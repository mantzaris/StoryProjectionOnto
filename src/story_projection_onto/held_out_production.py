"""Fail-closed production adapters for the reviewed held-out controller.

This module owns chronology, typed CAS resolution, cumulative schedule state, and
condition-service lifecycle.  Semantic generation is an injected narrow executor;
the controller never receives a lifecycle-capable vLLM object.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from pydantic import Field, model_validator

from story_projection_onto.benchmark_runtime import RuntimeStagingManifest
from story_projection_onto.conditions.base import ConditionAttemptRecord, ConditionPreparation
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    EvidencePacket,
    ImmutableRecord,
    PreconstructionRequest,
    PrequeryBarrier,
    PreQueryInventory,
    QueryAccessEvent,
    RetrievalMethod,
    ValidatedGeneration,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.development_adapter import (
    MeteredGenerationService,
    PackingTokenizer,
    encode_development_semantic_request,
)
from story_projection_onto.development_runtime import (
    DevelopmentExecutionResult,
    LiveServiceIdentity,
)
from story_projection_onto.held_out_primary import (
    PRIMARY_SERVICE_START_P95_SECONDS,
    GlobalGpuScheduleSnapshot,
    HeldOutAblationPrequeryReceipt,
    HeldOutC2PrequeryReceipt,
    HeldOutCallEnvelope,
    HeldOutCallSpec,
    HeldOutCASReference,
    HeldOutControlConfiguration,
    HeldOutControlError,
    HeldOutPrequeryUnit,
    HeldOutQueryOpening,
    HeldOutServiceResult,
    HeldOutServiceShutdownReceipt,
    HeldOutSessionIdentity,
    InjectedHeldOutSession,
    PublicStageReference,
    RepairReservePoolSnapshot,
    ReviewedHeldOutPlan,
)
from story_projection_onto.llm import (
    PackingReport,
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
)
from story_projection_onto.query_runtime import (
    AuditedBenchmarkRuntime,
    AuditedPacketMaterialization,
    AuditedQueryOpening,
)
from story_projection_onto.store import ArtifactRecord, ArtifactStore, GpuEventKind, ReleaseClass


class HeldOutProductionError(HeldOutControlError):
    """A production dependency or immutable artifact failed verification."""


RecordT = TypeVar("RecordT", bound=ImmutableRecord)


def _media_type(object_kind: str) -> str:
    return f"application/vnd.story-projection.{object_kind.replace('_', '-')}+json"


def _event_hash(event: object) -> str:
    return canonical_sha256(asdict(cast(Any, event)))


def gpu_ledger_chain_hash(artifacts: ArtifactStore) -> str:
    """Hash the complete ordered allocation-event ledger, including failures."""

    return canonical_sha256(tuple(asdict(item) for item in artifacts.ledger.gpu_events()))


@dataclass(frozen=True, slots=True)
class HeldOutArtifactResolver:
    artifacts: ArtifactStore

    def persist_record(
        self,
        value: ImmutableRecord,
        *,
        object_kind: str,
        release_class: ReleaseClass,
        created_at: datetime,
    ) -> HeldOutCASReference:
        artifact = self.artifacts.put_bytes(
            (value.to_canonical_json() + "\n").encode("utf-8"),
            media_type=_media_type(object_kind),
            release_class=release_class,
            created_at=created_at,
        )
        return HeldOutCASReference(
            artifact_hash=artifact.content_hash,
            logical_content_hash=value.content_hash,
            object_kind=cast(Any, object_kind),
            media_type=artifact.media_type,
            release_class=artifact.release_class.value,
        )

    def reference(
        self,
        artifact: ArtifactRecord,
        *,
        logical_content_hash: str,
        object_kind: str,
    ) -> HeldOutCASReference:
        return HeldOutCASReference(
            artifact_hash=artifact.content_hash,
            logical_content_hash=logical_content_hash,
            object_kind=cast(Any, object_kind),
            media_type=artifact.media_type,
            release_class=artifact.release_class.value,
        )

    def resolve_record(
        self,
        reference: HeldOutCASReference,
        model_type: type[RecordT],
        *,
        required_release: ReleaseClass | None = None,
    ) -> RecordT:
        artifact = self.artifacts.ledger.get_artifact(reference.artifact_hash)
        if (
            artifact.media_type != reference.media_type
            or artifact.release_class.value != reference.release_class
            or (required_release is not None and artifact.release_class is not required_release)
        ):
            raise HeldOutProductionError("CAS metadata differs from its typed reference")
        raw = self.artifacts.blobs.read_bytes(
            artifact,
            allow_restricted=artifact.release_class is ReleaseClass.RESTRICTED,
        )
        value = model_type.model_validate_json(raw)
        if value.content_hash != reference.logical_content_hash:
            raise HeldOutProductionError("CAS logical object hash changed")
        return value


class ShutdownCapableGenerationService(MeteredGenerationService, Protocol):
    def shutdown(self) -> object | None: ...


class HeldOutSemanticExecutor(Protocol):
    """Narrow semantic engine; implementations must durably receipt before return."""

    def execute(
        self,
        *,
        service: MeteredGenerationService,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        remaining_required_seconds: float,
    ) -> HeldOutServiceResult: ...

    def recover(
        self,
        *,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
    ) -> HeldOutServiceResult | None: ...


class ConditionServiceActivator(Protocol):
    """Outer lifecycle owner starts exactly one service for one condition block."""

    def __call__(
        self,
    ) -> tuple[
        ShutdownCapableGenerationService,
        LiveServiceIdentity,
        str,
    ]: ...


class HeldOutScheduleState(ImmutableRecord):
    call_manifest_hash: str
    development_execution_result_hash: str
    development_predecessor_allocated_gpu_seconds: float
    remaining_registered_p95_seconds: float
    consumed_primary_service_start_count: int = Field(ge=0, le=3, default=0)
    consumed_long: int = Field(ge=0, le=4, default=0)
    consumed_standard: int = Field(ge=0, le=8, default=0)
    consumed_short: int = Field(ge=0, le=4, default=0)
    completed_result_hashes: Mapping[str, str] = Field(default_factory=dict)
    session_identities: Mapping[str, HeldOutSessionIdentity] = Field(default_factory=dict)
    shutdown_receipts: Mapping[str, HeldOutServiceShutdownReceipt] = Field(default_factory=dict)
    active_call_id: str | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def service_lifecycle_is_one_to_one(self) -> HeldOutScheduleState:
        if len(self.session_identities) != self.consumed_primary_service_start_count:
            raise ValueError("service-start count differs from persisted condition identities")
        if not set(self.shutdown_receipts).issubset(self.session_identities):
            raise ValueError("shutdown receipt exists without a condition service identity")
        return self


@dataclass(slots=True)
class ProductionHeldOutRuntime:
    """Shared audited runtime with no access to scorer-gold namespaces."""

    repository: Path
    reviewed_plan: ReviewedHeldOutPlan
    configuration: HeldOutControlConfiguration
    development_result: DevelopmentExecutionResult
    artifacts: ArtifactStore
    tokenizer: PackingTokenizer
    neutral_by_model_visible_hash: Mapping[str, Any]
    state_path: Path
    semantic_executor: HeldOutSemanticExecutor
    activators: Mapping[ConditionName, ConditionServiceActivator]
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    resolver: HeldOutArtifactResolver = field(init=False)
    query_runtime: AuditedBenchmarkRuntime = field(init=False)
    _openings: dict[str, AuditedQueryOpening] = field(default_factory=dict, init=False)
    _packets: dict[str, AuditedPacketMaterialization] = field(default_factory=dict, init=False)
    _services: dict[ConditionName, ShutdownCapableGenerationService] = field(
        default_factory=dict, init=False
    )
    _identities: dict[ConditionName, HeldOutSessionIdentity] = field(
        default_factory=dict, init=False
    )
    _shutdowns: dict[ConditionName, HeldOutServiceShutdownReceipt] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self.repository = self.repository.resolve(strict=True)
        self.state_path = self.state_path.resolve(strict=False)
        manifest = self.reviewed_plan.call_manifest
        if (
            self.development_result.content_hash != manifest.development_execution_result_hash
            or not self.development_result.gate.passed
            or self.development_result.forecast.actual_allocated_seconds_after
            != manifest.allocated_gpu_seconds_before_heldout
            or self.development_result.forecast.post_development_mandatory_forecast_seconds
            != manifest.post_development_mandatory_forecast_seconds
        ):
            raise HeldOutProductionError(
                "production adapter differs from the exact passing development predecessor"
            )
        if set(self.activators) != {
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }:
            raise HeldOutProductionError("production adapter requires three condition activators")
        self.resolver = HeldOutArtifactResolver(self.artifacts)
        self.query_runtime = AuditedBenchmarkRuntime(
            ledger=self.artifacts.ledger,
            blobs=self.artifacts.blobs,
            clock=self.clock,
        )
        if self.state_path.exists():
            state = self._read_state()
            if (
                state.call_manifest_hash != manifest.content_hash
                or state.development_execution_result_hash != self.development_result.content_hash
            ):
                raise HeldOutProductionError("durable held-out state belongs to another run")
        else:
            self._write_state(
                HeldOutScheduleState(
                    call_manifest_hash=manifest.content_hash,
                    development_execution_result_hash=self.development_result.content_hash,
                    development_predecessor_allocated_gpu_seconds=(
                        self.development_result.forecast.actual_allocated_seconds_after
                    ),
                    remaining_registered_p95_seconds=(
                        self.development_result.forecast.post_development_mandatory_forecast_seconds
                    ),
                    updated_at=self._now(),
                )
            )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise HeldOutProductionError("production clock must be timezone-aware")
        return value.astimezone(UTC)

    def _read_state(self) -> HeldOutScheduleState:
        if self.state_path.is_symlink() or not self.state_path.is_file():
            raise HeldOutProductionError("held-out schedule state is missing or unsafe")
        return HeldOutScheduleState.model_validate_json(self.state_path.read_bytes())

    def _write_state(self, state: HeldOutScheduleState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((state.to_canonical_json() + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _replace_state(self, state: HeldOutScheduleState, **updates: object) -> None:
        payload = state.model_dump(mode="python", exclude={"content_hash"})
        payload.update(updates)
        payload["updated_at"] = self._now()
        self._write_state(HeldOutScheduleState.model_validate(payload))

    def schedule_snapshot(self) -> GlobalGpuScheduleSnapshot:
        state = self._read_state()
        allocated = max(
            self.artifacts.ledger.gpu_summary().total_allocated_seconds,
            *(service.actual_allocated_service_seconds for service in self._services.values()),
        )
        if not math.isfinite(allocated):
            raise HeldOutProductionError("cumulative GPU allocation is not finite")
        return GlobalGpuScheduleSnapshot(
            global_accounting_id=f"held-out-global-{self.development_result.content_hash[:16]}",
            gpu_call_inventory_file_sha256=(self.configuration.gpu_call_inventory_file_sha256),
            development_execution_result_hash=self.development_result.content_hash,
            development_predecessor_allocated_gpu_seconds=(
                state.development_predecessor_allocated_gpu_seconds
            ),
            ledger_chain_hash=gpu_ledger_chain_hash(self.artifacts),
            actual_allocated_gpu_seconds=allocated,
            remaining_registered_p95_seconds=state.remaining_registered_p95_seconds,
            repair_reserves=(
                RepairReservePoolSnapshot(
                    reserve_class="reserve_long",
                    total_slots=4,
                    consumed_slots=state.consumed_long,
                    watchdog_seconds=240,
                ),
                RepairReservePoolSnapshot(
                    reserve_class="reserve_standard",
                    total_slots=8,
                    consumed_slots=state.consumed_standard,
                    watchdog_seconds=150,
                ),
                RepairReservePoolSnapshot(
                    reserve_class="reserve_short",
                    total_slots=4,
                    consumed_slots=state.consumed_short,
                    watchdog_seconds=90,
                ),
            ),
            captured_at=self._now(),
        )

    def activate(self, condition: ConditionName) -> HeldOutSessionIdentity:
        existing = self._identities.get(condition)
        if existing is not None:
            return existing
        state = self._read_state()
        persisted = state.session_identities.get(condition.value)
        if persisted is not None and condition.value in state.shutdown_receipts:
            self._identities[condition] = persisted
            self._shutdowns[condition] = state.shutdown_receipts[condition.value]
            return persisted
        if self._services:
            raise HeldOutProductionError(
                "a prior condition service remains live; concurrent loads are forbidden"
            )
        before = self.schedule_snapshot()
        if (
            state.remaining_registered_p95_seconds < PRIMARY_SERVICE_START_P95_SECONDS
            or before.actual_allocated_gpu_seconds + PRIMARY_SERVICE_START_P95_SECONDS
            >= self.configuration.hard_gpu_seconds_limit
            or before.actual_allocated_gpu_seconds + state.remaining_registered_p95_seconds
            > self.configuration.scheduled_gpu_seconds_limit
        ):
            raise HeldOutProductionError(
                "condition service load is not admissible under the cumulative GPU gates"
            )
        service, live, model_load_event_id = self.activators[condition]()
        events = tuple(
            item
            for item in self.artifacts.ledger.gpu_events()
            if item.event_id == model_load_event_id
        )
        if len(events) != 1 or events[0].event_kind is not GpuEventKind.MODEL_LOAD:
            raise HeldOutProductionError("condition activation lacks one metered model load")
        identity = HeldOutSessionIdentity(
            session_id=f"held-out-session-{condition.value}",
            condition=condition,
            service_identity_hash=live.content_hash,
            service_pid=live.service_pid,
            service_start_ticks=live.service_start_ticks,
            global_accounting_id=f"held-out-global-{self.development_result.content_hash[:16]}",
            global_ledger_chain_hash=gpu_ledger_chain_hash(self.artifacts),
            allocation_event_id=live.gpu_session_event_id,
            model_load_event_id=model_load_event_id,
            model_load_event_hash=_event_hash(events[0]),
        )
        if persisted is not None and persisted != identity:
            raise HeldOutProductionError(
                "interrupted condition service could not be adopted without another load"
            )
        if any(
            identity.service_identity_hash == item.service_identity_hash
            or identity.allocation_event_id == item.allocation_event_id
            or identity.model_load_event_hash == item.model_load_event_hash
            or (identity.service_pid, identity.service_start_ticks)
            == (item.service_pid, item.service_start_ticks)
            for item in self._identities.values()
        ):
            raise HeldOutProductionError("condition activation reused a prior service allocation")
        self._services[condition] = service
        self._identities[condition] = identity
        identities = dict(state.session_identities)
        identities[condition.value] = identity
        remaining = state.remaining_registered_p95_seconds - PRIMARY_SERVICE_START_P95_SECONDS
        if remaining < 0:
            service.shutdown()
            self._services.pop(condition, None)
            self._identities.pop(condition, None)
            raise HeldOutProductionError(
                "condition service load consumed unregistered GPU schedule"
            )
        after_allocated = max(
            self.artifacts.ledger.gpu_summary().total_allocated_seconds,
            service.actual_allocated_service_seconds,
        )
        if (
            after_allocated >= self.configuration.hard_gpu_seconds_limit
            or after_allocated + remaining > self.configuration.scheduled_gpu_seconds_limit
        ):
            service.shutdown()
            self._services.pop(condition, None)
            self._identities.pop(condition, None)
            raise HeldOutProductionError("condition service load crossed a cumulative GPU gate")
        self._replace_state(
            state,
            session_identities=identities,
            remaining_registered_p95_seconds=remaining,
            consumed_primary_service_start_count=(state.consumed_primary_service_start_count + 1),
        )
        return identity

    def prepare_c2_empty_inventory(
        self, unit: HeldOutPrequeryUnit, *, seed_block: int
    ) -> HeldOutC2PrequeryReceipt:
        recorded_at = self._now()
        inventory = PreQueryInventory(
            inventory_id=f"empty-{unit.unit_id}-s{seed_block}",
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            recorded_at=recorded_at,
        )
        reference = self.resolver.persist_record(
            inventory,
            object_kind="pre_query_inventory",
            release_class=ReleaseClass.RESTRICTED,
            created_at=recorded_at,
        )
        preparation = ConditionPreparation(
            preparation_id=f"empty-preparation-{unit.unit_id}-s{seed_block}",
            condition=ConditionName.C2_LLM_QUERY,
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            completed_at=recorded_at,
            empty_inventory=inventory,
        )
        preparation_reference = self.resolver.persist_record(
            preparation,
            object_kind="condition_preparation",
            release_class=ReleaseClass.RESTRICTED,
            created_at=recorded_at,
        )
        return HeldOutC2PrequeryReceipt(
            unit_id=unit.unit_id,
            seed_block=cast(Any, seed_block),
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            inventory=inventory,
            inventory_artifact=reference,
            preparation=preparation,
            preparation_artifact=preparation_reference,
            completed_at=self._now(),
        )

    def validate_c2_prequery_receipt(self, receipt: HeldOutC2PrequeryReceipt) -> None:
        inventory = self.resolver.resolve_record(
            receipt.inventory_artifact,
            PreQueryInventory,
            required_release=ReleaseClass.RESTRICTED,
        )
        if inventory != receipt.inventory:
            raise HeldOutProductionError("C2 empty-inventory CAS object changed")
        preparation = self.resolver.resolve_record(
            receipt.preparation_artifact,
            ConditionPreparation,
            required_release=ReleaseClass.RESTRICTED,
        )
        if preparation != receipt.preparation:
            raise HeldOutProductionError("C2 condition-preparation CAS object changed")

    def prepare_ablation_empty_inventory(
        self,
        unit: HeldOutPrequeryUnit,
        *,
        condition: ConditionName,
    ) -> HeldOutAblationPrequeryReceipt:
        allowed = {
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        }
        if condition not in allowed:
            raise HeldOutProductionError("unsupported held-out ablation preparation")
        recorded_at = self._now()
        inventory = PreQueryInventory(
            inventory_id=f"empty-{condition.value}-{unit.unit_id}-s1",
            condition=condition,
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            recorded_at=recorded_at,
        )
        preparation = ConditionPreparation(
            preparation_id=f"empty-preparation-{condition.value}-{unit.unit_id}-s1",
            condition=condition,
            snapshot_hash=unit.prequery_stage.snapshot_hash,
            completed_at=recorded_at,
            empty_inventory=inventory,
        )
        inventory_reference = self.resolver.persist_record(
            inventory,
            object_kind="pre_query_inventory",
            release_class=ReleaseClass.RESTRICTED,
            created_at=recorded_at,
        )
        preparation_reference = self.resolver.persist_record(
            preparation,
            object_kind="condition_preparation",
            release_class=ReleaseClass.RESTRICTED,
            created_at=recorded_at,
        )
        return HeldOutAblationPrequeryReceipt(
            unit_id=unit.unit_id,
            condition=condition,
            prequery_stage_hash=unit.prequery_stage.staging_manifest_hash,
            preparation=preparation,
            inventory_artifact=inventory_reference,
            preparation_artifact=preparation_reference,
            completed_at=self._now(),
        )

    def validate_ablation_prequery_receipt(self, receipt: HeldOutAblationPrequeryReceipt) -> None:
        inventory = self.resolver.resolve_record(
            receipt.inventory_artifact,
            PreQueryInventory,
            required_release=ReleaseClass.RESTRICTED,
        )
        preparation = self.resolver.resolve_record(
            receipt.preparation_artifact,
            ConditionPreparation,
            required_release=ReleaseClass.RESTRICTED,
        )
        if inventory != receipt.preparation.empty_inventory or preparation != receipt.preparation:
            raise HeldOutProductionError("ablation pre-query CAS lineage changed")

    def persist_prequery_barrier(self, barrier: PrequeryBarrier) -> None:
        self.query_runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)

    def _stage_manifest(self, stage: PublicStageReference) -> tuple[Path, RuntimeStagingManifest]:
        root = (self.repository / stage.relative_path).resolve(strict=True)
        if not root.is_relative_to(self.repository):
            raise HeldOutProductionError("query stage escaped the repository")
        path = root / "manifest.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != stage.manifest_file_sha256:
            raise HeldOutProductionError("query-stage manifest bytes changed")
        manifest = RuntimeStagingManifest.model_validate_json(raw)
        if manifest.content_hash != stage.staging_manifest_hash:
            raise HeldOutProductionError("query-stage logical manifest changed")
        return root, manifest

    def _materialize(self, opening: AuditedQueryOpening) -> AuditedPacketMaterialization:
        neutral = self.neutral_by_model_visible_hash.get(opening.evidence.content_hash)
        if neutral is None or neutral.snapshot != opening.evidence.snapshot:
            raise HeldOutProductionError("query lacks its query-blind neutral evidence")

        def all_admissible(_evidence: object, _reveal: object) -> EvidencePacket:
            created_at = self._now()
            return EvidencePacket(
                packet_id=f"packet-{opening.access_event.stage_manifest_hash[:20]}",
                snapshot_hash=neutral.snapshot.content_hash,
                evidence=neutral.evidence,
                ordered_evidence_ids=neutral.snapshot.eligible_evidence_ids,
                retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
                token_count=sum(
                    len(self.tokenizer.encode(item.text, add_special_tokens=False))
                    for item in neutral.evidence
                ),
                created_at=created_at,
                release_class=neutral.snapshot.release_class,
            )

        return self.query_runtime.materialize_packet(
            opening,
            materialization_event_id=(
                f"{opening.access_event.execution_id}-packet-"
                f"{opening.access_event.stage_manifest_hash[:16]}"
            ),
            retrieval_config_hash=canonical_sha256(
                {
                    "policy": "complete_admissible_snapshot_v1",
                    "stage": opening.access_event.stage_manifest_hash,
                }
            ),
            materializer=all_admissible,
        )

    def open_or_recover_query(
        self,
        stage: PublicStageReference,
        *,
        barrier: PrequeryBarrier,
        persisted: HeldOutQueryOpening | None,
    ) -> HeldOutQueryOpening:
        if persisted is not None:
            event = self.resolver.resolve_record(
                persisted.query_access_artifact,
                QueryAccessEvent,
            )
            packet = self.resolver.resolve_record(
                persisted.evidence_packet_artifact, EvidencePacket
            )
            if (
                event != persisted.query_access_event
                or packet.content_hash != persisted.evidence_packet_hash
            ):
                raise HeldOutProductionError("persisted query opening CAS lineage changed")
            return persisted
        root, manifest = self._stage_manifest(stage)
        opening = self.query_runtime.open_query(
            staging_root=root,
            manifest=manifest,
            barrier=barrier,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            access_event_id=f"query-access-{stage.stage_id}",
        )
        materialized = self._materialize(opening)
        access_record = self.artifacts.ledger.get_artifact(
            opening.persistence.access_event_artifact_hash
        )
        packet_record = self.artifacts.ledger.get_artifact(
            materialized.persistence.packet_artifact_hash
        )
        opened_stage = PublicStageReference(
            **stage.model_dump(
                mode="python",
                exclude={
                    "content_hash",
                    "query_context_hash",
                    "horizon_hash",
                    "budget_hash",
                },
            ),
            query_context_hash=opening.context.content_hash,
            horizon_hash=opening.context.spoiler_horizon.content_hash,
            budget_hash=opening.context.budgets.content_hash,
        )
        value = HeldOutQueryOpening(
            sealed_stage_hash=stage.staging_manifest_hash,
            opened_stage=opened_stage,
            query_access_event=opening.access_event,
            query_access_artifact=self.resolver.reference(
                access_record,
                logical_content_hash=opening.access_event.content_hash,
                object_kind="query_access_event",
            ),
            evidence_packet_hash=materialized.packet.content_hash,
            evidence_packet_artifact=self.resolver.reference(
                packet_record,
                logical_content_hash=materialized.packet.content_hash,
                object_kind="evidence_packet",
            ),
            opened_at=opening.access_event.accessed_at,
        )
        self._openings[stage.staging_manifest_hash] = opening
        self._packets[stage.staging_manifest_hash] = materialized
        return value

    def validate_result_artifacts(
        self,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
        result: HeldOutServiceResult,
    ) -> None:
        if not result.request_started:
            return
        receipt = result.artifact_receipt
        if receipt is None or result.ledger_receipt_artifact_hash is None:
            raise HeldOutProductionError("started result lacks its typed audit receipt")
        receipt_ref = HeldOutCASReference(
            artifact_hash=result.ledger_receipt_artifact_hash,
            logical_content_hash=receipt.content_hash,
            object_kind="held_out_call_audit_receipt",
            media_type=_media_type("held_out_call_audit_receipt"),
            release_class="restricted",
        )
        persisted_receipt = self.resolver.resolve_record(
            receipt_ref, type(receipt), required_release=ReleaseClass.RESTRICTED
        )
        if persisted_receipt != receipt:
            raise HeldOutProductionError("held-out audit receipt CAS object changed")
        semantic_type = (
            PreconstructionRequest
            if call.condition is ConditionName.C1_LLM_PRE
            else ConstructionRequest
        )
        semantic = self.resolver.resolve_record(
            receipt.semantic_request,
            semantic_type,
            required_release=ReleaseClass.RESTRICTED,
        )
        packing = self.resolver.resolve_record(
            receipt.packing_report,
            PackingReport,
            required_release=ReleaseClass.RESTRICTED,
        )
        if semantic.content_hash != result.request_hash or packing.condition is not call.condition:
            raise HeldOutProductionError("request or packing receipt differs from the call")
        packing_sections = {item.name: item for item in packing.sections}
        if (
            packing_sections["output_schema"].section_content_hash
            != semantic.runtime.output_schema_hash
        ):
            raise HeldOutProductionError("packed constrained grammar hash changed")
        fixed_inventory = None
        if call.condition is ConditionName.A_FIXED_SELECT:
            fixed = cast(ConstructionRequest, semantic).fixed_ontology
            source_reference = envelope.source_c1_output
            source_preparation = (
                None
                if source_reference is None
                else self.resolver.resolve_record(
                    source_reference,
                    ConditionPreparation,
                    required_release=ReleaseClass.RESTRICTED,
                )
            )
            source = None if source_preparation is None else source_preparation.sealed_preontology
            if (
                fixed is None
                or source is None
                or fixed != source.as_fixed_ontology()
                or fixed.construction_seal.content_hash != envelope.source_c1_seal_hash
                or not packing.complete_sealed_ontology
            ):
                raise HeldOutProductionError(
                    "FixedSelect did not pack the complete inherited C1 ontology"
                )
            encoded_fixed = encode_development_semantic_request(
                cast(ConstructionRequest, semantic)
            ).sections["sealed_ontology"]
            expected_packed_hash = hashlib.sha256(
                canonical_json(encoded_fixed).encode("utf-8")
            ).hexdigest()
            if packing_sections["sealed_ontology"].section_content_hash != expected_packed_hash:
                raise HeldOutProductionError("FixedSelect sealed graph packing changed")
            fixed_inventory = sealed_inventory_from_fixed_ontology(
                fixed,
                seed_block=call.seed_block,
                source_draft=source.draft,
            )
        generation = None
        if receipt.validation is not None:
            generation = self.resolver.resolve_record(
                receipt.validation,
                ValidatedGeneration,
                required_release=ReleaseClass.RESTRICTED,
            )
        if fixed_inventory is not None and generation is not None:
            enforce_fixed_select_draft(
                generation.draft,
                sealed=fixed_inventory,
                seed_block=call.seed_block,
            )
        if receipt.output is not None:
            output_type = (
                ConditionPreparation
                if call.condition is ConditionName.C1_LLM_PRE
                else ConditionAttemptRecord
            )
            self.resolver.resolve_record(
                receipt.output,
                output_type,
                required_release=ReleaseClass.RESTRICTED,
            )
        event = next(
            (
                item
                for item in self.artifacts.ledger.gpu_events()
                if item.event_id == receipt.gpu_event_id
            ),
            None,
        )
        model_call = self.artifacts.ledger.get_model_call(receipt.model_call_id)
        if (
            event is None
            or _event_hash(event) != receipt.gpu_event_hash
            or canonical_sha256(asdict(model_call)) != receipt.model_call_record_hash
        ):
            raise HeldOutProductionError("GPU/model-call ledger receipt changed")

    def shutdown_condition(
        self, identity: HeldOutSessionIdentity
    ) -> tuple[HeldOutServiceShutdownReceipt, GlobalGpuScheduleSnapshot]:
        existing = self._shutdowns.get(identity.condition)
        if existing is not None:
            return existing, existing.schedule_after_shutdown
        state = self._read_state()
        persisted = state.shutdown_receipts.get(identity.condition.value)
        if persisted is not None:
            if persisted.session_identity_hash != identity.content_hash:
                raise HeldOutProductionError("persisted shutdown belongs to another service")
            self._shutdowns[identity.condition] = persisted
            return persisted, persisted.schedule_after_shutdown
        if self._identities.get(identity.condition) != identity:
            raise HeldOutProductionError("shutdown requested for another service identity")
        service = self._services.pop(identity.condition)
        service.shutdown()
        snapshot = self.schedule_snapshot()
        receipt = HeldOutServiceShutdownReceipt(
            condition=identity.condition,
            session_identity_hash=identity.content_hash,
            allocation_event_id=identity.allocation_event_id,
            model_load_event_id=identity.model_load_event_id,
            model_load_event_hash=identity.model_load_event_hash,
            global_accounting_id=identity.global_accounting_id,
            final_ledger_chain_hash=snapshot.ledger_chain_hash,
            schedule_after_shutdown=snapshot,
            stopped_at=snapshot.captured_at,
        )
        self._shutdowns[identity.condition] = receipt
        shutdowns = dict(state.shutdown_receipts)
        shutdowns[identity.condition.value] = receipt
        self._replace_state(state, shutdown_receipts=shutdowns)
        return receipt, snapshot

    def session(self, condition: ConditionName) -> ProductionHeldOutSession:
        return ProductionHeldOutSession(condition=condition, runtime=self)


@dataclass(frozen=True, slots=True)
class ProductionHeldOutSession:
    condition: ConditionName
    runtime: ProductionHeldOutRuntime = field(repr=False)

    def identity(self) -> HeldOutSessionIdentity:
        return self.runtime.activate(self.condition)

    def schedule_snapshot(self) -> GlobalGpuScheduleSnapshot:
        return self.runtime.schedule_snapshot()

    def execute_call(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult:
        if call.condition is not self.condition:
            raise HeldOutProductionError("call routed through another condition session")
        state = self.runtime._read_state()
        if state.active_call_id not in {None, call.call_id}:
            raise HeldOutProductionError("another held-out call remains active")
        if call.call_id in state.completed_result_hashes:
            raise HeldOutProductionError("completed call cannot be issued again")
        self.runtime._replace_state(state, active_call_id=call.call_id)
        result = self.runtime.semantic_executor.execute(
            service=self.runtime._services[self.condition],
            call=call,
            envelope=envelope,
            remaining_required_seconds=state.remaining_registered_p95_seconds,
        )
        self._finalize_state(call, result)
        return result

    def recover_call(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult | None:
        state = self.runtime._read_state()
        result = self.runtime.semantic_executor.recover(call=call, envelope=envelope)
        if result is None:
            return None
        if call.call_id not in state.completed_result_hashes:
            self._finalize_state(call, result)
        elif state.completed_result_hashes[call.call_id] != result.content_hash:
            raise HeldOutProductionError("recovered result differs from durable state")
        return result

    def _finalize_state(self, call: HeldOutCallSpec, result: HeldOutServiceResult) -> None:
        state = self.runtime._read_state()
        completed = dict(state.completed_result_hashes)
        prior = completed.get(call.call_id)
        if prior is not None:
            if prior != result.content_hash:
                raise HeldOutProductionError("call completion changed during resume")
            return
        remaining = state.remaining_registered_p95_seconds
        updates: dict[str, object] = {}
        if result.request_started:
            remaining -= call.p95_seconds + result.repair_attempts * call.watchdog_seconds
            reserve_field = {
                "reserve_long": "consumed_long",
                "reserve_standard": "consumed_standard",
                "reserve_short": "consumed_short",
            }[call.repair_reserve_class]
            updates[reserve_field] = getattr(state, reserve_field) + result.repair_attempts
        if remaining < 0:
            raise HeldOutProductionError("held-out schedule consumed unregistered GPU work")
        completed[call.call_id] = result.content_hash
        updates.update(
            remaining_registered_p95_seconds=remaining,
            completed_result_hashes=completed,
            active_call_id=None,
        )
        self.runtime._replace_state(state, **updates)


@dataclass(frozen=True, slots=True)
class ProductionHeldOutBundle:
    cpu: object
    sessions: Mapping[ConditionName, InjectedHeldOutSession]
    runtime: ProductionHeldOutRuntime


def create_production_held_out_bundle(
    *,
    repository: Path,
    reviewed_plan: ReviewedHeldOutPlan,
    configuration: HeldOutControlConfiguration,
    development_result: DevelopmentExecutionResult,
    artifacts: ArtifactStore,
    tokenizer: PackingTokenizer,
    neutral_by_model_visible_hash: Mapping[str, Any],
    state_path: Path,
    semantic_executor: HeldOutSemanticExecutor,
    activators: Mapping[ConditionName, ConditionServiceActivator],
    cpu: object,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ProductionHeldOutBundle:
    """Bind the frozen predecessor and three sequential service allocations."""

    runtime = ProductionHeldOutRuntime(
        repository=repository,
        reviewed_plan=reviewed_plan,
        configuration=configuration,
        development_result=development_result,
        artifacts=artifacts,
        tokenizer=tokenizer,
        neutral_by_model_visible_hash=neutral_by_model_visible_hash,
        state_path=state_path,
        semantic_executor=semantic_executor,
        activators=activators,
        clock=clock,
    )
    sessions = {
        condition: cast(InjectedHeldOutSession, runtime.session(condition))
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    }
    return ProductionHeldOutBundle(cpu=cpu, sessions=sessions, runtime=runtime)


__all__ = [
    "ConditionServiceActivator",
    "HeldOutArtifactResolver",
    "HeldOutProductionError",
    "HeldOutSemanticExecutor",
    "ProductionHeldOutBundle",
    "ProductionHeldOutRuntime",
    "ProductionHeldOutSession",
    "ShutdownCapableGenerationService",
    "create_production_held_out_bundle",
    "gpu_ledger_chain_hash",
]
