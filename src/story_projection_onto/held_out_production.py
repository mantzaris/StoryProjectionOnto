"""Fail-closed production adapters for the reviewed held-out controller.

This module owns chronology, typed CAS resolution, cumulative schedule state, and
condition-service lifecycle.  Semantic generation is an injected narrow executor;
the controller never receives a lifecycle-capable vLLM object.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from pydantic import Field, model_validator

from story_projection_onto.benchmark_runtime import RuntimeStagingManifest
from story_projection_onto.conditions.base import ConditionAttemptRecord, ConditionPreparation
from story_projection_onto.conditions.fixed_select import prepare_fixed_selection
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    EvidencePacket,
    ImmutableRecord,
    PacketMaterializationEvent,
    PreconstructionRequest,
    PrequeryBarrier,
    PreQueryInventory,
    QueryAccessEvent,
    RetrievalMethod,
    Sha256Digest,
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
    RunOutcome,
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
        repair_allowed: bool,
    ) -> HeldOutServiceResult: ...

    def recover(
        self,
        *,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
    ) -> HeldOutServiceResult | None: ...

    def recovery_pending(
        self,
        *,
        call: HeldOutCallSpec,
        envelope: HeldOutCallEnvelope,
    ) -> bool: ...


class HeldOutServiceActivationIntent(ImmutableRecord):
    """Path-free promise authorizing one condition's only physical model load."""

    condition: ConditionName
    call_manifest_hash: str
    development_execution_result_hash: str
    global_accounting_id: str
    expected_session_id: str
    expected_model_load_event_id: str
    launcher_configuration_hash: str
    service_start_watchdog_seconds: float = Field(ge=180, le=600)
    remaining_registered_seconds_before: float = Field(ge=0)
    requested_at: datetime


class HeldOutServiceShutdownIntent(ImmutableRecord):
    """Path-free durable boundary written before physically stopping a service."""

    condition: ConditionName
    session_identity_hash: str
    allocation_event_id: str
    model_load_event_id: str
    global_accounting_id: str
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class ConditionServiceActivation:
    """One live adoption/start or one terminally reconciled prior allocation."""

    service: ResumableVLLMService
    live_identity: LiveServiceIdentity
    model_load_event_id: str
    terminal_recovery: object | None = None

    @property
    def service_ready(self) -> bool:
        return self.terminal_recovery is None

    def __iter__(self):
        """Retain the historical three-value unpacking for fixture callers."""

        yield self.service
        yield self.live_identity
        yield self.model_load_event_id


class ConditionServiceActivator(Protocol):
    """Outer lifecycle owner starts exactly one service for one condition block."""

    def __call__(
        self,
        *,
        remaining_required_seconds: float,
        resume_required: bool = False,
        activation_intent: HeldOutServiceActivationIntent,
    ) -> ConditionServiceActivation: ...

    def detach_for_resume(self) -> None: ...


class ResumableVLLMService(ShutdownCapableGenerationService, Protocol):
    @property
    def pid(self) -> int: ...

    def start(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float,
        remaining_required_seconds: float,
    ) -> None: ...

    def write_resume_checkpoint(self, path: Path) -> None: ...

    def resume_from_checkpoint(self, path: Path) -> bool: ...

    def resume_live_service_lease(
        self,
        *,
        expected_session_id: str,
        expected_event_id: str,
        watchdog_seconds: float,
    ) -> bool: ...

    def recover_stale_service_lease(self) -> object | None: ...

    @property
    def last_recovered_process_identity(self) -> object | None: ...

    def detach_for_controller_restart(self, path: Path) -> None: ...


@dataclass(slots=True)
class SequentialVLLMConditionActivator:
    """Start or adopt exactly one persistent vLLM process for one condition block."""

    condition: ConditionName
    service_factory: Callable[[], ResumableVLLMService]
    checkpoint_path: Path
    owner_run_id: str
    launcher_configuration_hash: str
    model_snapshot_hash: str
    selected_model_freeze_hash: str
    source_execution_hash: str
    service_start_watchdog_seconds: float = PRIMARY_SERVICE_START_P95_SECONDS
    _service: ResumableVLLMService | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.condition not in {
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }:
            raise HeldOutProductionError("unsupported held-out service condition")
        if (
            not math.isfinite(self.service_start_watchdog_seconds)
            or self.service_start_watchdog_seconds < PRIMARY_SERVICE_START_P95_SECONDS
            or self.service_start_watchdog_seconds > 600
        ):
            raise HeldOutProductionError(
                "held-out startup watchdog must be frozen between 180 and 600 seconds"
            )
        self.checkpoint_path = self.checkpoint_path.resolve(strict=False)

    def _checkpoint_identity(self, model_load_event_id: str) -> tuple[int, int]:
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HeldOutProductionError("condition service checkpoint is invalid") from error
        if not isinstance(checkpoint, Mapping):
            raise HeldOutProductionError("condition service checkpoint is invalid")
        pid = checkpoint.get("pid")
        start_ticks = checkpoint.get("process_start_ticks")
        expected_session_id = f"{self.owner_run_id}-{self.condition.value}-service"
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(start_ticks, bool)
            or not isinstance(start_ticks, int)
            or start_ticks <= 0
            or checkpoint.get("accounting_session_id") != model_load_event_id
            or checkpoint.get("session_id") != expected_session_id
            or checkpoint.get("configuration_hash") != self.launcher_configuration_hash
        ):
            raise HeldOutProductionError("condition service checkpoint identity changed")
        return pid, start_ticks

    def __call__(
        self,
        *,
        remaining_required_seconds: float,
        resume_required: bool = False,
        activation_intent: HeldOutServiceActivationIntent,
    ) -> ConditionServiceActivation:
        if self._service is not None:
            raise HeldOutProductionError("condition activator cannot allocate twice")
        if not math.isfinite(remaining_required_seconds) or remaining_required_seconds < 0:
            raise HeldOutProductionError("remaining GPU schedule must be finite and nonnegative")
        model_load_event_id = f"{self.owner_run_id}-{self.condition.value}-model-load"
        expected_session_id = f"{self.owner_run_id}-{self.condition.value}-service"
        if (
            activation_intent.condition is not self.condition
            or activation_intent.expected_session_id != expected_session_id
            or activation_intent.expected_model_load_event_id != model_load_event_id
            or activation_intent.launcher_configuration_hash
            != self.launcher_configuration_hash
            or activation_intent.service_start_watchdog_seconds
            != self.service_start_watchdog_seconds
            or activation_intent.remaining_registered_seconds_before
            != remaining_required_seconds + PRIMARY_SERVICE_START_P95_SECONDS
        ):
            raise HeldOutProductionError("condition activation differs from its durable intent")
        service = self.service_factory()
        try:
            resumed = False
            terminal_recovery = None
            if self.checkpoint_path.exists():
                pid, start_ticks = self._checkpoint_identity(model_load_event_id)
                resumed = service.resume_from_checkpoint(self.checkpoint_path)
                if not resumed:
                    resumed = service.resume_live_service_lease(
                        expected_session_id=expected_session_id,
                        expected_event_id=model_load_event_id,
                        watchdog_seconds=self.service_start_watchdog_seconds,
                    )
                    if resumed:
                        service.write_resume_checkpoint(self.checkpoint_path)
                        pid, start_ticks = self._checkpoint_identity(model_load_event_id)
                    else:
                        terminal_recovery = service.recover_stale_service_lease()
                    if not resumed and terminal_recovery is None:
                        raise HeldOutProductionError(
                            "persisted condition service is neither adoptable nor "
                            "recoverable; a second load is forbidden"
                        )
            else:
                resumed = service.resume_live_service_lease(
                    expected_session_id=expected_session_id,
                    expected_event_id=model_load_event_id,
                    watchdog_seconds=self.service_start_watchdog_seconds,
                )
                if resumed:
                    self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    service.write_resume_checkpoint(self.checkpoint_path)
                    pid, start_ticks = self._checkpoint_identity(model_load_event_id)
                else:
                    terminal_recovery = service.recover_stale_service_lease()
                    if terminal_recovery is not None:
                        recovered_identity = service.last_recovered_process_identity
                        if recovered_identity is None:
                            raise HeldOutProductionError(
                                "terminal service lacks a recoverable process identity"
                            )
                        recovered_process = cast(Any, recovered_identity)
                        pid = cast(int, recovered_process.pid)
                        start_ticks = cast(
                            int,
                            recovered_process.process_start_ticks,
                        )
                        if (
                            recovered_process.configuration_hash
                            != self.launcher_configuration_hash
                            or recovered_process.session_id != expected_session_id
                            or recovered_process.accounting_session_id
                            != model_load_event_id
                        ):
                            raise HeldOutProductionError(
                                "terminal service recovery identity changed"
                            )
                    elif resume_required:
                        raise HeldOutProductionError(
                            "persisted condition identity lacks a recoverable service; "
                            "a replacement model load is forbidden"
                        )
                    else:
                        service.start(
                            session_id=expected_session_id,
                            event_id=model_load_event_id,
                            watchdog_seconds=self.service_start_watchdog_seconds,
                            remaining_required_seconds=remaining_required_seconds,
                        )
                        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        service.write_resume_checkpoint(self.checkpoint_path)
                        pid, start_ticks = self._checkpoint_identity(model_load_event_id)
            if resumed and service.pid != pid:
                raise HeldOutProductionError("condition service checkpoint identity changed")
        except BaseException:
            try:
                service.shutdown()
            except BaseException as shutdown_error:
                raise HeldOutProductionError(
                    "failed condition activation could not verify service shutdown"
                ) from shutdown_error
            raise
        identity = LiveServiceIdentity(
            owner_run_id=self.owner_run_id,
            service_pid=pid,
            service_start_ticks=start_ticks,
            gpu_session_event_id=model_load_event_id,
            launcher_configuration_hash=self.launcher_configuration_hash,
            model_snapshot_hash=self.model_snapshot_hash,
            selected_model_freeze_hash=self.selected_model_freeze_hash,
            source_execution_hash=self.source_execution_hash,
        )
        if terminal_recovery is None:
            self._service = service
        return ConditionServiceActivation(
            service=service,
            live_identity=identity,
            model_load_event_id=model_load_event_id,
            terminal_recovery=terminal_recovery,
        )

    def detach_for_resume(self) -> None:
        """Transfer a healthy live process to a later controller invocation."""

        service = self._service
        if service is None:
            return
        service.detach_for_controller_restart(self.checkpoint_path)
        self._service = None


class HeldOutScheduleState(ImmutableRecord):
    call_manifest_hash: str
    development_execution_result_hash: str
    development_predecessor_ledger_hash: Sha256Digest
    development_predecessor_allocated_gpu_seconds: float
    remaining_registered_p95_seconds: float
    consumed_primary_service_start_count: int = Field(ge=0, le=3, default=0)
    consumed_long: int = Field(ge=0, le=4, default=0)
    consumed_standard: int = Field(ge=0, le=8, default=0)
    consumed_short: int = Field(ge=0, le=4, default=0)
    completed_result_hashes: Mapping[str, str] = Field(default_factory=dict)
    activation_intents: Mapping[str, HeldOutServiceActivationIntent] = Field(
        default_factory=dict
    )
    session_identities: Mapping[str, HeldOutSessionIdentity] = Field(default_factory=dict)
    shutdown_intents: Mapping[str, HeldOutServiceShutdownIntent] = Field(default_factory=dict)
    shutdown_receipts: Mapping[str, HeldOutServiceShutdownReceipt] = Field(default_factory=dict)
    active_call_id: str | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def service_lifecycle_is_one_to_one(self) -> HeldOutScheduleState:
        if len(self.session_identities) != self.consumed_primary_service_start_count:
            raise ValueError("service-start count differs from persisted condition identities")
        if not set(self.session_identities).issubset(self.activation_intents):
            raise ValueError("service identity exists without its activation intent")
        if not set(self.shutdown_intents).issubset(self.session_identities):
            raise ValueError("shutdown intent exists without a condition service identity")
        if not set(self.shutdown_receipts).issubset(self.session_identities):
            raise ValueError("shutdown receipt exists without a condition service identity")
        if not set(self.shutdown_receipts).issubset(self.shutdown_intents):
            raise ValueError("shutdown receipt exists without its pre-stop intent")
        if any(
            key != intent.condition.value
            for key, intent in self.activation_intents.items()
        ):
            raise ValueError("activation intent is stored under another condition")
        if any(
            key != identity.condition.value
            for key, identity in self.session_identities.items()
        ):
            raise ValueError("persisted service identity is stored under another condition")
        for key, receipt in self.shutdown_receipts.items():
            identity = self.session_identities[key]
            shutdown_intent = self.shutdown_intents[key]
            if (
                key != receipt.condition.value
                or receipt.session_identity_hash != identity.content_hash
                or receipt.allocation_event_id != identity.allocation_event_id
                or receipt.model_load_event_id != identity.model_load_event_id
                or receipt.model_load_event_hash != identity.model_load_event_hash
                or receipt.global_accounting_id != identity.global_accounting_id
                or shutdown_intent.session_identity_hash != identity.content_hash
                or shutdown_intent.allocation_event_id != identity.allocation_event_id
                or shutdown_intent.model_load_event_id != identity.model_load_event_id
                or shutdown_intent.global_accounting_id != identity.global_accounting_id
            ):
                raise ValueError("persisted shutdown receipt differs from its service identity")
        return self


@dataclass(slots=True)
class ProductionHeldOutRuntime:
    """Shared audited runtime with no access to scorer-gold namespaces."""

    repository: Path
    reviewed_plan: ReviewedHeldOutPlan
    configuration: HeldOutControlConfiguration
    development_result: DevelopmentExecutionResult
    development_predecessor_ledger_hash: Sha256Digest
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
                or state.development_predecessor_ledger_hash
                != self.development_predecessor_ledger_hash
            ):
                raise HeldOutProductionError("durable held-out state belongs to another run")
        else:
            self._write_state(
                HeldOutScheduleState(
                    call_manifest_hash=manifest.content_hash,
                    development_execution_result_hash=self.development_result.content_hash,
                    development_predecessor_ledger_hash=(
                        self.development_predecessor_ledger_hash
                    ),
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

    def _global_accounting_id(self) -> str:
        return (
            f"held-out-global-{self.development_result.content_hash[:12]}-"
            f"{self.reviewed_plan.call_manifest.content_hash[:12]}"
        )

    def _read_state(self) -> HeldOutScheduleState:
        if self.state_path.is_symlink() or not self.state_path.is_file():
            raise HeldOutProductionError("held-out schedule state is missing or unsafe")
        return HeldOutScheduleState.model_validate_json(self.state_path.read_bytes())

    def _write_state(self, state: HeldOutScheduleState) -> None:
        parent_existed = self.state_path.parent.exists()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            descriptor = os.open(
                self.state_path.parent.parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
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
            directory_descriptor = os.open(
                self.state_path.parent,
                os.O_RDONLY | os.O_DIRECTORY,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
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
            (
                self.artifacts.ledger.gpu_summary().total_allocated_seconds,
                *(
                    service.actual_allocated_service_seconds
                    for service in self._services.values()
                ),
            )
        )
        if not math.isfinite(allocated):
            raise HeldOutProductionError("cumulative GPU allocation is not finite")
        return GlobalGpuScheduleSnapshot(
            global_accounting_id=self._global_accounting_id(),
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

    def _activation_intent(
        self,
        condition: ConditionName,
        state: HeldOutScheduleState,
    ) -> tuple[HeldOutServiceActivationIntent, HeldOutScheduleState]:
        existing = state.activation_intents.get(condition.value)
        expected_session_id = (
            f"{self.reviewed_plan.call_manifest.manifest_id}-{condition.value}-service"
        )
        expected_event_id = (
            f"{self.reviewed_plan.call_manifest.manifest_id}-{condition.value}-model-load"
        )
        if existing is not None:
            if (
                existing.condition is not condition
                or existing.call_manifest_hash
                != self.reviewed_plan.call_manifest.content_hash
                or existing.development_execution_result_hash
                != self.development_result.content_hash
                or existing.global_accounting_id != self._global_accounting_id()
                or existing.expected_session_id != expected_session_id
                or existing.expected_model_load_event_id != expected_event_id
                or existing.service_start_watchdog_seconds
                != self.configuration.service_start_watchdog_seconds
            ):
                raise HeldOutProductionError("persisted activation intent changed")
            return existing, state
        if condition.value in state.session_identities:
            raise HeldOutProductionError("condition identity lacks its activation intent")
        activator = self.activators[condition]
        launcher_hash = getattr(activator, "launcher_configuration_hash", None)
        if not isinstance(launcher_hash, str) or len(launcher_hash) != 64:
            raise HeldOutProductionError("condition activator lacks its frozen launcher hash")
        intent = HeldOutServiceActivationIntent(
            condition=condition,
            call_manifest_hash=self.reviewed_plan.call_manifest.content_hash,
            development_execution_result_hash=self.development_result.content_hash,
            global_accounting_id=self._global_accounting_id(),
            expected_session_id=expected_session_id,
            expected_model_load_event_id=expected_event_id,
            launcher_configuration_hash=launcher_hash,
            service_start_watchdog_seconds=self.configuration.service_start_watchdog_seconds,
            remaining_registered_seconds_before=state.remaining_registered_p95_seconds,
            requested_at=self._now(),
        )
        intents = dict(state.activation_intents)
        intents[condition.value] = intent
        self._replace_state(state, activation_intents=intents)
        return intent, self._read_state()

    def _persist_shutdown_intent(
        self,
        state: HeldOutScheduleState,
        identity: HeldOutSessionIdentity,
    ) -> tuple[HeldOutServiceShutdownIntent, HeldOutScheduleState]:
        existing = state.shutdown_intents.get(identity.condition.value)
        if existing is not None:
            if (
                existing.condition is not identity.condition
                or existing.session_identity_hash != identity.content_hash
                or existing.allocation_event_id != identity.allocation_event_id
                or existing.model_load_event_id != identity.model_load_event_id
                or existing.global_accounting_id != identity.global_accounting_id
            ):
                raise HeldOutProductionError("persisted shutdown intent changed")
            return existing, state
        intent = HeldOutServiceShutdownIntent(
            condition=identity.condition,
            session_identity_hash=identity.content_hash,
            allocation_event_id=identity.allocation_event_id,
            model_load_event_id=identity.model_load_event_id,
            global_accounting_id=identity.global_accounting_id,
            requested_at=self._now(),
        )
        intents = dict(state.shutdown_intents)
        intents[identity.condition.value] = intent
        self._replace_state(state, shutdown_intents=intents)
        return intent, self._read_state()

    def _persist_session_identity(
        self,
        state: HeldOutScheduleState,
        identity: HeldOutSessionIdentity,
    ) -> HeldOutScheduleState:
        existing = state.session_identities.get(identity.condition.value)
        if existing is not None:
            if existing != identity:
                raise HeldOutProductionError(
                    "persisted condition service identity changed"
                )
            return state
        identities = dict(state.session_identities)
        identities[identity.condition.value] = identity
        remaining = (
            state.remaining_registered_p95_seconds
            - PRIMARY_SERVICE_START_P95_SECONDS
        )
        if remaining < 0:
            raise HeldOutProductionError(
                "condition service load consumed unregistered GPU schedule"
            )
        self._replace_state(
            state,
            session_identities=identities,
            remaining_registered_p95_seconds=remaining,
            consumed_primary_service_start_count=(
                state.consumed_primary_service_start_count + 1
            ),
        )
        return self._read_state()

    def _terminalize_failed_activation(
        self,
        *,
        condition: ConditionName,
        activation_intent: HeldOutServiceActivationIntent,
        service: ShutdownCapableGenerationService,
        allocation_event_id: str,
        identity: HeldOutSessionIdentity | None,
    ) -> None:
        """Stop and account for a service that failed after its activator returned.

        The activation intent and the lower-level service journal are durable before
        the physical load.  We additionally try to persist the exact identity and a
        shutdown intent before stopping.  If the schedule-state write itself is the
        fault, physical shutdown and the immutable service-session row take priority;
        a later controller can reconstruct the held-out receipt from those records.
        """

        if identity is not None:
            # The exact failure under repair can be the atomic state write.
            # The activation intent plus GPU service journal still bind the
            # only authorized allocation, so continue to a verified stop.
            with suppress(BaseException):
                state = self._persist_session_identity(self._read_state(), identity)
                self._identities[condition] = identity
                self._persist_shutdown_intent(state, identity)

        shutdown_error: BaseException | None = None
        try:
            service.shutdown()
        except BaseException as error:
            shutdown_error = error
        terminal = self.artifacts.ledger.get_gpu_service_session(allocation_event_id)
        if terminal is None:
            # Retain ownership so the outer bundle gets one further fail-safe
            # cleanup opportunity instead of losing a possibly live process.
            self._services[condition] = service
            if identity is not None:
                self._identities[condition] = identity
            raise HeldOutProductionError(
                "failed activation could not verify terminal GPU accounting"
            ) from shutdown_error
        if terminal.session_id != activation_intent.expected_session_id:
            raise HeldOutProductionError(
                "failed activation terminal accounting identity changed"
            ) from shutdown_error

        self._services.pop(condition, None)

        if identity is not None:
            # The process is already absent and its complete allocation is
            # immutable in the GPU ledger.  Do not mask the triggering error;
            # restart recovery will replay this exact activation and receipt.
            with suppress(BaseException):
                state = self._persist_session_identity(self._read_state(), identity)
                self._identities[condition] = identity
                _shutdown_intent, state = self._persist_shutdown_intent(state, identity)
                self._persist_terminal_shutdown(state, identity)

        try:
            durable_state = self._read_state()
        except BaseException:
            durable_state = None
        if durable_state is None:
            self._identities.pop(condition, None)
            self._shutdowns.pop(condition, None)
            return
        durable_identity = durable_state.session_identities.get(condition.value)
        if durable_identity is None:
            self._identities.pop(condition, None)
        else:
            self._identities[condition] = durable_identity
        durable_shutdown = durable_state.shutdown_receipts.get(condition.value)
        if durable_shutdown is None:
            self._shutdowns.pop(condition, None)
        else:
            self._shutdowns[condition] = durable_shutdown

    def _persist_terminal_shutdown(
        self,
        state: HeldOutScheduleState,
        identity: HeldOutSessionIdentity,
    ) -> tuple[HeldOutServiceShutdownReceipt, GlobalGpuScheduleSnapshot]:
        session = self.artifacts.ledger.get_gpu_service_session(identity.allocation_event_id)
        if session is None or session.session_id != (
            f"{self.reviewed_plan.call_manifest.manifest_id}-{identity.condition.value}-service"
        ):
            raise HeldOutProductionError(
                "condition service stop lacks its terminal allocation record"
            )
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
        shutdowns = dict(state.shutdown_receipts)
        shutdowns[identity.condition.value] = receipt
        self._replace_state(state, shutdown_receipts=shutdowns)
        self._shutdowns[identity.condition] = receipt
        self._services.pop(identity.condition, None)
        return receipt, snapshot

    def service_available(self, condition: ConditionName) -> bool:
        """Report only a controller-owned, nonterminal service as callable."""

        return condition in self._services and condition not in self._shutdowns

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
        if persisted is None:
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
        elif (
            before.actual_allocated_gpu_seconds >= self.configuration.hard_gpu_seconds_limit
            or before.actual_allocated_gpu_seconds + state.remaining_registered_p95_seconds
            > self.configuration.scheduled_gpu_seconds_limit
        ):
            raise HeldOutProductionError(
                "condition service resume is not admissible under the cumulative GPU gates"
            )
        activation_intent, state = self._activation_intent(condition, state)
        persisted = state.session_identities.get(condition.value)
        shutdown_intent = state.shutdown_intents.get(condition.value)
        if persisted is not None and shutdown_intent is not None:
            terminal = self.artifacts.ledger.get_gpu_service_session(
                persisted.allocation_event_id
            )
            if terminal is not None:
                self._identities[condition] = persisted
                self._persist_terminal_shutdown(state, persisted)
                return persisted
        prior_events = tuple(
            item
            for item in self.artifacts.ledger.gpu_events()
            if item.event_id == activation_intent.expected_model_load_event_id
        )
        activation = self.activators[condition](
            remaining_required_seconds=max(
                0.0,
                activation_intent.remaining_registered_seconds_before
                - PRIMARY_SERVICE_START_P95_SECONDS,
            ),
            resume_required=persisted is not None or bool(prior_events),
            activation_intent=activation_intent,
        )
        service = activation.service
        live = activation.live_identity
        model_load_event_id = activation.model_load_event_id
        cleanup_identity: HeldOutSessionIdentity | None = None
        try:
            if activation.service_ready:
                # Ownership begins immediately when the activator returns.  Every
                # later validation/state-write failure is therefore visible to the
                # runtime and to the bundle cleanup boundary.
                self._services[condition] = service
            events = tuple(
                item
                for item in self.artifacts.ledger.gpu_events()
                if item.event_id == model_load_event_id
            )
            if len(events) != 1 or events[0].event_kind is not GpuEventKind.MODEL_LOAD:
                raise HeldOutProductionError(
                    "condition activation lacks one metered model load"
                )
            if persisted is not None:
                if (
                    persisted.condition is not condition
                    or persisted.service_identity_hash != live.content_hash
                    or persisted.service_pid != live.service_pid
                    or persisted.service_start_ticks != live.service_start_ticks
                    or persisted.allocation_event_id != live.gpu_session_event_id
                    or persisted.model_load_event_id != model_load_event_id
                    or persisted.model_load_event_hash != _event_hash(events[0])
                    or persisted.global_accounting_id != self._global_accounting_id()
                ):
                    self._identities[condition] = persisted
                    self.shutdown_condition(persisted)
                    raise HeldOutProductionError(
                        "interrupted condition service identity changed during adoption"
                    )
                cleanup_identity = persisted
                self._identities[condition] = persisted
                if not activation.service_ready:
                    _intent, state = self._persist_shutdown_intent(state, persisted)
                    self._persist_terminal_shutdown(state, persisted)
                    return persisted
                resumed_snapshot = self.schedule_snapshot()
                if (
                    resumed_snapshot.actual_allocated_gpu_seconds
                    >= self.configuration.hard_gpu_seconds_limit
                    or resumed_snapshot.actual_allocated_gpu_seconds
                    + state.remaining_registered_p95_seconds
                    > self.configuration.scheduled_gpu_seconds_limit
                ):
                    self.shutdown_condition(persisted)
                    raise HeldOutProductionError(
                        "recovered service allocation crossed a cumulative GPU gate"
                    )
                if condition.value in state.shutdown_intents:
                    self.shutdown_condition(persisted)
                return persisted
            identity = HeldOutSessionIdentity(
                session_id=(
                    f"{self.reviewed_plan.call_manifest.manifest_id}-"
                    f"{condition.value}-session"
                ),
                condition=condition,
                service_identity_hash=live.content_hash,
                service_pid=live.service_pid,
                service_start_ticks=live.service_start_ticks,
                global_accounting_id=self._global_accounting_id(),
                global_ledger_chain_hash=gpu_ledger_chain_hash(self.artifacts),
                allocation_event_id=live.gpu_session_event_id,
                model_load_event_id=model_load_event_id,
                model_load_event_hash=_event_hash(events[0]),
            )
            if any(
                identity.service_identity_hash == item.service_identity_hash
                or identity.allocation_event_id == item.allocation_event_id
                or identity.model_load_event_hash == item.model_load_event_hash
                or (identity.service_pid, identity.service_start_ticks)
                == (item.service_pid, item.service_start_ticks)
                for item in self._identities.values()
            ):
                raise HeldOutProductionError(
                    "condition activation reused a prior service allocation"
                )
            cleanup_identity = identity
            self._identities[condition] = identity
            state = self._persist_session_identity(state, identity)
            remaining = state.remaining_registered_p95_seconds
            if not activation.service_ready:
                _shutdown_intent, state = self._persist_shutdown_intent(state, identity)
                self._persist_terminal_shutdown(state, identity)
            after_allocated = max(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds,
                service.actual_allocated_service_seconds,
            )
            if (
                after_allocated >= self.configuration.hard_gpu_seconds_limit
                or after_allocated + remaining
                > self.configuration.scheduled_gpu_seconds_limit
            ):
                if self.service_available(condition):
                    self.shutdown_condition(identity)
                raise HeldOutProductionError(
                    "condition service load crossed a cumulative GPU gate"
                )
            if activation.service_ready and condition.value in state.shutdown_intents:
                self.shutdown_condition(identity)
            return identity
        except BaseException:
            if activation.service_ready:
                self._terminalize_failed_activation(
                    condition=condition,
                    activation_intent=activation_intent,
                    service=service,
                    allocation_event_id=live.gpu_session_event_id,
                    identity=cleanup_identity,
                )
            raise

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

    def _stage_manifest(self, stage: PublicStageReference) -> RuntimeStagingManifest:
        """Reconstruct the reviewed manifest without reopening a stage pathname."""

        if stage.query_artifact_hash is None or stage.stage_kind != "query_revealed":
            raise HeldOutProductionError("query-stage reference is incomplete")
        manifest = RuntimeStagingManifest(
            stage_id=stage.stage_id,
            stage_kind=stage.stage_kind,
            artifact_hashes=(
                stage.evidence_artifact_hash,
                stage.query_artifact_hash,
            ),
            file_names=("evidence.json", "query.json"),
        )
        if manifest.content_hash != stage.staging_manifest_hash:
            raise HeldOutProductionError("query-stage logical manifest changed")
        return manifest

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
                persisted.query_context is None
                or persisted.packet_materialization is None
                or persisted.packet_materialization_artifact is None
            ):
                raise HeldOutProductionError(
                    "persisted query opening lacks restart-safe semantic lineage"
                )
            materialization = self.resolver.resolve_record(
                persisted.packet_materialization_artifact,
                PacketMaterializationEvent,
            )
            if (
                event != persisted.query_access_event
                or packet.content_hash != persisted.evidence_packet_hash
                or materialization != persisted.packet_materialization
                or persisted.query_context.content_hash != event.query_context_hash
            ):
                raise HeldOutProductionError("persisted query opening CAS lineage changed")
            return persisted
        manifest = self._stage_manifest(stage)
        opening = self.query_runtime.open_query(
            staging_root=None,
            manifest=manifest,
            barrier=barrier,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            access_event_id=f"query-access-{stage.stage_id}",
            repository=self.repository,
            stage_relative_path=stage.relative_path,
            manifest_file_sha256=stage.manifest_file_sha256,
        )
        materialized = self._materialize(opening)
        access_record = self.artifacts.ledger.get_artifact(
            opening.persistence.access_event_artifact_hash
        )
        packet_record = self.artifacts.ledger.get_artifact(
            materialized.persistence.packet_artifact_hash
        )
        materialization_record = self.artifacts.ledger.get_artifact(
            materialized.persistence.materialization_event_artifact_hash
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
            query_context=opening.context,
            packet_materialization=materialized.event,
            packet_materialization_artifact=self.resolver.reference(
                materialization_record,
                logical_content_hash=materialized.event.content_hash,
                object_kind="packet_materialization_event",
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
        if persisted_receipt != receipt or receipt.call_spec_hash != call.content_hash:
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
        for raw_reference in (receipt.raw_response, receipt.repair_raw_response):
            if raw_reference is None:
                continue
            raw_record = self.artifacts.ledger.get_artifact(raw_reference.artifact_hash)
            raw_bytes = self.artifacts.blobs.read_bytes(
                raw_record,
                allow_restricted=True,
            )
            if (
                raw_record.media_type != raw_reference.media_type
                or raw_record.release_class is not ReleaseClass.RESTRICTED
                or hashlib.sha256(raw_bytes).hexdigest()
                != raw_reference.logical_content_hash
            ):
                raise HeldOutProductionError("raw response CAS lineage changed")
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
        repair_semantic: PreconstructionRequest | ConstructionRequest | None = None
        repair_packing: PackingReport | None = None
        if receipt.repair_semantic_request is not None:
            repair_semantic = self.resolver.resolve_record(
                receipt.repair_semantic_request,
                semantic_type,
                required_release=ReleaseClass.RESTRICTED,
            )
            repair_packing_reference = receipt.repair_packing_report
            if repair_packing_reference is None:
                raise HeldOutProductionError("repair lost its packing report")
            repair_packing = self.resolver.resolve_record(
                repair_packing_reference,
                PackingReport,
                required_release=ReleaseClass.RESTRICTED,
            )
        generation = None
        if receipt.validation is not None:
            generation = self.resolver.resolve_record(
                receipt.validation,
                ValidatedGeneration,
                required_release=ReleaseClass.RESTRICTED,
            )
            expected_semantic_hash = semantic.content_hash
            expected_packing_hash = packing.content_hash
            if repair_semantic is not None and repair_packing is not None:
                expected_semantic_hash = repair_semantic.content_hash
                expected_packing_hash = repair_packing.content_hash
            if (
                generation.request_hash != expected_semantic_hash
                or generation.packing_report_hash != expected_packing_hash
                or generation.repair_attempt
                != int(receipt.repair_semantic_request is not None)
            ):
                raise HeldOutProductionError("validated generation repair lineage changed")
        if fixed_inventory is not None and generation is not None:
            enforce_fixed_select_draft(
                generation.draft,
                sealed=fixed_inventory,
                seed_block=call.seed_block,
            )
        output_value: ConditionPreparation | ConditionAttemptRecord | None = None
        if receipt.output is not None:
            output_type = (
                ConditionPreparation
                if call.condition is ConditionName.C1_LLM_PRE
                else ConditionAttemptRecord
            )
            output_value = self.resolver.resolve_record(
                receipt.output,
                output_type,
                required_release=ReleaseClass.RESTRICTED,
            )
        if call.condition is ConditionName.C1_LLM_PRE and result.outcome is RunOutcome.SUCCEEDED:
            if not isinstance(output_value, ConditionPreparation):
                raise HeldOutProductionError("successful C1 has no typed preparation output")
            bindings = {item.condition: item for item in result.prequery_preparation_bindings}
            fixed_binding = bindings.get(ConditionName.A_FIXED_SELECT)
            if fixed_binding is None:
                raise HeldOutProductionError("successful C1 lacks its FixedSelect preparation")
            fixed_preparation = prepare_fixed_selection(
                output_value,
                seed_block=call.seed_block,
                prepared_at=fixed_binding.completed_at,
            )
            fixed_lineage = fixed_preparation.fixed_selection
            if (
                fixed_preparation.content_hash != fixed_binding.preparation_hash
                or fixed_lineage is None
                or fixed_lineage.content_hash != fixed_binding.lineage_artifact_hash
            ):
                raise HeldOutProductionError(
                    "FixedSelect pre-query binding is not derivable from sealed C1"
                )
        ledger_pairs = [
            (
                receipt.gpu_event_id,
                receipt.gpu_event_hash,
                receipt.model_call_id,
                receipt.model_call_record_hash,
            )
        ]
        if receipt.repair_gpu_event_id is not None:
            ledger_pairs.append(
                (
                    receipt.repair_gpu_event_id,
                    cast(str, receipt.repair_gpu_event_hash),
                    cast(str, receipt.repair_model_call_id),
                    cast(str, receipt.repair_model_call_record_hash),
                )
            )
        allocated = 0.0
        for event_id, event_hash, model_call_id, model_call_hash in ledger_pairs:
            event = next(
                (
                    item
                    for item in self.artifacts.ledger.gpu_events()
                    if item.event_id == event_id
                ),
                None,
            )
            model_call = self.artifacts.ledger.get_model_call(model_call_id)
            if (
                event is None
                or _event_hash(event) != event_hash
                or canonical_sha256(asdict(model_call)) != model_call_hash
            ):
                raise HeldOutProductionError("GPU/model-call ledger receipt changed")
            allocated += event.allocated_seconds
        if not math.isclose(allocated, result.allocated_gpu_seconds, abs_tol=1e-6):
            raise HeldOutProductionError("result GPU allocation differs from its events")

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
        _shutdown_intent, state = self._persist_shutdown_intent(state, identity)
        service = self._services.get(identity.condition)
        if service is None:
            return self._persist_terminal_shutdown(state, identity)
        service.shutdown()
        return self._persist_terminal_shutdown(self._read_state(), identity)

    def detach_live_services_for_resume(self) -> None:
        """Detach only verified live services, retaining their durable identities."""

        for condition in tuple(self._services):
            activator = self.activators[condition]
            activator.detach_for_resume()
            self._services.pop(condition, None)

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

    def service_available(self) -> bool:
        return self.runtime.service_available(self.condition)

    def execute_call(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult:
        if call.condition is not self.condition:
            raise HeldOutProductionError("call routed through another condition session")
        if not self.service_available():
            raise HeldOutProductionError("condition service is terminal and cannot execute calls")
        state = self.runtime._read_state()
        if state.active_call_id not in {None, call.call_id}:
            raise HeldOutProductionError("another held-out call remains active")
        if call.call_id in state.completed_result_hashes:
            raise HeldOutProductionError("completed call cannot be issued again")
        self.runtime._replace_state(state, active_call_id=call.call_id)
        reserve_field, reserve_total = {
            "reserve_long": ("consumed_long", 4),
            "reserve_standard": ("consumed_standard", 8),
            "reserve_short": ("consumed_short", 4),
        }[call.repair_reserve_class]
        result = self.runtime.semantic_executor.execute(
            service=self.runtime._services[self.condition],
            call=call,
            envelope=envelope,
            remaining_required_seconds=max(
                0.0,
                state.remaining_registered_p95_seconds - call.p95_seconds,
            ),
            repair_allowed=getattr(state, reserve_field) < reserve_total,
        )
        self._finalize_state(call, result)
        return result

    def recover_call(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> HeldOutServiceResult | None:
        state = self.runtime._read_state()
        result = self.runtime.semantic_executor.recover(call=call, envelope=envelope)
        if result is None:
            if self.runtime.semantic_executor.recovery_pending(
                call=call,
                envelope=envelope,
            ):
                return None
            # The append-only controller slot can survive a power loss before
            # the semantic intent is written. With no matching intent or
            # result, issuing this exact slotted call is not a duplicate.
            return self.execute_call(call, envelope)
        if call.call_id not in state.completed_result_hashes:
            self._finalize_state(call, result)
        elif state.completed_result_hashes[call.call_id] != result.content_hash:
            raise HeldOutProductionError("recovered result differs from durable state")
        return result

    def recovery_pending(
        self, call: HeldOutCallSpec, envelope: HeldOutCallEnvelope
    ) -> bool:
        if call.condition is not self.condition:
            raise HeldOutProductionError("recovery query was routed through another session")
        return self.runtime.semantic_executor.recovery_pending(
            call=call,
            envelope=envelope,
        )

    def finalize_unstarted_call(
        self, call: HeldOutCallSpec, result: HeldOutServiceResult
    ) -> GlobalGpuScheduleSnapshot:
        if call.condition is not self.condition or result.condition is not self.condition:
            raise HeldOutProductionError("unstarted call was routed through another session")
        if result.request_started or result.call_id != call.call_id:
            raise HeldOutProductionError("only the exact unstarted call may be finalized")
        self._finalize_state(call, result)
        return self.schedule_snapshot()

    def _finalize_state(self, call: HeldOutCallSpec, result: HeldOutServiceResult) -> None:
        state = self.runtime._read_state()
        completed = dict(state.completed_result_hashes)
        prior = completed.get(call.call_id)
        if prior is not None:
            if prior != result.content_hash:
                raise HeldOutProductionError("call completion changed during resume")
            return
        # Every terminal ITT slot, including a dependency skip or a condition
        # stopped by a prior timeout, removes its registered p95 work from the
        # remaining mandatory forecast. Only an actual repair consumes a reserve.
        remaining = state.remaining_registered_p95_seconds - call.p95_seconds
        updates: dict[str, object] = {}
        if result.request_started:
            remaining -= result.repair_attempts * call.watchdog_seconds
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


@dataclass(slots=True)
class ProductionHeldOutBundle:
    cpu: object
    sessions: Mapping[ConditionName, InjectedHeldOutSession]
    runtime: ProductionHeldOutRuntime
    close_callback: Callable[[], None] | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        for condition in tuple(self.runtime._services):
            identity = self.runtime._identities.get(condition)
            if identity is None:
                raise HeldOutProductionError("live service lacks its shutdown identity")
            self.runtime.shutdown_condition(identity)
        if self.close_callback is not None:
            self.close_callback()
        self._closed = True

    def preserve_for_resume(self) -> None:
        """Detach a healthy service; fall back to verified shutdown on failure."""

        if self._closed:
            return
        try:
            self.runtime.detach_live_services_for_resume()
        except BaseException:
            # A service that cannot prove a safe handoff must not be left
            # allocating unattended.
            for condition in tuple(self.runtime._services):
                identity = self.runtime._identities.get(condition)
                if identity is None:
                    raise HeldOutProductionError(
                        "failed handoff left a live service without its identity"
                    ) from None
                self.runtime.shutdown_condition(identity)
            if self.close_callback is not None:
                self.close_callback()
            self._closed = True
            raise
        if self.close_callback is not None:
            self.close_callback()
        self._closed = True


def create_production_held_out_bundle(
    *,
    repository: Path,
    reviewed_plan: ReviewedHeldOutPlan,
    configuration: HeldOutControlConfiguration,
    development_result: DevelopmentExecutionResult,
    development_predecessor_ledger_hash: Sha256Digest,
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
        development_predecessor_ledger_hash=development_predecessor_ledger_hash,
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
    "ConditionServiceActivation",
    "ConditionServiceActivator",
    "HeldOutArtifactResolver",
    "HeldOutProductionError",
    "HeldOutScheduleState",
    "HeldOutSemanticExecutor",
    "HeldOutServiceActivationIntent",
    "HeldOutServiceShutdownIntent",
    "ProductionHeldOutBundle",
    "ProductionHeldOutRuntime",
    "ProductionHeldOutSession",
    "ResumableVLLMService",
    "SequentialVLLMConditionActivator",
    "ShutdownCapableGenerationService",
    "create_production_held_out_bundle",
    "gpu_ledger_chain_hash",
]
