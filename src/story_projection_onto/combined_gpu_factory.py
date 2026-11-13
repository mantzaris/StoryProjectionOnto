"""Concrete pinned production factory for the registered combined GPU block.

This is the only factory accepted by ``run_combined_gpu_block.py``.  Construction
performs CPU-only validation; vLLM is started only by ``activate`` after the outer
controller has durably written its activation slot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from story_projection_onto.benchmark_runtime import (
    NeutralEvidenceArtifact,
    RuntimeStagingManifest,
    load_staged_neutral_evidence,
    load_staged_world,
)
from story_projection_onto.combined_gpu_block import (
    CombinedBlockConfiguration,
    CombinedCallClass,
    CombinedCallManifest,
    CombinedCallSpec,
    CombinedRuntimeBinding,
    CombinedUpstreamGate,
    load_registered_combined_selections,
)
from story_projection_onto.combined_gpu_materialize import (
    verify_combined_compiler_receipt,
)
from story_projection_onto.combined_gpu_production import (
    _GPU_EVENT_FIELDS,
    _MODEL_CALL_FIELDS,
    CASCombinedInputProvider,
    CombinedActivationSlot,
    CombinedLifecycleOwner,
    CombinedOwnedService,
    CombinedPreparedCall,
    CombinedProductionError,
    CombinedRecoveryRequired,
    CombinedRepairClaim,
    CombinedServiceCallResult,
    CombinedServiceIdentity,
    CombinedServiceShutdownReceipt,
    RepairAuthority,
    RestrictedArtifactPointer,
    _append_bytes_atomically,
    _load_public_evidence_packet,
    _record_hash,
    _require_contained_writable_path,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ConditionAttemptRecord,
    ConditionPreparation,
    RunConditionConfig,
)
from story_projection_onto.conditions.c2 import (
    build_c2_construction_request,
    failed_c2_attempt,
    finalize_c2_draft,
)
from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    ImmutableRecord,
    NoTemporalEpistemicOntologyDraft,
    PacketMaterializationEvent,
    PrequeryBarrier,
    QueryAccessEvent,
    ReleaseClass,
    RunOutcome,
    RuntimeIdentifiers,
    ValidatedGeneration,
    canonical_sha256,
    normalize_generation_metadata,
    runtime_structural_acceptance_record,
    to_model_visible_query,
)
from story_projection_onto.development_adapter import (
    DevelopmentConstructionConfiguration,
    PackingTokenizer,
    build_development_guided_request,
    development_request_runtime,
    encode_development_semantic_request,
    restore_model_output_source_aliases,
)
from story_projection_onto.development_continuation import development_validator_hash
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    ResourceLimits,
    StorageAllocationPlan,
)
from story_projection_onto.fallback_acceptance import validate_source_association
from story_projection_onto.feedback_runtime import FeedbackAttemptStatus
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    GenerationResult,
    GuidedJSONRequest,
    ResourceSampler,
    TokenizerManifest,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    capture_tokenizer_manifest,
)
from story_projection_onto.held_out_execution import (
    FrozenHeldOutSemanticExecutor,
    HeldOutProduceInputs,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    normalize_no_temporal_epistemic_draft,
)
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    validate_fallback_snapshot_manifest,
)
from story_projection_onto.phase5_execution import (
    C2RegenerationRequest,
    Phase5ExecutionInputManifest,
    PrimaryHeldOutResultsGate,
)
from story_projection_onto.phase5_production import (
    Phase5CASReference,
    Phase5OwnedServiceIdentity,
    Phase5OwnedServiceResult,
    persist_phase5_record,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    Compression,
    FailureKind,
    GpuEventKind,
    JobState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    PacketMaterializationRecord,
    QueryAccessRecord,
    RetryClass,
    StorageBudget,
    StoragePreflight,
    StorageReport,
)
from story_projection_onto.validate import (
    validate_draft_structure,
    validate_repair_preservation,
)


class CombinedFactoryError(CombinedProductionError):
    """A production dependency differs from the frozen combined runtime."""


def _registered_combined_storage_preflights(
    repository: Path,
    storage: StoragePreflight,
) -> tuple[tuple[str, StorageReport], ...]:
    plan = StorageAllocationPlan.load(
        repository / "configs/study/storage_phase_allocations.json"
    )
    return tuple(
        (
            phase,
            storage.check(**plan.reservation_for(phase).preflight_arguments()),
        )
        for phase in ("phase_4", "phase_5")
    )


def _require_registered_combined_storage_preflights(
    repository: Path,
    storage: StoragePreflight,
    ledger: Ledger,
) -> tuple[tuple[str, StorageReport], ...]:
    reports = _registered_combined_storage_preflights(repository, storage)
    for phase, report in reports:
        ledger.record_storage_sample(
            report,
            phase=f"{phase}:combined_gpu_block:factory",
        )
    if any(not report.allowed for _phase, report in reports):
        raise CombinedFactoryError("combined storage preflight failed")
    return reports


def _require_registered_phase_five_transition_storage_preflight(
    repository: Path,
    storage: StoragePreflight,
    ledger: Ledger,
) -> StorageReport:
    plan = StorageAllocationPlan.load(
        repository / "configs/study/storage_phase_allocations.json"
    )
    report = storage.check(
        **plan.reservation_for("phase_5").preflight_arguments()
    )
    ledger.record_storage_sample(
        report,
        phase="phase_5:combined_gpu_block:transition",
    )
    if not report.allowed:
        raise CombinedFactoryError("combined Phase 5 storage preflight failed")
    return report


class _SemanticIntent(ImmutableRecord):
    request_kind: Literal["ordinary", "phase5"] = "ordinary"
    run_id: str
    call_id: str
    call_spec_hash: str
    prepared_hash: str
    guided_request_hash: str
    gpu_event_id: str
    service_identity_hash: str
    cumulative_gpu_seconds_before: float = Field(ge=0.0)
    remaining_required_seconds: float = Field(gt=0.0)
    created_at: AwareDatetime
    ordinary_prepared_call: CombinedPreparedCall | None = None
    phase5_request: C2RegenerationRequest | None = None

    @model_validator(mode="after")
    def exact_request_payload(self) -> _SemanticIntent:
        ordinary = self.ordinary_prepared_call
        phase5 = self.phase5_request
        if self.request_kind == "ordinary":
            # Legacy pre-recovery intents did not embed the prepared call.  They
            # remain readable for terminal-completion verification but
            # ``_reconstruct_completion`` rejects them if that checkpoint is
            # absent.  Every newly issued intent supplies this payload.
            if phase5 is not None:
                raise ValueError("ordinary semantic intent cannot contain a Phase 5 request")
            if ordinary is not None and ordinary.content_hash != self.prepared_hash:
                raise ValueError("ordinary semantic intent prepared-call hash changed")
        elif phase5 is None or ordinary is not None:
            raise ValueError("Phase 5 semantic intent requires exactly its regeneration request")
        elif phase5.content_hash != self.prepared_hash:
            raise ValueError("Phase 5 semantic intent request hash changed")
        return self


class _OrdinarySemanticCompletion(ImmutableRecord):
    """Exact post-inference checkpoint sufficient to rebuild the service result."""

    run_id: str
    call_id: str
    call_spec_hash: str
    prepared_hash: str
    semantic_request_hash: str
    semantic_intent_hash: str
    service_identity_hash: str
    condition_attempt: ConditionAttemptRecord
    model_call_ids: tuple[str, ...]
    model_request_hashes: tuple[str, ...]
    cumulative_gpu_seconds_before: float = Field(ge=0.0)
    cumulative_gpu_seconds_after: float = Field(ge=0.0)
    started_at: AwareDatetime
    completed_at: AwareDatetime


class _RepairDisposition(ImmutableRecord):
    """Durable one-way decision made after an invalid terminal base output.

    A claimed disposition is written after the outer global-reserve claim and
    before any repair attempt row/GPU event.  An exhausted disposition is the
    terminal proof that the invalid base must enter ITT without a repair.
    """

    run_id: str
    call_id: str
    call_spec_hash: str
    request_hash: str
    base_model_call_id: str
    base_raw_output_hash: str
    repair_semantic_request_hash: str
    repair_guided_request_hash: str
    decision: Literal["repair_claimed", "reserve_exhausted"]
    repair_claim: CombinedRepairClaim | None = None
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def exact_terminal_decision(self) -> _RepairDisposition:
        if self.decision == "repair_claimed":
            claim = self.repair_claim
            if (
                claim is None
                or claim.call_id != self.call_id
                or claim.call_spec_hash != self.call_spec_hash
                or claim.request_hash != self.request_hash
                or claim.base_model_call_id != self.base_model_call_id
            ):
                raise ValueError("claimed repair disposition has no exact reserve claim")
        elif self.repair_claim is not None:
            raise ValueError("exhausted repair disposition cannot retain a reserve claim")
        return self


class _LifecycleBinding(ImmutableRecord):
    run_id: str
    activation_slot_hash: str
    service_identity: CombinedServiceIdentity
    checkpoint_file_sha256: str
    created_at: AwareDatetime


class _ActivationIntent(ImmutableRecord):
    """Path-free durable identity written before the sole physical model start."""

    run_id: str
    activation_slot_hash: str
    activation_slot: CombinedActivationSlot
    service_session_id: str
    gpu_event_id: str
    launcher_configuration_hash: str
    created_at: AwareDatetime


class _ShutdownIntent(ImmutableRecord):
    run_id: str
    service_identity_hash: str
    service_pid: int
    service_start_ticks: int
    requested_at: AwareDatetime


@dataclass(slots=True)
class CombinedProductionBundle:
    provider: CASCombinedInputProvider
    lifecycle_owner: CombinedLifecycleOwner
    artifacts: ArtifactStore
    phase5_storage_preflight: Callable[[], None]


def _safe_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise CombinedFactoryError(f"{label} must be a bounded regular file")
    return path.resolve(strict=True)


def _load_cas_record(
    artifacts: ArtifactStore,
    artifact_hash: str,
    logical_hash: str,
    model: type[ImmutableRecord],
    label: str,
) -> ImmutableRecord:
    try:
        record = artifacts.ledger.get_artifact(artifact_hash)
        if record.release_class.value != "restricted":
            raise CombinedFactoryError(f"{label} is not restricted")
        value = model.model_validate_json(
            artifacts.blobs.read_bytes(record, allow_restricted=True)
        )
    except CombinedFactoryError:
        raise
    except Exception as error:
        raise CombinedFactoryError(f"cannot resolve {label}: {error}") from error
    if value.content_hash != logical_hash:
        raise CombinedFactoryError(f"{label} logical hash changed")
    return value


def _load_neutral_sources(
    repository: Path,
    manifest: CombinedCallManifest,
) -> dict[str, NeutralEvidenceArtifact]:
    result: dict[str, NeutralEvidenceArtifact] = {}
    neutral_root = repository / "data/synthetic/condition_inputs/neutral_evidence"
    for call in manifest.calls:
        source = call.source
        if source.unit_id in result:
            continue
        stage_root = (repository / source.prequery_stage.relative_path).resolve(strict=True)
        if not stage_root.is_relative_to(repository):
            raise CombinedFactoryError("combined prequery stage escaped the repository")
        stage_manifest = RuntimeStagingManifest.model_validate_json(
            (stage_root / "manifest.json").read_bytes()
        )
        visible = load_staged_world(stage_root / "evidence.json", stage_root, stage_manifest)
        suffix = stage_root.name.removeprefix("artifact_")
        neutral_stage = neutral_root / f"neutral_{suffix}"
        neutral_manifest = RuntimeStagingManifest.model_validate_json(
            (neutral_stage / "manifest.json").read_bytes()
        )
        neutral, certificate = load_staged_neutral_evidence(
            neutral_stage,
            neutral_root,
            neutral_manifest,
            visible,
        )
        if (
            neutral.snapshot.content_hash != source.prequery_stage.snapshot_hash
            or certificate.model_visible_evidence_artifact_hash
            != source.prequery_stage.evidence_artifact_hash
        ):
            raise CombinedFactoryError("combined neutral/model-visible evidence changed")
        result[source.unit_id] = neutral
    return result


def _artifact_value(
    artifacts: ArtifactStore,
    artifact_hash: str,
    model: type[ImmutableRecord],
    logical_hash: str,
) -> ImmutableRecord:
    record = artifacts.ledger.get_artifact(artifact_hash)
    value = model.model_validate_json(
        artifacts.blobs.read_bytes(record, allow_restricted=True)
    )
    if value.content_hash != logical_hash:
        raise CombinedFactoryError("controller audit CAS logical hash changed")
    return value


class _ProductionInputProvider(CASCombinedInputProvider):
    """CAS provider that can also reconstruct fact-free finalization inputs."""

    def __init__(
        self,
        *,
        runtime: CombinedRuntimeBinding,
        construction: DevelopmentConstructionConfiguration,
        artifacts: ArtifactStore,
        neutral_by_unit: Mapping[str, NeutralEvidenceArtifact],
    ) -> None:
        super().__init__(
            runtime=runtime,
            upper_ontology=construction.upper_ontology,
            artifacts=artifacts,
        )
        self.construction = construction
        self.neutral_by_unit = dict(neutral_by_unit)

    def _preparation(self, call: CombinedCallSpec) -> ConditionPreparation:
        if call.condition is ConditionName.C2_LLM_QUERY:
            artifact_hash = call.source.c2_seed1_preparation_artifact_hash
            logical_hash = call.source.c2_seed1_preparation_hash
        else:
            lineage = call.source.preparation_for(call.condition)
            artifact_hash = lineage.preparation_artifact_hash
            logical_hash = lineage.preparation_hash
        value = _load_cas_record(
            self.artifacts,
            artifact_hash,
            logical_hash,
            ConditionPreparation,
            "combined condition preparation",
        )
        return cast(ConditionPreparation, value)

    def _barrier(self, call: CombinedCallSpec) -> PrequeryBarrier:
        row = self.artifacts.ledger.get_prequery_barrier(call.source.prequery_barrier_hash)
        value = _artifact_value(
            self.artifacts,
            row.barrier_artifact_hash,
            PrequeryBarrier,
            row.barrier_hash,
        )
        return cast(PrequeryBarrier, value)

    def _original_opening(
        self,
        call: CombinedCallSpec,
    ) -> tuple[QueryAccessEvent, PacketMaterializationEvent]:
        access_row = self.artifacts.ledger.get_query_access(call.source.query_access_event_hash)
        access = cast(
            QueryAccessEvent,
            _artifact_value(
                self.artifacts,
                access_row.access_event_artifact_hash,
                QueryAccessEvent,
                access_row.access_event_hash,
            ),
        )
        ledger_path = self.artifacts.ledger.path.absolute()
        if ledger_path.is_symlink() or not ledger_path.is_file():
            raise CombinedFactoryError("combined ledger is missing or unsafe")
        connection = sqlite3.connect(
            f"file:{ledger_path.as_posix()}?mode=ro", uri=True, timeout=30
        )
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT materialization_event_hash, materialization_event_artifact_hash "
                "FROM packet_materialization_events WHERE query_access_event_hash = ?",
                (access.content_hash,),
            ).fetchall()
        finally:
            connection.close()
        if len(rows) != 1:
            raise CombinedFactoryError("query access lacks one exact packet materialization")
        event = cast(
            PacketMaterializationEvent,
            _artifact_value(
                self.artifacts,
                rows[0]["materialization_event_artifact_hash"],
                PacketMaterializationEvent,
                rows[0]["materialization_event_hash"],
            ),
        )
        return access, event

    def produce_inputs(
        self,
        call: CombinedCallSpec,
        prepared: CombinedPreparedCall,
    ) -> HeldOutProduceInputs:
        packet = _load_public_evidence_packet(
            artifacts=self.artifacts,
            reference=call.source.packet_artifact,
        )
        if prepared.paraphrase_reveal is None:
            access, materialization = self._original_opening(call)
        else:
            access = prepared.paraphrase_reveal.access_event
            materialization = prepared.paraphrase_reveal.packet_reuse_event
        runtime = self.runtime
        run_config = RunConditionConfig(
            config_id=f"combined-run-config-{call.call_id}",
            condition=call.condition,
            budgets=call.source.context.budgets,
            maximum_input_tokens=runtime.maximum_input_tokens,
            maximum_output_tokens=runtime.maximum_output_tokens,
            repair_attempt_budget=call.maximum_repair_attempts,
            seed_block=call.seed_block,
            model_stack_hash=runtime.model_manifest_hash,
            decoding_manifest_hash=runtime.decoding_manifest_hashes[call.condition],
            decoding_family_hash=runtime.decoding_family_hash,
            seed_manifest_hash=runtime.seed_manifest_hash,
            resolved_seed=call.vllm_seed,
            prompt_hash=runtime.prompt_hashes[call.condition],
            output_schema_hash=runtime.output_schema_hashes[call.condition],
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            capability_manifest_hash=runtime.capability_manifest_hashes[call.condition],
            validator_hash=runtime.validator_hash,
            upper_ontology_hash=runtime.upper_ontology_hash,
        )
        context = call.paraphrase_context or call.source.context
        return HeldOutProduceInputs(
            controller_unit_id=call.source.unit_id,
            preparation=self._preparation(call),
            snapshot=self.neutral_by_unit[call.source.unit_id].snapshot,
            packet=packet,
            context=context,
            query_access=access,
            prequery_barrier=self._barrier(call),
            query_processing_started_at=prepared.request.requested_at,
            packet_materialization=materialization,
            upper_ontology=self.construction.upper_ontology,
            run_config=run_config,
        )


def _as_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CombinedFactoryError("GPU ledger timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _strictly_after(*values: datetime) -> datetime:
    threshold = max(value.astimezone(UTC) for value in values)
    observed = datetime.now(UTC)
    return observed if observed > threshold else threshold + timedelta(microseconds=1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registered_repair_policy_hash(
    *,
    configuration: CombinedBlockConfiguration,
    repair_prompt_file_sha256: str,
    validator_hash: str,
) -> str:
    """Recompute the registered repair-policy binding from its exact components."""

    return canonical_sha256(
        {
            "policy": "one-preservation-repair-with-short-global-reserve-v1",
            "repair_prompt_file_sha256": repair_prompt_file_sha256,
            "maximum_repairs_per_call": configuration.maximum_repairs_per_call,
            "repair_reserve_class": configuration.repair_reserve_class,
            "repair_watchdog_seconds": configuration.repair_watchdog_seconds,
            "validator_hash": validator_hash,
        }
    )


def _selected_freeze(path: Path) -> Mapping[str, object]:
    resolved = _safe_file(path, "selected-model freeze")
    try:
        outer = json.loads(resolved.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CombinedFactoryError("selected-model freeze is not valid JSON") from error
    if not isinstance(outer, Mapping):
        raise CombinedFactoryError("selected-model freeze root must be an object")
    nested = outer.get("selected_model_freeze")
    value = nested if isinstance(nested, Mapping) else outer
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != canonical_sha256(immutable):
        raise CombinedFactoryError("selected-model freeze self-hash changed")
    expected = {
        "kind": "selected_llm_model_freeze",
        "model_candidate": "fallback",
        "repository": FALLBACK_MODEL_REPOSITORY,
        "revision": FALLBACK_MODEL_REVISION,
        "served_model_name": FALLBACK_SERVED_MODEL_NAME,
        "mixed_model_candidates_forbidden": True,
        "held_out_requires_separate_development_and_review_gates": True,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise CombinedFactoryError("selected-model freeze is not the pinned fallback stack")
    if value.get("applies_symmetrically_to") != [
        "C1",
        "C2",
        "A-FixedSelect",
        "LLM_ablations",
    ]:
        raise CombinedFactoryError("selected-model freeze is not symmetric")
    return value


def _event(artifacts: ArtifactStore, event_id: str) -> Any:
    matches = tuple(
        item for item in artifacts.ledger.gpu_events() if item.event_id == event_id
    )
    if len(matches) != 1:
        raise CombinedRecoveryRequired(
            f"GPU intent {event_id!r} lacks one terminal allocation row"
        )
    return matches[0]


def _condition_outcome(error: BaseException) -> RunOutcome:
    if isinstance(error, TimeoutError):
        return RunOutcome.TIMED_OUT
    if isinstance(error, (ValidationError, ValueError)):
        return RunOutcome.INVALID
    return RunOutcome.FAILED


def _feedback_status(outcome: RunOutcome) -> FeedbackAttemptStatus:
    return {
        RunOutcome.SUCCEEDED: FeedbackAttemptStatus.SUCCEEDED,
        RunOutcome.INVALID: FeedbackAttemptStatus.INVALID,
        RunOutcome.TIMED_OUT: FeedbackAttemptStatus.TIMED_OUT,
        RunOutcome.FAILED: FeedbackAttemptStatus.FAILED,
        RunOutcome.INTERRUPTED: FeedbackAttemptStatus.FAILED,
    }[outcome]


class _CombinedSemanticExecutor:
    """Frozen active-construction engine over the held-out semantic machinery."""

    def __init__(
        self,
        *,
        repository: Path,
        run_id: str,
        manifest: CombinedCallManifest,
        runtime: CombinedRuntimeBinding,
        phase5_inputs: Phase5ExecutionInputManifest,
        provider: _ProductionInputProvider,
        construction: DevelopmentConstructionConfiguration,
        tokenizer: PackingTokenizer,
        tokenizer_manifest: TokenizerManifest,
        artifacts: ArtifactStore,
        runtime_root: Path,
    ) -> None:
        self.repository = repository
        self.run_id = run_id
        self.manifest = manifest
        self.runtime = runtime
        self.phase5_inputs = phase5_inputs
        self.provider = provider
        self.construction = construction
        self.tokenizer = tokenizer
        self.tokenizer_manifest = tokenizer_manifest
        self.artifacts = artifacts
        self.state_root = runtime_root / "combined-semantic"
        self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.state_root, 0o700)
        faux_manifest = SimpleNamespace(
            manifest_id=run_id,
            content_hash=manifest.content_hash,
            units=tuple(SimpleNamespace(unit_id=value) for value in provider.neutral_by_unit),
        )
        self.helper = FrozenHeldOutSemanticExecutor(
            root=repository,
            call_manifest=cast(Any, faux_manifest),
            construction=construction,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            model_stack_hash=runtime.model_manifest_hash,
            seed_manifest_hash=runtime.seed_manifest_hash,
            validator_hash=runtime.validator_hash,
            selected_model_freeze_hash=runtime.selected_model_freeze_hash,
            source_tree_hash=runtime.source_tree_association_hash,
            artifacts=artifacts,
            neutral_by_unit=provider.neutral_by_unit,
            state_path=self.state_root / "shared-validator-state.json",
        )
        self.calls_by_hash = {item.content_hash: item for item in manifest.calls}
        phase5_values = (*phase5_inputs.scripted_inputs, *phase5_inputs.researcher_trace_inputs)
        self.phase5_by_episode = {item.episode_id: item for item in phase5_values}
        if len(self.calls_by_hash) != 49 or len(self.phase5_by_episode) != 9:
            raise CombinedFactoryError("combined semantic inventory changed")

    def _intent_path(self, request_hash: str, kind: str) -> Path:
        return self.state_root / f"{kind}-{request_hash}.intent.json"

    def _result_path(self, request_hash: str, kind: str) -> Path:
        return self.state_root / f"{kind}-{request_hash}.result.json"

    def _completion_path(self, request_hash: str, kind: str = "ordinary") -> Path:
        if kind not in {"ordinary", "phase5"}:
            raise CombinedFactoryError("unknown combined semantic completion kind")
        return self.state_root / f"{kind}-{request_hash}.completion.json"

    def _repair_disposition_path(self, request_hash: str) -> Path:
        return self.state_root / f"repair-{request_hash}.disposition.json"

    def _append(self, path: Path, value: ImmutableRecord, label: str) -> None:
        _append_bytes_atomically(
            path,
            (value.to_canonical_json() + "\n").encode("utf-8"),
            label=label,
        )

    def _read(self, path: Path, model: type[ImmutableRecord]) -> Any:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise CombinedFactoryError(f"unsafe combined semantic state: {path.name}")
        try:
            return model.model_validate_json(path.read_bytes())
        except Exception as error:
            raise CombinedFactoryError(
                f"invalid combined semantic state {path.name}: {error}"
            ) from error

    def _fake_call(self, call: CombinedCallSpec, *, repair: bool) -> Any:
        return SimpleNamespace(
            call_id=call.call_id,
            call_class=call.call_class.value,
            condition=call.condition,
            unit_id=call.source.unit_id,
            ordinal=call.ordinal,
            seed_block=call.seed_block,
            vllm_seed=call.vllm_seed,
            watchdog_seconds=(
                call.repair_watchdog_seconds if repair else call.watchdog_seconds
            ),
            repair_reserve_class="reserve_short" if repair else "reserve_standard",
        )

    def _envelope(self, inputs: HeldOutProduceInputs) -> Any:
        return SimpleNamespace(
            prequery_stage=SimpleNamespace(
                staging_manifest_hash=inputs.prequery_barrier.execution_manifest_hash
            ),
            query_stage=SimpleNamespace(
                staging_manifest_hash=inputs.query_access.stage_manifest_hash
            ),
            query_opening=SimpleNamespace(query_access_event=inputs.query_access),
            prequery_barrier=inputs.prequery_barrier,
            source_c1_output=None,
        )

    def _validate_generation(
        self,
        *,
        call: CombinedCallSpec,
        attempt: Any,
        config: RunConditionConfig,
        inputs: HeldOutProduceInputs,
        repair_attempt: int,
        repair_parent_raw_hash: str | None,
        base_invalid: Mapping[str, object] | None = None,
        diagnosed_paths: tuple[str, ...] = (),
        validated_at_override: datetime | None = None,
    ) -> ValidatedGeneration:
        if call.condition is not ConditionName.A_NO_TEMPORAL_EPISTEMIC:
            return self.helper._validate_generation(
                call=self._fake_call(call, repair=bool(repair_attempt)),
                envelope=self._envelope(inputs),
                attempt=attempt,
                config=config,
                repair_attempt=repair_attempt,
                repair_parent_raw_hash=repair_parent_raw_hash,
                base_invalid=base_invalid,
                diagnosed_paths=diagnosed_paths,
                validated_at_override=validated_at_override,
            )
        generated = attempt.generated
        raw = attempt.raw_reference
        if generated is None or raw is None:
            raise attempt.failure or CombinedFactoryError("model returned no generation")
        guided = attempt.guided
        if (
            generated.request_id != guided.request_id
            or generated.request_hash != guided.request_hash
            or generated.response_sha256 != raw.logical_content_hash
            or generated.finish_reason != "stop"
            or generated.prompt_tokens != guided.rendered_input_token_count
            or generated.completion_tokens > guided.decoding.maximum_output_tokens
        ):
            raise CombinedFactoryError("vLLM response envelope changed")
        alias = encode_development_semantic_request(attempt.semantic).alias_manifest
        restored = restore_model_output_source_aliases(generated.parsed_object, alias)
        ablated = NoTemporalEpistemicOntologyDraft.model_validate(restored)
        if (
            ablated.budget_accounting.input_tokens != 0
            or ablated.budget_accounting.output_tokens != 0
        ):
            raise CombinedFactoryError("model token sentinels must remain zero")
        raw_draft = normalize_no_temporal_epistemic_draft(ablated)
        event = _event(self.artifacts, attempt.event_id)
        started_at = _as_utc(event.started_at)
        completed_at = _as_utc(event.ended_at)
        normalized = normalize_generation_metadata(
            raw_draft,
            decision_recorded_at=completed_at,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
        )
        boundary = validate_draft_structure(
            draft=normalized,
            upper_ontology=attempt.semantic.upper_ontology,
            evidence=attempt.semantic.packet.evidence,
            horizon=inputs.snapshot.horizon,
            budgets=attempt.semantic.budgets,
            capabilities=attempt.semantic.capabilities,
        )
        boundary.raise_for_errors()
        report_hashes = [boundary.content_hash]
        if repair_attempt:
            if base_invalid is None:
                raise CombinedFactoryError("repair lacks its preserved invalid parent")
            preservation = validate_repair_preservation(
                base_draft=base_invalid,
                repaired_draft=cast(Mapping[str, object], restored),
                diagnosed_paths=diagnosed_paths,
            )
            preservation.raise_for_errors()
            report_hashes.append(preservation.content_hash)
        validated_at = (
            _strictly_after(completed_at)
            if validated_at_override is None
            else validated_at_override.astimezone(UTC)
        )
        if validated_at <= completed_at:
            raise CombinedFactoryError(
                "validation timestamp must follow generation completion"
            )
        validation = runtime_structural_acceptance_record(
            validation_id=(
                "combined-validation-"
                + canonical_sha256((call.call_id, raw.artifact_hash))[:20]
            ),
            target_id=normalized.content_hash,
            diagnostics=(
                "registered ablation normalization accepted",
            ),
            repair_parent_hash=repair_parent_raw_hash,
            repair_attempt=repair_attempt,
            validated_at=validated_at,
        )
        return ValidatedGeneration(
            generation_id=(
                "combined-generation-"
                + canonical_sha256((call.call_id, raw.artifact_hash))[:20]
            ),
            condition=call.condition,
            request_hash=attempt.semantic.content_hash,
            raw_output_artifact_hash=raw.artifact_hash,
            raw_parsed_draft=raw_draft,
            draft=normalized,
            normalized_draft_hash=normalized.content_hash,
            stage_manifest_hash=inputs.query_access.stage_manifest_hash,
            query_access_event_hash=inputs.query_access.content_hash,
            prequery_barrier_hash=inputs.prequery_barrier.content_hash,
            packing_report_hash=guided.packing.content_hash,
            capability_manifest_hash=CapabilityManifest.for_condition(
                call.condition
            ).content_hash,
            seed_manifest_hash=self.runtime.seed_manifest_hash,
            prompt_hash=cast(str, config.prompt_hash),
            output_schema_hash=cast(str, config.output_schema_hash),
            decoding_manifest_hash=cast(str, config.decoding_manifest_hash),
            validator_hash=self.runtime.validator_hash,
            model_stack_hash=self.runtime.model_manifest_hash,
            seed=call.vllm_seed,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
            generation_started_at=started_at,
            generation_completed_at=completed_at,
            decision_recorded_at=completed_at,
            validation_records=(validation,),
            validator_report_hashes=tuple(report_hashes),
            validated_at=validated_at,
            repair_attempt=repair_attempt,
            repair_parent_raw_output_hash=repair_parent_raw_hash,
        )

    def _persist_condition_attempt(
        self,
        attempt: ConditionAttemptRecord,
        completed_at: datetime,
    ) -> RestrictedArtifactPointer:
        artifact = self.artifacts.put_bytes(
            (attempt.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.condition-attempt+json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=completed_at,
        )
        return RestrictedArtifactPointer(
            artifact_hash=artifact.content_hash,
            logical_content_hash=attempt.content_hash,
            object_kind="condition_attempt",
            media_type=artifact.media_type,
        )

    def _persist_phase5_attempt(
        self,
        *,
        service: VLLMService,
        call: CombinedCallSpec,
        semantic: Any,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
        repair: bool,
        parent_attempt_id: str | None,
        public_request_hash: str,
        remaining_required_seconds: float,
        prequery_sealed_at: datetime,
        query_revealed_at: datetime,
    ) -> Any:
        """Persist one honest guided request while retaining Phase 5's wrapper hash."""

        created_at = datetime.now(UTC)
        semantic_reference = self.helper.resolver.persist_record(
            semantic,
            object_kind="construction_request",
            release_class=ReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        packing_reference = self.helper.resolver.persist_record(
            guided.packing,
            object_kind="packing_report",
            release_class=ReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        job = self.artifacts.ledger.create_or_resume_job(
            {
                "execution_id": self.run_id,
                "call_id": call.call_id,
                "manifest_hash": self.manifest.content_hash,
                "condition": call.condition.value,
                "lifecycle_kind": "query_time_generation",
            },
            release_class=ReleaseClass.RESTRICTED,
            created_at=prequery_sealed_at,
        )
        self.artifacts.ledger.link_job_to_study(
            study_id=self.run_id,
            job_id=job.job_id,
            created_at=created_at,
        )
        self.artifacts.ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, prequery_sealed_at),
                (JobState.QUERY_REVEALED, query_revealed_at),
            ),
        )
        suffix = "repair" if repair else "base"
        attempt_id = f"{self.run_id}-{call.call_id}-{suffix}"
        self.artifacts.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.REPAIR if repair else AttemptKind.BASE,
            input_hash=guided.request_hash,
            config_hash=config.content_hash,
            seed=cast(int, call.frozen_seed),
            parent_attempt_id=parent_attempt_id,
            created_at=created_at,
        )
        event_id = f"{self.run_id}-{call.call_id}-{suffix}-gpu"
        generated: GenerationResult | None = None
        failure: BaseException | None = None
        try:
            generated = service.generate(
                guided,
                event_id=event_id,
                watchdog_seconds=(
                    call.repair_watchdog_seconds if repair else call.watchdog_seconds
                ),
                repair=repair,
                job_id=job.job_id,
                attempt_id=attempt_id,
                remaining_required_seconds=remaining_required_seconds,
                accounting_details={
                    "phase": "combined_phase5",
                    "ordinal": call.ordinal,
                    "call_class": call.call_class.value,
                    "attempt": suffix,
                    "semantic_request_hash": semantic.content_hash,
                    "phase5_request_hash": public_request_hash,
                },
            )
        except BaseException as error:
            failure = error
        event = _event(self.artifacts, event_id)
        raw_reference = None
        if generated is not None:
            raw_record = self.artifacts.put_bytes(
                generated.raw_response,
                media_type="application/vnd.story-projection.raw-model-response+json",
                release_class=ReleaseClass.RESTRICTED,
                created_at=_as_utc(event.ended_at),
            )
            raw_reference = self.helper.resolver.reference(
                raw_record,
                logical_content_hash=hashlib.sha256(generated.raw_response).hexdigest(),
                object_kind="raw_model_response",
            )
        model_call_id = f"{self.run_id}-{call.call_id}-{suffix}-model"
        self.artifacts.ledger.record_model_call(
            model_call_id=model_call_id,
            job_id=job.job_id,
            attempt_id=attempt_id,
            gpu_event_id=event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.REPAIR if repair else ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.SHORT if repair else RetryClass.STANDARD,
            model_manifest_hash=self.runtime.model_manifest_hash,
            decoding_manifest_hash=guided.decoding.content_hash,
            request_hash=public_request_hash,
            response_artifact_hash=(
                None if raw_reference is None else raw_reference.artifact_hash
            ),
            construction_unit_hash=canonical_sha256(
                {
                    "manifest": self.manifest.content_hash,
                    "ordinal": call.ordinal,
                    "condition": call.condition,
                    "guided_request_hash": guided.request_hash,
                }
            ),
            served_context_count=1,
            prompt_tokens=0 if generated is None else generated.prompt_tokens,
            completion_tokens=0 if generated is None else generated.completion_tokens,
            allocated_gpu_seconds=event.allocated_seconds,
            successful=generated is not None and failure is None,
            created_at=datetime.now(UTC),
        )
        self.helper._advance_state(
            job.job_id,
            JobState.REPAIRED if repair else JobState.GENERATED,
            _as_utc(event.ended_at),
        )
        model_call = self.artifacts.ledger.get_model_call(model_call_id)
        return SimpleNamespace(
            job_id=job.job_id,
            attempt_id=attempt_id,
            semantic=semantic,
            guided=guided,
            semantic_reference=semantic_reference,
            packing_reference=packing_reference,
            raw_reference=raw_reference,
            event_id=event_id,
            event_hash=_record_hash(event, _GPU_EVENT_FIELDS),
            model_call_id=model_call_id,
            model_call_hash=_record_hash(model_call, _MODEL_CALL_FIELDS),
            generated=generated,
            allocated_seconds=event.allocated_seconds,
            failure=failure,
        )

    def _run_generation(
        self,
        *,
        service: VLLMService,
        call: CombinedCallSpec,
        semantic: Any,
        config: RunConditionConfig,
        inputs: HeldOutProduceInputs,
        repair_authority: RepairAuthority,
        authority_request_hash: str,
        remaining_required_seconds: float,
        phase5_request_hash: str | None = None,
    ) -> tuple[ConditionAttemptRecord, tuple[Any, ...]]:
        if (
            not math.isfinite(remaining_required_seconds)
            or remaining_required_seconds <= 0
        ):
            raise CombinedFactoryError(
                "combined generation requires a nonzero remaining forecast receipt"
            )
        guided = build_development_guided_request(
            root=self.repository,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        if (
            guided.decoding.content_hash != config.decoding_manifest_hash
            or canonical_sha256(guided.output_schema) != config.output_schema_hash
            or guided.request_hash == semantic.content_hash
        ):
            raise CombinedFactoryError("combined guided request differs from its freeze")
        fake_base = self._fake_call(call, repair=False)
        if phase5_request_hash is None:
            base = self.helper._persist_attempt(
                service=service,
                call=fake_base,
                semantic=semantic,
                guided=guided,
                config=config,
                repair=False,
                remaining_required_seconds=remaining_required_seconds,
                parent_attempt_id=None,
                prequery_sealed_at=inputs.prequery_barrier.sealed_at,
                query_revealed_at=inputs.query_access.accessed_at,
            )
        else:
            base = self._persist_phase5_attempt(
                service=service,
                call=call,
                semantic=semantic,
                guided=guided,
                config=config,
                repair=False,
                parent_attempt_id=None,
                public_request_hash=phase5_request_hash,
                remaining_required_seconds=remaining_required_seconds,
                prequery_sealed_at=inputs.prequery_barrier.sealed_at,
                query_revealed_at=inputs.query_access.accessed_at,
            )
        generation = None
        final_attempt = base
        final_inputs = inputs
        failure: BaseException | None = base.failure
        base_validation_id: str | None = None
        invalid = cast(Mapping[str, object], base.generated.parsed_object) if base.generated else {}
        diagnosed_paths: tuple[str, ...] = ()
        if failure is None:
            try:
                generation = self._validate_generation(
                    call=call,
                    attempt=base,
                    config=config,
                    inputs=inputs,
                    repair_attempt=0,
                    repair_parent_raw_hash=None,
                    validated_at_override=(
                        _as_utc(_event(self.artifacts, base.event_id).ended_at)
                        + timedelta(microseconds=1)
                    ),
                )
            except BaseException as error:
                failure = error
                if isinstance(error, ValidationError):
                    diagnosed_paths = tuple(
                        "/" + "/".join(str(part) for part in row["loc"])
                        for row in error.errors(include_input=False)
                    )
                if not diagnosed_paths:
                    diagnosed_paths = ("/draft",)

        base_checked_at = (
            _as_utc(_event(self.artifacts, base.event_id).ended_at)
            + timedelta(microseconds=1)
        )
        base_validation_id = self.helper._record_attempt_validation(
            call=self._fake_call(call, repair=False),
            attempt=base,
            accepted=generation is not None and failure is None,
            checked_at=base_checked_at,
            error=failure,
            generation=generation,
            repair=False,
            terminal=failure is None,
        )
        attempts = [base]
        if failure is not None and base.generated is not None:
            repair_semantic, repair_config = self.helper._repair_request(
                call=self._fake_call(call, repair=True),
                base=semantic,
            )
            repair_guided = self.helper._repair_guided_request(
                call=self._fake_call(call, repair=True),
                semantic=repair_semantic,
                invalid=invalid,
                diagnostics=tuple(
                    {"code": "boundary_validation", "path": path}
                    for path in diagnosed_paths
                ),
            )
            repair_exists = self._attempt_trace_exists(call, "repair")
            disposition = self._repair_disposition(
                call=call,
                authority=repair_authority,
                authority_request_hash=authority_request_hash,
                base=base,
                repair_semantic=repair_semantic,
                repair_guided=repair_guided,
                repair_trace_exists=repair_exists,
            )
            if disposition.decision == "reserve_exhausted":
                failed = failed_c2_attempt(
                    inputs,
                    outcome=_condition_outcome(failure),
                    failure_code=f"combined_{type(failure).__name__.casefold()}",
                    raw_output_hash=base.raw_reference.artifact_hash,
                )
                self.helper._advance_state(
                    base.job_id, JobState.VALIDATED, disposition.decided_at
                )
                self.helper._advance_state(
                    base.job_id, JobState.FINALIZED, disposition.decided_at
                )
                return failed, tuple(attempts)
            if repair_exists:
                repair = self._restore_terminal_attempt(
                    call=call,
                    semantic=repair_semantic,
                    guided=repair_guided,
                    config=repair_config,
                    repair=True,
                    phase5_request_hash=phase5_request_hash,
                    inputs=inputs,
                )
            elif phase5_request_hash is None:
                repair = self.helper._persist_attempt(
                    service=service,
                    call=self._fake_call(call, repair=True),
                    semantic=repair_semantic,
                    guided=repair_guided,
                    config=repair_config,
                    repair=True,
                    remaining_required_seconds=remaining_required_seconds,
                    parent_attempt_id=f"{self.run_id}-{call.call_id}-base",
                    prequery_sealed_at=inputs.prequery_barrier.sealed_at,
                    query_revealed_at=inputs.query_access.accessed_at,
                )
            else:
                repair = self._persist_phase5_attempt(
                    service=service,
                    call=call,
                    semantic=repair_semantic,
                    guided=repair_guided,
                    config=repair_config,
                    repair=True,
                    parent_attempt_id=f"{self.run_id}-{call.call_id}-base",
                    public_request_hash=repair_guided.request_hash,
                    remaining_required_seconds=remaining_required_seconds,
                    prequery_sealed_at=inputs.prequery_barrier.sealed_at,
                    query_revealed_at=inputs.query_access.accessed_at,
                )
            attempts.append(repair)
            final_attempt = repair
            repaired_input_values = inputs.model_dump(
                mode="python", exclude={"content_hash"}
            )
            repaired_input_values["run_config"] = repair_config
            final_inputs = HeldOutProduceInputs.model_validate(repaired_input_values)
            failure = repair.failure
            if failure is None:
                try:
                    generation = self._validate_generation(
                        call=call,
                        attempt=repair,
                        config=repair_config,
                        inputs=inputs,
                        repair_attempt=1,
                        repair_parent_raw_hash=(
                            None
                            if base.raw_reference is None
                            else base.raw_reference.artifact_hash
                        ),
                        base_invalid=invalid,
                        diagnosed_paths=diagnosed_paths,
                        validated_at_override=(
                            _as_utc(_event(self.artifacts, repair.event_id).ended_at)
                            + timedelta(microseconds=1)
                        ),
                    )
                except BaseException as error:
                    failure = error
            repair_checked_at = (
                _as_utc(_event(self.artifacts, repair.event_id).ended_at)
                + timedelta(microseconds=1)
            )
            if failure is not None:
                self.helper._record_attempt_validation(
                    call=self._fake_call(call, repair=True),
                    attempt=repair,
                    accepted=False,
                    checked_at=repair_checked_at,
                    error=failure,
                    parent_validation_id=base_validation_id,
                    repair=True,
                    terminal=True,
                )
            elif generation is not None:
                base_validation_id = self.helper._record_attempt_validation(
                    call=self._fake_call(call, repair=True),
                    attempt=repair,
                    accepted=True,
                    checked_at=generation.validated_at,
                    generation=generation,
                    parent_validation_id=base_validation_id,
                    repair=True,
                    terminal=True,
                )
        result_object = None
        if generation is not None and failure is None:
            try:
                result_object = finalize_c2_draft(
                    final_inputs,
                    request=cast(Any, final_attempt.semantic),
                    generation=generation,
                )
            except BaseException as error:
                failure = error
        terminal_at = _strictly_after(
            _as_utc(_event(self.artifacts, final_attempt.event_id).ended_at)
        )
        if result_object is not None and failure is None:
            if result_object.projection is None or base_validation_id is None:
                raise CombinedFactoryError(
                    "successful combined output lacks canonical validation lineage"
                )
            self.helper._record_projection(
                call=self._fake_call(call, repair=final_attempt is not base),
                attempt=final_attempt,
                inputs=final_inputs,
                projection=result_object.projection,
                validation_id=base_validation_id,
                finalized_at=terminal_at,
            )
            self.helper._advance_state(
                final_attempt.job_id, JobState.FINALIZED, terminal_at
            )
            return result_object, tuple(attempts)
        terminal_error = failure or CombinedFactoryError("missing terminal output")
        raw_hash = (
            None
            if final_attempt.raw_reference is None
            else final_attempt.raw_reference.artifact_hash
        )
        failed = failed_c2_attempt(
            inputs,
            outcome=_condition_outcome(terminal_error),
            failure_code=f"combined_{type(terminal_error).__name__.casefold()}",
            raw_output_hash=raw_hash,
        )
        self.helper._advance_state(
            final_attempt.job_id, JobState.VALIDATED, terminal_at
        )
        if not any(
            item.attempt_id == final_attempt.attempt_id
            for item in self.artifacts.ledger.failures_for_lineage(final_attempt.attempt_id)
        ):
            self.artifacts.ledger.record_failure(
                attempt_id=final_attempt.attempt_id,
                failure_kind=FailureKind.VALIDATION,
                message="Combined output finalization failed; inspect restricted lineage",
                details={
                    "call_id": call.call_id,
                    "exception_type": type(terminal_error).__name__,
                },
                artifact_hash=raw_hash,
                occurred_at=terminal_at,
            )
        self.helper._advance_state(
            final_attempt.job_id, JobState.FINALIZED, terminal_at
        )
        return failed, tuple(attempts)

    def _model_call_or_none(self, model_call_id: str) -> Any | None:
        try:
            return self.artifacts.ledger.get_model_call(model_call_id)
        except KeyError:
            return None

    def _attempt_trace_exists(self, call: CombinedCallSpec, suffix: str) -> bool:
        attempt_id = f"{self.run_id}-{call.call_id}-{suffix}"
        event_id = f"{attempt_id}-gpu"
        model_call_id = f"{attempt_id}-model"
        if self._model_call_or_none(model_call_id) is not None:
            return True
        if any(item.event_id == event_id for item in self.artifacts.ledger.gpu_events()):
            return True
        try:
            self.artifacts.ledger.attempt_lineage(attempt_id)
        except KeyError:
            return False
        return True

    def _repair_disposition(
        self,
        *,
        call: CombinedCallSpec,
        authority: RepairAuthority,
        authority_request_hash: str,
        base: Any,
        repair_semantic: Any,
        repair_guided: GuidedJSONRequest,
        repair_trace_exists: bool,
    ) -> _RepairDisposition:
        """Recover or durably make the sole repair/reserve decision.

        A pre-existing repair trace may only be adopted when the immutable
        outer reserve claim already exists.  With no trace, the reserve is
        claimed first and the local disposition is appended before any repair
        attempt can be persisted.
        """

        if base.raw_reference is None:
            raise CombinedFactoryError("repair-eligible base lacks its raw output")
        common = {
            "run_id": self.run_id,
            "call_id": call.call_id,
            "call_spec_hash": call.content_hash,
            "request_hash": authority_request_hash,
            "base_model_call_id": base.model_call_id,
            "base_raw_output_hash": base.raw_reference.artifact_hash,
            "repair_semantic_request_hash": repair_semantic.content_hash,
            "repair_guided_request_hash": repair_guided.request_hash,
        }
        path = self._repair_disposition_path(authority_request_hash)
        if path.exists():
            disposition = self._read(path, _RepairDisposition)
            if any(getattr(disposition, name) != value for name, value in common.items()):
                raise CombinedFactoryError("repair disposition changed on recovery")
            if disposition.decision == "repair_claimed":
                recovered = authority.recover_claim(
                    request_hash=authority_request_hash,
                    base_model_call_id=base.model_call_id,
                )
                if recovered is None or recovered != disposition.repair_claim:
                    raise CombinedFactoryError(
                        "repair disposition lost its exact global reserve claim"
                    )
            elif repair_trace_exists:
                raise CombinedFactoryError(
                    "repair trace exists after a durable exhausted-reserve decision"
                )
            return disposition

        event_end = _as_utc(_event(self.artifacts, base.event_id).ended_at)
        if repair_trace_exists:
            claim = authority.recover_claim(
                request_hash=authority_request_hash,
                base_model_call_id=base.model_call_id,
            )
            if claim is None:
                raise CombinedRecoveryRequired(
                    "repair trace exists without its preceding global reserve claim"
                )
            disposition = _RepairDisposition(
                **common,
                decision="repair_claimed",
                repair_claim=claim,
                decided_at=_strictly_after(event_end, claim.claimed_at),
            )
        else:
            claimed_at = _strictly_after(event_end)
            try:
                claim = authority.claim(
                    request_hash=authority_request_hash,
                    base_model_call_id=base.model_call_id,
                    claimed_at=claimed_at,
                )
            except CombinedProductionError as error:
                if str(error) != "registered global short-repair reserve is exhausted":
                    raise
                disposition = _RepairDisposition(
                    **common,
                    decision="reserve_exhausted",
                    decided_at=claimed_at,
                )
            else:
                disposition = _RepairDisposition(
                    **common,
                    decision="repair_claimed",
                    repair_claim=claim,
                    decided_at=_strictly_after(claimed_at, claim.claimed_at),
                )
        self._append(path, disposition, "combined repair disposition")
        return disposition

    def _restore_terminal_attempt(
        self,
        *,
        call: CombinedCallSpec,
        semantic: Any,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
        repair: bool,
        phase5_request_hash: str | None,
        inputs: HeldOutProduceInputs,
    ) -> Any:
        """Rebuild one attempt only from a complete immutable GPU/model/CAS lineage."""

        suffix = "repair" if repair else "base"
        attempt_id = f"{self.run_id}-{call.call_id}-{suffix}"
        event_id = f"{attempt_id}-gpu"
        model_call_id = f"{attempt_id}-model"
        model_call = self._model_call_or_none(model_call_id)
        if model_call is None:
            raise CombinedRecoveryRequired(
                f"terminal combined {suffix} attempt lacks its model-call row"
            )
        event = _event(self.artifacts, event_id)
        try:
            lineage = self.artifacts.ledger.attempt_lineage(attempt_id)
        except KeyError as error:
            raise CombinedRecoveryRequired(
                f"terminal combined {suffix} attempt lacks its attempt row"
            ) from error
        expected_parent = (
            f"{self.run_id}-{call.call_id}-base" if repair else None
        )
        expected_role = (
            ModelCallRole.REPAIR
            if repair
            else (
                ModelCallRole.FIXED_SELECT
                if call.condition is ConditionName.A_FIXED_SELECT
                else (
                    ModelCallRole.PREBUILD
                    if call.condition is ConditionName.C1_LLM_PRE
                    else ModelCallRole.QUERY_TIME
                )
            )
        )
        expected_request_hash = (
            phase5_request_hash
            if phase5_request_hash is not None and not repair
            else guided.request_hash
        )
        expected_construction_hash = canonical_sha256(
            {
                "manifest": self.manifest.content_hash,
                "ordinal": call.ordinal,
                "condition": call.condition,
                **(
                    {"guided_request_hash": guided.request_hash}
                    if phase5_request_hash is not None
                    else {}
                ),
            }
        )
        attempt = lineage[-1]
        allowed_event_kinds = {
            GpuEventKind.REPAIR if repair else GpuEventKind.INFERENCE,
            GpuEventKind.FAILURE,
            GpuEventKind.TIMEOUT,
        }
        if (
            attempt.attempt_id != attempt_id
            or attempt.attempt_kind
            is not (AttemptKind.REPAIR if repair else AttemptKind.BASE)
            or attempt.parent_attempt_id != expected_parent
            or attempt.input_hash != guided.request_hash
            or attempt.config_hash != config.content_hash
            or attempt.seed != call.vllm_seed
            or model_call.job_id != attempt.job_id
            or model_call.attempt_id != attempt_id
            or model_call.gpu_event_id != event_id
            or model_call.backend is not ModelBackend.VLLM_GPU
            or model_call.call_role is not expected_role
            or model_call.retry_class
            is not (RetryClass.SHORT if repair else RetryClass.STANDARD)
            or model_call.model_manifest_hash != self.runtime.model_manifest_hash
            or model_call.decoding_manifest_hash != guided.decoding.content_hash
            or model_call.request_hash != expected_request_hash
            or model_call.construction_unit_hash != expected_construction_hash
            or model_call.served_context_count != 1
            or model_call.allocated_gpu_microseconds != event.allocated_microseconds
            or event.event_id != event_id
            or event.job_id != attempt.job_id
            or event.attempt_id != attempt_id
            or event.event_kind not in allowed_event_kinds
            or event.succeeded is None
        ):
            raise CombinedFactoryError(
                f"terminal combined {suffix} GPU/model/attempt lineage changed"
            )
        self.artifacts.ledger.advance_job_lifecycle(
            attempt.job_id,
            (
                (JobState.PREQUERY_SEALED, inputs.prequery_barrier.sealed_at),
                (JobState.QUERY_REVEALED, inputs.query_access.accessed_at),
            ),
        )
        if repair:
            base_event = _event(
                self.artifacts, f"{self.run_id}-{call.call_id}-base-gpu"
            )
            self.helper._advance_state(
                attempt.job_id,
                JobState.GENERATED,
                _as_utc(base_event.ended_at),
            )
        self.helper._advance_state(
            attempt.job_id,
            JobState.REPAIRED if repair else JobState.GENERATED,
            _as_utc(event.ended_at),
        )

        generated: GenerationResult | None = None
        raw_reference = None
        failure: BaseException | None = None
        if model_call.successful:
            if event.succeeded is not True or model_call.response_artifact_hash is None:
                raise CombinedFactoryError(
                    f"successful combined {suffix} call lacks a successful raw response"
                )
            try:
                raw_record = self.artifacts.ledger.get_artifact(
                    model_call.response_artifact_hash
                )
                if raw_record.release_class is not ReleaseClass.RESTRICTED:
                    raise CombinedFactoryError("combined raw model response is not restricted")
                raw = self.artifacts.blobs.read_bytes(
                    raw_record,
                    allow_restricted=True,
                )
                generated = VLLMGuidedJSONClient._decode_generation_response(
                    guided,
                    200,
                    raw,
                    {},
                )
            except CombinedFactoryError:
                raise
            except Exception as error:
                raise CombinedFactoryError(
                    f"cannot reconstruct combined {suffix} raw response"
                ) from error
            if (
                generated.response_sha256 != model_call.response_artifact_hash
                or generated.prompt_tokens != model_call.prompt_tokens
                or generated.completion_tokens != model_call.completion_tokens
            ):
                raise CombinedFactoryError(
                    f"combined {suffix} raw response differs from its model-call row"
                )
            raw_reference = SimpleNamespace(
                artifact_hash=raw_record.content_hash,
                logical_content_hash=generated.response_sha256,
            )
        else:
            if (
                event.succeeded is not False
                or model_call.response_artifact_hash is not None
                or model_call.prompt_tokens != 0
                or model_call.completion_tokens != 0
            ):
                raise CombinedFactoryError(
                    f"failed combined {suffix} call retains inconsistent output state"
                )
            failure = (
                TimeoutError("recovered terminal vLLM watchdog timeout")
                if event.event_kind is GpuEventKind.TIMEOUT
                else RuntimeError("recovered terminal vLLM failure")
            )
        return SimpleNamespace(
            job_id=attempt.job_id,
            attempt_id=attempt_id,
            semantic=semantic,
            guided=guided,
            raw_reference=raw_reference,
            event_id=event_id,
            model_call_id=model_call_id,
            generated=generated,
            allocated_seconds=event.allocated_seconds,
            failure=failure,
        )

    def _recover_terminal_condition_attempt(
        self,
        *,
        service: VLLMService | None,
        call: CombinedCallSpec,
        semantic: Any,
        inputs: HeldOutProduceInputs,
        repair_authority: RepairAuthority | None,
        authority_request_hash: str,
        remaining_required_seconds: float,
        phase5_request_hash: str | None,
    ) -> tuple[ConditionAttemptRecord, tuple[Any, ...]]:
        """Finish terminal lineage, issuing only a durably authorized missing repair."""

        config = inputs.run_config
        guided = build_development_guided_request(
            root=self.repository,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        base = self._restore_terminal_attempt(
            call=call,
            semantic=semantic,
            guided=guided,
            config=config,
            repair=False,
            phase5_request_hash=phase5_request_hash,
            inputs=inputs,
        )
        attempts = [base]
        generation = None
        failure: BaseException | None = base.failure
        diagnosed_paths: tuple[str, ...] = ()
        invalid = cast(Mapping[str, object], base.generated.parsed_object) if base.generated else {}
        if failure is None:
            try:
                generation = self._validate_generation(
                    call=call,
                    attempt=base,
                    config=config,
                    inputs=inputs,
                    repair_attempt=0,
                    repair_parent_raw_hash=None,
                    validated_at_override=(
                        _as_utc(_event(self.artifacts, base.event_id).ended_at)
                        + timedelta(microseconds=1)
                    ),
                )
            except BaseException as error:
                failure = error
                if isinstance(error, ValidationError):
                    diagnosed_paths = tuple(
                        "/" + "/".join(str(part) for part in row["loc"])
                        for row in error.errors(include_input=False)
                    )
                if not diagnosed_paths:
                    diagnosed_paths = ("/draft",)

        base_checked_at = (
            _as_utc(_event(self.artifacts, base.event_id).ended_at)
            + timedelta(microseconds=1)
        )
        base_validation_id = self.helper._record_attempt_validation(
            call=self._fake_call(call, repair=False),
            attempt=base,
            accepted=generation is not None and failure is None,
            checked_at=base_checked_at,
            error=failure,
            generation=generation,
            repair=False,
            terminal=failure is None,
        )

        final_attempt = base
        final_inputs = inputs
        repair_exists = self._attempt_trace_exists(call, "repair")
        if failure is None and repair_exists:
            raise CombinedFactoryError("valid base call unexpectedly has repair lineage")
        if failure is not None and base.generated is not None:
            if service is None or repair_authority is None:
                raise CombinedRecoveryRequired(
                    "invalid base recovery requires its live service and reserve authority"
                )
            repair_semantic, repair_config = self.helper._repair_request(
                call=self._fake_call(call, repair=True),
                base=semantic,
            )
            repair_guided = self.helper._repair_guided_request(
                call=self._fake_call(call, repair=True),
                semantic=repair_semantic,
                invalid=invalid,
                diagnostics=tuple(
                    {"code": "boundary_validation", "path": path}
                    for path in diagnosed_paths
                ),
            )
            disposition = self._repair_disposition(
                call=call,
                authority=repair_authority,
                authority_request_hash=authority_request_hash,
                base=base,
                repair_semantic=repair_semantic,
                repair_guided=repair_guided,
                repair_trace_exists=repair_exists,
            )
            if disposition.decision == "repair_claimed":
                if repair_exists:
                    repair = self._restore_terminal_attempt(
                        call=call,
                        semantic=repair_semantic,
                        guided=repair_guided,
                        config=repair_config,
                        repair=True,
                        phase5_request_hash=phase5_request_hash,
                        inputs=inputs,
                    )
                else:
                    if (
                        not math.isfinite(remaining_required_seconds)
                        or remaining_required_seconds <= 0
                    ):
                        raise CombinedFactoryError(
                            "recovered repair requires a nonzero remaining forecast receipt"
                        )
                    if phase5_request_hash is None:
                        repair = self.helper._persist_attempt(
                            service=service,
                            call=self._fake_call(call, repair=True),
                            semantic=repair_semantic,
                            guided=repair_guided,
                            config=repair_config,
                            repair=True,
                            remaining_required_seconds=remaining_required_seconds,
                            parent_attempt_id=f"{self.run_id}-{call.call_id}-base",
                            prequery_sealed_at=inputs.prequery_barrier.sealed_at,
                            query_revealed_at=inputs.query_access.accessed_at,
                        )
                    else:
                        repair = self._persist_phase5_attempt(
                            service=service,
                            call=call,
                            semantic=repair_semantic,
                            guided=repair_guided,
                            config=repair_config,
                            repair=True,
                            parent_attempt_id=f"{self.run_id}-{call.call_id}-base",
                            public_request_hash=repair_guided.request_hash,
                            remaining_required_seconds=remaining_required_seconds,
                            prequery_sealed_at=inputs.prequery_barrier.sealed_at,
                            query_revealed_at=inputs.query_access.accessed_at,
                        )
                attempts.append(repair)
                final_attempt = repair
                values = inputs.model_dump(mode="python", exclude={"content_hash"})
                values["run_config"] = repair_config
                final_inputs = HeldOutProduceInputs.model_validate(values)
                failure = repair.failure
                generation = None
                if failure is None:
                    try:
                        generation = self._validate_generation(
                            call=call,
                            attempt=repair,
                            config=repair_config,
                            inputs=inputs,
                            repair_attempt=1,
                            repair_parent_raw_hash=base.raw_reference.artifact_hash,
                            base_invalid=invalid,
                            diagnosed_paths=diagnosed_paths,
                            validated_at_override=(
                                _as_utc(_event(self.artifacts, repair.event_id).ended_at)
                                + timedelta(microseconds=1)
                            ),
                        )
                    except BaseException as error:
                        failure = error
                repair_checked_at = (
                    _as_utc(_event(self.artifacts, repair.event_id).ended_at)
                    + timedelta(microseconds=1)
                )
                base_validation_id = self.helper._record_attempt_validation(
                    call=self._fake_call(call, repair=True),
                    attempt=repair,
                    accepted=generation is not None and failure is None,
                    checked_at=repair_checked_at,
                    error=failure,
                    generation=generation,
                    parent_validation_id=base_validation_id,
                    repair=True,
                    terminal=True,
                )
        elif failure is not None and repair_exists:
            raise CombinedFactoryError("failed base call unexpectedly has repair lineage")

        result_object = None
        if generation is not None and failure is None:
            try:
                result_object = finalize_c2_draft(
                    final_inputs,
                    request=cast(Any, final_attempt.semantic),
                    generation=generation,
                )
            except BaseException as error:
                failure = error
        terminal_event = _event(self.artifacts, final_attempt.event_id)
        terminal_at = _as_utc(terminal_event.ended_at) + timedelta(microseconds=2)
        if result_object is not None and failure is None:
            if result_object.projection is None:
                raise CombinedFactoryError(
                    "recovered successful output lacks its ontology projection"
                )
            self.helper._record_projection(
                call=self._fake_call(call, repair=final_attempt is not base),
                attempt=final_attempt,
                inputs=final_inputs,
                projection=result_object.projection,
                validation_id=base_validation_id,
                finalized_at=terminal_at,
            )
            self.helper._advance_state(
                final_attempt.job_id, JobState.FINALIZED, terminal_at
            )
            return result_object, tuple(attempts)
        terminal_error = failure or CombinedFactoryError("missing recovered terminal output")
        self.helper._advance_state(
            final_attempt.job_id, JobState.VALIDATED, terminal_at
        )
        if not any(
            item.attempt_id == final_attempt.attempt_id
            for item in self.artifacts.ledger.failures_for_lineage(
                final_attempt.attempt_id
            )
        ):
            self.artifacts.ledger.record_failure(
                attempt_id=final_attempt.attempt_id,
                failure_kind=FailureKind.VALIDATION,
                message="Recovered combined finalization failed; inspect restricted lineage",
                details={
                    "call_id": call.call_id,
                    "exception_type": type(terminal_error).__name__,
                },
                artifact_hash=(
                    None
                    if final_attempt.raw_reference is None
                    else final_attempt.raw_reference.artifact_hash
                ),
                occurred_at=terminal_at,
            )
        self.helper._advance_state(
            final_attempt.job_id, JobState.FINALIZED, terminal_at
        )
        return (
            failed_c2_attempt(
                inputs,
                outcome=_condition_outcome(terminal_error),
                failure_code=f"combined_{type(terminal_error).__name__.casefold()}",
                raw_output_hash=(
                    None
                    if final_attempt.raw_reference is None
                    else final_attempt.raw_reference.artifact_hash
                ),
            ),
            tuple(attempts),
        )

    def _reconstruct_completion(
        self,
        *,
        service: VLLMService | None,
        intent: _SemanticIntent,
        identity_hash: str,
        repair_authority: RepairAuthority | None,
        remaining_required_seconds: float,
    ) -> tuple[_OrdinarySemanticCompletion, EvidencePacket | None]:
        try:
            call = self.calls_by_hash[intent.call_spec_hash]
        except KeyError as error:
            raise CombinedFactoryError("semantic recovery references an unknown call") from error
        after_packet: EvidencePacket | None = None
        phase5_hash: str | None = None
        if intent.request_kind == "ordinary":
            prepared = intent.ordinary_prepared_call
            if prepared is None:
                raise CombinedRecoveryRequired(
                    "ordinary GPU intent is inflight or ambiguous: legacy intent lost "
                    "its prepared call"
                )
            inputs = self.provider.produce_inputs(call, prepared)
            semantic = prepared.request
        else:
            request = intent.phase5_request
            if request is None:
                raise CombinedFactoryError("Phase 5 recovery lost its request")
            inputs, after_packet = self._phase5_produce_inputs(call, request)
            runtime = RuntimeIdentifiers(
                model_id=self.runtime.served_model_name,
                model_revision=self.runtime.model_revision,
                tokenizer_hash=self.runtime.tokenizer_manifest_hash,
                runtime_version=self.runtime.runtime_version,
                prompt_hash=self.runtime.prompt_hashes[ConditionName.C2_LLM_QUERY],
                output_schema_hash=self.runtime.output_schema_hashes[
                    ConditionName.C2_LLM_QUERY
                ],
                decoding_config_hash=self.runtime.decoding_manifest_hashes[
                    ConditionName.C2_LLM_QUERY
                ],
            )
            semantic = build_c2_construction_request(
                inputs,
                runtime=runtime,
                requested_at=inputs.query_processing_started_at + timedelta(microseconds=1),
            )
            phase5_hash = request.content_hash
        guided = build_development_guided_request(
            root=self.repository,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        if (
            intent.run_id != self.run_id
            or intent.call_id != call.call_id
            or intent.guided_request_hash != guided.request_hash
            or intent.service_identity_hash != identity_hash
        ):
            raise CombinedFactoryError("semantic recovery intent changed")
        condition_attempt, attempts = self._recover_terminal_condition_attempt(
            service=service,
            call=call,
            semantic=semantic,
            inputs=inputs,
            repair_authority=repair_authority,
            authority_request_hash=intent.prepared_hash,
            remaining_required_seconds=remaining_required_seconds,
            phase5_request_hash=phase5_hash,
        )
        first_event = _event(self.artifacts, attempts[0].event_id)
        final_event = _event(self.artifacts, attempts[-1].event_id)
        completion = _OrdinarySemanticCompletion(
            run_id=self.run_id,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            prepared_hash=intent.prepared_hash,
            semantic_request_hash=semantic.content_hash,
            semantic_intent_hash=intent.content_hash,
            service_identity_hash=identity_hash,
            condition_attempt=condition_attempt,
            model_call_ids=tuple(item.model_call_id for item in attempts),
            model_request_hashes=tuple(item.guided.request_hash for item in attempts),
            cumulative_gpu_seconds_before=intent.cumulative_gpu_seconds_before,
            cumulative_gpu_seconds_after=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
            started_at=_as_utc(first_event.started_at),
            completed_at=_strictly_after(_as_utc(final_event.ended_at)),
        )
        return completion, after_packet

    def _ordinary_result_from_completion(
        self,
        *,
        identity: CombinedServiceIdentity,
        intent: _SemanticIntent,
        completion: _OrdinarySemanticCompletion,
    ) -> CombinedServiceCallResult:
        try:
            call = self.calls_by_hash[intent.call_spec_hash]
        except KeyError as error:
            raise CombinedFactoryError(
                "ordinary completion references an unknown call"
            ) from error
        attempt = completion.condition_attempt
        if (
            completion.run_id != self.run_id
            or completion.call_id != call.call_id
            or completion.call_spec_hash != call.content_hash
            or completion.prepared_hash != intent.prepared_hash
            or not completion.semantic_request_hash
            or completion.semantic_intent_hash != intent.content_hash
            or completion.service_identity_hash != identity.content_hash
            or intent.service_identity_hash != identity.content_hash
            or completion.cumulative_gpu_seconds_before
            != intent.cumulative_gpu_seconds_before
            or completion.cumulative_gpu_seconds_after
            < completion.cumulative_gpu_seconds_before
            or attempt.condition is not call.condition
            or attempt.seed_block != call.seed_block
            or len(completion.model_call_ids) not in {1, 2}
            or len(completion.model_request_hashes)
            != len(completion.model_call_ids)
        ):
            raise CombinedFactoryError("ordinary terminal completion identity changed")
        events = {event.event_id: event for event in self.artifacts.ledger.gpu_events()}
        previous_end: datetime | None = None
        for index, (model_call_id, request_hash) in enumerate(
            zip(
                completion.model_call_ids,
                completion.model_request_hashes,
                strict=True,
            )
        ):
            try:
                model_call = self.artifacts.ledger.get_model_call(model_call_id)
                event = events[model_call.gpu_event_id]
            except KeyError as error:
                raise CombinedFactoryError(
                    "ordinary completion lost its model/GPU ledger lineage"
                ) from error
            event_start = _as_utc(event.started_at)
            event_end = _as_utc(event.ended_at)
            if (
                model_call.request_hash != request_hash
                or (index == 0 and model_call.gpu_event_id != intent.gpu_event_id)
                or event_end < event_start
                or (previous_end is not None and event_start < previous_end)
                or event_start < completion.started_at
                or event_end > completion.completed_at
            ):
                raise CombinedFactoryError(
                    "ordinary completion model/GPU lineage changed"
                )
            previous_end = event_end
        final_call = self.artifacts.ledger.get_model_call(
            completion.model_call_ids[-1]
        )
        if final_call.response_artifact_hash != attempt.raw_output_hash:
            raise CombinedFactoryError(
                "ordinary completion output differs from its terminal model call"
            )
        pointer = self._persist_condition_attempt(attempt, completion.completed_at)
        return CombinedServiceCallResult(
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            request_hash=intent.prepared_hash,
            semantic_request_hash=completion.semantic_request_hash,
            service_identity_hash=identity.content_hash,
            condition=call.condition,
            outcome=attempt.outcome,
            model_call_ids=completion.model_call_ids,
            model_request_hashes=completion.model_request_hashes,
            condition_attempt_artifact=pointer,
            cumulative_gpu_seconds_before=completion.cumulative_gpu_seconds_before,
            cumulative_gpu_seconds_after=completion.cumulative_gpu_seconds_after,
            started_at=completion.started_at,
            completed_at=completion.completed_at,
            checked_at=completion.completed_at,
        )

    def execute_ordinary(
        self,
        *,
        service: VLLMService,
        identity: CombinedServiceIdentity,
        prepared: CombinedPreparedCall,
        model_visible_payload: dict[str, object],
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> CombinedServiceCallResult:
        try:
            call = self.calls_by_hash[prepared.call_spec_hash]
        except KeyError as error:
            raise CombinedFactoryError(
                "prepared call is absent from the frozen manifest"
            ) from error
        if call.call_class in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }:
            raise CombinedFactoryError("Phase 5 call entered the ordinary executor")
        if model_visible_payload != prepared.model_visible_payload():
            raise CombinedFactoryError("ordinary model-visible payload changed after controller")
        result_path = self._result_path(prepared.content_hash, "ordinary")
        intent_path = self._intent_path(prepared.content_hash, "ordinary")
        if result_path.exists():
            recovered = self.recover_ordinary(
                prepared.content_hash,
                identity,
                service=service,
                repair_authority=repair_authority,
                remaining_required_seconds=remaining_required_seconds,
            )
            if recovered is None:
                raise CombinedFactoryError("ordinary result recovery changed")
            return recovered
        inputs = self.provider.produce_inputs(call, prepared)
        semantic = prepared.request
        guided = build_development_guided_request(
            root=self.repository,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        before = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        intent = _SemanticIntent(
            request_kind="ordinary",
            run_id=self.run_id,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            prepared_hash=prepared.content_hash,
            guided_request_hash=guided.request_hash,
            gpu_event_id=f"{self.run_id}-{call.call_id}-base-gpu",
            service_identity_hash=identity.content_hash,
            cumulative_gpu_seconds_before=before,
            remaining_required_seconds=remaining_required_seconds,
            created_at=datetime.now(UTC),
            ordinary_prepared_call=prepared,
        )
        if intent_path.exists():
            existing = self._read(intent_path, _SemanticIntent)
            # Timestamps make a newly constructed intent unequal.  Compare
            # every immutable scientific field explicitly on replay.
            if existing != intent and (
                existing.run_id != self.run_id
                or existing.call_id != call.call_id
                or existing.call_spec_hash != call.content_hash
                or existing.prepared_hash != prepared.content_hash
                or existing.guided_request_hash != guided.request_hash
                or existing.gpu_event_id != intent.gpu_event_id
                or existing.service_identity_hash != identity.content_hash
                or existing.remaining_required_seconds != remaining_required_seconds
                or existing.request_kind != "ordinary"
                or existing.ordinary_prepared_call != prepared
            ):
                raise CombinedFactoryError("ordinary intent changed on replay")
            raise CombinedRecoveryRequired(
                "ordinary GPU intent has no terminal result; inference cannot be resent"
            )
        self._append(intent_path, intent, "ordinary semantic intent")
        attempt, attempts = self._run_generation(
            service=service,
            call=call,
            semantic=semantic,
            config=inputs.run_config,
            inputs=inputs,
            repair_authority=repair_authority,
            authority_request_hash=prepared.content_hash,
            remaining_required_seconds=remaining_required_seconds,
        )
        final_event = _event(self.artifacts, attempts[-1].event_id)
        completed_at = _strictly_after(_as_utc(final_event.ended_at))
        after = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        completion = _OrdinarySemanticCompletion(
            run_id=self.run_id,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            prepared_hash=prepared.content_hash,
            semantic_request_hash=semantic.content_hash,
            semantic_intent_hash=intent.content_hash,
            service_identity_hash=identity.content_hash,
            condition_attempt=attempt,
            model_call_ids=tuple(item.model_call_id for item in attempts),
            model_request_hashes=tuple(item.guided.request_hash for item in attempts),
            cumulative_gpu_seconds_before=before,
            cumulative_gpu_seconds_after=after,
            started_at=_as_utc(_event(self.artifacts, attempts[0].event_id).started_at),
            completed_at=completed_at,
        )
        self._append(
            self._completion_path(prepared.content_hash),
            completion,
            "ordinary semantic completion",
        )
        result = self._ordinary_result_from_completion(
            identity=identity,
            intent=intent,
            completion=completion,
        )
        self._append(result_path, result, "ordinary semantic result")
        return result

    def recover_ordinary(
        self,
        request_hash: str,
        identity: CombinedServiceIdentity,
        *,
        service: VLLMService | None = None,
        repair_authority: RepairAuthority | None = None,
        remaining_required_seconds: float = 0.0,
    ) -> CombinedServiceCallResult | None:
        intent_path = self._intent_path(request_hash, "ordinary")
        result_path = self._result_path(request_hash, "ordinary")
        completion_path = self._completion_path(request_hash)
        if not result_path.exists():
            if intent_path.exists():
                intent = self._read(intent_path, _SemanticIntent)
                if intent.run_id != self.run_id or intent.prepared_hash != request_hash:
                    raise CombinedFactoryError("ordinary recovery intent belongs to another run")
                if completion_path.exists():
                    completion = self._read(
                        completion_path,
                        _OrdinarySemanticCompletion,
                    )
                    result = self._ordinary_result_from_completion(
                        identity=identity,
                        intent=intent,
                        completion=completion,
                    )
                    self._append(result_path, result, "recovered ordinary semantic result")
                    return result
                completion, _after_packet = self._reconstruct_completion(
                    service=service,
                    intent=intent,
                    identity_hash=identity.content_hash,
                    repair_authority=repair_authority,
                    remaining_required_seconds=remaining_required_seconds,
                )
                self._append(
                    completion_path,
                    completion,
                    "reconstructed ordinary semantic completion",
                )
                result = self._ordinary_result_from_completion(
                    identity=identity,
                    intent=intent,
                    completion=completion,
                )
                self._append(result_path, result, "reconstructed ordinary semantic result")
                return result
            return None
        if not intent_path.exists() or not completion_path.exists():
            raise CombinedFactoryError(
                "ordinary result lacks its pre-GPU intent/completion"
            )
        intent = self._read(intent_path, _SemanticIntent)
        completion = self._read(completion_path, _OrdinarySemanticCompletion)
        result = self._read(result_path, CombinedServiceCallResult)
        if (
            intent.run_id != self.run_id
            or intent.prepared_hash != request_hash
            or result.request_hash != request_hash
            or result.call_id != intent.call_id
            or result.call_spec_hash != intent.call_spec_hash
            or result.model_request_hashes[0] != intent.guided_request_hash
            or completion.semantic_intent_hash != intent.content_hash
            or completion.service_identity_hash != identity.content_hash
            or result != self._ordinary_result_from_completion(
                identity=identity,
                intent=intent,
                completion=completion,
            )
        ):
            raise CombinedFactoryError("ordinary recovery changed run/request lineage")
        return result

    def _phase5_produce_inputs(
        self,
        call: CombinedCallSpec,
        request: C2RegenerationRequest,
    ) -> tuple[HeldOutProduceInputs, EvidencePacket]:
        expected = self.phase5_by_episode.get(request.episode_id)
        if expected is None:
            raise CombinedFactoryError("Phase 5 episode is absent from the frozen inputs")
        before = expected.c2_before_projection
        if (
            expected.instruction != request.instruction
            or expected.packet != request.packet
            or before.projection_id != request.before_projection_id
            or before.content_hash != request.before_projection_hash
            or request.protocol_hash != self.phase5_inputs.protocol_hash
            or call.source.context.content_hash != request.before_context_hash
            or call.source.packet_hash != request.before_packet_hash
        ):
            raise CombinedFactoryError("Phase 5 request changed its exact frozen episode")
        preparation = self.provider._preparation(call)
        barrier = self.provider._barrier(call)
        context = request.instruction.after_context
        access_time = _strictly_after(
            request.instruction.revision.created_at,
            context.revealed_at,
            barrier.sealed_at,
            preparation.completed_at,
        )
        context_payload = (context.to_canonical_json() + "\n").encode("utf-8")
        query_artifact = self.artifacts.put_bytes(
            context_payload,
            media_type="application/vnd.story-projection.query-context+json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=access_time,
        )
        access = QueryAccessEvent(
            access_event_id=f"combined-phase5-access-{call.ordinal:03d}",
            execution_id=call.source.execution_id,
            query_context_hash=context.content_hash,
            model_visible_query_hash=to_model_visible_query(context).content_hash,
            snapshot_hash=call.source.prequery_stage.snapshot_hash,
            stage_manifest_hash=canonical_sha256(
                {
                    "combined_manifest": self.manifest.content_hash,
                    "phase5_inputs": self.phase5_inputs.content_hash,
                    "call_spec": call.content_hash,
                    "instruction": request.instruction.content_hash,
                }
            ),
            query_artifact_hash=query_artifact.content_hash,
            prequery_barrier_hash=barrier.content_hash,
            packet_hash=None,
            registered_revealed_at=context.revealed_at,
            accessed_at=access_time,
        )
        access_artifact = self.artifacts.put_bytes(
            (access.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.query-access-event+json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=access_time,
        )
        self.artifacts.ledger.persist_query_access(
            QueryAccessRecord(
                access_event_hash=access.content_hash,
                access_event_id=access.access_event_id,
                execution_id=access.execution_id,
                query_context_hash=access.query_context_hash,
                model_visible_query_hash=access.model_visible_query_hash,
                snapshot_hash=access.snapshot_hash,
                stage_manifest_hash=access.stage_manifest_hash,
                query_artifact_hash=access.query_artifact_hash,
                prequery_barrier_hash=access.prequery_barrier_hash,
                packet_hash=None,
                query_payload_artifact_hash=query_artifact.content_hash,
                access_event_artifact_hash=access_artifact.content_hash,
                registered_revealed_at=access.registered_revealed_at.isoformat(),
                accessed_at=access.accessed_at.isoformat(),
                release_class=ReleaseClass.RESTRICTED,
            ),
            query_payload_artifact=query_artifact,
            access_event_artifact=access_artifact,
        )
        packet_time = access_time + timedelta(microseconds=1)
        packet_values = request.packet.model_dump(mode="python", exclude={"content_hash"})
        packet_values.update(
            packet_id=f"combined-phase5-packet-{call.ordinal:03d}",
            created_at=packet_time,
            release_class=ReleaseClass.RESTRICTED,
        )
        after_packet = EvidencePacket.model_validate(packet_values)
        packet_artifact = self.artifacts.put_bytes(
            (after_packet.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.evidence-packet+json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=packet_time,
        )
        materialization = PacketMaterializationEvent(
            materialization_event_id=f"combined-phase5-packet-{call.ordinal:03d}",
            execution_id=call.source.execution_id,
            query_access_event_hash=access.content_hash,
            snapshot_hash=after_packet.snapshot_hash,
            packet_hash=after_packet.content_hash,
            retrieval_method=after_packet.retrieval_method,
            retrieval_config_hash=canonical_sha256(
                {
                    "operation": "phase5-byte-identical-evidence-reuse-v1",
                    "source_packet_hash": request.packet.content_hash,
                    "after_packet_hash": after_packet.content_hash,
                }
            ),
            started_at=packet_time,
            completed_at=packet_time,
        )
        materialization_artifact = self.artifacts.put_bytes(
            (materialization.to_canonical_json() + "\n").encode("utf-8"),
            media_type=(
                "application/vnd.story-projection.packet-materialization-event+json"
            ),
            release_class=ReleaseClass.RESTRICTED,
            created_at=packet_time,
        )
        self.artifacts.ledger.persist_packet_materialization(
            PacketMaterializationRecord(
                materialization_event_hash=materialization.content_hash,
                materialization_event_id=materialization.materialization_event_id,
                execution_id=materialization.execution_id,
                query_access_event_hash=materialization.query_access_event_hash,
                snapshot_hash=materialization.snapshot_hash,
                packet_hash=materialization.packet_hash,
                retrieval_method=materialization.retrieval_method.value,
                retrieval_config_hash=materialization.retrieval_config_hash,
                packet_artifact_hash=packet_artifact.content_hash,
                materialization_event_artifact_hash=materialization_artifact.content_hash,
                started_at=materialization.started_at.isoformat(),
                completed_at=materialization.completed_at.isoformat(),
                release_class=ReleaseClass.RESTRICTED,
            ),
            packet_artifact=packet_artifact,
            materialization_event_artifact=materialization_artifact,
        )
        run_config = RunConditionConfig(
            config_id=f"combined-phase5-config-{call.call_id}",
            condition=ConditionName.C2_LLM_QUERY,
            budgets=context.budgets,
            maximum_input_tokens=self.runtime.maximum_input_tokens,
            maximum_output_tokens=self.runtime.maximum_output_tokens,
            repair_attempt_budget=call.maximum_repair_attempts,
            seed_block=call.seed_block,
            model_stack_hash=self.runtime.model_manifest_hash,
            decoding_manifest_hash=self.runtime.decoding_manifest_hashes[
                ConditionName.C2_LLM_QUERY
            ],
            decoding_family_hash=self.runtime.decoding_family_hash,
            seed_manifest_hash=self.runtime.seed_manifest_hash,
            resolved_seed=call.vllm_seed,
            prompt_hash=self.runtime.prompt_hashes[ConditionName.C2_LLM_QUERY],
            output_schema_hash=self.runtime.output_schema_hashes[
                ConditionName.C2_LLM_QUERY
            ],
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            capability_manifest_hash=self.runtime.capability_manifest_hashes[
                ConditionName.C2_LLM_QUERY
            ],
            validator_hash=self.runtime.validator_hash,
            upper_ontology_hash=self.runtime.upper_ontology_hash,
        )
        query_processing_started_at = packet_time + timedelta(microseconds=1)
        inputs = HeldOutProduceInputs(
            controller_unit_id=call.source.unit_id,
            preparation=preparation,
            snapshot=self.provider.neutral_by_unit[call.source.unit_id].snapshot,
            packet=after_packet,
            context=context,
            query_access=access,
            prequery_barrier=barrier,
            query_processing_started_at=query_processing_started_at,
            packet_materialization=materialization,
            upper_ontology=self.construction.upper_ontology,
            revisions=(request.instruction.model_visible(),),
            run_config=run_config,
        )
        return inputs, after_packet

    def _phase5_request_for_hash(
        self,
        request_hash: str,
    ) -> tuple[CombinedCallSpec, C2RegenerationRequest]:
        for call in self.manifest.calls:
            if call.phase5_episode_id is None or call.phase5_episode_kind is None:
                continue
            expected = self.phase5_by_episode.get(call.phase5_episode_id)
            if expected is None:
                raise CombinedFactoryError("Phase 5 call lacks its frozen episode")
            before = expected.c2_before_projection
            request = C2RegenerationRequest(
                call_slot_id=f"phase5-c2-{expected.episode_id}",
                episode_id=expected.episode_id,
                kind=call.phase5_episode_kind,
                instruction=expected.instruction,
                packet=expected.packet,
                before_projection_id=before.projection_id,
                before_projection_hash=before.content_hash,
                before_context_hash=before.context_hash,
                before_packet_hash=before.packet_hash,
                protocol_hash=self.phase5_inputs.protocol_hash,
                seed=call.frozen_seed,
            )
            if request.content_hash == request_hash:
                return call, request
        raise CombinedFactoryError("Phase 5 recovery request is absent from the freeze")

    def _phase5_has_any_adapter_trace(
        self,
        call: CombinedCallSpec,
        request: C2RegenerationRequest,
    ) -> bool:
        """Conservatively detect the first possible adapter-side durable effect."""

        query_payload = (request.instruction.after_context.to_canonical_json() + "\n").encode(
            "utf-8"
        )
        query_artifact_hash = hashlib.sha256(query_payload).hexdigest()
        try:
            self.artifacts.ledger.get_artifact(query_artifact_hash)
        except KeyError:
            pass
        else:
            return True
        try:
            self.artifacts.ledger.get_query_access_by_event_id(
                f"combined-phase5-access-{call.ordinal:03d}"
            )
        except KeyError:
            pass
        else:
            return True
        return self._attempt_trace_exists(call, "base") or self._attempt_trace_exists(
            call,
            "repair",
        )

    def _phase5_result_from_completion(
        self,
        *,
        intent: _SemanticIntent,
        completion: _OrdinarySemanticCompletion,
        request: C2RegenerationRequest,
        after_packet: EvidencePacket,
    ) -> Phase5OwnedServiceResult:
        attempt = completion.condition_attempt
        if (
            intent.request_kind != "phase5"
            or intent.phase5_request != request
            or completion.run_id != self.run_id
            or completion.call_id != intent.call_id
            or completion.call_spec_hash != intent.call_spec_hash
            or completion.prepared_hash != request.content_hash
            or completion.semantic_intent_hash != intent.content_hash
            or completion.service_identity_hash != intent.service_identity_hash
            or attempt.condition is not ConditionName.C2_LLM_QUERY
            or not completion.model_call_ids
        ):
            raise CombinedFactoryError("Phase 5 terminal completion identity changed")
        projection_reference = None
        packet_reference = None
        failure_reference = None
        if attempt.outcome is RunOutcome.SUCCEEDED:
            if attempt.projection is None:
                raise CombinedFactoryError("successful Phase 5 completion lacks a projection")
            # The semantic executor has already persisted the exact projection
            # and packet while recording their ledger lineage.  Their CAS
            # address is a hash of the bytes, so attempting to persist those
            # same bytes again under Phase 5-specific media types would either
            # create conflicting metadata or obscure the authoritative source
            # record.  Reuse and verify the existing immutable bytes instead.
            projection_reference = self._phase5_existing_record_reference(
                attempt.projection,
                object_kind="phase5_after_projection",
            )
            packet_reference = self._phase5_existing_record_reference(
                after_packet,
                object_kind="phase5_after_packet",
            )
        else:
            failure_reference = persist_phase5_record(
                self.artifacts,
                attempt,
                object_kind="phase5_failure",
                created_at=completion.completed_at,
            )
        return Phase5OwnedServiceResult(
            request_hash=request.content_hash,
            service_identity_hash=intent.service_identity_hash,
            attempt_status=_feedback_status(attempt.outcome),
            model_call_ids=completion.model_call_ids,
            after_projection=projection_reference,
            after_packet=packet_reference,
            failure_artifact=failure_reference,
            resolver_hash=canonical_sha256(
                {
                    "implementation": "combined-phase5-fresh-c2-v1",
                    "validator": self.runtime.validator_hash,
                    "semantic_request": completion.semantic_request_hash,
                }
            ),
            resolved_at=completion.completed_at,
            checked_at=completion.completed_at,
            cumulative_gpu_seconds_before=completion.cumulative_gpu_seconds_before,
            cumulative_gpu_seconds_after=completion.cumulative_gpu_seconds_after,
        )

    def _phase5_existing_record_reference(
        self,
        value: ImmutableRecord,
        *,
        object_kind: Literal["phase5_after_projection", "phase5_after_packet"],
    ) -> Phase5CASReference:
        """Bind an exact already-persisted semantic output under its Phase 5 role."""

        payload = (value.to_canonical_json() + "\n").encode("utf-8")
        artifact_hash = hashlib.sha256(payload).hexdigest()
        try:
            artifact = self.artifacts.ledger.get_artifact(artifact_hash)
        except KeyError as error:
            raise CombinedFactoryError(
                f"successful {object_kind} lacks its authoritative CAS artifact"
            ) from error
        if artifact.release_class.value != ReleaseClass.RESTRICTED.value:
            raise CombinedFactoryError(
                f"successful {object_kind} artifact is not restricted"
            )
        if (
            artifact.content_hash != artifact_hash
            or self.artifacts.blobs.read_bytes(artifact, allow_restricted=True) != payload
        ):
            raise CombinedFactoryError(
                f"successful {object_kind} CAS bytes changed"
            )
        return Phase5CASReference(
            artifact_hash=artifact.content_hash,
            logical_content_hash=value.content_hash,
            object_kind=object_kind,
            media_type=artifact.media_type,
            release_class=ReleaseClass(artifact.release_class.value),
        )

    def execute_phase5(
        self,
        *,
        service: VLLMService,
        identity: CombinedServiceIdentity,
        request: C2RegenerationRequest,
        model_visible_payload: dict[str, object],
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> Phase5OwnedServiceResult:
        if model_visible_payload != request.model_visible_payload():
            raise CombinedFactoryError("Phase 5 model-visible payload changed")
        call = next(
            (
                item
                for item in self.manifest.calls
                if item.phase5_episode_id == request.episode_id
            ),
            None,
        )
        if call is None or request.seed != call.frozen_seed:
            raise CombinedFactoryError("Phase 5 request is absent from the 49-call freeze")
        result_path = self._result_path(request.content_hash, "phase5")
        intent_path = self._intent_path(request.content_hash, "phase5")
        if result_path.exists():
            recovered = self.recover_phase5(
                request.content_hash,
                service=service,
                repair_authority=repair_authority,
                remaining_required_seconds=remaining_required_seconds,
            )
            if recovered is None:
                raise CombinedFactoryError("Phase 5 result recovery changed")
            return recovered
        inputs, after_packet = self._phase5_produce_inputs(call, request)
        runtime = RuntimeIdentifiers(
            model_id=self.runtime.served_model_name,
            model_revision=self.runtime.model_revision,
            tokenizer_hash=self.runtime.tokenizer_manifest_hash,
            runtime_version=self.runtime.runtime_version,
            prompt_hash=self.runtime.prompt_hashes[ConditionName.C2_LLM_QUERY],
            output_schema_hash=self.runtime.output_schema_hashes[
                ConditionName.C2_LLM_QUERY
            ],
            decoding_config_hash=self.runtime.decoding_manifest_hashes[
                ConditionName.C2_LLM_QUERY
            ],
        )
        semantic = build_c2_construction_request(
            inputs,
            runtime=runtime,
            requested_at=inputs.query_processing_started_at + timedelta(microseconds=1),
        )
        guided = build_development_guided_request(
            root=self.repository,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=call.vllm_seed,
        )
        before = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        intent = _SemanticIntent(
            request_kind="phase5",
            run_id=self.run_id,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            prepared_hash=request.content_hash,
            guided_request_hash=guided.request_hash,
            gpu_event_id=f"{self.run_id}-{call.call_id}-base-gpu",
            service_identity_hash=identity.phase5_identity.content_hash,
            cumulative_gpu_seconds_before=before,
            remaining_required_seconds=remaining_required_seconds,
            created_at=datetime.now(UTC),
            phase5_request=request,
        )
        if intent_path.exists():
            existing = self._read(intent_path, _SemanticIntent)
            if (
                existing.run_id != self.run_id
                or existing.call_id != call.call_id
                or existing.call_spec_hash != call.content_hash
                or existing.prepared_hash != request.content_hash
                or existing.guided_request_hash != guided.request_hash
                or existing.service_identity_hash
                != identity.phase5_identity.content_hash
                or existing.remaining_required_seconds
                != remaining_required_seconds
                or existing.request_kind != "phase5"
                or existing.phase5_request != request
            ):
                raise CombinedFactoryError("Phase 5 intent changed on replay")
            raise CombinedRecoveryRequired(
                "Phase 5 GPU intent has no terminal result; inference cannot be resent"
            )
        self._append(intent_path, intent, "Phase 5 semantic intent")
        condition_attempt, attempts = self._run_generation(
            service=service,
            call=call,
            semantic=semantic,
            config=inputs.run_config,
            inputs=inputs,
            repair_authority=repair_authority,
            authority_request_hash=request.content_hash,
            remaining_required_seconds=remaining_required_seconds,
            phase5_request_hash=request.content_hash,
        )
        completed_at = _strictly_after(
            _as_utc(_event(self.artifacts, attempts[-1].event_id).ended_at)
        )
        after = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        completion = _OrdinarySemanticCompletion(
            run_id=self.run_id,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            prepared_hash=request.content_hash,
            semantic_request_hash=semantic.content_hash,
            semantic_intent_hash=intent.content_hash,
            service_identity_hash=identity.phase5_identity.content_hash,
            condition_attempt=condition_attempt,
            model_call_ids=tuple(item.model_call_id for item in attempts),
            model_request_hashes=tuple(item.guided.request_hash for item in attempts),
            cumulative_gpu_seconds_before=before,
            cumulative_gpu_seconds_after=after,
            started_at=_as_utc(_event(self.artifacts, attempts[0].event_id).started_at),
            completed_at=completed_at,
        )
        self._append(
            self._completion_path(request.content_hash, "phase5"),
            completion,
            "Phase 5 semantic completion",
        )
        result = self._phase5_result_from_completion(
            intent=intent,
            completion=completion,
            request=request,
            after_packet=after_packet,
        )
        self._append(result_path, result, "Phase 5 semantic result")
        return result

    def recover_phase5(
        self,
        request_hash: str,
        *,
        service: VLLMService,
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> Phase5OwnedServiceResult | None:
        intent_path = self._intent_path(request_hash, "phase5")
        result_path = self._result_path(request_hash, "phase5")
        completion_path = self._completion_path(request_hash, "phase5")
        if not result_path.exists():
            call, request = self._phase5_request_for_hash(request_hash)
            if not intent_path.exists():
                if completion_path.exists() or self._phase5_has_any_adapter_trace(call, request):
                    raise CombinedRecoveryRequired(
                        "Phase 5 adapter trace exists without its semantic intent"
                    )
                return None
            intent = self._read(intent_path, _SemanticIntent)
            if (
                intent.run_id != self.run_id
                or intent.prepared_hash != request_hash
                or intent.request_kind != "phase5"
                or intent.phase5_request != request
            ):
                raise CombinedFactoryError("Phase 5 recovery belongs to another run/request")
            if completion_path.exists():
                completion = self._read(
                    completion_path,
                    _OrdinarySemanticCompletion,
                )
                _inputs, after_packet = self._phase5_produce_inputs(call, request)
            else:
                completion, recovered_packet = self._reconstruct_completion(
                    service=service,
                    intent=intent,
                    identity_hash=intent.service_identity_hash,
                    repair_authority=repair_authority,
                    remaining_required_seconds=remaining_required_seconds,
                )
                if recovered_packet is None:
                    raise CombinedFactoryError("Phase 5 recovery lost its evidence packet")
                after_packet = recovered_packet
                self._append(
                    completion_path,
                    completion,
                    "reconstructed Phase 5 semantic completion",
                )
            result = self._phase5_result_from_completion(
                intent=intent,
                completion=completion,
                request=request,
                after_packet=after_packet,
            )
            self._append(result_path, result, "reconstructed Phase 5 semantic result")
            return result
        if not intent_path.exists():
            raise CombinedFactoryError("Phase 5 result lacks its pre-GPU intent")
        if not completion_path.exists():
            raise CombinedFactoryError("Phase 5 result lacks its terminal completion")
        intent = self._read(intent_path, _SemanticIntent)
        completion = self._read(completion_path, _OrdinarySemanticCompletion)
        result = self._read(result_path, Phase5OwnedServiceResult)
        if (
            intent.run_id != self.run_id
            or intent.prepared_hash != request_hash
            or result.request_hash != request_hash
            or result.model_call_ids[0] != f"{self.run_id}-{intent.call_id}-base-model"
            or result.model_call_ids != completion.model_call_ids
            or result.service_identity_hash != completion.service_identity_hash
        ):
            raise CombinedFactoryError("Phase 5 recovery changed run/request lineage")
        return result


class _FrozenCombinedOwnedService:
    """Lifecycle-free scientific surface around one exact live vLLM process."""

    def __init__(
        self,
        *,
        service: VLLMService,
        identity: CombinedServiceIdentity,
        semantic: _CombinedSemanticExecutor,
    ) -> None:
        self.service = service
        self._identity = identity
        self.semantic = semantic

    def identity(self) -> CombinedServiceIdentity:
        return self._identity

    @property
    def actual_allocated_service_seconds(self) -> float:
        return self.service.actual_allocated_service_seconds

    def execute_combined(
        self,
        request: CombinedPreparedCall,
        model_visible_payload: dict[str, object],
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> CombinedServiceCallResult:
        return self.semantic.execute_ordinary(
            service=self.service,
            identity=self._identity,
            prepared=request,
            model_visible_payload=model_visible_payload,
            repair_authority=repair_authority,
            remaining_required_seconds=remaining_required_seconds,
        )

    def recover_combined(
        self,
        request_hash: str,
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> CombinedServiceCallResult | None:
        return self.semantic.recover_ordinary(
            request_hash,
            self._identity,
            service=self.service,
            repair_authority=repair_authority,
            remaining_required_seconds=remaining_required_seconds,
        )

    def execute_phase5_owned(
        self,
        request: C2RegenerationRequest,
        model_visible_payload: dict[str, object],
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> Phase5OwnedServiceResult:
        return self.semantic.execute_phase5(
            service=self.service,
            identity=self._identity,
            request=request,
            model_visible_payload=model_visible_payload,
            repair_authority=repair_authority,
            remaining_required_seconds=remaining_required_seconds,
        )

    def recover_phase5_owned(
        self,
        request: C2RegenerationRequest,
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> Phase5OwnedServiceResult | None:
        return self.semantic.recover_phase5(
            request.content_hash,
            service=self.service,
            repair_authority=repair_authority,
            remaining_required_seconds=remaining_required_seconds,
        )


class _FrozenCombinedLifecycleOwner:
    """One-load lifecycle owner with restart-safe adoption and shutdown recovery."""

    def __init__(
        self,
        *,
        run_id: str,
        configuration: CombinedBlockConfiguration,
        runtime: CombinedRuntimeBinding,
        artifacts: ArtifactStore,
        runtime_root: Path,
        service_factory: Callable[[], VLLMService],
        semantic: _CombinedSemanticExecutor,
    ) -> None:
        self.run_id = run_id
        self.configuration = configuration
        self.runtime = runtime
        self.artifacts = artifacts
        self.runtime_root = runtime_root
        self.service_factory = service_factory
        self.semantic = semantic
        self.activation_intent_path = runtime_root / "combined-activation-intent.json"
        self.checkpoint_path = runtime_root / "combined-vllm-checkpoint.json"
        self.binding_path = runtime_root / "combined-lifecycle-binding.json"
        self.shutdown_intent_path = runtime_root / "combined-shutdown-intent.json"
        self.shutdown_receipt_path = runtime_root / "combined-shutdown-receipt.json"
        self._active: _FrozenCombinedOwnedService | None = None

    def _append(self, path: Path, value: ImmutableRecord, label: str) -> None:
        _append_bytes_atomically(
            path,
            (value.to_canonical_json() + "\n").encode("utf-8"),
            label=label,
        )

    def _read(self, path: Path, model: type[ImmutableRecord]) -> Any:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise CombinedFactoryError(f"unsafe lifecycle state: {path.name}")
        try:
            return model.model_validate_json(path.read_bytes())
        except Exception as error:
            raise CombinedFactoryError(f"invalid lifecycle state {path.name}: {error}") from error

    def _binding(self) -> _LifecycleBinding:
        binding = cast(_LifecycleBinding, self._read(self.binding_path, _LifecycleBinding))
        if binding.run_id != self.run_id:
            raise CombinedFactoryError("combined lifecycle binding belongs to another run")
        return binding

    def _checkpoint_identity(self) -> Mapping[str, object]:
        path = _safe_file(self.checkpoint_path, "combined service checkpoint")
        try:
            value = json.loads(path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CombinedFactoryError("combined service checkpoint is invalid") from error
        if not isinstance(value, Mapping):
            raise CombinedFactoryError("combined service checkpoint root is invalid")
        return value

    def _expected_activation_intent(
        self,
        slot: CombinedActivationSlot,
    ) -> _ActivationIntent:
        return _ActivationIntent(
            run_id=self.run_id,
            activation_slot_hash=slot.content_hash,
            activation_slot=slot,
            service_session_id=f"{self.run_id}-combined-service",
            gpu_event_id=f"{self.run_id}-combined-model-load",
            launcher_configuration_hash=self.runtime.launcher_configuration_hash,
            created_at=slot.created_at,
        )

    def _activation_intent(self, slot: CombinedActivationSlot) -> _ActivationIntent:
        expected = self._expected_activation_intent(slot)
        if self.activation_intent_path.exists():
            persisted = cast(
                _ActivationIntent,
                self._read(self.activation_intent_path, _ActivationIntent),
            )
            if persisted != expected:
                raise CombinedFactoryError("combined activation intent changed on resume")
            return persisted
        self._append(
            self.activation_intent_path,
            expected,
            "combined activation intent",
        )
        return expected

    def _bind_live_service(
        self,
        *,
        slot: CombinedActivationSlot,
        service: VLLMService,
    ) -> _FrozenCombinedOwnedService:
        """Checkpoint and bind an already-ready service without loading another model."""

        event_id = f"{self.run_id}-combined-model-load"
        service.write_resume_checkpoint(self.checkpoint_path)
        checkpoint = self._checkpoint_identity()
        if (
            checkpoint.get("session_id") != f"{self.run_id}-combined-service"
            or checkpoint.get("accounting_session_id") != event_id
            or checkpoint.get("configuration_hash")
            != self.runtime.launcher_configuration_hash
            or checkpoint.get("pid") != service.pid
        ):
            raise CombinedFactoryError("combined service checkpoint identity changed")
        event = _event(self.artifacts, event_id)
        activated_at = _strictly_after(_as_utc(event.ended_at))
        cumulative = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        phase5_identity = Phase5OwnedServiceIdentity(
            service_id=f"{self.run_id}-combined-service",
            global_accounting_id=slot.global_accounting_id,
            model_manifest_hash=self.runtime.model_manifest_hash,
            decoding_manifest_hash=self.runtime.decoding_manifest_hashes[
                ConditionName.C2_LLM_QUERY
            ],
            model_load_event_id=event_id,
            model_load_event_record_hash=_record_hash(event, _GPU_EVENT_FIELDS),
            cumulative_gpu_seconds_at_handoff=cumulative,
            handed_off_at=activated_at,
        )
        identity = CombinedServiceIdentity(
            service_id=f"{self.run_id}-combined-service",
            global_accounting_id=slot.global_accounting_id,
            activation_slot_hash=slot.content_hash,
            runtime_binding_hash=self.runtime.content_hash,
            model_manifest_hash=self.runtime.model_manifest_hash,
            decoding_manifest_hashes=self.runtime.decoding_manifest_hashes,
            model_load_event_id=event_id,
            model_load_event_record_hash=_record_hash(event, _GPU_EVENT_FIELDS),
            service_pid=cast(int, checkpoint["pid"]),
            service_start_ticks=cast(int, checkpoint["process_start_ticks"]),
            gpu_seconds_before_load=slot.gpu_seconds_before,
            cumulative_gpu_seconds_after_load=cumulative,
            activated_at=activated_at,
            phase5_identity=phase5_identity,
        )
        binding = _LifecycleBinding(
            run_id=self.run_id,
            activation_slot_hash=slot.content_hash,
            service_identity=identity,
            checkpoint_file_sha256=_file_sha256(self.checkpoint_path),
            created_at=activated_at,
        )
        self._append(self.binding_path, binding, "combined lifecycle binding")
        owned = _FrozenCombinedOwnedService(
            service=service,
            identity=identity,
            semantic=self.semantic,
        )
        self._active = owned
        return owned

    def activate(self, slot: CombinedActivationSlot) -> CombinedOwnedService:
        if slot.run_id != self.run_id:
            raise CombinedFactoryError("activation slot belongs to another run")
        if self.binding_path.exists() or self.checkpoint_path.exists() or self._active is not None:
            raise CombinedRecoveryRequired(
                "combined lifecycle has prior state; activation cannot consume a second load"
            )
        event_id = f"{self.run_id}-combined-model-load"
        if (
            any(item.event_id == event_id for item in self.artifacts.ledger.gpu_events())
            or self.artifacts.ledger.latest_gpu_service_journal(event_id) is not None
        ):
            raise CombinedRecoveryRequired(
                "combined model-load lineage already exists without a lifecycle binding"
            )
        self._activation_intent(slot)
        service = self.service_factory()
        try:
            service.start(
                session_id=f"{self.run_id}-combined-service",
                event_id=event_id,
                watchdog_seconds=self.configuration.model_load_watchdog_seconds,
                remaining_required_seconds=max(
                    0.0,
                    slot.remaining_registered_p95_seconds_before
                    - self.configuration.model_load_p95_seconds,
                ),
            )
            return self._bind_live_service(slot=slot, service=service)
        except BaseException as activation_error:
            try:
                service.shutdown()
            except BaseException as shutdown_error:
                raise CombinedRecoveryRequired(
                    "combined activation failed and physical shutdown/accounting "
                    "could not be verified"
                ) from shutdown_error
            raise activation_error

    def recover_active(self, activation_slot_hash: str) -> CombinedOwnedService | None:
        if not self.binding_path.exists():
            event_id = f"{self.run_id}-combined-model-load"
            slot_intent_exists = self.activation_intent_path.exists()
            physical_trace_exists = (
                self.checkpoint_path.exists()
                or any(item.event_id == event_id for item in self.artifacts.ledger.gpu_events())
                or self.artifacts.ledger.latest_gpu_service_journal(event_id) is not None
            )
            intent: _ActivationIntent | None = None
            if slot_intent_exists:
                intent = cast(
                    _ActivationIntent,
                    self._read(self.activation_intent_path, _ActivationIntent),
                )
                if (
                    intent.run_id != self.run_id
                    or intent.activation_slot_hash != activation_slot_hash
                    or intent.activation_slot.content_hash != activation_slot_hash
                    or intent.service_session_id != f"{self.run_id}-combined-service"
                    or intent.gpu_event_id != event_id
                    or intent.launcher_configuration_hash
                    != self.runtime.launcher_configuration_hash
                ):
                    raise CombinedFactoryError(
                        "combined activation intent changed on resume"
                    )
            elif physical_trace_exists:
                raise CombinedRecoveryRequired(
                    "combined activation began but lacks a complete lifecycle binding "
                    "and its durable intent"
                )
            if physical_trace_exists:
                service = self.service_factory()
                adopted = False
                try:
                    if self.checkpoint_path.exists():
                        adopted = service.resume_from_checkpoint(self.checkpoint_path)
                    if not adopted:
                        adopted = service.resume_live_service_lease(
                            expected_session_id=f"{self.run_id}-combined-service",
                            expected_event_id=event_id,
                            watchdog_seconds=(
                                self.configuration.model_load_watchdog_seconds
                            ),
                        )
                    if adopted:
                        assert intent is not None
                        return self._bind_live_service(
                            slot=intent.activation_slot,
                            service=service,
                        )
                    recovered = service.recover_stale_service_lease()
                except CombinedRecoveryRequired:
                    if adopted:
                        with suppress(Exception):
                            service.shutdown()
                    raise
                except BaseException as error:
                    if adopted:
                        with suppress(Exception):
                            service.shutdown()
                    raise CombinedRecoveryRequired(
                        "combined activation could not safely adopt its durable lease"
                    ) from error
                if recovered is not None:
                    raise CombinedRecoveryRequired(
                        "combined activation terminated before lifecycle binding; "
                        "a second model load is forbidden"
                    )
                raise CombinedRecoveryRequired(
                    "combined activation began but lacks a complete lifecycle binding"
                )
            # The outer activation slot was persisted, but lifecycle activation
            # never began.  Returning None authorizes the coordinator to consume
            # that same slot exactly once; activate() repeats these checks.
            return None
        binding = self._binding()
        if binding.activation_slot_hash != activation_slot_hash:
            raise CombinedFactoryError("active recovery changed its activation slot")
        if self.shutdown_receipt_path.exists() or self.shutdown_intent_path.exists():
            return None
        if self._active is not None:
            if self._active.identity() != binding.service_identity:
                raise CombinedFactoryError("in-memory combined service identity changed")
            return self._active
        if _file_sha256(_safe_file(self.checkpoint_path, "combined checkpoint")) != (
            binding.checkpoint_file_sha256
        ):
            raise CombinedFactoryError("combined checkpoint changed after activation")
        service = self.service_factory()
        if not service.resume_from_checkpoint(self.checkpoint_path):
            return None
        if service.pid != binding.service_identity.service_pid:
            raise CombinedFactoryError("adopted combined process identity changed")
        self._active = _FrozenCombinedOwnedService(
            service=service,
            identity=binding.service_identity,
            semantic=self.semantic,
        )
        return self._active

    def _finish_shutdown(
        self,
        service: VLLMService,
        intent: _ShutdownIntent,
    ) -> CombinedServiceShutdownReceipt:
        uptime = service.shutdown()
        if uptime is None:
            raise CombinedFactoryError("combined shutdown lacks accounted service uptime")
        receipt = CombinedServiceShutdownReceipt(
            service_identity_hash=intent.service_identity_hash,
            service_pid=intent.service_pid,
            service_start_ticks=intent.service_start_ticks,
            shutdown_started_at=intent.requested_at,
            stopped_at=uptime.ended_at,
            cumulative_gpu_seconds_after_shutdown=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
        )
        self._append(self.shutdown_receipt_path, receipt, "combined shutdown receipt")
        self._active = None
        return receipt

    def shutdown(
        self,
        service: CombinedOwnedService,
        requested_at: datetime,
    ) -> CombinedServiceShutdownReceipt:
        if not isinstance(service, _FrozenCombinedOwnedService):
            raise CombinedFactoryError("combined shutdown received an unowned service")
        binding = self._binding()
        if service.identity() != binding.service_identity:
            raise CombinedFactoryError("combined shutdown service identity changed")
        intent = _ShutdownIntent(
            run_id=self.run_id,
            service_identity_hash=binding.service_identity.content_hash,
            service_pid=binding.service_identity.service_pid,
            service_start_ticks=binding.service_identity.service_start_ticks,
            requested_at=requested_at,
        )
        self._append(self.shutdown_intent_path, intent, "combined shutdown intent")
        return self._finish_shutdown(service.service, intent)

    def recover_shutdown(
        self,
        service_identity_hash: str,
    ) -> CombinedServiceShutdownReceipt | None:
        if self.shutdown_receipt_path.exists():
            receipt = cast(
                CombinedServiceShutdownReceipt,
                self._read(self.shutdown_receipt_path, CombinedServiceShutdownReceipt),
            )
            if receipt.service_identity_hash != service_identity_hash:
                raise CombinedFactoryError("shutdown receipt belongs to another service")
            return receipt
        if not self.shutdown_intent_path.exists():
            return None
        intent = cast(_ShutdownIntent, self._read(self.shutdown_intent_path, _ShutdownIntent))
        binding = self._binding()
        if (
            intent.run_id != self.run_id
            or intent.service_identity_hash != service_identity_hash
            or binding.service_identity.content_hash != service_identity_hash
        ):
            raise CombinedFactoryError("shutdown recovery changed run/service identity")
        if self._active is not None:
            return self._finish_shutdown(self._active.service, intent)
        service = self.service_factory()
        if service.resume_from_checkpoint(self.checkpoint_path):
            return self._finish_shutdown(service, intent)
        recovered = service.recover_stale_service_lease()
        checkpoint = self._checkpoint_identity()
        accounting_id = checkpoint.get("accounting_session_id")
        if recovered is None and isinstance(accounting_id, str):
            recovered = self.artifacts.ledger.get_gpu_service_session(accounting_id)
        if recovered is None:
            raise CombinedRecoveryRequired(
                "shutdown intent exists but physical stop/accounting cannot be recovered"
            )
        receipt = CombinedServiceShutdownReceipt(
            service_identity_hash=service_identity_hash,
            service_pid=intent.service_pid,
            service_start_ticks=intent.service_start_ticks,
            shutdown_started_at=intent.requested_at,
            stopped_at=_as_utc(recovered.ended_at),
            cumulative_gpu_seconds_after_shutdown=(
                self.artifacts.ledger.gpu_summary().total_allocated_seconds
            ),
        )
        self._append(self.shutdown_receipt_path, receipt, "recovered combined shutdown")
        return receipt


def create_frozen_production_combined_bundle(
    *,
    repository: Path,
    run_id: str,
    configuration: CombinedBlockConfiguration,
    manifest: CombinedCallManifest,
    runtime: CombinedRuntimeBinding,
    upstream_gate: CombinedUpstreamGate,
    phase5_inputs: Phase5ExecutionInputManifest,
    phase5_protocol: Any,
    phase5_primary_gate: PrimaryHeldOutResultsGate,
    compiler_manifest_path: Path,
    compiled_call_manifest_path: Path,
    compiled_runtime_binding_path: Path,
    compiled_upstream_gate_path: Path,
    compiler_source_paths: Mapping[str, Path],
    ledger_path: Path,
    artifact_root: Path,
    runtime_root: Path,
    quota_root: Path,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    selected_model_freeze_path: Path,
    source_association_path: Path,
    port: int = 8000,
) -> CombinedProductionBundle:
    """Create the sole registered combined adapter without starting the GPU.

    All model, source, semantic, CAS, and resource bindings are replayed here.
    The returned lifecycle owner is the only object capable of loading vLLM.
    """

    del phase5_protocol  # already hash-validated by the outer controller
    repository = repository.resolve(strict=True)
    if not run_id or any(character.isspace() for character in run_id):
        raise CombinedFactoryError("combined run_id must be a plain nonempty identifier")
    if (
        manifest.configuration_hash != configuration.content_hash
        or manifest.runtime_binding_hash != runtime.content_hash
        or manifest.upstream_gate_hash != upstream_gate.content_hash
        or manifest.phase5_input_manifest_hash != phase5_inputs.content_hash
    ):
        raise CombinedFactoryError("combined factory inputs changed after compilation")
    selections = load_registered_combined_selections(repository, configuration)
    try:
        verify_combined_compiler_receipt(
            compiler_manifest_path=compiler_manifest_path,
            call_manifest_path=compiled_call_manifest_path,
            runtime_binding_path=compiled_runtime_binding_path,
            upstream_gate_path=compiled_upstream_gate_path,
            source_paths=compiler_source_paths,
            configuration=configuration,
            selections=selections,
            runtime_binding=runtime,
            upstream_gate=upstream_gate,
            call_manifest=manifest,
            phase5_inputs=phase5_inputs,
            phase5_primary_gate=phase5_primary_gate,
        )
    except Exception as error:
        raise CombinedFactoryError("combined compiler receipt verification failed") from error
    limits = ResourceLimits.load(repository / "configs/study/resource_limits.json")
    if (
        limits.scheduled_gpu_seconds != configuration.scheduled_limit_seconds
        or limits.hard_gpu_seconds != configuration.hard_limit_seconds
        or limits.maximum_cpu_workers > 8
        or limits.model_cpu_offload_allowed
        or limits.generation_concurrency != 1
    ):
        raise CombinedFactoryError("combined factory differs from global resource limits")
    quota_root = quota_root.resolve(strict=True)
    configured_output = Path(configuration.output_root)
    restricted_root = repository / configured_output.parent
    if restricted_root != repository / "artifacts" / "restricted":
        raise CombinedFactoryError("combined restricted-root authority changed")
    try:
        ledger_path = _require_contained_writable_path(
            ledger_path,
            authority_root=restricted_root,
            label="combined SQLite ledger",
        )
        artifact_root = _require_contained_writable_path(
            artifact_root,
            authority_root=restricted_root,
            label="combined restricted CAS",
        )
        runtime_root = _require_contained_writable_path(
            runtime_root,
            authority_root=restricted_root,
            label="combined runtime state",
        )
    except CombinedProductionError as error:
        raise CombinedFactoryError(str(error)) from error
    candidate_paths = (
        repository,
        shared_cache,
        ledger_path.parent,
        artifact_root,
        runtime_root,
    )
    for label, candidate in zip(
        ("repository", "shared cache", "ledger", "artifact root", "runtime root"),
        candidate_paths,
        strict=True,
    ):
        try:
            candidate.resolve(strict=False).relative_to(quota_root)
        except ValueError as error:
            raise CombinedFactoryError(f"{label} lies outside the controlled quota") from error
    for path in (ledger_path.parent, artifact_root, runtime_root):
        if path.is_symlink():
            raise CombinedFactoryError("combined writable path must not be a symlink")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(runtime_root, 0o700)

    ledger = Ledger(ledger_path)
    try:
        artifacts = ArtifactStore(
            BlobStore(artifact_root, compression=Compression.GZIP),
            ledger,
        )
        storage = StoragePreflight(
            quota_root,
            controlled_paths=candidate_paths,
            budget=StorageBudget(
                total_allocation_bytes=limits.maximum_project_allocation_bytes,
                max_occupied_bytes=limits.maximum_project_occupied_bytes,
                min_headroom_bytes=limits.minimum_storage_headroom_bytes,
            ),
        )
        _require_registered_combined_storage_preflights(
            repository,
            storage,
            ledger,
        )
        if not math.isclose(
            ledger.gpu_summary().total_allocated_seconds,
            upstream_gate.actual_allocated_gpu_seconds_before_block,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CombinedFactoryError("combined ledger differs from admitted predecessor")

        freeze = _selected_freeze(selected_model_freeze_path)
        if freeze.get("manifest_sha256") != runtime.selected_model_freeze_hash:
            raise CombinedFactoryError("combined runtime binds another selected-model freeze")
        association = validate_source_association(
            _safe_file(source_association_path, "source association"),
            source_root=repository,
        )
        if association.get("manifest_sha256") != runtime.source_tree_association_hash:
            raise CombinedFactoryError("combined source association changed")
        policy_path = repository / "configs/study/fallback_model.json"
        policy = FallbackModelPolicy.load(policy_path)
        snapshot_manifest = validate_fallback_snapshot_manifest(
            _safe_file(verified_model_manifest_path, "verified model manifest"),
            policy=policy,
            policy_path=policy_path,
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
        )
        snapshot_hash = cast(str, snapshot_manifest["manifest_sha256"])
        if snapshot_hash != freeze.get("snapshot_manifest_sha256"):
            raise CombinedFactoryError("combined model snapshot differs from selected freeze")
        launcher = VLLMLaunchConfiguration.from_model_configuration(
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            model_configuration_path=repository / "configs/study/model.json",
            model_candidate="fallback",
            verified_snapshot_manifest_sha256=snapshot_hash,
            port=port,
        )
        if (
            launcher.configuration_hash != freeze.get("launcher_configuration_sha256")
            or launcher.configuration_hash != runtime.launcher_configuration_hash
            or runtime.model_manifest_hash != runtime.launcher_configuration_hash
            or launcher.served_model_name != runtime.served_model_name
        ):
            raise CombinedFactoryError("combined vLLM launcher/model binding changed")
        tokenizer_manifest = capture_tokenizer_manifest(
            snapshot_path,
            repository=FALLBACK_MODEL_REPOSITORY,
            revision=FALLBACK_MODEL_REVISION,
        )
        if (
            tokenizer_manifest.manifest_sha256 != freeze.get("tokenizer_manifest_sha256")
            or tokenizer_manifest.manifest_sha256 != runtime.tokenizer_manifest_hash
        ):
            raise CombinedFactoryError("combined tokenizer changed")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
            revision=FALLBACK_MODEL_REVISION,
        )
        construction = DevelopmentConstructionConfiguration.load(
            repository / configuration.development_construction_path
        )
        validator_hash = development_validator_hash(repository)
        repair_prompt_file_sha256 = _file_sha256(
            repository / "prompts/repair/prompt_v1.md"
        )
        repair_policy_hash = _registered_repair_policy_hash(
            configuration=configuration,
            repair_prompt_file_sha256=repair_prompt_file_sha256,
            validator_hash=validator_hash,
        )
        if (
            construction.upper_ontology.content_hash != runtime.upper_ontology_hash
            or construction.first_pass_input_tokens != runtime.maximum_input_tokens
            or construction.first_pass_output_tokens != runtime.maximum_output_tokens
            or construction.repair_input_tokens != runtime.repair_maximum_input_tokens
            or construction.repair_output_tokens != runtime.repair_maximum_output_tokens
            or validator_hash != runtime.validator_hash
            or repair_policy_hash != runtime.repair_policy_hash
            or configuration.seed_manifest_hash != runtime.seed_manifest_hash
            or _file_sha256(repository / configuration.decoding_configuration_path)
            != runtime.decoding_configuration_file_sha256
        ):
            raise CombinedFactoryError("combined semantic constants changed")
        conditions = (
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        )
        observed_family = set()
        for condition in conditions:
            request_runtime = development_request_runtime(
                root=repository,
                condition=condition,
                tokenizer_manifest=tokenizer_manifest,
                seed=configuration.vllm_seed,
            )
            observed_family.add(request_runtime.decoding_manifest.comparison_family_hash)
            if (
                request_runtime.prompt_hash != runtime.prompt_hashes[condition]
                or request_runtime.output_schema_hash != runtime.output_schema_hashes[condition]
                or request_runtime.decoding_manifest.content_hash
                != runtime.decoding_manifest_hashes[condition]
                or request_runtime.capability_manifest.content_hash
                != runtime.capability_manifest_hashes[condition]
            ):
                raise CombinedFactoryError(
                    f"combined {condition.value} prompt/schema/decoder changed"
                )
        if observed_family != {runtime.decoding_family_hash}:
            raise CombinedFactoryError("combined decoding comparison family changed")
        neutral = _load_neutral_sources(repository, manifest)
        meter = AllocatedGPUMeter.from_limits(ledger, limits)
        sampler = ResourceSampler(limits=limits, storage=storage, ledger=ledger)
        provider = _ProductionInputProvider(
            runtime=runtime,
            construction=construction,
            artifacts=artifacts,
            neutral_by_unit=neutral,
        )
        semantic = _CombinedSemanticExecutor(
            repository=repository,
            run_id=run_id,
            manifest=manifest,
            runtime=runtime,
            phase5_inputs=phase5_inputs,
            provider=provider,
            construction=construction,
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            artifacts=artifacts,
            runtime_root=runtime_root,
        )

        def service_factory() -> VLLMService:
            client = VLLMGuidedJSONClient(launcher.base_url)
            return VLLMService(
                configuration=launcher,
                client=client,
                meter=meter,
                log_path=runtime_root / "combined-vllm.log",
                startup_resource_sampler=sampler,
                preflight_endpoint_check=lambda: client.health(0.25),
                readiness_check=lambda: client.ready(
                    2.0,
                    model_name=FALLBACK_SERVED_MODEL_NAME,
                ),
            )

        lifecycle = _FrozenCombinedLifecycleOwner(
            run_id=run_id,
            configuration=configuration,
            runtime=runtime,
            artifacts=artifacts,
            runtime_root=runtime_root,
            service_factory=service_factory,
            semantic=semantic,
        )

        def phase5_storage_preflight() -> None:
            _require_registered_phase_five_transition_storage_preflight(
                repository,
                storage,
                ledger,
            )

        return CombinedProductionBundle(
            provider=provider,
            lifecycle_owner=lifecycle,
            artifacts=artifacts,
            phase5_storage_preflight=phase5_storage_preflight,
        )
    except BaseException:
        ledger.close()
        raise


__all__ = [
    "CombinedFactoryError",
    "CombinedProductionBundle",
    "create_frozen_production_combined_bundle",
]
