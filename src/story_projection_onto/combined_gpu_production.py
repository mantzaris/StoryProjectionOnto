"""Fail-closed owner for the single registered 49-call GPU allocation.

The controller deliberately knows how to own exactly one service lifecycle but
does not know how to launch vLLM or invent ontology semantics.  A production
adapter supplies those operations.  Before every external side effect the
controller writes an immutable intent record; on resume it permits recovery
lookups only and never resends an uncertain request.

The nine Phase 5 calls continue through :mod:`phase5_production`.  They receive
a capability-narrow view of the same already-running service, so the Phase 5
code cannot load or stop the model.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Protocol, Self, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import scan_model_payload
from story_projection_onto.combined_gpu_block import (
    COMBINED_BASE_CALL_COUNT,
    CombinedBlockConfiguration,
    CombinedBlockError,
    CombinedCallClass,
    CombinedCallManifest,
    CombinedCallSpec,
    CombinedRuntimeBinding,
    CombinedUpstreamGate,
    OneSwitchFingerprint,
    PublicEvidencePacketPointer,
    RestrictedArtifactPointer,
    assert_one_switch_only,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ConditionAttemptRecord,
    ConditionPreparation,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCapabilities,
    ConstructionRequest,
    EvidencePacket,
    Identifier,
    ImmutableRecord,
    PacketMaterializationEvent,
    PreQueryInventory,
    QueryAccessEvent,
    ReleaseClass,
    RunOutcome,
    RuntimeIdentifiers,
    Sha256Digest,
    UpperOntology,
    canonical_sha256,
    to_model_visible_packet,
)
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
    FeedbackProtocolConfiguration,
)
from story_projection_onto.phase5_execution import (
    C2RegenerationRequest,
    C2RegenerationResult,
    Phase5ExecutionInputManifest,
    Phase5JournalIndex,
    PrimaryHeldOutResultsGate,
    SQLiteFeedbackLedgerVerifier,
    execute_phase5,
    materialize_phase5_cpu_ledger,
    validate_phase5_inputs,
    validate_phase5_ledger_preflight,
)
from story_projection_onto.phase5_production import (
    Phase5OwnedServiceIdentity,
    Phase5OwnedServiceResult,
    build_metered_phase5_adapter,
)
from story_projection_onto.store import (
    ArtifactRecord,
    ArtifactStore,
    AttemptKind,
    GpuEventKind,
    ModelBackend,
    ModelCallRole,
    PacketMaterializationRecord,
    QueryAccessRecord,
    RetryClass,
)
from story_projection_onto.store import (
    ReleaseClass as StoreReleaseClass,
)

COMBINED_ORDINARY_CALL_COUNT = 40
COMBINED_PHASE5_CALL_COUNT = 9
_TERMINAL_OUTCOMES = {
    RunOutcome.SUCCEEDED,
    RunOutcome.INVALID,
    RunOutcome.FAILED,
    RunOutcome.TIMED_OUT,
    RunOutcome.INTERRUPTED,
}


class CombinedProductionError(CombinedBlockError):
    """The combined production journal or injected service violated its contract."""


class CombinedRecoveryRequired(CombinedProductionError):
    """A durable side-effect intent has no safely recoverable terminal record."""


def _record_hash(value: object, fields: tuple[str, ...]) -> Sha256Digest:
    return canonical_sha256({name: getattr(value, name) for name in fields})


_MODEL_CALL_FIELDS = (
    "model_call_id",
    "job_id",
    "attempt_id",
    "gpu_event_id",
    "backend",
    "call_role",
    "retry_class",
    "model_manifest_hash",
    "decoding_manifest_hash",
    "request_hash",
    "response_artifact_hash",
    "construction_unit_hash",
    "served_context_count",
    "prompt_tokens",
    "completion_tokens",
    "allocated_gpu_microseconds",
    "successful",
    "created_at",
)
_GPU_EVENT_FIELDS = (
    "event_id",
    "event_kind",
    "allocated_microseconds",
    "started_at",
    "ended_at",
    "succeeded",
    "job_id",
    "attempt_id",
    "details_json",
)


def _aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CombinedProductionError("GPU ledger timestamp is not timezone-aware")
    return parsed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_temporary_path(path: Path, payload: bytes) -> Path:
    digest = hashlib.sha256(payload).hexdigest()
    return path.parent / f".{path.name}.{digest}.tmp"


def _verified_atomic_temporary(path: Path) -> Path | None:
    """Return a complete interrupted write, rejecting every ambiguous fragment."""

    name = path.name
    if not (name.startswith(".") and name.endswith(".tmp")):
        return None
    stem = name[1:-4]
    target_name, separator, claimed_hash = stem.rpartition(".")
    if (
        not separator
        or not target_name
        or len(claimed_hash) != 64
        or any(character not in "0123456789abcdef" for character in claimed_hash)
        or not path.is_file()
    ):
        raise CombinedProductionError(f"invalid atomic journal temporary: {name}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != claimed_hash:
        raise CombinedProductionError(f"truncated atomic journal temporary: {name}")
    return path.parent / target_name


def _append_bytes_atomically(path: Path, payload: bytes, *, label: str) -> None:
    """Publish immutable bytes without ever exposing a partial target file."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise CombinedProductionError(f"append-only {label} changed")
        temporary = _atomic_temporary_path(path, payload)
        if temporary.exists():
            recovered_target = _verified_atomic_temporary(temporary)
            if recovered_target != path or temporary.read_bytes() != payload:
                raise CombinedProductionError(f"ambiguous atomic {label} recovery")
            temporary.unlink()
            _fsync_directory(path.parent)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    CombinedJournal._assert_no_symlink_chain(path.parent)
    temporary = _atomic_temporary_path(path, payload)
    if temporary.exists():
        recovered_target = _verified_atomic_temporary(temporary)
        if recovered_target != path or temporary.read_bytes() != payload:
            raise CombinedProductionError(f"ambiguous atomic {label} recovery")
    else:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        _fsync_directory(path.parent)
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        if not path.is_file() or path.read_bytes() != payload:
            raise CombinedProductionError(
                f"concurrent append-only {label} changed"
            ) from error
    _fsync_directory(path.parent)
    temporary.unlink()
    _fsync_directory(path.parent)


class CombinedActivationSlot(ImmutableRecord):
    run_id: Identifier
    manifest_hash: Sha256Digest
    configuration_hash: Sha256Digest
    runtime_binding_hash: Sha256Digest
    upstream_gate_hash: Sha256Digest
    global_accounting_id: Identifier
    gpu_seconds_before: float = Field(ge=0.0)
    model_load_event_hashes_before: tuple[Sha256Digest, ...]
    short_reserve_slots_consumed_before: Annotated[int, Field(ge=0, le=4)]
    remaining_registered_p95_seconds_before: float = Field(ge=0.0)
    intended_model_load_count: Literal[1] = 1
    maximum_concurrency: Literal[1] = 1
    created_at: AwareDatetime


class CombinedServiceIdentity(ImmutableRecord):
    service_id: Identifier
    global_accounting_id: Identifier
    activation_slot_hash: Sha256Digest
    runtime_binding_hash: Sha256Digest
    model_manifest_hash: Sha256Digest
    decoding_manifest_hashes: Mapping[ConditionName, Sha256Digest]
    model_load_event_id: Identifier
    model_load_event_record_hash: Sha256Digest
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    gpu_seconds_before_load: float = Field(ge=0.0)
    cumulative_gpu_seconds_after_load: float = Field(gt=0.0)
    activated_at: AwareDatetime
    phase5_identity: Phase5OwnedServiceIdentity
    model_load_count: Literal[1] = 1
    maximum_concurrency: Literal[1] = 1
    lifecycle_owner: Literal["combined_gpu_controller"] = "combined_gpu_controller"

    @model_validator(mode="after")
    def phase5_is_the_same_service(self) -> Self:
        phase5 = self.phase5_identity
        if (
            phase5.service_id != self.service_id
            or phase5.global_accounting_id != self.global_accounting_id
            or phase5.model_manifest_hash != self.model_manifest_hash
            or phase5.model_load_event_id != self.model_load_event_id
            or phase5.model_load_event_record_hash != self.model_load_event_record_hash
            or phase5.decoding_manifest_hash
            != self.decoding_manifest_hashes.get(ConditionName.C2_LLM_QUERY)
            or phase5.cumulative_gpu_seconds_at_handoff < self.cumulative_gpu_seconds_after_load
            or phase5.handed_off_at < self.activated_at
        ):
            raise ValueError("Phase 5 lease is not a narrow view of the combined service")
        required = {
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        }
        if set(self.decoding_manifest_hashes) != required:
            raise ValueError("combined service must bind all four active decoding manifests")
        if self.cumulative_gpu_seconds_after_load <= self.gpu_seconds_before_load:
            raise ValueError("combined service identity does not account for model loading")
        return self


class ParaphraseRevealReceipt(ImmutableRecord):
    call_spec_hash: Sha256Digest
    base_context_hash: Sha256Digest
    paraphrase_context_hash: Sha256Digest
    semantic_invariant_hash: Sha256Digest
    reused_c2_preparation_hash: Sha256Digest
    reused_c2_inventory_hash: Sha256Digest
    prequery_barrier_hash: Sha256Digest
    prequery_barrier_sealed_at: AwareDatetime
    access_event: QueryAccessEvent
    access_event_artifact_hash: Sha256Digest
    packet_reuse_event: PacketMaterializationEvent
    packet_reuse_event_artifact_hash: Sha256Digest
    packet_hash: Sha256Digest
    packet_artifact_hash: Sha256Digest
    revealed_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def chronology_and_reuse_are_explicit(self) -> Self:
        if not (
            self.prequery_barrier_sealed_at
            < self.revealed_at
            <= self.access_event.accessed_at
            <= self.packet_reuse_event.started_at
            <= self.packet_reuse_event.completed_at
        ):
            raise ValueError("paraphrase reveal chronology is not strictly post-barrier")
        if (
            self.access_event.prequery_barrier_hash != self.prequery_barrier_hash
            or self.access_event.query_context_hash != self.paraphrase_context_hash
            or self.packet_reuse_event.query_access_event_hash != self.access_event.content_hash
            or self.packet_reuse_event.packet_hash != self.packet_hash
        ):
            raise ValueError("paraphrase reveal does not bind its reused packet")
        return self


class ParaphraseRevealIntent(ImmutableRecord):
    """Durable reveal plan written before either CAS or SQLite is mutated."""

    call_spec_hash: Sha256Digest
    query_payload_artifact_hash: Sha256Digest
    access_event: QueryAccessEvent
    access_event_artifact_hash: Sha256Digest
    packet_reuse_event: PacketMaterializationEvent
    packet_reuse_event_artifact_hash: Sha256Digest
    packet_artifact_hash: Sha256Digest
    created_at: AwareDatetime


class CombinedPreparedCall(ImmutableRecord):
    call_spec_hash: Sha256Digest
    condition: ConditionName
    request: ConstructionRequest
    preparation_hash: Sha256Digest
    empty_inventory_hash: Sha256Digest
    prequery_barrier_hash: Sha256Digest
    query_access_event_hash: Sha256Digest
    packet_artifact: PublicEvidencePacketPointer
    baseline_fingerprint: OneSwitchFingerprint | None = None
    ablated_fingerprint: OneSwitchFingerprint | None = None
    paraphrase_reveal: ParaphraseRevealReceipt | None = None
    prepared_at: AwareDatetime
    scorer_namespace_visible: Literal[False] = False

    @model_validator(mode="after")
    def request_is_active_and_gold_free(self) -> Self:
        if self.condition is not self.request.condition:
            raise ValueError("prepared condition and semantic request differ")
        if self.request.fixed_ontology is not None:
            raise ValueError("combined active construction cannot receive a fixed ontology")
        if self.request.packet.packet_hash != self.packet_artifact.logical_content_hash:
            raise ValueError("prepared request and public packet reference differ")
        if self.request.requested_at < self.prepared_at:
            raise ValueError("semantic request predates input preparation")
        try:
            scan_model_payload(self.model_visible_payload())
        except Exception as error:
            raise ValueError(
                f"combined prepared payload crosses the gold firewall: {error}"
            ) from error
        return self

    def model_visible_payload(self) -> dict[str, object]:
        return self.request.model_dump(mode="json", exclude={"content_hash"})


class CombinedCallSlot(ImmutableRecord):
    ordinal: Annotated[int, Field(ge=1, le=COMBINED_BASE_CALL_COUNT)]
    call_id: Identifier
    call_spec_hash: Sha256Digest
    manifest_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    prepared_call: CombinedPreparedCall
    cumulative_gpu_seconds_before: float = Field(ge=0.0)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def exact_prepared_spec(self) -> Self:
        if self.prepared_call.call_spec_hash != self.call_spec_hash:
            raise ValueError("combined slot prepared another call spec")
        return self


class CombinedRepairClaim(ImmutableRecord):
    claim_id: Identifier
    call_id: Identifier
    call_spec_hash: Sha256Digest
    request_hash: Sha256Digest
    base_model_call_id: Identifier
    global_slot_number: Annotated[int, Field(ge=1, le=4)]
    reserve_class: Literal["reserve_short"] = "reserve_short"
    watchdog_seconds: Literal[90] = 90
    claimed_at: AwareDatetime
    consumed_permanently: Literal[True] = True


class CombinedServiceCallResult(ImmutableRecord):
    call_id: Identifier
    call_spec_hash: Sha256Digest
    request_hash: Sha256Digest
    semantic_request_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    condition: ConditionName
    outcome: RunOutcome
    model_call_ids: tuple[Identifier, ...]
    model_request_hashes: tuple[Sha256Digest, ...]
    condition_attempt_artifact: RestrictedArtifactPointer
    cumulative_gpu_seconds_before: float = Field(ge=0.0)
    cumulative_gpu_seconds_after: float = Field(gt=0.0)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    checked_at: AwareDatetime
    included_in_intention_to_treat: Literal[True] = True

    @model_validator(mode="after")
    def terminal_bounded_result(self) -> Self:
        if self.outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("combined service result is not terminal")
        if not 1 <= len(self.model_call_ids) <= 2:
            raise ValueError("combined call requires one base and at most one repair")
        if len(set(self.model_call_ids)) != len(self.model_call_ids):
            raise ValueError("combined model-call IDs must be unique")
        if len(self.model_request_hashes) != len(self.model_call_ids):
            raise ValueError("combined rendered-request lineage is incomplete")
        if self.condition_attempt_artifact.object_kind != "condition_attempt":
            raise ValueError("combined result must retain a typed condition attempt")
        if not (
            self.cumulative_gpu_seconds_after > self.cumulative_gpu_seconds_before
            and self.started_at <= self.completed_at <= self.checked_at
        ):
            raise ValueError("combined result GPU/chronology counters are invalid")
        return self


class CombinedModelCallLineage(ImmutableRecord):
    model_call_id: Identifier
    model_call_record_hash: Sha256Digest
    gpu_event_id: Identifier
    gpu_event_record_hash: Sha256Digest
    attempt_id: Identifier
    attempt_kind: Literal["base", "repair"]
    call_role: Literal["query_time", "repair"]
    retry_class: Literal["standard", "short"]
    request_hash: Sha256Digest
    parent_model_call_record_hash: Sha256Digest | None
    allocated_gpu_seconds: float = Field(gt=0.0)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    successful: bool


class CombinedITTRecord(ImmutableRecord):
    ordinal: Annotated[int, Field(ge=1, le=COMBINED_BASE_CALL_COUNT)]
    call_id: Identifier
    call_spec_hash: Sha256Digest
    call_slot_hash: Sha256Digest
    service_result_hash: Sha256Digest
    condition: ConditionName
    outcome: RunOutcome
    model_calls: tuple[CombinedModelCallLineage, ...]
    condition_attempt_hash: Sha256Digest
    repair_claim_hash: Sha256Digest | None
    cumulative_gpu_seconds_before: float = Field(ge=0.0)
    cumulative_gpu_seconds_after: float = Field(gt=0.0)
    completed_at: AwareDatetime
    included_in_intention_to_treat: Literal[True] = True

    @model_validator(mode="after")
    def repair_lineage_matches_call_count(self) -> Self:
        if (len(self.model_calls) == 2) != (self.repair_claim_hash is not None):
            raise ValueError("combined repair call and short-reserve claim differ")
        return self


class CombinedServiceShutdownReceipt(ImmutableRecord):
    service_identity_hash: Sha256Digest
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    shutdown_started_at: AwareDatetime
    stopped_at: AwareDatetime
    cumulative_gpu_seconds_after_shutdown: float = Field(gt=0.0)
    physical_process_exited: Literal[True] = True
    vllm_service_stopped: Literal[True] = True
    gpu_allocation_closed: Literal[True] = True
    model_load_count: Literal[1] = 1

    @model_validator(mode="after")
    def stopped_after_request(self) -> Self:
        if self.stopped_at < self.shutdown_started_at:
            raise ValueError("combined service stop predates shutdown request")
        return self


class CombinedExecutionIndex(ImmutableRecord):
    run_id: Identifier
    manifest_hash: Sha256Digest
    configuration_hash: Sha256Digest
    runtime_binding_hash: Sha256Digest
    upstream_gate_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    activation_slot_hash: Sha256Digest
    ordinary_itt_record_hashes: tuple[Sha256Digest, ...]
    phase5_index_hash: Sha256Digest
    phase5_request_hashes: tuple[Sha256Digest, ...]
    phase5_result_hashes: tuple[Sha256Digest, ...]
    all_call_spec_hashes_in_order: tuple[Sha256Digest, ...]
    repair_claim_hashes: tuple[Sha256Digest, ...]
    short_reserve_slots_consumed_before: Annotated[int, Field(ge=0, le=4)]
    short_reserve_slots_consumed_after: Annotated[int, Field(ge=0, le=4)]
    model_load_event_record_hash: Sha256Digest
    shutdown_receipt_hash: Sha256Digest
    actual_gpu_seconds_before: float = Field(ge=0.0)
    actual_gpu_seconds_after: float = Field(gt=0.0)
    completed_at: AwareDatetime
    base_call_count: Literal[49] = 49
    phase5_call_count: Literal[9] = 9
    ordinary_call_count: Literal[40] = 40
    maximum_concurrency: Literal[1] = 1
    model_load_count: Literal[1] = 1
    vllm_service_stopped: Literal[True] = True
    runtime_namespace: Literal["gold_free"] = "gold_free"

    @model_validator(mode="after")
    def exact_completed_inventory(self) -> Self:
        if len(self.ordinary_itt_record_hashes) != COMBINED_ORDINARY_CALL_COUNT:
            raise ValueError("combined final index requires forty ordinary ITT records")
        if len(self.phase5_request_hashes) != COMBINED_PHASE5_CALL_COUNT:
            raise ValueError("combined final index requires nine Phase 5 requests")
        if len(self.phase5_result_hashes) != COMBINED_PHASE5_CALL_COUNT:
            raise ValueError("combined final index requires nine Phase 5 results")
        if len(self.all_call_spec_hashes_in_order) != COMBINED_BASE_CALL_COUNT:
            raise ValueError("combined final index does not bind all 49 registered calls")
        if any(
            len(values) != len(set(values))
            for values in (
                self.ordinary_itt_record_hashes,
                self.phase5_request_hashes,
                self.phase5_result_hashes,
                self.all_call_spec_hashes_in_order,
                self.repair_claim_hashes,
            )
        ):
            raise ValueError("combined final inventories must be duplicate-free")
        expected_after = self.short_reserve_slots_consumed_before + len(self.repair_claim_hashes)
        if self.short_reserve_slots_consumed_after != expected_after:
            raise ValueError("combined short-reserve count does not reconcile")
        if self.actual_gpu_seconds_after <= self.actual_gpu_seconds_before:
            raise ValueError("combined block did not advance cumulative GPU time")
        return self


class CombinedInputProvider(Protocol):
    """Gold-free semantic materializer; it may not own the model lifecycle."""

    def prepare(
        self,
        call: CombinedCallSpec,
        paraphrase_reveal: ParaphraseRevealReceipt | None,
        prepared_at: datetime,
    ) -> CombinedPreparedCall: ...


class CASCombinedInputProvider:
    """Concrete provider over the exact restricted predecessor CAS objects."""

    def __init__(
        self,
        *,
        runtime: CombinedRuntimeBinding,
        upper_ontology: UpperOntology,
        artifacts: ArtifactStore,
    ) -> None:
        if upper_ontology.content_hash != runtime.upper_ontology_hash:
            raise CombinedProductionError("combined provider upper ontology changed")
        self.runtime = runtime
        self.upper_ontology = upper_ontology
        self.artifacts = artifacts

    def prepare(
        self,
        call: CombinedCallSpec,
        paraphrase_reveal: ParaphraseRevealReceipt | None,
        prepared_at: datetime,
    ) -> CombinedPreparedCall:
        if prepared_at.tzinfo is None or prepared_at.utcoffset() is None:
            raise CombinedProductionError("combined preparation time must be aware")
        packet = _load_public_evidence_packet(
            artifacts=self.artifacts,
            reference=call.source.packet_artifact,
        )
        assert isinstance(packet, EvidencePacket)
        if call.call_class is CombinedCallClass.PARAPHRASE_C2:
            if paraphrase_reveal is None:
                raise CombinedProductionError("paraphrase provider lacks its reveal receipt")
            preparation_hash = call.source.c2_seed1_preparation_hash
            inventory_hash = call.source.c2_seed1_inventory_hash
            query_access_hash = paraphrase_reveal.access_event.content_hash
            threshold = paraphrase_reveal.packet_reuse_event.completed_at
        elif call.call_class in {
            CombinedCallClass.ABLATION_NO_CONTEXT,
            CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC,
            CombinedCallClass.ABLATION_NO_RARE_GUARD,
        }:
            if paraphrase_reveal is not None:
                raise CombinedProductionError("ablation provider received a paraphrase reveal")
            lineage = call.source.preparation_for(call.condition)
            preparation_hash = lineage.preparation_hash
            inventory_hash = lineage.inventory_hash
            query_access_hash = call.source.query_access_event_hash
            threshold = call.source.query_accessed_at
        else:
            raise CombinedProductionError("Phase 5 inputs bypass the ordinary provider")
        if prepared_at <= threshold:
            raise CombinedProductionError("combined preparation did not follow query access")
        identifiers = RuntimeIdentifiers(
            model_id=self.runtime.served_model_name,
            model_revision=self.runtime.model_revision,
            tokenizer_hash=self.runtime.tokenizer_manifest_hash,
            runtime_version=self.runtime.runtime_version,
            prompt_hash=self.runtime.prompt_hashes[call.condition],
            output_schema_hash=self.runtime.output_schema_hashes[call.condition],
            decoding_config_hash=self.runtime.decoding_manifest_hashes[call.condition],
        )
        request = ConstructionRequest(
            request_id=f"{call.call_id}-construction",
            condition=call.condition,
            snapshot_hash=call.source.prequery_stage.snapshot_hash,
            packet=to_model_visible_packet(packet),
            context=call.model_visible_context(),
            upper_ontology=self.upper_ontology,
            budgets=call.source.context.budgets,
            capabilities=(
                ConstructionCapabilities.active_without_temporal_epistemic()
                if call.condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC
                else ConstructionCapabilities.active_construction()
            ),
            runtime=identifiers,
            requested_at=prepared_at,
        )
        baseline = None
        ablated = None
        if call.configuration_delta is not None:
            baseline, ablated = derive_one_switch_fingerprints(
                call=call,
                runtime=self.runtime,
                packet=packet,
            )
        return CombinedPreparedCall(
            call_spec_hash=call.content_hash,
            condition=call.condition,
            request=request,
            preparation_hash=preparation_hash,
            empty_inventory_hash=inventory_hash,
            prequery_barrier_hash=call.source.prequery_barrier_hash,
            query_access_event_hash=query_access_hash,
            packet_artifact=call.source.packet_artifact,
            baseline_fingerprint=baseline,
            ablated_fingerprint=ablated,
            paraphrase_reveal=paraphrase_reveal,
            prepared_at=prepared_at,
        )


class RepairAuthority(Protocol):
    """One-use short-reserve capability passed into an in-flight request."""

    def recover_claim(
        self,
        *,
        request_hash: str,
        base_model_call_id: str,
    ) -> CombinedRepairClaim | None: ...

    def claim(
        self,
        *,
        request_hash: str,
        base_model_call_id: str,
        claimed_at: datetime,
    ) -> CombinedRepairClaim: ...


@runtime_checkable
class CombinedOwnedService(Protocol):
    """Already-running inference surface returned only to the combined owner."""

    def identity(self) -> CombinedServiceIdentity: ...

    @property
    def actual_allocated_service_seconds(self) -> float: ...

    def execute_combined(
        self,
        request: CombinedPreparedCall,
        model_visible_payload: dict[str, object],
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> CombinedServiceCallResult: ...

    def recover_combined(
        self,
        request_hash: str,
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> CombinedServiceCallResult | None: ...

    def execute_phase5_owned(
        self,
        request: C2RegenerationRequest,
        model_visible_payload: dict[str, object],
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> Phase5OwnedServiceResult: ...

    def recover_phase5_owned(
        self,
        request: C2RegenerationRequest,
        repair_authority: RepairAuthority,
        remaining_required_seconds: float,
    ) -> Phase5OwnedServiceResult | None: ...


@runtime_checkable
class CombinedLifecycleOwner(Protocol):
    """The only component allowed to create/adopt/stop the service process."""

    def activate(self, slot: CombinedActivationSlot) -> CombinedOwnedService: ...

    def recover_active(self, activation_slot_hash: str) -> CombinedOwnedService | None: ...

    def shutdown(
        self,
        service: CombinedOwnedService,
        requested_at: datetime,
    ) -> CombinedServiceShutdownReceipt: ...

    def recover_shutdown(
        self,
        service_identity_hash: str,
    ) -> CombinedServiceShutdownReceipt | None: ...


def _require_contained_writable_path(
    target: Path,
    *,
    authority_root: Path,
    label: str,
) -> Path:
    """Reject lexical/realpath escape and every symlink below an explicit root."""

    root = authority_root.absolute()
    candidate = target.absolute()
    if any(ancestor.is_symlink() for ancestor in (root, *root.parents)):
        raise CombinedProductionError(
            f"{label} has a symlinked authorized-root ancestor"
        )
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise CombinedProductionError(
            f"{label} lies outside its authorized writable root"
        ) from error
    current = root
    for part in relative.parts:
        if current.is_symlink():
            raise CombinedProductionError(f"{label} has a symlinked writable ancestor")
        current = current / part
    if current.is_symlink():
        raise CombinedProductionError(f"{label} must not be a symlink")
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise CombinedProductionError(
            f"{label} realpath escapes its authorized writable root"
        ) from error
    return candidate


def _repository_from_frozen_output(
    configuration: CombinedBlockConfiguration,
    output_root: Path,
) -> Path:
    relative = Path(configuration.output_root)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise CombinedProductionError("combined output configuration is not bounded")
    candidate = output_root.absolute()
    repository = candidate
    for _part in relative.parts:
        repository = repository.parent
    if candidate != repository / relative:
        raise CombinedProductionError(
            "combined restricted output differs from its frozen repository path"
        )
    return repository


class CombinedJournal:
    """Small append-only filesystem index layered over the cumulative SQLite/CAS."""

    _TOP_LEVEL_FILES: ClassVar[set[str]] = {
        "manifest.json",
        "activation_slot.json",
        "service_identity.json",
        "phase5_index.json",
        "shutdown_receipt.json",
        "execution_index.json",
    }
    _DIRECTORIES: ClassVar[set[str]] = {
        "calls",
        "paraphrase_reveals",
        "repair_claims",
        "phase5",
    }

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()
        self._assert_no_symlink_chain(self.root)

    @staticmethod
    def _assert_no_symlink_chain(path: Path) -> None:
        current = path.absolute()
        while True:
            if current.is_symlink():
                raise CombinedProductionError(f"symlinked combined journal path: {path}")
            if current.parent == current:
                break
            current = current.parent

    def initialize(self) -> None:
        self._assert_no_symlink_chain(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit()

    def audit(self) -> None:
        if not self.root.exists():
            return
        if not self.root.is_dir() or self.root.is_symlink():
            raise CombinedProductionError("combined journal root is not a regular directory")
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise CombinedProductionError(f"symlinked combined journal entry: {path}")
            relative = path.relative_to(self.root)
            if path.is_file() and path.name.startswith(".") and path.name.endswith(".tmp"):
                target = _verified_atomic_temporary(path)
                assert target is not None
                target_relative = target.relative_to(self.root)
                if len(target_relative.parts) > 2:
                    raise CombinedProductionError(
                        f"atomic journal temporary escaped bounded layout: {relative}"
                    )
                if target.exists() and (
                    not target.is_file() or target.read_bytes() != path.read_bytes()
                ):
                    raise CombinedProductionError(
                        f"atomic journal target/temporary disagree: {relative}"
                    )
                continue
            if relative.parts[0] == "phase5":
                # The delegated executor performs its own strict subtree audit.
                if path.is_file() and path.stat().st_size > 64 * 1024 * 1024:
                    raise CombinedProductionError(
                        f"combined Phase 5 artifact exceeds 64 MiB: {relative}"
                    )
                continue
            if path.is_dir():
                if len(relative.parts) != 1 or relative.name not in self._DIRECTORIES:
                    raise CombinedProductionError(
                        f"unexpected combined journal directory: {relative}"
                    )
                continue
            if len(relative.parts) == 1:
                if relative.name not in self._TOP_LEVEL_FILES:
                    raise CombinedProductionError(f"unexpected combined journal file: {relative}")
            elif len(relative.parts) != 2 or relative.parts[0] not in self._DIRECTORIES:
                raise CombinedProductionError(f"unexpected combined journal entry: {relative}")
            if path.stat().st_size > 64 * 1024 * 1024:
                raise CombinedProductionError(f"combined journal entry exceeds 64 MiB: {relative}")

    def append(self, relative: Path, value: ImmutableRecord) -> None:
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) > 2:
            raise CombinedProductionError("combined journal target is not bounded")
        payload = (value.to_canonical_json() + "\n").encode("utf-8")
        path = self.root / relative
        self._assert_no_symlink_chain(path)
        _append_bytes_atomically(
            path,
            payload,
            label=f"combined artifact {relative.as_posix()}",
        )

    def load(self, relative: Path, model: type[ImmutableRecord]):
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) > 2:
            raise CombinedProductionError("combined journal source is not bounded")
        path = self.root / relative
        self._assert_no_symlink_chain(path)
        if not path.is_file():
            raise CombinedProductionError(f"missing combined journal record: {relative}")
        try:
            return model.model_validate_json(path.read_bytes())
        except Exception as error:
            raise CombinedProductionError(
                f"invalid combined journal record {relative}: {error}"
            ) from error

    def exists(self, relative: Path) -> bool:
        path = self.root / relative
        self._assert_no_symlink_chain(path)
        return path.is_file()

    @staticmethod
    def call_slot_path(call: CombinedCallSpec) -> Path:
        return Path("calls") / f"{call.ordinal:03d}-{call.call_id}.slot.json"

    @staticmethod
    def call_result_path(call: CombinedCallSpec) -> Path:
        return Path("calls") / f"{call.ordinal:03d}-{call.call_id}.result.json"

    @staticmethod
    def call_itt_path(call: CombinedCallSpec) -> Path:
        return Path("calls") / f"{call.ordinal:03d}-{call.call_id}.itt.json"

    @staticmethod
    def reveal_path(call: CombinedCallSpec) -> Path:
        return Path("paraphrase_reveals") / f"{call.ordinal:03d}-{call.call_id}.json"

    @staticmethod
    def reveal_intent_path(call: CombinedCallSpec) -> Path:
        return Path("paraphrase_reveals") / f"{call.ordinal:03d}-{call.call_id}.intent.json"

    @staticmethod
    def repair_claim_path(request_hash: str) -> Path:
        return Path("repair_claims") / f"{request_hash}.json"

    def repair_claims(self) -> tuple[CombinedRepairClaim, ...]:
        directory = self.root / "repair_claims"
        if not directory.exists():
            return ()
        claims = tuple(
            self.load(Path("repair_claims") / path.name, CombinedRepairClaim)
            for path in sorted(directory.iterdir())
            if path.is_file()
        )
        if len({item.request_hash for item in claims}) != len(claims):
            raise CombinedProductionError("combined repair claims are not request-unique")
        return claims


class _RepairAuthority:
    def __init__(
        self,
        *,
        journal: CombinedJournal,
        call: CombinedCallSpec,
        request_hash: str,
        consumed_before: int,
        artifacts: ArtifactStore,
        clock: Callable[[], datetime],
        capacity_check: Callable[[float], None] | None = None,
    ) -> None:
        self.journal = journal
        self.call = call
        self.request_hash = request_hash
        self.consumed_before = consumed_before
        self.artifacts = artifacts
        self.clock = clock
        self.capacity_check = capacity_check

    def recover_claim(
        self,
        *,
        request_hash: str,
        base_model_call_id: str,
    ) -> CombinedRepairClaim | None:
        if request_hash != self.request_hash:
            raise CombinedProductionError("repair authority used for another request")
        path = self.journal.repair_claim_path(request_hash)
        if not self.journal.exists(path):
            return None
        existing = self.journal.load(path, CombinedRepairClaim)
        if (
            existing.call_id != self.call.call_id
            or existing.call_spec_hash != self.call.content_hash
            or existing.base_model_call_id != base_model_call_id
        ):
            raise CombinedProductionError("repair claim identity changed")
        return existing

    def claim(
        self,
        *,
        request_hash: str,
        base_model_call_id: str,
        claimed_at: datetime,
    ) -> CombinedRepairClaim:
        if request_hash != self.request_hash:
            raise CombinedProductionError("repair authority used for another request")
        existing = self.recover_claim(
            request_hash=request_hash,
            base_model_call_id=base_model_call_id,
        )
        if existing is not None:
            return existing
        path = self.journal.repair_claim_path(request_hash)
        if claimed_at.tzinfo is None or claimed_at.utcoffset() is None:
            raise CombinedProductionError("repair claim timestamp must be timezone-aware")
        claims = self.journal.repair_claims()
        global_slot = self.consumed_before + len(claims) + 1
        if global_slot > 4:
            raise CombinedProductionError("registered global short-repair reserve is exhausted")
        if self.capacity_check is None:
            self.artifacts.ledger.require_gpu_capacity(90)
        else:
            self.capacity_check(self.call.repair_watchdog_seconds)
        claim = CombinedRepairClaim(
            claim_id=f"combined-short-repair-{global_slot}",
            call_id=self.call.call_id,
            call_spec_hash=self.call.content_hash,
            request_hash=request_hash,
            base_model_call_id=base_model_call_id,
            global_slot_number=global_slot,
            claimed_at=claimed_at,
        )
        self.journal.append(path, claim)
        return claim


def _artifact_record(
    artifacts: ArtifactStore,
    reference: RestrictedArtifactPointer,
) -> ArtifactRecord:
    try:
        record = artifacts.ledger.get_artifact(reference.artifact_hash)
    except KeyError as error:
        raise CombinedProductionError(
            f"combined restricted artifact is absent: {reference.artifact_hash}"
        ) from error
    if (
        record.content_hash != reference.artifact_hash
        or record.media_type != reference.media_type
        or record.release_class is not StoreReleaseClass.RESTRICTED
    ):
        raise CombinedProductionError("combined artifact metadata changed")
    return record


def _load_condition_attempt(
    artifacts: ArtifactStore,
    reference: RestrictedArtifactPointer,
) -> ConditionAttemptRecord:
    record = _artifact_record(artifacts, reference)
    try:
        raw = artifacts.blobs.read_bytes(record, allow_restricted=True)
        attempt = ConditionAttemptRecord.model_validate_json(raw)
    except Exception as error:
        raise CombinedProductionError(f"invalid retained condition attempt: {error}") from error
    if attempt.content_hash != reference.logical_content_hash:
        raise CombinedProductionError("condition attempt logical hash changed")
    return attempt


def _load_restricted_record(
    *,
    artifacts: ArtifactStore,
    artifact_hash: str,
    logical_hash: str,
    model: type[ImmutableRecord],
    label: str,
) -> ImmutableRecord:
    try:
        record = artifacts.ledger.get_artifact(artifact_hash)
        if (
            record.content_hash != artifact_hash
            or record.release_class is not StoreReleaseClass.RESTRICTED
        ):
            raise CombinedProductionError(f"{label} CAS metadata changed")
        value = model.model_validate_json(
            artifacts.blobs.read_bytes(record, allow_restricted=True)
        )
    except CombinedProductionError:
        raise
    except Exception as error:
        raise CombinedProductionError(
            f"cannot resolve exact {label} CAS object: {error}"
        ) from error
    if value.content_hash != logical_hash:
        raise CombinedProductionError(f"{label} logical hash changed")
    return value


def _load_public_evidence_packet(
    *,
    artifacts: ArtifactStore,
    reference: PublicEvidencePacketPointer,
) -> EvidencePacket:
    """Resolve only the public packet type admitted by the combined source seal."""

    try:
        record = _public_evidence_packet_record(artifacts, reference)
        packet = EvidencePacket.model_validate_json(artifacts.blobs.read_bytes(record))
    except CombinedProductionError:
        raise
    except Exception as error:
        raise CombinedProductionError(
            f"cannot resolve exact public evidence packet CAS object: {error}"
        ) from error
    if (
        packet.content_hash != reference.logical_content_hash
        or packet.release_class is not ReleaseClass.PUBLIC
    ):
        raise CombinedProductionError("public evidence packet logical identity changed")
    return packet


def _public_evidence_packet_record(
    artifacts: ArtifactStore,
    reference: PublicEvidencePacketPointer,
) -> ArtifactRecord:
    try:
        record = artifacts.ledger.get_artifact(reference.artifact_hash)
    except KeyError as error:
        raise CombinedProductionError(
            f"combined public evidence packet is absent: {reference.artifact_hash}"
        ) from error
    if (
        record.content_hash != reference.artifact_hash
        or record.release_class is not StoreReleaseClass.PUBLIC
        or record.media_type != reference.media_type
    ):
        raise CombinedProductionError("evidence packet CAS metadata changed")
    return record


def derive_one_switch_fingerprints(
    *,
    call: CombinedCallSpec,
    runtime: CombinedRuntimeBinding,
    packet: EvidencePacket,
) -> tuple[OneSwitchFingerprint, OneSwitchFingerprint]:
    """Derive the comparison fingerprints from resolved bytes, never provider claims."""

    if call.configuration_delta is None:
        raise CombinedProductionError("non-ablation call has no one-switch fingerprint")
    common = {
        "model_manifest_hash": runtime.model_manifest_hash,
        "model_revision": runtime.model_revision,
        "tokenizer_hash": runtime.tokenizer_manifest_hash,
        "packet_hash": packet.content_hash,
        "ordered_evidence_hash": canonical_sha256(packet.ordered_evidence_ids),
        "horizon_hash": call.source.context.spoiler_horizon.content_hash,
        "upper_ontology_hash": runtime.upper_ontology_hash,
        "budgets_hash": call.source.context.budgets.content_hash,
        "seed_manifest_hash": runtime.seed_manifest_hash,
        "seed_block": call.seed_block,
        "vllm_seed": call.vllm_seed,
        "decoding_family_hash": runtime.decoding_family_hash,
        "maximum_input_tokens": runtime.maximum_input_tokens,
        "maximum_output_tokens": runtime.maximum_output_tokens,
        "repair_attempt_budget": call.maximum_repair_attempts,
        "repair_policy_hash": runtime.repair_policy_hash,
        "validator_hash": runtime.validator_hash,
    }
    baseline = OneSwitchFingerprint.model_validate(common)
    ablated_payload = baseline.model_dump(mode="python", exclude={"content_hash"})
    ablated_payload[call.configuration_delta.switch_name] = (
        call.configuration_delta.ablated_value
    )
    ablated = OneSwitchFingerprint.model_validate(ablated_payload)
    assert_one_switch_only(baseline, ablated, call.configuration_delta)
    return baseline, ablated


def bind_prepared_call_from_cas(
    *,
    call: CombinedCallSpec,
    prepared: CombinedPreparedCall,
    runtime: CombinedRuntimeBinding,
    artifacts: ArtifactStore,
) -> CombinedPreparedCall:
    """Re-resolve evidence/preparation and independently bind semantic constants."""

    packet = _load_public_evidence_packet(
        artifacts=artifacts,
        reference=call.source.packet_artifact,
    )
    if (
        packet.snapshot_hash != call.source.prequery_stage.snapshot_hash
        or prepared.request.packet != to_model_visible_packet(packet)
        or prepared.request.upper_ontology.content_hash != runtime.upper_ontology_hash
    ):
        raise CombinedProductionError(
            "prepared call differs from controller-resolved evidence or upper ontology"
        )

    if call.call_class is CombinedCallClass.PARAPHRASE_C2:
        preparation_hash = call.source.c2_seed1_preparation_hash
        preparation_artifact_hash = call.source.c2_seed1_preparation_artifact_hash
        inventory_hash = call.source.c2_seed1_inventory_hash
        inventory_artifact_hash = call.source.c2_seed1_inventory_artifact_hash
    elif call.call_class in {
        CombinedCallClass.ABLATION_NO_CONTEXT,
        CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC,
        CombinedCallClass.ABLATION_NO_RARE_GUARD,
    }:
        lineage = call.source.preparation_for(call.condition)
        preparation_hash = lineage.preparation_hash
        preparation_artifact_hash = lineage.preparation_artifact_hash
        inventory_hash = lineage.inventory_hash
        inventory_artifact_hash = lineage.inventory_artifact_hash
    else:
        raise CombinedProductionError("Phase 5 calls do not use ordinary CAS preparation")

    preparation = _load_restricted_record(
        artifacts=artifacts,
        artifact_hash=preparation_artifact_hash,
        logical_hash=preparation_hash,
        model=ConditionPreparation,
        label="condition preparation",
    )
    inventory = _load_restricted_record(
        artifacts=artifacts,
        artifact_hash=inventory_artifact_hash,
        logical_hash=inventory_hash,
        model=PreQueryInventory,
        label="pre-query inventory",
    )
    assert isinstance(preparation, ConditionPreparation)
    assert isinstance(inventory, PreQueryInventory)
    if (
        preparation.condition is not call.condition
        or preparation.snapshot_hash != call.source.prequery_stage.snapshot_hash
        or preparation.empty_inventory != inventory
        or inventory.condition is not call.condition
        or inventory.snapshot_hash != call.source.prequery_stage.snapshot_hash
        or preparation.completed_at >= call.source.prequery_barrier_sealed_at
        or prepared.preparation_hash != preparation.content_hash
        or prepared.empty_inventory_hash != inventory.content_hash
    ):
        raise CombinedProductionError("controller-resolved empty preparation lineage changed")

    if call.configuration_delta is None:
        if prepared.baseline_fingerprint is not None or prepared.ablated_fingerprint is not None:
            raise CombinedProductionError("non-ablation prepared call supplied fingerprints")
    else:
        expected = derive_one_switch_fingerprints(call=call, runtime=runtime, packet=packet)
        if (prepared.baseline_fingerprint, prepared.ablated_fingerprint) != expected:
            raise CombinedProductionError(
                "provider one-switch claims differ from controller-derived fingerprints"
            )
    return prepared


def _canonical_artifact_payload(value: ImmutableRecord) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _predicted_artifact_hash(payload: bytes) -> Sha256Digest:
    return hashlib.sha256(payload).hexdigest()


def persist_paraphrase_reveal(
    *,
    call: CombinedCallSpec,
    manifest: CombinedCallManifest,
    journal: CombinedJournal,
    artifacts: ArtifactStore,
    revealed_at: datetime,
) -> ParaphraseRevealReceipt:
    """Persist a real post-barrier query access while reusing only query-free C2 state."""

    if call.call_class is not CombinedCallClass.PARAPHRASE_C2:
        raise CombinedProductionError("only a registered paraphrase call may open a reveal")
    if call.paraphrase_context is None or call.paraphrase_semantic_invariant_hash is None:
        raise CombinedProductionError("paraphrase call lacks its frozen renderer variant")
    receipt_path = journal.reveal_path(call)
    if journal.exists(receipt_path):
        return journal.load(receipt_path, ParaphraseRevealReceipt)
    if revealed_at.tzinfo is None or revealed_at.utcoffset() is None:
        raise CombinedProductionError("paraphrase reveal timestamp must be timezone-aware")
    if revealed_at < manifest.created_at:
        raise CombinedProductionError("paraphrase reveal predates the frozen 49-call manifest")

    context_payload = _canonical_artifact_payload(call.paraphrase_context)
    query_artifact_hash = _predicted_artifact_hash(context_payload)
    access_event = QueryAccessEvent(
        access_event_id=f"combined-paraphrase-access-{call.ordinal:03d}",
        execution_id=call.source.execution_id,
        query_context_hash=call.paraphrase_context.content_hash,
        model_visible_query_hash=call.model_visible_context().content_hash,
        snapshot_hash=call.source.prequery_stage.snapshot_hash,
        stage_manifest_hash=canonical_sha256(
            {
                "manifest_hash": manifest.content_hash,
                "call_spec_hash": call.content_hash,
                "base_query_stage_hash": call.source.query_stage.staging_manifest_hash,
                "renderer_variant_hash": call.paraphrase_context.content_hash,
            }
        ),
        query_artifact_hash=query_artifact_hash,
        prequery_barrier_hash=call.source.prequery_barrier_hash,
        packet_hash=None,
        registered_revealed_at=manifest.created_at,
        accessed_at=revealed_at,
    )
    access_payload = _canonical_artifact_payload(access_event)
    access_artifact_hash = _predicted_artifact_hash(access_payload)
    packet_event = PacketMaterializationEvent(
        materialization_event_id=f"combined-paraphrase-packet-reuse-{call.ordinal:03d}",
        execution_id=call.source.execution_id,
        query_access_event_hash=access_event.content_hash,
        snapshot_hash=call.source.prequery_stage.snapshot_hash,
        packet_hash=call.source.packet_hash,
        retrieval_method="all_admissible",
        retrieval_config_hash=canonical_sha256(
            {
                "operation": "reuse-byte-identical-held-out-packet-v1",
                "source_query_access_event_hash": call.source.query_access_event_hash,
                "source_packet_hash": call.source.packet_hash,
                "source_packet_artifact_hash": call.source.packet_artifact.artifact_hash,
            }
        ),
        started_at=revealed_at,
        completed_at=revealed_at,
    )
    packet_event_payload = _canonical_artifact_payload(packet_event)
    packet_event_artifact_hash = _predicted_artifact_hash(packet_event_payload)
    intent = ParaphraseRevealIntent(
        call_spec_hash=call.content_hash,
        query_payload_artifact_hash=query_artifact_hash,
        access_event=access_event,
        access_event_artifact_hash=access_artifact_hash,
        packet_reuse_event=packet_event,
        packet_reuse_event_artifact_hash=packet_event_artifact_hash,
        packet_artifact_hash=call.source.packet_artifact.artifact_hash,
        created_at=revealed_at,
    )
    intent_path = journal.reveal_intent_path(call)
    if journal.exists(intent_path):
        persisted_intent = journal.load(intent_path, ParaphraseRevealIntent)
        if persisted_intent.call_spec_hash != call.content_hash:
            raise CombinedProductionError("paraphrase reveal intent belongs to another call")
        intent = persisted_intent
        access_event = intent.access_event
        packet_event = intent.packet_reuse_event
        access_payload = _canonical_artifact_payload(access_event)
        packet_event_payload = _canonical_artifact_payload(packet_event)
    else:
        journal.append(intent_path, intent)

    query_artifact = artifacts.put_bytes(
        context_payload,
        media_type="application/vnd.story-projection.query-context+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=intent.created_at,
    )
    access_artifact = artifacts.put_bytes(
        access_payload,
        media_type="application/vnd.story-projection.query-access-event+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=intent.created_at,
    )
    if (
        query_artifact.content_hash != intent.query_payload_artifact_hash
        or access_artifact.content_hash != intent.access_event_artifact_hash
    ):
        raise CombinedProductionError("paraphrase reveal CAS identity changed")
    artifacts.ledger.persist_query_access(
        QueryAccessRecord(
            access_event_hash=access_event.content_hash,
            access_event_id=access_event.access_event_id,
            execution_id=access_event.execution_id,
            query_context_hash=access_event.query_context_hash,
            model_visible_query_hash=access_event.model_visible_query_hash,
            snapshot_hash=access_event.snapshot_hash,
            stage_manifest_hash=access_event.stage_manifest_hash,
            query_artifact_hash=access_event.query_artifact_hash,
            prequery_barrier_hash=access_event.prequery_barrier_hash,
            packet_hash=None,
            query_payload_artifact_hash=query_artifact.content_hash,
            access_event_artifact_hash=access_artifact.content_hash,
            registered_revealed_at=access_event.registered_revealed_at.isoformat(),
            accessed_at=access_event.accessed_at.isoformat(),
            release_class=ReleaseClass.PUBLIC,
        ),
        query_payload_artifact=query_artifact,
        access_event_artifact=access_artifact,
    )
    packet_artifact = _public_evidence_packet_record(
        artifacts,
        call.source.packet_artifact,
    )
    packet_event_artifact = artifacts.put_bytes(
        packet_event_payload,
        media_type="application/vnd.story-projection.packet-materialization-event+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=intent.created_at,
    )
    if packet_event_artifact.content_hash != intent.packet_reuse_event_artifact_hash:
        raise CombinedProductionError("paraphrase packet-reuse CAS identity changed")
    artifacts.ledger.persist_packet_materialization(
        PacketMaterializationRecord(
            materialization_event_hash=packet_event.content_hash,
            materialization_event_id=packet_event.materialization_event_id,
            execution_id=packet_event.execution_id,
            query_access_event_hash=packet_event.query_access_event_hash,
            snapshot_hash=packet_event.snapshot_hash,
            packet_hash=packet_event.packet_hash,
            retrieval_method=packet_event.retrieval_method.value,
            retrieval_config_hash=packet_event.retrieval_config_hash,
            packet_artifact_hash=packet_artifact.content_hash,
            materialization_event_artifact_hash=packet_event_artifact.content_hash,
            started_at=packet_event.started_at.isoformat(),
            completed_at=packet_event.completed_at.isoformat(),
            release_class=ReleaseClass.PUBLIC,
        ),
        packet_artifact=packet_artifact,
        materialization_event_artifact=packet_event_artifact,
    )
    receipt = ParaphraseRevealReceipt(
        call_spec_hash=call.content_hash,
        base_context_hash=call.source.context.content_hash,
        paraphrase_context_hash=call.paraphrase_context.content_hash,
        semantic_invariant_hash=call.paraphrase_semantic_invariant_hash,
        reused_c2_preparation_hash=call.source.c2_seed1_preparation_hash,
        reused_c2_inventory_hash=call.source.c2_seed1_inventory_hash,
        prequery_barrier_hash=call.source.prequery_barrier_hash,
        prequery_barrier_sealed_at=call.source.prequery_barrier_sealed_at,
        access_event=access_event,
        access_event_artifact_hash=access_artifact.content_hash,
        packet_reuse_event=packet_event,
        packet_reuse_event_artifact_hash=packet_event_artifact.content_hash,
        packet_hash=call.source.packet_hash,
        packet_artifact_hash=packet_artifact.content_hash,
        revealed_at=intent.created_at,
    )
    journal.append(receipt_path, receipt)
    return receipt


def validate_prepared_call(
    *,
    call: CombinedCallSpec,
    prepared: CombinedPreparedCall,
    runtime: CombinedRuntimeBinding,
) -> None:
    """Apply all condition, one-switch, evidence, and chronology checks before a slot."""

    if (
        prepared.call_spec_hash != call.content_hash
        or prepared.condition is not call.condition
        or prepared.request.condition is not call.condition
        or prepared.request.snapshot_hash != call.source.prequery_stage.snapshot_hash
        or prepared.request.packet.packet_hash != call.source.packet_hash
        or prepared.packet_artifact != call.source.packet_artifact
        or prepared.prequery_barrier_hash != call.source.prequery_barrier_hash
        or prepared.request.budgets != call.source.context.budgets
        or prepared.request.upper_ontology.content_hash != runtime.upper_ontology_hash
    ):
        raise CombinedProductionError("prepared call changed its frozen source lineage")
    request_runtime = prepared.request.runtime
    if (
        request_runtime.model_id != runtime.served_model_name
        or request_runtime.model_revision != runtime.model_revision
        or request_runtime.tokenizer_hash != runtime.tokenizer_manifest_hash
        or request_runtime.runtime_version != runtime.runtime_version
        or request_runtime.prompt_hash != runtime.prompt_hashes[call.condition]
        or request_runtime.output_schema_hash != runtime.output_schema_hashes[call.condition]
        or request_runtime.decoding_config_hash != runtime.decoding_manifest_hashes[call.condition]
    ):
        raise CombinedProductionError("prepared call changed the frozen model/decoder binding")

    expected_context = call.model_visible_context()
    if prepared.request.context != expected_context:
        raise CombinedProductionError("prepared call changed its condition-visible context")
    if call.call_class is CombinedCallClass.PARAPHRASE_C2:
        reveal = prepared.paraphrase_reveal
        if (
            reveal is None
            or reveal.call_spec_hash != call.content_hash
            or prepared.preparation_hash != call.source.c2_seed1_preparation_hash
            or prepared.empty_inventory_hash != call.source.c2_seed1_inventory_hash
            or prepared.query_access_event_hash != reveal.access_event.content_hash
            or prepared.request.requested_at < reveal.packet_reuse_event.completed_at
            or prepared.baseline_fingerprint is not None
            or prepared.ablated_fingerprint is not None
        ):
            raise CombinedProductionError("paraphrase call lacks its honest post-barrier reveal")
    elif call.call_class in {
        CombinedCallClass.ABLATION_NO_CONTEXT,
        CombinedCallClass.ABLATION_NO_TEMPORAL_EPISTEMIC,
        CombinedCallClass.ABLATION_NO_RARE_GUARD,
    }:
        lineage = call.source.preparation_for(call.condition)
        if (
            prepared.preparation_hash != lineage.preparation_hash
            or prepared.empty_inventory_hash != lineage.inventory_hash
            or prepared.query_access_event_hash != call.source.query_access_event_hash
            or prepared.request.requested_at < call.source.query_accessed_at
            or prepared.paraphrase_reveal is not None
            or prepared.baseline_fingerprint is None
            or prepared.ablated_fingerprint is None
            or call.configuration_delta is None
        ):
            raise CombinedProductionError(
                "ablation call lacks its condition-matching empty preparation"
            )
        assert prepared.baseline_fingerprint is not None
        assert prepared.ablated_fingerprint is not None
        assert call.configuration_delta is not None
        assert_one_switch_only(
            prepared.baseline_fingerprint,
            prepared.ablated_fingerprint,
            call.configuration_delta,
        )
        for fingerprint in (
            prepared.baseline_fingerprint,
            prepared.ablated_fingerprint,
        ):
            if (
                fingerprint.model_manifest_hash != runtime.model_manifest_hash
                or fingerprint.model_revision != runtime.model_revision
                or fingerprint.tokenizer_hash != runtime.tokenizer_manifest_hash
                or fingerprint.packet_hash != call.source.packet_hash
                or fingerprint.ordered_evidence_hash
                != canonical_sha256(prepared.request.packet.ordered_evidence_ids)
                or fingerprint.horizon_hash != call.source.context.spoiler_horizon.content_hash
                or fingerprint.upper_ontology_hash != runtime.upper_ontology_hash
                or fingerprint.budgets_hash != call.source.context.budgets.content_hash
                or fingerprint.seed_manifest_hash != runtime.seed_manifest_hash
                or fingerprint.seed_block != call.seed_block
                or fingerprint.vllm_seed != call.vllm_seed
                or fingerprint.decoding_family_hash != runtime.decoding_family_hash
                or fingerprint.maximum_input_tokens != runtime.maximum_input_tokens
                or fingerprint.maximum_output_tokens != runtime.maximum_output_tokens
                or fingerprint.repair_attempt_budget != call.maximum_repair_attempts
                or fingerprint.repair_policy_hash != runtime.repair_policy_hash
                or fingerprint.validator_hash != runtime.validator_hash
                or fingerprint.scored_schema_hash != SCORED_PROJECTION_SCHEMA_HASH
            ):
                raise CombinedProductionError("ablation one-switch fingerprint changed a constant")
    else:
        raise CombinedProductionError("Phase 5 calls must use the lifecycle-free adapter")
    scan_model_payload(prepared.model_visible_payload())


def verify_service_identity(
    *,
    identity: CombinedServiceIdentity,
    slot: CombinedActivationSlot,
    runtime: CombinedRuntimeBinding,
    artifacts: ArtifactStore,
) -> None:
    if (
        identity.activation_slot_hash != slot.content_hash
        or identity.runtime_binding_hash != runtime.content_hash
        or identity.global_accounting_id != slot.global_accounting_id
        or identity.model_manifest_hash != runtime.model_manifest_hash
        or dict(identity.decoding_manifest_hashes) != dict(runtime.decoding_manifest_hashes)
        or not math.isclose(
            identity.gpu_seconds_before_load,
            slot.gpu_seconds_before,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise CombinedProductionError("combined service identity differs from activation intent")
    events = artifacts.ledger.gpu_events()
    load_events = tuple(item for item in events if item.event_kind is GpuEventKind.MODEL_LOAD)
    by_id = {item.event_id: item for item in load_events}
    try:
        load = by_id[identity.model_load_event_id]
    except KeyError as error:
        raise CombinedProductionError("combined model-load event is absent from SQLite") from error
    observed_hash = _record_hash(load, _GPU_EVENT_FIELDS)
    prior_hashes = set(slot.model_load_event_hashes_before)
    current_hashes = {_record_hash(item, _GPU_EVENT_FIELDS) for item in load_events}
    if (
        observed_hash != identity.model_load_event_record_hash
        or not load.succeeded
        or current_hashes != prior_hashes | {observed_hash}
        or observed_hash in prior_hashes
        or _aware(load.ended_at) > identity.activated_at
        or not math.isclose(
            identity.cumulative_gpu_seconds_after_load,
            identity.gpu_seconds_before_load + load.allocated_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or artifacts.ledger.gpu_summary().total_allocated_seconds + 1e-6
        < identity.cumulative_gpu_seconds_after_load
    ):
        raise CombinedProductionError("combined block did not consume exactly one model load")


def _verify_service_call(
    *,
    call: CombinedCallSpec,
    slot: CombinedCallSlot,
    result: CombinedServiceCallResult,
    identity: CombinedServiceIdentity,
    runtime: CombinedRuntimeBinding,
    artifacts: ArtifactStore,
    repair_claim: CombinedRepairClaim | None,
) -> CombinedITTRecord:
    if (
        slot.ordinal != call.ordinal
        or slot.call_id != call.call_id
        or slot.call_spec_hash != call.content_hash
        or slot.service_identity_hash != identity.content_hash
        or result.call_id != call.call_id
        or result.call_spec_hash != call.content_hash
        or result.request_hash != slot.prepared_call.content_hash
        or result.semantic_request_hash != slot.prepared_call.request.content_hash
        or result.service_identity_hash != identity.content_hash
        or result.condition is not call.condition
        or result.started_at < slot.created_at
        or result.cumulative_gpu_seconds_before < identity.cumulative_gpu_seconds_after_load
        or not math.isclose(
            result.cumulative_gpu_seconds_before,
            slot.cumulative_gpu_seconds_before,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise CombinedProductionError("combined service result changed its durable call slot")
    attempt_record = _load_condition_attempt(artifacts, result.condition_attempt_artifact)
    if (
        attempt_record.condition is not call.condition
        or attempt_record.outcome is not result.outcome
        or attempt_record.release_class is not ReleaseClass.RESTRICTED
    ):
        raise CombinedProductionError("retained condition attempt differs from service result")
    if attempt_record.outcome is RunOutcome.SUCCEEDED:
        projection = attempt_record.projection
        assert projection is not None
        effective_context = call.paraphrase_context or call.source.context
        certificate = projection.construction_certificate
        inventory = projection.pre_query_inventory
        if (
            projection.condition is not call.condition
            or projection.context_hash != effective_context.content_hash
            or projection.packet_hash != call.source.packet_hash
            or projection.snapshot_hash != call.source.prequery_stage.snapshot_hash
            or certificate is None
            or inventory is None
            or certificate.prequery_barrier_hash != call.source.prequery_barrier_hash
            or certificate.query_access_event_hash != slot.prepared_call.query_access_event_hash
            or inventory.content_hash != slot.prepared_call.empty_inventory_hash
            or inventory.recorded_at >= certificate.query_revealed_at
        ):
            raise CombinedProductionError(
                "successful combined projection has invalid construction lineage"
            )

    events = {item.event_id: item for item in artifacts.ledger.gpu_events()}
    lineages: list[CombinedModelCallLineage] = []
    base_attempt_id: str | None = None
    base_call_hash: str | None = None
    previous_completed_at: datetime | None = None
    for index, model_call_id in enumerate(result.model_call_ids):
        try:
            model_call = artifacts.ledger.get_model_call(model_call_id)
            event = events[model_call.gpu_event_id]
            attempt = artifacts.ledger.attempt_lineage(model_call.attempt_id)[-1]
        except KeyError as error:
            raise CombinedProductionError(
                "combined result omitted SQLite model/GPU lineage"
            ) from error
        is_base = index == 0
        expected_attempt = AttemptKind.BASE if is_base else AttemptKind.REPAIR
        expected_role = ModelCallRole.QUERY_TIME if is_base else ModelCallRole.REPAIR
        expected_retry = RetryClass.STANDARD if is_base else RetryClass.SHORT
        expected_decoding = runtime.decoding_manifest_hashes[call.condition]
        if (
            attempt.attempt_kind is not expected_attempt
            or model_call.backend is not ModelBackend.VLLM_GPU
            or model_call.call_role is not expected_role
            or model_call.retry_class is not expected_retry
            or model_call.model_manifest_hash != runtime.model_manifest_hash
            or model_call.decoding_manifest_hash != expected_decoding
            or model_call.job_id != attempt.job_id
            or event.job_id != attempt.job_id
            or event.attempt_id != attempt.attempt_id
            or event.allocated_microseconds != model_call.allocated_gpu_microseconds
            or event.succeeded is not model_call.successful
            or attempt.seed != call.vllm_seed
            or attempt.input_hash != result.model_request_hashes[index]
            or model_call.request_hash != result.model_request_hashes[index]
            or (not is_base and attempt.parent_attempt_id != base_attempt_id)
        ):
            raise CombinedProductionError("combined SQLite attempt/model/GPU lineage changed")
        is_final = index == len(result.model_call_ids) - 1
        if is_final and result.outcome is RunOutcome.TIMED_OUT:
            expected_event_kind = GpuEventKind.TIMEOUT
        elif is_final and result.outcome in {RunOutcome.FAILED, RunOutcome.INTERRUPTED}:
            expected_event_kind = GpuEventKind.FAILURE
        elif is_base:
            expected_event_kind = GpuEventKind.INFERENCE
        else:
            expected_event_kind = GpuEventKind.REPAIR
        if event.event_kind is not expected_event_kind:
            raise CombinedProductionError("combined terminal outcome has the wrong GPU event kind")
        start = _aware(event.started_at)
        end = _aware(event.ended_at)
        if (
            start < slot.prepared_call.request.requested_at
            or start < result.started_at
            or end < start
            or end > result.completed_at
            or (previous_completed_at is not None and start < previous_completed_at)
        ):
            raise CombinedProductionError("combined GPU event predates query-time request")
        lineage = CombinedModelCallLineage(
            model_call_id=model_call.model_call_id,
            model_call_record_hash=_record_hash(model_call, _MODEL_CALL_FIELDS),
            gpu_event_id=event.event_id,
            gpu_event_record_hash=_record_hash(event, _GPU_EVENT_FIELDS),
            attempt_id=attempt.attempt_id,
            attempt_kind="base" if is_base else "repair",
            call_role="query_time" if is_base else "repair",
            retry_class="standard" if is_base else "short",
            request_hash=model_call.request_hash,
            parent_model_call_record_hash=None if is_base else base_call_hash,
            allocated_gpu_seconds=model_call.allocated_gpu_microseconds / 1_000_000,
            started_at=start,
            completed_at=end,
            successful=model_call.successful,
        )
        lineages.append(lineage)
        if is_base:
            base_attempt_id = attempt.attempt_id
            base_call_hash = lineage.model_call_record_hash
        previous_completed_at = end
    expected_final_success = result.outcome in {RunOutcome.SUCCEEDED, RunOutcome.INVALID}
    if lineages[-1].successful is not expected_final_success:
        raise CombinedProductionError("terminal outcome differs from final model-call status")
    if (len(lineages) == 2) != (repair_claim is not None):
        raise CombinedProductionError("repair inference did not consume exactly one short slot")
    if repair_claim is not None and (
        repair_claim.request_hash != slot.prepared_call.content_hash
        or repair_claim.call_spec_hash != call.content_hash
        or repair_claim.base_model_call_id != lineages[0].model_call_id
    ):
        raise CombinedProductionError("repair claim differs from the base inference")
    allocated = sum(item.allocated_gpu_seconds for item in lineages)
    delta = result.cumulative_gpu_seconds_after - result.cumulative_gpu_seconds_before
    if not math.isclose(allocated, delta, rel_tol=0.0, abs_tol=1e-6):
        raise CombinedProductionError("model-call rows do not explain cumulative GPU delta")
    if artifacts.ledger.gpu_summary().total_allocated_seconds + 1e-6 < (
        result.cumulative_gpu_seconds_after
    ):
        raise CombinedProductionError("combined result exceeds the cumulative GPU ledger")
    return CombinedITTRecord(
        ordinal=call.ordinal,
        call_id=call.call_id,
        call_spec_hash=call.content_hash,
        call_slot_hash=slot.content_hash,
        service_result_hash=result.content_hash,
        condition=call.condition,
        outcome=result.outcome,
        model_calls=tuple(lineages),
        condition_attempt_hash=attempt_record.content_hash,
        repair_claim_hash=None if repair_claim is None else repair_claim.content_hash,
        cumulative_gpu_seconds_before=result.cumulative_gpu_seconds_before,
        cumulative_gpu_seconds_after=result.cumulative_gpu_seconds_after,
        completed_at=result.completed_at,
    )


class _CombinedPhase5ServiceView:
    """Lifecycle-free Phase 5 view over the combined owner's live service."""

    def __init__(
        self,
        *,
        service: CombinedOwnedService,
        identity: CombinedServiceIdentity,
        configuration: CombinedBlockConfiguration,
        manifest: CombinedCallManifest,
        upstream_gate: CombinedUpstreamGate,
        calls: tuple[CombinedCallSpec, ...],
        journal: CombinedJournal,
        artifacts: ArtifactStore,
        short_reserve_consumed_before: int,
        clock: Callable[[], datetime],
    ) -> None:
        self._service = service
        self._identity = identity
        self._configuration = configuration
        self._manifest = manifest
        self._upstream_gate = upstream_gate
        self._calls = {item.phase5_episode_id: item for item in calls}
        self._journal = journal
        self._artifacts = artifacts
        self._short_reserve_consumed_before = short_reserve_consumed_before
        self._clock = clock
        if len(self._calls) != COMBINED_PHASE5_CALL_COUNT or None in self._calls:
            raise CombinedProductionError("combined Phase 5 lease requires nine exact episodes")

    def identity(self) -> Phase5OwnedServiceIdentity:
        return self._identity.phase5_identity

    def execute_feedback(
        self,
        request: C2RegenerationRequest,
        model_visible_payload: dict[str, object],
    ) -> Phase5OwnedServiceResult:
        call = self._calls.get(request.episode_id)
        if (
            call is None
            or call.phase5_episode_kind is not request.kind
            or request.seed != call.frozen_seed
        ):
            raise CombinedProductionError("Phase 5 request is absent from the 49-call manifest")
        scan_model_payload(model_visible_payload)
        remaining = _remaining_mandatory_p95_after_call(
            call=call,
            configuration=self._configuration,
            manifest=self._manifest,
            upstream_gate=self._upstream_gate,
            journal=self._journal,
        )
        _require_combined_capacity(
            artifacts=self._artifacts,
            configuration=self._configuration,
            next_watchdog_seconds=call.watchdog_seconds,
            remaining_mandatory_p95_seconds=remaining,
            service=self._service,
        )
        authority = _RepairAuthority(
            journal=self._journal,
            call=call,
            request_hash=request.content_hash,
            consumed_before=self._short_reserve_consumed_before,
            artifacts=self._artifacts,
            clock=self._clock,
            capacity_check=lambda watchdog: _require_combined_capacity(
                artifacts=self._artifacts,
                configuration=self._configuration,
                next_watchdog_seconds=watchdog,
                remaining_mandatory_p95_seconds=(
                    _remaining_mandatory_p95_after_call(
                        call=call,
                        configuration=self._configuration,
                        manifest=self._manifest,
                        upstream_gate=self._upstream_gate,
                        journal=self._journal,
                    )
                ),
                service=self._service,
            ),
        )
        result = self._service.execute_phase5_owned(
            request,
            model_visible_payload,
            authority,
            remaining,
        )
        self._validate_result(call, request.content_hash, result)
        return result

    def recover_feedback(
        self,
        request: C2RegenerationRequest,
    ) -> Phase5OwnedServiceResult | None:
        call = self._calls.get(request.episode_id)
        if (
            call is None
            or call.phase5_episode_kind is not request.kind
            or request.seed != call.frozen_seed
        ):
            raise CombinedProductionError("Phase 5 request is absent from the 49-call manifest")
        request_hash = request.content_hash
        remaining = _remaining_mandatory_p95_after_call(
            call=call,
            configuration=self._configuration,
            manifest=self._manifest,
            upstream_gate=self._upstream_gate,
            journal=self._journal,
        )
        authority = _RepairAuthority(
            journal=self._journal,
            call=call,
            request_hash=request_hash,
            consumed_before=self._short_reserve_consumed_before,
            artifacts=self._artifacts,
            clock=self._clock,
            capacity_check=lambda watchdog: _require_combined_capacity(
                artifacts=self._artifacts,
                configuration=self._configuration,
                next_watchdog_seconds=watchdog,
                remaining_mandatory_p95_seconds=(
                    _remaining_mandatory_p95_after_call(
                        call=call,
                        configuration=self._configuration,
                        manifest=self._manifest,
                        upstream_gate=self._upstream_gate,
                        journal=self._journal,
                    )
                ),
                service=self._service,
            ),
        )
        result = self._service.recover_phase5_owned(request, authority, remaining)
        if result is None:
            return None
        if result.request_hash != request_hash:
            raise CombinedProductionError("recovered Phase 5 result belongs to another request")
        claim_path = self._journal.repair_claim_path(request_hash)
        has_claim = self._journal.exists(claim_path)
        if (len(result.model_call_ids) == 2) != has_claim:
            raise CombinedProductionError("recovered Phase 5 repair lacks its reserve claim")
        return result

    def _validate_result(
        self,
        call: CombinedCallSpec,
        request_hash: str,
        result: Phase5OwnedServiceResult,
    ) -> None:
        if (
            result.request_hash != request_hash
            or result.service_identity_hash != self._identity.phase5_identity.content_hash
        ):
            raise CombinedProductionError("Phase 5 result changed service/request identity")
        claim_path = self._journal.repair_claim_path(request_hash)
        has_claim = self._journal.exists(claim_path)
        if (len(result.model_call_ids) == 2) != has_claim:
            raise CombinedProductionError("Phase 5 repair did not consume one short-reserve slot")
        if has_claim:
            claim = self._journal.load(claim_path, CombinedRepairClaim)
            if (
                claim.call_spec_hash != call.content_hash
                or claim.base_model_call_id != result.model_call_ids[0]
            ):
                raise CombinedProductionError("Phase 5 repair claim changed its call lineage")


def _validate_phase5_inventory(
    *,
    manifest: CombinedCallManifest,
    inputs: Phase5ExecutionInputManifest,
) -> tuple[CombinedCallSpec, ...]:
    calls = tuple(
        item
        for item in manifest.calls
        if item.call_class
        in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }
    )
    expected = tuple((item.phase5_episode_id, item.phase5_episode_kind) for item in calls)
    observed_ids = tuple(item.episode_id for item in inputs.scripted_inputs) + tuple(
        item.episode_id for item in inputs.researcher_trace_inputs
    )
    # The typed input collections establish the first six/last three kinds.
    if tuple(item[0] for item in expected) != observed_ids:
        raise CombinedProductionError("Phase 5 input episode order differs from combined manifest")
    if inputs.content_hash != manifest.phase5_input_manifest_hash:
        raise CombinedProductionError("Phase 5 inputs differ from the frozen combined manifest")
    return calls


def validate_combined_execution_inputs(
    *,
    configuration: CombinedBlockConfiguration,
    manifest: CombinedCallManifest,
    runtime: CombinedRuntimeBinding,
    upstream_gate: CombinedUpstreamGate,
    phase5_inputs: Phase5ExecutionInputManifest,
    phase5_protocol: FeedbackProtocolConfiguration,
    phase5_prerequisites: PrimaryHeldOutResultsGate,
) -> tuple[CombinedCallSpec, ...]:
    """Replay every read-only admission check before lifecycle ownership."""

    gate = upstream_gate
    if (
        gate.content_hash != manifest.upstream_gate_hash
        or not gate.development_gate_passed
        or not gate.held_out_runtime_closed
        or not gate.all_held_out_failures_included_in_itt
        or gate.phase5_input_manifest_hash != phase5_inputs.content_hash
        or phase5_inputs.primary_results_gate_hash != phase5_prerequisites.content_hash
        or phase5_inputs.final_reviewed_seal_hash != gate.final_reviewed_seal_hash
        or manifest.configuration_hash != configuration.content_hash
        or manifest.runtime_binding_hash != runtime.content_hash
        or manifest.phase5_input_manifest_hash != phase5_inputs.content_hash
        or runtime.model_repository != configuration.selected_model_repository
        or runtime.model_revision != configuration.selected_model_revision
        or runtime.served_model_name != configuration.served_model_name
        or manifest.created_at < gate.verified_at
    ):
        raise CombinedProductionError(
            "combined execution requires accepted development and complete "
            "held-out primary artifacts"
        )
    if any(
        call.source.held_out_call_manifest_hash != gate.held_out_call_manifest_hash
        or call.source.held_out_execution_manifest_hash != gate.held_out_execution_manifest_hash
        or call.source.held_out_scorer_bridge_hash != gate.held_out_scorer_bridge_hash
        or call.source.final_reviewed_seal_hash != gate.final_reviewed_seal_hash
        or call.source.prequery_barrier_hash != gate.held_out_prequery_barrier_hash
        or call.source.ablation_prequery_registry_hash != gate.ablation_prequery_registry_hash
        for call in manifest.calls
    ):
        raise CombinedProductionError("combined manifest source lineage changed after admission")
    if gate.remaining_registered_p95_seconds_before_block < (
        configuration.combined_with_load_forecast_seconds
    ):
        raise CombinedProductionError("remaining registered schedule cannot admit combined block")
    expected_total = (
        gate.actual_allocated_gpu_seconds_before_block
        + gate.remaining_registered_p95_seconds_before_block
    )
    if (
        expected_total > configuration.scheduled_limit_seconds
        or gate.actual_allocated_gpu_seconds_before_block
        + configuration.combined_with_load_forecast_seconds
        >= configuration.hard_limit_seconds
    ):
        raise CombinedProductionError("combined block would violate the GPU schedule")
    validate_phase5_inputs(phase5_inputs, phase5_protocol)
    return _validate_phase5_inventory(manifest=manifest, inputs=phase5_inputs)


def _repair_claim_for(
    journal: CombinedJournal,
    request_hash: str,
) -> CombinedRepairClaim | None:
    path = journal.repair_claim_path(request_hash)
    return journal.load(path, CombinedRepairClaim) if journal.exists(path) else None


def _live_allocated_gpu_seconds(
    artifacts: ArtifactStore,
    service: CombinedOwnedService | None = None,
) -> float:
    ledger_seconds = artifacts.ledger.gpu_summary().total_allocated_seconds
    if service is None:
        return ledger_seconds
    try:
        service_seconds = float(service.actual_allocated_service_seconds)
    except (AttributeError, TypeError, ValueError) as error:
        raise CombinedProductionError(
            "combined service does not expose actual live allocation"
        ) from error
    if not math.isfinite(service_seconds) or service_seconds + 1e-6 < ledger_seconds:
        raise CombinedProductionError("combined live GPU allocation regressed below SQLite")
    return max(ledger_seconds, service_seconds)


def _registered_tail_p95_seconds(
    configuration: CombinedBlockConfiguration,
    upstream_gate: CombinedUpstreamGate,
) -> float:
    tail = (
        upstream_gate.remaining_registered_p95_seconds_before_block
        - configuration.combined_with_load_forecast_seconds
    )
    if tail < -1e-6:
        raise CombinedProductionError("combined block exceeds its admitted remaining forecast")
    return max(0.0, tail)


def _call_has_terminal_record(journal: CombinedJournal, call: CombinedCallSpec) -> bool:
    if call.call_class in {
        CombinedCallClass.SCRIPTED_FEEDBACK_C2,
        CombinedCallClass.RESEARCHER_TRACE_C2,
    }:
        assert call.phase5_episode_id is not None
        result = journal.root / "phase5" / "results" / f"{call.phase5_episode_id}.json"
        CombinedJournal._assert_no_symlink_chain(result)
        return result.is_file()
    return journal.exists(journal.call_itt_path(call))


def _remaining_mandatory_p95_after_call(
    *,
    call: CombinedCallSpec,
    configuration: CombinedBlockConfiguration,
    manifest: CombinedCallManifest,
    upstream_gate: CombinedUpstreamGate,
    journal: CombinedJournal,
) -> float:
    future = sum(
        item.p95_seconds
        for item in manifest.calls
        if item.ordinal > call.ordinal and not _call_has_terminal_record(journal, item)
    )
    return _registered_tail_p95_seconds(configuration, upstream_gate) + future


def _require_combined_capacity(
    *,
    artifacts: ArtifactStore,
    configuration: CombinedBlockConfiguration,
    next_watchdog_seconds: float,
    remaining_mandatory_p95_seconds: float,
    service: CombinedOwnedService | None = None,
) -> float:
    """Gate every allocation against live time, mandatory p95, and shutdown room."""

    if (
        not math.isfinite(next_watchdog_seconds)
        or next_watchdog_seconds <= 0
        or not math.isfinite(remaining_mandatory_p95_seconds)
        or remaining_mandatory_p95_seconds < 0
    ):
        raise CombinedProductionError("combined GPU capacity inputs are invalid")
    actual = _live_allocated_gpu_seconds(artifacts, service)
    if (
        actual
        + next_watchdog_seconds
        + configuration.protected_shutdown_margin_seconds
        >= configuration.hard_limit_seconds
    ):
        raise CombinedProductionError(
            "next combined allocation would consume the protected hard-stop margin"
        )
    if (
        actual + next_watchdog_seconds + remaining_mandatory_p95_seconds
        > configuration.scheduled_limit_seconds
    ):
        raise CombinedProductionError(
            "next combined allocation plus mandatory p95 exceeds the scheduled limit"
        )
    artifacts.ledger.require_gpu_capacity(
        next_watchdog_seconds + configuration.protected_shutdown_margin_seconds,
        hard_limit_seconds=configuration.hard_limit_seconds,
    )
    return actual


def _require_combined_shutdown_margin(
    *,
    artifacts: ArtifactStore,
    configuration: CombinedBlockConfiguration,
    service: CombinedOwnedService,
) -> float:
    actual = _live_allocated_gpu_seconds(artifacts, service)
    if (
        actual + configuration.protected_shutdown_margin_seconds
        >= configuration.hard_limit_seconds
    ):
        raise CombinedProductionError("combined service reached its protected shutdown margin")
    return actual


def _ordinary_calls(manifest: CombinedCallManifest) -> tuple[CombinedCallSpec, ...]:
    return tuple(
        item
        for item in manifest.calls
        if item.call_class
        not in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }
    )


def _execute_or_recover_ordinary(
    *,
    call: CombinedCallSpec,
    manifest: CombinedCallManifest,
    configuration: CombinedBlockConfiguration,
    upstream_gate: CombinedUpstreamGate,
    runtime: CombinedRuntimeBinding,
    service: CombinedOwnedService,
    identity: CombinedServiceIdentity,
    provider: CombinedInputProvider,
    journal: CombinedJournal,
    artifacts: ArtifactStore,
    short_reserve_consumed_before: int,
    clock: Callable[[], datetime],
) -> CombinedITTRecord:
    slot_path = journal.call_slot_path(call)
    result_path = journal.call_result_path(call)
    itt_path = journal.call_itt_path(call)

    def recovery_resources(
        slot: CombinedCallSlot,
    ) -> tuple[float, _RepairAuthority]:
        remaining = _remaining_mandatory_p95_after_call(
            call=call,
            configuration=configuration,
            manifest=manifest,
            upstream_gate=upstream_gate,
            journal=journal,
        )
        authority = _RepairAuthority(
            journal=journal,
            call=call,
            request_hash=slot.prepared_call.content_hash,
            consumed_before=short_reserve_consumed_before,
            artifacts=artifacts,
            clock=clock,
            capacity_check=lambda watchdog: _require_combined_capacity(
                artifacts=artifacts,
                configuration=configuration,
                next_watchdog_seconds=watchdog,
                remaining_mandatory_p95_seconds=(
                    _remaining_mandatory_p95_after_call(
                        call=call,
                        configuration=configuration,
                        manifest=manifest,
                        upstream_gate=upstream_gate,
                        journal=journal,
                    )
                ),
                service=service,
            ),
        )
        return remaining, authority

    def issue_slotted_call(slot: CombinedCallSlot) -> CombinedServiceCallResult:
        remaining, authority = recovery_resources(slot)
        _require_combined_capacity(
            artifacts=artifacts,
            configuration=configuration,
            next_watchdog_seconds=call.watchdog_seconds,
            remaining_mandatory_p95_seconds=remaining,
            service=service,
        )
        payload = slot.prepared_call.model_visible_payload()
        scan_model_payload(payload)
        return service.execute_combined(
            slot.prepared_call,
            payload,
            authority,
            remaining,
        )

    if journal.exists(itt_path):
        slot = journal.load(slot_path, CombinedCallSlot)
        result = journal.load(result_path, CombinedServiceCallResult)
        itt = journal.load(itt_path, CombinedITTRecord)
        if slot.manifest_hash != manifest.content_hash:
            raise CombinedProductionError("resumed call slot binds another combined manifest")
        bind_prepared_call_from_cas(
            call=call,
            prepared=slot.prepared_call,
            runtime=runtime,
            artifacts=artifacts,
        )
        validate_prepared_call(call=call, prepared=slot.prepared_call, runtime=runtime)
        verified = _verify_service_call(
            call=call,
            slot=slot,
            result=result,
            identity=identity,
            runtime=runtime,
            artifacts=artifacts,
            repair_claim=_repair_claim_for(journal, slot.prepared_call.content_hash),
        )
        if verified != itt:
            raise CombinedProductionError("resumed ordinary ITT record changed")
        return itt

    if journal.exists(slot_path):
        slot = journal.load(slot_path, CombinedCallSlot)
        if slot.manifest_hash != manifest.content_hash:
            raise CombinedProductionError("resumed call slot binds another combined manifest")
        bind_prepared_call_from_cas(
            call=call,
            prepared=slot.prepared_call,
            runtime=runtime,
            artifacts=artifacts,
        )
        validate_prepared_call(call=call, prepared=slot.prepared_call, runtime=runtime)
        if journal.exists(result_path):
            result = journal.load(result_path, CombinedServiceCallResult)
        else:
            remaining, authority = recovery_resources(slot)
            result = service.recover_combined(
                slot.prepared_call.content_hash,
                authority,
                remaining,
            )
            if result is None:
                # A durable outer slot precedes the semantic/GPU intent.  None
                # now has one narrow meaning: the semantic executor proved no
                # intent or terminal trace exists, so this is still the first
                # physical request.  An inflight intent raises instead.
                result = issue_slotted_call(slot)
            journal.append(result_path, result)
    else:
        reveal = None
        if call.call_class is CombinedCallClass.PARAPHRASE_C2:
            reveal = persist_paraphrase_reveal(
                call=call,
                manifest=manifest,
                journal=journal,
                artifacts=artifacts,
                revealed_at=clock(),
            )
        prepared_at = clock()
        prepared = provider.prepare(call, reveal, prepared_at)
        bind_prepared_call_from_cas(
            call=call,
            prepared=prepared,
            runtime=runtime,
            artifacts=artifacts,
        )
        validate_prepared_call(call=call, prepared=prepared, runtime=runtime)
        # Capacity is intentionally evaluated against live service allocation,
        # including between-event uptime.  Per-call cumulative boundaries remain
        # SQLite boundaries so their deltas can be explained exactly by the one
        # base and optional repair GPU-event rows.  Service overhead is reconciled
        # once, at physical shutdown.
        before = artifacts.ledger.gpu_summary().total_allocated_seconds
        slot = CombinedCallSlot(
            ordinal=call.ordinal,
            call_id=call.call_id,
            call_spec_hash=call.content_hash,
            manifest_hash=manifest.content_hash,
            service_identity_hash=identity.content_hash,
            prepared_call=prepared,
            cumulative_gpu_seconds_before=before,
            created_at=clock(),
        )
        journal.append(slot_path, slot)
        result = issue_slotted_call(slot)
        journal.append(result_path, result)

    repair_claim = _repair_claim_for(journal, slot.prepared_call.content_hash)
    itt = _verify_service_call(
        call=call,
        slot=slot,
        result=result,
        identity=identity,
        runtime=runtime,
        artifacts=artifacts,
        repair_claim=repair_claim,
    )
    journal.append(itt_path, itt)
    return itt


class CombinedPublicSummary(ImmutableRecord):
    run_id: Identifier
    manifest_hash: Sha256Digest
    execution_index_hash: Sha256Digest
    model_manifest_hash: Sha256Digest
    model_revision: str
    call_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    physical_gpu_request_count: int = Field(ge=49, le=53)
    repair_request_count: Annotated[int, Field(ge=0, le=4)]
    allocated_gpu_seconds: float = Field(gt=0.0)
    allocated_gpu_hours: float = Field(gt=0.0)
    model_load_count: Literal[1] = 1
    maximum_concurrency: Literal[1] = 1
    vllm_service_stopped: Literal[True] = True
    copyrighted_text_included: Literal[False] = False
    scorer_gold_included: Literal[False] = False
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC
    generated_at: AwareDatetime

    @model_validator(mode="after")
    def totals_reconcile(self) -> Self:
        if sum(self.call_counts.values()) != COMBINED_BASE_CALL_COUNT:
            raise ValueError("public combined call counts do not total 49")
        if sum(self.outcome_counts.values()) != COMBINED_BASE_CALL_COUNT:
            raise ValueError("public combined outcome counts do not total 49")
        if self.physical_gpu_request_count != self.base_call_count + self.repair_request_count:
            raise ValueError("public physical request count does not reconcile")
        if not math.isclose(
            self.allocated_gpu_hours,
            self.allocated_gpu_seconds / 3600,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("public combined GPU hours do not reconcile")
        return self

    @property
    def base_call_count(self) -> int:
        return sum(self.call_counts.values())


def _phase5_result_records(
    *,
    phase5_root: Path,
    phase5_calls: tuple[CombinedCallSpec, ...],
) -> tuple[C2RegenerationResult, ...]:
    records: list[C2RegenerationResult] = []
    for call in phase5_calls:
        assert call.phase5_episode_id is not None
        path = phase5_root / "results" / f"{call.phase5_episode_id}.json"
        if path.is_symlink() or not path.is_file():
            raise CombinedProductionError("combined Phase 5 result inventory is incomplete")
        try:
            records.append(C2RegenerationResult.model_validate_json(path.read_bytes()))
        except Exception as error:
            raise CombinedProductionError(f"invalid combined Phase 5 result: {error}") from error
    return tuple(records)


def _validate_phase5_claims(
    *,
    journal: CombinedJournal,
    phase5_calls: tuple[CombinedCallSpec, ...],
    phase5_index: Phase5JournalIndex,
) -> tuple[C2RegenerationResult, ...]:
    results = _phase5_result_records(
        phase5_root=journal.root / "phase5",
        phase5_calls=phase5_calls,
    )
    if tuple(item.request_hash for item in results) != phase5_index.c2_request_hashes:
        raise CombinedProductionError("Phase 5 result order differs from its final index")
    repair_count = 0
    for call, result in zip(phase5_calls, results, strict=True):
        claim = _repair_claim_for(journal, result.request_hash)
        has_repair = len(result.receipt.ledger_calls) == 2
        if has_repair != (claim is not None):
            raise CombinedProductionError("Phase 5 repair receipt differs from reserve claims")
        if claim is not None:
            repair_count += 1
            if (
                claim.call_spec_hash != call.content_hash
                or claim.base_model_call_id != result.receipt.ledger_calls[0].model_call_id
            ):
                raise CombinedProductionError("Phase 5 repair claim binds another call")
    if repair_count != phase5_index.repair_gpu_request_count:
        raise CombinedProductionError("Phase 5 repair count differs from its final index")
    return results


def _all_required_work_exists(
    journal: CombinedJournal,
    manifest: CombinedCallManifest,
) -> bool:
    return all(
        journal.exists(path)
        for call in _ordinary_calls(manifest)
        for path in (
            journal.call_slot_path(call),
            journal.call_result_path(call),
            journal.call_itt_path(call),
        )
    ) and journal.exists(Path("phase5_index.json"))


def _load_verified_ordinary_records(
    *,
    manifest: CombinedCallManifest,
    runtime: CombinedRuntimeBinding,
    identity: CombinedServiceIdentity,
    journal: CombinedJournal,
    artifacts: ArtifactStore,
) -> tuple[CombinedITTRecord, ...]:
    records: list[CombinedITTRecord] = []
    for call in _ordinary_calls(manifest):
        slot = journal.load(journal.call_slot_path(call), CombinedCallSlot)
        result = journal.load(journal.call_result_path(call), CombinedServiceCallResult)
        retained = journal.load(journal.call_itt_path(call), CombinedITTRecord)
        if slot.manifest_hash != manifest.content_hash:
            raise CombinedProductionError("ordinary call slot binds another combined manifest")
        bind_prepared_call_from_cas(
            call=call,
            prepared=slot.prepared_call,
            runtime=runtime,
            artifacts=artifacts,
        )
        validate_prepared_call(call=call, prepared=slot.prepared_call, runtime=runtime)
        verified = _verify_service_call(
            call=call,
            slot=slot,
            result=result,
            identity=identity,
            runtime=runtime,
            artifacts=artifacts,
            repair_claim=_repair_claim_for(journal, slot.prepared_call.content_hash),
        )
        if verified != retained:
            raise CombinedProductionError("retained ordinary ITT record changed")
        records.append(retained)
    return tuple(records)


def _validate_sequential_call_chronology(
    *,
    manifest: CombinedCallManifest,
    ordinary_records: tuple[CombinedITTRecord, ...],
    phase5_results: tuple[C2RegenerationResult, ...],
) -> datetime:
    ordinary_by_ordinal = {item.ordinal: item for item in ordinary_records}
    # Instruction hashes are not manifest identifiers.  Bind the Phase 5 records
    # positionally to the already-validated exact episode inventory.
    phase5_iter = iter(phase5_results)
    previous_end: datetime | None = None
    previous_ordinary_after: float | None = None
    seen_phase5 = 0
    for call in manifest.calls:
        if call.call_class in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }:
            try:
                result = next(phase5_iter)
            except StopIteration as error:
                raise CombinedProductionError(
                    "combined Phase 5 chronology is incomplete"
                ) from error
            calls = result.receipt.ledger_calls
            started_at = calls[0].started_at
            completed_at = calls[-1].completed_at
            seen_phase5 += 1
        else:
            try:
                record = ordinary_by_ordinal[call.ordinal]
            except KeyError as error:
                raise CombinedProductionError(
                    "combined ordinary chronology is incomplete"
                ) from error
            started_at = record.model_calls[0].started_at
            completed_at = record.model_calls[-1].completed_at
            if (
                previous_ordinary_after is not None
                and record.cumulative_gpu_seconds_before + 1e-6 < previous_ordinary_after
            ):
                raise CombinedProductionError("combined cumulative GPU counter moved backwards")
            previous_ordinary_after = record.cumulative_gpu_seconds_after
        if previous_end is not None and started_at < previous_end:
            raise CombinedProductionError("combined calls overlap or ran out of manifest order")
        if completed_at < started_at:
            raise CombinedProductionError("combined call has inverted chronology")
        previous_end = completed_at
    if seen_phase5 != COMBINED_PHASE5_CALL_COUNT:
        raise CombinedProductionError("combined chronology did not include nine Phase 5 calls")
    try:
        next(phase5_iter)
    except StopIteration:
        pass
    else:
        raise CombinedProductionError("combined chronology contains extra Phase 5 calls")
    if previous_end is None:
        raise CombinedProductionError("combined chronology is empty")
    return previous_end


def _verify_shutdown_receipt(
    *,
    receipt: CombinedServiceShutdownReceipt,
    identity: CombinedServiceIdentity,
    latest_call_completed_at: datetime,
    artifacts: ArtifactStore,
) -> None:
    current = artifacts.ledger.gpu_summary().total_allocated_seconds
    if (
        receipt.service_identity_hash != identity.content_hash
        or receipt.service_pid != identity.service_pid
        or receipt.service_start_ticks != identity.service_start_ticks
        or receipt.shutdown_started_at < latest_call_completed_at
        or receipt.cumulative_gpu_seconds_after_shutdown
        < identity.cumulative_gpu_seconds_after_load
        or not math.isclose(
            receipt.cumulative_gpu_seconds_after_shutdown,
            current,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise CombinedProductionError("combined physical shutdown receipt changed")


def _append_public_summary(path: Path, summary: CombinedPublicSummary) -> None:
    CombinedJournal._assert_no_symlink_chain(path)
    payload = (summary.to_canonical_json() + "\n").encode("utf-8")
    _append_bytes_atomically(path, payload, label="public combined summary")


def _public_summary_from_verified_records(
    *,
    run_id: str,
    index: CombinedExecutionIndex,
    manifest: CombinedCallManifest,
    runtime: CombinedRuntimeBinding,
    ordinary_records: tuple[CombinedITTRecord, ...],
    phase5_results: tuple[C2RegenerationResult, ...],
) -> CombinedPublicSummary:
    phase5_outcomes = tuple(
        {
            FeedbackAttemptStatus.SUCCEEDED: RunOutcome.SUCCEEDED,
            FeedbackAttemptStatus.INVALID: RunOutcome.INVALID,
            FeedbackAttemptStatus.FAILED: RunOutcome.FAILED,
            FeedbackAttemptStatus.TIMED_OUT: RunOutcome.TIMED_OUT,
        }[item.receipt.attempt_status]
        for item in phase5_results
    )
    outcomes = tuple(item.outcome for item in ordinary_records) + phase5_outcomes
    elapsed = index.actual_gpu_seconds_after - index.actual_gpu_seconds_before
    return CombinedPublicSummary(
        run_id=run_id,
        manifest_hash=manifest.content_hash,
        execution_index_hash=index.content_hash,
        model_manifest_hash=runtime.model_manifest_hash,
        model_revision=runtime.model_revision,
        call_counts=dict(Counter(item.call_class.value for item in manifest.calls)),
        outcome_counts=dict(Counter(item.value for item in outcomes)),
        physical_gpu_request_count=COMBINED_BASE_CALL_COUNT + len(index.repair_claim_hashes),
        repair_request_count=len(index.repair_claim_hashes),
        allocated_gpu_seconds=elapsed,
        allocated_gpu_hours=elapsed / 3600,
        generated_at=index.completed_at,
    )


def _verify_public_summary(
    path: Path,
    *,
    index: CombinedExecutionIndex,
    manifest: CombinedCallManifest,
    runtime: CombinedRuntimeBinding,
    journal: CombinedJournal,
    expected_run_id: str,
) -> None:
    CombinedJournal._assert_no_symlink_chain(path)
    if not path.is_file():
        raise CombinedProductionError("terminal combined index lost its public summary")
    try:
        summary = CombinedPublicSummary.model_validate_json(path.read_bytes())
    except Exception as error:
        raise CombinedProductionError(f"invalid public combined summary: {error}") from error
    ordinary = tuple(
        journal.load(journal.call_itt_path(call), CombinedITTRecord)
        for call in _ordinary_calls(manifest)
    )
    phase5_calls = tuple(
        call
        for call in manifest.calls
        if call.call_class
        in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }
    )
    phase5_results = _phase5_result_records(
        phase5_root=journal.root / "phase5",
        phase5_calls=phase5_calls,
    )
    expected = _public_summary_from_verified_records(
        run_id=expected_run_id,
        index=index,
        manifest=manifest,
        runtime=runtime,
        ordinary_records=ordinary,
        phase5_results=phase5_results,
    )
    if summary != expected:
        raise CombinedProductionError("public combined summary changed after finalization")


def _verify_final_index(
    *,
    index: CombinedExecutionIndex,
    expected_run_id: str,
    manifest: CombinedCallManifest,
    configuration: CombinedBlockConfiguration,
    runtime: CombinedRuntimeBinding,
    journal: CombinedJournal,
) -> None:
    activation = journal.load(Path("activation_slot.json"), CombinedActivationSlot)
    identity = journal.load(Path("service_identity.json"), CombinedServiceIdentity)
    shutdown = journal.load(Path("shutdown_receipt.json"), CombinedServiceShutdownReceipt)
    phase5 = journal.load(Path("phase5_index.json"), Phase5JournalIndex)
    claims = journal.repair_claims()
    ordinary = tuple(
        journal.load(journal.call_itt_path(call), CombinedITTRecord)
        for call in _ordinary_calls(manifest)
    )
    phase5_calls = tuple(
        item
        for item in manifest.calls
        if item.call_class
        in {
            CombinedCallClass.SCRIPTED_FEEDBACK_C2,
            CombinedCallClass.RESEARCHER_TRACE_C2,
        }
    )
    phase5_results = _phase5_result_records(
        phase5_root=journal.root / "phase5",
        phase5_calls=phase5_calls,
    )
    if (
        index.run_id != expected_run_id
        or activation.run_id != expected_run_id
        or index.manifest_hash != manifest.content_hash
        or index.configuration_hash != configuration.content_hash
        or index.runtime_binding_hash != runtime.content_hash
        or index.upstream_gate_hash != manifest.upstream_gate_hash
        or index.activation_slot_hash != activation.content_hash
        or index.service_identity_hash != identity.content_hash
        or identity.activation_slot_hash != activation.content_hash
        or index.model_load_event_record_hash != identity.model_load_event_record_hash
        or index.shutdown_receipt_hash != shutdown.content_hash
        or index.phase5_index_hash != phase5.content_hash
        or index.phase5_request_hashes != phase5.c2_request_hashes
        or index.phase5_result_hashes != phase5.c2_result_hashes
        or index.phase5_result_hashes
        != tuple(item.content_hash for item in phase5_results)
        or index.ordinary_itt_record_hashes != tuple(item.content_hash for item in ordinary)
        or index.repair_claim_hashes != tuple(item.content_hash for item in claims)
        or index.all_call_spec_hashes_in_order
        != tuple(item.content_hash for item in manifest.calls)
        or not math.isclose(
            index.actual_gpu_seconds_before,
            activation.gpu_seconds_before,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            index.actual_gpu_seconds_after,
            shutdown.cumulative_gpu_seconds_after_shutdown,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise CombinedProductionError("combined final index changed on replay")


class CombinedGpuController:
    """Synchronous, one-service owner for the registered combined block."""

    def __init__(
        self,
        *,
        run_id: str,
        configuration: CombinedBlockConfiguration,
        manifest: CombinedCallManifest,
        runtime: CombinedRuntimeBinding,
        upstream_gate: CombinedUpstreamGate,
        phase5_inputs: Phase5ExecutionInputManifest,
        phase5_protocol: FeedbackProtocolConfiguration,
        phase5_prerequisites: PrimaryHeldOutResultsGate,
        provider: CombinedInputProvider,
        lifecycle_owner: CombinedLifecycleOwner,
        artifacts: ArtifactStore,
        phase5_storage_preflight: Callable[[], None],
        output_root: Path,
        public_summary_path: Path | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.run_id = run_id
        self.configuration = configuration
        self.manifest = manifest
        self.runtime = runtime
        self.upstream_gate = upstream_gate
        self.phase5_inputs = phase5_inputs
        self.phase5_protocol = phase5_protocol
        self.phase5_prerequisites = phase5_prerequisites
        self.provider = provider
        self.lifecycle_owner = lifecycle_owner
        self.artifacts = artifacts
        self.phase5_storage_preflight = phase5_storage_preflight
        repository = _repository_from_frozen_output(configuration, output_root)
        restricted_root = repository / "artifacts" / "restricted"
        public_root = repository / "artifacts" / "public"
        bounded_output = _require_contained_writable_path(
            output_root,
            authority_root=restricted_root,
            label="combined restricted output",
        )
        _require_contained_writable_path(
            artifacts.ledger.path,
            authority_root=restricted_root,
            label="combined SQLite ledger",
        )
        _require_contained_writable_path(
            artifacts.blobs.root,
            authority_root=restricted_root,
            label="combined restricted CAS",
        )
        if public_summary_path is not None:
            expected_public = repository / configuration.public_summary_path
            if public_summary_path.absolute() != expected_public:
                raise CombinedProductionError(
                    "combined public summary differs from its frozen repository path"
                )
            self.public_summary_path = _require_contained_writable_path(
                public_summary_path,
                authority_root=public_root,
                label="combined public summary",
            )
        else:
            self.public_summary_path = None
        if restricted_root.resolve(strict=False) == public_root.resolve(strict=False):
            raise CombinedProductionError("combined public and restricted roots alias")
        self.journal = CombinedJournal(bounded_output)
        self.clock = clock

    def _validate_gate(self) -> tuple[CombinedCallSpec, ...]:
        phase5_calls = validate_combined_execution_inputs(
            configuration=self.configuration,
            manifest=self.manifest,
            runtime=self.runtime,
            upstream_gate=self.upstream_gate,
            phase5_inputs=self.phase5_inputs,
            phase5_protocol=self.phase5_protocol,
            phase5_prerequisites=self.phase5_prerequisites,
        )
        # This execution owns a distinct append-only job namespace.  Register
        # its exact frozen protocol/code/configuration identity before any job
        # link or Phase 5 preflight; never alias combined jobs into the held-out
        # predecessor study.
        self.artifacts.ledger.register_study(
            study_id=self.run_id,
            protocol_hash=self.manifest.content_hash,
            code_manifest_hash=self.runtime.source_tree_association_hash,
            configuration_hash=self.configuration.content_hash,
            release_class=StoreReleaseClass.RESTRICTED,
            created_at=self.manifest.created_at,
        )
        materialize_phase5_cpu_ledger(
            inputs=self.phase5_inputs,
            protocol=self.phase5_protocol,
            artifacts=self.artifacts,
            ledger_study_id=self.run_id,
        )
        validate_phase5_ledger_preflight(
            inputs=self.phase5_inputs,
            artifacts=self.artifacts,
            ledger_study_id=self.run_id,
        )
        return phase5_calls

    def _activation_slot(self) -> tuple[CombinedActivationSlot, bool]:
        path = Path("activation_slot.json")
        if self.journal.exists(path):
            slot = self.journal.load(path, CombinedActivationSlot)
            if (
                slot.run_id != self.run_id
                or slot.manifest_hash != self.manifest.content_hash
                or slot.configuration_hash != self.configuration.content_hash
                or slot.runtime_binding_hash != self.runtime.content_hash
                or slot.upstream_gate_hash != self.upstream_gate.content_hash
            ):
                raise CombinedProductionError("combined activation intent changed on resume")
            return slot, False
        current = self.artifacts.ledger.gpu_summary().total_allocated_seconds
        if not math.isclose(
            current,
            self.upstream_gate.actual_allocated_gpu_seconds_before_block,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise CombinedProductionError("global GPU ledger changed after combined admission")
        load_hashes = tuple(
            _record_hash(event, _GPU_EVENT_FIELDS)
            for event in self.artifacts.ledger.gpu_events()
            if event.event_kind is GpuEventKind.MODEL_LOAD
        )
        slot = CombinedActivationSlot(
            run_id=self.run_id,
            manifest_hash=self.manifest.content_hash,
            configuration_hash=self.configuration.content_hash,
            runtime_binding_hash=self.runtime.content_hash,
            upstream_gate_hash=self.upstream_gate.content_hash,
            global_accounting_id=self.upstream_gate.global_accounting_id,
            gpu_seconds_before=current,
            model_load_event_hashes_before=load_hashes,
            short_reserve_slots_consumed_before=(
                self.upstream_gate.consumed_short_reserve_slots_before_block
            ),
            remaining_registered_p95_seconds_before=(
                self.upstream_gate.remaining_registered_p95_seconds_before_block
            ),
            created_at=self.clock(),
        )
        self.journal.append(path, slot)
        return slot, True

    def _recover_shutdown_before_active(self) -> None:
        """Discover a completed physical stop before attempting service adoption."""

        shutdown_path = Path("shutdown_receipt.json")
        identity_path = Path("service_identity.json")
        if self.journal.exists(shutdown_path) or not self.journal.exists(identity_path):
            return
        slot = self.journal.load(Path("activation_slot.json"), CombinedActivationSlot)
        identity = self.journal.load(identity_path, CombinedServiceIdentity)
        if (
            slot.run_id != self.run_id
            or slot.manifest_hash != self.manifest.content_hash
            or identity.activation_slot_hash != slot.content_hash
        ):
            raise CombinedProductionError("combined shutdown recovery changed run lineage")
        receipt = self.lifecycle_owner.recover_shutdown(identity.content_hash)
        if receipt is not None:
            if (
                receipt.service_identity_hash != identity.content_hash
                or receipt.service_pid != identity.service_pid
                or receipt.service_start_ticks != identity.service_start_ticks
            ):
                raise CombinedProductionError("recovered shutdown belongs to another service")
            self.journal.append(shutdown_path, receipt)

    def _obtain_service(
        self,
        slot: CombinedActivationSlot,
        fresh_slot: bool,
    ) -> tuple[CombinedOwnedService, CombinedServiceIdentity]:
        def activate_once() -> CombinedOwnedService:
            _require_combined_capacity(
                artifacts=self.artifacts,
                configuration=self.configuration,
                next_watchdog_seconds=self.configuration.model_load_watchdog_seconds,
                remaining_mandatory_p95_seconds=max(
                    0.0,
                    self.upstream_gate.remaining_registered_p95_seconds_before_block
                    - self.configuration.model_load_p95_seconds,
                ),
            )
            return self.lifecycle_owner.activate(slot)

        identity_path = Path("service_identity.json")
        if self.journal.exists(identity_path):
            expected = self.journal.load(identity_path, CombinedServiceIdentity)
            service = self.lifecycle_owner.recover_active(slot.content_hash)
            if service is None:
                raise CombinedRecoveryRequired(
                    "combined service identity exists but the live process cannot be adopted"
                )
            identity = service.identity()
            if identity != expected:
                raise CombinedProductionError("adopted combined service identity changed")
        elif fresh_slot:
            service = activate_once()
            identity = service.identity()
            self.journal.append(identity_path, identity)
        else:
            service = self.lifecycle_owner.recover_active(slot.content_hash)
            if service is None:
                # A controller may stop after durably creating the outer slot
                # but before entering the lifecycle owner.  The production
                # owner returns None only when it has proved that no load event,
                # service journal, checkpoint, or binding exists.  Replaying
                # that unconsumed intent is therefore still the first load.
                service = activate_once()
            identity = service.identity()
            self.journal.append(identity_path, identity)
        verify_service_identity(
            identity=identity,
            slot=slot,
            runtime=self.runtime,
            artifacts=self.artifacts,
        )
        _require_combined_shutdown_margin(
            artifacts=self.artifacts,
            configuration=self.configuration,
            service=service,
        )
        return service, identity

    def _shutdown(
        self,
        *,
        service: CombinedOwnedService,
        identity: CombinedServiceIdentity,
        latest_call_completed_at: datetime,
    ) -> CombinedServiceShutdownReceipt:
        path = Path("shutdown_receipt.json")
        if self.journal.exists(path):
            receipt = self.journal.load(path, CombinedServiceShutdownReceipt)
        else:
            receipt = self.lifecycle_owner.recover_shutdown(identity.content_hash)
            if receipt is None:
                _require_combined_shutdown_margin(
                    artifacts=self.artifacts,
                    configuration=self.configuration,
                    service=service,
                )
                receipt = self.lifecycle_owner.shutdown(service, self.clock())
            self.journal.append(path, receipt)
        _verify_shutdown_receipt(
            receipt=receipt,
            identity=identity,
            latest_call_completed_at=latest_call_completed_at,
            artifacts=self.artifacts,
        )
        return receipt

    def _validate_completed_work(
        self,
        *,
        slot: CombinedActivationSlot,
        ordinary_records: tuple[CombinedITTRecord, ...],
        phase5_calls: tuple[CombinedCallSpec, ...],
        phase5_index: Phase5JournalIndex,
        phase5_results: tuple[C2RegenerationResult, ...],
    ) -> tuple[tuple[CombinedRepairClaim, ...], datetime]:
        if len(ordinary_records) != COMBINED_ORDINARY_CALL_COUNT:
            raise CombinedProductionError("combined ordinary execution did not retain 40 calls")
        claims = self.journal.repair_claims()
        ordinary_repair_hashes = {
            item.repair_claim_hash
            for item in ordinary_records
            if item.repair_claim_hash is not None
        }
        phase5_request_hashes = set(phase5_index.c2_request_hashes)
        phase5_claims = {
            item.content_hash for item in claims if item.request_hash in phase5_request_hashes
        }
        if {item.content_hash for item in claims} != ordinary_repair_hashes | phase5_claims:
            raise CombinedProductionError("combined repair claims include an unconsumed slot")
        if slot.short_reserve_slots_consumed_before + len(claims) > 4:
            raise CombinedProductionError("combined block oversubscribed global short repairs")
        if len(phase5_calls) != len(phase5_results):
            raise CombinedProductionError("combined Phase 5 result inventory changed")
        latest = _validate_sequential_call_chronology(
            manifest=self.manifest,
            ordinary_records=ordinary_records,
            phase5_results=phase5_results,
        )
        return claims, latest

    def _complete(
        self,
        *,
        slot: CombinedActivationSlot,
        identity: CombinedServiceIdentity,
        ordinary_records: tuple[CombinedITTRecord, ...],
        phase5_calls: tuple[CombinedCallSpec, ...],
        phase5_index: Phase5JournalIndex,
        phase5_results: tuple[C2RegenerationResult, ...],
        shutdown: CombinedServiceShutdownReceipt,
    ) -> CombinedExecutionIndex:
        claims, latest = self._validate_completed_work(
            slot=slot,
            ordinary_records=ordinary_records,
            phase5_calls=phase5_calls,
            phase5_index=phase5_index,
            phase5_results=phase5_results,
        )
        _verify_shutdown_receipt(
            receipt=shutdown,
            identity=identity,
            latest_call_completed_at=latest,
            artifacts=self.artifacts,
        )
        index = CombinedExecutionIndex(
            run_id=self.run_id,
            manifest_hash=self.manifest.content_hash,
            configuration_hash=self.configuration.content_hash,
            runtime_binding_hash=self.runtime.content_hash,
            upstream_gate_hash=self.upstream_gate.content_hash,
            service_identity_hash=identity.content_hash,
            activation_slot_hash=slot.content_hash,
            ordinary_itt_record_hashes=tuple(item.content_hash for item in ordinary_records),
            phase5_index_hash=phase5_index.content_hash,
            phase5_request_hashes=phase5_index.c2_request_hashes,
            phase5_result_hashes=phase5_index.c2_result_hashes,
            all_call_spec_hashes_in_order=tuple(item.content_hash for item in self.manifest.calls),
            repair_claim_hashes=tuple(item.content_hash for item in claims),
            short_reserve_slots_consumed_before=slot.short_reserve_slots_consumed_before,
            short_reserve_slots_consumed_after=(
                slot.short_reserve_slots_consumed_before + len(claims)
            ),
            model_load_event_record_hash=identity.model_load_event_record_hash,
            shutdown_receipt_hash=shutdown.content_hash,
            actual_gpu_seconds_before=slot.gpu_seconds_before,
            actual_gpu_seconds_after=shutdown.cumulative_gpu_seconds_after_shutdown,
            completed_at=shutdown.stopped_at,
        )
        if self.public_summary_path is not None:
            summary = _public_summary_from_verified_records(
                run_id=self.run_id,
                index=index,
                manifest=self.manifest,
                runtime=self.runtime,
                ordinary_records=ordinary_records,
                phase5_results=phase5_results,
            )
            _append_public_summary(self.public_summary_path, summary)
        # Publish the terminal restricted index last.  Its presence therefore
        # certifies that every requested public companion was also persisted.
        self.journal.append(Path("execution_index.json"), index)
        return index

    def _run_once(self) -> CombinedExecutionIndex:
        phase5_calls = self._validate_gate()
        self.journal.initialize()
        self.journal.append(Path("manifest.json"), self.manifest)
        final_path = Path("execution_index.json")
        if self.journal.exists(final_path):
            index = self.journal.load(final_path, CombinedExecutionIndex)
            _verify_final_index(
                index=index,
                expected_run_id=self.run_id,
                manifest=self.manifest,
                configuration=self.configuration,
                runtime=self.runtime,
                journal=self.journal,
            )
            identity = self.journal.load(
                Path("service_identity.json"), CombinedServiceIdentity
            )
            shutdown = self.journal.load(
                Path("shutdown_receipt.json"), CombinedServiceShutdownReceipt
            )
            ordinary_records = _load_verified_ordinary_records(
                manifest=self.manifest,
                runtime=self.runtime,
                identity=identity,
                journal=self.journal,
                artifacts=self.artifacts,
            )
            phase5_results = _validate_phase5_claims(
                journal=self.journal,
                phase5_calls=phase5_calls,
                phase5_index=self.journal.load(
                    Path("phase5_index.json"), Phase5JournalIndex
                ),
            )
            latest = _validate_sequential_call_chronology(
                manifest=self.manifest,
                ordinary_records=ordinary_records,
                phase5_results=phase5_results,
            )
            _verify_shutdown_receipt(
                receipt=shutdown,
                identity=identity,
                latest_call_completed_at=latest,
                artifacts=self.artifacts,
            )
            if self.public_summary_path is not None:
                _verify_public_summary(
                    self.public_summary_path,
                    index=index,
                    manifest=self.manifest,
                    runtime=self.runtime,
                    journal=self.journal,
                    expected_run_id=self.run_id,
                )
            return index

        self._recover_shutdown_before_active()
        shutdown_path = Path("shutdown_receipt.json")
        if self.journal.exists(shutdown_path):
            if not _all_required_work_exists(self.journal, self.manifest):
                raise CombinedProductionError(
                    "combined service is stopped but the required ITT journal is incomplete"
                )
            slot = self.journal.load(Path("activation_slot.json"), CombinedActivationSlot)
            identity = self.journal.load(Path("service_identity.json"), CombinedServiceIdentity)
            if (
                slot.run_id != self.run_id
                or slot.manifest_hash != self.manifest.content_hash
                or slot.configuration_hash != self.configuration.content_hash
                or slot.runtime_binding_hash != self.runtime.content_hash
                or slot.upstream_gate_hash != self.upstream_gate.content_hash
                or identity.activation_slot_hash != slot.content_hash
            ):
                raise CombinedProductionError("stopped combined run changed its activation lineage")
            verify_service_identity(
                identity=identity,
                slot=slot,
                runtime=self.runtime,
                artifacts=self.artifacts,
            )
            ordinary_records = _load_verified_ordinary_records(
                manifest=self.manifest,
                runtime=self.runtime,
                identity=identity,
                journal=self.journal,
                artifacts=self.artifacts,
            )
            phase5_index = self.journal.load(Path("phase5_index.json"), Phase5JournalIndex)
            phase5_results = _validate_phase5_claims(
                journal=self.journal,
                phase5_calls=phase5_calls,
                phase5_index=phase5_index,
            )
            shutdown = self.journal.load(shutdown_path, CombinedServiceShutdownReceipt)
            return self._complete(
                slot=slot,
                identity=identity,
                ordinary_records=ordinary_records,
                phase5_calls=phase5_calls,
                phase5_index=phase5_index,
                phase5_results=phase5_results,
                shutdown=shutdown,
            )

        slot, fresh_slot = self._activation_slot()
        service, identity = self._obtain_service(slot, fresh_slot)
        first = tuple(item for item in _ordinary_calls(self.manifest) if item.ordinal <= 12)
        last = tuple(item for item in _ordinary_calls(self.manifest) if item.ordinal >= 22)
        ordinary_records: list[CombinedITTRecord] = []
        for call in first:
            ordinary_records.append(
                _execute_or_recover_ordinary(
                    call=call,
                    manifest=self.manifest,
                    configuration=self.configuration,
                    upstream_gate=self.upstream_gate,
                    runtime=self.runtime,
                    service=service,
                    identity=identity,
                    provider=self.provider,
                    journal=self.journal,
                    artifacts=self.artifacts,
                    short_reserve_consumed_before=slot.short_reserve_slots_consumed_before,
                    clock=self.clock,
                )
            )

        phase5_path = Path("phase5_index.json")
        if self.journal.exists(phase5_path):
            phase5_index = self.journal.load(phase5_path, Phase5JournalIndex)
        else:
            self.phase5_storage_preflight()
            phase5_view = _CombinedPhase5ServiceView(
                service=service,
                identity=identity,
                configuration=self.configuration,
                manifest=self.manifest,
                upstream_gate=self.upstream_gate,
                calls=phase5_calls,
                journal=self.journal,
                artifacts=self.artifacts,
                short_reserve_consumed_before=slot.short_reserve_slots_consumed_before,
                clock=self.clock,
            )
            adapter = build_metered_phase5_adapter(
                service=phase5_view,
                artifacts=self.artifacts,
            )
            phase5_index = execute_phase5(
                inputs=self.phase5_inputs,
                protocol=self.phase5_protocol,
                prerequisites=self.phase5_prerequisites,
                adapter=adapter,
                ledger_verifier=SQLiteFeedbackLedgerVerifier(self.artifacts.ledger.path),
                output_root=self.journal.root / "phase5",
                completed_at=self.clock,
                artifacts=self.artifacts,
                ledger_study_id=self.run_id,
            )
            self.journal.append(phase5_path, phase5_index)
        phase5_results = _validate_phase5_claims(
            journal=self.journal,
            phase5_calls=phase5_calls,
            phase5_index=phase5_index,
        )

        for call in last:
            ordinary_records.append(
                _execute_or_recover_ordinary(
                    call=call,
                    manifest=self.manifest,
                    configuration=self.configuration,
                    upstream_gate=self.upstream_gate,
                    runtime=self.runtime,
                    service=service,
                    identity=identity,
                    provider=self.provider,
                    journal=self.journal,
                    artifacts=self.artifacts,
                    short_reserve_consumed_before=slot.short_reserve_slots_consumed_before,
                    clock=self.clock,
                )
            )
        ordinary_tuple = tuple(ordinary_records)
        _claims, latest = self._validate_completed_work(
            slot=slot,
            ordinary_records=ordinary_tuple,
            phase5_calls=phase5_calls,
            phase5_index=phase5_index,
            phase5_results=phase5_results,
        )
        shutdown = self._shutdown(
            service=service,
            identity=identity,
            latest_call_completed_at=latest,
        )
        return self._complete(
            slot=slot,
            identity=identity,
            ordinary_records=ordinary_tuple,
            phase5_calls=phase5_calls,
            phase5_index=phase5_index,
            phase5_results=phase5_results,
            shutdown=shutdown,
        )

    def _terminalize_after_controller_error(self) -> None:
        """Bound a post-acquisition failure with a durable physical-stop receipt."""

        identity_path = Path("service_identity.json")
        shutdown_path = Path("shutdown_receipt.json")
        if self.journal.exists(shutdown_path):
            return
        activation_path = Path("activation_slot.json")
        if not self.journal.exists(activation_path):
            return
        slot = self.journal.load(activation_path, CombinedActivationSlot)
        active: CombinedOwnedService | None = None
        if self.journal.exists(identity_path):
            identity = self.journal.load(identity_path, CombinedServiceIdentity)
        else:
            # The outer identity append can fail after lifecycle activation has
            # already returned a live owned service.  Recovering is side-effect
            # free with respect to model loading; persist that exact identity so
            # cleanup itself is restartable.
            active = self.lifecycle_owner.recover_active(slot.content_hash)
            if active is None:
                return
            identity = active.identity()
            self.journal.append(identity_path, identity)
        if identity.activation_slot_hash != slot.content_hash:
            raise CombinedProductionError(
                "cannot clean up a service whose activation lineage changed"
            )
        receipt = self.lifecycle_owner.recover_shutdown(identity.content_hash)
        if receipt is None:
            if active is None:
                active = self.lifecycle_owner.recover_active(slot.content_hash)
            if active is None:
                # A recovery attempt can itself discover and terminalize a
                # stale lease.  Give the owner one final receipt lookup before
                # failing closed with its exclusive ownership state intact.
                receipt = self.lifecycle_owner.recover_shutdown(identity.content_hash)
            else:
                receipt = self.lifecycle_owner.shutdown(active, self.clock())
        if receipt is None:
            raise CombinedRecoveryRequired(
                "combined controller failed and service shutdown is not yet provable"
            )
        if (
            receipt.service_identity_hash != identity.content_hash
            or receipt.service_pid != identity.service_pid
            or receipt.service_start_ticks != identity.service_start_ticks
        ):
            raise CombinedProductionError("controller cleanup receipt changed service identity")
        self.journal.append(shutdown_path, receipt)

    def run(self) -> CombinedExecutionIndex:
        try:
            return self._run_once()
        except BaseException as controller_error:
            try:
                self._terminalize_after_controller_error()
            except BaseException as cleanup_error:
                raise CombinedRecoveryRequired(
                    "combined controller failure could not be bounded by a durable stop"
                ) from cleanup_error
            raise controller_error


__all__ = [
    "CombinedActivationSlot",
    "CombinedCallSlot",
    "CombinedExecutionIndex",
    "CombinedGpuController",
    "CombinedITTRecord",
    "CombinedInputProvider",
    "CombinedJournal",
    "CombinedLifecycleOwner",
    "CombinedModelCallLineage",
    "CombinedOwnedService",
    "CombinedPreparedCall",
    "CombinedProductionError",
    "CombinedPublicSummary",
    "CombinedRecoveryRequired",
    "CombinedRepairClaim",
    "CombinedServiceCallResult",
    "CombinedServiceIdentity",
    "CombinedServiceShutdownReceipt",
    "ParaphraseRevealIntent",
    "ParaphraseRevealReceipt",
    "RepairAuthority",
    "persist_paraphrase_reveal",
    "validate_combined_execution_inputs",
    "validate_prepared_call",
    "verify_service_identity",
]
