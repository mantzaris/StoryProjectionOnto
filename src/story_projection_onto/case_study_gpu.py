"""Concrete restricted GPU adapter for the bounded first-novel case study.

The module contains no corpus discovery and no model-output fixtures.  It owns the
single registered case-study vLLM lifecycle supplied by the caller, constructs the
same lossless/schema-guided requests used by development, and persists every
attempt through the cumulative SQLite ledger and restricted CAS. Completed calls
replay from immutable receipts; interrupted calls recover from terminal ledger/CAS
lineage and are never silently issued a second time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from story_projection_onto.case_study_execution import (
    CASE_REQUIRED_NEXT_REPAIR_SECONDS,
    CASE_SERVICE_START_WATCHDOG_SECONDS,
    HARD_GPU_LIMIT_SECONDS,
    SCHEDULED_GPU_LIMIT_SECONDS,
    CaseArtifactReference,
    CaseExecutionAdmissionReceipt,
    CaseExecutionRepository,
    CaseGpuCallAuditReceipt,
    CaseRepairDiagnostic,
    CaseRepairInput,
    CaseStudyAdmissionError,
    CaseStudyExecutionError,
    OwnedCaseModelService,
    PreparedCaseC1Requests,
    _CaseProduceInputs,
    _CaseQueryAccessProxy,
    _configuration_hash,
    _fsync_directory,
    _gpu_event,
    _parse_record,
    _parse_timestamp,
    _persist_mapping,
    _persist_record,
    _publish_private_no_replace,
    _read_reference,
    _replace_private_pointer,
    _require_restricted_path,
    _total_allocated_seconds,
    preflight_case_c1_requests,
)
from story_projection_onto.case_study_runtime import (
    AttestedRestrictedCaseStudy,
    CaseC1RequestEnvelope,
    CaseC2RequestEnvelope,
    CaseGpuCallRole,
    CaseGpuCallSlot,
    CaseOutputReceipt,
    CasePrequeryBarrierReceipt,
    CasePrequeryKind,
    CasePrequeryReceipt,
    CaseQueryAccessReceipt,
    CaseStudyExecutionPlan,
    CaseWindowExecutionPlan,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ConditionPreparation,
    RunConditionConfig,
)
from story_projection_onto.conditions.c1 import (
    project_sealed_c1,
    seal_c1_preconstruction,
)
from story_projection_onto.conditions.c2 import (
    build_c2_construction_request,
    finalize_c2_draft,
)
from story_projection_onto.contracts import (
    CommitmentCheckStatus,
    ConditionName,
    ConstructionRequest,
    EvidencePacket,
    EvidenceSupportStatus,
    ImmutableRecord,
    OntologyDraft,
    PreconstructionRequest,
    QueryContext,
    ReleaseClass,
    RunOutcome,
    Sha256Digest,
    TemporalDeterminationStatus,
    ValidatedGeneration,
    ValidationRecord,
    ValidationStatus,
    canonical_json,
    canonical_sha256,
    normalize_generation_metadata,
    to_model_visible_query,
)
from story_projection_onto.development_adapter import (
    DevelopmentConstructionConfiguration,
    ModelWireAliasManifest,
    PackingTokenizer,
    build_development_guided_request,
    development_output_schema_for_request,
    development_request_runtime,
    development_runtime_identifiers,
    encode_development_semantic_request,
    restore_model_output_source_aliases,
)
from story_projection_onto.gpu_runtime import (
    ChatMessage,
    GenerationResult,
    GuidedJSONRequest,
    ServiceState,
    TokenizerManifest,
)
from story_projection_onto.llm import CapabilityManifest, PackingReport, PackingSection
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.novel_case import WindowEvidenceBundle
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    FailureKind,
    ModelBackend,
    ModelCallRole,
    RetryClass,
)
from story_projection_onto.store import (
    CommitmentCheckStatus as LedgerCommitmentCheckStatus,
)
from story_projection_onto.store import (
    EvidenceSupportStatus as LedgerEvidenceSupportStatus,
)
from story_projection_onto.store import ReleaseClass as LedgerReleaseClass
from story_projection_onto.store import (
    TemporalValidationStatus as LedgerTemporalValidationStatus,
)
from story_projection_onto.store import ValidationStatus as LedgerValidationStatus
from story_projection_onto.validate import (
    BoundaryValidationError,
    BoundaryValidationReport,
    validate_draft_structure,
    validate_repair_preservation,
)

CASE_GPU_ADAPTER_REVISION = "case-study-gpu-adapter-v1"
_LONG_RESERVE_LIMIT = 4
_STANDARD_RESERVE_LIMIT = 8


class CaseGpuAdapterError(CaseStudyExecutionError):
    """The concrete case GPU adapter detected an immutable-lineage violation."""


class CaseServiceIdentity(ImmutableRecord):
    """Physical identity of the one model process owned by the case controller."""

    owner_execution_id: str = Field(min_length=1)
    service_pid: int = Field(gt=0)
    service_process_start_ticks: int = Field(gt=0)
    service_configuration_hash: Sha256Digest
    model_runtime_hash: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    load_event_id: str = Field(min_length=1)
    started_at: AwareDatetime
    model_load_count: Literal[1] = 1
    lifecycle_owner: Literal["case_controller"] = "case_controller"


class CaseGpuActivationIntent(ImmutableRecord):
    """Path-free authority committed before the sole physical model start."""

    intent_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    admission_receipt_hash: Sha256Digest
    packing_preflight: CaseArtifactReference
    service_configuration_hash: Sha256Digest
    session_id: str = Field(min_length=1)
    load_event_id: str = Field(min_length=1)
    remaining_required_seconds: float = Field(gt=0.0)
    requested_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseGpuShutdownIntent(ImmutableRecord):
    """Durable authority committed before stopping the exact service."""

    intent_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    service_identity: CaseArtifactReference
    lifecycle_receipt: CaseArtifactReference
    requested_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseGpuAdapterState(ImmutableRecord):
    """Append-only adapter index; the private pointer contains no novel prose."""

    state_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=0)
    parent_state_hash: Sha256Digest | None = None
    execution_plan_hash: Sha256Digest
    admission_receipt_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    service_configuration_hash: Sha256Digest
    tokenizer_manifest_hash: Sha256Digest
    packing_preflight: CaseArtifactReference | None = None
    activation_intent: CaseArtifactReference | None = None
    service_identity: CaseArtifactReference | None = None
    lifecycle_receipt: CaseArtifactReference | None = None
    shutdown_intent: CaseArtifactReference | None = None
    shutdown_receipt: CaseArtifactReference | None = None
    completed_receipts: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    call_audits: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    c1_preparations: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    c1_projection_receipts: Mapping[str, CaseArtifactReference] = Field(default_factory=dict)
    active_call_id: str | None = None
    consumed_repair_reservation_ids: tuple[str, ...] = ()
    model_service_start_count: Literal[0, 1] = 0
    model_load_count: Literal[0, 1] = 0
    model_service_shutdown_count: Literal[0, 1] = 0
    updated_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def coherent_lifecycle_and_indexes(self) -> CaseGpuAdapterState:
        if self.sequence_number == 0 and self.parent_state_hash is not None:
            raise ValueError("initial case GPU state cannot have a parent")
        if self.sequence_number > 0 and self.parent_state_hash is None:
            raise ValueError("case GPU successor requires its parent hash")
        if self.model_service_start_count != self.model_load_count:
            raise ValueError("case GPU start and model-load counts differ")
        if self.model_service_shutdown_count > self.model_service_start_count:
            raise ValueError("case GPU shutdown count exceeds its sole start")
        if self.model_service_start_count != (self.service_identity is not None):
            raise ValueError("case GPU start count and physical identity differ")
        if self.model_service_start_count != (self.lifecycle_receipt is not None):
            raise ValueError("case GPU start count and lifecycle receipt differ")
        if self.model_service_start_count and self.activation_intent is None:
            raise ValueError("case GPU start lacks its durable activation intent")
        if self.model_service_start_count and self.packing_preflight is None:
            raise ValueError("case GPU lifecycle cannot precede packing preflight")
        if self.model_service_shutdown_count != (self.shutdown_receipt is not None):
            raise ValueError("case GPU shutdown count and shutdown receipt differ")
        if self.model_service_shutdown_count and self.shutdown_intent is None:
            raise ValueError("case GPU shutdown lacks its durable shutdown intent")
        if self.shutdown_intent is not None and self.model_service_start_count != 1:
            raise ValueError("case GPU shutdown intent requires the registered start")
        if set(self.call_audits) != set(self.completed_receipts):
            raise ValueError("case GPU result and audit indexes differ")
        if self.active_call_id in self.completed_receipts:
            raise ValueError("completed case GPU call cannot remain active")
        if len(self.consumed_repair_reservation_ids) != len(
            set(self.consumed_repair_reservation_ids)
        ):
            raise ValueError("case repair reservation IDs must be unique")
        return self


class CaseGpuLifecycleReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    activation_intent: CaseArtifactReference
    service_identity: CaseArtifactReference
    packing_preflight: CaseArtifactReference
    allocated_gpu_seconds_at_start: float = Field(ge=0.0)
    remaining_required_seconds_at_start: float = Field(gt=0.0)
    start_count: Literal[1] = 1
    model_load_count: Literal[1] = 1
    created_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class CaseGpuShutdownReceipt(ImmutableRecord):
    """Proof that the controller reclaimed its exact one-load service."""

    receipt_id: str = Field(min_length=1)
    execution_plan_hash: Sha256Digest
    shutdown_intent: CaseArtifactReference
    service_identity: CaseArtifactReference
    lifecycle_receipt: CaseArtifactReference
    cumulative_allocated_gpu_seconds: float = Field(ge=0.0)
    service_state: Literal["stopped"] = "stopped"
    start_count: Literal[1] = 1
    model_load_count: Literal[1] = 1
    shutdown_count: Literal[1] = 1
    stopped_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


@dataclass(frozen=True, slots=True)
class _OperationalSnapshot:
    content_hash: str


@dataclass(frozen=True, slots=True)
class _AttemptResult:
    outcome: RunOutcome
    job_id: str
    base_attempt_id: str
    base_model_call_id: str
    base_event_id: str
    base_raw_hash: str
    validation_ids: tuple[str, ...]
    semantic: PreconstructionRequest | ConstructionRequest
    config: RunConditionConfig
    guided: GuidedJSONRequest
    alias_manifest: ModelWireAliasManifest
    generation: ValidatedGeneration | None
    boundary: BoundaryValidationReport | None
    repair_attempt_id: str | None = None
    repair_model_call_id: str | None = None
    repair_event_id: str | None = None
    repair_input_reference: CaseArtifactReference | None = None
    repair_raw_hash: str | None = None
    repair_preservation_reference: CaseArtifactReference | None = None
    failure_lineage_hash: str | None = None


def _failure_kind(error: BaseException) -> FailureKind:
    if isinstance(error, TimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(error, (ValidationError, BoundaryValidationError, ValueError)):
        return FailureKind.INVALID_OUTPUT
    if isinstance(error, MemoryError) or "out of memory" in str(error).casefold():
        return FailureKind.OUT_OF_MEMORY
    return FailureKind.SERVICE


def _outcome(error: BaseException) -> RunOutcome:
    if isinstance(error, TimeoutError):
        return RunOutcome.TIMED_OUT
    if isinstance(error, (ValidationError, BoundaryValidationError, ValueError)):
        return RunOutcome.INVALID
    return RunOutcome.FAILED


def _process_start_ticks(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        value = int(fields[21])
    except (OSError, IndexError, ValueError) as exc:
        raise CaseGpuAdapterError("case model process identity is unavailable") from exc
    if value <= 0:
        raise CaseGpuAdapterError("case model process start ticks are invalid")
    return value


def _atomic_pointer(path: Path, reference: CaseArtifactReference) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical_json(reference).encode("utf-8") + b"\n"
    history = path.parent / f"{path.name}.history"
    if history.exists() and (history.is_symlink() or not history.is_dir()):
        raise CaseGpuAdapterError("case GPU pointer history is not a real directory")
    if not history.exists():
        history.mkdir(mode=0o700)
        _fsync_directory(history.parent)
    _publish_private_no_replace(history / f"{reference.logical_content_hash}.json", payload)
    _replace_private_pointer(path, payload)


def _strictly_after(clock: Callable[[], datetime], threshold: datetime) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise CaseGpuAdapterError("case GPU clock must be timezone-aware")
    value = value.astimezone(UTC)
    return value if value > threshold else threshold + timedelta(microseconds=1)


def _case_query_stage_hash(plan: CaseStudyExecutionPlan, context: QueryContext) -> str:
    return canonical_sha256(
        {
            "execution_plan_hash": plan.content_hash,
            "context_id": context.context_id,
            "context_hash": context.content_hash,
            "registered_revealed_at": context.revealed_at,
        }
    )


def _repair_guided_request(
    *,
    root: Path,
    call_id: str,
    semantic_request: PreconstructionRequest | ConstructionRequest,
    repair_input: CaseRepairInput,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    model_name: str,
    model_revision: str,
    seed: int,
) -> GuidedJSONRequest:
    """Build the complete fact-free repair request without semantic supplementation."""

    if repair_input.semantic_request_hash != semantic_request.content_hash:
        raise CaseGpuAdapterError("case repair input belongs to another semantic request")
    runtime = development_request_runtime(
        root=root,
        condition=semantic_request.condition,
        tokenizer_manifest=tokenizer_manifest,
        seed=seed,
        repair=True,
    )
    expected = semantic_request.runtime
    if (
        expected.model_id,
        expected.model_revision,
        expected.tokenizer_hash,
        expected.prompt_hash,
        expected.output_schema_hash,
        expected.decoding_config_hash,
    ) != (
        model_name,
        model_revision,
        tokenizer_manifest.manifest_sha256,
        runtime.prompt_hash,
        runtime.output_schema_hash,
        runtime.decoding_manifest.content_hash,
    ):
        raise CaseGpuAdapterError("case repair request differs from the selected stack")
    prompt = (root / "prompts/repair/prompt_v1.md").read_text(encoding="utf-8")
    encoded = encode_development_semantic_request(semantic_request)
    sections = dict(encoded.sections)
    sections["invalid_draft"] = dict(repair_input.invalid_draft)
    sections["validation_diagnostics"] = [
        item.model_dump(mode="json", exclude={"schema_version", "content_hash"})
        for item in repair_input.diagnostics
    ]
    messages = (
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=canonical_json(sections)),
    )
    rendered = tokenizer.apply_chat_template(
        [asdict(item) for item in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes, bytearray)):
        raise CaseGpuAdapterError("case tokenizer did not return token IDs")
    schema = development_output_schema_for_request(semantic_request)
    section_values: dict[str, object] = {"system_prompt": prompt, **sections}
    packing_sections = [
        PackingSection(
            name="output_schema",
            section_content_hash=canonical_sha256(schema),
            token_count=0,
        )
    ]
    for name, value in section_values.items():
        text = value if isinstance(value, str) else canonical_json(value)
        packing_sections.append(
            PackingSection(
                name=name,
                section_content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                token_count=len(tokenizer.encode(text, add_special_tokens=False)),
            )
        )
    individually_encoded = sum(item.token_count for item in packing_sections)
    if len(rendered) < individually_encoded:
        raise CaseGpuAdapterError("repair section token counts exceed rendered prompt")
    if len(rendered) > individually_encoded:
        packing_sections.append(
            PackingSection(
                name="chat_protocol",
                section_content_hash=tokenizer_manifest.nonthinking_probe_sha256,
                token_count=len(rendered) - individually_encoded,
            )
        )
    packing = PackingReport.build(
        condition=semantic_request.condition,
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_model_tokens=runtime.decoding_manifest.maximum_model_tokens,
        maximum_input_tokens=runtime.decoding_manifest.maximum_input_tokens,
        reserved_output_tokens=runtime.decoding_manifest.maximum_output_tokens,
        sections=packing_sections,
        required_section_names=("output_schema", *section_values),
        complete_evidence_snapshot=(
            True if isinstance(semantic_request, PreconstructionRequest) else None
        ),
        complete_evidence_packet=(
            None if isinstance(semantic_request, PreconstructionRequest) else True
        ),
    )
    return GuidedJSONRequest(
        request_id=f"{call_id}-repair",
        model_name=model_name,
        condition=semantic_request.condition,
        messages=messages,
        output_schema=schema,
        decoding=runtime.decoding_manifest,
        packing=packing,
        rendered_input_token_count=len(rendered),
    )


@dataclass(slots=True)
class ProductionCaseStudyGpuAdapter:
    """Real one-load adapter used by :class:`CaseStudyProductionController`."""

    root: Path
    plan: CaseStudyExecutionPlan
    admission: CaseExecutionAdmissionReceipt
    construction: DevelopmentConstructionConfiguration
    tokenizer: PackingTokenizer = field(repr=False)
    tokenizer_manifest: TokenizerManifest
    service: OwnedCaseModelService = field(repr=False)
    artifacts: ArtifactStore
    repository: CaseExecutionRepository
    state_pointer_path: Path
    model_manifest_hash: Sha256Digest
    service_start_watchdog_seconds: int = CASE_SERVICE_START_WATCHDOG_SECONDS
    process_start_ticks: Callable[[int], int] = _process_start_ticks
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    backend: Literal["vllm_gpu"] = "vllm_gpu"
    _prepared: PreparedCaseC1Requests | None = field(default=None, init=False, repr=False)
    _bounded_packets: dict[str, WindowEvidenceBundle] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.root = self.root.resolve(strict=True)
        restricted = self.repository.restricted_root.resolve(strict=True)
        parent = _require_restricted_path(
            self.state_pointer_path.parent,
            restricted,
            label="case GPU state",
        )
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.resolve(strict=True) != parent:
            raise CaseStudyAdmissionError("case GPU state must remain in restricted storage")
        if self.state_pointer_path.is_symlink():
            raise CaseStudyAdmissionError("case GPU state pointer cannot be a symlink")
        if self.service_start_watchdog_seconds != CASE_SERVICE_START_WATCHDOG_SECONDS:
            raise CaseStudyAdmissionError("case service startup watchdog changed")
        if self.admission.execution_plan_hash != self.plan.content_hash:
            raise CaseStudyAdmissionError("case GPU admission belongs to another plan")
        if self.construction.content_hash != self.admission.construction_configuration_hash:
            raise CaseStudyAdmissionError("case GPU construction configuration changed")
        runtime = self.plan.model_runtime
        if (
            self.tokenizer_manifest.manifest_sha256 != runtime.selected_tokenizer_manifest_hash
            or self.tokenizer_manifest.tokenizer_revision != runtime.model_revision
            or self.model_manifest_hash != runtime.selected_snapshot_manifest_hash
            or _configuration_hash(self.service) != runtime.selected_launcher_configuration_hash
            or self.construction.upper_ontology.content_hash != runtime.upper_ontology_hash
            or runtime.c1_capability_manifest_hash
            != CapabilityManifest.for_condition(ConditionName.C1_LLM_PRE).content_hash
            or runtime.c2_capability_manifest_hash
            != CapabilityManifest.for_condition(ConditionName.C2_LLM_QUERY).content_hash
        ):
            raise CaseStudyAdmissionError("case GPU adapter differs from selected-model freeze")
        if _total_allocated_seconds(self.artifacts.ledger) + 1e-6 < (
            self.admission.allocated_gpu_seconds_before_case
        ):
            raise CaseStudyAdmissionError("case GPU adapter reset cumulative accounting")
        if self.state_pointer_path.exists():
            self._state()

    @property
    def service_configuration_hash(self) -> Sha256Digest:
        return _configuration_hash(self.service)

    @property
    def actual_allocated_service_seconds(self) -> float:
        value = self.service.actual_allocated_service_seconds
        if not math.isfinite(value) or value < 0:
            raise CaseGpuAdapterError("case GPU allocation counter is invalid")
        return value

    @property
    def model_service_start_count(self) -> Literal[0, 1]:
        return self._state().model_service_start_count

    @property
    def model_service_shutdown_count(self) -> Literal[0, 1]:
        return self._state().model_service_shutdown_count

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise CaseGpuAdapterError("case GPU clock must be timezone-aware")
        return value.astimezone(UTC)

    def _initial_state(self) -> CaseGpuAdapterState:
        return CaseGpuAdapterState(
            state_id=f"case-gpu-state-{self.plan.execution_id}-0000",
            sequence_number=0,
            execution_plan_hash=self.plan.content_hash,
            admission_receipt_hash=self.admission.content_hash,
            source_manifest_hash=self.admission.source_manifest.logical_content_hash,
            service_configuration_hash=self.service_configuration_hash,
            tokenizer_manifest_hash=self.tokenizer_manifest.manifest_sha256,
            updated_at=self._now(),
        )

    def _state(self) -> CaseGpuAdapterState:
        candidates: list[bytes] = []
        if self.state_pointer_path.is_file() and not self.state_pointer_path.is_symlink():
            candidates.append(self.state_pointer_path.read_bytes())
        history = self.state_pointer_path.parent / f"{self.state_pointer_path.name}.history"
        if history.is_dir() and not history.is_symlink():
            candidates.extend(path.read_bytes() for path in sorted(history.glob("*.json")))
        states: dict[str, CaseGpuAdapterState] = {}
        for payload in candidates:
            try:
                reference = CaseArtifactReference.model_validate_json(payload)
                state = cast(
                    CaseGpuAdapterState,
                    _parse_record(self.artifacts, reference, CaseGpuAdapterState),
                )
            except Exception:
                continue
            states[state.content_hash] = state
        if not states:
            if candidates:
                raise CaseGpuAdapterError("case GPU state pointer history is invalid")
            return self._initial_state()
        latest_sequence = max(item.sequence_number for item in states.values())
        latest = tuple(item for item in states.values() if item.sequence_number == latest_sequence)
        if len(latest) != 1:
            raise CaseGpuAdapterError("case GPU state pointer history forks at latest state")
        state = latest[0]
        expected = (
            self.plan.content_hash,
            self.admission.content_hash,
            self.admission.source_manifest.logical_content_hash,
            self.service_configuration_hash,
            self.tokenizer_manifest.manifest_sha256,
        )
        observed = (
            state.execution_plan_hash,
            state.admission_receipt_hash,
            state.source_manifest_hash,
            state.service_configuration_hash,
            state.tokenizer_manifest_hash,
        )
        if observed != expected:
            raise CaseGpuAdapterError("case GPU state belongs to another frozen execution")
        return state

    def _write_state(self, state: CaseGpuAdapterState, **updates: object) -> CaseGpuAdapterState:
        values = state.model_dump(mode="python", exclude={"content_hash"})
        values.update(updates)
        values.update(
            {
                "state_id": (
                    f"case-gpu-state-{self.plan.execution_id}-{state.sequence_number + 1:04d}"
                ),
                "sequence_number": state.sequence_number + 1,
                "parent_state_hash": state.content_hash,
                "updated_at": _strictly_after(self.clock, state.updated_at),
            }
        )
        successor = CaseGpuAdapterState.model_validate(values)
        reference = _persist_record(
            self.artifacts,
            successor,
            object_kind="case_gpu_adapter_state",
            created_at=successor.updated_at,
        )
        _atomic_pointer(self.state_pointer_path, reference)
        return successor

    def _persist_initial_if_needed(self) -> CaseGpuAdapterState:
        state = self._state()
        if not self.state_pointer_path.exists():
            reference = _persist_record(
                self.artifacts,
                state,
                object_kind="case_gpu_adapter_state",
                created_at=state.updated_at,
            )
            _atomic_pointer(self.state_pointer_path, reference)
        return state

    def preflight(
        self,
        *,
        plan: CaseStudyExecutionPlan,
        bounded_packets: Mapping[str, WindowEvidenceBundle],
    ) -> CaseArtifactReference:
        if plan != self.plan:
            raise CaseGpuAdapterError("case packing preflight received another plan")
        prepared = preflight_case_c1_requests(
            root=self.root,
            plan=self.plan,
            construction=self.construction,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            bounded_packets=bounded_packets,
            artifacts=self.artifacts,
            clock=self.clock,
        )
        state = self._persist_initial_if_needed()
        if state.packing_preflight is not None and (
            state.packing_preflight != prepared.receipt_reference
        ):
            raise CaseGpuAdapterError("case GPU packing preflight changed on resume")
        if state.packing_preflight is None:
            self._write_state(state, packing_preflight=prepared.receipt_reference)
        self._prepared = prepared
        self._bounded_packets = dict(bounded_packets)
        return prepared.receipt_reference

    def _identity(self) -> CaseServiceIdentity:
        state = self._state()
        if state.service_identity is None:
            raise CaseGpuAdapterError("case service identity has not been persisted")
        return cast(
            CaseServiceIdentity,
            _parse_record(self.artifacts, state.service_identity, CaseServiceIdentity),
        )

    def _verify_live_identity(self) -> CaseServiceIdentity:
        if self.service.state is not ServiceState.READY:
            raise CaseGpuAdapterError("case model service is not ready")
        identity = self._identity()
        if (
            self.service.pid != identity.service_pid
            or self.process_start_ticks(self.service.pid) != identity.service_process_start_ticks
            or self.service_configuration_hash != identity.service_configuration_hash
        ):
            raise CaseGpuAdapterError("case model physical identity changed")
        return identity

    def _admit_remaining(self, required_seconds: float) -> None:
        actual = self.actual_allocated_service_seconds
        if actual + required_seconds > SCHEDULED_GPU_LIMIT_SECONDS:
            raise CaseStudyAdmissionError("remaining case schedule exceeds nine GPU hours")
        if actual >= HARD_GPU_LIMIT_SECONDS or actual + required_seconds >= HARD_GPU_LIMIT_SECONDS:
            raise CaseStudyAdmissionError("case GPU execution would reach the hard stop")

    def _activation_intent(self, state: CaseGpuAdapterState) -> CaseGpuActivationIntent | None:
        if state.activation_intent is None:
            return None
        intent = cast(
            CaseGpuActivationIntent,
            _parse_record(
                self.artifacts,
                state.activation_intent,
                CaseGpuActivationIntent,
            ),
        )
        if (
            intent.execution_plan_hash != self.plan.content_hash
            or intent.admission_receipt_hash != self.admission.content_hash
            or intent.packing_preflight != state.packing_preflight
            or intent.service_configuration_hash != self.service_configuration_hash
            or intent.session_id != f"{self.plan.execution_id}-session"
            or intent.load_event_id != f"{self.plan.execution_id}-model-load"
        ):
            raise CaseGpuAdapterError("case activation intent differs from the frozen service")
        return intent

    def _ensure_activation_intent(
        self,
        state: CaseGpuAdapterState,
        *,
        remaining_required_seconds: float,
    ) -> tuple[CaseGpuAdapterState, CaseGpuActivationIntent]:
        existing = self._activation_intent(state)
        if existing is not None:
            if abs(existing.remaining_required_seconds - remaining_required_seconds) > 1e-6:
                raise CaseGpuAdapterError("case activation reserve changed on resume")
            return state, existing
        if state.packing_preflight is None:
            raise CaseGpuAdapterError("case service cannot start before complete C1 packing")
        intent = CaseGpuActivationIntent(
            intent_id=f"activation-{self.plan.execution_id}",
            execution_plan_hash=self.plan.content_hash,
            admission_receipt_hash=self.admission.content_hash,
            packing_preflight=state.packing_preflight,
            service_configuration_hash=self.service_configuration_hash,
            session_id=f"{self.plan.execution_id}-session",
            load_event_id=f"{self.plan.execution_id}-model-load",
            remaining_required_seconds=remaining_required_seconds,
            requested_at=self._now(),
        )
        reference = _persist_record(
            self.artifacts,
            intent,
            object_kind="case_gpu_activation_intent",
            created_at=intent.requested_at,
        )
        return self._write_state(state, activation_intent=reference), intent

    def _service_started_at(self, intent: CaseGpuActivationIntent) -> datetime:
        journal = self.artifacts.ledger.latest_gpu_service_journal(intent.load_event_id)
        if journal is not None:
            if (
                journal.session_id != intent.session_id
                or journal.configuration_hash != intent.service_configuration_hash
            ):
                raise CaseGpuAdapterError("case service journal differs from activation intent")
            return _parse_timestamp(journal.service_started_at)
        # Lightweight test services record only the load event. Production vLLM
        # always opens its service journal before spawning.
        return _parse_timestamp(_gpu_event(self.artifacts.ledger, intent.load_event_id).started_at)

    def _persist_started_lifecycle(
        self,
        state: CaseGpuAdapterState,
        intent: CaseGpuActivationIntent,
        *,
        service_pid: int,
        process_start_ticks: int,
        started_at: datetime,
    ) -> CaseGpuAdapterState:
        if state.model_service_start_count == 1:
            self._verify_live_identity()
            return state
        if state.activation_intent is None:
            raise CaseGpuAdapterError("case service start lacks durable activation authority")
        now = self._now()
        identity = CaseServiceIdentity(
            owner_execution_id=self.plan.execution_id,
            service_pid=service_pid,
            service_process_start_ticks=process_start_ticks,
            service_configuration_hash=self.service_configuration_hash,
            model_runtime_hash=self.plan.model_runtime.content_hash,
            selected_model_freeze_hash=self.plan.model_runtime.selected_model_freeze_hash,
            source_manifest_hash=self.admission.source_manifest.logical_content_hash,
            load_event_id=intent.load_event_id,
            started_at=started_at,
        )
        identity_ref = _persist_record(
            self.artifacts,
            identity,
            object_kind="case_service_identity",
            created_at=now,
        )
        lifecycle = CaseGpuLifecycleReceipt(
            receipt_id=f"lifecycle-{self.plan.execution_id}",
            execution_plan_hash=self.plan.content_hash,
            activation_intent=state.activation_intent,
            service_identity=identity_ref,
            packing_preflight=intent.packing_preflight,
            allocated_gpu_seconds_at_start=self.actual_allocated_service_seconds,
            remaining_required_seconds_at_start=intent.remaining_required_seconds,
            created_at=now,
        )
        lifecycle_ref = _persist_record(
            self.artifacts,
            lifecycle,
            object_kind="case_gpu_lifecycle_receipt",
            created_at=now,
        )
        return self._write_state(
            state,
            service_identity=identity_ref,
            lifecycle_receipt=lifecycle_ref,
            model_service_start_count=1,
            model_load_count=1,
        )

    def _ensure_shutdown_intent(
        self,
        state: CaseGpuAdapterState,
    ) -> tuple[CaseGpuAdapterState, CaseGpuShutdownIntent]:
        if state.service_identity is None or state.lifecycle_receipt is None:
            raise CaseGpuAdapterError("case shutdown lacks lifecycle lineage")
        if state.shutdown_intent is not None:
            intent = cast(
                CaseGpuShutdownIntent,
                _parse_record(
                    self.artifacts,
                    state.shutdown_intent,
                    CaseGpuShutdownIntent,
                ),
            )
            if (
                intent.execution_plan_hash != self.plan.content_hash
                or intent.service_identity != state.service_identity
                or intent.lifecycle_receipt != state.lifecycle_receipt
            ):
                raise CaseGpuAdapterError("case shutdown intent changed")
            return state, intent
        intent = CaseGpuShutdownIntent(
            intent_id=f"shutdown-intent-{self.plan.execution_id}",
            execution_plan_hash=self.plan.content_hash,
            service_identity=state.service_identity,
            lifecycle_receipt=state.lifecycle_receipt,
            requested_at=self._now(),
        )
        reference = _persist_record(
            self.artifacts,
            intent,
            object_kind="case_gpu_shutdown_intent",
            created_at=intent.requested_at,
        )
        return self._write_state(state, shutdown_intent=reference), intent

    def _finalize_terminal_service(
        self,
        state: CaseGpuAdapterState,
        *,
        stopped_at: datetime,
    ) -> CaseGpuAdapterState:
        if state.model_service_shutdown_count == 1:
            return state
        state, _intent = self._ensure_shutdown_intent(state)
        assert state.shutdown_intent is not None
        assert state.service_identity is not None
        assert state.lifecycle_receipt is not None
        receipt = CaseGpuShutdownReceipt(
            receipt_id=f"shutdown-{self.plan.execution_id}",
            execution_plan_hash=self.plan.content_hash,
            shutdown_intent=state.shutdown_intent,
            service_identity=state.service_identity,
            lifecycle_receipt=state.lifecycle_receipt,
            cumulative_allocated_gpu_seconds=self.actual_allocated_service_seconds,
            stopped_at=stopped_at,
        )
        receipt_ref = _persist_record(
            self.artifacts,
            receipt,
            object_kind="case_gpu_shutdown_receipt",
            created_at=stopped_at,
        )
        return self._write_state(
            state,
            model_service_shutdown_count=1,
            shutdown_receipt=receipt_ref,
            active_call_id=None,
        )

    def _recover_terminal_service(
        self,
        state: CaseGpuAdapterState,
        intent: CaseGpuActivationIntent,
    ) -> bool:
        # Replay the lease recovery first even when the terminal accounting row
        # already exists. The runtime repopulates the exact path-free process
        # identity needed after a crash between journal reconciliation and this
        # adapter's state publication.
        recover = getattr(self.service, "recover_stale_service_lease", None)
        session = None if recover is None else recover()
        if session is not None and session.service_session_id != intent.load_event_id:
            # A stopped, fully accounted service from an earlier phase may share
            # this launcher's lock path. Its verified absence is not evidence that
            # the current activation ran.
            session = None
        if session is None:
            session = self.artifacts.ledger.get_gpu_service_session(intent.load_event_id)
        if session is None:
            return False
        if state.model_service_start_count == 0:
            recovered = getattr(self.service, "last_recovered_process_identity", None)
            if recovered is None:
                raise CaseGpuAdapterError(
                    "terminal case service lacks its exact recovered process identity"
                )
            if (
                recovered.configuration_hash != self.service_configuration_hash
                or recovered.session_id != intent.session_id
                or recovered.accounting_session_id != intent.load_event_id
            ):
                raise CaseGpuAdapterError("recovered terminal case service identity changed")
            state = self._persist_started_lifecycle(
                state,
                intent,
                service_pid=recovered.pid,
                process_start_ticks=recovered.process_start_ticks,
                started_at=recovered.service_started_at,
            )
        stopped_at = _parse_timestamp(session.ended_at)
        self._finalize_terminal_service(state, stopped_at=stopped_at)
        return True

    def start_once(self, *, execution_id: str, remaining_required_seconds: float) -> None:
        state = self._state()
        if execution_id != self.plan.execution_id:
            raise CaseGpuAdapterError("case service start belongs to another execution")
        self._admit_remaining(remaining_required_seconds)
        state, intent = self._ensure_activation_intent(
            state,
            remaining_required_seconds=remaining_required_seconds,
        )
        if state.model_service_shutdown_count == 1:
            raise CaseGpuAdapterError("case adapter refuses a second model load")
        if state.model_service_start_count == 1:
            if self.service.state is ServiceState.READY:
                self._verify_live_identity()
                return
            raise CaseGpuAdapterError("case adapter refuses a second model load")
        resume_lease = getattr(self.service, "resume_live_service_lease", None)
        if resume_lease is not None and resume_lease(
            expected_session_id=intent.session_id,
            expected_event_id=intent.load_event_id,
            watchdog_seconds=self.service_start_watchdog_seconds,
        ):
            self._persist_started_lifecycle(
                state,
                intent,
                service_pid=self.service.pid,
                process_start_ticks=self.process_start_ticks(self.service.pid),
                started_at=self._service_started_at(intent),
            )
            return
        if self._recover_terminal_service(state, intent):
            raise CaseGpuAdapterError("case model load already terminated and cannot be repeated")
        self.service.start(
            session_id=intent.session_id,
            event_id=intent.load_event_id,
            watchdog_seconds=self.service_start_watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        if self.service.state is not ServiceState.READY:
            raise CaseGpuAdapterError("case service did not become ready")
        self._persist_started_lifecycle(
            state,
            intent,
            service_pid=self.service.pid,
            process_start_ticks=self.process_start_ticks(self.service.pid),
            started_at=self._service_started_at(intent),
        )

    def resume_live(self, checkpoint_path: Path) -> bool:
        state = self._state()
        if state.model_service_shutdown_count != 0:
            return False
        intent = self._activation_intent(state)
        if intent is None:
            return False
        if self.service.state is ServiceState.READY:
            if state.model_service_start_count == 0:
                self._persist_started_lifecycle(
                    state,
                    intent,
                    service_pid=self.service.pid,
                    process_start_ticks=self.process_start_ticks(self.service.pid),
                    started_at=self._service_started_at(intent),
                )
            self._verify_live_identity()
            if state.shutdown_intent is not None:
                self.shutdown()
                return False
            return True
        resumed = self.service.resume_from_checkpoint(checkpoint_path)
        if not resumed:
            resume_lease = getattr(self.service, "resume_live_service_lease", None)
            resumed = bool(
                resume_lease is not None
                and resume_lease(
                    expected_session_id=intent.session_id,
                    expected_event_id=intent.load_event_id,
                    watchdog_seconds=self.service_start_watchdog_seconds,
                )
            )
        if resumed:
            if state.model_service_start_count == 0:
                state = self._persist_started_lifecycle(
                    state,
                    intent,
                    service_pid=self.service.pid,
                    process_start_ticks=self.process_start_ticks(self.service.pid),
                    started_at=self._service_started_at(intent),
                )
            self._verify_live_identity()
            if state.shutdown_intent is not None:
                self.shutdown()
                return False
            return True
        self._recover_terminal_service(state, intent)
        return False

    def checkpoint(self, checkpoint_path: Path) -> None:
        self._verify_live_identity()
        self.service.write_resume_checkpoint(checkpoint_path)

    def detach_for_resume(self, checkpoint_path: Path) -> None:
        self._verify_live_identity()
        detach = getattr(self.service, "detach_for_controller_restart", None)
        if detach is None:
            raise CaseGpuAdapterError("case service cannot perform a durable restart handoff")
        detach(checkpoint_path)

    def shutdown(self) -> object | None:
        state = self._state()
        if state.model_service_shutdown_count == 1:
            if self.service.state is not ServiceState.STOPPED:
                raise CaseGpuAdapterError("case state claims shutdown while service is live")
            assert state.shutdown_receipt is not None
            return _parse_record(self.artifacts, state.shutdown_receipt, CaseGpuShutdownReceipt)
        if state.model_service_start_count == 0:
            intent = self._activation_intent(state)
            if intent is not None and self.service.state is ServiceState.READY:
                state = self._persist_started_lifecycle(
                    state,
                    intent,
                    service_pid=self.service.pid,
                    process_start_ticks=self.process_start_ticks(self.service.pid),
                    started_at=self._service_started_at(intent),
                )
            elif intent is not None and self._recover_terminal_service(state, intent):
                return None
            else:
                if self.service.state is not ServiceState.STOPPED:
                    raise CaseGpuAdapterError("unregistered case model service is live")
                return None
        state, _intent = self._ensure_shutdown_intent(state)
        result: object | None = None
        shutdown_error: BaseException | None = None
        try:
            result = self.service.shutdown()
        except BaseException as error:
            shutdown_error = error
        if self.service.state is not ServiceState.STOPPED:
            if shutdown_error is not None:
                raise shutdown_error
            raise CaseGpuAdapterError("case model service remained live after shutdown")
        session = self.artifacts.ledger.get_gpu_service_session(
            f"{self.plan.execution_id}-model-load"
        )
        stopped_at = self._now() if session is None else _parse_timestamp(session.ended_at)
        self._finalize_terminal_service(state, stopped_at=stopped_at)
        if shutdown_error is not None:
            raise shutdown_error
        return result

    def _window(self, window_id: str) -> CaseWindowExecutionPlan:
        matches = tuple(item for item in self.plan.windows if item.window_id == window_id)
        if len(matches) != 1:
            raise CaseGpuAdapterError("case call names an unknown window")
        return matches[0]

    def _prequery_receipt(self, job_id: str) -> CasePrequeryReceipt:
        controller_state = self.repository.load_state()
        if controller_state is None:
            raise CaseGpuAdapterError("case controller state is unavailable")
        resume = self.repository.load_resume(controller_state.resume_manifest)
        matches = tuple(
            item for item in resume.prequery_receipts if item.preparation_job_id == job_id
        )
        if len(matches) != 1:
            raise CaseGpuAdapterError("case prequery lineage is unavailable")
        return matches[0]

    def _run_config(
        self,
        *,
        call_id: str,
        condition: ConditionName,
        budgets: Any,
        repair: bool = False,
    ) -> RunConditionConfig:
        runtime = development_request_runtime(
            root=self.root,
            condition=condition,
            tokenizer_manifest=self.tokenizer_manifest,
            seed=self.plan.model_runtime.resolved_vllm_seed,
            repair=repair,
        )
        return RunConditionConfig(
            config_id=f"case-config-{call_id}{'-repair' if repair else ''}",
            condition=condition,
            budgets=budgets,
            maximum_input_tokens=(10752 if repair else 10240),
            maximum_output_tokens=(1536 if repair else 2048),
            repair_attempt_budget=budgets.repair_attempt_budget,
            seed_block=1,
            model_stack_hash=self.plan.model_runtime.content_hash,
            decoding_manifest_hash=runtime.decoding_manifest.content_hash,
            decoding_family_hash=runtime.decoding_manifest.comparison_family_hash,
            seed_manifest_hash=self.plan.model_runtime.seed_manifest_hash,
            resolved_seed=self.plan.model_runtime.resolved_vllm_seed,
            prompt_hash=runtime.prompt_hash,
            output_schema_hash=runtime.output_schema_hash,
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            capability_manifest_hash=runtime.capability_manifest.content_hash,
            validator_hash=self.plan.model_runtime.validator_hash,
            upper_ontology_hash=self.construction.upper_ontology.content_hash,
        )

    def _runtime_identifiers(self, condition: ConditionName, *, repair: bool = False) -> Any:
        return development_runtime_identifiers(
            root=self.root,
            condition=condition,
            tokenizer_manifest=self.tokenizer_manifest,
            model_name=self.plan.model_runtime.served_model_name,
            model_revision=self.plan.model_runtime.model_revision,
            seed=self.plan.model_runtime.resolved_vllm_seed,
            repair=repair,
        )

    def _completed(self, call_id: str, record_type: type[ImmutableRecord]) -> Any | None:
        state = self._state()
        reference = state.completed_receipts.get(call_id)
        if reference is None:
            return None
        return _parse_record(self.artifacts, reference, record_type)

    def _terminal_event(self, event_id: str) -> Any | None:
        events = tuple(
            item
            for item in self.artifacts.ledger.gpu_events_with_prefix(event_id)
            if item.event_id == event_id
        )
        if len(events) > 1:
            raise CaseGpuAdapterError("case GPU event identity is not unique")
        return None if not events else events[0]

    def _generation_from_model_call(
        self,
        *,
        call: CaseGpuCallSlot,
        guided: GuidedJSONRequest,
        job_id: str,
        attempt_id: str,
        event_id: str,
        repair: bool,
    ) -> tuple[GenerationResult | None, BaseException | None]:
        """Rebuild a terminal generation without allocating another GPU second."""

        label = "repair" if repair else "base"
        model_call_id = f"{self.plan.execution_id}-{call.call_id}-{label}-model"
        try:
            model_call = self.artifacts.ledger.get_model_call(model_call_id)
        except KeyError:
            return None, CaseGpuAdapterError(
                "terminal case GPU event has no recoverable model-call artifact"
            )
        if (
            model_call.job_id != job_id
            or model_call.attempt_id != attempt_id
            or model_call.gpu_event_id != event_id
            or model_call.request_hash != guided.request_hash
            or model_call.model_manifest_hash != self.model_manifest_hash
        ):
            raise CaseGpuAdapterError("terminal case model-call lineage changed")
        if model_call.response_artifact_hash is None:
            return None, CaseGpuAdapterError("terminal case model call retained no response")
        record = self.artifacts.ledger.get_artifact(model_call.response_artifact_hash)
        raw = self.artifacts.blobs.read_bytes(record, allow_restricted=True)
        if hashlib.sha256(raw).hexdigest() != model_call.response_artifact_hash:
            raise CaseGpuAdapterError("terminal case response artifact changed")
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, Mapping):
                raise TypeError
            if decoded.get("raw_response_available") is False:
                return None, CaseGpuAdapterError("terminal case call recorded no raw response")
            if "choices" in decoded:
                choice = cast(Any, decoded["choices"])[0]
                content = choice["message"]["content"]
                parsed = json.loads(content)
                if not isinstance(parsed, Mapping):
                    raise TypeError
                finish_reason = choice.get("finish_reason")
                service_request_id = decoded.get("id")
            else:
                # Unit services store the already-decoded guided object.
                parsed = decoded
                finish_reason = "stop"
                service_request_id = None
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            return None, CaseGpuAdapterError("terminal case response is not guided JSON")
        return (
            GenerationResult(
                request_id=guided.request_id,
                request_hash=guided.request_hash,
                response_sha256=model_call.response_artifact_hash,
                parsed_object=dict(parsed),
                raw_response=raw,
                prompt_tokens=model_call.prompt_tokens,
                completion_tokens=model_call.completion_tokens,
                finish_reason=finish_reason,
                service_request_id=service_request_id,
            ),
            None,
        )

    def _remaining_after(self, call: CaseGpuCallSlot, *, include_repair: bool) -> float:
        state = self._state()
        base = sum(
            item.watchdog_seconds
            for item in self.plan.gpu_call_slots
            if item.ordinal > call.ordinal and item.call_id not in state.completed_receipts
        )
        return float(base + (CASE_REQUIRED_NEXT_REPAIR_SECONDS if include_repair else 0))

    def _reserve_repair(self, call: CaseGpuCallSlot) -> str:
        state = self._state()
        reservation_id = f"{self.plan.execution_id}:{call.call_id}:repair"
        if reservation_id in state.consumed_repair_reservation_ids:
            return reservation_id
        tier = call.repair_reserve_class
        existing: set[str] = set()
        for event in self.artifacts.ledger.gpu_events():
            try:
                details = json.loads(event.details_json)
            except json.JSONDecodeError:
                continue
            observed_reservation = details.get(
                "reserve_reservation_id", details.get("reservation_id")
            )
            if details.get("reserve_call_class") == tier and isinstance(observed_reservation, str):
                existing.add(observed_reservation)
        limit = _LONG_RESERVE_LIMIT if tier == "reserve_long" else _STANDARD_RESERVE_LIMIT
        if len(existing) >= limit:
            raise CaseGpuAdapterError(f"{tier} is exhausted; invalid output remains ITT invalid")
        self._write_state(
            state,
            consumed_repair_reservation_ids=(
                *state.consumed_repair_reservation_ids,
                reservation_id,
            ),
        )
        return reservation_id

    def _record_model_call(
        self,
        *,
        call: CaseGpuCallSlot,
        guided: GuidedJSONRequest,
        job_id: str,
        attempt_id: str,
        event_id: str,
        response_hash: str | None,
        generated: GenerationResult | None,
        successful: bool,
        repair: bool,
        created_at: datetime,
    ) -> str:
        event = _gpu_event(self.artifacts.ledger, event_id)
        attempt_label = "repair" if repair else "base"
        model_call_id = f"{self.plan.execution_id}-{call.call_id}-{attempt_label}-model"
        role = (
            ModelCallRole.REPAIR
            if repair
            else ModelCallRole.PREBUILD
            if call.condition is ConditionName.C1_LLM_PRE
            else ModelCallRole.QUERY_TIME
        )
        retry_class = (
            RetryClass.LONG if call.repair_reserve_class == "reserve_long" else RetryClass.STANDARD
        )
        self.artifacts.ledger.record_model_call(
            model_call_id=model_call_id,
            job_id=job_id,
            attempt_id=attempt_id,
            gpu_event_id=event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=role,
            retry_class=retry_class,
            model_manifest_hash=self.model_manifest_hash,
            decoding_manifest_hash=guided.decoding.content_hash,
            request_hash=guided.request_hash,
            response_artifact_hash=response_hash,
            construction_unit_hash=canonical_sha256(
                {"plan": self.plan.content_hash, "slot": call.content_hash}
            ),
            served_context_count=1,
            prompt_tokens=0 if generated is None else generated.prompt_tokens,
            completion_tokens=0 if generated is None else generated.completion_tokens,
            allocated_gpu_seconds=event.allocated_seconds,
            successful=successful,
            created_at=created_at,
        )
        return model_call_id

    def _validate_generation(
        self,
        *,
        call: CaseGpuCallSlot,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        alias_manifest: ModelWireAliasManifest,
        generated: GenerationResult,
        raw_hash: str,
        event_id: str,
        stage_hash: str,
        query_access_hash: str | None,
        barrier_hash: str | None,
        config: RunConditionConfig,
        repair_parent_hash: str | None = None,
        repair_input: CaseRepairInput | None = None,
    ) -> tuple[ValidatedGeneration, BoundaryValidationReport, Any | None]:
        event = _gpu_event(self.artifacts.ledger, event_id)
        if (
            generated.request_id != guided.request_id
            or generated.request_hash != guided.request_hash
            or generated.response_sha256 != raw_hash
            or generated.finish_reason != "stop"
            or generated.prompt_tokens != guided.rendered_input_token_count
            or generated.completion_tokens > guided.decoding.maximum_output_tokens
        ):
            raise CaseGpuAdapterError("case vLLM response envelope changed")
        restored = restore_model_output_source_aliases(generated.parsed_object, alias_manifest)
        raw_draft = OntologyDraft.model_validate(restored)
        if (
            raw_draft.budget_accounting.input_tokens != 0
            or raw_draft.budget_accounting.output_tokens != 0
        ):
            raise CaseGpuAdapterError("case model token sentinels must remain zero")
        completed = _parse_timestamp(event.ended_at)
        started = _parse_timestamp(event.started_at)
        normalized = normalize_generation_metadata(
            raw_draft,
            decision_recorded_at=completed,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
        )
        evidence = (
            semantic.evidence
            if isinstance(semantic, PreconstructionRequest)
            else semantic.packet.evidence
        )
        boundary = validate_draft_structure(
            draft=normalized,
            upper_ontology=semantic.upper_ontology,
            evidence=evidence,
            budgets=semantic.budgets,
            capabilities=semantic.capabilities,
        )
        boundary.raise_for_errors()
        preservation = None
        if repair_input is not None:
            restored_invalid = restore_model_output_source_aliases(
                repair_input.invalid_draft,
                alias_manifest,
            )
            preservation = validate_repair_preservation(
                base_draft=cast(Mapping[str, object], restored_invalid),
                repaired_draft=cast(Mapping[str, object], restored),
                diagnosed_paths=tuple(item.path for item in repair_input.diagnostics),
            )
            preservation.raise_for_errors()
        validated_at = _strictly_after(self.clock, completed)
        validation = ValidationRecord(
            validation_id=f"case-validation-{call.call_id}-{raw_hash[:16]}",
            target_id=normalized.content_hash,
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.SUPPORTED,
            temporal_status=TemporalDeterminationStatus.VALID,
            commitment_status=CommitmentCheckStatus.VALID,
            diagnostics=("deterministic case boundary checks accepted",),
            validated_at=validated_at,
        )
        generation = ValidatedGeneration(
            generation_id=f"case-generation-{call.call_id}-{raw_hash[:16]}",
            condition=call.condition,
            request_hash=semantic.content_hash,
            raw_output_artifact_hash=raw_hash,
            raw_parsed_draft=raw_draft,
            draft=normalized,
            normalized_draft_hash=normalized.content_hash,
            stage_manifest_hash=stage_hash,
            query_access_event_hash=query_access_hash,
            prequery_barrier_hash=barrier_hash,
            packing_report_hash=guided.packing.content_hash,
            capability_manifest_hash=CapabilityManifest.for_condition(call.condition).content_hash,
            seed_manifest_hash=self.plan.model_runtime.seed_manifest_hash,
            prompt_hash=cast(str, config.prompt_hash),
            output_schema_hash=cast(str, config.output_schema_hash),
            decoding_manifest_hash=cast(str, config.decoding_manifest_hash),
            validator_hash=config.validator_hash,
            model_stack_hash=cast(str, config.model_stack_hash),
            seed=call.resolved_vllm_seed,
            input_tokens=generated.prompt_tokens,
            output_tokens=generated.completion_tokens,
            generation_started_at=started,
            generation_completed_at=completed,
            decision_recorded_at=completed,
            validation_records=(validation,),
            validator_report_hashes=tuple(
                dict.fromkeys(
                    (boundary.content_hash,)
                    if preservation is None
                    else (boundary.content_hash, preservation.content_hash)
                )
            ),
            validated_at=validated_at,
            repair_attempt=1 if repair_parent_hash is not None else 0,
            repair_parent_raw_output_hash=repair_parent_hash,
        )
        return generation, boundary, preservation

    def _diagnostics_for(
        self,
        error: BaseException,
        *,
        invalid: Mapping[str, object],
    ) -> tuple[CaseRepairDiagnostic, ...]:
        del invalid
        rows: list[CaseRepairDiagnostic] = []
        if isinstance(error, BoundaryValidationError):
            for index, item in enumerate(error.report.diagnostics):
                rows.append(
                    CaseRepairDiagnostic(
                        code=f"{item.code.value}-{index}",
                        path=item.path,
                        message=item.message,
                        related_ids=item.related_ids,
                    )
                )
        elif isinstance(error, ValidationError):
            for index, item in enumerate(error.errors(include_url=False)):
                location = ".".join(str(part) for part in item.get("loc", ())) or "$"
                rows.append(
                    CaseRepairDiagnostic(
                        code=f"schema-validation-{index}",
                        path=location,
                        message=f"The value at {location} violates the frozen output schema.",
                    )
                )
        elif isinstance(error, CaseGpuAdapterError):
            rows.append(
                CaseRepairDiagnostic(
                    code="runtime-envelope-0",
                    path="$",
                    message="The returned object violates the frozen response envelope.",
                )
            )
        if not rows:
            rows.append(
                CaseRepairDiagnostic(
                    code="deterministic-validation-0",
                    path="$",
                    message="The returned object failed a deterministic validation check.",
                )
            )
        # Bound an adversarially large error tree without adding semantic advice.
        return tuple(rows[:64])

    def _repair_semantic(
        self,
        semantic: PreconstructionRequest | ConstructionRequest,
        *,
        call: CaseGpuCallSlot,
        after: datetime,
    ) -> PreconstructionRequest | ConstructionRequest:
        values = semantic.model_dump(mode="python", exclude={"content_hash"})
        values.update(
            {
                "request_id": f"{semantic.request_id}-repair",
                "runtime": self._runtime_identifiers(call.condition, repair=True),
                "requested_at": _strictly_after(self.clock, after),
            }
        )
        return type(semantic).model_validate(values)

    def _record_validation(
        self,
        *,
        call: CaseGpuCallSlot,
        job_id: str,
        attempt_id: str,
        input_artifact_hash: str,
        accepted: bool,
        diagnostics: BoundaryValidationReport | Mapping[str, object],
        parent_validation_id: str | None = None,
        repair_attempt_id: str | None = None,
        created_at: datetime,
    ) -> str:
        diagnostics_ref = (
            _persist_record(
                self.artifacts,
                diagnostics,
                object_kind="case_boundary_validation_report",
                created_at=created_at,
            )
            if isinstance(diagnostics, ImmutableRecord)
            else _persist_mapping(
                self.artifacts,
                diagnostics,
                object_kind="case_validation_failure",
                created_at=created_at,
            )
        )
        suffix = "repair" if repair_attempt_id is not None else "base"
        validation_id = f"{self.plan.execution_id}-{call.call_id}-{suffix}-validation"
        self.artifacts.ledger.record_validation(
            validation_id=validation_id,
            job_id=job_id,
            attempt_id=attempt_id,
            input_artifact_hash=input_artifact_hash,
            validator_manifest_hash=self.plan.model_runtime.validator_hash,
            validation_status=(
                LedgerValidationStatus.ACCEPTED if accepted else LedgerValidationStatus.REJECTED
            ),
            evidence_support_status=(
                LedgerEvidenceSupportStatus.SUPPORTED
                if accepted
                else LedgerEvidenceSupportStatus.UNSUPPORTED
            ),
            temporal_status=(
                LedgerTemporalValidationStatus.VALID
                if accepted
                else LedgerTemporalValidationStatus.UNDERDETERMINED
            ),
            commitment_status=(
                LedgerCommitmentCheckStatus.VALID
                if accepted
                else LedgerCommitmentCheckStatus.INVALID
            ),
            diagnostics_artifact_hash=diagnostics_ref.artifact_hash,
            parent_validation_id=parent_validation_id,
            repair_attempt_id=repair_attempt_id,
            created_at=created_at,
        )
        return validation_id

    def _raw_or_failure_artifact(
        self,
        generated: GenerationResult | None,
        error: BaseException | None,
        *,
        created_at: datetime,
    ) -> str:
        if generated is not None:
            artifact = self.artifacts.put_bytes(
                generated.raw_response,
                media_type="application/json",
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=created_at,
            )
            if artifact.content_hash != generated.response_sha256:
                raise CaseGpuAdapterError("case response bytes differ from response digest")
            return artifact.content_hash
        assert error is not None
        artifact = self.artifacts.put_bytes(
            canonical_json(
                {
                    "execution_plan_hash": self.plan.content_hash,
                    "exception_type": type(error).__name__,
                    "raw_response_available": False,
                }
            ).encode("utf-8"),
            media_type="application/vnd.story-projection.case-call-failure+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        return artifact.content_hash

    def _invoke(
        self,
        *,
        call: CaseGpuCallSlot,
        guided: GuidedJSONRequest,
        job_id: str,
        attempt_id: str,
        event_id: str,
        repair: bool,
        remaining_required_seconds: float,
        reservation_id: str | None = None,
    ) -> tuple[GenerationResult | None, BaseException | None, Any]:
        self._verify_live_identity()
        self._admit_remaining(remaining_required_seconds)
        terminal_event = self._terminal_event(event_id)
        if terminal_event is not None:
            generated, error = self._generation_from_model_call(
                call=call,
                guided=guided,
                job_id=job_id,
                attempt_id=attempt_id,
                event_id=event_id,
                repair=repair,
            )
            return generated, error, terminal_event
        generated: GenerationResult | None = None
        error: BaseException | None = None
        details: dict[str, object] = {
            "phase": "bounded-first-novel-case-study",
            "call_id": call.call_id,
            "call_class": call.call_class,
            "ordinal": call.ordinal,
        }
        if reservation_id is not None:
            details.update(
                {
                    "reserve_reservation_id": reservation_id,
                    "reserve_call_class": call.repair_reserve_class,
                }
            )
        try:
            generated = self.service.generate(
                guided,
                event_id=event_id,
                watchdog_seconds=call.watchdog_seconds,
                repair=repair,
                job_id=job_id,
                attempt_id=attempt_id,
                remaining_required_seconds=remaining_required_seconds,
                accounting_details=details,
            )
        except BaseException as exc:
            error = exc
        event = _gpu_event(self.artifacts.ledger, event_id)
        if not math.isfinite(event.allocated_seconds) or event.allocated_seconds < 0:
            raise CaseGpuAdapterError("case GPU event duration is invalid")
        return generated, error, event

    def _execute_attempts(
        self,
        *,
        call: CaseGpuCallSlot,
        semantic: PreconstructionRequest | ConstructionRequest,
        config: RunConditionConfig,
        guided: GuidedJSONRequest,
        stage_hash: str,
        query_access_hash: str | None,
        barrier_hash: str | None,
    ) -> _AttemptResult:
        state = self._state()
        if state.active_call_id not in {None, call.call_id}:
            raise CaseGpuAdapterError("another case GPU call remains active")
        recovering_active = state.active_call_id == call.call_id
        created_at = self._now()
        job = self.artifacts.ledger.create_or_resume_job(
            {
                "execution_id": self.plan.execution_id,
                "execution_plan_hash": self.plan.content_hash,
                "call_slot_hash": call.content_hash,
            },
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=created_at,
        )
        self.artifacts.ledger.link_job_to_study(
            study_id=self.plan.execution_id,
            job_id=job.job_id,
            created_at=created_at,
        )
        attempt_id = f"{self.plan.execution_id}-{call.call_id}-base"
        self.artifacts.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=guided.request_hash,
            config_hash=config.content_hash,
            seed=call.resolved_vllm_seed,
            created_at=None if recovering_active else created_at,
        )
        if not recovering_active:
            self._write_state(state, active_call_id=call.call_id)
        event_id = f"{self.plan.execution_id}-{call.call_id}-base-gpu"
        generated, transport_error, event = self._invoke(
            call=call,
            guided=guided,
            job_id=job.job_id,
            attempt_id=attempt_id,
            event_id=event_id,
            repair=False,
            remaining_required_seconds=self._remaining_after(call, include_repair=True),
        )
        raw_hash = self._raw_or_failure_artifact(
            generated,
            transport_error,
            created_at=_parse_timestamp(event.ended_at),
        )
        alias_manifest = encode_development_semantic_request(semantic).alias_manifest
        base_error = transport_error
        generation = None
        boundary = None
        preservation = None
        if generated is not None:
            try:
                generation, boundary, preservation = self._validate_generation(
                    call=call,
                    semantic=semantic,
                    guided=guided,
                    alias_manifest=alias_manifest,
                    generated=generated,
                    raw_hash=raw_hash,
                    event_id=event_id,
                    stage_hash=stage_hash,
                    query_access_hash=query_access_hash,
                    barrier_hash=barrier_hash,
                    config=config,
                )
            except BaseException as exc:
                base_error = exc
        model_call_id = self._record_model_call(
            call=call,
            guided=guided,
            job_id=job.job_id,
            attempt_id=attempt_id,
            event_id=event_id,
            response_hash=raw_hash,
            generated=generated,
            successful=generation is not None,
            repair=False,
            created_at=_parse_timestamp(event.ended_at),
        )
        if generation is not None and boundary is not None:
            validation_id = self._record_validation(
                call=call,
                job_id=job.job_id,
                attempt_id=attempt_id,
                input_artifact_hash=raw_hash,
                accepted=True,
                diagnostics=boundary,
                created_at=generation.validated_at,
            )
            return _AttemptResult(
                outcome=RunOutcome.SUCCEEDED,
                job_id=job.job_id,
                base_attempt_id=attempt_id,
                base_model_call_id=model_call_id,
                base_event_id=event_id,
                base_raw_hash=raw_hash,
                validation_ids=(validation_id,),
                semantic=semantic,
                config=config,
                guided=guided,
                alias_manifest=alias_manifest,
                generation=generation,
                boundary=boundary,
            )
        assert base_error is not None
        invalid = (
            dict(generated.parsed_object)
            if generated is not None and isinstance(generated.parsed_object, Mapping)
            else {}
        )
        diagnostics = self._diagnostics_for(base_error, invalid=invalid)
        rejected_report = {
            "accepted": False,
            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
            "exception_type": type(base_error).__name__,
        }
        base_validation_id = self._record_validation(
            call=call,
            job_id=job.job_id,
            attempt_id=attempt_id,
            input_artifact_hash=raw_hash,
            accepted=False,
            diagnostics=rejected_report,
            created_at=_strictly_after(self.clock, _parse_timestamp(event.ended_at)),
        )
        if generated is None or not invalid or config.repair_attempt_budget != 1:
            failure_hash = self._persist_failure(
                call=call,
                attempt_id=attempt_id,
                error=base_error,
                raw_hash=raw_hash,
            )
            return _AttemptResult(
                outcome=_outcome(base_error),
                job_id=job.job_id,
                base_attempt_id=attempt_id,
                base_model_call_id=model_call_id,
                base_event_id=event_id,
                base_raw_hash=raw_hash,
                validation_ids=(base_validation_id,),
                semantic=semantic,
                config=config,
                guided=guided,
                alias_manifest=alias_manifest,
                generation=None,
                boundary=None,
                failure_lineage_hash=failure_hash,
            )

        reservation_id = self._reserve_repair(call)
        repair_semantic = self._repair_semantic(
            semantic,
            call=call,
            after=_parse_timestamp(event.ended_at),
        )
        repair_input = CaseRepairInput(
            parent_attempt_id=attempt_id,
            semantic_request_hash=repair_semantic.content_hash,
            parent_raw_output_hash=raw_hash,
            invalid_draft=invalid,
            diagnostics=diagnostics,
            created_at=_strictly_after(self.clock, _parse_timestamp(event.ended_at)),
        )
        repair_ref = _persist_record(
            self.artifacts,
            repair_input,
            object_kind="case_repair_input",
            created_at=repair_input.created_at,
        )
        repair_guided = _repair_guided_request(
            root=self.root,
            call_id=call.call_id,
            semantic_request=repair_semantic,
            repair_input=repair_input,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            model_name=self.plan.model_runtime.served_model_name,
            model_revision=self.plan.model_runtime.model_revision,
            seed=call.resolved_vllm_seed,
        )
        repair_config = self._run_config(
            call_id=call.call_id,
            condition=call.condition,
            budgets=semantic.budgets,
            repair=True,
        )
        repair_attempt_id = f"{self.plan.execution_id}-{call.call_id}-repair"
        self.artifacts.ledger.record_attempt(
            attempt_id=repair_attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=attempt_id,
            input_hash=repair_guided.request_hash,
            config_hash=repair_config.content_hash,
            seed=call.resolved_vllm_seed,
            created_at=None if recovering_active else repair_input.created_at,
        )
        repair_event_id = f"{self.plan.execution_id}-{call.call_id}-repair-gpu"
        repaired, repair_error, repair_event = self._invoke(
            call=call,
            guided=repair_guided,
            job_id=job.job_id,
            attempt_id=repair_attempt_id,
            event_id=repair_event_id,
            repair=True,
            remaining_required_seconds=self._remaining_after(call, include_repair=True),
            reservation_id=reservation_id,
        )
        repair_raw_hash = self._raw_or_failure_artifact(
            repaired,
            repair_error,
            created_at=_parse_timestamp(repair_event.ended_at),
        )
        repair_aliases = encode_development_semantic_request(repair_semantic).alias_manifest
        repair_generation = None
        repair_boundary = None
        if repaired is not None:
            try:
                repair_generation, repair_boundary, preservation = self._validate_generation(
                    call=call,
                    semantic=repair_semantic,
                    guided=repair_guided,
                    alias_manifest=repair_aliases,
                    generated=repaired,
                    raw_hash=repair_raw_hash,
                    event_id=repair_event_id,
                    stage_hash=stage_hash,
                    query_access_hash=query_access_hash,
                    barrier_hash=barrier_hash,
                    config=repair_config,
                    repair_parent_hash=raw_hash,
                    repair_input=repair_input,
                )
            except BaseException as exc:
                repair_error = exc
        repair_model_call_id = self._record_model_call(
            call=call,
            guided=repair_guided,
            job_id=job.job_id,
            attempt_id=repair_attempt_id,
            event_id=repair_event_id,
            response_hash=repair_raw_hash,
            generated=repaired,
            successful=repair_generation is not None,
            repair=True,
            created_at=_parse_timestamp(repair_event.ended_at),
        )
        repair_validation_id = self._record_validation(
            call=call,
            job_id=job.job_id,
            attempt_id=repair_attempt_id,
            input_artifact_hash=repair_raw_hash,
            accepted=repair_generation is not None,
            diagnostics=(
                cast(BoundaryValidationReport, repair_boundary)
                if repair_generation is not None
                else {
                    "accepted": False,
                    "exception_type": type(repair_error).__name__,
                    "parent_validation_id": base_validation_id,
                    "diagnostics": [
                        item.model_dump(mode="json")
                        for item in self._diagnostics_for(
                            cast(BaseException, repair_error),
                            invalid=(
                                dict(repaired.parsed_object)
                                if repaired is not None
                                and isinstance(repaired.parsed_object, Mapping)
                                else {}
                            ),
                        )
                    ],
                }
            ),
            parent_validation_id=base_validation_id,
            repair_attempt_id=repair_attempt_id,
            created_at=_strictly_after(self.clock, _parse_timestamp(repair_event.ended_at)),
        )
        preservation_ref = (
            None
            if preservation is None
            else _persist_record(
                self.artifacts,
                preservation,
                object_kind="case_repair_preservation_report",
                created_at=_strictly_after(self.clock, _parse_timestamp(repair_event.ended_at)),
            )
        )
        if repair_generation is not None and repair_boundary is not None:
            return _AttemptResult(
                outcome=RunOutcome.SUCCEEDED,
                job_id=job.job_id,
                base_attempt_id=attempt_id,
                base_model_call_id=model_call_id,
                base_event_id=event_id,
                base_raw_hash=raw_hash,
                validation_ids=(base_validation_id, repair_validation_id),
                semantic=repair_semantic,
                config=repair_config,
                guided=repair_guided,
                alias_manifest=repair_aliases,
                generation=repair_generation,
                boundary=repair_boundary,
                repair_attempt_id=repair_attempt_id,
                repair_model_call_id=repair_model_call_id,
                repair_event_id=repair_event_id,
                repair_input_reference=repair_ref,
                repair_raw_hash=repair_raw_hash,
                repair_preservation_reference=preservation_ref,
            )
        assert repair_error is not None
        failure_hash = self._persist_failure(
            call=call,
            attempt_id=repair_attempt_id,
            error=repair_error,
            raw_hash=repair_raw_hash,
        )
        return _AttemptResult(
            outcome=_outcome(repair_error),
            job_id=job.job_id,
            base_attempt_id=attempt_id,
            base_model_call_id=model_call_id,
            base_event_id=event_id,
            base_raw_hash=raw_hash,
            validation_ids=(base_validation_id, repair_validation_id),
            semantic=repair_semantic,
            config=repair_config,
            guided=repair_guided,
            alias_manifest=repair_aliases,
            generation=None,
            boundary=None,
            repair_attempt_id=repair_attempt_id,
            repair_model_call_id=repair_model_call_id,
            repair_event_id=repair_event_id,
            repair_input_reference=repair_ref,
            repair_raw_hash=repair_raw_hash,
            failure_lineage_hash=failure_hash,
        )

    def _persist_failure(
        self,
        *,
        call: CaseGpuCallSlot,
        attempt_id: str,
        error: BaseException,
        raw_hash: str,
    ) -> str:
        created_at = self._now()
        reference = _persist_mapping(
            self.artifacts,
            {
                "execution_plan_hash": self.plan.content_hash,
                "call_slot_hash": call.content_hash,
                "attempt_id": attempt_id,
                "exception_type": type(error).__name__,
                "raw_artifact_hash": raw_hash,
            },
            object_kind="case_gpu_failure_lineage",
            created_at=created_at,
        )
        self.artifacts.ledger.record_failure(
            attempt_id=attempt_id,
            failure_kind=_failure_kind(error),
            message="case model attempt did not pass deterministic validation",
            details={
                "call_id": call.call_id,
                "execution_plan_hash": self.plan.content_hash,
            },
            artifact_hash=reference.artifact_hash,
            occurred_at=created_at,
        )
        return reference.logical_content_hash

    def _request_references(
        self,
        *,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
        alias_manifest: ModelWireAliasManifest,
        created_at: datetime,
    ) -> dict[str, CaseArtifactReference | str]:
        return {
            # Novel-bearing semantic and rendered requests remain in memory.
            # Their exact hashes bind the ledger without duplicating source text.
            "semantic": semantic.content_hash,
            "alias": _persist_record(
                self.artifacts,
                alias_manifest,
                object_kind="case_model_wire_alias_manifest",
                created_at=created_at,
            ),
            "rendered": guided.request_hash,
            "packing": _persist_record(
                self.artifacts,
                guided.packing,
                object_kind="case_packing_report",
                created_at=created_at,
            ),
            "decoding": _persist_record(
                self.artifacts,
                guided.decoding,
                object_kind="case_decoding_manifest",
                created_at=created_at,
            ),
            "capability": _persist_record(
                self.artifacts,
                CapabilityManifest.for_condition(semantic.condition),
                object_kind="case_capability_manifest",
                created_at=created_at,
            ),
            "config": _persist_record(
                self.artifacts,
                config,
                object_kind="case_run_condition_config",
                created_at=created_at,
            ),
        }

    def _persist_gpu_audit(
        self,
        *,
        call: CaseGpuCallSlot,
        base_semantic: PreconstructionRequest | ConstructionRequest,
        base_guided: GuidedJSONRequest,
        base_config: RunConditionConfig,
        attempt: _AttemptResult,
        condition_result_reference: CaseArtifactReference | None,
        query_access: CaseQueryAccessReceipt | None,
        prequery_barrier: CasePrequeryBarrierReceipt | None,
        request_envelope_hash: str,
        stage_manifest_hash: str,
        completed_receipt: CasePrequeryReceipt | CaseOutputReceipt,
    ) -> None:
        created_at = completed_receipt.completed_at
        base_alias = encode_development_semantic_request(base_semantic).alias_manifest
        refs = self._request_references(
            semantic=base_semantic,
            guided=base_guided,
            config=base_config,
            alias_manifest=base_alias,
            created_at=created_at,
        )
        generation_ref = (
            None
            if attempt.generation is None
            else _persist_record(
                self.artifacts,
                attempt.generation,
                object_kind="case_validated_generation",
                created_at=created_at,
            )
        )
        failure_hash = attempt.failure_lineage_hash
        if failure_hash is None and attempt.outcome is not RunOutcome.SUCCEEDED:
            raise CaseGpuAdapterError("failed case attempt lacks failure lineage")
        state = self._state()
        if state.service_identity is None:
            raise CaseGpuAdapterError("case call audit lacks the owned service identity")
        query_access_ref = (
            None
            if query_access is None
            else _persist_record(
                self.artifacts,
                query_access,
                object_kind="case_query_access_receipt",
                created_at=created_at,
            )
        )
        barrier_ref = (
            None
            if prequery_barrier is None
            else _persist_record(
                self.artifacts,
                prequery_barrier,
                object_kind="case_prequery_barrier_receipt",
                created_at=created_at,
            )
        )
        repair_refs = None
        if attempt.repair_attempt_id is not None:
            repair_refs = self._request_references(
                semantic=attempt.semantic,
                guided=attempt.guided,
                config=attempt.config,
                alias_manifest=attempt.alias_manifest,
                created_at=created_at,
            )
        audit = CaseGpuCallAuditReceipt(
            receipt_id=f"case-gpu-audit-{call.call_id}",
            execution_plan_hash=self.plan.content_hash,
            call_slot_hash=call.content_hash,
            call_id=call.call_id,
            condition=call.condition,
            job_id=attempt.job_id,
            base_attempt_id=attempt.base_attempt_id,
            base_model_call_id=attempt.base_model_call_id,
            base_gpu_event_id=attempt.base_event_id,
            service_identity=state.service_identity,
            request_envelope_hash=request_envelope_hash,
            stage_manifest_hash=stage_manifest_hash,
            semantic_request_hash=cast(str, refs["semantic"]),
            wire_alias_manifest=cast(CaseArtifactReference, refs["alias"]),
            rendered_request_hash=cast(str, refs["rendered"]),
            packing_report=cast(CaseArtifactReference, refs["packing"]),
            decoding_manifest=cast(CaseArtifactReference, refs["decoding"]),
            capability_manifest=cast(CaseArtifactReference, refs["capability"]),
            run_configuration=cast(CaseArtifactReference, refs["config"]),
            query_access_receipt=query_access_ref,
            prequery_barrier=barrier_ref,
            raw_base_artifact_hash=attempt.base_raw_hash,
            validation_ids=attempt.validation_ids,
            terminal_outcome=attempt.outcome,
            terminal_receipt_hash=completed_receipt.content_hash,
            validated_generation=generation_ref,
            condition_result=condition_result_reference,
            repair_attempt_id=attempt.repair_attempt_id,
            repair_model_call_id=attempt.repair_model_call_id,
            repair_gpu_event_id=attempt.repair_event_id,
            repair_input=attempt.repair_input_reference,
            repair_semantic_request_hash=(
                None if repair_refs is None else cast(str, repair_refs["semantic"])
            ),
            repair_wire_alias_manifest=(
                None if repair_refs is None else cast(CaseArtifactReference, repair_refs["alias"])
            ),
            repair_rendered_request_hash=(
                None if repair_refs is None else cast(str, repair_refs["rendered"])
            ),
            repair_packing_report=(
                None if repair_refs is None else cast(CaseArtifactReference, repair_refs["packing"])
            ),
            repair_decoding_manifest=(
                None
                if repair_refs is None
                else cast(CaseArtifactReference, repair_refs["decoding"])
            ),
            repair_run_configuration=(
                None if repair_refs is None else cast(CaseArtifactReference, repair_refs["config"])
            ),
            raw_repair_artifact_hash=attempt.repair_raw_hash,
            repair_preservation_report=attempt.repair_preservation_reference,
            failure_lineage_hash=failure_hash,
            created_at=created_at,
        )
        audit_ref = _persist_record(
            self.artifacts,
            audit,
            object_kind="case_gpu_call_audit",
            created_at=created_at,
        )
        receipt_ref = _persist_record(
            self.artifacts,
            completed_receipt,
            object_kind=(
                "case_prequery_receipt"
                if isinstance(completed_receipt, CasePrequeryReceipt)
                else "case_output_receipt"
            ),
            created_at=created_at,
        )
        state = self._state()
        if state.active_call_id != call.call_id:
            raise CaseGpuAdapterError("case call lost its durable active marker")
        receipts = dict(state.completed_receipts)
        audits = dict(state.call_audits)
        if call.call_id in receipts or call.call_id in audits:
            raise CaseGpuAdapterError("case call completion index is not append-only")
        receipts[call.call_id] = receipt_ref
        audits[call.call_id] = audit_ref
        preparations = dict(state.c1_preparations)
        if (
            isinstance(completed_receipt, CasePrequeryReceipt)
            and condition_result_reference is not None
            and completed_receipt.terminal_outcome is RunOutcome.SUCCEEDED
        ):
            preparations[completed_receipt.window_id] = condition_result_reference
        self._write_state(
            state,
            completed_receipts=receipts,
            call_audits=audits,
            c1_preparations=preparations,
            active_call_id=None,
        )

    def _assert_base_request_runtime(
        self,
        *,
        call: CaseGpuCallSlot,
        semantic: PreconstructionRequest | ConstructionRequest,
        guided: GuidedJSONRequest,
        config: RunConditionConfig,
    ) -> None:
        if (
            call.runtime_binding_hash != self.plan.model_runtime.content_hash
            or guided.model_name != self.plan.model_runtime.served_model_name
            or guided.decoding.seed != call.resolved_vllm_seed
            or guided.rendered_input_token_count > self.plan.model_runtime.maximum_input_tokens
            or guided.decoding.maximum_output_tokens
            != self.plan.model_runtime.maximum_output_tokens
            or config.decoding_manifest_hash != guided.decoding.content_hash
            or config.decoding_family_hash != guided.decoding.comparison_family_hash
            or config.output_schema_hash != semantic.runtime.output_schema_hash
            or config.capability_manifest_hash
            != CapabilityManifest.for_condition(call.condition).content_hash
        ):
            raise CaseGpuAdapterError("case request differs from the frozen selected stack")

    def preconstruct_c1(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC1RequestEnvelope,
        protected_packet: WindowEvidenceBundle,
    ) -> CasePrequeryReceipt:
        recovered = self._completed(call.call_id, CasePrequeryReceipt)
        if recovered is not None:
            return cast(CasePrequeryReceipt, recovered)
        self._verify_live_identity()
        if self._prepared is None:
            raise CaseGpuAdapterError("case C1 call lacks the complete packing preflight")
        if call.role is not CaseGpuCallRole.C1_WINDOW_PRECONSTRUCTION:
            raise CaseGpuAdapterError("case C1 adapter received another call role")
        snapshot = protected_packet.snapshot_assembly.snapshot
        packet = protected_packet.packet
        window = self._window(cast(str, call.window_id))
        if (
            envelope.execution_plan_hash != self.plan.content_hash
            or envelope.call_slot_hash != call.content_hash
            or envelope.evidence_binding_hash != window.evidence_binding_hash
            or envelope.snapshot_hash != snapshot.content_hash
            or envelope.packet_hash != packet.content_hash
            or envelope.query_context_supplied is not False
        ):
            raise CaseGpuAdapterError("case C1 request envelope changed")
        semantic = cast(
            PreconstructionRequest,
            self._prepared.semantic_requests.get(call.call_id),
        )
        guided = self._prepared.guided_requests.get(call.call_id)
        if not isinstance(semantic, PreconstructionRequest) or guided is None:
            raise CaseGpuAdapterError("case C1 preflight omitted its request")
        config = self._run_config(
            call_id=call.call_id,
            condition=ConditionName.C1_LLM_PRE,
            budgets=semantic.budgets,
        )
        self._assert_base_request_runtime(
            call=call,
            semantic=semantic,
            guided=guided,
            config=config,
        )
        attempt = self._execute_attempts(
            call=call,
            semantic=semantic,
            config=config,
            guided=guided,
            stage_hash=envelope.content_hash,
            query_access_hash=None,
            barrier_hash=None,
        )
        started_at = _parse_timestamp(
            _gpu_event(self.artifacts.ledger, attempt.base_event_id).started_at
        )
        preparation = None
        preparation_ref = None
        completed_at = self._now()
        if attempt.generation is not None:
            if not isinstance(attempt.semantic, PreconstructionRequest):
                raise CaseGpuAdapterError("case C1 terminal request is not preconstruction")
            sealed_at = _strictly_after(self.clock, attempt.generation.validated_at)
            preparation = seal_c1_preconstruction(
                request=attempt.semantic,
                generation=attempt.generation,
                seed_block=1,
                sealed_at=sealed_at,
            )
            preparation_ref = _persist_record(
                self.artifacts,
                preparation,
                object_kind="case_c1_condition_preparation",
                created_at=sealed_at,
            )
            completed_at = sealed_at
        repair_used = attempt.repair_attempt_id is not None
        receipt = CasePrequeryReceipt(
            receipt_id=f"receipt-{call.call_id}",
            execution_plan_hash=self.plan.content_hash,
            preparation_job_id=call.call_id,
            kind=CasePrequeryKind.C1_WINDOW_PRECONSTRUCTION,
            window_id=window.window_id,
            condition=ConditionName.C1_LLM_PRE,
            evidence_binding_hash=window.evidence_binding_hash,
            snapshot_hash=snapshot.content_hash,
            backend="vllm_gpu",
            terminal_outcome=attempt.outcome,
            started_at=started_at,
            completed_at=_strictly_after(self.clock, completed_at),
            base_attempt_artifact_hash=attempt.base_raw_hash,
            repair_attempt_count=1 if repair_used else 0,
            repair_attempt_artifact_hash=attempt.repair_raw_hash,
            repair_parent_artifact_hash=(attempt.base_raw_hash if repair_used else None),
            repair_reserve_class=("reserve_long" if repair_used else None),
            construction_seal=(
                None
                if preparation is None
                else cast(Any, preparation.sealed_preontology).construction_seal
            ),
            failure_lineage_hash=attempt.failure_lineage_hash,
        )
        self._persist_gpu_audit(
            call=call,
            base_semantic=semantic,
            base_guided=guided,
            base_config=config,
            attempt=attempt,
            condition_result_reference=preparation_ref,
            query_access=None,
            prequery_barrier=None,
            request_envelope_hash=envelope.content_hash,
            stage_manifest_hash=envelope.content_hash,
            completed_receipt=receipt,
        )
        return receipt

    def _query_proxy(
        self,
        *,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        context: QueryContext,
        packet: EvidencePacket,
    ) -> _CaseQueryAccessProxy:
        if (
            query_access.execution_plan_hash != self.plan.content_hash
            or query_access.context_hash != context.content_hash
            or query_access.prequery_barrier_hash != prequery_barrier.content_hash
            or query_access.packet_hash != packet.content_hash
            or query_access.accessed_at < context.revealed_at
        ):
            raise CaseGpuAdapterError("case query access lineage changed")
        return _CaseQueryAccessProxy(
            content_hash=query_access.content_hash,
            execution_id=self.plan.execution_id,
            query_context_hash=context.content_hash,
            model_visible_query_hash=to_model_visible_query(context).content_hash,
            snapshot_hash=query_access.snapshot_hash,
            stage_manifest_hash=_case_query_stage_hash(self.plan, context),
            prequery_barrier_hash=prequery_barrier.content_hash,
            packet_hash=packet.content_hash,
            registered_revealed_at=context.revealed_at,
            accessed_at=query_access.accessed_at,
        )

    def _query_inputs(
        self,
        *,
        condition: ConditionName,
        preparation: ConditionPreparation,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        packet: EvidencePacket,
        context: QueryContext,
        config: RunConditionConfig,
        snapshot: object,
    ) -> _CaseProduceInputs:
        proxy = self._query_proxy(
            query_access=query_access,
            prequery_barrier=prequery_barrier,
            context=context,
            packet=packet,
        )
        if preparation.condition is not condition:
            raise CaseGpuAdapterError("case preparation and query condition differ")
        return _CaseProduceInputs(
            preparation=preparation,
            snapshot=cast(Any, snapshot),
            packet=packet,
            context=context,
            query_access=proxy,
            prequery_barrier=prequery_barrier,
            query_processing_started_at=_strictly_after(self.clock, query_access.accessed_at),
            packet_materialization=None,
            upper_ontology=self.construction.upper_ontology,
            run_config=config,
        )

    def _persist_cpu_projection_state(
        self,
        *,
        job_id: str,
        receipt: CaseOutputReceipt,
    ) -> None:
        reference = _persist_record(
            self.artifacts,
            receipt,
            object_kind="case_output_receipt",
            created_at=receipt.completed_at,
        )
        state = self._state()
        values = dict(state.c1_projection_receipts)
        existing = values.get(job_id)
        if existing is not None and existing != reference:
            raise CaseGpuAdapterError("case C1 projection changed during resume")
        if existing is None:
            values[job_id] = reference
            self._write_state(state, c1_projection_receipts=values)

    def project_c1(
        self,
        *,
        window: CaseWindowExecutionPlan,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        protected_packet: WindowEvidenceBundle,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt:
        index = window.context_ids.index(query_access.context_id)
        job_id = window.c1_projection_job_ids[index]
        state = self._state()
        prior = state.c1_projection_receipts.get(job_id)
        if prior is not None:
            return cast(
                CaseOutputReceipt,
                _parse_record(self.artifacts, prior, CaseOutputReceipt),
            )
        source_receipt = self._prequery_receipt(window.c1_preconstruction_call_id)
        packet = protected_packet.packet
        snapshot = protected_packet.snapshot_assembly.snapshot
        if (
            source_receipt.execution_plan_hash != self.plan.content_hash
            or query_access.window_id != window.window_id
            or query_access.packet_hash != packet.content_hash
            or query_access.snapshot_hash != snapshot.content_hash
        ):
            raise CaseGpuAdapterError("case C1 fixed projection lineage changed")
        if source_receipt.terminal_outcome is not RunOutcome.SUCCEEDED:
            completed = _strictly_after(self.clock, query_access.accessed_at)
            failure = _persist_mapping(
                self.artifacts,
                {
                    "dependency_preparation_receipt_hash": source_receipt.content_hash,
                    "projection_job_id": job_id,
                    "reason": "c1_preconstruction_not_successful",
                },
                object_kind="case_c1_projection_dependency_failure",
                created_at=completed,
            )
            receipt = CaseOutputReceipt(
                receipt_id=f"receipt-{job_id}",
                execution_plan_hash=self.plan.content_hash,
                projection_job_id=job_id,
                condition=ConditionName.C1_LLM_PRE,
                context_id=protected_query.context_id,
                window_id=window.window_id,
                backend="fixed_projection_cpu",
                query_access_receipt_hash=query_access.content_hash,
                preparation_receipt_hash=source_receipt.content_hash,
                evidence_binding_hash=window.evidence_binding_hash,
                packet_equality_group_id=window.packet_equality_group_id,
                snapshot_hash=snapshot.content_hash,
                packet_hash=packet.content_hash,
                terminal_outcome=RunOutcome.FAILED,
                failure_lineage_hash=failure.logical_content_hash,
                query_accessed_at=query_access.accessed_at,
                completed_at=completed,
                operational_only=False,
                causal_comparison_eligible=True,
            )
            self._persist_cpu_projection_state(job_id=job_id, receipt=receipt)
            return receipt
        preparation_ref = state.c1_preparations.get(window.window_id)
        if preparation_ref is None:
            raise CaseGpuAdapterError("successful case C1 receipt lacks its sealed preparation")
        preparation = cast(
            ConditionPreparation,
            _parse_record(self.artifacts, preparation_ref, ConditionPreparation),
        )
        assert preparation.sealed_preontology is not None
        if source_receipt.construction_seal != preparation.sealed_preontology.construction_seal:
            raise CaseGpuAdapterError("case C1 preparation and prequery seal differ")
        config = self._run_config(
            call_id=job_id,
            condition=ConditionName.C1_LLM_PRE,
            budgets=protected_query.budgets,
        )
        inputs = self._query_inputs(
            condition=ConditionName.C1_LLM_PRE,
            preparation=preparation,
            query_access=query_access,
            prequery_barrier=prequery_barrier,
            packet=packet,
            context=protected_query,
            config=config,
            snapshot=snapshot,
        )
        projection = project_sealed_c1(
            preparation.sealed_preontology,
            inputs,  # type: ignore[arg-type]
        )
        projection_artifact = self.artifacts.put_bytes(
            canonical_json(projection).encode("utf-8"),
            media_type="application/vnd.story-projection.ontology-projection+json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=_strictly_after(self.clock, query_access.accessed_at),
        )
        completed = _strictly_after(self.clock, query_access.accessed_at)
        receipt = CaseOutputReceipt(
            receipt_id=f"receipt-{job_id}",
            execution_plan_hash=self.plan.content_hash,
            projection_job_id=job_id,
            condition=ConditionName.C1_LLM_PRE,
            context_id=protected_query.context_id,
            window_id=window.window_id,
            backend="fixed_projection_cpu",
            query_access_receipt_hash=query_access.content_hash,
            preparation_receipt_hash=source_receipt.content_hash,
            evidence_binding_hash=window.evidence_binding_hash,
            packet_equality_group_id=window.packet_equality_group_id,
            snapshot_hash=snapshot.content_hash,
            packet_hash=packet.content_hash,
            source_construction_seal_hash=(
                preparation.sealed_preontology.construction_seal.content_hash
            ),
            terminal_outcome=RunOutcome.SUCCEEDED,
            projection_artifact_hash=projection_artifact.content_hash,
            query_accessed_at=query_access.accessed_at,
            completed_at=completed,
            operational_only=False,
            causal_comparison_eligible=True,
        )
        self._persist_cpu_projection_state(job_id=job_id, receipt=receipt)
        return receipt

    def _c2_preparation(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC2RequestEnvelope,
    ) -> tuple[CasePrequeryReceipt, ConditionPreparation]:
        job_id = (
            self.plan.operational.c2_empty_inventory_job_id
            if call.operational_only
            else self._window(cast(str, call.window_id)).c2_empty_inventory_job_id
        )
        receipt = self._prequery_receipt(job_id)
        inventory = receipt.empty_inventory
        if (
            inventory is None
            or receipt.content_hash == envelope.query_access_receipt_hash
            or inventory.content_hash != envelope.empty_inventory_hash
            or inventory.snapshot_hash != receipt.snapshot_hash
        ):
            raise CaseGpuAdapterError("case C2 empty-inventory lineage changed")
        return (
            receipt,
            ConditionPreparation(
                preparation_id=f"case-preparation-{job_id}",
                condition=ConditionName.C2_LLM_QUERY,
                snapshot_hash=receipt.snapshot_hash,
                completed_at=inventory.recorded_at,
                empty_inventory=inventory,
            ),
        )

    def construct_c2(
        self,
        *,
        call: CaseGpuCallSlot,
        envelope: CaseC2RequestEnvelope,
        query_access: CaseQueryAccessReceipt,
        prequery_barrier: CasePrequeryBarrierReceipt,
        protected_packet: EvidencePacket,
        protected_query: QueryContext,
    ) -> CaseOutputReceipt:
        recovered = self._completed(call.call_id, CaseOutputReceipt)
        if recovered is not None:
            return cast(CaseOutputReceipt, recovered)
        self._verify_live_identity()
        if call.role not in {
            CaseGpuCallRole.C2_BOUNDED_CONSTRUCTION,
            CaseGpuCallRole.C2_FULL_INDEX_OPERATIONAL,
        }:
            raise CaseGpuAdapterError("case C2 adapter received another call role")
        if (
            envelope.execution_plan_hash != self.plan.content_hash
            or envelope.call_slot_hash != call.content_hash
            or envelope.query_access_receipt_hash != query_access.content_hash
            or envelope.packet_hash != protected_packet.content_hash
            or query_access.context_id != protected_query.context_id
            or call.context_id != protected_query.context_id
            or envelope.inherited_ontology_hash is not None
        ):
            raise CaseGpuAdapterError("case C2 request envelope changed")
        window = None if call.operational_only else self._window(cast(str, call.window_id))
        expected_binding = (
            self.plan.operational.full_index_binding_hash
            if call.operational_only
            else cast(CaseWindowExecutionPlan, window).evidence_binding_hash
        )
        if envelope.evidence_binding_hash != expected_binding:
            raise CaseGpuAdapterError("case C2 evidence binding changed")
        preparation_receipt, preparation = self._c2_preparation(
            call=call,
            envelope=envelope,
        )
        snapshot: object
        if call.operational_only:
            snapshot = _OperationalSnapshot(query_access.snapshot_hash)
        else:
            bundle = self._bounded_packets.get(cast(str, call.window_id))
            if bundle is None:
                raise CaseGpuAdapterError("bounded C2 call lacks its preflight snapshot")
            snapshot = bundle.snapshot_assembly.snapshot
            if protected_packet != bundle.packet:
                raise CaseGpuAdapterError("bounded C2 packet differs from C1 preflight evidence")
        config = self._run_config(
            call_id=call.call_id,
            condition=ConditionName.C2_LLM_QUERY,
            budgets=protected_query.budgets,
        )
        inputs = self._query_inputs(
            condition=ConditionName.C2_LLM_QUERY,
            preparation=preparation,
            query_access=query_access,
            prequery_barrier=prequery_barrier,
            packet=protected_packet,
            context=protected_query,
            config=config,
            snapshot=snapshot,
        )
        runtime = self._runtime_identifiers(ConditionName.C2_LLM_QUERY)
        requested_at = _strictly_after(self.clock, query_access.accessed_at)
        semantic = build_c2_construction_request(
            inputs,  # type: ignore[arg-type]
            runtime=runtime,
            requested_at=requested_at,
        )
        guided = build_development_guided_request(
            root=self.root,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            model_name=self.plan.model_runtime.served_model_name,
            model_revision=self.plan.model_runtime.model_revision,
            seed=call.resolved_vllm_seed,
        )
        self._assert_base_request_runtime(
            call=call,
            semantic=semantic,
            guided=guided,
            config=config,
        )
        attempt = self._execute_attempts(
            call=call,
            semantic=semantic,
            config=config,
            guided=guided,
            stage_hash=inputs.query_access.stage_manifest_hash,
            query_access_hash=query_access.content_hash,
            barrier_hash=prequery_barrier.content_hash,
        )
        completed_at = self._now()
        condition_result = None
        condition_ref = None
        projection_hash = None
        certificate = None
        if attempt.generation is not None:
            if not isinstance(attempt.semantic, ConstructionRequest):
                raise CaseGpuAdapterError("case C2 terminal request is not query construction")
            # A terminal repair carries its own exact decoding configuration.
            terminal_inputs = replace(inputs, run_config=attempt.config)
            condition_result = finalize_c2_draft(
                terminal_inputs,  # type: ignore[arg-type]
                request=attempt.semantic,
                generation=attempt.generation,
            )
            condition_ref = _persist_record(
                self.artifacts,
                condition_result,
                object_kind="case_c2_condition_attempt",
                created_at=attempt.generation.validated_at,
            )
            assert condition_result.projection is not None
            projection_artifact = self.artifacts.put_bytes(
                canonical_json(condition_result.projection).encode("utf-8"),
                media_type="application/vnd.story-projection.ontology-projection+json",
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=attempt.generation.validated_at,
            )
            projection_hash = projection_artifact.content_hash
            certificate = condition_result.projection.construction_certificate
            completed_at = _strictly_after(self.clock, attempt.generation.validated_at)
        repair_used = attempt.repair_attempt_id is not None
        packet_group = (
            "case-full-index-operational-packet"
            if call.operational_only
            else cast(CaseWindowExecutionPlan, window).packet_equality_group_id
        )
        terminal_raw = attempt.repair_raw_hash if repair_used else attempt.base_raw_hash
        if certificate is not None and certificate.raw_output_artifact_hash != terminal_raw:
            raise CaseGpuAdapterError("case C2 certificate does not cite terminal raw output")
        receipt = CaseOutputReceipt(
            receipt_id=f"receipt-{call.call_id}",
            execution_plan_hash=self.plan.content_hash,
            projection_job_id=call.call_id,
            condition=ConditionName.C2_LLM_QUERY,
            context_id=protected_query.context_id,
            window_id=cast(str, call.window_id),
            backend="vllm_gpu",
            query_access_receipt_hash=query_access.content_hash,
            preparation_receipt_hash=preparation_receipt.content_hash,
            evidence_binding_hash=expected_binding,
            packet_equality_group_id=packet_group,
            snapshot_hash=query_access.snapshot_hash,
            packet_hash=protected_packet.content_hash,
            pre_query_inventory_hash=envelope.empty_inventory_hash,
            base_attempt_artifact_hash=attempt.base_raw_hash,
            repair_attempt_count=1 if repair_used else 0,
            repair_attempt_artifact_hash=attempt.repair_raw_hash,
            repair_parent_artifact_hash=(attempt.base_raw_hash if repair_used else None),
            repair_reserve_class=("reserve_standard" if repair_used else None),
            terminal_outcome=attempt.outcome,
            projection_artifact_hash=projection_hash,
            failure_lineage_hash=attempt.failure_lineage_hash,
            construction_certificate=certificate,
            query_accessed_at=query_access.accessed_at,
            completed_at=_strictly_after(self.clock, completed_at),
            operational_only=call.operational_only,
            causal_comparison_eligible=call.causal_comparison_eligible,
        )
        self._persist_gpu_audit(
            call=call,
            base_semantic=semantic,
            base_guided=guided,
            base_config=config,
            attempt=attempt,
            condition_result_reference=condition_ref,
            query_access=query_access,
            prequery_barrier=prequery_barrier,
            request_envelope_hash=envelope.content_hash,
            stage_manifest_hash=inputs.query_access.stage_manifest_hash,
            completed_receipt=receipt,
        )
        return receipt


def build_production_case_study_gpu_adapter(
    *,
    root: Path,
    plan: CaseStudyExecutionPlan,
    loaded: AttestedRestrictedCaseStudy,
    admission: CaseExecutionAdmissionReceipt,
    construction: DevelopmentConstructionConfiguration,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    service: OwnedCaseModelService,
    artifacts: ArtifactStore,
    repository: CaseExecutionRepository,
    state_pointer_path: Path,
    model_manifest_hash: Sha256Digest,
    service_start_watchdog_seconds: int = CASE_SERVICE_START_WATCHDOG_SECONDS,
    process_start_ticks: Callable[[int], int] = _process_start_ticks,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ProductionCaseStudyGpuAdapter:
    """Verify all production prerequisites and return the concrete GPU adapter.

    No file is discovered here: callers must supply the explicitly attested case
    input, cumulative ledger-backed artifact store, tokenizer, and owned service.
    """

    if (
        loaded.manifest.content_hash != plan.restricted_index_manifest_hash
        or loaded.preregistration.content_hash != plan.preregistration_hash
        or loaded.attestation.content_hash != plan.input_attestation_hash
    ):
        raise CaseStudyAdmissionError("lawful restricted case input differs from the plan")
    if artifacts.ledger is not repository.artifacts.ledger:
        raise CaseStudyAdmissionError("case factory must use one cumulative SQLite ledger")
    source = build_source_manifest(root.resolve(strict=True), admission.source_revision)
    source_payload = canonical_json(source.to_dict()).encode("utf-8")
    if (
        canonical_sha256(source.to_dict()) != admission.source_manifest.logical_content_hash
        or _read_reference(artifacts, admission.source_manifest) != source_payload
    ):
        raise CaseStudyAdmissionError("case source association changed before execution")
    if _total_allocated_seconds(artifacts.ledger) + 1e-6 < (
        admission.allocated_gpu_seconds_before_case
    ):
        raise CaseStudyAdmissionError("case factory cannot reset cumulative GPU accounting")
    return ProductionCaseStudyGpuAdapter(
        root=root,
        plan=plan,
        admission=admission,
        construction=construction,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        service=service,
        artifacts=artifacts,
        repository=repository,
        state_pointer_path=state_pointer_path,
        model_manifest_hash=model_manifest_hash,
        service_start_watchdog_seconds=service_start_watchdog_seconds,
        process_start_ticks=process_start_ticks,
        clock=clock,
    )


__all__ = [
    "CASE_GPU_ADAPTER_REVISION",
    "CaseGpuAdapterError",
    "CaseGpuAdapterState",
    "CaseGpuLifecycleReceipt",
    "CaseGpuShutdownReceipt",
    "CaseServiceIdentity",
    "ProductionCaseStudyGpuAdapter",
    "build_production_case_study_gpu_adapter",
]
