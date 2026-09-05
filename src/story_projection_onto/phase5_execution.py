"""Production orchestration for the bounded Phase 5 feedback study.

The orchestrator consumes frozen, externally produced held-out inputs.  It never
constructs ontology semantics itself: CPU conditions may only reproject an
existing construction seal, and C2 work is delegated to an injected, metered
adapter.  Its journal is append-only and safe to resume exactly.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    ConditionName,
    ConstructionOperator,
    EvidencePacket,
    FeedbackAction,
    FeedbackResolutionStatus,
    Identifier,
    ImmutableRecord,
    OntologyProjection,
    Sha256Digest,
    canonical_sha256,
    to_model_visible_packet,
)
from story_projection_onto.feedback_provenance import ResearcherTraceSubmissionReceipt
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
    FeedbackEpisodeExecution,
    FeedbackProtocolConfiguration,
    FeedbackRegenerationReceipt,
    FeedbackStudyExecutionManifest,
    ScriptedRevisionFreeze,
    assert_feedback_manifest_matches_protocol,
    compile_feedback_execution,
    load_feedback_protocol,
)
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
)
from story_projection_onto.independent_review_runtime import load_completed_review
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    CommitmentCheckStatus,
    EvidenceSupportStatus,
    FailureKind,
    InputKind,
    JobRecord,
    JobState,
    ProjectionRecord,
    SemanticAssessmentScope,
    StoreError,
    StudyRecord,
    TemporalValidationStatus,
    ValidationStatus,
)
from story_projection_onto.store import (
    FeedbackAction as LedgerFeedbackAction,
)
from story_projection_onto.store import (
    FeedbackKind as LedgerFeedbackKind,
)
from story_projection_onto.store import (
    FeedbackResolutionStatus as LedgerFeedbackResolutionStatus,
)
from story_projection_onto.store import (
    ReleaseClass as LedgerReleaseClass,
)
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    RevisionInstruction,
    VisualizationBundle,
    load_visualization_configuration,
)

CPU_CONDITIONS = (ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE)
DEFAULT_PHASE5_OUTPUT_ROOT = Path("artifacts/restricted/phase5_feedback")
DEFAULT_PHASE5_RUNNER_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs/study/phase5_execution.json"
)


class Phase5ExecutionError(RuntimeError):
    """The production feedback execution cannot safely proceed."""


class Phase5PrerequisiteError(Phase5ExecutionError):
    """A real reviewed held-out result is absent or inconsistent."""


class InterruptedFeedbackCallRecoveryRequired(Phase5ExecutionError):
    """A durable C2 slot exists but its immutable ledger result is unavailable."""


class Phase5LedgerPreflightError(Phase5ExecutionError):
    """The production ledger lacks an exact, already-persisted Phase 5 binding."""


class Phase5LedgerPersistenceError(Phase5ExecutionError):
    """A fully resolved Phase 5 ledger write could not be replayed exactly."""


class Phase5CpuLedgerMaterializationError(Phase5LedgerPersistenceError):
    """A frozen CPU feedback result could not be materialized without invention."""


class Phase5RunnerConfiguration(ImmutableRecord):
    configuration_id: Identifier
    feedback_protocol_hash: Sha256Digest
    required_upstream_state: Literal["held_out_primary_results_frozen"]
    output_root: Literal["artifacts/restricted/phase5_feedback"]
    append_only: Literal[True] = True
    scripted_episode_count: Literal[6] = 6
    researcher_trace_count: Literal[3] = 3
    cpu_reprojection_call_count: Literal[12] = 12
    c2_regeneration_slot_count: Literal[9] = 9
    maximum_repair_gpu_request_count: Literal[9] = 9
    authoritative_materialization_receipt_required: Literal[True] = True
    permit_gpu_execution_without_adapter: Literal[False] = False
    controller_owns_model_service_lifecycle: Literal[False] = False
    usability_claims: Literal[False] = False


def load_phase5_runner_configuration(
    path: Path = DEFAULT_PHASE5_RUNNER_CONFIG_PATH,
) -> Phase5RunnerConfiguration:
    return _read(path, Phase5RunnerConfiguration)


class PrimaryHeldOutResultsGate(ImmutableRecord):
    """Typed wrapper around the real held-out execution and scorer bridge.

    The wrapper cannot authorize Phase 5 by assertion alone.  Prerequisite replay
    parses both upstream artifacts through their owning contracts and checks their
    logical and byte hashes before this receipt is accepted.
    """

    gate_id: Identifier
    lifecycle_state: Literal["held_out_primary_results_frozen"]
    benchmark_draft_seal_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    held_out_execution_manifest_relative_path: Identifier
    held_out_execution_manifest_hash: Sha256Digest
    held_out_execution_manifest_file_sha256: Sha256Digest
    scorer_bridge_relative_path: Identifier
    scorer_bridge_hash: Sha256Digest
    scorer_bridge_file_sha256: Sha256Digest
    result_artifact_hashes: tuple[Sha256Digest, ...]
    world_count: Literal[12] = 12
    context_count: Literal[36] = 36
    conditions: tuple[ConditionName, ...]
    intention_to_treat_complete: Literal[True] = True
    test_fixture: Literal[False] = False
    frozen_at: AwareDatetime

    @model_validator(mode="after")
    def exact_primary_inventory(self) -> Self:
        expected = {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        }
        if set(self.conditions) != expected or len(self.conditions) != 4:
            raise ValueError("held-out result gate requires the four registered conditions")
        if not self.result_artifact_hashes:
            raise ValueError("held-out result gate requires immutable result artifact hashes")
        if len(self.result_artifact_hashes) != len(set(self.result_artifact_hashes)):
            raise ValueError("held-out result artifact hashes must be unique")
        for label, value in (
            ("held-out execution manifest", self.held_out_execution_manifest_relative_path),
            ("scorer bridge", self.scorer_bridge_relative_path),
        ):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError(f"{label} path must be safe and relative")
        if self.held_out_execution_manifest_relative_path == self.scorer_bridge_relative_path:
            raise ValueError("held-out execution and scorer bridge paths must be distinct")
        return self


class CpuReprojectionInput(ImmutableRecord):
    condition: ConditionName
    before_projection: OntologyProjection
    after_projection: OntologyProjection | None = None
    instruction_hash: Sha256Digest
    source_query_stage_hash: Sha256Digest
    source_preparation_hash: Sha256Digest
    resolver_hash: Sha256Digest
    started_at: AwareDatetime
    completed_at: AwareDatetime
    resolved_at: AwareDatetime
    checked_at: AwareDatetime

    @model_validator(mode="after")
    def only_same_seal_cpu_work(self) -> Self:
        if self.condition not in CPU_CONDITIONS:
            raise ValueError("CPU feedback input is restricted to C0 and C1")
        if self.before_projection.condition is not self.condition:
            raise ValueError("CPU input before projection belongs to another condition")
        if self.completed_at < self.started_at or self.resolved_at < self.completed_at:
            raise ValueError("CPU reprojection timestamps are not monotonic")
        if self.checked_at < self.resolved_at:
            raise ValueError("CPU replay check predates resolution")
        if self.after_projection is not None:
            if self.after_projection.condition is not self.condition:
                raise ValueError("CPU input after projection belongs to another condition")
            before_seal = self.before_projection.construction_seal
            after_seal = self.after_projection.construction_seal
            if (
                before_seal is None
                or after_seal is None
                or before_seal.content_hash != after_seal.content_hash
            ):
                raise ValueError("CPU feedback may only reproject the identical sealed ontology")
        return self


class CpuReprojectionReceipt(ImmutableRecord):
    call_id: Identifier
    episode_id: Identifier
    condition: ConditionName
    instruction_hash: Sha256Digest
    construction_seal_hash: Sha256Digest
    before_projection_hash: Sha256Digest
    after_projection_hash: Sha256Digest | None
    status: Literal["reprojected_same_seal", "capability_limited"]
    resolver_hash: Sha256Digest
    started_at: AwareDatetime
    completed_at: AwareDatetime
    allocated_gpu_seconds: Literal[0.0] = 0.0


class ScorerOnlyFeedbackBinding(ImmutableRecord):
    """Restricted link for later metric compilation; never model-visible."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    episode_id: Identifier
    condition: ConditionName
    scorer_binding_key: Identifier
    before_gold_projection_hash: Sha256Digest
    after_gold_projection_hash: Sha256Digest
    required_target_change_hash: Sha256Digest


class ScriptedFeedbackInput(ImmutableRecord):
    episode_id: Identifier
    instruction: RevisionInstruction
    packet: EvidencePacket
    freeze: ScriptedRevisionFreeze
    cpu_inputs: tuple[CpuReprojectionInput, CpuReprojectionInput]
    c2_before_projection: OntologyProjection
    scorer_bindings: tuple[
        ScorerOnlyFeedbackBinding,
        ScorerOnlyFeedbackBinding,
        ScorerOnlyFeedbackBinding,
    ]

    @model_validator(mode="after")
    def exact_condition_independent_script(self) -> Self:
        if self.freeze.episode_id != self.episode_id:
            raise ValueError("script freeze belongs to another episode")
        if self.freeze.instruction_hash != self.instruction.content_hash:
            raise ValueError("script instruction differs from its pre-output freeze")
        if self.freeze.anchor_hashes != tuple(
            item.content_hash for item in self.instruction.revision.anchors
        ):
            raise ValueError("script anchors differ from their pre-output freeze")
        if self.freeze.frozen_at > self.instruction.revision.created_at:
            raise ValueError("script instruction predates its freeze")
        if {item.condition for item in self.cpu_inputs} != set(CPU_CONDITIONS):
            raise ValueError("script requires exactly one C0 and one C1 CPU input")
        if self.c2_before_projection.condition is not ConditionName.C2_LLM_QUERY:
            raise ValueError("script C2 before projection is not C2")
        conditions = [item.condition for item in self.scorer_bindings]
        if set(conditions) != {*CPU_CONDITIONS, ConditionName.C2_LLM_QUERY}:
            raise ValueError("script scorer bindings require exactly C0, C1, and C2")
        if any(item.episode_id != self.episode_id for item in self.scorer_bindings):
            raise ValueError("script scorer binding belongs to another episode")
        if any(
            item.instruction_hash != self.instruction.content_hash
            for item in self.cpu_inputs
        ):
            raise ValueError("CPU input does not bind the exact scripted instruction")
        cpu_after = tuple(item.after_projection for item in self.cpu_inputs)
        if self.instruction.revision.action is FeedbackAction.REFINE_CONTEXT:
            if any(item is None for item in cpu_after):
                raise ValueError("REFINE_CONTEXT requires both same-seal CPU reprojections")
            for item in cpu_after:
                assert item is not None
                if (
                    item.context_hash != self.instruction.after_context.content_hash
                    or not item.decisions
                    or any(
                        decision.operator is not ConstructionOperator.SELECTION
                        for decision in item.decisions
                    )
                ):
                    raise ValueError("CPU context refinement must remain selection-only")
        elif any(item is not None for item in cpu_after):
            raise ValueError("CPU merge/split feedback must be capability_limited")
        gold_targets = {
            (
                item.before_gold_projection_hash,
                item.after_gold_projection_hash,
                item.required_target_change_hash,
            )
            for item in self.scorer_bindings
        }
        if len(gold_targets) != 1:
            raise ValueError("all conditions must be scored against one shared known answer")
        if any(item.started_at <= self.instruction.revision.created_at for item in self.cpu_inputs):
            raise ValueError("CPU feedback work must start after the revision is recorded")
        before = (
            *(item.before_projection for item in self.cpu_inputs),
            self.c2_before_projection,
        )
        if len({item.snapshot_hash for item in before}) != 1:
            raise ValueError("feedback conditions must share one evidence snapshot")
        certificate = self.c2_before_projection.construction_certificate
        if certificate is None or certificate.completed_at >= self.instruction.revision.created_at:
            raise ValueError("the C2 parent projection must complete before its feedback revision")
        return self


class ResearcherTraceInput(ImmutableRecord):
    episode_id: Identifier
    instruction: RevisionInstruction
    submission_receipt: ResearcherTraceSubmissionReceipt
    packet: EvidencePacket
    c2_before_projection: OntologyProjection

    @model_validator(mode="after")
    def c2_only(self) -> Self:
        if self.c2_before_projection.condition is not ConditionName.C2_LLM_QUERY:
            raise ValueError("researcher trace input must be C2")
        receipt = self.submission_receipt
        if (
            receipt.episode_id != self.episode_id
            or receipt.action is not self.instruction.revision.action
            or receipt.requested_at != self.instruction.revision.created_at
            or receipt.instruction_hash != self.instruction.content_hash
            or receipt.before_projection_id != self.c2_before_projection.projection_id
            or receipt.before_projection_hash != self.c2_before_projection.content_hash
        ):
            raise ValueError("researcher trace input differs from its UI submission receipt")
        certificate = self.c2_before_projection.construction_certificate
        if certificate is None or certificate.completed_at >= self.instruction.revision.created_at:
            raise ValueError("the C2 parent projection must complete before its feedback revision")
        return self


class Phase5ExecutionInputManifest(ImmutableRecord):
    run_id: Identifier
    protocol_hash: Sha256Digest
    script_commitment_manifest_hash: Sha256Digest
    primary_results_gate_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    held_out_call_manifest_hash: Sha256Digest
    scorer_binding_authorization_hash: Sha256Digest
    frozen_at: AwareDatetime
    scripted_inputs: tuple[
        ScriptedFeedbackInput,
        ScriptedFeedbackInput,
        ScriptedFeedbackInput,
        ScriptedFeedbackInput,
        ScriptedFeedbackInput,
        ScriptedFeedbackInput,
    ]
    researcher_trace_inputs: tuple[
        ResearcherTraceInput,
        ResearcherTraceInput,
        ResearcherTraceInput,
    ]

    @model_validator(mode="after")
    def exact_input_inventory(self) -> Self:
        script_ids = [item.episode_id for item in self.scripted_inputs]
        trace_ids = [item.episode_id for item in self.researcher_trace_inputs]
        if len(set(script_ids)) != 6 or len(set(trace_ids)) != 3:
            raise ValueError("Phase 5 input requires six distinct scripts and three traces")
        if set(script_ids).intersection(trace_ids):
            raise ValueError("script and trace episode IDs must be disjoint")
        freezes = [item.freeze for item in self.scripted_inputs]
        if any(item.frozen_at > self.frozen_at for item in freezes):
            raise ValueError("input manifest freeze predates a scripted revision freeze")
        instructions = [item.instruction for item in self.scripted_inputs]
        instructions.extend(item.instruction for item in self.researcher_trace_inputs)
        if any(item.revision.created_at > self.frozen_at for item in instructions):
            raise ValueError("input manifest freeze predates a feedback instruction")
        if any(
            item.submission_receipt.recorded_at > self.frozen_at
            for item in self.researcher_trace_inputs
        ):
            raise ValueError("input manifest freeze predates a UI submission receipt")
        return self


class C2RegenerationRequest(ImmutableRecord):
    call_slot_id: Identifier
    episode_id: Identifier
    kind: FeedbackEpisodeKind
    instruction: RevisionInstruction
    packet: EvidencePacket
    before_projection_id: Identifier
    before_projection_hash: Sha256Digest
    before_context_hash: Sha256Digest
    before_packet_hash: Sha256Digest
    protocol_hash: Sha256Digest
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def request_is_post_revision_c2(self) -> Self:
        if self.before_context_hash != self.instruction.before_context.content_hash:
            raise ValueError("C2 regeneration request context changed")
        if self.before_packet_hash != self.packet.content_hash:
            raise ValueError("C2 regeneration request packet changed")
        return self

    def model_visible_payload(self) -> dict[str, object]:
        """Return only query/evidence/revision input, never scorer bindings."""

        return {
            "instruction": self.instruction.model_visible().model_dump(mode="json"),
            "revised_context": self.instruction.after_context.model_dump(mode="json"),
            "evidence_packet": to_model_visible_packet(self.packet).model_dump(mode="json"),
            "parent_projection_hash": self.before_projection_hash,
            "seed": self.seed,
        }


class C2RegenerationResult(ImmutableRecord):
    request_hash: Sha256Digest
    after_projection: OntologyProjection | None = None
    after_packet: EvidencePacket | None = None
    receipt: FeedbackRegenerationReceipt
    resolver_hash: Sha256Digest
    resolved_at: AwareDatetime
    checked_at: AwareDatetime
    latency_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def result_is_itt_complete(self) -> Self:
        if self.checked_at < self.resolved_at:
            raise ValueError("C2 replay check predates resolution")
        if self.receipt.completed_at > self.resolved_at:
            raise ValueError("C2 resolution predates its metered attempt")
        if self.receipt.attempt_status is FeedbackAttemptStatus.SUCCEEDED:
            if self.after_projection is None:
                raise ValueError("successful C2 result requires its projection")
        elif self.receipt.failure_artifact_hash is None:
            raise ValueError("ITT C2 failure requires a retained failure artifact")
        if self.after_projection is None and self.after_packet is not None:
            raise ValueError("C2 result cannot retain a new packet without a projection")
        observed_after_hash = (
            None if self.after_projection is None else self.after_projection.content_hash
        )
        if self.receipt.after_projection_hash != observed_after_hash:
            raise ValueError("C2 result and GPU receipt retain different projections")
        return self


@runtime_checkable
class C2RegenerationAdapter(Protocol):
    """Live adapter with a side-effect-free recovery lookup.

    ``recover`` must only consult durable state for the exact request and must
    never start inference.  It may return ``None`` only after proving that no
    adapter intent, evidence-preparation artifact, GPU event, model-call row, or
    terminal output exists.  That exact-absence receipt permits the caller to
    consume a pre-existing outer slot as the first adapter request.
    """

    def regenerate(self, request: C2RegenerationRequest) -> C2RegenerationResult: ...

    def recover(self, request: C2RegenerationRequest) -> C2RegenerationResult | None: ...


class FeedbackLedgerVerifier(Protocol):
    """Read-only verifier for persisted SQLite model-call and GPU-event rows."""

    def verify(
        self,
        request: C2RegenerationRequest,
        receipt: FeedbackRegenerationReceipt,
    ) -> None: ...


_MODEL_CALL_HASH_FIELDS = (
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
_GPU_EVENT_HASH_FIELDS = (
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


def _record_value(record: object, name: str):
    value = record[name] if isinstance(record, sqlite3.Row) else getattr(record, name)
    if name == "successful":
        return bool(value)
    if name == "succeeded":
        return None if value is None else bool(value)
    return value.value if hasattr(value, "value") else value


def feedback_model_call_record_hash(record: object) -> str:
    """Canonical hash of every immutable field in one SQLite model-call row."""

    return canonical_sha256({name: _record_value(record, name) for name in _MODEL_CALL_HASH_FIELDS})


def feedback_gpu_event_record_hash(record: object) -> str:
    """Canonical hash of every immutable field in one SQLite GPU-event row."""

    return canonical_sha256({name: _record_value(record, name) for name in _GPU_EVENT_HASH_FIELDS})


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Phase5ExecutionError("GPU ledger contains a timezone-naive timestamp")
    return parsed


class SQLiteFeedbackLedgerVerifier:
    """Verify feedback receipts against the existing append-only SQLite ledger.

    The connection is opened in read-only mode.  This verifier neither creates
    ledger rows nor accepts a receipt merely because its hashes are well formed.
    """

    def __init__(self, ledger_path: Path) -> None:
        self.ledger_path = ledger_path.absolute()

    def verify(
        self,
        request: C2RegenerationRequest,
        receipt: FeedbackRegenerationReceipt,
    ) -> None:
        _assert_no_symlink_chain(self.ledger_path)
        if not self.ledger_path.is_file():
            raise Phase5ExecutionError("Phase 5 GPU ledger is missing or not a regular file")
        try:
            connection = sqlite3.connect(
                f"file:{self.ledger_path.as_posix()}?mode=ro",
                uri=True,
                timeout=30,
            )
            connection.row_factory = sqlite3.Row
            try:
                self._verify_rows(connection, request, receipt)
            finally:
                connection.close()
        except Phase5ExecutionError:
            raise
        except sqlite3.Error as error:
            raise Phase5ExecutionError(
                f"cannot replay Phase 5 GPU ledger receipt: {error}"
            ) from error

    @staticmethod
    def _verify_rows(
        connection: sqlite3.Connection,
        request: C2RegenerationRequest,
        receipt: FeedbackRegenerationReceipt,
    ) -> None:
        for index, reference in enumerate(receipt.ledger_calls):
            model_call = connection.execute(
                "SELECT * FROM model_calls WHERE model_call_id = ?",
                (reference.model_call_id,),
            ).fetchone()
            if model_call is None:
                raise Phase5ExecutionError("feedback receipt cites a missing model-call row")
            event = connection.execute(
                "SELECT * FROM gpu_events WHERE event_id = ?",
                (reference.gpu_event_id,),
            ).fetchone()
            if event is None:
                raise Phase5ExecutionError("feedback receipt cites a missing GPU-event row")
            if feedback_model_call_record_hash(model_call) != reference.model_call_record_hash:
                raise Phase5ExecutionError("feedback model-call ledger row hash changed")
            if feedback_gpu_event_record_hash(event) != reference.gpu_event_record_hash:
                raise Phase5ExecutionError("feedback GPU-event ledger row hash changed")
            expected_microseconds = round(reference.allocated_gpu_seconds * 1_000_000)
            is_final_call = index == len(receipt.ledger_calls) - 1
            if is_final_call and receipt.attempt_status is FeedbackAttemptStatus.TIMED_OUT:
                expected_event_kinds = {"timeout"}
            elif is_final_call and receipt.attempt_status is FeedbackAttemptStatus.FAILED:
                expected_event_kinds = {"failure"}
            elif reference.attempt_kind == "repair":
                expected_event_kinds = {"repair"}
            else:
                expected_event_kinds = {"inference"}
            checks = (
                model_call["gpu_event_id"] == reference.gpu_event_id,
                model_call["attempt_id"] == reference.attempt_id,
                model_call["backend"] == "vllm_gpu",
                model_call["call_role"] == reference.call_role,
                model_call["retry_class"] == reference.retry_class,
                model_call["request_hash"] == reference.request_hash,
                model_call["allocated_gpu_microseconds"] == expected_microseconds,
                bool(model_call["successful"]) is reference.successful,
                event["attempt_id"] == reference.attempt_id,
                event["job_id"] == model_call["job_id"],
                event["allocated_microseconds"] == expected_microseconds,
                event["event_kind"] in expected_event_kinds,
                (None if event["succeeded"] is None else bool(event["succeeded"]))
                is reference.successful,
                _aware_datetime(event["started_at"]) == reference.started_at,
                _aware_datetime(event["ended_at"]) == reference.completed_at,
            )
            if not all(checks):
                raise Phase5ExecutionError("feedback receipt differs from its GPU ledger rows")
            if reference.successful and model_call["response_artifact_hash"] is None:
                raise Phase5ExecutionError("completed feedback inference lacks its raw artifact")
        if receipt.ledger_calls[0].request_hash != request.content_hash:
            raise Phase5ExecutionError("feedback base ledger call hashes another request")


class FeedbackExecutionRecord(ImmutableRecord):
    input_manifest_hash: Sha256Digest
    source_input_hash: Sha256Digest
    execution: FeedbackEpisodeExecution
    cpu_receipt: CpuReprojectionReceipt | None = None
    c2_request_hash: Sha256Digest | None = None
    c2_result_hash: Sha256Digest | None = None
    scorer_binding: ScorerOnlyFeedbackBinding | None = None

    @model_validator(mode="after")
    def receipt_and_gold_policy(self) -> Self:
        execution = self.execution
        if execution.condition in CPU_CONDITIONS:
            if self.cpu_receipt is None:
                raise ValueError("C0/C1 feedback requires an auditable CPU receipt")
            if self.cpu_receipt.condition is not execution.condition:
                raise ValueError("CPU receipt condition differs from its execution")
            if (
                self.cpu_receipt.episode_id != execution.episode_id
                or self.cpu_receipt.instruction_hash != execution.instruction.content_hash
                or self.cpu_receipt.before_projection_hash
                != execution.before_bundle.projection_hash
            ):
                raise ValueError("CPU receipt lineage differs from its execution")
            if self.c2_request_hash is not None or self.c2_result_hash is not None:
                raise ValueError("CPU execution cannot claim C2 call artifacts")
        else:
            if self.cpu_receipt is not None:
                raise ValueError("C2 execution cannot carry a CPU receipt")
            if self.c2_request_hash is None:
                raise ValueError("C2 execution must bind its persisted call request")
            if self.c2_result_hash is None:
                raise ValueError("C2 execution must bind its persisted ITT result")
        if execution.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER:
            if self.scorer_binding is None:
                raise ValueError("known-answer execution requires a scorer-only binding")
            if self.scorer_binding.condition is not execution.condition:
                raise ValueError("scorer binding condition differs from execution")
            if self.scorer_binding.episode_id != execution.episode_id:
                raise ValueError("scorer binding episode differs from execution")
        elif self.scorer_binding is not None:
            raise ValueError("researcher traces must keep gold fields NA")
        return self


class Phase5ArtifactBinding(ImmutableRecord):
    relative_path: Identifier
    artifact_kind: Literal[
        "input_manifest",
        "script_freeze",
        "c2_call_slot",
        "c2_result",
        "execution_record",
        "feedback_manifest",
    ]
    logical_content_hash: Sha256Digest
    file_sha256: Sha256Digest

    @model_validator(mode="after")
    def safe_relative_path(self) -> Self:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.relative_path:
            raise ValueError("Phase 5 artifact binding path must be safe and relative")
        return self


class Phase5JournalIndex(ImmutableRecord):
    run_id: Identifier
    input_manifest_hash: Sha256Digest
    protocol_hash: Sha256Digest
    primary_results_gate_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    feedback_manifest_hash: Sha256Digest
    execution_record_hashes: tuple[Sha256Digest, ...]
    c2_request_hashes: tuple[Sha256Digest, ...]
    c2_result_hashes: tuple[Sha256Digest, ...]
    ledger_model_call_hashes: tuple[Sha256Digest, ...]
    ledger_gpu_event_hashes: tuple[Sha256Digest, ...]
    actual_gpu_request_count: int = Field(ge=9, le=18)
    repair_gpu_request_count: int = Field(ge=0, le=9)
    allocated_gpu_seconds: float = Field(gt=0.0)
    artifact_bindings: tuple[Phase5ArtifactBinding, ...]
    cpu_call_record_count: Literal[12] = 12
    c2_regeneration_slot_count: Literal[9] = 9
    scripted_metric_binding_count: Literal[18] = 18
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def exact_final_inventory(self) -> Self:
        if len(self.execution_record_hashes) != 21:
            raise ValueError("Phase 5 index requires exactly 21 execution records")
        if len(self.c2_request_hashes) != 9 or len(set(self.c2_request_hashes)) != 9:
            raise ValueError("Phase 5 index requires nine distinct C2 call requests")
        if len(self.c2_result_hashes) != 9 or len(set(self.c2_result_hashes)) != 9:
            raise ValueError("Phase 5 index requires nine distinct C2 ITT results")
        if self.actual_gpu_request_count != len(self.ledger_model_call_hashes):
            raise ValueError("Phase 5 GPU request count differs from its ledger model calls")
        if len(self.ledger_gpu_event_hashes) != self.actual_gpu_request_count:
            raise ValueError("Phase 5 GPU request count differs from its ledger GPU events")
        if (
            len(set(self.ledger_model_call_hashes)) != self.actual_gpu_request_count
            or len(set(self.ledger_gpu_event_hashes)) != self.actual_gpu_request_count
        ):
            raise ValueError("Phase 5 ledger calls and GPU events must be globally unique")
        if self.repair_gpu_request_count != self.actual_gpu_request_count - 9:
            raise ValueError("Phase 5 repair count differs from base-plus-repair ledger lineage")
        expected_artifacts = Counter(
            {
                "input_manifest": 1,
                "script_freeze": 6,
                "c2_call_slot": 9,
                "c2_result": 9,
                "execution_record": 21,
                "feedback_manifest": 1,
            }
        )
        if Counter(item.artifact_kind for item in self.artifact_bindings) != expected_artifacts:
            raise ValueError("Phase 5 final artifact inventory is incomplete")
        paths = [item.relative_path for item in self.artifact_bindings]
        if len(paths) != len(set(paths)):
            raise ValueError("Phase 5 final artifact paths must be unique")
        return self


def _read(path: Path, model_type):
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5ExecutionError(f"missing or symlinked immutable input: {path}")
    try:
        raw = path.read_bytes()
        if len(raw) > 100 * 1024 * 1024:
            raise Phase5ExecutionError(f"immutable input exceeds 100 MiB: {path}")
        return model_type.model_validate_json(raw)
    except Exception as error:
        raise Phase5ExecutionError(f"invalid {model_type.__name__}: {error}") from error


def load_phase5_input_manifest(path: Path) -> Phase5ExecutionInputManifest:
    """Load a self-hashed, immutable Phase 5 input inventory."""

    return _read(path, Phase5ExecutionInputManifest)


def validate_phase5_prerequisites(
    *,
    inputs: Phase5ExecutionInputManifest,
    primary_results_gate_path: Path,
    benchmark_root: Path,
    review_completion_root: Path,
) -> PrimaryHeldOutResultsGate:
    """Reproduce both upstream gates; a status boolean is insufficient."""

    gate, execution, _bridge = replay_phase5_prerequisites(
        expected_gate_hash=inputs.primary_results_gate_hash,
        expected_final_reviewed_seal_hash=inputs.final_reviewed_seal_hash,
        expected_consumer_frozen_at=inputs.frozen_at,
        primary_results_gate_path=primary_results_gate_path,
        benchmark_root=benchmark_root,
        review_completion_root=review_completion_root,
    )
    if execution.content_hash != inputs.held_out_execution_manifest_hash:
        raise Phase5PrerequisiteError("Phase 5 inputs bind another held-out execution manifest")
    return gate


def replay_phase5_prerequisites(
    *,
    expected_gate_hash: str,
    expected_final_reviewed_seal_hash: str,
    expected_consumer_frozen_at: datetime,
    primary_results_gate_path: Path,
    benchmark_root: Path,
    review_completion_root: Path,
) -> tuple[
    PrimaryHeldOutResultsGate,
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
]:
    """Return typed upstream records only after replaying every Phase 5 gate."""

    try:
        review = load_completed_review(
            benchmark_root=benchmark_root, output_root=review_completion_root
        )
    except Exception as error:
        raise Phase5PrerequisiteError(
            f"independent review is not reproducibly complete: {error}"
        ) from error
    gate = _read(primary_results_gate_path, PrimaryHeldOutResultsGate)
    if (
        gate.content_hash != expected_gate_hash
        or gate.final_reviewed_seal_hash != review.final_seal.content_hash
        or expected_final_reviewed_seal_hash != review.final_seal.content_hash
        or gate.benchmark_draft_seal_hash != review.draft.content_hash
        or gate.frozen_at > expected_consumer_frozen_at
    ):
        raise Phase5PrerequisiteError("Phase 5 inputs do not bind the reviewed held-out results")
    primary_root = primary_results_gate_path.absolute().parent
    _assert_no_symlink_chain(primary_root)

    def load_upstream(relative: str, model_type, expected_file_hash: str):
        path = primary_root / relative
        _assert_no_symlink_chain(path)
        try:
            if not path.resolve(strict=True).is_relative_to(primary_root.resolve(strict=True)):
                raise Phase5PrerequisiteError("held-out prerequisite escapes its result root")
        except OSError as error:
            raise Phase5PrerequisiteError("held-out prerequisite artifact is absent") from error
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_file_hash:
            raise Phase5PrerequisiteError("held-out prerequisite artifact file hash changed")
        try:
            return _read(path, model_type)
        except Phase5ExecutionError as error:
            raise Phase5PrerequisiteError(str(error)) from error

    execution = load_upstream(
        gate.held_out_execution_manifest_relative_path,
        HeldOutExecutionManifest,
        gate.held_out_execution_manifest_file_sha256,
    )
    bridge = load_upstream(
        gate.scorer_bridge_relative_path,
        ScorerBridgeAuthorization,
        gate.scorer_bridge_file_sha256,
    )
    if (
        execution.content_hash != gate.held_out_execution_manifest_hash
        or bridge.content_hash != gate.scorer_bridge_hash
        or bridge.execution_manifest_hash != execution.content_hash
        or bridge.call_manifest_hash != execution.call_manifest_hash
        or execution.final_reviewed_seal_hash != review.final_seal.content_hash
        or bridge.final_reviewed_seal_hash != review.final_seal.content_hash
        or bridge.output_artifact_hashes != gate.result_artifact_hashes
        or bridge.authorized_at < execution.completed_at
        or gate.frozen_at < bridge.authorized_at
    ):
        raise Phase5PrerequisiteError(
            "Phase 5 gate does not reproduce the held-out execution/scorer bridge"
        )
    return gate, execution, bridge


def validate_phase5_inputs(
    inputs: Phase5ExecutionInputManifest,
    protocol: FeedbackProtocolConfiguration,
) -> None:
    if inputs.protocol_hash != protocol.content_hash:
        raise Phase5ExecutionError("Phase 5 inputs use another feedback protocol")
    scripts = {item.episode_id: item for item in inputs.scripted_inputs}
    traces = {item.episode_id: item for item in inputs.researcher_trace_inputs}
    if set(scripts) != {item.episode_id for item in protocol.scripted_episodes}:
        raise Phase5ExecutionError("script input IDs differ from the frozen protocol")
    if set(traces) != {item.episode_id for item in protocol.researcher_trace_slots}:
        raise Phase5ExecutionError("trace input IDs differ from the frozen protocol")

    def assert_sealed_cpu_projection(projection: OntologyProjection) -> None:
        seal = projection.construction_seal
        if seal is None:
            raise Phase5ExecutionError("CPU feedback projection lacks a construction seal")
        graph = projection.instance_graph
        semantic_ids = {
            *(item.entity_id for item in graph.entities),
            *(item.event_id for item in graph.events),
            *(item.proposition_content_id for item in graph.proposition_contents),
            *(item.assertion_id for item in graph.assertions),
        }
        if not semantic_ids.issubset(seal.sealed_object_ids):
            raise Phase5ExecutionError("CPU feedback projection contains an unsealed semantic ID")
        if any(item.operator in CONSTRUCTIVE_OPERATORS for item in projection.decisions):
            raise Phase5ExecutionError("CPU feedback projection claims query-time construction")

    for selection in protocol.scripted_episodes:
        item = scripts[selection.episode_id]
        if (
            item.instruction.before_context.context_id != selection.context_id
            or item.instruction.revision.action is not selection.action
            or item.freeze.context_id != selection.context_id
            or item.freeze.action is not selection.action
            or any(
                binding.scorer_binding_key != selection.scorer_binding_key
                for binding in item.scorer_bindings
            )
        ):
            raise Phase5ExecutionError("script input differs from its frozen selection")
        before = (*[cpu.before_projection for cpu in item.cpu_inputs], item.c2_before_projection)
        if any(
            projection.context_hash != item.instruction.before_context.content_hash
            for projection in before
        ):
            raise Phase5ExecutionError("script before projection context differs")
        if any(projection.packet_hash != item.packet.content_hash for projection in before):
            raise Phase5ExecutionError("script conditions do not share the exact packet")
        for cpu in item.cpu_inputs:
            seal = cpu.before_projection.construction_seal
            if seal is None or seal.sealed_at >= item.instruction.before_context.revealed_at:
                raise Phase5ExecutionError(
                    "C0/C1 feedback source was not sealed before the original query reveal"
                )
            assert_sealed_cpu_projection(cpu.before_projection)
            if cpu.after_projection is not None:
                assert_sealed_cpu_projection(cpu.after_projection)
    for slot in protocol.researcher_trace_slots:
        item = traces[slot.episode_id]
        if (
            item.instruction.before_context.context_id != slot.context_id
            or item.instruction.revision.action not in slot.allowed_actions
            or item.c2_before_projection.context_hash
            != item.instruction.before_context.content_hash
            or item.c2_before_projection.packet_hash != item.packet.content_hash
        ):
            raise Phase5ExecutionError("researcher trace differs from its frozen slot")


_TERMINAL_LEDGER_JOB_STATES = frozenset(
    {JobState.FINALIZED, JobState.SCORED, JobState.RENDERED}
)


@dataclass(frozen=True)
class _Phase5LedgerResolutionBinding:
    record: FeedbackExecutionRecord
    before_projection: ProjectionRecord
    receiving_job: JobRecord
    after_projection: ProjectionRecord | None


def _canonical_record_payload(value: ImmutableRecord) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _projection_artifact_hash(projection: OntologyProjection) -> str:
    return hashlib.sha256(_canonical_record_payload(projection)).hexdigest()


def _resolve_exact_ledger_projection(
    *,
    artifacts: ArtifactStore,
    projection: OntologyProjection,
) -> tuple[StudyRecord, ProjectionRecord]:
    """Resolve a semantic projection through its exact immutable CAS bytes."""

    payload = _canonical_record_payload(projection)
    artifact_hash = _projection_artifact_hash(projection)
    artifact = artifacts.ledger.get_artifact(artifact_hash)
    if (
        artifact.release_class.value != projection.release_class.value
        or artifacts.blobs.read_bytes(
            artifact,
            allow_restricted=artifact.release_class is LedgerReleaseClass.RESTRICTED,
        )
        != payload
    ):
        raise Phase5LedgerPreflightError(
            "projection CAS bytes or release class differ from the Phase 5 input"
        )
    return artifacts.ledger.resolve_projection_artifact_owner(
        projection_artifact_hash=artifact_hash,
        projection_semantic_hash=projection.content_hash,
        condition_id=projection.condition.value,
        context_hash=projection.context_hash,
    )


def phase5_cpu_receiving_job_identity(
    *,
    inputs: Phase5ExecutionInputManifest,
    ledger_study_id: str,
    script: ScriptedFeedbackInput,
    cpu: CpuReprojectionInput,
) -> dict[str, object]:
    """Return the only accepted identity for an already-persisted CPU feedback job.

    This function is resolve-only.  Phase 5 never creates the job from this identity;
    the upstream CPU execution owner must have durably recorded and finalized it.
    """

    return {
        "execution_id": ledger_study_id,
        "phase5_run_id": inputs.run_id,
        "episode_id": script.episode_id,
        "condition": cpu.condition.value,
        "lifecycle_kind": "phase5_cpu_reprojection",
        "input_manifest_hash": inputs.content_hash,
        "instruction_hash": script.instruction.content_hash,
        "source_input_hash": cpu.content_hash,
        "source_query_stage_hash": cpu.source_query_stage_hash,
        "source_preparation_hash": cpu.source_preparation_hash,
        "before_projection_hash": cpu.before_projection.content_hash,
        "after_projection_hash": (
            None if cpu.after_projection is None else cpu.after_projection.content_hash
        ),
        "resolver_hash": cpu.resolver_hash,
    }


def _phase5_cpu_ledger_id(prefix: str, payload: object) -> str:
    return f"phase5-cpu-{prefix}-{canonical_sha256(payload)[:32]}"


def _clone_cpu_projection_inputs(
    *,
    artifacts: ArtifactStore,
    ledger_study_id: str,
    source_study: StudyRecord,
    source_projection: ProjectionRecord,
    semantic_projection: OntologyProjection,
) -> tuple[str, str]:
    ledger = artifacts.ledger
    source_snapshot = ledger.get_evidence_snapshot(source_projection.snapshot_id)
    source_snapshot_input = ledger.get_input(source_snapshot.input_id)
    source_packet_input = ledger.get_input(source_projection.packet_input_id)
    if (
        source_snapshot_input.study_id != source_study.study_id
        or source_packet_input.study_id != source_study.study_id
        or source_snapshot_input.input_kind is not InputKind.EVIDENCE_SNAPSHOT
        or source_packet_input.input_kind is not InputKind.EVIDENCE_PACKET
        or source_snapshot_input.content_hash != semantic_projection.snapshot_hash
        or source_packet_input.content_hash != semantic_projection.packet_hash
        or source_snapshot_input.artifact_hash is None
        or source_packet_input.artifact_hash is None
    ):
        raise Phase5CpuLedgerMaterializationError(
            "CPU source projection input metadata is incomplete or inconsistent"
        )
    # Resolve both artifacts before cloning their immutable references.  The aliases
    # change only study-local identifiers; all scientific hashes remain byte-identical.
    ledger.get_artifact(source_snapshot_input.artifact_hash)
    ledger.get_artifact(source_packet_input.artifact_hash)
    snapshot_key = canonical_sha256(
        {
            "study_id": ledger_study_id,
            "source_study_id": source_study.study_id,
            "source_snapshot_id": source_snapshot.snapshot_id,
            "snapshot_hash": source_snapshot_input.content_hash,
        }
    )
    packet_key = canonical_sha256(
        {
            "study_id": ledger_study_id,
            "source_study_id": source_study.study_id,
            "source_packet_input_id": source_packet_input.input_id,
            "packet_hash": source_packet_input.content_hash,
        }
    )
    snapshot_input_id = _phase5_cpu_ledger_id("snapshot-input", snapshot_key)
    snapshot_id = _phase5_cpu_ledger_id("snapshot", snapshot_key)
    packet_input_id = _phase5_cpu_ledger_id("packet-input", packet_key)
    ledger.register_input(
        input_id=snapshot_input_id,
        study_id=ledger_study_id,
        input_kind=InputKind.EVIDENCE_SNAPSHOT,
        content_hash=source_snapshot_input.content_hash,
        artifact_hash=source_snapshot_input.artifact_hash,
        release_class=source_snapshot_input.release_class,
        created_at=source_snapshot_input.created_at,
    )
    ledger.register_evidence_snapshot(
        snapshot_id=snapshot_id,
        input_id=snapshot_input_id,
        horizon_hash=source_snapshot.horizon_hash,
        evidence_manifest_hash=source_snapshot.evidence_manifest_hash,
        index_configuration_hash=source_snapshot.index_configuration_hash,
        prequery_seal_hash=source_snapshot.prequery_seal_hash,
        eligible_evidence_count=source_snapshot.eligible_evidence_count,
        created_at=source_snapshot.created_at,
    )
    ledger.register_input(
        input_id=packet_input_id,
        study_id=ledger_study_id,
        input_kind=InputKind.EVIDENCE_PACKET,
        content_hash=source_packet_input.content_hash,
        artifact_hash=source_packet_input.artifact_hash,
        release_class=source_packet_input.release_class,
        created_at=source_packet_input.created_at,
    )
    return snapshot_id, packet_input_id


def _verify_cpu_job_lifecycle(
    *,
    artifacts: ArtifactStore,
    job: JobRecord,
    sealed_at: datetime,
    revealed_at: datetime,
    completed_at: datetime,
    checked_at: datetime,
) -> None:
    expected = (
        (JobState.PLANNED, sealed_at),
        (JobState.PREQUERY_SEALED, sealed_at),
        (JobState.QUERY_REVEALED, revealed_at),
        (JobState.GENERATED, completed_at),
        (JobState.VALIDATED, checked_at),
        (JobState.FINALIZED, checked_at),
    )
    observed = artifacts.ledger.transitions(job.job_id)
    if len(observed) != len(expected) or any(
        transition.to_state is not state
        or _aware_datetime(transition.occurred_at) != timestamp.astimezone(UTC)
        for transition, (state, timestamp) in zip(observed, expected, strict=True)
    ):
        raise Phase5CpuLedgerMaterializationError(
            "CPU receiving job lifecycle differs from its frozen timestamps"
        )
    if _aware_datetime(job.created_at) != sealed_at.astimezone(UTC):
        raise Phase5CpuLedgerMaterializationError(
            "CPU receiving job creation predates or postdates its source seal"
        )


def materialize_phase5_cpu_ledger(
    *,
    inputs: Phase5ExecutionInputManifest,
    protocol: FeedbackProtocolConfiguration,
    artifacts: ArtifactStore,
    ledger_study_id: str,
) -> None:
    """Persist the twelve frozen CPU executions without constructing semantics."""

    validate_phase5_inputs(inputs, protocol)
    try:
        study = artifacts.ledger.get_study(ledger_study_id)
        if study.release_class is not LedgerReleaseClass.RESTRICTED:
            raise Phase5CpuLedgerMaterializationError(
                "Phase 5 CPU receiving study must be restricted"
            )
        for script in inputs.scripted_inputs:
            for cpu in script.cpu_inputs:
                source_study, source_projection = _resolve_exact_ledger_projection(
                    artifacts=artifacts,
                    projection=cpu.before_projection,
                )
                if source_study.protocol_hash != inputs.held_out_call_manifest_hash:
                    raise Phase5CpuLedgerMaterializationError(
                        "CPU source projection belongs to another held-out protocol"
                    )
                after = cpu.after_projection
                if after is not None and (
                    after.snapshot_hash != cpu.before_projection.snapshot_hash
                    or after.packet_hash != cpu.before_projection.packet_hash
                ):
                    raise Phase5CpuLedgerMaterializationError(
                        "CPU reprojection changed its frozen snapshot or evidence packet"
                    )
                snapshot_id, packet_input_id = _clone_cpu_projection_inputs(
                    artifacts=artifacts,
                    ledger_study_id=ledger_study_id,
                    source_study=source_study,
                    source_projection=source_projection,
                    semantic_projection=cpu.before_projection,
                )
                identity = phase5_cpu_receiving_job_identity(
                    inputs=inputs,
                    ledger_study_id=ledger_study_id,
                    script=script,
                    cpu=cpu,
                )
                seal = cpu.before_projection.construction_seal
                if seal is None:
                    raise Phase5CpuLedgerMaterializationError(
                        "CPU receiving job lacks its source construction seal"
                    )
                job = artifacts.ledger.create_or_resume_job(
                    identity,
                    release_class=LedgerReleaseClass.RESTRICTED,
                    created_at=seal.sealed_at,
                )
                artifacts.ledger.link_job_to_study(
                    study_id=ledger_study_id,
                    job_id=job.job_id,
                    created_at=seal.sealed_at,
                )
                artifacts.ledger.advance_job_lifecycle(
                    job.job_id,
                    (
                        (JobState.PREQUERY_SEALED, seal.sealed_at),
                        (JobState.QUERY_REVEALED, script.instruction.revision.created_at),
                    ),
                )
                identity_hash = canonical_sha256(identity)
                attempt_id = _phase5_cpu_ledger_id("attempt", identity_hash)
                artifacts.ledger.record_attempt(
                    attempt_id=attempt_id,
                    job_id=job.job_id,
                    attempt_kind=AttemptKind.BASE,
                    input_hash=cpu.content_hash,
                    config_hash=canonical_sha256(
                        {
                            "protocol_hash": protocol.content_hash,
                            "resolver_hash": cpu.resolver_hash,
                            "source_query_stage_hash": cpu.source_query_stage_hash,
                            "source_preparation_hash": cpu.source_preparation_hash,
                        }
                    ),
                    seed=0,
                    created_at=cpu.started_at,
                )
                artifacts.ledger.advance_job_lifecycle(
                    job.job_id,
                    (
                        (JobState.PREQUERY_SEALED, seal.sealed_at),
                        (JobState.QUERY_REVEALED, script.instruction.revision.created_at),
                        (JobState.GENERATED, cpu.completed_at),
                    ),
                )
                cpu_record = _cpu_record(
                    inputs=inputs,
                    script=script,
                    cpu=cpu,
                    protocol=protocol,
                )
                resolution = cpu_record.execution.resolution
                resolution_artifact_hash = _persist_phase5_artifact(
                    artifacts=artifacts,
                    value=resolution,
                    media_type=(
                        "application/vnd.story-projection.feedback-resolution+json"
                    ),
                    created_at=resolution.resolved_at,
                )
                projection_artifact_hash = None
                if after is not None:
                    projection_artifact_hash = _persist_phase5_artifact(
                        artifacts=artifacts,
                        value=after,
                        media_type=(
                            "application/vnd.story-projection.ontology-projection+json"
                        ),
                        created_at=cpu.completed_at,
                        release_class=LedgerReleaseClass(after.release_class.value),
                    )
                validation_id = _phase5_cpu_ledger_id("validation", identity_hash)
                accepted = after is not None
                artifacts.ledger.record_validation(
                    validation_id=validation_id,
                    job_id=job.job_id,
                    attempt_id=attempt_id,
                    input_artifact_hash=(
                        projection_artifact_hash
                        if projection_artifact_hash is not None
                        else resolution_artifact_hash
                    ),
                    validator_manifest_hash=cpu.resolver_hash,
                    validation_status=(
                        ValidationStatus.ACCEPTED
                        if accepted
                        else ValidationStatus.REJECTED
                    ),
                    evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
                    temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
                    commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
                    semantic_assessment_scope=(
                        SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
                    ),
                    diagnostics_artifact_hash=resolution_artifact_hash,
                    created_at=cpu.checked_at,
                )
                if after is not None:
                    after_seal = after.construction_seal
                    if after_seal is None or projection_artifact_hash is None:
                        raise Phase5CpuLedgerMaterializationError(
                            "applied CPU feedback lacks its sealed result projection"
                        )
                    artifacts.ledger.record_projection(
                        projection_id=_phase5_cpu_ledger_id("projection", identity_hash),
                        job_id=job.job_id,
                        validation_id=validation_id,
                        snapshot_id=snapshot_id,
                        packet_input_id=packet_input_id,
                        condition_id=cpu.condition.value,
                        context_hash=after.context_hash,
                        upper_ontology_hash=after.upper_ontology.content_hash,
                        construction_certificate_hash=after_seal.content_hash,
                        projection_artifact_hash=projection_artifact_hash,
                        projection_semantic_hash=after.content_hash,
                        release_class=LedgerReleaseClass(after.release_class.value),
                        finalized_at=cpu.checked_at,
                    )
                else:
                    artifacts.ledger.record_failure(
                        attempt_id=attempt_id,
                        failure_kind=FailureKind.OTHER,
                        message=(
                            "CPU condition cannot apply query-time merge/split construction"
                        ),
                        details={
                            "episode_id": script.episode_id,
                            "condition": cpu.condition.value,
                            "resolution_hash": resolution.content_hash,
                            "resolution_status": resolution.status.value,
                        },
                        artifact_hash=resolution_artifact_hash,
                        occurred_at=resolution.resolved_at,
                    )
                artifacts.ledger.advance_job_lifecycle(
                    job.job_id,
                    (
                        (JobState.PREQUERY_SEALED, seal.sealed_at),
                        (JobState.QUERY_REVEALED, script.instruction.revision.created_at),
                        (JobState.GENERATED, cpu.completed_at),
                        (JobState.VALIDATED, cpu.checked_at),
                        (JobState.FINALIZED, cpu.checked_at),
                    ),
                )
                _verify_cpu_job_lifecycle(
                    artifacts=artifacts,
                    job=artifacts.ledger.get_job(job.job_id),
                    sealed_at=seal.sealed_at,
                    revealed_at=script.instruction.revision.created_at,
                    completed_at=cpu.completed_at,
                    checked_at=cpu.checked_at,
                )
    except Phase5ExecutionError:
        raise
    except (KeyError, StoreError, ValueError, OSError) as error:
        raise Phase5CpuLedgerMaterializationError(
            f"Phase 5 CPU ledger materialization failed closed: {error}"
        ) from error


def _require_terminal_receiving_job(job: JobRecord, *, label: str) -> None:
    if job.state not in _TERMINAL_LEDGER_JOB_STATES:
        raise Phase5LedgerPreflightError(
            f"{label} receiving job is not finalized (state={job.state.value})"
        )


def validate_phase5_ledger_preflight(
    *,
    inputs: Phase5ExecutionInputManifest,
    artifacts: ArtifactStore,
    ledger_study_id: str,
) -> None:
    """Prove every pre-existing Phase 5 ledger binding before GPU activation.

    C2 result jobs and projections necessarily arrive after their registered calls, so
    this gate checks their source projections and target study only.  CPU work is
    already materialized in the frozen input and therefore must have its exact job and,
    when applied, result projection in SQLite before production may start.
    """

    try:
        target_study = artifacts.ledger.get_study(ledger_study_id)
    except (KeyError, StoreError, ValueError) as error:
        raise Phase5LedgerPreflightError(
            f"Phase 5 receiving study {ledger_study_id!r} is absent from the ledger"
        ) from error
    if target_study.release_class is not LedgerReleaseClass.RESTRICTED:
        raise Phase5LedgerPreflightError(
            "Phase 5 receiving study must be restricted"
        )

    gaps: list[str] = []
    source_study_ids: set[str] = set()

    def bind_source(
        episode_id: str,
        condition: ConditionName,
        projection: OntologyProjection,
    ) -> None:
        label = f"{episode_id}/{condition.value}"
        try:
            study, _projection = _resolve_exact_ledger_projection(
                artifacts=artifacts,
                projection=projection,
            )
            source_study_ids.add(study.study_id)
            if study.protocol_hash != inputs.held_out_call_manifest_hash:
                gaps.append(f"{label}: source projection belongs to another protocol")
        except (KeyError, StoreError, ValueError, Phase5LedgerPreflightError) as error:
            gaps.append(f"{label}: exact source projection is absent ({error})")

    for script in inputs.scripted_inputs:
        for cpu in script.cpu_inputs:
            label = f"{script.episode_id}/{cpu.condition.value}"
            bind_source(script.episode_id, cpu.condition, cpu.before_projection)
            try:
                job = artifacts.ledger.resolve_job_identity(
                    phase5_cpu_receiving_job_identity(
                        inputs=inputs,
                        ledger_study_id=ledger_study_id,
                        script=script,
                        cpu=cpu,
                    ),
                    study_id=ledger_study_id,
                )
                _require_terminal_receiving_job(job, label=label)
            except (KeyError, StoreError, ValueError, Phase5LedgerPreflightError) as error:
                gaps.append(f"{label}: exact CPU receiving job is absent ({error})")
                job = None
            if cpu.after_projection is not None:
                try:
                    after_study, after = _resolve_exact_ledger_projection(
                        artifacts=artifacts,
                        projection=cpu.after_projection,
                    )
                    if after_study.study_id != ledger_study_id:
                        raise Phase5LedgerPreflightError(
                            "CPU result projection belongs to another study"
                        )
                    if job is not None and after.job_id != job.job_id:
                        raise Phase5LedgerPreflightError(
                            "CPU result projection belongs to another receiving job"
                        )
                except (
                    KeyError,
                    StoreError,
                    ValueError,
                    Phase5LedgerPreflightError,
                ) as error:
                    gaps.append(
                        f"{label}: exact CPU receiving projection is absent ({error})"
                    )
        bind_source(
            script.episode_id,
            ConditionName.C2_LLM_QUERY,
            script.c2_before_projection,
        )
    for trace in inputs.researcher_trace_inputs:
        bind_source(
            trace.episode_id,
            ConditionName.C2_LLM_QUERY,
            trace.c2_before_projection,
        )
    if len(source_study_ids) > 1:
        gaps.append("held-out source projections cross multiple owning studies")
    if gaps:
        raise Phase5LedgerPreflightError(
            "Phase 5 ledger preflight found unbound production lineage: "
            + "; ".join(gaps)
        )


def _phase5_projection_models(
    *,
    inputs: Phase5ExecutionInputManifest,
    record: FeedbackExecutionRecord,
    results_root: Path,
) -> tuple[OntologyProjection, OntologyProjection | None]:
    episode_id = record.execution.episode_id
    condition = record.execution.condition
    scripts = {item.episode_id: item for item in inputs.scripted_inputs}
    if episode_id in scripts:
        script = scripts[episode_id]
        if condition in CPU_CONDITIONS:
            cpu = next(item for item in script.cpu_inputs if item.condition is condition)
            return cpu.before_projection, cpu.after_projection
        before = script.c2_before_projection
    else:
        traces = {item.episode_id: item for item in inputs.researcher_trace_inputs}
        before = traces[episode_id].c2_before_projection
    result = _read(results_root / f"{episode_id}.json", C2RegenerationResult)
    if result.content_hash != record.c2_result_hash:
        raise Phase5LedgerPersistenceError(
            "Phase 5 C2 ledger persistence loaded another result"
        )
    return before, result.after_projection


def _build_phase5_ledger_bindings(
    *,
    inputs: Phase5ExecutionInputManifest,
    records: tuple[FeedbackExecutionRecord, ...],
    results_root: Path,
    artifacts: ArtifactStore,
    ledger_study_id: str,
) -> tuple[_Phase5LedgerResolutionBinding, ...]:
    scripts = {item.episode_id: item for item in inputs.scripted_inputs}
    bindings: list[_Phase5LedgerResolutionBinding] = []
    for record in records:
        execution = record.execution
        before, after = _phase5_projection_models(
            inputs=inputs,
            record=record,
            results_root=results_root,
        )
        source_study, before_row = _resolve_exact_ledger_projection(
            artifacts=artifacts,
            projection=before,
        )
        if source_study.protocol_hash != inputs.held_out_call_manifest_hash:
            raise Phase5LedgerPersistenceError(
                "Phase 5 source projection belongs to another held-out protocol"
            )

        if execution.condition in CPU_CONDITIONS:
            script = scripts[execution.episode_id]
            cpu = next(
                item for item in script.cpu_inputs if item.condition is execution.condition
            )
            receiving_job = artifacts.ledger.resolve_job_identity(
                phase5_cpu_receiving_job_identity(
                    inputs=inputs,
                    ledger_study_id=ledger_study_id,
                    script=script,
                    cpu=cpu,
                ),
                study_id=ledger_study_id,
            )
        else:
            receipt = execution.regeneration_receipt
            if receipt is None:
                raise Phase5LedgerPersistenceError(
                    "C2 condition resolution lacks its model-call receipt"
                )
            receiving_job = artifacts.ledger.resolve_model_call_job(
                study_id=ledger_study_id,
                model_call_ids=tuple(item.model_call_id for item in receipt.ledger_calls),
            )
            if receipt.failure_artifact_hash is not None:
                artifacts.ledger.get_artifact(receipt.failure_artifact_hash)
        _require_terminal_receiving_job(
            receiving_job,
            label=f"{execution.episode_id}/{execution.condition.value}",
        )

        after_row = None
        if after is not None:
            after_study, after_row = _resolve_exact_ledger_projection(
                artifacts=artifacts,
                projection=after,
            )
            if (
                after_study.study_id != ledger_study_id
                or after_row.job_id != receiving_job.job_id
            ):
                raise Phase5LedgerPersistenceError(
                    "Phase 5 after projection is not owned by its receiving study/job"
                )
        expected_after_hash = execution.resolution.after_projection_hash
        if (after is None) != (expected_after_hash is None) or (
            after is not None and after.content_hash != expected_after_hash
        ):
            raise Phase5LedgerPersistenceError(
                "Phase 5 ledger projection binding differs from its resolution"
            )
        bindings.append(
            _Phase5LedgerResolutionBinding(
                record=record,
                before_projection=before_row,
                receiving_job=receiving_job,
                after_projection=after_row,
            )
        )
    return tuple(bindings)


def _persist_phase5_artifact(
    *,
    artifacts: ArtifactStore,
    value: ImmutableRecord,
    media_type: str,
    created_at: datetime,
    release_class: LedgerReleaseClass = LedgerReleaseClass.RESTRICTED,
) -> str:
    payload = _canonical_record_payload(value)
    artifact = artifacts.put_bytes(
        payload,
        media_type=media_type,
        release_class=release_class,
        created_at=created_at,
    )
    if artifact.content_hash != hashlib.sha256(payload).hexdigest():
        raise Phase5LedgerPersistenceError("Phase 5 CAS write changed artifact identity")
    return artifact.content_hash


def _feedback_row_id(
    *,
    ledger_study_id: str,
    episode_id: str,
    condition: ConditionName | None,
    revision_hash: str,
) -> str:
    kind = "user" if condition is None else condition.value.casefold()
    identity = canonical_sha256(
        {
            "study_id": ledger_study_id,
            "episode_id": episode_id,
            "condition": condition,
            "revision_hash": revision_hash,
        }
    )
    return f"phase5-feedback-{kind}-{identity[:32]}"


def _visualization_row_id(
    *,
    feedback_id: str,
    stage: Literal["before", "after"],
    bundle: VisualizationBundle,
) -> str:
    identity = canonical_sha256(
        {
            "feedback_id": feedback_id,
            "stage": stage,
            "bundle_hash": bundle.content_hash,
        }
    )
    return f"phase5-visualization-{stage}-{identity[:32]}"


def persist_phase5_ledger(
    *,
    inputs: Phase5ExecutionInputManifest,
    records: tuple[FeedbackExecutionRecord, ...],
    results_root: Path,
    artifacts: ArtifactStore,
    ledger_study_id: str,
) -> None:
    """Append shared revisions, condition resolutions, and before/after views.

    Every source projection, receiving job, and result projection is resolved before
    the first write.  A partial append caused by process loss is safe to replay because
    all row and artifact identities are deterministic and immutable.
    """

    try:
        study = artifacts.ledger.get_study(ledger_study_id)
        if study.release_class is not LedgerReleaseClass.RESTRICTED:
            raise Phase5LedgerPersistenceError(
                "Phase 5 receiving study must be restricted"
            )
        bindings = _build_phase5_ledger_bindings(
            inputs=inputs,
            records=records,
            results_root=results_root,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
        if len(bindings) != 21:
            raise Phase5LedgerPersistenceError(
                "Phase 5 ledger persistence requires exactly 21 condition resolutions"
            )

        instructions: dict[str, RevisionInstruction] = {}
        for binding in bindings:
            execution = binding.record.execution
            previous = instructions.setdefault(execution.episode_id, execution.instruction)
            if previous != execution.instruction:
                raise Phase5LedgerPersistenceError(
                    "Phase 5 episode conditions do not share one exact revision"
                )
        if len(instructions) != 9:
            raise Phase5LedgerPersistenceError(
                "Phase 5 ledger persistence requires exactly nine user revisions"
            )

        for episode_id, instruction in instructions.items():
            revision = instruction.revision
            artifacts.ledger.record_feedback(
                feedback_id=_feedback_row_id(
                    ledger_study_id=ledger_study_id,
                    episode_id=episode_id,
                    condition=None,
                    revision_hash=revision.content_hash,
                ),
                study_id=ledger_study_id,
                job_id=None,
                feedback_kind=LedgerFeedbackKind.USER_REVISION,
                action=LedgerFeedbackAction(revision.action.value),
                revision_hash=revision.content_hash,
                anchor_manifest_hash=canonical_sha256(
                    tuple(item.content_hash for item in revision.anchors)
                ),
                before_context_hash=revision.before_context_hash,
                after_context_hash=revision.after_context_hash,
                receiving_condition=None,
                before_projection_id=None,
                after_projection_id=None,
                resolution_status=LedgerFeedbackResolutionStatus.PENDING,
                resolution_artifact_hash=None,
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=revision.created_at,
            )

        visualization_configuration = load_visualization_configuration()
        expected_layout_hash = canonical_sha256(
            {
                "algorithm": "preset",
                "coordinate_rule": visualization_configuration.coordinate_rule,
                "seed": visualization_configuration.layout_seed,
                "seed_manifest_hash": visualization_configuration.seed_manifest_hash,
                "layout_seed_entry_hash": (
                    visualization_configuration.layout_seed_entry_hash
                ),
            }
        )
        expected_style_hash = canonical_sha256(
            {
                "style": visualization_configuration.style_name,
                "progressive_disclosure": (
                    visualization_configuration.progressive_disclosure
                ),
            }
        )
        expected_font_hash = canonical_sha256(
            {
                "family": visualization_configuration.font_family,
                "base_px": visualization_configuration.font_base_px,
            }
        )
        for binding in bindings:
            execution = binding.record.execution
            revision = execution.instruction.revision
            feedback_id = _feedback_row_id(
                ledger_study_id=ledger_study_id,
                episode_id=execution.episode_id,
                condition=execution.condition,
                revision_hash=revision.content_hash,
            )
            resolution_artifact_hash = _persist_phase5_artifact(
                artifacts=artifacts,
                value=execution.resolution,
                media_type=(
                    "application/vnd.story-projection.feedback-resolution+json"
                ),
                created_at=execution.resolution.resolved_at,
            )
            status = {
                FeedbackResolutionStatus.RESOLVED: LedgerFeedbackResolutionStatus.APPLIED,
                FeedbackResolutionStatus.CAPABILITY_LIMITED: (
                    LedgerFeedbackResolutionStatus.CAPABILITY_LIMITED
                ),
                FeedbackResolutionStatus.INVALID: LedgerFeedbackResolutionStatus.REJECTED,
            }[execution.resolution.status]
            artifacts.ledger.record_feedback(
                feedback_id=feedback_id,
                study_id=ledger_study_id,
                job_id=binding.receiving_job.job_id,
                feedback_kind=LedgerFeedbackKind.CONDITION_RESOLUTION,
                action=LedgerFeedbackAction(execution.action.value),
                revision_hash=revision.content_hash,
                anchor_manifest_hash=canonical_sha256(
                    tuple(item.content_hash for item in revision.anchors)
                ),
                before_context_hash=revision.before_context_hash,
                after_context_hash=revision.after_context_hash,
                receiving_condition=execution.condition.value,
                before_projection_id=binding.before_projection.projection_id,
                after_projection_id=(
                    None
                    if binding.after_projection is None
                    else binding.after_projection.projection_id
                ),
                resolution_status=status,
                resolution_artifact_hash=resolution_artifact_hash,
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=execution.resolution.resolved_at,
            )

            visualizations = [
                ("before", execution.before_bundle, binding.before_projection)
            ]
            if execution.after_bundle is not None:
                if binding.after_projection is None:
                    raise Phase5LedgerPersistenceError(
                        "after visualization lacks its exact ledger projection"
                    )
                visualizations.append(
                    ("after", execution.after_bundle, binding.after_projection)
                )
            for stage, bundle, projection in visualizations:
                if (
                    bundle.projection_hash != projection.projection_semantic_hash
                    or bundle.state.layout_name
                    != visualization_configuration.layout_name
                    or bundle.state.layout_config_hash != expected_layout_hash
                    or bundle.state.style_config_hash != expected_style_hash
                    or bundle.state.font_config_hash != expected_font_hash
                    or bundle.state.layout_seed != visualization_configuration.layout_seed
                    or bundle.state.viewport != visualization_configuration.viewport
                ):
                    raise Phase5LedgerPersistenceError(
                        "Phase 5 visualization differs from its ledger projection/configuration"
                    )
                visualization_artifact_hash = _persist_phase5_artifact(
                    artifacts=artifacts,
                    value=bundle,
                    media_type=(
                        "application/vnd.story-projection.visualization-bundle+json"
                    ),
                    created_at=execution.resolution.resolved_at,
                    release_class=LedgerReleaseClass(bundle.release_class.value),
                )
                artifacts.ledger.record_visualization(
                    visualization_id=_visualization_row_id(
                        feedback_id=feedback_id,
                        stage=stage,
                        bundle=bundle,
                    ),
                    projection_id=projection.projection_id,
                    renderer_configuration_hash=(
                        visualization_configuration.content_hash
                    ),
                    semantic_hash=bundle.projection_hash,
                    visualization_artifact_hash=visualization_artifact_hash,
                    layout_seed=bundle.state.layout_seed,
                    release_class=LedgerReleaseClass(bundle.release_class.value),
                    created_at=execution.resolution.resolved_at,
                )
    except Phase5ExecutionError:
        raise
    except (KeyError, StoreError, ValueError, OSError) as error:
        raise Phase5LedgerPersistenceError(
            f"Phase 5 ledger persistence failed closed: {error}"
        ) from error


def _append_exact(path: Path, record: ImmutableRecord) -> bool:
    payload = record.to_canonical_json().encode() + b"\n"
    _assert_no_symlink_chain(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise Phase5ExecutionError(f"append-only journal drift: {path.name}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_chain(path.parent)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        pass
    else:
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return True


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5ExecutionError(f"symlinked Phase 5 path is forbidden: {path}")
        if current.parent == current:
            return
        current = current.parent


def _record_name(episode_id: str, condition: ConditionName) -> str:
    return f"{episode_id}--{condition.value}.json"


def _cpu_record(
    *,
    inputs: Phase5ExecutionInputManifest,
    script: ScriptedFeedbackInput,
    cpu: CpuReprojectionInput,
    protocol: FeedbackProtocolConfiguration,
) -> FeedbackExecutionRecord:
    after = cpu.after_projection
    if (
        script.instruction.revision.action is FeedbackAction.REQUEST_MERGE_SPLIT
        and after is not None
    ):
        raise Phase5ExecutionError("C0/C1 merge/split must be capability_limited")
    execution = compile_feedback_execution(
        episode_id=script.episode_id,
        kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
        instruction=script.instruction,
        packet=script.packet,
        condition=cpu.condition,
        before_projection=cpu.before_projection,
        after_projection=after,
        after_packet=None,
        resolver_hash=cpu.resolver_hash,
        seed=protocol.llm_seed,
        resolved_at=cpu.resolved_at,
        checked_at=cpu.checked_at,
        latency_seconds=(cpu.completed_at - cpu.started_at).total_seconds(),
        regeneration_receipt=None,
    )
    seal = cpu.before_projection.construction_seal
    if seal is None:
        raise Phase5ExecutionError("CPU feedback source lacks its construction seal")
    receipt = CpuReprojectionReceipt(
        call_id=f"cpu-{script.episode_id}-{cpu.condition.value}",
        episode_id=script.episode_id,
        condition=cpu.condition,
        instruction_hash=script.instruction.content_hash,
        construction_seal_hash=seal.content_hash,
        before_projection_hash=cpu.before_projection.content_hash,
        after_projection_hash=None if after is None else after.content_hash,
        status=(
            "capability_limited"
            if execution.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
            else "reprojected_same_seal"
        ),
        resolver_hash=cpu.resolver_hash,
        started_at=cpu.started_at,
        completed_at=cpu.completed_at,
    )
    binding = next(item for item in script.scorer_bindings if item.condition is cpu.condition)
    return FeedbackExecutionRecord(
        input_manifest_hash=inputs.content_hash,
        source_input_hash=cpu.content_hash,
        execution=execution,
        cpu_receipt=receipt,
        scorer_binding=binding,
    )


def _c2_record(
    *,
    inputs: Phase5ExecutionInputManifest,
    episode_id: str,
    kind: FeedbackEpisodeKind,
    instruction: RevisionInstruction,
    packet: EvidencePacket,
    before: OntologyProjection,
    protocol: FeedbackProtocolConfiguration,
    adapter: C2RegenerationAdapter,
    ledger_verifier: FeedbackLedgerVerifier,
    scorer_binding: ScorerOnlyFeedbackBinding | None,
    source_input_hash: str,
    call_slots_root: Path,
    results_root: Path,
) -> FeedbackExecutionRecord:
    request = C2RegenerationRequest(
        call_slot_id=f"phase5-c2-{episode_id}",
        episode_id=episode_id,
        kind=kind,
        instruction=instruction,
        packet=packet,
        before_projection_id=before.projection_id,
        before_projection_hash=before.content_hash,
        before_context_hash=before.context_hash,
        before_packet_hash=before.packet_hash,
        protocol_hash=protocol.content_hash,
        seed=protocol.llm_seed,
    )
    created = _append_exact(call_slots_root / f"{episode_id}.json", request)
    result_path = results_root / f"{episode_id}.json"
    if result_path.exists():
        result = _read(result_path, C2RegenerationResult)
    elif created:
        result = adapter.regenerate(request)
    else:
        result = adapter.recover(request)
        if result is None:
            # The adapter's recovery contract makes ``None`` a positive proof
            # that the outer slot was persisted before the adapter performed
            # any side effect.  This is therefore the first issuance, not a
            # resend.  Partial/inflight traces must raise from ``recover``.
            result = adapter.regenerate(request)
    if result.request_hash != request.content_hash:
        raise Phase5ExecutionError("C2 adapter result belongs to another call slot")
    if (
        result.receipt.instruction_hash != instruction.content_hash
        or result.receipt.before_projection_hash != request.before_projection_hash
        or result.receipt.seed != protocol.llm_seed
    ):
        raise Phase5ExecutionError("C2 receipt does not bind its regeneration request")
    ledger_verifier.verify(request, result.receipt)
    _append_exact(result_path, result)
    execution = compile_feedback_execution(
        episode_id=episode_id,
        kind=kind,
        instruction=instruction,
        packet=packet,
        condition=ConditionName.C2_LLM_QUERY,
        before_projection=before,
        after_projection=result.after_projection,
        after_packet=result.after_packet,
        resolver_hash=result.resolver_hash,
        seed=protocol.llm_seed,
        resolved_at=result.resolved_at,
        checked_at=result.checked_at,
        latency_seconds=result.latency_seconds,
        regeneration_receipt=result.receipt,
    )
    return FeedbackExecutionRecord(
        input_manifest_hash=inputs.content_hash,
        source_input_hash=source_input_hash,
        execution=execution,
        c2_request_hash=request.content_hash,
        c2_result_hash=result.content_hash,
        scorer_binding=scorer_binding,
    )


def execute_phase5(
    *,
    inputs: Phase5ExecutionInputManifest,
    protocol: FeedbackProtocolConfiguration,
    prerequisites: PrimaryHeldOutResultsGate,
    adapter: C2RegenerationAdapter,
    ledger_verifier: FeedbackLedgerVerifier,
    output_root: Path,
    completed_at: AwareDatetime | Callable[[], datetime],
    artifacts: ArtifactStore | None = None,
    ledger_study_id: str | None = None,
) -> Phase5JournalIndex:
    """Execute/resume the frozen inventory after prerequisites were verified."""

    validate_phase5_inputs(inputs, protocol)
    if (artifacts is None) != (ledger_study_id is None):
        raise Phase5LedgerPreflightError(
            "ArtifactStore and ledger_study_id must be supplied together"
        )
    if (
        prerequisites.content_hash != inputs.primary_results_gate_hash
        or prerequisites.final_reviewed_seal_hash != inputs.final_reviewed_seal_hash
    ):
        raise Phase5PrerequisiteError("execution does not bind its verified upstream gates")
    if artifacts is not None and ledger_study_id is not None:
        materialize_phase5_cpu_ledger(
            inputs=inputs,
            protocol=protocol,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
        validate_phase5_ledger_preflight(
            inputs=inputs,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
    root = output_root.absolute()
    _assert_no_symlink_chain(root)
    root.mkdir(parents=True, exist_ok=True)
    allowed_dirs = {
        root / "call_slots",
        root / "freezes",
        root / "records",
        root / "results",
    }
    allowed_top_files = {
        root / "input_manifest.json",
        root / "feedback_manifest.json",
        root / "execution_index.json",
    }
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Phase5ExecutionError(f"symlinked Phase 5 journal entry: {path}")
        if path.is_dir() and path not in allowed_dirs:
            raise Phase5ExecutionError(f"unexpected Phase 5 journal directory: {path}")
        if path.is_file() and path.parent == root and path not in allowed_top_files:
            raise Phase5ExecutionError(f"unexpected Phase 5 journal file: {path}")
    _append_exact(root / "input_manifest.json", inputs)
    for script in inputs.scripted_inputs:
        _append_exact(root / "freezes" / f"{script.episode_id}.json", script.freeze)

    records: list[FeedbackExecutionRecord] = []
    scripts = {item.episode_id: item for item in inputs.scripted_inputs}
    traces = {item.episode_id: item for item in inputs.researcher_trace_inputs}
    work: list[tuple[str, ConditionName]] = [
        (selection.episode_id, condition)
        for selection in protocol.scripted_episodes
        for condition in (*CPU_CONDITIONS, ConditionName.C2_LLM_QUERY)
    ]
    work.extend(
        (slot.episode_id, ConditionName.C2_LLM_QUERY) for slot in protocol.researcher_trace_slots
    )
    expected_names = {_record_name(*item) for item in work}
    expected_call_slots = {
        f"{episode_id}.json"
        for episode_id, condition in work
        if condition is ConditionName.C2_LLM_QUERY
    }
    expected_freezes = {f"{item.episode_id}.json" for item in protocol.scripted_episodes}
    freezes_dir = root / "freezes"
    if (
        freezes_dir.exists()
        and {path.name for path in freezes_dir.iterdir() if path.is_file()} - expected_freezes
    ):
        raise Phase5ExecutionError("unexpected freeze in append-only Phase 5 journal")
    records_dir = root / "records"
    call_slots_dir = root / "call_slots"
    results_dir = root / "results"
    if call_slots_dir.exists():
        actual_call_slots = {path.name for path in call_slots_dir.iterdir() if path.is_file()}
        if not actual_call_slots.issubset(expected_call_slots):
            raise Phase5ExecutionError("unexpected C2 call slot in Phase 5 journal")
        if (
            (root / "feedback_manifest.json").exists() or (root / "execution_index.json").exists()
        ) and actual_call_slots != expected_call_slots:
            raise Phase5ExecutionError("finalized Phase 5 journal lost a C2 call slot")
    if records_dir.exists():
        actual_names = {path.name for path in records_dir.iterdir() if path.is_file()}
        if not actual_names.issubset(expected_names):
            raise Phase5ExecutionError("unexpected record in append-only Phase 5 journal")
        if (
            (root / "feedback_manifest.json").exists() or (root / "execution_index.json").exists()
        ) and actual_names != expected_names:
            raise Phase5ExecutionError("finalized Phase 5 journal lost an execution record")
    if results_dir.exists():
        actual_results = {path.name for path in results_dir.iterdir() if path.is_file()}
        if not actual_results.issubset(expected_call_slots):
            raise Phase5ExecutionError("unexpected C2 result in Phase 5 journal")
        if (
            (root / "feedback_manifest.json").exists() or (root / "execution_index.json").exists()
        ) and actual_results != expected_call_slots:
            raise Phase5ExecutionError("finalized Phase 5 journal lost a C2 ITT result")
    for episode_id, condition in work:
        path = records_dir / _record_name(episode_id, condition)
        if path.exists():
            record = _read(path, FeedbackExecutionRecord)
            expected_source = (
                next(
                    item for item in scripts[episode_id].cpu_inputs if item.condition is condition
                ).content_hash
                if episode_id in scripts and condition in CPU_CONDITIONS
                else (
                    scripts[episode_id].content_hash
                    if episode_id in scripts
                    else traces[episode_id].content_hash
                )
            )
            if (
                record.input_manifest_hash != inputs.content_hash
                or record.source_input_hash != expected_source
                or record.execution.episode_id != episode_id
                or record.execution.condition is not condition
            ):
                raise Phase5ExecutionError("resumed record belongs to another input manifest")
            if condition is ConditionName.C2_LLM_QUERY:
                request = _read(call_slots_dir / f"{episode_id}.json", C2RegenerationRequest)
                expected_input = scripts.get(episode_id) or traces[episode_id]
                if (
                    request.content_hash != record.c2_request_hash
                    or request.episode_id != episode_id
                    or request.protocol_hash != protocol.content_hash
                    or request.instruction.content_hash != expected_input.instruction.content_hash
                    or request.before_projection_hash
                    != expected_input.c2_before_projection.content_hash
                ):
                    raise Phase5ExecutionError("resumed C2 call slot lineage changed")
                result = _read(results_dir / f"{episode_id}.json", C2RegenerationResult)
                if (
                    result.content_hash != record.c2_result_hash
                    or result.request_hash != request.content_hash
                    or result.receipt != record.execution.regeneration_receipt
                ):
                    raise Phase5ExecutionError("resumed C2 ITT result lineage changed")
                ledger_verifier.verify(request, result.receipt)
            records.append(record)
            continue
        if episode_id in scripts:
            script = scripts[episode_id]
            if condition in CPU_CONDITIONS:
                cpu = next(item for item in script.cpu_inputs if item.condition is condition)
                record = _cpu_record(inputs=inputs, script=script, cpu=cpu, protocol=protocol)
            else:
                binding = next(
                    item
                    for item in script.scorer_bindings
                    if item.condition is ConditionName.C2_LLM_QUERY
                )
                record = _c2_record(
                    inputs=inputs,
                    episode_id=episode_id,
                    kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
                    instruction=script.instruction,
                    packet=script.packet,
                    before=script.c2_before_projection,
                    protocol=protocol,
                    adapter=adapter,
                    ledger_verifier=ledger_verifier,
                    scorer_binding=binding,
                    source_input_hash=script.content_hash,
                    call_slots_root=call_slots_dir,
                    results_root=results_dir,
                )
        else:
            trace = traces[episode_id]
            record = _c2_record(
                inputs=inputs,
                episode_id=episode_id,
                kind=FeedbackEpisodeKind.RESEARCHER_TRACE,
                instruction=trace.instruction,
                packet=trace.packet,
                before=trace.c2_before_projection,
                protocol=protocol,
                adapter=adapter,
                ledger_verifier=ledger_verifier,
                scorer_binding=None,
                source_input_hash=trace.content_hash,
                call_slots_root=call_slots_dir,
                results_root=results_dir,
            )
        _append_exact(path, record)
        records.append(record)

    resolved_completed_at = completed_at() if callable(completed_at) else completed_at
    if (
        not isinstance(resolved_completed_at, datetime)
        or resolved_completed_at.tzinfo is None
        or resolved_completed_at.utcoffset() is None
    ):
        raise Phase5ExecutionError("Phase 5 completion clock must return an aware datetime")
    manifest = FeedbackStudyExecutionManifest(
        protocol_hash=protocol.content_hash,
        scripted_revision_freezes=tuple(item.freeze for item in inputs.scripted_inputs),
        executions=tuple(item.execution for item in records),
        generated_at=resolved_completed_at,
    )
    assert_feedback_manifest_matches_protocol(manifest, protocol)
    if any(
        item.execution.replay is not None
        and item.execution.replay.checked_at > resolved_completed_at
        for item in records
    ):
        raise Phase5ExecutionError("Phase 5 completion timestamp predates replay validation")
    if resolved_completed_at < inputs.frozen_at or any(
        item.cpu_receipt is not None and item.cpu_receipt.completed_at > resolved_completed_at
        for item in records
    ):
        raise Phase5ExecutionError("Phase 5 completion timestamp predates frozen work")
    c2_receipts = [
        item.execution.regeneration_receipt
        for item in records
        if item.execution.condition is ConditionName.C2_LLM_QUERY
    ]
    if len(c2_receipts) != 9 or any(item is None for item in c2_receipts):
        raise Phase5ExecutionError("Phase 5 did not retain all nine C2 call receipts")
    concrete_c2_receipts = [item for item in c2_receipts if item is not None]
    if any(item.completed_at > resolved_completed_at for item in concrete_c2_receipts):
        raise Phase5ExecutionError("Phase 5 completion timestamp predates GPU work")
    if Counter(item.execution.kind for item in records) != Counter(
        {
            FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER: 18,
            FeedbackEpisodeKind.RESEARCHER_TRACE: 3,
        }
    ):
        raise Phase5ExecutionError("Phase 5 execution inventory changed")
    if artifacts is not None and ledger_study_id is not None:
        persist_phase5_ledger(
            inputs=inputs,
            records=tuple(records),
            results_root=results_dir,
            artifacts=artifacts,
            ledger_study_id=ledger_study_id,
        )
    _append_exact(root / "feedback_manifest.json", manifest)

    def artifact_binding(
        relative: str,
        artifact_kind: Literal[
            "input_manifest",
            "script_freeze",
            "c2_call_slot",
            "c2_result",
            "execution_record",
            "feedback_manifest",
        ],
        logical_hash: str,
    ) -> Phase5ArtifactBinding:
        path = root / relative
        return Phase5ArtifactBinding(
            relative_path=relative,
            artifact_kind=artifact_kind,
            logical_content_hash=logical_hash,
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    artifact_bindings = [
        artifact_binding("input_manifest.json", "input_manifest", inputs.content_hash)
    ]
    artifact_bindings.extend(
        artifact_binding(
            f"freezes/{script.episode_id}.json",
            "script_freeze",
            script.freeze.content_hash,
        )
        for script in inputs.scripted_inputs
    )
    c2_records = [
        item for item in records if item.execution.condition is ConditionName.C2_LLM_QUERY
    ]
    artifact_bindings.extend(
        artifact_binding(
            f"call_slots/{item.execution.episode_id}.json",
            "c2_call_slot",
            item.c2_request_hash,
        )
        for item in c2_records
        if item.c2_request_hash is not None
    )
    artifact_bindings.extend(
        artifact_binding(
            f"results/{item.execution.episode_id}.json",
            "c2_result",
            item.c2_result_hash,
        )
        for item in c2_records
        if item.c2_result_hash is not None
    )
    artifact_bindings.extend(
        artifact_binding(
            f"records/{_record_name(item.execution.episode_id, item.execution.condition)}",
            "execution_record",
            item.content_hash,
        )
        for item in records
    )
    artifact_bindings.append(
        artifact_binding("feedback_manifest.json", "feedback_manifest", manifest.content_hash)
    )
    ledger_calls = [call for receipt in concrete_c2_receipts for call in receipt.ledger_calls]
    index = Phase5JournalIndex(
        run_id=inputs.run_id,
        input_manifest_hash=inputs.content_hash,
        protocol_hash=protocol.content_hash,
        primary_results_gate_hash=inputs.primary_results_gate_hash,
        final_reviewed_seal_hash=inputs.final_reviewed_seal_hash,
        feedback_manifest_hash=manifest.content_hash,
        execution_record_hashes=tuple(item.content_hash for item in records),
        c2_request_hashes=tuple(
            item.c2_request_hash for item in records if item.c2_request_hash is not None
        ),
        c2_result_hashes=tuple(
            item.c2_result_hash for item in c2_records if item.c2_result_hash is not None
        ),
        ledger_model_call_hashes=tuple(item.model_call_record_hash for item in ledger_calls),
        ledger_gpu_event_hashes=tuple(item.gpu_event_record_hash for item in ledger_calls),
        actual_gpu_request_count=len(ledger_calls),
        repair_gpu_request_count=sum(item.attempt_kind == "repair" for item in ledger_calls),
        allocated_gpu_seconds=sum(item.allocated_gpu_seconds for item in ledger_calls),
        artifact_bindings=tuple(artifact_bindings),
        completed_at=resolved_completed_at,
    )
    _append_exact(root / "execution_index.json", index)
    return index


def run_phase5_from_files(
    *,
    input_manifest_path: Path,
    materialization_receipt_path: Path,
    primary_results_gate_path: Path,
    benchmark_root: Path,
    review_completion_root: Path,
    output_root: Path,
    adapter: C2RegenerationAdapter,
    ledger_verifier: FeedbackLedgerVerifier,
    completed_at: AwareDatetime | Callable[[], datetime],
    protocol_path: Path | None = None,
) -> Phase5JournalIndex:
    inputs = load_phase5_input_manifest(input_manifest_path)
    from story_projection_onto.phase5_production import (
        load_phase5_materialization_receipt,
        validate_phase5_materialization_receipt,
    )

    validate_phase5_materialization_receipt(
        inputs,
        load_phase5_materialization_receipt(materialization_receipt_path),
    )
    protocol = (
        load_feedback_protocol() if protocol_path is None else load_feedback_protocol(protocol_path)
    )
    validate_phase5_inputs(inputs, protocol)
    prerequisites = validate_phase5_prerequisites(
        inputs=inputs,
        primary_results_gate_path=primary_results_gate_path,
        benchmark_root=benchmark_root,
        review_completion_root=review_completion_root,
    )
    return execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=prerequisites,
        adapter=adapter,
        ledger_verifier=ledger_verifier,
        output_root=output_root,
        completed_at=completed_at,
    )


__all__ = [
    "CPU_CONDITIONS",
    "DEFAULT_PHASE5_OUTPUT_ROOT",
    "DEFAULT_PHASE5_RUNNER_CONFIG_PATH",
    "C2RegenerationAdapter",
    "C2RegenerationRequest",
    "C2RegenerationResult",
    "CpuReprojectionInput",
    "CpuReprojectionReceipt",
    "FeedbackExecutionRecord",
    "FeedbackLedgerVerifier",
    "InterruptedFeedbackCallRecoveryRequired",
    "Phase5ArtifactBinding",
    "Phase5CpuLedgerMaterializationError",
    "Phase5ExecutionError",
    "Phase5ExecutionInputManifest",
    "Phase5JournalIndex",
    "Phase5LedgerPersistenceError",
    "Phase5LedgerPreflightError",
    "Phase5PrerequisiteError",
    "Phase5RunnerConfiguration",
    "PrimaryHeldOutResultsGate",
    "ResearcherTraceInput",
    "SQLiteFeedbackLedgerVerifier",
    "ScorerOnlyFeedbackBinding",
    "ScriptedFeedbackInput",
    "execute_phase5",
    "feedback_gpu_event_record_hash",
    "feedback_model_call_record_hash",
    "load_phase5_input_manifest",
    "load_phase5_runner_configuration",
    "materialize_phase5_cpu_ledger",
    "persist_phase5_ledger",
    "phase5_cpu_receiving_job_identity",
    "replay_phase5_prerequisites",
    "run_phase5_from_files",
    "validate_phase5_inputs",
    "validate_phase5_ledger_preflight",
    "validate_phase5_prerequisites",
]
