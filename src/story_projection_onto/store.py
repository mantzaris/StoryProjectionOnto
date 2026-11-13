"""Append-only experiment storage, accounting, and resource preflight.

The scientific payload is kept out of SQLite.  Payloads live in a bounded,
content-addressed blob store while the database records immutable hashes and
lineage.  All mutable-looking job progress is represented as an append-only
transition log; database triggers reject updates and deletes even if a caller
bypasses this module.

Compression is part of an artifact's identity metadata.  Asking for zstd when
the optional codec is unavailable is an error: this module never writes gzip
bytes under a ``zstd`` label (or silently changes the requested codec).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import decimal
import enum
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import stat
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

SCHEMA_VERSION = 8
DECIMAL_GIGABYTE = 1_000_000_000
DEFAULT_TOTAL_ALLOCATION_BYTES = 30 * DECIMAL_GIGABYTE
DEFAULT_MAX_OCCUPIED_BYTES = 25 * DECIMAL_GIGABYTE
DEFAULT_MIN_HEADROOM_BYTES = 5 * DECIMAL_GIGABYTE
DEFAULT_GPU_HARD_LIMIT_SECONDS = 10 * 60 * 60
DEFAULT_MAX_BLOB_RAW_BYTES = 64 * 1024 * 1024

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MICROSECONDS_PER_SECOND = decimal.Decimal(1_000_000)


class StoreError(RuntimeError):
    """Base class for persistence and accounting failures."""


class DuplicateConflictError(StoreError):
    """An idempotency key was reused with different immutable content."""


class InvalidTransitionError(StoreError):
    """A requested job transition is not in the frozen state machine."""


class ArtifactIntegrityError(StoreError):
    """A stored blob does not match its immutable metadata."""


class CompressionUnavailableError(StoreError):
    """The exact requested compression codec is unavailable."""


class BlobTooLargeError(StoreError):
    """A payload exceeds the configured one-blob bound."""


class ReleaseViolationError(StoreError):
    """Restricted material was requested through a public-release path."""


class StorageBudgetExceeded(StoreError):
    """A storage preflight would violate occupancy or headroom limits."""

    def __init__(self, report: StorageReport) -> None:
        self.report = report
        super().__init__("storage preflight failed: " + ", ".join(report.violations))


class GpuBudgetExceeded(StoreError):
    """Starting a GPU allocation would reach or cross the hard stop."""


class ReleaseClass(enum.StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"


class Compression(enum.StrEnum):
    GZIP = "gzip"
    ZSTD = "zstd"


class JobState(enum.StrEnum):
    PLANNED = "planned"
    PREQUERY_SEALED = "prequery_sealed"
    QUERY_REVEALED = "query_revealed"
    GENERATED = "generated"
    REPAIRED = "repaired"
    VALIDATED = "validated"
    FINALIZED = "finalized"
    SCORED = "scored"
    RENDERED = "rendered"


ALLOWED_JOB_TRANSITIONS: Mapping[JobState, frozenset[JobState]] = {
    JobState.PLANNED: frozenset({JobState.PREQUERY_SEALED}),
    # Query-blind C0/C1 preconstruction has no query-access event.  Those jobs
    # take the typed PREQUERY_SEALED -> GENERATED branch; every query-time job
    # must still pass through QUERY_REVEALED.  ``transition_job`` verifies the
    # immutable job identity before permitting the branch.
    JobState.PREQUERY_SEALED: frozenset(
        {JobState.QUERY_REVEALED, JobState.GENERATED}
    ),
    JobState.QUERY_REVEALED: frozenset({JobState.GENERATED}),
    JobState.GENERATED: frozenset({JobState.REPAIRED, JobState.VALIDATED}),
    JobState.REPAIRED: frozenset({JobState.VALIDATED}),
    JobState.VALIDATED: frozenset({JobState.FINALIZED}),
    JobState.FINALIZED: frozenset({JobState.SCORED}),
    JobState.SCORED: frozenset({JobState.RENDERED}),
    JobState.RENDERED: frozenset(),
}


# One already-durable fallback C1 attempt predates typed lifecycle identities. Its
# second-recovery retry must remain in the same job so the RETRY parent stays a
# real same-job lineage edge; rewriting the historical job identity or inventing
# a query reveal would be scientifically false. This frozen record is the only
# legacy job allowed to take the query-blind PREQUERY_SEALED -> GENERATED branch.
_LEGACY_FALLBACK_C1_RETRY_JOB_IDENTITY: Mapping[str, str] = {
    "run_id": "fallback-qwen3-8b-awq-development-v3",
    "call_id": "fallback-c1-01",
    "plan_hash": "24bd99189fdf5157d6de1c3c13edaa0a86749c97b1b4957b220e7a62de9aa200",
}
_LEGACY_FALLBACK_C1_RETRY_JOB_ID = (
    "f5646ba427f98e76afc07fabcb6c53f65d73d255856493ac94bf9d0c642f444f"
)
_LEGACY_FALLBACK_C1_PARENT_ATTEMPT_ID = (
    "fallback-qwen3-8b-awq-development-v3-fallback-c1-01-attempt"
)
_LEGACY_FALLBACK_C1_PARENT_MODEL_CALL_ID = (
    "fallback-qwen3-8b-awq-development-v3-fallback-c1-01"
)
_LEGACY_FALLBACK_C1_PARENT_GPU_EVENT_ID = (
    "fallback-qwen3-8b-awq-development-v3-fallback-c1-01-gpu"
)
_LEGACY_FALLBACK_C1_REQUEST_HASH = (
    "1cc73c5525e096a4df830892f37cdc8062899363a0b75835bb2f04b3a14a0d44"
)
_LEGACY_FALLBACK_C1_CONSTRUCTION_UNIT_HASH = (
    "c91e2eeb87d9f9c05713396b3403ddab702573da73c14893eb7ed0c7158e6317"
)
_LEGACY_FALLBACK_C1_ALLOCATED_MICROSECONDS = 852_878


def frozen_legacy_fallback_c1_retry_prebuild_matches(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    identity: Mapping[str, Any],
) -> bool:
    """Authenticate the sole pre-typed query-blind recovery job.

    This is intentionally not a general legacy escape hatch. Both the writable
    transition path and the independent read-only verifier call this predicate,
    which binds the exact immutable v3 job, failed attempt, model call, metered
    event, and failure row before accepting a direct query-blind generation edge.
    """

    if (
        job_id != _LEGACY_FALLBACK_C1_RETRY_JOB_ID
        or dict(identity) != dict(_LEGACY_FALLBACK_C1_RETRY_JOB_IDENTITY)
    ):
        return False
    attempt = connection.execute(
        "SELECT * FROM attempts WHERE attempt_id = ?",
        (_LEGACY_FALLBACK_C1_PARENT_ATTEMPT_ID,),
    ).fetchone()
    model_call = connection.execute(
        "SELECT * FROM model_calls WHERE model_call_id = ?",
        (_LEGACY_FALLBACK_C1_PARENT_MODEL_CALL_ID,),
    ).fetchone()
    event = connection.execute(
        "SELECT * FROM gpu_events WHERE event_id = ?",
        (_LEGACY_FALLBACK_C1_PARENT_GPU_EVENT_ID,),
    ).fetchone()
    failures = connection.execute(
        """SELECT * FROM failures WHERE attempt_id = ?
           ORDER BY failure_id""",
        (_LEGACY_FALLBACK_C1_PARENT_ATTEMPT_ID,),
    ).fetchall()
    if attempt is None or model_call is None or event is None or len(failures) != 1:
        return False
    failure = failures[0]
    try:
        failure_details = json.loads(failure["details_json"])
        event_details = json.loads(event["details_json"])
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(
        attempt["job_id"] == job_id
        and attempt["attempt_kind"] == AttemptKind.BASE.value
        and attempt["parent_attempt_id"] is None
        and attempt["input_hash"] == _LEGACY_FALLBACK_C1_REQUEST_HASH
        and attempt["config_hash"] == model_call["decoding_manifest_hash"]
        and attempt["seed"] == 0
        and model_call["job_id"] == job_id
        and model_call["attempt_id"] == _LEGACY_FALLBACK_C1_PARENT_ATTEMPT_ID
        and model_call["gpu_event_id"] == _LEGACY_FALLBACK_C1_PARENT_GPU_EVENT_ID
        and model_call["backend"] == ModelBackend.VLLM_GPU.value
        and model_call["call_role"] == ModelCallRole.PILOT.value
        and model_call["retry_class"] == RetryClass.LONG.value
        and model_call["request_hash"] == _LEGACY_FALLBACK_C1_REQUEST_HASH
        and model_call["response_artifact_hash"] is None
        and model_call["construction_unit_hash"]
        == _LEGACY_FALLBACK_C1_CONSTRUCTION_UNIT_HASH
        and model_call["prompt_tokens"] == 0
        and model_call["completion_tokens"] == 0
        and model_call["allocated_gpu_microseconds"]
        == _LEGACY_FALLBACK_C1_ALLOCATED_MICROSECONDS
        and model_call["successful"] == 0
        and event["job_id"] == job_id
        and event["attempt_id"] == _LEGACY_FALLBACK_C1_PARENT_ATTEMPT_ID
        and event["event_kind"] == GpuEventKind.FAILURE.value
        and event["allocated_microseconds"]
        == _LEGACY_FALLBACK_C1_ALLOCATED_MICROSECONDS
        and event["succeeded"] == 0
        and event_details.get("reserve_call_class") == "reserve_long"
        and event_details.get("reserve_reservation_id")
        == "fallback-qwen3-8b-awq-development-v3:fallback-c1-01"
        and failure["failure_kind"] == FailureKind.SERVICE.value
        and failure["message"] == "Fallback micro-pilot model call failed"
        and failure["artifact_hash"] is None
        and failure_details
        == {
            "call_id": "fallback-c1-01",
            "exception_type": "RuntimeTransportError",
        }
    )


class AttemptKind(enum.StrEnum):
    BASE = "base"
    REPAIR = "repair"
    RETRY = "retry"


class FailureKind(enum.StrEnum):
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    VALIDATION = "validation"
    INTERRUPTED = "interrupted"
    SERVICE = "service"
    OTHER = "other"


class GpuEventKind(enum.StrEnum):
    """Mutually exclusive allocated-GPU service intervals.

    Every row contributes to actual allocated time, including unsuccessful
    work.  Callers must not also record an enclosing interval for the same
    elapsed period, which would double-count it.
    """

    GPU_SESSION_START = "gpu_session_start"
    # Compatibility spelling for callers that treat the event as service allocation.
    SERVICE_START = "gpu_session_start"
    MODEL_LOAD = "model_load"
    WARM_UP = "warm_up"
    SCHEMA_PROBE = "schema_probe"
    INFERENCE = "inference"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    REPAIR = "repair"
    RESTART = "restart"
    FALLBACK_TEST = "fallback_test"
    RERUN = "rerun"
    # Durable complement of classified events within a live model-service
    # session. Stored in gpu_service_sessions, never as an enclosing event.
    SERVICE_OVERHEAD = "service_overhead"


class GpuAllocationJournalState(enum.StrEnum):
    OPENED = "opened"
    HEARTBEAT = "heartbeat"
    CLOSED = "closed"
    RECOVERED = "recovered"


class GpuServiceJournalState(enum.StrEnum):
    """Durable lifecycle states for one exclusive GPU model service.

    The journal is distinct from ``gpu_service_sessions``: it is written while
    a service is live so controller death cannot erase idle, load, or restart
    allocation.  ``CLOSED`` and ``RECOVERED`` are terminal observations made
    only after the immutable service-session accounting row exists.
    """

    OPENED = "opened"
    HEARTBEAT = "heartbeat"
    PROCESS_STOPPED = "process_stopped"
    CLOSED = "closed"
    RECOVERED = "recovered"


class InputKind(enum.StrEnum):
    EVIDENCE_SNAPSHOT = "evidence_snapshot"
    EVIDENCE_PACKET = "evidence_packet"
    CONFIGURATION = "configuration"
    PROMPT_TEMPLATE = "prompt_template"
    OUTPUT_SCHEMA = "output_schema"
    UPPER_ONTOLOGY = "upper_ontology"
    SEALED_ONTOLOGY = "sealed_ontology"
    OTHER_MANIFEST = "other_manifest"


class ModelBackend(enum.StrEnum):
    VLLM_GPU = "vllm_gpu"
    HAND_AUTHORED_FIXTURE = "hand_authored_fixture"


class ModelCallRole(enum.StrEnum):
    PREBUILD = "prebuild"
    QUERY_TIME = "query_time"
    FIXED_SELECT = "fixed_select"
    REPAIR = "repair"
    PILOT = "pilot"


class RetryClass(enum.StrEnum):
    BASE = "base"
    LONG = "long"
    STANDARD = "standard"
    SHORT = "short"


class ValidationStatus(enum.StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"


class EvidenceSupportStatus(enum.StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


class TemporalValidationStatus(enum.StrEnum):
    VALID = "valid"
    UNDERDETERMINED = "underdetermined"
    CONTRADICTION = "contradiction"
    NOT_APPLICABLE = "not_applicable"


class CommitmentCheckStatus(enum.StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    NOT_APPLICABLE = "not_applicable"


class SemanticAssessmentScope(enum.StrEnum):
    """Whether semantic correctness was actually assessed for a validation row."""

    RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED = "runtime_structural_only_not_assessed"
    POSTHOC_SCORER_OR_REVIEWER = "posthoc_scorer_or_reviewer"
    LEGACY_UNSPECIFIED = "legacy_unspecified"


class FeedbackKind(enum.StrEnum):
    USER_REVISION = "user_revision"
    CONDITION_RESOLUTION = "condition_resolution"


class FeedbackAction(enum.StrEnum):
    REFINE_CONTEXT = "REFINE_CONTEXT"
    REQUEST_MERGE_SPLIT = "REQUEST_MERGE_SPLIT"


class FeedbackResolutionStatus(enum.StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    CAPABILITY_LIMITED = "capability_limited"
    REJECTED = "rejected"


class MetricStatus(enum.StrEnum):
    VALUE = "value"
    NOT_APPLICABLE = "not_applicable"
    UNDEFINED = "undefined"
    INVALID = "invalid"


@dataclass(frozen=True)
class ArtifactRecord:
    content_hash: str
    compression: Compression
    media_type: str
    raw_size_bytes: int
    stored_size_bytes: int
    relative_path: str
    release_class: ReleaseClass
    created_at: str


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    identity_hash: str
    identity_json: str
    release_class: ReleaseClass
    state: JobState
    created_at: str


@dataclass(frozen=True)
class JobTransition:
    event_id: int
    job_id: str
    sequence: int
    from_state: JobState | None
    to_state: JobState
    occurred_at: str


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    job_id: str
    attempt_kind: AttemptKind
    parent_attempt_id: str | None
    input_hash: str
    config_hash: str
    seed: int
    created_at: str


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    attempt_id: str
    failure_kind: FailureKind
    message: str
    details_json: str
    artifact_hash: str | None
    occurred_at: str


@dataclass(frozen=True)
class GpuEvent:
    event_id: str
    event_kind: GpuEventKind
    allocated_microseconds: int
    started_at: str
    ended_at: str
    succeeded: bool | None
    job_id: str | None
    attempt_id: str | None
    details_json: str

    @property
    def allocated_seconds(self) -> float:
        return self.allocated_microseconds / 1_000_000


@dataclass(frozen=True)
class GpuServiceSession:
    """Durable, non-double-counting reconciliation of one live GPU service."""

    service_session_id: str
    session_id: str
    service_microseconds: int
    classified_event_microseconds: int
    overhead_microseconds: int
    started_at: str
    ended_at: str
    details_json: str

    @property
    def service_seconds(self) -> float:
        return self.service_microseconds / 1_000_000

    @property
    def overhead_seconds(self) -> float:
        return self.overhead_microseconds / 1_000_000


@dataclass(frozen=True)
class GpuAllocationJournalRecord:
    journal_id: str
    allocation_id: str
    sequence: int
    state: GpuAllocationJournalState
    intended_event_kind: GpuEventKind
    elapsed_microseconds: int
    maximum_microseconds: int
    observed_at: str
    job_id: str | None
    attempt_id: str | None
    details_json: str


@dataclass(frozen=True)
class GpuServiceJournalRecord:
    journal_id: str
    service_session_id: str
    sequence: int
    state: GpuServiceJournalState
    session_id: str
    configuration_hash: str
    service_started_at: str
    elapsed_microseconds: int
    ledger_allocated_microseconds_before_session: int
    hard_limit_microseconds: int
    observed_at: str
    details_json: str


@dataclass(frozen=True)
class GpuSummary:
    total_allocated_microseconds: int
    event_count: int
    by_kind_microseconds: tuple[tuple[GpuEventKind, int], ...]
    service_session_count: int = 0

    @property
    def total_allocated_seconds(self) -> float:
        return self.total_allocated_microseconds / 1_000_000

    @property
    def total_allocated_hours(self) -> float:
        return self.total_allocated_seconds / 3600

    def seconds_for(self, kind: GpuEventKind) -> float:
        wanted = GpuEventKind(kind)
        return dict(self.by_kind_microseconds).get(wanted, 0) / 1_000_000


@dataclass(frozen=True)
class StudyRecord:
    study_id: str
    protocol_hash: str
    code_manifest_hash: str
    configuration_hash: str
    release_class: ReleaseClass
    created_at: str


@dataclass(frozen=True)
class StudyJobRecord:
    link_id: str
    study_id: str
    job_id: str
    created_at: str


@dataclass(frozen=True)
class InputRecord:
    input_id: str
    study_id: str
    input_kind: InputKind
    content_hash: str
    artifact_hash: str | None
    release_class: ReleaseClass
    created_at: str


@dataclass(frozen=True)
class EvidenceSnapshotRecord:
    snapshot_id: str
    input_id: str
    horizon_hash: str
    evidence_manifest_hash: str
    index_configuration_hash: str
    prequery_seal_hash: str
    eligible_evidence_count: int
    created_at: str


@dataclass(frozen=True)
class PrequeryBarrierRecord:
    barrier_hash: str
    barrier_id: str
    execution_id: str
    execution_manifest_hash: str
    barrier_artifact_hash: str
    preparation_count: int
    sealed_at: str
    persisted_at: str
    release_class: ReleaseClass


@dataclass(frozen=True)
class QueryAccessRecord:
    access_event_hash: str
    access_event_id: str
    execution_id: str
    query_context_hash: str
    model_visible_query_hash: str
    snapshot_hash: str
    stage_manifest_hash: str
    query_artifact_hash: str
    prequery_barrier_hash: str
    packet_hash: str | None
    query_payload_artifact_hash: str
    access_event_artifact_hash: str
    registered_revealed_at: str
    accessed_at: str
    release_class: ReleaseClass


@dataclass(frozen=True)
class PacketMaterializationRecord:
    materialization_event_hash: str
    materialization_event_id: str
    execution_id: str
    query_access_event_hash: str
    snapshot_hash: str
    packet_hash: str
    retrieval_method: str
    retrieval_config_hash: str
    packet_artifact_hash: str
    materialization_event_artifact_hash: str
    started_at: str
    completed_at: str
    release_class: ReleaseClass


@dataclass(frozen=True)
class ModelCallRecord:
    model_call_id: str
    job_id: str
    attempt_id: str
    gpu_event_id: str | None
    backend: ModelBackend
    call_role: ModelCallRole
    retry_class: RetryClass
    model_manifest_hash: str
    decoding_manifest_hash: str
    request_hash: str
    response_artifact_hash: str | None
    construction_unit_hash: str
    served_context_count: int
    prompt_tokens: int
    completion_tokens: int
    allocated_gpu_microseconds: int
    successful: bool
    created_at: str


@dataclass(frozen=True)
class ValidationRecord:
    validation_id: str
    job_id: str
    attempt_id: str
    input_artifact_hash: str
    validator_manifest_hash: str
    validation_status: ValidationStatus
    evidence_support_status: EvidenceSupportStatus
    temporal_status: TemporalValidationStatus
    commitment_status: CommitmentCheckStatus
    semantic_assessment_scope: SemanticAssessmentScope
    diagnostics_artifact_hash: str | None
    parent_validation_id: str | None
    repair_attempt_id: str | None
    created_at: str


@dataclass(frozen=True)
class ProjectionRecord:
    projection_id: str
    job_id: str
    validation_id: str
    snapshot_id: str
    packet_input_id: str
    condition_id: str
    context_hash: str
    upper_ontology_hash: str
    construction_certificate_hash: str
    projection_artifact_hash: str
    projection_semantic_hash: str | None
    parent_projection_id: str | None
    release_class: ReleaseClass
    finalized_at: str


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    study_id: str
    job_id: str | None
    feedback_kind: FeedbackKind
    action: FeedbackAction
    revision_hash: str
    anchor_manifest_hash: str
    before_context_hash: str
    after_context_hash: str
    receiving_condition: str | None
    before_projection_id: str | None
    after_projection_id: str | None
    resolution_status: FeedbackResolutionStatus
    resolution_artifact_hash: str | None
    release_class: ReleaseClass
    created_at: str


@dataclass(frozen=True)
class MetricRecord:
    metric_id: str
    study_id: str
    job_id: str | None
    projection_id: str | None
    unit_hash: str
    metric_name: str
    metric_version_hash: str
    status: MetricStatus
    value: float | None
    numerator: float | None
    denominator: float | None
    result_artifact_hash: str | None
    created_at: str


@dataclass(frozen=True)
class VisualizationRecord:
    visualization_id: str
    projection_id: str
    renderer_configuration_hash: str
    semantic_hash: str
    visualization_artifact_hash: str
    layout_seed: int
    release_class: ReleaseClass
    created_at: str


@dataclass(frozen=True)
class ResourceSampleRecord:
    sample_id: str
    job_id: str | None
    gpu_event_id: str | None
    process_ram_bytes: int
    system_available_ram_bytes: int
    gpu_vram_bytes: int
    project_storage_bytes: int
    cpu_worker_count: int
    sampled_at: str


@dataclass(frozen=True)
class StorageSampleRecord:
    """Durable storage/headroom observation used for interrupted-run replay."""

    sample_id: str
    phase: str
    sampled_at: str
    current_occupied_bytes: int
    additional_reserved_bytes: int
    projected_occupied_bytes: int
    filesystem_free_bytes: int
    effective_projected_headroom_bytes: int
    allowed: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class StorageBudget:
    total_allocation_bytes: int = DEFAULT_TOTAL_ALLOCATION_BYTES
    max_occupied_bytes: int = DEFAULT_MAX_OCCUPIED_BYTES
    min_headroom_bytes: int = DEFAULT_MIN_HEADROOM_BYTES

    def __post_init__(self) -> None:
        _nonnegative_int("total_allocation_bytes", self.total_allocation_bytes)
        _nonnegative_int("max_occupied_bytes", self.max_occupied_bytes)
        _nonnegative_int("min_headroom_bytes", self.min_headroom_bytes)
        if self.max_occupied_bytes + self.min_headroom_bytes > self.total_allocation_bytes:
            raise ValueError("max occupied plus minimum headroom exceeds the writable allocation")


@dataclass(frozen=True)
class StorageReport:
    current_occupied_bytes: int
    declared_growth_bytes: int
    largest_atomic_temporary_bytes: int
    quarantine_allowance_bytes: int
    release_staging_bytes: int
    projected_occupied_bytes: int
    filesystem_free_bytes: int
    projected_allocation_free_bytes: int
    projected_filesystem_free_bytes: int
    effective_projected_headroom_bytes: int
    budget: StorageBudget
    allowed: bool
    violations: tuple[str, ...]

    @property
    def additional_reserved_bytes(self) -> int:
        return (
            self.declared_growth_bytes
            + self.largest_atomic_temporary_bytes
            + self.quarantine_allowance_bytes
            + self.release_staging_bytes
        )


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject non-standard NaN values."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def derive_job_identity(identity_components: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(identity_components).encode("utf-8"))


def _normalise_hash(name: str, value: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _metadata_token(name: str, value: str, *, maximum_length: int = 256) -> str:
    """Validate a compact metadata identifier, never a prompt or source-text field."""

    if not isinstance(value, str) or not value or len(value) > maximum_length:
        raise ValueError(f"{name} must be a nonempty metadata token")
    if any(character in value for character in ("\n", "\r", "\x00")):
        raise ValueError(f"{name} cannot contain line breaks or NUL bytes")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_optional_number(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise ValueError(f"{name} must be a finite number or None")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _normalise_timestamp(value: Any | None = None) -> str:
    if value is None:
        parsed = dt.datetime.now(dt.UTC)
    elif isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, str):
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = dt.datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc
    else:
        raise TypeError("timestamp must be an aware datetime, ISO-8601 string, or None")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed.astimezone(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)


def _seconds_to_microseconds(seconds: Any) -> int:
    try:
        numeric = decimal.Decimal(str(seconds))
    except decimal.InvalidOperation as exc:
        raise ValueError("allocated_seconds must be a finite nonnegative number") from exc
    if not numeric.is_finite() or numeric < 0:
        raise ValueError("allocated_seconds must be a finite nonnegative number")
    micros = (numeric * _MICROSECONDS_PER_SECOND).quantize(
        decimal.Decimal(1), rounding=decimal.ROUND_HALF_UP
    )
    return int(micros)


def _import_zstandard() -> Any:
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CompressionUnavailableError(
            "zstd compression was requested but the 'zstandard' package is unavailable"
        ) from exc
    return zstandard


class BlobStore:
    """A fixed-codec, content-addressed store with atomic verified writes."""

    def __init__(
        self,
        root: Path,
        *,
        compression: Compression = Compression.ZSTD,
        max_raw_bytes: int = DEFAULT_MAX_BLOB_RAW_BYTES,
    ) -> None:
        self.root = Path(root).resolve()
        self.compression = Compression(compression)
        self.max_raw_bytes = _nonnegative_int("max_raw_bytes", max_raw_bytes)
        if self.max_raw_bytes == 0:
            raise ValueError("max_raw_bytes must be positive")
        # Fail at construction rather than silently selecting another codec.
        self._zstandard = _import_zstandard() if self.compression is Compression.ZSTD else None
        self.root.mkdir(parents=True, exist_ok=True)
        self.partial_directory = self.root / ".partial"
        self.quarantine_directory = self.root / ".quarantine"
        self.partial_directory.mkdir(exist_ok=True)

    @property
    def suffix(self) -> str:
        return ".jsonl.zst" if self.compression is Compression.ZSTD else ".jsonl.gz"

    def _target(self, digest: str) -> Path:
        _normalise_hash("content_hash", digest)
        return self.root / digest[:2] / (digest + self.suffix)

    def _compress(self, payload: bytes) -> bytes:
        if self.compression is Compression.GZIP:
            return gzip.compress(payload, compresslevel=9, mtime=0)
        assert self._zstandard is not None
        return self._zstandard.ZstdCompressor(
            level=10, write_checksum=True, write_content_size=True
        ).compress(payload)

    def _decompress(self, payload: bytes) -> bytes:
        if self.compression is Compression.GZIP:
            return gzip.decompress(payload)
        assert self._zstandard is not None
        return self._zstandard.ZstdDecompressor().decompress(payload)

    def _safe_record_path(self, record: ArtifactRecord) -> Path:
        relative = PurePosixPath(record.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactIntegrityError("artifact path escapes the blob root")
        candidate = (self.root / Path(*relative.parts)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactIntegrityError("artifact path escapes the blob root") from exc
        return candidate

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> ArtifactRecord:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if len(payload) > self.max_raw_bytes:
            raise BlobTooLargeError(
                f"payload is {len(payload)} bytes; bound is {self.max_raw_bytes} bytes"
            )
        if not media_type or not isinstance(media_type, str):
            raise ValueError("media_type must be a nonempty string")
        release = ReleaseClass(release_class)
        digest = sha256_bytes(payload)
        target = self._target(digest)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not target.exists():
            temporary = self.partial_directory / f"{digest}.{uuid.uuid4().hex}.partial"
            compressed = self._compress(payload)
            try:
                with temporary.open("xb") as stream:
                    stream.write(compressed)
                    stream.flush()
                    os.fsync(stream.fileno())
                verified = self._decompress(temporary.read_bytes())
                if verified != payload or sha256_bytes(verified) != digest:
                    raise ArtifactIntegrityError("temporary blob failed round-trip verification")
                os.replace(str(temporary), str(target))
                self._fsync_directory(target.parent)
            except Exception:
                # Deliberately retain a bounded partial for explicit quarantine/recovery.
                raise

        stored = target.read_bytes()
        try:
            decoded = self._decompress(stored)
        except Exception as exc:
            raise ArtifactIntegrityError(f"cannot decode stored blob {digest}") from exc
        if decoded != payload or sha256_bytes(decoded) != digest:
            raise ArtifactIntegrityError(f"stored blob {digest} does not match its address")
        return ArtifactRecord(
            content_hash=digest,
            compression=self.compression,
            media_type=media_type,
            raw_size_bytes=len(payload),
            stored_size_bytes=len(stored),
            relative_path=target.relative_to(self.root).as_posix(),
            release_class=release,
            created_at=_normalise_timestamp(created_at),
        )

    def put_jsonl(
        self,
        records: Iterable[Any],
        *,
        media_type: str = "application/x-ndjson",
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> ArtifactRecord:
        payload = bytearray()
        for record in records:
            line = canonical_json(record).encode("utf-8") + b"\n"
            if len(payload) + len(line) > self.max_raw_bytes:
                raise BlobTooLargeError("canonical JSONL payload exceeds the one-blob bound")
            payload.extend(line)
        return self.put_bytes(
            bytes(payload),
            media_type=media_type,
            release_class=release_class,
            created_at=created_at,
        )

    def read_bytes(self, record: ArtifactRecord, *, allow_restricted: bool = False) -> bytes:
        if record.compression is not self.compression:
            raise ArtifactIntegrityError(
                f"record codec {record.compression.value} does not match store codec "
                f"{self.compression.value}"
            )
        if record.release_class is ReleaseClass.RESTRICTED and not allow_restricted:
            raise ReleaseViolationError("restricted artifact requires explicit restricted access")
        path = self._safe_record_path(record)
        try:
            encoded = path.read_bytes()
            decoded = self._decompress(encoded)
        except (OSError, EOFError, ValueError) as exc:
            raise ArtifactIntegrityError(f"cannot read artifact {record.content_hash}") from exc
        if len(decoded) != record.raw_size_bytes:
            raise ArtifactIntegrityError("artifact raw size does not match ledger metadata")
        if len(encoded) != record.stored_size_bytes:
            raise ArtifactIntegrityError("artifact stored size does not match ledger metadata")
        if sha256_bytes(decoded) != record.content_hash:
            raise ArtifactIntegrityError("artifact content hash verification failed")
        return decoded

    def quarantine_partials(self) -> tuple[Path, ...]:
        """Atomically move interrupted writes out of the active partial directory."""

        self.quarantine_directory.mkdir(exist_ok=True)
        moved = []
        for partial in sorted(self.partial_directory.glob("*.partial")):
            if not partial.is_file() or partial.is_symlink():
                continue
            destination = self.quarantine_directory / (
                partial.name + "." + uuid.uuid4().hex + ".quarantined"
            )
            os.replace(str(partial), str(destination))
            moved.append(destination)
        if moved:
            self._fsync_directory(self.quarantine_directory)
        return tuple(moved)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        try:
            descriptor = os.open(str(directory), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)


class Ledger:
    """SQLite metadata ledger whose scientific rows are append-only."""

    _APPEND_ONLY_TABLES: ClassVar[tuple[str, ...]] = (
        "schema_metadata",
        "studies",
        "jobs",
        "study_jobs",
        "job_transitions",
        "attempts",
        "failures",
        "artifacts",
        "job_artifacts",
        "inputs",
        "evidence_snapshots",
        "prequery_barriers",
        "query_access_events",
        "packet_materialization_events",
        "gpu_allocation_journal",
        "gpu_service_journal",
        "gpu_events",
        "gpu_service_sessions",
        "model_calls",
        "validations",
        "projections",
        "feedback",
        "metrics",
        "visualizations",
        "resource_samples",
        "storage_samples",
    )

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=30, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                yield cursor
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")
            finally:
                cursor.close()

    def _create_schema(self) -> None:
        release_values = "'public','restricted'"
        state_values = ",".join(f"'{state.value}'" for state in JobState)
        attempt_values = ",".join(f"'{kind.value}'" for kind in AttemptKind)
        failure_values = ",".join(f"'{kind.value}'" for kind in FailureKind)
        gpu_values = ",".join(f"'{kind.value}'" for kind in GpuEventKind)
        input_values = ",".join(f"'{kind.value}'" for kind in InputKind)
        backend_values = ",".join(f"'{kind.value}'" for kind in ModelBackend)
        call_role_values = ",".join(f"'{kind.value}'" for kind in ModelCallRole)
        retry_values = ",".join(f"'{kind.value}'" for kind in RetryClass)
        validation_values = ",".join(f"'{kind.value}'" for kind in ValidationStatus)
        evidence_support_values = ",".join(f"'{kind.value}'" for kind in EvidenceSupportStatus)
        temporal_validation_values = ",".join(
            f"'{kind.value}'" for kind in TemporalValidationStatus
        )
        commitment_values = ",".join(f"'{kind.value}'" for kind in CommitmentCheckStatus)
        assessment_scope_values = ",".join(
            f"'{kind.value}'" for kind in SemanticAssessmentScope
        )
        feedback_kind_values = ",".join(f"'{kind.value}'" for kind in FeedbackKind)
        feedback_action_values = ",".join(f"'{kind.value}'" for kind in FeedbackAction)
        feedback_status_values = ",".join(f"'{kind.value}'" for kind in FeedbackResolutionStatus)
        metric_status_values = ",".join(f"'{kind.value}'" for kind in MetricStatus)
        compression_values = "'gzip','zstd'"
        schema = f"""
        CREATE TABLE IF NOT EXISTS schema_metadata (
            schema_version INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            identity_hash TEXT NOT NULL UNIQUE,
            identity_json TEXT NOT NULL,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS job_transitions (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            sequence INTEGER NOT NULL CHECK (sequence >= 0),
            from_state TEXT CHECK (from_state IS NULL OR from_state IN ({state_values})),
            to_state TEXT NOT NULL CHECK (to_state IN ({state_values})),
            occurred_at TEXT NOT NULL,
            UNIQUE(job_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            attempt_kind TEXT NOT NULL CHECK (attempt_kind IN ({attempt_values})),
            parent_attempt_id TEXT REFERENCES attempts(attempt_id),
            input_hash TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            seed INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (
                (attempt_kind = 'base' AND parent_attempt_id IS NULL) OR
                (attempt_kind != 'base' AND parent_attempt_id IS NOT NULL)
            )
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_repair_per_parent
            ON attempts(parent_attempt_id) WHERE attempt_kind = 'repair';
        CREATE TABLE IF NOT EXISTS failures (
            failure_id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
            failure_kind TEXT NOT NULL CHECK (failure_kind IN ({failure_values})),
            message TEXT NOT NULL,
            details_json TEXT NOT NULL,
            artifact_hash TEXT,
            occurred_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            content_hash TEXT PRIMARY KEY,
            compression TEXT NOT NULL CHECK (compression IN ({compression_values})),
            media_type TEXT NOT NULL,
            raw_size_bytes INTEGER NOT NULL CHECK (raw_size_bytes >= 0),
            stored_size_bytes INTEGER NOT NULL CHECK (stored_size_bytes >= 0),
            relative_path TEXT NOT NULL UNIQUE,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS job_artifacts (
            link_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            content_hash TEXT NOT NULL REFERENCES artifacts(content_hash),
            role TEXT NOT NULL,
            attempt_id TEXT REFERENCES attempts(attempt_id),
            parent_content_hash TEXT REFERENCES artifacts(content_hash),
            created_at TEXT NOT NULL,
            UNIQUE(job_id, content_hash, role, attempt_id)
        );
        CREATE TABLE IF NOT EXISTS gpu_events (
            event_id TEXT PRIMARY KEY,
            event_kind TEXT NOT NULL CHECK (event_kind IN ({gpu_values})),
            allocated_microseconds INTEGER NOT NULL CHECK (allocated_microseconds >= 0),
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            succeeded INTEGER CHECK (succeeded IS NULL OR succeeded IN (0, 1)),
            job_id TEXT REFERENCES jobs(job_id),
            attempt_id TEXT REFERENCES attempts(attempt_id),
            details_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS gpu_allocation_journal (
            journal_id TEXT PRIMARY KEY,
            allocation_id TEXT NOT NULL,
            sequence INTEGER NOT NULL CHECK (sequence >= 0),
            state TEXT NOT NULL CHECK (
                state IN ('opened','heartbeat','closed','recovered')
            ),
            intended_event_kind TEXT NOT NULL CHECK (intended_event_kind IN ({gpu_values})),
            elapsed_microseconds INTEGER NOT NULL CHECK (elapsed_microseconds >= 0),
            maximum_microseconds INTEGER NOT NULL CHECK (maximum_microseconds > 0),
            observed_at TEXT NOT NULL,
            job_id TEXT REFERENCES jobs(job_id),
            attempt_id TEXT REFERENCES attempts(attempt_id),
            details_json TEXT NOT NULL,
            UNIQUE(allocation_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS gpu_service_journal (
            journal_id TEXT PRIMARY KEY,
            service_session_id TEXT NOT NULL,
            sequence INTEGER NOT NULL CHECK (sequence >= 0),
            state TEXT NOT NULL CHECK (
                state IN ('opened','heartbeat','process_stopped','closed','recovered')
            ),
            session_id TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            service_started_at TEXT NOT NULL,
            elapsed_microseconds INTEGER NOT NULL CHECK (elapsed_microseconds >= 0),
            ledger_allocated_microseconds_before_session INTEGER NOT NULL
                CHECK (ledger_allocated_microseconds_before_session >= 0),
            hard_limit_microseconds INTEGER NOT NULL CHECK (hard_limit_microseconds > 0),
            observed_at TEXT NOT NULL,
            details_json TEXT NOT NULL,
            UNIQUE(service_session_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS gpu_service_sessions (
            service_session_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            service_microseconds INTEGER NOT NULL CHECK (service_microseconds >= 0),
            classified_event_microseconds INTEGER NOT NULL
                CHECK (classified_event_microseconds >= 0),
            overhead_microseconds INTEGER NOT NULL CHECK (overhead_microseconds >= 0),
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            details_json TEXT NOT NULL,
            CHECK (service_microseconds >= classified_event_microseconds),
            CHECK (
                overhead_microseconds = service_microseconds - classified_event_microseconds
            )
        );
        CREATE TABLE IF NOT EXISTS storage_samples (
            sample_id TEXT PRIMARY KEY,
            phase TEXT NOT NULL,
            sampled_at TEXT NOT NULL,
            current_occupied_bytes INTEGER NOT NULL,
            additional_reserved_bytes INTEGER NOT NULL,
            projected_occupied_bytes INTEGER NOT NULL,
            filesystem_free_bytes INTEGER NOT NULL,
            effective_projected_headroom_bytes INTEGER NOT NULL,
            allowed INTEGER NOT NULL CHECK (allowed IN (0, 1)),
            violations_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS studies (
            study_id TEXT PRIMARY KEY,
            protocol_hash TEXT NOT NULL,
            code_manifest_hash TEXT NOT NULL,
            configuration_hash TEXT NOT NULL,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS study_jobs (
            link_id TEXT PRIMARY KEY,
            study_id TEXT NOT NULL REFERENCES studies(study_id),
            job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
            created_at TEXT NOT NULL,
            UNIQUE(study_id, job_id)
        );
        CREATE TABLE IF NOT EXISTS inputs (
            input_id TEXT PRIMARY KEY,
            study_id TEXT NOT NULL REFERENCES studies(study_id),
            input_kind TEXT NOT NULL CHECK (input_kind IN ({input_values})),
            content_hash TEXT NOT NULL,
            artifact_hash TEXT REFERENCES artifacts(content_hash),
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evidence_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            input_id TEXT NOT NULL UNIQUE REFERENCES inputs(input_id),
            horizon_hash TEXT NOT NULL,
            evidence_manifest_hash TEXT NOT NULL,
            index_configuration_hash TEXT NOT NULL,
            prequery_seal_hash TEXT NOT NULL,
            eligible_evidence_count INTEGER NOT NULL CHECK (eligible_evidence_count >= 0),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prequery_barriers (
            barrier_hash TEXT PRIMARY KEY,
            barrier_id TEXT NOT NULL UNIQUE,
            execution_id TEXT NOT NULL,
            execution_manifest_hash TEXT NOT NULL,
            barrier_artifact_hash TEXT NOT NULL UNIQUE REFERENCES artifacts(content_hash),
            preparation_count INTEGER NOT NULL CHECK (preparation_count >= 1),
            sealed_at TEXT NOT NULL,
            persisted_at TEXT NOT NULL,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            CHECK (sealed_at <= persisted_at)
        );
        CREATE TABLE IF NOT EXISTS query_access_events (
            access_event_hash TEXT PRIMARY KEY,
            access_event_id TEXT NOT NULL UNIQUE,
            execution_id TEXT NOT NULL,
            query_context_hash TEXT NOT NULL,
            model_visible_query_hash TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL,
            stage_manifest_hash TEXT NOT NULL,
            query_artifact_hash TEXT NOT NULL,
            prequery_barrier_hash TEXT NOT NULL REFERENCES prequery_barriers(barrier_hash),
            packet_hash TEXT CHECK (packet_hash IS NULL),
            query_payload_artifact_hash TEXT NOT NULL REFERENCES artifacts(content_hash),
            access_event_artifact_hash TEXT NOT NULL UNIQUE REFERENCES artifacts(content_hash),
            registered_revealed_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            UNIQUE (execution_id, query_artifact_hash),
            UNIQUE (execution_id, stage_manifest_hash),
            CHECK (registered_revealed_at <= accessed_at)
        );
        CREATE TABLE IF NOT EXISTS packet_materialization_events (
            materialization_event_hash TEXT PRIMARY KEY,
            materialization_event_id TEXT NOT NULL UNIQUE,
            execution_id TEXT NOT NULL,
            query_access_event_hash TEXT NOT NULL UNIQUE
                REFERENCES query_access_events(access_event_hash),
            snapshot_hash TEXT NOT NULL,
            packet_hash TEXT NOT NULL,
            retrieval_method TEXT NOT NULL CHECK (
                retrieval_method IN ('all_admissible','sqlite_fts5_bm25')
            ),
            retrieval_config_hash TEXT NOT NULL,
            packet_artifact_hash TEXT NOT NULL REFERENCES artifacts(content_hash),
            materialization_event_artifact_hash TEXT NOT NULL UNIQUE
                REFERENCES artifacts(content_hash),
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            CHECK (started_at <= completed_at)
        );
        CREATE TABLE IF NOT EXISTS model_calls (
            model_call_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(attempt_id),
            gpu_event_id TEXT UNIQUE REFERENCES gpu_events(event_id),
            backend TEXT NOT NULL CHECK (backend IN ({backend_values})),
            call_role TEXT NOT NULL CHECK (call_role IN ({call_role_values})),
            retry_class TEXT NOT NULL CHECK (retry_class IN ({retry_values})),
            model_manifest_hash TEXT NOT NULL,
            decoding_manifest_hash TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            response_artifact_hash TEXT REFERENCES artifacts(content_hash),
            construction_unit_hash TEXT NOT NULL,
            served_context_count INTEGER NOT NULL CHECK (served_context_count >= 1),
            prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens >= 0),
            completion_tokens INTEGER NOT NULL CHECK (completion_tokens >= 0),
            allocated_gpu_microseconds INTEGER NOT NULL
                CHECK (allocated_gpu_microseconds >= 0),
            successful INTEGER NOT NULL CHECK (successful IN (0, 1)),
            created_at TEXT NOT NULL,
            CHECK (
                (backend = 'vllm_gpu' AND gpu_event_id IS NOT NULL) OR
                (backend = 'hand_authored_fixture' AND gpu_event_id IS NULL
                    AND allocated_gpu_microseconds = 0)
            )
        );
        CREATE TABLE IF NOT EXISTS validations (
            validation_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
            input_artifact_hash TEXT NOT NULL REFERENCES artifacts(content_hash),
            validator_manifest_hash TEXT NOT NULL,
            validation_status TEXT NOT NULL CHECK (validation_status IN ({validation_values})),
            evidence_support_status TEXT NOT NULL
                CHECK (evidence_support_status IN ({evidence_support_values})),
            temporal_status TEXT NOT NULL
                CHECK (temporal_status IN ({temporal_validation_values})),
            commitment_status TEXT NOT NULL
                CHECK (commitment_status IN ({commitment_values})),
            semantic_assessment_scope TEXT NOT NULL
                CHECK (semantic_assessment_scope IN ({assessment_scope_values})),
            diagnostics_artifact_hash TEXT REFERENCES artifacts(content_hash),
            parent_validation_id TEXT REFERENCES validations(validation_id),
            repair_attempt_id TEXT REFERENCES attempts(attempt_id),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projections (
            projection_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            validation_id TEXT NOT NULL UNIQUE REFERENCES validations(validation_id),
            snapshot_id TEXT NOT NULL REFERENCES evidence_snapshots(snapshot_id),
            packet_input_id TEXT NOT NULL REFERENCES inputs(input_id),
            condition_id TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            upper_ontology_hash TEXT NOT NULL,
            construction_certificate_hash TEXT NOT NULL,
            projection_artifact_hash TEXT NOT NULL REFERENCES artifacts(content_hash),
            projection_semantic_hash TEXT NOT NULL,
            parent_projection_id TEXT REFERENCES projections(projection_id),
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            finalized_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id TEXT PRIMARY KEY,
            study_id TEXT NOT NULL REFERENCES studies(study_id),
            job_id TEXT REFERENCES jobs(job_id),
            feedback_kind TEXT NOT NULL CHECK (feedback_kind IN ({feedback_kind_values})),
            action TEXT NOT NULL CHECK (action IN ({feedback_action_values})),
            revision_hash TEXT NOT NULL,
            anchor_manifest_hash TEXT NOT NULL,
            before_context_hash TEXT NOT NULL,
            after_context_hash TEXT NOT NULL,
            receiving_condition TEXT,
            before_projection_id TEXT REFERENCES projections(projection_id),
            after_projection_id TEXT REFERENCES projections(projection_id),
            resolution_status TEXT NOT NULL
                CHECK (resolution_status IN ({feedback_status_values})),
            resolution_artifact_hash TEXT REFERENCES artifacts(content_hash),
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            created_at TEXT NOT NULL,
            CHECK (
                (feedback_kind = 'user_revision' AND job_id IS NULL
                    AND receiving_condition IS NULL AND before_projection_id IS NULL
                    AND after_projection_id IS NULL AND resolution_status = 'pending') OR
                (feedback_kind = 'condition_resolution' AND job_id IS NOT NULL
                    AND receiving_condition IS NOT NULL AND before_projection_id IS NOT NULL
                    AND resolution_status != 'pending')
            )
        );
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id TEXT PRIMARY KEY,
            study_id TEXT NOT NULL REFERENCES studies(study_id),
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            projection_id TEXT REFERENCES projections(projection_id),
            unit_hash TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_version_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ({metric_status_values})),
            value REAL,
            numerator REAL,
            denominator REAL,
            result_artifact_hash TEXT REFERENCES artifacts(content_hash),
            created_at TEXT NOT NULL,
            CHECK ((status = 'value' AND value IS NOT NULL) OR
                   (status != 'value' AND value IS NULL))
        );
        CREATE TABLE IF NOT EXISTS visualizations (
            visualization_id TEXT PRIMARY KEY,
            projection_id TEXT NOT NULL REFERENCES projections(projection_id),
            renderer_configuration_hash TEXT NOT NULL,
            semantic_hash TEXT NOT NULL,
            visualization_artifact_hash TEXT NOT NULL REFERENCES artifacts(content_hash),
            layout_seed INTEGER NOT NULL,
            release_class TEXT NOT NULL CHECK (release_class IN ({release_values})),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS resource_samples (
            sample_id TEXT PRIMARY KEY,
            job_id TEXT REFERENCES jobs(job_id),
            gpu_event_id TEXT REFERENCES gpu_events(event_id),
            process_ram_bytes INTEGER NOT NULL CHECK (process_ram_bytes >= 0),
            system_available_ram_bytes INTEGER NOT NULL
                CHECK (system_available_ram_bytes >= 0),
            gpu_vram_bytes INTEGER NOT NULL CHECK (gpu_vram_bytes >= 0),
            project_storage_bytes INTEGER NOT NULL CHECK (project_storage_bytes >= 0),
            cpu_worker_count INTEGER NOT NULL CHECK (cpu_worker_count >= 0),
            sampled_at TEXT NOT NULL
        );
        """
        lineage_triggers = """
        CREATE TRIGGER IF NOT EXISTS query_access_require_barrier_lineage
        BEFORE INSERT ON query_access_events
        WHEN NOT EXISTS (
            SELECT 1 FROM prequery_barriers
            WHERE barrier_hash = NEW.prequery_barrier_hash
              AND execution_id = NEW.execution_id
              AND sealed_at < NEW.accessed_at
              AND persisted_at < NEW.accessed_at
        )
        BEGIN
            SELECT RAISE(ABORT, 'query access has invalid prequery lineage');
        END;
        CREATE TRIGGER IF NOT EXISTS packet_materialization_require_access_lineage
        BEFORE INSERT ON packet_materialization_events
        WHEN NOT EXISTS (
            SELECT 1 FROM query_access_events
            WHERE access_event_hash = NEW.query_access_event_hash
              AND execution_id = NEW.execution_id
              AND snapshot_hash = NEW.snapshot_hash
              AND accessed_at <= NEW.started_at
        )
        BEGIN
            SELECT RAISE(ABORT, 'packet materialization has invalid query lineage');
        END;
        """
        append_only_triggers = "".join(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_reject_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'append-only table: {table}');
            END;
            CREATE TRIGGER IF NOT EXISTS {table}_reject_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'append-only table: {table}');
            END;
            """
            for table in self._APPEND_ONLY_TABLES
        )
        existing_validation_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(validations)").fetchall()
        }
        validation_scope_migration = (
            ""
            if not existing_validation_columns
            or "semantic_assessment_scope" in existing_validation_columns
            else (
                "ALTER TABLE validations ADD COLUMN semantic_assessment_scope TEXT "
                "NOT NULL DEFAULT 'legacy_unspecified' CHECK (semantic_assessment_scope IN "
                f"({assessment_scope_values}));"
            )
        )
        existing_projection_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(projections)").fetchall()
        }
        projection_semantic_migration = (
            ""
            if not existing_projection_columns
            or "projection_semantic_hash" in existing_projection_columns
            else "ALTER TABLE projections ADD COLUMN projection_semantic_hash TEXT;"
        )
        existing_metric_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(metrics)").fetchall()
        }
        metric_job_migration = (
            ""
            if not existing_metric_columns or "job_id" in existing_metric_columns
            else "ALTER TABLE metrics ADD COLUMN job_id TEXT REFERENCES jobs(job_id);"
        )
        migration = f"""
        BEGIN IMMEDIATE;
        {schema}
        {validation_scope_migration}
        {projection_semantic_migration}
        {metric_job_migration}
        {lineage_triggers}
        {append_only_triggers}
        INSERT OR IGNORE INTO schema_metadata(schema_version, created_at)
            VALUES ({SCHEMA_VERSION}, '{_normalise_timestamp()}');
        COMMIT;
        """
        with self._lock:
            try:
                self._connection.executescript(migration)
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def _append_metadata_record(
        self,
        *,
        table: str,
        key_column: str,
        stable_fields: Mapping[str, Any],
        timestamp_column: str,
        timestamp: Any | None,
    ) -> sqlite3.Row:
        """Insert immutable metadata, or return an identical row on replay."""

        if table not in self._APPEND_ONLY_TABLES or table == "schema_metadata":
            raise ValueError("unsupported append-only metadata table")
        if key_column not in stable_fields or timestamp_column in stable_fields:
            raise ValueError("invalid immutable-record field declaration")
        for identifier in (key_column, timestamp_column, *stable_fields):
            if not identifier.replace("_", "").isalnum():
                raise ValueError("invalid SQL metadata identifier")

        key = stable_fields[key_column]
        columns = (*stable_fields, timestamp_column)
        with self._transaction() as cursor:
            existing = cursor.execute(
                f"SELECT * FROM {table} WHERE {key_column} = ?", (key,)
            ).fetchone()
            if existing is not None:
                actual = tuple(existing[column] for column in stable_fields)
                expected = tuple(stable_fields.values())
                if actual != expected:
                    raise DuplicateConflictError(
                        f"{table}.{key_column} was reused with different immutable metadata"
                    )
                return existing
            values = (*stable_fields.values(), _normalise_timestamp(timestamp))
            placeholders = ",".join("?" for _ in columns)
            cursor.execute(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )
            inserted = cursor.execute(
                f"SELECT * FROM {table} WHERE {key_column} = ?", (key,)
            ).fetchone()
            assert inserted is not None
            return inserted

    def _insert_immutable_row(
        self,
        cursor: sqlite3.Cursor,
        *,
        table: str,
        key_column: str,
        fields: Mapping[str, Any],
    ) -> sqlite3.Row:
        """Insert one fully timestamped immutable row inside a caller transaction."""

        if table not in self._APPEND_ONLY_TABLES or table == "schema_metadata":
            raise ValueError("unsupported append-only metadata table")
        if key_column not in fields:
            raise ValueError("immutable row omits its primary identity")
        for identifier in (key_column, *fields):
            if not identifier.replace("_", "").isalnum():
                raise ValueError("invalid SQL metadata identifier")
        key = fields[key_column]
        existing = cursor.execute(
            f"SELECT * FROM {table} WHERE {key_column} = ?", (key,)
        ).fetchone()
        if existing is not None:
            if tuple(existing[name] for name in fields) != tuple(fields.values()):
                raise DuplicateConflictError(
                    f"{table}.{key_column} was reused with different immutable metadata"
                )
            return existing
        columns = tuple(fields)
        try:
            cursor.execute(
                f"INSERT INTO {table} ({','.join(columns)}) "
                f"VALUES ({','.join('?' for _ in columns)})",
                tuple(fields.values()),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateConflictError(
                f"{table} conflicts with an existing immutable event"
            ) from exc
        inserted = cursor.execute(
            f"SELECT * FROM {table} WHERE {key_column} = ?", (key,)
        ).fetchone()
        assert inserted is not None
        return inserted

    def schema_versions(self) -> tuple[int, ...]:
        rows = self._connection.execute(
            "SELECT schema_version FROM schema_metadata ORDER BY schema_version"
        ).fetchall()
        return tuple(int(row["schema_version"]) for row in rows)

    def _ensure_public_compatible_artifacts(
        self, release_class: ReleaseClass, *artifact_hashes: str | None
    ) -> None:
        if release_class is not ReleaseClass.PUBLIC:
            return
        for artifact_hash in artifact_hashes:
            if artifact_hash is None:
                continue
            artifact = self.get_artifact(artifact_hash)
            if artifact.release_class is ReleaseClass.RESTRICTED:
                raise ReleaseViolationError(
                    "public metadata cannot reference a restricted artifact"
                )

    def register_study(
        self,
        *,
        study_id: str,
        protocol_hash: str,
        code_manifest_hash: str,
        configuration_hash: str,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> StudyRecord:
        _metadata_token("study_id", study_id)
        for name, value in (
            ("protocol_hash", protocol_hash),
            ("code_manifest_hash", code_manifest_hash),
            ("configuration_hash", configuration_hash),
        ):
            _normalise_hash(name, value)
        release = ReleaseClass(release_class)
        row = self._append_metadata_record(
            table="studies",
            key_column="study_id",
            stable_fields={
                "study_id": study_id,
                "protocol_hash": protocol_hash,
                "code_manifest_hash": code_manifest_hash,
                "configuration_hash": configuration_hash,
                "release_class": release.value,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return self._study_from_row(row)

    @staticmethod
    def _study_from_row(row: sqlite3.Row) -> StudyRecord:
        return StudyRecord(
            study_id=row["study_id"],
            protocol_hash=row["protocol_hash"],
            code_manifest_hash=row["code_manifest_hash"],
            configuration_hash=row["configuration_hash"],
            release_class=ReleaseClass(row["release_class"]),
            created_at=row["created_at"],
        )

    def get_study(self, study_id: str) -> StudyRecord:
        row = self._connection.execute(
            "SELECT * FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown study {study_id}")
        return self._study_from_row(row)

    def link_job_to_study(
        self, *, study_id: str, job_id: str, created_at: Any | None = None
    ) -> StudyJobRecord:
        study = self.get_study(study_id)
        job = self.get_job(job_id)
        if (
            study.release_class is ReleaseClass.PUBLIC
            and job.release_class is ReleaseClass.RESTRICTED
        ):
            raise ReleaseViolationError("a public study cannot link a restricted job")
        link_id = sha256_bytes(
            canonical_json({"study_id": study_id, "job_id": job_id}).encode("utf-8")
        )
        row = self._append_metadata_record(
            table="study_jobs",
            key_column="link_id",
            stable_fields={"link_id": link_id, "study_id": study_id, "job_id": job_id},
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return StudyJobRecord(
            link_id=row["link_id"],
            study_id=row["study_id"],
            job_id=row["job_id"],
            created_at=row["created_at"],
        )

    def get_study_job(self, *, study_id: str, job_id: str) -> StudyJobRecord:
        """Resolve one exact study/job link without creating missing metadata."""

        self.get_study(study_id)
        self.get_job(job_id)
        link_id = sha256_bytes(
            canonical_json({"study_id": study_id, "job_id": job_id}).encode("utf-8")
        )
        row = self._connection.execute(
            "SELECT * FROM study_jobs WHERE link_id = ?", (link_id,)
        ).fetchone()
        if row is None or row["study_id"] != study_id or row["job_id"] != job_id:
            raise ArtifactIntegrityError("job is not linked to the required study")
        return StudyJobRecord(
            link_id=row["link_id"],
            study_id=row["study_id"],
            job_id=row["job_id"],
            created_at=row["created_at"],
        )

    def study_job_for_job(self, job_id: str) -> StudyJobRecord:
        """Resolve the one immutable owning study for a generation job."""

        self.get_job(job_id)
        rows = self._connection.execute(
            "SELECT * FROM study_jobs WHERE job_id = ? ORDER BY study_id",
            (job_id,),
        ).fetchall()
        if len(rows) != 1:
            raise ArtifactIntegrityError(
                "generation job does not resolve to exactly one owning study"
            )
        row = rows[0]
        return StudyJobRecord(
            link_id=row["link_id"],
            study_id=row["study_id"],
            job_id=row["job_id"],
            created_at=row["created_at"],
        )

    def register_input(
        self,
        *,
        input_id: str,
        study_id: str,
        input_kind: InputKind,
        content_hash: str,
        artifact_hash: str | None,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> InputRecord:
        _metadata_token("input_id", input_id)
        self.get_study(study_id)
        kind = InputKind(input_kind)
        _normalise_hash("content_hash", content_hash)
        if artifact_hash is not None:
            _normalise_hash("artifact_hash", artifact_hash)
            self.get_artifact(artifact_hash)
        release = ReleaseClass(release_class)
        self._ensure_public_compatible_artifacts(release, artifact_hash)
        row = self._append_metadata_record(
            table="inputs",
            key_column="input_id",
            stable_fields={
                "input_id": input_id,
                "study_id": study_id,
                "input_kind": kind.value,
                "content_hash": content_hash,
                "artifact_hash": artifact_hash,
                "release_class": release.value,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return self._input_from_row(row)

    @staticmethod
    def _input_from_row(row: sqlite3.Row) -> InputRecord:
        return InputRecord(
            input_id=row["input_id"],
            study_id=row["study_id"],
            input_kind=InputKind(row["input_kind"]),
            content_hash=row["content_hash"],
            artifact_hash=row["artifact_hash"],
            release_class=ReleaseClass(row["release_class"]),
            created_at=row["created_at"],
        )

    def get_input(self, input_id: str) -> InputRecord:
        row = self._connection.execute(
            "SELECT * FROM inputs WHERE input_id = ?", (input_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown input {input_id}")
        return self._input_from_row(row)

    def register_evidence_snapshot(
        self,
        *,
        snapshot_id: str,
        input_id: str,
        horizon_hash: str,
        evidence_manifest_hash: str,
        index_configuration_hash: str,
        prequery_seal_hash: str,
        eligible_evidence_count: int,
        created_at: Any | None = None,
    ) -> EvidenceSnapshotRecord:
        _metadata_token("snapshot_id", snapshot_id)
        input_record = self.get_input(input_id)
        if input_record.input_kind is not InputKind.EVIDENCE_SNAPSHOT:
            raise ValueError("an evidence snapshot must reference an evidence_snapshot input")
        for name, value in (
            ("horizon_hash", horizon_hash),
            ("evidence_manifest_hash", evidence_manifest_hash),
            ("index_configuration_hash", index_configuration_hash),
            ("prequery_seal_hash", prequery_seal_hash),
        ):
            _normalise_hash(name, value)
        count = _nonnegative_int("eligible_evidence_count", eligible_evidence_count)
        row = self._append_metadata_record(
            table="evidence_snapshots",
            key_column="snapshot_id",
            stable_fields={
                "snapshot_id": snapshot_id,
                "input_id": input_id,
                "horizon_hash": horizon_hash,
                "evidence_manifest_hash": evidence_manifest_hash,
                "index_configuration_hash": index_configuration_hash,
                "prequery_seal_hash": prequery_seal_hash,
                "eligible_evidence_count": count,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return EvidenceSnapshotRecord(
            snapshot_id=row["snapshot_id"],
            input_id=row["input_id"],
            horizon_hash=row["horizon_hash"],
            evidence_manifest_hash=row["evidence_manifest_hash"],
            index_configuration_hash=row["index_configuration_hash"],
            prequery_seal_hash=row["prequery_seal_hash"],
            eligible_evidence_count=row["eligible_evidence_count"],
            created_at=row["created_at"],
        )

    def get_evidence_snapshot(self, snapshot_id: str) -> EvidenceSnapshotRecord:
        """Resolve one immutable evidence-snapshot metadata row."""

        _metadata_token("snapshot_id", snapshot_id)
        row = self._connection.execute(
            "SELECT * FROM evidence_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evidence snapshot {snapshot_id}")
        return EvidenceSnapshotRecord(
            snapshot_id=row["snapshot_id"],
            input_id=row["input_id"],
            horizon_hash=row["horizon_hash"],
            evidence_manifest_hash=row["evidence_manifest_hash"],
            index_configuration_hash=row["index_configuration_hash"],
            prequery_seal_hash=row["prequery_seal_hash"],
            eligible_evidence_count=row["eligible_evidence_count"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _prequery_barrier_from_row(row: sqlite3.Row) -> PrequeryBarrierRecord:
        return PrequeryBarrierRecord(
            barrier_hash=row["barrier_hash"],
            barrier_id=row["barrier_id"],
            execution_id=row["execution_id"],
            execution_manifest_hash=row["execution_manifest_hash"],
            barrier_artifact_hash=row["barrier_artifact_hash"],
            preparation_count=row["preparation_count"],
            sealed_at=row["sealed_at"],
            persisted_at=row["persisted_at"],
            release_class=ReleaseClass(row["release_class"]),
        )

    def get_prequery_barrier(self, barrier_hash: str) -> PrequeryBarrierRecord:
        _normalise_hash("barrier_hash", barrier_hash)
        row = self._connection.execute(
            "SELECT * FROM prequery_barriers WHERE barrier_hash = ?", (barrier_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown prequery barrier {barrier_hash}")
        return self._prequery_barrier_from_row(row)

    def persist_prequery_barrier(
        self,
        record: PrequeryBarrierRecord,
        *,
        barrier_artifact: ArtifactRecord,
    ) -> PrequeryBarrierRecord:
        """Atomically register one barrier artifact and its append-only receipt."""

        for name, value in (
            ("barrier_hash", record.barrier_hash),
            ("execution_manifest_hash", record.execution_manifest_hash),
            ("barrier_artifact_hash", record.barrier_artifact_hash),
        ):
            _normalise_hash(name, value)
        for name, value in (
            ("barrier_id", record.barrier_id),
            ("execution_id", record.execution_id),
        ):
            _metadata_token(name, value)
        if record.preparation_count < 1:
            raise ValueError("prequery barrier must contain at least one preparation")
        sealed_at = _normalise_timestamp(record.sealed_at)
        persisted_at = _normalise_timestamp(record.persisted_at)
        if _parse_timestamp(sealed_at) > _parse_timestamp(persisted_at):
            raise ValueError("prequery barrier cannot be persisted before it is sealed")
        release = ReleaseClass(record.release_class)
        if record.barrier_artifact_hash != barrier_artifact.content_hash:
            raise ValueError("prequery barrier receipt cites a different CAS artifact")
        if barrier_artifact.release_class is not release:
            raise ReleaseViolationError(
                "prequery barrier and its CAS artifact require the same release class"
            )
        fields = {
            "barrier_hash": record.barrier_hash,
            "barrier_id": record.barrier_id,
            "execution_id": record.execution_id,
            "execution_manifest_hash": record.execution_manifest_hash,
            "barrier_artifact_hash": record.barrier_artifact_hash,
            "preparation_count": record.preparation_count,
            "sealed_at": sealed_at,
            "persisted_at": persisted_at,
            "release_class": release.value,
        }
        with self._transaction() as cursor:
            self._register_artifact_with_cursor(cursor, barrier_artifact)
            row = self._insert_immutable_row(
                cursor,
                table="prequery_barriers",
                key_column="barrier_hash",
                fields=fields,
            )
        return self._prequery_barrier_from_row(row)

    @staticmethod
    def _query_access_from_row(row: sqlite3.Row) -> QueryAccessRecord:
        return QueryAccessRecord(
            access_event_hash=row["access_event_hash"],
            access_event_id=row["access_event_id"],
            execution_id=row["execution_id"],
            query_context_hash=row["query_context_hash"],
            model_visible_query_hash=row["model_visible_query_hash"],
            snapshot_hash=row["snapshot_hash"],
            stage_manifest_hash=row["stage_manifest_hash"],
            query_artifact_hash=row["query_artifact_hash"],
            prequery_barrier_hash=row["prequery_barrier_hash"],
            packet_hash=row["packet_hash"],
            query_payload_artifact_hash=row["query_payload_artifact_hash"],
            access_event_artifact_hash=row["access_event_artifact_hash"],
            registered_revealed_at=row["registered_revealed_at"],
            accessed_at=row["accessed_at"],
            release_class=ReleaseClass(row["release_class"]),
        )

    def get_query_access(self, access_event_hash: str) -> QueryAccessRecord:
        _normalise_hash("access_event_hash", access_event_hash)
        row = self._connection.execute(
            "SELECT * FROM query_access_events WHERE access_event_hash = ?",
            (access_event_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown query access event {access_event_hash}")
        return self._query_access_from_row(row)

    def get_query_access_by_event_id(self, access_event_id: str) -> QueryAccessRecord:
        """Resolve exactly one audited access by its controller-issued identity.

        This narrow lookup lets an interrupted query-opening controller detect a
        durable commit without enumerating query records or rereading query.json.
        """

        _metadata_token("access_event_id", access_event_id)
        rows = self._connection.execute(
            "SELECT * FROM query_access_events WHERE access_event_id = ?",
            (access_event_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f"unknown query access event {access_event_id}")
        if len(rows) != 1:
            raise ArtifactIntegrityError("query access event identity is not unique")
        return self._query_access_from_row(rows[0])

    def persist_query_access(
        self,
        record: QueryAccessRecord,
        *,
        query_payload_artifact: ArtifactRecord,
        access_event_artifact: ArtifactRecord,
    ) -> QueryAccessRecord:
        """Commit query payload, access receipt, and lineage in one SQLite transaction."""

        for name, value in (
            ("access_event_hash", record.access_event_hash),
            ("query_context_hash", record.query_context_hash),
            ("model_visible_query_hash", record.model_visible_query_hash),
            ("snapshot_hash", record.snapshot_hash),
            ("stage_manifest_hash", record.stage_manifest_hash),
            ("query_artifact_hash", record.query_artifact_hash),
            ("prequery_barrier_hash", record.prequery_barrier_hash),
            ("query_payload_artifact_hash", record.query_payload_artifact_hash),
            ("access_event_artifact_hash", record.access_event_artifact_hash),
        ):
            _normalise_hash(name, value)
        if record.packet_hash is not None:
            raise ValueError("query access cannot carry a packet hash before materialization")
        for name, value in (
            ("access_event_id", record.access_event_id),
            ("execution_id", record.execution_id),
        ):
            _metadata_token(name, value)
        registered = _normalise_timestamp(record.registered_revealed_at)
        accessed = _normalise_timestamp(record.accessed_at)
        if _parse_timestamp(accessed) < _parse_timestamp(registered):
            raise ValueError("query access cannot predate its registered reveal")
        release = ReleaseClass(record.release_class)
        expected_artifacts = {
            record.query_payload_artifact_hash: query_payload_artifact,
            record.access_event_artifact_hash: access_event_artifact,
        }
        if len(expected_artifacts) != 2 or any(
            digest != artifact.content_hash
            for digest, artifact in expected_artifacts.items()
        ):
            raise ValueError("query access CAS artifact bindings are inconsistent")
        if any(artifact.release_class is not release for artifact in expected_artifacts.values()):
            raise ReleaseViolationError(
                "query access and both CAS artifacts require the same release class"
            )
        fields = {
            "access_event_hash": record.access_event_hash,
            "access_event_id": record.access_event_id,
            "execution_id": record.execution_id,
            "query_context_hash": record.query_context_hash,
            "model_visible_query_hash": record.model_visible_query_hash,
            "snapshot_hash": record.snapshot_hash,
            "stage_manifest_hash": record.stage_manifest_hash,
            "query_artifact_hash": record.query_artifact_hash,
            "prequery_barrier_hash": record.prequery_barrier_hash,
            "packet_hash": None,
            "query_payload_artifact_hash": record.query_payload_artifact_hash,
            "access_event_artifact_hash": record.access_event_artifact_hash,
            "registered_revealed_at": registered,
            "accessed_at": accessed,
            "release_class": release.value,
        }
        with self._transaction() as cursor:
            barrier = cursor.execute(
                "SELECT * FROM prequery_barriers WHERE barrier_hash = ?",
                (record.prequery_barrier_hash,),
            ).fetchone()
            if barrier is None:
                raise ValueError("query access requires a persisted prequery barrier")
            if (
                barrier["execution_id"] != record.execution_id
                or _parse_timestamp(barrier["sealed_at"]) >= _parse_timestamp(accessed)
                or _parse_timestamp(barrier["persisted_at"]) >= _parse_timestamp(accessed)
            ):
                raise ValueError("query access does not strictly follow its prequery barrier")
            if barrier["release_class"] != release.value:
                raise ReleaseViolationError(
                    "query access and prequery barrier require the same release class"
                )
            for artifact in expected_artifacts.values():
                self._register_artifact_with_cursor(cursor, artifact)
            row = self._insert_immutable_row(
                cursor,
                table="query_access_events",
                key_column="access_event_hash",
                fields=fields,
            )
        return self._query_access_from_row(row)

    @staticmethod
    def _packet_materialization_from_row(
        row: sqlite3.Row,
    ) -> PacketMaterializationRecord:
        return PacketMaterializationRecord(
            materialization_event_hash=row["materialization_event_hash"],
            materialization_event_id=row["materialization_event_id"],
            execution_id=row["execution_id"],
            query_access_event_hash=row["query_access_event_hash"],
            snapshot_hash=row["snapshot_hash"],
            packet_hash=row["packet_hash"],
            retrieval_method=row["retrieval_method"],
            retrieval_config_hash=row["retrieval_config_hash"],
            packet_artifact_hash=row["packet_artifact_hash"],
            materialization_event_artifact_hash=row[
                "materialization_event_artifact_hash"
            ],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            release_class=ReleaseClass(row["release_class"]),
        )

    def get_packet_materialization(
        self,
        materialization_event_hash: str,
    ) -> PacketMaterializationRecord:
        _normalise_hash("materialization_event_hash", materialization_event_hash)
        row = self._connection.execute(
            "SELECT * FROM packet_materialization_events "
            "WHERE materialization_event_hash = ?",
            (materialization_event_hash,),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"unknown packet materialization event {materialization_event_hash}"
            )
        return self._packet_materialization_from_row(row)

    def persist_packet_materialization(
        self,
        record: PacketMaterializationRecord,
        *,
        packet_artifact: ArtifactRecord,
        materialization_event_artifact: ArtifactRecord,
    ) -> PacketMaterializationRecord:
        """Atomically persist one post-access packet and its timing receipt."""

        for name, value in (
            ("materialization_event_hash", record.materialization_event_hash),
            ("query_access_event_hash", record.query_access_event_hash),
            ("snapshot_hash", record.snapshot_hash),
            ("packet_hash", record.packet_hash),
            ("retrieval_config_hash", record.retrieval_config_hash),
            ("packet_artifact_hash", record.packet_artifact_hash),
            (
                "materialization_event_artifact_hash",
                record.materialization_event_artifact_hash,
            ),
        ):
            _normalise_hash(name, value)
        for name, value in (
            ("materialization_event_id", record.materialization_event_id),
            ("execution_id", record.execution_id),
            ("retrieval_method", record.retrieval_method),
        ):
            _metadata_token(name, value)
        if record.retrieval_method not in {"all_admissible", "sqlite_fts5_bm25"}:
            raise ValueError("packet materialization uses an unregistered retrieval method")
        started = _normalise_timestamp(record.started_at)
        completed = _normalise_timestamp(record.completed_at)
        if _parse_timestamp(completed) < _parse_timestamp(started):
            raise ValueError("packet materialization completion predates its start")
        release = ReleaseClass(record.release_class)
        expected_artifacts = {
            record.packet_artifact_hash: packet_artifact,
            record.materialization_event_artifact_hash: materialization_event_artifact,
        }
        if len(expected_artifacts) != 2 or any(
            digest != artifact.content_hash
            for digest, artifact in expected_artifacts.items()
        ):
            raise ValueError("packet materialization CAS bindings are inconsistent")
        if any(artifact.release_class is not release for artifact in expected_artifacts.values()):
            raise ReleaseViolationError(
                "packet materialization and both CAS artifacts require the same release class"
            )
        fields = {
            "materialization_event_hash": record.materialization_event_hash,
            "materialization_event_id": record.materialization_event_id,
            "execution_id": record.execution_id,
            "query_access_event_hash": record.query_access_event_hash,
            "snapshot_hash": record.snapshot_hash,
            "packet_hash": record.packet_hash,
            "retrieval_method": record.retrieval_method,
            "retrieval_config_hash": record.retrieval_config_hash,
            "packet_artifact_hash": record.packet_artifact_hash,
            "materialization_event_artifact_hash": (
                record.materialization_event_artifact_hash
            ),
            "started_at": started,
            "completed_at": completed,
            "release_class": release.value,
        }
        with self._transaction() as cursor:
            access = cursor.execute(
                "SELECT * FROM query_access_events WHERE access_event_hash = ?",
                (record.query_access_event_hash,),
            ).fetchone()
            if access is None:
                raise ValueError("packet materialization requires persisted query access")
            if (
                access["execution_id"] != record.execution_id
                or access["snapshot_hash"] != record.snapshot_hash
                or _parse_timestamp(started) < _parse_timestamp(access["accessed_at"])
            ):
                raise ValueError("packet materialization timing or lineage is invalid")
            if access["release_class"] != release.value:
                raise ReleaseViolationError(
                    "packet materialization and query access require the same release class"
                )
            for artifact in expected_artifacts.values():
                self._register_artifact_with_cursor(cursor, artifact)
            row = self._insert_immutable_row(
                cursor,
                table="packet_materialization_events",
                key_column="materialization_event_hash",
                fields=fields,
            )
        return self._packet_materialization_from_row(row)

    def record_model_call(
        self,
        *,
        model_call_id: str,
        job_id: str,
        attempt_id: str,
        gpu_event_id: str | None,
        backend: ModelBackend,
        call_role: ModelCallRole,
        retry_class: RetryClass,
        model_manifest_hash: str,
        decoding_manifest_hash: str,
        request_hash: str,
        response_artifact_hash: str | None,
        construction_unit_hash: str,
        served_context_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        allocated_gpu_seconds: Any,
        successful: bool,
        created_at: Any | None = None,
    ) -> ModelCallRecord:
        _metadata_token("model_call_id", model_call_id)
        model_backend = ModelBackend(backend)
        role = ModelCallRole(call_role)
        retry = RetryClass(retry_class)
        for name, value in (
            ("model_manifest_hash", model_manifest_hash),
            ("decoding_manifest_hash", decoding_manifest_hash),
            ("request_hash", request_hash),
            ("construction_unit_hash", construction_unit_hash),
        ):
            _normalise_hash(name, value)
        if response_artifact_hash is not None:
            _normalise_hash("response_artifact_hash", response_artifact_hash)
            self.get_artifact(response_artifact_hash)
        contexts = _nonnegative_int("served_context_count", served_context_count)
        if contexts == 0:
            raise ValueError("served_context_count must be positive")
        prompt_count = _nonnegative_int("prompt_tokens", prompt_tokens)
        completion_count = _nonnegative_int("completion_tokens", completion_tokens)
        allocated_microseconds = _seconds_to_microseconds(allocated_gpu_seconds)
        if not isinstance(successful, bool):
            raise ValueError("successful must be a bool")

        job = self.get_job(job_id)
        attempt = self._connection.execute(
            "SELECT job_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if attempt is None:
            raise KeyError(f"unknown attempt {attempt_id}")
        if attempt["job_id"] != job_id:
            raise ValueError("model call attempt cannot cross jobs")
        if model_backend is ModelBackend.VLLM_GPU:
            if gpu_event_id is None:
                raise ValueError("vllm_gpu model calls require a GPU event")
            gpu_event = self._connection.execute(
                "SELECT * FROM gpu_events WHERE event_id = ?", (gpu_event_id,)
            ).fetchone()
            if gpu_event is None:
                raise KeyError(f"unknown GPU event {gpu_event_id}")
            if gpu_event["job_id"] not in (None, job_id):
                raise ValueError("model call GPU event cannot cross jobs")
            if gpu_event["allocated_microseconds"] != allocated_microseconds:
                raise ValueError("model call GPU duration must match its metered GPU event")
        elif gpu_event_id is not None or allocated_microseconds != 0:
            raise ValueError("hand-authored fixtures cannot claim allocated GPU time")
        self._ensure_public_compatible_artifacts(job.release_class, response_artifact_hash)
        row = self._append_metadata_record(
            table="model_calls",
            key_column="model_call_id",
            stable_fields={
                "model_call_id": model_call_id,
                "job_id": job_id,
                "attempt_id": attempt_id,
                "gpu_event_id": gpu_event_id,
                "backend": model_backend.value,
                "call_role": role.value,
                "retry_class": retry.value,
                "model_manifest_hash": model_manifest_hash,
                "decoding_manifest_hash": decoding_manifest_hash,
                "request_hash": request_hash,
                "response_artifact_hash": response_artifact_hash,
                "construction_unit_hash": construction_unit_hash,
                "served_context_count": contexts,
                "prompt_tokens": prompt_count,
                "completion_tokens": completion_count,
                "allocated_gpu_microseconds": allocated_microseconds,
                "successful": int(successful),
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return self._model_call_from_row(row)

    @staticmethod
    def _model_call_from_row(row: sqlite3.Row) -> ModelCallRecord:
        return ModelCallRecord(
            model_call_id=row["model_call_id"],
            job_id=row["job_id"],
            attempt_id=row["attempt_id"],
            gpu_event_id=row["gpu_event_id"],
            backend=ModelBackend(row["backend"]),
            call_role=ModelCallRole(row["call_role"]),
            retry_class=RetryClass(row["retry_class"]),
            model_manifest_hash=row["model_manifest_hash"],
            decoding_manifest_hash=row["decoding_manifest_hash"],
            request_hash=row["request_hash"],
            response_artifact_hash=row["response_artifact_hash"],
            construction_unit_hash=row["construction_unit_hash"],
            served_context_count=row["served_context_count"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            allocated_gpu_microseconds=row["allocated_gpu_microseconds"],
            successful=bool(row["successful"]),
            created_at=row["created_at"],
        )

    def get_model_call(self, model_call_id: str) -> ModelCallRecord:
        row = self._connection.execute(
            "SELECT * FROM model_calls WHERE model_call_id = ?", (model_call_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown model call {model_call_id}")
        return self._model_call_from_row(row)

    def record_validation(
        self,
        *,
        validation_id: str,
        job_id: str,
        attempt_id: str,
        input_artifact_hash: str,
        validator_manifest_hash: str,
        validation_status: ValidationStatus,
        evidence_support_status: EvidenceSupportStatus,
        temporal_status: TemporalValidationStatus,
        commitment_status: CommitmentCheckStatus,
        semantic_assessment_scope: SemanticAssessmentScope,
        diagnostics_artifact_hash: str | None = None,
        parent_validation_id: str | None = None,
        repair_attempt_id: str | None = None,
        created_at: Any | None = None,
    ) -> ValidationRecord:
        _metadata_token("validation_id", validation_id)
        for name, value in (
            ("input_artifact_hash", input_artifact_hash),
            ("validator_manifest_hash", validator_manifest_hash),
        ):
            _normalise_hash(name, value)
        if diagnostics_artifact_hash is not None:
            _normalise_hash("diagnostics_artifact_hash", diagnostics_artifact_hash)
        structural = ValidationStatus(validation_status)
        evidence = EvidenceSupportStatus(evidence_support_status)
        temporal = TemporalValidationStatus(temporal_status)
        commitment = CommitmentCheckStatus(commitment_status)
        assessment_scope = SemanticAssessmentScope(semantic_assessment_scope)
        semantic_statuses = (evidence, temporal, commitment)
        unassessed_statuses = (
            EvidenceSupportStatus.NOT_APPLICABLE,
            TemporalValidationStatus.NOT_APPLICABLE,
            CommitmentCheckStatus.NOT_APPLICABLE,
        )
        if assessment_scope is SemanticAssessmentScope.LEGACY_UNSPECIFIED:
            raise ValueError(
                "legacy_unspecified is migration provenance and cannot be written anew"
            )
        if (
            assessment_scope
            is SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            and semantic_statuses != unassessed_statuses
        ):
            raise ValueError(
                "runtime structural-only assessment requires unassessed semantic statuses"
            )
        if (
            assessment_scope is SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER
            and semantic_statuses == unassessed_statuses
        ):
            raise ValueError(
                "post-hoc assessment requires at least one assessed semantic status"
            )
        job = self.get_job(job_id)
        for linked_attempt_id in (attempt_id, repair_attempt_id):
            if linked_attempt_id is None:
                continue
            attempt = self._connection.execute(
                "SELECT job_id FROM attempts WHERE attempt_id = ?", (linked_attempt_id,)
            ).fetchone()
            if attempt is None:
                raise KeyError(f"unknown attempt {linked_attempt_id}")
            if attempt["job_id"] != job_id:
                raise ValueError("validation attempt lineage cannot cross jobs")
        if parent_validation_id is not None:
            parent = self._connection.execute(
                "SELECT job_id FROM validations WHERE validation_id = ?",
                (parent_validation_id,),
            ).fetchone()
            if parent is None:
                raise KeyError(f"unknown parent validation {parent_validation_id}")
            if parent["job_id"] != job_id:
                raise ValueError("validation lineage cannot cross jobs")
        self.get_artifact(input_artifact_hash)
        if diagnostics_artifact_hash is not None:
            self.get_artifact(diagnostics_artifact_hash)
        self._ensure_public_compatible_artifacts(
            job.release_class, input_artifact_hash, diagnostics_artifact_hash
        )
        row = self._append_metadata_record(
            table="validations",
            key_column="validation_id",
            stable_fields={
                "validation_id": validation_id,
                "job_id": job_id,
                "attempt_id": attempt_id,
                "input_artifact_hash": input_artifact_hash,
                "validator_manifest_hash": validator_manifest_hash,
                "validation_status": structural.value,
                "evidence_support_status": evidence.value,
                "temporal_status": temporal.value,
                "commitment_status": commitment.value,
                "semantic_assessment_scope": assessment_scope.value,
                "diagnostics_artifact_hash": diagnostics_artifact_hash,
                "parent_validation_id": parent_validation_id,
                "repair_attempt_id": repair_attempt_id,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return self._validation_from_row(row)

    @staticmethod
    def _validation_from_row(row: sqlite3.Row) -> ValidationRecord:
        return ValidationRecord(
            validation_id=row["validation_id"],
            job_id=row["job_id"],
            attempt_id=row["attempt_id"],
            input_artifact_hash=row["input_artifact_hash"],
            validator_manifest_hash=row["validator_manifest_hash"],
            validation_status=ValidationStatus(row["validation_status"]),
            evidence_support_status=EvidenceSupportStatus(row["evidence_support_status"]),
            temporal_status=TemporalValidationStatus(row["temporal_status"]),
            commitment_status=CommitmentCheckStatus(row["commitment_status"]),
            semantic_assessment_scope=SemanticAssessmentScope(
                row["semantic_assessment_scope"]
            ),
            diagnostics_artifact_hash=row["diagnostics_artifact_hash"],
            parent_validation_id=row["parent_validation_id"],
            repair_attempt_id=row["repair_attempt_id"],
            created_at=row["created_at"],
        )

    def get_validation(self, validation_id: str) -> ValidationRecord:
        row = self._connection.execute(
            "SELECT * FROM validations WHERE validation_id = ?", (validation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown validation {validation_id}")
        return self._validation_from_row(row)

    def record_projection(
        self,
        *,
        projection_id: str,
        job_id: str,
        validation_id: str,
        snapshot_id: str,
        packet_input_id: str,
        condition_id: str,
        context_hash: str,
        upper_ontology_hash: str,
        construction_certificate_hash: str,
        projection_artifact_hash: str,
        projection_semantic_hash: str,
        parent_projection_id: str | None = None,
        release_class: ReleaseClass,
        finalized_at: Any | None = None,
    ) -> ProjectionRecord:
        _metadata_token("projection_id", projection_id)
        _metadata_token("condition_id", condition_id)
        for name, value in (
            ("context_hash", context_hash),
            ("upper_ontology_hash", upper_ontology_hash),
            ("construction_certificate_hash", construction_certificate_hash),
            ("projection_artifact_hash", projection_artifact_hash),
            ("projection_semantic_hash", projection_semantic_hash),
        ):
            _normalise_hash(name, value)
        job = self.get_job(job_id)
        validation = self.get_validation(validation_id)
        if validation.job_id != job_id:
            raise ValueError("projection validation cannot cross jobs")
        if (
            validation.validation_status is not ValidationStatus.ACCEPTED
            or validation.semantic_assessment_scope
            is not SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            or validation.evidence_support_status
            is not EvidenceSupportStatus.NOT_APPLICABLE
            or validation.temporal_status is not TemporalValidationStatus.NOT_APPLICABLE
            or validation.commitment_status
            is not CommitmentCheckStatus.NOT_APPLICABLE
        ):
            raise ValueError(
                "projection requires an accepted runtime structural-only validation"
            )
        snapshot = self._connection.execute(
            """SELECT evidence_snapshots.snapshot_id, inputs.study_id
               FROM evidence_snapshots JOIN inputs USING (input_id)
               WHERE evidence_snapshots.snapshot_id = ?""",
            (snapshot_id,),
        ).fetchone()
        if snapshot is None:
            raise KeyError(f"unknown evidence snapshot {snapshot_id}")
        packet = self.get_input(packet_input_id)
        if packet.input_kind is not InputKind.EVIDENCE_PACKET:
            raise ValueError("projection packet_input_id must reference an evidence packet")
        if packet.study_id != snapshot["study_id"]:
            raise ValueError("snapshot and evidence packet cannot cross studies")
        linked = self._connection.execute(
            "SELECT 1 FROM study_jobs WHERE study_id = ? AND job_id = ?",
            (packet.study_id, job_id),
        ).fetchone()
        if linked is None:
            raise ValueError("projection job must be linked to the input study")
        self.get_artifact(projection_artifact_hash)
        if parent_projection_id is not None:
            parent = self._connection.execute(
                "SELECT job_id FROM projections WHERE projection_id = ?",
                (parent_projection_id,),
            ).fetchone()
            if parent is None:
                raise KeyError(f"unknown parent projection {parent_projection_id}")
            if parent["job_id"] != job_id:
                raise ValueError("projection lineage cannot cross jobs")
        release = ReleaseClass(release_class)
        if job.release_class is ReleaseClass.PUBLIC and release is ReleaseClass.RESTRICTED:
            raise ReleaseViolationError("a public job cannot finalize a restricted projection")
        self._ensure_public_compatible_artifacts(release, projection_artifact_hash)
        row = self._append_metadata_record(
            table="projections",
            key_column="projection_id",
            stable_fields={
                "projection_id": projection_id,
                "job_id": job_id,
                "validation_id": validation_id,
                "snapshot_id": snapshot_id,
                "packet_input_id": packet_input_id,
                "condition_id": condition_id,
                "context_hash": context_hash,
                "upper_ontology_hash": upper_ontology_hash,
                "construction_certificate_hash": construction_certificate_hash,
                "projection_artifact_hash": projection_artifact_hash,
                "projection_semantic_hash": projection_semantic_hash,
                "parent_projection_id": parent_projection_id,
                "release_class": release.value,
            },
            timestamp_column="finalized_at",
            timestamp=finalized_at,
        )
        return self._projection_from_row(row)

    @staticmethod
    def _projection_from_row(row: sqlite3.Row) -> ProjectionRecord:
        return ProjectionRecord(
            projection_id=row["projection_id"],
            job_id=row["job_id"],
            validation_id=row["validation_id"],
            snapshot_id=row["snapshot_id"],
            packet_input_id=row["packet_input_id"],
            condition_id=row["condition_id"],
            context_hash=row["context_hash"],
            upper_ontology_hash=row["upper_ontology_hash"],
            construction_certificate_hash=row["construction_certificate_hash"],
            projection_artifact_hash=row["projection_artifact_hash"],
            projection_semantic_hash=row["projection_semantic_hash"],
            parent_projection_id=row["parent_projection_id"],
            release_class=ReleaseClass(row["release_class"]),
            finalized_at=row["finalized_at"],
        )

    def get_projection(self, projection_id: str) -> ProjectionRecord:
        row = self._connection.execute(
            "SELECT * FROM projections WHERE projection_id = ?", (projection_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown projection {projection_id}")
        return self._projection_from_row(row)

    def projections_for_job(self, job_id: str) -> tuple[ProjectionRecord, ...]:
        """Return the immutable projection rows owned by one exact job.

        This deliberately requires a ledger job identifier rather than accepting
        descriptive condition/unit fields.  Downstream scoring may therefore
        resolve only identities already bound by generation artifacts.
        """

        self.get_job(job_id)
        rows = self._connection.execute(
            "SELECT * FROM projections WHERE job_id = ? ORDER BY projection_id",
            (job_id,),
        ).fetchall()
        return tuple(self._projection_from_row(row) for row in rows)

    def resolve_projection_artifact(
        self,
        *,
        study_id: str,
        projection_artifact_hash: str,
        condition_id: str,
        context_hash: str,
    ) -> ProjectionRecord:
        """Resolve one hash-bound study projection, failing closed on ambiguity."""

        self.get_study(study_id)
        _normalise_hash("projection_artifact_hash", projection_artifact_hash)
        _metadata_token("condition_id", condition_id)
        _normalise_hash("context_hash", context_hash)
        rows = self._connection.execute(
            """SELECT p.* FROM projections p
               JOIN study_jobs sj ON sj.job_id = p.job_id
               WHERE sj.study_id = ?
                 AND p.projection_artifact_hash = ?
                 AND p.condition_id = ?
                 AND p.context_hash = ?
               ORDER BY p.projection_id""",
            (study_id, projection_artifact_hash, condition_id, context_hash),
        ).fetchall()
        if len(rows) != 1:
            raise ArtifactIntegrityError(
                "projection artifact did not resolve to exactly one study ledger row"
            )
        return self._projection_from_row(rows[0])

    def resolve_projection_artifact_owner(
        self,
        *,
        projection_artifact_hash: str,
        projection_semantic_hash: str,
        condition_id: str,
        context_hash: str,
    ) -> tuple[StudyRecord, ProjectionRecord]:
        """Resolve one exact projection and its unique owning study globally.

        Feedback consumes held-out projections from a later combined-study job,
        so its input manifest cannot truthfully assume the receiving study owns
        the source.  All four immutable semantic bindings are required and an
        ambiguous/missing study ownership fails closed.
        """

        for name, value in (
            ("projection_artifact_hash", projection_artifact_hash),
            ("projection_semantic_hash", projection_semantic_hash),
            ("context_hash", context_hash),
        ):
            _normalise_hash(name, value)
        _metadata_token("condition_id", condition_id)
        rows = self._connection.execute(
            """SELECT sj.study_id, p.* FROM projections p
               JOIN study_jobs sj ON sj.job_id = p.job_id
               WHERE p.projection_artifact_hash = ?
                 AND p.projection_semantic_hash = ?
                 AND p.condition_id = ?
                 AND p.context_hash = ?
               ORDER BY sj.study_id, p.projection_id""",
            (
                projection_artifact_hash,
                projection_semantic_hash,
                condition_id,
                context_hash,
            ),
        ).fetchall()
        if len(rows) != 1:
            raise ArtifactIntegrityError(
                "projection did not resolve to exactly one immutable study owner"
            )
        row = rows[0]
        return self.get_study(row["study_id"]), self._projection_from_row(row)

    def resolve_failure_job(
        self,
        *,
        study_id: str,
        failure_artifact_hash: str,
    ) -> JobRecord:
        """Resolve one failed attempt by its immutable ledger failure artifact."""

        self.get_study(study_id)
        _normalise_hash("failure_artifact_hash", failure_artifact_hash)
        rows = self._connection.execute(
            """SELECT DISTINCT j.*,
                      (SELECT to_state FROM job_transitions t
                       WHERE t.job_id = j.job_id ORDER BY sequence DESC LIMIT 1) AS state
               FROM failures f
               JOIN attempts a ON a.attempt_id = f.attempt_id
               JOIN jobs j ON j.job_id = a.job_id
               JOIN study_jobs sj ON sj.job_id = j.job_id
               WHERE sj.study_id = ? AND f.artifact_hash = ?
               ORDER BY j.job_id""",
            (study_id, failure_artifact_hash),
        ).fetchall()
        if len(rows) != 1:
            raise ArtifactIntegrityError(
                "failure artifact did not resolve to exactly one study ledger job"
            )
        row = rows[0]
        return JobRecord(
            job_id=row["job_id"],
            identity_hash=row["identity_hash"],
            identity_json=row["identity_json"],
            release_class=ReleaseClass(row["release_class"]),
            state=JobState(row["state"]),
            created_at=row["created_at"],
        )

    def record_feedback(
        self,
        *,
        feedback_id: str,
        study_id: str,
        job_id: str | None,
        feedback_kind: FeedbackKind,
        action: FeedbackAction,
        revision_hash: str,
        anchor_manifest_hash: str,
        before_context_hash: str,
        after_context_hash: str,
        receiving_condition: str | None,
        before_projection_id: str | None,
        after_projection_id: str | None,
        resolution_status: FeedbackResolutionStatus,
        resolution_artifact_hash: str | None,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> FeedbackRecord:
        _metadata_token("feedback_id", feedback_id)
        self.get_study(study_id)
        kind = FeedbackKind(feedback_kind)
        typed_action = FeedbackAction(action)
        status = FeedbackResolutionStatus(resolution_status)
        for name, value in (
            ("revision_hash", revision_hash),
            ("anchor_manifest_hash", anchor_manifest_hash),
            ("before_context_hash", before_context_hash),
            ("after_context_hash", after_context_hash),
        ):
            _normalise_hash(name, value)
        if kind is FeedbackKind.USER_REVISION:
            if (
                any(
                    value is not None
                    for value in (
                        job_id,
                        receiving_condition,
                        before_projection_id,
                        after_projection_id,
                        resolution_artifact_hash,
                    )
                )
                or status is not FeedbackResolutionStatus.PENDING
            ):
                raise ValueError("a shared user revision cannot contain condition output")
        else:
            if (
                job_id is None
                or receiving_condition is None
                or before_projection_id is None
                or resolution_artifact_hash is None
                or status is FeedbackResolutionStatus.PENDING
            ):
                raise ValueError(
                    "a condition resolution requires its own job/projection/status/artifact"
                )
            if (
                status is FeedbackResolutionStatus.APPLIED
                and after_projection_id is None
            ):
                raise ValueError("applied feedback requires its after projection")
            _metadata_token("receiving_condition", receiving_condition)
            receiving_job = self.get_job(job_id)
            before = self.get_projection(before_projection_id)
            if before.condition_id != receiving_condition:
                raise ValueError("feedback source condition differs from receiving condition")
            if before.context_hash != before_context_hash:
                raise ValueError("feedback source projection context changed")
            # A feedback job receives a revision to a projection constructed by
            # an earlier source job.  The source projection may come from the
            # held-out study while the receiving feedback job belongs to the
            # combined block.  Require the source to have one unambiguous study
            # owner, but never rewrite/link it into the receiving study.
            self.study_job_for_job(before.job_id)
            if after_projection_id is not None:
                after = self.get_projection(after_projection_id)
                if after.job_id != job_id:
                    raise ValueError(
                        "feedback result projection must belong to its receiving job"
                    )
                if (
                    after.condition_id != receiving_condition
                    or after.context_hash != after_context_hash
                ):
                    raise ValueError("feedback result condition/context changed")
            if (
                self._connection.execute(
                    "SELECT 1 FROM study_jobs WHERE study_id = ? AND job_id = ?",
                    (study_id, job_id),
                ).fetchone()
                is None
            ):
                raise ValueError("feedback job must be linked to its study")
            if receiving_job.state not in {
                JobState.FINALIZED,
                JobState.SCORED,
                JobState.RENDERED,
            }:
                raise ValueError("feedback receiving job must be finalized")
        if resolution_artifact_hash is not None:
            _normalise_hash("resolution_artifact_hash", resolution_artifact_hash)
            self.get_artifact(resolution_artifact_hash)
        release = ReleaseClass(release_class)
        self._ensure_public_compatible_artifacts(release, resolution_artifact_hash)
        row = self._append_metadata_record(
            table="feedback",
            key_column="feedback_id",
            stable_fields={
                "feedback_id": feedback_id,
                "study_id": study_id,
                "job_id": job_id,
                "feedback_kind": kind.value,
                "action": typed_action.value,
                "revision_hash": revision_hash,
                "anchor_manifest_hash": anchor_manifest_hash,
                "before_context_hash": before_context_hash,
                "after_context_hash": after_context_hash,
                "receiving_condition": receiving_condition,
                "before_projection_id": before_projection_id,
                "after_projection_id": after_projection_id,
                "resolution_status": status.value,
                "resolution_artifact_hash": resolution_artifact_hash,
                "release_class": release.value,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return self._feedback_from_row(row)

    @staticmethod
    def _feedback_from_row(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            feedback_id=row["feedback_id"],
            study_id=row["study_id"],
            job_id=row["job_id"],
            feedback_kind=FeedbackKind(row["feedback_kind"]),
            action=FeedbackAction(row["action"]),
            revision_hash=row["revision_hash"],
            anchor_manifest_hash=row["anchor_manifest_hash"],
            before_context_hash=row["before_context_hash"],
            after_context_hash=row["after_context_hash"],
            receiving_condition=row["receiving_condition"],
            before_projection_id=row["before_projection_id"],
            after_projection_id=row["after_projection_id"],
            resolution_status=FeedbackResolutionStatus(row["resolution_status"]),
            resolution_artifact_hash=row["resolution_artifact_hash"],
            release_class=ReleaseClass(row["release_class"]),
            created_at=row["created_at"],
        )

    def record_metric(
        self,
        *,
        metric_id: str,
        study_id: str,
        job_id: str,
        projection_id: str | None,
        unit_hash: str,
        metric_name: str,
        metric_version_hash: str,
        status: MetricStatus,
        value: float | int | None,
        numerator: float | int | None = None,
        denominator: float | int | None = None,
        result_artifact_hash: str | None = None,
        created_at: Any | None = None,
    ) -> MetricRecord:
        _metadata_token("metric_id", metric_id)
        _metadata_token("metric_name", metric_name)
        self.get_study(study_id)
        _normalise_hash("unit_hash", unit_hash)
        _normalise_hash("metric_version_hash", metric_version_hash)
        metric_status = MetricStatus(status)
        numeric_value = _finite_optional_number("value", value)
        numeric_numerator = _finite_optional_number("numerator", numerator)
        numeric_denominator = _finite_optional_number("denominator", denominator)
        if metric_status is MetricStatus.VALUE and numeric_value is None:
            raise ValueError("a metric with value status requires a numeric value")
        if metric_status is not MetricStatus.VALUE and numeric_value is not None:
            raise ValueError("an unavailable metric cannot carry a numeric value")
        if numeric_denominator is not None and numeric_denominator < 0:
            raise ValueError("metric denominator cannot be negative")
        job = self.get_job(job_id)
        linked = self._connection.execute(
            "SELECT 1 FROM study_jobs WHERE study_id = ? AND job_id = ?",
            (study_id, job_id),
        ).fetchone()
        if linked is None:
            raise ValueError("metric job must belong to its study")
        if job.state not in {
            JobState.FINALIZED,
            JobState.SCORED,
            JobState.RENDERED,
        }:
            raise ValueError("metric job must be finalized before scoring")
        if projection_id is not None:
            projection = self.get_projection(projection_id)
            if projection.job_id != job_id:
                raise ValueError("metric projection must belong to its exact job")
        if result_artifact_hash is not None:
            _normalise_hash("result_artifact_hash", result_artifact_hash)
            self.get_artifact(result_artifact_hash)
        study = self.get_study(study_id)
        self._ensure_public_compatible_artifacts(study.release_class, result_artifact_hash)
        row = self._append_metadata_record(
            table="metrics",
            key_column="metric_id",
            stable_fields={
                "metric_id": metric_id,
                "study_id": study_id,
                "job_id": job_id,
                "projection_id": projection_id,
                "unit_hash": unit_hash,
                "metric_name": metric_name,
                "metric_version_hash": metric_version_hash,
                "status": metric_status.value,
                "value": numeric_value,
                "numerator": numeric_numerator,
                "denominator": numeric_denominator,
                "result_artifact_hash": result_artifact_hash,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return MetricRecord(
            metric_id=row["metric_id"],
            study_id=row["study_id"],
            job_id=row["job_id"],
            projection_id=row["projection_id"],
            unit_hash=row["unit_hash"],
            metric_name=row["metric_name"],
            metric_version_hash=row["metric_version_hash"],
            status=MetricStatus(row["status"]),
            value=row["value"],
            numerator=row["numerator"],
            denominator=row["denominator"],
            result_artifact_hash=row["result_artifact_hash"],
            created_at=row["created_at"],
        )

    def metrics_for_job(self, *, study_id: str, job_id: str) -> tuple[MetricRecord, ...]:
        """Return the exact durable metric inventory for one study generation job."""

        self.get_study_job(study_id=study_id, job_id=job_id)
        rows = self._connection.execute(
            """SELECT * FROM metrics
               WHERE study_id = ? AND job_id = ? ORDER BY metric_id""",
            (study_id, job_id),
        ).fetchall()
        return tuple(
            MetricRecord(
                metric_id=row["metric_id"],
                study_id=row["study_id"],
                job_id=row["job_id"],
                projection_id=row["projection_id"],
                unit_hash=row["unit_hash"],
                metric_name=row["metric_name"],
                metric_version_hash=row["metric_version_hash"],
                status=MetricStatus(row["status"]),
                value=row["value"],
                numerator=row["numerator"],
                denominator=row["denominator"],
                result_artifact_hash=row["result_artifact_hash"],
                created_at=row["created_at"],
            )
            for row in rows
        )

    def record_visualization(
        self,
        *,
        visualization_id: str,
        projection_id: str,
        renderer_configuration_hash: str,
        semantic_hash: str,
        visualization_artifact_hash: str,
        layout_seed: int,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> VisualizationRecord:
        _metadata_token("visualization_id", visualization_id)
        for name, value in (
            ("renderer_configuration_hash", renderer_configuration_hash),
            ("semantic_hash", semantic_hash),
            ("visualization_artifact_hash", visualization_artifact_hash),
        ):
            _normalise_hash(name, value)
        if isinstance(layout_seed, bool) or not isinstance(layout_seed, int):
            raise ValueError("layout_seed must be an integer")
        projection = self.get_projection(projection_id)
        if (
            projection.projection_semantic_hash is None
            or semantic_hash != projection.projection_semantic_hash
        ):
            raise ValueError(
                "visualization semantic hash must name its exact projection content"
            )
        self.get_artifact(visualization_artifact_hash)
        release = ReleaseClass(release_class)
        if projection.release_class is ReleaseClass.PUBLIC and release is ReleaseClass.RESTRICTED:
            raise ReleaseViolationError(
                "a public projection cannot have a restricted visualization"
            )
        self._ensure_public_compatible_artifacts(release, visualization_artifact_hash)
        row = self._append_metadata_record(
            table="visualizations",
            key_column="visualization_id",
            stable_fields={
                "visualization_id": visualization_id,
                "projection_id": projection_id,
                "renderer_configuration_hash": renderer_configuration_hash,
                "semantic_hash": semantic_hash,
                "visualization_artifact_hash": visualization_artifact_hash,
                "layout_seed": layout_seed,
                "release_class": release.value,
            },
            timestamp_column="created_at",
            timestamp=created_at,
        )
        return VisualizationRecord(
            visualization_id=row["visualization_id"],
            projection_id=row["projection_id"],
            renderer_configuration_hash=row["renderer_configuration_hash"],
            semantic_hash=row["semantic_hash"],
            visualization_artifact_hash=row["visualization_artifact_hash"],
            layout_seed=row["layout_seed"],
            release_class=ReleaseClass(row["release_class"]),
            created_at=row["created_at"],
        )

    def record_resource_sample(
        self,
        *,
        sample_id: str,
        job_id: str | None,
        gpu_event_id: str | None,
        process_ram_bytes: int,
        system_available_ram_bytes: int,
        gpu_vram_bytes: int,
        project_storage_bytes: int,
        cpu_worker_count: int,
        sampled_at: Any,
    ) -> ResourceSampleRecord:
        _metadata_token("sample_id", sample_id)
        values = {
            "process_ram_bytes": _nonnegative_int("process_ram_bytes", process_ram_bytes),
            "system_available_ram_bytes": _nonnegative_int(
                "system_available_ram_bytes", system_available_ram_bytes
            ),
            "gpu_vram_bytes": _nonnegative_int("gpu_vram_bytes", gpu_vram_bytes),
            "project_storage_bytes": _nonnegative_int(
                "project_storage_bytes", project_storage_bytes
            ),
            "cpu_worker_count": _nonnegative_int("cpu_worker_count", cpu_worker_count),
        }
        if job_id is not None:
            self.get_job(job_id)
        if gpu_event_id is not None:
            event = self._connection.execute(
                "SELECT job_id FROM gpu_events WHERE event_id = ?", (gpu_event_id,)
            ).fetchone()
            if event is None:
                raise KeyError(f"unknown GPU event {gpu_event_id}")
            if job_id is not None and event["job_id"] not in (None, job_id):
                raise ValueError("resource sample GPU event cannot cross jobs")
        row = self._append_metadata_record(
            table="resource_samples",
            key_column="sample_id",
            stable_fields={
                "sample_id": sample_id,
                "job_id": job_id,
                "gpu_event_id": gpu_event_id,
                **values,
            },
            timestamp_column="sampled_at",
            timestamp=sampled_at,
        )
        return ResourceSampleRecord(
            sample_id=row["sample_id"],
            job_id=row["job_id"],
            gpu_event_id=row["gpu_event_id"],
            process_ram_bytes=row["process_ram_bytes"],
            system_available_ram_bytes=row["system_available_ram_bytes"],
            gpu_vram_bytes=row["gpu_vram_bytes"],
            project_storage_bytes=row["project_storage_bytes"],
            cpu_worker_count=row["cpu_worker_count"],
            sampled_at=row["sampled_at"],
        )

    def resource_samples_with_prefix(
        self, sample_id_prefix: str
    ) -> tuple[ResourceSampleRecord, ...]:
        """Read prior append-only samples for interrupted-run gate replay."""

        if not sample_id_prefix:
            raise ValueError("sample_id_prefix must be nonempty")
        return tuple(
            sample
            for sample in self.resource_samples()
            if sample.sample_id.startswith(sample_id_prefix)
        )

    def resource_samples(self) -> tuple[ResourceSampleRecord, ...]:
        """Return every immutable resource sample for cumulative safety gates."""

        rows = self._connection.execute(
            "SELECT * FROM resource_samples ORDER BY sampled_at, sample_id"
        ).fetchall()
        return tuple(
            ResourceSampleRecord(
                sample_id=row["sample_id"],
                job_id=row["job_id"],
                gpu_event_id=row["gpu_event_id"],
                process_ram_bytes=row["process_ram_bytes"],
                system_available_ram_bytes=row["system_available_ram_bytes"],
                gpu_vram_bytes=row["gpu_vram_bytes"],
                project_storage_bytes=row["project_storage_bytes"],
                cpu_worker_count=row["cpu_worker_count"],
                sampled_at=row["sampled_at"],
            )
            for row in rows
        )

    def create_or_resume_job(
        self,
        identity_components: Mapping[str, Any],
        *,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> JobRecord:
        identity_json = canonical_json(identity_components)
        identity_hash = sha256_bytes(identity_json.encode("utf-8"))
        release = ReleaseClass(release_class)
        existing = self._connection.execute(
            "SELECT * FROM jobs WHERE identity_hash = ?", (identity_hash,)
        ).fetchone()
        if existing is not None:
            if (
                existing["identity_json"] != identity_json
                or existing["release_class"] != release.value
            ):
                raise DuplicateConflictError("job identity was reused with incompatible metadata")
            return self.get_job(existing["job_id"])

        timestamp = _normalise_timestamp(created_at)
        job_id = identity_hash
        try:
            with self._transaction() as cursor:
                cursor.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                    (job_id, identity_hash, identity_json, release.value, timestamp),
                )
                cursor.execute(
                    """INSERT INTO job_transitions
                       (job_id, sequence, from_state, to_state, occurred_at)
                       VALUES (?, 0, NULL, ?, ?)""",
                    (job_id, JobState.PLANNED.value, timestamp),
                )
        except sqlite3.IntegrityError as exc:
            # A concurrent process may have inserted the same immutable job.
            resumed = self._connection.execute(
                "SELECT * FROM jobs WHERE identity_hash = ?", (identity_hash,)
            ).fetchone()
            if resumed is None:
                raise
            if (
                resumed["identity_json"] != identity_json
                or resumed["release_class"] != release.value
            ):
                raise DuplicateConflictError("concurrent job metadata conflict") from exc
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        row = self._connection.execute(
            """SELECT j.*,
                      (SELECT to_state FROM job_transitions t
                       WHERE t.job_id = j.job_id ORDER BY sequence DESC LIMIT 1) AS state
               FROM jobs j WHERE j.job_id = ?""",
            (job_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown job {job_id}")
        return JobRecord(
            job_id=row["job_id"],
            identity_hash=row["identity_hash"],
            identity_json=row["identity_json"],
            release_class=ReleaseClass(row["release_class"]),
            state=JobState(row["state"]),
            created_at=row["created_at"],
        )

    def is_frozen_legacy_fallback_c1_retry_prebuild(self, job_id: str) -> bool:
        """Return whether ``job_id`` is the one authenticated pre-typed C1 retry job."""

        job = self.get_job(job_id)
        try:
            identity = json.loads(job.identity_json)
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(identity, Mapping) and (
            frozen_legacy_fallback_c1_retry_prebuild_matches(
                self._connection,
                job_id=job_id,
                identity=identity,
            )
        )

    def resolve_job_identity(
        self,
        identity_components: Mapping[str, Any],
        *,
        study_id: str | None = None,
    ) -> JobRecord:
        """Resolve an exact canonical identity without creating a missing job."""

        identity_json = canonical_json(identity_components)
        identity_hash = sha256_bytes(identity_json.encode("utf-8"))
        try:
            job = self.get_job(identity_hash)
        except KeyError as error:
            raise ArtifactIntegrityError(
                "canonical job identity is absent from the ledger"
            ) from error
        if job.identity_hash != identity_hash or job.identity_json != identity_json:
            raise ArtifactIntegrityError("canonical job identity metadata changed")
        if study_id is not None:
            self.get_study_job(study_id=study_id, job_id=job.job_id)
        return job

    def resolve_model_call_job(
        self,
        *,
        study_id: str,
        model_call_ids: Sequence[str],
    ) -> JobRecord:
        """Resolve one study job from a nonempty immutable model-call lineage."""

        identifiers = tuple(model_call_ids)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ArtifactIntegrityError(
                "model-call job resolution requires nonempty unique identifiers"
            )
        try:
            calls = tuple(self.get_model_call(identifier) for identifier in identifiers)
        except KeyError as error:
            raise ArtifactIntegrityError(
                "model-call lineage is absent from the ledger"
            ) from error
        job_ids = {call.job_id for call in calls}
        if len(job_ids) != 1:
            raise ArtifactIntegrityError("model-call lineage crosses ledger jobs")
        job = self.get_job(job_ids.pop())
        self.get_study_job(study_id=study_id, job_id=job.job_id)
        return job

    def transition_job(
        self,
        job_id: str,
        to_state: JobState,
        *,
        occurred_at: Any | None = None,
        frozen_legacy_fallback_c1_retry: bool = False,
    ) -> JobTransition:
        target = JobState(to_state)
        explicit_timestamp = _normalise_timestamp(occurred_at) if occurred_at is not None else None
        with self._transaction() as cursor:
            row = cursor.execute(
                """SELECT event_id, sequence, from_state, to_state, occurred_at
                   FROM job_transitions WHERE job_id = ?
                   ORDER BY sequence DESC LIMIT 1""",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown job {job_id}")
            current = JobState(row["to_state"])
            if current is target:
                # A replay cannot rewrite the original transition timestamp.
                return self._transition_from_row(job_id, row)
            if current is JobState.PREQUERY_SEALED and target is JobState.GENERATED:
                identity_row = cursor.execute(
                    "SELECT identity_json FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if identity_row is None:
                    raise KeyError(f"unknown job {job_id}")
                try:
                    identity = json.loads(identity_row["identity_json"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ArtifactIntegrityError("job identity is not canonical JSON") from exc
                typed_query_blind = isinstance(identity, Mapping) and (
                    identity.get("lifecycle_kind") == "query_blind_prebuild"
                    and identity.get("condition") in {"C0", "C1"}
                )
                frozen_legacy_query_blind = (
                    frozen_legacy_fallback_c1_retry
                    and isinstance(identity, Mapping)
                    and frozen_legacy_fallback_c1_retry_prebuild_matches(
                        cursor,
                        job_id=job_id,
                        identity=identity,
                    )
                )
                if not typed_query_blind and not frozen_legacy_query_blind:
                    raise InvalidTransitionError(
                        "only an explicitly typed C0/C1 query-blind prebuild may "
                        "transition from prequery_sealed directly to generated"
                    )
            if target not in ALLOWED_JOB_TRANSITIONS[current]:
                raise InvalidTransitionError(
                    f"cannot transition job from {current.value} to {target.value}"
                )
            timestamp = explicit_timestamp or _normalise_timestamp()
            previous_timestamp = _parse_timestamp(row["occurred_at"])
            parsed_timestamp = _parse_timestamp(timestamp)
            if parsed_timestamp < previous_timestamp:
                raise InvalidTransitionError(
                    "job lifecycle timestamps must be nondecreasing"
                )
            if (
                current is JobState.PREQUERY_SEALED
                and target is JobState.QUERY_REVEALED
                and parsed_timestamp <= previous_timestamp
            ):
                raise InvalidTransitionError(
                    "query reveal must strictly follow the prequery seal"
                )
            cursor.execute(
                """INSERT INTO job_transitions
                   (job_id, sequence, from_state, to_state, occurred_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (job_id, row["sequence"] + 1, current.value, target.value, timestamp),
            )
            event_id = int(cursor.lastrowid)
        return JobTransition(
            event_id=event_id,
            job_id=job_id,
            sequence=row["sequence"] + 1,
            from_state=current,
            to_state=target,
            occurred_at=timestamp,
        )

    def advance_job_lifecycle(
        self,
        job_id: str,
        milestones: Sequence[tuple[JobState, Any]],
        *,
        frozen_legacy_fallback_c1_retry: bool = False,
    ) -> tuple[JobTransition, ...]:
        """Append a frozen lifecycle prefix, or verify its exact replay.

        Runtime recovery frequently revisits a job after a later milestone is
        already durable.  Calling :meth:`transition_job` for an earlier state
        would then look like an illegal backwards transition.  This helper
        compares the complete append-only prefix first and appends only the
        missing suffix.  A changed repair/non-repair path fails closed.

        ``milestones`` deliberately omits ``PLANNED`` because job creation
        writes that event atomically.  Supplied timestamps must be monotonic;
        timestamps for already-durable events are never rewritten.
        """

        requested = tuple((JobState(state), timestamp) for state, timestamp in milestones)
        requested_states = (JobState.PLANNED, *(state for state, _ in requested))
        if len(set(requested_states)) != len(requested_states):
            raise ValueError("job lifecycle milestones must not repeat a state")

        existing = self.transitions(job_id)
        existing_states = tuple(item.to_state for item in existing)
        shared = min(len(existing_states), len(requested_states))
        if existing_states[:shared] != requested_states[:shared]:
            raise InvalidTransitionError(
                "durable job lifecycle differs from the requested replay prefix"
            )
        if len(existing_states) >= len(requested_states):
            return existing

        previous_time = _parse_timestamp(existing[-1].occurred_at)
        for state, occurred_at in requested[len(existing_states) - 1 :]:
            timestamp = _normalise_timestamp(occurred_at)
            parsed = _parse_timestamp(timestamp)
            if parsed < previous_time or (
                existing[-1].to_state is JobState.PREQUERY_SEALED
                and state is JobState.QUERY_REVEALED
                and parsed <= previous_time
            ):
                raise InvalidTransitionError(
                    "job lifecycle timestamps are not valid for the requested boundary"
                )
            self.transition_job(
                job_id,
                state,
                occurred_at=timestamp,
                frozen_legacy_fallback_c1_retry=(
                    frozen_legacy_fallback_c1_retry
                    and existing[-1].to_state is JobState.PREQUERY_SEALED
                    and state is JobState.GENERATED
                ),
            )
            previous_time = parsed
            existing = self.transitions(job_id)
        return self.transitions(job_id)

    @staticmethod
    def _transition_from_row(job_id: str, row: sqlite3.Row) -> JobTransition:
        return JobTransition(
            event_id=row["event_id"],
            job_id=job_id,
            sequence=row["sequence"],
            from_state=JobState(row["from_state"]) if row["from_state"] else None,
            to_state=JobState(row["to_state"]),
            occurred_at=row["occurred_at"],
        )

    def transitions(self, job_id: str) -> tuple[JobTransition, ...]:
        rows = self._connection.execute(
            "SELECT * FROM job_transitions WHERE job_id = ? ORDER BY sequence", (job_id,)
        ).fetchall()
        if not rows:
            raise KeyError(f"unknown job {job_id}")
        return tuple(self._transition_from_row(job_id, row) for row in rows)

    def record_attempt(
        self,
        *,
        attempt_id: str,
        job_id: str,
        attempt_kind: AttemptKind,
        input_hash: str,
        config_hash: str,
        seed: int,
        parent_attempt_id: str | None = None,
        created_at: Any | None = None,
    ) -> AttemptRecord:
        if not attempt_id:
            raise ValueError("attempt_id must be nonempty")
        kind = AttemptKind(attempt_kind)
        _normalise_hash("input_hash", input_hash)
        _normalise_hash("config_hash", config_hash)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if (kind is AttemptKind.BASE) != (parent_attempt_id is None):
            raise ValueError("only base attempts omit a parent attempt")

        existing = self._connection.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        expected = (job_id, kind.value, parent_attempt_id, input_hash, config_hash, seed)
        if existing is not None:
            actual = tuple(
                existing[key]
                for key in (
                    "job_id",
                    "attempt_kind",
                    "parent_attempt_id",
                    "input_hash",
                    "config_hash",
                    "seed",
                )
            )
            if actual != expected:
                raise DuplicateConflictError("attempt_id was reused with different content")
            if (
                created_at is not None
                and _normalise_timestamp(created_at) != existing["created_at"]
            ):
                raise DuplicateConflictError("attempt_id was reused with a different timestamp")
            return self._attempt_from_row(existing)

        timestamp = _normalise_timestamp(created_at)
        with self._transaction() as cursor:
            if cursor.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone() is None:
                raise KeyError(f"unknown job {job_id}")
            if parent_attempt_id is not None:
                parent = cursor.execute(
                    "SELECT job_id FROM attempts WHERE attempt_id = ?", (parent_attempt_id,)
                ).fetchone()
                if parent is None:
                    raise KeyError(f"unknown parent attempt {parent_attempt_id}")
                if parent["job_id"] != job_id:
                    raise ValueError("attempt lineage cannot cross jobs")
            try:
                cursor.execute(
                    "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        attempt_id,
                        job_id,
                        kind.value,
                        parent_attempt_id,
                        input_hash,
                        config_hash,
                        seed,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if kind is AttemptKind.REPAIR:
                    raise DuplicateConflictError(
                        "the parent attempt already has its single permitted repair"
                    ) from exc
                raise
        return AttemptRecord(
            attempt_id=attempt_id,
            job_id=job_id,
            attempt_kind=kind,
            parent_attempt_id=parent_attempt_id,
            input_hash=input_hash,
            config_hash=config_hash,
            seed=seed,
            created_at=timestamp,
        )

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> AttemptRecord:
        return AttemptRecord(
            attempt_id=row["attempt_id"],
            job_id=row["job_id"],
            attempt_kind=AttemptKind(row["attempt_kind"]),
            parent_attempt_id=row["parent_attempt_id"],
            input_hash=row["input_hash"],
            config_hash=row["config_hash"],
            seed=row["seed"],
            created_at=row["created_at"],
        )

    def attempt_lineage(self, attempt_id: str) -> tuple[AttemptRecord, ...]:
        lineage = []
        seen = set()
        current: str | None = attempt_id
        while current is not None:
            if current in seen:
                raise ArtifactIntegrityError("attempt lineage contains a cycle")
            seen.add(current)
            row = self._connection.execute(
                "SELECT * FROM attempts WHERE attempt_id = ?", (current,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown attempt {current}")
            record = self._attempt_from_row(row)
            lineage.append(record)
            current = record.parent_attempt_id
        lineage.reverse()
        return tuple(lineage)

    def record_failure(
        self,
        *,
        attempt_id: str,
        failure_kind: FailureKind,
        message: str,
        details: Mapping[str, Any] | None = None,
        artifact_hash: str | None = None,
        occurred_at: Any | None = None,
    ) -> FailureRecord:
        kind = FailureKind(failure_kind)
        if not message:
            raise ValueError("failure message must be nonempty")
        details_json = canonical_json(details or {})
        if artifact_hash is not None:
            _normalise_hash("artifact_hash", artifact_hash)
        immutable = {
            "attempt_id": attempt_id,
            "failure_kind": kind.value,
            "message": message,
            "details": json.loads(details_json),
            "artifact_hash": artifact_hash,
        }
        failure_id = sha256_bytes(canonical_json(immutable).encode("utf-8"))
        existing = self._connection.execute(
            "SELECT * FROM failures WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        if existing is not None:
            expected = (failure_id, kind.value, message, details_json, artifact_hash)
            actual = tuple(
                existing[key]
                for key in (
                    "failure_id",
                    "failure_kind",
                    "message",
                    "details_json",
                    "artifact_hash",
                )
            )
            if actual != expected:
                raise DuplicateConflictError("attempt already has a different failure record")
            if (
                occurred_at is not None
                and _normalise_timestamp(occurred_at) != existing["occurred_at"]
            ):
                raise DuplicateConflictError("failure was reused with a different timestamp")
            return self._failure_from_row(existing)
        timestamp = _normalise_timestamp(occurred_at)
        try:
            with self._transaction() as cursor:
                cursor.execute(
                    "INSERT INTO failures VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        failure_id,
                        attempt_id,
                        kind.value,
                        message,
                        details_json,
                        artifact_hash,
                        timestamp,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise KeyError(f"unknown attempt {attempt_id}") from exc
        return FailureRecord(
            failure_id=failure_id,
            attempt_id=attempt_id,
            failure_kind=kind,
            message=message,
            details_json=details_json,
            artifact_hash=artifact_hash,
            occurred_at=timestamp,
        )

    @staticmethod
    def _failure_from_row(row: sqlite3.Row) -> FailureRecord:
        return FailureRecord(
            failure_id=row["failure_id"],
            attempt_id=row["attempt_id"],
            failure_kind=FailureKind(row["failure_kind"]),
            message=row["message"],
            details_json=row["details_json"],
            artifact_hash=row["artifact_hash"],
            occurred_at=row["occurred_at"],
        )

    def failures_for_lineage(self, attempt_id: str) -> tuple[FailureRecord, ...]:
        attempt_ids = [attempt.attempt_id for attempt in self.attempt_lineage(attempt_id)]
        placeholders = ",".join("?" for _ in attempt_ids)
        rows = self._connection.execute(
            f"SELECT * FROM failures WHERE attempt_id IN ({placeholders}) ORDER BY occurred_at",
            attempt_ids,
        ).fetchall()
        return tuple(self._failure_from_row(row) for row in rows)

    @staticmethod
    def _validate_artifact_record(record: ArtifactRecord) -> tuple[Compression, ReleaseClass]:
        _normalise_hash("content_hash", record.content_hash)
        compression = Compression(record.compression)
        release = ReleaseClass(record.release_class)
        _nonnegative_int("raw_size_bytes", record.raw_size_bytes)
        _nonnegative_int("stored_size_bytes", record.stored_size_bytes)
        if not record.media_type or not record.relative_path:
            raise ValueError("artifact media type and relative path must be nonempty")
        relative = PurePosixPath(record.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact relative_path must remain inside the blob root")
        return compression, release

    def _register_artifact_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        record: ArtifactRecord,
    ) -> ArtifactRecord:
        compression, release = self._validate_artifact_record(record)
        existing = cursor.execute(
            "SELECT * FROM artifacts WHERE content_hash = ?", (record.content_hash,)
        ).fetchone()
        if existing is not None:
            expected = (
                compression.value,
                record.media_type,
                record.raw_size_bytes,
                record.stored_size_bytes,
                record.relative_path,
                release.value,
            )
            actual = tuple(
                existing[key]
                for key in (
                    "compression",
                    "media_type",
                    "raw_size_bytes",
                    "stored_size_bytes",
                    "relative_path",
                    "release_class",
                )
            )
            if actual != expected:
                raise DuplicateConflictError(
                    "content hash is already registered with different metadata or release class"
                )
            return self._artifact_from_row(existing)
        try:
            cursor.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.content_hash,
                    compression.value,
                    record.media_type,
                    record.raw_size_bytes,
                    record.stored_size_bytes,
                    record.relative_path,
                    release.value,
                    _normalise_timestamp(record.created_at),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateConflictError(
                "artifact path or hash conflicts with an existing row"
            ) from exc
        return record

    def register_artifact(self, record: ArtifactRecord) -> ArtifactRecord:
        with self._transaction() as cursor:
            return self._register_artifact_with_cursor(cursor, record)

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            content_hash=row["content_hash"],
            compression=Compression(row["compression"]),
            media_type=row["media_type"],
            raw_size_bytes=row["raw_size_bytes"],
            stored_size_bytes=row["stored_size_bytes"],
            relative_path=row["relative_path"],
            release_class=ReleaseClass(row["release_class"]),
            created_at=row["created_at"],
        )

    def get_artifact(
        self, content_hash: str, *, for_public_release: bool = False
    ) -> ArtifactRecord:
        _normalise_hash("content_hash", content_hash)
        row = self._connection.execute(
            "SELECT * FROM artifacts WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact {content_hash}")
        record = self._artifact_from_row(row)
        if for_public_release and record.release_class is ReleaseClass.RESTRICTED:
            raise ReleaseViolationError("restricted artifact cannot enter a public release")
        return record

    def link_artifact(
        self,
        *,
        job_id: str,
        content_hash: str,
        role: str,
        attempt_id: str | None = None,
        parent_content_hash: str | None = None,
        created_at: Any | None = None,
    ) -> str:
        if not role:
            raise ValueError("artifact role must be nonempty")
        _normalise_hash("content_hash", content_hash)
        if parent_content_hash is not None:
            _normalise_hash("parent_content_hash", parent_content_hash)
            if parent_content_hash == content_hash:
                raise ValueError("an artifact cannot be its own lineage parent")
        link_components = {
            "job_id": job_id,
            "content_hash": content_hash,
            "role": role,
            "attempt_id": attempt_id,
            "parent_content_hash": parent_content_hash,
        }
        link_id = sha256_bytes(canonical_json(link_components).encode("utf-8"))
        existing = self._connection.execute(
            "SELECT link_id FROM job_artifacts WHERE link_id = ?", (link_id,)
        ).fetchone()
        if existing is not None:
            return link_id
        timestamp = _normalise_timestamp(created_at)
        with self._transaction() as cursor:
            job = cursor.execute(
                "SELECT release_class FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"unknown job {job_id}")
            artifact = cursor.execute(
                "SELECT release_class FROM artifacts WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            if artifact is None:
                raise KeyError(f"unknown artifact {content_hash}")
            if (
                job["release_class"] == ReleaseClass.PUBLIC.value
                and artifact["release_class"] == ReleaseClass.RESTRICTED.value
            ):
                raise ReleaseViolationError("a public job cannot reference a restricted artifact")
            if attempt_id is not None:
                attempt = cursor.execute(
                    "SELECT job_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
                if attempt is None:
                    raise KeyError(f"unknown attempt {attempt_id}")
                if attempt["job_id"] != job_id:
                    raise ValueError("artifact attempt lineage cannot cross jobs")
            try:
                cursor.execute(
                    "INSERT INTO job_artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        link_id,
                        job_id,
                        content_hash,
                        role,
                        attempt_id,
                        parent_content_hash,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise KeyError("unknown parent artifact or conflicting artifact link") from exc
            return link_id

    @staticmethod
    def _gpu_allocation_journal_from_row(
        row: sqlite3.Row,
    ) -> GpuAllocationJournalRecord:
        return GpuAllocationJournalRecord(
            journal_id=row["journal_id"],
            allocation_id=row["allocation_id"],
            sequence=row["sequence"],
            state=GpuAllocationJournalState(row["state"]),
            intended_event_kind=GpuEventKind(row["intended_event_kind"]),
            elapsed_microseconds=row["elapsed_microseconds"],
            maximum_microseconds=row["maximum_microseconds"],
            observed_at=row["observed_at"],
            job_id=row["job_id"],
            attempt_id=row["attempt_id"],
            details_json=row["details_json"],
        )

    def record_gpu_allocation_observation(
        self,
        *,
        allocation_id: str,
        state: GpuAllocationJournalState,
        intended_event_kind: GpuEventKind,
        elapsed_seconds: Any,
        maximum_seconds: Any,
        observed_at: Any,
        job_id: str | None = None,
        attempt_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> GpuAllocationJournalRecord:
        """Append an open/heartbeat/close record before final event reconciliation."""

        _metadata_token("allocation_id", allocation_id)
        journal_state = GpuAllocationJournalState(state)
        intended = GpuEventKind(intended_event_kind)
        if intended is GpuEventKind.SERVICE_OVERHEAD:
            raise ValueError("service overhead uses service-session reconciliation")
        elapsed = _seconds_to_microseconds(elapsed_seconds)
        maximum = _seconds_to_microseconds(maximum_seconds)
        if maximum <= 0:
            raise ValueError("GPU allocation journal maximum must be positive")
        timestamp = _normalise_timestamp(observed_at)
        details_json = canonical_json(details or {})
        with self._transaction() as cursor:
            previous = cursor.execute(
                """SELECT * FROM gpu_allocation_journal
                   WHERE allocation_id = ? ORDER BY sequence DESC LIMIT 1""",
                (allocation_id,),
            ).fetchone()
            if previous is None:
                if journal_state is not GpuAllocationJournalState.OPENED or elapsed != 0:
                    raise ValueError("first GPU allocation observation must open at zero")
                sequence = 0
            else:
                previous_record = self._gpu_allocation_journal_from_row(previous)
                if previous_record.state in {
                    GpuAllocationJournalState.CLOSED,
                    GpuAllocationJournalState.RECOVERED,
                }:
                    raise DuplicateConflictError("GPU allocation journal is already terminal")
                if journal_state is GpuAllocationJournalState.OPENED:
                    raise DuplicateConflictError("GPU allocation cannot be opened twice")
                if intended is not previous_record.intended_event_kind:
                    raise DuplicateConflictError("GPU allocation event kind changed")
                if maximum != previous_record.maximum_microseconds:
                    raise DuplicateConflictError("GPU allocation maximum changed")
                if job_id != previous_record.job_id or attempt_id != previous_record.attempt_id:
                    raise DuplicateConflictError("GPU allocation lineage changed")
                if elapsed < previous_record.elapsed_microseconds:
                    raise ValueError("GPU allocation elapsed time regressed")
                if _parse_timestamp(timestamp) < _parse_timestamp(previous_record.observed_at):
                    raise ValueError("GPU allocation observation time regressed")
                sequence = previous_record.sequence + 1
            if job_id is not None:
                job = cursor.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if job is None:
                    raise KeyError(f"unknown job {job_id}")
            if attempt_id is not None:
                attempt = cursor.execute(
                    "SELECT job_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
                if attempt is None:
                    raise KeyError(f"unknown attempt {attempt_id}")
                if job_id is not None and attempt["job_id"] != job_id:
                    raise ValueError("GPU allocation attempt cannot cross jobs")
            contents = {
                "allocation_id": allocation_id,
                "sequence": sequence,
                "state": journal_state.value,
                "intended_event_kind": intended.value,
                "elapsed_microseconds": elapsed,
                "maximum_microseconds": maximum,
                "observed_at": timestamp,
                "job_id": job_id,
                "attempt_id": attempt_id,
                "details_json": details_json,
            }
            journal_id = sha256_bytes(canonical_json(contents).encode("utf-8"))
            cursor.execute(
                """INSERT INTO gpu_allocation_journal
                   (journal_id, allocation_id, sequence, state, intended_event_kind,
                    elapsed_microseconds, maximum_microseconds, observed_at, job_id,
                    attempt_id, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (journal_id, *contents.values()),
            )
            inserted = cursor.execute(
                "SELECT * FROM gpu_allocation_journal WHERE journal_id = ?",
                (journal_id,),
            ).fetchone()
            assert inserted is not None
            return self._gpu_allocation_journal_from_row(inserted)

    def unresolved_gpu_allocations(self) -> tuple[GpuAllocationJournalRecord, ...]:
        """Return latest nonterminal journals that have no reconciled GPU event."""

        rows = self._connection.execute(
            """SELECT journal.*
               FROM gpu_allocation_journal AS journal
               JOIN (
                   SELECT allocation_id, MAX(sequence) AS maximum_sequence
                   FROM gpu_allocation_journal GROUP BY allocation_id
               ) AS latest
                 ON latest.allocation_id = journal.allocation_id
                AND latest.maximum_sequence = journal.sequence
               LEFT JOIN gpu_events AS event ON event.event_id = journal.allocation_id
               WHERE event.event_id IS NULL
                 AND journal.state IN ('opened', 'heartbeat')
               ORDER BY journal.observed_at, journal.allocation_id"""
        ).fetchall()
        return tuple(self._gpu_allocation_journal_from_row(row) for row in rows)

    def gpu_allocation_journal_records(self) -> tuple[GpuAllocationJournalRecord, ...]:
        """Return the complete immutable allocation journal in logical row order."""

        rows = self._connection.execute(
            """SELECT * FROM gpu_allocation_journal
               ORDER BY allocation_id, sequence, journal_id"""
        ).fetchall()
        return tuple(self._gpu_allocation_journal_from_row(row) for row in rows)

    def recover_unclosed_gpu_allocations(self, *, recovered_at: Any) -> tuple[GpuEvent, ...]:
        """Conservatively charge crash-open intervals before any new allocation."""

        recovery_timestamp = _normalise_timestamp(recovered_at)
        recovery_time = _parse_timestamp(recovery_timestamp)
        recovered: list[GpuEvent] = []
        for latest in self.unresolved_gpu_allocations():
            opened_row = self._connection.execute(
                """SELECT * FROM gpu_allocation_journal
                   WHERE allocation_id = ? ORDER BY sequence LIMIT 1""",
                (latest.allocation_id,),
            ).fetchone()
            assert opened_row is not None
            opened = self._gpu_allocation_journal_from_row(opened_row)
            wall_microseconds = max(
                0,
                int(
                    decimal.Decimal(
                        str((recovery_time - _parse_timestamp(opened.observed_at)).total_seconds())
                    )
                    * _MICROSECONDS_PER_SECOND
                ),
            )
            charged_microseconds = min(
                latest.maximum_microseconds,
                max(latest.elapsed_microseconds, wall_microseconds),
            )
            event = self.record_gpu_event(
                event_id=latest.allocation_id,
                event_kind=GpuEventKind.FAILURE,
                allocated_seconds=decimal.Decimal(charged_microseconds) / _MICROSECONDS_PER_SECOND,
                started_at=opened.observed_at,
                ended_at=recovery_timestamp,
                succeeded=False,
                job_id=latest.job_id,
                attempt_id=latest.attempt_id,
                details={
                    "recovered_from_open_journal": True,
                    "intended_event_kind": latest.intended_event_kind.value,
                    "admitted_maximum_seconds": latest.maximum_microseconds / 1_000_000,
                },
            )
            self.record_gpu_allocation_observation(
                allocation_id=latest.allocation_id,
                state=GpuAllocationJournalState.RECOVERED,
                intended_event_kind=latest.intended_event_kind,
                elapsed_seconds=decimal.Decimal(charged_microseconds) / _MICROSECONDS_PER_SECOND,
                maximum_seconds=decimal.Decimal(latest.maximum_microseconds)
                / _MICROSECONDS_PER_SECOND,
                observed_at=recovery_timestamp,
                job_id=latest.job_id,
                attempt_id=latest.attempt_id,
                details={"event_id": event.event_id},
            )
            recovered.append(event)
        return tuple(recovered)

    def record_gpu_event(
        self,
        *,
        event_id: str,
        event_kind: GpuEventKind,
        allocated_seconds: Any,
        started_at: Any,
        ended_at: Any,
        succeeded: bool | None,
        job_id: str | None = None,
        attempt_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> GpuEvent:
        if not event_id:
            raise ValueError("event_id must be nonempty")
        kind = GpuEventKind(event_kind)
        if kind is GpuEventKind.SERVICE_OVERHEAD:
            raise ValueError("service_overhead is session-derived; use record_gpu_service_session")
        micros = _seconds_to_microseconds(allocated_seconds)
        start = _normalise_timestamp(started_at)
        end = _normalise_timestamp(ended_at)
        if _parse_timestamp(end) < _parse_timestamp(start):
            raise ValueError("GPU event end precedes its start")
        if succeeded is not None and not isinstance(succeeded, bool):
            raise ValueError("succeeded must be bool or None")
        if kind in {GpuEventKind.FAILURE, GpuEventKind.TIMEOUT} and succeeded is True:
            raise ValueError("failure and timeout intervals cannot be marked successful")
        details_json = canonical_json(details or {})
        if attempt_id is not None:
            attempt = self._connection.execute(
                "SELECT job_id FROM attempts WHERE attempt_id = ?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise KeyError(f"unknown attempt {attempt_id}")
            if job_id is not None and attempt["job_id"] != job_id:
                raise ValueError("GPU event attempt cannot cross jobs")
            if job_id is None:
                job_id = attempt["job_id"]
        elif job_id is not None:
            if (
                self._connection.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown job {job_id}")
        values = (
            kind.value,
            micros,
            start,
            end,
            None if succeeded is None else int(succeeded),
            job_id,
            attempt_id,
            details_json,
        )
        existing = self._connection.execute(
            "SELECT * FROM gpu_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing is not None:
            actual = tuple(
                existing[key]
                for key in (
                    "event_kind",
                    "allocated_microseconds",
                    "started_at",
                    "ended_at",
                    "succeeded",
                    "job_id",
                    "attempt_id",
                    "details_json",
                )
            )
            if actual != values:
                raise DuplicateConflictError("GPU event_id was reused with different content")
            return self._gpu_event_from_row(existing)
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT INTO gpu_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event_id, *values),
            )
        return GpuEvent(
            event_id=event_id,
            event_kind=kind,
            allocated_microseconds=micros,
            started_at=start,
            ended_at=end,
            succeeded=succeeded,
            job_id=job_id,
            attempt_id=attempt_id,
            details_json=details_json,
        )

    @staticmethod
    def _gpu_event_from_row(row: sqlite3.Row) -> GpuEvent:
        succeeded = None if row["succeeded"] is None else bool(row["succeeded"])
        return GpuEvent(
            event_id=row["event_id"],
            event_kind=GpuEventKind(row["event_kind"]),
            allocated_microseconds=row["allocated_microseconds"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            succeeded=succeeded,
            job_id=row["job_id"],
            attempt_id=row["attempt_id"],
            details_json=row["details_json"],
        )

    def gpu_events_with_prefix(self, event_id_prefix: str) -> tuple[GpuEvent, ...]:
        """Return an ordered, typed view of one runner's allocation events.

        Prefix filtering is performed in Python so ``%`` and ``_`` remain
        ordinary identifier characters rather than SQL pattern operators.
        The event table is intentionally small (the registered study has 286
        allocation events), making this both bounded and unambiguous.
        """

        if not event_id_prefix:
            raise ValueError("event_id_prefix must be nonempty")
        return tuple(
            event for event in self.gpu_events() if event.event_id.startswith(event_id_prefix)
        )

    def gpu_events(self) -> tuple[GpuEvent, ...]:
        """Return every immutable allocation event in deterministic ledger order."""

        rows = self._connection.execute(
            "SELECT * FROM gpu_events ORDER BY started_at, event_id"
        ).fetchall()
        return tuple(self._gpu_event_from_row(row) for row in rows)

    @staticmethod
    def _gpu_service_journal_from_row(
        row: sqlite3.Row,
    ) -> GpuServiceJournalRecord:
        return GpuServiceJournalRecord(
            journal_id=row["journal_id"],
            service_session_id=row["service_session_id"],
            sequence=row["sequence"],
            state=GpuServiceJournalState(row["state"]),
            session_id=row["session_id"],
            configuration_hash=row["configuration_hash"],
            service_started_at=row["service_started_at"],
            elapsed_microseconds=row["elapsed_microseconds"],
            ledger_allocated_microseconds_before_session=row[
                "ledger_allocated_microseconds_before_session"
            ],
            hard_limit_microseconds=row["hard_limit_microseconds"],
            observed_at=row["observed_at"],
            details_json=row["details_json"],
        )

    def record_gpu_service_observation(
        self,
        *,
        service_session_id: str,
        state: GpuServiceJournalState,
        session_id: str,
        configuration_hash: str,
        service_started_at: Any,
        elapsed_seconds: Any,
        ledger_allocated_seconds_before_session: Any,
        hard_limit_seconds: Any,
        observed_at: Any,
        details: Mapping[str, Any] | None = None,
    ) -> GpuServiceJournalRecord:
        """Append a crash-durable observation of one exclusive GPU service.

        A journal is opened before process creation and becomes terminal only
        after its immutable ``gpu_service_sessions`` accounting row exists.
        Stable identity and budget fields are repeated on every observation so
        a corrupt or mixed recovery cannot silently adopt another service.
        """

        _metadata_token("service_session_id", service_session_id)
        _metadata_token("session_id", session_id)
        configuration = _normalise_hash("configuration_hash", configuration_hash)
        journal_state = GpuServiceJournalState(state)
        started = _normalise_timestamp(service_started_at)
        elapsed = _seconds_to_microseconds(elapsed_seconds)
        baseline = _seconds_to_microseconds(ledger_allocated_seconds_before_session)
        hard_limit = _seconds_to_microseconds(hard_limit_seconds)
        if hard_limit <= 0:
            raise ValueError("GPU service journal hard limit must be positive")
        timestamp = _normalise_timestamp(observed_at)
        if _parse_timestamp(timestamp) < _parse_timestamp(started):
            raise ValueError("GPU service observation predates service start")
        details_json = canonical_json(details or {})
        with self._transaction() as cursor:
            previous = cursor.execute(
                """SELECT * FROM gpu_service_journal
                   WHERE service_session_id = ? ORDER BY sequence DESC LIMIT 1""",
                (service_session_id,),
            ).fetchone()
            if previous is None:
                unresolved_other = cursor.execute(
                    """SELECT latest.service_session_id
                       FROM gpu_service_journal AS latest
                       JOIN (
                           SELECT service_session_id, MAX(sequence) AS maximum_sequence
                           FROM gpu_service_journal GROUP BY service_session_id
                       ) AS terminal
                         ON terminal.service_session_id = latest.service_session_id
                        AND terminal.maximum_sequence = latest.sequence
                       WHERE latest.state NOT IN ('closed','recovered')
                         AND latest.service_session_id != ?
                       LIMIT 1""",
                    (service_session_id,),
                ).fetchone()
                if unresolved_other is not None:
                    raise DuplicateConflictError("another GPU service journal remains unresolved")
                if journal_state is not GpuServiceJournalState.OPENED or elapsed != 0:
                    raise ValueError("first GPU service observation must open at zero")
                if timestamp != started:
                    raise ValueError("opening GPU service observation must equal start time")
                sequence = 0
            else:
                previous_record = self._gpu_service_journal_from_row(previous)
                if previous_record.state in {
                    GpuServiceJournalState.CLOSED,
                    GpuServiceJournalState.RECOVERED,
                }:
                    raise DuplicateConflictError("GPU service journal is already terminal")
                stable_actual = (
                    previous_record.session_id,
                    previous_record.configuration_hash,
                    previous_record.service_started_at,
                    previous_record.ledger_allocated_microseconds_before_session,
                    previous_record.hard_limit_microseconds,
                )
                stable_expected = (session_id, configuration, started, baseline, hard_limit)
                if stable_actual != stable_expected:
                    raise DuplicateConflictError("GPU service journal identity changed")
                if journal_state is GpuServiceJournalState.OPENED:
                    raise DuplicateConflictError("GPU service journal cannot be opened twice")
                allowed_states = {
                    GpuServiceJournalState.OPENED: {
                        GpuServiceJournalState.HEARTBEAT,
                        GpuServiceJournalState.PROCESS_STOPPED,
                        GpuServiceJournalState.RECOVERED,
                    },
                    GpuServiceJournalState.HEARTBEAT: {
                        GpuServiceJournalState.HEARTBEAT,
                        GpuServiceJournalState.PROCESS_STOPPED,
                        GpuServiceJournalState.RECOVERED,
                    },
                    GpuServiceJournalState.PROCESS_STOPPED: {
                        GpuServiceJournalState.CLOSED,
                        GpuServiceJournalState.RECOVERED,
                    },
                }
                contradicted_stop_reverification = (
                    previous_record.state is GpuServiceJournalState.PROCESS_STOPPED
                    and journal_state is GpuServiceJournalState.PROCESS_STOPPED
                    and isinstance(details, Mapping)
                    and details.get("prior_stop_contradicted_by_exact_live_identity")
                    is True
                    and _parse_timestamp(timestamp)
                    > _parse_timestamp(previous_record.observed_at)
                    and elapsed > previous_record.elapsed_microseconds
                )
                if (
                    journal_state not in allowed_states[previous_record.state]
                    and not contradicted_stop_reverification
                ):
                    raise ValueError(
                        "invalid GPU service journal transition "
                        f"{previous_record.state.value}->{journal_state.value}"
                    )
                if elapsed < previous_record.elapsed_microseconds:
                    raise ValueError("GPU service elapsed time regressed")
                if _parse_timestamp(timestamp) < _parse_timestamp(previous_record.observed_at):
                    raise ValueError("GPU service observation time regressed")
                sequence = previous_record.sequence + 1
            contents = {
                "service_session_id": service_session_id,
                "sequence": sequence,
                "state": journal_state.value,
                "session_id": session_id,
                "configuration_hash": configuration,
                "service_started_at": started,
                "elapsed_microseconds": elapsed,
                "ledger_allocated_microseconds_before_session": baseline,
                "hard_limit_microseconds": hard_limit,
                "observed_at": timestamp,
                "details_json": details_json,
            }
            journal_id = sha256_bytes(canonical_json(contents).encode("utf-8"))
            cursor.execute(
                """INSERT INTO gpu_service_journal
                   (journal_id, service_session_id, sequence, state, session_id,
                    configuration_hash, service_started_at, elapsed_microseconds,
                    ledger_allocated_microseconds_before_session,
                    hard_limit_microseconds, observed_at, details_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (journal_id, *contents.values()),
            )
            inserted = cursor.execute(
                "SELECT * FROM gpu_service_journal WHERE journal_id = ?",
                (journal_id,),
            ).fetchone()
            assert inserted is not None
            return self._gpu_service_journal_from_row(inserted)

    def latest_gpu_service_journal(self, service_session_id: str) -> GpuServiceJournalRecord | None:
        _metadata_token("service_session_id", service_session_id)
        row = self._connection.execute(
            """SELECT * FROM gpu_service_journal
               WHERE service_session_id = ? ORDER BY sequence DESC LIMIT 1""",
            (service_session_id,),
        ).fetchone()
        return None if row is None else self._gpu_service_journal_from_row(row)

    def gpu_service_journal_records(self) -> tuple[GpuServiceJournalRecord, ...]:
        """Return the complete immutable service journal in logical row order."""

        rows = self._connection.execute(
            """SELECT * FROM gpu_service_journal
               ORDER BY service_session_id, sequence, journal_id"""
        ).fetchall()
        return tuple(self._gpu_service_journal_from_row(row) for row in rows)

    def unresolved_gpu_service_journals(self) -> tuple[GpuServiceJournalRecord, ...]:
        """Return latest nonterminal service journals in deterministic order."""

        rows = self._connection.execute(
            """SELECT journal.*
               FROM gpu_service_journal AS journal
               JOIN (
                   SELECT service_session_id, MAX(sequence) AS maximum_sequence
                   FROM gpu_service_journal GROUP BY service_session_id
               ) AS latest
                 ON latest.service_session_id = journal.service_session_id
                AND latest.maximum_sequence = journal.sequence
               WHERE journal.state NOT IN ('closed','recovered')
               ORDER BY journal.service_started_at, journal.service_session_id"""
        ).fetchall()
        return tuple(self._gpu_service_journal_from_row(row) for row in rows)

    def get_gpu_service_session(self, service_session_id: str) -> GpuServiceSession | None:
        _metadata_token("service_session_id", service_session_id)
        row = self._connection.execute(
            "SELECT * FROM gpu_service_sessions WHERE service_session_id = ?",
            (service_session_id,),
        ).fetchone()
        return None if row is None else self._gpu_service_session_from_row(row)

    def gpu_service_sessions(self) -> tuple[GpuServiceSession, ...]:
        """Return all terminal service-accounting rows ordered by stable identity."""

        rows = self._connection.execute(
            "SELECT * FROM gpu_service_sessions ORDER BY service_session_id"
        ).fetchall()
        return tuple(self._gpu_service_session_from_row(row) for row in rows)

    def recover_gpu_service_journal(
        self,
        *,
        service_session_id: str,
        recovered_at: Any,
        details: Mapping[str, Any] | None = None,
    ) -> GpuServiceSession:
        """Conservatively close one service after verified process/endpoint absence.

        The caller owns OS-level absence checks.  A live/open journal is charged
        through recovery time; a durably ``process_stopped`` journal uses its
        earlier verified stop observation.  Replaying after a crash between the
        accounting row and terminal journal observation is idempotent.
        """

        latest = self.latest_gpu_service_journal(service_session_id)
        if latest is None:
            raise KeyError(f"unknown GPU service journal {service_session_id}")
        if latest.state in {
            GpuServiceJournalState.CLOSED,
            GpuServiceJournalState.RECOVERED,
        }:
            existing_terminal = self.get_gpu_service_session(service_session_id)
            if existing_terminal is None:
                raise StoreError("terminal GPU service journal has no accounting row")
            return existing_terminal
        recovery_timestamp = _normalise_timestamp(recovered_at)
        recovery_time = _parse_timestamp(recovery_timestamp)
        if latest.state is GpuServiceJournalState.PROCESS_STOPPED:
            ended_at = latest.observed_at
            elapsed_micros = latest.elapsed_microseconds
        else:
            if recovery_time < _parse_timestamp(latest.observed_at):
                raise ValueError("GPU service recovery time regressed")
            ended_at = recovery_timestamp
            wall_seconds = (
                recovery_time - _parse_timestamp(latest.service_started_at)
            ).total_seconds()
            wall_micros = max(
                0,
                int(decimal.Decimal(str(wall_seconds)) * _MICROSECONDS_PER_SECOND),
            )
            elapsed_micros = max(latest.elapsed_microseconds, wall_micros)

        existing = self.get_gpu_service_session(service_session_id)
        if existing is None:
            current_micros = self.gpu_summary().total_allocated_microseconds
            baseline_micros = latest.ledger_allocated_microseconds_before_session
            if current_micros < baseline_micros:
                raise StoreError("GPU ledger total predates service journal baseline")
            classified_micros = current_micros - baseline_micros
            service_micros = max(elapsed_micros, classified_micros)
            recovery_details = dict(details or {})
            recovery_details.update(
                {
                    "accounting_method": "conservative_service_journal_recovery",
                    "configuration_hash": latest.configuration_hash,
                    "latest_journal_state": latest.state.value,
                    "recovered_from_open_service_journal": True,
                }
            )
            existing = self.record_gpu_service_session(
                service_session_id=service_session_id,
                session_id=latest.session_id,
                service_seconds=decimal.Decimal(service_micros) / _MICROSECONDS_PER_SECOND,
                classified_event_seconds=decimal.Decimal(classified_micros)
                / _MICROSECONDS_PER_SECOND,
                started_at=latest.service_started_at,
                ended_at=ended_at,
                details=recovery_details,
            )
        self.record_gpu_service_observation(
            service_session_id=service_session_id,
            state=GpuServiceJournalState.RECOVERED,
            session_id=latest.session_id,
            configuration_hash=latest.configuration_hash,
            service_started_at=latest.service_started_at,
            elapsed_seconds=decimal.Decimal(existing.service_microseconds)
            / _MICROSECONDS_PER_SECOND,
            ledger_allocated_seconds_before_session=decimal.Decimal(
                latest.ledger_allocated_microseconds_before_session
            )
            / _MICROSECONDS_PER_SECOND,
            hard_limit_seconds=decimal.Decimal(latest.hard_limit_microseconds)
            / _MICROSECONDS_PER_SECOND,
            observed_at=max(
                _parse_timestamp(recovery_timestamp),
                _parse_timestamp(latest.observed_at),
            ),
            details={"accounting_row_created": True},
        )
        return existing

    def close_gpu_service_journal(
        self,
        *,
        service_session_id: str,
        session_id: str,
        service_seconds: Any,
        classified_event_seconds: Any,
        started_at: Any,
        ended_at: Any,
        details: Mapping[str, Any] | None = None,
    ) -> GpuServiceSession:
        """Create the accounting row, then durably terminalize a normal stop.

        A replay after a crash between those two append-only writes is safe: the
        immutable accounting insert is idempotent and the still-pending journal
        receives its terminal observation on the second call.
        """

        latest = self.latest_gpu_service_journal(service_session_id)
        if latest is None:
            raise KeyError(f"unknown GPU service journal {service_session_id}")
        if latest.state is GpuServiceJournalState.RECOVERED:
            raise DuplicateConflictError("a recovered GPU service cannot close normally")
        if latest.state is GpuServiceJournalState.CLOSED:
            return self.record_gpu_service_session(
                service_session_id=service_session_id,
                session_id=session_id,
                service_seconds=service_seconds,
                classified_event_seconds=classified_event_seconds,
                started_at=started_at,
                ended_at=ended_at,
                details=details,
            )
        if latest.state is not GpuServiceJournalState.PROCESS_STOPPED:
            raise ValueError("GPU service must be durably process-stopped before close")
        normalized_start = _normalise_timestamp(started_at)
        normalized_end = _normalise_timestamp(ended_at)
        if session_id != latest.session_id or normalized_start != latest.service_started_at:
            raise DuplicateConflictError("GPU service close identity changed")
        if normalized_end != latest.observed_at:
            raise DuplicateConflictError("GPU service close end differs from verified stop")
        service_micros = _seconds_to_microseconds(service_seconds)
        if service_micros < latest.elapsed_microseconds:
            raise ValueError("GPU service close time is below the stopped observation")
        record = self.record_gpu_service_session(
            service_session_id=service_session_id,
            session_id=session_id,
            service_seconds=service_seconds,
            classified_event_seconds=classified_event_seconds,
            started_at=normalized_start,
            ended_at=normalized_end,
            details=details,
        )
        self.record_gpu_service_observation(
            service_session_id=service_session_id,
            state=GpuServiceJournalState.CLOSED,
            session_id=latest.session_id,
            configuration_hash=latest.configuration_hash,
            service_started_at=latest.service_started_at,
            elapsed_seconds=decimal.Decimal(record.service_microseconds) / _MICROSECONDS_PER_SECOND,
            ledger_allocated_seconds_before_session=decimal.Decimal(
                latest.ledger_allocated_microseconds_before_session
            )
            / _MICROSECONDS_PER_SECOND,
            hard_limit_seconds=decimal.Decimal(latest.hard_limit_microseconds)
            / _MICROSECONDS_PER_SECOND,
            observed_at=normalized_end,
            details={"accounting_row_created": True},
        )
        return record

    def record_gpu_service_session(
        self,
        *,
        service_session_id: str,
        session_id: str,
        service_seconds: Any,
        classified_event_seconds: Any,
        started_at: Any,
        ended_at: Any,
        details: Mapping[str, Any] | None = None,
    ) -> GpuServiceSession:
        """Persist only the unclassified complement of a service session.

        Classified load/inference/repair/etc. intervals already live in
        ``gpu_events``. The session row stores the complete service duration and
        its classified subtotal, while :meth:`gpu_summary` adds only their
        difference. Thus allocation between calls survives controller restarts
        without recording an overlapping enclosing GPU event.
        """

        if not service_session_id or not session_id:
            raise ValueError("GPU service session identifiers must be nonempty")
        service_micros = _seconds_to_microseconds(service_seconds)
        classified_micros = _seconds_to_microseconds(classified_event_seconds)
        # Each classified interval and the encompassing monotonic service duration
        # are rounded independently to integer microseconds.  A valid set of
        # nested intervals can therefore exceed the rounded service total by a
        # few microseconds. Never lose already-accounted work or fail shutdown:
        # conservatively promote the durable service total to that subtotal.
        service_micros = max(service_micros, classified_micros)
        overhead_micros = service_micros - classified_micros
        start = _normalise_timestamp(started_at)
        end = _normalise_timestamp(ended_at)
        if _parse_timestamp(end) < _parse_timestamp(start):
            raise ValueError("GPU service session end precedes its start")
        details_json = canonical_json(details or {})
        values = (
            session_id,
            service_micros,
            classified_micros,
            overhead_micros,
            start,
            end,
            details_json,
        )
        existing = self._connection.execute(
            "SELECT * FROM gpu_service_sessions WHERE service_session_id = ?",
            (service_session_id,),
        ).fetchone()
        if existing is not None:
            actual = tuple(
                existing[key]
                for key in (
                    "session_id",
                    "service_microseconds",
                    "classified_event_microseconds",
                    "overhead_microseconds",
                    "started_at",
                    "ended_at",
                    "details_json",
                )
            )
            if actual != values:
                raise DuplicateConflictError(
                    "GPU service_session_id was reused with different content"
                )
            return self._gpu_service_session_from_row(existing)
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT INTO gpu_service_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (service_session_id, *values),
            )
        return GpuServiceSession(
            service_session_id=service_session_id,
            session_id=session_id,
            service_microseconds=service_micros,
            classified_event_microseconds=classified_micros,
            overhead_microseconds=overhead_micros,
            started_at=start,
            ended_at=end,
            details_json=details_json,
        )

    @staticmethod
    def _gpu_service_session_from_row(row: sqlite3.Row) -> GpuServiceSession:
        return GpuServiceSession(
            service_session_id=row["service_session_id"],
            session_id=row["session_id"],
            service_microseconds=row["service_microseconds"],
            classified_event_microseconds=row["classified_event_microseconds"],
            overhead_microseconds=row["overhead_microseconds"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            details_json=row["details_json"],
        )

    def gpu_summary(self) -> GpuSummary:
        rows = self._connection.execute(
            """SELECT event_kind, SUM(allocated_microseconds) AS total, COUNT(*) AS count
               FROM gpu_events GROUP BY event_kind ORDER BY event_kind"""
        ).fetchall()
        by_kind_values = {GpuEventKind(row["event_kind"]): int(row["total"]) for row in rows}
        service_row = self._connection.execute(
            """SELECT COALESCE(SUM(overhead_microseconds), 0) AS total, COUNT(*) AS count
               FROM gpu_service_sessions"""
        ).fetchone()
        service_overhead = int(service_row["total"])
        if service_overhead:
            by_kind_values[GpuEventKind.SERVICE_OVERHEAD] = (
                by_kind_values.get(GpuEventKind.SERVICE_OVERHEAD, 0) + service_overhead
            )
        by_kind = tuple(sorted(by_kind_values.items(), key=lambda item: item[0].value))
        return GpuSummary(
            total_allocated_microseconds=sum(value for _, value in by_kind),
            event_count=sum(int(row["count"]) for row in rows),
            by_kind_microseconds=by_kind,
            service_session_count=int(service_row["count"]),
        )

    def require_gpu_capacity(
        self,
        planned_next_seconds: Any,
        *,
        hard_limit_seconds: Any = DEFAULT_GPU_HARD_LIMIT_SECONDS,
    ) -> None:
        planned = _seconds_to_microseconds(planned_next_seconds)
        hard_limit = _seconds_to_microseconds(hard_limit_seconds)
        used = self.gpu_summary().total_allocated_microseconds
        # The plan requires shutdown *before* 10.00 actual hours.
        if used + planned >= hard_limit:
            raise GpuBudgetExceeded(
                f"GPU allocation would reach/cross hard limit: used={used / 1e6:.6f}s, "
                f"next={planned / 1e6:.6f}s, hard={hard_limit / 1e6:.6f}s"
            )

    def record_storage_sample(
        self,
        report: StorageReport,
        *,
        phase: str,
        sampled_at: Any | None = None,
    ) -> str:
        if not phase:
            raise ValueError("phase must be nonempty")
        timestamp = _normalise_timestamp(sampled_at)
        contents = {
            "phase": phase,
            "sampled_at": timestamp,
            "current": report.current_occupied_bytes,
            "additional": report.additional_reserved_bytes,
            "projected": report.projected_occupied_bytes,
            "filesystem_free": report.filesystem_free_bytes,
            "headroom": report.effective_projected_headroom_bytes,
            "allowed": report.allowed,
            "violations": report.violations,
        }
        sample_id = sha256_bytes(canonical_json(contents).encode("utf-8"))
        existing = self._connection.execute(
            "SELECT 1 FROM storage_samples WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        if existing is not None:
            return sample_id
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT INTO storage_samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sample_id,
                    phase,
                    timestamp,
                    report.current_occupied_bytes,
                    report.additional_reserved_bytes,
                    report.projected_occupied_bytes,
                    report.filesystem_free_bytes,
                    report.effective_projected_headroom_bytes,
                    int(report.allowed),
                    canonical_json(report.violations),
                ),
            )
        return sample_id

    def storage_samples_with_phase_prefix(
        self, phase_prefix: str
    ) -> tuple[StorageSampleRecord, ...]:
        """Read immutable storage observations for a run or phase prefix."""

        if not phase_prefix:
            raise ValueError("phase_prefix must be nonempty")
        rows = self._connection.execute(
            "SELECT * FROM storage_samples ORDER BY sampled_at, sample_id"
        ).fetchall()
        records: list[StorageSampleRecord] = []
        for row in rows:
            if not row["phase"].startswith(phase_prefix):
                continue
            violations = json.loads(row["violations_json"])
            if not isinstance(violations, list) or not all(
                isinstance(item, str) for item in violations
            ):
                raise ArtifactIntegrityError("storage sample contains invalid violation metadata")
            records.append(
                StorageSampleRecord(
                    sample_id=row["sample_id"],
                    phase=row["phase"],
                    sampled_at=row["sampled_at"],
                    current_occupied_bytes=row["current_occupied_bytes"],
                    additional_reserved_bytes=row["additional_reserved_bytes"],
                    projected_occupied_bytes=row["projected_occupied_bytes"],
                    filesystem_free_bytes=row["filesystem_free_bytes"],
                    effective_projected_headroom_bytes=row["effective_projected_headroom_bytes"],
                    allowed=bool(row["allowed"]),
                    violations=tuple(violations),
                )
            )
        return tuple(records)

    def storage_samples(self) -> tuple[StorageSampleRecord, ...]:
        """Return all immutable storage preflights without SQL-pattern filtering."""

        rows = self._connection.execute(
            "SELECT DISTINCT phase FROM storage_samples ORDER BY phase"
        ).fetchall()
        return tuple(
            sample
            for row in rows
            for sample in self.storage_samples_with_phase_prefix(row["phase"])
            if sample.phase == row["phase"]
        )

    def count_rows(self, table: str) -> int:
        """Return a test/verification count for a known append-only table."""

        if table not in self._APPEND_ONLY_TABLES:
            raise ValueError("unknown or non-scientific table")
        row = self._connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"])


def _validate_artifact_record_metadata(record: ArtifactRecord) -> None:
    """Reject malformed ledger metadata before it is used as a filesystem capability."""

    try:
        digest = _normalise_hash("content_hash", record.content_hash)
        compression = Compression(record.compression)
        release = ReleaseClass(record.release_class)
        raw_size = _nonnegative_int("raw_size_bytes", record.raw_size_bytes)
        stored_size = _nonnegative_int("stored_size_bytes", record.stored_size_bytes)
        created_at = _normalise_timestamp(record.created_at)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("artifact ledger metadata is invalid") from exc
    if not isinstance(record.media_type, str) or not record.media_type:
        raise ArtifactIntegrityError("artifact media type is invalid")
    if any(character in record.media_type for character in ("\n", "\r", "\x00")):
        raise ArtifactIntegrityError("artifact media type is invalid")
    if not isinstance(record.created_at, str) or created_at != record.created_at:
        raise ArtifactIntegrityError("artifact creation timestamp is not canonical")
    if stored_size == 0:
        raise ArtifactIntegrityError("artifact stored size must be positive")
    if release is not record.release_class or compression is not record.compression:
        raise ArtifactIntegrityError("artifact enum metadata is not canonical")
    suffix = ".jsonl.zst" if compression is Compression.ZSTD else ".jsonl.gz"
    if not isinstance(record.relative_path, str):
        raise ArtifactIntegrityError("artifact path is invalid")
    expected_path = PurePosixPath(digest[:2], digest + suffix).as_posix()
    relative = PurePosixPath(record.relative_path)
    if (
        not record.relative_path
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != record.relative_path
        or record.relative_path != expected_path
    ):
        raise ArtifactIntegrityError("artifact path does not match its content address")
    if raw_size != record.raw_size_bytes or stored_size != record.stored_size_bytes:
        raise ArtifactIntegrityError("artifact size metadata is not canonical")


class ReadOnlyBlobStore:
    """Strictly read-only view of a fixed-codec content-addressed blob directory.

    Construction never creates a directory. Reads walk the CAS with ``O_NOFOLLOW``
    and validate every byte against the immutable ledger record.
    """

    def __init__(
        self,
        root: Path,
        *,
        compression: Compression = Compression.ZSTD,
        max_raw_bytes: int = DEFAULT_MAX_BLOB_RAW_BYTES,
    ) -> None:
        supplied_root = Path(root)
        if supplied_root.is_symlink():
            raise ArtifactIntegrityError("blob root cannot be a symbolic link")
        try:
            resolved_root = supplied_root.resolve(strict=True)
        except OSError as exc:
            raise ArtifactIntegrityError("blob root does not exist") from exc
        if not resolved_root.is_dir():
            raise ArtifactIntegrityError("blob root must be a directory")
        self.root = resolved_root
        self.compression = Compression(compression)
        self.max_raw_bytes = _nonnegative_int("max_raw_bytes", max_raw_bytes)
        if self.max_raw_bytes == 0:
            raise ValueError("max_raw_bytes must be positive")
        self._zstandard = (
            _import_zstandard() if self.compression is Compression.ZSTD else None
        )

    @property
    def suffix(self) -> str:
        return ".jsonl.zst" if self.compression is Compression.ZSTD else ".jsonl.gz"

    @staticmethod
    def _open_flags(*, directory: bool) -> int:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        return flags

    def _read_encoded(self, relative: PurePosixPath, *, expected_size: int) -> bytes:
        """Read one regular CAS file without following any path-component symlink."""

        descriptors: list[int] = []
        try:
            current = os.open(self.root, self._open_flags(directory=True))
            descriptors.append(current)
            for component in relative.parts[:-1]:
                current = os.open(
                    component,
                    self._open_flags(directory=True),
                    dir_fd=current,
                )
                descriptors.append(current)
            artifact_fd = os.open(
                relative.parts[-1],
                self._open_flags(directory=False),
                dir_fd=current,
            )
            descriptors.append(artifact_fd)
            metadata = os.fstat(artifact_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactIntegrityError("artifact path is not a regular file")
            if metadata.st_size != expected_size:
                raise ArtifactIntegrityError(
                    "artifact stored size does not match ledger metadata"
                )
            with os.fdopen(os.dup(artifact_fd), "rb") as stream:
                encoded = stream.read(expected_size + 1)
            if len(encoded) != expected_size:
                raise ArtifactIntegrityError(
                    "artifact stored size does not match ledger metadata"
                )
            return encoded
        except ArtifactIntegrityError:
            raise
        except OSError as exc:
            raise ArtifactIntegrityError("cannot safely open artifact path") from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _decompress_bounded(self, encoded: bytes) -> bytes:
        try:
            if self.compression is Compression.GZIP:
                with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode="rb") as stream:
                    return stream.read(self.max_raw_bytes + 1)
            assert self._zstandard is not None
            return self._zstandard.ZstdDecompressor().decompress(
                encoded,
                max_output_size=self.max_raw_bytes + 1,
            )
        except Exception as exc:
            raise ArtifactIntegrityError("cannot decode stored artifact") from exc

    def read_bytes(self, record: ArtifactRecord, *, allow_restricted: bool = False) -> bytes:
        _validate_artifact_record_metadata(record)
        if record.compression is not self.compression:
            raise ArtifactIntegrityError(
                f"record codec {record.compression.value} does not match store codec "
                f"{self.compression.value}"
            )
        if record.raw_size_bytes > self.max_raw_bytes:
            raise BlobTooLargeError(
                f"artifact declares {record.raw_size_bytes} raw bytes; "
                f"bound is {self.max_raw_bytes} bytes"
            )
        if record.release_class is ReleaseClass.RESTRICTED and not allow_restricted:
            raise ReleaseViolationError("restricted artifact requires explicit restricted access")
        relative = PurePosixPath(record.relative_path)
        encoded = self._read_encoded(relative, expected_size=record.stored_size_bytes)
        decoded = self._decompress_bounded(encoded)
        if len(decoded) != record.raw_size_bytes:
            raise ArtifactIntegrityError("artifact raw size does not match ledger metadata")
        if sha256_bytes(decoded) != record.content_hash:
            raise ArtifactIntegrityError("artifact content hash verification failed")
        return decoded


class ReadOnlyLedger:
    """Typed query view opened with SQLite ``mode=ro`` and ``immutable=1``.

    This is intentionally a small verification surface, not a writable Ledger
    subtype. It assumes a closed/checkpointed database and refuses a nonempty
    rollback journal or WAL rather than making SQLite recover either sidecar.

    ``allow_live_wal`` is an explicit exception for control-plane observation
    while the one bound writer still owns a WAL database.  It uses SQLite's
    normal read-only locking protocol and pins one coherent read transaction;
    it never makes the immutable reader silently accept an uncheckpointed WAL.
    """

    def __init__(self, path: Path, *, allow_live_wal: bool = False) -> None:
        supplied_path = Path(path)
        if supplied_path.is_symlink():
            raise ArtifactIntegrityError("read-only ledger cannot be a symbolic link")
        try:
            resolved_path = supplied_path.resolve(strict=True)
        except OSError as exc:
            raise ArtifactIntegrityError("read-only ledger does not exist") from exc
        if not resolved_path.is_file():
            raise ArtifactIntegrityError("read-only ledger must be a regular file")
        for suffix in ("-wal", "-journal"):
            sidecar = Path(str(resolved_path) + suffix)
            if sidecar.is_symlink():
                raise ArtifactIntegrityError("SQLite sidecar cannot be a symbolic link")
            if sidecar.exists() and (
                not sidecar.is_file() or sidecar.stat().st_size
            ):
                if suffix == "-wal" and allow_live_wal and sidecar.is_file():
                    continue
                raise ArtifactIntegrityError(
                    "read-only ledger must be closed and checkpointed"
                )
        if allow_live_wal:
            wal = Path(str(resolved_path) + "-wal")
            shared_memory = Path(str(resolved_path) + "-shm")
            if shared_memory.is_symlink():
                raise ArtifactIntegrityError("SQLite sidecar cannot be a symbolic link")
            if wal.exists() and wal.stat().st_size and (
                not shared_memory.exists() or not shared_memory.is_file()
            ):
                raise ArtifactIntegrityError(
                    "live read-only ledger requires its regular SQLite shared-memory sidecar"
                )
        self.path = resolved_path
        self.live_wal_snapshot = allow_live_wal
        self._closed = False
        uri = self.path.as_uri() + (
            "?mode=ro" if allow_live_wal else "?mode=ro&immutable=1"
        )
        try:
            self._connection = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                timeout=30,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only = ON")
            if allow_live_wal:
                # Keep every status query on one SQLite-coordinated snapshot.
                # The read transaction participates in WAL locking but cannot
                # write ledger rows through this mode=ro connection.
                self._connection.execute("BEGIN")
            versions = self._connection.execute(
                "SELECT schema_version FROM schema_metadata ORDER BY schema_version"
            ).fetchall()
            schema_versions = tuple(int(row["schema_version"]) for row in versions)
            if (
                not schema_versions
                or schema_versions != tuple(sorted(set(schema_versions)))
                or schema_versions[-1] != SCHEMA_VERSION
            ):
                raise ArtifactIntegrityError("read-only ledger schema version is unsupported")
        except (sqlite3.Error, ArtifactIntegrityError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            if isinstance(exc, ArtifactIntegrityError):
                raise
            raise ArtifactIntegrityError("cannot open read-only ledger") from exc

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> ReadOnlyLedger:
        if self._closed:
            raise StoreError("read-only ledger is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _execute(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Cursor:
        if self._closed:
            raise StoreError("read-only ledger is closed")
        return self._connection.execute(sql, parameters)

    @staticmethod
    def _convert_artifact(row: sqlite3.Row) -> ArtifactRecord:
        try:
            record = Ledger._artifact_from_row(row)
            _validate_artifact_record_metadata(record)
            return record
        except ArtifactIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("artifact ledger metadata is invalid") from exc

    def get_artifact(
        self, content_hash: str, *, for_public_release: bool = False
    ) -> ArtifactRecord:
        try:
            digest = _normalise_hash("content_hash", content_hash)
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("requested artifact hash is invalid") from exc
        row = self._execute(
            "SELECT * FROM artifacts WHERE content_hash = ?", (digest,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact {digest}")
        record = self._convert_artifact(row)
        if for_public_release and record.release_class is ReleaseClass.RESTRICTED:
            raise ReleaseViolationError("restricted artifact cannot enter a public release")
        return record

    def gpu_events(self) -> tuple[GpuEvent, ...]:
        rows = self._execute(
            "SELECT * FROM gpu_events ORDER BY started_at, event_id"
        ).fetchall()
        try:
            return tuple(Ledger._gpu_event_from_row(row) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU event metadata is invalid") from exc

    def gpu_allocation_journal_records(self) -> tuple[GpuAllocationJournalRecord, ...]:
        rows = self._execute(
            """SELECT * FROM gpu_allocation_journal
               ORDER BY allocation_id, sequence, journal_id"""
        ).fetchall()
        try:
            return tuple(Ledger._gpu_allocation_journal_from_row(row) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU allocation journal metadata is invalid") from exc

    def unresolved_gpu_allocations(self) -> tuple[GpuAllocationJournalRecord, ...]:
        rows = self._execute(
            """SELECT journal.*
               FROM gpu_allocation_journal AS journal
               JOIN (
                   SELECT allocation_id, MAX(sequence) AS maximum_sequence
                   FROM gpu_allocation_journal GROUP BY allocation_id
               ) AS latest
                 ON latest.allocation_id = journal.allocation_id
                AND latest.maximum_sequence = journal.sequence
               LEFT JOIN gpu_events AS event ON event.event_id = journal.allocation_id
               WHERE event.event_id IS NULL
                 AND journal.state IN ('opened', 'heartbeat')
               ORDER BY journal.observed_at, journal.allocation_id"""
        ).fetchall()
        try:
            return tuple(Ledger._gpu_allocation_journal_from_row(row) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU allocation journal metadata is invalid") from exc

    def gpu_service_journal_records(self) -> tuple[GpuServiceJournalRecord, ...]:
        rows = self._execute(
            """SELECT * FROM gpu_service_journal
               ORDER BY service_session_id, sequence, journal_id"""
        ).fetchall()
        try:
            return tuple(Ledger._gpu_service_journal_from_row(row) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU service journal metadata is invalid") from exc

    def unresolved_gpu_service_journals(self) -> tuple[GpuServiceJournalRecord, ...]:
        rows = self._execute(
            """SELECT journal.*
               FROM gpu_service_journal AS journal
               JOIN (
                   SELECT service_session_id, MAX(sequence) AS maximum_sequence
                   FROM gpu_service_journal GROUP BY service_session_id
               ) AS latest
                 ON latest.service_session_id = journal.service_session_id
                AND latest.maximum_sequence = journal.sequence
               WHERE journal.state NOT IN ('closed','recovered')
               ORDER BY journal.service_started_at, journal.service_session_id"""
        ).fetchall()
        try:
            return tuple(Ledger._gpu_service_journal_from_row(row) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU service journal metadata is invalid") from exc

    def gpu_service_sessions(self) -> tuple[GpuServiceSession, ...]:
        rows = self._execute(
            "SELECT * FROM gpu_service_sessions ORDER BY service_session_id"
        ).fetchall()
        try:
            return tuple(Ledger._gpu_service_session_from_row(row) for row in rows)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU service session metadata is invalid") from exc

    def gpu_summary(self) -> GpuSummary:
        rows = self._execute(
            """SELECT event_kind, SUM(allocated_microseconds) AS total, COUNT(*) AS count
               FROM gpu_events GROUP BY event_kind ORDER BY event_kind"""
        ).fetchall()
        try:
            by_kind_values = {
                GpuEventKind(row["event_kind"]): int(row["total"]) for row in rows
            }
            service_row = self._execute(
                """SELECT COALESCE(SUM(overhead_microseconds), 0) AS total,
                          COUNT(*) AS count
                   FROM gpu_service_sessions"""
            ).fetchone()
            assert service_row is not None
            service_overhead = int(service_row["total"])
            if service_overhead:
                by_kind_values[GpuEventKind.SERVICE_OVERHEAD] = (
                    by_kind_values.get(GpuEventKind.SERVICE_OVERHEAD, 0)
                    + service_overhead
                )
            by_kind = tuple(sorted(by_kind_values.items(), key=lambda item: item[0].value))
            return GpuSummary(
                total_allocated_microseconds=sum(value for _, value in by_kind),
                event_count=sum(int(row["count"]) for row in rows),
                by_kind_microseconds=by_kind,
                service_session_count=int(service_row["count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("GPU accounting metadata is invalid") from exc

    def storage_samples_with_phase_prefix(
        self, phase_prefix: str
    ) -> tuple[StorageSampleRecord, ...]:
        if not phase_prefix:
            raise ValueError("phase_prefix must be nonempty")
        rows = self._execute(
            "SELECT * FROM storage_samples ORDER BY sampled_at, sample_id"
        ).fetchall()
        records: list[StorageSampleRecord] = []
        for row in rows:
            if not row["phase"].startswith(phase_prefix):
                continue
            try:
                violations = json.loads(row["violations_json"])
                if not isinstance(violations, list) or not all(
                    isinstance(item, str) for item in violations
                ):
                    raise ValueError
                records.append(
                    StorageSampleRecord(
                        sample_id=row["sample_id"],
                        phase=row["phase"],
                        sampled_at=row["sampled_at"],
                        current_occupied_bytes=row["current_occupied_bytes"],
                        additional_reserved_bytes=row["additional_reserved_bytes"],
                        projected_occupied_bytes=row["projected_occupied_bytes"],
                        filesystem_free_bytes=row["filesystem_free_bytes"],
                        effective_projected_headroom_bytes=row[
                            "effective_projected_headroom_bytes"
                        ],
                        allowed=bool(row["allowed"]),
                        violations=tuple(violations),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ArtifactIntegrityError("storage sample metadata is invalid") from exc
        return tuple(records)

    def storage_samples(self) -> tuple[StorageSampleRecord, ...]:
        rows = self._execute(
            "SELECT DISTINCT phase FROM storage_samples ORDER BY phase"
        ).fetchall()
        return tuple(
            sample
            for row in rows
            for sample in self.storage_samples_with_phase_prefix(row["phase"])
            if sample.phase == row["phase"]
        )

    def count_rows(self, table: str) -> int:
        if table not in Ledger._APPEND_ONLY_TABLES:
            raise ValueError("unknown or non-scientific table")
        row = self._execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        assert row is not None
        return int(row["count"])


class ReadOnlyArtifactStore:
    """ArtifactStore-compatible read surface with explicit ownership semantics."""

    def __init__(self, blobs: ReadOnlyBlobStore, ledger: ReadOnlyLedger) -> None:
        self.blobs = blobs
        self.ledger = ledger
        self._owns_ledger = False

    @classmethod
    def from_paths(
        cls,
        *,
        blob_root: Path,
        ledger_path: Path,
        compression: Compression = Compression.ZSTD,
        max_raw_bytes: int = DEFAULT_MAX_BLOB_RAW_BYTES,
    ) -> ReadOnlyArtifactStore:
        ledger = ReadOnlyLedger(ledger_path)
        try:
            blobs = ReadOnlyBlobStore(
                blob_root,
                compression=compression,
                max_raw_bytes=max_raw_bytes,
            )
        except Exception:
            ledger.close()
            raise
        instance = cls(blobs, ledger)
        instance._owns_ledger = True
        return instance

    def close(self) -> None:
        if self._owns_ledger:
            self.ledger.close()

    def __enter__(self) -> ReadOnlyArtifactStore:
        if self.ledger.closed:
            raise StoreError("read-only artifact store is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class ArtifactStore:
    """Coordinates atomic CAS writes with append-only artifact registration."""

    def __init__(self, blobs: BlobStore, ledger: Ledger) -> None:
        self.blobs = blobs
        self.ledger = ledger

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
        release_class: ReleaseClass,
        created_at: Any | None = None,
    ) -> ArtifactRecord:
        pending = self.blobs.put_bytes(
            payload,
            media_type=media_type,
            release_class=release_class,
            created_at=created_at,
        )
        # If a process stopped after the atomic rename but before this insert,
        # replay reaches the same hash/path and safely completes registration.
        return self.ledger.register_artifact(pending)


class StoragePreflight:
    """Measure and enforce the 30/25/5 GB writable-storage envelope."""

    def __init__(
        self,
        quota_root: Path,
        *,
        controlled_paths: Sequence[Path] | None = None,
        budget: StorageBudget | None = None,
    ) -> None:
        self.quota_root = Path(quota_root).resolve()
        if not self.quota_root.exists() or not self.quota_root.is_dir():
            raise ValueError("quota_root must be an existing directory")
        self.budget = budget or StorageBudget()
        paths = controlled_paths if controlled_paths is not None else (self.quota_root,)
        if not paths:
            raise ValueError("at least one project-controlled path is required")
        normalised = []
        for path in paths:
            candidate = Path(path).resolve()
            try:
                candidate.relative_to(self.quota_root)
            except ValueError as exc:
                raise ValueError("every controlled path must be within quota_root") from exc
            self._require_quota_device(candidate)
            normalised.append(candidate)
        # Keep every declared location for later device-boundary checks, even
        # when an outer tree makes a nested location redundant for occupancy
        # traversal.  In particular, this preserves detection if a missing
        # future nested path later appears on a different mounted device.
        self._device_check_paths = tuple(sorted(set(normalised), key=os.fspath))
        self.controlled_paths = self._minimal_controlled_paths(
            self._device_check_paths
        )

    @staticmethod
    def _minimal_controlled_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
        """Return deterministic non-overlapping filesystem namespace roots."""

        return tuple(
            candidate
            for candidate in paths
            if not any(
                candidate != possible_parent
                and candidate.is_relative_to(possible_parent)
                for possible_parent in paths
            )
        )

    @staticmethod
    def _nearest_existing_ancestor(path: Path) -> tuple[Path, os.stat_result]:
        candidate = path
        while True:
            try:
                return candidate, candidate.stat()
            except FileNotFoundError:
                parent = candidate.parent
                if parent == candidate:
                    raise ValueError(
                        "controlled path has no inspectable existing ancestor"
                    ) from None
                candidate = parent
            except OSError as exc:
                raise ValueError(
                    "controlled path's nearest existing ancestor cannot be inspected"
                ) from exc

    def _require_quota_device(self, controlled: Path) -> None:
        try:
            quota_device = self.quota_root.stat().st_dev
        except OSError as exc:
            raise ValueError("quota_root device cannot be inspected") from exc
        _ancestor, ancestor_stat = self._nearest_existing_ancestor(controlled)
        if ancestor_stat.st_dev != quota_device:
            raise ValueError(
                "controlled path's nearest existing ancestor is on a different device "
                "from quota_root"
            )

    def measure_occupied_bytes(self) -> int:
        """Measure allocated filesystem blocks once, deduplicating hard links."""

        seen = set()
        total = 0
        # Recheck every originally declared location, including nested paths
        # removed from the traversal roots by semantic deduplication.
        for controlled in self._device_check_paths:
            self._require_quota_device(controlled)
        for controlled in self.controlled_paths:
            if not controlled.exists() and not controlled.is_symlink():
                continue
            candidates: Iterator[Path]
            if not controlled.is_dir() or controlled.is_symlink():
                candidates = iter((controlled,))
            else:
                def traversal_error(error: OSError) -> None:
                    raise OSError(
                        "cannot completely traverse project-controlled storage"
                    ) from error

                def all_entries(root: Path) -> Iterator[Path]:
                    yield root
                    for directory, directories, filenames in os.walk(
                        root, followlinks=False, onerror=traversal_error
                    ):
                        # Directory blocks and directory symlinks occupy quota
                        # too. Hard-link identity deduplication remains below.
                        for name in (*directories, *filenames):
                            yield Path(directory) / name

                candidates = all_entries(controlled)
            for candidate in candidates:
                stat = candidate.lstat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                blocks = getattr(stat, "st_blocks", 0)
                total += blocks * 512 if blocks else stat.st_size
        return total

    def check(
        self,
        *,
        declared_growth_bytes: int = 0,
        largest_atomic_temporary_bytes: int = 0,
        quarantine_allowance_bytes: int = 0,
        release_staging_bytes: int = 0,
        current_occupied_bytes: int | None = None,
        filesystem_free_bytes: int | None = None,
    ) -> StorageReport:
        # An explicit occupancy observation must not bypass device-boundary
        # enforcement if a controlled path became a mount after construction.
        for controlled in self._device_check_paths:
            self._require_quota_device(controlled)
        growth = _nonnegative_int("declared_growth_bytes", declared_growth_bytes)
        temporary = _nonnegative_int(
            "largest_atomic_temporary_bytes", largest_atomic_temporary_bytes
        )
        quarantine = _nonnegative_int("quarantine_allowance_bytes", quarantine_allowance_bytes)
        release = _nonnegative_int("release_staging_bytes", release_staging_bytes)
        current = (
            self.measure_occupied_bytes()
            if current_occupied_bytes is None
            else _nonnegative_int("current_occupied_bytes", current_occupied_bytes)
        )
        filesystem_free = (
            shutil.disk_usage(str(self.quota_root)).free
            if filesystem_free_bytes is None
            else _nonnegative_int("filesystem_free_bytes", filesystem_free_bytes)
        )
        additional = growth + temporary + quarantine + release
        projected = current + additional
        allocation_free = self.budget.total_allocation_bytes - projected
        projected_filesystem_free = filesystem_free - additional
        effective_headroom = min(allocation_free, projected_filesystem_free)
        violations = []
        if projected > self.budget.max_occupied_bytes:
            violations.append("projected_occupancy_exceeds_limit")
        if filesystem_free < self.budget.min_headroom_bytes:
            violations.append("actual_filesystem_headroom_below_minimum")
        if allocation_free < self.budget.min_headroom_bytes:
            violations.append("projected_allocation_headroom_below_minimum")
        if projected_filesystem_free < self.budget.min_headroom_bytes:
            violations.append("projected_filesystem_headroom_below_minimum")
        return StorageReport(
            current_occupied_bytes=current,
            declared_growth_bytes=growth,
            largest_atomic_temporary_bytes=temporary,
            quarantine_allowance_bytes=quarantine,
            release_staging_bytes=release,
            projected_occupied_bytes=projected,
            filesystem_free_bytes=filesystem_free,
            projected_allocation_free_bytes=allocation_free,
            projected_filesystem_free_bytes=projected_filesystem_free,
            effective_projected_headroom_bytes=effective_headroom,
            budget=self.budget,
            allowed=not violations,
            violations=tuple(violations),
        )

    def require(self, **reservations: Any) -> StorageReport:
        report = self.check(**reservations)
        if not report.allowed:
            raise StorageBudgetExceeded(report)
        return report


__all__ = [
    "ALLOWED_JOB_TRANSITIONS",
    "DEFAULT_GPU_HARD_LIMIT_SECONDS",
    "DEFAULT_MAX_OCCUPIED_BYTES",
    "DEFAULT_MIN_HEADROOM_BYTES",
    "DEFAULT_TOTAL_ALLOCATION_BYTES",
    "SCHEMA_VERSION",
    "ArtifactIntegrityError",
    "ArtifactRecord",
    "ArtifactStore",
    "AttemptKind",
    "AttemptRecord",
    "BlobStore",
    "BlobTooLargeError",
    "CommitmentCheckStatus",
    "Compression",
    "CompressionUnavailableError",
    "DuplicateConflictError",
    "EvidenceSnapshotRecord",
    "EvidenceSupportStatus",
    "FailureKind",
    "FailureRecord",
    "FeedbackAction",
    "FeedbackKind",
    "FeedbackRecord",
    "FeedbackResolutionStatus",
    "GpuAllocationJournalRecord",
    "GpuAllocationJournalState",
    "GpuBudgetExceeded",
    "GpuEvent",
    "GpuEventKind",
    "GpuServiceJournalRecord",
    "GpuServiceJournalState",
    "GpuServiceSession",
    "GpuSummary",
    "InputKind",
    "InputRecord",
    "InvalidTransitionError",
    "JobRecord",
    "JobState",
    "JobTransition",
    "Ledger",
    "MetricRecord",
    "MetricStatus",
    "ModelBackend",
    "ModelCallRecord",
    "ModelCallRole",
    "PacketMaterializationRecord",
    "PrequeryBarrierRecord",
    "ProjectionRecord",
    "QueryAccessRecord",
    "ReadOnlyArtifactStore",
    "ReadOnlyBlobStore",
    "ReadOnlyLedger",
    "ReleaseClass",
    "ReleaseViolationError",
    "ResourceSampleRecord",
    "RetryClass",
    "SemanticAssessmentScope",
    "StorageBudget",
    "StorageBudgetExceeded",
    "StoragePreflight",
    "StorageReport",
    "StorageSampleRecord",
    "StudyJobRecord",
    "StudyRecord",
    "TemporalValidationStatus",
    "ValidationRecord",
    "ValidationStatus",
    "VisualizationRecord",
    "canonical_json",
    "derive_job_identity",
    "sha256_bytes",
]
