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
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

SCHEMA_VERSION = 3
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
    JobState.PREQUERY_SEALED: frozenset({JobState.QUERY_REVEALED}),
    JobState.QUERY_REVEALED: frozenset({JobState.GENERATED}),
    JobState.GENERATED: frozenset({JobState.REPAIRED, JobState.VALIDATED}),
    JobState.REPAIRED: frozenset({JobState.VALIDATED}),
    JobState.VALIDATED: frozenset({JobState.FINALIZED}),
    JobState.FINALIZED: frozenset({JobState.SCORED}),
    JobState.SCORED: frozenset({JobState.RENDERED}),
    JobState.RENDERED: frozenset(),
}


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
            except Exception:
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
        INSERT OR IGNORE INTO schema_metadata(schema_version, created_at)
            VALUES ({SCHEMA_VERSION}, '{_normalise_timestamp()}');

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
        with self._lock:
            self._connection.executescript(schema)
            for table in self._APPEND_ONLY_TABLES:
                self._connection.executescript(
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
                )

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
        ):
            _normalise_hash(name, value)
        job = self.get_job(job_id)
        validation = self.get_validation(validation_id)
        if validation.job_id != job_id:
            raise ValueError("projection validation cannot cross jobs")
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
                or status is FeedbackResolutionStatus.PENDING
            ):
                raise ValueError("a condition resolution requires its own job/projection/status")
            _metadata_token("receiving_condition", receiving_condition)
            self.get_job(job_id)
            before = self.get_projection(before_projection_id)
            if before.job_id != job_id:
                raise ValueError("feedback projection cannot cross jobs")
            if after_projection_id is not None:
                after = self.get_projection(after_projection_id)
                if after.job_id != job_id:
                    raise ValueError("feedback projection cannot cross jobs")
            if (
                self._connection.execute(
                    "SELECT 1 FROM study_jobs WHERE study_id = ? AND job_id = ?",
                    (study_id, job_id),
                ).fetchone()
                is None
            ):
                raise ValueError("feedback job must be linked to its study")
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
        if projection_id is not None:
            projection = self.get_projection(projection_id)
            linked = self._connection.execute(
                """SELECT 1 FROM study_jobs
                   WHERE study_id = ? AND job_id = ?""",
                (study_id, projection.job_id),
            ).fetchone()
            if linked is None:
                raise ValueError("metric projection must belong to its study")
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

    def transition_job(
        self, job_id: str, to_state: JobState, *, occurred_at: Any | None = None
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
            if target not in ALLOWED_JOB_TRANSITIONS[current]:
                raise InvalidTransitionError(
                    f"cannot transition job from {current.value} to {target.value}"
                )
            timestamp = explicit_timestamp or _normalise_timestamp()
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

    def register_artifact(self, record: ArtifactRecord) -> ArtifactRecord:
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
        existing = self._connection.execute(
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
            with self._transaction() as cursor:
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
            raise ValueError(
                "service_overhead is session-derived; use record_gpu_service_session"
            )
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
        by_kind_values = {
            GpuEventKind(row["event_kind"]): int(row["total"])
            for row in rows
        }
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

    def count_rows(self, table: str) -> int:
        """Return a test/verification count for a known append-only table."""

        if table not in self._APPEND_ONLY_TABLES:
            raise ValueError("unknown or non-scientific table")
        row = self._connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        return int(row["count"])


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
            normalised.append(candidate)
        self.controlled_paths = tuple(normalised)

    def measure_occupied_bytes(self) -> int:
        """Measure allocated filesystem blocks once, deduplicating hard links."""

        seen = set()
        total = 0
        for controlled in self.controlled_paths:
            if not controlled.exists():
                continue
            candidates: Iterator[Path]
            if controlled.is_file():
                candidates = iter((controlled,))
            else:
                candidates = (
                    Path(directory) / filename
                    for directory, _, filenames in os.walk(controlled, followlinks=False)
                    for filename in filenames
                )
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
    "GpuBudgetExceeded",
    "GpuEvent",
    "GpuEventKind",
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
    "ProjectionRecord",
    "ReleaseClass",
    "ReleaseViolationError",
    "ResourceSampleRecord",
    "RetryClass",
    "StorageBudget",
    "StorageBudgetExceeded",
    "StoragePreflight",
    "StorageReport",
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
