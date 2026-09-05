"""Exhaustive, read-only verification for a study ledger and its CAS.

This module deliberately does not construct :class:`store.Ledger` or
:class:`store.BlobStore`: both normal runtime objects prepare writable state.
Recovery and release checks instead open SQLite with ``mode=ro`` and decode
registered blobs directly from their immutable metadata.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from story_projection_onto.contracts import OntologyProjection
from story_projection_onto.store import (
    ALLOWED_JOB_TRANSITIONS,
    DEFAULT_GPU_HARD_LIMIT_SECONDS,
    DEFAULT_MAX_OCCUPIED_BYTES,
    DEFAULT_MIN_HEADROOM_BYTES,
    DEFAULT_TOTAL_ALLOCATION_BYTES,
    SCHEMA_VERSION,
    JobState,
    SemanticAssessmentScope,
    canonical_json,
    frozen_legacy_fallback_c1_retry_prebuild_matches,
)

_SHA256_LENGTH = 64
_REQUIRED_TABLES = frozenset(
    {
        "schema_metadata",
        "jobs",
        "job_transitions",
        "attempts",
        "failures",
        "artifacts",
        "job_artifacts",
        "gpu_events",
        "gpu_allocation_journal",
        "gpu_service_journal",
        "gpu_service_sessions",
        "model_calls",
        "studies",
        "study_jobs",
        "inputs",
        "evidence_snapshots",
        "prequery_barriers",
        "query_access_events",
        "packet_materialization_events",
        "validations",
        "projections",
        "feedback",
        "metrics",
        "visualizations",
        "resource_samples",
        "storage_samples",
    }
)
_TERMINAL_ALLOCATION_STATES = frozenset({"closed", "recovered"})
_TERMINAL_SERVICE_STATES = frozenset({"closed", "recovered"})


@dataclass(frozen=True, slots=True)
class LedgerVerificationIssue:
    """One independently detected integrity failure."""

    code: str
    subject: str
    message: str


@dataclass(frozen=True, slots=True)
class LedgerVerificationReport:
    """Canonical summary of every completed verification pass."""

    schema_versions: tuple[int, ...]
    artifact_count: int
    artifact_raw_bytes: int
    artifact_stored_bytes: int
    job_count: int
    attempt_count: int
    artifact_link_count: int
    model_call_count: int
    gpu_event_count: int
    gpu_service_session_count: int
    gpu_total_allocated_microseconds: int
    gpu_by_kind_microseconds: tuple[tuple[str, int], ...]
    unresolved_gpu_allocation_count: int
    unresolved_gpu_service_count: int
    storage_sample_count: int
    peak_recorded_project_storage_bytes: int
    peak_projected_storage_bytes: int
    legacy_validation_scope_count: int
    issues: tuple[LedgerVerificationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["valid"] = self.valid
        payload["status"] = "verified" if self.valid else "invalid"
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


class LedgerVerificationError(RuntimeError):
    """Raised when an exhaustive audit finds one or more failures."""

    def __init__(self, report: LedgerVerificationReport) -> None:
        self.report = report
        super().__init__(
            f"ledger/CAS verification found {len(report.issues)} integrity issue(s)"
        )


class _Issues:
    def __init__(self) -> None:
        self.values: list[LedgerVerificationIssue] = []

    def add(self, code: str, subject: object, message: str) -> None:
        self.values.append(
            LedgerVerificationIssue(code=code, subject=str(subject), message=message)
        )

    def ordered(self) -> tuple[LedgerVerificationIssue, ...]:
        return tuple(sorted(self.values, key=lambda item: (item.code, item.subject, item.message)))


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    # Hold one coherent read snapshot even if an experiment process is still appending.
    connection.execute("BEGIN")
    return connection


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_timestamp(value: object) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not text")
    parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp has no UTC offset")
    return parsed


def _decoded_size_and_hash(
    path: Path,
    compression: str,
    expected_size: int,
) -> tuple[int, str]:
    """Stream a compressed payload, bounding corrupt expansion at expected+1."""

    digest = hashlib.sha256()
    decoded_size = 0
    with path.open("rb") as encoded:
        if compression == "gzip":
            decoded = gzip.GzipFile(fileobj=encoded, mode="rb")
        elif compression == "zstd":
            import zstandard  # type: ignore[import-not-found]

            decoded = zstandard.ZstdDecompressor().stream_reader(encoded)
        else:
            raise ValueError(f"unsupported compression codec {compression!r}")
        try:
            while True:
                remaining = expected_size - decoded_size
                chunk = decoded.read(min(1024 * 1024, max(1, remaining + 1)))
                if not chunk:
                    break
                decoded_size += len(chunk)
                digest.update(chunk)
                if decoded_size > expected_size:
                    break
        finally:
            decoded.close()
    return decoded_size, digest.hexdigest()


def _read_registered_payload(row: sqlite3.Row, blob_root: Path) -> bytes:
    """Read one already-registered CAS payload with its declared size bound."""

    raw_size = row["raw_size_bytes"]
    if not isinstance(raw_size, int) or raw_size < 0:
        raise ValueError("artifact raw size is invalid")
    relative = PurePosixPath(row["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path escapes the CAS root")
    root = blob_root.resolve()
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("artifact is not a regular CAS file")
    with candidate.open("rb") as encoded:
        if row["compression"] == "gzip":
            decoded = gzip.GzipFile(fileobj=encoded, mode="rb")
        elif row["compression"] == "zstd":
            import zstandard  # type: ignore[import-not-found]

            decoded = zstandard.ZstdDecompressor().stream_reader(encoded)
        else:
            raise ValueError("artifact compression is invalid")
        try:
            payload = decoded.read(raw_size + 1)
        finally:
            decoded.close()
    if len(payload) != raw_size or hashlib.sha256(payload).hexdigest() != row["content_hash"]:
        raise ValueError("artifact payload differs from registered metadata")
    return payload


def _find_cycles(
    parents_by_node: Mapping[str, Iterable[str]],
) -> tuple[tuple[str, str], ...]:
    """Return deterministic DFS back-edges without recursion depth risk."""

    state: dict[str, int] = {}
    back_edges: set[tuple[str, str]] = set()
    for start in sorted(parents_by_node):
        if state.get(start, 0) != 0:
            continue
        state[start] = 1
        stack: list[tuple[str, Iterator[str]]] = [
            (start, iter(sorted(set(parents_by_node.get(start, ())))))
        ]
        while stack:
            node, iterator = stack[-1]
            try:
                parent = next(iterator)
            except StopIteration:
                state[node] = 2
                stack.pop()
                continue
            parent_state = state.get(parent, 0)
            if parent_state == 1:
                back_edges.add((node, parent))
            elif parent_state == 0 and parent in parents_by_node:
                state[parent] = 1
                stack.append((parent, iter(sorted(set(parents_by_node.get(parent, ()))))))
    return tuple(sorted(back_edges))


def _json_mapping(value: object) -> bool:
    try:
        decoded = json.loads(value) if isinstance(value, str) else None
    except (TypeError, ValueError):
        return False
    return isinstance(decoded, dict)


def _verify_artifacts(
    rows: list[sqlite3.Row], blob_root: Path, issues: _Issues
) -> tuple[int, int]:
    raw_total = 0
    stored_total = 0
    root = blob_root.resolve()
    if not root.is_dir():
        issues.add("cas_root_missing", root, "CAS root is not an existing directory")

    for row in rows:
        digest = row["content_hash"]
        subject = digest
        raw_size = row["raw_size_bytes"]
        stored_size = row["stored_size_bytes"]
        if not isinstance(raw_size, int) or raw_size < 0:
            issues.add("cas_raw_size_invalid", subject, "raw size is not a nonnegative integer")
            continue
        if not isinstance(stored_size, int) or stored_size < 0:
            issues.add(
                "cas_stored_size_invalid", subject, "stored size is not a nonnegative integer"
            )
            continue
        raw_total += raw_size
        stored_total += stored_size
        if not _is_sha256(digest):
            issues.add("cas_hash_invalid", subject, "content address is not lowercase SHA-256")
            continue
        compression = row["compression"]
        suffix = {"gzip": ".jsonl.gz", "zstd": ".jsonl.zst"}.get(compression)
        if suffix is None:
            issues.add("cas_compression_invalid", subject, f"unknown codec {compression!r}")
            continue
        relative_text = row["relative_path"]
        if not isinstance(relative_text, str):
            issues.add("cas_path_invalid", subject, "relative path is not text")
            continue
        relative = PurePosixPath(relative_text)
        expected = f"{digest[:2]}/{digest}{suffix}"
        if relative.is_absolute() or ".." in relative.parts:
            issues.add("cas_path_escape", subject, "relative path escapes the CAS root")
            continue
        if relative.as_posix() != expected:
            issues.add(
                "cas_path_not_content_addressed",
                subject,
                f"registered path {relative.as_posix()!r} differs from {expected!r}",
            )
        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            issues.add("cas_path_unresolvable", subject, f"cannot resolve artifact path: {exc}")
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            issues.add("cas_path_escape", subject, "resolved path escapes the CAS root")
            continue
        if candidate.is_symlink():
            issues.add("cas_symlink", subject, "artifact path is a symbolic link")
            continue
        if not candidate.is_file():
            issues.add(
                "cas_file_missing",
                subject,
                f"artifact is not readable at {relative_text!r}",
            )
            continue
        try:
            actual_stored_size = candidate.stat().st_size
        except OSError as exc:
            issues.add("cas_unreadable", subject, f"cannot stat compressed artifact: {exc}")
            continue
        if actual_stored_size != stored_size:
            issues.add(
                "cas_stored_size_mismatch",
                subject,
                f"ledger={stored_size}, actual={actual_stored_size}",
            )
        try:
            actual_raw_size, actual_hash = _decoded_size_and_hash(
                candidate,
                compression,
                raw_size,
            )
        except Exception as exc:
            issues.add("cas_decode_failed", subject, f"cannot decode registered artifact: {exc}")
            continue
        if actual_raw_size != raw_size:
            issues.add(
                "cas_raw_size_mismatch",
                subject,
                f"ledger={raw_size}, actual={actual_raw_size}",
            )
        if actual_hash != digest:
            issues.add("cas_hash_mismatch", subject, f"decoded SHA-256 is {actual_hash}")
    return raw_total, stored_total


def _verify_attempts(
    rows: list[sqlite3.Row], jobs: set[str], issues: _Issues
) -> dict[str, sqlite3.Row]:
    attempts = {row["attempt_id"]: row for row in rows}
    parents: dict[str, tuple[str, ...]] = {}
    for attempt_id, row in attempts.items():
        job_id = row["job_id"]
        parent = row["parent_attempt_id"]
        if job_id not in jobs:
            issues.add("attempt_job_missing", attempt_id, f"unknown job {job_id!r}")
        if (row["attempt_kind"] == "base") != (parent is None):
            issues.add(
                "attempt_kind_parent_mismatch",
                attempt_id,
                "only base attempts may omit a parent",
            )
        if parent is None:
            parents[attempt_id] = ()
            continue
        parents[attempt_id] = (parent,)
        parent_row = attempts.get(parent)
        if parent_row is None:
            issues.add("attempt_parent_missing", attempt_id, f"unknown parent {parent!r}")
        elif parent_row["job_id"] != job_id:
            issues.add("attempt_parent_cross_job", attempt_id, "parent belongs to another job")
    for child, parent in _find_cycles(parents):
        issues.add("attempt_cycle", child, f"back-edge to {parent!r}")
    return attempts


def _verify_job_lifecycles(
    connection: sqlite3.Connection,
    jobs: dict[str, sqlite3.Row],
    issues: _Issues,
) -> dict[str, JobState]:
    """Verify every append-only job state chain and its typed query boundary."""

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM job_transitions ORDER BY job_id, sequence, event_id"
    ):
        grouped[row["job_id"]].append(row)

    for transition_job_id in sorted(set(grouped) - set(jobs)):
        issues.add(
            "job_transition_job_missing",
            transition_job_id,
            "transition chain belongs to an unknown job",
        )
    final_states: dict[str, JobState] = {}
    for job_id, job in jobs.items():
        rows = grouped.get(job_id, [])
        if not rows:
            issues.add("job_transition_missing", job_id, "job has no planned transition")
            continue
        try:
            identity = json.loads(job["identity_json"])
        except (TypeError, ValueError):
            identity = None
        if not isinstance(identity, dict):
            issues.add(
                "job_identity_invalid",
                job_id,
                "job identity is not a canonical JSON object",
            )
        else:
            canonical_identity = canonical_json(identity)
            if canonical_identity != job["identity_json"]:
                issues.add(
                    "job_identity_noncanonical",
                    job_id,
                    "job identity JSON is not canonical",
                )
            if hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest() != job[
                "identity_hash"
            ]:
                issues.add(
                    "job_identity_hash_mismatch",
                    job_id,
                    "job identity hash differs from canonical identity JSON",
                )
            if job_id != job["identity_hash"]:
                issues.add(
                    "job_id_identity_hash_mismatch",
                    job_id,
                    "job ID is not its immutable identity hash",
                )
        previous_state: JobState | None = None
        previous_time: dt.datetime | None = None
        for index, row in enumerate(rows):
            subject = f"{job_id}:{index}"
            if row["sequence"] != index:
                issues.add(
                    "job_transition_sequence_gap",
                    subject,
                    f"found sequence {row['sequence']!r}",
                )
            try:
                state = JobState(row["to_state"])
            except (TypeError, ValueError):
                issues.add(
                    "job_transition_state_invalid",
                    subject,
                    f"unknown target state {row['to_state']!r}",
                )
                continue
            expected_from = None if previous_state is None else previous_state.value
            if row["from_state"] != expected_from:
                issues.add(
                    "job_transition_from_state_mismatch",
                    subject,
                    f"expected {expected_from!r}, found {row['from_state']!r}",
                )
            if index == 0:
                if state is not JobState.PLANNED:
                    issues.add(
                        "job_transition_initial_state_invalid",
                        job_id,
                        "transition chain does not begin at planned",
                    )
                if row["occurred_at"] != job["created_at"]:
                    issues.add(
                        "job_transition_creation_time_mismatch",
                        job_id,
                        "planned transition time differs from immutable job creation time",
                    )
            elif previous_state is not None:
                if state not in ALLOWED_JOB_TRANSITIONS[previous_state]:
                    issues.add(
                        "job_transition_edge_invalid",
                        subject,
                        f"{previous_state.value}->{state.value} is not registered",
                    )
                if (
                    previous_state is JobState.PREQUERY_SEALED
                    and state is JobState.GENERATED
                    and (
                        not isinstance(identity, dict)
                        or (
                            (
                                identity.get("lifecycle_kind")
                                != "query_blind_prebuild"
                                or identity.get("condition") not in {"C0", "C1"}
                            )
                            and not frozen_legacy_fallback_c1_retry_prebuild_matches(
                                connection,
                                job_id=job_id,
                                identity=identity,
                            )
                        )
                    )
                ):
                    issues.add(
                        "job_transition_query_boundary_bypass",
                        subject,
                        "direct generation is not a typed C0/C1 query-blind prebuild",
                    )
            try:
                occurred = _parse_timestamp(row["occurred_at"])
                if previous_time is not None and occurred < previous_time:
                    raise ValueError("transition time regressed")
                if (
                    previous_time is not None
                    and previous_state is JobState.PREQUERY_SEALED
                    and state is JobState.QUERY_REVEALED
                    and occurred <= previous_time
                ):
                    raise ValueError("query reveal does not strictly follow prequery seal")
                previous_time = occurred
            except ValueError as exc:
                issues.add("job_transition_time_invalid", subject, str(exc))
            previous_state = state
        if previous_state is not None:
            final_states[job_id] = previous_state
    return final_states


def _verify_validations_and_projections(
    connection: sqlite3.Connection,
    attempts: dict[str, sqlite3.Row],
    artifacts: dict[str, sqlite3.Row],
    blob_root: Path,
    issues: _Issues,
) -> int:
    """Check explicit semantic scope and exact validation/projection lineage."""

    validation_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(validations)")
    }
    if "semantic_assessment_scope" not in validation_columns:
        issues.add(
            "validation_assessment_scope_missing",
            "validations",
            "schema cannot distinguish runtime-unassessed from semantic assessment",
        )
        return 0
    validations = {
        row["validation_id"]: row
        for row in connection.execute("SELECT * FROM validations ORDER BY validation_id")
    }
    model_calls_by_attempt = {
        row["attempt_id"]: row
        for row in connection.execute(
            "SELECT * FROM model_calls ORDER BY model_call_id"
        )
    }
    allowed_scopes = {item.value for item in SemanticAssessmentScope}
    unassessed = ("not_applicable", "not_applicable", "not_applicable")
    legacy_count = 0
    for validation_id, row in validations.items():
        scope = row["semantic_assessment_scope"]
        statuses = (
            row["evidence_support_status"],
            row["temporal_status"],
            row["commitment_status"],
        )
        if scope not in allowed_scopes:
            issues.add(
                "validation_assessment_scope_invalid",
                validation_id,
                f"unknown scope {scope!r}",
            )
        elif scope == SemanticAssessmentScope.LEGACY_UNSPECIFIED.value:
            legacy_count += 1
            issues.add(
                "validation_legacy_scope_unresolved",
                validation_id,
                "legacy validation has no proof of whether semantics were assessed",
            )
        elif (
            scope
            == SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED.value
            and statuses != unassessed
        ):
            issues.add(
                "validation_runtime_scope_claims_semantics",
                validation_id,
                "runtime structural-only row contains assessed semantic statuses",
            )
        elif (
            scope == SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER.value
            and statuses == unassessed
        ):
            issues.add(
                "validation_posthoc_scope_unassessed",
                validation_id,
                "post-hoc row contains no assessed semantic status",
            )
        attempt = attempts.get(row["attempt_id"])
        if attempt is not None and attempt["job_id"] != row["job_id"]:
            issues.add(
                "validation_attempt_cross_job",
                validation_id,
                "validation and attempt identify different jobs",
            )
        if row["input_artifact_hash"] not in artifacts:
            issues.add(
                "validation_input_artifact_missing",
                validation_id,
                "validation input is absent from the registered CAS",
            )
        if (
            row["diagnostics_artifact_hash"] is not None
            and row["diagnostics_artifact_hash"] not in artifacts
        ):
            issues.add(
                "validation_diagnostics_artifact_missing",
                validation_id,
                "validation diagnostics are absent from the registered CAS",
            )
        model_call = model_calls_by_attempt.get(row["attempt_id"])
        if (
            model_call is not None
            and model_call["response_artifact_hash"] is not None
            and model_call["response_artifact_hash"] != row["input_artifact_hash"]
        ):
            issues.add(
                "validation_model_response_mismatch",
                validation_id,
                "validation input differs from its attempt's immutable model response",
            )

    projection_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(projections)")
    }
    if "projection_semantic_hash" not in projection_columns:
        issues.add(
            "projection_semantic_hash_missing",
            "projections",
            "schema cannot bind renderer semantics to projection content",
        )
        return legacy_count
    snapshot_hashes = {
        row["snapshot_id"]: row["content_hash"]
        for row in connection.execute(
            """SELECT s.snapshot_id, i.content_hash
               FROM evidence_snapshots s JOIN inputs i ON i.input_id = s.input_id"""
        )
    }
    packet_hashes = {
        row["input_id"]: row["content_hash"]
        for row in connection.execute(
            "SELECT input_id, content_hash FROM inputs WHERE input_kind = 'evidence_packet'"
        )
    }
    for row in connection.execute("SELECT * FROM projections ORDER BY projection_id"):
        projection_id = row["projection_id"]
        validation = validations.get(row["validation_id"])
        if validation is None:
            # Normally covered by SQLite's foreign-key check, but retain a
            # domain-specific issue when auditing a damaged database.
            issues.add(
                "projection_validation_missing",
                projection_id,
                f"unknown validation {row['validation_id']!r}",
            )
            continue
        if validation["job_id"] != row["job_id"]:
            issues.add(
                "projection_validation_cross_job",
                projection_id,
                "projection and validation identify different jobs",
            )
        if (
            validation["attempt_id"] not in model_calls_by_attempt
            and validation["input_artifact_hash"] != row["projection_artifact_hash"]
        ):
            issues.add(
                "projection_cpu_validation_input_mismatch",
                projection_id,
                "CPU projection validation does not name its exact projection artifact",
            )
        if validation["validation_status"] != "accepted":
            issues.add(
                "projection_validation_rejected",
                projection_id,
                "projection points to a non-accepted validation",
            )
        if (
            validation["semantic_assessment_scope"]
            != SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED.value
            or (
                validation["evidence_support_status"],
                validation["temporal_status"],
                validation["commitment_status"],
            )
            != ("not_applicable", "not_applicable", "not_applicable")
        ):
            issues.add(
                "projection_validation_scope_invalid",
                projection_id,
                "projection validation is not accepted runtime structural-only lineage",
            )
        semantic_hash = row["projection_semantic_hash"]
        if not _is_sha256(semantic_hash):
            issues.add(
                "projection_semantic_hash_invalid",
                projection_id,
                "projection semantic identity is absent or is not lowercase SHA-256",
            )
            continue
        artifact = artifacts.get(row["projection_artifact_hash"])
        if artifact is None:
            issues.add(
                "projection_artifact_missing",
                projection_id,
                f"unknown artifact {row['projection_artifact_hash']!r}",
            )
            continue
        try:
            payload_bytes = _read_registered_payload(artifact, blob_root)
            projection = OntologyProjection.model_validate_json(payload_bytes)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            issues.add(
                "projection_artifact_payload_invalid",
                projection_id,
                f"cannot parse canonical OntologyProjection: {error}",
            )
            continue
        canonical = projection.to_canonical_json().encode("utf-8")
        if payload_bytes not in {canonical, canonical + b"\n"}:
            issues.add(
                "projection_artifact_noncanonical",
                projection_id,
                "projection artifact bytes are not a canonical serialization",
            )
        if projection.content_hash != semantic_hash:
            issues.add(
                "projection_semantic_hash_mismatch",
                projection_id,
                "ledger semantic hash differs from the projection artifact content hash",
            )
        certificate = (
            projection.construction_certificate or projection.construction_seal
        )
        expected_bindings = {
            "condition_id": projection.condition.value,
            "context_hash": projection.context_hash,
            "snapshot_hash": projection.snapshot_hash,
            "packet_hash": projection.packet_hash,
            "upper_ontology_hash": projection.upper_ontology.content_hash,
            "construction_certificate_hash": (
                None if certificate is None else certificate.content_hash
            ),
        }
        observed_bindings = {
            "condition_id": row["condition_id"],
            "context_hash": row["context_hash"],
            "snapshot_hash": snapshot_hashes.get(row["snapshot_id"]),
            "packet_hash": packet_hashes.get(row["packet_input_id"]),
            "upper_ontology_hash": row["upper_ontology_hash"],
            "construction_certificate_hash": row[
                "construction_certificate_hash"
            ],
        }
        for binding_name, expected in expected_bindings.items():
            if observed_bindings[binding_name] != expected:
                issues.add(
                    "projection_artifact_ledger_binding_mismatch",
                    f"{projection_id}:{binding_name}",
                    "projection artifact differs from its immutable ledger binding",
                )
        if row["release_class"] == "public" and projection.release_class.value != "public":
            issues.add(
                "projection_release_class_underclassified",
                projection_id,
                "public ledger row contains a restricted projection payload",
            )
    return legacy_count


def _verify_scientific_lineage(
    connection: sqlite3.Connection,
    *,
    jobs: dict[str, sqlite3.Row],
    job_states: Mapping[str, JobState],
    artifacts: dict[str, sqlite3.Row],
    issues: _Issues,
) -> None:
    """Verify study ownership and downstream metric/render/feedback bindings."""

    studies = {
        row["study_id"]: row
        for row in connection.execute("SELECT * FROM studies ORDER BY study_id")
    }
    study_by_job: dict[str, str] = {}
    for row in connection.execute("SELECT * FROM study_jobs ORDER BY link_id"):
        link_id = row["link_id"]
        study_id = row["study_id"]
        job_id = row["job_id"]
        if study_id not in studies:
            issues.add("study_job_study_missing", link_id, f"unknown study {study_id!r}")
        if job_id not in jobs:
            issues.add("study_job_job_missing", link_id, f"unknown job {job_id!r}")
        prior = study_by_job.setdefault(job_id, study_id)
        if prior != study_id:
            issues.add("study_job_ambiguous", job_id, "job is linked to multiple studies")

    inputs = {
        row["input_id"]: row
        for row in connection.execute("SELECT * FROM inputs ORDER BY input_id")
    }
    snapshots = {
        row["snapshot_id"]: row
        for row in connection.execute(
            "SELECT * FROM evidence_snapshots ORDER BY snapshot_id"
        )
    }
    projections = {
        row["projection_id"]: row
        for row in connection.execute("SELECT * FROM projections ORDER BY projection_id")
    }
    for projection_id, projection in projections.items():
        study_id = study_by_job.get(projection["job_id"])
        snapshot = snapshots.get(projection["snapshot_id"])
        packet = inputs.get(projection["packet_input_id"])
        snapshot_input = None if snapshot is None else inputs.get(snapshot["input_id"])
        if study_id is None:
            issues.add(
                "projection_study_job_missing",
                projection_id,
                "projection job is not linked to a study",
            )
        if snapshot_input is None or snapshot_input["input_kind"] != "evidence_snapshot":
            issues.add(
                "projection_snapshot_input_invalid",
                projection_id,
                "projection snapshot does not resolve to a snapshot input",
            )
        elif study_id is not None and snapshot_input["study_id"] != study_id:
            issues.add(
                "projection_snapshot_cross_study",
                projection_id,
                "projection snapshot belongs to another study",
            )
        if packet is None or packet["input_kind"] != "evidence_packet":
            issues.add(
                "projection_packet_input_invalid",
                projection_id,
                "projection packet does not resolve to an evidence packet input",
            )
        elif study_id is not None and packet["study_id"] != study_id:
            issues.add(
                "projection_packet_cross_study",
                projection_id,
                "projection packet belongs to another study",
            )

    for row in connection.execute("SELECT * FROM feedback ORDER BY feedback_id"):
        feedback_id = row["feedback_id"]
        study_id = row["study_id"]
        if study_id not in studies:
            issues.add("feedback_study_missing", feedback_id, f"unknown study {study_id!r}")
        if row["feedback_kind"] == "user_revision":
            if any(
                row[name] is not None
                for name in (
                    "job_id",
                    "receiving_condition",
                    "before_projection_id",
                    "after_projection_id",
                    "resolution_artifact_hash",
                )
            ) or row["resolution_status"] != "pending":
                issues.add(
                    "feedback_user_revision_shape_invalid",
                    feedback_id,
                    "shared revision contains condition-specific output",
                )
            continue
        job_id = row["job_id"]
        before = projections.get(row["before_projection_id"])
        after = projections.get(row["after_projection_id"])
        if job_id is None or study_by_job.get(job_id) != study_id:
            issues.add(
                "feedback_receiving_job_invalid",
                feedback_id,
                "receiving job is absent or belongs to another study",
            )
        if before is None or before["job_id"] not in study_by_job:
            issues.add(
                "feedback_source_projection_invalid",
                feedback_id,
                "before projection is absent or its source job has no unique study owner",
            )
        elif (
            before["condition_id"] != row["receiving_condition"]
            or before["context_hash"] != row["before_context_hash"]
        ):
            issues.add(
                "feedback_source_projection_mismatch",
                feedback_id,
                "before projection condition/context differs from feedback metadata",
            )
        if row["resolution_status"] == "applied" and after is None:
            issues.add(
                "feedback_applied_projection_missing",
                feedback_id,
                "applied condition resolution lacks its after projection",
            )
        if after is not None and (
            after["job_id"] != job_id
            or after["condition_id"] != row["receiving_condition"]
            or after["context_hash"] != row["after_context_hash"]
        ):
            issues.add(
                "feedback_result_projection_mismatch",
                feedback_id,
                "after projection is not owned by the receiving job/condition/context",
            )
        if row["resolution_status"] == "pending":
            issues.add(
                "feedback_condition_resolution_pending",
                feedback_id,
                "condition-specific feedback cannot remain pending",
            )
        if row["resolution_artifact_hash"] not in artifacts:
            issues.add(
                "feedback_resolution_artifact_missing",
                feedback_id,
                "condition resolution lacks its registered result artifact",
            )

    metrics_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM metrics ORDER BY metric_id"):
        metric_id = row["metric_id"]
        study_id = row["study_id"]
        job_id = row["job_id"]
        if study_id not in studies:
            issues.add("metric_study_missing", metric_id, f"unknown study {study_id!r}")
        if job_id is None or study_by_job.get(job_id) != study_id:
            issues.add(
                "metric_job_lineage_invalid",
                metric_id,
                "metric lacks an exact job in its study",
            )
        else:
            metrics_by_job[job_id].append(row)
            if job_states.get(job_id) not in {
                JobState.FINALIZED,
                JobState.SCORED,
                JobState.RENDERED,
            }:
                issues.add(
                    "metric_job_not_finalized",
                    metric_id,
                    "metric predates a finalized generation job",
                )
        projection = projections.get(row["projection_id"])
        if row["projection_id"] is not None and (
            projection is None or projection["job_id"] != job_id
        ):
            issues.add(
                "metric_projection_lineage_invalid",
                metric_id,
                "metric projection does not belong to its exact job",
            )
        if row["result_artifact_hash"] not in artifacts:
            issues.add(
                "metric_result_artifact_missing",
                metric_id,
                "metric lacks its registered immutable result artifact",
            )

    visualizations_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM visualizations ORDER BY visualization_id"
    ):
        visualization_id = row["visualization_id"]
        projection = projections.get(row["projection_id"])
        if projection is None:
            issues.add(
                "visualization_projection_missing",
                visualization_id,
                f"unknown projection {row['projection_id']!r}",
            )
            continue
        visualizations_by_job[projection["job_id"]].append(row)
        if row["semantic_hash"] != projection["projection_semantic_hash"]:
            issues.add(
                "visualization_semantic_hash_mismatch",
                visualization_id,
                "visualization semantic identity differs from its projection",
            )
        if row["visualization_artifact_hash"] not in artifacts:
            issues.add(
                "visualization_artifact_missing",
                visualization_id,
                "visualization artifact is absent from the CAS ledger",
            )

    attempts_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM attempts ORDER BY attempt_id"):
        attempts_by_job[row["job_id"]].append(row)
    calls_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM model_calls ORDER BY model_call_id"):
        calls_by_job[row["job_id"]].append(row)
    validations_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute("SELECT * FROM validations ORDER BY validation_id"):
        validations_by_job[row["job_id"]].append(row)
    projections_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in projections.values():
        projections_by_job[row["job_id"]].append(row)
    failures_by_job: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        """SELECT f.*, a.job_id FROM failures f
           JOIN attempts a ON a.attempt_id = f.attempt_id ORDER BY f.failure_id"""
    ):
        failures_by_job[row["job_id"]].append(row)

    state_order = {
        JobState.PLANNED: 0,
        JobState.PREQUERY_SEALED: 1,
        JobState.QUERY_REVEALED: 2,
        JobState.GENERATED: 3,
        JobState.REPAIRED: 4,
        JobState.VALIDATED: 5,
        JobState.FINALIZED: 6,
        JobState.SCORED: 7,
        JobState.RENDERED: 8,
    }
    phase1_conditions = {
        "c1-01": "C1",
        "c1-02": "C1",
        "c2-01": "C2",
        "c2-02": "C2",
        "c2-03": "C2",
        "fixed-01": "A-FixedSelect",
        "fixed-02": "A-FixedSelect",
        "repair-base-fixture": "C2",
        "fallback-c1-01": "C1",
        "fallback-c2-01": "C2",
        "fallback-c2-02": "C2",
        "fallback-fixed-01": "A-FixedSelect",
    }
    for job_id, state in job_states.items():
        try:
            identity = json.loads(jobs[job_id]["identity_json"])
        except (TypeError, ValueError):
            identity = {}
        identity = identity if isinstance(identity, dict) else {}
        lifecycle_kind = identity.get("lifecycle_kind")
        condition = identity.get("condition")
        if frozen_legacy_fallback_c1_retry_prebuild_matches(
            connection,
            job_id=job_id,
            identity=identity,
        ):
            condition = "C1"
            lifecycle_kind = "query_blind_prebuild"
        is_cpu_generation = (
            condition == "C0"
            and lifecycle_kind
            in {
                "query_blind_prebuild",
                "query_time_projection",
                "phase5_cpu_reprojection",
            }
        ) or (
            condition == "C1"
            and lifecycle_kind
            in {"query_time_projection", "phase5_cpu_reprojection"}
        )
        rank = state_order[state]
        if rank >= state_order[JobState.GENERATED]:
            if not attempts_by_job[job_id]:
                issues.add(
                    "job_generated_attempt_missing",
                    job_id,
                    "generated job has no durable attempt",
                )
            if not calls_by_job[job_id] and not failures_by_job[job_id] and not (
                is_cpu_generation and validations_by_job[job_id]
            ):
                issues.add(
                    "job_generated_evidence_missing",
                    job_id,
                    "generated job has neither a model call nor typed CPU output evidence",
                )
        if rank >= state_order[JobState.VALIDATED] and not validations_by_job[job_id]:
            issues.add(
                "job_validated_row_missing",
                job_id,
                "validated job has no validation row",
            )
        if rank >= state_order[JobState.FINALIZED]:
            accepted_prebuild = (
                lifecycle_kind == "query_blind_prebuild"
                and any(
                    row["validation_status"] == "accepted"
                    for row in validations_by_job[job_id]
                )
                and (
                    condition == "C0"
                    or (
                        condition == "C1"
                        and any(
                            bool(call["successful"])
                            and call["response_artifact_hash"] is not None
                            and any(
                                validation["validation_status"] == "accepted"
                                and validation["attempt_id"] == call["attempt_id"]
                                and validation["input_artifact_hash"]
                                == call["response_artifact_hash"]
                                for validation in validations_by_job[job_id]
                            )
                            for call in calls_by_job[job_id]
                        )
                    )
                )
            )
            phase1_call_id = identity.get("call_id")
            phase1_terminal_validations = {
                row["attempt_id"]: row
                for row in validations_by_job[job_id]
                if row["validation_status"] == "accepted"
            }
            phase1_terminal_artifact_bound = any(
                call["backend"] == "vllm_gpu"
                and bool(call["successful"])
                and call["response_artifact_hash"] is not None
                and call["attempt_id"] in phase1_terminal_validations
                and phase1_terminal_validations[call["attempt_id"]][
                    "input_artifact_hash"
                ]
                == call["response_artifact_hash"]
                for call in calls_by_job[job_id]
            )
            phase1_acceptance = (
                set(identity)
                == {"run_id", "call_id", "plan_hash", "condition", "lifecycle_kind"}
                and isinstance(phase1_call_id, str)
                and phase1_conditions.get(phase1_call_id) == condition
                and lifecycle_kind
                == (
                    "query_blind_prebuild"
                    if condition == "C1"
                    else "query_time_generation"
                )
                and phase1_terminal_artifact_bound
            )
            if not (
                projections_by_job[job_id]
                or failures_by_job[job_id]
                or accepted_prebuild
                or phase1_acceptance
            ):
                issues.add(
                    "job_finalized_output_missing",
                    job_id,
                    "finalized job has no projection, terminal failure, or typed prebuild",
                )
        if rank >= state_order[JobState.SCORED]:
            metrics = metrics_by_job[job_id]
            if not metrics or any(row["result_artifact_hash"] is None for row in metrics):
                issues.add(
                    "job_scored_metric_inventory_missing",
                    job_id,
                    "scored job lacks a durable job-bound metric inventory",
                )
        if rank >= state_order[JobState.RENDERED] and not visualizations_by_job[job_id]:
            issues.add(
                "job_rendered_visualization_missing",
                job_id,
                "rendered job lacks a projection-bound visualization",
            )


def _verify_boundary_and_failures(
    connection: sqlite3.Connection,
    *,
    jobs: dict[str, sqlite3.Row],
    attempts: dict[str, sqlite3.Row],
    artifacts: dict[str, sqlite3.Row],
    blob_root: Path,
    issues: _Issues,
) -> None:
    """Replay query-blind/query-time boundaries and terminal failure materialization."""

    def artifact_json(
        *,
        subject: str,
        artifact_hash: str,
        code_prefix: str,
    ) -> Mapping[str, object] | None:
        artifact = artifacts.get(artifact_hash)
        if artifact is None:
            issues.add(
                f"{code_prefix}_artifact_missing",
                subject,
                f"unknown artifact {artifact_hash!r}",
            )
            return None
        try:
            value = json.loads(_read_registered_payload(artifact, blob_root))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            issues.add(
                f"{code_prefix}_artifact_invalid",
                subject,
                f"cannot decode registered JSON artifact: {error}",
            )
            return None
        if not isinstance(value, dict):
            issues.add(
                f"{code_prefix}_artifact_invalid",
                subject,
                "registered JSON artifact is not an object",
            )
            return None
        return value

    for row in connection.execute("SELECT * FROM failures ORDER BY failure_id"):
        failure_id = row["failure_id"]
        attempt = attempts.get(row["attempt_id"])
        if attempt is None:
            issues.add(
                "failure_attempt_missing",
                failure_id,
                f"unknown attempt {row['attempt_id']!r}",
            )
        artifact_hash = row["artifact_hash"]
        if artifact_hash is not None:
            artifact = artifacts.get(artifact_hash)
            if artifact is None:
                issues.add(
                    "failure_artifact_missing",
                    failure_id,
                    f"unknown artifact {artifact_hash!r}",
                )
            elif attempt is not None:
                job = jobs.get(attempt["job_id"])
                if (
                    job is not None
                    and job["release_class"] == "public"
                    and artifact["release_class"] == "restricted"
                ):
                    issues.add(
                        "failure_artifact_release_violation",
                        failure_id,
                        "public failure lineage cites a restricted artifact",
                    )
        if not _json_mapping(row["details_json"]):
            issues.add(
                "failure_details_invalid",
                failure_id,
                "failure details are not a JSON object",
            )
        try:
            _parse_timestamp(row["occurred_at"])
        except ValueError as error:
            issues.add("failure_time_invalid", failure_id, str(error))

    failure_attempt_ids = {
        row["attempt_id"]
        for row in connection.execute("SELECT attempt_id FROM failures")
    }
    for row in connection.execute(
        "SELECT validation_id, attempt_id FROM validations "
        "WHERE validation_status = 'rejected' ORDER BY validation_id"
    ):
        if row["attempt_id"] not in failure_attempt_ids:
            issues.add(
                "rejected_validation_failure_missing",
                row["validation_id"],
                "rejected validation has no retained failure row for its attempt",
            )

    barriers = {
        row["barrier_hash"]: row
        for row in connection.execute(
            "SELECT * FROM prequery_barriers ORDER BY barrier_hash"
        )
    }
    barrier_payloads: dict[str, Mapping[str, object]] = {}
    for barrier_hash, row in barriers.items():
        try:
            sealed = _parse_timestamp(row["sealed_at"])
            persisted = _parse_timestamp(row["persisted_at"])
            if sealed > persisted:
                raise ValueError("barrier persistence predates sealing")
        except ValueError as error:
            issues.add("prequery_barrier_time_invalid", barrier_hash, str(error))
        payload = artifact_json(
            subject=barrier_hash,
            artifact_hash=row["barrier_artifact_hash"],
            code_prefix="prequery_barrier",
        )
        if payload is not None and payload.get("content_hash") != barrier_hash:
            issues.add(
                "prequery_barrier_artifact_hash_mismatch",
                barrier_hash,
                "barrier artifact does not bind the barrier logical hash",
            )
        elif payload is not None:
            barrier_payloads[barrier_hash] = payload
            bindings = payload.get("preparation_bindings")
            if (
                payload.get("execution_id") != row["execution_id"]
                or payload.get("execution_manifest_hash")
                != row["execution_manifest_hash"]
                or not isinstance(bindings, list)
                or len(bindings) != row["preparation_count"]
            ):
                issues.add(
                    "prequery_barrier_artifact_lineage_mismatch",
                    barrier_hash,
                    "barrier artifact differs from its execution/count ledger row",
                )
        artifact = artifacts.get(row["barrier_artifact_hash"])
        if artifact is not None and artifact["release_class"] != row["release_class"]:
            issues.add(
                "prequery_barrier_release_mismatch",
                barrier_hash,
                "barrier and artifact release classes differ",
            )

    accesses = {
        row["access_event_hash"]: row
        for row in connection.execute(
            "SELECT * FROM query_access_events ORDER BY access_event_hash"
        )
    }
    for access_hash, row in accesses.items():
        barrier = barriers.get(row["prequery_barrier_hash"])
        try:
            revealed = _parse_timestamp(row["registered_revealed_at"])
            accessed = _parse_timestamp(row["accessed_at"])
            if revealed > accessed:
                raise ValueError("query access predates registered reveal")
            if barrier is not None and (
                _parse_timestamp(barrier["sealed_at"]) >= accessed
                or _parse_timestamp(barrier["persisted_at"]) >= accessed
            ):
                raise ValueError("query access does not strictly follow its barrier")
        except ValueError as error:
            issues.add("query_access_time_invalid", access_hash, str(error))
        if barrier is None:
            issues.add(
                "query_access_barrier_missing",
                access_hash,
                f"unknown barrier {row['prequery_barrier_hash']!r}",
            )
        elif (
            barrier["execution_id"] != row["execution_id"]
            or barrier["release_class"] != row["release_class"]
        ):
            issues.add(
                "query_access_barrier_lineage_mismatch",
                access_hash,
                "query access crosses execution or release boundary",
            )
        barrier_payload = barrier_payloads.get(row["prequery_barrier_hash"])
        if barrier_payload is not None:
            bindings = barrier_payload.get("preparation_bindings")
            snapshot_hashes = (
                {
                    binding.get("snapshot_hash")
                    for binding in bindings
                    if isinstance(binding, dict)
                }
                if isinstance(bindings, list)
                else set()
            )
            if row["snapshot_hash"] not in snapshot_hashes:
                issues.add(
                    "query_access_snapshot_not_sealed",
                    access_hash,
                    "query access snapshot is absent from the sealed prequery inventory",
                )
        if row["packet_hash"] is not None:
            issues.add(
                "query_access_pre_materialization_packet",
                access_hash,
                "query access claims a packet before materialization",
            )
        access_payload = artifact_json(
            subject=access_hash,
            artifact_hash=row["access_event_artifact_hash"],
            code_prefix="query_access_event",
        )
        if access_payload is not None and access_payload.get("content_hash") != access_hash:
            issues.add(
                "query_access_event_artifact_hash_mismatch",
                access_hash,
                "query-access artifact does not bind the event logical hash",
            )
        elif access_payload is not None and any(
            access_payload.get(field) != row[field]
            for field in (
                "execution_id",
                "query_context_hash",
                "snapshot_hash",
                "stage_manifest_hash",
                "prequery_barrier_hash",
            )
        ):
            issues.add(
                "query_access_event_artifact_lineage_mismatch",
                access_hash,
                "query-access artifact differs from its immutable ledger row",
            )
        query_payload = artifact_json(
            subject=access_hash,
            artifact_hash=row["query_payload_artifact_hash"],
            code_prefix="query_payload",
        )
        if query_payload is not None and query_payload.get("content_hash") not in {
            row["query_artifact_hash"],
            row["query_context_hash"],
        }:
            issues.add(
                "query_payload_logical_hash_mismatch",
                access_hash,
                "query payload does not bind its registered query/context hash",
            )
        for artifact_hash in (
            row["access_event_artifact_hash"],
            row["query_payload_artifact_hash"],
        ):
            artifact = artifacts.get(artifact_hash)
            if artifact is not None and artifact["release_class"] != row["release_class"]:
                issues.add(
                    "query_access_release_mismatch",
                    access_hash,
                    "query access and artifact release classes differ",
                )

    for row in connection.execute(
        "SELECT * FROM packet_materialization_events ORDER BY materialization_event_hash"
    ):
        event_hash = row["materialization_event_hash"]
        access = accesses.get(row["query_access_event_hash"])
        try:
            started = _parse_timestamp(row["started_at"])
            completed = _parse_timestamp(row["completed_at"])
            if started > completed:
                raise ValueError("packet materialization completion predates start")
            if access is not None and _parse_timestamp(access["accessed_at"]) > started:
                raise ValueError("packet materialization predates query access")
        except ValueError as error:
            issues.add("packet_materialization_time_invalid", event_hash, str(error))
        if access is None:
            issues.add(
                "packet_materialization_access_missing",
                event_hash,
                f"unknown access event {row['query_access_event_hash']!r}",
            )
        elif (
            access["execution_id"] != row["execution_id"]
            or access["snapshot_hash"] != row["snapshot_hash"]
            or access["release_class"] != row["release_class"]
        ):
            issues.add(
                "packet_materialization_lineage_mismatch",
                event_hash,
                "packet materialization crosses execution/snapshot/release lineage",
            )
        event_payload = artifact_json(
            subject=event_hash,
            artifact_hash=row["materialization_event_artifact_hash"],
            code_prefix="packet_materialization_event",
        )
        if event_payload is not None and event_payload.get("content_hash") != event_hash:
            issues.add(
                "packet_materialization_artifact_hash_mismatch",
                event_hash,
                "materialization artifact does not bind the event logical hash",
            )
        elif event_payload is not None and any(
            event_payload.get(field) != row[field]
            for field in (
                "execution_id",
                "snapshot_hash",
                "packet_hash",
                "query_access_event_hash",
            )
        ):
            issues.add(
                "packet_materialization_artifact_lineage_mismatch",
                event_hash,
                "packet-materialization artifact differs from its ledger row",
            )
        packet_payload = artifact_json(
            subject=event_hash,
            artifact_hash=row["packet_artifact_hash"],
            code_prefix="packet",
        )
        if packet_payload is not None and packet_payload.get(
            "content_hash",
            packet_payload.get("packet_hash"),
        ) != row["packet_hash"]:
            issues.add(
                "packet_artifact_hash_mismatch",
                event_hash,
                "packet artifact does not bind the registered packet logical hash",
            )
        elif packet_payload is not None and packet_payload.get("snapshot_hash") != row[
            "snapshot_hash"
        ]:
            issues.add(
                "packet_artifact_snapshot_mismatch",
                event_hash,
                "packet artifact does not bind the registered evidence snapshot",
            )
        for artifact_hash in (
            row["materialization_event_artifact_hash"],
            row["packet_artifact_hash"],
        ):
            artifact = artifacts.get(artifact_hash)
            if artifact is not None and artifact["release_class"] != row["release_class"]:
                issues.add(
                    "packet_materialization_release_mismatch",
                    event_hash,
                    "packet materialization and artifact release classes differ",
                )


def _verify_artifact_links(
    rows: list[sqlite3.Row],
    artifacts: dict[str, sqlite3.Row],
    jobs: dict[str, sqlite3.Row],
    attempts: dict[str, sqlite3.Row],
    issues: _Issues,
) -> None:
    parents: dict[str, set[str]] = {digest: set() for digest in artifacts}
    links_by_job: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        links_by_job[row["job_id"]].add(row["content_hash"])
    for row in rows:
        link_id = row["link_id"]
        job_id = row["job_id"]
        digest = row["content_hash"]
        attempt_id = row["attempt_id"]
        parent = row["parent_content_hash"]
        expected_link = hashlib.sha256(
            canonical_json(
                {
                    "job_id": job_id,
                    "content_hash": digest,
                    "role": row["role"],
                    "attempt_id": attempt_id,
                    "parent_content_hash": parent,
                }
            ).encode("utf-8")
        ).hexdigest()
        if link_id != expected_link:
            issues.add("artifact_link_id_mismatch", link_id, f"expected {expected_link}")
        job = jobs.get(job_id)
        artifact = artifacts.get(digest)
        if job is None:
            issues.add("artifact_link_job_missing", link_id, f"unknown job {job_id!r}")
        if artifact is None:
            issues.add("artifact_link_target_missing", link_id, f"unknown artifact {digest!r}")
        elif job is not None and job["release_class"] == "public" and artifact[
            "release_class"
        ] == "restricted":
            issues.add(
                "artifact_link_release_violation",
                link_id,
                "public job links restricted CAS",
            )
        if attempt_id is not None:
            attempt = attempts.get(attempt_id)
            if attempt is None:
                issues.add(
                    "artifact_link_attempt_missing", link_id, f"unknown attempt {attempt_id!r}"
                )
            elif attempt["job_id"] != job_id:
                issues.add(
                    "artifact_link_attempt_cross_job",
                    link_id,
                    "attempt belongs to another job",
                )
        if parent is not None:
            parents.setdefault(digest, set()).add(parent)
            if parent not in artifacts:
                issues.add("artifact_parent_missing", link_id, f"unknown parent {parent!r}")
            elif parent not in links_by_job.get(job_id, set()):
                issues.add(
                    "artifact_parent_not_linked_to_job",
                    link_id,
                    f"parent {parent!r} is not registered to job {job_id!r}",
                )
    for child, parent in _find_cycles(parents):
        issues.add("artifact_cycle", child, f"back-edge to {parent!r}")


def _verify_gpu_and_calls(
    connection: sqlite3.Connection,
    jobs: dict[str, sqlite3.Row],
    attempts: dict[str, sqlite3.Row],
    artifacts: dict[str, sqlite3.Row],
    issues: _Issues,
) -> dict[str, object]:
    events = {
        row["event_id"]: row
        for row in connection.execute("SELECT * FROM gpu_events ORDER BY event_id")
    }
    event_by_kind: dict[str, int] = defaultdict(int)
    event_total = 0
    for event_id, row in events.items():
        micros = row["allocated_microseconds"]
        event_total += micros
        event_by_kind[row["event_kind"]] += micros
        job_id = row["job_id"]
        attempt_id = row["attempt_id"]
        if job_id is not None and job_id not in jobs:
            issues.add("gpu_event_job_missing", event_id, f"unknown job {job_id!r}")
        if attempt_id is not None:
            attempt = attempts.get(attempt_id)
            if attempt is None:
                issues.add("gpu_event_attempt_missing", event_id, f"unknown attempt {attempt_id!r}")
            elif job_id != attempt["job_id"]:
                issues.add("gpu_event_attempt_cross_job", event_id, "attempt/job binding differs")
        try:
            if _parse_timestamp(row["ended_at"]) < _parse_timestamp(row["started_at"]):
                raise ValueError("end precedes start")
        except ValueError as exc:
            issues.add("gpu_event_time_invalid", event_id, str(exc))
        if not _json_mapping(row["details_json"]):
            issues.add("gpu_event_details_invalid", event_id, "details_json is not a JSON object")
        if row["event_kind"] in {"failure", "timeout"} and row["succeeded"] == 1:
            issues.add(
                "gpu_event_success_invalid",
                event_id,
                "failure/timeout event is marked successful",
            )

    sessions = {
        row["service_session_id"]: row
        for row in connection.execute(
            "SELECT * FROM gpu_service_sessions ORDER BY service_session_id"
        )
    }
    overhead_total = 0
    classified_total = 0
    for session_id, row in sessions.items():
        service = row["service_microseconds"]
        classified = row["classified_event_microseconds"]
        overhead = row["overhead_microseconds"]
        classified_total += classified
        overhead_total += overhead
        if service != classified + overhead:
            issues.add(
                "gpu_service_arithmetic_mismatch",
                session_id,
                "service duration does not equal classified plus overhead",
            )
        try:
            if _parse_timestamp(row["ended_at"]) < _parse_timestamp(row["started_at"]):
                raise ValueError("end precedes start")
        except ValueError as exc:
            issues.add("gpu_service_time_invalid", session_id, str(exc))
        if not _json_mapping(row["details_json"]):
            issues.add(
                "gpu_service_details_invalid",
                session_id,
                "details_json is not a JSON object",
            )
    if classified_total > event_total:
        issues.add(
            "gpu_classified_total_unreconciled",
            "gpu_service_sessions",
            f"classified={classified_total} exceeds all GPU events={event_total}",
        )

    calls = list(connection.execute("SELECT * FROM model_calls ORDER BY model_call_id"))
    for row in calls:
        call_id = row["model_call_id"]
        job_id = row["job_id"]
        attempt_id = row["attempt_id"]
        event_id = row["gpu_event_id"]
        job = jobs.get(job_id)
        attempt = attempts.get(attempt_id)
        if job is None:
            issues.add("model_call_job_missing", call_id, f"unknown job {job_id!r}")
        if attempt is None:
            issues.add("model_call_attempt_missing", call_id, f"unknown attempt {attempt_id!r}")
        elif attempt["job_id"] != job_id:
            issues.add("model_call_attempt_cross_job", call_id, "attempt belongs to another job")
        response_hash = row["response_artifact_hash"]
        if response_hash is not None:
            response = artifacts.get(response_hash)
            if response is None:
                issues.add(
                    "model_call_response_missing", call_id, f"unknown response {response_hash!r}"
                )
            elif job is not None and job["release_class"] == "public" and response[
                "release_class"
            ] == "restricted":
                issues.add(
                    "model_call_release_violation", call_id, "public call links restricted response"
                )
        if row["backend"] == "hand_authored_fixture":
            if event_id is not None or row["allocated_gpu_microseconds"] != 0:
                issues.add(
                    "fixture_gpu_claim",
                    call_id,
                    "hand-authored fixture claims a GPU event or allocation",
                )
            continue
        if row["backend"] != "vllm_gpu":
            issues.add("model_call_backend_invalid", call_id, f"unknown backend {row['backend']!r}")
            continue
        event = events.get(event_id)
        if event is None:
            issues.add("model_call_gpu_event_missing", call_id, f"unknown GPU event {event_id!r}")
            continue
        if event["job_id"] != job_id or event["attempt_id"] != attempt_id:
            issues.add(
                "model_call_gpu_binding_mismatch",
                call_id,
                "GPU event does not identify the call's exact job and attempt",
            )
        if event["allocated_microseconds"] != row["allocated_gpu_microseconds"]:
            issues.add(
                "model_call_gpu_duration_mismatch",
                call_id,
                "model call and GPU event allocations differ",
            )
        if event["succeeded"] is not None and event["succeeded"] != row["successful"]:
            issues.add(
                "model_call_gpu_success_mismatch", call_id, "model call and GPU event status differ"
            )

    unresolved_allocations = _verify_allocation_journals(
        connection, jobs, attempts, events, issues
    )
    unresolved_services = _verify_service_journals(connection, sessions, issues)
    total = event_total + overhead_total
    if overhead_total:
        event_by_kind["service_overhead"] += overhead_total
    if total >= DEFAULT_GPU_HARD_LIMIT_SECONDS * 1_000_000:
        issues.add(
            "gpu_hard_limit_reached",
            "gpu_summary",
            f"allocated {total} microseconds reaches/crosses the hard stop",
        )
    return {
        "event_count": len(events),
        "session_count": len(sessions),
        "call_count": len(calls),
        "total": total,
        "by_kind": tuple(sorted(event_by_kind.items())),
        "unresolved_allocations": unresolved_allocations,
        "unresolved_services": unresolved_services,
    }


def _verify_allocation_journals(
    connection: sqlite3.Connection,
    jobs: dict[str, sqlite3.Row],
    attempts: dict[str, sqlite3.Row],
    events: dict[str, sqlite3.Row],
    issues: _Issues,
) -> int:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM gpu_allocation_journal ORDER BY allocation_id, sequence"
    ):
        grouped[row["allocation_id"]].append(row)
    unresolved = 0
    for allocation_id, rows in grouped.items():
        first = rows[0]
        stable = (
            first["intended_event_kind"],
            first["maximum_microseconds"],
            first["job_id"],
            first["attempt_id"],
        )
        if first["sequence"] != 0 or first["state"] != "opened" or first[
            "elapsed_microseconds"
        ] != 0:
            issues.add(
                "gpu_allocation_journal_open_invalid",
                allocation_id,
                "journal must begin with sequence-zero opened/zero observation",
            )
        previous_elapsed = -1
        previous_time: dt.datetime | None = None
        for index, row in enumerate(rows):
            if row["sequence"] != index:
                issues.add(
                    "gpu_allocation_journal_sequence_gap", allocation_id, f"at row {index}"
                )
            if (
                row["intended_event_kind"],
                row["maximum_microseconds"],
                row["job_id"],
                row["attempt_id"],
            ) != stable:
                issues.add(
                    "gpu_allocation_journal_identity_changed", allocation_id, f"at sequence {index}"
                )
            if row["elapsed_microseconds"] < previous_elapsed:
                issues.add(
                    "gpu_allocation_journal_elapsed_regressed",
                    allocation_id,
                    f"at sequence {index}",
                )
            try:
                observed = _parse_timestamp(row["observed_at"])
                if previous_time is not None and observed < previous_time:
                    raise ValueError("observation time regressed")
                previous_time = observed
            except ValueError as exc:
                issues.add("gpu_allocation_journal_time_invalid", allocation_id, str(exc))
            previous_elapsed = row["elapsed_microseconds"]
            if index > 0 and row["state"] == "opened":
                issues.add(
                    "gpu_allocation_journal_reopened",
                    allocation_id,
                    f"at sequence {index}",
                )
            if index < len(rows) - 1 and row["state"] in _TERMINAL_ALLOCATION_STATES:
                issues.add(
                    "gpu_allocation_journal_after_terminal",
                    allocation_id,
                    f"terminal state appears at sequence {index}",
                )
            if not _json_mapping(row["details_json"]):
                issues.add(
                    "gpu_allocation_journal_details_invalid",
                    allocation_id,
                    f"sequence {index} details are not a JSON object",
                )
            contents = {
                "allocation_id": allocation_id,
                "sequence": row["sequence"],
                "state": row["state"],
                "intended_event_kind": row["intended_event_kind"],
                "elapsed_microseconds": row["elapsed_microseconds"],
                "maximum_microseconds": row["maximum_microseconds"],
                "observed_at": row["observed_at"],
                "job_id": row["job_id"],
                "attempt_id": row["attempt_id"],
                "details_json": row["details_json"],
            }
            expected_journal_id = hashlib.sha256(
                canonical_json(contents).encode("utf-8")
            ).hexdigest()
            if row["journal_id"] != expected_journal_id:
                issues.add(
                    "gpu_allocation_journal_id_mismatch",
                    row["journal_id"],
                    f"expected {expected_journal_id}",
                )
        job_id, attempt_id = first["job_id"], first["attempt_id"]
        if job_id is not None and job_id not in jobs:
            issues.add("gpu_allocation_job_missing", allocation_id, f"unknown job {job_id!r}")
        if attempt_id is not None:
            attempt = attempts.get(attempt_id)
            if attempt is None:
                issues.add(
                    "gpu_allocation_attempt_missing",
                    allocation_id,
                    f"unknown attempt {attempt_id!r}",
                )
            elif attempt["job_id"] != job_id:
                issues.add(
                    "gpu_allocation_attempt_cross_job", allocation_id, "attempt/job binding differs"
                )
        latest = rows[-1]
        if latest["state"] not in _TERMINAL_ALLOCATION_STATES:
            unresolved += 1
            continue
        event = events.get(allocation_id)
        if event is None:
            issues.add(
                "gpu_allocation_terminal_event_missing",
                allocation_id,
                "terminal allocation journal has no same-ID GPU event",
            )
            continue
        if event["allocated_microseconds"] != latest["elapsed_microseconds"]:
            issues.add(
                "gpu_allocation_terminal_duration_mismatch",
                allocation_id,
                "terminal journal and GPU event allocations differ",
            )
        if event["job_id"] != job_id or event["attempt_id"] != attempt_id:
            issues.add(
                "gpu_allocation_terminal_binding_mismatch",
                allocation_id,
                "terminal GPU event has different job/attempt lineage",
            )
        allowed_kinds = (
            {"failure"}
            if latest["state"] == "recovered"
            else {first["intended_event_kind"], "failure", "timeout"}
        )
        if event["event_kind"] not in allowed_kinds:
            issues.add(
                "gpu_allocation_terminal_kind_mismatch",
                allocation_id,
                f"expected one of {sorted(allowed_kinds)!r}, found {event['event_kind']!r}",
            )
    return unresolved


def _verify_service_journals(
    connection: sqlite3.Connection,
    sessions: dict[str, sqlite3.Row],
    issues: _Issues,
) -> int:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM gpu_service_journal ORDER BY service_session_id, sequence"
    ):
        grouped[row["service_session_id"]].append(row)
    unresolved = 0
    for service_id, rows in grouped.items():
        first = rows[0]
        stable = (
            first["session_id"],
            first["configuration_hash"],
            first["service_started_at"],
            first["ledger_allocated_microseconds_before_session"],
            first["hard_limit_microseconds"],
        )
        if first["sequence"] != 0 or first["state"] != "opened" or first[
            "elapsed_microseconds"
        ] != 0:
            issues.add(
                "gpu_service_journal_open_invalid",
                service_id,
                "journal must begin with sequence-zero opened/zero observation",
            )
        previous_elapsed = -1
        previous_time: dt.datetime | None = None
        allowed_transitions = {
            "opened": {"heartbeat", "process_stopped", "recovered"},
            "heartbeat": {"heartbeat", "process_stopped", "recovered"},
            "process_stopped": {"closed", "recovered"},
        }
        for index, row in enumerate(rows):
            if row["sequence"] != index:
                issues.add("gpu_service_journal_sequence_gap", service_id, f"at row {index}")
            if (
                row["session_id"],
                row["configuration_hash"],
                row["service_started_at"],
                row["ledger_allocated_microseconds_before_session"],
                row["hard_limit_microseconds"],
            ) != stable:
                issues.add(
                    "gpu_service_journal_identity_changed",
                    service_id,
                    f"at sequence {index}",
                )
            if row["elapsed_microseconds"] < previous_elapsed:
                issues.add(
                    "gpu_service_journal_elapsed_regressed",
                    service_id,
                    f"at sequence {index}",
                )
            try:
                observed = _parse_timestamp(row["observed_at"])
                if previous_time is not None and observed < previous_time:
                    raise ValueError("observation time regressed")
                previous_time = observed
            except ValueError as exc:
                issues.add("gpu_service_journal_time_invalid", service_id, str(exc))
            previous_elapsed = row["elapsed_microseconds"]
            if index > 0 and row["state"] == "opened":
                issues.add("gpu_service_journal_reopened", service_id, f"at sequence {index}")
            if index > 0 and row["state"] not in allowed_transitions.get(
                rows[index - 1]["state"], set()
            ):
                issues.add(
                    "gpu_service_journal_transition_invalid",
                    service_id,
                    f"{rows[index - 1]['state']}->{row['state']} at sequence {index}",
                )
            if index < len(rows) - 1 and row["state"] in _TERMINAL_SERVICE_STATES:
                issues.add(
                    "gpu_service_journal_after_terminal",
                    service_id,
                    f"terminal state appears at sequence {index}",
                )
            if not _json_mapping(row["details_json"]):
                issues.add(
                    "gpu_service_journal_details_invalid",
                    service_id,
                    f"sequence {index} details are not a JSON object",
                )
            contents = {
                "service_session_id": service_id,
                "sequence": row["sequence"],
                "state": row["state"],
                "session_id": row["session_id"],
                "configuration_hash": row["configuration_hash"],
                "service_started_at": row["service_started_at"],
                "elapsed_microseconds": row["elapsed_microseconds"],
                "ledger_allocated_microseconds_before_session": row[
                    "ledger_allocated_microseconds_before_session"
                ],
                "hard_limit_microseconds": row["hard_limit_microseconds"],
                "observed_at": row["observed_at"],
                "details_json": row["details_json"],
            }
            expected_journal_id = hashlib.sha256(
                canonical_json(contents).encode("utf-8")
            ).hexdigest()
            if row["journal_id"] != expected_journal_id:
                issues.add(
                    "gpu_service_journal_id_mismatch",
                    row["journal_id"],
                    f"expected {expected_journal_id}",
                )
        latest = rows[-1]
        if latest["state"] not in _TERMINAL_SERVICE_STATES:
            unresolved += 1
            continue
        session = sessions.get(service_id)
        if session is None:
            issues.add(
                "gpu_service_terminal_session_missing",
                service_id,
                "terminal service journal has no accounting row",
            )
            continue
        if (
            session["session_id"] != first["session_id"]
            or session["started_at"] != first["service_started_at"]
        ):
            issues.add(
                "gpu_service_terminal_identity_mismatch",
                service_id,
                "journal and accounting session identities differ",
            )
        if session["service_microseconds"] != latest["elapsed_microseconds"]:
            issues.add(
                "gpu_service_terminal_duration_mismatch",
                service_id,
                "terminal journal and accounting service durations differ",
            )
        if (
            first["ledger_allocated_microseconds_before_session"]
            + session["service_microseconds"]
            >= first["hard_limit_microseconds"]
        ):
            issues.add(
                "gpu_service_hard_limit_reached",
                service_id,
                "session accounting reaches/crosses its registered hard stop",
            )
    return unresolved


def _verify_storage(
    connection: sqlite3.Connection,
    jobs: dict[str, sqlite3.Row],
    issues: _Issues,
) -> tuple[int, int, int]:
    samples = list(connection.execute("SELECT * FROM storage_samples ORDER BY sample_id"))
    peak_projected = 0
    for row in samples:
        sample_id = row["sample_id"]
        current = row["current_occupied_bytes"]
        additional = row["additional_reserved_bytes"]
        projected = row["projected_occupied_bytes"]
        filesystem_free = row["filesystem_free_bytes"]
        allocation_free = DEFAULT_TOTAL_ALLOCATION_BYTES - projected
        projected_filesystem_free = filesystem_free - additional
        effective = min(allocation_free, projected_filesystem_free)
        peak_projected = max(peak_projected, projected)
        expected_violations: list[str] = []
        if projected > DEFAULT_MAX_OCCUPIED_BYTES:
            expected_violations.append("projected_occupancy_exceeds_limit")
        if filesystem_free < DEFAULT_MIN_HEADROOM_BYTES:
            expected_violations.append("actual_filesystem_headroom_below_minimum")
        if allocation_free < DEFAULT_MIN_HEADROOM_BYTES:
            expected_violations.append("projected_allocation_headroom_below_minimum")
        if projected_filesystem_free < DEFAULT_MIN_HEADROOM_BYTES:
            expected_violations.append("projected_filesystem_headroom_below_minimum")
        try:
            violations = json.loads(row["violations_json"])
        except (TypeError, ValueError):
            violations = None
        if projected != current + additional:
            issues.add(
                "storage_projection_mismatch",
                sample_id,
                "projected occupied bytes do not equal current plus reserved",
            )
        if row["effective_projected_headroom_bytes"] != effective:
            issues.add(
                "storage_headroom_mismatch", sample_id, f"expected effective headroom {effective}"
            )
        if violations != expected_violations:
            issues.add(
                "storage_violations_mismatch",
                sample_id,
                f"expected {expected_violations!r}, found {violations!r}",
            )
        if bool(row["allowed"]) != (not expected_violations):
            issues.add("storage_allowed_mismatch", sample_id, "allowed flag disagrees with limits")
        expected_id = hashlib.sha256(
            canonical_json(
                {
                    "phase": row["phase"],
                    "sampled_at": row["sampled_at"],
                    "current": current,
                    "additional": additional,
                    "projected": projected,
                    "filesystem_free": filesystem_free,
                    "headroom": row["effective_projected_headroom_bytes"],
                    "allowed": bool(row["allowed"]),
                    "violations": tuple(violations) if isinstance(violations, list) else violations,
                }
            ).encode("utf-8")
        ).hexdigest()
        if sample_id != expected_id:
            issues.add("storage_sample_id_mismatch", sample_id, f"expected {expected_id}")

    resources = list(connection.execute("SELECT * FROM resource_samples ORDER BY sample_id"))
    resource_events = {
        row["event_id"]: row
        for row in connection.execute("SELECT * FROM gpu_events ORDER BY event_id")
    }
    for row in resources:
        sample_id = row["sample_id"]
        job_id = row["job_id"]
        event_id = row["gpu_event_id"]
        if job_id is not None and job_id not in jobs:
            issues.add("resource_sample_job_missing", sample_id, f"unknown job {job_id!r}")
        if event_id is None:
            continue
        event = resource_events.get(event_id)
        if event is None:
            issues.add(
                "resource_sample_gpu_event_missing",
                sample_id,
                f"unknown GPU event {event_id!r}",
            )
        elif job_id is not None and event["job_id"] not in (None, job_id):
            issues.add(
                "resource_sample_gpu_cross_job",
                sample_id,
                "resource sample and GPU event belong to different jobs",
            )
    peak_recorded = max((row["project_storage_bytes"] for row in resources), default=0)
    peak_recorded = max(
        peak_recorded,
        max((row["current_occupied_bytes"] for row in samples), default=0),
    )
    return len(samples), peak_recorded, peak_projected


def audit_ledger(ledger_path: Path, blob_root: Path) -> LedgerVerificationReport:
    """Run every ledger/CAS check and return all issues without writing state."""

    database = Path(ledger_path).resolve()
    blobs = Path(blob_root).resolve()
    issues = _Issues()
    empty: dict[str, object] = {
        "schema_versions": (),
        "artifact_count": 0,
        "artifact_raw_bytes": 0,
        "artifact_stored_bytes": 0,
        "job_count": 0,
        "attempt_count": 0,
        "artifact_link_count": 0,
        "model_call_count": 0,
        "gpu_event_count": 0,
        "gpu_service_session_count": 0,
        "gpu_total_allocated_microseconds": 0,
        "gpu_by_kind_microseconds": (),
        "unresolved_gpu_allocation_count": 0,
        "unresolved_gpu_service_count": 0,
        "storage_sample_count": 0,
        "peak_recorded_project_storage_bytes": 0,
        "peak_projected_storage_bytes": 0,
        "legacy_validation_scope_count": 0,
    }
    try:
        connection = _readonly_connection(database)
    except (OSError, sqlite3.Error) as exc:
        issues.add("sqlite_open_failed", database, str(exc))
        return LedgerVerificationReport(**empty, issues=issues.ordered())  # type: ignore[arg-type]

    try:
        try:
            integrity_rows = [row[0] for row in connection.execute("PRAGMA integrity_check")]
            if integrity_rows != ["ok"]:
                for result in integrity_rows:
                    issues.add("sqlite_integrity_failed", database.name, str(result))
        except sqlite3.Error as exc:
            issues.add("sqlite_integrity_check_failed", database.name, str(exc))

        try:
            for row in connection.execute("PRAGMA foreign_key_check"):
                issues.add(
                    "sqlite_foreign_key_failed",
                    f"{row[0]}:{row[1]}",
                    f"missing parent table={row[2]!r}, constraint={row[3]}",
                )
        except sqlite3.Error as exc:
            issues.add("sqlite_foreign_key_check_failed", database.name, str(exc))

        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for missing in sorted(_REQUIRED_TABLES - tables):
            issues.add("schema_table_missing", missing, "required verification table is absent")
        if _REQUIRED_TABLES - tables:
            return LedgerVerificationReport(
                **empty,
                issues=issues.ordered(),
            )  # type: ignore[arg-type]

        versions = tuple(
            row["schema_version"]
            for row in connection.execute(
                "SELECT schema_version FROM schema_metadata ORDER BY schema_version"
            )
        )
        empty["schema_versions"] = versions
        if not versions or versions[-1] != SCHEMA_VERSION:
            issues.add(
                "schema_version_unsupported",
                database.name,
                f"expected latest schema version {SCHEMA_VERSION}; found {versions!r}",
            )

        artifact_rows = list(connection.execute("SELECT * FROM artifacts ORDER BY content_hash"))
        raw_total, stored_total = _verify_artifacts(artifact_rows, blobs, issues)
        artifacts = {row["content_hash"]: row for row in artifact_rows}
        job_rows = list(connection.execute("SELECT * FROM jobs ORDER BY job_id"))
        jobs = {row["job_id"]: row for row in job_rows}
        job_states = _verify_job_lifecycles(connection, jobs, issues)
        attempt_rows = list(connection.execute("SELECT * FROM attempts ORDER BY attempt_id"))
        attempts = _verify_attempts(attempt_rows, set(jobs), issues)
        legacy_scope_count = _verify_validations_and_projections(
            connection, attempts, artifacts, blobs, issues
        )
        _verify_boundary_and_failures(
            connection,
            jobs=jobs,
            attempts=attempts,
            artifacts=artifacts,
            blob_root=blobs,
            issues=issues,
        )
        _verify_scientific_lineage(
            connection,
            jobs=jobs,
            job_states=job_states,
            artifacts=artifacts,
            issues=issues,
        )
        link_rows = list(connection.execute("SELECT * FROM job_artifacts ORDER BY link_id"))
        _verify_artifact_links(link_rows, artifacts, jobs, attempts, issues)
        gpu = _verify_gpu_and_calls(connection, jobs, attempts, artifacts, issues)
        storage_count, peak_recorded, peak_projected = _verify_storage(
            connection,
            jobs,
            issues,
        )

        empty.update(
            {
                "artifact_count": len(artifact_rows),
                "artifact_raw_bytes": raw_total,
                "artifact_stored_bytes": stored_total,
                "job_count": len(job_rows),
                "attempt_count": len(attempt_rows),
                "artifact_link_count": len(link_rows),
                "model_call_count": gpu["call_count"],
                "gpu_event_count": gpu["event_count"],
                "gpu_service_session_count": gpu["session_count"],
                "gpu_total_allocated_microseconds": gpu["total"],
                "gpu_by_kind_microseconds": gpu["by_kind"],
                "unresolved_gpu_allocation_count": gpu["unresolved_allocations"],
                "unresolved_gpu_service_count": gpu["unresolved_services"],
                "storage_sample_count": storage_count,
                "peak_recorded_project_storage_bytes": peak_recorded,
                "peak_projected_storage_bytes": peak_projected,
                "legacy_validation_scope_count": legacy_scope_count,
            }
        )
    except sqlite3.Error as exc:
        issues.add("sqlite_scan_failed", database.name, str(exc))
    finally:
        connection.close()
    return LedgerVerificationReport(**empty, issues=issues.ordered())  # type: ignore[arg-type]


def verify_ledger(ledger_path: Path, blob_root: Path) -> LedgerVerificationReport:
    """Return a valid report or raise with the complete invalid report attached."""

    report = audit_ledger(ledger_path, blob_root)
    if not report.valid:
        raise LedgerVerificationError(report)
    return report


__all__ = [
    "LedgerVerificationError",
    "LedgerVerificationIssue",
    "LedgerVerificationReport",
    "audit_ledger",
    "verify_ledger",
]
