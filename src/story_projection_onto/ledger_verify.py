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

from story_projection_onto.store import (
    DEFAULT_GPU_HARD_LIMIT_SECONDS,
    DEFAULT_MAX_OCCUPIED_BYTES,
    DEFAULT_MIN_HEADROOM_BYTES,
    DEFAULT_TOTAL_ALLOCATION_BYTES,
    SCHEMA_VERSION,
    canonical_json,
)

_SHA256_LENGTH = 64
_REQUIRED_TABLES = frozenset(
    {
        "schema_metadata",
        "jobs",
        "attempts",
        "artifacts",
        "job_artifacts",
        "gpu_events",
        "gpu_allocation_journal",
        "gpu_service_journal",
        "gpu_service_sessions",
        "model_calls",
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
        attempt_rows = list(connection.execute("SELECT * FROM attempts ORDER BY attempt_id"))
        attempts = _verify_attempts(attempt_rows, set(jobs), issues)
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
