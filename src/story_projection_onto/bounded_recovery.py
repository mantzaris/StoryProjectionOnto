"""One explicitly authorized V10 recovery; no new scientific inference inventory.

The old V9 overlay remains immutable and terminal. This module binds the new
resource amendment to its terminal evidence, not to V9's exhausted permission.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.manifest import load_source_manifest, write_json_atomic

RUN_ID = "fallback-qwen3-8b-awq-development-v10"
V9_ID = "fallback-qwen3-8b-awq-development-v9"
CAPS = {
    "startup": 300,
    "live_checks": 120,
    "c1_retry": 240,
    "validation_drain": 120,
    "shutdown": 60,
}
TOTAL = 840
PRIOR_ACTUAL = 2936.238858
PRIOR_LOAD_PROXY = 333.6688687896926
REQUEST_HASH = "190ee0e2f382bc4d356e5ce24e543ad7cd963aae0ad596ba5a4c310c447a29d5"
RESULT_HASH = "dbd5c1f5defccba571879548b152de59ffd59e90e5e1a6a6ea6f22eede2a03dc"
LEDGER_HASH = "53e4341b14f53dce48482cd765eb4654777df76061ff4948e0ea9bcf8a93dc5c"
RECOVERY_IDS = tuple(
    f"fallback-qwen3-8b-awq-development-v{v}-service-start-001" for v in (3, 7, 8, 9, 10)
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_result(root: Path) -> dict:
    path = root / "artifacts/public/results/fallback_gpu_acceptance_development_v9.json"
    if file_hash(path) != RESULT_HASH:
        raise ValueError("V9 terminal result changed")
    result = json.loads(path.read_bytes())
    if result["vllm_service_stopped"] is not True:
        raise ValueError("V9 did not stop")
    return result


def load_authorization(path: Path, *, root: Path, run_id: str) -> dict:
    expected = root / "configs/study/bounded_recovery_v10.json"
    if path.resolve() != expected.resolve() or path.is_symlink():
        raise ValueError("bounded recovery requires the exact amendment file")
    value = json.loads(path.read_bytes())
    required = {
        "kind": "user_authorized_resource_feasibility_amendment",
        "run_id": RUN_ID,
        "authorized_by": "user",
        "original_scheduled_seconds": 32400,
        "amended_scheduled_seconds": 33660,
        "strict_hard_seconds": 36000,
        "original_nine_hour_target_met": False,
        "authorized_service_starts": 1,
        "additional_inference_slots": 0,
        "earmarked_existing_reserve_long_slots": 1,
        "stage_caps_seconds": CAPS,
        "whole_recovery_seconds": TOTAL,
        "model_revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
        "request_sha256": REQUEST_HASH,
        "historical_allocated_seconds": PRIOR_ACTUAL,
        "historical_ledger_sha256": LEDGER_HASH,
        "historical_result_sha256": RESULT_HASH,
    }
    if run_id != RUN_ID or any(
        type(value.get(k)) is not type(v) or value[k] != v for k, v in required.items()
    ):
        raise ValueError("bounded recovery authority or caps changed")
    limits = json.loads((root / "configs/study/resource_limits.json").read_bytes())
    if limits["scheduled_gpu_seconds"] != 33660 or limits["hard_gpu_seconds"] != 36000:
        raise ValueError("resource amendment is not the approved 33660/36000 pair")
    prior_result(root)
    return {**value, "manifest_sha256": canonical_sha256(value)}


def validate_history(root: Path, ledger, *, before_start: bool = False) -> None:
    events = ledger.gpu_events()
    prior = [e for e in events if not e.event_id.startswith(RUN_ID + "-")]
    # Compare the historical rows to the preserved, hash-pinned terminal ledger.
    from dataclasses import asdict

    from story_projection_onto.store import ReadOnlyLedger

    preserved = (
        root
        / "artifacts/restricted/recovery_validation/v9_terminal_20260906T2313Z"
        / "phase1_acceptance.sqlite"
    )
    if not preserved.is_file() or file_hash(preserved) != LEDGER_HASH:
        raise ValueError("hash-pinned V9 ledger snapshot is missing or changed")
    with ReadOnlyLedger(preserved) as historical:
        if [asdict(e) for e in prior] != [asdict(e) for e in historical.gpu_events()]:
            raise ValueError("historical GPU event rows changed")
        prior_sessions = [s for s in ledger.gpu_service_sessions() if s.session_id != RUN_ID]
        if [asdict(s) for s in prior_sessions] != [
            asdict(s) for s in historical.gpu_service_sessions()
        ]:
            raise ValueError("historical service allocation rows changed")
    if before_start and (
        len(events) != len(prior)
        or ledger.unresolved_gpu_allocations()
        or ledger.unresolved_gpu_service_journals()
    ):
        raise ValueError("V10 already allocated or predecessor unresolved")
    if (
        sum(e.allocated_microseconds for e in prior)
        + sum(s.overhead_microseconds for s in prior_sessions)
    ) != 2936238858:
        raise ValueError("historical allocation must never reset")


def validate_frozen_science(root: Path) -> None:
    """Reuse unchanged V9 science; permit only the declared engineering edits."""
    manifest = (
        root / "artifacts/public/manifests/source_tree_fallback_second_recovery_v9.local.json"
    )
    allowed = {
        "configs/study/resource_limits.json",
        "src/story_projection_onto/gpu_runtime.py",
        "src/story_projection_onto/fallback_acceptance.py",
        "src/story_projection_onto/phase1_acceptance.py",
        "src/story_projection_onto/development_artifacts.py",
        "src/story_projection_onto/development_continuation.py",
        "tests/unit/test_fallback_acceptance.py",
        "tests/unit/test_experiment.py",
    }
    for entry in load_source_manifest(manifest)["files"]:
        if entry["path"] not in allowed and file_hash(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"unapproved scientific/source change: {entry['path']}")


def prelaunch_forecast(root: Path, actual: float) -> dict:
    rows = prior_result(root)["post_fallback_full_manifest_forecast"]["rows"]
    remaining = math.fsum(row["remaining_forecast_seconds"] for row in rows)
    reserve = next(row for row in rows if row["call_class"] == "reserve_long")
    if reserve["remaining_count"] != 2 or reserve["forecast_p95_seconds"] != 240:
        raise ValueError("earmarked reserve is not available in the full inventory")
    all_in = actual + remaining - 240 + TOTAL
    result = {
        "actual_allocated_seconds": actual,
        "mandatory_rows": rows,
        "remaining_registered_seconds": remaining,
        "earmark_subtraction_seconds": 240,
        "recovery_envelope_seconds": TOTAL,
        "all_in_seconds": all_in,
        "scheduled_seconds": 33660,
        "scheduled_reserve_seconds": 33660 - all_in,
        "hard_margin_seconds": 36000 - all_in,
        "admitted": all_in <= 33660 and actual + TOTAL < 36000,
    }
    if not result["admitted"]:
        raise RuntimeError(f"bounded recovery admission failed: {all_in:.9f} > 33660")
    return result


def diagnostic_storage_probe(root: Path) -> dict:
    """Persist a labeled CPU fixture in the actual restricted diagnostics store."""
    from story_projection_onto.http_diagnostics import RestrictedResponseJournal

    journal = RestrictedResponseJournal(
        root / "artifacts/restricted/http_diagnostics/v10-cpu-preflight",
        request_hash=REQUEST_HASH,
    )
    body = b'{"cpu_storage_probe_only":true}'
    journal.headers(200, {"Content-Type": "application/json"})
    journal.fragment(body)
    journal.complete()
    events = [
        json.loads(line) for line in (journal.path / "events.jsonl").read_bytes().splitlines()
    ]
    if [event["event"] for event in events] != [
        "request",
        "response_headers",
        "response_fragment",
        "response_complete",
    ]:
        raise ValueError("diagnostic storage probe did not persist complete evidence")
    record = events[2]["artifact"]
    from story_projection_onto.store import ArtifactRecord, Compression, ReleaseClass

    record["compression"] = Compression(record["compression"])
    record["release_class"] = ReleaseClass(record["release_class"])
    if journal.blobs.read_bytes(ArtifactRecord(**record), allow_restricted=True) != body:
        raise ValueError("diagnostic fragment round-trip failed")
    return {
        "cpu_fixture_only": True,
        "gpu_calls": 0,
        "bytes_round_tripped": len(body),
        "event_count": len(events),
        "response_complete": journal.response_complete,
    }


class RecoveryDeadlineExceeded(RuntimeError):
    pass


class RecoveryGate:
    """Append-only stage records shared with the independent service guardian.

    Linux monotonic time is shared by all pod processes. A boot ID prevents a
    reboot from resetting these deadlines. A one-second early trigger leaves
    room for the guardian poll; every shutdown retains a full 60-second slot.
    """

    def __init__(self, checkpoint: Path, authorization_hash: str, *, clock=time.monotonic):
        self.root = checkpoint.with_name(checkpoint.name + ".recovery-stages")
        self.authorization_hash = authorization_hash
        self.clock = clock

    def records(self) -> list[dict]:
        records = []
        if not self.root.exists():
            return records
        if self.root.is_symlink():
            raise ValueError("recovery stage directory is symlinked")
        for path in sorted(self.root.glob("*.json")):
            value = json.loads(path.read_bytes())
            payload = {k: v for k, v in value.items() if k != "manifest_sha256"}
            if (
                path.is_symlink()
                or value["manifest_sha256"] != canonical_sha256(payload)
                or value["authorization_sha256"] != self.authorization_hash
                or value["run_id"] != RUN_ID
                or value["sequence"] != len(records) + 1
                or value["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            ):
                raise ValueError("recovery stage identity changed")
            records.append(value)
        return records

    def deadline_reason(self) -> str | None:
        records = self.records()
        if not records:
            return None
        latest = records[-1]
        if latest["stage"] in {"accepted", "stopped"}:
            return None
        now = self.clock()
        if now < latest["monotonic"]:
            return "recovery_clock_regression"
        shutdown = latest["stage"] == "shutdown"
        # Once C1 is accepted, ordinary continuation is not capped at 840s;
        # its eventual shutdown still has its own 60s limit.
        accepted = any(record["stage"] == "accepted" for record in records)
        # Give the cooperative controller 10s, then leave 50s for guardian
        # takeover, TERM/KILL and verification within the 60s shutdown cap.
        deadline = min(
            latest["monotonic"] + (10 if shutdown else CAPS[latest["stage"]]),
            math.inf if accepted else records[0]["monotonic"] + TOTAL - (0 if shutdown else 60),
        )
        return f"recovery_{latest['stage']}_deadline" if now >= deadline - 1 else None

    def transition(self, stage: str, *, details: Mapping | None = None) -> None:
        records = self.records()
        previous = None if not records else records[-1]["stage"]
        allowed = {
            None: {"startup"},
            "startup": {"live_checks", "shutdown"},
            "live_checks": {"c1_retry", "shutdown"},
            "c1_retry": {"validation_drain", "shutdown"},
            "validation_drain": {"accepted", "shutdown"},
            "accepted": {"shutdown"},
            "shutdown": {"stopped"},
            "stopped": set(),
        }
        if stage not in allowed[previous]:
            raise ValueError(f"illegal recovery transition {previous} -> {stage}")
        reason = self.deadline_reason()
        if reason and stage not in {"shutdown", "stopped"}:
            raise RecoveryDeadlineExceeded(reason)
        if stage == "accepted" and (
            details is None
            or details.get("c1_valid") is not True
            or details.get("fresh_admission") is not True
        ):
            raise ValueError("recovery release requires C1 validation and fresh admission")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "run_id": RUN_ID,
            "authorization_sha256": self.authorization_hash,
            "sequence": len(records) + 1,
            "stage": stage,
            "monotonic": self.clock(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "recorded_at": datetime.now(UTC).isoformat(),
            "details": dict(details or {}),
        }
        path = self.root / f"{len(records) + 1:02d}-{stage}.json"
        if path.exists():
            raise FileExistsError(path)
        write_json_atomic({**payload, "manifest_sha256": canonical_sha256(payload)}, path)

    def begin_shutdown(self) -> None:
        records = self.records()
        if records and records[-1]["stage"] not in {"shutdown", "stopped"}:
            self.transition("shutdown")
