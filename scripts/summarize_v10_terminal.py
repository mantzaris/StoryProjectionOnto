#!/usr/bin/env python3
"""Verify V10 on its owning host and emit aggregates, never response content.

No service start, inference, ledger mutation, or restricted network transfer.
The absent ordinary runner result is disclosed, not reconstructed as a success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import zstandard

from story_projection_onto import bounded_recovery as recovery
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.ledger_verify import verify_ledger
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.store import ReadOnlyLedger


def hashed_object(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    assert value["manifest_sha256"] == canonical_sha256(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    ), path.name
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-fixture-tokens", action="store_true")
    options = parser.parse_args()
    root = options.root.resolve(strict=True)
    if options.output.exists():
        raise FileExistsError("terminal summaries are append-only")
    run_root = root / "artifacts/restricted/fallback-development-v10"
    ledger_path = root / "artifacts/restricted/phase1_acceptance.sqlite"
    guardian = hashed_object(run_root / "checkpoint.json.guardian-result.json")
    assert guardian["physical_shutdown_verified"] is True
    assert guardian["run_id"] == recovery.RUN_ID
    checkpoint = json.loads((run_root / "checkpoint.json").read_bytes())
    assert checkpoint["completed_call_ids"] == []
    assert checkpoint["failed_call_id"] == "fallback-c1-01"
    with ReadOnlyLedger(ledger_path) as ledger:
        recovery.validate_history(root, ledger)
        assert not ledger.unresolved_gpu_allocations()
        assert not ledger.unresolved_gpu_service_journals()
        events = [e for e in ledger.gpu_events() if e.event_id.startswith(recovery.RUN_ID + "-")]
        sessions = [s for s in ledger.gpu_service_sessions() if s.session_id == recovery.RUN_ID]
        assert len(events) == 2 and len(sessions) == 1
        summary = ledger.gpu_summary()
    session = sessions[0]
    assert canonical_sha256(asdict(session)) == guardian["service_session_accounting_sha256"]
    assert summary.total_allocated_seconds == guardian["actual_allocated_gpu_seconds"]
    call = next(e for e in events if "fallback-c1-01-gpu" in e.event_id)
    startup = next(e for e in events if "service-start-001" in e.event_id)
    details = json.loads(call.details_json)
    assert details["request_hash"] == recovery.REQUEST_HASH
    assert details["reserve_call_class"] == "reserve_long"
    assert (
        session.service_microseconds
        == sum(e.allocated_microseconds for e in events) + session.overhead_microseconds
    )

    diagnostic_root = root / "artifacts/restricted/http_diagnostics" / recovery.RUN_ID
    logs = list(diagnostic_root.glob("*/events.jsonl"))
    assert len(logs) == 1
    journal = [json.loads(line) for line in logs[0].read_bytes().splitlines()]
    response = bytearray()
    for event in journal:
        if event["event"] == "response_fragment":
            artifact = event["artifact"]
            relative = Path(artifact["relative_path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            stored = (logs[0].parent / "fragments" / relative).read_bytes()
            raw = zstandard.ZstdDecompressor().decompress(stored)
            assert hashlib.sha256(raw).hexdigest() == artifact["content_hash"]
            assert event["offset"] == len(response)
            response.extend(raw)
    complete = next(e for e in journal if e["event"] == "response_complete")
    assert complete["complete"] and complete["received_bytes"] == len(response)
    assert hashlib.sha256(response).hexdigest() == complete["response_sha256"]
    envelope = json.loads(response)
    metadata = next(e for e in journal if e["event"] == "completion_metadata")
    failure = next(e for e in journal if e["event"] == "failure")
    assert metadata["finish_reason"] == "length"
    assert metadata["usage"]["completion_tokens"] == 2048
    assert failure["stage"] == "decoding"
    try:
        json.loads(envelope["choices"][0]["message"]["content"])
    except json.JSONDecodeError as error:
        json_error = {
            "message": error.msg,
            "line": error.lineno,
            "column": error.colno,
            "character": error.pos,
        }
    else:
        raise AssertionError("captured V10 C1 was expected to be incomplete JSON")
    stages = [
        hashed_object(p)
        for p in sorted((run_root / "checkpoint.json.recovery-stages").glob("*.json"))
    ]
    assert [s["stage"] for s in stages] == ["startup", "live_checks", "c1_retry", "shutdown"]
    durations = {
        stages[i]["stage"]: stages[i + 1]["monotonic"] - stages[i]["monotonic"]
        for i in range(len(stages) - 1)
    }
    durations["shutdown"] = (
        datetime.fromisoformat(session.ended_at) - datetime.fromisoformat(stages[-1]["recorded_at"])
    ).total_seconds()
    assert all(0 <= seconds < recovery.CAPS[name] for name, seconds in durations.items())
    assert session.service_seconds < recovery.TOTAL
    verification = verify_ledger(ledger_path, root / "artifacts/blobs/phase1_acceptance")
    assert verification.valid, verification.issues
    rows = recovery.prior_result(root)["post_fallback_full_manifest_forecast"]["rows"]
    current_load_proxy = max(
        recovery.PRIOR_LOAD_PROXY, startup.allocated_seconds + session.overhead_seconds
    )
    for row in rows:
        if row["call_class"] == "reserve_long":
            row["consumed_count"] += 1
            row["remaining_count"] -= 1
        if row["call_class"] == "gpu_session_start":
            row["forecast_p95_seconds"] = current_load_proxy
        row["remaining_forecast_seconds"] = row["remaining_count"] * row["forecast_p95_seconds"]
    remaining = math.fsum(row["remaining_forecast_seconds"] for row in rows)
    actual = summary.total_allocated_seconds
    connection = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    resources = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM resource_samples WHERE sample_id LIKE ?", (recovery.RUN_ID + "%",)
        )
    ]
    failures = [
        dict(row)
        for row in connection.execute(
            "SELECT failure_kind, message FROM failures WHERE attempt_id LIKE ?",
            (recovery.RUN_ID + "%",),
        )
    ]
    assert len(failures) == 1 and failures[0]["failure_kind"] == "invalid_output"
    peaks = {
        key: max((row[key] for row in resources), default=0)
        for key in (
            "gpu_vram_bytes",
            "process_ram_bytes",
            "project_storage_bytes",
            "cpu_worker_count",
        )
    }
    connection.close()
    fixture_counts = None
    if options.count_fixture_tokens:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from transformers import AutoTokenizer

        snapshot = (
            root
            / ".cache/shared/hub/models--Qwen--Qwen3-8B-AWQ/snapshots"
            / "4da05a8edb55c6046cce958586c33b61da07bb79"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False
        )
        fixture = json.loads((root / "tests/fixtures/phase1/c1_pre_output.json").read_bytes())
        fixture_counts = {
            "authored_fixture_not_model_output": True,
            "raw_fixture_compact_tokens": len(
                tokenizer.encode(
                    json.dumps(fixture, separators=(",", ":")), add_special_tokens=False
                )
            ),
            "raw_fixture_pretty_tokens": len(
                tokenizer.encode(json.dumps(fixture, indent=2), add_special_tokens=False)
            ),
            "raw_fixture_hashes_not_replaced": True,
            "cannot_establish_required_completion_length": True,
        }
    payload = {
        "kind": "bounded_recovery_v10_terminal_diagnostic",
        "run_id": recovery.RUN_ID,
        "recorded_at": datetime.now(UTC).isoformat(),
        "executed_source_commit": json.loads((run_root / "launch.json").read_bytes())["git_commit"],
        "summary_script_sha256": recovery.file_hash(Path(__file__)),
        "ledger_sha256": recovery.file_hash(ledger_path),
        "guardian_manifest_sha256": guardian["manifest_sha256"],
        "response_sha256": complete["response_sha256"],
        "response_bytes": len(response),
        "http_status": 200,
        "response_complete": True,
        "request_sha256": recovery.REQUEST_HASH,
        "finish_reason": metadata["finish_reason"],
        "diagnostic_usage": metadata["usage"],
        "usage_not_backfilled_into_immutable_failed_call": True,
        "failure_stage": failure["stage"],
        "json_error": json_error,
        "accepted_outputs": 0,
        "new_attempts": 1,
        "new_repairs": 0,
        "development_calls": 0,
        "complete_acceptance_passed": False,
        "startup_seconds": startup.allocated_seconds,
        "failed_call_seconds": call.allocated_seconds,
        "service_overhead_seconds": session.overhead_seconds,
        "new_service_seconds": session.service_seconds,
        "actual_allocated_seconds": actual,
        "stage_wall_seconds": durations,
        "validation_stage_reached": False,
        "cpu_preallocation_seconds": hashed_object(
            run_root / "checkpoint.json.cpu-preparation.json"
        )["preparation_seconds"],
        "resource_peaks_this_attempt": peaks,
        "ledger_verification": verification.to_dict(),
        "remaining_inventory_rows": rows,
        "remaining_forecast_seconds": remaining,
        "all_in_seconds": actual + remaining,
        "scheduled_limit_seconds": 33660,
        "scheduled_reserve_seconds": 33660 - actual - remaining,
        "strict_hard_seconds": 36000,
        "remaining_actual_hard_seconds_exclusive": 36000 - actual,
        "future_load_proxy_seconds": current_load_proxy,
        "failed_output_throughput_credited": False,
        "hypothetical_same_envelope_next_attempt_all_in_seconds": actual + remaining - 240 + 840,
        "hypothetical_next_attempt_authorized": False,
        "standard_runner_result_published": (
            root / "artifacts/public/results/fallback_gpu_acceptance_development_v10.json"
        ).exists(),
        "guardian_takeover_trigger": guardian["trigger"],
        "physical_shutdown_verified": True,
        "resume_authorized": False,
        "restricted_artifacts_remain_on_owning_host": True,
        "local_restricted_transfer_approved": False,
        "fixture_token_counts": fixture_counts,
        "v9_exact_failure_still_unknown": True,
    }
    write_json_atomic({**payload, "manifest_sha256": canonical_sha256(payload)}, options.output)
    print(
        json.dumps(
            {
                "verified": True,
                "actual": actual,
                "remaining": remaining,
                "all_in": actual + remaining,
                "peaks": peaks,
                "fixture": fixture_counts,
            }
        )
    )


if __name__ == "__main__":
    main()
