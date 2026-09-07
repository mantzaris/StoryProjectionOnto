#!/usr/bin/env python3
"""Recompute the bounded diagnostic outcome from immutable restricted records."""

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import zstandard

from story_projection_onto.ledger_verify import verify_ledger
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.output_capacity_gate import BASELINE_SECONDS, capacity_forecast


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--v10-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or "restricted" not in args.output.parts:
        raise ValueError("new restricted receipt required")
    ledger_path = args.recovery / "artifacts/restricted/phase1_acceptance.sqlite"
    block = args.recovery / "artifacts/restricted/output-capacity-recovery-v1"
    verified = verify_ledger(ledger_path, args.recovery / "artifacts/blobs/phase1_acceptance")
    assert verified.valid
    total = verified.gpu_total_allocated_microseconds / 1e6
    diagnostics = []
    for path in sorted(block.glob("run-*/capacity-diagnostic-*/outcome.json")):
        outcome = json.loads(path.read_bytes())
        if outcome["accepted"]:
            raise ValueError("successful diagnostics require a fresh class-specific forecast")
        matches = list(
            path.parent.parent.glob("http/" + outcome["request_hash"][:16] + "-*/events.jsonl")
        )
        assert len(matches) == 1
        journal = matches[0]
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        chunks = []
        for event in events:
            if event["event"] == "response_fragment":
                record = event["artifact"]
                raw = zstandard.ZstdDecompressor().decompress(
                    (journal.parent / "fragments" / record["relative_path"]).read_bytes()
                )
                assert hashlib.sha256(raw).hexdigest() == record["content_hash"]
                assert event["offset"] == sum(map(len, chunks))
                chunks.append(raw)
        body = b"".join(chunks)
        complete = next(e for e in events if e["event"] == "response_complete")
        assert hashlib.sha256(body).hexdigest() == complete["response_sha256"]
        response = json.loads(body)
        choice = response["choices"][0]
        text = choice["message"]["content"]
        diagnostics.append(
            {
                "attempt_id": outcome["attempt_id"],
                "condition": outcome["condition"],
                "request_hash": outcome["request_hash"],
                "configuration_hash": outcome["configuration_hash"],
                "accepted": outcome["accepted"],
                "usage": response["usage"],
                "finish_reason": choice["finish_reason"],
                "http_complete": True,
                "http_status": next(
                    e["http_status"] for e in events if e["event"] == "response_headers"
                ),
                "failure_stage": next(
                    e["stage"] for e in reversed(events) if e["event"] == "failure"
                ),
                "generation_seconds": outcome["generation_event_seconds"],
                "content_characters": len(text),
                "whitespace_characters": sum(c.isspace() for c in text),
                "newlines": text.count("\n"),
                "prefix": text[:120],
                "response_sha256": complete["response_sha256"],
                "schema_and_scientific_validation": "not_reached",
            }
        )
    inventory = json.loads(args.v10_terminal.read_bytes())["remaining_inventory_rows"]
    forecast = capacity_forecast(inventory)
    # The diagnostic service is stopped and did not complete acceptance/resume.
    # Retain one future acceptance/resume service envelope in addition to the
    # five registered main-study loads. The proxy ALREADY includes service
    # overhead (fallback_acceptance.py's startup + complement convention).
    extra_load = next(
        r["forecast_p95_seconds"] for r in inventory if r["call_class"] == "gpu_session_start"
    )
    remaining = forecast["remaining_forecast_seconds"] + extra_load
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        peaks = dict(
            connection.execute(
                "SELECT max(process_ram_bytes) process_tree_ram_bytes, "
                "max(gpu_vram_bytes) gpu_vram_bytes, "
                "max(project_storage_bytes) project_storage_bytes, "
                "max(cpu_worker_count) workers FROM resource_samples "
                "WHERE sample_id LIKE 'run-20260907%'"
            ).fetchone()
        )
        events = [
            dict(r)
            for r in connection.execute(
                "SELECT event_id,event_kind,allocated_microseconds,"
                "started_at,ended_at FROM gpu_events WHERE event_id LIKE 'capacity-%'"
            )
        ]
    load = math.fsum(
        e["allocated_microseconds"] / 1e6 for e in events if e["event_kind"] == "gpu_session_start"
    )
    generation = math.fsum(d["generation_seconds"] for d in diagnostics)
    result = {
        "kind": "bounded_feasibility_diagnostics_not_acceptance",
        "ledger_sha256": sha(ledger_path),
        "ledger_verification": verified.to_dict(),
        "diagnostics": diagnostics,
        "accepted_by_condition": {"C1": 0, "C2": 0, "A-FixedSelect": 0},
        "attempted_by_condition": {"C1": len(diagnostics), "C2": 0, "A-FixedSelect": 0},
        "actual_allocated_seconds": total,
        "preserved_pre_block_seconds": BASELINE_SECONDS,
        "block_allocated_seconds": total - BASELINE_SECONDS,
        "block_remaining_seconds": 1200 - (total - BASELINE_SECONDS),
        "block_start_count": len(list(block.glob("start-*.json"))),
        "block_attempt_count": len(list(block.glob("attempt-*.json"))),
        "measured_load_seconds": load,
        "measured_generation_seconds": generation,
        "measured_other_service_seconds": total - BASELINE_SECONDS - load - generation,
        "resource_peaks": peaks,
        "gpu_events": events,
        "forecast": {
            "inventory_rows": forecast["rows"],
            "unchanged_inventory_remaining_seconds": forecast["remaining_forecast_seconds"],
            "additional_pending_acceptance_resume_service_envelope": extra_load,
            "all_remaining_seconds": remaining,
            "all_in_seconds": total + remaining,
            "scheduled_deficit_seconds": total + remaining - 33660,
            "hard_deficit_seconds": total + remaining - 36000,
            "valid_generation_latency_samples": 0,
            "valid_completion_p95_established": False,
            "generation_proxy": (
                "unmeasured allowance-ratio sensitivity capped at unchanged watchdogs"
            ),
            "loads_include_overhead": True,
            "invalid_throughput_credited": False,
            "mandatory_generation_plus_reserve_slots": sum(
                r["remaining_count"] for r in inventory if r["call_class"] != "gpu_session_start"
            ),
            "acceptance_earmarks_inside_existing_reserves": {
                "C1": 240,
                "C2": 300,
                "A-FixedSelect": 90,
            },
        },
        "ordinary_admission": False,
        "held_out_review_gate_retained": True,
        "restricted_inventory": [
            {"path": str(p.relative_to(args.recovery)), "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(args.recovery.rglob("*"))
            if p.is_file()
        ],
    }
    write_json_atomic(result, args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "actual_allocated_seconds",
                    "block_allocated_seconds",
                    "block_start_count",
                    "block_attempt_count",
                )
            }
        )
    )
    print(
        json.dumps(
            {
                k: result["forecast"][k]
                for k in ("all_remaining_seconds", "all_in_seconds", "scheduled_deficit_seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
