#!/usr/bin/env python3
"""Append a restricted receipt for the V10 recovery and CPU capacity work only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.ledger_verify import verify_ledger
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.output_capacity_gate import BASELINE_SECONDS, capacity_forecast


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.recovery_root.resolve(strict=True)
    if "restricted" not in root.parts or "restricted" not in args.output.parts:
        raise ValueError("restricted inputs and output required")
    if args.output.exists():
        raise FileExistsError("receipt is append-only")
    ledger = root / "artifacts/restricted/phase1_acceptance.sqlite"
    terminal = json.loads(
        (root / "artifacts/restricted/v10_validation/terminal-verification.json").read_bytes()
    )
    assert terminal["manifest_sha256"] == canonical_sha256(
        {k: v for k, v in terminal.items() if k != "manifest_sha256"}
    )
    assert sha(ledger) == terminal["ledger_sha256"]
    verification = verify_ledger(ledger, root / "artifacts/blobs/phase1_acceptance")
    assert verification.valid
    assert verification.gpu_total_allocated_microseconds / 1e6 == BASELINE_SECONDS
    preserved_v9 = Path("artifacts/restricted/phase1_acceptance.sqlite")
    assert sha(preserved_v9) == "53e4341b14f53dce48482cd765eb4654777df76061ff4948e0ea9bcf8a93dc5c"
    capacity_path = root / "artifacts/restricted/output-capacity-v9/capacity.json"
    capacity = json.loads(capacity_path.read_bytes())
    for relative, digest in capacity["cpu_job"]["source_sha256"].items():
        assert sha(Path(relative)) == digest, "measured source differs from checkpoint source"
    assert all(row["candidate"]["fits"] for row in capacity["production_requests"])
    assert capacity["intact_c1_fixedselect_stress"]["fits"]
    calibration_path = root / "artifacts/restricted/c0-calibration-v4/calibration.json"
    calibration = json.loads(calibration_path.read_bytes())
    for name, digest in calibration["files"].items():
        assert sha(calibration_path.parent / name) == digest
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        calls = [
            dict(row)
            for row in connection.execute(
                "SELECT model_call_id, call_role, successful FROM model_calls ORDER BY created_at"
            )
        ]
    tests = []
    for name in ("output_capacity_unit_final_v5.xml", "output_capacity_http_final_v4.xml"):
        path = Path("artifacts/restricted") / name
        suites = [node.attrib for node in ET.parse(path).getroot()]
        assert all(row["errors"] == row["failures"] == "0" for row in suites)
        tests.append({"path": str(path), "sha256": sha(path), "suites": suites})
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("no symlink traversal in restricted recovery receipt")
        if path.is_file():
            stat = path.stat()
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": stat.st_size,
                    "sha256": sha(path),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                    "git_status": "ignored_restricted_recovery_copy",
                    "release_class": "restricted",
                    "apparent_phase": "V10_or_CPU_development_capacity",
                }
            )
    forecast = capacity_forecast(terminal["remaining_inventory_rows"])
    forecast["fallback_acceptance_earmarks_already_in_remaining_reserves"] = {
        "fallback-c2-01": ["reserve_standard", 150],
        "fallback-c2-02": ["reserve_standard", 150],
        "fallback-fixed-01": ["reserve_short", 90],
    }
    forecast["new_c1_diagnostics_charged_to_new_recovery_block_not_reserve_long"] = True
    payload = {
        "kind": "restricted_cpu_output_capacity_repair_receipt",
        "recorded_at": datetime.now(UTC).isoformat(),
        "recovery_inventory": inventory,
        "authoritative_v10_ledger_sha256": sha(ledger),
        "preserved_stale_v9_ledger_sha256": sha(preserved_v9),
        "ledger_verification": verification.to_dict(),
        "historical_model_calls": calls,
        "new_gpu_service_starts": 0,
        "new_gpu_generation_attempts": 0,
        "additional_gpu_allocated_seconds": 0,
        "capacity_measurement_sha256": sha(capacity_path),
        "production_capacity": capacity["production_requests"],
        "intact_c1_fixedselect_stress": capacity["intact_c1_fixedselect_stress"],
        "forecast": forecast,
        "c0_calibration": calibration,
        "tests": tests,
        "candidate_activated": False,
        "complete_acceptance_passed": False,
        "held_out_authorized": False,
    }
    write_json_atomic({**payload, "manifest_sha256": canonical_sha256(payload)}, args.output)
    print(json.dumps({"receipt": str(args.output), "forecast": forecast["all_in_seconds"]}))


if __name__ == "__main__":
    main()
