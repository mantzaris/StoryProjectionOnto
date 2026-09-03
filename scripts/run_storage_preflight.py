#!/usr/bin/env python3
"""Measure, record, and enforce the study's complete writable-storage envelope."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from story_projection_onto.store import Ledger, StorageBudget, StoragePreflight


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("artifacts/restricted/study.sqlite"),
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument("--declared-growth-bytes", type=int, default=0)
    parser.add_argument("--largest-atomic-temporary-bytes", type=int, default=0)
    parser.add_argument("--quarantine-allowance-bytes", type=int, default=0)
    parser.add_argument("--release-staging-bytes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    project_root = arguments.project_root.resolve(strict=True)
    limits_path = project_root / "configs" / "study" / "resource_limits.json"
    limits = json.loads(limits_path.read_text(encoding="utf-8"))
    budget = StorageBudget(
        total_allocation_bytes=limits["maximum_project_allocation_bytes"],
        max_occupied_bytes=limits["maximum_project_occupied_bytes"],
        min_headroom_bytes=limits["minimum_storage_headroom_bytes"],
    )
    preflight = StoragePreflight(project_root, budget=budget)
    report = preflight.require(
        declared_growth_bytes=arguments.declared_growth_bytes,
        largest_atomic_temporary_bytes=arguments.largest_atomic_temporary_bytes,
        quarantine_allowance_bytes=arguments.quarantine_allowance_bytes,
        release_staging_bytes=arguments.release_staging_bytes,
    )
    ledger_path = arguments.ledger
    if not ledger_path.is_absolute():
        ledger_path = project_root / ledger_path
    ledger_path.resolve().relative_to(project_root)
    with Ledger(ledger_path) as ledger:
        sample_id = ledger.record_storage_sample(report, phase=arguments.phase)
    payload = {**asdict(report), "phase": arguments.phase, "sample_id": sample_id}
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
