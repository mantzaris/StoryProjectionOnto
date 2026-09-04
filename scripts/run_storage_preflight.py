#!/usr/bin/env python3
"""Measure, record, and enforce the study's complete writable-storage envelope."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from story_projection_onto.experiment import ResourceLimits, StorageAllocationPlan
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
    parser.add_argument(
        "--allocation-plan",
        type=Path,
        default=Path("configs/study/storage_phase_allocations.json"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    project_root = arguments.project_root.resolve(strict=True)
    limits_path = project_root / "configs" / "study" / "resource_limits.json"
    limits = ResourceLimits.load(limits_path)
    budget = StorageBudget(
        total_allocation_bytes=limits.maximum_project_allocation_bytes,
        max_occupied_bytes=limits.maximum_project_occupied_bytes,
        min_headroom_bytes=limits.minimum_storage_headroom_bytes,
    )
    preflight = StoragePreflight(project_root, budget=budget)
    plan_path = arguments.allocation_plan
    if not plan_path.is_absolute():
        plan_path = project_root / plan_path
    plan_path.resolve(strict=True).relative_to(project_root)
    reservation = StorageAllocationPlan.load(plan_path).reservation_for(arguments.phase)
    report = preflight.check(**reservation.preflight_arguments())
    ledger_path = arguments.ledger
    if not ledger_path.is_absolute():
        ledger_path = project_root / ledger_path
    ledger_path.resolve().relative_to(project_root)
    with Ledger(ledger_path) as ledger:
        sample_id = ledger.record_storage_sample(report, phase=arguments.phase)
    payload = {**asdict(report), "phase": arguments.phase, "sample_id": sample_id}
    print(json.dumps(payload, sort_keys=True))
    return 0 if report.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
