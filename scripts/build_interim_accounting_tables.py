#!/usr/bin/env python3
"""Regenerate interim resource/failure tables from immutable pilot results.

This script is deliberately limited to observed-to-date accounting. It does not
create scientific efficacy rows or claim that the study has completed.
"""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-v1",
        type=Path,
        default=Path("artifacts/public/results/phase1_gpu_acceptance_v1_failed.json"),
    )
    parser.add_argument(
        "--pilot-v2",
        type=Path,
        default=Path("artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports/tables"))
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("gate_passed") is not False or value.get("vllm_service_stopped") is not True:
        raise ValueError(f"interim pilot artifact lacks rejected/stopped status: {path}")
    return value


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _seconds(microseconds: int) -> str:
    return f"{Decimal(microseconds) / Decimal(1_000_000):.6f}"


def build_tables(pilot_v1: Path, pilot_v2: Path, output_root: Path) -> None:
    first = _load(pilot_v1)
    second = _load(pilot_v2)
    first_accounting = first["runtime"]["gpu_accounting"]
    second_accounting = second["runtime"]["gpu_accounting"]
    total_microseconds = int(second_accounting["total_allocated_microseconds"])
    if total_microseconds < int(first_accounting["total_allocated_microseconds"]):
        raise ValueError("cumulative v2 GPU accounting regresses below v1")
    samples = second["runtime"]["resource_samples"]
    resource_rows: list[list[object]] = [
        [
            "primary_14b_pilot_to_date",
            "actual_allocated_gpu_time",
            _seconds(total_microseconds),
            "seconds",
            "observed",
            "Includes both failed service-start allocations recorded in the cumulative ledger.",
        ],
        [
            "primary_14b_pilot_to_date",
            "actual_allocated_gpu_time",
            f"{Decimal(total_microseconds) / Decimal(3_600_000_000):.9f}",
            "hours",
            "observed",
            "Derived exactly from cumulative allocated microseconds for display only.",
        ],
        [
            "primary_14b_pilot_v2",
            "peak_gpu_vram",
            max(int(item["gpu_vram_bytes"]) for item in samples),
            "bytes",
            "observed",
            "Maximum persisted resource sample in the rejected v2 run.",
        ],
        [
            "primary_14b_pilot_v2",
            "peak_process_ram",
            max(int(item["process_ram_bytes"]) for item in samples),
            "bytes",
            "observed",
            "Maximum persisted resource sample in the rejected v2 run.",
        ],
        [
            "primary_14b_pilot_v2",
            "peak_project_storage",
            max(int(item["project_storage_bytes"]) for item in samples),
            "bytes",
            "observed",
            "Maximum persisted resource sample in the rejected v2 run.",
        ],
        [
            "study",
            "current_vllm_service_stopped",
            "true",
            "boolean",
            "observed",
            "The rejected primary pilot result records that the service stopped.",
        ],
        [
            "study",
            "final_scheduled_gpu_forecast",
            "",
            "hours",
            "incomplete",
            "Requires the accepted fallback micro-pilot plus exact development timing block.",
        ],
        [
            "study",
            "final_total_runpod_wall_time",
            "",
            "hours",
            "incomplete",
            "No final pod-session accounting artifact exists yet.",
        ],
    ]
    _write_csv(
        output_root / "resource_accounting.csv",
        ["scope", "metric", "value", "unit", "status", "source_note"],
        resource_rows,
    )
    failures: list[list[object]] = []
    for payload in (first, second):
        by_kind = payload["runtime"]["gpu_accounting"]["by_kind_microseconds"]
        incremental = sum(
            int(value) for key, value in by_kind.items() if key in {"failure", "timeout"}
        )
        if payload is second:
            incremental -= int(first_accounting["by_kind_microseconds"].get("failure", 0))
        failures.append(
            [
                payload["run_id"],
                payload["runtime"]["launcher"]["repository"],
                payload["runtime"]["launcher"]["revision"],
                payload["failure_type"],
                payload["completed_call_count"],
                "false",
                "true",
                _seconds(incremental),
                "Rejected primary-model lifecycle start; not a scientific model output.",
            ]
        )
    _write_csv(
        output_root / "failure_accounting.csv",
        [
            "run_id",
            "model_repository",
            "immutable_revision",
            "failure_type",
            "completed_generation_calls",
            "gate_passed",
            "vllm_service_stopped",
            "allocated_gpu_seconds",
            "interpretation",
        ],
        failures,
    )


def main() -> int:
    args = parse_args()
    build_tables(args.pilot_v1, args.pilot_v2, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
