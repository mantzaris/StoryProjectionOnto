#!/usr/bin/env python3
"""Regenerate interim resource/failure tables from immutable pilot results.

This script is deliberately limited to observed-to-date accounting. It does not
create scientific efficacy rows or claim that the study has completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

from story_projection_onto.fallback_control_plane_incident import (
    load_fallback_control_plane_incident,
)


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
    parser.add_argument(
        "--fallback-v1",
        type=Path,
        default=Path(
            "artifacts/public/results/"
            "fallback_gpu_acceptance_development_v1.json.controller-handoff.json"
        ),
    )
    parser.add_argument(
        "--fallback-v3",
        type=Path,
        default=Path(
            "artifacts/public/results/fallback_gpu_acceptance_development_v3.json"
        ),
    )
    parser.add_argument(
        "--fallback-v3-incident",
        type=Path,
        default=Path(
            "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v3_incident.json"
        ),
    )
    parser.add_argument(
        "--fallback-v4-control-plane-incident",
        type=Path,
        default=Path(
            "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v4_control_plane_incident.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports/tables"))
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_self_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = value.get("manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if supplied_hash != _canonical_sha256(immutable):
        raise ValueError(f"interim source has an invalid canonical self-hash: {path}")
    return value


def _load_rejected_result(path: Path) -> dict[str, Any]:
    value = _load_self_hashed(path)
    if value.get("gate_passed") is not False or value.get("vllm_service_stopped") is not True:
        raise ValueError(f"interim pilot artifact lacks rejected/stopped status: {path}")
    return value


def _load_v3_incident(path: Path, fallback_v3_path: Path) -> dict[str, Any]:
    value = _load_self_hashed(path)
    terminal = value.get("terminal_state", {})
    diagnosis = value.get("diagnosis", {})
    if (
        terminal.get("gate_passed") is not False
        or terminal.get("vllm_service_stopped") is not True
        or terminal.get("accepted_output_count") != 0
        or diagnosis.get("inference_call_reached_generation") is not False
    ):
        raise ValueError(f"fallback-v3 incident lacks its rejected terminal state: {path}")
    provenance = value.get("provenance", {}).get("failed_result", {})
    if (
        provenance.get("file_sha256")
        != hashlib.sha256(fallback_v3_path.read_bytes()).hexdigest()
    ):
        raise ValueError("fallback-v3 incident does not bind the supplied failed result")
    return value


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _seconds(microseconds: int) -> str:
    return f"{Decimal(microseconds) / Decimal(1_000_000):.6f}"


def build_tables(
    pilot_v1: Path,
    pilot_v2: Path,
    fallback_v1: Path,
    fallback_v3: Path,
    fallback_v3_incident: Path,
    fallback_v4_control_plane_incident: Path,
    output_root: Path,
) -> None:
    first = _load_rejected_result(pilot_v1)
    second = _load_rejected_result(pilot_v2)
    third = _load_rejected_result(fallback_v1)
    fourth = _load_rejected_result(fallback_v3)
    incident = _load_v3_incident(fallback_v3_incident, fallback_v3)
    control_plane_incident = load_fallback_control_plane_incident(
        fallback_v4_control_plane_incident
    )
    first_accounting = first["runtime"]["gpu_accounting"]
    second_accounting = second["runtime"]["gpu_accounting"]
    third_accounting = third["runtime"]["gpu_accounting"]
    fourth_accounting = fourth["runtime"]["gpu_accounting"]
    cumulative_microseconds = [
        int(item["total_allocated_microseconds"])
        for item in (
            first_accounting,
            second_accounting,
            third_accounting,
            fourth_accounting,
        )
    ]
    if cumulative_microseconds != sorted(cumulative_microseconds) or len(
        set(cumulative_microseconds)
    ) != len(cumulative_microseconds):
        raise ValueError("cumulative GPU accounting is not strictly increasing")
    incident_accounting = incident["accounting"]
    total_microseconds = cumulative_microseconds[-1]
    if (
        int(incident_accounting["cumulative_gpu_microseconds"]) != total_microseconds
        or int(incident_accounting["prior_gpu_microseconds"])
        != cumulative_microseconds[-2]
        or int(incident_accounting["v3_service_microseconds"])
        != cumulative_microseconds[-1] - cumulative_microseconds[-2]
    ):
        raise ValueError("fallback-v3 incident disagrees with cumulative GPU accounting")
    control_plane_summary = control_plane_incident.accounting.after.summary
    v3_terminal = incident["terminal_state"]
    if (
        control_plane_incident.accounting.before
        != control_plane_incident.accounting.after
        or control_plane_summary.total_allocated_microseconds != total_microseconds
        or control_plane_summary.event_count
        != int(incident_accounting["gpu_event_count"])
        or control_plane_summary.service_session_count
        != int(incident_accounting["gpu_service_session_count"])
        or control_plane_summary.model_call_count
        != int(incident_accounting["model_call_count"])
        or control_plane_summary.unresolved_gpu_allocation_count
        != int(v3_terminal["unresolved_gpu_allocation_count"])
        or control_plane_summary.unresolved_gpu_service_count
        != int(v3_terminal["unresolved_gpu_service_count"])
        or control_plane_summary.by_kind_microseconds
        != {
            str(kind): int(microseconds)
            for kind, microseconds in fourth_accounting["by_kind_microseconds"].items()
        }
    ):
        raise ValueError(
            "fallback-v4 control-plane incident changes cumulative GPU accounting"
        )
    v3_provenance = incident["provenance"]["failed_result"]
    if v3_provenance.get("manifest_sha256") != fourth.get("manifest_sha256"):
        raise ValueError("fallback-v3 incident names another logical failed result")
    resource_maxima = incident["resource_maxima"]
    if int(resource_maxima.get("resource_violation_count", -1)) != 0:
        raise ValueError("fallback-v3 incident reports a resource violation")
    samples = [
        sample
        for payload in (first, second, third, fourth)
        for sample in payload["runtime"]["resource_samples"]
    ]
    observed_sample_maxima = {
        field: max((int(item[field]) for item in samples), default=0)
        for field in ("gpu_vram_bytes", "process_ram_bytes", "project_storage_bytes")
    }
    if any(
        int(resource_maxima[field]) < observed_sample_maxima[field]
        for field in observed_sample_maxima
    ):
        raise ValueError("fallback-v3 incident resource maximum regresses below a source sample")
    resource_rows: list[list[object]] = [
        [
            "study_through_fallback_v3",
            "actual_allocated_gpu_time",
            _seconds(total_microseconds),
            "seconds",
            "observed",
            "Exact cumulative allocation through the rejected fallback-v3 attempt; "
            "includes all four service sessions and every failure, timeout, startup, "
            "and service-overhead event.",
        ],
        [
            "study_through_fallback_v3",
            "actual_allocated_gpu_time",
            f"{Decimal(total_microseconds) / Decimal(3_600_000_000):.9f}",
            "hours",
            "observed",
            "Derived exactly from cumulative allocated microseconds for display only.",
        ],
        [
            "study_through_fallback_v3",
            "peak_gpu_vram",
            int(resource_maxima["gpu_vram_bytes"]),
            "bytes",
            "observed",
            "Recovered cumulative maximum authenticated by the fallback-v3 incident record.",
        ],
        [
            "study_through_fallback_v3",
            "peak_process_ram",
            int(resource_maxima["process_ram_bytes"]),
            "bytes",
            "observed",
            "Recovered cumulative maximum authenticated by the fallback-v3 incident record.",
        ],
        [
            "study_through_fallback_v3",
            "peak_project_storage",
            int(resource_maxima["project_storage_bytes"]),
            "bytes",
            "observed",
            "Recovered cumulative maximum authenticated by the fallback-v3 incident record.",
        ],
        [
            "study",
            "current_vllm_service_stopped",
            "true",
            "boolean",
            "observed",
            "The fallback-v3 terminal result and incident record both verify that "
            "the service stopped.",
        ],
        [
            "study",
            "final_scheduled_gpu_forecast",
            "",
            "hours",
            "incomplete",
            "The v3 incident invalidates its published forecast; a corrected "
            "gate-admitted timing result is still required.",
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
    increments = [
        cumulative_microseconds[0],
        *(
            current - prior
            for prior, current in pairwise(cumulative_microseconds)
        ),
    ]
    if sum(increments) != total_microseconds or any(item <= 0 for item in increments):
        raise ValueError("incremental failure allocations do not reconcile to cumulative time")
    failures: list[list[object]] = []
    failure_payloads = (
        (
            first,
            first["failure_type"],
            int(first["completed_call_count"]),
            "Rejected primary-model lifecycle start; not a scientific model output.",
        ),
        (
            second,
            second["failure_type"],
            int(second["completed_call_count"]),
            "Rejected primary-model lifecycle start; not a scientific model output.",
        ),
        (
            third,
            third["failure_type"],
            int(third["completed_base_call_count"]),
            "Rejected fallback-model watchdog start; not a scientific model output.",
        ),
        (
            fourth,
            incident["diagnosis"]["exception_type"],
            int(incident["terminal_state"]["accepted_output_count"]),
            "Fallback-v3 request was rejected by decoder-schema validation before "
            "generation; no scientific model output was accepted.",
        ),
    )
    for (payload, failure_type, completed_generation_calls, interpretation), incremental in zip(
        failure_payloads, increments, strict=True
    ):
        failures.append(
            [
                payload["run_id"],
                payload["runtime"]["launcher"]["repository"],
                payload["runtime"]["launcher"]["revision"],
                failure_type,
                completed_generation_calls,
                "false",
                "true",
                _seconds(incremental),
                interpretation,
            ]
        )
    failures.append(
        [
            control_plane_incident.run_id,
            "not_applicable",
            "not_applicable",
            control_plane_incident.failure.safe_error_class,
            control_plane_incident.terminal_state.accepted_output_count,
            "false",
            "true",
            _seconds(control_plane_incident.accounting.delta.allocated_gpu_microseconds),
            "Operational control-plane failure before guardian readiness; no model "
            "process, service start, GPU allocation, inference attempt, authorized "
            "retry slot, or scientific generation.",
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
    build_tables(
        args.pilot_v1,
        args.pilot_v2,
        args.fallback_v1,
        args.fallback_v3,
        args.fallback_v3_incident,
        args.fallback_v4_control_plane_incident,
        args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
