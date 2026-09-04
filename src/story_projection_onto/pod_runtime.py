"""Public-safe RunPod container wall-time accounting.

Scientific GPU allocation is authoritative in the SQLite ledger.  This module
records the separate lifetime of PID 1 in the current Linux container as a
transparent operational lower/upper context; it never treats that duration as
GPU-service time or as a billing-system timestamp.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from story_projection_onto.contracts import canonical_sha256


class PodRuntimeMeasurementError(RuntimeError):
    """The current Linux process namespace could not be measured safely."""


def _boot_epoch_seconds(proc_root: Path) -> int:
    try:
        lines = (proc_root / "stat").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PodRuntimeMeasurementError("cannot read Linux boot time") from exc
    values = [line.split()[1] for line in lines if line.startswith("btime ")]
    if len(values) != 1 or not values[0].isdecimal():
        raise PodRuntimeMeasurementError("Linux proc stat has no unique boot epoch")
    return int(values[0])


def _pid_one_start_ticks(proc_root: Path) -> int:
    try:
        value = (proc_root / "1" / "stat").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise PodRuntimeMeasurementError("cannot read container PID 1 start ticks") from exc
    # comm is parenthesized and may itself contain spaces or parentheses.  The
    # final ') ' precedes field 3, making field 22 index 19 in this suffix.
    boundary = value.rfind(") ")
    suffix = value[boundary + 2 :].split() if boundary >= 0 else []
    if len(suffix) <= 19 or not suffix[19].isdecimal():
        raise PodRuntimeMeasurementError("container PID 1 stat record is malformed")
    return int(suffix[19])


def capture_pod_wall_time(
    *,
    proc_root: Path = Path("/proc"),
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    clock_ticks_per_second: int | None = None,
) -> dict[str, object]:
    """Measure current-container PID-1 lifetime without invoking GPU APIs."""

    ticks_per_second = (
        os.sysconf("SC_CLK_TCK")
        if clock_ticks_per_second is None
        else clock_ticks_per_second
    )
    if isinstance(ticks_per_second, bool) or not isinstance(ticks_per_second, int):
        raise PodRuntimeMeasurementError("clock tick frequency must be an integer")
    if ticks_per_second <= 0:
        raise PodRuntimeMeasurementError("clock tick frequency must be positive")
    sampled_at = clock()
    if sampled_at.tzinfo is None or sampled_at.utcoffset() is None:
        raise PodRuntimeMeasurementError("pod wall-time clock must be timezone-aware")
    sampled_at = sampled_at.astimezone(UTC)
    started_at = datetime.fromtimestamp(_boot_epoch_seconds(proc_root), tz=UTC) + timedelta(
        seconds=_pid_one_start_ticks(proc_root) / ticks_per_second
    )
    elapsed_microseconds = int((sampled_at - started_at).total_seconds() * 1_000_000)
    if elapsed_microseconds < 0:
        raise PodRuntimeMeasurementError("container PID 1 appears to start in the future")
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "runpod_container_wall_time_sample",
        "scope": "current_container_pid_1_lifetime",
        "container_pid": 1,
        "container_pid_1_started_at": started_at.isoformat(),
        "sampled_at": sampled_at.isoformat(),
        "elapsed_microseconds": elapsed_microseconds,
        "measurement_method": "linux_proc_boot_epoch_plus_pid1_start_ticks",
        "scientific_gpu_accounting": False,
        "billing_time_equivalence_claimed": False,
        "caveat": (
            "PID-1 lifetime can differ from provider session and billable pod lifetime; "
            "scientific GPU-service time remains the append-only ledger total."
        ),
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


__all__ = [
    "PodRuntimeMeasurementError",
    "capture_pod_wall_time",
]
