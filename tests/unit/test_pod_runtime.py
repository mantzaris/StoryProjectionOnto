from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.pod_runtime import (
    PodRuntimeMeasurementError,
    capture_pod_wall_time,
)


def _proc_fixture(root: Path, *, boot: int = 1_700_000_000, ticks: int = 250) -> Path:
    root.mkdir()
    (root / "1").mkdir()
    (root / "stat").write_text(f"cpu  1 2 3\nbtime {boot}\n", encoding="ascii")
    fields_after_comm = ["S", *("0" for _ in range(18)), str(ticks), "0"]
    (root / "1" / "stat").write_text(
        "1 (container init) " + " ".join(fields_after_comm) + "\n",
        encoding="ascii",
    )
    return root


def test_pod_wall_time_is_separate_hash_bound_operational_accounting(
    tmp_path: Path,
) -> None:
    proc = _proc_fixture(tmp_path / "proc")
    sampled = datetime.fromtimestamp(1_700_000_012, tz=UTC)

    result = capture_pod_wall_time(
        proc_root=proc,
        clock=lambda: sampled,
        clock_ticks_per_second=100,
    )

    assert result["elapsed_microseconds"] == 9_500_000
    assert result["scientific_gpu_accounting"] is False
    assert result["billing_time_equivalence_claimed"] is False
    payload = {key: value for key, value in result.items() if key != "manifest_sha256"}
    assert result["manifest_sha256"] == canonical_sha256(payload)


def test_pod_wall_time_rejects_malformed_or_future_process_state(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "1").mkdir()
    (malformed / "stat").write_text("cpu 1\n", encoding="ascii")
    (malformed / "1" / "stat").write_text("broken\n", encoding="ascii")
    with pytest.raises(PodRuntimeMeasurementError, match="boot epoch"):
        capture_pod_wall_time(
            proc_root=malformed,
            clock=lambda: datetime.fromtimestamp(1, tz=UTC),
            clock_ticks_per_second=100,
        )

    future = _proc_fixture(tmp_path / "future", boot=1_700_000_000, ticks=2_000)
    with pytest.raises(PodRuntimeMeasurementError, match="future"):
        capture_pod_wall_time(
            proc_root=future,
            clock=lambda: datetime.fromtimestamp(1_700_000_010, tz=UTC),
            clock_ticks_per_second=100,
        )
