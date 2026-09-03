from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.experiment import (  # noqa: E402
    REGISTERED_PROVISIONAL_SECONDS,
    AdmissionSignal,
    AllocatedGPUMeter,
    CostEntry,
    CostRole,
    ForecastAdmissionError,
    ForecastSource,
    GPUCallInventory,
    InventoryValidationError,
    ReserveExhausted,
    ReservePool,
    ReserveState,
    ReserveTier,
    ResourceLimits,
    TimingObservation,
    forecast_gpu_schedule,
    nearest_rank_percentile,
    summarize_call_class_timings,
    summarize_condition_block_costs,
)
from story_projection_onto.store import (  # noqa: E402
    GpuBudgetExceeded,
    GpuEventKind,
    Ledger,
)

INVENTORY_PATH = REPOSITORY_ROOT / "configs" / "study" / "gpu_call_inventory.json"
LIMITS_PATH = REPOSITORY_ROOT / "configs" / "study" / "resource_limits.json"
T0 = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def inventory() -> GPUCallInventory:
    return GPUCallInventory.load(INVENTORY_PATH)


def test_typed_inventory_reconciles_exact_registered_manifest() -> None:
    registered = inventory()

    assert registered.maximum_inference_attempts == 278
    assert registered.computed_inference_attempts == 278
    assert registered.session_start_count == 8
    assert registered.accounting_events == registered.computed_accounting_events == 286
    assert registered.provisional_planned_seconds == REGISTERED_PROVISIONAL_SECONDS == 31_229
    assert registered.call_class("reserve_long").count == 4
    assert registered.call_class("reserve_standard").count == 8
    assert registered.call_class("reserve_short").count == 4


def test_inventory_rejects_internal_and_registered_plan_drift() -> None:
    raw = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    raw["classes"][1]["count"] += 1
    with pytest.raises(InventoryValidationError, match="accounting_events"):
        GPUCallInventory.from_mapping(raw)

    raw = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    raw["classes"][-3]["count"] = 3
    raw["classes"][-2]["count"] += 1
    # Internal totals still reconcile, but cross-tier borrowing has changed the protocol.
    with pytest.raises(InventoryValidationError, match="reserve tiers"):
        GPUCallInventory.from_mapping(raw)


def test_resource_limits_are_typed_and_preserve_strict_ordering() -> None:
    limits = ResourceLimits.load(LIMITS_PATH)

    assert limits.preferred_forecast_gpu_seconds == 8.25 * 3600
    assert limits.scheduled_gpu_seconds == 9 * 3600
    assert limits.hard_gpu_seconds == 10 * 3600
    assert limits.model_cpu_offload_allowed is False
    assert limits.generation_concurrency == 1


def test_nearest_rank_p50_and_p95_are_not_interpolated() -> None:
    observations = list(range(1, 21))

    assert nearest_rank_percentile(observations, 0.50) == 10
    assert nearest_rank_percentile(observations, 0.95) == 19
    assert nearest_rank_percentile([9.5], 0.95) == 9.5
    with pytest.raises(ValueError, match="at least one"):
        nearest_rank_percentile([], 0.95)


def test_timing_summary_is_per_call_class() -> None:
    summaries = summarize_call_class_timings(
        [TimingObservation("acceptance_c1", seconds) for seconds in (4.0, 1.0, 3.0, 2.0)]
        + [TimingObservation("acceptance_c2", 8.0)]
    )

    by_name = {summary.call_class: summary for summary in summaries}
    assert by_name["acceptance_c1"].sample_count == 4
    assert by_name["acceptance_c1"].p50_seconds == 2.0
    assert by_name["acceptance_c1"].p95_seconds == 4.0
    assert by_name["acceptance_c2"].p50_seconds == 8.0


def test_provisional_forecast_is_admitted_but_above_preferred_signal() -> None:
    forecast = forecast_gpu_schedule(inventory())

    assert forecast.total_seconds == 31_229
    assert forecast.total_hours == pytest.approx(8.6747222222)
    assert forecast.admitted
    assert not forecast.within_preferred_margin
    assert forecast.admission_signal is AdmissionSignal.ADMITTED
    forecast.require_admission()


def test_small_sample_uses_longer_same_family_proxy_and_reserve_stays_fixed() -> None:
    forecast = forecast_gpu_schedule(
        inventory(),
        {
            "test_c2": [35.0],
            "development_c2": [42.0, 44.0],
            "reserve_standard": [1.0] * 20,
        },
    )
    rows = {row.call_class: row for row in forecast.rows}

    assert rows["test_c2"].forecast_p95_seconds == 44.0
    assert rows["test_c2"].source is ForecastSource.CONSERVATIVE_PROXY
    assert rows["test_c2"].proxy_call_class == "development_c2"
    assert rows["reserve_standard"].forecast_p95_seconds == 150
    assert rows["reserve_standard"].source is ForecastSource.REGISTERED_RESERVE


def test_forecast_rejects_more_than_nine_hours() -> None:
    samples = {
        item.name: [item.provisional_p95_seconds + 100.0] * 20 for item in inventory().classes
    }
    forecast = forecast_gpu_schedule(inventory(), samples)

    assert not forecast.admitted
    assert forecast.admission_signal is AdmissionSignal.REJECTED
    with pytest.raises(ForecastAdmissionError, match="exceeds scheduled"):
        forecast.require_admission()


def test_reserve_pool_is_resume_idempotent_and_never_borrows() -> None:
    pool = ReservePool(inventory())
    slots = [pool.consume(ReserveTier.LONG, f"long-{index}") for index in range(4)]

    assert [slot.index for slot in slots] == [1, 2, 3, 4]
    assert all(slot.watchdog_seconds == 240 for slot in slots)
    assert pool.consume(ReserveTier.LONG, "long-2") == slots[2]
    assert pool.remaining(ReserveTier.LONG) == 0
    assert pool.remaining(ReserveTier.STANDARD) == 8
    with pytest.raises(ReserveExhausted, match="borrowing"):
        pool.consume(ReserveTier.LONG, "long-overflow")

    resumed = ReservePool(inventory(), ReserveState(pool.state.consumed))
    assert resumed.remaining(ReserveTier.LONG) == 0
    assert resumed.consume(ReserveTier.LONG, "long-0") == slots[0]


def test_cost_accounting_exposes_c1_amortization_and_fixed_select_inheritance() -> None:
    entries = [
        CostEntry(
            condition="C1",
            block_id="world-01/seed-0",
            call_class="test_c1",
            role=CostRole.PREBUILD,
            allocated_gpu_seconds=120,
            prompt_tokens=100,
            completion_tokens=20,
            served_context_count=3,
        ),
        *[
            CostEntry(
                condition="C2",
                block_id="world-01/seed-0",
                call_class="test_c2",
                role=CostRole.QUERY_TIME,
                allocated_gpu_seconds=seconds,
                context_id=f"q-{ordinal}",
                query_ordinal=ordinal,
            )
            for ordinal, seconds in enumerate((30, 40, 50), start=1)
        ],
        CostEntry(
            condition="A-FixedSelect",
            block_id="world-01/seed-0",
            call_class="test_c1",
            role=CostRole.INHERITED_C1_PREBUILD,
            allocated_gpu_seconds=120,
            prompt_tokens=100,
            completion_tokens=20,
            served_context_count=3,
        ),
        *[
            CostEntry(
                condition="A-FixedSelect",
                block_id="world-01/seed-0",
                call_class="test_fixed_select",
                role=CostRole.QUERY_TIME,
                allocated_gpu_seconds=10,
                context_id=f"q-{ordinal}",
                query_ordinal=ordinal,
            )
            for ordinal in range(1, 4)
        ],
    ]
    summaries = {summary.condition: summary for summary in summarize_condition_block_costs(entries)}

    assert summaries["C1"].own_allocated_gpu_seconds == 120
    assert summaries["C1"].served_context_count == 3
    assert summaries["C1"].amortized_seconds_per_context == 40
    assert summaries["C2"].first_query_workload_seconds == 30
    assert summaries["C2"].complete_workload_seconds == 120
    assert summaries["A-FixedSelect"].inherited_c1_prebuild_seconds == 120
    assert summaries["A-FixedSelect"].own_allocated_gpu_seconds == 30
    assert summaries["A-FixedSelect"].first_query_workload_seconds == 130
    assert summaries["A-FixedSelect"].complete_workload_seconds == 150
    assert summaries["A-FixedSelect"].amortized_seconds_per_context == 50


class FakeMonotonicClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FakeWallClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        value = T0 + timedelta(seconds=self.calls)
        self.calls += 1
        return value


def test_meter_context_records_load_warmup_inference_repair_and_restart(tmp_path: Path) -> None:
    with Ledger(tmp_path / "meter.sqlite3") as ledger:
        meter = AllocatedGPUMeter(
            ledger,
            monotonic_clock=FakeMonotonicClock([0, 2, 2, 3, 3, 7, 7, 8.5, 8.5, 10]),
            wall_clock=FakeWallClock(),
        )
        contexts = (
            meter.model_load(event_id="load", maximum_seconds=10),
            meter.warmup(event_id="warmup", maximum_seconds=10),
            meter.inference(event_id="inference", maximum_seconds=10),
            meter.repair(event_id="repair", maximum_seconds=10),
            meter.restart(event_id="restart", maximum_seconds=10),
        )
        for context in contexts:
            with context:
                pass

        summary = ledger.gpu_summary()
        assert summary.total_allocated_seconds == 10
        assert summary.seconds_for(GpuEventKind.MODEL_LOAD) == 2
        assert summary.seconds_for(GpuEventKind.WARM_UP) == 1
        assert summary.seconds_for(GpuEventKind.INFERENCE) == 4
        assert summary.seconds_for(GpuEventKind.REPAIR) == 1.5
        assert summary.seconds_for(GpuEventKind.RESTART) == 1.5
        assert meter.actual_allocated_gpu_seconds == 10


def test_meter_classifies_exceptions_and_resumes_existing_total(tmp_path: Path) -> None:
    database = tmp_path / "resume.sqlite3"
    with Ledger(database) as ledger:
        meter = AllocatedGPUMeter(
            ledger,
            monotonic_clock=FakeMonotonicClock([0, 2, 2, 5]),
            wall_clock=FakeWallClock(),
        )
        with (
            pytest.raises(TimeoutError),
            meter.inference(event_id="timed-out", maximum_seconds=5),
        ):
            raise TimeoutError("watchdog")
        with (
            pytest.raises(RuntimeError),
            meter.inference(event_id="failed", maximum_seconds=5),
        ):
            raise RuntimeError("service failed")
        assert ledger.gpu_summary().seconds_for(GpuEventKind.TIMEOUT) == 2
        assert ledger.gpu_summary().seconds_for(GpuEventKind.FAILURE) == 3

    with Ledger(database) as resumed_ledger:
        resumed = AllocatedGPUMeter(resumed_ledger)
        assert resumed.actual_allocated_gpu_seconds == 5


def test_meter_enforces_scheduled_forecast_and_strict_hard_stop(tmp_path: Path) -> None:
    with Ledger(tmp_path / "limits.sqlite3") as ledger:
        ledger.record_gpu_event(
            event_id="prior",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=8,
            started_at=T0,
            ended_at=T0 + timedelta(seconds=8),
            succeeded=True,
        )
        meter = AllocatedGPUMeter(
            ledger,
            scheduled_limit_seconds=9,
            hard_limit_seconds=10,
        )

        with pytest.raises(ForecastAdmissionError, match="scheduled"):
            meter.require_capacity(0.5, remaining_required_seconds=0.6)
        meter.require_capacity(
            0.5,
            remaining_required_seconds=0.6,
            contingency_unlocked=True,
            essential_recovery=True,
        )
        with pytest.raises(GpuBudgetExceeded, match="reach/cross"):
            meter.require_capacity(2)
