from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

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
    StorageAllocationPlan,
    TimingObservation,
    forecast_gpu_schedule,
    nearest_rank_percentile,
    summarize_call_class_timings,
    summarize_condition_block_costs,
)
from story_projection_onto.store import (  # noqa: E402
    GpuAllocationJournalState,
    GpuBudgetExceeded,
    GpuEventKind,
    GpuServiceJournalState,
    Ledger,
)

INVENTORY_PATH = REPOSITORY_ROOT / "configs" / "study" / "gpu_call_inventory.json"
LIMITS_PATH = REPOSITORY_ROOT / "configs" / "study" / "resource_limits.json"
STORAGE_PLAN_PATH = REPOSITORY_ROOT / "configs" / "study" / "storage_phase_allocations.json"
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


def test_storage_allocation_plan_has_exact_typed_phase_reservations() -> None:
    plan = StorageAllocationPlan.load(STORAGE_PLAN_PATH)

    phase_one = plan.reservation_for("phase_1")
    assert tuple(plan.reservations) == tuple(f"phase_{index}" for index in range(1, 8))
    assert phase_one.additional_reserved_bytes == 1_085_544_320
    assert phase_one.preflight_arguments() == {
        "declared_growth_bytes": 750_000_000,
        "largest_atomic_temporary_bytes": 67_108_864,
        "quarantine_allowance_bytes": 134_217_728,
        "release_staging_bytes": 134_217_728,
    }


def test_storage_allocation_plan_rejects_missing_or_untyped_phases() -> None:
    raw = json.loads(STORAGE_PLAN_PATH.read_text(encoding="utf-8"))
    del raw["reservations"]["phase_7"]
    with pytest.raises(ValueError, match="exactly"):
        StorageAllocationPlan.from_mapping(raw)

    raw = json.loads(STORAGE_PLAN_PATH.read_text(encoding="utf-8"))
    raw["reservations"]["phase_1"]["declared_growth_bytes"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        StorageAllocationPlan.from_mapping(raw)


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
        assert ledger.count_rows("gpu_allocation_journal") == 10
        journal = ledger.gpu_allocation_journal_records()
        assert len(journal) == 10
        assert tuple((item.allocation_id, item.sequence) for item in journal) == tuple(
            (allocation_id, sequence)
            for allocation_id in ("inference", "load", "repair", "restart", "warmup")
            for sequence in (0, 1)
        )
        assert ledger.unresolved_gpu_allocations() == ()


def test_meter_recovers_crash_open_interval_before_new_allocation(tmp_path: Path) -> None:
    database = tmp_path / "crash-open.sqlite3"
    with Ledger(database) as ledger:
        ledger.record_gpu_allocation_observation(
            allocation_id="crashed-call",
            state=GpuAllocationJournalState.OPENED,
            intended_event_kind=GpuEventKind.INFERENCE,
            elapsed_seconds=0,
            maximum_seconds=10,
            observed_at=T0,
        )
        ledger.record_gpu_allocation_observation(
            allocation_id="crashed-call",
            state=GpuAllocationJournalState.HEARTBEAT,
            intended_event_kind=GpuEventKind.INFERENCE,
            elapsed_seconds=3,
            maximum_seconds=10,
            observed_at=T0 + timedelta(seconds=3),
        )

    with Ledger(database) as resumed_ledger:
        resumed = AllocatedGPUMeter(
            resumed_ledger,
            wall_clock=lambda: T0 + timedelta(seconds=20),
        )
        assert resumed.actual_allocated_gpu_seconds == 10
        assert resumed_ledger.gpu_summary().seconds_for(GpuEventKind.FAILURE) == 10
        assert resumed_ledger.unresolved_gpu_allocations() == ()


def test_meter_service_journal_closes_without_double_counting(tmp_path: Path) -> None:
    with Ledger(tmp_path / "service-journal.sqlite3") as ledger:
        meter = AllocatedGPUMeter(ledger)
        opened = meter.open_service_journal(
            service_session_id="pilot-service",
            session_id="pilot",
            configuration_hash="a" * 64,
            started_at=T0,
            ledger_allocated_seconds_before_session=0,
        )
        meter.observe_service_journal(
            service_session_id=opened.service_session_id,
            elapsed_seconds=1,
            observed_at=T0 + timedelta(seconds=1),
        )
        ledger.record_gpu_event(
            event_id="pilot-inference",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=0.5,
            started_at=T0 + timedelta(seconds=1),
            ended_at=T0 + timedelta(seconds=1, microseconds=500_000),
            succeeded=True,
        )
        meter.observe_service_journal(
            service_session_id=opened.service_session_id,
            elapsed_seconds=2,
            observed_at=T0 + timedelta(seconds=2),
            process_stopped=True,
        )
        record = meter.reconcile_service_session(
            service_session_id=opened.service_session_id,
            session_id="pilot",
            service_seconds=2,
            classified_event_seconds=0.5,
            started_at=T0,
            ended_at=T0 + timedelta(seconds=2),
        )
        assert record.service_seconds == 2
        assert meter.actual_allocated_gpu_seconds == 2
        assert (
            ledger.latest_gpu_service_journal(opened.service_session_id).state
            is GpuServiceJournalState.CLOSED
        )


def test_meter_service_recovery_is_durable_before_hard_limit_error(tmp_path: Path) -> None:
    with Ledger(tmp_path / "service-recovery.sqlite3") as ledger:
        meter = AllocatedGPUMeter(
            ledger,
            scheduled_limit_seconds=9,
            hard_limit_seconds=10,
        )
        meter.open_service_journal(
            service_session_id="crashed-service",
            session_id="pilot",
            configuration_hash="b" * 64,
            started_at=T0,
            ledger_allocated_seconds_before_session=0,
        )
        with pytest.raises(GpuBudgetExceeded, match="recovered GPU service"):
            meter.recover_service_journal(
                service_session_id="crashed-service",
                recovered_at=T0 + timedelta(seconds=10),
                details={"pid_absent": True, "endpoint_absent": True},
            )
        assert ledger.gpu_summary().total_allocated_seconds == 10
        assert ledger.unresolved_gpu_service_journals() == ()


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
        with pytest.raises(ValueError, match="inside the scheduled envelope"):
            meter.require_capacity(
                0.5,
                remaining_required_seconds=0,
                contingency_unlocked=True,
                essential_recovery=True,
            )
        with pytest.raises(GpuBudgetExceeded, match="next and remaining required work"):
            meter.require_capacity(
                0.5,
                remaining_required_seconds=1.5,
                contingency_unlocked=True,
                essential_recovery=True,
            )
        with pytest.raises(GpuBudgetExceeded, match="reach/cross"):
            meter.require_capacity(2)


def test_meter_contingency_is_paired_and_restricted_to_service_start(tmp_path: Path) -> None:
    with Ledger(tmp_path / "contingency-scope.sqlite3") as ledger:
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

        with pytest.raises(ValueError, match="paired unlock"):
            meter.require_capacity(
                0.5,
                remaining_required_seconds=0.6,
                contingency_unlocked=True,
            )
        with pytest.raises(ValueError, match="exact booleans"):
            meter.require_capacity(
                0.5,
                remaining_required_seconds=0.6,
                contingency_unlocked=cast(bool, 1),
                essential_recovery=True,
            )
        with pytest.raises(ValueError, match="only an essential-recovery service start"):
            meter.inference(
                event_id="forbidden-contingency-inference",
                maximum_seconds=0.5,
                remaining_required_seconds=0.6,
                contingency_unlocked=True,
                essential_recovery=True,
            )

        assert tuple(event.event_id for event in ledger.gpu_events()) == ("prior",)
        assert ledger.unresolved_gpu_allocations() == ()


def test_meter_accounting_reads_are_serialized_across_watchdog_threads(
    tmp_path: Path,
) -> None:
    with Ledger(tmp_path / "meter-thread-lock.sqlite3") as ledger:
        meter = AllocatedGPUMeter(ledger)
        original_summary = ledger.gpu_summary
        first_entered = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        state_lock = threading.Lock()
        calls = 0
        active = 0
        maximum_active = 0

        def controlled_summary():
            nonlocal calls, active, maximum_active
            with state_lock:
                calls += 1
                call_number = calls
                active += 1
                maximum_active = max(maximum_active, active)
            if call_number == 1:
                first_entered.set()
                assert release_first.wait(1)
            else:
                second_entered.set()
            try:
                return original_summary()
            finally:
                with state_lock:
                    active -= 1

        ledger.gpu_summary = controlled_summary  # type: ignore[method-assign]
        results: list[float] = []
        first = threading.Thread(target=lambda: results.append(meter.actual_allocated_gpu_seconds))
        second = threading.Thread(target=lambda: results.append(meter.actual_allocated_gpu_seconds))
        first.start()
        assert first_entered.wait(1)
        second.start()
        assert not second_entered.wait(0.1)
        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)
        assert not first.is_alive() and not second.is_alive()
        assert sorted(results) == [0, 0]
        assert maximum_active == 1
