import pytest

from story_projection_onto.output_capacity_gate import (
    BASELINE_SECONDS,
    SMALL_START_BASELINE_SECONDS,
    CapacityRecoveryState,
    capacity_forecast,
)


def test_fifth_start_whole_envelope_and_single_call_preserve_prior_usage():
    args = dict(
        remaining_mandatory_seconds=40000,
        stage_seconds=510,
        complete_packing=True,
        feasibility_diagnostic_exception=True,
    )
    receipt = CapacityRecoveryState(SMALL_START_BASELINE_SECONDS, 4, 3).admit(
        **args, starting_service=True
    )
    assert receipt["block_reserve_seconds"] == pytest.approx(5)
    with pytest.raises(ValueError, match="560-second"):
        CapacityRecoveryState(SMALL_START_BASELINE_SECONDS, 4, 3).admit(
            **(args | {"stage_seconds": 511}), starting_service=True
        )
    with pytest.raises(ValueError, match="only one small"):
        CapacityRecoveryState(SMALL_START_BASELINE_SECONDS + 200, 5, 4).admit(
            **(args | {"stage_seconds": 120}), diagnostic_generation=True
        )


def test_admits_only_complete_all_in_with_preserved_baseline():
    state = CapacityRecoveryState(BASELINE_SECONDS, 0, 0)
    receipt = state.admit(
        remaining_mandatory_seconds=1000,
        stage_seconds=100,
        complete_packing=True,
        diagnostic_generation=True,
    )
    assert receipt["all_in_seconds"] == BASELINE_SECONDS + 1000 + 100 + 45


@pytest.mark.parametrize(
    "state,kwargs,reason",
    [
        (CapacityRecoveryState(0, 0, 0), {}, "must not reset"),
        (CapacityRecoveryState(BASELINE_SECONDS, 5, 0), {"starting_service": True}, "five-start"),
        (
            CapacityRecoveryState(BASELINE_SECONDS, 0, 5),
            {"diagnostic_generation": True},
            "five-diagnostic",
        ),
        (CapacityRecoveryState(BASELINE_SECONDS + 1900, 1, 1), {}, "whole recovery"),
        (
            CapacityRecoveryState(BASELINE_SECONDS, 0, 0),
            {"complete_packing": False},
            "capacity gate",
        ),
        (
            CapacityRecoveryState(BASELINE_SECONDS, 0, 0),
            {"remaining_mandatory_seconds": 31000},
            "all-in",
        ),
    ],
)
def test_rejects_each_bound(state, kwargs, reason):
    params = {
        "remaining_mandatory_seconds": 1000,
        "stage_seconds": 100,
        "complete_packing": True,
    } | kwargs
    with pytest.raises(ValueError, match=reason):
        state.admit(**params)


def test_forecast_keeps_all_mandatory_rows_and_loads():
    rows = [
        {"call_class": "gpu_session_start", "remaining_count": 5, "forecast_p95_seconds": 333},
        {"call_class": "synthetic_c1", "remaining_count": 24, "forecast_p95_seconds": 180},
    ]
    result = capacity_forecast(rows)
    assert len(result["rows"]) == len(rows)
    assert result["remaining_forecast_seconds"] == 5 * 333 + 24 * 240
    assert result["rows"][1]["capacity_demand_exceeds_watchdog"]
    assert rows[1]["forecast_p95_seconds"] == 180
    assert result["truncated_output_throughput_credited"] is False


def test_exception_only_bypasses_forecast_not_block_or_count_limits():
    kwargs = dict(
        remaining_mandatory_seconds=40000,
        stage_seconds=240,
        complete_packing=True,
        diagnostic_generation=True,
        feasibility_diagnostic_exception=True,
    )
    assert CapacityRecoveryState(BASELINE_SECONDS, 1, 0).admit(**kwargs)[
        "complete_forecast_exception_applied"
    ]
    with pytest.raises(ValueError, match="whole recovery"):
        CapacityRecoveryState(BASELINE_SECONDS + 1700, 1, 0).admit(**kwargs)
    with pytest.raises(ValueError, match="five-diagnostic"):
        CapacityRecoveryState(BASELINE_SECONDS, 1, 5).admit(**kwargs)
    with pytest.raises(ValueError, match="diagnostic-only"):
        CapacityRecoveryState(BASELINE_SECONDS, 0, 0).admit(
            **(kwargs | {"diagnostic_generation": False})
        )
