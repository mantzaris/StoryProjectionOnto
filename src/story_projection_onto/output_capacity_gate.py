"""Small CPU-only admission calculation for the authorized capacity recovery.

This does not start or control a service. Unchanged historical ledger allocation
is included in every bound. Candidate output-cap scaling is an explicitly
conservative unmeasured forecast, never truncated-response throughput.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

BASELINE_SECONDS = 3227.826324
SCHEDULED_SECONDS = 33660
HARD_SECONDS = 36000
BLOCK_SECONDS = 1200
SHUTDOWN_SECONDS = 60


@dataclass(frozen=True)
class CapacityRecoveryState:
    actual_allocated_seconds: float
    service_starts: int
    diagnostic_attempts: int

    def admit(
        self,
        *,
        remaining_mandatory_seconds: float,
        stage_seconds: float,
        starting_service: bool = False,
        diagnostic_generation: bool = False,
        complete_packing: bool = False,
    ) -> dict[str, float]:
        values = (self.actual_allocated_seconds, remaining_mandatory_seconds, stage_seconds)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("invalid allocation measurement")
        if self.actual_allocated_seconds < BASELINE_SECONDS:
            raise ValueError("authoritative V10 allocation must not reset")
        if not complete_packing:
            raise ValueError("complete input/output capacity gate has not passed")
        if not 0 <= self.service_starts + int(starting_service) <= 2:
            raise ValueError("two-start recovery limit")
        if not 0 <= self.diagnostic_attempts + int(diagnostic_generation) <= 3:
            raise ValueError("three-diagnostic recovery limit")
        additional = self.actual_allocated_seconds - BASELINE_SECONDS
        needed = stage_seconds + SHUTDOWN_SECONDS
        if additional + needed >= BLOCK_SECONDS - 1:
            raise ValueError("whole recovery allocation exhausted before safe shutdown")
        if self.actual_allocated_seconds + needed >= HARD_SECONDS - 1:
            raise ValueError("strict hard allocation stop")
        all_in = self.actual_allocated_seconds + needed + remaining_mandatory_seconds
        if all_in > SCHEDULED_SECONDS:
            raise ValueError(f"all-in forecast {all_in:.6f} exceeds {SCHEDULED_SECONDS}")
        return {
            "all_in_seconds": all_in,
            "scheduled_reserve_seconds": SCHEDULED_SECONDS - all_in,
            "block_reserve_seconds": BLOCK_SECONDS - additional - needed,
            "hard_contingency_seconds": HARD_SECONDS - all_in,
        }


def capacity_forecast(rows: Sequence[Mapping], *, output_tokens: int = 6144) -> dict:
    """Preserve every remaining row; scale only unmeasured generation proxies.

    Output-cap ratio is a sensitivity/admission proxy, not measured p95. No
    latency reduction is credited for compression, short fixtures, or invalid
    generations. Model-load/allocation forecasts remain unchanged.
    """
    if output_tokens < 2048:
        raise ValueError("capacity repair cannot silently lower required output allocation")
    updated = []
    for row in rows:
        item = dict(row)
        name = item["call_class"]
        denominator = 1536 if "repair" in name or name == "reserve_short" else 2048
        allowance = 3072 if "fixed_select" in name else output_tokens
        ratio = 1.0 if name == "gpu_session_start" else allowance / denominator
        item["candidate_output_allowance"] = allowance
        item["previous_p95_proxy_seconds"] = item["forecast_p95_seconds"]
        item["capacity_scaling_factor_not_measured"] = ratio
        capacity_demand = item["forecast_p95_seconds"] * ratio
        watchdog = (
            240
            if "c1" in name or name == "reserve_long"
            else 90
            if "fixed_select" in name or "repair" in name or name == "reserve_short"
            else 150
        )
        item["uncapped_capacity_demand_proxy_seconds"] = capacity_demand
        item["registered_watchdog_seconds"] = None if name == "gpu_session_start" else watchdog
        item["capacity_demand_exceeds_watchdog"] = (
            name != "gpu_session_start" and capacity_demand > watchdog
        )
        item["forecast_p95_seconds"] = (
            capacity_demand if name == "gpu_session_start" else min(capacity_demand, watchdog)
        )
        item["remaining_forecast_seconds"] = item["remaining_count"] * item["forecast_p95_seconds"]
        updated.append(item)
    remaining = math.fsum(row["remaining_forecast_seconds"] for row in updated)
    return {
        "kind": "unmeasured_capacity_proxy_capped_at_unchanged_watchdogs",
        "valid_completion_forecast_established": False,
        "rows": updated,
        "actual_allocated_seconds": BASELINE_SECONDS,
        "remaining_forecast_seconds": remaining,
        "all_in_seconds": BASELINE_SECONDS + remaining,
        "plus_full_authorized_recovery_block": BASELINE_SECONDS + remaining + BLOCK_SECONDS,
        "original_nine_hour_target_met": False,
        "fits_scheduled_before_new_recovery": BASELINE_SECONDS + remaining <= SCHEDULED_SECONDS,
        "truncated_output_throughput_credited": False,
    }
