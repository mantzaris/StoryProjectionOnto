from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from story_projection_onto.combined_gpu_block import CombinedBlockError
from story_projection_onto.combined_gpu_inputs import replay_combined_upstream_gate
from story_projection_onto.development_runtime import DevelopmentPhase
from tests.unit.test_combined_gpu_block import digest

NOW = datetime(2026, 9, 4, tzinfo=UTC)


def _gate_inputs():
    development = SimpleNamespace(
        content_hash=digest("development"),
        phase=DevelopmentPhase.COMPLETED,
        gate=SimpleNamespace(passed=True),
        admission_failure_code=None,
    )
    held_out_calls = SimpleNamespace(
        content_hash=digest("held-out-calls"),
        development_execution_result_hash=development.content_hash,
    )
    itt_records = tuple(
        SimpleNamespace(
            content_hash=digest(f"held-out-itt-{index}"),
            schedule_after=SimpleNamespace(actual_allocated_gpu_seconds=1000 + index),
        )
        for index in range(168)
    )
    barrier = SimpleNamespace(content_hash=digest("barrier"))
    ablation_receipts = tuple(
        SimpleNamespace(content_hash=digest(f"ablation-prequery-{index}")) for index in range(36)
    )
    held_out_execution = SimpleNamespace(
        content_hash=digest("held-out-execution"),
        call_manifest_hash=held_out_calls.content_hash,
        final_reviewed_seal_hash=digest("final-review"),
        itt_records=itt_records,
        final_schedule_snapshot_hash=digest("final-schedule"),
        prequery_barrier=barrier,
        ablation_prequery_receipts=ablation_receipts,
    )
    scorer_bridge = SimpleNamespace(
        content_hash=digest("scorer-bridge"),
        execution_manifest_hash=held_out_execution.content_hash,
        call_manifest_hash=held_out_calls.content_hash,
        final_reviewed_seal_hash=held_out_execution.final_reviewed_seal_hash,
        runtime_namespace_closed=True,
        model_input_open=False,
        itt_record_hashes=tuple(item.content_hash for item in itt_records),
        authorized_at=NOW,
    )
    final_schedule = SimpleNamespace(
        content_hash=held_out_execution.final_schedule_snapshot_hash,
        global_accounting_id="test-global-accounting",
        actual_allocated_gpu_seconds=2000,
        remaining_registered_p95_seconds=9000,
        repair_reserves=(
            SimpleNamespace(
                reserve_class="reserve_short",
                consumed_slots=2,
            ),
        ),
    )
    phase5_inputs = SimpleNamespace(
        content_hash=digest("phase5-inputs"),
        held_out_execution_manifest_hash=held_out_execution.content_hash,
        held_out_call_manifest_hash=held_out_calls.content_hash,
        final_reviewed_seal_hash=held_out_execution.final_reviewed_seal_hash,
        frozen_at=NOW + timedelta(seconds=1),
    )
    return (
        development,
        held_out_calls,
        held_out_execution,
        scorer_bridge,
        final_schedule,
        phase5_inputs,
    )


def test_upstream_gate_replays_all_168_itt_and_36_ablation_preparations() -> None:
    values = _gate_inputs()
    gate = replay_combined_upstream_gate(
        development=values[0],
        held_out_calls=values[1],
        held_out_execution=values[2],
        scorer_bridge=values[3],
        final_schedule=values[4],
        phase5_inputs=values[5],
        verified_at=NOW + timedelta(seconds=2),
    )
    assert len(gate.held_out_itt_record_hashes) == 168
    assert gate.consumed_short_reserve_slots_before_block == 2
    assert gate.global_accounting_id == "test-global-accounting"
    assert gate.phase5_input_manifest_hash == values[5].content_hash


@pytest.mark.parametrize("mutation", ("development", "itt", "chronology"))
def test_upstream_gate_fails_closed_on_incomplete_or_backdated_predecessor(
    mutation: str,
) -> None:
    values = list(_gate_inputs())
    development, _calls, _execution, bridge, _schedule, phase5 = values
    if mutation == "development":
        development.gate.passed = False
    elif mutation == "itt":
        bridge.itt_record_hashes = bridge.itt_record_hashes[:-1]
    else:
        phase5.frozen_at = bridge.authorized_at
    with pytest.raises(CombinedBlockError, match="accepted development"):
        replay_combined_upstream_gate(
            development=values[0],
            held_out_calls=values[1],
            held_out_execution=values[2],
            scorer_bridge=values[3],
            final_schedule=values[4],
            phase5_inputs=values[5],
            verified_at=NOW + timedelta(seconds=2),
        )
