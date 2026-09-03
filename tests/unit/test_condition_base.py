from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from story_projection_onto.conditions.base import (
    ComparisonInputManifest,
    ConditionAttemptRecord,
    ConditionExecutionTrace,
    ConditionIntegrityError,
    ExecutionStage,
    assert_comparison_fairness,
)
from story_projection_onto.contracts import (
    ConditionName,
    OutputBudgets,
    ReleaseClass,
    RunOutcome,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def budgets() -> OutputBudgets:
    return OutputBudgets(
        node_budget=10,
        assertion_budget=12,
        display_node_budget=8,
        display_assertion_budget=10,
    )


def fairness_manifest(condition: ConditionName) -> ComparisonInputManifest:
    is_c0 = condition is ConditionName.C0_CLASSICAL_PRE
    seed_block = None if is_c0 else 1
    return ComparisonInputManifest(
        condition=condition,
        snapshot_hash=HASH_A,
        packet_hash=HASH_B,
        ordered_evidence_ids=("evidence-1", "evidence-2"),
        horizon_hash=HASH_C,
        context_semantics_hash=HASH_D,
        upper_ontology_hash=HASH_E,
        budgets=budgets(),
        maximum_input_tokens=10_240,
        maximum_output_tokens=2_048,
        repair_attempt_budget=1,
        seed_block=seed_block,
        source_c1_seed_block=(seed_block if condition is ConditionName.A_FIXED_SELECT else None),
        model_stack_hash=None if is_c0 else HASH_F,
        decoding_family_hash=None if is_c0 else HASH_A,
        validator_hash=HASH_B,
    )


def complete_fairness_block() -> tuple[ComparisonInputManifest, ...]:
    return tuple(
        fairness_manifest(condition)
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    )


def test_execution_trace_enforces_registered_contiguous_state_machine() -> None:
    trace = ConditionExecutionTrace(trace_id="trace-1", condition=ConditionName.C2_LLM_QUERY)
    trace = trace.advance(ExecutionStage.PREQUERY_SEALED, occurred_at=NOW)
    trace = trace.advance(ExecutionStage.QUERY_REVEALED, occurred_at=NOW + timedelta(seconds=1))
    trace = trace.advance(ExecutionStage.GENERATED, occurred_at=NOW + timedelta(seconds=2))
    trace = trace.advance(ExecutionStage.REPAIRED, occurred_at=NOW + timedelta(seconds=3))
    trace = trace.advance(ExecutionStage.VALIDATED, occurred_at=NOW + timedelta(seconds=4))
    trace = trace.advance(ExecutionStage.FINALIZED, occurred_at=NOW + timedelta(seconds=5))
    assert trace.current_stage is ExecutionStage.FINALIZED

    with pytest.raises(ValidationError, match="unregistered condition transition"):
        ConditionExecutionTrace(
            trace_id="trace-bad",
            condition=ConditionName.C2_LLM_QUERY,
        ).advance(ExecutionStage.GENERATED, occurred_at=NOW)


def test_execution_trace_preserves_terminal_failures() -> None:
    trace = ConditionExecutionTrace(trace_id="trace-failure", condition=ConditionName.C1_LLM_PRE)
    trace = trace.advance(ExecutionStage.PREQUERY_SEALED, occurred_at=NOW)
    trace = trace.advance(
        ExecutionStage.TIMED_OUT,
        occurred_at=NOW + timedelta(seconds=1),
        diagnostic_code="watchdog_timeout",
    )
    assert trace.current_stage is ExecutionStage.TIMED_OUT
    with pytest.raises(ValidationError, match="unregistered condition transition"):
        trace.advance(ExecutionStage.QUERY_REVEALED, occurred_at=NOW + timedelta(seconds=2))


def test_fairness_manifest_accepts_only_exact_common_inputs_and_paired_seeds() -> None:
    assert_comparison_fairness(complete_fairness_block())

    manifests = list(complete_fairness_block())
    c2 = manifests[2].model_dump(exclude={"content_hash"})
    c2["packet_hash"] = HASH_C
    manifests[2] = ComparisonInputManifest(**c2)
    with pytest.raises(ConditionIntegrityError, match="nonidentical evidence"):
        assert_comparison_fairness(manifests)

    manifests = list(complete_fairness_block())
    fixed = manifests[3].model_dump(exclude={"content_hash"})
    fixed["seed_block"] = 2
    fixed["source_c1_seed_block"] = 2
    manifests[3] = ComparisonInputManifest(**fixed)
    with pytest.raises(ConditionIntegrityError, match="paired seed"):
        assert_comparison_fairness(manifests)


def test_fairness_manifest_rejects_budget_or_model_asymmetry() -> None:
    manifests = list(complete_fairness_block())
    changed = manifests[1].model_dump(exclude={"content_hash"})
    changed["maximum_output_tokens"] = 1_024
    manifests[1] = ComparisonInputManifest(**changed)
    with pytest.raises(ConditionIntegrityError, match="maximum_output_tokens"):
        assert_comparison_fairness(manifests)

    manifests = list(complete_fairness_block())
    changed = manifests[2].model_dump(exclude={"content_hash"})
    changed["model_stack_hash"] = HASH_D
    manifests[2] = ComparisonInputManifest(**changed)
    with pytest.raises(ConditionIntegrityError, match="model stacks"):
        assert_comparison_fairness(manifests)


@pytest.mark.parametrize(
    "outcome",
    [
        RunOutcome.INVALID,
        RunOutcome.FAILED,
        RunOutcome.TIMED_OUT,
        RunOutcome.INTERRUPTED,
    ],
)
def test_intention_to_treat_assigns_semantic_zero_to_unsuccessful_output(
    outcome: RunOutcome,
) -> None:
    attempt = ConditionAttemptRecord(
        attempt_id=f"attempt-{outcome.value}",
        condition=ConditionName.C2_LLM_QUERY,
        unit_id="world-01-context-a",
        seed_block=1,
        outcome=outcome,
        failure_code="recorded_failure",
        release_class=ReleaseClass.PUBLIC,
    )
    assert attempt.included_in_intention_to_treat is True
    assert attempt.semantic_score_for_nonempty_gold(None) == 0.0


def test_intention_to_treat_rejects_unrecorded_or_masquerading_failure() -> None:
    with pytest.raises(ValidationError, match="failure code"):
        ConditionAttemptRecord(
            attempt_id="attempt-invalid",
            condition=ConditionName.C2_LLM_QUERY,
            unit_id="unit",
            seed_block=1,
            outcome=RunOutcome.INVALID,
            release_class=ReleaseClass.PUBLIC,
        )
