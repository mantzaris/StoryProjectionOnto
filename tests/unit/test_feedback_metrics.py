from __future__ import annotations

import pytest
from pydantic import ValidationError

from story_projection_onto.metrics.feedback import (
    FeedbackEpisodeKind,
    FeedbackScore,
    capability_limited_rate,
    capability_limited_rates_by_kind,
    record_researcher_trace,
    score_scripted_feedback,
)


def test_known_answer_feedback_scores_registered_fields_with_denominators() -> None:
    result = score_scripted_feedback(
        episode_id="script-01-c2",
        before_strict_f1=0.5,
        after_strict_f1=0.75,
        required_target_change_ids=("merge-a-b", "qualify-belief"),
        realized_target_change_ids=("merge-a-b",),
        changed_semantic_ids=("merge-a-b", "collateral"),
        requested_scope_ids=("merge-a-b", "qualify-belief"),
        unsupported_changed_ids=("collateral",),
        before_rare_pivotal_recall=0.5,
        after_rare_pivotal_recall=1.0,
        capability_limited=False,
        latency_seconds=2.5,
        replay_hash_success=True,
    )
    assert result.strict_f1_change == pytest.approx(0.25)
    assert result.target_change_recall is not None
    assert result.target_change_recall.value == pytest.approx(0.5)
    assert result.edit_locality is not None
    assert result.edit_locality.value == pytest.approx(0.5)
    assert result.unsupported_change_count == 1
    assert result.rare_pivotal_recall_change == pytest.approx(0.5)


def test_no_changes_makes_locality_na_without_imputing_success() -> None:
    result = score_scripted_feedback(
        episode_id="script-02-c1",
        before_strict_f1=None,
        after_strict_f1=None,
        required_target_change_ids=("split-a",),
        realized_target_change_ids=(),
        changed_semantic_ids=(),
        requested_scope_ids=("split-a",),
        unsupported_changed_ids=(),
        before_rare_pivotal_recall=None,
        after_rare_pivotal_recall=None,
        capability_limited=True,
        latency_seconds=0.02,
        replay_hash_success=True,
    )
    assert result.target_change_recall is not None
    assert result.target_change_recall.value == 0
    assert result.edit_locality is not None
    assert result.edit_locality.value is None
    assert result.strict_f1_change is None


def test_researcher_trace_has_operational_fields_and_gold_na() -> None:
    trace = record_researcher_trace(
        episode_id="trace-01",
        changed_semantic_ids=("assertion-a", "node-b"),
        capability_limited=False,
        latency_seconds=1.2,
        replay_hash_success=True,
    )
    assert trace.kind is FeedbackEpisodeKind.RESEARCHER_TRACE
    assert trace.semantic_change_count == 2
    assert trace.strict_f1_change is None
    assert trace.target_change_recall is None
    assert trace.edit_locality is None
    assert trace.unsupported_change_count is None
    assert trace.rare_pivotal_recall_change is None


def test_trace_contract_rejects_gold_dependent_fields() -> None:
    with pytest.raises(ValidationError, match="gold-dependent"):
        FeedbackScore(
            episode_id="trace-invalid",
            kind=FeedbackEpisodeKind.RESEARCHER_TRACE,
            strict_f1_change=0.1,
            semantic_change_count=1,
            capability_limited=False,
            latency_seconds=1.0,
            replay_hash_success=True,
        )


def test_script_inputs_enforce_change_lineage_and_joint_values() -> None:
    arguments = {
        "episode_id": "script-invalid",
        "before_strict_f1": 0.5,
        "after_strict_f1": 0.6,
        "required_target_change_ids": ("target",),
        "realized_target_change_ids": ("target",),
        "changed_semantic_ids": (),
        "requested_scope_ids": ("target",),
        "unsupported_changed_ids": (),
        "before_rare_pivotal_recall": None,
        "after_rare_pivotal_recall": None,
        "capability_limited": False,
        "latency_seconds": 1.0,
        "replay_hash_success": True,
    }
    with pytest.raises(ValueError, match="realized target"):
        score_scripted_feedback(**arguments)
    arguments["realized_target_change_ids"] = ()
    arguments["after_strict_f1"] = None
    with pytest.raises(ValueError, match="jointly present"):
        score_scripted_feedback(**arguments)


def test_capability_limited_rate_is_reported_for_both_episode_kinds() -> None:
    script = score_scripted_feedback(
        episode_id="script-limited",
        before_strict_f1=None,
        after_strict_f1=None,
        required_target_change_ids=("target",),
        realized_target_change_ids=(),
        changed_semantic_ids=(),
        requested_scope_ids=("target",),
        unsupported_changed_ids=(),
        before_rare_pivotal_recall=None,
        after_rare_pivotal_recall=None,
        capability_limited=True,
        latency_seconds=0.01,
        replay_hash_success=True,
    )
    trace = record_researcher_trace(
        episode_id="trace-resolved",
        changed_semantic_ids=(),
        capability_limited=False,
        latency_seconds=0.01,
        replay_hash_success=True,
    )
    result = capability_limited_rate((script, trace))
    assert result.numerator == 1
    assert result.denominator == 2
    assert result.value == pytest.approx(0.5)
    separated = capability_limited_rates_by_kind((script, trace))
    assert separated.scripted_known_answer.value == pytest.approx(1.0)
    assert separated.researcher_trace.value == pytest.approx(0.0)
