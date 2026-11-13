from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import story_projection_onto.scorer_only.phase5_feedback as phase5_scoring
from story_projection_onto.contracts import ConditionName
from story_projection_onto.feedback_runtime import FeedbackAttemptStatus
from story_projection_onto.metrics.feedback import score_scripted_feedback
from story_projection_onto.scorer_only.phase5_feedback import (
    Phase5ProjectionGoldScore,
    Phase5ScriptedFeedbackMetricRecord,
    Phase5ScriptedFeedbackMetrics,
    _projection_gold_score,
)

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def gold_score(label: str, *, strict: float, rare: float) -> Phase5ProjectionGoldScore:
    return Phase5ProjectionGoldScore(
        projection_hash=digest(f"projection-{label}"),
        gold_projection_hash=digest(f"gold-{label}"),
        gold_source_hash=digest(f"source-{label}"),
        scorer_plan_hash=digest(f"plan-{label}"),
        alignment_plan_hash=digest(f"alignment-{label}"),
        strict_qualified_assertion_f1=strict,
        rare_pivotal_recall=rare,
        strict_match_target_ids=(f"target-{label}",),
        rare_score_hash=digest(f"rare-{label}"),
        structurally_valid=True,
        content_bearing=True,
    )


def test_empty_structurally_valid_projection_is_an_itt_semantic_failure(monkeypatch) -> None:
    invalid_flags: list[bool] = []

    monkeypatch.setattr(
        phase5_scoring,
        "prediction_records",
        lambda *args, **kwargs: ((), (), ()),
    )
    monkeypatch.setattr(
        phase5_scoring,
        "audit_qualified_assertion_grounding",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        phase5_scoring,
        "projection_is_structurally_valid",
        lambda projection: True,
    )
    monkeypatch.setattr(
        phase5_scoring,
        "projection_is_content_bearing",
        lambda projection: False,
    )

    def alignment(*args, invalid_semantic_output, **kwargs):
        invalid_flags.append(invalid_semantic_output)
        return SimpleNamespace(
            strict_assertion_matches=(),
            strict_assertion_score=SimpleNamespace(f1=0.0),
        )

    def rare(*args, invalid_semantic_output, **kwargs):
        invalid_flags.append(invalid_semantic_output)
        return SimpleNamespace(
            qualified_assertion_recall=SimpleNamespace(value=0.0),
            content_hash=digest("empty-rare-score"),
        )

    monkeypatch.setattr(phase5_scoring, "score_alignment", alignment)
    monkeypatch.setattr(phase5_scoring, "score_rare_pivotal", rare)
    plan = SimpleNamespace(
        valid_evidence_ids=(),
        predicate_aliases=(),
        alignment_plan=SimpleNamespace(content_hash=digest("empty-alignment-plan")),
        rare_annotations=(),
        content_hash=digest("empty-scorer-plan"),
    )
    source = SimpleNamespace(
        scorer_plan=plan,
        gold_projection=SimpleNamespace(content_hash=digest("empty-gold")),
        content_hash=digest("empty-source"),
    )
    projection = SimpleNamespace(
        instance_graph=SimpleNamespace(assertions=()),
        content_hash=digest("empty-projection"),
    )

    score = _projection_gold_score(projection, source)

    assert invalid_flags == [True, True]
    assert score.structurally_valid is True
    assert score.content_bearing is False
    assert score.strict_qualified_assertion_f1 == 0.0
    assert score.rare_pivotal_recall == 0.0


def metric_record(
    episode: str,
    condition: ConditionName,
    *,
    ordinal: int,
    c2_status: FeedbackAttemptStatus | None = None,
    capability_limited: bool = False,
) -> Phase5ScriptedFeedbackMetricRecord:
    before = None if capability_limited else gold_score(
        f"before-{episode}-{condition.value}", strict=0.75, rare=0.5
    )
    unsuccessful = c2_status in {
        FeedbackAttemptStatus.INVALID,
        FeedbackAttemptStatus.FAILED,
        FeedbackAttemptStatus.TIMED_OUT,
    }
    after = (
        None
        if capability_limited or unsuccessful
        else gold_score(f"after-{episode}-{condition.value}", strict=0.875, rare=0.75)
    )
    before_f1 = None if before is None else before.strict_qualified_assertion_f1
    before_rare = None if before is None else before.rare_pivotal_recall
    after_f1 = 0.0 if unsuccessful else (
        None if after is None else after.strict_qualified_assertion_f1
    )
    after_rare = 0.0 if unsuccessful else (
        None if after is None else after.rare_pivotal_recall
    )
    score = score_scripted_feedback(
        episode_id=f"{episode}--{condition.value.lower()}",
        before_strict_f1=before_f1,
        after_strict_f1=after_f1,
        required_target_change_ids=("required",),
        realized_target_change_ids=(
            () if capability_limited or unsuccessful else ("required",)
        ),
        changed_semantic_ids=(
            () if capability_limited or unsuccessful else ("required",)
        ),
        requested_scope_ids=("required",),
        unsupported_changed_ids=(),
        before_rare_pivotal_recall=before_rare,
        after_rare_pivotal_recall=after_rare,
        capability_limited=capability_limited,
        latency_seconds=1.0,
        replay_hash_success=not capability_limited and not unsuccessful,
    )
    return Phase5ScriptedFeedbackMetricRecord(
        episode_id=episode,
        condition=condition,
        scorer_binding_key=f"binding-{episode}",
        execution_record_hash=digest(f"record-{ordinal}"),
        execution_hash=digest(f"execution-{ordinal}"),
        instruction_hash=digest(f"instruction-{episode}"),
        before_projection_hash=digest(f"before-projection-{ordinal}"),
        after_projection_hash=(
            None
            if capability_limited or unsuccessful
            else digest(f"after-projection-{ordinal}")
        ),
        c2_attempt_status=c2_status,
        itt_zero_endpoint_applied=unsuccessful,
        before_gold_projection_hash=digest(f"before-gold-{episode}"),
        after_gold_projection_hash=digest(f"after-gold-{episode}"),
        before_gold_source_artifact_hash=digest(f"before-source-{episode}"),
        after_gold_source_artifact_hash=digest(f"after-source-{episode}"),
        required_target_change_hash=digest(f"target-{episode}"),
        required_target_source_artifact_hash=digest(f"target-source-{episode}"),
        diff_hash=(
            None if capability_limited or unsuccessful else digest(f"diff-{ordinal}")
        ),
        before_gold_score=before,
        after_gold_score=after,
        before_strict_f1_endpoint=before_f1,
        after_strict_f1_endpoint=after_f1,
        before_rare_pivotal_endpoint=before_rare,
        after_rare_pivotal_endpoint=after_rare,
        score=score,
        compiled_at=NOW,
    )


@pytest.mark.parametrize(
    "status",
    (
        FeedbackAttemptStatus.FAILED,
        FeedbackAttemptStatus.TIMED_OUT,
        FeedbackAttemptStatus.INVALID,
    ),
)
def test_unsuccessful_c2_itt_rows_use_zero_endpoint_without_projection(status) -> None:
    record = metric_record("script-1", ConditionName.C2_LLM_QUERY, ordinal=1, c2_status=status)

    assert record.after_projection_hash is None
    assert record.after_gold_score is None
    assert record.diff_hash is None
    assert record.after_strict_f1_endpoint == 0.0
    assert record.after_rare_pivotal_endpoint == 0.0
    assert record.score.strict_f1_change == -0.75
    assert record.score.target_change_recall.value == 0.0
    assert record.score.capability_limited is False


def test_exact_eighteen_rows_include_cpu_capability_limits_and_c2_itt_failures() -> None:
    statuses = (
        FeedbackAttemptStatus.FAILED,
        FeedbackAttemptStatus.TIMED_OUT,
        FeedbackAttemptStatus.INVALID,
        FeedbackAttemptStatus.SUCCEEDED,
        FeedbackAttemptStatus.SUCCEEDED,
        FeedbackAttemptStatus.SUCCEEDED,
    )
    records = []
    ordinal = 0
    for episode_index, status in enumerate(statuses, 1):
        episode = f"script-{episode_index}"
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        ):
            ordinal += 1
            records.append(
                metric_record(
                    episode,
                    condition,
                    ordinal=ordinal,
                    c2_status=status if condition is ConditionName.C2_LLM_QUERY else None,
                    capability_limited=(
                        condition in {
                            ConditionName.C0_CLASSICAL_PRE,
                            ConditionName.C1_LLM_PRE,
                        }
                        and episode_index % 2 == 0
                    ),
                )
            )
    metrics = Phase5ScriptedFeedbackMetrics(
        metrics_id="TEST-ONLY-phase5-metrics",
        protocol_hash=digest("protocol"),
        source_manifest_hash=digest("source"),
        known_answer_source_hash=digest("known"),
        input_manifest_hash=digest("inputs"),
        feedback_manifest_hash=digest("feedback"),
        execution_index_hash=digest("index"),
        scorer_binding_authorization_hash=digest("authorization"),
        records=tuple(records),
        generated_at=NOW,
    )

    assert len(metrics.records) == 18
    assert sum(item.itt_zero_endpoint_applied for item in metrics.records) == 3
    assert sum(item.score.capability_limited for item in metrics.records) == 6


def test_unsuccessful_c2_row_cannot_fabricate_after_projection() -> None:
    record = metric_record(
        "script-1",
        ConditionName.C2_LLM_QUERY,
        ordinal=1,
        c2_status=FeedbackAttemptStatus.FAILED,
    )
    with pytest.raises(ValidationError, match="no fabricated after projection"):
        Phase5ScriptedFeedbackMetricRecord.model_validate(
            {
                **record.model_dump(mode="python", exclude={"content_hash"}),
                "after_projection_hash": digest("fabricated"),
            }
        )
