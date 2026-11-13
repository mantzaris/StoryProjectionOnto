from __future__ import annotations

import pytest

from story_projection_onto.contracts import (
    ConstructionOperator,
    GoldContrastDecision,
    GoldContrastDirection,
)
from story_projection_onto.metrics.alignment import DecisionFamily, NormalizedDecision
from story_projection_onto.metrics.common import MetricStatus
from story_projection_onto.metrics.contrastive import (
    ChangeDirection,
    SignedDecisionChange,
    derive_signed_changes,
    normalize_gold_contrast,
    score_contrastive_changes,
    score_paraphrase_stability,
)
from story_projection_onto.metrics.rare import (
    RarePivotalAnnotation,
    RarityPivotalityStratum,
    RunArtifactFingerprint,
    ScorerLabelSet,
    audit_gold_label_firewall,
    score_rare_pivotal,
)


def decision(
    family: DecisionFamily,
    signature: str,
    anchors=("mention-1",),
    *,
    operator: ConstructionOperator | None = None,
    slot_key: str | None = None,
) -> NormalizedDecision:
    default_operators = {
        DecisionFamily.MERGE_SPLIT: ConstructionOperator.MERGE,
        DecisionFamily.CONTEXTUAL_TYPE: ConstructionOperator.CONTEXTUAL_TYPE,
        DecisionFamily.SCHEMA_RELATION: ConstructionOperator.SCHEMA_RELATION,
        DecisionFamily.EVENT_REIFICATION: ConstructionOperator.EVENT_REIFICATION,
        DecisionFamily.ABSTRACTION: ConstructionOperator.ABSTRACTION,
        DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION: (
            ConstructionOperator.TEMPORAL_QUALIFICATION
        ),
    }
    return NormalizedDecision(
        slot_key=slot_key or f"{family.value}:{':'.join(anchors)}",
        family=family,
        operator=operator or default_operators[family],
        anchor_ids=anchors,
        semantic_signature=signature,
    )


def change(
    family: DecisionFamily,
    direction: ChangeDirection,
    signature: str,
    anchors=("mention-1",),
    *,
    operator: ConstructionOperator | None = None,
) -> SignedDecisionChange:
    default_operators = {
        DecisionFamily.MERGE_SPLIT: ConstructionOperator.MERGE,
        DecisionFamily.CONTEXTUAL_TYPE: ConstructionOperator.CONTEXTUAL_TYPE,
        DecisionFamily.SCHEMA_RELATION: ConstructionOperator.SCHEMA_RELATION,
        DecisionFamily.EVENT_REIFICATION: ConstructionOperator.EVENT_REIFICATION,
        DecisionFamily.ABSTRACTION: ConstructionOperator.ABSTRACTION,
        DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION: (
            ConstructionOperator.TEMPORAL_QUALIFICATION
        ),
    }
    return SignedDecisionChange(
        family=family,
        operator=operator or default_operators[family],
        direction=direction,
        anchor_ids=anchors,
        semantic_signature=signature,
    )


def test_signed_delta_derivation_records_substitution_addition_and_removal() -> None:
    before = (
        decision(DecisionFamily.MERGE_SPLIT, "merged", slot_key="identity"),
        decision(DecisionFamily.ABSTRACTION, "actor", ("evidence-2",)),
    )
    after = (
        decision(
            DecisionFamily.MERGE_SPLIT,
            "split",
            operator=ConstructionOperator.SPLIT,
            slot_key="identity",
        ),
        decision(DecisionFamily.EVENT_REIFICATION, "reified", ("evidence-3",)),
    )
    changes = derive_signed_changes(before, after)
    assert {(item.family, item.direction) for item in changes} == {
        (DecisionFamily.MERGE_SPLIT, ChangeDirection.SUBSTITUTE),
        (DecisionFamily.ABSTRACTION, ChangeDirection.REMOVE),
        (DecisionFamily.EVENT_REIFICATION, ChangeDirection.ADD),
    }
    identity_change = next(item for item in changes if item.family is DecisionFamily.MERGE_SPLIT)
    assert identity_change.operator is ConstructionOperator.SPLIT
    assert identity_change.semantic_signature == "merged=>split"


def test_gold_and_predicted_changes_share_operator_signature_and_all_anchors() -> None:
    gold = normalize_gold_contrast(
        (
            GoldContrastDecision(
                contrast_decision_id="change-1",
                operator=ConstructionOperator.SPLIT,
                direction=GoldContrastDirection.SUBSTITUTE,
                anchor_ids=("mention-1",),
                expected_signature="merged=>split",
                evidence_ids=("evidence-1",),
            ),
        )
    )
    predicted = derive_signed_changes(
        (
            decision(
                DecisionFamily.MERGE_SPLIT,
                "merged",
                ("evidence-1", "mention-1"),
                slot_key="identity",
            ),
        ),
        (
            decision(
                DecisionFamily.MERGE_SPLIT,
                "split",
                ("evidence-1", "mention-1"),
                operator=ConstructionOperator.SPLIT,
                slot_key="identity",
            ),
        ),
    )
    assert predicted == gold
    assert score_contrastive_changes(gold=gold, predicted=predicted).decision_change_score.f1 == 1.0


def test_exact_operator_change_is_not_hidden_when_final_state_text_is_equal() -> None:
    changes = derive_signed_changes(
        (
            decision(
                DecisionFamily.MERGE_SPLIT,
                "partition-state",
                operator=ConstructionOperator.MERGE,
                slot_key="identity",
            ),
        ),
        (
            decision(
                DecisionFamily.MERGE_SPLIT,
                "partition-state",
                operator=ConstructionOperator.SPLIT,
                slot_key="identity",
            ),
        ),
    )
    assert len(changes) == 1
    assert changes[0].operator is ConstructionOperator.SPLIT
    assert changes[0].semantic_signature == "partition-state=>partition-state"


def test_contrastive_change_f1_and_primary_collapse_ignore_visible_selection() -> None:
    gold = (change(DecisionFamily.EVENT_REIFICATION, ChangeDirection.ADD, "reified"),)
    correct = score_contrastive_changes(gold=gold, predicted=gold)
    assert correct.decision_change_score.f1 == 1.0
    assert not correct.ontological_collapse

    # No nonselection decision was emitted. A different visible subset would not be
    # represented here and therefore cannot prevent the registered collapse.
    collapsed = score_contrastive_changes(gold=gold, predicted=())
    assert collapsed.decision_change_score.f1 == 0.0
    assert collapsed.ontological_collapse
    assert collapsed.collapse_rate_contribution == 1.0


def test_contrastive_empty_gold_and_prediction_is_na_not_success() -> None:
    result = score_contrastive_changes(gold=(), predicted=())
    assert result.decision_change_score.status is MetricStatus.NOT_APPLICABLE
    assert result.decision_change_score.f1 is None
    assert not result.ontological_collapse


def test_invalid_contrast_is_scored_as_no_accepted_nonselection_delta() -> None:
    gold = (change(DecisionFamily.MERGE_SPLIT, ChangeDirection.ADD, "merge"),)
    attempted = (change(DecisionFamily.MERGE_SPLIT, ChangeDirection.ADD, "merge"),)
    result = score_contrastive_changes(
        gold=gold,
        predicted=attempted,
        invalid_semantic_output=True,
    )
    assert result.decision_change_score.f1 == 0.0
    assert result.decision_change_score.status is MetricStatus.INVALID_SEMANTIC_ZERO
    assert result.predicted_nonselection_change_count == 0
    assert result.ontological_collapse


def test_paraphrase_divergence_and_f1_change_use_no_unregistered_threshold() -> None:
    stable_but_wrong = score_paraphrase_stability(
        base_signature=frozenset({"same"}),
        paraphrase_signature=frozenset({"same"}),
        base_strict_f1=0.2,
        paraphrase_strict_f1=0.2,
    )
    assert stable_but_wrong.signature_divergence == 0.0

    correct = score_paraphrase_stability(
        base_signature=frozenset({"a", "b"}),
        paraphrase_signature=frozenset({"a", "c"}),
        base_strict_f1=0.8,
        paraphrase_strict_f1=0.7,
    )
    assert correct.signature_divergence == pytest.approx(2 / 3)
    assert correct.intersection_signature_size == 1
    assert correct.union_signature_size == 3
    assert correct.strict_f1_change == pytest.approx(-0.1)


def rare_annotations() -> tuple[RarePivotalAnnotation, ...]:
    return (
        RarePivotalAnnotation(
            assertion_target_id="support-1",
            is_rare=False,
            is_pivotal=False,
        ),
        RarePivotalAnnotation(
            assertion_target_id="rare-1",
            is_rare=True,
            is_pivotal=True,
            support_path_assertion_target_ids=("support-1", "rare-1"),
        ),
        RarePivotalAnnotation(
            assertion_target_id="common-1",
            is_rare=False,
            is_pivotal=True,
        ),
    )


def test_rare_pivotal_recall_requires_strict_match_and_complete_support_path() -> None:
    rare_only = score_rare_pivotal(
        annotations=rare_annotations(),
        strictly_matched_assertion_target_ids=frozenset({"rare-1"}),
    )
    assert rare_only.qualified_assertion_recall.value == 1.0
    assert rare_only.complete_support_path_survival.value == 0.0

    complete = score_rare_pivotal(
        annotations=rare_annotations(),
        strictly_matched_assertion_target_ids=frozenset({"support-1", "rare-1"}),
    )
    assert complete.complete_support_path_survival.value == 1.0


def test_rare_invalid_output_scores_zero_and_missing_denominator_is_na() -> None:
    invalid = score_rare_pivotal(
        annotations=rare_annotations(),
        strictly_matched_assertion_target_ids=frozenset({"support-1", "rare-1"}),
        invalid_semantic_output=True,
    )
    assert invalid.qualified_assertion_recall.value == 0.0
    assert invalid.qualified_assertion_recall.status is MetricStatus.INVALID_SEMANTIC_ZERO
    assert invalid.complete_support_path_survival.value == 0.0

    empty = score_rare_pivotal(
        annotations=(),
        strictly_matched_assertion_target_ids=frozenset(),
    )
    assert empty.qualified_assertion_recall.value is None
    assert empty.qualified_assertion_recall.status is MetricStatus.NOT_APPLICABLE


def test_common_rare_by_pivotal_nonpivotal_breakdown_exposes_all_denominators() -> None:
    annotations = (
        RarePivotalAnnotation(
            assertion_target_id="common-nonpivotal",
            is_rare=False,
            is_pivotal=False,
        ),
        RarePivotalAnnotation(
            assertion_target_id="common-pivotal",
            is_rare=False,
            is_pivotal=True,
        ),
        RarePivotalAnnotation(
            assertion_target_id="rare-nonpivotal",
            is_rare=True,
            is_pivotal=False,
        ),
        RarePivotalAnnotation(
            assertion_target_id="rare-pivotal",
            is_rare=True,
            is_pivotal=True,
            support_path_assertion_target_ids=("common-nonpivotal", "rare-pivotal"),
        ),
    )
    result = score_rare_pivotal(
        annotations=annotations,
        strictly_matched_assertion_target_ids=frozenset({"common-nonpivotal", "rare-pivotal"}),
    )
    assert tuple(row.stratum for row in result.stratified_qualified_assertion_recall) == (
        RarityPivotalityStratum.COMMON_NONPIVOTAL,
        RarityPivotalityStratum.COMMON_PIVOTAL,
        RarityPivotalityStratum.RARE_NONPIVOTAL,
        RarityPivotalityStratum.RARE_PIVOTAL,
    )
    assert tuple(
        row.qualified_assertion_recall.denominator
        for row in result.stratified_qualified_assertion_recall
    ) == (1, 1, 1, 1)
    assert tuple(
        row.qualified_assertion_recall.value for row in result.stratified_qualified_assertion_recall
    ) == (1.0, 0.0, 0.0, 1.0)
    assert result.complete_support_path_survival.value == 1.0
    assert len(result.content_hash) == 64


def test_duplicate_rarity_annotations_are_rejected_across_every_stratum() -> None:
    duplicated = (
        RarePivotalAnnotation(
            assertion_target_id="same",
            is_rare=False,
            is_pivotal=False,
        ),
        RarePivotalAnnotation(
            assertion_target_id="same",
            is_rare=True,
            is_pivotal=False,
        ),
    )
    with pytest.raises(ValueError, match="duplicate assertion targets"):
        score_rare_pivotal(
            annotations=duplicated,
            strictly_matched_assertion_target_ids=frozenset(),
        )


def test_support_path_must_reference_registered_annotation_targets() -> None:
    annotations = (
        RarePivotalAnnotation(
            assertion_target_id="rare",
            is_rare=True,
            is_pivotal=True,
            support_path_assertion_target_ids=("unregistered", "rare"),
        ),
    )
    with pytest.raises(ValueError, match="registered annotation target IDs"):
        score_rare_pivotal(
            annotations=annotations,
            strictly_matched_assertion_target_ids=frozenset(),
        )


def test_gold_label_mutation_cannot_change_generation_artifact_hashes() -> None:
    original = ScorerLabelSet(labels={"assertion-1": (False, False)})
    mutated = ScorerLabelSet(labels={"assertion-1": (True, True)})
    artifacts = RunArtifactFingerprint(
        request_hash="a" * 64,
        raw_output_hash="b" * 64,
        projection_hash="c" * 64,
    )
    audit = audit_gold_label_firewall(
        original_labels=original,
        mutated_labels=mutated,
        original_artifacts=artifacts,
        artifacts_after_label_mutation=artifacts,
    )
    assert audit.labels_changed
    assert audit.run_artifacts_unchanged

    changed_artifacts = RunArtifactFingerprint(
        request_hash="a" * 64,
        raw_output_hash="b" * 64,
        projection_hash="d" * 64,
    )
    with pytest.raises(ValueError, match="changed a run artifact"):
        audit_gold_label_firewall(
            original_labels=original,
            mutated_labels=mutated,
            original_artifacts=artifacts,
            artifacts_after_label_mutation=changed_artifacts,
        )
