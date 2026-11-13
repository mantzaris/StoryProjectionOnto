from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.benchmark import compile_alignment_alternatives
from story_projection_onto.contracts import (
    ConstructionOperator,
    GoldAlternativeSet,
    GoldContextualProjection,
    NarrativeCommitment,
    TemporalKind,
)
from story_projection_onto.metrics.alignment import (
    AlignmentPlan,
    AnchorKind,
    AssertionAlignmentTarget,
    CompiledGoldAlternatives,
    DecisionFamily,
    EpistemicSignature,
    ExplicitNodeAlternative,
    GroundingStatus,
    NodeAlignmentTarget,
    NodeKind,
    NormalizedDecision,
    PermissibleAssertionAlternative,
    PredictedAssertion,
    PredictedNode,
    QualifiedAssertionSignature,
    TemporalExtentSignature,
    audit_qualified_assertion_grounding,
    build_alignment_plan,
    score_alignment,
    score_ontology_decisions,
)
from story_projection_onto.metrics.common import (
    MetricStatus,
    PrecisionRecallF1,
    RateResult,
)

ROOT = Path(__file__).resolve().parents[2]
HASH_A = "a" * 64
HASH_B = "b" * 64


def test_direct_enabled_label_uses_common_causal_normalization_not_precedence():
    from story_projection_onto.metrics.alignment import _normalize_predicate

    for label in ("enabled", "enables", "causally enables", "contextual_enabled"):
        assert _normalize_predicate(label, {}) == "causally_enables"
        assert _normalize_predicate(label, {"causally_enables": "causal"}) == "causal"
    for label in ("precedes", "occurred before", "supported", "causes"):
        assert _normalize_predicate(label, {}) != "causally_enables"


def test_development_office_state_labels_normalize_without_relaxing_events():
    from story_projection_onto.metrics.alignment import _normalize_predicate

    assert _normalize_predicate("served as", {}) == _normalize_predicate("holds-office", {})
    assert _normalize_predicate("serves_as", {}) == "holds_office"
    assert _normalize_predicate("appointed to", {}) == "appointed_to"
    assert _normalize_predicate("succeeds", {}) == "succeeds"


def time_point(point: int) -> TemporalExtentSignature:
    return TemporalExtentSignature(kind=TemporalKind.POINT, point=point)


def validity(start: int, end: int) -> TemporalExtentSignature:
    return TemporalExtentSignature(kind=TemporalKind.INTERVAL, start=start, end=end)


def world_signature(*, story_point: int = 2, predicate: str = "supports"):
    return QualifiedAssertionSignature(
        predicate=predicate,
        direction="forward",
        subject_target_id="gold-node-alice",
        object_target_id="gold-node-bob",
        story_time=time_point(story_point),
        validity_time=validity(2, 4),
        epistemic=EpistemicSignature(narrative_commitment=NarrativeCommitment.WORLD_COMMITTED),
    )


def plan(*, alternatives=()) -> AlignmentPlan:
    primary = PermissibleAssertionAlternative(
        alternative_id="assertion-1:primary",
        signature=world_signature(),
        supporting_evidence_ids=("evidence-1",),
    )
    return AlignmentPlan(
        matcher_revision="matcher-v1",
        source_gold_hash=HASH_A,
        source_alternative_set_hash=HASH_B,
        node_targets=(
            NodeAlignmentTarget(
                target_id="gold-node-alice",
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                permissible_anchor_sets=(("mention-alice",),),
            ),
            NodeAlignmentTarget(
                target_id="gold-node-bob",
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                permissible_anchor_sets=(("mention-bob",),),
            ),
        ),
        assertion_targets=(
            AssertionAlignmentTarget(
                target_id="assertion-1",
                alternatives=(primary, *alternatives),
                essential_temporal=True,
            ),
        ),
    )


def predicted_nodes() -> tuple[PredictedNode, ...]:
    return (
        PredictedNode(
            prediction_id="local-alice",
            kind=NodeKind.ENTITY,
            anchor_kind=AnchorKind.MENTION,
            anchor_ids=("mention-alice",),
        ),
        PredictedNode(
            prediction_id="local-bob",
            kind=NodeKind.ENTITY,
            anchor_kind=AnchorKind.MENTION,
            anchor_ids=("mention-bob",),
        ),
    )


def predicted_assertion(
    *,
    signature=None,
    evidence_ids=("evidence-1",),
    valid_evidence_ids=("evidence-1",),
    grounding=GroundingStatus.SUPPORTED,
) -> PredictedAssertion:
    return PredictedAssertion(
        prediction_id="local-assertion",
        signature=signature or world_signature(),
        evidence_ids=evidence_ids,
        valid_evidence_ids=valid_evidence_ids,
        grounding_status=grounding,
    )


def test_contextual_node_and_strict_qualified_assertion_f1_are_anchor_based() -> None:
    result = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(),),
    )
    assert result.node_score.f1 == 1.0
    assert result.strict_assertion_score.f1 == 1.0
    assert result.essential_temporal_accuracy.value == 1.0
    assert result.evidence_citation_validity.value == 1.0
    assert result.grounding_precision.value == 1.0
    assert result.unsupported_assertion_rate.value == 0.0
    assert result.strict_assertion_matches[0].target_id == "assertion-1"


def test_metric_records_reject_values_inconsistent_with_exposed_counts() -> None:
    with pytest.raises(ValidationError, match="precision does not match"):
        PrecisionRecallF1(
            true_positive_count=1,
            false_positive_count=1,
            false_negative_count=0,
            predicted_count=2,
            gold_count=1,
            precision_denominator=2,
            recall_denominator=1,
            precision=0.9,
            recall=1.0,
            f1=0.95,
            status=MetricStatus.DEFINED,
        )
    with pytest.raises(ValidationError, match="does not match"):
        RateResult(
            numerator=1,
            denominator=2,
            value=0.9,
            status=MetricStatus.DEFINED,
        )


def test_contested_holder_scope_is_scoreable_but_world_commitment_forbids_it() -> None:
    contested = EpistemicSignature(
        holder_target_id="gold-node-alice",
        attitude="uncertain",
        holder_relative_time=time_point(2),
        narrative_commitment=NarrativeCommitment.CONTESTED,
    )
    assert contested.holder_target_id == "gold-node-alice"
    with pytest.raises(ValidationError, match="world-committed"):
        EpistemicSignature(
            holder_target_id="gold-node-alice",
            attitude="known",
            holder_relative_time=time_point(2),
            narrative_commitment=NarrativeCommitment.WORLD_COMMITTED,
        )


@pytest.mark.parametrize(
    "changed_signature",
    [
        world_signature(predicate="opposes"),
        QualifiedAssertionSignature(
            predicate="supports",
            direction="inverse",
            subject_target_id="gold-node-alice",
            object_target_id="gold-node-bob",
            story_time=time_point(2),
            validity_time=validity(2, 4),
            epistemic=EpistemicSignature(narrative_commitment=NarrativeCommitment.WORLD_COMMITTED),
        ),
        QualifiedAssertionSignature(
            predicate="supports",
            direction="forward",
            subject_target_id="gold-node-bob",
            object_target_id="gold-node-alice",
            story_time=time_point(2),
            validity_time=validity(2, 4),
            epistemic=EpistemicSignature(narrative_commitment=NarrativeCommitment.WORLD_COMMITTED),
        ),
    ],
)
def test_strict_assertion_requires_predicate_direction_and_endpoints(
    changed_signature: QualifiedAssertionSignature,
) -> None:
    result = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(signature=changed_signature),),
    )
    assert result.strict_assertion_score.f1 == 0.0


def test_strict_assertion_requires_essential_time_epistemic_and_valid_support() -> None:
    wrong_time = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(signature=world_signature(story_point=3)),),
    )
    assert wrong_time.strict_assertion_score.f1 == 0.0
    assert wrong_time.structurally_aligned_assertion_matches
    assert wrong_time.essential_temporal_accuracy.value == 0.0

    attributed = QualifiedAssertionSignature(
        predicate="supports",
        direction="forward",
        subject_target_id="gold-node-alice",
        object_target_id="gold-node-bob",
        story_time=time_point(2),
        validity_time=validity(2, 4),
        epistemic=EpistemicSignature(
            holder_target_id="gold-node-alice",
            attitude="reported",
            holder_relative_time=time_point(2),
            narrative_commitment=NarrativeCommitment.HOLDER_ATTRIBUTED,
        ),
    )
    wrong_epistemic = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(signature=attributed),),
    )
    assert wrong_epistemic.strict_assertion_score.f1 == 0.0

    mixed_citations = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(
            predicted_assertion(
                evidence_ids=("evidence-1", "future-evidence"),
                valid_evidence_ids=("evidence-1",),
            ),
        ),
    )
    assert mixed_citations.strict_assertion_score.f1 == 1.0
    assert mixed_citations.evidence_citation_validity.value == 0.5

    unsupported_reference = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(
            predicted_assertion(
                evidence_ids=("future-evidence",),
                valid_evidence_ids=(),
                grounding=GroundingStatus.UNSUPPORTED,
            ),
        ),
    )
    assert unsupported_reference.strict_assertion_score.f1 == 0.0

    with pytest.raises(ValidationError, match="valid evidence citation"):
        predicted_assertion(
            evidence_ids=("future-evidence",),
            valid_evidence_ids=(),
            grounding=GroundingStatus.SUPPORTED,
        )


def test_grounding_audit_requires_exact_qualification_and_only_reviewed_support() -> None:
    attributed = QualifiedAssertionSignature(
        predicate="supports",
        direction="forward",
        subject_target_id="gold-node-alice",
        object_target_id="gold-node-bob",
        story_time=time_point(2),
        validity_time=validity(2, 4),
        epistemic=EpistemicSignature(
            holder_target_id="gold-node-alice",
            attitude="reported",
            holder_relative_time=time_point(2),
            narrative_commitment=NarrativeCommitment.HOLDER_ATTRIBUTED,
        ),
    )
    assertions = (
        predicted_assertion().model_copy(update={"prediction_id": "exact"}),
        predicted_assertion(signature=world_signature(story_point=3)).model_copy(
            update={"prediction_id": "wrong-time"}
        ),
        predicted_assertion(signature=attributed).model_copy(
            update={"prediction_id": "wrong-holder-scope"}
        ),
        predicted_assertion(
            evidence_ids=("evidence-1", "irrelevant-in-packet"),
            valid_evidence_ids=("evidence-1", "irrelevant-in-packet"),
        ).model_copy(update={"prediction_id": "extra-citation"}),
        predicted_assertion(
            evidence_ids=("evidence-1", "outside-packet"),
            valid_evidence_ids=("evidence-1",),
        ).model_copy(update={"prediction_id": "invalid-citation"}),
    )

    statuses = dict(
        audit_qualified_assertion_grounding(
            plan=plan(),
            predicted_assertions=assertions,
        )
    )

    assert statuses == {
        "exact": GroundingStatus.SUPPORTED,
        "extra-citation": GroundingStatus.UNSUPPORTED,
        "invalid-citation": GroundingStatus.UNSUPPORTED,
        "wrong-holder-scope": GroundingStatus.UNSUPPORTED,
        "wrong-time": GroundingStatus.UNSUPPORTED,
    }


def test_explicit_permissible_alternative_can_match_without_relaxing_other_targets() -> None:
    alternative = PermissibleAssertionAlternative(
        alternative_id="assertion-1:reviewed-synonym",
        signature=world_signature(predicate="aids"),
        supporting_evidence_ids=("evidence-1",),
    )
    result = score_alignment(
        plan=plan(alternatives=(alternative,)),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(signature=world_signature(predicate="aids")),),
    )
    assert result.strict_assertion_score.f1 == 1.0
    assert result.strict_assertion_matches[0].alternative_id.endswith("reviewed-synonym")


def test_compiled_alternative_contract_requires_every_source_to_change_a_target() -> None:
    compiled = CompiledGoldAlternatives(
        source_alternative_set_hash=HASH_B,
        consumed_alternative_ids=("alternative-1",),
        node_alternatives=(
            ExplicitNodeAlternative(
                source_alternative_id="alternative-1",
                target_id="gold-node-alice",
                alternative_anchor_ids=("mention-alice-alias",),
            ),
        ),
    )
    assert compiled.consumed_alternative_ids == ("alternative-1",)
    with pytest.raises(ValidationError, match="every consumed"):
        CompiledGoldAlternatives(
            source_alternative_set_hash=HASH_B,
            consumed_alternative_ids=("alternative-1", "alternative-2"),
            node_alternatives=compiled.node_alternatives,
        )


def test_unsupported_and_unknown_grounding_are_exposed_separately() -> None:
    unsupported = predicted_assertion(grounding=GroundingStatus.UNSUPPORTED)
    unknown = PredictedAssertion(
        prediction_id="local-assertion-unknown",
        signature=world_signature(predicate="opposes"),
        evidence_ids=("evidence-1",),
        valid_evidence_ids=("evidence-1",),
        grounding_status=GroundingStatus.UNKNOWN,
    )
    result = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(unsupported, unknown),
    )
    assert result.grounding_precision.value == 0.0
    assert result.unsupported_assertion_rate.value == 0.5
    assert result.strict_assertion_score.f1 == 0.0


def test_invalid_output_is_semantic_zero_and_empty_secondary_rates_are_na() -> None:
    result = score_alignment(
        plan=plan(),
        predicted_nodes=(),
        predicted_assertions=(),
        invalid_semantic_output=True,
    )
    assert result.node_score.f1 == 0.0
    assert result.node_score.status is MetricStatus.INVALID_SEMANTIC_ZERO
    assert result.strict_assertion_score.f1 == 0.0
    assert result.evidence_citation_validity.value is None
    assert result.grounding_precision.value is None
    assert result.unsupported_assertion_rate.value is None

    invalid_with_parseable_fragments = score_alignment(
        plan=plan(),
        predicted_nodes=predicted_nodes(),
        predicted_assertions=(predicted_assertion(),),
        invalid_semantic_output=True,
    )
    assert invalid_with_parseable_fragments.node_score.f1 == 0.0
    assert invalid_with_parseable_fragments.strict_assertion_score.f1 == 0.0
    assert invalid_with_parseable_fragments.evidence_citation_validity.value is None
    assert invalid_with_parseable_fragments.grounding_precision.value is None


def decision(
    family: DecisionFamily,
    signature: str,
    anchors=("evidence-1",),
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


def test_ontology_decision_macro_uses_six_families_and_union_activation() -> None:
    gold = (
        decision(DecisionFamily.MERGE_SPLIT, "merge"),
        decision(DecisionFamily.EVENT_REIFICATION, "reify", ("evidence-2",)),
    )
    predicted = (
        decision(DecisionFamily.MERGE_SPLIT, "merge"),
        decision(DecisionFamily.SCHEMA_RELATION, "invented", ("evidence-3",)),
    )
    result = score_ontology_decisions(gold=gold, predicted=predicted)
    assert len(result.components) == 6
    assert result.active_family_count == 3
    assert result.macro_f1 == pytest.approx(1 / 3)
    by_family = {item.family: item for item in result.components}
    assert by_family[DecisionFamily.MERGE_SPLIT].score.f1 == 1.0
    assert by_family[DecisionFamily.EVENT_REIFICATION].score.f1 == 0.0
    assert by_family[DecisionFamily.SCHEMA_RELATION].score.f1 == 0.0
    assert by_family[DecisionFamily.ABSTRACTION].score.status is MetricStatus.NOT_APPLICABLE


def test_ontology_decision_invalid_output_is_zero_for_active_gold_families() -> None:
    result = score_ontology_decisions(
        gold=(decision(DecisionFamily.CONTEXTUAL_TYPE, "type"),),
        predicted=(),
        invalid_semantic_output=True,
    )
    assert result.macro_f1 == 0.0
    component = next(
        item for item in result.components if item.family is DecisionFamily.CONTEXTUAL_TYPE
    )
    assert component.score.status is MetricStatus.INVALID_SEMANTIC_ZERO


def test_duplicate_predicted_decisions_are_not_hidden_by_set_deduplication() -> None:
    gold = (decision(DecisionFamily.MERGE_SPLIT, "merged"),)
    predicted = (gold[0], gold[0])
    result = score_ontology_decisions(gold=gold, predicted=predicted)
    component = next(
        item for item in result.components if item.family is DecisionFamily.MERGE_SPLIT
    )
    assert component.score.true_positive_count == 1
    assert component.score.false_positive_count == 1
    assert component.score.f1 == pytest.approx(2 / 3)


def test_generated_gold_and_reviewed_alternative_compile_to_anchor_plan() -> None:
    payload = json.loads(
        (ROOT / "data/synthetic/scorer_only/development/syn-dev-01.json").read_text()
    )
    gold = GoldContextualProjection.model_validate(payload["gold_projections"][0])
    alternatives = GoldAlternativeSet.model_validate(payload["alternatives"][0])
    compiled_alternatives = compile_alignment_alternatives(gold, alternatives)
    compiled = build_alignment_plan(
        gold,
        alternatives,
        compiled_alternatives=compiled_alternatives,
    )
    assert compiled.scorer_namespace == "scorer_only"
    assert compiled.source_gold_hash == gold.content_hash
    assert compiled.source_alternative_set_hash == alternatives.content_hash
    assert compiled.node_targets
    assert compiled.assertion_targets
