from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import (
    AllenRelation,
    DiscoursePosition,
    EpistemicAttitude,
    EpistemicScope,
    HolderRelativeTime,
    NarrativeCommitment,
    PartialOrderConstraint,
    ProvenanceReference,
    QualifiedAssertion,
    RevelationPosition,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
    TemporalScope,
    ValidityTime,
)
from story_projection_onto.temporal import (
    TemporalDiagnosticCode,
    TemporalValidationStatus,
    discourse_is_within_horizon,
    revelation_is_within_horizon,
    validate_assertion_temporality,
    validate_partial_order,
    validate_temporal_extent,
    validate_temporal_scope,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def horizon(*, discourse: int = 10, revelation: int | None = 10) -> SpoilerHorizon:
    return SpoilerHorizon(
        horizon_id="horizon-1",
        max_discourse_position=DiscoursePosition(passage_order=discourse),
        max_revelation_position=(
            RevelationPosition(revelation_order=revelation) if revelation is not None else None
        ),
    )


def scope(
    *,
    story_point: int = 3,
    discourse: int = 2,
    revelation: int = 2,
) -> TemporalScope:
    return TemporalScope(
        story_time=StoryTime(kind=TemporalKind.POINT, point=story_point),
        validity_time=ValidityTime(kind=TemporalKind.INTERVAL, start=2, end=5),
        discourse_position=DiscoursePosition(passage_order=discourse),
        revelation_position=RevelationPosition(revelation_order=revelation),
    )


def provenance() -> ProvenanceReference:
    return ProvenanceReference(
        provenance_id="prov-1",
        evidence_id="ev-1",
        extraction_method="fixture",
        locator="fixture:1",
        source_artifact_hash=digest("fixture"),
        confidence=1.0,
    )


def test_interval_bound_contradiction_is_typed_and_does_not_repair_facts() -> None:
    reversed_interval = ValidityTime(kind=TemporalKind.INTERVAL, start=8, end=3)
    result = validate_temporal_extent(reversed_interval, field_path="validity_time")
    assert result.status is TemporalValidationStatus.CONTRADICTION
    assert [item.code for item in result.diagnostics] == [
        TemporalDiagnosticCode.INVALID_INTERVAL_BOUNDS
    ]
    assert reversed_interval.start == 8
    assert reversed_interval.end == 3


def test_open_and_exact_intervals_validate_without_fabricating_missing_bounds() -> None:
    open_interval = ValidityTime(kind=TemporalKind.INTERVAL, start=4)
    result = validate_temporal_extent(open_interval)
    assert result.status is TemporalValidationStatus.VALID
    assert open_interval.end is None

    exact_story_point = StoryTime(kind=TemporalKind.POINT, point=4)
    assert validate_temporal_extent(exact_story_point).status is TemporalValidationStatus.VALID


def test_unknown_withheld_invalid_and_not_applicable_remain_distinct() -> None:
    unknown = validate_temporal_extent(
        StoryTime(kind=TemporalKind.UNKNOWN, reason="the evidence gives no date")
    )
    withheld = validate_temporal_extent(
        StoryTime(kind=TemporalKind.HORIZON_WITHHELD, reason="revealed later")
    )
    invalid = validate_temporal_extent(
        StoryTime(kind=TemporalKind.INVALID, reason="conflicting exact dates")
    )
    not_applicable = validate_temporal_extent(StoryTime(kind=TemporalKind.NOT_APPLICABLE))

    assert unknown.status is TemporalValidationStatus.UNDERDETERMINED
    assert unknown.diagnostics[0].code is TemporalDiagnosticCode.UNKNOWN_TIME
    assert withheld.status is TemporalValidationStatus.UNDERDETERMINED
    assert withheld.diagnostics[0].code is TemporalDiagnosticCode.HORIZON_WITHHELD
    assert invalid.status is TemporalValidationStatus.CONTRADICTION
    assert invalid.diagnostics[0].code is TemporalDiagnosticCode.INVALID_EXPLICIT_TIME
    assert not_applicable.status is TemporalValidationStatus.VALID


def test_malformed_temporal_shapes_fail_contract_validation() -> None:
    with pytest.raises(ValidationError, match="point time requires point"):
        StoryTime(kind=TemporalKind.POINT)
    with pytest.raises(ValidationError, match="relative time requires"):
        StoryTime(kind=TemporalKind.RELATIVE, anchor_id="event-1")
    with pytest.raises(ValidationError, match="requires an explicit reason"):
        StoryTime(kind=TemporalKind.UNKNOWN)


def test_partial_order_accepts_dag_and_rejects_cycle() -> None:
    acyclic = (
        PartialOrderConstraint(
            left_id="event-a", relation=AllenRelation.BEFORE, right_id="event-b"
        ),
        PartialOrderConstraint(left_id="event-b", relation=AllenRelation.MEETS, right_id="event-c"),
    )
    assert (
        validate_partial_order(acyclic, known_node_ids=("event-a", "event-b", "event-c")).status
        is TemporalValidationStatus.VALID
    )

    cyclic = (
        *acyclic,
        PartialOrderConstraint(
            left_id="event-c", relation=AllenRelation.BEFORE, right_id="event-a"
        ),
    )
    result = validate_partial_order(cyclic, known_node_ids=("event-a", "event-b", "event-c"))
    assert result.status is TemporalValidationStatus.CONTRADICTION
    assert result.diagnostics[-1].code is TemporalDiagnosticCode.PARTIAL_ORDER_CYCLE
    assert result.diagnostics[-1].involved_ids == ("event-a", "event-b", "event-c")


def test_partial_order_reports_unknown_nodes_and_does_not_assume_overlap_order() -> None:
    unknown = validate_partial_order(
        (
            PartialOrderConstraint(
                left_id="event-a", relation=AllenRelation.BEFORE, right_id="event-missing"
            ),
        ),
        known_node_ids=("event-a",),
    )
    assert unknown.status is TemporalValidationStatus.CONTRADICTION
    assert unknown.diagnostics[0].code is TemporalDiagnosticCode.PARTIAL_ORDER_UNKNOWN_NODE

    overlap_pair = (
        PartialOrderConstraint(
            left_id="event-a", relation=AllenRelation.OVERLAPS, right_id="event-b"
        ),
        PartialOrderConstraint(
            left_id="event-b", relation=AllenRelation.OVERLAPS, right_id="event-a"
        ),
    )
    assert validate_partial_order(overlap_pair).status is TemporalValidationStatus.VALID


def test_story_time_is_not_compared_to_spoiler_or_discourse_coordinates() -> None:
    result = validate_temporal_scope(scope(story_point=999), horizon=horizon(discourse=10))
    assert result.status is TemporalValidationStatus.VALID
    assert result.diagnostics == ()


def test_discourse_and_revelation_horizon_failures_are_independent() -> None:
    disclosure_failure = validate_temporal_scope(
        scope(discourse=11, revelation=2),
        horizon=horizon(discourse=10, revelation=10),
    )
    assert disclosure_failure.status is TemporalValidationStatus.CONTRADICTION
    assert {item.code for item in disclosure_failure.diagnostics} == {
        TemporalDiagnosticCode.EVIDENCE_AFTER_SPOILER_HORIZON
    }

    revelation_failure = validate_temporal_scope(
        scope(discourse=2, revelation=11),
        horizon=horizon(discourse=10, revelation=10),
    )
    assert revelation_failure.status is TemporalValidationStatus.CONTRADICTION
    assert {item.code for item in revelation_failure.diagnostics} == {
        TemporalDiagnosticCode.REVELATION_AFTER_SPOILER_HORIZON
    }


def test_position_comparators_use_only_their_own_coordinates() -> None:
    registered_horizon = horizon(discourse=4, revelation=7)
    assert discourse_is_within_horizon(
        DiscoursePosition(passage_order=4, sentence_order=0), registered_horizon
    )
    assert not discourse_is_within_horizon(
        DiscoursePosition(passage_order=4, sentence_order=1), registered_horizon
    )
    assert revelation_is_within_horizon(RevelationPosition(revelation_order=7), registered_horizon)
    assert not revelation_is_within_horizon(
        RevelationPosition(revelation_order=8), registered_horizon
    )


def test_assertion_holder_time_is_validated_without_global_truth_promotion() -> None:
    epistemic = EpistemicScope(
        holder_id="entity-mira",
        attitude=EpistemicAttitude.BELIEVED,
        proposition_content_id="proposition-1",
        holder_relative_time=HolderRelativeTime(
            kind=TemporalKind.UNKNOWN,
            reason="the report does not say when Mira formed the belief",
        ),
        evidence_ids=("ev-1",),
    )
    assertion = QualifiedAssertion(
        assertion_id="assertion-1",
        proposition_content_id="proposition-1",
        predicate_id="believed",
        subject_id="entity-mira",
        object_id="proposition-1",
        temporal_scope=scope(),
        epistemic_scope=epistemic,
        narrative_commitment=NarrativeCommitment.HOLDER_ATTRIBUTED,
        confidence=0.8,
        evidence_ids=("ev-1",),
        provenance=(provenance(),),
        contextual_relevance=0.9,
        why_matters="The belief explains Mira's action without endorsing its content.",
        why_matters_evidence_ids=("ev-1",),
    )
    result = validate_assertion_temporality(assertion, horizon=horizon())
    assert result.status is TemporalValidationStatus.UNDERDETERMINED
    assert result.diagnostics[0].code is TemporalDiagnosticCode.UNKNOWN_TIME
    assert assertion.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED
