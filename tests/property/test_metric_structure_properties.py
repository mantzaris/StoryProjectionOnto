from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from story_projection_onto.contracts import NarrativeCommitment, TemporalKind
from story_projection_onto.metrics.alignment import (
    AlignmentPlan,
    AnchorKind,
    AssertionAlignmentTarget,
    EpistemicSignature,
    GroundingStatus,
    NodeAlignmentTarget,
    NodeKind,
    PermissibleAssertionAlternative,
    PredictedAssertion,
    PredictedNode,
    QualifiedAssertionSignature,
    TemporalExtentSignature,
    score_alignment,
)
from story_projection_onto.metrics.clutter import Point, crossing_measure
from story_projection_onto.metrics.entropy import SemanticEdge, entropy_profile


def edge(edge_id: str, source: str, target: str, relation: str) -> SemanticEdge:
    return SemanticEdge(
        edge_id=edge_id,
        source_id=source,
        target_id=target,
        native_predicate_id=relation,
        canonical_predicate_id=relation,
    )


NODES = ("a", "b", "c", "d", "isolated")
EDGES = (
    edge("e1", "a", "b", "related_to"),
    edge("e2", "b", "a", "reports"),
    edge("e3", "c", "d", "related_to"),
    edge("e4", "d", "d", "reports"),
)
POSITIONS = {
    "a": Point(x=0, y=0),
    "b": Point(x=1, y=1),
    "c": Point(x=0, y=1),
    "d": Point(x=1, y=0),
    "isolated": Point(x=2, y=2),
}


@settings(max_examples=12, deadline=None)
@given(st.permutations(NODES), st.permutations(EDGES))
def test_entropy_and_crossings_are_invariant_to_node_assertion_order(
    node_permutation: list[str],
    edge_permutation: list[SemanticEdge],
) -> None:
    expected_entropy = entropy_profile(
        NODES,
        EDGES,
        canonical_vocabulary=("related_to", "reports", "OTHER"),
    )
    actual_entropy = entropy_profile(
        tuple(node_permutation),
        tuple(edge_permutation),
        canonical_vocabulary=("related_to", "reports", "OTHER"),
    )
    assert actual_entropy.content_hash == expected_entropy.content_hash

    expected_crossings = crossing_measure(EDGES, POSITIONS)
    actual_crossings = crossing_measure(tuple(edge_permutation), POSITIONS)
    assert actual_crossings.content_hash == expected_crossings.content_hash


@settings(max_examples=20, deadline=None)
@given(
    st.lists(
        st.sampled_from(("related_to", "reports", "OTHER")),
        min_size=1,
        max_size=12,
    )
)
def test_registered_entropy_normalizers_stay_in_unit_interval(relations: list[str]) -> None:
    edges = tuple(edge(f"e{index}", "a", "b", relation) for index, relation in enumerate(relations))
    result = entropy_profile(
        ("a", "b"),
        edges,
        canonical_vocabulary=("related_to", "reports", "OTHER"),
    )
    for measure in (
        result.degree_histogram,
        result.degree_mass,
        result.mean_local_relation_neighborhood,
        result.native_schema_relation,
        result.canonical_mapped_relation,
    ):
        assert measure.normalized is not None
        assert 0.0 <= measure.normalized <= 1.0


GRAPH_FAMILY = {
    "empty": ((), ()),
    "singleton": (("a",), ()),
    "disconnected": (
        ("a", "b", "c", "d"),
        (
            edge("disconnected-1", "a", "b", "related_to"),
            edge("disconnected-2", "c", "d", "reports"),
        ),
    ),
    "multiedge": (
        ("a", "b"),
        (
            edge("parallel-1", "a", "b", "related_to"),
            edge("parallel-2", "a", "b", "reports"),
            edge("reverse", "b", "a", "reports"),
            edge("self-loop", "a", "a", "related_to"),
        ),
    ),
}


@settings(max_examples=8, deadline=None)
@given(st.sampled_from(tuple(GRAPH_FAMILY)))
def test_empty_singleton_disconnected_and_multiedge_graph_family(
    family_name: str,
) -> None:
    nodes, edges = GRAPH_FAMILY[family_name]
    result = entropy_profile(
        nodes,
        edges,
        canonical_vocabulary=("related_to", "reports", "OTHER"),
    )

    assert result.node_count == len(nodes)
    assert result.assertion_edge_count == len(edges)
    assert result.topology_edge_count == {
        "empty": 0,
        "singleton": 0,
        "disconnected": 2,
        "multiedge": 1,
    }[family_name]
    assert result.isolate_count == {
        "empty": 0,
        "singleton": 1,
        "disconnected": 0,
        "multiedge": 0,
    }[family_name]
    if family_name == "empty":
        assert not result.degree_histogram.defined
        assert not result.degree_mass.defined
    elif family_name == "singleton":
        assert result.degree_histogram.defined
        assert result.degree_histogram.denominator == 1
        assert not result.degree_mass.defined
    elif family_name == "multiedge":
        # Parallel/reverse assertions remain distinct relation observations while
        # topology collapses them to one simple undirected skeleton edge.
        assert result.native_schema_relation.denominator == 4
        assert result.degree_mass.denominator == 2


def _point(point: int) -> TemporalExtentSignature:
    return TemporalExtentSignature(kind=TemporalKind.POINT, point=point)


def _interval(start: int, end: int) -> TemporalExtentSignature:
    return TemporalExtentSignature(kind=TemporalKind.INTERVAL, start=start, end=end)


def _assertion_signature(
    *,
    predicate: str,
    story_point: int,
) -> QualifiedAssertionSignature:
    return QualifiedAssertionSignature(
        predicate=predicate,
        direction="forward",
        subject_target_id="gold-alice",
        object_target_id="gold-bob",
        story_time=_point(story_point),
        validity_time=_interval(story_point, story_point + 1),
        epistemic=EpistemicSignature(
            narrative_commitment=NarrativeCommitment.WORLD_COMMITTED
        ),
    )


@settings(max_examples=12, deadline=None)
@given(
    use_node_alternative=st.booleans(),
    use_assertion_alternative=st.booleans(),
    reverse_prediction_order=st.booleans(),
)
def test_permissible_gold_alternative_family_scores_every_declared_form(
    use_node_alternative: bool,
    use_assertion_alternative: bool,
    reverse_prediction_order: bool,
) -> None:
    primary_signature = _assertion_signature(predicate="supports", story_point=2)
    alternative_signature = _assertion_signature(predicate="protects", story_point=3)
    plan = AlignmentPlan(
        matcher_revision="property-alternative-matcher-v1",
        source_gold_hash="a" * 64,
        source_alternative_set_hash="b" * 64,
        executed_alternative_ids=("assertion-alternative", "node-alternative"),
        node_targets=(
            NodeAlignmentTarget(
                target_id="gold-alice",
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                permissible_anchor_sets=(("mention-alice-alias",), ("mention-alice-primary",)),
            ),
            NodeAlignmentTarget(
                target_id="gold-bob",
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                permissible_anchor_sets=(("mention-bob",),),
            ),
        ),
        assertion_targets=(
            AssertionAlignmentTarget(
                target_id="gold-assertion",
                alternatives=(
                    PermissibleAssertionAlternative(
                        alternative_id="primary",
                        signature=primary_signature,
                        supporting_evidence_ids=("evidence-primary",),
                    ),
                    PermissibleAssertionAlternative(
                        alternative_id="permissible-alternative",
                        signature=alternative_signature,
                        supporting_evidence_ids=("evidence-alternative",),
                    ),
                ),
                essential_temporal=True,
            ),
        ),
    )
    nodes = [
        PredictedNode(
            prediction_id="predicted-alice",
            kind=NodeKind.ENTITY,
            anchor_kind=AnchorKind.MENTION,
            anchor_ids=(
                "mention-alice-alias"
                if use_node_alternative
                else "mention-alice-primary",
            ),
        ),
        PredictedNode(
            prediction_id="predicted-bob",
            kind=NodeKind.ENTITY,
            anchor_kind=AnchorKind.MENTION,
            anchor_ids=("mention-bob",),
        ),
    ]
    if reverse_prediction_order:
        nodes.reverse()
    selected_signature = (
        alternative_signature if use_assertion_alternative else primary_signature
    )
    evidence_id = (
        "evidence-alternative" if use_assertion_alternative else "evidence-primary"
    )
    assertion = PredictedAssertion(
        prediction_id="predicted-assertion",
        signature=selected_signature,
        evidence_ids=(evidence_id,),
        valid_evidence_ids=(evidence_id,),
        grounding_status=GroundingStatus.SUPPORTED,
    )

    result = score_alignment(
        plan=plan,
        predicted_nodes=tuple(nodes),
        predicted_assertions=(assertion,),
    )
    assert result.node_score.f1 == 1.0
    assert result.strict_assertion_score.f1 == 1.0
    assert result.essential_temporal_accuracy.value == 1.0
    assert result.strict_assertion_matches[0].alternative_id == (
        "permissible-alternative" if use_assertion_alternative else "primary"
    )
