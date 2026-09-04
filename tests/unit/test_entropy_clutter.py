from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from story_projection_onto.metrics.clutter import (
    LabelRectangle,
    Point,
    clutter_profile,
    crossing_measure,
)
from story_projection_onto.metrics.entropy import SemanticEdge, entropy_profile


def edge(
    edge_id: str,
    source: str,
    target: str,
    native: str = "native-related",
    canonical: str | None = "related_to",
) -> SemanticEdge:
    return SemanticEdge(
        edge_id=edge_id,
        source_id=source,
        target_id=target,
        native_predicate_id=native,
        canonical_predicate_id=canonical,
    )


def test_entropy_panel_uses_simple_topology_but_all_relation_assertions() -> None:
    nodes = ("a", "b", "c", "isolated")
    edges = (
        edge("e1", "a", "b", "allied-with", "related_to"),
        edge("e2", "a", "b", "trusts", "related_to"),
        edge("e3", "b", "c", "reported-to", "reports"),
        edge("e4", "c", "c", "self-state", None),
    )
    result = entropy_profile(
        nodes,
        edges,
        canonical_vocabulary=("related_to", "reports", "OTHER"),
    )

    assert result.node_count == 4
    assert result.topology_edge_count == 2
    assert result.assertion_edge_count == 4
    assert result.isolate_count == 1
    assert result.positive_degree_node_count == 3
    assert result.canonical_other_count == 1
    assert result.canonical_other_rate == pytest.approx(0.25)
    assert result.degree_histogram.defined
    expected_degree_entropy = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert result.degree_histogram.raw == pytest.approx(expected_degree_entropy)
    assert result.degree_histogram.normalization_size == 3
    assert result.degree_histogram.normalized == pytest.approx(
        result.degree_histogram.raw / math.log(3)
    )
    assert result.degree_mass.defined
    assert result.degree_mass.denominator == 2 * result.topology_edge_count
    assert result.degree_mass.raw == pytest.approx(expected_degree_entropy)
    assert result.degree_mass.normalization_size == 3
    assert result.degree_mass.normalized == pytest.approx(result.degree_mass.raw / math.log(3))
    assert result.mean_local_relation_neighborhood.support_size == 3
    expected_local_entropy = (
        0.0 - (2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3) + math.log(2)
    ) / 3
    assert result.mean_local_relation_neighborhood.raw == pytest.approx(expected_local_entropy)
    assert result.mean_local_relation_neighborhood.normalization_size == 3
    assert result.mean_local_relation_neighborhood.normalized == pytest.approx(
        result.mean_local_relation_neighborhood.raw / math.log(3)
    )
    assert result.native_schema_relation.normalized is not None
    assert result.native_schema_relation.denominator == result.assertion_edge_count
    assert result.canonical_mapped_relation.normalized is not None
    assert result.canonical_mapped_relation.denominator == result.assertion_edge_count
    assert len(result.content_hash) == 64
    assert result.schema_version == "1.0.0"
    with pytest.raises(ValidationError, match="frozen"):
        result.node_count = 99


def test_one_relation_category_normalizes_to_zero_and_empty_assertions_are_na() -> None:
    singleton = entropy_profile(
        ("a", "b"),
        (edge("e1", "a", "b"),),
        canonical_vocabulary=("related_to", "OTHER"),
    )
    assert singleton.native_schema_relation.raw == 0
    assert singleton.native_schema_relation.normalized == 0

    empty = entropy_profile(
        ("a", "b"),
        (),
        canonical_vocabulary=("related_to", "OTHER"),
    )
    assert not empty.degree_mass.defined
    assert not empty.native_schema_relation.defined
    assert not empty.canonical_mapped_relation.defined
    assert empty.canonical_other_rate is None


def test_crossing_two_part_rule_never_imputes_zero_when_no_opportunity() -> None:
    positions = {
        "a": Point(x=0, y=0),
        "b": Point(x=1, y=1),
        "c": Point(x=0, y=1),
    }
    result = crossing_measure(
        (edge("e1", "a", "b"), edge("e2", "a", "c")),
        positions,
    )
    assert result.opportunity_indicator == 0
    assert result.opportunity_count == 0
    assert result.crossing_count == 0
    assert result.conditional_crossing_rate is None


def test_crossing_count_uses_nonadjacent_edge_opportunities() -> None:
    positions = {
        "a": Point(x=0, y=0),
        "b": Point(x=1, y=1),
        "c": Point(x=0, y=1),
        "d": Point(x=1, y=0),
        "e": Point(x=2, y=0),
    }
    result = crossing_measure(
        (
            edge("e1", "a", "b"),
            edge("e2", "c", "d"),
            edge("e3", "b", "e"),
        ),
        positions,
    )
    assert result.opportunity_indicator == 1
    assert result.opportunity_count == 2
    assert result.crossing_count == 1
    assert result.conditional_crossing_rate == pytest.approx(0.5)


def test_crossing_geometry_collapses_parallel_and_reverse_assertion_edges() -> None:
    positions = {
        "a": Point(x=0, y=0),
        "b": Point(x=1, y=1),
        "c": Point(x=0, y=1),
        "d": Point(x=1, y=0),
    }
    result = crossing_measure(
        (
            edge("e1", "a", "b"),
            edge("e2", "b", "a"),
            edge("e3", "a", "b", native="reported"),
            edge("e4", "c", "d"),
        ),
        positions,
    )
    assert result.opportunity_count == 1
    assert result.crossing_count == 1
    assert result.conditional_crossing_rate == 1


def test_clutter_profile_reports_geometry_content_and_discoverability_together() -> None:
    nodes = ("a", "b", "c", "d")
    edges = (edge("e1", "a", "b"), edge("e2", "c", "d"))
    positions = {
        "a": Point(x=0, y=0),
        "b": Point(x=1, y=1),
        "c": Point(x=0, y=1),
        "d": Point(x=1, y=0),
    }
    rectangles = (
        LabelRectangle(label_id="a", left=0, top=0, right=2, bottom=1),
        LabelRectangle(label_id="b", left=1, top=0.5, right=3, bottom=1.5),
        LabelRectangle(label_id="c", left=5, top=5, right=6, bottom=6),
    )
    result = clutter_profile(
        nodes,
        edges,
        positions=positions,
        label_rectangles=rectangles,
        visible_semantic_ids=("a", "b", "e1", "e2"),
        irrelevant_semantic_ids=frozenset({"b", "e2"}),
        rare_pivotal_discoverability={"rare-1": 2, "rare-2": 4},
        rare_pivotal_target_ids=("rare-1", "rare-2", "rare-omitted"),
    )
    assert result.valid_content_bearing
    assert result.density == pytest.approx(2 / 6)
    assert result.component_count == 2
    assert result.isolate_count == 0
    assert result.crossings.crossing_count == 1
    assert result.label_overlap_count == 1
    assert result.label_overlap_area == pytest.approx(0.5)
    assert result.irrelevant_visible_fraction == pytest.approx(0.5)
    assert result.visible_semantic_count == 4
    assert result.rare_pivotal_discoverability_denominator == 3
    assert result.rare_pivotal_discovered_count == 2
    assert result.rare_pivotal_omitted_count == 1
    assert result.rare_pivotal_discoverability_mean_interactions == pytest.approx(3)
    assert result.rare_pivotal_discoverability_max_interactions == 4
    assert result.schema_version == "1.0.0"
    assert len(result.content_hash) == 64


@pytest.mark.parametrize("structurally_valid,nodes", [(False, ("a",)), (True, ())])
def test_invalid_or_empty_output_has_no_favorable_geometry(
    structurally_valid: bool,
    nodes: tuple[str, ...],
) -> None:
    result = clutter_profile(
        nodes,
        (),
        positions={node: Point(x=0, y=0) for node in nodes},
        label_rectangles=(),
        visible_semantic_ids=(),
        irrelevant_semantic_ids=frozenset(),
        rare_pivotal_discoverability={},
        rare_pivotal_target_ids=(),
        structurally_valid=structurally_valid,
    )
    assert not result.valid_content_bearing
    assert result.density is None
    assert result.crossings.opportunity_indicator is None
    assert result.crossings.conditional_crossing_rate is None
    assert result.label_overlap_count is None
