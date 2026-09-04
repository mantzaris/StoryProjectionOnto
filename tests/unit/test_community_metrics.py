from __future__ import annotations

import pytest
from pydantic import ValidationError

from story_projection_onto.metrics.community import (
    LeidenConfiguration,
    adjusted_mutual_information,
    conductance,
    cross_seed_stability,
    run_leiden_sensitivity,
    score_partition,
    variation_of_information,
)
from story_projection_onto.metrics.entropy import SemanticEdge


def edge(edge_id: str, source: str, target: str) -> SemanticEdge:
    return SemanticEdge(
        edge_id=edge_id,
        source_id=source,
        target_id=target,
        native_predicate_id="related",
        canonical_predicate_id="related_to",
    )


def test_identical_partition_has_perfect_ami_purity_and_pairwise_errors() -> None:
    gold = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    predicted = {"a": "p1", "b": "p1", "c": "p2", "d": "p2"}
    result = score_partition(
        gold_assignments=gold,
        predicted_assignments=predicted,
        anchors=("a", "b", "c", "d"),
    )
    assert result.adjusted_mutual_information == pytest.approx(1)
    assert result.purity == pytest.approx(1)
    assert result.fragmentation_error == pytest.approx(0)
    assert result.merging_error == pytest.approx(0)
    assert result.predicted_cluster_count == 2
    assert result.schema_version == "1.0.0"
    assert len(result.content_hash) == 64
    with pytest.raises(ValidationError, match="frozen"):
        result.anchor_count = 5


def test_singletons_expose_purity_inflation_beside_fragmentation_and_cluster_count() -> None:
    gold = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    predicted = {node: f"singleton-{node}" for node in gold}
    result = score_partition(
        gold_assignments=gold,
        predicted_assignments=predicted,
        anchors=tuple(gold),
    )
    assert result.purity == 1
    assert result.predicted_cluster_count == 4
    assert result.fragmentation_error == 1
    assert result.merging_error == 0


def test_omitted_anchor_receives_unique_absent_label() -> None:
    result = score_partition(
        gold_assignments={"a": "g1", "b": "g1", "c": "g2"},
        predicted_assignments={"a": "p1", "b": "p1"},
        anchors=("a", "b", "c"),
    )
    assert result.omitted_anchor_count == 1
    assert result.predicted_cluster_count == 2
    assert result.adjusted_mutual_information is not None


def test_absent_assignment_namespace_is_reserved_for_scorer_omissions() -> None:
    with pytest.raises(ValueError, match="reserved"):
        score_partition(
            gold_assignments={"a": "g1"},
            predicted_assignments={"a": "ABSENT::forged"},
            anchors=("a",),
        )


def test_conductance_reports_zero_for_disconnected_supported_communities() -> None:
    nodes = ("a", "b", "c", "d")
    edges = (edge("e1", "a", "b"), edge("e2", "c", "d"))
    result = conductance(
        nodes,
        edges,
        {"a": "left", "b": "left", "c": "right", "d": "right"},
    )
    assert result.unweighted_mean == 0
    assert result.defined_cluster_count == 2
    assert result.undefined_cluster_count == 0


def test_zero_volume_conductance_is_na_with_counts() -> None:
    result = conductance(("a", "b"), (), {"a": "one", "b": "two"})
    assert result.unweighted_mean is None
    assert result.defined_cluster_count == 0
    assert result.undefined_cluster_count == 2


def test_cross_seed_stability_uses_fixed_anchor_universe_and_absence() -> None:
    result = cross_seed_stability(
        seed_a_assignments={"a": "one", "b": "one"},
        seed_b_assignments={"a": "alpha", "b": "alpha"},
        anchors=("a", "b", "c"),
    )
    assert result.anchor_count == 3
    assert result.omitted_seed_a_count == 1
    assert result.omitted_seed_b_count == 1
    assert result.adjusted_mutual_information == pytest.approx(1)
    assert result.variation_of_information == pytest.approx(0)


def test_ami_and_vi_empty_and_mismatched_cases_are_explicit() -> None:
    assert adjusted_mutual_information((), ()) is None
    assert variation_of_information((), ()) is None
    with pytest.raises(ValueError, match="equal length"):
        adjusted_mutual_information(("a",), ("a", "b"))


def test_degenerate_equivalent_partitions_are_label_invariant() -> None:
    assert adjusted_mutual_information(("a", "a"), ("renamed", "renamed")) == 1


def test_leiden_runs_only_three_registered_cpm_resolutions() -> None:
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    nodes = ("a", "b", "c", "d")
    edges = (edge("e1", "a", "b"), edge("e2", "c", "d"))
    partitions = run_leiden_sensitivity(
        nodes,
        edges,
        LeidenConfiguration(resolution=0.5, seed=17),
    )
    assert tuple(result.resolution for result in partitions) == (0.25, 0.5, 1.0)
    assert all(set(result.by_node) == set(nodes) for result in partitions)
    assert all(result.seed == 17 for result in partitions)


def test_leiden_canonicalizes_node_and_edge_input_order() -> None:
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    from story_projection_onto.metrics.community import run_leiden_cpm

    configuration = LeidenConfiguration(resolution=0.5, seed=17)
    nodes = ("d", "b", "a", "c")
    edges = (edge("e2", "c", "d"), edge("e1", "a", "b"))
    first = run_leiden_cpm(nodes, edges, configuration)
    second = run_leiden_cpm(tuple(reversed(nodes)), tuple(reversed(edges)), configuration)
    assert first.assignments == second.assignments
    assert first.content_hash == second.content_hash
    assert tuple(node_id for node_id, _ in first.assignments) == tuple(sorted(nodes))
