"""Leiden-CPM community detection and registered anchor-aligned measures."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import ImmutableRecord
from story_projection_onto.metrics.entropy import SemanticEdge

ABSENT_ASSIGNMENT_PREFIX = "ABSENT::"
_NUMERIC_TOLERANCE = 1e-12


class LeidenConfiguration(ImmutableRecord):
    resolution: float = Field(gt=0.0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def values_are_finite(self) -> Self:
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("Leiden CPM resolution must be positive and finite")
        return self


class LeidenPartition(ImmutableRecord):
    resolution: float = Field(gt=0.0)
    seed: int = Field(ge=0)
    assignments: tuple[tuple[str, str], ...]
    cluster_count: int = Field(ge=0)
    modularity: float | None = Field(default=None, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def assignments_are_canonical(self) -> Self:
        if not math.isfinite(self.resolution):
            raise ValueError("Leiden resolution must be finite")
        node_ids = tuple(node_id for node_id, _ in self.assignments)
        labels = tuple(label for _, label in self.assignments)
        if any(not node_id for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
            raise ValueError("Leiden assignment node IDs must be nonempty and unique")
        if node_ids != tuple(sorted(node_ids)):
            raise ValueError("Leiden assignments must use canonical node order")
        _reject_reserved_or_empty_labels(labels)
        if self.cluster_count != len(set(labels)):
            raise ValueError("Leiden cluster count must match assignment labels")
        return self

    @property
    def by_node(self) -> Mapping[str, str]:
        return dict(self.assignments)


class PartitionAgreement(ImmutableRecord):
    anchor_count: int = Field(ge=0)
    gold_cluster_count: int = Field(ge=0)
    predicted_cluster_count: int = Field(ge=0)
    omitted_anchor_count: int = Field(ge=0)
    adjusted_mutual_information: float | None = Field(default=None, ge=-1.0, le=1.0)
    purity: float | None = Field(default=None, ge=0.0, le=1.0)
    fragmentation_error: float | None = Field(default=None, ge=0.0, le=1.0)
    fragmentation_pair_denominator: int = Field(ge=0)
    merging_error: float | None = Field(default=None, ge=0.0, le=1.0)
    merging_pair_denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def counts_and_pairwise_denominators_agree(self) -> Self:
        for name, count in (
            ("gold cluster", self.gold_cluster_count),
            ("predicted cluster", self.predicted_cluster_count),
            ("omitted anchor", self.omitted_anchor_count),
        ):
            if count > self.anchor_count:
                raise ValueError(f"{name} count cannot exceed anchor count")
        if self.anchor_count == 0:
            if self.gold_cluster_count != 0 or self.predicted_cluster_count != 0:
                raise ValueError("empty anchor universes cannot contain clusters")
            if self.adjusted_mutual_information is not None or self.purity is not None:
                raise ValueError("empty anchor universes require NA AMI and purity")
        elif (
            self.gold_cluster_count == 0
            or self.predicted_cluster_count == 0
            or self.adjusted_mutual_information is None
            or self.purity is None
        ):
            raise ValueError("nonempty anchor universes require clusters, AMI, and purity")
        for name, denominator, value in (
            ("fragmentation", self.fragmentation_pair_denominator, self.fragmentation_error),
            ("merging", self.merging_pair_denominator, self.merging_error),
        ):
            if (denominator == 0) != (value is None):
                raise ValueError(f"{name} error is NA exactly when its denominator is zero")
        total_pairs = self.anchor_count * (self.anchor_count - 1) // 2
        if self.fragmentation_pair_denominator + self.merging_pair_denominator != total_pairs:
            raise ValueError("fragmentation and merging denominators must partition anchor pairs")
        return self


class ConductanceResult(ImmutableRecord):
    unweighted_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    defined_cluster_count: int = Field(ge=0)
    undefined_cluster_count: int = Field(ge=0)
    by_cluster: tuple[tuple[str, float | None], ...]

    @model_validator(mode="after")
    def cluster_values_match_summary(self) -> Self:
        cluster_ids = tuple(cluster for cluster, _ in self.by_cluster)
        if cluster_ids != tuple(sorted(cluster_ids)) or len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("conductance clusters must be unique and canonically ordered")
        _reject_reserved_or_empty_labels(cluster_ids)
        values = tuple(value for _, value in self.by_cluster)
        if any(value is not None and not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("defined conductance values must lie in [0, 1]")
        defined = tuple(value for value in values if value is not None)
        if self.defined_cluster_count != len(defined):
            raise ValueError("defined conductance count does not match cluster rows")
        if self.undefined_cluster_count != len(values) - len(defined):
            raise ValueError("undefined conductance count does not match cluster rows")
        expected_mean = sum(defined) / len(defined) if defined else None
        if expected_mean is None:
            if self.unweighted_mean is not None:
                raise ValueError("no defined clusters require an NA conductance mean")
        elif self.unweighted_mean is None or not math.isclose(
            self.unweighted_mean,
            expected_mean,
            rel_tol=_NUMERIC_TOLERANCE,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError("mean conductance does not match cluster rows")
        return self


class CrossSeedStability(ImmutableRecord):
    anchor_count: int = Field(ge=0)
    adjusted_mutual_information: float | None = Field(default=None, ge=-1.0, le=1.0)
    variation_of_information: float | None = Field(default=None, ge=0.0)
    omitted_seed_a_count: int = Field(ge=0)
    omitted_seed_b_count: int = Field(ge=0)

    @model_validator(mode="after")
    def omitted_counts_and_definedness_agree(self) -> Self:
        if self.omitted_seed_a_count > self.anchor_count:
            raise ValueError("seed-A omissions cannot exceed anchor count")
        if self.omitted_seed_b_count > self.anchor_count:
            raise ValueError("seed-B omissions cannot exceed anchor count")
        metrics_are_na = (
            self.adjusted_mutual_information is None and self.variation_of_information is None
        )
        if metrics_are_na != (self.anchor_count == 0):
            raise ValueError("cross-seed metrics are NA exactly for an empty anchor universe")
        return self


def _reject_reserved_or_empty_labels(labels: Sequence[str]) -> None:
    if any(not label for label in labels):
        raise ValueError("community labels must be nonempty")
    if any(label.startswith(ABSENT_ASSIGNMENT_PREFIX) for label in labels):
        raise ValueError(f"{ABSENT_ASSIGNMENT_PREFIX} is reserved for scorer-created omissions")


def _validate_assignments(
    assignments: Mapping[str, str],
    anchors: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    if any(not node_id for node_id in assignments):
        raise ValueError("community assignment node IDs must be nonempty")
    _reject_reserved_or_empty_labels(tuple(assignments.values()))
    if any(not anchor for anchor in anchors):
        raise ValueError("gold anchor IDs must be nonempty")
    if len(anchors) != len(set(anchors)):
        raise ValueError("gold anchor universe must be unique")
    gold_anchors = tuple(anchors)
    labels = []
    omitted = 0
    for anchor in gold_anchors:
        if anchor in assignments:
            label = assignments[anchor]
        else:
            label = f"{ABSENT_ASSIGNMENT_PREFIX}{anchor}"
            omitted += 1
        labels.append(label)
    return gold_anchors, tuple(labels), omitted


def _contingency(labels_a: Sequence[str], labels_b: Sequence[str]) -> list[list[int]]:
    unique_a = tuple(sorted(set(labels_a)))
    unique_b = tuple(sorted(set(labels_b)))
    index_a = {label: index for index, label in enumerate(unique_a)}
    index_b = {label: index for index, label in enumerate(unique_b)}
    table = [[0 for _ in unique_b] for _ in unique_a]
    for first, second in zip(labels_a, labels_b, strict=True):
        table[index_a[first]][index_b[second]] += 1
    return table


def _entropy_from_labels(labels: Sequence[str]) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in Counter(labels).values())


def _mutual_information(table: Sequence[Sequence[int]]) -> float:
    total = sum(sum(row) for row in table)
    if total == 0:
        return 0.0
    row_sums = tuple(sum(row) for row in table)
    column_sums = tuple(sum(row[column] for row in table) for column in range(len(table[0])))
    result = 0.0
    for row_index, row in enumerate(table):
        for column_index, value in enumerate(row):
            if value:
                result += (value / total) * math.log(
                    total * value / (row_sums[row_index] * column_sums[column_index])
                )
    return result


def _expected_mutual_information(table: Sequence[Sequence[int]]) -> float:
    total = sum(sum(row) for row in table)
    if total <= 1:
        return 0.0
    row_sums = tuple(sum(row) for row in table)
    column_sums = tuple(sum(row[column] for row in table) for column in range(len(table[0])))
    expectation = 0.0
    for row_total in row_sums:
        for column_total in column_sums:
            lower = max(1, row_total + column_total - total)
            upper = min(row_total, column_total)
            denominator = math.comb(total, column_total)
            for overlap in range(lower, upper + 1):
                probability = (
                    math.comb(row_total, overlap)
                    * math.comb(total - row_total, column_total - overlap)
                    / denominator
                )
                expectation += (
                    overlap
                    / total
                    * math.log(total * overlap / (row_total * column_total))
                    * probability
                )
    return expectation


def adjusted_mutual_information(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float | None:
    """Arithmetic-normalized adjusted MI with the permutation expectation."""

    if len(labels_a) != len(labels_b):
        raise ValueError("partition label vectors must have equal length")
    if not labels_a:
        return None
    table = _contingency(labels_a, labels_b)
    mutual_information = _mutual_information(table)
    expected = _expected_mutual_information(table)
    normalizer = (_entropy_from_labels(labels_a) + _entropy_from_labels(labels_b)) / 2
    denominator = normalizer - expected
    numerator = mutual_information - expected
    if abs(denominator) <= 1e-15:
        return 1.0 if abs(numerator) <= 1e-15 else 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def variation_of_information(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float | None:
    if len(labels_a) != len(labels_b):
        raise ValueError("partition label vectors must have equal length")
    if not labels_a:
        return None
    table = _contingency(labels_a, labels_b)
    value = (
        _entropy_from_labels(labels_a)
        + _entropy_from_labels(labels_b)
        - 2 * _mutual_information(table)
    )
    return max(0.0, value)


def score_partition(
    *,
    gold_assignments: Mapping[str, str],
    predicted_assignments: Mapping[str, str],
    anchors: Sequence[str],
) -> PartitionAgreement:
    if any(not node_id for node_id in gold_assignments):
        raise ValueError("gold community assignment node IDs must be nonempty")
    _reject_reserved_or_empty_labels(tuple(gold_assignments.values()))
    anchor_ids, predicted, omitted = _validate_assignments(predicted_assignments, anchors)
    missing_gold = tuple(anchor for anchor in anchor_ids if anchor not in gold_assignments)
    if missing_gold:
        raise ValueError(f"gold partition misses anchors: {', '.join(missing_gold)}")
    gold = tuple(gold_assignments[anchor] for anchor in anchor_ids)
    _reject_reserved_or_empty_labels(gold)

    predicted_clusters: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(predicted):
        predicted_clusters[label].append(index)
    purity_numerator = sum(
        max(Counter(gold[index] for index in members).values())
        for members in predicted_clusters.values()
    )

    fragmented = merged = gold_same = gold_different = 0
    for left in range(len(anchor_ids)):
        for right in range(left + 1, len(anchor_ids)):
            if gold[left] == gold[right]:
                gold_same += 1
                fragmented += predicted[left] != predicted[right]
            else:
                gold_different += 1
                merged += predicted[left] == predicted[right]
    return PartitionAgreement(
        anchor_count=len(anchor_ids),
        gold_cluster_count=len(set(gold)),
        predicted_cluster_count=len(set(predicted)),
        omitted_anchor_count=omitted,
        adjusted_mutual_information=adjusted_mutual_information(gold, predicted),
        purity=purity_numerator / len(anchor_ids) if anchor_ids else None,
        fragmentation_error=(fragmented / gold_same if gold_same else None),
        fragmentation_pair_denominator=gold_same,
        merging_error=(merged / gold_different if gold_different else None),
        merging_pair_denominator=gold_different,
    )


def conductance(
    nodes: Sequence[str],
    edges: Sequence[SemanticEdge],
    assignments: Mapping[str, str],
) -> ConductanceResult:
    if any(not node_id for node_id in nodes):
        raise ValueError("conductance node IDs must be nonempty")
    if len(nodes) != len(set(nodes)):
        raise ValueError("conductance node IDs must be unique")
    if set(assignments) != set(nodes):
        raise ValueError("conductance assignments must exactly cover graph nodes")
    _reject_reserved_or_empty_labels(tuple(assignments.values()))
    topology_edges = {
        tuple(sorted((edge.source_id, edge.target_id)))
        for edge in edges
        if edge.source_id != edge.target_id
    }
    if any(endpoint not in assignments for edge in topology_edges for endpoint in edge):
        raise ValueError("edge endpoint is absent from the assigned nodes")
    degrees = Counter(endpoint for edge in topology_edges for endpoint in edge)
    total_volume = sum(degrees.values())
    clusters: dict[str, set[str]] = defaultdict(set)
    for node, cluster in assignments.items():
        clusters[cluster].add(node)
    values = []
    by_cluster = []
    for cluster, members in sorted(clusters.items()):
        volume = sum(degrees[node] for node in members)
        denominator = min(volume, total_volume - volume)
        if denominator == 0:
            value = None
        else:
            boundary = sum(
                (left in members) != (right in members) for left, right in topology_edges
            )
            value = boundary / denominator
            values.append(value)
        by_cluster.append((cluster, value))
    return ConductanceResult(
        unweighted_mean=(sum(values) / len(values) if values else None),
        defined_cluster_count=len(values),
        undefined_cluster_count=len(clusters) - len(values),
        by_cluster=tuple(by_cluster),
    )


def run_leiden_cpm(
    nodes: Sequence[str],
    edges: Sequence[SemanticEdge],
    configuration: LeidenConfiguration,
) -> LeidenPartition:
    """Run only Leiden's Constant Potts Model on the simple unit-weight skeleton."""

    import igraph as ig
    import leidenalg

    node_ids = tuple(sorted(nodes))
    if any(not node_id for node_id in node_ids):
        raise ValueError("Leiden node IDs must be nonempty")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Leiden node IDs must be unique")
    node_index = {node: index for index, node in enumerate(node_ids)}
    if any(
        endpoint not in node_index
        for edge in edges
        for endpoint in (edge.source_id, edge.target_id)
    ):
        raise ValueError("edge endpoint is absent from the Leiden node universe")
    topology_edges = tuple(
        sorted(
            {
                tuple(sorted((node_index[edge.source_id], node_index[edge.target_id])))
                for edge in edges
                if edge.source_id != edge.target_id
            }
        )
    )
    graph = ig.Graph(n=len(node_ids), edges=topology_edges, directed=False)
    if node_ids:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.CPMVertexPartition,
            resolution_parameter=configuration.resolution,
            seed=configuration.seed,
            n_iterations=-1,
        )
        membership = tuple(int(value) for value in partition.membership)
    else:
        membership = ()
    community_members: dict[int, list[str]] = defaultdict(list)
    for index, node_id in enumerate(node_ids):
        community_members[membership[index]].append(node_id)
    canonical_community_order = sorted(
        community_members,
        key=lambda community_id: tuple(community_members[community_id]),
    )
    canonical_labels = {
        community_id: f"cluster-{index:03d}"
        for index, community_id in enumerate(canonical_community_order)
    }
    assignments = tuple(
        (node, canonical_labels[membership[index]]) for index, node in enumerate(node_ids)
    )
    modularity = None
    if topology_edges and membership:
        candidate = float(graph.modularity(membership))
        modularity = candidate if math.isfinite(candidate) else None
    return LeidenPartition(
        resolution=configuration.resolution,
        seed=configuration.seed,
        assignments=assignments,
        cluster_count=len(set(membership)),
        modularity=modularity,
    )


def run_leiden_sensitivity(
    nodes: Sequence[str],
    edges: Sequence[SemanticEdge],
    configuration: LeidenConfiguration,
) -> tuple[LeidenPartition, LeidenPartition, LeidenPartition]:
    return (
        run_leiden_cpm(
            nodes,
            edges,
            LeidenConfiguration(
                resolution=configuration.resolution * 0.5,
                seed=configuration.seed,
            ),
        ),
        run_leiden_cpm(nodes, edges, configuration),
        run_leiden_cpm(
            nodes,
            edges,
            LeidenConfiguration(
                resolution=configuration.resolution * 2.0,
                seed=configuration.seed,
            ),
        ),
    )


def cross_seed_stability(
    *,
    seed_a_assignments: Mapping[str, str],
    seed_b_assignments: Mapping[str, str],
    anchors: Sequence[str],
) -> CrossSeedStability:
    _, first, omitted_first = _validate_assignments(seed_a_assignments, anchors)
    _, second, omitted_second = _validate_assignments(seed_b_assignments, anchors)
    return CrossSeedStability(
        anchor_count=len(anchors),
        adjusted_mutual_information=adjusted_mutual_information(first, second),
        variation_of_information=variation_of_information(first, second),
        omitted_seed_a_count=omitted_first,
        omitted_seed_b_count=omitted_second,
    )


__all__ = [
    "ABSENT_ASSIGNMENT_PREFIX",
    "ConductanceResult",
    "CrossSeedStability",
    "LeidenConfiguration",
    "LeidenPartition",
    "PartitionAgreement",
    "adjusted_mutual_information",
    "conductance",
    "cross_seed_stability",
    "run_leiden_cpm",
    "run_leiden_sensitivity",
    "score_partition",
    "variation_of_information",
]
