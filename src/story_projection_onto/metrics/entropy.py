"""Registered entropy panel over the named entity-event skeleton."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import Identifier, ImmutableRecord

OTHER_RELATION_BIN = "OTHER"
_NUMERIC_TOLERANCE = 1e-12


class SemanticEdge(ImmutableRecord):
    """One unit-weight incidence edge in the entity/event topology."""

    edge_id: Identifier
    source_id: Identifier
    target_id: Identifier
    native_predicate_id: Identifier
    canonical_predicate_id: Identifier | None


class AssertionRelationRecord(ImmutableRecord):
    """One qualified assertion for global relation-frequency measures.

    This is deliberately separate from :class:`SemanticEdge`: one n-ary assertion
    can induce several topology incidences but contributes exactly one observation
    to native/canonical relation entropy and the assertion-count panel.
    """

    assertion_id: Identifier
    native_predicate_id: Identifier
    canonical_predicate_id: Identifier | None


class EntropyMeasure(ImmutableRecord):
    """One Shannon entropy with both sampling and normalization denominators."""

    raw: float | None = Field(default=None, ge=0.0)
    normalized: float | None = Field(default=None, ge=0.0, le=1.0)
    support_size: int = Field(ge=0)
    denominator: int | None = Field(default=None, gt=0)
    normalization_size: int = Field(ge=0)
    normalization_log_denominator: float | None = Field(default=None, ge=0.0)
    defined: bool

    @model_validator(mode="after")
    def values_match_declared_denominators(self) -> Self:
        values = (
            self.raw,
            self.normalized,
            self.denominator,
            self.normalization_log_denominator,
        )
        if not self.defined:
            if self.support_size != 0 or self.normalization_size != 0:
                raise ValueError("undefined entropy must have zero support and normalization size")
            if any(value is not None for value in values):
                raise ValueError("undefined entropy values and denominators must be NA")
            return self

        if self.support_size <= 0 or self.normalization_size <= 0:
            raise ValueError("defined entropy requires positive support and normalization size")
        if self.raw is None or self.normalized is None or self.denominator is None:
            raise ValueError("defined entropy requires raw, normalized, and sampling denominator")
        expected_log = math.log(self.normalization_size)
        if self.normalization_log_denominator is None or not math.isclose(
            self.normalization_log_denominator,
            expected_log,
            rel_tol=_NUMERIC_TOLERANCE,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError("entropy normalization denominator does not match normalization size")
        expected_normalized = 0.0 if expected_log == 0.0 else self.raw / expected_log
        if not math.isclose(
            self.normalized,
            expected_normalized,
            rel_tol=_NUMERIC_TOLERANCE,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError("normalized entropy does not match the declared normalizer")
        return self


class EntropyProfile(ImmutableRecord):
    node_count: int = Field(ge=0)
    topology_edge_count: int = Field(ge=0)
    assertion_edge_count: int = Field(ge=0)
    isolate_count: int = Field(ge=0)
    positive_degree_node_count: int = Field(ge=0)
    occupied_degree_bin_count: int = Field(ge=0)
    active_native_relation_count: int = Field(ge=0)
    canonical_vocabulary_size: int = Field(gt=0)
    canonical_other_count: int = Field(ge=0)
    canonical_other_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    degree_histogram: EntropyMeasure
    degree_mass: EntropyMeasure
    mean_local_relation_neighborhood: EntropyMeasure
    native_schema_relation: EntropyMeasure
    canonical_mapped_relation: EntropyMeasure

    @model_validator(mode="after")
    def counts_and_measure_denominators_agree(self) -> Self:
        maximum_topology_edges = self.node_count * (self.node_count - 1) // 2
        if self.topology_edge_count > maximum_topology_edges:
            raise ValueError("topology edge count exceeds a simple undirected skeleton")
        if self.isolate_count + self.positive_degree_node_count != self.node_count:
            raise ValueError("isolate and positive-degree counts must partition the nodes")
        if self.occupied_degree_bin_count > self.node_count:
            raise ValueError("occupied degree bins cannot exceed node count")
        if self.active_native_relation_count > self.assertion_edge_count:
            raise ValueError("active native relation count cannot exceed assertion count")
        if self.canonical_other_count > self.assertion_edge_count:
            raise ValueError("OTHER count cannot exceed assertion count")

        if self.assertion_edge_count == 0:
            if self.canonical_other_rate is not None:
                raise ValueError("empty assertion sets require an NA OTHER rate")
        else:
            expected_other_rate = self.canonical_other_count / self.assertion_edge_count
            if self.canonical_other_rate is None or not math.isclose(
                self.canonical_other_rate,
                expected_other_rate,
                rel_tol=_NUMERIC_TOLERANCE,
                abs_tol=_NUMERIC_TOLERANCE,
            ):
                raise ValueError("OTHER rate does not match its exposed counts")

        if self.node_count == 0:
            if self.occupied_degree_bin_count != 0 or self.degree_histogram.defined:
                raise ValueError("an empty graph has no degree-histogram entropy")
        elif (
            not self.degree_histogram.defined
            or self.degree_histogram.denominator != self.node_count
            or self.degree_histogram.support_size != self.occupied_degree_bin_count
            or self.degree_histogram.normalization_size != self.occupied_degree_bin_count
        ):
            raise ValueError("degree-histogram denominators do not match the profile")

        has_positive_degree = self.positive_degree_node_count > 0
        for name, measure in (
            ("degree-mass", self.degree_mass),
            ("local relation-neighborhood", self.mean_local_relation_neighborhood),
        ):
            if measure.defined != has_positive_degree:
                raise ValueError(f"{name} definedness must match positive-degree support")
        if has_positive_degree:
            if (
                self.degree_mass.support_size != self.positive_degree_node_count
                or self.degree_mass.denominator != 2 * self.topology_edge_count
                or self.degree_mass.normalization_size != self.positive_degree_node_count
            ):
                raise ValueError(
                    "degree-mass sampling denominator must be twice the topology-edge "
                    "count and its normalizer must be the positive-degree node count"
                )
            if (
                self.mean_local_relation_neighborhood.support_size
                != self.positive_degree_node_count
                or self.mean_local_relation_neighborhood.denominator
                != self.positive_degree_node_count
                or self.mean_local_relation_neighborhood.normalization_size
                != self.canonical_vocabulary_size
            ):
                raise ValueError("local relation entropy denominators do not match the profile")

        has_assertions = self.assertion_edge_count > 0
        if self.native_schema_relation.defined != has_assertions:
            raise ValueError("native relation entropy definedness must match assertion presence")
        if self.canonical_mapped_relation.defined != has_assertions:
            raise ValueError("canonical relation entropy definedness must match assertion presence")
        if has_assertions:
            if (
                self.native_schema_relation.support_size != self.active_native_relation_count
                or self.native_schema_relation.denominator != self.assertion_edge_count
                or self.native_schema_relation.normalization_size
                != self.active_native_relation_count
            ):
                raise ValueError(
                    "native entropy denominator must be assertion count and its normalizer "
                    "must be the active native vocabulary"
                )
            if (
                self.canonical_mapped_relation.support_size > self.canonical_vocabulary_size
                or self.canonical_mapped_relation.denominator != self.assertion_edge_count
                or self.canonical_mapped_relation.normalization_size
                != self.canonical_vocabulary_size
            ):
                raise ValueError(
                    "canonical entropy denominator must be assertion count and its "
                    "normalizer must be the frozen vocabulary"
                )
        return self


def _entropy(counts: Iterable[int]) -> tuple[float | None, int, int]:
    positive = tuple(sorted(count for count in counts if count > 0))
    total = sum(positive)
    if total == 0:
        return None, 0, 0
    raw = -sum((count / total) * math.log(count / total) for count in positive)
    if abs(raw) <= _NUMERIC_TOLERANCE:
        raw = 0.0
    return raw, len(positive), total


def _undefined_measure() -> EntropyMeasure:
    return EntropyMeasure(
        raw=None,
        normalized=None,
        support_size=0,
        denominator=None,
        normalization_size=0,
        normalization_log_denominator=None,
        defined=False,
    )


def _measure(
    counts: Iterable[int],
    *,
    normalization_size: int,
) -> EntropyMeasure:
    raw, support_size, denominator = _entropy(counts)
    if raw is None:
        return _undefined_measure()
    if normalization_size < support_size:
        raise ValueError("normalization universe cannot be smaller than observed entropy support")
    log_denominator = math.log(normalization_size)
    normalized = 0.0 if log_denominator == 0.0 else raw / log_denominator
    normalized = min(1.0, max(0.0, normalized))
    return EntropyMeasure(
        raw=raw,
        normalized=normalized,
        support_size=support_size,
        denominator=denominator,
        normalization_size=normalization_size,
        normalization_log_denominator=log_denominator,
        defined=True,
    )


def _mean_local_measure(
    local_values: Sequence[float],
    *,
    canonical_vocabulary_size: int,
) -> EntropyMeasure:
    if not local_values:
        return _undefined_measure()
    raw = sum(local_values) / len(local_values)
    if abs(raw) <= _NUMERIC_TOLERANCE:
        raw = 0.0
    log_denominator = math.log(canonical_vocabulary_size)
    normalized = 0.0 if log_denominator == 0.0 else raw / log_denominator
    normalized = min(1.0, max(0.0, normalized))
    return EntropyMeasure(
        raw=raw,
        normalized=normalized,
        support_size=len(local_values),
        denominator=len(local_values),
        normalization_size=canonical_vocabulary_size,
        normalization_log_denominator=log_denominator,
        defined=True,
    )


def _validate_graph(nodes: Sequence[str], edges: Sequence[SemanticEdge]) -> tuple[str, ...]:
    node_ids = tuple(nodes)
    if any(not node_id for node_id in node_ids):
        raise ValueError("skeleton node IDs must be nonempty")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("skeleton node IDs must be unique")
    if len({edge.edge_id for edge in edges}) != len(edges):
        raise ValueError("assertion edge IDs must be unique")
    known = set(node_ids)
    unknown = sorted(
        {
            endpoint
            for edge in edges
            for endpoint in (edge.source_id, edge.target_id)
            if endpoint not in known
        }
    )
    if unknown:
        raise ValueError(f"edge endpoints are absent from the skeleton: {', '.join(unknown)}")
    return tuple(sorted(node_ids))


def entropy_profile(
    nodes: Sequence[str],
    edges: Sequence[SemanticEdge],
    *,
    canonical_vocabulary: Sequence[str],
    assertion_relations: Sequence[AssertionRelationRecord] | None = None,
) -> EntropyProfile:
    """Compute every registered entropy without treating a direction as preferable."""

    node_ids = _validate_graph(nodes, edges)
    vocabulary = tuple(canonical_vocabulary)
    if any(not relation for relation in vocabulary):
        raise ValueError("canonical vocabulary entries must be nonempty")
    if len(vocabulary) != len(set(vocabulary)) or OTHER_RELATION_BIN not in vocabulary:
        raise ValueError("canonical vocabulary must be unique and include OTHER")

    topology_edges = {
        tuple(sorted((edge.source_id, edge.target_id)))
        for edge in edges
        if edge.source_id != edge.target_id
    }
    neighbors: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for left, right in topology_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    degrees = {node_id: len(neighbors[node_id]) for node_id in node_ids}
    degree_counts = Counter(degrees.values())
    degree_histogram = _measure(
        degree_counts.values(),
        normalization_size=len(degree_counts),
    )
    positive_degrees = tuple(degree for degree in degrees.values() if degree > 0)
    degree_mass = _measure(
        positive_degrees,
        normalization_size=len(positive_degrees),
    )

    relation_records = (
        tuple(assertion_relations)
        if assertion_relations is not None
        else tuple(
            AssertionRelationRecord(
                assertion_id=edge.edge_id,
                native_predicate_id=edge.native_predicate_id,
                canonical_predicate_id=edge.canonical_predicate_id,
            )
            for edge in edges
        )
    )
    relation_ids = tuple(item.assertion_id for item in relation_records)
    if len(relation_ids) != len(set(relation_ids)):
        raise ValueError("qualified assertion relation records must have unique IDs")
    canonical_counts: Counter[str] = Counter()
    native_counts = Counter(item.native_predicate_id for item in relation_records)
    incident: dict[str, Counter[str]] = defaultdict(Counter)
    vocabulary_set = set(vocabulary)
    for item in relation_records:
        canonical = (
            item.canonical_predicate_id
            if item.canonical_predicate_id in vocabulary_set
            else OTHER_RELATION_BIN
        )
        canonical_counts[canonical] += 1
    for edge in edges:
        canonical = (
            edge.canonical_predicate_id
            if edge.canonical_predicate_id in vocabulary_set
            else OTHER_RELATION_BIN
        )
        for endpoint in {edge.source_id, edge.target_id}:
            incident[endpoint][canonical] += 1

    local_values: list[float] = []
    for node_id in node_ids:
        if degrees[node_id] == 0:
            continue
        raw, _, _ = _entropy(incident[node_id].values())
        if raw is None:
            raise ValueError("a positive-degree node must have an incident relation assertion")
        local_values.append(raw)
    local_measure = _mean_local_measure(
        local_values,
        canonical_vocabulary_size=len(vocabulary),
    )
    native_measure = _measure(
        native_counts.values(),
        normalization_size=len(native_counts),
    )
    canonical_measure = _measure(
        (canonical_counts[relation] for relation in vocabulary),
        normalization_size=len(vocabulary),
    )
    assertion_count = len(relation_records)
    other_count = canonical_counts[OTHER_RELATION_BIN]
    return EntropyProfile(
        node_count=len(node_ids),
        topology_edge_count=len(topology_edges),
        assertion_edge_count=assertion_count,
        isolate_count=sum(degree == 0 for degree in degrees.values()),
        positive_degree_node_count=len(positive_degrees),
        occupied_degree_bin_count=len(degree_counts),
        active_native_relation_count=len(native_counts),
        canonical_vocabulary_size=len(vocabulary),
        canonical_other_count=other_count,
        canonical_other_rate=(other_count / assertion_count if assertion_count else None),
        degree_histogram=degree_histogram,
        degree_mass=degree_mass,
        mean_local_relation_neighborhood=local_measure,
        native_schema_relation=native_measure,
        canonical_mapped_relation=canonical_measure,
    )


__all__ = [
    "OTHER_RELATION_BIN",
    "AssertionRelationRecord",
    "EntropyMeasure",
    "EntropyProfile",
    "SemanticEdge",
    "entropy_profile",
]
