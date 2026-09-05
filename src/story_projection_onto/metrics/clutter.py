"""Direct clutter and two-part edge-crossing measures for frozen layouts."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import Identifier, ImmutableRecord
from story_projection_onto.metrics.entropy import SemanticEdge

_NUMERIC_TOLERANCE = 1e-12


class Point(ImmutableRecord):
    x: float
    y: float

    @model_validator(mode="after")
    def coordinates_are_finite(self) -> Self:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("layout coordinates must be finite")
        return self


class LabelRectangle(ImmutableRecord):
    label_id: Identifier
    left: float
    top: float
    right: float
    bottom: float

    @model_validator(mode="after")
    def coordinates_form_a_finite_rectangle(self) -> Self:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("label rectangle coordinates must be finite")
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("label rectangle has negative extent")
        return self


class CrossingResult(ImmutableRecord):
    opportunity_indicator: int | None = Field(default=None, ge=0, le=1)
    opportunity_count: int | None = Field(default=None, ge=0)
    crossing_count: int | None = Field(default=None, ge=0)
    conditional_crossing_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def crossing_rate_matches_counts(self) -> Self:
        counts = (
            self.opportunity_indicator,
            self.opportunity_count,
            self.crossing_count,
        )
        if all(value is None for value in counts):
            if self.conditional_crossing_rate is not None:
                raise ValueError("unavailable crossing counts require an NA rate")
            return self
        if any(value is None for value in counts):
            raise ValueError("crossing counts must be all numeric or all NA")
        assert self.opportunity_indicator is not None
        assert self.opportunity_count is not None
        assert self.crossing_count is not None
        if self.opportunity_indicator != int(self.opportunity_count > 0):
            raise ValueError("opportunity indicator does not match opportunity count")
        if self.crossing_count > self.opportunity_count:
            raise ValueError("crossing count cannot exceed opportunity count")
        if self.opportunity_count == 0:
            if self.crossing_count != 0 or self.conditional_crossing_rate is not None:
                raise ValueError("zero opportunities require zero crossings and an NA rate")
        else:
            expected = self.crossing_count / self.opportunity_count
            if self.conditional_crossing_rate is None or not math.isclose(
                self.conditional_crossing_rate,
                expected,
                rel_tol=_NUMERIC_TOLERANCE,
                abs_tol=_NUMERIC_TOLERANCE,
            ):
                raise ValueError("conditional crossing rate does not match its counts")
        return self


class ClutterProfile(ImmutableRecord):
    valid_content_bearing: bool
    node_count: int = Field(ge=0)
    assertion_edge_count: int = Field(ge=0)
    topology_edge_count: int | None = Field(default=None, ge=0)
    density: float | None = Field(default=None, ge=0.0, le=1.0)
    isolate_count: int | None = Field(default=None, ge=0)
    component_count: int | None = Field(default=None, ge=0)
    crossings: CrossingResult
    label_overlap_count: int | None = Field(default=None, ge=0)
    label_overlap_area: float | None = Field(default=None, ge=0.0)
    visible_semantic_count: int | None = Field(default=None, ge=0)
    irrelevant_visible_count: int | None = Field(default=None, ge=0)
    irrelevant_visible_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    rare_pivotal_discoverability_denominator: int | None = Field(default=None, ge=0)
    rare_pivotal_discovered_count: int | None = Field(default=None, ge=0)
    rare_pivotal_omitted_count: int | None = Field(default=None, ge=0)
    rare_pivotal_discoverability_mean_interactions: float | None = Field(
        default=None,
        ge=0.0,
    )
    rare_pivotal_discoverability_max_interactions: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def derived_values_match_exposed_counts(self) -> Self:
        derived = (
            self.topology_edge_count,
            self.density,
            self.isolate_count,
            self.component_count,
            self.label_overlap_count,
            self.label_overlap_area,
            self.visible_semantic_count,
            self.irrelevant_visible_count,
            self.irrelevant_visible_fraction,
            self.rare_pivotal_discoverability_denominator,
            self.rare_pivotal_discovered_count,
            self.rare_pivotal_omitted_count,
            self.rare_pivotal_discoverability_mean_interactions,
            self.rare_pivotal_discoverability_max_interactions,
        )
        if not self.valid_content_bearing:
            if any(value is not None for value in derived):
                raise ValueError("invalid or empty outputs require NA clutter values")
            if self.crossings.opportunity_count is not None:
                raise ValueError("invalid or empty outputs require unavailable crossing geometry")
            return self

        if self.node_count == 0:
            raise ValueError("a content-bearing clutter profile requires at least one node")
        required = (
            self.topology_edge_count,
            self.density,
            self.isolate_count,
            self.component_count,
            self.label_overlap_count,
            self.label_overlap_area,
            self.visible_semantic_count,
            self.irrelevant_visible_count,
            self.rare_pivotal_discoverability_denominator,
            self.rare_pivotal_discovered_count,
            self.rare_pivotal_omitted_count,
        )
        if any(value is None for value in required):
            raise ValueError("content-bearing clutter profiles require all count fields")
        assert self.topology_edge_count is not None
        assert self.density is not None
        assert self.isolate_count is not None
        assert self.component_count is not None
        assert self.visible_semantic_count is not None
        assert self.irrelevant_visible_count is not None
        assert self.rare_pivotal_discoverability_denominator is not None
        assert self.rare_pivotal_discovered_count is not None
        assert self.rare_pivotal_omitted_count is not None

        maximum_edges = self.node_count * (self.node_count - 1) / 2
        if self.topology_edge_count > maximum_edges:
            raise ValueError("topology edge count exceeds a simple undirected skeleton")
        expected_density = self.topology_edge_count / maximum_edges if maximum_edges else 0.0
        if not math.isclose(
            self.density,
            expected_density,
            rel_tol=_NUMERIC_TOLERANCE,
            abs_tol=_NUMERIC_TOLERANCE,
        ):
            raise ValueError("density does not match node and topology-edge counts")
        if self.isolate_count > self.node_count:
            raise ValueError("isolate count cannot exceed node count")
        if not 1 <= self.component_count <= self.node_count:
            raise ValueError("component count must lie between one and node count")
        if self.crossings.opportunity_count is None:
            raise ValueError("content-bearing profiles require crossing geometry")
        if self.irrelevant_visible_count > self.visible_semantic_count:
            raise ValueError("irrelevant visible count cannot exceed visible count")
        if self.visible_semantic_count == 0:
            if self.irrelevant_visible_fraction is not None:
                raise ValueError("zero visible objects require an NA irrelevant fraction")
        else:
            expected_fraction = self.irrelevant_visible_count / self.visible_semantic_count
            if self.irrelevant_visible_fraction is None or not math.isclose(
                self.irrelevant_visible_fraction,
                expected_fraction,
                rel_tol=_NUMERIC_TOLERANCE,
                abs_tol=_NUMERIC_TOLERANCE,
            ):
                raise ValueError("irrelevant visible fraction does not match its counts")

        if (
            self.rare_pivotal_discovered_count + self.rare_pivotal_omitted_count
            != self.rare_pivotal_discoverability_denominator
        ):
            raise ValueError("rare-pivotal discovered and omitted counts must exhaust targets")
        if self.rare_pivotal_discovered_count == 0:
            if (
                self.rare_pivotal_discoverability_mean_interactions is not None
                or self.rare_pivotal_discoverability_max_interactions is not None
            ):
                raise ValueError("no discovered targets require NA interaction summaries")
        elif (
            self.rare_pivotal_discoverability_mean_interactions is None
            or self.rare_pivotal_discoverability_max_interactions is None
        ):
            raise ValueError("discovered rare-pivotal targets require interaction summaries")
        elif (
            self.rare_pivotal_discoverability_mean_interactions
            > self.rare_pivotal_discoverability_max_interactions
        ):
            raise ValueError("mean discoverability interactions cannot exceed the maximum")
        return self


def _properly_crosses(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(first: Point, second: Point, third: Point) -> float:
        return (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (
            third.x - first.x
        )

    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    tolerance = 1e-12
    if any(abs(value) <= tolerance for value in (first, second, third, fourth)):
        return False
    return (first > 0) != (second > 0) and (third > 0) != (fourth > 0)


def _simple_undirected_edges(edges: Sequence[SemanticEdge]) -> tuple[tuple[str, str], ...]:
    """Collapse self-loops, parallel assertions, and reverse-direction duplicates."""

    return tuple(
        sorted(
            {
                tuple(sorted((edge.source_id, edge.target_id)))
                for edge in edges
                if edge.source_id != edge.target_id
            }
        )
    )


def crossing_measure(
    edges: Sequence[SemanticEdge],
    positions: Mapping[str, Point],
) -> CrossingResult:
    """Count crossings over the registered simple undirected skeleton."""

    skeleton_edges = _simple_undirected_edges(edges)
    missing_positions = sorted(
        {endpoint for edge in skeleton_edges for endpoint in edge if endpoint not in positions}
    )
    if missing_positions:
        raise ValueError("every topology-edge endpoint requires a frozen layout position")
    eligible: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for index, first in enumerate(skeleton_edges):
        for second in skeleton_edges[index + 1 :]:
            if set(first) & set(second):
                continue
            eligible.append((first, second))
    opportunities = len(eligible)
    crossings = sum(
        _properly_crosses(
            positions[first[0]],
            positions[first[1]],
            positions[second[0]],
            positions[second[1]],
        )
        for first, second in eligible
    )
    return CrossingResult(
        opportunity_indicator=int(opportunities > 0),
        opportunity_count=opportunities,
        crossing_count=crossings,
        conditional_crossing_rate=(crossings / opportunities if opportunities else None),
    )


def label_overlap(rectangles: Sequence[LabelRectangle]) -> tuple[int, float]:
    label_ids = tuple(rectangle.label_id for rectangle in rectangles)
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("label rectangle IDs must be unique")
    count = 0
    area = 0.0
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            width = min(first.right, second.right) - max(first.left, second.left)
            height = min(first.bottom, second.bottom) - max(first.top, second.top)
            if width > 0 and height > 0:
                count += 1
                area += width * height
    return count, area


def _topology(
    nodes: Sequence[str],
    edges: Sequence[SemanticEdge],
) -> tuple[set[tuple[str, str]], Mapping[str, set[str]]]:
    node_set = set(nodes)
    if any(not node for node in nodes):
        raise ValueError("node IDs must be nonempty")
    if len(node_set) != len(nodes):
        raise ValueError("node IDs must be unique")
    if len({edge.edge_id for edge in edges}) != len(edges):
        raise ValueError("assertion edge IDs must be unique")
    adjacency = {node: set() for node in nodes}
    topology_edges = set(_simple_undirected_edges(edges))
    if any(endpoint not in node_set for edge in topology_edges for endpoint in edge):
        raise ValueError("edge endpoint is absent from the graph")
    if any(
        endpoint not in node_set for edge in edges for endpoint in (edge.source_id, edge.target_id)
    ):
        raise ValueError("edge endpoint is absent from the graph")
    for left, right in topology_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return topology_edges, adjacency


def _component_count(adjacency: Mapping[str, set[str]]) -> int:
    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        queue = deque((remaining.pop(),))
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node] & remaining:
                remaining.remove(neighbor)
                queue.append(neighbor)
    return count


def clutter_profile(
    nodes: Sequence[str],
    edges: Sequence[SemanticEdge],
    *,
    positions: Mapping[str, Point],
    label_rectangles: Sequence[LabelRectangle],
    label_semantic_ids: Mapping[str, str] | None = None,
    visible_semantic_ids: Sequence[str],
    irrelevant_semantic_ids: frozenset[str],
    rare_pivotal_discoverability: Mapping[str, int],
    rare_pivotal_target_ids: Sequence[str],
    structurally_valid: bool = True,
    assertion_record_count: int | None = None,
    assertion_semantic_ids: Sequence[str] = (),
) -> ClutterProfile:
    """Compute clutter while making invalid/empty geometry explicitly unavailable."""

    assertion_count = len(edges) if assertion_record_count is None else assertion_record_count
    if assertion_count < 0:
        raise ValueError("qualified assertion record count must be nonnegative")
    content_bearing = structurally_valid and bool(nodes)
    if not content_bearing:
        return ClutterProfile(
            valid_content_bearing=False,
            node_count=len(nodes),
            assertion_edge_count=assertion_count,
            topology_edge_count=None,
            density=None,
            isolate_count=None,
            component_count=None,
            crossings=CrossingResult(),
            label_overlap_count=None,
            label_overlap_area=None,
            visible_semantic_count=None,
            irrelevant_visible_count=None,
            irrelevant_visible_fraction=None,
            rare_pivotal_discoverability_denominator=None,
            rare_pivotal_discovered_count=None,
            rare_pivotal_omitted_count=None,
            rare_pivotal_discoverability_mean_interactions=None,
            rare_pivotal_discoverability_max_interactions=None,
        )

    topology_edges, adjacency = _topology(nodes, edges)
    if set(positions) != set(nodes):
        raise ValueError("frozen layout positions must exactly cover graph nodes")
    overlap_count, overlap_area = label_overlap(label_rectangles)
    visible = tuple(visible_semantic_ids)
    if len(visible) != len(set(visible)):
        raise ValueError("visible semantic IDs must be unique")
    known_semantic_ids = (
        set(nodes) | {edge.edge_id for edge in edges} | set(assertion_semantic_ids)
    )
    rectangle_ids = [rectangle.label_id for rectangle in label_rectangles]
    if label_semantic_ids is None:
        resolved_label_ids = set(rectangle_ids)
    else:
        if set(label_semantic_ids) != set(rectangle_ids):
            raise ValueError("label semantic bindings must exactly cover label rectangles")
        resolved_label_ids = set(label_semantic_ids.values())
    unknown_label_ids = sorted(resolved_label_ids - known_semantic_ids)
    if unknown_label_ids:
        raise ValueError("label rectangles must resolve to a graph node or assertion")
    unknown_visible_ids = sorted(set(visible) - known_semantic_ids)
    if unknown_visible_ids:
        raise ValueError("visible semantic IDs must reference a graph node or assertion")
    irrelevant_count = sum(item in irrelevant_semantic_ids for item in visible)

    target_ids = tuple(rare_pivotal_target_ids)
    if any(not target_id for target_id in target_ids):
        raise ValueError("rare-pivotal discoverability target IDs must be nonempty")
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("rare-pivotal discoverability target IDs must be unique")
    unexpected_discoverability = sorted(set(rare_pivotal_discoverability) - set(target_ids))
    if unexpected_discoverability:
        raise ValueError("discoverability observations must reference registered targets")
    interaction_values = tuple(rare_pivotal_discoverability.values())
    if any(value < 0 for value in interaction_values):
        raise ValueError("discoverability interactions must be nonnegative")

    node_count = len(nodes)
    maximum_edges = node_count * (node_count - 1) / 2
    discovered_count = len(interaction_values)
    target_count = len(target_ids)
    return ClutterProfile(
        valid_content_bearing=True,
        node_count=node_count,
        assertion_edge_count=assertion_count,
        topology_edge_count=len(topology_edges),
        density=(len(topology_edges) / maximum_edges if maximum_edges else 0.0),
        isolate_count=sum(not neighbors for neighbors in adjacency.values()),
        component_count=_component_count(adjacency),
        crossings=crossing_measure(edges, positions),
        label_overlap_count=overlap_count,
        label_overlap_area=overlap_area,
        visible_semantic_count=len(visible),
        irrelevant_visible_count=irrelevant_count,
        irrelevant_visible_fraction=(irrelevant_count / len(visible) if visible else None),
        rare_pivotal_discoverability_denominator=target_count,
        rare_pivotal_discovered_count=discovered_count,
        rare_pivotal_omitted_count=target_count - discovered_count,
        rare_pivotal_discoverability_mean_interactions=(
            sum(interaction_values) / discovered_count if interaction_values else None
        ),
        rare_pivotal_discoverability_max_interactions=(
            max(interaction_values) if interaction_values else None
        ),
    )


__all__ = [
    "ClutterProfile",
    "CrossingResult",
    "LabelRectangle",
    "Point",
    "clutter_profile",
    "crossing_measure",
    "label_overlap",
]
