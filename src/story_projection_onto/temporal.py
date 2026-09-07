"""Deterministic temporal and horizon validation.

This module checks explicit claims and reports typed contradictions or
underdetermination.  It never fills missing dates, orders events from prose, or
turns temporal precedence into causation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from enum import StrEnum

from pydantic import Field

from story_projection_onto.contracts import (
    AllenRelation,
    DiscoursePosition,
    HolderRelativeTime,
    ImmutableRecord,
    PartialOrderConstraint,
    QualifiedAssertion,
    RevelationPosition,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
    TemporalScope,
    ValidityTime,
)


class TemporalValidationStatus(StrEnum):
    VALID = "valid"
    UNDERDETERMINED = "underdetermined"
    CONTRADICTION = "contradiction"


def query_time_visibility(
    story_time: StoryTime, validity_time: ValidityTime, query_time: StoryTime
) -> bool | None:
    """Common, non-mutating query restriction for every condition.

    Explicit intrinsic validity can extend an observation's visibility. Unknown
    validity cannot establish exclusion outside its observed story occurrence;
    it remains undetermined, not asserted persistent. None means undetermined.
    This is selection/display logic, not ontology construction or qualification.
    """
    extent = (
        validity_time
        if validity_time.kind in {TemporalKind.POINT, TemporalKind.INTERVAL}
        else story_time
    )

    def bounds(value: StoryTime | ValidityTime) -> tuple[float, float] | None:
        if value.kind is TemporalKind.POINT:
            assert value.point is not None
            return value.point, value.point
        if value.kind is TemporalKind.INTERVAL:
            return (
                float("-inf") if value.start is None else value.start,
                float("inf") if value.end is None else value.end,
            )
        return None

    source, query = bounds(extent), bounds(query_time)
    if source is None or query is None:
        return None
    overlaps = max(source[0], query[0]) <= min(source[1], query[1])
    if not overlaps and validity_time.kind is TemporalKind.UNKNOWN:
        return None
    return overlaps


class TemporalDiagnosticCode(StrEnum):
    INVALID_INTERVAL_BOUNDS = "invalid_interval_bounds"
    INVALID_EXPLICIT_TIME = "invalid_explicit_time"
    UNRESOLVED_RELATIVE_ANCHOR = "unresolved_relative_anchor"
    UNKNOWN_TIME = "unknown_time"
    HORIZON_WITHHELD = "horizon_withheld"
    PARTIAL_ORDER_UNKNOWN_NODE = "partial_order_unknown_node"
    PARTIAL_ORDER_SELF_PRECEDENCE = "partial_order_self_precedence"
    PARTIAL_ORDER_CYCLE = "partial_order_cycle"
    EVIDENCE_AFTER_SPOILER_HORIZON = "evidence_after_spoiler_horizon"
    REVELATION_AFTER_SPOILER_HORIZON = "revelation_after_spoiler_horizon"


class TemporalDiagnostic(ImmutableRecord):
    code: TemporalDiagnosticCode
    field_path: str
    message: str = Field(min_length=1)
    involved_ids: tuple[str, ...] = ()


class TemporalValidationResult(ImmutableRecord):
    status: TemporalValidationStatus
    diagnostics: tuple[TemporalDiagnostic, ...] = ()


_PRECEDENCE_RELATIONS = {
    AllenRelation.BEFORE,
    AllenRelation.MEETS,
}
_REVERSE_PRECEDENCE_RELATIONS = {
    AllenRelation.AFTER,
    AllenRelation.MET_BY,
}


def _result(diagnostics: Iterable[TemporalDiagnostic]) -> TemporalValidationResult:
    frozen = tuple(diagnostics)
    contradiction_codes = {
        TemporalDiagnosticCode.INVALID_INTERVAL_BOUNDS,
        TemporalDiagnosticCode.INVALID_EXPLICIT_TIME,
        TemporalDiagnosticCode.PARTIAL_ORDER_UNKNOWN_NODE,
        TemporalDiagnosticCode.PARTIAL_ORDER_SELF_PRECEDENCE,
        TemporalDiagnosticCode.PARTIAL_ORDER_CYCLE,
        TemporalDiagnosticCode.EVIDENCE_AFTER_SPOILER_HORIZON,
        TemporalDiagnosticCode.REVELATION_AFTER_SPOILER_HORIZON,
    }
    if any(item.code in contradiction_codes for item in frozen):
        status = TemporalValidationStatus.CONTRADICTION
    elif frozen:
        status = TemporalValidationStatus.UNDERDETERMINED
    else:
        status = TemporalValidationStatus.VALID
    return TemporalValidationResult(status=status, diagnostics=frozen)


def combine_temporal_results(
    *results: TemporalValidationResult,
) -> TemporalValidationResult:
    """Combine results while retaining all diagnostics in call order."""

    return _result(item for result in results for item in result.diagnostics)


def validate_partial_order(
    constraints: Iterable[PartialOrderConstraint],
    *,
    known_node_ids: Iterable[str] | None = None,
    field_path: str = "partial_order",
) -> TemporalValidationResult:
    """Validate the strict-precedence subset of declared Allen constraints."""

    frozen_constraints = tuple(constraints)
    known = set(known_node_ids) if known_node_ids is not None else None
    diagnostics: list[TemporalDiagnostic] = []
    graph: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()

    for constraint in frozen_constraints:
        left = constraint.left_id
        right = constraint.right_id
        all_nodes.update((left, right))
        if known is not None:
            missing = tuple(sorted({left, right} - known))
            if missing:
                diagnostics.append(
                    TemporalDiagnostic(
                        code=TemporalDiagnosticCode.PARTIAL_ORDER_UNKNOWN_NODE,
                        field_path=field_path,
                        message="partial-order constraint refers to an unknown node",
                        involved_ids=missing,
                    )
                )
                continue

        edge: tuple[str, str] | None = None
        if constraint.relation in _PRECEDENCE_RELATIONS:
            edge = (left, right)
        elif constraint.relation in _REVERSE_PRECEDENCE_RELATIONS:
            edge = (right, left)

        if edge is not None:
            source, target = edge
            if source == target:
                diagnostics.append(
                    TemporalDiagnostic(
                        code=TemporalDiagnosticCode.PARTIAL_ORDER_SELF_PRECEDENCE,
                        field_path=field_path,
                        message="an object cannot strictly precede itself",
                        involved_ids=(source,),
                    )
                )
            else:
                graph[source].add(target)

    indegree = dict.fromkeys(all_nodes, 0)
    for source, targets in graph.items():
        indegree.setdefault(source, 0)
        for target in targets:
            indegree[target] = indegree.get(target, 0) + 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited: list[str] = []
    while queue:
        node = queue.popleft()
        visited.append(node)
        for target in sorted(graph.get(node, ())):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(visited) != len(indegree):
        cyclic_nodes = tuple(sorted(node for node, degree in indegree.items() if degree > 0))
        diagnostics.append(
            TemporalDiagnostic(
                code=TemporalDiagnosticCode.PARTIAL_ORDER_CYCLE,
                field_path=field_path,
                message="strict temporal precedence contains a cycle",
                involved_ids=cyclic_nodes,
            )
        )

    return _result(diagnostics)


def validate_temporal_extent(
    extent: StoryTime | ValidityTime | HolderRelativeTime,
    *,
    known_anchor_ids: Iterable[str] | None = None,
    field_path: str = "time",
) -> TemporalValidationResult:
    """Validate one explicit extent without inferring unknown information."""

    diagnostics: list[TemporalDiagnostic] = []
    if extent.kind is TemporalKind.INTERVAL:
        if extent.start is not None and extent.end is not None and extent.start > extent.end:
            diagnostics.append(
                TemporalDiagnostic(
                    code=TemporalDiagnosticCode.INVALID_INTERVAL_BOUNDS,
                    field_path=field_path,
                    message="interval start is after interval end",
                )
            )
    elif extent.kind is TemporalKind.RELATIVE:
        if known_anchor_ids is None or extent.anchor_id not in set(known_anchor_ids):
            diagnostics.append(
                TemporalDiagnostic(
                    code=TemporalDiagnosticCode.UNRESOLVED_RELATIVE_ANCHOR,
                    field_path=field_path,
                    message="relative temporal anchor is not resolved by this validation context",
                    involved_ids=(extent.anchor_id,) if extent.anchor_id else (),
                )
            )
    elif extent.kind is TemporalKind.PARTIAL_ORDER:
        partial_result = validate_partial_order(
            extent.partial_order,
            known_node_ids=known_anchor_ids,
            field_path=f"{field_path}.partial_order",
        )
        diagnostics.extend(partial_result.diagnostics)
    elif extent.kind is TemporalKind.UNKNOWN:
        diagnostics.append(
            TemporalDiagnostic(
                code=TemporalDiagnosticCode.UNKNOWN_TIME,
                field_path=field_path,
                message=extent.reason or "time is explicitly unknown",
            )
        )
    elif extent.kind is TemporalKind.HORIZON_WITHHELD:
        diagnostics.append(
            TemporalDiagnostic(
                code=TemporalDiagnosticCode.HORIZON_WITHHELD,
                field_path=field_path,
                message=extent.reason or "time is withheld by the registered horizon",
            )
        )
    elif extent.kind is TemporalKind.INVALID:
        diagnostics.append(
            TemporalDiagnostic(
                code=TemporalDiagnosticCode.INVALID_EXPLICIT_TIME,
                field_path=field_path,
                message=extent.reason or "time was marked invalid",
            )
        )
    return _result(diagnostics)


def discourse_is_within_horizon(
    position: DiscoursePosition,
    horizon: SpoilerHorizon,
) -> bool:
    """Compare discourse coordinates only; story time is intentionally ignored."""

    return position.ordering_key <= horizon.max_discourse_position.ordering_key


def revelation_is_within_horizon(
    position: RevelationPosition,
    horizon: SpoilerHorizon,
) -> bool:
    """Compare proposition revelation only when the horizon declares that coordinate."""

    maximum = horizon.max_revelation_position
    return maximum is None or position.revelation_order <= maximum.revelation_order


def validate_horizon(
    *,
    discourse_position: DiscoursePosition,
    revelation_position: RevelationPosition | None,
    horizon: SpoilerHorizon,
    field_path: str = "temporal_scope",
) -> TemporalValidationResult:
    diagnostics: list[TemporalDiagnostic] = []
    if not discourse_is_within_horizon(discourse_position, horizon):
        diagnostics.append(
            TemporalDiagnostic(
                code=TemporalDiagnosticCode.EVIDENCE_AFTER_SPOILER_HORIZON,
                field_path=f"{field_path}.discourse_position",
                message="discourse evidence occurs after the registered spoiler horizon",
            )
        )
    if revelation_position is not None and not revelation_is_within_horizon(
        revelation_position, horizon
    ):
        diagnostics.append(
            TemporalDiagnostic(
                code=TemporalDiagnosticCode.REVELATION_AFTER_SPOILER_HORIZON,
                field_path=f"{field_path}.revelation_position",
                message="proposition revelation occurs after the registered spoiler horizon",
            )
        )
    return _result(diagnostics)


def validate_evidence_horizon(
    discourse_position: DiscoursePosition,
    *,
    horizon: SpoilerHorizon,
    evidence_id: str | None = None,
) -> TemporalValidationResult:
    """Validate query-blind evidence admission against discourse position only."""

    path = f"evidence.{evidence_id}" if evidence_id is not None else "evidence"
    return validate_horizon(
        discourse_position=discourse_position,
        revelation_position=None,
        horizon=horizon,
        field_path=path,
    )


def validate_temporal_scope(
    scope: TemporalScope,
    *,
    horizon: SpoilerHorizon | None = None,
    known_anchor_ids: Iterable[str] | None = None,
    field_path: str = "temporal_scope",
) -> TemporalValidationResult:
    frozen_anchor_ids = tuple(known_anchor_ids) if known_anchor_ids is not None else None
    results = [
        validate_temporal_extent(
            scope.story_time,
            known_anchor_ids=frozen_anchor_ids,
            field_path=f"{field_path}.story_time",
        ),
        validate_temporal_extent(
            scope.validity_time,
            known_anchor_ids=frozen_anchor_ids,
            field_path=f"{field_path}.validity_time",
        ),
    ]
    if horizon is not None:
        results.append(
            validate_horizon(
                discourse_position=scope.discourse_position,
                revelation_position=scope.revelation_position,
                horizon=horizon,
                field_path=field_path,
            )
        )
    return combine_temporal_results(*results)


def validate_assertion_temporality(
    assertion: QualifiedAssertion,
    *,
    horizon: SpoilerHorizon,
    known_anchor_ids: Iterable[str] | None = None,
) -> TemporalValidationResult:
    """Validate assertion time and optional holder-relative time independently."""

    frozen_anchor_ids = tuple(known_anchor_ids) if known_anchor_ids is not None else None
    results = [
        validate_temporal_scope(
            assertion.temporal_scope,
            horizon=horizon,
            known_anchor_ids=frozen_anchor_ids,
            field_path=f"assertions.{assertion.assertion_id}.temporal_scope",
        )
    ]
    if assertion.epistemic_scope is not None:
        results.append(
            validate_temporal_extent(
                assertion.epistemic_scope.holder_relative_time,
                known_anchor_ids=frozen_anchor_ids,
                field_path=(
                    f"assertions.{assertion.assertion_id}.epistemic_scope.holder_relative_time"
                ),
            )
        )
    return combine_temporal_results(*results)
