"""Condition-neutral selection for the registered display-budget view.

This module contains no renderer, scorer, gold, or condition-specific logic.  Both
structural validation and visualization compilation call the same function so an
output that cannot satisfy the registered display budget is repairable at the model
boundary rather than failing for the first time during post-hoc geometry capture.
"""

from __future__ import annotations

from dataclasses import dataclass

from story_projection_onto.contracts import (
    BudgetAccounting,
    InstanceGraph,
    OutputBudgets,
    QualifiedAssertion,
)

DISPLAY_SELECTION_RULE_REVISION = "context-relevance-qualified-dependency-closure-v2"


class DisplaySelectionError(ValueError):
    """The declared accounting cannot produce the registered display view."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        repair_paths: tuple[str, ...] | None = None,
    ) -> None:
        self.path = path
        self.repair_paths = repair_paths or (path,)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RegisteredDisplaySelection:
    node_ids: tuple[str, ...]
    assertion_ids: tuple[str, ...]


def _node_id(record: object) -> str:
    entity_id = getattr(record, "entity_id", None)
    event_id = getattr(record, "event_id", None)
    identifier = entity_id if entity_id is not None else event_id
    if not isinstance(identifier, str):
        raise DisplaySelectionError(
            "display graph contains an unidentified node",
            path="instance_graph",
        )
    return identifier


def assertion_endpoint_ids(assertion: QualifiedAssertion) -> frozenset[str]:
    """Return every materialized node needed to interpret an assertion.

    The epistemic holder is a semantic dependency, not merely display metadata: an
    attributed belief/report without its holder would be ambiguous and could leak an
    opaque projection-local identifier in the renderer.  Proposition-content endpoints
    need no separate expansion because structural validation requires them to match the
    assertion's binary/role shape exactly.
    """

    if assertion.subject_id is not None:
        assert assertion.object_id is not None
        node_ids = {assertion.subject_id, assertion.object_id}
    else:
        node_ids = {item.object_id for item in assertion.roles}
    if assertion.epistemic_scope is not None:
        node_ids.add(assertion.epistemic_scope.holder_id)
    return frozenset(node_ids)


def compile_registered_display_selection(
    graph: InstanceGraph,
    accounting: BudgetAccounting,
    budgets: OutputBudgets,
) -> RegisteredDisplaySelection:
    """Select exact common capped counts with qualified semantic dependencies."""

    node_records = (*graph.entities, *graph.events)
    assertion_records = graph.assertions
    node_ids = tuple(_node_id(item) for item in node_records)
    if accounting.nodes_used != len(node_records):
        raise DisplaySelectionError(
            "declared node use differs from the entity/event graph count",
            path="budget_accounting.nodes_used",
        )
    if accounting.assertions_used != len(assertion_records):
        raise DisplaySelectionError(
            "declared assertion use differs from the qualified-assertion count",
            path="budget_accounting.assertions_used",
        )
    try:
        accounting.validate_against(budgets)
    except ValueError as error:
        raise DisplaySelectionError(
            str(error),
            path="instance_graph",
            repair_paths=("instance_graph", "decisions", "budget_accounting"),
        ) from error

    expected_node_count = min(len(node_records), budgets.display_node_budget)
    expected_assertion_count = min(len(assertion_records), budgets.display_assertion_budget)
    if accounting.display_nodes_used != expected_node_count:
        raise DisplaySelectionError(
            "declared display-node use must equal graph nodes capped by the display budget",
            path="budget_accounting.display_nodes_used",
        )
    if accounting.display_assertions_used != expected_assertion_count:
        raise DisplaySelectionError(
            "declared display-assertion use must equal graph assertions capped by the "
            "display budget",
            path="budget_accounting.display_assertions_used",
        )

    ranked_assertions = sorted(
        assertion_records,
        key=lambda item: (
            -item.contextual_relevance,
            -item.confidence,
            item.content_hash,
            item.assertion_id,
        ),
    )
    selected_assertions: list[QualifiedAssertion] = []
    required_node_ids: set[str] = set()
    available_node_ids = frozenset(node_ids)
    for assertion in ranked_assertions:
        endpoints = assertion_endpoint_ids(assertion)
        if not endpoints.issubset(available_node_ids):
            raise DisplaySelectionError(
                f"display assertion {assertion.assertion_id!r} has an unknown endpoint",
                path=f"instance_graph.assertions.{assertion.assertion_id}",
            )
        if len(required_node_ids.union(endpoints)) > expected_node_count:
            continue
        selected_assertions.append(assertion)
        required_node_ids.update(endpoints)
        if len(selected_assertions) == expected_assertion_count:
            break
    if len(selected_assertions) != expected_assertion_count:
        raise DisplaySelectionError(
            "declared display budgets cannot retain the required assertion count with "
            "endpoint closure and epistemic-holder dependency closure",
            path="instance_graph",
            repair_paths=("instance_graph", "decisions", "budget_accounting"),
        )

    incident_relevance: dict[str, float] = {node_id: 0.0 for node_id in node_ids}
    for assertion in assertion_records:
        for node_id in assertion_endpoint_ids(assertion):
            incident_relevance[node_id] = max(
                incident_relevance.get(node_id, 0.0), assertion.contextual_relevance
            )
    node_by_id = {_node_id(item): item for item in node_records}
    ranked_remaining_nodes = sorted(
        (node_by_id[node_id] for node_id in node_ids if node_id not in required_node_ids),
        key=lambda item: (
            -incident_relevance[_node_id(item)],
            -item.confidence,
            item.content_hash,
            _node_id(item),
        ),
    )
    for item in ranked_remaining_nodes:
        if len(required_node_ids) == expected_node_count:
            break
        required_node_ids.add(_node_id(item))
    if len(required_node_ids) != expected_node_count:
        raise DisplaySelectionError(
            "registered display rule could not materialize the declared node count",
            path="instance_graph",
            repair_paths=("instance_graph", "decisions", "budget_accounting"),
        )
    return RegisteredDisplaySelection(
        node_ids=tuple(sorted(required_node_ids)),
        assertion_ids=tuple(sorted(item.assertion_id for item in selected_assertions)),
    )


__all__ = [
    "DISPLAY_SELECTION_RULE_REVISION",
    "DisplaySelectionError",
    "RegisteredDisplaySelection",
    "assertion_endpoint_ids",
    "compile_registered_display_selection",
]
