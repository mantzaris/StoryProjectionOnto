"""Common final-state compiler for gold and predicted ontology decisions.

The compiler never copies a gold label into a prediction.  It independently reduces
both graphs to the same anchor-based semantic representation, hashes those states,
and only then performs one-to-one matching against emitted construction operations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    GoldContextualProjection,
    GoldContrastInvariant,
    ImmutableRecord,
    OntologyProjection,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.metrics.adapters import AssertionSemanticRecord
from story_projection_onto.metrics.alignment import DecisionFamily, NormalizedDecision
from story_projection_onto.metrics.common import (
    maximum_cardinality_matching,
    revalidated_copy,
)

DECISION_STATE_COMPILER_REVISION = "anchor-final-state-v2"


class DecisionCompilationIssue(ImmutableRecord):
    side: Literal["gold", "prediction"]
    slot_key: str = Field(min_length=1)
    family: DecisionFamily
    reason: str = Field(min_length=1)


class DecisionStateCompilation(ImmutableRecord):
    compiler_revision: Literal["anchor-final-state-v2"] = DECISION_STATE_COMPILER_REVISION
    source_gold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_decisions: tuple[NormalizedDecision, ...]
    predicted_decisions: tuple[NormalizedDecision, ...]
    issues: tuple[DecisionCompilationIssue, ...]
    unrepresentable_gold_count: int = Field(ge=0)
    unrepresentable_prediction_count: int = Field(ge=0)

    @model_validator(mode="after")
    def issue_counts_agree(self) -> Self:
        gold_count = sum(item.side == "gold" for item in self.issues)
        prediction_count = sum(item.side == "prediction" for item in self.issues)
        if gold_count != self.unrepresentable_gold_count:
            raise ValueError("gold decision-compilation issue count does not agree")
        if prediction_count != self.unrepresentable_prediction_count:
            raise ValueError("prediction decision-compilation issue count does not agree")
        return self


class InvariantCompilationIssue(ImmutableRecord):
    invariant_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def _normalize_label(value: str, aliases: Mapping[str, str]) -> str:
    direct = aliases.get(value)
    if direct is not None:
        return direct
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    for prefix in ("contextual_", "actor_", "collective_", "event_role_"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
    return aliases.get(normalized, normalized)


def _extent(extent: Any) -> dict[str, Any]:
    return {
        "kind": extent.kind.value,
        "point": extent.point,
        "start": extent.start,
        "end": extent.end,
        "anchor_id": extent.anchor_id,
        "relation": extent.relation.value if extent.relation is not None else None,
        "partial_order": tuple(
            sorted(
                (item.left_id, item.relation.value, item.right_id)
                for item in extent.partial_order
            )
        ),
    }


def _temporal_scope(scope: Any) -> dict[str, Any]:
    return {
        "story_time": _extent(scope.story_time),
        "validity_time": _extent(scope.validity_time),
        "discourse_position": scope.discourse_position.ordering_key,
        "revelation_order": scope.revelation_position.revelation_order,
    }


def _type_semantics(schema: Any, aliases: Mapping[str, str]) -> dict[str, tuple[Any, ...]]:
    return {
        item.type_id: (
            _normalize_label(item.label, aliases),
            _normalize_label(item.parent_upper_type, aliases),
            item.abstraction.value,
        )
        for item in schema.contextual_types
    }


def _predicate_semantics(
    schema: Any,
    aliases: Mapping[str, str],
) -> dict[str, tuple[Any, ...]]:
    type_semantics = _type_semantics(schema, aliases)

    def type_reference(type_id: str) -> tuple[Any, ...]:
        return type_semantics.get(
            type_id,
            ("upper_or_unresolved", _normalize_label(type_id, aliases)),
        )

    return {
        item.predicate_id: (
            _normalize_label(item.label, aliases),
            _normalize_label(item.parent_upper_relation, aliases),
            item.arity,
            tuple(sorted(_normalize_label(role, aliases) for role in item.role_names)),
            tuple(sorted(type_reference(type_id) for type_id in item.domain_type_ids)),
            tuple(sorted(type_reference(type_id) for type_id in item.range_type_ids)),
        )
        for item in schema.predicates
    }


def _gold_node_keys(gold: GoldContextualProjection) -> dict[str, tuple[Any, ...]]:
    result = {
        item.cluster_id: ("entity", tuple(sorted(item.mention_candidate_ids)))
        for item in gold.entity_partition
    }
    result.update(
        {
            item.event_id: (
                "event",
                tuple(sorted(item.evidence_ids)),
                _extent(item.occurrence_time),
            )
            for item in gold.events
        }
    )
    return result


def _prediction_node_keys(
    projection: OntologyProjection,
    gold: GoldContextualProjection,
) -> dict[str, tuple[Any, ...]]:
    mention_universe = {
        anchor for cluster in gold.entity_partition for anchor in cluster.mention_candidate_ids
    }
    evidence_universe = {
        *(evidence for event in gold.events for evidence in event.evidence_ids),
        *(
            evidence
            for assertion in gold.qualified_assertions
            for evidence in assertion.evidence_ids
        ),
        *(
            evidence
            for item in (
                *gold.local_schema.contextual_types,
                *gold.local_schema.predicates,
            )
            for evidence in item.evidence_ids
        ),
    }
    result = {}
    for entity in projection.instance_graph.entities:
        stable_mentions = tuple(
            sorted(set(entity.supported_mention_candidate_ids) & mention_universe)
        )
        result[entity.entity_id] = (
            "entity",
            stable_mentions
            if stable_mentions
            else (f"UNMATCHED::{canonical_sha256(entity.supported_mention_candidate_ids)}",),
        )
    for event in projection.instance_graph.events:
        stable_evidence = tuple(sorted(set(event.evidence_ids) & evidence_universe))
        result[event.event_id] = (
            "event",
            stable_evidence
            if stable_evidence
            else (f"UNMATCHED::{canonical_sha256(event.evidence_ids)}",),
            _extent(event.occurrence_time),
        )
    return result


def _partition_payload(
    *,
    gold: GoldContextualProjection,
    projection: OntologyProjection | None,
    anchor_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    all_mentions = {
        anchor
        for cluster in gold.entity_partition
        for anchor in cluster.mention_candidate_ids
    }
    universe = tuple(
        sorted(
            all_mentions.intersection(anchor_ids)
        )
    )
    if projection is None:
        groups = [
            tuple(sorted(set(item.mention_candidate_ids).intersection(universe)))
            for item in gold.entity_partition
        ]
        groups = [item for item in groups if item]
    else:
        universe_set = set(universe)
        groups = [
            tuple(sorted(set(item.supported_mention_candidate_ids) & universe_set))
            for item in projection.instance_graph.entities
        ]
        groups = [item for item in groups if item]
        represented = {anchor for group in groups for anchor in group}
        groups.extend((f"ABSENT::{anchor}",) for anchor in universe_set - represented)
    return tuple(sorted(groups))


def _assertion_payloads(
    assertions: Sequence[Any],
    *,
    node_keys: Mapping[str, tuple[Any, ...]],
    predicates: Mapping[str, tuple[Any, ...]],
    mode: Literal["structure", "qualification"],
) -> tuple[Any, ...]:
    values = [
        _assertion_payload(
            assertion,
            node_keys=node_keys,
            predicates=predicates,
            mode=mode,
        )
        for assertion in assertions
    ]
    return tuple(sorted(values, key=canonical_json))


def _assertion_payload(
    assertion: Any,
    *,
    node_keys: Mapping[str, tuple[Any, ...]],
    predicates: Mapping[str, tuple[Any, ...]],
    mode: Literal["structure", "qualification"],
) -> dict[str, Any]:
    predicate = predicates.get(assertion.predicate_id)
    if predicate is None:
        predicate = ("UNDECLARED", assertion.predicate_id)

    def endpoint(object_id: str | None) -> Any:
        if object_id is None:
            return None
        return node_keys.get(object_id, ("UNRESOLVED", object_id))

    subject = endpoint(assertion.subject_id)
    object_ = endpoint(assertion.object_id)
    direction = assertion.direction
    if assertion.subject_id is not None and direction == "inverse":
        # A binary inverse with swapped endpoints is a registered representation-
        # preserving alternative.  Canonicalize its orientation before hashing so
        # the common gold/prediction compiler does not penalize that encoding.
        subject, object_ = object_, subject
        direction = "forward"
    shape = {
        "predicate": predicate,
        "direction": direction,
        "subject": subject,
        "object": object_,
        "roles": tuple(
            sorted(
                ((role.role, endpoint(role.object_id)) for role in assertion.roles),
                key=canonical_json,
            )
        ),
        "evidence_ids": tuple(sorted(set(assertion.evidence_ids))),
    }
    if mode == "qualification":
        epistemic = None
        if assertion.epistemic_scope is not None:
            epistemic = {
                "holder": endpoint(assertion.epistemic_scope.holder_id),
                "attitude": assertion.epistemic_scope.attitude.value,
                "holder_relative_time": _extent(assertion.epistemic_scope.holder_relative_time),
                "evidence_ids": tuple(sorted(set(assertion.epistemic_scope.evidence_ids))),
            }
        shape["temporal_scope"] = _temporal_scope(assertion.temporal_scope)
        shape["epistemic_scope"] = epistemic
        shape["narrative_commitment"] = assertion.narrative_commitment.value
    return shape


def _node_anchor_values(node_key: tuple[Any, ...]) -> tuple[str, ...]:
    if len(node_key) >= 2 and isinstance(node_key[1], tuple):
        return tuple(str(item) for item in node_key[1])
    return ()


def _assertion_semantic_records(
    assertions: Sequence[Any],
    *,
    node_keys: Mapping[str, tuple[Any, ...]],
    predicates: Mapping[str, tuple[Any, ...]],
) -> tuple[AssertionSemanticRecord, ...]:
    records = []
    for assertion in assertions:
        endpoint_ids = (
            (assertion.subject_id, assertion.object_id)
            if assertion.subject_id is not None
            else tuple(role.object_id for role in assertion.roles)
        )
        anchors = set(assertion.evidence_ids)
        for object_id in endpoint_ids:
            if object_id is not None:
                anchors.update(_node_anchor_values(node_keys.get(object_id, ())))
        payload = _assertion_payload(
            assertion,
            node_keys=node_keys,
            predicates=predicates,
            mode="qualification",
        )
        records.append(
            AssertionSemanticRecord(
                assertion_id=assertion.assertion_id,
                anchor_ids=tuple(sorted(anchors)),
                evidence_ids=tuple(sorted(set(assertion.evidence_ids))),
                semantic_signature=canonical_sha256(
                    {
                        "compiler_revision": DECISION_STATE_COMPILER_REVISION,
                        "qualified_assertion": payload,
                    }
                ),
            )
        )
    return tuple(records)


def _state_payload(
    target: NormalizedDecision,
    *,
    gold: GoldContextualProjection,
    projection: OntologyProjection | None,
    aliases: Mapping[str, str],
) -> tuple[dict[str, Any] | None, str | None]:
    family = target.family
    anchor_ids = set(target.anchor_ids)
    schema = gold.local_schema if projection is None else projection.local_schema
    type_semantics = _type_semantics(schema, aliases)
    predicate_semantics = _predicate_semantics(schema, aliases)
    if family is DecisionFamily.MERGE_SPLIT:
        partition = _partition_payload(
            gold=gold,
            projection=projection,
            anchor_ids=target.anchor_ids,
        )
        represented_anchors = {anchor for group in partition for anchor in group}
        if len(represented_anchors) < 2:
            return None, "merge_split_target_lacks_two_gold_mention_anchors"
        return {"entity_partition": partition}, None
    if family is DecisionFamily.ABSTRACTION:
        return {"abstraction": schema.abstraction.value}, None
    if family is DecisionFamily.SCHEMA_RELATION:
        selected = tuple(
            sorted(
                predicate_semantics[item.predicate_id]
                for item in schema.predicates
                if anchor_ids.intersection(item.evidence_ids)
            )
        )
        if not selected:
            return None, "schema_relation_target_has_no_evidence_bound_predicate"
        return {"predicates": selected}, None
    if (
        family is DecisionFamily.CONTEXTUAL_TYPE
        and not target.slot_key.startswith("home-collective/")
    ):
        # GoldEntityCluster intentionally contains no contextual_type_id or role.
        return None, "gold_entity_clusters_lack_contextual_type_and_role_assignments"
    if projection is None:
        node_keys = _gold_node_keys(gold)
        events = gold.events
        assertions = gold.qualified_assertions
    else:
        node_keys = _prediction_node_keys(projection, gold)
        events = projection.instance_graph.events
        assertions = projection.instance_graph.assertions
    selected_assertions = tuple(
        assertion for assertion in assertions if anchor_ids.intersection(assertion.evidence_ids)
    )
    if family is DecisionFamily.CONTEXTUAL_TYPE:
        if not selected_assertions:
            return None, "contextual_type_proxy_has_no_evidence_bound_assertion"
        return {
            "evidence_bound_contextual_relation": _assertion_payloads(
                selected_assertions,
                node_keys=node_keys,
                predicates=predicate_semantics,
                mode="structure",
            )
        }, None
    if family is DecisionFamily.EVENT_REIFICATION:
        selected_events = tuple(
            event for event in events if anchor_ids.intersection(event.evidence_ids)
        )
        if not selected_events and not selected_assertions:
            return None, "event_reification_target_has_no_evidence_bound_event_or_assertion"
        event_payloads = tuple(
            sorted(
                (
                    (
                        tuple(sorted(event.evidence_ids)),
                        type_semantics.get(
                            event.contextual_type_id,
                            ("UNDECLARED", event.contextual_type_id),
                        ),
                        _extent(event.occurrence_time),
                    )
                    for event in selected_events
                ),
                key=canonical_json,
            )
        )
        return {
            "events": event_payloads,
            "assertion_structure": _assertion_payloads(
                selected_assertions,
                node_keys=node_keys,
                predicates=predicate_semantics,
                mode="structure",
            ),
        }, None
    if family is DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION:
        qualification_assertions = (
            tuple(assertions)
            if target.slot_key == "qualification/story-scope"
            else selected_assertions
        )
        if not qualification_assertions:
            return None, "qualification_target_has_no_evidence_bound_assertion"
        return {
            "qualified_assertions": _assertion_payloads(
                qualification_assertions,
                node_keys=node_keys,
                predicates=predicate_semantics,
                mode="qualification",
            )
        }, None
    return None, f"unsupported_decision_family:{family.value}"


def _state_signature(family: DecisionFamily, payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "compiler_revision": DECISION_STATE_COMPILER_REVISION,
            "family": family.value,
            "final_state": payload,
        }
    )


def compile_gold_decision_targets(
    gold: GoldContextualProjection,
    targets: Sequence[NormalizedDecision],
    *,
    predicate_aliases: Mapping[str, str] | None = None,
) -> tuple[tuple[NormalizedDecision, ...], tuple[DecisionCompilationIssue, ...]]:
    """Compile every registered target from the gold final graph, not its prose signature."""

    aliases = predicate_aliases or {}
    compiled = []
    issues = []
    for target in targets:
        payload, reason = _state_payload(
            target,
            gold=gold,
            projection=None,
            aliases=aliases,
        )
        signature = _state_signature(target.family, payload) if payload is not None else None
        if signature is None:
            assert reason is not None
            issues.append(
                DecisionCompilationIssue(
                    side="gold",
                    slot_key=target.slot_key,
                    family=target.family,
                    reason=reason,
                )
            )
            signature = canonical_sha256(
                {
                    "compiler_revision": DECISION_STATE_COMPILER_REVISION,
                    "unrepresentable_gold_slot": target.slot_key,
                    "reason": reason,
                }
            )
        compiled.append(revalidated_copy(target, semantic_signature=signature))
    return tuple(compiled), tuple(issues)


def compile_and_verify_decision_states(
    *,
    gold: GoldContextualProjection,
    projection: OntologyProjection,
    emitted: Sequence[NormalizedDecision],
    gold_targets: Sequence[NormalizedDecision],
    predicate_aliases: Mapping[str, str] | None = None,
) -> DecisionStateCompilation:
    """Compile both final graphs and one-to-one match emitted operations to gold slots."""

    aliases = predicate_aliases or {}
    compiled_gold, gold_issues = compile_gold_decision_targets(
        gold,
        gold_targets,
        predicate_aliases=aliases,
    )
    unrepresentable_gold_slots = {item.slot_key for item in gold_issues}
    emitted_by_key = {
        f"emitted:{index:06d}": item for index, item in enumerate(emitted)
    }
    adjacency: dict[str, tuple[str, ...]] = {}
    prediction_issues = []
    for emitted_key, candidate in emitted_by_key.items():
        candidate_targets = []
        for target in compiled_gold:
            if (
                target.slot_key in unrepresentable_gold_slots
                or candidate.family is not target.family
                or candidate.operator is not target.operator
                or not set(target.anchor_ids).issubset(candidate.anchor_ids)
            ):
                continue
            payload, _ = _state_payload(
                target,
                gold=gold,
                projection=projection,
                aliases=aliases,
            )
            if payload is not None and (
                _state_signature(target.family, payload) == target.semantic_signature
            ):
                candidate_targets.append(target.slot_key)
        adjacency[emitted_key] = tuple(sorted(candidate_targets))
    matches = maximum_cardinality_matching(adjacency)
    target_by_slot = {item.slot_key: item for item in compiled_gold}
    matched_emitted = {emitted_key for emitted_key, _ in matches}
    predicted = [
        target_by_slot[target_slot]
        for emitted_key, target_slot in matches
    ]
    for emitted_key, candidate in emitted_by_key.items():
        if emitted_key in matched_emitted:
            continue
        payload, reason = _state_payload(
            candidate,
            gold=gold,
            projection=projection,
            aliases=aliases,
        )
        signature = (
            _state_signature(candidate.family, payload) if payload is not None else None
        )
        if signature is None:
            prediction_issues.append(
                DecisionCompilationIssue(
                    side="prediction",
                    slot_key=candidate.slot_key,
                    family=candidate.family,
                    reason=reason or "prediction_state_is_unrepresentable",
                )
            )
        predicted.append(
            revalidated_copy(
                candidate,
                semantic_signature=signature
                or canonical_sha256(
                    {
                        "compiler_revision": DECISION_STATE_COMPILER_REVISION,
                        "unrepresentable_prediction_slot": candidate.slot_key,
                        "reason": reason,
                    }
                ),
            )
        )
    issues = tuple(
        sorted((*gold_issues, *prediction_issues), key=lambda item: (item.side, item.slot_key))
    )
    return DecisionStateCompilation(
        source_gold_hash=gold.content_hash,
        source_projection_hash=projection.content_hash,
        gold_decisions=tuple(compiled_gold),
        predicted_decisions=tuple(predicted),
        issues=issues,
        unrepresentable_gold_count=len(gold_issues),
        unrepresentable_prediction_count=len(prediction_issues),
    )


def compile_projection_assertion_semantics(
    gold: GoldContextualProjection,
    projection: OntologyProjection,
    *,
    predicate_aliases: Mapping[str, str] | None = None,
) -> tuple[AssertionSemanticRecord, ...]:
    """Compile prediction assertions in the same anchor namespace as gold invariants."""

    aliases = predicate_aliases or {}
    return _assertion_semantic_records(
        projection.instance_graph.assertions,
        node_keys=_prediction_node_keys(projection, gold),
        predicates=_predicate_semantics(projection.local_schema, aliases),
    )


def compile_gold_contrast_invariants(
    gold: GoldContextualProjection,
    invariants: Sequence[GoldContrastInvariant],
    *,
    predicate_aliases: Mapping[str, str] | None = None,
) -> tuple[tuple[GoldContrastInvariant, ...], tuple[InvariantCompilationIssue, ...]]:
    """Replace prose invariant signatures with exact hashes of gold assertion states."""

    aliases = predicate_aliases or {}
    assertions = _assertion_semantic_records(
        gold.qualified_assertions,
        node_keys=_gold_node_keys(gold),
        predicates=_predicate_semantics(gold.local_schema, aliases),
    )
    invariant_ids = tuple(item.invariant_id for item in invariants)
    if len(invariant_ids) != len(set(invariant_ids)):
        raise ValueError("contrast invariants must have unique IDs")
    assertion_by_key = {
        f"assertion:{index:06d}": assertion for index, assertion in enumerate(assertions)
    }
    candidate_keys = {
        invariant.invariant_id: tuple(
            assertion_key
            for assertion_key, assertion in assertion_by_key.items()
            if set(invariant.anchor_ids).issubset(assertion.anchor_ids)
            and set(invariant.evidence_ids).issubset(assertion.evidence_ids)
        )
        for invariant in invariants
    }
    adjacency = {
        assertion_key: tuple(
            sorted(
                invariant.invariant_id
                for invariant in invariants
                if len(candidate_keys[invariant.invariant_id]) == 1
                if set(invariant.anchor_ids).issubset(assertion.anchor_ids)
                and set(invariant.evidence_ids).issubset(assertion.evidence_ids)
            )
        )
        for assertion_key, assertion in assertion_by_key.items()
    }
    matches = maximum_cardinality_matching(adjacency)
    signature_by_invariant = {
        invariant_id: assertion_by_key[assertion_key].semantic_signature
        for assertion_key, invariant_id in matches
    }
    compiled = []
    issues = []
    for invariant in invariants:
        signature = signature_by_invariant.get(invariant.invariant_id)
        if signature is None:
            candidate_count = len(candidate_keys[invariant.invariant_id])
            reason = (
                "no_anchor_and_evidence_grounded_gold_assertion"
                if candidate_count == 0
                else "ambiguous_anchor_and_evidence_grounded_gold_assertion"
                if candidate_count > 1
                else "gold_assertion_reused_by_multiple_invariants"
            )
            issues.append(
                InvariantCompilationIssue(
                    invariant_id=invariant.invariant_id,
                    reason=reason,
                )
            )
            signature = canonical_sha256(
                {
                    "compiler_revision": DECISION_STATE_COMPILER_REVISION,
                    "unrepresentable_gold_invariant": invariant.invariant_id,
                    "reason": reason,
                }
            )
        compiled.append(revalidated_copy(invariant, expected_signature=signature))
    return tuple(compiled), tuple(issues)


__all__ = [
    "DECISION_STATE_COMPILER_REVISION",
    "DecisionCompilationIssue",
    "DecisionStateCompilation",
    "InvariantCompilationIssue",
    "compile_and_verify_decision_states",
    "compile_gold_contrast_invariants",
    "compile_gold_decision_targets",
    "compile_projection_assertion_semantics",
]
