"""Typed adapters from validated projections to condition-blind metric records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    ConstructionOperator,
    ImmutableRecord,
    OntologyDecision,
    OntologyProjection,
    ValidationStatus,
    canonical_sha256,
)
from story_projection_onto.metrics.alignment import (
    AlignmentIntegrityError,
    DecisionFamily,
    NormalizedDecision,
)
from story_projection_onto.metrics.common import revalidated_copy
from story_projection_onto.metrics.config import StudyMetricConfiguration
from story_projection_onto.metrics.entropy import AssertionRelationRecord, SemanticEdge

_OPERATOR_FAMILY = {
    ConstructionOperator.MERGE: DecisionFamily.MERGE_SPLIT,
    ConstructionOperator.SPLIT: DecisionFamily.MERGE_SPLIT,
    ConstructionOperator.CONTEXTUAL_TYPE: DecisionFamily.CONTEXTUAL_TYPE,
    ConstructionOperator.SCHEMA_RELATION: DecisionFamily.SCHEMA_RELATION,
    ConstructionOperator.EVENT_REIFICATION: DecisionFamily.EVENT_REIFICATION,
    ConstructionOperator.ABSTRACTION: DecisionFamily.ABSTRACTION,
    ConstructionOperator.TEMPORAL_QUALIFICATION: (
        DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
    ),
    ConstructionOperator.EPISTEMIC_QUALIFICATION: (
        DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
    ),
}


class AssertionSemanticRecord(ImmutableRecord):
    assertion_id: str = Field(min_length=1)
    anchor_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    semantic_signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def anchors_and_evidence_are_canonical(self) -> Self:
        for values in (self.anchor_ids, self.evidence_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("assertion anchors and evidence must be sorted and unique")
        return self


class ProjectionMetricAdapter(ImmutableRecord):
    """Canonical, ID-renaming-resistant view used by structural/scoring code."""

    projection_id: str = Field(min_length=1)
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    structurally_valid: bool
    validation_failure_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    edges: tuple[SemanticEdge, ...]
    assertion_relations: tuple[AssertionRelationRecord, ...]
    object_anchors: tuple[tuple[str, tuple[str, ...]], ...]
    assertion_semantics: tuple[AssertionSemanticRecord, ...]
    normalized_decisions: tuple[NormalizedDecision, ...]

    @property
    def anchors_by_object(self) -> Mapping[str, tuple[str, ...]]:
        return dict(self.object_anchors)


def projection_is_structurally_valid(projection: OntologyProjection) -> bool:
    """Fail closed: an output is valid only with a nonempty all-accepted validation set."""

    return bool(projection.validation_records) and all(
        record.validation_status is ValidationStatus.ACCEPTED
        for record in projection.validation_records
    )


def projection_is_content_bearing(projection: OntologyProjection) -> bool:
    """Return whether the projection contains at least one entity/event graph node.

    The registered analysis treats node-empty output as semantic failure even when its
    serialization and structural-validation envelope are well formed.  Keeping this
    predicate separate from structural validity prevents an empty response from earning
    favorable entropy, community, or clutter values.
    """

    return bool(
        projection.instance_graph.entities or projection.instance_graph.events
    )


def _object_payloads_and_anchors(
    projection: OntologyProjection,
) -> tuple[dict[str, object], dict[str, tuple[str, ...]]]:
    payloads: dict[str, object] = {}
    anchors: dict[str, tuple[str, ...]] = {}
    schema_evidence = tuple(
        sorted(
            {
                evidence_id
                for item in (
                    *projection.local_schema.contextual_types,
                    *projection.local_schema.predicates,
                )
                for evidence_id in item.evidence_ids
            }
        )
    )
    payloads[projection.local_schema.schema_id] = {
        "kind": "schema",
        "abstraction": projection.local_schema.abstraction,
    }
    anchors[projection.local_schema.schema_id] = schema_evidence
    for local_type in projection.local_schema.contextual_types:
        payloads[local_type.type_id] = {
            "kind": "type",
            "label": local_type.label,
            "definition": local_type.definition,
            "parent": local_type.parent_upper_type,
            "abstraction": local_type.abstraction,
        }
        anchors[local_type.type_id] = tuple(sorted(set(local_type.evidence_ids)))
    for predicate in projection.local_schema.predicates:
        domain = tuple(
            sorted(
                canonical_sha256(payloads[type_id])
                if type_id in payloads
                else type_id
                for type_id in predicate.domain_type_ids
            )
        )
        range_ = tuple(
            sorted(
                canonical_sha256(payloads[type_id])
                if type_id in payloads
                else type_id
                for type_id in predicate.range_type_ids
            )
        )
        payloads[predicate.predicate_id] = {
            "kind": "predicate",
            "label": predicate.label,
            "definition": predicate.definition,
            "arity": predicate.arity,
            "roles": predicate.role_names,
            "domain": domain,
            "range": range_,
            "parent": predicate.parent_upper_relation,
        }
        anchors[predicate.predicate_id] = tuple(sorted(set(predicate.evidence_ids)))
    for entity in projection.instance_graph.entities:
        object_anchors = tuple(
            sorted(set(entity.supported_mention_candidate_ids) | set(entity.evidence_ids))
        )
        payloads[entity.entity_id] = {
            "kind": "entity",
            "anchors": object_anchors,
            "type": canonical_sha256(payloads[entity.contextual_type_id]),
            "role": entity.contextual_role,
            "abstraction": entity.abstraction,
        }
        anchors[entity.entity_id] = object_anchors
    for event in projection.instance_graph.events:
        object_anchors = tuple(sorted(set(event.evidence_ids)))
        payloads[event.event_id] = {
            "kind": "event",
            "anchors": object_anchors,
            "type": canonical_sha256(payloads[event.contextual_type_id]),
            "occurrence": event.occurrence_time,
            "reason": event.reification_reason,
        }
        anchors[event.event_id] = object_anchors
    for proposition in projection.instance_graph.proposition_contents:
        payloads[proposition.proposition_content_id] = {
            "kind": "proposition",
            "predicate": canonical_sha256(payloads[proposition.predicate_id]),
            "subject": anchors.get(proposition.subject_id or "", ()),
            "object": anchors.get(proposition.object_id or "", ()),
            "roles": tuple(
                (role.role, anchors.get(role.object_id, (role.object_id,)))
                for role in proposition.roles
            ),
            "temporal": proposition.temporal_content,
        }
        anchors[proposition.proposition_content_id] = tuple(
            sorted(set(proposition.evidence_ids))
        )
    for assertion in projection.instance_graph.assertions:
        endpoint_ids = (
            (assertion.subject_id, assertion.object_id)
            if assertion.subject_id is not None
            else tuple(role.object_id for role in assertion.roles)
        )
        endpoint_anchors = {
            anchor
            for object_id in endpoint_ids
            if object_id is not None
            for anchor in anchors.get(object_id, ())
        }
        assertion_anchors = tuple(sorted(endpoint_anchors | set(assertion.evidence_ids)))
        payloads[assertion.assertion_id] = {
            "kind": "assertion",
            "predicate": canonical_sha256(payloads[assertion.predicate_id]),
            "endpoints": tuple(anchors.get(item or "", (item,)) for item in endpoint_ids),
            "direction": assertion.direction,
            "temporal": assertion.temporal_scope,
            "epistemic": assertion.epistemic_scope,
            "commitment": assertion.narrative_commitment,
        }
        anchors[assertion.assertion_id] = assertion_anchors
    return payloads, anchors


def _semantic_signature(
    decision: OntologyDecision,
    *,
    payloads: Mapping[str, object],
    anchors: Mapping[str, Sequence[str]],
) -> tuple[str, tuple[str, ...]]:
    referenced_ids = (
        *decision.input_object_ids,
        *decision.created_object_ids,
        *decision.removed_object_ids,
    )
    unresolved_created = sorted(set(decision.created_object_ids) - set(payloads))
    if unresolved_created:
        raise AlignmentIntegrityError(
            f"decision {decision.decision_id!r} has unresolved created objects: "
            f"{unresolved_created!r}"
        )
    resolved_anchors = set(decision.evidence_ids)
    decision_evidence = set(decision.evidence_ids)
    # Close over projection objects grounded in the decision evidence so scorer
    # mention anchors can be verified even when a decision names only a schema ID.
    for object_anchor_values in anchors.values():
        if decision_evidence.intersection(object_anchor_values):
            resolved_anchors.update(object_anchor_values)
    for object_id in referenced_ids:
        # Input/removed IDs may lawfully be sealed evidence-candidate anchors rather
        # than surviving ontology objects. They remain stable scorer-side anchors.
        resolved_anchors.update(anchors.get(object_id, (object_id,)))
    normalized_anchors = tuple(sorted(resolved_anchors))
    final_objects = tuple(
        sorted(
            canonical_sha256(payloads[object_id])
            for object_id in set(decision.input_object_ids) | set(decision.created_object_ids)
            if object_id in payloads
        )
    )
    signature = canonical_sha256(
        {
            "operator": decision.operator.value,
            "anchors": normalized_anchors,
            "final_objects": final_objects,
            "removed_anchor_sets": tuple(
                sorted(
                    tuple(anchors.get(object_id, (object_id,)))
                    for object_id in decision.removed_object_ids
                )
            ),
        }
    )
    return signature, normalized_anchors


def normalize_projection_decisions(
    projection: OntologyProjection,
) -> tuple[NormalizedDecision, ...]:
    """Derive slots and semantics exclusively from the emitted projection."""

    payloads, anchors = _object_payloads_and_anchors(projection)
    candidates: list[
        tuple[DecisionFamily, ConstructionOperator, tuple[str, ...], str]
    ] = []
    for decision in projection.decisions:
        family = _OPERATOR_FAMILY.get(decision.operator)
        if family is None:
            continue
        signature, decision_anchors = _semantic_signature(
            decision,
            payloads=payloads,
            anchors=anchors,
        )
        candidates.append(
            (family, decision.operator, decision_anchors, signature)
        )

    # The evidence/mention anchor group is the stable semantic slot across contexts,
    # which lets a changed final state become a substitution in contrastive scoring.
    # A deterministic rank permits multiple legitimate operations over the same
    # anchors instead of collapsing or rejecting them. Exact duplicates remain as
    # separate predictions and therefore incur the appropriate precision penalty.
    grouped: dict[
        tuple[DecisionFamily, tuple[str, ...]],
        list[tuple[ConstructionOperator, str]],
    ] = {}
    for family, operator, decision_anchors, signature in candidates:
        grouped.setdefault((family, decision_anchors), []).append((operator, signature))
    normalized: list[NormalizedDecision] = []
    for (family, decision_anchors), operations in sorted(
        grouped.items(),
        key=lambda item: (item[0][0].value, item[0][1]),
    ):
        anchor_hash = canonical_sha256(decision_anchors)[:24]
        for rank, (operator, signature) in enumerate(
            sorted(operations, key=lambda item: (item[0].value, item[1]))
        ):
            normalized.append(
                NormalizedDecision(
                    slot_key=f"emitted/{family.value}/{anchor_hash}/{rank:03d}",
                    family=family,
                    operator=operator,
                    anchor_ids=decision_anchors,
                    semantic_signature=signature,
                )
            )
    if len(normalized) != len(candidates):
        raise AssertionError(
            "every constructive decision must receive exactly one normalized slot"
        )
    return tuple(normalized)


def verify_projection_decisions(
    projection: OntologyProjection,
    *,
    gold_targets: Sequence[NormalizedDecision],
) -> tuple[NormalizedDecision, ...]:
    """Match emitted decisions to frozen targets without rewriting their semantics.

    Gold and prediction compilers must emit the same canonical final-state signature.
    An operator label and overlapping evidence are only candidate-generation signals;
    they are never enough to turn an emitted action into a gold decision.  Matching is
    one-to-one so one broad model decision cannot satisfy several gold targets.
    """

    return verify_normalized_decisions(
        normalize_projection_decisions(projection),
        gold_targets=gold_targets,
    )


def verify_normalized_decisions(
    emitted: Sequence[NormalizedDecision],
    *,
    gold_targets: Sequence[NormalizedDecision],
) -> tuple[NormalizedDecision, ...]:
    """Map exact emitted final states to frozen slots with one-to-one matching."""

    target_slots = [item.slot_key for item in gold_targets]
    if len(target_slots) != len(set(target_slots)):
        raise AlignmentIntegrityError("gold ontology-decision targets require unique slots")
    adjacency: dict[str, tuple[str, ...]] = {}
    emitted_by_key: dict[str, NormalizedDecision] = {}
    for index, candidate in enumerate(emitted):
        emitted_key = f"emitted:{index:06d}"
        emitted_by_key[emitted_key] = candidate
        adjacency[emitted_key] = tuple(
            sorted(
                target.slot_key
                for target in gold_targets
                if candidate.family is target.family
                and candidate.operator is target.operator
                and candidate.anchor_ids == target.anchor_ids
                and candidate.semantic_signature == target.semantic_signature
            )
        )

    from story_projection_onto.metrics.common import maximum_cardinality_matching

    matches = maximum_cardinality_matching(adjacency)
    matched_emitted = {emitted_key for emitted_key, _ in matches}
    # Return the emitted records with the frozen target slot only after every semantic
    # field agrees.  Keeping the emitted signature makes accidental gold substitution
    # mechanically visible in reviews and hashes.
    verified = [
        revalidated_copy(emitted_by_key[emitted_key], slot_key=target_slot)
        for emitted_key, target_slot in matches
    ]
    verified.extend(
        candidate
        for emitted_key, candidate in emitted_by_key.items()
        if emitted_key not in matched_emitted
    )
    slots = [item.slot_key for item in verified]
    if len(slots) != len(set(slots)):
        raise AlignmentIntegrityError("verified and extra decision slots collide")
    return tuple(verified)


def adapt_projection_for_metrics(
    projection: OntologyProjection,
    configuration: StudyMetricConfiguration,
) -> ProjectionMetricAdapter:
    """Build the sole canonical projection-to-structural-metrics boundary."""

    payloads, anchors = _object_payloads_and_anchors(projection)
    del payloads
    node_ids = tuple(
        sorted(
            [item.entity_id for item in projection.instance_graph.entities]
            + [item.event_id for item in projection.instance_graph.events]
        )
    )
    node_set = set(node_ids)
    event_ids = {item.event_id for item in projection.instance_graph.events}
    predicates = {item.predicate_id: item for item in projection.local_schema.predicates}
    relation_map = configuration.upper_relation_map
    edges: list[SemanticEdge] = []
    assertion_relations: list[AssertionRelationRecord] = []
    semantics: list[AssertionSemanticRecord] = []
    for assertion in projection.instance_graph.assertions:
        predicate = predicates.get(assertion.predicate_id)
        native = assertion.predicate_id
        predicate_label = predicate.label if predicate is not None else assertion.predicate_id
        parent = predicate.parent_upper_relation if predicate is not None else ""
        canonical = relation_map.get(parent)
        assertion_relations.append(
            AssertionRelationRecord(
                assertion_id=assertion.assertion_id,
                native_predicate_id=native,
                canonical_predicate_id=canonical,
            )
        )
        if assertion.subject_id is not None and assertion.object_id is not None:
            pairs = ((assertion.subject_id, assertion.object_id),)
        else:
            role_objects = tuple(role.object_id for role in assertion.roles)
            hubs = tuple(item for item in role_objects if item in event_ids)
            if hubs:
                pairs = tuple(
                    (hub, item)
                    for hub in hubs
                    for item in role_objects
                    if item != hub
                )
            else:
                pairs = tuple(
                    (role_objects[left], role_objects[right])
                    for left in range(len(role_objects))
                    for right in range(left + 1, len(role_objects))
                )
        unknown = sorted({item for pair in pairs for item in pair} - node_set)
        if unknown:
            raise AlignmentIntegrityError(
                f"assertion {assertion.assertion_id!r} has non-node metric endpoints: {unknown!r}"
            )
        for index, (source, target) in enumerate(pairs):
            edges.append(
                SemanticEdge(
                    edge_id=f"{assertion.assertion_id}:skeleton:{index}",
                    source_id=source,
                    target_id=target,
                    native_predicate_id=native,
                    canonical_predicate_id=canonical,
                )
            )
        assertion_anchors = tuple(sorted(set(anchors[assertion.assertion_id])))
        semantics.append(
            AssertionSemanticRecord(
                assertion_id=assertion.assertion_id,
                anchor_ids=assertion_anchors,
                evidence_ids=tuple(sorted(set(assertion.evidence_ids))),
                semantic_signature=canonical_sha256(
                    {
                        "predicate": native.casefold(),
                        "predicate_label": predicate_label.casefold(),
                        "parent": parent,
                        "anchors": assertion_anchors,
                        "temporal": assertion.temporal_scope,
                        "epistemic": assertion.epistemic_scope,
                        "commitment": assertion.narrative_commitment,
                    }
                ),
            )
        )
    failures = tuple(
        sorted(
            item.validation_id
            for item in projection.validation_records
            if item.validation_status is not ValidationStatus.ACCEPTED
        )
    )
    if not projection.validation_records:
        failures = ("missing-validation-records",)
    return ProjectionMetricAdapter(
        projection_id=projection.projection_id,
        projection_hash=projection.content_hash,
        structurally_valid=projection_is_structurally_valid(projection),
        validation_failure_ids=failures,
        node_ids=node_ids,
        edges=tuple(edges),
        assertion_relations=tuple(assertion_relations),
        object_anchors=tuple(sorted(anchors.items())),
        assertion_semantics=tuple(sorted(semantics, key=lambda item: item.assertion_id)),
        normalized_decisions=normalize_projection_decisions(projection),
    )


__all__ = [
    "AssertionSemanticRecord",
    "ProjectionMetricAdapter",
    "adapt_projection_for_metrics",
    "normalize_projection_decisions",
    "projection_is_content_bearing",
    "projection_is_structurally_valid",
    "verify_normalized_decisions",
    "verify_projection_decisions",
]
