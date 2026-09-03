"""Query-blind LLM preconstruction and deterministic sealed-ID projection.

No model client lives here.  The module builds the C1 request, accepts a parsed
``OntologyDraft`` returned by the separately metered GPU runtime, seals it, and
projects it after reveal without semantic construction.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from story_projection_onto.conditions.base import (
    ConditionAttemptRecord,
    ConditionIntegrityError,
    ConditionPreparation,
    ProduceInputs,
    SealedPreontology,
    preontology_semantic_hash,
    sealed_semantic_ids,
)
from story_projection_onto.contracts import (
    BudgetAccounting,
    CommitmentCheckStatus,
    ConditionName,
    ConstructionCapabilities,
    ConstructionOperator,
    ConstructionSeal,
    Entity,
    Event,
    EvidenceRecord,
    EvidenceSupportStatus,
    InstanceGraph,
    OntologyDecision,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    PreconstructionRequest,
    PropositionContent,
    QualifiedAssertion,
    RunOutcome,
    RuntimeIdentifiers,
    TemporalDeterminationStatus,
    TemporalKind,
    UpperOntology,
    ValidationRecord,
    ValidationStatus,
    to_model_visible_evidence,
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _identifier(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def build_c1_preconstruction_request(
    *,
    snapshot_hash: str,
    snapshot_sealed_at: datetime,
    ordered_snapshot_evidence_ids: Sequence[str],
    evidence: Sequence[EvidenceRecord],
    upper_ontology: UpperOntology,
    preconstruction_budgets: OutputBudgets,
    runtime: RuntimeIdentifiers,
    requested_at: datetime,
) -> PreconstructionRequest:
    """Serialize a complete query-blind request with no context-shaped fields."""

    if requested_at < snapshot_sealed_at:
        raise ConditionIntegrityError("C1 request predates its sealed evidence snapshot")
    if tuple(item.evidence_id for item in evidence) != tuple(ordered_snapshot_evidence_ids):
        raise ConditionIntegrityError("C1 request requires complete ordered snapshot evidence")
    return PreconstructionRequest(
        request_id=_identifier("c1-pre-request", snapshot_hash, runtime.content_hash),
        snapshot_hash=snapshot_hash,
        evidence=tuple(to_model_visible_evidence(item) for item in evidence),
        upper_ontology=upper_ontology,
        budgets=preconstruction_budgets,
        capabilities=ConstructionCapabilities.prequery_construction(),
        runtime=runtime,
        requested_at=requested_at,
    )


def seal_c1_preconstruction(
    *,
    request: PreconstructionRequest,
    draft: OntologyDraft,
    seed_block: int,
    constructed_at: datetime,
    sealed_at: datetime,
) -> ConditionPreparation:
    """Seal exactly one complete C1 graph for reuse across its world's contexts."""

    if constructed_at < request.requested_at:
        raise ConditionIntegrityError("C1 completion predates its GPU request")
    if sealed_at < constructed_at:
        raise ConditionIntegrityError("C1 seal predates construction completion")
    draft.budget_accounting.validate_against(request.budgets)
    ontology_hash = preontology_semantic_hash(request.upper_ontology, draft)
    seal = ConstructionSeal(
        seal_id=_identifier("c1-seal", request.content_hash, ontology_hash, seed_block),
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=request.snapshot_hash,
        ontology_hash=ontology_hash,
        constructed_at=constructed_at,
        sealed_at=sealed_at,
        sealed_object_ids=sealed_semantic_ids(draft),
    )
    preontology = SealedPreontology(
        artifact_id=_identifier("c1-preontology", seal.content_hash),
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=request.snapshot_hash,
        upper_ontology=request.upper_ontology,
        draft=draft,
        construction_seal=seal,
        seed_block=seed_block,
    )
    return ConditionPreparation(
        preparation_id=_identifier("c1-preparation", preontology.content_hash),
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=request.snapshot_hash,
        completed_at=sealed_at,
        sealed_preontology=preontology,
    )


def _terms(value: str) -> frozenset[str]:
    return frozenset(item.group(0).casefold() for item in _TOKEN.finditer(value))


def _endpoint_ids(assertion: QualifiedAssertion) -> frozenset[str]:
    endpoints = {item for item in (assertion.subject_id, assertion.object_id) if item is not None}
    endpoints.update(role.object_id for role in assertion.roles)
    return frozenset(endpoints)


def _time_score(assertion: QualifiedAssertion, inputs: ProduceInputs) -> float:
    query = inputs.context.story_scope
    found = assertion.temporal_scope.story_time
    if query.kind is TemporalKind.POINT:
        query_start = query_end = query.point
    elif query.kind is TemporalKind.INTERVAL:
        query_start, query_end = query.start, query.end
    else:
        return 0.0
    if found.kind is TemporalKind.POINT:
        found_start = found_end = found.point
    elif found.kind is TemporalKind.INTERVAL:
        found_start, found_end = found.start, found.end
    else:
        return 0.0
    query_left = float("-inf") if query_start is None else query_start
    query_right = float("inf") if query_end is None else query_end
    found_left = float("-inf") if found_start is None else found_start
    found_right = float("inf") if found_end is None else found_end
    return 0.2 if max(query_left, found_left) <= min(query_right, found_right) else -0.35


def project_sealed_c1(
    preontology: SealedPreontology,
    inputs: ProduceInputs,
) -> OntologyProjection:
    """Apply frozen relevance/support/path selection to a sealed comprehensive C1 graph."""

    if preontology.condition is not ConditionName.C1_LLM_PRE:
        raise ConditionIntegrityError("C1 projector requires a C1 sealed preontology")
    if preontology.construction_seal.sealed_at >= inputs.context.revealed_at:
        raise ConditionIntegrityError("C1 was not sealed strictly before query reveal")
    if preontology.seed_block != inputs.run_config.seed_block:
        raise ConditionIntegrityError("C1 projection and preconstruction seeds differ")
    graph = preontology.draft.instance_graph
    entities = {item.entity_id: item for item in graph.entities}
    events = {item.event_id: item for item in graph.events}
    nodes: dict[str, Entity | Event] = {**entities, **events}
    assertions = {item.assertion_id: item for item in graph.assertions}
    query_terms = _terms(
        " ".join((inputs.context.wording, inputs.context.lens, inputs.context.target))
    )
    evidence_frequency = Counter(
        evidence_id for item in graph.assertions for evidence_id in item.evidence_ids
    )

    def rank(assertion: QualifiedAssertion) -> tuple[float, str]:
        text = [assertion.predicate_id, assertion.why_matters]
        text.extend(nodes[item].label for item in _endpoint_ids(assertion) if item in nodes)
        lexical = len(query_terms & _terms(" ".join(text))) / max(1, len(query_terms))
        low_frequency = min(
            (1.0 / evidence_frequency[item] for item in assertion.evidence_ids),
            default=0.0,
        )
        return (
            lexical
            + _time_score(assertion, inputs)
            + 0.1 * assertion.confidence
            + 0.12 * low_frequency,
            assertion.assertion_id,
        )

    ordered = sorted(graph.assertions, key=lambda item: (-rank(item)[0], rank(item)[1]))
    chosen_assertions: set[str] = set()
    chosen_nodes: set[str] = set()

    def closure(assertion_id: str) -> tuple[set[str], set[str]]:
        needed_assertions = {assertion_id}
        needed_nodes: set[str] = set()
        changed = True
        while changed:
            changed = False
            for needed_id in tuple(needed_assertions):
                for node_id in _endpoint_ids(assertions[needed_id]):
                    if node_id not in nodes:
                        continue
                    if node_id not in needed_nodes:
                        needed_nodes.add(node_id)
                        changed = True
                    for support_id in nodes[node_id].description_assertion_ids:
                        if support_id in assertions and support_id not in needed_assertions:
                            needed_assertions.add(support_id)
                            changed = True
        return needed_assertions, needed_nodes

    while ordered:
        ordered.sort(
            key=lambda item: (
                -(rank(item)[0] + (0.15 if chosen_nodes & _endpoint_ids(item) else 0.0)),
                item.assertion_id,
            )
        )
        candidate = ordered.pop(0)
        required_assertions, required_nodes = closure(candidate.assertion_id)
        next_assertions = chosen_assertions | required_assertions
        next_nodes = chosen_nodes | required_nodes
        if len(next_assertions) > inputs.context.budgets.assertion_budget:
            continue
        if len(next_nodes) > inputs.context.budgets.node_budget:
            continue
        chosen_assertions, chosen_nodes = next_assertions, next_nodes

    selected_assertions = tuple(
        item for item in graph.assertions if item.assertion_id in chosen_assertions
    )
    selected_entities = tuple(item for item in graph.entities if item.entity_id in chosen_nodes)
    selected_events = tuple(item for item in graph.events if item.event_id in chosen_nodes)
    selected_proposition_ids = {
        item.proposition_content_id
        for item in selected_assertions
        if item.proposition_content_id is not None
    }
    selected_propositions: tuple[PropositionContent, ...] = tuple(
        item
        for item in graph.proposition_contents
        if item.proposition_content_id in selected_proposition_ids
    )
    selected_ids = {
        *(item.entity_id for item in selected_entities),
        *(item.event_id for item in selected_events),
        *(item.proposition_content_id for item in selected_propositions),
        *(item.assertion_id for item in selected_assertions),
    }
    if not selected_ids.issubset(preontology.construction_seal.sealed_object_ids):
        raise ConditionIntegrityError("C1 fixed projector emitted an unsealed semantic ID")
    evidence_ids = tuple(
        sorted({item for assertion in selected_assertions for item in assertion.evidence_ids})
    )
    decision = OntologyDecision(
        decision_id=_identifier(
            "c1-selection", preontology.content_hash, inputs.context.content_hash
        ),
        operator=ConstructionOperator.SELECTION,
        evidence_ids=evidence_ids or (inputs.packet.ordered_evidence_ids[0],),
        rationale="Frozen query-time scorer selected only complete-seal C1 semantic objects.",
        decided_at=inputs.context.revealed_at,
        input_object_ids=tuple(sorted(selected_ids)),
    )
    node_count = len(selected_entities) + len(selected_events)
    assertion_count = len(selected_assertions)
    accounting = BudgetAccounting(
        nodes_used=node_count,
        assertions_used=assertion_count,
        display_nodes_used=min(node_count, inputs.context.budgets.display_node_budget),
        display_assertions_used=min(
            assertion_count, inputs.context.budgets.display_assertion_budget
        ),
        input_tokens=0,
        output_tokens=0,
    )
    projection_id = _identifier(
        "c1-projection", preontology.content_hash, inputs.context.content_hash
    )
    validation = ValidationRecord(
        validation_id=_identifier("c1-validation", projection_id),
        target_id=projection_id,
        validation_status=ValidationStatus.ACCEPTED,
        evidence_support_status=EvidenceSupportStatus.SUPPORTED,
        temporal_status=TemporalDeterminationStatus.VALID,
        commitment_status=CommitmentCheckStatus.VALID,
        validated_at=inputs.context.revealed_at,
    )
    return OntologyProjection(
        projection_id=projection_id,
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        context_hash=inputs.context.content_hash,
        upper_ontology=preontology.upper_ontology,
        local_schema=preontology.draft.local_schema,
        instance_graph=InstanceGraph(
            entities=selected_entities,
            events=selected_events,
            proposition_contents=selected_propositions,
            assertions=selected_assertions,
        ),
        decisions=(decision,),
        validation_records=(validation,),
        budget_accounting=accounting,
        budgets=inputs.context.budgets,
        construction_seal=preontology.construction_seal,
        run_id=_identifier("c1-run", inputs.run_config.content_hash, projection_id),
        release_class=inputs.packet.release_class,
    )


class LLMPreCondition:
    """Pure orchestration half of C1; GPU execution is deliberately injected elsewhere."""

    condition = ConditionName.C1_LLM_PRE

    def produce(self, inputs: ProduceInputs) -> ConditionAttemptRecord:
        preontology = inputs.preparation.sealed_preontology
        if preontology is None:
            raise ConditionIntegrityError("C1 produce requires its sealed preontology")
        projection = project_sealed_c1(preontology, inputs)
        return ConditionAttemptRecord(
            attempt_id=_identifier(
                "c1-attempt", inputs.context.content_hash, inputs.run_config.seed_block
            ),
            condition=self.condition,
            unit_id=inputs.context.context_id,
            seed_block=inputs.run_config.seed_block,
            outcome=RunOutcome.SUCCEEDED,
            projection=projection,
            release_class=inputs.packet.release_class,
        )
