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
    ConditionName,
    ConstructionCapabilities,
    ConstructionOperator,
    ConstructionSeal,
    Entity,
    Event,
    EvidenceRecord,
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
    SpoilerHorizon,
    TemporalKind,
    UpperOntology,
    ValidatedGeneration,
    canonical_sha256,
    projection_validation_target_hash,
    runtime_structural_acceptance_record,
    to_model_visible_evidence,
)
from story_projection_onto.validate import (
    close_projection_dependencies,
    validate_draft_structure,
    validate_projection_lineage,
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _identifier(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def build_c1_preconstruction_request(
    *,
    snapshot_hash: str,
    snapshot_sealed_at: datetime,
    sealed_horizon: SpoilerHorizon,
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
        sealed_horizon=sealed_horizon,
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
    generation: ValidatedGeneration,
    seed_block: int,
    sealed_at: datetime,
) -> ConditionPreparation:
    """Seal exactly one complete C1 graph for reuse across its world's contexts."""

    sealed_horizon = request.sealed_horizon
    if sealed_horizon is None:
        raise ConditionIntegrityError("live C1 sealing requires its trusted sealed horizon")
    if generation.condition is not ConditionName.C1_LLM_PRE:
        raise ConditionIntegrityError("C1 sealer received another condition's generation")
    if generation.request_hash != request.content_hash:
        raise ConditionIntegrityError("C1 validated generation cites a different request")
    if (
        generation.prompt_hash != request.runtime.prompt_hash
        or generation.output_schema_hash != request.runtime.output_schema_hash
        or generation.decoding_manifest_hash != request.runtime.decoding_config_hash
    ):
        raise ConditionIntegrityError("C1 generation differs from the request runtime")
    if generation.generation_started_at < request.requested_at:
        raise ConditionIntegrityError("C1 completion predates its GPU request")
    if sealed_at <= generation.validated_at:
        raise ConditionIntegrityError("C1 seal predates construction completion")
    draft = generation.draft
    draft.budget_accounting.validate_against(request.budgets)
    validation = validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=request.evidence,
        horizon=sealed_horizon,
        budgets=request.budgets,
        capabilities=request.capabilities,
    )
    if not validation.accepted:
        raise ConditionIntegrityError("C1 draft failed deterministic boundary validation")
    if validation.content_hash not in generation.validator_report_hashes:
        raise ConditionIntegrityError("C1 generation does not bind its boundary report")
    ontology_hash = preontology_semantic_hash(request.upper_ontology, draft)
    seal = ConstructionSeal(
        seal_id=_identifier("c1-seal", request.content_hash, ontology_hash, seed_block),
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=request.snapshot_hash,
        ontology_hash=ontology_hash,
        constructed_at=generation.validated_at,
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
        validated_generation=generation,
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
    if preontology.construction_seal.sealed_at >= inputs.query_access.accessed_at:
        raise ConditionIntegrityError("C1 was not sealed before physical query access")
    if preontology.seed_block != inputs.run_config.seed_block:
        raise ConditionIntegrityError("C1 projection and preconstruction seeds differ")
    graph = preontology.draft.instance_graph
    entities = {item.entity_id: item for item in graph.entities}
    events = {item.event_id: item for item in graph.events}
    nodes: dict[str, Entity | Event] = {**entities, **events}
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
        required = close_projection_dependencies(
            graph,
            seed_assertion_ids=(assertion_id,),
        )
        return set(required.assertion_ids), set(required.node_ids)

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

    final_closure = close_projection_dependencies(
        graph,
        seed_assertion_ids=chosen_assertions,
    )
    if (
        final_closure.assertion_ids != frozenset(chosen_assertions)
        or final_closure.node_ids != frozenset(chosen_nodes)
    ):
        raise ConditionIntegrityError("C1 dependency closure changed after selection")
    selected_assertions = tuple(
        item for item in graph.assertions if item.assertion_id in chosen_assertions
    )
    selected_entities = tuple(item for item in graph.entities if item.entity_id in chosen_nodes)
    selected_events = tuple(item for item in graph.events if item.event_id in chosen_nodes)
    selected_propositions: tuple[PropositionContent, ...] = tuple(
        item
        for item in graph.proposition_contents
        if item.proposition_content_id in final_closure.proposition_content_ids
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
        decided_at=inputs.query_processing_started_at,
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
    instance_graph = InstanceGraph(
        entities=selected_entities,
        events=selected_events,
        proposition_contents=selected_propositions,
        assertions=selected_assertions,
    )
    final_draft = OntologyDraft(
        contextual_interpretation=(
            "Deterministic selection from the sealed query-blind C1 ontology."
        ),
        local_schema=preontology.draft.local_schema,
        instance_graph=instance_graph,
        decisions=(decision,),
        budget_accounting=accounting,
    )
    structural_report = validate_draft_structure(
        draft=final_draft,
        upper_ontology=preontology.upper_ontology,
        evidence=tuple(to_model_visible_evidence(item) for item in inputs.packet.evidence),
        horizon=inputs.snapshot.horizon,
        budgets=inputs.context.budgets,
        capabilities=ConstructionCapabilities.fixed_selection(),
    )
    if not structural_report.accepted:
        raise ConditionIntegrityError(
            "C1 selected projection failed deterministic structural validation"
        )
    projection_id = _identifier(
        "c1-projection", preontology.content_hash, inputs.context.content_hash
    )
    validation_target = projection_validation_target_hash(
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        context_hash=inputs.context.content_hash,
        upper_ontology=preontology.upper_ontology,
        local_schema=preontology.draft.local_schema,
        instance_graph=instance_graph,
        decisions=(decision,),
        omissions=(),
        budget_accounting=accounting,
        budgets=inputs.context.budgets,
    )
    validation = runtime_structural_acceptance_record(
        validation_id=_identifier("c1-validation", projection_id, validation_target),
        target_id=validation_target,
        validated_at=inputs.query_processing_started_at,
        diagnostics=(f"structural_report_sha256:{structural_report.content_hash}",),
    )
    generation = preontology.validated_generation
    if generation is None:
        raise ConditionIntegrityError("C1 projection lacks validated preconstruction lineage")
    projection = OntologyProjection(
        projection_id=projection_id,
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        context_hash=inputs.context.content_hash,
        query_access_event_hash=inputs.query_access.content_hash,
        generation_lineage_hash=generation.content_hash,
        raw_output_artifact_hash=generation.raw_output_artifact_hash,
        normalized_draft_hash=generation.normalized_draft_hash,
        validation_bundle_hash=canonical_sha256((validation,)),
        upper_ontology=preontology.upper_ontology,
        local_schema=preontology.draft.local_schema,
        instance_graph=instance_graph,
        decisions=(decision,),
        validation_records=(validation,),
        budget_accounting=accounting,
        budgets=inputs.context.budgets,
        construction_seal=preontology.construction_seal,
        run_id=_identifier("c1-run", inputs.run_config.content_hash, projection_id),
        release_class=inputs.packet.release_class,
    )
    lineage = validate_projection_lineage(
        projection,
        inputs.context,
        query_access=inputs.query_access,
        packet_materialization=inputs.packet_materialization,
        request_created_at=None,
        seed_block=inputs.run_config.seed_block or 0,
        sealed_seed_block=preontology.seed_block,
    )
    if not lineage.accepted:
        raise ConditionIntegrityError("C1 projection failed timing/lineage validation")
    return projection


class LLMPreCondition:
    """Pure orchestration half of C1; GPU execution is deliberately injected elsewhere."""

    condition = ConditionName.C1_LLM_PRE

    def prepare(
        self,
        *,
        request: PreconstructionRequest,
        generation: ValidatedGeneration,
        seed_block: int,
        sealed_at: datetime,
    ) -> ConditionPreparation:
        return seal_c1_preconstruction(
            request=request,
            generation=generation,
            seed_block=seed_block,
            sealed_at=sealed_at,
        )

    def produce(self, inputs: ProduceInputs) -> ConditionAttemptRecord:
        preontology = inputs.preparation.sealed_preontology
        if preontology is None:
            raise ConditionIntegrityError("C1 produce requires its sealed preontology")
        generation = preontology.validated_generation
        if generation is None:
            raise ConditionIntegrityError("C1 produce lacks validated generation lineage")
        config = inputs.run_config
        expected_bindings = (
            (generation.model_stack_hash, config.model_stack_hash),
            (generation.decoding_manifest_hash, config.decoding_manifest_hash),
            (generation.seed_manifest_hash, config.seed_manifest_hash),
            (generation.seed, config.resolved_seed),
            (generation.prompt_hash, config.prompt_hash),
            (generation.output_schema_hash, config.output_schema_hash),
            (generation.capability_manifest_hash, config.capability_manifest_hash),
            (generation.validator_hash, config.validator_hash),
        )
        if any(observed != expected for observed, expected in expected_bindings):
            raise ConditionIntegrityError("C1 generation differs from its frozen run configuration")
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
