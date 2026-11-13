from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import (
    AbstractionLevel,
    BudgetAccounting,
    ConditionName,
    ConstructionCertificate,
    ConstructionOperator,
    ConstructionSeal,
    DiscoursePosition,
    Entity,
    EpistemicAttitude,
    EpistemicScope,
    EvidencePacket,
    EvidenceRecord,
    ExplicitValueState,
    FeedbackAction,
    FeedbackAnchor,
    FeedbackResolutionStatus,
    HolderRelativeTime,
    InstanceGraph,
    LocalContextSchema,
    LocalPredicateDefinition,
    LocalTypeDefinition,
    MentionCandidate,
    NarrativeCommitment,
    OntologyDecision,
    OntologyProjection,
    OutputBudgets,
    PreQueryInventory,
    PropositionContent,
    ProvenanceReference,
    QualifiedAssertion,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RevelationPosition,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
    TemporalScope,
    UpperOntology,
    ValidationRecord,
    ValidityTime,
    VisualizationTemporalFilter,
    canonical_sha256,
    projection_validation_target_hash,
    runtime_structural_acceptance_record,
)
from story_projection_onto.ui import (
    NODE_DESCRIPTION_WITHHELD,
    WHY_MATTERS_WITHHELD,
    FeedbackEpisodeKind,
    FeedbackEpisodePlan,
    FeedbackReplayExpectation,
    FeedbackStudyPlan,
    MergeSplitIntent,
    MergeSplitOperation,
    RefineContextIntent,
    RevisionExecutionResult,
    SemanticAssessmentSourceKind,
    SemanticDisplayMode,
    SemanticSupportStatus,
    VisualizationAssertionSemanticAssessment,
    VisualizationChangeKind,
    VisualizationCompilationError,
    VisualizationContentScope,
    VisualizationSemanticOverlay,
    _structurally_accepted_assertion_ids,
    apply_context_refinement,
    assert_revision_anchors_resolve_in_packet,
    assert_revision_has_no_projection_local_anchors,
    build_merge_split_instruction,
    build_refine_context_instruction,
    build_visualization_bundle,
    compare_visualizations,
    filter_visualization_bundle,
    resolve_feedback_for_condition,
    verify_feedback_replay,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def provenance(evidence_id: str, order: int) -> ProvenanceReference:
    return ProvenanceReference(
        provenance_id=f"prov-{evidence_id}",
        evidence_id=evidence_id,
        extraction_method="hand-authored synthetic fixture",
        locator=f"fixture:{order}",
        source_artifact_hash=digest(f"fixture-source:{evidence_id}:{order}"),
        confidence=1.0,
    )


def mention(candidate_id: str, evidence_id: str, surface: str) -> MentionCandidate:
    return MentionCandidate(
        candidate_id=candidate_id,
        evidence_id=evidence_id,
        start_char=0,
        end_char=len(surface),
        surface=surface,
        surface_hash=digest(surface),
    )


def context() -> QueryContext:
    return QueryContext(
        context_id="context-alpha",
        wording="Who enabled the beacon signal?",
        lens="causal responsibility",
        target="beacon signal",
        story_scope=StoryTime(kind=TemporalKind.INTERVAL, start=1, end=9),
        spoiler_horizon=SpoilerHorizon(
            horizon_id="horizon-nine",
            max_discourse_position=DiscoursePosition(passage_order=9),
            max_revelation_position=RevelationPosition(revelation_order=9),
        ),
        abstraction=AbstractionLevel.EVENT_ROLE,
        budgets=OutputBudgets(
            node_budget=10,
            assertion_budget=12,
            display_node_budget=10,
            display_assertion_budget=12,
        ),
        revealed_at=NOW,
    )


def packet() -> EvidencePacket:
    records = []
    for order, (evidence_id, candidate_id, text) in enumerate(
        (
            ("ev-a", "m-a", "Ari carried the key."),
            ("ev-b", "m-b", "Bex guarded the beacon."),
            ("ev-c", "m-c", "Cato received the signal."),
        ),
        1,
    ):
        records.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                passage_id=f"passage-{order}",
                text=text,
                text_hash=digest(text),
                discourse_position=DiscoursePosition(passage_order=order),
                mention_candidates=(mention(candidate_id, evidence_id, text.split()[0]),),
                provenance=provenance(evidence_id, order),
                confidence=1.0,
                release_class=ReleaseClass.PUBLIC,
            )
        )
    return EvidencePacket(
        packet_id="packet-alpha",
        snapshot_hash=digest("snapshot-alpha"),
        evidence=tuple(records),
        ordered_evidence_ids=tuple(item.evidence_id for item in records),
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        token_count=20,
        created_at=NOW,
        release_class=ReleaseClass.PUBLIC,
    )


def schema() -> LocalContextSchema:
    return LocalContextSchema(
        schema_id="schema-signal",
        contextual_types=(
            LocalTypeDefinition(
                type_id="type-participant",
                label="Signal participant",
                definition="A participant grounded in the signal sequence.",
                parent_upper_type="person",
                abstraction=AbstractionLevel.EVENT_ROLE,
                evidence_ids=("ev-a", "ev-b", "ev-c"),
            ),
        ),
        predicates=(
            LocalPredicateDefinition(
                predicate_id="pred-enabled",
                label="enabled signal for",
                definition="The source supplied an evidenced prerequisite for the target.",
                arity=2,
                domain_type_ids=("type-participant",),
                range_type_ids=("type-participant",),
                parent_upper_relation="causes",
                evidence_ids=("ev-a", "ev-b", "ev-c"),
            ),
        ),
        abstraction=AbstractionLevel.EVENT_ROLE,
    )


def assertion(
    assertion_id: str,
    subject_id: str,
    object_id: str,
    evidence_ids: tuple[str, ...],
    *,
    point: int,
    discourse: int,
) -> QualifiedAssertion:
    packet_by_id = {item.evidence_id: item for item in packet().evidence}
    return QualifiedAssertion(
        assertion_id=assertion_id,
        predicate_id="pred-enabled",
        subject_id=subject_id,
        object_id=object_id,
        temporal_scope=TemporalScope(
            story_time=StoryTime(kind=TemporalKind.POINT, point=point),
            validity_time=ValidityTime(kind=TemporalKind.INTERVAL, start=point, end=point + 1),
            discourse_position=DiscoursePosition(passage_order=discourse),
            revelation_position=RevelationPosition(revelation_order=discourse),
        ),
        narrative_commitment=NarrativeCommitment.WORLD_COMMITTED,
        confidence=0.9,
        evidence_ids=evidence_ids,
        provenance=tuple(packet_by_id[item].provenance for item in evidence_ids),
        contextual_relevance=0.95,
        why_matters="This supported relation links a prerequisite to the signal outcome.",
        why_matters_evidence_ids=evidence_ids,
    )


def entity(
    entity_id: str,
    label: str,
    mention_id: str,
    evidence_ids: tuple[str, ...],
    support_ids: tuple[str, ...],
    *,
    point: int,
) -> Entity:
    return Entity(
        entity_id=entity_id,
        label=label,
        supported_mention_candidate_ids=(mention_id,),
        aliases=(label,),
        contextual_type_id="type-participant",
        contextual_role="participant in the evidence-grounded signal sequence",
        abstraction=AbstractionLevel.EVENT_ROLE,
        temporal_state=StoryTime(kind=TemporalKind.POINT, point=point),
        uncertainty=ExplicitValueState.KNOWN,
        confidence=0.9,
        evidence_ids=evidence_ids,
        description=f"{label} is supported as a signal-sequence participant.",
        description_assertion_ids=support_ids,
    )


def validation(target_id: str) -> ValidationRecord:
    return runtime_structural_acceptance_record(
        validation_id=f"validation-{target_id[:32]}",
        target_id=target_id,
        validated_at=NOW + timedelta(seconds=3),
    )


def graph(*, merged: bool, requalified: bool = False) -> InstanceGraph:
    if merged:
        merged_assertion = assertion(
            "assert-ab-c",
            "entity-ab",
            "entity-c",
            ("ev-a", "ev-b", "ev-c"),
            point=2 if requalified else 1,
            discourse=1,
        )
        entities = (
            Entity(
                entity_id="entity-ab",
                label="Ari and Bex",
                supported_mention_candidate_ids=("m-a", "m-b"),
                aliases=("Ari", "Bex"),
                contextual_type_id="type-participant",
                contextual_role="merged participant requested by evidence anchors",
                abstraction=AbstractionLevel.EVENT_ROLE,
                temporal_state=StoryTime(kind=TemporalKind.POINT, point=1),
                uncertainty=ExplicitValueState.KNOWN,
                confidence=0.85,
                evidence_ids=("ev-a", "ev-b"),
                description="Ari and Bex are represented as one requested participant.",
                description_assertion_ids=(merged_assertion.assertion_id,),
            ),
            entity(
                "entity-c",
                "Cato",
                "m-c",
                ("ev-c",),
                (merged_assertion.assertion_id,),
                point=1,
            ),
        )
        return InstanceGraph(entities=entities, events=(), assertions=(merged_assertion,))

    assertion_a = assertion(
        "assert-a-c",
        "entity-a",
        "entity-c",
        ("ev-a", "ev-c"),
        point=1,
        discourse=1,
    )
    assertion_b = assertion(
        "assert-b-c",
        "entity-b",
        "entity-c",
        ("ev-b", "ev-c"),
        point=9,
        discourse=2,
    )
    return InstanceGraph(
        entities=(
            entity(
                "entity-a",
                "Ari",
                "m-a",
                ("ev-a",),
                (assertion_a.assertion_id,),
                point=1,
            ),
            entity(
                "entity-b",
                "Bex",
                "m-b",
                ("ev-b",),
                (assertion_b.assertion_id,),
                point=9,
            ),
            entity(
                "entity-c",
                "Cato",
                "m-c",
                ("ev-c",),
                (assertion_a.assertion_id, assertion_b.assertion_id),
                point=1,
            ),
        ),
        events=(),
        assertions=(assertion_a, assertion_b),
    )


def upper() -> UpperOntology:
    return UpperOntology(
        ontology_id="upper-v1",
        primitive_types=("person",),
        primitive_relations=("causes",),
        temporal_terms=("story_time", "validity_time"),
        epistemic_terms=("reported",),
        revision="fixture-v1",
    )


def projection(
    condition: ConditionName,
    query_context: QueryContext,
    evidence_packet: EvidencePacket,
    *,
    merged: bool = False,
    split_decision: bool = False,
    decision_offset_seconds: int = 1,
    instance_graph_override: InstanceGraph | None = None,
) -> OntologyProjection:
    instance_graph = instance_graph_override or graph(merged=merged)
    budget = BudgetAccounting(
        nodes_used=len(instance_graph.entities),
        assertions_used=len(instance_graph.assertions),
        display_nodes_used=min(
            len(instance_graph.entities), query_context.budgets.display_node_budget
        ),
        display_assertions_used=min(
            len(instance_graph.assertions), query_context.budgets.display_assertion_budget
        ),
        input_tokens=100,
        output_tokens=100,
    )
    variant = (
        "custom"
        if instance_graph_override is not None
        else ("merged" if merged else ("split" if split_decision else "base"))
    )
    common = {
        "projection_id": (
            f"projection-{condition.value.lower()}-{variant}-{query_context.context_id}"
        ),
        "condition": condition,
        "snapshot_hash": evidence_packet.snapshot_hash,
        "packet_hash": evidence_packet.content_hash,
        "context_hash": query_context.content_hash,
        "query_access_event_hash": digest("ui-query-access"),
        "upper_ontology": upper(),
        "local_schema": schema(),
        "instance_graph": instance_graph,
        "omissions": (),
        "budget_accounting": budget,
        "budgets": query_context.budgets,
        "run_id": f"run-{condition.value.lower()}-{variant}-{query_context.context_id}",
        "release_class": ReleaseClass.PUBLIC,
    }
    if condition is ConditionName.C0_CLASSICAL_PRE:
        validation_target = projection_validation_target_hash(
            condition=condition,
            snapshot_hash=evidence_packet.snapshot_hash,
            packet_hash=evidence_packet.content_hash,
            context_hash=query_context.content_hash,
            upper_ontology=common["upper_ontology"],
            local_schema=common["local_schema"],
            instance_graph=instance_graph,
            decisions=(),
            omissions=(),
            budget_accounting=budget,
            budgets=query_context.budgets,
        )
        records = (validation(validation_target),)
        seal = ConstructionSeal(
            seal_id="seal-c0-shared",
            condition=condition,
            snapshot_hash=evidence_packet.snapshot_hash,
            ontology_hash=digest("fixture-preontology"),
            constructed_at=NOW - timedelta(minutes=10),
            sealed_at=NOW - timedelta(minutes=5),
            sealed_object_ids=("sealed-fixture-object",),
        )
        return OntologyProjection(
            **common,
            decisions=(),
            validation_records=records,
            construction_seal=seal,
        )

    inventory = PreQueryInventory(
        inventory_id="inventory-c2",
        condition=condition,
        snapshot_hash=evidence_packet.snapshot_hash,
        recorded_at=NOW - timedelta(minutes=5),
    )
    object_ids = tuple(item.entity_id for item in instance_graph.entities)
    if merged:
        operator = ConstructionOperator.MERGE
    elif split_decision:
        operator = ConstructionOperator.SPLIT
    else:
        operator = ConstructionOperator.CONTEXTUAL_TYPE
    decision = OntologyDecision(
        decision_id=f"decision-{operator.value}",
        operator=operator,
        evidence_ids=("ev-a", "ev-b"),
        rationale="Fixture construction decision made only after query reveal.",
        decided_at=NOW + timedelta(seconds=decision_offset_seconds),
        input_object_ids=(
            ("entity-a", "entity-b") if merged else (("entity-ab",) if split_decision else ("m-a",))
        ),
        created_object_ids=object_ids,
        removed_object_ids=(
            ("entity-a", "entity-b") if merged else (("entity-ab",) if split_decision else ())
        ),
    )
    normalized_draft_hash = digest(f"ui-normalized-draft-{variant}")
    records = (validation(normalized_draft_hash),)
    generation_fields = {
        "generation_lineage_hash": digest("ui-generation-lineage"),
        "raw_output_artifact_hash": digest("ui-raw-output"),
        "normalized_draft_hash": normalized_draft_hash,
        "validation_bundle_hash": canonical_sha256(records),
    }
    certificate = ConstructionCertificate(
        certificate_id=f"certificate-{'merge' if merged else 'base'}",
        condition=condition,
        snapshot_hash=evidence_packet.snapshot_hash,
        packet_hash=evidence_packet.content_hash,
        query_context_hash=query_context.content_hash,
        query_access_event_hash=digest("ui-query-access"),
        stage_manifest_hash=digest("ui-query-stage"),
        prequery_barrier_hash=digest("ui-prequery-barrier"),
        **generation_fields,
        query_revealed_at=query_context.revealed_at,
        completed_at=NOW + timedelta(seconds=decision_offset_seconds + 1),
        decisions=(decision,),
        pre_query_inventory_hash=inventory.content_hash,
    )
    return OntologyProjection(
        **common,
        **generation_fields,
        decisions=(decision,),
        validation_records=records,
        pre_query_inventory=inventory,
        construction_certificate=certificate,
    )


def c1_projection(
    query_context: QueryContext,
    evidence_packet: EvidencePacket,
) -> OntologyProjection:
    """Construct the production C1 validation shape for renderer regressions."""

    source = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    payload = source.model_dump(mode="python", exclude={"content_hash"})
    projection_id = "projection-c1-aggregate-context-alpha"
    payload.update(
        {
            "projection_id": projection_id,
            "condition": ConditionName.C1_LLM_PRE,
            "generation_lineage_hash": digest("ui-c1-generation-lineage"),
            "raw_output_artifact_hash": digest("ui-c1-raw-output"),
            "normalized_draft_hash": digest("ui-c1-normalized-draft"),
            "construction_seal": ConstructionSeal(
                seal_id="seal-c1-shared",
                condition=ConditionName.C1_LLM_PRE,
                snapshot_hash=evidence_packet.snapshot_hash,
                ontology_hash=digest("fixture-c1-preontology"),
                constructed_at=NOW - timedelta(minutes=10),
                sealed_at=NOW - timedelta(minutes=5),
                sealed_object_ids=("sealed-c1-fixture-object",),
            ),
            "run_id": "run-c1-aggregate-context-alpha",
        }
    )
    validation_target = projection_validation_target_hash(
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=payload["snapshot_hash"],
        packet_hash=payload["packet_hash"],
        context_hash=payload["context_hash"],
        upper_ontology=payload["upper_ontology"],
        local_schema=payload["local_schema"],
        instance_graph=payload["instance_graph"],
        decisions=payload["decisions"],
        omissions=payload["omissions"],
        budget_accounting=payload["budget_accounting"],
        budgets=payload["budgets"],
    )
    records = (validation(validation_target),)
    payload["validation_records"] = records
    payload["validation_bundle_hash"] = canonical_sha256(records)
    return OntologyProjection.model_validate(payload)


def semantic_overlay(
    source: OntologyProjection,
    query_context: QueryContext,
    evidence_packet: EvidencePacket,
    *,
    statuses: dict[
        str,
        tuple[SemanticSupportStatus, SemanticSupportStatus],
    ]
    | None = None,
    supported_evidence: dict[str, tuple[str, ...]] | None = None,
) -> VisualizationSemanticOverlay:
    status_by_id = statuses or {
        item.assertion_id: (
            SemanticSupportStatus.SUPPORTED,
            SemanticSupportStatus.SUPPORTED,
        )
        for item in source.instance_graph.assertions
    }
    evidence_by_id = supported_evidence or {}
    rows = []
    for item in sorted(source.instance_graph.assertions, key=lambda row: row.assertion_id):
        assertion_status, description_status = status_by_id[item.assertion_id]
        default_evidence = (
            tuple(sorted(item.why_matters_evidence_ids))
            if description_status is SemanticSupportStatus.SUPPORTED
            else ()
        )
        rows.append(
            VisualizationAssertionSemanticAssessment(
                projection_assertion_id=item.assertion_id,
                projection_assertion_hash=item.content_hash,
                assertion_support_status=assertion_status,
                description_support_status=description_status,
                supported_description_evidence_ids=evidence_by_id.get(
                    item.assertion_id, default_evidence
                ),
            )
        )
    return VisualizationSemanticOverlay(
        overlay_id=f"overlay-{source.projection_id}",
        projection_hash=source.content_hash,
        snapshot_hash=source.snapshot_hash,
        packet_hash=evidence_packet.content_hash,
        context_hash=query_context.content_hash,
        source_kind=SemanticAssessmentSourceKind.SCORER,
        source_artifact_hash=digest(f"scorer-artifact-{source.projection_id}"),
        assessment_revision="semantic-overlay-fixture-v1",
        assertion_assessments=tuple(rows),
        assessed_at=NOW + timedelta(minutes=1),
        release_class=ReleaseClass.PUBLIC,
    )


def feedback_anchor(evidence_id: str, mention_id: str) -> FeedbackAnchor:
    return FeedbackAnchor(
        evidence_ids=(evidence_id,),
        mention_candidate_ids=(mention_id,),
        requested_semantic_signature=f"identity anchored by {mention_id}",
    )


def test_compiler_exposes_rich_grounded_details_and_fixed_anchors() -> None:
    query_context = context()
    evidence_packet = packet()
    c0 = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    bundle = build_visualization_bundle(
        c0,
        query_context,
        evidence_packet,
        include_public_evidence_text=True,
    )

    assert len(bundle.state.nodes) == 3
    assert len(bundle.state.assertions) == 2
    assert bundle.state.semantic_hash == c0.content_hash
    assert all(item.evidence_badge.available for item in bundle.state.assertions)
    assert all(item.why_matters for item in bundle.state.assertions)
    assert all(
        item.contextual_relevance == pytest.approx(0.95) for item in bundle.assertion_details
    )
    assert all(item.why_matters_evidence_ids for item in bundle.assertion_details)
    assert {item.contextual_type_label for item in bundle.node_details} == {"Signal participant"}
    assert all(item.public_text for item in bundle.evidence_metadata)

    c2 = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    c2_bundle = build_visualization_bundle(c2, query_context, evidence_packet)
    c0_positions = {item.visualization_node_id: (item.x, item.y) for item in bundle.state.positions}
    c2_positions = {
        item.visualization_node_id: (item.x, item.y) for item in c2_bundle.state.positions
    }
    assert c0_positions == c2_positions


def test_compiler_renders_full_structural_output_with_semantic_assessment_pending() -> None:
    query_context = context()
    evidence_packet = packet()
    projections = (
        projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet),
        c1_projection(query_context, evidence_packet),
        projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet),
    )
    for source in projections:
        bundle = build_visualization_bundle(source, query_context, evidence_packet)
        assert len(bundle.state.nodes) == 3
        assert len(bundle.state.assertions) == 2
        assert bundle.semantic_assessment_status == "pending_scorer_or_reviewer"
        assert _structurally_accepted_assertion_ids(source) == frozenset(
            {"assert-a-c", "assert-b-c"}
        )


def test_registered_display_is_exact_deterministic_and_endpoint_closed() -> None:
    evidence_packet = packet()
    base_context = context()
    context_payload = base_context.model_dump(
        mode="python", exclude={"content_hash", "budgets"}
    )
    context_payload["budgets"] = OutputBudgets(
        node_budget=10,
        assertion_budget=12,
        display_node_budget=2,
        display_assertion_budget=1,
    )
    limited_context = QueryContext.model_validate(context_payload)
    source = projection(ConditionName.C0_CLASSICAL_PRE, limited_context, evidence_packet)

    raw = build_visualization_bundle(source, limited_context, evidence_packet)
    registered = build_visualization_bundle(
        source,
        limited_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
    )
    replay = build_visualization_bundle(
        source,
        limited_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
    )

    assert len(raw.state.nodes) == 3
    assert len(raw.state.assertions) == 2
    assert len(registered.state.nodes) == source.budget_accounting.display_nodes_used == 2
    assert (
        len(registered.state.assertions)
        == source.budget_accounting.display_assertions_used
        == 1
    )
    assert registered == replay
    assert registered.display_selection.content_scope is (
        VisualizationContentScope.REGISTERED_DISPLAY
    )
    selected_node_ids = {
        item.projection_object_id for item in registered.state.nodes
    }
    selected_assertion = registered.state.assertions[0]
    assert selected_assertion.source_visualization_node_id is not None
    visual_to_projection = {
        item.visualization_node_id: item.projection_object_id
        for item in registered.state.nodes
    }
    assert {
        visual_to_projection[selected_assertion.source_visualization_node_id],
        visual_to_projection[selected_assertion.target_visualization_node_id],
    } == selected_node_ids


def test_registered_display_fails_closed_when_budget_cannot_preserve_endpoints() -> None:
    evidence_packet = packet()
    base_context = context()
    context_payload = base_context.model_dump(
        mode="python", exclude={"content_hash", "budgets"}
    )
    context_payload["budgets"] = OutputBudgets(
        node_budget=10,
        assertion_budget=12,
        display_node_budget=1,
        display_assertion_budget=1,
    )
    impossible_context = QueryContext.model_validate(context_payload)
    source = projection(ConditionName.C0_CLASSICAL_PRE, impossible_context, evidence_packet)

    with pytest.raises(VisualizationCompilationError, match="dependency closure"):
        build_visualization_bundle(
            source,
            impossible_context,
            evidence_packet,
            content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        )


def test_registered_display_treats_epistemic_holder_as_required_dependency() -> None:
    evidence_packet = packet()
    attributed = assertion(
        "assert-attributed-a-c",
        "entity-a",
        "entity-c",
        ("ev-a", "ev-b", "ev-c"),
        point=1,
        discourse=1,
    )
    attributed_payload = attributed.model_dump(mode="python", exclude={"content_hash"})
    proposition = PropositionContent(
        proposition_content_id="proposition-a-c",
        predicate_id=attributed.predicate_id,
        subject_id=attributed.subject_id,
        object_id=attributed.object_id,
        temporal_content=attributed.temporal_scope,
        evidence_ids=attributed.evidence_ids,
    )
    attributed_payload.update(
        {
            "proposition_content_id": proposition.proposition_content_id,
            "epistemic_scope": EpistemicScope(
                holder_id="entity-b",
                attitude=EpistemicAttitude.REPORTED,
                proposition_content_id=proposition.proposition_content_id,
                holder_relative_time=HolderRelativeTime(
                    kind=TemporalKind.POINT,
                    point=1,
                ),
                evidence_ids=("ev-b",),
            ),
            "narrative_commitment": NarrativeCommitment.HOLDER_ATTRIBUTED,
        }
    )
    attributed = QualifiedAssertion.model_validate(attributed_payload)
    attributed_graph = InstanceGraph(
        entities=(
            entity(
                "entity-a",
                "Ari",
                "m-a",
                ("ev-a",),
                (attributed.assertion_id,),
                point=1,
            ),
            entity(
                "entity-b",
                "Bex",
                "m-b",
                ("ev-b",),
                (attributed.assertion_id,),
                point=1,
            ),
            entity(
                "entity-c",
                "Cato",
                "m-c",
                ("ev-c",),
                (attributed.assertion_id,),
                point=1,
            ),
        ),
        events=(),
        proposition_contents=(proposition,),
        assertions=(attributed,),
    )
    base_context = context()
    context_payload = base_context.model_dump(
        mode="python", exclude={"content_hash", "budgets"}
    )
    context_payload["budgets"] = OutputBudgets(
        node_budget=10,
        assertion_budget=12,
        display_node_budget=2,
        display_assertion_budget=1,
    )
    insufficient_context = QueryContext.model_validate(context_payload)
    insufficient = projection(
        ConditionName.C0_CLASSICAL_PRE,
        insufficient_context,
        evidence_packet,
        instance_graph_override=attributed_graph,
    )
    with pytest.raises(VisualizationCompilationError, match="dependency closure"):
        build_visualization_bundle(
            insufficient,
            insufficient_context,
            evidence_packet,
            content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        )

    context_payload["budgets"] = OutputBudgets(
        node_budget=10,
        assertion_budget=12,
        display_node_budget=3,
        display_assertion_budget=1,
    )
    sufficient_context = QueryContext.model_validate(context_payload)
    sufficient = projection(
        ConditionName.C0_CLASSICAL_PRE,
        sufficient_context,
        evidence_packet,
        instance_graph_override=attributed_graph,
    )
    bundle = build_visualization_bundle(
        sufficient,
        sufficient_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
    )
    projection_by_visual = {
        item.visualization_node_id: item.projection_object_id for item in bundle.state.nodes
    }
    assertion_view = bundle.state.assertions[0]
    assert assertion_view.epistemic_holder_id in projection_by_visual
    assert projection_by_visual[assertion_view.epistemic_holder_id] == "entity-b"
    assert set(bundle.display_selection.projection_node_ids) == {
        "entity-a",
        "entity-b",
        "entity-c",
    }
    supported = build_visualization_bundle(
        sufficient,
        sufficient_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        semantic_overlay=semantic_overlay(
            sufficient,
            sufficient_context,
            evidence_packet,
        ),
        semantic_display_mode=SemanticDisplayMode.SUPPORTED_ONLY,
    )
    assert {
        item.projection_object_id for item in supported.state.nodes
    } == {"entity-a", "entity-b", "entity-c"}


def test_verified_overlay_exposes_verdicts_without_changing_raw_projection_content() -> None:
    query_context = context()
    evidence_packet = packet()
    source = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    assessment = semantic_overlay(
        source,
        query_context,
        evidence_packet,
        statuses={
            "assert-a-c": (
                SemanticSupportStatus.SUPPORTED,
                SemanticSupportStatus.SUPPORTED,
            ),
            "assert-b-c": (
                SemanticSupportStatus.UNSUPPORTED,
                SemanticSupportStatus.UNSUPPORTED,
            ),
        },
    )
    pending = build_visualization_bundle(source, query_context, evidence_packet)
    verified = build_visualization_bundle(
        source,
        query_context,
        evidence_packet,
        semantic_overlay=assessment,
    )

    assert verified.semantic_assessment_status == "verified_scorer_or_reviewer"
    assert verified.semantic_overlay == assessment
    assert verified.projection_hash == pending.projection_hash == source.content_hash
    assert [item.model_dump(exclude={"content_hash"}) for item in verified.state.nodes] == [
        item.model_dump(exclude={"content_hash"}) for item in pending.state.nodes
    ]
    assert [
        item.model_dump(exclude={"content_hash"}) for item in verified.state.assertions
    ] == [item.model_dump(exclude={"content_hash"}) for item in pending.state.assertions]
    assert {
        item.projection_assertion_id: item.assertion_support_status
        for item in verified.assertion_details
    } == {"assert-a-c": "supported", "assert-b-c": "unsupported"}


def test_supported_display_omits_unsupported_and_withholds_unverified_prose() -> None:
    query_context = context()
    evidence_packet = packet()
    source = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    assessment = semantic_overlay(
        source,
        query_context,
        evidence_packet,
        statuses={
            "assert-a-c": (
                SemanticSupportStatus.SUPPORTED,
                SemanticSupportStatus.INSUFFICIENT,
            ),
            "assert-b-c": (
                SemanticSupportStatus.UNSUPPORTED,
                SemanticSupportStatus.UNSUPPORTED,
            ),
        },
    )
    supported = build_visualization_bundle(
        source,
        query_context,
        evidence_packet,
        semantic_overlay=assessment,
        semantic_display_mode=SemanticDisplayMode.SUPPORTED_ONLY,
    )

    assert [
        item.projection_assertion_id for item in supported.state.assertions
    ] == ["assert-a-c"]
    assert {
        item.projection_object_id for item in supported.state.nodes
    } == {"entity-a", "entity-c"}
    assert all(item.description == NODE_DESCRIPTION_WITHHELD for item in supported.state.nodes)
    assert supported.state.assertions[0].why_matters == WHY_MATTERS_WITHHELD
    assert supported.assertion_details[0].why_matters_evidence_ids == ()
    assert supported.assertion_details[0].assertion_support_status == "supported"
    assert supported.assertion_details[0].description_support_status == "insufficient"


def test_semantic_overlay_is_exhaustive_and_hash_bound() -> None:
    query_context = context()
    evidence_packet = packet()
    source = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    complete = semantic_overlay(source, query_context, evidence_packet)

    missing_payload = complete.model_dump(mode="python", exclude={"content_hash"})
    missing_payload["assertion_assessments"] = complete.assertion_assessments[:1]
    missing = VisualizationSemanticOverlay.model_validate(missing_payload)
    with pytest.raises(VisualizationCompilationError, match="every and only"):
        build_visualization_bundle(
            source, query_context, evidence_packet, semantic_overlay=missing
        )

    wrong_projection_payload = complete.model_dump(
        mode="python", exclude={"content_hash"}
    )
    wrong_projection_payload["projection_hash"] = digest("another-projection")
    wrong_projection = VisualizationSemanticOverlay.model_validate(
        wrong_projection_payload
    )
    with pytest.raises(VisualizationCompilationError, match="another projection"):
        build_visualization_bundle(
            source, query_context, evidence_packet, semantic_overlay=wrong_projection
        )

    first = complete.assertion_assessments[0]
    wrong_row_payload = first.model_dump(mode="python", exclude={"content_hash"})
    wrong_row_payload["projection_assertion_hash"] = digest("another-assertion")
    wrong_row = VisualizationAssertionSemanticAssessment.model_validate(wrong_row_payload)
    wrong_assertion_payload = complete.model_dump(
        mode="python", exclude={"content_hash"}
    )
    wrong_assertion_payload["assertion_assessments"] = (
        wrong_row,
        *complete.assertion_assessments[1:],
    )
    wrong_assertion = VisualizationSemanticOverlay.model_validate(
        wrong_assertion_payload
    )
    with pytest.raises(VisualizationCompilationError, match="assertion hash differs"):
        build_visualization_bundle(
            source, query_context, evidence_packet, semantic_overlay=wrong_assertion
        )

    outside_lineage = semantic_overlay(
        source,
        query_context,
        evidence_packet,
        supported_evidence={"assert-a-c": ("ev-b",)},
    )
    with pytest.raises(VisualizationCompilationError, match="projection assertion"):
        build_visualization_bundle(
            source, query_context, evidence_packet, semantic_overlay=outside_lineage
        )


def test_static_interface_persistently_discloses_pending_semantic_assessment() -> None:
    ui_root = Path(__file__).resolve().parents[2] / "ui"
    html = (ui_root / "index.html").read_text(encoding="utf-8")
    script = (ui_root / "app.js").read_text(encoding="utf-8")

    assert 'id="semantic-assessment-status"' in html
    assert "Semantic support: pending scorer or reviewer assessment" in html
    assert 'currentBundle.semantic_assessment_status === "pending_scorer_or_reviewer"' in script
    assert '"verified_scorer_or_reviewer"' in script
    assert "graph shows structurally accepted output in its declared content scope" in script
    assert "Projection-authored node description (not verified)" in script
    assert "Projection-claimed description evidence (not verified)" in script
    assert 'detail.description_support_status === "supported"' in script


def test_story_and_spoiler_filters_change_visibility_not_semantics() -> None:
    query_context = context()
    evidence_packet = packet()
    c0 = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    bundle = build_visualization_bundle(c0, query_context, evidence_packet)
    filtered = filter_visualization_bundle(
        bundle,
        VisualizationTemporalFilter(
            story_scope=StoryTime(kind=TemporalKind.POINT, point=1),
            spoiler_horizon=SpoilerHorizon(
                horizon_id="ui-horizon-one",
                max_discourse_position=DiscoursePosition(passage_order=1),
                max_revelation_position=RevelationPosition(revelation_order=1),
            ),
        ),
    )
    assert filtered.state.semantic_hash == bundle.state.semantic_hash
    assert filtered.projection_hash == bundle.projection_hash
    assert filtered.semantic_assessment_status == "pending_scorer_or_reviewer"
    assert filtered.state.content_hash != bundle.state.content_hash
    assert len(filtered.state.visible_node_ids) == 1
    assert filtered.state.visible_assertion_ids == ()


def test_diff_identifies_merge_and_preserves_unmoved_anchors() -> None:
    query_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
    )
    after_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
    )
    before = build_visualization_bundle(before_projection, query_context, evidence_packet)
    after = build_visualization_bundle(after_projection, query_context, evidence_packet)
    result = compare_visualizations(before, after)
    merge = [item for item in result.changes if item.kind is VisualizationChangeKind.MERGED]
    assert len(merge) == 1
    assert len(merge[0].before_visualization_ids) == 2
    assert len(merge[0].after_visualization_ids) == 1
    assert result.fixed_anchor_positions_preserved is True
    assert result.shared_fixed_anchor_ids


def test_diff_marks_contextual_relevance_only_change_as_requalified() -> None:
    query_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        query_context,
        evidence_packet,
    )
    graph_payload = before_projection.instance_graph.model_dump(
        mode="python",
        exclude={
            "content_hash": True,
            "assertions": {"__all__": {"content_hash"}},
        },
    )
    graph_payload["assertions"][0]["contextual_relevance"] = 0.35
    changed_graph = InstanceGraph(**graph_payload)
    projection_payload = before_projection.model_dump(
        mode="python",
        exclude={"content_hash", "instance_graph"},
    )
    validation_target = projection_validation_target_hash(
        condition=projection_payload["condition"],
        snapshot_hash=projection_payload["snapshot_hash"],
        packet_hash=projection_payload["packet_hash"],
        context_hash=projection_payload["context_hash"],
        upper_ontology=projection_payload["upper_ontology"],
        local_schema=projection_payload["local_schema"],
        instance_graph=changed_graph,
        decisions=projection_payload["decisions"],
        omissions=projection_payload["omissions"],
        budget_accounting=projection_payload["budget_accounting"],
        budgets=projection_payload["budgets"],
    )
    projection_payload["validation_records"] = (validation(validation_target),)
    after_projection = OntologyProjection(
        **projection_payload,
        instance_graph=changed_graph,
    )
    result = compare_visualizations(
        build_visualization_bundle(before_projection, query_context, evidence_packet),
        build_visualization_bundle(after_projection, query_context, evidence_packet),
    )
    assert any(
        item.kind is VisualizationChangeKind.REQUALIFIED and item.object_kind == "assertion"
        for item in result.changes
    )


def test_refine_context_instruction_replays_exact_context_hash() -> None:
    before = context()
    created_at = NOW + timedelta(minutes=1)
    intent = RefineContextIntent(
        after_context_id="context-alpha-revised",
        after_revealed_at=created_at,
        lens="epistemic responsibility",
        story_scope=StoryTime(kind=TemporalKind.POINT, point=1),
    )
    instruction = build_refine_context_instruction(
        revision_id="revision-refine",
        before_context=before,
        intent=intent,
        anchors=(feedback_anchor("ev-a", "m-a"),),
        rationale="Focus the graph on what was known at the first signal.",
        sequence=1,
        created_at=created_at,
    )
    replayed = apply_context_refinement(before, intent)
    assert replayed.content_hash == instruction.revision.after_context_hash
    assert instruction.model_visible().revision == instruction.revision
    assert "epistemic responsibility" in instruction.revision.requested_change


def test_merge_split_anchors_are_packet_scoped_and_projection_independent() -> None:
    query_context = context()
    evidence_packet = packet()
    anchors = (feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b"))
    instruction = build_merge_split_instruction(
        revision_id="revision-merge",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=anchors,
        rationale="The two evidence-anchored mentions refer to one requested identity.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    assert_revision_anchors_resolve_in_packet(instruction, evidence_packet)
    before = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    assert_revision_has_no_projection_local_anchors(instruction, (before,))

    bad_payload = instruction.revision.anchors[0].model_dump(exclude={"content_hash"})
    bad_payload["mention_candidate_ids"] = ("entity-a",)
    bad_anchor = FeedbackAnchor(**bad_payload)
    with pytest.raises(ValueError, match="outside the frozen packet"):
        bad_instruction = build_merge_split_instruction(
            revision_id="revision-local-leak",
            context=query_context,
            intent=MergeSplitIntent(
                operation=MergeSplitOperation.MERGE,
                grouped_mention_candidate_ids=(("entity-a",), ("m-b",)),
            ),
            anchors=(bad_anchor, anchors[1]),
            rationale="Invalid local identifier leak.",
            sequence=2,
            created_at=NOW + timedelta(minutes=2),
        )
        assert_revision_anchors_resolve_in_packet(bad_instruction, evidence_packet)


def test_per_condition_resolution_records_capability_and_c2_result() -> None:
    query_context = context()
    evidence_packet = packet()
    anchors = (feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b"))
    instruction = build_merge_split_instruction(
        revision_id="revision-merge",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=anchors,
        rationale="Merge the two anchored identities.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    c0 = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    limited = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C0_CLASSICAL_PRE,
        before_projection=c0,
        after_projection=None,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=2),
    )
    assert limited.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
    assert limited.after_projection_hash is None

    c2_before = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    c2_after = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
        decision_offset_seconds=65,
    )
    resolved = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C2_LLM_QUERY,
        before_projection=c2_before,
        after_projection=c2_after,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=2),
    )
    assert resolved.status is FeedbackResolutionStatus.RESOLVED
    assert set(resolved.resolved_object_ids) == {"entity-a", "entity-b"}


def test_c2_feedback_rejects_an_output_constructed_before_the_revision() -> None:
    query_context = context()
    evidence_packet = packet()
    instruction = build_merge_split_instruction(
        revision_id="revision-no-stale-output",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=(feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b")),
        rationale="Require a genuine post-revision reconstruction.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    before = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    stale_after = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
    )
    with pytest.raises(ValueError, match="not reconstructed after the revision"):
        resolve_feedback_for_condition(
            instruction=instruction,
            packet=evidence_packet,
            receiving_condition=ConditionName.C2_LLM_QUERY,
            before_projection=before,
            after_projection=stale_after,
            resolver_hash=digest("resolver-v1"),
            seed=0,
            resolved_at=NOW + timedelta(minutes=2),
        )


def test_capability_limited_revision_execution_has_no_fabricated_output() -> None:
    query_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        query_context,
        evidence_packet,
    )
    before_bundle = build_visualization_bundle(
        before_projection,
        query_context,
        evidence_packet,
    )
    instruction = build_merge_split_instruction(
        revision_id="revision-c0-limited",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=(feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b")),
        rationale="Ask the sealed condition for an unsupported identity change.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    resolution = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C0_CLASSICAL_PRE,
        before_projection=before_projection,
        after_projection=None,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=1, seconds=1),
    )
    result = RevisionExecutionResult(
        instruction=instruction,
        resolution=resolution,
        before_bundle_hash=before_bundle.content_hash,
        latency_seconds=0.01,
    )
    assert result.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
    assert result.after_bundle is None
    assert result.call is None


def test_split_resolution_requires_split_certificate_and_separate_groups() -> None:
    query_context = context()
    evidence_packet = packet()
    anchors = (feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b"))
    instruction = build_merge_split_instruction(
        revision_id="revision-split",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.SPLIT,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=anchors,
        rationale="Separate the two mention groups.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    before = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
    )
    without_split_certificate = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        decision_offset_seconds=65,
    )
    invalid = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C2_LLM_QUERY,
        before_projection=before,
        after_projection=without_split_certificate,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=2),
    )
    assert invalid.status is FeedbackResolutionStatus.INVALID

    after = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        split_decision=True,
        decision_offset_seconds=65,
    )
    resolved = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C2_LLM_QUERY,
        before_projection=before,
        after_projection=after,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=2),
    )
    assert resolved.status is FeedbackResolutionStatus.RESOLVED
    assert resolved.resolved_object_ids == ("entity-ab",)


def test_refine_context_can_reproject_the_same_c0_seal() -> None:
    before_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        before_context,
        evidence_packet,
    )
    created_at = NOW + timedelta(minutes=1)
    intent = RefineContextIntent(
        after_context_id="context-alpha-refined",
        after_revealed_at=created_at,
        lens="identity evidence",
    )
    instruction = build_refine_context_instruction(
        revision_id="revision-refine-c0",
        before_context=before_context,
        intent=intent,
        anchors=(feedback_anchor("ev-a", "m-a"),),
        rationale="Reproject the sealed ontology for an identity lens.",
        sequence=1,
        created_at=created_at,
    )
    after_context = instruction.after_context
    after_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        after_context,
        evidence_packet,
    )
    result = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C0_CLASSICAL_PRE,
        before_projection=before_projection,
        after_projection=after_projection,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=created_at + timedelta(seconds=1),
    )
    assert result.status is FeedbackResolutionStatus.RESOLVED
    assert (
        before_projection.construction_seal.content_hash
        == after_projection.construction_seal.content_hash
    )


def test_replay_recomputes_diff_and_reports_hash_mismatch() -> None:
    query_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
    )
    after_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
        decision_offset_seconds=65,
    )
    anchors = (feedback_anchor("ev-a", "m-a"), feedback_anchor("ev-b", "m-b"))
    instruction = build_merge_split_instruction(
        revision_id="revision-replay",
        context=query_context,
        intent=MergeSplitIntent(
            operation=MergeSplitOperation.MERGE,
            grouped_mention_candidate_ids=(("m-a",), ("m-b",)),
        ),
        anchors=anchors,
        rationale="Replay the merge exactly.",
        sequence=1,
        created_at=NOW + timedelta(minutes=1),
    )
    resolution = resolve_feedback_for_condition(
        instruction=instruction,
        packet=evidence_packet,
        receiving_condition=ConditionName.C2_LLM_QUERY,
        before_projection=before_projection,
        after_projection=after_projection,
        resolver_hash=digest("resolver-v1"),
        seed=0,
        resolved_at=NOW + timedelta(minutes=2),
    )
    before_bundle = build_visualization_bundle(
        before_projection,
        query_context,
        evidence_packet,
    )
    after_bundle = build_visualization_bundle(
        after_projection,
        query_context,
        evidence_packet,
    )
    diff = compare_visualizations(before_bundle, after_bundle)
    expectation = FeedbackReplayExpectation(
        instruction_hash=instruction.content_hash,
        resolution_hash=resolution.content_hash,
        before_bundle_hash=before_bundle.content_hash,
        after_bundle_hash=after_bundle.content_hash,
        diff_hash=diff.content_hash,
    )
    replay = verify_feedback_replay(
        expectation=expectation,
        instruction=instruction,
        resolution=resolution,
        before_bundle=before_bundle,
        after_bundle=after_bundle,
        diff=diff,
        checked_at=NOW + timedelta(minutes=3),
    )
    assert replay.replay_hash_success is True
    assert replay.diagnostics == ()

    bad_expectation = FeedbackReplayExpectation(
        **{
            **expectation.model_dump(mode="python", exclude={"content_hash"}),
            "diff_hash": digest("wrong-diff"),
        }
    )
    failed = verify_feedback_replay(
        expectation=bad_expectation,
        instruction=instruction,
        resolution=resolution,
        before_bundle=before_bundle,
        after_bundle=after_bundle,
        diff=diff,
        checked_at=NOW + timedelta(minutes=3),
    )
    assert failed.replay_hash_success is False
    assert "diff_hash" in failed.diagnostics[0]


def test_feedback_plan_is_exactly_six_balanced_scripts_and_three_traces() -> None:
    anchor_hash = digest("condition-independent-anchor")
    episodes = []
    for index in range(6):
        episodes.append(
            FeedbackEpisodePlan(
                episode_id=f"script-{index}",
                kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
                action=(
                    FeedbackAction.REFINE_CONTEXT
                    if index < 3
                    else FeedbackAction.REQUEST_MERGE_SPLIT
                ),
                context_id=f"context-{index}",
                seed=0,
                anchor_hashes=(anchor_hash,),
                known_answer_target_ids=(f"target-{index}",),
            )
        )
    for index in range(3):
        episodes.append(
            FeedbackEpisodePlan(
                episode_id=f"trace-{index}",
                kind=FeedbackEpisodeKind.RESEARCHER_TRACE,
                action=FeedbackAction.REFINE_CONTEXT,
                context_id=f"trace-context-{index}",
                seed=0,
                anchor_hashes=(anchor_hash,),
            )
        )
    plan = FeedbackStudyPlan(episodes=tuple(episodes))
    assert len(plan.episodes) == 9

    bad_trace = episodes[-1].model_dump(mode="python", exclude={"content_hash"})
    bad_trace["known_answer_target_ids"] = ("gold-leak",)
    with pytest.raises(ValidationError, match="gold-dependent targets NA"):
        FeedbackEpisodePlan(**bad_trace)
