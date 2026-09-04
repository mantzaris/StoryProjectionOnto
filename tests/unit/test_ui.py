from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import (
    AbstractionLevel,
    BudgetAccounting,
    CommitmentCheckStatus,
    ConditionName,
    ConstructionCertificate,
    ConstructionOperator,
    ConstructionSeal,
    DiscoursePosition,
    Entity,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSupportStatus,
    ExplicitValueState,
    FeedbackAction,
    FeedbackAnchor,
    FeedbackResolutionStatus,
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
    ProvenanceReference,
    QualifiedAssertion,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RevelationPosition,
    SpoilerHorizon,
    StoryTime,
    TemporalDeterminationStatus,
    TemporalKind,
    TemporalScope,
    UpperOntology,
    ValidationRecord,
    ValidationStatus,
    ValidityTime,
    VisualizationTemporalFilter,
)
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    FeedbackEpisodePlan,
    FeedbackReplayExpectation,
    FeedbackStudyPlan,
    MergeSplitIntent,
    MergeSplitOperation,
    RefineContextIntent,
    VisualizationChangeKind,
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


def validation(assertion_id: str, *, supported: bool = True) -> ValidationRecord:
    return ValidationRecord(
        validation_id=f"validation-{assertion_id}",
        target_id=assertion_id,
        validation_status=ValidationStatus.ACCEPTED,
        evidence_support_status=(
            EvidenceSupportStatus.SUPPORTED if supported else EvidenceSupportStatus.UNSUPPORTED
        ),
        temporal_status=TemporalDeterminationStatus.VALID,
        commitment_status=CommitmentCheckStatus.VALID,
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
) -> OntologyProjection:
    instance_graph = graph(merged=merged)
    assertion_ids = tuple(item.assertion_id for item in instance_graph.assertions)
    budget = BudgetAccounting(
        nodes_used=len(instance_graph.entities),
        assertions_used=len(instance_graph.assertions),
        display_nodes_used=len(instance_graph.entities),
        display_assertions_used=len(instance_graph.assertions),
        input_tokens=100,
        output_tokens=100,
    )
    variant = "merged" if merged else ("split" if split_decision else "base")
    common = {
        "projection_id": (
            f"projection-{condition.value.lower()}-{variant}-{query_context.context_id}"
        ),
        "condition": condition,
        "snapshot_hash": evidence_packet.snapshot_hash,
        "packet_hash": evidence_packet.content_hash,
        "context_hash": query_context.content_hash,
        "upper_ontology": upper(),
        "local_schema": schema(),
        "instance_graph": instance_graph,
        "omissions": (),
        "validation_records": tuple(validation(item) for item in assertion_ids),
        "budget_accounting": budget,
        "budgets": query_context.budgets,
        "run_id": f"run-{condition.value.lower()}-{variant}-{query_context.context_id}",
        "release_class": ReleaseClass.PUBLIC,
    }
    if condition is ConditionName.C0_CLASSICAL_PRE:
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
        decided_at=NOW + timedelta(seconds=1),
        input_object_ids=(
            ("entity-a", "entity-b") if merged else (("entity-ab",) if split_decision else ("m-a",))
        ),
        created_object_ids=object_ids,
        removed_object_ids=(
            ("entity-a", "entity-b") if merged else (("entity-ab",) if split_decision else ())
        ),
    )
    certificate = ConstructionCertificate(
        certificate_id=f"certificate-{'merge' if merged else 'base'}",
        condition=condition,
        snapshot_hash=evidence_packet.snapshot_hash,
        packet_hash=evidence_packet.content_hash,
        query_context_hash=query_context.content_hash,
        query_revealed_at=query_context.revealed_at,
        completed_at=NOW + timedelta(seconds=2),
        decisions=(decision,),
        pre_query_inventory_hash=inventory.content_hash,
    )
    return OntologyProjection(
        **common,
        decisions=(decision,),
        pre_query_inventory=inventory,
        construction_certificate=certificate,
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
    assert {item.contextual_type_label for item in bundle.node_details} == {"Signal participant"}
    assert all(item.public_text for item in bundle.evidence_metadata)

    c2 = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    c2_bundle = build_visualization_bundle(c2, query_context, evidence_packet)
    c0_positions = {item.visualization_node_id: (item.x, item.y) for item in bundle.state.positions}
    c2_positions = {
        item.visualization_node_id: (item.x, item.y) for item in c2_bundle.state.positions
    }
    assert c0_positions == c2_positions


def test_compiler_never_displays_unsupported_assertions_or_descriptions() -> None:
    query_context = context()
    evidence_packet = packet()
    c0 = projection(ConditionName.C0_CLASSICAL_PRE, query_context, evidence_packet)
    payload = c0.model_dump(mode="python", exclude={"content_hash"})
    payload["validation_records"] = (
        validation("assert-a-c", supported=True),
        validation("assert-b-c", supported=False),
    )
    changed = OntologyProjection(**payload)
    bundle = build_visualization_bundle(changed, query_context, evidence_packet)
    # The otherwise supported assertion is also hidden because its target node's
    # description depends on the unsupported assertion. Rendering no orphan edge is safer
    # than silently inventing a replacement description.
    assert bundle.state.assertions == ()
    assert {item.projection_object_id for item in bundle.state.nodes} == {"entity-a"}


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
