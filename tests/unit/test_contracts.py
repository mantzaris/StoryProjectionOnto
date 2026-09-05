from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import story_projection_onto.contracts as contracts_module
from story_projection_onto.contracts import (
    MODEL_VISIBLE_GENERIC_CONTEXT_FIELDS,
    RUNTIME_STRUCTURAL_ONLY_DIAGNOSTIC,
    AbstractionLevel,
    ArtifactHashReference,
    BenchmarkSplit,
    BudgetAccounting,
    CommitmentCheckStatus,
    ConditionName,
    ConstructionCapabilities,
    ConstructionCertificate,
    ConstructionOperator,
    DiscoursePosition,
    EpistemicAttitude,
    EpistemicScope,
    EvidenceBadge,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSupportStatus,
    ExplicitValueState,
    FailureLineage,
    GoldAdjudicationStatus,
    GoldAlternativeSet,
    GoldAssertionAnnotation,
    GoldContextualProjection,
    GoldEntityCluster,
    GoldRelevanceAnnotation,
    GoldReviewStatus,
    GoldTargetKind,
    HolderRelativeTime,
    InstanceGraph,
    LocalContextSchema,
    ModelVisibleGenericConstructionContext,
    ModelVisibleQueryContext,
    NarrativeCommitment,
    NodePosition,
    OntologyDecision,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    ParentProjectionRef,
    PreQueryInventory,
    ProvenanceReference,
    QualifiedAssertion,
    QueryAccessEvent,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RevelationPosition,
    RunManifest,
    RunOutcome,
    RunResourceUsage,
    RunTiming,
    SemanticAssessmentScope,
    SpoilerHorizon,
    StoryTime,
    TemporalDeterminationStatus,
    TemporalKind,
    TemporalScope,
    UpperOntology,
    ValidatedGeneration,
    ValidationRecord,
    ValidationStatus,
    ValidityTime,
    Viewport,
    VisualizationNode,
    VisualizationState,
    VisualizationTemporalFilter,
    assert_evidence_boundary,
    assert_public_release,
    canonical_json,
    canonical_sha256,
    normalize_generation_metadata,
    runtime_structural_acceptance_record,
    to_model_visible_context_for_condition,
    to_model_visible_packet,
    to_model_visible_query,
)

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def provenance(evidence_id: str = "ev-1") -> ProvenanceReference:
    return ProvenanceReference(
        provenance_id=f"prov-{evidence_id}",
        evidence_id=evidence_id,
        extraction_method="hand-authored fixture",
        locator=f"fixture:{evidence_id}",
        source_artifact_hash=digest("fixture-source"),
        confidence=1.0,
    )


def evidence_record(
    evidence_id: str = "ev-1", release_class: ReleaseClass = ReleaseClass.PUBLIC
) -> EvidenceRecord:
    text = "Mira reports that the bridge is closed."
    return EvidenceRecord(
        evidence_id=evidence_id,
        passage_id="passage-1",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        provenance=provenance(evidence_id),
        confidence=0.98,
        release_class=release_class,
    )


def horizon() -> SpoilerHorizon:
    return SpoilerHorizon(
        horizon_id="horizon-1",
        max_discourse_position=DiscoursePosition(passage_order=10),
        max_revelation_position=RevelationPosition(revelation_order=10),
    )


def budgets() -> OutputBudgets:
    return OutputBudgets(
        node_budget=10,
        assertion_budget=12,
        display_node_budget=10,
        display_assertion_budget=12,
    )


def temporal_scope() -> TemporalScope:
    return TemporalScope(
        story_time=StoryTime(kind=TemporalKind.POINT, point=3, label="day three"),
        validity_time=ValidityTime(kind=TemporalKind.INTERVAL, start=3, end=5),
        discourse_position=DiscoursePosition(passage_order=1),
        revelation_position=RevelationPosition(revelation_order=1),
    )


def query_context() -> QueryContext:
    return QueryContext(
        context_id="context-1",
        wording="Which relationships explain the bridge closure?",
        lens="causal consequence",
        target="bridge closure",
        story_scope=StoryTime(kind=TemporalKind.INTERVAL, start=0, end=5),
        spoiler_horizon=horizon(),
        viewpoint=None,
        abstraction=AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN,
        budgets=budgets(),
        revealed_at=NOW,
    )


def decision(
    operator: ConstructionOperator = ConstructionOperator.CONTEXTUAL_TYPE,
) -> OntologyDecision:
    return OntologyDecision(
        decision_id=f"decision-{operator.value}",
        operator=operator,
        evidence_ids=("ev-1",),
        rationale="The contextual lens requires this supported distinction.",
        decided_at=NOW + timedelta(seconds=1),
        created_object_ids=("local-object-1",)
        if operator
        not in {
            ConstructionOperator.SELECTION,
            ConstructionOperator.COMPRESSION,
            ConstructionOperator.SUPPORTED_DESCRIPTION,
        }
        else (),
    )


def upper_ontology() -> UpperOntology:
    return UpperOntology(
        ontology_id="upper-1",
        primitive_types=("entity", "event", "proposition"),
        primitive_relations=("related_to",),
        temporal_terms=("story_time", "validity_time"),
        epistemic_terms=("believed", "reported", "denied"),
        revision="fixture-v1",
    )


def local_schema() -> LocalContextSchema:
    return LocalContextSchema(
        schema_id="schema-1",
        contextual_types=(),
        predicates=(),
        abstraction=AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN,
    )


def test_canonical_serialization_and_hash_are_stable() -> None:
    first = {"z": (2, 1), "a": {"right": True, "left": "é"}}
    second = {"a": {"left": "é", "right": True}, "z": [2, 1]}
    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)

    utc_record = DiscoursePosition(passage_order=1)
    assert utc_record.content_hash == canonical_sha256(utc_record)
    assert len(utc_record.content_hash) == 64


def test_records_are_immutable_and_serialized_hashes_are_verified() -> None:
    record = DiscoursePosition(passage_order=1)
    with pytest.raises(ValidationError):
        record.passage_order = 2  # type: ignore[misc]

    serialized = record.model_dump(mode="json")
    serialized["content_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="content_hash mismatch"):
        DiscoursePosition.model_validate(serialized)


def test_evidence_contract_cannot_carry_constructed_ontology_fields() -> None:
    record = evidence_record()
    assert_evidence_boundary(record)
    with pytest.raises(ValueError, match="constructed ontology field"):
        assert_evidence_boundary(
            {
                "evidence_id": "ev-1",
                "candidate": {"canonical_entity_id": "entity-final"},
            }
        )

    payload = record.model_dump(mode="json")
    payload["entity_id"] = "entity-final"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceRecord.model_validate(payload)


def test_model_visible_packet_is_an_allowlist() -> None:
    evidence = evidence_record()
    packet = EvidencePacket(
        packet_id="packet-1",
        snapshot_hash=digest("snapshot"),
        evidence=(evidence,),
        ordered_evidence_ids=(evidence.evidence_id,),
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        ranks={evidence.evidence_id: 1},
        scores={evidence.evidence_id: 1.0},
        token_count=12,
        created_at=NOW,
        release_class=ReleaseClass.PUBLIC,
    )
    visible = to_model_visible_packet(packet).model_dump(mode="json")
    visible_evidence = visible["evidence"][0]
    assert "release_class" not in visible_evidence
    assert visible_evidence["passage_id"] == evidence.passage_id
    assert visible_evidence["text_hash"] == evidence.text_hash
    assert visible_evidence["provenance"] == evidence.provenance.model_dump(mode="json")
    assert visible_evidence["confidence"] == evidence.confidence
    assert visible["packet_hash"] == packet.content_hash


def test_query_model_view_strips_runner_identifiers_and_rejects_gold_fields() -> None:
    visible = to_model_visible_query(query_context())
    payload = visible.model_dump(mode="json")
    assert "context_id" not in payload
    assert "revealed_at" not in payload
    assert "world_id" not in payload
    assert "split" not in payload
    assert "gold_id" not in payload
    assert "expected_effect" not in payload

    payload["gold_id"] = "gold-secret"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelVisibleQueryContext.model_validate(payload)


def test_no_context_model_view_is_exactly_generic_request_and_budgets() -> None:
    context = query_context()
    visible = to_model_visible_context_for_condition(context, ConditionName.A_NO_CONTEXT)
    assert isinstance(visible, ModelVisibleGenericConstructionContext)
    payload = visible.model_dump(mode="json", exclude={"schema_version", "content_hash"})
    assert set(payload) == MODEL_VISIBLE_GENERIC_CONTEXT_FIELDS
    serialized = canonical_json(payload)
    for hidden_query_field in (
        "wording",
        "lens",
        "target",
        "story_scope",
        "spoiler_horizon",
        "viewpoint",
        "abstraction",
    ):
        assert hidden_query_field not in serialized
    assert visible.budgets == context.budgets
    assert isinstance(
        to_model_visible_context_for_condition(context, ConditionName.C2_LLM_QUERY),
        ModelVisibleQueryContext,
    )


def test_story_validity_discourse_revelation_and_spoiler_are_distinct() -> None:
    scope = temporal_scope()
    query = query_context()
    assert isinstance(scope.story_time, StoryTime)
    assert isinstance(scope.validity_time, ValidityTime)
    assert scope.discourse_position.passage_order == 1
    assert scope.revelation_position.revelation_order == 1
    assert query.spoiler_horizon.max_discourse_position.passage_order == 10

    schema = TemporalScope.model_json_schema()
    assert set(schema["properties"]) >= {
        "story_time",
        "validity_time",
        "discourse_position",
        "revelation_position",
    }
    assert "spoiler_horizon" not in schema["properties"]


def test_holder_attribution_cannot_be_promoted_to_world_truth() -> None:
    epistemic = EpistemicScope(
        holder_id="entity-mira",
        attitude=EpistemicAttitude.REPORTED,
        proposition_content_id="proposition-bridge-closed",
        holder_relative_time=HolderRelativeTime(kind=TemporalKind.POINT, point=3),
        evidence_ids=("ev-1",),
    )
    common = dict(
        assertion_id="assertion-report",
        proposition_content_id="proposition-bridge-closed",
        predicate_id="reported",
        subject_id="entity-mira",
        object_id="proposition-bridge-closed",
        temporal_scope=temporal_scope(),
        epistemic_scope=epistemic,
        confidence=0.9,
        evidence_ids=("ev-1",),
        provenance=(provenance(),),
        contextual_relevance=0.8,
        why_matters="The report changes what Mira can act on.",
        why_matters_evidence_ids=("ev-1",),
    )
    attributed = QualifiedAssertion(
        **common,
        narrative_commitment=NarrativeCommitment.HOLDER_ATTRIBUTED,
    )
    assert attributed.epistemic_scope is not None
    assert attributed.epistemic_scope.holder_id == "entity-mira"

    with pytest.raises(ValidationError, match="cannot be promoted to world truth"):
        QualifiedAssertion(
            **common,
            narrative_commitment=NarrativeCommitment.WORLD_COMMITTED,
        )


def test_prequery_inventory_mechanically_requires_no_c2_ontology() -> None:
    inventory = PreQueryInventory(
        inventory_id="inventory-1",
        snapshot_hash=digest("snapshot"),
        recorded_at=NOW - timedelta(seconds=1),
    )
    assert not inventory.entity_ids
    with pytest.raises(ValidationError, match="must contain no constructed ontology"):
        PreQueryInventory(
            inventory_id="inventory-bad",
            snapshot_hash=digest("snapshot"),
            recorded_at=NOW,
            entity_ids=("hidden-entity",),
        )


def test_query_access_event_separates_registered_and_physical_reveal_time() -> None:
    event = QueryAccessEvent(
        access_event_id="access-1",
        execution_id="execution-1",
        query_context_hash=digest("context"),
        model_visible_query_hash=digest("visible-query"),
        snapshot_hash=digest("snapshot"),
        stage_manifest_hash=digest("stage-manifest"),
        query_artifact_hash=digest("query-artifact"),
        prequery_barrier_hash=digest("prequery-barrier"),
        packet_hash=digest("packet"),
        registered_revealed_at=NOW,
        accessed_at=NOW + timedelta(seconds=1),
    )
    assert event.accessed_at > event.registered_revealed_at

    with pytest.raises(ValidationError, match="cannot predate"):
        QueryAccessEvent(
            **event.model_dump(
                mode="python",
                exclude={"content_hash", "accessed_at"},
            ),
            accessed_at=NOW - timedelta(microseconds=1),
        )


def test_c2_certificate_requires_post_reveal_nonselection_decision() -> None:
    inventory = PreQueryInventory(
        inventory_id="inventory-1",
        snapshot_hash=digest("snapshot"),
        recorded_at=NOW - timedelta(seconds=1),
    )
    certificate = ConstructionCertificate(
        certificate_id="certificate-1",
        condition=ConditionName.C2_LLM_QUERY,
        snapshot_hash=digest("snapshot"),
        packet_hash=digest("packet"),
        query_context_hash=digest("context"),
        query_access_event_hash=digest("query-access"),
        stage_manifest_hash=digest("stage-manifest"),
        prequery_barrier_hash=digest("prequery-barrier"),
        generation_lineage_hash=digest("generation-lineage"),
        raw_output_artifact_hash=digest("raw-output"),
        normalized_draft_hash=digest("normalized-draft"),
        validation_bundle_hash=digest("validation-bundle"),
        query_revealed_at=NOW,
        completed_at=NOW + timedelta(seconds=2),
        decisions=(decision(),),
        pre_query_inventory_hash=inventory.content_hash,
    )
    assert certificate.decisions[0].operator is ConstructionOperator.CONTEXTUAL_TYPE

    with pytest.raises(ValidationError, match="nonselection construction decision"):
        ConstructionCertificate(
            certificate_id="certificate-selection-only",
            condition=ConditionName.C2_LLM_QUERY,
            snapshot_hash=digest("snapshot"),
            packet_hash=digest("packet"),
            query_context_hash=digest("context"),
            query_access_event_hash=digest("query-access"),
            stage_manifest_hash=digest("stage-manifest"),
            prequery_barrier_hash=digest("prequery-barrier"),
            generation_lineage_hash=digest("generation-lineage"),
            raw_output_artifact_hash=digest("raw-output"),
            normalized_draft_hash=digest("normalized-draft"),
            validation_bundle_hash=digest("validation-bundle"),
            query_revealed_at=NOW,
            completed_at=NOW + timedelta(seconds=2),
            decisions=(decision(ConstructionOperator.SELECTION),),
            pre_query_inventory_hash=inventory.content_hash,
        )

    for audit_only_operator in (
        ConstructionOperator.INCLUDE_EXCLUDE,
        ConstructionOperator.RARE_PRESERVATION,
    ):
        with pytest.raises(
            ValidationError,
            match="decisions alone do not establish ontology construction",
        ):
            ConstructionCertificate(
                certificate_id=f"certificate-{audit_only_operator.value}-only",
                condition=ConditionName.C2_LLM_QUERY,
                snapshot_hash=digest("snapshot"),
                packet_hash=digest("packet"),
                query_context_hash=digest("context"),
                query_access_event_hash=digest("query-access"),
                stage_manifest_hash=digest("stage-manifest"),
                prequery_barrier_hash=digest("prequery-barrier"),
                generation_lineage_hash=digest("generation-lineage"),
                raw_output_artifact_hash=digest("raw-output"),
                normalized_draft_hash=digest("normalized-draft"),
                validation_bundle_hash=digest("validation-bundle"),
                query_revealed_at=NOW,
                completed_at=NOW + timedelta(seconds=2),
                decisions=(decision(audit_only_operator),),
                pre_query_inventory_hash=inventory.content_hash,
            )


def test_fixed_select_certificate_rejects_every_constructive_operator() -> None:
    with pytest.raises(ValidationError, match="forbidden construction operators"):
        ConstructionCertificate(
            certificate_id="fixed-certificate-bad",
            condition=ConditionName.A_FIXED_SELECT,
            snapshot_hash=digest("snapshot"),
            packet_hash=digest("packet"),
            query_context_hash=digest("context"),
            query_access_event_hash=digest("query-access"),
            stage_manifest_hash=digest("stage-manifest"),
            prequery_barrier_hash=digest("prequery-barrier"),
            generation_lineage_hash=digest("generation-lineage"),
            raw_output_artifact_hash=digest("raw-output"),
            normalized_draft_hash=digest("normalized-draft"),
            validation_bundle_hash=digest("validation-bundle"),
            query_revealed_at=NOW,
            completed_at=NOW + timedelta(seconds=2),
            decisions=(decision(ConstructionOperator.MERGE),),
            inherited_construction_seal_hash=digest("seal"),
        )


def test_c2_projection_requires_inventory_certificate_and_enforces_budget() -> None:
    inventory = PreQueryInventory(
        inventory_id="inventory-1",
        snapshot_hash=digest("snapshot"),
        recorded_at=NOW - timedelta(seconds=1),
    )
    construction_decision = decision()
    validation = runtime_structural_acceptance_record(
        validation_id="validation-1",
        target_id=digest("normalized-draft"),
        validated_at=NOW + timedelta(seconds=3),
    )
    assert validation.diagnostics == (RUNTIME_STRUCTURAL_ONLY_DIAGNOSTIC,)
    assert validation.semantic_assessment_scope is (
        SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
    )
    unassessed_posthoc = validation.model_dump(mode="python", exclude={"content_hash"})
    unassessed_posthoc["semantic_assessment_scope"] = (
        SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER
    )
    with pytest.raises(ValidationError, match="at least one assessed semantic status"):
        ValidationRecord.model_validate(unassessed_posthoc)
    validation_bundle_hash = canonical_sha256((validation,))
    certificate = ConstructionCertificate(
        certificate_id="certificate-1",
        condition=ConditionName.C2_LLM_QUERY,
        snapshot_hash=digest("snapshot"),
        packet_hash=digest("packet"),
        query_context_hash=digest("context"),
        query_access_event_hash=digest("query-access"),
        stage_manifest_hash=digest("stage-manifest"),
        prequery_barrier_hash=digest("prequery-barrier"),
        generation_lineage_hash=digest("generation-lineage"),
        raw_output_artifact_hash=digest("raw-output"),
        normalized_draft_hash=digest("normalized-draft"),
        validation_bundle_hash=validation_bundle_hash,
        query_revealed_at=NOW,
        completed_at=NOW + timedelta(seconds=2),
        decisions=(construction_decision,),
        pre_query_inventory_hash=inventory.content_hash,
    )
    projection = OntologyProjection(
        projection_id="projection-1",
        condition=ConditionName.C2_LLM_QUERY,
        snapshot_hash=digest("snapshot"),
        packet_hash=digest("packet"),
        context_hash=digest("context"),
        query_access_event_hash=digest("query-access"),
        generation_lineage_hash=digest("generation-lineage"),
        raw_output_artifact_hash=digest("raw-output"),
        normalized_draft_hash=digest("normalized-draft"),
        validation_bundle_hash=validation_bundle_hash,
        upper_ontology=upper_ontology(),
        local_schema=local_schema(),
        instance_graph=InstanceGraph(entities=(), events=(), assertions=()),
        decisions=(construction_decision,),
        validation_records=(validation,),
        budget_accounting=BudgetAccounting(
            nodes_used=0,
            assertions_used=0,
            display_nodes_used=0,
            display_assertions_used=0,
            input_tokens=100,
            output_tokens=20,
        ),
        budgets=budgets(),
        pre_query_inventory=inventory,
        construction_certificate=certificate,
        run_id="run-1",
        release_class=ReleaseClass.PUBLIC,
    )
    assert projection.pre_query_inventory is not None

    semantic_claim_payload = validation.model_dump(mode="python", exclude={"content_hash"})
    semantic_claim_payload["evidence_support_status"] = EvidenceSupportStatus.SUPPORTED
    with pytest.raises(ValidationError, match="unassessed semantic statuses"):
        ValidationRecord.model_validate(semantic_claim_payload)
    semantic_claim_payload["semantic_assessment_scope"] = (
        SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER
    )
    semantic_claim = ValidationRecord.model_validate(semantic_claim_payload)
    semantic_claim_hash = canonical_sha256((semantic_claim,))
    projection_payload = projection.model_dump(
        mode="python",
        exclude={"content_hash", "construction_certificate"},
    )
    projection_payload["validation_records"] = (semantic_claim,)
    projection_payload["validation_bundle_hash"] = semantic_claim_hash
    certificate_payload = certificate.model_dump(mode="python", exclude={"content_hash"})
    certificate_payload["validation_bundle_hash"] = semantic_claim_hash
    projection_payload["construction_certificate"] = ConstructionCertificate.model_validate(
        certificate_payload
    )
    with pytest.raises(ValidationError, match="cannot claim post-hoc semantic status"):
        OntologyProjection.model_validate(projection_payload)

    payload = projection.model_dump(mode="python")
    payload["pre_query_inventory"] = None
    payload["content_hash"] = ""
    with pytest.raises(ValidationError, match="requires empty inventory"):
        OntologyProjection.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("validation_status", ValidationStatus.REJECTED),
        ("evidence_support_status", EvidenceSupportStatus.SUPPORTED),
        ("temporal_status", TemporalDeterminationStatus.VALID),
        ("commitment_status", CommitmentCheckStatus.VALID),
    ),
)
def test_validated_generation_requires_exact_structural_only_runtime_statuses(
    field: str,
    value: object,
) -> None:
    raw_draft = OntologyDraft.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "phase1"
            / "c1_pre_output.json"
        ).read_text(encoding="utf-8")
    )
    decision_at = NOW + timedelta(seconds=1)
    completed_at = NOW + timedelta(seconds=2)
    validated_at = NOW + timedelta(seconds=3)
    normalized = normalize_generation_metadata(
        raw_draft,
        decision_recorded_at=decision_at,
        input_tokens=11,
        output_tokens=22,
    )
    validation = runtime_structural_acceptance_record(
        validation_id="validation-c1-structural",
        target_id=normalized.content_hash,
        validated_at=validated_at,
    )
    generation = ValidatedGeneration(
        generation_id="generation-c1-structural",
        condition=ConditionName.C1_LLM_PRE,
        request_hash=digest("request"),
        raw_output_artifact_hash=digest("raw-output"),
        raw_parsed_draft=raw_draft,
        draft=normalized,
        normalized_draft_hash=normalized.content_hash,
        stage_manifest_hash=digest("stage"),
        packing_report_hash=digest("packing"),
        capability_manifest_hash=digest("capability"),
        seed_manifest_hash=digest("seed-manifest"),
        prompt_hash=digest("prompt"),
        output_schema_hash=digest("schema"),
        decoding_manifest_hash=digest("decoding"),
        validator_hash=digest("validator"),
        model_stack_hash=digest("model-stack"),
        seed=17,
        input_tokens=11,
        output_tokens=22,
        generation_started_at=NOW,
        generation_completed_at=completed_at,
        decision_recorded_at=decision_at,
        validation_records=(validation,),
        validator_report_hashes=(digest("validator-report"),),
        validated_at=validated_at,
    )
    assert generation.validation_records == (validation,)

    status_payload = validation.model_dump(mode="python", exclude={"content_hash"})
    status_payload[field] = value
    if field != "validation_status":
        with pytest.raises(ValidationError, match="unassessed semantic statuses"):
            ValidationRecord.model_validate(status_payload)
        return
    status_record = ValidationRecord.model_validate(status_payload)
    generation_payload = generation.model_dump(mode="python", exclude={"content_hash"})
    generation_payload["validation_records"] = (status_record,)
    with pytest.raises(
        ValidationError,
        match="accepted structural validation with semantic statuses not_applicable",
    ):
        ValidatedGeneration.model_validate(generation_payload)


@pytest.mark.parametrize(
    ("repair_attempt", "repair_parent_hash"),
    (
        (0, digest("unexpected-repair-parent")),
        (1, None),
    ),
)
def test_validation_record_requires_paired_repair_lineage(
    repair_attempt: int,
    repair_parent_hash: str | None,
) -> None:
    with pytest.raises(
        ValidationError,
        match="validation repair attempt and repair-parent hash disagree",
    ):
        runtime_structural_acceptance_record(
            validation_id="validation-invalid-repair-lineage",
            target_id=digest("normalized-draft"),
            repair_attempt=repair_attempt,
            repair_parent_hash=repair_parent_hash,
            validated_at=NOW,
        )


def test_validated_generation_rejects_validation_record_repair_lineage_mismatch() -> None:
    raw_draft = OntologyDraft.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "phase1"
            / "c1_pre_output.json"
        ).read_text(encoding="utf-8")
    )
    decision_at = NOW + timedelta(seconds=1)
    normalized = normalize_generation_metadata(
        raw_draft,
        decision_recorded_at=decision_at,
        input_tokens=11,
        output_tokens=22,
    )
    parent_hash = digest("first-raw-output")
    repair_validation = runtime_structural_acceptance_record(
        validation_id="validation-repair",
        target_id=normalized.content_hash,
        repair_parent_hash=parent_hash,
        repair_attempt=1,
        validated_at=NOW + timedelta(seconds=3),
    )
    with pytest.raises(
        ValidationError,
        match="validation record repair lineage differs from generation lineage",
    ):
        ValidatedGeneration(
            generation_id="generation-mismatched-repair",
            condition=ConditionName.C1_LLM_PRE,
            request_hash=digest("request"),
            raw_output_artifact_hash=digest("repair-raw-output"),
            raw_parsed_draft=raw_draft,
            draft=normalized,
            normalized_draft_hash=normalized.content_hash,
            stage_manifest_hash=digest("stage"),
            packing_report_hash=digest("packing"),
            capability_manifest_hash=digest("capability"),
            seed_manifest_hash=digest("seed-manifest"),
            prompt_hash=digest("prompt"),
            output_schema_hash=digest("schema"),
            decoding_manifest_hash=digest("decoding"),
            validator_hash=digest("validator"),
            model_stack_hash=digest("model-stack"),
            seed=17,
            input_tokens=11,
            output_tokens=22,
            generation_started_at=NOW,
            generation_completed_at=NOW + timedelta(seconds=2),
            decision_recorded_at=decision_at,
            validation_records=(repair_validation,),
            validator_report_hashes=(digest("validator-report"),),
            validated_at=NOW + timedelta(seconds=3),
        )


def test_capability_sets_are_symmetric_or_mechanically_selection_only() -> None:
    active = ConstructionCapabilities.active_construction()
    fixed = ConstructionCapabilities.fixed_selection()
    assert active.create_entities and active.merge_split and active.reify_events
    assert not fixed.create_entities
    assert not fixed.merge_split
    assert not fixed.create_schema_predicates
    assert not fixed.reify_events
    assert not fixed.change_abstraction
    assert fixed.select_existing and fixed.compress_existing


def test_public_release_scan_rejects_restricted_nested_records() -> None:
    assert_public_release(evidence_record())
    with pytest.raises(ValueError, match="restricted material"):
        assert_public_release(evidence_record(release_class=ReleaseClass.RESTRICTED))


def test_gold_projection_is_scorer_only_anchor_based_and_review_audited() -> None:
    gold = GoldContextualProjection(
        gold_projection_id="gold-projection-1",
        split=BenchmarkSplit.DEVELOPMENT,
        world_id="world-1",
        query_id="query-1",
        local_schema=local_schema(),
        entity_partition=(
            GoldEntityCluster(
                cluster_id="cluster-mira",
                mention_candidate_ids=("mention-mira-1", "mention-mira-2"),
            ),
        ),
        events=(),
        qualified_assertions=(),
        relevance=(
            GoldRelevanceAnnotation(
                target_id="cluster-mira",
                target_kind=GoldTargetKind.ENTITY_CLUSTER,
                is_relevant=True,
            ),
        ),
        assertion_annotations=(),
        matcher_revision="matcher-v1",
        review_status=GoldReviewStatus.NOT_SELECTED,
        adjudication_status=GoldAdjudicationStatus.NOT_REQUIRED,
        compiled_at=NOW,
    )
    assert gold.scorer_namespace == "scorer_only"
    assert gold.entity_partition[0].mention_candidate_ids == (
        "mention-mira-1",
        "mention-mira-2",
    )

    with pytest.raises(ValidationError, match="review record hash"):
        GoldContextualProjection(
            **{
                **gold.model_dump(mode="python", exclude={"content_hash"}),
                "review_status": GoldReviewStatus.REVIEWED,
            }
        )

    adjudicated = GoldContextualProjection(
        **{
            **gold.model_dump(mode="python", exclude={"content_hash"}),
            "review_status": GoldReviewStatus.DISAGREEMENT_LOGGED,
            "adjudication_status": GoldAdjudicationStatus.ADJUDICATED,
            "independent_review_record_hash": digest("independent-review"),
            "adjudication_record_hash": digest("adjudication"),
        }
    )
    assert adjudicated.adjudication_record_hash == digest("adjudication")

    with pytest.raises(ValidationError, match="exactly one adjudication"):
        GoldContextualProjection(
            **{
                **adjudicated.model_dump(mode="python", exclude={"content_hash"}),
                "adjudication_record_hash": None,
            }
        )


def test_gold_rare_and_pivotal_labels_are_independent() -> None:
    rare_nonpivotal = GoldAssertionAnnotation(
        assertion_id="assertion-rare",
        is_rare=True,
        is_pivotal=False,
    )
    common_pivotal = GoldAssertionAnnotation(
        assertion_id="assertion-pivotal",
        is_rare=False,
        is_pivotal=True,
    )
    assert rare_nonpivotal.is_rare and not rare_nonpivotal.is_pivotal
    assert common_pivotal.is_pivotal and not common_pivotal.is_rare

    with pytest.raises(ValidationError, match="require a support path"):
        GoldAssertionAnnotation(
            assertion_id="assertion-rare-pivotal",
            is_rare=True,
            is_pivotal=True,
        )


def test_gold_relevance_and_community_references_must_resolve() -> None:
    base = GoldContextualProjection(
        gold_projection_id="gold-projection-integrity",
        split=BenchmarkSplit.DEVELOPMENT,
        world_id="world-integrity",
        query_id="query-integrity",
        local_schema=local_schema(),
        entity_partition=(
            GoldEntityCluster(
                cluster_id="cluster-mira",
                mention_candidate_ids=("mention-mira-1",),
            ),
        ),
        events=(),
        qualified_assertions=(),
        relevance=(),
        assertion_annotations=(),
        matcher_revision="matcher-v1",
        review_status=GoldReviewStatus.NOT_SELECTED,
        adjudication_status=GoldAdjudicationStatus.NOT_REQUIRED,
        compiled_at=NOW,
    )
    with pytest.raises(ValidationError, match="unknown target"):
        GoldContextualProjection(
            **{
                **base.model_dump(mode="python", exclude={"content_hash"}),
                "relevance": (
                    GoldRelevanceAnnotation(
                        target_id="missing-target",
                        target_kind=GoldTargetKind.ENTITY_CLUSTER,
                        is_relevant=True,
                    ),
                ),
            }
        )
    with pytest.raises(ValidationError, match="target kind"):
        GoldContextualProjection(
            **{
                **base.model_dump(mode="python", exclude={"content_hash"}),
                "relevance": (
                    GoldRelevanceAnnotation(
                        target_id="cluster-mira",
                        target_kind=GoldTargetKind.EVENT,
                        is_relevant=True,
                    ),
                ),
            }
        )


def test_gold_alternatives_cannot_be_empty_or_enter_model_visible_registry() -> None:
    with pytest.raises(ValidationError, match="require a projection ID or constraints"):
        GoldAlternativeSet(
            alternative_set_id="alternatives-empty",
            gold_projection_id="gold-projection-1",
            equivalence_rule="Anchor-equivalent structures are accepted.",
            matching_rule="Use the frozen anchor matcher.",
            matcher_revision="matcher-v1",
            review_status=GoldReviewStatus.NOT_SELECTED,
            adjudication_status=GoldAdjudicationStatus.NOT_REQUIRED,
            compiled_at=NOW,
        )

    alternatives = GoldAlternativeSet(
        alternative_set_id="alternatives-1",
        gold_projection_id="gold-projection-1",
        permissible_projection_ids=("permissible-1",),
        equivalence_rule="Anchor-equivalent structures are accepted.",
        matching_rule="Use the frozen anchor matcher.",
        matcher_revision="matcher-v1",
        review_status=GoldReviewStatus.NOT_SELECTED,
        adjudication_status=GoldAdjudicationStatus.NOT_REQUIRED,
        compiled_at=NOW,
    )
    assert alternatives.scorer_namespace == "scorer_only"
    assert set(contracts_module.SCORER_ONLY_SCHEMA_TYPES).isdisjoint(
        contracts_module.MODEL_VISIBLE_SCHEMA_TYPES
    )
    assert not hasattr(contracts_module, "to_model_visible_gold")

    with pytest.raises(ValidationError, match="projection IDs must be unique"):
        GoldAlternativeSet(
            **{
                **alternatives.model_dump(mode="python", exclude={"content_hash"}),
                "permissible_projection_ids": ("permissible-1", "permissible-1"),
            }
        )


def test_parent_projection_reference_is_same_context_lineage_not_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot be referenced before"):
        ParentProjectionRef(
            parent_projection_id="projection-parent",
            parent_projection_hash=digest("projection-parent"),
            snapshot_hash=digest("snapshot"),
            packet_hash=digest("packet"),
            context_hash=digest("context"),
            model_visible_context_hash=digest("visible-context"),
            parent_completed_at=NOW + timedelta(seconds=1),
            referenced_at=NOW,
        )


def visualization_node() -> VisualizationNode:
    return VisualizationNode(
        visualization_node_id="visual-node-mira",
        projection_object_id="entity-mira",
        projection_object_hash=digest("entity-mira"),
        contextual_label="Mira — witness",
        contextual_type_id="witness",
        contextual_role="reports the closure",
        abstraction=AbstractionLevel.ACTOR,
        temporal_state=StoryTime(kind=TemporalKind.POINT, point=3),
        uncertainty=ExplicitValueState.KNOWN,
        confidence=0.9,
        evidence_badge=EvidenceBadge(available=True, evidence_ids=("ev-1",), count=1),
        description="Mira is the evidence-grounded reporting witness.",
        description_assertion_ids=("assertion-1",),
        release_class=ReleaseClass.PUBLIC,
    )


def visualization_state(*, center_x: float = 0.0) -> VisualizationState:
    node = visualization_node()
    projection_hash = digest("projection")
    return VisualizationState(
        visualization_state_id=f"visual-state-{center_x}",
        projection_hash=projection_hash,
        semantic_hash=projection_hash,
        layout_name="cose",
        layout_config_hash=digest("layout"),
        style_config_hash=digest("style"),
        font_config_hash=digest("font"),
        layout_seed=17,
        viewport=Viewport(
            center_x=center_x,
            center_y=0.0,
            zoom=1.0,
            width=1200,
            height=800,
        ),
        nodes=(node,),
        assertions=(),
        positions=(NodePosition(visualization_node_id=node.visualization_node_id, x=1, y=2),),
        visible_node_ids=(node.visualization_node_id,),
        visible_assertion_ids=(),
        labels_visible=True,
        temporal_filter=VisualizationTemporalFilter(),
        release_class=ReleaseClass.PUBLIC,
    )


def test_renderer_changes_content_hash_but_never_semantic_hash() -> None:
    before_pan = visualization_state(center_x=0.0)
    after_pan = visualization_state(center_x=100.0)
    assert before_pan.semantic_hash == after_pan.semantic_hash
    assert before_pan.content_hash != after_pan.content_hash

    payload = before_pan.model_dump(mode="python", exclude={"content_hash"})
    payload["semantic_hash"] = digest("invented-renderer-semantics")
    with pytest.raises(ValidationError, match="must equal the immutable projection hash"):
        VisualizationState.model_validate(payload)


def run_resources() -> RunResourceUsage:
    return RunResourceUsage(
        peak_gpu_vram_bytes=0,
        peak_process_ram_bytes=100_000,
        peak_project_storage_bytes=1_000_000,
        final_project_storage_bytes=900_000,
        minimum_free_storage_bytes=10_000_000,
        cpu_worker_threads=2,
    )


def test_run_manifest_records_reproducible_cpu_success() -> None:
    output = ArtifactHashReference(
        artifact_name="projection",
        artifact_hash=digest("projection-output"),
        release_class=ReleaseClass.PUBLIC,
    )
    manifest = RunManifest(
        manifest_id="run-manifest-1",
        condition=ConditionName.C0_CLASSICAL_PRE,
        run_role="development fixture",
        code_revision=digest("code")[:40],
        dirty_worktree=False,
        environment_hash=digest("environment"),
        model_stack_status=ExplicitValueState.NOT_APPLICABLE,
        prompt_hash=digest("no-prompt"),
        schema_hash=digest("schema"),
        config_hash=digest("config"),
        seed=7,
        seed_manifest_hash=digest("seed-manifest"),
        input_artifacts=(
            ArtifactHashReference(
                artifact_name="evidence",
                artifact_hash=digest("evidence"),
                release_class=ReleaseClass.PUBLIC,
            ),
        ),
        output_artifacts=(output,),
        timing=RunTiming(
            started_at=NOW,
            ended_at=NOW + timedelta(seconds=2),
            allocated_gpu_seconds=0,
        ),
        resources=run_resources(),
        release_class=ReleaseClass.PUBLIC,
        outcome=RunOutcome.SUCCEEDED,
    )
    assert manifest.output_artifacts == (output,)
    assert manifest.model_revision is None


def test_run_manifest_preserves_failure_and_retry_lineage() -> None:
    failure = FailureLineage(
        failure_id="failure-1",
        failure_code="structured-output-invalid",
        public_summary="The response failed structural validation.",
        diagnostic_blob_hash=digest("diagnostic"),
        failed_at=NOW + timedelta(seconds=4),
        parent_attempt_manifest_hash=digest("attempt-1"),
    )
    manifest = RunManifest(
        manifest_id="run-manifest-2",
        condition=ConditionName.C2_LLM_QUERY,
        run_role="repair attempt",
        code_revision=digest("code")[:40],
        dirty_worktree=True,
        dirty_patch_hash=digest("patch"),
        environment_hash=digest("environment"),
        model_stack_status=ExplicitValueState.INVALID,
        model_stack_reason="The runtime rejected initialization before immutable IDs loaded.",
        prompt_hash=digest("prompt"),
        schema_hash=digest("schema"),
        config_hash=digest("config"),
        seed=7,
        seed_manifest_hash=digest("seed-manifest"),
        input_artifacts=(),
        output_artifacts=(),
        timing=RunTiming(
            started_at=NOW,
            ended_at=NOW + timedelta(seconds=4),
            allocated_gpu_seconds=4,
            failed_request_seconds=4,
        ),
        resources=run_resources(),
        parent_manifest_hashes=(digest("attempt-1"),),
        release_class=ReleaseClass.PUBLIC,
        outcome=RunOutcome.FAILED,
        attempt_number=2,
        retry_of_manifest_hash=digest("attempt-1"),
        failure_lineage=failure,
    )
    assert manifest.failure_lineage == failure
    assert manifest.timing.failed_request_seconds == 4
