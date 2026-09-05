from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from story_projection_onto.analysis import (
    CrossingObservation,
    MetricObservation,
    RarePivotalCountObservation,
    compare_crossing_profiles,
    validate_cross_condition_metric_design,
    validate_cross_condition_rare_design,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    CommitmentCheckStatus,
    ConditionName,
    ConstructionOperator,
    EvidenceSupportStatus,
    GoldContextualProjection,
    GoldContrastInvariant,
    OntologyDraft,
    OntologyProjection,
    RoleBinding,
    SemanticAssessmentScope,
    TemporalDeterminationStatus,
    ValidationRecord,
    ValidationStatus,
    canonical_sha256,
)
from story_projection_onto.metrics.adapters import (
    adapt_projection_for_metrics,
    projection_is_content_bearing,
    verify_projection_decisions,
)
from story_projection_onto.metrics.alignment import (
    AlignmentIntegrityError,
    AlignmentPlan,
    AnchorKind,
    DecisionFamily,
    NodeAlignmentTarget,
    NodeKind,
    NormalizedDecision,
)
from story_projection_onto.metrics.common import revalidated_copy
from story_projection_onto.metrics.config import (
    LEIDEN_LIBRARY_SEED_MODULUS,
    CommunityReviewTemplate,
    StudyMetricConfiguration,
)
from story_projection_onto.metrics.contrastive import (
    aggregate_contrastive_collapse,
    score_contrast_invariants,
    score_contrastive_changes,
)
from story_projection_onto.metrics.decision_states import compile_and_verify_decision_states
from story_projection_onto.metrics.entropy import entropy_profile
from story_projection_onto.metrics.pipeline import (
    FailedMetricOutput,
    IntendedMetricManifest,
    IntendedMetricUnit,
    OutputFailureKind,
    PipelineMetricStatus,
    ScorerMetricPlan,
    analysis_observations_from_scores,
    materialize_intention_to_treat_scores,
    score_failed_output,
    score_intended_projection,
    score_projection,
)

ROOT = Path(__file__).resolve().parents[2]
HASH = "a" * 64
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def metric_configuration() -> StudyMetricConfiguration:
    return StudyMetricConfiguration.load(ROOT / "configs/study/metrics.json")


def fixture_projection(*, accepted: bool = True) -> OntologyProjection:
    payload = json.loads((ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text())
    draft = OntologyDraft.model_validate(payload)
    validation = ValidationRecord(
        validation_id="projection-validation",
        target_id="projection",
        validation_status=(ValidationStatus.ACCEPTED if accepted else ValidationStatus.INVALID),
        evidence_support_status=EvidenceSupportStatus.SUPPORTED,
        temporal_status=TemporalDeterminationStatus.VALID,
        commitment_status=CommitmentCheckStatus.VALID,
        semantic_assessment_scope=SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER,
        validated_at=NOW,
    )
    # The adapter deliberately consumes only the condition-neutral semantic payload
    # and validation lineage; model_construct avoids irrelevant condition lineage here.
    return OntologyProjection.model_construct(
        projection_id="projection-fixture",
        content_hash=HASH,
        condition=ConditionName.C2_LLM_QUERY,
        snapshot_hash="d" * 64,
        packet_hash="e" * 64,
        context_hash="f" * 64,
        local_schema=draft.local_schema,
        instance_graph=draft.instance_graph,
        decisions=draft.decisions,
        validation_records=(validation,),
    )


def empty_alignment_plan() -> AlignmentPlan:
    return AlignmentPlan(
        matcher_revision="matcher-v1",
        source_gold_hash="b" * 64,
        source_alternative_set_hash="c" * 64,
        node_targets=(),
        assertion_targets=(),
    )


def test_frozen_metric_configuration_and_empty_review_contract() -> None:
    configuration = metric_configuration()
    review = CommunityReviewTemplate.load(
        ROOT / "configs/study/community_review_template.json"
    )
    assert configuration.leiden_half_resolution == configuration.leiden_base_resolution / 2
    assert configuration.leiden_double_resolution == configuration.leiden_base_resolution * 2
    assert configuration.von_neumann_entropy == "omitted"
    assert "not inherently better" in configuration.simplification_claim_boundary
    visualization = json.loads(
        (ROOT / "configs/study/visualization.json").read_text(encoding="utf-8")
    )
    assert configuration.renderer.layout_name == "preset-anchor-hash-v2"
    assert configuration.renderer.layout_seed == visualization["layout_seed"]
    assert configuration.renderer.visualization_configuration_hash == canonical_sha256(
        visualization
    )
    assert configuration.renderer.layout_config_hash == canonical_sha256(
        {
            "algorithm": "preset",
            "coordinate_rule": visualization["coordinate_rule"],
            "seed": visualization["layout_seed"],
            "seed_manifest_hash": visualization["seed_manifest_hash"],
            "layout_seed_entry_hash": visualization["layout_seed_entry_hash"],
        }
    )
    seed_manifest = json.loads(
        (ROOT / "data/synthetic/manifests/seed_manifest.json").read_text(encoding="utf-8")
    )
    leiden_entry = next(
        item for item in seed_manifest["entries"] if item["purpose"] == "leiden"
    )
    assert configuration.seed_manifest_hash == canonical_sha256(seed_manifest)
    assert configuration.renderer.seed_manifest_hash == configuration.seed_manifest_hash
    assert configuration.leiden_seed_entry_hash == canonical_sha256(leiden_entry)
    assert configuration.leiden_source_seed == leiden_entry["seed"]
    assert configuration.leiden_seed_mapping == "modulo-2147483647-v1"
    assert configuration.leiden_seed == (
        configuration.leiden_source_seed % LEIDEN_LIBRARY_SEED_MODULUS
    )
    assert configuration.metric_formula_revision == "conference-metric-formulas-v3"
    assert configuration.upper_relation_map["located_at"] == "descriptive"
    assert review.condition_blind and review.reviews == ()


def test_metric_configuration_fails_closed_on_visualization_drift(tmp_path: Path) -> None:
    metric_path = tmp_path / "metrics.json"
    visualization_path = tmp_path / "visualization.json"
    metric_path.write_text(
        (ROOT / "configs/study/metrics.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    visualization = json.loads(
        (ROOT / "configs/study/visualization.json").read_text(encoding="utf-8")
    )
    visualization["layout_seed"] += 1
    visualization_path.write_text(json.dumps(visualization), encoding="utf-8")
    with pytest.raises(ValueError, match="visualization configuration hash"):
        StudyMetricConfiguration.load(
            metric_path,
            seed_manifest_path=ROOT / "data/synthetic/manifests/seed_manifest.json",
        )


def test_metric_configuration_fails_closed_on_leiden_seed_manifest_drift(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "seed_manifest.json"
    seed_manifest = json.loads(
        (ROOT / "data/synthetic/manifests/seed_manifest.json").read_text(encoding="utf-8")
    )
    leiden_entry = next(
        item for item in seed_manifest["entries"] if item["purpose"] == "leiden"
    )
    leiden_entry["seed"] += 1
    manifest_path.write_text(json.dumps(seed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="seed manifest hash"):
        StudyMetricConfiguration.load(
            ROOT / "configs/study/metrics.json",
            seed_manifest_path=manifest_path,
        )


def test_projection_adapter_derives_decisions_and_assertion_skeleton() -> None:
    adapter = adapt_projection_for_metrics(fixture_projection(), metric_configuration())
    assert adapter.structurally_valid
    assert adapter.node_ids
    assert adapter.edges
    assert adapter.normalized_decisions
    assert all(item.anchor_ids for item in adapter.normalized_decisions)
    assert all(len(item.semantic_signature) == 64 for item in adapter.normalized_decisions)


def test_projection_adapter_allows_distinct_same_anchor_construction_decisions() -> None:
    projection = fixture_projection()
    source_type = projection.local_schema.contextual_types[0]
    second_type = revalidated_copy(
        source_type,
        type_id="t-c2-alternate",
        label="Alternate role",
    )
    schema = projection.local_schema.model_copy(
        update={
            "contextual_types": (*projection.local_schema.contextual_types, second_type),
        }
    )
    source_decision = next(
        item
        for item in projection.decisions
        if item.operator is ConstructionOperator.CONTEXTUAL_TYPE
    )
    second_decision = revalidated_copy(
        source_decision,
        decision_id="d-c2-type-alternate",
        created_object_ids=(second_type.type_id,),
    )
    adapted = adapt_projection_for_metrics(
        projection.model_copy(
            update={
                "local_schema": schema,
                "decisions": (*projection.decisions, second_decision),
            }
        ),
        metric_configuration(),
    )
    type_decisions = tuple(
        item
        for item in adapted.normalized_decisions
        if item.family is DecisionFamily.CONTEXTUAL_TYPE
    )
    assert len(type_decisions) == 2
    assert len({item.anchor_ids for item in type_decisions}) == 1
    assert len({item.slot_key for item in type_decisions}) == 2
    assert len({item.semantic_signature for item in type_decisions}) == 2


def test_projection_adapter_keeps_anchor_slot_stable_across_semantic_change() -> None:
    projection = fixture_projection()
    original = adapt_projection_for_metrics(projection, metric_configuration())
    source_type = projection.local_schema.contextual_types[0]
    changed_type = revalidated_copy(source_type, label="Changed contextual role")
    changed_schema = projection.local_schema.model_copy(
        update={
            "contextual_types": (
                changed_type,
                *projection.local_schema.contextual_types[1:],
            )
        }
    )
    changed = adapt_projection_for_metrics(
        projection.model_copy(update={"local_schema": changed_schema}),
        metric_configuration(),
    )
    original_type = next(
        item
        for item in original.normalized_decisions
        if item.family is DecisionFamily.CONTEXTUAL_TYPE
    )
    changed_type_decision = next(
        item
        for item in changed.normalized_decisions
        if item.family is DecisionFamily.CONTEXTUAL_TYPE
    )
    assert changed_type_decision.slot_key == original_type.slot_key
    assert changed_type_decision.semantic_signature != original_type.semantic_signature


def test_projection_adapter_reifies_nary_assertion_as_semantic_skeleton() -> None:
    projection = fixture_projection()
    graph = projection.instance_graph
    entity_ids = tuple(item.entity_id for item in graph.entities[:3])
    original = graph.assertions[0]
    nary = original.model_copy(
        update={
            "subject_id": None,
            "object_id": None,
            "roles": tuple(
                RoleBinding(role=f"role-{index}", object_id=object_id, evidence_ids=("ev-02",))
                for index, object_id in enumerate(entity_ids)
            ),
        }
    )
    projection = projection.model_copy(
        update={
            "instance_graph": graph.model_copy(
                update={"assertions": (nary, *graph.assertions[1:])}
            )
        }
    )
    adapter = adapt_projection_for_metrics(projection, metric_configuration())
    nary_edges = tuple(
        item for item in adapter.edges if item.edge_id.startswith(f"{nary.assertion_id}:")
    )
    assert len(nary_edges) == 3
    assert len(adapter.assertion_relations) == len(graph.assertions)
    assert sum(item.assertion_id == nary.assertion_id for item in adapter.assertion_relations) == 1
    entropy = entropy_profile(
        adapter.node_ids,
        adapter.edges,
        canonical_vocabulary=metric_configuration().canonical_relation_vocabulary,
        assertion_relations=adapter.assertion_relations,
    )
    assert entropy.assertion_edge_count == len(graph.assertions)
    assert entropy.topology_edge_count <= len(adapter.edges)
    assert {item.source_id for item in nary_edges} | {item.target_id for item in nary_edges} == set(
        entity_ids
    )


def test_projection_adapter_rejects_unresolved_created_decision_object() -> None:
    projection = fixture_projection()
    first_constructive = next(
        item for item in projection.decisions if item.operator.value == "split"
    )
    invalid_decision = first_constructive.model_copy(
        update={"created_object_ids": ("not-an-emitted-object",)}
    )
    decisions = tuple(
        invalid_decision if item.decision_id == invalid_decision.decision_id else item
        for item in projection.decisions
    )
    with pytest.raises(AlignmentIntegrityError, match="unresolved created objects"):
        adapt_projection_for_metrics(
            projection.model_copy(update={"decisions": decisions}), metric_configuration()
        )


def test_projection_decision_verifier_uses_frozen_target_only_after_emitted_support() -> None:
    projection = fixture_projection()
    adapter = adapt_projection_for_metrics(projection, metric_configuration())
    emitted_split = next(
        item for item in adapter.normalized_decisions if item.operator.value == "split"
    )
    gold_target = revalidated_copy(
        emitted_split,
        slot_key="identity/fixture-pair",
        semantic_signature=emitted_split.semantic_signature,
    )
    verified = verify_projection_decisions(projection, gold_targets=(gold_target,))
    assert gold_target in verified
    verified_target = next(item for item in verified if item.slot_key == gold_target.slot_key)
    assert verified_target.content_hash == gold_target.content_hash
    wrong_semantics = revalidated_copy(gold_target, semantic_signature="0" * 64)
    assert wrong_semantics not in verify_projection_decisions(
        projection, gold_targets=(wrong_semantics,)
    )
    second_target = revalidated_copy(gold_target, slot_key="identity/second-slot")
    one_to_one = verify_projection_decisions(
        projection,
        gold_targets=(gold_target, second_target),
    )
    assert sum(
        item.slot_key in {gold_target.slot_key, second_target.slot_key}
        for item in one_to_one
    ) == 1
    unsupported = revalidated_copy(gold_target, anchor_ids=("missing-gold-anchor",))
    assert unsupported not in verify_projection_decisions(
        projection, gold_targets=(unsupported,)
    )


def test_common_final_state_compiler_rejects_same_operator_anchor_with_wrong_state() -> None:
    scorer_payload = json.loads(
        (ROOT / "data/synthetic/scorer_only/development/syn-dev-01.json").read_text()
    )
    gold = GoldContextualProjection.model_validate(scorer_payload["gold_projections"][0])
    target = NormalizedDecision(
        slot_key="abstraction/root",
        family=DecisionFamily.ABSTRACTION,
        operator=ConstructionOperator.ABSTRACTION,
        anchor_ids=("stable-anchor",),
        semantic_signature="legacy-human-readable-signature",
    )
    emitted = target.model_copy(
        update={
            "slot_key": "emitted-slot",
            "semantic_signature": "untrusted-model-provided-signature",
        }
    )
    projection = fixture_projection().model_copy(update={"local_schema": gold.local_schema})
    correct = compile_and_verify_decision_states(
        gold=gold,
        projection=projection,
        emitted=(emitted,),
        gold_targets=(target,),
    )
    assert correct.predicted_decisions == correct.gold_decisions
    assert NormalizedDecision.model_validate(
        correct.gold_decisions[0].model_dump(mode="python")
    ) == correct.gold_decisions[0]
    different = next(
        item for item in AbstractionLevel if item is not gold.local_schema.abstraction
    )
    wrong_projection = projection.model_copy(
        update={
            "local_schema": projection.local_schema.model_copy(
                update={"abstraction": different}
            )
        }
    )
    wrong = compile_and_verify_decision_states(
        gold=gold,
        projection=wrong_projection,
        emitted=(emitted,),
        gold_targets=(target,),
    )
    assert wrong.predicted_decisions != wrong.gold_decisions


def test_common_final_state_compiler_includes_predicate_domain_and_range() -> None:
    scorer_payload = json.loads(
        (ROOT / "data/synthetic/scorer_only/development/syn-dev-01.json").read_text()
    )
    gold = GoldContextualProjection.model_validate(scorer_payload["gold_projections"][0])
    predicate = gold.local_schema.predicates[0]
    target = NormalizedDecision(
        slot_key="schema-relation/domain-range",
        family=DecisionFamily.SCHEMA_RELATION,
        operator=ConstructionOperator.SCHEMA_RELATION,
        anchor_ids=(predicate.evidence_ids[0],),
        semantic_signature="legacy-human-readable-signature",
    )
    emitted = revalidated_copy(target, slot_key="emitted-schema-relation")
    projection = fixture_projection().model_copy(update={"local_schema": gold.local_schema})
    correct = compile_and_verify_decision_states(
        gold=gold,
        projection=projection,
        emitted=(emitted,),
        gold_targets=(target,),
    )
    assert correct.predicted_decisions == correct.gold_decisions

    changed_predicate = revalidated_copy(
        predicate,
        domain_type_ids=("different-domain",),
    )
    wrong_schema = revalidated_copy(
        gold.local_schema,
        predicates=(changed_predicate, *gold.local_schema.predicates[1:]),
    )
    wrong = compile_and_verify_decision_states(
        gold=gold,
        projection=projection.model_copy(update={"local_schema": wrong_schema}),
        emitted=(emitted,),
        gold_targets=(target,),
    )
    assert wrong.predicted_decisions != wrong.gold_decisions


def test_unrepresentable_contextual_type_gold_fails_closed_with_reason() -> None:
    scorer_payload = json.loads(
        (ROOT / "data/synthetic/scorer_only/development/syn-dev-01.json").read_text()
    )
    gold = GoldContextualProjection.model_validate(scorer_payload["gold_projections"][0])
    target = NormalizedDecision(
        slot_key="contextual-type/actor",
        family=DecisionFamily.CONTEXTUAL_TYPE,
        operator=ConstructionOperator.CONTEXTUAL_TYPE,
        anchor_ids=("stable-anchor",),
        semantic_signature="legacy-human-readable-signature",
    )
    emitted = target.model_copy(update={"slot_key": "emitted-slot"})
    result = compile_and_verify_decision_states(
        gold=gold,
        projection=fixture_projection(),
        emitted=(emitted,),
        gold_targets=(target,),
    )
    assert result.unrepresentable_gold_count == 1
    assert result.predicted_decisions != result.gold_decisions
    assert "lack_contextual_type" in result.issues[0].reason


def test_invalid_projection_scores_semantics_itt_and_structures_fail_closed() -> None:
    bundle = score_projection(
        fixture_projection(accepted=False),
        configuration=metric_configuration(),
        alignment_plan=empty_alignment_plan(),
        gold_decisions=(),
        valid_evidence_ids=frozenset(),
        grounding_by_assertion_id={},
    )
    assert not bundle.structural.valid_content_bearing
    assert bundle.structural.entropy is None
    assert bundle.structural.leiden_base is None
    structural_rows = {
        row.metric_name: row for row in bundle.metric_rows if "entropy" in row.metric_name
    }
    assert structural_rows
    assert all(row.status is PipelineMetricStatus.INVALID for row in structural_rows.values())


def test_mathematically_undefined_entropy_rows_are_not_marked_not_applicable() -> None:
    projection = fixture_projection()
    empty_graph = projection.instance_graph.model_copy(
        update={"assertions": (), "proposition_contents": ()}
    )
    bundle = score_projection(
        projection.model_copy(update={"instance_graph": empty_graph, "decisions": ()}),
        configuration=metric_configuration(),
        alignment_plan=empty_alignment_plan(),
        gold_decisions=(),
        valid_evidence_ids=frozenset(),
        grounding_by_assertion_id={},
        compute_community=False,
    )
    rows = {item.metric_name: item for item in bundle.metric_rows}
    for name in (
        "degree_mass_entropy",
        "degree_mass_denominator",
        "native_schema_relation_entropy",
        "native_schema_relation_denominator",
        "canonical_mapped_relation_entropy",
        "canonical_mapped_relation_denominator",
        "canonical_other_rate",
    ):
        assert rows[name].status is PipelineMetricStatus.UNDEFINED
        assert rows[name].value is None
    assert rows["active_native_relation_count"].value == 0
    assert rows["assertion_edge_count"].value == 0


def test_node_empty_projection_is_failed_itt_output_with_no_geometry_metrics() -> None:
    projection = fixture_projection()
    empty_graph = projection.instance_graph.model_copy(
        update={
            "entities": (),
            "events": (),
            "proposition_contents": (),
            "assertions": (),
        }
    )
    empty_projection = projection.model_copy(
        update={"instance_graph": empty_graph, "decisions": ()}
    )
    scorer_plan = ScorerMetricPlan(
        plan_id="empty-output-scorer-plan",
        source_gold_hash="b" * 64,
        alignment_plan=empty_alignment_plan(),
        gold_decisions=(),
        rare_annotations=(),
        valid_evidence_ids=(),
        valid_evidence_manifest_hash=canonical_sha256(()),
    )
    intended = IntendedMetricUnit(
        unit_id="empty-output-intended",
        job_id="empty-output-job",
        condition=ConditionName.C2_LLM_QUERY,
        world_id="world-empty",
        context_id="context-empty",
        seed_block=1,
        snapshot_hash="d" * 64,
        packet_hash="e" * 64,
        context_hash="f" * 64,
        scorer_plan_hash=scorer_plan.content_hash,
        relevant_node_gold_count=0,
        strict_assertion_gold_count=0,
        ontology_decision_gold_count=0,
        rare_pivotal_gold_count=0,
    )

    assert not projection_is_content_bearing(empty_projection)
    score = score_intended_projection(
        intended,
        empty_projection,
        context=None,  # type: ignore[arg-type] -- must return before semantic inputs
        evidence_packet=None,  # type: ignore[arg-type]
        configuration=metric_configuration(),
        scorer_plan=scorer_plan,
        grounding_audit=None,  # type: ignore[arg-type]
        geometry=None,
    )

    assert score.output_valid is False
    assert score.failure_kind is OutputFailureKind.VALIDATION_INVALID
    assert score.failure_artifact_hash == HASH
    density = next(item for item in score.rows if item.metric_name == "density")
    assert density.status is PipelineMetricStatus.INVALID
    assert density.value is None


def test_contrast_invariants_and_collapse_summary_expose_denominators() -> None:
    adapter = adapt_projection_for_metrics(fixture_projection(), metric_configuration())
    assertion = adapter.assertion_semantics[0]
    invariant = GoldContrastInvariant(
        invariant_id="invariant-1",
        anchor_ids=assertion.anchor_ids[:1],
        evidence_ids=assertion.evidence_ids,
        expected_signature=assertion.semantic_signature,
    )
    invariant_score = score_contrast_invariants(
        invariants=(invariant,),
        before_assertions=adapter.assertion_semantics,
        after_assertions=adapter.assertion_semantics,
    )
    assert invariant_score.preservation_rate.numerator == 1
    assert invariant_score.preservation_rate.denominator == 1
    wrong_invariant = invariant.model_copy(update={"expected_signature": "wrong-final-state"})
    wrong_score = score_contrast_invariants(
        invariants=(wrong_invariant,),
        before_assertions=adapter.assertion_semantics,
        after_assertions=adapter.assertion_semantics,
    )
    assert wrong_score.preserved_in_both_count == 0
    duplicated_target = invariant.model_copy(update={"invariant_id": "invariant-2"})
    one_to_one = score_contrast_invariants(
        invariants=(invariant, duplicated_target),
        before_assertions=(assertion,),
        after_assertions=(assertion,),
    )
    assert one_to_one.preserved_in_both_count == 1
    assert one_to_one.invariant_count == 2

    eligible_collapse = score_contrastive_changes(gold=(), predicted=())
    # Empty-gold pairs are explicit but excluded from the collapse denominator.
    summary = aggregate_contrastive_collapse((eligible_collapse,))
    assert summary.pair_count == 1
    assert summary.eligible_pair_count == 0
    assert summary.collapse_rate.value is None


def test_cross_condition_completeness_and_rare_denominators_are_enforced() -> None:
    metric_rows = (
        MetricObservation("C1", "w", "a", 1, "m", 1.0),
        MetricObservation("C2", "w", "b", 1, "m", 1.0),
    )
    with pytest.raises(ValueError, match="world/context cells differ"):
        validate_cross_condition_metric_design(
            metric_rows,
            metric_names=("m",),
            conditions=("C1", "C2"),
        )
    rare_rows = (
        RarePivotalCountObservation("C1", "w", "a", 1, 1, 1),
        RarePivotalCountObservation("C2", "w", "a", 1, 1, 2),
    )
    with pytest.raises(ValueError, match="bindings differ across conditions"):
        validate_cross_condition_rare_design(rare_rows, conditions=("C1", "C2"))


def test_failed_output_is_materialized_from_frozen_intended_manifest() -> None:
    alignment = AlignmentPlan(
        matcher_revision="matcher-v1",
        source_gold_hash="b" * 64,
        source_alternative_set_hash="c" * 64,
        node_targets=(
            NodeAlignmentTarget(
                target_id="gold-node",
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                permissible_anchor_sets=(("mention-1",),),
            ),
        ),
        assertion_targets=(),
    )
    gold_decision = NormalizedDecision(
        slot_key="abstraction/root",
        family=DecisionFamily.ABSTRACTION,
        operator=ConstructionOperator.ABSTRACTION,
        anchor_ids=("mention-1",),
        semantic_signature="a" * 64,
    )
    scorer_plan = ScorerMetricPlan(
        plan_id="scorer-plan",
        source_gold_hash="b" * 64,
        alignment_plan=alignment,
        gold_decisions=(gold_decision,),
        rare_annotations=(),
        valid_evidence_ids=(),
        valid_evidence_manifest_hash=canonical_sha256(()),
    )
    intended = IntendedMetricUnit(
        unit_id="intended-1",
        job_id="job-1",
        condition=ConditionName.C2_LLM_QUERY,
        world_id="world-1",
        context_id="context-1",
        seed_block=1,
        snapshot_hash="d" * 64,
        packet_hash="e" * 64,
        context_hash="f" * 64,
        scorer_plan_hash=scorer_plan.content_hash,
        relevant_node_gold_count=1,
        strict_assertion_gold_count=0,
        ontology_decision_gold_count=1,
        rare_pivotal_gold_count=0,
    )
    score = score_failed_output(
        FailedMetricOutput(
            intended_unit=intended,
            failure_kind=OutputFailureKind.TIMEOUT,
            allocated_gpu_seconds=150,
        ),
        configuration=metric_configuration(),
        scorer_plan=scorer_plan,
    )
    rows = {item.metric_name: item for item in score.rows}
    assert not score.output_valid
    assert rows["contextual_node_f1"].value == 0
    assert rows["ontology_decision_macro_f1"].value == 0
    assert rows["density"].status is PipelineMetricStatus.INVALID
    manifest = IntendedMetricManifest(manifest_id="manifest", units=(intended,))
    metric_rows, rare_rows = analysis_observations_from_scores(
        (score,), intended_manifest=manifest
    )
    assert len(metric_rows) == 2
    assert all(not item.output_valid for item in metric_rows)
    assert len(rare_rows) == 1
    auto_missing = materialize_intention_to_treat_scores(
        manifest,
        successful_scores=(),
        recorded_failures=(),
        scorer_plans_by_hash={scorer_plan.content_hash: scorer_plan},
        configuration=metric_configuration(),
    )
    assert auto_missing[0].failure_kind is OutputFailureKind.MISSING
    assert next(
        item
        for item in auto_missing[0].rows
        if item.metric_name == "ontology_decision_macro_f1"
    ).value == 0
    stale_projection_id = "stale-projection"
    stale_rows = tuple(
        revalidated_copy(
            row,
            projection_id=stale_projection_id,
            metric_version_hash="0" * 64,
        )
        for row in score.rows
    )
    stale_success = revalidated_copy(
        score,
        output_valid=True,
        projection_id=stale_projection_id,
        projection_bundle_hash="1" * 64,
        failure_kind=None,
        failure_artifact_hash=None,
        allocated_gpu_seconds=0.0,
        rows=stale_rows,
    )
    with pytest.raises(ValueError, match="active metric formula/configuration version"):
        materialize_intention_to_treat_scores(
            manifest,
            successful_scores=(stale_success,),
            recorded_failures=(),
            scorer_plans_by_hash={scorer_plan.content_hash: scorer_plan},
            configuration=metric_configuration(),
        )
    with pytest.raises(ValueError, match="score set differs from intended-unit manifest"):
        analysis_observations_from_scores(
            (),
            intended_manifest=manifest,
        )


def test_crossing_comparison_does_not_duplicate_deterministic_c0() -> None:
    rows = []
    for world in range(12):
        for context in range(3):
            rows.append(CrossingObservation("C0", f"w{world}", f"c{context}", None, 1, 2))
            for seed in (1, 2):
                rows.append(
                    CrossingObservation("C2", f"w{world}", f"c{context}", seed, 1, 2)
                )
    result = compare_crossing_profiles(
        rows,
        treatment_condition="C2",
        comparator_condition="C0",
        comparison="C2-vs-C0",
        bootstrap_root_seed=13,
    )
    assert result.expected_output_pair_count == 36
    assert result.available_output_pair_count == 36
    assert result.comparator_pooled_rate.positive_opportunity_output_count == 36
    assert result.treatment_pooled_rate.positive_opportunity_output_count == 72
