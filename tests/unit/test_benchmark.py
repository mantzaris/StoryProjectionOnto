from __future__ import annotations

import itertools
import json
from collections import Counter

import pytest

from story_projection_onto.benchmark import (
    AlternativeField,
    ANoContextSelectionManifest,
    BenchmarkFactor,
    CompilerPolicy,
    Difficulty,
    HorizonBlock,
    IndependentReviewGateError,
    IndependentReviewResponse,
    IndependentReviewResponseItem,
    LensFamily,
    ReviewAdjudication,
    ReviewDisposition,
    ReviewLifecycleError,
    SeedPurpose,
    StoryScopeBlock,
    bind_review_response,
    build_seed_manifest,
    compile_alignment_alternatives,
    compile_benchmark,
    compile_gold_projection,
    derive_seed,
    derive_signed_changes,
    evaluate_alternative_set,
    finalize_reviewed_seal,
    require_independent_review_complete,
    reviewed_artifact_from_original,
    sample_held_out_queries,
    scientific_audit,
    seed_for,
    semantic_atom_to_normalized_decision,
    validate_adjudication,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    BenchmarkSplit,
    ConstructionOperator,
    EpistemicAttitude,
    GoldReviewStatus,
    NarrativeCommitment,
)
from story_projection_onto.metrics.alignment import build_alignment_plan
from story_projection_onto.metrics.contrastive import (
    derive_signed_changes as derive_metric_changes,
)
from story_projection_onto.metrics.contrastive import (
    normalize_gold_contrast,
)


@pytest.fixture(scope="module")
def benchmark_build():
    return compile_benchmark()


def test_seed_derivation_is_stable_complete_and_namespaced(benchmark_build) -> None:
    manifest = benchmark_build.seeds
    assert {item.purpose for item in manifest.entries} == set(SeedPurpose)
    assert len({item.seed for item in manifest.entries}) == len(SeedPurpose)
    entry = next(item for item in manifest.entries if item.purpose is SeedPurpose.WORLD)
    assert entry.seed == derive_seed(manifest.root_seed, entry.namespace)
    assert seed_for(manifest, SeedPurpose.WORLD) == entry.seed
    held_out = [
        item for item in benchmark_build.world_specs if item.split is BenchmarkSplit.HELD_OUT
    ]
    changed_seed_audit = sample_held_out_queries(
        held_out, build_seed_manifest(manifest.root_seed + 1)
    )
    assert tuple(item.lenses for item in changed_seed_audit.assignments) != tuple(
        item.lenses for item in benchmark_build.query_sampling_audit.assignments
    )
    assert set(benchmark_build.draft_seal.compiler_dependency_hashes) == {
        "synthetic_benchmark.py",
        "benchmark_runtime.py",
        "contracts.py",
        "metrics/alignment.py",
    }


def test_registered_splits_strata_and_blocked_query_allocation(benchmark_build) -> None:
    held_out = [
        item for item in benchmark_build.world_specs if item.split is BenchmarkSplit.HELD_OUT
    ]
    development = [
        item for item in benchmark_build.world_specs if item.split is BenchmarkSplit.DEVELOPMENT
    ]
    assert len(development) == 4
    assert len(held_out) == 12
    assert Counter(item.difficulty for item in held_out) == Counter(
        {item: 4 for item in Difficulty}
    )
    assert Counter(item.horizon_block for item in held_out) == {
        HorizonBlock.INTERMEDIATE: 6,
        HorizonBlock.FINAL: 6,
    }
    assignments = benchmark_build.query_sampling_audit.assignments
    assert Counter(lens for item in assignments for lens in item.lenses) == {
        item: 6 for item in LensFamily
    }
    assert Counter(scope for item in assignments for scope in item.story_scopes) == {
        item: 12 for item in StoryScopeBlock
    }
    assert Counter(level for item in assignments for level in item.abstractions) == {
        item: 12 for item in AbstractionLevel
    }
    assert Counter(item.node_budget for item in assignments) == {10: 4, 15: 4, 20: 4}
    assert all(len(set(item.lenses)) == 3 for item in assignments)
    allocation_by_world = {
        item.world_id: item for item in benchmark_build.query_sampling_audit.conditional_allocations
    }
    assert set(allocation_by_world) == {item.world_id for item in assignments}
    assert all(
        item.lenses in allocation_by_world[item.world_id].eligible_lens_orders
        and item.story_scopes in allocation_by_world[item.world_id].eligible_scope_orders
        and item.abstractions in allocation_by_world[item.world_id].eligible_abstraction_orders
        and item.viewpoint_context_index
        in allocation_by_world[item.world_id].eligible_viewpoint_indices
        for item in assignments
    )
    context_lookup = {
        context.context_id: (spec.world_id, spec.difficulty, context)
        for spec in held_out
        for context in benchmark_build.contexts_by_world[spec.world_id]
    }
    eligibility = benchmark_build.eligibility
    assert len({context_lookup[item][0] for item in eligibility.rare_guard_context_ids}) == 8
    assert (
        len({context_lookup[item][0] for item in eligibility.temporal_epistemic_context_ids}) == 8
    )
    assert {context_lookup[item][1] for item in eligibility.temporal_case_context_ids} == set(
        Difficulty
    )
    assert {context_lookup[item][1] for item in eligibility.holder_status_context_ids} == set(
        Difficulty
    )
    assert all(
        context_lookup[item][2].viewpoint is None for item in eligibility.temporal_case_context_ids
    )
    assert all(
        context_lookup[item][2].viewpoint is not None
        for item in eligibility.holder_status_context_ids
    )


def test_held_out_structures_schema_grounding_and_temporal_coordinates(benchmark_build) -> None:
    held_out = [
        item for item in benchmark_build.world_specs if item.split is BenchmarkSplit.HELD_OUT
    ]
    development = [
        item for item in benchmark_build.world_specs if item.split is BenchmarkSplit.DEVELOPMENT
    ]
    assert len({item.structure_template_id for item in held_out}) == 12
    assert {item.structure_template_id for item in held_out}.isdisjoint(
        item.structure_template_id for item in development
    )
    coordinate_differences = 0
    for spec in held_out:
        model = benchmark_build.model_artifacts[spec.world_id]
        evidence_by_id = {item.evidence_id: item for item in model.evidence}
        narrative = benchmark_build.narratives[spec.world_id]
        coordinate_differences += sum(
            (left.discourse_position.passage_order - right.discourse_position.passage_order)
            * (
                narrative.revelation_order_by_evidence_id[left.evidence_id]
                - narrative.revelation_order_by_evidence_id[right.evidence_id]
            )
            < 0
            for left, right in itertools.combinations(model.evidence, 2)
        )
        for projection in benchmark_build.scorer_artifacts[spec.world_id].gold_projections:
            evidence_by_predicate = {
                predicate.predicate_id: set(predicate.evidence_ids)
                for predicate in projection.local_schema.predicates
            }
            for assertion in projection.qualified_assertions:
                assert set(assertion.evidence_ids) <= evidence_by_predicate[assertion.predicate_id]
            for local_type in projection.local_schema.contextual_types:
                kind = local_type.parent_upper_type.removeprefix("upper.")
                assert all(
                    any(
                        candidate.provisional_type == f"{kind}_candidate"
                        for candidate in evidence_by_id[evidence_id].mention_candidates
                    )
                    or (kind == "event" and evidence_by_id[evidence_id].event_candidates)
                    for evidence_id in local_type.evidence_ids
                )
    assert coordinate_differences > 0


def test_context_fields_causally_change_compiled_semantics(benchmark_build) -> None:
    for spec in (
        item for item in benchmark_build.world_specs if item.split is BenchmarkSplit.HELD_OUT
    ):
        scorer = benchmark_build.scorer_artifacts[spec.world_id]
        contexts = benchmark_build.contexts_by_world[spec.world_id]
        narrative = benchmark_build.narratives[spec.world_id]
        base_context = contexts[0]
        base = compile_gold_projection(
            spec,
            base_context,
            0,
            narrative,
            benchmark_build.configuration,
            policy=CompilerPolicy.QUERY_DEPENDENT,
            selected_for_review=False,
        )
        alternate_scope = contexts[
            next(
                index for index in (1, 2) if contexts[index].story_scope != base_context.story_scope
            )
        ].story_scope
        scope_changed = compile_gold_projection(
            spec,
            base_context.model_copy(update={"story_scope": alternate_scope}),
            0,
            narrative,
            benchmark_build.configuration,
            policy=CompilerPolicy.QUERY_DEPENDENT,
            selected_for_review=False,
        )
        assert next(
            item.signature
            for item in base.semantic_atoms
            if item.slot_key == "qualification/story-scope"
        ) != next(
            item.signature
            for item in scope_changed.semantic_atoms
            if item.slot_key == "qualification/story-scope"
        )
        changed_by_id = {
            item.assertion_id: item for item in scope_changed.projection.qualified_assertions
        }
        assert any(
            item.temporal_scope != changed_by_id[item.assertion_id].temporal_scope
            for item in base.projection.qualified_assertions
        )
        alternate_abstraction = next(
            item for item in AbstractionLevel if item is not base_context.abstraction
        )
        abstraction_changed = compile_gold_projection(
            spec,
            base_context.model_copy(update={"abstraction": alternate_abstraction}),
            0,
            narrative,
            benchmark_build.configuration,
            policy=CompilerPolicy.QUERY_DEPENDENT,
            selected_for_review=False,
        )
        assert (
            base.projection.local_schema.abstraction
            != abstraction_changed.projection.local_schema.abstraction
        )
        viewpoint_context = contexts[scorer.query_assignment.viewpoint_context_index]
        viewpoint_framed = compile_gold_projection(
            spec,
            viewpoint_context,
            scorer.query_assignment.viewpoint_context_index,
            narrative,
            benchmark_build.configuration,
            policy=CompilerPolicy.QUERY_DEPENDENT,
            selected_for_review=False,
        )
        viewpoint_unframed = compile_gold_projection(
            spec,
            viewpoint_context.model_copy(update={"viewpoint": None}),
            scorer.query_assignment.viewpoint_context_index,
            narrative,
            benchmark_build.configuration,
            policy=CompilerPolicy.QUERY_DEPENDENT,
            selected_for_review=False,
        )
        assert next(
            item.signature
            for item in viewpoint_framed.semantic_atoms
            if item.slot_key == "qualification/epistemic-frame"
        ) != next(
            item.signature
            for item in viewpoint_unframed.semantic_atoms
            if item.slot_key == "qualification/epistemic-frame"
        )
        if BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE in spec.factors:
            framed_denial = next(
                item
                for item in viewpoint_framed.projection.qualified_assertions
                if item.epistemic_scope is not None
                and item.epistemic_scope.attitude is EpistemicAttitude.DENIED
            )
            unframed_denial = next(
                item
                for item in viewpoint_unframed.projection.qualified_assertions
                if item.assertion_id == framed_denial.assertion_id
            )
            assert framed_denial.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED
            assert unframed_denial.narrative_commitment is NarrativeCommitment.CONTESTED


def test_signed_contrasts_are_exact_compiler_diffs_and_grounded(benchmark_build) -> None:
    for world_id, scorer in benchmark_build.scorer_artifacts.items():
        model = benchmark_build.model_artifacts[world_id]
        contexts = benchmark_build.contexts_by_world[world_id]
        expected = derive_signed_changes(
            scorer.semantic_atoms_by_query[contexts[0].context_id],
            scorer.semantic_atoms_by_query[contexts[1].context_id],
        )
        assert expected
        assert scorer.gold_projections[0].signed_contrast_decisions == expected
        assert scorer.gold_projections[1].signed_contrast_decisions == expected
        evidence = {item.evidence_id: item for item in model.evidence}
        for decision in expected:
            candidates = {
                candidate
                for evidence_id in decision.evidence_ids
                for candidate in (
                    *(item.candidate_id for item in evidence[evidence_id].mention_candidates),
                    *(item.candidate_id for item in evidence[evidence_id].event_candidates),
                    *(
                        item.candidate_id
                        for item in evidence[evidence_id].relation_phrase_candidates
                    ),
                    *(item.clue_id for item in evidence[evidence_id].temporal_clues),
                )
            }
            assert set(decision.anchor_ids) <= candidates
        metric_expected = derive_metric_changes(
            tuple(
                semantic_atom_to_normalized_decision(item)
                for item in scorer.semantic_atoms_by_query[contexts[0].context_id]
            ),
            tuple(
                semantic_atom_to_normalized_decision(item)
                for item in scorer.semantic_atoms_by_query[contexts[1].context_id]
            ),
        )
        assert normalize_gold_contrast(expected) == metric_expected


def test_identity_and_event_representation_are_evidence_defensible(benchmark_build) -> None:
    merge_world = next(
        item
        for item in benchmark_build.world_specs
        if item.split is BenchmarkSplit.HELD_OUT and item.contrast_family.value == "merge_split"
    )
    scorer = benchmark_build.scorer_artifacts[merge_world.world_id]
    context_a, context_b, _ = benchmark_build.contexts_by_world[merge_world.world_id]
    atoms_a = {item.slot_key: item for item in scorer.semantic_atoms_by_query[context_a.context_id]}
    atoms_b = {item.slot_key: item for item in scorer.semantic_atoms_by_query[context_b.context_id]}
    assert atoms_a["identity/title-continuity"].signature == "partition:separate"
    assert atoms_b["identity/title-continuity"].signature == "partition:co-clustered"
    assert {
        item.operator
        for item in derive_signed_changes(tuple(atoms_a.values()), tuple(atoms_b.values()))
    } >= {ConstructionOperator.MERGE, ConstructionOperator.SPLIT}

    event_world = next(
        item
        for item in benchmark_build.world_specs
        if item.split is BenchmarkSplit.HELD_OUT
        and item.contrast_family.value == "event_reification"
    )
    event_scorer = benchmark_build.scorer_artifacts[event_world.world_id]
    event_contexts = benchmark_build.contexts_by_world[event_world.world_id]
    event_a, event_b = event_scorer.gold_projections[:2]
    assert len(event_b.events) == len(event_a.events)
    representations = tuple(
        next(
            item.signature
            for item in event_scorer.semantic_atoms_by_query[context.context_id]
            if item.slot_key == "event/focal-representation"
        )
        for context in event_contexts[:2]
    )
    assert set(representations) == {
        "qualified-nary-event-relation",
        "event-object-with-qualified-roles",
    }
    focal_label = next(item.label for item in event_world.events if item.focal)
    by_representation = dict(zip(representations, (event_a, event_b), strict=True))
    assert focal_label not in {
        item.label for item in by_representation["qualified-nary-event-relation"].events
    }
    assert focal_label in {
        item.label for item in by_representation["event-object-with-qualified-roles"].events
    }


def test_registered_world_fields_are_consumed_and_holder_truth_is_safe(benchmark_build) -> None:
    for scorer in benchmark_build.scorer_artifacts.values():
        spec = scorer.world_spec
        atoms = [item for values in scorer.semantic_atoms_by_query.values() for item in values]
        assert {item.slot_key for item in atoms if item.slot_key.startswith("causal/")}
        assert len(
            {item.slot_key for item in atoms if item.slot_key.startswith("event-predecessor/")}
        ) == sum(item.causal_predecessor_ref is not None for item in spec.events)
        assert len(
            {item.slot_key for item in atoms if item.slot_key.startswith("home-collective/")}
        ) == len(spec.personas)
        has_conflict = any(item.slot_key.startswith("conflict/") for item in atoms)
        assert has_conflict == (BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE in spec.factors)
        for projection in scorer.gold_projections:
            attributed = [
                item for item in projection.qualified_assertions if item.epistemic_scope is not None
            ]
            assert attributed
            assert all(
                item.narrative_commitment
                in {NarrativeCommitment.HOLDER_ATTRIBUTED, NarrativeCommitment.CONTESTED}
                for item in attributed
            )
        viewpoint_index = scorer.query_assignment.viewpoint_context_index
        viewpoint_projection = scorer.gold_projections[viewpoint_index]
        viewpoint_relevance = {
            item.target_id: item.is_relevant for item in viewpoint_projection.relevance
        }
        relevant_attributed = [
            item
            for item in viewpoint_projection.qualified_assertions
            if item.epistemic_scope is not None and viewpoint_relevance[item.assertion_id]
        ]
        assert relevant_attributed
        assert all(
            item.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED
            for item in relevant_attributed
        )
        if BenchmarkFactor.INCOMPLETE_CONFLICTING_EVIDENCE in spec.factors:
            assert any(
                item.epistemic_scope is not None
                and item.epistemic_scope.attitude is EpistemicAttitude.DENIED
                for item in relevant_attributed
            )


def test_rare_paths_are_connected_and_every_deletion_is_pivotal(benchmark_build) -> None:
    proofs = benchmark_build.mutation_manifest.rare_deletion_proofs
    assert len(proofs) >= 8
    assert {
        item.world_id
        for item in benchmark_build.world_specs
        if item.split is BenchmarkSplit.HELD_OUT
    } <= {item.world_id for item in proofs}
    for proof in benchmark_build.mutation_manifest.rare_deletion_proofs:
        assert proof.before_answer_hash != proof.after_answer_hash
        assert proof.before_regenerated_bundle_hash != proof.after_regenerated_bundle_hash
        assert "answer_facts" in proof.changed_components
        assert proof.impact_kind.value in proof.changed_components
    for scorer in benchmark_build.scorer_artifacts.values():
        contexts = benchmark_build.contexts_by_world[scorer.world_spec.world_id]
        for context, projection in zip(contexts, scorer.gold_projections, strict=True):
            assertions = {item.assertion_id: item for item in projection.qualified_assertions}
            rare = next(item for item in projection.assertion_annotations if item.is_rare)
            path = scorer.rare_support_paths_by_query[context.context_id]
            assert rare.is_pivotal == (path is not None)
            if path is None:
                assert rare.support_path_assertion_ids == ()
                continue
            assert rare.support_path_assertion_ids[0] == rare.assertion_id
            assert len(path.steps) == 2
            assert (
                tuple((item.source_fact_id, item.target_fact_id) for item in path.steps)
                == scorer.world_spec.causal_dependencies
            )
            assert path.steps[0].target_fact_id == path.steps[1].source_fact_id
            assert all(
                step.source_assertion_id in assertions
                and step.causal_assertion_id in assertions
                and step.target_assertion_id in assertions
                for step in path.steps
            )


def test_fixed_friendly_and_null_cases_are_real_compiler_modes(benchmark_build) -> None:
    held_out_scorers = [
        item
        for item in benchmark_build.scorer_artifacts.values()
        if item.world_spec.split is BenchmarkSplit.HELD_OUT
    ]
    fixed = [
        item
        for item in held_out_scorers
        if BenchmarkFactor.FIXED_ONTOLOGY_FRIENDLY in item.world_spec.factors
    ]
    assert len(fixed) >= 3
    assert all(
        item.query_assignment.compiler_policies[2] is CompilerPolicy.FIXED_REFERENCE
        for item in fixed
    )
    assert all(
        item.fixed_friendly_proof is not None
        and item.fixed_friendly_proof.fixed_reference_semantic_hash
        == item.fixed_friendly_proof.query_dependent_semantic_hash
        and item.fixed_friendly_proof.query_id
        == benchmark_build.contexts_by_world[item.world_spec.world_id][2].context_id
        for item in fixed
    )
    assert all(
        any(annotation.is_relevant for annotation in item.gold_projections[2].relevance)
        for item in fixed
    )
    nulls = [item.null_case_proof for item in held_out_scorers if item.null_case_proof]
    assert len(nulls) == 3
    assert all(
        item.base_nonselection_signature == item.perturbed_nonselection_signature for item in nulls
    )
    assert {item.perturbation for item in nulls} == {"wording-only-under-fixed-reference"}


def test_executable_alternatives_are_consumed_and_reject_wrong_holder(benchmark_build) -> None:
    scorer = next(iter(benchmark_build.scorer_artifacts.values()))
    for gold, alternatives in zip(scorer.gold_projections, scorer.alternatives, strict=True):
        compiled = compile_alignment_alternatives(gold, alternatives)
        assert set(compiled.consumed_alternative_ids) == {
            item.alternative_id for item in alternatives.constraint_alternatives
        }
        plan = build_alignment_plan(
            gold,
            alternatives,
            compiled_alternatives=compiled,
        )
        assert plan.executed_alternative_ids == compiled.consumed_alternative_ids
        candidate = next(
            item
            for item in benchmark_build.scorer_artifacts[
                scorer.world_spec.world_id
            ].gold_projections
            if item.gold_projection_id == gold.gold_projection_id
        )
        from story_projection_onto.synthetic_benchmark import _alternative_view

        view = _alternative_view(candidate)
        assert evaluate_alternative_set(alternatives, view).matched
        constraints = {
            item.field_path: item for item in alternatives.constraint_alternatives[0].constraints
        }
        joint = constraints[AlternativeField.JOINT_REPRESENTATION.value]
        assert len(constraints) == 1
        alternate = json.loads(joint.accepted_values[1])
        equivalent = type(view)(
            **alternate,
            joint_representation_signature=joint.accepted_values[1],
        )
        assert evaluate_alternative_set(alternatives, equivalent).matched
        hybrid = view.model_copy(update={"assertion_signature": equivalent.assertion_signature})
        assert not evaluate_alternative_set(alternatives, hybrid).matched
        wrong_holder = view.model_copy(
            update={"epistemic_signature": view.epistemic_signature + ":wrong-holder"}
        )
        assert not evaluate_alternative_set(alternatives, wrong_holder).matched


def test_review_lifecycle_binds_exact_ids_and_pending_seal_blocks(benchmark_build) -> None:
    package = benchmark_build.review_package
    templates = [
        item
        for world in package.worlds
        for projection in world.projections
        for item in projection.review_items
    ]
    assert len(templates) == 72
    response = IndependentReviewResponse(
        package_hash=package.content_hash,
        reviewer_pseudonym="contract-fixture-only",
        reviewed_at=benchmark_build.configuration.frozen_at,
        items=tuple(
            IndependentReviewResponseItem(
                review_item_id=item.review_item_id,
                blind_projection_id=item.blind_projection_id,
                criterion=item.criterion,
                disposition=ReviewDisposition.AGREE,
                notes="contract fixture; not a scientific review",
            )
            for item in templates
        ),
    )
    assert len(bind_review_response(package, response)) == 72
    altered = response.items[0].model_copy(update={"review_item_id": "arbitrary_response_id"})
    invalid = response.model_copy(update={"items": (altered, *response.items[1:])})
    with pytest.raises(ReviewLifecycleError):
        bind_review_response(package, invalid)
    with pytest.raises(IndependentReviewGateError):
        require_independent_review_complete(
            benchmark_build.draft_seal, None, None, None, None, None, None
        )
    adjudication = ReviewAdjudication(
        package_hash=package.content_hash,
        response_hash=response.content_hash,
        adjudicator_pseudonym="contract-fixture-adjudicator",
        adjudicated_at=benchmark_build.configuration.frozen_at,
        items=(),
    )
    same_reviewer_and_adjudicator = adjudication.model_copy(
        update={"adjudicator_pseudonym": response.reviewer_pseudonym.upper()}
    )
    validate_adjudication(package, response, same_reviewer_and_adjudicator)
    reviewed_artifacts = []
    for binding in benchmark_build.review_bindings.entries:
        scorer = benchmark_build.scorer_artifacts[binding.world_id]
        gold = next(item for item in scorer.gold_projections if item.query_id == binding.query_id)
        alternatives = next(
            item
            for item in scorer.alternatives
            if item.gold_projection_id == gold.gold_projection_id
        )
        reviewed_artifacts.append(
            reviewed_artifact_from_original(
                binding=binding,
                package=package,
                response=response,
                adjudication=adjudication,
                gold=gold,
                alternatives=alternatives,
            )
        )
    with pytest.raises(ReviewLifecycleError):
        finalize_reviewed_seal(
            benchmark_build.draft_seal,
            package,
            benchmark_build.review_bindings,
            response,
            adjudication,
            (*reviewed_artifacts[:-1], reviewed_artifacts[0]),
        )
    fixture_final_seal = finalize_reviewed_seal(
        benchmark_build.draft_seal,
        package,
        benchmark_build.review_bindings,
        response,
        adjudication,
        reviewed_artifacts,
    )
    require_independent_review_complete(
        benchmark_build.draft_seal,
        package,
        benchmark_build.review_bindings,
        response,
        adjudication,
        reviewed_artifacts,
        fixture_final_seal,
    )
    selected = {item.world_id for item in benchmark_build.review_selection.selected}
    assert len(selected) == 3
    assert all(
        projection.review_status is GoldReviewStatus.PENDING
        for world_id in selected
        for projection in benchmark_build.scorer_artifacts[world_id].gold_projections
    )


def test_no_context_selection_is_frozen_and_exactly_balanced(benchmark_build) -> None:
    manifest: ANoContextSelectionManifest = benchmark_build.no_context_selection
    assert len(manifest.entries) == 12
    assert Counter(item.lens for item in manifest.entries) == {item: 2 for item in LensFamily}
    assert Counter(item.difficulty for item in manifest.entries) == {item: 4 for item in Difficulty}


def test_scientific_audit_regresses_all_measured_failures(benchmark_build) -> None:
    audit = scientific_audit(benchmark_build)
    assert audit.invalid_contrast_anchor_count == 0  # rejected candidate: 32/32 invalid
    assert audit.disconnected_rare_support_path_count == 0  # rejected candidate: 36/36
    assert audit.relevant_event_outside_scope_count == 0  # rejected candidate: 25 events
    assert audit.arbitrary_review_id_rejected
    assert benchmark_build.rejected_candidate.condition_outputs_generated is False
