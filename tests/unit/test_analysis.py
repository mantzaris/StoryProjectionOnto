from __future__ import annotations

import json

import pytest

from story_projection_onto.analysis import (
    PRIMARY_ORGANIZATION_HYPOTHESIS,
    PRIMARY_SEMANTIC_HYPOTHESIS,
    REGISTERED_BOOTSTRAP_RESAMPLES,
    REGISTERED_SIGN_FLIP_ASSIGNMENTS,
    CrossingObservation,
    MetricObservation,
    RarePivotalCountObservation,
    aggregate_rare_pivotal_world_recalls,
    aggregate_world_means,
    analysis_to_json,
    analyze_paired_worlds,
    compare_crossing_profiles,
    exact_sign_flip,
    holm_adjust,
    mechanism_comparison_is_admissible,
    paired_differences,
    rare_pivotal_noninferiority,
    run_registered_analysis,
    run_registered_analysis_from_observations,
    world_bootstrap_interval,
)


def primary_rows() -> tuple[MetricObservation, ...]:
    rows = []
    for condition in ("C1 LLMPre", "C2 LLMQuery"):
        for world_number in range(1, 13):
            for context_number in range(1, 4):
                for seed in (1, 2):
                    value = 0.5 + 0.01 * world_number
                    if condition == "C2 LLMQuery":
                        value += 0.1
                    rows.append(
                        MetricObservation(
                            condition=condition,
                            world_id=f"world-{world_number:02d}",
                            context_id=f"context-{context_number}",
                            seed_block=seed,
                            metric_name="strict_f1",
                            value=value,
                        )
                    )
    return tuple(rows)


def test_world_aggregation_averages_seed_then_context() -> None:
    means = aggregate_world_means(primary_rows(), metric_name="strict_f1")
    lookup = {(row.condition, row.world_id): row for row in means}

    assert len(means) == 24
    assert lookup[("C1 LLMPre", "world-01")].value == pytest.approx(0.51)
    assert lookup[("C2 LLMQuery", "world-12")].value == pytest.approx(0.72)
    assert all(row.context_count == 3 and row.row_count == 6 for row in means)


def test_world_aggregation_rejects_duplicate_and_unseeded_llm_rows() -> None:
    rows = list(primary_rows())
    with pytest.raises(ValueError, match="one row per seed block"):
        aggregate_world_means((*rows, rows[0]), metric_name="strict_f1")

    first = rows[0]
    rows[0] = MetricObservation(
        condition=first.condition,
        world_id=first.world_id,
        context_id=first.context_id,
        seed_block=None,
        metric_name=first.metric_name,
        value=first.value,
    )
    with pytest.raises(ValueError, match="one row per seed block"):
        aggregate_world_means(rows, metric_name="strict_f1")


def test_rare_pivotal_aggregation_pools_counts_before_averaging_seeds() -> None:
    rows = []
    for seed, true_positives in ((1, (1, 0, 1)), (2, (0, 1, 1))):
        for context, true_positive, gold_count in zip(
            ("a", "b", "c"),
            true_positives,
            (1, 3, 1),
            strict=True,
        ):
            rows.append(
                RarePivotalCountObservation(
                    condition="C2 LLMQuery",
                    world_id="world-01",
                    context_id=context,
                    seed_block=seed,
                    true_positive_count=true_positive,
                    gold_count=gold_count,
                )
            )
    result = aggregate_rare_pivotal_world_recalls(rows)
    assert len(result) == 1
    assert result[0].value == pytest.approx(0.4)
    assert tuple(item.gold_count for item in result[0].seed_pools) == (5, 5)
    assert result[0].value != pytest.approx(
        sum(row.true_positive_count / row.gold_count for row in rows) / len(rows)
    )


def test_rare_pivotal_aggregation_zeros_invalid_rows_and_does_not_duplicate_c0() -> None:
    rows = tuple(
        RarePivotalCountObservation(
            condition="C0 ClassicalPre",
            world_id="world-01",
            context_id=f"context-{index}",
            seed_block=None,
            true_positive_count=1,
            gold_count=1,
            output_valid=index != 2,
        )
        for index in range(1, 4)
    )
    result = aggregate_rare_pivotal_world_recalls(rows)
    assert result[0].value == pytest.approx(2 / 3)
    assert result[0].seed_pools[0].seed_block is None

    with pytest.raises(ValueError, match="repeats a context"):
        aggregate_rare_pivotal_world_recalls((*rows, rows[0]))


def test_rare_pivotal_aggregation_requires_seed_invariant_gold_denominators() -> None:
    rows = []
    for seed in (1, 2):
        for context in range(1, 4):
            rows.append(
                RarePivotalCountObservation(
                    condition="C2",
                    world_id="world-01",
                    context_id=f"context-{context}",
                    seed_block=seed,
                    true_positive_count=0,
                    gold_count=1 + int(seed == 2 and context == 1),
                )
            )
    with pytest.raises(ValueError, match="gold denominators differ"):
        aggregate_rare_pivotal_world_recalls(rows)


def test_invalid_semantic_output_is_zero_but_structural_na_stays_undefined() -> None:
    rows = []
    for context in range(1, 4):
        for seed in (1, 2):
            rows.append(
                MetricObservation(
                    condition="C2 LLMQuery",
                    world_id="world-01",
                    context_id=f"context-{context}",
                    seed_block=seed,
                    metric_name="strict_f1",
                    value=None,
                    output_valid=False,
                    gold_nonempty=True,
                )
            )
    semantic = aggregate_world_means(
        rows,
        metric_name="strict_f1",
        require_complete_design=True,
    )
    structural = aggregate_world_means(
        rows,
        metric_name="strict_f1",
        invalid_with_nonempty_gold_scores_zero=False,
        require_complete_design=True,
    )

    assert semantic[0].value == 0
    assert structural[0].value is None
    assert structural[0].undefined_context_count == 3


def test_deterministic_condition_is_not_duplicated_across_seeds() -> None:
    rows = tuple(
        MetricObservation(
            condition="C0 ClassicalPre",
            world_id="world-01",
            context_id=f"context-{context}",
            seed_block=None,
            metric_name="strict_f1",
            value=0.4 + context / 10,
        )
        for context in range(1, 4)
    )
    result = aggregate_world_means(rows, metric_name="strict_f1")
    assert result[0].value == pytest.approx(0.6)

    duplicated = (*rows, rows[0])
    with pytest.raises(ValueError, match="one unseeded row"):
        aggregate_world_means(duplicated, metric_name="strict_f1")

    canonical = tuple(
        MetricObservation(
            condition="C0",
            world_id=row.world_id,
            context_id=row.context_id,
            seed_block=None,
            metric_name=row.metric_name,
            value=row.value,
        )
        for row in rows
    )
    assert aggregate_world_means(canonical, metric_name="strict_f1")[0].value == pytest.approx(0.6)


def test_exact_sign_flip_enumerates_all_4096_world_assignments() -> None:
    result = exact_sign_flip(tuple(range(1, 13)))
    assert result.assignment_count == REGISTERED_SIGN_FLIP_ASSIGNMENTS
    assert result.one_sided_p_value == pytest.approx(1 / 4096)
    assert result.two_sided_p_value == pytest.approx(2 / 4096)
    assert "symmetric" in result.exchangeability_assumption


def test_world_bootstrap_is_deterministic_and_uses_10000_resamples() -> None:
    differences = tuple(value / 100 for value in range(1, 13))
    first = world_bootstrap_interval(differences, seed=20260903)
    second = world_bootstrap_interval(differences, seed=20260903)
    assert first == second
    assert first.resamples == 10_000
    assert first.unit_count == 12
    assert first.lower < sum(differences) / 12 < first.upper
    assert first.unit == "world"

    with pytest.raises(ValueError, match="exactly 10000"):
        world_bootstrap_interval(differences, seed=20260903, resamples=9_999)
    with pytest.raises(ValueError, match=r"confidence must equal 0\.95"):
        world_bootstrap_interval(differences, seed=20260903, confidence=0.9)


def test_registered_paired_analysis_reports_world_level_outputs() -> None:
    comparator = {f"world-{index:02d}": 0.4 + index / 100 for index in range(1, 13)}
    treatment = {
        world: value + (0.03 if index % 3 else 0.01)
        for index, (world, value) in enumerate(comparator.items(), start=1)
    }
    result = analyze_paired_worlds(
        treatment,
        comparator,
        comparison="C2-C1",
        metric_name="strict_qualified_assertion_f1",
        bootstrap_seed=42,
    )

    assert result.world_count == 12
    assert result.degrees_of_freedom == 11
    assert len(result.differences) == 12
    assert result.two_sided_ci_lower < result.mean_difference < result.two_sided_ci_upper
    assert result.one_sided_superiority_p_value < 0.05
    assert result.small_sample_correction == pytest.approx(0.9299598099757785)
    assert result.small_sample_corrected_effect_gz < result.paired_effect_dz
    assert result.sign_flip.assignment_count == 4096


def test_holm_controls_exactly_supplied_family_and_mechanism_gate() -> None:
    results = holm_adjust({"semantic": 0.03, "organization": 0.01})
    by_name = {result.hypothesis: result for result in results}
    assert by_name["organization"].adjusted_p_value == pytest.approx(0.02)
    assert by_name["semantic"].adjusted_p_value == pytest.approx(0.03)
    assert all(result.rejected for result in results)
    assert mechanism_comparison_is_admissible(
        results,
        organization_hypothesis="organization",
    )

    blocked = holm_adjust({"semantic": 0.01, "organization": 0.06})
    assert not mechanism_comparison_is_admissible(
        blocked,
        organization_hypothesis="organization",
    )

    with pytest.raises(ValueError, match="exactly two"):
        holm_adjust({"semantic": 0.01})
    with pytest.raises(ValueError, match="exactly two"):
        holm_adjust({"semantic": 0.01, "organization": 0.02, "extra": 0.03})


def test_rare_noninferiority_requires_lower_bound_strictly_above_margin() -> None:
    comparator = {f"world-{index:02d}": 0.7 for index in range(1, 13)}
    treatment = {
        world: value + difference
        for (world, value), difference in zip(
            comparator.items(),
            (-0.03, -0.02, -0.01, 0.0) * 3,
            strict=True,
        )
    }
    result = rare_pivotal_noninferiority(
        treatment,
        comparator,
        comparison="C2-C1",
        bootstrap_seed=99,
    )
    assert result.one_sided_95_lower_bound > -0.05
    assert result.passes
    assert result.one_sided_noninferiority_p_value < 0.05
    assert result.bootstrap.resamples == REGISTERED_BOOTSTRAP_RESAMPLES
    assert result.bootstrap_lower_exceeds_margin


def test_rare_noninferiority_strict_boundary_does_not_pass() -> None:
    comparator = {f"world-{index:02d}": 0.7 for index in range(1, 13)}
    treatment = {world: value - 0.05 for world, value in comparator.items()}
    result = rare_pivotal_noninferiority(
        treatment,
        comparator,
        comparison="C2-C1",
        bootstrap_seed=100,
    )
    assert result.one_sided_95_lower_bound == pytest.approx(-0.05)
    assert result.one_sided_noninferiority_p_value == pytest.approx(0.5)
    assert not result.passes
    assert not result.bootstrap_lower_exceeds_margin


def test_paired_analysis_rejects_missing_or_undefined_worlds() -> None:
    treatment = {f"world-{index:02d}": 0.5 for index in range(1, 13)}
    comparator = dict(treatment)
    comparator.pop("world-12")
    with pytest.raises(ValueError, match="different worlds"):
        paired_differences(treatment, comparator)

    comparator["world-12"] = None
    with pytest.raises(ValueError, match="undefined"):
        paired_differences(treatment, comparator)


def test_zero_variance_effect_is_explicit_and_strict_json_serializable() -> None:
    comparator = {f"world-{index:02d}": 0.4 for index in range(1, 13)}
    treatment = {world: value + 0.1 for world, value in comparator.items()}
    result = analyze_paired_worlds(
        treatment,
        comparator,
        comparison="C2-C1",
        metric_name="strict_f1",
        bootstrap_seed=1,
    )
    assert result.paired_effect_dz is None
    assert result.small_sample_corrected_effect_gz is None
    assert result.t_statistic is None
    assert result.degeneracy == "zero_variance_nonzero_mean"
    assert result.one_sided_superiority_p_value == 0
    payload = json.loads(analysis_to_json(result))
    assert payload["paired_effect_dz"] is None


def test_all_zero_differences_have_explicit_degenerate_conventions() -> None:
    values = {f"world-{index:02d}": 0.4 for index in range(1, 13)}
    result = analyze_paired_worlds(
        values,
        values,
        comparison="C2-C1",
        metric_name="strict_f1",
        bootstrap_seed=2,
    )
    assert result.degeneracy == "zero_variance_zero_mean"
    assert result.t_statistic is None
    assert result.paired_effect_dz is None
    assert result.one_sided_superiority_p_value == 0.5
    assert result.two_sided_p_value == 1
    assert result.sign_flip.one_sided_p_value == 1
    assert result.sign_flip.two_sided_p_value == 1


def _world_panel(*, c0: float, c1: float, c2: float, fixed: float) -> dict[str, dict[str, float]]:
    return {
        condition: {f"world-{world:02d}": baseline + (world % 4) * 0.001 for world in range(1, 13)}
        for condition, baseline in (
            ("C0", c0),
            ("C1", c1),
            ("C2", c2),
            ("A-FixedSelect", fixed),
        )
    }


def test_registered_analysis_enforces_two_primary_tests_secondary_and_gated_mechanism() -> None:
    result = run_registered_analysis(
        strict_qualified_assertion_f1=_world_panel(c0=0.40, c1=0.50, c2=0.65, fixed=0.52),
        ontology_decision_macro_f1=_world_panel(c0=0.35, c1=0.45, c2=0.63, fixed=0.48),
        rare_pivotal_qualified_assertion_recall=_world_panel(
            c0=0.60,
            c1=0.70,
            c2=0.72,
            fixed=0.70,
        ),
        bootstrap_root_seed=901,
    )

    assert {row.hypothesis for row in result.primary_holm} == {
        PRIMARY_SEMANTIC_HYPOTHESIS,
        PRIMARY_ORGANIZATION_HYPOTHESIS,
    }
    assert len(result.primary_holm) == 2
    assert all(row.rejected for row in result.primary_holm)
    assert result.semantic.secondary_c2_vs_c0.comparison == "C2-C0"
    assert result.organization.secondary_c2_vs_c0.comparison == "C2-C0"
    assert result.mechanism_status == "tested"
    assert result.mechanism_c2_vs_fixed_select is not None
    assert result.rare_c2_vs_fixed_select is not None
    assert result.mechanism_statistical_rare_gate_passes
    assert result.semantic.primary_c2_vs_c1.sign_flip.assignment_count == 4096
    assert result.semantic.primary_c2_vs_c1.bootstrap.resamples == 10_000
    json.loads(analysis_to_json(result))


def test_registered_analysis_does_not_open_fixed_select_gate_when_organization_fails() -> None:
    organization = _world_panel(c0=0.35, c1=0.50, c2=0.50, fixed=0.1)
    organization.pop("A-FixedSelect")
    rare = _world_panel(c0=0.60, c1=0.70, c2=0.72, fixed=0.1)
    rare.pop("A-FixedSelect")
    result = run_registered_analysis(
        strict_qualified_assertion_f1=_world_panel(c0=0.40, c1=0.50, c2=0.65, fixed=0.52),
        ontology_decision_macro_f1=organization,
        rare_pivotal_qualified_assertion_recall=rare,
        bootstrap_root_seed=902,
    )
    assert result.mechanism_status == "not_tested_organization_gate_failed"
    assert result.mechanism_c2_vs_fixed_select is None
    assert result.rare_c2_vs_fixed_select is None
    assert not result.mechanism_statistical_rare_gate_passes


def test_ledger_facing_registered_analysis_applies_complete_world_aggregation() -> None:
    metric_rows = []
    rare_rows = []
    baselines = {"C0": 0.35, "C1": 0.45, "C2": 0.65, "A-FixedSelect": 0.50}
    for condition, baseline in baselines.items():
        seeds = (None,) if condition == "C0" else (1, 2)
        for world in range(1, 13):
            for context in range(1, 4):
                for seed in seeds:
                    for metric_name in (
                        "strict_qualified_assertion_f1",
                        "ontology_decision_macro_f1",
                    ):
                        metric_rows.append(
                            MetricObservation(
                                condition=condition,
                                world_id=f"world-{world:02d}",
                                context_id=f"context-{context}",
                                seed_block=seed,
                                metric_name=metric_name,
                                value=baseline + context / 100 + (seed or 0) / 1000,
                            )
                        )
                    rare_rows.append(
                        RarePivotalCountObservation(
                            condition=condition,
                            world_id=f"world-{world:02d}",
                            context_id=f"context-{context}",
                            seed_block=seed,
                            true_positive_count=(
                                context if condition in {"C2", "A-FixedSelect"} else context - 1
                            ),
                            gold_count=context,
                        )
                    )
    result = run_registered_analysis_from_observations(
        metric_rows,
        rare_rows,
        bootstrap_root_seed=903,
    )
    assert result.semantic.primary_c2_vs_c1.mean_difference == pytest.approx(0.20)
    assert result.organization.primary_c2_vs_c1.world_count == 12
    assert result.rare_c2_vs_c1.mean_difference == pytest.approx(0.5)


def _crossing_rows() -> list[CrossingObservation]:
    rows = []
    for world in range(1, 13):
        for context in range(1, 4):
            for seed in (1, 2):
                rows.extend(
                    (
                        CrossingObservation(
                            condition="C2",
                            world_id=f"world-{world:02d}",
                            context_id=f"context-{context}",
                            seed_block=seed,
                            crossing_count=1,
                            opportunity_count=2,
                        ),
                        CrossingObservation(
                            condition="C1",
                            world_id=f"world-{world:02d}",
                            context_id=f"context-{context}",
                            seed_block=seed,
                            crossing_count=1,
                            opportunity_count=4,
                        ),
                    )
                )
    return rows


def test_crossing_comparison_uses_world_units_and_reports_exposure_sensitivity() -> None:
    result = compare_crossing_profiles(
        _crossing_rows(),
        treatment_condition="C2",
        comparator_condition="C1",
        comparison="C2-C1",
        bootstrap_root_seed=77,
    )
    assert result.expected_output_pair_count == 72
    assert result.available_output_pair_count == 72
    assert result.both_positive_pair_count == 72
    assert result.conditional_rate_retained_world_count == 12
    assert result.conditional_rate is not None
    assert result.conditional_rate.world_count == 12
    assert result.conditional_rate.mean_difference == pytest.approx(0.25)
    assert result.opportunity_incidence is not None
    assert result.opportunity_incidence.mean_difference == 0
    assert result.opportunity_count is not None
    assert result.opportunity_count.mean_difference == pytest.approx(-2)
    assert result.treatment_pooled_rate.crossing_count == 72
    assert result.treatment_pooled_rate.opportunity_count == 144
    assert result.treatment_pooled_rate.rate == pytest.approx(0.5)
    assert result.comparator_pooled_rate.rate == pytest.approx(0.25)
    assert result.pooled_rate_difference == pytest.approx(0.25)
    assert result.analysis_warning is None


def test_crossing_comparison_reports_discordant_zero_and_unavailable_pairs() -> None:
    rows = _crossing_rows()

    def replace(
        condition: str,
        world_id: str,
        context_id: str,
        seed: int,
        crossing: int | None,
        opportunity: int | None,
    ) -> None:
        for index, row in enumerate(rows):
            if (
                row.condition,
                row.world_id,
                row.context_id,
                row.seed_block,
            ) == (condition, world_id, context_id, seed):
                rows[index] = CrossingObservation(
                    condition=condition,
                    world_id=world_id,
                    context_id=context_id,
                    seed_block=seed,
                    crossing_count=crossing,
                    opportunity_count=opportunity,
                )
                return
        raise AssertionError("fixture row not found")

    replace("C2", "world-01", "context-1", 1, 0, 0)
    replace("C1", "world-02", "context-1", 1, 0, 0)
    replace("C2", "world-03", "context-1", 1, 0, 0)
    replace("C1", "world-03", "context-1", 1, 0, 0)
    replace("C2", "world-04", "context-1", 1, None, None)
    replace("C2", "world-04", "context-1", 2, 1, 10)
    result = compare_crossing_profiles(
        rows,
        treatment_condition="C2",
        comparator_condition="C1",
        comparison="C2-C1",
        bootstrap_root_seed=78,
    )
    assert result.treatment_only_positive_pair_count == 1
    assert result.comparator_only_positive_pair_count == 1
    assert result.neither_positive_pair_count == 1
    assert result.unavailable_output_pair_count == 1
    assert result.both_positive_pair_count == 68
    assert result.conditional_rate_retained_world_count == 12
    assert result.opportunity_count is not None
    world_four_index = result.opportunity_count.world_ids.index("world-04")
    assert result.opportunity_count.differences[world_four_index] == pytest.approx(2 / 3)
    assert result.analysis_warning is not None
    assert "unavailable geometry" in result.analysis_warning


def test_paired_inputs_reject_nan_and_infinite_values() -> None:
    finite = {f"world-{index:02d}": 0.5 for index in range(1, 13)}
    nonfinite = dict(finite)
    nonfinite["world-01"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_differences(nonfinite, finite)
    nonfinite["world-01"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        paired_differences(nonfinite, finite)
