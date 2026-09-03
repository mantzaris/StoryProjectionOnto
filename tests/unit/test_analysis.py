from __future__ import annotations

import math

import pytest

from story_projection_onto.analysis import (
    REGISTERED_SIGN_FLIP_ASSIGNMENTS,
    MetricObservation,
    aggregate_world_means,
    analyze_paired_worlds,
    exact_sign_flip,
    holm_adjust,
    mechanism_comparison_is_admissible,
    paired_differences,
    rare_pivotal_noninferiority,
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
    assert first.lower < sum(differences) / 12 < first.upper
    assert first.unit == "world"


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


def test_paired_analysis_rejects_missing_or_undefined_worlds() -> None:
    treatment = {f"world-{index:02d}": 0.5 for index in range(1, 13)}
    comparator = dict(treatment)
    comparator.pop("world-12")
    with pytest.raises(ValueError, match="different worlds"):
        paired_differences(treatment, comparator)

    comparator["world-12"] = None
    with pytest.raises(ValueError, match="undefined"):
        paired_differences(treatment, comparator)


def test_zero_variance_effect_is_explicit_not_silently_regularized() -> None:
    comparator = {f"world-{index:02d}": 0.4 for index in range(1, 13)}
    treatment = {world: value + 0.1 for world, value in comparator.items()}
    result = analyze_paired_worlds(
        treatment,
        comparator,
        comparison="C2-C1",
        metric_name="strict_f1",
        bootstrap_seed=1,
    )
    assert math.isinf(result.paired_effect_dz)
    assert result.one_sided_superiority_p_value == 0
