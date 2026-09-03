"""Registered world-level paired analyses for the conference study.

Contexts, seeds, nodes, assertions, and edges are never treated as independent
experimental units here.  Callers must first materialize one value per condition and
world with :func:`aggregate_world_means`.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np
from scipy import stats

REGISTERED_WORLD_COUNT = 12
REGISTERED_CONTEXTS_PER_WORLD = 3
REGISTERED_LLM_SEEDS = (1, 2)
REGISTERED_BOOTSTRAP_RESAMPLES = 10_000
REGISTERED_SIGN_FLIP_ASSIGNMENTS = 2**REGISTERED_WORLD_COUNT
RARE_PIVOTAL_NONINFERIORITY_MARGIN = -0.05


@dataclass(frozen=True)
class MetricObservation:
    """One intended condition/world/context/seed result row.

    A missing, timed-out, invalid, or unrepaired semantic output must still have a
    row.  For registered semantic failure metrics, ``gold_nonempty=True`` maps that
    row to zero.  Structurally undefined metrics remain ``None`` and are not made
    favorable by empty output.
    """

    condition: str
    world_id: str
    context_id: str
    seed_block: int | None
    metric_name: str
    value: float | None
    output_valid: bool = True
    gold_nonempty: bool = True

    def __post_init__(self) -> None:
        for name in ("condition", "world_id", "context_id", "metric_name"):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and stripped")
        if self.seed_block is not None and self.seed_block <= 0:
            raise ValueError("seed_block must be positive when present")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric value must be finite or None")


@dataclass(frozen=True)
class WorldMean:
    condition: str
    world_id: str
    metric_name: str
    value: float | None
    context_count: int
    row_count: int
    undefined_context_count: int


@dataclass(frozen=True)
class SignFlipResult:
    assignment_count: int
    observed_mean_difference: float
    one_sided_p_value: float
    two_sided_p_value: float
    exchangeability_assumption: str


@dataclass(frozen=True)
class BootstrapInterval:
    resamples: int
    lower: float
    upper: float
    seed: int
    unit: str = "world"
    warning: str = "Only 12 independent worlds; percentile bootstrap is sensitivity analysis."


@dataclass(frozen=True)
class PairedEstimate:
    comparison: str
    metric_name: str
    world_ids: tuple[str, ...]
    differences: tuple[float, ...]
    world_count: int
    mean_difference: float
    median_difference: float
    standard_deviation_difference: float
    standard_error: float
    degrees_of_freedom: int
    t_statistic: float
    one_sided_superiority_p_value: float
    two_sided_p_value: float
    two_sided_ci_lower: float
    two_sided_ci_upper: float
    paired_effect_dz: float
    small_sample_corrected_effect_gz: float
    sign_flip: SignFlipResult
    bootstrap: BootstrapInterval


@dataclass(frozen=True)
class HolmResult:
    hypothesis: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool
    rank: int


@dataclass(frozen=True)
class NoninferiorityResult:
    comparison: str
    metric_name: str
    margin: float
    mean_difference: float
    one_sided_95_lower_bound: float
    passes: bool
    world_count: int
    bootstrap: BootstrapInterval


def _semantic_value(
    row: MetricObservation,
    *,
    invalid_with_nonempty_gold_scores_zero: bool,
) -> float | None:
    if not row.output_valid and row.gold_nonempty and invalid_with_nonempty_gold_scores_zero:
        return 0.0
    return row.value


def aggregate_world_means(
    observations: Iterable[MetricObservation],
    *,
    metric_name: str,
    deterministic_conditions: frozenset[str] = frozenset({"C0 ClassicalPre"}),
    expected_contexts_per_world: int = REGISTERED_CONTEXTS_PER_WORLD,
    expected_llm_seeds: tuple[int, ...] = REGISTERED_LLM_SEEDS,
    invalid_with_nonempty_gold_scores_zero: bool = True,
    require_complete_design: bool = True,
) -> tuple[WorldMean, ...]:
    """Average seeds within context, then contexts within each independent world."""

    filtered = tuple(row for row in observations if row.metric_name == metric_name)
    if not filtered:
        raise ValueError(f"no observations for metric {metric_name!r}")
    grouped: dict[tuple[str, str], list[MetricObservation]] = defaultdict(list)
    for row in filtered:
        grouped[(row.condition, row.world_id)].append(row)

    results: list[WorldMean] = []
    for (condition, world_id), rows in sorted(grouped.items()):
        by_context: dict[str, list[MetricObservation]] = defaultdict(list)
        for row in rows:
            by_context[row.context_id].append(row)
        if require_complete_design and len(by_context) != expected_contexts_per_world:
            raise ValueError(
                f"{condition}/{world_id} has {len(by_context)} contexts; "
                f"expected {expected_contexts_per_world}"
            )

        context_values: list[float] = []
        undefined_contexts = 0
        for context_id, context_rows in sorted(by_context.items()):
            seed_ids = tuple(sorted(row.seed_block for row in context_rows if row.seed_block))
            if condition in deterministic_conditions:
                if len(context_rows) != 1 or context_rows[0].seed_block is not None:
                    raise ValueError(
                        f"deterministic {condition}/{world_id}/{context_id} must have "
                        "one unseeded row"
                    )
            elif require_complete_design and seed_ids != expected_llm_seeds:
                raise ValueError(
                    f"{condition}/{world_id}/{context_id} seed blocks {seed_ids!r} "
                    f"do not equal {expected_llm_seeds!r}"
                )
            values = tuple(
                value
                for row in context_rows
                if (
                    value := _semantic_value(
                        row,
                        invalid_with_nonempty_gold_scores_zero=(
                            invalid_with_nonempty_gold_scores_zero
                        ),
                    )
                )
                is not None
            )
            if not values:
                undefined_contexts += 1
                continue
            if len(values) != len(context_rows):
                raise ValueError(
                    f"{condition}/{world_id}/{context_id} mixes defined and undefined rows"
                )
            context_values.append(float(np.mean(values)))

        if not context_values or len(context_values) != len(by_context):
            world_value = None
        else:
            world_value = float(np.mean(context_values))
        results.append(
            WorldMean(
                condition=condition,
                world_id=world_id,
                metric_name=metric_name,
                value=world_value,
                context_count=len(by_context),
                row_count=len(rows),
                undefined_context_count=undefined_contexts,
            )
        )
    return tuple(results)


def paired_differences(
    treatment: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
    *,
    require_world_count: int | None = REGISTERED_WORLD_COUNT,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if set(treatment) != set(comparator):
        missing_treatment = sorted(set(comparator) - set(treatment))
        missing_comparator = sorted(set(treatment) - set(comparator))
        raise ValueError(
            "paired conditions have different worlds: "
            f"missing treatment={missing_treatment}, missing comparator={missing_comparator}"
        )
    world_ids = tuple(sorted(treatment))
    if require_world_count is not None and len(world_ids) != require_world_count:
        raise ValueError(f"paired analysis requires exactly {require_world_count} worlds")
    undefined = tuple(
        world_id
        for world_id in world_ids
        if treatment[world_id] is None or comparator[world_id] is None
    )
    if undefined:
        raise ValueError(f"paired endpoint is undefined for worlds: {', '.join(undefined)}")
    differences = tuple(
        float(treatment[world_id]) - float(comparator[world_id])  # type: ignore[arg-type]
        for world_id in world_ids
    )
    return world_ids, differences


def exact_sign_flip(differences: Sequence[float]) -> SignFlipResult:
    """Enumerate all sign assignments; for 12 worlds this is exactly 4,096."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("sign-flip differences must be a nonempty finite vector")
    if len(values) > 20:
        raise ValueError("exact sign-flip enumeration is bounded to at most 20 units")
    observed = float(np.mean(values))
    means = np.fromiter(
        (
            float(np.mean(values * signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ),
        dtype=float,
        count=2 ** len(values),
    )
    tolerance = 1e-15
    return SignFlipResult(
        assignment_count=len(means),
        observed_mean_difference=observed,
        one_sided_p_value=float(np.mean(means >= observed - tolerance)),
        two_sided_p_value=float(np.mean(np.abs(means) >= abs(observed) - tolerance)),
        exchangeability_assumption=(
            "World-level paired differences are symmetric/exchangeable under sign reversal."
        ),
    )


def world_bootstrap_interval(
    differences: Sequence[float],
    *,
    seed: int,
    resamples: int = REGISTERED_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
) -> BootstrapInterval:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap differences must be a nonempty finite vector")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[indexes].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, (tail, 1 - tail), method="linear")
    return BootstrapInterval(
        resamples=resamples,
        lower=float(lower),
        upper=float(upper),
        seed=seed,
    )


def _paired_summary(differences: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired analysis requires at least two finite world differences")
    count = len(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    standard_error = standard_deviation / math.sqrt(count)
    degrees = count - 1
    if standard_error == 0:
        statistic = math.copysign(math.inf, mean) if mean != 0 else math.nan
        one_sided = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
        two_sided = 0.0 if mean != 0 else 1.0
        lower = upper = mean
        effect = math.copysign(math.inf, mean) if mean != 0 else 0.0
    else:
        statistic = mean / standard_error
        one_sided = float(stats.t.sf(statistic, degrees))
        two_sided = float(2 * stats.t.sf(abs(statistic), degrees))
        critical = float(stats.t.ppf(0.975, degrees))
        lower = mean - critical * standard_error
        upper = mean + critical * standard_error
        effect = mean / standard_deviation
    correction = 1 - 3 / (4 * degrees - 1)
    return {
        "count": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "degrees": degrees,
        "statistic": statistic,
        "one_sided": one_sided,
        "two_sided": two_sided,
        "lower": lower,
        "upper": upper,
        "effect": effect,
        "corrected_effect": effect * correction,
    }


def analyze_paired_worlds(
    treatment: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
    *,
    comparison: str,
    metric_name: str,
    bootstrap_seed: int,
    require_world_count: int | None = REGISTERED_WORLD_COUNT,
) -> PairedEstimate:
    world_ids, differences = paired_differences(
        treatment,
        comparator,
        require_world_count=require_world_count,
    )
    summary = _paired_summary(differences)
    return PairedEstimate(
        comparison=comparison,
        metric_name=metric_name,
        world_ids=world_ids,
        differences=differences,
        world_count=int(summary["count"]),
        mean_difference=float(summary["mean"]),
        median_difference=float(median(differences)),
        standard_deviation_difference=float(summary["standard_deviation"]),
        standard_error=float(summary["standard_error"]),
        degrees_of_freedom=int(summary["degrees"]),
        t_statistic=float(summary["statistic"]),
        one_sided_superiority_p_value=float(summary["one_sided"]),
        two_sided_p_value=float(summary["two_sided"]),
        two_sided_ci_lower=float(summary["lower"]),
        two_sided_ci_upper=float(summary["upper"]),
        paired_effect_dz=float(summary["effect"]),
        small_sample_corrected_effect_gz=float(summary["corrected_effect"]),
        sign_flip=exact_sign_flip(differences),
        bootstrap=world_bootstrap_interval(differences, seed=bootstrap_seed),
    )


def holm_adjust(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> tuple[HolmResult, ...]:
    """Holm step-down correction; the confirmatory caller supplies exactly two tests."""

    if not p_values:
        raise ValueError("Holm adjustment requires at least one hypothesis")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    for hypothesis, value in p_values.items():
        if not hypothesis or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("hypothesis names and p-values must be valid")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running_adjusted = 0.0
    rejection_open = True
    results: list[HolmResult] = []
    for index, (hypothesis, raw) in enumerate(ordered):
        rank = index + 1
        adjusted = min(1.0, max(running_adjusted, (count - index) * raw))
        running_adjusted = adjusted
        threshold = alpha / (count - index)
        rejected = rejection_open and raw <= threshold
        if not rejected:
            rejection_open = False
        results.append(
            HolmResult(
                hypothesis=hypothesis,
                raw_p_value=raw,
                adjusted_p_value=adjusted,
                rejected=rejected,
                rank=rank,
            )
        )
    return tuple(sorted(results, key=lambda result: result.hypothesis))


def rare_pivotal_noninferiority(
    treatment: Mapping[str, float | None],
    comparator: Mapping[str, float | None],
    *,
    comparison: str,
    bootstrap_seed: int,
    margin: float = RARE_PIVOTAL_NONINFERIORITY_MARGIN,
) -> NoninferiorityResult:
    _, differences = paired_differences(treatment, comparator)
    summary = _paired_summary(differences)
    critical = float(stats.t.ppf(0.95, int(summary["degrees"])))
    lower = float(summary["mean"]) - critical * float(summary["standard_error"])
    return NoninferiorityResult(
        comparison=comparison,
        metric_name="rare_pivotal_qualified_assertion_recall",
        margin=margin,
        mean_difference=float(summary["mean"]),
        one_sided_95_lower_bound=lower,
        passes=lower > margin,
        world_count=int(summary["count"]),
        bootstrap=world_bootstrap_interval(differences, seed=bootstrap_seed),
    )


def mechanism_comparison_is_admissible(
    holm_results: Sequence[HolmResult],
    *,
    organization_hypothesis: str,
) -> bool:
    matches = tuple(
        result for result in holm_results if result.hypothesis == organization_hypothesis
    )
    if len(matches) != 1:
        raise ValueError("organization hypothesis must identify exactly one Holm result")
    return matches[0].rejected


__all__ = [
    "RARE_PIVOTAL_NONINFERIORITY_MARGIN",
    "REGISTERED_BOOTSTRAP_RESAMPLES",
    "REGISTERED_CONTEXTS_PER_WORLD",
    "REGISTERED_LLM_SEEDS",
    "REGISTERED_SIGN_FLIP_ASSIGNMENTS",
    "REGISTERED_WORLD_COUNT",
    "BootstrapInterval",
    "HolmResult",
    "MetricObservation",
    "NoninferiorityResult",
    "PairedEstimate",
    "SignFlipResult",
    "WorldMean",
    "aggregate_world_means",
    "analyze_paired_worlds",
    "exact_sign_flip",
    "holm_adjust",
    "mechanism_comparison_is_admissible",
    "paired_differences",
    "rare_pivotal_noninferiority",
    "world_bootstrap_interval",
]
