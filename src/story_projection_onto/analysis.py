"""Registered world-level paired analyses for the conference study.

Contexts, seeds, nodes, assertions, and edges are never treated as independent
experimental units here.  Callers must first materialize one value per condition and
world with :func:`aggregate_world_means`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from statistics import median
from typing import Any, Literal

import numpy as np
from scipy import special, stats

REGISTERED_WORLD_COUNT = 12
REGISTERED_CONTEXTS_PER_WORLD = 3
REGISTERED_LLM_SEEDS = (1, 2)
REGISTERED_BOOTSTRAP_RESAMPLES = 10_000
REGISTERED_SIGN_FLIP_ASSIGNMENTS = 2**REGISTERED_WORLD_COUNT
RARE_PIVOTAL_NONINFERIORITY_MARGIN = -0.05
PRIMARY_SEMANTIC_METRIC = "strict_qualified_assertion_f1"
PRIMARY_ORGANIZATION_METRIC = "ontology_decision_macro_f1"
PRIMARY_SEMANTIC_HYPOTHESIS = "H1-semantic"
PRIMARY_ORGANIZATION_HYPOTHESIS = "H1-organization"
REGISTERED_ALPHA = 0.05


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
    gold_count: int | None = None
    scorer_plan_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("condition", "world_id", "context_id", "metric_name"):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and stripped")
        if self.seed_block is not None and self.seed_block <= 0:
            raise ValueError("seed_block must be positive when present")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric value must be finite or None")
        if self.gold_count is not None:
            if self.gold_count < 0:
                raise ValueError("gold_count must be nonnegative")
            if self.gold_nonempty != (self.gold_count > 0):
                raise ValueError("gold_nonempty must agree with gold_count")
        if self.scorer_plan_hash is not None and (
            len(self.scorer_plan_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.scorer_plan_hash)
        ):
            raise ValueError("scorer_plan_hash must be a lowercase SHA-256 digest")


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
class RarePivotalCountObservation:
    """One context/seed contribution to the registered count-pooled safeguard."""

    condition: str
    world_id: str
    context_id: str
    seed_block: int | None
    true_positive_count: int
    gold_count: int
    output_valid: bool = True
    scorer_plan_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("condition", "world_id", "context_id"):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and stripped")
        if self.seed_block is not None and self.seed_block <= 0:
            raise ValueError("seed_block must be positive when present")
        if self.gold_count < 0 or self.true_positive_count < 0:
            raise ValueError("rare-pivotal counts must be nonnegative")
        if self.true_positive_count > self.gold_count:
            raise ValueError("rare-pivotal true positives cannot exceed gold count")
        if self.scorer_plan_hash is not None and (
            len(self.scorer_plan_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.scorer_plan_hash)
        ):
            raise ValueError("scorer_plan_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class SeedPooledRareRecall:
    seed_block: int | None
    true_positive_count: int
    gold_count: int
    recall: float


@dataclass(frozen=True)
class RarePivotalWorldRecall:
    condition: str
    world_id: str
    value: float
    seed_pools: tuple[SeedPooledRareRecall, ...]
    context_count: int
    row_count: int


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
    unit_count: int
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
    t_statistic: float | None
    one_sided_superiority_p_value: float
    two_sided_p_value: float
    two_sided_ci_lower: float
    two_sided_ci_upper: float
    paired_effect_dz: float | None
    small_sample_correction: float | None
    small_sample_corrected_effect_gz: float | None
    degeneracy: Literal["none", "zero_variance_nonzero_mean", "zero_variance_zero_mean"]
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
    one_sided_noninferiority_p_value: float
    passes: bool
    world_count: int
    bootstrap: BootstrapInterval
    bootstrap_lower_exceeds_margin: bool


@dataclass(frozen=True)
class RegisteredEndpointResult:
    """Primary C2--C1 estimate and required C2--C0 secondary estimate."""

    metric_name: str
    primary_c2_vs_c1: PairedEstimate
    secondary_c2_vs_c0: PairedEstimate


@dataclass(frozen=True)
class RegisteredAnalysisResult:
    """Complete registered confirmatory family and its guarded mechanism test."""

    semantic: RegisteredEndpointResult
    organization: RegisteredEndpointResult
    primary_holm: tuple[HolmResult, HolmResult]
    rare_c2_vs_c1: NoninferiorityResult
    rare_secondary_c2_vs_c0: PairedEstimate
    mechanism_status: Literal["tested", "not_tested_organization_gate_failed"]
    mechanism_c2_vs_fixed_select: PairedEstimate | None
    rare_c2_vs_fixed_select: NoninferiorityResult | None
    semantic_statistical_rare_gate_passes: bool
    organization_statistical_rare_gate_passes: bool
    mechanism_statistical_rare_gate_passes: bool
    alpha: float
    claim_boundary: str = (
        "Statistical and rare-pivotal gates only; grounding, unsupported-assertion, "
        "construction-certificate, and lineage gates remain separate requirements."
    )
    independent_unit: Literal["world"] = "world"
    registered_world_count: int = REGISTERED_WORLD_COUNT
    registered_bootstrap_resamples: int = REGISTERED_BOOTSTRAP_RESAMPLES
    registered_sign_flip_assignments: int = REGISTERED_SIGN_FLIP_ASSIGNMENTS


@dataclass(frozen=True)
class CrossingObservation:
    """Crossing geometry for one condition/world/context/seed output."""

    condition: str
    world_id: str
    context_id: str
    seed_block: int | None
    crossing_count: int | None
    opportunity_count: int | None

    def __post_init__(self) -> None:
        for name in ("condition", "world_id", "context_id"):
            value = getattr(self, name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty and stripped")
        if self.seed_block is not None and self.seed_block <= 0:
            raise ValueError("seed_block must be positive when present")
        if (self.crossing_count is None) != (self.opportunity_count is None):
            raise ValueError("crossing and opportunity counts must be jointly numeric or NA")
        if self.crossing_count is None:
            return
        assert self.opportunity_count is not None
        if self.crossing_count < 0 or self.opportunity_count < 0:
            raise ValueError("crossing and opportunity counts must be nonnegative")
        if self.crossing_count > self.opportunity_count:
            raise ValueError("crossing count cannot exceed opportunity count")


@dataclass(frozen=True)
class ExposureWeightedCrossingRate:
    crossing_count: int
    opportunity_count: int
    rate: float | None
    positive_opportunity_output_count: int


@dataclass(frozen=True)
class WorldRetainedPairCount:
    world_id: str
    available_pair_count: int
    both_positive_pair_count: int


@dataclass(frozen=True)
class CrossingComparisonResult:
    """Registered two-part crossing comparison for two seeded conditions."""

    comparison: str
    treatment_condition: str
    comparator_condition: str
    expected_output_pair_count: int
    available_output_pair_count: int
    unavailable_output_pair_count: int
    both_positive_pair_count: int
    treatment_only_positive_pair_count: int
    comparator_only_positive_pair_count: int
    neither_positive_pair_count: int
    world_pair_counts: tuple[WorldRetainedPairCount, ...]
    opportunity_incidence: PairedEstimate | None
    opportunity_count: PairedEstimate | None
    conditional_rate: PairedEstimate | None
    conditional_rate_retained_world_count: int
    conditional_rate_excluded_world_count: int
    treatment_pooled_rate: ExposureWeightedCrossingRate
    comparator_pooled_rate: ExposureWeightedCrossingRate
    pooled_rate_difference: float | None
    analysis_warning: str | None


def _semantic_value(
    row: MetricObservation,
    *,
    invalid_with_nonempty_gold_scores_zero: bool,
) -> float | None:
    if not row.output_valid and row.gold_nonempty and invalid_with_nonempty_gold_scores_zero:
        return 0.0
    return row.value


def validate_cross_condition_metric_design(
    observations: Iterable[MetricObservation],
    *,
    metric_names: Sequence[str],
    conditions: Sequence[str],
    require_scorer_bindings: bool = False,
) -> None:
    """Require the same world/context and scorer-gold cells across conditions."""

    rows = tuple(
        row
        for row in observations
        if row.metric_name in set(metric_names) and row.condition in set(conditions)
    )
    for metric_name in metric_names:
        metric_rows = tuple(row for row in rows if row.metric_name == metric_name)
        panels: dict[
            str, dict[tuple[str, str], tuple[bool, int | None, str | None]]
        ] = {}
        for condition in conditions:
            condition_rows = tuple(row for row in metric_rows if row.condition == condition)
            if not condition_rows:
                raise ValueError(f"missing {metric_name!r} rows for condition {condition!r}")
            cells: dict[
                tuple[str, str], tuple[bool, int | None, str | None]
            ] = {}
            for row in condition_rows:
                key = (row.world_id, row.context_id)
                if require_scorer_bindings and (
                    row.gold_count is None or row.scorer_plan_hash is None
                ):
                    raise ValueError(
                        "registered analysis requires gold_count and scorer_plan_hash "
                        f"for {condition}/{row.world_id}/{row.context_id}"
                    )
                binding = (row.gold_nonempty, row.gold_count, row.scorer_plan_hash)
                previous = cells.setdefault(key, binding)
                if previous != binding:
                    raise ValueError(
                        "scorer/gold binding differs within "
                        f"{condition}/{row.world_id}/{row.context_id}"
                    )
            panels[condition] = cells
        reference_condition = conditions[0]
        reference = panels[reference_condition]
        for condition in conditions[1:]:
            if set(panels[condition]) != set(reference):
                raise ValueError(
                    f"{metric_name!r} world/context cells differ between "
                    f"{reference_condition!r} and {condition!r}"
                )
            if panels[condition] != reference:
                raise ValueError(
                    f"{metric_name!r} scorer/gold bindings differ across conditions"
                )


def validate_cross_condition_rare_design(
    observations: Iterable[RarePivotalCountObservation],
    *,
    conditions: Sequence[str],
    require_scorer_bindings: bool = False,
) -> None:
    """Require identical context identities and rare-gold denominators by condition."""

    rows = tuple(row for row in observations if row.condition in set(conditions))
    panels: dict[str, dict[tuple[str, str], tuple[int, str | None]]] = {}
    for condition in conditions:
        condition_rows = tuple(row for row in rows if row.condition == condition)
        if not condition_rows:
            raise ValueError(f"missing rare-pivotal rows for condition {condition!r}")
        cells: dict[tuple[str, str], tuple[int, str | None]] = {}
        for row in condition_rows:
            key = (row.world_id, row.context_id)
            if require_scorer_bindings and row.scorer_plan_hash is None:
                raise ValueError(
                    "registered rare-pivotal analysis requires scorer_plan_hash "
                    f"for {condition}/{row.world_id}/{row.context_id}"
                )
            binding = (row.gold_count, row.scorer_plan_hash)
            previous = cells.setdefault(key, binding)
            if previous != binding:
                raise ValueError(
                    f"rare scorer/gold binding differs within {condition}/{row.world_id}/"
                    f"{row.context_id}"
                )
        panels[condition] = cells
    reference_condition = conditions[0]
    reference = panels[reference_condition]
    for condition in conditions[1:]:
        if set(panels[condition]) != set(reference):
            raise ValueError(
                f"rare-pivotal world/context cells differ between {reference_condition!r} "
                f"and {condition!r}"
            )
        if panels[condition] != reference:
            raise ValueError("rare-pivotal scorer/gold bindings differ across conditions")


def aggregate_world_means(
    observations: Iterable[MetricObservation],
    *,
    metric_name: str,
    deterministic_conditions: frozenset[str] = frozenset({"C0", "C0 ClassicalPre"}),
    expected_contexts_per_world: int = REGISTERED_CONTEXTS_PER_WORLD,
    expected_llm_seeds: tuple[int, ...] = REGISTERED_LLM_SEEDS,
    invalid_with_nonempty_gold_scores_zero: bool = True,
    require_complete_design: bool = True,
) -> tuple[WorldMean, ...]:
    """Average seeds within context, then contexts within each independent world."""

    if expected_contexts_per_world <= 0:
        raise ValueError("expected_contexts_per_world must be positive")
    if (
        not expected_llm_seeds
        or len(set(expected_llm_seeds)) != len(expected_llm_seeds)
        or any(seed <= 0 for seed in expected_llm_seeds)
        or tuple(sorted(expected_llm_seeds)) != expected_llm_seeds
    ):
        raise ValueError("expected_llm_seeds must be sorted unique positive integers")

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
            seed_ids = tuple(
                sorted(row.seed_block for row in context_rows if row.seed_block is not None)
            )
            if condition in deterministic_conditions:
                if len(context_rows) != 1 or context_rows[0].seed_block is not None:
                    raise ValueError(
                        f"deterministic {condition}/{world_id}/{context_id} must have "
                        "one unseeded row"
                    )
            else:
                if len(seed_ids) != len(set(seed_ids)) or len(seed_ids) != len(context_rows):
                    raise ValueError(
                        f"{condition}/{world_id}/{context_id} must have one row per seed block"
                    )
                if require_complete_design and seed_ids != expected_llm_seeds:
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


def aggregate_rare_pivotal_world_recalls(
    observations: Iterable[RarePivotalCountObservation],
    *,
    deterministic_conditions: frozenset[str] = frozenset({"C0", "C0 ClassicalPre"}),
    expected_contexts_per_world: int = REGISTERED_CONTEXTS_PER_WORLD,
    expected_llm_seeds: tuple[int, ...] = REGISTERED_LLM_SEEDS,
    require_complete_design: bool = True,
) -> tuple[RarePivotalWorldRecall, ...]:
    """Pool counts over a world's contexts per seed, then average seed recalls.

    This is intentionally distinct from :func:`aggregate_world_means`: the registered
    rare safeguard weights contexts by their rare-pivotal denominators.
    """

    if expected_contexts_per_world <= 0:
        raise ValueError("expected_contexts_per_world must be positive")
    if (
        not expected_llm_seeds
        or len(set(expected_llm_seeds)) != len(expected_llm_seeds)
        or any(seed <= 0 for seed in expected_llm_seeds)
        or tuple(sorted(expected_llm_seeds)) != expected_llm_seeds
    ):
        raise ValueError("expected_llm_seeds must be sorted unique positive integers")

    rows = tuple(observations)
    if not rows:
        raise ValueError("rare-pivotal aggregation requires observations")
    grouped: dict[tuple[str, str], list[RarePivotalCountObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.condition, row.world_id)].append(row)

    results: list[RarePivotalWorldRecall] = []
    for (condition, world_id), world_rows in sorted(grouped.items()):
        contexts = {row.context_id for row in world_rows}
        if require_complete_design and len(contexts) != expected_contexts_per_world:
            raise ValueError(
                f"{condition}/{world_id} has {len(contexts)} contexts; "
                f"expected {expected_contexts_per_world}"
            )
        by_seed: dict[int | None, list[RarePivotalCountObservation]] = defaultdict(list)
        for row in world_rows:
            by_seed[row.seed_block].append(row)
        if condition in deterministic_conditions:
            if set(by_seed) != {None}:
                raise ValueError(f"deterministic {condition}/{world_id} must be unseeded")
        elif None in by_seed:
            raise ValueError(f"seeded {condition}/{world_id} cannot contain unseeded rows")
        expected_seed_keys: tuple[int | None, ...] = (
            (None,) if condition in deterministic_conditions else expected_llm_seeds
        )
        if (
            require_complete_design
            and tuple(sorted(by_seed, key=lambda value: -1 if value is None else value))
            != expected_seed_keys
        ):
            raise ValueError(
                f"{condition}/{world_id} seed blocks do not equal {expected_seed_keys!r}"
            )

        ordered_seed_keys = tuple(sorted(by_seed, key=lambda value: -1 if value is None else value))
        if len(by_seed) > 1:
            gold_by_seed = {
                seed: {row.context_id: row.gold_count for row in seed_rows}
                for seed, seed_rows in by_seed.items()
            }
            reference_seed = ordered_seed_keys[0]
            reference = gold_by_seed[reference_seed]
            for seed, gold_by_context in gold_by_seed.items():
                if gold_by_context != reference:
                    raise ValueError(
                        f"{condition}/{world_id} gold denominators differ between "
                        f"seed={reference_seed} and seed={seed}"
                    )

        seed_pools: list[SeedPooledRareRecall] = []
        for seed in expected_seed_keys if require_complete_design else ordered_seed_keys:
            seed_rows = by_seed[seed]
            if len({row.context_id for row in seed_rows}) != len(seed_rows):
                raise ValueError(f"{condition}/{world_id}/seed={seed} repeats a context")
            if require_complete_design and {row.context_id for row in seed_rows} != contexts:
                raise ValueError(f"{condition}/{world_id}/seed={seed} misses a context")
            gold_count = sum(row.gold_count for row in seed_rows)
            if gold_count <= 0:
                raise ValueError(f"{condition}/{world_id}/seed={seed} has no rare-pivotal gold")
            true_positives = sum(
                row.true_positive_count if row.output_valid else 0 for row in seed_rows
            )
            seed_pools.append(
                SeedPooledRareRecall(
                    seed_block=seed,
                    true_positive_count=true_positives,
                    gold_count=gold_count,
                    recall=true_positives / gold_count,
                )
            )
        results.append(
            RarePivotalWorldRecall(
                condition=condition,
                world_id=world_id,
                value=float(np.mean([item.recall for item in seed_pools])),
                seed_pools=tuple(seed_pools),
                context_count=len(contexts),
                row_count=len(world_rows),
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
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("paired endpoint values and differences must be finite")
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
    if resamples != REGISTERED_BOOTSTRAP_RESAMPLES:
        raise ValueError(
            f"registered bootstrap requires exactly {REGISTERED_BOOTSTRAP_RESAMPLES} resamples"
        )
    if confidence != 0.95:
        raise ValueError("registered bootstrap confidence must equal 0.95")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(values), size=(resamples, len(values)))
    means = values[indexes].mean(axis=1)
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, (tail, 1 - tail), method="linear")
    return BootstrapInterval(
        resamples=resamples,
        unit_count=len(values),
        lower=float(lower),
        upper=float(upper),
        seed=seed,
        warning=(
            f"Only {len(values)} independent worlds; percentile bootstrap is sensitivity analysis."
        ),
    )


def _small_sample_correction(degrees_of_freedom: int) -> float | None:
    """Exact Hedges correction for a standardized paired mean difference."""

    if degrees_of_freedom <= 1:
        return None
    log_correction = (
        special.gammaln(degrees_of_freedom / 2)
        - 0.5 * math.log(degrees_of_freedom / 2)
        - special.gammaln((degrees_of_freedom - 1) / 2)
    )
    return float(math.exp(log_correction))


def _paired_summary(differences: Sequence[float]) -> dict[str, float | int | str | None]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired analysis requires at least two finite world differences")
    count = len(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    standard_error = standard_deviation / math.sqrt(count)
    degrees = count - 1
    if standard_error == 0:
        statistic = None
        one_sided = 0.0 if mean > 0 else (1.0 if mean < 0 else 0.5)
        two_sided = 0.0 if mean != 0 else 1.0
        lower = upper = mean
        effect = None
        degeneracy = "zero_variance_nonzero_mean" if mean != 0 else "zero_variance_zero_mean"
    else:
        statistic = mean / standard_error
        one_sided = float(stats.t.sf(statistic, degrees))
        two_sided = float(2 * stats.t.sf(abs(statistic), degrees))
        critical = float(stats.t.ppf(0.975, degrees))
        lower = mean - critical * standard_error
        upper = mean + critical * standard_error
        effect = mean / standard_deviation
        degeneracy = "none"
    correction = _small_sample_correction(degrees)
    corrected_effect = None if effect is None or correction is None else effect * correction
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
        "correction": correction,
        "corrected_effect": corrected_effect,
        "degeneracy": degeneracy,
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
    for name, value in (("comparison", comparison), ("metric_name", metric_name)):
        if not value or value.strip() != value:
            raise ValueError(f"{name} must be nonempty and stripped")
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
        t_statistic=(None if summary["statistic"] is None else float(summary["statistic"])),
        one_sided_superiority_p_value=float(summary["one_sided"]),
        two_sided_p_value=float(summary["two_sided"]),
        two_sided_ci_lower=float(summary["lower"]),
        two_sided_ci_upper=float(summary["upper"]),
        paired_effect_dz=(None if summary["effect"] is None else float(summary["effect"])),
        small_sample_correction=(
            None if summary["correction"] is None else float(summary["correction"])
        ),
        small_sample_corrected_effect_gz=(
            None if summary["corrected_effect"] is None else float(summary["corrected_effect"])
        ),
        degeneracy=str(summary["degeneracy"]),  # type: ignore[arg-type]
        sign_flip=exact_sign_flip(differences),
        bootstrap=world_bootstrap_interval(differences, seed=bootstrap_seed),
    )


def holm_adjust(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> tuple[HolmResult, ...]:
    """Holm step-down correction for the exactly two registered primary tests."""

    if len(p_values) != 2:
        raise ValueError("registered Holm family requires exactly two hypotheses")
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
    if not math.isfinite(margin):
        raise ValueError("noninferiority margin must be finite")
    _, differences = paired_differences(treatment, comparator)
    summary = _paired_summary(differences)
    critical = float(stats.t.ppf(0.95, int(summary["degrees"])))
    lower = float(summary["mean"]) - critical * float(summary["standard_error"])
    if float(summary["standard_error"]) == 0:
        mean_minus_margin = float(summary["mean"]) - margin
        if math.isclose(mean_minus_margin, 0.0, rel_tol=1e-12, abs_tol=1e-15):
            p_value = 0.5
        else:
            p_value = 0.0 if mean_minus_margin > 0 else 1.0
    else:
        statistic = (float(summary["mean"]) - margin) / float(summary["standard_error"])
        p_value = float(stats.t.sf(statistic, int(summary["degrees"])))
    bootstrap = world_bootstrap_interval(differences, seed=bootstrap_seed)
    t_bound_passes = lower > margin and not math.isclose(
        lower, margin, rel_tol=1e-12, abs_tol=1e-15
    )
    bootstrap_bound_passes = bootstrap.lower > margin and not math.isclose(
        bootstrap.lower, margin, rel_tol=1e-12, abs_tol=1e-15
    )
    return NoninferiorityResult(
        comparison=comparison,
        metric_name="rare_pivotal_qualified_assertion_recall",
        margin=margin,
        mean_difference=float(summary["mean"]),
        one_sided_95_lower_bound=lower,
        one_sided_noninferiority_p_value=p_value,
        passes=t_bound_passes,
        world_count=int(summary["count"]),
        bootstrap=bootstrap,
        bootstrap_lower_exceeds_margin=bootstrap_bound_passes,
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


def _derived_bootstrap_seed(root_seed: int, label: str) -> int:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("bootstrap root seed must be a nonnegative integer")
    payload = f"story-projection-onto/analysis/bootstrap/{root_seed}/{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _condition_panel(
    panel: Mapping[str, Mapping[str, float | None]],
    condition: str,
    *,
    metric_name: str,
) -> Mapping[str, float | None]:
    try:
        values = panel[condition]
    except KeyError as error:
        raise ValueError(f"{metric_name} is missing condition {condition!r}") from error
    if not values:
        raise ValueError(f"{metric_name}/{condition} has no world values")
    for world_id, value in values.items():
        if not world_id or world_id.strip() != world_id:
            raise ValueError(f"{metric_name}/{condition} has an invalid world ID")
        if value is not None and (
            not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"registered rate {metric_name}/{condition}/{world_id} must lie in [0, 1]"
            )
    return values


def run_registered_analysis(
    *,
    strict_qualified_assertion_f1: Mapping[str, Mapping[str, float | None]],
    ontology_decision_macro_f1: Mapping[str, Mapping[str, float | None]],
    rare_pivotal_qualified_assertion_recall: Mapping[str, Mapping[str, float | None]],
    bootstrap_root_seed: int,
    alpha: float = REGISTERED_ALPHA,
    c0_condition: str = "C0",
    c1_condition: str = "C1",
    c2_condition: str = "C2",
    fixed_select_condition: str = "A-FixedSelect",
) -> RegisteredAnalysisResult:
    """Run the frozen confirmatory family, secondary C0 estimates, and gated control.

    Inputs must already be the registered per-world values.  In particular, ordinary
    metrics use :func:`aggregate_world_means`, while rare-pivotal recall uses
    :func:`aggregate_rare_pivotal_world_recalls`.  This entry point deliberately
    exposes no option to change the 12-world unit, 10,000 bootstrap resamples, 4,096
    sign assignments, two primary hypotheses, or -0.05 rare-pivotal margin.
    """

    if alpha != REGISTERED_ALPHA:
        raise ValueError(f"registered confirmatory alpha must equal {REGISTERED_ALPHA}")
    condition_names = (c0_condition, c1_condition, c2_condition, fixed_select_condition)
    if any(not value or value.strip() != value for value in condition_names):
        raise ValueError("condition names must be nonempty and stripped")
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("registered condition names must be distinct")

    semantic_c2 = _condition_panel(
        strict_qualified_assertion_f1,
        c2_condition,
        metric_name=PRIMARY_SEMANTIC_METRIC,
    )
    semantic_c1 = _condition_panel(
        strict_qualified_assertion_f1,
        c1_condition,
        metric_name=PRIMARY_SEMANTIC_METRIC,
    )
    semantic_c0 = _condition_panel(
        strict_qualified_assertion_f1,
        c0_condition,
        metric_name=PRIMARY_SEMANTIC_METRIC,
    )
    organization_c2 = _condition_panel(
        ontology_decision_macro_f1,
        c2_condition,
        metric_name=PRIMARY_ORGANIZATION_METRIC,
    )
    organization_c1 = _condition_panel(
        ontology_decision_macro_f1,
        c1_condition,
        metric_name=PRIMARY_ORGANIZATION_METRIC,
    )
    organization_c0 = _condition_panel(
        ontology_decision_macro_f1,
        c0_condition,
        metric_name=PRIMARY_ORGANIZATION_METRIC,
    )

    semantic_primary = analyze_paired_worlds(
        semantic_c2,
        semantic_c1,
        comparison="C2-C1",
        metric_name=PRIMARY_SEMANTIC_METRIC,
        bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "semantic/C2-C1"),
    )
    organization_primary = analyze_paired_worlds(
        organization_c2,
        organization_c1,
        comparison="C2-C1",
        metric_name=PRIMARY_ORGANIZATION_METRIC,
        bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "organization/C2-C1"),
    )
    primary_holm_raw = holm_adjust(
        {
            PRIMARY_SEMANTIC_HYPOTHESIS: semantic_primary.one_sided_superiority_p_value,
            PRIMARY_ORGANIZATION_HYPOTHESIS: (organization_primary.one_sided_superiority_p_value),
        },
        alpha=alpha,
    )
    primary_holm = (primary_holm_raw[0], primary_holm_raw[1])
    holm_by_hypothesis = {row.hypothesis: row for row in primary_holm}

    semantic_secondary = analyze_paired_worlds(
        semantic_c2,
        semantic_c0,
        comparison="C2-C0",
        metric_name=PRIMARY_SEMANTIC_METRIC,
        bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "semantic/C2-C0"),
    )
    organization_secondary = analyze_paired_worlds(
        organization_c2,
        organization_c0,
        comparison="C2-C0",
        metric_name=PRIMARY_ORGANIZATION_METRIC,
        bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "organization/C2-C0"),
    )

    rare_c2 = _condition_panel(
        rare_pivotal_qualified_assertion_recall,
        c2_condition,
        metric_name="rare_pivotal_qualified_assertion_recall",
    )
    rare_c1 = _condition_panel(
        rare_pivotal_qualified_assertion_recall,
        c1_condition,
        metric_name="rare_pivotal_qualified_assertion_recall",
    )
    rare_c0 = _condition_panel(
        rare_pivotal_qualified_assertion_recall,
        c0_condition,
        metric_name="rare_pivotal_qualified_assertion_recall",
    )
    rare_primary = rare_pivotal_noninferiority(
        rare_c2,
        rare_c1,
        comparison="C2-C1",
        bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "rare/C2-C1"),
    )
    rare_secondary = analyze_paired_worlds(
        rare_c2,
        rare_c0,
        comparison="C2-C0",
        metric_name="rare_pivotal_qualified_assertion_recall",
        bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "rare/C2-C0"),
    )

    organization_holm_passes = holm_by_hypothesis[PRIMARY_ORGANIZATION_HYPOTHESIS].rejected
    mechanism: PairedEstimate | None = None
    rare_mechanism: NoninferiorityResult | None = None
    if organization_holm_passes:
        organization_fixed = _condition_panel(
            ontology_decision_macro_f1,
            fixed_select_condition,
            metric_name=PRIMARY_ORGANIZATION_METRIC,
        )
        rare_fixed = _condition_panel(
            rare_pivotal_qualified_assertion_recall,
            fixed_select_condition,
            metric_name="rare_pivotal_qualified_assertion_recall",
        )
        mechanism = analyze_paired_worlds(
            organization_c2,
            organization_fixed,
            comparison="C2-A-FixedSelect",
            metric_name=PRIMARY_ORGANIZATION_METRIC,
            bootstrap_seed=_derived_bootstrap_seed(
                bootstrap_root_seed, "organization/C2-A-FixedSelect"
            ),
        )
        rare_mechanism = rare_pivotal_noninferiority(
            rare_c2,
            rare_fixed,
            comparison="C2-A-FixedSelect",
            bootstrap_seed=_derived_bootstrap_seed(bootstrap_root_seed, "rare/C2-A-FixedSelect"),
        )

    semantic_holm_passes = holm_by_hypothesis[PRIMARY_SEMANTIC_HYPOTHESIS].rejected
    mechanism_superiority_passes = (
        mechanism is not None and mechanism.one_sided_superiority_p_value <= alpha
    )
    construction_claim_passes = bool(
        organization_holm_passes
        and mechanism_superiority_passes
        and rare_primary.passes
        and rare_mechanism is not None
        and rare_mechanism.passes
    )
    return RegisteredAnalysisResult(
        semantic=RegisteredEndpointResult(
            metric_name=PRIMARY_SEMANTIC_METRIC,
            primary_c2_vs_c1=semantic_primary,
            secondary_c2_vs_c0=semantic_secondary,
        ),
        organization=RegisteredEndpointResult(
            metric_name=PRIMARY_ORGANIZATION_METRIC,
            primary_c2_vs_c1=organization_primary,
            secondary_c2_vs_c0=organization_secondary,
        ),
        primary_holm=primary_holm,
        rare_c2_vs_c1=rare_primary,
        rare_secondary_c2_vs_c0=rare_secondary,
        mechanism_status=(
            "tested" if organization_holm_passes else "not_tested_organization_gate_failed"
        ),
        mechanism_c2_vs_fixed_select=mechanism,
        rare_c2_vs_fixed_select=rare_mechanism,
        semantic_statistical_rare_gate_passes=semantic_holm_passes and rare_primary.passes,
        organization_statistical_rare_gate_passes=(
            organization_holm_passes and rare_primary.passes
        ),
        mechanism_statistical_rare_gate_passes=construction_claim_passes,
        alpha=alpha,
    )


def _world_value_panel(
    rows: Iterable[WorldMean | RarePivotalWorldRecall],
) -> dict[str, dict[str, float | None]]:
    panel: dict[str, dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        if row.world_id in panel[row.condition]:
            raise ValueError(f"duplicate world aggregate for {row.condition}/{row.world_id}")
        panel[row.condition][row.world_id] = row.value
    return dict(panel)


def run_registered_analysis_from_observations(
    metric_observations: Iterable[MetricObservation],
    rare_pivotal_observations: Iterable[RarePivotalCountObservation],
    *,
    bootstrap_root_seed: int,
    c0_condition: str = "C0",
    c1_condition: str = "C1",
    c2_condition: str = "C2",
    fixed_select_condition: str = "A-FixedSelect",
) -> RegisteredAnalysisResult:
    """Aggregate the complete design and run :func:`run_registered_analysis`.

    This is the preferred ledger-facing entry point because it mechanically applies
    the frozen seed-within-context, context-within-world estimand and the separate
    count-pooled rare-pivotal formula before any inferential calculation.
    """

    registered_conditions = {c0_condition, c1_condition, c2_condition, fixed_select_condition}
    metric_rows = tuple(
        row for row in metric_observations if row.condition in registered_conditions
    )
    rare_rows = tuple(
        row for row in rare_pivotal_observations if row.condition in registered_conditions
    )
    ordered_conditions = (c0_condition, c1_condition, c2_condition, fixed_select_condition)
    validate_cross_condition_metric_design(
        metric_rows,
        metric_names=(PRIMARY_SEMANTIC_METRIC, PRIMARY_ORGANIZATION_METRIC),
        conditions=ordered_conditions,
        require_scorer_bindings=True,
    )
    validate_cross_condition_rare_design(
        rare_rows,
        conditions=ordered_conditions,
        require_scorer_bindings=True,
    )
    semantic_worlds = aggregate_world_means(
        metric_rows,
        metric_name=PRIMARY_SEMANTIC_METRIC,
        deterministic_conditions=frozenset({c0_condition}),
    )
    organization_worlds = aggregate_world_means(
        metric_rows,
        metric_name=PRIMARY_ORGANIZATION_METRIC,
        deterministic_conditions=frozenset({c0_condition}),
    )
    rare_worlds = aggregate_rare_pivotal_world_recalls(
        rare_rows,
        deterministic_conditions=frozenset({c0_condition}),
    )
    return run_registered_analysis(
        strict_qualified_assertion_f1=_world_value_panel(semantic_worlds),
        ontology_decision_macro_f1=_world_value_panel(organization_worlds),
        rare_pivotal_qualified_assertion_recall=_world_value_panel(rare_worlds),
        bootstrap_root_seed=bootstrap_root_seed,
        c0_condition=c0_condition,
        c1_condition=c1_condition,
        c2_condition=c2_condition,
        fixed_select_condition=fixed_select_condition,
    )


def _paired_estimate_from_retained_worlds(
    treatment: Mapping[str, float],
    comparator: Mapping[str, float],
    *,
    comparison: str,
    metric_name: str,
    bootstrap_seed: int,
) -> PairedEstimate | None:
    if set(treatment) != set(comparator):
        raise ValueError("internal retained-world mismatch")
    if len(treatment) < 2:
        return None
    return analyze_paired_worlds(
        treatment,
        comparator,
        comparison=comparison,
        metric_name=metric_name,
        bootstrap_seed=bootstrap_seed,
        require_world_count=len(treatment),
    )


def _pooled_crossing_rate(rows: Iterable[CrossingObservation]) -> ExposureWeightedCrossingRate:
    positive = tuple(
        row for row in rows if row.opportunity_count is not None and row.opportunity_count > 0
    )
    crossings = sum(row.crossing_count or 0 for row in positive)
    opportunities = sum(row.opportunity_count or 0 for row in positive)
    return ExposureWeightedCrossingRate(
        crossing_count=crossings,
        opportunity_count=opportunities,
        rate=(crossings / opportunities if opportunities else None),
        positive_opportunity_output_count=len(positive),
    )


def _two_stage_pair_mean(
    pairs: Sequence[tuple[CrossingObservation, CrossingObservation]],
    *,
    member_index: int,
    value_kind: Literal["incidence", "opportunity_count", "conditional_rate"],
) -> float:
    """Average retained seeds within context and then retained contexts within world."""

    by_context: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        row = pair[member_index]
        assert row.opportunity_count is not None
        if value_kind == "incidence":
            value = float(row.opportunity_count > 0)
        elif value_kind == "opportunity_count":
            value = float(row.opportunity_count)
        else:
            assert row.opportunity_count > 0
            assert row.crossing_count is not None
            value = row.crossing_count / row.opportunity_count
        by_context[row.context_id].append(value)
    return float(np.mean([np.mean(seed_values) for seed_values in by_context.values()]))


def compare_crossing_profiles(
    observations: Iterable[CrossingObservation],
    *,
    treatment_condition: str,
    comparator_condition: str,
    comparison: str,
    bootstrap_root_seed: int,
    require_complete_design: bool = True,
) -> CrossingComparisonResult:
    """Apply the registered opportunity/conditional crossing comparison.

    Seeded conditions are paired at world/context/seed.  When exactly one condition
    is deterministic (C0), its one unseeded output is compared with the other
    condition's seed mean at world/context; the deterministic graph is never copied
    into artificial seed rows. Exposure-weighted rates remain separately descriptive.
    """

    if treatment_condition == comparator_condition:
        raise ValueError("crossing comparison requires distinct conditions")
    if not comparison or comparison.strip() != comparison:
        raise ValueError("comparison must be nonempty and stripped")
    selected = tuple(
        row for row in observations if row.condition in {treatment_condition, comparator_condition}
    )
    if not selected:
        raise ValueError("crossing comparison has no observations")
    grouped: dict[str, dict[tuple[str, str], list[CrossingObservation]]] = {
        treatment_condition: defaultdict(list),
        comparator_condition: defaultdict(list),
    }
    for row in selected:
        grouped[row.condition][(row.world_id, row.context_id)].append(row)
    if set(grouped[treatment_condition]) != set(grouped[comparator_condition]):
        raise ValueError("crossing conditions have different world/context cells")
    cells = tuple(sorted(grouped[treatment_condition]))
    worlds = tuple(sorted({key[0] for key in cells}))

    deterministic: dict[str, bool] = {}
    for condition in (treatment_condition, comparator_condition):
        seed_kinds = {
            row.seed_block is None
            for rows in grouped[condition].values()
            for row in rows
        }
        if len(seed_kinds) != 1:
            raise ValueError(f"crossing condition {condition!r} mixes seeded and unseeded rows")
        deterministic[condition] = True in seed_kinds
    paired_by_seed = not (
        deterministic[treatment_condition] or deterministic[comparator_condition]
    )

    if require_complete_design:
        if len(worlds) != REGISTERED_WORLD_COUNT:
            raise ValueError(
                f"crossing comparison requires exactly {REGISTERED_WORLD_COUNT} worlds"
            )
        for world_id in worlds:
            world_cells = tuple(key for key in cells if key[0] == world_id)
            contexts = {key[1] for key in world_cells}
            if len(contexts) != REGISTERED_CONTEXTS_PER_WORLD:
                raise ValueError(
                    f"{world_id} has {len(contexts)} crossing contexts; "
                    f"expected {REGISTERED_CONTEXTS_PER_WORLD}"
                )
    for condition in (treatment_condition, comparator_condition):
        for (world_id, context_id), rows in grouped[condition].items():
            seeds = tuple(sorted(row.seed_block for row in rows if row.seed_block is not None))
            if len(rows) != len({row.seed_block for row in rows}):
                raise ValueError(f"duplicate crossing seed for {condition}/{world_id}/{context_id}")
            expected = () if deterministic[condition] else REGISTERED_LLM_SEEDS
            if deterministic[condition]:
                valid = len(rows) == 1 and rows[0].seed_block is None
            else:
                valid = seeds == expected and all(row.seed_block is not None for row in rows)
            if require_complete_design and not valid:
                expected_description = (
                    "one deterministic unseeded row"
                    if deterministic[condition]
                    else repr(expected)
                )
                raise ValueError(
                    f"{condition}/{world_id}/{context_id} crossing seeds do not equal "
                    f"{expected_description}"
                )

    # Each unit holds one output per side for seeded-vs-seeded comparisons, or all
    # seed rows per side for a context-level deterministic comparison.
    units: list[
        tuple[str, str, tuple[CrossingObservation, ...], tuple[CrossingObservation, ...]]
    ] = []
    for world_id, context_id in cells:
        treatment_cell = tuple(grouped[treatment_condition][(world_id, context_id)])
        comparator_cell = tuple(grouped[comparator_condition][(world_id, context_id)])
        if paired_by_seed:
            treatment_by_seed = {row.seed_block: row for row in treatment_cell}
            comparator_by_seed = {row.seed_block: row for row in comparator_cell}
            if set(treatment_by_seed) != set(comparator_by_seed):
                raise ValueError("seeded crossing conditions have different seed keys")
            for seed in sorted(treatment_by_seed):
                units.append(
                    (
                        world_id,
                        context_id,
                        (treatment_by_seed[seed],),
                        (comparator_by_seed[seed],),
                    )
                )
        else:
            units.append((world_id, context_id, treatment_cell, comparator_cell))

    def values(
        rows: tuple[CrossingObservation, ...],
        kind: Literal["incidence", "opportunity_count", "conditional_rate"],
    ) -> tuple[float, ...] | None:
        if any(row.opportunity_count is None for row in rows):
            return None
        if kind == "incidence":
            return tuple(float((row.opportunity_count or 0) > 0) for row in rows)
        if kind == "opportunity_count":
            return tuple(float(row.opportunity_count or 0) for row in rows)
        return tuple(
            (row.crossing_count or 0) / (row.opportunity_count or 1)
            for row in rows
            if (row.opportunity_count or 0) > 0
        )

    available_values: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    positive_values: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    unavailable = treatment_only = comparator_only = neither = both = 0
    for world_id, context_id, treatment_unit, comparator_unit in units:
        treatment_incidence = values(treatment_unit, "incidence")
        comparator_incidence = values(comparator_unit, "incidence")
        if treatment_incidence is None or comparator_incidence is None:
            unavailable += 1
            continue
        treatment_count = values(treatment_unit, "opportunity_count")
        comparator_count = values(comparator_unit, "opportunity_count")
        assert treatment_count is not None and comparator_count is not None
        available_values[world_id].append(
            (
                context_id,
                float(np.mean(treatment_incidence)),
                float(np.mean(comparator_incidence)),
                float(np.mean(treatment_count)),
                float(np.mean(comparator_count)),
            )
        )
        treatment_positive = bool(any(treatment_incidence))
        comparator_positive = bool(any(comparator_incidence))
        if treatment_positive and comparator_positive:
            both += 1
            treatment_rates = values(treatment_unit, "conditional_rate")
            comparator_rates = values(comparator_unit, "conditional_rate")
            assert treatment_rates and comparator_rates
            positive_values[world_id].append(
                (context_id, float(np.mean(treatment_rates)), float(np.mean(comparator_rates)))
            )
        elif treatment_positive:
            treatment_only += 1
        elif comparator_positive:
            comparator_only += 1
        else:
            neither += 1

    incidence_treatment: dict[str, float] = {}
    incidence_comparator: dict[str, float] = {}
    count_treatment: dict[str, float] = {}
    count_comparator: dict[str, float] = {}
    conditional_treatment: dict[str, float] = {}
    conditional_comparator: dict[str, float] = {}
    world_counts: list[WorldRetainedPairCount] = []

    def two_stage_mean(rows: Sequence[tuple[Any, ...]], value_index: int) -> float:
        by_context: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_context[row[0]].append(float(row[value_index]))
        return float(np.mean([np.mean(items) for items in by_context.values()]))

    for world_id in worlds:
        available = available_values[world_id]
        positive = positive_values[world_id]
        world_counts.append(
            WorldRetainedPairCount(
                world_id=world_id,
                available_pair_count=len(available),
                both_positive_pair_count=len(positive),
            )
        )
        if available:
            incidence_treatment[world_id] = two_stage_mean(available, 1)
            incidence_comparator[world_id] = two_stage_mean(available, 2)
            count_treatment[world_id] = two_stage_mean(available, 3)
            count_comparator[world_id] = two_stage_mean(available, 4)
        if positive:
            conditional_treatment[world_id] = two_stage_mean(positive, 1)
            conditional_comparator[world_id] = two_stage_mean(positive, 2)

    incidence = _paired_estimate_from_retained_worlds(
        incidence_treatment,
        incidence_comparator,
        comparison=comparison,
        metric_name="crossing_opportunity_incidence",
        bootstrap_seed=_derived_bootstrap_seed(
            bootstrap_root_seed, f"crossing/{comparison}/opportunity-incidence"
        ),
    )
    opportunity_count = _paired_estimate_from_retained_worlds(
        count_treatment,
        count_comparator,
        comparison=comparison,
        metric_name="crossing_opportunity_count",
        bootstrap_seed=_derived_bootstrap_seed(
            bootstrap_root_seed, f"crossing/{comparison}/opportunity-count"
        ),
    )
    conditional = _paired_estimate_from_retained_worlds(
        conditional_treatment,
        conditional_comparator,
        comparison=comparison,
        metric_name="conditional_crossing_rate",
        bootstrap_seed=_derived_bootstrap_seed(
            bootstrap_root_seed, f"crossing/{comparison}/conditional-rate"
        ),
    )
    treatment_pooled = _pooled_crossing_rate(
        row for rows in grouped[treatment_condition].values() for row in rows
    )
    comparator_pooled = _pooled_crossing_rate(
        row for rows in grouped[comparator_condition].values() for row in rows
    )
    pooled_difference = (
        None
        if treatment_pooled.rate is None or comparator_pooled.rate is None
        else treatment_pooled.rate - comparator_pooled.rate
    )
    warnings: list[str] = []
    if unavailable:
        warnings.append(f"{unavailable} matched output pairs had unavailable geometry")
    if len(conditional_treatment) < len(worlds):
        warnings.append(
            "conditional rate excludes worlds without any matched positive-opportunity pair"
        )
    if conditional is None:
        warnings.append("fewer than two worlds support a conditional paired estimate")
    return CrossingComparisonResult(
        comparison=comparison,
        treatment_condition=treatment_condition,
        comparator_condition=comparator_condition,
        expected_output_pair_count=len(units),
        available_output_pair_count=len(units) - unavailable,
        unavailable_output_pair_count=unavailable,
        both_positive_pair_count=both,
        treatment_only_positive_pair_count=treatment_only,
        comparator_only_positive_pair_count=comparator_only,
        neither_positive_pair_count=neither,
        world_pair_counts=tuple(world_counts),
        opportunity_incidence=incidence,
        opportunity_count=opportunity_count,
        conditional_rate=conditional,
        conditional_rate_retained_world_count=len(conditional_treatment),
        conditional_rate_excluded_world_count=len(worlds) - len(conditional_treatment),
        treatment_pooled_rate=treatment_pooled,
        comparator_pooled_rate=comparator_pooled,
        pooled_rate_difference=pooled_difference,
        analysis_warning="; ".join(warnings) if warnings else None,
    )


def analysis_to_json(value: Any) -> str:
    """Serialize an analysis record as strict, deterministic, machine-readable JSON."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("analysis_to_json requires a dataclass record instance")
    return json.dumps(
        asdict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "PRIMARY_ORGANIZATION_HYPOTHESIS",
    "PRIMARY_ORGANIZATION_METRIC",
    "PRIMARY_SEMANTIC_HYPOTHESIS",
    "PRIMARY_SEMANTIC_METRIC",
    "RARE_PIVOTAL_NONINFERIORITY_MARGIN",
    "REGISTERED_ALPHA",
    "REGISTERED_BOOTSTRAP_RESAMPLES",
    "REGISTERED_CONTEXTS_PER_WORLD",
    "REGISTERED_LLM_SEEDS",
    "REGISTERED_SIGN_FLIP_ASSIGNMENTS",
    "REGISTERED_WORLD_COUNT",
    "BootstrapInterval",
    "CrossingComparisonResult",
    "CrossingObservation",
    "ExposureWeightedCrossingRate",
    "HolmResult",
    "MetricObservation",
    "NoninferiorityResult",
    "PairedEstimate",
    "RarePivotalCountObservation",
    "RarePivotalWorldRecall",
    "RegisteredAnalysisResult",
    "RegisteredEndpointResult",
    "SeedPooledRareRecall",
    "SignFlipResult",
    "WorldMean",
    "WorldRetainedPairCount",
    "aggregate_rare_pivotal_world_recalls",
    "aggregate_world_means",
    "analysis_to_json",
    "analyze_paired_worlds",
    "compare_crossing_profiles",
    "exact_sign_flip",
    "holm_adjust",
    "mechanism_comparison_is_admissible",
    "paired_differences",
    "rare_pivotal_noninferiority",
    "run_registered_analysis",
    "run_registered_analysis_from_observations",
    "validate_cross_condition_metric_design",
    "validate_cross_condition_rare_design",
    "world_bootstrap_interval",
]
