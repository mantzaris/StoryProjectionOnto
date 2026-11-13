from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import ConditionName, ReleaseClass, canonical_sha256
from story_projection_onto.metrics.pipeline import MetricResultRow, PipelineMetricStatus
from story_projection_onto.scorer_only.phase4_analysis import (
    AblationPairResult,
    CanonicalComparisonRow,
    ExploratoryComparison,
    Phase4AnalysisConfiguration,
    Phase4AnalysisError,
    Phase4ReportGateStatus,
    Phase4TableEntry,
    Phase4TableManifest,
    ScoredMetricObservation,
    WorldMetricRow,
    _append_exact,
    _comparison_csv,
    _entropy_multiplicity,
    _exploratory_comparisons,
    _ScoreCollection,
    _world_metric_rows,
)

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "a" * 64


def _metric_observation(
    *,
    condition: ConditionName,
    world: int,
    context: int,
    seed: int | None,
    value: float | None,
) -> ScoredMetricObservation:
    unit_id = f"unit:{condition.value}:w{world:02d}:c{context}:s{seed or 0}"
    return ScoredMetricObservation(
        source_block="primary",
        unit_id=unit_id,
        condition=condition,
        world_id=f"world-{world:02d}",
        context_id=f"world-{world:02d}-context-{context}",
        seed_block=seed,
        output_valid=value is not None,
        metric=MetricResultRow(
            unit_hash=canonical_sha256(unit_id),
            metric_name="strict_qualified_assertion_f1",
            metric_version_hash=DIGEST,
            status=(
                PipelineMetricStatus.VALUE
                if value is not None
                else PipelineMetricStatus.INVALID
            ),
            value=value,
        ),
    )


def _primary_panel(*, missing_c2_cell: bool = False) -> _ScoreCollection:
    observations = []
    for world in range(1, 13):
        for context in range(1, 4):
            observations.append(
                _metric_observation(
                    condition=ConditionName.C0_CLASSICAL_PRE,
                    world=world,
                    context=context,
                    seed=None,
                    value=0.25,
                )
            )
            for condition, value in (
                (ConditionName.C1_LLM_PRE, 0.5),
                (ConditionName.C2_LLM_QUERY, 0.75),
                (ConditionName.A_FIXED_SELECT, 0.625),
            ):
                for seed in (1, 2):
                    cell_value = value
                    if (
                        missing_c2_cell
                        and condition is ConditionName.C2_LLM_QUERY
                        and world == 1
                        and context == 1
                        and seed == 2
                    ):
                        cell_value = None
                    observations.append(
                        _metric_observation(
                            condition=condition,
                            world=world,
                            context=context,
                            seed=seed,
                            value=cell_value,
                        )
                    )
    return _ScoreCollection(
        scores=(),
        bundles={},
        projections={},
        grounding_audits={},
        geometries={},
        observations=tuple(observations),
    )


def test_checked_in_phase4_configuration_binds_registered_inputs() -> None:
    configuration = Phase4AnalysisConfiguration.load(
        ROOT,
        ROOT / "configs/study/phase4_analysis.json",
    )

    assert configuration.registered_world_count == 12
    assert configuration.primary_score_count == 252
    assert configuration.combined_ordinary_score_count == 40
    assert configuration.registered_sign_flip_assignments == 4096
    assert configuration.registered_bootstrap_resamples == 10_000
    assert configuration.rare_pivotal_noninferiority_margin == -0.05


def test_world_metrics_average_seeds_then_contexts_without_pseudoreplication() -> None:
    rows = _world_metric_rows(_primary_panel())
    lookup = {(row.condition, row.world_id): row for row in rows}

    assert len(rows) == 48
    assert lookup[(ConditionName.C0_CLASSICAL_PRE, "world-01")].value == 0.25
    assert lookup[(ConditionName.C0_CLASSICAL_PRE, "world-01")].row_count == 3
    assert lookup[(ConditionName.C2_LLM_QUERY, "world-01")].value == 0.75
    assert lookup[(ConditionName.C2_LLM_QUERY, "world-01")].row_count == 6


def test_world_metrics_retain_structural_na_instead_of_rewarding_failure() -> None:
    rows = _world_metric_rows(_primary_panel(missing_c2_cell=True))
    target = next(
        row
        for row in rows
        if row.condition is ConditionName.C2_LLM_QUERY and row.world_id == "world-01"
    )

    assert target.value is None
    assert target.undefined_context_count == 1


def test_entropy_family_uses_planned_ten_test_bh_denominator() -> None:
    metrics = (
        "degree_histogram_entropy",
        "degree_mass_entropy",
        "local_relation_neighborhood_entropy",
        "native_schema_relation_entropy",
        "canonical_mapped_relation_entropy",
    )
    comparisons = tuple(
        ExploratoryComparison(
            metric_name=metric,
            comparison=comparison,
            status="estimated",
            retained_world_count=12,
            estimate={"two_sided_p_value": index / 100},
        )
        for index, (metric, comparison) in enumerate(
            ((metric, comparison) for metric in metrics for comparison in ("C2-C0", "C2-C1")),
            1,
        )
    )

    adjusted = _entropy_multiplicity(comparisons)

    assert len(adjusted) == 10
    assert all(row.status == "adjusted" for row in adjusted)
    assert all(row.benjamini_hochberg_adjusted_p_value == pytest.approx(0.1) for row in adjusted)


def test_mechanism_support_adds_fixed_select_change_and_collapse_world_tests() -> None:
    rows = tuple(
        WorldMetricRow(
            condition=condition,
            world_id=f"world-{world:02d}",
            metric_name=metric,
            value=(
                0.8
                if condition is ConditionName.C2_LLM_QUERY
                else 0.4
                if metric == "contrastive_decision_change_f1"
                else 0.7
            ),
            context_count=1,
            row_count=2,
            undefined_context_count=0,
        )
        for metric in (
            "contrastive_decision_change_f1",
            "contrastive_collapse",
        )
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
        for world in range(1, 13)
    )

    comparisons = _exploratory_comparisons(rows, bootstrap_root_seed=7331)
    mechanism = {
        item.metric_name: item
        for item in comparisons
        if item.comparison == "C2-A-FixedSelect"
    }

    assert set(mechanism) == {
        "contrastive_decision_change_f1",
        "contrastive_collapse",
    }
    assert all(item.status == "estimated" for item in mechanism.values())
    assert all(item.retained_world_count == 12 for item in mechanism.values())
    assert all(item.estimate["two_sided_p_value"] is not None for item in mechanism.values())


def test_table_manifest_is_self_hashed_sorted_and_selection_explicit() -> None:
    entry = Phase4TableEntry(
        table_id="unit_metric_observations",
        relative_path="tables/unit_metrics.jsonl",
        columns=("world_id", "condition", "metric"),
        row_count=123,
        file_sha256=DIGEST,
        logical_content_hash="b" * 64,
        release_class=ReleaseClass.PUBLIC,
        independent_unit="metric_observation",
        selection_ready=True,
    )
    manifest = Phase4TableManifest(
        manifest_id="phase4-table-test",
        analysis_configuration_hash="c" * 64,
        metric_version_hash="d" * 64,
        tables=(entry,),
    )

    assert manifest.content_hash == canonical_sha256(manifest)
    assert manifest.tables[0].selection_ready

    later_entry = Phase4TableEntry.model_validate(
        {
            **entry.model_dump(mode="json", exclude={"content_hash"}),
            "table_id": "z",
        }
    )
    with pytest.raises(ValidationError, match="sorted by table ID"):
        Phase4TableManifest(
            manifest_id="phase4-table-unsorted",
            analysis_configuration_hash="c" * 64,
            metric_version_hash="d" * 64,
            tables=(later_entry, entry),
        )


def test_ablation_difference_cannot_be_hand_transcribed_inconsistently() -> None:
    with pytest.raises(ValidationError, match="ablation minus parent"):
        AblationPairResult(
            call_id="ablation-test",
            condition=ConditionName.A_NO_CONTEXT,
            world_id="world-01",
            context_id="context-01",
            parent_score_hash=DIGEST,
            ablation_score_hash="b" * 64,
            principal_metric="ontology_decision_macro_f1",
            parent_value=0.5,
            ablation_value=0.25,
            difference=0.25,
        )


def test_append_only_output_accepts_exact_replay_and_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "tables" / "values.jsonl"
    _append_exact(path, b'{"value":1}\n', restricted=False)
    _append_exact(path, b'{"value":1}\n', restricted=False)

    with pytest.raises(Phase4AnalysisError, match="output drift"):
        _append_exact(path, b'{"value":2}\n', restricted=False)


def test_canonical_comparison_csv_has_report_contract_columns_in_order() -> None:
    row = CanonicalComparisonRow(
        comparison="C2-C1",
        metric="strict_qualified_assertion_f1",
        estimate=0.1,
        paired_median=0.09,
        standardized_effect=0.4,
        ci_lower=-0.1,
        ci_upper=0.3,
        p_value=0.04,
        adjusted_p_value=0.05,
        sign_flip_p_value=0.06,
        bootstrap_lower=-0.08,
        bootstrap_upper=0.29,
        paired_world_count=12,
        status="estimated",
        analysis_family="registered_primary",
        p_value_kind="one_sided_superiority",
        multiplicity_method="Holm_two_primary_hypotheses",
    )

    header, values = _comparison_csv((row,)).decode("utf-8").splitlines()

    assert header == (
        "comparison,metric,estimate,paired_median,standardized_effect,ci_lower,ci_upper,"
        "p_value,adjusted_p_value,sign_flip_p_value,bootstrap_lower,bootstrap_upper,"
        "paired_world_count,status,analysis_family,p_value_kind,multiplicity_method,"
        "independent_unit"
    )
    assert values.startswith("C2-C1,strict_qualified_assertion_f1,0.1")


def test_report_gate_status_is_mechanically_derived_and_hash_bound() -> None:
    registered_hash = "e" * 64
    gate = Phase4ReportGateStatus(
        gate_status_id="phase4-report-gates-test",
        registered_analysis_hash=registered_hash,
        organization_adjusted_p_value=0.04,
        mechanism_one_sided_p_value=0.03,
        c2_vs_c1_rare_lower_bound=-0.01,
        c2_vs_fixed_rare_lower_bound=-0.02,
        c2_vs_fixed_rare_p_value=0.02,
        organization_gate_passed=True,
        construction_freedom_gate_passed=True,
        c2_vs_c1_rare_safeguard_gate_passed=True,
        c2_vs_fixed_rare_safeguard_status="passed",
        c2_vs_fixed_rare_safeguard_gate_passed=True,
    )

    assert gate.registered_analysis_hash == registered_hash
    assert gate.rare_safeguard_margin == -0.05
    assert gate.content_hash == canonical_sha256(gate)

    with pytest.raises(ValidationError, match="organization gate"):
        Phase4ReportGateStatus(
            gate_status_id="phase4-report-gates-invalid",
            registered_analysis_hash=registered_hash,
            organization_adjusted_p_value=0.04,
            mechanism_one_sided_p_value=0.03,
            c2_vs_c1_rare_lower_bound=-0.01,
            c2_vs_fixed_rare_lower_bound=-0.02,
            c2_vs_fixed_rare_p_value=0.02,
            organization_gate_passed=False,
            construction_freedom_gate_passed=False,
            c2_vs_c1_rare_safeguard_gate_passed=True,
            c2_vs_fixed_rare_safeguard_status="passed",
            c2_vs_fixed_rare_safeguard_gate_passed=True,
        )
