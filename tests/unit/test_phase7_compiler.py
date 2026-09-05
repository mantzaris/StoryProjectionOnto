from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from story_projection_onto.phase7_compiler import (
    C2IntentionToTreatScore,
    ExactRowFilter,
    MetricPartitionRequirement,
    MetricRowRequirement,
    MetricSourceFilter,
    Phase7CompilationError,
    Phase7CompilerConfiguration,
    Phase7TableBinding,
    PhaseInputState,
    PublicSourceCategory,
    QualitativeCandidate,
    QualitativeCandidateSet,
    QualitativeDisplayRow,
    QualitativeExample,
    SupplementalMetricSourceContract,
    TableContract,
    TableProducer,
    _canonical_singleton_csv,
    _canonical_source_csv,
    _choose_qualitative_candidates,
    _inspect_supplemental_metric_source_csv,
    _study_status_source_artifact_ids,
    _validate_mechanism_table_rows,
    _validate_registered_table_binding,
    _validate_registered_table_rows,
    _verify_phase4_table_manifest_binding,
    compile_phase7_results,
    expand_public_reproduction_source_paths,
    load_phase7_source_registry,
    load_public_reproduction_source_manifest,
    verify_phase7_results,
)
from story_projection_onto.public_release import (
    PublicEntry,
    scan_public_bytes,
    scan_public_entries,
)
from story_projection_onto.report_ingestion import SourceArtifactSpec
from story_projection_onto.reporting import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = ROOT / "configs/study/phase7_compiler.json"
TEMPLATE = ROOT / "configs/study/phase7_source_registry.template.json"
POLICY = ROOT / "configs/study/reporting.json"
HASH = "1" * 64


def _copy_public_reproduction_sources(destination: Path) -> tuple[str, ...]:
    configuration = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    manifest = load_public_reproduction_source_manifest(
        ROOT / configuration["public_source_manifest_path"]
    )
    relative_paths = expand_public_reproduction_source_paths(
        manifest,
        source_root=ROOT,
    )
    for relative in relative_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return relative_paths


def _display(context: str, condition: str) -> QualitativeDisplayRow:
    return QualitativeDisplayRow(
        context_id=context,
        condition=condition,
        node_summary="Two context-dependent entities with readable roles.",
        assertion_summary="A grounded, directed assertion with validity and confidence.",
        temporal_sequence="Event one precedes event two at story time t2.",
        why_matters="The distinction changes the answer to the registered query.",
        opaque_evidence_ids=("ev-opaque-1",),
        projection_hash=HASH,
    )


def _candidate(
    candidate_id: str,
    example: QualitativeExample,
    *,
    split: str,
    world: str,
    contexts: tuple[str, ...],
    query_ordinal: str,
    stratum: str,
    window: int | None = None,
    holder: bool = False,
    rare: int = 0,
    support: bool = False,
    temporal: bool = False,
    scores: tuple[C2IntentionToTreatScore, ...] = (),
) -> QualitativeCandidate:
    conditions = (
        "c0_classical_pre",
        "c1_llm_pre",
        "c2_llm_query",
        "a_fixed_select",
    )
    return QualitativeCandidate(
        candidate_id=candidate_id,
        eligible_examples=(example,),
        evidence_split=split,
        world_id=world,
        context_ids=contexts,
        query_ordinal=query_ordinal,
        stratum=stratum,
        window_ordinal=window,
        rare_pivotal_denominator=rare,
        complete_support_path=support,
        temporal_epistemic_eligible=temporal,
        holder_attributed=holder,
        c2_itt_scores=scores,
        reviewed_failure_id=(
            f"reviewed-failure-{candidate_id}"
            if example is QualitativeExample.COUNTEREXAMPLE
            else None
        ),
        display_rows=tuple(
            _display(context, condition)
            for context in contexts
            for condition in conditions
        ),
        composite_figure_artifact_id=f"{candidate_id}-figure",
        source_artifact_ids=(f"{candidate_id}-figure",),
        paraphrase_only=True,
        opaque_evidence_only=True,
        contains_verbatim_copyrighted_text=False,
    )


def _candidate_set() -> QualitativeCandidateSet:
    policy_hash = json.loads(POLICY.read_text(encoding="utf-8"))["policy_sha256"]
    values = (
        _candidate(
            "tutorial",
            QualitativeExample.DEVELOPMENT_TUTORIAL,
            split="development",
            world="syn-dev-01",
            contexts=("ctx-a", "ctx-b"),
            query_ordinal="A+B",
            stratum="easy",
        ),
        _candidate(
            "rare",
            QualitativeExample.RARE_PIVOTAL,
            split="development",
            world="syn-dev-02",
            contexts=("ctx-c",),
            query_ordinal="A",
            stratum="medium",
            rare=1,
            support=True,
        ),
        _candidate(
            "temporal-no-holder",
            QualitativeExample.TEMPORAL_EPISTEMIC,
            split="development",
            world="syn-dev-01",
            contexts=("ctx-d",),
            query_ordinal="A",
            stratum="easy",
            temporal=True,
        ),
        _candidate(
            "temporal-holder",
            QualitativeExample.TEMPORAL_EPISTEMIC,
            split="development",
            world="syn-dev-04",
            contexts=("ctx-e",),
            query_ordinal="B",
            stratum="hard",
            temporal=True,
            holder=True,
        ),
        _candidate(
            "held",
            QualitativeExample.HELD_OUT_ILLUSTRATION,
            split="held_out",
            world="syn-test-03",
            contexts=("ctx_556b0577875afcad7419",),
            query_ordinal="A",
            stratum="easy",
        ),
        _candidate(
            "counter-high",
            QualitativeExample.COUNTEREXAMPLE,
            split="held_out",
            world="syn-test-10",
            contexts=("ctx-hard-z",),
            query_ordinal="A",
            stratum="hard",
            scores=(
                C2IntentionToTreatScore(
                    seed_block=1,
                    outcome="succeeded",
                    strict_qualified_assertion_f1=0.5,
                ),
                C2IntentionToTreatScore(
                    seed_block=2,
                    outcome="succeeded",
                    strict_qualified_assertion_f1=0.5,
                ),
            ),
        ),
        _candidate(
            "counter-itt-failure",
            QualitativeExample.COUNTEREXAMPLE,
            split="held_out",
            world="syn-test-11",
            contexts=("ctx-hard-a",),
            query_ordinal="B",
            stratum="hard",
            scores=(
                C2IntentionToTreatScore(
                    seed_block=1,
                    outcome="timed_out",
                    strict_qualified_assertion_f1=0.0,
                ),
                C2IntentionToTreatScore(
                    seed_block=2,
                    outcome="succeeded",
                    strict_qualified_assertion_f1=0.2,
                ),
            ),
        ),
        _candidate(
            "narrative",
            QualitativeExample.NARRATIVE_ILLUSTRATION,
            split="case_study",
            world="window-1",
            contexts=("case-a", "case-b"),
            query_ordinal="A+B",
            stratum="case_study",
            window=1,
        ),
    )
    raw = {
        "schema_version": "1.0.0",
        "candidate_set_id": "test-candidates",
        "frozen_reporting_policy_sha256": policy_hash,
        "candidates": [item.model_dump(mode="json") for item in values],
        "copyright_release_attestation_hash": "2" * 64,
    }
    raw["content_hash"] = canonical_sha256(raw)
    return QualitativeCandidateSet.model_validate(raw)


def test_checked_in_configuration_and_incomplete_template_are_self_hashed() -> None:
    configuration = Phase7CompilerConfiguration.load(CONFIGURATION, source_root=ROOT)
    registry = load_phase7_source_registry(TEMPLATE)
    public_manifest = load_public_reproduction_source_manifest(
        ROOT / configuration.public_source_manifest_path
    )
    public_paths = expand_public_reproduction_source_paths(
        public_manifest,
        source_root=ROOT,
    )
    assert {item.category for item in public_manifest.groups} == set(PublicSourceCategory)
    assert len(public_paths) == sum(
        item.expected_path_count for item in public_manifest.groups
    )
    assert {
        "configs/study/public_reproduction_sources.json",
        "data/synthetic/manifests/benchmark_manifest.json",
        "data/synthetic/scorer_only/held_out/syn-test-12.json",
        "prompts/c2_query/prompt_v1.md",
        "schemas/jsonschema/fallback_control_plane_incident.schema.json",
        "schemas/jsonschema/ontology_projection.schema.json",
        "scripts/build_fallback_control_plane_incident.py",
        "scripts/build_final_accounting_source_recipe.py",
        "src/story_projection_onto/fallback_control_plane_incident.py",
        "src/story_projection_onto/scorer_only/phase4_analysis.py",
        "tests/integration/ui_browser_smoke.mjs",
        "tests/property/test_benchmark_properties.py",
        "tests/unit/test_fallback_control_plane_incident.py",
        "tests/unit/test_phase7_compiler.py",
        "ui/index.html",
    }.issubset(public_paths)
    assert len(configuration.table_contracts) == 14
    assert all(1 <= len(item.display_columns) <= 8 for item in configuration.table_contracts)
    supplemental = {
        item.table_id: item.supplemental_metric_sources
        for item in configuration.table_contracts
        if item.supplemental_metric_sources
    }
    assert set(supplemental) == {
        "primary_c2_vs_c1",
        "rare_pivotal",
        "entropy_clutter",
        "community",
        "mechanism_c2_vs_fixed",
    }
    sources_by_role = {
        source.source_role: source
        for sources in supplemental.values()
        for source in sources
    }
    primary_partition = {
        "C0": 36,
        "C1": 72,
        "C2": 72,
        "A-FixedSelect": 72,
    }
    for role in (
        "primary_unit_temporal_epistemic",
        "primary_unit_rare_pivotal",
        "primary_unit_entropy_clutter",
        "primary_unit_community",
    ):
        source = sources_by_role[role]
        assert {metric.expected_row_count for metric in source.metrics} == {252}
        assert {item.value: item.expected_rows_per_metric for item in source.partitions} == (
            primary_partition
        )
    cross_seed_source = sources_by_role["cross_seed_unit_community"]
    assert {metric.expected_row_count for metric in cross_seed_source.metrics} == {108}
    assert {
        item.value: item.expected_rows_per_metric for item in cross_seed_source.partitions
    } == {"C1": 36, "C2": 36, "A-FixedSelect": 36}
    mechanism_source = sources_by_role["contrastive_mechanism_diagnostics"]
    assert {metric.metric_name for metric in mechanism_source.metrics} == {
        "contrastive_decision_change_f1",
        "contrastive_collapse",
    }
    assert {metric.expected_row_count for metric in mechanism_source.metrics} == {84}
    assert {
        item.value: item.expected_rows_per_metric
        for item in mechanism_source.partitions
    } == {"C0": 12, "C1": 24, "C2": 24, "A-FixedSelect": 24}
    assert len(configuration.sections) == 19
    assert registry.tables[0].table_id == "study_status"


def test_public_source_manifest_fails_closed_on_remove_add_policy_and_symlink(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    public_paths = _copy_public_reproduction_sources(source_root)
    manifest = load_public_reproduction_source_manifest(
        source_root / "configs/study/public_reproduction_sources.json"
    )
    assert expand_public_reproduction_source_paths(
        manifest, source_root=source_root
    ) == public_paths

    required = source_root / "prompts/c2_query/prompt_v1.md"
    required.unlink()
    with pytest.raises(Phase7CompilationError, match="lacks required paths"):
        expand_public_reproduction_source_paths(manifest, source_root=source_root)
    required.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "prompts/c2_query/prompt_v1.md", required)

    unexpected = source_root / "src/story_projection_onto/operator_override.py"
    unexpected.write_text("# TEST-ONLY unexpected science source\n", encoding="utf-8")
    with pytest.raises(Phase7CompilationError, match="path inventory changed"):
        expand_public_reproduction_source_paths(manifest, source_root=source_root)
    unexpected.unlink()

    science = next(
        item
        for item in manifest.groups
        if item.category is PublicSourceCategory.SCIENCE_CODE
    )
    weakened = science.model_copy(update={"roots": ("src/story_projection_onto/metrics",)})
    with pytest.raises(ValueError, match="reviewed allowlist"):
        type(science).model_validate(weakened.model_dump(mode="json"))

    outside = tmp_path / "outside.py"
    outside.write_text("# TEST-ONLY outside source\n", encoding="utf-8")
    (source_root / "src/story_projection_onto/linked.py").symlink_to(outside)
    with pytest.raises(Phase7CompilationError, match="symlink is prohibited"):
        expand_public_reproduction_source_paths(manifest, source_root=source_root)
    (source_root / "src/story_projection_onto/linked.py").unlink()

    verification_required = source_root / "tests/integration/ui_browser_smoke.mjs"
    verification_required.unlink()
    with pytest.raises(Phase7CompilationError, match="lacks required paths"):
        expand_public_reproduction_source_paths(manifest, source_root=source_root)
    shutil.copyfile(ROOT / "tests/integration/ui_browser_smoke.mjs", verification_required)

    unexpected_test = source_root / "tests/unit/test_operator_override.py"
    unexpected_test.write_text("# TEST-ONLY unexpected verification source\n", encoding="utf-8")
    with pytest.raises(Phase7CompilationError, match="path inventory changed"):
        expand_public_reproduction_source_paths(manifest, source_root=source_root)
    unexpected_test.unlink()

    outside_test = tmp_path / "outside_test.py"
    outside_test.write_text("# TEST-ONLY outside verification source\n", encoding="utf-8")
    (source_root / "tests/unit/test_linked.py").symlink_to(outside_test)
    with pytest.raises(Phase7CompilationError, match="symlink is prohibited"):
        expand_public_reproduction_source_paths(manifest, source_root=source_root)


def test_verification_test_inventory_is_exact_and_public_safe() -> None:
    configuration = Phase7CompilerConfiguration.load(CONFIGURATION, source_root=ROOT)
    manifest = load_public_reproduction_source_manifest(
        ROOT / configuration.public_source_manifest_path
    )
    public_paths = expand_public_reproduction_source_paths(manifest, source_root=ROOT)
    verification_paths = tuple(path for path in public_paths if path.startswith("tests/"))
    assert len(verification_paths) == 128
    assert sum(path.endswith(".py") for path in verification_paths) == 115
    assert sum(path.endswith(".json") for path in verification_paths) == 12
    assert sum(path.endswith(".mjs") for path in verification_paths) == 1
    records = scan_public_entries(
        ROOT,
        tuple(
            PublicEntry(
                source_relative_path=relative,
                bundle_relative_path=relative,
                sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                release_class="public",
            )
            for relative in verification_paths
        ),
    )
    assert len(records) == 128


def test_phase6_guide_is_public_safe_and_recovery_guide_remains_excluded() -> None:
    configuration = Phase7CompilerConfiguration.load(CONFIGURATION, source_root=ROOT)
    manifest = load_public_reproduction_source_manifest(
        ROOT / configuration.public_source_manifest_path
    )
    public_paths = expand_public_reproduction_source_paths(manifest, source_root=ROOT)

    phase6_path = "docs/PHASE6_EXECUTION.md"
    assert phase6_path in public_paths
    assert "docs/FALLBACK_SECOND_RECOVERY.md" not in public_paths
    scan_public_bytes((ROOT / phase6_path).read_bytes(), relative_path=phase6_path)


def test_all_phase4_report_tables_have_frozen_exact_selection_contracts() -> None:
    configuration = Phase7CompilerConfiguration.load(CONFIGURATION, source_root=ROOT)
    contracts = {item.table_id: item for item in configuration.table_contracts}
    expected_counts = {
        "primary_c2_vs_c1": 2,
        "secondary_c2_vs_c0": 3,
        "mechanism_c2_vs_fixed": 4,
        "rare_pivotal": 5,
        "entropy_clutter": 32,
        "community": 52,
        "paraphrase_contrastive": 12,
        "ablations": 28,
    }
    assert {
        table_id: contracts[table_id].expected_output_row_count
        for table_id in expected_counts
    } == expected_counts
    assert all(contracts[item].source_producer_table_id for item in expected_counts)

    for table_id in (
        "primary_c2_vs_c1",
        "secondary_c2_vs_c0",
        "mechanism_c2_vs_fixed",
        "rare_pivotal",
        "entropy_clutter",
        "community",
    ):
        contract = contracts[table_id]
        rows = tuple(
            dict(zip(contract.row_identity_columns, identity, strict=True))
            for identity in sorted(contract.expected_row_identities())
        )
        _validate_registered_table_rows(contract, rows)
        with pytest.raises(Phase7CompilationError, match="row count changed"):
            _validate_registered_table_rows(contract, rows[:-1])
        substituted = list(rows)
        substituted[0] = {**substituted[0], "metric": "substituted_metric"}
        with pytest.raises(Phase7CompilationError, match="row identities changed"):
            _validate_registered_table_rows(contract, substituted)

        valid_binding = Phase7TableBinding(
            table_id=table_id,
            status="complete",
            reason="TEST-ONLY exact Phase 4 table binding.",
            scope="final",
            source_artifact_ids=("producer-csv", "producer-manifest"),
            source_table_artifact_id="producer-csv",
            producer_manifest_artifact_id="producer-manifest",
            source_row_count=116,
            output_row_count=contract.expected_output_row_count,
            row_filters=contract.registered_row_filters,
        )
        _validate_registered_table_binding(contract, valid_binding)
        tampered_binding = valid_binding.model_copy(update={"row_filters": ()})
        with pytest.raises(Phase7CompilationError, match="frozen row-selection"):
            _validate_registered_table_binding(contract, tampered_binding)
        missing_manifest = valid_binding.model_copy(
            update={"producer_manifest_artifact_id": None}
        )
        with pytest.raises(Phase7CompilationError, match="lacks its Phase 4 producer"):
            _validate_registered_table_binding(contract, missing_manifest)


def test_full_phase4_pair_tables_require_exact_paraphrase_and_ablation_inventories() -> None:
    configuration = Phase7CompilerConfiguration.load(CONFIGURATION, source_root=ROOT)
    contracts = {item.table_id: item for item in configuration.table_contracts}
    paraphrase_rows = tuple(
        {
            "call_id": f"paraphrase-{index:02d}",
            "context_id": f"context-{index:02d}",
            "world_id": f"world-{index:02d}",
        }
        for index in range(12)
    )
    _validate_registered_table_rows(
        contracts["paraphrase_contrastive"], paraphrase_rows
    )
    duplicate_world = list(paraphrase_rows)
    duplicate_world[-1] = {**duplicate_world[-1], "world_id": "world-00"}
    with pytest.raises(Phase7CompilationError, match="each of 12 worlds"):
        _validate_registered_table_rows(
            contracts["paraphrase_contrastive"], duplicate_world
        )

    principal = {
        "A-NoContext": "ontology_decision_macro_f1",
        "A-NoTemporalEpistemic": "essential_temporal_qualification_accuracy",
        "A-NoRareGuard": "rare_pivotal_qualified_assertion_recall",
    }
    counts = {"A-NoContext": 12, "A-NoTemporalEpistemic": 8, "A-NoRareGuard": 8}
    ablation_rows = tuple(
        {
            "call_id": f"{condition}-{index:02d}",
            "condition": condition,
            "context_id": f"{condition}-context-{index:02d}",
            "world_id": f"world-{index:02d}",
            "principal_metric": principal[condition],
        }
        for condition, count in counts.items()
        for index in range(count)
    )
    _validate_registered_table_rows(contracts["ablations"], ablation_rows)
    wrong_principal = list(ablation_rows)
    wrong_principal[0] = {**wrong_principal[0], "principal_metric": "wrong_metric"}
    with pytest.raises(Phase7CompilationError, match="12/8/8"):
        _validate_registered_table_rows(contracts["ablations"], wrong_principal)
    registry = load_phase7_source_registry(TEMPLATE)
    supplemental = {
        item.table_id: item.supplemental_metric_sources
        for item in configuration.table_contracts
        if item.supplemental_metric_sources
    }
    metric_names = {
        table_id: {
            metric.metric_name
            for source in sources
            for metric in source.metrics
        }
        for table_id, sources in supplemental.items()
    }
    source_metric_names = {
        source.source_role: {metric.metric_name for metric in source.metrics}
        for sources in supplemental.values()
        for source in sources
    }
    expected_primary_metrics = {
        "contextual_node_precision",
        "contextual_node_recall",
        "contextual_node_f1",
        "strict_qualified_assertion_precision",
        "strict_qualified_assertion_recall",
        "strict_qualified_assertion_f1",
        "essential_temporal_qualification_accuracy",
        "evidence_citation_validity",
        "grounding_precision",
        "unsupported_assertion_rate",
        "ontology_decision_macro_f1",
        "ontology_decision_unrepresentable_gold_count",
        "ontology_decision_unrepresentable_prediction_count",
    } | {
        f"ontology_decision_{family}_{suffix}"
        for family in (
            "merge_split",
            "contextual_type",
            "schema_relation",
            "event_reification",
            "abstraction",
            "temporal_epistemic_qualification",
        )
        for suffix in ("precision", "recall", "f1")
    }
    assert metric_names["primary_c2_vs_c1"] == expected_primary_metrics
    assert source_metric_names["primary_unit_temporal_epistemic"] == (
        expected_primary_metrics
    )
    assert metric_names["rare_pivotal"] == {
        "rare_pivotal_qualified_assertion_recall",
        "rare_pivotal_support_path_survival",
        "qualified_assertion_recall_common_nonpivotal",
        "qualified_assertion_recall_common_pivotal",
        "qualified_assertion_recall_rare_nonpivotal",
        "qualified_assertion_recall_rare_pivotal",
    }
    assert source_metric_names["primary_unit_rare_pivotal"] == metric_names[
        "rare_pivotal"
    ]
    assert metric_names["entropy_clutter"] == {
        "degree_histogram_entropy",
        "degree_histogram_entropy_raw",
        "degree_histogram_support_size",
        "occupied_degree_bin_count",
        "degree_histogram_denominator",
        "degree_mass_entropy",
        "degree_mass_entropy_raw",
        "degree_mass_support_size",
        "degree_mass_denominator",
        "local_relation_neighborhood_entropy",
        "local_relation_neighborhood_entropy_raw",
        "local_relation_neighborhood_support_size",
        "local_relation_neighborhood_denominator",
        "native_schema_relation_entropy",
        "native_schema_relation_entropy_raw",
        "native_schema_relation_support_size",
        "active_native_relation_count",
        "native_schema_relation_denominator",
        "canonical_mapped_relation_entropy",
        "canonical_mapped_relation_entropy_raw",
        "canonical_mapped_relation_support_size",
        "canonical_mapped_relation_denominator",
        "canonical_relation_vocabulary_size",
        "canonical_other_count",
        "canonical_other_rate",
        "isolate_count",
        "positive_degree_node_count",
        "node_count",
        "assertion_edge_count",
        "topology_edge_count",
        "density",
        "component_count",
        "label_overlap_count",
        "label_overlap_area",
        "irrelevant_visible_load",
        "crossing_opportunity_indicator",
        "crossing_opportunity_count",
        "crossing_count",
        "conditional_crossing_rate",
        "rare_pivotal_discoverability_mean_interactions",
        "rare_pivotal_discoverability_max_interactions",
        "rare_pivotal_discovered_count",
        "rare_pivotal_omitted_count",
    }
    assert source_metric_names["primary_unit_entropy_clutter"] == metric_names[
        "entropy_clutter"
    ]
    community_metric_bases = {
        "community_adjusted_mutual_information",
        "community_purity",
        "community_mean_conductance",
        "community_conductance_defined_cluster_count",
        "community_conductance_undefined_cluster_count",
        "community_fragmentation_error",
        "community_merging_error",
        "community_omitted_anchor_count",
    }
    expected_community_metrics = {
        f"leiden_{measure}_{resolution}"
        for measure in ("cluster_count", "modularity")
        for resolution in ("half", "base", "double")
    } | {
        f"{metric}{suffix}"
        for metric in community_metric_bases
        for suffix in ("", "_half", "_double")
    } | {
        "cross_seed_community_adjusted_mutual_information",
        "cross_seed_community_variation_of_information",
        "cross_seed_community_omitted_seed_1_count",
        "cross_seed_community_omitted_seed_2_count",
    }
    assert metric_names["community"] == expected_community_metrics
    assert source_metric_names["cross_seed_unit_community"] == {
        metric
        for metric in expected_community_metrics
        if metric.startswith("cross_seed_")
    }
    assert source_metric_names["primary_unit_community"] == {
        metric
        for metric in expected_community_metrics
        if not metric.startswith("cross_seed_")
    }
    assert len(configuration.sections) == 19
    assert registry.tables[0].table_id == "study_status"


def test_generated_complete_study_status_inherits_all_phase_lineage() -> None:
    phases = tuple(
        PhaseInputState(
            phase_id=f"phase_{ordinal}",
            status="complete",
            reason=f"TEST-ONLY phase {ordinal} is complete.",
            source_artifact_ids=(
                f"phase-{ordinal}-source",
                "shared-environment-source",
            ),
        )
        for ordinal in range(1, 8)
    )

    assert _study_status_source_artifact_ids(phases) == (
        "phase-1-source",
        "phase-2-source",
        "phase-3-source",
        "phase-4-source",
        "phase-5-source",
        "phase-6-source",
        "phase-7-source",
        "shared-environment-source",
    )


def _small_metric_source_contract() -> SupplementalMetricSourceContract:
    return SupplementalMetricSourceContract(
        source_role="test_metric_rows",
        producer_table_id="report_test_metrics_csv",
        description="TEST-ONLY exact supplemental metric rows.",
        required_columns=(
            "source_block",
            "unit_id",
            "condition",
            "metric_name",
            "value",
            "metric_version_hash",
        ),
        key_columns=("unit_id",),
        row_filters=(MetricSourceFilter(column="source_block", equals="primary"),),
        metrics=(MetricRowRequirement(metric_name="registered_metric", expected_row_count=2),),
        partitions=(
            MetricPartitionRequirement(
                column="condition", value="C0", expected_rows_per_metric=1
            ),
            MetricPartitionRequirement(
                column="condition", value="C2", expected_rows_per_metric=1
            ),
        ),
    )


def test_supplemental_metric_source_is_exact_and_phase4_manifest_bound(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tables/test_metrics.csv"
    source.parent.mkdir()
    source.write_text(
        "source_block,unit_id,condition,metric_name,value,metric_version_hash\n"
        f"primary,u-0,C0,registered_metric,0.25,{'a' * 64}\n"
        f"primary,u-1,C2,registered_metric,0.75,{'a' * 64}\n"
        f"combined,u-2,C2,another_metric,0.5,{'a' * 64}\n",
        encoding="utf-8",
        newline="\n",
    )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    total, selected, counts, columns, metric_version_hash = (
        _inspect_supplemental_metric_source_csv(
            source,
            expected_hash=source_hash,
            contract=_small_metric_source_contract(),
        )
    )
    assert (total, selected) == (3, 2)
    assert metric_version_hash == "a" * 64
    assert [(item.metric_name, item.row_count) for item in counts] == [
        ("registered_metric", 2)
    ]

    manifest_payload = {
        "schema_version": "1.0.0",
        "manifest_id": "test-phase4-tables",
        "analysis_configuration_hash": "b" * 64,
        "metric_version_hash": "a" * 64,
        "independent_confirmatory_unit": "world",
        "tables": [
            {
                "table_id": "report_test_metrics_csv",
                "relative_path": "tables/test_metrics.csv",
                "columns": list(columns),
                "row_count": total,
                "file_sha256": source_hash,
                "logical_content_hash": "c" * 64,
                "release_class": "public",
                "independent_unit": "metric_observation",
                "selection_ready": False,
            }
        ],
    }
    manifest_payload["content_hash"] = canonical_sha256(manifest_payload)
    manifest = tmp_path / "table_manifest.json"
    manifest.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    source_spec = SourceArtifactSpec(
        artifact_id="test-metrics",
        family="heldout",
        relative_path="tables/test_metrics.csv",
        file_sha256=source_hash,
        media_type="text/csv",
        release_class="public",
    )
    _verify_phase4_table_manifest_binding(
        manifest,
        expected_manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source=source_spec,
        source_columns=columns,
        source_row_count=total,
        producer_table_id="report_test_metrics_csv",
        metric_version_hash=metric_version_hash,
    )
    manifest_payload["tables"][0]["row_count"] = total - 1
    manifest_payload["content_hash"] = canonical_sha256(
        {key: value for key, value in manifest_payload.items() if key != "content_hash"}
    )
    manifest.write_text(
        json.dumps(
            manifest_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(Phase7CompilationError, match="manifest metadata mismatch"):
        _verify_phase4_table_manifest_binding(
            manifest,
            expected_manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            source=source_spec,
            source_columns=columns,
            source_row_count=total,
            producer_table_id="report_test_metrics_csv",
            metric_version_hash=metric_version_hash,
        )


def test_supplemental_metric_source_rejects_missing_or_duplicated_registered_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "metrics.csv"
    source.write_text(
        "source_block,unit_id,condition,metric_name,value,metric_version_hash\n"
        f"primary,u-0,C0,registered_metric,0.25,{'a' * 64}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(Phase7CompilationError, match="row inventory changed"):
        _inspect_supplemental_metric_source_csv(
            source,
            expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            contract=_small_metric_source_contract(),
        )

    source.write_text(
        "source_block,unit_id,condition,metric_name,value,metric_version_hash\n"
        f"primary,u-0,C0,registered_metric,0.25,{'a' * 64}\n"
        f"primary,u-0,C2,registered_metric,0.75,{'a' * 64}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(Phase7CompilationError, match="repeats a registered key"):
        _inspect_supplemental_metric_source_csv(
            source,
            expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            contract=_small_metric_source_contract(),
        )

    source.write_text(
        "source_block,unit_id,condition,metric_name,value,metric_version_hash\n"
        f"primary,u-0,C2,registered_metric,0.25,{'a' * 64}\n"
        f"primary,u-1,C2,registered_metric,0.75,{'a' * 64}\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(Phase7CompilationError, match="partition inventory changed"):
        _inspect_supplemental_metric_source_csv(
            source,
            expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            contract=_small_metric_source_contract(),
        )


def test_shared_canonical_csv_is_split_only_by_exact_declared_filter(tmp_path: Path) -> None:
    source = tmp_path / "comparisons.csv"
    source.write_text(
        "analysis_family,comparison,metric\n"
        "primary,C2-C1,strict_f1\n"
        "secondary,C2-C0,strict_f1\n"
        "primary,C2-C1,decision_f1\n",
        encoding="utf-8",
        newline="\n",
    )
    contract = TableContract(
        table_id="primary_c2_vs_c1",
        producer=TableProducer.CANONICAL_CSV,
        required_columns=("analysis_family", "comparison", "metric"),
        sort_columns=("metric",),
        display_columns=("comparison", "metric"),
        description="TEST-ONLY primary table.",
    )
    _, rows, payload = _canonical_source_csv(
        source,
        expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        expected_rows=3,
        expected_output_rows=2,
        row_filters=(ExactRowFilter(column="analysis_family", equals="primary"),),
        contract=contract,
    )
    assert [item["metric"] for item in rows] == ["decision_f1", "strict_f1"]
    assert payload.endswith(b"primary,C2-C1,strict_f1\n")
    with pytest.raises(Phase7CompilationError, match="filter count changed"):
        _canonical_source_csv(
            source,
            expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_rows=3,
            expected_output_rows=3,
            row_filters=(ExactRowFilter(column="analysis_family", equals="primary"),),
            contract=contract,
        )


def test_singleton_gate_csv_is_hash_bound_and_exactly_one_row(tmp_path: Path) -> None:
    source = tmp_path / "report_gate_status.csv"
    source.write_text(
        "organization_gate_passed,construction_freedom_gate_passed\n"
        "true,false\n",
        encoding="utf-8",
        newline="\n",
    )
    columns, row = _canonical_singleton_csv(
        source,
        expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        table_id="mechanism_c2_vs_fixed",
    )
    assert columns == (
        "organization_gate_passed",
        "construction_freedom_gate_passed",
    )
    assert row["organization_gate_passed"] == "true"

    source.write_text(
        "organization_gate_passed,construction_freedom_gate_passed\n"
        "true,false\n"
        "false,false\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(Phase7CompilationError, match="exactly one row"):
        _canonical_singleton_csv(
            source,
            expected_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            table_id="mechanism_c2_vs_fixed",
        )


def test_mechanism_table_requires_endpoint_safeguard_change_and_collapse_rows() -> None:
    rows = (
        {
            "analysis_family": "gated_mechanism",
            "comparison": "C2-A-FixedSelect",
            "metric": "ontology_decision_macro_f1",
        },
        {
            "analysis_family": "gated_mechanism_safeguard",
            "comparison": "C2-A-FixedSelect",
            "metric": "rare_pivotal_qualified_assertion_recall",
        },
        {
            "analysis_family": "mechanism_support",
            "comparison": "C2-A-FixedSelect",
            "metric": "contrastive_decision_change_f1",
        },
        {
            "analysis_family": "mechanism_support",
            "comparison": "C2-A-FixedSelect",
            "metric": "contrastive_collapse",
        },
    )
    _validate_mechanism_table_rows(rows)
    with pytest.raises(Phase7CompilationError, match="change-F1, or collapse"):
        _validate_mechanism_table_rows(rows[:-1])


def test_singleton_gate_csv_must_be_in_table_lineage() -> None:
    with pytest.raises(ValueError, match="singleton join CSV must be part of table lineage"):
        Phase7TableBinding(
            table_id="mechanism_c2_vs_fixed",
            status="complete",
            reason="TEST-ONLY complete binding.",
            scope="final",
            source_artifact_ids=("comparisons",),
            source_table_artifact_id="comparisons",
            source_row_count=3,
            output_row_count=1,
            singleton_join_artifact_id="report-gates",
            singleton_join_row_count=1,
        )


def test_qualitative_rules_choose_fixed_units_and_itt_failure_case() -> None:
    selected = _choose_qualitative_candidates(_candidate_set(), policy_path=POLICY)
    by_example = {example.value: candidate.candidate_id for example, candidate, _ in selected}
    assert by_example["development_tutorial"] == "tutorial"
    assert by_example["temporal_epistemic"] == "temporal-holder"
    assert by_example["held_out_illustration"] == "held"
    assert by_example["counterexample"] == "counter-itt-failure"
    assert by_example["narrative_illustration"] == "narrative"


def test_qualitative_comparison_requires_every_condition_in_each_context() -> None:
    candidates = _candidate_set()
    tutorial = next(item for item in candidates.candidates if item.candidate_id == "tutorial")
    payload = tutorial.model_dump(mode="json")
    payload["display_rows"] = [
        item
        for item in payload["display_rows"]
        if not (
            item["context_id"] == "ctx-b"
            and item["condition"] == "a_fixed_select"
        )
    ]
    incomplete = QualitativeCandidate.model_validate(payload)
    changed = candidates.model_copy(
        update={
            "candidates": tuple(
                incomplete if item.candidate_id == "tutorial" else item
                for item in candidates.candidates
            )
        }
    )
    with pytest.raises(Phase7CompilationError, match="tutorial/ctx-b"):
        _choose_qualitative_candidates(changed, policy_path=POLICY)


def test_non_success_counterexample_score_cannot_escape_itt_zero() -> None:
    with pytest.raises(ValueError, match="registered ITT zero"):
        C2IntentionToTreatScore(
            seed_block=1,
            outcome="invalid",
            strict_qualified_assertion_f1=0.1,
        )


def test_incomplete_production_compile_is_reproducible_and_has_no_efficacy_rows(
    tmp_path: Path,
) -> None:
    pytest.importorskip("reportlab")
    source_root = tmp_path / "source"
    source_root.mkdir()
    _copy_public_reproduction_sources(source_root)
    registry_path = source_root / "registry.json"
    shutil.copyfile(TEMPLATE, registry_path)
    output = source_root / "reports"
    result = compile_phase7_results(
        source_root=source_root,
        configuration_path=source_root / "configs/study/phase7_compiler.json",
        registry_path=registry_path,
        output_root=output,
    )
    assert result.study_status.value == "blocked"
    assert (output / "RESULTS_REPORT.pdf").read_bytes().startswith(b"%PDF-")
    assert not tuple((output / "tables").glob("primary_c2_vs_c1.*.csv"))
    status = next((output / "tables").glob("study_status.*.csv"))
    assert len(status.read_text(encoding="utf-8").splitlines()) == 8
    selection = next((output / "qualitative").glob("selection_manifest.*.json"))
    assert json.loads(selection.read_text(encoding="utf-8"))["selection_count"] == 0
    reproduced = verify_phase7_results(
        source_root=source_root,
        configuration_path=source_root / "configs/study/phase7_compiler.json",
        registry_path=registry_path,
        output_root=output,
    )
    assert reproduced.output_hashes == result.output_hashes
