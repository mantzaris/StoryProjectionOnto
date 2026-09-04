from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from story_projection_onto.phase7_compiler import (
    C2IntentionToTreatScore,
    ExactRowFilter,
    Phase7CompilationError,
    Phase7CompilerConfiguration,
    Phase7TableBinding,
    QualitativeCandidate,
    QualitativeCandidateSet,
    QualitativeDisplayRow,
    QualitativeExample,
    TableContract,
    TableProducer,
    _canonical_singleton_csv,
    _canonical_source_csv,
    _choose_qualitative_candidates,
    compile_phase7_results,
    load_phase7_source_registry,
    verify_phase7_results,
)
from story_projection_onto.reporting import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION = ROOT / "configs/study/phase7_compiler.json"
TEMPLATE = ROOT / "configs/study/phase7_source_registry.template.json"
POLICY = ROOT / "configs/study/reporting.json"
HASH = "1" * 64


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
    assert len(configuration.table_contracts) == 14
    assert len(configuration.sections) == 19
    assert registry.tables[0].table_id == "study_status"


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
    configuration = json.loads(CONFIGURATION.read_text(encoding="utf-8"))
    for relative in configuration["public_static_paths"]:
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
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
