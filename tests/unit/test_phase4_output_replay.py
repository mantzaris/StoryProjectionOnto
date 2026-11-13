from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import story_projection_onto.scorer_only.phase4_replay as replay_module
from story_projection_onto.contracts import ConditionName, ReleaseClass, canonical_sha256
from story_projection_onto.metrics.alignment import AlignmentPlan
from story_projection_onto.metrics.common import revalidated_copy
from story_projection_onto.metrics.config import StudyMetricConfiguration
from story_projection_onto.metrics.pipeline import (
    FailedMetricOutput,
    IntendedMetricUnit,
    OutputFailureKind,
    ScorerMetricPlan,
    score_failed_output,
)
from story_projection_onto.scorer_only.phase4_analysis import Phase4AnalysisError
from story_projection_onto.scorer_only.phase4_replay import (
    _FIXED_ROW_COUNTS,
    _NON_TABLE_OUTPUTS,
    _TABLE_CONTRACTS,
    _canonical_jsonl,
    _exact_file_inventory,
    _recompute_intended_cell_scores,
    _validate_registered_inventory_metadata,
)

DIGEST = "a" * 64
CONFIGURATION_HASH = "b" * 64
METRIC_CONFIGURATION_HASH = "c" * 64
METRIC_VERSION_HASH = "d" * 64
ROOT = Path(__file__).resolve().parents[2]


def _empty_scorer_plan() -> ScorerMetricPlan:
    alignment = AlignmentPlan(
        matcher_revision="matcher-v1",
        source_gold_hash="1" * 64,
        source_alternative_set_hash="2" * 64,
        node_targets=(),
        assertion_targets=(),
    )
    return ScorerMetricPlan(
        plan_id="replay-plan",
        source_gold_hash=alignment.source_gold_hash,
        alignment_plan=alignment,
        gold_decisions=(),
        rare_annotations=(),
        valid_evidence_ids=(),
        valid_evidence_manifest_hash=canonical_sha256(()),
    )


def _intended_unit(plan: ScorerMetricPlan) -> IntendedMetricUnit:
    return IntendedMetricUnit(
        unit_id="replay-unit",
        job_id="replay-job",
        condition=ConditionName.C2_LLM_QUERY,
        world_id="world-1",
        context_id="context-1",
        seed_block=1,
        snapshot_hash="3" * 64,
        packet_hash="4" * 64,
        context_hash="5" * 64,
        scorer_plan_hash=plan.content_hash,
        relevant_node_gold_count=0,
        strict_assertion_gold_count=0,
        ontology_decision_gold_count=0,
        rare_pivotal_gold_count=0,
    )


def _inventory_fixture() -> tuple[
    SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace
]:
    release_by_path = {
        **_NON_TABLE_OUTPUTS,
        **{
            contract.relative_path: contract.release_class
            for contract in _TABLE_CONTRACTS.values()
        },
        "table_manifest.json": ReleaseClass.PUBLIC,
    }
    outputs = tuple(
        SimpleNamespace(
            relative_path=path,
            file_sha256=DIGEST,
            logical_content_hash=DIGEST,
            row_count=_FIXED_ROW_COUNTS.get(path, 0),
            release_class=release,
        )
        for path, release in sorted(release_by_path.items())
    )
    output_by_path = {item.relative_path: item for item in outputs}
    tables = tuple(
        SimpleNamespace(
            table_id=table_id,
            relative_path=contract.relative_path,
            columns=contract.columns,
            row_count=output_by_path[contract.relative_path].row_count,
            file_sha256=DIGEST,
            logical_content_hash=DIGEST,
            release_class=contract.release_class,
            independent_unit=contract.independent_unit,
            selection_ready=contract.selection_ready,
        )
        for table_id, contract in sorted(_TABLE_CONTRACTS.items())
    )
    index = SimpleNamespace(
        analysis_configuration_hash=CONFIGURATION_HASH,
        metric_configuration_hash=METRIC_CONFIGURATION_HASH,
        metric_version_hash=METRIC_VERSION_HASH,
        output_files=outputs,
    )
    manifest = SimpleNamespace(
        analysis_configuration_hash=CONFIGURATION_HASH,
        metric_version_hash=METRIC_VERSION_HASH,
        tables=tables,
    )
    configuration = SimpleNamespace(content_hash=CONFIGURATION_HASH)
    metric_configuration = SimpleNamespace(
        content_hash=METRIC_CONFIGURATION_HASH,
        metric_version_hash=METRIC_VERSION_HASH,
    )
    return index, manifest, configuration, metric_configuration


def test_registered_phase4_contract_requires_36_outputs_and_26_tables() -> None:
    index, manifest, configuration, metric_configuration = _inventory_fixture()

    outputs = _validate_registered_inventory_metadata(
        index=index,
        table_manifest=manifest,
        configuration=configuration,
        metric_configuration=metric_configuration,
    )

    assert len(outputs) == 36
    assert len(_TABLE_CONTRACTS) == 26
    assert "analysis/registered_analysis.json" in outputs
    assert "analysis/report_gate_status.json" in outputs
    assert "tables/comparisons.csv" in outputs
    assert "table_manifest.json" in outputs


def test_registered_phase4_contract_rejects_subset_and_column_drift() -> None:
    index, manifest, configuration, metric_configuration = _inventory_fixture()
    with pytest.raises(Phase4AnalysisError, match="exact output contract"):
        _validate_registered_inventory_metadata(
            index=SimpleNamespace(
                **{
                    **vars(index),
                    "output_files": index.output_files[:-1],
                }
            ),
            table_manifest=manifest,
            configuration=configuration,
            metric_configuration=metric_configuration,
        )

    first = manifest.tables[0]
    changed = SimpleNamespace(**{**vars(first), "columns": (*first.columns, "drift")})
    with pytest.raises(Phase4AnalysisError, match="registered table contract changed"):
        _validate_registered_inventory_metadata(
            index=index,
            table_manifest=SimpleNamespace(
                **{
                    **vars(manifest),
                    "tables": (changed, *manifest.tables[1:]),
                }
            ),
            configuration=configuration,
            metric_configuration=metric_configuration,
        )


def test_registered_phase4_contract_rejects_configuration_or_metric_drift() -> None:
    index, manifest, _, metric_configuration = _inventory_fixture()

    with pytest.raises(Phase4AnalysisError, match="frozen analysis or metric configuration"):
        _validate_registered_inventory_metadata(
            index=index,
            table_manifest=manifest,
            configuration=SimpleNamespace(content_hash="e" * 64),
            metric_configuration=metric_configuration,
        )


def test_phase4_jsonl_replay_requires_canonical_bytes(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    canonical.write_bytes(b'{"a":1,"b":2}\n')
    assert _canonical_jsonl(canonical) == ({"a": 1, "b": 2},)

    noncanonical = tmp_path / "noncanonical.jsonl"
    noncanonical.write_bytes(b'{"b": 2, "a": 1}\n')
    with pytest.raises(Phase4AnalysisError, match="noncanonical"):
        _canonical_jsonl(noncanonical)


def test_phase4_output_replay_rejects_extra_or_symlinked_files(tmp_path: Path) -> None:
    expected = tmp_path / "expected.json"
    expected.write_text("{}\n", encoding="utf-8")
    _exact_file_inventory(tmp_path, {"expected.json"})

    extra = tmp_path / "extra.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Phase4AnalysisError, match="output inventory changed"):
        _exact_file_inventory(tmp_path, {"expected.json"})

    extra.unlink()
    link = tmp_path / "link.json"
    link.symlink_to(expected)
    with pytest.raises(Phase4AnalysisError, match="symlinked"):
        _exact_file_inventory(tmp_path, {"expected.json"})


def test_phase4_replay_recomputes_failed_itt_rows_instead_of_trusting_them() -> None:
    configuration = StudyMetricConfiguration.load(ROOT / "configs/study/metrics.json")
    plan = _empty_scorer_plan()
    intended = _intended_unit(plan)
    score = score_failed_output(
        FailedMetricOutput(
            intended_unit=intended,
            failure_kind=OutputFailureKind.TIMEOUT,
            failure_artifact_hash="6" * 64,
            allocated_gpu_seconds=19.5,
        ),
        configuration=configuration,
        scorer_plan=plan,
    )

    _recompute_intended_cell_scores(
        scores=(score,),
        complete_bundles=(),
        scorer_plans=(plan,),
        grounding_audits=(),
        geometries=(),
        metric_configuration=configuration,
    )

    forged_row = revalidated_copy(score.rows[0], unit_hash="7" * 64)
    forged_score = revalidated_copy(score, rows=(forged_row, *score.rows[1:]))
    with pytest.raises(
        Phase4AnalysisError,
        match="failed-output ITT score failed deterministic recomputation",
    ):
        _recompute_intended_cell_scores(
            scores=(forged_score,),
            complete_bundles=(),
            scorer_plans=(plan,),
            grounding_audits=(),
            geometries=(),
            metric_configuration=configuration,
        )


def test_phase4_replay_rejects_coherently_relinked_valid_bundle_and_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing a bundle, its hash, and the score link cannot pass replay together."""

    projection = SimpleNamespace(content_hash="8" * 64)
    replay = SimpleNamespace(
        projection=projection,
        context=SimpleNamespace(),
        evidence_packet=SimpleNamespace(),
    )
    plan = SimpleNamespace(content_hash="9" * 64)
    intended = SimpleNamespace(scorer_plan_hash=plan.content_hash)
    grounding_audit = SimpleNamespace(projection_hash=projection.content_hash)
    canonical_bundle = SimpleNamespace(
        content_hash="a" * 64,
        scorer_only_replay=replay,
    )
    canonical_score = SimpleNamespace(
        intended_unit=intended,
        output_valid=True,
        projection_bundle_hash=canonical_bundle.content_hash,
    )
    configuration = SimpleNamespace()
    monkeypatch.setattr(
        replay_module,
        "_grounding_audit",
        lambda candidate_projection, candidate_plan: grounding_audit,
    )
    monkeypatch.setattr(
        replay_module,
        "score_complete_projection",
        lambda *args, **kwargs: canonical_bundle,
    )
    monkeypatch.setattr(
        replay_module,
        "score_intended_projection",
        lambda *args, **kwargs: canonical_score,
    )

    _recompute_intended_cell_scores(
        scores=(canonical_score,),
        complete_bundles=(canonical_bundle,),
        scorer_plans=(plan,),
        grounding_audits=(grounding_audit,),
        geometries=(),
        metric_configuration=configuration,
    )

    forged_bundle = SimpleNamespace(
        content_hash="b" * 64,
        scorer_only_replay=replay,
        forged_metric_payload=True,
    )
    forged_score = SimpleNamespace(
        intended_unit=intended,
        output_valid=True,
        projection_bundle_hash=forged_bundle.content_hash,
        forged_rows=True,
    )
    with pytest.raises(
        Phase4AnalysisError,
        match="valid score/bundle failed deterministic recomputation",
    ):
        _recompute_intended_cell_scores(
            scores=(forged_score,),
            complete_bundles=(forged_bundle,),
            scorer_plans=(plan,),
            grounding_audits=(grounding_audit,),
            geometries=(),
            metric_configuration=configuration,
        )
