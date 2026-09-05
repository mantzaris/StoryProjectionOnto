from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.contracts import ConditionName, RunOutcome
from story_projection_onto.metrics.pipeline import (
    IntendedMetricUnit,
    IntendedUnitScore,
    MetricResultRow,
    OutputFailureKind,
    PipelineMetricStatus,
)
from story_projection_onto.scorer_only.blinded_postrun_review import (
    FROZEN_REGISTERED_FAILURE_SIGNAL_GATES,
    BlindedReviewError,
    _neutral_failure_panel,
    _primary_score_lineage,
    _score_failure_gate_ids,
    prepare_held_out_failure_source,
)

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "a" * 64
NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _receipt(index: int, **values: object) -> SimpleNamespace:
    return SimpleNamespace(content_hash=f"{index:064x}", **values)


def _primary_runtime_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    units = []
    openings = []
    projections = []
    c0_receipts = []
    calls = []
    itt_records = []
    receipt_index = 1
    ordinal = 1
    for world in range(1, 13):
        unit_id = f"opaque-unit-{world:02d}"
        stages = []
        c0_receipts.append(_receipt(receipt_index, unit_id=unit_id))
        receipt_index += 1
        for context in range(1, 4):
            stage_hash = f"{world * 10 + context:064x}"
            stage = SimpleNamespace(
                staging_manifest_hash=stage_hash,
                stage_id=f"opaque-stage-{world:02d}-{context}",
                snapshot_hash=f"{1000 + world * 10 + context:064x}",
            )
            stages.append(stage)
            query_context = SimpleNamespace(
                context_id=f"context-{world:02d}-{context}",
                content_hash=f"{2000 + world * 10 + context:064x}",
            )
            openings.append(
                SimpleNamespace(
                    sealed_stage_hash=stage_hash,
                    query_context=query_context,
                    evidence_packet_hash=f"{3000 + world * 10 + context:064x}",
                )
            )
            for condition, seeds in (
                (ConditionName.C0_CLASSICAL_PRE, (None,)),
                (ConditionName.C1_LLM_PRE, (1, 2)),
            ):
                for seed in seeds:
                    projections.append(
                        _receipt(
                            receipt_index,
                            unit_id=unit_id,
                            query_stage_hash=stage_hash,
                            condition=condition,
                            seed_block=seed,
                            outcome=RunOutcome.SUCCEEDED,
                        )
                    )
                    receipt_index += 1
        units.append(SimpleNamespace(unit_id=unit_id, query_stages=tuple(stages)))

    for unit in units:
        for seed in (1, 2):
            call = _receipt(
                10_000 + ordinal,
                ordinal=ordinal,
                call_id=f"call-{ordinal:03d}",
                unit_id=unit.unit_id,
                condition=ConditionName.C1_LLM_PRE,
                seed_block=seed,
                query_stage_hash=None,
            )
            calls.append(call)
            itt_records.append(
                _receipt(
                    receipt_index,
                    ordinal=ordinal,
                    call_spec_hash=call.content_hash,
                    result=SimpleNamespace(
                        call_id=call.call_id,
                        condition=call.condition,
                        outcome=RunOutcome.SUCCEEDED,
                    ),
                )
            )
            receipt_index += 1
            ordinal += 1
        for stage in unit.query_stages:
            for condition in (
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_FIXED_SELECT,
            ):
                for seed in (1, 2):
                    call = _receipt(
                        10_000 + ordinal,
                        ordinal=ordinal,
                        call_id=f"call-{ordinal:03d}",
                        unit_id=unit.unit_id,
                        condition=condition,
                        seed_block=seed,
                        query_stage_hash=stage.staging_manifest_hash,
                    )
                    calls.append(call)
                    itt_records.append(
                        _receipt(
                            receipt_index,
                            ordinal=ordinal,
                            call_spec_hash=call.content_hash,
                            result=SimpleNamespace(
                                call_id=call.call_id,
                                condition=call.condition,
                                outcome=RunOutcome.SUCCEEDED,
                            ),
                        )
                    )
                    receipt_index += 1
                    ordinal += 1

    call_manifest = SimpleNamespace(
        content_hash="f" * 64,
        units=tuple(units),
        calls=tuple(calls),
    )
    execution = SimpleNamespace(
        call_manifest_hash=call_manifest.content_hash,
        c0_constructions=tuple(c0_receipts),
        query_openings=tuple(openings),
        preconstructed_projections=tuple(projections),
        itt_records=tuple(itt_records),
    )
    return call_manifest, execution


def _score(*, strict_f1: float, unsupported_rate: float) -> IntendedUnitScore:
    intended = IntendedMetricUnit(
        unit_id="primary:opaque-unit:opaque-stage:C2:s1",
        job_id="phase4-primary-opaque",
        condition=ConditionName.C2_LLM_QUERY,
        world_id="world-01",
        context_id="context-01",
        seed_block=1,
        snapshot_hash="1" * 64,
        packet_hash="2" * 64,
        context_hash="3" * 64,
        scorer_plan_hash="4" * 64,
        relevant_node_gold_count=1,
        strict_assertion_gold_count=1,
        ontology_decision_gold_count=1,
        rare_pivotal_gold_count=1,
    )
    values = {
        gate.metric_name: (
            strict_f1
            if gate.metric_name == "strict_qualified_assertion_f1"
            else unsupported_rate
            if gate.metric_name == "unsupported_assertion_rate"
            else 0
            if gate.direction.value == "above"
            else 1
        )
        for gate in FROZEN_REGISTERED_FAILURE_SIGNAL_GATES
    }
    return IntendedUnitScore(
        intended_unit=intended,
        output_valid=True,
        projection_id="projection-opaque",
        projection_bundle_hash="5" * 64,
        rows=tuple(
            MetricResultRow(
                projection_id="projection-opaque",
                unit_hash="6" * 64,
                metric_name=name,
                metric_version_hash="7" * 64,
                status=PipelineMetricStatus.VALUE,
                value=value,
            )
            for name, value in sorted(values.items())
        ),
    )


def test_primary_lineage_binds_252_scores_and_all_288_receipts() -> None:
    call_manifest, execution = _primary_runtime_fixture()

    lineage, inventory_hash = _primary_score_lineage(
        call_manifest=call_manifest,
        execution=execution,
    )

    assert len(lineage) == 252
    assert len(execution.c0_constructions) == 12
    assert len(execution.preconstructed_projections) == 108
    assert len(execution.itt_records) == 168
    assert len(inventory_hash) == 64

    execution.preconstructed_projections = execution.preconstructed_projections[:-1]
    with pytest.raises(BlindedReviewError, match="108-cell inventory"):
        _primary_score_lineage(call_manifest=call_manifest, execution=execution)


def test_registered_failure_signal_mapping_is_fixed_and_directional() -> None:
    perfect = _score(strict_f1=1.0, unsupported_rate=0.0)
    failed = _score(strict_f1=0.75, unsupported_rate=0.25)

    assert _score_failure_gate_ids(perfect, output_status="succeeded") == ()
    assert _score_failure_gate_ids(failed, output_status="succeeded") == (
        "strict-qualified-assertion-f1-below-1",
        "unsupported-assertion-rate-above-0",
    )

    invalid = IntendedUnitScore(
        intended_unit=perfect.intended_unit,
        output_valid=False,
        failure_kind=OutputFailureKind.VALIDATION_INVALID,
        rows=tuple(
            MetricResultRow(
                unit_hash=row.unit_hash,
                metric_name=row.metric_name,
                metric_version_hash=row.metric_version_hash,
                status=PipelineMetricStatus.INVALID,
            )
            for row in perfect.rows
        ),
    )
    assert _score_failure_gate_ids(invalid, output_status="invalid") == (
        "output-status-invalid",
    )


def test_generated_reviewer_panel_replaces_every_experimental_identifier() -> None:
    score = _score(strict_f1=0.75, unsupported_rate=0.0)
    adapter = SimpleNamespace(
        node_ids=("C2-world-01-node-a", "context-01-node-b"),
        edges=(
            SimpleNamespace(
                edge_id="world-01-edge",
                source_id="C2-world-01-node-a",
                target_id="context-01-node-b",
                native_predicate_id="secret-relation",
                canonical_predicate_id=None,
            ),
        ),
        assertion_relations=(
            SimpleNamespace(
                native_predicate_id="secret-relation",
                canonical_predicate_id=None,
            ),
        ),
        normalized_decisions=(
            SimpleNamespace(
                family=SimpleNamespace(value="merge_split"),
                operator=SimpleNamespace(value="merge"),
                anchor_ids=("secret-evidence-1",),
            ),
        ),
        structurally_valid=True,
        projection_hash="8" * 64,
    )
    bundle = SimpleNamespace(
        projection=SimpleNamespace(adapter=adapter),
        metric_rows=score.rows,
    )

    panel = _neutral_failure_panel(
        score=score,
        bundle=bundle,
        output_status="succeeded",
        failed_gate_ids=("strict-qualified-assertion-f1-below-1",),
    )
    visible = panel.to_canonical_json()

    for secret in (
        "C2-world-01-node-a",
        "context-01-node-b",
        "world-01-edge",
        "secret-relation",
        "secret-evidence-1",
        "8" * 64,
    ):
        assert secret not in visible
    assert "node-001" in visible
    assert "relation-001" in visible


def test_failure_source_refuses_a_partial_phase4_directory(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    phase4.mkdir()
    (phase4 / "analysis_index.json").write_text("{}\n", encoding="utf-8")
    held_out = tmp_path / "held-out"
    held_out.mkdir()

    with pytest.raises(BlindedReviewError, match="complete Phase 4 replay failed"):
        prepare_held_out_failure_source(
            repository=ROOT,
            phase4_configuration_path=ROOT / "configs/study/phase4_analysis.json",
            phase4_output_root=phase4,
            held_out_root=held_out,
            selected_at=NOW,
        )
