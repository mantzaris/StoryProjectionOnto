from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import timedelta
from pathlib import Path

import pytest

from story_projection_onto.contracts import ConditionName, FeedbackResolutionStatus
from story_projection_onto.feedback_runtime import FeedbackEpisodeKind
from story_projection_onto.metrics.feedback import score_scripted_feedback
from story_projection_onto.phase5_execution import execute_phase5
from story_projection_onto.scorer_only.phase5_feedback import (
    METRICS_FILE_NAME,
    RECEIPT_FILE_NAME,
    SESSION_FILE_NAME,
    Phase5FeedbackScoringReceipt,
    Phase5FeedbackScoringSession,
    Phase5ProjectionGoldScore,
    Phase5ScriptedFeedbackMetricRecord,
    Phase5ScriptedFeedbackMetrics,
    _verify_journal,
)
from story_projection_onto.scorer_only.phase5_report import (
    FEEDBACK_TABLE_COLUMNS,
    FEEDBACK_TABLE_FILE_NAME,
    Phase5FeedbackReportError,
    materialize_phase5_feedback_table,
    prepare_phase5_feedback_table,
    verify_phase5_feedback_table,
)
from tests.unit.test_phase5_execution import _Adapter, _inputs, _Verifier
from tests.unit.test_ui import NOW, digest


def _canonical_bytes(value) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _gold_score(
    label: str,
    *,
    projection_hash: str,
    strict: float,
    rare: float,
) -> Phase5ProjectionGoldScore:
    return Phase5ProjectionGoldScore(
        projection_hash=projection_hash,
        gold_projection_hash=digest(f"gold-{label}"),
        gold_source_hash=digest(f"gold-source-{label}"),
        scorer_plan_hash=digest(f"scorer-plan-{label}"),
        alignment_plan_hash=digest(f"alignment-plan-{label}"),
        strict_qualified_assertion_f1=strict,
        rare_pivotal_recall=rare,
        strict_match_target_ids=(),
        rare_score_hash=digest(f"rare-score-{label}"),
        structurally_valid=True,
        content_bearing=True,
    )


def _write_complete_scoring_bundle(journal_root: Path, scoring_root: Path) -> None:
    inputs, manifest, index, execution_map = _verify_journal(journal_root)
    compiled_at = NOW + timedelta(minutes=10)
    records = []
    gold_artifacts = set()
    target_artifacts = set()
    for (episode_id, condition), execution_record in sorted(
        execution_map.items(), key=lambda item: (item[0][0], item[0][1].value)
    ):
        execution = execution_record.execution
        if execution.kind is not FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER:
            continue
        capability_limited = (
            execution.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
        )
        changes = () if execution.diff is None else tuple(
            item.change_id for item in execution.diff.changes
        )
        before_score = None
        after_score = None
        before_f1 = after_f1 = before_rare = after_rare = None
        if not capability_limited:
            before_f1, after_f1 = 0.5, 0.625
            before_rare, after_rare = 0.25, 0.5
            before_score = _gold_score(
                f"before-{episode_id}-{condition.value}",
                projection_hash=execution.before_bundle.projection_hash,
                strict=before_f1,
                rare=before_rare,
            )
            assert execution.after_bundle is not None
            after_score = _gold_score(
                f"after-{episode_id}-{condition.value}",
                projection_hash=execution.after_bundle.projection_hash,
                strict=after_f1,
                rare=after_rare,
            )
        score = score_scripted_feedback(
            episode_id=f"{episode_id}--{condition.value.lower()}",
            before_strict_f1=before_f1,
            after_strict_f1=after_f1,
            required_target_change_ids=("required",),
            realized_target_change_ids=(),
            changed_semantic_ids=changes,
            requested_scope_ids=(),
            unsupported_changed_ids=changes,
            before_rare_pivotal_recall=before_rare,
            after_rare_pivotal_recall=after_rare,
            capability_limited=capability_limited,
            latency_seconds=execution.latency_seconds,
            replay_hash_success=(
                execution.replay is not None and execution.replay.replay_hash_success
            ),
        )
        assert execution_record.scorer_binding is not None
        binding = execution_record.scorer_binding
        before_source = digest(f"before-source-{episode_id}")
        after_source = digest(f"after-source-{episode_id}")
        target_source = digest(f"target-source-{episode_id}")
        gold_artifacts.update((before_source, after_source))
        target_artifacts.add(target_source)
        records.append(
            Phase5ScriptedFeedbackMetricRecord(
                episode_id=episode_id,
                condition=condition,
                scorer_binding_key=binding.scorer_binding_key,
                execution_record_hash=execution_record.content_hash,
                execution_hash=execution.content_hash,
                instruction_hash=execution.instruction.content_hash,
                before_projection_hash=execution.before_bundle.projection_hash,
                after_projection_hash=(
                    None
                    if execution.after_bundle is None
                    else execution.after_bundle.projection_hash
                ),
                c2_attempt_status=(
                    None
                    if execution.regeneration_receipt is None
                    else execution.regeneration_receipt.attempt_status
                ),
                before_gold_projection_hash=binding.before_gold_projection_hash,
                after_gold_projection_hash=binding.after_gold_projection_hash,
                before_gold_source_artifact_hash=before_source,
                after_gold_source_artifact_hash=after_source,
                required_target_change_hash=binding.required_target_change_hash,
                required_target_source_artifact_hash=target_source,
                diff_hash=None if execution.diff is None else execution.diff.content_hash,
                before_gold_score=before_score,
                after_gold_score=after_score,
                before_strict_f1_endpoint=before_f1,
                after_strict_f1_endpoint=after_f1,
                before_rare_pivotal_endpoint=before_rare,
                after_rare_pivotal_endpoint=after_rare,
                score=score,
                compiled_at=compiled_at,
            )
        )
    metrics = Phase5ScriptedFeedbackMetrics(
        metrics_id="TEST-ONLY-phase5-scripted-feedback",
        protocol_hash=inputs.protocol_hash,
        source_manifest_hash=inputs.source_manifest_hash,
        known_answer_source_hash=digest("known-answer-source"),
        input_manifest_hash=inputs.content_hash,
        feedback_manifest_hash=manifest.content_hash,
        execution_index_hash=index.content_hash,
        scorer_binding_authorization_hash=inputs.scorer_binding_authorization_hash,
        records=tuple(records),
        generated_at=compiled_at,
    )
    session = Phase5FeedbackScoringSession(
        session_id="TEST-ONLY-phase5-scoring-session",
        source_manifest_hash=metrics.source_manifest_hash,
        known_answer_source_hash=metrics.known_answer_source_hash,
        input_manifest_hash=metrics.input_manifest_hash,
        execution_index_hash=metrics.execution_index_hash,
        compiler_source_file_sha256=digest("TEST-ONLY-scorer-source"),
        compiled_at=compiled_at,
    )
    metrics_bytes = _canonical_bytes(metrics)
    receipt = Phase5FeedbackScoringReceipt(
        receipt_id="TEST-ONLY-phase5-scoring-receipt",
        session_hash=session.content_hash,
        metrics_hash=metrics.content_hash,
        metrics_file_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
        source_manifest_hash=metrics.source_manifest_hash,
        known_answer_source_hash=metrics.known_answer_source_hash,
        input_manifest_hash=metrics.input_manifest_hash,
        feedback_manifest_hash=metrics.feedback_manifest_hash,
        execution_index_hash=metrics.execution_index_hash,
        scorer_binding_authorization_hash=metrics.scorer_binding_authorization_hash,
        execution_record_hashes=tuple(item.execution_record_hash for item in metrics.records),
        gold_source_artifact_hashes=tuple(sorted(gold_artifacts)),
        target_source_artifact_hashes=tuple(sorted(target_artifacts)),
        compiled_at=compiled_at,
    )
    scoring_root.mkdir(parents=True)
    (scoring_root / SESSION_FILE_NAME).write_bytes(_canonical_bytes(session))
    (scoring_root / METRICS_FILE_NAME).write_bytes(metrics_bytes)
    (scoring_root / RECEIPT_FILE_NAME).write_bytes(_canonical_bytes(receipt))


def _complete_sources(tmp_path: Path) -> tuple[Path, Path]:
    protocol, gate, inputs = _inputs()
    journal = tmp_path / "journal"
    execute_phase5(
        inputs=inputs,
        protocol=protocol,
        prerequisites=gate,
        adapter=_Adapter(),
        ledger_verifier=_Verifier(),
        output_root=journal,
        completed_at=NOW + timedelta(minutes=5),
    )
    scoring = tmp_path / "scoring"
    _write_complete_scoring_bundle(journal, scoring)
    return journal, scoring


def test_complete_phase5_inventory_materializes_and_replays_exact_table(
    tmp_path: Path,
) -> None:
    journal, scoring = _complete_sources(tmp_path)
    rows, table_bytes, receipt = prepare_phase5_feedback_table(
        feedback_journal_root=journal,
        scoring_root=scoring,
    )
    assert receipt.scripted_condition_score_count == 18
    assert len(receipt.scripted_metric_record_hashes) == 18
    assert receipt.researcher_trace_count == 3
    assert len(receipt.researcher_trace_execution_record_hashes) == 3
    assert receipt.researcher_trace_gold_fields == "NA"
    assert receipt.scripted_table_row_count == 60
    assert receipt.researcher_trace_table_row_count == 10
    assert receipt.table_row_count == 70 == len(rows)

    parsed = tuple(csv.DictReader(io.StringIO(table_bytes.decode("utf-8"))))
    assert tuple(parsed[0]) == FEEDBACK_TABLE_COLUMNS
    trace_rows = [item for item in parsed if item["episode_class"] == "researcher_trace"]
    scripted_rows = [
        item for item in parsed if item["episode_class"] == "scripted_known_answer"
    ]
    assert len(scripted_rows) == 60
    assert {int(item["episode_count"]) for item in scripted_rows} == {3}
    assert {
        int(item["defined_episode_count"])
        for item in scripted_rows
        if item["metric"] == "unsupported_change_count"
    } == {3}
    assert len(trace_rows) == 10
    assert {item["condition"] for item in trace_rows} == {ConditionName.C2_LLM_QUERY.value}
    assert {item["gold_applicability"] for item in trace_rows} == {"NA"}
    assert sum(
        int(item["episode_count"])
        for item in trace_rows
        if item["metric"] == "semantic_change_count"
    ) == 3

    output = tmp_path / "public-feedback"
    assert materialize_phase5_feedback_table(
        output_root=output,
        table_bytes=table_bytes,
        receipt=receipt,
    )
    assert not materialize_phase5_feedback_table(
        output_root=output,
        table_bytes=table_bytes,
        receipt=receipt,
    )
    replayed = verify_phase5_feedback_table(
        feedback_journal_root=journal,
        scoring_root=scoring,
        output_root=output,
    )
    assert replayed == receipt
    assert (output / FEEDBACK_TABLE_FILE_NAME).read_bytes() == table_bytes


def test_feedback_table_rejects_mutated_scripted_metric_source(tmp_path: Path) -> None:
    journal, scoring = _complete_sources(tmp_path)
    metrics_path = scoring / METRICS_FILE_NAME
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["records"][0]["score"]["latency_seconds"] += 1.0
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Phase5FeedbackReportError, match="invalid Phase5ScriptedFeedbackMetrics"):
        prepare_phase5_feedback_table(
            feedback_journal_root=journal,
            scoring_root=scoring,
        )


def test_feedback_table_replay_rejects_changed_public_csv(tmp_path: Path) -> None:
    journal, scoring = _complete_sources(tmp_path)
    _rows, table_bytes, receipt = prepare_phase5_feedback_table(
        feedback_journal_root=journal,
        scoring_root=scoring,
    )
    output = tmp_path / "public-feedback"
    materialize_phase5_feedback_table(
        output_root=output,
        table_bytes=table_bytes,
        receipt=receipt,
    )
    table_path = output / FEEDBACK_TABLE_FILE_NAME
    table_path.write_bytes(table_path.read_bytes() + b"\n")
    with pytest.raises(Phase5FeedbackReportError, match="table bytes do not replay"):
        verify_phase5_feedback_table(
            feedback_journal_root=journal,
            scoring_root=scoring,
            output_root=output,
        )


def test_feedback_table_rejects_allowed_name_symlink(tmp_path: Path) -> None:
    journal, scoring = _complete_sources(tmp_path)
    _rows, table_bytes, receipt = prepare_phase5_feedback_table(
        feedback_journal_root=journal,
        scoring_root=scoring,
    )
    output = tmp_path / "public-feedback"
    output.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("do not follow\n", encoding="utf-8")
    (output / FEEDBACK_TABLE_FILE_NAME).symlink_to(outside)
    with pytest.raises(Phase5FeedbackReportError, match="non-regular"):
        materialize_phase5_feedback_table(
            output_root=output,
            table_bytes=table_bytes,
            receipt=receipt,
        )
