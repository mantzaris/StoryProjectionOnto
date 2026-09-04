from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from story_projection_onto.reporting import (
    REGISTERED_COMPLETE_TABLE_IDS,
    QualitativeSelectionRule,
    ReportingError,
    build_document,
    build_results_report,
    canonical_manifest_payload,
    canonical_sha256,
    load_reporting_policy,
    render_markdown,
    verify_report_build,
)

POLICY = Path("configs/study/reporting.json")
ZERO_REVISION = "0" * 40
SOURCE_HASH = "1" * 64


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _incomplete_manifest(tmp_path: Path) -> Path:
    phase_rows = [(f"phase_{number}", "incomplete", "Not executed.") for number in range(1, 8)]
    table = (
        "phase_id,status,reason\n"
        + "".join(f"{phase_id},{status},{reason}\n" for phase_id, status, reason in phase_rows)
    ).encode()
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "study_status.csv").write_bytes(table)
    receipt_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "source_manifest_sha256": "2" * 64,
        "source_manifest_file_sha256": "3" * 64,
        "predecessors": [
            {
                "family": family,
                "status": "incomplete",
                "reason": "TEST-ONLY predecessor is incomplete.",
                "artifact_ids": ["test-source"] if family == "runtime" else [],
            }
            for family in (
                "fallback",
                "development",
                "heldout",
                "ablations",
                "feedback",
                "case_study",
                "runtime",
                "storage",
            )
        ],
        "artifacts": [
            {
                "artifact_id": "test-source",
                "family": "runtime",
                "file_sha256": SOURCE_HASH,
                "logical_hash": None,
                "media_type": "application/json",
                "measurement_domains": [],
            }
        ],
        "tables": [
            (
                {
                    "table_id": table_id,
                    "status": "complete",
                    "reason": "TEST-ONLY current status table.",
                    "scope": "interim",
                    "relative_path": "tables/study_status.csv",
                    "file_sha256": hashlib.sha256(table).hexdigest(),
                    "row_count": 7,
                    "columns": ["phase_id", "status", "reason"],
                    "source_artifact_hashes": [],
                }
                if table_id == "study_status"
                else {
                    "table_id": table_id,
                    "status": "incomplete",
                    "reason": "TEST-ONLY table is incomplete.",
                    "scope": "final",
                    "relative_path": None,
                    "file_sha256": None,
                    "row_count": None,
                    "columns": [],
                    "source_artifact_hashes": [],
                }
            )
            for table_id in sorted(REGISTERED_COMPLETE_TABLE_IDS)
        ],
        "compiled_at_utc": "2026-09-04T00:00:00Z",
        "gpu_service_time_artifact_ids": [],
        "runpod_wall_time_artifact_ids": [],
    }
    receipt_payload["receipt_sha256"] = canonical_sha256(receipt_payload)
    receipt = tmp_path / "report_ingestion_receipt.json"
    _write_json(receipt, receipt_payload)
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "study_id": "test-study",
        "study_status": "incomplete",
        "status_reason": "Registered execution has not completed.",
        "generated_at_utc": "2026-09-04T00:00:00Z",
        "code_revision": ZERO_REVISION,
        "dirty_worktree": True,
        "model_repository": None,
        "model_revision": None,
        "phases": [
            {
                "phase_id": f"phase_{number}",
                "label": f"Phase {number}",
                "status": "incomplete",
                "reason": "Not executed.",
                "source_artifact_hashes": [],
            }
            for number in range(1, 8)
        ],
        "tables": [
            {
                "table_id": "study_status",
                "relative_path": "tables/study_status.csv",
                "sha256": hashlib.sha256(table).hexdigest(),
                "row_count": 7,
                "required_columns": ["phase_id", "status", "reason"],
                "status": "complete",
                "description": "Phase completion status",
                "source_artifact_hashes": [],
            }
        ],
        "figures": [
            {
                "figure_id": "phase_status",
                "kind": "phase_status",
                "title": "Registered study phase status",
                "relative_path": "figures/study_status.pdf",
            }
        ],
        "sections": [
            {
                "section_id": "primary_results",
                "title": "Primary C2 versus C1 results",
                "required_phases": ["phase_3", "phase_4"],
                "table_ids": ["study_status"],
                "fixed_text": [
                    "Aggregate statistical comparisons, when available, are the efficacy evidence."
                ],
            }
        ],
        "source_artifact_hashes": [SOURCE_HASH],
        "ingestion_receipt_relative_path": "report_ingestion_receipt.json",
        "ingestion_receipt_file_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "ingestion_receipt_sha256": receipt_payload["receipt_sha256"],
        "reporting_policy_sha256": json.loads(POLICY.read_text())["policy_sha256"],
    }
    path = tmp_path / "results_manifest.json"
    _write_json(path, canonical_manifest_payload(payload))
    return path


def test_incomplete_report_cannot_imply_null_or_completed_results(tmp_path: Path) -> None:
    manifest = _incomplete_manifest(tmp_path)
    document = build_document(manifest, POLICY)
    rendered = render_markdown(document)
    assert "**STATUS: INCOMPLETE**" in rendered
    assert "No confirmatory or descriptive outcome is available" in rendered
    assert "not evidence of a null result" in rendered
    assert "GPU fallback not yet accepted" not in rendered


def test_complete_label_requires_every_registered_phase_and_output(tmp_path: Path) -> None:
    manifest = _incomplete_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["study_status"] = "complete"
    payload["dirty_worktree"] = False
    payload["model_repository"] = "Example/model"
    payload["model_revision"] = "2" * 40
    payload["source_artifact_hashes"] = [SOURCE_HASH]
    payload.pop("manifest_sha256")
    _write_json(manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="missing/incomplete phases"):
        build_document(manifest, POLICY)


def test_table_hash_and_row_count_are_blocking(tmp_path: Path) -> None:
    manifest = _incomplete_manifest(tmp_path)
    table_path = tmp_path / "tables" / "study_status.csv"
    table_path.write_text("phase_id,status,reason\nphase_1,complete,tampered\n")
    with pytest.raises(ReportingError, match="table hash mismatch"):
        build_document(manifest, POLICY)


def test_symlinked_result_manifest_is_blocking(tmp_path: Path) -> None:
    manifest = _incomplete_manifest(tmp_path)
    link = tmp_path / "linked-results.json"
    link.symlink_to(manifest)
    with pytest.raises(ReportingError, match="symlink is prohibited"):
        build_document(link, POLICY)


def test_markdown_values_regenerate_exactly_and_pdf_is_bound(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    manifest = _incomplete_manifest(tmp_path)
    build_results_report(manifest, POLICY)
    verify_report_build(manifest, POLICY)
    report = tmp_path / "RESULTS_REPORT.md"
    report.write_text(report.read_text() + "manually transcribed p = .001\n")
    with pytest.raises(ReportingError, match="manually transcribed"):
        verify_report_build(manifest, POLICY)


def test_pdf_must_reproduce_even_if_tampered_hash_manifest_is_rewritten(
    tmp_path: Path,
) -> None:
    pytest.importorskip("reportlab")
    manifest = _incomplete_manifest(tmp_path)
    build_results_report(manifest, POLICY)
    pdf = tmp_path / "RESULTS_REPORT.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"\n% hash-consistent tamper\n")
    build_manifest = tmp_path / "result_figure_manifest.json"
    payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    for record in payload["reports"]:
        if record["relative_path"] == "RESULTS_REPORT.pdf":
            record["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
    _write_json(build_manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="does not reproduce"):
        verify_report_build(manifest, POLICY)


def test_result_figure_manifest_requires_exact_table_inventory(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    manifest = _incomplete_manifest(tmp_path)
    build_results_report(manifest, POLICY)
    build_manifest = tmp_path / "result_figure_manifest.json"
    payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["tables"] = []
    _write_json(build_manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="table inventory"):
        verify_report_build(manifest, POLICY)


def test_policy_hash_is_a_blocking_input(tmp_path: Path) -> None:
    manifest = _incomplete_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["reporting_policy_sha256"] = "3" * 64
    payload.pop("manifest_sha256")
    _write_json(manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="not bound"):
        build_document(manifest, POLICY)


def test_phase_cannot_cite_source_outside_verified_ingestion(tmp_path: Path) -> None:
    manifest = _incomplete_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["phases"][0]["source_artifact_hashes"] = ["9" * 64]
    payload.pop("manifest_sha256")
    _write_json(manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="unverified predecessor"):
        build_document(manifest, POLICY)


def test_complete_section_inventory_is_bound_before_outcomes(tmp_path: Path) -> None:
    source = Path("reports/results_manifest.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sections"][0]["fixed_text"][0] += " Post-output prose drift."
    payload.pop("manifest_sha256")
    manifest = tmp_path / "results_manifest.json"
    _write_json(manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="prose differs"):
        build_document(manifest, POLICY)


def test_result_figure_manifest_binds_ingestion_receipt(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    manifest = _incomplete_manifest(tmp_path)
    build_results_report(manifest, POLICY)
    build_manifest = tmp_path / "result_figure_manifest.json"
    payload = json.loads(build_manifest.read_text(encoding="utf-8"))
    payload["source_ingestion_receipt_sha256"] = "8" * 64
    payload.pop("manifest_sha256")
    _write_json(build_manifest, canonical_manifest_payload(payload))
    with pytest.raises(ReportingError, match="ingestion lineage"):
        verify_report_build(manifest, POLICY)


def test_counterexample_rule_is_frozen_before_outputs_but_applied_after_itt() -> None:
    policy = load_reporting_policy(POLICY)
    counterexample = next(
        item for item in policy.qualitative_selection_rules if item.example_id == "counterexample"
    )
    assert counterexample.selection_time == "rule_frozen_before_outputs_applied_after_itt_scoring"
    payload = counterexample.model_dump(mode="python")
    payload["selection_time"] = "before_condition_outputs"
    with pytest.raises(ValueError, match="ITT counterexample"):
        QualitativeSelectionRule.model_validate(payload)
