from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from story_projection_onto.report_visual_inspection import (
    VisualCriterion,
    VisualInspectionReceipt,
    _ordered_poppler_rasters,
    load_raster_manifest,
    prepare_report_rasters,
    record_visual_inspection,
    verify_visual_inspection,
)
from story_projection_onto.reporting import (
    REGISTERED_COMPLETE_TABLE_IDS,
    LoadedTable,
    QualitativeSelectionRule,
    ReportingError,
    SupplementalMetricCount,
    SupplementalMetricSourceSpec,
    TableSpec,
    _markdown_supplemental_metric_sources,
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


def _supplemental_source() -> SupplementalMetricSourceSpec:
    return SupplementalMetricSourceSpec(
        source_role="primary_unit_temporal_epistemic",
        source_relative_path="artifacts/public/phase4/tables/unit_metrics.csv",
        source_file_sha256="1" * 64,
        table_manifest_relative_path="artifacts/public/phase4/table_manifest.json",
        table_manifest_file_sha256="2" * 64,
        source_row_count=10_000,
        selected_row_count=9_000,
        metric_row_counts=(
            SupplementalMetricCount(metric_name="registered_metric", row_count=252),
        ),
        description="TEST-ONLY complete numeric diagnostic rows.",
    )


def test_complete_registered_panel_requires_and_reports_supplemental_numeric_source() -> None:
    common = {
        "table_id": "primary_c2_vs_c1",
        "relative_path": "tables/primary.csv",
        "sha256": "3" * 64,
        "row_count": 2,
        "required_columns": ("metric", "estimate"),
        "display_columns": ("metric", "estimate"),
        "status": "complete",
        "description": "TEST-ONLY compact primary table.",
        "source_artifact_hashes": ("1" * 64, "2" * 64),
    }
    with pytest.raises(ValueError, match="supplemental metric sources"):
        TableSpec(**common)
    spec = TableSpec(**common, supplemental_metric_sources=(_supplemental_source(),))
    rendered = _markdown_supplemental_metric_sources(
        LoadedTable(
            spec=spec,
            columns=("metric", "estimate"),
            rows=({"metric": "strict_f1", "estimate": "0.1"},),
        )
    )
    assert "registered_metric" in rendered
    assert "(252)" in rendered
    assert "10000 total rows" in rendered
    assert "9000 rows after" in rendered
    assert "unit_metrics.csv" in rendered
    assert "does not recompute" in rendered


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _visual_fixture(tmp_path: Path) -> tuple[Path, Path, datetime]:
    visual_root = tmp_path / "visual"
    page = visual_root / "fixture-token" / "pages" / "page-0001.png"
    page.parent.mkdir(parents=True)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (200).to_bytes(4, "big")
        + (300).to_bytes(4, "big")
        + b"\x00" * 1000
    )
    page.write_bytes(png)
    pdf = tmp_path / "RESULTS_REPORT.pdf"
    pdf.write_bytes(b"%PDF-1.4\nTEST-ONLY\n")
    prepared_at = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    checks = [
        {
            "check_id": check_id,
            "status": "passed",
            "detail": "TEST-ONLY structural check.",
        }
        for check_id in (
            "pdf_header",
            "page_count",
            "section_text_inventory",
            "figure_text_inventory",
            "table_text_inventory",
            "raster_dimensions",
        )
    ]
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "report_raster_manifest",
        "source_result_manifest_sha256": "1" * 64,
        "source_pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "source_pdf_name": "RESULTS_REPORT.pdf",
        "renderer": "pdftoppm",
        "renderer_version": "TEST-ONLY pdftoppm",
        "raster_prepared_at_utc": prepared_at.isoformat().replace("+00:00", "Z"),
        "dpi": 144,
        "page_count": 1,
        "representative_pages": [1],
        "pages": [
            {
                "page_number": 1,
                "relative_path": "fixture-token/pages/page-0001.png",
                "file_sha256": hashlib.sha256(png).hexdigest(),
                "width_pixels": 200,
                "height_pixels": 300,
                "size_bytes": len(png),
            }
        ],
        "automated_checks": checks,
        "visual_review_status": "pending",
        "visual_review_notice": (
            "Automated structural checks are not a visual-quality acceptance."
        ),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    manifest = visual_root / "report_raster_manifest.fixture.json"
    _write_json(manifest, payload)
    return manifest, pdf, prepared_at


def _passing_visual_criteria() -> tuple[VisualCriterion, ...]:
    return tuple(
        VisualCriterion(
            criterion=criterion,
            status="passed",
            note="TEST-ONLY explicit visual assessment.",
        )
        for criterion in (
            "clipping",
            "figure_presence",
            "graph_label_readability",
            "mathematical_notation",
            "page_breaks",
            "references_and_provenance",
        )
    )


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


def test_blinded_review_supplement_is_hash_bound_and_rendered_from_csv(
    tmp_path: Path,
) -> None:
    manifest = _incomplete_manifest(tmp_path)
    supplement = (
        b"error_code,failure_count,world_count,reviewed_failure_count,independent_unit\n"
        b"invalid_structure,2,1,3,world\n"
    )
    supplement_path = tmp_path / "tables/held_out_error_review_summary.test.csv"
    supplement_path.write_bytes(supplement)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["review_supplements"] = [
        {
            "supplement_id": "held_out_error_review",
            "section_id": "error_limitations",
            "relative_path": "tables/held_out_error_review_summary.test.csv",
            "sha256": hashlib.sha256(supplement).hexdigest(),
            "row_count": 1,
            "required_columns": [
                "error_code",
                "failure_count",
                "world_count",
                "reviewed_failure_count",
                "independent_unit",
            ],
            "display_columns": [
                "error_code",
                "failure_count",
                "world_count",
                "reviewed_failure_count",
            ],
            "description": "TEST-ONLY mechanically derived blinded error counts.",
            "source_artifact_hashes": [SOURCE_HASH],
        }
    ]
    payload["sections"][0]["section_id"] = "error_limitations"
    payload["sections"][0]["title"] = "Error analysis and limitations"
    payload.pop("manifest_sha256")
    _write_json(manifest, canonical_manifest_payload(payload))
    rendered = render_markdown(build_document(manifest, POLICY))
    assert "invalid_structure" in rendered
    assert "| 2 | 1 | 3 |" in rendered

    supplement_path.write_bytes(supplement.replace(b",2,1,3,", b",0,0,3,"))
    with pytest.raises(ReportingError, match="review supplement hash mismatch"):
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


def test_generated_figure_is_embedded_and_manifested_inside_report(tmp_path: Path) -> None:
    pytest.importorskip("reportlab")
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext is unavailable")
    manifest = _incomplete_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sections"][0]["section_id"] = "resource_controls"
    payload["sections"][0]["title"] = "Resource and reproducibility controls"
    payload.pop("manifest_sha256")
    _write_json(manifest, canonical_manifest_payload(payload))
    build_results_report(manifest, POLICY)
    extracted = subprocess.run(
        ["pdftotext", str(tmp_path / "RESULTS_REPORT.pdf"), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Registered study phase status" in extracted
    figure_manifest = json.loads(
        (tmp_path / "result_figure_manifest.json").read_text(encoding="utf-8")
    )
    figure = figure_manifest["figures"][0]
    assert figure["embedded_in_report"] is True
    assert figure["report_section_id"] == "resource_controls"
    assert figure["kind"] == "phase_status"


def test_raster_preflight_remains_pending_until_explicit_visual_attestation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("reportlab")
    if any(shutil.which(item) is None for item in ("pdfinfo", "pdftoppm", "pdftotext")):
        pytest.skip("Poppler inspection tools are unavailable")
    manifest = _incomplete_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sections"][0]["section_id"] = "resource_controls"
    payload["sections"][0]["title"] = "Resource and reproducibility controls"
    payload.pop("manifest_sha256")
    _write_json(manifest, canonical_manifest_payload(payload))
    build_results_report(manifest, POLICY)
    raster_manifest_path = prepare_report_rasters(
        result_manifest_path=manifest,
        policy_path=POLICY,
        pdf_path=tmp_path / "RESULTS_REPORT.pdf",
        output_root=tmp_path / "visual",
    )
    raster = load_raster_manifest(raster_manifest_path)
    assert raster.visual_review_status == "pending"
    with pytest.raises(ReportingError, match="accepted human/agent"):
        verify_visual_inspection(
            raster_manifest_path=raster_manifest_path,
            pdf_path=tmp_path / "RESULTS_REPORT.pdf",
            require_accepted=True,
        )

    criteria = tuple(
        VisualCriterion(
            criterion=criterion,
            status="passed",
            note="TEST-ONLY explicit visual assessment after fixture-page viewing.",
        )
        for criterion in (
            "clipping",
            "figure_presence",
            "graph_label_readability",
            "mathematical_notation",
            "page_breaks",
            "references_and_provenance",
        )
    )
    with pytest.raises(ReportingError, match="representative page"):
        record_visual_inspection(
            raster_manifest_path=raster_manifest_path,
            output_path=raster_manifest_path.parent / "invalid-receipt.json",
            reviewer_kind="agent",
            reviewed_at_utc=datetime.now(UTC),
            inspected_pages=(),
            criteria=criteria,
        )
    receipt_path = raster_manifest_path.parent / "receipt.json"
    receipt = record_visual_inspection(
        raster_manifest_path=raster_manifest_path,
        output_path=receipt_path,
        reviewer_kind="agent",
        reviewed_at_utc=datetime.now(UTC),
        inspected_pages=raster.representative_pages,
        criteria=criteria,
    )
    assert receipt.status == "accepted"
    assert receipt.reviewed_at_utc > raster.raster_prepared_at_utc
    verify_visual_inspection(
        raster_manifest_path=raster_manifest_path,
        pdf_path=tmp_path / "RESULTS_REPORT.pdf",
        receipt_path=receipt_path,
        require_accepted=True,
    )


def test_poppler_rasters_are_bound_in_numeric_page_order(tmp_path: Path) -> None:
    paths = tuple(tmp_path / name for name in ("page-10.png", "page-2.png", "page-1.png"))
    with pytest.raises(ReportingError, match="not contiguous"):
        _ordered_poppler_rasters(paths)
    contiguous = tuple(tmp_path / f"page-{number}.png" for number in range(12, 0, -1))
    assert [item.name for item in _ordered_poppler_rasters(contiguous)] == [
        f"page-{number}.png" for number in range(1, 13)
    ]


def test_visual_receipt_timestamp_is_normalized_to_utc() -> None:
    with pytest.raises(ValueError, match="normalized to UTC"):
        VisualInspectionReceipt(
            source_raster_manifest_sha256="1" * 64,
            source_raster_manifest_file_sha256="2" * 64,
            source_pdf_sha256="3" * 64,
            reviewer_kind="agent",
            reviewed_at_utc=datetime.now(timezone(timedelta(hours=-4))),
            inspected_pages=(1,),
            criteria=tuple(
                VisualCriterion(
                    criterion=criterion,
                    status="passed",
                    note="TEST-ONLY criterion.",
                )
                for criterion in (
                    "clipping",
                    "figure_presence",
                    "graph_label_readability",
                    "mathematical_notation",
                    "page_breaks",
                    "references_and_provenance",
                )
            ),
            status="accepted",
            receipt_sha256="4" * 64,
        )


def test_visual_review_must_strictly_postdate_raster_preparation(tmp_path: Path) -> None:
    manifest, _, prepared_at = _visual_fixture(tmp_path)
    with pytest.raises(ReportingError, match="after raster preparation"):
        record_visual_inspection(
            raster_manifest_path=manifest,
            output_path=manifest.parent / "receipt.json",
            reviewer_kind="agent",
            reviewed_at_utc=prepared_at,
            inspected_pages=(1,),
            criteria=_passing_visual_criteria(),
        )


def test_visual_paths_reject_escape_and_symlinked_ancestry(tmp_path: Path) -> None:
    manifest, pdf, prepared_at = _visual_fixture(tmp_path)
    with pytest.raises(ReportingError, match="inside the visual-inspection root"):
        record_visual_inspection(
            raster_manifest_path=manifest,
            output_path=tmp_path / "outside-receipt.json",
            reviewer_kind="agent",
            reviewed_at_utc=prepared_at + timedelta(seconds=1),
            inspected_pages=(1,),
            criteria=_passing_visual_criteria(),
        )

    real_root = manifest.parent
    linked_root = tmp_path / "linked-visual"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ReportingError, match="symlinked ancestor"):
        verify_visual_inspection(
            raster_manifest_path=linked_root / manifest.name,
            pdf_path=pdf,
        )

    pages = real_root / "fixture-token" / "pages"
    real_pages = tmp_path / "real-pages"
    pages.rename(real_pages)
    pages.symlink_to(real_pages, target_is_directory=True)
    with pytest.raises(ReportingError, match="symlinked ancestor"):
        verify_visual_inspection(
            raster_manifest_path=manifest,
            pdf_path=pdf,
        )


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
