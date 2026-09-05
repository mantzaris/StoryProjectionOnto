from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.report_ingestion import (
    IngestedTableSpec,
    ReportingIngestionError,
    compile_ingestion_receipt,
    load_ingestion_manifest,
    verify_ingestion_from_files,
)
from story_projection_onto.reporting import ReportStatus, canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs/study/report_ingestion.json"
RECEIPT_PATH = ROOT / "reports/report_ingestion_receipt.json"


def _copy_registered_inputs(tmp_path: Path):
    manifest = load_ingestion_manifest(MANIFEST_PATH)
    source_root = tmp_path / "source"
    table_root = tmp_path / "report"
    for predecessor in manifest.predecessors:
        for artifact in predecessor.artifacts:
            target = source_root / artifact.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / artifact.relative_path, target)
    for table in manifest.tables:
        if table.status is ReportStatus.COMPLETE:
            assert table.relative_path is not None
            target = table_root / table.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "reports" / table.relative_path, target)
    return manifest, source_root, table_root


def _compile(manifest, source_root: Path, table_root: Path):
    return compile_ingestion_receipt(
        manifest,
        source_root=source_root,
        table_root=table_root,
        manifest_file_sha256=hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
    )


def test_checked_in_ingestion_replays_all_families_and_table_slots() -> None:
    receipt = verify_ingestion_from_files(
        MANIFEST_PATH,
        RECEIPT_PATH,
        source_root=ROOT,
        table_root=ROOT / "reports",
    )
    assert len(receipt.predecessors) == 8
    assert len(receipt.tables) == 14
    assert sum(item.status is ReportStatus.COMPLETE for item in receipt.tables) == 3
    assert sum(item.status is not ReportStatus.COMPLETE for item in receipt.tables) == 11
    assert receipt.gpu_service_time_artifact_ids == (
        "primary-pilot-v1-failure",
        "primary-pilot-v2-failure",
        "fallback-pilot-v1-failure",
        "fallback-pilot-v3-failure",
    )
    assert receipt.runpod_wall_time_artifact_ids == ()


def test_predecessor_byte_tamper_is_blocking(tmp_path: Path) -> None:
    manifest, source_root, table_root = _copy_registered_inputs(tmp_path)
    artifact = manifest.predecessors[0].artifacts[0]
    path = source_root / artifact.relative_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ReportingIngestionError, match="predecessor file hash changed"):
        _compile(manifest, source_root, table_root)


def test_symlinked_predecessor_is_blocking(tmp_path: Path) -> None:
    manifest, source_root, table_root = _copy_registered_inputs(tmp_path)
    artifact = manifest.predecessors[0].artifacts[0]
    path = source_root / artifact.relative_path
    path.unlink()
    path.symlink_to(ROOT / artifact.relative_path)
    with pytest.raises(ReportingIngestionError, match="symlinked reporting input"):
        _compile(manifest, source_root, table_root)


def test_final_table_refuses_incomplete_predecessor(tmp_path: Path) -> None:
    manifest, source_root, table_root = _copy_registered_inputs(tmp_path)
    table = next(item for item in manifest.tables if item.table_id == "primary_c2_vs_c1")
    status = next(item for item in manifest.tables if item.table_id == "study_status")
    table = table.model_copy(
        update={
            "status": ReportStatus.COMPLETE,
            "relative_path": status.relative_path,
            "file_sha256": status.file_sha256,
            "row_count": status.row_count,
            "required_columns": status.required_columns,
            "source_artifact_ids": ("synthetic-benchmark-manifest",),
        }
    )
    manifest = manifest.model_copy(
        update={
            "tables": tuple(
                table if item.table_id == table.table_id else item for item in manifest.tables
            )
        }
    )
    with pytest.raises(ReportingIngestionError, match="incomplete predecessors: heldout"):
        _compile(manifest, source_root, table_root)


def test_final_table_requires_lineage_from_each_dependency_family(tmp_path: Path) -> None:
    manifest, source_root, table_root = _copy_registered_inputs(tmp_path)
    predecessors = tuple(
        item.model_copy(update={"status": ReportStatus.COMPLETE})
        if item.family.value == "heldout"
        else item
        for item in manifest.predecessors
    )
    table = next(item for item in manifest.tables if item.table_id == "primary_c2_vs_c1")
    status = next(item for item in manifest.tables if item.table_id == "study_status")
    table = table.model_copy(
        update={
            "status": ReportStatus.COMPLETE,
            "relative_path": status.relative_path,
            "file_sha256": status.file_sha256,
            "row_count": status.row_count,
            "required_columns": status.required_columns,
            "source_artifact_ids": ("primary-pilot-v1-failure",),
        }
    )
    manifest = manifest.model_copy(
        update={
            "predecessors": predecessors,
            "tables": tuple(
                table if item.table_id == table.table_id else item for item in manifest.tables
            ),
        }
    )
    with pytest.raises(ReportingIngestionError, match="lacks source lineage for: heldout"):
        _compile(manifest, source_root, table_root)


def test_observed_runpod_wall_time_needs_distinct_source(tmp_path: Path) -> None:
    manifest, source_root, table_root = _copy_registered_inputs(tmp_path)
    table = next(item for item in manifest.tables if item.table_id == "resource_accounting")
    assert table.relative_path is not None
    table_path = table_root / table.relative_path
    text = table_path.read_text(encoding="utf-8")
    text = text.replace(
        "study,final_total_runpod_wall_time,,hours,incomplete,",
        "study,final_total_runpod_wall_time,1.5,hours,observed,",
    )
    table_path.write_text(text, encoding="utf-8", newline="\n")
    table = table.model_copy(
        update={"file_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest()}
    )
    manifest = manifest.model_copy(
        update={
            "tables": tuple(
                table if item.table_id == table.table_id else item for item in manifest.tables
            )
        }
    )
    with pytest.raises(ReportingIngestionError, match="distinct wall-time artifact"):
        _compile(manifest, source_root, table_root)


def test_incomplete_table_cannot_name_a_csv() -> None:
    with pytest.raises(ValidationError, match="incomplete table cannot point"):
        IngestedTableSpec(
            table_id="primary_c2_vs_c1",
            status=ReportStatus.INCOMPLETE,
            reason="TEST-ONLY absent result.",
            scope="final",
            relative_path="tables/fake.csv",
        )


def test_receipt_with_recomputed_but_changed_content_does_not_replay(
    tmp_path: Path,
) -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    receipt["predecessors"][0]["reason"] = "tampered but self-hashed"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ReportingIngestionError, match="does not reproduce"):
        verify_ingestion_from_files(
            MANIFEST_PATH,
            receipt_path,
            source_root=ROOT,
            table_root=ROOT / "reports",
        )
