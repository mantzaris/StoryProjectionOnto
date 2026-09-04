from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from story_projection_onto.case_study_execution import validate_case_admission_evidence
from story_projection_onto.case_study_factory import (
    CASE_PRODUCTION_FACTORY,
    CaseStudyFactoryError,
    create_frozen_production_case_study_bundle,
    stage_case_admission_evidence,
)
from story_projection_onto.case_study_runtime import CaseStudyAdmissionAttestation
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
)

T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
HASH = "a" * 64
GATE_NAMES = (
    "synthetic_run_closure_hash",
    "timing_lineage_audit_hash",
    "gold_firewall_audit_hash",
    "registered_metric_regeneration_hash",
    "blinded_error_review_hash",
    "storage_preflight_hash",
    "gpu_schedule_admission_hash",
    "public_release_scan_hash",
)


def _admission_and_gates(
    restricted_root: Path,
) -> tuple[Path, CaseStudyAdmissionAttestation, dict[str, Path]]:
    values: dict[str, object] = {}
    paths: dict[str, Path] = {}
    for name in GATE_NAMES:
        value = {"gate": name, "passed": True}
        values[name] = canonical_sha256(value)
        path = restricted_root / f"{name}.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[name] = path
    admission = CaseStudyAdmissionAttestation(
        admission_id="artificial-case-admission",
        **values,
        selected_model_freeze_file_sha256=HASH,
        selected_model_freeze_hash=HASH,
        upper_ontology_hash=HASH,
        validator_hash=HASH,
        c1_capability_manifest_hash=HASH,
        c2_capability_manifest_hash=HASH,
        attested_by="artificial fixture",
        attested_at=T0,
    )
    path = restricted_root / "admission.json"
    path.write_text(canonical_json(admission) + "\n", encoding="utf-8")
    return path, admission, paths


def test_stage_case_admission_evidence_is_restricted_and_replayable(tmp_path: Path) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, admission, gate_paths = _admission_and_gates(restricted)
    ledger_path = restricted / "study.sqlite"
    artifact_root = restricted / "blobs"
    ledger = Ledger(ledger_path)
    ledger.record_gpu_event(
        event_id="artificial-predecessor",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1.0,
        started_at=T0 - timedelta(seconds=1),
        ended_at=T0,
        succeeded=True,
        details={"artificial_fixture": True},
    )
    ledger.close()
    bundle_path = restricted / "staged" / "bundle.json"
    reference_path = restricted / "staged" / "reference.json"

    bundle, reference = stage_case_admission_evidence(
        restricted_root=restricted,
        admission_attestation_path=admission_path,
        gate_paths=gate_paths,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        bundle_output_path=bundle_path,
        reference_output_path=reference_path,
        staged_at=T0 + timedelta(minutes=1),
    )

    assert bundle.admission_attestation_hash == admission.content_hash
    assert len(bundle.evidence) == 8
    assert bundle_path.is_file()
    assert reference_path.is_file()
    assert reference.object_kind == "case_admission_evidence_bundle"
    assert CASE_PRODUCTION_FACTORY.endswith("create_frozen_production_case_study_bundle")
    ledger = Ledger(ledger_path)
    artifacts = ArtifactStore(
        BlobStore(artifact_root, compression=Compression.ZSTD),
        ledger,
    )
    validate_case_admission_evidence(
        admission=admission,
        bundle=bundle,
        artifacts=artifacts,
    )
    ledger.close()

    replayed, replayed_reference = stage_case_admission_evidence(
        restricted_root=restricted,
        admission_attestation_path=admission_path,
        gate_paths=gate_paths,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        bundle_output_path=bundle_path,
        reference_output_path=reference_path,
        staged_at=T0 + timedelta(minutes=1),
    )
    assert replayed == bundle
    assert replayed_reference == reference


def test_stage_case_admission_evidence_rejects_tampered_gate(tmp_path: Path) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, _admission, gate_paths = _admission_and_gates(restricted)
    gate_paths["gold_firewall_audit_hash"].write_text(
        '{"gate":"different","passed":true}\n',
        encoding="utf-8",
    )
    ledger_path = restricted / "study.sqlite"
    ledger = Ledger(ledger_path)
    ledger.record_gpu_event(
        event_id="artificial-predecessor",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1.0,
        started_at=T0 - timedelta(seconds=1),
        ended_at=T0,
        succeeded=True,
        details={"artificial_fixture": True},
    )
    ledger.close()

    with pytest.raises(CaseStudyFactoryError, match="differs from its attestation"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            admission_attestation_path=admission_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=restricted / "blobs",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )


def test_stage_case_admission_evidence_does_not_create_a_fresh_ledger(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, _admission, gate_paths = _admission_and_gates(restricted)
    ledger_path = restricted / "missing.sqlite"

    with pytest.raises(CaseStudyFactoryError, match="existing study ledger"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            admission_attestation_path=admission_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=restricted / "blobs",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )

    assert not ledger_path.exists()


def test_stage_requires_ledger_and_cas_realpaths_under_restricted_root(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, _admission, gate_paths = _admission_and_gates(restricted)
    outside_ledger = tmp_path / "outside.sqlite"
    ledger = Ledger(outside_ledger)
    ledger.record_gpu_event(
        event_id="artificial-predecessor",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1.0,
        started_at=T0 - timedelta(seconds=1),
        ended_at=T0,
        succeeded=True,
    )
    ledger.close()
    with pytest.raises(CaseStudyFactoryError, match="inside the restricted root"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            admission_attestation_path=admission_path,
            gate_paths=gate_paths,
            ledger_path=outside_ledger,
            artifact_root=restricted / "blobs",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )

    inside_ledger = restricted / "study.sqlite"
    inside_ledger.write_bytes(outside_ledger.read_bytes())
    outside_blobs = tmp_path / "outside-blobs"
    outside_blobs.mkdir()
    (restricted / "linked").symlink_to(outside_blobs, target_is_directory=True)
    with pytest.raises(CaseStudyFactoryError, match="symbolic link"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            admission_attestation_path=admission_path,
            gate_paths=gate_paths,
            ledger_path=inside_ledger,
            artifact_root=restricted / "linked" / "blobs",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )


def test_full_production_factory_rejects_external_ledger_before_model_construction(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    placeholder = restricted / "placeholder.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    outside_ledger = tmp_path / "outside.sqlite"
    outside_ledger.write_bytes(b"not opened")

    with pytest.raises(CaseStudyFactoryError, match="inside the restricted root"):
        create_frozen_production_case_study_bundle(
            repository=Path(__file__).resolve().parents[2],
            restricted_root=restricted,
            plan_path=placeholder,
            index_path=placeholder,
            index_manifest_path=placeholder,
            preregistration_path=placeholder,
            input_attestation_path=placeholder,
            admission_attestation_path=placeholder,
            selected_model_freeze_path=placeholder,
            admission_evidence_bundle_path=placeholder,
            admission_evidence_bundle_reference_path=placeholder,
            construction_path=placeholder,
            ledger_path=outside_ledger,
            artifact_root=restricted / "blobs",
            runtime_root=restricted / "runtime",
            quota_root=tmp_path,
            snapshot_path=tmp_path,
            shared_cache=tmp_path,
            verified_model_manifest_path=placeholder,
            source_association_path=placeholder,
            expected_predecessor_ledger_sha256="a" * 64,
            source_revision="artificial-source",
        )
