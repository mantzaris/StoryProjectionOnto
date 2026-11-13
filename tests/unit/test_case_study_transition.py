from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import story_projection_onto.case_study_transition as transition_module
from story_projection_onto.case_study_execution import (
    CaseAdmissionEvidenceBundle,
    CaseAdmissionEvidenceReference,
    CaseArtifactReference,
)
from story_projection_onto.case_study_transition import (
    CasePostH1ArtifactDescriptor,
    CaseStagedPayloadDescriptor,
    CaseStudyTransitionError,
    capture_case_admission_staging_intent,
    finalize_case_admission_staging_transition,
    verify_case_admission_staging_transition,
    verify_case_post_h1_artifact_prefix,
)
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuAllocationJournalState,
    GpuEventKind,
    Ledger,
    ReleaseClass,
)

T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(seconds=1)
PLAN_HASH = "1" * 64
ADMISSION_HASH = "2" * 64
SEMANTIC_HASH = "3" * 64
GATE_MEDIA_TYPE = "application/vnd.story-projection.case-admission-gate+json"
GATE_ROLES = (
    "blinded_error_review_hash",
    "gold_firewall_audit_hash",
    "gpu_schedule_admission_hash",
    "public_release_scan_hash",
    "registered_metric_regeneration_hash",
    "storage_preflight_hash",
    "synthetic_run_closure_hash",
    "timing_lineage_audit_hash",
)


@dataclass(frozen=True)
class _TransitionFixture:
    restricted_root: Path
    ledger_path: Path
    cas_root: Path
    transition_directory: Path
    payloads: dict[str, bytes]
    descriptors: tuple[CaseStagedPayloadDescriptor, ...]
    bundle_hash: str
    bundle_artifact_hash: str
    bundle_reference_hash: str


def _make_fixture(tmp_path: Path) -> _TransitionFixture:
    root = tmp_path / "restricted"
    root.mkdir()
    ledger_path = root / "study.sqlite3"
    cas_root = tmp_path / "artifacts" / "blobs" / "study"
    transition_directory = root / "case-transition"
    with Ledger(ledger_path) as ledger:
        artifacts = ArtifactStore(
            BlobStore(cas_root, compression=Compression.GZIP),
            ledger,
        )
        artifacts.put_bytes(
            b'{"preexisting":true}',
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=T0,
        )
        ledger.record_gpu_allocation_observation(
            allocation_id="pre-case-allocation",
            state=GpuAllocationJournalState.OPENED,
            intended_event_kind=GpuEventKind.SCHEMA_PROBE,
            elapsed_seconds=0,
            maximum_seconds=2,
            observed_at=T0,
        )
        ledger.record_gpu_event(
            event_id="pre-case-allocation",
            event_kind=GpuEventKind.SCHEMA_PROBE,
            allocated_seconds=1,
            started_at=T0,
            ended_at=T1,
            succeeded=True,
            details={"fixture": "pre-case"},
        )
        ledger.record_gpu_allocation_observation(
            allocation_id="pre-case-allocation",
            state=GpuAllocationJournalState.CLOSED,
            intended_event_kind=GpuEventKind.SCHEMA_PROBE,
            elapsed_seconds=1,
            maximum_seconds=2,
            observed_at=T1,
        )

    payloads: dict[str, bytes] = {}
    descriptors: list[CaseStagedPayloadDescriptor] = []
    for role in GATE_ROLES:
        value = {"gate": role, "status": "verified"}
        raw = canonical_json(value).encode("utf-8") + b"\n"
        payloads[role] = raw
        descriptors.append(
            CaseStagedPayloadDescriptor(
                role=role,
                logical_content_hash=canonical_sha256(value),
                artifact_hash=hashlib.sha256(raw).hexdigest(),
                raw_size_bytes=len(raw),
                compression="gzip",
                media_type=GATE_MEDIA_TYPE,
            )
        )
    bundle = CaseAdmissionEvidenceBundle(
        bundle_id="case-admission-evidence",
        admission_attestation_hash=ADMISSION_HASH,
        evidence=tuple(
            CaseAdmissionEvidenceReference(
                name=descriptor.role,
                logical_content_hash=descriptor.logical_content_hash,
                artifact_hash=descriptor.artifact_hash,
            )
            for descriptor in descriptors
        ),
        frozen_at=T1,
    )
    bundle_hash = bundle.content_hash
    bundle_raw = canonical_json(bundle).encode("utf-8")
    bundle_artifact_hash = hashlib.sha256(bundle_raw).hexdigest()
    bundle_reference_hash = CaseArtifactReference(
        logical_content_hash=bundle_hash,
        artifact_hash=bundle_artifact_hash,
        object_kind="case_admission_evidence_bundle",
    ).content_hash
    payloads["admission_evidence_bundle"] = bundle_raw
    descriptors.append(
        CaseStagedPayloadDescriptor(
            role="admission_evidence_bundle",
            logical_content_hash=bundle_hash,
            artifact_hash=bundle_artifact_hash,
            raw_size_bytes=len(bundle_raw),
            compression="gzip",
            media_type="application/json",
        )
    )
    return _TransitionFixture(
        restricted_root=root,
        ledger_path=ledger_path,
        cas_root=cas_root,
        transition_directory=transition_directory,
        payloads=payloads,
        descriptors=tuple(descriptors),
        bundle_hash=bundle_hash,
        bundle_artifact_hash=bundle_artifact_hash,
        bundle_reference_hash=bundle_reference_hash,
    )


def _capture(fixture: _TransitionFixture):
    return capture_case_admission_staging_intent(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        transition_id="phase6-admission-staging",
        execution_plan_hash=PLAN_HASH,
        admission_attestation_hash=ADMISSION_HASH,
        semantic_bundle_hash=SEMANTIC_HASH,
        expected_payloads=fixture.descriptors,
        evidence_bundle_hash=fixture.bundle_hash,
        evidence_bundle_artifact_hash=fixture.bundle_artifact_hash,
        evidence_bundle_reference_hash=fixture.bundle_reference_hash,
        captured_at=T1,
    )


def _stage(fixture: _TransitionFixture) -> None:
    with Ledger(fixture.ledger_path) as ledger:
        artifacts = ArtifactStore(
            BlobStore(fixture.cas_root, compression=Compression.GZIP),
            ledger,
        )
        descriptors = {item.role: item for item in fixture.descriptors}
        for role, raw in fixture.payloads.items():
            descriptor = descriptors[role]
            record = artifacts.put_bytes(
                raw,
                media_type=descriptor.media_type,
                release_class=ReleaseClass.RESTRICTED,
                created_at=T1,
            )
            assert record.content_hash == descriptor.artifact_hash


def _verify(fixture: _TransitionFixture, *, require_current_h1: bool = True):
    return verify_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        execution_plan_hash=PLAN_HASH,
        admission_attestation_hash=ADMISSION_HASH,
        semantic_bundle_hash=SEMANTIC_HASH,
        evidence_bundle_hash=fixture.bundle_hash,
        evidence_bundle_artifact_hash=fixture.bundle_artifact_hash,
        evidence_bundle_reference_hash=fixture.bundle_reference_hash,
        require_current_h1=require_current_h1,
    )


def _complete_staging(fixture: _TransitionFixture) -> None:
    _capture(fixture)
    _stage(fixture)
    finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )


def _post_h1_payloads() -> tuple[
    tuple[CasePostH1ArtifactDescriptor, ...],
    dict[str, bytes],
]:
    descriptors: list[CasePostH1ArtifactDescriptor] = []
    payloads: dict[str, bytes] = {}
    for role in ("case_source_manifest", "case_execution_admission"):
        value = {"bootstrap_object": role, "lineage": PLAN_HASH}
        raw = canonical_json(value).encode("utf-8")
        payloads[role] = raw
        descriptors.append(
            CasePostH1ArtifactDescriptor(
                role=role,
                logical_content_hash=canonical_sha256(value),
                artifact_hash=hashlib.sha256(raw).hexdigest(),
                raw_size_bytes=len(raw),
                compression="gzip",
                media_type="application/json",
            )
        )
    return tuple(descriptors), payloads


def _verify_post_h1(
    fixture: _TransitionFixture,
    descriptors: tuple[CasePostH1ArtifactDescriptor, ...],
):
    return verify_case_post_h1_artifact_prefix(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        expected_payloads=descriptors,
    )


def test_staging_transition_binds_exact_h0_h1_and_nine_payloads(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    intent = _capture(fixture)
    assert len(intent.expected_payloads) == 9
    assert intent.h0_ledger.allocated_gpu_microseconds == 1_000_000
    assert intent.h0_ledger.gpu_inventory_hash
    table_counts = {item.table_name: item.row_count for item in intent.h0_ledger.tables}
    assert table_counts["gpu_allocation_journal"] == 2
    assert table_counts["gpu_events"] == 1

    _stage(fixture)
    receipt = finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )
    assert len(receipt.added_artifacts) == 9
    assert len(receipt.added_cas_entries) == 9
    assert receipt.reused_expected_artifact_hashes == ()
    assert receipt.gpu_inventory_hash == intent.h0_ledger.gpu_inventory_hash
    assert receipt.allocated_gpu_microseconds == intent.h0_ledger.allocated_gpu_microseconds

    verified = _verify(fixture)
    assert verified.current_matches_archived_h1
    assert verified.current_is_append_only_successor
    for record_name in ("staging-intent.json", "staging-receipt.json"):
        serialized = (fixture.transition_directory / record_name).read_text()
        assert str(fixture.restricted_root) not in serialized
        private_home_prefix = "/ho" + "me/"
        assert private_home_prefix not in serialized
        assert "PRIVATE KEY" not in serialized


def test_finalize_recovers_power_loss_after_h1_ledger_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    paths = transition_module.case_admission_transition_paths(
        restricted_root=fixture.restricted_root,
        transition_directory=fixture.transition_directory,
    )
    publish_record = transition_module._publish_record_no_replace

    def interrupt_before_h1_manifest(path: Path, record) -> None:
        if path == paths.h1_cas_manifest:
            raise RuntimeError("simulated power loss")
        publish_record(path, record)

    monkeypatch.setattr(
        transition_module,
        "_publish_record_no_replace",
        interrupt_before_h1_manifest,
    )
    with pytest.raises(RuntimeError, match="simulated power loss"):
        finalize_case_admission_staging_transition(
            restricted_root=fixture.restricted_root,
            ledger_path=fixture.ledger_path,
            cas_root=fixture.cas_root,
            transition_directory=fixture.transition_directory,
            completed_at=T1 + timedelta(seconds=1),
        )
    assert paths.h1_ledger.is_file()
    assert not paths.h1_cas_manifest.exists()
    assert not paths.receipt.exists()

    monkeypatch.setattr(
        transition_module,
        "_publish_record_no_replace",
        publish_record,
    )
    receipt = finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )
    assert paths.h1_cas_manifest.is_file()
    assert receipt.h1_ledger.file_sha256 == hashlib.sha256(
        paths.h1_ledger.read_bytes()
    ).hexdigest()
    assert _verify(fixture).current_matches_archived_h1


def test_finalize_repairs_publisher_hardlink_left_by_power_loss(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    paths = transition_module.case_admission_transition_paths(
        restricted_root=fixture.restricted_root,
        transition_directory=fixture.transition_directory,
    )
    transition_module._publish_copy_no_replace(
        fixture.ledger_path,
        paths.h1_ledger,
    )
    interrupted_temporary = (
        paths.directory / f".{paths.h1_ledger.name}.{'a' * 32}.partial"
    )
    os.link(paths.h1_ledger, interrupted_temporary)
    assert paths.h1_ledger.stat().st_nlink == 2

    receipt = finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )

    assert not interrupted_temporary.exists()
    assert paths.h1_ledger.stat().st_nlink == 1
    assert receipt.h1_ledger.file_sha256 == hashlib.sha256(
        paths.h1_ledger.read_bytes()
    ).hexdigest()


def test_finalize_recovers_a_valid_manifest_only_h1_boundary(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    paths = transition_module.case_admission_transition_paths(
        restricted_root=fixture.restricted_root,
        transition_directory=fixture.transition_directory,
    )
    transition_module._publish_record_no_replace(
        paths.h1_cas_manifest,
        transition_module._scan_cas(fixture.cas_root),
    )

    receipt = finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )
    assert paths.h1_ledger.is_file()
    assert receipt.h1_cas_manifest_hash
    assert _verify(fixture).current_matches_archived_h1


def test_finalize_rejects_h1_archive_hardlinked_to_live_ledger(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    paths = transition_module.case_admission_transition_paths(
        restricted_root=fixture.restricted_root,
        transition_directory=fixture.transition_directory,
    )
    os.link(fixture.ledger_path, paths.h1_ledger)

    with pytest.raises(
        CaseStudyTransitionError,
        match=r"singly-linked|aliases its live source",
    ):
        finalize_case_admission_staging_transition(
            restricted_root=fixture.restricted_root,
            ledger_path=fixture.ledger_path,
            cas_root=fixture.cas_root,
            transition_directory=fixture.transition_directory,
            completed_at=T1 + timedelta(seconds=1),
        )


def test_verification_rejects_using_archived_h1_as_the_live_ledger(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _complete_staging(fixture)
    paths = transition_module.case_admission_transition_paths(
        restricted_root=fixture.restricted_root,
        transition_directory=fixture.transition_directory,
    )

    with pytest.raises(CaseStudyTransitionError, match="live ledger cannot be"):
        verify_case_admission_staging_transition(
            restricted_root=fixture.restricted_root,
            ledger_path=paths.h1_ledger,
            cas_root=fixture.cas_root,
            transition_directory=fixture.transition_directory,
            execution_plan_hash=PLAN_HASH,
            admission_attestation_hash=ADMISSION_HASH,
            semantic_bundle_hash=SEMANTIC_HASH,
            evidence_bundle_hash=fixture.bundle_hash,
            evidence_bundle_artifact_hash=fixture.bundle_artifact_hash,
            evidence_bundle_reference_hash=fixture.bundle_reference_hash,
        )


def test_resume_accepts_append_only_successor_but_returns_archived_h1(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    receipt = finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )
    with Ledger(fixture.ledger_path) as ledger:
        ArtifactStore(
            BlobStore(fixture.cas_root, compression=Compression.GZIP),
            ledger,
        ).put_bytes(
            b'{"bootstrap":"successor"}',
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=T1 + timedelta(seconds=2),
        )

    with pytest.raises(CaseStudyTransitionError, match="archived H1 boundary"):
        _verify(fixture)
    verified = _verify(fixture, require_current_h1=False)
    assert not verified.current_matches_archived_h1
    assert verified.current_is_append_only_successor
    assert verified.archived_h1_ledger_sha256 == receipt.h1_ledger.file_sha256
    assert verified.current_ledger_sha256 != verified.archived_h1_ledger_sha256


def test_post_h1_artifact_prefix_accepts_zero_one_and_two_artifacts(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _complete_staging(fixture)
    descriptors, payloads = _post_h1_payloads()
    zero = _verify_post_h1(fixture, descriptors)
    assert zero.completed_prefix_count == 0
    assert zero.newly_registered_prefix_roles == ()

    for expected_count, descriptor in enumerate(descriptors, start=1):
        with Ledger(fixture.ledger_path) as ledger:
            ArtifactStore(
                BlobStore(fixture.cas_root, compression=Compression.GZIP),
                ledger,
            ).put_bytes(
                payloads[descriptor.role],
                media_type=descriptor.media_type,
                release_class=ReleaseClass.RESTRICTED,
                created_at=T1 + timedelta(seconds=expected_count + 2),
            )
        verified = _verify_post_h1(fixture, descriptors)
        assert verified.completed_prefix_count == expected_count
        assert verified.newly_registered_prefix_roles == tuple(
            item.role for item in descriptors[:expected_count]
        )
        assert verified.archived_h1_gpu_inventory_hash == verified.current_gpu_inventory_hash


def test_post_h1_artifact_prefix_accepts_one_next_cas_orphan(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _complete_staging(fixture)
    descriptors, payloads = _post_h1_payloads()
    first = descriptors[0]
    BlobStore(fixture.cas_root, compression=Compression.GZIP).put_bytes(
        payloads[first.role],
        media_type=first.media_type,
        release_class=ReleaseClass.RESTRICTED,
        created_at=T1 + timedelta(seconds=3),
    )
    verified = _verify_post_h1(fixture, descriptors)
    assert verified.completed_prefix_count == 0
    assert verified.next_orphan_cas_role == first.role


def test_post_h1_artifact_prefix_rejects_unrelated_artifact(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _complete_staging(fixture)
    descriptors, _payloads = _post_h1_payloads()
    with Ledger(fixture.ledger_path) as ledger:
        ArtifactStore(
            BlobStore(fixture.cas_root, compression=Compression.GZIP),
            ledger,
        ).put_bytes(
            b'{"unrelated":"bootstrap"}',
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=T1 + timedelta(seconds=3),
        )
    with pytest.raises(CaseStudyTransitionError, match="not the declared post-H1 prefix"):
        _verify_post_h1(fixture, descriptors)


def test_post_h1_artifact_prefix_rejects_gpu_row(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _complete_staging(fixture)
    descriptors, _payloads = _post_h1_payloads()
    with Ledger(fixture.ledger_path) as ledger:
        ledger.record_gpu_event(
            event_id="post-h1-forbidden-gpu",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=0.25,
            started_at=T1 + timedelta(seconds=3),
            ended_at=T1 + timedelta(seconds=3, microseconds=250_000),
            succeeded=True,
            details={"forbidden": "bootstrap-prefix"},
        )
    with pytest.raises(CaseStudyTransitionError, match="GPU allocation inventory changed"):
        _verify_post_h1(fixture, descriptors)


def test_staging_recovers_preexisting_unregistered_expected_cas_blob(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    role = GATE_ROLES[0]
    descriptor = next(item for item in fixture.descriptors if item.role == role)
    orphan = BlobStore(fixture.cas_root, compression=Compression.GZIP).put_bytes(
        fixture.payloads[role],
        media_type=descriptor.media_type,
        release_class=ReleaseClass.RESTRICTED,
        created_at=T1,
    )
    assert orphan.content_hash == descriptor.artifact_hash
    _capture(fixture)
    _stage(fixture)
    receipt = finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )
    assert len(receipt.added_artifacts) == 9
    assert len(receipt.added_cas_entries) == 8
    assert descriptor.expected_relative_path not in {
        item.relative_path for item in receipt.added_cas_entries
    }
    assert _verify(fixture).current_matches_archived_h1


def test_finalize_rejects_undeclared_artifact_and_cas_delta(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    with Ledger(fixture.ledger_path) as ledger:
        ArtifactStore(
            BlobStore(fixture.cas_root, compression=Compression.GZIP),
            ledger,
        ).put_bytes(
            b'{"undeclared":true}',
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=T1,
        )
    with pytest.raises(CaseStudyTransitionError, match="declared staging set"):
        finalize_case_admission_staging_transition(
            restricted_root=fixture.restricted_root,
            ledger_path=fixture.ledger_path,
            cas_root=fixture.cas_root,
            transition_directory=fixture.transition_directory,
            completed_at=T1 + timedelta(seconds=1),
        )


def test_finalize_rejects_gpu_change_during_staging(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    with Ledger(fixture.ledger_path) as ledger:
        ledger.record_gpu_event(
            event_id="forbidden-during-stage",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=0.25,
            started_at=T1,
            ended_at=T1 + timedelta(microseconds=250_000),
            succeeded=True,
            details={"forbidden": "during-stage"},
        )
    with pytest.raises(CaseStudyTransitionError, match="GPU allocation inventory changed"):
        finalize_case_admission_staging_transition(
            restricted_root=fixture.restricted_root,
            ledger_path=fixture.ledger_path,
            cas_root=fixture.cas_root,
            transition_directory=fixture.transition_directory,
            completed_at=T1 + timedelta(seconds=1),
        )


def test_intent_replay_does_not_substitute_live_successor_for_h0(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    first = _capture(fixture)
    _stage(fixture)
    replay = _capture(fixture)
    assert replay == first
    assert replay.h0_ledger == first.h0_ledger


def test_transition_rejects_symlinked_output_directory(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    real = fixture.restricted_root / "real-transition"
    real.mkdir()
    alias = fixture.restricted_root / "alias-transition"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(CaseStudyTransitionError, match="symbolic links"):
        capture_case_admission_staging_intent(
            restricted_root=fixture.restricted_root,
            ledger_path=fixture.ledger_path,
            cas_root=fixture.cas_root,
            transition_directory=alias,
            transition_id="phase6-admission-staging",
            execution_plan_hash=PLAN_HASH,
            admission_attestation_hash=ADMISSION_HASH,
            semantic_bundle_hash=SEMANTIC_HASH,
            expected_payloads=fixture.descriptors,
            evidence_bundle_hash=fixture.bundle_hash,
            evidence_bundle_artifact_hash=fixture.bundle_artifact_hash,
            evidence_bundle_reference_hash=fixture.bundle_reference_hash,
            captured_at=T1,
        )


def test_serialized_records_are_valid_json(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _capture(fixture)
    _stage(fixture)
    finalize_case_admission_staging_transition(
        restricted_root=fixture.restricted_root,
        ledger_path=fixture.ledger_path,
        cas_root=fixture.cas_root,
        transition_directory=fixture.transition_directory,
        completed_at=T1 + timedelta(seconds=1),
    )
    for path in fixture.transition_directory.glob("*.json"):
        assert isinstance(json.loads(path.read_bytes()), dict)
