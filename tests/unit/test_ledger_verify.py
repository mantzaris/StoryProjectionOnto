from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from story_projection_onto.cli import main
from story_projection_onto.contracts import (
    ConditionName,
    PacketMaterializationEvent,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    QueryAccessEvent,
)
from story_projection_onto.ledger_verify import (
    LedgerVerificationError,
    audit_ledger,
    verify_ledger,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    CommitmentCheckStatus,
    Compression,
    EvidenceSupportStatus,
    GpuAllocationJournalState,
    GpuEventKind,
    GpuServiceJournalState,
    InputKind,
    JobState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    PacketMaterializationRecord,
    PrequeryBarrierRecord,
    QueryAccessRecord,
    ReleaseClass,
    RetryClass,
    SemanticAssessmentScope,
    StoragePreflight,
    TemporalValidationStatus,
    ValidationStatus,
)
from tests.unit.test_ui import context, packet, projection

T0 = "2026-09-01T00:00:00.000000Z"
T1 = "2026-09-01T00:00:01.000000Z"
T15 = "2026-09-01T00:00:01.500000Z"
T2 = "2026-09-01T00:00:02.000000Z"
T3 = "2026-09-01T00:00:03.000000Z"
HASH_A = "a" * 64
HASH_B = "b" * 64


@dataclass(frozen=True)
class _Fixture:
    database: Path
    cas_root: Path
    base_attempt_id: str
    retry_attempt_id: str
    raw_hash: str
    response_hash: str
    raw_link_id: str


@dataclass(frozen=True)
class _ProjectionFixture:
    database: Path
    cas_root: Path
    projection_id: str
    projection_artifact_hash: str
    projection_semantic_hash: str


@dataclass(frozen=True)
class _BoundaryFixture:
    database: Path
    cas_root: Path
    access_event_hash: str


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fixture(tmp_path: Path) -> _Fixture:
    database = tmp_path / "study.sqlite"
    cas_root = tmp_path / "cas"
    with Ledger(database) as ledger:
        artifacts = ArtifactStore(
            BlobStore(cas_root, compression=Compression.GZIP),
            ledger,
        )
        job = ledger.create_or_resume_job(
            {"run": "ledger-verification"},
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        base = ledger.record_attempt(
            attempt_id="base-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=1,
            created_at=T0,
        )
        retry = ledger.record_attempt(
            attempt_id="retry-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.RETRY,
            parent_attempt_id=base.attempt_id,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=1,
            created_at=T1,
        )
        raw = artifacts.put_bytes(
            b'{"request":"grounded"}\n',
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        response = artifacts.put_bytes(
            b'{"response":"qualified assertion"}\n',
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T1,
        )
        raw_link = ledger.link_artifact(
            job_id=job.job_id,
            content_hash=raw.content_hash,
            role="request",
            attempt_id=base.attempt_id,
            created_at=T0,
        )
        ledger.link_artifact(
            job_id=job.job_id,
            content_hash=response.content_hash,
            role="response",
            attempt_id=base.attempt_id,
            parent_content_hash=raw.content_hash,
            created_at=T1,
        )
        ledger.record_gpu_service_observation(
            service_session_id="service-session",
            state=GpuServiceJournalState.OPENED,
            session_id="persistent-service",
            configuration_hash=HASH_A,
            service_started_at=T0,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T0,
        )
        ledger.record_gpu_allocation_observation(
            allocation_id="inference-event",
            state=GpuAllocationJournalState.OPENED,
            intended_event_kind=GpuEventKind.INFERENCE,
            elapsed_seconds=0,
            maximum_seconds=10,
            observed_at=T0,
            job_id=job.job_id,
            attempt_id=base.attempt_id,
        )
        event = ledger.record_gpu_event(
            event_id="inference-event",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=1,
            started_at=T0,
            ended_at=T1,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=base.attempt_id,
            details={"meter": "test"},
        )
        ledger.record_gpu_allocation_observation(
            allocation_id="inference-event",
            state=GpuAllocationJournalState.CLOSED,
            intended_event_kind=GpuEventKind.INFERENCE,
            elapsed_seconds=1,
            maximum_seconds=10,
            observed_at=T1,
            job_id=job.job_id,
            attempt_id=base.attempt_id,
        )
        ledger.record_model_call(
            model_call_id="model-call",
            job_id=job.job_id,
            attempt_id=base.attempt_id,
            gpu_event_id=event.event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.BASE,
            model_manifest_hash=HASH_A,
            decoding_manifest_hash=HASH_B,
            request_hash=raw.content_hash,
            response_artifact_hash=response.content_hash,
            construction_unit_hash=HASH_A,
            served_context_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            allocated_gpu_seconds=1,
            successful=True,
            created_at=T1,
        )
        ledger.record_gpu_service_observation(
            service_session_id="service-session",
            state=GpuServiceJournalState.PROCESS_STOPPED,
            session_id="persistent-service",
            configuration_hash=HASH_A,
            service_started_at=T0,
            elapsed_seconds=1.5,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T15,
        )
        ledger.close_gpu_service_journal(
            service_session_id="service-session",
            session_id="persistent-service",
            service_seconds=1.5,
            classified_event_seconds=1,
            started_at=T0,
            ended_at=T15,
            details={"accounting": "test"},
        )
        report = StoragePreflight(tmp_path).check(
            current_occupied_bytes=1_000_000_000,
            declared_growth_bytes=100_000_000,
            filesystem_free_bytes=10_000_000_000,
        )
        ledger.record_storage_sample(report, phase="test", sampled_at=T1)
        ledger.record_resource_sample(
            sample_id="resource-sample",
            job_id=job.job_id,
            gpu_event_id=event.event_id,
            process_ram_bytes=100,
            system_available_ram_bytes=1_000,
            gpu_vram_bytes=200,
            project_storage_bytes=1_000_000_000,
            cpu_worker_count=1,
            sampled_at=T1,
        )
    return _Fixture(
        database=database,
        cas_root=cas_root,
        base_attempt_id=base.attempt_id,
        retry_attempt_id=retry.attempt_id,
        raw_hash=raw.content_hash,
        response_hash=response.content_hash,
        raw_link_id=raw_link,
    )


def _projection_fixture(
    tmp_path: Path,
    *,
    serialization: str = "canonical",
) -> _ProjectionFixture:
    database = tmp_path / "projection-study.sqlite"
    cas_root = tmp_path / "projection-cas"
    value = projection(ConditionName.C2_LLM_QUERY, context(), packet())
    canonical_payload = value.to_canonical_json()
    if serialization == "canonical":
        payload = (canonical_payload + "\n").encode("utf-8")
    elif serialization == "noncanonical":
        payload = json.dumps(json.loads(canonical_payload), indent=2).encode("utf-8")
    elif serialization == "forged":
        forged = json.loads(canonical_payload)
        forged["local_schema"]["contextual_types"][0]["label"] = "forged label"
        payload = json.dumps(forged, sort_keys=True, separators=(",", ":")).encode("utf-8")
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(serialization)
    with Ledger(database) as ledger:
        artifacts = ArtifactStore(BlobStore(cas_root, compression=Compression.GZIP), ledger)
        projection_artifact = artifacts.put_bytes(
            payload,
            media_type="application/vnd.story-projection.ontology-projection+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T2,
        )
        evidence_artifact = artifacts.put_bytes(
            (packet().to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.evidence-packet+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        ledger.register_study(
            study_id="projection-study",
            protocol_hash=HASH_A,
            code_manifest_hash=HASH_B,
            configuration_hash=HASH_A,
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        job = ledger.create_or_resume_job(
            {
                "run": "projection-audit",
                "condition": "C2",
                "lifecycle_kind": "query_time_projection",
            },
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        ledger.link_job_to_study(
            study_id="projection-study", job_id=job.job_id, created_at=T0
        )
        snapshot_input = ledger.register_input(
            input_id="projection-snapshot-input",
            study_id="projection-study",
            input_kind=InputKind.EVIDENCE_SNAPSHOT,
            content_hash=value.snapshot_hash,
            artifact_hash=evidence_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        packet_input = ledger.register_input(
            input_id="projection-packet-input",
            study_id="projection-study",
            input_kind=InputKind.EVIDENCE_PACKET,
            content_hash=value.packet_hash,
            artifact_hash=evidence_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        snapshot = ledger.register_evidence_snapshot(
            snapshot_id="projection-snapshot",
            input_id=snapshot_input.input_id,
            horizon_hash=HASH_A,
            evidence_manifest_hash=HASH_B,
            index_configuration_hash=HASH_A,
            prequery_seal_hash=HASH_B,
            eligible_evidence_count=len(packet().evidence),
            created_at=T0,
        )
        attempt = ledger.record_attempt(
            attempt_id="projection-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=7,
            created_at=T1,
        )
        ledger.record_model_call(
            model_call_id="projection-call",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id=None,
            backend=ModelBackend.HAND_AUTHORED_FIXTURE,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.BASE,
            model_manifest_hash=HASH_A,
            decoding_manifest_hash=HASH_B,
            request_hash=HASH_A,
            response_artifact_hash=projection_artifact.content_hash,
            construction_unit_hash=HASH_B,
            served_context_count=1,
            prompt_tokens=0,
            completion_tokens=0,
            allocated_gpu_seconds=0,
            successful=True,
            created_at=T2,
        )
        validation = ledger.record_validation(
            validation_id="projection-validation",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            input_artifact_hash=projection_artifact.content_hash,
            validator_manifest_hash=HASH_A,
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            created_at=T2,
        )
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, T0),
                (JobState.QUERY_REVEALED, T1),
                (JobState.GENERATED, T2),
                (JobState.VALIDATED, T3),
            ),
        )
        ledger.record_projection(
            projection_id="projection-ledger-row",
            job_id=job.job_id,
            validation_id=validation.validation_id,
            snapshot_id=snapshot.snapshot_id,
            packet_input_id=packet_input.input_id,
            condition_id="C2",
            context_hash=value.context_hash,
            upper_ontology_hash=value.upper_ontology.content_hash,
            construction_certificate_hash=value.construction_certificate.content_hash,
            projection_artifact_hash=projection_artifact.content_hash,
            projection_semantic_hash=value.content_hash,
            release_class=ReleaseClass.PUBLIC,
            finalized_at=T3,
        )
        ledger.transition_job(job.job_id, JobState.FINALIZED, occurred_at=T3)
    return _ProjectionFixture(
        database=database,
        cas_root=cas_root,
        projection_id="projection-ledger-row",
        projection_artifact_hash=projection_artifact.content_hash,
        projection_semantic_hash=value.content_hash,
    )


def _boundary_fixture(tmp_path: Path) -> _BoundaryFixture:
    database = tmp_path / "boundary-study.sqlite"
    cas_root = tmp_path / "boundary-cas"
    evidence_packet = packet()
    barrier = PrequeryBarrier(
        barrier_id="boundary-fixture-barrier",
        execution_id="boundary-fixture-execution",
        execution_manifest_hash=_digest("boundary execution manifest"),
        neutral_evidence_artifact_hashes=(_digest("neutral evidence"),),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id="boundary-unit",
                condition=ConditionName.C0_CLASSICAL_PRE,
                snapshot_hash=evidence_packet.snapshot_hash,
                preparation_hash=_digest("preparation"),
                lineage_artifact_hash=_digest("lineage artifact"),
                completed_at=T0,
            ),
        ),
        sealed_at=T1,
    )
    query_artifact_hash = _digest("query logical object")
    access = QueryAccessEvent(
        access_event_id="boundary-fixture-access",
        execution_id=barrier.execution_id,
        query_context_hash=context().content_hash,
        model_visible_query_hash=_digest("model visible query"),
        snapshot_hash=evidence_packet.snapshot_hash,
        stage_manifest_hash=_digest("query stage"),
        query_artifact_hash=query_artifact_hash,
        prequery_barrier_hash=barrier.content_hash,
        registered_revealed_at=T2,
        accessed_at=T2,
    )
    materialization = PacketMaterializationEvent(
        materialization_event_id="boundary-fixture-materialization",
        execution_id=barrier.execution_id,
        query_access_event_hash=access.content_hash,
        snapshot_hash=evidence_packet.snapshot_hash,
        packet_hash=evidence_packet.content_hash,
        retrieval_method="all_admissible",
        retrieval_config_hash=_digest("retrieval configuration"),
        started_at=T3,
        completed_at=T3,
    )
    with Ledger(database) as ledger:
        artifacts = ArtifactStore(BlobStore(cas_root, compression=Compression.GZIP), ledger)
        barrier_artifact = artifacts.put_bytes(
            (barrier.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.prequery-barrier+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T1,
        )
        query_artifact = artifacts.put_bytes(
            (json.dumps({"content_hash": query_artifact_hash}) + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.query+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T2,
        )
        access_artifact = artifacts.put_bytes(
            (access.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.query-access+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T2,
        )
        packet_artifact = artifacts.put_bytes(
            (evidence_packet.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.evidence-packet+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )
        materialization_artifact = artifacts.put_bytes(
            (materialization.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.packet-materialization+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )
        ledger.persist_prequery_barrier(
            PrequeryBarrierRecord(
                barrier_hash=barrier.content_hash,
                barrier_id=barrier.barrier_id,
                execution_id=barrier.execution_id,
                execution_manifest_hash=barrier.execution_manifest_hash,
                barrier_artifact_hash=barrier_artifact.content_hash,
                preparation_count=len(barrier.preparation_bindings),
                sealed_at=T1,
                persisted_at=T1,
                release_class=ReleaseClass.PUBLIC,
            ),
            barrier_artifact=barrier_artifact,
        )
        ledger.persist_query_access(
            QueryAccessRecord(
                access_event_hash=access.content_hash,
                access_event_id=access.access_event_id,
                execution_id=access.execution_id,
                query_context_hash=access.query_context_hash,
                model_visible_query_hash=access.model_visible_query_hash,
                snapshot_hash=access.snapshot_hash,
                stage_manifest_hash=access.stage_manifest_hash,
                query_artifact_hash=access.query_artifact_hash,
                prequery_barrier_hash=access.prequery_barrier_hash,
                packet_hash=None,
                query_payload_artifact_hash=query_artifact.content_hash,
                access_event_artifact_hash=access_artifact.content_hash,
                registered_revealed_at=T2,
                accessed_at=T2,
                release_class=ReleaseClass.PUBLIC,
            ),
            query_payload_artifact=query_artifact,
            access_event_artifact=access_artifact,
        )
        ledger.persist_packet_materialization(
            PacketMaterializationRecord(
                materialization_event_hash=materialization.content_hash,
                materialization_event_id=materialization.materialization_event_id,
                execution_id=materialization.execution_id,
                query_access_event_hash=materialization.query_access_event_hash,
                snapshot_hash=materialization.snapshot_hash,
                packet_hash=materialization.packet_hash,
                retrieval_method=materialization.retrieval_method.value,
                retrieval_config_hash=materialization.retrieval_config_hash,
                packet_artifact_hash=packet_artifact.content_hash,
                materialization_event_artifact_hash=(
                    materialization_artifact.content_hash
                ),
                started_at=T3,
                completed_at=T3,
                release_class=ReleaseClass.PUBLIC,
            ),
            packet_artifact=packet_artifact,
            materialization_event_artifact=materialization_artifact,
        )
    return _BoundaryFixture(
        database=database,
        cas_root=cas_root,
        access_event_hash=access.content_hash,
    )


def _drop_update_guard(connection: sqlite3.Connection, table: str) -> None:
    connection.execute(f"DROP TRIGGER {table}_reject_update")


def test_exhaustive_verification_is_read_only_and_reconciles_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    database_before = fixture.database.read_bytes()
    database_mtime_before = fixture.database.stat().st_mtime_ns
    blobs_before = {
        path.relative_to(fixture.cas_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fixture.cas_root.rglob("*")
        if path.is_file()
    }

    report = verify_ledger(fixture.database, fixture.cas_root)

    assert report.valid
    assert report.artifact_count == 2
    assert report.attempt_count == 2
    assert report.artifact_link_count == 2
    assert report.model_call_count == 1
    assert report.gpu_event_count == 1
    assert report.gpu_service_session_count == 1
    assert report.gpu_total_allocated_microseconds == 1_500_000
    assert dict(report.gpu_by_kind_microseconds) == {
        "inference": 1_000_000,
        "service_overhead": 500_000,
    }
    assert report.storage_sample_count == 1
    assert report.peak_recorded_project_storage_bytes == 1_000_000_000
    assert fixture.database.read_bytes() == database_before
    assert fixture.database.stat().st_mtime_ns == database_mtime_before
    assert {
        path.relative_to(fixture.cas_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in fixture.cas_root.rglob("*")
        if path.is_file()
    } == blobs_before

    assert (
        main(
            [
                "verify-ledger",
                "--ledger",
                str(fixture.database),
                "--cas-root",
                str(fixture.cas_root),
            ]
        )
        == 0
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "verified"
    assert cli_payload["gpu_total_allocated_microseconds"] == 1_500_000


def test_verification_detects_corrupt_registered_cas_payload(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    blob = next(fixture.cas_root.rglob(f"{fixture.response_hash}.jsonl.gz"))
    encoded = bytearray(blob.read_bytes())
    encoded[len(encoded) // 2] ^= 0xFF
    blob.write_bytes(encoded)

    report = audit_ledger(fixture.database, fixture.cas_root)

    assert not report.valid
    assert any(
        issue.code in {"cas_decode_failed", "cas_hash_mismatch"}
        and issue.subject == fixture.response_hash
        for issue in report.issues
    )
    with pytest.raises(LedgerVerificationError) as caught:
        verify_ledger(fixture.database, fixture.cas_root)
    assert caught.value.report == report


def test_verification_detects_attempt_and_artifact_lineage_cycles(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with sqlite3.connect(fixture.database) as connection:
        _drop_update_guard(connection, "attempts")
        connection.execute(
            "UPDATE attempts SET attempt_kind = 'retry', parent_attempt_id = ? "
            "WHERE attempt_id = ?",
            (fixture.retry_attempt_id, fixture.base_attempt_id),
        )
        _drop_update_guard(connection, "job_artifacts")
        connection.execute(
            "UPDATE job_artifacts SET parent_content_hash = ? WHERE link_id = ?",
            (fixture.response_hash, fixture.raw_link_id),
        )

    codes = {issue.code for issue in audit_ledger(fixture.database, fixture.cas_root).issues}

    assert "attempt_cycle" in codes
    assert "artifact_cycle" in codes


def test_verification_detects_query_boundary_bypass_and_semantic_scope_claim(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with sqlite3.connect(fixture.database) as connection:
        job_id = connection.execute("SELECT job_id FROM jobs").fetchone()[0]
        connection.execute(
            """INSERT INTO job_transitions
                   (job_id, sequence, from_state, to_state, occurred_at)
               VALUES (?, 1, 'planned', 'prequery_sealed', ?)""",
            (job_id, T0),
        )
        connection.execute(
            """INSERT INTO job_transitions
                   (job_id, sequence, from_state, to_state, occurred_at)
               VALUES (?, 2, 'prequery_sealed', 'generated', ?)""",
            (job_id, T1),
        )
        connection.execute(
            """INSERT INTO validations (
                   validation_id, job_id, attempt_id, input_artifact_hash,
                   validator_manifest_hash, validation_status,
                   evidence_support_status, temporal_status, commitment_status,
                   semantic_assessment_scope, diagnostics_artifact_hash,
                   parent_validation_id, repair_attempt_id, created_at
               ) VALUES (?, ?, ?, ?, ?, 'accepted', 'supported', 'not_applicable',
                         'not_applicable', 'runtime_structural_only_not_assessed',
                         NULL, NULL, NULL, ?)""",
            (
                "invalid-runtime-scope",
                job_id,
                fixture.base_attempt_id,
                fixture.raw_hash,
                HASH_A,
                T1,
            ),
        )

    codes = {issue.code for issue in audit_ledger(fixture.database, fixture.cas_root).issues}

    assert "job_transition_query_boundary_bypass" in codes
    assert "validation_runtime_scope_claims_semantics" in codes


def test_verification_replays_a_canonical_projection_end_to_end(tmp_path: Path) -> None:
    fixture = _projection_fixture(tmp_path)

    report = verify_ledger(fixture.database, fixture.cas_root)

    assert report.valid
    assert report.job_count == 1
    assert report.attempt_count == 1
    assert report.legacy_validation_scope_count == 0


@pytest.mark.parametrize(
    ("serialization", "expected_code"),
    [
        ("noncanonical", "projection_artifact_noncanonical"),
        ("forged", "projection_artifact_payload_invalid"),
    ],
)
def test_verification_rejects_noncanonical_or_internally_forged_projection(
    tmp_path: Path,
    serialization: str,
    expected_code: str,
) -> None:
    fixture = _projection_fixture(tmp_path, serialization=serialization)

    codes = {issue.code for issue in audit_ledger(fixture.database, fixture.cas_root).issues}

    assert expected_code in codes


def test_verification_rejects_projection_semantic_identity_tamper(tmp_path: Path) -> None:
    fixture = _projection_fixture(tmp_path)
    with sqlite3.connect(fixture.database) as connection:
        _drop_update_guard(connection, "projections")
        connection.execute(
            "UPDATE projections SET projection_semantic_hash = ? WHERE projection_id = ?",
            (HASH_A, fixture.projection_id),
        )

    codes = {issue.code for issue in audit_ledger(fixture.database, fixture.cas_root).issues}

    assert "projection_semantic_hash_mismatch" in codes


def test_verification_rejects_projection_row_to_artifact_binding_tamper(
    tmp_path: Path,
) -> None:
    fixture = _projection_fixture(tmp_path)
    with sqlite3.connect(fixture.database) as connection:
        _drop_update_guard(connection, "projections")
        connection.execute(
            "UPDATE projections SET context_hash = ? WHERE projection_id = ?",
            (_digest("wrong projection context"), fixture.projection_id),
        )

    issues = audit_ledger(fixture.database, fixture.cas_root).issues

    assert any(
        issue.code == "projection_artifact_ledger_binding_mismatch"
        and issue.subject == f"{fixture.projection_id}:context_hash"
        for issue in issues
    )


def test_verification_requires_failure_lineage_for_every_rejected_validation(
    tmp_path: Path,
) -> None:
    fixture = _projection_fixture(tmp_path)
    with sqlite3.connect(fixture.database) as connection:
        _drop_update_guard(connection, "validations")
        connection.execute(
            "UPDATE validations SET validation_status = 'rejected' "
            "WHERE validation_id = 'projection-validation'"
        )

    codes = {issue.code for issue in audit_ledger(fixture.database, fixture.cas_root).issues}

    assert "rejected_validation_failure_missing" in codes


def test_verification_replays_and_rejects_tampered_query_boundary(tmp_path: Path) -> None:
    fixture = _boundary_fixture(tmp_path)
    assert verify_ledger(fixture.database, fixture.cas_root).valid

    with sqlite3.connect(fixture.database) as connection:
        _drop_update_guard(connection, "query_access_events")
        connection.execute(
            "UPDATE query_access_events SET snapshot_hash = ? "
            "WHERE access_event_hash = ?",
            (_digest("unsealed snapshot"), fixture.access_event_hash),
        )

    codes = {issue.code for issue in audit_ledger(fixture.database, fixture.cas_root).issues}
    assert "query_access_snapshot_not_sealed" in codes
    assert "query_access_event_artifact_lineage_mismatch" in codes
    assert "packet_materialization_lineage_mismatch" in codes


def test_phase1_looking_identity_without_terminal_gpu_artifact_is_not_exempt(
    tmp_path: Path,
) -> None:
    database = tmp_path / "phase1-impostor.sqlite"
    cas_root = tmp_path / "phase1-impostor-cas"
    with Ledger(database) as ledger:
        artifacts = ArtifactStore(BlobStore(cas_root, compression=Compression.GZIP), ledger)
        output = artifacts.put_bytes(
            b'{"structurally":"valid"}\n',
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T2,
        )
        job = ledger.create_or_resume_job(
            {
                "run_id": "phase1-impostor",
                "call_id": "c2-01",
                "plan_hash": HASH_A,
                "condition": "C2",
                "lifecycle_kind": "query_time_generation",
            },
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        attempt = ledger.record_attempt(
            attempt_id="phase1-impostor-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=1,
            created_at=T1,
        )
        ledger.record_model_call(
            model_call_id="phase1-impostor-call",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id=None,
            backend=ModelBackend.HAND_AUTHORED_FIXTURE,
            call_role=ModelCallRole.PILOT,
            retry_class=RetryClass.BASE,
            model_manifest_hash=HASH_A,
            decoding_manifest_hash=HASH_B,
            request_hash=HASH_A,
            response_artifact_hash=output.content_hash,
            construction_unit_hash=HASH_B,
            served_context_count=1,
            prompt_tokens=0,
            completion_tokens=0,
            allocated_gpu_seconds=0,
            successful=True,
            created_at=T2,
        )
        ledger.record_validation(
            validation_id="phase1-impostor-validation",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            input_artifact_hash=output.content_hash,
            validator_manifest_hash=HASH_A,
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            created_at=T2,
        )
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, T0),
                (JobState.QUERY_REVEALED, T1),
                (JobState.GENERATED, T2),
                (JobState.VALIDATED, T3),
                (JobState.FINALIZED, T3),
            ),
        )

    codes = {issue.code for issue in audit_ledger(database, cas_root).issues}

    assert "job_finalized_output_missing" in codes


def test_verification_aggregates_fk_gpu_and_storage_inconsistency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    with sqlite3.connect(fixture.database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        _drop_update_guard(connection, "model_calls")
        connection.execute(
            "UPDATE model_calls SET job_id = 'missing-job', "
            "allocated_gpu_microseconds = allocated_gpu_microseconds + 1"
        )
        _drop_update_guard(connection, "storage_samples")
        connection.execute(
            "UPDATE storage_samples SET projected_occupied_bytes = "
            "projected_occupied_bytes + 1"
        )
        _drop_update_guard(connection, "attempts")
        connection.execute(
            "UPDATE attempts SET parent_attempt_id = ? WHERE attempt_id = ?",
            ("c" * 64, fixture.retry_attempt_id),
        )
        _drop_update_guard(connection, "job_artifacts")
        connection.execute(
            "UPDATE job_artifacts SET parent_content_hash = ? WHERE content_hash = ?",
            ("d" * 64, fixture.response_hash),
        )

    report = audit_ledger(fixture.database, fixture.cas_root)
    codes = {issue.code for issue in report.issues}

    assert "sqlite_foreign_key_failed" in codes
    assert "model_call_job_missing" in codes
    assert "model_call_gpu_binding_mismatch" in codes
    assert "model_call_gpu_duration_mismatch" in codes
    assert "attempt_parent_missing" in codes
    assert "artifact_parent_missing" in codes
    assert "storage_projection_mismatch" in codes
    assert "storage_sample_id_mismatch" in codes
    assert (
        main(
            [
                "verify-ledger",
                "--ledger",
                str(fixture.database),
                "--cas-root",
                str(fixture.cas_root),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"
