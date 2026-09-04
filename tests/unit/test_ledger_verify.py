from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from story_projection_onto.cli import main
from story_projection_onto.ledger_verify import (
    LedgerVerificationError,
    audit_ledger,
    verify_ledger,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    Compression,
    GpuAllocationJournalState,
    GpuEventKind,
    GpuServiceJournalState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
    StoragePreflight,
)

T0 = "2026-09-01T00:00:00.000000Z"
T1 = "2026-09-01T00:00:01.000000Z"
T15 = "2026-09-01T00:00:01.500000Z"
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
