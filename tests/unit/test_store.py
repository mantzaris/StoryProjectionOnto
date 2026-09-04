from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

import story_projection_onto.store as store_module
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    CommitmentCheckStatus,
    Compression,
    CompressionUnavailableError,
    DuplicateConflictError,
    EvidenceSupportStatus,
    FailureKind,
    FeedbackAction,
    FeedbackKind,
    FeedbackResolutionStatus,
    GpuBudgetExceeded,
    GpuEventKind,
    GpuServiceJournalState,
    InputKind,
    InvalidTransitionError,
    JobState,
    Ledger,
    MetricStatus,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    ReleaseViolationError,
    RetryClass,
    StorageBudgetExceeded,
    StoragePreflight,
    TemporalValidationStatus,
    ValidationStatus,
)

T0 = "2026-09-03T12:00:00Z"
T1 = "2026-09-03T12:00:01Z"
T2 = "2026-09-03T12:00:02Z"
T3 = "2026-09-03T12:00:03Z"
HASH_A = "a" * 64
HASH_B = "b" * 64


@pytest.fixture
def ledger(tmp_path: Path):
    active = Ledger(tmp_path / "study.sqlite3")
    try:
        yield active
    finally:
        active.close()


def test_job_state_is_append_only_strict_and_resumable(ledger: Ledger) -> None:
    identity = {"snapshot_hash": HASH_A, "config_hash": HASH_B, "condition": "C2"}
    job = ledger.create_or_resume_job(identity, release_class=ReleaseClass.PUBLIC, created_at=T0)

    resumed = ledger.create_or_resume_job(identity, release_class=ReleaseClass.PUBLIC)
    assert resumed.job_id == job.job_id
    assert resumed.state is JobState.PLANNED
    assert ledger.count_rows("jobs") == 1
    assert ledger.count_rows("job_transitions") == 1

    sealed = ledger.transition_job(job.job_id, JobState.PREQUERY_SEALED, occurred_at=T1)
    duplicate = ledger.transition_job(job.job_id, JobState.PREQUERY_SEALED)
    assert duplicate.event_id == sealed.event_id
    assert ledger.count_rows("job_transitions") == 2

    with pytest.raises(InvalidTransitionError):
        ledger.transition_job(job.job_id, JobState.GENERATED)

    for state in (
        JobState.QUERY_REVEALED,
        JobState.GENERATED,
        JobState.REPAIRED,
        JobState.VALIDATED,
        JobState.FINALIZED,
        JobState.SCORED,
        JobState.RENDERED,
    ):
        ledger.transition_job(job.job_id, state)

    transitions = ledger.transitions(job.job_id)
    assert transitions[0].from_state is None
    assert transitions[0].to_state is JobState.PLANNED
    assert [event.sequence for event in transitions] == list(range(len(transitions)))
    assert ledger.get_job(job.job_id).state is JobState.RENDERED

    with pytest.raises(DuplicateConflictError):
        ledger.create_or_resume_job(identity, release_class=ReleaseClass.RESTRICTED)


def test_sqlite_triggers_reject_update_and_delete(tmp_path: Path) -> None:
    database = tmp_path / "study.sqlite3"
    with Ledger(database) as ledger:
        job = ledger.create_or_resume_job(
            {"run": "immutable"}, release_class=ReleaseClass.PUBLIC, created_at=T0
        )

        bypass = sqlite3.connect(str(database))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                bypass.execute(
                    "UPDATE jobs SET release_class = 'restricted' WHERE job_id = ?",
                    (job.job_id,),
                )
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                bypass.execute("DELETE FROM job_transitions WHERE job_id = ?", (job.job_id,))
        finally:
            bypass.close()


def test_content_addressed_blobs_deduplicate_and_enforce_release_class(
    tmp_path: Path, ledger: Ledger
) -> None:
    blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
    artifacts = ArtifactStore(blobs, ledger)
    payload = b'{"answer":"grounded"}\n'

    first = artifacts.put_bytes(
        payload,
        media_type="application/x-ndjson",
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    second = artifacts.put_bytes(
        payload,
        media_type="application/x-ndjson",
        release_class=ReleaseClass.PUBLIC,
    )

    assert second == first
    assert ledger.count_rows("artifacts") == 1
    assert len(list((tmp_path / "blobs").rglob("*.jsonl.gz"))) == 1
    assert blobs.read_bytes(first) == payload
    assert ledger.get_artifact(first.content_hash, for_public_release=True) == first

    restricted = artifacts.put_bytes(
        b"opaque restricted payload",
        media_type="application/octet-stream",
        release_class=ReleaseClass.RESTRICTED,
        created_at=T1,
    )
    with pytest.raises(ReleaseViolationError):
        blobs.read_bytes(restricted)
    assert blobs.read_bytes(restricted, allow_restricted=True) == b"opaque restricted payload"
    with pytest.raises(ReleaseViolationError):
        ledger.get_artifact(restricted.content_hash, for_public_release=True)

    public_job = ledger.create_or_resume_job(
        {"run": "public"}, release_class=ReleaseClass.PUBLIC, created_at=T0
    )
    with pytest.raises(ReleaseViolationError):
        ledger.link_artifact(
            job_id=public_job.job_id,
            content_hash=restricted.content_hash,
            role="raw_response",
        )

    # Content classification is immutable; the same bytes cannot later be relabeled.
    relabeled = blobs.put_bytes(
        payload,
        media_type="application/x-ndjson",
        release_class=ReleaseClass.RESTRICTED,
    )
    with pytest.raises(DuplicateConflictError):
        ledger.register_artifact(relabeled)


def test_requested_zstd_never_falls_back_to_gzip(tmp_path: Path, monkeypatch) -> None:
    def unavailable():
        raise CompressionUnavailableError("deliberately unavailable")

    monkeypatch.setattr(store_module, "_import_zstandard", unavailable)
    with pytest.raises(CompressionUnavailableError, match="deliberately unavailable"):
        BlobStore(tmp_path / "zstd", compression=Compression.ZSTD)
    assert not list(tmp_path.rglob("*.gz"))


def test_interrupted_blob_is_quarantined_and_resume_registers_existing_blob(
    tmp_path: Path, ledger: Ledger
) -> None:
    blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
    interrupted = blobs.partial_directory / ("f" * 64 + ".deadbeef.partial")
    interrupted.write_bytes(b"incomplete")

    quarantined = blobs.quarantine_partials()
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"incomplete"
    assert not interrupted.exists()

    # Simulate a stop after atomic rename but before the SQLite artifact insert.
    pending = blobs.put_bytes(
        b"completed response\n",
        media_type="application/x-ndjson",
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    assert ledger.count_rows("artifacts") == 0

    resumed = ArtifactStore(blobs, ledger).put_bytes(
        b"completed response\n",
        media_type="application/x-ndjson",
        release_class=ReleaseClass.PUBLIC,
    )
    assert resumed.content_hash == pending.content_hash
    assert ledger.count_rows("artifacts") == 1
    assert len(list((tmp_path / "blobs").rglob("*.jsonl.gz"))) == 1


def test_failure_repair_and_artifact_lineage_are_immutable(tmp_path: Path, ledger: Ledger) -> None:
    job = ledger.create_or_resume_job(
        {"run": "repair"}, release_class=ReleaseClass.RESTRICTED, created_at=T0
    )
    base = ledger.record_attempt(
        attempt_id="base-1",
        job_id=job.job_id,
        attempt_kind=AttemptKind.BASE,
        input_hash=HASH_A,
        config_hash=HASH_B,
        seed=17,
        created_at=T0,
    )
    failure = ledger.record_failure(
        attempt_id=base.attempt_id,
        failure_kind=FailureKind.INVALID_OUTPUT,
        message="schema validation failed",
        details={"path": "assertions[0].time"},
        occurred_at=T1,
    )
    assert (
        ledger.record_failure(
            attempt_id=base.attempt_id,
            failure_kind=FailureKind.INVALID_OUTPUT,
            message="schema validation failed",
            details={"path": "assertions[0].time"},
        )
        == failure
    )

    repair = ledger.record_attempt(
        attempt_id="repair-1",
        job_id=job.job_id,
        attempt_kind=AttemptKind.REPAIR,
        parent_attempt_id=base.attempt_id,
        input_hash=HASH_A,
        config_hash=HASH_B,
        seed=17,
        created_at=T2,
    )
    retry = ledger.record_attempt(
        attempt_id="retry-1",
        job_id=job.job_id,
        attempt_kind=AttemptKind.RETRY,
        parent_attempt_id=repair.attempt_id,
        input_hash=HASH_A,
        config_hash=HASH_B,
        seed=17,
        created_at=T3,
    )
    assert [entry.attempt_id for entry in ledger.attempt_lineage(retry.attempt_id)] == [
        "base-1",
        "repair-1",
        "retry-1",
    ]
    assert ledger.failures_for_lineage(retry.attempt_id) == (failure,)

    with pytest.raises(DuplicateConflictError, match="single permitted repair"):
        ledger.record_attempt(
            attempt_id="repair-2",
            job_id=job.job_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=base.attempt_id,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=17,
        )
    with pytest.raises(DuplicateConflictError):
        ledger.record_failure(
            attempt_id=base.attempt_id,
            failure_kind=FailureKind.TIMEOUT,
            message="different immutable failure",
        )

    blob_store = BlobStore(tmp_path / "lineage-blobs", compression=Compression.GZIP)
    artifact_store = ArtifactStore(blob_store, ledger)
    raw = artifact_store.put_bytes(
        b"invalid raw output",
        media_type="application/x-ndjson",
        release_class=ReleaseClass.RESTRICTED,
    )
    fixed = artifact_store.put_bytes(
        b"repaired raw output",
        media_type="application/x-ndjson",
        release_class=ReleaseClass.RESTRICTED,
    )
    ledger.link_artifact(
        job_id=job.job_id,
        content_hash=raw.content_hash,
        role="invalid_raw",
        attempt_id=base.attempt_id,
    )
    link = ledger.link_artifact(
        job_id=job.job_id,
        content_hash=fixed.content_hash,
        parent_content_hash=raw.content_hash,
        role="repair_raw",
        attempt_id=repair.attempt_id,
    )
    assert (
        ledger.link_artifact(
            job_id=job.job_id,
            content_hash=fixed.content_hash,
            parent_content_hash=raw.content_hash,
            role="repair_raw",
            attempt_id=repair.attempt_id,
        )
        == link
    )
    assert ledger.count_rows("job_artifacts") == 2


def test_gpu_accounting_counts_load_failure_and_restart_and_is_monotonic(
    ledger: Ledger,
) -> None:
    job = ledger.create_or_resume_job(
        {"run": "gpu-meter"}, release_class=ReleaseClass.PUBLIC, created_at=T0
    )
    attempt = ledger.record_attempt(
        attempt_id="gpu-attempt",
        job_id=job.job_id,
        attempt_kind=AttemptKind.BASE,
        input_hash=HASH_A,
        config_hash=HASH_B,
        seed=3,
        created_at=T0,
    )
    events = (
        ("load", GpuEventKind.MODEL_LOAD, 2.25, T0, T1, True),
        ("failed", GpuEventKind.FAILURE, 3.5, T1, T2, False),
        ("restart", GpuEventKind.RESTART, 1.5, T2, T3, True),
    )
    for event_id, kind, seconds, start, end, succeeded in events:
        ledger.record_gpu_event(
            event_id=event_id,
            event_kind=kind,
            allocated_seconds=seconds,
            started_at=start,
            ended_at=end,
            succeeded=succeeded,
            attempt_id=attempt.attempt_id if event_id == "failed" else None,
            job_id=job.job_id,
            details={"meter": "allocated-service"},
        )

    summary = ledger.gpu_summary()
    assert summary.event_count == 3
    assert summary.total_allocated_seconds == pytest.approx(7.25)
    assert summary.seconds_for(GpuEventKind.MODEL_LOAD) == pytest.approx(2.25)
    assert summary.seconds_for(GpuEventKind.FAILURE) == pytest.approx(3.5)
    assert summary.seconds_for(GpuEventKind.RESTART) == pytest.approx(1.5)
    prefixed = ledger.gpu_events_with_prefix("f")
    assert [event.event_id for event in prefixed] == ["failed"]
    assert ledger.gpu_events_with_prefix("missing-") == ()
    with pytest.raises(ValueError, match="event_id_prefix"):
        ledger.gpu_events_with_prefix("")

    duplicate = ledger.record_gpu_event(
        event_id="failed",
        event_kind=GpuEventKind.FAILURE,
        allocated_seconds=3.5,
        started_at=T1,
        ended_at=T2,
        succeeded=False,
        attempt_id=attempt.attempt_id,
        job_id=job.job_id,
        details={"meter": "allocated-service"},
    )
    assert duplicate.event_id == "failed"
    assert ledger.gpu_summary().total_allocated_seconds == pytest.approx(7.25)

    with pytest.raises(DuplicateConflictError):
        ledger.record_gpu_event(
            event_id="failed",
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds=99,
            started_at=T1,
            ended_at=T2,
            succeeded=False,
            attempt_id=attempt.attempt_id,
            job_id=job.job_id,
        )

    ledger.require_gpu_capacity(2.749, hard_limit_seconds=10)
    with pytest.raises(GpuBudgetExceeded, match="reach/cross"):
        ledger.require_gpu_capacity(2.75, hard_limit_seconds=10)


def test_gpu_service_reconciliation_absorbs_only_microsecond_rounding(
    ledger: Ledger,
) -> None:
    for event_id in ("rounded-a", "rounded-b"):
        ledger.record_gpu_event(
            event_id=event_id,
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=0.0000005,
            started_at=T0,
            ended_at=T0,
            succeeded=True,
        )
    assert ledger.gpu_summary().total_allocated_microseconds == 2

    session = ledger.record_gpu_service_session(
        service_session_id="rounded-session",
        session_id="rounding-regression",
        service_seconds=0.0000014,
        classified_event_seconds=0.000002,
        started_at=T0,
        ended_at=T0,
    )
    assert session.service_microseconds == 2
    assert session.classified_event_microseconds == 2
    assert session.overhead_microseconds == 0
    assert ledger.gpu_summary().total_allocated_microseconds == 2

    with pytest.raises(ValueError, match="session-derived"):
        ledger.record_gpu_event(
            event_id="forbidden-synthetic-overhead",
            event_kind=GpuEventKind.SERVICE_OVERHEAD,
            allocated_seconds=1,
            started_at=T0,
            ended_at=T1,
            succeeded=True,
        )


def test_gpu_service_journal_closes_only_after_durable_process_stop(
    ledger: Ledger,
) -> None:
    opened = ledger.record_gpu_service_observation(
        service_session_id="service-journal-normal",
        state=GpuServiceJournalState.OPENED,
        session_id="pilot",
        configuration_hash=HASH_A,
        service_started_at=T0,
        elapsed_seconds=0,
        ledger_allocated_seconds_before_session=0,
        hard_limit_seconds=36_000,
        observed_at=T0,
    )
    assert opened.sequence == 0
    assert ledger.unresolved_gpu_service_journals() == (opened,)
    ledger.record_gpu_service_observation(
        service_session_id=opened.service_session_id,
        state=GpuServiceJournalState.HEARTBEAT,
        session_id=opened.session_id,
        configuration_hash=opened.configuration_hash,
        service_started_at=opened.service_started_at,
        elapsed_seconds=1,
        ledger_allocated_seconds_before_session=0,
        hard_limit_seconds=36_000,
        observed_at=T1,
    )
    with pytest.raises(ValueError, match="process-stopped"):
        ledger.close_gpu_service_journal(
            service_session_id=opened.service_session_id,
            session_id=opened.session_id,
            service_seconds=2,
            classified_event_seconds=0,
            started_at=T0,
            ended_at=T2,
        )
    ledger.record_gpu_event(
        event_id="service-journal-inference",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=0.5,
        started_at=T1,
        ended_at=T2,
        succeeded=True,
    )
    ledger.record_gpu_service_observation(
        service_session_id=opened.service_session_id,
        state=GpuServiceJournalState.PROCESS_STOPPED,
        session_id=opened.session_id,
        configuration_hash=opened.configuration_hash,
        service_started_at=opened.service_started_at,
        elapsed_seconds=2,
        ledger_allocated_seconds_before_session=0,
        hard_limit_seconds=36_000,
        observed_at=T2,
    )
    closed = ledger.close_gpu_service_journal(
        service_session_id=opened.service_session_id,
        session_id=opened.session_id,
        service_seconds=2,
        classified_event_seconds=0.5,
        started_at=T0,
        ended_at=T2,
        details={"normal_shutdown": True},
    )
    assert closed.service_seconds == 2
    assert ledger.gpu_summary().total_allocated_seconds == 2
    assert ledger.unresolved_gpu_service_journals() == ()
    assert (
        ledger.latest_gpu_service_journal(opened.service_session_id).state
        is GpuServiceJournalState.CLOSED
    )
    assert (
        ledger.close_gpu_service_journal(
            service_session_id=opened.service_session_id,
            session_id=opened.session_id,
            service_seconds=2,
            classified_event_seconds=0.5,
            started_at=T0,
            ended_at=T2,
            details={"normal_shutdown": True},
        )
        == closed
    )


def test_gpu_service_journal_recovery_charges_through_verified_absence(
    ledger: Ledger,
) -> None:
    ledger.record_gpu_service_observation(
        service_session_id="service-journal-crash",
        state=GpuServiceJournalState.OPENED,
        session_id="pilot-crash",
        configuration_hash=HASH_B,
        service_started_at=T0,
        elapsed_seconds=0,
        ledger_allocated_seconds_before_session=0,
        hard_limit_seconds=36_000,
        observed_at=T0,
    )
    ledger.record_gpu_service_observation(
        service_session_id="service-journal-crash",
        state=GpuServiceJournalState.HEARTBEAT,
        session_id="pilot-crash",
        configuration_hash=HASH_B,
        service_started_at=T0,
        elapsed_seconds=1,
        ledger_allocated_seconds_before_session=0,
        hard_limit_seconds=36_000,
        observed_at=T1,
    )
    recovered = ledger.recover_gpu_service_journal(
        service_session_id="service-journal-crash",
        recovered_at=T3,
        details={"pid_absent": True, "endpoint_absent": True},
    )
    assert recovered.service_seconds == 3
    assert recovered.overhead_seconds == 3
    assert ledger.gpu_summary().total_allocated_seconds == 3
    assert ledger.unresolved_gpu_service_journals() == ()
    assert (
        ledger.latest_gpu_service_journal("service-journal-crash").state
        is GpuServiceJournalState.RECOVERED
    )
    assert (
        ledger.recover_gpu_service_journal(
            service_session_id="service-journal-crash", recovered_at=T3
        )
        == recovered
    )


def test_gpu_service_journal_rejects_mixed_identity_and_parallel_open(
    ledger: Ledger,
) -> None:
    ledger.record_gpu_service_observation(
        service_session_id="exclusive-a",
        state=GpuServiceJournalState.OPENED,
        session_id="pilot",
        configuration_hash=HASH_A,
        service_started_at=T0,
        elapsed_seconds=0,
        ledger_allocated_seconds_before_session=0,
        hard_limit_seconds=36_000,
        observed_at=T0,
    )
    with pytest.raises(DuplicateConflictError, match="another GPU service"):
        ledger.record_gpu_service_observation(
            service_session_id="exclusive-b",
            state=GpuServiceJournalState.OPENED,
            session_id="pilot",
            configuration_hash=HASH_A,
            service_started_at=T1,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T1,
        )
    with pytest.raises(DuplicateConflictError, match="identity changed"):
        ledger.record_gpu_service_observation(
            service_session_id="exclusive-a",
            state=GpuServiceJournalState.HEARTBEAT,
            session_id="different-session",
            configuration_hash=HASH_A,
            service_started_at=T0,
            elapsed_seconds=1,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T1,
        )


def test_storage_preflight_enforces_25gb_occupied_and_5gb_headroom(
    tmp_path: Path, ledger: Ledger
) -> None:
    preflight = StoragePreflight(tmp_path)
    gigabyte = 1_000_000_000

    boundary = preflight.require(
        current_occupied_bytes=24 * gigabyte,
        declared_growth_bytes=500_000_000,
        largest_atomic_temporary_bytes=200_000_000,
        quarantine_allowance_bytes=100_000_000,
        release_staging_bytes=200_000_000,
        filesystem_free_bytes=6 * gigabyte,
    )
    assert boundary.projected_occupied_bytes == 25 * gigabyte
    assert boundary.projected_allocation_free_bytes == 5 * gigabyte
    assert boundary.projected_filesystem_free_bytes == 5 * gigabyte
    assert boundary.effective_projected_headroom_bytes == 5 * gigabyte
    assert boundary.allowed

    first_sample = ledger.record_storage_sample(boundary, phase="phase_1", sampled_at=T0)
    second_sample = ledger.record_storage_sample(boundary, phase="phase_1", sampled_at=T0)
    assert first_sample == second_sample
    assert ledger.count_rows("storage_samples") == 1
    stored = ledger.storage_samples_with_phase_prefix("phase_")
    assert len(stored) == 1
    assert stored[0].sample_id == first_sample
    assert stored[0].filesystem_free_bytes == 6 * gigabyte
    assert stored[0].effective_projected_headroom_bytes == 5 * gigabyte
    assert stored[0].allowed is True
    assert stored[0].violations == ()
    assert ledger.storage_samples_with_phase_prefix("other") == ()
    with pytest.raises(ValueError, match="phase_prefix"):
        ledger.storage_samples_with_phase_prefix("")

    with pytest.raises(StorageBudgetExceeded) as occupied_error:
        preflight.require(
            current_occupied_bytes=25 * gigabyte,
            declared_growth_bytes=1,
            filesystem_free_bytes=10 * gigabyte,
        )
    assert "projected_occupancy_exceeds_limit" in occupied_error.value.report.violations
    assert "projected_allocation_headroom_below_minimum" in occupied_error.value.report.violations

    with pytest.raises(StorageBudgetExceeded) as headroom_error:
        preflight.require(
            current_occupied_bytes=20 * gigabyte,
            filesystem_free_bytes=5 * gigabyte - 1,
        )
    assert "actual_filesystem_headroom_below_minimum" in headroom_error.value.report.violations


def test_storage_measurement_deduplicates_hardlinks(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    alias = tmp_path / "same-inode.bin"
    source.write_bytes(b"x" * 8192)
    try:
        os.link(str(source), str(alias))
    except OSError:
        pytest.skip("hard links are unavailable on this test filesystem")

    preflight = StoragePreflight(tmp_path)
    occupied = preflight.measure_occupied_bytes()
    allocated_once = source.stat().st_blocks * 512 or source.stat().st_size
    assert occupied == allocated_once


def test_phase_one_metadata_families_are_typed_deduplicated_and_append_only(
    tmp_path: Path,
) -> None:
    database = tmp_path / "complete-ledger.sqlite3"
    canary = b"COPYRIGHTED_SOURCE_CANARY_MUST_NOT_ENTER_SQLITE"
    ledger = Ledger(database)
    try:
        blobs = BlobStore(tmp_path / "metadata-blobs", compression=Compression.GZIP)
        artifacts = ArtifactStore(blobs, ledger)
        evidence_artifact = artifacts.put_bytes(
            canary,
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        packet_artifact = artifacts.put_bytes(
            b"opaque packet artifact",
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        response_artifact = artifacts.put_bytes(
            b"opaque model response",
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T1,
        )
        diagnostics_artifact = artifacts.put_bytes(
            b"typed diagnostic codes only",
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T1,
        )
        projection_artifact = artifacts.put_bytes(
            b"validated projection",
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T2,
        )
        visualization_artifact = artifacts.put_bytes(
            b"visualization DTO",
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )

        study = ledger.register_study(
            study_id="conference-study",
            protocol_hash=HASH_A,
            code_manifest_hash=HASH_B,
            configuration_hash=HASH_A,
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        assert (
            ledger.register_study(
                study_id="conference-study",
                protocol_hash=HASH_A,
                code_manifest_hash=HASH_B,
                configuration_hash=HASH_A,
                release_class=ReleaseClass.PUBLIC,
            )
            == study
        )
        job = ledger.create_or_resume_job(
            {"run": "complete-metadata"},
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        study_job = ledger.link_job_to_study(
            study_id=study.study_id, job_id=job.job_id, created_at=T0
        )
        assert study_job.study_id == study.study_id

        snapshot_input = ledger.register_input(
            input_id="snapshot-input",
            study_id=study.study_id,
            input_kind=InputKind.EVIDENCE_SNAPSHOT,
            content_hash=evidence_artifact.content_hash,
            artifact_hash=evidence_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        packet_input = ledger.register_input(
            input_id="packet-input",
            study_id=study.study_id,
            input_kind=InputKind.EVIDENCE_PACKET,
            content_hash=packet_artifact.content_hash,
            artifact_hash=packet_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        snapshot = ledger.register_evidence_snapshot(
            snapshot_id="snapshot-1",
            input_id=snapshot_input.input_id,
            horizon_hash=HASH_A,
            evidence_manifest_hash=HASH_B,
            index_configuration_hash=HASH_A,
            prequery_seal_hash=HASH_B,
            eligible_evidence_count=12,
            created_at=T0,
        )
        assert snapshot.eligible_evidence_count == 12

        attempt = ledger.record_attempt(
            attempt_id="metadata-attempt",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=41,
            created_at=T0,
        )
        ledger.record_gpu_event(
            event_id="metadata-inference",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=1,
            started_at=T0,
            ended_at=T1,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
        )
        model_call = ledger.record_model_call(
            model_call_id="model-call-1",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id="metadata-inference",
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.QUERY_TIME,
            retry_class=RetryClass.BASE,
            model_manifest_hash=HASH_A,
            decoding_manifest_hash=HASH_B,
            request_hash=HASH_A,
            response_artifact_hash=response_artifact.content_hash,
            construction_unit_hash=HASH_B,
            served_context_count=1,
            prompt_tokens=120,
            completion_tokens=40,
            allocated_gpu_seconds=1,
            successful=True,
            created_at=T1,
        )
        assert model_call.allocated_gpu_microseconds == 1_000_000

        validation = ledger.record_validation(
            validation_id="validation-1",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            input_artifact_hash=response_artifact.content_hash,
            validator_manifest_hash=HASH_A,
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.SUPPORTED,
            temporal_status=TemporalValidationStatus.VALID,
            commitment_status=CommitmentCheckStatus.VALID,
            diagnostics_artifact_hash=diagnostics_artifact.content_hash,
            created_at=T2,
        )
        projection = ledger.record_projection(
            projection_id="projection-1",
            job_id=job.job_id,
            validation_id=validation.validation_id,
            snapshot_id=snapshot.snapshot_id,
            packet_input_id=packet_input.input_id,
            condition_id="C2",
            context_hash=HASH_A,
            upper_ontology_hash=HASH_B,
            construction_certificate_hash=HASH_A,
            projection_artifact_hash=projection_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            finalized_at=T2,
        )
        feedback = ledger.record_feedback(
            feedback_id="feedback-resolution-1",
            study_id=study.study_id,
            job_id=job.job_id,
            feedback_kind=FeedbackKind.CONDITION_RESOLUTION,
            action=FeedbackAction.REQUEST_MERGE_SPLIT,
            revision_hash=HASH_A,
            anchor_manifest_hash=HASH_B,
            before_context_hash=HASH_A,
            after_context_hash=HASH_B,
            receiving_condition="C2",
            before_projection_id=projection.projection_id,
            after_projection_id=None,
            resolution_status=FeedbackResolutionStatus.CAPABILITY_LIMITED,
            resolution_artifact_hash=diagnostics_artifact.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )
        assert feedback.action is FeedbackAction.REQUEST_MERGE_SPLIT

        metric = ledger.record_metric(
            metric_id="metric-1",
            study_id=study.study_id,
            projection_id=projection.projection_id,
            unit_hash=HASH_A,
            metric_name="strict_qualified_assertion_f1",
            metric_version_hash=HASH_B,
            status=MetricStatus.VALUE,
            value=0.75,
            numerator=3,
            denominator=4,
            result_artifact_hash=diagnostics_artifact.content_hash,
            created_at=T3,
        )
        assert metric.value == pytest.approx(0.75)

        visualization = ledger.record_visualization(
            visualization_id="visualization-1",
            projection_id=projection.projection_id,
            renderer_configuration_hash=HASH_A,
            semantic_hash=projection.projection_artifact_hash,
            visualization_artifact_hash=visualization_artifact.content_hash,
            layout_seed=91,
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )
        assert visualization.semantic_hash == projection.projection_artifact_hash
        resource = ledger.record_resource_sample(
            sample_id="resource-sample-1",
            job_id=job.job_id,
            gpu_event_id="metadata-inference",
            process_ram_bytes=2_000_000_000,
            system_available_ram_bytes=20_000_000_000,
            gpu_vram_bytes=18_000_000_000,
            project_storage_bytes=12_000_000_000,
            cpu_worker_count=8,
            sampled_at=T1,
        )
        assert resource.cpu_worker_count == 8

        expected_counts = {
            "studies": 1,
            "study_jobs": 1,
            "inputs": 2,
            "evidence_snapshots": 1,
            "model_calls": 1,
            "validations": 1,
            "projections": 1,
            "feedback": 1,
            "metrics": 1,
            "visualizations": 1,
            "resource_samples": 1,
        }
        for table, expected in expected_counts.items():
            assert ledger.count_rows(table) == expected

        prohibited_inline_columns = {
            "payload",
            "prompt",
            "prompt_text",
            "raw_payload",
            "source_text",
            "prose",
            "document_text",
        }
        for table in expected_counts:
            columns = {row[1] for row in ledger._connection.execute(f"PRAGMA table_info({table})")}
            assert columns.isdisjoint(prohibited_inline_columns)

        bypass = sqlite3.connect(str(database))
        try:
            for table in expected_counts:
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    bypass.execute(f"UPDATE {table} SET rowid = rowid")
                bypass.rollback()
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    bypass.execute(f"DELETE FROM {table}")
                bypass.rollback()
        finally:
            bypass.close()
    finally:
        ledger.close()

    assert canary not in database.read_bytes()


def test_v2_ledger_migrates_additively_without_rewriting_existing_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v2-ledger.sqlite3"
    legacy = sqlite3.connect(str(database))
    try:
        legacy.executescript(
            """
            CREATE TABLE schema_metadata (
                schema_version INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                identity_hash TEXT NOT NULL UNIQUE,
                identity_json TEXT NOT NULL,
                release_class TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE job_transitions (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                sequence INTEGER NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                UNIQUE(job_id, sequence)
            );
            """
        )
        legacy.execute(
            "INSERT INTO schema_metadata VALUES (1, ?)",
            ("2026-09-03T11:59:00.000000Z",),
        )
        legacy.execute(
            "INSERT INTO schema_metadata VALUES (2, ?)",
            ("2026-09-03T11:59:30.000000Z",),
        )
        legacy.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
            (HASH_A, HASH_A, '{"legacy":true}', "public", T0),
        )
        legacy.execute(
            "INSERT INTO job_transitions VALUES (1, ?, 0, NULL, 'planned', ?)",
            (HASH_A, T0),
        )
        legacy.commit()
    finally:
        legacy.close()

    with Ledger(database) as migrated:
        assert migrated.schema_versions() == (1, 2, store_module.SCHEMA_VERSION)
        assert migrated.get_job(HASH_A).state is JobState.PLANNED
        expected_new_tables = {
            "studies",
            "study_jobs",
            "inputs",
            "evidence_snapshots",
            "gpu_service_journal",
            "gpu_service_sessions",
            "model_calls",
            "validations",
            "projections",
            "feedback",
            "metrics",
            "visualizations",
            "resource_samples",
        }
        actual_tables = {
            row[0]
            for row in migrated._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert expected_new_tables <= actual_tables

        migrated.register_study(
            study_id="migrated-study",
            protocol_hash=HASH_A,
            code_manifest_hash=HASH_B,
            configuration_hash=HASH_A,
            release_class=ReleaseClass.PUBLIC,
            created_at=T1,
        )
        assert migrated.count_rows("studies") == 1

        bypass = sqlite3.connect(str(database))
        try:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                bypass.execute("UPDATE schema_metadata SET schema_version = 99")
        finally:
            bypass.close()
