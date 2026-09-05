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
    PrequeryBarrierRecord,
    QueryAccessRecord,
    ReleaseClass,
    ReleaseViolationError,
    RetryClass,
    SemanticAssessmentScope,
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


@pytest.mark.parametrize(
    "condition",
    ["C2", "A-FixedSelect", "A-NoContext", "feedback"],
)
def test_only_typed_query_blind_prebuild_can_skip_query_reveal(
    ledger: Ledger, condition: str
) -> None:
    job = ledger.create_or_resume_job(
        {
            "condition": condition,
            "lifecycle_kind": "query_time_generation",
            "run": f"query-time-{condition}",
        },
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    ledger.transition_job(job.job_id, JobState.PREQUERY_SEALED, occurred_at=T1)
    with pytest.raises(InvalidTransitionError, match="query-blind prebuild"):
        ledger.transition_job(job.job_id, JobState.GENERATED, occurred_at=T2)


def test_query_blind_prebuild_branch_and_lifecycle_replay_are_append_only(
    ledger: Ledger,
) -> None:
    job = ledger.create_or_resume_job(
        {
            "condition": "C1",
            "lifecycle_kind": "query_blind_prebuild",
            "run": "query-blind-c1",
        },
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    first = ledger.advance_job_lifecycle(
        job.job_id,
        (
            (JobState.PREQUERY_SEALED, T1),
            (JobState.GENERATED, T2),
            (JobState.VALIDATED, T3),
        ),
    )
    replay = ledger.advance_job_lifecycle(
        job.job_id,
        (
            (JobState.PREQUERY_SEALED, "2030-01-01T00:00:00Z"),
            (JobState.GENERATED, "2030-01-01T00:00:01Z"),
        ),
    )
    assert replay == first
    assert [item.to_state for item in replay] == [
        JobState.PLANNED,
        JobState.PREQUERY_SEALED,
        JobState.GENERATED,
        JobState.VALIDATED,
    ]


def test_only_exact_frozen_v3_failure_can_use_legacy_query_blind_retry_branch(
    ledger: Ledger,
) -> None:
    identity = {
        "run_id": "fallback-qwen3-8b-awq-development-v3",
        "call_id": "fallback-c1-01",
        "plan_hash": "24bd99189fdf5157d6de1c3c13edaa0a86749c97b1b4957b220e7a62de9aa200",
    }
    job = ledger.create_or_resume_job(
        identity,
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    assert job.job_id == "f5646ba427f98e76afc07fabcb6c53f65d73d255856493ac94bf9d0c642f444f"
    attempt_id = "fallback-qwen3-8b-awq-development-v3-fallback-c1-01-attempt"
    model_call_id = "fallback-qwen3-8b-awq-development-v3-fallback-c1-01"
    event_id = f"{model_call_id}-gpu"
    request_hash = "1cc73c5525e096a4df830892f37cdc8062899363a0b75835bb2f04b3a14a0d44"
    ledger.record_attempt(
        attempt_id=attempt_id,
        job_id=job.job_id,
        attempt_kind=AttemptKind.BASE,
        input_hash=request_hash,
        config_hash=HASH_B,
        seed=0,
        created_at=T1,
    )
    ledger.record_gpu_event(
        event_id=event_id,
        event_kind=GpuEventKind.FAILURE,
        allocated_seconds=0.852878,
        started_at=T1,
        ended_at="2026-09-03T12:00:01.852878Z",
        succeeded=False,
        job_id=job.job_id,
        attempt_id=attempt_id,
        details={
            "reserve_call_class": "reserve_long",
            "reserve_reservation_id": (
                "fallback-qwen3-8b-awq-development-v3:fallback-c1-01"
            ),
        },
    )
    ledger.record_model_call(
        model_call_id=model_call_id,
        job_id=job.job_id,
        attempt_id=attempt_id,
        gpu_event_id=event_id,
        backend=ModelBackend.VLLM_GPU,
        call_role=ModelCallRole.PILOT,
        retry_class=RetryClass.LONG,
        model_manifest_hash=HASH_A,
        decoding_manifest_hash=HASH_B,
        request_hash=request_hash,
        response_artifact_hash=None,
        construction_unit_hash=(
            "c91e2eeb87d9f9c05713396b3403ddab702573da73c14893eb7ed0c7158e6317"
        ),
        served_context_count=1,
        prompt_tokens=0,
        completion_tokens=0,
        allocated_gpu_seconds=0.852878,
        successful=False,
        created_at=T1,
    )
    ledger.record_failure(
        attempt_id=attempt_id,
        failure_kind=FailureKind.SERVICE,
        message="Fallback micro-pilot model call failed",
        details={
            "call_id": "fallback-c1-01",
            "exception_type": "RuntimeTransportError",
        },
        occurred_at=T1,
    )
    assert ledger.is_frozen_legacy_fallback_c1_retry_prebuild(job.job_id)
    ledger.transition_job(job.job_id, JobState.PREQUERY_SEALED, occurred_at=T2)
    ledger.transition_job(
        job.job_id,
        JobState.GENERATED,
        occurred_at=T3,
        frozen_legacy_fallback_c1_retry=True,
    )
    assert ledger.get_job(job.job_id).state is JobState.GENERATED

    generic = ledger.create_or_resume_job(
        {**identity, "run_id": "fallback-qwen3-8b-awq-development-v3-mutated"},
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    ledger.transition_job(generic.job_id, JobState.PREQUERY_SEALED, occurred_at=T2)
    with pytest.raises(InvalidTransitionError, match="query-blind prebuild"):
        ledger.transition_job(
            generic.job_id,
            JobState.GENERATED,
            occurred_at=T3,
            frozen_legacy_fallback_c1_retry=True,
        )
    with pytest.raises(InvalidTransitionError, match="requested replay prefix"):
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, T1),
                (JobState.QUERY_REVEALED, T2),
            ),
        )


def test_query_reveal_must_strictly_follow_prequery_seal(ledger: Ledger) -> None:
    job = ledger.create_or_resume_job(
        {
            "condition": "C2",
            "lifecycle_kind": "query_time_projection",
            "run": "strict-query-boundary",
        },
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    ledger.transition_job(job.job_id, JobState.PREQUERY_SEALED, occurred_at=T1)

    with pytest.raises(InvalidTransitionError, match="strictly follow"):
        ledger.transition_job(job.job_id, JobState.QUERY_REVEALED, occurred_at=T1)

    revealed = ledger.transition_job(
        job.job_id, JobState.QUERY_REVEALED, occurred_at=T2
    )
    assert revealed.occurred_at == "2026-09-03T12:00:02.000000Z"


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


def test_query_access_event_id_lookup_is_exact_unknown_safe_and_unique(
    tmp_path: Path,
    ledger: Ledger,
) -> None:
    blobs = BlobStore(tmp_path / "query-lookup-blobs", compression=Compression.GZIP)
    barrier_artifact = blobs.put_bytes(
        b'{"barrier":"query-lookup"}\n',
        media_type="application/vnd.story-projection.prequery-barrier+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=T0,
    )
    barrier = ledger.persist_prequery_barrier(
        PrequeryBarrierRecord(
            barrier_hash=HASH_A,
            barrier_id="query-lookup-barrier",
            execution_id="query-lookup-execution",
            execution_manifest_hash=HASH_B,
            barrier_artifact_hash=barrier_artifact.content_hash,
            preparation_count=1,
            sealed_at=T0,
            persisted_at=T0,
            release_class=ReleaseClass.PUBLIC,
        ),
        barrier_artifact=barrier_artifact,
    )
    query_payload_artifact = blobs.put_bytes(
        b'{"query":"exact"}\n',
        media_type="application/vnd.story-projection.query-reveal+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=T1,
    )
    access_event_artifact = blobs.put_bytes(
        b'{"access":"first"}\n',
        media_type="application/vnd.story-projection.query-access+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=T1,
    )
    persisted = ledger.persist_query_access(
        QueryAccessRecord(
            access_event_hash="c" * 64,
            access_event_id="query-access-exact-01",
            execution_id=barrier.execution_id,
            query_context_hash="d" * 64,
            model_visible_query_hash="e" * 64,
            snapshot_hash="f" * 64,
            stage_manifest_hash="1" * 64,
            query_artifact_hash="2" * 64,
            prequery_barrier_hash=barrier.barrier_hash,
            packet_hash=None,
            query_payload_artifact_hash=query_payload_artifact.content_hash,
            access_event_artifact_hash=access_event_artifact.content_hash,
            registered_revealed_at=T0,
            accessed_at=T1,
            release_class=ReleaseClass.PUBLIC,
        ),
        query_payload_artifact=query_payload_artifact,
        access_event_artifact=access_event_artifact,
    )

    assert ledger.get_query_access_by_event_id("query-access-exact-01") == persisted
    assert ledger.get_query_access(persisted.access_event_hash) == persisted
    with pytest.raises(KeyError, match="unknown query access event"):
        ledger.get_query_access_by_event_id("query-access-exact")
    with pytest.raises(KeyError, match="unknown query access event"):
        ledger.get_query_access_by_event_id("query-access-exact-01 ")
    with pytest.raises(ValueError):
        ledger.get_query_access_by_event_id("")

    conflicting_access_artifact = blobs.put_bytes(
        b'{"access":"conflicting-duplicate-id"}\n',
        media_type="application/vnd.story-projection.query-access+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=T2,
    )
    with pytest.raises(DuplicateConflictError, match="immutable event"):
        ledger.persist_query_access(
            QueryAccessRecord(
                access_event_hash="3" * 64,
                access_event_id=persisted.access_event_id,
                execution_id=barrier.execution_id,
                query_context_hash="4" * 64,
                model_visible_query_hash="5" * 64,
                snapshot_hash=persisted.snapshot_hash,
                stage_manifest_hash="6" * 64,
                query_artifact_hash="7" * 64,
                prequery_barrier_hash=barrier.barrier_hash,
                packet_hash=None,
                query_payload_artifact_hash=query_payload_artifact.content_hash,
                access_event_artifact_hash=conflicting_access_artifact.content_hash,
                registered_revealed_at=T0,
                accessed_at=T2,
                release_class=ReleaseClass.PUBLIC,
            ),
            query_payload_artifact=query_payload_artifact,
            access_event_artifact=conflicting_access_artifact,
        )

    assert ledger.count_rows("query_access_events") == 1
    assert ledger.get_query_access_by_event_id(persisted.access_event_id) == persisted


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
    assert [event.event_id for event in ledger.gpu_events()] == [
        "load",
        "failed",
        "restart",
    ]
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


def test_storage_preflight_allows_missing_future_path_on_quota_device(
    tmp_path: Path,
) -> None:
    future_output = tmp_path / "future" / "nested" / "result.json"

    preflight = StoragePreflight(
        tmp_path,
        controlled_paths=(future_output,),
    )

    assert preflight.controlled_paths == (future_output,)
    assert preflight.measure_occupied_bytes() == 0


def test_storage_preflight_rejects_controlled_ancestor_on_another_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign_ancestor = tmp_path / "external-mount"
    foreign_ancestor.mkdir()
    controlled = foreign_ancestor / "future" / "result.json"
    original_stat = Path.stat

    def device_overridden_stat(path: Path, *args: object, **kwargs: object):
        observed = original_stat(path, *args, **kwargs)
        if path == foreign_ancestor:
            fields = list(observed)
            fields[2] = observed.st_dev + 1
            return os.stat_result(fields)
        return observed

    monkeypatch.setattr(Path, "stat", device_overridden_stat)

    with pytest.raises(ValueError, match="different device from quota_root"):
        StoragePreflight(tmp_path, controlled_paths=(controlled,))


def test_storage_override_cannot_bypass_late_cross_device_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controlled = tmp_path / "future" / "result.json"
    preflight = StoragePreflight(tmp_path, controlled_paths=(controlled,))
    foreign_ancestor = controlled.parent
    foreign_ancestor.mkdir()
    original_stat = Path.stat

    def device_overridden_stat(path: Path, *args: object, **kwargs: object):
        observed = original_stat(path, *args, **kwargs)
        if path == foreign_ancestor:
            fields = list(observed)
            fields[2] = observed.st_dev + 1
            return os.stat_result(fields)
        return observed

    monkeypatch.setattr(Path, "stat", device_overridden_stat)

    with pytest.raises(ValueError, match="different device from quota_root"):
        preflight.check(
            current_occupied_bytes=0,
            filesystem_free_bytes=30_000_000_000,
        )


def test_storage_measurement_fails_closed_on_traversal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = StoragePreflight(tmp_path)

    def failed_walk(
        _root: Path,
        *,
        followlinks: bool,
        onerror,
    ) -> tuple[()]:
        assert followlinks is False
        assert onerror is not None
        onerror(PermissionError("synthetic unreadable subtree"))
        return ()

    monkeypatch.setattr(store_module.os, "walk", failed_walk)

    with pytest.raises(
        OSError,
        match="cannot completely traverse project-controlled storage",
    ) as error:
        preflight.measure_occupied_bytes()
    assert isinstance(error.value.__cause__, PermissionError)


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
            semantic_assessment_scope=(
                SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER
            ),
            diagnostics_artifact_hash=diagnostics_artifact.content_hash,
            created_at=T2,
        )
        assert validation.semantic_assessment_scope is (
            SemanticAssessmentScope.POSTHOC_SCORER_OR_REVIEWER
        )
        with pytest.raises(ValueError, match="runtime structural-only"):
            ledger.record_validation(
                validation_id="invalid-runtime-semantic-claim",
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
                input_artifact_hash=response_artifact.content_hash,
                validator_manifest_hash=HASH_A,
                validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.SUPPORTED,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            created_at=T2,
            )
        with pytest.raises(ValueError, match="migration provenance"):
            ledger.record_validation(
                validation_id="invalid-new-legacy-scope",
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
                input_artifact_hash=response_artifact.content_hash,
                validator_manifest_hash=HASH_A,
                validation_status=ValidationStatus.ACCEPTED,
                evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
                temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
                commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
                semantic_assessment_scope=SemanticAssessmentScope.LEGACY_UNSPECIFIED,
                created_at=T2,
            )
        projection_validation = ledger.record_validation(
            validation_id="projection-validation-1",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            input_artifact_hash=response_artifact.content_hash,
            validator_manifest_hash=HASH_A,
            validation_status=ValidationStatus.ACCEPTED,
            evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            diagnostics_artifact_hash=diagnostics_artifact.content_hash,
            created_at=T2,
        )
        ledger.advance_job_lifecycle(
            job.job_id,
            (
                (JobState.PREQUERY_SEALED, T0),
                (JobState.QUERY_REVEALED, T1),
                (JobState.GENERATED, T1),
                (JobState.VALIDATED, T2),
                (JobState.FINALIZED, T2),
            ),
        )
        projection = ledger.record_projection(
            projection_id="projection-1",
            job_id=job.job_id,
            validation_id=projection_validation.validation_id,
            snapshot_id=snapshot.snapshot_id,
            packet_input_id=packet_input.input_id,
            condition_id="C2",
            context_hash=HASH_A,
            upper_ontology_hash=HASH_B,
            construction_certificate_hash=HASH_A,
            projection_artifact_hash=projection_artifact.content_hash,
            projection_semantic_hash=HASH_B,
            release_class=ReleaseClass.PUBLIC,
            finalized_at=T2,
        )
        owner, resolved_projection = ledger.resolve_projection_artifact_owner(
            projection_artifact_hash=projection.projection_artifact_hash,
            projection_semantic_hash=projection.projection_semantic_hash,
            condition_id=projection.condition_id,
            context_hash=projection.context_hash,
        )
        assert owner == study
        assert resolved_projection == projection
        with pytest.raises(ValueError, match="accepted runtime structural-only"):
            ledger.record_projection(
                projection_id="projection-with-posthoc-validation",
                job_id=job.job_id,
                validation_id=validation.validation_id,
                snapshot_id=snapshot.snapshot_id,
                packet_input_id=packet_input.input_id,
                condition_id="C2",
                context_hash=HASH_A,
                upper_ontology_hash=HASH_B,
                construction_certificate_hash=HASH_A,
                projection_artifact_hash=projection_artifact.content_hash,
                projection_semantic_hash=HASH_B,
                release_class=ReleaseClass.PUBLIC,
                finalized_at=T2,
            )
        with pytest.raises(ValueError, match="after projection"):
            ledger.record_feedback(
                feedback_id="feedback-applied-without-projection",
                study_id=study.study_id,
                job_id=job.job_id,
                feedback_kind=FeedbackKind.CONDITION_RESOLUTION,
                action=FeedbackAction.REFINE_CONTEXT,
                revision_hash=HASH_A,
                anchor_manifest_hash=HASH_B,
                before_context_hash=HASH_A,
                after_context_hash=HASH_B,
                receiving_condition="C2",
                before_projection_id=projection.projection_id,
                after_projection_id=None,
                resolution_status=FeedbackResolutionStatus.APPLIED,
                resolution_artifact_hash=diagnostics_artifact.content_hash,
                release_class=ReleaseClass.PUBLIC,
                created_at=T3,
            )
        with pytest.raises(ValueError, match="status/artifact"):
            ledger.record_feedback(
                feedback_id="feedback-resolution-without-artifact",
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
                resolution_artifact_hash=None,
                release_class=ReleaseClass.PUBLIC,
                created_at=T3,
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
            job_id=job.job_id,
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
            semantic_hash=HASH_B,
            visualization_artifact_hash=visualization_artifact.content_hash,
            layout_seed=91,
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )
        assert visualization.semantic_hash == projection.projection_semantic_hash
        with pytest.raises(ValueError, match="exact projection content"):
            ledger.record_visualization(
                visualization_id="visualization-wrong-semantic-hash",
                projection_id=projection.projection_id,
                renderer_configuration_hash=HASH_A,
                semantic_hash=HASH_A,
                visualization_artifact_hash=visualization_artifact.content_hash,
                layout_seed=91,
                release_class=ReleaseClass.PUBLIC,
                created_at=T3,
            )
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
            "validations": 2,
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
            CREATE TABLE validations (
                validation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                input_artifact_hash TEXT NOT NULL,
                validator_manifest_hash TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                evidence_support_status TEXT NOT NULL,
                temporal_status TEXT NOT NULL,
                commitment_status TEXT NOT NULL,
                diagnostics_artifact_hash TEXT,
                parent_validation_id TEXT,
                repair_attempt_id TEXT,
                created_at TEXT NOT NULL
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
        legacy.execute(
            """INSERT INTO validations VALUES (
                   'legacy-validation', ?, 'legacy-attempt', ?, ?,
                   'accepted', 'supported', 'valid', 'valid',
                   NULL, NULL, NULL, ?
               )""",
            (HASH_A, HASH_A, HASH_B, T0),
        )
        legacy.commit()
    finally:
        legacy.close()

    with Ledger(database) as migrated:
        assert migrated.schema_versions() == (1, 2, store_module.SCHEMA_VERSION)
        assert migrated.get_job(HASH_A).state is JobState.PLANNED
        assert migrated.get_validation(
            "legacy-validation"
        ).semantic_assessment_scope is SemanticAssessmentScope.LEGACY_UNSPECIFIED
        expected_new_tables = {
            "studies",
            "study_jobs",
            "inputs",
            "evidence_snapshots",
            "prequery_barriers",
            "query_access_events",
            "packet_materialization_events",
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
