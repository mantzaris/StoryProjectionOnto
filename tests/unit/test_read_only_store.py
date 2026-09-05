from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from story_projection_onto.store import (
    ArtifactIntegrityError,
    ArtifactStore,
    BlobStore,
    Compression,
    GpuAllocationJournalState,
    GpuEventKind,
    GpuServiceJournalState,
    Ledger,
    ReadOnlyArtifactStore,
    ReadOnlyBlobStore,
    ReadOnlyLedger,
    ReleaseClass,
    ReleaseViolationError,
    StorageBudget,
    StorageReport,
    StoreError,
)

T0 = "2026-09-03T12:00:00Z"
T1 = "2026-09-03T12:00:01Z"


def _file_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _storage_report() -> StorageReport:
    return StorageReport(
        current_occupied_bytes=100,
        declared_growth_bytes=20,
        largest_atomic_temporary_bytes=10,
        quarantine_allowance_bytes=5,
        release_staging_bytes=5,
        projected_occupied_bytes=140,
        filesystem_free_bytes=10_000,
        projected_allocation_free_bytes=29_999_999_860,
        projected_filesystem_free_bytes=9_960,
        effective_projected_headroom_bytes=9_960,
        budget=StorageBudget(),
        allowed=True,
        violations=(),
    )


@pytest.mark.parametrize("compression", [Compression.GZIP, Compression.ZSTD])
def test_read_only_artifact_view_preserves_ledger_and_cas_across_all_queries(
    tmp_path: Path,
    compression: Compression,
) -> None:
    database = tmp_path / "study.sqlite3"
    cas_root = tmp_path / "cas"
    public_payload = b'{"scope":"public"}\n'
    restricted_payload = b'{"scope":"restricted"}\n'

    with Ledger(database) as writable:
        artifacts = ArtifactStore(
            BlobStore(cas_root, compression=compression),
            writable,
        )
        public_record = artifacts.put_bytes(
            public_payload,
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        restricted_record = artifacts.put_bytes(
            restricted_payload,
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=T1,
        )
        writable.record_gpu_allocation_observation(
            allocation_id="still-open-allocation",
            state=GpuAllocationJournalState.OPENED,
            intended_event_kind=GpuEventKind.INFERENCE,
            elapsed_seconds=0,
            maximum_seconds=10,
            observed_at=T0,
        )
        expected_allocation_journal = writable.gpu_allocation_journal_records()
        expected_unresolved_allocations = writable.unresolved_gpu_allocations()
        writable.record_gpu_service_observation(
            service_session_id="still-open-service",
            state=GpuServiceJournalState.OPENED,
            session_id="server-session",
            configuration_hash="a" * 64,
            service_started_at=T0,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T0,
        )
        expected_service_journal = writable.gpu_service_journal_records()
        expected_unresolved_services = writable.unresolved_gpu_service_journals()
        writable.record_gpu_event(
            event_id="completed-warmup",
            event_kind=GpuEventKind.WARM_UP,
            allocated_seconds=2,
            started_at=T0,
            ended_at=T1,
            succeeded=True,
        )
        writable.record_gpu_service_session(
            service_session_id="completed-service",
            session_id="earlier-server-session",
            service_seconds=3,
            classified_event_seconds=2,
            started_at=T0,
            ended_at=T1,
        )
        expected_service_sessions = writable.gpu_service_sessions()
        expected_gpu_events = writable.gpu_events()
        expected_gpu_summary = writable.gpu_summary()
        writable.record_storage_sample(_storage_report(), phase="phase_6/final", sampled_at=T1)
        expected_storage = writable.storage_samples()

    assert not Path(str(database) + "-wal").exists()
    assert not Path(str(database) + "-journal").exists()
    ledger_before = hashlib.sha256(database.read_bytes()).hexdigest()
    ledger_size_before = database.stat().st_size
    ledger_mtime_before = database.stat().st_mtime_ns
    cas_before = _file_inventory(cas_root)
    directory_entries_before = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
    )

    with ReadOnlyArtifactStore.from_paths(
        blob_root=cas_root,
        ledger_path=database,
        compression=compression,
    ) as read_only:
        assert read_only.ledger.get_artifact(public_record.content_hash) == public_record
        assert read_only.blobs.read_bytes(public_record) == public_payload
        with pytest.raises(ReleaseViolationError):
            read_only.blobs.read_bytes(restricted_record)
        assert (
            read_only.blobs.read_bytes(restricted_record, allow_restricted=True)
            == restricted_payload
        )
        with pytest.raises(ReleaseViolationError):
            read_only.ledger.get_artifact(
                restricted_record.content_hash,
                for_public_release=True,
            )
        assert (
            read_only.ledger.gpu_allocation_journal_records()
            == expected_allocation_journal
        )
        assert (
            read_only.ledger.unresolved_gpu_allocations()
            == expected_unresolved_allocations
        )
        assert read_only.ledger.gpu_service_journal_records() == expected_service_journal
        assert (
            read_only.ledger.unresolved_gpu_service_journals()
            == expected_unresolved_services
        )
        assert read_only.ledger.gpu_service_sessions() == expected_service_sessions
        assert read_only.ledger.gpu_events() == expected_gpu_events
        assert read_only.ledger.gpu_summary() == expected_gpu_summary
        assert read_only.ledger.storage_samples() == expected_storage
        assert read_only.ledger.storage_samples_with_phase_prefix("phase_6") == expected_storage
        assert read_only.ledger.count_rows("artifacts") == 2
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            read_only.ledger._connection.execute(
                "INSERT INTO schema_metadata VALUES (999, 'forbidden')"
            )

    assert read_only.ledger.closed
    with pytest.raises(StoreError, match="closed"):
        read_only.ledger.count_rows("artifacts")
    assert hashlib.sha256(database.read_bytes()).hexdigest() == ledger_before
    assert database.stat().st_size == ledger_size_before
    assert database.stat().st_mtime_ns == ledger_mtime_before
    assert _file_inventory(cas_root) == cas_before
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == (
        directory_entries_before
    )
    assert not Path(str(database) + "-wal").exists()
    assert not Path(str(database) + "-journal").exists()
    assert not Path(str(database) + "-shm").exists()


def test_read_only_blob_view_rejects_noncanonical_metadata_and_symlinks(
    tmp_path: Path,
) -> None:
    database = tmp_path / "study.sqlite3"
    cas_root = tmp_path / "cas"
    payload = b"immutable payload"
    with Ledger(database) as writable:
        record = ArtifactStore(
            BlobStore(cas_root, compression=Compression.GZIP), writable
        ).put_bytes(
            payload,
            media_type="application/octet-stream",
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )

    blobs = ReadOnlyBlobStore(cas_root, compression=Compression.GZIP)
    with pytest.raises(ArtifactIntegrityError, match="content address"):
        blobs.read_bytes(replace(record, relative_path="wrong/path.jsonl.gz"))

    artifact_path = cas_root / record.relative_path
    encoded_copy = tmp_path / "outside-cas.gz"
    encoded_copy.write_bytes(artifact_path.read_bytes())
    artifact_path.unlink()
    artifact_path.symlink_to(encoded_copy)
    with pytest.raises(ArtifactIntegrityError, match="safely open"):
        blobs.read_bytes(record)

    cas_alias = tmp_path / "cas-alias"
    cas_alias.symlink_to(cas_root, target_is_directory=True)
    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        ReadOnlyBlobStore(cas_alias, compression=Compression.GZIP)

    ledger_alias = tmp_path / "ledger-alias.sqlite3"
    ledger_alias.symlink_to(database)
    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        ReadOnlyLedger(ledger_alias)


def test_read_only_ledger_rejects_uncheckpointed_wal_without_touching_it(
    tmp_path: Path,
) -> None:
    database = tmp_path / "study.sqlite3"
    with Ledger(database):
        pass
    wal = Path(str(database) + "-wal")
    wal.write_bytes(b"not-a-valid-but-nonempty-wal")
    database_before = hashlib.sha256(database.read_bytes()).hexdigest()
    wal_before = wal.read_bytes()

    with pytest.raises(ArtifactIntegrityError, match="closed and checkpointed"):
        ReadOnlyLedger(database)

    assert hashlib.sha256(database.read_bytes()).hexdigest() == database_before
    assert wal.read_bytes() == wal_before
    assert os.path.lexists(wal)


def test_read_only_ledger_accepts_valid_migration_history(tmp_path: Path) -> None:
    database = tmp_path / "migrated-study.sqlite3"
    with Ledger(database):
        pass
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "INSERT INTO schema_metadata(schema_version, created_at) VALUES (?, ?)",
            (5, T0),
        )
        connection.commit()
    finally:
        connection.close()

    with ReadOnlyLedger(database) as read_only:
        assert read_only.count_rows("schema_metadata") == 2


def test_read_only_ledger_rejects_symlinked_sidecar(tmp_path: Path) -> None:
    database = tmp_path / "study.sqlite3"
    with Ledger(database):
        pass
    outside = tmp_path / "outside-wal"
    outside.write_bytes(b"")
    wal = Path(str(database) + "-wal")
    wal.symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError, match=r"sidecar.*symbolic link"):
        ReadOnlyLedger(database)

    assert wal.is_symlink()
    assert outside.read_bytes() == b""
