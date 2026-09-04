from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import PreQueryInventory
from story_projection_onto.held_out_primary import HeldOutCASReference
from story_projection_onto.held_out_production import (
    HeldOutArtifactResolver,
    HeldOutProductionError,
    HeldOutScheduleState,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    Ledger,
    ReleaseClass,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_typed_held_out_cas_resolves_physical_and_logical_hashes(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        resolver = HeldOutArtifactResolver(
            ArtifactStore(
                BlobStore(tmp_path / "cas", compression=Compression.GZIP),
                ledger,
            )
        )
        inventory = PreQueryInventory(
            inventory_id="TEST-ONLY-empty-c2",
            snapshot_hash="a" * 64,
            recorded_at=NOW,
        )
        reference = resolver.persist_record(
            inventory,
            object_kind="pre_query_inventory",
            release_class=ReleaseClass.RESTRICTED,
            created_at=NOW,
        )
        assert (
            resolver.resolve_record(
                reference,
                PreQueryInventory,
                required_release=ReleaseClass.RESTRICTED,
            )
            == inventory
        )

        changed = HeldOutCASReference(
            **reference.model_dump(
                mode="python",
                exclude={"content_hash", "logical_content_hash"},
            ),
            logical_content_hash="b" * 64,
        )
        with pytest.raises(HeldOutProductionError, match="logical object hash"):
            resolver.resolve_record(changed, PreQueryInventory)
    finally:
        ledger.close()


def test_durable_schedule_refuses_unmetered_or_duplicate_service_identity() -> None:
    with pytest.raises(ValidationError, match="service-start count"):
        HeldOutScheduleState(
            call_manifest_hash="a" * 64,
            development_execution_result_hash="b" * 64,
            development_predecessor_allocated_gpu_seconds=100,
            remaining_registered_p95_seconds=200,
            consumed_primary_service_start_count=1,
            updated_at=NOW,
        )
