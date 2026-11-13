from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.phase5_capture import (
    ResearcherTraceCaptureBinding,
    ResearcherTraceCaptureServerManifest,
    materialize_capture_server_manifest,
    prepare_researcher_trace_capture_app,
)
from story_projection_onto.phase5_sources import Phase5SourceProductionError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_capture_factory_rejects_noncanonical_restricted_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    canonical = repository / "artifacts/restricted"
    canonical.mkdir(parents=True)
    arbitrary = repository / "arbitrary"
    arbitrary.mkdir()

    with pytest.raises(Phase5SourceProductionError, match="canonical repository"):
        prepare_researcher_trace_capture_app(
            repository=repository,
            restricted_root=arbitrary,
            primary_results_gate_path=arbitrary / "gate.json",
            held_out_journal_root=arbitrary / "journal",
            capture_root=arbitrary / "capture",
            protocol=load_feedback_protocol(),
            artifacts=None,
        )


def test_capture_materializer_rejects_symlinked_destination(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    restricted = repository / "artifacts/restricted"
    restricted.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    capture = restricted / "capture"
    capture.symlink_to(outside, target_is_directory=True)
    manifest = ResearcherTraceCaptureServerManifest(
        protocol_hash=digest("protocol"),
        primary_results_gate_hash=digest("gate"),
        held_out_execution_manifest_hash=digest("execution"),
        bindings=tuple(
            ResearcherTraceCaptureBinding(
                episode_id=f"trace-{index}",
                context_id=f"context-{index}",
                projection_id=f"projection-{index}",
                projection_hash=digest(f"projection-{index}"),
                bundle_hash=digest(f"bundle-{index}"),
                source_export_hash=digest(f"export-{index}"),
            )
            for index in range(3)
        ),
        opened_at=datetime(2026, 9, 4, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(Phase5SourceProductionError, match="symlinked"):
        materialize_capture_server_manifest(
            repository=repository,
            restricted_root=restricted,
            capture_root=capture,
            manifest=manifest,
        )
