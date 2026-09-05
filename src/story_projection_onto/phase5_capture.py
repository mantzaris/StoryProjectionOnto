"""Capture-only UI factory for the three registered researcher traces."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
)
from story_projection_onto.feedback_provenance import (
    AppendOnlyResearcherTraceCaptureStore,
)
from story_projection_onto.feedback_runtime import FeedbackProtocolConfiguration
from story_projection_onto.phase5_sources import (
    Phase5SourceProductionError,
    _build_export,
    _episode_boundary,
    _selected_openings,
    load_closed_held_out_sources,
)
from story_projection_onto.store import ArtifactStore
from story_projection_onto.ui import (
    LocalUiRepository,
    VisualizationBundle,
    VisualizationContentScope,
    build_visualization_bundle,
    create_app,
)

CAPTURE_SERVER_MANIFEST_FILE = "capture_server_manifest.json"


class ResearcherTraceCaptureBinding(ImmutableRecord):
    episode_id: str
    context_id: str
    projection_id: str
    projection_hash: Sha256Digest
    bundle_hash: Sha256Digest
    source_export_hash: Sha256Digest


class ResearcherTraceCaptureServerManifest(ImmutableRecord):
    protocol_hash: Sha256Digest
    primary_results_gate_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    bindings: tuple[
        ResearcherTraceCaptureBinding,
        ResearcherTraceCaptureBinding,
        ResearcherTraceCaptureBinding,
    ]
    opened_at: AwareDatetime
    endpoint: Literal["/api/revisions"] = "/api/revisions"
    capture_only: Literal[True] = True
    revision_runner_configured: Literal[False] = False
    model_service_called: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_three_traces(self) -> Self:
        if (
            len({item.episode_id for item in self.bindings}) != 3
            or len({item.projection_id for item in self.bindings}) != 3
        ):
            raise ValueError("capture server requires three distinct traces and projections")
        return self


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5SourceProductionError(
                f"symlinked trace capture path is forbidden: {path}"
            )
        if current.parent == current:
            return
        current = current.parent


def _canonical_restricted_root(repository: Path, restricted_root: Path) -> Path:
    repository = repository.resolve(strict=True)
    _assert_no_symlink_chain(restricted_root)
    restricted_root = restricted_root.resolve(strict=True)
    expected = repository / "artifacts/restricted"
    _assert_no_symlink_chain(expected)
    if restricted_root != expected.resolve(strict=True):
        raise Phase5SourceProductionError(
            "restricted root is not the canonical repository artifacts/restricted root"
        )
    return restricted_root


def _restricted_descendant(root: Path, path: Path, label: str) -> Path:
    path = path.absolute()
    _assert_no_symlink_chain(path)
    try:
        if not path.resolve(strict=path.exists()).is_relative_to(root):
            raise Phase5SourceProductionError(f"{label} is outside restricted storage")
    except OSError as error:
        raise Phase5SourceProductionError(f"cannot resolve {label}") from error
    if path.resolve(strict=path.exists()) == root:
        raise Phase5SourceProductionError(f"{label} must be below restricted storage")
    return path


def _append_exact(path: Path, value: ImmutableRecord) -> bool:
    payload = (value.to_canonical_json() + "\n").encode("utf-8")
    _assert_no_symlink_chain(path)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise Phase5SourceProductionError("append-only trace capture manifest changed")
        return False
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise Phase5SourceProductionError(
                    "concurrent trace capture manifest changed"
                ) from None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def prepare_researcher_trace_capture_app(
    *,
    repository: Path,
    restricted_root: Path,
    primary_results_gate_path: Path,
    held_out_journal_root: Path,
    capture_root: Path,
    protocol: FeedbackProtocolConfiguration,
    artifacts: ArtifactStore,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
):
    """Load the exact three seed-1 C2 parents and construct a no-runner app."""

    repository = repository.resolve(strict=True)
    restricted_root = _canonical_restricted_root(repository, restricted_root)

    gate_path = _restricted_descendant(
        restricted_root, primary_results_gate_path, "primary results gate"
    )
    journal_root = _restricted_descendant(
        restricted_root, held_out_journal_root, "held-out journal"
    )
    capture_root = _restricted_descendant(
        restricted_root, capture_root, "researcher trace capture root"
    )
    closed = load_closed_held_out_sources(
        primary_results_gate_path=gate_path,
        held_out_journal_root=journal_root,
    )
    selected = _selected_openings(protocol, closed)
    opened_at = clock()
    if opened_at.tzinfo is None or opened_at.utcoffset() is None:
        raise Phase5SourceProductionError("trace capture clock must be timezone-aware")
    if opened_at <= closed.gate.frozen_at:
        raise Phase5SourceProductionError("trace capture must follow the frozen primary results")
    bundles: list[VisualizationBundle] = []
    bindings: list[ResearcherTraceCaptureBinding] = []
    for slot in protocol.researcher_trace_slots:
        boundary = _episode_boundary(
            episode_id=slot.episode_id,
            context_id=slot.context_id,
            selected=selected,
            repository=repository,
            artifacts=artifacts,
        )
        resolved = _build_export(
            boundary=boundary,
            condition=ConditionName.C2_LLM_QUERY,
            closed=closed,
            journal_root=journal_root,
            artifacts=artifacts,
            exported_at=opened_at,
            expected_frozen_seed=protocol.llm_seed,
        )
        bundle = build_visualization_bundle(
            resolved.projection,
            boundary.opening.query_context,
            boundary.packet,
            content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        )
        bundles.append(bundle)
        bindings.append(
            ResearcherTraceCaptureBinding(
                episode_id=slot.episode_id,
                context_id=slot.context_id,
                projection_id=resolved.projection.projection_id,
                projection_hash=resolved.projection.content_hash,
                bundle_hash=bundle.content_hash,
                source_export_hash=resolved.export.content_hash,
            )
        )
    manifest = ResearcherTraceCaptureServerManifest(
        protocol_hash=protocol.content_hash,
        primary_results_gate_hash=closed.gate.content_hash,
        held_out_execution_manifest_hash=closed.execution.content_hash,
        bindings=tuple(bindings),
        opened_at=opened_at,
    )
    application = create_app(
        LocalUiRepository(tuple(bundles)),
        revision_seed=protocol.llm_seed,
        allow_restricted_evidence_metadata=True,
        clock=clock,
        researcher_trace_episode_by_projection_id={
            item.projection_id: item.episode_id for item in bindings
        },
        researcher_trace_capture_sink=AppendOnlyResearcherTraceCaptureStore(capture_root),
    )
    return application, manifest


def materialize_capture_server_manifest(
    *,
    repository: Path,
    restricted_root: Path,
    capture_root: Path,
    manifest: ResearcherTraceCaptureServerManifest,
) -> bool:
    restricted_root = _canonical_restricted_root(repository, restricted_root)
    capture_root = _restricted_descendant(
        restricted_root,
        capture_root,
        "researcher trace capture root",
    )
    return _append_exact(capture_root / CAPTURE_SERVER_MANIFEST_FILE, manifest)


__all__ = [
    "CAPTURE_SERVER_MANIFEST_FILE",
    "ResearcherTraceCaptureBinding",
    "ResearcherTraceCaptureServerManifest",
    "materialize_capture_server_manifest",
    "prepare_researcher_trace_capture_app",
]
