"""Auditable Cytoscape geometry capture for the registered Phase 4 metrics.

The browser reports only renderer state.  Gold-dependent relevance and rare-pivotal
labels are joined afterwards from a sealed scorer plan, so no scorer data is exposed to
the UI or used to alter layout.  Geometry observations remain per projection; the
receipt fixes ``world`` as the only inferentially independent unit.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    OntologyProjection,
    Sha256Digest,
    Viewport,
)
from story_projection_onto.metrics.alignment import prediction_records, score_alignment
from story_projection_onto.metrics.clutter import LabelRectangle, Point
from story_projection_onto.metrics.pipeline import (
    GeometryMetricInput,
    GroundingAuditInput,
    ScorerMetricPlan,
)
from story_projection_onto.ui import (
    LocalUiRepository,
    SemanticDisplayMode,
    VisualizationBundle,
    VisualizationContentScope,
    create_app,
)

CAPTURE_PROTOCOL = "cytoscape-browser-geometry-v1"
DISCOVERABILITY_RULE = "one-click-why-matters-v1"
GEOMETRY_FILENAME_RULE = "projection-content-hash.json"
FROZEN_FONT_FAMILY = "system-ui,sans-serif"
FROZEN_FONT_BASE_PX = 14
RENDERER_HUB_PREFIX = "render-hub:"
RENDERER_LABEL_PREFIX = "render-label:"
RENDERER_SOURCE_PATHS = (
    "configs/study/visualization.json",
    "scripts/capture_renderer_geometry.mjs",
    "ui/app.js",
    "ui/cytoscape-loader.js",
    "ui/cytoscape.lock.json",
    "ui/cytoscape.min.js",
    "ui/index.html",
    "ui/style.css",
)


class RendererGeometryError(RuntimeError):
    """A frozen renderer input, capture, or append-only output failed validation."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_regular_file(path))


def _assert_no_symlink(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise RendererGeometryError(f"symlinked geometry path is forbidden: {path}")
        if current.parent == current:
            return
        current = current.parent


def _regular_file(path: Path, *, maximum_bytes: int = 100 * 1024 * 1024) -> bytes:
    _assert_no_symlink(path)
    if not path.is_file():
        raise RendererGeometryError(f"missing geometry input: {path}")
    if path.stat().st_size > maximum_bytes:
        raise RendererGeometryError(f"geometry input exceeds {maximum_bytes} bytes: {path}")
    return path.read_bytes()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError("geometry paths must be bounded portable relative paths")
    return path


def _append_exact(path: Path, payload: bytes) -> bool:
    """Atomically create one file, accepting only an exact interrupted-run replay."""

    _assert_no_symlink(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RendererGeometryError(f"append-only geometry output drift: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _assert_no_symlink(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    created = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise RendererGeometryError(
                    f"concurrent append-only geometry drift: {path}"
                ) from error
            created = False
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return created


def _record_bytes(record: ImmutableRecord) -> bytes:
    return (record.to_canonical_json() + "\n").encode("utf-8")


def renderer_source_file_hashes(repository: Path) -> tuple[tuple[str, str], ...]:
    """Hash every tracked source that can affect the browser geometry observation."""

    return tuple(
        (relative, _sha256_file(repository / Path(*PurePosixPath(relative).parts)))
        for relative in RENDERER_SOURCE_PATHS
    )


class RendererTypographyStyle(ImmutableRecord):
    """Canonical effective Cytoscape typography for one renderer element kind."""

    element_kind: Literal["node", "edge", "hub"]
    font_family: Literal["system-ui,sans-serif"] = FROZEN_FONT_FAMILY
    font_size_px: Literal[14] = FROZEN_FONT_BASE_PX
    font_style: Literal["normal"] = "normal"
    font_weight: Literal["normal"] = "normal"


class RendererRuntimeIdentity(ImmutableRecord):
    cytoscape_version: str = Field(min_length=1)
    user_agent: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    device_pixel_ratio: float = Field(gt=0.0)
    container_width: int = Field(gt=0)
    container_height: int = Field(gt=0)
    document_fonts_status: Literal["loaded"] = "loaded"
    font_probe_available: Literal[True] = True
    font_probe_css: str = Field(min_length=1)
    font_probe_widths: tuple[float, float, float]
    typography_styles: tuple[
        RendererTypographyStyle,
        RendererTypographyStyle,
        RendererTypographyStyle,
    ]

    @model_validator(mode="after")
    def finite_measurements(self) -> Self:
        if not math.isfinite(self.device_pixel_ratio) or not all(
            math.isfinite(value) and value > 0 for value in self.font_probe_widths
        ):
            raise ValueError("renderer runtime measurements must be finite and positive")
        expected_kinds = ("edge", "hub", "node")
        if tuple(item.element_kind for item in self.typography_styles) != expected_kinds:
            raise ValueError("renderer typography styles must cover edge, hub, and node")
        normalized_probe = (
            self.font_probe_css.casefold()
            .replace(" ", "")
            .replace('"', "")
            .replace("'", "")
        )
        if normalized_probe not in {
            "14pxsystem-ui,sans-serif",
            "normalnormal14pxsystem-ui,sans-serif",
        }:
            raise ValueError("renderer font probe differs from frozen 14px system-ui typography")
        return self


class RendererRawCapture(ImmutableRecord):
    """Condition-blind values returned by the real Cytoscape browser renderer."""

    capture_protocol: Literal["cytoscape-browser-geometry-v1"] = CAPTURE_PROTOCOL
    projection_hash: Sha256Digest
    visualization_bundle_hash: Sha256Digest
    visualization_state_hash: Sha256Digest
    layout_config_hash: Sha256Digest
    style_config_hash: Sha256Digest
    font_config_hash: Sha256Digest
    viewport: Viewport
    labels_visible: Literal[True] = True
    temporal_filter_hash: Sha256Digest
    positions: tuple[tuple[str, Point], ...]
    renderer_only_positions: tuple[tuple[str, Point], ...]
    label_rectangles: tuple[LabelRectangle, ...]
    label_semantic_ids: tuple[tuple[str, str], ...]
    visible_semantic_ids: tuple[str, ...]
    interactive_assertion_ids: tuple[str, ...]
    renderer_runtime: RendererRuntimeIdentity

    @model_validator(mode="after")
    def canonical_renderer_rows(self) -> Self:
        position_ids = tuple(item[0] for item in self.positions)
        if self.positions != tuple(sorted(self.positions, key=lambda item: item[0])) or len(
            position_ids
        ) != len(set(position_ids)):
            raise ValueError("renderer positions must be canonical and unique")
        renderer_only_ids = tuple(item[0] for item in self.renderer_only_positions)
        if self.renderer_only_positions != tuple(
            sorted(self.renderer_only_positions, key=lambda item: item[0])
        ) or len(renderer_only_ids) != len(set(renderer_only_ids)):
            raise ValueError("renderer-only positions must be canonical and unique")
        if set(position_ids) & set(renderer_only_ids):
            raise ValueError("semantic and renderer-only position IDs must be disjoint")
        rectangles = tuple(item.label_id for item in self.label_rectangles)
        if self.label_rectangles != tuple(
            sorted(self.label_rectangles, key=lambda item: item.label_id)
        ) or len(rectangles) != len(set(rectangles)):
            raise ValueError("renderer label rectangles must be canonical and unique")
        label_component_ids = tuple(item[0] for item in self.label_semantic_ids)
        if self.label_semantic_ids != tuple(sorted(self.label_semantic_ids)) or len(
            label_component_ids
        ) != len(set(label_component_ids)):
            raise ValueError("renderer label-to-semantic bindings must be canonical and unique")
        if set(label_component_ids) != set(rectangles):
            raise ValueError("every renderer label rectangle requires one semantic binding")
        for values, label in (
            (self.visible_semantic_ids, "visible semantic IDs"),
            (self.interactive_assertion_ids, "interactive assertion IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"renderer {label} must be sorted and unique")
        if not set(self.interactive_assertion_ids).issubset(self.visible_semantic_ids):
            raise ValueError("interactive assertions must be visible")
        if not {item[1] for item in self.label_semantic_ids}.issubset(
            self.visible_semantic_ids
        ):
            raise ValueError("renderer label bindings must name visible semantics")
        return self


class GeometryObservationBinding(ImmutableRecord):
    source_block: Literal["primary", "combined"]
    intended_unit_hash: Sha256Digest
    unit_id: str = Field(min_length=1)
    source_result_hash: Sha256Digest
    world_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    condition: ConditionName
    seed_block: int | None = Field(default=None, ge=0)


class RarePivotalVisualMatch(ImmutableRecord):
    assertion_target_id: str = Field(min_length=1)
    projection_assertion_id: str = Field(min_length=1)


class GeometrySourceEntry(ImmutableRecord):
    """One projection and scorer join, prepared before browser capture."""

    projection_hash: Sha256Digest
    projection_id: str = Field(min_length=1)
    condition: ConditionName
    world_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    context_hash: Sha256Digest
    packet_hash: Sha256Digest
    scorer_plan_hash: Sha256Digest
    visualization_bundle_hash: Sha256Digest
    visualization_state_hash: Sha256Digest
    bundle_relative_path: str
    bundle_file_sha256: Sha256Digest
    layout_config_hash: Sha256Digest
    style_config_hash: Sha256Digest
    font_config_hash: Sha256Digest
    viewport_hash: Sha256Digest
    projection_node_ids: tuple[str, ...]
    projection_assertion_ids: tuple[str, ...]
    irrelevant_semantic_ids: tuple[str, ...]
    rare_pivotal_matches: tuple[RarePivotalVisualMatch, ...]
    observations: tuple[GeometryObservationBinding, ...] = Field(min_length=1)
    structurally_valid_success: Literal[True] = True

    @model_validator(mode="after")
    def canonical_source(self) -> Self:
        _relative_path(self.bundle_relative_path)
        for values, label in (
            (self.projection_node_ids, "projection node IDs"),
            (self.projection_assertion_ids, "projection assertion IDs"),
            (self.irrelevant_semantic_ids, "irrelevant semantic IDs"),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be sorted and unique")
        semantic_ids = set(self.projection_node_ids) | set(self.projection_assertion_ids)
        if not set(self.irrelevant_semantic_ids).issubset(semantic_ids):
            raise ValueError("irrelevant IDs must reference the source projection")
        if self.rare_pivotal_matches != tuple(
            sorted(self.rare_pivotal_matches, key=lambda item: item.assertion_target_id)
        ):
            raise ValueError("rare-pivotal visual matches must be canonically ordered")
        target_ids = tuple(item.assertion_target_id for item in self.rare_pivotal_matches)
        prediction_ids = tuple(item.projection_assertion_id for item in self.rare_pivotal_matches)
        if len(target_ids) != len(set(target_ids)) or len(prediction_ids) != len(
            set(prediction_ids)
        ):
            raise ValueError("rare-pivotal visual matches must be one-to-one")
        if not set(prediction_ids).issubset(self.projection_assertion_ids):
            raise ValueError("rare-pivotal matches reference unknown assertions")
        if self.observations != tuple(
            sorted(self.observations, key=lambda item: item.intended_unit_hash)
        ):
            raise ValueError("geometry observations must be canonically ordered")
        hashes = tuple(item.intended_unit_hash for item in self.observations)
        if len(hashes) != len(set(hashes)):
            raise ValueError("geometry observations must be unique")
        if len({item.world_id for item in self.observations}) != 1:
            raise ValueError("one projection cannot cross independent worlds")
        if any(
            item.world_id != self.world_id
            or item.context_id != self.context_id
            or item.condition is not self.condition
            for item in self.observations
        ):
            raise ValueError("geometry observations name another projection cell")
        return self

    @property
    def source_result_hashes(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_result_hash for item in self.observations}))


class GeometrySourceManifest(ImmutableRecord):
    manifest_id: str = Field(min_length=1)
    phase4_analysis_configuration_hash: Sha256Digest
    metric_configuration_hash: Sha256Digest
    phase4_source_binding_hash: Sha256Digest
    source_association_hash: Sha256Digest
    source_tree_hash: Sha256Digest
    visualization_configuration_hash: Sha256Digest
    renderer_source_files: tuple[tuple[str, Sha256Digest], ...]
    cytoscape_asset_sha256: Sha256Digest
    cytoscape_version: str = Field(min_length=1)
    layout_config_hash: Sha256Digest
    style_config_hash: Sha256Digest
    font_config_hash: Sha256Digest
    font_family: Literal["system-ui,sans-serif"] = FROZEN_FONT_FAMILY
    font_base_px: Literal[14] = FROZEN_FONT_BASE_PX
    viewport_hash: Sha256Digest
    entries: tuple[GeometrySourceEntry, ...]
    prepared_at: AwareDatetime
    geometry_filename_rule: Literal["projection-content-hash.json"] = GEOMETRY_FILENAME_RULE
    capture_protocol: Literal["cytoscape-browser-geometry-v1"] = CAPTURE_PROTOCOL
    discoverability_rule: Literal["one-click-why-matters-v1"] = DISCOVERABILITY_RULE
    observation_unit: Literal["projection"] = "projection"
    independent_confirmatory_unit: Literal["world"] = "world"
    contexts_are_independent_samples: Literal[False] = False
    model_service_called: Literal[False] = False

    @model_validator(mode="after")
    def canonical_inventory(self) -> Self:
        if tuple(item[0] for item in self.renderer_source_files) != RENDERER_SOURCE_PATHS:
            raise ValueError("geometry source manifest has an incomplete renderer source set")
        if len(dict(self.renderer_source_files)) != len(self.renderer_source_files):
            raise ValueError("renderer source paths must be unique")
        if dict(self.renderer_source_files)["ui/cytoscape.min.js"] != (self.cytoscape_asset_sha256):
            raise ValueError("Cytoscape hash differs from the renderer source inventory")
        if self.entries != tuple(sorted(self.entries, key=lambda item: item.projection_hash)):
            raise ValueError("geometry source entries must be sorted by projection hash")
        hashes = tuple(item.projection_hash for item in self.entries)
        if len(hashes) != len(set(hashes)):
            raise ValueError("geometry source requires one entry per projection hash")
        expected = (
            self.layout_config_hash,
            self.style_config_hash,
            self.font_config_hash,
            self.viewport_hash,
        )
        if any(
            (
                entry.layout_config_hash,
                entry.style_config_hash,
                entry.font_config_hash,
                entry.viewport_hash,
            )
            != expected
            for entry in self.entries
        ):
            raise ValueError("geometry source entries use different renderer configurations")
        return self


class GeometryArtifactBinding(ImmutableRecord):
    projection_hash: Sha256Digest
    source_entry_hash: Sha256Digest
    source_result_hashes: tuple[Sha256Digest, ...] = Field(min_length=1)
    raw_capture_hash: Sha256Digest
    raw_capture_file_sha256: Sha256Digest
    geometry_hash: Sha256Digest
    geometry_file_sha256: Sha256Digest
    geometry_relative_path: str
    content_addressed_relative_path: str

    @model_validator(mode="after")
    def bounded_paths_and_sources(self) -> Self:
        _relative_path(self.geometry_relative_path)
        _relative_path(self.content_addressed_relative_path)
        if self.source_result_hashes != tuple(sorted(set(self.source_result_hashes))):
            raise ValueError("receipt source-result hashes must be canonical")
        return self


class GeometryMaterializationReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    source_manifest_hash: Sha256Digest
    source_manifest_file_sha256: Sha256Digest
    phase4_analysis_configuration_hash: Sha256Digest
    metric_configuration_hash: Sha256Digest
    phase4_source_binding_hash: Sha256Digest
    source_association_hash: Sha256Digest
    source_tree_hash: Sha256Digest
    renderer_runtime_hash: Sha256Digest | None = None
    artifacts: tuple[GeometryArtifactBinding, ...]
    completed_at: AwareDatetime
    geometry_filename_rule: Literal["projection-content-hash.json"] = GEOMETRY_FILENAME_RULE
    capture_protocol: Literal["cytoscape-browser-geometry-v1"] = CAPTURE_PROTOCOL
    discoverability_rule: Literal["one-click-why-matters-v1"] = DISCOVERABILITY_RULE
    observation_unit: Literal["projection"] = "projection"
    independent_confirmatory_unit: Literal["world"] = "world"
    contexts_are_independent_samples: Literal[False] = False
    model_service_called: Literal[False] = False
    renderer_values_fabricated: Literal[False] = False

    @model_validator(mode="after")
    def canonical_artifacts(self) -> Self:
        if self.artifacts != tuple(sorted(self.artifacts, key=lambda item: item.projection_hash)):
            raise ValueError("geometry receipt artifacts must be sorted")
        hashes = tuple(item.projection_hash for item in self.artifacts)
        if len(hashes) != len(set(hashes)):
            raise ValueError("geometry receipt repeats a projection")
        if bool(self.artifacts) != (self.renderer_runtime_hash is not None):
            raise ValueError("nonempty geometry receipts require one renderer runtime")
        return self


class GeometryPreflight(ImmutableRecord):
    source_manifest_hash: Sha256Digest
    expected_geometry_count: int = Field(ge=0)
    exact_existing_geometry_count: int = Field(ge=0)
    missing_projection_hashes: tuple[Sha256Digest, ...]
    exact_existing_capture_count: int = Field(ge=0)
    missing_capture_hashes: tuple[Sha256Digest, ...]
    browser_capture_available: bool
    replay_receipt_present: bool
    ready_to_capture: bool


def derive_geometry_scorer_annotations(
    projection: OntologyProjection,
    *,
    scorer_plan: ScorerMetricPlan,
    grounding_audit: GroundingAuditInput,
) -> tuple[tuple[str, ...], tuple[RarePivotalVisualMatch, ...]]:
    """Derive relevance and rare joins without exposing scorer labels to the browser."""

    if grounding_audit.projection_hash != projection.content_hash:
        raise RendererGeometryError("grounding audit names another geometry projection")
    predicted_nodes, predicted_assertions, _ = prediction_records(
        projection,
        predicate_aliases=dict(scorer_plan.predicate_aliases),
        valid_evidence_ids=frozenset(scorer_plan.valid_evidence_ids),
        grounding_by_assertion_id=dict(grounding_audit.assertion_statuses),
        plan=scorer_plan.alignment_plan,
    )
    alignment = score_alignment(
        plan=scorer_plan.alignment_plan,
        predicted_nodes=predicted_nodes,
        predicted_assertions=predicted_assertions,
    )
    relevant_nodes = {item.prediction_id for item in alignment.node_matches}
    relevant_assertions = {
        item.prediction_id for item in alignment.structurally_aligned_assertion_matches
    }
    irrelevant = tuple(
        sorted(
            ({item.prediction_id for item in predicted_nodes} - relevant_nodes)
            | ({item.prediction_id for item in predicted_assertions} - relevant_assertions)
        )
    )
    rare_targets = {
        item.assertion_target_id
        for item in scorer_plan.rare_annotations
        if item.is_rare and item.is_pivotal
    }
    rare_matches = tuple(
        RarePivotalVisualMatch(
            assertion_target_id=item.target_id,
            projection_assertion_id=item.prediction_id,
        )
        for item in sorted(
            alignment.strict_assertion_matches,
            key=lambda value: value.target_id,
        )
        if item.target_id in rare_targets
    )
    return irrelevant, rare_matches


def _require_registered_geometry_bundle(bundle: VisualizationBundle) -> None:
    """Reject raw/full or post-hoc-filtered views from confirmatory clutter input."""

    if (
        bundle.display_selection.content_scope
        is not VisualizationContentScope.REGISTERED_DISPLAY
        or bundle.semantic_display_mode is not SemanticDisplayMode.ALL_STRUCTURAL
        or bundle.semantic_overlay is not None
        or bundle.semantic_assessment_status != "pending_scorer_or_reviewer"
    ):
        raise RendererGeometryError(
            "confirmatory geometry requires the unfiltered registered-display view"
        )


def build_geometry_source_entry(
    *,
    projection: OntologyProjection,
    bundle: VisualizationBundle,
    scorer_plan: ScorerMetricPlan,
    grounding_audit: GroundingAuditInput,
    observations: Sequence[GeometryObservationBinding],
    bundle_relative_path: str,
) -> GeometrySourceEntry:
    """Bind a validated projection, its renderer DTO, source results, and scorer join."""

    if bundle.projection_hash != projection.content_hash:
        raise RendererGeometryError("visualization bundle names another projection")
    _require_registered_geometry_bundle(bundle)
    canonical_observations = tuple(sorted(observations, key=lambda item: item.intended_unit_hash))
    if not canonical_observations:
        raise RendererGeometryError("geometry source requires at least one result observation")
    if any(
        item.condition is not projection.condition or item.context_id != bundle.context.context_id
        for item in canonical_observations
    ):
        raise RendererGeometryError("geometry result observations name another projection")
    node_ids = bundle.display_selection.projection_node_ids
    assertion_ids = bundle.display_selection.projection_assertion_ids
    rendered_nodes = tuple(sorted(item.projection_object_id for item in bundle.state.nodes))
    rendered_assertions = tuple(
        sorted(item.projection_assertion_id for item in bundle.state.assertions)
    )
    if rendered_nodes != node_ids or rendered_assertions != assertion_ids:
        raise RendererGeometryError(
            "registered geometry must exactly cover the frozen display selection"
        )
    irrelevant, rare_matches = derive_geometry_scorer_annotations(
        projection,
        scorer_plan=scorer_plan,
        grounding_audit=grounding_audit,
    )
    selected_semantic_ids = set(node_ids) | set(assertion_ids)
    irrelevant = tuple(sorted(set(irrelevant).intersection(selected_semantic_ids)))
    rare_matches = tuple(
        item for item in rare_matches if item.projection_assertion_id in assertion_ids
    )
    payload = _record_bytes(bundle)
    return GeometrySourceEntry(
        projection_hash=projection.content_hash,
        projection_id=projection.projection_id,
        condition=projection.condition,
        world_id=canonical_observations[0].world_id,
        context_id=bundle.context.context_id,
        context_hash=bundle.context_hash,
        packet_hash=bundle.packet_hash,
        scorer_plan_hash=scorer_plan.content_hash,
        visualization_bundle_hash=bundle.content_hash,
        visualization_state_hash=bundle.state.content_hash,
        bundle_relative_path=bundle_relative_path,
        bundle_file_sha256=_sha256_bytes(payload),
        layout_config_hash=bundle.state.layout_config_hash,
        style_config_hash=bundle.state.style_config_hash,
        font_config_hash=bundle.state.font_config_hash,
        viewport_hash=bundle.state.viewport.content_hash,
        projection_node_ids=node_ids,
        projection_assertion_ids=assertion_ids,
        irrelevant_semantic_ids=irrelevant,
        rare_pivotal_matches=rare_matches,
        observations=canonical_observations,
    )


def write_geometry_source(
    *,
    root: Path,
    manifest: GeometrySourceManifest,
    bundles: Mapping[str, VisualizationBundle],
) -> None:
    """Write the immutable capture plan and content-bound renderer DTOs."""

    if set(bundles) != {item.projection_hash for item in manifest.entries}:
        raise RendererGeometryError("geometry bundle inventory differs from its source manifest")
    for entry in manifest.entries:
        bundle = bundles[entry.projection_hash]
        payload = _record_bytes(bundle)
        if (
            bundle.content_hash != entry.visualization_bundle_hash
            or bundle.state.content_hash != entry.visualization_state_hash
            or _sha256_bytes(payload) != entry.bundle_file_sha256
        ):
            raise RendererGeometryError("geometry renderer DTO changed after source planning")
        _append_exact(root / Path(*_relative_path(entry.bundle_relative_path).parts), payload)
    _append_exact(root / "source_manifest.json", _record_bytes(manifest))


def load_geometry_source(
    root: Path,
) -> tuple[GeometrySourceManifest, dict[str, VisualizationBundle]]:
    manifest_payload = _regular_file(root / "source_manifest.json")
    try:
        manifest = GeometrySourceManifest.model_validate_json(manifest_payload)
    except Exception as error:
        raise RendererGeometryError(f"invalid geometry source manifest: {error}") from error
    if manifest_payload != _record_bytes(manifest):
        raise RendererGeometryError("geometry source manifest is not canonical")
    bundles: dict[str, VisualizationBundle] = {}
    for entry in manifest.entries:
        path = root / Path(*_relative_path(entry.bundle_relative_path).parts)
        payload = _regular_file(path)
        if _sha256_bytes(payload) != entry.bundle_file_sha256:
            raise RendererGeometryError("renderer DTO physical hash changed")
        try:
            bundle = VisualizationBundle.model_validate_json(payload)
        except Exception as error:
            raise RendererGeometryError(f"invalid renderer DTO: {error}") from error
        _require_registered_geometry_bundle(bundle)
        if (
            bundle.content_hash != entry.visualization_bundle_hash
            or bundle.state.content_hash != entry.visualization_state_hash
            or bundle.projection_hash != entry.projection_hash
            or bundle.context_hash != entry.context_hash
            or bundle.packet_hash != entry.packet_hash
            or bundle.state.layout_config_hash != entry.layout_config_hash
            or bundle.state.style_config_hash != entry.style_config_hash
            or bundle.state.font_config_hash != entry.font_config_hash
            or bundle.state.viewport.content_hash != entry.viewport_hash
        ):
            raise RendererGeometryError("renderer DTO lineage differs from source entry")
        bundles[entry.projection_hash] = bundle
    return manifest, bundles


def _expected_visible_semantic_ids(bundle: VisualizationBundle) -> tuple[str, ...]:
    nodes = {item.visualization_node_id: item.projection_object_id for item in bundle.state.nodes}
    assertions = {
        item.visualization_assertion_id: item.projection_assertion_id
        for item in bundle.state.assertions
    }
    return tuple(
        sorted(
            [nodes[item] for item in bundle.state.visible_node_ids]
            + [assertions[item] for item in bundle.state.visible_assertion_ids]
        )
    )


def _round_renderer_coordinate(value: float) -> float:
    """Match JavaScript ``Math.round(value * 1e6) / 1e6`` exactly."""

    return math.floor(value * 1_000_000 + 0.5) / 1_000_000


def _stable_code_unit_hash(value: str) -> int:
    """FNV-1a over legal ASCII identifiers, mirrored byte-for-code-unit in the UI."""

    if not value.isascii():
        raise RendererGeometryError("renderer identifiers must remain ASCII")
    result = 2_166_136_261
    for character in value:
        result = ((result ^ ord(character)) * 16_777_619) & 0xFFFF_FFFF
    return result


def _renderer_hub_id(visualization_assertion_id: str) -> str:
    return f"{RENDERER_HUB_PREFIX}{visualization_assertion_id}"


def _renderer_hub_label_id(visualization_assertion_id: str) -> str:
    return f"{RENDERER_LABEL_PREFIX}{visualization_assertion_id}:hub"


def _renderer_role_label_id(visualization_assertion_id: str, index: int) -> str:
    return f"{RENDERER_LABEL_PREFIX}{visualization_assertion_id}:role:{index}"


def _renderer_hub_model_positions(bundle: VisualizationBundle) -> tuple[tuple[str, Point], ...]:
    """Derive collision-free n-ary hub positions from role positions and stable IDs."""

    state = bundle.state
    position_by_id = {
        item.visualization_node_id: Point(x=item.x, y=item.y) for item in state.positions
    }
    occupied = {
        (_round_renderer_coordinate(item.x), _round_renderer_coordinate(item.y))
        for item in position_by_id.values()
    }
    visible = set(state.visible_assertion_ids)
    result: list[tuple[str, Point]] = []
    for assertion in sorted(state.assertions, key=lambda item: item.visualization_assertion_id):
        if assertion.visualization_assertion_id not in visible or not assertion.roles:
            continue
        try:
            role_positions = tuple(position_by_id[item.object_id] for item in assertion.roles)
        except KeyError as error:
            raise RendererGeometryError(
                "n-ary renderer hub references an unpositioned role"
            ) from error
        center_x = sum(item.x for item in role_positions) / len(role_positions)
        center_y = sum(item.y for item in role_positions) / len(role_positions)
        identifier_hash = _stable_code_unit_hash(assertion.visualization_assertion_id)
        offset_x = identifier_hash % 49 - 24
        offset_y = (identifier_hash // 49) % 49 - 24
        if offset_x == 0 and offset_y == 0:
            offset_x = 25
        attempt = 0
        while True:
            x = _round_renderer_coordinate(center_x + offset_x + attempt * 53)
            y = _round_renderer_coordinate(center_y + offset_y + attempt * 47)
            key = (x, y)
            if key != (0.0, 0.0) and key not in occupied:
                occupied.add(key)
                result.append(
                    (
                        _renderer_hub_id(assertion.visualization_assertion_id),
                        Point(x=x, y=y),
                    )
                )
                break
            attempt += 1
    return tuple(sorted(result, key=lambda item: item[0]))


def _expected_renderer_only_positions(
    bundle: VisualizationBundle,
) -> tuple[tuple[str, Point], ...]:
    viewport = bundle.state.viewport
    return tuple(
        (
            identifier,
            Point(
                x=_round_renderer_coordinate(
                    (position.x - viewport.center_x) * viewport.zoom + viewport.width / 2
                ),
                y=_round_renderer_coordinate(
                    (position.y - viewport.center_y) * viewport.zoom + viewport.height / 2
                ),
            ),
        )
        for identifier, position in _renderer_hub_model_positions(bundle)
    )


def _expected_label_semantic_ids(bundle: VisualizationBundle) -> tuple[tuple[str, str], ...]:
    state = bundle.state
    visible_nodes = set(state.visible_node_ids)
    visible_assertions = set(state.visible_assertion_ids)
    bindings: list[tuple[str, str]] = [
        (item.projection_object_id, item.projection_object_id)
        for item in state.nodes
        if item.visualization_node_id in visible_nodes
    ]
    for assertion in state.assertions:
        if assertion.visualization_assertion_id not in visible_assertions:
            continue
        if assertion.roles:
            bindings.append(
                (
                    _renderer_hub_label_id(assertion.visualization_assertion_id),
                    assertion.projection_assertion_id,
                )
            )
            bindings.extend(
                (
                    _renderer_role_label_id(assertion.visualization_assertion_id, index),
                    assertion.projection_assertion_id,
                )
                for index, _ in enumerate(assertion.roles)
            )
        else:
            bindings.append(
                (assertion.projection_assertion_id, assertion.projection_assertion_id)
            )
    return tuple(sorted(bindings))


def materialize_geometry_record(
    *,
    source_manifest_hash: str,
    entry: GeometrySourceEntry,
    bundle: VisualizationBundle,
    capture: RendererRawCapture,
    expected_cytoscape_version: str,
) -> GeometryMetricInput:
    """Validate one raw browser capture and join scorer-only annotations."""

    _require_registered_geometry_bundle(bundle)
    state = bundle.state
    expected_identities = (
        entry.projection_hash,
        entry.visualization_bundle_hash,
        entry.visualization_state_hash,
        entry.layout_config_hash,
        entry.style_config_hash,
        entry.font_config_hash,
        entry.viewport_hash,
    )
    captured_identities = (
        capture.projection_hash,
        capture.visualization_bundle_hash,
        capture.visualization_state_hash,
        capture.layout_config_hash,
        capture.style_config_hash,
        capture.font_config_hash,
        capture.viewport.content_hash,
    )
    if captured_identities != expected_identities:
        raise RendererGeometryError("browser capture names different source or renderer inputs")
    if capture.renderer_runtime.cytoscape_version != expected_cytoscape_version:
        raise RendererGeometryError("browser loaded a different Cytoscape version")
    if (
        not math.isclose(capture.renderer_runtime.device_pixel_ratio, 1.0, abs_tol=1e-12)
        or capture.renderer_runtime.container_width != state.viewport.width
        or capture.renderer_runtime.container_height != state.viewport.height
    ):
        raise RendererGeometryError("browser capture did not use the frozen unit-scale viewport")
    expected_visible = _expected_visible_semantic_ids(bundle)
    if capture.visible_semantic_ids != expected_visible:
        raise RendererGeometryError("browser visibility differs from the visualization state")
    if (
        capture.labels_visible != state.labels_visible
        or capture.temporal_filter_hash != state.temporal_filter.content_hash
    ):
        raise RendererGeometryError("browser disclosure/filter state differs from its source")
    if tuple(item[0] for item in capture.positions) != entry.projection_node_ids:
        raise RendererGeometryError("browser positions do not exactly cover projection nodes")
    position_by_visual_id = {item.visualization_node_id: item for item in state.positions}
    visual_id_by_semantic_id = {
        item.projection_object_id: item.visualization_node_id for item in state.nodes
    }
    viewport = state.viewport
    for semantic_id, observed in capture.positions:
        model = position_by_visual_id[visual_id_by_semantic_id[semantic_id]]
        expected_x = (model.x - viewport.center_x) * viewport.zoom + viewport.width / 2
        expected_y = (model.y - viewport.center_y) * viewport.zoom + viewport.height / 2
        if not (
            math.isclose(observed.x, expected_x, abs_tol=1e-5)
            and math.isclose(observed.y, expected_y, abs_tol=1e-5)
        ):
            raise RendererGeometryError("Cytoscape changed a frozen preset position")
    expected_renderer_only = _expected_renderer_only_positions(bundle)
    if len(capture.renderer_only_positions) != len(expected_renderer_only) or any(
        identifier != expected_identifier
        or not math.isclose(observed.x, expected_position.x, abs_tol=1e-5)
        or not math.isclose(observed.y, expected_position.y, abs_tol=1e-5)
        for (identifier, observed), (expected_identifier, expected_position) in zip(
            capture.renderer_only_positions,
            expected_renderer_only,
            strict=True,
        )
    ):
        raise RendererGeometryError("Cytoscape changed a frozen n-ary hub position")
    expected_label_bindings = _expected_label_semantic_ids(bundle)
    if (
        capture.label_semantic_ids != expected_label_bindings
        or tuple(item.label_id for item in capture.label_rectangles)
        != tuple(item[0] for item in expected_label_bindings)
    ):
        raise RendererGeometryError("browser label geometry does not cover renderer components")
    visible_assertions = set(entry.projection_assertion_ids) & set(expected_visible)
    if set(capture.interactive_assertion_ids) != visible_assertions:
        raise RendererGeometryError("browser assertion interaction surface is incomplete")
    discoverability = tuple(
        sorted(
            (
                match.assertion_target_id,
                1,
            )
            for match in entry.rare_pivotal_matches
            if match.projection_assertion_id in capture.interactive_assertion_ids
        )
    )
    irrelevant = tuple(
        sorted(set(entry.irrelevant_semantic_ids) & set(capture.visible_semantic_ids))
    )
    return GeometryMetricInput(
        projection_hash=entry.projection_hash,
        visualization_state_hash=entry.visualization_state_hash,
        materialization_source_hash=source_manifest_hash,
        source_result_hashes=entry.source_result_hashes,
        layout_config_hash=entry.layout_config_hash,
        style_config_hash=entry.style_config_hash,
        font_config_hash=entry.font_config_hash,
        viewport_hash=entry.viewport_hash,
        renderer_runtime_hash=capture.renderer_runtime.content_hash,
        positions=capture.positions,
        renderer_only_positions=capture.renderer_only_positions,
        label_rectangles=capture.label_rectangles,
        label_semantic_ids=capture.label_semantic_ids,
        visible_semantic_ids=capture.visible_semantic_ids,
        irrelevant_semantic_ids=irrelevant,
        rare_pivotal_discoverability=discoverability,
    )


def materialize_geometry_captures(
    *,
    source_root: Path,
    geometry_root: Path,
    captures: Mapping[str, RendererRawCapture],
    completed_at: datetime,
) -> GeometryMaterializationReceipt:
    """Write raw captures, Phase 4 records, a local CAS, and one immutable receipt."""

    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise RendererGeometryError("geometry completion timestamp must be timezone-aware")
    manifest, bundles = load_geometry_source(source_root)
    expected = {item.projection_hash for item in manifest.entries}
    if set(captures) != expected:
        raise RendererGeometryError("raw capture inventory differs from planned projections")
    runtime_hashes = {item.renderer_runtime.content_hash for item in captures.values()}
    if len(runtime_hashes) > 1:
        raise RendererGeometryError("one geometry run cannot mix renderer runtimes")
    artifacts = []
    for entry in manifest.entries:
        capture = captures[entry.projection_hash]
        geometry = materialize_geometry_record(
            source_manifest_hash=manifest.content_hash,
            entry=entry,
            bundle=bundles[entry.projection_hash],
            capture=capture,
            expected_cytoscape_version=manifest.cytoscape_version,
        )
        raw_payload = _record_bytes(capture)
        geometry_payload = _record_bytes(geometry)
        geometry_relative = f"{entry.projection_hash}.json"
        content_relative = f"by-content/{geometry.content_hash}.json"
        _append_exact(geometry_root / "raw" / f"{entry.projection_hash}.json", raw_payload)
        _append_exact(geometry_root / geometry_relative, geometry_payload)
        _append_exact(geometry_root / content_relative, geometry_payload)
        artifacts.append(
            GeometryArtifactBinding(
                projection_hash=entry.projection_hash,
                source_entry_hash=entry.content_hash,
                source_result_hashes=entry.source_result_hashes,
                raw_capture_hash=capture.content_hash,
                raw_capture_file_sha256=_sha256_bytes(raw_payload),
                geometry_hash=geometry.content_hash,
                geometry_file_sha256=_sha256_bytes(geometry_payload),
                geometry_relative_path=geometry_relative,
                content_addressed_relative_path=content_relative,
            )
        )
    source_payload = _regular_file(source_root / "source_manifest.json")
    receipt = GeometryMaterializationReceipt(
        receipt_id=f"phase4-renderer-geometry-{manifest.content_hash[:24]}",
        source_manifest_hash=manifest.content_hash,
        source_manifest_file_sha256=_sha256_bytes(source_payload),
        phase4_analysis_configuration_hash=manifest.phase4_analysis_configuration_hash,
        metric_configuration_hash=manifest.metric_configuration_hash,
        phase4_source_binding_hash=manifest.phase4_source_binding_hash,
        source_association_hash=manifest.source_association_hash,
        source_tree_hash=manifest.source_tree_hash,
        renderer_runtime_hash=(None if not runtime_hashes else next(iter(runtime_hashes))),
        artifacts=tuple(artifacts),
        completed_at=completed_at,
    )
    _append_exact(geometry_root / "materialization_receipt.json", _record_bytes(receipt))
    return receipt


def replay_geometry_materialization(
    *,
    source_root: Path,
    geometry_root: Path,
) -> GeometryMaterializationReceipt:
    """Verify every source, raw capture, derived record, CAS copy, and receipt byte."""

    manifest, bundles = load_geometry_source(source_root)
    receipt_payload = _regular_file(geometry_root / "materialization_receipt.json")
    try:
        receipt = GeometryMaterializationReceipt.model_validate_json(receipt_payload)
    except Exception as error:
        raise RendererGeometryError(f"invalid geometry materialization receipt: {error}") from error
    if receipt_payload != _record_bytes(receipt):
        raise RendererGeometryError("geometry materialization receipt is not canonical")
    if (
        receipt.source_manifest_hash != manifest.content_hash
        or receipt.source_manifest_file_sha256 != _sha256_file(source_root / "source_manifest.json")
        or receipt.phase4_analysis_configuration_hash != manifest.phase4_analysis_configuration_hash
        or receipt.metric_configuration_hash != manifest.metric_configuration_hash
        or receipt.phase4_source_binding_hash != manifest.phase4_source_binding_hash
        or receipt.source_association_hash != manifest.source_association_hash
        or receipt.source_tree_hash != manifest.source_tree_hash
    ):
        raise RendererGeometryError("geometry receipt names different frozen sources")
    entries = {item.projection_hash: item for item in manifest.entries}
    if {item.projection_hash for item in receipt.artifacts} != set(entries):
        raise RendererGeometryError("geometry receipt does not cover the source manifest exactly")
    runtime_hashes = set()
    for artifact in receipt.artifacts:
        entry = entries[artifact.projection_hash]
        raw_payload = _regular_file(geometry_root / "raw" / f"{artifact.projection_hash}.json")
        if _sha256_bytes(raw_payload) != artifact.raw_capture_file_sha256:
            raise RendererGeometryError("raw capture file hash changed")
        try:
            capture = RendererRawCapture.model_validate_json(raw_payload)
        except Exception as error:
            raise RendererGeometryError(f"invalid replay capture: {error}") from error
        if raw_payload != _record_bytes(capture):
            raise RendererGeometryError("raw capture is not canonical")
        if capture.content_hash != artifact.raw_capture_hash:
            raise RendererGeometryError("raw capture logical hash changed")
        geometry = materialize_geometry_record(
            source_manifest_hash=manifest.content_hash,
            entry=entry,
            bundle=bundles[artifact.projection_hash],
            capture=capture,
            expected_cytoscape_version=manifest.cytoscape_version,
        )
        geometry_payload = _record_bytes(geometry)
        primary = geometry_root / Path(*_relative_path(artifact.geometry_relative_path).parts)
        addressed = geometry_root / Path(
            *_relative_path(artifact.content_addressed_relative_path).parts
        )
        if (
            artifact.source_entry_hash != entry.content_hash
            or artifact.source_result_hashes != entry.source_result_hashes
            or geometry.content_hash != artifact.geometry_hash
            or _sha256_bytes(geometry_payload) != artifact.geometry_file_sha256
            or _regular_file(primary) != geometry_payload
            or _regular_file(addressed) != geometry_payload
        ):
            raise RendererGeometryError("derived geometry differs from its receipt")
        runtime_hashes.add(capture.renderer_runtime.content_hash)
    expected_runtime = None if not runtime_hashes else next(iter(runtime_hashes))
    if len(runtime_hashes) > 1 or receipt.renderer_runtime_hash != expected_runtime:
        raise RendererGeometryError("replayed captures do not use one recorded renderer runtime")
    return receipt


def load_existing_renderer_captures(
    *,
    source_root: Path,
    geometry_root: Path,
) -> dict[str, RendererRawCapture]:
    """Load only exact, reusable raw captures from an interrupted materialization."""

    manifest, bundles = load_geometry_source(source_root)
    captures: dict[str, RendererRawCapture] = {}
    runtime_hashes: set[str] = set()
    for entry in manifest.entries:
        path = geometry_root / "raw" / f"{entry.projection_hash}.json"
        if not path.exists() and not path.is_symlink():
            continue
        payload = _regular_file(path)
        try:
            capture = RendererRawCapture.model_validate_json(payload)
        except Exception as error:
            raise RendererGeometryError(
                f"invalid interrupted renderer capture {entry.projection_hash}: {error}"
            ) from error
        if payload != _record_bytes(capture):
            raise RendererGeometryError("interrupted renderer capture is not canonical")
        materialize_geometry_record(
            source_manifest_hash=manifest.content_hash,
            entry=entry,
            bundle=bundles[entry.projection_hash],
            capture=capture,
            expected_cytoscape_version=manifest.cytoscape_version,
        )
        captures[entry.projection_hash] = capture
        runtime_hashes.add(capture.renderer_runtime.content_hash)
    if len(runtime_hashes) > 1:
        raise RendererGeometryError("interrupted captures mix renderer runtimes")
    return captures


def browser_capture_executables(
    *,
    node_executable: str | None = None,
    browser_executable: str | None = None,
) -> tuple[str | None, str | None]:
    node = shutil.which(node_executable) if node_executable else shutil.which("node")
    browser = (
        shutil.which(browser_executable)
        if browser_executable
        else next(
            (
                path
                for name in ("google-chrome", "chromium", "chromium-browser")
                if (path := shutil.which(name)) is not None
            ),
            None,
        )
    )
    return node, browser


def preflight_geometry_materialization(
    *,
    source_root: Path,
    geometry_root: Path,
    node_executable: str | None = None,
    browser_executable: str | None = None,
) -> GeometryPreflight:
    manifest, bundles = load_geometry_source(source_root)
    captures = load_existing_renderer_captures(
        source_root=source_root,
        geometry_root=geometry_root,
    )
    exact = 0
    missing = []
    for entry in manifest.entries:
        path = geometry_root / f"{entry.projection_hash}.json"
        if path.is_symlink():
            raise RendererGeometryError(f"symlinked geometry output is forbidden: {path}")
        if not path.is_file():
            missing.append(entry.projection_hash)
            continue
        payload = _regular_file(path)
        try:
            geometry = GeometryMetricInput.model_validate_json(payload)
        except Exception as error:
            raise RendererGeometryError(
                f"invalid existing geometry {entry.projection_hash}: {error}"
            ) from error
        if payload != _record_bytes(geometry):
            raise RendererGeometryError("existing renderer geometry is not canonical")
        if (
            geometry.projection_hash != entry.projection_hash
            or geometry.materialization_source_hash != manifest.content_hash
            or geometry.source_result_hashes != entry.source_result_hashes
        ):
            raise RendererGeometryError("existing geometry belongs to another source plan")
        capture = captures.get(entry.projection_hash)
        if capture is not None:
            expected_geometry = materialize_geometry_record(
                source_manifest_hash=manifest.content_hash,
                entry=entry,
                bundle=bundles[entry.projection_hash],
                capture=capture,
                expected_cytoscape_version=manifest.cytoscape_version,
            )
            if geometry != expected_geometry:
                raise RendererGeometryError(
                    "existing geometry differs from its interrupted renderer capture"
                )
        exact += 1
    node, browser = browser_capture_executables(
        node_executable=node_executable,
        browser_executable=browser_executable,
    )
    receipt_path = geometry_root / "materialization_receipt.json"
    receipt_present = receipt_path.exists() or receipt_path.is_symlink()
    if receipt_present:
        replay_geometry_materialization(source_root=source_root, geometry_root=geometry_root)
    missing_captures = tuple(
        item.projection_hash for item in manifest.entries if item.projection_hash not in captures
    )
    ready = (
        not missing
        if receipt_present
        else not manifest.entries or not missing_captures or bool(node and browser)
    )
    return GeometryPreflight(
        source_manifest_hash=manifest.content_hash,
        expected_geometry_count=len(manifest.entries),
        exact_existing_geometry_count=exact,
        missing_projection_hashes=tuple(missing),
        exact_existing_capture_count=len(captures),
        missing_capture_hashes=missing_captures,
        browser_capture_available=bool(node and browser),
        replay_receipt_present=receipt_present,
        ready_to_capture=ready,
    )


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(url: str) -> None:
    deadline = time.monotonic() + 15.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as error:  # pragma: no cover - depends on server startup timing
            last_error = error
        time.sleep(0.05)
    raise RendererGeometryError(f"local geometry renderer did not become healthy: {last_error}")


def capture_with_system_browser(
    bundles: Sequence[VisualizationBundle],
    *,
    ui_directory: Path,
    driver_path: Path,
    node_executable: str | None = None,
    browser_executable: str | None = None,
    timeout_seconds: float = 180.0,
) -> dict[str, RendererRawCapture]:
    """Serve immutable DTOs locally and capture them with the vendored Cytoscape asset."""

    node, browser = browser_capture_executables(
        node_executable=node_executable,
        browser_executable=browser_executable,
    )
    if node is None or browser is None:
        raise RendererGeometryError("geometry capture requires Node.js and Chrome/Chromium")
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - optional study environment only
        raise RendererGeometryError(
            "geometry capture requires the study UI dependencies"
        ) from error
    repository = LocalUiRepository(tuple(bundles))
    application = create_app(
        repository,
        static_directory=ui_directory,
        allow_restricted_evidence_metadata=True,
    )
    port = _available_port()
    configuration = uvicorn.Config(
        application,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(configuration)
    thread = threading.Thread(target=server.run, name="geometry-renderer", daemon=True)
    thread.start()
    application_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(application_url)
        projection_ids = [item.projection_id for item in bundles]
        completed = subprocess.run(
            [node, str(driver_path), browser, application_url, *projection_ids],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RendererGeometryError(
                "browser geometry capture failed: " + completed.stderr[-4000:]
            )
        lines = tuple(line for line in completed.stdout.splitlines() if line.strip())
        if len(lines) != len(bundles):
            raise RendererGeometryError("browser returned an incomplete geometry inventory")
        captures: dict[str, RendererRawCapture] = {}
        for line in lines:
            try:
                capture = RendererRawCapture.model_validate_json(line)
            except Exception as error:
                raise RendererGeometryError(f"invalid browser geometry capture: {error}") from error
            if capture.projection_hash in captures:
                raise RendererGeometryError("browser returned duplicate projection geometry")
            captures[capture.projection_hash] = capture
        expected_hashes = {item.projection_hash for item in bundles}
        if set(captures) != expected_hashes:
            raise RendererGeometryError("browser returned geometry for different projections")
        return captures
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RendererGeometryError("local geometry renderer did not stop")


__all__ = [
    "CAPTURE_PROTOCOL",
    "DISCOVERABILITY_RULE",
    "FROZEN_FONT_BASE_PX",
    "FROZEN_FONT_FAMILY",
    "GEOMETRY_FILENAME_RULE",
    "RENDERER_SOURCE_PATHS",
    "GeometryArtifactBinding",
    "GeometryMaterializationReceipt",
    "GeometryObservationBinding",
    "GeometryPreflight",
    "GeometrySourceEntry",
    "GeometrySourceManifest",
    "RarePivotalVisualMatch",
    "RendererGeometryError",
    "RendererRawCapture",
    "RendererRuntimeIdentity",
    "RendererTypographyStyle",
    "browser_capture_executables",
    "build_geometry_source_entry",
    "capture_with_system_browser",
    "derive_geometry_scorer_annotations",
    "load_existing_renderer_captures",
    "load_geometry_source",
    "materialize_geometry_captures",
    "materialize_geometry_record",
    "preflight_geometry_materialization",
    "renderer_source_file_hashes",
    "replay_geometry_materialization",
    "write_geometry_source",
]
