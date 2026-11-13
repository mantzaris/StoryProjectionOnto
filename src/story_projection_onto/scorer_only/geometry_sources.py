"""Prepare the post-execution, scorer-bound renderer input inventory for Phase 4."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from story_projection_onto.contracts import RunOutcome, canonical_sha256
from story_projection_onto.metrics.adapters import (
    projection_is_content_bearing,
    projection_is_structurally_valid,
)
from story_projection_onto.renderer_geometry import (
    GeometryObservationBinding,
    GeometrySourceManifest,
    build_geometry_source_entry,
    renderer_source_file_hashes,
    write_geometry_source,
)
from story_projection_onto.scorer_only.phase4_analysis import (
    _Cell,
    _grounding_audit,
    _prepare_inputs,
)
from story_projection_onto.ui import (
    VisualizationBundle,
    VisualizationContentScope,
    build_visualization_bundle,
    load_visualization_configuration,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cytoscape_identity(repository: Path) -> tuple[str, str]:
    ui_root = repository / "ui"
    lock = json.loads((ui_root / "cytoscape.lock.json").read_text(encoding="utf-8"))
    asset_hash = _file_sha256(ui_root / "cytoscape.min.js")
    if lock.get("sha256") != asset_hash or not isinstance(lock.get("version"), str):
        raise ValueError("vendored Cytoscape bytes differ from their frozen lock")
    return asset_hash, lock["version"]


def _observation(cell: _Cell) -> GeometryObservationBinding:
    return GeometryObservationBinding(
        source_block=cell.source_block,
        intended_unit_hash=cell.intended.content_hash,
        unit_id=cell.intended.unit_id,
        source_result_hash=cell.source_result_hash,
        world_id=cell.intended.world_id,
        context_id=cell.intended.context_id,
        condition=cell.intended.condition,
        seed_block=cell.intended.seed_block,
    )


def prepare_phase4_geometry_sources(
    *,
    repository: Path,
    configuration_path: Path,
    held_out_root: Path,
    scorer_bridge_path: Path,
    combined_root: Path,
    review_root: Path,
    benchmark_root: Path,
    ledger_path: Path,
    artifact_root: Path,
    source_association_path: Path,
    output_root: Path,
    prepared_at: datetime,
) -> GeometrySourceManifest:
    """Reproduce all valid-success sources and freeze one DTO per projection hash."""

    if prepared_at.tzinfo is None or prepared_at.utcoffset() is None:
        raise ValueError("geometry source preparation timestamp must be timezone-aware")
    prepared, ledger = _prepare_inputs(
        repository=repository,
        configuration_path=configuration_path,
        held_out_root=held_out_root,
        scorer_bridge_path=scorer_bridge_path,
        combined_root=combined_root,
        review_root=review_root,
        benchmark_root=benchmark_root,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        source_association_path=source_association_path,
    )
    try:
        visualization_configuration = load_visualization_configuration(
            repository / "configs/study/visualization.json"
        )
        grouped: dict[str, list[_Cell]] = defaultdict(list)
        for cell in (*prepared.primary_cells, *prepared.combined_cells):
            if (
                cell.outcome is RunOutcome.SUCCEEDED
                and cell.projection is not None
                and projection_is_structurally_valid(cell.projection)
                and projection_is_content_bearing(cell.projection)
            ):
                grouped[cell.projection.content_hash].append(cell)

        entries = []
        bundles: dict[str, VisualizationBundle] = {}
        for projection_hash in sorted(grouped):
            cells = sorted(grouped[projection_hash], key=lambda item: item.intended.content_hash)
            first = cells[0]
            assert first.projection is not None
            bundle = build_visualization_bundle(
                first.projection,
                first.context,
                first.packet,
                visualization_config=visualization_configuration,
                content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
            )
            audit = _grounding_audit(first.projection, first.scorer_plan)
            entry = build_geometry_source_entry(
                projection=first.projection,
                bundle=bundle,
                scorer_plan=first.scorer_plan,
                grounding_audit=audit,
                observations=tuple(_observation(item) for item in cells),
                bundle_relative_path=f"bundles/{projection_hash}.json",
            )
            for duplicate in cells[1:]:
                assert duplicate.projection is not None
                duplicate_bundle = build_visualization_bundle(
                    duplicate.projection,
                    duplicate.context,
                    duplicate.packet,
                    visualization_config=visualization_configuration,
                    content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
                )
                duplicate_audit = _grounding_audit(
                    duplicate.projection,
                    duplicate.scorer_plan,
                )
                duplicate_entry = build_geometry_source_entry(
                    projection=duplicate.projection,
                    bundle=duplicate_bundle,
                    scorer_plan=duplicate.scorer_plan,
                    grounding_audit=duplicate_audit,
                    observations=tuple(_observation(item) for item in cells),
                    bundle_relative_path=f"bundles/{projection_hash}.json",
                )
                if duplicate_entry != entry or duplicate_bundle != bundle:
                    raise ValueError("one projection hash produced conflicting geometry sources")
            entries.append(entry)
            bundles[projection_hash] = bundle

        renderer = prepared.metric_configuration.renderer
        renderer_sources = renderer_source_file_hashes(repository)
        cytoscape_hash, cytoscape_version = _cytoscape_identity(repository)
        manifest = GeometrySourceManifest(
            manifest_id=(
                "phase4-renderer-sources-"
                f"{prepared.held_out_execution.content_hash[:12]}-"
                f"{prepared.combined_execution.content_hash[:12]}"
            ),
            phase4_analysis_configuration_hash=prepared.analysis_configuration.content_hash,
            metric_configuration_hash=prepared.metric_configuration.content_hash,
            phase4_source_binding_hash=canonical_sha256(prepared.source_bindings),
            source_association_hash=prepared.source_association_hash,
            source_tree_hash=prepared.source_tree_hash,
            visualization_configuration_hash=renderer.visualization_configuration_hash,
            renderer_source_files=renderer_sources,
            cytoscape_asset_sha256=cytoscape_hash,
            cytoscape_version=cytoscape_version,
            layout_config_hash=renderer.layout_config_hash,
            style_config_hash=renderer.style_config_hash,
            font_config_hash=renderer.font_config_hash,
            font_family=visualization_configuration.font_family,
            font_base_px=visualization_configuration.font_base_px,
            viewport_hash=renderer.viewport_hash,
            entries=tuple(entries),
            prepared_at=prepared_at,
        )
        write_geometry_source(root=output_root, manifest=manifest, bundles=bundles)
        return manifest
    finally:
        ledger.close()


__all__ = ["prepare_phase4_geometry_sources"]
