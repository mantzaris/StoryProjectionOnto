from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    RoleBinding,
    RunOutcome,
    Viewport,
    canonical_sha256,
)
from story_projection_onto.renderer_geometry import (
    GeometryObservationBinding,
    GeometrySourceEntry,
    GeometrySourceManifest,
    RendererGeometryError,
    RendererRawCapture,
    RendererRuntimeIdentity,
    RendererTypographyStyle,
    _expected_label_semantic_ids,
    _expected_renderer_only_positions,
    _renderer_hub_model_positions,
    load_existing_renderer_captures,
    materialize_geometry_captures,
    materialize_geometry_record,
    preflight_geometry_materialization,
    renderer_source_file_hashes,
    replay_geometry_materialization,
    write_geometry_source,
)
from story_projection_onto.scorer_only.phase4_analysis import (
    Phase4AnalysisError,
    _replay_geometry_prerequisite,
    _validate_geometry_materialization_receipt,
)
from story_projection_onto.ui import (
    SemanticDisplayMode,
    VisualizationBundle,
    VisualizationContentScope,
    build_visualization_bundle,
)
from tests.unit.test_ui import context, packet, projection, semantic_overlay

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "a" * 64


def _copy_record(record, **updates):
    payload = record.model_dump(mode="python", exclude={"content_hash"})
    payload.update(updates)
    return type(record).model_validate(payload)


def nary_mixed_identifier_bundle() -> VisualizationBundle:
    """A renderer-only fixture with two n-ary assertions and legal ASCII edge cases."""

    query_context = context()
    evidence_packet = packet()
    base = build_visualization_bundle(
        projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet),
        query_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
    )
    semantic_node_ids = ("A-node", "A.Node", "a/node")
    node_semantics = {
        item.visualization_node_id: semantic_node_ids[index]
        for index, item in enumerate(base.state.nodes)
    }
    nodes = tuple(
        _copy_record(item, projection_object_id=node_semantics[item.visualization_node_id])
        for item in base.state.nodes
    )
    node_details = tuple(
        _copy_record(item, projection_object_id=node_semantics[item.visualization_node_id])
        for item in base.node_details
    )
    visual_node_ids = tuple(item.visualization_node_id for item in base.state.nodes)
    semantic_assertion_ids = ("A/assertion", "a.assertion")
    assertions = []
    assertion_details = []
    details_by_visual_id = {
        item.visualization_assertion_id: item for item in base.assertion_details
    }
    for index, item in enumerate(base.state.assertions):
        roles = (
            RoleBinding(
                role=f"role-{index}-A",
                object_id=visual_node_ids[index],
                evidence_ids=item.evidence_badge.evidence_ids,
            ),
            RoleBinding(
                role=f"role-{index}.b",
                object_id=visual_node_ids[index + 1],
                evidence_ids=item.evidence_badge.evidence_ids,
            ),
        )
        semantic_id = semantic_assertion_ids[index]
        assertions.append(
            _copy_record(
                item,
                projection_assertion_id=semantic_id,
                source_visualization_node_id=None,
                target_visualization_node_id=None,
                roles=roles,
                why_matters_assertion_id=semantic_id,
            )
        )
        assertion_details.append(
            _copy_record(
                details_by_visual_id[item.visualization_assertion_id],
                projection_assertion_id=semantic_id,
                role_labels=tuple(role.role for role in roles),
            )
        )
    state = _copy_record(base.state, nodes=nodes, assertions=tuple(assertions))
    display_selection = _copy_record(
        base.display_selection,
        projection_node_ids=tuple(sorted(semantic_node_ids)),
        projection_assertion_ids=tuple(sorted(semantic_assertion_ids)),
    )
    return _copy_record(
        base,
        state=state,
        node_details=node_details,
        assertion_details=tuple(assertion_details),
        display_selection=display_selection,
    )


def _fixture():
    query_context = context()
    evidence_packet = packet()
    ontology_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
    )
    bundle = build_visualization_bundle(
        ontology_projection,
        query_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
    )
    payload = (bundle.to_canonical_json() + "\n").encode()
    node_ids = tuple(sorted(item.projection_object_id for item in bundle.state.nodes))
    assertion_ids = tuple(sorted(item.projection_assertion_id for item in bundle.state.assertions))
    observation = GeometryObservationBinding(
        source_block="primary",
        intended_unit_hash="b" * 64,
        unit_id="primary:test-world:test-context:C2:s1",
        source_result_hash="c" * 64,
        world_id="test-world",
        context_id=query_context.context_id,
        condition=ConditionName.C2_LLM_QUERY,
        seed_block=1,
    )
    entry = GeometrySourceEntry(
        projection_hash=ontology_projection.content_hash,
        projection_id=ontology_projection.projection_id,
        condition=ontology_projection.condition,
        world_id=observation.world_id,
        context_id=query_context.context_id,
        context_hash=query_context.content_hash,
        packet_hash=evidence_packet.content_hash,
        scorer_plan_hash="d" * 64,
        visualization_bundle_hash=bundle.content_hash,
        visualization_state_hash=bundle.state.content_hash,
        bundle_relative_path=f"bundles/{ontology_projection.content_hash}.json",
        bundle_file_sha256=hashlib.sha256(payload).hexdigest(),
        layout_config_hash=bundle.state.layout_config_hash,
        style_config_hash=bundle.state.style_config_hash,
        font_config_hash=bundle.state.font_config_hash,
        viewport_hash=bundle.state.viewport.content_hash,
        projection_node_ids=node_ids,
        projection_assertion_ids=assertion_ids,
        irrelevant_semantic_ids=assertion_ids[-1:],
        rare_pivotal_matches=(),
        observations=(observation,),
    )
    lock = json.loads((ROOT / "ui/cytoscape.lock.json").read_text(encoding="utf-8"))
    manifest = GeometrySourceManifest(
        manifest_id="geometry-source-test",
        phase4_analysis_configuration_hash=DIGEST,
        metric_configuration_hash="e" * 64,
        phase4_source_binding_hash=canonical_sha256(()),
        source_association_hash="1" * 64,
        source_tree_hash="2" * 64,
        visualization_configuration_hash="3" * 64,
        renderer_source_files=renderer_source_file_hashes(ROOT),
        cytoscape_asset_sha256=lock["sha256"],
        cytoscape_version=lock["version"],
        layout_config_hash=entry.layout_config_hash,
        style_config_hash=entry.style_config_hash,
        font_config_hash=entry.font_config_hash,
        viewport_hash=entry.viewport_hash,
        entries=(entry,),
        prepared_at=datetime(2026, 9, 4, 12, tzinfo=UTC),
    )
    semantic_by_visual = {
        item.visualization_node_id: item.projection_object_id for item in bundle.state.nodes
    }
    assertion_by_visual = {
        item.visualization_assertion_id: item.projection_assertion_id
        for item in bundle.state.assertions
    }
    viewport = bundle.state.viewport
    positions = tuple(
        sorted(
            (
                semantic_by_visual[item.visualization_node_id],
                {
                    "x": (item.x - viewport.center_x) * viewport.zoom + viewport.width / 2,
                    "y": (item.y - viewport.center_y) * viewport.zoom + viewport.height / 2,
                },
            )
            for item in bundle.state.positions
        )
    )
    visible_ids = tuple(
        sorted(
            [semantic_by_visual[item] for item in bundle.state.visible_node_ids]
            + [assertion_by_visual[item] for item in bundle.state.visible_assertion_ids]
        )
    )
    rectangles = tuple(
        {
            "label_id": semantic_id,
            "left": float(index * 20),
            "top": 0.0,
            "right": float(index * 20 + 10),
            "bottom": 10.0,
        }
        for index, semantic_id in enumerate(visible_ids)
    )
    capture = RendererRawCapture(
        projection_hash=ontology_projection.content_hash,
        visualization_bundle_hash=bundle.content_hash,
        visualization_state_hash=bundle.state.content_hash,
        layout_config_hash=bundle.state.layout_config_hash,
        style_config_hash=bundle.state.style_config_hash,
        font_config_hash=bundle.state.font_config_hash,
        viewport=bundle.state.viewport,
        temporal_filter_hash=bundle.state.temporal_filter.content_hash,
        positions=positions,
        renderer_only_positions=(),
        label_rectangles=rectangles,
        label_semantic_ids=tuple(
            (item["label_id"], item["label_id"]) for item in rectangles
        ),
        visible_semantic_ids=visible_ids,
        interactive_assertion_ids=assertion_ids,
        renderer_runtime=RendererRuntimeIdentity(
            cytoscape_version=lock["version"],
            user_agent="geometry-test-browser/1",
            platform="test-platform",
            device_pixel_ratio=1.0,
            container_width=viewport.width,
            container_height=viewport.height,
            font_probe_css="14px system-ui, sans-serif",
            font_probe_widths=(100.0, 80.0, 60.0),
            typography_styles=tuple(
                RendererTypographyStyle(element_kind=kind)
                for kind in ("edge", "hub", "node")
            ),
        ),
    )
    return bundle, entry, manifest, capture


def _nary_fixture():
    bundle = nary_mixed_identifier_bundle()
    _, base_entry, base_manifest, base_capture = _fixture()
    payload = (bundle.to_canonical_json() + "\n").encode()
    node_ids = tuple(sorted(item.projection_object_id for item in bundle.state.nodes))
    assertion_ids = tuple(
        sorted(item.projection_assertion_id for item in bundle.state.assertions)
    )
    entry = _copy_record(
        base_entry,
        projection_hash=bundle.projection_hash,
        projection_id=bundle.projection_id,
        context_hash=bundle.context_hash,
        packet_hash=bundle.packet_hash,
        visualization_bundle_hash=bundle.content_hash,
        visualization_state_hash=bundle.state.content_hash,
        bundle_relative_path=f"bundles/{bundle.projection_hash}.json",
        bundle_file_sha256=hashlib.sha256(payload).hexdigest(),
        projection_node_ids=node_ids,
        projection_assertion_ids=assertion_ids,
        irrelevant_semantic_ids=(),
    )
    manifest = _copy_record(base_manifest, entries=(entry,))
    semantic_by_visual = {
        item.visualization_node_id: item.projection_object_id for item in bundle.state.nodes
    }
    viewport = bundle.state.viewport
    positions = tuple(
        sorted(
            (
                semantic_by_visual[item.visualization_node_id],
                {
                    "x": (item.x - viewport.center_x) * viewport.zoom + viewport.width / 2,
                    "y": (item.y - viewport.center_y) * viewport.zoom + viewport.height / 2,
                },
            )
            for item in bundle.state.positions
        )
    )
    label_bindings = _expected_label_semantic_ids(bundle)
    rectangles = tuple(
        {
            "label_id": component_id,
            "left": float(index * 20),
            "top": 0.0,
            "right": float(index * 20 + 10),
            "bottom": 10.0,
        }
        for index, (component_id, _) in enumerate(label_bindings)
    )
    capture = _copy_record(
        base_capture,
        projection_hash=bundle.projection_hash,
        visualization_bundle_hash=bundle.content_hash,
        visualization_state_hash=bundle.state.content_hash,
        positions=positions,
        renderer_only_positions=_expected_renderer_only_positions(bundle),
        label_rectangles=rectangles,
        label_semantic_ids=label_bindings,
        visible_semantic_ids=tuple(sorted((*node_ids, *assertion_ids))),
        interactive_assertion_ids=assertion_ids,
    )
    return bundle, entry, manifest, capture


def _write_materialized_fixture(tmp_path: Path):
    bundle, entry, manifest, capture = _fixture()
    source_root = tmp_path / "restricted-source"
    geometry_root = tmp_path / "geometry"
    write_geometry_source(
        root=source_root,
        manifest=manifest,
        bundles={entry.projection_hash: bundle},
    )
    receipt = materialize_geometry_captures(
        source_root=source_root,
        geometry_root=geometry_root,
        captures={entry.projection_hash: capture},
        completed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )
    return source_root, geometry_root, entry, receipt


def test_materialization_writes_one_projection_record_and_replays(tmp_path: Path) -> None:
    bundle, entry, manifest, capture = _fixture()
    source_root = tmp_path / "restricted-source"
    geometry_root = tmp_path / "geometry"
    write_geometry_source(
        root=source_root,
        manifest=manifest,
        bundles={entry.projection_hash: bundle},
    )

    receipt = materialize_geometry_captures(
        source_root=source_root,
        geometry_root=geometry_root,
        captures={entry.projection_hash: capture},
        completed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )
    replayed = replay_geometry_materialization(
        source_root=source_root,
        geometry_root=geometry_root,
    )

    geometry = json.loads((geometry_root / f"{entry.projection_hash}.json").read_text())
    assert replayed == receipt
    assert len(receipt.artifacts) == 1
    assert receipt.independent_confirmatory_unit == "world"
    assert receipt.contexts_are_independent_samples is False
    assert geometry["projection_hash"] == entry.projection_hash
    assert geometry["materialization_source_hash"] == manifest.content_hash
    assert geometry["source_result_hashes"] == ["c" * 64]
    assert geometry["renderer_runtime_hash"] == capture.renderer_runtime.content_hash
    assert geometry["irrelevant_semantic_ids"] == list(entry.irrelevant_semantic_ids)
    assert (geometry_root / receipt.artifacts[0].content_addressed_relative_path).read_bytes() == (
        geometry_root / f"{entry.projection_hash}.json"
    ).read_bytes()
    preflight = preflight_geometry_materialization(
        source_root=source_root,
        geometry_root=geometry_root,
        node_executable="/definitely/absent/node",
        browser_executable="/definitely/absent/chrome",
    )
    assert preflight.ready_to_capture is True
    assert preflight.browser_capture_available is False
    assert preflight.exact_existing_capture_count == 1
    assert preflight.missing_capture_hashes == ()


def test_confirmatory_geometry_rejects_raw_and_posthoc_filtered_views() -> None:
    registered, entry, manifest, capture = _fixture()
    query_context = context()
    evidence_packet = packet()
    source = projection(ConditionName.C2_LLM_QUERY, query_context, evidence_packet)
    raw = build_visualization_bundle(source, query_context, evidence_packet)
    supported_only = build_visualization_bundle(
        source,
        query_context,
        evidence_packet,
        content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        semantic_overlay=semantic_overlay(source, query_context, evidence_packet),
        semantic_display_mode=SemanticDisplayMode.SUPPORTED_ONLY,
    )

    assert registered.display_selection.content_scope is (
        VisualizationContentScope.REGISTERED_DISPLAY
    )
    for disallowed in (raw, supported_only):
        with pytest.raises(RendererGeometryError, match="unfiltered registered-display"):
            materialize_geometry_record(
                source_manifest_hash=manifest.content_hash,
                entry=entry,
                bundle=disallowed,
                capture=capture,
                expected_cytoscape_version=manifest.cytoscape_version,
            )


def test_nary_hubs_are_distinct_nonzero_and_replayed_with_component_labels(
    tmp_path: Path,
) -> None:
    bundle, entry, manifest, capture = _nary_fixture()
    model_positions = _renderer_hub_model_positions(bundle)
    assert len(model_positions) == 2
    assert len({(item.x, item.y) for _, item in model_positions}) == 2
    assert all((item.x, item.y) != (0.0, 0.0) for _, item in model_positions)
    assert model_positions == _renderer_hub_model_positions(bundle)

    source_root = tmp_path / "restricted-nary-source"
    geometry_root = tmp_path / "nary-geometry"
    write_geometry_source(
        root=source_root,
        manifest=manifest,
        bundles={entry.projection_hash: bundle},
    )
    materialize_geometry_captures(
        source_root=source_root,
        geometry_root=geometry_root,
        captures={entry.projection_hash: capture},
        completed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )

    replay_geometry_materialization(
        source_root=source_root,
        geometry_root=geometry_root,
    )
    geometry = json.loads((geometry_root / f"{entry.projection_hash}.json").read_text())
    assert len(geometry["renderer_only_positions"]) == 2
    assert len(geometry["label_rectangles"]) == 9
    assertion_components = [
        component
        for component, semantic in geometry["label_semantic_ids"]
        if semantic in entry.projection_assertion_ids
    ]
    assert len(assertion_components) == 6
    assert all(component.startswith("render-label:") for component in assertion_components)

    drifted_capture = _copy_record(
        capture,
        renderer_only_positions=(
            (
                capture.renderer_only_positions[0][0],
                {"x": 1.0, "y": 1.0},
            ),
            capture.renderer_only_positions[1],
        ),
    )
    with pytest.raises(RendererGeometryError, match="n-ary hub position"):
        materialize_geometry_record(
            source_manifest_hash=manifest.content_hash,
            entry=entry,
            bundle=bundle,
            capture=drifted_capture,
            expected_cytoscape_version=manifest.cytoscape_version,
        )


def test_empty_success_inventory_materializes_without_a_browser(tmp_path: Path) -> None:
    _, _, manifest, _ = _fixture()
    empty_manifest = GeometrySourceManifest.model_validate(
        {
            **manifest.model_dump(mode="json", exclude={"content_hash", "entries"}),
            "manifest_id": "geometry-source-empty-test",
            "entries": [],
        }
    )
    source_root = tmp_path / "restricted-empty-source"
    geometry_root = tmp_path / "empty-geometry"
    write_geometry_source(root=source_root, manifest=empty_manifest, bundles={})

    preflight = preflight_geometry_materialization(
        source_root=source_root,
        geometry_root=geometry_root,
        node_executable="/definitely/absent/node",
        browser_executable="/definitely/absent/chrome",
    )
    receipt = materialize_geometry_captures(
        source_root=source_root,
        geometry_root=geometry_root,
        captures={},
        completed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )

    assert preflight.ready_to_capture is True
    assert preflight.expected_geometry_count == 0
    assert receipt.artifacts == ()
    assert receipt.renderer_runtime_hash is None
    assert (
        replay_geometry_materialization(
            source_root=source_root,
            geometry_root=geometry_root,
        )
        == receipt
    )


def test_interrupted_raw_capture_is_reused_without_browser(tmp_path: Path) -> None:
    bundle, entry, manifest, capture = _fixture()
    source_root = tmp_path / "restricted-source"
    geometry_root = tmp_path / "interrupted-geometry"
    write_geometry_source(
        root=source_root,
        manifest=manifest,
        bundles={entry.projection_hash: bundle},
    )
    raw_root = geometry_root / "raw"
    raw_root.mkdir(parents=True)
    (raw_root / f"{entry.projection_hash}.json").write_text(
        capture.to_canonical_json() + "\n",
        encoding="utf-8",
    )

    preflight = preflight_geometry_materialization(
        source_root=source_root,
        geometry_root=geometry_root,
        node_executable="/definitely/absent/node",
        browser_executable="/definitely/absent/chrome",
    )
    captures = load_existing_renderer_captures(
        source_root=source_root,
        geometry_root=geometry_root,
    )
    receipt = materialize_geometry_captures(
        source_root=source_root,
        geometry_root=geometry_root,
        captures=captures,
        completed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )

    assert preflight.ready_to_capture is True
    assert preflight.missing_projection_hashes == (entry.projection_hash,)
    assert preflight.missing_capture_hashes == ()
    assert len(receipt.artifacts) == 1


def test_capture_fails_closed_on_viewport_or_result_lineage_drift() -> None:
    bundle, entry, manifest, capture = _fixture()
    bad_viewport = Viewport(
        center_x=capture.viewport.center_x,
        center_y=capture.viewport.center_y,
        zoom=capture.viewport.zoom,
        width=capture.viewport.width + 1,
        height=capture.viewport.height,
    )
    bad = RendererRawCapture(
        **capture.model_dump(
            mode="python",
            exclude={"content_hash", "viewport"},
        ),
        viewport=bad_viewport,
    )

    with pytest.raises(RendererGeometryError, match="different source or renderer"):
        materialize_geometry_record(
            source_manifest_hash=manifest.content_hash,
            entry=entry,
            bundle=bundle,
            capture=bad,
            expected_cytoscape_version=manifest.cytoscape_version,
        )

    bad_filter = RendererRawCapture(
        **capture.model_dump(
            mode="python",
            exclude={"content_hash", "temporal_filter_hash"},
        ),
        temporal_filter_hash="f" * 64,
    )
    with pytest.raises(RendererGeometryError, match="disclosure/filter state"):
        materialize_geometry_record(
            source_manifest_hash=manifest.content_hash,
            entry=entry,
            bundle=bundle,
            capture=bad_filter,
            expected_cytoscape_version=manifest.cytoscape_version,
        )


@pytest.mark.parametrize(
    "drift",
    ("missing-source", "missing-raw", "source-entry-hash", "raw-logical-hash"),
)
def test_phase4_replay_prerequisite_rejects_source_and_raw_lineage_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    source_root, geometry_root, entry, receipt = _write_materialized_fixture(tmp_path)
    if drift == "missing-source":
        (source_root / "source_manifest.json").unlink()
    elif drift == "missing-raw":
        (geometry_root / "raw" / f"{entry.projection_hash}.json").unlink()
    else:
        artifact = receipt.artifacts[0]
        if drift == "source-entry-hash":
            artifact = _copy_record(artifact, source_entry_hash="f" * 64)
        else:
            artifact = _copy_record(artifact, raw_capture_hash="f" * 64)
        changed_receipt = _copy_record(receipt, artifacts=(artifact,))
        (geometry_root / "materialization_receipt.json").write_text(
            changed_receipt.to_canonical_json() + "\n",
            encoding="utf-8",
        )

    with pytest.raises(Phase4AnalysisError, match="replay prerequisite failed"):
        _replay_geometry_prerequisite(
            geometry_source_root=source_root,
            geometry_root=geometry_root,
        )

def test_phase4_revalidates_receipt_against_successful_result_inventory(
    tmp_path: Path,
) -> None:
    bundle, entry, manifest, capture = _fixture()
    ontology_projection = projection(
        ConditionName.C2_LLM_QUERY,
        context(),
        packet(),
    )
    assert ontology_projection.content_hash == entry.projection_hash
    source_root = tmp_path / "restricted-source"
    geometry_root = tmp_path / "geometry"
    write_geometry_source(
        root=source_root,
        manifest=manifest,
        bundles={entry.projection_hash: bundle},
    )
    expected = materialize_geometry_captures(
        source_root=source_root,
        geometry_root=geometry_root,
        captures={entry.projection_hash: capture},
        completed_at=datetime(2026, 9, 4, 13, tzinfo=UTC),
    )
    prepared = SimpleNamespace(
        primary_cells=(
                SimpleNamespace(
                    outcome=RunOutcome.SUCCEEDED,
                    projection=ontology_projection,
                    source_result_hash="c" * 64,
                ),
        ),
        combined_cells=(),
        analysis_configuration=SimpleNamespace(content_hash=DIGEST),
        metric_configuration=SimpleNamespace(content_hash="e" * 64),
        source_bindings=(),
        source_association_hash="1" * 64,
        source_tree_hash="2" * 64,
    )

    observed = _validate_geometry_materialization_receipt(
        prepared=prepared,
        geometry_source_root=source_root,
        geometry_root=geometry_root,
    )

    assert observed == expected


def test_source_manifest_hash_changes_with_source_result_but_not_context_count() -> None:
    _, entry, manifest, _ = _fixture()
    changed_observation = entry.observations[0].model_copy(update={"source_result_hash": "9" * 64})
    changed_entry = GeometrySourceEntry.model_validate(
        {
            **entry.model_dump(mode="json", exclude={"content_hash", "observations"}),
            "observations": [changed_observation.model_dump(mode="json", exclude={"content_hash"})],
        }
    )
    changed_manifest = GeometrySourceManifest.model_validate(
        {
            **manifest.model_dump(mode="json", exclude={"content_hash", "entries"}),
            "entries": [changed_entry.model_dump(mode="json", exclude={"content_hash"})],
        }
    )

    assert changed_manifest.content_hash != manifest.content_hash
    assert changed_manifest.independent_confirmatory_unit == "world"
    assert canonical_sha256(changed_manifest.entries[0].observations) != canonical_sha256(
        entry.observations
    )
