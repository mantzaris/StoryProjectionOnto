from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from story_projection_onto.renderer_geometry import (
    _expected_renderer_only_positions,
    capture_with_system_browser,
)
from tests.unit.test_renderer_geometry import nary_mixed_identifier_bundle


@pytest.mark.integration
def test_system_browser_captures_frozen_cytoscape_geometry() -> None:
    node = shutil.which("node")
    browser = next(
        (
            path
            for name in ("google-chrome", "chromium", "chromium-browser")
            if (path := shutil.which(name)) is not None
        ),
        None,
    )
    if node is None or browser is None:
        pytest.skip("renderer geometry capture requires Node.js and Chrome/Chromium")
    bundle = nary_mixed_identifier_bundle()
    root = Path(__file__).resolve().parents[2]

    captures = capture_with_system_browser(
        (bundle,),
        ui_directory=root / "ui",
        driver_path=root / "scripts/capture_renderer_geometry.mjs",
        node_executable=node,
        browser_executable=browser,
        timeout_seconds=45,
    )

    capture = captures[bundle.projection_hash]
    expected_nodes = {item.projection_object_id for item in bundle.state.nodes}
    expected_assertions = {item.projection_assertion_id for item in bundle.state.assertions}
    assert {item[0] for item in capture.positions} == expected_nodes
    assert {item[1] for item in capture.label_semantic_ids} == (
        expected_nodes | expected_assertions
    )
    assert set(capture.interactive_assertion_ids) == expected_assertions
    assert capture.renderer_only_positions == _expected_renderer_only_positions(bundle)
    assert len(capture.renderer_only_positions) == 2
    assert len({(item.x, item.y) for _, item in capture.renderer_only_positions}) == 2
    assert tuple(item[0] for item in capture.positions) == tuple(sorted(expected_nodes))
    assert tuple(item.label_id for item in capture.label_rectangles) == tuple(
        sorted(item.label_id for item in capture.label_rectangles)
    )
    assert len(capture.label_rectangles) == 9
    assert len(capture.label_semantic_ids) == 9
    assert capture.renderer_runtime.font_probe_css.replace(" ", "") == (
        "14pxsystem-ui,sans-serif"
    )
    assert tuple(item.element_kind for item in capture.renderer_runtime.typography_styles) == (
        "edge",
        "hub",
        "node",
    )
    assert capture.renderer_runtime.container_width == bundle.state.viewport.width
    assert capture.renderer_runtime.container_height == bundle.state.viewport.height
    assert capture.renderer_runtime.device_pixel_ratio == 1.0
