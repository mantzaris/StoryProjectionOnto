#!/usr/bin/env python3
"""Prepare, capture, and replay the frozen renderer geometry required by Phase 4."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from story_projection_onto.renderer_geometry import (
    RendererGeometryError,
    capture_with_system_browser,
    load_existing_renderer_captures,
    load_geometry_source,
    materialize_geometry_captures,
    preflight_geometry_materialization,
    renderer_source_file_hashes,
    replay_geometry_materialization,
)
from story_projection_onto.scorer_only.geometry_sources import (
    prepare_phase4_geometry_sources,
)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Capture real Cytoscape geometry without opening the model runtime."
    )
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--replay", action="store_true")
    command.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/restricted/scorer_only/phase4_geometry_source"),
    )
    command.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("artifacts/public/phase4_geometry"),
    )
    command.add_argument("--prepared-at", type=_timestamp)
    command.add_argument("--completed-at", type=_timestamp)
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument(
        "--configuration",
        type=Path,
        default=Path("configs/study/phase4_analysis.json"),
    )
    command.add_argument("--held-out-root", type=Path)
    command.add_argument("--scorer-bridge", type=Path)
    command.add_argument("--combined-root", type=Path)
    command.add_argument("--review-root", type=Path)
    command.add_argument("--benchmark-root", type=Path, default=Path("data/synthetic"))
    command.add_argument("--ledger", type=Path)
    command.add_argument("--artifact-root", type=Path)
    command.add_argument("--source-association", type=Path)
    command.add_argument("--node-executable")
    command.add_argument("--browser-executable")
    command.add_argument("--browser-timeout-seconds", type=float, default=300.0)
    return command


def _required(options: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(options, name) is None]
    if missing:
        raise RendererGeometryError(
            "mode requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    try:
        if options.prepare:
            _required(
                options,
                (
                    "prepared_at",
                    "held_out_root",
                    "scorer_bridge",
                    "combined_root",
                    "review_root",
                    "ledger",
                    "artifact_root",
                    "source_association",
                ),
            )
            manifest = prepare_phase4_geometry_sources(
                repository=options.repository,
                configuration_path=options.configuration,
                held_out_root=options.held_out_root,
                scorer_bridge_path=options.scorer_bridge,
                combined_root=options.combined_root,
                review_root=options.review_root,
                benchmark_root=options.benchmark_root,
                ledger_path=options.ledger,
                artifact_root=options.artifact_root,
                source_association_path=options.source_association,
                output_root=options.source_root,
                prepared_at=options.prepared_at,
            )
            summary = {
                "state": "prepared",
                "source_manifest_hash": manifest.content_hash,
                "projection_count": len(manifest.entries),
                "independent_confirmatory_unit": manifest.independent_confirmatory_unit,
                "model_service_called": False,
            }
        elif options.replay:
            receipt = replay_geometry_materialization(
                source_root=options.source_root,
                geometry_root=options.geometry_root,
            )
            summary = {
                "state": "replayed",
                "receipt_hash": receipt.content_hash,
                "geometry_count": len(receipt.artifacts),
                "independent_confirmatory_unit": receipt.independent_confirmatory_unit,
                "model_service_called": False,
            }
        elif options.validate_only:
            preflight = preflight_geometry_materialization(
                source_root=options.source_root,
                geometry_root=options.geometry_root,
                node_executable=options.node_executable,
                browser_executable=options.browser_executable,
            )
            manifest, _ = load_geometry_source(options.source_root)
            if (
                preflight.missing_capture_hashes
                and renderer_source_file_hashes(options.repository)
                != manifest.renderer_source_files
            ):
                raise RendererGeometryError("renderer sources changed after geometry preparation")
            summary = {
                "state": "ready" if preflight.ready_to_capture else "blocked",
                "source_manifest_hash": preflight.source_manifest_hash,
                "expected_geometry_count": preflight.expected_geometry_count,
                "exact_existing_geometry_count": preflight.exact_existing_geometry_count,
                "missing_geometry_count": len(preflight.missing_projection_hashes),
                "exact_existing_capture_count": preflight.exact_existing_capture_count,
                "missing_capture_count": len(preflight.missing_capture_hashes),
                "browser_capture_available": preflight.browser_capture_available,
                "receipt_present": preflight.replay_receipt_present,
                "model_service_called": False,
            }
            print(json.dumps(summary, sort_keys=True))
            return 0 if preflight.ready_to_capture else 2
        else:
            _required(options, ("completed_at",))
            preflight = preflight_geometry_materialization(
                source_root=options.source_root,
                geometry_root=options.geometry_root,
                node_executable=options.node_executable,
                browser_executable=options.browser_executable,
            )
            if not preflight.ready_to_capture:
                raise RendererGeometryError("renderer geometry preflight is not ready")
            manifest, bundles_by_hash = load_geometry_source(options.source_root)
            ui_root = options.repository / "ui"
            captures = load_existing_renderer_captures(
                source_root=options.source_root,
                geometry_root=options.geometry_root,
            )
            missing_bundles = tuple(
                bundles_by_hash[item.projection_hash]
                for item in manifest.entries
                if item.projection_hash not in captures
            )
            if missing_bundles:
                if renderer_source_file_hashes(options.repository) != (
                    manifest.renderer_source_files
                ):
                    raise RendererGeometryError(
                        "renderer sources changed after geometry preparation"
                    )
                captures.update(
                    capture_with_system_browser(
                        missing_bundles,
                        ui_directory=ui_root,
                        driver_path=options.repository / "scripts/capture_renderer_geometry.mjs",
                        node_executable=options.node_executable,
                        browser_executable=options.browser_executable,
                        timeout_seconds=options.browser_timeout_seconds,
                    )
                )
            receipt = materialize_geometry_captures(
                source_root=options.source_root,
                geometry_root=options.geometry_root,
                captures=captures,
                completed_at=options.completed_at,
            )
            replay_geometry_materialization(
                source_root=options.source_root,
                geometry_root=options.geometry_root,
            )
            summary = {
                "state": "materialized",
                "receipt_hash": receipt.content_hash,
                "geometry_count": len(receipt.artifacts),
                "renderer_runtime_hash": receipt.renderer_runtime_hash,
                "independent_confirmatory_unit": receipt.independent_confirmatory_unit,
                "model_service_called": False,
            }
    except (RendererGeometryError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
