#!/usr/bin/env python3
"""Validate or serve the capture-only three-trace Phase 5 interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.phase5_capture import (
    materialize_capture_server_manifest,
    prepare_researcher_trace_capture_app,
)
from story_projection_onto.phase5_sources import Phase5SourceProductionError
from story_projection_onto.store import ArtifactStore, BlobStore, Compression, Ledger


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument("--restricted-root", type=Path, required=True)
    command.add_argument("--primary-results-gate", type=Path, required=True)
    command.add_argument("--held-out-journal-root", type=Path, required=True)
    command.add_argument("--capture-root", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--protocol", type=Path, default=Path("configs/study/feedback.json"))
    command.add_argument("--compression", choices=("zstd", "gzip"), default="zstd")
    command.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    command.add_argument("--port", type=int, default=8765)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--serve", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    ledger: Ledger | None = None
    try:
        if not 1024 <= options.port <= 65535:
            raise Phase5SourceProductionError("capture server port must be in [1024, 65535]")
        repository = options.repository.resolve(strict=True)

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else repository / path

        restricted_root = resolve(options.restricted_root).resolve(strict=True)
        ledger_path = resolve(options.ledger)
        artifact_root = resolve(options.artifact_root)
        if not ledger_path.is_file() or ledger_path.is_symlink():
            raise Phase5SourceProductionError("held-out ledger is absent or symlinked")
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise Phase5SourceProductionError("held-out CAS root is absent or symlinked")
        ledger = Ledger(ledger_path)
        artifacts = ArtifactStore(
            BlobStore(artifact_root, compression=Compression(options.compression)),
            ledger,
        )
        application, manifest = prepare_researcher_trace_capture_app(
            repository=repository,
            restricted_root=restricted_root,
            primary_results_gate_path=resolve(options.primary_results_gate),
            held_out_journal_root=resolve(options.held_out_journal_root),
            capture_root=resolve(options.capture_root),
            protocol=load_feedback_protocol(resolve(options.protocol)),
            artifacts=artifacts,
        )
        print(
            json.dumps(
                {
                    "state": "ready_to_serve" if options.serve else "validated",
                    "capture_server_manifest_hash": manifest.content_hash,
                    "trace_count": len(manifest.bindings),
                    "capture_only": manifest.capture_only,
                    "revision_runner_configured": manifest.revision_runner_configured,
                    "model_service_called": manifest.model_service_called,
                },
                sort_keys=True,
            )
        )
        if not options.serve:
            return 0
        materialize_capture_server_manifest(
            repository=repository,
            restricted_root=restricted_root,
            capture_root=resolve(options.capture_root),
            manifest=manifest,
        )
        try:
            import uvicorn
        except ImportError as error:
            raise Phase5SourceProductionError(
                "the optional study dependencies are required to serve the capture UI"
            ) from error
        uvicorn.run(application, host=options.host, port=options.port, log_level="info")
        return 0
    except (Phase5SourceProductionError, ValueError, OSError, KeyError) as error:
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "phase5_trace_capture_blocked",
                    "error": str(error),
                    "model_service_called": False,
                },
                sort_keys=True,
            )
        )
        return 2
    finally:
        if ledger is not None:
            ledger.close()


if __name__ == "__main__":
    raise SystemExit(main())
