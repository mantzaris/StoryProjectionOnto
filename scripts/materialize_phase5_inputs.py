#!/usr/bin/env python3
"""Materialize production Phase 5 inputs from verified restricted artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.phase5_execution import Phase5ExecutionError
from story_projection_onto.phase5_production import (
    RestrictedPhase5CAS,
    load_phase5_materialization_source,
    materialize_phase5_inputs_to_directory,
)
from story_projection_onto.store import ArtifactStore, BlobStore, Compression, Ledger


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Reproduce the review-gated Phase 5 input manifest from restricted CAS exports."
        )
    )
    command.add_argument("--source-manifest", type=Path, required=True)
    command.add_argument("--primary-results-gate", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--compression", choices=("zstd", "gzip"), default="zstd")
    command.add_argument("--benchmark-root", type=Path, default=Path("data/synthetic"))
    command.add_argument(
        "--review-completion-root",
        type=Path,
        default=Path("artifacts/restricted/scorer_only/independent_review"),
    )
    command.add_argument("--protocol", type=Path, default=Path("configs/study/feedback.json"))
    command.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/restricted/phase5_inputs"),
    )
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    if not options.ledger.is_file() or options.ledger.is_symlink():
        print(json.dumps({"state": "blocked", "error": "ledger is absent or symlinked"}))
        return 2
    if not options.artifact_root.is_dir() or options.artifact_root.is_symlink():
        print(json.dumps({"state": "blocked", "error": "CAS root is absent or symlinked"}))
        return 2
    ledger = Ledger(options.ledger)
    try:
        source = load_phase5_materialization_source(options.source_manifest)
        protocol = load_feedback_protocol(options.protocol)
        artifacts = ArtifactStore(
            BlobStore(
                options.artifact_root,
                compression=Compression(options.compression),
            ),
            ledger,
        )
        receipt = materialize_phase5_inputs_to_directory(
            source=source,
            protocol=protocol,
            primary_results_gate_path=options.primary_results_gate,
            benchmark_root=options.benchmark_root,
            review_completion_root=options.review_completion_root,
            cas=RestrictedPhase5CAS(artifacts),
            output_root=options.output_root,
        )
    except (Phase5ExecutionError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    finally:
        ledger.close()
    print(
        json.dumps(
            {
                "state": "materialized",
                "input_manifest_hash": receipt.input_manifest_hash,
                "materialization_receipt_hash": receipt.content_hash,
                "episode_count": len(receipt.episodes),
                "scorer_gold_read": receipt.scorer_gold_read,
                "model_service_called": receipt.model_service_called,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
