#!/usr/bin/env python3
"""Compile or validate the 18 restricted known-answer Phase 5 scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.phase5_production import Phase5MaterializationError
from story_projection_onto.scorer_only.phase5_feedback import (
    DEFAULT_PHASE5_SCORING_ROOT,
    Phase5FeedbackScoringError,
    materialize_phase5_feedback_scoring,
    prepare_phase5_feedback_scoring,
)
from story_projection_onto.store import ArtifactStore, BlobStore, Compression, Ledger


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument("--restricted-root", type=Path, required=True)
    command.add_argument("--source-manifest", type=Path, required=True)
    command.add_argument("--known-answer-source", type=Path, required=True)
    command.add_argument("--feedback-journal-root", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--output-root", type=Path, default=DEFAULT_PHASE5_SCORING_ROOT)
    command.add_argument("--compression", choices=("zstd", "gzip"), default="zstd")
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    ledger: Ledger | None = None
    try:
        repository = options.repository.resolve(strict=True)

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else repository / path

        restricted_root = resolve(options.restricted_root).resolve(strict=True)
        ledger_path = resolve(options.ledger)
        artifact_root = resolve(options.artifact_root)
        if not ledger_path.is_file() or ledger_path.is_symlink():
            raise Phase5FeedbackScoringError("scorer ledger is absent or symlinked")
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise Phase5FeedbackScoringError("scorer CAS root is absent or symlinked")
        ledger = Ledger(ledger_path)
        artifacts = ArtifactStore(
            BlobStore(artifact_root, compression=Compression(options.compression)),
            ledger,
        )
        session, metrics, receipt = prepare_phase5_feedback_scoring(
            repository=repository,
            restricted_root=restricted_root,
            source_manifest_path=resolve(options.source_manifest),
            known_answer_source_path=resolve(options.known_answer_source),
            feedback_journal_root=resolve(options.feedback_journal_root),
            output_root=resolve(options.output_root),
            artifacts=artifacts,
        )
        writes_performed = False
        if options.materialize:
            writes_performed = materialize_phase5_feedback_scoring(
                output_root=resolve(options.output_root),
                restricted_root=restricted_root,
                session=session,
                metrics=metrics,
                receipt=receipt,
            )
        print(
            json.dumps(
                {
                    "state": "materialized" if options.materialize else "validated",
                    "writes_performed": writes_performed,
                    "metrics_hash": metrics.content_hash,
                    "receipt_hash": receipt.content_hash,
                    "score_count": len(metrics.records),
                    "researcher_trace_gold_fields": metrics.researcher_trace_gold_fields,
                    "model_service_called": metrics.model_service_called,
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        Phase5FeedbackScoringError,
        Phase5MaterializationError,
        ValueError,
        OSError,
        KeyError,
    ) as error:
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "phase5_scorer_only_compilation_blocked",
                    "error": str(error),
                    "writes_performed": False,
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
