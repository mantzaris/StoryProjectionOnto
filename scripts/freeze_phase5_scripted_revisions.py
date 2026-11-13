#!/usr/bin/env python3
"""Validate or freeze the exact six pre-output scripted Phase 5 revisions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.held_out_binding import (
    DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH,
)
from story_projection_onto.held_out_primary import DEFAULT_HELD_OUT_CONTROL_PATH
from story_projection_onto.phase5_commitment import (
    DEFAULT_COMMITMENT_OUTPUT_ROOT,
    Phase5CommitmentError,
    load_scripted_revision_draft,
    materialize_phase5_script_commitment,
    open_phase5_hash_bound_reviewed_plan,
    prepare_phase5_script_commitment,
    validate_phase5_commitment_destinations,
)
from story_projection_onto.store import ArtifactStore, BlobStore, Compression, Ledger


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument("--draft-manifest", type=Path, required=True)
    command.add_argument("--protocol", type=Path, default=Path("configs/study/feedback.json"))
    command.add_argument(
        "--review-completion-root",
        type=Path,
        default=Path("artifacts/restricted/scorer_only/independent_review"),
    )
    command.add_argument(
        "--held-out-control",
        type=Path,
        default=DEFAULT_HELD_OUT_CONTROL_PATH,
    )
    command.add_argument(
        "--held-out-runtime-binding",
        type=Path,
        default=DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH,
    )
    command.add_argument("--held-out-journal-root", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--restricted-root", type=Path, required=True)
    command.add_argument("--output-root", type=Path, default=DEFAULT_COMMITMENT_OUTPUT_ROOT)
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

        def repository_path(path: Path) -> Path:
            return path if path.is_absolute() else repository / path

        restricted_root = repository_path(options.restricted_root).resolve(strict=True)
        draft_path = repository_path(options.draft_manifest)
        ledger_path = repository_path(options.ledger)
        artifact_root = repository_path(options.artifact_root)
        output_root = repository_path(options.output_root)
        validate_phase5_commitment_destinations(
            repository=repository,
            restricted_root=restricted_root,
            draft_path=draft_path,
            ledger_path=ledger_path,
            artifact_root=artifact_root,
            output_root=output_root,
        )
        draft = load_scripted_revision_draft(draft_path)
        protocol = load_feedback_protocol(repository_path(options.protocol))
        reviewed_plan, held_out_configuration, review_gate = (
            open_phase5_hash_bound_reviewed_plan(
                repository=repository,
                restricted_root=restricted_root,
                review_completion_root=repository_path(options.review_completion_root),
                configuration_path=options.held_out_control,
                runtime_binding_path=options.held_out_runtime_binding,
            )
        )
        prepared = prepare_phase5_script_commitment(
            draft=draft,
            protocol=protocol,
            reviewed_plan=reviewed_plan,
            held_out_configuration=held_out_configuration,
            review_gate=review_gate,
            repository=repository,
            restricted_root=restricted_root,
            held_out_journal_root=repository_path(options.held_out_journal_root),
            committed_at=datetime.now(UTC),
        )
        created = False
        if options.materialize:
            ledger = Ledger(ledger_path)
            manifest, created = materialize_phase5_script_commitment(
                prepared=prepared,
                artifacts=ArtifactStore(
                    BlobStore(artifact_root, compression=Compression(options.compression)),
                    ledger,
                ),
                output_root=output_root,
                restricted_root=restricted_root,
            )
        else:
            manifest = prepared.manifest
        print(
            json.dumps(
                {
                    "state": "materialized" if options.materialize else "validated",
                    "writes_performed": bool(options.materialize and created),
                    "commitment_manifest_hash": manifest.content_hash,
                    "draft_manifest_hash": manifest.draft_manifest_hash,
                    "review_completion_manifest_hash": (
                        manifest.review_gate.review_completion_manifest_hash
                    ),
                    "final_reviewed_seal_hash": (
                        manifest.review_gate.final_reviewed_seal_hash
                    ),
                    "held_out_call_manifest_hash": (
                        manifest.parent_readiness_floor.held_out_call_manifest_hash
                    ),
                    "registered_watchdog_floor_seconds": (
                        manifest.parent_readiness_floor.registered_watchdog_floor_seconds
                    ),
                    "script_count": len(manifest.entries),
                    "scheduled_activation_at": manifest.scheduled_activation_at.isoformat(),
                    "condition_output_inspected": manifest.condition_output_inspected,
                    "held_out_output_read": manifest.held_out_output_read,
                    "scorer_gold_read": manifest.scorer_gold_read,
                    "model_service_called": manifest.model_service_called,
                },
                sort_keys=True,
            )
        )
        return 0
    except (Phase5CommitmentError, ValueError, OSError, KeyError):
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "phase5_script_commitment_blocked",
                    "error": "Phase 5 script commitment is blocked; inspect restricted inputs",
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
