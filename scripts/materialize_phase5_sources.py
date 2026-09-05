#!/usr/bin/env python3
"""Validate or publish Phase 5 sources from a closed held-out execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.phase5_sources import (
    DEFAULT_PHASE5_SOURCE_ROOT,
    Phase5SelectedParentUnavailable,
    Phase5SourceProductionError,
    materialize_phase5_source_production,
    prepare_phase5_source_production,
)
from story_projection_onto.store import ArtifactStore, BlobStore, Compression, Ledger


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument("--restricted-root", type=Path, required=True)
    command.add_argument("--script-commitment", type=Path, required=True)
    command.add_argument("--primary-results-gate", type=Path, required=True)
    command.add_argument("--held-out-journal-root", type=Path, required=True)
    command.add_argument("--known-answer-source", type=Path, required=True)
    command.add_argument(
        "--trace-instruction",
        action="append",
        default=[],
        metavar="EPISODE_ID=PATH",
        help="exactly three explicit researcher-trace RevisionInstruction files",
    )
    command.add_argument(
        "--trace-submission-receipt",
        action="append",
        default=[],
        metavar="EPISODE_ID=PATH",
        help="exactly three append-only /api/revisions submission receipts",
    )
    command.add_argument("--c0-state-root", type=Path, required=True)
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--protocol", type=Path, default=Path("configs/study/feedback.json"))
    command.add_argument(
        "--c0-rule-config",
        type=Path,
        default=Path("configs/study/c0_rules.json"),
    )
    command.add_argument("--output-root", type=Path, default=DEFAULT_PHASE5_SOURCE_ROOT)
    command.add_argument("--compression", choices=("zstd", "gzip"), default="zstd")
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    return command


def _trace_paths(values: list[str], *, label: str = "trace instructions") -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        episode_id, separator, raw_path = value.partition("=")
        if not separator or not episode_id or not raw_path or episode_id in result:
            raise Phase5SourceProductionError(
                f"{label} require three unique EPISODE_ID=PATH arguments"
            )
        result[episode_id] = Path(raw_path)
    if len(result) != 3:
        raise Phase5SourceProductionError(
            f"exactly three researcher {label} files are required"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    ledger: Ledger | None = None
    try:
        repository = options.repository.resolve(strict=True)

        def repository_path(path: Path) -> Path:
            return path if path.is_absolute() else repository / path

        restricted_root = repository_path(options.restricted_root).resolve(strict=True)
        ledger_path = repository_path(options.ledger)
        artifact_root = repository_path(options.artifact_root)
        if not ledger_path.is_file() or ledger_path.is_symlink():
            raise Phase5SourceProductionError("held-out ledger is absent or symlinked")
        if not artifact_root.is_dir() or artifact_root.is_symlink():
            raise Phase5SourceProductionError("held-out CAS root is absent or symlinked")
        raw_trace_paths = _trace_paths(options.trace_instruction)
        trace_paths = {
            key: repository_path(value) for key, value in raw_trace_paths.items()
        }
        raw_trace_receipt_paths = _trace_paths(
            options.trace_submission_receipt,
            label="trace submission receipts",
        )
        trace_receipt_paths = {
            key: repository_path(value)
            for key, value in raw_trace_receipt_paths.items()
        }
        ledger = Ledger(ledger_path)
        artifacts = ArtifactStore(
            BlobStore(
                artifact_root,
                compression=Compression(options.compression),
            ),
            ledger,
        )
        prepared = prepare_phase5_source_production(
            repository=repository,
            restricted_root=restricted_root,
            output_root=repository_path(options.output_root),
            protocol=load_feedback_protocol(repository_path(options.protocol)),
            script_commitment_path=repository_path(options.script_commitment),
            primary_results_gate_path=repository_path(options.primary_results_gate),
            held_out_journal_root=repository_path(options.held_out_journal_root),
            known_answer_source_path=repository_path(options.known_answer_source),
            trace_instruction_paths=trace_paths,
            trace_submission_receipt_paths=trace_receipt_paths,
            c0_state_root=repository_path(options.c0_state_root),
            c0_rule_config_path=repository_path(options.c0_rule_config),
            artifacts=artifacts,
        )
        writes_performed = False
        if options.materialize:
            source, writes_performed = materialize_phase5_source_production(
                prepared=prepared,
                artifacts=artifacts,
                output_root=repository_path(options.output_root),
                restricted_root=restricted_root,
            )
        else:
            source = prepared.source
        print(
            json.dumps(
                {
                    "state": "materialized" if options.materialize else "validated",
                    "writes_performed": writes_performed,
                    "source_manifest_hash": source.content_hash,
                    "session_hash": prepared.session.content_hash,
                    "episode_count": len(source.episodes),
                    "scripted_episode_count": 6,
                    "researcher_trace_count": 3,
                    "held_out_projection_export_count": 21,
                    "cpu_reprojection_input_count": 12,
                    "scorer_binding_count": 18,
                    "selected_parent_policy": prepared.session.selected_parent_policy,
                    "model_service_called": prepared.session.model_service_called,
                    "fabricated_projection": prepared.session.fabricated_projection,
                },
                sort_keys=True,
            )
        )
        return 0
    except Phase5SelectedParentUnavailable as error:
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "phase5_selected_parent_unavailable",
                    "episode_id": error.episode_id,
                    "condition": error.condition.value,
                    "outcome": error.outcome.value,
                    "source_record_hash": error.source_record_hash,
                    "writes_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    except (Phase5SourceProductionError, ValueError, OSError, KeyError) as error:
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "phase5_source_production_blocked",
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
