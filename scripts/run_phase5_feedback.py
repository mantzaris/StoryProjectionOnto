#!/usr/bin/env python3
"""Validate or execute the frozen Phase 5 feedback inventory."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.phase5_execution import (
    DEFAULT_PHASE5_OUTPUT_ROOT,
    C2RegenerationAdapter,
    Phase5ExecutionError,
    SQLiteFeedbackLedgerVerifier,
    load_phase5_input_manifest,
    load_phase5_runner_configuration,
    run_phase5_from_files,
    validate_phase5_inputs,
    validate_phase5_prerequisites,
)
from story_projection_onto.phase5_production import (
    load_phase5_materialization_receipt,
    validate_phase5_materialization_receipt,
)


def _adapter_factory(reference: str) -> C2RegenerationAdapter:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise Phase5ExecutionError("adapter factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory()
    if not all(callable(getattr(adapter, method, None)) for method in ("regenerate", "recover")):
        raise Phase5ExecutionError("adapter factory did not return a regeneration adapter")
    return adapter


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Run the review-gated, append-only Phase 5 feedback inventory."
    )
    command.add_argument("--inputs", type=Path, required=True)
    command.add_argument("--materialization-receipt", type=Path, required=True)
    command.add_argument("--primary-results-gate", type=Path, required=True)
    command.add_argument("--benchmark-root", type=Path, default=Path("data/synthetic"))
    command.add_argument(
        "--review-completion-root",
        type=Path,
        default=Path("artifacts/restricted/scorer_only/independent_review"),
    )
    command.add_argument("--protocol", type=Path, default=Path("configs/study/feedback.json"))
    command.add_argument("--output-root", type=Path, default=DEFAULT_PHASE5_OUTPUT_ROOT)
    command.add_argument(
        "--ledger",
        type=Path,
        help="Existing append-only SQLite ledger; required for --run receipt replay.",
    )
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    command.add_argument("--adapter-factory")
    command.add_argument(
        "--completed-at",
        help="Required aware ISO-8601 timestamp for a reproducible final manifest.",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    try:
        inputs = load_phase5_input_manifest(options.inputs)
        validate_phase5_materialization_receipt(
            inputs,
            load_phase5_materialization_receipt(options.materialization_receipt),
        )
        protocol = load_feedback_protocol(options.protocol)
        runner_configuration = load_phase5_runner_configuration()
        if runner_configuration.feedback_protocol_hash != protocol.content_hash:
            raise Phase5ExecutionError("runner configuration binds another feedback protocol")
        if (
            runner_configuration.scripted_episode_count != len(protocol.scripted_episodes)
            or runner_configuration.researcher_trace_count != len(protocol.researcher_trace_slots)
            or runner_configuration.cpu_reprojection_call_count != 12
            or runner_configuration.c2_regeneration_slot_count
            != protocol.total_c2_regeneration_calls
        ):
            raise Phase5ExecutionError("runner configuration inventory differs from protocol")
        validate_phase5_inputs(inputs, protocol)
        validate_phase5_prerequisites(
            inputs=inputs,
            primary_results_gate_path=options.primary_results_gate,
            benchmark_root=options.benchmark_root,
            review_completion_root=options.review_completion_root,
        )
        if options.validate_only:
            print(
                json.dumps(
                    {
                        "state": "validated",
                        "writes_performed": False,
                        "protocol_hash": protocol.content_hash,
                        "planned_cpu_calls": 12,
                        "planned_c2_regeneration_slots": 9,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if not options.adapter_factory or not options.completed_at or options.ledger is None:
            raise Phase5ExecutionError(
                "--run requires --adapter-factory, --ledger, and explicit --completed-at"
            )
        completed_at = datetime.fromisoformat(options.completed_at.replace("Z", "+00:00"))
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise Phase5ExecutionError("--completed-at must include a timezone")
        index = run_phase5_from_files(
            input_manifest_path=options.inputs,
            materialization_receipt_path=options.materialization_receipt,
            primary_results_gate_path=options.primary_results_gate,
            benchmark_root=options.benchmark_root,
            review_completion_root=options.review_completion_root,
            output_root=options.output_root,
            adapter=_adapter_factory(options.adapter_factory),
            ledger_verifier=SQLiteFeedbackLedgerVerifier(options.ledger),
            completed_at=completed_at,
            protocol_path=options.protocol,
        )
        print(
            json.dumps(
                {
                    "state": "complete",
                    "execution_index_hash": index.content_hash,
                    "feedback_manifest_hash": index.feedback_manifest_hash,
                    "cpu_calls": index.cpu_call_record_count,
                    "c2_regeneration_slots": index.c2_regeneration_slot_count,
                    "actual_gpu_requests": index.actual_gpu_request_count,
                    "repair_gpu_requests": index.repair_gpu_request_count,
                    "allocated_gpu_seconds": index.allocated_gpu_seconds,
                },
                sort_keys=True,
            )
        )
        return 0
    except (Phase5ExecutionError, ValueError, ImportError, AttributeError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
