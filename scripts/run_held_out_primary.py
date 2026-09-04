#!/usr/bin/env python3
"""Review-gated entry point for the held-out primary control plane."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

from story_projection_onto.held_out_controller import execute_reviewed_held_out_manifest
from story_projection_onto.held_out_primary import (
    DEFAULT_HELD_OUT_CONTROL_PATH,
    DEFAULT_HELD_OUT_OUTPUT_ROOT,
    HeldOutControlError,
    load_held_out_control_configuration,
    open_reviewed_held_out_plan,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Validate or run the review-gated 168-call held-out primary manifest."
    )
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument(
        "--review-completion-root",
        type=Path,
        default=Path("artifacts/restricted/scorer_only/independent_review"),
    )
    command.add_argument("--control", type=Path, default=DEFAULT_HELD_OUT_CONTROL_PATH)
    command.add_argument("--output-root", type=Path, default=DEFAULT_HELD_OUT_OUTPUT_ROOT)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    command.add_argument(
        "--adapter-factory",
        help="module.path:callable returning an object with cpu and sessions attributes",
    )
    command.add_argument("--completed-at")
    return command


def _adapter_bundle(reference: str):
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise HeldOutControlError("adapter factory must use module.path:callable syntax")
    bundle = getattr(importlib.import_module(module_name), attribute)()
    if not all(hasattr(bundle, name) for name in ("cpu", "sessions", "runtime")):
        raise HeldOutControlError("adapter factory must return cpu, sessions, and runtime")
    return bundle


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    repository = options.repository.resolve()
    review_root = (
        options.review_completion_root
        if options.review_completion_root.is_absolute()
        else repository / options.review_completion_root
    )
    try:
        # This is intentionally the first operation that can lead to runtime inputs.
        reviewed_plan = open_reviewed_held_out_plan(
            repository=repository,
            review_completion_root=review_root,
            configuration_path=options.control,
        )
        manifest = reviewed_plan.call_manifest
        configuration = load_held_out_control_configuration(repository, options.control)
        if options.validate_only:
            print(
                json.dumps(
                    {
                        "state": "reviewed_plan_validated",
                        "writes_performed": False,
                        "call_manifest_hash": manifest.content_hash,
                        "world_count": len(manifest.units),
                        "context_count": sum(len(item.query_stages) for item in manifest.units),
                        "c0_cpu_constructions": 12,
                        "c1_calls": 24,
                        "c2_calls": 72,
                        "fixed_select_calls": 72,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if not options.adapter_factory or not options.completed_at:
            raise HeldOutControlError(
                "--run requires an injected adapter factory and aware --completed-at"
            )
        if configuration.production_adapter_factory == "PENDING":
            raise HeldOutControlError(
                "scientific execution is blocked until the production adapter is frozen"
            )
        if options.adapter_factory != configuration.production_adapter_factory:
            raise HeldOutControlError(
                "adapter factory differs from the frozen production implementation"
            )
        completed_at = datetime.fromisoformat(options.completed_at.replace("Z", "+00:00"))
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise HeldOutControlError("--completed-at must include a timezone")
        bundle = _adapter_bundle(options.adapter_factory)
        output_root = (
            options.output_root
            if options.output_root.is_absolute()
            else repository / options.output_root
        )
        execution = execute_reviewed_held_out_manifest(
            reviewed_plan=reviewed_plan,
            configuration=configuration,
            repository=repository,
            review_completion_root=review_root,
            configuration_path=options.control,
            cpu=bundle.cpu,
            sessions=bundle.sessions,
            runtime=bundle.runtime,
            output_root=output_root,
            completed_at=completed_at,
        )
        print(
            json.dumps(
                {
                    "state": "complete",
                    "execution_manifest_hash": execution.content_hash,
                    "itt_call_count": len(execution.itt_records),
                    "allocated_gpu_seconds": "read_from_global_ledger",
                },
                sort_keys=True,
            )
        )
        return 0
    except (HeldOutControlError, ValueError, OSError, ImportError, AttributeError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
