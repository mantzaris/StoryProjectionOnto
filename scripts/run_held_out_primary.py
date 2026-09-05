#!/usr/bin/env python3
"""Review-gated entry point for the held-out primary control plane."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.held_out_binding import (
    DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH,
    load_runtime_bound_held_out_configuration,
    persist_restricted_error,
)
from story_projection_onto.held_out_controller import (
    InterruptedCallRecoveryRequired,
    execute_reviewed_held_out_manifest,
)
from story_projection_onto.held_out_factory import (
    _canonical_restricted_root,
    _restricted_descendant,
)
from story_projection_onto.held_out_primary import (
    DEFAULT_HELD_OUT_CONTROL_PATH,
    DEFAULT_HELD_OUT_OUTPUT_ROOT,
    HeldOutControlError,
    open_reviewed_held_out_plan,
)
from story_projection_onto.held_out_results import (
    DEFAULT_HELD_OUT_RESULTS_ROOT,
    close_held_out_primary_results,
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
    command.add_argument(
        "--runtime-binding",
        type=Path,
        default=DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH,
    )
    command.add_argument("--output-root", type=Path, default=DEFAULT_HELD_OUT_OUTPUT_ROOT)
    command.add_argument(
        "--results-gate-root",
        type=Path,
        default=DEFAULT_HELD_OUT_RESULTS_ROOT,
    )
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    command.add_argument(
        "--adapter-factory",
        help="module.path:callable returning an object with cpu and sessions attributes",
    )
    command.add_argument("--snapshot", type=Path)
    command.add_argument("--shared-cache", type=Path)
    command.add_argument("--verified-model-manifest", type=Path)
    command.add_argument("--selected-model-freeze", type=Path)
    command.add_argument("--source-association", type=Path)
    command.add_argument("--development-prequery-inputs-artifact-hash")
    command.add_argument("--ledger", type=Path)
    command.add_argument("--artifact-root", type=Path)
    command.add_argument("--runtime-root", type=Path)
    command.add_argument("--restricted-root", type=Path)
    command.add_argument("--quota-root", type=Path)
    command.add_argument("--port", type=int, default=8000)
    return command


def _adapter_bundle(reference: str, **kwargs):
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise HeldOutControlError("adapter factory must use module.path:callable syntax")
    bundle = getattr(importlib.import_module(module_name), attribute)(**kwargs)
    if not all(hasattr(bundle, name) for name in ("cpu", "sessions", "runtime")):
        raise HeldOutControlError("adapter factory must return cpu, sessions, and runtime")
    return bundle


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    repository: Path | None = None
    restricted_root: Path | None = None
    try:
        repository = options.repository.resolve()
        if options.restricted_root is not None:
            restricted_root = _canonical_restricted_root(
                repository,
                options.restricted_root,
            )
        review_root = (
            options.review_completion_root
            if options.review_completion_root.is_absolute()
            else repository / options.review_completion_root
        )
        # This is intentionally the first operation that can lead to runtime inputs.
        reviewed_plan = open_reviewed_held_out_plan(
            repository=repository,
            review_completion_root=review_root,
            configuration_path=options.control,
            runtime_binding_path=options.runtime_binding,
        )
        manifest = reviewed_plan.call_manifest
        configuration, runtime_binding = load_runtime_bound_held_out_configuration(
            repository=repository,
            configuration_path=options.control,
            binding_path=options.runtime_binding,
            expected_plan_hash=manifest.content_hash,
        )
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
        if not options.adapter_factory:
            raise HeldOutControlError("--run requires an injected adapter factory")
        runtime_arguments = (
            "snapshot",
            "shared_cache",
            "verified_model_manifest",
            "selected_model_freeze",
            "source_association",
            "development_prequery_inputs_artifact_hash",
            "ledger",
            "artifact_root",
            "runtime_root",
            "restricted_root",
            "quota_root",
        )
        missing = [name for name in runtime_arguments if getattr(options, name) is None]
        if missing:
            raise HeldOutControlError(
                "--run requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        if configuration.production_adapter_factory == "PENDING":
            raise HeldOutControlError(
                "scientific execution is blocked until the production adapter is frozen"
            )
        if options.adapter_factory != configuration.production_adapter_factory:
            raise HeldOutControlError(
                "adapter factory differs from the frozen production implementation"
            )

        def runtime_path(value: Path) -> Path:
            return value if value.is_absolute() else repository / value

        bundle = _adapter_bundle(
            options.adapter_factory,
            repository=repository,
            reviewed_plan=reviewed_plan,
            configuration=configuration,
            snapshot_path=runtime_path(options.snapshot),
            shared_cache=runtime_path(options.shared_cache),
            verified_model_manifest_path=runtime_path(options.verified_model_manifest),
            selected_model_freeze_path=runtime_path(options.selected_model_freeze),
            source_association_path=runtime_path(options.source_association),
            development_prequery_inputs_artifact_hash=(
                options.development_prequery_inputs_artifact_hash
            ),
            ledger_path=runtime_path(options.ledger),
            artifact_root=runtime_path(options.artifact_root),
            runtime_root=runtime_path(options.runtime_root),
            restricted_root=restricted_root,
            quota_root=runtime_path(options.quota_root),
            predecessor_ledger_binding=(
                runtime_binding.development_predecessor_ledger
            ),
            port=options.port,
        )
        output_root = (
            options.output_root
            if options.output_root.is_absolute()
            else repository / options.output_root
        )
        _restricted_descendant(
            restricted_root,
            output_root,
            label="held-out output root",
        )
        results_gate_root = (
            options.results_gate_root
            if options.results_gate_root.is_absolute()
            else repository / options.results_gate_root
        )
        _restricted_descendant(
            restricted_root,
            results_gate_root,
            label="held-out results gate root",
        )
        try:
            execution = execute_reviewed_held_out_manifest(
                reviewed_plan=reviewed_plan,
                configuration=configuration,
                repository=repository,
                review_completion_root=review_root,
                configuration_path=options.control,
                runtime_binding_path=options.runtime_binding,
                cpu=bundle.cpu,
                sessions=bundle.sessions,
                runtime=bundle.runtime,
                output_root=output_root,
            )
            scorer_bridge, primary_results_gate = close_held_out_primary_results(
                reviewed_plan=reviewed_plan,
                call_manifest=manifest,
                execution=execution,
                held_out_output_root=output_root,
                results_root=results_gate_root,
                runtime=bundle.runtime,
                clock=lambda: datetime.now(UTC),
            )
        except InterruptedCallRecoveryRequired:
            preserve = getattr(bundle, "preserve_for_resume", None)
            if callable(preserve):
                preserve()
            else:
                close = getattr(bundle, "close", None)
                if callable(close):
                    close()
            raise
        except BaseException as error:
            callback_name = (
                "preserve_for_resume"
                if isinstance(error, (OSError, KeyboardInterrupt))
                else "close"
            )
            callback = getattr(bundle, callback_name, None)
            if callable(callback):
                callback()
            raise
        else:
            close = getattr(bundle, "close", None)
            if callable(close):
                close()
        print(
            json.dumps(
                {
                    "state": "complete",
                    "execution_manifest_hash": execution.content_hash,
                    "scorer_bridge_hash": scorer_bridge.content_hash,
                    "primary_results_gate_hash": primary_results_gate.content_hash,
                    "itt_call_count": len(execution.itt_records),
                    "allocated_gpu_seconds": "read_from_global_ledger",
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        if restricted_root is not None:
            persist_restricted_error(
                restricted_root=restricted_root,
                namespace="held_out_primary",
                error=error,
            )
        if isinstance(error, KeyboardInterrupt):
            state = "interrupted"
            error_code = "held_out_interrupted"
            return_code = 130
        elif isinstance(error, SystemExit):
            state = "terminated"
            error_code = "held_out_system_exit"
            return_code = error.code if isinstance(error.code, int) else 2
        else:
            state = "blocked"
            error_code = "held_out_control_blocked"
            return_code = 2
        print(
            json.dumps(
                {
                    "state": state,
                    "error_code": error_code,
                    "error": "Held-out execution is blocked; inspect restricted logs",
                },
                sort_keys=True,
            )
        )
        return return_code


if __name__ == "__main__":
    raise SystemExit(main())
