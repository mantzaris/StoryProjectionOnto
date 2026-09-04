#!/usr/bin/env python3
"""Audit or execute the single registered 49-call GPU allocation block."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.combined_gpu_block import (
    DEFAULT_CONFIGURATION_PATH,
    CombinedCallManifest,
    CombinedRuntimeBinding,
    CombinedUpstreamGate,
    combined_selection_registry_hash,
    load_combined_configuration,
    load_registered_combined_selections,
)
from story_projection_onto.combined_gpu_materialize import (
    verify_combined_compiler_receipt,
)
from story_projection_onto.combined_gpu_production import (
    CombinedGpuController,
    CombinedProductionError,
    validate_combined_execution_inputs,
)
from story_projection_onto.feedback_runtime import load_feedback_protocol
from story_projection_onto.phase5_execution import (
    Phase5ExecutionInputManifest,
    PrimaryHeldOutResultsGate,
)


def _read_typed(path: Path, model_type):
    absolute = path.absolute()
    current = absolute
    while True:
        if current.is_symlink():
            raise CombinedProductionError(f"symlinked combined input: {path}")
        if current.parent == current:
            break
        current = current.parent
    if not absolute.is_file() or absolute.stat().st_size > 64 * 1024 * 1024:
        raise CombinedProductionError(f"combined input is absent or oversized: {path}")
    try:
        return model_type.model_validate_json(absolute.read_bytes())
    except Exception as error:
        raise CombinedProductionError(f"invalid combined input {path}: {error}") from error


def _bundle_factory(reference: str, *, expected_reference: str, **kwargs):
    if reference != expected_reference:
        raise CombinedProductionError(
            "adapter factory differs from the frozen production implementation"
        )
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise CombinedProductionError("adapter factory must use module.path:callable syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise CombinedProductionError("combined adapter factory is not callable")
    bundle = factory(**kwargs)
    if not all(hasattr(bundle, name) for name in ("provider", "lifecycle_owner", "artifacts")):
        raise CombinedProductionError(
            "combined adapter factory must return provider, lifecycle_owner, and artifacts"
        )
    return bundle


def _persist_private_diagnostic(repository: Path, error: Exception) -> None:
    """Best-effort private diagnostics; the public CLI never echoes path-bearing text."""

    try:
        repository = repository.resolve(strict=True)
        restricted = repository / "artifacts/restricted"
        if restricted.is_symlink():
            return
        restricted.mkdir(mode=0o700, parents=True, exist_ok=True)
        restricted = restricted.resolve(strict=True)
        diagnostic_root = restricted / "combined_gpu_block/diagnostics"
        current = restricted
        for part in diagnostic_root.relative_to(restricted).parts:
            current /= part
            if current.is_symlink():
                return
        diagnostic_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = diagnostic_root / "cli-errors.jsonl"
        if path.is_symlink():
            return
        payload = json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:4096],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        descriptor = os.open(
            path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("private diagnostic append made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(diagnostic_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, ValueError):
        return


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Validate or run the fail-closed paraphrase/feedback/trace/ablation block "
            "against one injected, lifecycle-owned model service."
        )
    )
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION_PATH)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--runtime-binding", type=Path, required=True)
    command.add_argument("--upstream-gate", type=Path, required=True)
    command.add_argument("--compiler-manifest", type=Path, required=True)
    command.add_argument("--development-result", type=Path, required=True)
    command.add_argument("--held-out-manifest", type=Path, required=True)
    command.add_argument("--held-out-execution", type=Path, required=True)
    command.add_argument("--scorer-bridge", type=Path, required=True)
    command.add_argument("--final-schedule", type=Path, required=True)
    command.add_argument("--phase5-inputs", type=Path, required=True)
    command.add_argument("--phase5-primary-gate", type=Path, required=True)
    command.add_argument("--selected-model-freeze", type=Path, required=True)
    command.add_argument("--source-association", type=Path, required=True)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    command.add_argument("--adapter-factory")
    command.add_argument("--run-id")
    command.add_argument("--ledger", type=Path)
    command.add_argument("--artifact-root", type=Path)
    command.add_argument("--runtime-root", type=Path)
    command.add_argument("--quota-root", type=Path)
    command.add_argument("--snapshot", type=Path)
    command.add_argument("--shared-cache", type=Path)
    command.add_argument("--verified-model-manifest", type=Path)
    command.add_argument("--output-root", type=Path)
    command.add_argument("--public-summary", type=Path)
    command.add_argument("--port", type=int, default=8000)
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    diagnostic_repository = options.repository.absolute()
    try:
        repository = options.repository.resolve(strict=True)
        diagnostic_repository = repository

        def repository_path(value: Path) -> Path:
            return value if value.is_absolute() else repository / value

        configuration = load_combined_configuration(repository, options.configuration)
        selections = load_registered_combined_selections(repository, configuration)
        manifest = _read_typed(repository_path(options.manifest), CombinedCallManifest)
        runtime = _read_typed(repository_path(options.runtime_binding), CombinedRuntimeBinding)
        upstream = _read_typed(repository_path(options.upstream_gate), CombinedUpstreamGate)
        phase5_inputs = _read_typed(
            repository_path(options.phase5_inputs), Phase5ExecutionInputManifest
        )
        phase5_gate = _read_typed(
            repository_path(options.phase5_primary_gate), PrimaryHeldOutResultsGate
        )
        compiler_source_paths = {
            "development_result": repository_path(options.development_result),
            "held_out_call_manifest": repository_path(options.held_out_manifest),
            "held_out_execution": repository_path(options.held_out_execution),
            "scorer_bridge": repository_path(options.scorer_bridge),
            "final_schedule": repository_path(options.final_schedule),
            "phase5_inputs": repository_path(options.phase5_inputs),
            "phase5_primary_gate": repository_path(options.phase5_primary_gate),
            "selected_model_freeze": repository_path(options.selected_model_freeze),
            "source_association": repository_path(options.source_association),
        }
        verify_combined_compiler_receipt(
            compiler_manifest_path=repository_path(options.compiler_manifest),
            call_manifest_path=repository_path(options.manifest),
            runtime_binding_path=repository_path(options.runtime_binding),
            upstream_gate_path=repository_path(options.upstream_gate),
            source_paths=compiler_source_paths,
            configuration=configuration,
            selections=selections,
            runtime_binding=runtime,
            upstream_gate=upstream,
            call_manifest=manifest,
            phase5_inputs=phase5_inputs,
            phase5_primary_gate=phase5_gate,
        )
        protocol = load_feedback_protocol(repository / configuration.feedback_protocol_path)
        phase5_calls = validate_combined_execution_inputs(
            configuration=configuration,
            manifest=manifest,
            runtime=runtime,
            upstream_gate=upstream,
            phase5_inputs=phase5_inputs,
            phase5_protocol=protocol,
            phase5_prerequisites=phase5_gate,
        )
        if manifest.registered_selections_hash != combined_selection_registry_hash(selections):
            raise CombinedProductionError("combined manifest selection lineage changed")
        if options.validate_only:
            print(
                json.dumps(
                    {
                        "state": "validated",
                        "writes_performed": False,
                        "manifest_hash": manifest.content_hash,
                        "runtime_binding_hash": runtime.content_hash,
                        "upstream_gate_hash": upstream.content_hash,
                        "base_call_count": len(manifest.calls),
                        "phase5_call_count": len(phase5_calls),
                        "model_load_count": manifest.model_load_count,
                        "load_inclusive_forecast_seconds": (
                            manifest.load_inclusive_forecast_seconds
                        ),
                        "maximum_concurrency": manifest.maximum_concurrency,
                    },
                    sort_keys=True,
                )
            )
            return 0
        required = {
            "run_id": options.run_id,
            "ledger": options.ledger,
            "artifact_root": options.artifact_root,
            "runtime_root": options.runtime_root,
            "quota_root": options.quota_root,
            "snapshot": options.snapshot,
            "shared_cache": options.shared_cache,
            "verified_model_manifest": options.verified_model_manifest,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise CombinedProductionError(
                "--run requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        adapter_factory = options.adapter_factory or configuration.production_adapter_factory
        bundle = _bundle_factory(
            adapter_factory,
            expected_reference=configuration.production_adapter_factory,
            repository=repository,
            run_id=options.run_id,
            configuration=configuration,
            manifest=manifest,
            runtime=runtime,
            upstream_gate=upstream,
            phase5_inputs=phase5_inputs,
            phase5_protocol=protocol,
            phase5_primary_gate=phase5_gate,
            compiler_manifest_path=repository_path(options.compiler_manifest),
            compiled_call_manifest_path=repository_path(options.manifest),
            compiled_runtime_binding_path=repository_path(options.runtime_binding),
            compiled_upstream_gate_path=repository_path(options.upstream_gate),
            compiler_source_paths=compiler_source_paths,
            ledger_path=repository_path(options.ledger),
            artifact_root=repository_path(options.artifact_root),
            runtime_root=repository_path(options.runtime_root),
            quota_root=repository_path(options.quota_root),
            snapshot_path=repository_path(options.snapshot),
            shared_cache=repository_path(options.shared_cache),
            verified_model_manifest_path=repository_path(options.verified_model_manifest),
            selected_model_freeze_path=repository_path(options.selected_model_freeze),
            source_association_path=repository_path(options.source_association),
            port=options.port,
        )
        output_root = repository_path(options.output_root or Path(configuration.output_root))
        public_summary = repository_path(
            options.public_summary or Path(configuration.public_summary_path)
        )
        index = CombinedGpuController(
            run_id=options.run_id,
            configuration=configuration,
            manifest=manifest,
            runtime=runtime,
            upstream_gate=upstream,
            phase5_inputs=phase5_inputs,
            phase5_protocol=protocol,
            phase5_prerequisites=phase5_gate,
            provider=bundle.provider,
            lifecycle_owner=bundle.lifecycle_owner,
            artifacts=bundle.artifacts,
            output_root=output_root,
            public_summary_path=public_summary,
        ).run()
        print(
            json.dumps(
                {
                    "state": "complete",
                    "execution_index_hash": index.content_hash,
                    "base_call_count": index.base_call_count,
                    "model_load_count": index.model_load_count,
                    "repair_request_count": len(index.repair_claim_hashes),
                    "allocated_gpu_seconds": (
                        index.actual_gpu_seconds_after - index.actual_gpu_seconds_before
                    ),
                    "vllm_service_stopped": index.vllm_service_stopped,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        _persist_private_diagnostic(diagnostic_repository, error)
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "combined_execution_blocked",
                    "error_type": type(error).__name__,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
