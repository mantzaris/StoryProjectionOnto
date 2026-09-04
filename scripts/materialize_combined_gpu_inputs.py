#!/usr/bin/env python3
"""Compile the three immutable inputs for the registered combined GPU block."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from story_projection_onto.combined_gpu_block import (
    DEFAULT_CONFIGURATION_PATH,
    load_combined_configuration,
    load_registered_combined_selections,
)
from story_projection_onto.combined_gpu_materialize import (
    CombinedMaterializationError,
    compile_combined_production_inputs,
    compile_combined_runtime_binding,
    load_mapping_snapshot,
    load_record_snapshot,
    materialize_combined_inputs,
    validate_source_association_snapshot,
)
from story_projection_onto.development_runtime import DevelopmentExecutionResult
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
)
from story_projection_onto.held_out_primary import (
    GlobalGpuScheduleSnapshot,
    HeldOutCallManifest,
)
from story_projection_onto.phase5_execution import (
    Phase5ExecutionInputManifest,
    PrimaryHeldOutResultsGate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION_PATH)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--held-out-manifest", type=Path, required=True)
    parser.add_argument("--held-out-execution", type=Path, required=True)
    parser.add_argument("--scorer-bridge", type=Path, required=True)
    parser.add_argument("--final-schedule", type=Path, required=True)
    parser.add_argument("--phase5-inputs", type=Path, required=True)
    parser.add_argument("--phase5-primary-gate", type=Path, required=True)
    parser.add_argument("--selected-model-freeze", type=Path, required=True)
    parser.add_argument("--source-association", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/restricted/combined_gpu_inputs"),
    )
    parser.add_argument("--compiled-at", required=True)
    parser.add_argument("--port", type=int, default=8000)
    return parser


def _absolute(repository: Path, value: Path) -> Path:
    return value if value.is_absolute() else repository / value


def main(argv: list[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        repository = options.repository.resolve(strict=True)
        paths = {
            "development_result": _absolute(repository, options.development_result),
            "held_out_call_manifest": _absolute(repository, options.held_out_manifest),
            "held_out_execution": _absolute(repository, options.held_out_execution),
            "scorer_bridge": _absolute(repository, options.scorer_bridge),
            "final_schedule": _absolute(repository, options.final_schedule),
            "phase5_inputs": _absolute(repository, options.phase5_inputs),
            "phase5_primary_gate": _absolute(repository, options.phase5_primary_gate),
            "selected_model_freeze": _absolute(
                repository, options.selected_model_freeze
            ),
            "source_association": _absolute(repository, options.source_association),
        }
        compiled_at = datetime.fromisoformat(options.compiled_at.replace("Z", "+00:00"))
        if compiled_at.tzinfo is None or compiled_at.utcoffset() is None:
            raise CombinedMaterializationError("--compiled-at must be timezone-aware")
        configuration = load_combined_configuration(repository, options.configuration)
        selections = load_registered_combined_selections(repository, configuration)
        development_source = load_record_snapshot(
            paths["development_result"],
            DevelopmentExecutionResult,
            label="development result",
        )
        held_out_calls_source = load_record_snapshot(
            paths["held_out_call_manifest"],
            HeldOutCallManifest,
            label="held-out call manifest",
        )
        held_out_execution_source = load_record_snapshot(
            paths["held_out_execution"],
            HeldOutExecutionManifest,
            label="held-out execution",
        )
        scorer_bridge_source = load_record_snapshot(
            paths["scorer_bridge"],
            ScorerBridgeAuthorization,
            label="scorer bridge",
        )
        final_schedule_source = load_record_snapshot(
            paths["final_schedule"],
            GlobalGpuScheduleSnapshot,
            label="final GPU schedule",
        )
        phase5_inputs_source = load_record_snapshot(
            paths["phase5_inputs"],
            Phase5ExecutionInputManifest,
            label="Phase 5 inputs",
        )
        phase5_gate_source = load_record_snapshot(
            paths["phase5_primary_gate"],
            PrimaryHeldOutResultsGate,
            label="Phase 5 primary gate",
        )
        selected_freeze_source = load_mapping_snapshot(
            paths["selected_model_freeze"], label="selected-model freeze"
        )
        source_association_source = load_mapping_snapshot(
            paths["source_association"], label="source association"
        )
        source_association = dict(
            validate_source_association_snapshot(
                source_association_source,
                association_path=paths["source_association"],
                source_root=repository,
            )
        )
        development = development_source.value
        held_out_calls = held_out_calls_source.value
        held_out_execution = held_out_execution_source.value
        scorer_bridge = scorer_bridge_source.value
        final_schedule = final_schedule_source.value
        phase5_inputs = phase5_inputs_source.value
        phase5_gate = phase5_gate_source.value
        selected_freeze = selected_freeze_source.value
        runtime = compile_combined_runtime_binding(
            repository=repository,
            configuration=configuration,
            selected_model_freeze=selected_freeze,
            source_association=source_association,
            snapshot_path=_absolute(repository, options.snapshot),
            shared_cache=_absolute(repository, options.shared_cache),
            port=options.port,
        )
        compiled = compile_combined_production_inputs(
            configuration=configuration,
            selections=selections,
            runtime_binding=runtime,
            development=development,
            held_out_calls=held_out_calls,
            held_out_execution=held_out_execution,
            scorer_bridge=scorer_bridge,
            final_schedule=final_schedule,
            phase5_inputs=phase5_inputs,
            phase5_primary_gate=phase5_gate,
            compiled_at=compiled_at,
        )
        source_file_hashes = {
            "development_result": development_source.file_sha256,
            "held_out_call_manifest": held_out_calls_source.file_sha256,
            "held_out_execution": held_out_execution_source.file_sha256,
            "scorer_bridge": scorer_bridge_source.file_sha256,
            "final_schedule": final_schedule_source.file_sha256,
            "phase5_inputs": phase5_inputs_source.file_sha256,
            "phase5_primary_gate": phase5_gate_source.file_sha256,
            "selected_model_freeze": selected_freeze_source.file_sha256,
            "source_association": source_association_source.file_sha256,
        }
        source_logical_hashes = {
            "development_result": development.content_hash,
            "held_out_call_manifest": held_out_calls.content_hash,
            "held_out_execution": held_out_execution.content_hash,
            "scorer_bridge": scorer_bridge.content_hash,
            "final_schedule": final_schedule.content_hash,
            "phase5_inputs": phase5_inputs.content_hash,
            "phase5_primary_gate": phase5_gate.content_hash,
            "selected_model_freeze": str(selected_freeze["manifest_sha256"]),
            "source_association": str(source_association["manifest_sha256"]),
            "registered_selections": selections.content_hash,
            "runtime_binding": runtime.content_hash,
        }
        compiler = materialize_combined_inputs(
            repository=repository,
            output_root=options.output_root,
            compiled=compiled,
            source_file_sha256s=source_file_hashes,
            source_logical_hashes=source_logical_hashes,
            compiled_at=compiled_at,
        )
        print(
            json.dumps(
                {
                    "state": "complete",
                    "compiler_manifest_hash": compiler.content_hash,
                    "runtime_binding_hash": runtime.content_hash,
                    "upstream_gate_hash": compiled.upstream_gate.content_hash,
                    "call_manifest_hash": compiled.call_manifest.content_hash,
                    "base_call_count": len(compiled.call_manifest.calls),
                    "model_calls_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            state = "interrupted"
            error_code = "combined_input_compilation_interrupted"
            return_code = 130
        elif isinstance(error, SystemExit):
            state = "terminated"
            error_code = "combined_input_compilation_system_exit"
            return_code = error.code if isinstance(error.code, int) else 2
        else:
            state = "blocked"
            error_code = "combined_input_compilation_failed"
            return_code = 2
        print(
            json.dumps(
                {"state": state, "error_code": error_code},
                sort_keys=True,
            )
        )
        return return_code


if __name__ == "__main__":
    raise SystemExit(main())
