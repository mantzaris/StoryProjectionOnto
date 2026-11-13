"""CPU-only production compiler for the registered 49-call combined block."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from story_projection_onto.combined_gpu_block import (
    ACTIVE_ABLATION_CONDITIONS,
    CombinedBlockConfiguration,
    CombinedBlockError,
    CombinedCallManifest,
    CombinedRuntimeBinding,
    CombinedUpstreamGate,
    RegisteredCombinedSelections,
    combined_selection_registry_hash,
    compile_combined_call_manifest,
)
from story_projection_onto.combined_gpu_inputs import (
    compile_combined_query_sources,
    replay_combined_upstream_gate,
)
from story_projection_onto.combined_gpu_production import (
    validate_combined_execution_inputs,
)
from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.development_adapter import (
    STRUCTURED_DECODER,
    DevelopmentConstructionConfiguration,
    development_request_runtime,
)
from story_projection_onto.development_continuation import development_validator_hash
from story_projection_onto.development_runtime import DevelopmentExecutionResult
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    VLLMLaunchConfiguration,
    capture_tokenizer_manifest,
)
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    ScorerBridgeAuthorization,
)
from story_projection_onto.held_out_primary import (
    GlobalGpuScheduleSnapshot,
    HeldOutCallManifest,
)
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.phase5_execution import (
    Phase5ExecutionInputManifest,
    PrimaryHeldOutResultsGate,
)


class CombinedMaterializationError(CombinedBlockError):
    """A production compiler input or immutable publication is invalid."""


class CombinedCompilerOutput(ImmutableRecord):
    logical_name: str = Field(min_length=1)
    file_sha256: Sha256Digest
    logical_content_hash: Sha256Digest


class CombinedCompilerManifest(ImmutableRecord):
    compiler_id: str = Field(min_length=1)
    configuration_hash: Sha256Digest
    source_file_sha256s: Mapping[str, Sha256Digest]
    source_logical_hashes: Mapping[str, Sha256Digest]
    outputs: tuple[
        CombinedCompilerOutput,
        CombinedCompilerOutput,
        CombinedCompilerOutput,
    ]
    compiled_at: datetime
    model_calls_performed: Literal[False] = False
    scorer_selection_read_by_trusted_compiler_only: Literal[True] = True
    scorer_gold_in_outputs: Literal[False] = False

    @model_validator(mode="after")
    def exact_outputs(self) -> CombinedCompilerManifest:
        if self.compiled_at.tzinfo is None or self.compiled_at.utcoffset() is None:
            raise ValueError("combined compiler timestamp must be timezone-aware")
        expected = ("runtime_binding", "upstream_gate", "call_manifest")
        if tuple(item.logical_name for item in self.outputs) != expected:
            raise ValueError("combined compiler output inventory changed")
        required_sources = {
            "development_result",
            "held_out_call_manifest",
            "held_out_execution",
            "scorer_bridge",
            "final_schedule",
            "phase5_inputs",
            "phase5_primary_gate",
            "selected_model_freeze",
            "source_association",
        }
        if set(self.source_file_sha256s) != required_sources:
            raise ValueError("combined compiler source-file inventory changed")
        expected_logical = required_sources | {"registered_selections", "runtime_binding"}
        if set(self.source_logical_hashes) != expected_logical:
            raise ValueError("combined compiler source-logical inventory changed")
        return self


@dataclass(frozen=True, slots=True)
class CompiledCombinedInputs:
    runtime_binding: CombinedRuntimeBinding
    upstream_gate: CombinedUpstreamGate
    call_manifest: CombinedCallManifest


@dataclass(frozen=True, slots=True)
class CompilerSourceSnapshot[T]:
    """One descriptor-read compiler source and the value parsed from those bytes."""

    value: T
    raw: bytes
    file_sha256: Sha256Digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hash(value: Mapping[str, object], *, label: str) -> str:
    supplied = value.get("manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    expected = canonical_sha256(immutable)
    if supplied != expected:
        raise CombinedMaterializationError(f"{label} canonical self-hash changed")
    return expected


def load_mapping_snapshot(
    path: Path,
    *,
    label: str,
) -> CompilerSourceSnapshot[dict[str, object]]:
    """Read, hash, and parse one mapping without reopening its pathname."""

    raw = _safe_file(path, label=label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CombinedMaterializationError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise CombinedMaterializationError(f"{label} must be one JSON object")
    return CompilerSourceSnapshot(
        value=cast(dict[str, object], value),
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_record_snapshot[T: ImmutableRecord](
    path: Path,
    model_type: type[T],
    *,
    label: str,
) -> CompilerSourceSnapshot[T]:
    """Read, hash, and parse one typed record without reopening its pathname."""

    raw = _safe_file(path, label=label)
    try:
        value = model_type.model_validate_json(raw)
    except Exception as error:
        raise CombinedMaterializationError(f"invalid {label}") from error
    return CompilerSourceSnapshot(
        value=value,
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_mapping_file(path: Path, *, label: str) -> dict[str, object]:
    """Compatibility wrapper returning the parsed value from one safe snapshot."""

    return load_mapping_snapshot(path, label=label).value


def load_record_file[T: ImmutableRecord](
    path: Path,
    model_type: type[T],
    *,
    label: str,
) -> T:
    """Compatibility wrapper returning the parsed value from one safe snapshot."""

    return load_record_snapshot(path, model_type, label=label).value


def _safe_file(path: Path, *, label: str) -> bytes:
    """Read a bounded regular file through no-follow directory descriptors."""

    absolute = path.absolute()
    if len(absolute.parts) < 2:
        raise CombinedMaterializationError(f"{label} is absent or oversized")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | nofollow,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | nofollow,
            dir_fd=directory_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 64 * 1024 * 1024:
                raise CombinedMaterializationError(f"{label} is absent or oversized")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                block = os.read(descriptor, min(1024 * 1024, remaining))
                if not block:
                    raise CombinedMaterializationError(f"{label} changed while being read")
                chunks.append(block)
                remaining -= len(block)
            if os.read(descriptor, 1):
                raise CombinedMaterializationError(f"{label} changed while being read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise CombinedMaterializationError(
            f"{label} is absent or uses a symbolic link"
        ) from error
    finally:
        os.close(directory_fd)


def validate_source_association_snapshot(
    snapshot: CompilerSourceSnapshot[Mapping[str, object]],
    *,
    association_path: Path,
    source_root: Path,
) -> Mapping[str, object]:
    """Validate a source association already parsed from its captured bytes.

    The association pathname is used only to locate its separately hash-bound
    sibling source manifest. In particular, the association itself is never
    reopened after ``snapshot`` was captured.
    """

    association = snapshot.value
    association_hash = _self_hash(association, label="source association")
    if association.get("manifest_sha256") != association_hash:
        raise CombinedMaterializationError(
            "source association manifest hash does not match its contents"
        )
    local_tree = association.get("local_tree_sha256")
    remote_tree = association.get("remote_tree_sha256")
    if (
        association.get("kind") != "local_remote_source_tree_association"
        or not isinstance(local_tree, str)
        or len(local_tree) != 64
        or local_tree != remote_tree
        or association.get("branch")
        != "implementation/query-dependent-temporal-ontology"
    ):
        raise CombinedMaterializationError(
            "source association does not bind one identical local/remote tree"
        )
    revision = association.get("revision_label")
    manifest_name = association.get("local_manifest")
    manifest_file_hash = association.get("local_manifest_file_sha256")
    if (
        not isinstance(revision, str)
        or not revision
        or not isinstance(manifest_name, str)
        or Path(manifest_name).name != manifest_name
        or not isinstance(manifest_file_hash, str)
    ):
        raise CombinedMaterializationError(
            "source association lacks one safe local manifest binding"
        )
    manifest_path = association_path.absolute().parent / manifest_name
    manifest_raw = _safe_file(manifest_path, label="source association local manifest")
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_file_hash:
        raise CombinedMaterializationError(
            "source association local manifest file hash changed"
        )
    try:
        local_manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CombinedMaterializationError(
            "source association local manifest is invalid"
        ) from error
    if not isinstance(local_manifest, dict):
        raise CombinedMaterializationError(
            "source association local manifest is invalid"
        )
    rebuilt = build_source_manifest(source_root, revision).to_dict()
    if local_manifest != rebuilt:
        raise CombinedMaterializationError(
            "current source tree differs from its associated local manifest"
        )
    if rebuilt.get("tree_sha256") != local_tree:
        raise CombinedMaterializationError(
            "rebuilt source-tree hash differs from its association"
        )
    return association


def compile_combined_runtime_binding(
    *,
    repository: Path,
    configuration: CombinedBlockConfiguration,
    selected_model_freeze: Mapping[str, object],
    source_association: Mapping[str, object],
    snapshot_path: Path,
    shared_cache: Path,
    port: int = 8000,
) -> CombinedRuntimeBinding:
    """Reproduce the model, prompt, schema, decoder, and source bindings."""

    repository = repository.resolve(strict=True)
    freeze_hash = _self_hash(selected_model_freeze, label="selected-model freeze")
    association_hash = _self_hash(source_association, label="source association")
    freeze_digests = (
        selected_model_freeze.get("snapshot_manifest_sha256"),
        selected_model_freeze.get("launcher_configuration_sha256"),
        selected_model_freeze.get("tokenizer_manifest_sha256"),
        selected_model_freeze.get("source_tree_sha256"),
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in freeze_digests
    ):
        raise CombinedMaterializationError("combined selected-model freeze lacks a digest")
    if (
        selected_model_freeze.get("repository") != configuration.selected_model_repository
        or selected_model_freeze.get("revision") != configuration.selected_model_revision
        or selected_model_freeze.get("served_model_name") != configuration.served_model_name
        or selected_model_freeze.get("source_association_manifest_sha256")
        != association_hash
        or selected_model_freeze.get("source_tree_sha256")
        != source_association.get("local_tree_sha256")
    ):
        raise CombinedMaterializationError("combined selected-model/source freeze changed")
    launcher = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot_path,
        shared_cache=shared_cache,
        model_configuration_path=repository / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=cast(str, freeze_digests[0]),
        port=port,
    )
    if (
        launcher.configuration_hash
        != selected_model_freeze.get("launcher_configuration_sha256")
        or (launcher.repository, launcher.revision, launcher.served_model_name)
        != (
            FALLBACK_MODEL_REPOSITORY,
            FALLBACK_MODEL_REVISION,
            FALLBACK_SERVED_MODEL_NAME,
        )
    ):
        raise CombinedMaterializationError("combined launcher differs from selected freeze")
    tokenizer = capture_tokenizer_manifest(
        snapshot_path,
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
    )
    if tokenizer.manifest_sha256 != selected_model_freeze.get("tokenizer_manifest_sha256"):
        raise CombinedMaterializationError("combined tokenizer differs from selected freeze")

    conditions = (ConditionName.C2_LLM_QUERY, *ACTIVE_ABLATION_CONDITIONS)
    runtimes = {
        condition: development_request_runtime(
            root=repository,
            condition=condition,
            tokenizer_manifest=tokenizer,
            seed=configuration.vllm_seed,
        )
        for condition in conditions
    }
    decoding_families = {
        runtime.decoding_manifest.comparison_family_hash for runtime in runtimes.values()
    }
    if len(decoding_families) != 1:
        raise CombinedMaterializationError("combined decoding families are not comparable")
    construction = DevelopmentConstructionConfiguration.load(
        repository / configuration.development_construction_path
    )
    validator_hash = development_validator_hash(repository)
    repair_policy_hash = canonical_sha256(
        {
            "policy": "one-preservation-repair-with-short-global-reserve-v1",
            "repair_prompt_file_sha256": _file_sha256(
                repository / "prompts/repair/prompt_v1.md"
            ),
            "maximum_repairs_per_call": configuration.maximum_repairs_per_call,
            "repair_reserve_class": configuration.repair_reserve_class,
            "repair_watchdog_seconds": configuration.repair_watchdog_seconds,
            "validator_hash": validator_hash,
        }
    )
    return CombinedRuntimeBinding(
        source_tree_association_hash=association_hash,
        selected_model_freeze_hash=freeze_hash,
        model_manifest_hash=launcher.configuration_hash,
        model_repository=configuration.selected_model_repository,
        model_revision=configuration.selected_model_revision,
        served_model_name=configuration.served_model_name,
        tokenizer_manifest_hash=tokenizer.manifest_sha256,
        launcher_configuration_hash=launcher.configuration_hash,
        runtime_version="vllm-0.10.2",
        decoding_configuration_file_sha256=(
            configuration.decoding_configuration_file_sha256
        ),
        seed_manifest_hash=configuration.seed_manifest_hash,
        validator_hash=validator_hash,
        upper_ontology_hash=construction.upper_ontology.content_hash,
        repair_policy_hash=repair_policy_hash,
        prompt_hashes={key: value.prompt_hash for key, value in runtimes.items()},
        output_schema_hashes={
            key: value.output_schema_hash for key, value in runtimes.items()
        },
        decoding_manifest_hashes={
            key: value.decoding_manifest.content_hash for key, value in runtimes.items()
        },
        decoding_family_hash=next(iter(decoding_families)),
        capability_manifest_hashes={
            key: value.capability_manifest.content_hash for key, value in runtimes.items()
        },
        maximum_input_tokens=construction.first_pass_input_tokens,
        maximum_output_tokens=construction.first_pass_output_tokens,
        repair_maximum_input_tokens=construction.repair_input_tokens,
        repair_maximum_output_tokens=construction.repair_output_tokens,
        structured_decoder=STRUCTURED_DECODER,
    )


def compile_combined_production_inputs(
    *,
    configuration: CombinedBlockConfiguration,
    selections: RegisteredCombinedSelections,
    runtime_binding: CombinedRuntimeBinding,
    development: DevelopmentExecutionResult,
    held_out_calls: HeldOutCallManifest,
    held_out_execution: HeldOutExecutionManifest,
    scorer_bridge: ScorerBridgeAuthorization,
    final_schedule: GlobalGpuScheduleSnapshot,
    phase5_inputs: Phase5ExecutionInputManifest,
    phase5_primary_gate: PrimaryHeldOutResultsGate,
    compiled_at: datetime,
) -> CompiledCombinedInputs:
    """Compile and cross-validate every required combined controller input."""

    contexts = {
        opening.sealed_stage_hash: opening.query_context
        for opening in held_out_execution.query_openings
        if opening.query_context is not None
    }
    if len(contexts) != 36:
        raise CombinedMaterializationError("held-out execution lacks 36 opened contexts")
    if (
        phase5_inputs.primary_results_gate_hash != phase5_primary_gate.content_hash
        or phase5_primary_gate.held_out_execution_manifest_hash
        != held_out_execution.content_hash
        or phase5_primary_gate.scorer_bridge_hash != scorer_bridge.content_hash
        or phase5_primary_gate.result_artifact_hashes
        != scorer_bridge.output_artifact_hashes
    ):
        raise CombinedMaterializationError("Phase 5 primary gate differs from held-out closure")
    upstream = replay_combined_upstream_gate(
        development=development,
        held_out_calls=held_out_calls,
        held_out_execution=held_out_execution,
        scorer_bridge=scorer_bridge,
        final_schedule=final_schedule,
        phase5_inputs=phase5_inputs,
        verified_at=compiled_at,
    )
    sources = compile_combined_query_sources(
        held_out_calls=held_out_calls,
        held_out_execution=held_out_execution,
        scorer_bridge=scorer_bridge,
        contexts_by_query_stage_hash=cast(Mapping[str, Any], contexts),
    )
    manifest = compile_combined_call_manifest(
        configuration=configuration,
        selections=selections,
        sources_by_context_id=sources,
        runtime_binding=runtime_binding,
        upstream_gate=upstream,
        created_at=compiled_at,
    )
    validate_combined_execution_inputs(
        configuration=configuration,
        manifest=manifest,
        runtime=runtime_binding,
        upstream_gate=upstream,
        phase5_inputs=phase5_inputs,
        phase5_protocol=selections.feedback,
        phase5_prerequisites=phase5_primary_gate,
    )
    return CompiledCombinedInputs(runtime_binding, upstream, manifest)


def verify_combined_compiler_receipt(
    *,
    compiler_manifest_path: Path,
    call_manifest_path: Path,
    runtime_binding_path: Path,
    upstream_gate_path: Path,
    source_paths: Mapping[str, Path],
    configuration: CombinedBlockConfiguration,
    selections: RegisteredCombinedSelections,
    runtime_binding: CombinedRuntimeBinding,
    upstream_gate: CombinedUpstreamGate,
    call_manifest: CombinedCallManifest,
    phase5_inputs: Phase5ExecutionInputManifest,
    phase5_primary_gate: PrimaryHeldOutResultsGate,
) -> CombinedCompilerManifest:
    """Replay the CPU compiler receipt and every source/output byte binding."""

    output_paths = {
        "runtime_binding": runtime_binding_path.absolute(),
        "upstream_gate": upstream_gate_path.absolute(),
        "call_manifest": call_manifest_path.absolute(),
    }
    compiler_path = compiler_manifest_path.absolute()
    expected_names = {
        "runtime_binding": "runtime_binding.json",
        "upstream_gate": "upstream_gate.json",
        "call_manifest": "call_manifest.json",
    }
    parents = {path.parent for path in (*output_paths.values(), compiler_path)}
    if len(parents) != 1 or compiler_path.name != "compiler_manifest.json":
        raise CombinedMaterializationError(
            "combined compiler receipt and outputs must share one canonical directory"
        )
    if any(path.name != expected_names[name] for name, path in output_paths.items()):
        raise CombinedMaterializationError("combined compiler output filename changed")
    compiler_raw = _safe_file(compiler_path, label="combined compiler receipt")
    try:
        receipt = CombinedCompilerManifest.model_validate_json(compiler_raw)
    except Exception as error:
        raise CombinedMaterializationError("invalid combined compiler receipt") from error
    if receipt.configuration_hash != configuration.content_hash:
        raise CombinedMaterializationError("combined compiler configuration hash changed")

    typed_outputs = {
        "runtime_binding": runtime_binding,
        "upstream_gate": upstream_gate,
        "call_manifest": call_manifest,
    }
    for output in receipt.outputs:
        raw = _safe_file(
            output_paths[output.logical_name],
            label=f"combined compiler {output.logical_name}",
        )
        if (
            hashlib.sha256(raw).hexdigest() != output.file_sha256
            or typed_outputs[output.logical_name].content_hash
            != output.logical_content_hash
            or raw != _record_bytes(typed_outputs[output.logical_name])
        ):
            raise CombinedMaterializationError(
                f"combined compiler {output.logical_name} byte/logical hash changed"
            )

    if set(source_paths) != set(receipt.source_file_sha256s):
        raise CombinedMaterializationError("combined compiler source-path inventory changed")
    source_raw = {
        name: _safe_file(path, label=f"combined compiler source {name}")
        for name, path in source_paths.items()
    }
    if any(
        hashlib.sha256(source_raw[name]).hexdigest() != expected
        for name, expected in receipt.source_file_sha256s.items()
    ):
        raise CombinedMaterializationError("combined compiler source-file hash changed")
    try:
        development = DevelopmentExecutionResult.model_validate_json(
            source_raw["development_result"]
        )
        held_out_calls = HeldOutCallManifest.model_validate_json(
            source_raw["held_out_call_manifest"]
        )
        held_out_execution = HeldOutExecutionManifest.model_validate_json(
            source_raw["held_out_execution"]
        )
        scorer_bridge = ScorerBridgeAuthorization.model_validate_json(
            source_raw["scorer_bridge"]
        )
        final_schedule = GlobalGpuScheduleSnapshot.model_validate_json(
            source_raw["final_schedule"]
        )
        source_phase5_inputs = Phase5ExecutionInputManifest.model_validate_json(
            source_raw["phase5_inputs"]
        )
        source_phase5_gate = PrimaryHeldOutResultsGate.model_validate_json(
            source_raw["phase5_primary_gate"]
        )
        selected_model_freeze = json.loads(source_raw["selected_model_freeze"])
        source_association = json.loads(source_raw["source_association"])
    except Exception as error:
        raise CombinedMaterializationError(
            "combined compiler source record is invalid"
        ) from error
    if not isinstance(selected_model_freeze, Mapping) or not isinstance(
        source_association,
        Mapping,
    ):
        raise CombinedMaterializationError("combined compiler mapping source is invalid")
    selected_hash = _self_hash(
        cast(Mapping[str, object], selected_model_freeze),
        label="selected-model freeze",
    )
    association_hash = _self_hash(
        cast(Mapping[str, object], source_association),
        label="source association",
    )
    expected_logical = {
        "development_result": development.content_hash,
        "held_out_call_manifest": held_out_calls.content_hash,
        "held_out_execution": held_out_execution.content_hash,
        "scorer_bridge": scorer_bridge.content_hash,
        "final_schedule": final_schedule.content_hash,
        "phase5_inputs": source_phase5_inputs.content_hash,
        "phase5_primary_gate": source_phase5_gate.content_hash,
        "selected_model_freeze": selected_hash,
        "source_association": association_hash,
        "registered_selections": selections.content_hash,
        "runtime_binding": runtime_binding.content_hash,
    }
    if dict(receipt.source_logical_hashes) != expected_logical:
        raise CombinedMaterializationError("combined compiler source-logical hashes changed")
    if (
        source_phase5_inputs != phase5_inputs
        or source_phase5_gate != phase5_primary_gate
        or runtime_binding.selected_model_freeze_hash != selected_hash
        or runtime_binding.source_tree_association_hash != association_hash
        or call_manifest.registered_selections_hash
        != combined_selection_registry_hash(selections)
        or call_manifest.created_at != receipt.compiled_at
        or upstream_gate.verified_at != receipt.compiled_at
    ):
        raise CombinedMaterializationError("combined compiler receipt binding changed")
    replay = compile_combined_production_inputs(
        configuration=configuration,
        selections=selections,
        runtime_binding=runtime_binding,
        development=development,
        held_out_calls=held_out_calls,
        held_out_execution=held_out_execution,
        scorer_bridge=scorer_bridge,
        final_schedule=final_schedule,
        phase5_inputs=phase5_inputs,
        phase5_primary_gate=phase5_primary_gate,
        compiled_at=receipt.compiled_at,
    )
    if (
        replay.runtime_binding != runtime_binding
        or replay.upstream_gate != upstream_gate
        or replay.call_manifest != call_manifest
    ):
        raise CombinedMaterializationError(
            "combined controller inputs do not replay from the compiler sources"
        )
    return receipt


def _record_bytes(record: ImmutableRecord) -> bytes:
    return (record.to_canonical_json() + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exact(path: Path, payload: bytes) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | nofollow)
    temporary_name = f".{path.name}.{hashlib.sha256(payload).hexdigest()}.tmp"
    try:
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow,
                0o600,
                dir_fd=directory,
            )
        except FileExistsError as error:
            try:
                existing_temporary = os.open(
                    temporary_name,
                    os.O_RDONLY | nofollow,
                    dir_fd=directory,
                )
            except OSError as open_error:
                raise CombinedMaterializationError(
                    f"immutable combined compiler temporary changed: {path.name}"
                ) from open_error
            try:
                observed = b""
                while block := os.read(existing_temporary, 1024 * 1024):
                    observed += block
            finally:
                os.close(existing_temporary)
            if observed != payload:
                raise CombinedMaterializationError(
                    f"immutable combined compiler temporary changed: {path.name}"
                ) from error
        else:
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("combined compiler write made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            existing = os.open(path.name, os.O_RDONLY | nofollow, dir_fd=directory)
            try:
                observed = b""
                while block := os.read(existing, 1024 * 1024):
                    observed += block
            finally:
                os.close(existing)
            if observed != payload:
                raise CombinedMaterializationError(
                    f"immutable combined compiler output changed: {path.name}"
                ) from error
        os.fsync(directory)
        if os.fstat(directory) != os.stat(path.parent, follow_symlinks=False):
            raise CombinedMaterializationError(
                "combined compiler output directory changed during publication"
            )
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=directory)
        os.close(directory)


def materialize_combined_inputs(
    *,
    repository: Path,
    output_root: Path,
    compiled: CompiledCombinedInputs,
    source_file_sha256s: Mapping[str, str],
    source_logical_hashes: Mapping[str, str],
    compiled_at: datetime,
) -> CombinedCompilerManifest:
    """Publish the three controller inputs and a self-hashed compiler receipt."""

    repository = repository.resolve(strict=True)
    restricted_root = (repository / "artifacts/restricted").resolve(strict=True)
    candidate = output_root if output_root.is_absolute() else repository / output_root
    current = repository
    try:
        relative = candidate.absolute().relative_to(repository)
    except ValueError as error:
        raise CombinedMaterializationError("combined output root escaped repository") from error
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise CombinedMaterializationError("combined output root uses a symbolic link")
    candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    output_root = candidate.resolve(strict=True)
    try:
        output_root.relative_to(restricted_root)
    except ValueError as error:
        raise CombinedMaterializationError(
            "combined compiler outputs must remain under artifacts/restricted"
        ) from error
    os.chmod(output_root, 0o700)

    records: tuple[tuple[str, str, ImmutableRecord], ...] = (
        ("runtime_binding", "runtime_binding.json", compiled.runtime_binding),
        ("upstream_gate", "upstream_gate.json", compiled.upstream_gate),
        ("call_manifest", "call_manifest.json", compiled.call_manifest),
    )
    outputs = []
    for logical_name, filename, record in records:
        payload = _record_bytes(record)
        _publish_exact(output_root / filename, payload)
        outputs.append(
            CombinedCompilerOutput(
                logical_name=logical_name,
                file_sha256=hashlib.sha256(payload).hexdigest(),
                logical_content_hash=record.content_hash,
            )
        )
    typed_outputs = cast(
        tuple[
            CombinedCompilerOutput,
            CombinedCompilerOutput,
            CombinedCompilerOutput,
        ],
        tuple(outputs),
    )
    compiler = CombinedCompilerManifest(
        compiler_id=f"combined-inputs-{compiled.call_manifest.content_hash[:20]}",
        configuration_hash=compiled.call_manifest.configuration_hash,
        source_file_sha256s=source_file_sha256s,
        source_logical_hashes=source_logical_hashes,
        outputs=typed_outputs,
        compiled_at=compiled_at,
    )
    _publish_exact(output_root / "compiler_manifest.json", _record_bytes(compiler))
    return compiler


__all__ = [
    "CombinedCompilerManifest",
    "CombinedCompilerOutput",
    "CombinedMaterializationError",
    "CompiledCombinedInputs",
    "CompilerSourceSnapshot",
    "compile_combined_production_inputs",
    "compile_combined_runtime_binding",
    "load_mapping_file",
    "load_mapping_snapshot",
    "load_record_file",
    "load_record_snapshot",
    "materialize_combined_inputs",
    "validate_source_association_snapshot",
    "verify_combined_compiler_receipt",
]
