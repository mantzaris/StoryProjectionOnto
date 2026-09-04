from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import story_projection_onto.combined_gpu_materialize as materialize
from story_projection_onto.combined_gpu_block import compile_combined_call_manifest
from story_projection_onto.combined_gpu_materialize import (
    CombinedMaterializationError,
    CompiledCombinedInputs,
    CompilerSourceSnapshot,
    load_mapping_snapshot,
    materialize_combined_inputs,
    validate_source_association_snapshot,
    verify_combined_compiler_receipt,
)
from story_projection_onto.contracts import canonical_sha256
from tests.unit.test_combined_gpu_block import combined_fixture
from tests.unit.test_combined_gpu_production import _rebuild
from tests.unit.test_phase5_execution import _inputs

NOW = datetime(2026, 9, 4, 20, 30, tzinfo=UTC)
SOURCE_NAMES = (
    "development_result",
    "held_out_call_manifest",
    "held_out_execution",
    "scorer_bridge",
    "final_schedule",
    "phase5_inputs",
    "phase5_primary_gate",
    "selected_model_freeze",
    "source_association",
)


def _compiled() -> CompiledCombinedInputs:
    _configuration, _selections, _sources, runtime, upstream, manifest = (
        combined_fixture()
    )
    return CompiledCombinedInputs(runtime, upstream, manifest)


def _script_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/materialize_combined_gpu_inputs.py"
    )
    spec = importlib.util.spec_from_file_location("materialize_combined_gpu_inputs", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hashes() -> tuple[dict[str, str], dict[str, str]]:
    files = {name: canonical_sha256({"test-file": name}) for name in SOURCE_NAMES}
    logical = {name: canonical_sha256({"test-logical": name}) for name in SOURCE_NAMES}
    logical.update(
        {
            "registered_selections": canonical_sha256({"test": "selections"}),
            "runtime_binding": canonical_sha256({"test": "runtime"}),
        }
    )
    return files, logical


def test_materializer_publishes_exact_three_inputs_and_self_hashed_receipt(
    tmp_path: Path,
) -> None:
    (tmp_path / "artifacts/restricted").mkdir(parents=True)
    files, logical = _hashes()
    compiled = _compiled()
    receipt = materialize_combined_inputs(
        repository=tmp_path,
        output_root=Path("artifacts/restricted/combined-inputs"),
        compiled=compiled,
        source_file_sha256s=files,
        source_logical_hashes=logical,
        compiled_at=NOW,
    )
    root = tmp_path / "artifacts/restricted/combined-inputs"
    assert sorted(path.name for path in root.iterdir()) == [
        "call_manifest.json",
        "compiler_manifest.json",
        "runtime_binding.json",
        "upstream_gate.json",
    ]
    assert json.loads((root / "compiler_manifest.json").read_text())["content_hash"] == (
        receipt.content_hash
    )
    replay = materialize_combined_inputs(
        repository=tmp_path,
        output_root=root,
        compiled=compiled,
        source_file_sha256s=files,
        source_logical_hashes=logical,
        compiled_at=NOW,
    )
    assert replay == receipt


def test_materializer_rejects_drift_and_restricted_root_escape(tmp_path: Path) -> None:
    (tmp_path / "artifacts/restricted").mkdir(parents=True)
    files, logical = _hashes()
    compiled = _compiled()
    arguments = {
        "repository": tmp_path,
        "compiled": compiled,
        "source_file_sha256s": files,
        "source_logical_hashes": logical,
        "compiled_at": NOW,
    }
    with pytest.raises(CombinedMaterializationError, match="artifacts/restricted"):
        materialize_combined_inputs(output_root=tmp_path / "public", **arguments)

    root = tmp_path / "artifacts/restricted/combined-inputs"
    materialize_combined_inputs(output_root=root, **arguments)
    (root / "call_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CombinedMaterializationError, match="output changed"):
        materialize_combined_inputs(output_root=root, **arguments)


def test_materializer_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    (tmp_path / "artifacts/restricted").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "artifacts/restricted/link").symlink_to(outside, target_is_directory=True)
    files, logical = _hashes()
    with pytest.raises(CombinedMaterializationError, match="symbolic link"):
        materialize_combined_inputs(
            repository=tmp_path,
            output_root=Path("artifacts/restricted/link/combined"),
            compiled=_compiled(),
            source_file_sha256s=files,
            source_logical_hashes=logical,
            compiled_at=NOW,
        )


def test_source_association_validation_does_not_reopen_captured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree_hash = canonical_sha256({"tree": "captured"})
    local_manifest = {
        "schema_version": "1.0.0",
        "revision": "captured-revision",
        "files": [],
        "tree_sha256": tree_hash,
    }
    local_manifest_raw = (json.dumps(local_manifest, sort_keys=True) + "\n").encode()
    (tmp_path / "source-manifest.json").write_bytes(local_manifest_raw)
    association = {
        "kind": "local_remote_source_tree_association",
        "branch": "implementation/query-dependent-temporal-ontology",
        "revision_label": "captured-revision",
        "local_manifest": "source-manifest.json",
        "local_manifest_file_sha256": hashlib.sha256(
            local_manifest_raw
        ).hexdigest(),
        "local_tree_sha256": tree_hash,
        "remote_tree_sha256": tree_hash,
    }
    association["manifest_sha256"] = canonical_sha256(association)
    association_path = tmp_path / "source-association.json"
    association_path.write_text(json.dumps(association), encoding="utf-8")
    snapshot = load_mapping_snapshot(association_path, label="source association")

    replacement = tmp_path / "replacement-association.json"
    replacement.write_text('{"kind":"attacker-controlled"}\n', encoding="utf-8")
    association_path.unlink()
    association_path.symlink_to(replacement)
    monkeypatch.setattr(
        materialize,
        "build_source_manifest",
        lambda _root, _revision: SimpleNamespace(
            to_dict=lambda: local_manifest,
        ),
    )

    validated = validate_source_association_snapshot(
        snapshot,
        association_path=association_path,
        source_root=tmp_path,
    )
    assert validated == association
    assert snapshot.file_sha256 == hashlib.sha256(snapshot.raw).hexdigest()


def test_compiler_cli_receipt_uses_single_snapshots_during_symlink_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _script_module()
    source_root = tmp_path / "compiler-sources"
    source_root.mkdir()
    source_paths: dict[str, Path] = {}
    original_bytes: dict[str, bytes] = {}
    for name in SOURCE_NAMES:
        payload = {"source": name, "manifest_sha256": canonical_sha256({"source": name})}
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
        path = source_root / f"{name}.json"
        path.write_bytes(raw)
        source_paths[name] = path
        original_bytes[name] = raw
    replacement = source_root / "replacement.json"
    replacement.write_text('{"source":"swapped"}\n', encoding="utf-8")

    class SourceParser:
        @staticmethod
        def model_validate_json(raw: bytes) -> SimpleNamespace:
            name = json.loads(raw)["source"]
            return SimpleNamespace(content_hash=canonical_sha256({"logical": name}))

    for model_name in (
        "DevelopmentExecutionResult",
        "HeldOutCallManifest",
        "HeldOutExecutionManifest",
        "ScorerBridgeAuthorization",
        "GlobalGpuScheduleSnapshot",
        "Phase5ExecutionInputManifest",
        "PrimaryHeldOutResultsGate",
    ):
        monkeypatch.setattr(module, model_name, SourceParser)

    original_safe_file = materialize._safe_file
    read_counts = {name: 0 for name in SOURCE_NAMES}

    def swap_after_descriptor_read(path: Path, *, label: str) -> bytes:
        raw = original_safe_file(path, label=label)
        for name, source_path in source_paths.items():
            if path.absolute() == source_path.absolute():
                read_counts[name] += 1
                source_path.unlink()
                source_path.symlink_to(replacement)
                break
        return raw

    monkeypatch.setattr(materialize, "_safe_file", swap_after_descriptor_read)
    configuration = SimpleNamespace(content_hash=canonical_sha256({"configuration": 1}))
    selections = SimpleNamespace(content_hash=canonical_sha256({"selections": 1}))
    runtime = SimpleNamespace(content_hash=canonical_sha256({"runtime": 1}))
    upstream = SimpleNamespace(content_hash=canonical_sha256({"upstream": 1}))
    call_manifest = SimpleNamespace(
        content_hash=canonical_sha256({"calls": 1}),
        calls=(object(),),
    )
    compiled = SimpleNamespace(upstream_gate=upstream, call_manifest=call_manifest)
    receipt = SimpleNamespace(content_hash=canonical_sha256({"receipt": 1}))
    captured_hashes: dict[str, str] = {}

    monkeypatch.setattr(module, "load_combined_configuration", lambda *_args: configuration)
    monkeypatch.setattr(
        module,
        "load_registered_combined_selections",
        lambda *_args: selections,
    )

    def validate_snapshot(
        snapshot: CompilerSourceSnapshot[dict[str, object]],
        **_kwargs: object,
    ) -> dict[str, object]:
        assert snapshot.value["source"] == "source_association"
        assert source_paths["source_association"].is_symlink()
        return snapshot.value

    monkeypatch.setattr(module, "validate_source_association_snapshot", validate_snapshot)
    monkeypatch.setattr(
        module,
        "compile_combined_runtime_binding",
        lambda **_kwargs: runtime,
    )
    monkeypatch.setattr(
        module,
        "compile_combined_production_inputs",
        lambda **_kwargs: compiled,
    )

    def capture_receipt(**kwargs: object) -> SimpleNamespace:
        captured_hashes.update(kwargs["source_file_sha256s"])
        return receipt

    monkeypatch.setattr(module, "materialize_combined_inputs", capture_receipt)
    arguments = ["--repository", str(tmp_path)]
    options = {
        "development-result": "development_result",
        "held-out-manifest": "held_out_call_manifest",
        "held-out-execution": "held_out_execution",
        "scorer-bridge": "scorer_bridge",
        "final-schedule": "final_schedule",
        "phase5-inputs": "phase5_inputs",
        "phase5-primary-gate": "phase5_primary_gate",
        "selected-model-freeze": "selected_model_freeze",
        "source-association": "source_association",
    }
    for option, name in options.items():
        arguments.extend((f"--{option}", str(source_paths[name])))
    arguments.extend(
        (
            "--snapshot",
            str(tmp_path / "unused-snapshot"),
            "--shared-cache",
            str(tmp_path / "unused-cache"),
            "--compiled-at",
            "2026-09-04T20:30:00Z",
        )
    )

    assert module.main(arguments) == 0
    assert read_counts == dict.fromkeys(SOURCE_NAMES, 1)
    assert captured_hashes == {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in original_bytes.items()
    }
    assert all(path.is_symlink() for path in source_paths.values())
    assert json.loads(capsys.readouterr().out)["state"] == "complete"


def test_compiler_receipt_replays_sources_and_rejects_tampered_output_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration, selections, sources, runtime, upstream, _manifest = combined_fixture()
    protocol, phase5_gate, phase5_inputs = _inputs()
    phase5_gate = _rebuild(
        phase5_gate,
        final_reviewed_seal_hash=upstream.final_reviewed_seal_hash,
    )
    phase5_inputs = _rebuild(
        phase5_inputs,
        protocol_hash=protocol.content_hash,
        primary_results_gate_hash=phase5_gate.content_hash,
        final_reviewed_seal_hash=upstream.final_reviewed_seal_hash,
    )
    selected_payload = {"kind": "test-selected-model"}
    selected_payload["manifest_sha256"] = canonical_sha256(selected_payload)
    association_payload = {"kind": "test-source-association"}
    association_payload["manifest_sha256"] = canonical_sha256(association_payload)
    runtime = _rebuild(
        runtime,
        selected_model_freeze_hash=selected_payload["manifest_sha256"],
        source_tree_association_hash=association_payload["manifest_sha256"],
    )
    upstream = _rebuild(
        upstream,
        phase5_input_manifest_hash=phase5_inputs.content_hash,
    )
    manifest = compile_combined_call_manifest(
        configuration=configuration,
        selections=selections,
        sources_by_context_id=sources,
        runtime_binding=runtime,
        upstream_gate=upstream,
        created_at=upstream.verified_at,
    )
    def record(content_hash: str):
        return type("Record", (), {"content_hash": content_hash})()

    parsed_sources = {
        "development_result": record(upstream.development_execution_result_hash),
        "held_out_call_manifest": record(upstream.held_out_call_manifest_hash),
        "held_out_execution": record(upstream.held_out_execution_manifest_hash),
        "scorer_bridge": record(upstream.held_out_scorer_bridge_hash),
        "final_schedule": record(upstream.held_out_final_schedule_snapshot_hash),
        "phase5_inputs": phase5_inputs,
        "phase5_primary_gate": phase5_gate,
    }
    parser_names = {
        "development_result": "DevelopmentExecutionResult",
        "held_out_call_manifest": "HeldOutCallManifest",
        "held_out_execution": "HeldOutExecutionManifest",
        "scorer_bridge": "ScorerBridgeAuthorization",
        "final_schedule": "GlobalGpuScheduleSnapshot",
        "phase5_inputs": "Phase5ExecutionInputManifest",
        "phase5_primary_gate": "PrimaryHeldOutResultsGate",
    }
    for source_name, class_name in parser_names.items():
        parsed = parsed_sources[source_name]
        monkeypatch.setattr(
            materialize,
            class_name,
            type(
                f"{class_name}Parser",
                (),
                {"model_validate_json": staticmethod(lambda _raw, value=parsed: value)},
            ),
        )

    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_paths: dict[str, Path] = {}
    for source_name in SOURCE_NAMES:
        path = source_root / f"{source_name}.json"
        if source_name == "selected_model_freeze":
            payload = selected_payload
        elif source_name == "source_association":
            payload = association_payload
        else:
            payload = {"source": source_name}
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        source_paths[source_name] = path
    source_file_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    source_logical_hashes = {
        name: record.content_hash for name, record in parsed_sources.items()
    }
    source_logical_hashes.update(
        {
            "selected_model_freeze": selected_payload["manifest_sha256"],
            "source_association": association_payload["manifest_sha256"],
            "registered_selections": selections.content_hash,
            "runtime_binding": runtime.content_hash,
        }
    )
    compiled = CompiledCombinedInputs(runtime, upstream, manifest)
    output_root = tmp_path / "artifacts/restricted/combined-inputs"
    output_root.parent.mkdir(parents=True)
    materialize_combined_inputs(
        repository=tmp_path,
        output_root=output_root,
        compiled=compiled,
        source_file_sha256s=source_file_hashes,
        source_logical_hashes=source_logical_hashes,
        compiled_at=upstream.verified_at,
    )
    monkeypatch.setattr(
        materialize,
        "compile_combined_production_inputs",
        lambda **_kwargs: compiled,
    )
    arguments = dict(
        compiler_manifest_path=output_root / "compiler_manifest.json",
        call_manifest_path=output_root / "call_manifest.json",
        runtime_binding_path=output_root / "runtime_binding.json",
        upstream_gate_path=output_root / "upstream_gate.json",
        source_paths=source_paths,
        configuration=configuration,
        selections=selections,
        runtime_binding=runtime,
        upstream_gate=upstream,
        call_manifest=manifest,
        phase5_inputs=phase5_inputs,
        phase5_primary_gate=phase5_gate,
    )
    assert verify_combined_compiler_receipt(**arguments).configuration_hash == (
        configuration.content_hash
    )

    (output_root / "call_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CombinedMaterializationError, match="byte/logical hash changed"):
        verify_combined_compiler_receipt(**arguments)


def test_compiler_cli_reports_only_stable_public_error_code(
    tmp_path: Path,
    capsys,
) -> None:
    module = _script_module()
    missing = tmp_path / "private-corpus-title-and-path.json"
    arguments = ["--repository", str(Path(__file__).resolve().parents[2])]
    for option in (
        "development-result",
        "held-out-manifest",
        "held-out-execution",
        "scorer-bridge",
        "final-schedule",
        "phase5-inputs",
        "phase5-primary-gate",
        "selected-model-freeze",
        "source-association",
        "snapshot",
        "shared-cache",
    ):
        arguments.extend((f"--{option}", str(missing)))
    arguments.extend(("--compiled-at", "2026-09-04T20:30:00Z"))
    assert module.main(arguments) == 2
    assert json.loads(capsys.readouterr().out) == {
        "state": "blocked",
        "error_code": "combined_input_compilation_failed",
    }


def test_compiler_cli_redacts_real_repository_symlink_loop(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _script_module()
    repository_loop = tmp_path / "private-compiler-repository-loop"
    repository_loop.symlink_to(repository_loop.name)
    private_source = tmp_path / "private-source.json"
    arguments = ["--repository", str(repository_loop)]
    for option in (
        "development-result",
        "held-out-manifest",
        "held-out-execution",
        "scorer-bridge",
        "final-schedule",
        "phase5-inputs",
        "phase5-primary-gate",
        "selected-model-freeze",
        "source-association",
        "snapshot",
        "shared-cache",
    ):
        arguments.extend((f"--{option}", str(private_source)))
    arguments.extend(("--compiled-at", "2026-09-04T20:30:00Z"))

    assert module.main(arguments) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "state": "blocked",
        "error_code": "combined_input_compilation_failed",
    }
    assert str(repository_loop) not in captured.out
    assert str(repository_loop) not in captured.err


@pytest.mark.parametrize(
    ("error", "expected_state", "expected_code", "expected_return"),
    (
        (
            KeyboardInterrupt("private compiler interrupt"),
            "interrupted",
            "combined_input_compilation_interrupted",
            130,
        ),
        (
            SystemExit(17),
            "terminated",
            "combined_input_compilation_system_exit",
            17,
        ),
        (
            SystemExit("private compiler exit"),
            "terminated",
            "combined_input_compilation_system_exit",
            2,
        ),
    ),
)
def test_compiler_cli_preserves_interrupt_and_system_exit_policy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    expected_state: str,
    expected_code: str,
    expected_return: int,
) -> None:
    module = _script_module()

    def stop(*_args: object) -> None:
        raise error

    monkeypatch.setattr(module, "load_combined_configuration", stop)
    private_source = tmp_path / "private-source.json"
    arguments = ["--repository", str(tmp_path)]
    for option in (
        "development-result",
        "held-out-manifest",
        "held-out-execution",
        "scorer-bridge",
        "final-schedule",
        "phase5-inputs",
        "phase5-primary-gate",
        "selected-model-freeze",
        "source-association",
        "snapshot",
        "shared-cache",
    ):
        arguments.extend((f"--{option}", str(private_source)))
    arguments.extend(("--compiled-at", "2026-09-04T20:30:00Z"))

    assert module.main(arguments) == expected_return
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "state": expected_state,
        "error_code": expected_code,
    }
    assert str(error) not in captured.out
    assert str(error) not in captured.err
