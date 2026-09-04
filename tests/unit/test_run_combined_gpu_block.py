from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

from story_projection_onto.combined_gpu_block import compile_combined_call_manifest
from tests.unit.test_combined_gpu_block import REPOSITORY, combined_fixture
from tests.unit.test_combined_gpu_production import _rebuild
from tests.unit.test_phase5_execution import _inputs


def _script_module():
    path = REPOSITORY / "scripts/run_combined_gpu_block.py"
    spec = importlib.util.spec_from_file_location("run_combined_gpu_block", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path):
    configuration, selections, sources, runtime, upstream, _manifest = combined_fixture()
    protocol, original_phase5_gate, original_phase5_inputs = _inputs()
    phase5_gate = _rebuild(
        original_phase5_gate,
        final_reviewed_seal_hash=upstream.final_reviewed_seal_hash,
    )
    phase5_inputs = _rebuild(
        original_phase5_inputs,
        protocol_hash=protocol.content_hash,
        primary_results_gate_hash=phase5_gate.content_hash,
        final_reviewed_seal_hash=upstream.final_reviewed_seal_hash,
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
    values = {
        "manifest": manifest,
        "runtime": runtime,
        "upstream": upstream,
        "phase5-inputs": phase5_inputs,
        "phase5-gate": phase5_gate,
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(value.to_canonical_json() + "\n", encoding="utf-8")
        paths[name] = path
    for name in (
        "development-result",
        "held-out-manifest",
        "held-out-execution",
        "scorer-bridge",
        "final-schedule",
        "selected-model-freeze",
        "source-association",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    compiler = tmp_path / "compiler_manifest.json"
    compiler.write_text("{}\n", encoding="utf-8")
    paths["compiler-manifest"] = compiler
    return manifest, paths


def _compiler_arguments(paths: dict[str, Path]) -> list[str]:
    return [
        "--compiler-manifest",
        str(paths["compiler-manifest"]),
        "--development-result",
        str(paths["development-result"]),
        "--held-out-manifest",
        str(paths["held-out-manifest"]),
        "--held-out-execution",
        str(paths["held-out-execution"]),
        "--scorer-bridge",
        str(paths["scorer-bridge"]),
        "--final-schedule",
        str(paths["final-schedule"]),
        "--selected-model-freeze",
        str(paths["selected-model-freeze"]),
        "--source-association",
        str(paths["source-association"]),
    ]


def test_cli_validate_only_reports_exact_inventory_without_writes(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    manifest, paths = _write_inputs(tmp_path)
    module = _script_module()
    monkeypatch.setattr(module, "verify_combined_compiler_receipt", lambda **_kwargs: None)
    result = module.main(
        [
            "--repository",
            str(REPOSITORY),
            "--manifest",
            str(paths["manifest"]),
            "--runtime-binding",
            str(paths["runtime"]),
            "--upstream-gate",
            str(paths["upstream"]),
            "--phase5-inputs",
            str(paths["phase5-inputs"]),
            "--phase5-primary-gate",
            str(paths["phase5-gate"]),
            *_compiler_arguments(paths),
            "--validate-only",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output == {
        "state": "validated",
        "writes_performed": False,
        "manifest_hash": manifest.content_hash,
        "runtime_binding_hash": paths["runtime"]
        and json.loads(paths["runtime"].read_text())["content_hash"],
        "upstream_gate_hash": json.loads(paths["upstream"].read_text())["content_hash"],
        "base_call_count": 49,
        "phase5_call_count": 9,
        "model_load_count": 1,
        "load_inclusive_forecast_seconds": 4835,
        "maximum_concurrency": 1,
    }


def test_cli_fails_closed_before_adapter_import_on_tampered_upstream(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _manifest, paths = _write_inputs(tmp_path)
    payload = json.loads(paths["upstream"].read_text(encoding="utf-8"))
    payload["phase5_input_manifest_hash"] = "0" * 64
    payload.pop("content_hash", None)
    paths["upstream"].write_text(json.dumps(payload), encoding="utf-8")
    module = _script_module()
    private_errors = []
    monkeypatch.setattr(
        module,
        "_persist_private_diagnostic",
        lambda _repository, error: private_errors.append(str(error)),
    )
    monkeypatch.setattr(module, "verify_combined_compiler_receipt", lambda **_kwargs: None)
    result = module.main(
        [
            "--repository",
            str(REPOSITORY),
            "--manifest",
            str(paths["manifest"]),
            "--runtime-binding",
            str(paths["runtime"]),
            "--upstream-gate",
            str(paths["upstream"]),
            "--phase5-inputs",
            str(paths["phase5-inputs"]),
            "--phase5-primary-gate",
            str(paths["phase5-gate"]),
            *_compiler_arguments(paths),
            "--validate-only",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert output == {
        "state": "blocked",
        "error_code": "combined_execution_blocked",
        "error_type": "CombinedProductionError",
    }
    assert private_errors and "accepted development" in private_errors[0]


def test_cli_diagnostic_keeps_path_bearing_error_private(tmp_path: Path) -> None:
    module = _script_module()
    (tmp_path / "artifacts/restricted").mkdir(parents=True)
    secret = tmp_path / "private-corpus-location.txt"
    module._persist_private_diagnostic(tmp_path, ValueError(f"failed at {secret}"))
    path = tmp_path / "artifacts/restricted/combined_gpu_block/diagnostics/cli-errors.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert str(secret) in payload["error_message"]
    assert path.stat().st_mode & 0o777 == 0o600


def test_cli_blocks_missing_or_invalid_compiler_receipt_before_factory(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _manifest, paths = _write_inputs(tmp_path)
    module = _script_module()
    monkeypatch.setattr(
        module,
        "_bundle_factory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("factory must not be reached")
        ),
    )
    for missing in (False, True):
        receipt = paths["compiler-manifest"]
        if missing:
            receipt.unlink(missing_ok=True)
        else:
            receipt.write_text("{}\n", encoding="utf-8")
        result = module.main(
            [
                "--repository",
                str(REPOSITORY),
                "--manifest",
                str(paths["manifest"]),
                "--runtime-binding",
                str(paths["runtime"]),
                "--upstream-gate",
                str(paths["upstream"]),
                "--phase5-inputs",
                str(paths["phase5-inputs"]),
                "--phase5-primary-gate",
                str(paths["phase5-gate"]),
                *_compiler_arguments(paths),
                "--validate-only",
            ]
        )
        assert result == 2
        assert json.loads(capsys.readouterr().out)["state"] == "blocked"


def test_cli_factory_is_one_frozen_concrete_export() -> None:
    configuration, *_rest = combined_fixture()
    module = _script_module()
    factory_module_name, _, export_name = (
        configuration.production_adapter_factory.partition(":")
    )
    export = getattr(importlib.import_module(factory_module_name), export_name)
    assert callable(export)
    try:
        module._bundle_factory(
            "tests.fake:factory",
            expected_reference=configuration.production_adapter_factory,
        )
    except Exception as error:
        assert "differs from the frozen" in str(error)
    else:
        raise AssertionError("arbitrary combined factory was accepted")
