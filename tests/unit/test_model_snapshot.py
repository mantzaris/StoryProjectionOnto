from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_model_snapshot.py"
SPEC = importlib.util.spec_from_file_location("verify_model_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    revision = "a" * 40
    cache = tmp_path / "cache"
    snapshot = cache / "hub" / "models--Example--Model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    files = {
        "LICENSE": "Apache License\nVersion 2.0, January 2004\n",
        "config.json": json.dumps({"quantization_config": {"quant_method": "awq"}}),
        "generation_config.json": "{}",
        "model.safetensors.index.json": "{}",
        "model-00001-of-00001.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
    }
    for name, content in files.items():
        (snapshot / name).write_text(content, encoding="utf-8")
    configuration = tmp_path / "model.json"
    configuration.write_text(
        json.dumps({"repository": "Example/Model", "revision": revision}),
        encoding="utf-8",
    )
    return cache, snapshot, configuration


def test_verify_snapshot_hashes_one_pinned_awq_snapshot(tmp_path: Path) -> None:
    cache, snapshot, configuration = make_snapshot(tmp_path)
    manifest = MODULE.verify_snapshot(
        shared_cache=cache,
        snapshot=snapshot,
        model_configuration_path=configuration,
    )

    assert manifest["repository"] == "Example/Model"
    assert manifest["revision"] == "a" * 40
    assert manifest["license"] == "Apache-2.0"
    assert manifest["file_count"] == 7
    assert len(manifest["manifest_sha256"]) == 64


def test_verify_snapshot_rejects_second_revision_and_partial(tmp_path: Path) -> None:
    cache, snapshot, configuration = make_snapshot(tmp_path)
    second = snapshot.parent / ("b" * 40)
    second.mkdir()
    with pytest.raises(ValueError, match="exactly one pinned snapshot"):
        MODULE.verify_snapshot(
            shared_cache=cache,
            snapshot=snapshot,
            model_configuration_path=configuration,
        )
    second.rmdir()
    partial = cache / "hub" / "stale.incomplete"
    partial.write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.verify_snapshot(
            shared_cache=cache,
            snapshot=snapshot,
            model_configuration_path=configuration,
        )


def test_verify_snapshot_rejects_right_revision_under_wrong_repository(tmp_path: Path) -> None:
    cache, snapshot, configuration = make_snapshot(tmp_path)
    configured = json.loads(configuration.read_text(encoding="utf-8"))
    configured["repository"] = "Different/Model"
    configuration.write_text(json.dumps(configured), encoding="utf-8")

    with pytest.raises(ValueError, match="configured model repository"):
        MODULE.verify_snapshot(
            shared_cache=cache,
            snapshot=snapshot,
            model_configuration_path=configuration,
        )
