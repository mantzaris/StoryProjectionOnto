from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from story_projection_onto.held_out_factory import HeldOutFactoryError

ROOT = Path(__file__).resolve().parents[2]


def _test_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    restricted = repository / "artifacts" / "restricted"
    restricted.mkdir(parents=True)
    return repository, restricted


def _script_module():
    path = ROOT / "scripts/run_held_out_primary.py"
    specification = importlib.util.spec_from_file_location(
        "run_held_out_primary_test_module",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "error",
    (
        HeldOutFactoryError("private-factory-detail:/restricted/factory"),
        RuntimeError("private-runtime-detail:/restricted/runtime"),
        sqlite3.DatabaseError("private-sqlite-detail:/restricted/ledger.sqlite3"),
        OSError(5, "private-path-detail", "/restricted/private-input.json"),
    ),
)
def test_arbitrary_control_error_is_redacted_publicly_and_retained_only_restricted(
    tmp_path: Path,
    capsys,
    monkeypatch,
    error: BaseException,
) -> None:
    module = _script_module()
    repository, restricted = _test_repository(tmp_path)
    secret = str(error)

    def fail_review(**_kwargs):
        raise error

    monkeypatch.setattr(module, "open_reviewed_held_out_plan", fail_review)
    assert (
        module.main(
            [
                "--repository",
                str(repository),
                "--validate-only",
                "--restricted-root",
                str(restricted),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    public = json.loads(captured.out)
    assert public == {
        "error": "Held-out execution is blocked; inspect restricted logs",
        "error_code": "held_out_control_blocked",
        "state": "blocked",
    }
    assert secret not in captured.out
    assert secret not in captured.err
    records = tuple((restricted / "held_out_primary/control_errors").glob("*.json"))
    assert len(records) == 1
    assert secret in records[0].read_text(encoding="utf-8")
    assert records[0].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("error", "expected_state", "expected_code", "expected_return"),
    (
        (KeyboardInterrupt("private interrupt"), "interrupted", "held_out_interrupted", 130),
        (SystemExit("private exit"), "terminated", "held_out_system_exit", 2),
    ),
)
def test_control_boundary_preserves_interrupt_exit_policy_without_detail_leak(
    tmp_path: Path,
    capsys,
    monkeypatch,
    error: BaseException,
    expected_state: str,
    expected_code: str,
    expected_return: int,
) -> None:
    module = _script_module()
    repository, restricted = _test_repository(tmp_path)

    def stop(**_kwargs):
        raise error

    monkeypatch.setattr(module, "open_reviewed_held_out_plan", stop)
    result = module.main(
        [
            "--repository",
            str(repository),
            "--validate-only",
            "--restricted-root",
            str(restricted),
        ]
    )
    public = json.loads(capsys.readouterr().out)
    assert result == expected_return
    assert public["state"] == expected_state
    assert public["error_code"] == expected_code
    assert str(error) not in json.dumps(public)


def test_repository_symlink_loop_is_redacted_before_any_review_input_is_opened(
    tmp_path: Path,
    capsys,
) -> None:
    module = _script_module()
    repository_loop = tmp_path / "private-repository-loop"
    repository_loop.symlink_to(repository_loop.name)
    restricted = tmp_path / "restricted"
    restricted.mkdir()

    result = module.main(
        [
            "--repository",
            str(repository_loop),
            "--validate-only",
            "--restricted-root",
            str(restricted),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "error": "Held-out execution is blocked; inspect restricted logs",
        "error_code": "held_out_control_blocked",
        "state": "blocked",
    }
    assert str(repository_loop) not in captured.out
    assert str(repository_loop) not in captured.err
    records = tuple((restricted / "held_out_primary/control_errors").glob("*.json"))
    assert records == ()


def test_public_directory_cannot_be_selected_as_restricted_root(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    module = _script_module()
    repository, canonical_restricted = _test_repository(tmp_path)
    public = repository / "artifacts" / "public"
    public.mkdir()
    review_calls: list[object] = []

    def record_review_call(**kwargs):
        review_calls.append(kwargs)
        raise AssertionError("review gate must not open for an untrusted restricted root")

    monkeypatch.setattr(module, "open_reviewed_held_out_plan", record_review_call)
    result = module.main(
        [
            "--repository",
            str(repository),
            "--validate-only",
            "--restricted-root",
            str(public),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "error": "Held-out execution is blocked; inspect restricted logs",
        "error_code": "held_out_control_blocked",
        "state": "blocked",
    }
    assert captured.err == ""
    assert review_calls == []
    assert tuple(public.rglob("*")) == ()
    assert tuple(canonical_restricted.rglob("*")) == ()


def test_successful_run_publishes_results_gate_before_adapter_close(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    module = _script_module()
    repository, restricted = _test_repository(tmp_path)
    manifest = SimpleNamespace(
        content_hash="a" * 64,
        units=(),
    )
    reviewed_plan = SimpleNamespace(call_manifest=manifest)
    configuration = SimpleNamespace(production_adapter_factory="example:create")
    runtime_binding = SimpleNamespace(development_predecessor_ledger="binding")
    execution = SimpleNamespace(content_hash="b" * 64, itt_records=(1, 2, 3))
    bridge = SimpleNamespace(content_hash="c" * 64)
    gate = SimpleNamespace(content_hash="d" * 64)
    events: list[str] = []

    class Bundle:
        cpu = object()
        sessions = object()
        runtime = object()

        @staticmethod
        def close() -> None:
            events.append("close")

    monkeypatch.setattr(
        module,
        "open_reviewed_held_out_plan",
        lambda **_kwargs: reviewed_plan,
    )
    monkeypatch.setattr(
        module,
        "load_runtime_bound_held_out_configuration",
        lambda **_kwargs: (configuration, runtime_binding),
    )
    monkeypatch.setattr(module, "_adapter_bundle", lambda *_args, **_kwargs: Bundle())

    def execute(**_kwargs):
        events.append("execute")
        return execution

    def close_results(**kwargs):
        assert kwargs["execution"] is execution
        assert kwargs["runtime"] is Bundle.runtime
        assert kwargs["results_root"] == repository / "artifacts/restricted/held_out"
        events.append("results_gate")
        return bridge, gate

    monkeypatch.setattr(module, "execute_reviewed_held_out_manifest", execute)
    monkeypatch.setattr(module, "close_held_out_primary_results", close_results)
    arguments = [
        "--repository",
        str(repository),
        "--review-completion-root",
        "review",
        "--run",
        "--adapter-factory",
        "example:create",
        "--snapshot",
        "snapshot",
        "--shared-cache",
        "cache",
        "--verified-model-manifest",
        "model.json",
        "--selected-model-freeze",
        "freeze.json",
        "--source-association",
        "association.json",
        "--development-prequery-inputs-artifact-hash",
        "e" * 64,
        "--ledger",
        "ledger.sqlite3",
        "--artifact-root",
        "artifacts/restricted/cas",
        "--runtime-root",
        "artifacts/restricted/runtime",
        "--restricted-root",
        str(restricted),
        "--quota-root",
        "artifacts/restricted",
    ]

    assert module.main(arguments) == 0
    assert events == ["execute", "results_gate", "close"]
    public = json.loads(capsys.readouterr().out)
    assert public == {
        "allocated_gpu_seconds": "read_from_global_ledger",
        "execution_manifest_hash": execution.content_hash,
        "itt_call_count": 3,
        "primary_results_gate_hash": gate.content_hash,
        "scorer_bridge_hash": bridge.content_hash,
        "state": "complete",
    }
