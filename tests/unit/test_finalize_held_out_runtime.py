from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _test_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    restricted = repository / "artifacts" / "restricted"
    restricted.mkdir(parents=True)
    return repository, restricted


def _script_module():
    path = ROOT / "scripts/finalize_held_out_runtime.py"
    specification = importlib.util.spec_from_file_location(
        "finalize_held_out_runtime_test_module",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("private-runtime-detail:/restricted/runtime"),
        sqlite3.DatabaseError("private-sqlite-detail:/restricted/ledger.sqlite3"),
        OSError(5, "private-path-detail", "/restricted/private-input.json"),
    ),
)
def test_finalizer_never_prints_exception_details(
    tmp_path: Path,
    capsys,
    monkeypatch,
    error: BaseException,
) -> None:
    module = _script_module()
    repository, restricted = _test_repository(tmp_path)

    def fail_materialization(**_kwargs):
        raise error

    monkeypatch.setattr(
        module,
        "materialize_held_out_runtime_binding",
        fail_materialization,
    )
    result = module.main(
        [
            "--repository",
            str(repository),
            "--fallback-result",
            str(tmp_path / "private-fallback.json"),
            "--source-association",
            str(tmp_path / "private-association.json"),
            "--ledger",
            str(restricted / "private-ledger.sqlite3"),
            "--artifact-root",
            str(restricted / "private-cas"),
            "--restricted-root",
            str(restricted),
            "--created-at",
            "2026-09-04T00:00:00Z",
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "error": "Held-out finalization is blocked; inspect restricted logs",
        "error_code": "held_out_finalizer_blocked",
        "state": "blocked",
    }
    assert str(error) not in captured.out
    assert str(error) not in captured.err
    records = tuple((restricted / "held_out_finalizer/control_errors").glob("*.json"))
    assert len(records) == 1
    assert str(error) in records[0].read_text(encoding="utf-8")
    assert records[0].stat().st_mode & 0o777 == 0o600


def test_finalizer_repository_symlink_loop_is_redacted(
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
            "--fallback-result",
            "private-fallback.json",
            "--source-association",
            "private-association.json",
            "--ledger",
            str(restricted / "private-ledger.sqlite3"),
            "--artifact-root",
            str(restricted / "private-cas"),
            "--restricted-root",
            str(restricted),
            "--created-at",
            "2026-09-04T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "error": "Held-out finalization is blocked; inspect restricted logs",
        "error_code": "held_out_finalizer_blocked",
        "state": "blocked",
    }
    assert str(repository_loop) not in captured.out
    assert str(repository_loop) not in captured.err
    records = tuple((restricted / "held_out_finalizer/control_errors").glob("*.json"))
    assert records == ()


def test_finalizer_rejects_public_directory_as_restricted_root_before_materialization(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    module = _script_module()
    repository, canonical_restricted = _test_repository(tmp_path)
    public = repository / "artifacts" / "public"
    public.mkdir()
    materialization_calls: list[object] = []

    def record_materialization_call(**kwargs):
        materialization_calls.append(kwargs)
        raise AssertionError("materialization must not see an untrusted restricted root")

    monkeypatch.setattr(
        module,
        "materialize_held_out_runtime_binding",
        record_materialization_call,
    )
    result = module.main(
        [
            "--repository",
            str(repository),
            "--fallback-result",
            "private-fallback.json",
            "--source-association",
            "private-association.json",
            "--ledger",
            str(public / "private-ledger.sqlite3"),
            "--artifact-root",
            str(public / "private-cas"),
            "--restricted-root",
            str(public),
            "--created-at",
            "2026-09-04T00:00:00Z",
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert json.loads(captured.out) == {
        "error": "Held-out finalization is blocked; inspect restricted logs",
        "error_code": "held_out_finalizer_blocked",
        "state": "blocked",
    }
    assert captured.err == ""
    assert materialization_calls == []
    assert tuple(public.rglob("*")) == ()
    assert tuple(canonical_restricted.rglob("*")) == ()
