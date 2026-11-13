"""Real CPU subprocess checks for the preparation/release boundary."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from story_projection_onto.controller_preparation import (
    PreparedController,
    wait_for_allocation_release,
)


def child_program(path: Path, *, after: str = ""):
    return (
        sys.executable,
        "-c",
        f"""
from pathlib import Path
from story_projection_onto.controller_preparation import wait_for_allocation_release
Path({str(path / "before")!r}).write_text('prepared on CPU')
wait_for_allocation_release(timeout=2)
{after}
Path({str(path / "after")!r}).write_text('released')
""",
    )


def test_real_controller_preparation_precedes_release_without_reexecuting_setup(tmp_path):
    controller = PreparedController(child_program(tmp_path), timeout=2)
    try:
        assert (tmp_path / "before").is_file()
        assert not (tmp_path / "after").exists()
        assert controller.process.poll() is None
        pid = controller.process.pid
        assert 0 < controller.preparation_seconds < 2
        assert controller.release_and_wait() == 0
        assert controller.process.pid == pid
        assert (tmp_path / "after").read_text() == "released"
        with pytest.raises(RuntimeError, match="twice"):
            controller.release_and_wait()
    finally:
        controller.close()


def test_owner_loss_does_not_release_controller(tmp_path):
    controller = PreparedController(child_program(tmp_path), timeout=2)
    controller.close()
    assert controller.process.poll() is not None
    assert not (tmp_path / "after").exists()


def test_live_identity_failure_after_release_remains_fatal(tmp_path):
    controller = PreparedController(
        child_program(tmp_path, after="raise RuntimeError('live identity rejected')"), timeout=2
    )
    try:
        assert controller.release_and_wait() != 0
        assert not (tmp_path / "after").exists()
    finally:
        controller.close()


@pytest.mark.parametrize(
    "program,exception",
    [
        ("raise ValueError('CPU setup failed')", RuntimeError),
        ("import time; time.sleep(5)", TimeoutError),
    ],
)
def test_preparation_failures_are_bounded(program, exception):
    with pytest.raises(exception):
        PreparedController((sys.executable, "-c", program), timeout=0.1)


def test_invalid_release_does_not_run_postrelease_actions(tmp_path):
    controller = PreparedController(child_program(tmp_path), timeout=2)
    try:
        os.write(controller.release_fd, b"wrong\n")
        assert controller.process.wait(timeout=2) != 0
        assert not (tmp_path / "after").exists()
    finally:
        controller.close()


def test_no_pipe_means_no_change_to_resume_path(monkeypatch):
    monkeypatch.delenv("SPO_CPU_PREPARED_FD", raising=False)
    monkeypatch.delenv("SPO_CPU_RELEASE_FD", raising=False)
    assert wait_for_allocation_release() is False


def test_production_gate_precedes_ledger_and_retains_fresh_checks():
    # A source-order regression supplements the real subprocess/pipe tests.
    source = Path("src/story_projection_onto/fallback_acceptance.py").read_text()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    gate = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If) and "wait_for_allocation_release" in ast.unparse(n.test)
    )
    ledger = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.With)
        and "Ledger(options.ledger)" in ast.unparse(n.items[0].context_expr)
    )
    assert gate.end_lineno < ledger.lineno
    body = ast.unparse(gate)
    for live_check in (
        "_validate_orchestrator_guard",
        "validate_source_association",
        "validate_fallback_snapshot_manifest",
        "capture_gpu_hardware_identity",
        "storage.check",
    ):
        assert live_check in body
