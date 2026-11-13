from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from story_projection_onto.contracts import ConditionName
from story_projection_onto.ui import LocalUiRepository, build_visualization_bundle, create_app
from tests.unit.test_ui import context, packet, projection


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_health(url: str) -> None:
    deadline = time.monotonic() + 10.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as error:
            last_error = error
        time.sleep(0.05)
    raise RuntimeError(f"local UI server did not become healthy: {last_error}")


@pytest.mark.integration
def test_real_system_browser_exercises_local_ui_controls() -> None:
    """Use the installed browser, not a DOM shim, for the bounded interface smoke."""

    uvicorn = pytest.importorskip("uvicorn")
    node = shutil.which("node")
    chrome = next(
        (
            executable
            for name in ("google-chrome", "chromium", "chromium-browser")
            if (executable := shutil.which(name)) is not None
        ),
        None,
    )
    if node is None or chrome is None:
        pytest.skip("the system-browser smoke requires Node.js and Chrome/Chromium")

    query_context = context()
    evidence_packet = packet()
    ontology_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
    )
    bundle = build_visualization_bundle(
        ontology_projection,
        query_context,
        evidence_packet,
        include_public_evidence_text=True,
    )
    ui_directory = Path(__file__).resolve().parents[2] / "ui"
    app = create_app(
        LocalUiRepository((bundle,)),
        revision_seed=0,
        static_directory=ui_directory,
    )
    port = _available_port()
    configuration = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(configuration)
    thread = threading.Thread(target=server.run, name="bounded-ui-smoke", daemon=True)
    thread.start()
    application_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(application_url)
        driver = Path(__file__).with_name("ui_browser_smoke.mjs")
        completed = subprocess.run(
            [node, str(driver), chrome, application_url],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert completed.returncode == 0, (
            f"browser smoke failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
        assert completed.stdout.strip() == "system browser smoke passed"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "the bounded local UI server did not stop"
