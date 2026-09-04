from __future__ import annotations

import json
from pathlib import Path

from story_projection_onto import __version__
from story_projection_onto.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_cli_version_is_the_package_version(capsys) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_development_plan_is_read_only_and_complete(capsys) -> None:
    assert main(["validate-development-plan", "--project-root", str(ROOT)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["call_count"] == 24
    assert payload["model_load_permitted"] is False
    assert payload["model_service_start_permitted"] is False
    assert payload["model_service_shutdown_permitted"] is False
