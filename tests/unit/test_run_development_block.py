from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_development_block.py"


def test_development_manifest_cli_remains_query_blind() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--validate-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["call_count"] == 24
    assert payload["query_json_read_during_manifest_load"] is False
    assert payload["model_service_start_permitted"] is False


def test_cpu_packing_mode_fails_before_tokenizer_import_when_inputs_are_missing() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--packing-preflight"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--snapshot" in completed.stderr
    assert "--launcher-configuration-hash" in completed.stderr
    assert "Traceback" not in completed.stderr
