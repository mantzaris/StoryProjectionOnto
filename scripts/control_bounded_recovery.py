#!/usr/bin/env python3
"""CPU preflight/status or one persistent, explicitly authorized V10 launch."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto import bounded_recovery as recovery
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.fallback_acceptance import validate_source_association
from story_projection_onto.store import ReadOnlyLedger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "launch", "status"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    options = parser.parse_args()
    root = options.root.resolve(strict=True)
    run_root = root / "artifacts/restricted/fallback-development-v10"
    result = root / "artifacts/public/results/fallback_gpu_acceptance_development_v10.json"
    preflight = (
        root / "artifacts/public/manifests/fallback_gpu_acceptance_development_v10.preflight.json"
    )
    source = root / "artifacts/public/manifests/source_tree_bounded_recovery_v10.association.json"
    ledger = root / "artifacts/restricted/phase1_acceptance.sqlite"
    cache = root / ".cache/shared"
    snapshot = (
        cache / "hub/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79"
    )
    files = {
        "primary-result": root / "artifacts/public/results/phase1_gpu_acceptance_v2_failed.json",
        "activation-certificate": root / "artifacts/public/manifests/fallback_activation_v2.json",
        "cache-replacement-receipt": root
        / "artifacts/public/manifests/model_cache_replacement_runtime.json",
        "snapshot": snapshot,
        "shared-cache": cache,
        "verified-model-manifest": root / "artifacts/public/manifests/model_snapshot_fallback.json",
        "source-association": source,
        "bounded-recovery-authorization": root / "configs/study/bounded_recovery_v10.json",
        "ledger": ledger,
        "artifact-root": root / "artifacts/blobs/phase1_acceptance",
        "checkpoint": run_root / "checkpoint.json",
        "quota-root": root,
    }
    common = [
        "--project-root",
        str(root),
        "--controller-stage",
        "orchestrate",
        "--run-id",
        recovery.RUN_ID,
        "--port",
        "8000",
    ]
    for flag, path in files.items():
        common.extend((f"--{flag}", str(path)))
    command = [sys.executable, str(root / "scripts/run_fallback_gpu_acceptance.py")]
    if options.mode == "preflight":
        if preflight.exists():
            raise FileExistsError("preflight exists; inspect instead of replacing")
        return subprocess.run(
            [*command, "--validate-only", "--output", str(preflight), *common], check=False
        ).returncode
    if options.mode == "status":
        return subprocess.run(
            [*command, "--status", "--output", str(result), *common], check=False
        ).returncode
    proof = json.loads(preflight.read_bytes())
    if proof["manifest_sha256"] != canonical_sha256(
        {k: v for k, v in proof.items() if k != "manifest_sha256"}
    ):
        raise ValueError("preflight checksum changed")
    association = validate_source_association(source, source_root=root)
    if (
        proof.get("passed") is not True
        or proof.get("execution_authorized") is not True
        or proof["source_association_sha256"] != association["manifest_sha256"]
    ):
        raise ValueError("source-bound preflight does not authorize this launch")
    recovery.load_authorization(
        files["bounded-recovery-authorization"], root=root, run_id=recovery.RUN_ID
    )
    with ReadOnlyLedger(ledger) as opened:
        recovery.validate_history(root, opened, before_start=True)
        recovery.prelaunch_forecast(root, opened.gpu_summary().total_allocated_seconds)
    if run_root.exists() or result.exists():
        raise FileExistsError("V10 was invoked; no second start is authorized")
    run_root.mkdir(mode=0o700)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log = run_root / f"orchestrator.{stamp}.log"
    launch = [
        "env",
        f"PYTHONPATH={root / 'src'}",
        *command,
        "--execute",
        "--output",
        str(result),
        *common,
    ]
    with (run_root / "launch.json").open("x") as stream:
        json.dump(
            {
                "command": launch,
                "started_at": stamp,
                "git_commit": association["git_commit"],
                "source_tree_sha256": association["local_tree_sha256"],
                "preflight_sha256": proof["manifest_sha256"],
            },
            stream,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    shell = "exec " + shlex.join(launch) + " >>" + shlex.quote(str(log)) + " 2>&1"
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "storyprojection-study-v10", shell], check=True
    )
    print(json.dumps({"session": "storyprojection-study-v10", "log": str(log)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
