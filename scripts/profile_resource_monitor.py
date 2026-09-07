#!/usr/bin/env python3
"""CPU-only component timings on a stopped pod; never imports a model backend."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.experiment import ResourceLimits
from story_projection_onto.gpu_runtime import ResourceSampler, sample_gpu_vram, sample_process_tree
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.store import StoragePreflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repaired", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.output.exists() or "restricted" not in args.output.parts:
        raise ValueError("new restricted profile required")
    # Read-only authoritative ledger access; no schema initialization or allocation.
    ledger = root / "artifacts/restricted/phase1_acceptance.sqlite"
    digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
    conn = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True, timeout=2)
    rows = []

    def timed(name, fn):
        start = datetime.now(UTC).isoformat()
        tic = time.monotonic()
        value = fn()
        rows.append(
            {
                "component": name,
                "started_at": start,
                "ended_at": datetime.now(UTC).isoformat(),
                "seconds": time.monotonic() - tic,
                "result": value,
            }
        )

    for _ in range(3):
        timed(
            "process_tree_and_ram",
            lambda: {
                "pids": sorted(sample_process_tree(os.getpid()).pids),
                "rss_bytes": sample_process_tree(os.getpid()).rss_bytes,
            },
        )
        timed("gpu_query", lambda: sample_gpu_vram(sample_process_tree(os.getpid()).pids))
        timed("ledger_read", lambda: conn.execute("SELECT count(*) FROM gpu_events").fetchone()[0])
        timed(
            "status_read",
            lambda: len(
                (
                    root
                    / "artifacts/restricted/output-capacity-recovery-v1"
                    / "run-20260907T163234617188/guardian-terminal.json"
                ).read_bytes()
            ),
        )
        lock = threading.RLock()

        def locking(lock=lock):
            for _ in range(1000):
                with lock:
                    pass
            return 1000

        timed("uncontended_rlock_1000", locking)
    with (root / "artifacts/restricted/output-capacity-recovery-v1/controller.lock").open(
        "r"
    ) as file:

        def lock_check():
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(file, fcntl.LOCK_UN)
            return "uncontended_existing_lock"

        timed("controller_flock", lock_check)
    timed("full_storage_traversal", StoragePreflight(root).measure_occupied_bytes)
    if args.repaired:
        sampler = ResourceSampler(
            limits=ResourceLimits.load(root / "configs/study/resource_limits.json"),
            storage=StoragePreflight(root),
        )
        try:
            timed("repaired_full_storage_census", lambda: sampler.prepare(timeout_seconds=150))
            for index in range(3):
                timed(
                    "repaired_fast_observation",
                    lambda index=index: sampler.sample(
                        sample_id=f"cpu-profile-{index}", root_pid=os.getpid()
                    ).public_manifest(),
                )
            timed("repaired_observer_cancel_and_reap", sampler.close_probes)
        finally:
            sampler.close_probes()
    conn.close()
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == digest
    source_paths = (
        "scripts/profile_resource_monitor.py",
        "src/story_projection_onto/resource_probe.py",
        "src/story_projection_onto/gpu_runtime.py",
        "src/story_projection_onto/store.py",
    )
    write_json_atomic(
        {
            "kind": "stopped_pod_cpu_component_profile",
            "gpu_allocation_seconds": 0,
            "ledger_sha256_unchanged": digest,
            "profile_implementation_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in source_paths
            },
            "measurements": rows,
        },
        args.output,
    )
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
