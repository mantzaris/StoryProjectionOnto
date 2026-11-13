#!/usr/bin/env python3
"""Measure only local invariant CPU work and itemize a proposed, unapproved budget.

No runtime factory, GPU service, model, inference client, or live ledger is opened.
The HTTP integration tests separately use authored fixture responses, not inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.controller_preparation import PreparedController
from story_projection_onto.manifest import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    options = parser.parse_args()
    if options.output.exists() or not 1 <= options.samples <= 10:
        raise ValueError("use a new output and 1..10 CPU samples")
    root = Path(__file__).resolve().parents[1]
    observations = []
    # Explicitly hide GPU devices from the CPU benchmark's subprocesses.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "8"
    with tempfile.TemporaryDirectory(prefix="spo-cpu-preparation-") as temporary:
        for sample in range(options.samples):
            target = Path(temporary) / f"sample-{sample}.json"
            program = f"""
import json, time
from pathlib import Path
started = time.perf_counter()
import story_projection_onto.fallback_acceptance as fallback
import_done = time.perf_counter()
root = Path({str(root)!r})
fallback.FallbackModelPolicy.load(root / 'configs/study/fallback_model.json')
fallback.ResourceLimits.load(root / 'configs/study/resource_limits.json')
fallback.StorageAllocationPlan.load(root / 'configs/study/storage_phase_allocations.json')
policy_done = time.perf_counter()
fallback.Phase1LegacyEvidenceProvenanceBridge.load(root)
provenance_done = time.perf_counter()
with Path({str(target)!r}).open('x') as stream:
    json.dump({{'module_import_seconds': import_done-started,
               'policy_parsing_seconds': policy_done-import_done,
               'provenance_loading_seconds': provenance_done-policy_done}}, stream)
fallback.wait_for_allocation_release()
"""
            child = PreparedController((sys.executable, "-c", program), timeout=30)
            try:
                sample_data = json.loads(target.read_bytes())
                sample_data["process_to_prepared_seconds"] = child.preparation_seconds
                assert child.release_and_wait() == 0
            finally:
                child.close()
            observations.append(sample_data)
    result_path = root / "artifacts/public/results/fallback_gpu_acceptance_development_v9.json"
    result = json.loads(result_path.read_bytes())
    assert (
        canonical_sha256({k: v for k, v in result.items() if k != "manifest_sha256"})
        == result["manifest_sha256"]
    )
    forecast = result["actual_plus_remaining_forecast"]
    consumed = forecast["actual_allocated_seconds"]
    remaining = forecast["remaining_forecast_seconds"]
    assert consumed == 2936.238858
    assert (
        sum(
            row["remaining_forecast_seconds"]
            for row in result["post_fallback_full_manifest_forecast"]["rows"]
        )
        == remaining
    )
    assert forecast["consumed_reserve_slots"]["reserve_long"] == 2
    proposed_attempt = {
        "startup_timeout_seconds": 300,
        "live_control_checks_cap_seconds": 120,
        "single_c1_retry_watchdog_seconds": 240,
        "validation_and_resource_drain_cap_seconds": 120,
        "shutdown_cap_seconds": 60,
    }
    attempt_bound = sum(proposed_attempt.values())
    # One of the two still-unconsumed reserve-long slots already covers this
    # proposed call. Earmarking it adds no inference slot or duplicate forecast.
    all_in = consumed + remaining - 240 + attempt_bound
    source_paths = [
        "scripts/measure_cpu_preparation.py",
        "src/story_projection_onto/controller_preparation.py",
        "src/story_projection_onto/fallback_acceptance.py",
        "src/story_projection_onto/gpu_runtime.py",
        "src/story_projection_onto/http_diagnostics.py",
        "src/story_projection_onto/phase1_acceptance.py",
    ]
    payload = {
        "schema_version": "1.0.0",
        "kind": "cpu_preparation_measurement_and_budget_proposal",
        "measurement_scope": "local CPU only; not a RunPod or GPU timing observation",
        "python_version": sys.version.split()[0],
        "samples": observations,
        "local_medians_seconds": {
            key: statistics.median(row[key] for row in observations) for key in observations[0]
        },
        "pod_preparation_savings_demonstrated_seconds": 0,
        "gpu_throughput_improvement_claimed": False,
        "unmeasured_prepared_components": [
            "pinned torch import",
            "tokenizer loading/template capture",
            "RunPod filesystem and scheduling costs",
        ],
        "checks_retained_after_release": [
            "orchestrator authority",
            "source hash",
            "snapshot hash",
            "GPU identity",
            "storage",
            "ledger",
            "service identity",
            "ordinary admission",
            "live resource watchdog",
        ],
        "v9_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "actual_allocated_seconds": consumed,
        "remaining_registered_forecast_seconds": remaining,
        "proposed_attempt_limits_seconds": proposed_attempt,
        "proposed_attempt_total_seconds": attempt_bound,
        "already_forecast_reserve_long_earmarked_seconds": 240,
        "remaining_forecast_excluding_earmarked_retry_seconds": remaining - 240,
        "all_in_with_proposed_attempt_seconds": all_in,
        "ordinary_ceiling_seconds": 32400,
        "ordinary_admitted": all_in <= 32400,
        "ordinary_deficit_seconds": all_in - 32400,
        "proposed_scheduled_ceiling_seconds": 33660,
        "proposed_scheduled_ceiling_hours": 9.35,
        "proposed_scheduled_reserve_seconds": 33660 - all_in,
        "unchanged_hard_ceiling_exclusive_seconds": 36000,
        "hard_margin_after_forecast_seconds": 36000 - all_in,
        "proposal_approved": False,
        "new_start_authorized": False,
        "attempt_caps_enforced_in_existing_runner": False,
        "qualification": (
            "Before any future launch, approve the amendment and enforce the proposed "
            "whole-attempt/live-control caps. No savings are deducted; no mandatory "
            "comparison or prior second is removed."
        ),
        "source_file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_paths
        },
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    write_json_atomic(payload, options.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
