from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG = REPOSITORY_ROOT / "configs" / "study"


def load_config(name: str) -> dict[str, object]:
    return json.loads((STUDY_CONFIG / name).read_text(encoding="utf-8"))


def test_call_inventory_reconciles_with_authoritative_plan() -> None:
    inventory = load_config("gpu_call_inventory.json")
    classes = inventory["classes"]
    assert isinstance(classes, list)
    inference_attempts = sum(
        entry["count"] for entry in classes if entry["name"] != "gpu_session_start"
    )
    accounting_events = sum(entry["count"] for entry in classes)
    planned_seconds = sum(entry["count"] * entry["provisional_p95_seconds"] for entry in classes)

    assert inference_attempts == inventory["maximum_inference_attempts"] == 278
    assert accounting_events == inventory["accounting_events"] == 286
    assert planned_seconds == 31_229
    assert planned_seconds / 3_600 == 8.674722222222222


def test_reserve_tiers_cannot_be_silently_reallocated() -> None:
    inventory = load_config("gpu_call_inventory.json")
    reserve = {
        entry["name"]: (entry["count"], entry["provisional_p95_seconds"])
        for entry in inventory["classes"]
        if entry["name"].startswith("reserve_")
    }
    assert reserve == {
        "reserve_long": (4, 240),
        "reserve_standard": (8, 150),
        "reserve_short": (4, 90),
    }


def test_resource_limits_preserve_reduced_study_envelope() -> None:
    limits = load_config("resource_limits.json")
    assert limits["maximum_cpu_workers"] == 8
    assert limits["maximum_process_ram_bytes"] == 25_000_000_000
    assert limits["maximum_peak_vram_bytes"] == 23 * 1024**3
    assert limits["maximum_project_occupied_bytes"] == 25_000_000_000
    assert limits["minimum_storage_headroom_bytes"] == 5_000_000_000
    assert limits["maximum_project_allocation_bytes"] == 30_000_000_000
    assert limits["scheduled_gpu_seconds"] == 9 * 3_600
    assert limits["hard_gpu_seconds"] == 10 * 3_600
    assert limits["model_cpu_offload_allowed"] is False
    assert limits["generation_concurrency"] == 1


def test_model_and_decoding_are_single_pinned_nonthinking_stack() -> None:
    model = load_config("model.json")
    decoding = load_config("decoding.json")
    assert model["repository"] == "Qwen/Qwen3-14B-AWQ"
    assert model["revision"] == "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"
    assert len(model["revision"]) == 40
    assert model["runtime_version"] == "0.10.2"
    assert model["transformers_version"] == "4.55.2"
    assert model["cpu_offload_gb"] == 0
    assert model["thinking_mode"] is False
    assert model["max_model_len"] == 12_288
    assert decoding["first_pass_maximum_input_tokens"] == 10_240
    assert decoding["first_pass_maximum_output_tokens"] == 2_048
    assert decoding["repair_maximum_input_tokens"] == 10_752
    assert decoding["repair_maximum_output_tokens"] == 1_536


def test_authoritative_plan_hashes_are_frozen() -> None:
    authority = load_config("authority.json")
    entries = authority["authoritative_plans"]
    assert isinstance(entries, list)
    assert len(entries) == 2
    for entry in entries:
        path = REPOSITORY_ROOT / entry["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]

    assert set(authority["historical_provenance_only"]) == {
        "plan_notes/IMPLEMENTATION_PLAN_ADMISSIBILITY_CONSTRAINED_PROJECTION.md",
        "plan_notes/IMPLEMENTATION_PLAN_ADMISSIBILITY_CONSTRAINED_PROJECTION_V2.md",
        "plan_notes/SOLO_RESEARCHER_CONTEXTUAL_NARRATIVE_PROJECTION_PLAN_V2.md",
    }
