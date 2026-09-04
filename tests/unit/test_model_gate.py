from __future__ import annotations

import json
from pathlib import Path

import pytest

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.model_gate import (
    REGISTERED_PRIMARY_REPOSITORY,
    REGISTERED_PRIMARY_REVISION,
    FallbackModelPolicy,
    fallback_activation_certificate,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs/study/fallback_model.json"


def rejected_primary_result() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "phase1_gpu_acceptance_result",
        "gate_passed": False,
        "vllm_service_stopped": True,
        "runtime": {
            "launcher": {
                "model_candidate": "primary",
                "repository": REGISTERED_PRIMARY_REPOSITORY,
                "revision": REGISTERED_PRIMARY_REVISION,
            }
        },
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def rehash(payload: dict[str, object]) -> dict[str, object]:
    immutable = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = canonical_sha256(immutable)
    return payload


def test_fallback_policy_is_one_exact_candidate_and_bounded_micro_pilot() -> None:
    policy = FallbackModelPolicy.load(POLICY_PATH)

    assert policy.repository == "Qwen/Qwen3-8B-AWQ"
    assert policy.revision == "4da05a8edb55c6046cce958586c33b61da07bb79"
    assert policy.maximum_simultaneous_model_snapshots == 1
    assert policy.model_search_allowed is False
    assert policy.service_start_events == 1
    assert policy.maximum_inference_seconds == 720


def test_fallback_policy_rejects_candidate_search_or_call_drift() -> None:
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    raw["model_search_allowed"] = True
    with pytest.raises(ValueError, match="differs at"):
        FallbackModelPolicy.from_mapping(raw)

    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    raw["micro_pilot"]["calls"][1]["form"] = "C1"
    with pytest.raises(ValueError, match="call-form counts"):
        FallbackModelPolicy.from_mapping(raw)


def test_fallback_activation_requires_failed_stopped_registered_primary() -> None:
    policy = FallbackModelPolicy.load(POLICY_PATH)
    certificate = fallback_activation_certificate(
        policy=policy,
        primary_result=rejected_primary_result(),
        cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
    )
    assert certificate["authorized"] is True
    assert certificate["maximum_inference_seconds"] == 720

    passing = rejected_primary_result()
    passing["gate_passed"] = True
    with pytest.raises(ValueError, match="gate rejected"):
        fallback_activation_certificate(
            policy=policy,
            primary_result=rehash(passing),
            cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
        )

    live = rejected_primary_result()
    live["vllm_service_stopped"] = False
    with pytest.raises(ValueError, match="service may be live"):
        fallback_activation_certificate(
            policy=policy,
            primary_result=rehash(live),
            cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
        )

    with pytest.raises(ValueError, match="exactly the rejected primary"):
        fallback_activation_certificate(
            policy=policy,
            primary_result=rejected_primary_result(),
            cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY, policy.repository),
        )
