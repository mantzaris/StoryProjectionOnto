from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.authorize_fallback_model import main as authorize_fallback_main
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.model_gate import (
    REGISTERED_FALLBACK_REPOSITORY,
    REGISTERED_FALLBACK_REVISION,
    REGISTERED_PRIMARY_REPOSITORY,
    REGISTERED_PRIMARY_REVISION,
    FallbackModelPolicy,
    classify_primary_failure_for_fallback,
    discover_cached_model_repositories,
    fallback_activation_certificate,
    fallback_cache_replacement_receipt,
    migrate_legacy_fallback_artifacts,
    validate_fallback_activation_certificate,
    validate_fallback_cache_replacement_receipt,
    validate_fallback_snapshot_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs/study/fallback_model.json"


def rejected_primary_result() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "phase1_gpu_acceptance_result",
        "gate_passed": False,
        "failure_type": "RuntimeWatchdogTimeout",
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
    assert certificate["primary_failure_classification"] == "latency"
    assert (
        validate_fallback_activation_certificate(
            policy=policy,
            certificate=certificate,
            primary_result=rejected_primary_result(),
        )["manifest_sha256"]
        == certificate["manifest_sha256"]
    )

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


def test_fallback_activation_rejects_controller_bug_but_accepts_generated_invalidity() -> None:
    policy = FallbackModelPolicy.load(POLICY_PATH)
    bug = rejected_primary_result()
    bug["failure_type"] = "TypeError"
    with pytest.raises(ValueError, match="not classified"):
        fallback_activation_certificate(
            policy=policy,
            primary_result=rehash(bug),
            cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
        )

    disguised_bug = rejected_primary_result()
    disguised_bug["failure_type"] = "TypeError"
    disguised_bug["fallback_eligible_failure_class"] = "latency"
    with pytest.raises(ValueError, match="not classified"):
        fallback_activation_certificate(
            policy=policy,
            primary_result=rehash(disguised_bug),
            cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
        )

    generated = rejected_primary_result()
    generated.pop("failure_type")
    generated["calls"] = [
        {
            "status": "failed",
            "response_artifact_hash": "a" * 64,
            "exception_type": "ValueError",
        }
    ]
    assert classify_primary_failure_for_fallback(rehash(generated)) == "structured_output"


def _make_fallback_cache(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    snapshot = (
        cache
        / "hub"
        / "models--Qwen--Qwen3-8B-AWQ"
        / "snapshots"
        / REGISTERED_FALLBACK_REVISION
    )
    snapshot.mkdir(parents=True)
    files = {
        "LICENSE": "Apache License\nVersion 2.0, January 2004\n",
        "config.json": json.dumps({"quantization_config": {"quant_method": "awq"}}),
        "generation_config.json": "{}",
        "model.safetensors.index.json": "{}",
        "model.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
    }
    for name, value in files.items():
        (snapshot / name).write_text(value, encoding="utf-8")
    return cache, snapshot


def test_cache_replacement_receipt_requires_only_exact_fallback(tmp_path: Path) -> None:
    policy = FallbackModelPolicy.load(POLICY_PATH)
    activation = fallback_activation_certificate(
        policy=policy,
        primary_result=rejected_primary_result(),
        cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
    )
    cache, _ = _make_fallback_cache(tmp_path)
    assert discover_cached_model_repositories(cache) == (REGISTERED_FALLBACK_REPOSITORY,)
    receipt = fallback_cache_replacement_receipt(
        policy=policy,
        activation_certificate=activation,
        shared_cache=cache,
    )
    assert receipt["rejected_primary_repository_absent"] is True
    assert (
        validate_fallback_cache_replacement_receipt(
            policy=policy,
            activation_certificate=activation,
            receipt=receipt,
            shared_cache=cache,
        )["manifest_sha256"]
        == receipt["manifest_sha256"]
    )


def test_fallback_snapshot_manifest_rehashes_exact_candidate(tmp_path: Path) -> None:
    policy = FallbackModelPolicy.load(POLICY_PATH)
    cache, snapshot = _make_fallback_cache(tmp_path)
    entries = [
        {
            "path": path.relative_to(snapshot).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    ]
    payload = {
        "schema_version": "1.0.0",
        "repository": policy.repository,
        "revision": policy.revision,
        "license": policy.license,
        "quantization": policy.quantization,
        "file_count": len(entries),
        "total_bytes": sum(row["size_bytes"] for row in entries),
        "files": entries,
        "model_configuration_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        "single_repository_in_shared_cache": True,
        "single_snapshot_in_shared_cache": True,
        "incomplete_file_count": 0,
    }
    manifest = {**payload, "manifest_sha256": canonical_sha256(payload)}
    path = tmp_path / "fallback-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = validate_fallback_snapshot_manifest(
        path,
        policy=policy,
        policy_path=POLICY_PATH,
        snapshot_path=snapshot,
        shared_cache=cache,
    )
    assert validated["repository"] == REGISTERED_FALLBACK_REPOSITORY


def test_checked_in_legacy_artifacts_have_a_strict_nonrewriting_migration(
    tmp_path: Path,
) -> None:
    policy = FallbackModelPolicy.load(POLICY_PATH)
    cache, _ = _make_fallback_cache(tmp_path)
    manifests = ROOT / "artifacts/public/manifests"
    results = ROOT / "artifacts/public/results"
    legacy_activation = json.loads((manifests / "fallback_activation.json").read_text())
    operational_receipt = json.loads(
        (manifests / "model_cache_replacement.json").read_text()
    )
    primary_result = json.loads(
        (results / "phase1_gpu_acceptance_v2_failed.json").read_text()
    )

    upgraded, runtime_receipt = migrate_legacy_fallback_artifacts(
        policy=policy,
        primary_result=primary_result,
        legacy_activation=legacy_activation,
        operational_replacement_receipt=operational_receipt,
        shared_cache=cache,
    )

    assert upgraded["primary_failure_classification"] == "latency"
    assert upgraded["fallback_served_model_name"] == "qwen3-8b-awq-fallback"
    assert upgraded["legacy_activation_manifest_sha256"] == legacy_activation[
        "manifest_sha256"
    ]
    assert upgraded["operational_replacement_receipt_manifest_sha256"] == (
        operational_receipt["manifest_sha256"]
    )
    assert runtime_receipt["activation_certificate_sha256"] == upgraded["manifest_sha256"]
    assert runtime_receipt["legacy_activation_manifest_sha256"] == legacy_activation[
        "manifest_sha256"
    ]
    assert operational_receipt["kind"] == "model_cache_replacement_receipt"


def test_migration_cli_serializes_immutable_certificates(tmp_path: Path) -> None:
    cache, _ = _make_fallback_cache(tmp_path)
    manifests = ROOT / "artifacts/public/manifests"
    results = ROOT / "artifacts/public/results"
    activation_output = tmp_path / "upgraded-activation.json"
    receipt_output = tmp_path / "runtime-receipt.json"

    assert authorize_fallback_main(
        [
            "--policy",
            str(POLICY_PATH),
            "migrate-legacy",
            "--primary-result",
            str(results / "phase1_gpu_acceptance_v2_failed.json"),
            "--legacy-activation",
            str(manifests / "fallback_activation.json"),
            "--operational-replacement-receipt",
            str(manifests / "model_cache_replacement.json"),
            "--shared-cache",
            str(cache),
            "--activation-output",
            str(activation_output),
            "--runtime-receipt-output",
            str(receipt_output),
        ]
    ) == 0
    activation = json.loads(activation_output.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_output.read_text(encoding="utf-8"))
    assert activation["legacy_activation_manifest_sha256"] == (
        "84cd76f0de47a1ceb9154df150c8ead022aa4e544bd0a1c4e31cfee36aab9eb7"
    )
    assert receipt["activation_certificate_sha256"] == activation["manifest_sha256"]
