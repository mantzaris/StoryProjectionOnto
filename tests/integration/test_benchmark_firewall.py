from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

from story_projection_onto.benchmark import (
    BenchmarkDriftError,
    GoldFirewallError,
    compile_benchmark,
    load_model_eligible_query,
    load_model_eligible_world,
    materialize_benchmark,
    verify_materialized_benchmark,
)
from story_projection_onto.benchmark_runtime import (
    RuntimeStageKind,
    RuntimeStagingManifest,
    load_staged_world,
    model_request_payload,
    preconstruction_request_payload,
    scan_model_payload,
)
from story_projection_onto.contracts import canonical_sha256


@pytest.mark.integration
def test_materialized_runtime_stage_is_opaque_and_scorer_unreachable(tmp_path: Path) -> None:
    manifest = materialize_benchmark(tmp_path)
    assert manifest.review_complete is False
    assert manifest.held_out_launch_authorized is False
    assert manifest.final_reviewed_seal_hash is None
    verify_materialized_benchmark(tmp_path)
    routing = json.loads(
        (tmp_path / "scorer_only/routing/model_artifacts.json").read_text(encoding="utf-8")
    )
    prequery_stage = tmp_path / routing[0]["relative_prequery_stage_path"]
    model_path = prequery_stage / "evidence.json"
    assert {item.name for item in prequery_stage.iterdir()} == {"evidence.json", "manifest.json"}
    staged_text = model_path.read_text(encoding="utf-8").casefold()
    for forbidden in ("syn-test", "syn-dev", "held_out", "scorer_only"):
        assert forbidden not in staged_text
    loaded = load_model_eligible_world(model_path, tmp_path)
    runtime_manifest = RuntimeStagingManifest.model_validate_json(
        (prequery_stage / "manifest.json").read_text(encoding="utf-8")
    )
    assert runtime_manifest.stage_kind is RuntimeStageKind.PREQUERY_EVIDENCE
    staging_alias = tmp_path / "runtime-object-alias"
    staging_alias.symlink_to(prequery_stage, target_is_directory=True)
    with pytest.raises(GoldFirewallError):
        load_staged_world(staging_alias / "evidence.json", staging_alias, runtime_manifest)
    assert "query" not in preconstruction_request_payload(loaded)
    query_hashes = []
    for query_relative in routing[0]["relative_query_stage_paths"]:
        query_stage = tmp_path / query_relative
        assert {item.name for item in query_stage.iterdir()} == {
            "evidence.json",
            "query.json",
            "manifest.json",
        }
        query_evidence, reveal = load_model_eligible_query(query_stage, tmp_path)
        query_hashes.append(query_evidence.content_hash)
        serialized = json.dumps(
            model_request_payload(query_evidence, reveal), sort_keys=True
        ).casefold()
        for forbidden in (
            "syn-test",
            "syn-dev",
            "held_out",
            "development",
            "scorer_only",
            "expected_effect",
            "contrast_pair",
            '"gold',
        ):
            assert forbidden not in serialized
    assert query_hashes == [loaded.content_hash] * 3
    scorer_path = tmp_path / "scorer_only/held_out/syn-test-01.json"
    with pytest.raises(GoldFirewallError):
        load_model_eligible_world(scorer_path, tmp_path)


@pytest.mark.integration
def test_model_payload_is_invariant_to_scorer_gold_mutation() -> None:
    build = compile_benchmark()
    world_id = "syn-test-01"
    reveal = build.query_reveals_by_world[world_id][0]
    before = model_request_payload(build.model_artifacts[world_id], reveal)
    scorer_dump = build.scorer_artifacts[world_id].model_dump(mode="python")
    annotation = scorer_dump["gold_projections"][0]["assertion_annotations"][0]
    annotation["is_rare"] = not annotation["is_rare"]
    assert canonical_sha256(scorer_dump) != build.scorer_artifacts[world_id].content_hash
    assert canonical_sha256(
        model_request_payload(build.model_artifacts[world_id], reveal)
    ) == canonical_sha256(before)


@pytest.mark.integration
def test_runtime_module_has_no_scorer_import_or_gold_type_reference() -> None:
    path = Path("src/story_projection_onto/benchmark_runtime.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "story_projection_onto.benchmark" not in imports
    assert "story_projection_onto.synthetic_benchmark" not in imports
    assert "GoldContextualProjection" not in source
    assert "GoldAlternativeSet" not in source
    assert "scorer_only" not in {
        value.value
        for value in ast.walk(tree)
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }


@pytest.mark.integration
def test_worker_process_runs_with_only_runtime_contracts_and_one_stage(tmp_path: Path) -> None:
    materialize_benchmark(tmp_path / "corpus")
    routing = json.loads(
        (tmp_path / "corpus/scorer_only/routing/model_artifacts.json").read_text(encoding="utf-8")
    )
    source_stage = tmp_path / "corpus" / routing[0]["relative_query_stage_paths"][0]
    worker_stage = tmp_path / "worker_stage"
    shutil.copytree(source_stage, worker_stage)
    runtime_package = tmp_path / "runtime_only/story_projection_onto"
    runtime_package.mkdir(parents=True)
    (runtime_package / "__init__.py").write_text("", encoding="utf-8")
    for name in ("contracts.py", "benchmark_runtime.py"):
        shutil.copy2(Path("src/story_projection_onto") / name, runtime_package / name)
    site_packages = Path(pydantic.__file__).resolve().parents[1]
    script = """
import importlib.util
from pathlib import Path
from story_projection_onto.benchmark_runtime import RuntimeStagingManifest, load_staged_query
stage = Path(__import__('sys').argv[1])
assert importlib.util.find_spec('story_projection_onto.synthetic_benchmark') is None
manifest = RuntimeStagingManifest.model_validate_json(
    (stage / 'manifest.json').read_text(encoding='utf-8')
)
evidence, reveal = load_staged_query(stage, manifest)
assert reveal.evidence_artifact_hash == evidence.content_hash
"""
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(tmp_path / "runtime_only"), str(site_packages))),
    }
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script, str(worker_stage)],
        cwd=worker_stage,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.integration
def test_value_level_leak_scan_rejects_split_metadata() -> None:
    with pytest.raises(GoldFirewallError):
        scan_model_payload({"innocent_key": "syn-test-01"})
    with pytest.raises(GoldFirewallError):
        scan_model_payload({"innocent_key": "held_out"})
    with pytest.raises(GoldFirewallError):
        scan_model_payload({"expected_effect": "anything"})


@pytest.mark.integration
def test_one_job_stage_rejects_any_extra_query_or_scorer_file(tmp_path: Path) -> None:
    materialize_benchmark(tmp_path)
    routing = json.loads(
        (tmp_path / "scorer_only/routing/model_artifacts.json").read_text(encoding="utf-8")
    )
    query_stage = tmp_path / routing[0]["relative_query_stage_paths"][0]
    (query_stage / "second_query.json").write_text("{}", encoding="utf-8")
    with pytest.raises(GoldFirewallError, match="runtime stage is not exact"):
        load_model_eligible_query(query_stage, tmp_path)


@pytest.mark.integration
def test_review_package_is_exactly_condition_blind_and_has_no_final_seal(tmp_path: Path) -> None:
    materialize_benchmark(tmp_path)
    package = json.loads(
        (tmp_path / "scorer_only/review/blind_review_package.json").read_text(encoding="utf-8")
    )
    assert package["condition_blind"] is True
    assert package["contains_method_outputs"] is False
    assert package["lifecycle_state"] == "draft_for_external_review"
    serialized = json.dumps(package, sort_keys=True).casefold()
    for forbidden in (
        "syn-test",
        "syn-dev",
        "held_out",
        "c0_classical_pre",
        "c1_llm_pre",
        "c2_llm_query",
        "a_fixed_select",
    ):
        assert forbidden not in serialized
    projections = [projection for world in package["worlds"] for projection in world["projections"]]
    assert len(projections) == 9
    assert (
        len(
            {
                item["review_item_id"]
                for projection in projections
                for item in projection["review_items"]
            }
        )
        == 72
    )
    assert not (tmp_path / "scorer_only/held_out/final_reviewed_seal.json").exists()
    assert (tmp_path / "scorer_only/held_out/draft_seal.json").is_file()
    bindings = json.loads(
        (tmp_path / "scorer_only/review/scorer_bindings.json").read_text(encoding="utf-8")
    )
    assert bindings["package_hash"] == package["content_hash"]
    assert len(bindings["entries"]) == 9


@pytest.mark.integration
def test_verifier_rejects_unmanifested_stale_candidate_file(tmp_path: Path) -> None:
    materialize_benchmark(tmp_path)
    stale = tmp_path / "model_visible/held_out/syn-test-01.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkDriftError):
        verify_materialized_benchmark(tmp_path)


@pytest.mark.integration
def test_verifier_binds_the_exact_configuration(tmp_path: Path) -> None:
    output = tmp_path / "output"
    materialize_benchmark(output)
    config = json.loads(Path("configs/study/synthetic_benchmark.json").read_text(encoding="utf-8"))
    config["matcher_revision"] += "-tampered"
    changed_config = tmp_path / "changed-config.json"
    changed_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BenchmarkDriftError, match="supplied configuration"):
        verify_materialized_benchmark(output, changed_config)


@pytest.mark.integration
def test_two_fresh_regenerations_are_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = materialize_benchmark(first)
    second_manifest = materialize_benchmark(second)
    assert first_manifest.content_hash == second_manifest.content_hash
    first_bytes = {
        str(path.relative_to(first)): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_bytes = {
        str(path.relative_to(second)): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_bytes == second_bytes
