from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pytest

from story_projection_onto.development_runtime import (
    DEVELOPMENT_CLASS_ADMISSION_P95_SECONDS,
    DEVELOPMENT_CLASS_COUNTS,
    DEVELOPMENT_CLASS_WATCHDOGS,
    DevelopmentCallKind,
    DevelopmentIntegrityError,
    canonical_public_summary,
    load_development_call_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/study/development_call_manifest.json"


def test_registered_development_manifest_is_exact_and_deterministic() -> None:
    manifest = load_development_call_manifest(ROOT)

    assert len(manifest.calls) == 24
    assert Counter(call.call_class for call in manifest.calls) == Counter(DEVELOPMENT_CLASS_COUNTS)
    assert [call.kind for call in manifest.calls[:4]] == [
        DevelopmentCallKind.C1_PRECONSTRUCTION
    ] * 4
    assert [call.kind for call in manifest.calls[4:16]] == [
        DevelopmentCallKind.C2_CONSTRUCTION
    ] * 12
    assert [call.kind for call in manifest.calls[16:20]] == [
        DevelopmentCallKind.FIXED_SELECTION
    ] * 4
    assert [call.kind for call in manifest.calls[20:23]] == [DevelopmentCallKind.ABLATION_PROBE] * 3
    assert manifest.calls[23].kind is DevelopmentCallKind.REPAIR_PROBE
    assert (
        len(
            {
                call.query_stage.staging_manifest_hash
                for call in manifest.calls[4:16]
                if call.query_stage is not None
            }
        )
        == 12
    )
    assert all(call.seed_block == 1 for call in manifest.calls)
    assert all(call.vllm_seed == 1_988_649_846 for call in manifest.calls)
    assert all(
        call.admission_p95_seconds == DEVELOPMENT_CLASS_ADMISSION_P95_SECONDS[call.call_class]
        for call in manifest.calls
    )
    assert all(
        call.watchdog_seconds == DEVELOPMENT_CLASS_WATCHDOGS[call.call_class]
        for call in manifest.calls
    )
    assert [item.unit_id for item in manifest.neutral_evidence_stages] == [
        "dev-unit-01",
        "dev-unit-02",
        "dev-unit-03",
        "dev-unit-04",
    ]
    assert all(
        item.neutral_evidence_artifact_hash != item.model_visible_evidence_artifact_hash
        for item in manifest.neutral_evidence_stages
    )


def test_manifest_loading_does_not_open_query_json(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "query.json":
            raise AssertionError("query semantics were opened during prequery manifest loading")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    assert len(load_development_call_manifest(ROOT).calls) == 24


@pytest.mark.parametrize("marker", ["scorer_only", "held_out", "syn-test-01"])
def test_manifest_source_fails_closed_on_forbidden_namespace(
    tmp_path: Path,
    marker: str,
) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["units"][0]["prequery_stage"]["relative_path"] = (
        f"data/synthetic/model_visible/{marker}/artifact"
    )
    changed = tmp_path / "development-plan.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((DevelopmentIntegrityError, ValueError), match="marker"):
        load_development_call_manifest(ROOT, changed)


def test_manifest_source_rejects_neutral_namespace_or_hash_tamper(tmp_path: Path) -> None:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["units"][0]["neutral_evidence_stage"]["relative_path"] = (
        "data/synthetic/model_visible/prequery_stages/not-neutral"
    )
    changed_path = tmp_path / "changed-path.json"
    changed_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((DevelopmentIntegrityError, ValueError), match="neutral evidence"):
        load_development_call_manifest(ROOT, changed_path)

    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    payload["units"][0]["neutral_evidence_stage"]["manifest_file_sha256"] = "0" * 64
    changed_hash = tmp_path / "changed-hash.json"
    changed_hash.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="neutral stage"):
        load_development_call_manifest(ROOT, changed_hash)


def test_manifest_path_itself_cannot_enter_scorer_namespace() -> None:
    with pytest.raises(DevelopmentIntegrityError, match="marker"):
        load_development_call_manifest(ROOT, ROOT / "scorer_only/development-plan.json")


def test_public_manifest_summary_contains_no_query_or_scorer_payload() -> None:
    summary = canonical_public_summary(load_development_call_manifest(ROOT)).casefold()

    assert '"call_count":24' in summary
    assert "scorer_only" not in summary
    assert "held_out" not in summary
    assert "syn-test" not in summary
    assert '"wording"' not in summary
    assert '"target"' not in summary
    assert '"query_json_read_during_manifest_load":false' in summary


def test_runtime_module_imports_no_scorer_or_gold_implementation() -> None:
    source = (ROOT / "src/story_projection_onto/development_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "story_projection_onto.benchmark" not in modules
    assert "story_projection_onto.synthetic_benchmark" not in modules
    assert "story_projection_onto.metrics" not in modules
    assert "GoldContextualProjection" not in imported_names
    assert "GoldAlternativeSet" not in imported_names


def test_fixed_and_probe_lineage_is_same_unit_and_same_query() -> None:
    manifest = load_development_call_manifest(ROOT)
    by_id = {call.call_id: call for call in manifest.calls}

    for call in manifest.calls:
        if call.source_c1_call_id is not None:
            source = by_id[call.source_c1_call_id]
            assert source.unit_id == call.unit_id
            assert source.seed_block == call.seed_block
        if call.parent_call_id is not None:
            parent = by_id[call.parent_call_id]
            assert parent.unit_id == call.unit_id
            assert parent.query_stage == call.query_stage
            assert parent.seed_block == call.seed_block
        if call.kind is DevelopmentCallKind.ABLATION_PROBE:
            assert call.configuration_delta is not None
            assert call.configuration_delta.changed_field_count == 1
