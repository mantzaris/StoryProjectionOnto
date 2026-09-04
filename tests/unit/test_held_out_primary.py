from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import story_projection_onto.held_out_primary as held_out
from story_projection_onto.contracts import ConditionName
from story_projection_onto.held_out_primary import (
    FixedSelectCapabilityAudit,
    GlobalGpuScheduleSnapshot,
    HeldOutCallEnvelope,
    HeldOutCallManifest,
    HeldOutControlError,
    HeldOutReviewGateError,
    RepairReservePoolSnapshot,
    admit_call,
    load_held_out_control_configuration,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _reserves():
    return (
        RepairReservePoolSnapshot(
            reserve_class="reserve_long", total_slots=4, consumed_slots=0, watchdog_seconds=240
        ),
        RepairReservePoolSnapshot(
            reserve_class="reserve_standard",
            total_slots=8,
            consumed_slots=0,
            watchdog_seconds=150,
        ),
        RepairReservePoolSnapshot(
            reserve_class="reserve_short", total_slots=4, consumed_slots=0, watchdog_seconds=90
        ),
    )


def _plan():
    configuration = load_held_out_control_configuration(ROOT)
    return configuration, held_out._derive_call_manifest(ROOT, configuration)


def test_frozen_held_out_plan_has_exact_registered_inventory_and_no_gold_paths() -> None:
    configuration, manifest = _plan()
    assert manifest.content_hash == configuration.expected_plan_hash
    assert len(manifest.units) == 12
    assert sum(len(item.query_stages) for item in manifest.units) == 36
    assert len(manifest.calls) == 168
    counts = {
        name: sum(item.call_class == name for item in manifest.calls)
        for name in ("test_c1", "test_c2", "test_fixed_select")
    }
    assert counts == {"test_c1": 24, "test_c2": 72, "test_fixed_select": 72}
    serialized = manifest.to_canonical_json().casefold()
    assert "scorer_only" not in serialized
    assert "gold_projection" not in serialized
    assert all("syn-test" not in item.unit_id for item in manifest.units)
    assert sum(item.p95_seconds for item in manifest.calls) == 16344


def test_review_failure_occurs_before_stage_plan_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("held-out stages opened before review")

    monkeypatch.setattr(held_out, "_derive_call_manifest", forbidden)
    with pytest.raises(HeldOutReviewGateError, match="independent review"):
        held_out.open_reviewed_held_out_plan(
            repository=ROOT,
            review_completion_root=tmp_path / "missing-review",
        )
    assert called is False


def test_fixed_select_envelope_requires_complete_c1_or_dependency_failure() -> None:
    _configuration, manifest = _plan()
    fixed_call = next(
        item for item in manifest.calls if item.condition is ConditionName.A_FIXED_SELECT
    )
    unit = next(item for item in manifest.units if item.unit_id == fixed_call.unit_id)
    query = next(
        item
        for item in unit.query_stages
        if item.staging_manifest_hash == fixed_call.query_stage_hash
    )
    with pytest.raises(ValidationError, match="complete C1 source or dependency"):
        HeldOutCallEnvelope(
            call_spec_hash="0" * 64,
            prequery_stage=unit.prequery_stage,
            query_stage=query,
            source_c1_call_id="c1-source",
            require_empty_prequery_inventory=False,
            construction_operations_permitted=False,
        )
    audit = FixedSelectCapabilityAudit(
        complete_c1_graph_hash="2" * 64,
        source_c1_seal_hash="3" * 64,
        constructive_operator_attempt_count=2,
        mechanically_rejected_operator_count=2,
    )
    assert audit.accepted_constructive_operator_count == 0
    with pytest.raises(ValidationError):
        FixedSelectCapabilityAudit(
            complete_c1_graph_hash="2" * 64,
            source_c1_seal_hash="3" * 64,
            constructive_operator_attempt_count=2,
            mechanically_rejected_operator_count=1,
        )


def test_global_schedule_and_watchdog_admission_are_hard_gates() -> None:
    configuration, manifest = _plan()
    call = next(item for item in manifest.calls if item.condition is ConditionName.C2_LLM_QUERY)
    admit_call(
        call=call,
        snapshot=GlobalGpuScheduleSnapshot(
            global_accounting_id="TEST-ONLY-global-accounting",
            gpu_call_inventory_file_sha256=configuration.gpu_call_inventory_file_sha256,
            development_execution_result_hash="PENDING",
            development_predecessor_allocated_gpu_seconds=1000,
            ledger_chain_hash="4" * 64,
            actual_allocated_gpu_seconds=1000,
            remaining_registered_p95_seconds=30000,
            repair_reserves=_reserves(),
            captured_at=NOW,
        ),
        configuration=configuration,
    )
    with pytest.raises(HeldOutControlError, match="9-hour"):
        admit_call(
            call=call,
            snapshot=GlobalGpuScheduleSnapshot(
                global_accounting_id="TEST-ONLY-global-accounting",
                gpu_call_inventory_file_sha256=configuration.gpu_call_inventory_file_sha256,
                development_execution_result_hash="PENDING",
                development_predecessor_allocated_gpu_seconds=1000,
                ledger_chain_hash="4" * 64,
                actual_allocated_gpu_seconds=3000,
                remaining_registered_p95_seconds=30000,
                repair_reserves=_reserves(),
                captured_at=NOW,
            ),
            configuration=configuration,
        )


def test_stage_reference_rejects_tampered_artifact_bytes(tmp_path: Path) -> None:
    source = next((ROOT / "data/synthetic/model_visible/prequery_stages").glob("*/"))
    target = tmp_path / "stage"
    shutil.copytree(source, target)
    evidence_path = target / "evidence.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["artifact_id"] = "TEST-ONLY-tampered-artifact"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(HeldOutControlError, match="invalid runtime stage"):
        held_out._stage_reference(tmp_path, "stage")


def test_call_manifest_derivation_never_opens_held_out_query_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.read_bytes
    opened_query_payload = False

    def guarded(path: Path) -> bytes:
        nonlocal opened_query_payload
        if path.name == "query.json" and "query_stages" in path.parts:
            opened_query_payload = True
            raise AssertionError("query semantics opened while deriving the pre-query plan")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    configuration, manifest = _plan()
    assert manifest.content_hash == configuration.expected_plan_hash
    assert not opened_query_payload
    assert all(
        not query.query_semantics_opened for unit in manifest.units for query in unit.query_stages
    )


def test_call_manifest_rejects_cross_unit_rebinding() -> None:
    _configuration, manifest = _plan()
    payload = manifest.model_dump(mode="python", exclude={"content_hash"})
    calls = list(payload["calls"])
    target_index = next(
        index for index, item in enumerate(calls) if item["call_class"] == "test_c2"
    )
    replacement = dict(calls[target_index])
    replacement.pop("content_hash", None)
    replacement["unit_id"] = manifest.units[-1].unit_id
    calls[target_index] = replacement
    payload["calls"] = calls
    with pytest.raises(ValidationError, match="cover each unit/context/seed"):
        HeldOutCallManifest.model_validate(payload)
