from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.build_fallback_v6_control_plane_incident import main as build_incident_main
from story_projection_onto import fallback_v6_control_plane_incident as incident_module
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_v6_control_plane_incident import (
    DERIVED_V6_RECOVERY_SERVICE_EVENT_IDS,
    EXPECTED_V6_ABSENT_BASENAMES,
    EXPECTED_V6_RUN_ROOT_BASENAMES,
    FALLBACK_V6_CONTROL_PLANE_ERROR_CLASS,
    FALLBACK_V6_CONTROL_PLANE_INCIDENT_KIND,
    FALLBACK_V6_RUN_ID,
    FROZEN_RECOVERY_SERVICE_EVENT_IDS,
    FallbackV6ControlPlaneIncident,
    build_fallback_v6_control_plane_incident,
    load_fallback_v6_control_plane_incident,
    validate_fallback_v6_control_plane_incident,
    write_fallback_v6_control_plane_incident,
)
from story_projection_onto.store import GpuEventKind, Ledger, StoragePreflight

ROOT = Path(__file__).resolve().parents[2]


def _write_self_hashed(path: Path, payload: dict[str, object]) -> dict[str, object]:
    value = {**payload, "manifest_sha256": canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return value


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    public = tmp_path / "artifacts" / "public"
    manifests = public / "manifests"
    results = public / "results"
    restricted = tmp_path / "artifacts" / "restricted"
    run_root = restricted / "fallback-development-v6"
    run_root.mkdir(parents=True)
    manifests.mkdir(parents=True)
    results.mkdir(parents=True)

    live_ledger = restricted / "phase1_acceptance.after-v6.sqlite"
    with Ledger(live_ledger) as ledger:
        ledger.record_gpu_event(
            event_id="registered-prior-allocation",
            event_kind=GpuEventKind.TIMEOUT,
            allocated_seconds="815.215409",
            started_at="2026-09-05T00:00:00Z",
            ended_at="2026-09-05T00:14:00Z",
            succeeded=False,
        )
    before = tmp_path / "before" / "phase1_acceptance.before-v6.sqlite"
    before.parent.mkdir()
    shutil.copyfile(live_ledger, before)
    with Ledger(live_ledger) as ledger:
        storage = StoragePreflight(tmp_path).require(
            current_occupied_bytes=100,
            declared_growth_bytes=10,
            filesystem_free_bytes=6_000_000_000,
        )
        ledger.record_storage_sample(
            storage,
            phase=f"phase1_fallback:{FALLBACK_V6_RUN_ID}",
            sampled_at="2026-09-05T17:02:47.151672Z",
        )
    accounting = {
        "total_allocated_microseconds": 815_215_409,
        "event_count": 1,
        "service_session_count": 0,
        "by_kind_microseconds": {"timeout": 815_215_409},
    }

    association_path = (
        manifests / "source_tree_fallback_second_recovery_v6.association.json"
    )
    association = _write_self_hashed(
        association_path,
        {
            "schema_version": "1.0.0",
            "kind": "local_remote_source_tree_association",
            "branch": "implementation/query-dependent-temporal-ontology",
            "git_commit": "a" * 40,
            "revision_label": "fallback-second-recovery-v6",
            "recorded_at": "2026-09-05T16:43:17Z",
            "local_manifest": "source.local.json",
            "local_manifest_file_sha256": "b" * 64,
            "remote_manifest": "source.remote.json",
            "remote_manifest_file_sha256": "b" * 64,
            "local_tree_sha256": "c" * 64,
            "remote_tree_sha256": "c" * 64,
            "byte_identity_checks": ["independently verified"],
        },
    )
    overlay_path = restricted / "fallback-second-recovery-v6.authorized.json"
    overlay = _write_self_hashed(
        overlay_path,
        {
            "schema_version": "1.4.0",
            "kind": "phase1_fallback_second_recovery_overlay",
            "authorized_recovery_run_id": FALLBACK_V6_RUN_ID,
            "authorization": {"status": "authorized"},
            "source": {
                "current_association_manifest_sha256": association["manifest_sha256"],
                "current_association_file_sha256": hashlib.sha256(
                    association_path.read_bytes()
                ).hexdigest(),
                "current_tree_sha256": association["local_tree_sha256"],
            },
            "cumulative_gpu_accounting": accounting,
            "intervening_control_plane_incident": {
                "incident_manifest_sha256": (
                    incident_module.FALLBACK_V4_INCIDENT_MANIFEST_SHA256
                )
            },
            "intervening_v5_control_plane_incident": {
                "incident_manifest_sha256": (
                    incident_module.FALLBACK_V5_INCIDENT_MANIFEST_SHA256
                )
            },
        },
    )
    preflight_path = manifests / "fallback_gpu_acceptance_development_v6.preflight.json"
    preflight = _write_self_hashed(
        preflight_path,
        {
            "schema_version": "1.0.0",
            "kind": "phase1_fallback_execution_preflight",
            "run_id": FALLBACK_V6_RUN_ID,
            "source_association_sha256": association["manifest_sha256"],
            "source_tree_sha256": association["local_tree_sha256"],
            "second_recovery_overlay_sha256": overlay["manifest_sha256"],
            "prior_control_plane_incident_sha256": (
                incident_module.FALLBACK_V4_INCIDENT_MANIFEST_SHA256
            ),
            "prior_v5_control_plane_incident_sha256": (
                incident_module.FALLBACK_V5_INCIDENT_MANIFEST_SHA256
            ),
            "gpu_accounting_before_start": accounting,
            "execution_authorized": True,
            "passed": True,
            "gpu_allocation_performed": False,
            "model_process_started": False,
            "checkpoint_absent": True,
        },
    )
    monkeypatch.setattr(
        incident_module,
        "FALLBACK_V6_PREFLIGHT_MANIFEST_SHA256",
        preflight["manifest_sha256"],
    )

    result = results / "fallback_gpu_acceptance_development_v6.json"
    execution_hash = "d" * 64
    invocation = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-invocation.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestration_invocation",
            "run_id": FALLBACK_V6_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "checkpoint": str(run_root / "checkpoint.json"),
            "result_output": str(result),
            "created_at": "2026-09-05T17:01:13.659373Z",
            "gpu_seconds_before_invocation": 815.215409,
            "service_session_id": FALLBACK_V6_RUN_ID,
            "service_event_id": f"{FALLBACK_V6_RUN_ID}-service-start-001",
        },
    )
    ticket = _write_self_hashed(
        run_root / "checkpoint.json.guardian-ticket.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_ticket",
            "run_id": FALLBACK_V6_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_command_sha256": "e" * 64,
            "service_session_id": FALLBACK_V6_RUN_ID,
            "service_event_id": f"{FALLBACK_V6_RUN_ID}-service-start-001",
        },
    )
    guard = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-guard-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestrator_guard",
            "run_id": FALLBACK_V6_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "orchestrator_command_sha256": "f" * 64,
            "sequence": 1,
            "previous_guard_sha256": None,
            "state": "active",
        },
    )
    _write_self_hashed(
        run_root / "status.after-failure.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestration_status",
            "run_id": FALLBACK_V6_RUN_ID,
            "invocation_present": True,
            "invocation_sha256": invocation["manifest_sha256"],
            "latest_orchestrator_guard_sha256": guard["manifest_sha256"],
            "orchestrator_invocation_count": 1,
            "guardian_identity_verified": True,
            "guardian_ready_receipt_count": 0,
            "guardian_live": False,
            "guardian_terminal": False,
            "internal_controller_receipt_count": 0,
            "live_internal_controller_count": 0,
            "checkpoint_present": False,
            "handoff_present": False,
            "cleanup_present": False,
            "result_present": False,
            "service_start_attempted": None,
            "latest_orchestrator_live": False,
            "latest_control_group_live": False,
            "ledger_snapshot_verified": True,
            "ledger_snapshot_state": "verified_immutable_read",
            "resume_accounting_state_verified": True,
            "checkpoint_identity_verified": True,
            "unresolved_gpu_allocation_count": 0,
            "unresolved_gpu_service_count": 0,
            "actual_allocated_gpu_seconds": 815.215409,
            "resume_allowed": False,
        },
    )
    (run_root / "checkpoint.json.guardian.log").write_text(
        "DevelopmentContinuationError: second recovery requires the exact ordered "
        "v3+v5 service IDs\n",
        encoding="utf-8",
    )
    (run_root / "orchestrator.20260905T170000Z.log").write_text(
        "RuntimeError: fallback guardian exited before readiness (1)\n",
        encoding="utf-8",
    )
    return {
        "association": association_path,
        "overlay": overlay_path,
        "preflight": preflight_path,
        "run_root": run_root,
        "result": result,
        "before": before,
        "after": live_ledger,
    }


def _build(paths: dict[str, Path]) -> FallbackV6ControlPlaneIncident:
    return build_fallback_v6_control_plane_incident(
        run_id=FALLBACK_V6_RUN_ID,
        source_association_path=paths["association"],
        authorization_overlay_path=paths["overlay"],
        preflight_path=paths["preflight"],
        restricted_run_root=paths["run_root"],
        result_path=paths["result"],
        ledger_before_path=paths["before"],
        ledger_after_path=paths["after"],
        audited_at=datetime(2026, 9, 5, 17, 20, tzinfo=UTC),
    )


def test_builder_validator_and_append_only_writer_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = _build(_fixture(tmp_path, monkeypatch))
    destination = tmp_path / "fallback_v6_incident.json"

    write_fallback_v6_control_plane_incident(destination, incident)
    write_fallback_v6_control_plane_incident(destination, incident)
    loaded = load_fallback_v6_control_plane_incident(destination)
    validated = validate_fallback_v6_control_plane_incident(
        destination,
        expected_source_association_manifest_sha256=(
            incident.source_and_authorization.source_association.manifest_sha256
        ),
        expected_overlay_manifest_sha256=(
            incident.source_and_authorization.authorization_overlay.manifest_sha256
        ),
        expected_preflight_manifest_sha256=(
            incident.source_and_authorization.execution_preflight.manifest_sha256
        ),
        expected_before_ledger_file_sha256=incident.accounting.before.file_sha256,
        expected_after_ledger_file_sha256=incident.accounting.after.file_sha256,
        expected_before_ledger_summary=incident.accounting.before.summary.model_dump(
            mode="python"
        ),
        expected_after_ledger_summary=incident.accounting.after.summary.model_dump(
            mode="python"
        ),
    )
    assert loaded == validated == incident
    assert incident.kind == FALLBACK_V6_CONTROL_PLANE_INCIDENT_KIND
    assert incident.root_cause.safe_error_class == FALLBACK_V6_CONTROL_PLANE_ERROR_CLASS
    assert incident.root_cause.frozen_expected_service_event_ids == (
        FROZEN_RECOVERY_SERVICE_EVENT_IDS
    )
    assert incident.root_cause.factory_derived_service_event_ids == (
        DERIVED_V6_RECOVERY_SERVICE_EVENT_IDS
    )
    assert incident.accounting.delta.allocated_gpu_microseconds == 0
    assert incident.accounting.delta.inference_model_calls == 0
    assert incident.accounting.all_non_storage_tables_identical is True
    assert incident.terminal_state.resume_allowed is False
    assert incident.absence_inventory.observed_run_root_basenames == (
        EXPECTED_V6_RUN_ROOT_BASENAMES
    )
    assert incident.absence_inventory.absent_basenames == EXPECTED_V6_ABSENT_BASENAMES
    public_text = destination.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert "launch_command=" not in public_text
    assert "Traceback" not in public_text
    assert "DevelopmentContinuationError:" not in public_text

    conflict = tmp_path / "conflict.json"
    conflict.write_text("preserve me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="append-only"):
        write_fallback_v6_control_plane_incident(conflict, incident)
    assert conflict.read_text(encoding="utf-8") == "preserve me\n"


def test_builder_rejects_inventory_storage_and_non_storage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path / "inventory", monkeypatch)
    (paths["run_root"] / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="exact preserved artifact set"):
        _build(paths)

    paths = _fixture(tmp_path / "storage", monkeypatch)
    with Ledger(paths["after"]) as ledger:
        storage = StoragePreflight(tmp_path).require(
            current_occupied_bytes=200,
            filesystem_free_bytes=6_000_000_000,
        )
        ledger.record_storage_sample(
            storage,
            phase=f"phase1_fallback:{FALLBACK_V6_RUN_ID}",
            sampled_at="2026-09-05T17:03:00Z",
        )
    with pytest.raises(ValueError, match="beyond one appended sample"):
        _build(paths)

    paths = _fixture(tmp_path / "non-storage", monkeypatch)
    connection = sqlite3.connect(paths["after"])
    try:
        trigger_row = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name = 'gpu_events' "
            "AND sql LIKE '%append-only%' LIMIT 1"
        ).fetchone()
        assert trigger_row is not None
        trigger_name, trigger_sql = trigger_row
        connection.execute(f'DROP TRIGGER "{trigger_name}"')
        connection.execute(
            "UPDATE gpu_events SET started_at = ? WHERE event_id = ?",
            ("2026-09-05T00:00:01Z", "registered-prior-allocation"),
        )
        connection.execute(trigger_sql)
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="non-storage content changed"):
        _build(paths)


def test_contract_forbids_extra_fields_and_validator_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = _build(_fixture(tmp_path, monkeypatch))
    value = incident.model_dump(mode="python")
    value["remote_path"] = "/restricted/remote/path"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FallbackV6ControlPlaneIncident.model_validate(value)

    destination = tmp_path / "incident.json"
    write_fallback_v6_control_plane_incident(destination, incident)
    with pytest.raises(ValueError, match="expected identity"):
        validate_fallback_v6_control_plane_incident(
            destination,
            expected_source_association_manifest_sha256="0" * 64,
            expected_overlay_manifest_sha256=(
                incident.source_and_authorization.authorization_overlay.manifest_sha256
            ),
            expected_preflight_manifest_sha256=(
                incident.source_and_authorization.execution_preflight.manifest_sha256
            ),
            expected_before_ledger_file_sha256=incident.accounting.before.file_sha256,
            expected_after_ledger_file_sha256=incident.accounting.after.file_sha256,
            expected_before_ledger_summary=incident.accounting.before.summary.model_dump(
                mode="python"
            ),
            expected_after_ledger_summary=incident.accounting.after.summary.model_dump(
                mode="python"
            ),
        )


def test_builder_rejects_rehashed_status_and_receipt_chain_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path / "status", monkeypatch)
    status_path = paths["run_root"] / "status.after-failure.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.pop("manifest_sha256")
    status["resume_allowed"] = True
    _write_self_hashed(status_path, status)
    with pytest.raises(ValueError, match="post-failure status changed"):
        _build(paths)

    paths = _fixture(tmp_path / "chain", monkeypatch)
    ticket_path = paths["run_root"] / "checkpoint.json.guardian-ticket.json"
    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    ticket.pop("manifest_sha256")
    ticket["execution_arguments_sha256"] = "0" * 64
    _write_self_hashed(ticket_path, ticket)
    with pytest.raises(ValueError, match="ticket, or orchestrator guard chain changed"):
        _build(paths)


def test_cpu_only_cli_builds_same_deterministic_incident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback_v6_incident.json"
    arguments = [
        "--source-association",
        str(paths["association"]),
        "--authorization-overlay",
        str(paths["overlay"]),
        "--preflight",
        str(paths["preflight"]),
        "--restricted-run-root",
        str(paths["run_root"]),
        "--result",
        str(paths["result"]),
        "--ledger-before",
        str(paths["before"]),
        "--ledger-after",
        str(paths["after"]),
        "--audited-at",
        "2026-09-05T17:20:00Z",
        "--output",
        str(output),
    ]
    assert build_incident_main(arguments) == 0
    first = output.read_bytes()
    assert build_incident_main(arguments) == 0
    assert output.read_bytes() == first
    assert load_fallback_v6_control_plane_incident(output) == _build(paths)


def test_checked_in_v6_incident_binds_exact_zero_gpu_failure() -> None:
    path = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v6_control_plane_incident.json"
    )
    incident = load_fallback_v6_control_plane_incident(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "e8fb3aac180838a4212435ca51905178b7a22ca76990321027f01e182311df56"
    )
    assert incident.manifest_sha256 == (
        "d052713bd262745afc2060e0a75a3b565cf91db93365832b1a8b27dac09a81e9"
    )
    assert incident.accounting.before.file_sha256 == (
        "9040aea2431053fa99667b06ac0e33287c7f0839d131a06bcd462616536b57a6"
    )
    assert incident.accounting.after.file_sha256 == (
        "8698837637cc3d5c86e3abb3593f5647d25a17bfdaaf01fb35a36ff1e0c34a9d"
    )
    assert incident.accounting.appended_storage_sample.sample_id == (
        "f1c86890d0951c709f54f0976bb018d70bc9ecd107033763f99b91bf9bea0f7a"
    )
    assert incident.accounting.delta.allocated_gpu_microseconds == 0
    assert incident.accounting.delta.inference_model_calls == 0
    assert incident.terminal_state.model_process_started is False
    assert incident.terminal_state.guardian_ready is False
    assert incident.terminal_state.resume_allowed is False
    public_text = path.read_text(encoding="utf-8")
    assert "/workspace" not in public_text
    assert "Traceback" not in public_text
    assert "DevelopmentContinuationError:" not in public_text
    assert json.loads(public_text)["manifest_sha256"] == (
        incident.manifest_sha256
    )
