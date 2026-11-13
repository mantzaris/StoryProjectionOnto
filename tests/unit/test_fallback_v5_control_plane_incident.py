from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.build_fallback_v5_control_plane_incident import main as build_incident_main
from story_projection_onto import fallback_v5_control_plane_incident as incident_module
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_v5_control_plane_incident import (
    EXPECTED_V5_ABSENT_BASENAMES,
    EXPECTED_V5_RUN_ROOT_BASENAMES,
    FALLBACK_V5_ACTIVE_STATUS_ERROR_CLASS,
    FALLBACK_V5_CONTROL_PLANE_ERROR_CLASS,
    FALLBACK_V5_CONTROL_PLANE_INCIDENT_KIND,
    FALLBACK_V5_RUN_ID,
    FallbackV5ControlPlaneIncident,
    build_fallback_v5_control_plane_incident,
    load_fallback_v5_control_plane_incident,
    validate_fallback_v5_control_plane_incident,
    write_fallback_v5_control_plane_incident,
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
    run_root = restricted / "fallback-development-v5"
    run_root.mkdir(parents=True)
    manifests.mkdir(parents=True)
    results.mkdir(parents=True)

    live_ledger = restricted / "phase1_acceptance.sqlite"
    with Ledger(live_ledger) as ledger:
        ledger.record_gpu_event(
            event_id="registered-prior-allocation",
            event_kind=GpuEventKind.TIMEOUT,
            allocated_seconds="815.215409",
            started_at="2026-09-05T00:00:00Z",
            ended_at="2026-09-05T00:14:00Z",
            succeeded=False,
        )
    before = tmp_path / "before" / "phase1_acceptance.before-v5.sqlite"
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
            phase=f"phase1_fallback:{FALLBACK_V5_RUN_ID}",
            sampled_at="2026-09-05T15:57:29.466508Z",
        )
    accounting = {
        "total_allocated_microseconds": 815_215_409,
        "event_count": 1,
        "service_session_count": 0,
        "by_kind_microseconds": {"timeout": 815_215_409},
    }

    association_path = (
        manifests / "source_tree_fallback_second_recovery_v5.association.json"
    )
    association = _write_self_hashed(
        association_path,
        {
            "schema_version": "1.0.0",
            "kind": "local_remote_source_tree_association",
            "branch": "implementation/query-dependent-temporal-ontology",
            "git_commit": "a" * 40,
            "revision_label": "fallback-second-recovery-v5",
            "recorded_at": "2026-09-05T15:47:18Z",
            "local_manifest": "source.local.json",
            "local_manifest_file_sha256": "b" * 64,
            "remote_manifest": "source.remote.json",
            "remote_manifest_file_sha256": "b" * 64,
            "local_tree_sha256": "c" * 64,
            "remote_tree_sha256": "c" * 64,
            "byte_identity_checks": ["independently verified"],
        },
    )
    overlay_path = restricted / "fallback-second-recovery-v5.authorized.json"
    overlay = _write_self_hashed(
        overlay_path,
        {
            "schema_version": "1.3.0",
            "kind": "phase1_fallback_second_recovery_overlay",
            "authorized_recovery_run_id": FALLBACK_V5_RUN_ID,
            "authorization": {"status": "authorized"},
            "source": {
                "current_association_manifest_sha256": association["manifest_sha256"],
                "current_tree_sha256": association["local_tree_sha256"],
            },
            "cumulative_gpu_accounting": accounting,
            "intervening_control_plane_incident": {
                "incident_manifest_sha256": (
                    incident_module.FALLBACK_V4_INCIDENT_MANIFEST_SHA256
                )
            },
        },
    )
    preflight_path = manifests / "fallback_gpu_acceptance_development_v5.preflight.json"
    preflight = _write_self_hashed(
        preflight_path,
        {
            "schema_version": "1.0.0",
            "kind": "phase1_fallback_execution_preflight",
            "run_id": FALLBACK_V5_RUN_ID,
            "source_association_sha256": association["manifest_sha256"],
            "source_tree_sha256": association["local_tree_sha256"],
            "second_recovery_overlay_sha256": overlay["manifest_sha256"],
            "prior_control_plane_incident_sha256": (
                incident_module.FALLBACK_V4_INCIDENT_MANIFEST_SHA256
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
        "FALLBACK_V5_PREFLIGHT_MANIFEST_SHA256",
        preflight["manifest_sha256"],
    )

    result = results / "fallback_gpu_acceptance_development_v5.json"
    execution_hash = "d" * 64
    invocation = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-invocation.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestration_invocation",
            "run_id": FALLBACK_V5_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "checkpoint": str(run_root / "checkpoint.json"),
            "result_output": str(result),
            "created_at": "2026-09-05T15:55:46.738299Z",
            "gpu_seconds_before_invocation": 815.215409,
            "service_session_id": FALLBACK_V5_RUN_ID,
            "service_event_id": f"{FALLBACK_V5_RUN_ID}-service-start-001",
        },
    )
    ticket = _write_self_hashed(
        run_root / "checkpoint.json.guardian-ticket.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_ticket",
            "run_id": FALLBACK_V5_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_command_sha256": "e" * 64,
            "service_session_id": FALLBACK_V5_RUN_ID,
            "service_event_id": f"{FALLBACK_V5_RUN_ID}-service-start-001",
        },
    )
    guard = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-guard-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestrator_guard",
            "run_id": FALLBACK_V5_RUN_ID,
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
        run_root / "checkpoint.json.guardian-ready-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_ready",
            "run_id": FALLBACK_V5_RUN_ID,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "guardian_command_sha256": ticket["guardian_command_sha256"],
            "ready_at": "2026-09-05T15:57:32.289467Z",
        },
    )
    takeover = _write_self_hashed(
        run_root / "checkpoint.json.guardian-controller-takeover.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_guardian_controller_takeover",
            "run_id": FALLBACK_V5_RUN_ID,
            "trigger": "orchestrator_lost",
            "execution_arguments_sha256": execution_hash,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "latest_orchestrator_guard_sha256": guard["manifest_sha256"],
            "controller_receipt_sha256s_before_takeover": [],
            "controller_launch_authority_revoked_at": "2026-09-05T16:02:32.622807Z",
        },
    )
    _write_self_hashed(
        run_root / "checkpoint.json.guardian-result.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_result",
            "run_id": FALLBACK_V5_RUN_ID,
            "trigger": "orchestrator_lost",
            "execution_arguments_sha256": execution_hash,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "controller_takeover_sha256": takeover["manifest_sha256"],
            "physical_shutdown_verified": True,
            "actual_allocated_gpu_seconds": 815.215409,
            "accounted_service_seconds": None,
            "service_session_accounting_sha256": None,
            "terminal_request_sha256": None,
            "checkpoint_service_adopted": False,
            "uptime_recovered": False,
            "control_group_outcomes": [
                {
                    "orchestrator_guard_sha256": guard["manifest_sha256"],
                    "bound_controller_allocation_absent": True,
                    "exact_live_identity_count_before_signal": 0,
                    "process_group_absent": True,
                    "sigterm_sent": False,
                    "sigkill_sent": False,
                }
            ],
        },
    )
    _write_self_hashed(
        run_root / "status.after-terminal.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestration_status",
            "run_id": FALLBACK_V5_RUN_ID,
            "invocation_present": True,
            "invocation_sha256": invocation["manifest_sha256"],
            "latest_orchestrator_guard_sha256": guard["manifest_sha256"],
            "guardian_ready_receipt_count": 1,
            "internal_controller_receipt_count": 0,
            "live_internal_controller_count": 0,
            "checkpoint_present": False,
            "handoff_present": False,
            "cleanup_present": False,
            "result_present": False,
            "service_start_attempted": None,
            "guardian_terminal": True,
            "controller_launch_authority_revoked": True,
            "guardian_physical_shutdown_verified": True,
            "latest_orchestrator_live": False,
            "latest_control_group_live": False,
            "unresolved_gpu_allocation_count": 0,
            "unresolved_gpu_service_count": 0,
            "actual_allocated_gpu_seconds": 815.215409,
            "resume_allowed": False,
        },
    )
    (run_root / "checkpoint.json.guardian.log").write_bytes(b"")
    (run_root / "checkpoint.json.guardian.lock").write_bytes(b"")
    (run_root / "orchestrator.20260905T155544Z.log").write_text(
        "RuntimeError: fallback guardian did not publish readiness before its watchdog\n",
        encoding="utf-8",
    )
    (run_root / "status.active-guardian-attempt-000001.log").write_text(
        "ArtifactIntegrityError: read-only ledger must be closed and checkpointed\n",
        encoding="utf-8",
    )
    (run_root / "status.active-guardian-attempt-000001.exit-code.txt").write_text(
        "1\n",
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


def _build(paths: dict[str, Path]) -> FallbackV5ControlPlaneIncident:
    return build_fallback_v5_control_plane_incident(
        run_id=FALLBACK_V5_RUN_ID,
        source_association_path=paths["association"],
        authorization_overlay_path=paths["overlay"],
        preflight_path=paths["preflight"],
        restricted_run_root=paths["run_root"],
        result_path=paths["result"],
        ledger_before_path=paths["before"],
        ledger_after_path=paths["after"],
        audited_at=datetime(2026, 9, 5, 16, 15, tzinfo=UTC),
    )


def test_builder_validator_and_append_only_writer_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = _build(_fixture(tmp_path, monkeypatch))
    destination = tmp_path / "fallback_gpu_acceptance_development_v5_incident.json"

    write_fallback_v5_control_plane_incident(destination, incident)
    write_fallback_v5_control_plane_incident(destination, incident)
    loaded = load_fallback_v5_control_plane_incident(destination)
    validated = validate_fallback_v5_control_plane_incident(
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
    assert incident.kind == FALLBACK_V5_CONTROL_PLANE_INCIDENT_KIND
    assert incident.failure.safe_error_class == FALLBACK_V5_CONTROL_PLANE_ERROR_CLASS
    assert (
        incident.active_status_diagnostic.safe_error_class
        == FALLBACK_V5_ACTIVE_STATUS_ERROR_CLASS
    )
    assert incident.failure.guardian_ready_latency_microseconds == 105_551_168
    assert incident.accounting.delta.storage_samples == 1
    assert incident.accounting.delta.allocated_gpu_microseconds == 0
    assert incident.accounting.delta.inference_model_calls == 0
    assert incident.terminal_state.resume_allowed is False
    assert incident.absence_inventory.observed_run_root_basenames == (
        EXPECTED_V5_RUN_ROOT_BASENAMES
    )
    assert incident.absence_inventory.absent_basenames == EXPECTED_V5_ABSENT_BASENAMES
    public_text = destination.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert "launch_command=" not in public_text
    assert "Traceback" not in public_text
    assert "closed and checkpointed" not in public_text


def test_builder_rejects_unexpected_file_and_second_storage_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
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
            phase=f"phase1_fallback:{FALLBACK_V5_RUN_ID}",
            sampled_at="2026-09-05T15:58:00Z",
        )
    with pytest.raises(ValueError, match="beyond one appended sample"):
        _build(paths)


def test_contract_forbids_extra_fields_and_validator_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incident = _build(_fixture(tmp_path, monkeypatch))
    value = incident.model_dump(mode="python")
    value["remote_path"] = "/restricted/remote/path"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FallbackV5ControlPlaneIncident.model_validate(value)

    destination = tmp_path / "incident.json"
    write_fallback_v5_control_plane_incident(destination, incident)
    with pytest.raises(ValueError, match="expected identity"):
        validate_fallback_v5_control_plane_incident(
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


def test_cpu_only_cli_builds_same_deterministic_incident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback_gpu_acceptance_development_v5_incident.json"
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
        "2026-09-05T16:15:00Z",
        "--output",
        str(output),
    ]

    assert build_incident_main(arguments) == 0
    first = output.read_bytes()
    assert build_incident_main(arguments) == 0
    assert output.read_bytes() == first
    assert load_fallback_v5_control_plane_incident(output) == _build(paths)


def test_checked_in_v5_incident_binds_exact_terminal_evidence(tmp_path: Path) -> None:
    path = (
        ROOT
        / "artifacts/public/manifests/"
        "fallback_gpu_acceptance_development_v5_control_plane_incident.json"
    )
    incident = load_fallback_v5_control_plane_incident(path)
    # The canonical ledger advances after every later recovery.  The immutable
    # post-v5 bytes survive as the byte-identical pre-v6 snapshot; copy them to
    # the historical basename required by the incident contract.
    post_v5_ledger = tmp_path / "phase1_acceptance.sqlite"
    shutil.copyfile(
        ROOT
        / "artifacts/restricted/recovery_validation/v6_control_plane_incident/"
        "phase1_acceptance.before-v6.sqlite",
        post_v5_ledger,
    )
    rebuilt = build_fallback_v5_control_plane_incident(
        run_id=FALLBACK_V5_RUN_ID,
        source_association_path=(
            ROOT
            / "artifacts/public/manifests/"
            "source_tree_fallback_second_recovery_v5.association.json"
        ),
        authorization_overlay_path=(
            ROOT / "artifacts/restricted/fallback-second-recovery-v5.authorized.json"
        ),
        preflight_path=(
            ROOT
            / "artifacts/public/manifests/"
            "fallback_gpu_acceptance_development_v5.preflight.json"
        ),
        restricted_run_root=ROOT / "artifacts/restricted/fallback-development-v5",
        result_path=(
            ROOT
            / "artifacts/public/results/fallback_gpu_acceptance_development_v5.json"
        ),
        ledger_before_path=(
            ROOT
            / "artifacts/restricted/recovery_validation/v5_control_plane_incident/"
            "phase1_acceptance.before-v5.sqlite"
        ),
        ledger_after_path=post_v5_ledger,
        audited_at=incident.audited_at,
    )

    assert rebuilt == incident
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "47ce61fe2d65eab23e967eb06ff8eb3920d3e21f6191564f8b2a73bea4f4709f"
    )
    assert incident.manifest_sha256 == (
        "06b7bf28427266efa9ebae3640a8a4fe883b0233956d313d98fb9103775dbfdf"
    )
    assert incident.accounting.before.file_sha256 == (
        "38775d1fe3c27cb93afbf78f3c692300a05ed52cb083cb4656d0378b0b94429f"
    )
    assert incident.accounting.after.file_sha256 == (
        "9040aea2431053fa99667b06ac0e33287c7f0839d131a06bcd462616536b57a6"
    )
    assert incident.accounting.before.summary.storage_sample_count == 113
    assert incident.accounting.after.summary.storage_sample_count == 114
    assert incident.accounting.appended_storage_sample.sample_id == (
        "fffd24589a024ce828055420a75e860315856c7a0ee36ea3d4a066e8eefde6f8"
    )
    assert incident.terminal_state.physical_shutdown_verified is True
    assert incident.terminal_state.resume_allowed is False
