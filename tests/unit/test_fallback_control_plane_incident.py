from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.build_fallback_control_plane_incident import main as build_incident_main
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_control_plane_incident import (
    EXPECTED_ABSENT_BASENAMES,
    EXPECTED_RUN_ROOT_BASENAMES,
    FALLBACK_CONTROL_PLANE_ERROR_CLASS,
    FALLBACK_CONTROL_PLANE_INCIDENT_KIND,
    FALLBACK_V4_RUN_ID,
    FallbackControlPlaneIncident,
    build_fallback_control_plane_incident,
    load_fallback_control_plane_incident,
    validate_fallback_control_plane_incident,
    write_fallback_control_plane_incident,
)
from story_projection_onto.store import Ledger


def _write_self_hashed(path: Path, payload: dict[str, object]) -> dict[str, object]:
    value = {**payload, "manifest_sha256": canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return value


def _fixture(tmp_path: Path) -> dict[str, Path]:
    public = tmp_path / "artifacts" / "public"
    manifests = public / "manifests"
    results = public / "results"
    restricted = tmp_path / "artifacts" / "restricted"
    run_root = restricted / "fallback-development-v4"
    run_root.mkdir(parents=True)
    manifests.mkdir(parents=True)
    results.mkdir(parents=True)

    ledger_live = restricted / "phase1_acceptance.sqlite"
    with Ledger(ledger_live):
        pass
    before = tmp_path / "before" / "phase1_acceptance.sqlite"
    after = tmp_path / "after" / "phase1_acceptance.sqlite"
    before.parent.mkdir()
    after.parent.mkdir()
    shutil.copyfile(ledger_live, before)
    shutil.copyfile(ledger_live, after)
    empty_accounting = {
        "total_allocated_microseconds": 0,
        "event_count": 0,
        "service_session_count": 0,
        "by_kind_microseconds": {},
    }

    association_path = manifests / "source_tree_fallback_second_recovery_v4.association.json"
    association = _write_self_hashed(
        association_path,
        {
            "schema_version": "1.0.0",
            "kind": "local_remote_source_tree_association",
            "branch": "implementation/query-dependent-temporal-ontology",
            "git_commit": "a" * 40,
            "revision_label": "fallback-second-recovery-v4",
            "recorded_at": "2026-09-05T14:22:06Z",
            "local_manifest": "source.local.json",
            "local_manifest_file_sha256": "b" * 64,
            "remote_manifest": "source.remote.json",
            "remote_manifest_file_sha256": "b" * 64,
            "local_tree_sha256": "c" * 64,
            "remote_tree_sha256": "c" * 64,
            "byte_identity_checks": ["independently verified"],
        },
    )
    overlay_path = restricted / "fallback-second-recovery-v4.authorized.json"
    overlay = _write_self_hashed(
        overlay_path,
        {
            "schema_version": "1.2.0",
            "kind": "phase1_fallback_second_recovery_overlay",
            "authorized_recovery_run_id": FALLBACK_V4_RUN_ID,
            "authorization": {"status": "authorized"},
            "source": {
                "current_association_file_sha256": "unused-in-fixture",
                "current_association_manifest_sha256": association["manifest_sha256"],
                "current_tree_sha256": association["local_tree_sha256"],
            },
            "cumulative_gpu_accounting": empty_accounting,
        },
    )
    preflight_path = manifests / "fallback_gpu_acceptance_development_v4.preflight.json"
    _write_self_hashed(
        preflight_path,
        {
            "schema_version": "1.0.0",
            "kind": "phase1_fallback_execution_preflight",
            "run_id": FALLBACK_V4_RUN_ID,
            "source_association_sha256": association["manifest_sha256"],
            "source_tree_sha256": association["local_tree_sha256"],
            "second_recovery_overlay_sha256": overlay["manifest_sha256"],
            "gpu_accounting_before_start": empty_accounting,
            "execution_authorized": True,
            "passed": True,
            "gpu_allocation_performed": False,
            "model_process_started": False,
            "checkpoint_absent": True,
        },
    )
    result_path = results / "fallback_gpu_acceptance_development_v4.json"
    execution_hash = "d" * 64
    invocation = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-invocation.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestration_invocation",
            "run_id": FALLBACK_V4_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "checkpoint": str(run_root / "checkpoint.json"),
            "result_output": str(result_path),
            "gpu_seconds_before_invocation": 0.0,
            "service_session_id": FALLBACK_V4_RUN_ID,
            "service_event_id": f"{FALLBACK_V4_RUN_ID}-service-start-001",
        },
    )
    ticket = _write_self_hashed(
        run_root / "checkpoint.json.guardian-ticket.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_ticket",
            "run_id": FALLBACK_V4_RUN_ID,
            "execution_arguments_sha256": execution_hash,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_command_sha256": "e" * 64,
            "service_session_id": FALLBACK_V4_RUN_ID,
            "service_event_id": f"{FALLBACK_V4_RUN_ID}-service-start-001",
        },
    )
    guard = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-guard-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestrator_guard",
            "run_id": FALLBACK_V4_RUN_ID,
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
            "run_id": FALLBACK_V4_RUN_ID,
            "invocation_present": True,
            "invocation_sha256": invocation["manifest_sha256"],
            "latest_orchestrator_guard_sha256": guard["manifest_sha256"],
            "guardian_ready_receipt_count": 0,
            "internal_controller_receipt_count": 0,
            "live_internal_controller_count": 0,
            "checkpoint_present": False,
            "handoff_present": False,
            "cleanup_present": False,
            "result_present": False,
            "service_start_attempted": None,
            "guardian_terminal": False,
            "latest_orchestrator_live": False,
            "latest_control_group_live": False,
            "unresolved_gpu_allocation_count": 0,
            "unresolved_gpu_service_count": 0,
            "actual_allocated_gpu_seconds": 0.0,
            "resume_allowed": True,
        },
    )
    (run_root / "checkpoint.json.guardian.log").write_text(
        "ValueError: fallback orchestration invocation identity changed on resume\n",
        encoding="utf-8",
    )
    (run_root / "orchestrator.20260905T143000Z.log").write_text(
        "RuntimeError: fallback guardian exited before readiness (1)\n",
        encoding="utf-8",
    )
    return {
        "association": association_path,
        "overlay": overlay_path,
        "preflight": preflight_path,
        "run_root": run_root,
        "result": result_path,
        "before": before,
        "after": after,
    }


def _build(paths: dict[str, Path]) -> FallbackControlPlaneIncident:
    return build_fallback_control_plane_incident(
        run_id=FALLBACK_V4_RUN_ID,
        source_association_path=paths["association"],
        authorization_overlay_path=paths["overlay"],
        preflight_path=paths["preflight"],
        restricted_run_root=paths["run_root"],
        result_path=paths["result"],
        ledger_before_path=paths["before"],
        ledger_after_path=paths["after"],
        audited_at=datetime(2026, 9, 5, 14, 45, tzinfo=UTC),
    )


def test_builder_load_validator_and_append_only_writer_round_trip(tmp_path: Path) -> None:
    incident = _build(_fixture(tmp_path))
    destination = tmp_path / "fallback_gpu_acceptance_development_v4_control_plane_incident.json"

    write_fallback_control_plane_incident(destination, incident)
    write_fallback_control_plane_incident(destination, incident)
    loaded = load_fallback_control_plane_incident(destination)
    validated = validate_fallback_control_plane_incident(
        destination,
        expected_run_id=FALLBACK_V4_RUN_ID,
        expected_source_association_manifest_sha256=(
            incident.source_and_authorization.source_association.manifest_sha256
        ),
        expected_overlay_manifest_sha256=(
            incident.source_and_authorization.authorization_overlay.manifest_sha256
        ),
        expected_preflight_manifest_sha256=(
            incident.source_and_authorization.execution_preflight.manifest_sha256
        ),
        expected_ledger_file_sha256=incident.accounting.before.file_sha256,
        expected_ledger_summary=incident.accounting.before.summary.model_dump(mode="python"),
    )

    assert loaded == validated == incident
    assert incident.kind == FALLBACK_CONTROL_PLANE_INCIDENT_KIND
    assert incident.failure.safe_error_class == FALLBACK_CONTROL_PLANE_ERROR_CLASS
    assert incident.accounting.delta.allocated_gpu_microseconds == 0
    assert incident.accounting.delta.inference_model_calls == 0
    assert incident.absence_inventory.observed_run_root_basenames == (EXPECTED_RUN_ROOT_BASENAMES)
    assert incident.absence_inventory.absent_basenames == EXPECTED_ABSENT_BASENAMES
    public_text = destination.read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text
    assert "launch_command=" not in public_text
    assert "Traceback" not in public_text
    assert "identity changed on resume" not in public_text


def test_builder_rejects_tampered_receipt_unexpected_file_and_changed_ledger(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    status_path = paths["run_root"] / "status.after-failure.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["resume_allowed"] = False
    status_path.write_text(canonical_json(status) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        _build(paths)

    paths = _fixture(tmp_path / "unexpected")
    (paths["run_root"] / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="six-file"):
        _build(paths)

    paths = _fixture(tmp_path / "ledger")
    paths["after"].write_bytes(paths["after"].read_bytes() + b"changed")
    with pytest.raises((ValueError, RuntimeError), match=r"ledger|SQLite"):
        _build(paths)


def test_contract_forbids_extra_fields_and_validator_fails_closed(tmp_path: Path) -> None:
    incident = _build(_fixture(tmp_path))
    value = incident.model_dump(mode="python")
    value["remote_path"] = "/restricted/remote/path"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FallbackControlPlaneIncident.model_validate(value)

    destination = tmp_path / "incident.json"
    write_fallback_control_plane_incident(destination, incident)
    with pytest.raises(ValueError, match="expected v4 identity"):
        validate_fallback_control_plane_incident(
            destination,
            expected_run_id=FALLBACK_V4_RUN_ID,
            expected_source_association_manifest_sha256="0" * 64,
            expected_overlay_manifest_sha256=(
                incident.source_and_authorization.authorization_overlay.manifest_sha256
            ),
            expected_preflight_manifest_sha256=(
                incident.source_and_authorization.execution_preflight.manifest_sha256
            ),
            expected_ledger_file_sha256=incident.accounting.before.file_sha256,
            expected_ledger_summary=incident.accounting.before.summary.model_dump(mode="python"),
        )


def test_contract_schema_is_strict_and_contains_no_payload_text_or_path_fields() -> None:
    schema = FallbackControlPlaneIncident.model_json_schema(mode="validation")
    assert schema["additionalProperties"] is False
    definitions = schema.get("$defs", {})
    assert definitions
    assert all(
        definition.get("additionalProperties") is False
        for definition in definitions.values()
        if definition.get("type") == "object"
    )
    serialized = canonical_json(schema)
    assert '"command"' not in serialized
    assert '"absolute_path"' not in serialized
    assert '"log_text"' not in serialized


def test_cpu_only_cli_builds_the_same_deterministic_incident(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "fallback_gpu_acceptance_development_v4_control_plane_incident.json"
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
        "2026-09-05T14:45:00Z",
        "--output",
        str(output),
    ]

    assert build_incident_main(arguments) == 0
    first = output.read_bytes()
    assert build_incident_main(arguments) == 0
    assert output.read_bytes() == first
    assert load_fallback_control_plane_incident(output) == _build(paths)
