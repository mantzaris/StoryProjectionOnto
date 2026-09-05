from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.build_fallback_v7_runtime_incident import main as build_incident_main
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_v7_runtime_incident import (
    EXPECTED_CLASSIFIED_MICROSECONDS,
    EXPECTED_FINAL_TOTAL_MICROSECONDS,
    EXPECTED_OVERHEAD_MICROSECONDS,
    EXPECTED_SERVICE_MICROSECONDS,
    FALLBACK_V7_ROOT_CAUSE,
    FALLBACK_V7_RUN_ID,
    FALLBACK_V7_RUNTIME_INCIDENT_KIND,
    FallbackV7RuntimeIncident,
    build_fallback_v7_runtime_incident,
    load_fallback_v7_runtime_incident,
    validate_fallback_v7_runtime_incident,
    write_fallback_v7_runtime_incident,
)
from story_projection_onto.store import (
    GpuAllocationJournalState,
    GpuEventKind,
    GpuServiceJournalState,
    Ledger,
    StoragePreflight,
)

ROOT = Path(__file__).resolve().parents[2]
SERVICE_EVENT_ID = f"{FALLBACK_V7_RUN_ID}-service-start-001"
CONFIGURATION_HASH = "7" * 64
STARTED_AT = "2026-09-05T17:49:49.471491Z"
FAILED_AT = "2026-09-05T17:53:33.666129Z"
RECOVERED_AT = "2026-09-05T18:01:22.107047Z"


def _write_self_hashed(path: Path, payload: dict[str, object]) -> dict[str, object]:
    value = {**payload, "manifest_sha256": canonical_sha256(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return value


def _build_ledgers(tmp_path: Path) -> dict[str, Path]:
    restricted = tmp_path / "artifacts" / "restricted"
    live = restricted / "phase1_acceptance.sqlite"
    live.parent.mkdir(parents=True)
    with Ledger(live) as ledger:
        ledger.record_gpu_event(
            event_id="prior-allocation",
            event_kind=GpuEventKind.TIMEOUT,
            allocated_seconds="815.215409",
            started_at="2026-09-05T00:00:00Z",
            ended_at="2026-09-05T00:13:35.215409Z",
            succeeded=False,
        )
    before = restricted / "recovery_validation" / "v7" / "phase1_acceptance.before_v7.sqlite"
    before.parent.mkdir(parents=True)
    shutil.copyfile(live, before)

    with Ledger(live) as ledger:
        ledger.record_gpu_allocation_observation(
            allocation_id=SERVICE_EVENT_ID,
            state=GpuAllocationJournalState.OPENED,
            intended_event_kind=GpuEventKind.GPU_SESSION_START,
            elapsed_seconds=0,
            maximum_seconds=300,
            observed_at=STARTED_AT,
        )
        ledger.record_gpu_service_observation(
            service_session_id=SERVICE_EVENT_ID,
            state=GpuServiceJournalState.OPENED,
            session_id=FALLBACK_V7_RUN_ID,
            configuration_hash=CONFIGURATION_HASH,
            service_started_at=STARTED_AT,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session="815.215409",
            hard_limit_seconds=36_000,
            observed_at=STARTED_AT,
        )
        for sequence in range(1, 45):
            elapsed = sequence * 5
            ledger.record_gpu_allocation_observation(
                allocation_id=SERVICE_EVENT_ID,
                state=GpuAllocationJournalState.HEARTBEAT,
                intended_event_kind=GpuEventKind.GPU_SESSION_START,
                elapsed_seconds=elapsed,
                maximum_seconds=300,
                observed_at=STARTED_AT,
            )
            ledger.record_gpu_service_observation(
                service_session_id=SERVICE_EVENT_ID,
                state=GpuServiceJournalState.HEARTBEAT,
                session_id=FALLBACK_V7_RUN_ID,
                configuration_hash=CONFIGURATION_HASH,
                service_started_at=STARTED_AT,
                elapsed_seconds=elapsed,
                ledger_allocated_seconds_before_session="815.215409",
                hard_limit_seconds=36_000,
                observed_at=STARTED_AT,
            )
        ledger.record_gpu_allocation_observation(
            allocation_id=SERVICE_EVENT_ID,
            state=GpuAllocationJournalState.CLOSED,
            intended_event_kind=GpuEventKind.GPU_SESSION_START,
            elapsed_seconds="224.234089",
            maximum_seconds=300,
            observed_at=FAILED_AT,
        )
        ledger.record_gpu_event(
            event_id=SERVICE_EVENT_ID,
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds="224.234089",
            started_at="2026-09-05T17:49:49.432034Z",
            ended_at=FAILED_AT,
            succeeded=False,
            details={
                "admitted_maximum_seconds": 300.0,
                "configuration_hash": CONFIGURATION_HASH,
                "exception_type": "RuntimeConfigurationError",
                "intended_event_kind": "gpu_session_start",
                "session_id": FALLBACK_V7_RUN_ID,
            },
        )
        storage = StoragePreflight(tmp_path)
        for index in range(1, 9):
            report = storage.check(
                current_occupied_bytes=100 + index,
                filesystem_free_bytes=10_000_000_000,
            )
            ledger.record_storage_sample(
                report,
                phase=f"v7-fixture-{index}",
                sampled_at=f"2026-09-05T17:5{index % 4}:00Z",
            )
        for index in range(1, 6):
            ledger.record_resource_sample(
                sample_id=f"{SERVICE_EVENT_ID}-startup-{index:06d}",
                job_id=None,
                gpu_event_id=SERVICE_EVENT_ID,
                process_ram_bytes=index,
                system_available_ram_bytes=100_000,
                gpu_vram_bytes=index,
                project_storage_bytes=100 + index,
                cpu_worker_count=8,
                sampled_at=f"2026-09-05T17:5{index}:00Z",
            )
    before_manual = (
        restricted
        / "recovery_validation"
        / "v7"
        / "phase1_acceptance.before_manual_recovery.sqlite"
    )
    shutil.copyfile(live, before_manual)

    with Ledger(live) as ledger:
        recovered = ledger.recover_gpu_service_journal(
            service_session_id=SERVICE_EVENT_ID,
            recovered_at=RECOVERED_AT,
            details={
                "endpoint_absence_verified": True,
                "gpu_process_absence_verified": True,
                "manual_exact_process_group_stop": True,
                "pid_absence_verified": True,
                "process_cmdline_sha256": "a" * 64,
                "process_group_absence_verified": True,
                "process_group_id": 836_363,
                "process_session_id": 836_363,
                "process_start_ticks": 5_567_562_04,
                "service_instance_token_sha256": "b" * 64,
                "service_pid": 836_363,
                "source_run_id": FALLBACK_V7_RUN_ID,
                "stop_signal": "SIGTERM",
                "stop_verified_at": RECOVERED_AT.replace("Z", "+00:00"),
            },
        )
        assert recovered.service_microseconds == EXPECTED_SERVICE_MICROSECONDS
        assert recovered.classified_event_microseconds == EXPECTED_CLASSIFIED_MICROSECONDS
        assert recovered.overhead_microseconds == EXPECTED_OVERHEAD_MICROSECONDS
        assert (
            ledger.gpu_summary().total_allocated_microseconds
            == EXPECTED_FINAL_TOTAL_MICROSECONDS
        )
    return {"before": before, "before_manual": before_manual, "terminal": live}


def _fixture(tmp_path: Path) -> dict[str, Path]:
    ledgers = _build_ledgers(tmp_path)
    public = tmp_path / "artifacts" / "public"
    manifests = public / "manifests"
    results = public / "results"
    restricted = tmp_path / "artifacts" / "restricted"
    run_root = restricted / "fallback-development-v7"
    run_root.mkdir(parents=True)
    manifests.mkdir(parents=True)
    results.mkdir(parents=True)

    source_path = manifests / "source_tree_fallback_second_recovery_v7.association.json"
    source = _write_self_hashed(
        source_path,
        {
            "schema_version": "1.0.0",
            "kind": "local_remote_source_tree_association",
            "branch": "implementation/query-dependent-temporal-ontology",
            "git_commit": "0" * 40,
            "revision_label": "fallback-second-recovery-v7",
            "local_tree_sha256": "1" * 64,
            "remote_tree_sha256": "1" * 64,
        },
    )
    initial_accounting = {
        "total_allocated_microseconds": 815_215_409,
        "event_count": 1,
        "service_session_count": 0,
        "by_kind_microseconds": {"timeout": 815_215_409},
    }
    overlay_path = restricted / "fallback-second-recovery-v7.authorized.json"
    overlay = _write_self_hashed(
        overlay_path,
        {
            "schema_version": "1.5.0",
            "kind": "phase1_fallback_second_recovery_overlay",
            "authorized_recovery_run_id": FALLBACK_V7_RUN_ID,
            "authorization": {"status": "authorized"},
            "source": {
                "current_association_manifest_sha256": source["manifest_sha256"],
                "current_association_file_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "current_tree_sha256": source["local_tree_sha256"],
            },
            "cumulative_gpu_accounting": initial_accounting,
            "intervening_v6_control_plane_incident": {
                "incident_manifest_sha256": "2" * 64
            },
        },
    )
    preflight_path = manifests / "fallback_gpu_acceptance_development_v7.preflight.json"
    _write_self_hashed(
        preflight_path,
        {
            "schema_version": "1.0.0",
            "kind": "phase1_fallback_execution_preflight",
            "run_id": FALLBACK_V7_RUN_ID,
            "source_association_sha256": source["manifest_sha256"],
            "source_tree_sha256": source["local_tree_sha256"],
            "second_recovery_overlay_sha256": overlay["manifest_sha256"],
            "prior_v6_control_plane_incident_sha256": "2" * 64,
            "gpu_accounting_before_start": initial_accounting,
            "execution_authorized": True,
            "passed": True,
            "gpu_allocation_performed": False,
            "model_process_started": False,
            "checkpoint_absent": True,
        },
    )

    result = results / "fallback_gpu_acceptance_development_v7.json"
    execution_arguments = "3" * 64
    invocation = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-invocation.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestration_invocation",
            "run_id": FALLBACK_V7_RUN_ID,
            "execution_arguments_sha256": execution_arguments,
            "result_output": str(result),
            "gpu_seconds_before_invocation": 815.215409,
            "service_session_id": FALLBACK_V7_RUN_ID,
            "service_event_id": SERVICE_EVENT_ID,
        },
    )
    ticket = _write_self_hashed(
        run_root / "checkpoint.json.guardian-ticket.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_ticket",
            "run_id": FALLBACK_V7_RUN_ID,
            "execution_arguments_sha256": execution_arguments,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_command_sha256": "4" * 64,
            "service_session_id": FALLBACK_V7_RUN_ID,
            "service_event_id": SERVICE_EVENT_ID,
        },
    )
    guard = _write_self_hashed(
        run_root / "checkpoint.json.orchestrator-guard-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestrator_guard",
            "run_id": FALLBACK_V7_RUN_ID,
            "sequence": 1,
            "state": "active",
            "previous_guard_sha256": None,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "execution_arguments_sha256": execution_arguments,
            "orchestrator_command_sha256": "5" * 64,
        },
    )
    _write_self_hashed(
        run_root / "checkpoint.json.guardian-ready-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_service_guardian_ready",
            "run_id": FALLBACK_V7_RUN_ID,
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
        },
    )
    _write_self_hashed(
        run_root / "checkpoint.json.guardian-terminal-request.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_guardian_terminal_request",
            "run_id": FALLBACK_V7_RUN_ID,
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "orchestrator_guard_sha256": guard["manifest_sha256"],
            "controller_result_sha256": None,
        },
    )
    prepare_receipt = _write_self_hashed(
        run_root / "checkpoint.json.internal-controller-000001.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_internal_controller_receipt",
            "run_id": FALLBACK_V7_RUN_ID,
            "sequence": 1,
            "controller_stage": "prepare",
            "previous_controller_receipt_sha256": None,
            "execution_arguments_sha256": execution_arguments,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "orchestrator_guard_sha256": guard["manifest_sha256"],
        },
    )
    cleanup_receipt = _write_self_hashed(
        run_root / "checkpoint.json.internal-controller-000002.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_internal_controller_receipt",
            "run_id": FALLBACK_V7_RUN_ID,
            "sequence": 2,
            "controller_stage": "cleanup",
            "previous_controller_receipt_sha256": prepare_receipt["manifest_sha256"],
            "execution_arguments_sha256": execution_arguments,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "orchestrator_guard_sha256": guard["manifest_sha256"],
        },
    )
    _write_self_hashed(
        run_root / ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
        {
            "schema_version": "1.0.0",
            "kind": "fallback_controller_orchestrator_closed",
            "run_id": FALLBACK_V7_RUN_ID,
            "orchestrator_guard_sha256": guard["manifest_sha256"],
            "prepare_return_code": 2,
            "cleanup_return_code": 2,
            "run_return_code": None,
            "physical_shutdown_verified": False,
        },
    )
    checkpoint = {
        "run_id": FALLBACK_V7_RUN_ID,
        "service_start_attempted": True,
        "controller_handoff_complete": False,
        "failed_call_id": "controller-restart-prepare",
        "active_attempt": None,
        "active_call_id": None,
        "completed_call_ids": [],
        "accepted_outputs": {},
        "orphan_cleanup_completed": False,
        "development_continuation_completed": False,
    }
    (run_root / "checkpoint.json").write_text(
        canonical_json(checkpoint) + "\n", encoding="utf-8"
    )
    (run_root / "checkpoint.json.guardian.lock").touch()
    (run_root / "checkpoint.json.guardian.log").write_text(
        "live vLLM lease lacks an exact recoverable process identity\n"
        "fallback service adoption failed and physical shutdown was not verified\n",
        encoding="utf-8",
    )
    (run_root / "orchestrator.20260905T174607Z.log").write_text(
        "RuntimeError: fallback prepare controller failed\n", encoding="utf-8"
    )
    (run_root / f"{FALLBACK_V7_RUN_ID}.vllm.log").write_text(
        "INFO 09-05 17:53:28 [api_server.py] "
        "Starting vLLM API server 0 on http://127.0.0.1:8000\n"
        "Shutting down FastAPI HTTP server\n",
        encoding="utf-8",
    )

    resource_samples = [
        {
            "sample_id": f"{SERVICE_EVENT_ID}-startup-{index:06d}",
            "violations": [],
        }
        for index in range(1, 6)
    ]

    def public_failure(stage: str, failure_type: str, receipt_hash: object) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "kind": "phase1_fallback_micro_pilot_result",
            "run_id": FALLBACK_V7_RUN_ID,
            "orchestration_controller_stage": stage,
            "failure_type": failure_type,
            "cleanup_failure_type": failure_type,
            "failed_call_id": "controller-restart-prepare",
            "failure_stage": "fallback_micro_pilot",
            "micro_pilot_passed": False,
            "phase1_gate_passed": False,
            "gate_passed": False,
            "normal_acceptance_block_executed": False,
            "completed_base_call_count": 0,
            "completed_call_ids": [],
            "accepted_micro_pilot_result": None,
            "physical_service_state_unverified": True,
            "vllm_service_stopped": False,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "orchestrator_guard_sha256": guard["manifest_sha256"],
            "controller_process_receipt_sha256": receipt_hash,
            "execution_arguments_sha256": execution_arguments,
            "runtime": {"resource_samples": resource_samples if stage == "prepare" else []},
        }

    handoff_path = result.with_name(result.name + ".controller-handoff.json")
    cleanup_path = result.with_name(result.name + ".orphan-cleanup.json")
    _write_self_hashed(
        handoff_path,
        public_failure("prepare", "RuntimeConfigurationError", prepare_receipt["manifest_sha256"]),
    )
    _write_self_hashed(
        cleanup_path,
        public_failure("cleanup", "RuntimeError", cleanup_receipt["manifest_sha256"]),
    )
    return {
        **ledgers,
        "source": source_path,
        "overlay": overlay_path,
        "preflight": preflight_path,
        "handoff": handoff_path,
        "cleanup": cleanup_path,
        "run_root": run_root,
        "result": result,
    }


def _build(paths: dict[str, Path]) -> FallbackV7RuntimeIncident:
    return build_fallback_v7_runtime_incident(
        run_id=FALLBACK_V7_RUN_ID,
        source_association_path=paths["source"],
        authorization_overlay_path=paths["overlay"],
        preflight_path=paths["preflight"],
        controller_handoff_path=paths["handoff"],
        orphan_cleanup_path=paths["cleanup"],
        restricted_run_root=paths["run_root"],
        result_path=paths["result"],
        ledger_before_v7_path=paths["before"],
        ledger_before_manual_recovery_path=paths["before_manual"],
        terminal_ledger_path=paths["terminal"],
        audited_at=datetime(2026, 9, 5, 18, 15, tzinfo=UTC),
    )


def test_builds_exact_terminal_incident_and_validates_self_hash(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    incident = _build(paths)

    assert incident.kind == FALLBACK_V7_RUNTIME_INCIDENT_KIND
    assert incident.failure_sequence.root_cause == FALLBACK_V7_ROOT_CAUSE
    assert incident.failure_sequence.healthy_endpoint_before_failure is True
    assert incident.accounting.delta.inference_model_calls == 0
    assert incident.accounting.service.service_microseconds == EXPECTED_SERVICE_MICROSECONDS
    assert incident.accounting.service.overhead_microseconds == EXPECTED_OVERHEAD_MICROSECONDS
    assert incident.accounting.terminal.summary.total_allocated_microseconds == (
        EXPECTED_FINAL_TOTAL_MICROSECONDS
    )
    assert incident.manual_stop.stop_signal == "SIGTERM"
    assert incident.terminal_state.resume_allowed is False
    assert incident.terminal_state.fresh_repaired_source_required is True
    assert incident.manifest_sha256 == canonical_sha256(
        incident.model_dump(mode="python", exclude={"manifest_sha256"})
    )
    serialized = canonical_json(incident)
    assert "/workspace/" not in serialized
    assert "127.0.0.1" not in serialized
    assert "Traceback" not in serialized


def test_tampering_with_evidence_or_terminal_accounting_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    handoff = json.loads(paths["handoff"].read_text(encoding="utf-8"))
    handoff["runtime"]["resource_samples"].pop()
    immutable = {key: value for key, value in handoff.items() if key != "manifest_sha256"}
    paths["handoff"].write_text(
        canonical_json({**immutable, "manifest_sha256": canonical_sha256(immutable)}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resource-sample count"):
        _build(paths)

    paths = _fixture(tmp_path / "accounting")
    with Ledger(paths["terminal"]) as ledger:
        ledger.record_gpu_event(
            event_id="tampered-extra",
            event_kind=GpuEventKind.FAILURE,
            allocated_seconds="0.000001",
            started_at=RECOVERED_AT,
            ended_at=RECOVERED_AT,
            succeeded=False,
        )
    with pytest.raises((ValueError, ValidationError), match="GPU accounting"):
        _build(paths)

    paths = _fixture(tmp_path / "timing")
    vllm_log = paths["run_root"] / f"{FALLBACK_V7_RUN_ID}.vllm.log"
    vllm_log.write_text(
        "Starting vLLM API server 0 on http://127.0.0.1:8000\n"
        "Shutting down FastAPI HTTP server\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="healthy-endpoint timestamp"):
        _build(paths)


def test_append_only_writer_and_expected_identity_validator(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    incident = _build(paths)
    destination = tmp_path / "incident.json"
    write_fallback_v7_runtime_incident(destination, incident)
    write_fallback_v7_runtime_incident(destination, incident)
    loaded = load_fallback_v7_runtime_incident(destination)
    assert loaded == incident
    validate_fallback_v7_runtime_incident(
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
        expected_terminal_ledger_file_sha256=incident.accounting.terminal.file_sha256,
    )
    destination.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already differs"):
        write_fallback_v7_runtime_incident(destination, incident)


def test_schema_is_strict_and_cli_replays_deterministically(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    schema = FallbackV7RuntimeIncident.model_json_schema(mode="validation")
    assert schema["additionalProperties"] is False
    assert all(
        definition.get("additionalProperties") is False
        for definition in schema.get("$defs", {}).values()
        if definition.get("type") == "object"
    )
    serialized_schema = canonical_json(schema)
    assert '"absolute_path"' not in serialized_schema
    assert '"log_text"' not in serialized_schema
    assert '"command"' not in serialized_schema

    output = tmp_path / "cli-incident.json"
    arguments = [
        "--source-association",
        str(paths["source"]),
        "--authorization-overlay",
        str(paths["overlay"]),
        "--preflight",
        str(paths["preflight"]),
        "--controller-handoff",
        str(paths["handoff"]),
        "--orphan-cleanup",
        str(paths["cleanup"]),
        "--restricted-run-root",
        str(paths["run_root"]),
        "--result",
        str(paths["result"]),
        "--ledger-before-v7",
        str(paths["before"]),
        "--ledger-before-manual-recovery",
        str(paths["before_manual"]),
        "--terminal-ledger",
        str(paths["terminal"]),
        "--audited-at",
        "2026-09-05T18:15:00Z",
        "--output",
        str(output),
    ]
    assert build_incident_main(arguments) == 0
    assert load_fallback_v7_runtime_incident(output) == _build(paths)
