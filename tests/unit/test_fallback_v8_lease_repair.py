from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.restore_fallback_v8_terminal_lease import main as repair_cli_main
from story_projection_onto import fallback_v8_lease_repair as repair_module
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_v8_lease_repair import (
    FALLBACK_V8_LEASE_REPAIR_KIND,
    restore_fallback_v8_terminal_lease,
)
from story_projection_onto.fallback_v8_runtime_incident import (
    EXPECTED_BASELINE_MICROSECONDS,
    EXPECTED_CLASSIFIED_MICROSECONDS,
    EXPECTED_OVERHEAD_MICROSECONDS,
    EXPECTED_SERVICE_MICROSECONDS,
    EXPECTED_TERMINAL_MICROSECONDS,
    FALLBACK_V8_RUN_ID,
    FALLBACK_V8_SERVICE_EVENT_ID,
)
from story_projection_onto.gpu_runtime import (
    DURABLE_EXEC_GATE_PROTOCOL,
    FALLBACK_MODEL_REVISION,
    SERVICE_LEASE_SNAPSHOT_FILENAME,
    SERVICE_LOCK_FILENAME,
    VLLMLaunchConfiguration,
)
from story_projection_onto.store import GpuServiceSession

ROOT = Path(__file__).resolve().parents[2]
STARTED_AT = datetime(2026, 9, 5, 20, 22, 15, 908094, tzinfo=UTC)
ENDED_AT = datetime(2026, 9, 5, 20, 40, 9, 324832, tzinfo=UTC)
REPAIRED_AT = datetime(2026, 9, 5, 21, 30, tzinfo=UTC)
PROCESS_GROUP = 885_933
START_TICKS = 557_670_858
SNAPSHOT_MANIFEST_SHA256 = "b" * 64
TOKEN_SHA256 = "c" * 64
LAUNCH_TOKEN_SHA256 = "d" * 64
SUPERVISOR_SHA256 = "e" * 64


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _self_hashed(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "manifest_sha256": canonical_sha256(value)}


def _lease_payload(
    configuration: VLLMLaunchConfiguration,
    *,
    state: str,
) -> dict[str, Any]:
    stopped = state == "stopped_verified"
    payload = {
        "schema_version": "1.0.0",
        "configuration_hash": configuration.configuration_hash,
        "controller_pid": os.getpid(),
        "lease_state": state,
        "session_id": FALLBACK_V8_RUN_ID,
        "accounting_session_id": FALLBACK_V8_SERVICE_EVENT_ID,
        "service_pid": PROCESS_GROUP,
        "process_start_ticks": START_TICKS,
        "process_command_sha256": canonical_sha256(list(configuration.command())),
        "process_group_id": PROCESS_GROUP,
        "process_session_id": PROCESS_GROUP,
        "service_instance_token_sha256": TOKEN_SHA256,
        "launch_protocol": DURABLE_EXEC_GATE_PROTOCOL,
        "launch_gate_token_sha256": LAUNCH_TOKEN_SHA256,
        "launch_supervisor_command_sha256": SUPERVISOR_SHA256,
        "service_started_at": STARTED_AT.isoformat(),
        "service_ended_at": (
            ENDED_AT.isoformat()
            if stopped
            else datetime(2026, 9, 5, 20, 27, tzinfo=UTC).isoformat()
        ),
        "ledger_allocated_seconds_before_session": (
            EXPECTED_BASELINE_MICROSECONDS / 1_000_000
        ),
        "observed_service_seconds": (
            EXPECTED_SERVICE_MICROSECONDS / 1_000_000 if stopped else 285.0
        ),
        "updated_at": (
            REPAIRED_AT.isoformat()
            if stopped
            else datetime(2026, 9, 5, 20, 27, tzinfo=UTC).isoformat()
        ),
    }
    return {**payload, "lease_manifest_sha256": canonical_sha256(payload)}


def _service_record() -> GpuServiceSession:
    return GpuServiceSession(
        service_session_id=FALLBACK_V8_SERVICE_EVENT_ID,
        session_id=FALLBACK_V8_RUN_ID,
        service_microseconds=EXPECTED_SERVICE_MICROSECONDS,
        classified_event_microseconds=EXPECTED_CLASSIFIED_MICROSECONDS,
        overhead_microseconds=EXPECTED_OVERHEAD_MICROSECONDS,
        started_at=STARTED_AT.isoformat().replace("+00:00", "Z"),
        ended_at=ENDED_AT.isoformat().replace("+00:00", "Z"),
        details_json="{}",
    )


class _FakeLedger:
    record = _service_record()

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> _FakeLedger:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_gpu_service_session(self, service_session_id: str) -> GpuServiceSession | None:
        assert service_session_id == FALLBACK_V8_SERVICE_EVENT_ID
        return self.record


class _FakeMeter:
    def __init__(self, ledger: _FakeLedger) -> None:
        self.ledger = ledger


@pytest.fixture
def repair_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    project = tmp_path / "project"
    configuration_root = project / "configs/study"
    configuration_root.mkdir(parents=True)
    shutil.copy2(ROOT / "configs/study/fallback_model.json", configuration_root)
    shutil.copy2(ROOT / "configs/study/model.json", configuration_root)

    cache = project / ".cache/shared"
    snapshot = (
        cache
        / "hub/models--Qwen--Qwen3-8B-AWQ/snapshots"
        / FALLBACK_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    public_manifests = project / "artifacts/public/manifests"
    public_manifests.mkdir(parents=True)
    incident_path = public_manifests / "v8_incident.json"
    incident_path.write_text("{}\n", encoding="utf-8")
    snapshot_manifest = public_manifests / "verified_snapshot.json"
    snapshot_manifest.write_text("{}\n", encoding="utf-8")
    restricted = project / "artifacts/restricted"
    restricted.mkdir(parents=True)
    ledger = restricted / "phase1_acceptance.sqlite"
    ledger.write_bytes(b"immutable terminal V8 ledger bytes")
    output = restricted / "v8_lease_repair.json"

    configuration = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot,
        shared_cache=cache,
        model_configuration_path=configuration_root / "model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256,
        port=8123,
    )
    lease_path = cache / SERVICE_LEASE_SNAPSHOT_FILENAME
    _write_private_json(lease_path, _lease_payload(configuration, state="shutdown_unverified"))
    lock_path = cache / SERVICE_LOCK_FILENAME
    lock_path.write_text("{}\n", encoding="utf-8")
    lock_path.chmod(0o600)
    damaged_lease = json.loads(lease_path.read_text(encoding="utf-8"))

    intent_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "fallback_v8_exact_orphan_stop_intent",
        "run_id": FALLBACK_V8_RUN_ID,
        "service_event_id": FALLBACK_V8_SERVICE_EVENT_ID,
        "recorded_at": datetime(2026, 9, 5, 20, 40, 8, tzinfo=UTC),
        "lease_file_sha256": _sha256_file(lease_path),
        "lease_manifest_sha256": damaged_lease["lease_manifest_sha256"],
        "process_group_id": PROCESS_GROUP,
        "member_identities": [
            {
                "pid": PROCESS_GROUP,
                "ppid": 1,
                "process_group": PROCESS_GROUP,
                "session_id": PROCESS_GROUP,
                "start_ticks": START_TICKS,
                "state": "S",
                "comm": "python",
                "raw_command_sha256": "1" * 64,
                "token_matches": True,
            },
            {
                "pid": PROCESS_GROUP + 1,
                "ppid": PROCESS_GROUP,
                "process_group": PROCESS_GROUP,
                "session_id": PROCESS_GROUP,
                "start_ticks": START_TICKS + 1,
                "state": "S",
                "comm": "python",
                "raw_command_sha256": "2" * 64,
                "token_matches": True,
            },
            {
                "pid": PROCESS_GROUP + 2,
                "ppid": PROCESS_GROUP,
                "process_group": PROCESS_GROUP,
                "session_id": PROCESS_GROUP,
                "start_ticks": START_TICKS + 2,
                "state": "S",
                "comm": "VLLM::EngineCor",
                "raw_command_sha256": "3" * 64,
                "token_matches": False,
            },
        ],
        "sole_gpu_process": {
            "pid": PROCESS_GROUP + 2,
            "used_gpu_memory_mib": 21482,
        },
        "sole_port_8000_owner_verified": True,
        "signal_plan": ["SIGTERM", "SIGKILL only after 30 seconds if exact members remain"],
        "reason": "terminal V8 controllers exited after fail-closed token loss",
        "utility_file_sha256": "4" * 64,
    }
    stop_intent = restricted / "manual_stop_intent.json"
    _write_private_json(stop_intent, _self_hashed(intent_payload))
    stop_intent_value = json.loads(stop_intent.read_text(encoding="utf-8"))
    stop_binding = SimpleNamespace(
        basename=stop_intent.name,
        size_bytes=stop_intent.stat().st_size,
        file_sha256=_sha256_file(stop_intent),
        kind=stop_intent_value["kind"],
        manifest_sha256=stop_intent_value["manifest_sha256"],
    )
    process_members = tuple(
        SimpleNamespace(
            role=role,
            pid=member["pid"],
            parent_pid=member["ppid"],
            process_group_id=member["process_group"],
            session_id=member["session_id"],
            start_ticks=member["start_ticks"],
            process_name=member["comm"],
            process_state=member["state"],
            argv_sha256=member["raw_command_sha256"],
        )
        for role, member in zip(
            ("service_leader", "service_worker", "engine_core"),
            intent_payload["member_identities"],
            strict=True,
        )
    )
    incident = SimpleNamespace(
        run_id=FALLBACK_V8_RUN_ID,
        manifest_sha256="a" * 64,
        manual_recovery=SimpleNamespace(
            stop_intent=stop_binding,
            pre_stop_lease_file_sha256=stop_intent_value["lease_file_sha256"],
            pre_stop_lease_manifest_sha256=stop_intent_value[
                "lease_manifest_sha256"
            ],
            process_group_id=PROCESS_GROUP,
            process_members=process_members,
            engine_core_token_matched=False,
            sole_gpu_process_memory_mib=21_482,
            sole_port_owner_verified=True,
        ),
        accounting=SimpleNamespace(
            terminal=SimpleNamespace(
                file_sha256=_sha256_file(ledger),
                size_bytes=ledger.stat().st_size,
                summary=SimpleNamespace(
                    total_allocated_microseconds=EXPECTED_TERMINAL_MICROSECONDS,
                    unresolved_gpu_allocation_count=0,
                    unresolved_gpu_service_count=0,
                ),
            ),
            service=SimpleNamespace(
                service_session_id=FALLBACK_V8_SERVICE_EVENT_ID,
                session_id=FALLBACK_V8_RUN_ID,
                service_microseconds=EXPECTED_SERVICE_MICROSECONDS,
                classified_event_microseconds=EXPECTED_CLASSIFIED_MICROSECONDS,
                overhead_microseconds=EXPECTED_OVERHEAD_MICROSECONDS,
                started_at=STARTED_AT,
                ended_at=ENDED_AT,
            ),
        ),
    )

    def validate_snapshot(
        path: Path,
        *,
        policy: object,
        policy_path: Path,
        snapshot_path: Path,
        shared_cache: Path,
    ) -> dict[str, object]:
        assert path == snapshot_manifest.resolve()
        assert policy is not None
        assert policy_path == (configuration_root / "fallback_model.json").resolve()
        assert snapshot_path == snapshot.resolve()
        assert shared_cache == cache.resolve()
        return {"manifest_sha256": SNAPSHOT_MANIFEST_SHA256}

    calls: list[dict[str, object]] = []
    ledger_paths: list[Path] = []

    class ObservedLedger(_FakeLedger):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            ledger_paths.append(path)

    class FakeService:
        def __init__(self, *, configuration: VLLMLaunchConfiguration, **_: object) -> None:
            self.configuration = configuration
            self.process_liveness_check = lambda pid: False
            self.process_group_liveness_check = lambda pid: False

        def _endpoint_live(self, timeout_seconds: float) -> bool:
            assert timeout_seconds == 0.25
            return False

        def recover_stale_service_lease(self, **kwargs: object) -> GpuServiceSession:
            calls.append(kwargs)
            current = json.loads(lease_path.read_text(encoding="utf-8"))
            assert kwargs == {
                "expected_current_lease_manifest_sha256": current[
                    "lease_manifest_sha256"
                ]
            }
            _write_private_json(
                lease_path,
                _lease_payload(self.configuration, state="stopped_verified"),
            )
            return _service_record()

    monkeypatch.setattr(
        "story_projection_onto.fallback_v8_lease_repair.load_fallback_v8_runtime_incident",
        lambda path: incident,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v8_lease_repair.validate_fallback_snapshot_manifest",
        validate_snapshot,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v8_lease_repair.Ledger",
        ObservedLedger,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v8_lease_repair.AllocatedGPUMeter",
        _FakeMeter,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v8_lease_repair.VLLMService",
        FakeService,
    )
    monkeypatch.setattr(repair_module, "_gpu_compute_process_count", lambda: 0)
    return SimpleNamespace(
        project=project,
        incident_path=incident_path,
        stop_intent=stop_intent,
        snapshot=snapshot,
        cache=cache,
        snapshot_manifest=snapshot_manifest,
        ledger=ledger,
        output=output,
        lease_path=lease_path,
        configuration=configuration,
        incident=incident,
        calls=calls,
        ledger_paths=ledger_paths,
    )


def _run(fixture: SimpleNamespace):
    return restore_fallback_v8_terminal_lease(
        project_root=fixture.project,
        incident_path=fixture.incident_path,
        stop_intent_path=fixture.stop_intent,
        snapshot_path=fixture.snapshot,
        shared_cache=fixture.cache,
        verified_snapshot_manifest_path=fixture.snapshot_manifest,
        ledger_path=fixture.ledger,
        output_path=fixture.output,
        port=8123,
    )


def test_exact_v8_lease_repair_is_ledger_neutral_and_idempotent(
    repair_fixture: SimpleNamespace,
) -> None:
    ledger_before = repair_fixture.ledger.read_bytes()
    receipt = _run(repair_fixture)

    assert receipt.kind == FALLBACK_V8_LEASE_REPAIR_KIND
    assert receipt.ledger_file_sha256_before == receipt.ledger_file_sha256_after
    assert repair_fixture.ledger.read_bytes() == ledger_before
    assert stat.S_IMODE(repair_fixture.output.stat().st_mode) == 0o600
    assert len(repair_fixture.calls) == 1
    assert receipt.gpu_events_added == 0
    assert receipt.accounting_rows_added == 0
    assert receipt.inference_attempts_added == 0
    assert receipt.inference_calls_added == 0
    assert len(repair_fixture.ledger_paths) == 1
    assert repair_fixture.ledger_paths[0] != repair_fixture.ledger
    assert not repair_fixture.ledger_paths[0].exists()
    assert json.loads(repair_fixture.lease_path.read_text(encoding="utf-8"))[
        "lease_state"
    ] == "stopped_verified"

    replayed = _run(repair_fixture)
    assert replayed == receipt
    assert len(repair_fixture.calls) == 1
    assert len(repair_fixture.ledger_paths) == 2
    assert not any(path.exists() for path in repair_fixture.ledger_paths)


def test_v8_repair_rejects_validly_rehashed_lease_drift(
    repair_fixture: SimpleNamespace,
) -> None:
    lease = json.loads(repair_fixture.lease_path.read_text(encoding="utf-8"))
    lease["controller_pid"] += 1
    immutable = {key: value for key, value in lease.items() if key != "lease_manifest_sha256"}
    lease["lease_manifest_sha256"] = canonical_sha256(immutable)
    _write_private_json(repair_fixture.lease_path, lease)
    lease_before = repair_fixture.lease_path.read_bytes()

    with pytest.raises(ValueError, match="incident-bound V8 damage state"):
        _run(repair_fixture)

    assert repair_fixture.lease_path.read_bytes() == lease_before
    assert repair_fixture.calls == []


def test_v8_repair_rejects_terminal_ledger_drift(
    repair_fixture: SimpleNamespace,
) -> None:
    repair_fixture.ledger.write_bytes(b"changed terminal ledger")

    with pytest.raises(ValueError, match="terminal ledger differs"):
        _run(repair_fixture)

    assert repair_fixture.calls == []


@pytest.mark.parametrize(
    ("changed_target", "expected_message"),
    (
        ("shadow", "attempted to change ledger contents"),
        ("canonical", "terminal V8 ledger changed during lease restoration"),
    ),
)
def test_v8_ledger_shadow_revalidates_during_exception_unwinding(
    repair_fixture: SimpleNamespace,
    changed_target: str,
    expected_message: str,
) -> None:
    ledger_sha256 = _sha256_file(repair_fixture.ledger)
    ledger_size = repair_fixture.ledger.stat().st_size

    with (
        pytest.raises(RuntimeError, match=expected_message),
        repair_module._terminal_ledger_shadow(
            repair_fixture.ledger,
            expected_sha256=ledger_sha256,
            expected_size_bytes=ledger_size,
        ) as shadow,
    ):
        changed_path = shadow.path if changed_target == "shadow" else repair_fixture.ledger
        changed_path.write_bytes(b"changed while the shadow body failed")
        raise ValueError("simulated repair-body failure")


def test_v8_repair_rejects_incident_pre_stop_lease_binding_drift(
    repair_fixture: SimpleNamespace,
) -> None:
    repair_fixture.incident.manual_recovery.pre_stop_lease_file_sha256 = "f" * 64

    with pytest.raises(ValueError, match="stop intent differs"):
        _run(repair_fixture)

    assert repair_fixture.calls == []


def test_v8_repair_rejects_live_incident_process(
    repair_fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_service = repair_module.VLLMService

    class LiveService(base_service):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.process_liveness_check = lambda pid: True

    monkeypatch.setattr(repair_module, "VLLMService", LiveService)
    lease_before = repair_fixture.lease_path.read_bytes()

    with pytest.raises(ValueError, match="service or endpoint is live"):
        _run(repair_fixture)

    assert repair_fixture.lease_path.read_bytes() == lease_before
    assert repair_fixture.calls == []


def test_v8_repair_rejects_live_gpu_compute_process(
    repair_fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repair_module, "_gpu_compute_process_count", lambda: 1)
    lease_before = repair_fixture.lease_path.read_bytes()

    with pytest.raises(ValueError, match="GPU compute process is live"):
        _run(repair_fixture)

    assert repair_fixture.lease_path.read_bytes() == lease_before
    assert repair_fixture.calls == []


def test_v8_repair_fails_closed_after_unreceipted_post_transition_crash(
    repair_fixture: SimpleNamespace,
) -> None:
    _write_private_json(
        repair_fixture.lease_path,
        _lease_payload(repair_fixture.configuration, state="stopped_verified"),
    )
    lease_before = repair_fixture.lease_path.read_bytes()

    with pytest.raises(ValueError, match="no independent post-transition binding"):
        _run(repair_fixture)

    assert repair_fixture.lease_path.read_bytes() == lease_before
    assert not repair_fixture.output.exists()
    assert repair_fixture.calls == []


def test_v8_repair_rejects_unreceipted_stopped_lease_with_substituted_token(
    repair_fixture: SimpleNamespace,
) -> None:
    lease = _lease_payload(repair_fixture.configuration, state="stopped_verified")
    lease["service_instance_token_sha256"] = "f" * 64
    immutable = {key: value for key, value in lease.items() if key != "lease_manifest_sha256"}
    lease["lease_manifest_sha256"] = canonical_sha256(immutable)
    _write_private_json(repair_fixture.lease_path, lease)
    lease_before = repair_fixture.lease_path.read_bytes()

    with pytest.raises(ValueError, match="no independent post-transition binding"):
        _run(repair_fixture)

    assert repair_fixture.lease_path.read_bytes() == lease_before
    assert not repair_fixture.output.exists()
    assert repair_fixture.calls == []


def test_existing_different_v8_receipt_fails_without_service_call(
    repair_fixture: SimpleNamespace,
) -> None:
    repair_fixture.output.write_text("{}\n", encoding="utf-8")
    repair_fixture.output.chmod(0o600)
    lease_before = repair_fixture.lease_path.read_bytes()

    with pytest.raises(FileExistsError, match="existing V8 lease repair receipt"):
        _run(repair_fixture)

    assert repair_fixture.lease_path.read_bytes() == lease_before
    assert repair_fixture.calls == []


def test_v8_receipt_output_traversal_is_rejected(
    repair_fixture: SimpleNamespace,
) -> None:
    repair_fixture.output = (
        repair_fixture.project / "artifacts/restricted/../public/traversed.json"
    )

    with pytest.raises(ValueError, match="cannot contain traversal"):
        _run(repair_fixture)

    assert not (repair_fixture.project / "artifacts/public/traversed.json").exists()
    assert repair_fixture.calls == []


def test_v8_repair_cli_forwards_only_declared_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_restore(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(
        "scripts.restore_fallback_v8_terminal_lease.restore_fallback_v8_terminal_lease",
        fake_restore,
    )
    assert (
        repair_cli_main(
            [
                "--project-root",
                "root",
                "--incident",
                "incident",
                "--stop-intent",
                "intent",
                "--snapshot",
                "snapshot",
                "--shared-cache",
                "cache",
                "--verified-snapshot-manifest",
                "manifest",
                "--ledger",
                "ledger",
                "--output",
                "output",
                "--port",
                "8123",
            ]
        )
        == 0
    )
    assert observed == {
        "project_root": Path("root"),
        "incident_path": Path("incident"),
        "stop_intent_path": Path("intent"),
        "snapshot_path": Path("snapshot"),
        "shared_cache": Path("cache"),
        "verified_snapshot_manifest_path": Path("manifest"),
        "ledger_path": Path("ledger"),
        "output_path": Path("output"),
        "port": 8123,
    }
