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

from scripts.restore_fallback_v7_terminal_lease import main as repair_cli_main
from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.fallback_v7_lease_repair import (
    FALLBACK_V7_LEASE_REPAIR_KIND,
    restore_fallback_v7_terminal_lease,
)
from story_projection_onto.fallback_v7_runtime_incident import (
    EXPECTED_CLASSIFIED_MICROSECONDS,
    EXPECTED_OVERHEAD_MICROSECONDS,
    EXPECTED_SERVICE_MICROSECONDS,
    FALLBACK_V7_RUN_ID,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REVISION,
    SERVICE_LEASE_SNAPSHOT_FILENAME,
    SERVICE_LOCK_FILENAME,
    VLLMLaunchConfiguration,
)
from story_projection_onto.store import GpuServiceSession

ROOT = Path(__file__).resolve().parents[2]
SERVICE_EVENT_ID = f"{FALLBACK_V7_RUN_ID}-service-start-001"
STARTED_AT = datetime(2026, 9, 5, 17, 49, 49, 471491, tzinfo=UTC)
ENDED_AT = datetime(2026, 9, 5, 18, 1, 22, 107047, tzinfo=UTC)
REPAIRED_AT = datetime(2026, 9, 5, 18, 20, 0, tzinfo=UTC)
SNAPSHOT_MANIFEST_SHA256 = "b" * 64


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _lease_payload(
    configuration: VLLMLaunchConfiguration,
    *,
    state: str,
    identity: SimpleNamespace | None = None,
) -> dict[str, Any]:
    final = state == "stopped_verified"
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "configuration_hash": configuration.configuration_hash,
        "controller_pid": os.getpid(),
        "lease_state": state,
        "session_id": FALLBACK_V7_RUN_ID if final else None,
        "accounting_session_id": SERVICE_EVENT_ID if final else None,
        "service_pid": identity.service_pid if final else None,
        "process_start_ticks": identity.process_start_ticks if final else None,
        "process_command_sha256": (
            canonical_sha256(list(configuration.command())) if final else None
        ),
        "process_group_id": identity.process_group_id if final else None,
        "process_session_id": identity.process_session_id if final else None,
        "service_instance_token_sha256": (
            identity.service_instance_token_sha256 if final else None
        ),
        "launch_protocol": None,
        "launch_gate_token_sha256": None,
        "launch_supervisor_command_sha256": None,
        "service_started_at": STARTED_AT.isoformat() if final else None,
        "service_ended_at": ENDED_AT.isoformat() if final else None,
        "ledger_allocated_seconds_before_session": 815.215409 if final else None,
        "observed_service_seconds": EXPECTED_SERVICE_MICROSECONDS / 1_000_000 if final else None,
        "updated_at": (REPAIRED_AT if final else ENDED_AT).isoformat(),
    }
    return {**payload, "lease_manifest_sha256": canonical_sha256(payload)}


def _incident(ledger_sha256: str) -> SimpleNamespace:
    identity = SimpleNamespace(
        service_pid=60_310,
        process_group_id=60_310,
        process_session_id=60_310,
        process_start_ticks=71_010,
        process_cmdline_sha256="c" * 64,
        service_instance_token_sha256="d" * 64,
    )
    service = SimpleNamespace(
        service_session_id=SERVICE_EVENT_ID,
        session_id=FALLBACK_V7_RUN_ID,
        service_microseconds=EXPECTED_SERVICE_MICROSECONDS,
        classified_event_microseconds=EXPECTED_CLASSIFIED_MICROSECONDS,
        overhead_microseconds=EXPECTED_OVERHEAD_MICROSECONDS,
        started_at=STARTED_AT,
        ended_at=ENDED_AT,
    )
    return SimpleNamespace(
        run_id=FALLBACK_V7_RUN_ID,
        manifest_sha256="a" * 64,
        orchestration_identity=SimpleNamespace(
            service_session_id=FALLBACK_V7_RUN_ID,
            service_event_id=SERVICE_EVENT_ID,
        ),
        manual_stop=SimpleNamespace(identity=identity),
        accounting=SimpleNamespace(
            terminal=SimpleNamespace(file_sha256=ledger_sha256),
            before_v7=SimpleNamespace(
                summary=SimpleNamespace(total_allocated_microseconds=815_215_409)
            ),
            service=service,
        ),
    )


class _FakeLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> _FakeLedger:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _FakeMeter:
    def __init__(self, ledger: _FakeLedger) -> None:
        self.ledger = ledger


@pytest.fixture
def repair_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    project = tmp_path / "project"
    config_root = project / "configs/study"
    config_root.mkdir(parents=True)
    shutil.copy2(ROOT / "configs/study/fallback_model.json", config_root)
    shutil.copy2(ROOT / "configs/study/model.json", config_root)

    cache = project / ".cache/shared"
    snapshot = (
        cache
        / "hub/models--Qwen--Qwen3-8B-AWQ/snapshots"
        / FALLBACK_MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    manifest = project / "verified_snapshot.json"
    manifest.write_text("{}\n", encoding="utf-8")
    incident_path = project / "incident.json"
    incident_path.write_text("{}\n", encoding="utf-8")
    ledger = project / "terminal.sqlite"
    ledger.write_bytes(b"immutable terminal ledger bytes")
    output = project / "restricted/lease_repair.json"
    incident = _incident(_sha256_file(ledger))

    def validate_snapshot(
        path: Path,
        *,
        policy: object,
        policy_path: Path,
        snapshot_path: Path,
        shared_cache: Path,
    ) -> dict[str, object]:
        assert path == manifest.resolve()
        assert policy_path == (config_root / "fallback_model.json").resolve()
        assert snapshot_path == snapshot.resolve()
        assert shared_cache == cache.resolve()
        assert policy is not None
        return {"manifest_sha256": SNAPSHOT_MANIFEST_SHA256}

    monkeypatch.setattr(
        "story_projection_onto.fallback_v7_lease_repair.load_fallback_v7_runtime_incident",
        lambda path: incident,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v7_lease_repair.validate_fallback_snapshot_manifest",
        validate_snapshot,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v7_lease_repair.Ledger",
        _FakeLedger,
    )
    monkeypatch.setattr(
        "story_projection_onto.fallback_v7_lease_repair.AllocatedGPUMeter",
        _FakeMeter,
    )

    configuration = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot,
        shared_cache=cache,
        model_configuration_path=config_root / "model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=SNAPSHOT_MANIFEST_SHA256,
        port=8123,
    )
    _write_private_json(
        cache / SERVICE_LEASE_SNAPSHOT_FILENAME,
        _lease_payload(configuration, state="shutdown_unverified"),
    )
    (cache / SERVICE_LOCK_FILENAME).write_text("{}\n", encoding="utf-8")
    (cache / SERVICE_LOCK_FILENAME).chmod(0o600)

    calls: list[dict[str, object]] = []

    class FakeService:
        def __init__(self, *, configuration: VLLMLaunchConfiguration, **_: object) -> None:
            self.configuration = configuration

        def restore_terminal_service_lease_from_identity(
            self, **kwargs: object
        ) -> GpuServiceSession:
            calls.append(kwargs)
            _write_private_json(
                cache / SERVICE_LEASE_SNAPSHOT_FILENAME,
                _lease_payload(
                    self.configuration,
                    state="stopped_verified",
                    identity=incident.manual_stop.identity,
                ),
            )
            return GpuServiceSession(
                service_session_id=SERVICE_EVENT_ID,
                session_id=FALLBACK_V7_RUN_ID,
                service_microseconds=EXPECTED_SERVICE_MICROSECONDS,
                classified_event_microseconds=EXPECTED_CLASSIFIED_MICROSECONDS,
                overhead_microseconds=EXPECTED_OVERHEAD_MICROSECONDS,
                started_at=STARTED_AT.isoformat().replace("+00:00", "Z"),
                ended_at=ENDED_AT.isoformat().replace("+00:00", "Z"),
                details_json="{}",
            )

    monkeypatch.setattr(
        "story_projection_onto.fallback_v7_lease_repair.VLLMService",
        FakeService,
    )
    return SimpleNamespace(
        project=project,
        incident_path=incident_path,
        snapshot=snapshot,
        cache=cache,
        manifest=manifest,
        ledger=ledger,
        output=output,
        incident=incident,
        calls=calls,
    )


def _run(fixture: SimpleNamespace):
    return restore_fallback_v7_terminal_lease(
        project_root=fixture.project,
        incident_path=fixture.incident_path,
        snapshot_path=fixture.snapshot,
        shared_cache=fixture.cache,
        verified_snapshot_manifest_path=fixture.manifest,
        ledger_path=fixture.ledger,
        output_path=fixture.output,
        port=8123,
    )


def test_one_shot_repair_derives_identity_and_replays_idempotently(
    repair_fixture: SimpleNamespace,
) -> None:
    ledger_before = repair_fixture.ledger.read_bytes()
    receipt = _run(repair_fixture)

    assert receipt.kind == FALLBACK_V7_LEASE_REPAIR_KIND
    assert receipt.ledger_file_sha256_before == receipt.ledger_file_sha256_after
    assert repair_fixture.ledger.read_bytes() == ledger_before
    assert stat.S_IMODE(repair_fixture.output.stat().st_mode) == 0o600
    assert len(repair_fixture.calls) == 1
    assert repair_fixture.calls[0] == {
        "expected_session_id": FALLBACK_V7_RUN_ID,
        "expected_event_id": SERVICE_EVENT_ID,
        "service_pid": 60_310,
        "process_start_ticks": 71_010,
        "observed_process_command_sha256": "c" * 64,
        "process_group_id": 60_310,
        "process_session_id": 60_310,
        "service_instance_token_sha256": "d" * 64,
    }
    raw_receipt = json.loads(repair_fixture.output.read_text(encoding="utf-8"))
    assert not any(
        forbidden in key.casefold()
        for key in raw_receipt
        for forbidden in ("path", "command", "token", "pid")
    )

    replayed = _run(repair_fixture)
    assert replayed == receipt
    assert len(repair_fixture.calls) == 1


def test_existing_different_receipt_fails_without_mutating_lease(
    repair_fixture: SimpleNamespace,
) -> None:
    repair_fixture.output.parent.mkdir(parents=True)
    repair_fixture.output.write_text("{}\n", encoding="utf-8")
    repair_fixture.output.chmod(0o600)
    lease_before = (repair_fixture.cache / SERVICE_LEASE_SNAPSHOT_FILENAME).read_bytes()

    with pytest.raises(FileExistsError, match="existing v7 lease repair receipt"):
        _run(repair_fixture)

    assert (repair_fixture.cache / SERVICE_LEASE_SNAPSHOT_FILENAME).read_bytes() == lease_before
    assert repair_fixture.calls == []


def test_terminal_ledger_hash_mismatch_fails_before_service_call(
    repair_fixture: SimpleNamespace,
) -> None:
    repair_fixture.ledger.write_bytes(b"changed")
    with pytest.raises(ValueError, match="terminal ledger SHA-256"):
        _run(repair_fixture)
    assert repair_fixture.calls == []


def test_damaged_lease_must_be_exact_and_self_hashed(
    repair_fixture: SimpleNamespace,
) -> None:
    lease_path = repair_fixture.cache / SERVICE_LEASE_SNAPSHOT_FILENAME
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["service_pid"] = 1
    _write_private_json(lease_path, lease)

    with pytest.raises(ValueError, match="self-hash is invalid"):
        _run(repair_fixture)
    assert repair_fixture.calls == []


def test_cli_forwards_only_declared_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_restore(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(
        "scripts.restore_fallback_v7_terminal_lease.restore_fallback_v7_terminal_lease",
        fake_restore,
    )
    assert (
        repair_cli_main(
            [
                "--project-root",
                "root",
                "--incident",
                "incident",
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
        "snapshot_path": Path("snapshot"),
        "shared_cache": Path("cache"),
        "verified_snapshot_manifest_path": Path("manifest"),
        "ledger_path": Path("ledger"),
        "output_path": Path("output"),
        "port": 8123,
    }
