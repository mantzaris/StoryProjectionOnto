"""One-shot restoration of the terminal fallback-v7 vLLM lease.

This module is deliberately narrower than general service recovery.  It accepts
only the self-hashed, null-identity ``shutdown_unverified`` lease documented by
the final typed v7 incident, delegates the actual transition to
``VLLMService.restore_terminal_service_lease_from_identity``, and proves that
the terminal accounting ledger did not change.  The resulting restricted
receipt contains hashes and safe identifiers only; it never records paths,
commands, process identifiers, or token material.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from story_projection_onto.contracts import Sha256Digest, canonical_json, canonical_sha256
from story_projection_onto.experiment import AllocatedGPUMeter
from story_projection_onto.fallback_v7_runtime_incident import (
    EXPECTED_SERVICE_MICROSECONDS,
    FALLBACK_V7_RUN_ID,
    FALLBACK_V7_RUNTIME_INCIDENT_KIND,
    FallbackV7RuntimeIncident,
    load_fallback_v7_runtime_incident,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    SCHEMA_VERSION,
    SERVICE_LEASE_SNAPSHOT_FILENAME,
    SERVICE_LOCK_FILENAME,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
)
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    validate_fallback_snapshot_manifest,
)
from story_projection_onto.store import GpuServiceSession, Ledger

FALLBACK_V7_LEASE_REPAIR_KIND = "fallback_v7_terminal_lease_repair_receipt"
FALLBACK_V7_SERVICE_EVENT_ID = f"{FALLBACK_V7_RUN_ID}-service-start-001"

_LEASE_FIELDS = frozenset(
    {
        "schema_version",
        "configuration_hash",
        "controller_pid",
        "lease_state",
        "session_id",
        "accounting_session_id",
        "service_pid",
        "process_start_ticks",
        "process_command_sha256",
        "process_group_id",
        "process_session_id",
        "service_instance_token_sha256",
        "launch_protocol",
        "launch_gate_token_sha256",
        "launch_supervisor_command_sha256",
        "service_started_at",
        "service_ended_at",
        "ledger_allocated_seconds_before_session",
        "observed_service_seconds",
        "updated_at",
        "lease_manifest_sha256",
    }
)
_NULL_DAMAGE_FIELDS = (
    "session_id",
    "accounting_session_id",
    "service_pid",
    "process_start_ticks",
    "process_command_sha256",
    "process_group_id",
    "process_session_id",
    "service_instance_token_sha256",
    "launch_protocol",
    "launch_gate_token_sha256",
    "launch_supervisor_command_sha256",
    "service_started_at",
    "service_ended_at",
    "ledger_allocated_seconds_before_session",
    "observed_service_seconds",
)


class FallbackV7LeaseRepairReceipt(BaseModel):
    """Restricted, self-hashed proof of the one permitted lease transition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_v7_terminal_lease_repair_receipt"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v7"]
    source_incident_kind: Literal["fallback_gpu_acceptance_v7_runtime_incident"]
    source_incident_manifest_sha256: Sha256Digest
    source_incident_file_sha256: Sha256Digest
    model_repository: Literal["Qwen/Qwen3-8B-AWQ"]
    model_revision: Literal["4da05a8edb55c6046cce958586c33b61da07bb79"]
    served_model_name: Literal["qwen3-8b-awq-fallback"]
    verified_snapshot_manifest_sha256: Sha256Digest
    verified_snapshot_manifest_file_sha256: Sha256Digest
    configuration_hash: Sha256Digest
    configuration_manifest_sha256: Sha256Digest
    service_session_id: Literal["fallback-qwen3-8b-awq-development-v7"]
    service_event_id: Literal[
        "fallback-qwen3-8b-awq-development-v7-service-start-001"
    ]
    process_identity_sha256: Sha256Digest
    terminal_service_record_sha256: Sha256Digest
    service_microseconds: Literal[692635556]
    ledger_size_bytes: int = Field(gt=0, strict=True)
    ledger_file_sha256_before: Sha256Digest
    ledger_file_sha256_after: Sha256Digest
    ledger_bytes_unchanged: Literal[True]
    accounting_rows_added: Literal[0]
    inference_calls_added: Literal[0]
    damaged_lease_manifest_sha256: Sha256Digest
    damaged_lease_file_sha256: Sha256Digest
    restored_lease_manifest_sha256: Sha256Digest
    restored_lease_file_sha256: Sha256Digest
    restored_lease_state: Literal["stopped_verified"]
    terminal_absence_revalidated: Literal[True]
    repaired_at: AwareDatetime
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_reconciliation_and_manifest(self) -> Self:
        if self.ledger_file_sha256_before != self.ledger_file_sha256_after:
            raise ValueError("v7 lease repair changed terminal ledger bytes")
        if self.damaged_lease_manifest_sha256 == self.restored_lease_manifest_sha256:
            raise ValueError("v7 lease repair did not change lease state")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_sha256(immutable):
            raise ValueError("v7 lease repair receipt manifest hash changed")
        return self


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require_no_symlink_ancestry(path: Path, *, label: str) -> None:
    supplied = Path(path).absolute()
    for component in (supplied, *supplied.parents):
        if component.is_symlink():
            raise ValueError(f"{label} cannot have symlink ancestry")


def _require_regular_file(path: Path, *, label: str) -> Path:
    supplied = Path(path).absolute()
    _require_no_symlink_ancestry(supplied, label=label)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _require_directory(path: Path, *, label: str) -> Path:
    supplied = Path(path).absolute()
    _require_no_symlink_ancestry(supplied, label=label)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _parse_aware_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        )
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an offset")
    return parsed


def _load_lease(path: Path, *, label: str) -> tuple[dict[str, Any], str, str]:
    resolved = _require_regular_file(path, label=label)
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > 65_536:
        raise ValueError(f"{label} size is invalid")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != _LEASE_FIELDS:
        raise ValueError(f"{label} fields differ from the exact vLLM lease contract")
    supplied_hash = value.get("lease_manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "lease_manifest_sha256"}
    if not isinstance(supplied_hash, str) or supplied_hash != canonical_sha256(immutable):
        raise ValueError(f"{label} self-hash is invalid")
    return cast(dict[str, Any], value), supplied_hash, _sha256_file(resolved)


def _require_private_mode(path: Path, *, label: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"{label} must have mode 0600")


def _validate_damaged_lease(
    lease: Mapping[str, Any],
    *,
    configuration: VLLMLaunchConfiguration,
) -> None:
    if (
        lease.get("schema_version") != SCHEMA_VERSION
        or lease.get("configuration_hash") != configuration.configuration_hash
        or lease.get("lease_state") != "shutdown_unverified"
        or any(lease.get(name) is not None for name in _NULL_DAMAGE_FIELDS)
    ):
        raise ValueError("service lease is not the exact v7 null-identity damage state")
    controller_pid = lease.get("controller_pid")
    if (
        isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 0
    ):
        raise ValueError("damaged v7 lease controller identity is invalid")
    _parse_aware_timestamp(lease.get("updated_at"), label="damaged v7 lease update")


def _identity_sha256(incident: FallbackV7RuntimeIncident) -> str:
    identity = incident.manual_stop.identity
    return canonical_sha256(
        {
            "service_pid": identity.service_pid,
            "process_group_id": identity.process_group_id,
            "process_session_id": identity.process_session_id,
            "process_start_ticks": identity.process_start_ticks,
            "process_cmdline_sha256": identity.process_cmdline_sha256,
            "service_instance_token_sha256": identity.service_instance_token_sha256,
        }
    )


def _service_record_payload(record: GpuServiceSession) -> dict[str, object]:
    return {
        "service_session_id": record.service_session_id,
        "session_id": record.session_id,
        "service_microseconds": record.service_microseconds,
        "classified_event_microseconds": record.classified_event_microseconds,
        "overhead_microseconds": record.overhead_microseconds,
        "started_at": _parse_aware_timestamp(record.started_at, label="service record start"),
        "ended_at": _parse_aware_timestamp(record.ended_at, label="service record end"),
    }


def _validate_service_record(
    record: GpuServiceSession,
    incident: FallbackV7RuntimeIncident,
) -> None:
    expected = incident.accounting.service
    if _service_record_payload(record) != {
        "service_session_id": expected.service_session_id,
        "session_id": expected.session_id,
        "service_microseconds": expected.service_microseconds,
        "classified_event_microseconds": expected.classified_event_microseconds,
        "overhead_microseconds": expected.overhead_microseconds,
        "started_at": expected.started_at,
        "ended_at": expected.ended_at,
    }:
        raise ValueError("restored lease service row differs from the final v7 incident")


def _validate_restored_lease(
    lease: Mapping[str, Any],
    *,
    configuration: VLLMLaunchConfiguration,
    incident: FallbackV7RuntimeIncident,
) -> datetime:
    identity = incident.manual_stop.identity
    service = incident.accounting.service
    expected_command_sha256 = canonical_sha256(list(configuration.command()))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": configuration.configuration_hash,
        "lease_state": "stopped_verified",
        "session_id": incident.orchestration_identity.service_session_id,
        "accounting_session_id": incident.orchestration_identity.service_event_id,
        "service_pid": identity.service_pid,
        "process_start_ticks": identity.process_start_ticks,
        "process_command_sha256": expected_command_sha256,
        "process_group_id": identity.process_group_id,
        "process_session_id": identity.process_session_id,
        "service_instance_token_sha256": identity.service_instance_token_sha256,
        "launch_protocol": None,
        "launch_gate_token_sha256": None,
        "launch_supervisor_command_sha256": None,
        "service_started_at": service.started_at.isoformat(),
        "service_ended_at": service.ended_at.isoformat(),
        "ledger_allocated_seconds_before_session": (
            incident.accounting.before_v7.summary.total_allocated_microseconds / 1_000_000
        ),
        "observed_service_seconds": service.service_microseconds / 1_000_000,
    }
    if any(lease.get(name) != value for name, value in expected.items()):
        raise ValueError("restored v7 lease differs from its terminal incident identity")
    controller_pid = lease.get("controller_pid")
    if (
        isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 0
    ):
        raise ValueError("restored v7 lease controller identity is invalid")
    return _parse_aware_timestamp(lease.get("updated_at"), label="restored v7 lease update")


def _receipt_payload(
    *,
    incident: FallbackV7RuntimeIncident,
    incident_file_sha256: str,
    snapshot_manifest: Mapping[str, object],
    snapshot_manifest_file_sha256: str,
    configuration: VLLMLaunchConfiguration,
    service_record_sha256: str,
    process_identity_sha256: str,
    ledger_size_bytes: int,
    ledger_sha256: str,
    damaged_lease_manifest_sha256: str,
    damaged_lease_file_sha256: str,
    restored_lease_manifest_sha256: str,
    restored_lease_file_sha256: str,
    repaired_at: datetime,
) -> dict[str, object]:
    configuration_manifest = configuration.public_manifest()
    return {
        "schema_version": "1.0.0",
        "kind": FALLBACK_V7_LEASE_REPAIR_KIND,
        "run_id": incident.run_id,
        "source_incident_kind": FALLBACK_V7_RUNTIME_INCIDENT_KIND,
        "source_incident_manifest_sha256": incident.manifest_sha256,
        "source_incident_file_sha256": incident_file_sha256,
        "model_repository": configuration.repository,
        "model_revision": configuration.revision,
        "served_model_name": configuration.served_model_name,
        "verified_snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "verified_snapshot_manifest_file_sha256": snapshot_manifest_file_sha256,
        "configuration_hash": configuration.configuration_hash,
        "configuration_manifest_sha256": configuration_manifest["manifest_sha256"],
        "service_session_id": incident.orchestration_identity.service_session_id,
        "service_event_id": incident.orchestration_identity.service_event_id,
        "process_identity_sha256": process_identity_sha256,
        "terminal_service_record_sha256": service_record_sha256,
        "service_microseconds": EXPECTED_SERVICE_MICROSECONDS,
        "ledger_size_bytes": ledger_size_bytes,
        "ledger_file_sha256_before": ledger_sha256,
        "ledger_file_sha256_after": ledger_sha256,
        "ledger_bytes_unchanged": True,
        "accounting_rows_added": 0,
        "inference_calls_added": 0,
        "damaged_lease_manifest_sha256": damaged_lease_manifest_sha256,
        "damaged_lease_file_sha256": damaged_lease_file_sha256,
        "restored_lease_manifest_sha256": restored_lease_manifest_sha256,
        "restored_lease_file_sha256": restored_lease_file_sha256,
        "restored_lease_state": "stopped_verified",
        "terminal_absence_revalidated": True,
        "repaired_at": repaired_at,
    }


def _load_receipt(path: Path) -> FallbackV7LeaseRepairReceipt:
    resolved = _require_regular_file(path, label="v7 lease repair receipt")
    _require_private_mode(resolved, label="v7 lease repair receipt")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("v7 lease repair receipt is not valid UTF-8 JSON") from exc
    try:
        return FallbackV7LeaseRepairReceipt.model_validate(value)
    except ValidationError as exc:
        raise ValueError("v7 lease repair receipt contract is invalid") from exc


def _write_receipt(path: Path, receipt: FallbackV7LeaseRepairReceipt) -> None:
    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="v7 lease repair receipt output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(destination, label="v7 lease repair receipt output")
    payload = (canonical_json(receipt) + "\n").encode("utf-8")
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and stat.S_IMODE(destination.stat().st_mode) == 0o600
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError("append-only v7 lease repair receipt already differs")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _validate_replay(
    receipt: FallbackV7LeaseRepairReceipt,
    *,
    incident: FallbackV7RuntimeIncident,
    incident_file_sha256: str,
    snapshot_manifest: Mapping[str, object],
    snapshot_manifest_file_sha256: str,
    configuration: VLLMLaunchConfiguration,
    ledger_size_bytes: int,
    ledger_sha256: str,
    lease: Mapping[str, Any],
    lease_manifest_sha256: str,
    lease_file_sha256: str,
) -> None:
    repaired_at = _validate_restored_lease(
        lease,
        configuration=configuration,
        incident=incident,
    )
    service = incident.accounting.service
    service_payload = {
        "service_session_id": service.service_session_id,
        "session_id": service.session_id,
        "service_microseconds": service.service_microseconds,
        "classified_event_microseconds": service.classified_event_microseconds,
        "overhead_microseconds": service.overhead_microseconds,
        "started_at": service.started_at,
        "ended_at": service.ended_at,
    }
    expected = _receipt_payload(
        incident=incident,
        incident_file_sha256=incident_file_sha256,
        snapshot_manifest=snapshot_manifest,
        snapshot_manifest_file_sha256=snapshot_manifest_file_sha256,
        configuration=configuration,
        service_record_sha256=canonical_sha256(service_payload),
        process_identity_sha256=_identity_sha256(incident),
        ledger_size_bytes=ledger_size_bytes,
        ledger_sha256=ledger_sha256,
        damaged_lease_manifest_sha256=receipt.damaged_lease_manifest_sha256,
        damaged_lease_file_sha256=receipt.damaged_lease_file_sha256,
        restored_lease_manifest_sha256=lease_manifest_sha256,
        restored_lease_file_sha256=lease_file_sha256,
        repaired_at=repaired_at,
    )
    expected_receipt = FallbackV7LeaseRepairReceipt.model_validate(
        {**expected, "manifest_sha256": canonical_sha256(expected)}
    )
    if receipt != expected_receipt:
        raise FileExistsError("existing v7 lease repair receipt differs from current state")


def restore_fallback_v7_terminal_lease(
    *,
    project_root: Path,
    incident_path: Path,
    snapshot_path: Path,
    shared_cache: Path,
    verified_snapshot_manifest_path: Path,
    ledger_path: Path,
    output_path: Path,
    port: int,
) -> FallbackV7LeaseRepairReceipt:
    """Restore the exact terminal v7 lease and append its restricted receipt."""

    root = _require_directory(project_root, label="project root")
    incident_resolved = _require_regular_file(incident_path, label="fallback-v7 incident")
    incident = load_fallback_v7_runtime_incident(incident_resolved)
    if incident.run_id != FALLBACK_V7_RUN_ID:
        raise ValueError("lease repair is restricted to the final fallback-v7 incident")

    cache = _require_directory(shared_cache, label="shared cache")
    snapshot = _require_directory(snapshot_path, label="fallback snapshot")
    snapshot_manifest_resolved = _require_regular_file(
        verified_snapshot_manifest_path,
        label="verified fallback snapshot manifest",
    )
    policy_path = _require_regular_file(
        root / "configs/study/fallback_model.json",
        label="fallback model policy",
    )
    model_configuration_path = _require_regular_file(
        root / "configs/study/model.json",
        label="model configuration",
    )
    policy = FallbackModelPolicy.load(policy_path)
    snapshot_manifest = validate_fallback_snapshot_manifest(
        snapshot_manifest_resolved,
        policy=policy,
        policy_path=policy_path,
        snapshot_path=snapshot,
        shared_cache=cache,
    )
    configuration = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot,
        shared_cache=cache,
        model_configuration_path=model_configuration_path,
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=cast(str, snapshot_manifest["manifest_sha256"]),
        port=port,
    )
    if (
        configuration.repository != FALLBACK_MODEL_REPOSITORY
        or configuration.revision != FALLBACK_MODEL_REVISION
        or configuration.served_model_name != FALLBACK_SERVED_MODEL_NAME
        or configuration.model_candidate != "fallback"
    ):
        raise ValueError("lease repair configuration is not the pinned 8B fallback")

    ledger = _require_regular_file(ledger_path, label="terminal GPU ledger")
    ledger_sha256 = _sha256_file(ledger)
    ledger_size_bytes = ledger.stat().st_size
    if ledger_sha256 != incident.accounting.terminal.file_sha256:
        raise ValueError("current terminal ledger SHA-256 differs from the v7 incident")

    incident_file_sha256 = _sha256_file(incident_resolved)
    snapshot_manifest_file_sha256 = _sha256_file(snapshot_manifest_resolved)
    lease_path = cache / SERVICE_LEASE_SNAPSHOT_FILENAME
    lock_path = _require_regular_file(
        cache / SERVICE_LOCK_FILENAME,
        label="vLLM service lock",
    )
    _require_private_mode(lock_path, label="vLLM service lock")

    output = Path(output_path).absolute()
    _require_no_symlink_ancestry(output, label="v7 lease repair receipt output")
    if output.exists():
        try:
            receipt = _load_receipt(output)
        except (ValueError, OSError) as exc:
            raise FileExistsError("existing v7 lease repair receipt is invalid") from exc
        restored_lease, restored_manifest_sha256, restored_file_sha256 = _load_lease(
            lease_path,
            label="restored v7 service lease",
        )
        _require_private_mode(lease_path, label="restored v7 service lease")
        _validate_replay(
            receipt,
            incident=incident,
            incident_file_sha256=incident_file_sha256,
            snapshot_manifest=snapshot_manifest,
            snapshot_manifest_file_sha256=snapshot_manifest_file_sha256,
            configuration=configuration,
            ledger_size_bytes=ledger_size_bytes,
            ledger_sha256=ledger_sha256,
            lease=restored_lease,
            lease_manifest_sha256=restored_manifest_sha256,
            lease_file_sha256=restored_file_sha256,
        )
        return receipt

    damaged_lease, damaged_manifest_sha256, damaged_file_sha256 = _load_lease(
        lease_path,
        label="damaged v7 service lease",
    )
    _require_private_mode(lease_path, label="damaged v7 service lease")
    _validate_damaged_lease(damaged_lease, configuration=configuration)

    identity = incident.manual_stop.identity
    with Ledger(ledger) as ledger_store:
        meter = AllocatedGPUMeter(ledger_store)
        service = VLLMService(
            configuration=configuration,
            client=VLLMGuidedJSONClient(configuration.base_url),
            meter=meter,
        )
        record = service.restore_terminal_service_lease_from_identity(
            expected_session_id=incident.orchestration_identity.service_session_id,
            expected_event_id=incident.orchestration_identity.service_event_id,
            service_pid=identity.service_pid,
            process_start_ticks=identity.process_start_ticks,
            observed_process_command_sha256=identity.process_cmdline_sha256,
            process_group_id=identity.process_group_id,
            process_session_id=identity.process_session_id,
            service_instance_token_sha256=identity.service_instance_token_sha256,
        )
        _validate_service_record(record, incident)

    ledger_sha256_after = _sha256_file(ledger)
    if ledger.stat().st_size != ledger_size_bytes or ledger_sha256_after != ledger_sha256:
        raise RuntimeError("terminal v7 lease restoration changed ledger bytes")

    restored_lease, restored_manifest_sha256, restored_file_sha256 = _load_lease(
        lease_path,
        label="restored v7 service lease",
    )
    _require_private_mode(lease_path, label="restored v7 service lease")
    repaired_at = _validate_restored_lease(
        restored_lease,
        configuration=configuration,
        incident=incident,
    )
    payload = _receipt_payload(
        incident=incident,
        incident_file_sha256=incident_file_sha256,
        snapshot_manifest=snapshot_manifest,
        snapshot_manifest_file_sha256=snapshot_manifest_file_sha256,
        configuration=configuration,
        service_record_sha256=canonical_sha256(_service_record_payload(record)),
        process_identity_sha256=_identity_sha256(incident),
        ledger_size_bytes=ledger_size_bytes,
        ledger_sha256=ledger_sha256,
        damaged_lease_manifest_sha256=damaged_manifest_sha256,
        damaged_lease_file_sha256=damaged_file_sha256,
        restored_lease_manifest_sha256=restored_manifest_sha256,
        restored_lease_file_sha256=restored_file_sha256,
        repaired_at=repaired_at,
    )
    receipt = FallbackV7LeaseRepairReceipt.model_validate(
        {**payload, "manifest_sha256": canonical_sha256(payload)}
    )
    _write_receipt(output, receipt)
    return _load_receipt(output)


__all__ = [
    "FALLBACK_V7_LEASE_REPAIR_KIND",
    "FallbackV7LeaseRepairReceipt",
    "restore_fallback_v7_terminal_lease",
]
