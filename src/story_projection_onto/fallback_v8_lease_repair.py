"""Exact CPU-only restoration of the terminal fallback-v8 service lease.

The V8 service was stopped and fully accounted before this utility runs, but
its durable lease remains ``shutdown_unverified``.  This module accepts only
the incident-bound preserved lease and terminal ledger, delegates the lease
transition to the standard stale-service recovery path with an under-lock
manifest pin, and proves that no ledger byte or inference record changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from story_projection_onto.contracts import Sha256Digest, canonical_json, canonical_sha256
from story_projection_onto.experiment import AllocatedGPUMeter
from story_projection_onto.fallback_v7_lease_repair import (
    FallbackV7LeaseRepairAccessControl,
)
from story_projection_onto.fallback_v7_lease_repair import (
    _validate_private_access_control as _validate_v7_private_access_control,
)
from story_projection_onto.fallback_v8_runtime_incident import (
    EXPECTED_BASELINE_MICROSECONDS,
    EXPECTED_CLASSIFIED_MICROSECONDS,
    EXPECTED_OVERHEAD_MICROSECONDS,
    EXPECTED_SERVICE_MICROSECONDS,
    EXPECTED_TERMINAL_MICROSECONDS,
    FALLBACK_V8_RUN_ID,
    FALLBACK_V8_RUNTIME_INCIDENT_KIND,
    FALLBACK_V8_SERVICE_EVENT_ID,
    FallbackV8RuntimeIncident,
    load_fallback_v8_runtime_incident,
)
from story_projection_onto.gpu_runtime import (
    DURABLE_EXEC_GATE_PROTOCOL,
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

FALLBACK_V8_LEASE_REPAIR_KIND = "fallback_v8_terminal_lease_repair_receipt"

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
_STOP_INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "run_id",
        "service_event_id",
        "recorded_at",
        "lease_file_sha256",
        "lease_manifest_sha256",
        "process_group_id",
        "member_identities",
        "sole_gpu_process",
        "sole_port_8000_owner_verified",
        "signal_plan",
        "reason",
        "utility_file_sha256",
        "manifest_sha256",
    }
)
_SAFE_RECEIPT_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\.json$")


class FallbackV8LeaseRepairAccessControl(FallbackV7LeaseRepairAccessControl):
    """The already-audited ordinary/FUSE protection proof under a V8 name."""


class FallbackV8LeaseRepairReceipt(BaseModel):
    """Restricted, self-hashed proof of the exact V8 lease-only transition."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_v8_terminal_lease_repair_receipt"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v8"]
    source_incident_kind: Literal["fallback_gpu_acceptance_v8_runtime_incident"]
    source_incident_manifest_sha256: Sha256Digest
    source_incident_file_sha256: Sha256Digest
    stop_intent_manifest_sha256: Sha256Digest
    stop_intent_file_sha256: Sha256Digest
    model_repository: Literal["Qwen/Qwen3-8B-AWQ"]
    model_revision: Literal["4da05a8edb55c6046cce958586c33b61da07bb79"]
    served_model_name: Literal["qwen3-8b-awq-fallback"]
    verified_snapshot_manifest_sha256: Sha256Digest
    verified_snapshot_manifest_file_sha256: Sha256Digest
    configuration_hash: Sha256Digest
    configuration_manifest_sha256: Sha256Digest
    access_control: FallbackV8LeaseRepairAccessControl
    service_session_id: Literal["fallback-qwen3-8b-awq-development-v8"]
    service_event_id: Literal[
        "fallback-qwen3-8b-awq-development-v8-service-start-001"
    ]
    process_identity_sha256: Sha256Digest
    terminal_service_record_sha256: Sha256Digest
    service_microseconds: Literal[1073416738]
    classified_event_microseconds: Literal[227686586]
    overhead_microseconds: Literal[845730152]
    terminal_allocated_microseconds: Literal[2581267703]
    ledger_size_bytes: int = Field(gt=0, strict=True)
    ledger_file_sha256_before: Sha256Digest
    ledger_file_sha256_after: Sha256Digest
    ledger_bytes_unchanged: Literal[True]
    gpu_events_added: Literal[0]
    accounting_rows_added: Literal[0]
    inference_attempts_added: Literal[0]
    inference_calls_added: Literal[0]
    damaged_lease_manifest_sha256: Sha256Digest
    damaged_lease_file_sha256: Sha256Digest
    restored_lease_manifest_sha256: Sha256Digest
    restored_lease_file_sha256: Sha256Digest
    restored_lease_state: Literal["stopped_verified"]
    terminal_process_absence_revalidated: Literal[True]
    terminal_gpu_compute_absence_revalidated: Literal[True]
    repaired_at: AwareDatetime
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_transition_and_manifest(self) -> Self:
        if self.ledger_file_sha256_before != self.ledger_file_sha256_after:
            raise ValueError("v8 lease repair changed terminal ledger bytes")
        if self.damaged_lease_manifest_sha256 == self.restored_lease_manifest_sha256:
            raise ValueError("v8 lease repair did not change lease state")
        if self.service_microseconds != (
            self.classified_event_microseconds + self.overhead_microseconds
        ):
            raise ValueError("v8 lease repair service accounting does not reconcile")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_sha256(immutable):
            raise ValueError("v8 lease repair receipt manifest hash changed")
        return self


@dataclass(frozen=True, slots=True)
class _LeaseIdentity:
    service_pid: int
    process_start_ticks: int
    process_command_sha256: str
    process_group_id: int
    process_session_id: int
    service_instance_token_sha256: str
    launch_protocol: str
    launch_gate_token_sha256: str
    launch_supervisor_command_sha256: str


def _is_canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _require_beneath(path: Path, parent: Path, *, label: str) -> None:
    try:
        relative = path.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{label} is outside its required project namespace") from exc
    if not relative.parts:
        raise ValueError(f"{label} must name a child in its required project namespace")


def _normalize_output_path(path: Path, *, root: Path) -> tuple[Path, Path]:
    supplied = Path(path).absolute()
    if ".." in supplied.parts:
        raise ValueError("v8 lease repair receipt output cannot contain traversal")
    if _SAFE_RECEIPT_BASENAME.fullmatch(supplied.name) is None:
        raise ValueError("v8 lease repair receipt output has an unsafe basename")
    restricted_root = (root / "artifacts/restricted").resolve(strict=True)
    _require_beneath(supplied, restricted_root, label="v8 lease repair receipt output")
    _require_no_symlink_ancestry(supplied, label="v8 lease repair receipt output")
    supplied.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(supplied, label="v8 lease repair receipt output")
    parent = supplied.parent.resolve(strict=True)
    normalized = parent / supplied.name
    _require_beneath(normalized, restricted_root, label="v8 lease repair receipt output")
    return normalized, parent


def _require_exact_layout(
    *,
    root: Path,
    incident: Path,
    stop_intent: Path,
    snapshot: Path,
    cache: Path,
    snapshot_manifest: Path,
    ledger: Path,
    output: Path,
) -> tuple[Path, Path]:
    expected_cache = (root / ".cache/shared").resolve(strict=True)
    if cache != expected_cache:
        raise ValueError("fixed fallback repair requires the project shared cache")
    _require_beneath(snapshot, cache, label="fallback snapshot")
    public_root = (root / "artifacts/public").resolve(strict=True)
    restricted_root = (root / "artifacts/restricted").resolve(strict=True)
    _require_beneath(incident, public_root, label="fallback-v8 incident")
    _require_beneath(
        snapshot_manifest,
        public_root,
        label="verified fallback snapshot manifest",
    )
    _require_beneath(stop_intent, restricted_root, label="fallback-v8 stop intent")
    _require_beneath(ledger, restricted_root, label="terminal GPU ledger")
    return _normalize_output_path(output, root=root)


def _load_self_hashed_json(
    path: Path,
    *,
    label: str,
    expected_fields: frozenset[str] | None = None,
) -> tuple[Path, dict[str, Any], str, str]:
    resolved = _require_regular_file(path, label=label)
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > 1_048_576:
        raise ValueError(f"{label} size is invalid")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if expected_fields is not None and set(value) != expected_fields:
        raise ValueError(f"{label} fields differ from its exact contract")
    supplied_hash = value.get("manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not isinstance(supplied_hash, str) or supplied_hash != canonical_sha256(immutable):
        raise ValueError(f"{label} self-hash is invalid")
    return resolved, cast(dict[str, Any], value), supplied_hash, _sha256_file(resolved)


def _load_lease(path: Path, *, label: str) -> tuple[dict[str, Any], str, str]:
    resolved = _require_regular_file(path, label=label)
    if resolved.stat().st_size <= 0 or resolved.stat().st_size > 1_048_576:
        raise ValueError(f"{label} size is invalid")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    lease = cast(dict[str, Any], value)
    if set(lease) != _LEASE_FIELDS:
        raise ValueError(f"{label} fields differ from the exact vLLM lease contract")
    manifest = lease["lease_manifest_sha256"]
    immutable = {
        key: item for key, item in lease.items() if key != "lease_manifest_sha256"
    }
    if not isinstance(manifest, str) or manifest != canonical_sha256(immutable):
        raise ValueError(f"{label} self-hash is invalid")
    return lease, manifest, _sha256_file(resolved)


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


def _microseconds(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return round(converted * 1_000_000)


def _load_and_validate_stop_intent(
    path: Path,
    *,
    incident: FallbackV8RuntimeIncident,
) -> tuple[dict[str, Any], str, str]:
    resolved, value, manifest, file_sha256 = _load_self_hashed_json(
        path,
        label="fallback-v8 stop intent",
        expected_fields=_STOP_INTENT_FIELDS,
    )
    binding = incident.manual_recovery.stop_intent
    if (
        resolved.name != binding.basename
        or resolved.stat().st_size != binding.size_bytes
        or value.get("kind") != binding.kind
        or manifest != binding.manifest_sha256
        or file_sha256 != binding.file_sha256
        or value.get("run_id") != FALLBACK_V8_RUN_ID
        or value.get("service_event_id") != FALLBACK_V8_SERVICE_EVENT_ID
        or value.get("process_group_id") != incident.manual_recovery.process_group_id
        or value.get("lease_manifest_sha256")
        != incident.manual_recovery.pre_stop_lease_manifest_sha256
        or value.get("lease_file_sha256")
        != incident.manual_recovery.pre_stop_lease_file_sha256
    ):
        raise ValueError("fallback-v8 stop intent differs from its incident binding")
    members = value.get("member_identities")
    if not isinstance(members, list) or len(members) != 3:
        raise ValueError("fallback-v8 stop intent process inventory changed")
    for observed, expected in zip(
        members,
        incident.manual_recovery.process_members,
        strict=True,
    ):
        if not isinstance(observed, dict) or any(
            observed.get(intent_field) != getattr(expected, incident_field)
            for intent_field, incident_field in (
                ("pid", "pid"),
                ("ppid", "parent_pid"),
                ("process_group", "process_group_id"),
                ("session_id", "session_id"),
                ("start_ticks", "start_ticks"),
                ("comm", "process_name"),
                ("state", "process_state"),
                ("raw_command_sha256", "argv_sha256"),
            )
        ):
            raise ValueError("fallback-v8 stop intent process inventory changed")
    if (
        members[0].get("token_matches") is not True
        or members[1].get("token_matches") is not True
        or members[2].get("token_matches")
        is not incident.manual_recovery.engine_core_token_matched
    ):
        raise ValueError("fallback-v8 stop intent token-match inventory changed")
    gpu_process = value.get("sole_gpu_process")
    if (
        not isinstance(gpu_process, dict)
        or gpu_process.get("pid") != incident.manual_recovery.process_members[2].pid
        or gpu_process.get("used_gpu_memory_mib")
        != incident.manual_recovery.sole_gpu_process_memory_mib
        or value.get("sole_port_8000_owner_verified")
        is not incident.manual_recovery.sole_port_owner_verified
    ):
        raise ValueError("fallback-v8 stop intent resource ownership changed")
    return value, manifest, file_sha256


def _lease_identity(
    lease: Mapping[str, Any],
    *,
    configuration: VLLMLaunchConfiguration,
    incident: FallbackV8RuntimeIncident,
) -> _LeaseIdentity:
    group = incident.manual_recovery.process_group_id
    command_sha256 = canonical_sha256(list(configuration.command()))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": configuration.configuration_hash,
        "session_id": FALLBACK_V8_RUN_ID,
        "accounting_session_id": FALLBACK_V8_SERVICE_EVENT_ID,
        "service_pid": group,
        "process_command_sha256": command_sha256,
        "process_group_id": group,
        "process_session_id": group,
        "launch_protocol": DURABLE_EXEC_GATE_PROTOCOL,
    }
    if any(lease.get(name) != value for name, value in expected.items()):
        raise ValueError("fallback-v8 lease identity differs from its incident and configuration")
    controller_pid = lease.get("controller_pid")
    start_ticks = lease.get("process_start_ticks")
    token_sha256 = lease.get("service_instance_token_sha256")
    launch_token_sha256 = lease.get("launch_gate_token_sha256")
    supervisor_sha256 = lease.get("launch_supervisor_command_sha256")
    if (
        isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
        or not _is_canonical_sha256(token_sha256)
        or not _is_canonical_sha256(launch_token_sha256)
        or not _is_canonical_sha256(supervisor_sha256)
    ):
        raise ValueError("fallback-v8 lease process identity is incomplete")
    if (
        _parse_aware_timestamp(
            lease.get("service_started_at"),
            label="fallback-v8 lease service start",
        )
        != incident.accounting.service.started_at
        or _microseconds(
            lease.get("ledger_allocated_seconds_before_session"),
            label="fallback-v8 lease baseline",
        )
        != EXPECTED_BASELINE_MICROSECONDS
    ):
        raise ValueError("fallback-v8 lease accounting identity changed")
    _parse_aware_timestamp(lease.get("updated_at"), label="fallback-v8 lease update")
    return _LeaseIdentity(
        service_pid=group,
        process_start_ticks=start_ticks,
        process_command_sha256=command_sha256,
        process_group_id=group,
        process_session_id=group,
        service_instance_token_sha256=cast(str, token_sha256),
        launch_protocol=DURABLE_EXEC_GATE_PROTOCOL,
        launch_gate_token_sha256=cast(str, launch_token_sha256),
        launch_supervisor_command_sha256=cast(str, supervisor_sha256),
    )


def _validate_damaged_lease(
    lease: Mapping[str, Any],
    *,
    lease_manifest_sha256: str,
    lease_file_sha256: str,
    identity: _LeaseIdentity,
    incident: FallbackV8RuntimeIncident,
) -> None:
    if (
        lease.get("lease_state") != "shutdown_unverified"
        or lease_manifest_sha256
        != incident.manual_recovery.pre_stop_lease_manifest_sha256
        or lease_file_sha256 != incident.manual_recovery.pre_stop_lease_file_sha256
        or identity.process_start_ticks
        != incident.manual_recovery.process_members[0].start_ticks
    ):
        raise ValueError("service lease is not the exact incident-bound V8 damage state")
    ended_at = lease.get("service_ended_at")
    if ended_at is not None:
        parsed_end = _parse_aware_timestamp(ended_at, label="damaged V8 lease end")
        if parsed_end > incident.accounting.service.ended_at:
            raise ValueError("damaged V8 lease end exceeds the verified physical stop")
    observed = lease.get("observed_service_seconds")
    if observed is not None and _microseconds(observed, label="damaged V8 observed service") > (
        EXPECTED_SERVICE_MICROSECONDS
    ):
        raise ValueError("damaged V8 lease exceeds terminal service accounting")


def _process_identity_sha256(identity: _LeaseIdentity) -> str:
    return canonical_sha256(
        {
            "service_pid": identity.service_pid,
            "process_start_ticks": identity.process_start_ticks,
            "process_command_sha256": identity.process_command_sha256,
            "process_group_id": identity.process_group_id,
            "process_session_id": identity.process_session_id,
            "service_instance_token_sha256": identity.service_instance_token_sha256,
            "launch_protocol": identity.launch_protocol,
            "launch_gate_token_sha256": identity.launch_gate_token_sha256,
            "launch_supervisor_command_sha256": identity.launch_supervisor_command_sha256,
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


def _expected_service_payload(incident: FallbackV8RuntimeIncident) -> dict[str, object]:
    service = incident.accounting.service
    return {
        "service_session_id": service.service_session_id,
        "session_id": service.session_id,
        "service_microseconds": service.service_microseconds,
        "classified_event_microseconds": service.classified_event_microseconds,
        "overhead_microseconds": service.overhead_microseconds,
        "started_at": service.started_at,
        "ended_at": service.ended_at,
    }


def _validate_service_record(
    record: GpuServiceSession,
    incident: FallbackV8RuntimeIncident,
) -> None:
    if _service_record_payload(record) != _expected_service_payload(incident):
        raise ValueError("restored lease service row differs from the terminal V8 incident")


def _validate_restored_lease(
    lease: Mapping[str, Any],
    *,
    configuration: VLLMLaunchConfiguration,
    incident: FallbackV8RuntimeIncident,
    identity: _LeaseIdentity,
) -> datetime:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "configuration_hash": configuration.configuration_hash,
        "lease_state": "stopped_verified",
        "session_id": FALLBACK_V8_RUN_ID,
        "accounting_session_id": FALLBACK_V8_SERVICE_EVENT_ID,
        "service_pid": identity.service_pid,
        "process_start_ticks": identity.process_start_ticks,
        "process_command_sha256": identity.process_command_sha256,
        "process_group_id": identity.process_group_id,
        "process_session_id": identity.process_session_id,
        "service_instance_token_sha256": identity.service_instance_token_sha256,
        "launch_protocol": identity.launch_protocol,
        "launch_gate_token_sha256": identity.launch_gate_token_sha256,
        "launch_supervisor_command_sha256": identity.launch_supervisor_command_sha256,
        "service_started_at": incident.accounting.service.started_at.isoformat(),
        "service_ended_at": incident.accounting.service.ended_at.isoformat(),
        "ledger_allocated_seconds_before_session": EXPECTED_BASELINE_MICROSECONDS / 1_000_000,
        "observed_service_seconds": EXPECTED_SERVICE_MICROSECONDS / 1_000_000,
    }
    if any(lease.get(name) != value for name, value in expected.items()):
        raise ValueError("restored V8 lease differs from its terminal incident identity")
    controller_pid = lease.get("controller_pid")
    if (
        isinstance(controller_pid, bool)
        or not isinstance(controller_pid, int)
        or controller_pid <= 0
    ):
        raise ValueError("restored V8 lease controller identity is invalid")
    repaired_at = _parse_aware_timestamp(lease.get("updated_at"), label="restored V8 update")
    if repaired_at < incident.accounting.service.ended_at:
        raise ValueError("restored V8 lease predates the verified stop")
    return repaired_at


def _require_terminal_absence(service: VLLMService, identity: _LeaseIdentity) -> None:
    try:
        pid_live = service.process_liveness_check(identity.service_pid)
        group_live = service.process_group_liveness_check(identity.process_group_id)
        endpoint_live = service._endpoint_live(0.25)
    except BaseException as exc:
        raise ValueError("cannot revalidate terminal V8 service absence") from exc
    if pid_live or group_live or endpoint_live:
        raise ValueError("terminal V8 service or endpoint is live")


def _gpu_compute_process_count() -> int:
    import subprocess

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("cannot revalidate terminal GPU compute-process absence") from exc
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    for row in rows:
        fields = tuple(field.strip() for field in row.split(","))
        if len(fields) != 2 or not all(field.isdecimal() for field in fields):
            raise ValueError("GPU compute-process inventory is invalid")
    return len(rows)


def _require_gpu_compute_absence() -> None:
    if _gpu_compute_process_count() != 0:
        raise ValueError("terminal V8 GPU compute process is live")


@contextmanager
def _terminal_ledger_shadow(
    ledger_path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> Iterator[Ledger]:
    """Open an exact scratch copy so lease restoration cannot mutate the ledger."""

    source_bytes = ledger_path.read_bytes()
    if (
        len(source_bytes) != expected_size_bytes
        or hashlib.sha256(source_bytes).hexdigest() != expected_sha256
    ):
        raise ValueError("terminal V8 ledger changed before scratch isolation")
    try:
        with tempfile.TemporaryDirectory(prefix="story-projection-v8-ledger-") as directory:
            shadow = Path(directory) / "terminal.sqlite"
            shadow.write_bytes(source_bytes)
            shadow.chmod(0o600)
            shadow_sha256 = _sha256_file(shadow)
            try:
                with Ledger(shadow) as ledger_store:
                    yield ledger_store
            finally:
                if (
                    shadow.stat().st_size != expected_size_bytes
                    or _sha256_file(shadow) != shadow_sha256
                ):
                    raise RuntimeError("V8 lease restoration attempted to change ledger contents")
    finally:
        if (
            ledger_path.stat().st_size != expected_size_bytes
            or _sha256_file(ledger_path) != expected_sha256
        ):
            raise RuntimeError("terminal V8 ledger changed during lease restoration")


def _validate_access_control(
    *,
    root: Path,
    cache: Path,
    output_parent: Path,
    lease_path: Path,
    lock_path: Path,
    ledger_path: Path,
    output_path: Path,
) -> FallbackV8LeaseRepairAccessControl:
    proof = _validate_v7_private_access_control(
        root=root,
        cache=cache,
        output_parent=output_parent,
        lease_path=lease_path,
        lock_path=lock_path,
        ledger_path=ledger_path,
        output_path=output_path,
    )
    return FallbackV8LeaseRepairAccessControl.model_validate(proof.model_dump(mode="python"))


def _receipt_payload(
    *,
    incident: FallbackV8RuntimeIncident,
    incident_file_sha256: str,
    stop_intent_manifest_sha256: str,
    stop_intent_file_sha256: str,
    snapshot_manifest: Mapping[str, object],
    snapshot_manifest_file_sha256: str,
    configuration: VLLMLaunchConfiguration,
    access_control: FallbackV8LeaseRepairAccessControl,
    identity: _LeaseIdentity,
    ledger_size_bytes: int,
    ledger_sha256: str,
    damaged_lease_manifest_sha256: str,
    damaged_lease_file_sha256: str,
    restored_lease_manifest_sha256: str,
    restored_lease_file_sha256: str,
    repaired_at: datetime,
) -> dict[str, object]:
    configuration_manifest = configuration.public_manifest()
    service_payload = _expected_service_payload(incident)
    return {
        "schema_version": "1.0.0",
        "kind": FALLBACK_V8_LEASE_REPAIR_KIND,
        "run_id": FALLBACK_V8_RUN_ID,
        "source_incident_kind": FALLBACK_V8_RUNTIME_INCIDENT_KIND,
        "source_incident_manifest_sha256": incident.manifest_sha256,
        "source_incident_file_sha256": incident_file_sha256,
        "stop_intent_manifest_sha256": stop_intent_manifest_sha256,
        "stop_intent_file_sha256": stop_intent_file_sha256,
        "model_repository": configuration.repository,
        "model_revision": configuration.revision,
        "served_model_name": configuration.served_model_name,
        "verified_snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "verified_snapshot_manifest_file_sha256": snapshot_manifest_file_sha256,
        "configuration_hash": configuration.configuration_hash,
        "configuration_manifest_sha256": configuration_manifest["manifest_sha256"],
        "access_control": access_control,
        "service_session_id": FALLBACK_V8_RUN_ID,
        "service_event_id": FALLBACK_V8_SERVICE_EVENT_ID,
        "process_identity_sha256": _process_identity_sha256(identity),
        "terminal_service_record_sha256": canonical_sha256(service_payload),
        "service_microseconds": EXPECTED_SERVICE_MICROSECONDS,
        "classified_event_microseconds": EXPECTED_CLASSIFIED_MICROSECONDS,
        "overhead_microseconds": EXPECTED_OVERHEAD_MICROSECONDS,
        "terminal_allocated_microseconds": EXPECTED_TERMINAL_MICROSECONDS,
        "ledger_size_bytes": ledger_size_bytes,
        "ledger_file_sha256_before": ledger_sha256,
        "ledger_file_sha256_after": ledger_sha256,
        "ledger_bytes_unchanged": True,
        "gpu_events_added": 0,
        "accounting_rows_added": 0,
        "inference_attempts_added": 0,
        "inference_calls_added": 0,
        "damaged_lease_manifest_sha256": damaged_lease_manifest_sha256,
        "damaged_lease_file_sha256": damaged_lease_file_sha256,
        "restored_lease_manifest_sha256": restored_lease_manifest_sha256,
        "restored_lease_file_sha256": restored_lease_file_sha256,
        "restored_lease_state": "stopped_verified",
        "terminal_process_absence_revalidated": True,
        "terminal_gpu_compute_absence_revalidated": True,
        "repaired_at": repaired_at,
    }


def _load_receipt(path: Path) -> FallbackV8LeaseRepairReceipt:
    resolved = _require_regular_file(path, label="V8 lease repair receipt")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
        return FallbackV8LeaseRepairReceipt.model_validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError("V8 lease repair receipt contract is invalid") from exc


def load_fallback_v8_lease_repair_receipt(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_file_sha256: str,
) -> FallbackV8LeaseRepairReceipt:
    """Load only the receipt bound by both registered logical and file hashes."""

    if not _is_canonical_sha256(expected_manifest_sha256):
        raise ValueError("expected V8 lease repair receipt manifest hash is invalid")
    if not _is_canonical_sha256(expected_file_sha256):
        raise ValueError("expected V8 lease repair receipt file hash is invalid")
    resolved = _require_regular_file(path, label="V8 lease repair receipt")
    if _sha256_file(resolved) != expected_file_sha256:
        raise ValueError("V8 lease repair receipt file hash changed")
    receipt = _load_receipt(resolved)
    if receipt.manifest_sha256 != expected_manifest_sha256:
        raise ValueError("V8 lease repair receipt manifest hash changed")
    return receipt


def _write_receipt(path: Path, receipt: FallbackV8LeaseRepairReceipt) -> None:
    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="V8 lease repair receipt output")
    payload = (canonical_json(receipt) + "\n").encode("utf-8")
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError("append-only V8 lease repair receipt already differs")
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


def _validate_receipt_identity(
    receipt: FallbackV8LeaseRepairReceipt,
    *,
    incident: FallbackV8RuntimeIncident,
    incident_file_sha256: str,
    stop_intent_manifest_sha256: str,
    stop_intent_file_sha256: str,
    snapshot_manifest: Mapping[str, object],
    snapshot_manifest_file_sha256: str,
    configuration: VLLMLaunchConfiguration,
    access_control: FallbackV8LeaseRepairAccessControl,
    identity: _LeaseIdentity,
    ledger_size_bytes: int,
    ledger_sha256: str,
    restored_lease_manifest_sha256: str,
    restored_lease_file_sha256: str,
    repaired_at: datetime,
) -> None:
    payload = _receipt_payload(
        incident=incident,
        incident_file_sha256=incident_file_sha256,
        stop_intent_manifest_sha256=stop_intent_manifest_sha256,
        stop_intent_file_sha256=stop_intent_file_sha256,
        snapshot_manifest=snapshot_manifest,
        snapshot_manifest_file_sha256=snapshot_manifest_file_sha256,
        configuration=configuration,
        access_control=access_control,
        identity=identity,
        ledger_size_bytes=ledger_size_bytes,
        ledger_sha256=ledger_sha256,
        damaged_lease_manifest_sha256=receipt.damaged_lease_manifest_sha256,
        damaged_lease_file_sha256=receipt.damaged_lease_file_sha256,
        restored_lease_manifest_sha256=restored_lease_manifest_sha256,
        restored_lease_file_sha256=restored_lease_file_sha256,
        repaired_at=repaired_at,
    )
    expected = FallbackV8LeaseRepairReceipt.model_validate(
        {**payload, "manifest_sha256": canonical_sha256(payload)}
    )
    if receipt != expected:
        raise FileExistsError("existing V8 lease repair receipt differs from current state")


def restore_fallback_v8_terminal_lease(
    *,
    project_root: Path,
    incident_path: Path,
    stop_intent_path: Path,
    snapshot_path: Path,
    shared_cache: Path,
    verified_snapshot_manifest_path: Path,
    ledger_path: Path,
    output_path: Path,
    port: int,
) -> FallbackV8LeaseRepairReceipt:
    """Restore only V8's incident-bound terminal lease without GPU allocation."""

    root = _require_directory(project_root, label="project root")
    incident_resolved = _require_regular_file(incident_path, label="fallback-v8 incident")
    incident = load_fallback_v8_runtime_incident(incident_resolved)
    if incident.run_id != FALLBACK_V8_RUN_ID:
        raise ValueError("lease repair is restricted to the final fallback-v8 incident")
    intent_resolved = _require_regular_file(stop_intent_path, label="fallback-v8 stop intent")
    _stop_intent, intent_manifest_sha256, intent_file_sha256 = (
        _load_and_validate_stop_intent(intent_resolved, incident=incident)
    )

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
    if (
        ledger_sha256 != incident.accounting.terminal.file_sha256
        or ledger_size_bytes != incident.accounting.terminal.size_bytes
        or incident.accounting.terminal.summary.total_allocated_microseconds
        != EXPECTED_TERMINAL_MICROSECONDS
        or incident.accounting.terminal.summary.unresolved_gpu_allocation_count != 0
        or incident.accounting.terminal.summary.unresolved_gpu_service_count != 0
    ):
        raise ValueError("current terminal ledger differs from the V8 incident")

    output, output_parent = _require_exact_layout(
        root=root,
        incident=incident_resolved,
        stop_intent=intent_resolved,
        snapshot=snapshot,
        cache=cache,
        snapshot_manifest=snapshot_manifest_resolved,
        ledger=ledger,
        output=output_path,
    )
    incident_file_sha256 = _sha256_file(incident_resolved)
    snapshot_manifest_file_sha256 = _sha256_file(snapshot_manifest_resolved)
    lease_path = cache / SERVICE_LEASE_SNAPSHOT_FILENAME
    lock_path = _require_regular_file(cache / SERVICE_LOCK_FILENAME, label="vLLM service lock")
    lease, lease_manifest_sha256, lease_file_sha256 = _load_lease(
        lease_path,
        label="current V8 service lease",
    )
    identity = _lease_identity(
        lease,
        configuration=configuration,
        incident=incident,
    )
    if (
        identity.process_start_ticks
        != incident.manual_recovery.process_members[0].start_ticks
    ):
        raise ValueError("current V8 lease and stop intent process identity differ")
    access_control = _validate_access_control(
        root=root,
        cache=cache,
        output_parent=output_parent,
        lease_path=lease_path,
        lock_path=lock_path,
        ledger_path=ledger,
        output_path=output,
    )

    if output.exists():
        try:
            receipt = _load_receipt(output)
        except (ValueError, OSError) as exc:
            raise FileExistsError("existing V8 lease repair receipt is invalid") from exc
        if (
            lease_manifest_sha256 != receipt.restored_lease_manifest_sha256
            or lease_file_sha256 != receipt.restored_lease_file_sha256
        ):
            raise FileExistsError("existing V8 lease repair receipt differs from current lease")
        repaired_at = _validate_restored_lease(
            lease,
            configuration=configuration,
            incident=incident,
            identity=identity,
        )
        with _terminal_ledger_shadow(
            ledger,
            expected_sha256=ledger_sha256,
            expected_size_bytes=ledger_size_bytes,
        ) as ledger_store:
            meter = AllocatedGPUMeter(ledger_store)
            service = VLLMService(
                configuration=configuration,
                client=VLLMGuidedJSONClient(configuration.base_url),
                meter=meter,
            )
            _require_terminal_absence(service, identity)
        _require_gpu_compute_absence()
        if _sha256_file(ledger) != ledger_sha256 or ledger.stat().st_size != ledger_size_bytes:
            raise RuntimeError("V8 lease repair replay changed terminal ledger bytes")
        _validate_receipt_identity(
            receipt,
            incident=incident,
            incident_file_sha256=incident_file_sha256,
            stop_intent_manifest_sha256=intent_manifest_sha256,
            stop_intent_file_sha256=intent_file_sha256,
            snapshot_manifest=snapshot_manifest,
            snapshot_manifest_file_sha256=snapshot_manifest_file_sha256,
            configuration=configuration,
            access_control=access_control,
            identity=identity,
            ledger_size_bytes=ledger_size_bytes,
            ledger_sha256=ledger_sha256,
            restored_lease_manifest_sha256=lease_manifest_sha256,
            restored_lease_file_sha256=lease_file_sha256,
            repaired_at=repaired_at,
        )
        return receipt

    record: GpuServiceSession
    if lease.get("lease_state") == "shutdown_unverified":
        _validate_damaged_lease(
            lease,
            lease_manifest_sha256=lease_manifest_sha256,
            lease_file_sha256=lease_file_sha256,
            identity=identity,
            incident=incident,
        )
        _require_gpu_compute_absence()
        with _terminal_ledger_shadow(
            ledger,
            expected_sha256=ledger_sha256,
            expected_size_bytes=ledger_size_bytes,
        ) as ledger_store:
            meter = AllocatedGPUMeter(ledger_store)
            service = VLLMService(
                configuration=configuration,
                client=VLLMGuidedJSONClient(configuration.base_url),
                meter=meter,
            )
            _require_terminal_absence(service, identity)
            record = service.recover_stale_service_lease(
                expected_current_lease_manifest_sha256=lease_manifest_sha256,
            )
            if record is None:
                raise RuntimeError("V8 terminal lease recovery returned no service record")
            _validate_service_record(record, incident)
        damaged_manifest_sha256 = lease_manifest_sha256
        damaged_file_sha256 = lease_file_sha256
    elif lease.get("lease_state") == "stopped_verified":
        # Without the append-only receipt there is no independent binding for
        # the post-transition token and durable-launch identity.  Treating the
        # current lease as its own expected identity would let a validly rehashed
        # substitute masquerade as a crash after the atomic lease rewrite.
        raise ValueError(
            "unreceipted stopped V8 lease has no independent post-transition binding"
        )
    else:
        raise ValueError("current V8 lease is not repairable")

    if _sha256_file(ledger) != ledger_sha256 or ledger.stat().st_size != ledger_size_bytes:
        raise RuntimeError("terminal V8 lease restoration changed ledger bytes")
    _require_gpu_compute_absence()
    restored_lease, restored_manifest_sha256, restored_file_sha256 = _load_lease(
        lease_path,
        label="restored V8 service lease",
    )
    restored_identity = _lease_identity(
        restored_lease,
        configuration=configuration,
        incident=incident,
    )
    if restored_identity != identity:
        raise ValueError("restored V8 lease process identity changed")
    repaired_at = _validate_restored_lease(
        restored_lease,
        configuration=configuration,
        incident=incident,
        identity=identity,
    )
    payload = _receipt_payload(
        incident=incident,
        incident_file_sha256=incident_file_sha256,
        stop_intent_manifest_sha256=intent_manifest_sha256,
        stop_intent_file_sha256=intent_file_sha256,
        snapshot_manifest=snapshot_manifest,
        snapshot_manifest_file_sha256=snapshot_manifest_file_sha256,
        configuration=configuration,
        access_control=access_control,
        identity=identity,
        ledger_size_bytes=ledger_size_bytes,
        ledger_sha256=ledger_sha256,
        damaged_lease_manifest_sha256=damaged_manifest_sha256,
        damaged_lease_file_sha256=damaged_file_sha256,
        restored_lease_manifest_sha256=restored_manifest_sha256,
        restored_lease_file_sha256=restored_file_sha256,
        repaired_at=repaired_at,
    )
    receipt = FallbackV8LeaseRepairReceipt.model_validate(
        {**payload, "manifest_sha256": canonical_sha256(payload)}
    )
    _write_receipt(output, receipt)
    final_access_control = _validate_access_control(
        root=root,
        cache=cache,
        output_parent=output_parent,
        lease_path=lease_path,
        lock_path=lock_path,
        ledger_path=ledger,
        output_path=output,
    )
    if final_access_control != access_control:
        raise RuntimeError("V8 lease repair access-control proof changed after receipt write")
    return _load_receipt(output)


__all__ = [
    "FALLBACK_V8_LEASE_REPAIR_KIND",
    "FallbackV8LeaseRepairAccessControl",
    "FallbackV8LeaseRepairReceipt",
    "load_fallback_v8_lease_repair_receipt",
    "restore_fallback_v8_terminal_lease",
]
