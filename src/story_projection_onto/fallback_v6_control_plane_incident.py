"""Typed public record for the fallback-v6 zero-GPU control-plane incident.

The record authenticates the exact restricted receipt/log set without exposing
command lines, absolute paths, or log text.  The v6 guardian failed during
CPU-only development-continuation construction, before publishing readiness or
starting a model service.  The two ledger snapshots differ by exactly one
storage-admission observation; every non-storage table is identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from story_projection_onto.contracts import GitRevision, Sha256Digest, canonical_json
from story_projection_onto.contracts import canonical_sha256 as _canonical_sha256
from story_projection_onto.fallback_control_plane_incident import (
    OpaqueFileBinding,
    SafeBasename,
    SelfHashedFileBinding,
)
from story_projection_onto.store import (
    GpuEventKind,
    GpuSummary,
    ReadOnlyLedger,
    StorageSampleRecord,
)

FALLBACK_V6_RUN_ID = "fallback-qwen3-8b-awq-development-v6"
FALLBACK_V6_SOURCE_REVISION = "fallback-second-recovery-v6"
FALLBACK_V6_CONTROL_PLANE_INCIDENT_KIND = (
    "fallback_gpu_acceptance_v6_control_plane_incident"
)
FALLBACK_V6_CONTROL_PLANE_ERROR_CLASS = (
    "development_continuation_service_identity_mismatch_before_guardian_readiness"
)
FALLBACK_V6_PREFLIGHT_MANIFEST_SHA256 = (
    "f8a05eeacf706e53b0d9d88aee6768230dfe741b808ab3a4e39628bef953db49"
)
FALLBACK_V4_INCIDENT_MANIFEST_SHA256 = (
    "e8b30397068a97f4f169395ec0a70ec1f709515c1df07b6a17400c7568c2993c"
)
FALLBACK_V5_INCIDENT_MANIFEST_SHA256 = (
    "06b7bf28427266efa9ebae3640a8a4fe883b0233956d313d98fb9103775dbfdf"
)
FROZEN_RECOVERY_SERVICE_EVENT_IDS = (
    "fallback-qwen3-8b-awq-development-v3-service-start-001",
    "fallback-qwen3-8b-awq-development-v5-service-start-001",
)
DERIVED_V6_RECOVERY_SERVICE_EVENT_IDS = (
    "fallback-qwen3-8b-awq-development-v3-service-start-001",
    "fallback-qwen3-8b-awq-development-v6-service-start-001",
)

EXPECTED_V6_RUN_ROOT_BASENAMES = (
    "checkpoint.json.guardian-ticket.json",
    "checkpoint.json.guardian.log",
    "checkpoint.json.orchestrator-guard-000001.json",
    "checkpoint.json.orchestrator-invocation.json",
    "orchestrator.20260905T170000Z.log",
    "status.after-failure.json",
)
EXPECTED_V6_ABSENT_BASENAMES = (
    ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
    "checkpoint.json",
    "checkpoint.json.guardian-controller-takeover.json",
    "checkpoint.json.guardian-ready-000001.json",
    "checkpoint.json.guardian-result.json",
    "checkpoint.json.guardian-terminal-request.json",
    "checkpoint.json.guardian.lock",
    "checkpoint.json.service",
    "fallback-qwen3-8b-awq-development-v6.vllm.log",
    "fallback_gpu_acceptance_development_v6.json",
    "fallback_gpu_acceptance_development_v6.json.controller-handoff.json",
    "fallback_gpu_acceptance_development_v6.json.orphan-cleanup.json",
)


class _StrictIncidentRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class FallbackV6SourceAndAuthorization(_StrictIncidentRecord):
    source_revision: Literal["fallback-second-recovery-v6"]
    source_git_commit: GitRevision
    source_tree_sha256: Sha256Digest
    source_association: SelfHashedFileBinding
    authorization_overlay: SelfHashedFileBinding
    execution_preflight: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_names_and_kinds(self) -> Self:
        expected = (
            (
                self.source_association,
                "source_tree_fallback_second_recovery_v6.association.json",
                "local_remote_source_tree_association",
            ),
            (
                self.authorization_overlay,
                "fallback-second-recovery-v6.authorized.json",
                "phase1_fallback_second_recovery_overlay",
            ),
            (
                self.execution_preflight,
                "fallback_gpu_acceptance_development_v6.preflight.json",
                "phase1_fallback_execution_preflight",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected
        ):
            raise ValueError("v6 source, overlay, or preflight identity changed")
        return self


class FallbackV6RestrictedEvidence(_StrictIncidentRecord):
    orchestration_invocation: SelfHashedFileBinding
    guardian_ticket: SelfHashedFileBinding
    orchestrator_guard: SelfHashedFileBinding
    post_failure_status: SelfHashedFileBinding
    guardian_log: OpaqueFileBinding
    orchestrator_log: OpaqueFileBinding
    run_root_inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def require_exact_names_and_kinds(self) -> Self:
        expected_json = (
            (
                self.orchestration_invocation,
                "checkpoint.json.orchestrator-invocation.json",
                "fallback_controller_orchestration_invocation",
            ),
            (
                self.guardian_ticket,
                "checkpoint.json.guardian-ticket.json",
                "fallback_service_guardian_ticket",
            ),
            (
                self.orchestrator_guard,
                "checkpoint.json.orchestrator-guard-000001.json",
                "fallback_controller_orchestrator_guard",
            ),
            (
                self.post_failure_status,
                "status.after-failure.json",
                "fallback_controller_orchestration_status",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected_json
        ):
            raise ValueError("v6 restricted JSON identity changed")
        if (
            self.guardian_log.basename != "checkpoint.json.guardian.log"
            or self.orchestrator_log.basename != "orchestrator.20260905T170000Z.log"
        ):
            raise ValueError("v6 restricted diagnostic identity changed")
        return self


class FallbackV6OrchestrationIdentity(_StrictIncidentRecord):
    execution_arguments_sha256: Sha256Digest
    orchestration_invocation_manifest_sha256: Sha256Digest
    guardian_ticket_manifest_sha256: Sha256Digest
    orchestrator_guard_manifest_sha256: Sha256Digest
    post_failure_status_manifest_sha256: Sha256Digest
    guardian_command_sha256: Sha256Digest
    orchestrator_command_sha256: Sha256Digest
    service_session_id: Literal["fallback-qwen3-8b-awq-development-v6"]
    service_event_id: Literal[
        "fallback-qwen3-8b-awq-development-v6-service-start-001"
    ]


class FallbackV6LedgerSummary(_StrictIncidentRecord):
    total_allocated_microseconds: int = Field(ge=0, strict=True)
    event_count: int = Field(ge=0, strict=True)
    service_session_count: int = Field(ge=0, strict=True)
    service_start_event_count: int = Field(ge=0, strict=True)
    attempt_count: int = Field(ge=0, strict=True)
    model_call_count: int = Field(ge=0, strict=True)
    artifact_count: int = Field(ge=0, strict=True)
    storage_sample_count: int = Field(ge=0, strict=True)
    unresolved_gpu_allocation_count: int = Field(ge=0, strict=True)
    unresolved_gpu_service_count: int = Field(ge=0, strict=True)
    by_kind_microseconds: dict[str, int]

    @model_validator(mode="after")
    def reconcile_gpu_total(self) -> Self:
        if any(
            not key or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.by_kind_microseconds.items()
        ):
            raise ValueError("GPU accounting kinds must be nonnegative integers")
        if self.total_allocated_microseconds != sum(self.by_kind_microseconds.values()):
            raise ValueError("GPU accounting kinds do not reconcile")
        return self


class FallbackV6LedgerBinding(_StrictIncidentRecord):
    basename: SafeBasename
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest
    schema_sha256: Sha256Digest
    non_storage_content_sha256: Sha256Digest
    non_storage_table_count: int = Field(gt=0, strict=True)
    summary: FallbackV6LedgerSummary


class FallbackV6StorageSample(_StrictIncidentRecord):
    sample_id: Sha256Digest
    phase: Literal["phase1_fallback:fallback-qwen3-8b-awq-development-v6"]
    sampled_at: AwareDatetime
    current_occupied_bytes: int = Field(ge=0, strict=True)
    additional_reserved_bytes: int = Field(ge=0, strict=True)
    projected_occupied_bytes: int = Field(ge=0, strict=True)
    filesystem_free_bytes: int = Field(ge=0, strict=True)
    effective_projected_headroom_bytes: int = Field(ge=0, strict=True)
    allowed: Literal[True]
    violations: tuple[str, ...]

    @model_validator(mode="after")
    def require_allowed_exact_arithmetic(self) -> Self:
        if self.violations:
            raise ValueError("v6 storage admission must have no violations")
        if self.projected_occupied_bytes != (
            self.current_occupied_bytes + self.additional_reserved_bytes
        ):
            raise ValueError("v6 storage admission arithmetic changed")
        return self


class FallbackV6ZeroGpuDelta(_StrictIncidentRecord):
    allocated_gpu_microseconds: Literal[0]
    gpu_events: Literal[0]
    service_sessions: Literal[0]
    service_start_events: Literal[0]
    attempts: Literal[0]
    inference_model_calls: Literal[0]
    artifacts: Literal[0]
    storage_samples: Literal[1]
    unresolved_gpu_allocations: Literal[0]
    unresolved_gpu_services: Literal[0]
    accepted_outputs: Literal[0]
    guardian_process_launch_attempts: Literal[1]
    internal_controller_launch_attempts: Literal[0]
    authorized_retry_inference_consumed: Literal[False]
    authorized_service_start_consumed: Literal[False]


class FallbackV6Accounting(_StrictIncidentRecord):
    before: FallbackV6LedgerBinding
    after: FallbackV6LedgerBinding
    appended_storage_sample: FallbackV6StorageSample
    prior_storage_samples_preserved: Literal[True]
    all_non_storage_tables_identical: Literal[True]
    delta: FallbackV6ZeroGpuDelta

    @model_validator(mode="after")
    def require_one_storage_only_delta(self) -> Self:
        if (
            self.before.basename != "phase1_acceptance.before-v6.sqlite"
            or self.after.basename != "phase1_acceptance.after-v6.sqlite"
            or self.before.file_sha256 == self.after.file_sha256
            or self.before.schema_sha256 != self.after.schema_sha256
            or self.before.non_storage_table_count
            != self.after.non_storage_table_count
            or self.before.non_storage_content_sha256
            != self.after.non_storage_content_sha256
        ):
            raise ValueError("v6 ledger identity or non-storage content changed")
        before = self.before.summary.model_dump(mode="python")
        expected_after = {**before, "storage_sample_count": before["storage_sample_count"] + 1}
        if self.after.summary.model_dump(mode="python") != expected_after:
            raise ValueError("v6 ledger changed beyond one storage observation")
        return self


class FallbackV6AbsenceInventory(_StrictIncidentRecord):
    observed_run_root_basenames: tuple[SafeBasename, ...]
    absent_basenames: tuple[SafeBasename, ...]
    guardian_ready_receipt_count: Literal[0]
    internal_controller_receipt_count: Literal[0]
    all_required_paths_absent: Literal[True]

    @model_validator(mode="after")
    def require_exact_inventories(self) -> Self:
        if self.observed_run_root_basenames != EXPECTED_V6_RUN_ROOT_BASENAMES:
            raise ValueError("v6 run-root inventory changed")
        if self.absent_basenames != EXPECTED_V6_ABSENT_BASENAMES:
            raise ValueError("v6 absence inventory changed")
        return self


class FallbackV6TerminalState(_StrictIncidentRecord):
    guardian_ready: Literal[False]
    guardian_terminal_receipt_present: Literal[False]
    guardian_live: Literal[False]
    latest_orchestrator_live: Literal[False]
    latest_control_group_live: Literal[False]
    live_internal_controller_count: Literal[0]
    checkpoint_present: Literal[False]
    controller_handoff_present: Literal[False]
    cleanup_present: Literal[False]
    public_result_present: Literal[False]
    service_start_attempted: Literal[None]
    model_process_started: Literal[False]
    vllm_service_started: Literal[False]
    gpu_allocation_performed: Literal[False]
    inference_attempt_count: Literal[0]
    accepted_output_count: Literal[0]
    unresolved_gpu_allocation_count: Literal[0]
    unresolved_gpu_service_count: Literal[0]
    ledger_snapshot_verified: Literal[True]
    resume_allowed: Literal[False]
    source_bound_v6_resume_permitted: Literal[False]
    fresh_repaired_source_required: Literal[True]


class FallbackV6RootCause(_StrictIncidentRecord):
    safe_error_class: Literal[
        "development_continuation_service_identity_mismatch_before_guardian_readiness"
    ]
    guardian_exception_type: Literal["DevelopmentContinuationError"]
    outer_exception_type: Literal["RuntimeError"]
    failure_stage: Literal["guardian_construction_before_readiness"]
    guardian_exit_code: Literal[1]
    frozen_expected_service_event_ids: tuple[str, str]
    factory_derived_service_event_ids: tuple[str, str]
    mismatch_position: Literal[1]
    scientific_generation_reached: Literal[False]
    command_line_public: Literal[False]
    remote_absolute_paths_public: Literal[False]
    full_log_text_public: Literal[False]

    @model_validator(mode="after")
    def require_exact_identity_mismatch(self) -> Self:
        if (
            self.frozen_expected_service_event_ids != FROZEN_RECOVERY_SERVICE_EVENT_IDS
            or self.factory_derived_service_event_ids
            != DERIVED_V6_RECOVERY_SERVICE_EVENT_IDS
            or self.frozen_expected_service_event_ids[0]
            != self.factory_derived_service_event_ids[0]
            or self.frozen_expected_service_event_ids[1]
            == self.factory_derived_service_event_ids[1]
        ):
            raise ValueError("v6 recovery service identity mismatch changed")
        return self


class FallbackV6ControlPlaneIncident(_StrictIncidentRecord):
    """Self-hashed public description of the exact fallback-v6 incident."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_gpu_acceptance_v6_control_plane_incident"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v6"]
    audited_at: AwareDatetime
    source_and_authorization: FallbackV6SourceAndAuthorization
    restricted_evidence: FallbackV6RestrictedEvidence
    orchestration_identity: FallbackV6OrchestrationIdentity
    accounting: FallbackV6Accounting
    absence_inventory: FallbackV6AbsenceInventory
    terminal_state: FallbackV6TerminalState
    root_cause: FallbackV6RootCause
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_cross_record_identity_and_manifest(self) -> Self:
        evidence = self.restricted_evidence
        identity = self.orchestration_identity
        expected = (
            (
                evidence.orchestration_invocation.manifest_sha256,
                identity.orchestration_invocation_manifest_sha256,
            ),
            (
                evidence.guardian_ticket.manifest_sha256,
                identity.guardian_ticket_manifest_sha256,
            ),
            (
                evidence.orchestrator_guard.manifest_sha256,
                identity.orchestrator_guard_manifest_sha256,
            ),
            (
                evidence.post_failure_status.manifest_sha256,
                identity.post_failure_status_manifest_sha256,
            ),
        )
        if any(observed != required for observed, required in expected):
            raise ValueError("v6 restricted receipts disagree with orchestration identity")
        if self.accounting.before.summary.total_allocated_microseconds != 815_215_409:
            raise ValueError("v6 incident changed cumulative GPU allocation")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _canonical_sha256(immutable):
            raise ValueError("fallback-v6 incident manifest hash changed")
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


def _load_self_hashed_json(
    path: Path,
    *,
    label: str,
    expected_kind: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = _require_regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    supplied_hash = value.get("manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if (
        value.get("kind") != expected_kind
        or not isinstance(supplied_hash, str)
        or supplied_hash != _canonical_sha256(immutable)
    ):
        raise ValueError(f"{label} has an invalid kind or canonical self-hash")
    return resolved, value


def _self_hashed_binding(path: Path, value: Mapping[str, Any]) -> SelfHashedFileBinding:
    return SelfHashedFileBinding(
        basename=path.name,
        size_bytes=path.stat().st_size,
        file_sha256=_sha256_file(path),
        kind=value["kind"],
        manifest_sha256=value["manifest_sha256"],
    )


def _opaque_binding(path: Path, *, label: str) -> OpaqueFileBinding:
    resolved = _require_regular_file(path, label=label)
    if resolved.stat().st_size <= 0:
        raise ValueError(f"{label} cannot be empty")
    return OpaqueFileBinding(
        basename=resolved.name,
        size_bytes=resolved.stat().st_size,
        file_sha256=_sha256_file(resolved),
    )


def _summary_payload(summary: GpuSummary, ledger: ReadOnlyLedger) -> dict[str, object]:
    return {
        "total_allocated_microseconds": summary.total_allocated_microseconds,
        "event_count": summary.event_count,
        "service_session_count": summary.service_session_count,
        "service_start_event_count": sum(
            1 for event in ledger.gpu_events() if event.event_kind is GpuEventKind.GPU_SESSION_START
        ),
        "attempt_count": ledger.count_rows("attempts"),
        "model_call_count": ledger.count_rows("model_calls"),
        "artifact_count": ledger.count_rows("artifacts"),
        "storage_sample_count": ledger.count_rows("storage_samples"),
        "unresolved_gpu_allocation_count": len(ledger.unresolved_gpu_allocations()),
        "unresolved_gpu_service_count": len(ledger.unresolved_gpu_service_journals()),
        "by_kind_microseconds": {
            kind.value: value for kind, value in summary.by_kind_microseconds
        },
    }


def _typed_sql_value(value: object) -> dict[str, object]:
    if value is None:
        return {"type": "null", "value": None}
    if isinstance(value, bytes):
        return {"type": "blob", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "text", "value": value}
    if isinstance(value, int):
        return {"type": "integer", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ledger contains a non-finite floating-point value")
        return {"type": "real", "value": value.hex()}
    raise ValueError("ledger contains an unsupported SQLite value")


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _ledger_content_fingerprints(path: Path) -> tuple[str, str, int]:
    """Hash schema and every non-storage row without publishing row contents."""

    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        schema_payload = [
            {
                "type": row[0],
                "name": row[1],
                "table": row[2],
                "sql": row[3],
            }
            for row in objects
        ]
        table_names = sorted(row[1] for row in objects if row[0] == "table")
        non_storage_payload: list[dict[str, object]] = []
        for table_name in table_names:
            if table_name == "storage_samples":
                continue
            columns = [
                row[1]
                for row in connection.execute(
                    f"PRAGMA table_info({_quoted_identifier(table_name)})"
                ).fetchall()
            ]
            encoded_rows = [
                [_typed_sql_value(value) for value in row]
                for row in connection.execute(
                    f"SELECT * FROM {_quoted_identifier(table_name)}"
                ).fetchall()
            ]
            encoded_rows.sort(key=canonical_json)
            non_storage_payload.append(
                {"table": table_name, "columns": columns, "rows": encoded_rows}
            )
        return (
            _canonical_sha256(schema_payload),
            _canonical_sha256(non_storage_payload),
            len(non_storage_payload),
        )
    finally:
        connection.close()


def _ledger_binding_and_samples(
    path: Path,
    *,
    label: str,
) -> tuple[FallbackV6LedgerBinding, dict[str, StorageSampleRecord]]:
    resolved = _require_regular_file(path, label=label)
    with ReadOnlyLedger(resolved) as ledger:
        summary = FallbackV6LedgerSummary.model_validate(
            _summary_payload(ledger.gpu_summary(), ledger)
        )
        samples = {sample.sample_id: sample for sample in ledger.storage_samples()}
    schema_hash, non_storage_hash, non_storage_count = _ledger_content_fingerprints(
        resolved
    )
    return (
        FallbackV6LedgerBinding(
            basename=resolved.name,
            size_bytes=resolved.stat().st_size,
            file_sha256=_sha256_file(resolved),
            schema_sha256=schema_hash,
            non_storage_content_sha256=non_storage_hash,
            non_storage_table_count=non_storage_count,
            summary=summary,
        ),
        samples,
    )


def _typed_storage_sample(sample: StorageSampleRecord) -> FallbackV6StorageSample:
    return FallbackV6StorageSample(
        sample_id=sample.sample_id,
        phase=sample.phase,
        sampled_at=sample.sampled_at,
        current_occupied_bytes=sample.current_occupied_bytes,
        additional_reserved_bytes=sample.additional_reserved_bytes,
        projected_occupied_bytes=sample.projected_occupied_bytes,
        filesystem_free_bytes=sample.filesystem_free_bytes,
        effective_projected_headroom_bytes=sample.effective_projected_headroom_bytes,
        allowed=sample.allowed,
        violations=sample.violations,
    )


def _microseconds_from_seconds(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative number")
    microseconds = round(float(value) * 1_000_000)
    if abs(float(value) - microseconds / 1_000_000) > 1e-9:
        raise ValueError(f"{label} is not exact to microsecond precision")
    return microseconds


def _require_json_relationships(
    *,
    run_id: str,
    result_path: Path,
    checkpoint_path: Path,
    source_path: Path,
    source: Mapping[str, Any],
    overlay: Mapping[str, Any],
    preflight: Mapping[str, Any],
    invocation: Mapping[str, Any],
    ticket: Mapping[str, Any],
    guard: Mapping[str, Any],
    status: Mapping[str, Any],
    before: FallbackV6LedgerBinding,
) -> None:
    source_manifest = source.get("manifest_sha256")
    overlay_manifest = overlay.get("manifest_sha256")
    invocation_manifest = invocation.get("manifest_sha256")
    ticket_manifest = ticket.get("manifest_sha256")
    guard_manifest = guard.get("manifest_sha256")
    execution_arguments = invocation.get("execution_arguments_sha256")
    core_accounting = {
        "total_allocated_microseconds": before.summary.total_allocated_microseconds,
        "event_count": before.summary.event_count,
        "service_session_count": before.summary.service_session_count,
        "by_kind_microseconds": before.summary.by_kind_microseconds,
    }
    source_binding = overlay.get("source")
    authorization = overlay.get("authorization")
    v4_binding = overlay.get("intervening_control_plane_incident")
    v5_binding = overlay.get("intervening_v5_control_plane_incident")
    if (
        source.get("revision_label") != FALLBACK_V6_SOURCE_REVISION
        or source.get("local_tree_sha256") != source.get("remote_tree_sha256")
        or overlay.get("schema_version") != "1.4.0"
        or overlay.get("authorized_recovery_run_id") != run_id
        or not isinstance(authorization, Mapping)
        or authorization.get("status") != "authorized"
        or not isinstance(source_binding, Mapping)
        or source_binding.get("current_association_manifest_sha256") != source_manifest
        or source_binding.get("current_association_file_sha256")
        != _sha256_file(source_path)
        or source_binding.get("current_tree_sha256") != source.get("local_tree_sha256")
        or not isinstance(v4_binding, Mapping)
        or v4_binding.get("incident_manifest_sha256")
        != FALLBACK_V4_INCIDENT_MANIFEST_SHA256
        or not isinstance(v5_binding, Mapping)
        or v5_binding.get("incident_manifest_sha256")
        != FALLBACK_V5_INCIDENT_MANIFEST_SHA256
        or preflight.get("manifest_sha256") != FALLBACK_V6_PREFLIGHT_MANIFEST_SHA256
        or preflight.get("run_id") != run_id
        or preflight.get("execution_authorized") is not True
        or preflight.get("passed") is not True
        or preflight.get("gpu_allocation_performed") is not False
        or preflight.get("model_process_started") is not False
        or preflight.get("checkpoint_absent") is not True
        or preflight.get("source_association_sha256") != source_manifest
        or preflight.get("source_tree_sha256") != source.get("local_tree_sha256")
        or preflight.get("second_recovery_overlay_sha256") != overlay_manifest
        or preflight.get("prior_control_plane_incident_sha256")
        != FALLBACK_V4_INCIDENT_MANIFEST_SHA256
        or preflight.get("prior_v5_control_plane_incident_sha256")
        != FALLBACK_V5_INCIDENT_MANIFEST_SHA256
        or preflight.get("gpu_accounting_before_start") != core_accounting
        or overlay.get("cumulative_gpu_accounting") != core_accounting
    ):
        raise ValueError("v6 source, authorization, preflight, or ledger binding changed")
    if (
        invocation.get("run_id") != run_id
        or Path(str(invocation.get("result_output"))).name != result_path.name
        or Path(str(invocation.get("checkpoint"))).name != checkpoint_path.name
        or Path(str(invocation.get("checkpoint"))).parent.name != checkpoint_path.parent.name
        or invocation.get("service_session_id") != run_id
        or invocation.get("service_event_id") != f"{run_id}-service-start-001"
        or _microseconds_from_seconds(
            invocation.get("gpu_seconds_before_invocation"),
            label="invocation GPU accounting",
        )
        != before.summary.total_allocated_microseconds
        or ticket.get("run_id") != run_id
        or ticket.get("orchestration_invocation_sha256") != invocation_manifest
        or ticket.get("execution_arguments_sha256") != execution_arguments
        or ticket.get("service_session_id") != run_id
        or ticket.get("service_event_id") != f"{run_id}-service-start-001"
        or guard.get("run_id") != run_id
        or guard.get("sequence") != 1
        or guard.get("previous_guard_sha256") is not None
        or guard.get("state") != "active"
        or guard.get("orchestration_invocation_sha256") != invocation_manifest
        or guard.get("guardian_ticket_sha256") != ticket_manifest
        or guard.get("execution_arguments_sha256") != execution_arguments
    ):
        raise ValueError("v6 invocation, ticket, or orchestrator guard chain changed")
    terminal_required = {
        "run_id": run_id,
        "invocation_present": True,
        "invocation_sha256": invocation_manifest,
        "latest_orchestrator_guard_sha256": guard_manifest,
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
        "resume_allowed": False,
    }
    if any(status.get(key) != value for key, value in terminal_required.items()):
        raise ValueError("v6 post-failure status changed")
    if _microseconds_from_seconds(
        status.get("actual_allocated_gpu_seconds"),
        label="post-failure GPU accounting",
    ) != before.summary.total_allocated_microseconds:
        raise ValueError("v6 status changed cumulative GPU accounting")


def build_fallback_v6_control_plane_incident(
    *,
    run_id: str,
    source_association_path: Path,
    authorization_overlay_path: Path,
    preflight_path: Path,
    restricted_run_root: Path,
    result_path: Path,
    ledger_before_path: Path,
    ledger_after_path: Path,
    audited_at: datetime,
) -> FallbackV6ControlPlaneIncident:
    """Build the v6 incident from the exact public and restricted evidence."""

    if run_id != FALLBACK_V6_RUN_ID:
        raise ValueError("v6 incident builder is restricted to the exact v6 run")
    if audited_at.tzinfo is None or audited_at.utcoffset() is None:
        raise ValueError("incident audit timestamp must be timezone-aware")
    run_root = Path(restricted_run_root)
    _require_no_symlink_ancestry(run_root, label="restricted v6 run root")
    if not run_root.resolve(strict=True).is_dir():
        raise ValueError("restricted v6 run root must be a directory")
    run_root = run_root.resolve(strict=True)
    observed_names = tuple(sorted(item.name for item in run_root.iterdir()))
    if observed_names != EXPECTED_V6_RUN_ROOT_BASENAMES:
        raise ValueError("restricted v6 run root is not the exact preserved artifact set")

    source_path, source = _load_self_hashed_json(
        source_association_path,
        label="v6 source association",
        expected_kind="local_remote_source_tree_association",
    )
    overlay_path, overlay = _load_self_hashed_json(
        authorization_overlay_path,
        label="v6 authorization overlay",
        expected_kind="phase1_fallback_second_recovery_overlay",
    )
    preflight_resolved, preflight = _load_self_hashed_json(
        preflight_path,
        label="v6 execution preflight",
        expected_kind="phase1_fallback_execution_preflight",
    )
    specs = {
        "invocation": (
            "checkpoint.json.orchestrator-invocation.json",
            "fallback_controller_orchestration_invocation",
        ),
        "ticket": (
            "checkpoint.json.guardian-ticket.json",
            "fallback_service_guardian_ticket",
        ),
        "guard": (
            "checkpoint.json.orchestrator-guard-000001.json",
            "fallback_controller_orchestrator_guard",
        ),
        "status": (
            "status.after-failure.json",
            "fallback_controller_orchestration_status",
        ),
    }
    loaded = {
        label: _load_self_hashed_json(
            run_root / basename,
            label=f"v6 {label}",
            expected_kind=kind,
        )
        for label, (basename, kind) in specs.items()
    }
    invocation_path, invocation = loaded["invocation"]
    ticket_path, ticket = loaded["ticket"]
    guard_path, guard = loaded["guard"]
    status_path, status = loaded["status"]
    guardian_log_path = _require_regular_file(
        run_root / "checkpoint.json.guardian.log", label="v6 guardian log"
    )
    orchestrator_log_path = _require_regular_file(
        run_root / "orchestrator.20260905T170000Z.log",
        label="v6 orchestrator log",
    )
    try:
        guardian_log = guardian_log_path.read_text(encoding="utf-8")
        orchestrator_log = orchestrator_log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("v6 diagnostic logs are not readable UTF-8") from exc
    if (
        "DevelopmentContinuationError: second recovery requires the exact ordered "
        "v3+v5 service IDs" not in guardian_log
    ):
        raise ValueError("v6 guardian log lacks the exact service-identity failure")
    if "RuntimeError: fallback guardian exited before readiness (1)" not in orchestrator_log:
        raise ValueError("v6 orchestrator log lacks the exact readiness failure")

    result = Path(os.path.abspath(os.fspath(result_path)))
    _require_no_symlink_ancestry(result, label="v6 public result")
    checkpoint = run_root / "checkpoint.json"
    required_absent = (
        run_root / ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
        checkpoint,
        run_root / "checkpoint.json.guardian-controller-takeover.json",
        run_root / "checkpoint.json.guardian-ready-000001.json",
        run_root / "checkpoint.json.guardian-result.json",
        run_root / "checkpoint.json.guardian-terminal-request.json",
        run_root / "checkpoint.json.guardian.lock",
        run_root / "checkpoint.json.service",
        run_root / f"{run_id}.vllm.log",
        result,
        result.with_name(result.name + ".controller-handoff.json"),
        result.with_name(result.name + ".orphan-cleanup.json"),
    )
    absent_names = tuple(path.name for path in required_absent)
    if absent_names != EXPECTED_V6_ABSENT_BASENAMES:
        raise AssertionError("internal v6 absence inventory changed")
    if any(path.exists() or path.is_symlink() for path in required_absent):
        raise ValueError("a required v6 zero-state path exists")
    ready_receipts = tuple(run_root.glob("checkpoint.json.guardian-ready-*.json"))
    controller_receipts = tuple(run_root.glob("checkpoint.json.internal-controller-*.json"))
    if ready_receipts or controller_receipts:
        raise ValueError("v6 guardian-ready or internal-controller receipt count changed")

    before, before_samples = _ledger_binding_and_samples(
        ledger_before_path, label="pre-v6 ledger snapshot"
    )
    after, after_samples = _ledger_binding_and_samples(
        ledger_after_path, label="post-v6 ledger snapshot"
    )
    added_ids = after_samples.keys() - before_samples.keys()
    removed_ids = before_samples.keys() - after_samples.keys()
    changed_ids = {
        sample_id
        for sample_id in before_samples.keys() & after_samples.keys()
        if before_samples[sample_id] != after_samples[sample_id]
    }
    if len(added_ids) != 1 or removed_ids or changed_ids:
        raise ValueError("v6 storage history changed beyond one appended sample")
    appended_sample = _typed_storage_sample(after_samples[next(iter(added_ids))])

    _require_json_relationships(
        run_id=run_id,
        result_path=result,
        checkpoint_path=checkpoint,
        source_path=source_path,
        source=source,
        overlay=overlay,
        preflight=preflight,
        invocation=invocation,
        ticket=ticket,
        guard=guard,
        status=status,
        before=before,
    )
    accounting = FallbackV6Accounting(
        before=before,
        after=after,
        appended_storage_sample=appended_sample,
        prior_storage_samples_preserved=True,
        all_non_storage_tables_identical=True,
        delta=FallbackV6ZeroGpuDelta(
            allocated_gpu_microseconds=0,
            gpu_events=0,
            service_sessions=0,
            service_start_events=0,
            attempts=0,
            inference_model_calls=0,
            artifacts=0,
            storage_samples=1,
            unresolved_gpu_allocations=0,
            unresolved_gpu_services=0,
            accepted_outputs=0,
            guardian_process_launch_attempts=1,
            internal_controller_launch_attempts=0,
            authorized_retry_inference_consumed=False,
            authorized_service_start_consumed=False,
        ),
    )
    run_root_inventory = {
        "files": [
            {
                "basename": name,
                "size_bytes": (run_root / name).stat().st_size,
                "file_sha256": _sha256_file(run_root / name),
            }
            for name in observed_names
        ]
    }
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": FALLBACK_V6_CONTROL_PLANE_INCIDENT_KIND,
        "run_id": run_id,
        "audited_at": audited_at,
        "source_and_authorization": FallbackV6SourceAndAuthorization(
            source_revision=FALLBACK_V6_SOURCE_REVISION,
            source_git_commit=source["git_commit"],
            source_tree_sha256=source["local_tree_sha256"],
            source_association=_self_hashed_binding(source_path, source),
            authorization_overlay=_self_hashed_binding(overlay_path, overlay),
            execution_preflight=_self_hashed_binding(preflight_resolved, preflight),
        ),
        "restricted_evidence": FallbackV6RestrictedEvidence(
            orchestration_invocation=_self_hashed_binding(invocation_path, invocation),
            guardian_ticket=_self_hashed_binding(ticket_path, ticket),
            orchestrator_guard=_self_hashed_binding(guard_path, guard),
            post_failure_status=_self_hashed_binding(status_path, status),
            guardian_log=_opaque_binding(guardian_log_path, label="v6 guardian log"),
            orchestrator_log=_opaque_binding(
                orchestrator_log_path, label="v6 orchestrator log"
            ),
            run_root_inventory_sha256=_canonical_sha256(run_root_inventory),
        ),
        "orchestration_identity": FallbackV6OrchestrationIdentity(
            execution_arguments_sha256=invocation["execution_arguments_sha256"],
            orchestration_invocation_manifest_sha256=invocation["manifest_sha256"],
            guardian_ticket_manifest_sha256=ticket["manifest_sha256"],
            orchestrator_guard_manifest_sha256=guard["manifest_sha256"],
            post_failure_status_manifest_sha256=status["manifest_sha256"],
            guardian_command_sha256=ticket["guardian_command_sha256"],
            orchestrator_command_sha256=guard["orchestrator_command_sha256"],
            service_session_id=run_id,
            service_event_id=f"{run_id}-service-start-001",
        ),
        "accounting": accounting,
        "absence_inventory": FallbackV6AbsenceInventory(
            observed_run_root_basenames=observed_names,
            absent_basenames=absent_names,
            guardian_ready_receipt_count=0,
            internal_controller_receipt_count=0,
            all_required_paths_absent=True,
        ),
        "terminal_state": FallbackV6TerminalState(
            guardian_ready=False,
            guardian_terminal_receipt_present=False,
            guardian_live=False,
            latest_orchestrator_live=False,
            latest_control_group_live=False,
            live_internal_controller_count=0,
            checkpoint_present=False,
            controller_handoff_present=False,
            cleanup_present=False,
            public_result_present=False,
            service_start_attempted=None,
            model_process_started=False,
            vllm_service_started=False,
            gpu_allocation_performed=False,
            inference_attempt_count=0,
            accepted_output_count=0,
            unresolved_gpu_allocation_count=0,
            unresolved_gpu_service_count=0,
            ledger_snapshot_verified=True,
            resume_allowed=False,
            source_bound_v6_resume_permitted=False,
            fresh_repaired_source_required=True,
        ),
        "root_cause": FallbackV6RootCause(
            safe_error_class=FALLBACK_V6_CONTROL_PLANE_ERROR_CLASS,
            guardian_exception_type="DevelopmentContinuationError",
            outer_exception_type="RuntimeError",
            failure_stage="guardian_construction_before_readiness",
            guardian_exit_code=1,
            frozen_expected_service_event_ids=FROZEN_RECOVERY_SERVICE_EVENT_IDS,
            factory_derived_service_event_ids=DERIVED_V6_RECOVERY_SERVICE_EVENT_IDS,
            mismatch_position=1,
            scientific_generation_reached=False,
            command_line_public=False,
            remote_absolute_paths_public=False,
            full_log_text_public=False,
        ),
    }
    return FallbackV6ControlPlaneIncident.model_validate(
        {**payload, "manifest_sha256": _canonical_sha256(payload)}
    )


def load_fallback_v6_control_plane_incident(path: Path) -> FallbackV6ControlPlaneIncident:
    """Load and fully validate one regular self-hashed v6 incident artifact."""

    resolved = _require_regular_file(path, label="fallback-v6 control-plane incident")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fallback-v6 incident is not valid UTF-8 JSON") from exc
    try:
        return FallbackV6ControlPlaneIncident.model_validate(value)
    except ValidationError as exc:
        raise ValueError("fallback-v6 incident contract is invalid") from exc


def validate_fallback_v6_control_plane_incident(
    path: Path,
    *,
    expected_source_association_manifest_sha256: str,
    expected_overlay_manifest_sha256: str,
    expected_preflight_manifest_sha256: str,
    expected_before_ledger_file_sha256: str,
    expected_after_ledger_file_sha256: str,
    expected_before_ledger_summary: Mapping[str, object],
    expected_after_ledger_summary: Mapping[str, object],
) -> FallbackV6ControlPlaneIncident:
    """Validate v6 against exact identities frozen by a later lineage record."""

    incident = load_fallback_v6_control_plane_incident(path)
    source = incident.source_and_authorization
    if (
        source.source_association.manifest_sha256
        != expected_source_association_manifest_sha256
        or source.authorization_overlay.manifest_sha256 != expected_overlay_manifest_sha256
        or source.execution_preflight.manifest_sha256 != expected_preflight_manifest_sha256
        or incident.accounting.before.file_sha256 != expected_before_ledger_file_sha256
        or incident.accounting.after.file_sha256 != expected_after_ledger_file_sha256
        or incident.accounting.before.summary
        != FallbackV6LedgerSummary.model_validate(expected_before_ledger_summary)
        or incident.accounting.after.summary
        != FallbackV6LedgerSummary.model_validate(expected_after_ledger_summary)
    ):
        raise ValueError("fallback-v6 incident differs from its expected identity")
    return incident


def write_fallback_v6_control_plane_incident(
    path: Path,
    incident: FallbackV6ControlPlaneIncident,
) -> None:
    """Append one canonical public incident; exact replay is idempotent."""

    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="v6 incident output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(destination, label="v6 incident output")
    payload = (canonical_json(incident) + "\n").encode("utf-8")
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError("append-only v6 incident output already differs")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


__all__ = [
    "DERIVED_V6_RECOVERY_SERVICE_EVENT_IDS",
    "EXPECTED_V6_ABSENT_BASENAMES",
    "EXPECTED_V6_RUN_ROOT_BASENAMES",
    "FALLBACK_V6_CONTROL_PLANE_ERROR_CLASS",
    "FALLBACK_V6_CONTROL_PLANE_INCIDENT_KIND",
    "FALLBACK_V6_PREFLIGHT_MANIFEST_SHA256",
    "FALLBACK_V6_RUN_ID",
    "FALLBACK_V6_SOURCE_REVISION",
    "FROZEN_RECOVERY_SERVICE_EVENT_IDS",
    "FallbackV6Accounting",
    "FallbackV6ControlPlaneIncident",
    "FallbackV6LedgerSummary",
    "build_fallback_v6_control_plane_incident",
    "load_fallback_v6_control_plane_incident",
    "validate_fallback_v6_control_plane_incident",
    "write_fallback_v6_control_plane_incident",
]
