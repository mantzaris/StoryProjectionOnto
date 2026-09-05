"""Typed public record for the fallback-v4 zero-GPU control-plane incident.

The incident record deliberately exposes only safe artifact identities.  The
restricted receipts and logs remain outside the public artifact tree; their
basenames, byte counts, file digests, and (for JSON receipts) canonical logical
digests are sufficient to authenticate a later private audit without publishing
commands, absolute paths, or log text.
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
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from story_projection_onto.contracts import GitRevision, Sha256Digest, canonical_json
from story_projection_onto.contracts import canonical_sha256 as _canonical_sha256
from story_projection_onto.store import GpuEventKind, GpuSummary, ReadOnlyLedger

FALLBACK_V4_RUN_ID = "fallback-qwen3-8b-awq-development-v4"
FALLBACK_V4_SOURCE_REVISION = "fallback-second-recovery-v4"
FALLBACK_CONTROL_PLANE_INCIDENT_KIND = "fallback_gpu_acceptance_control_plane_incident"
FALLBACK_CONTROL_PLANE_ERROR_CLASS = "guardian_outer_result_path_identity_mismatch_before_readiness"

SafeBasename = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9.][A-Za-z0-9._-]*$",
    ),
]

EXPECTED_RUN_ROOT_BASENAMES = (
    "checkpoint.json.guardian-ticket.json",
    "checkpoint.json.guardian.log",
    "checkpoint.json.orchestrator-guard-000001.json",
    "checkpoint.json.orchestrator-invocation.json",
    "orchestrator.20260905T143000Z.log",
    "status.after-failure.json",
)
EXPECTED_ABSENT_BASENAMES = (
    ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
    "checkpoint.json",
    "checkpoint.json.guardian-controller-takeover.json",
    "checkpoint.json.guardian-result.json",
    "checkpoint.json.guardian-terminal-request.json",
    "checkpoint.json.guardian.lock",
    "checkpoint.json.service",
    "fallback-qwen3-8b-awq-development-v4.vllm.log",
    "fallback_gpu_acceptance_development_v4.json",
    "fallback_gpu_acceptance_development_v4.json.controller-handoff.json",
    "fallback_gpu_acceptance_development_v4.json.orphan-cleanup.json",
)


class _StrictIncidentRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class SelfHashedFileBinding(_StrictIncidentRecord):
    """Safe public identity for one private or public self-hashed JSON file."""

    basename: SafeBasename
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest
    kind: SafeBasename
    manifest_sha256: Sha256Digest

    @field_validator("basename")
    @classmethod
    def require_one_basename(cls, value: str) -> str:
        if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
            raise ValueError("artifact identity must contain one safe basename")
        return value


class OpaqueFileBinding(_StrictIncidentRecord):
    """Safe public identity for one restricted non-JSON artifact."""

    basename: SafeBasename
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest

    @field_validator("basename")
    @classmethod
    def require_one_basename(cls, value: str) -> str:
        if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
            raise ValueError("artifact identity must contain one safe basename")
        return value


class FallbackV4SourceAndAuthorization(_StrictIncidentRecord):
    source_revision: Literal["fallback-second-recovery-v4"]
    source_git_commit: GitRevision
    source_tree_sha256: Sha256Digest
    source_association: SelfHashedFileBinding
    authorization_overlay: SelfHashedFileBinding
    execution_preflight: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_v4_artifact_names_and_kinds(self) -> Self:
        expected = (
            (
                self.source_association,
                "source_tree_fallback_second_recovery_v4.association.json",
                "local_remote_source_tree_association",
            ),
            (
                self.authorization_overlay,
                "fallback-second-recovery-v4.authorized.json",
                "phase1_fallback_second_recovery_overlay",
            ),
            (
                self.execution_preflight,
                "fallback_gpu_acceptance_development_v4.preflight.json",
                "phase1_fallback_execution_preflight",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected
        ):
            raise ValueError("v4 source, overlay, or preflight artifact identity changed")
        return self


class FallbackV4RestrictedEvidence(_StrictIncidentRecord):
    orchestration_invocation: SelfHashedFileBinding
    guardian_ticket: SelfHashedFileBinding
    orchestrator_guard: SelfHashedFileBinding
    post_failure_status: SelfHashedFileBinding
    guardian_log: OpaqueFileBinding
    orchestrator_log: OpaqueFileBinding
    run_root_inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def require_exact_restricted_artifact_names_and_kinds(self) -> Self:
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
            raise ValueError("v4 restricted JSON artifact identity changed")
        if (
            self.guardian_log.basename != "checkpoint.json.guardian.log"
            or self.orchestrator_log.basename != "orchestrator.20260905T143000Z.log"
        ):
            raise ValueError("v4 restricted log identity changed")
        return self


class FallbackV4OrchestrationIdentity(_StrictIncidentRecord):
    execution_arguments_sha256: Sha256Digest
    orchestration_invocation_manifest_sha256: Sha256Digest
    guardian_ticket_manifest_sha256: Sha256Digest
    orchestrator_guard_manifest_sha256: Sha256Digest
    guardian_command_sha256: Sha256Digest
    orchestrator_command_sha256: Sha256Digest
    service_session_id: Literal["fallback-qwen3-8b-awq-development-v4"]
    service_event_id: Literal["fallback-qwen3-8b-awq-development-v4-service-start-001"]


class FallbackV4LedgerSummary(_StrictIncidentRecord):
    total_allocated_microseconds: int = Field(ge=0, strict=True)
    event_count: int = Field(ge=0, strict=True)
    service_session_count: int = Field(ge=0, strict=True)
    service_start_event_count: int = Field(ge=0, strict=True)
    attempt_count: int = Field(ge=0, strict=True)
    model_call_count: int = Field(ge=0, strict=True)
    artifact_count: int = Field(ge=0, strict=True)
    unresolved_gpu_allocation_count: int = Field(ge=0, strict=True)
    unresolved_gpu_service_count: int = Field(ge=0, strict=True)
    by_kind_microseconds: dict[str, int]

    @model_validator(mode="after")
    def reconcile_summary(self) -> Self:
        if any(
            not key or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.by_kind_microseconds.items()
        ):
            raise ValueError("GPU accounting kinds must be nonempty nonnegative integers")
        if self.total_allocated_microseconds != sum(self.by_kind_microseconds.values()):
            raise ValueError("GPU accounting kinds do not reconcile to the total")
        return self


class FallbackV4LedgerBinding(_StrictIncidentRecord):
    basename: Literal["phase1_acceptance.sqlite"]
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest
    summary: FallbackV4LedgerSummary


class FallbackV4ZeroDelta(_StrictIncidentRecord):
    allocated_gpu_microseconds: Literal[0]
    gpu_events: Literal[0]
    service_sessions: Literal[0]
    service_start_events: Literal[0]
    attempts: Literal[0]
    inference_model_calls: Literal[0]
    artifacts: Literal[0]
    unresolved_gpu_allocations: Literal[0]
    unresolved_gpu_services: Literal[0]
    accepted_outputs: Literal[0]
    authorized_retry_inference_consumed: Literal[False]
    authorized_service_start_consumed: Literal[False]
    control_plane_launch_attempts: Literal[1]


class FallbackV4Accounting(_StrictIncidentRecord):
    before: FallbackV4LedgerBinding
    after: FallbackV4LedgerBinding
    delta: FallbackV4ZeroDelta

    @model_validator(mode="after")
    def require_byte_identical_zero_delta(self) -> Self:
        if self.before != self.after:
            raise ValueError("zero-GPU incident requires byte-identical before/after ledgers")
        return self


class FallbackV4AbsenceInventory(_StrictIncidentRecord):
    observed_run_root_basenames: tuple[SafeBasename, ...]
    absent_basenames: tuple[SafeBasename, ...]
    guardian_ready_receipt_count: Literal[0]
    internal_controller_receipt_count: Literal[0]
    all_required_paths_absent: Literal[True]

    @model_validator(mode="after")
    def require_exact_inventories(self) -> Self:
        if self.observed_run_root_basenames != EXPECTED_RUN_ROOT_BASENAMES:
            raise ValueError("v4 run-root inventory is not the exact preserved six-file set")
        if self.absent_basenames != EXPECTED_ABSENT_BASENAMES:
            raise ValueError("v4 absence inventory changed")
        return self


class FallbackV4TerminalState(_StrictIncidentRecord):
    outer_exit_code: Literal[1]
    guardian_exit_code: Literal[1]
    guardian_ready: Literal[False]
    guardian_terminal: Literal[False]
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
    latest_orchestrator_live: Literal[False]
    latest_control_group_live: Literal[False]
    live_internal_controller_count: Literal[0]
    historical_status_resume_allowed: Literal[True]
    source_bound_v4_resume_permitted: Literal[False]
    fresh_source_bound_run_required: Literal[True]


class FallbackV4FailureClassification(_StrictIncidentRecord):
    safe_error_class: Literal["guardian_outer_result_path_identity_mismatch_before_readiness"]
    outer_exception_type: Literal["RuntimeError"]
    guardian_exception_type: Literal["ValueError"]
    failure_stage: Literal["guardian_identity_validation_before_readiness"]
    scientific_generation_reached: Literal[False]
    full_log_text_public: Literal[False]
    command_line_public: Literal[False]
    remote_absolute_paths_public: Literal[False]


class FallbackControlPlaneIncident(_StrictIncidentRecord):
    """Self-hashed public description of the exact fallback-v4 incident."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_gpu_acceptance_control_plane_incident"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v4"]
    audited_at: AwareDatetime
    source_and_authorization: FallbackV4SourceAndAuthorization
    restricted_evidence: FallbackV4RestrictedEvidence
    orchestration_identity: FallbackV4OrchestrationIdentity
    accounting: FallbackV4Accounting
    absence_inventory: FallbackV4AbsenceInventory
    terminal_state: FallbackV4TerminalState
    failure: FallbackV4FailureClassification
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_cross_record_identity_and_manifest(self) -> Self:
        evidence = self.restricted_evidence
        identity = self.orchestration_identity
        if (
            evidence.orchestration_invocation.manifest_sha256
            != identity.orchestration_invocation_manifest_sha256
            or evidence.guardian_ticket.manifest_sha256 != identity.guardian_ticket_manifest_sha256
            or evidence.orchestrator_guard.manifest_sha256
            != identity.orchestrator_guard_manifest_sha256
        ):
            raise ValueError("restricted receipt bindings disagree with orchestration identity")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _canonical_sha256(immutable):
            raise ValueError("fallback control-plane incident manifest hash changed")
        return self


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, *, label: str) -> Path:
    supplied = Path(path).absolute()
    _require_no_symlink_ancestry(supplied, label=label)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _require_no_symlink_ancestry(path: Path, *, label: str) -> None:
    supplied = Path(path).absolute()
    for component in (supplied, *supplied.parents):
        if component.is_symlink():
            raise ValueError(f"{label} cannot have symlink ancestry")


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
    by_kind = {kind.value: value for kind, value in summary.by_kind_microseconds}
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
        "unresolved_gpu_allocation_count": len(ledger.unresolved_gpu_allocations()),
        "unresolved_gpu_service_count": len(ledger.unresolved_gpu_service_journals()),
        "by_kind_microseconds": by_kind,
    }


def _ledger_binding(path: Path, *, label: str) -> FallbackV4LedgerBinding:
    resolved = _require_regular_file(path, label=label)
    with ReadOnlyLedger(resolved) as ledger:
        summary = FallbackV4LedgerSummary.model_validate(
            _summary_payload(ledger.gpu_summary(), ledger)
        )
    return FallbackV4LedgerBinding(
        basename=resolved.name,
        size_bytes=resolved.stat().st_size,
        file_sha256=_sha256_file(resolved),
        summary=summary,
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
    source: Mapping[str, Any],
    overlay: Mapping[str, Any],
    preflight: Mapping[str, Any],
    invocation: Mapping[str, Any],
    ticket: Mapping[str, Any],
    guard: Mapping[str, Any],
    status: Mapping[str, Any],
    before: FallbackV4LedgerBinding,
) -> None:
    source_manifest = source.get("manifest_sha256")
    overlay_manifest = overlay.get("manifest_sha256")
    invocation_manifest = invocation.get("manifest_sha256")
    ticket_manifest = ticket.get("manifest_sha256")
    guard_manifest = guard.get("manifest_sha256")
    execution_arguments = invocation.get("execution_arguments_sha256")
    accounting = preflight.get("gpu_accounting_before_start")
    expected_core_accounting = {
        "total_allocated_microseconds": before.summary.total_allocated_microseconds,
        "event_count": before.summary.event_count,
        "service_session_count": before.summary.service_session_count,
        "by_kind_microseconds": before.summary.by_kind_microseconds,
    }
    source_binding = overlay.get("source")
    authorization = overlay.get("authorization")
    if (
        source.get("revision_label") != FALLBACK_V4_SOURCE_REVISION
        or source.get("local_tree_sha256") != source.get("remote_tree_sha256")
        or overlay.get("authorized_recovery_run_id") != run_id
        or not isinstance(authorization, Mapping)
        or authorization.get("status") != "authorized"
        or preflight.get("run_id") != run_id
        or preflight.get("execution_authorized") is not True
        or preflight.get("passed") is not True
        or preflight.get("gpu_allocation_performed") is not False
        or preflight.get("model_process_started") is not False
        or preflight.get("checkpoint_absent") is not True
        or preflight.get("source_association_sha256") != source_manifest
        or preflight.get("source_tree_sha256") != source.get("local_tree_sha256")
        or preflight.get("second_recovery_overlay_sha256") != overlay_manifest
        or not isinstance(source_binding, Mapping)
        or source_binding.get("current_association_manifest_sha256") != source_manifest
        or source_binding.get("current_tree_sha256") != source.get("local_tree_sha256")
        or not isinstance(accounting, Mapping)
        or dict(accounting) != expected_core_accounting
        or overlay.get("cumulative_gpu_accounting") != expected_core_accounting
    ):
        raise ValueError("v4 source, authorization, preflight, or ledger binding changed")
    if (
        invocation.get("run_id") != run_id
        or Path(os.path.abspath(str(invocation.get("result_output")))) != result_path
        or Path(os.path.abspath(str(invocation.get("checkpoint")))) != checkpoint_path
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
        or status.get("run_id") != run_id
        or status.get("invocation_present") is not True
        or status.get("invocation_sha256") != invocation_manifest
        or status.get("latest_orchestrator_guard_sha256") != guard_manifest
        or status.get("actual_allocated_gpu_seconds") is None
        or _microseconds_from_seconds(
            status.get("actual_allocated_gpu_seconds"),
            label="status GPU accounting",
        )
        != before.summary.total_allocated_microseconds
    ):
        raise ValueError("v4 orchestration receipt chain changed")


def _require_terminal_status(status: Mapping[str, Any]) -> None:
    required = {
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
        "resume_allowed": True,
    }
    if any(status.get(key) != expected for key, expected in required.items()):
        raise ValueError("v4 post-failure status is not the exact pre-service zero state")


def build_fallback_control_plane_incident(
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
) -> FallbackControlPlaneIncident:
    """Build the v4 incident from exact public and restricted durable evidence."""

    if run_id != FALLBACK_V4_RUN_ID:
        raise ValueError("control-plane incident builder is restricted to the exact v4 run")
    if audited_at.tzinfo is None or audited_at.utcoffset() is None:
        raise ValueError("incident audit timestamp must be timezone-aware")
    run_root = Path(restricted_run_root)
    if run_root.is_symlink() or not run_root.resolve(strict=True).is_dir():
        raise ValueError("restricted run root must be a regular non-symlink directory")
    run_root = run_root.resolve(strict=True)
    observed_names = tuple(sorted(item.name for item in run_root.iterdir()))
    if observed_names != EXPECTED_RUN_ROOT_BASENAMES:
        raise ValueError("restricted v4 run root is not the exact preserved six-file set")

    source_path, source = _load_self_hashed_json(
        source_association_path,
        label="v4 source association",
        expected_kind="local_remote_source_tree_association",
    )
    overlay_path, overlay = _load_self_hashed_json(
        authorization_overlay_path,
        label="v4 authorization overlay",
        expected_kind="phase1_fallback_second_recovery_overlay",
    )
    preflight_resolved, preflight = _load_self_hashed_json(
        preflight_path,
        label="v4 execution preflight",
        expected_kind="phase1_fallback_execution_preflight",
    )
    receipt_specs = {
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
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for label, (basename, kind) in receipt_specs.items():
        loaded[label] = _load_self_hashed_json(
            run_root / basename,
            label=f"v4 {label}",
            expected_kind=kind,
        )
    invocation_path, invocation = loaded["invocation"]
    ticket_path, ticket = loaded["ticket"]
    guard_path, guard = loaded["guard"]
    status_path, status = loaded["status"]

    guardian_log_path = _require_regular_file(
        run_root / "checkpoint.json.guardian.log",
        label="v4 guardian log",
    )
    orchestrator_log_path = _require_regular_file(
        run_root / "orchestrator.20260905T143000Z.log",
        label="v4 orchestrator log",
    )
    try:
        guardian_log = guardian_log_path.read_text(encoding="utf-8")
        orchestrator_log = orchestrator_log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("v4 restricted logs are not readable UTF-8") from exc
    if "fallback orchestration invocation identity changed on resume" not in guardian_log:
        raise ValueError("guardian log does not contain the expected identity failure")
    if "fallback guardian exited before readiness (1)" not in orchestrator_log:
        raise ValueError("orchestrator log does not contain the expected readiness failure")

    result = Path(os.path.abspath(os.fspath(result_path)))
    _require_no_symlink_ancestry(result, label="v4 public result")
    checkpoint_path = run_root / "checkpoint.json"
    required_absent = (
        run_root / ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
        run_root / "checkpoint.json",
        run_root / "checkpoint.json.guardian-controller-takeover.json",
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
    if absent_names != EXPECTED_ABSENT_BASENAMES:
        raise AssertionError("internal v4 absence inventory changed")
    if any(path.exists() or path.is_symlink() for path in required_absent):
        raise ValueError("a required v4 zero-state path exists")
    if list(run_root.glob("checkpoint.json.guardian-ready-*.json")) or list(
        run_root.glob("checkpoint.json.internal-controller-*.json")
    ):
        raise ValueError("v4 run root contains a controller or guardian-ready receipt")

    before = _ledger_binding(ledger_before_path, label="pre-v4 ledger snapshot")
    after = _ledger_binding(ledger_after_path, label="post-v4 ledger snapshot")
    if before != after:
        raise ValueError("v4 changed the canonical GPU ledger")
    _require_json_relationships(
        run_id=run_id,
        result_path=result,
        checkpoint_path=checkpoint_path,
        source=source,
        overlay=overlay,
        preflight=preflight,
        invocation=invocation,
        ticket=ticket,
        guard=guard,
        status=status,
        before=before,
    )
    _require_terminal_status(status)

    run_root_inventory = {
        "basenames": observed_names,
        "file_sha256": {name: _sha256_file(run_root / name) for name in observed_names},
    }
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": FALLBACK_CONTROL_PLANE_INCIDENT_KIND,
        "run_id": run_id,
        "audited_at": audited_at,
        "source_and_authorization": FallbackV4SourceAndAuthorization(
            source_revision=FALLBACK_V4_SOURCE_REVISION,
            source_git_commit=source["git_commit"],
            source_tree_sha256=source["local_tree_sha256"],
            source_association=_self_hashed_binding(source_path, source),
            authorization_overlay=_self_hashed_binding(overlay_path, overlay),
            execution_preflight=_self_hashed_binding(preflight_resolved, preflight),
        ),
        "restricted_evidence": FallbackV4RestrictedEvidence(
            orchestration_invocation=_self_hashed_binding(invocation_path, invocation),
            guardian_ticket=_self_hashed_binding(ticket_path, ticket),
            orchestrator_guard=_self_hashed_binding(guard_path, guard),
            post_failure_status=_self_hashed_binding(status_path, status),
            guardian_log=_opaque_binding(guardian_log_path, label="v4 guardian log"),
            orchestrator_log=_opaque_binding(orchestrator_log_path, label="v4 orchestrator log"),
            run_root_inventory_sha256=_canonical_sha256(run_root_inventory),
        ),
        "orchestration_identity": FallbackV4OrchestrationIdentity(
            execution_arguments_sha256=invocation["execution_arguments_sha256"],
            orchestration_invocation_manifest_sha256=invocation["manifest_sha256"],
            guardian_ticket_manifest_sha256=ticket["manifest_sha256"],
            orchestrator_guard_manifest_sha256=guard["manifest_sha256"],
            guardian_command_sha256=ticket["guardian_command_sha256"],
            orchestrator_command_sha256=guard["orchestrator_command_sha256"],
            service_session_id=run_id,
            service_event_id=f"{run_id}-service-start-001",
        ),
        "accounting": FallbackV4Accounting(
            before=before,
            after=after,
            delta=FallbackV4ZeroDelta(
                allocated_gpu_microseconds=0,
                gpu_events=0,
                service_sessions=0,
                service_start_events=0,
                attempts=0,
                inference_model_calls=0,
                artifacts=0,
                unresolved_gpu_allocations=0,
                unresolved_gpu_services=0,
                accepted_outputs=0,
                authorized_retry_inference_consumed=False,
                authorized_service_start_consumed=False,
                control_plane_launch_attempts=1,
            ),
        ),
        "absence_inventory": FallbackV4AbsenceInventory(
            observed_run_root_basenames=observed_names,
            absent_basenames=absent_names,
            guardian_ready_receipt_count=0,
            internal_controller_receipt_count=0,
            all_required_paths_absent=True,
        ),
        "terminal_state": FallbackV4TerminalState(
            outer_exit_code=1,
            guardian_exit_code=1,
            guardian_ready=False,
            guardian_terminal=False,
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
            latest_orchestrator_live=False,
            latest_control_group_live=False,
            live_internal_controller_count=0,
            historical_status_resume_allowed=True,
            source_bound_v4_resume_permitted=False,
            fresh_source_bound_run_required=True,
        ),
        "failure": FallbackV4FailureClassification(
            safe_error_class=FALLBACK_CONTROL_PLANE_ERROR_CLASS,
            outer_exception_type="RuntimeError",
            guardian_exception_type="ValueError",
            failure_stage="guardian_identity_validation_before_readiness",
            scientific_generation_reached=False,
            full_log_text_public=False,
            command_line_public=False,
            remote_absolute_paths_public=False,
        ),
    }
    manifest_sha256 = _canonical_sha256(payload)
    return FallbackControlPlaneIncident.model_validate(
        {**payload, "manifest_sha256": manifest_sha256}
    )


def load_fallback_control_plane_incident(path: Path) -> FallbackControlPlaneIncident:
    """Load and fully validate one regular self-hashed incident artifact."""

    resolved = _require_regular_file(path, label="fallback control-plane incident")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fallback control-plane incident is not valid UTF-8 JSON") from exc
    try:
        return FallbackControlPlaneIncident.model_validate(value)
    except ValidationError as exc:
        raise ValueError("fallback control-plane incident contract is invalid") from exc


def validate_fallback_control_plane_incident(
    path: Path,
    *,
    expected_run_id: str,
    expected_source_association_manifest_sha256: str,
    expected_overlay_manifest_sha256: str,
    expected_preflight_manifest_sha256: str,
    expected_ledger_file_sha256: str,
    expected_ledger_summary: Mapping[str, object],
) -> FallbackControlPlaneIncident:
    """Validate the incident against identities frozen by a later v5 overlay."""

    incident = load_fallback_control_plane_incident(path)
    expected_summary = FallbackV4LedgerSummary.model_validate(expected_ledger_summary)
    source = incident.source_and_authorization
    if (
        incident.run_id != expected_run_id
        or source.source_association.manifest_sha256 != expected_source_association_manifest_sha256
        or source.authorization_overlay.manifest_sha256 != expected_overlay_manifest_sha256
        or source.execution_preflight.manifest_sha256 != expected_preflight_manifest_sha256
        or incident.accounting.before.file_sha256 != expected_ledger_file_sha256
        or incident.accounting.after.file_sha256 != expected_ledger_file_sha256
        or incident.accounting.before.summary != expected_summary
        or incident.accounting.after.summary != expected_summary
    ):
        raise ValueError("fallback control-plane incident differs from its expected v4 identity")
    return incident


def write_fallback_control_plane_incident(
    path: Path,
    incident: FallbackControlPlaneIncident,
) -> None:
    """Append one canonical public incident; exact replay is idempotent."""

    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="incident output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(destination, label="incident output")
    payload = (canonical_json(incident) + "\n").encode("utf-8")
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return
        raise FileExistsError("append-only incident output already differs")
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
    "EXPECTED_ABSENT_BASENAMES",
    "EXPECTED_RUN_ROOT_BASENAMES",
    "FALLBACK_CONTROL_PLANE_ERROR_CLASS",
    "FALLBACK_CONTROL_PLANE_INCIDENT_KIND",
    "FALLBACK_V4_RUN_ID",
    "FALLBACK_V4_SOURCE_REVISION",
    "FallbackControlPlaneIncident",
    "FallbackV4Accounting",
    "FallbackV4LedgerSummary",
    "SelfHashedFileBinding",
    "build_fallback_control_plane_incident",
    "load_fallback_control_plane_incident",
    "validate_fallback_control_plane_incident",
    "write_fallback_control_plane_incident",
]
