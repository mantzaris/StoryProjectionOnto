"""Typed public record for the fallback-v5 zero-GPU control-plane incident.

The public record authenticates the exact restricted receipt set without
publishing command lines, absolute paths, or log text.  Unlike the earlier v4
incident, the before/after ledger files are intentionally not byte-identical:
the guardian appended one storage-admission observation before the outer
orchestrator timed out.  Scientific and GPU-accounting tables remain unchanged.
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

FALLBACK_V5_RUN_ID = "fallback-qwen3-8b-awq-development-v5"
FALLBACK_V5_SOURCE_REVISION = "fallback-second-recovery-v5"
FALLBACK_V5_CONTROL_PLANE_INCIDENT_KIND = (
    "fallback_gpu_acceptance_v5_control_plane_incident"
)
FALLBACK_V5_CONTROL_PLANE_ERROR_CLASS = (
    "guardian_readiness_watchdog_elapsed_before_ready_publish"
)
FALLBACK_V5_ACTIVE_STATUS_ERROR_CLASS = "active_guardian_status_refused_live_wal"
FALLBACK_V5_PREFLIGHT_MANIFEST_SHA256 = (
    "f3b90b16a323f81066b4392fccecdb5af69718e76b67f0325dec9d8adaf70b24"
)
FALLBACK_V4_INCIDENT_MANIFEST_SHA256 = (
    "e8b30397068a97f4f169395ec0a70ec1f709515c1df07b6a17400c7568c2993c"
)

EXPECTED_V5_RUN_ROOT_BASENAMES = (
    "checkpoint.json.guardian-controller-takeover.json",
    "checkpoint.json.guardian-ready-000001.json",
    "checkpoint.json.guardian-result.json",
    "checkpoint.json.guardian-ticket.json",
    "checkpoint.json.guardian.lock",
    "checkpoint.json.guardian.log",
    "checkpoint.json.orchestrator-guard-000001.json",
    "checkpoint.json.orchestrator-invocation.json",
    "orchestrator.20260905T155544Z.log",
    "status.active-guardian-attempt-000001.exit-code.txt",
    "status.active-guardian-attempt-000001.log",
    "status.after-terminal.json",
)
EXPECTED_V5_ABSENT_BASENAMES = (
    ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
    "checkpoint.json",
    "checkpoint.json.guardian-terminal-request.json",
    "checkpoint.json.service",
    "fallback-qwen3-8b-awq-development-v5.vllm.log",
    "fallback_gpu_acceptance_development_v5.json",
    "fallback_gpu_acceptance_development_v5.json.controller-handoff.json",
    "fallback_gpu_acceptance_development_v5.json.orphan-cleanup.json",
)


class _StrictIncidentRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class FallbackV5SourceAndAuthorization(_StrictIncidentRecord):
    source_revision: Literal["fallback-second-recovery-v5"]
    source_git_commit: GitRevision
    source_tree_sha256: Sha256Digest
    source_association: SelfHashedFileBinding
    authorization_overlay: SelfHashedFileBinding
    execution_preflight: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_v5_artifact_names_and_kinds(self) -> Self:
        expected = (
            (
                self.source_association,
                "source_tree_fallback_second_recovery_v5.association.json",
                "local_remote_source_tree_association",
            ),
            (
                self.authorization_overlay,
                "fallback-second-recovery-v5.authorized.json",
                "phase1_fallback_second_recovery_overlay",
            ),
            (
                self.execution_preflight,
                "fallback_gpu_acceptance_development_v5.preflight.json",
                "phase1_fallback_execution_preflight",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected
        ):
            raise ValueError("v5 source, overlay, or preflight artifact identity changed")
        return self


class FallbackV5RestrictedEvidence(_StrictIncidentRecord):
    orchestration_invocation: SelfHashedFileBinding
    guardian_ticket: SelfHashedFileBinding
    orchestrator_guard: SelfHashedFileBinding
    guardian_ready: SelfHashedFileBinding
    controller_takeover: SelfHashedFileBinding
    guardian_result: SelfHashedFileBinding
    post_terminal_status: SelfHashedFileBinding
    orchestrator_log: OpaqueFileBinding
    active_status_log: OpaqueFileBinding
    active_status_exit_code: OpaqueFileBinding
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
                self.guardian_ready,
                "checkpoint.json.guardian-ready-000001.json",
                "fallback_service_guardian_ready",
            ),
            (
                self.controller_takeover,
                "checkpoint.json.guardian-controller-takeover.json",
                "fallback_guardian_controller_takeover",
            ),
            (
                self.guardian_result,
                "checkpoint.json.guardian-result.json",
                "fallback_service_guardian_result",
            ),
            (
                self.post_terminal_status,
                "status.after-terminal.json",
                "fallback_controller_orchestration_status",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected_json
        ):
            raise ValueError("v5 restricted JSON artifact identity changed")
        expected_opaque = (
            (self.orchestrator_log, "orchestrator.20260905T155544Z.log"),
            (
                self.active_status_log,
                "status.active-guardian-attempt-000001.log",
            ),
            (
                self.active_status_exit_code,
                "status.active-guardian-attempt-000001.exit-code.txt",
            ),
        )
        if any(binding.basename != basename for binding, basename in expected_opaque):
            raise ValueError("v5 restricted diagnostic artifact identity changed")
        return self


class FallbackV5OrchestrationIdentity(_StrictIncidentRecord):
    execution_arguments_sha256: Sha256Digest
    orchestration_invocation_manifest_sha256: Sha256Digest
    guardian_ticket_manifest_sha256: Sha256Digest
    orchestrator_guard_manifest_sha256: Sha256Digest
    guardian_ready_manifest_sha256: Sha256Digest
    controller_takeover_manifest_sha256: Sha256Digest
    guardian_result_manifest_sha256: Sha256Digest
    guardian_command_sha256: Sha256Digest
    orchestrator_command_sha256: Sha256Digest
    service_session_id: Literal["fallback-qwen3-8b-awq-development-v5"]
    service_event_id: Literal["fallback-qwen3-8b-awq-development-v5-service-start-001"]


class FallbackV5LedgerSummary(_StrictIncidentRecord):
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
    def reconcile_summary(self) -> Self:
        if any(
            not key or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.by_kind_microseconds.items()
        ):
            raise ValueError("GPU accounting kinds must be nonempty nonnegative integers")
        if self.total_allocated_microseconds != sum(self.by_kind_microseconds.values()):
            raise ValueError("GPU accounting kinds do not reconcile to the total")
        return self


class FallbackV5LedgerBinding(_StrictIncidentRecord):
    basename: SafeBasename
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest
    summary: FallbackV5LedgerSummary


class FallbackV5StorageSample(_StrictIncidentRecord):
    sample_id: Sha256Digest
    phase: Literal["phase1_fallback:fallback-qwen3-8b-awq-development-v5"]
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
            raise ValueError("v5 storage admission must have no violations")
        if self.projected_occupied_bytes != (
            self.current_occupied_bytes + self.additional_reserved_bytes
        ):
            raise ValueError("v5 storage admission arithmetic changed")
        return self


class FallbackV5ZeroGpuDelta(_StrictIncidentRecord):
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
    authorized_retry_inference_consumed: Literal[False]
    authorized_service_start_consumed: Literal[False]
    control_plane_launch_attempts: Literal[1]


class FallbackV5Accounting(_StrictIncidentRecord):
    before: FallbackV5LedgerBinding
    after: FallbackV5LedgerBinding
    appended_storage_sample: FallbackV5StorageSample
    prior_storage_samples_preserved: Literal[True]
    delta: FallbackV5ZeroGpuDelta

    @model_validator(mode="after")
    def require_one_storage_only_delta(self) -> Self:
        if (
            self.before.basename != "phase1_acceptance.before-v5.sqlite"
            or self.after.basename != "phase1_acceptance.sqlite"
            or self.before.file_sha256 == self.after.file_sha256
        ):
            raise ValueError("v5 before/after ledger identity changed")
        before = self.before.summary.model_dump(mode="python")
        expected_after = {**before, "storage_sample_count": before["storage_sample_count"] + 1}
        if self.after.summary.model_dump(mode="python") != expected_after:
            raise ValueError("v5 ledger changed beyond one storage observation")
        return self


class FallbackV5AbsenceInventory(_StrictIncidentRecord):
    observed_run_root_basenames: tuple[SafeBasename, ...]
    absent_basenames: tuple[SafeBasename, ...]
    guardian_ready_receipt_count: Literal[1]
    internal_controller_receipt_count: Literal[0]
    all_required_paths_absent: Literal[True]

    @model_validator(mode="after")
    def require_exact_inventories(self) -> Self:
        if self.observed_run_root_basenames != EXPECTED_V5_RUN_ROOT_BASENAMES:
            raise ValueError("v5 run-root inventory is not the exact preserved artifact set")
        if self.absent_basenames != EXPECTED_V5_ABSENT_BASENAMES:
            raise ValueError("v5 absence inventory changed")
        return self


class FallbackV5TerminalState(_StrictIncidentRecord):
    outer_exit_code: Literal[1]
    guardian_ready: Literal[True]
    guardian_terminal: Literal[True]
    owner_loss_takeover: Literal[True]
    controller_launch_authority_revoked: Literal[True]
    physical_shutdown_verified: Literal[True]
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
    resume_allowed: Literal[False]
    source_bound_v5_resume_permitted: Literal[False]
    fresh_source_bound_v6_run_required: Literal[True]


class FallbackV5ActiveStatusDiagnostic(_StrictIncidentRecord):
    attempt_count: Literal[1]
    exit_code: Literal[1]
    safe_error_class: Literal["active_guardian_status_refused_live_wal"]
    exception_type: Literal["ArtifactIntegrityError"]
    guardian_was_active: Literal[True]
    scientific_generation_reached: Literal[False]
    full_log_text_public: Literal[False]


class FallbackV5FailureClassification(_StrictIncidentRecord):
    safe_error_class: Literal[
        "guardian_readiness_watchdog_elapsed_before_ready_publish"
    ]
    outer_exception_type: Literal["RuntimeError"]
    failure_stage: Literal["guardian_readiness_wait"]
    readiness_watchdog_seconds: Literal[30]
    guardian_ready_latency_microseconds: int = Field(gt=30_000_000, strict=True)
    guardian_eventually_published_ready: Literal[True]
    scientific_generation_reached: Literal[False]
    command_line_public: Literal[False]
    remote_absolute_paths_public: Literal[False]


class FallbackV5ControlPlaneIncident(_StrictIncidentRecord):
    """Self-hashed public description of the exact fallback-v5 incident."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_gpu_acceptance_v5_control_plane_incident"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v5"]
    audited_at: AwareDatetime
    source_and_authorization: FallbackV5SourceAndAuthorization
    restricted_evidence: FallbackV5RestrictedEvidence
    orchestration_identity: FallbackV5OrchestrationIdentity
    accounting: FallbackV5Accounting
    absence_inventory: FallbackV5AbsenceInventory
    terminal_state: FallbackV5TerminalState
    active_status_diagnostic: FallbackV5ActiveStatusDiagnostic
    failure: FallbackV5FailureClassification
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
            (evidence.guardian_ticket.manifest_sha256, identity.guardian_ticket_manifest_sha256),
            (
                evidence.orchestrator_guard.manifest_sha256,
                identity.orchestrator_guard_manifest_sha256,
            ),
            (evidence.guardian_ready.manifest_sha256, identity.guardian_ready_manifest_sha256),
            (
                evidence.controller_takeover.manifest_sha256,
                identity.controller_takeover_manifest_sha256,
            ),
            (evidence.guardian_result.manifest_sha256, identity.guardian_result_manifest_sha256),
        )
        if any(observed != required for observed, required in expected):
            raise ValueError("restricted receipt bindings disagree with orchestration identity")
        if self.accounting.before.summary.total_allocated_microseconds != 815_215_409:
            raise ValueError("v5 incident changed the registered cumulative GPU allocation")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _canonical_sha256(immutable):
            raise ValueError("fallback-v5 control-plane incident manifest hash changed")
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


def _ledger_binding_and_samples(
    path: Path,
    *,
    label: str,
) -> tuple[FallbackV5LedgerBinding, dict[str, StorageSampleRecord]]:
    resolved = _require_regular_file(path, label=label)
    with ReadOnlyLedger(resolved) as ledger:
        summary = FallbackV5LedgerSummary.model_validate(
            _summary_payload(ledger.gpu_summary(), ledger)
        )
        samples = {sample.sample_id: sample for sample in ledger.storage_samples()}
    return (
        FallbackV5LedgerBinding(
            basename=resolved.name,
            size_bytes=resolved.stat().st_size,
            file_sha256=_sha256_file(resolved),
            summary=summary,
        ),
        samples,
    )


def _typed_storage_sample(sample: StorageSampleRecord) -> FallbackV5StorageSample:
    return FallbackV5StorageSample(
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


def _aware_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed


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
    ready: Mapping[str, Any],
    takeover: Mapping[str, Any],
    guardian_result: Mapping[str, Any],
    status: Mapping[str, Any],
    before: FallbackV5LedgerBinding,
) -> int:
    source_manifest = source.get("manifest_sha256")
    overlay_manifest = overlay.get("manifest_sha256")
    invocation_manifest = invocation.get("manifest_sha256")
    ticket_manifest = ticket.get("manifest_sha256")
    guard_manifest = guard.get("manifest_sha256")
    ready_manifest = ready.get("manifest_sha256")
    takeover_manifest = takeover.get("manifest_sha256")
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
    if (
        source.get("revision_label") != FALLBACK_V5_SOURCE_REVISION
        or source.get("local_tree_sha256") != source.get("remote_tree_sha256")
        or overlay.get("schema_version") != "1.3.0"
        or overlay.get("authorized_recovery_run_id") != run_id
        or not isinstance(authorization, Mapping)
        or authorization.get("status") != "authorized"
        or not isinstance(v4_binding, Mapping)
        or v4_binding.get("incident_manifest_sha256")
        != FALLBACK_V4_INCIDENT_MANIFEST_SHA256
        or preflight.get("manifest_sha256") != FALLBACK_V5_PREFLIGHT_MANIFEST_SHA256
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
        or not isinstance(source_binding, Mapping)
        or source_binding.get("current_association_manifest_sha256") != source_manifest
        or source_binding.get("current_tree_sha256") != source.get("local_tree_sha256")
        or preflight.get("gpu_accounting_before_start") != core_accounting
        or overlay.get("cumulative_gpu_accounting") != core_accounting
    ):
        raise ValueError("v5 source, authorization, preflight, or ledger binding changed")
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
        raise ValueError("v5 orchestration invocation, ticket, or guard chain changed")
    if (
        ready.get("run_id") != run_id
        or ready.get("orchestration_invocation_sha256") != invocation_manifest
        or ready.get("guardian_ticket_sha256") != ticket_manifest
        or ready.get("guardian_command_sha256") != ticket.get("guardian_command_sha256")
        or takeover.get("run_id") != run_id
        or takeover.get("trigger") != "orchestrator_lost"
        or takeover.get("controller_receipt_sha256s_before_takeover") != []
        or takeover.get("orchestration_invocation_sha256") != invocation_manifest
        or takeover.get("guardian_ticket_sha256") != ticket_manifest
        or takeover.get("execution_arguments_sha256") != execution_arguments
        or takeover.get("latest_orchestrator_guard_sha256") != guard_manifest
        or guardian_result.get("run_id") != run_id
        or guardian_result.get("trigger") != "orchestrator_lost"
        or guardian_result.get("orchestration_invocation_sha256") != invocation_manifest
        or guardian_result.get("guardian_ticket_sha256") != ticket_manifest
        or guardian_result.get("controller_takeover_sha256") != takeover_manifest
        or guardian_result.get("execution_arguments_sha256") != execution_arguments
        or guardian_result.get("physical_shutdown_verified") is not True
        or guardian_result.get("accounted_service_seconds") is not None
        or guardian_result.get("service_session_accounting_sha256") is not None
        or guardian_result.get("terminal_request_sha256") is not None
        or guardian_result.get("checkpoint_service_adopted") is not False
        or guardian_result.get("uptime_recovered") is not False
        or _microseconds_from_seconds(
            guardian_result.get("actual_allocated_gpu_seconds"),
            label="guardian-result GPU accounting",
        )
        != before.summary.total_allocated_microseconds
    ):
        raise ValueError("v5 guardian ready, takeover, or terminal chain changed")
    outcomes = guardian_result.get("control_group_outcomes")
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 1
        or not isinstance(outcomes[0], Mapping)
        or outcomes[0].get("orchestrator_guard_sha256") != guard_manifest
        or outcomes[0].get("bound_controller_allocation_absent") is not True
        or outcomes[0].get("exact_live_identity_count_before_signal") != 0
        or outcomes[0].get("process_group_absent") is not True
        or outcomes[0].get("sigterm_sent") is not False
        or outcomes[0].get("sigkill_sent") is not False
    ):
        raise ValueError("v5 guardian terminal control-group outcome changed")
    terminal_required = {
        "run_id": run_id,
        "invocation_present": True,
        "invocation_sha256": invocation_manifest,
        "latest_orchestrator_guard_sha256": guard_manifest,
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
        "resume_allowed": False,
    }
    if any(status.get(key) != value for key, value in terminal_required.items()):
        raise ValueError("v5 post-terminal status changed")
    if _microseconds_from_seconds(
        status.get("actual_allocated_gpu_seconds"),
        label="terminal-status GPU accounting",
    ) != before.summary.total_allocated_microseconds:
        raise ValueError("v5 terminal status changed cumulative GPU accounting")
    created_at = _aware_datetime(invocation.get("created_at"), label="invocation creation")
    ready_at = _aware_datetime(ready.get("ready_at"), label="guardian readiness")
    takeover_at = _aware_datetime(
        takeover.get("controller_launch_authority_revoked_at"),
        label="guardian takeover",
    )
    ready_latency_microseconds = round((ready_at - created_at).total_seconds() * 1_000_000)
    if ready_latency_microseconds <= 30_000_000 or takeover_at <= ready_at:
        raise ValueError("v5 guardian timing does not prove the readiness watchdog failure")
    if guardian_result.get("manifest_sha256") is None or ready_manifest is None:
        raise ValueError("v5 guardian receipt chain lacks self-hashed identities")
    return ready_latency_microseconds


def build_fallback_v5_control_plane_incident(
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
) -> FallbackV5ControlPlaneIncident:
    """Build the v5 incident from exact public and restricted durable evidence."""

    if run_id != FALLBACK_V5_RUN_ID:
        raise ValueError("v5 incident builder is restricted to the exact v5 run")
    if audited_at.tzinfo is None or audited_at.utcoffset() is None:
        raise ValueError("incident audit timestamp must be timezone-aware")
    run_root = Path(restricted_run_root)
    if run_root.is_symlink() or not run_root.resolve(strict=True).is_dir():
        raise ValueError("restricted run root must be a regular non-symlink directory")
    run_root = run_root.resolve(strict=True)
    observed_names = tuple(sorted(item.name for item in run_root.iterdir()))
    if observed_names != EXPECTED_V5_RUN_ROOT_BASENAMES:
        raise ValueError("restricted v5 run root is not the exact preserved artifact set")

    source_path, source = _load_self_hashed_json(
        source_association_path,
        label="v5 source association",
        expected_kind="local_remote_source_tree_association",
    )
    overlay_path, overlay = _load_self_hashed_json(
        authorization_overlay_path,
        label="v5 authorization overlay",
        expected_kind="phase1_fallback_second_recovery_overlay",
    )
    preflight_resolved, preflight = _load_self_hashed_json(
        preflight_path,
        label="v5 execution preflight",
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
        "ready": (
            "checkpoint.json.guardian-ready-000001.json",
            "fallback_service_guardian_ready",
        ),
        "takeover": (
            "checkpoint.json.guardian-controller-takeover.json",
            "fallback_guardian_controller_takeover",
        ),
        "guardian_result": (
            "checkpoint.json.guardian-result.json",
            "fallback_service_guardian_result",
        ),
        "status": (
            "status.after-terminal.json",
            "fallback_controller_orchestration_status",
        ),
    }
    loaded = {
        label: _load_self_hashed_json(
            run_root / basename,
            label=f"v5 {label}",
            expected_kind=kind,
        )
        for label, (basename, kind) in receipt_specs.items()
    }
    invocation_path, invocation = loaded["invocation"]
    ticket_path, ticket = loaded["ticket"]
    guard_path, guard = loaded["guard"]
    ready_path, ready = loaded["ready"]
    takeover_path, takeover = loaded["takeover"]
    guardian_result_path, guardian_result = loaded["guardian_result"]
    status_path, status = loaded["status"]

    orchestrator_log_path = _require_regular_file(
        run_root / "orchestrator.20260905T155544Z.log",
        label="v5 orchestrator log",
    )
    active_status_log_path = _require_regular_file(
        run_root / "status.active-guardian-attempt-000001.log",
        label="v5 active status log",
    )
    active_status_exit_path = _require_regular_file(
        run_root / "status.active-guardian-attempt-000001.exit-code.txt",
        label="v5 active status exit code",
    )
    guardian_log_path = _require_regular_file(
        run_root / "checkpoint.json.guardian.log",
        label="v5 guardian log",
    )
    guardian_lock_path = _require_regular_file(
        run_root / "checkpoint.json.guardian.lock",
        label="v5 guardian lock",
    )
    try:
        orchestrator_log = orchestrator_log_path.read_text(encoding="utf-8")
        active_status_log = active_status_log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("v5 restricted diagnostic logs are not readable UTF-8") from exc
    if "fallback guardian did not publish readiness before its watchdog" not in orchestrator_log:
        raise ValueError("v5 orchestrator log lacks the readiness-watchdog failure")
    if "ArtifactIntegrityError: read-only ledger must be closed and checkpointed" not in (
        active_status_log
    ):
        raise ValueError("v5 active status log lacks the live-WAL refusal")
    if active_status_exit_path.read_bytes() != b"1\n":
        raise ValueError("v5 active status diagnostic exit code changed")
    if guardian_log_path.stat().st_size != 0 or guardian_lock_path.stat().st_size != 0:
        raise ValueError("v5 empty guardian coordination files changed")

    result = Path(os.path.abspath(os.fspath(result_path)))
    _require_no_symlink_ancestry(result, label="v5 public result")
    checkpoint_path = run_root / "checkpoint.json"
    required_absent = (
        run_root / ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
        checkpoint_path,
        run_root / "checkpoint.json.guardian-terminal-request.json",
        run_root / "checkpoint.json.service",
        run_root / f"{run_id}.vllm.log",
        result,
        result.with_name(result.name + ".controller-handoff.json"),
        result.with_name(result.name + ".orphan-cleanup.json"),
    )
    absent_names = tuple(path.name for path in required_absent)
    if absent_names != EXPECTED_V5_ABSENT_BASENAMES:
        raise AssertionError("internal v5 absence inventory changed")
    if any(path.exists() or path.is_symlink() for path in required_absent):
        raise ValueError("a required v5 zero-state path exists")
    ready_receipts = tuple(run_root.glob("checkpoint.json.guardian-ready-*.json"))
    controller_receipts = tuple(run_root.glob("checkpoint.json.internal-controller-*.json"))
    if ready_receipts != (ready_path,) or controller_receipts:
        raise ValueError("v5 guardian-ready or internal-controller receipt count changed")

    before, before_samples = _ledger_binding_and_samples(
        ledger_before_path,
        label="pre-v5 ledger snapshot",
    )
    after, after_samples = _ledger_binding_and_samples(
        ledger_after_path,
        label="post-v5 canonical ledger",
    )
    added_ids = after_samples.keys() - before_samples.keys()
    removed_ids = before_samples.keys() - after_samples.keys()
    changed_ids = {
        sample_id
        for sample_id in before_samples.keys() & after_samples.keys()
        if before_samples[sample_id] != after_samples[sample_id]
    }
    if len(added_ids) != 1 or removed_ids or changed_ids:
        raise ValueError("v5 ledger storage history changed beyond one appended sample")
    appended_sample = _typed_storage_sample(after_samples[next(iter(added_ids))])

    ready_latency = _require_json_relationships(
        run_id=run_id,
        result_path=result,
        checkpoint_path=checkpoint_path,
        source=source,
        overlay=overlay,
        preflight=preflight,
        invocation=invocation,
        ticket=ticket,
        guard=guard,
        ready=ready,
        takeover=takeover,
        guardian_result=guardian_result,
        status=status,
        before=before,
    )
    accounting = FallbackV5Accounting(
        before=before,
        after=after,
        appended_storage_sample=appended_sample,
        prior_storage_samples_preserved=True,
        delta=FallbackV5ZeroGpuDelta(
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
            authorized_retry_inference_consumed=False,
            authorized_service_start_consumed=False,
            control_plane_launch_attempts=1,
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
        "kind": FALLBACK_V5_CONTROL_PLANE_INCIDENT_KIND,
        "run_id": run_id,
        "audited_at": audited_at,
        "source_and_authorization": FallbackV5SourceAndAuthorization(
            source_revision=FALLBACK_V5_SOURCE_REVISION,
            source_git_commit=source["git_commit"],
            source_tree_sha256=source["local_tree_sha256"],
            source_association=_self_hashed_binding(source_path, source),
            authorization_overlay=_self_hashed_binding(overlay_path, overlay),
            execution_preflight=_self_hashed_binding(preflight_resolved, preflight),
        ),
        "restricted_evidence": FallbackV5RestrictedEvidence(
            orchestration_invocation=_self_hashed_binding(invocation_path, invocation),
            guardian_ticket=_self_hashed_binding(ticket_path, ticket),
            orchestrator_guard=_self_hashed_binding(guard_path, guard),
            guardian_ready=_self_hashed_binding(ready_path, ready),
            controller_takeover=_self_hashed_binding(takeover_path, takeover),
            guardian_result=_self_hashed_binding(guardian_result_path, guardian_result),
            post_terminal_status=_self_hashed_binding(status_path, status),
            orchestrator_log=_opaque_binding(
                orchestrator_log_path,
                label="v5 orchestrator log",
            ),
            active_status_log=_opaque_binding(
                active_status_log_path,
                label="v5 active status log",
            ),
            active_status_exit_code=_opaque_binding(
                active_status_exit_path,
                label="v5 active status exit code",
            ),
            run_root_inventory_sha256=_canonical_sha256(run_root_inventory),
        ),
        "orchestration_identity": FallbackV5OrchestrationIdentity(
            execution_arguments_sha256=invocation["execution_arguments_sha256"],
            orchestration_invocation_manifest_sha256=invocation["manifest_sha256"],
            guardian_ticket_manifest_sha256=ticket["manifest_sha256"],
            orchestrator_guard_manifest_sha256=guard["manifest_sha256"],
            guardian_ready_manifest_sha256=ready["manifest_sha256"],
            controller_takeover_manifest_sha256=takeover["manifest_sha256"],
            guardian_result_manifest_sha256=guardian_result["manifest_sha256"],
            guardian_command_sha256=ticket["guardian_command_sha256"],
            orchestrator_command_sha256=guard["orchestrator_command_sha256"],
            service_session_id=run_id,
            service_event_id=f"{run_id}-service-start-001",
        ),
        "accounting": accounting,
        "absence_inventory": FallbackV5AbsenceInventory(
            observed_run_root_basenames=observed_names,
            absent_basenames=absent_names,
            guardian_ready_receipt_count=1,
            internal_controller_receipt_count=0,
            all_required_paths_absent=True,
        ),
        "terminal_state": FallbackV5TerminalState(
            outer_exit_code=1,
            guardian_ready=True,
            guardian_terminal=True,
            owner_loss_takeover=True,
            controller_launch_authority_revoked=True,
            physical_shutdown_verified=True,
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
            resume_allowed=False,
            source_bound_v5_resume_permitted=False,
            fresh_source_bound_v6_run_required=True,
        ),
        "active_status_diagnostic": FallbackV5ActiveStatusDiagnostic(
            attempt_count=1,
            exit_code=1,
            safe_error_class=FALLBACK_V5_ACTIVE_STATUS_ERROR_CLASS,
            exception_type="ArtifactIntegrityError",
            guardian_was_active=True,
            scientific_generation_reached=False,
            full_log_text_public=False,
        ),
        "failure": FallbackV5FailureClassification(
            safe_error_class=FALLBACK_V5_CONTROL_PLANE_ERROR_CLASS,
            outer_exception_type="RuntimeError",
            failure_stage="guardian_readiness_wait",
            readiness_watchdog_seconds=30,
            guardian_ready_latency_microseconds=ready_latency,
            guardian_eventually_published_ready=True,
            scientific_generation_reached=False,
            command_line_public=False,
            remote_absolute_paths_public=False,
        ),
    }
    return FallbackV5ControlPlaneIncident.model_validate(
        {**payload, "manifest_sha256": _canonical_sha256(payload)}
    )


def load_fallback_v5_control_plane_incident(path: Path) -> FallbackV5ControlPlaneIncident:
    """Load and fully validate one regular self-hashed v5 incident artifact."""

    resolved = _require_regular_file(path, label="fallback-v5 control-plane incident")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fallback-v5 incident is not valid UTF-8 JSON") from exc
    try:
        return FallbackV5ControlPlaneIncident.model_validate(value)
    except ValidationError as exc:
        raise ValueError("fallback-v5 incident contract is invalid") from exc


def validate_fallback_v5_control_plane_incident(
    path: Path,
    *,
    expected_source_association_manifest_sha256: str,
    expected_overlay_manifest_sha256: str,
    expected_preflight_manifest_sha256: str,
    expected_before_ledger_file_sha256: str,
    expected_after_ledger_file_sha256: str,
    expected_before_ledger_summary: Mapping[str, object],
    expected_after_ledger_summary: Mapping[str, object],
) -> FallbackV5ControlPlaneIncident:
    """Validate v5 against exact identities frozen by a later v6 overlay."""

    incident = load_fallback_v5_control_plane_incident(path)
    source = incident.source_and_authorization
    if (
        source.source_association.manifest_sha256
        != expected_source_association_manifest_sha256
        or source.authorization_overlay.manifest_sha256 != expected_overlay_manifest_sha256
        or source.execution_preflight.manifest_sha256 != expected_preflight_manifest_sha256
        or incident.accounting.before.file_sha256 != expected_before_ledger_file_sha256
        or incident.accounting.after.file_sha256 != expected_after_ledger_file_sha256
        or incident.accounting.before.summary
        != FallbackV5LedgerSummary.model_validate(expected_before_ledger_summary)
        or incident.accounting.after.summary
        != FallbackV5LedgerSummary.model_validate(expected_after_ledger_summary)
    ):
        raise ValueError("fallback-v5 incident differs from its expected identity")
    return incident


def write_fallback_v5_control_plane_incident(
    path: Path,
    incident: FallbackV5ControlPlaneIncident,
) -> None:
    """Append one canonical public incident; exact replay is idempotent."""

    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="v5 incident output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(destination, label="v5 incident output")
    payload = (canonical_json(incident) + "\n").encode("utf-8")
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError("append-only v5 incident output already differs")
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
    "EXPECTED_V5_ABSENT_BASENAMES",
    "EXPECTED_V5_RUN_ROOT_BASENAMES",
    "FALLBACK_V5_ACTIVE_STATUS_ERROR_CLASS",
    "FALLBACK_V5_CONTROL_PLANE_ERROR_CLASS",
    "FALLBACK_V5_CONTROL_PLANE_INCIDENT_KIND",
    "FALLBACK_V5_PREFLIGHT_MANIFEST_SHA256",
    "FALLBACK_V5_RUN_ID",
    "FALLBACK_V5_SOURCE_REVISION",
    "FallbackV5Accounting",
    "FallbackV5ControlPlaneIncident",
    "FallbackV5LedgerSummary",
    "build_fallback_v5_control_plane_incident",
    "load_fallback_v5_control_plane_incident",
    "validate_fallback_v5_control_plane_incident",
    "write_fallback_v5_control_plane_incident",
]
