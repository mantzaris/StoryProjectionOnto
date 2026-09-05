"""Typed public record for the terminal fallback-v7 runtime incident.

The record binds the exact public results, restricted control receipts and
diagnostic logs, and three immutable ledger states without publishing command
lines, absolute paths, or log contents.  The failed run reached a healthy vLLM
endpoint but made no inference call.  A slow in-flight storage measurement
outlived the poll-derived watchdog join, after which the controller lost the
exact live-service lease and automated cleanup could not verify shutdown.  The
later exact-identity SIGTERM stop and conservative ledger recovery are recorded
separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

from story_projection_onto.contracts import GitRevision, Sha256Digest, canonical_json
from story_projection_onto.contracts import canonical_sha256 as _canonical_sha256
from story_projection_onto.fallback_control_plane_incident import (
    OpaqueFileBinding,
    SafeBasename,
    SelfHashedFileBinding,
)
from story_projection_onto.store import GpuEventKind, ReadOnlyLedger

FALLBACK_V7_RUN_ID = "fallback-qwen3-8b-awq-development-v7"
FALLBACK_V7_SOURCE_REVISION = "fallback-second-recovery-v7"
FALLBACK_V7_RUNTIME_INCIDENT_KIND = "fallback_gpu_acceptance_v7_runtime_incident"
FALLBACK_V7_ROOT_CAUSE = (
    "slow in-flight storage resource sample exceeded poll-derived watchdog stop join"
)
FALLBACK_V7_SAFE_ERROR_CLASS = (
    "slow_in_flight_storage_resource_sample_exceeded_poll_derived_watchdog_stop_join"
)
FALLBACK_V7_SERVICE_EVENT_ID = f"{FALLBACK_V7_RUN_ID}-service-start-001"

EXPECTED_BEFORE_TOTAL_MICROSECONDS = 815_215_409
EXPECTED_CLASSIFIED_MICROSECONDS = 224_234_089
EXPECTED_OVERHEAD_MICROSECONDS = 468_401_467
EXPECTED_SERVICE_MICROSECONDS = 692_635_556
EXPECTED_FINAL_TOTAL_MICROSECONDS = 1_507_850_965

EXPECTED_RUN_ROOT_BASENAMES = (
    ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
    "checkpoint.json",
    "checkpoint.json.guardian-ready-000001.json",
    "checkpoint.json.guardian-terminal-request.json",
    "checkpoint.json.guardian-ticket.json",
    "checkpoint.json.guardian.lock",
    "checkpoint.json.guardian.log",
    "checkpoint.json.internal-controller-000001.json",
    "checkpoint.json.internal-controller-000002.json",
    "checkpoint.json.orchestrator-guard-000001.json",
    "checkpoint.json.orchestrator-invocation.json",
    "fallback-qwen3-8b-awq-development-v7.vllm.log",
    "orchestrator.20260905T174607Z.log",
)
EXPECTED_ABSENT_BASENAMES = (
    "checkpoint.json.guardian-controller-takeover.json",
    "checkpoint.json.guardian-result.json",
    "checkpoint.json.service",
    "fallback_gpu_acceptance_development_v7.json",
)


class _StrictIncidentRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class FallbackV7SourceAndAuthorization(_StrictIncidentRecord):
    source_revision: Literal["fallback-second-recovery-v7"]
    source_git_commit: GitRevision
    source_tree_sha256: Sha256Digest
    source_association: SelfHashedFileBinding
    authorization_overlay: SelfHashedFileBinding
    execution_preflight: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_artifact_identities(self) -> Self:
        expected = (
            (
                self.source_association,
                "source_tree_fallback_second_recovery_v7.association.json",
                "local_remote_source_tree_association",
            ),
            (
                self.authorization_overlay,
                "fallback-second-recovery-v7.authorized.json",
                "phase1_fallback_second_recovery_overlay",
            ),
            (
                self.execution_preflight,
                "fallback_gpu_acceptance_development_v7.preflight.json",
                "phase1_fallback_execution_preflight",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected
        ):
            raise ValueError("v7 source, authorization, or preflight identity changed")
        return self


class FallbackV7PublicFailureEvidence(_StrictIncidentRecord):
    controller_handoff: SelfHashedFileBinding
    orphan_cleanup: SelfHashedFileBinding

    @model_validator(mode="after")
    def require_exact_artifact_identities(self) -> Self:
        if (
            self.controller_handoff.basename
            != "fallback_gpu_acceptance_development_v7.json.controller-handoff.json"
            or self.orphan_cleanup.basename
            != "fallback_gpu_acceptance_development_v7.json.orphan-cleanup.json"
            or self.controller_handoff.kind != "phase1_fallback_micro_pilot_result"
            or self.orphan_cleanup.kind != "phase1_fallback_micro_pilot_result"
        ):
            raise ValueError("v7 public controller failure identity changed")
        return self


class FallbackV7RestrictedEvidence(_StrictIncidentRecord):
    orchestration_invocation: SelfHashedFileBinding
    guardian_ticket: SelfHashedFileBinding
    guardian_ready: SelfHashedFileBinding
    guardian_terminal_request: SelfHashedFileBinding
    orchestrator_guard: SelfHashedFileBinding
    closed_orchestrator_guard: SelfHashedFileBinding
    prepare_controller_receipt: SelfHashedFileBinding
    cleanup_controller_receipt: SelfHashedFileBinding
    checkpoint: OpaqueFileBinding
    guardian_log: OpaqueFileBinding
    orchestrator_log: OpaqueFileBinding
    vllm_log: OpaqueFileBinding
    run_root_inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def require_exact_artifact_identities(self) -> Self:
        expected = (
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
                self.guardian_ready,
                "checkpoint.json.guardian-ready-000001.json",
                "fallback_service_guardian_ready",
            ),
            (
                self.guardian_terminal_request,
                "checkpoint.json.guardian-terminal-request.json",
                "fallback_guardian_terminal_request",
            ),
            (
                self.orchestrator_guard,
                "checkpoint.json.orchestrator-guard-000001.json",
                "fallback_controller_orchestrator_guard",
            ),
            (
                self.closed_orchestrator_guard,
                ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
                "fallback_controller_orchestrator_closed",
            ),
            (
                self.prepare_controller_receipt,
                "checkpoint.json.internal-controller-000001.json",
                "fallback_internal_controller_receipt",
            ),
            (
                self.cleanup_controller_receipt,
                "checkpoint.json.internal-controller-000002.json",
                "fallback_internal_controller_receipt",
            ),
        )
        if any(
            binding.basename != basename or binding.kind != kind
            for binding, basename, kind in expected
        ):
            raise ValueError("v7 restricted receipt identity changed")
        opaque_names = (
            self.checkpoint.basename,
            self.guardian_log.basename,
            self.orchestrator_log.basename,
            self.vllm_log.basename,
        )
        if opaque_names != (
            "checkpoint.json",
            "checkpoint.json.guardian.log",
            "orchestrator.20260905T174607Z.log",
            "fallback-qwen3-8b-awq-development-v7.vllm.log",
        ):
            raise ValueError("v7 restricted diagnostic identity changed")
        return self


class FallbackV7OrchestrationIdentity(_StrictIncidentRecord):
    execution_arguments_sha256: Sha256Digest
    orchestration_invocation_manifest_sha256: Sha256Digest
    guardian_ticket_manifest_sha256: Sha256Digest
    guardian_ready_manifest_sha256: Sha256Digest
    guardian_terminal_request_manifest_sha256: Sha256Digest
    orchestrator_guard_manifest_sha256: Sha256Digest
    closed_orchestrator_guard_manifest_sha256: Sha256Digest
    prepare_controller_receipt_manifest_sha256: Sha256Digest
    cleanup_controller_receipt_manifest_sha256: Sha256Digest
    guardian_command_sha256: Sha256Digest
    orchestrator_command_sha256: Sha256Digest
    service_session_id: Literal["fallback-qwen3-8b-awq-development-v7"]
    service_event_id: Literal[
        "fallback-qwen3-8b-awq-development-v7-service-start-001"
    ]


class FallbackV7LedgerSummary(_StrictIncidentRecord):
    total_allocated_microseconds: int = Field(ge=0, strict=True)
    event_count: int = Field(ge=0, strict=True)
    service_session_count: int = Field(ge=0, strict=True)
    attempt_count: int = Field(ge=0, strict=True)
    model_call_count: int = Field(ge=0, strict=True)
    artifact_count: int = Field(ge=0, strict=True)
    storage_sample_count: int = Field(ge=0, strict=True)
    resource_sample_count: int = Field(ge=0, strict=True)
    allocation_journal_count: int = Field(ge=0, strict=True)
    service_journal_count: int = Field(ge=0, strict=True)
    unresolved_gpu_allocation_count: int = Field(ge=0, strict=True)
    unresolved_gpu_service_count: int = Field(ge=0, strict=True)
    by_kind_microseconds: dict[str, int]

    @model_validator(mode="after")
    def reconcile_total(self) -> Self:
        if any(
            not key or isinstance(value, bool) or not isinstance(value, int) or value < 0
            for key, value in self.by_kind_microseconds.items()
        ):
            raise ValueError("GPU accounting kinds must be nonnegative integers")
        if self.total_allocated_microseconds != sum(self.by_kind_microseconds.values()):
            raise ValueError("GPU accounting kinds do not reconcile")
        return self


class FallbackV7LedgerBinding(_StrictIncidentRecord):
    basename: SafeBasename
    size_bytes: int = Field(gt=0, strict=True)
    file_sha256: Sha256Digest
    summary: FallbackV7LedgerSummary


class FallbackV7FailureEvent(_StrictIncidentRecord):
    event_id: Literal["fallback-qwen3-8b-awq-development-v7-service-start-001"]
    event_kind: Literal["failure"]
    allocated_microseconds: Literal[224234089]
    started_at: AwareDatetime
    ended_at: AwareDatetime
    succeeded: Literal[False]
    intended_event_kind: Literal["gpu_session_start"]
    exception_type: Literal["RuntimeConfigurationError"]
    admitted_maximum_microseconds: Literal[300000000]

    @model_validator(mode="after")
    def require_time_order(self) -> Self:
        if self.ended_at <= self.started_at:
            raise ValueError("v7 failure event time order changed")
        return self


class FallbackV7ServiceAccounting(_StrictIncidentRecord):
    service_session_id: Literal[
        "fallback-qwen3-8b-awq-development-v7-service-start-001"
    ]
    session_id: Literal["fallback-qwen3-8b-awq-development-v7"]
    service_microseconds: Literal[692635556]
    classified_event_microseconds: Literal[224234089]
    overhead_microseconds: Literal[468401467]
    started_at: AwareDatetime
    ended_at: AwareDatetime
    accounting_method: Literal["conservative_service_journal_recovery"]
    recovery_journal_sequence: Literal[45]
    recovery_journal_state: Literal["recovered"]

    @model_validator(mode="after")
    def reconcile_service(self) -> Self:
        if (
            self.service_microseconds
            != self.classified_event_microseconds + self.overhead_microseconds
            or self.ended_at <= self.started_at
        ):
            raise ValueError("v7 service accounting does not reconcile")
        return self


class FallbackV7AccountingDelta(_StrictIncidentRecord):
    allocated_gpu_microseconds: Literal[692635556]
    classified_failure_microseconds: Literal[224234089]
    conservative_service_overhead_microseconds: Literal[468401467]
    gpu_events: Literal[1]
    service_sessions: Literal[1]
    attempts: Literal[0]
    inference_model_calls: Literal[0]
    artifacts: Literal[0]
    storage_samples: Literal[8]
    resource_samples: Literal[5]
    allocation_journal_rows: Literal[46]
    service_journal_rows: Literal[46]
    accepted_outputs: Literal[0]


class FallbackV7Accounting(_StrictIncidentRecord):
    before_v7: FallbackV7LedgerBinding
    before_manual_recovery: FallbackV7LedgerBinding
    terminal: FallbackV7LedgerBinding
    failure_event: FallbackV7FailureEvent
    service: FallbackV7ServiceAccounting
    delta: FallbackV7AccountingDelta

    @model_validator(mode="after")
    def require_exact_three_state_reconciliation(self) -> Self:
        before = self.before_v7.summary
        failed = self.before_manual_recovery.summary
        terminal = self.terminal.summary
        if (
            self.before_v7.basename != "phase1_acceptance.before_v7.sqlite"
            or self.before_manual_recovery.basename
            != "phase1_acceptance.before_manual_recovery.sqlite"
            or self.terminal.basename != "phase1_acceptance.sqlite"
            or before.total_allocated_microseconds != EXPECTED_BEFORE_TOTAL_MICROSECONDS
            or failed.total_allocated_microseconds
            != EXPECTED_BEFORE_TOTAL_MICROSECONDS + EXPECTED_CLASSIFIED_MICROSECONDS
            or terminal.total_allocated_microseconds != EXPECTED_FINAL_TOTAL_MICROSECONDS
            or terminal.total_allocated_microseconds - before.total_allocated_microseconds
            != EXPECTED_SERVICE_MICROSECONDS
            or terminal.total_allocated_microseconds
            - failed.total_allocated_microseconds
            != EXPECTED_OVERHEAD_MICROSECONDS
        ):
            raise ValueError("v7 cumulative GPU accounting changed")
        unchanged_counts = ("attempt_count", "model_call_count", "artifact_count")
        if any(
            len({getattr(before, name), getattr(failed, name), getattr(terminal, name)})
            != 1
            for name in unchanged_counts
        ):
            raise ValueError("v7 changed an inference or artifact count")
        if (
            failed.event_count - before.event_count != 1
            or terminal.event_count != failed.event_count
            or failed.service_session_count != before.service_session_count
            or terminal.service_session_count - failed.service_session_count != 1
            or failed.storage_sample_count - before.storage_sample_count != 8
            or terminal.storage_sample_count != failed.storage_sample_count
            or failed.resource_sample_count - before.resource_sample_count != 5
            or terminal.resource_sample_count != failed.resource_sample_count
            or failed.allocation_journal_count - before.allocation_journal_count != 46
            or terminal.allocation_journal_count != failed.allocation_journal_count
            or failed.service_journal_count - before.service_journal_count != 45
            or terminal.service_journal_count - failed.service_journal_count != 1
            or before.unresolved_gpu_allocation_count != 0
            or failed.unresolved_gpu_allocation_count != 0
            or terminal.unresolved_gpu_allocation_count != 0
            or before.unresolved_gpu_service_count != 0
            or failed.unresolved_gpu_service_count != 1
            or terminal.unresolved_gpu_service_count != 0
        ):
            raise ValueError("v7 three-state ledger delta changed")
        failure_before = before.by_kind_microseconds.get("failure", 0)
        overhead_before = before.by_kind_microseconds.get("service_overhead", 0)
        if (
            failed.by_kind_microseconds.get("failure", 0) - failure_before
            != EXPECTED_CLASSIFIED_MICROSECONDS
            or terminal.by_kind_microseconds.get("failure", 0)
            != failed.by_kind_microseconds.get("failure", 0)
            or failed.by_kind_microseconds.get("service_overhead", 0) != overhead_before
            or terminal.by_kind_microseconds.get("service_overhead", 0) - overhead_before
            != EXPECTED_OVERHEAD_MICROSECONDS
        ):
            raise ValueError("v7 failure/overhead classification changed")
        return self


class FallbackV7OperatorObservedIdentity(_StrictIncidentRecord):
    observation_basis: Literal[
        "operator_observed_then_bound_in_terminal_ledger_recovery"
    ]
    service_pid: int = Field(gt=0, strict=True)
    process_group_id: int = Field(gt=0, strict=True)
    process_session_id: int = Field(gt=0, strict=True)
    process_start_ticks: int = Field(gt=0, strict=True)
    process_cmdline_sha256: Sha256Digest
    service_instance_token_sha256: Sha256Digest
    command_line_public: Literal[False]
    remote_absolute_paths_public: Literal[False]

    @model_validator(mode="after")
    def require_single_exact_process_group(self) -> Self:
        if not (self.service_pid == self.process_group_id == self.process_session_id):
            raise ValueError("v7 observed service process identity changed")
        return self


class FallbackV7ManualStopEvidence(_StrictIncidentRecord):
    identity: FallbackV7OperatorObservedIdentity
    stop_signal: Literal["SIGTERM"]
    stopped_at: AwareDatetime
    manual_exact_process_group_stop: Literal[True]
    pid_absence_verified: Literal[True]
    process_group_absence_verified: Literal[True]
    gpu_process_absence_verified: Literal[True]
    endpoint_absence_verified: Literal[True]
    recovered_from_open_service_journal: Literal[True]
    evidence_source: Literal["terminal_ledger_service_details_and_bound_vllm_log"]


class FallbackV7FailureSequence(_StrictIncidentRecord):
    safe_error_class: Literal[
        "slow_in_flight_storage_resource_sample_exceeded_poll_derived_watchdog_stop_join"
    ]
    root_cause: Literal[
        "slow in-flight storage resource sample exceeded poll-derived watchdog stop join"
    ]
    root_cause_basis: Literal["operator_observation_and_source_bound_control_flow"]
    failure_stage: Literal["service_start_post_health_watchdog_stop"]
    healthy_endpoint_before_failure: Literal[True]
    endpoint_healthy_at: AwareDatetime
    failure_event_ended_at: AwareDatetime
    completed_startup_resource_samples: Literal[5]
    in_flight_storage_resource_sample_operator_observed: Literal[True]
    exact_live_lease_recovery_failed: Literal[True]
    automated_physical_shutdown_verification_failed: Literal[True]
    prepare_controller_failure_type: Literal["RuntimeConfigurationError"]
    cleanup_controller_failure_type: Literal["RuntimeError"]
    guardian_terminal_failure_type: Literal["RuntimeError"]
    scientific_inference_reached: Literal[False]
    command_line_public: Literal[False]
    full_log_text_public: Literal[False]

    @model_validator(mode="after")
    def require_endpoint_before_failure(self) -> Self:
        if self.endpoint_healthy_at >= self.failure_event_ended_at:
            raise ValueError("v7 endpoint was not healthy before the classified failure")
        return self


class FallbackV7TerminalState(_StrictIncidentRecord):
    public_final_result_present: Literal[False]
    accepted_output_count: Literal[0]
    inference_attempt_count_delta: Literal[0]
    inference_model_call_count_delta: Literal[0]
    phase1_gate_passed: Literal[False]
    guardian_result_present: Literal[False]
    automated_cleanup_succeeded: Literal[False]
    manual_physical_stop_verified: Literal[True]
    endpoint_live: Literal[False]
    gpu_process_live: Literal[False]
    exact_service_process_live: Literal[False]
    unresolved_gpu_allocation_count: Literal[0]
    unresolved_gpu_service_count: Literal[0]
    terminal_ledger_verified: Literal[True]
    resume_allowed: Literal[False]
    source_bound_v7_resume_permitted: Literal[False]
    fresh_repaired_source_required: Literal[True]


class FallbackV7AbsenceInventory(_StrictIncidentRecord):
    observed_run_root_basenames: tuple[SafeBasename, ...]
    absent_basenames: tuple[SafeBasename, ...]
    all_required_paths_absent: Literal[True]

    @model_validator(mode="after")
    def require_exact_inventory(self) -> Self:
        if self.observed_run_root_basenames != EXPECTED_RUN_ROOT_BASENAMES:
            raise ValueError("v7 run-root inventory changed")
        if self.absent_basenames != EXPECTED_ABSENT_BASENAMES:
            raise ValueError("v7 absence inventory changed")
        return self


class FallbackV7RuntimeIncident(_StrictIncidentRecord):
    """Self-hashed public description of the exact fallback-v7 incident."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    kind: Literal["fallback_gpu_acceptance_v7_runtime_incident"]
    run_id: Literal["fallback-qwen3-8b-awq-development-v7"]
    audited_at: AwareDatetime
    source_and_authorization: FallbackV7SourceAndAuthorization
    public_failure_evidence: FallbackV7PublicFailureEvidence
    restricted_evidence: FallbackV7RestrictedEvidence
    orchestration_identity: FallbackV7OrchestrationIdentity
    accounting: FallbackV7Accounting
    failure_sequence: FallbackV7FailureSequence
    manual_stop: FallbackV7ManualStopEvidence
    absence_inventory: FallbackV7AbsenceInventory
    terminal_state: FallbackV7TerminalState
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_cross_record_identity_and_manifest(self) -> Self:
        evidence = self.restricted_evidence
        identity = self.orchestration_identity
        pairs = (
            (
                evidence.orchestration_invocation.manifest_sha256,
                identity.orchestration_invocation_manifest_sha256,
            ),
            (evidence.guardian_ticket.manifest_sha256, identity.guardian_ticket_manifest_sha256),
            (evidence.guardian_ready.manifest_sha256, identity.guardian_ready_manifest_sha256),
            (
                evidence.guardian_terminal_request.manifest_sha256,
                identity.guardian_terminal_request_manifest_sha256,
            ),
            (
                evidence.orchestrator_guard.manifest_sha256,
                identity.orchestrator_guard_manifest_sha256,
            ),
            (
                evidence.closed_orchestrator_guard.manifest_sha256,
                identity.closed_orchestrator_guard_manifest_sha256,
            ),
            (
                evidence.prepare_controller_receipt.manifest_sha256,
                identity.prepare_controller_receipt_manifest_sha256,
            ),
            (
                evidence.cleanup_controller_receipt.manifest_sha256,
                identity.cleanup_controller_receipt_manifest_sha256,
            ),
        )
        if any(observed != expected for observed, expected in pairs):
            raise ValueError("v7 restricted evidence disagrees with orchestration identity")
        if self.failure_sequence.failure_event_ended_at != self.accounting.failure_event.ended_at:
            raise ValueError("v7 failure timing changed across records")
        if self.manual_stop.stopped_at != self.accounting.service.ended_at:
            raise ValueError("v7 manual stop and service accounting times disagree")
        immutable = self.model_dump(mode="python", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _canonical_sha256(immutable):
            raise ValueError("fallback-v7 incident manifest hash changed")
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


def _load_json_object(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _require_regular_file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
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


def _summary(ledger: ReadOnlyLedger) -> FallbackV7LedgerSummary:
    gpu = ledger.gpu_summary()
    return FallbackV7LedgerSummary(
        total_allocated_microseconds=gpu.total_allocated_microseconds,
        event_count=gpu.event_count,
        service_session_count=gpu.service_session_count,
        attempt_count=ledger.count_rows("attempts"),
        model_call_count=ledger.count_rows("model_calls"),
        artifact_count=ledger.count_rows("artifacts"),
        storage_sample_count=ledger.count_rows("storage_samples"),
        resource_sample_count=ledger.count_rows("resource_samples"),
        allocation_journal_count=ledger.count_rows("gpu_allocation_journal"),
        service_journal_count=ledger.count_rows("gpu_service_journal"),
        unresolved_gpu_allocation_count=len(ledger.unresolved_gpu_allocations()),
        unresolved_gpu_service_count=len(ledger.unresolved_gpu_service_journals()),
        by_kind_microseconds={
            kind.value: microseconds for kind, microseconds in gpu.by_kind_microseconds
        },
    )


def _ledger_binding(path: Path, *, label: str) -> FallbackV7LedgerBinding:
    resolved = _require_regular_file(path, label=label)
    with ReadOnlyLedger(resolved) as ledger:
        summary = _summary(ledger)
    return FallbackV7LedgerBinding(
        basename=resolved.name,
        size_bytes=resolved.stat().st_size,
        file_sha256=_sha256_file(resolved),
        summary=summary,
    )


def _json_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _microseconds_from_seconds(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative number")
    microseconds = round(float(value) * 1_000_000)
    if abs(float(value) - microseconds / 1_000_000) > 1e-9:
        raise ValueError(f"{label} is not exact to microsecond precision")
    return microseconds


def _endpoint_healthy_timestamp(
    vllm_log: str,
    *,
    failure_event: FallbackV7FailureEvent,
) -> datetime:
    matches = re.findall(
        r"INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) .*"
        r"Starting vLLM API server 0 on http://127\.0\.0\.1:8000",
        vllm_log,
    )
    if len(matches) != 1:
        raise ValueError("v7 vLLM log lacks one unambiguous healthy-endpoint timestamp")
    month, day, hour, minute, second = (int(value) for value in matches[0])
    observed = datetime(
        failure_event.started_at.year,
        month,
        day,
        hour,
        minute,
        second,
        tzinfo=UTC,
    )
    if observed < failure_event.started_at or observed >= failure_event.ended_at:
        raise ValueError("v7 vLLM healthy-endpoint timestamp is outside the failure event")
    return observed


def _require_source_chain(
    *,
    source_path: Path,
    source: Mapping[str, Any],
    overlay: Mapping[str, Any],
    preflight: Mapping[str, Any],
    before: FallbackV7LedgerBinding,
) -> None:
    source_hash = source.get("manifest_sha256")
    overlay_hash = overlay.get("manifest_sha256")
    overlay_source = _json_mapping(overlay.get("source"), label="v7 overlay source")
    authorization = _json_mapping(
        overlay.get("authorization"), label="v7 overlay authorization"
    )
    v6 = _json_mapping(
        overlay.get("intervening_v6_control_plane_incident"),
        label="v7 overlay v6 incident",
    )
    accounting = {
        "total_allocated_microseconds": before.summary.total_allocated_microseconds,
        "event_count": before.summary.event_count,
        "service_session_count": before.summary.service_session_count,
        "by_kind_microseconds": before.summary.by_kind_microseconds,
    }
    if (
        source.get("revision_label") != FALLBACK_V7_SOURCE_REVISION
        or source.get("local_tree_sha256") != source.get("remote_tree_sha256")
        or overlay.get("schema_version") != "1.5.0"
        or overlay.get("authorized_recovery_run_id") != FALLBACK_V7_RUN_ID
        or authorization.get("status") != "authorized"
        or overlay_source.get("current_association_manifest_sha256") != source_hash
        or overlay_source.get("current_association_file_sha256") != _sha256_file(source_path)
        or overlay_source.get("current_tree_sha256") != source.get("local_tree_sha256")
        or overlay.get("cumulative_gpu_accounting") != accounting
        or preflight.get("run_id") != FALLBACK_V7_RUN_ID
        or preflight.get("source_association_sha256") != source_hash
        or preflight.get("source_tree_sha256") != source.get("local_tree_sha256")
        or preflight.get("second_recovery_overlay_sha256") != overlay_hash
        or preflight.get("prior_v6_control_plane_incident_sha256")
        != v6.get("incident_manifest_sha256")
        or preflight.get("gpu_accounting_before_start") != accounting
        or preflight.get("execution_authorized") is not True
        or preflight.get("passed") is not True
        or preflight.get("gpu_allocation_performed") is not False
        or preflight.get("model_process_started") is not False
        or preflight.get("checkpoint_absent") is not True
    ):
        raise ValueError("v7 source, authorization, preflight, or initial accounting changed")


def _require_control_chain(
    *,
    invocation: Mapping[str, Any],
    ticket: Mapping[str, Any],
    ready: Mapping[str, Any],
    terminal_request: Mapping[str, Any],
    guard: Mapping[str, Any],
    closed_guard: Mapping[str, Any],
    prepare_receipt: Mapping[str, Any],
    cleanup_receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    handoff: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    result_path: Path,
    before: FallbackV7LedgerBinding,
) -> None:
    invocation_hash = invocation.get("manifest_sha256")
    ticket_hash = ticket.get("manifest_sha256")
    guard_hash = guard.get("manifest_sha256")
    execution_arguments = invocation.get("execution_arguments_sha256")
    common = (
        invocation.get("run_id") == FALLBACK_V7_RUN_ID
        and invocation.get("service_session_id") == FALLBACK_V7_RUN_ID
        and invocation.get("service_event_id") == FALLBACK_V7_SERVICE_EVENT_ID
        and Path(str(invocation.get("result_output"))).name == result_path.name
        and _microseconds_from_seconds(
            invocation.get("gpu_seconds_before_invocation"), label="v7 invocation accounting"
        )
        == before.summary.total_allocated_microseconds
        and ticket.get("run_id") == FALLBACK_V7_RUN_ID
        and ticket.get("orchestration_invocation_sha256") == invocation_hash
        and ticket.get("execution_arguments_sha256") == execution_arguments
        and ticket.get("service_session_id") == FALLBACK_V7_RUN_ID
        and ticket.get("service_event_id") == FALLBACK_V7_SERVICE_EVENT_ID
        and ready.get("run_id") == FALLBACK_V7_RUN_ID
        and ready.get("guardian_ticket_sha256") == ticket_hash
        and ready.get("orchestration_invocation_sha256") == invocation_hash
        and terminal_request.get("run_id") == FALLBACK_V7_RUN_ID
        and terminal_request.get("guardian_ticket_sha256") == ticket_hash
        and terminal_request.get("orchestration_invocation_sha256") == invocation_hash
        and terminal_request.get("orchestrator_guard_sha256") == guard_hash
        and terminal_request.get("controller_result_sha256") is None
        and guard.get("run_id") == FALLBACK_V7_RUN_ID
        and guard.get("sequence") == 1
        and guard.get("state") == "active"
        and guard.get("previous_guard_sha256") is None
        and guard.get("orchestration_invocation_sha256") == invocation_hash
        and guard.get("guardian_ticket_sha256") == ticket_hash
        and guard.get("execution_arguments_sha256") == execution_arguments
        and closed_guard.get("run_id") == FALLBACK_V7_RUN_ID
        and closed_guard.get("orchestrator_guard_sha256") == guard_hash
        and closed_guard.get("prepare_return_code") == 2
        and closed_guard.get("cleanup_return_code") == 2
        and closed_guard.get("run_return_code") is None
        and closed_guard.get("physical_shutdown_verified") is False
    )
    if not common:
        raise ValueError("v7 orchestration receipt chain changed")
    receipts = (prepare_receipt, cleanup_receipt)
    if any(
        receipt.get("run_id") != FALLBACK_V7_RUN_ID
        or receipt.get("sequence") != index
        or receipt.get("controller_stage") != stage
        or receipt.get("execution_arguments_sha256") != execution_arguments
        or receipt.get("orchestration_invocation_sha256") != invocation_hash
        or receipt.get("orchestrator_guard_sha256") != guard_hash
        for index, (receipt, stage) in enumerate(
            zip(receipts, ("prepare", "cleanup"), strict=True), start=1
        )
    ):
        raise ValueError("v7 controller receipt sequence changed")
    if (
        prepare_receipt.get("previous_controller_receipt_sha256") is not None
        or cleanup_receipt.get("previous_controller_receipt_sha256")
        != prepare_receipt.get("manifest_sha256")
    ):
        raise ValueError("v7 controller receipt lineage changed")
    required_checkpoint = {
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
    if any(checkpoint.get(key) != value for key, value in required_checkpoint.items()):
        raise ValueError("v7 terminal checkpoint changed")
    for value, stage, failure_type, receipt in (
        (handoff, "prepare", "RuntimeConfigurationError", prepare_receipt),
        (cleanup, "cleanup", "RuntimeError", cleanup_receipt),
    ):
        if (
            value.get("run_id") != FALLBACK_V7_RUN_ID
            or value.get("orchestration_controller_stage") != stage
            or value.get("failure_type") != failure_type
            or value.get("cleanup_failure_type") != failure_type
            or value.get("failed_call_id") != "controller-restart-prepare"
            or value.get("failure_stage") != "fallback_micro_pilot"
            or value.get("micro_pilot_passed") is not False
            or value.get("phase1_gate_passed") is not False
            or value.get("gate_passed") is not False
            or value.get("normal_acceptance_block_executed") is not False
            or value.get("completed_base_call_count") != 0
            or value.get("completed_call_ids") != []
            or value.get("accepted_micro_pilot_result") is not None
            or value.get("physical_service_state_unverified") is not True
            or value.get("vllm_service_stopped") is not False
            or value.get("orchestration_invocation_sha256") != invocation_hash
            or value.get("orchestrator_guard_sha256") != guard_hash
            or value.get("controller_process_receipt_sha256")
            != receipt.get("manifest_sha256")
            or value.get("execution_arguments_sha256") != execution_arguments
        ):
            raise ValueError(f"v7 {stage} failure result changed")


def _extract_final_accounting(
    ledger_path: Path,
) -> tuple[
    FallbackV7FailureEvent,
    FallbackV7ServiceAccounting,
    FallbackV7ManualStopEvidence,
]:
    with ReadOnlyLedger(ledger_path) as ledger:
        events = [
            event
            for event in ledger.gpu_events()
            if event.event_id == FALLBACK_V7_SERVICE_EVENT_ID
        ]
        sessions = [
            session
            for session in ledger.gpu_service_sessions()
            if session.service_session_id == FALLBACK_V7_SERVICE_EVENT_ID
        ]
        journals = [
            journal
            for journal in ledger.gpu_service_journal_records()
            if journal.service_session_id == FALLBACK_V7_SERVICE_EVENT_ID
        ]
    if len(events) != 1 or len(sessions) != 1 or not journals:
        raise ValueError("v7 terminal ledger lacks one exact event and service session")
    event = events[0]
    service = sessions[0]
    journal = journals[-1]
    try:
        event_details = json.loads(event.details_json)
        service_details = json.loads(service.details_json)
    except json.JSONDecodeError as exc:
        raise ValueError("v7 terminal GPU accounting details are invalid JSON") from exc
    if not isinstance(event_details, dict) or not isinstance(service_details, dict):
        raise ValueError("v7 terminal GPU accounting details must be objects")
    required_manual = {
        "manual_exact_process_group_stop": True,
        "pid_absence_verified": True,
        "process_group_absence_verified": True,
        "gpu_process_absence_verified": True,
        "endpoint_absence_verified": True,
        "recovered_from_open_service_journal": True,
        "stop_signal": "SIGTERM",
    }
    if any(service_details.get(key) != value for key, value in required_manual.items()):
        raise ValueError("v7 manual stop evidence changed")
    if (
        event.event_kind is not GpuEventKind.FAILURE
        or event.allocated_microseconds != EXPECTED_CLASSIFIED_MICROSECONDS
        or event.succeeded is not False
        or event_details.get("intended_event_kind") != "gpu_session_start"
        or event_details.get("exception_type") != "RuntimeConfigurationError"
        or _microseconds_from_seconds(
            event_details.get("admitted_maximum_seconds"), label="v7 startup watchdog"
        )
        != 300_000_000
        or service.service_microseconds != EXPECTED_SERVICE_MICROSECONDS
        or service.classified_event_microseconds != EXPECTED_CLASSIFIED_MICROSECONDS
        or service.overhead_microseconds != EXPECTED_OVERHEAD_MICROSECONDS
        or service.details_json == "{}"
        or service_details.get("accounting_method")
        != "conservative_service_journal_recovery"
        or service_details.get("source_run_id") != FALLBACK_V7_RUN_ID
        or journal.sequence != 45
        or journal.state.value != "recovered"
        or journal.elapsed_microseconds != EXPECTED_SERVICE_MICROSECONDS
    ):
        raise ValueError("v7 exact event or service accounting changed")
    identity_values = {
        name: service_details.get(name)
        for name in (
            "service_pid",
            "process_group_id",
            "process_session_id",
            "process_start_ticks",
        )
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in identity_values.values()
    ):
        raise ValueError("v7 operator-observed process identity is invalid")
    identity = FallbackV7OperatorObservedIdentity(
        observation_basis="operator_observed_then_bound_in_terminal_ledger_recovery",
        service_pid=identity_values["service_pid"],
        process_group_id=identity_values["process_group_id"],
        process_session_id=identity_values["process_session_id"],
        process_start_ticks=identity_values["process_start_ticks"],
        process_cmdline_sha256=service_details.get("process_cmdline_sha256"),
        service_instance_token_sha256=service_details.get("service_instance_token_sha256"),
        command_line_public=False,
        remote_absolute_paths_public=False,
    )
    failure = FallbackV7FailureEvent(
        event_id=event.event_id,
        event_kind=event.event_kind.value,
        allocated_microseconds=event.allocated_microseconds,
        started_at=event.started_at,
        ended_at=event.ended_at,
        succeeded=event.succeeded,
        intended_event_kind=event_details["intended_event_kind"],
        exception_type=event_details["exception_type"],
        admitted_maximum_microseconds=300_000_000,
    )
    service_record = FallbackV7ServiceAccounting(
        service_session_id=service.service_session_id,
        session_id=service.session_id,
        service_microseconds=service.service_microseconds,
        classified_event_microseconds=service.classified_event_microseconds,
        overhead_microseconds=service.overhead_microseconds,
        started_at=service.started_at,
        ended_at=service.ended_at,
        accounting_method=service_details["accounting_method"],
        recovery_journal_sequence=journal.sequence,
        recovery_journal_state=journal.state.value,
    )
    stopped_at = service_details.get("stop_verified_at")
    if not isinstance(stopped_at, str):
        raise ValueError("v7 manual stop timestamp is absent")
    manual = FallbackV7ManualStopEvidence(
        identity=identity,
        stop_signal=service_details["stop_signal"],
        stopped_at=stopped_at,
        manual_exact_process_group_stop=service_details["manual_exact_process_group_stop"],
        pid_absence_verified=service_details["pid_absence_verified"],
        process_group_absence_verified=service_details["process_group_absence_verified"],
        gpu_process_absence_verified=service_details["gpu_process_absence_verified"],
        endpoint_absence_verified=service_details["endpoint_absence_verified"],
        recovered_from_open_service_journal=service_details[
            "recovered_from_open_service_journal"
        ],
        evidence_source="terminal_ledger_service_details_and_bound_vllm_log",
    )
    return failure, service_record, manual


def build_fallback_v7_runtime_incident(
    *,
    run_id: str,
    source_association_path: Path,
    authorization_overlay_path: Path,
    preflight_path: Path,
    controller_handoff_path: Path,
    orphan_cleanup_path: Path,
    restricted_run_root: Path,
    result_path: Path,
    ledger_before_v7_path: Path,
    ledger_before_manual_recovery_path: Path,
    terminal_ledger_path: Path,
    audited_at: datetime,
) -> FallbackV7RuntimeIncident:
    """Build the v7 runtime incident from exact public and restricted evidence."""

    if run_id != FALLBACK_V7_RUN_ID:
        raise ValueError("v7 incident builder is restricted to the exact v7 run")
    if audited_at.tzinfo is None or audited_at.utcoffset() is None:
        raise ValueError("incident audit timestamp must be timezone-aware")
    run_root = Path(restricted_run_root).absolute()
    _require_no_symlink_ancestry(run_root, label="restricted v7 run root")
    if not run_root.resolve(strict=True).is_dir():
        raise ValueError("restricted v7 run root must be a directory")
    run_root = run_root.resolve(strict=True)
    observed_names = tuple(sorted(item.name for item in run_root.iterdir()))
    if observed_names != EXPECTED_RUN_ROOT_BASENAMES:
        raise ValueError("restricted v7 run root is not the exact preserved artifact set")

    source_path, source = _load_self_hashed_json(
        source_association_path,
        label="v7 source association",
        expected_kind="local_remote_source_tree_association",
    )
    overlay_path, overlay = _load_self_hashed_json(
        authorization_overlay_path,
        label="v7 authorization overlay",
        expected_kind="phase1_fallback_second_recovery_overlay",
    )
    preflight_resolved, preflight = _load_self_hashed_json(
        preflight_path,
        label="v7 execution preflight",
        expected_kind="phase1_fallback_execution_preflight",
    )
    handoff_path, handoff = _load_self_hashed_json(
        controller_handoff_path,
        label="v7 controller handoff failure",
        expected_kind="phase1_fallback_micro_pilot_result",
    )
    cleanup_path, cleanup = _load_self_hashed_json(
        orphan_cleanup_path,
        label="v7 orphan cleanup failure",
        expected_kind="phase1_fallback_micro_pilot_result",
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
        "ready": (
            "checkpoint.json.guardian-ready-000001.json",
            "fallback_service_guardian_ready",
        ),
        "terminal_request": (
            "checkpoint.json.guardian-terminal-request.json",
            "fallback_guardian_terminal_request",
        ),
        "guard": (
            "checkpoint.json.orchestrator-guard-000001.json",
            "fallback_controller_orchestrator_guard",
        ),
        "closed_guard": (
            ".checkpoint.json.orchestrator-guard-000001.json.closed.json",
            "fallback_controller_orchestrator_closed",
        ),
        "prepare_receipt": (
            "checkpoint.json.internal-controller-000001.json",
            "fallback_internal_controller_receipt",
        ),
        "cleanup_receipt": (
            "checkpoint.json.internal-controller-000002.json",
            "fallback_internal_controller_receipt",
        ),
    }
    loaded = {
        label: _load_self_hashed_json(
            run_root / basename,
            label=f"v7 {label}",
            expected_kind=kind,
        )
        for label, (basename, kind) in specs.items()
    }
    invocation_path, invocation = loaded["invocation"]
    ticket_path, ticket = loaded["ticket"]
    ready_path, ready = loaded["ready"]
    terminal_request_path, terminal_request = loaded["terminal_request"]
    guard_path, guard = loaded["guard"]
    closed_guard_path, closed_guard = loaded["closed_guard"]
    prepare_receipt_path, prepare_receipt = loaded["prepare_receipt"]
    cleanup_receipt_path, cleanup_receipt = loaded["cleanup_receipt"]
    checkpoint_path, checkpoint = _load_json_object(
        run_root / "checkpoint.json", label="v7 terminal checkpoint"
    )

    guardian_log_path = _require_regular_file(
        run_root / "checkpoint.json.guardian.log", label="v7 guardian log"
    )
    orchestrator_log_path = _require_regular_file(
        run_root / "orchestrator.20260905T174607Z.log", label="v7 orchestrator log"
    )
    vllm_log_path = _require_regular_file(
        run_root / f"{run_id}.vllm.log", label="v7 vLLM log"
    )
    try:
        guardian_log = guardian_log_path.read_text(encoding="utf-8")
        orchestrator_log = orchestrator_log_path.read_text(encoding="utf-8")
        vllm_log = vllm_log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("v7 diagnostic logs are not readable UTF-8") from exc
    if (
        "live vLLM lease lacks an exact recoverable process identity" not in guardian_log
        or "fallback service adoption failed and physical shutdown was not verified"
        not in guardian_log
        or "RuntimeError: fallback prepare controller failed" not in orchestrator_log
        or "Starting vLLM API server 0 on http://127.0.0.1:8000" not in vllm_log
        or "Shutting down FastAPI HTTP server" not in vllm_log
    ):
        raise ValueError("v7 diagnostic evidence markers changed")

    before = _ledger_binding(ledger_before_v7_path, label="pre-v7 ledger snapshot")
    pre_recovery = _ledger_binding(
        ledger_before_manual_recovery_path,
        label="pre-manual-recovery ledger snapshot",
    )
    terminal_ledger_resolved = _require_regular_file(
        terminal_ledger_path, label="terminal v7 ledger"
    )
    terminal = _ledger_binding(terminal_ledger_resolved, label="terminal v7 ledger")
    failure_event, service, manual_stop = _extract_final_accounting(
        terminal_ledger_resolved
    )
    accounting = FallbackV7Accounting(
        before_v7=before,
        before_manual_recovery=pre_recovery,
        terminal=terminal,
        failure_event=failure_event,
        service=service,
        delta=FallbackV7AccountingDelta(
            allocated_gpu_microseconds=EXPECTED_SERVICE_MICROSECONDS,
            classified_failure_microseconds=EXPECTED_CLASSIFIED_MICROSECONDS,
            conservative_service_overhead_microseconds=EXPECTED_OVERHEAD_MICROSECONDS,
            gpu_events=1,
            service_sessions=1,
            attempts=0,
            inference_model_calls=0,
            artifacts=0,
            storage_samples=8,
            resource_samples=5,
            allocation_journal_rows=46,
            service_journal_rows=46,
            accepted_outputs=0,
        ),
    )
    _require_source_chain(
        source_path=source_path,
        source=source,
        overlay=overlay,
        preflight=preflight,
        before=before,
    )
    result = Path(os.path.abspath(os.fspath(result_path)))
    _require_no_symlink_ancestry(result, label="v7 public final result")
    _require_control_chain(
        invocation=invocation,
        ticket=ticket,
        ready=ready,
        terminal_request=terminal_request,
        guard=guard,
        closed_guard=closed_guard,
        prepare_receipt=prepare_receipt,
        cleanup_receipt=cleanup_receipt,
        checkpoint=checkpoint,
        handoff=handoff,
        cleanup=cleanup,
        result_path=result,
        before=before,
    )
    absent_paths = (
        run_root / "checkpoint.json.guardian-controller-takeover.json",
        run_root / "checkpoint.json.guardian-result.json",
        run_root / "checkpoint.json.service",
        result,
    )
    absent_names = tuple(path.name for path in absent_paths)
    if absent_names != EXPECTED_ABSENT_BASENAMES:
        raise AssertionError("internal v7 absence inventory changed")
    if any(path.exists() or path.is_symlink() for path in absent_paths):
        raise ValueError("a required v7 terminal-state path exists")

    resource_samples = _json_mapping(handoff.get("runtime"), label="v7 handoff runtime").get(
        "resource_samples"
    )
    if not isinstance(resource_samples, list) or len(resource_samples) != 5:
        raise ValueError("v7 completed startup resource-sample count changed")
    if any(
        not isinstance(sample, Mapping)
        or sample.get("sample_id")
        != f"{FALLBACK_V7_SERVICE_EVENT_ID}-startup-{index:06d}"
        or sample.get("violations") != []
        for index, sample in enumerate(resource_samples, start=1)
    ):
        raise ValueError("v7 startup resource-sample sequence changed")
    endpoint_healthy_at = _endpoint_healthy_timestamp(
        vllm_log,
        failure_event=failure_event,
    )
    if audited_at < manual_stop.stopped_at:
        raise ValueError("v7 incident audit predates the verified manual stop")

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
        "kind": FALLBACK_V7_RUNTIME_INCIDENT_KIND,
        "run_id": run_id,
        "audited_at": audited_at,
        "source_and_authorization": FallbackV7SourceAndAuthorization(
            source_revision=FALLBACK_V7_SOURCE_REVISION,
            source_git_commit=source["git_commit"],
            source_tree_sha256=source["local_tree_sha256"],
            source_association=_self_hashed_binding(source_path, source),
            authorization_overlay=_self_hashed_binding(overlay_path, overlay),
            execution_preflight=_self_hashed_binding(preflight_resolved, preflight),
        ),
        "public_failure_evidence": FallbackV7PublicFailureEvidence(
            controller_handoff=_self_hashed_binding(handoff_path, handoff),
            orphan_cleanup=_self_hashed_binding(cleanup_path, cleanup),
        ),
        "restricted_evidence": FallbackV7RestrictedEvidence(
            orchestration_invocation=_self_hashed_binding(invocation_path, invocation),
            guardian_ticket=_self_hashed_binding(ticket_path, ticket),
            guardian_ready=_self_hashed_binding(ready_path, ready),
            guardian_terminal_request=_self_hashed_binding(
                terminal_request_path, terminal_request
            ),
            orchestrator_guard=_self_hashed_binding(guard_path, guard),
            closed_orchestrator_guard=_self_hashed_binding(
                closed_guard_path, closed_guard
            ),
            prepare_controller_receipt=_self_hashed_binding(
                prepare_receipt_path, prepare_receipt
            ),
            cleanup_controller_receipt=_self_hashed_binding(
                cleanup_receipt_path, cleanup_receipt
            ),
            checkpoint=_opaque_binding(checkpoint_path, label="v7 terminal checkpoint"),
            guardian_log=_opaque_binding(guardian_log_path, label="v7 guardian log"),
            orchestrator_log=_opaque_binding(
                orchestrator_log_path, label="v7 orchestrator log"
            ),
            vllm_log=_opaque_binding(vllm_log_path, label="v7 vLLM log"),
            run_root_inventory_sha256=_canonical_sha256(run_root_inventory),
        ),
        "orchestration_identity": FallbackV7OrchestrationIdentity(
            execution_arguments_sha256=invocation["execution_arguments_sha256"],
            orchestration_invocation_manifest_sha256=invocation["manifest_sha256"],
            guardian_ticket_manifest_sha256=ticket["manifest_sha256"],
            guardian_ready_manifest_sha256=ready["manifest_sha256"],
            guardian_terminal_request_manifest_sha256=terminal_request[
                "manifest_sha256"
            ],
            orchestrator_guard_manifest_sha256=guard["manifest_sha256"],
            closed_orchestrator_guard_manifest_sha256=closed_guard["manifest_sha256"],
            prepare_controller_receipt_manifest_sha256=prepare_receipt[
                "manifest_sha256"
            ],
            cleanup_controller_receipt_manifest_sha256=cleanup_receipt[
                "manifest_sha256"
            ],
            guardian_command_sha256=ticket["guardian_command_sha256"],
            orchestrator_command_sha256=guard["orchestrator_command_sha256"],
            service_session_id=FALLBACK_V7_RUN_ID,
            service_event_id=FALLBACK_V7_SERVICE_EVENT_ID,
        ),
        "accounting": accounting,
        "failure_sequence": FallbackV7FailureSequence(
            safe_error_class=FALLBACK_V7_SAFE_ERROR_CLASS,
            root_cause=FALLBACK_V7_ROOT_CAUSE,
            root_cause_basis="operator_observation_and_source_bound_control_flow",
            failure_stage="service_start_post_health_watchdog_stop",
            healthy_endpoint_before_failure=True,
            endpoint_healthy_at=endpoint_healthy_at,
            failure_event_ended_at=failure_event.ended_at,
            completed_startup_resource_samples=5,
            in_flight_storage_resource_sample_operator_observed=True,
            exact_live_lease_recovery_failed=True,
            automated_physical_shutdown_verification_failed=True,
            prepare_controller_failure_type="RuntimeConfigurationError",
            cleanup_controller_failure_type="RuntimeError",
            guardian_terminal_failure_type="RuntimeError",
            scientific_inference_reached=False,
            command_line_public=False,
            full_log_text_public=False,
        ),
        "manual_stop": manual_stop,
        "absence_inventory": FallbackV7AbsenceInventory(
            observed_run_root_basenames=observed_names,
            absent_basenames=absent_names,
            all_required_paths_absent=True,
        ),
        "terminal_state": FallbackV7TerminalState(
            public_final_result_present=False,
            accepted_output_count=0,
            inference_attempt_count_delta=0,
            inference_model_call_count_delta=0,
            phase1_gate_passed=False,
            guardian_result_present=False,
            automated_cleanup_succeeded=False,
            manual_physical_stop_verified=True,
            endpoint_live=False,
            gpu_process_live=False,
            exact_service_process_live=False,
            unresolved_gpu_allocation_count=0,
            unresolved_gpu_service_count=0,
            terminal_ledger_verified=True,
            resume_allowed=False,
            source_bound_v7_resume_permitted=False,
            fresh_repaired_source_required=True,
        ),
    }
    return FallbackV7RuntimeIncident.model_validate(
        {**payload, "manifest_sha256": _canonical_sha256(payload)}
    )


def load_fallback_v7_runtime_incident(path: Path) -> FallbackV7RuntimeIncident:
    """Load and fully validate one regular self-hashed v7 incident artifact."""

    resolved = _require_regular_file(path, label="fallback-v7 runtime incident")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("fallback-v7 incident is not valid UTF-8 JSON") from exc
    try:
        return FallbackV7RuntimeIncident.model_validate(value)
    except ValidationError as exc:
        raise ValueError("fallback-v7 incident contract is invalid") from exc


def validate_fallback_v7_runtime_incident(
    path: Path,
    *,
    expected_source_association_manifest_sha256: str,
    expected_overlay_manifest_sha256: str,
    expected_preflight_manifest_sha256: str,
    expected_terminal_ledger_file_sha256: str,
) -> FallbackV7RuntimeIncident:
    """Validate v7 against the exact identities frozen by a later lineage record."""

    incident = load_fallback_v7_runtime_incident(path)
    source = incident.source_and_authorization
    if (
        source.source_association.manifest_sha256
        != expected_source_association_manifest_sha256
        or source.authorization_overlay.manifest_sha256
        != expected_overlay_manifest_sha256
        or source.execution_preflight.manifest_sha256
        != expected_preflight_manifest_sha256
        or incident.accounting.terminal.file_sha256
        != expected_terminal_ledger_file_sha256
    ):
        raise ValueError("fallback-v7 incident differs from its expected identity")
    return incident


def write_fallback_v7_runtime_incident(
    path: Path,
    incident: FallbackV7RuntimeIncident,
) -> None:
    """Append one canonical public incident; exact replay is idempotent."""

    destination = Path(path).absolute()
    _require_no_symlink_ancestry(destination, label="v7 incident output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestry(destination, label="v7 incident output")
    payload = (canonical_json(incident) + "\n").encode("utf-8")
    if destination.exists():
        if (
            destination.is_file()
            and not destination.is_symlink()
            and destination.read_bytes() == payload
        ):
            return
        raise FileExistsError("append-only v7 incident output already differs")
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
    "FALLBACK_V7_ROOT_CAUSE",
    "FALLBACK_V7_RUNTIME_INCIDENT_KIND",
    "FALLBACK_V7_RUN_ID",
    "FallbackV7RuntimeIncident",
    "build_fallback_v7_runtime_incident",
    "load_fallback_v7_runtime_incident",
    "validate_fallback_v7_runtime_incident",
    "write_fallback_v7_runtime_incident",
]
