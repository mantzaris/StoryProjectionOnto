"""Restricted post-development binding for the held-out control template.

The tracked held-out configuration is part of the source tree accepted by the
model pilot, so it cannot be edited after development merely to paste in the
new result hash.  This module materializes that dynamic predecessor in a
restricted, content-addressed binding.  It deliberately does not inspect any
held-out runtime stage or scorer artifact; the independent-review gate remains
the first operation allowed to open the held-out plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ImmutableRecord,
    RunOutcome,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.development_artifacts import DevelopmentCallAuditReceipt
from story_projection_onto.development_runtime import (
    DevelopmentExecutionResult,
    DevelopmentPhase,
    RequestStartState,
)
from story_projection_onto.fallback_acceptance import (
    DevelopmentContinuationReceipt,
    validate_source_association,
)
from story_projection_onto.held_out_primary import (
    HeldOutControlConfiguration,
    HeldOutControlError,
    load_held_out_control_configuration,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    GpuEventKind,
    Ledger,
    ModelBackend,
    ReleaseClass,
    StoreError,
)

DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH = Path(
    "artifacts/restricted/held_out_primary/runtime_binding.json"
)
FROZEN_HELD_OUT_PRODUCTION_FACTORY = (
    "story_projection_onto.held_out_factory:create_frozen_production_held_out_bundle"
)


class DevelopmentPredecessorLedgerBinding(ImmutableRecord):
    """Exact accounting prefix and 24-call development ledger lineage."""

    development_execution_id: str = Field(min_length=1)
    development_execution_result_hash: Sha256Digest
    predecessor_gpu_event_count: int = Field(ge=1)
    predecessor_gpu_event_ids: tuple[str, ...]
    predecessor_gpu_event_chain_hash: Sha256Digest
    predecessor_total_allocated_microseconds: int = Field(ge=1)
    predecessor_gpu_summary_hash: Sha256Digest
    predecessor_gpu_service_session_count: int = Field(ge=0)
    predecessor_gpu_service_session_rows: tuple[tuple[str, Sha256Digest], ...]
    predecessor_gpu_service_sessions_hash: Sha256Digest
    predecessor_gpu_service_journal_rows: tuple[tuple[str, Sha256Digest], ...]
    predecessor_gpu_service_journal_hash: Sha256Digest
    development_handoff_allocated_microseconds: int = Field(ge=1)
    development_gpu_event_ids: tuple[str, ...]
    development_gpu_events_hash: Sha256Digest
    development_model_call_ids: tuple[str, ...]
    development_model_calls_hash: Sha256Digest
    development_call_lineage_hash: Sha256Digest

    @model_validator(mode="after")
    def exact_inventory_sizes(self) -> Self:
        if (
            len(self.predecessor_gpu_event_ids) != self.predecessor_gpu_event_count
            or len(set(self.predecessor_gpu_event_ids))
            != self.predecessor_gpu_event_count
            or len(self.development_gpu_event_ids) != 24
            or len(set(self.development_gpu_event_ids)) != 24
            or len(self.development_model_call_ids) != 24
            or len(set(self.development_model_call_ids)) != 24
            or len(self.predecessor_gpu_service_session_rows)
            != self.predecessor_gpu_service_session_count
            or len({item[0] for item in self.predecessor_gpu_service_session_rows})
            != len(self.predecessor_gpu_service_session_rows)
            or len({item[0] for item in self.predecessor_gpu_service_journal_rows})
            != len(self.predecessor_gpu_service_journal_rows)
        ):
            raise ValueError("development predecessor ledger inventory changed")
        if (
            canonical_sha256(self.predecessor_gpu_service_session_rows)
            != self.predecessor_gpu_service_sessions_hash
            or canonical_sha256(self.predecessor_gpu_service_journal_rows)
            != self.predecessor_gpu_service_journal_hash
        ):
            raise ValueError("development predecessor service-ledger digest changed")
        return self


def _gpu_summary_payload(ledger: Ledger) -> dict[str, object]:
    summary = ledger.gpu_summary()
    return {
        "total_allocated_microseconds": summary.total_allocated_microseconds,
        "event_count": summary.event_count,
        "service_session_count": summary.service_session_count,
        "by_kind_microseconds": {
            kind.value: microseconds for kind, microseconds in summary.by_kind_microseconds
        },
    }


def verify_predecessor_gpu_event_chain(
    ledger: Ledger,
    expected: DevelopmentPredecessorLedgerBinding,
) -> bool:
    """Authenticate the frozen event prefix; return whether successors exist."""

    events = ledger.gpu_events()
    sessions = ledger.gpu_service_sessions()
    journal = ledger.gpu_service_journal_records()
    if len(events) < expected.predecessor_gpu_event_count:
        raise HeldOutControlError("predecessor GPU event prefix is incomplete")
    prefix = events[: expected.predecessor_gpu_event_count]
    if (
        tuple(item.event_id for item in prefix) != expected.predecessor_gpu_event_ids
        or canonical_sha256(tuple(asdict(item) for item in prefix))
        != expected.predecessor_gpu_event_chain_hash
    ):
        raise HeldOutControlError("predecessor GPU event chain differs from its binding")
    summary_payload = _gpu_summary_payload(ledger)
    total = cast(int, summary_payload["total_allocated_microseconds"])
    session_count = cast(int, summary_payload["service_session_count"])
    if (
        total < expected.predecessor_total_allocated_microseconds
        or session_count < expected.predecessor_gpu_service_session_count
    ):
        raise HeldOutControlError("cumulative GPU ledger regressed below its predecessor")
    expected_session_ids = {
        row_id for row_id, _ in expected.predecessor_gpu_service_session_rows
    }
    actual_predecessor_sessions = tuple(
        (item.service_session_id, canonical_sha256(asdict(item)))
        for item in sessions
        if item.service_session_id in expected_session_ids
    )
    expected_journal_ids = {
        row_id for row_id, _ in expected.predecessor_gpu_service_journal_rows
    }
    actual_predecessor_journal = tuple(
        (item.journal_id, canonical_sha256(asdict(item)))
        for item in journal
        if item.journal_id in expected_journal_ids
    )
    if actual_predecessor_sessions != expected.predecessor_gpu_service_session_rows:
        raise HeldOutControlError(
            "predecessor GPU service-session rows differ from their binding"
        )
    if actual_predecessor_journal != expected.predecessor_gpu_service_journal_rows:
        raise HeldOutControlError(
            "predecessor GPU service-journal rows differ from their binding"
        )
    event_successors = len(events) != expected.predecessor_gpu_event_count
    session_successors = len(sessions) != len(
        expected.predecessor_gpu_service_session_rows
    )
    journal_successors = len(journal) != len(
        expected.predecessor_gpu_service_journal_rows
    )
    has_successors = event_successors or session_successors or journal_successors
    if not has_successors and (
        total != expected.predecessor_total_allocated_microseconds
        or canonical_sha256(summary_payload) != expected.predecessor_gpu_summary_hash
    ):
        raise HeldOutControlError("predecessor GPU summary differs from its binding")
    return has_successors


class HeldOutRuntimeBinding(ImmutableRecord):
    """Dynamic predecessor hashes attached to one immutable source template."""

    binding_id: str = Field(min_length=1)
    control_configuration_path: str
    control_configuration_file_sha256: Sha256Digest
    control_configuration_hash: Sha256Digest
    template_call_manifest_hash: Sha256Digest
    post_development_call_manifest_hash: Sha256Digest
    runtime_control_configuration_hash: Sha256Digest
    production_adapter_factory: Literal[
        "story_projection_onto.held_out_factory:create_frozen_production_held_out_bundle"
    ]
    fallback_result_path: str
    fallback_result_file_sha256: Sha256Digest
    fallback_result_manifest_sha256: Sha256Digest
    fallback_run_id: str = Field(min_length=1)
    development_execution_result_path: str
    development_execution_result_file_sha256: Sha256Digest
    development_execution_result_hash: Sha256Digest
    development_execution_id: str = Field(min_length=1)
    development_execution_manifest_hash: Sha256Digest
    development_prequery_inputs_hash: Sha256Digest
    development_continuation_receipt_hash: Sha256Digest
    development_predecessor_ledger: DevelopmentPredecessorLedgerBinding
    selected_model_freeze_path: str
    selected_model_freeze_file_sha256: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    source_association_path: str
    source_association_file_sha256: Sha256Digest
    source_association_manifest_sha256: Sha256Digest
    source_tree_sha256: Sha256Digest
    development_gate_passed: Literal[True] = True
    development_forecast_admitted: Literal[True] = True
    model_service_stopped_after_development: Literal[True] = True
    held_out_review_still_required: Literal[True] = True
    created_at: AwareDatetime

    @model_validator(mode="after")
    def safe_restricted_paths(self) -> Self:
        for value, label in (
            (self.control_configuration_path, "control configuration"),
            (self.fallback_result_path, "fallback result"),
            (self.development_execution_result_path, "development result"),
            (self.selected_model_freeze_path, "selected-model freeze"),
            (self.source_association_path, "source association"),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError(f"{label} path must be safe and repository-relative")
        if not self.development_execution_result_path.startswith("artifacts/restricted/"):
            raise ValueError("development result must remain in the restricted namespace")
        if not self.selected_model_freeze_path.startswith("artifacts/restricted/"):
            raise ValueError("selected-model freeze must remain in the restricted namespace")
        return self


def capture_development_predecessor_ledger(
    *,
    artifacts: ArtifactStore,
    development: DevelopmentExecutionResult,
    predecessor_gpu_event_count: int | None = None,
) -> DevelopmentPredecessorLedgerBinding:
    """Replay the exact completed development accounting and CAS/ledger bridge."""

    if (
        development.phase is not DevelopmentPhase.COMPLETED
        or not development.gate.passed
        or len(development.itt_records) != 24
    ):
        raise HeldOutControlError(
            "predecessor ledger capture requires the passing 24-call development result"
        )
    ledger = artifacts.ledger
    all_events = ledger.gpu_events()
    all_service_sessions = ledger.gpu_service_sessions()
    all_service_journal = ledger.gpu_service_journal_records()
    summary_payload = _gpu_summary_payload(ledger)
    event_count = (
        len(all_events)
        if predecessor_gpu_event_count is None
        else predecessor_gpu_event_count
    )
    if event_count <= 0 or len(all_events) < event_count:
        raise HeldOutControlError("predecessor GPU event prefix is incomplete")
    predecessor_events = all_events[:event_count]
    predecessor_event_ids = tuple(item.event_id for item in predecessor_events)
    if len(set(predecessor_event_ids)) != len(predecessor_event_ids):
        raise HeldOutControlError("predecessor GPU event prefix contains duplicate IDs")
    events_by_id = {item.event_id: item for item in predecessor_events}
    development_events = []
    development_model_calls = []
    lineage = []
    for ordinal, row in enumerate(development.itt_records, start=1):
        if (
            row.ordinal != ordinal
            or row.request_start_state is not RequestStartState.STARTED
            or row.outcome is not RunOutcome.SUCCEEDED
            or row.gpu_event_id is None
            or row.ledger_receipt_hash is None
            or row.run_condition_config_hash is None
        ):
            raise HeldOutControlError(
                "development result lacks one exact successful started call"
            )
        try:
            event = events_by_id[row.gpu_event_id]
            model_call_id = f"{development.execution_id}-{row.call_id}-model"
            model_call = ledger.get_model_call(model_call_id)
            attempt_chain = ledger.attempt_lineage(model_call.attempt_id)
            job = ledger.get_job(model_call.job_id)
            receipt_record = ledger.get_artifact(row.ledger_receipt_hash)
            receipt = DevelopmentCallAuditReceipt.model_validate_json(
                artifacts.blobs.read_bytes(receipt_record)
            )
            response_record = ledger.get_artifact(cast(str, row.response_artifact_hash))
            artifacts.blobs.read_bytes(
                response_record,
                allow_restricted=response_record.release_class is ReleaseClass.RESTRICTED,
            )
            assert receipt.rendered_model_request is not None
            rendered_record = ledger.get_artifact(
                receipt.rendered_model_request.artifact_hash
            )
            artifacts.blobs.read_bytes(
                rendered_record,
                allow_restricted=rendered_record.release_class is ReleaseClass.RESTRICTED,
            )
            validations = tuple(
                ledger.get_validation(validation_id)
                for validation_id in receipt.ledger_validation_ids
            )
        except (KeyError, OSError, StoreError, ValueError) as error:
            raise HeldOutControlError(
                "development result lost its exact CAS/ledger lineage"
            ) from error
        if (
            event.event_id != row.gpu_event_id
            or event.job_id != model_call.job_id
            or event.attempt_id != model_call.attempt_id
            or event.succeeded is not True
            or event.event_kind
            is not (
                GpuEventKind.REPAIR
                if row.call_id == "dev-repair-probe-u04-q02"
                else GpuEventKind.INFERENCE
            )
            or model_call.model_call_id != model_call_id
            or model_call.backend is not ModelBackend.VLLM_GPU
            or model_call.gpu_event_id != event.event_id
            or model_call.request_hash != attempt_chain[-1].input_hash
            or model_call.request_hash
            != receipt.rendered_model_request.logical_content_hash
            or model_call.response_artifact_hash != row.response_artifact_hash
            or model_call.allocated_gpu_microseconds != event.allocated_microseconds
            or not math.isclose(
                event.allocated_seconds,
                row.allocated_gpu_seconds,
                abs_tol=1e-6,
            )
            or model_call.prompt_tokens != row.prompt_tokens
            or model_call.completion_tokens != row.completion_tokens
            or not model_call.successful
            or attempt_chain[-1].config_hash != row.run_condition_config_hash
            or attempt_chain[-1].job_id != job.job_id
            or attempt_chain[-1].attempt_id != model_call.attempt_id
            or attempt_chain[-1].attempt_kind
            is not (
                AttemptKind.REPAIR
                if row.call_id == "dev-repair-probe-u04-q02"
                else AttemptKind.BASE
            )
            or receipt_record.release_class is not ReleaseClass.PUBLIC
            or receipt_record.media_type
            != "application/vnd.story-projection.development-call-receipt+json"
            or receipt.ordinal != row.ordinal
            or receipt.call_id != row.call_id
            or receipt.condition is not row.condition
            or receipt.outcome is not row.outcome
            or not receipt.request_started
            or receipt.model_call_id != model_call_id
            or receipt.job_id != model_call.job_id
            or receipt.attempt_id != model_call.attempt_id
            or receipt.raw_response_artifact_hash != row.response_artifact_hash
            or receipt.run_condition_config is None
            or receipt.run_condition_config.logical_content_hash
            != row.run_condition_config_hash
            or tuple(item.validation_id for item in validations)
            != receipt.ledger_validation_ids
            or any(
                item.job_id != model_call.job_id
                or item.attempt_id != model_call.attempt_id
                for item in validations
            )
        ):
            raise HeldOutControlError(
                "development GPU event/model-call lineage differs from its result"
            )
        development_events.append(event)
        development_model_calls.append(model_call)
        lineage.append(
            {
                "itt_record_hash": row.content_hash,
                "receipt_artifact": asdict(receipt_record),
                "receipt_hash": receipt.content_hash,
                "event": asdict(event),
                "model_call": asdict(model_call),
                "attempt_chain": tuple(asdict(item) for item in attempt_chain),
                "job": asdict(job),
                "validations": tuple(asdict(item) for item in validations),
                "rendered_request_artifact": asdict(rendered_record),
                "response_artifact": asdict(response_record),
            }
        )
    development_handoff = round(
        development.forecast.actual_allocated_seconds_after * 1_000_000
    )
    predecessor_total = cast(int, summary_payload["total_allocated_microseconds"])
    if predecessor_total < development_handoff:
        raise HeldOutControlError(
            "predecessor accounting regressed below the development handoff"
        )
    service_rows = tuple(
        (item.service_session_id, canonical_sha256(asdict(item)))
        for item in all_service_sessions
    )
    journal_rows = tuple(
        (item.journal_id, canonical_sha256(asdict(item)))
        for item in all_service_journal
    )
    return DevelopmentPredecessorLedgerBinding(
        development_execution_id=development.execution_id,
        development_execution_result_hash=development.content_hash,
        predecessor_gpu_event_count=event_count,
        predecessor_gpu_event_ids=predecessor_event_ids,
        predecessor_gpu_event_chain_hash=canonical_sha256(
            tuple(asdict(item) for item in predecessor_events)
        ),
        predecessor_total_allocated_microseconds=predecessor_total,
        predecessor_gpu_summary_hash=canonical_sha256(summary_payload),
        predecessor_gpu_service_session_count=cast(
            int, summary_payload["service_session_count"]
        ),
        predecessor_gpu_service_session_rows=service_rows,
        predecessor_gpu_service_sessions_hash=canonical_sha256(service_rows),
        predecessor_gpu_service_journal_rows=journal_rows,
        predecessor_gpu_service_journal_hash=canonical_sha256(journal_rows),
        development_handoff_allocated_microseconds=development_handoff,
        development_gpu_event_ids=tuple(item.event_id for item in development_events),
        development_gpu_events_hash=canonical_sha256(
            tuple(asdict(item) for item in development_events)
        ),
        development_model_call_ids=tuple(
            item.model_call_id for item in development_model_calls
        ),
        development_model_calls_hash=canonical_sha256(
            tuple(asdict(item) for item in development_model_calls)
        ),
        development_call_lineage_hash=canonical_sha256(tuple(lineage)),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HeldOutControlError(f"{label} must be an existing regular file")
    return path.resolve(strict=True)


def _restricted_descendant(
    restricted_root: Path,
    candidate: Path,
    *,
    label: str,
) -> Path:
    """Reject escape and every symlink below one explicit private root."""

    if restricted_root.is_symlink() or not restricted_root.is_dir():
        raise HeldOutControlError("restricted root must be an existing real directory")
    root = restricted_root.resolve(strict=True)
    absolute = candidate if candidate.is_absolute() else root / candidate
    if ".." in absolute.parts:
        raise HeldOutControlError(f"{label} contains parent traversal")
    try:
        lexical = absolute.absolute().relative_to(root)
        resolved = absolute.resolve(strict=False)
        resolved.relative_to(root)
    except ValueError as error:
        raise HeldOutControlError(f"{label} lies outside the restricted root") from error
    if lexical == Path("."):
        raise HeldOutControlError(f"{label} must be below the restricted root")
    probe = root
    for component in lexical.parts:
        probe /= component
        if probe.is_symlink():
            raise HeldOutControlError(f"{label} has a symlinked ancestor")
    return resolved


def _load_mapping(path: Path, *, label: str) -> tuple[Path, dict[str, object]]:
    resolved = _safe_file(path, label=label)
    try:
        value = json.loads(resolved.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeldOutControlError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise HeldOutControlError(f"{label} root must be an object")
    return resolved, cast(dict[str, object], value)


def _verify_external_manifest(value: Mapping[str, object], *, label: str) -> str:
    supplied = value.get("manifest_sha256")
    payload = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise HeldOutControlError(f"{label} manifest hash does not reproduce")
    return supplied


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Create an immutable private file, or accept an exact prior materialization."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise HeldOutControlError(f"existing restricted artifact differs: {path.name}")
        return
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise HeldOutControlError(f"concurrent restricted artifact differs: {path.name}")
        else:
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def persist_restricted_error(
    *,
    restricted_root: Path,
    namespace: str,
    error: BaseException,
) -> None:
    """Best-effort private traceback retention with no public error rendering."""

    if not namespace or not namespace.replace("_", "").replace("-", "").isalnum():
        return
    try:
        root = restricted_root
        if root.is_symlink() or not root.is_dir():
            return
        root = root.resolve(strict=True)
        directory = _restricted_descendant(
            root,
            root / namespace / "control_errors",
            label="restricted error directory",
        )
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        payload = (
            json.dumps(
                {
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "traceback": "".join(traceback.format_exception(error)),
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        target = _restricted_descendant(
            root,
            directory / f"{digest}.json",
            label="restricted error record",
        )
        _atomic_write_new(target, payload)
    except Exception:
        return


def _relative_to_repository(repository: Path, path: Path, *, label: str) -> str:
    try:
        relative = path.resolve(strict=False).relative_to(repository).as_posix()
    except ValueError as error:
        raise HeldOutControlError(f"{label} lies outside the repository") from error
    if not relative.startswith("artifacts/restricted/"):
        raise HeldOutControlError(f"{label} must be inside artifacts/restricted")
    return relative


def materialize_held_out_runtime_binding(
    *,
    repository: Path,
    fallback_result_path: Path,
    source_association_path: Path,
    control_configuration_path: Path,
    development_result_path: Path,
    selected_model_freeze_path: Path,
    binding_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    restricted_root: Path,
    created_at: datetime,
) -> HeldOutRuntimeBinding:
    """Extract and bind a passing development predecessor without opening held-out data."""

    repository = repository.resolve(strict=True)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise HeldOutControlError("binding timestamp must be timezone-aware")
    control_path = (
        control_configuration_path
        if control_configuration_path.is_absolute()
        else repository / control_configuration_path
    )
    control_path = _safe_file(control_path, label="held-out control configuration")
    try:
        control_relative = control_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise HeldOutControlError(
            "held-out control configuration lies outside repository"
        ) from error
    configuration = load_held_out_control_configuration(repository, control_path)
    if configuration.production_adapter_factory != FROZEN_HELD_OUT_PRODUCTION_FACTORY:
        raise HeldOutControlError("held-out production factory is not frozen in source")
    if configuration.development_execution_result_file_sha256 != "PENDING":
        raise HeldOutControlError("tracked control must remain a post-development template")
    if configuration.expected_plan_hash == "PENDING":
        raise HeldOutControlError("tracked held-out template call manifest is not frozen")

    fallback_path, fallback = _load_mapping(
        fallback_result_path,
        label="fallback/development result",
    )
    fallback_manifest_hash = _verify_external_manifest(
        fallback,
        label="fallback/development result",
    )
    if (
        fallback.get("phase1_gate_passed") is not True
        or fallback.get("gate_passed") is not True
        or fallback.get("vllm_service_stopped") is not True
        or fallback.get("physical_service_live") is not False
    ):
        raise HeldOutControlError(
            "held-out binding requires a passing development run and verified shutdown"
        )
    run_id = fallback.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise HeldOutControlError("fallback/development result lacks its run ID")

    raw_development = fallback.get("development_execution_result")
    raw_receipt = fallback.get("development_continuation_receipt")
    raw_freeze = fallback.get("selected_model_freeze")
    if not isinstance(raw_development, Mapping):
        raise HeldOutControlError("fallback result lacks its development execution result")
    if not isinstance(raw_receipt, Mapping):
        raise HeldOutControlError("fallback result lacks its development continuation receipt")
    if not isinstance(raw_freeze, Mapping):
        raise HeldOutControlError("fallback result lacks its selected-model freeze")
    development = DevelopmentExecutionResult.model_validate(raw_development)
    receipt = DevelopmentContinuationReceipt.model_validate(raw_receipt)
    freeze = dict(raw_freeze)
    freeze_hash = _verify_external_manifest(freeze, label="selected-model freeze")
    if (
        development.phase is not DevelopmentPhase.COMPLETED
        or not development.gate.passed
        or not development.forecast.scheduled_admitted
        or not development.forecast.below_hard_stop
        or not development.service_returned_live_to_owner
        or receipt.development_result_hash != development.content_hash
        or receipt.development_execution_id != development.execution_id
        or receipt.development_execution_manifest_hash
        != development.execution_manifest_hash
        or receipt.development_prequery_inputs_hash != development.prequery_inputs_hash
        or not receipt.development_gate_passed
        or not receipt.development_forecast_admitted
        or receipt.selected_model_freeze_hash != freeze_hash
    ):
        raise HeldOutControlError("development result, continuation receipt, and gate disagree")

    association_path = _safe_file(source_association_path, label="source association")
    association = validate_source_association(association_path, source_root=repository)
    association_hash = cast(str, association["manifest_sha256"])
    source_tree_hash = cast(str, association["local_tree_sha256"])
    if (
        freeze.get("source_association_manifest_sha256") != association_hash
        or freeze.get("source_tree_sha256") != source_tree_hash
        or receipt.source_execution_hash != source_tree_hash
    ):
        raise HeldOutControlError("development outputs do not bind the current frozen source tree")
    if created_at < receipt.completed_at:
        raise HeldOutControlError("runtime binding predates the development completion receipt")

    restricted_candidate = (
        restricted_root if restricted_root.is_absolute() else repository / restricted_root
    )
    if restricted_candidate.is_symlink() or not restricted_candidate.is_dir():
        raise HeldOutControlError("restricted root must be an existing real directory")
    restricted = restricted_candidate.resolve(strict=True)
    ledger_candidate = ledger_path if ledger_path.is_absolute() else repository / ledger_path
    artifact_candidate = (
        artifact_root if artifact_root.is_absolute() else repository / artifact_root
    )
    ledger_source = _safe_file(
        _restricted_descendant(
            restricted,
            ledger_candidate,
            label="development GPU ledger",
        ),
        label="development GPU ledger",
    )
    artifact_candidate = _restricted_descendant(
        restricted,
        artifact_candidate,
        label="development CAS root",
    )
    if artifact_candidate.is_symlink() or not artifact_candidate.is_dir():
        raise HeldOutControlError("development CAS root must be an existing real directory")
    ledger = Ledger(ledger_source)
    try:
        if ledger.unresolved_gpu_service_journals():
            raise HeldOutControlError(
                "development GPU service journal is not terminal at binding"
            )
        predecessor_ledger = capture_development_predecessor_ledger(
            artifacts=ArtifactStore(BlobStore(artifact_candidate), ledger),
            development=development,
        )
        runtime = fallback.get("runtime")
        accounting = runtime.get("gpu_accounting") if isinstance(runtime, Mapping) else None
        if not isinstance(accounting, Mapping) or dict(accounting) != _gpu_summary_payload(ledger):
            raise HeldOutControlError(
                "fallback terminal GPU accounting differs from the predecessor ledger"
            )
    finally:
        ledger.close()

    try:
        fallback_relative = fallback_path.relative_to(repository).as_posix()
        association_relative = association_path.relative_to(repository).as_posix()
    except ValueError as error:
        raise HeldOutControlError(
            "fallback result and source association must remain inside the repository"
        ) from error

    development_destination = (
        development_result_path
        if development_result_path.is_absolute()
        else repository / development_result_path
    )
    freeze_destination = (
        selected_model_freeze_path
        if selected_model_freeze_path.is_absolute()
        else repository / selected_model_freeze_path
    )
    binding_destination = binding_path if binding_path.is_absolute() else repository / binding_path
    for label, destination in (
        ("development result destination", development_destination),
        ("selected-model freeze destination", freeze_destination),
        ("runtime binding destination", binding_destination),
    ):
        _restricted_descendant(restricted, destination, label=label)
    development_relative = _relative_to_repository(
        repository,
        development_destination,
        label="development result destination",
    )
    freeze_relative = _relative_to_repository(
        repository,
        freeze_destination,
        label="selected-model freeze destination",
    )
    _relative_to_repository(repository, binding_destination, label="runtime binding destination")

    development_bytes = (development.to_canonical_json() + "\n").encode("utf-8")
    freeze_bytes = (
        json.dumps(
            freeze,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_new(development_destination, development_bytes)
    _atomic_write_new(freeze_destination, freeze_bytes)

    runtime_configuration_payload = configuration.model_dump(
        mode="python",
        exclude={"content_hash"},
    )
    runtime_configuration_payload.update(
        development_execution_result_path=development_relative,
        development_execution_result_file_sha256=(
            hashlib.sha256(development_bytes).hexdigest()
        ),
        expected_plan_hash="PENDING",
    )
    runtime_configuration = HeldOutControlConfiguration.model_validate(
        runtime_configuration_payload
    )
    from story_projection_onto.held_out_primary import _derive_call_manifest

    post_development_plan_hash = _derive_call_manifest(
        repository,
        runtime_configuration,
    ).content_hash

    binding = HeldOutRuntimeBinding(
        binding_id=f"held-out-runtime-{development.content_hash[:20]}",
        control_configuration_path=control_relative,
        control_configuration_file_sha256=_file_sha256(control_path),
        control_configuration_hash=configuration.content_hash,
        template_call_manifest_hash=configuration.expected_plan_hash,
        post_development_call_manifest_hash=post_development_plan_hash,
        runtime_control_configuration_hash=runtime_configuration.content_hash,
        production_adapter_factory=FROZEN_HELD_OUT_PRODUCTION_FACTORY,
        fallback_result_path=fallback_relative,
        fallback_result_file_sha256=_file_sha256(fallback_path),
        fallback_result_manifest_sha256=fallback_manifest_hash,
        fallback_run_id=run_id,
        development_execution_result_path=development_relative,
        development_execution_result_file_sha256=hashlib.sha256(development_bytes).hexdigest(),
        development_execution_result_hash=development.content_hash,
        development_execution_id=development.execution_id,
        development_execution_manifest_hash=development.execution_manifest_hash,
        development_prequery_inputs_hash=development.prequery_inputs_hash,
        development_continuation_receipt_hash=receipt.content_hash,
        development_predecessor_ledger=predecessor_ledger,
        selected_model_freeze_path=freeze_relative,
        selected_model_freeze_file_sha256=hashlib.sha256(freeze_bytes).hexdigest(),
        selected_model_freeze_hash=freeze_hash,
        source_association_path=association_relative,
        source_association_file_sha256=_file_sha256(association_path),
        source_association_manifest_sha256=association_hash,
        source_tree_sha256=source_tree_hash,
        created_at=created_at,
    )
    _atomic_write_new(
        binding_destination,
        (binding.to_canonical_json() + "\n").encode("utf-8"),
    )
    return binding


def load_runtime_bound_held_out_configuration(
    *,
    repository: Path,
    configuration_path: Path,
    binding_path: Path,
    expected_plan_hash: str | None = None,
) -> tuple[HeldOutControlConfiguration, HeldOutRuntimeBinding]:
    """Verify a restricted binding and return its in-memory control configuration."""

    repository = repository.resolve(strict=True)
    control_path = (
        configuration_path
        if configuration_path.is_absolute()
        else repository / configuration_path
    )
    control_path = _safe_file(control_path, label="held-out control configuration")
    binding_file = binding_path if binding_path.is_absolute() else repository / binding_path
    binding_file = _safe_file(binding_file, label="held-out runtime binding")
    try:
        binding = HeldOutRuntimeBinding.model_validate_json(binding_file.read_bytes())
    except Exception as error:
        raise HeldOutControlError(f"invalid held-out runtime binding: {error}") from error
    template = load_held_out_control_configuration(repository, control_path)
    if (
        binding.control_configuration_path != control_path.relative_to(repository).as_posix()
        or binding.control_configuration_file_sha256 != _file_sha256(control_path)
        or binding.control_configuration_hash != template.content_hash
        or template.production_adapter_factory != binding.production_adapter_factory
        or template.development_execution_result_file_sha256 != "PENDING"
    ):
        raise HeldOutControlError("held-out runtime binding names another source template")

    development_path = repository / binding.development_execution_result_path
    freeze_path = repository / binding.selected_model_freeze_path
    fallback_path = repository / binding.fallback_result_path
    association_path = repository / binding.source_association_path
    if (
        _file_sha256(_safe_file(development_path, label="development result"))
        != binding.development_execution_result_file_sha256
        or _file_sha256(_safe_file(freeze_path, label="selected-model freeze"))
        != binding.selected_model_freeze_file_sha256
        or _file_sha256(_safe_file(fallback_path, label="fallback/development result"))
        != binding.fallback_result_file_sha256
        or _file_sha256(_safe_file(association_path, label="source association"))
        != binding.source_association_file_sha256
    ):
        raise HeldOutControlError("held-out runtime predecessor file hash changed")
    development = DevelopmentExecutionResult.model_validate_json(development_path.read_bytes())
    _, freeze = _load_mapping(freeze_path, label="selected-model freeze")
    _, fallback = _load_mapping(fallback_path, label="fallback/development result")
    association = validate_source_association(association_path, source_root=repository)
    raw_receipt = fallback.get("development_continuation_receipt")
    if not isinstance(raw_receipt, Mapping):
        raise HeldOutControlError("fallback result lost its development continuation receipt")
    receipt = DevelopmentContinuationReceipt.model_validate(raw_receipt)
    if (
        development.content_hash != binding.development_execution_result_hash
        or development.execution_id != binding.development_execution_id
        or development.execution_manifest_hash
        != binding.development_execution_manifest_hash
        or development.prequery_inputs_hash != binding.development_prequery_inputs_hash
        or not development.gate.passed
        or development.phase is not DevelopmentPhase.COMPLETED
        or _verify_external_manifest(freeze, label="selected-model freeze")
        != binding.selected_model_freeze_hash
        or _verify_external_manifest(fallback, label="fallback/development result")
        != binding.fallback_result_manifest_sha256
        or fallback.get("run_id") != binding.fallback_run_id
        or fallback.get("phase1_gate_passed") is not True
        or fallback.get("gate_passed") is not True
        or fallback.get("vllm_service_stopped") is not True
        or fallback.get("physical_service_live") is not False
        or receipt.content_hash != binding.development_continuation_receipt_hash
        or receipt.development_result_hash != development.content_hash
        or association.get("manifest_sha256")
        != binding.source_association_manifest_sha256
        or association.get("local_tree_sha256") != binding.source_tree_sha256
        or freeze.get("source_tree_sha256") != binding.source_tree_sha256
        or freeze.get("source_association_manifest_sha256")
        != binding.source_association_manifest_sha256
    ):
        raise HeldOutControlError("held-out runtime predecessor content changed")

    payload = template.model_dump(mode="python", exclude={"content_hash"})
    payload.update(
        development_execution_result_path=binding.development_execution_result_path,
        development_execution_result_file_sha256=(
            binding.development_execution_result_file_sha256
        ),
        production_adapter_factory=binding.production_adapter_factory,
        expected_plan_hash=binding.post_development_call_manifest_hash,
    )
    configuration = HeldOutControlConfiguration.model_validate(payload)
    pending_payload = configuration.model_dump(mode="python", exclude={"content_hash"})
    pending_payload["expected_plan_hash"] = "PENDING"
    pending_configuration = HeldOutControlConfiguration.model_validate(pending_payload)
    from story_projection_onto.held_out_primary import _derive_call_manifest

    reproduced_plan_hash = _derive_call_manifest(
        repository,
        pending_configuration,
    ).content_hash
    if (
        binding.template_call_manifest_hash != template.expected_plan_hash
        or binding.runtime_control_configuration_hash
        != pending_configuration.content_hash
        or binding.post_development_call_manifest_hash != reproduced_plan_hash
        or expected_plan_hash not in {
            None,
            "PENDING",
            binding.post_development_call_manifest_hash,
        }
    ):
        raise HeldOutControlError(
            "held-out runtime configuration or post-development plan hash changed"
        )
    return configuration, binding


__all__ = [
    "DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH",
    "FROZEN_HELD_OUT_PRODUCTION_FACTORY",
    "DevelopmentPredecessorLedgerBinding",
    "HeldOutRuntimeBinding",
    "capture_development_predecessor_ledger",
    "load_runtime_bound_held_out_configuration",
    "materialize_held_out_runtime_binding",
    "persist_restricted_error",
    "verify_predecessor_gpu_event_chain",
]
