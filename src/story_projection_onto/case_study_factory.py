"""Concrete, fail-closed production factory for the bounded novel transfer study.

The execution plan is deliberately path-free.  This module is the only outer
lifecycle boundary that turns explicitly supplied restricted paths into the
single owned vLLM service used by Phase 6.  It does not discover a corpus, copy
novel prose, or create any public case-study result.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AwareDatetime

from story_projection_onto.case_study_execution import (
    CASE_REQUIRED_NEXT_REPAIR_SECONDS,
    CASE_SERVICE_START_WATCHDOG_SECONDS,
    CaseAdmissionEvidenceBundle,
    CaseAdmissionEvidenceReference,
    CaseArtifactReference,
    CaseControllerState,
    CaseExecutionAdmissionReceipt,
    CaseExecutionRepository,
    CaseStudyAdmissionError,
    CaseStudyExecutionPhase,
    CaseStudyExecutionResult,
    CaseStudyProductionController,
    ProductionCaseClassicalAdapter,
    _gpu_inventory_hash,
    _logical_json_hash,
    _parse_record,
    _persist_mapping,
    _persist_record,
    _publish_private_no_replace,
    _read_reference,
    _total_allocated_seconds,
    validate_case_admission_evidence,
)
from story_projection_onto.case_study_gpu import (
    ProductionCaseStudyGpuAdapter,
    build_production_case_study_gpu_adapter,
)
from story_projection_onto.case_study_runtime import (
    CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES,
    AttestedRestrictedCaseStudy,
    AttestedSelectedModelFreeze,
    CaseBlindedErrorReviewGate,
    CaseGoldFirewallGate,
    CaseGpuScheduleGate,
    CaseMetricRegenerationGate,
    CaseNativeGateArtifactReference,
    CasePublicReleaseScanGate,
    CaseStoragePreflightGate,
    CaseStudyAdmissionAttestation,
    CaseStudyExecutionPlan,
    CaseStudyResumeManifest,
    CaseStudyRuntimePolicy,
    CaseStudySemanticAdmissionBundle,
    CaseSyntheticClosureGate,
    CaseTimingLineageGate,
    audit_case_study_resume,
    load_attested_restricted_case_study,
    load_attested_selected_model_freeze,
    load_case_study_admission_attestation,
    load_case_study_execution_plan,
    load_case_study_semantic_admission_bundle,
    validate_case_study_semantic_admission,
)
from story_projection_onto.case_study_transition import (
    CaseAdmissionStagingReceipt,
    CasePostH1ArtifactDescriptor,
    CaseStagedPayloadDescriptor,
    VerifiedCaseAdmissionStagingTransition,
    capture_case_admission_staging_intent,
    case_admission_transition_paths,
    finalize_case_admission_staging_transition,
    verify_case_admission_staging_transition,
    verify_case_post_h1_artifact_prefix,
)
from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    DevelopmentConstructionConfiguration,
    PackingTokenizer,
)
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    ResourceLimits,
    StorageAllocationPlan,
)
from story_projection_onto.fallback_acceptance import validate_source_association
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    SERVICE_LOCK_FILENAME,
    ResourceSampler,
    ServiceState,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    capture_tokenizer_manifest,
)
from story_projection_onto.ledger_verify import audit_ledger
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    validate_fallback_snapshot_manifest,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    Ledger,
    ReadOnlyArtifactStore,
    StorageBudget,
    StoragePreflight,
    StorageReport,
)
from story_projection_onto.store import ReleaseClass as LedgerReleaseClass

CASE_PRODUCTION_FACTORY = (
    "story_projection_onto.case_study_factory:"
    "create_frozen_production_case_study_bundle"
)
CASE_FACTORY_REVISION = "case-study-production-factory-v2"


class CaseStudyFactoryError(CaseStudyAdmissionError):
    """An explicitly supplied production dependency differs from the freeze."""


@dataclass(frozen=True, slots=True)
class CaseStudyProductionPreflight:
    """Path-free result of the no-write launch-equivalent validation gate."""

    execution_plan_hash: str
    staging_transition_receipt_hash: str
    current_ledger_sha256: str
    actual_allocated_gpu_seconds: float
    remaining_required_gpu_seconds: float
    projected_storage_bytes: int
    filesystem_free_bytes: int
    selected_snapshot_manifest_hash: str
    launcher_configuration_hash: str
    bootstrap_state: Literal["fresh", "interrupted_prefix", "admitted"]
    controller_state: Literal["not_started", "in_progress", "complete"]
    controller_lock_available: Literal[True] = True
    service_lock_available: Literal[True] = True
    semantic_admission_replayed: Literal[True] = True
    storage_preflight_passed: Literal[True] = True
    model_snapshot_verified: Literal[True] = True
    gpu_accounting_verified: Literal[True] = True
    runtime_state_verified: Literal[True] = True
    writes_performed: Literal[False] = False
    execution_ready: Literal[True] = True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path, *, label: str, maximum_bytes: int = 64 * 1024 * 1024) -> Path:
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot traverse a symbolic link")
    if not lexical.is_file():
        raise CaseStudyFactoryError(f"{label} must be one regular non-symlink file")
    if lexical.stat().st_size > maximum_bytes:
        raise CaseStudyFactoryError(f"{label} exceeds its bounded input size")
    return lexical.resolve(strict=True)


def _ensure_private_directory(path: Path, *, restricted_root: Path, label: str) -> Path:
    _safe_real_directory(restricted_root, label="restricted root")
    root = restricted_root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise CaseStudyFactoryError(f"{label} must remain inside the restricted root") from error
    cursor = root
    relative = lexical.relative_to(root)
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot traverse a symbolic link")
    lexical.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(lexical, 0o700)
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise CaseStudyFactoryError(f"{label} requires a stable real path")
    return resolved


def _safe_real_directory(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor /= component
        if cursor.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot traverse a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise CaseStudyFactoryError(f"{label} is unavailable") from error
    if resolved != lexical or not resolved.is_dir():
        raise CaseStudyFactoryError(f"{label} must be one real directory")
    return resolved


def _exact_regular_file_inventory(
    root: Path,
    *,
    label: str,
    maximum_files: int = 1_024,
) -> frozenset[str]:
    """Return a bounded tree inventory while rejecting links and special files."""

    directory = _safe_real_directory(root, label=label)
    inventory: set[str] = set()
    for candidate in directory.rglob("*"):
        relative = candidate.relative_to(directory).as_posix()
        if candidate.is_symlink():
            raise CaseStudyFactoryError(f"{label} contains a symbolic link: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise CaseStudyFactoryError(f"{label} contains a special file: {relative}")
        inventory.add(relative)
        if len(inventory) > maximum_files:
            raise CaseStudyFactoryError(f"{label} exceeds its bounded file inventory")
    return frozenset(inventory)


def _require_exact_regular_file_inventory(
    root: Path,
    *,
    expected: frozenset[str],
    label: str,
) -> None:
    if _exact_regular_file_inventory(root, label=label) != expected:
        raise CaseStudyFactoryError(f"{label} inventory changed")


def _shared_blob_root(path: Path, *, evidence_root: Path, label: str) -> Path:
    """Resolve the one study CAS beneath the repository's artifact namespace."""

    root = _safe_real_directory(evidence_root, label="semantic evidence root")
    blob_parent = _safe_real_directory(
        root / "artifacts" / "blobs",
        label="study CAS parent",
    )
    candidate = _safe_real_directory(path, label=label)
    try:
        candidate.relative_to(blob_parent)
    except ValueError as error:
        raise CaseStudyFactoryError(
            f"{label} must remain beneath the repository artifacts/blobs root"
        ) from error
    return candidate


def _inside_quota(path: Path, quota_root: Path, *, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(quota_root.resolve(strict=True))
    except ValueError as error:
        raise CaseStudyFactoryError(f"{label} lies outside the controlled quota root") from error


def _atomic_private_record(path: Path, value: ImmutableRecord) -> None:
    payload = (value.to_canonical_json() + "\n").encode("utf-8")
    try:
        _publish_private_no_replace(path, payload)
    except Exception as error:
        raise CaseStudyFactoryError("immutable case bootstrap record changed") from error


def _acquire_runtime_lock(
    runtime_root: Path,
    *,
    ledger_path: Path | None = None,
) -> Any:
    """Hold one crash-releasing controller lock for the exact cumulative ledger."""

    path = _runtime_lock_path(runtime_root, ledger_path=ledger_path)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        lock_info = os.fstat(descriptor)
        path_info = path.lstat()
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or (lock_info.st_dev, lock_info.st_ino)
            != (path_info.st_dev, path_info.st_ino)
        ):
            raise OSError("controller lock is not one stable regular file")
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = -1
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if "stream" in locals():
            stream.close()
        elif "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)
        raise CaseStudyFactoryError("another case-study controller owns the runtime") from error
    return stream


def _runtime_lock_path(
    runtime_root: Path,
    *,
    ledger_path: Path | None,
) -> Path:
    if ledger_path is None:
        # Retained for focused lock tests and callers without a ledger capability.
        return runtime_root / ".case-study-controller.lock"
    ledger = Path(os.path.abspath(ledger_path))
    parent = ledger.parent
    if (
        ledger.is_symlink()
        or not ledger.is_file()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise CaseStudyFactoryError(
            "case-study controller lock requires one real cumulative ledger"
        )
    return parent / f".{ledger.name}.case-study-controller.lock"


def _probe_exclusive_lock(path: Path, *, label: str) -> None:
    """Check an execution lock without creating or changing its lock file."""

    if path.is_symlink():
        raise CaseStudyFactoryError(f"{label} cannot be a symbolic link")
    if not path.exists():
        parent = _safe_real_directory(path.parent, label=f"{label} parent")
        if not os.access(parent, os.W_OK | os.X_OK):
            raise CaseStudyFactoryError(f"{label} cannot be created for execution")
        return
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        lock_info = os.fstat(descriptor)
        path_info = path.lstat()
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_nlink != 1
            or (lock_info.st_dev, lock_info.st_ino)
            != (path_info.st_dev, path_info.st_ino)
        ):
            raise CaseStudyFactoryError(f"{label} is not one stable regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except BlockingIOError as error:
        raise CaseStudyFactoryError(f"{label} is held by another controller") from error
    except OSError as error:
        raise CaseStudyFactoryError(f"{label} is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_private_runtime_bytes(path: Path, *, label: str) -> bytes:
    resolved = _safe_file(path, label=label)
    if resolved.lstat().st_nlink != 1:
        raise CaseStudyFactoryError(f"{label} must be a singly-linked regular file")
    return resolved.read_bytes()


def _preflight_controller_state(
    *,
    runtime_root: Path,
    plan: CaseStudyExecutionPlan,
    artifacts: ReadOnlyArtifactStore,
    admission_reference: CaseArtifactReference | None,
) -> tuple[Literal["not_started", "in_progress", "complete"], float]:
    """Replay the durable controller pointer without creating runtime directories."""

    _exact_regular_file_inventory(
        runtime_root,
        label="case runtime root",
        maximum_files=4_096,
    )
    pointer_path = runtime_root / "controller" / "current.json"
    history_path = pointer_path.parent / f"{pointer_path.name}.history"
    if pointer_path.exists() and (
        pointer_path.is_symlink() or not pointer_path.is_file()
    ):
        raise CaseStudyFactoryError("case state pointer is not a regular private file")
    if history_path.exists() and (
        history_path.is_symlink() or not history_path.is_dir()
    ):
        raise CaseStudyFactoryError("case state history is not a real directory")

    candidates: list[tuple[Path, bytes]] = []
    if pointer_path.is_file():
        candidates.append(
            (
                pointer_path,
                _read_private_runtime_bytes(
                    pointer_path,
                    label="case state pointer",
                ),
            )
        )
    if history_path.is_dir():
        history = _safe_real_directory(history_path, label="case state history")
        for candidate in sorted(history.iterdir()):
            if (
                candidate.name.endswith(".json")
                and len(candidate.stem) == 64
                and all(character in "0123456789abcdef" for character in candidate.stem)
            ):
                candidates.append(
                    (
                        candidate,
                        _read_private_runtime_bytes(
                            candidate,
                            label="case state history entry",
                        ),
                    )
                )
            else:
                raise CaseStudyFactoryError("case state history has an unexpected entry")
    if not candidates:
        return (
            "not_started",
            float(
                CASE_SERVICE_START_WATCHDOG_SECONDS
                + sum(item.watchdog_seconds for item in plan.gpu_call_slots)
                + CASE_REQUIRED_NEXT_REPAIR_SECONDS
            ),
        )

    states: dict[str, CaseControllerState] = {}
    for candidate, payload in candidates:
        try:
            pointer = json.loads(payload)
            if not isinstance(pointer, Mapping):
                raise ValueError
            reference = CaseArtifactReference(
                logical_content_hash=cast(str, pointer.get("state_logical_hash")),
                artifact_hash=cast(str, pointer.get("state_artifact_hash")),
                object_kind="case_controller_state",
            )
            if pointer.get("execution_plan_hash") != plan.content_hash:
                raise ValueError
            state = cast(
                CaseControllerState,
                _parse_record(
                    cast(Any, artifacts),
                    reference,
                    CaseControllerState,
                ),
            )
            if state.execution_plan_hash != plan.content_hash:
                raise ValueError
            if candidate.parent == history_path and candidate.stem != state.content_hash:
                raise ValueError
        except Exception as error:
            if candidate == pointer_path:
                # The immutable history is the crash-recovery authority; a torn
                # convenience pointer may be ignored only when history survives.
                continue
            raise CaseStudyFactoryError("case state history failed replay") from error
        states[state.content_hash] = state
    if not states:
        raise CaseStudyFactoryError("case state pointer history is invalid")
    latest_sequence = max(item.sequence_number for item in states.values())
    latest = tuple(item for item in states.values() if item.sequence_number == latest_sequence)
    if len(latest) != 1:
        raise CaseStudyFactoryError("case state pointer history forks at its latest state")
    state = latest[0]
    if admission_reference is None or state.admission_receipt != admission_reference:
        raise CaseStudyFactoryError("case runtime state belongs to another admission")
    if state.resume_manifest.object_kind != "case_resume_manifest":
        raise CaseStudyFactoryError("case state references another resume object kind")
    resume = cast(
        CaseStudyResumeManifest,
        _parse_record(
            cast(Any, artifacts),
            state.resume_manifest,
            CaseStudyResumeManifest,
        ),
    )
    status = audit_case_study_resume(plan, resume)
    if state.model_service_shutdown_count == 1 and (
        state.phase is not CaseStudyExecutionPhase.COMPLETED
    ):
        raise CaseStudyFactoryError("incomplete case state already stopped its sole model load")
    if state.phase is CaseStudyExecutionPhase.COMPLETED and not status.complete:
        raise CaseStudyFactoryError("terminal case state has an incomplete resume manifest")
    completed_call_ids = {
        item.preparation_job_id for item in resume.prequery_receipts
    } | {item.projection_job_id for item in resume.output_receipts}
    pending = tuple(
        item for item in plan.gpu_call_slots if item.call_id not in completed_call_ids
    )
    remaining = sum(item.watchdog_seconds for item in pending)
    if pending:
        remaining += CASE_REQUIRED_NEXT_REPAIR_SECONDS
        if state.model_service_start_count == 0:
            remaining += CASE_SERVICE_START_WATCHDOG_SECONDS
    return (
        "complete" if state.phase is CaseStudyExecutionPhase.COMPLETED else "in_progress",
        float(remaining),
    )


def _release_runtime_lock(stream: Any) -> None:
    """Release the advisory lock and always close its owning descriptor."""

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _cleanup_runtime_ownership(
    *,
    ledger: Ledger | None,
    runtime_lock: Any,
    primary_error: BaseException,
) -> None:
    """Best-effort cleanup that cannot replace the scientifically useful error."""

    if ledger is not None:
        try:
            ledger.close()
        except BaseException as cleanup_error:
            primary_error.add_note(
                "ledger cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
    try:
        _release_runtime_lock(runtime_lock)
    except BaseException as cleanup_error:
        primary_error.add_note(
            "runtime-lock cleanup also failed: "
            f"{type(cleanup_error).__name__}"
        )


def _restricted_location(
    path: Path,
    *,
    restricted_root: Path,
    label: str,
    directory: bool,
) -> Path:
    """Require a stable real path with no ancestor symlink under restricted root."""

    root = _safe_real_directory(restricted_root, label="restricted root")
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise CaseStudyFactoryError(f"{label} must remain inside the restricted root") from error
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot traverse a symbolic link")
    target = lexical if directory else lexical.parent
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.resolve(strict=True) != target:
        raise CaseStudyFactoryError(f"{label} requires a stable real path")
    if directory:
        os.chmod(lexical, 0o700)
    elif lexical.exists() and (lexical.is_symlink() or not lexical.is_file()):
        raise CaseStudyFactoryError(f"{label} must be one regular non-symlink file")
    return lexical


def _load_record(path: Path, model: type[ImmutableRecord], *, label: str) -> ImmutableRecord:
    resolved = _safe_file(path, label=label)
    try:
        return model.model_validate_json(resolved.read_bytes())
    except Exception as error:
        raise CaseStudyFactoryError(f"{label} is not a valid immutable record") from error


class CaseAdmissionBootstrapIntent(ImmutableRecord):
    """Path-free intent written before admission mutates the cumulative ledger."""

    factory_revision: str = CASE_FACTORY_REVISION
    execution_plan_hash: Sha256Digest
    admission_attestation_hash: Sha256Digest
    evidence_bundle_hash: Sha256Digest
    evidence_bundle_reference_hash: Sha256Digest
    construction_configuration_hash: Sha256Digest
    construction_configuration_file_sha256: Sha256Digest
    source_revision: str
    source_tree_sha256: Sha256Digest
    staging_transition_receipt_hash: Sha256Digest
    predecessor_ledger_sha256: Sha256Digest
    prior_gpu_event_inventory_hash: Sha256Digest
    allocated_gpu_seconds_before_case: float
    admitted_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


@dataclass(slots=True)
class CaseStudyProductionBundle:
    """Owned Phase-6 runtime; the controller remains the sole service owner."""

    controller: CaseStudyProductionController
    admission_reference: CaseArtifactReference
    artifacts: ArtifactStore
    ledger: Ledger
    service: VLLMService = field(repr=False)
    gpu: ProductionCaseStudyGpuAdapter = field(repr=False)
    runtime_root: Path = field(repr=False)
    storage_report: StorageReport
    _runtime_lock: Any = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def persist_result(self, result: CaseStudyExecutionResult) -> CaseArtifactReference:
        """Persist one path-free terminal result in restricted CAS and immutable JSON."""

        reference = _persist_record(
            self.artifacts,
            result,
            object_kind="case_study_execution_result",
            created_at=result.completed_at,
        )
        result_path = self.runtime_root / "results" / f"{result.content_hash}.json"
        _atomic_private_record(result_path, result)
        _atomic_private_record(
            self.runtime_root / "terminal-result-reference.json",
            reference,
        )
        return reference

    def close(self) -> None:
        """Reclaim any live process and close the ledger without hiding failures."""

        if self._closed:
            return
        failure: BaseException | None = None
        try:
            if self.service.state is not ServiceState.STOPPED:
                try:
                    self.gpu.shutdown()
                except BaseException as adapter_error:
                    # A startup failure can stop the physical process before the
                    # adapter has a persisted service identity.  The raw owner is
                    # then the only component capable of closing its journal.
                    try:
                        self.service.shutdown()
                    except BaseException as service_error:
                        raise service_error from adapter_error
                    raise
        except BaseException as error:
            failure = error
        try:
            self.ledger.close()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(
                    "ledger cleanup also failed: "
                    f"{type(error).__name__}"
                )
        try:
            _release_runtime_lock(self._runtime_lock)
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(
                    "runtime-lock cleanup also failed: "
                    f"{type(error).__name__}"
                )
        finally:
            self._closed = True
        if failure is not None:
            raise failure


_SEMANTIC_GATE_MODELS: Mapping[str, type[ImmutableRecord]] = {
    "synthetic_run_closure_hash": CaseSyntheticClosureGate,
    "timing_lineage_audit_hash": CaseTimingLineageGate,
    "gold_firewall_audit_hash": CaseGoldFirewallGate,
    "registered_metric_regeneration_hash": CaseMetricRegenerationGate,
    "blinded_error_review_hash": CaseBlindedErrorReviewGate,
    "storage_preflight_hash": CaseStoragePreflightGate,
    "gpu_schedule_admission_hash": CaseGpuScheduleGate,
    "public_release_scan_hash": CasePublicReleaseScanGate,
}
_NATIVE_GATE_ROLES = CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES


def _native_gate_reference(
    *, evidence_root: Path, role: str, path: Path
) -> CaseNativeGateArtifactReference:
    source = _safe_file(path, label=f"native gate artifact {role}", maximum_bytes=512 * 1024 * 1024)
    try:
        relative = source.relative_to(evidence_root)
    except ValueError as error:
        raise CaseStudyFactoryError(
            f"native gate artifact {role} must remain inside the evidence root"
        ) from error
    raw = source.read_bytes()
    logical_hash = hashlib.sha256(raw).hexdigest()
    if source.suffix == ".json":
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaseStudyFactoryError(f"native gate artifact {role} is invalid JSON") from error
        if not isinstance(value, Mapping):
            raise CaseStudyFactoryError(f"native gate artifact {role} must be one JSON object")
        logical_hash = _logical_json_hash(value)
    return CaseNativeGateArtifactReference(
        role=role,
        relative_path=relative.as_posix(),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        logical_content_hash=logical_hash,
    )


def _resolve_native_gate_artifacts(
    *,
    bundle: CaseStudySemanticAdmissionBundle,
    evidence_root: Path,
    cumulative_ledger_override: Path | None = None,
) -> dict[str, Path]:
    """Resolve and re-hash every native input named by a semantic bundle."""

    root = _safe_real_directory(evidence_root, label="semantic evidence root")
    resolved: dict[str, Path] = {}
    for reference in bundle.native_artifacts:
        candidate = (
            cumulative_ledger_override
            if reference.role == "cumulative_ledger"
            and cumulative_ledger_override is not None
            else root.joinpath(*Path(reference.relative_path).parts)
        )
        source = _safe_file(
            candidate,
            label=f"native gate artifact {reference.role}",
            maximum_bytes=512 * 1024 * 1024,
        )
        try:
            source.relative_to(root)
        except ValueError as error:
            raise CaseStudyFactoryError("native gate artifact escapes its evidence root") from error
        raw = source.read_bytes()
        observed_file_sha256 = hashlib.sha256(raw).hexdigest()
        if observed_file_sha256 != reference.file_sha256:
            raise CaseStudyFactoryError(
                f"native gate artifact bytes changed for role {reference.role}"
            )
        logical_hash = observed_file_sha256
        if source.suffix == ".json":
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaseStudyFactoryError(
                    f"native gate artifact {reference.role} is invalid JSON"
                ) from error
            if not isinstance(value, Mapping):
                raise CaseStudyFactoryError(
                    f"native gate artifact {reference.role} must be one JSON object"
                )
            logical_hash = _logical_json_hash(value)
        if logical_hash != reference.logical_content_hash:
            raise CaseStudyFactoryError(
                f"native gate artifact logical hash changed for role {reference.role}"
            )
        resolved[reference.role] = source
    if set(resolved) != _NATIVE_GATE_ROLES:
        raise CaseStudyFactoryError("semantic bundle has an incomplete native inventory")
    return resolved


def _typed_native(
    paths: Mapping[str, Path], role: str, model: type[ImmutableRecord]
) -> ImmutableRecord:
    return _load_record(paths[role], model, label=f"native {role}")


def _canonical_phase4_output_root(*, index_path: Path, table_manifest_path: Path) -> Path:
    root = table_manifest_path.parent
    if (
        table_manifest_path != root / "table_manifest.json"
        or index_path != root / "analysis_index.json"
    ):
        raise CaseStudyFactoryError(
            "Phase 4 native index and table manifest must use their canonical sibling paths"
        )
    return root


def _canonical_community_review_roots(
    *,
    evidence_root: Path,
    rubric_template_path: Path,
    source_manifest_path: Path,
    package_path: Path,
    rejoin_path: Path,
    completion_path: Path,
    finalization_path: Path,
    table_path: Path,
) -> tuple[Path, Path, Path]:
    """Resolve the three append-only community-review bundles and frozen rubric."""

    repository = _safe_real_directory(evidence_root, label="semantic evidence root")
    expected_rubric = repository / "configs" / "study" / "community_review_template.json"
    if rubric_template_path != expected_rubric:
        raise CaseStudyFactoryError(
            "community-review rubric must use its canonical tracked configuration path"
        )
    source_root = source_manifest_path.parent
    package_root = package_path.parent.parent
    final_root = finalization_path.parent
    if source_manifest_path != source_root / "source_manifest.json":
        raise CaseStudyFactoryError(
            "community-review source manifest must use its canonical bundle path"
        )
    if (
        package_path != package_root / "reviewer" / "review_package.json"
        or rejoin_path != package_root / "scorer_only" / "rejoin_map.json"
    ):
        raise CaseStudyFactoryError(
            "community-review package and rejoin map must use canonical sibling paths"
        )
    if (
        completion_path != final_root / "completion.json"
        or finalization_path != final_root / "finalization.json"
        or table_path != final_root / "community_blind_review.csv"
    ):
        raise CaseStudyFactoryError(
            "community-review completion, finalization, and table must use canonical sibling paths"
        )
    return source_root, package_root, final_root


def _validate_error_review_chronology(
    *,
    analysis_completed_at: datetime,
    source_selected_at: datetime,
    reviewer_completed_at: datetime,
    adjudicated_at: datetime,
    review_gate_completed_at: datetime,
    semantic_bundle_frozen_at: datetime,
) -> None:
    chronology = (
        analysis_completed_at,
        source_selected_at,
        reviewer_completed_at,
        adjudicated_at,
        review_gate_completed_at,
        semantic_bundle_frozen_at,
    )
    if any(later < earlier for earlier, later in pairwise(chronology)):
        raise CaseStudyFactoryError("condition-blind review chronology is not monotonic")


def _validate_community_review_chronology(
    *,
    analysis_completed_at: datetime,
    source_selected_at: datetime,
    reviewer_completed_at: datetime,
    binding_gate_completed_at: datetime,
    semantic_bundle_frozen_at: datetime,
) -> None:
    chronology = (
        analysis_completed_at,
        source_selected_at,
        reviewer_completed_at,
        binding_gate_completed_at,
        semantic_bundle_frozen_at,
    )
    if any(later < earlier for earlier, later in pairwise(chronology)):
        raise CaseStudyFactoryError("community-review chronology is not monotonic")


def _assert_neutral_community_panel_payload(
    payload: bytes,
    *,
    forbidden_identity_values: tuple[str, ...],
) -> None:
    """Reject rejoin fields and normalized embedded restricted identifiers."""

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaseStudyFactoryError("community-review panel is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise CaseStudyFactoryError("community-review panel must be one JSON object")
    forbidden_keys = {
        "condition",
        "context_id",
        "expected_effect",
        "gold",
        "gold_hash",
        "partition_hash",
        "projection_hash",
        "resolution",
        "seed_block",
        "source_partition_id",
        "world_id",
    }

    def keys(current: object) -> set[str]:
        if isinstance(current, Mapping):
            return set(current) | {
                nested
                for child in current.values()
                for nested in keys(child)
            }
        if isinstance(current, list):
            return {nested for child in current for nested in keys(child)}
        return set()

    normalized_keys = {
        unicodedata.normalize("NFKC", item).casefold() for item in keys(value)
    }
    if normalized_keys & forbidden_keys or any(
        item.startswith("gold_") or "expected_effect" in item
        for item in normalized_keys
    ):
        raise CaseStudyFactoryError("community-review panel exposes a rejoin field")

    def strings(current: object) -> tuple[str, ...]:
        if isinstance(current, str):
            return (current,)
        if isinstance(current, Mapping):
            return tuple(
                child
                for key, nested_value in current.items()
                for child in (str(key), *strings(nested_value))
            )
        if isinstance(current, list):
            return tuple(child for item in current for child in strings(item))
        return ()

    normalized_strings = tuple(
        unicodedata.normalize("NFKC", item).casefold() for item in strings(value)
    )

    def contains_identity(text: str, token: str) -> bool:
        normalized = unicodedata.normalize("NFKC", token).casefold()
        if not normalized:
            return False
        if len(normalized) >= 4:
            return normalized in text
        start = 0
        while True:
            index = text.find(normalized, start)
            if index < 0:
                return False
            before = text[index - 1] if index else ""
            after_index = index + len(normalized)
            after = text[after_index] if after_index < len(text) else ""
            if (not before or not before.isalnum()) and (
                not after or not after.isalnum()
            ):
                return True
            start = index + 1

    if any(
        contains_identity(text, token)
        for token in forbidden_identity_values
        for text in normalized_strings
    ):
        raise CaseStudyFactoryError("community-review panel leaked its rejoin identity")


def _require_content_addressed_root(
    root: Path,
    *,
    expected_hash: str,
    label: str,
) -> None:
    if root.name != expected_hash:
        raise CaseStudyFactoryError(f"{label} root is not named by its content hash")


def replay_case_study_semantic_admission(
    *,
    bundle: CaseStudySemanticAdmissionBundle,
    evidence_root: Path,
    ledger_path: Path,
    blob_root: Path,
    cumulative_ledger_override: Path | None = None,
) -> None:
    """Replay native Phase 4/5 and accounting artifacts; never trust gate booleans.

    This function is intentionally called at bundle compilation, plan compilation,
    staging, and execution admission.  The summary gates are accepted only when
    their hashes can be regenerated from the exact typed artifacts below.
    """

    from story_projection_onto.combined_gpu_block import CombinedCallManifest
    from story_projection_onto.combined_gpu_production import CombinedExecutionIndex
    from story_projection_onto.held_out_controller import (
        HeldOutExecutionManifest,
        ScorerBridgeAuthorization,
        _assert_construction_lineage,
        _assert_query_fairness,
        _audit_journal_tree,
        _replay_complete_journal,
    )
    from story_projection_onto.held_out_primary import HeldOutCallManifest
    from story_projection_onto.independent_review_runtime import (
        IndependentReviewCompletionManifest,
    )
    from story_projection_onto.metrics.config import CommunityReviewTemplate
    from story_projection_onto.metrics.rare import (
        GoldLabelFirewallAudit,
        RunArtifactFingerprint,
        ScorerLabelSet,
        audit_gold_label_firewall,
    )
    from story_projection_onto.phase5_execution import (
        Phase5JournalIndex,
        PrimaryHeldOutResultsGate,
    )
    from story_projection_onto.public_release import (
        load_protected_prose_canaries,
        load_public_entries,
        scan_public_entries,
    )
    from story_projection_onto.scorer_only.blinded_postrun_review import (
        BlindedCommunityReviewPackage,
        BlindedErrorReviewPackage,
        CommunityReviewCompletion,
        CommunityReviewFinalization,
        CommunityReviewRejoinMap,
        CommunityReviewSourceManifest,
        ErrorReviewAdjudication,
        ErrorReviewCompletion,
        ErrorReviewFinalization,
        ErrorReviewRejoinMap,
        HeldOutErrorTaxonomy,
        HeldOutFailureSourceManifest,
        NeutralCommunityReviewPanel,
        prepare_community_review_finalization,
        prepare_community_review_package,
        prepare_community_review_source,
        prepare_error_review_finalization,
        prepare_error_review_package,
        prepare_held_out_failure_source,
    )
    from story_projection_onto.scorer_only.phase4_analysis import (
        Phase4AnalysisIndex,
        Phase4TableManifest,
    )
    from story_projection_onto.scorer_only.phase4_replay import (
        replay_phase4_analysis_outputs,
    )
    from story_projection_onto.scorer_only.phase5_feedback import (
        Phase5FeedbackScoringReceipt,
        Phase5FeedbackScoringSession,
        Phase5ScriptedFeedbackMetrics,
        prepare_phase5_feedback_scoring,
    )
    from story_projection_onto.synthetic_benchmark import (
        SyntheticBenchmarkManifest,
        verify_materialized_benchmark,
    )

    paths = _resolve_native_gate_artifacts(
        bundle=bundle,
        evidence_root=evidence_root,
        cumulative_ledger_override=cumulative_ledger_override,
    )
    ledger = _safe_file(
        ledger_path,
        label="live cumulative ledger",
        maximum_bytes=512 * 1024 * 1024,
    )
    if ledger != paths["cumulative_ledger"]:
        raise CaseStudyFactoryError("semantic replay was given another cumulative ledger")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(ledger) + suffix)
        if sidecar.is_symlink() or (
            sidecar.exists() and (not sidecar.is_file() or sidecar.stat().st_size)
        ):
            raise CaseStudyFactoryError(
                "semantic replay requires a closed, checkpointed cumulative ledger"
            )
    blobs = _shared_blob_root(
        blob_root,
        evidence_root=evidence_root,
        label="cumulative CAS root",
    )
    ledger_audit = audit_ledger(ledger, blobs)
    if not ledger_audit.valid:
        raise CaseStudyFactoryError("cumulative ledger/CAS failed exhaustive native audit")

    association = validate_source_association(
        paths["source_association"], source_root=evidence_root
    )
    source_tree_hash = association.get("local_tree_sha256")
    if not isinstance(source_tree_hash, str):
        raise CaseStudyFactoryError("source association lacks its local tree hash")
    gates = (
        bundle.synthetic_run_closure,
        bundle.timing_lineage_audit,
        bundle.gold_firewall_audit,
        bundle.registered_metric_regeneration,
        bundle.blinded_error_review,
        bundle.storage_preflight,
        bundle.gpu_schedule_admission,
        bundle.public_release_scan,
    )
    if any(gate.source_tree_sha256 != source_tree_hash for gate in gates):
        raise CaseStudyFactoryError(
            "semantic gate source tree was not replayed from source association"
        )

    try:
        freeze_outer = json.loads(paths["selected_model_freeze"].read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaseStudyFactoryError("selected-model freeze is invalid JSON") from error
    freeze = freeze_outer.get("selected_model_freeze", freeze_outer)
    if not isinstance(freeze, Mapping):
        raise CaseStudyFactoryError("selected-model freeze is not an object")
    freeze_payload = {key: value for key, value in freeze.items() if key != "manifest_sha256"}
    freeze_hash = canonical_sha256(freeze_payload)
    if (
        freeze.get("manifest_sha256") != freeze_hash
        or freeze.get("kind") != "selected_llm_model_freeze"
        or freeze.get("source_tree_sha256") != source_tree_hash
        or freeze.get("source_association_manifest_sha256")
        != association.get("manifest_sha256")
        or any(gate.selected_model_freeze_hash != freeze_hash for gate in gates)
    ):
        raise CaseStudyFactoryError("semantic gates do not replay the selected-model freeze")

    benchmark = cast(
        SyntheticBenchmarkManifest,
        _typed_native(paths, "synthetic_benchmark_manifest", SyntheticBenchmarkManifest),
    )
    benchmark_root = paths["synthetic_benchmark_manifest"].parent.parent
    try:
        verify_materialized_benchmark(benchmark_root, paths["synthetic_benchmark_config"])
    except Exception as error:
        raise CaseStudyFactoryError("synthetic benchmark tree failed native replay") from error

    heldout_gate = cast(
        PrimaryHeldOutResultsGate,
        _typed_native(paths, "held_out_results_gate", PrimaryHeldOutResultsGate),
    )
    execution = cast(
        HeldOutExecutionManifest,
        _typed_native(paths, "held_out_execution_manifest", HeldOutExecutionManifest),
    )
    bridge = cast(
        ScorerBridgeAuthorization,
        _typed_native(paths, "held_out_scorer_bridge", ScorerBridgeAuthorization),
    )
    journal_root = paths["held_out_execution_manifest"].parent
    call_manifest = cast(
        HeldOutCallManifest,
        _typed_native(paths, "held_out_call_manifest", HeldOutCallManifest),
    )
    journal_call_manifest = cast(
        HeldOutCallManifest,
        _load_record(
            journal_root / "call_manifest.json",
            HeldOutCallManifest,
            label="held-out journal call manifest",
        ),
    )
    if (
        journal_call_manifest != call_manifest
        or _file_sha256(journal_root / "call_manifest.json")
        != _file_sha256(paths["held_out_call_manifest"])
    ):
        raise CaseStudyFactoryError("held-out call manifest differs from its journal")
    try:
        _audit_journal_tree(journal_root, call_manifest, require_complete=True)
        # Re-read and compare every persisted journal record with the terminal
        # execution manifest.  The original production runtime is deliberately
        # unavailable in the scorer/case namespace, so these three callbacks do
        # not claim to replay its CAS payloads; those require a separately bound
        # native CAS inventory.  They only let the native journal replayer reach
        # all of its typed record and call-envelope comparisons.
        class _JournalRecordReplay:
            @staticmethod
            def validate_c2_prequery_receipt(_receipt: object) -> None:
                return None

            @staticmethod
            def validate_ablation_prequery_receipt(_receipt: object) -> None:
                return None

            @staticmethod
            def validate_result_artifacts(*_args: object) -> None:
                return None

        _replay_complete_journal(
            journal_root,
            call_manifest,
            execution,
            cast(Any, _JournalRecordReplay()),
        )
        _assert_query_fairness(
            call_manifest,
            execution.preconstructed_projections,
            execution.itt_records,
        )
        _assert_construction_lineage(
            call_manifest,
            execution.c0_constructions,
            execution.c2_prequery_receipts,
            execution.ablation_prequery_receipts,
            execution.prequery_barrier,
            execution.preconstructed_projections,
            execution.itt_records,
        )
    except Exception as error:
        raise CaseStudyFactoryError(
            "held-out journal fairness/construction replay failed"
        ) from error
    if (
        call_manifest.content_hash != execution.call_manifest_hash
        or heldout_gate.held_out_execution_manifest_hash != execution.content_hash
        or heldout_gate.held_out_execution_manifest_file_sha256
        != _file_sha256(paths["held_out_execution_manifest"])
        or heldout_gate.scorer_bridge_hash != bridge.content_hash
        or heldout_gate.scorer_bridge_file_sha256 != _file_sha256(paths["held_out_scorer_bridge"])
        or bridge.execution_manifest_hash != execution.content_hash
        or bridge.call_manifest_hash != execution.call_manifest_hash
        or bridge.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
        or tuple(bridge.output_artifact_hashes) != tuple(heldout_gate.result_artifact_hashes)
        or bridge.c0_receipt_hashes
        != tuple(item.content_hash for item in execution.c0_constructions)
        or bridge.projection_receipt_hashes
        != tuple(item.content_hash for item in execution.preconstructed_projections)
        or bridge.c2_prequery_receipt_hashes
        != tuple(item.content_hash for item in execution.c2_prequery_receipts)
        or bridge.ablation_prequery_receipt_hashes
        != tuple(item.content_hash for item in execution.ablation_prequery_receipts)
        or bridge.prequery_barrier_hash != execution.prequery_barrier.content_hash
        or bridge.query_opening_hashes
        != tuple(item.content_hash for item in execution.query_openings)
        or bridge.service_shutdown_receipt_hashes
        != tuple(item.content_hash for item in execution.service_shutdown_receipts)
        or bridge.itt_record_hashes
        != tuple(item.content_hash for item in execution.itt_records)
    ):
        raise CaseStudyFactoryError("held-out closure artifacts do not form one exact lineage")
    closure = bundle.synthetic_run_closure
    output_inventory_hash = canonical_sha256(
        {
            "domain": "held_out_primary_output_receipts/v1",
            "c0_receipt_hashes": bridge.c0_receipt_hashes,
            "projection_receipt_hashes": bridge.projection_receipt_hashes,
            "itt_record_hashes": bridge.itt_record_hashes,
        }
    )
    if (
        closure.held_out_execution_plan_hash != execution.call_manifest_hash
        or closure.held_out_results_closure_hash != heldout_gate.content_hash
        or closure.synthetic_benchmark_hash != benchmark.content_hash
        or closure.output_receipt_inventory_hash != output_inventory_hash
    ):
        raise CaseStudyFactoryError(
            "synthetic closure summary was not derived from held-out artifacts"
        )

    combined = cast(
        CombinedExecutionIndex,
        _typed_native(paths, "combined_execution_index", CombinedExecutionIndex),
    )
    combined_manifest = cast(
        CombinedCallManifest,
        _typed_native(paths, "combined_call_manifest", CombinedCallManifest),
    )
    combined_journal_manifest = cast(
        CombinedCallManifest,
        _load_record(
            paths["combined_execution_index"].parent / "manifest.json",
            CombinedCallManifest,
            label="combined journal call manifest",
        ),
    )
    independent_review = cast(
        IndependentReviewCompletionManifest,
        _typed_native(
            paths,
            "independent_review_completion",
            IndependentReviewCompletionManifest,
        ),
    )
    phase5 = cast(
        Phase5JournalIndex,
        _typed_native(paths, "phase5_journal_index", Phase5JournalIndex),
    )
    if (
        combined_manifest != combined_journal_manifest
        or _file_sha256(paths["combined_call_manifest"])
        != _file_sha256(paths["combined_execution_index"].parent / "manifest.json")
        or combined.manifest_hash != combined_manifest.content_hash
        or execution.review_completion_manifest_hash != independent_review.content_hash
        or execution.final_reviewed_seal_hash != independent_review.final_seal_hash
        or combined.phase5_index_hash != phase5.content_hash
        or phase5.primary_results_gate_hash != heldout_gate.content_hash
        or phase5.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
    ):
        raise CaseStudyFactoryError(
            "combined/feedback execution lineage does not close held-out results"
        )

    table_manifest = cast(
        Phase4TableManifest,
        _typed_native(paths, "phase4_table_manifest", Phase4TableManifest),
    )
    analysis_index = cast(
        Phase4AnalysisIndex,
        _typed_native(paths, "phase4_analysis_index", Phase4AnalysisIndex),
    )
    table_root = _canonical_phase4_output_root(
        index_path=paths["phase4_analysis_index"],
        table_manifest_path=paths["phase4_table_manifest"],
    )
    try:
        phase4_replay = replay_phase4_analysis_outputs(
            repository=_safe_real_directory(evidence_root, label="semantic evidence root"),
            configuration_path=(
                Path(evidence_root) / "configs" / "study" / "phase4_analysis.json"
            ),
            output_root=table_root,
        )
    except Exception as error:
        raise CaseStudyFactoryError("Phase 4 complete output replay failed") from error
    expected_phase4_sources: Mapping[str, tuple[Path, ImmutableRecord]] = {
        "held_out_call_manifest": (paths["held_out_call_manifest"], call_manifest),
        "held_out_execution": (paths["held_out_execution_manifest"], execution),
        "held_out_scorer_bridge": (paths["held_out_scorer_bridge"], bridge),
        "combined_call_manifest": (paths["combined_call_manifest"], combined_manifest),
        "combined_execution": (paths["combined_execution_index"], combined),
        "review_completion_manifest": (
            paths["independent_review_completion"],
            independent_review,
        ),
    }
    source_binding_by_role = {item.role: item for item in analysis_index.source_bindings}
    if set(source_binding_by_role) != set(expected_phase4_sources):
        raise CaseStudyFactoryError("Phase 4 source-binding inventory changed")
    for role, (source_path, source_record) in expected_phase4_sources.items():
        binding = source_binding_by_role[role]
        if (
            binding.file_sha256 != _file_sha256(source_path)
            or binding.logical_content_hash != source_record.content_hash
        ):
            raise CaseStudyFactoryError(f"Phase 4 source binding changed for {role}")
    association_manifest_hash = association.get("manifest_sha256")
    if not isinstance(association_manifest_hash, str):
        raise CaseStudyFactoryError("source association lacks its manifest hash")
    if (
        phase4_replay.analysis_index_hash != analysis_index.content_hash
        or phase4_replay.table_manifest_hash != table_manifest.content_hash
        or phase4_replay.registered_analysis_hash
        != analysis_index.registered_analysis_hash
        or phase4_replay.report_gate_status_hash
        != analysis_index.report_gate_status_hash
    ):
        raise CaseStudyFactoryError("Phase 4 replay receipt differs from its native index")
    if (
        analysis_index.table_manifest_hash != table_manifest.content_hash
        or analysis_index.table_manifest_file_sha256
        != _file_sha256(paths["phase4_table_manifest"])
        or analysis_index.final_reviewed_seal_hash != independent_review.final_seal_hash
        or analysis_index.review_completion_manifest_hash
        != independent_review.content_hash
        or analysis_index.source_tree_association_hash != association_manifest_hash
        or analysis_index.source_tree_association_file_sha256
        != _file_sha256(paths["source_association"])
    ):
        raise CaseStudyFactoryError("Phase 4 analysis index differs from held-out/table lineage")
    output_by_path = {item.relative_path: item for item in analysis_index.output_files}
    for output in analysis_index.output_files:
        candidate = _safe_file(
            table_root / output.relative_path,
            label=f"Phase 4 output {output.relative_path}",
            maximum_bytes=512 * 1024 * 1024,
        )
        if _file_sha256(candidate) != output.file_sha256:
            raise CaseStudyFactoryError("Phase 4 output-file inventory failed byte replay")
        if output.row_count is not None:
            raw = candidate.read_bytes()
            if candidate.suffix == ".csv":
                observed_rows = max(0, len(raw.decode("utf-8").splitlines()) - 1)
            elif candidate.suffix == ".jsonl":
                observed_rows = sum(bool(line.strip()) for line in raw.splitlines())
            else:
                observed_rows = 1
            if observed_rows != output.row_count:
                raise CaseStudyFactoryError("Phase 4 output-file row count failed replay")
    for entry in table_manifest.tables:
        output = output_by_path.get(entry.relative_path)
        candidate = _safe_file(
            table_root / entry.relative_path,
            label=f"Phase 4 table {entry.table_id}",
        )
        if (
            output is None
            or output.file_sha256 != entry.file_sha256
            or output.logical_content_hash != entry.logical_content_hash
            or output.row_count != entry.row_count
            or _file_sha256(candidate) != entry.file_sha256
        ):
            raise CaseStudyFactoryError("Phase 4 canonical table manifest failed replay")
    metric_gate = bundle.registered_metric_regeneration
    if (
        metric_gate.held_out_results_closure_hash != heldout_gate.content_hash
        or metric_gate.synthetic_benchmark_hash != benchmark.content_hash
        or metric_gate.registered_analysis_manifest_hash != analysis_index.registered_analysis_hash
        or metric_gate.canonical_table_manifest_hash != table_manifest.content_hash
    ):
        raise CaseStudyFactoryError("registered metric gate was not regenerated from Phase 4")

    feedback_session = cast(
        Phase5FeedbackScoringSession,
        _typed_native(paths, "phase5_feedback_scoring_session", Phase5FeedbackScoringSession),
    )
    feedback_metrics = cast(
        Phase5ScriptedFeedbackMetrics,
        _typed_native(paths, "phase5_feedback_metrics", Phase5ScriptedFeedbackMetrics),
    )
    feedback_receipt = cast(
        Phase5FeedbackScoringReceipt,
        _typed_native(paths, "phase5_feedback_scoring_receipt", Phase5FeedbackScoringReceipt),
    )
    try:
        read_only_artifacts = ReadOnlyArtifactStore.from_paths(
            blob_root=blobs,
            ledger_path=ledger,
            compression=Compression.ZSTD,
        )
        with read_only_artifacts:
            (
                replayed_feedback_session,
                replayed_feedback_metrics,
                replayed_feedback_receipt,
            ) = prepare_phase5_feedback_scoring(
                repository=_safe_real_directory(
                    evidence_root,
                    label="semantic evidence root",
                ),
                restricted_root=_safe_real_directory(
                    Path(evidence_root) / "artifacts" / "restricted",
                    label="canonical restricted root",
                ),
                source_manifest_path=paths["phase5_source_manifest"],
                known_answer_source_path=paths["phase5_known_answer_source"],
                feedback_journal_root=paths["phase5_journal_index"].parent,
                output_root=paths["phase5_feedback_scoring_receipt"].parent,
                artifacts=read_only_artifacts,
            )
    except Exception as error:
        raise CaseStudyFactoryError(
            "Phase 5 feedback metrics could not be regenerated"
        ) from error
    if (
        replayed_feedback_session != feedback_session
        or replayed_feedback_metrics != feedback_metrics
        or replayed_feedback_receipt != feedback_receipt
        or feedback_session.execution_index_hash != phase5.content_hash
        or feedback_metrics.execution_index_hash != phase5.content_hash
        or feedback_receipt.session_hash != feedback_session.content_hash
        or feedback_receipt.metrics_hash != feedback_metrics.content_hash
        or feedback_receipt.metrics_file_sha256 != _file_sha256(paths["phase5_feedback_metrics"])
        or feedback_receipt.execution_index_hash != phase5.content_hash
    ):
        raise CaseStudyFactoryError("Phase 5 feedback metrics failed native lineage replay")

    taxonomy = cast(
        HeldOutErrorTaxonomy,
        _typed_native(paths, "blinded_error_taxonomy", HeldOutErrorTaxonomy),
    )
    source = cast(
        HeldOutFailureSourceManifest,
        _typed_native(
            paths,
            "blinded_error_source_manifest",
            HeldOutFailureSourceManifest,
        ),
    )
    try:
        replayed_failure_source, replayed_failure_files = (
            prepare_held_out_failure_source(
                repository=_safe_real_directory(
                    evidence_root,
                    label="semantic evidence root",
                ),
                phase4_configuration_path=(
                    Path(evidence_root)
                    / "configs"
                    / "study"
                    / "phase4_analysis.json"
                ),
                phase4_output_root=table_root,
                held_out_root=journal_root,
                selected_at=source.selected_at,
            )
        )
    except Exception as error:
        raise CaseStudyFactoryError(
            "held-out failure source could not be regenerated"
        ) from error
    if replayed_failure_source != source:
        raise CaseStudyFactoryError("held-out failure source failed producer replay")
    failure_source_root = paths["blinded_error_source_manifest"].parent
    _require_content_addressed_root(
        failure_source_root,
        expected_hash=source.content_hash,
        label="held-out failure-source bundle",
    )
    _require_exact_regular_file_inventory(
        failure_source_root,
        expected=frozenset(replayed_failure_files),
        label="held-out failure-source bundle",
    )
    for relative_path, expected_payload in replayed_failure_files.items():
        source_file = _safe_file(
            failure_source_root.joinpath(*Path(relative_path).parts),
            label=f"regenerated failure-source file {relative_path}",
        )
        if source_file.read_bytes() != expected_payload:
            raise CaseStudyFactoryError("held-out failure-source bytes failed replay")
    package = cast(
        BlindedErrorReviewPackage,
        _typed_native(paths, "blinded_error_package", BlindedErrorReviewPackage),
    )
    rejoin = cast(
        ErrorReviewRejoinMap,
        _typed_native(paths, "blinded_error_rejoin", ErrorReviewRejoinMap),
    )
    completion = cast(
        ErrorReviewCompletion,
        _typed_native(paths, "blinded_error_completion", ErrorReviewCompletion),
    )
    adjudication = cast(
        ErrorReviewAdjudication,
        _typed_native(
            paths,
            "blinded_error_adjudication",
            ErrorReviewAdjudication,
        ),
    )
    finalization = cast(
        ErrorReviewFinalization,
        _typed_native(
            paths,
            "blinded_error_finalization",
            ErrorReviewFinalization,
        ),
    )
    _validate_error_review_chronology(
        analysis_completed_at=analysis_index.completed_at,
        source_selected_at=source.selected_at,
        reviewer_completed_at=completion.completed_at,
        adjudicated_at=adjudication.adjudicated_at,
        review_gate_completed_at=bundle.blinded_error_review.completed_at,
        semantic_bundle_frozen_at=bundle.frozen_at,
    )
    package_root = paths["blinded_error_package"].parent.parent
    _require_content_addressed_root(
        package_root,
        expected_hash=package.content_hash,
        label="condition-blind error-review package",
    )
    try:
        replayed_package, replayed_rejoin, replayed_package_files = (
            prepare_error_review_package(
                restricted_root=_safe_real_directory(
                    evidence_root,
                    label="semantic evidence root",
                ),
                source_manifest_path=paths["blinded_error_source_manifest"],
                taxonomy_path=paths["blinded_error_taxonomy"],
            )
        )
        replayed_finalization, replayed_review_files = prepare_error_review_finalization(
            restricted_root=_safe_real_directory(evidence_root, label="semantic evidence root"),
            package_root=package_root,
            completion_path=paths["blinded_error_completion"],
            adjudication_path=paths["blinded_error_adjudication"],
        )
    except Exception as error:
        raise CaseStudyFactoryError("condition-blind review could not be regenerated") from error
    if replayed_package != package or replayed_rejoin != rejoin:
        raise CaseStudyFactoryError("condition-blind review package failed producer replay")
    _require_exact_regular_file_inventory(
        package_root,
        expected=frozenset(replayed_package_files),
        label="condition-blind error-review package bundle",
    )
    for relative_path, expected_payload in replayed_package_files.items():
        package_file = _safe_file(
            package_root.joinpath(*Path(relative_path).parts),
            label=f"regenerated blinded review file {relative_path}",
        )
        if package_file.read_bytes() != expected_payload:
            raise CaseStudyFactoryError(
                "condition-blind review package bytes failed producer replay"
            )
    error_final_root = paths["blinded_error_finalization"].parent
    _require_content_addressed_root(
        error_final_root,
        expected_hash=finalization.content_hash,
        label="condition-blind error-review final bundle",
    )
    _require_exact_regular_file_inventory(
        error_final_root,
        expected=frozenset(replayed_review_files),
        label="condition-blind error-review final bundle",
    )
    for relative_path, expected_payload in replayed_review_files.items():
        review_file = _safe_file(
            error_final_root.joinpath(*Path(relative_path).parts),
            label=f"regenerated blinded review final file {relative_path}",
        )
        if review_file.read_bytes() != expected_payload:
            raise CaseStudyFactoryError(
                "condition-blind review final bytes failed producer replay"
            )
    blind_ids = tuple(item.blind_item_id for item in package.items)
    rejoin_by_blind = {item.blind_item_id: item for item in rejoin.entries}
    for item in package.items:
        panel = _safe_file(
            paths["blinded_error_package"].parent / item.panel_file,
            label=f"blinded review panel {item.blind_item_id}",
        )
        payload = panel.read_bytes()
        secret = rejoin_by_blind.get(item.blind_item_id)
        if (
            hashlib.sha256(payload).hexdigest() != item.panel_sha256
            or secret is None
            or any(
                token.encode("utf-8") in payload
                for token in (
                    secret.world_id,
                    secret.context_id,
                    secret.condition.value,
                )
            )
        ):
            raise CaseStudyFactoryError("blinded review panel leaked its rejoin identity")
    alias_hash = canonical_sha256(
        tuple(
            (item.blind_item_id, item.condition.value)
            for item in sorted(rejoin.entries, key=lambda value: value.blind_item_id)
        )
    )
    if (
        package.taxonomy_hash != taxonomy.content_hash
        or package.source_manifest_hash != source.content_hash
        or source.analysis_artifact_hash != analysis_index.content_hash
        or rejoin.package_hash != package.content_hash
        or rejoin.source_manifest_hash != source.content_hash
        or completion.package_hash != package.content_hash
        or adjudication.package_hash != package.content_hash
        or adjudication.completion_hash != completion.content_hash
        or finalization.package_hash != package.content_hash
        or finalization.rejoin_map_hash != rejoin.content_hash
        or finalization.completion_hash != completion.content_hash
        or finalization.adjudication_hash != adjudication.content_hash
        or tuple(sorted(item.blind_item_id for item in completion.judgments)) != blind_ids
        or tuple(sorted(item.blind_item_id for item in adjudication.entries)) != blind_ids
        or finalization.reviewed_failure_count != len(blind_ids)
        or adjudication.adjudicator_id == completion.reviewer_id
        or finalization.canonical_table_sha256 != _file_sha256(paths["blinded_error_table"])
        or finalization != replayed_finalization
        or replayed_review_files["held_out_error_review.csv"]
        != paths["blinded_error_table"].read_bytes()
    ):
        raise CaseStudyFactoryError("condition-blind review completion/adjudication failed replay")
    review_gate = bundle.blinded_error_review
    if (
        review_gate.held_out_results_closure_hash != heldout_gate.content_hash
        or review_gate.output_receipt_inventory_hash != output_inventory_hash
        or review_gate.frozen_selection_manifest_hash != source.content_hash
        or review_gate.condition_alias_manifest_hash != alias_hash
        or review_gate.completed_response_hash != completion.content_hash
        or review_gate.adjudication_hash != adjudication.content_hash
        or review_gate.reviewed_unit_count != finalization.reviewed_failure_count
    ):
        raise CaseStudyFactoryError("blinded-review summary was not derived from completed review")

    community_template = cast(
        CommunityReviewTemplate,
        _typed_native(
            paths,
            "blinded_community_rubric_template",
            CommunityReviewTemplate,
        ),
    )
    community_source = cast(
        CommunityReviewSourceManifest,
        _typed_native(
            paths,
            "blinded_community_source_manifest",
            CommunityReviewSourceManifest,
        ),
    )
    community_package = cast(
        BlindedCommunityReviewPackage,
        _typed_native(
            paths,
            "blinded_community_package",
            BlindedCommunityReviewPackage,
        ),
    )
    community_rejoin = cast(
        CommunityReviewRejoinMap,
        _typed_native(
            paths,
            "blinded_community_rejoin",
            CommunityReviewRejoinMap,
        ),
    )
    community_completion = cast(
        CommunityReviewCompletion,
        _typed_native(
            paths,
            "blinded_community_completion",
            CommunityReviewCompletion,
        ),
    )
    community_finalization = cast(
        CommunityReviewFinalization,
        _typed_native(
            paths,
            "blinded_community_finalization",
            CommunityReviewFinalization,
        ),
    )
    community_source_root, community_package_root, community_final_root = (
        _canonical_community_review_roots(
            evidence_root=_safe_real_directory(
                evidence_root,
                label="semantic evidence root",
            ),
            rubric_template_path=paths["blinded_community_rubric_template"],
            source_manifest_path=paths["blinded_community_source_manifest"],
            package_path=paths["blinded_community_package"],
            rejoin_path=paths["blinded_community_rejoin"],
            completion_path=paths["blinded_community_completion"],
            finalization_path=paths["blinded_community_finalization"],
            table_path=paths["blinded_community_table"],
        )
    )
    for root, expected_hash, label in (
        (
            community_source_root,
            community_source.content_hash,
            "community-review source bundle",
        ),
        (
            community_package_root,
            community_package.content_hash,
            "community-review package bundle",
        ),
        (
            community_final_root,
            community_finalization.content_hash,
            "community-review final bundle",
        ),
    ):
        _require_content_addressed_root(
            root,
            expected_hash=expected_hash,
            label=label,
        )
    try:
        replayed_community_source, replayed_community_source_files = (
            prepare_community_review_source(
                repository=_safe_real_directory(
                    evidence_root,
                    label="semantic evidence root",
                ),
                phase4_configuration_path=(
                    Path(evidence_root)
                    / "configs"
                    / "study"
                    / "phase4_analysis.json"
                ),
                phase4_output_root=table_root,
                selected_at=community_source.selected_at,
            )
        )
    except Exception as error:
        raise CaseStudyFactoryError(
            "community-review source could not be regenerated"
        ) from error
    if replayed_community_source != community_source:
        raise CaseStudyFactoryError("community-review source failed producer replay")
    _require_exact_regular_file_inventory(
        community_source_root,
        expected=frozenset(replayed_community_source_files),
        label="community-review source bundle",
    )
    for relative_path, expected_payload in replayed_community_source_files.items():
        candidate = _safe_file(
            community_source_root.joinpath(*Path(relative_path).parts),
            label=f"regenerated community-source file {relative_path}",
        )
        if candidate.read_bytes() != expected_payload:
            raise CaseStudyFactoryError("community-review source bytes failed replay")

    _validate_community_review_chronology(
        analysis_completed_at=analysis_index.completed_at,
        source_selected_at=community_source.selected_at,
        reviewer_completed_at=community_completion.completed_at,
        binding_gate_completed_at=bundle.gold_firewall_audit.completed_at,
        semantic_bundle_frozen_at=bundle.frozen_at,
    )
    try:
        (
            replayed_community_package,
            replayed_community_rejoin,
            replayed_community_package_files,
        ) = prepare_community_review_package(
            restricted_root=_safe_real_directory(
                evidence_root,
                label="semantic evidence root",
            ),
            source_manifest_path=paths["blinded_community_source_manifest"],
            rubric_template_path=paths["blinded_community_rubric_template"],
        )
        (
            replayed_community_finalization,
            replayed_community_final_files,
        ) = prepare_community_review_finalization(
            restricted_root=_safe_real_directory(
                evidence_root,
                label="semantic evidence root",
            ),
            package_root=community_package_root,
            completion_path=paths["blinded_community_completion"],
        )
    except Exception as error:
        raise CaseStudyFactoryError("community review could not be regenerated") from error
    if (
        replayed_community_package != community_package
        or replayed_community_rejoin != community_rejoin
        or replayed_community_finalization != community_finalization
    ):
        raise CaseStudyFactoryError("community review failed typed producer replay")
    for root, expected_files, label in (
        (
            community_package_root,
            replayed_community_package_files,
            "community-review package bundle",
        ),
        (
            community_final_root,
            replayed_community_final_files,
            "community-review final bundle",
        ),
    ):
        _require_exact_regular_file_inventory(
            root,
            expected=frozenset(expected_files),
            label=label,
        )
        for relative_path, expected_payload in expected_files.items():
            candidate = _safe_file(
                root.joinpath(*Path(relative_path).parts),
                label=f"regenerated {label} file {relative_path}",
            )
            if candidate.read_bytes() != expected_payload:
                raise CaseStudyFactoryError(f"{label} bytes failed replay")

    expected_community_conditions = (
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    )
    community_items_by_condition = {
        condition: tuple(
            item for item in community_source.items if item.condition is condition
        )
        for condition in expected_community_conditions
    }
    source_ids = {item.source_partition_id for item in community_source.items}
    package_ids = {item.blinded_output_id for item in community_package.items}
    rejoin_ids = {item.blinded_output_id for item in community_rejoin.entries}
    completion_ids = {
        item.blinded_output_id for item in community_completion.reviews
    }
    table_payload = paths["blinded_community_table"].read_bytes()
    if (
        community_template.reviews
        or community_package.rubric_template_hash != community_template.content_hash
        or community_package.source_manifest_hash != community_source.content_hash
        or community_source.analysis_artifact_hash != analysis_index.content_hash
        or community_rejoin.package_hash != community_package.content_hash
        or community_rejoin.source_manifest_hash != community_source.content_hash
        or community_completion.package_hash != community_package.content_hash
        or community_finalization.package_hash != community_package.content_hash
        or community_finalization.rejoin_map_hash != community_rejoin.content_hash
        or community_finalization.completion_hash != community_completion.content_hash
        or community_finalization.canonical_table_sha256
        != _file_sha256(paths["blinded_community_table"])
        or replayed_community_final_files["community_blind_review.csv"]
        != table_payload
        or community_source.selected_cell_count != 4
        or community_source.partitions_per_selected_cell != 3
        or community_source.eligible_partition_count != 12
        or len(community_source.items) != 12
        or len(community_package.items) != 12
        or len(community_rejoin.entries) != 12
        or len(community_completion.reviews) != 12
        or community_finalization.reviewed_partition_count != 12
        or community_source.registered_primary_conditions
        != expected_community_conditions
        or len({(item.world_id, item.context_id) for item in community_source.items})
        != 1
        or source_ids
        != {item.source_partition_id for item in community_rejoin.entries}
        or package_ids != rejoin_ids
        or package_ids != completion_ids
        or len(table_payload.decode("utf-8").splitlines()) != 13
    ):
        raise CaseStudyFactoryError("community-review lineage or fixed design changed")
    registered_community_resolutions = set(
        community_source.registered_resolution_values
    )
    for condition, condition_items in community_items_by_condition.items():
        expected_seed = None if condition is ConditionName.C0_CLASSICAL_PRE else 1
        if (
            len(condition_items) != 3
            or {item.resolution for item in condition_items}
            != registered_community_resolutions
            or {item.seed_block for item in condition_items} != {expected_seed}
        ):
            raise CaseStudyFactoryError(
                "community-review condition, resolution, or seed balance changed"
            )
    source_by_id = {
        item.source_partition_id: item for item in community_source.items
    }
    for rejoin_entry in community_rejoin.entries:
        source_item = source_by_id[rejoin_entry.source_partition_id]
        panel_path = _safe_file(
            community_package_root
            / "reviewer"
            / next(
                item.panel_file
                for item in community_package.items
                if item.blinded_output_id == rejoin_entry.blinded_output_id
            ),
            label=f"community-review panel {rejoin_entry.blinded_output_id}",
        )
        panel_payload = panel_path.read_bytes()
        try:
            NeutralCommunityReviewPanel.model_validate_json(panel_payload)
        except Exception as error:
            raise CaseStudyFactoryError(
                "community-review panel differs from its neutral typed contract"
            ) from error
        _assert_neutral_community_panel_payload(
            panel_payload,
            forbidden_identity_values=(
                source_item.source_partition_id,
                source_item.world_id,
                source_item.context_id,
                source_item.condition.value,
                source_item.projection_hash,
                source_item.partition_hash,
            ),
        )

    original_labels = cast(
        ScorerLabelSet,
        _typed_native(paths, "gold_original_labels", ScorerLabelSet),
    )
    mutated_labels = cast(
        ScorerLabelSet,
        _typed_native(paths, "gold_mutated_labels", ScorerLabelSet),
    )
    original_artifacts = cast(
        RunArtifactFingerprint,
        _typed_native(
            paths,
            "gold_original_artifacts",
            RunArtifactFingerprint,
        ),
    )
    mutated_artifacts = cast(
        RunArtifactFingerprint,
        _typed_native(
            paths,
            "gold_mutated_artifacts",
            RunArtifactFingerprint,
        ),
    )
    recorded_firewall = cast(
        GoldLabelFirewallAudit,
        _typed_native(paths, "gold_firewall_audit", GoldLabelFirewallAudit),
    )
    replayed_firewall = audit_gold_label_firewall(
        original_labels=original_labels,
        mutated_labels=mutated_labels,
        original_artifacts=original_artifacts,
        artifacts_after_label_mutation=mutated_artifacts,
    )
    model_visible_hash = canonical_sha256(
        {
            "domain": "held_out_model_visible_semantic_inputs/v1",
            "call_spec_hashes": tuple(item.call_spec_hash for item in execution.itt_records),
            "started_request_bindings": tuple(
                (
                    item.result.evidence_packet_hash,
                    item.result.horizon_hash,
                    item.result.budget_hash,
                )
                for item in execution.itt_records
                if item.result.request_started
            ),
        }
    )
    scorer_generated = tuple(
        (item.relative_path, item.sha256)
        for item in benchmark.generated_files
        if item.namespace == "scorer_only"
    )
    scorer_only_hash = canonical_sha256(
        {
            "domain": "held_out_scorer_only_inventory/v1",
            "benchmark_files": scorer_generated,
            "phase4_analysis_index": analysis_index.content_hash,
            "feedback_scoring_receipt": feedback_receipt.content_hash,
            "blinded_review_finalization": finalization.content_hash,
            "blinded_community_finalization": community_finalization.content_hash,
            "gold_mutation_audit": recorded_firewall.content_hash,
        }
    )
    gold_gate = bundle.gold_firewall_audit
    if (
        recorded_firewall != replayed_firewall
        or gold_gate.held_out_execution_plan_hash != execution.call_manifest_hash
        or gold_gate.held_out_results_closure_hash != heldout_gate.content_hash
        or gold_gate.synthetic_benchmark_hash != benchmark.content_hash
        or gold_gate.model_visible_payload_inventory_hash != model_visible_hash
        or gold_gate.scorer_only_inventory_hash != scorer_only_hash
    ):
        raise CaseStudyFactoryError("gold firewall summary failed mutation/runtime replay")

    from story_projection_onto.experiment import GPUCallInventory
    from story_projection_onto.experiment import (
        ResourceLimits as RegisteredResourceLimits,
    )

    try:
        inventory = GPUCallInventory.load(paths["gpu_call_inventory"])
        limits = RegisteredResourceLimits.load(paths["resource_limits"])
        storage_plan = StorageAllocationPlan.load(paths["storage_allocation_plan"])
    except Exception as error:
        raise CaseStudyFactoryError(
            "registered resource configurations failed typed replay"
        ) from error
    if (
        inventory.call_class("case_c1").count != 4
        or inventory.call_class("case_c2").count != 8
        or inventory.call_class("case_full_index_c2").count != 1
    ):
        raise CaseStudyFactoryError("registered GPU call inventory lacks the Phase 6 envelope")
    phase6_reservation = storage_plan.reservation_for("phase_6")

    wal_path = ledger.with_name(ledger.name + "-wal")
    ledger_sha = _file_sha256(ledger)
    semantic_ledger_sha = ledger_sha
    with ReadOnlyArtifactStore.from_paths(
        blob_root=blobs,
        ledger_path=ledger,
        compression=Compression.ZSTD,
    ) as read_only_artifacts:
        live = read_only_artifacts.ledger
        event_inventory_hash = _gpu_inventory_hash(live)
        gpu_inventory_hash = canonical_sha256(
            {
                "domain": "cumulative_gpu_allocation_inventory/v3",
                "event_inventory_hash": event_inventory_hash,
                "allocation_journal": tuple(
                    asdict(item) for item in live.gpu_allocation_journal_records()
                ),
                "service_sessions": tuple(
                    asdict(item) for item in live.gpu_service_sessions()
                ),
                "service_journal": tuple(
                    asdict(item) for item in live.gpu_service_journal_records()
                ),
                "unresolved_allocation_count": len(live.unresolved_gpu_allocations()),
                "unresolved_service_count": len(
                    live.unresolved_gpu_service_journals()
                ),
            }
        )
        live_summary = live.gpu_summary()
        actual_seconds = live_summary.total_allocated_microseconds / 1_000_000
        unresolved_allocation_count = len(live.unresolved_gpu_allocations())
        unresolved_service_count = len(live.unresolved_gpu_service_journals())
        unresolved = unresolved_allocation_count + unresolved_service_count
        samples = tuple(
            item
            for item in live.storage_samples_with_phase_prefix("phase_6")
            if item.phase == "phase_6"
        )
    if _file_sha256(ledger) != ledger_sha or (wal_path.exists() and wal_path.stat().st_size):
        raise CaseStudyFactoryError("cumulative ledger changed during semantic replay")
    if (
        ledger_audit.gpu_total_allocated_microseconds
        != live_summary.total_allocated_microseconds
        or ledger_audit.gpu_event_count != live_summary.event_count
        or ledger_audit.gpu_service_session_count
        != live_summary.service_session_count
        or ledger_audit.unresolved_gpu_allocation_count
        != unresolved_allocation_count
        or ledger_audit.unresolved_gpu_service_count
        != unresolved_service_count
    ):
        raise CaseStudyFactoryError(
            "read-only ledger audit differs from the typed GPU accounting replay"
        )
    if unresolved or actual_seconds >= limits.hard_gpu_seconds or not samples:
        raise CaseStudyFactoryError(
            "cumulative ledger has unresolved GPU state or no storage sample"
        )
    latest = max(samples, key=lambda item: item.sampled_at)
    timing = bundle.timing_lineage_audit
    storage = bundle.storage_preflight
    schedule = bundle.gpu_schedule_admission
    expected_forecast = actual_seconds + 300 + 2310 + 240
    if (
        timing.cumulative_ledger_sha256 != semantic_ledger_sha
        or storage.cumulative_ledger_sha256 != semantic_ledger_sha
        or schedule.cumulative_ledger_sha256 != semantic_ledger_sha
        or any(
            gate.gpu_event_inventory_hash != gpu_inventory_hash
            for gate in (timing, storage, schedule)
        )
        or not math.isclose(timing.actual_allocated_gpu_seconds, actual_seconds, abs_tol=1e-6)
        or not math.isclose(schedule.actual_allocated_gpu_seconds, actual_seconds, abs_tol=1e-6)
        or not math.isclose(
            schedule.projected_scheduled_gpu_seconds,
            expected_forecast,
            abs_tol=1e-6,
        )
        or not math.isclose(schedule.projected_hard_gpu_seconds, expected_forecast, abs_tol=1e-6)
        or expected_forecast > limits.scheduled_gpu_seconds
        or expected_forecast >= limits.hard_gpu_seconds
        or storage.occupied_bytes != latest.current_occupied_bytes
        or storage.projected_occupied_bytes_after_case != latest.projected_occupied_bytes
        or storage.filesystem_free_bytes != latest.filesystem_free_bytes
        or latest.phase != "phase_6"
        or latest.additional_reserved_bytes != phase6_reservation.additional_reserved_bytes
        or storage.controlled_allocation_bytes != limits.maximum_project_allocation_bytes
        or latest.projected_occupied_bytes > limits.maximum_project_occupied_bytes
        or latest.effective_projected_headroom_bytes < limits.minimum_storage_headroom_bytes
        or not latest.allowed
    ):
        raise CaseStudyFactoryError("live ledger/storage/schedule differs from semantic admission")

    release_payload = json.loads(paths["public_release_allowlist"].read_bytes())
    release_manifest_hash = release_payload.get("manifest_sha256")
    canary_relative = release_payload.get("protected_prose_canary_relative_path")
    expected_canary_hash = release_payload.get("protected_prose_canary_manifest_hash")
    if not isinstance(canary_relative, str) or not isinstance(expected_canary_hash, str):
        raise CaseStudyFactoryError("public release admission lacks protected-prose canary lineage")
    try:
        canary_path = Path(canary_relative)
        if canary_path.is_absolute() or ".." in canary_path.parts or "\\" in canary_relative:
            raise CaseStudyFactoryError("protected prose canary path is not bounded")
        canary_manifest, canaries = load_protected_prose_canaries(
            _safe_real_directory(evidence_root, label="semantic evidence root"),
            _safe_real_directory(evidence_root, label="semantic evidence root")
            / canary_path,
        )
    except Exception as error:
        raise CaseStudyFactoryError("protected-prose canary manifest failed replay") from error
    if canary_manifest.content_hash != expected_canary_hash:
        raise CaseStudyFactoryError("protected-prose canary manifest hash changed")
    entries = load_public_entries(paths["public_release_allowlist"])
    entry_by_source = {entry.source_relative_path: entry for entry in entries}
    for table in table_manifest.tables:
        if table.release_class is ReleaseClass.PUBLIC:
            source_relative = (table_root / table.relative_path).relative_to(
                _safe_real_directory(evidence_root, label="semantic evidence root")
            ).as_posix()
            entry = entry_by_source.get(source_relative)
            if entry is None or entry.sha256 != table.file_sha256:
                raise CaseStudyFactoryError(
                    "public allowlist does not cover every canonical public table"
                )
    scan_records = scan_public_entries(evidence_root, entries, forbidden_canaries=canaries)
    scan_hash = canonical_sha256(
        {
            "source_tree_sha256": source_tree_hash,
            "allowlist_manifest_hash": release_manifest_hash,
            "protected_prose_canary_manifest_hash": canary_manifest.content_hash,
            "records": scan_records,
        }
    )
    release_gate = bundle.public_release_scan
    if (
        release_gate.canonical_table_manifest_hash != table_manifest.content_hash
        or release_gate.public_candidate_manifest_hash != release_manifest_hash
        or release_gate.release_scan_receipt_hash != scan_hash
    ):
        raise CaseStudyFactoryError("public release scan gate failed allowlist/coverage replay")


def _replay_execution_semantic_admission(
    *,
    plan: CaseStudyExecutionPlan,
    admission: CaseStudyAdmissionAttestation,
    bundle: CaseStudySemanticAdmissionBundle,
    evidence_bundle: CaseAdmissionEvidenceBundle,
    evidence_bundle_reference: CaseArtifactReference,
    repository: Path,
    restricted_root: Path,
    ledger_path: Path,
    blob_root: Path,
    transition_directory: Path,
    expected_predecessor_ledger_sha256: str,
    require_current_h1: bool,
) -> VerifiedCaseAdmissionStagingTransition:
    """Replay semantic admission from H0 and bind runtime to proven H1."""

    if (
        evidence_bundle.admission_attestation_hash != admission.content_hash
        or evidence_bundle_reference.logical_content_hash != evidence_bundle.content_hash
        or evidence_bundle_reference.object_kind != "case_admission_evidence_bundle"
    ):
        raise CaseStudyFactoryError("case evidence bundle differs from semantic admission")
    verified = verify_case_admission_staging_transition(
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        cas_root=blob_root,
        transition_directory=transition_directory,
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission.content_hash,
        semantic_bundle_hash=bundle.content_hash,
        evidence_bundle_hash=evidence_bundle.content_hash,
        evidence_bundle_artifact_hash=evidence_bundle_reference.artifact_hash,
        evidence_bundle_reference_hash=evidence_bundle_reference.content_hash,
        require_current_h1=require_current_h1,
    )
    if expected_predecessor_ledger_sha256 != verified.archived_h1_ledger_sha256:
        raise CaseStudyFactoryError("declared predecessor is not the archived staging H1")
    transition_paths = case_admission_transition_paths(
        restricted_root=restricted_root,
        transition_directory=transition_directory,
    )
    replay_case_study_semantic_admission(
        bundle=bundle,
        evidence_root=repository,
        ledger_path=transition_paths.h0_ledger,
        blob_root=blob_root,
        cumulative_ledger_override=transition_paths.h0_ledger,
    )
    return verified


def compile_case_study_semantic_admission_bundle(
    *,
    restricted_root: Path,
    evidence_root: Path,
    gate_paths: Mapping[str, Path],
    native_artifact_paths: Mapping[str, Path],
    ledger_path: Path,
    blob_root: Path,
    output_path: Path,
    bundle_id: str,
    frozen_at: datetime,
) -> CaseStudySemanticAdmissionBundle:
    """Validate and bind the eight typed pre-case gates without inventing a pass."""

    if set(gate_paths) != set(_SEMANTIC_GATE_MODELS):
        raise CaseStudyFactoryError("semantic admission requires exactly eight typed gates")
    if set(native_artifact_paths) != _NATIVE_GATE_ROLES:
        raise CaseStudyFactoryError(
            "semantic admission requires the exact native replay artifact inventory"
        )
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise CaseStudyFactoryError("semantic admission freeze time must be timezone-aware")
    root = _safe_real_directory(restricted_root, label="restricted root")
    evidence_root = _safe_real_directory(evidence_root, label="semantic evidence root")
    ledger = _safe_file(ledger_path, label="cumulative ledger", maximum_bytes=512 * 1024 * 1024)
    if ledger != _safe_file(native_artifact_paths["cumulative_ledger"], label="native ledger"):
        raise CaseStudyFactoryError("semantic native ledger path differs from the live ledger")
    output = _restricted_location(
        output_path,
        restricted_root=root,
        label="semantic admission bundle output",
        directory=False,
    )
    loaded: dict[str, ImmutableRecord] = {}
    for name, model in _SEMANTIC_GATE_MODELS.items():
        path = _safe_file(gate_paths[name], label=name)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise CaseStudyFactoryError(f"{name} must remain inside the restricted root") from error
        loaded[name] = _load_record(path, model, label=name)
    native_references = tuple(
        _native_gate_reference(
            evidence_root=evidence_root,
            role=role,
            path=native_artifact_paths[role],
        )
        for role in sorted(_NATIVE_GATE_ROLES)
    )
    bundle = CaseStudySemanticAdmissionBundle(
        bundle_id=bundle_id,
        synthetic_run_closure=cast(
            CaseSyntheticClosureGate, loaded["synthetic_run_closure_hash"]
        ),
        timing_lineage_audit=cast(
            CaseTimingLineageGate, loaded["timing_lineage_audit_hash"]
        ),
        gold_firewall_audit=cast(
            CaseGoldFirewallGate, loaded["gold_firewall_audit_hash"]
        ),
        registered_metric_regeneration=cast(
            CaseMetricRegenerationGate,
            loaded["registered_metric_regeneration_hash"],
        ),
        blinded_error_review=cast(
            CaseBlindedErrorReviewGate, loaded["blinded_error_review_hash"]
        ),
        storage_preflight=cast(
            CaseStoragePreflightGate, loaded["storage_preflight_hash"]
        ),
        gpu_schedule_admission=cast(
            CaseGpuScheduleGate, loaded["gpu_schedule_admission_hash"]
        ),
        public_release_scan=cast(
            CasePublicReleaseScanGate, loaded["public_release_scan_hash"]
        ),
        native_artifacts=native_references,
        frozen_at=frozen_at.astimezone(UTC),
    )
    replay_case_study_semantic_admission(
        bundle=bundle,
        evidence_root=evidence_root,
        ledger_path=ledger,
        blob_root=blob_root,
    )
    _atomic_private_record(output, bundle)
    return bundle


def stage_case_admission_evidence(
    *,
    restricted_root: Path,
    semantic_evidence_root: Path,
    execution_plan_path: Path,
    admission_attestation_path: Path,
    semantic_gate_bundle_path: Path,
    gate_paths: Mapping[str, Path],
    ledger_path: Path,
    artifact_root: Path,
    transition_directory: Path,
    bundle_output_path: Path,
    reference_output_path: Path,
    staged_at: datetime,
) -> tuple[
    CaseAdmissionEvidenceBundle,
    CaseArtifactReference,
    CaseAdmissionStagingReceipt,
]:
    """Copy exactly eight hash-valid gate records into the restricted CAS.

    The source gate files may be public reports, but their CAS copies are always
    marked restricted so the later admission validator has one uniform access
    boundary.  JSON is reserialized canonically; no source path is retained.
    """

    expected_names = (
        "synthetic_run_closure_hash",
        "timing_lineage_audit_hash",
        "gold_firewall_audit_hash",
        "registered_metric_regeneration_hash",
        "blinded_error_review_hash",
        "storage_preflight_hash",
        "gpu_schedule_admission_hash",
        "public_release_scan_hash",
    )
    if set(gate_paths) != set(expected_names):
        raise CaseStudyFactoryError("case admission staging requires exactly eight named gates")
    if staged_at.tzinfo is None or staged_at.utcoffset() is None:
        raise CaseStudyFactoryError("case admission staging time must be timezone-aware")
    if not bundle_output_path.is_absolute() or not reference_output_path.is_absolute():
        raise CaseStudyFactoryError("case gate outputs require explicit absolute paths")
    root = _safe_real_directory(restricted_root, label="restricted root")
    semantic_evidence_root = _safe_real_directory(
        semantic_evidence_root,
        label="semantic evidence root",
    )
    _ensure_private_directory(
        bundle_output_path.parent,
        restricted_root=root,
        label="case admission bundle output",
    )
    _ensure_private_directory(
        reference_output_path.parent,
        restricted_root=root,
        label="case admission reference output",
    )
    transition_directory = _ensure_private_directory(
        transition_directory,
        restricted_root=root,
        label="case admission staging transition",
    )
    ledger_path = _restricted_location(
        ledger_path,
        restricted_root=root,
        label="case cumulative ledger",
        directory=False,
    )
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise CaseStudyFactoryError("case gate staging requires the existing study ledger")
    artifact_root = _shared_blob_root(
        artifact_root,
        evidence_root=semantic_evidence_root,
        label="case shared CAS",
    )
    admission = load_case_study_admission_attestation(
        admission_attestation_path,
        restricted_root=restricted_root,
    )
    semantic_bundle = load_case_study_semantic_admission_bundle(
        semantic_gate_bundle_path,
        restricted_root=restricted_root,
    )
    validate_case_study_semantic_admission(admission, semantic_bundle)
    execution_plan = load_case_study_execution_plan(
        execution_plan_path,
        restricted_root=restricted_root,
    )
    if (
        execution_plan.admission_attestation_hash != admission.content_hash
        or execution_plan.semantic_gate_bundle_hash != semantic_bundle.content_hash
    ):
        raise CaseStudyFactoryError("case execution plan differs from staged admission")
    if staged_at < max(admission.attested_at, semantic_bundle.frozen_at):
        raise CaseStudyFactoryError("case gate staging predates semantic admission")

    # Materialize every byte and reference in memory before capturing H0.  The
    # transition intent can therefore declare the complete, exact H0->H1 delta
    # before any CAS or ledger mutation occurs.
    gate_payloads: dict[str, bytes] = {}
    references: list[CaseAdmissionEvidenceReference] = []
    payload_descriptors: list[CaseStagedPayloadDescriptor] = []
    for name in expected_names:
        source = _safe_file(gate_paths[name], label=f"case gate {name}")
        try:
            value = json.loads(source.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CaseStudyFactoryError(f"case gate {name} is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise CaseStudyFactoryError(f"case gate {name} must contain one JSON object")
        expected_hash = cast(str, getattr(admission, name))
        if _logical_json_hash(value) != expected_hash:
            raise CaseStudyFactoryError(f"case gate {name} differs from its attestation")
        payload = canonical_json(value).encode("utf-8") + b"\n"
        artifact_hash = hashlib.sha256(payload).hexdigest()
        gate_payloads[name] = payload
        references.append(
            CaseAdmissionEvidenceReference(
                name=cast(Any, name),
                logical_content_hash=expected_hash,
                artifact_hash=artifact_hash,
            )
        )
        payload_descriptors.append(
            CaseStagedPayloadDescriptor(
                role=cast(Any, name),
                logical_content_hash=expected_hash,
                artifact_hash=artifact_hash,
                raw_size_bytes=len(payload),
                compression=Compression.ZSTD.value,
                media_type="application/vnd.story-projection.case-admission-gate+json",
            )
        )
    bundle = CaseAdmissionEvidenceBundle(
        bundle_id=f"case-admission-evidence-{admission.content_hash[:16]}",
        admission_attestation_hash=admission.content_hash,
        evidence=tuple(references),
        frozen_at=staged_at.astimezone(UTC),
    )
    bundle_payload = canonical_json(bundle).encode("utf-8")
    bundle_artifact_hash = hashlib.sha256(bundle_payload).hexdigest()
    reference = CaseArtifactReference(
        logical_content_hash=bundle.content_hash,
        artifact_hash=bundle_artifact_hash,
        object_kind="case_admission_evidence_bundle",
    )
    payload_descriptors.append(
        CaseStagedPayloadDescriptor(
            role="admission_evidence_bundle",
            logical_content_hash=bundle.content_hash,
            artifact_hash=bundle_artifact_hash,
            raw_size_bytes=len(bundle_payload),
            compression=Compression.ZSTD.value,
            media_type="application/json",
        )
    )

    transition_paths = case_admission_transition_paths(
        restricted_root=root,
        transition_directory=transition_directory,
    )
    phase6_reservation = StorageAllocationPlan.load(
        Path(semantic_evidence_root) / "configs" / "study" / "storage_phase_allocations.json"
    ).reservation_for("phase_6")
    ledger_size = ledger_path.stat().st_size
    if (
        ledger_size > phase6_reservation.largest_atomic_temporary_bytes
        or 2 * ledger_size > phase6_reservation.declared_growth_bytes
    ):
        raise CaseStudyFactoryError(
            "case staging snapshots do not fit the registered Phase 6 storage reservation"
        )
    intent_already_existed = transition_paths.intent.exists()
    if not intent_already_existed:
        replay_case_study_semantic_admission(
            bundle=semantic_bundle,
            evidence_root=semantic_evidence_root,
            ledger_path=ledger_path,
            blob_root=artifact_root,
        )
    capture_case_admission_staging_intent(
        restricted_root=root,
        ledger_path=ledger_path,
        cas_root=artifact_root,
        transition_directory=transition_directory,
        transition_id=(
            f"case-admission-{execution_plan.content_hash[:16]}-{admission.content_hash[:16]}"
        ),
        execution_plan_hash=execution_plan.content_hash,
        admission_attestation_hash=admission.content_hash,
        semantic_bundle_hash=semantic_bundle.content_hash,
        expected_payloads=payload_descriptors,
        evidence_bundle_hash=bundle.content_hash,
        evidence_bundle_artifact_hash=bundle_artifact_hash,
        evidence_bundle_reference_hash=reference.content_hash,
        captured_at=staged_at.astimezone(UTC),
    )
    if intent_already_existed:
        replay_case_study_semantic_admission(
            bundle=semantic_bundle,
            evidence_root=semantic_evidence_root,
            ledger_path=transition_paths.h0_ledger,
            blob_root=artifact_root,
            cumulative_ledger_override=transition_paths.h0_ledger,
        )

    ledger = Ledger(ledger_path)
    try:
        if _total_allocated_seconds(ledger) <= 0:
            raise CaseStudyFactoryError("case gate staging requires the cumulative study ledger")
        artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
        for name in expected_names:
            artifact = artifacts.put_bytes(
                gate_payloads[name],
                media_type="application/vnd.story-projection.case-admission-gate+json",
                release_class=LedgerReleaseClass.RESTRICTED,
                created_at=staged_at,
            )
            expected_artifact_hash = next(
                item.artifact_hash for item in references if item.name == name
            )
            if artifact.content_hash != expected_artifact_hash:
                raise CaseStudyFactoryError("case gate CAS identity changed during staging")
        bundle_artifact = artifacts.put_bytes(
            bundle_payload,
            media_type="application/json",
            release_class=LedgerReleaseClass.RESTRICTED,
            created_at=staged_at,
        )
        if bundle_artifact.content_hash != reference.artifact_hash:
            raise CaseStudyFactoryError("case evidence-bundle CAS identity changed")
        _atomic_private_record(bundle_output_path, bundle)
        _atomic_private_record(reference_output_path, reference)
    finally:
        ledger.close()
    receipt = finalize_case_admission_staging_transition(
        restricted_root=root,
        ledger_path=ledger_path,
        cas_root=artifact_root,
        transition_directory=transition_directory,
        completed_at=staged_at.astimezone(UTC),
    )
    return bundle, reference, receipt


def _verify_interrupted_bootstrap_prefix(
    *,
    repository: Path,
    restricted_root: Path,
    runtime_root: Path,
    transition_directory: Path,
    ledger_path: Path,
    artifact_root: Path,
    plan: CaseStudyExecutionPlan,
    admission_attestation: CaseStudyAdmissionAttestation,
    evidence_bundle: CaseAdmissionEvidenceBundle,
    evidence_bundle_reference: CaseArtifactReference,
    construction: DevelopmentConstructionConfiguration,
    source_revision: str,
    source_tree_sha256: str,
    staging_transition: VerifiedCaseAdmissionStagingTransition,
) -> None:
    """Allow only the deterministic zero/one/two-artifact bootstrap prefix."""

    intent_path = runtime_root / "case-admission-bootstrap-intent.json"
    reference_path = runtime_root / "case-execution-admission-reference.json"
    if reference_path.exists() and (reference_path.is_symlink() or not reference_path.is_file()):
        raise CaseStudyFactoryError("case admission reference is not a regular private file")
    if not intent_path.is_file() or reference_path.is_file():
        return
    intent = cast(
        CaseAdmissionBootstrapIntent,
        _load_record(
            intent_path,
            CaseAdmissionBootstrapIntent,
            label="case admission bootstrap intent",
        ),
    )
    if (
        intent.factory_revision != CASE_FACTORY_REVISION
        or intent.execution_plan_hash != plan.content_hash
        or intent.admission_attestation_hash != admission_attestation.content_hash
        or intent.evidence_bundle_hash != evidence_bundle.content_hash
        or intent.evidence_bundle_reference_hash != evidence_bundle_reference.content_hash
        or intent.construction_configuration_hash != construction.content_hash
        or intent.construction_configuration_file_sha256
        != construction.source_file_sha256
        or intent.source_revision != source_revision
        or intent.source_tree_sha256 != source_tree_sha256
        or intent.staging_transition_receipt_hash
        != staging_transition.receipt.content_hash
        or intent.predecessor_ledger_sha256
        != staging_transition.archived_h1_ledger_sha256
        or not math.isclose(
            intent.allocated_gpu_seconds_before_case,
            staging_transition.archived_h1_allocated_gpu_microseconds / 1_000_000,
            abs_tol=1e-6,
        )
    ):
        raise CaseStudyFactoryError("interrupted case bootstrap intent changed")
    source = build_source_manifest(repository, source_revision)
    if source.tree_sha256 != source_tree_sha256:
        raise CaseStudyFactoryError("interrupted bootstrap source tree changed")
    source_value = source.to_dict()
    source_payload = canonical_json(source_value).encode("utf-8")
    source_reference = CaseArtifactReference(
        logical_content_hash=canonical_sha256(source_value),
        artifact_hash=hashlib.sha256(source_payload).hexdigest(),
        object_kind="case_source_manifest",
    )
    admission = CaseExecutionAdmissionReceipt(
        receipt_id=f"admission-{plan.execution_id}",
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission_attestation.content_hash,
        admission_evidence_bundle=evidence_bundle_reference,
        source_manifest=source_reference,
        source_revision=source_revision,
        construction_configuration_file_sha256=construction.source_file_sha256,
        construction_configuration_hash=construction.content_hash,
        predecessor_ledger_sha256=intent.predecessor_ledger_sha256,
        prior_gpu_event_inventory_hash=intent.prior_gpu_event_inventory_hash,
        allocated_gpu_seconds_before_case=intent.allocated_gpu_seconds_before_case,
        admitted_at=intent.admitted_at,
    )
    admission_payload = canonical_json(admission).encode("utf-8")
    verified = verify_case_post_h1_artifact_prefix(
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        cas_root=artifact_root,
        transition_directory=transition_directory,
        expected_payloads=(
            CasePostH1ArtifactDescriptor(
                role="case_source_manifest",
                logical_content_hash=source_reference.logical_content_hash,
                artifact_hash=source_reference.artifact_hash,
                raw_size_bytes=len(source_payload),
                compression=Compression.ZSTD.value,
                media_type="application/json",
            ),
            CasePostH1ArtifactDescriptor(
                role="case_execution_admission",
                logical_content_hash=admission.content_hash,
                artifact_hash=hashlib.sha256(admission_payload).hexdigest(),
                raw_size_bytes=len(admission_payload),
                compression=Compression.ZSTD.value,
                media_type="application/json",
            ),
        ),
    )
    if verified.transition_receipt_hash != staging_transition.receipt.content_hash:
        raise CaseStudyFactoryError("bootstrap prefix uses another staging transition")


def _load_bootstrap_reference(
    *,
    path: Path,
    artifacts: ArtifactStore,
    plan: CaseStudyExecutionPlan,
    admission_attestation: CaseStudyAdmissionAttestation,
    construction: DevelopmentConstructionConfiguration,
    evidence_bundle_reference: CaseArtifactReference,
    source_tree_sha256: str,
    source_revision: str,
    expected_predecessor_ledger_sha256: str,
    staging_transition: VerifiedCaseAdmissionStagingTransition,
    intent_path: Path,
    ledger: Ledger,
) -> tuple[CaseExecutionAdmissionReceipt, CaseArtifactReference] | None:
    if not path.exists():
        return None
    reference = cast(
        CaseArtifactReference,
        _load_record(path, CaseArtifactReference, label="case admission reference"),
    )
    if reference.object_kind != "case_execution_admission":
        raise CaseStudyFactoryError("case admission reference has another object kind")
    admission = cast(
        CaseExecutionAdmissionReceipt,
        _parse_record(artifacts, reference, CaseExecutionAdmissionReceipt),
    )
    if not intent_path.is_file() or intent_path.is_symlink():
        raise CaseStudyFactoryError("persisted case admission lacks its bootstrap intent")
    intent = cast(
        CaseAdmissionBootstrapIntent,
        _load_record(
            intent_path,
            CaseAdmissionBootstrapIntent,
            label="case admission bootstrap intent",
        ),
    )
    source_payload = json.loads(_read_reference(artifacts, admission.source_manifest))
    if (
        intent.factory_revision != CASE_FACTORY_REVISION
        or intent.execution_plan_hash != plan.content_hash
        or intent.admission_attestation_hash != admission_attestation.content_hash
        or intent.evidence_bundle_hash
        != evidence_bundle_reference.logical_content_hash
        or intent.evidence_bundle_reference_hash
        != evidence_bundle_reference.content_hash
        or intent.construction_configuration_hash != construction.content_hash
        or intent.construction_configuration_file_sha256
        != construction.source_file_sha256
        or intent.source_revision != source_revision
        or intent.source_tree_sha256 != source_tree_sha256
        or intent.staging_transition_receipt_hash
        != staging_transition.receipt.content_hash
        or intent.predecessor_ledger_sha256 != expected_predecessor_ledger_sha256
        or intent.predecessor_ledger_sha256
        != staging_transition.archived_h1_ledger_sha256
        or admission.execution_plan_hash != plan.content_hash
        or admission.admission_attestation_hash != admission_attestation.content_hash
        or admission.construction_configuration_hash != construction.content_hash
        or admission.construction_configuration_file_sha256 != construction.source_file_sha256
        or admission.admission_evidence_bundle != evidence_bundle_reference
        or admission.source_revision != source_revision
        or admission.predecessor_ledger_sha256 != intent.predecessor_ledger_sha256
        or admission.prior_gpu_event_inventory_hash
        != intent.prior_gpu_event_inventory_hash
        or abs(
            admission.allocated_gpu_seconds_before_case
            - intent.allocated_gpu_seconds_before_case
        )
        > 1e-6
        or admission.admitted_at != intent.admitted_at
        or not isinstance(source_payload, Mapping)
        or source_payload.get("tree_sha256") != source_tree_sha256
    ):
        raise CaseStudyFactoryError("persisted case admission differs from current frozen inputs")
    current_seconds = _total_allocated_seconds(ledger)
    if (
        not math.isfinite(current_seconds)
        or current_seconds + 1e-6 < intent.allocated_gpu_seconds_before_case
    ):
        raise CaseStudyFactoryError("cumulative GPU allocation regressed after case admission")
    if (
        abs(current_seconds - intent.allocated_gpu_seconds_before_case) <= 1e-6
        and _gpu_inventory_hash(ledger) != intent.prior_gpu_event_inventory_hash
    ):
        raise CaseStudyFactoryError("pre-case GPU event inventory changed")
    return admission, reference


def _build_or_resume_admission(
    *,
    repository: Path,
    runtime_root: Path,
    source_revision: str,
    source_tree_sha256: str,
    plan: CaseStudyExecutionPlan,
    admission_attestation: CaseStudyAdmissionAttestation,
    evidence_bundle: CaseAdmissionEvidenceBundle,
    evidence_bundle_reference: CaseArtifactReference,
    construction: DevelopmentConstructionConfiguration,
    construction_path: Path,
    ledger: Ledger,
    artifacts: ArtifactStore,
    ledger_path: Path,
    expected_predecessor_ledger_sha256: str,
    staging_transition: VerifiedCaseAdmissionStagingTransition,
    clock: Callable[[], datetime],
) -> tuple[CaseExecutionAdmissionReceipt, CaseArtifactReference]:
    reference_path = runtime_root / "case-execution-admission-reference.json"
    intent_path = runtime_root / "case-admission-bootstrap-intent.json"
    validate_case_admission_evidence(
        admission=admission_attestation,
        bundle=evidence_bundle,
        artifacts=artifacts,
    )
    if evidence_bundle_reference.logical_content_hash != evidence_bundle.content_hash:
        raise CaseStudyFactoryError("case admission bundle reference changed")
    if _read_reference(artifacts, evidence_bundle_reference) != canonical_json(
        evidence_bundle
    ).encode("utf-8"):
        raise CaseStudyFactoryError("case admission bundle CAS bytes changed")
    if construction.source_file_sha256 != _file_sha256(construction_path):
        raise CaseStudyFactoryError("case construction configuration bytes changed")
    if construction.upper_ontology.content_hash != admission_attestation.upper_ontology_hash:
        raise CaseStudyFactoryError("case upper ontology differs from admission")
    recovered = _load_bootstrap_reference(
        path=reference_path,
        artifacts=artifacts,
        plan=plan,
        admission_attestation=admission_attestation,
        construction=construction,
        evidence_bundle_reference=evidence_bundle_reference,
        source_tree_sha256=source_tree_sha256,
        source_revision=source_revision,
        expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
        staging_transition=staging_transition,
        intent_path=intent_path,
        ledger=ledger,
    )
    if recovered is not None:
        return recovered

    source = build_source_manifest(repository, source_revision)
    if source.tree_sha256 != source_tree_sha256:
        raise CaseStudyFactoryError("case source association and rebuilt source tree differ")
    prior_seconds = _total_allocated_seconds(ledger)
    if not math.isfinite(prior_seconds) or prior_seconds <= 0:
        raise CaseStudyFactoryError("case execution cannot reset cumulative GPU accounting")
    prior_inventory_hash = _gpu_inventory_hash(ledger)
    now = clock()
    if now.tzinfo is None or now.utcoffset() is None:
        raise CaseStudyFactoryError("case factory clock must be timezone-aware")
    intent = CaseAdmissionBootstrapIntent(
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission_attestation.content_hash,
        evidence_bundle_hash=evidence_bundle.content_hash,
        evidence_bundle_reference_hash=evidence_bundle_reference.content_hash,
        construction_configuration_hash=construction.content_hash,
        construction_configuration_file_sha256=construction.source_file_sha256,
        source_revision=source_revision,
        source_tree_sha256=source.tree_sha256,
        staging_transition_receipt_hash=staging_transition.receipt.content_hash,
        predecessor_ledger_sha256=staging_transition.archived_h1_ledger_sha256,
        prior_gpu_event_inventory_hash=prior_inventory_hash,
        allocated_gpu_seconds_before_case=prior_seconds,
        admitted_at=now.astimezone(UTC),
    )
    if intent_path.exists():
        retained = cast(
            CaseAdmissionBootstrapIntent,
            _load_record(
                intent_path,
                CaseAdmissionBootstrapIntent,
                label="case admission bootstrap intent",
            ),
        )
        static_fields = intent.model_dump(mode="python", exclude={"content_hash", "admitted_at"})
        retained_fields = retained.model_dump(
            mode="python", exclude={"content_hash", "admitted_at"}
        )
        if static_fields != retained_fields:
            raise CaseStudyFactoryError("case admission bootstrap intent changed")
        intent = retained
        if (
            prior_inventory_hash != intent.prior_gpu_event_inventory_hash
            or abs(prior_seconds - intent.allocated_gpu_seconds_before_case) > 1e-6
        ):
            raise CaseStudyFactoryError(
                "case admission bootstrap was interrupted after GPU accounting changed"
            )
    else:
        if (
            not staging_transition.current_matches_archived_h1
            or expected_predecessor_ledger_sha256
            != staging_transition.archived_h1_ledger_sha256
            or _file_sha256(ledger_path) != staging_transition.archived_h1_ledger_sha256
        ):
            raise CaseStudyFactoryError("cumulative predecessor is not the archived H1")
        _atomic_private_record(intent_path, intent)

    source_reference = _persist_mapping(
        artifacts,
        source.to_dict(),
        object_kind="case_source_manifest",
        created_at=intent.admitted_at,
    )
    admission = CaseExecutionAdmissionReceipt(
        receipt_id=f"admission-{plan.execution_id}",
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission_attestation.content_hash,
        admission_evidence_bundle=evidence_bundle_reference,
        source_manifest=source_reference,
        source_revision=source_revision,
        construction_configuration_file_sha256=construction.source_file_sha256,
        construction_configuration_hash=construction.content_hash,
        predecessor_ledger_sha256=intent.predecessor_ledger_sha256,
        prior_gpu_event_inventory_hash=intent.prior_gpu_event_inventory_hash,
        allocated_gpu_seconds_before_case=intent.allocated_gpu_seconds_before_case,
        admitted_at=intent.admitted_at,
    )
    reference = _persist_record(
        artifacts,
        admission,
        object_kind="case_execution_admission",
        created_at=intent.admitted_at,
    )
    _atomic_private_record(reference_path, reference)
    return admission, reference


def _verify_runtime_model(
    *,
    repository: Path,
    plan: CaseStudyExecutionPlan,
    selected_freeze: AttestedSelectedModelFreeze,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    port: int,
) -> tuple[VLLMLaunchConfiguration, Any, Any, str]:
    policy_path = repository / "configs/study/fallback_model.json"
    policy = FallbackModelPolicy.load(policy_path)
    snapshot_manifest = validate_fallback_snapshot_manifest(
        verified_model_manifest_path,
        policy=policy,
        policy_path=policy_path,
        snapshot_path=snapshot_path,
        shared_cache=shared_cache,
    )
    snapshot_hash = cast(str, snapshot_manifest["manifest_sha256"])
    if (
        selected_freeze.repository != FALLBACK_MODEL_REPOSITORY
        or selected_freeze.revision != FALLBACK_MODEL_REVISION
        or selected_freeze.payload.get("served_model_name") != FALLBACK_SERVED_MODEL_NAME
        or snapshot_hash != plan.model_runtime.selected_snapshot_manifest_hash
    ):
        raise CaseStudyFactoryError("case model snapshot differs from selected fallback freeze")
    launcher = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=snapshot_path,
        shared_cache=shared_cache,
        model_configuration_path=repository / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=snapshot_hash,
        port=port,
    )
    if launcher.configuration_hash != plan.model_runtime.selected_launcher_configuration_hash:
        raise CaseStudyFactoryError("case vLLM launcher differs from execution plan")
    tokenizer_manifest = capture_tokenizer_manifest(
        snapshot_path,
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
    )
    if tokenizer_manifest.manifest_sha256 != plan.model_runtime.selected_tokenizer_manifest_hash:
        raise CaseStudyFactoryError("case tokenizer differs from execution plan")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        revision=FALLBACK_MODEL_REVISION,
    )
    return launcher, tokenizer, tokenizer_manifest, snapshot_hash


def preflight_frozen_production_case_study_bundle(
    *,
    repository: Path,
    restricted_root: Path,
    plan_path: Path,
    index_path: Path,
    index_manifest_path: Path,
    preregistration_path: Path,
    input_attestation_path: Path,
    admission_attestation_path: Path,
    semantic_gate_bundle_path: Path,
    selected_model_freeze_path: Path,
    admission_evidence_bundle_path: Path,
    admission_evidence_bundle_reference_path: Path,
    construction_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    staging_transition_directory: Path,
    runtime_root: Path,
    quota_root: Path,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    source_association_path: Path,
    expected_predecessor_ledger_sha256: str,
    source_revision: str,
    port: int = 8000,
) -> CaseStudyProductionPreflight:
    """Replay every launch dependency without constructing writable runtime objects."""

    if repository.is_symlink():
        raise CaseStudyFactoryError("repository root cannot be a symbolic link")
    repository = repository.resolve(strict=True)
    restricted_root = _safe_real_directory(restricted_root, label="restricted root")
    ledger_path = _safe_file(
        ledger_path,
        label="case cumulative ledger",
        maximum_bytes=512 * 1024 * 1024,
    )
    try:
        ledger_path.relative_to(restricted_root)
    except ValueError as error:
        raise CaseStudyFactoryError(
            "case cumulative ledger must remain inside the restricted root"
        ) from error
    artifact_root = _shared_blob_root(
        artifact_root,
        evidence_root=repository,
        label="case shared CAS",
    )
    staging_transition_directory = _safe_real_directory(
        staging_transition_directory,
        label="case staging transition directory",
    )
    runtime_root = _safe_real_directory(runtime_root, label="case runtime root")
    for label, path in (
        ("case staging transition directory", staging_transition_directory),
        ("case runtime root", runtime_root),
    ):
        try:
            path.relative_to(restricted_root)
        except ValueError as error:
            raise CaseStudyFactoryError(f"{label} escapes restricted storage") from error
    if stat.S_IMODE(runtime_root.stat().st_mode) & 0o077:
        raise CaseStudyFactoryError("case runtime root must already be private for preflight")
    quota_root = _safe_real_directory(quota_root, label="quota root")
    shared_cache = _safe_real_directory(shared_cache, label="shared model cache")
    snapshot_path = _safe_real_directory(snapshot_path, label="model snapshot")
    for label, path in (
        ("repository", repository),
        ("restricted root", restricted_root),
        ("ledger", ledger_path),
        ("artifact root", artifact_root),
        ("staging transition", staging_transition_directory),
        ("runtime root", runtime_root),
        ("shared cache", shared_cache),
        ("snapshot", snapshot_path),
    ):
        _inside_quota(path, quota_root, label=label)
    if (
        len(expected_predecessor_ledger_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_predecessor_ledger_sha256
        )
    ):
        raise CaseStudyFactoryError("predecessor ledger hash must be lowercase SHA-256")
    if not source_revision or source_revision.strip() != source_revision:
        raise CaseStudyFactoryError("source revision must be explicit and stripped")

    plan = load_case_study_execution_plan(plan_path, restricted_root=restricted_root)
    loaded = load_attested_restricted_case_study(
        restricted_root=restricted_root,
        index_path=index_path,
        manifest_path=index_manifest_path,
        preregistration_path=preregistration_path,
        attestation_path=input_attestation_path,
    )
    admission = load_case_study_admission_attestation(
        admission_attestation_path,
        restricted_root=restricted_root,
    )
    semantic_bundle = load_case_study_semantic_admission_bundle(
        semantic_gate_bundle_path,
        restricted_root=restricted_root,
    )
    validate_case_study_semantic_admission(admission, semantic_bundle)
    evidence_bundle = cast(
        CaseAdmissionEvidenceBundle,
        _load_record(
            admission_evidence_bundle_path,
            CaseAdmissionEvidenceBundle,
            label="case admission evidence bundle",
        ),
    )
    evidence_reference = cast(
        CaseArtifactReference,
        _load_record(
            admission_evidence_bundle_reference_path,
            CaseArtifactReference,
            label="case admission evidence bundle reference",
        ),
    )
    selected_freeze = load_attested_selected_model_freeze(
        restricted_root=restricted_root,
        selected_model_freeze_path=selected_model_freeze_path,
        admission=admission,
    )
    if (
        plan.input_attestation_hash != loaded.attestation.content_hash
        or plan.admission_attestation_hash != admission.content_hash
        or plan.semantic_gate_bundle_hash != semantic_bundle.content_hash
        or plan.model_runtime.selected_model_freeze_hash
        != selected_freeze.manifest_sha256
    ):
        raise CaseStudyFactoryError("case plan differs from its exact restricted attestations")
    policy = CaseStudyRuntimePolicy.load(repository / "configs/case_study/runtime.json")
    if (
        plan.runtime_policy_hash != policy.content_hash
        or policy.production_adapter_factory != CASE_PRODUCTION_FACTORY
        or policy.service_start_watchdog_seconds != CASE_SERVICE_START_WATCHDOG_SECONDS
    ):
        raise CaseStudyFactoryError("case execution plan does not bind the production factory")
    association = validate_source_association(
        _safe_file(source_association_path, label="source association"),
        source_root=repository,
    )
    if (
        association.get("revision_label") != source_revision
        or not isinstance(association.get("local_tree_sha256"), str)
    ):
        raise CaseStudyFactoryError("source revision differs from its current association")
    source_tree_sha256 = cast(str, association["local_tree_sha256"])
    expected_construction_path = (
        repository / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
    ).resolve(strict=True)
    if construction_path.resolve(strict=True) != expected_construction_path:
        raise CaseStudyFactoryError("case construction path differs from frozen configuration")
    construction = DevelopmentConstructionConfiguration.load(expected_construction_path)
    if construction.source_file_sha256 != _file_sha256(expected_construction_path):
        raise CaseStudyFactoryError("case construction configuration bytes changed")
    if (
        construction.upper_ontology.content_hash != admission.upper_ontology_hash
        or construction.upper_ontology.content_hash != plan.model_runtime.upper_ontology_hash
    ):
        raise CaseStudyFactoryError("case upper ontology differs from frozen admission")

    controller_lock_path = _runtime_lock_path(
        runtime_root,
        ledger_path=ledger_path,
    )
    _probe_exclusive_lock(controller_lock_path, label="case-study controller lock")
    _probe_exclusive_lock(
        shared_cache / SERVICE_LOCK_FILENAME,
        label="shared vLLM service lock",
    )
    bootstrap_intent_path = runtime_root / "case-admission-bootstrap-intent.json"
    if bootstrap_intent_path.exists() and (
        bootstrap_intent_path.is_symlink() or not bootstrap_intent_path.is_file()
    ):
        raise CaseStudyFactoryError("case bootstrap intent is not a regular private file")
    bootstrap_intent_exists = bootstrap_intent_path.is_file()
    staging_transition = _replay_execution_semantic_admission(
        plan=plan,
        admission=admission,
        bundle=semantic_bundle,
        evidence_bundle=evidence_bundle,
        evidence_bundle_reference=evidence_reference,
        repository=repository,
        restricted_root=restricted_root,
        ledger_path=ledger_path,
        blob_root=artifact_root,
        transition_directory=staging_transition_directory,
        expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
        require_current_h1=not bootstrap_intent_exists,
    )
    _verify_interrupted_bootstrap_prefix(
        repository=repository,
        restricted_root=restricted_root,
        runtime_root=runtime_root,
        transition_directory=staging_transition_directory,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        plan=plan,
        admission_attestation=admission,
        evidence_bundle=evidence_bundle,
        evidence_bundle_reference=evidence_reference,
        construction=construction,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        staging_transition=staging_transition,
    )

    reference_path = runtime_root / "case-execution-admission-reference.json"
    if reference_path.exists() and (
        reference_path.is_symlink() or not reference_path.is_file()
    ):
        raise CaseStudyFactoryError("case admission reference is not a regular private file")
    with ReadOnlyArtifactStore.from_paths(
        blob_root=artifact_root,
        ledger_path=ledger_path,
        compression=Compression.ZSTD,
    ) as read_only_artifacts:
        validate_case_admission_evidence(
            admission=admission,
            bundle=evidence_bundle,
            artifacts=cast(Any, read_only_artifacts),
        )
        if _read_reference(
            cast(Any, read_only_artifacts),
            evidence_reference,
        ) != canonical_json(evidence_bundle).encode("utf-8"):
            raise CaseStudyFactoryError("case admission bundle CAS bytes changed")
        admission_reference: CaseArtifactReference | None = None
        if reference_path.is_file():
            recovered = _load_bootstrap_reference(
                path=reference_path,
                artifacts=cast(Any, read_only_artifacts),
                plan=plan,
                admission_attestation=admission,
                construction=construction,
                evidence_bundle_reference=evidence_reference,
                source_tree_sha256=source_tree_sha256,
                source_revision=source_revision,
                expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
                staging_transition=staging_transition,
                intent_path=bootstrap_intent_path,
                ledger=cast(Any, read_only_artifacts.ledger),
            )
            if recovered is None:  # pragma: no cover - guarded by is_file()
                raise CaseStudyFactoryError("case admission reference could not be replayed")
            admission_reference = recovered[1]
        actual_allocated_seconds = (
            read_only_artifacts.ledger.gpu_summary().total_allocated_seconds
        )
        if (
            read_only_artifacts.ledger.unresolved_gpu_allocations()
            or read_only_artifacts.ledger.unresolved_gpu_service_journals()
        ):
            raise CaseStudyFactoryError(
                "case cumulative GPU accounting requires recovery before launch preflight"
            )
        if (
            not math.isfinite(actual_allocated_seconds)
            or actual_allocated_seconds <= 0
            or abs(
                actual_allocated_seconds
                - staging_transition.current_allocated_gpu_microseconds / 1_000_000
            )
            > 1e-6
        ):
            raise CaseStudyFactoryError("case cumulative GPU accounting changed during preflight")
        controller_state, remaining_required_seconds = _preflight_controller_state(
            runtime_root=runtime_root,
            plan=plan,
            artifacts=read_only_artifacts,
            admission_reference=admission_reference,
        )

    limits = ResourceLimits.load(repository / "configs/study/resource_limits.json")
    projected_gpu_seconds = actual_allocated_seconds + remaining_required_seconds
    if projected_gpu_seconds > limits.scheduled_gpu_seconds:
        raise CaseStudyFactoryError("remaining case schedule exceeds nine GPU hours")
    if (
        actual_allocated_seconds >= limits.hard_gpu_seconds
        or projected_gpu_seconds >= limits.hard_gpu_seconds
    ):
        raise CaseStudyFactoryError("case GPU execution would reach the hard stop")
    phase6_reservation = StorageAllocationPlan.load(
        repository / "configs/study/storage_phase_allocations.json"
    ).reservation_for("phase_6")
    storage = StoragePreflight(
        quota_root,
        controlled_paths=(
            repository,
            restricted_root,
            ledger_path.parent,
            artifact_root,
            staging_transition_directory,
            shared_cache,
        ),
        budget=StorageBudget(
            total_allocation_bytes=limits.maximum_project_allocation_bytes,
            max_occupied_bytes=limits.maximum_project_occupied_bytes,
            min_headroom_bytes=limits.minimum_storage_headroom_bytes,
        ),
    )
    storage_report = storage.check(**phase6_reservation.preflight_arguments())
    if not storage_report.allowed:
        raise CaseStudyFactoryError("case storage preflight failed")
    launcher, tokenizer, _tokenizer_manifest, snapshot_hash = _verify_runtime_model(
        repository=repository,
        plan=plan,
        selected_freeze=selected_freeze,
        snapshot_path=snapshot_path,
        shared_cache=shared_cache,
        verified_model_manifest_path=verified_model_manifest_path,
        port=port,
    )
    close_tokenizer = getattr(tokenizer, "close", None)
    if callable(close_tokenizer):
        close_tokenizer()
    del tokenizer
    return CaseStudyProductionPreflight(
        execution_plan_hash=plan.content_hash,
        staging_transition_receipt_hash=staging_transition.receipt.content_hash,
        current_ledger_sha256=staging_transition.current_ledger_sha256,
        actual_allocated_gpu_seconds=actual_allocated_seconds,
        remaining_required_gpu_seconds=remaining_required_seconds,
        projected_storage_bytes=storage_report.projected_occupied_bytes,
        filesystem_free_bytes=storage_report.filesystem_free_bytes,
        selected_snapshot_manifest_hash=snapshot_hash,
        launcher_configuration_hash=launcher.configuration_hash,
        bootstrap_state=(
            "admitted"
            if reference_path.is_file()
            else "interrupted_prefix"
            if bootstrap_intent_exists
            else "fresh"
        ),
        controller_state=controller_state,
    )


def create_frozen_production_case_study_bundle(
    *,
    repository: Path,
    restricted_root: Path,
    plan_path: Path,
    index_path: Path,
    index_manifest_path: Path,
    preregistration_path: Path,
    input_attestation_path: Path,
    admission_attestation_path: Path,
    semantic_gate_bundle_path: Path,
    selected_model_freeze_path: Path,
    admission_evidence_bundle_path: Path,
    admission_evidence_bundle_reference_path: Path,
    construction_path: Path,
    ledger_path: Path,
    artifact_root: Path,
    staging_transition_directory: Path,
    runtime_root: Path,
    quota_root: Path,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    source_association_path: Path,
    expected_predecessor_ledger_sha256: str,
    source_revision: str,
    port: int = 8000,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CaseStudyProductionBundle:
    """Build the exact 4/8/1 controller without starting the model service.

    All path-bearing inputs are explicit.  The returned controller starts vLLM
    only after bounded-packet materialization and lossless C1 packing succeed.
    """

    if repository.is_symlink():
        raise CaseStudyFactoryError("repository root cannot be a symbolic link")
    repository = repository.resolve(strict=True)
    restricted_root = _safe_real_directory(restricted_root, label="restricted root")
    ledger_path = _restricted_location(
        ledger_path,
        restricted_root=restricted_root,
        label="case cumulative ledger",
        directory=False,
    )
    artifact_root = _shared_blob_root(
        artifact_root,
        evidence_root=repository,
        label="case shared CAS",
    )
    if not staging_transition_directory.is_dir():
        raise CaseStudyFactoryError("case staging transition directory is unavailable")
    staging_transition_directory = _restricted_location(
        staging_transition_directory,
        restricted_root=restricted_root,
        label="case staging transition directory",
        directory=True,
    )
    runtime_root = _ensure_private_directory(
        runtime_root,
        restricted_root=restricted_root,
        label="case runtime root",
    )
    if quota_root.is_symlink():
        raise CaseStudyFactoryError("quota root cannot be a symbolic link")
    quota_root = quota_root.resolve(strict=True)
    for label, path in (
        ("repository", repository),
        ("restricted root", restricted_root),
        ("ledger", ledger_path),
        ("artifact root", artifact_root),
        ("staging transition", staging_transition_directory),
        ("runtime root", runtime_root),
        ("shared cache", shared_cache),
        ("snapshot", snapshot_path),
    ):
        _inside_quota(path, quota_root, label=label)
    if (
        len(expected_predecessor_ledger_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_predecessor_ledger_sha256
        )
    ):
        raise CaseStudyFactoryError("predecessor ledger hash must be lowercase SHA-256")
    if not source_revision or source_revision.strip() != source_revision:
        raise CaseStudyFactoryError("source revision must be explicit and stripped")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise CaseStudyFactoryError("case execution requires the existing cumulative ledger")
    for label, path in (
        ("artifact root", artifact_root),
        ("shared cache", shared_cache),
        ("snapshot", snapshot_path),
    ):
        if path.is_symlink():
            raise CaseStudyFactoryError(f"{label} cannot be a symbolic link")

    plan = load_case_study_execution_plan(plan_path, restricted_root=restricted_root)
    loaded: AttestedRestrictedCaseStudy = load_attested_restricted_case_study(
        restricted_root=restricted_root,
        index_path=index_path,
        manifest_path=index_manifest_path,
        preregistration_path=preregistration_path,
        attestation_path=input_attestation_path,
    )
    admission_attestation = load_case_study_admission_attestation(
        admission_attestation_path,
        restricted_root=restricted_root,
    )
    semantic_bundle = load_case_study_semantic_admission_bundle(
        semantic_gate_bundle_path,
        restricted_root=restricted_root,
    )
    validate_case_study_semantic_admission(admission_attestation, semantic_bundle)
    evidence_bundle = cast(
        CaseAdmissionEvidenceBundle,
        _load_record(
            admission_evidence_bundle_path,
            CaseAdmissionEvidenceBundle,
            label="case admission evidence bundle",
        ),
    )
    evidence_bundle_reference = cast(
        CaseArtifactReference,
        _load_record(
            admission_evidence_bundle_reference_path,
            CaseArtifactReference,
            label="case admission evidence bundle reference",
        ),
    )
    selected_freeze = load_attested_selected_model_freeze(
        restricted_root=restricted_root,
        selected_model_freeze_path=selected_model_freeze_path,
        admission=admission_attestation,
    )
    if (
        plan.input_attestation_hash != loaded.attestation.content_hash
        or plan.admission_attestation_hash != admission_attestation.content_hash
        or plan.semantic_gate_bundle_hash != semantic_bundle.content_hash
        or plan.model_runtime.selected_model_freeze_hash != selected_freeze.manifest_sha256
    ):
        raise CaseStudyFactoryError("case plan differs from its exact restricted attestations")
    policy = CaseStudyRuntimePolicy.load(repository / "configs/case_study/runtime.json")
    if (
        plan.runtime_policy_hash != policy.content_hash
        or policy.production_adapter_factory != CASE_PRODUCTION_FACTORY
        or policy.service_start_watchdog_seconds != CASE_SERVICE_START_WATCHDOG_SECONDS
    ):
        raise CaseStudyFactoryError("case execution plan does not bind the production factory")
    association = validate_source_association(
        _safe_file(source_association_path, label="source association"),
        source_root=repository,
    )
    if (
        association.get("revision_label") != source_revision
        or not isinstance(association.get("local_tree_sha256"), str)
    ):
        raise CaseStudyFactoryError("source revision differs from its current association")
    source_tree_sha256 = cast(str, association["local_tree_sha256"])

    expected_construction_path = (
        repository / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
    ).resolve(strict=True)
    if construction_path.resolve(strict=True) != expected_construction_path:
        raise CaseStudyFactoryError("case construction path differs from the frozen configuration")
    construction = DevelopmentConstructionConfiguration.load(expected_construction_path)
    runtime_lock = _acquire_runtime_lock(runtime_root, ledger_path=ledger_path)
    ledger: Ledger | None = None
    try:
        bootstrap_intent_path = runtime_root / "case-admission-bootstrap-intent.json"
        if bootstrap_intent_path.exists() and (
            bootstrap_intent_path.is_symlink() or not bootstrap_intent_path.is_file()
        ):
            raise CaseStudyFactoryError("case bootstrap intent is not a regular private file")
        bootstrap_intent_exists = bootstrap_intent_path.is_file()
        staging_transition = _replay_execution_semantic_admission(
            plan=plan,
            admission=admission_attestation,
            bundle=semantic_bundle,
            evidence_bundle=evidence_bundle,
            evidence_bundle_reference=evidence_bundle_reference,
            repository=repository,
            restricted_root=restricted_root,
            ledger_path=ledger_path,
            blob_root=artifact_root,
            transition_directory=staging_transition_directory,
            expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
            require_current_h1=not bootstrap_intent_exists,
        )
        _verify_interrupted_bootstrap_prefix(
            repository=repository,
            restricted_root=restricted_root,
            runtime_root=runtime_root,
            transition_directory=staging_transition_directory,
            ledger_path=ledger_path,
            artifact_root=artifact_root,
            plan=plan,
            admission_attestation=admission_attestation,
            evidence_bundle=evidence_bundle,
            evidence_bundle_reference=evidence_bundle_reference,
            construction=construction,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            staging_transition=staging_transition,
        )
        limits = ResourceLimits.load(repository / "configs/study/resource_limits.json")
        phase6_reservation = StorageAllocationPlan.load(
            repository / "configs/study/storage_phase_allocations.json"
        ).reservation_for("phase_6")
        storage = StoragePreflight(
            quota_root,
            controlled_paths=(
                repository,
                restricted_root,
                ledger_path.parent,
                artifact_root,
                staging_transition_directory,
                shared_cache,
            ),
            budget=StorageBudget(
                total_allocation_bytes=limits.maximum_project_allocation_bytes,
                max_occupied_bytes=limits.maximum_project_occupied_bytes,
                min_headroom_bytes=limits.minimum_storage_headroom_bytes,
            ),
        )
        storage_report = storage.check(**phase6_reservation.preflight_arguments())
        if not storage_report.allowed:
            raise CaseStudyFactoryError("case storage preflight failed")

        ledger = Ledger(ledger_path)
    except BaseException as error:
        _cleanup_runtime_ownership(
            ledger=ledger,
            runtime_lock=runtime_lock,
            primary_error=error,
        )
        raise
    assert ledger is not None
    try:
        artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
        meter = AllocatedGPUMeter.from_limits(ledger, limits)
        admission, admission_reference = _build_or_resume_admission(
            repository=repository,
            runtime_root=runtime_root,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            plan=plan,
            admission_attestation=admission_attestation,
            evidence_bundle=evidence_bundle,
            evidence_bundle_reference=evidence_bundle_reference,
            construction=construction,
            construction_path=expected_construction_path,
            ledger=ledger,
            artifacts=artifacts,
            ledger_path=ledger_path,
            expected_predecessor_ledger_sha256=expected_predecessor_ledger_sha256,
            staging_transition=staging_transition,
            clock=clock,
        )
        storage_report = storage.check(**phase6_reservation.preflight_arguments())
        if not storage_report.allowed:
            raise CaseStudyFactoryError("case storage preflight failed after admission bootstrap")
        ledger.record_storage_sample(storage_report, phase="phase_6:factory")

        launcher, tokenizer, tokenizer_manifest, snapshot_hash = _verify_runtime_model(
            repository=repository,
            plan=plan,
            selected_freeze=selected_freeze,
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            verified_model_manifest_path=verified_model_manifest_path,
            port=port,
        )
        sampler = ResourceSampler(limits=limits, storage=storage, ledger=ledger)
        client = VLLMGuidedJSONClient(launcher.base_url)
        log_path = runtime_root / "case-study.vllm.log"
        if log_path.is_symlink():
            raise CaseStudyFactoryError("case vLLM log cannot be a symbolic link")
        service = VLLMService(
            configuration=launcher,
            client=client,
            meter=meter,
            log_path=log_path,
            startup_resource_sampler=sampler,
            preflight_endpoint_check=lambda: client.health(0.25),
            readiness_check=lambda: client.ready(
                2.0,
                model_name=FALLBACK_SERVED_MODEL_NAME,
            ),
        )
        repository_state = CaseExecutionRepository(
            plan=plan,
            artifacts=artifacts,
            restricted_root=restricted_root,
            resume_directory=runtime_root / "resume",
            state_pointer_path=runtime_root / "controller" / "current.json",
            clock=clock,
        )
        classical = ProductionCaseClassicalAdapter(
            root=repository,
            plan=plan,
            construction=construction,
            artifacts=artifacts,
            restricted_state_directory=runtime_root / "c0-state",
            clock=clock,
        )
        gpu = build_production_case_study_gpu_adapter(
            root=repository,
            plan=plan,
            loaded=loaded,
            admission=admission,
            construction=construction,
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            service=service,
            artifacts=artifacts,
            repository=repository_state,
            state_pointer_path=runtime_root / "gpu" / "current.json",
            model_manifest_hash=snapshot_hash,
            service_start_watchdog_seconds=policy.service_start_watchdog_seconds,
            clock=clock,
        )
        def token_counter(value: str) -> int:
            return len(tokenizer.encode(value, add_special_tokens=False))
        controller = CaseStudyProductionController(
            root=repository,
            plan=plan,
            loaded=loaded,
            admission=admission,
            repository=repository_state,
            ledger=ledger,
            artifacts=artifacts,
            classical=classical,
            gpu=gpu,
            token_counter=token_counter,
            service_checkpoint_path=runtime_root / "service-checkpoint.json",
            clock=clock,
        )
        return CaseStudyProductionBundle(
            controller=controller,
            admission_reference=admission_reference,
            artifacts=artifacts,
            ledger=ledger,
            service=service,
            gpu=gpu,
            runtime_root=runtime_root,
            storage_report=storage_report,
            _runtime_lock=runtime_lock,
        )
    except BaseException as error:
        _cleanup_runtime_ownership(
            ledger=ledger,
            runtime_lock=runtime_lock,
            primary_error=error,
        )
        raise


__all__ = [
    "CASE_FACTORY_REVISION",
    "CASE_PRODUCTION_FACTORY",
    "CaseAdmissionBootstrapIntent",
    "CaseStudyFactoryError",
    "CaseStudyProductionBundle",
    "CaseStudyProductionPreflight",
    "compile_case_study_semantic_admission_bundle",
    "create_frozen_production_case_study_bundle",
    "preflight_frozen_production_case_study_bundle",
    "replay_case_study_semantic_admission",
    "stage_case_admission_evidence",
]
