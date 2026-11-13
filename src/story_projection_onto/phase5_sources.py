"""Append-only production of the nine Phase 5 feedback source episodes.

This module consumes only a closed held-out journal, its frozen results gate,
the pre-output scripted-revision commitment, explicitly supplied researcher
trace instructions, and an explicitly authored scorer-only known-answer file.
It never creates a missing ontology or a missing gold answer.  If a selected
held-out parent is invalid, failed, or timed out, production stops with the
exact retained ITT/receipt hash instead of silently replacing that parent.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import (
    ModelEligibleWorldArtifact,
    RuntimeStagingManifest,
    load_staged_world,
)
from story_projection_onto.conditions.base import (
    ConditionAttemptRecord,
    ConditionPreparation,
)
from story_projection_onto.conditions.c0 import (
    ClassicalRuleConfig,
    load_classical_rule_config,
    project_sealed_c0,
)
from story_projection_onto.conditions.c1 import project_sealed_c1
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionOperator,
    EvidencePacket,
    EvidenceSnapshot,
    FeedbackAction,
    GoldContextualProjection,
    ImmutableRecord,
    OntologyProjection,
    QueryAccessEvent,
    ReleaseClass,
    Sha256Digest,
    canonical_sha256,
    to_model_visible_query,
)
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.feedback_provenance import ResearcherTraceSubmissionReceipt
from story_projection_onto.feedback_runtime import (
    FeedbackProtocolConfiguration,
    ScriptedRevisionFreeze,
)
from story_projection_onto.held_out_c0 import HeldOutC0PreparationState
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    HeldOutITTRecord,
    PreconstructedProjectionReceipt,
    ScorerBridgeAuthorization,
    _audit_journal_tree,
)
from story_projection_onto.held_out_primary import (
    HeldOutCallArtifactReceipt,
    HeldOutCallManifest,
    HeldOutCallSpec,
    HeldOutCASReference,
)
from story_projection_onto.metrics.pipeline import ScorerMetricPlan
from story_projection_onto.metrics.rare import annotations_from_gold
from story_projection_onto.phase5_commitment import (
    Phase5ScriptCommitmentManifest,
    load_phase5_script_commitment,
)
from story_projection_onto.phase5_execution import (
    CpuReprojectionInput,
    PrimaryHeldOutResultsGate,
    ScorerOnlyFeedbackBinding,
)
from story_projection_onto.phase5_production import (
    Phase5CASReference,
    Phase5EpisodeSource,
    Phase5HeldOutProjectionExport,
    Phase5MaterializationSourceManifest,
    Phase5ObjectKind,
    Phase5ScorerBindingAuthorization,
    Phase5SourceModelCallBinding,
    RestrictedPhase5CAS,
    persist_phase5_record,
    phase5_record_reference,
    validate_phase5_source_model_calls,
)
from story_projection_onto.store import ArtifactStore
from story_projection_onto.synthetic_benchmark import load_model_eligible_query
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    RevisionDraftSubmission,
    RevisionInstruction,
    VisualizationChangeKind,
    VisualizationContentScope,
    _compile_submission,
    assert_revision_anchors_resolve_in_packet,
    build_visualization_bundle,
)

DEFAULT_PHASE5_SOURCE_ROOT = Path("artifacts/restricted/phase5_sources")
SESSION_FILE_NAME = "production_session.json"
SOURCE_FILE_NAME = "source_manifest.json"
SCORER_AUTHORIZATION_FILE_NAME = "scorer_binding_authorization.json"
CALL_MANIFEST_FILE_NAME = "held_out_call_manifest.json"


class Phase5SourceProductionError(RuntimeError):
    """A Phase 5 source artifact cannot be reproduced without fabrication."""


class Phase5SelectedParentUnavailable(Phase5SourceProductionError):
    """A preregistered feedback parent has no valid ontology projection."""

    def __init__(
        self,
        *,
        episode_id: str,
        condition: ConditionName,
        outcome: RunOutcome,
        source_record_hash: str,
    ) -> None:
        self.episode_id = episode_id
        self.condition = condition
        self.outcome = outcome
        self.source_record_hash = source_record_hash
        super().__init__(
            "selected Phase 5 parent is unavailable: "
            f"episode={episode_id} condition={condition.value} "
            f"outcome={outcome.value} source_record_hash={source_record_hash}"
        )


class Phase5KnownAnswerEntry(ImmutableRecord):
    """One explicitly authored scorer-only answer, shared across conditions."""

    episode_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    scorer_binding_key: str = Field(min_length=1)
    before_gold_projection_hash: Sha256Digest
    after_gold_projection_hash: Sha256Digest
    required_target_change_hash: Sha256Digest
    before_gold_projection_source: Phase5CASReference
    after_gold_projection_source: Phase5CASReference
    required_target_change_source: Phase5CASReference

    @model_validator(mode="after")
    def typed_scorer_sources_are_restricted(self) -> Self:
        expected = (
            (self.before_gold_projection_source, "feedback_gold_projection_source"),
            (self.after_gold_projection_source, "feedback_gold_projection_source"),
            (self.required_target_change_source, "feedback_required_target_change"),
        )
        if any(
            reference.object_kind != kind
            or reference.release_class is not ReleaseClass.RESTRICTED
            for reference, kind in expected
        ):
            raise ValueError("known-answer scorer sources require typed restricted CAS refs")
        if (
            self.required_target_change_source.logical_content_hash
            != self.required_target_change_hash
        ):
            raise ValueError("target-change hash differs from its typed scorer CAS ref")
        return self


class Phase5GoldProjectionSource(ImmutableRecord):
    """Restricted executable scorer plan around one reviewed synthetic gold graph."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    source_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    role: Literal["before", "after"]
    gold_projection: GoldContextualProjection
    scorer_plan: ScorerMetricPlan
    review_completion_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    prepared_at: AwareDatetime
    condition_outputs_inspected: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_gold_plan(self) -> Self:
        expected_annotations = tuple(
            sorted(
                annotations_from_gold(self.gold_projection),
                key=lambda item: item.assertion_target_id,
            )
        )
        if (
            self.scorer_plan.gold_projection != self.gold_projection
            or self.scorer_plan.source_gold_hash != self.gold_projection.content_hash
            or self.scorer_plan.alignment_plan.source_gold_hash
            != self.gold_projection.content_hash
            or self.scorer_plan.rare_annotations != expected_annotations
        ):
            raise ValueError("Phase 5 gold source does not contain its exact scorer plan")
        return self


class Phase5FeedbackChangeTarget(ImmutableRecord):
    """Output-independent exact matcher for one supported graph-diff change."""

    target_change_id: str = Field(min_length=1)
    kind: VisualizationChangeKind
    object_kind: Literal["node", "assertion"]
    anchor_ids: tuple[str, ...]
    required: bool
    within_requested_scope: bool

    @model_validator(mode="after")
    def canonical_target(self) -> Self:
        if not self.anchor_ids or self.anchor_ids != tuple(sorted(set(self.anchor_ids))):
            raise ValueError("feedback change target anchors must be sorted and unique")
        if self.required and not self.within_requested_scope:
            raise ValueError("a required target change must be inside requested scope")
        return self


class Phase5RequiredTargetChange(ImmutableRecord):
    """Frozen scorer-only target rules; never available to a condition runner."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    target_set_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    scorer_binding_key: str = Field(min_length=1)
    action: FeedbackAction
    instruction_hash: Sha256Digest
    before_gold_projection_hash: Sha256Digest
    after_gold_projection_hash: Sha256Digest
    targets: tuple[Phase5FeedbackChangeTarget, ...]
    review_completion_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    authored_at: AwareDatetime
    condition_outputs_inspected: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_target_inventory(self) -> Self:
        ids = tuple(item.target_change_id for item in self.targets)
        signatures = tuple(
            (item.kind, item.object_kind, item.anchor_ids) for item in self.targets
        )
        if (
            not self.targets
            or self.targets != tuple(sorted(self.targets, key=lambda item: item.target_change_id))
            or len(ids) != len(set(ids))
            or len(signatures) != len(set(signatures))
            or not any(item.required for item in self.targets)
            or not any(item.within_requested_scope for item in self.targets)
        ):
            raise ValueError("required target-change inventory is incomplete or ambiguous")
        return self


class Phase5KnownAnswerSourceManifest(ImmutableRecord):
    """Restricted six-answer source; this producer never authors its semantics."""

    source_id: str = Field(min_length=1)
    protocol_hash: Sha256Digest
    review_completion_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    answers: tuple[
        Phase5KnownAnswerEntry,
        Phase5KnownAnswerEntry,
        Phase5KnownAnswerEntry,
        Phase5KnownAnswerEntry,
        Phase5KnownAnswerEntry,
        Phase5KnownAnswerEntry,
    ]
    authored_at: AwareDatetime
    condition_outputs_inspected: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_answer_inventory(self) -> Self:
        ids = tuple(item.episode_id for item in self.answers)
        keys = tuple(item.scorer_binding_key for item in self.answers)
        if len(set(ids)) != 6 or len(set(keys)) != 6:
            raise ValueError("known-answer source requires six unique episodes and keys")
        return self


class Phase5TraceInstructionBinding(ImmutableRecord):
    episode_id: str = Field(min_length=1)
    submission_hash: Sha256Digest
    submission_file_sha256: Sha256Digest
    instruction_hash: Sha256Digest
    instruction_file_sha256: Sha256Digest
    submission_receipt_hash: Sha256Digest
    submission_receipt_file_sha256: Sha256Digest


class Phase5SourceProductionSession(ImmutableRecord):
    """Stable timestamps and immutable inputs for crash-safe replay."""

    session_id: str = Field(min_length=1)
    protocol_hash: Sha256Digest
    script_commitment_manifest_hash: Sha256Digest
    primary_results_gate_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    held_out_call_manifest_hash: Sha256Digest
    known_answer_source_hash: Sha256Digest
    known_answer_source_file_sha256: Sha256Digest
    trace_instructions: tuple[
        Phase5TraceInstructionBinding,
        Phase5TraceInstructionBinding,
        Phase5TraceInstructionBinding,
    ]
    producer_source_file_sha256: Sha256Digest
    query_accessed_at: AwareDatetime
    started_at: AwareDatetime
    completed_at: AwareDatetime
    selected_parent_policy: Literal["fail_closed_without_projection"] = (
        "fail_closed_without_projection"
    )
    model_service_called: Literal[False] = False
    fabricated_projection: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def stable_session(self) -> Self:
        if not self.query_accessed_at < self.started_at <= self.completed_at:
            raise ValueError("Phase 5 source production timestamps are not monotonic")
        if len({item.episode_id for item in self.trace_instructions}) != 3:
            raise ValueError("source production requires three trace instructions")
        return self


@dataclass(frozen=True, slots=True)
class ClosedHeldOutSources:
    gate: PrimaryHeldOutResultsGate
    execution: HeldOutExecutionManifest
    bridge: ScorerBridgeAuthorization
    call_manifest: HeldOutCallManifest


@dataclass(frozen=True, slots=True)
class _CASWrite:
    value: ImmutableRecord
    object_kind: Phase5ObjectKind
    release_class: ReleaseClass


@dataclass(frozen=True, slots=True)
class _PublishedRecord:
    relative_path: Path
    value: ImmutableRecord


@dataclass(frozen=True, slots=True)
class PreparedPhase5SourceProduction:
    session: Phase5SourceProductionSession
    source: Phase5MaterializationSourceManifest
    cas_writes: tuple[_CASWrite, ...]
    published_records: tuple[_PublishedRecord, ...]


def _canonical_bytes(value: ImmutableRecord) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise Phase5SourceProductionError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5SourceProductionError(
                f"symlinked Phase 5 source path is forbidden: {path}"
            )
        if current.parent == current:
            return
        current = current.parent


def _restricted_descendant(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve(strict=True)
    _assert_no_symlink_chain(path)
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as error:
        raise Phase5SourceProductionError(
            f"{label} must remain under the restricted root"
        ) from error
    if not relative.parts:
        raise Phase5SourceProductionError(f"{label} must be below the restricted root")
    return path.absolute()


def _load_canonical(path: Path, model_type: type[Any]) -> Any:
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5SourceProductionError(f"required Phase 5 source is absent: {path}")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise Phase5SourceProductionError(
            f"restricted Phase 5 source permissions are too broad: {path.name}"
        )
    raw = path.read_bytes()
    if len(raw) > 100 * 1024 * 1024:
        raise Phase5SourceProductionError(f"Phase 5 source is oversized: {path.name}")
    try:
        value = model_type.model_validate_json(raw)
    except Exception as error:
        raise Phase5SourceProductionError(
            f"invalid {model_type.__name__}: {path.name}"
        ) from error
    if raw != _canonical_bytes(value):
        raise Phase5SourceProductionError(
            f"Phase 5 source is not canonical: {path.name}"
        )
    return value


def _safe_relative_child(root: Path, relative: str) -> Path:
    logical = Path(relative)
    if logical.is_absolute() or ".." in logical.parts or "\\" in relative:
        raise Phase5SourceProductionError("held-out result gate contains an unsafe path")
    path = root / logical
    _assert_no_symlink_chain(path)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise Phase5SourceProductionError(
            "held-out result artifact escaped its result root"
        ) from error
    return path


def load_closed_held_out_sources(
    *,
    primary_results_gate_path: Path,
    held_out_journal_root: Path,
) -> ClosedHeldOutSources:
    """Replay the closed result copy and exact append-only journal inventory."""

    gate = cast(
        PrimaryHeldOutResultsGate,
        _load_canonical(primary_results_gate_path, PrimaryHeldOutResultsGate),
    )
    result_root = primary_results_gate_path.absolute().parent
    execution_path = _safe_relative_child(
        result_root, gate.held_out_execution_manifest_relative_path
    )
    bridge_path = _safe_relative_child(result_root, gate.scorer_bridge_relative_path)
    execution = cast(
        HeldOutExecutionManifest,
        _load_canonical(execution_path, HeldOutExecutionManifest),
    )
    bridge = cast(
        ScorerBridgeAuthorization,
        _load_canonical(bridge_path, ScorerBridgeAuthorization),
    )
    if (
        _file_sha256(execution_path)
        != gate.held_out_execution_manifest_file_sha256
        or _file_sha256(bridge_path) != gate.scorer_bridge_file_sha256
        or execution.content_hash != gate.held_out_execution_manifest_hash
        or bridge.content_hash != gate.scorer_bridge_hash
        or bridge.execution_manifest_hash != execution.content_hash
        or bridge.call_manifest_hash != execution.call_manifest_hash
        or bridge.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
        or gate.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
        or bridge.authorized_at < execution.completed_at
        or gate.frozen_at < bridge.authorized_at
    ):
        raise Phase5SourceProductionError("held-out result gate lineage changed")

    journal = held_out_journal_root.absolute()
    _assert_no_symlink_chain(journal)
    call_manifest = cast(
        HeldOutCallManifest,
        _load_canonical(journal / "call_manifest.json", HeldOutCallManifest),
    )
    journal_execution = cast(
        HeldOutExecutionManifest,
        _load_canonical(journal / "execution_manifest.json", HeldOutExecutionManifest),
    )
    if journal_execution != execution or call_manifest.content_hash != execution.call_manifest_hash:
        raise Phase5SourceProductionError(
            "closed held-out journal differs from its result-gate copy"
        )
    try:
        _audit_journal_tree(journal, call_manifest, require_complete=True)
    except Exception as error:
        raise Phase5SourceProductionError("closed held-out journal is not exact") from error

    expected_outputs = tuple(
        item.result.output_artifact_hash
        for item in execution.itt_records
        if item.result.output_artifact_hash is not None
    ) + tuple(
        item.projection_artifact_hash
        for item in execution.preconstructed_projections
        if item.projection_artifact_hash is not None
    )
    if (
        bridge.output_artifact_hashes != expected_outputs
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
        raise Phase5SourceProductionError("scorer bridge inventory changed")
    return ClosedHeldOutSources(
        gate=gate,
        execution=execution,
        bridge=bridge,
        call_manifest=call_manifest,
    )


def load_phase5_known_answer_source(path: Path) -> Phase5KnownAnswerSourceManifest:
    return cast(
        Phase5KnownAnswerSourceManifest,
        _load_canonical(path, Phase5KnownAnswerSourceManifest),
    )


def _artifact_bytes(
    artifacts: ArtifactStore,
    artifact_hash: str,
    *,
    expected_media_type: str | None = None,
    expected_release: ReleaseClass | None = None,
) -> bytes:
    try:
        record = artifacts.ledger.get_artifact(artifact_hash)
    except KeyError as error:
        raise Phase5SourceProductionError(
            f"held-out CAS artifact is absent: {artifact_hash}"
        ) from error
    if (
        record.content_hash != artifact_hash
        or (expected_media_type is not None and record.media_type != expected_media_type)
        or (expected_release is not None and record.release_class is not expected_release)
    ):
        raise Phase5SourceProductionError("held-out CAS metadata changed")
    try:
        raw = artifacts.blobs.read_bytes(
            record,
            allow_restricted=record.release_class is ReleaseClass.RESTRICTED,
        )
    except Exception as error:
        raise Phase5SourceProductionError("held-out CAS bytes failed verification") from error
    if hashlib.sha256(raw).hexdigest() != artifact_hash:
        raise Phase5SourceProductionError("held-out CAS physical hash changed")
    return raw


def _load_artifact_model(
    artifacts: ArtifactStore,
    artifact_hash: str,
    model_type: type[Any],
    *,
    expected_media_type: str | None = None,
    expected_release: ReleaseClass | None = None,
) -> Any:
    raw = _artifact_bytes(
        artifacts,
        artifact_hash,
        expected_media_type=expected_media_type,
        expected_release=expected_release,
    )
    try:
        return model_type.model_validate_json(raw)
    except Exception as error:
        raise Phase5SourceProductionError(
            f"held-out CAS does not contain {model_type.__name__}"
        ) from error


def _load_held_out_reference(
    artifacts: ArtifactStore,
    reference: HeldOutCASReference,
    model_type: type[Any] | None = None,
    *,
    expected_release: ReleaseClass | None = None,
) -> Any:
    release = ReleaseClass(reference.release_class)
    raw = _artifact_bytes(
        artifacts,
        reference.artifact_hash,
        expected_media_type=reference.media_type,
        expected_release=expected_release or release,
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase5SourceProductionError("held-out typed CAS payload is not JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("content_hash") != reference.logical_content_hash
    ):
        raise Phase5SourceProductionError("held-out typed CAS logical hash changed")
    if model_type is None:
        return payload
    try:
        value = model_type.model_validate(payload)
    except Exception as error:
        raise Phase5SourceProductionError(
            f"held-out typed CAS does not contain {model_type.__name__}"
        ) from error
    if value.content_hash != reference.logical_content_hash:
        raise Phase5SourceProductionError("held-out typed CAS object hash changed")
    return value


def _journal_record(
    root: Path,
    relative: Path,
    model_type: type[Any],
    expected: Any,
) -> Any:
    value = _load_canonical(root / relative, model_type)
    if value != expected:
        raise Phase5SourceProductionError(
            f"held-out journal record changed: {relative.as_posix()}"
        )
    return value


def _load_public_snapshot(
    repository: Path,
    unit: Any,
) -> tuple[EvidenceSnapshot, ModelEligibleWorldArtifact]:
    stage = repository / unit.prequery_stage.relative_path
    _assert_no_symlink_chain(stage)
    manifest_path = stage / "manifest.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != unit.prequery_stage.manifest_file_sha256:
        raise Phase5SourceProductionError("prequery stage manifest file hash changed")
    try:
        manifest = RuntimeStagingManifest.model_validate_json(raw)
        visible = load_staged_world(stage / "evidence.json", stage, manifest)
    except Exception as error:
        raise Phase5SourceProductionError("public prequery evidence failed replay") from error
    if (
        manifest.content_hash != unit.prequery_stage.staging_manifest_hash
        or visible.content_hash != unit.prequery_stage.evidence_artifact_hash
        or visible.snapshot.content_hash != unit.prequery_stage.snapshot_hash
        or visible.snapshot.release_class is not ReleaseClass.PUBLIC
    ):
        raise Phase5SourceProductionError("public prequery snapshot lineage changed")
    return visible.snapshot, visible


def _source_call_bindings(
    *,
    result: Any,
    artifacts: ArtifactStore,
) -> tuple[Phase5SourceModelCallBinding, ...]:
    receipt = result.artifact_receipt
    if receipt is None or result.ledger_receipt_artifact_hash is None:
        raise Phase5SourceProductionError("held-out result lacks its typed audit receipt")
    persisted = _load_artifact_model(
        artifacts,
        result.ledger_receipt_artifact_hash,
        HeldOutCallArtifactReceipt,
        expected_media_type=(
            "application/vnd.story-projection.held-out-call-audit-receipt+json"
        ),
        expected_release=ReleaseClass.RESTRICTED,
    )
    if persisted != receipt or persisted.content_hash != result.ledger_receipt_hash:
        raise Phase5SourceProductionError("held-out audit receipt CAS changed")
    references = (
        receipt.semantic_request,
        receipt.packing_report,
        receipt.output,
        receipt.validation,
        receipt.raw_response,
        receipt.repair_semantic_request,
        receipt.repair_packing_report,
        receipt.repair_raw_response,
    )
    for reference in references:
        if reference is not None:
            _load_held_out_reference(artifacts, reference)
    values = [
        Phase5SourceModelCallBinding(
            model_call_id=receipt.model_call_id,
            model_call_record_hash=receipt.model_call_record_hash,
            gpu_event_id=receipt.gpu_event_id,
            gpu_event_record_hash=receipt.gpu_event_hash,
        )
    ]
    if receipt.repair_model_call_id is not None:
        if (
            receipt.repair_model_call_record_hash is None
            or receipt.repair_gpu_event_id is None
            or receipt.repair_gpu_event_hash is None
        ):
            raise Phase5SourceProductionError("held-out repair receipt is incomplete")
        values.append(
            Phase5SourceModelCallBinding(
                model_call_id=receipt.repair_model_call_id,
                model_call_record_hash=receipt.repair_model_call_record_hash,
                gpu_event_id=receipt.repair_gpu_event_id,
                gpu_event_record_hash=receipt.repair_gpu_event_hash,
            )
        )
    return tuple(values)


def _selected_openings(
    protocol: FeedbackProtocolConfiguration,
    closed: ClosedHeldOutSources,
) -> dict[str, tuple[Any, Any, Any]]:
    planned = {
        stage.staging_manifest_hash: (unit, stage)
        for unit in closed.call_manifest.units
        for stage in unit.query_stages
    }
    by_context: dict[str, tuple[Any, Any, Any]] = {}
    for opening in closed.execution.query_openings:
        context = opening.query_context
        pair = planned.get(opening.sealed_stage_hash)
        if context is None or pair is None:
            raise Phase5SourceProductionError("held-out query opening lacks planned semantics")
        unit, stage = pair
        static_fields = (
            "relative_path",
            "manifest_file_sha256",
            "staging_manifest_hash",
            "stage_id",
            "stage_kind",
            "evidence_artifact_hash",
            "snapshot_hash",
            "query_artifact_hash",
        )
        if any(
            getattr(opening.opened_stage, name) != getattr(stage, name)
            for name in static_fields
        ):
            raise Phase5SourceProductionError("opened held-out stage changed from its plan")
        if context.context_id in by_context:
            raise Phase5SourceProductionError("held-out query context is duplicated")
        by_context[context.context_id] = (unit, stage, opening)
    expected = {
        item.context_id
        for item in (*protocol.scripted_episodes, *protocol.researcher_trace_slots)
    }
    if not expected.issubset(by_context):
        raise Phase5SourceProductionError("Phase 5 context is absent from held-out openings")
    return {key: by_context[key] for key in expected}


def _unavailable(
    *,
    episode_id: str,
    condition: ConditionName,
    outcome: RunOutcome,
    source_record_hash: str,
) -> None:
    raise Phase5SelectedParentUnavailable(
        episode_id=episode_id,
        condition=condition,
        outcome=outcome,
        source_record_hash=source_record_hash,
    )


@dataclass(frozen=True, slots=True)
class _EpisodeBoundary:
    episode_id: str
    unit: Any
    stage: Any
    opening: Any
    packet: EvidencePacket
    snapshot: EvidenceSnapshot


@dataclass(frozen=True, slots=True)
class _ResolvedExport:
    export: Phase5HeldOutProjectionExport
    projection: OntologyProjection
    preparation: ConditionPreparation | None
    cas_writes: tuple[_CASWrite, ...]


@dataclass(frozen=True, slots=True)
class _CpuDraft:
    condition: ConditionName
    before_projection: OntologyProjection
    after_projection: OntologyProjection | None
    instruction_hash: str
    source_query_stage_hash: str
    source_preparation_hash: str
    resolver_hash: str


def _episode_boundary(
    *,
    episode_id: str,
    context_id: str,
    selected: dict[str, tuple[Any, Any, Any]],
    repository: Path,
    artifacts: ArtifactStore,
) -> _EpisodeBoundary:
    unit, stage, opening = selected[context_id]
    packet = cast(
        EvidencePacket,
        _load_held_out_reference(
            artifacts,
            opening.evidence_packet_artifact,
            EvidencePacket,
            expected_release=ReleaseClass.PUBLIC,
        ),
    )
    query_access = _load_held_out_reference(
        artifacts,
        opening.query_access_artifact,
        QueryAccessEvent,
    )
    if query_access != opening.query_access_event:
        raise Phase5SourceProductionError("held-out query-access CAS changed")
    if opening.packet_materialization_artifact is not None:
        materialization = _load_held_out_reference(
            artifacts,
            opening.packet_materialization_artifact,
            type(opening.packet_materialization),
        )
        if materialization != opening.packet_materialization:
            raise Phase5SourceProductionError(
                "held-out packet-materialization CAS changed"
            )
    snapshot, visible = _load_public_snapshot(repository, unit)
    query_stage = repository / stage.relative_path
    manifest_raw = (query_stage / "manifest.json").read_bytes()
    try:
        query_manifest = RuntimeStagingManifest.model_validate_json(manifest_raw)
        query_visible, reveal = load_model_eligible_query(
            query_stage,
            query_stage.parents[2],
        )
    except Exception as error:
        raise Phase5SourceProductionError("public held-out query stage failed replay") from error
    if (
        packet.content_hash != opening.evidence_packet_hash
        or packet.snapshot_hash != snapshot.content_hash
        or packet.evidence != visible.evidence
        or packet.ordered_evidence_ids != snapshot.eligible_evidence_ids
        or opening.opened_stage.snapshot_hash != snapshot.content_hash
        or opening.opened_stage.evidence_artifact_hash != visible.content_hash
        or hashlib.sha256(manifest_raw).hexdigest() != stage.manifest_file_sha256
        or query_manifest.content_hash != stage.staging_manifest_hash
        or query_visible != visible
        or reveal.content_hash != stage.query_artifact_hash
        or to_model_visible_query(opening.query_context) != reveal.query
        or query_visible.content_hash not in query_manifest.artifact_hashes
        or reveal.content_hash not in query_manifest.artifact_hashes
    ):
        raise Phase5SourceProductionError("held-out packet/snapshot boundary changed")
    return _EpisodeBoundary(
        episode_id=episode_id,
        unit=unit,
        stage=stage,
        opening=opening,
        packet=packet,
        snapshot=snapshot,
    )


def _projection_receipt_path(boundary: _EpisodeBoundary, condition: ConditionName) -> Path:
    suffix = (
        "c0" if condition is ConditionName.C0_CLASSICAL_PRE else "c1-s1"
    )
    return Path("projections") / (
        f"{boundary.unit.unit_id}-{boundary.stage.stage_id}-{suffix}.json"
    )


def _itt_path(call: HeldOutCallSpec) -> Path:
    return Path("itt") / f"{call.ordinal:03d}-{call.call_id}.json"


def _find_call(
    closed: ClosedHeldOutSources,
    *,
    boundary: _EpisodeBoundary,
    call_class: str,
) -> tuple[HeldOutCallSpec, HeldOutITTRecord]:
    matches = tuple(
        call
        for call in closed.call_manifest.calls
        if call.call_class == call_class
        and call.unit_id == boundary.unit.unit_id
        and call.seed_block == 1
        and (
            call_class == "test_c1"
            or call.query_stage_hash == boundary.stage.staging_manifest_hash
        )
    )
    if len(matches) != 1:
        raise Phase5SourceProductionError("selected held-out parent call is not unique")
    call = matches[0]
    itt = closed.execution.itt_records[call.ordinal - 1]
    if (
        itt.ordinal != call.ordinal
        or itt.call_spec_hash != call.content_hash
        or itt.result.call_id != call.call_id
        or itt.result.condition is not call.condition
    ):
        raise Phase5SourceProductionError("selected held-out ITT lineage changed")
    return call, itt


def _common_cas_writes(
    projection: OntologyProjection,
    boundary: _EpisodeBoundary,
) -> tuple[_CASWrite, _CASWrite, _CASWrite]:
    return (
        _CASWrite(projection, "ontology_projection", ReleaseClass.RESTRICTED),
        _CASWrite(boundary.packet, "evidence_packet", ReleaseClass.PUBLIC),
        _CASWrite(boundary.snapshot, "evidence_snapshot", ReleaseClass.PUBLIC),
    )


def _build_export(
    *,
    boundary: _EpisodeBoundary,
    condition: ConditionName,
    closed: ClosedHeldOutSources,
    journal_root: Path,
    artifacts: ArtifactStore,
    exported_at: datetime,
    expected_frozen_seed: int,
) -> _ResolvedExport:
    execution = closed.execution
    packet_ref = phase5_record_reference(
        boundary.packet,
        object_kind="evidence_packet",
        release_class=ReleaseClass.PUBLIC,
    )
    snapshot_ref = phase5_record_reference(
        boundary.snapshot,
        object_kind="evidence_snapshot",
        release_class=ReleaseClass.PUBLIC,
    )
    preparation: ConditionPreparation | None = None

    if condition in {ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE}:
        expected_seed = None if condition is ConditionName.C0_CLASSICAL_PRE else 1
        receipt_matches = tuple(
            item
            for item in execution.preconstructed_projections
            if item.unit_id == boundary.unit.unit_id
            and item.query_stage_hash == boundary.stage.staging_manifest_hash
            and item.condition is condition
            and item.seed_block == expected_seed
        )
        if len(receipt_matches) != 1:
            raise Phase5SourceProductionError(
                "selected preconstructed projection receipt is not unique"
            )
        receipt = receipt_matches[0]
        _journal_record(
            journal_root,
            _projection_receipt_path(boundary, condition),
            PreconstructedProjectionReceipt,
            receipt,
        )
        if receipt.outcome is not RunOutcome.SUCCEEDED:
            _unavailable(
                episode_id=boundary.episode_id,
                condition=condition,
                outcome=receipt.outcome,
                source_record_hash=receipt.content_hash,
            )
        assert receipt.projection_artifact_hash is not None
        projection = cast(
            OntologyProjection,
            _load_artifact_model(
                artifacts,
                receipt.projection_artifact_hash,
                OntologyProjection,
                expected_media_type=(
                    "application/vnd.story-projection.ontology-projection+json"
                ),
                expected_release=ReleaseClass.RESTRICTED,
            ),
        )
        if (
            projection.condition is not condition
            or projection.context_hash != boundary.opening.query_context.content_hash
            or projection.packet_hash != boundary.packet.content_hash
            or projection.snapshot_hash != boundary.snapshot.content_hash
            or projection.construction_seal is None
            or projection.construction_seal.content_hash
            != receipt.source_construction_seal_hash
            or projection.construction_seal.ontology_hash
            != receipt.source_complete_graph_hash
            or receipt.evidence_packet_hash != boundary.packet.content_hash
            or receipt.horizon_hash != boundary.snapshot.horizon.content_hash
            or receipt.budget_hash != boundary.opening.query_context.budgets.content_hash
        ):
            raise Phase5SourceProductionError(
                "selected preconstructed projection differs from its receipt"
            )
        projection_ref = phase5_record_reference(
            projection,
            object_kind="ontology_projection",
        )
        if projection_ref.artifact_hash != receipt.projection_artifact_hash:
            raise Phase5SourceProductionError(
                "selected preconstructed projection serialization changed"
            )
        if condition is ConditionName.C0_CLASSICAL_PRE:
            export = Phase5HeldOutProjectionExport(
                export_id=f"phase5-export-{boundary.episode_id}-c0",
                condition=condition,
                unit_id=boundary.unit.unit_id,
                query_stage_hash=boundary.stage.staging_manifest_hash,
                seed_block=None,
                projection_receipt_hash=receipt.content_hash,
                source_output_artifact_hash=receipt.projection_artifact_hash,
                projection=projection_ref,
                packet=packet_ref,
                snapshot=snapshot_ref,
                source_completed_at=receipt.completed_at,
                exported_at=exported_at,
            )
            return _ResolvedExport(
                export=export,
                projection=projection,
                preparation=None,
                cas_writes=_common_cas_writes(projection, boundary),
            )

        call, itt = _find_call(closed, boundary=boundary, call_class="test_c1")
        _journal_record(journal_root, _itt_path(call), HeldOutITTRecord, itt)
        if itt.result.outcome is not RunOutcome.SUCCEEDED:
            _unavailable(
                episode_id=boundary.episode_id,
                condition=condition,
                outcome=itt.result.outcome,
                source_record_hash=itt.content_hash,
            )
        output_reference = cast(Any, itt.result.artifact_receipt).output
        if output_reference is None:
            raise Phase5SourceProductionError("successful C1 parent lacks its preparation")
        preparation = cast(
            ConditionPreparation,
            _load_held_out_reference(
                artifacts,
                output_reference,
                ConditionPreparation,
                expected_release=ReleaseClass.RESTRICTED,
            ),
        )
        preontology = preparation.sealed_preontology
        if (
            preontology is None
            or preparation.condition is not condition
            or preontology.construction_seal != projection.construction_seal
            or itt.result.output_artifact_hash != preparation.content_hash
            or itt.result.construction_seal_hash
            != preontology.construction_seal.content_hash
            or itt.result.complete_c1_graph_hash
            != preontology.construction_seal.ontology_hash
            or call.frozen_seed != expected_frozen_seed
        ):
            raise Phase5SourceProductionError("selected C1 preparation lineage changed")
        source_calls = _source_call_bindings(result=itt.result, artifacts=artifacts)
        export = Phase5HeldOutProjectionExport(
            export_id=f"phase5-export-{boundary.episode_id}-c1",
            condition=condition,
            unit_id=boundary.unit.unit_id,
            query_stage_hash=boundary.stage.staging_manifest_hash,
            seed_block=1,
            projection_receipt_hash=receipt.content_hash,
            itt_record_hash=itt.content_hash,
            source_output_artifact_hash=receipt.projection_artifact_hash,
            source_condition_output_artifact_hash=itt.result.output_artifact_hash,
            source_validation_artifact_hash=itt.result.validation_artifact_hash,
            source_ledger_receipt_hash=itt.result.ledger_receipt_hash,
            source_model_calls=source_calls,
            projection=projection_ref,
            packet=packet_ref,
            snapshot=snapshot_ref,
            source_completed_at=max(receipt.completed_at, itt.result.completed_at),
            exported_at=exported_at,
        )
        validate_phase5_source_model_calls(
            export=export,
            source_result=itt,
            call=call,
            artifacts=artifacts,
        )
        return _ResolvedExport(
            export=export,
            projection=projection,
            preparation=preparation,
            cas_writes=_common_cas_writes(projection, boundary),
        )

    if condition is not ConditionName.C2_LLM_QUERY:
        raise Phase5SourceProductionError("Phase 5 export condition is unsupported")
    call, itt = _find_call(closed, boundary=boundary, call_class="test_c2")
    _journal_record(journal_root, _itt_path(call), HeldOutITTRecord, itt)
    if itt.result.outcome is not RunOutcome.SUCCEEDED:
        _unavailable(
            episode_id=boundary.episode_id,
            condition=condition,
            outcome=itt.result.outcome,
            source_record_hash=itt.content_hash,
        )
    output_reference = cast(Any, itt.result.artifact_receipt).output
    if output_reference is None:
        raise Phase5SourceProductionError("successful C2 parent lacks its output")
    attempt = cast(
        ConditionAttemptRecord,
        _load_held_out_reference(
            artifacts,
            output_reference,
            ConditionAttemptRecord,
            expected_release=ReleaseClass.RESTRICTED,
        ),
    )
    projection = attempt.projection
    if (
        attempt.outcome is not RunOutcome.SUCCEEDED
        or projection is None
        or attempt.condition is not condition
        or projection.condition is not condition
        or projection.context_hash != boundary.opening.query_context.content_hash
        or projection.packet_hash != boundary.packet.content_hash
        or projection.snapshot_hash != boundary.snapshot.content_hash
        or itt.result.output_artifact_hash != attempt.content_hash
        or itt.result.evidence_packet_hash != boundary.packet.content_hash
        or itt.result.horizon_hash != boundary.snapshot.horizon.content_hash
        or itt.result.budget_hash != boundary.opening.query_context.budgets.content_hash
        or projection.construction_certificate is None
        or projection.pre_query_inventory is None
        or projection.construction_certificate.content_hash
        != itt.result.construction_certificate_hash
        or projection.pre_query_inventory.content_hash
        != itt.result.empty_prequery_inventory_hash
        or call.frozen_seed != expected_frozen_seed
    ):
        raise Phase5SourceProductionError("selected C2 attempt lineage changed")
    projection_ref = phase5_record_reference(
        projection,
        object_kind="ontology_projection",
    )
    source_calls = _source_call_bindings(result=itt.result, artifacts=artifacts)
    export = Phase5HeldOutProjectionExport(
        export_id=f"phase5-export-{boundary.episode_id}-c2",
        condition=condition,
        unit_id=boundary.unit.unit_id,
        query_stage_hash=boundary.stage.staging_manifest_hash,
        seed_block=1,
        itt_record_hash=itt.content_hash,
        source_output_artifact_hash=itt.result.output_artifact_hash,
        source_condition_output_artifact_hash=itt.result.output_artifact_hash,
        source_validation_artifact_hash=itt.result.validation_artifact_hash,
        source_ledger_receipt_hash=itt.result.ledger_receipt_hash,
        source_model_calls=source_calls,
        projection=projection_ref,
        packet=packet_ref,
        snapshot=snapshot_ref,
        source_completed_at=itt.result.completed_at,
        exported_at=exported_at,
    )
    validate_phase5_source_model_calls(
        export=export,
        source_result=itt,
        call=call,
        artifacts=artifacts,
    )
    return _ResolvedExport(
        export=export,
        projection=projection,
        preparation=None,
        cas_writes=_common_cas_writes(projection, boundary),
    )


def _prequery_binding(
    closed: ClosedHeldOutSources,
    *,
    unit_id: str,
    condition: ConditionName,
    seed_block: int | None,
) -> Any:
    matches = tuple(
        item
        for item in closed.execution.prequery_barrier.preparation_bindings
        if item.unit_id == unit_id
        and item.condition is condition
        and item.seed_block == seed_block
    )
    if len(matches) != 1:
        raise Phase5SourceProductionError(
            "selected parent has no unique prequery-barrier binding"
        )
    return matches[0]


def _load_c0_preparation(
    *,
    boundary: _EpisodeBoundary,
    closed: ClosedHeldOutSources,
    c0_state_root: Path,
    rule_config: ClassicalRuleConfig,
    artifacts: ArtifactStore,
) -> ConditionPreparation:
    """Resolve the exact durable C0 preontology used by the held-out controller."""

    state = cast(
        HeldOutC0PreparationState,
        _load_canonical(
            c0_state_root / f"{boundary.unit.unit_id}.json",
            HeldOutC0PreparationState,
        ),
    )
    try:
        state.validate_pointer()
    except Exception as error:
        raise Phase5SourceProductionError("C0 preparation pointer is invalid") from error
    preparation = cast(
        ConditionPreparation,
        _load_held_out_reference(
            artifacts,
            state.preparation,
            ConditionPreparation,
            expected_release=ReleaseClass.RESTRICTED,
        ),
    )
    preontology = preparation.sealed_preontology
    constructions = tuple(
        item
        for item in closed.execution.c0_constructions
        if item.unit_id == boundary.unit.unit_id
    )
    if len(constructions) != 1:
        raise Phase5SourceProductionError("C0 construction receipt is not unique")
    construction = constructions[0]
    if construction.outcome is not RunOutcome.SUCCEEDED:
        _unavailable(
            episode_id=boundary.episode_id,
            condition=ConditionName.C0_CLASSICAL_PRE,
            outcome=construction.outcome,
            source_record_hash=construction.content_hash,
        )
    binding = _prequery_binding(
        closed,
        unit_id=boundary.unit.unit_id,
        condition=ConditionName.C0_CLASSICAL_PRE,
        seed_block=None,
    )
    if (
        state.unit_id != boundary.unit.unit_id
        or state.prequery_stage_hash
        != boundary.unit.prequery_stage.staging_manifest_hash
        or state.model_visible_evidence_hash
        != boundary.unit.prequery_stage.evidence_artifact_hash
        or state.rule_config_hash != rule_config.content_hash
        or state.preparation.logical_content_hash != preparation.content_hash
        or preontology is None
        or preparation.condition is not ConditionName.C0_CLASSICAL_PRE
        or preparation.snapshot_hash != boundary.snapshot.content_hash
        or state.construction_seal_hash
        != preontology.construction_seal.content_hash
        or state.complete_graph_hash != preontology.construction_seal.ontology_hash
        or construction.construction_seal_hash != state.construction_seal_hash
        or construction.complete_graph_hash != state.complete_graph_hash
        or construction.prequery_stage_hash
        != boundary.unit.prequery_stage.staging_manifest_hash
        or binding.snapshot_hash != boundary.snapshot.content_hash
        or binding.preparation_hash != construction.content_hash
        or binding.lineage_artifact_hash != construction.complete_graph_hash
        or binding.completed_at != construction.completed_at
    ):
        raise Phase5SourceProductionError("C0 preparation lineage changed")
    return preparation


def _validate_c1_preparation(
    *,
    boundary: _EpisodeBoundary,
    closed: ClosedHeldOutSources,
    preparation: ConditionPreparation,
) -> None:
    preontology = preparation.sealed_preontology
    binding = _prequery_binding(
        closed,
        unit_id=boundary.unit.unit_id,
        condition=ConditionName.C1_LLM_PRE,
        seed_block=1,
    )
    if (
        preontology is None
        or preparation.condition is not ConditionName.C1_LLM_PRE
        or preparation.snapshot_hash != boundary.snapshot.content_hash
        or preontology.seed_block != 1
        or binding.snapshot_hash != boundary.snapshot.content_hash
        or binding.preparation_hash != preparation.content_hash
        or binding.lineage_artifact_hash
        != preontology.construction_seal.content_hash
        or binding.completed_at > preparation.completed_at
    ):
        raise Phase5SourceProductionError("C1 preparation lineage changed")


def _load_commitment_records(
    *,
    commitment_path: Path,
    protocol: FeedbackProtocolConfiguration,
    closed: ClosedHeldOutSources,
    cas: RestrictedPhase5CAS,
) -> tuple[
    Phase5ScriptCommitmentManifest,
    dict[str, tuple[RevisionInstruction, ScriptedRevisionFreeze]],
]:
    commitment = load_phase5_script_commitment(commitment_path)
    if commitment_path.read_bytes() != _canonical_bytes(commitment):
        raise Phase5SourceProductionError("script commitment is not canonical")
    if (
        commitment.protocol_hash != protocol.content_hash
        or commitment.benchmark_draft_seal_hash
        != protocol.benchmark_draft_seal_hash
        or commitment.review_gate.review_completion_manifest_hash
        != closed.execution.review_completion_manifest_hash
        or commitment.review_gate.final_reviewed_seal_hash
        != closed.execution.final_reviewed_seal_hash
        or commitment.parent_readiness_floor.held_out_call_manifest_hash
        != closed.call_manifest.content_hash
        or commitment.scheduled_activation_at <= commitment.committed_at
    ):
        raise Phase5SourceProductionError(
            "script commitment and closed held-out execution differ"
        )
    selections = {item.episode_id: item for item in protocol.scripted_episodes}
    entries = {item.episode_id: item for item in commitment.entries}
    if set(entries) != set(selections):
        raise Phase5SourceProductionError("script commitment episode inventory changed")
    resolved: dict[str, tuple[RevisionInstruction, ScriptedRevisionFreeze]] = {}
    for episode_id, selection in selections.items():
        entry = entries[episode_id]
        instruction = cas.load(
            entry.instruction,
            RevisionInstruction,
            object_kind="revision_instruction",
        )
        freeze = cas.load(
            entry.freeze,
            ScriptedRevisionFreeze,
            object_kind="scripted_revision_freeze",
        )
        if (
            entry.context_id != selection.context_id
            or entry.action is not selection.action
            or instruction.before_context.context_id != selection.context_id
            or instruction.revision.action is not selection.action
            or instruction.content_hash != freeze.instruction_hash
            or entry.instruction.logical_content_hash != instruction.content_hash
            or entry.freeze.logical_content_hash != freeze.content_hash
            or freeze.episode_id != episode_id
            or freeze.context_id != selection.context_id
            or freeze.frozen_at != commitment.committed_at
            or instruction.revision.created_at
            != commitment.scheduled_activation_at
        ):
            raise Phase5SourceProductionError("committed scripted revision changed")
        resolved[episode_id] = (instruction, freeze)
    return commitment, resolved


def _validate_known_answer_source(
    *,
    source: Phase5KnownAnswerSourceManifest,
    protocol: FeedbackProtocolConfiguration,
    commitment: Phase5ScriptCommitmentManifest,
    closed: ClosedHeldOutSources,
) -> tuple[ScorerOnlyFeedbackBinding, ...]:
    selections = {item.episode_id: item for item in protocol.scripted_episodes}
    answers = {item.episode_id: item for item in source.answers}
    if (
        source.protocol_hash != protocol.content_hash
        or source.review_completion_manifest_hash
        != closed.execution.review_completion_manifest_hash
        or source.final_reviewed_seal_hash
        != closed.execution.final_reviewed_seal_hash
        or source.authored_at > commitment.committed_at
        or set(answers) != set(selections)
    ):
        raise Phase5SourceProductionError(
            "known-answer source is not the output-blind reviewed six-script inventory"
        )
    bindings: list[ScorerOnlyFeedbackBinding] = []
    for selection in protocol.scripted_episodes:
        answer = answers[selection.episode_id]
        if (
            answer.context_id != selection.context_id
            or answer.scorer_binding_key != selection.scorer_binding_key
            or answer.required_target_change_source.logical_content_hash
            != answer.required_target_change_hash
        ):
            raise Phase5SourceProductionError(
                "known-answer scorer key or context changed"
            )
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        ):
            bindings.append(
                ScorerOnlyFeedbackBinding(
                    episode_id=answer.episode_id,
                    condition=condition,
                    scorer_binding_key=answer.scorer_binding_key,
                    before_gold_projection_hash=answer.before_gold_projection_hash,
                    after_gold_projection_hash=answer.after_gold_projection_hash,
                    required_target_change_hash=answer.required_target_change_hash,
                )
            )
    return tuple(bindings)


def _load_trace_instructions(
    *,
    trace_paths: Mapping[str, Path],
    trace_receipt_paths: Mapping[str, Path],
    restricted_root: Path,
    protocol: FeedbackProtocolConfiguration,
    selected: dict[str, tuple[Any, Any, Any]],
    repository: Path,
    artifacts: ArtifactStore,
    query_accessed_at: datetime,
) -> tuple[
    dict[
        str,
        tuple[RevisionInstruction, ResearcherTraceSubmissionReceipt, _EpisodeBoundary],
    ],
    tuple[
        Phase5TraceInstructionBinding,
        Phase5TraceInstructionBinding,
        Phase5TraceInstructionBinding,
    ],
]:
    slots = {item.episode_id: item for item in protocol.researcher_trace_slots}
    if set(trace_paths) != set(slots):
        raise Phase5SourceProductionError(
            "exactly the three registered researcher trace instruction files are required"
        )
    if set(trace_receipt_paths) != set(slots):
        raise Phase5SourceProductionError(
            "exactly the three registered researcher trace receipt files are required"
        )
    resolved: dict[
        str,
        tuple[RevisionInstruction, ResearcherTraceSubmissionReceipt, _EpisodeBoundary],
    ] = {}
    bindings: list[Phase5TraceInstructionBinding] = []
    for slot in protocol.researcher_trace_slots:
        path = _restricted_descendant(
            restricted_root,
            trace_paths[slot.episode_id],
            label=f"trace instruction {slot.episode_id}",
        )
        instruction = cast(
            RevisionInstruction,
            _load_canonical(path, RevisionInstruction),
        )
        receipt_path = _restricted_descendant(
            restricted_root,
            trace_receipt_paths[slot.episode_id],
            label=f"trace submission receipt {slot.episode_id}",
        )
        receipt = cast(
            ResearcherTraceSubmissionReceipt,
            _load_canonical(receipt_path, ResearcherTraceSubmissionReceipt),
        )
        try:
            submission = RevisionDraftSubmission.model_validate_json(
                receipt.submission_canonical_json
            )
        except Exception as error:
            raise Phase5SourceProductionError(
                "researcher trace receipt contains an invalid UI submission"
            ) from error
        boundary = _episode_boundary(
            episode_id=slot.episode_id,
            context_id=slot.context_id,
            selected=selected,
            repository=repository,
            artifacts=artifacts,
        )
        if (
            instruction.before_context != boundary.opening.query_context
            or instruction.before_context.context_id != slot.context_id
            or instruction.revision.action not in slot.allowed_actions
            or instruction.revision.created_at >= query_accessed_at
            or instruction.after_context.spoiler_horizon
            != boundary.snapshot.horizon
            or receipt.episode_id != slot.episode_id
            or receipt.action is not instruction.revision.action
            or receipt.requested_at != instruction.revision.created_at
            or submission.to_canonical_json() != receipt.submission_canonical_json
            or submission.content_hash != receipt.submission_hash
            or submission.action is not receipt.action
            or submission.seed != protocol.llm_seed
            or receipt.instruction_hash != instruction.content_hash
            or receipt.instruction_file_sha256 != _file_sha256(path)
            or receipt.recorded_at >= query_accessed_at
        ):
            raise Phase5SourceProductionError(
                "researcher trace instruction changed its registered source boundary"
            )
        try:
            assert_revision_anchors_resolve_in_packet(instruction, boundary.packet)
        except ValueError as error:
            raise Phase5SourceProductionError(
                "researcher trace anchor is outside its held-out packet"
            ) from error
        resolved[slot.episode_id] = (instruction, receipt, boundary)
        bindings.append(
            Phase5TraceInstructionBinding(
                episode_id=slot.episode_id,
                submission_hash=receipt.submission_hash,
                submission_file_sha256=receipt.submission_file_sha256,
                instruction_hash=instruction.content_hash,
                instruction_file_sha256=_file_sha256(path),
                submission_receipt_hash=receipt.content_hash,
                submission_receipt_file_sha256=_file_sha256(receipt_path),
            )
        )
    return resolved, cast(
        tuple[
            Phase5TraceInstructionBinding,
            Phase5TraceInstructionBinding,
            Phase5TraceInstructionBinding,
        ],
        tuple(bindings),
    )


def _cpu_resolver_hash(
    *,
    condition: ConditionName,
    instruction: RevisionInstruction,
    boundary: _EpisodeBoundary,
    preparation: ConditionPreparation,
    rule_config: ClassicalRuleConfig | None,
) -> str:
    return canonical_sha256(
        {
            "resolver": "phase5-sealed-reprojection-v1",
            "condition": condition.value,
            "action": instruction.revision.action.value,
            "instruction_hash": instruction.content_hash,
            "source_query_stage_hash": boundary.stage.staging_manifest_hash,
            "source_preparation_hash": preparation.content_hash,
            "source_seal_hash": cast(
                Any, preparation.sealed_preontology
            ).construction_seal.content_hash,
            "rule_config_hash": None if rule_config is None else rule_config.content_hash,
            "policy": (
                "selection_only"
                if instruction.revision.action is FeedbackAction.REFINE_CONTEXT
                else "capability_limited"
            ),
        }
    )


def _build_cpu_draft(
    *,
    instruction: RevisionInstruction,
    boundary: _EpisodeBoundary,
    condition: ConditionName,
    before_projection: OntologyProjection,
    preparation: ConditionPreparation,
    prequery_barrier: Any,
    query_accessed_at: datetime,
    processing_started_at: datetime,
    rule_config: ClassicalRuleConfig | None = None,
) -> _CpuDraft:
    """Perform only fixed same-seal selection, never post-query construction."""

    preontology = preparation.sealed_preontology
    if (
        preontology is None
        or preparation.condition is not condition
        or before_projection.condition is not condition
        or before_projection.construction_seal != preontology.construction_seal
        or before_projection.snapshot_hash != boundary.snapshot.content_hash
        or before_projection.packet_hash != boundary.packet.content_hash
        or instruction.before_context != boundary.opening.query_context
        or instruction.revision.created_at >= query_accessed_at
        or query_accessed_at >= processing_started_at
    ):
        raise Phase5SourceProductionError("CPU feedback parent or chronology changed")
    if instruction.after_context.spoiler_horizon != boundary.snapshot.horizon:
        raise Phase5SourceProductionError(
            "Phase 5 same-packet feedback cannot change the spoiler horizon"
        )
    try:
        assert_revision_anchors_resolve_in_packet(instruction, boundary.packet)
    except ValueError as error:
        raise Phase5SourceProductionError(
            "scripted revision anchor is outside its held-out packet"
        ) from error
    resolver_hash = _cpu_resolver_hash(
        condition=condition,
        instruction=instruction,
        boundary=boundary,
        preparation=preparation,
        rule_config=rule_config,
    )
    after_projection: OntologyProjection | None = None
    if instruction.revision.action is FeedbackAction.REFINE_CONTEXT:
        stage_hash = canonical_sha256(
            {
                "kind": "phase5-revised-query-stage-v1",
                "source_query_stage_hash": boundary.stage.staging_manifest_hash,
                "instruction_hash": instruction.content_hash,
            }
        )
        query_artifact_hash = canonical_sha256(
            {
                "kind": "phase5-revised-model-visible-query-v1",
                "query": to_model_visible_query(instruction.after_context),
                "instruction_hash": instruction.content_hash,
            }
        )
        query_access = QueryAccessEvent(
            access_event_id=(
                f"phase5-cpu-query-{canonical_sha256((condition, instruction.content_hash))[:20]}"
            ),
            execution_id=prequery_barrier.execution_id,
            query_context_hash=instruction.after_context.content_hash,
            model_visible_query_hash=to_model_visible_query(
                instruction.after_context
            ).content_hash,
            snapshot_hash=boundary.snapshot.content_hash,
            stage_manifest_hash=stage_hash,
            query_artifact_hash=query_artifact_hash,
            prequery_barrier_hash=prequery_barrier.content_hash,
            packet_hash=boundary.packet.content_hash,
            registered_revealed_at=instruction.after_context.revealed_at,
            accessed_at=query_accessed_at,
        )
        inputs = SimpleNamespace(
            preparation=preparation,
            snapshot=boundary.snapshot,
            packet=boundary.packet,
            context=instruction.after_context,
            query_access=query_access,
            prequery_barrier=prequery_barrier,
            query_processing_started_at=processing_started_at,
            packet_materialization=None,
            upper_ontology=preontology.upper_ontology,
            revisions=(),
            run_config=SimpleNamespace(
                content_hash=resolver_hash,
                seed_block=(1 if condition is ConditionName.C1_LLM_PRE else None),
            ),
        )
        try:
            after_projection = (
                project_sealed_c0(preontology, inputs, rule_config=rule_config)
                if condition is ConditionName.C0_CLASSICAL_PRE
                else project_sealed_c1(preontology, inputs)
            )
        except Exception as error:
            raise Phase5SourceProductionError(
                "fixed same-seal CPU reprojection failed"
            ) from error
        if (
            after_projection.construction_seal != before_projection.construction_seal
            or after_projection.context_hash != instruction.after_context.content_hash
            or after_projection.packet_hash != boundary.packet.content_hash
            or after_projection.snapshot_hash != boundary.snapshot.content_hash
            or not after_projection.decisions
            or any(
                item.operator is not ConstructionOperator.SELECTION
                for item in after_projection.decisions
            )
        ):
            raise Phase5SourceProductionError(
                "CPU feedback attempted construction or changed its evidence/seal"
            )
    elif instruction.revision.action is not FeedbackAction.REQUEST_MERGE_SPLIT:
        raise Phase5SourceProductionError("unsupported Phase 5 revision action")
    return _CpuDraft(
        condition=condition,
        before_projection=before_projection,
        after_projection=after_projection,
        instruction_hash=instruction.content_hash,
        source_query_stage_hash=boundary.stage.staging_manifest_hash,
        source_preparation_hash=preparation.content_hash,
        resolver_hash=resolver_hash,
    )


def _finalize_cpu_input(
    draft: _CpuDraft,
    *,
    started_at: datetime,
    completed_at: datetime,
) -> CpuReprojectionInput:
    return CpuReprojectionInput(
        condition=draft.condition,
        before_projection=draft.before_projection,
        after_projection=draft.after_projection,
        instruction_hash=draft.instruction_hash,
        source_query_stage_hash=draft.source_query_stage_hash,
        source_preparation_hash=draft.source_preparation_hash,
        resolver_hash=draft.resolver_hash,
        started_at=started_at,
        completed_at=completed_at,
        resolved_at=completed_at,
        checked_at=completed_at,
    )


def _strictly_after(
    clock: Callable[[], datetime],
    predecessor: datetime,
    *,
    label: str,
) -> datetime:
    observed = _aware(clock(), label=label)
    return max(observed, predecessor + timedelta(microseconds=1))


def _safe_component(value: str) -> str:
    if Path(value).name != value or value in {"", ".", ".."} or "\\" in value:
        raise Phase5SourceProductionError("unsafe Phase 5 episode identifier")
    return value


def _existing_session(output_root: Path) -> Phase5SourceProductionSession | None:
    path = output_root / SESSION_FILE_NAME
    if not path.exists():
        return None
    return cast(
        Phase5SourceProductionSession,
        _load_canonical(path, Phase5SourceProductionSession),
    )


def _expected_output_paths(protocol: FeedbackProtocolConfiguration) -> tuple[Path, ...]:
    values: list[Path] = [
        Path(SESSION_FILE_NAME),
        Path(CALL_MANIFEST_FILE_NAME),
        Path(SCORER_AUTHORIZATION_FILE_NAME),
    ]
    for item in protocol.scripted_episodes:
        episode_id = _safe_component(item.episode_id)
        values.extend(
            (
                Path("exports") / f"{episode_id}--c0.json",
                Path("exports") / f"{episode_id}--c1.json",
                Path("exports") / f"{episode_id}--c2.json",
                Path("cpu") / f"{episode_id}--c0.json",
                Path("cpu") / f"{episode_id}--c1.json",
            )
        )
    for item in protocol.researcher_trace_slots:
        episode_id = _safe_component(item.episode_id)
        values.append(Path("exports") / f"{episode_id}--c2.json")
    values.append(Path(SOURCE_FILE_NAME))
    return tuple(values)


def _deduplicate_cas_writes(values: Sequence[_CASWrite]) -> tuple[_CASWrite, ...]:
    by_artifact_hash: dict[str, tuple[Phase5CASReference, _CASWrite]] = {}
    for value in values:
        reference = phase5_record_reference(
            value.value,
            object_kind=value.object_kind,
            release_class=value.release_class,
        )
        previous = by_artifact_hash.get(reference.artifact_hash)
        if previous is not None and previous[0] != reference:
            raise Phase5SourceProductionError(
                "one Phase 5 CAS byte string was assigned conflicting metadata"
            )
        by_artifact_hash[reference.artifact_hash] = (reference, value)
    return tuple(
        item[1]
        for item in sorted(by_artifact_hash.values(), key=lambda item: item[0].artifact_hash)
    )


def prepare_phase5_source_production(
    *,
    repository: Path,
    restricted_root: Path,
    output_root: Path,
    protocol: FeedbackProtocolConfiguration,
    script_commitment_path: Path,
    primary_results_gate_path: Path,
    held_out_journal_root: Path,
    known_answer_source_path: Path,
    trace_instruction_paths: Mapping[str, Path],
    trace_submission_receipt_paths: Mapping[str, Path],
    c0_state_root: Path,
    c0_rule_config_path: Path,
    artifacts: ArtifactStore,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> PreparedPhase5SourceProduction:
    """Build the exact source graph in memory, performing no persistence.

    All 21 selected parent projections must exist and validate.  A registered
    failure remains in the held-out ITT journal, but cannot be converted into a
    feedback before-projection; this function therefore fails closed before any
    Phase 5 output write.
    """

    repository = repository.resolve(strict=True)
    restricted_root = restricted_root.resolve(strict=True)
    canonical_restricted_root = (
        repository / "artifacts/restricted"
    ).resolve(strict=True)
    if restricted_root != canonical_restricted_root:
        raise Phase5SourceProductionError(
            "restricted root is not the canonical repository artifacts/restricted root"
        )
    output_root = _restricted_descendant(
        restricted_root,
        output_root,
        label="Phase 5 source output root",
    )
    script_commitment_path = _restricted_descendant(
        restricted_root,
        script_commitment_path,
        label="Phase 5 script commitment",
    )
    primary_results_gate_path = _restricted_descendant(
        restricted_root,
        primary_results_gate_path,
        label="primary held-out results gate",
    )
    held_out_journal_root = _restricted_descendant(
        restricted_root,
        held_out_journal_root,
        label="held-out journal",
    )
    known_answer_source_path = _restricted_descendant(
        restricted_root,
        known_answer_source_path,
        label="known-answer source",
    )
    c0_state_root = _restricted_descendant(
        restricted_root,
        c0_state_root,
        label="C0 preparation state root",
    )
    rule_path = c0_rule_config_path
    if not rule_path.is_absolute():
        rule_path = repository / rule_path
    try:
        rule_config = load_classical_rule_config(rule_path)
    except Exception as error:
        raise Phase5SourceProductionError("frozen C0 rule configuration is invalid") from error

    closed = load_closed_held_out_sources(
        primary_results_gate_path=primary_results_gate_path,
        held_out_journal_root=held_out_journal_root,
    )
    if (
        closed.gate.benchmark_draft_seal_hash
        != protocol.benchmark_draft_seal_hash
        or closed.gate.result_artifact_hashes
        != closed.bridge.output_artifact_hashes
    ):
        raise Phase5SourceProductionError("held-out result gate inventory changed")
    cas = RestrictedPhase5CAS(artifacts)
    commitment, scripts = _load_commitment_records(
        commitment_path=script_commitment_path,
        protocol=protocol,
        closed=closed,
        cas=cas,
    )
    known_answer = load_phase5_known_answer_source(known_answer_source_path)
    scorer_bindings = _validate_known_answer_source(
        source=known_answer,
        protocol=protocol,
        commitment=commitment,
        closed=closed,
    )
    selected = _selected_openings(protocol, closed)
    _audit_source_output(output_root, _expected_output_paths(protocol))
    retained_session = _existing_session(output_root)
    if retained_session is None:
        query_accessed_at = _aware(clock(), label="source query-access clock")
        minimum_access = max(
            commitment.scheduled_activation_at,
            closed.gate.frozen_at,
        )
        if query_accessed_at <= minimum_access:
            raise Phase5SourceProductionError(
                "Phase 5 source production must occur after activation and held-out freeze"
            )
        processing_started_at = _strictly_after(
            clock,
            query_accessed_at,
            label="CPU reprojection start clock",
        )
        completed_at: datetime | None = None
    else:
        query_accessed_at = retained_session.query_accessed_at
        processing_started_at = retained_session.started_at
        completed_at = retained_session.completed_at
        if query_accessed_at <= max(
            commitment.scheduled_activation_at,
            closed.gate.frozen_at,
        ):
            raise Phase5SourceProductionError("retained Phase 5 session predates its gates")

    traces, trace_bindings = _load_trace_instructions(
        trace_paths=trace_instruction_paths,
        trace_receipt_paths=trace_submission_receipt_paths,
        restricted_root=restricted_root,
        protocol=protocol,
        selected=selected,
        repository=repository,
        artifacts=artifacts,
        query_accessed_at=query_accessed_at,
    )
    exports: dict[tuple[str, ConditionName], _ResolvedExport] = {}
    boundaries: dict[str, _EpisodeBoundary] = {}
    cpu_drafts: dict[tuple[str, ConditionName], _CpuDraft] = {}

    for selection in protocol.scripted_episodes:
        episode_id = selection.episode_id
        instruction, _freeze = scripts[episode_id]
        boundary = _episode_boundary(
            episode_id=episode_id,
            context_id=selection.context_id,
            selected=selected,
            repository=repository,
            artifacts=artifacts,
        )
        boundaries[episode_id] = boundary
        try:
            assert_revision_anchors_resolve_in_packet(instruction, boundary.packet)
        except ValueError as error:
            raise Phase5SourceProductionError(
                "committed scripted anchor is outside its held-out packet"
            ) from error
        if (
            instruction.before_context != boundary.opening.query_context
            or instruction.after_context.spoiler_horizon != boundary.snapshot.horizon
        ):
            raise Phase5SourceProductionError(
                "committed script changed query semantics or requires a different packet"
            )
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        ):
            exports[(episode_id, condition)] = _build_export(
                boundary=boundary,
                condition=condition,
                closed=closed,
                journal_root=held_out_journal_root,
                artifacts=artifacts,
                exported_at=query_accessed_at,
                expected_frozen_seed=protocol.llm_seed,
            )
        resolved = [
            exports[(episode_id, condition)]
            for condition in (
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C1_LLM_PRE,
                ConditionName.C2_LLM_QUERY,
            )
        ]
        if max(item.export.source_completed_at for item in resolved) >= (
            instruction.revision.created_at
        ):
            raise Phase5SourceProductionError(
                "script activation did not strictly follow every selected parent"
            )
        c0_preparation = _load_c0_preparation(
            boundary=boundary,
            closed=closed,
            c0_state_root=c0_state_root,
            rule_config=rule_config,
            artifacts=artifacts,
        )
        c1_preparation = resolved[1].preparation
        if c1_preparation is None:
            raise Phase5SourceProductionError("selected C1 parent lost its preparation")
        _validate_c1_preparation(
            boundary=boundary,
            closed=closed,
            preparation=c1_preparation,
        )
        cpu_drafts[(episode_id, ConditionName.C0_CLASSICAL_PRE)] = _build_cpu_draft(
            instruction=instruction,
            boundary=boundary,
            condition=ConditionName.C0_CLASSICAL_PRE,
            before_projection=resolved[0].projection,
            preparation=c0_preparation,
            prequery_barrier=closed.execution.prequery_barrier,
            query_accessed_at=query_accessed_at,
            processing_started_at=processing_started_at,
            rule_config=rule_config,
        )
        cpu_drafts[(episode_id, ConditionName.C1_LLM_PRE)] = _build_cpu_draft(
            instruction=instruction,
            boundary=boundary,
            condition=ConditionName.C1_LLM_PRE,
            before_projection=resolved[1].projection,
            preparation=c1_preparation,
            prequery_barrier=closed.execution.prequery_barrier,
            query_accessed_at=query_accessed_at,
            processing_started_at=processing_started_at,
        )

    for slot in protocol.researcher_trace_slots:
        instruction, submission_receipt, boundary = traces[slot.episode_id]
        boundaries[slot.episode_id] = boundary
        resolved = _build_export(
            boundary=boundary,
            condition=ConditionName.C2_LLM_QUERY,
            closed=closed,
            journal_root=held_out_journal_root,
            artifacts=artifacts,
            exported_at=query_accessed_at,
            expected_frozen_seed=protocol.llm_seed,
        )
        certificate = resolved.projection.construction_certificate
        before_bundle = build_visualization_bundle(
            resolved.projection,
            instruction.before_context,
            boundary.packet,
            content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
        )
        submission = RevisionDraftSubmission.model_validate_json(
            submission_receipt.submission_canonical_json
        )
        replayed_instruction = _compile_submission(
            submission,
            before_bundle,
            created_at=submission_receipt.requested_at,
        )
        if (
            certificate is None
            or resolved.export.source_completed_at >= instruction.revision.created_at
            or certificate.completed_at >= instruction.revision.created_at
            or submission_receipt.before_projection_id
            != resolved.projection.projection_id
            or submission_receipt.before_projection_hash
            != resolved.projection.content_hash
            or submission_receipt.before_bundle_hash != before_bundle.content_hash
            or replayed_instruction != instruction
        ):
            raise Phase5SourceProductionError(
                "researcher trace was not recorded after its selected C2 parent"
            )
        exports[(slot.episode_id, ConditionName.C2_LLM_QUERY)] = resolved

    if completed_at is None:
        completed_at = _strictly_after(
            clock,
            processing_started_at,
            label="source completion clock",
        )
    cpu_inputs = {
        key: _finalize_cpu_input(
            value,
            started_at=processing_started_at,
            completed_at=completed_at,
        )
        for key, value in cpu_drafts.items()
    }
    scorer = Phase5ScorerBindingAuthorization(
        authorization_id=f"phase5-scorer-{known_answer.content_hash[:20]}",
        protocol_hash=protocol.content_hash,
        scorer_bridge_hash=closed.bridge.content_hash,
        known_answer_source_hash=known_answer.content_hash,
        review_completion_manifest_hash=closed.execution.review_completion_manifest_hash,
        final_reviewed_seal_hash=closed.execution.final_reviewed_seal_hash,
        bindings=scorer_bindings,
        authorized_at=completed_at,
    )

    source_file_hash = _file_sha256(Path(__file__))
    session = Phase5SourceProductionSession(
        session_id=(
            "phase5-source-session-"
            + canonical_sha256(
                {
                    "protocol": protocol.content_hash,
                    "commitment": commitment.content_hash,
                    "gate": closed.gate.content_hash,
                    "known_answer": known_answer.content_hash,
                    "traces": trace_bindings,
                }
            )[:20]
        ),
        protocol_hash=protocol.content_hash,
        script_commitment_manifest_hash=commitment.content_hash,
        primary_results_gate_hash=closed.gate.content_hash,
        held_out_execution_manifest_hash=closed.execution.content_hash,
        held_out_call_manifest_hash=closed.call_manifest.content_hash,
        known_answer_source_hash=known_answer.content_hash,
        known_answer_source_file_sha256=_file_sha256(known_answer_source_path),
        trace_instructions=trace_bindings,
        producer_source_file_sha256=source_file_hash,
        query_accessed_at=query_accessed_at,
        started_at=processing_started_at,
        completed_at=completed_at,
    )
    if retained_session is not None and retained_session != session:
        raise Phase5SourceProductionError(
            "retained Phase 5 source session differs from replayed inputs or code"
        )

    cas_writes: list[_CASWrite] = [
        _CASWrite(closed.call_manifest, "held_out_call_manifest", ReleaseClass.RESTRICTED),
        _CASWrite(scorer, "scorer_binding_authorization", ReleaseClass.RESTRICTED),
    ]
    published: list[_PublishedRecord] = [
        _PublishedRecord(Path(CALL_MANIFEST_FILE_NAME), closed.call_manifest),
        _PublishedRecord(Path(SCORER_AUTHORIZATION_FILE_NAME), scorer),
    ]
    call_manifest_ref = phase5_record_reference(
        closed.call_manifest,
        object_kind="held_out_call_manifest",
    )
    scorer_ref = phase5_record_reference(
        scorer,
        object_kind="scorer_binding_authorization",
    )
    episodes: list[Phase5EpisodeSource] = []

    for selection in protocol.scripted_episodes:
        episode_id = _safe_component(selection.episode_id)
        instruction, _freeze = scripts[episode_id]
        entry = next(item for item in commitment.entries if item.episode_id == episode_id)
        c0 = exports[(episode_id, ConditionName.C0_CLASSICAL_PRE)]
        c1 = exports[(episode_id, ConditionName.C1_LLM_PRE)]
        c2 = exports[(episode_id, ConditionName.C2_LLM_QUERY)]
        c0_cpu = cpu_inputs[(episode_id, ConditionName.C0_CLASSICAL_PRE)]
        c1_cpu = cpu_inputs[(episode_id, ConditionName.C1_LLM_PRE)]
        export_refs = {
            condition: phase5_record_reference(
                resolved.export,
                object_kind="held_out_projection_export",
            )
            for condition, resolved in (
                (ConditionName.C0_CLASSICAL_PRE, c0),
                (ConditionName.C1_LLM_PRE, c1),
                (ConditionName.C2_LLM_QUERY, c2),
            )
        }
        c0_cpu_ref = phase5_record_reference(
            c0_cpu,
            object_kind="cpu_reprojection_input",
        )
        c1_cpu_ref = phase5_record_reference(
            c1_cpu,
            object_kind="cpu_reprojection_input",
        )
        episodes.append(
            Phase5EpisodeSource(
                episode_id=episode_id,
                context_id=selection.context_id,
                kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
                unit_id=boundaries[episode_id].unit.unit_id,
                query_stage_hash=boundaries[episode_id].stage.staging_manifest_hash,
                instruction=entry.instruction,
                scripted_freeze=entry.freeze,
                c0_export=export_refs[ConditionName.C0_CLASSICAL_PRE],
                c1_export=export_refs[ConditionName.C1_LLM_PRE],
                c0_cpu_input=c0_cpu_ref,
                c1_cpu_input=c1_cpu_ref,
                c2_export=export_refs[ConditionName.C2_LLM_QUERY],
            )
        )
        cas_writes.extend(
            (
                _CASWrite(c0.export, "held_out_projection_export", ReleaseClass.RESTRICTED),
                _CASWrite(c1.export, "held_out_projection_export", ReleaseClass.RESTRICTED),
                _CASWrite(c2.export, "held_out_projection_export", ReleaseClass.RESTRICTED),
                _CASWrite(c0_cpu, "cpu_reprojection_input", ReleaseClass.RESTRICTED),
                _CASWrite(c1_cpu, "cpu_reprojection_input", ReleaseClass.RESTRICTED),
            )
        )
        for resolved in (c0, c1, c2):
            cas_writes.extend(resolved.cas_writes)
        published.extend(
            (
                _PublishedRecord(
                    Path("exports") / f"{episode_id}--c0.json",
                    c0.export,
                ),
                _PublishedRecord(
                    Path("exports") / f"{episode_id}--c1.json",
                    c1.export,
                ),
                _PublishedRecord(
                    Path("exports") / f"{episode_id}--c2.json",
                    c2.export,
                ),
                _PublishedRecord(
                    Path("cpu") / f"{episode_id}--c0.json",
                    c0_cpu,
                ),
                _PublishedRecord(
                    Path("cpu") / f"{episode_id}--c1.json",
                    c1_cpu,
                ),
            )
        )

    for slot in protocol.researcher_trace_slots:
        episode_id = _safe_component(slot.episode_id)
        instruction, submission_receipt, _boundary = traces[episode_id]
        c2 = exports[(episode_id, ConditionName.C2_LLM_QUERY)]
        instruction_ref = phase5_record_reference(
            instruction,
            object_kind="revision_instruction",
        )
        c2_ref = phase5_record_reference(
            c2.export,
            object_kind="held_out_projection_export",
        )
        submission_receipt_ref = phase5_record_reference(
            submission_receipt,
            object_kind="researcher_trace_submission_receipt",
        )
        episodes.append(
            Phase5EpisodeSource(
                episode_id=episode_id,
                context_id=slot.context_id,
                kind=FeedbackEpisodeKind.RESEARCHER_TRACE,
                unit_id=boundaries[episode_id].unit.unit_id,
                query_stage_hash=boundaries[episode_id].stage.staging_manifest_hash,
                instruction=instruction_ref,
                trace_submission_receipt=submission_receipt_ref,
                c2_export=c2_ref,
            )
        )
        cas_writes.extend(
            (
                _CASWrite(instruction, "revision_instruction", ReleaseClass.RESTRICTED),
                _CASWrite(
                    submission_receipt,
                    "researcher_trace_submission_receipt",
                    ReleaseClass.RESTRICTED,
                ),
                _CASWrite(c2.export, "held_out_projection_export", ReleaseClass.RESTRICTED),
                *c2.cas_writes,
            )
        )
        published.append(
            _PublishedRecord(
                Path("exports") / f"{episode_id}--c2.json",
                c2.export,
            )
        )

    source = Phase5MaterializationSourceManifest(
        source_id=f"phase5-source-{session.content_hash[:20]}",
        run_id=f"phase5-feedback-{closed.execution.content_hash[:20]}",
        protocol_hash=protocol.content_hash,
        script_commitment_manifest_hash=commitment.content_hash,
        primary_results_gate_hash=closed.gate.content_hash,
        final_reviewed_seal_hash=closed.execution.final_reviewed_seal_hash,
        held_out_execution_manifest_hash=closed.execution.content_hash,
        held_out_call_manifest=call_manifest_ref,
        scorer_binding_authorization=scorer_ref,
        episodes=tuple(episodes),
        captured_at=completed_at,
    )
    published.append(_PublishedRecord(Path(SOURCE_FILE_NAME), source))
    for item in published:
        existing = output_root / item.relative_path
        if existing.exists() and (
            not existing.is_file() or existing.read_bytes() != _canonical_bytes(item.value)
        ):
            raise Phase5SourceProductionError(
                f"retained Phase 5 source record changed: {item.relative_path.as_posix()}"
            )
    return PreparedPhase5SourceProduction(
        session=session,
        source=source,
        cas_writes=_deduplicate_cas_writes(cas_writes),
        published_records=tuple(published),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_exact(path: Path, value: ImmutableRecord) -> bool:
    payload = _canonical_bytes(value)
    _assert_no_symlink_chain(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise Phase5SourceProductionError(
                f"append-only Phase 5 source drift: {path.name}"
            )
        return False
    parent_existed = path.parent.exists()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    _assert_no_symlink_chain(path.parent)
    if not parent_existed:
        _fsync_directory(path.parent.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise Phase5SourceProductionError(
                    f"concurrent Phase 5 source drift: {path.name}"
                ) from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _audit_source_output(
    output_root: Path,
    ordered_paths: Sequence[Path],
) -> None:
    if not output_root.exists():
        return
    _assert_no_symlink_chain(output_root)
    if not output_root.is_dir():
        raise Phase5SourceProductionError("Phase 5 source output root is not a directory")
    allowed_directories = {
        path.parent.as_posix()
        for path in ordered_paths
        if path.parent != Path(".")
    }
    observed: set[Path] = set()
    for item in output_root.rglob("*"):
        if item.is_symlink():
            raise Phase5SourceProductionError("Phase 5 source output contains a symlink")
        relative = item.relative_to(output_root)
        if item.is_dir():
            if relative.as_posix() not in allowed_directories:
                raise Phase5SourceProductionError(
                    f"unexpected Phase 5 source directory: {relative.as_posix()}"
                )
            continue
        interrupted = any(
            relative.parent == expected.parent
            and item.name.startswith(f".{expected.name}.")
            and item.name.endswith(".tmp")
            for expected in ordered_paths
        )
        if interrupted:
            continue
        observed.add(relative)
    expected = list(ordered_paths)
    prefixes = {frozenset(expected[:index]) for index in range(len(expected) + 1)}
    if frozenset(observed) not in prefixes:
        unexpected = sorted(item.as_posix() for item in observed - set(expected))
        detail = unexpected[0] if unexpected else "out-of-order partial publication"
        raise Phase5SourceProductionError(
            f"unexpected Phase 5 source artifact: {detail}"
        )


def materialize_phase5_source_production(
    *,
    prepared: PreparedPhase5SourceProduction,
    artifacts: ArtifactStore,
    output_root: Path,
    restricted_root: Path,
) -> tuple[Phase5MaterializationSourceManifest, bool]:
    """Persist a prepared source graph in deterministic append-only order."""

    restricted_root = restricted_root.resolve(strict=True)
    output_root = _restricted_descendant(
        restricted_root,
        output_root,
        label="Phase 5 source output root",
    )
    ordered = (
        Path(SESSION_FILE_NAME),
        *(item.relative_path for item in prepared.published_records),
    )
    _audit_source_output(output_root, ordered)
    source_path = output_root / SOURCE_FILE_NAME
    created = not source_path.exists()
    _append_exact(output_root / SESSION_FILE_NAME, prepared.session)
    for item in prepared.cas_writes:
        expected = phase5_record_reference(
            item.value,
            object_kind=item.object_kind,
            release_class=item.release_class,
        )
        observed = persist_phase5_record(
            artifacts,
            item.value,
            object_kind=item.object_kind,
            created_at=prepared.session.completed_at,
            release_class=item.release_class,
        )
        if observed != expected:
            raise Phase5SourceProductionError("Phase 5 CAS write changed after preparation")
    for item in prepared.published_records:
        _append_exact(output_root / item.relative_path, item.value)
    _audit_source_output(output_root, ordered)
    persisted = cast(
        Phase5MaterializationSourceManifest,
        _load_canonical(source_path, Phase5MaterializationSourceManifest),
    )
    if persisted != prepared.source:
        raise Phase5SourceProductionError("persisted Phase 5 source manifest changed")
    return persisted, created


__all__ = [
    "CALL_MANIFEST_FILE_NAME",
    "DEFAULT_PHASE5_SOURCE_ROOT",
    "SCORER_AUTHORIZATION_FILE_NAME",
    "SESSION_FILE_NAME",
    "SOURCE_FILE_NAME",
    "Phase5FeedbackChangeTarget",
    "Phase5GoldProjectionSource",
    "Phase5KnownAnswerEntry",
    "Phase5KnownAnswerSourceManifest",
    "Phase5RequiredTargetChange",
    "Phase5SelectedParentUnavailable",
    "Phase5SourceProductionError",
    "Phase5SourceProductionSession",
    "Phase5TraceInstructionBinding",
    "PreparedPhase5SourceProduction",
    "load_closed_held_out_sources",
    "load_phase5_known_answer_source",
    "materialize_phase5_source_production",
    "prepare_phase5_source_production",
]
