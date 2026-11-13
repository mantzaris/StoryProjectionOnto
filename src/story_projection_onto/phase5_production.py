"""Fail-closed production materialization and metered Phase 5 C2 adaptation.

This module deliberately owns neither held-out execution nor a model-service
lifecycle.  It consumes hash-bound exports from the closed, gold-free held-out
runtime and wraps an already-owned C2 service whose only capabilities are execute
and side-effect-free recovery.  Scorer data crosses the boundary only as opaque
known-answer hashes.
"""

from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import scan_model_payload
from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    EvidenceSnapshot,
    ImmutableRecord,
    OntologyProjection,
    ReleaseClass,
    Sha256Digest,
)
from story_projection_onto.development_runtime import RunOutcome
from story_projection_onto.feedback_provenance import ResearcherTraceSubmissionReceipt
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
    FeedbackLedgerCallReference,
    FeedbackProtocolConfiguration,
    FeedbackRegenerationReceipt,
    ScriptedRevisionFreeze,
)
from story_projection_onto.held_out_controller import (
    HeldOutExecutionManifest,
    HeldOutITTRecord,
)
from story_projection_onto.held_out_primary import (
    HeldOutCallManifest,
    HeldOutCallSpec,
)
from story_projection_onto.phase5_execution import (
    C2RegenerationRequest,
    C2RegenerationResult,
    CpuReprojectionInput,
    Phase5ExecutionError,
    Phase5ExecutionInputManifest,
    ResearcherTraceInput,
    ScorerOnlyFeedbackBinding,
    ScriptedFeedbackInput,
    feedback_gpu_event_record_hash,
    feedback_model_call_record_hash,
    replay_phase5_prerequisites,
    validate_phase5_inputs,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    GpuEvent,
    GpuEventKind,
    ModelBackend,
    ModelCallRole,
    ReadOnlyArtifactStore,
    RetryClass,
)
from story_projection_onto.ui import (
    FeedbackEpisodeKind,
    RevisionDraftSubmission,
    RevisionInstruction,
    VisualizationContentScope,
    _compile_submission,
    assert_revision_anchors_resolve_in_packet,
    build_visualization_bundle,
)

_RecordT = TypeVar("_RecordT", bound=ImmutableRecord)


class Phase5MaterializationError(Phase5ExecutionError):
    """A production Phase 5 input could not be reproduced from frozen artifacts."""


class Phase5AdapterError(Phase5ExecutionError):
    """An already-owned service or its cumulative ledger violated the adapter contract."""


Phase5ObjectKind = Literal[
    "held_out_call_manifest",
    "held_out_projection_export",
    "evidence_packet",
    "evidence_snapshot",
    "ontology_projection",
    "revision_instruction",
    "scripted_revision_freeze",
    "cpu_reprojection_input",
    "scorer_binding_authorization",
    "researcher_trace_submission_receipt",
    "feedback_gold_projection_source",
    "feedback_required_target_change",
    "phase5_after_packet",
    "phase5_after_projection",
    "phase5_failure",
]


class Phase5CASReference(ImmutableRecord):
    """A logical object and its exact CAS serialization.

    Synthetic evidence packets and snapshots already exist in the shared public
    CAS.  Reusing those exact bytes is safe; every constructed, instructional,
    scorer-bound, result, and failure object remains restricted.
    """

    artifact_hash: Sha256Digest
    logical_content_hash: Sha256Digest
    object_kind: Phase5ObjectKind
    media_type: str = Field(min_length=1)
    release_class: ReleaseClass = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def public_only_for_synthetic_evidence(self) -> Self:
        if self.release_class is ReleaseClass.PUBLIC and self.object_kind not in {
            "evidence_packet",
            "evidence_snapshot",
        }:
            raise ValueError("only Phase 5 evidence packets and snapshots may be public")
        return self


class RestrictedPhase5CAS:
    """Typed reader over restricted Phase 5 records and shared public evidence."""

    def __init__(
        self,
        artifacts: ArtifactStore | ReadOnlyArtifactStore,
    ) -> None:
        self.artifacts = artifacts

    def bytes(self, reference: Phase5CASReference) -> bytes:
        try:
            record = self.artifacts.ledger.get_artifact(reference.artifact_hash)
        except KeyError as error:
            raise Phase5MaterializationError(
                f"missing Phase 5 CAS artifact {reference.artifact_hash}"
            ) from error
        if (
            record.content_hash != reference.artifact_hash
            or record.media_type != reference.media_type
            or record.release_class.value != reference.release_class.value
        ):
            raise Phase5MaterializationError("Phase 5 CAS metadata differs from its reference")
        try:
            return self.artifacts.blobs.read_bytes(
                record,
                allow_restricted=reference.release_class is ReleaseClass.RESTRICTED,
            )
        except Exception as error:
            raise Phase5MaterializationError(
                f"cannot verify Phase 5 CAS artifact {reference.artifact_hash}: {error}"
            ) from error

    def load(
        self,
        reference: Phase5CASReference,
        model_type: type[_RecordT],
        *,
        object_kind: Phase5ObjectKind,
    ) -> _RecordT:
        if reference.object_kind != object_kind:
            raise Phase5MaterializationError(
                f"expected {object_kind}, received {reference.object_kind}"
            )
        try:
            value = model_type.model_validate_json(self.bytes(reference))
        except Exception as error:
            raise Phase5MaterializationError(
                f"invalid {object_kind} CAS object: {error}"
            ) from error
        if value.content_hash != reference.logical_content_hash:
            raise Phase5MaterializationError(f"{object_kind} logical hash changed")
        return value


_MEDIA_TYPES: dict[Phase5ObjectKind, str] = {
    kind: f"application/vnd.story-projection.{kind.replace('_', '-')}+json"
    for kind in (
        "held_out_call_manifest",
        "held_out_projection_export",
        "evidence_packet",
        "evidence_snapshot",
        "ontology_projection",
        "revision_instruction",
        "scripted_revision_freeze",
        "cpu_reprojection_input",
        "scorer_binding_authorization",
        "researcher_trace_submission_receipt",
        "feedback_gold_projection_source",
        "feedback_required_target_change",
        "phase5_after_packet",
        "phase5_after_projection",
    )
}
_MEDIA_TYPES["phase5_failure"] = "application/vnd.story-projection.phase5-failure+json"


def persist_phase5_record(
    artifacts: ArtifactStore,
    value: ImmutableRecord,
    *,
    object_kind: Phase5ObjectKind,
    created_at: datetime,
    release_class: ReleaseClass = ReleaseClass.RESTRICTED,
) -> Phase5CASReference:
    """Persist an already-created record; this helper never invents semantic content."""

    expected = phase5_record_reference(
        value,
        object_kind=object_kind,
        release_class=release_class,
    )
    release = expected.release_class
    record = artifacts.put_bytes(
        (value.to_canonical_json() + "\n").encode("utf-8"),
        media_type=expected.media_type,
        release_class=release,
        created_at=created_at,
    )
    observed = Phase5CASReference(
        artifact_hash=record.content_hash,
        logical_content_hash=value.content_hash,
        object_kind=object_kind,
        media_type=record.media_type,
        release_class=record.release_class,
    )
    if observed != expected:
        raise Phase5MaterializationError("persisted Phase 5 CAS metadata changed")
    return observed


def phase5_record_reference(
    value: ImmutableRecord,
    *,
    object_kind: Phase5ObjectKind,
    release_class: ReleaseClass = ReleaseClass.RESTRICTED,
) -> Phase5CASReference:
    """Derive the exact CAS reference without performing a write."""

    release = ReleaseClass(release_class)
    if release is ReleaseClass.PUBLIC and object_kind not in {
        "evidence_packet",
        "evidence_snapshot",
    }:
        raise Phase5MaterializationError(
            "only Phase 5 evidence packets and snapshots may be persisted as public"
        )
    payload = (value.to_canonical_json() + "\n").encode("utf-8")
    return Phase5CASReference(
        artifact_hash=hashlib.sha256(payload).hexdigest(),
        logical_content_hash=value.content_hash,
        object_kind=object_kind,
        media_type=_MEDIA_TYPES[object_kind],
        release_class=release,
    )


def _assert_safe_path(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5MaterializationError(f"symlinked Phase 5 path is forbidden: {path}")
        if current.parent == current:
            return
        current = current.parent


def load_phase5_materialization_source(
    path: Path,
) -> Phase5MaterializationSourceManifest:
    """Load one self-hashed source manifest without following symlinks."""

    _assert_safe_path(path)
    if not path.is_file():
        raise Phase5MaterializationError("Phase 5 materialization source is absent")
    raw = path.read_bytes()
    if len(raw) > 10 * 1024 * 1024:
        raise Phase5MaterializationError("Phase 5 materialization source exceeds 10 MiB")
    try:
        return Phase5MaterializationSourceManifest.model_validate_json(raw)
    except Exception as error:
        raise Phase5MaterializationError(
            f"invalid Phase 5 materialization source: {error}"
        ) from error


def _append_materialization(path: Path, value: ImmutableRecord) -> None:
    payload = (value.to_canonical_json() + "\n").encode("utf-8")
    _assert_safe_path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise Phase5MaterializationError(
                f"append-only Phase 5 materialization changed: {path.name}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_safe_path(path.parent)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Phase5SourceModelCallBinding(ImmutableRecord):
    """Exact held-out ledger rows used to produce one exported projection."""

    model_call_id: str = Field(min_length=1)
    model_call_record_hash: Sha256Digest
    gpu_event_id: str = Field(min_length=1)
    gpu_event_record_hash: Sha256Digest


class Phase5HeldOutProjectionExport(ImmutableRecord):
    """Gold-free bridge from an upstream receipt to typed CAS objects.

    The upstream result hash is not assumed to serialize an ``OntologyProjection``.
    A held-out adapter instead exports an explicit mapping to a separately verified
    projection, packet, and snapshot.
    """

    export_id: str = Field(min_length=1)
    condition: Literal[
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
    ]
    unit_id: str = Field(min_length=1)
    query_stage_hash: Sha256Digest
    seed_block: Literal[1] | None
    projection_receipt_hash: Sha256Digest | None = None
    itt_record_hash: Sha256Digest | None = None
    source_output_artifact_hash: Sha256Digest
    source_condition_output_artifact_hash: Sha256Digest | None = None
    source_validation_artifact_hash: Sha256Digest | None = None
    source_ledger_receipt_hash: Sha256Digest | None = None
    source_model_calls: tuple[Phase5SourceModelCallBinding, ...] = ()
    projection: Phase5CASReference
    packet: Phase5CASReference
    snapshot: Phase5CASReference
    source_completed_at: AwareDatetime
    exported_at: AwareDatetime
    runtime_namespace: Literal["gold_free"] = "gold_free"
    scorer_fields_included: Literal[False] = False
    model_input_open: Literal[False] = False

    @model_validator(mode="after")
    def condition_specific_source(self) -> Self:
        if self.condition is ConditionName.C0_CLASSICAL_PRE:
            if (
                self.seed_block is not None
                or self.projection_receipt_hash is None
                or self.itt_record_hash is not None
                or self.source_model_calls
                or self.source_ledger_receipt_hash is not None
                or self.source_condition_output_artifact_hash is not None
            ):
                raise ValueError("C0 export must be CPU-only and projection-receipt bound")
        elif self.condition is ConditionName.C1_LLM_PRE:
            if (
                self.seed_block != 1
                or self.projection_receipt_hash is None
                or self.itt_record_hash is None
                or not self.source_model_calls
                or self.source_ledger_receipt_hash is None
                or self.source_condition_output_artifact_hash is None
            ):
                raise ValueError("C1 export requires seed-1 projection and model lineage")
        elif (
            self.seed_block != 1
            or self.projection_receipt_hash is not None
            or self.itt_record_hash is None
            or not self.source_model_calls
            or self.source_ledger_receipt_hash is None
            or self.source_condition_output_artifact_hash != self.source_output_artifact_hash
        ):
            raise ValueError("C2 export requires its seed-1 ITT and model lineage")
        for reference, expected_kind in (
            (self.projection, "ontology_projection"),
            (self.packet, "evidence_packet"),
            (self.snapshot, "evidence_snapshot"),
        ):
            if reference.object_kind != expected_kind:
                raise ValueError("held-out export has an incorrectly typed CAS reference")
        if self.exported_at < self.source_completed_at:
            raise ValueError("held-out projection export predates its source result")
        return self


class Phase5ScorerBindingAuthorization(ImmutableRecord):
    """Hash-only post-freeze scorer handoff; gold objects are intentionally absent."""

    authorization_id: str = Field(min_length=1)
    protocol_hash: Sha256Digest
    scorer_bridge_hash: Sha256Digest
    known_answer_source_hash: Sha256Digest
    review_completion_manifest_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    bindings: tuple[ScorerOnlyFeedbackBinding, ...]
    authorized_at: AwareDatetime
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    model_input_open: Literal[False] = False
    contains_gold_objects: Literal[False] = False

    @model_validator(mode="after")
    def exact_hash_only_inventory(self) -> Self:
        keys = [(item.episode_id, item.condition) for item in self.bindings]
        if len(keys) != 18 or len(set(keys)) != 18:
            raise ValueError("scorer authorization requires six scripts by three conditions")
        return self


class Phase5EpisodeSource(ImmutableRecord):
    episode_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    kind: FeedbackEpisodeKind
    unit_id: str = Field(min_length=1)
    query_stage_hash: Sha256Digest
    held_out_seed_block: Literal[1] = 1
    instruction: Phase5CASReference
    scripted_freeze: Phase5CASReference | None = None
    c0_export: Phase5CASReference | None = None
    c1_export: Phase5CASReference | None = None
    c0_cpu_input: Phase5CASReference | None = None
    c1_cpu_input: Phase5CASReference | None = None
    trace_submission_receipt: Phase5CASReference | None = None
    c2_export: Phase5CASReference

    @model_validator(mode="after")
    def exact_episode_surface(self) -> Self:
        if self.instruction.object_kind != "revision_instruction":
            raise ValueError("episode instruction reference has the wrong object kind")
        if self.c2_export.object_kind != "held_out_projection_export":
            raise ValueError("episode C2 export reference has the wrong object kind")
        scripted_values = (
            self.scripted_freeze,
            self.c0_export,
            self.c1_export,
            self.c0_cpu_input,
            self.c1_cpu_input,
        )
        if self.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER:
            if self.trace_submission_receipt is not None:
                raise ValueError("scripted episode cannot carry a researcher receipt")
            if any(item is None for item in scripted_values):
                raise ValueError("scripted episode lacks freeze/C0/C1 materialization")
            expected_kinds = (
                "scripted_revision_freeze",
                "held_out_projection_export",
                "held_out_projection_export",
                "cpu_reprojection_input",
                "cpu_reprojection_input",
            )
            if any(
                item is not None and item.object_kind != expected
                for item, expected in zip(scripted_values, expected_kinds, strict=True)
            ):
                raise ValueError("scripted episode CAS object kind changed")
        elif any(item is not None for item in scripted_values):
            raise ValueError("researcher trace cannot carry scripted or scorer-bound inputs")
        elif (
            self.trace_submission_receipt is None
            or self.trace_submission_receipt.object_kind
            != "researcher_trace_submission_receipt"
        ):
            raise ValueError("researcher trace requires its typed UI submission receipt")
        return self


class Phase5MaterializationSourceManifest(ImmutableRecord):
    source_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    protocol_hash: Sha256Digest
    script_commitment_manifest_hash: Sha256Digest
    primary_results_gate_hash: Sha256Digest
    final_reviewed_seal_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    held_out_call_manifest: Phase5CASReference
    scorer_binding_authorization: Phase5CASReference
    episodes: tuple[Phase5EpisodeSource, ...]
    captured_at: AwareDatetime
    generated_from_test_fixture: Literal[False] = False
    manually_authored_scientific_values: Literal[False] = False

    @model_validator(mode="after")
    def exact_source_inventory(self) -> Self:
        if len(self.episodes) != 9 or len({item.episode_id for item in self.episodes}) != 9:
            raise ValueError("Phase 5 source requires exactly nine distinct episodes")
        kinds = [item.kind for item in self.episodes]
        if kinds.count(FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER) != 6:
            raise ValueError("Phase 5 source requires six scripted episodes")
        if kinds.count(FeedbackEpisodeKind.RESEARCHER_TRACE) != 3:
            raise ValueError("Phase 5 source requires three researcher traces")
        if self.held_out_call_manifest.object_kind != "held_out_call_manifest":
            raise ValueError("Phase 5 source has the wrong call-manifest object kind")
        if self.scorer_binding_authorization.object_kind != "scorer_binding_authorization":
            raise ValueError("Phase 5 source has the wrong scorer authorization object kind")
        return self


class Phase5InputEpisodeLineage(ImmutableRecord):
    episode_id: str
    query_stage_hash: Sha256Digest
    horizon_hash: Sha256Digest
    packet_hash: Sha256Digest
    held_out_seed_block: Literal[1] = 1
    source_receipt_hashes: tuple[Sha256Digest, ...]
    source_model_call_record_hashes: tuple[Sha256Digest, ...]


class Phase5InputMaterializationReceipt(ImmutableRecord):
    source_manifest_hash: Sha256Digest
    input_manifest_hash: Sha256Digest
    script_commitment_manifest_hash: Sha256Digest
    primary_results_gate_hash: Sha256Digest
    held_out_execution_manifest_hash: Sha256Digest
    held_out_call_manifest_hash: Sha256Digest
    scorer_bridge_hash: Sha256Digest
    scorer_binding_authorization_hash: Sha256Digest
    episodes: tuple[Phase5InputEpisodeLineage, ...]
    materialized_at: AwareDatetime
    scorer_gold_read: Literal[False] = False
    model_service_called: Literal[False] = False


def load_phase5_materialization_receipt(
    path: Path,
) -> Phase5InputMaterializationReceipt:
    _assert_safe_path(path)
    if not path.is_file():
        raise Phase5MaterializationError("Phase 5 materialization receipt is absent")
    try:
        return Phase5InputMaterializationReceipt.model_validate_json(path.read_bytes())
    except Exception as error:
        raise Phase5MaterializationError(
            f"invalid Phase 5 materialization receipt: {error}"
        ) from error


def validate_phase5_materialization_receipt(
    inputs: Phase5ExecutionInputManifest,
    receipt: Phase5InputMaterializationReceipt,
) -> None:
    expected_inputs = {
        item.episode_id: item for item in (*inputs.scripted_inputs, *inputs.researcher_trace_inputs)
    }
    received = {item.episode_id: item for item in receipt.episodes}
    if (
        receipt.input_manifest_hash != inputs.content_hash
        or receipt.source_manifest_hash != inputs.source_manifest_hash
        or receipt.script_commitment_manifest_hash
        != inputs.script_commitment_manifest_hash
        or receipt.primary_results_gate_hash != inputs.primary_results_gate_hash
        or receipt.held_out_execution_manifest_hash != inputs.held_out_execution_manifest_hash
        or receipt.held_out_call_manifest_hash != inputs.held_out_call_manifest_hash
        or receipt.scorer_binding_authorization_hash != inputs.scorer_binding_authorization_hash
        or len(receipt.episodes) != 9
        or len(received) != 9
        or set(received) != set(expected_inputs)
    ):
        raise Phase5MaterializationError(
            "Phase 5 input differs from its authoritative materialization receipt"
        )
    for episode_id, item in expected_inputs.items():
        lineage = received[episode_id]
        if (
            lineage.packet_hash != item.packet.content_hash
            or lineage.horizon_hash != item.instruction.before_context.spoiler_horizon.content_hash
            or not lineage.source_receipt_hashes
            or not lineage.source_model_call_record_hashes
        ):
            raise Phase5MaterializationError(
                "Phase 5 episode differs from its materialization lineage"
            )


def _event_by_id(artifacts: ArtifactStore, event_id: str) -> GpuEvent:
    matches = [item for item in artifacts.ledger.gpu_events() if item.event_id == event_id]
    if len(matches) != 1:
        raise Phase5MaterializationError(f"GPU event {event_id!r} is missing or duplicated")
    return matches[0]


def _ledger_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Phase5AdapterError("GPU ledger timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Phase5AdapterError("GPU ledger timestamp must be timezone-aware")
    return parsed


def _verify_source_model_calls(
    *,
    export: Phase5HeldOutProjectionExport,
    source_result: HeldOutITTRecord,
    call: HeldOutCallSpec,
    artifacts: ArtifactStore,
) -> None:
    references = export.source_model_calls
    if not 1 <= len(references) <= 2:
        raise Phase5MaterializationError("held-out export requires one base and at most one repair")
    model_calls = []
    attempts = []
    events = []
    for reference in references:
        try:
            model_call = artifacts.ledger.get_model_call(reference.model_call_id)
            lineage = artifacts.ledger.attempt_lineage(model_call.attempt_id)
        except KeyError as error:
            raise Phase5MaterializationError(
                "held-out export cites a missing model call"
            ) from error
        event = _event_by_id(artifacts, reference.gpu_event_id)
        if (
            model_call.gpu_event_id != event.event_id
            or feedback_model_call_record_hash(model_call) != reference.model_call_record_hash
            or feedback_gpu_event_record_hash(event) != reference.gpu_event_record_hash
            or model_call.backend is not ModelBackend.VLLM_GPU
            or model_call.allocated_gpu_microseconds != event.allocated_microseconds
            or model_call.successful is not event.succeeded
            or model_call.job_id != event.job_id
            or model_call.attempt_id != event.attempt_id
        ):
            raise Phase5MaterializationError("held-out export model/GPU ledger rows changed")
        model_calls.append(model_call)
        attempts.append(lineage[-1])
        events.append(event)
    first_expected_role = (
        ModelCallRole.PREBUILD
        if export.condition is ConditionName.C1_LLM_PRE
        else ModelCallRole.QUERY_TIME
    )
    first_expected_retry = (
        RetryClass.LONG if export.condition is ConditionName.C1_LLM_PRE else RetryClass.STANDARD
    )
    if (
        attempts[0].attempt_kind is not AttemptKind.BASE
        or model_calls[0].call_role is not first_expected_role
        or model_calls[0].retry_class is not first_expected_retry
        or events[0].event_kind is not GpuEventKind.INFERENCE
        or model_calls[0].request_hash != source_result.result.request_hash
        or attempts[0].seed != call.vllm_seed
    ):
        raise Phase5MaterializationError("held-out base call differs from its registered slot")
    if len(model_calls) == 2 and (
        attempts[1].attempt_kind is not AttemptKind.REPAIR
        or attempts[1].parent_attempt_id != attempts[0].attempt_id
        or model_calls[1].call_role is not ModelCallRole.REPAIR
        or model_calls[1].retry_class is not first_expected_retry
        or events[1].event_kind is not GpuEventKind.REPAIR
        or attempts[1].seed != call.vllm_seed
    ):
        raise Phase5MaterializationError("held-out repair lineage changed")
    if len(model_calls) != source_result.result.repair_attempts + 1:
        raise Phase5MaterializationError("held-out export repair count differs from ITT")
    allocated = sum(item.allocated_seconds for item in events)
    if not math.isclose(
        allocated,
        source_result.result.allocated_gpu_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise Phase5MaterializationError("held-out export GPU allocation differs from ITT")


def validate_phase5_source_model_calls(
    *,
    export: Phase5HeldOutProjectionExport,
    source_result: HeldOutITTRecord,
    call: HeldOutCallSpec,
    artifacts: ArtifactStore,
) -> None:
    """Public producer-side replay of exact held-out model/GPU ledger rows."""

    _verify_source_model_calls(
        export=export,
        source_result=source_result,
        call=call,
        artifacts=artifacts,
    )


def _load_export(
    *,
    reference: Phase5CASReference,
    expected_condition: ConditionName,
    episode: Phase5EpisodeSource,
    expected_frozen_seed: int,
    call_manifest: HeldOutCallManifest,
    execution: HeldOutExecutionManifest,
    cas: RestrictedPhase5CAS,
) -> tuple[
    Phase5HeldOutProjectionExport,
    OntologyProjection,
    EvidencePacket,
    EvidenceSnapshot,
    tuple[str, ...],
]:
    export = cas.load(
        reference,
        Phase5HeldOutProjectionExport,
        object_kind="held_out_projection_export",
    )
    if (
        export.condition is not expected_condition
        or export.unit_id != episode.unit_id
        or export.query_stage_hash != episode.query_stage_hash
        or export.seed_block
        != (None if expected_condition is ConditionName.C0_CLASSICAL_PRE else 1)
    ):
        raise Phase5MaterializationError("held-out projection export uses another slot")
    projection = cas.load(export.projection, OntologyProjection, object_kind="ontology_projection")
    packet = cas.load(export.packet, EvidencePacket, object_kind="evidence_packet")
    snapshot = cas.load(export.snapshot, EvidenceSnapshot, object_kind="evidence_snapshot")
    if (
        projection.condition is not expected_condition
        or projection.packet_hash != packet.content_hash
        or projection.snapshot_hash != snapshot.content_hash
        or packet.snapshot_hash != snapshot.content_hash
    ):
        raise Phase5MaterializationError("held-out exported projection boundary changed")

    calls_by_hash = {item.content_hash: item for item in call_manifest.calls}
    planned_unit = next(
        (item for item in call_manifest.units if item.unit_id == episode.unit_id), None
    )
    planned_query = (
        None
        if planned_unit is None
        else next(
            (
                item
                for item in planned_unit.query_stages
                if item.staging_manifest_hash == episode.query_stage_hash
            ),
            None,
        )
    )
    if planned_query is None:
        raise Phase5MaterializationError("held-out export query stage is unregistered")
    itt_by_hash = {item.content_hash: item for item in execution.itt_records}
    receipt_by_hash = {item.content_hash: item for item in execution.preconstructed_projections}
    source_hashes: list[str] = []
    if expected_condition in {
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
    }:
        receipt = receipt_by_hash.get(export.projection_receipt_hash or "")
        if receipt is None:
            raise Phase5MaterializationError("exported CPU projection receipt is absent")
        expected_seed = None if expected_condition is ConditionName.C0_CLASSICAL_PRE else 1
        if (
            receipt.condition is not expected_condition
            or receipt.unit_id != episode.unit_id
            or receipt.query_stage_hash != episode.query_stage_hash
            or receipt.seed_block != expected_seed
            or receipt.outcome is not RunOutcome.SUCCEEDED
            or receipt.projection_artifact_hash != export.source_output_artifact_hash
            or receipt.evidence_packet_hash != packet.content_hash
            or receipt.horizon_hash != snapshot.horizon.content_hash
            or (
                expected_condition is ConditionName.C0_CLASSICAL_PRE
                and export.source_completed_at != receipt.completed_at
            )
        ):
            raise Phase5MaterializationError("projection export differs from its CPU receipt")
        seal = projection.construction_seal
        if seal is None or seal.content_hash != receipt.source_construction_seal_hash:
            raise Phase5MaterializationError("exported CPU projection uses another seal")
        source_hashes.append(receipt.content_hash)
        if expected_condition is ConditionName.C0_CLASSICAL_PRE:
            return export, projection, packet, snapshot, tuple(source_hashes)

    itt = itt_by_hash.get(export.itt_record_hash or "")
    if itt is None:
        raise Phase5MaterializationError("exported LLM projection ITT record is absent")
    call = calls_by_hash.get(itt.call_spec_hash)
    if call is None:
        raise Phase5MaterializationError("exported ITT record has no registered call")
    wanted_class = "test_c1" if expected_condition is ConditionName.C1_LLM_PRE else "test_c2"
    if (
        call.call_class != wanted_class
        or call.unit_id != episode.unit_id
        or call.seed_block != 1
        or call.frozen_seed != expected_frozen_seed
        or (
            expected_condition is ConditionName.C2_LLM_QUERY
            and call.query_stage_hash != episode.query_stage_hash
        )
        or itt.result.outcome is not RunOutcome.SUCCEEDED
        or not itt.result.request_started
        or itt.result.call_id != call.call_id
        or itt.result.condition is not expected_condition
        or itt.result.output_artifact_hash != export.source_condition_output_artifact_hash
        or itt.result.validation_artifact_hash != export.source_validation_artifact_hash
        or itt.result.ledger_receipt_hash != export.source_ledger_receipt_hash
        or itt.result.evidence_packet_hash not in {None, packet.content_hash}
        or (
            expected_condition is ConditionName.C2_LLM_QUERY
            and (
                itt.result.evidence_packet_hash != packet.content_hash
                or itt.result.horizon_hash != planned_query.horizon_hash
                or itt.result.budget_hash != planned_query.budget_hash
            )
        )
    ):
        raise Phase5MaterializationError("projection export differs from its LLM ITT record")
    _verify_source_model_calls(
        export=export,
        source_result=itt,
        call=call,
        artifacts=cas.artifacts,
    )
    if expected_condition is ConditionName.C1_LLM_PRE:
        receipt = receipt_by_hash[export.projection_receipt_hash]
        if export.source_completed_at != max(receipt.completed_at, itt.result.completed_at):
            raise Phase5MaterializationError("C1 export source completion time changed")
        seal = projection.construction_seal
        if (
            seal is None
            or seal.content_hash != itt.result.construction_seal_hash
            or seal.ontology_hash != itt.result.complete_c1_graph_hash
        ):
            raise Phase5MaterializationError("C1 export differs from its query-blind seal")
    else:
        if export.source_completed_at != itt.result.completed_at:
            raise Phase5MaterializationError("C2 export source completion time changed")
        certificate = projection.construction_certificate
        inventory = projection.pre_query_inventory
        if (
            certificate is None
            or inventory is None
            or certificate.content_hash != itt.result.construction_certificate_hash
            or inventory.content_hash != itt.result.empty_prequery_inventory_hash
            or certificate.completed_at > itt.result.completed_at
        ):
            raise Phase5MaterializationError("C2 export lost its construction certificate")
    source_hashes.append(itt.content_hash)
    return export, projection, packet, snapshot, tuple(source_hashes)


def _assert_source_ledger_calls_unique(
    exports: list[Phase5HeldOutProjectionExport],
) -> None:
    call_ids = [
        reference.model_call_id for export in exports for reference in export.source_model_calls
    ]
    if len(call_ids) != len(set(call_ids)):
        raise Phase5MaterializationError("one held-out model-call row was reused across exports")


def materialize_phase5_input_manifest(
    *,
    source: Phase5MaterializationSourceManifest,
    protocol: FeedbackProtocolConfiguration,
    script_commitment_path: Path,
    primary_results_gate_path: Path,
    benchmark_root: Path,
    review_completion_root: Path,
    cas: RestrictedPhase5CAS,
) -> tuple[Phase5ExecutionInputManifest, Phase5InputMaterializationReceipt]:
    """Materialize the nine exact inputs without reading scorer gold or calling a model."""

    if source.protocol_hash != protocol.content_hash:
        raise Phase5MaterializationError("source manifest uses another feedback protocol")
    # Import locally so the pre-output commitment module can reuse the restricted
    # CAS contracts here without introducing an import cycle.
    from story_projection_onto.phase5_commitment import load_phase5_script_commitment

    commitment = load_phase5_script_commitment(script_commitment_path)
    if (
        commitment.content_hash != source.script_commitment_manifest_hash
        or commitment.protocol_hash != protocol.content_hash
        or commitment.benchmark_draft_seal_hash != protocol.benchmark_draft_seal_hash
        or commitment.committed_at >= commitment.scheduled_activation_at
        or commitment.scheduled_activation_at > source.captured_at
    ):
        raise Phase5MaterializationError(
            "source manifest uses another or invalid pre-output script commitment"
        )
    gate, execution, bridge = replay_phase5_prerequisites(
        expected_gate_hash=source.primary_results_gate_hash,
        expected_final_reviewed_seal_hash=source.final_reviewed_seal_hash,
        expected_consumer_frozen_at=source.captured_at,
        primary_results_gate_path=primary_results_gate_path,
        benchmark_root=benchmark_root,
        review_completion_root=review_completion_root,
    )
    if (
        source.held_out_execution_manifest_hash != execution.content_hash
        or source.primary_results_gate_hash != gate.content_hash
        or commitment.review_gate.review_completion_manifest_hash
        != execution.review_completion_manifest_hash
        or commitment.review_gate.final_reviewed_seal_hash
        != execution.final_reviewed_seal_hash
        or commitment.review_gate.final_reviewed_seal_hash
        != source.final_reviewed_seal_hash
    ):
        raise Phase5MaterializationError("source manifest binds another held-out execution")
    call_manifest = cas.load(
        source.held_out_call_manifest,
        HeldOutCallManifest,
        object_kind="held_out_call_manifest",
    )
    if (
        call_manifest.content_hash != execution.call_manifest_hash
        or call_manifest.content_hash != source.held_out_call_manifest.logical_content_hash
        or call_manifest.content_hash
        != commitment.parent_readiness_floor.held_out_call_manifest_hash
    ):
        raise Phase5MaterializationError("held-out call manifest changed after execution")
    scorer = cas.load(
        source.scorer_binding_authorization,
        Phase5ScorerBindingAuthorization,
        object_kind="scorer_binding_authorization",
    )
    if (
        scorer.protocol_hash != protocol.content_hash
        or scorer.scorer_bridge_hash != bridge.content_hash
        or scorer.review_completion_manifest_hash
        != execution.review_completion_manifest_hash
        or scorer.final_reviewed_seal_hash != execution.final_reviewed_seal_hash
        or scorer.authorized_at < bridge.authorized_at
        or scorer.authorized_at > source.captured_at
    ):
        raise Phase5MaterializationError("hash-only scorer authorization is not post-freeze")

    source_by_id = {item.episode_id: item for item in source.episodes}
    expected_scripts = {item.episode_id: item for item in protocol.scripted_episodes}
    expected_traces = {item.episode_id: item for item in protocol.researcher_trace_slots}
    commitment_by_id = {item.episode_id: item for item in commitment.entries}
    if set(source_by_id) != set(expected_scripts) | set(expected_traces):
        raise Phase5MaterializationError("source episode IDs differ from the frozen protocol")
    if set(commitment_by_id) != set(expected_scripts):
        raise Phase5MaterializationError(
            "script commitment differs from the six frozen protocol episodes"
        )
    scorer_by_key = {(item.episode_id, item.condition): item for item in scorer.bindings}
    expected_scorer_keys = {
        (episode_id, condition)
        for episode_id in expected_scripts
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        )
    }
    if set(scorer_by_key) != expected_scorer_keys:
        raise Phase5MaterializationError(
            "scorer authorization differs from the six frozen scripted episodes"
        )
    scripted_inputs: list[ScriptedFeedbackInput] = []
    trace_inputs: list[ResearcherTraceInput] = []
    lineages: list[Phase5InputEpisodeLineage] = []
    all_exports: list[Phase5HeldOutProjectionExport] = []

    units = {item.unit_id: item for item in call_manifest.units}
    for episode_id in (
        *(item.episode_id for item in protocol.scripted_episodes),
        *(item.episode_id for item in protocol.researcher_trace_slots),
    ):
        episode = source_by_id[episode_id]
        selection = expected_scripts.get(episode_id)
        trace_slot = expected_traces.get(episode_id)
        expected_kind = (
            FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
            if selection is not None
            else FeedbackEpisodeKind.RESEARCHER_TRACE
        )
        if episode.kind is not expected_kind:
            raise Phase5MaterializationError("episode kind differs from the frozen protocol")
        instruction = cas.load(
            episode.instruction,
            RevisionInstruction,
            object_kind="revision_instruction",
        )
        if instruction.before_context.context_id != episode.context_id or episode.context_id != (
            selection.context_id if selection is not None else trace_slot.context_id
        ):
            raise Phase5MaterializationError("episode context differs from the frozen protocol")
        unit = units.get(episode.unit_id)
        query_stage = (
            None
            if unit is None
            else next(
                (
                    item
                    for item in unit.query_stages
                    if item.staging_manifest_hash == episode.query_stage_hash
                ),
                None,
            )
        )
        if query_stage is None or (
            query_stage.query_context_hash != instruction.before_context.content_hash
            or query_stage.horizon_hash != instruction.before_context.spoiler_horizon.content_hash
            or query_stage.budget_hash != instruction.before_context.budgets.content_hash
        ):
            raise Phase5MaterializationError("episode query/horizon/budgets changed")

        c2_export, c2_projection, packet, snapshot, c2_sources = _load_export(
            reference=episode.c2_export,
            expected_condition=ConditionName.C2_LLM_QUERY,
            episode=episode,
            expected_frozen_seed=protocol.llm_seed,
            call_manifest=call_manifest,
            execution=execution,
            cas=cas,
        )
        all_exports.append(c2_export)
        if (
            snapshot.horizon.content_hash != query_stage.horizon_hash
            or snapshot.sealed_at >= instruction.before_context.revealed_at
            or c2_projection.context_hash != instruction.before_context.content_hash
            or c2_projection.construction_certificate is None
            or c2_projection.construction_certificate.completed_at
            >= instruction.revision.created_at
        ):
            raise Phase5MaterializationError("episode violates prequery/query/revision chronology")
        assert_revision_anchors_resolve_in_packet(instruction, packet)

        source_receipts = list(c2_sources)
        if selection is not None:
            committed_script = commitment_by_id[episode_id]
            if instruction.revision.action is not selection.action:
                raise Phase5MaterializationError("scripted action differs from its selection")
            assert episode.scripted_freeze is not None
            assert episode.c0_export is not None and episode.c1_export is not None
            assert episode.c0_cpu_input is not None and episode.c1_cpu_input is not None
            freeze = cas.load(
                episode.scripted_freeze,
                ScriptedRevisionFreeze,
                object_kind="scripted_revision_freeze",
            )
            c0_export, c0_projection, c0_packet, c0_snapshot, c0_sources = _load_export(
                reference=episode.c0_export,
                expected_condition=ConditionName.C0_CLASSICAL_PRE,
                episode=episode,
                expected_frozen_seed=protocol.llm_seed,
                call_manifest=call_manifest,
                execution=execution,
                cas=cas,
            )
            c1_export, c1_projection, c1_packet, c1_snapshot, c1_sources = _load_export(
                reference=episode.c1_export,
                expected_condition=ConditionName.C1_LLM_PRE,
                episode=episode,
                expected_frozen_seed=protocol.llm_seed,
                call_manifest=call_manifest,
                execution=execution,
                cas=cas,
            )
            all_exports.extend((c0_export, c1_export))
            if not (packet == c0_packet == c1_packet and snapshot == c0_snapshot == c1_snapshot):
                raise Phase5MaterializationError("script conditions do not share exact evidence")
            c0_cpu = cas.load(
                episode.c0_cpu_input,
                CpuReprojectionInput,
                object_kind="cpu_reprojection_input",
            )
            c1_cpu = cas.load(
                episode.c1_cpu_input,
                CpuReprojectionInput,
                object_kind="cpu_reprojection_input",
            )
            if (
                c0_cpu.before_projection != c0_projection
                or c1_cpu.before_projection != c1_projection
                or c0_cpu.source_query_stage_hash != episode.query_stage_hash
                or c1_cpu.source_query_stage_hash != episode.query_stage_hash
                or freeze.instruction_hash != instruction.content_hash
                or committed_script.context_id != episode.context_id
                or committed_script.action is not selection.action
                or committed_script.query_stage_relative_path
                != query_stage.relative_path
                or committed_script.query_stage_manifest_hash != episode.query_stage_hash
                or committed_script.query_stage_manifest_file_sha256
                != query_stage.manifest_file_sha256
                or committed_script.query_artifact_hash != query_stage.query_artifact_hash
                or committed_script.model_visible_evidence_hash
                != query_stage.evidence_artifact_hash
                or committed_script.instruction != episode.instruction
                or committed_script.freeze != episode.scripted_freeze
                or committed_script.scheduled_activation_at
                != instruction.revision.created_at
                or freeze.frozen_at != commitment.committed_at
                or max(
                    c0_export.source_completed_at,
                    c1_export.source_completed_at,
                    c2_export.source_completed_at,
                )
                >= commitment.scheduled_activation_at
                or freeze.frozen_at
                >= min(
                    c0_export.source_completed_at,
                    c1_export.source_completed_at,
                    c2_export.source_completed_at,
                )
            ):
                raise Phase5MaterializationError("script freeze or CPU source lineage changed")
            bindings = tuple(
                scorer_by_key[(episode_id, condition)]
                for condition in (
                    ConditionName.C0_CLASSICAL_PRE,
                    ConditionName.C1_LLM_PRE,
                    ConditionName.C2_LLM_QUERY,
                )
            )
            scripted_inputs.append(
                ScriptedFeedbackInput(
                    episode_id=episode_id,
                    instruction=instruction,
                    packet=packet,
                    freeze=freeze,
                    cpu_inputs=(c0_cpu, c1_cpu),
                    c2_before_projection=c2_projection,
                    scorer_bindings=bindings,
                )
            )
            source_receipts.extend((*c0_sources, *c1_sources))
        else:
            assert trace_slot is not None
            assert episode.trace_submission_receipt is not None
            submission_receipt = cas.load(
                episode.trace_submission_receipt,
                ResearcherTraceSubmissionReceipt,
                object_kind="researcher_trace_submission_receipt",
            )
            if instruction.revision.action not in trace_slot.allowed_actions:
                raise Phase5MaterializationError("trace action is outside its registered surface")
            before_bundle = build_visualization_bundle(
                c2_projection,
                instruction.before_context,
                packet,
                content_scope=VisualizationContentScope.REGISTERED_DISPLAY,
            )
            try:
                submission = RevisionDraftSubmission.model_validate_json(
                    submission_receipt.submission_canonical_json
                )
                replayed_instruction = _compile_submission(
                    submission,
                    before_bundle,
                    created_at=submission_receipt.requested_at,
                )
            except Exception as error:
                raise Phase5MaterializationError(
                    "trace UI submission cannot reproduce its instruction"
                ) from error
            if (
                submission_receipt.episode_id != episode_id
                or submission_receipt.action is not instruction.revision.action
                or submission_receipt.requested_at != instruction.revision.created_at
                or submission_receipt.instruction_hash != instruction.content_hash
                or submission_receipt.before_projection_id != c2_projection.projection_id
                or submission_receipt.before_projection_hash != c2_projection.content_hash
                or submission_receipt.before_bundle_hash != before_bundle.content_hash
                or submission.content_hash != submission_receipt.submission_hash
                or replayed_instruction != instruction
                or submission_receipt.recorded_at > source.captured_at
            ):
                raise Phase5MaterializationError(
                    "trace instruction differs from its UI submission receipt"
                )
            trace_inputs.append(
                ResearcherTraceInput(
                    episode_id=episode_id,
                    instruction=instruction,
                    submission_receipt=submission_receipt,
                    packet=packet,
                    c2_before_projection=c2_projection,
                )
            )
            source_receipts.append(submission_receipt.content_hash)

        episode_exports = [c2_export] if selection is None else [c0_export, c1_export, c2_export]
        lineages.append(
            Phase5InputEpisodeLineage(
                episode_id=episode_id,
                query_stage_hash=episode.query_stage_hash,
                horizon_hash=query_stage.horizon_hash,
                packet_hash=packet.content_hash,
                source_receipt_hashes=tuple(source_receipts),
                source_model_call_record_hashes=tuple(
                    reference.model_call_record_hash
                    for export in episode_exports
                    for reference in export.source_model_calls
                ),
            )
        )

    authorized_outputs = set(bridge.output_artifact_hashes)
    authorized_projection_receipts = set(bridge.projection_receipt_hashes)
    authorized_itt_records = set(bridge.itt_record_hashes)
    for export in all_exports:
        if export.source_output_artifact_hash not in authorized_outputs:
            raise Phase5MaterializationError(
                "held-out projection export is absent from the frozen scorer bridge"
            )
        if (
            export.source_condition_output_artifact_hash is not None
            and export.source_condition_output_artifact_hash not in authorized_outputs
        ):
            raise Phase5MaterializationError(
                "held-out condition output is absent from the frozen scorer bridge"
            )
        if (
            export.projection_receipt_hash is not None
            and export.projection_receipt_hash not in authorized_projection_receipts
        ):
            raise Phase5MaterializationError(
                "projection receipt is absent from the frozen scorer bridge"
            )
        if (
            export.itt_record_hash is not None
            and export.itt_record_hash not in authorized_itt_records
        ):
            raise Phase5MaterializationError("ITT record is absent from the frozen scorer bridge")
    _assert_source_ledger_calls_unique(all_exports)
    inputs = Phase5ExecutionInputManifest(
        run_id=source.run_id,
        protocol_hash=protocol.content_hash,
        script_commitment_manifest_hash=commitment.content_hash,
        primary_results_gate_hash=gate.content_hash,
        final_reviewed_seal_hash=source.final_reviewed_seal_hash,
        source_manifest_hash=source.content_hash,
        held_out_execution_manifest_hash=execution.content_hash,
        held_out_call_manifest_hash=call_manifest.content_hash,
        scorer_binding_authorization_hash=scorer.content_hash,
        frozen_at=source.captured_at,
        scripted_inputs=tuple(scripted_inputs),
        researcher_trace_inputs=tuple(trace_inputs),
    )
    validate_phase5_inputs(inputs, protocol)
    receipt = Phase5InputMaterializationReceipt(
        source_manifest_hash=source.content_hash,
        input_manifest_hash=inputs.content_hash,
        script_commitment_manifest_hash=commitment.content_hash,
        primary_results_gate_hash=gate.content_hash,
        held_out_execution_manifest_hash=execution.content_hash,
        held_out_call_manifest_hash=call_manifest.content_hash,
        scorer_bridge_hash=bridge.content_hash,
        scorer_binding_authorization_hash=scorer.content_hash,
        episodes=tuple(lineages),
        materialized_at=source.captured_at,
    )
    return inputs, receipt


def materialize_phase5_inputs_to_directory(
    *,
    source: Phase5MaterializationSourceManifest,
    protocol: FeedbackProtocolConfiguration,
    script_commitment_path: Path,
    primary_results_gate_path: Path,
    benchmark_root: Path,
    review_completion_root: Path,
    cas: RestrictedPhase5CAS,
    output_root: Path,
) -> Phase5InputMaterializationReceipt:
    """Write the exact typed input and its lineage receipt, idempotently."""

    root = output_root.absolute()
    _assert_safe_path(root)
    allowed = {root / "input_manifest.json", root / "materialization_receipt.json"}
    if root.exists():
        if not root.is_dir():
            raise Phase5MaterializationError("Phase 5 materialization root is not a directory")
        for path in root.iterdir():
            if path.is_symlink() or not path.is_file() or path not in allowed:
                raise Phase5MaterializationError(
                    f"unexpected Phase 5 materialization path: {path.name}"
                )
    inputs, receipt = materialize_phase5_input_manifest(
        source=source,
        protocol=protocol,
        script_commitment_path=script_commitment_path,
        primary_results_gate_path=primary_results_gate_path,
        benchmark_root=benchmark_root,
        review_completion_root=review_completion_root,
        cas=cas,
    )
    root.mkdir(parents=True, exist_ok=True)
    _append_materialization(root / "input_manifest.json", inputs)
    _append_materialization(root / "materialization_receipt.json", receipt)
    return receipt


class Phase5OwnedServiceIdentity(ImmutableRecord):
    """Identity of a live service handed to, but never owned by, Phase 5."""

    service_id: str = Field(min_length=1)
    condition: Literal[ConditionName.C2_LLM_QUERY] = ConditionName.C2_LLM_QUERY
    global_accounting_id: str = Field(min_length=1)
    model_manifest_hash: Sha256Digest
    decoding_manifest_hash: Sha256Digest
    model_load_event_id: str = Field(min_length=1)
    model_load_event_record_hash: Sha256Digest
    cumulative_gpu_seconds_at_handoff: float = Field(gt=0.0)
    handed_off_at: AwareDatetime
    already_running: Literal[True] = True
    lifecycle_owner: Literal["external"] = "external"
    controller_can_start_service: Literal[False] = False
    controller_can_stop_service: Literal[False] = False
    exclusive_accounting_lease: Literal[True] = True


class Phase5OwnedServiceResult(ImmutableRecord):
    """Raw service handoff; the adapter independently reconstructs its receipt."""

    request_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    attempt_status: FeedbackAttemptStatus
    model_call_ids: tuple[str, ...]
    after_projection: Phase5CASReference | None = None
    after_packet: Phase5CASReference | None = None
    failure_artifact: Phase5CASReference | None = None
    resolver_hash: Sha256Digest
    resolved_at: AwareDatetime
    checked_at: AwareDatetime
    cumulative_gpu_seconds_before: float = Field(gt=0.0)
    cumulative_gpu_seconds_after: float = Field(gt=0.0)

    @model_validator(mode="after")
    def terminal_and_bounded(self) -> Self:
        if not 1 <= len(self.model_call_ids) <= 2:
            raise ValueError("feedback service requires one base and at most one repair")
        if len(self.model_call_ids) != len(set(self.model_call_ids)):
            raise ValueError("feedback service model-call IDs must be unique")
        if self.cumulative_gpu_seconds_after <= self.cumulative_gpu_seconds_before:
            raise ValueError("feedback service cumulative GPU allocation did not advance")
        if self.checked_at < self.resolved_at:
            raise ValueError("feedback service check predates resolution")
        if self.attempt_status is FeedbackAttemptStatus.SUCCEEDED:
            if self.after_projection is None or self.failure_artifact is not None:
                raise ValueError("successful feedback service result requires only a projection")
        elif self.failure_artifact is None or self.after_projection is not None:
            raise ValueError(
                "nonsuccessful feedback service result requires only a failure artifact"
            )
        if self.after_projection is None and self.after_packet is not None:
            raise ValueError("feedback service packet cannot exist without a projection")
        if self.after_projection is not None and self.after_projection.object_kind != (
            "phase5_after_projection"
        ):
            raise ValueError("feedback service projection reference has the wrong kind")
        if self.after_packet is not None and self.after_packet.object_kind != (
            "phase5_after_packet"
        ):
            raise ValueError("feedback service packet reference has the wrong kind")
        if self.failure_artifact is not None and self.failure_artifact.object_kind != (
            "phase5_failure"
        ):
            raise ValueError("feedback failure reference has the wrong kind")
        return self


@runtime_checkable
class AlreadyOwnedPhase5C2Service(Protocol):
    """Narrow service lease with no load/start/stop capability."""

    def identity(self) -> Phase5OwnedServiceIdentity: ...

    def execute_feedback(
        self,
        request: C2RegenerationRequest,
        model_visible_payload: dict[str, object],
    ) -> Phase5OwnedServiceResult: ...

    def recover_feedback(
        self,
        request: C2RegenerationRequest,
    ) -> Phase5OwnedServiceResult | None: ...


class MeteredPhase5C2RegenerationAdapter:
    """Concrete adapter over one externally managed, cumulatively metered service."""

    def __init__(
        self,
        *,
        service: AlreadyOwnedPhase5C2Service,
        cas: RestrictedPhase5CAS,
    ) -> None:
        self.service = service
        self.cas = cas
        self._identity = service.identity()
        self._verify_identity()

    def _verify_identity(self) -> None:
        identity = self._identity
        event = _event_by_id(self.cas.artifacts, identity.model_load_event_id)
        total = self.cas.artifacts.ledger.gpu_summary().total_allocated_seconds
        if (
            event.event_kind is not GpuEventKind.MODEL_LOAD
            or not event.succeeded
            or feedback_gpu_event_record_hash(event) != identity.model_load_event_record_hash
            or _ledger_datetime(event.ended_at) > identity.handed_off_at
            or total + 1e-6 < identity.cumulative_gpu_seconds_at_handoff
        ):
            raise Phase5AdapterError("owned Phase 5 service identity/ledger handoff changed")

    def regenerate(self, request: C2RegenerationRequest) -> C2RegenerationResult:
        """Recover an exact prior request before consuming a new registered call."""

        payload = request.model_visible_payload()
        scan_model_payload(payload)
        recovered = self.service.recover_feedback(request)
        if recovered is not None:
            current = self.cas.artifacts.ledger.gpu_summary().total_allocated_seconds
            if current + 1e-6 < recovered.cumulative_gpu_seconds_after:
                raise Phase5AdapterError("recovered feedback result exceeds cumulative GPU ledger")
            return self._compile(request, recovered, recovering=True)
        before = self.cas.artifacts.ledger.gpu_summary().total_allocated_seconds
        if before + 1e-6 < self._identity.cumulative_gpu_seconds_at_handoff:
            raise Phase5AdapterError("cumulative GPU ledger regressed before feedback")
        self.cas.artifacts.ledger.require_gpu_capacity(150)
        service_result = self.service.execute_feedback(request, payload)
        after = self.cas.artifacts.ledger.gpu_summary().total_allocated_seconds
        if not math.isclose(
            service_result.cumulative_gpu_seconds_before,
            before,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            service_result.cumulative_gpu_seconds_after,
            after,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise Phase5AdapterError("service result does not bind the cumulative GPU ledger")
        return self._compile(request, service_result, recovering=False)

    def recover(self, request: C2RegenerationRequest) -> C2RegenerationResult | None:
        """Recover ledger/CAS state without any inference or lifecycle side effect."""

        service_result = self.service.recover_feedback(request)
        if service_result is None:
            return None
        current = self.cas.artifacts.ledger.gpu_summary().total_allocated_seconds
        if current + 1e-6 < service_result.cumulative_gpu_seconds_after:
            raise Phase5AdapterError("recovered feedback result exceeds cumulative GPU ledger")
        return self._compile(request, service_result, recovering=True)

    def _compile(
        self,
        request: C2RegenerationRequest,
        service_result: Phase5OwnedServiceResult,
        *,
        recovering: bool,
    ) -> C2RegenerationResult:
        if (
            service_result.request_hash != request.content_hash
            or service_result.service_identity_hash != self._identity.content_hash
        ):
            raise Phase5AdapterError("owned service result belongs to another request/session")
        references = self._ledger_references(request, service_result)
        allocated = sum(item.allocated_gpu_seconds for item in references)
        delta = (
            service_result.cumulative_gpu_seconds_after
            - service_result.cumulative_gpu_seconds_before
        )
        if not math.isclose(allocated, delta, rel_tol=0.0, abs_tol=1e-6):
            raise Phase5AdapterError("feedback model rows do not explain cumulative GPU delta")

        after_projection = None
        after_packet = None
        if service_result.after_projection is not None:
            after_projection = self.cas.load(
                service_result.after_projection,
                OntologyProjection,
                object_kind="phase5_after_projection",
            )
            after_packet = (
                request.packet
                if service_result.after_packet is None
                else self.cas.load(
                    service_result.after_packet,
                    EvidencePacket,
                    object_kind="phase5_after_packet",
                )
            )
            certificate = after_projection.construction_certificate
            inventory = after_projection.pre_query_inventory
            if (
                after_projection.condition is not ConditionName.C2_LLM_QUERY
                or after_projection.context_hash != request.instruction.after_context.content_hash
                or after_projection.packet_hash != after_packet.content_hash
                or after_projection.snapshot_hash != request.packet.snapshot_hash
                or certificate is None
                or inventory is None
                or inventory.recorded_at >= certificate.query_revealed_at
                or certificate.query_revealed_at < request.instruction.revision.created_at
                or certificate.completed_at > references[-1].completed_at
                or after_projection.parent_projection_ref is not None
            ):
                raise Phase5AdapterError("feedback C2 output is not a fresh post-revision build")
        failure_hash = None
        if service_result.failure_artifact is not None:
            self.cas.bytes(service_result.failure_artifact)
            failure_hash = service_result.failure_artifact.artifact_hash
        started_at = references[0].started_at
        completed_at = references[-1].completed_at
        if started_at <= request.instruction.revision.created_at:
            raise Phase5AdapterError("feedback model call did not start after revision reveal")
        receipt = FeedbackRegenerationReceipt(
            ledger_calls=references,
            attempt_status=service_result.attempt_status,
            instruction_hash=request.instruction.content_hash,
            before_projection_hash=request.before_projection_hash,
            after_projection_hash=(
                None if after_projection is None else after_projection.content_hash
            ),
            failure_artifact_hash=failure_hash,
            seed=request.seed,
            allocated_gpu_seconds=allocated,
            started_at=started_at,
            completed_at=completed_at,
        )
        if service_result.resolved_at < completed_at:
            raise Phase5AdapterError("feedback resolution predates its final model call")
        latency = (completed_at - started_at).total_seconds()
        if recovering and service_result.checked_at < service_result.resolved_at:
            raise Phase5AdapterError("recovered feedback replay chronology changed")
        return C2RegenerationResult(
            request_hash=request.content_hash,
            after_projection=after_projection,
            after_packet=(
                None if after_projection is None or after_packet == request.packet else after_packet
            ),
            receipt=receipt,
            resolver_hash=service_result.resolver_hash,
            resolved_at=service_result.resolved_at,
            checked_at=service_result.checked_at,
            latency_seconds=latency,
        )

    def _ledger_references(
        self,
        request: C2RegenerationRequest,
        service_result: Phase5OwnedServiceResult,
    ) -> tuple[FeedbackLedgerCallReference, ...]:
        references: list[FeedbackLedgerCallReference] = []
        events = {item.event_id: item for item in self.cas.artifacts.ledger.gpu_events()}
        base_hash: str | None = None
        base_attempt_id: str | None = None
        for index, model_call_id in enumerate(service_result.model_call_ids):
            try:
                model_call = self.cas.artifacts.ledger.get_model_call(model_call_id)
                event = events[model_call.gpu_event_id]
                attempt = self.cas.artifacts.ledger.attempt_lineage(model_call.attempt_id)[-1]
            except KeyError as error:
                raise Phase5AdapterError("owned service omitted a model/GPU ledger row") from error
            is_base = index == 0
            expected_attempt = AttemptKind.BASE if is_base else AttemptKind.REPAIR
            expected_role = ModelCallRole.QUERY_TIME if is_base else ModelCallRole.REPAIR
            expected_retry = RetryClass.STANDARD if is_base else RetryClass.SHORT
            if (
                attempt.attempt_kind is not expected_attempt
                or model_call.call_role is not expected_role
                or model_call.retry_class is not expected_retry
                or model_call.backend is not ModelBackend.VLLM_GPU
                or model_call.model_manifest_hash != self._identity.model_manifest_hash
                or model_call.decoding_manifest_hash != self._identity.decoding_manifest_hash
                or attempt.seed != request.seed
                or model_call.job_id != attempt.job_id
                or event.job_id != attempt.job_id
                or event.attempt_id != attempt.attempt_id
                or event.allocated_microseconds != model_call.allocated_gpu_microseconds
                or event.succeeded is not model_call.successful
                or (is_base and model_call.request_hash != request.content_hash)
                or (not is_base and attempt.parent_attempt_id != base_attempt_id)
            ):
                raise Phase5AdapterError("owned service model-call lineage changed")
            started_at = _ledger_datetime(event.started_at)
            completed_at = _ledger_datetime(event.ended_at)
            reference = FeedbackLedgerCallReference(
                model_call_id=model_call.model_call_id,
                model_call_record_hash=feedback_model_call_record_hash(model_call),
                gpu_event_id=event.event_id,
                gpu_event_record_hash=feedback_gpu_event_record_hash(event),
                attempt_id=attempt.attempt_id,
                attempt_kind="base" if is_base else "repair",
                call_role="query_time" if is_base else "repair",
                retry_class="standard" if is_base else "short",
                request_hash=model_call.request_hash,
                parent_model_call_record_hash=None if is_base else base_hash,
                allocated_gpu_seconds=model_call.allocated_gpu_microseconds / 1_000_000,
                started_at=started_at,
                completed_at=completed_at,
                successful=model_call.successful,
            )
            references.append(reference)
            if is_base:
                base_hash = reference.model_call_record_hash
                base_attempt_id = attempt.attempt_id
        expected_success = service_result.attempt_status in {
            FeedbackAttemptStatus.SUCCEEDED,
            FeedbackAttemptStatus.INVALID,
        }
        if references[-1].successful is not expected_success:
            raise Phase5AdapterError("owned service status differs from final model call")
        return tuple(references)


def build_metered_phase5_adapter(
    *,
    service: AlreadyOwnedPhase5C2Service,
    artifacts: ArtifactStore,
) -> MeteredPhase5C2RegenerationAdapter:
    """Factory for a service that was loaded and leased by another controller."""

    if not isinstance(service, AlreadyOwnedPhase5C2Service):
        raise Phase5AdapterError("service does not implement the lifecycle-free Phase 5 lease")
    forbidden = ("start", "load", "shutdown", "terminate", "close_service")
    if any(callable(getattr(service, name, None)) for name in forbidden):
        raise Phase5AdapterError("Phase 5 service lease exposes forbidden lifecycle control")
    return MeteredPhase5C2RegenerationAdapter(
        service=service,
        cas=RestrictedPhase5CAS(artifacts),
    )


__all__ = [
    "AlreadyOwnedPhase5C2Service",
    "MeteredPhase5C2RegenerationAdapter",
    "Phase5AdapterError",
    "Phase5CASReference",
    "Phase5EpisodeSource",
    "Phase5HeldOutProjectionExport",
    "Phase5InputEpisodeLineage",
    "Phase5InputMaterializationReceipt",
    "Phase5MaterializationError",
    "Phase5MaterializationSourceManifest",
    "Phase5OwnedServiceIdentity",
    "Phase5OwnedServiceResult",
    "Phase5ScorerBindingAuthorization",
    "Phase5SourceModelCallBinding",
    "RestrictedPhase5CAS",
    "build_metered_phase5_adapter",
    "load_phase5_materialization_receipt",
    "load_phase5_materialization_source",
    "materialize_phase5_input_manifest",
    "materialize_phase5_inputs_to_directory",
    "persist_phase5_record",
    "phase5_record_reference",
    "validate_phase5_materialization_receipt",
    "validate_phase5_source_model_calls",
]
