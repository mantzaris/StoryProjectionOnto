"""Restricted production compiler for the 18 scripted Phase 5 scores.

This module is the only Phase 5 component that resolves known-answer CAS
objects.  It consumes a closed feedback journal and never imports or invokes a
model adapter.  Researcher traces are intentionally excluded from gold scoring.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    FeedbackResolutionStatus,
    ImmutableRecord,
    OntologyProjection,
    ReleaseClass,
    Sha256Digest,
)
from story_projection_onto.feedback_runtime import (
    FeedbackAttemptStatus,
    FeedbackStudyExecutionManifest,
    ScriptedRevisionFreeze,
)
from story_projection_onto.metrics.adapters import (
    projection_is_content_bearing,
    projection_is_structurally_valid,
)
from story_projection_onto.metrics.alignment import (
    GroundingStatus,
    audit_qualified_assertion_grounding,
    prediction_records,
    score_alignment,
)
from story_projection_onto.metrics.feedback import FeedbackScore, score_scripted_feedback
from story_projection_onto.metrics.rare import RarePivotalScore, score_rare_pivotal
from story_projection_onto.phase5_execution import (
    C2RegenerationRequest,
    C2RegenerationResult,
    FeedbackExecutionRecord,
    Phase5ExecutionInputManifest,
    Phase5JournalIndex,
    load_phase5_input_manifest,
)
from story_projection_onto.phase5_production import (
    Phase5ScorerBindingAuthorization,
    RestrictedPhase5CAS,
    load_phase5_materialization_source,
)
from story_projection_onto.phase5_sources import (
    Phase5GoldProjectionSource,
    Phase5KnownAnswerEntry,
    Phase5KnownAnswerSourceManifest,
    Phase5RequiredTargetChange,
    load_phase5_known_answer_source,
)
from story_projection_onto.store import ArtifactStore, ReadOnlyArtifactStore
from story_projection_onto.ui import VisualizationDiff

DEFAULT_PHASE5_SCORING_ROOT = Path("artifacts/restricted/phase5_feedback_scoring")
SESSION_FILE_NAME = "scoring_session.json"
METRICS_FILE_NAME = "scripted_feedback_metrics.json"
RECEIPT_FILE_NAME = "scoring_receipt.json"


class Phase5FeedbackScoringError(RuntimeError):
    """A scorer-only input or closed feedback artifact failed replay."""


class Phase5ProjectionGoldScore(ImmutableRecord):
    projection_hash: Sha256Digest
    gold_projection_hash: Sha256Digest
    gold_source_hash: Sha256Digest
    scorer_plan_hash: Sha256Digest
    alignment_plan_hash: Sha256Digest
    strict_qualified_assertion_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    rare_pivotal_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    strict_match_target_ids: tuple[str, ...]
    rare_score_hash: Sha256Digest
    structurally_valid: bool
    content_bearing: bool


class Phase5ScriptedFeedbackMetricRecord(ImmutableRecord):
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    episode_id: str = Field(min_length=1)
    condition: ConditionName
    scorer_binding_key: str = Field(min_length=1)
    execution_record_hash: Sha256Digest
    execution_hash: Sha256Digest
    instruction_hash: Sha256Digest
    before_projection_hash: Sha256Digest
    after_projection_hash: Sha256Digest | None
    c2_attempt_status: FeedbackAttemptStatus | None = None
    itt_zero_endpoint_applied: bool = False
    before_gold_projection_hash: Sha256Digest
    after_gold_projection_hash: Sha256Digest
    before_gold_source_artifact_hash: Sha256Digest
    after_gold_source_artifact_hash: Sha256Digest
    required_target_change_hash: Sha256Digest
    required_target_source_artifact_hash: Sha256Digest
    diff_hash: Sha256Digest | None
    before_gold_score: Phase5ProjectionGoldScore | None
    after_gold_score: Phase5ProjectionGoldScore | None
    before_strict_f1_endpoint: float | None = Field(default=None, ge=0.0, le=1.0)
    after_strict_f1_endpoint: float | None = Field(default=None, ge=0.0, le=1.0)
    before_rare_pivotal_endpoint: float | None = Field(default=None, ge=0.0, le=1.0)
    after_rare_pivotal_endpoint: float | None = Field(default=None, ge=0.0, le=1.0)
    score: FeedbackScore
    compiled_at: AwareDatetime
    model_service_called: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_metric_lineage(self) -> Self:
        if self.condition not in {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        }:
            raise ValueError("Phase 5 scripted metrics support only C0, C1, and C2")
        if self.score.episode_id != f"{self.episode_id}--{self.condition.value.lower()}":
            raise ValueError("feedback score ID does not bind its episode and condition")
        gold_scores = (self.before_gold_score, self.after_gold_score)
        endpoints = (
            self.before_strict_f1_endpoint,
            self.after_strict_f1_endpoint,
            self.before_rare_pivotal_endpoint,
            self.after_rare_pivotal_endpoint,
        )
        unsuccessful_c2 = self.c2_attempt_status in {
            FeedbackAttemptStatus.INVALID,
            FeedbackAttemptStatus.FAILED,
            FeedbackAttemptStatus.TIMED_OUT,
        }
        if self.condition is ConditionName.C2_LLM_QUERY:
            if self.c2_attempt_status is None:
                raise ValueError("C2 feedback metrics require the metered attempt status")
        elif self.c2_attempt_status is not None:
            raise ValueError("CPU feedback metrics cannot claim a C2 attempt status")
        if self.score.capability_limited:
            if (
                self.after_projection_hash is not None
                or self.diff_hash is not None
                or self.itt_zero_endpoint_applied
            ):
                raise ValueError("capability-limited score cannot claim an after projection")
            if any(item is not None for item in gold_scores):
                raise ValueError("capability-limited gold values must remain NA")
            if any(item is not None for item in endpoints):
                raise ValueError("capability-limited endpoint values must remain NA")
        elif unsuccessful_c2:
            if (
                self.after_projection_hash is not None
                or self.diff_hash is not None
                or self.before_gold_score is None
                or self.after_gold_score is not None
                or not self.itt_zero_endpoint_applied
                or self.after_strict_f1_endpoint != 0.0
                or self.after_rare_pivotal_endpoint != 0.0
                or self.before_strict_f1_endpoint
                != self.before_gold_score.strict_qualified_assertion_f1
                or self.before_rare_pivotal_endpoint
                != self.before_gold_score.rare_pivotal_recall
            ):
                raise ValueError(
                    "unsuccessful C2 ITT rows require a scored before projection, "
                    "a zero endpoint, and no fabricated after projection"
                )
        elif (
            self.after_projection_hash is None
            or self.diff_hash is None
            or any(item is None for item in gold_scores)
        ):
            raise ValueError("resolved score requires both gold evaluations and graph diff")
        elif self.itt_zero_endpoint_applied:
            raise ValueError("resolved rows cannot claim an ITT zero endpoint")
        elif self.before_gold_score is not None and self.after_gold_score is not None:
            if endpoints != (
                self.before_gold_score.strict_qualified_assertion_f1,
                self.after_gold_score.strict_qualified_assertion_f1,
                self.before_gold_score.rare_pivotal_recall,
                self.after_gold_score.rare_pivotal_recall,
            ):
                raise ValueError("resolved endpoint values differ from their gold scores")
        if (
            self.score.strict_f1_change is not None
            and self.before_strict_f1_endpoint is not None
            and self.after_strict_f1_endpoint is not None
            and abs(
                self.score.strict_f1_change
                - (
                    self.after_strict_f1_endpoint
                    - self.before_strict_f1_endpoint
                )
            )
            > 1e-12
        ):
            raise ValueError("strict F1 change differs from recorded endpoints")
        return self


class Phase5ScriptedFeedbackMetrics(ImmutableRecord):
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    metrics_id: str = Field(min_length=1)
    protocol_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    known_answer_source_hash: Sha256Digest
    input_manifest_hash: Sha256Digest
    feedback_manifest_hash: Sha256Digest
    execution_index_hash: Sha256Digest
    scorer_binding_authorization_hash: Sha256Digest
    records: tuple[Phase5ScriptedFeedbackMetricRecord, ...]
    generated_at: AwareDatetime
    researcher_trace_gold_fields: Literal["NA"] = "NA"
    model_service_called: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_eighteen_scores(self) -> Self:
        pairs = tuple((item.episode_id, item.condition) for item in self.records)
        if (
            len(pairs) != 18
            or len(set(pairs)) != 18
            or Counter(condition for _, condition in pairs)
            != Counter(
                {
                    ConditionName.C0_CLASSICAL_PRE: 6,
                    ConditionName.C1_LLM_PRE: 6,
                    ConditionName.C2_LLM_QUERY: 6,
                }
            )
        ):
            raise ValueError("scripted feedback metrics require six scores per condition")
        return self


class Phase5FeedbackScoringSession(ImmutableRecord):
    session_id: str = Field(min_length=1)
    source_manifest_hash: Sha256Digest
    known_answer_source_hash: Sha256Digest
    input_manifest_hash: Sha256Digest
    execution_index_hash: Sha256Digest
    compiler_source_file_sha256: Sha256Digest
    compiled_at: AwareDatetime
    scorer_only: Literal[True] = True
    model_service_called: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


class Phase5FeedbackScoringReceipt(ImmutableRecord):
    receipt_id: str = Field(min_length=1)
    session_hash: Sha256Digest
    metrics_hash: Sha256Digest
    metrics_file_sha256: Sha256Digest
    source_manifest_hash: Sha256Digest
    known_answer_source_hash: Sha256Digest
    input_manifest_hash: Sha256Digest
    feedback_manifest_hash: Sha256Digest
    execution_index_hash: Sha256Digest
    scorer_binding_authorization_hash: Sha256Digest
    execution_record_hashes: tuple[Sha256Digest, ...]
    gold_source_artifact_hashes: tuple[Sha256Digest, ...]
    target_source_artifact_hashes: tuple[Sha256Digest, ...]
    compiled_at: AwareDatetime
    score_count: Literal[18] = 18
    researcher_trace_gold_fields: Literal["NA"] = "NA"
    gold_objects_model_visible: Literal[False] = False
    model_service_called: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_receipt_inventory(self) -> Self:
        if len(self.execution_record_hashes) != 18 or len(
            set(self.execution_record_hashes)
        ) != 18:
            raise ValueError("scoring receipt requires 18 unique execution records")
        if not self.gold_source_artifact_hashes or not self.target_source_artifact_hashes:
            raise ValueError("scoring receipt requires resolvable gold and target sources")
        for values in (
            self.gold_source_artifact_hashes,
            self.target_source_artifact_hashes,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("scoring source artifact hashes must be canonical")
        return self


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5FeedbackScoringError(f"symlinked scorer path is forbidden: {path}")
        if current.parent == current:
            return
        current = current.parent


def _canonical_bytes(value: ImmutableRecord) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _read(path: Path, model_type):
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5FeedbackScoringError(f"missing scorer input: {path}")
    raw = path.read_bytes()
    if len(raw) > 100 * 1024 * 1024:
        raise Phase5FeedbackScoringError(f"scorer input exceeds 100 MiB: {path}")
    try:
        value = model_type.model_validate_json(raw)
    except Exception as error:
        raise Phase5FeedbackScoringError(
            f"invalid {model_type.__name__}: {error}"
        ) from error
    if raw != _canonical_bytes(value):
        raise Phase5FeedbackScoringError(f"noncanonical scorer input: {path}")
    return value


def _projection_gold_score(
    projection: OntologyProjection,
    source: Phase5GoldProjectionSource,
) -> Phase5ProjectionGoldScore:
    plan = source.scorer_plan
    valid_evidence = frozenset(plan.valid_evidence_ids)
    provisional = {
        item.assertion_id: (
            GroundingStatus.SUPPORTED
            if valid_evidence.intersection(item.evidence_ids)
            else GroundingStatus.UNSUPPORTED
        )
        for item in projection.instance_graph.assertions
    }
    predicted_nodes, predicted_assertions, _ = prediction_records(
        projection,
        predicate_aliases=dict(plan.predicate_aliases),
        valid_evidence_ids=valid_evidence,
        grounding_by_assertion_id=provisional,
        plan=plan.alignment_plan,
    )
    statuses = dict(
        audit_qualified_assertion_grounding(
            plan=plan.alignment_plan,
            predicted_assertions=predicted_assertions,
        )
    )
    predicted_nodes, predicted_assertions, _ = prediction_records(
        projection,
        predicate_aliases=dict(plan.predicate_aliases),
        valid_evidence_ids=valid_evidence,
        grounding_by_assertion_id=statuses,
        plan=plan.alignment_plan,
    )
    structurally_valid = projection_is_structurally_valid(projection)
    content_bearing = projection_is_content_bearing(projection)
    semantic_output_valid = structurally_valid and content_bearing
    alignment = score_alignment(
        plan=plan.alignment_plan,
        predicted_nodes=predicted_nodes,
        predicted_assertions=predicted_assertions,
        invalid_semantic_output=not semantic_output_valid,
    )
    matched = frozenset(item.target_id for item in alignment.strict_assertion_matches)
    rare: RarePivotalScore = score_rare_pivotal(
        annotations=plan.rare_annotations,
        strictly_matched_assertion_target_ids=matched,
        invalid_semantic_output=not semantic_output_valid,
    )
    return Phase5ProjectionGoldScore(
        projection_hash=projection.content_hash,
        gold_projection_hash=source.gold_projection.content_hash,
        gold_source_hash=source.content_hash,
        scorer_plan_hash=plan.content_hash,
        alignment_plan_hash=plan.alignment_plan.content_hash,
        strict_qualified_assertion_f1=alignment.strict_assertion_score.f1,
        rare_pivotal_recall=rare.qualified_assertion_recall.value,
        strict_match_target_ids=tuple(sorted(matched)),
        rare_score_hash=rare.content_hash,
        structurally_valid=structurally_valid,
        content_bearing=content_bearing,
    )


def _change_inputs(
    diff: VisualizationDiff,
    target: Phase5RequiredTargetChange,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    by_signature = {
        (item.kind, item.object_kind, item.anchor_ids): item for item in target.targets
    }
    changed: list[str] = []
    realized: list[str] = []
    unsupported: list[str] = []
    for change in diff.changes:
        match = by_signature.get((change.kind, change.object_kind, change.anchor_ids))
        change_id = (
            match.target_change_id
            if match is not None
            else f"unregistered-change-{change.content_hash}"
        )
        if change_id in changed:
            raise Phase5FeedbackScoringError(
                "one target rule ambiguously matches multiple observed changes"
            )
        changed.append(change_id)
        if match is None:
            unsupported.append(change_id)
        else:
            realized.append(change_id)
    return (
        tuple(sorted(item.target_change_id for item in target.targets if item.required)),
        tuple(sorted(realized)),
        tuple(sorted(changed)),
        tuple(sorted(unsupported)),
    )


def _verify_journal(
    root: Path,
) -> tuple[
    Phase5ExecutionInputManifest,
    FeedbackStudyExecutionManifest,
    Phase5JournalIndex,
    dict[tuple[str, ConditionName], FeedbackExecutionRecord],
]:
    root = root.resolve(strict=True)
    inputs = load_phase5_input_manifest(root / "input_manifest.json")
    manifest = _read(root / "feedback_manifest.json", FeedbackStudyExecutionManifest)
    index = _read(root / "execution_index.json", Phase5JournalIndex)
    if (
        index.input_manifest_hash != inputs.content_hash
        or index.feedback_manifest_hash != manifest.content_hash
        or index.protocol_hash != inputs.protocol_hash
    ):
        raise Phase5FeedbackScoringError("feedback journal index lineage changed")
    model_by_kind = {
        "input_manifest": Phase5ExecutionInputManifest,
        "script_freeze": ScriptedRevisionFreeze,
        "c2_call_slot": C2RegenerationRequest,
        "c2_result": C2RegenerationResult,
        "execution_record": FeedbackExecutionRecord,
        "feedback_manifest": FeedbackStudyExecutionManifest,
    }
    records: dict[tuple[str, ConditionName], FeedbackExecutionRecord] = {}
    for binding in index.artifact_bindings:
        path = root / binding.relative_path
        try:
            if not path.resolve(strict=True).is_relative_to(root):
                raise Phase5FeedbackScoringError("feedback artifact escapes its journal root")
        except OSError as error:
            raise Phase5FeedbackScoringError("feedback journal artifact is absent") from error
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding.file_sha256:
            raise Phase5FeedbackScoringError("feedback journal file hash changed")
        value = _read(path, model_by_kind[binding.artifact_kind])
        if value.content_hash != binding.logical_content_hash:
            raise Phase5FeedbackScoringError("feedback journal logical hash changed")
        if isinstance(value, FeedbackExecutionRecord):
            key = (value.execution.episode_id, value.execution.condition)
            if key in records:
                raise Phase5FeedbackScoringError("feedback journal repeats an execution")
            records[key] = value
    if (
        len(records) != 21
        or tuple(item.content_hash for item in records.values())
        != tuple(
            item.logical_content_hash
            for item in index.artifact_bindings
            if item.artifact_kind == "execution_record"
        )
        or Counter(item.execution.content_hash for item in records.values())
        != Counter(item.content_hash for item in manifest.executions)
    ):
        raise Phase5FeedbackScoringError("feedback execution inventory changed")
    return inputs, manifest, index, records


def _resolved_projections(
    *,
    root: Path,
    inputs: Phase5ExecutionInputManifest,
    record: FeedbackExecutionRecord,
) -> tuple[OntologyProjection, OntologyProjection | None]:
    execution = record.execution
    script = next(
        (item for item in inputs.scripted_inputs if item.episode_id == execution.episode_id),
        None,
    )
    if script is None:
        raise Phase5FeedbackScoringError("scorer attempted to open a researcher trace")
    if execution.condition in {
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
    }:
        cpu = next(
            item for item in script.cpu_inputs if item.condition is execution.condition
        )
        before, after = cpu.before_projection, cpu.after_projection
    else:
        before = script.c2_before_projection
        result = _read(
            root / "results" / f"{execution.episode_id}.json",
            C2RegenerationResult,
        )
        if result.content_hash != record.c2_result_hash:
            raise Phase5FeedbackScoringError("C2 result hash differs from execution record")
        after = result.after_projection
    if before.content_hash != execution.before_bundle.projection_hash:
        raise Phase5FeedbackScoringError("before projection differs from execution bundle")
    observed_after = None if after is None else after.content_hash
    expected_after = (
        None if execution.after_bundle is None else execution.after_bundle.projection_hash
    )
    if observed_after != expected_after:
        raise Phase5FeedbackScoringError("after projection differs from execution bundle")
    return before, after


def _load_known_answer_objects(
    *,
    entry: Phase5KnownAnswerEntry,
    known: Phase5KnownAnswerSourceManifest,
    instruction_hash: str,
    before_context_id: str,
    after_context_id: str,
    cas: RestrictedPhase5CAS,
) -> tuple[
    Phase5GoldProjectionSource,
    Phase5GoldProjectionSource,
    Phase5RequiredTargetChange,
]:
    before = cast(
        Phase5GoldProjectionSource,
        cas.load(
            entry.before_gold_projection_source,
            Phase5GoldProjectionSource,
            object_kind="feedback_gold_projection_source",
        ),
    )
    after = cast(
        Phase5GoldProjectionSource,
        cas.load(
            entry.after_gold_projection_source,
            Phase5GoldProjectionSource,
            object_kind="feedback_gold_projection_source",
        ),
    )
    target = cast(
        Phase5RequiredTargetChange,
        cas.load(
            entry.required_target_change_source,
            Phase5RequiredTargetChange,
            object_kind="feedback_required_target_change",
        ),
    )
    if (
        before.episode_id != entry.episode_id
        or after.episode_id != entry.episode_id
        or before.role != "before"
        or after.role != "after"
        or before.gold_projection.content_hash != entry.before_gold_projection_hash
        or after.gold_projection.content_hash != entry.after_gold_projection_hash
        or before.gold_projection.query_id != before_context_id
        or after.gold_projection.query_id != after_context_id
        or target.content_hash != entry.required_target_change_hash
        or target.episode_id != entry.episode_id
        or target.context_id != entry.context_id
        or target.scorer_binding_key != entry.scorer_binding_key
        or target.instruction_hash != instruction_hash
        or target.before_gold_projection_hash != entry.before_gold_projection_hash
        or target.after_gold_projection_hash != entry.after_gold_projection_hash
        or any(
            item.review_completion_manifest_hash
            != known.review_completion_manifest_hash
            or item.final_reviewed_seal_hash != known.final_reviewed_seal_hash
            or item.prepared_at > known.authored_at
            for item in (before, after)
        )
        or target.review_completion_manifest_hash
        != known.review_completion_manifest_hash
        or target.final_reviewed_seal_hash != known.final_reviewed_seal_hash
        or target.authored_at > known.authored_at
    ):
        raise Phase5FeedbackScoringError("known-answer scorer CAS lineage changed")
    return before, after, target


def prepare_phase5_feedback_scoring(
    *,
    repository: Path,
    restricted_root: Path,
    source_manifest_path: Path,
    known_answer_source_path: Path,
    feedback_journal_root: Path,
    output_root: Path,
    artifacts: ArtifactStore | ReadOnlyArtifactStore,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[
    Phase5FeedbackScoringSession,
    Phase5ScriptedFeedbackMetrics,
    Phase5FeedbackScoringReceipt,
]:
    repository = repository.resolve(strict=True)
    restricted_root = restricted_root.resolve(strict=True)
    canonical_restricted = (repository / "artifacts/restricted").resolve(strict=True)
    if restricted_root != canonical_restricted:
        raise Phase5FeedbackScoringError(
            "restricted root is not the canonical repository artifacts/restricted root"
        )

    def restricted(path: Path, label: str) -> Path:
        path = path.absolute()
        _assert_no_symlink_chain(path)
        try:
            if not path.resolve(strict=path.exists()).is_relative_to(restricted_root):
                raise Phase5FeedbackScoringError(f"{label} is outside restricted storage")
        except OSError as error:
            raise Phase5FeedbackScoringError(f"cannot resolve {label}") from error
        return path

    source_path = restricted(source_manifest_path, "Phase 5 source manifest")
    known_path = restricted(known_answer_source_path, "known-answer source")
    journal_root = restricted(feedback_journal_root, "feedback journal")
    destination = restricted(output_root, "scoring output root")
    source = load_phase5_materialization_source(source_path)
    if source_path.read_bytes() != _canonical_bytes(source):
        raise Phase5FeedbackScoringError("Phase 5 source manifest is noncanonical")
    known = load_phase5_known_answer_source(known_path)
    inputs, feedback_manifest, index, records = _verify_journal(journal_root)
    cas = RestrictedPhase5CAS(artifacts)
    authorization = cast(
        Phase5ScorerBindingAuthorization,
        cas.load(
            source.scorer_binding_authorization,
            Phase5ScorerBindingAuthorization,
            object_kind="scorer_binding_authorization",
        ),
    )
    if (
        source.content_hash != inputs.source_manifest_hash
        or source.protocol_hash != inputs.protocol_hash
        or source.held_out_execution_manifest_hash
        != inputs.held_out_execution_manifest_hash
        or authorization.content_hash != inputs.scorer_binding_authorization_hash
        or authorization.known_answer_source_hash != known.content_hash
        or authorization.protocol_hash != inputs.protocol_hash
        or authorization.review_completion_manifest_hash
        != known.review_completion_manifest_hash
        or authorization.final_reviewed_seal_hash != known.final_reviewed_seal_hash
        or index.input_manifest_hash != inputs.content_hash
    ):
        raise Phase5FeedbackScoringError("Phase 5 scorer source lineage changed")

    session_path = destination / SESSION_FILE_NAME
    if session_path.exists():
        session = _read(session_path, Phase5FeedbackScoringSession)
        compiled_at = session.compiled_at
    else:
        compiled_at = clock()
        if compiled_at.tzinfo is None or compiled_at.utcoffset() is None:
            raise Phase5FeedbackScoringError("scoring clock must be timezone-aware")
        if compiled_at < index.completed_at:
            raise Phase5FeedbackScoringError("scoring timestamp predates feedback completion")
        session = Phase5FeedbackScoringSession(
            session_id=(
                "phase5-scoring-"
                + hashlib.sha256(
                    (
                        source.content_hash
                        + known.content_hash
                        + index.content_hash
                    ).encode("ascii")
                ).hexdigest()[:24]
            ),
            source_manifest_hash=source.content_hash,
            known_answer_source_hash=known.content_hash,
            input_manifest_hash=inputs.content_hash,
            execution_index_hash=index.content_hash,
            compiler_source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            compiled_at=compiled_at,
        )
    expected_session = session.model_copy(
        update={
            "source_manifest_hash": source.content_hash,
            "known_answer_source_hash": known.content_hash,
            "input_manifest_hash": inputs.content_hash,
            "execution_index_hash": index.content_hash,
            "compiler_source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    if expected_session != session:
        raise Phase5FeedbackScoringError("retained scoring session differs from inputs or code")

    known_by_id = {item.episode_id: item for item in known.answers}
    authorization_by_pair = {
        (item.episode_id, item.condition): item for item in authorization.bindings
    }
    metric_records: list[Phase5ScriptedFeedbackMetricRecord] = []
    gold_artifacts: set[str] = set()
    target_artifacts: set[str] = set()
    for script in inputs.scripted_inputs:
        try:
            entry = known_by_id[script.episode_id]
        except KeyError as error:
            raise Phase5FeedbackScoringError("script has no known-answer source") from error
        before_gold, after_gold, target = _load_known_answer_objects(
            entry=entry,
            known=known,
            instruction_hash=script.instruction.content_hash,
            before_context_id=script.instruction.before_context.context_id,
            after_context_id=script.instruction.after_context.context_id,
            cas=cas,
        )
        if target.action is not script.instruction.revision.action:
            raise Phase5FeedbackScoringError("target change action differs from script")
        if not set(before_gold.scorer_plan.valid_evidence_ids).issubset(
            script.packet.ordered_evidence_ids
        ) or not set(after_gold.scorer_plan.valid_evidence_ids).issubset(
            script.packet.ordered_evidence_ids
        ):
            raise Phase5FeedbackScoringError("gold scorer plan references evidence outside packet")
        gold_artifacts.update(
            {
                entry.before_gold_projection_source.artifact_hash,
                entry.after_gold_projection_source.artifact_hash,
            }
        )
        target_artifacts.add(entry.required_target_change_source.artifact_hash)
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        ):
            record = records.get((script.episode_id, condition))
            if record is None:
                raise Phase5FeedbackScoringError("missing scripted feedback execution")
            binding = authorization_by_pair.get((script.episode_id, condition))
            if binding is None or binding != record.scorer_binding or (
                binding.scorer_binding_key != entry.scorer_binding_key
                or binding.before_gold_projection_hash
                != entry.before_gold_projection_hash
                or binding.after_gold_projection_hash != entry.after_gold_projection_hash
                or binding.required_target_change_hash != entry.required_target_change_hash
            ):
                raise Phase5FeedbackScoringError("execution scorer binding changed")
            execution = record.execution
            before_projection, after_projection = _resolved_projections(
                root=journal_root,
                inputs=inputs,
                record=record,
            )
            c2_attempt_status = (
                None
                if execution.regeneration_receipt is None
                else execution.regeneration_receipt.attempt_status
            )
            capability_limited = (
                execution.resolution.status
                is FeedbackResolutionStatus.CAPABILITY_LIMITED
            )
            unsuccessful_c2 = c2_attempt_status in {
                FeedbackAttemptStatus.INVALID,
                FeedbackAttemptStatus.FAILED,
                FeedbackAttemptStatus.TIMED_OUT,
            }
            before_score = after_score = None
            required_ids = tuple(
                sorted(item.target_change_id for item in target.targets if item.required)
            )
            requested_scope_ids = tuple(
                sorted(
                    item.target_change_id
                    for item in target.targets
                    if item.within_requested_scope
                )
            )
            realized_ids: tuple[str, ...] = ()
            changed_ids: tuple[str, ...] = ()
            unsupported_ids: tuple[str, ...] = ()
            diff: VisualizationDiff | None = execution.diff
            before_f1: float | None = None
            after_f1: float | None = None
            before_rare: float | None = None
            after_rare: float | None = None
            if unsuccessful_c2:
                if after_projection is not None or diff is not None:
                    raise Phase5FeedbackScoringError(
                        "unsuccessful C2 ITT result retained an after projection"
                    )
                before_score = _projection_gold_score(before_projection, before_gold)
                before_f1 = before_score.strict_qualified_assertion_f1
                before_rare = before_score.rare_pivotal_recall
                if not after_gold.gold_projection.qualified_assertions:
                    raise Phase5FeedbackScoringError(
                        "registered unsuccessful-C2 ITT zero endpoint requires "
                        "nonempty after-context assertion gold"
                    )
                if before_f1 is None or before_rare is None:
                    raise Phase5FeedbackScoringError(
                        "unsuccessful-C2 ITT before endpoint is undefined"
                    )
                # Intention-to-treat: a requested query-time regeneration that
                # yields no valid projection scores zero against the nonempty
                # after-context gold.  No synthetic graph or graph diff is made.
                after_f1 = 0.0
                after_rare = 0.0
            elif not capability_limited:
                if after_projection is None or diff is None:
                    raise Phase5FeedbackScoringError("resolved execution lost its output or diff")
                before_score = _projection_gold_score(before_projection, before_gold)
                after_score = _projection_gold_score(after_projection, after_gold)
                required_ids, realized_ids, changed_ids, unsupported_ids = _change_inputs(
                    diff,
                    target,
                )
                if (
                    before_score.strict_qualified_assertion_f1 is None
                    or after_score.strict_qualified_assertion_f1 is None
                ):
                    raise Phase5FeedbackScoringError(
                        "resolved known-answer script has undefined strict F1"
                    )
                before_f1 = before_score.strict_qualified_assertion_f1
                after_f1 = after_score.strict_qualified_assertion_f1
                before_rare = before_score.rare_pivotal_recall
                after_rare = after_score.rare_pivotal_recall
            score = score_scripted_feedback(
                episode_id=f"{script.episode_id}--{condition.value.lower()}",
                before_strict_f1=before_f1,
                after_strict_f1=after_f1,
                required_target_change_ids=required_ids,
                realized_target_change_ids=realized_ids,
                changed_semantic_ids=changed_ids,
                requested_scope_ids=requested_scope_ids,
                unsupported_changed_ids=unsupported_ids,
                before_rare_pivotal_recall=before_rare,
                after_rare_pivotal_recall=after_rare,
                capability_limited=capability_limited,
                latency_seconds=execution.latency_seconds,
                replay_hash_success=(
                    False
                    if execution.replay is None
                    else execution.replay.replay_hash_success
                ),
            )
            metric_records.append(
                Phase5ScriptedFeedbackMetricRecord(
                    episode_id=script.episode_id,
                    condition=condition,
                    scorer_binding_key=entry.scorer_binding_key,
                    execution_record_hash=record.content_hash,
                    execution_hash=execution.content_hash,
                    instruction_hash=script.instruction.content_hash,
                    before_projection_hash=before_projection.content_hash,
                    after_projection_hash=(
                        None if after_projection is None else after_projection.content_hash
                    ),
                    c2_attempt_status=c2_attempt_status,
                    itt_zero_endpoint_applied=unsuccessful_c2,
                    before_gold_projection_hash=entry.before_gold_projection_hash,
                    after_gold_projection_hash=entry.after_gold_projection_hash,
                    before_gold_source_artifact_hash=(
                        entry.before_gold_projection_source.artifact_hash
                    ),
                    after_gold_source_artifact_hash=(
                        entry.after_gold_projection_source.artifact_hash
                    ),
                    required_target_change_hash=entry.required_target_change_hash,
                    required_target_source_artifact_hash=(
                        entry.required_target_change_source.artifact_hash
                    ),
                    diff_hash=None if diff is None else diff.content_hash,
                    before_gold_score=before_score,
                    after_gold_score=after_score,
                    before_strict_f1_endpoint=before_f1,
                    after_strict_f1_endpoint=after_f1,
                    before_rare_pivotal_endpoint=before_rare,
                    after_rare_pivotal_endpoint=after_rare,
                    score=score,
                    compiled_at=compiled_at,
                )
            )
    metric_records.sort(key=lambda item: (item.episode_id, item.condition.value))
    metrics = Phase5ScriptedFeedbackMetrics(
        metrics_id=f"phase5-scripted-feedback-{session.content_hash[:24]}",
        protocol_hash=inputs.protocol_hash,
        source_manifest_hash=source.content_hash,
        known_answer_source_hash=known.content_hash,
        input_manifest_hash=inputs.content_hash,
        feedback_manifest_hash=feedback_manifest.content_hash,
        execution_index_hash=index.content_hash,
        scorer_binding_authorization_hash=authorization.content_hash,
        records=tuple(metric_records),
        generated_at=compiled_at,
    )
    metrics_bytes = _canonical_bytes(metrics)
    receipt = Phase5FeedbackScoringReceipt(
        receipt_id=f"phase5-scoring-receipt-{metrics.content_hash[:24]}",
        session_hash=session.content_hash,
        metrics_hash=metrics.content_hash,
        metrics_file_sha256=hashlib.sha256(metrics_bytes).hexdigest(),
        source_manifest_hash=source.content_hash,
        known_answer_source_hash=known.content_hash,
        input_manifest_hash=inputs.content_hash,
        feedback_manifest_hash=feedback_manifest.content_hash,
        execution_index_hash=index.content_hash,
        scorer_binding_authorization_hash=authorization.content_hash,
        execution_record_hashes=tuple(item.execution_record_hash for item in metric_records),
        gold_source_artifact_hashes=tuple(sorted(gold_artifacts)),
        target_source_artifact_hashes=tuple(sorted(target_artifacts)),
        compiled_at=compiled_at,
    )
    return session, metrics, receipt


def _append_exact(path: Path, value: ImmutableRecord) -> bool:
    payload = _canonical_bytes(value)
    _assert_no_symlink_chain(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise Phase5FeedbackScoringError(f"append-only scorer output drift: {path.name}")
        return False
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
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
            if not path.is_file() or path.read_bytes() != payload:
                raise Phase5FeedbackScoringError(
                    f"concurrent scorer output drift: {path.name}"
                ) from None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def materialize_phase5_feedback_scoring(
    *,
    output_root: Path,
    restricted_root: Path,
    session: Phase5FeedbackScoringSession,
    metrics: Phase5ScriptedFeedbackMetrics,
    receipt: Phase5FeedbackScoringReceipt,
) -> bool:
    restricted_root = restricted_root.resolve(strict=True)
    output_root = output_root.absolute()
    _assert_no_symlink_chain(output_root)
    try:
        if not output_root.resolve(strict=output_root.exists()).is_relative_to(restricted_root):
            raise Phase5FeedbackScoringError("scoring output root is outside restricted storage")
    except OSError as error:
        raise Phase5FeedbackScoringError("cannot resolve scoring output root") from error
    allowed = {SESSION_FILE_NAME, METRICS_FILE_NAME, RECEIPT_FILE_NAME}
    if output_root.exists():
        if any(item.is_symlink() for item in output_root.iterdir()):
            raise Phase5FeedbackScoringError("symlinked scorer output artifact")
        extras = {
            item.name
            for item in output_root.iterdir()
            if not (
                (item.is_file() and item.name in allowed)
                or (
                    item.is_file()
                    and item.name.startswith(".")
                    and item.name.endswith(".tmp")
                )
            )
        }
        if extras:
            raise Phase5FeedbackScoringError(
                f"unexpected scorer output artifact: {sorted(extras)[0]}"
            )
    created = not (output_root / RECEIPT_FILE_NAME).exists()
    _append_exact(output_root / SESSION_FILE_NAME, session)
    _append_exact(output_root / METRICS_FILE_NAME, metrics)
    _append_exact(output_root / RECEIPT_FILE_NAME, receipt)
    return created


__all__ = [
    "DEFAULT_PHASE5_SCORING_ROOT",
    "METRICS_FILE_NAME",
    "RECEIPT_FILE_NAME",
    "SESSION_FILE_NAME",
    "Phase5FeedbackScoringError",
    "Phase5FeedbackScoringReceipt",
    "Phase5FeedbackScoringSession",
    "Phase5ProjectionGoldScore",
    "Phase5ScriptedFeedbackMetricRecord",
    "Phase5ScriptedFeedbackMetrics",
    "materialize_phase5_feedback_scoring",
    "prepare_phase5_feedback_scoring",
]
