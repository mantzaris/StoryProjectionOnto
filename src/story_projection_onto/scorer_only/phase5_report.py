"""Canonical public table compiler for the registered Phase 5 feedback study.

The compiler is deliberately downstream of the restricted scorer.  It opens only
the closed Phase 5 journal and the scorer's immutable metric bundle, verifies their
complete lineage, and reduces the 18 scripted condition scores plus three C2
researcher traces to one public-safe table.  Researcher traces are reconstructed
from their execution records and never receive gold-dependent values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    ConditionName,
    FeedbackAction,
    FeedbackResolutionStatus,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.metrics.common import RateResult
from story_projection_onto.metrics.feedback import (
    FeedbackEpisodeKind as FeedbackMetricEpisodeKind,
)
from story_projection_onto.metrics.feedback import (
    FeedbackScore,
    record_researcher_trace,
)
from story_projection_onto.phase5_execution import FeedbackExecutionRecord
from story_projection_onto.scorer_only.phase5_feedback import (
    METRICS_FILE_NAME,
    RECEIPT_FILE_NAME,
    SESSION_FILE_NAME,
    Phase5FeedbackScoringError,
    Phase5FeedbackScoringReceipt,
    Phase5FeedbackScoringSession,
    Phase5ScriptedFeedbackMetricRecord,
    Phase5ScriptedFeedbackMetrics,
    _verify_journal,
)
from story_projection_onto.ui import FeedbackEpisodeKind as FeedbackExecutionKind

DEFAULT_PHASE5_REPORT_ROOT = Path("artifacts/public/phase5_feedback_report")
FEEDBACK_TABLE_FILE_NAME = "feedback.csv"
FEEDBACK_TABLE_RECEIPT_FILE_NAME = "feedback_table_receipt.json"

SCRIPTED_METRICS = (
    "capability_limited_rate",
    "edit_locality",
    "latency_seconds",
    "post_minus_pre_strict_qualified_assertion_f1",
    "rare_pivotal_recall_change",
    "replay_success_rate",
    "resolution_success_rate",
    "semantic_change_count",
    "target_change_recall",
    "unsupported_change_count",
)
TRACE_METRICS = (
    "capability_limited_rate",
    "latency_seconds",
    "replay_success_rate",
    "resolution_success_rate",
    "semantic_change_count",
)

FeedbackMetricName = Literal[
    "capability_limited_rate",
    "edit_locality",
    "latency_seconds",
    "post_minus_pre_strict_qualified_assertion_f1",
    "rare_pivotal_recall_change",
    "replay_success_rate",
    "resolution_success_rate",
    "semantic_change_count",
    "target_change_recall",
    "unsupported_change_count",
]
FeedbackMetricUnit = Literal[
    "mean_difference",
    "mean_seconds",
    "pooled_proportion",
    "total_count",
]
FeedbackMetricStatus = Literal[
    "complete",
    "not_applicable",
    "partial_defined_panel",
]

FEEDBACK_TABLE_COLUMNS = (
    "episode_class",
    "condition",
    "action",
    "metric",
    "value",
    "unit",
    "episode_count",
    "capability_limited_count",
    "gold_applicability",
    "status",
    "defined_episode_count",
    "numerator",
    "denominator",
)


class Phase5FeedbackReportError(RuntimeError):
    """The public Phase 5 table could not be reproduced from immutable inputs."""


class Phase5FeedbackTableRow(ImmutableRecord):
    """One action-by-condition descriptive feedback estimate with denominators."""

    episode_class: FeedbackMetricEpisodeKind
    condition: ConditionName
    action: FeedbackAction
    metric: FeedbackMetricName
    value: float | None = None
    unit: FeedbackMetricUnit
    episode_count: int = Field(ge=1, le=3)
    capability_limited_count: int = Field(ge=0, le=3)
    gold_applicability: Literal["applicable", "NA"]
    status: FeedbackMetricStatus
    defined_episode_count: int = Field(ge=0, le=3)
    numerator: float | None = None
    denominator: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def row_is_an_exact_registered_reduction(self) -> Self:
        if self.capability_limited_count > self.episode_count:
            raise ValueError("capability-limited count exceeds the episode count")
        if self.defined_episode_count > self.episode_count:
            raise ValueError("defined-episode count exceeds the episode count")
        if self.episode_class is FeedbackMetricEpisodeKind.RESEARCHER_TRACE:
            if (
                self.condition is not ConditionName.C2_LLM_QUERY
                or self.gold_applicability != "NA"
                or self.metric not in TRACE_METRICS
            ):
                raise ValueError("researcher-trace rows must remain C2-only and gold-NA")
        elif (
            self.condition
            not in {
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C1_LLM_PRE,
                ConditionName.C2_LLM_QUERY,
            }
            or self.gold_applicability != "applicable"
            or self.metric not in SCRIPTED_METRICS
        ):
            raise ValueError("scripted feedback row is outside the registered inventory")

        if self.status == "not_applicable":
            if (
                self.value is not None
                or self.numerator is not None
                or self.defined_episode_count != 0
            ):
                raise ValueError("not-applicable feedback rows cannot carry an estimate")
        else:
            if self.value is None or self.numerator is None or self.defined_episode_count == 0:
                raise ValueError("defined feedback rows require an estimate and numerator")
            expected_status = (
                "complete"
                if self.defined_episode_count == self.episode_count
                else "partial_defined_panel"
            )
            if self.status != expected_status:
                raise ValueError("feedback row status differs from its defined episode count")

        if self.unit in {"mean_difference", "mean_seconds"}:
            if self.status == "not_applicable":
                if self.denominator != 0:
                    raise ValueError("undefined mean requires a zero denominator")
            elif self.denominator != self.defined_episode_count or not math.isclose(
                self.value,
                self.numerator / self.denominator,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("mean feedback estimate differs from numerator/denominator")
        elif self.unit == "pooled_proportion":
            if self.denominator is None:
                raise ValueError("pooled rates require an explicit denominator")
            if self.status == "not_applicable":
                if self.denominator != 0:
                    raise ValueError("not-applicable pooled rate requires zero denominator")
            elif (
                self.denominator <= 0
                or not 0.0 <= self.numerator <= self.denominator
                or not math.isclose(
                    self.value,
                    self.numerator / self.denominator,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("pooled feedback rate differs from its support counts")
        elif (
            self.denominator is not None
            or self.status != "complete"
            or self.defined_episode_count != self.episode_count
            or not math.isclose(self.value, self.numerator, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError("total-count feedback rows require a complete integer total")
        if (
            self.metric.endswith("_rate")
            or self.metric in {"edit_locality", "target_change_recall"}
        ) and self.value is not None and not 0.0 <= self.value <= 1.0:
            raise ValueError("feedback proportion lies outside [0, 1]")
        return self


class Phase5FeedbackTableReceipt(ImmutableRecord):
    """Self-hashed proof that the public table covers the exact Phase 5 inventory."""

    receipt_id: str = Field(min_length=1)
    execution_index_hash: Sha256Digest
    feedback_manifest_hash: Sha256Digest
    scoring_session_hash: Sha256Digest
    scripted_metrics_hash: Sha256Digest
    scripted_scoring_receipt_hash: Sha256Digest
    scripted_scoring_metrics_file_sha256: Sha256Digest
    source_execution_record_hashes: tuple[Sha256Digest, ...]
    scripted_execution_record_hashes: tuple[Sha256Digest, ...]
    researcher_trace_execution_record_hashes: tuple[Sha256Digest, ...]
    scripted_metric_record_hashes: tuple[Sha256Digest, ...]
    researcher_trace_score_hashes: tuple[Sha256Digest, ...]
    aggregation_policy_sha256: Sha256Digest
    compiler_source_file_sha256: Sha256Digest
    table_file_sha256: Sha256Digest
    table_logical_hash: Sha256Digest
    table_columns: tuple[str, ...]
    table_row_count: int = Field(ge=65, le=70)
    scripted_table_row_count: Literal[60] = 60
    researcher_trace_table_row_count: int = Field(ge=5, le=10)
    scripted_condition_score_count: Literal[18] = 18
    researcher_trace_count: Literal[3] = 3
    trace_action_group_count: int = Field(ge=1, le=2)
    generated_at: AwareDatetime
    researcher_trace_gold_fields: Literal["NA"] = "NA"
    usability_claims: Literal[False] = False
    model_service_called: Literal[False] = False
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC

    @model_validator(mode="after")
    def exact_inventory_is_bound(self) -> Self:
        inventories = (
            (self.source_execution_record_hashes, 21, "source execution"),
            (self.scripted_execution_record_hashes, 18, "scripted execution"),
            (self.researcher_trace_execution_record_hashes, 3, "trace execution"),
            (self.scripted_metric_record_hashes, 18, "scripted metric"),
            (self.researcher_trace_score_hashes, 3, "trace score"),
        )
        for values, count, label in inventories:
            if values != tuple(sorted(set(values))) or len(values) != count:
                raise ValueError(f"{label} hashes must be canonical and exact")
        if (
            set(self.scripted_execution_record_hashes)
            | set(self.researcher_trace_execution_record_hashes)
            != set(self.source_execution_record_hashes)
            or set(self.scripted_execution_record_hashes)
            & set(self.researcher_trace_execution_record_hashes)
        ):
            raise ValueError("script and trace executions do not partition the Phase 5 journal")
        if self.table_columns != FEEDBACK_TABLE_COLUMNS:
            raise ValueError("feedback table columns differ from the Phase 7 contract")
        if (
            self.researcher_trace_table_row_count != 5 * self.trace_action_group_count
            or self.table_row_count
            != self.scripted_table_row_count + self.researcher_trace_table_row_count
        ):
            raise ValueError("feedback table row inventory differs from its action groups")
        return self


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise Phase5FeedbackReportError(f"symlinked feedback report path: {path}")
        if current.parent == current:
            return
        current = current.parent


def _canonical_record_bytes(value: ImmutableRecord) -> bytes:
    return (value.to_canonical_json() + "\n").encode("utf-8")


def _load_canonical(path: Path, model_type):
    _assert_no_symlink_chain(path)
    if not path.is_file():
        raise Phase5FeedbackReportError(f"missing feedback report source: {path}")
    raw = path.read_bytes()
    try:
        value = model_type.model_validate_json(raw)
    except Exception as error:
        raise Phase5FeedbackReportError(
            f"invalid {model_type.__name__} feedback report source: {error}"
        ) from error
    if raw != _canonical_record_bytes(value):
        raise Phase5FeedbackReportError(
            f"noncanonical {model_type.__name__} feedback report source"
        )
    return value, raw


def _status(defined_count: int, episode_count: int) -> FeedbackMetricStatus:
    if defined_count == 0:
        return "not_applicable"
    if defined_count == episode_count:
        return "complete"
    return "partial_defined_panel"


def _mean_row(
    *,
    episode_class: FeedbackMetricEpisodeKind,
    condition: ConditionName,
    action: FeedbackAction,
    metric: FeedbackMetricName,
    values: Sequence[float | None],
    capability_limited_count: int,
    gold_applicability: Literal["applicable", "NA"],
    unit: Literal["mean_difference", "mean_seconds"],
) -> Phase5FeedbackTableRow:
    defined = tuple(float(item) for item in values if item is not None)
    numerator = math.fsum(defined) if defined else None
    denominator = len(defined)
    return Phase5FeedbackTableRow(
        episode_class=episode_class,
        condition=condition,
        action=action,
        metric=metric,
        value=None if numerator is None else numerator / denominator,
        unit=unit,
        episode_count=len(values),
        capability_limited_count=capability_limited_count,
        gold_applicability=gold_applicability,
        status=_status(denominator, len(values)),
        defined_episode_count=denominator,
        numerator=numerator,
        denominator=denominator,
    )


def _rate_row(
    *,
    episode_class: FeedbackMetricEpisodeKind,
    condition: ConditionName,
    action: FeedbackAction,
    metric: FeedbackMetricName,
    rates: Sequence[RateResult],
    capability_limited_count: int,
    gold_applicability: Literal["applicable", "NA"],
) -> Phase5FeedbackTableRow:
    numerator = sum(item.numerator for item in rates)
    denominator = sum(item.denominator for item in rates)
    defined_count = sum(item.denominator > 0 for item in rates)
    return Phase5FeedbackTableRow(
        episode_class=episode_class,
        condition=condition,
        action=action,
        metric=metric,
        value=None if denominator == 0 else numerator / denominator,
        unit="pooled_proportion",
        episode_count=len(rates),
        capability_limited_count=capability_limited_count,
        gold_applicability=gold_applicability,
        status=_status(defined_count, len(rates)),
        defined_episode_count=defined_count,
        numerator=float(numerator) if denominator else None,
        denominator=denominator,
    )


def _binary_rate_row(
    *,
    episode_class: FeedbackMetricEpisodeKind,
    condition: ConditionName,
    action: FeedbackAction,
    metric: FeedbackMetricName,
    values: Sequence[bool],
    capability_limited_count: int,
    gold_applicability: Literal["applicable", "NA"],
) -> Phase5FeedbackTableRow:
    numerator = sum(values)
    denominator = len(values)
    return Phase5FeedbackTableRow(
        episode_class=episode_class,
        condition=condition,
        action=action,
        metric=metric,
        value=numerator / denominator,
        unit="pooled_proportion",
        episode_count=denominator,
        capability_limited_count=capability_limited_count,
        gold_applicability=gold_applicability,
        status="complete",
        defined_episode_count=denominator,
        numerator=float(numerator),
        denominator=denominator,
    )


def _count_row(
    *,
    episode_class: FeedbackMetricEpisodeKind,
    condition: ConditionName,
    action: FeedbackAction,
    metric: FeedbackMetricName,
    values: Sequence[int],
    capability_limited_count: int,
    gold_applicability: Literal["applicable", "NA"],
) -> Phase5FeedbackTableRow:
    total = sum(values)
    return Phase5FeedbackTableRow(
        episode_class=episode_class,
        condition=condition,
        action=action,
        metric=metric,
        value=float(total),
        unit="total_count",
        episode_count=len(values),
        capability_limited_count=capability_limited_count,
        gold_applicability=gold_applicability,
        status="complete",
        defined_episode_count=len(values),
        numerator=float(total),
    )


def _scripted_rows(
    records: Sequence[Phase5ScriptedFeedbackMetricRecord],
    executions: dict[tuple[str, ConditionName], FeedbackExecutionRecord],
) -> tuple[Phase5FeedbackTableRow, ...]:
    grouped: dict[
        tuple[ConditionName, FeedbackAction],
        list[tuple[Phase5ScriptedFeedbackMetricRecord, FeedbackExecutionRecord]],
    ] = defaultdict(list)
    for metric_record in records:
        key = (metric_record.episode_id, metric_record.condition)
        execution_record = executions.get(key)
        if execution_record is None:
            raise Phase5FeedbackReportError("scripted metric references a missing execution")
        execution = execution_record.execution
        score = metric_record.score
        expected_replay = execution.replay is not None and execution.replay.replay_hash_success
        expected_capability = (
            execution.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
        )
        expected_change_count = 0 if execution.diff is None else len(execution.diff.changes)
        if (
            execution.kind is not FeedbackExecutionKind.SCRIPTED_KNOWN_ANSWER
            or metric_record.execution_record_hash != execution_record.content_hash
            or metric_record.execution_hash != execution.content_hash
            or metric_record.instruction_hash != execution.instruction.content_hash
            or metric_record.before_projection_hash != execution.before_bundle.projection_hash
            or metric_record.after_projection_hash
            != (
                None
                if execution.after_bundle is None
                else execution.after_bundle.projection_hash
            )
            or metric_record.diff_hash
            != (None if execution.diff is None else execution.diff.content_hash)
            or score.latency_seconds != execution.latency_seconds
            or score.capability_limited is not expected_capability
            or score.replay_hash_success is not expected_replay
            or score.semantic_change_count != expected_change_count
        ):
            raise Phase5FeedbackReportError(
                "scripted metric record differs from its immutable execution"
            )
        grouped[(metric_record.condition, execution.action)].append(
            (metric_record, execution_record)
        )

    expected_groups = {
        (condition, action)
        for condition in (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        )
        for action in (FeedbackAction.REFINE_CONTEXT, FeedbackAction.REQUEST_MERGE_SPLIT)
    }
    if set(grouped) != expected_groups or any(len(values) != 3 for values in grouped.values()):
        raise Phase5FeedbackReportError(
            "scripted scores must retain three episodes per condition and action"
        )

    rows: list[Phase5FeedbackTableRow] = []
    for (condition, action), values in sorted(
        grouped.items(), key=lambda item: (item[0][0].value, item[0][1].value)
    ):
        scores = tuple(item[0].score for item in values)
        episode_executions = tuple(item[1].execution for item in values)
        capability_count = sum(item.capability_limited for item in scores)
        target_rates = tuple(item.target_change_recall for item in scores)
        locality_rates = tuple(item.edit_locality for item in scores)
        unsupported_counts = tuple(item.unsupported_change_count for item in scores)
        if any(
            item is None for item in (*target_rates, *locality_rates, *unsupported_counts)
        ):
            raise Phase5FeedbackReportError(
                "scripted scorer omitted target/locality/unsupported metrics"
            )
        concrete_targets = tuple(item for item in target_rates if item is not None)
        concrete_locality = tuple(item for item in locality_rates if item is not None)
        concrete_unsupported = tuple(
            item for item in unsupported_counts if item is not None
        )
        common = {
            "episode_class": FeedbackMetricEpisodeKind.SCRIPTED_KNOWN_ANSWER,
            "condition": condition,
            "action": action,
            "capability_limited_count": capability_count,
            "gold_applicability": "applicable",
        }
        rows.extend(
            (
                _mean_row(
                    **common,
                    metric="post_minus_pre_strict_qualified_assertion_f1",
                    values=tuple(item.strict_f1_change for item in scores),
                    unit="mean_difference",
                ),
                _rate_row(
                    **common,
                    metric="target_change_recall",
                    rates=concrete_targets,
                ),
                _rate_row(
                    **common,
                    metric="edit_locality",
                    rates=concrete_locality,
                ),
                _count_row(
                    **common,
                    metric="unsupported_change_count",
                    values=concrete_unsupported,
                ),
                _mean_row(
                    **common,
                    metric="rare_pivotal_recall_change",
                    values=tuple(item.rare_pivotal_recall_change for item in scores),
                    unit="mean_difference",
                ),
                _count_row(
                    **common,
                    metric="semantic_change_count",
                    values=tuple(item.semantic_change_count for item in scores),
                ),
                _binary_rate_row(
                    **common,
                    metric="capability_limited_rate",
                    values=tuple(item.capability_limited for item in scores),
                ),
                _mean_row(
                    **common,
                    metric="latency_seconds",
                    values=tuple(item.latency_seconds for item in scores),
                    unit="mean_seconds",
                ),
                _binary_rate_row(
                    **common,
                    metric="replay_success_rate",
                    values=tuple(item.replay_hash_success for item in scores),
                ),
                _binary_rate_row(
                    **common,
                    metric="resolution_success_rate",
                    values=tuple(
                        item.resolution.status is FeedbackResolutionStatus.RESOLVED
                        for item in episode_executions
                    ),
                ),
            )
        )
    return tuple(rows)


def _trace_rows(
    records: Sequence[FeedbackExecutionRecord],
) -> tuple[tuple[Phase5FeedbackTableRow, ...], tuple[FeedbackScore, ...]]:
    grouped: dict[FeedbackAction, list[tuple[FeedbackScore, FeedbackExecutionRecord]]] = (
        defaultdict(list)
    )
    scores: list[FeedbackScore] = []
    for record in records:
        execution = record.execution
        if (
            execution.kind is not FeedbackExecutionKind.RESEARCHER_TRACE
            or execution.condition is not ConditionName.C2_LLM_QUERY
            or record.scorer_binding is not None
            or execution.gold_dependent_fields != "NA"
        ):
            raise Phase5FeedbackReportError("researcher trace crossed its gold-free boundary")
        changes = () if execution.diff is None else tuple(
            item.change_id for item in execution.diff.changes
        )
        if len(changes) != len(set(changes)):
            raise Phase5FeedbackReportError("researcher trace contains duplicate graph changes")
        score = record_researcher_trace(
            episode_id=execution.episode_id,
            changed_semantic_ids=changes,
            capability_limited=(
                execution.resolution.status is FeedbackResolutionStatus.CAPABILITY_LIMITED
            ),
            latency_seconds=execution.latency_seconds,
            replay_hash_success=(
                execution.replay is not None and execution.replay.replay_hash_success
            ),
        )
        scores.append(score)
        grouped[execution.action].append((score, record))
    if len(scores) != 3 or len({item.episode_id for item in scores}) != 3:
        raise Phase5FeedbackReportError("feedback report requires exactly three trace scores")

    rows: list[Phase5FeedbackTableRow] = []
    for action, values in sorted(grouped.items(), key=lambda item: item[0].value):
        action_scores = tuple(item[0] for item in values)
        executions = tuple(item[1].execution for item in values)
        capability_count = sum(item.capability_limited for item in action_scores)
        common = {
            "episode_class": FeedbackMetricEpisodeKind.RESEARCHER_TRACE,
            "condition": ConditionName.C2_LLM_QUERY,
            "action": action,
            "capability_limited_count": capability_count,
            "gold_applicability": "NA",
        }
        rows.extend(
            (
                _count_row(
                    **common,
                    metric="semantic_change_count",
                    values=tuple(item.semantic_change_count for item in action_scores),
                ),
                _binary_rate_row(
                    **common,
                    metric="capability_limited_rate",
                    values=tuple(item.capability_limited for item in action_scores),
                ),
                _mean_row(
                    **common,
                    metric="latency_seconds",
                    values=tuple(item.latency_seconds for item in action_scores),
                    unit="mean_seconds",
                ),
                _binary_rate_row(
                    **common,
                    metric="replay_success_rate",
                    values=tuple(item.replay_hash_success for item in action_scores),
                ),
                _binary_rate_row(
                    **common,
                    metric="resolution_success_rate",
                    values=tuple(
                        item.resolution.status is FeedbackResolutionStatus.RESOLVED
                        for item in executions
                    ),
                ),
            )
        )
    return tuple(rows), tuple(scores)


def _csv_bytes(rows: Sequence[Phase5FeedbackTableRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FEEDBACK_TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        payload = row.model_dump(mode="json", exclude={"schema_version", "content_hash"})
        writer.writerow({column: payload[column] for column in FEEDBACK_TABLE_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _read_scoring_bundle(
    scoring_root: Path,
) -> tuple[
    Phase5FeedbackScoringSession,
    Phase5ScriptedFeedbackMetrics,
    Phase5FeedbackScoringReceipt,
]:
    _assert_no_symlink_chain(scoring_root)
    root = scoring_root.resolve(strict=True)
    session, _ = _load_canonical(root / SESSION_FILE_NAME, Phase5FeedbackScoringSession)
    metrics, metrics_bytes = _load_canonical(
        root / METRICS_FILE_NAME, Phase5ScriptedFeedbackMetrics
    )
    receipt, _ = _load_canonical(root / RECEIPT_FILE_NAME, Phase5FeedbackScoringReceipt)
    if (
        receipt.session_hash != session.content_hash
        or receipt.metrics_hash != metrics.content_hash
        or receipt.metrics_file_sha256 != hashlib.sha256(metrics_bytes).hexdigest()
        or tuple(item.execution_record_hash for item in metrics.records)
        != receipt.execution_record_hashes
        or metrics.generated_at != session.compiled_at
        or receipt.compiled_at != session.compiled_at
        or metrics.source_manifest_hash != session.source_manifest_hash
        or metrics.known_answer_source_hash != session.known_answer_source_hash
        or metrics.input_manifest_hash != session.input_manifest_hash
        or metrics.execution_index_hash != session.execution_index_hash
        or receipt.source_manifest_hash != metrics.source_manifest_hash
        or receipt.known_answer_source_hash != metrics.known_answer_source_hash
        or receipt.input_manifest_hash != metrics.input_manifest_hash
        or receipt.feedback_manifest_hash != metrics.feedback_manifest_hash
        or receipt.execution_index_hash != metrics.execution_index_hash
        or receipt.scorer_binding_authorization_hash
        != metrics.scorer_binding_authorization_hash
    ):
        raise Phase5FeedbackReportError("scripted scorer bundle lineage changed")
    return session, metrics, receipt


def _aggregation_policy_hash() -> str:
    return canonical_sha256(
        {
            "schema_version": "1.0.0",
            "grouping": ("episode_class", "condition", "action"),
            "scripted_metrics": SCRIPTED_METRICS,
            "trace_metrics": TRACE_METRICS,
            "mean_policy": "arithmetic_mean_over_defined_episodes_with_count",
            "rate_policy": "pooled_numerator_over_pooled_denominator",
            "count_policy": "sum_over_all_episodes",
            "na_policy": "empty_value_and_zero_denominator",
            "trace_gold_policy": "NA",
        }
    )


def prepare_phase5_feedback_table(
    *,
    feedback_journal_root: Path,
    scoring_root: Path,
) -> tuple[tuple[Phase5FeedbackTableRow, ...], bytes, Phase5FeedbackTableReceipt]:
    """Reproduce the public table and receipt without writing any files."""

    try:
        inputs, manifest, index, execution_map = _verify_journal(feedback_journal_root)
    except (OSError, ValueError, Phase5FeedbackScoringError) as error:
        raise Phase5FeedbackReportError(f"invalid Phase 5 execution journal: {error}") from error
    session, metrics, scoring_receipt = _read_scoring_bundle(scoring_root)
    if (
        metrics.protocol_hash != inputs.protocol_hash
        or metrics.input_manifest_hash != inputs.content_hash
        or metrics.feedback_manifest_hash != manifest.content_hash
        or metrics.execution_index_hash != index.content_hash
        or scoring_receipt.execution_index_hash != index.content_hash
    ):
        raise Phase5FeedbackReportError("feedback scorer bundle references another journal")

    scripted_executions = {
        key: value
        for key, value in execution_map.items()
        if value.execution.kind is FeedbackExecutionKind.SCRIPTED_KNOWN_ANSWER
    }
    traces = tuple(
        value
        for value in execution_map.values()
        if value.execution.kind is FeedbackExecutionKind.RESEARCHER_TRACE
    )
    if len(scripted_executions) != 18 or len(traces) != 3:
        raise Phase5FeedbackReportError("Phase 5 journal lacks the exact 18+3 inventory")
    scripted_rows = _scripted_rows(metrics.records, scripted_executions)
    trace_rows, trace_scores = _trace_rows(traces)
    rows = tuple(
        sorted(
            (*scripted_rows, *trace_rows),
            key=lambda item: (
                item.episode_class.value,
                item.condition.value,
                item.action.value,
                item.metric,
            ),
        )
    )
    keys = tuple(
        (item.episode_class, item.condition, item.action, item.metric) for item in rows
    )
    if len(keys) != len(set(keys)) or len(scripted_rows) != 60:
        raise Phase5FeedbackReportError("feedback report table inventory is not exact")
    table_bytes = _csv_bytes(rows)
    receipt = Phase5FeedbackTableReceipt(
        receipt_id=f"phase5-feedback-table-{hashlib.sha256(table_bytes).hexdigest()[:24]}",
        execution_index_hash=index.content_hash,
        feedback_manifest_hash=manifest.content_hash,
        scoring_session_hash=session.content_hash,
        scripted_metrics_hash=metrics.content_hash,
        scripted_scoring_receipt_hash=scoring_receipt.content_hash,
        scripted_scoring_metrics_file_sha256=scoring_receipt.metrics_file_sha256,
        source_execution_record_hashes=tuple(
            sorted(item.content_hash for item in execution_map.values())
        ),
        scripted_execution_record_hashes=tuple(
            sorted(item.content_hash for item in scripted_executions.values())
        ),
        researcher_trace_execution_record_hashes=tuple(
            sorted(item.content_hash for item in traces)
        ),
        scripted_metric_record_hashes=tuple(
            sorted(item.content_hash for item in metrics.records)
        ),
        researcher_trace_score_hashes=tuple(sorted(item.content_hash for item in trace_scores)),
        aggregation_policy_sha256=_aggregation_policy_hash(),
        compiler_source_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        table_file_sha256=hashlib.sha256(table_bytes).hexdigest(),
        table_logical_hash=canonical_sha256(rows),
        table_columns=FEEDBACK_TABLE_COLUMNS,
        table_row_count=len(rows),
        researcher_trace_table_row_count=len(trace_rows),
        trace_action_group_count=len({item.execution.action for item in traces}),
        generated_at=scoring_receipt.compiled_at,
    )
    return rows, table_bytes, receipt


def _append_exact(path: Path, payload: bytes) -> bool:
    _assert_no_symlink_chain(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise Phase5FeedbackReportError(f"append-only feedback report drift: {path.name}")
        return False
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise Phase5FeedbackReportError(
                    f"concurrent feedback report drift: {path.name}"
                ) from None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _validate_output_root(output_root: Path) -> Path:
    root = output_root.absolute()
    _assert_no_symlink_chain(root)
    allowed = {FEEDBACK_TABLE_FILE_NAME, FEEDBACK_TABLE_RECEIPT_FILE_NAME}
    if root.exists():
        if not root.is_dir():
            raise Phase5FeedbackReportError("feedback report output root is not a directory")
        entries = tuple(root.iterdir())
        unsafe = tuple(
            item.name for item in entries if item.is_symlink() or not item.is_file()
        )
        if unsafe:
            raise Phase5FeedbackReportError(
                f"non-regular feedback report output artifact: {sorted(unsafe)[0]}"
            )
        extras = {
            item.name
            for item in entries
            if not (
                item.name in allowed
                or (item.name.startswith(".") and item.name.endswith(".tmp"))
            )
        }
        if extras:
            raise Phase5FeedbackReportError(
                f"unexpected feedback report output artifact: {sorted(extras)[0]}"
            )
    return root


def materialize_phase5_feedback_table(
    *,
    output_root: Path,
    table_bytes: bytes,
    receipt: Phase5FeedbackTableReceipt,
) -> bool:
    """Atomically append the canonical table and its self-hashed receipt."""

    root = _validate_output_root(output_root)
    created = not (root / FEEDBACK_TABLE_RECEIPT_FILE_NAME).exists()
    _append_exact(root / FEEDBACK_TABLE_FILE_NAME, table_bytes)
    _append_exact(
        root / FEEDBACK_TABLE_RECEIPT_FILE_NAME,
        _canonical_record_bytes(receipt),
    )
    return created


def verify_phase5_feedback_table(
    *,
    feedback_journal_root: Path,
    scoring_root: Path,
    output_root: Path,
) -> Phase5FeedbackTableReceipt:
    """Rebuild the complete table and require byte-identical retained outputs."""

    _rows, table_bytes, expected = prepare_phase5_feedback_table(
        feedback_journal_root=feedback_journal_root,
        scoring_root=scoring_root,
    )
    root = _validate_output_root(output_root)
    observed, observed_bytes = _load_canonical(
        root / FEEDBACK_TABLE_RECEIPT_FILE_NAME,
        Phase5FeedbackTableReceipt,
    )
    if observed != expected or observed_bytes != _canonical_record_bytes(expected):
        raise Phase5FeedbackReportError("feedback table receipt does not replay")
    table_path = root / FEEDBACK_TABLE_FILE_NAME
    _assert_no_symlink_chain(table_path)
    if not table_path.is_file() or table_path.read_bytes() != table_bytes:
        raise Phase5FeedbackReportError("feedback table bytes do not replay")
    return observed


__all__ = [
    "DEFAULT_PHASE5_REPORT_ROOT",
    "FEEDBACK_TABLE_COLUMNS",
    "FEEDBACK_TABLE_FILE_NAME",
    "FEEDBACK_TABLE_RECEIPT_FILE_NAME",
    "Phase5FeedbackReportError",
    "Phase5FeedbackTableReceipt",
    "Phase5FeedbackTableRow",
    "materialize_phase5_feedback_table",
    "prepare_phase5_feedback_table",
    "verify_phase5_feedback_table",
]
