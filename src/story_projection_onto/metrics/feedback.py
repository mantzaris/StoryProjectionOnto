"""Descriptive metrics for the registered feedback and interface episodes."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import ImmutableRecord
from story_projection_onto.metrics.common import RateResult, rate


class FeedbackEpisodeKind(StrEnum):
    SCRIPTED_KNOWN_ANSWER = "scripted_known_answer"
    RESEARCHER_TRACE = "researcher_trace"


class FeedbackScore(ImmutableRecord):
    """One episode, keeping confirmatory gold fields absent from researcher traces."""

    episode_id: str = Field(min_length=1)
    kind: FeedbackEpisodeKind
    strict_f1_change: float | None = Field(default=None, ge=-1.0, le=1.0)
    target_change_recall: RateResult | None = None
    edit_locality: RateResult | None = None
    unsupported_change_count: int | None = Field(default=None, ge=0)
    rare_pivotal_recall_change: float | None = Field(default=None, ge=-1.0, le=1.0)
    semantic_change_count: int = Field(ge=0)
    capability_limited: bool
    latency_seconds: float = Field(ge=0.0)
    replay_hash_success: bool

    @model_validator(mode="after")
    def enforce_trace_na_policy(self) -> Self:
        gold_fields = (
            self.strict_f1_change,
            self.target_change_recall,
            self.edit_locality,
            self.unsupported_change_count,
            self.rare_pivotal_recall_change,
        )
        if self.kind is FeedbackEpisodeKind.RESEARCHER_TRACE and any(
            value is not None for value in gold_fields
        ):
            raise ValueError("researcher traces must keep every gold-dependent field NA")
        if self.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER:
            if self.target_change_recall is None or self.edit_locality is None:
                raise ValueError("known-answer scripts require target and locality results")
            if self.unsupported_change_count is None:
                raise ValueError("known-answer scripts require unsupported-change accounting")
            if not self.capability_limited and self.strict_f1_change is None:
                raise ValueError("resolved known-answer scripts require strict F1 change")
        if self.capability_limited and self.semantic_change_count != 0:
            raise ValueError("capability-limited episodes cannot claim semantic changes")
        return self


class FeedbackCapabilityLimitedRates(ImmutableRecord):
    """Separate rates for scripts and traces; the two episode sets are never pooled."""

    scripted_known_answer: RateResult
    researcher_trace: RateResult


def _normalized_ids(values: Sequence[str], *, field_name: str) -> frozenset[str]:
    if any(not value for value in values):
        raise ValueError(f"{field_name} cannot contain empty IDs")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique IDs")
    return frozenset(values)


def score_scripted_feedback(
    *,
    episode_id: str,
    before_strict_f1: float | None,
    after_strict_f1: float | None,
    required_target_change_ids: Sequence[str],
    realized_target_change_ids: Sequence[str],
    changed_semantic_ids: Sequence[str],
    requested_scope_ids: Sequence[str],
    unsupported_changed_ids: Sequence[str],
    before_rare_pivotal_recall: float | None,
    after_rare_pivotal_recall: float | None,
    capability_limited: bool,
    latency_seconds: float,
    replay_hash_success: bool,
) -> FeedbackScore:
    """Score one of the six frozen known-answer feedback scripts.

    Edit locality is the fraction of changed semantic objects inside the script's
    independently frozen requested scope. An episode with no semantic changes has an
    explicit zero denominator and therefore ``NA`` locality.
    """

    required = _normalized_ids(
        required_target_change_ids,
        field_name="required_target_change_ids",
    )
    realized = _normalized_ids(
        realized_target_change_ids,
        field_name="realized_target_change_ids",
    )
    changed = _normalized_ids(changed_semantic_ids, field_name="changed_semantic_ids")
    requested_scope = _normalized_ids(requested_scope_ids, field_name="requested_scope_ids")
    unsupported = _normalized_ids(
        unsupported_changed_ids,
        field_name="unsupported_changed_ids",
    )
    if not required:
        raise ValueError("known-answer scripts require at least one target change")
    if not realized.issubset(changed):
        raise ValueError("realized target changes must be among changed semantic objects")
    if not unsupported.issubset(changed):
        raise ValueError("unsupported changes must be among changed semantic objects")

    paired_f1 = (before_strict_f1, after_strict_f1)
    paired_rare = (before_rare_pivotal_recall, after_rare_pivotal_recall)
    for field_name, values in (("strict F1", paired_f1), ("rare-pivotal recall", paired_rare)):
        if (values[0] is None) != (values[1] is None):
            raise ValueError(f"before and after {field_name} must be jointly present or NA")
        if any(value is not None and not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{field_name} values must lie in [0, 1]")
    if not capability_limited and paired_f1[0] is None:
        raise ValueError("a resolved known-answer episode requires before/after strict F1")
    if capability_limited and (realized or changed or unsupported):
        raise ValueError("capability-limited scripts cannot claim realized semantic changes")
    if capability_limited and any(value is not None for value in (*paired_f1, *paired_rare)):
        raise ValueError("capability-limited script gold scores must remain NA")

    return FeedbackScore(
        episode_id=episode_id,
        kind=FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER,
        strict_f1_change=(
            None
            if before_strict_f1 is None or after_strict_f1 is None
            else after_strict_f1 - before_strict_f1
        ),
        target_change_recall=rate(len(required.intersection(realized)), len(required)),
        edit_locality=rate(len(changed.intersection(requested_scope)), len(changed)),
        unsupported_change_count=len(unsupported),
        rare_pivotal_recall_change=(
            None
            if paired_rare[0] is None or paired_rare[1] is None
            else paired_rare[1] - paired_rare[0]
        ),
        semantic_change_count=len(changed),
        capability_limited=capability_limited,
        latency_seconds=latency_seconds,
        replay_hash_success=replay_hash_success,
    )


def record_researcher_trace(
    *,
    episode_id: str,
    changed_semantic_ids: Sequence[str],
    capability_limited: bool,
    latency_seconds: float,
    replay_hash_success: bool,
) -> FeedbackScore:
    """Record operational fields only; researcher traces never consume gold."""

    changed = _normalized_ids(changed_semantic_ids, field_name="changed_semantic_ids")
    return FeedbackScore(
        episode_id=episode_id,
        kind=FeedbackEpisodeKind.RESEARCHER_TRACE,
        semantic_change_count=len(changed),
        capability_limited=capability_limited,
        latency_seconds=latency_seconds,
        replay_hash_success=replay_hash_success,
    )


def capability_limited_rate(scores: Sequence[FeedbackScore]) -> RateResult:
    return rate(sum(score.capability_limited for score in scores), len(scores))


def capability_limited_rates_by_kind(
    scores: Sequence[FeedbackScore],
) -> FeedbackCapabilityLimitedRates:
    """Report the registered capability diagnostic separately for both episode sets."""

    scripts = tuple(
        item for item in scores if item.kind is FeedbackEpisodeKind.SCRIPTED_KNOWN_ANSWER
    )
    traces = tuple(item for item in scores if item.kind is FeedbackEpisodeKind.RESEARCHER_TRACE)
    return FeedbackCapabilityLimitedRates(
        scripted_known_answer=capability_limited_rate(scripts),
        researcher_trace=capability_limited_rate(traces),
    )


__all__ = [
    "FeedbackCapabilityLimitedRates",
    "FeedbackEpisodeKind",
    "FeedbackScore",
    "capability_limited_rate",
    "capability_limited_rates_by_kind",
    "record_researcher_trace",
    "score_scripted_feedback",
]
