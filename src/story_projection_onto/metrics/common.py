"""Shared explicit-denominator records for scientific metrics."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, cast

from pydantic import Field, model_validator

from story_projection_onto.contracts import ImmutableRecord


def revalidated_copy[ImmutableRecordT: ImmutableRecord](
    record: ImmutableRecordT,
    /,
    **updates: Any,
) -> ImmutableRecordT:
    """Return an immutable-record copy with a freshly verified content hash.

    Pydantic's ``model_copy(update=...)`` intentionally skips validation.  That is
    unsafe for the content-addressed records used by the scorer because it retains
    the source record's hash after changing scientific fields.  Reconstruct through
    validation instead, recursively allowing nested records to verify themselves.
    """

    payload = record.model_dump(mode="python", exclude={"content_hash"})
    payload.update(updates)
    return cast(ImmutableRecordT, type(record).model_validate(payload))


class MetricStatus(StrEnum):
    DEFINED = "defined"
    NOT_APPLICABLE = "not_applicable"
    INVALID_SEMANTIC_ZERO = "invalid_semantic_zero"


class PrecisionRecallF1(ImmutableRecord):
    true_positive_count: int = Field(ge=0)
    false_positive_count: int = Field(ge=0)
    false_negative_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    gold_count: int = Field(ge=0)
    precision_denominator: int = Field(ge=0)
    recall_denominator: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    status: MetricStatus

    @model_validator(mode="after")
    def counts_and_denominators_agree(self) -> PrecisionRecallF1:
        if self.predicted_count != self.true_positive_count + self.false_positive_count:
            raise ValueError("predicted count must equal TP + FP")
        if self.gold_count != self.true_positive_count + self.false_negative_count:
            raise ValueError("gold count must equal TP + FN")
        if self.precision_denominator != self.predicted_count:
            raise ValueError("precision denominator must expose predicted count")
        if self.recall_denominator != self.gold_count:
            raise ValueError("recall denominator must expose gold count")
        values = (self.precision, self.recall, self.f1)
        if self.status is MetricStatus.NOT_APPLICABLE and any(
            value is not None for value in values
        ):
            raise ValueError("not-applicable PRF values must be NA")
        if self.status is not MetricStatus.NOT_APPLICABLE and any(
            value is None for value in values
        ):
            raise ValueError("defined or invalid-zero PRF values must be numeric")
        if self.status is MetricStatus.INVALID_SEMANTIC_ZERO and values != (0.0, 0.0, 0.0):
            raise ValueError("invalid semantic output must receive an explicit zero")
        both_empty = self.predicted_count == 0 and self.gold_count == 0
        if (self.status is MetricStatus.NOT_APPLICABLE) != both_empty:
            raise ValueError("PRF is not-applicable exactly when prediction and gold are empty")
        if self.status is MetricStatus.INVALID_SEMANTIC_ZERO:
            if self.true_positive_count != 0:
                raise ValueError("invalid semantic PRF cannot retain true positives")
            return self
        if self.status is MetricStatus.DEFINED:
            expected_precision = (
                self.true_positive_count / self.predicted_count if self.predicted_count else 0.0
            )
            expected_recall = self.true_positive_count / self.gold_count if self.gold_count else 0.0
            expected_f1 = (
                2 * expected_precision * expected_recall / (expected_precision + expected_recall)
                if expected_precision + expected_recall
                else 0.0
            )
            for name, actual, expected in (
                ("precision", self.precision, expected_precision),
                ("recall", self.recall, expected_recall),
                ("F1", self.f1, expected_f1),
            ):
                if actual is None or not math.isclose(actual, expected, abs_tol=1e-15):
                    raise ValueError(f"{name} does not match the exposed counts")
        return self


def precision_recall_f1(
    *,
    true_positives: int,
    predicted_count: int,
    gold_count: int,
    invalid_semantic_output: bool = False,
) -> PrecisionRecallF1:
    """Compute PRF with registered empty/invalid behavior and visible denominators."""

    if not 0 <= true_positives <= min(predicted_count, gold_count):
        raise ValueError("true positives must lie within both set sizes")
    false_positives = predicted_count - true_positives
    false_negatives = gold_count - true_positives
    if invalid_semantic_output and (predicted_count > 0 or gold_count > 0):
        return PrecisionRecallF1(
            true_positive_count=0,
            false_positive_count=predicted_count,
            false_negative_count=gold_count,
            predicted_count=predicted_count,
            gold_count=gold_count,
            precision_denominator=predicted_count,
            recall_denominator=gold_count,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status=MetricStatus.INVALID_SEMANTIC_ZERO,
        )
    if predicted_count == 0 and gold_count == 0:
        return PrecisionRecallF1(
            true_positive_count=0,
            false_positive_count=0,
            false_negative_count=0,
            predicted_count=0,
            gold_count=0,
            precision_denominator=0,
            recall_denominator=0,
            status=MetricStatus.NOT_APPLICABLE,
        )
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PrecisionRecallF1(
        true_positive_count=true_positives,
        false_positive_count=false_positives,
        false_negative_count=false_negatives,
        predicted_count=predicted_count,
        gold_count=gold_count,
        precision_denominator=predicted_count,
        recall_denominator=gold_count,
        precision=precision,
        recall=recall,
        f1=f1,
        status=MetricStatus.DEFINED,
    )


class RateResult(ImmutableRecord):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    status: MetricStatus

    @model_validator(mode="after")
    def validate_rate(self) -> RateResult:
        if self.numerator > self.denominator:
            raise ValueError("rate numerator cannot exceed denominator")
        if self.denominator == 0:
            if (
                self.numerator != 0
                or self.value is not None
                or self.status is not MetricStatus.NOT_APPLICABLE
            ):
                raise ValueError("zero-denominator rates must be not-applicable")
        elif self.value is None or self.status is MetricStatus.NOT_APPLICABLE:
            raise ValueError("positive-denominator rates must be numeric")
        elif not math.isclose(self.value, self.numerator / self.denominator, abs_tol=1e-15):
            raise ValueError("rate value does not match its numerator and denominator")
        if self.status is MetricStatus.INVALID_SEMANTIC_ZERO and (
            self.numerator != 0 or self.value != 0.0
        ):
            raise ValueError("invalid semantic rate must be an explicit zero")
        return self


def rate(
    numerator: int,
    denominator: int,
    *,
    invalid_semantic_output: bool = False,
) -> RateResult:
    if denominator < 0 or numerator < 0 or numerator > denominator:
        raise ValueError("invalid rate counts")
    if denominator == 0:
        return RateResult(
            numerator=0,
            denominator=0,
            value=None,
            status=MetricStatus.NOT_APPLICABLE,
        )
    if invalid_semantic_output:
        return RateResult(
            numerator=0,
            denominator=denominator,
            value=0.0,
            status=MetricStatus.INVALID_SEMANTIC_ZERO,
        )
    return RateResult(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator,
        status=MetricStatus.DEFINED,
    )


def maximum_cardinality_matching(
    adjacency: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, str], ...]:
    """Deterministic Kuhn matching for the study's bounded bipartite graphs."""

    target_owner: dict[str, str] = {}

    def augment(prediction_id: str, visited: set[str]) -> bool:
        for target_id in adjacency.get(prediction_id, ()):
            if target_id in visited:
                continue
            visited.add(target_id)
            owner = target_owner.get(target_id)
            if owner is None or augment(owner, visited):
                target_owner[target_id] = prediction_id
                return True
        return False

    for prediction_id in sorted(adjacency):
        augment(prediction_id, set())
    return tuple(
        sorted((prediction_id, target_id) for target_id, prediction_id in target_owner.items())
    )
