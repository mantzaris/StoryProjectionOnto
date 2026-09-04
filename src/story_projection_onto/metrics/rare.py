"""Scorer-only rare-pivotal preservation and support-path safeguards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    GoldContextualProjection,
    ImmutableRecord,
    canonical_sha256,
)
from story_projection_onto.metrics.common import MetricStatus, RateResult, rate


class RarePivotalAnnotation(ImmutableRecord):
    """Gold-only annotation that is structurally impossible to pass to a runner."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    assertion_target_id: str = Field(min_length=1)
    is_rare: bool
    is_pivotal: bool
    support_path_assertion_target_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def path_is_normalized_and_contains_target(self) -> Self:
        if any(not target_id for target_id in self.support_path_assertion_target_ids):
            raise ValueError("support path target IDs must be nonempty")
        if self.support_path_assertion_target_ids != tuple(
            dict.fromkeys(self.support_path_assertion_target_ids)
        ):
            raise ValueError("support path target IDs must be unique and ordered")
        if self.is_rare and self.is_pivotal:
            if not self.support_path_assertion_target_ids:
                raise ValueError("rare-pivotal assertions require a complete support path")
            if self.assertion_target_id not in self.support_path_assertion_target_ids:
                raise ValueError("rare-pivotal support path must contain its target assertion")
        return self


class RarityPivotalityStratum(StrEnum):
    COMMON_NONPIVOTAL = "common_nonpivotal"
    COMMON_PIVOTAL = "common_pivotal"
    RARE_NONPIVOTAL = "rare_nonpivotal"
    RARE_PIVOTAL = "rare_pivotal"


_REGISTERED_STRATA = (
    RarityPivotalityStratum.COMMON_NONPIVOTAL,
    RarityPivotalityStratum.COMMON_PIVOTAL,
    RarityPivotalityStratum.RARE_NONPIVOTAL,
    RarityPivotalityStratum.RARE_PIVOTAL,
)
_STRATUM_FLAGS = {
    RarityPivotalityStratum.COMMON_NONPIVOTAL: (False, False),
    RarityPivotalityStratum.COMMON_PIVOTAL: (False, True),
    RarityPivotalityStratum.RARE_NONPIVOTAL: (True, False),
    RarityPivotalityStratum.RARE_PIVOTAL: (True, True),
}


class StratifiedQualifiedAssertionRecall(ImmutableRecord):
    stratum: RarityPivotalityStratum
    is_rare: bool
    is_pivotal: bool
    qualified_assertion_recall: RateResult
    assertion_target_ids: tuple[str, ...]
    matched_assertion_target_ids: tuple[str, ...]

    @model_validator(mode="after")
    def flags_ids_and_rate_agree(self) -> Self:
        if (self.is_rare, self.is_pivotal) != _STRATUM_FLAGS[self.stratum]:
            raise ValueError("rarity/pivotality flags do not match the named stratum")
        if any(not target_id for target_id in self.assertion_target_ids):
            raise ValueError("stratified assertion target IDs must be nonempty")
        if self.assertion_target_ids != tuple(sorted(set(self.assertion_target_ids))):
            raise ValueError("stratified assertion target IDs must be unique and sorted")
        if self.matched_assertion_target_ids != tuple(
            sorted(set(self.matched_assertion_target_ids))
        ):
            raise ValueError("matched stratified target IDs must be unique and sorted")
        if not set(self.matched_assertion_target_ids).issubset(self.assertion_target_ids):
            raise ValueError("matched stratified target IDs must belong to their stratum")
        if self.qualified_assertion_recall.denominator != len(self.assertion_target_ids):
            raise ValueError("stratified recall denominator does not match target count")
        if self.qualified_assertion_recall.numerator != len(self.matched_assertion_target_ids):
            raise ValueError("stratified recall numerator does not match matched target count")
        return self


class RarePivotalScore(ImmutableRecord):
    qualified_assertion_recall: RateResult
    complete_support_path_survival: RateResult
    stratified_qualified_assertion_recall: tuple[StratifiedQualifiedAssertionRecall, ...]
    rare_pivotal_target_ids: tuple[str, ...]
    matched_rare_pivotal_target_ids: tuple[str, ...]
    complete_path_target_ids: tuple[str, ...]
    invalid_semantic_output: bool

    @model_validator(mode="after")
    def registered_breakdown_and_top_level_counts_agree(self) -> Self:
        for name, target_ids in (
            ("rare-pivotal", self.rare_pivotal_target_ids),
            ("matched rare-pivotal", self.matched_rare_pivotal_target_ids),
            ("complete support-path", self.complete_path_target_ids),
        ):
            if target_ids != tuple(sorted(set(target_ids))):
                raise ValueError(f"{name} target IDs must be unique and sorted")
        if not set(self.matched_rare_pivotal_target_ids).issubset(self.rare_pivotal_target_ids):
            raise ValueError("matched rare-pivotal target IDs must belong to the gold targets")
        strata = tuple(item.stratum for item in self.stratified_qualified_assertion_recall)
        if strata != _REGISTERED_STRATA:
            raise ValueError("rare/common by pivotal/nonpivotal rows must use registered order")
        rare_pivotal_row = self.stratified_qualified_assertion_recall[-1]
        if rare_pivotal_row.assertion_target_ids != self.rare_pivotal_target_ids:
            raise ValueError("rare-pivotal target IDs disagree with stratified results")
        if rare_pivotal_row.matched_assertion_target_ids != self.matched_rare_pivotal_target_ids:
            raise ValueError("rare-pivotal matched IDs disagree with stratified results")
        if rare_pivotal_row.qualified_assertion_recall != self.qualified_assertion_recall:
            raise ValueError("rare-pivotal recall disagrees with stratified results")
        if not set(self.complete_path_target_ids).issubset(self.matched_rare_pivotal_target_ids):
            raise ValueError("a complete support path requires its rare-pivotal target match")
        if self.complete_support_path_survival.denominator != len(self.rare_pivotal_target_ids):
            raise ValueError("support-path denominator does not match rare-pivotal targets")
        if self.complete_support_path_survival.numerator != len(self.complete_path_target_ids):
            raise ValueError("support-path numerator does not match complete paths")
        rates = (
            self.qualified_assertion_recall,
            self.complete_support_path_survival,
            *(
                item.qualified_assertion_recall
                for item in self.stratified_qualified_assertion_recall
            ),
        )
        if self.invalid_semantic_output:
            if self.matched_rare_pivotal_target_ids or self.complete_path_target_ids:
                raise ValueError("invalid semantic output cannot retain accepted rare-pivotal hits")
            if any(
                result.denominator > 0 and result.status is not MetricStatus.INVALID_SEMANTIC_ZERO
                for result in rates
            ):
                raise ValueError("invalid semantic output requires explicit zero statuses")
        elif any(result.status is MetricStatus.INVALID_SEMANTIC_ZERO for result in rates):
            raise ValueError("valid semantic output cannot carry invalid-zero rate statuses")
        return self


def annotations_from_gold(
    gold: GoldContextualProjection,
) -> tuple[RarePivotalAnnotation, ...]:
    """Copy only scorer-side rare/pivotal labels into a scoring-only record."""

    return tuple(
        RarePivotalAnnotation(
            assertion_target_id=item.assertion_id,
            is_rare=item.is_rare,
            is_pivotal=item.is_pivotal,
            support_path_assertion_target_ids=item.support_path_assertion_ids,
        )
        for item in gold.assertion_annotations
    )


def score_rare_pivotal(
    *,
    annotations: Sequence[RarePivotalAnnotation],
    strictly_matched_assertion_target_ids: frozenset[str],
    invalid_semantic_output: bool = False,
) -> RarePivotalScore:
    """Require strict qualified matches and every assertion on each support path."""

    annotation_target_ids = tuple(item.assertion_target_id for item in annotations)
    if len(annotation_target_ids) != len(set(annotation_target_ids)):
        raise ValueError("rarity/pivotality annotations contain duplicate assertion targets")
    registered_target_ids = set(annotation_target_ids)
    unknown_support_path_ids = sorted(
        {
            support_target_id
            for item in annotations
            for support_target_id in item.support_path_assertion_target_ids
            if support_target_id not in registered_target_ids
        }
    )
    if unknown_support_path_ids:
        raise ValueError("support paths must reference registered annotation target IDs")
    rare_targets = tuple(
        sorted(item.assertion_target_id for item in annotations if item.is_rare and item.is_pivotal)
    )
    effective_matches = (
        frozenset() if invalid_semantic_output else strictly_matched_assertion_target_ids
    )
    matched = tuple(item for item in rare_targets if item in effective_matches)
    complete_paths = tuple(
        sorted(
            item.assertion_target_id
            for item in annotations
            if item.is_rare
            and item.is_pivotal
            and set(item.support_path_assertion_target_ids).issubset(effective_matches)
        )
    )
    stratified_rows = []
    for stratum in _REGISTERED_STRATA:
        is_rare, is_pivotal = _STRATUM_FLAGS[stratum]
        target_ids = tuple(
            sorted(
                item.assertion_target_id
                for item in annotations
                if item.is_rare is is_rare and item.is_pivotal is is_pivotal
            )
        )
        matched_target_ids = tuple(
            target_id for target_id in target_ids if target_id in effective_matches
        )
        stratified_rows.append(
            StratifiedQualifiedAssertionRecall(
                stratum=stratum,
                is_rare=is_rare,
                is_pivotal=is_pivotal,
                qualified_assertion_recall=rate(
                    len(matched_target_ids),
                    len(target_ids),
                    invalid_semantic_output=invalid_semantic_output,
                ),
                assertion_target_ids=target_ids,
                matched_assertion_target_ids=matched_target_ids,
            )
        )
    return RarePivotalScore(
        qualified_assertion_recall=rate(
            len(matched),
            len(rare_targets),
            invalid_semantic_output=invalid_semantic_output,
        ),
        complete_support_path_survival=rate(
            len(complete_paths),
            len(rare_targets),
            invalid_semantic_output=invalid_semantic_output,
        ),
        stratified_qualified_assertion_recall=tuple(stratified_rows),
        rare_pivotal_target_ids=rare_targets,
        matched_rare_pivotal_target_ids=matched,
        complete_path_target_ids=complete_paths,
        invalid_semantic_output=invalid_semantic_output,
    )


class ScorerLabelSet(ImmutableRecord):
    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    labels: Mapping[str, tuple[bool, bool]]


class RunArtifactFingerprint(ImmutableRecord):
    """Only generation-side hashes; scorer labels have no representable field."""

    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class GoldLabelFirewallAudit(ImmutableRecord):
    original_label_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutated_label_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_artifact_fingerprint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_changed: Literal[True] = True
    run_artifacts_unchanged: Literal[True] = True


def audit_gold_label_firewall(
    *,
    original_labels: ScorerLabelSet,
    mutated_labels: ScorerLabelSet,
    original_artifacts: RunArtifactFingerprint,
    artifacts_after_label_mutation: RunArtifactFingerprint,
) -> GoldLabelFirewallAudit:
    """Prove a scorer-label mutation changes labels but no generation artifact."""

    if original_labels.content_hash == mutated_labels.content_hash:
        raise ValueError("firewall mutation test did not change scorer labels")
    if original_artifacts != artifacts_after_label_mutation:
        raise ValueError("scorer-label mutation changed a run artifact")
    return GoldLabelFirewallAudit(
        original_label_hash=original_labels.content_hash,
        mutated_label_hash=mutated_labels.content_hash,
        run_artifact_fingerprint_hash=canonical_sha256(original_artifacts),
    )


__all__ = [
    "GoldLabelFirewallAudit",
    "RarePivotalAnnotation",
    "RarePivotalScore",
    "RarityPivotalityStratum",
    "RunArtifactFingerprint",
    "ScorerLabelSet",
    "StratifiedQualifiedAssertionRecall",
    "annotations_from_gold",
    "audit_gold_label_firewall",
    "score_rare_pivotal",
]
