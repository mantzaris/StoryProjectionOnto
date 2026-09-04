"""Signed nonselection contrast and paraphrase-stability metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    ConstructionOperator,
    GoldContrastDecision,
    GoldContrastDirection,
    GoldContrastInvariant,
    ImmutableRecord,
    canonical_sha256,
)
from story_projection_onto.metrics.adapters import AssertionSemanticRecord
from story_projection_onto.metrics.alignment import DecisionFamily, NormalizedDecision
from story_projection_onto.metrics.common import (
    PrecisionRecallF1,
    RateResult,
    precision_recall_f1,
    rate,
)


class ChangeDirection(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    SUBSTITUTE = "substitute"


class SignedDecisionChange(ImmutableRecord):
    family: DecisionFamily
    operator: ConstructionOperator
    direction: ChangeDirection
    anchor_ids: tuple[str, ...]
    semantic_signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def anchors_are_normalized(self) -> SignedDecisionChange:
        if self.semantic_signature.strip() != self.semantic_signature:
            raise ValueError("contrastive semantic signatures must be stripped")
        if not self.anchor_ids:
            raise ValueError("contrastive decisions require common anchors")
        if self.anchor_ids != tuple(sorted(set(self.anchor_ids))):
            raise ValueError("contrastive anchors must be sorted and unique")
        expected_family = {
            ConstructionOperator.MERGE: DecisionFamily.MERGE_SPLIT,
            ConstructionOperator.SPLIT: DecisionFamily.MERGE_SPLIT,
            ConstructionOperator.CONTEXTUAL_TYPE: DecisionFamily.CONTEXTUAL_TYPE,
            ConstructionOperator.SCHEMA_RELATION: DecisionFamily.SCHEMA_RELATION,
            ConstructionOperator.EVENT_REIFICATION: DecisionFamily.EVENT_REIFICATION,
            ConstructionOperator.ABSTRACTION: DecisionFamily.ABSTRACTION,
            ConstructionOperator.TEMPORAL_QUALIFICATION: (
                DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
            ),
            ConstructionOperator.EPISTEMIC_QUALIFICATION: (
                DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
            ),
        }.get(self.operator)
        if expected_family is None or self.family is not expected_family:
            raise ValueError("contrast family must agree with its exact construction operator")
        delimiter_count = self.semantic_signature.count("=>")
        if self.direction is ChangeDirection.SUBSTITUTE and delimiter_count != 1:
            raise ValueError("substitution signatures require exactly one old=>new transition")
        if self.direction is not ChangeDirection.SUBSTITUTE and delimiter_count:
            raise ValueError("add/remove signatures cannot contain the substitution delimiter")
        return self


def _unique_by_slot(
    decisions: Sequence[NormalizedDecision],
) -> dict[str, NormalizedDecision]:
    result: dict[str, NormalizedDecision] = {}
    for decision in decisions:
        if decision.slot_key in result:
            raise ValueError(f"duplicate normalized decision slot {decision.slot_key!r}")
        result[decision.slot_key] = decision
    return result


def derive_signed_changes(
    before: Sequence[NormalizedDecision],
    after: Sequence[NormalizedDecision],
) -> tuple[SignedDecisionChange, ...]:
    """Derive A-to-B changes; selection/display-only decisions are absent by construction."""

    before_by_slot = _unique_by_slot(before)
    after_by_slot = _unique_by_slot(after)
    changes: list[SignedDecisionChange] = []
    for slot_key in sorted(set(before_by_slot) | set(after_by_slot)):
        old = before_by_slot.get(slot_key)
        new = after_by_slot.get(slot_key)
        if (
            old is not None
            and new is not None
            and old.operator is new.operator
            and old.semantic_signature == new.semantic_signature
        ):
            continue
        if old is not None and new is not None:
            if old.family is not new.family:
                raise ValueError("one semantic slot cannot change across decision families")
            changes.append(
                SignedDecisionChange(
                    family=new.family,
                    operator=new.operator,
                    direction=ChangeDirection.SUBSTITUTE,
                    anchor_ids=tuple(sorted(set(old.anchor_ids) | set(new.anchor_ids))),
                    semantic_signature=f"{old.semantic_signature}=>{new.semantic_signature}",
                )
            )
        elif old is not None:
            changes.append(
                SignedDecisionChange(
                    family=old.family,
                    operator=old.operator,
                    direction=ChangeDirection.REMOVE,
                    anchor_ids=old.anchor_ids,
                    semantic_signature=old.semantic_signature,
                )
            )
        elif new is not None:
            changes.append(
                SignedDecisionChange(
                    family=new.family,
                    operator=new.operator,
                    direction=ChangeDirection.ADD,
                    anchor_ids=new.anchor_ids,
                    semantic_signature=new.semantic_signature,
                )
            )
    return tuple(changes)


def _family_from_gold(item: GoldContrastDecision) -> DecisionFamily:
    operator = item.operator.value
    if operator in {"merge", "split"}:
        return DecisionFamily.MERGE_SPLIT
    if operator == "contextual_type":
        return DecisionFamily.CONTEXTUAL_TYPE
    if operator == "schema_relation":
        return DecisionFamily.SCHEMA_RELATION
    if operator == "event_reification":
        return DecisionFamily.EVENT_REIFICATION
    if operator == "abstraction":
        return DecisionFamily.ABSTRACTION
    if operator in {"temporal_qualification", "epistemic_qualification"}:
        return DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION
    raise ValueError(f"gold contrast contains a selection-only operator: {operator}")


def normalize_gold_contrast(
    decisions: Sequence[GoldContrastDecision],
) -> tuple[SignedDecisionChange, ...]:
    """Strip gold IDs while retaining signed anchor-grounded expected changes."""

    direction_map = {
        GoldContrastDirection.ADD: ChangeDirection.ADD,
        GoldContrastDirection.REMOVE: ChangeDirection.REMOVE,
        GoldContrastDirection.SUBSTITUTE: ChangeDirection.SUBSTITUTE,
    }
    return tuple(
        SignedDecisionChange(
            family=_family_from_gold(item),
            operator=item.operator,
            direction=direction_map[item.direction],
            anchor_ids=tuple(sorted(set(item.anchor_ids) | set(item.evidence_ids))),
            semantic_signature=item.expected_signature,
        )
        for item in decisions
    )


class ContrastiveScore(ImmutableRecord):
    decision_change_score: PrecisionRecallF1
    gold_nonselection_change_count: int = Field(ge=0)
    predicted_nonselection_change_count: int = Field(ge=0)
    ontological_collapse: bool
    collapse_rate_contribution: float = Field(ge=0.0, le=1.0)
    invalid_semantic_output: bool

    @model_validator(mode="after")
    def collapse_definition_is_exact(self) -> ContrastiveScore:
        expected = (
            self.gold_nonselection_change_count > 0
            and self.predicted_nonselection_change_count == 0
        )
        if self.ontological_collapse != expected:
            raise ValueError("collapse must mean nonempty gold delta and empty predicted delta")
        if self.collapse_rate_contribution != float(expected):
            raise ValueError("per-pair collapse-rate contribution must be zero or one")
        return self


def score_contrastive_changes(
    *,
    gold: Sequence[SignedDecisionChange],
    predicted: Sequence[SignedDecisionChange],
    invalid_semantic_output: bool = False,
) -> ContrastiveScore:
    """Score only signed ontological changes; visible-node selection cannot avert collapse."""

    gold_counts = Counter(
        (item.family, item.operator, item.direction, item.anchor_ids, item.semantic_signature)
        for item in gold
    )
    raw_predicted_counts = Counter(
        (item.family, item.operator, item.direction, item.anchor_ids, item.semantic_signature)
        for item in predicted
    )
    predicted_counts = Counter() if invalid_semantic_output else raw_predicted_counts
    true_positives = (
        0 if invalid_semantic_output else sum((gold_counts & predicted_counts).values())
    )
    score = precision_recall_f1(
        true_positives=true_positives,
        predicted_count=sum(predicted_counts.values()),
        gold_count=sum(gold_counts.values()),
        invalid_semantic_output=invalid_semantic_output,
    )
    collapse = bool(gold_counts) and not predicted_counts
    return ContrastiveScore(
        decision_change_score=score,
        gold_nonselection_change_count=sum(gold_counts.values()),
        predicted_nonselection_change_count=sum(predicted_counts.values()),
        ontological_collapse=collapse,
        collapse_rate_contribution=float(collapse),
        invalid_semantic_output=invalid_semantic_output,
    )


class ContrastInvariantScore(ImmutableRecord):
    """Condition-blind preservation of registered unchanged semantic facts."""

    invariant_count: int = Field(ge=0)
    preserved_in_both_count: int = Field(ge=0)
    preservation_rate: RateResult
    evaluated_invariant_hashes: tuple[str, ...]
    missing_in_before_ids: tuple[str, ...]
    missing_in_after_ids: tuple[str, ...]

    @model_validator(mode="after")
    def counts_agree(self) -> ContrastInvariantScore:
        if self.preserved_in_both_count > self.invariant_count:
            raise ValueError("preserved invariants cannot exceed registered invariants")
        if self.preservation_rate.numerator != self.preserved_in_both_count:
            raise ValueError("invariant preservation numerator does not match")
        if self.preservation_rate.denominator != self.invariant_count:
            raise ValueError("invariant preservation denominator does not match")
        if len(self.evaluated_invariant_hashes) != self.invariant_count:
            raise ValueError("every registered invariant must have an audit hash")
        return self


def _matched_invariant_ids(
    invariants: Sequence[GoldContrastInvariant],
    assertions: Sequence[AssertionSemanticRecord],
) -> frozenset[str]:
    """Return exact, one-to-one final-state matches for registered invariants."""

    from story_projection_onto.metrics.common import maximum_cardinality_matching

    adjacency: dict[str, tuple[str, ...]] = {}
    for index, assertion in enumerate(assertions):
        adjacency[f"assertion:{index:06d}"] = tuple(
            sorted(
                invariant.invariant_id
                for invariant in invariants
                if set(invariant.anchor_ids).issubset(assertion.anchor_ids)
                and set(invariant.evidence_ids).issubset(assertion.evidence_ids)
                and assertion.semantic_signature == invariant.expected_signature
            )
        )
    return frozenset(target_id for _, target_id in maximum_cardinality_matching(adjacency))


def score_contrast_invariants(
    *,
    invariants: Sequence[GoldContrastInvariant],
    before_assertions: Sequence[AssertionSemanticRecord],
    after_assertions: Sequence[AssertionSemanticRecord],
    before_invalid: bool = False,
    after_invalid: bool = False,
) -> ContrastInvariantScore:
    """Consume scorer-only invariants and require support in both paired outputs."""

    invariant_ids = tuple(item.invariant_id for item in invariants)
    if len(invariant_ids) != len(set(invariant_ids)):
        raise ValueError("contrast invariants must have unique IDs")
    before_present = (
        frozenset()
        if before_invalid
        else _matched_invariant_ids(invariants, before_assertions)
    )
    after_present = (
        frozenset()
        if after_invalid
        else _matched_invariant_ids(invariants, after_assertions)
    )
    preserved = before_present & after_present
    return ContrastInvariantScore(
        invariant_count=len(invariants),
        preserved_in_both_count=len(preserved),
        preservation_rate=rate(len(preserved), len(invariants)),
        evaluated_invariant_hashes=tuple(sorted(canonical_sha256(item) for item in invariants)),
        missing_in_before_ids=tuple(sorted(set(invariant_ids) - before_present)),
        missing_in_after_ids=tuple(sorted(set(invariant_ids) - after_present)),
    )


class ContrastiveCollapseSummary(ImmutableRecord):
    pair_count: int = Field(ge=0)
    eligible_pair_count: int = Field(ge=0)
    collapse_count: int = Field(ge=0)
    collapse_rate: RateResult

    @model_validator(mode="after")
    def counts_agree(self) -> ContrastiveCollapseSummary:
        if self.eligible_pair_count > self.pair_count:
            raise ValueError("eligible contrast pairs cannot exceed all pairs")
        if self.collapse_count > self.eligible_pair_count:
            raise ValueError("collapse count cannot exceed eligible pairs")
        if (
            self.collapse_rate.numerator != self.collapse_count
            or self.collapse_rate.denominator != self.eligible_pair_count
        ):
            raise ValueError("collapse rate must expose count and eligible denominator")
        return self


def aggregate_contrastive_collapse(
    scores: Sequence[ContrastiveScore],
) -> ContrastiveCollapseSummary:
    eligible = tuple(item for item in scores if item.gold_nonselection_change_count > 0)
    collapse_count = sum(item.ontological_collapse for item in eligible)
    return ContrastiveCollapseSummary(
        pair_count=len(scores),
        eligible_pair_count=len(eligible),
        collapse_count=collapse_count,
        collapse_rate=rate(collapse_count, len(eligible)),
    )


class ParaphraseStabilityScore(ImmutableRecord):
    base_signature_size: int = Field(ge=0)
    paraphrase_signature_size: int = Field(ge=0)
    intersection_signature_size: int = Field(ge=0)
    union_signature_size: int = Field(ge=0)
    signature_divergence: float = Field(ge=0.0, le=1.0)
    base_strict_f1: float = Field(ge=0.0, le=1.0)
    paraphrase_strict_f1: float = Field(ge=0.0, le=1.0)
    strict_f1_change: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def values_match_declared_denominators(self) -> ParaphraseStabilityScore:
        if self.intersection_signature_size > min(
            self.base_signature_size,
            self.paraphrase_signature_size,
        ):
            raise ValueError("paraphrase signature intersection exceeds an input signature")
        expected_union = (
            self.base_signature_size
            + self.paraphrase_signature_size
            - self.intersection_signature_size
        )
        if self.union_signature_size != expected_union:
            raise ValueError("paraphrase signature union does not match input set sizes")
        expected_divergence = (
            0.0
            if self.union_signature_size == 0
            else 1.0 - self.intersection_signature_size / self.union_signature_size
        )
        if abs(self.signature_divergence - expected_divergence) > 1e-15:
            raise ValueError("paraphrase divergence does not match intersection/union counts")
        if abs(
            self.strict_f1_change
            - (self.paraphrase_strict_f1 - self.base_strict_f1)
        ) > 1e-15:
            raise ValueError("strict-F1 change does not match paraphrase minus base")
        return self


def score_paraphrase_stability(
    *,
    base_signature: frozenset[str],
    paraphrase_signature: frozenset[str],
    base_strict_f1: float,
    paraphrase_strict_f1: float,
) -> ParaphraseStabilityScore:
    """Report the two registered paraphrase outcomes without an unregistered threshold."""

    union = base_signature | paraphrase_signature
    divergence = 0.0 if not union else 1.0 - len(base_signature & paraphrase_signature) / len(union)
    return ParaphraseStabilityScore(
        base_signature_size=len(base_signature),
        paraphrase_signature_size=len(paraphrase_signature),
        intersection_signature_size=len(base_signature & paraphrase_signature),
        union_signature_size=len(union),
        signature_divergence=divergence,
        base_strict_f1=base_strict_f1,
        paraphrase_strict_f1=paraphrase_strict_f1,
        strict_f1_change=paraphrase_strict_f1 - base_strict_f1,
    )
