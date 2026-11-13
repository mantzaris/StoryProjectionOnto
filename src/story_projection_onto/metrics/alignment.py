"""Condition-blind anchor alignment and registered semantic fidelity metrics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import (
    ConstructionOperator,
    EpistemicAttitude,
    GoldAlternativeSet,
    GoldContextualProjection,
    HolderRelativeTime,
    ImmutableRecord,
    InstanceGraph,
    LocalContextSchema,
    NarrativeCommitment,
    OntologyDecision,
    OntologyProjection,
    QualifiedAssertion,
    StoryTime,
    TemporalKind,
    ValidityTime,
    canonical_sha256,
)
from story_projection_onto.metrics.common import (
    PrecisionRecallF1,
    RateResult,
    maximum_cardinality_matching,
    precision_recall_f1,
    rate,
)


class AlignmentIntegrityError(ValueError):
    """A scorer-side gold/alternative/anchor contract is inconsistent."""


class NodeKind(StrEnum):
    ENTITY = "entity"
    EVENT = "event"


class AnchorKind(StrEnum):
    MENTION = "mention"
    EVIDENCE = "evidence"


class GroundingStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


QUALIFIED_GROUNDING_AUDIT_REVISION = "known-answer-qualified-evidence-support-v2"


class TemporalExtentSignature(ImmutableRecord):
    """Essential temporal semantics; display labels and explanatory prose are excluded."""

    kind: TemporalKind
    point: int | None = None
    start: int | None = None
    end: int | None = None
    anchor_id: str | None = None
    relation: str | None = None
    partial_order: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def from_extent(
        cls, extent: StoryTime | ValidityTime | HolderRelativeTime
    ) -> TemporalExtentSignature:
        return cls(
            kind=extent.kind,
            point=extent.point,
            start=extent.start,
            end=extent.end,
            anchor_id=extent.anchor_id,
            relation=extent.relation.value if extent.relation is not None else None,
            partial_order=tuple(
                sorted(
                    (item.left_id, item.relation.value, item.right_id)
                    for item in extent.partial_order
                )
            ),
        )


class NodeAlignmentTarget(ImmutableRecord):
    target_id: str = Field(min_length=1)
    kind: NodeKind
    anchor_kind: AnchorKind
    permissible_anchor_sets: tuple[tuple[str, ...], ...]
    qualifier_hashes: tuple[str, ...] = ()
    is_contextually_relevant: bool = True

    @model_validator(mode="after")
    def alternatives_are_nonempty_unique_sets(self) -> Self:
        if not self.permissible_anchor_sets:
            raise ValueError("node target needs at least one permissible anchor set")
        normalized = tuple(tuple(sorted(set(item))) for item in self.permissible_anchor_sets)
        if any(not item for item in normalized):
            raise ValueError("node anchor alternatives cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("node anchor alternatives must be unique")
        if self.permissible_anchor_sets != normalized:
            raise ValueError("node anchor alternatives must be sorted and deduplicated")
        if self.qualifier_hashes and len(self.qualifier_hashes) != len(
            self.permissible_anchor_sets
        ):
            raise ValueError("node qualifier hashes must correspond to every anchor alternative")
        return self


class PredictedNode(ImmutableRecord):
    prediction_id: str = Field(min_length=1)
    kind: NodeKind
    anchor_kind: AnchorKind
    anchor_ids: tuple[str, ...]
    qualifier_hash: str | None = None

    @model_validator(mode="after")
    def anchors_are_nonempty_and_normalized(self) -> Self:
        if not self.anchor_ids:
            raise ValueError("predicted nodes require evidence or mention anchors")
        if self.anchor_ids != tuple(sorted(set(self.anchor_ids))):
            raise ValueError("predicted node anchors must be sorted and unique")
        return self


class EpistemicSignature(ImmutableRecord):
    holder_target_id: str | None = None
    attitude: EpistemicAttitude | None = None
    holder_relative_time: TemporalExtentSignature | None = None
    narrative_commitment: NarrativeCommitment

    @model_validator(mode="after")
    def holder_and_attitude_are_joint(self) -> Self:
        holder_parts = (
            self.holder_target_id,
            self.attitude,
            self.holder_relative_time,
        )
        if any(item is None for item in holder_parts) != all(item is None for item in holder_parts):
            raise ValueError("epistemic holder, attitude, and relative time appear together")
        if self.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED:
            if self.holder_target_id is None:
                raise ValueError("holder-attributed signature requires a holder")
        elif (
            self.narrative_commitment is NarrativeCommitment.WORLD_COMMITTED
            and self.holder_target_id is not None
        ):
            raise ValueError("world-committed signatures cannot carry a holder scope")
        return self


class QualifiedAssertionSignature(ImmutableRecord):
    predicate: str = Field(min_length=1)
    direction: Literal["forward", "inverse"]
    subject_target_id: str | None = None
    object_target_id: str | None = None
    roles: tuple[tuple[str, str], ...] = ()
    story_time: TemporalExtentSignature
    validity_time: TemporalExtentSignature
    epistemic: EpistemicSignature

    @model_validator(mode="after")
    def binary_or_role_shape_is_exclusive(self) -> Self:
        binary = self.subject_target_id is not None and self.object_target_id is not None
        partial = (self.subject_target_id is None) != (self.object_target_id is None)
        if partial or binary == bool(self.roles):
            raise ValueError("normalized assertion requires exactly one binary or role shape")
        if self.roles != tuple(sorted(set(self.roles))):
            raise ValueError("normalized roles must be sorted and unique")
        return self


class PermissibleAssertionAlternative(ImmutableRecord):
    alternative_id: str = Field(min_length=1)
    signature: QualifiedAssertionSignature
    supporting_evidence_ids: tuple[str, ...]

    @model_validator(mode="after")
    def support_is_nonempty(self) -> Self:
        if not self.supporting_evidence_ids:
            raise ValueError("assertion alternative requires supporting evidence")
        if self.supporting_evidence_ids != tuple(sorted(set(self.supporting_evidence_ids))):
            raise ValueError("supporting evidence IDs must be sorted and unique")
        return self


class AssertionAlignmentTarget(ImmutableRecord):
    target_id: str = Field(min_length=1)
    alternatives: tuple[PermissibleAssertionAlternative, ...]
    essential_temporal: bool

    @model_validator(mode="after")
    def target_has_unique_alternatives(self) -> Self:
        if not self.alternatives:
            raise ValueError("assertion target requires a primary or permissible alternative")
        ids = tuple(item.alternative_id for item in self.alternatives)
        if len(ids) != len(set(ids)):
            raise ValueError("assertion alternative IDs must be unique")
        return self


class ExplicitNodeAlternative(ImmutableRecord):
    source_alternative_id: str = Field(min_length=1)
    target_id: str
    alternative_anchor_ids: tuple[str, ...]
    qualifier_hash: str | None = None

    @model_validator(mode="after")
    def anchors_are_canonical(self) -> Self:
        if not self.alternative_anchor_ids:
            raise ValueError("explicit node alternatives require anchors")
        if self.alternative_anchor_ids != tuple(sorted(set(self.alternative_anchor_ids))):
            raise ValueError("explicit node-alternative anchors must be sorted and unique")
        return self


class ExplicitAssertionAlternative(ImmutableRecord):
    source_alternative_id: str = Field(min_length=1)
    target_id: str
    alternative: PermissibleAssertionAlternative


class CompiledGoldAlternatives(ImmutableRecord):
    """Executable scorer targets compiled from every declared gold alternative."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    source_alternative_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_alternative_ids: tuple[str, ...]
    node_alternatives: tuple[ExplicitNodeAlternative, ...] = ()
    assertion_alternatives: tuple[ExplicitAssertionAlternative, ...] = ()

    @model_validator(mode="after")
    def consumed_ids_are_canonical(self) -> Self:
        if not self.consumed_alternative_ids:
            raise ValueError("compiled gold alternatives cannot be empty")
        if self.consumed_alternative_ids != tuple(sorted(set(self.consumed_alternative_ids))):
            raise ValueError("consumed gold-alternative IDs must be sorted and unique")
        if not self.node_alternatives and not self.assertion_alternatives:
            raise ValueError("compiled gold alternatives must alter at least one executable target")
        represented_ids = {
            *(item.source_alternative_id for item in self.node_alternatives),
            *(item.source_alternative_id for item in self.assertion_alternatives),
        }
        if represented_ids != set(self.consumed_alternative_ids):
            raise ValueError("every consumed gold alternative must alter an executable target")
        return self


class AlignmentPlan(ImmutableRecord):
    """Scorer-only, condition-free anchor plan compiled before evaluated outputs."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    matcher_revision: str = Field(min_length=1)
    source_gold_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_alternative_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    executed_alternative_ids: tuple[str, ...] = ()
    node_targets: tuple[NodeAlignmentTarget, ...]
    assertion_targets: tuple[AssertionAlignmentTarget, ...]

    @model_validator(mode="after")
    def target_ids_are_unique(self) -> Self:
        node_ids = tuple(item.target_id for item in self.node_targets)
        assertion_ids = tuple(item.target_id for item in self.assertion_targets)
        if len(node_ids) != len(set(node_ids)) or len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("alignment target IDs must be unique within target kind")
        node_id_set = set(node_ids)
        for target in self.assertion_targets:
            for alternative in target.alternatives:
                signature = alternative.signature
                endpoint_ids = {
                    item
                    for item in (signature.subject_target_id, signature.object_target_id)
                    if item is not None
                }
                endpoint_ids.update(item[1] for item in signature.roles)
                if signature.epistemic.holder_target_id is not None:
                    endpoint_ids.add(signature.epistemic.holder_target_id)
                if not endpoint_ids.issubset(node_id_set):
                    raise ValueError("assertion target references an unknown node target")
        return self


class PredictedAssertion(ImmutableRecord):
    prediction_id: str = Field(min_length=1)
    signature: QualifiedAssertionSignature
    evidence_ids: tuple[str, ...]
    valid_evidence_ids: tuple[str, ...]
    grounding_status: GroundingStatus

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> Self:
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ValueError("predicted assertion evidence IDs must be sorted and unique")
        if self.valid_evidence_ids != tuple(sorted(set(self.valid_evidence_ids))):
            raise ValueError("valid evidence IDs must be sorted and unique")
        if not set(self.valid_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("valid evidence IDs must be a subset of cited evidence IDs")
        if self.grounding_status is GroundingStatus.SUPPORTED and not self.valid_evidence_ids:
            raise ValueError("supported assertions require at least one valid evidence citation")
        return self


class AlignmentMatch(ImmutableRecord):
    prediction_id: str
    target_id: str
    alternative_id: str | None = None


class AlignmentResult(ImmutableRecord):
    node_score: PrecisionRecallF1
    strict_assertion_score: PrecisionRecallF1
    essential_temporal_accuracy: RateResult
    evidence_citation_validity: RateResult
    grounding_precision: RateResult
    unsupported_assertion_rate: RateResult
    all_anchor_node_matches: tuple[AlignmentMatch, ...]
    node_matches: tuple[AlignmentMatch, ...]
    strict_assertion_matches: tuple[AlignmentMatch, ...]
    structurally_aligned_assertion_matches: tuple[AlignmentMatch, ...]
    invalid_semantic_output: bool


PREDICATE_NORMALIZATION_REVISION = "development-lexical-normalization-v2"


def _normalize_predicate(value: str, aliases: Mapping[str, str]) -> str:
    if value in aliases:
        return aliases[value]
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    for prefix in ("contextual_", "actor_", "collective_", "event_role_"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
    # Development-frozen lexical equivalents, applied identically to gold and
    # every condition. This is predicate normalization, not relaxed assertion
    # matching: endpoints/roles, direction, time, epistemic scope and evidence
    # must still match. Appointment/succession events are NOT office states.
    lexical_aliases = {
        "served_as": "holds_office",
        "serves_as": "holds_office",
        # Direct causal witness uses "enabled"; the compiler labels that SAME
        # directed relation "causally_enables". Not equivalent to precedence,
        # correlation, support, or all relations under the causal upper parent.
        "enabled": "causally_enables",
        "enables": "causally_enables",
    }
    normalized = lexical_aliases.get(normalized, normalized)
    return aliases.get(normalized, normalized)


def _epistemic_signature(
    assertion: QualifiedAssertion,
    node_mapping: Mapping[str, str] | None = None,
) -> EpistemicSignature:
    if assertion.epistemic_scope is None:
        return EpistemicSignature(narrative_commitment=assertion.narrative_commitment)
    holder = assertion.epistemic_scope.holder_id
    if node_mapping is not None:
        holder = node_mapping.get(holder, f"unmatched:{holder}")
    return EpistemicSignature(
        holder_target_id=holder,
        attitude=assertion.epistemic_scope.attitude,
        holder_relative_time=TemporalExtentSignature.from_extent(
            assertion.epistemic_scope.holder_relative_time
        ),
        narrative_commitment=assertion.narrative_commitment,
    )


def _assertion_signature(
    assertion: QualifiedAssertion,
    *,
    predicate: str,
    node_mapping: Mapping[str, str] | None = None,
) -> QualifiedAssertionSignature:
    def aligned(object_id: str | None) -> str | None:
        if object_id is None or node_mapping is None:
            return object_id
        return node_mapping.get(object_id, f"unmatched:{object_id}")

    return QualifiedAssertionSignature(
        predicate=predicate,
        direction=assertion.direction,
        subject_target_id=aligned(assertion.subject_id),
        object_target_id=aligned(assertion.object_id),
        roles=tuple(sorted((role.role, aligned(role.object_id)) for role in assertion.roles)),
        story_time=TemporalExtentSignature.from_extent(assertion.temporal_scope.story_time),
        validity_time=TemporalExtentSignature.from_extent(assertion.temporal_scope.validity_time),
        epistemic=_epistemic_signature(assertion, node_mapping),
    )


def build_alignment_plan(
    gold: GoldContextualProjection,
    alternatives: GoldAlternativeSet,
    *,
    compiled_alternatives: CompiledGoldAlternatives,
    predicate_aliases: Mapping[str, str] | None = None,
    include_context_excluded_assertions: bool = False,
) -> AlignmentPlan:
    """Compile reviewed gold into executable anchor targets before condition scoring."""

    if alternatives.gold_projection_id != gold.gold_projection_id:
        raise AlignmentIntegrityError("gold and permissible-alternative set IDs differ")
    if alternatives.matcher_revision != gold.matcher_revision:
        raise AlignmentIntegrityError("gold and alternative matcher revisions differ")
    if compiled_alternatives.source_alternative_set_hash != alternatives.content_hash:
        raise AlignmentIntegrityError("compiled alternatives name a different source set")
    expected_alternative_ids = {
        *alternatives.permissible_projection_ids,
        *(item.alternative_id for item in alternatives.constraint_alternatives),
    }
    if set(compiled_alternatives.consumed_alternative_ids) != expected_alternative_ids:
        missing = expected_alternative_ids - set(compiled_alternatives.consumed_alternative_ids)
        extra = set(compiled_alternatives.consumed_alternative_ids) - expected_alternative_ids
        raise AlignmentIntegrityError(
            f"gold alternatives were not consumed exactly; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    aliases = predicate_aliases or {}
    relevance = {item.target_id: item.is_relevant for item in gold.relevance}
    extra_nodes: dict[str, list[ExplicitNodeAlternative]] = defaultdict(list)
    for item in compiled_alternatives.node_alternatives:
        extra_nodes[item.target_id].append(item)
    known_node_ids = {item.cluster_id for item in gold.entity_partition} | {
        item.event_id for item in gold.events
    }
    unknown_node_alternatives = set(extra_nodes) - known_node_ids
    if unknown_node_alternatives:
        raise AlignmentIntegrityError(
            f"node alternatives reference unknown targets: {sorted(unknown_node_alternatives)}"
        )

    node_targets: list[NodeAlignmentTarget] = []
    for cluster in gold.entity_partition:
        anchors = [tuple(sorted(cluster.mention_candidate_ids))]
        anchors.extend(
            tuple(sorted(item.alternative_anchor_ids)) for item in extra_nodes[cluster.cluster_id]
        )
        node_targets.append(
            NodeAlignmentTarget(
                target_id=cluster.cluster_id,
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                permissible_anchor_sets=tuple(anchors),
                is_contextually_relevant=relevance.get(cluster.cluster_id, True),
            )
        )
    for event in gold.events:
        primary_qualifier = canonical_sha256(
            TemporalExtentSignature.from_extent(event.occurrence_time)
        )
        anchors = [tuple(sorted(event.evidence_ids))]
        qualifiers = [primary_qualifier]
        for item in extra_nodes[event.event_id]:
            anchors.append(tuple(sorted(item.alternative_anchor_ids)))
            qualifiers.append(item.qualifier_hash or primary_qualifier)
        node_targets.append(
            NodeAlignmentTarget(
                target_id=event.event_id,
                kind=NodeKind.EVENT,
                anchor_kind=AnchorKind.EVIDENCE,
                permissible_anchor_sets=tuple(anchors),
                qualifier_hashes=tuple(qualifiers),
                is_contextually_relevant=relevance.get(event.event_id, True),
            )
        )

    predicate_by_id = {
        item.predicate_id: _normalize_predicate(item.label, aliases)
        for item in gold.local_schema.predicates
    }
    explicit_assertions: dict[str, list[PermissibleAssertionAlternative]] = defaultdict(list)
    for item in compiled_alternatives.assertion_alternatives:
        explicit_assertions[item.target_id].append(item.alternative)
    known_assertion_ids = {item.assertion_id for item in gold.qualified_assertions}
    unknown_assertion_alternatives = set(explicit_assertions) - known_assertion_ids
    if unknown_assertion_alternatives:
        raise AlignmentIntegrityError(
            "assertion alternatives reference unknown targets: "
            f"{sorted(unknown_assertion_alternatives)}"
        )
    assertion_targets: list[AssertionAlignmentTarget] = []
    for assertion in gold.qualified_assertions:
        if not include_context_excluded_assertions and not relevance.get(
            assertion.assertion_id, True
        ):
            continue
        predicate = predicate_by_id.get(
            assertion.predicate_id,
            _normalize_predicate(assertion.predicate_id, aliases),
        )
        primary = PermissibleAssertionAlternative(
            alternative_id=f"{assertion.assertion_id}:primary",
            signature=_assertion_signature(assertion, predicate=predicate),
            supporting_evidence_ids=tuple(sorted(assertion.evidence_ids)),
        )
        story_kind = assertion.temporal_scope.story_time.kind
        validity_kind = assertion.temporal_scope.validity_time.kind
        essential = story_kind not in {
            TemporalKind.UNKNOWN,
            TemporalKind.NOT_APPLICABLE,
            TemporalKind.HORIZON_WITHHELD,
        } or validity_kind not in {
            TemporalKind.UNKNOWN,
            TemporalKind.NOT_APPLICABLE,
            TemporalKind.HORIZON_WITHHELD,
        }
        assertion_targets.append(
            AssertionAlignmentTarget(
                target_id=assertion.assertion_id,
                alternatives=(primary, *explicit_assertions[assertion.assertion_id]),
                essential_temporal=essential,
            )
        )
    return AlignmentPlan(
        matcher_revision=gold.matcher_revision,
        source_gold_hash=gold.content_hash,
        source_alternative_set_hash=alternatives.content_hash,
        executed_alternative_ids=compiled_alternatives.consumed_alternative_ids,
        node_targets=tuple(node_targets),
        assertion_targets=tuple(assertion_targets),
    )


def prediction_records(
    projection: OntologyProjection,
    *,
    predicate_aliases: Mapping[str, str] | None,
    valid_evidence_ids: frozenset[str],
    grounding_by_assertion_id: Mapping[str, GroundingStatus],
    plan: AlignmentPlan,
) -> tuple[tuple[PredictedNode, ...], tuple[PredictedAssertion, ...], tuple[AlignmentMatch, ...]]:
    """Normalize one projection using only frozen mappings and gold anchor targets."""

    return prediction_records_from_components(
        local_schema=projection.local_schema,
        instance_graph=projection.instance_graph,
        predicate_aliases=predicate_aliases,
        valid_evidence_ids=valid_evidence_ids,
        grounding_by_assertion_id=grounding_by_assertion_id,
        plan=plan,
    )


def prediction_records_from_components(
    *,
    local_schema: LocalContextSchema,
    instance_graph: InstanceGraph,
    predicate_aliases: Mapping[str, str] | None,
    valid_evidence_ids: frozenset[str],
    grounding_by_assertion_id: Mapping[str, GroundingStatus],
    plan: AlignmentPlan,
) -> tuple[tuple[PredictedNode, ...], tuple[PredictedAssertion, ...], tuple[AlignmentMatch, ...]]:
    """Normalize a sealed draft without inventing a query-time projection.

    The development competence gate scores C1's comprehensive pre-query ontology
    against the union of its three contextual gold projections.  Constructing a
    synthetic :class:`OntologyProjection` merely to call the scorer would create
    fictitious packet, context, and access lineage.  This component-level entry
    point preserves the exact same normalization while accepting only the real
    schema and instance graph from the sealed preontology.
    """

    aliases = predicate_aliases or {}
    nodes: list[PredictedNode] = []
    for entity in instance_graph.entities:
        nodes.append(
            PredictedNode(
                prediction_id=entity.entity_id,
                kind=NodeKind.ENTITY,
                anchor_kind=AnchorKind.MENTION,
                anchor_ids=tuple(sorted(entity.supported_mention_candidate_ids)),
            )
        )
    for event in instance_graph.events:
        nodes.append(
            PredictedNode(
                prediction_id=event.event_id,
                kind=NodeKind.EVENT,
                anchor_kind=AnchorKind.EVIDENCE,
                anchor_ids=tuple(sorted(event.evidence_ids)),
                qualifier_hash=canonical_sha256(
                    TemporalExtentSignature.from_extent(event.occurrence_time)
                ),
            )
        )
    node_matches = _align_nodes(tuple(nodes), plan.node_targets)
    node_mapping = {item.prediction_id: item.target_id for item in node_matches}
    predicate_by_id = {
        item.predicate_id: _normalize_predicate(item.label, aliases)
        for item in local_schema.predicates
    }
    predicted_assertions = tuple(
        PredictedAssertion(
            prediction_id=assertion.assertion_id,
            signature=_assertion_signature(
                assertion,
                predicate=predicate_by_id.get(
                    assertion.predicate_id,
                    _normalize_predicate(assertion.predicate_id, aliases),
                ),
                node_mapping=node_mapping,
            ),
            evidence_ids=tuple(sorted(assertion.evidence_ids)),
            valid_evidence_ids=tuple(
                sorted(
                    evidence_id
                    for evidence_id in assertion.evidence_ids
                    if evidence_id in valid_evidence_ids
                )
            ),
            grounding_status=grounding_by_assertion_id.get(
                assertion.assertion_id, GroundingStatus.UNKNOWN
            ),
        )
        for assertion in instance_graph.assertions
    )
    return tuple(nodes), predicted_assertions, node_matches


def _align_nodes(
    predictions: Sequence[PredictedNode],
    targets: Sequence[NodeAlignmentTarget],
) -> tuple[AlignmentMatch, ...]:
    adjacency: dict[str, tuple[str, ...]] = {}
    for prediction in predictions:
        compatible: list[str] = []
        for target in targets:
            if (
                prediction.kind is not target.kind
                or prediction.anchor_kind is not target.anchor_kind
            ):
                continue
            matches = [
                index
                for index, anchors in enumerate(target.permissible_anchor_sets)
                if prediction.anchor_ids == anchors
            ]
            if not matches:
                continue
            if target.qualifier_hashes and not any(
                prediction.qualifier_hash == target.qualifier_hashes[index] for index in matches
            ):
                continue
            compatible.append(target.target_id)
        adjacency[prediction.prediction_id] = tuple(sorted(compatible))
    return tuple(
        AlignmentMatch(prediction_id=prediction_id, target_id=target_id)
        for prediction_id, target_id in maximum_cardinality_matching(adjacency)
    )


def _structural_signature(signature: QualifiedAssertionSignature) -> tuple[object, ...]:
    return (
        signature.predicate,
        signature.direction,
        signature.subject_target_id,
        signature.object_target_id,
        signature.roles,
    )


def audit_qualified_assertion_grounding(
    *,
    plan: AlignmentPlan,
    predicted_assertions: Sequence[PredictedAssertion],
) -> tuple[tuple[str, GroundingStatus], ...]:
    """Audit support using exact qualified semantics and relevant citations.

    Grounding is deliberately stricter than packet-membership citation validity.  A
    prediction is supported only when its complete qualified signature matches a
    permissible gold alternative, every emitted citation is inside the admissible
    evidence packet, and every emitted citation is registered as support for an
    exact-signature alternative.  Thus a correct endpoint pair cannot mask a wrong
    story time, validity interval, holder-relative attitude, or narrative
    commitment, and an unrelated in-packet citation cannot ride along for free.

    Equivalent permissible alternatives with the same signature contribute the
    union of their reviewed support IDs.  The audit is assertion-local: duplicate
    true statements remain grounded, while one-to-one cardinality is handled by the
    registered fidelity matcher rather than by the grounding label.
    """

    prediction_ids = tuple(item.prediction_id for item in predicted_assertions)
    if len(prediction_ids) != len(set(prediction_ids)):
        raise AlignmentIntegrityError("qualified grounding audit requires unique prediction IDs")

    support_by_signature: dict[str, set[str]] = defaultdict(set)
    signature_by_hash: dict[str, QualifiedAssertionSignature] = {}
    for target in plan.assertion_targets:
        for alternative in target.alternatives:
            signature_hash = alternative.signature.content_hash
            existing = signature_by_hash.setdefault(signature_hash, alternative.signature)
            if existing != alternative.signature:
                raise AlignmentIntegrityError("qualified grounding signature hash collision")
            support_by_signature[signature_hash].update(alternative.supporting_evidence_ids)

    statuses: list[tuple[str, GroundingStatus]] = []
    for prediction in predicted_assertions:
        cited = frozenset(prediction.evidence_ids)
        admissible_cited = frozenset(prediction.valid_evidence_ids)
        reviewed_support = support_by_signature.get(prediction.signature.content_hash, set())
        supported = (
            bool(cited)
            and cited == admissible_cited
            and prediction.signature == signature_by_hash.get(prediction.signature.content_hash)
            and cited.issubset(reviewed_support)
        )
        statuses.append(
            (
                prediction.prediction_id,
                GroundingStatus.SUPPORTED if supported else GroundingStatus.UNSUPPORTED,
            )
        )
    return tuple(sorted(statuses))


def _align_assertions(
    predictions: Sequence[PredictedAssertion],
    targets: Sequence[AssertionAlignmentTarget],
    *,
    strict: bool,
) -> tuple[AlignmentMatch, ...]:
    alternative_by_pair: dict[tuple[str, str], str] = {}
    adjacency: dict[str, tuple[str, ...]] = {}
    for prediction in predictions:
        compatible: list[str] = []
        for target in targets:
            for alternative in target.alternatives:
                signature_matches = (
                    prediction.signature == alternative.signature
                    if strict
                    else _structural_signature(prediction.signature)
                    == _structural_signature(alternative.signature)
                )
                evidence_matches = (
                    bool(
                        set(prediction.valid_evidence_ids)
                        & set(alternative.supporting_evidence_ids)
                    )
                    and prediction.grounding_status is GroundingStatus.SUPPORTED
                )
                if signature_matches and evidence_matches:
                    compatible.append(target.target_id)
                    alternative_by_pair[(prediction.prediction_id, target.target_id)] = (
                        alternative.alternative_id
                    )
                    break
        adjacency[prediction.prediction_id] = tuple(sorted(set(compatible)))
    return tuple(
        AlignmentMatch(
            prediction_id=prediction_id,
            target_id=target_id,
            alternative_id=alternative_by_pair[(prediction_id, target_id)],
        )
        for prediction_id, target_id in maximum_cardinality_matching(adjacency)
    )


def score_alignment(
    *,
    plan: AlignmentPlan,
    predicted_nodes: Sequence[PredictedNode],
    predicted_assertions: Sequence[PredictedAssertion],
    invalid_semantic_output: bool = False,
) -> AlignmentResult:
    """Score contextual nodes, strict qualified assertions, time, and grounding."""

    if invalid_semantic_output:
        all_node_matches: tuple[AlignmentMatch, ...] = ()
        strict_matches: tuple[AlignmentMatch, ...] = ()
        structural_matches: tuple[AlignmentMatch, ...] = ()
    else:
        all_node_matches = _align_nodes(predicted_nodes, plan.node_targets)
        strict_matches = _align_assertions(
            predicted_assertions, plan.assertion_targets, strict=True
        )
        structural_matches = _align_assertions(
            predicted_assertions, plan.assertion_targets, strict=False
        )
    relevant_target_ids = {
        item.target_id for item in plan.node_targets if item.is_contextually_relevant
    }
    node_matches = tuple(item for item in all_node_matches if item.target_id in relevant_target_ids)
    node_score = precision_recall_f1(
        true_positives=len(node_matches),
        predicted_count=len(predicted_nodes),
        gold_count=len(relevant_target_ids),
        invalid_semantic_output=invalid_semantic_output,
    )
    assertion_score = precision_recall_f1(
        true_positives=len(strict_matches),
        predicted_count=len(predicted_assertions),
        gold_count=len(plan.assertion_targets),
        invalid_semantic_output=invalid_semantic_output,
    )

    prediction_by_id = {item.prediction_id: item for item in predicted_assertions}
    target_by_id = {item.target_id: item for item in plan.assertion_targets}
    temporal_denominator = sum(item.essential_temporal for item in plan.assertion_targets)
    temporal_correct = 0
    if not invalid_semantic_output:
        for match in structural_matches:
            target = target_by_id[match.target_id]
            if not target.essential_temporal:
                continue
            prediction = prediction_by_id[match.prediction_id]
            alternative = next(
                item for item in target.alternatives if item.alternative_id == match.alternative_id
            )
            if (
                prediction.signature.story_time == alternative.signature.story_time
                and prediction.signature.validity_time == alternative.signature.validity_time
            ):
                temporal_correct += 1
    temporal_accuracy = rate(temporal_correct, temporal_denominator)

    if invalid_semantic_output:
        citation_validity = rate(0, 0)
        grounding_precision = rate(0, 0)
        unsupported_rate = rate(0, 0)
    else:
        citation_denominator = sum(len(item.evidence_ids) for item in predicted_assertions)
        valid_citations = sum(len(item.valid_evidence_ids) for item in predicted_assertions)
        supported = sum(
            item.grounding_status is GroundingStatus.SUPPORTED for item in predicted_assertions
        )
        unsupported = sum(
            item.grounding_status is GroundingStatus.UNSUPPORTED for item in predicted_assertions
        )
        citation_validity = rate(valid_citations, citation_denominator)
        grounding_precision = rate(supported, len(predicted_assertions))
        unsupported_rate = rate(unsupported, len(predicted_assertions))
    return AlignmentResult(
        node_score=node_score,
        strict_assertion_score=assertion_score,
        essential_temporal_accuracy=temporal_accuracy,
        evidence_citation_validity=citation_validity,
        grounding_precision=grounding_precision,
        unsupported_assertion_rate=unsupported_rate,
        all_anchor_node_matches=all_node_matches,
        node_matches=node_matches,
        strict_assertion_matches=strict_matches,
        structurally_aligned_assertion_matches=structural_matches,
        invalid_semantic_output=invalid_semantic_output,
    )


class DecisionFamily(StrEnum):
    MERGE_SPLIT = "merge_split"
    CONTEXTUAL_TYPE = "contextual_type"
    SCHEMA_RELATION = "schema_relation"
    EVENT_REIFICATION = "event_reification"
    ABSTRACTION = "abstraction"
    TEMPORAL_EPISTEMIC_QUALIFICATION = "temporal_epistemic_qualification"


_OPERATOR_FAMILY = {
    ConstructionOperator.MERGE: DecisionFamily.MERGE_SPLIT,
    ConstructionOperator.SPLIT: DecisionFamily.MERGE_SPLIT,
    ConstructionOperator.CONTEXTUAL_TYPE: DecisionFamily.CONTEXTUAL_TYPE,
    ConstructionOperator.SCHEMA_RELATION: DecisionFamily.SCHEMA_RELATION,
    ConstructionOperator.EVENT_REIFICATION: DecisionFamily.EVENT_REIFICATION,
    ConstructionOperator.ABSTRACTION: DecisionFamily.ABSTRACTION,
    ConstructionOperator.TEMPORAL_QUALIFICATION: (DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION),
    ConstructionOperator.EPISTEMIC_QUALIFICATION: (DecisionFamily.TEMPORAL_EPISTEMIC_QUALIFICATION),
}


class NormalizedDecision(ImmutableRecord):
    slot_key: str = Field(min_length=1)
    family: DecisionFamily
    operator: ConstructionOperator
    anchor_ids: tuple[str, ...]
    semantic_signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_is_canonical(self) -> Self:
        if self.slot_key.strip() != self.slot_key:
            raise ValueError("normalized decision slot keys must be stripped")
        if self.semantic_signature.strip() != self.semantic_signature:
            raise ValueError("normalized final-state signatures must be stripped")
        if not self.anchor_ids:
            raise ValueError("normalized decisions require mention/evidence anchors")
        if self.anchor_ids != tuple(sorted(set(self.anchor_ids))):
            raise ValueError("decision anchors must be sorted and unique")
        expected_family = _OPERATOR_FAMILY.get(self.operator)
        if expected_family is None:
            raise ValueError("selection/display operators are not ontology-decision targets")
        if self.family is not expected_family:
            raise ValueError("decision family must agree with its exact construction operator")
        if "=>" in self.semantic_signature:
            raise ValueError("final-state signatures cannot contain the reserved change delimiter")
        return self


def normalize_ontology_decisions(
    decisions: Sequence[OntologyDecision],
    *,
    slot_keys_by_decision_id: Mapping[str, str],
    semantic_signatures_by_decision_id: Mapping[str, str],
    object_anchors: Mapping[str, Sequence[str]] | None = None,
) -> tuple[NormalizedDecision, ...]:
    """Resolve condition-local decisions to explicit, common final-state semantics.

    Slot keys and final-state signatures must be derived from the emitted projection;
    falling back to an operator label would silently count different semantic decisions
    as equivalent.
    """

    resolved: list[NormalizedDecision] = []
    anchors_by_object = object_anchors or {}
    for decision in decisions:
        family = _OPERATOR_FAMILY.get(decision.operator)
        if family is None:
            continue
        if decision.decision_id not in slot_keys_by_decision_id:
            raise AlignmentIntegrityError(
                f"missing normalized slot key for decision {decision.decision_id!r}"
            )
        if decision.decision_id not in semantic_signatures_by_decision_id:
            raise AlignmentIntegrityError(
                f"missing final-state signature for decision {decision.decision_id!r}"
            )
        anchors: set[str] = set(decision.evidence_ids)
        for object_id in (
            *decision.input_object_ids,
            *decision.created_object_ids,
            *decision.removed_object_ids,
        ):
            anchors.update(anchors_by_object.get(object_id, ()))
        resolved.append(
            NormalizedDecision(
                slot_key=slot_keys_by_decision_id[decision.decision_id],
                family=family,
                operator=decision.operator,
                anchor_ids=tuple(sorted(anchors)),
                semantic_signature=semantic_signatures_by_decision_id[decision.decision_id],
            )
        )
    slot_keys = [item.slot_key for item in resolved]
    if len(slot_keys) != len(set(slot_keys)):
        raise AlignmentIntegrityError("normalized ontology decisions require unique slot keys")
    return tuple(resolved)


class DecisionComponentScore(ImmutableRecord):
    family: DecisionFamily
    score: PrecisionRecallF1
    active_in_gold: bool
    active_in_prediction: bool


class OntologyDecisionScore(ImmutableRecord):
    macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    active_family_count: int = Field(ge=0)
    components: tuple[DecisionComponentScore, ...]
    invalid_semantic_output: bool

    @model_validator(mode="after")
    def macro_matches_defined_components(self) -> Self:
        defined = [item.score.f1 for item in self.components if item.score.f1 is not None]
        if len(defined) != self.active_family_count:
            raise ValueError("active family count must equal non-NA decision components")
        expected = sum(defined) / len(defined) if defined else None
        if self.macro_f1 != expected:
            raise ValueError("ontology-decision macro must be the unweighted active-family mean")
        return self


def score_ontology_decisions(
    *,
    gold: Sequence[NormalizedDecision],
    predicted: Sequence[NormalizedDecision],
    invalid_semantic_output: bool = False,
) -> OntologyDecisionScore:
    """Compute the six registered components over union activation and their macro."""

    components: list[DecisionComponentScore] = []
    gold_counts = {
        family: Counter(
            (item.slot_key, item.operator, item.anchor_ids, item.semantic_signature)
            for item in gold
            if item.family is family
        )
        for family in DecisionFamily
    }
    predicted_counts = {
        family: Counter(
            (item.slot_key, item.operator, item.anchor_ids, item.semantic_signature)
            for item in predicted
            if item.family is family
        )
        for family in DecisionFamily
    }
    for family in DecisionFamily:
        gold_count = gold_counts[family]
        predicted_count = predicted_counts[family]
        true_positive_count = (
            0 if invalid_semantic_output else sum((gold_count & predicted_count).values())
        )
        score = precision_recall_f1(
            true_positives=true_positive_count,
            predicted_count=sum(predicted_count.values()),
            gold_count=sum(gold_count.values()),
            invalid_semantic_output=invalid_semantic_output,
        )
        components.append(
            DecisionComponentScore(
                family=family,
                score=score,
                active_in_gold=bool(gold_count),
                active_in_prediction=bool(predicted_count),
            )
        )
    defined = [item.score.f1 for item in components if item.score.f1 is not None]
    return OntologyDecisionScore(
        macro_f1=(sum(defined) / len(defined) if defined else None),
        active_family_count=len(defined),
        components=tuple(components),
        invalid_semantic_output=invalid_semantic_output,
    )
