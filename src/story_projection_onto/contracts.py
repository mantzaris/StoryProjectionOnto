"""Immutable records for evidence, ontology construction, and experiment lineage.

The contracts in this module deliberately keep query-blind evidence candidates
separate from constructed semantic objects.  Evidence records can contain text,
mentions, surface relation phrases, and temporal clues, but their schemas have no
field in which to hide a finalized entity partition, event, predicate, or
qualified assertion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    ),
]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


def _canonical_value(value: Any, *, exclude_content_hash: bool) -> Any:
    """Convert supported values to a deterministic JSON-compatible tree."""

    if isinstance(value, BaseModel):
        fields = value.__class__.model_fields
        return {
            name: _canonical_value(getattr(value, name), exclude_content_hash=exclude_content_hash)
            for name in sorted(fields)
            if not (exclude_content_hash and name == "content_hash")
        }
    if isinstance(value, Enum):
        return _canonical_value(value.value, exclude_content_hash=exclude_content_hash)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical timestamps must be timezone-aware")
        normalized = value.astimezone(UTC).isoformat(timespec="microseconds")
        return normalized.replace("+00:00", "Z")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {
            key: _canonical_value(value[key], exclude_content_hash=exclude_content_hash)
            for key in sorted(value)
            if not (exclude_content_hash and key == "content_hash")
        }
    if isinstance(value, (set, frozenset)):
        converted = [
            _canonical_value(item, exclude_content_hash=exclude_content_hash) for item in value
        ]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item, exclude_content_hash=exclude_content_hash) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any, *, include_content_hash: bool = True) -> str:
    """Serialize a record deterministically as UTF-8 JSON text."""

    normalized = _canonical_value(value, exclude_content_hash=not include_content_hash)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    """Hash the canonical payload, recursively excluding hash fields."""

    payload = canonical_json(value, include_content_hash=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# A concise alias is useful at artifact boundaries.
canonical_hash = canonical_sha256


def canonical_json_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    """Return a validation schema suitable for constrained generation and hashing."""

    schema = model_type.model_json_schema(mode="validation")
    # Round-tripping through canonical JSON also normalizes definitions and key order.
    return json.loads(canonical_json(schema))


class ImmutableRecord(BaseModel):
    """Base for versioned, content-addressed, immutable scientific records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    content_hash: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")

    @model_validator(mode="after")
    def populate_and_verify_content_hash(self) -> Self:
        expected = canonical_sha256(self)
        if self.content_hash and self.content_hash != expected:
            raise ValueError(
                f"content_hash mismatch: supplied {self.content_hash}, expected {expected}"
            )
        object.__setattr__(self, "content_hash", expected)
        return self

    def to_canonical_json(self) -> str:
        """Return the complete canonical serialization, including its checksum."""

        return canonical_json(self)


class ReleaseClass(StrEnum):
    PUBLIC = "public"
    RESTRICTED = "restricted"


class RightsClass(StrEnum):
    PUBLIC_SYNTHETIC = "public_synthetic"
    PUBLIC_DOMAIN = "public_domain"
    LICENSED_PUBLIC = "licensed_public"
    RESTRICTED_COPYRIGHTED = "restricted_copyrighted"


class ExplicitValueState(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    HORIZON_WITHHELD = "horizon_withheld"
    INVALID = "invalid"


class ConditionName(StrEnum):
    C0_CLASSICAL_PRE = "C0"
    C1_LLM_PRE = "C1"
    C2_LLM_QUERY = "C2"
    A_FIXED_SELECT = "A-FixedSelect"
    A_NO_CONTEXT = "A-NoContext"
    A_NO_TEMPORAL_EPISTEMIC = "A-NoTemporalEpistemic"
    A_NO_RARE_GUARD = "A-NoRareGuard"


ACTIVE_QUERY_CONSTRUCTION_CONDITIONS = frozenset(
    {
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ConditionName.A_NO_RARE_GUARD,
    }
)


class TemporalKind(StrEnum):
    POINT = "point"
    INTERVAL = "interval"
    RELATIVE = "relative"
    PARTIAL_ORDER = "partial_order"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    HORIZON_WITHHELD = "horizon_withheld"
    INVALID = "invalid"


class AllenRelation(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


class PartialOrderConstraint(ImmutableRecord):
    left_id: Identifier
    relation: AllenRelation
    right_id: Identifier


class _TemporalExtent(ImmutableRecord):
    """Common shape; semantic subclasses prevent accidental concept conflation."""

    kind: TemporalKind
    point: int | None = None
    start: int | None = None
    end: int | None = None
    label: str | None = None
    anchor_id: Identifier | None = None
    relation: AllenRelation | None = None
    partial_order: tuple[PartialOrderConstraint, ...] = ()
    reason: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        coordinate_fields = (self.point, self.start, self.end)
        if self.kind is TemporalKind.POINT:
            if self.point is None:
                raise ValueError("point time requires point")
            if any(value is not None for value in (self.start, self.end, self.anchor_id)):
                raise ValueError("point time cannot contain interval or relative fields")
            if self.relation is not None or self.partial_order:
                raise ValueError("point time cannot contain relation constraints")
        elif self.kind is TemporalKind.INTERVAL:
            if self.start is None and self.end is None:
                raise ValueError("interval time requires at least one bound")
            if self.point is not None or self.anchor_id is not None or self.relation is not None:
                raise ValueError("interval time cannot contain point or relative fields")
            if self.partial_order:
                raise ValueError("interval time cannot contain partial-order constraints")
        elif self.kind is TemporalKind.RELATIVE:
            if self.anchor_id is None or self.relation is None:
                raise ValueError("relative time requires anchor_id and relation")
            if any(value is not None for value in coordinate_fields) or self.partial_order:
                raise ValueError("relative time cannot contain coordinates or a partial order")
        elif self.kind is TemporalKind.PARTIAL_ORDER:
            if not self.partial_order:
                raise ValueError("partial-order time requires constraints")
            if any(value is not None for value in coordinate_fields):
                raise ValueError("partial-order time cannot contain numeric coordinates")
            if self.anchor_id is not None or self.relation is not None:
                raise ValueError("partial-order time cannot contain a relative anchor")
        else:
            if any(value is not None for value in coordinate_fields):
                raise ValueError(f"{self.kind.value} time cannot contain coordinates")
            if self.anchor_id is not None or self.relation is not None or self.partial_order:
                raise ValueError(f"{self.kind.value} time cannot contain order relations")
            if self.kind is not TemporalKind.NOT_APPLICABLE and not self.reason:
                raise ValueError(f"{self.kind.value} time requires an explicit reason")
        return self


class StoryTime(_TemporalExtent):
    """Time at which an event occurs in the story world."""


class ValidityTime(_TemporalExtent):
    """Interval or order over which a state/relation is valid."""


class HolderRelativeTime(_TemporalExtent):
    """Time associated with a holder's epistemic attitude."""


class DiscoursePosition(ImmutableRecord):
    passage_order: NonNegativeInt
    sentence_order: NonNegativeInt = 0
    token_order: NonNegativeInt = 0

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (self.passage_order, self.sentence_order, self.token_order)


class RevelationPosition(ImmutableRecord):
    revelation_order: NonNegativeInt
    label: str | None = None


class SpoilerHorizon(ImmutableRecord):
    horizon_id: Identifier
    max_discourse_position: DiscoursePosition
    max_revelation_position: RevelationPosition | None = None
    withheld_after_horizon: Literal[True] = True


class TemporalScope(ImmutableRecord):
    story_time: StoryTime
    validity_time: ValidityTime
    discourse_position: DiscoursePosition
    revelation_position: RevelationPosition


class ProvenanceReference(ImmutableRecord):
    provenance_id: Identifier
    evidence_id: Identifier
    extraction_method: NonEmptyText
    locator: NonEmptyText
    source_artifact_hash: Sha256Digest | None = None
    confidence: UnitInterval


class Document(ImmutableRecord):
    document_id: Identifier
    corpus_id: Identifier
    edition: NonEmptyText
    chapter_order: NonNegativeInt | None = None
    restricted_text_handle: NonEmptyText
    source_text_hash: Sha256Digest
    rights_class: RightsClass
    release_class: ReleaseClass

    @model_validator(mode="after")
    def enforce_rights_release(self) -> Self:
        if (
            self.rights_class is RightsClass.RESTRICTED_COPYRIGHTED
            and self.release_class is not ReleaseClass.RESTRICTED
        ):
            raise ValueError("copyrighted source records must be restricted")
        return self


class Passage(ImmutableRecord):
    passage_id: Identifier
    document_id: Identifier
    passage_order: NonNegativeInt
    restricted_text_handle: NonEmptyText
    source_text_hash: Sha256Digest
    discourse_position: DiscoursePosition
    release_class: ReleaseClass


class EvidenceSpan(ImmutableRecord):
    evidence_id: Identifier
    passage_id: Identifier
    start_char: NonNegativeInt
    end_char: PositiveInt
    surface_hash: Sha256Digest
    discourse_position: DiscoursePosition
    confidence: UnitInterval
    provenance: ProvenanceReference
    release_class: ReleaseClass

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("evidence span end_char must be greater than start_char")
        if self.provenance.evidence_id != self.evidence_id:
            raise ValueError("provenance must reference the containing evidence span")
        return self


class MentionCandidate(ImmutableRecord):
    candidate_id: Identifier
    evidence_id: Identifier
    start_char: NonNegativeInt
    end_char: PositiveInt
    surface: NonEmptyText
    surface_hash: Sha256Digest
    provisional_type: str | None = None
    alias_candidate_ids: tuple[Identifier, ...] = ()
    coreference_scores: dict[Identifier, UnitInterval] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mention(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("mention end_char must be greater than start_char")
        if hashlib.sha256(self.surface.encode("utf-8")).hexdigest() != self.surface_hash:
            raise ValueError("surface_hash does not match surface")
        return self


class EventCandidate(ImmutableRecord):
    candidate_id: Identifier
    evidence_id: Identifier
    trigger_start_char: NonNegativeInt
    trigger_end_char: PositiveInt
    trigger_surface: NonEmptyText
    participant_mention_candidate_ids: tuple[Identifier, ...] = ()
    confidence: UnitInterval

    @model_validator(mode="after")
    def validate_trigger_offsets(self) -> Self:
        if self.trigger_end_char <= self.trigger_start_char:
            raise ValueError("event trigger offsets are reversed or empty")
        return self


class RelationPhraseCandidate(ImmutableRecord):
    candidate_id: Identifier
    evidence_id: Identifier
    subject_mention_candidate_id: Identifier
    object_mention_candidate_id: Identifier | None = None
    surface_phrase: NonEmptyText
    confidence: UnitInterval


class TemporalClue(ImmutableRecord):
    clue_id: Identifier
    evidence_id: Identifier
    normalized_expression: NonEmptyText
    target_candidate_ids: tuple[Identifier, ...]
    relation: AllenRelation | None = None
    confidence: UnitInterval


class EvidenceRecord(ImmutableRecord):
    """A model-eligible passage fragment containing only defeasible candidates."""

    evidence_id: Identifier
    passage_id: Identifier
    text: NonEmptyText
    text_hash: Sha256Digest
    discourse_position: DiscoursePosition
    mention_candidates: tuple[MentionCandidate, ...] = ()
    event_candidates: tuple[EventCandidate, ...] = ()
    relation_phrase_candidates: tuple[RelationPhraseCandidate, ...] = ()
    temporal_clues: tuple[TemporalClue, ...] = ()
    provenance: ProvenanceReference
    confidence: UnitInterval
    release_class: ReleaseClass

    @model_validator(mode="after")
    def validate_evidence_record(self) -> Self:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_hash:
            raise ValueError("text_hash does not match text")
        children = (
            *self.mention_candidates,
            *self.event_candidates,
            *self.relation_phrase_candidates,
            *self.temporal_clues,
        )
        if any(child.evidence_id != self.evidence_id for child in children):
            raise ValueError("all evidence candidates must reference the containing evidence_id")
        if self.provenance.evidence_id != self.evidence_id:
            raise ValueError("provenance must reference the containing evidence_id")
        return self


class ModelVisibleEvidenceRecord(ImmutableRecord):
    """Strict allowlist for evidence passed to a model."""

    evidence_id: Identifier
    text: NonEmptyText
    discourse_position: DiscoursePosition
    mention_candidates: tuple[MentionCandidate, ...] = ()
    event_candidates: tuple[EventCandidate, ...] = ()
    relation_phrase_candidates: tuple[RelationPhraseCandidate, ...] = ()
    temporal_clues: tuple[TemporalClue, ...] = ()


MODEL_VISIBLE_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "text",
        "discourse_position",
        "mention_candidates",
        "event_candidates",
        "relation_phrase_candidates",
        "temporal_clues",
    }
)


def to_model_visible_evidence(record: EvidenceRecord) -> ModelVisibleEvidenceRecord:
    values = {name: getattr(record, name) for name in MODEL_VISIBLE_EVIDENCE_FIELDS}
    return ModelVisibleEvidenceRecord(**values)


class EvidenceSnapshot(ImmutableRecord):
    snapshot_id: Identifier
    corpus_id: Identifier
    world_or_window_id: Identifier
    horizon: SpoilerHorizon
    eligible_evidence_ids: tuple[Identifier, ...]
    index_config_hash: Sha256Digest
    created_at: AwareDatetime
    sealed_at: AwareDatetime
    release_class: ReleaseClass

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if not self.eligible_evidence_ids:
            raise ValueError("an evidence snapshot must contain eligible evidence")
        if len(set(self.eligible_evidence_ids)) != len(self.eligible_evidence_ids):
            raise ValueError("eligible evidence IDs must be unique")
        if self.sealed_at < self.created_at:
            raise ValueError("snapshot cannot be sealed before it is created")
        return self


class RetrievalMethod(StrEnum):
    ALL_ADMISSIBLE = "all_admissible"
    SQLITE_FTS5_BM25 = "sqlite_fts5_bm25"


class EvidencePacket(ImmutableRecord):
    packet_id: Identifier
    snapshot_hash: Sha256Digest
    evidence: tuple[EvidenceRecord, ...]
    ordered_evidence_ids: tuple[Identifier, ...]
    retrieval_method: RetrievalMethod
    ranks: dict[Identifier, PositiveInt] = Field(default_factory=dict)
    scores: dict[Identifier, float] = Field(default_factory=dict)
    token_count: NonNegativeInt
    horizon_rejections: tuple[Identifier, ...] = ()
    created_at: AwareDatetime
    release_class: ReleaseClass

    @model_validator(mode="after")
    def validate_packet(self) -> Self:
        evidence_ids = tuple(record.evidence_id for record in self.evidence)
        if not evidence_ids:
            raise ValueError("an evidence packet must contain evidence")
        if evidence_ids != self.ordered_evidence_ids:
            raise ValueError("evidence records must match ordered_evidence_ids exactly")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("packet evidence IDs must be unique")
        evidence_id_set = set(evidence_ids)
        if not set(self.ranks).issubset(evidence_id_set):
            raise ValueError("retrieval ranks contain IDs outside the packet")
        if not set(self.scores).issubset(evidence_id_set):
            raise ValueError("retrieval scores contain IDs outside the packet")
        if evidence_id_set.intersection(self.horizon_rejections):
            raise ValueError("horizon-rejected evidence cannot appear in the packet")
        if self.release_class is ReleaseClass.PUBLIC and any(
            item.release_class is ReleaseClass.RESTRICTED for item in self.evidence
        ):
            raise ValueError("a public packet cannot contain restricted evidence")
        return self


class ModelVisibleEvidencePacket(ImmutableRecord):
    packet_hash: Sha256Digest
    evidence: tuple[ModelVisibleEvidenceRecord, ...]
    ordered_evidence_ids: tuple[Identifier, ...]
    retrieval_method: RetrievalMethod

    @model_validator(mode="after")
    def validate_visible_packet_order(self) -> Self:
        if not self.evidence:
            raise ValueError("a model-visible evidence packet cannot be empty")
        if tuple(item.evidence_id for item in self.evidence) != self.ordered_evidence_ids:
            raise ValueError("visible evidence must match ordered_evidence_ids exactly")
        return self


def to_model_visible_packet(packet: EvidencePacket) -> ModelVisibleEvidencePacket:
    return ModelVisibleEvidencePacket(
        packet_hash=packet.content_hash,
        evidence=tuple(to_model_visible_evidence(record) for record in packet.evidence),
        ordered_evidence_ids=packet.ordered_evidence_ids,
        retrieval_method=packet.retrieval_method,
    )


_EVIDENCE_FORBIDDEN_KEYS = frozenset(
    {
        "canonical_entity_id",
        "entity_id",
        "event_id",
        "predicate_id",
        "assertion_id",
        "qualified_assertion",
        "local_context_schema",
        "local_schema",
        "instance_graph",
        "ontology",
        "ontology_projection",
        "query_relevance",
        "why_matters",
    }
)


def assert_evidence_boundary(value: BaseModel | Mapping[str, Any]) -> None:
    """Reject ontology-bearing keys from an evidence-side namespace."""

    data = value.model_dump(mode="python") if isinstance(value, BaseModel) else dict(value)

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized_key = str(key).casefold()
                child_path = f"{path}.{key}" if path else str(key)
                if normalized_key in _EVIDENCE_FORBIDDEN_KEYS:
                    raise ValueError(
                        f"constructed ontology field {key!r} is forbidden in evidence at "
                        f"{child_path}"
                    )
                visit(child, child_path)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(data, "")


class UpperOntology(ImmutableRecord):
    ontology_id: Identifier
    primitive_types: tuple[Identifier, ...]
    primitive_relations: tuple[Identifier, ...]
    temporal_terms: tuple[Identifier, ...]
    epistemic_terms: tuple[Identifier, ...]
    revision: NonEmptyText


class AbstractionLevel(StrEnum):
    ACTOR = "actor"
    EVENT_ROLE = "event_role"
    COLLECTIVE_CAUSAL_CHAIN = "collective_causal_chain"


class LocalTypeDefinition(ImmutableRecord):
    type_id: Identifier
    label: NonEmptyText
    definition: NonEmptyText
    parent_upper_type: Identifier
    abstraction: AbstractionLevel
    evidence_ids: tuple[Identifier, ...]


class LocalPredicateDefinition(ImmutableRecord):
    predicate_id: Identifier
    label: NonEmptyText
    definition: NonEmptyText
    arity: Annotated[int, Field(ge=2)]
    domain_type_ids: tuple[Identifier, ...] = ()
    range_type_ids: tuple[Identifier, ...] = ()
    role_names: tuple[Identifier, ...] = ()
    parent_upper_relation: Identifier
    evidence_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_predicate_arity(self) -> Self:
        if self.role_names and len(self.role_names) != self.arity:
            raise ValueError("role_names length must equal predicate arity")
        return self


class LocalContextSchema(ImmutableRecord):
    schema_id: Identifier
    contextual_types: tuple[LocalTypeDefinition, ...]
    predicates: tuple[LocalPredicateDefinition, ...]
    abstraction: AbstractionLevel

    @model_validator(mode="after")
    def unique_schema_ids(self) -> Self:
        type_ids = [item.type_id for item in self.contextual_types]
        predicate_ids = [item.predicate_id for item in self.predicates]
        if len(type_ids) != len(set(type_ids)):
            raise ValueError("contextual type IDs must be unique")
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValueError("local predicate IDs must be unique")
        return self


class Entity(ImmutableRecord):
    entity_id: Identifier
    label: NonEmptyText
    supported_mention_candidate_ids: tuple[Identifier, ...]
    aliases: tuple[str, ...] = ()
    contextual_type_id: Identifier
    contextual_role: NonEmptyText
    abstraction: AbstractionLevel
    temporal_state: StoryTime
    uncertainty: ExplicitValueState
    confidence: UnitInterval
    evidence_ids: tuple[Identifier, ...]
    description: NonEmptyText
    description_assertion_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def require_entity_support(self) -> Self:
        if not self.supported_mention_candidate_ids or not self.evidence_ids:
            raise ValueError("entities require mention and evidence support")
        if not self.description_assertion_ids:
            raise ValueError("entity descriptions require supporting assertion IDs")
        return self


class Event(ImmutableRecord):
    event_id: Identifier
    label: NonEmptyText
    contextual_type_id: Identifier
    occurrence_time: StoryTime
    reification_reason: NonEmptyText
    uncertainty: ExplicitValueState
    confidence: UnitInterval
    evidence_ids: tuple[Identifier, ...]
    description: NonEmptyText
    description_assertion_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def require_event_support(self) -> Self:
        if not self.evidence_ids:
            raise ValueError("events require evidence support")
        if not self.description_assertion_ids:
            raise ValueError("event descriptions require supporting assertion IDs")
        return self


class RoleBinding(ImmutableRecord):
    role: Identifier
    object_id: Identifier
    evidence_ids: tuple[Identifier, ...]


class PropositionContent(ImmutableRecord):
    """Content that is not thereby endorsed as global narrative truth."""

    proposition_content_id: Identifier
    predicate_id: Identifier
    subject_id: Identifier | None = None
    object_id: Identifier | None = None
    roles: tuple[RoleBinding, ...] = ()
    temporal_content: TemporalScope
    evidence_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_content_shape(self) -> Self:
        binary = self.subject_id is not None and self.object_id is not None
        partial_binary = (self.subject_id is None) != (self.object_id is None)
        if partial_binary or binary == bool(self.roles):
            raise ValueError("proposition content requires exactly one binary or n-ary shape")
        if self.roles and len(self.roles) < 2:
            raise ValueError("n-ary proposition content requires at least two roles")
        if not self.evidence_ids:
            raise ValueError("proposition content requires evidence")
        return self


class EpistemicAttitude(StrEnum):
    KNOWN = "known"
    BELIEVED = "believed"
    REPORTED = "reported"
    DENIED = "denied"
    UNCERTAIN = "uncertain"


class EpistemicScope(ImmutableRecord):
    holder_id: Identifier
    attitude: EpistemicAttitude
    proposition_content_id: Identifier
    holder_relative_time: HolderRelativeTime
    evidence_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def require_epistemic_evidence(self) -> Self:
        if not self.evidence_ids:
            raise ValueError("holder-level epistemic scope requires evidence")
        return self


class NarrativeCommitment(StrEnum):
    WORLD_COMMITTED = "world_committed"
    HOLDER_ATTRIBUTED = "holder_attributed"
    CONTESTED = "contested"
    UNKNOWN = "unknown"


class QualifiedAssertion(ImmutableRecord):
    assertion_id: Identifier
    proposition_content_id: Identifier | None = None
    predicate_id: Identifier
    subject_id: Identifier | None = None
    object_id: Identifier | None = None
    roles: tuple[RoleBinding, ...] = ()
    direction: Literal["forward", "inverse"] = "forward"
    temporal_scope: TemporalScope
    epistemic_scope: EpistemicScope | None = None
    narrative_commitment: NarrativeCommitment
    confidence: UnitInterval
    evidence_ids: tuple[Identifier, ...]
    provenance: tuple[ProvenanceReference, ...]
    contextual_relevance: UnitInterval
    why_matters: NonEmptyText
    why_matters_evidence_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_assertion_shape_and_commitment(self) -> Self:
        binary = self.subject_id is not None and self.object_id is not None
        partial_binary = (self.subject_id is None) != (self.object_id is None)
        if partial_binary or binary == bool(self.roles):
            raise ValueError("assertion requires exactly one binary or n-ary shape")
        if self.roles and len(self.roles) < 2:
            raise ValueError("n-ary assertions require at least two roles")
        if not self.evidence_ids or not self.provenance:
            raise ValueError("qualified assertions require evidence and provenance")
        if not self.why_matters_evidence_ids:
            raise ValueError("why_matters requires explicit evidence support")
        if not set(self.why_matters_evidence_ids).issubset(self.evidence_ids):
            raise ValueError("why_matters evidence must be assertion evidence")
        if any(item.evidence_id not in self.evidence_ids for item in self.provenance):
            raise ValueError("all provenance references must name assertion evidence")
        if self.epistemic_scope is None:
            if self.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED:
                raise ValueError("holder-attributed assertions require EpistemicScope")
            if self.proposition_content_id is not None:
                raise ValueError("proposition content references require EpistemicScope")
        else:
            if self.narrative_commitment is NarrativeCommitment.WORLD_COMMITTED:
                raise ValueError(
                    "an attributed proposition cannot be promoted to world truth; create a "
                    "separate world-committed assertion"
                )
            if self.proposition_content_id != self.epistemic_scope.proposition_content_id:
                raise ValueError("epistemic and assertion proposition references must match")
            if not set(self.epistemic_scope.evidence_ids).issubset(self.evidence_ids):
                raise ValueError("epistemic evidence must be assertion evidence")
        return self


class ValidationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"


class EvidenceSupportStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT = "insufficient"
    NOT_APPLICABLE = "not_applicable"


class TemporalDeterminationStatus(StrEnum):
    VALID = "valid"
    UNDERDETERMINED = "underdetermined"
    CONTRADICTION = "contradiction"
    NOT_APPLICABLE = "not_applicable"


class CommitmentCheckStatus(StrEnum):
    VALID = "valid"
    INVALID_PROMOTION = "invalid_promotion"
    CONTESTED = "contested"
    NOT_APPLICABLE = "not_applicable"


class ValidationRecord(ImmutableRecord):
    validation_id: Identifier
    target_id: Identifier
    validation_status: ValidationStatus
    evidence_support_status: EvidenceSupportStatus
    temporal_status: TemporalDeterminationStatus
    commitment_status: CommitmentCheckStatus
    diagnostics: tuple[NonEmptyText, ...] = ()
    repair_parent_hash: Sha256Digest | None = None
    repair_attempt: Annotated[int, Field(ge=0, le=1)] = 0
    validated_at: AwareDatetime


class OutputBudgets(ImmutableRecord):
    node_budget: PositiveInt
    assertion_budget: PositiveInt
    display_node_budget: PositiveInt
    display_assertion_budget: PositiveInt
    repair_attempt_budget: Annotated[int, Field(ge=0, le=1)] = 1

    @model_validator(mode="after")
    def display_budgets_do_not_exceed_semantic_budgets(self) -> Self:
        if self.display_node_budget > self.node_budget:
            raise ValueError("display_node_budget cannot exceed node_budget")
        if self.display_assertion_budget > self.assertion_budget:
            raise ValueError("display_assertion_budget cannot exceed assertion_budget")
        return self


class EpistemicViewpoint(ImmutableRecord):
    holder_id: Identifier
    attitude_scope: EpistemicAttitude | None = None


class QueryContext(ImmutableRecord):
    """Runner-side query record; scorer and contrast metadata have no fields here."""

    context_id: Identifier
    wording: NonEmptyText
    lens: NonEmptyText
    target: NonEmptyText
    story_scope: StoryTime
    spoiler_horizon: SpoilerHorizon
    viewpoint: EpistemicViewpoint | None = None
    abstraction: AbstractionLevel
    budgets: OutputBudgets
    revealed_at: AwareDatetime


class ModelVisibleQueryContext(ImmutableRecord):
    """Exact query-time allowlist passed to C2 and A-FixedSelect."""

    wording: NonEmptyText
    lens: NonEmptyText
    target: NonEmptyText
    story_scope: StoryTime
    spoiler_horizon: SpoilerHorizon
    viewpoint: EpistemicViewpoint | None = None
    abstraction: AbstractionLevel
    budgets: OutputBudgets


MODEL_VISIBLE_QUERY_FIELDS = frozenset(
    {
        "wording",
        "lens",
        "target",
        "story_scope",
        "spoiler_horizon",
        "viewpoint",
        "abstraction",
        "budgets",
    }
)


def to_model_visible_query(context: QueryContext) -> ModelVisibleQueryContext:
    values = {name: getattr(context, name) for name in MODEL_VISIBLE_QUERY_FIELDS}
    return ModelVisibleQueryContext(**values)


class ConstructionOperator(StrEnum):
    SELECTION = "selection"
    COMPRESSION = "compression"
    SUPPORTED_DESCRIPTION = "supported_description"
    INCLUDE_EXCLUDE = "include_exclude"
    MERGE = "merge"
    SPLIT = "split"
    CONTEXTUAL_TYPE = "contextual_type"
    SCHEMA_RELATION = "schema_relation"
    EVENT_REIFICATION = "event_reification"
    ABSTRACTION = "abstraction"
    TEMPORAL_QUALIFICATION = "temporal_qualification"
    EPISTEMIC_QUALIFICATION = "epistemic_qualification"
    RARE_PRESERVATION = "rare_preservation"


FIXED_SELECT_ALLOWED_OPERATORS = frozenset(
    {
        ConstructionOperator.SELECTION,
        ConstructionOperator.COMPRESSION,
        ConstructionOperator.SUPPORTED_DESCRIPTION,
    }
)
CONSTRUCTIVE_OPERATORS = frozenset(ConstructionOperator) - FIXED_SELECT_ALLOWED_OPERATORS


class OntologyDecision(ImmutableRecord):
    decision_id: Identifier
    operator: ConstructionOperator
    evidence_ids: tuple[Identifier, ...]
    rationale: NonEmptyText
    decided_at: AwareDatetime
    input_object_ids: tuple[Identifier, ...] = ()
    created_object_ids: tuple[Identifier, ...] = ()
    removed_object_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def decision_has_evidence(self) -> Self:
        if not self.evidence_ids:
            raise ValueError("ontology decisions require evidence")
        if self.operator in CONSTRUCTIVE_OPERATORS and not (
            self.created_object_ids or self.removed_object_ids
        ):
            raise ValueError("constructive decisions must identify created or removed objects")
        return self


class ConstructionCapabilities(ImmutableRecord):
    create_entities: bool
    merge_split: bool
    create_schema_predicates: bool
    reify_events: bool
    change_abstraction: bool
    add_temporal_qualification: bool
    add_epistemic_qualification: bool
    select_existing: bool = True
    compress_existing: bool = True
    write_supported_descriptions: bool = True

    @classmethod
    def active_construction(cls) -> ConstructionCapabilities:
        return cls(
            create_entities=True,
            merge_split=True,
            create_schema_predicates=True,
            reify_events=True,
            change_abstraction=True,
            add_temporal_qualification=True,
            add_epistemic_qualification=True,
        )

    @classmethod
    def fixed_selection(cls) -> ConstructionCapabilities:
        return cls(
            create_entities=False,
            merge_split=False,
            create_schema_predicates=False,
            reify_events=False,
            change_abstraction=False,
            add_temporal_qualification=False,
            add_epistemic_qualification=False,
        )


class InstanceGraph(ImmutableRecord):
    entities: tuple[Entity, ...]
    events: tuple[Event, ...]
    proposition_contents: tuple[PropositionContent, ...] = ()
    assertions: tuple[QualifiedAssertion, ...]

    @model_validator(mode="after")
    def graph_object_ids_are_unique(self) -> Self:
        object_ids = [item.entity_id for item in self.entities]
        object_ids.extend(item.event_id for item in self.events)
        object_ids.extend(item.proposition_content_id for item in self.proposition_contents)
        object_ids.extend(item.assertion_id for item in self.assertions)
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("instance graph object IDs must be globally unique")
        return self


class OmissionRecord(ImmutableRecord):
    evidence_id: Identifier
    reason: NonEmptyText
    confidence: UnitInterval


class BudgetAccounting(ImmutableRecord):
    nodes_used: NonNegativeInt
    assertions_used: NonNegativeInt
    display_nodes_used: NonNegativeInt
    display_assertions_used: NonNegativeInt
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt

    def validate_against(self, budgets: OutputBudgets) -> None:
        if self.nodes_used > budgets.node_budget:
            raise ValueError("node budget exceeded")
        if self.assertions_used > budgets.assertion_budget:
            raise ValueError("assertion budget exceeded")
        if self.display_nodes_used > budgets.display_node_budget:
            raise ValueError("display node budget exceeded")
        if self.display_assertions_used > budgets.display_assertion_budget:
            raise ValueError("display assertion budget exceeded")


class OntologyDraft(ImmutableRecord):
    contextual_interpretation: NonEmptyText
    local_schema: LocalContextSchema
    instance_graph: InstanceGraph
    decisions: tuple[OntologyDecision, ...]
    omissions: tuple[OmissionRecord, ...] = ()
    uncertainty_and_abstentions: tuple[NonEmptyText, ...] = ()
    budget_accounting: BudgetAccounting


class ConstructionSeal(ImmutableRecord):
    seal_id: Identifier
    condition: ConditionName
    snapshot_hash: Sha256Digest
    ontology_hash: Sha256Digest
    constructed_at: AwareDatetime
    sealed_at: AwareDatetime
    sealed_object_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def validate_prequery_seal(self) -> Self:
        if self.condition not in {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
        }:
            raise ValueError("only C0/C1 preconstructions can carry ConstructionSeal")
        if self.sealed_at < self.constructed_at:
            raise ValueError("construction cannot be sealed before completion")
        if not self.sealed_object_ids:
            raise ValueError("a construction seal must enumerate sealed objects")
        return self


class PreQueryInventory(ImmutableRecord):
    inventory_id: Identifier
    condition: ConditionName = ConditionName.C2_LLM_QUERY
    snapshot_hash: Sha256Digest
    recorded_at: AwareDatetime
    ontology_refs: tuple[Identifier, ...] = ()
    local_schema_ids: tuple[Identifier, ...] = ()
    entity_ids: tuple[Identifier, ...] = ()
    event_ids: tuple[Identifier, ...] = ()
    assertion_ids: tuple[Identifier, ...] = ()
    finalized_predicate_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def prove_empty_prequery_ontology(self) -> Self:
        if self.condition not in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            raise ValueError("pre-query empty inventories are for active query construction")
        forbidden_inventory = (
            self.ontology_refs,
            self.local_schema_ids,
            self.entity_ids,
            self.event_ids,
            self.assertion_ids,
            self.finalized_predicate_ids,
        )
        if any(forbidden_inventory):
            raise ValueError("C2 pre-query inventory must contain no constructed ontology")
        return self


class ConstructionCertificate(ImmutableRecord):
    certificate_id: Identifier
    condition: ConditionName
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    query_context_hash: Sha256Digest
    query_revealed_at: AwareDatetime
    completed_at: AwareDatetime
    decisions: tuple[OntologyDecision, ...]
    pre_query_inventory_hash: Sha256Digest | None = None
    inherited_construction_seal_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_construction_lineage(self) -> Self:
        if self.condition not in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS | {
            ConditionName.A_FIXED_SELECT
        }:
            raise ValueError("construction certificates are query-time C2/fixed-select records")
        if self.completed_at < self.query_revealed_at:
            raise ValueError("construction completion predates query reveal")
        if any(decision.decided_at < self.query_revealed_at for decision in self.decisions):
            raise ValueError("every query-time decision must follow query reveal")
        if self.condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            if self.pre_query_inventory_hash is None:
                raise ValueError("C2 certificate requires its empty pre-query inventory hash")
            if self.inherited_construction_seal_hash is not None:
                raise ValueError("C2 cannot inherit a preconstructed ontology seal")
            if not any(decision.operator in CONSTRUCTIVE_OPERATORS for decision in self.decisions):
                raise ValueError("C2 certificate must record a nonselection construction decision")
        else:
            if self.inherited_construction_seal_hash is None:
                raise ValueError("A-FixedSelect must cite its inherited C1 construction seal")
            if self.pre_query_inventory_hash is not None:
                raise ValueError("A-FixedSelect does not use a C2 empty inventory")
            forbidden = [
                decision.operator
                for decision in self.decisions
                if decision.operator not in FIXED_SELECT_ALLOWED_OPERATORS
            ]
            if forbidden:
                raise ValueError(
                    "A-FixedSelect certificate contains forbidden construction operators: "
                    + ", ".join(item.value for item in forbidden)
                )
        return self


class ParentProjectionRef(ImmutableRecord):
    """A same-context, post-reveal C2 revision parent—not an evidence substitute."""

    parent_projection_id: Identifier
    parent_projection_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    context_hash: Sha256Digest
    model_visible_context_hash: Sha256Digest
    parent_completed_at: AwareDatetime
    referenced_at: AwareDatetime
    relationship: Literal["same_context_revision_parent"] = "same_context_revision_parent"

    @model_validator(mode="after")
    def parent_must_precede_reference(self) -> Self:
        if self.referenced_at < self.parent_completed_at:
            raise ValueError("a parent projection cannot be referenced before it completes")
        return self


class OntologyProjection(ImmutableRecord):
    projection_id: Identifier
    condition: ConditionName
    snapshot_hash: Sha256Digest
    packet_hash: Sha256Digest
    context_hash: Sha256Digest
    upper_ontology: UpperOntology
    local_schema: LocalContextSchema
    instance_graph: InstanceGraph
    decisions: tuple[OntologyDecision, ...]
    omissions: tuple[OmissionRecord, ...] = ()
    validation_records: tuple[ValidationRecord, ...]
    budget_accounting: BudgetAccounting
    budgets: OutputBudgets
    construction_seal: ConstructionSeal | None = None
    pre_query_inventory: PreQueryInventory | None = None
    construction_certificate: ConstructionCertificate | None = None
    parent_projection_ref: ParentProjectionRef | None = None
    run_id: Identifier
    release_class: ReleaseClass

    @model_validator(mode="after")
    def validate_projection_lineage_and_budget(self) -> Self:
        self.budget_accounting.validate_against(self.budgets)
        if self.condition in {
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
        }:
            if self.construction_seal is None:
                raise ValueError("C0/C1 projections require a pre-query construction seal")
            if self.construction_seal.condition is not self.condition:
                raise ValueError("projection and construction-seal conditions differ")
            if self.construction_seal.snapshot_hash != self.snapshot_hash:
                raise ValueError("projection and construction-seal snapshots differ")
            if self.pre_query_inventory is not None or self.construction_certificate is not None:
                raise ValueError("C0/C1 projections cannot carry C2 query-time lineage")
            if self.parent_projection_ref is not None:
                raise ValueError("C0/C1 cannot use a C2 parent projection")
        elif self.condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            if self.pre_query_inventory is None or self.construction_certificate is None:
                raise ValueError("C2 requires empty inventory and construction certificate")
            if self.construction_seal is not None:
                raise ValueError("C2 cannot carry a preconstructed ontology seal")
            if self.pre_query_inventory.snapshot_hash != self.snapshot_hash:
                raise ValueError("projection and empty-inventory snapshots differ")
            if self.pre_query_inventory.condition is not self.condition:
                raise ValueError("projection and empty-inventory conditions differ")
            if self.construction_certificate.condition is not self.condition:
                raise ValueError("C2 projection and certificate conditions differ")
            if self.pre_query_inventory.content_hash != (
                self.construction_certificate.pre_query_inventory_hash
            ):
                raise ValueError("C2 certificate does not cite its pre-query inventory")
            if self.parent_projection_ref is not None:
                parent = self.parent_projection_ref
                if parent.snapshot_hash != self.snapshot_hash:
                    raise ValueError("C2 parent and child snapshots differ")
                if parent.packet_hash != self.packet_hash:
                    raise ValueError("C2 parent and child packets differ")
                if parent.context_hash != self.context_hash:
                    raise ValueError("C2 parent must belong to the exact same context")
        elif self.condition is ConditionName.A_FIXED_SELECT:
            if self.construction_seal is None or self.construction_certificate is None:
                raise ValueError("A-FixedSelect requires inherited seal and query certificate")
            if self.construction_seal.condition is not ConditionName.C1_LLM_PRE:
                raise ValueError("A-FixedSelect must inherit a C1 seal")
            if self.construction_seal.snapshot_hash != self.snapshot_hash:
                raise ValueError("projection and inherited-seal snapshots differ")
            if self.pre_query_inventory is not None:
                raise ValueError("A-FixedSelect cannot claim a C2 empty inventory")
            if self.parent_projection_ref is not None:
                raise ValueError("A-FixedSelect cannot inherit a C2 parent projection")
            if self.construction_certificate.inherited_construction_seal_hash != (
                self.construction_seal.content_hash
            ):
                raise ValueError("A-FixedSelect certificate does not cite the inherited C1 seal")
        certificate = self.construction_certificate
        if certificate is not None:
            if certificate.decisions != self.decisions:
                raise ValueError("projection decisions must equal certified decisions")
            if certificate.snapshot_hash != self.snapshot_hash:
                raise ValueError("projection and certificate snapshots differ")
            if certificate.packet_hash != self.packet_hash:
                raise ValueError("projection and certificate packets differ")
            if certificate.query_context_hash != self.context_hash:
                raise ValueError("projection and certificate contexts differ")
        return self


class BenchmarkSplit(StrEnum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class GoldReviewStatus(StrEnum):
    NOT_SELECTED = "not_selected"
    PENDING = "pending"
    REVIEWED = "reviewed"
    DISAGREEMENT_LOGGED = "disagreement_logged"


class GoldAdjudicationStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    ADJUDICATED = "adjudicated"


class GoldTargetKind(StrEnum):
    ENTITY_CLUSTER = "entity_cluster"
    EVENT = "event"
    ASSERTION = "assertion"


class GoldEntityCluster(ImmutableRecord):
    cluster_id: Identifier
    mention_candidate_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def cluster_has_unique_anchors(self) -> Self:
        if not self.mention_candidate_ids:
            raise ValueError("a gold entity cluster requires mention anchors")
        if len(self.mention_candidate_ids) != len(set(self.mention_candidate_ids)):
            raise ValueError("gold entity-cluster anchors must be unique")
        return self


class GoldRelevanceAnnotation(ImmutableRecord):
    target_id: Identifier
    target_kind: GoldTargetKind
    is_relevant: bool


class GoldAssertionAnnotation(ImmutableRecord):
    assertion_id: Identifier
    is_rare: bool
    is_pivotal: bool
    support_path_assertion_ids: tuple[Identifier, ...] = ()


class GoldCommunityAssignment(ImmutableRecord):
    anchor_id: Identifier
    community_id: Identifier


class GoldContrastDirection(StrEnum):
    ADD = "add"
    REMOVE = "remove"
    SUBSTITUTE = "substitute"


class GoldContrastDecision(ImmutableRecord):
    contrast_decision_id: Identifier
    operator: ConstructionOperator
    direction: GoldContrastDirection
    anchor_ids: tuple[Identifier, ...]
    expected_signature: NonEmptyText
    evidence_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def contrast_decision_has_anchors_and_evidence(self) -> Self:
        if not self.anchor_ids or not self.evidence_ids:
            raise ValueError("gold contrast decisions require anchors and evidence")
        return self


class GoldContrastInvariant(ImmutableRecord):
    invariant_id: Identifier
    anchor_ids: tuple[Identifier, ...]
    expected_signature: NonEmptyText
    evidence_ids: tuple[Identifier, ...]


class GoldContextualProjection(ImmutableRecord):
    """Scorer-only contextual truth; no model-visible conversion is provided."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    gold_projection_id: Identifier
    split: BenchmarkSplit
    world_id: Identifier
    query_id: Identifier
    local_schema: LocalContextSchema
    entity_partition: tuple[GoldEntityCluster, ...]
    events: tuple[Event, ...]
    qualified_assertions: tuple[QualifiedAssertion, ...]
    relevance: tuple[GoldRelevanceAnnotation, ...]
    assertion_annotations: tuple[GoldAssertionAnnotation, ...]
    communities: tuple[GoldCommunityAssignment, ...] = ()
    signed_contrast_decisions: tuple[GoldContrastDecision, ...] = ()
    contrast_invariants: tuple[GoldContrastInvariant, ...] = ()
    matcher_revision: NonEmptyText
    review_status: GoldReviewStatus
    adjudication_status: GoldAdjudicationStatus
    independent_review_record_hash: Sha256Digest | None = None
    adjudication_record_hash: Sha256Digest | None = None
    compiled_at: AwareDatetime
    release_class: ReleaseClass = ReleaseClass.PUBLIC

    @model_validator(mode="after")
    def validate_gold_integrity_and_review(self) -> Self:
        cluster_ids = [cluster.cluster_id for cluster in self.entity_partition]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("gold entity-cluster IDs must be unique")
        mention_ids = [
            mention_id
            for cluster in self.entity_partition
            for mention_id in cluster.mention_candidate_ids
        ]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("a mention anchor cannot belong to two gold entity clusters")

        assertion_ids = [item.assertion_id for item in self.qualified_assertions]
        annotation_ids = [item.assertion_id for item in self.assertion_annotations]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("gold assertion IDs must be unique")
        if set(annotation_ids) != set(assertion_ids) or len(annotation_ids) != len(
            set(annotation_ids)
        ):
            raise ValueError("every gold assertion requires exactly one rare/pivotal annotation")

        relevance_targets = [item.target_id for item in self.relevance]
        if len(relevance_targets) != len(set(relevance_targets)):
            raise ValueError("gold relevance targets must be unique")
        community_anchors = [item.anchor_id for item in self.communities]
        if len(community_anchors) != len(set(community_anchors)):
            raise ValueError("gold community anchors must be unique")

        reviewed = self.review_status in {
            GoldReviewStatus.REVIEWED,
            GoldReviewStatus.DISAGREEMENT_LOGGED,
        }
        if reviewed != (self.independent_review_record_hash is not None):
            raise ValueError("completed independent review requires exactly one review record hash")
        if (
            self.adjudication_status is GoldAdjudicationStatus.ADJUDICATED
            and self.adjudication_record_hash is None
        ):
            raise ValueError("adjudicated gold requires an adjudication record hash")
        elif self.adjudication_record_hash is not None:
            raise ValueError("non-adjudicated gold cannot carry an adjudication record hash")
        if (
            self.review_status is GoldReviewStatus.DISAGREEMENT_LOGGED
            and self.adjudication_status is GoldAdjudicationStatus.NOT_REQUIRED
        ):
            raise ValueError(
                "logged review disagreement requires pending or completed adjudication"
            )
        return self


class GoldConstraintOperator(StrEnum):
    EQUALS = "equals"
    ONE_OF = "one_of"
    ENTITY_PARTITION_EQUIVALENT = "entity_partition_equivalent"
    TEMPORALLY_EQUIVALENT = "temporally_equivalent"
    EPISTEMICALLY_EQUIVALENT = "epistemically_equivalent"


class GoldMatchingConstraint(ImmutableRecord):
    field_path: NonEmptyText
    operator: GoldConstraintOperator
    accepted_values: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def matching_constraint_has_values(self) -> Self:
        if not self.accepted_values:
            raise ValueError("a gold matching constraint requires accepted values")
        return self


class GoldConstraintAlternative(ImmutableRecord):
    alternative_id: Identifier
    description: NonEmptyText
    constraints: tuple[GoldMatchingConstraint, ...]

    @model_validator(mode="after")
    def alternative_has_constraints(self) -> Self:
        if not self.constraints:
            raise ValueError("a constraint-based gold alternative cannot be empty")
        return self


class GoldAlternativeSet(ImmutableRecord):
    """Scorer-only permissible equivalents; deliberately absent from request schemas."""

    scorer_namespace: Literal["scorer_only"] = "scorer_only"
    alternative_set_id: Identifier
    gold_projection_id: Identifier
    permissible_projection_ids: tuple[Identifier, ...] = ()
    constraint_alternatives: tuple[GoldConstraintAlternative, ...] = ()
    equivalence_rule: NonEmptyText
    matching_rule: NonEmptyText
    matcher_revision: NonEmptyText
    review_status: GoldReviewStatus
    adjudication_status: GoldAdjudicationStatus
    compiled_at: AwareDatetime
    release_class: ReleaseClass = ReleaseClass.PUBLIC

    @model_validator(mode="after")
    def alternative_set_is_nonempty(self) -> Self:
        if not self.permissible_projection_ids and not self.constraint_alternatives:
            raise ValueError("gold alternatives require a projection ID or constraints")
        return self


class RuntimeIdentifiers(ImmutableRecord):
    backend: Literal["vllm_gpu"] = "vllm_gpu"
    model_id: Identifier
    model_revision: GitRevision
    tokenizer_hash: Sha256Digest
    runtime_version: NonEmptyText
    prompt_hash: Sha256Digest
    output_schema_hash: Sha256Digest
    decoding_config_hash: Sha256Digest


class PreconstructionRequest(ImmutableRecord):
    request_id: Identifier
    condition: Literal[ConditionName.C1_LLM_PRE] = ConditionName.C1_LLM_PRE
    snapshot_hash: Sha256Digest
    evidence: tuple[ModelVisibleEvidenceRecord, ...]
    upper_ontology: UpperOntology
    budgets: OutputBudgets
    capabilities: ConstructionCapabilities
    runtime: RuntimeIdentifiers
    requested_at: AwareDatetime

    @model_validator(mode="after")
    def preconstruction_has_active_capabilities(self) -> Self:
        if self.capabilities != ConstructionCapabilities.active_construction():
            raise ValueError("C1 preconstruction requires the complete construction capability set")
        return self


class FixedOntologyInput(ImmutableRecord):
    construction_seal: ConstructionSeal
    upper_ontology: UpperOntology
    local_schema: LocalContextSchema
    instance_graph: InstanceGraph

    @model_validator(mode="after")
    def sealed_graph_matches_seal(self) -> Self:
        if self.construction_seal.condition is not ConditionName.C1_LLM_PRE:
            raise ValueError("fixed selection input must be a sealed C1 ontology")
        return self


class ConstructionRequest(ImmutableRecord):
    request_id: Identifier
    condition: ConditionName
    snapshot_hash: Sha256Digest
    packet: ModelVisibleEvidencePacket
    context: ModelVisibleQueryContext
    upper_ontology: UpperOntology
    budgets: OutputBudgets
    capabilities: ConstructionCapabilities
    runtime: RuntimeIdentifiers
    requested_at: AwareDatetime
    fixed_ontology: FixedOntologyInput | None = None
    model_visible_revisions: tuple[ModelVisibleRevision, ...] = ()
    same_context_parent: ParentProjectionRef | None = None

    @model_validator(mode="after")
    def validate_condition_capabilities(self) -> Self:
        if self.condition in ACTIVE_QUERY_CONSTRUCTION_CONDITIONS:
            if self.fixed_ontology is not None:
                raise ValueError("C2 cannot receive a hidden preconstructed ontology")
            if self.capabilities != ConstructionCapabilities.active_construction():
                raise ValueError("C2 requires the active construction capability set")
            if self.same_context_parent is not None:
                parent = self.same_context_parent
                if parent.snapshot_hash != self.snapshot_hash:
                    raise ValueError("C2 request and parent snapshots differ")
                if parent.packet_hash != self.packet.packet_hash:
                    raise ValueError("C2 request and parent packets differ")
                if parent.model_visible_context_hash != self.context.content_hash:
                    raise ValueError("C2 request parent belongs to a different context")
                if self.requested_at < parent.parent_completed_at:
                    raise ValueError("C2 request predates its parent projection")
        elif self.condition is ConditionName.A_FIXED_SELECT:
            if self.fixed_ontology is None:
                raise ValueError("A-FixedSelect requires the complete sealed C1 ontology")
            if self.capabilities != ConstructionCapabilities.fixed_selection():
                raise ValueError("A-FixedSelect capabilities must be selection-only")
            if self.same_context_parent is not None:
                raise ValueError("A-FixedSelect cannot consume a C2 parent projection")
        else:
            raise ValueError("ConstructionRequest is only valid for C2 and A-FixedSelect")
        if self.budgets != self.context.budgets:
            raise ValueError("request and model-visible context budgets differ")
        revision_sequences = [item.revision.sequence for item in self.model_visible_revisions]
        if revision_sequences != sorted(set(revision_sequences)):
            raise ValueError("model-visible revisions must have unique increasing sequences")
        if any(
            item.own_resolution is not None
            and item.own_resolution.receiving_condition is not self.condition
            for item in self.model_visible_revisions
        ):
            raise ValueError("a request may see only its receiving condition's resolution")
        return self


class FeedbackAction(StrEnum):
    REFINE_CONTEXT = "REFINE_CONTEXT"
    REQUEST_MERGE_SPLIT = "REQUEST_MERGE_SPLIT"


class FeedbackAnchor(ImmutableRecord):
    evidence_ids: tuple[Identifier, ...]
    mention_candidate_ids: tuple[Identifier, ...]
    requested_semantic_signature: NonEmptyText
    time_constraint: StoryTime | None = None
    role_constraint: Identifier | None = None

    @model_validator(mode="after")
    def feedback_anchor_is_condition_independent(self) -> Self:
        if not self.evidence_ids and not self.mention_candidate_ids:
            raise ValueError("feedback anchors require evidence or mention candidates")
        return self


class UserRevision(ImmutableRecord):
    revision_id: Identifier
    action: FeedbackAction
    anchors: tuple[FeedbackAnchor, ...]
    requested_change: NonEmptyText
    rationale: NonEmptyText
    sequence: PositiveInt
    before_context_hash: Sha256Digest
    after_context_hash: Sha256Digest
    created_at: AwareDatetime


class FeedbackResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    CAPABILITY_LIMITED = "capability_limited"
    INVALID = "invalid"


class FeedbackResolution(ImmutableRecord):
    resolution_id: Identifier
    revision_hash: Sha256Digest
    receiving_condition: ConditionName
    resolved_object_ids: tuple[Identifier, ...]
    status: FeedbackResolutionStatus
    before_projection_id: Identifier
    before_projection_hash: Sha256Digest
    after_projection_id: Identifier | None = None
    after_projection_hash: Sha256Digest | None = None
    resolver_hash: Sha256Digest
    seed: NonNegativeInt
    resolved_at: AwareDatetime

    @model_validator(mode="after")
    def projection_identity_is_complete(self) -> Self:
        if (self.after_projection_id is None) != (self.after_projection_hash is None):
            raise ValueError("after-projection ID and hash must be present or absent together")
        if self.status is FeedbackResolutionStatus.RESOLVED and self.after_projection_id is None:
            raise ValueError("a resolved revision requires its resulting projection")
        return self


class ModelVisibleRevision(ImmutableRecord):
    revision: UserRevision
    own_resolution: FeedbackResolution | None = None

    @model_validator(mode="after")
    def own_resolution_matches_revision(self) -> Self:
        if self.own_resolution is not None and (
            self.own_resolution.revision_hash != self.revision.content_hash
        ):
            raise ValueError("resolution must reference the visible revision")
        return self


class EvidenceBadge(ImmutableRecord):
    available: bool
    evidence_ids: tuple[Identifier, ...]
    count: NonNegativeInt

    @model_validator(mode="after")
    def evidence_availability_matches_ids(self) -> Self:
        if self.count != len(self.evidence_ids):
            raise ValueError("evidence badge count must equal its evidence ID count")
        if self.available != bool(self.evidence_ids):
            raise ValueError("evidence availability must match evidence IDs")
        return self


class VisualizationNode(ImmutableRecord):
    visualization_node_id: Identifier
    projection_object_id: Identifier
    projection_object_hash: Sha256Digest
    contextual_label: NonEmptyText
    contextual_type_id: Identifier
    contextual_role: NonEmptyText
    abstraction: AbstractionLevel
    temporal_state: StoryTime
    uncertainty: ExplicitValueState
    confidence: UnitInterval
    evidence_badge: EvidenceBadge
    description: NonEmptyText
    description_assertion_ids: tuple[Identifier, ...]
    release_class: ReleaseClass

    @model_validator(mode="after")
    def node_description_is_traceable(self) -> Self:
        if not self.description_assertion_ids:
            raise ValueError("visual node descriptions require projection assertion references")
        return self


class VisualizationAssertion(ImmutableRecord):
    visualization_assertion_id: Identifier
    projection_assertion_id: Identifier
    projection_assertion_hash: Sha256Digest
    source_visualization_node_id: Identifier | None = None
    target_visualization_node_id: Identifier | None = None
    roles: tuple[RoleBinding, ...] = ()
    contextual_label: NonEmptyText
    predicate_id: Identifier
    direction: Literal["forward", "inverse"]
    temporal_scope: TemporalScope
    uncertainty: ExplicitValueState
    epistemic_holder_id: Identifier | None = None
    epistemic_attitude: EpistemicAttitude | None = None
    confidence: UnitInterval
    evidence_badge: EvidenceBadge
    provenance: tuple[ProvenanceReference, ...]
    why_matters: NonEmptyText
    why_matters_assertion_id: Identifier
    release_class: ReleaseClass

    @model_validator(mode="after")
    def assertion_view_is_traceable(self) -> Self:
        binary = (
            self.source_visualization_node_id is not None
            and self.target_visualization_node_id is not None
        )
        partial_binary = (self.source_visualization_node_id is None) != (
            self.target_visualization_node_id is None
        )
        if partial_binary or binary == bool(self.roles):
            raise ValueError("visual assertion requires exactly one binary or n-ary shape")
        if self.roles and len(self.roles) < 2:
            raise ValueError("n-ary visual assertions require at least two roles")
        if (self.epistemic_holder_id is None) != (self.epistemic_attitude is None):
            raise ValueError("epistemic holder and attitude must be present or absent together")
        if self.why_matters_assertion_id != self.projection_assertion_id:
            raise ValueError("visual why_matters must cite its source projection assertion")
        if self.evidence_badge.available and not self.provenance:
            raise ValueError("available visual evidence requires provenance")
        badge_ids = set(self.evidence_badge.evidence_ids)
        if any(item.evidence_id not in badge_ids for item in self.provenance):
            raise ValueError("visual provenance must be represented by the evidence badge")
        return self


class NodePosition(ImmutableRecord):
    visualization_node_id: Identifier
    x: float
    y: float


class Viewport(ImmutableRecord):
    center_x: float
    center_y: float
    zoom: Annotated[float, Field(gt=0.0)]
    width: PositiveInt
    height: PositiveInt


class VisualizationTemporalFilter(ImmutableRecord):
    story_scope: StoryTime | None = None
    spoiler_horizon: SpoilerHorizon | None = None
    epistemic_holder_id: Identifier | None = None


class VisualizationState(ImmutableRecord):
    visualization_state_id: Identifier
    projection_hash: Sha256Digest
    semantic_hash: Sha256Digest
    layout_name: NonEmptyText
    layout_config_hash: Sha256Digest
    style_config_hash: Sha256Digest
    font_config_hash: Sha256Digest
    layout_seed: NonNegativeInt
    viewport: Viewport
    nodes: tuple[VisualizationNode, ...]
    assertions: tuple[VisualizationAssertion, ...]
    positions: tuple[NodePosition, ...]
    visible_node_ids: tuple[Identifier, ...]
    visible_assertion_ids: tuple[Identifier, ...]
    labels_visible: bool
    temporal_filter: VisualizationTemporalFilter
    release_class: ReleaseClass

    @model_validator(mode="after")
    def renderer_state_cannot_change_semantics(self) -> Self:
        if self.semantic_hash != self.projection_hash:
            raise ValueError("renderer semantic_hash must equal the immutable projection hash")
        node_ids = [item.visualization_node_id for item in self.nodes]
        assertion_ids = [item.visualization_assertion_id for item in self.assertions]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("visualization node IDs must be unique")
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("visualization assertion IDs must be unique")
        position_ids = [item.visualization_node_id for item in self.positions]
        if len(position_ids) != len(set(position_ids)) or set(position_ids) != set(node_ids):
            raise ValueError("every visualization node requires exactly one position")
        if not set(self.visible_node_ids).issubset(node_ids):
            raise ValueError("visible node IDs must reference visualization nodes")
        if not set(self.visible_assertion_ids).issubset(assertion_ids):
            raise ValueError("visible assertion IDs must reference visualization assertions")
        if any(
            assertion.source_visualization_node_id is not None
            and assertion.source_visualization_node_id not in node_ids
            for assertion in self.assertions
        ):
            raise ValueError("visual assertion sources must reference visualization nodes")
        if any(
            assertion.target_visualization_node_id is not None
            and assertion.target_visualization_node_id not in node_ids
            for assertion in self.assertions
        ):
            raise ValueError("visual assertion targets must reference visualization nodes")
        if self.release_class is ReleaseClass.PUBLIC and any(
            item.release_class is ReleaseClass.RESTRICTED
            for item in (*self.nodes, *self.assertions)
        ):
            raise ValueError("a public visualization cannot contain restricted DTOs")
        return self


class RunOutcome(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    INVALID = "invalid"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"


class RetryClass(StrEnum):
    LONG = "long"
    STANDARD = "standard"
    SHORT = "short"


class ArtifactHashReference(ImmutableRecord):
    artifact_name: Identifier
    artifact_hash: Sha256Digest
    release_class: ReleaseClass


class RunTiming(ImmutableRecord):
    started_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    allocated_gpu_seconds: NonNegativeFloat
    model_loading_seconds: NonNegativeFloat = 0.0
    warmup_seconds: NonNegativeFloat = 0.0
    schema_probe_seconds: NonNegativeFloat = 0.0
    inference_seconds: NonNegativeFloat = 0.0
    failed_request_seconds: NonNegativeFloat = 0.0
    repair_seconds: NonNegativeFloat = 0.0
    restart_seconds: NonNegativeFloat = 0.0
    runpod_session_wall_seconds: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_timing_accounting(self) -> Self:
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("run cannot end before it starts")
        classified_gpu_seconds = sum(
            (
                self.model_loading_seconds,
                self.warmup_seconds,
                self.schema_probe_seconds,
                self.inference_seconds,
                self.failed_request_seconds,
                self.repair_seconds,
                self.restart_seconds,
            )
        )
        if classified_gpu_seconds > self.allocated_gpu_seconds + 1e-9:
            raise ValueError("classified GPU activity exceeds allocated GPU service time")
        return self


class RunResourceUsage(ImmutableRecord):
    peak_gpu_vram_bytes: NonNegativeInt
    peak_process_ram_bytes: NonNegativeInt
    peak_project_storage_bytes: NonNegativeInt
    final_project_storage_bytes: NonNegativeInt
    minimum_free_storage_bytes: NonNegativeInt
    cpu_worker_threads: PositiveInt


class FailureLineage(ImmutableRecord):
    failure_id: Identifier
    failure_code: Identifier
    public_summary: NonEmptyText
    diagnostic_blob_hash: Sha256Digest | None = None
    failed_at: AwareDatetime
    parent_attempt_manifest_hash: Sha256Digest | None = None
    retry_class: RetryClass | None = None


class RunManifest(ImmutableRecord):
    manifest_id: Identifier
    condition: ConditionName | None = None
    run_role: NonEmptyText
    code_revision: GitRevision
    dirty_worktree: bool
    dirty_patch_hash: Sha256Digest | None = None
    environment_hash: Sha256Digest
    model_stack_status: ExplicitValueState
    model_stack_reason: str | None = None
    model_id: Identifier | None = None
    model_revision: GitRevision | None = None
    model_file_manifest_hash: Sha256Digest | None = None
    tokenizer_hash: Sha256Digest | None = None
    awq_config_hash: Sha256Digest | None = None
    runtime_hash: Sha256Digest | None = None
    chat_template_hash: Sha256Digest | None = None
    prompt_hash: Sha256Digest
    schema_hash: Sha256Digest
    config_hash: Sha256Digest
    seed: NonNegativeInt
    seed_manifest_hash: Sha256Digest
    input_artifacts: tuple[ArtifactHashReference, ...]
    output_artifacts: tuple[ArtifactHashReference, ...]
    timing: RunTiming
    resources: RunResourceUsage
    parent_manifest_hashes: tuple[Sha256Digest, ...] = ()
    release_class: ReleaseClass
    outcome: RunOutcome
    attempt_number: PositiveInt = 1
    retry_of_manifest_hash: Sha256Digest | None = None
    failure_lineage: FailureLineage | None = None

    @model_validator(mode="after")
    def validate_run_reproducibility_and_lineage(self) -> Self:
        model_fields = (
            self.model_id,
            self.model_revision,
            self.model_file_manifest_hash,
            self.tokenizer_hash,
            self.awq_config_hash,
            self.runtime_hash,
            self.chat_template_hash,
        )
        if self.model_stack_status is ExplicitValueState.KNOWN:
            if any(item is None for item in model_fields):
                raise ValueError("known model stack requires every immutable model/runtime field")
            if self.model_stack_reason is not None:
                raise ValueError("known model stack cannot carry an unknown-status reason")
        else:
            if any(item is not None for item in model_fields):
                raise ValueError("non-known model stack cannot carry model/runtime values")
            if self.model_stack_status is not ExplicitValueState.NOT_APPLICABLE and not (
                self.model_stack_reason
            ):
                raise ValueError("unknown, withheld, or invalid model stack requires a reason")

        if self.dirty_worktree != (self.dirty_patch_hash is not None):
            raise ValueError("dirty worktree status must match dirty patch hash presence")
        if self.attempt_number == 1 and self.retry_of_manifest_hash is not None:
            raise ValueError("a first attempt cannot cite a retry parent")
        if self.attempt_number > 1 and self.retry_of_manifest_hash is None:
            raise ValueError("a retry attempt must cite the prior manifest")

        terminal_failures = {
            RunOutcome.INVALID,
            RunOutcome.FAILED,
            RunOutcome.TIMED_OUT,
            RunOutcome.INTERRUPTED,
        }
        if (self.outcome in terminal_failures) != (self.failure_lineage is not None):
            raise ValueError("terminal failure outcome must match failure-lineage presence")
        if self.outcome is RunOutcome.SUCCEEDED and (
            self.timing.ended_at is None or not self.output_artifacts
        ):
            raise ValueError("successful runs require end time and output artifacts")
        if self.outcome in terminal_failures and self.timing.ended_at is None:
            raise ValueError("terminal failed runs require an end time")
        if self.outcome in {RunOutcome.PLANNED, RunOutcome.RUNNING} and (
            self.timing.ended_at is not None or self.output_artifacts
        ):
            raise ValueError("nonterminal runs cannot claim an end or final outputs")
        if self.release_class is ReleaseClass.PUBLIC and any(
            item.release_class is ReleaseClass.RESTRICTED
            for item in (*self.input_artifacts, *self.output_artifacts)
        ):
            raise ValueError("a public run manifest cannot expose restricted artifact references")
        return self


# Only these contracts may be emitted for model interaction.  Scorer-only gold
# records are intentionally absent and no ``to_model_visible_gold`` function exists.
MODEL_VISIBLE_SCHEMA_TYPES: tuple[type[ImmutableRecord], ...] = (
    ModelVisibleEvidenceRecord,
    ModelVisibleEvidencePacket,
    ModelVisibleQueryContext,
    ModelVisibleRevision,
    PreconstructionRequest,
    ConstructionRequest,
    OntologyDraft,
)

PUBLIC_SCHEMA_TYPES: tuple[type[ImmutableRecord], ...] = (
    EvidenceSnapshot,
    EvidencePacket,
    QueryContext,
    QualifiedAssertion,
    OntologyProjection,
    VisualizationNode,
    VisualizationAssertion,
    VisualizationState,
    RunManifest,
)

SCORER_ONLY_SCHEMA_TYPES: tuple[type[ImmutableRecord], ...] = (
    GoldContextualProjection,
    GoldAlternativeSet,
)

# Resolve the deliberately forward-declared model-visible revision type only
# after every feedback contract exists.
ConstructionRequest.model_rebuild()


def _scan_for_restricted_material(value: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, BaseModel):
        for name in value.__class__.model_fields:
            child = getattr(value, name)
            child_path = f"{path}.{name}" if path else name
            if name == "release_class" and child is ReleaseClass.RESTRICTED:
                findings.append(child_path)
            if name == "rights_class" and child is RightsClass.RESTRICTED_COPYRIGHTED:
                findings.append(child_path)
            findings.extend(_scan_for_restricted_material(child, child_path))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key == "release_class" and child in {
                ReleaseClass.RESTRICTED,
                ReleaseClass.RESTRICTED.value,
            }:
                findings.append(child_path)
            if key == "rights_class" and child in {
                RightsClass.RESTRICTED_COPYRIGHTED,
                RightsClass.RESTRICTED_COPYRIGHTED.value,
            }:
                findings.append(child_path)
            findings.extend(_scan_for_restricted_material(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            findings.extend(_scan_for_restricted_material(child, f"{path}[{index}]"))
    return findings


def assert_public_release(value: Any) -> None:
    """Raise when a proposed public artifact contains restricted-class records."""

    findings = _scan_for_restricted_material(value)
    if findings:
        raise ValueError("public release contains restricted material at: " + ", ".join(findings))
