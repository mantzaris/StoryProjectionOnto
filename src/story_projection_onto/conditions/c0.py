"""Credible deterministic ``C0 ClassicalPre`` baseline.

The baseline constructs a compact conventional ontology before query reveal from
query-blind candidate records.  Candidate-aware rules are primary; a deterministic
regex fallback keeps hand-authored fixtures executable when a spaCy model is not
installed.  Query time performs only sealed-object relevance/path/support selection.
It never creates entities, predicates, events, identity decisions, abstractions, or
temporal/epistemic qualifications after reveal.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

from pydantic import Field, model_validator

from story_projection_onto.conditions.base import (
    ConditionAttemptRecord,
    ConditionIntegrityError,
    ConditionPreparation,
    ProduceInputs,
    SealedPreontology,
    preontology_semantic_hash,
    sealed_semantic_ids,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    BudgetAccounting,
    CommitmentCheckStatus,
    ConditionName,
    ConstructionOperator,
    ConstructionSeal,
    Entity,
    EpistemicAttitude,
    EpistemicScope,
    Event,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceSupportStatus,
    ExplicitValueState,
    HolderRelativeTime,
    ImmutableRecord,
    InstanceGraph,
    LocalContextSchema,
    LocalPredicateDefinition,
    LocalTypeDefinition,
    NarrativeCommitment,
    OntologyDecision,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    PropositionContent,
    QualifiedAssertion,
    ReleaseClass,
    RevelationPosition,
    RoleBinding,
    RunOutcome,
    StoryTime,
    TemporalDeterminationStatus,
    TemporalKind,
    TemporalScope,
    UpperOntology,
    ValidationRecord,
    ValidationStatus,
    ValidityTime,
)

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")
_NAME_PATTERN = re.compile(
    r"\b(?:(?:Captain|Commander|Doctor|Dr|King|Lady|Lord|Queen|Ser)\.?\s+)?"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)
_DAY_RANGE_PATTERN = re.compile(
    r"\bfrom\s+day\s+(\d+)\s+(?:to|through|until)\s+day\s+(\d+)\b",
    re.IGNORECASE,
)
_DAY_POINT_PATTERN = re.compile(r"\b(?:on\s+)?day\s+(\d+)\b", re.IGNORECASE)
_PRONOUNS = frozenset({"he", "her", "hers", "him", "his", "she", "they", "them", "their"})
_NAME_STOPWORDS = frozenset(
    {
        "After",
        "Before",
        "Day",
        "During",
        "From",
        "Later",
        "Meanwhile",
        "On",
        "The",
        "Then",
        "Through",
        "Until",
        "When",
    }
)


def _identifier(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:20]}"


def _words(value: str) -> frozenset[str]:
    return frozenset(match.group(0).casefold() for match in _WORD_PATTERN.finditer(value))


def _normalized_surface(surface: str, honorifics: frozenset[str]) -> str:
    tokens = [token.casefold().rstrip(".") for token in _WORD_PATTERN.findall(surface)]
    while tokens and tokens[0] in honorifics:
        tokens.pop(0)
    return " ".join(tokens)


class ClassicalEntityKind(StrEnum):
    PERSON = "person"
    COLLECTIVE = "collective"
    PLACE = "place"
    OTHER = "entity"


class ClassicalMention(ImmutableRecord):
    mention_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    surface: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    entity_kind: ClassicalEntityKind
    alias_candidate_ids: tuple[str, ...] = ()
    is_pronoun: bool = False
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def offsets_are_ordered(self) -> ClassicalMention:
        if self.end_char <= self.start_char:
            raise ValueError("classical mention offsets must be ordered")
        return self


class ClassicalRelation(ImmutableRecord):
    relation_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    subject_mention_id: str = Field(min_length=1)
    object_mention_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    surface_phrase: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ClassicalEvent(ImmutableRecord):
    event_candidate_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    participant_mention_ids: tuple[str, ...]
    confidence: float = Field(ge=0.0, le=1.0)


class ClassicalEvidenceAnalysis(ImmutableRecord):
    evidence_id: str = Field(min_length=1)
    mentions: tuple[ClassicalMention, ...]
    relations: tuple[ClassicalRelation, ...]
    events: tuple[ClassicalEvent, ...]
    temporal_expressions: tuple[str, ...]
    dependency_backend: str = Field(min_length=1)


class ClassicalRuleConfig(ImmutableRecord):
    """Frozen development-tunable rule vocabulary for the classical baseline."""

    config_id: str = "c0-rules-v1"
    honorifics: frozenset[str] = frozenset(
        {"captain", "commander", "doctor", "dr", "king", "lady", "lord", "queen", "ser"}
    )
    collective_markers: frozenset[str] = frozenset(
        {"alliance", "army", "clan", "company", "council", "crew", "guild", "house", "team"}
    )
    place_markers: frozenset[str] = frozenset(
        {"bay", "bridge", "castle", "city", "fort", "harbor", "island", "river", "road", "tower"}
    )
    event_triggers: frozenset[str] = frozenset(
        {
            "arrived",
            "attacked",
            "betrayed",
            "closed",
            "departed",
            "died",
            "fled",
            "joined",
            "left",
            "met",
            "opened",
            "reported",
            "rescued",
            "revealed",
            "warned",
        }
    )
    relation_lemmas: Mapping[str, str] = Field(
        default_factory=lambda: {
            "allied with": "allied_with",
            "believed": "believed",
            "belongs to": "member_of",
            "denied": "denied",
            "followed": "followed",
            "joined": "member_of",
            "knows": "knows",
            "led": "led",
            "left": "left",
            "located at": "located_at",
            "opposed": "opposed",
            "reported": "reported",
            "supported": "supported",
            "trusted": "trusted",
            "warned": "warned",
        }
    )
    alias_suffix_merge: bool = True
    local_pronoun_coreference: bool = True
    low_frequency_support_bonus: float = Field(default=0.12, ge=0.0, le=1.0)
    typed_path_continuity_bonus: float = Field(default=0.15, ge=0.0, le=1.0)


class ClassicalCandidateBackend(Protocol):
    """Hook implemented by spaCy-backed or deterministic candidate analyzers."""

    backend_name: str

    def analyze(
        self,
        evidence: EvidenceRecord,
        config: ClassicalRuleConfig,
    ) -> ClassicalEvidenceAnalysis:
        """Return query-blind candidates; never return final ontology objects."""


def _classify_entity(
    surface: str, provisional_type: str | None, config: ClassicalRuleConfig
) -> ClassicalEntityKind:
    type_hint = (provisional_type or "").casefold()
    if any(token in type_hint for token in ("person", "actor", "character", "human")):
        return ClassicalEntityKind.PERSON
    if any(token in type_hint for token in ("collective", "group", "organization", "army")):
        return ClassicalEntityKind.COLLECTIVE
    if any(token in type_hint for token in ("place", "location", "geo")):
        return ClassicalEntityKind.PLACE
    surface_words = _words(surface)
    if surface_words & config.collective_markers:
        return ClassicalEntityKind.COLLECTIVE
    if surface_words & config.place_markers:
        return ClassicalEntityKind.PLACE
    if surface.casefold() in _PRONOUNS:
        return ClassicalEntityKind.PERSON
    return ClassicalEntityKind.PERSON


def _normalize_predicate(surface: str, config: ClassicalRuleConfig) -> str:
    folded = " ".join(_WORD_PATTERN.findall(surface.casefold()))
    for phrase, predicate in sorted(
        config.relation_lemmas.items(), key=lambda item: (-len(item[0]), item[0])
    ):
        if phrase in folded:
            return predicate
    if folded:
        return folded.replace(" ", "_")
    return "related_to"


class RuleCandidateBackend:
    """Candidate-aware fallback used when no full spaCy pipeline is supplied."""

    backend_name = "deterministic-candidate-regex-v1"

    def analyze(
        self,
        evidence: EvidenceRecord,
        config: ClassicalRuleConfig,
    ) -> ClassicalEvidenceAnalysis:
        mentions: list[ClassicalMention] = []
        if evidence.mention_candidates:
            for candidate in sorted(
                evidence.mention_candidates,
                key=lambda item: (item.start_char, item.end_char, item.candidate_id),
            ):
                mentions.append(
                    ClassicalMention(
                        mention_id=candidate.candidate_id,
                        evidence_id=evidence.evidence_id,
                        surface=candidate.surface,
                        start_char=candidate.start_char,
                        end_char=candidate.end_char,
                        entity_kind=_classify_entity(
                            candidate.surface, candidate.provisional_type, config
                        ),
                        alias_candidate_ids=candidate.alias_candidate_ids,
                        is_pronoun=candidate.surface.casefold() in _PRONOUNS,
                        confidence=max(candidate.coreference_scores.values(), default=0.9),
                    )
                )
        else:
            for index, match in enumerate(_NAME_PATTERN.finditer(evidence.text)):
                surface = match.group(0)
                if surface in _NAME_STOPWORDS:
                    continue
                mentions.append(
                    ClassicalMention(
                        mention_id=_identifier(
                            "c0-mention", evidence.evidence_id, index, match.start(), surface
                        ),
                        evidence_id=evidence.evidence_id,
                        surface=surface,
                        start_char=match.start(),
                        end_char=match.end(),
                        entity_kind=_classify_entity(surface, None, config),
                        confidence=0.78,
                    )
                )

        mention_by_id = {item.mention_id: item for item in mentions}
        relations: list[ClassicalRelation] = []
        for candidate in sorted(
            evidence.relation_phrase_candidates,
            key=lambda item: item.candidate_id,
        ):
            if (
                candidate.object_mention_candidate_id is None
                or candidate.subject_mention_candidate_id not in mention_by_id
                or candidate.object_mention_candidate_id not in mention_by_id
            ):
                continue
            relations.append(
                ClassicalRelation(
                    relation_id=candidate.candidate_id,
                    evidence_id=evidence.evidence_id,
                    subject_mention_id=candidate.subject_mention_candidate_id,
                    object_mention_id=candidate.object_mention_candidate_id,
                    predicate=_normalize_predicate(candidate.surface_phrase, config),
                    surface_phrase=candidate.surface_phrase,
                    confidence=candidate.confidence,
                )
            )
        if not relations and len(mentions) >= 2:
            ordered_mentions = sorted(mentions, key=lambda item: item.start_char)
            for subject, object_ in pairwise(ordered_mentions):
                between = evidence.text[subject.end_char : object_.start_char]
                normalized = _normalize_predicate(between, config)
                if normalized in set(config.relation_lemmas.values()):
                    relations.append(
                        ClassicalRelation(
                            relation_id=_identifier(
                                "c0-relation",
                                evidence.evidence_id,
                                subject.mention_id,
                                object_.mention_id,
                                normalized,
                            ),
                            evidence_id=evidence.evidence_id,
                            subject_mention_id=subject.mention_id,
                            object_mention_id=object_.mention_id,
                            predicate=normalized,
                            surface_phrase=between.strip() or normalized,
                            confidence=0.74,
                        )
                    )

        events: list[ClassicalEvent] = []
        for candidate in sorted(evidence.event_candidates, key=lambda item: item.candidate_id):
            participants = tuple(
                item
                for item in candidate.participant_mention_candidate_ids
                if item in mention_by_id
            )
            events.append(
                ClassicalEvent(
                    event_candidate_id=candidate.candidate_id,
                    evidence_id=evidence.evidence_id,
                    trigger=candidate.trigger_surface,
                    participant_mention_ids=participants,
                    confidence=candidate.confidence,
                )
            )
        if not events:
            lowered_tokens = tuple(_WORD_PATTERN.finditer(evidence.text))
            for index, token in enumerate(lowered_tokens):
                trigger = token.group(0).casefold()
                if trigger not in config.event_triggers:
                    continue
                nearest = sorted(
                    mentions,
                    key=lambda item: (
                        min(abs(item.start_char - token.start()), abs(item.end_char - token.end())),
                        item.start_char,
                    ),
                )[:2]
                events.append(
                    ClassicalEvent(
                        event_candidate_id=_identifier(
                            "c0-event-candidate", evidence.evidence_id, index, trigger
                        ),
                        evidence_id=evidence.evidence_id,
                        trigger=trigger,
                        participant_mention_ids=tuple(item.mention_id for item in nearest),
                        confidence=0.7,
                    )
                )

        temporal_expressions = tuple(
            dict.fromkeys(
                (
                    *(clue.normalized_expression for clue in evidence.temporal_clues),
                    *(match.group(0) for match in _DAY_RANGE_PATTERN.finditer(evidence.text)),
                    *(match.group(0) for match in _DAY_POINT_PATTERN.finditer(evidence.text)),
                )
            )
        )
        return ClassicalEvidenceAnalysis(
            evidence_id=evidence.evidence_id,
            mentions=tuple(mentions),
            relations=tuple(relations),
            events=tuple(events),
            temporal_expressions=temporal_expressions,
            dependency_backend=self.backend_name,
        )


class SpacyCandidateBackend:
    """Adapter for an already-loaded spaCy pipeline, with deterministic fallback.

    The adapter never downloads a model.  When ``ner`` or ``parser`` is absent it
    retains indexed candidates and fills only missing candidate families through
    :class:`RuleCandidateBackend`.  This keeps setup bounded while exposing the
    planned tokenizer/NER/dependency hook for the frozen study environment.
    """

    backend_name = "spacy-plus-deterministic-rules-v1"

    def __init__(self, nlp: object) -> None:
        if not callable(nlp):
            raise TypeError("spaCy backend requires a callable Language-like object")
        self._nlp = nlp
        self._fallback = RuleCandidateBackend()

    def analyze(
        self,
        evidence: EvidenceRecord,
        config: ClassicalRuleConfig,
    ) -> ClassicalEvidenceAnalysis:
        base = self._fallback.analyze(evidence, config)
        doc = self._nlp(evidence.text)
        mentions = list(base.mentions)
        occupied = {(item.start_char, item.end_char) for item in mentions}
        for index, entity in enumerate(getattr(doc, "ents", ())):
            start = int(entity.start_char)
            end = int(entity.end_char)
            if (start, end) in occupied:
                continue
            surface = str(entity.text)
            mentions.append(
                ClassicalMention(
                    mention_id=_identifier(
                        "c0-spacy-mention", evidence.evidence_id, index, start, surface
                    ),
                    evidence_id=evidence.evidence_id,
                    surface=surface,
                    start_char=start,
                    end_char=end,
                    entity_kind=_classify_entity(
                        surface, str(getattr(entity, "label_", "")), config
                    ),
                    confidence=0.8,
                )
            )
        # Indexed relation/event candidates remain authoritative inputs.  Dependency
        # parse availability is recorded so the frozen competence report is honest.
        pipe_names = set(getattr(self._nlp, "pipe_names", ()))
        backend = self.backend_name if "parser" in pipe_names else f"{self.backend_name}:no-parser"
        return ClassicalEvidenceAnalysis(
            evidence_id=base.evidence_id,
            mentions=tuple(sorted(mentions, key=lambda item: (item.start_char, item.mention_id))),
            relations=base.relations,
            events=base.events,
            temporal_expressions=base.temporal_expressions,
            dependency_backend=backend,
        )


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        if left not in self.parent or right not in self.parent:
            return
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _story_and_validity(
    evidence: EvidenceRecord,
    expressions: Sequence[str],
    *,
    state_like: bool,
) -> tuple[StoryTime, ValidityTime]:
    joined = " ".join((*expressions, evidence.text))
    range_match = _DAY_RANGE_PATTERN.search(joined)
    if range_match:
        start, end = (int(range_match.group(1)), int(range_match.group(2)))
        story = StoryTime(
            kind=TemporalKind.INTERVAL,
            start=start,
            end=end,
            label=f"day {start} through day {end}",
        )
        validity = ValidityTime(
            kind=TemporalKind.INTERVAL,
            start=start,
            end=end,
            label="explicit validity interval",
        )
        return story, validity
    point_match = _DAY_POINT_PATTERN.search(joined)
    if point_match:
        point = int(point_match.group(1))
        story = StoryTime(kind=TemporalKind.POINT, point=point, label=f"day {point}")
        if state_like:
            validity = ValidityTime(
                kind=TemporalKind.INTERVAL,
                start=point,
                label=f"valid from day {point}; end unknown",
            )
        else:
            validity = ValidityTime(kind=TemporalKind.NOT_APPLICABLE)
        return story, validity
    return (
        StoryTime(
            kind=TemporalKind.UNKNOWN,
            reason="no exact story-time expression in supporting evidence",
        ),
        ValidityTime(
            kind=TemporalKind.UNKNOWN,
            reason="supporting evidence does not determine validity bounds",
        ),
    )


def _holder_time(story_time: StoryTime) -> HolderRelativeTime:
    values = story_time.model_dump(exclude={"content_hash"})
    return HolderRelativeTime(**values)


def _revelation(evidence: EvidenceRecord) -> RevelationPosition:
    return RevelationPosition(
        revelation_order=evidence.discourse_position.passage_order,
        label="first supported discourse occurrence",
    )


def _temporal_scope(
    evidence: EvidenceRecord,
    expressions: Sequence[str],
    *,
    state_like: bool,
) -> TemporalScope:
    story, validity = _story_and_validity(evidence, expressions, state_like=state_like)
    return TemporalScope(
        story_time=story,
        validity_time=validity,
        discourse_position=evidence.discourse_position,
        revelation_position=_revelation(evidence),
    )


class _AssertionSpec(ImmutableRecord):
    assertion_id: str
    predicate: str
    evidence_id: str
    subject_id: str | None = None
    object_id: str | None = None
    roles: tuple[RoleBinding, ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)
    temporal_scope: TemporalScope
    epistemic_attitude: EpistemicAttitude | None = None
    holder_id: str | None = None
    why_matters: str


class ClassicalPreBuilder:
    """Build and seal one comprehensive deterministic C0 ontology."""

    condition = ConditionName.C0_CLASSICAL_PRE

    def __init__(
        self,
        *,
        config: ClassicalRuleConfig | None = None,
        candidate_backend: ClassicalCandidateBackend | None = None,
    ) -> None:
        self.config = config or ClassicalRuleConfig()
        self.candidate_backend = candidate_backend or RuleCandidateBackend()

    def prepare(
        self,
        *,
        snapshot: EvidenceSnapshot,
        evidence: Sequence[EvidenceRecord],
        upper_ontology: UpperOntology,
        preconstruction_budgets: OutputBudgets,
        constructed_at: datetime,
        sealed_at: datetime,
    ) -> ConditionPreparation:
        if sealed_at < constructed_at:
            raise ConditionIntegrityError("C0 cannot seal before construction completes")
        if sealed_at >= snapshot.sealed_at and constructed_at < snapshot.sealed_at:
            # Construction may begin exactly when the snapshot is sealed but never before.
            raise ConditionIntegrityError("C0 construction began before evidence snapshot seal")
        if constructed_at < snapshot.sealed_at:
            raise ConditionIntegrityError("C0 construction began before evidence snapshot seal")
        evidence_by_id = {item.evidence_id: item for item in evidence}
        if tuple(evidence_by_id) != snapshot.eligible_evidence_ids:
            raise ConditionIntegrityError(
                "C0 preconstruction requires the complete ordered snapshot evidence"
            )
        if any(
            item.discourse_position.ordering_key
            > snapshot.horizon.max_discourse_position.ordering_key
            for item in evidence
        ):
            raise ConditionIntegrityError("C0 preconstruction received evidence after horizon")

        analyses = tuple(self.candidate_backend.analyze(item, self.config) for item in evidence)
        draft = self._construct_draft(
            evidence_by_id=evidence_by_id,
            analyses=analyses,
            upper_ontology=upper_ontology,
            budgets=preconstruction_budgets,
            constructed_at=constructed_at,
        )
        semantic_hash = preontology_semantic_hash(upper_ontology, draft)
        seal = ConstructionSeal(
            seal_id=_identifier("c0-seal", snapshot.content_hash, semantic_hash),
            condition=self.condition,
            snapshot_hash=snapshot.content_hash,
            ontology_hash=semantic_hash,
            constructed_at=constructed_at,
            sealed_at=sealed_at,
            sealed_object_ids=sealed_semantic_ids(draft),
        )
        preontology = SealedPreontology(
            artifact_id=_identifier("c0-preontology", snapshot.content_hash, semantic_hash),
            condition=self.condition,
            snapshot_hash=snapshot.content_hash,
            upper_ontology=upper_ontology,
            draft=draft,
            construction_seal=seal,
        )
        return ConditionPreparation(
            preparation_id=_identifier("c0-preparation", preontology.content_hash),
            condition=self.condition,
            snapshot_hash=snapshot.content_hash,
            completed_at=sealed_at,
            sealed_preontology=preontology,
        )

    def _construct_draft(
        self,
        *,
        evidence_by_id: Mapping[str, EvidenceRecord],
        analyses: Sequence[ClassicalEvidenceAnalysis],
        upper_ontology: UpperOntology,
        budgets: OutputBudgets,
        constructed_at: datetime,
    ) -> OntologyDraft:
        mentions = {
            mention.mention_id: mention for analysis in analyses for mention in analysis.mentions
        }
        if not mentions:
            raise ConditionIntegrityError("C0 found no named or indexed mention candidates")

        union = _UnionFind(mentions)
        for mention in mentions.values():
            for alias_id in mention.alias_candidate_ids:
                union.union(mention.mention_id, alias_id)

        by_normalized: dict[str, list[str]] = defaultdict(list)
        for mention in mentions.values():
            if not mention.is_pronoun:
                normalized = _normalized_surface(mention.surface, self.config.honorifics)
                if normalized:
                    by_normalized[normalized].append(mention.mention_id)
        for ids in by_normalized.values():
            for other in ids[1:]:
                union.union(ids[0], other)

        if self.config.alias_suffix_merge:
            normalized_values = tuple(sorted(by_normalized))
            for short in normalized_values:
                short_tokens = short.split()
                matching_long = [
                    long
                    for long in normalized_values
                    if len(long.split()) > len(short_tokens)
                    and long.split()[-len(short_tokens) :] == short_tokens
                ]
                if len(matching_long) == 1:
                    union.union(by_normalized[short][0], by_normalized[matching_long[0]][0])

        if self.config.local_pronoun_coreference:
            for analysis in analyses:
                ordered = sorted(analysis.mentions, key=lambda item: item.start_char)
                previous_named: ClassicalMention | None = None
                for mention in ordered:
                    if mention.is_pronoun and previous_named is not None:
                        union.union(mention.mention_id, previous_named.mention_id)
                    elif not mention.is_pronoun:
                        previous_named = mention

        grouped: dict[str, list[ClassicalMention]] = defaultdict(list)
        for mention in mentions.values():
            grouped[union.find(mention.mention_id)].append(mention)
        groups = tuple(
            sorted(
                (
                    tuple(sorted(items, key=lambda item: item.mention_id))
                    for items in grouped.values()
                ),
                key=lambda items: items[0].mention_id,
            )
        )
        entity_id_by_mention: dict[str, str] = {}
        entity_specs: list[dict[str, object]] = []
        for group in groups:
            entity_id = _identifier("c0-entity", *(item.mention_id for item in group))
            for mention in group:
                entity_id_by_mention[mention.mention_id] = entity_id
            named = [item for item in group if not item.is_pronoun] or list(group)
            label = sorted(named, key=lambda item: (-len(item.surface), item.surface))[0].surface
            kinds = Counter(item.entity_kind for item in group)
            kind = sorted(kinds, key=lambda item: (-kinds[item], item.value))[0]
            evidence_ids = tuple(sorted({item.evidence_id for item in group}))
            entity_specs.append(
                {
                    "entity_id": entity_id,
                    "label": label,
                    "mention_ids": tuple(item.mention_id for item in group),
                    "aliases": tuple(
                        sorted({item.surface for item in group if item.surface != label})
                    ),
                    "kind": kind,
                    "evidence_ids": evidence_ids,
                    "confidence": min(item.confidence for item in group),
                }
            )

        expressions_by_evidence = {
            analysis.evidence_id: analysis.temporal_expressions for analysis in analyses
        }
        assertion_specs: list[_AssertionSpec] = []
        event_specs: list[dict[str, object]] = []
        relation_rows = [relation for analysis in analyses for relation in analysis.relations]
        for relation in relation_rows:
            subject_id = entity_id_by_mention.get(relation.subject_mention_id)
            object_id = entity_id_by_mention.get(relation.object_mention_id)
            if subject_id is None or object_id is None:
                continue
            evidence_record = evidence_by_id[relation.evidence_id]
            state_like = relation.predicate in {
                "allied_with",
                "believed",
                "knows",
                "member_of",
                "opposed",
                "supported",
                "trusted",
            }
            attitude = {
                "believed": EpistemicAttitude.BELIEVED,
                "denied": EpistemicAttitude.DENIED,
                "reported": EpistemicAttitude.REPORTED,
            }.get(relation.predicate)
            assertion_specs.append(
                _AssertionSpec(
                    assertion_id=_identifier("c0-assertion", relation.relation_id),
                    predicate=relation.predicate,
                    evidence_id=relation.evidence_id,
                    subject_id=subject_id,
                    object_id=object_id,
                    confidence=relation.confidence,
                    temporal_scope=_temporal_scope(
                        evidence_record,
                        expressions_by_evidence[relation.evidence_id],
                        state_like=state_like,
                    ),
                    epistemic_attitude=attitude,
                    holder_id=subject_id if attitude is not None else None,
                    why_matters=(
                        f"The explicit {relation.surface_phrase.strip()} relation connects "
                        "two supported narrative objects."
                    ),
                )
            )

        for event_candidate in (event for analysis in analyses for event in analysis.events):
            event_id = _identifier("c0-event", event_candidate.event_candidate_id)
            participants = tuple(
                dict.fromkeys(
                    entity_id_by_mention[mention_id]
                    for mention_id in event_candidate.participant_mention_ids
                    if mention_id in entity_id_by_mention
                )
            )
            evidence_record = evidence_by_id[event_candidate.evidence_id]
            temporal = _temporal_scope(
                evidence_record,
                expressions_by_evidence[event_candidate.evidence_id],
                state_like=False,
            )
            role_assertion_ids: list[str] = []
            for index, participant_id in enumerate(participants):
                predicate = "event_agent" if index == 0 else "event_participant"
                assertion_id = _identifier(
                    "c0-assertion", event_candidate.event_candidate_id, predicate, participant_id
                )
                role_assertion_ids.append(assertion_id)
                assertion_specs.append(
                    _AssertionSpec(
                        assertion_id=assertion_id,
                        predicate=predicate,
                        evidence_id=event_candidate.evidence_id,
                        subject_id=event_id,
                        object_id=participant_id,
                        confidence=event_candidate.confidence,
                        temporal_scope=temporal,
                        why_matters=(
                            f"Evidence identifies a {predicate.replace('_', ' ')} role "
                            f"in the {event_candidate.trigger} event."
                        ),
                    )
                )
            if not role_assertion_ids:
                role_assertion_ids.append(
                    _identifier("c0-assertion", event_candidate.event_candidate_id, "occurrence")
                )
                assertion_specs.append(
                    _AssertionSpec(
                        assertion_id=role_assertion_ids[0],
                        predicate="event_occurrence",
                        evidence_id=event_candidate.evidence_id,
                        subject_id=event_id,
                        object_id=event_id,
                        confidence=event_candidate.confidence,
                        temporal_scope=temporal,
                        why_matters="Evidence explicitly records this event occurrence.",
                    )
                )
            event_specs.append(
                {
                    "event_id": event_id,
                    "trigger": event_candidate.trigger,
                    "evidence_id": event_candidate.evidence_id,
                    "confidence": event_candidate.confidence,
                    "temporal_scope": temporal,
                    "description_assertion_ids": tuple(role_assertion_ids),
                }
            )

        incident_assertions: dict[str, list[str]] = defaultdict(list)
        for spec in assertion_specs:
            endpoints = [spec.subject_id, spec.object_id]
            endpoints.extend(role.object_id for role in spec.roles)
            for endpoint in endpoints:
                if endpoint is not None:
                    incident_assertions[endpoint].append(spec.assertion_id)
        for spec in entity_specs:
            entity_id = str(spec["entity_id"])
            if incident_assertions[entity_id]:
                continue
            evidence_id = next(iter(spec["evidence_ids"]))
            evidence_record = evidence_by_id[evidence_id]
            assertion_id = _identifier("c0-assertion", entity_id, "mentioned")
            assertion_specs.append(
                _AssertionSpec(
                    assertion_id=assertion_id,
                    predicate="mentioned_as",
                    evidence_id=evidence_id,
                    subject_id=entity_id,
                    object_id=entity_id,
                    confidence=float(spec["confidence"]),
                    temporal_scope=_temporal_scope(
                        evidence_record,
                        expressions_by_evidence[evidence_id],
                        state_like=False,
                    ),
                    why_matters="The evidence explicitly mentions this narrative entity.",
                )
            )
            incident_assertions[entity_id].append(assertion_id)

        proposition_contents: list[PropositionContent] = []
        assertions: list[QualifiedAssertion] = []
        for spec in assertion_specs:
            evidence_record = evidence_by_id[spec.evidence_id]
            proposition_id: str | None = None
            epistemic_scope: EpistemicScope | None = None
            commitment = NarrativeCommitment.WORLD_COMMITTED
            if spec.epistemic_attitude is not None and spec.holder_id is not None:
                proposition_id = _identifier("c0-proposition", spec.assertion_id)
                proposition_contents.append(
                    PropositionContent(
                        proposition_content_id=proposition_id,
                        predicate_id=f"c0-predicate-{spec.predicate}",
                        subject_id=spec.subject_id,
                        object_id=spec.object_id,
                        roles=spec.roles,
                        temporal_content=spec.temporal_scope,
                        evidence_ids=(spec.evidence_id,),
                    )
                )
                epistemic_scope = EpistemicScope(
                    holder_id=spec.holder_id,
                    attitude=spec.epistemic_attitude,
                    proposition_content_id=proposition_id,
                    holder_relative_time=_holder_time(spec.temporal_scope.story_time),
                    evidence_ids=(spec.evidence_id,),
                )
                commitment = NarrativeCommitment.HOLDER_ATTRIBUTED
            assertions.append(
                QualifiedAssertion(
                    assertion_id=spec.assertion_id,
                    proposition_content_id=proposition_id,
                    predicate_id=f"c0-predicate-{spec.predicate}",
                    subject_id=spec.subject_id,
                    object_id=spec.object_id,
                    roles=spec.roles,
                    temporal_scope=spec.temporal_scope,
                    epistemic_scope=epistemic_scope,
                    narrative_commitment=commitment,
                    confidence=spec.confidence,
                    evidence_ids=(spec.evidence_id,),
                    provenance=(evidence_record.provenance,),
                    contextual_relevance=0.5,
                    why_matters=spec.why_matters,
                    why_matters_evidence_ids=(spec.evidence_id,),
                )
            )
            for endpoint in (spec.subject_id, spec.object_id):
                if endpoint is not None and spec.assertion_id not in incident_assertions[endpoint]:
                    incident_assertions[endpoint].append(spec.assertion_id)

        evidence_ids_by_kind: dict[ClassicalEntityKind, set[str]] = defaultdict(set)
        for spec in entity_specs:
            evidence_ids_by_kind[spec["kind"]].update(spec["evidence_ids"])
        event_evidence_ids = {str(spec["evidence_id"]) for spec in event_specs}
        upper_entity = (
            "entity"
            if "entity" in upper_ontology.primitive_types
            else upper_ontology.primitive_types[0]
        )
        upper_event = (
            "event"
            if "event" in upper_ontology.primitive_types
            else upper_ontology.primitive_types[0]
        )
        type_definitions = [
            LocalTypeDefinition(
                type_id=f"c0-type-{kind.value}",
                label=kind.value.replace("_", " ").title(),
                definition=f"Fixed classical {kind.value} category inferred before query reveal.",
                parent_upper_type=upper_entity,
                abstraction=AbstractionLevel.ACTOR,
                evidence_ids=tuple(sorted(evidence_ids)),
            )
            for kind, evidence_ids in sorted(
                evidence_ids_by_kind.items(), key=lambda item: item[0].value
            )
        ]
        if event_specs:
            type_definitions.append(
                LocalTypeDefinition(
                    type_id="c0-type-event",
                    label="Event",
                    definition="Fixed event category produced by query-blind trigger templates.",
                    parent_upper_type=upper_event,
                    abstraction=AbstractionLevel.EVENT_ROLE,
                    evidence_ids=tuple(sorted(event_evidence_ids)),
                )
            )

        relation_evidence: dict[str, set[str]] = defaultdict(set)
        for spec in assertion_specs:
            relation_evidence[spec.predicate].add(spec.evidence_id)
        upper_relation = (
            "related_to"
            if "related_to" in upper_ontology.primitive_relations
            else upper_ontology.primitive_relations[0]
        )
        predicates = tuple(
            LocalPredicateDefinition(
                predicate_id=f"c0-predicate-{predicate}",
                label=predicate.replace("_", " "),
                definition=(
                    f"Fixed classical relation normalized from explicit {predicate} evidence."
                ),
                arity=2,
                parent_upper_relation=upper_relation,
                evidence_ids=tuple(sorted(evidence_ids)),
            )
            for predicate, evidence_ids in sorted(relation_evidence.items())
        )
        local_schema = LocalContextSchema(
            schema_id=_identifier("c0-schema", self.config.content_hash),
            contextual_types=tuple(type_definitions),
            predicates=predicates,
            abstraction=AbstractionLevel.COLLECTIVE_CAUSAL_CHAIN,
        )

        entities: list[Entity] = []
        for spec in entity_specs:
            evidence_ids = tuple(spec["evidence_ids"])
            earliest_id = min(
                evidence_ids,
                key=lambda item: evidence_by_id[item].discourse_position.ordering_key,
            )
            story_time, _ = _story_and_validity(
                evidence_by_id[earliest_id],
                expressions_by_evidence[earliest_id],
                state_like=False,
            )
            entity_id = str(spec["entity_id"])
            entities.append(
                Entity(
                    entity_id=entity_id,
                    label=str(spec["label"]),
                    supported_mention_candidate_ids=tuple(spec["mention_ids"]),
                    aliases=tuple(spec["aliases"]),
                    contextual_type_id=f"c0-type-{spec['kind'].value}",
                    contextual_role="query-blind narrative participant",
                    abstraction=AbstractionLevel.ACTOR,
                    temporal_state=story_time,
                    uncertainty=ExplicitValueState.KNOWN,
                    confidence=float(spec["confidence"]),
                    evidence_ids=evidence_ids,
                    description=(
                        f"{spec['label']} is an evidence-grounded classical entity assembled "
                        "before query reveal."
                    ),
                    description_assertion_ids=(incident_assertions[entity_id][0],),
                )
            )

        events = tuple(
            Event(
                event_id=str(spec["event_id"]),
                label=f"{str(spec['trigger']).title()} event",
                contextual_type_id="c0-type-event",
                occurrence_time=spec["temporal_scope"].story_time,
                reification_reason=(
                    "A frozen lexical trigger with participant roles or independent time "
                    "requires an event object."
                ),
                uncertainty=ExplicitValueState.KNOWN,
                confidence=float(spec["confidence"]),
                evidence_ids=(str(spec["evidence_id"]),),
                description=f"Evidence-grounded {spec['trigger']} occurrence.",
                description_assertion_ids=tuple(spec["description_assertion_ids"]),
            )
            for spec in event_specs
        )

        decisions: list[OntologyDecision] = []
        for entity, spec in zip(entities, entity_specs, strict=True):
            decisions.append(
                OntologyDecision(
                    decision_id=_identifier("c0-decision", "entity", entity.entity_id),
                    operator=ConstructionOperator.CONTEXTUAL_TYPE,
                    evidence_ids=entity.evidence_ids,
                    rationale="Frozen name/type rules created this pre-query entity.",
                    decided_at=constructed_at,
                    input_object_ids=entity.supported_mention_candidate_ids,
                    created_object_ids=(entity.entity_id,),
                )
            )
            if len(tuple(spec["mention_ids"])) > 1:
                decisions.append(
                    OntologyDecision(
                        decision_id=_identifier("c0-decision", "merge", entity.entity_id),
                        operator=ConstructionOperator.MERGE,
                        evidence_ids=entity.evidence_ids,
                        rationale="Frozen alias/coreference heuristics merged supported mentions.",
                        decided_at=constructed_at,
                        input_object_ids=tuple(spec["mention_ids"]),
                        created_object_ids=(entity.entity_id,),
                    )
                )
        for event in events:
            decisions.append(
                OntologyDecision(
                    decision_id=_identifier("c0-decision", "event", event.event_id),
                    operator=ConstructionOperator.EVENT_REIFICATION,
                    evidence_ids=event.evidence_ids,
                    rationale="Frozen event trigger/role rule reified this occurrence.",
                    decided_at=constructed_at,
                    created_object_ids=(event.event_id,),
                )
            )
        for predicate in predicates:
            decisions.append(
                OntologyDecision(
                    decision_id=_identifier("c0-decision", "predicate", predicate.predicate_id),
                    operator=ConstructionOperator.SCHEMA_RELATION,
                    evidence_ids=predicate.evidence_ids,
                    rationale="Frozen surface-relation normalization created this predicate.",
                    decided_at=constructed_at,
                    created_object_ids=(predicate.predicate_id,),
                )
            )
        for assertion in assertions:
            if assertion.temporal_scope.story_time.kind is not TemporalKind.UNKNOWN:
                decisions.append(
                    OntologyDecision(
                        decision_id=_identifier("c0-decision", "temporal", assertion.assertion_id),
                        operator=ConstructionOperator.TEMPORAL_QUALIFICATION,
                        evidence_ids=assertion.evidence_ids,
                        rationale="Frozen temporal-expression rules qualified this assertion.",
                        decided_at=constructed_at,
                        input_object_ids=(assertion.assertion_id,),
                        created_object_ids=(assertion.temporal_scope.content_hash,),
                    )
                )
            if assertion.epistemic_scope is not None:
                decisions.append(
                    OntologyDecision(
                        decision_id=_identifier("c0-decision", "epistemic", assertion.assertion_id),
                        operator=ConstructionOperator.EPISTEMIC_QUALIFICATION,
                        evidence_ids=assertion.evidence_ids,
                        rationale="Frozen report/belief/denial rule retained holder attribution.",
                        decided_at=constructed_at,
                        input_object_ids=(assertion.assertion_id,),
                        created_object_ids=(assertion.epistemic_scope.content_hash,),
                    )
                )

        node_count = len(entities) + len(events)
        assertion_count = len(assertions)
        accounting = BudgetAccounting(
            nodes_used=node_count,
            assertions_used=assertion_count,
            display_nodes_used=min(node_count, budgets.display_node_budget),
            display_assertions_used=min(assertion_count, budgets.display_assertion_budget),
            input_tokens=0,
            output_tokens=0,
        )
        try:
            accounting.validate_against(budgets)
        except ValueError as exc:
            raise ConditionIntegrityError(
                "C0 comprehensive preconstruction exceeds its declared prebuild budget; "
                "refusing silent truncation"
            ) from exc
        return OntologyDraft(
            contextual_interpretation=(
                "Query-blind comprehensive classical ontology from frozen NER, dependency, "
                "alias/coreference, event, relation, and temporal rules."
            ),
            local_schema=local_schema,
            instance_graph=InstanceGraph(
                entities=tuple(entities),
                events=events,
                proposition_contents=tuple(proposition_contents),
                assertions=tuple(assertions),
            ),
            decisions=tuple(decisions),
            budget_accounting=accounting,
        )

    def produce(self, inputs: ProduceInputs) -> ConditionAttemptRecord:
        if inputs.run_config.condition is not self.condition:
            raise ConditionIntegrityError("ClassicalPre can produce only C0")
        preontology = inputs.preparation.sealed_preontology
        if preontology is None or preontology.condition is not self.condition:
            raise ConditionIntegrityError("C0 query projection requires its sealed preontology")
        projection = project_sealed_c0(preontology, inputs)
        return ConditionAttemptRecord(
            attempt_id=_identifier("c0-attempt", inputs.context.content_hash),
            condition=self.condition,
            unit_id=inputs.context.context_id,
            outcome=RunOutcome.SUCCEEDED,
            projection=projection,
            release_class=inputs.packet.release_class,
        )


def _extent_interval(extent: StoryTime) -> tuple[int | None, int | None]:
    if extent.kind is TemporalKind.POINT:
        return extent.point, extent.point
    if extent.kind is TemporalKind.INTERVAL:
        return extent.start, extent.end
    return None, None


def _time_compatibility(assertion: QualifiedAssertion, query_story_time: StoryTime) -> float:
    query_start, query_end = _extent_interval(query_story_time)
    assertion_start, assertion_end = _extent_interval(assertion.temporal_scope.story_time)
    if query_start is None and query_end is None:
        return 0.0
    if assertion_start is None and assertion_end is None:
        return 0.0
    left_start = float("-inf") if query_start is None else query_start
    left_end = float("inf") if query_end is None else query_end
    right_start = float("-inf") if assertion_start is None else assertion_start
    right_end = float("inf") if assertion_end is None else assertion_end
    return 0.2 if max(left_start, right_start) <= min(left_end, right_end) else -0.35


def _assertion_endpoints(assertion: QualifiedAssertion) -> frozenset[str]:
    values = {value for value in (assertion.subject_id, assertion.object_id) if value is not None}
    values.update(role.object_id for role in assertion.roles)
    return frozenset(values)


def project_sealed_c0(
    preontology: SealedPreontology,
    inputs: ProduceInputs,
) -> OntologyProjection:
    """Fixed support-aware projection; every semantic atom must inherit a seal ID."""

    if preontology.construction_seal.sealed_at >= inputs.context.revealed_at:
        raise ConditionIntegrityError("C0 ontology was not sealed strictly before query reveal")
    source_graph = preontology.draft.instance_graph
    entity_by_id = {item.entity_id: item for item in source_graph.entities}
    event_by_id = {item.event_id: item for item in source_graph.events}
    assertion_by_id = {item.assertion_id: item for item in source_graph.assertions}
    proposition_by_id = {
        item.proposition_content_id: item for item in source_graph.proposition_contents
    }
    node_ids = set(entity_by_id) | set(event_by_id)
    query_terms = _words(
        " ".join((inputs.context.wording, inputs.context.lens, inputs.context.target))
    )
    evidence_frequency = Counter(
        evidence_id
        for assertion in source_graph.assertions
        for evidence_id in assertion.evidence_ids
    )

    def score(assertion: QualifiedAssertion) -> tuple[float, str]:
        predicate = assertion.predicate_id.removeprefix("c0-predicate-").replace("_", " ")
        labels = [predicate, assertion.why_matters]
        labels.extend(
            entity_by_id[item].label
            for item in _assertion_endpoints(assertion)
            if item in entity_by_id
        )
        labels.extend(
            event_by_id[item].label
            for item in _assertion_endpoints(assertion)
            if item in event_by_id
        )
        terms = _words(" ".join(labels))
        lexical = len(query_terms & terms) / max(1, len(query_terms))
        support = 0.1 * assertion.confidence
        rarity = min(
            (1.0 / evidence_frequency[evidence_id] for evidence_id in assertion.evidence_ids),
            default=0.0,
        )
        rare_bonus = rarity * 0.12
        temporal = _time_compatibility(assertion, inputs.context.story_scope)
        return lexical + support + rare_bonus + temporal, assertion.assertion_id

    ranked = sorted(source_graph.assertions, key=lambda item: (-score(item)[0], score(item)[1]))
    selected_assertion_ids: set[str] = set()
    selected_node_ids: set[str] = set()

    def support_closure(assertion_id: str) -> tuple[set[str], set[str]]:
        closure_assertions = {assertion_id}
        closure_nodes: set[str] = set()
        changed = True
        while changed:
            changed = False
            for current_id in tuple(closure_assertions):
                current = assertion_by_id[current_id]
                for endpoint in _assertion_endpoints(current) & node_ids:
                    if endpoint not in closure_nodes:
                        closure_nodes.add(endpoint)
                        changed = True
                    node = entity_by_id.get(endpoint) or event_by_id.get(endpoint)
                    if node is not None:
                        for support_id in node.description_assertion_ids:
                            if support_id not in closure_assertions:
                                closure_assertions.add(support_id)
                                changed = True
        return closure_assertions, closure_nodes

    remaining = list(ranked)
    while remaining:
        # Re-score with typed-path continuity once an initial focus exists.
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                score(remaining[index])[0]
                + (0.15 if selected_node_ids & _assertion_endpoints(remaining[index]) else 0.0),
                tuple(-ord(char) for char in remaining[index].assertion_id),
            ),
        )
        candidate = remaining.pop(best_index)
        closure_assertions, closure_nodes = support_closure(candidate.assertion_id)
        proposed_assertions = selected_assertion_ids | closure_assertions
        proposed_nodes = selected_node_ids | closure_nodes
        if len(proposed_assertions) > inputs.context.budgets.assertion_budget:
            continue
        if len(proposed_nodes) > inputs.context.budgets.node_budget:
            continue
        selected_assertion_ids = proposed_assertions
        selected_node_ids = proposed_nodes
        if (
            len(selected_assertion_ids) == inputs.context.budgets.assertion_budget
            or len(selected_node_ids) == inputs.context.budgets.node_budget
        ):
            break

    selected_assertions = tuple(
        item for item in source_graph.assertions if item.assertion_id in selected_assertion_ids
    )
    selected_entities = tuple(
        item for item in source_graph.entities if item.entity_id in selected_node_ids
    )
    selected_events = tuple(
        item for item in source_graph.events if item.event_id in selected_node_ids
    )
    selected_proposition_ids = {
        item.proposition_content_id
        for item in selected_assertions
        if item.proposition_content_id is not None
    }
    selected_propositions = tuple(
        proposition_by_id[item]
        for item in sorted(selected_proposition_ids)
        if item in proposition_by_id
    )
    selected_semantic_ids = {
        *(item.entity_id for item in selected_entities),
        *(item.event_id for item in selected_events),
        *(item.proposition_content_id for item in selected_propositions),
        *(item.assertion_id for item in selected_assertions),
    }
    if not selected_semantic_ids.issubset(preontology.construction_seal.sealed_object_ids):
        raise ConditionIntegrityError("C0 fixed projector emitted an unsealed semantic ID")

    evidence_ids = tuple(
        sorted({evidence_id for item in selected_assertions for evidence_id in item.evidence_ids})
    )
    decisions = (
        OntologyDecision(
            decision_id=_identifier(
                "c0-selection", preontology.content_hash, inputs.context.content_hash
            ),
            operator=ConstructionOperator.SELECTION,
            evidence_ids=evidence_ids or (inputs.packet.ordered_evidence_ids[0],),
            rationale=(
                "Frozen lexical/time/support/path scoring selected only sealed pre-query objects."
            ),
            decided_at=inputs.context.revealed_at,
            input_object_ids=tuple(sorted(selected_semantic_ids)),
        ),
    )
    node_count = len(selected_entities) + len(selected_events)
    assertion_count = len(selected_assertions)
    accounting = BudgetAccounting(
        nodes_used=node_count,
        assertions_used=assertion_count,
        display_nodes_used=min(node_count, inputs.context.budgets.display_node_budget),
        display_assertions_used=min(
            assertion_count, inputs.context.budgets.display_assertion_budget
        ),
        input_tokens=0,
        output_tokens=0,
    )
    validation = ValidationRecord(
        validation_id=_identifier(
            "c0-validation", preontology.content_hash, inputs.context.content_hash
        ),
        target_id=_identifier(
            "c0-projection", preontology.content_hash, inputs.context.content_hash
        ),
        validation_status=ValidationStatus.ACCEPTED,
        evidence_support_status=EvidenceSupportStatus.SUPPORTED,
        temporal_status=TemporalDeterminationStatus.VALID,
        commitment_status=CommitmentCheckStatus.VALID,
        validated_at=inputs.context.revealed_at,
    )
    projection_id = validation.target_id
    return OntologyProjection(
        projection_id=projection_id,
        condition=ConditionName.C0_CLASSICAL_PRE,
        snapshot_hash=inputs.snapshot.content_hash,
        packet_hash=inputs.packet.content_hash,
        context_hash=inputs.context.content_hash,
        upper_ontology=preontology.upper_ontology,
        local_schema=preontology.draft.local_schema,
        instance_graph=InstanceGraph(
            entities=selected_entities,
            events=selected_events,
            proposition_contents=selected_propositions,
            assertions=selected_assertions,
        ),
        decisions=decisions,
        validation_records=(validation,),
        budget_accounting=accounting,
        budgets=inputs.context.budgets,
        construction_seal=preontology.construction_seal,
        run_id=_identifier("c0-run", inputs.run_config.content_hash, projection_id),
        release_class=(
            ReleaseClass.RESTRICTED
            if inputs.packet.release_class is ReleaseClass.RESTRICTED
            else ReleaseClass.PUBLIC
        ),
    )
