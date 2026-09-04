"""Known-answer semantic grounding audit for the bounded Phase-1 pilot.

This module is deliberately isolated under ``scorer_only``.  It consumes an
already parsed draft plus the exact seven hand-authored evidence records.  It
does not generate prompt content and is not imported by request packing code.
The audit maps model-created IDs back to sealed mention anchors and supported
event/fact alternatives; citation presence alone is never a positive verdict.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from story_projection_onto.contracts import (
    EpistemicAttitude,
    ModelVisibleEvidenceRecord,
    NarrativeCommitment,
    OntologyDraft,
    QualifiedAssertion,
    TemporalKind,
)


class SemanticSupportStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SemanticGroundingAssessment:
    record_id: str
    record_kind: str
    status: SemanticSupportStatus
    evidence_ids: tuple[str, ...]
    matched_fact_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class AcceptanceGroundingAudit:
    assessments: tuple[SemanticGroundingAssessment, ...]

    @property
    def supported_count(self) -> int:
        return sum(item.status is SemanticSupportStatus.SUPPORTED for item in self.assessments)

    @property
    def unsupported_count(self) -> int:
        return sum(item.status is SemanticSupportStatus.UNSUPPORTED for item in self.assessments)

    @property
    def unknown_count(self) -> int:
        return sum(item.status is SemanticSupportStatus.UNKNOWN for item in self.assessments)

    @property
    def grounding_precision(self) -> float:
        if not self.assessments:
            return 0.0
        return self.supported_count / len(self.assessments)

    @property
    def complete(self) -> bool:
        return bool(self.assessments) and all(
            item.status is SemanticSupportStatus.SUPPORTED for item in self.assessments
        )

    def raise_for_failure(self) -> None:
        if self.complete:
            return
        failures = [
            f"{item.record_kind}:{item.record_id}:{item.status.value}:{item.reason}"
            for item in self.assessments
            if item.status is not SemanticSupportStatus.SUPPORTED
        ]
        raise ValueError("semantic grounding audit failed: " + "; ".join(failures))

    def public_manifest(self) -> dict[str, object]:
        return {
            "assessment_count": len(self.assessments),
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "unknown_count": self.unknown_count,
            "grounding_precision": self.grounding_precision,
            "grounding_complete": self.complete,
            "assessment_statuses": {item.record_id: item.status.value for item in self.assessments},
            "matched_fact_ids": {
                item.record_id: item.matched_fact_id
                for item in self.assessments
                if item.matched_fact_id is not None
            },
            "scorer_only_oracle_revision": "phase1-known-answer-v1",
        }


@dataclass(frozen=True, slots=True)
class _KnownFact:
    fact_id: str
    parent_relations: frozenset[str]
    required_evidence: frozenset[str]
    allowed_evidence: frozenset[str]
    relation_stems: tuple[str, ...]
    subject_concepts: frozenset[str] = frozenset()
    object_concepts: frozenset[str] = frozenset()
    required_role_concepts: frozenset[str] = frozenset()
    symmetric: bool = False
    story_point: int | None = None
    holder_concept: str | None = None
    attitude: EpistemicAttitude | None = None
    commitment: NarrativeCommitment = NarrativeCommitment.WORLD_COMMITTED


_MENTION_CONCEPTS: Mapping[str, str] = {
    "m-lio-01": "courier_lio",
    "m-ash-courier": "courier_lio",
    "m-lio-roster": "courier_lio",
    "m-lio-03": "courier_lio",
    "m-lio-04": "courier_lio",
    "m-ash-mechanic": "mechanic_ash",
    "m-gate-01": "north_gate",
    "m-gate-03": "north_gate",
    "m-seal-01": "seal",
    "m-seal-02": "seal",
    "m-seal-04": "seal",
    "m-seal-06": "seal",
    "m-pump-03": "river_pump",
    "m-council-04": "council",
    "m-floodgate-04": "floodgate",
    "m-mara-05": "mara",
    "m-bridge-05": "bridge",
    "m-safe-05": "safe_state",
    "m-mark-06": "mark",
    "m-key-06": "floodgate_key",
    "m-bell-07": "bell",
    "m-vendors-07": "vendors",
}

_EVENT_SPECS: Mapping[str, tuple[frozenset[str], tuple[str, ...], int | None]] = {
    "event:arrival": (frozenset({"ev-01"}), ("arriv",), 1),
    "event:repair": (frozenset({"ev-03"}), ("repair",), None),
    "event:delivery": (frozenset({"ev-04"}), ("deliver", "handoff"), 2),
    "event:opening": (frozenset({"ev-04"}), ("open", "floodgate"), 2),
    "event:report": (frozenset({"ev-05"}), ("report",), None),
    "event:bell": (frozenset({"ev-07"}), ("bell", "ring", "rang"), None),
    "event:trade": (frozenset({"ev-07"}), ("trade", "market", "vendor"), None),
}

_KNOWN_FACTS = (
    _KnownFact(
        fact_id="lio_arrived_at_north_gate",
        parent_relations=frozenset({"located_at"}),
        required_evidence=frozenset({"ev-01"}),
        allowed_evidence=frozenset({"ev-01"}),
        relation_stems=("arriv", "located", "at"),
        subject_concepts=frozenset({"courier_lio"}),
        object_concepts=frozenset({"north_gate"}),
        story_point=1,
    ),
    _KnownFact(
        fact_id="lio_remained_at_north_gate",
        parent_relations=frozenset({"located_at"}),
        required_evidence=frozenset({"ev-03"}),
        allowed_evidence=frozenset({"ev-03"}),
        relation_stems=("remain", "located", "stay"),
        subject_concepts=frozenset({"courier_lio"}),
        object_concepts=frozenset({"north_gate"}),
    ),
    _KnownFact(
        fact_id="mechanic_repaired_pump",
        parent_relations=frozenset({"related_to", "participates_in"}),
        required_evidence=frozenset({"ev-03"}),
        allowed_evidence=frozenset({"ev-03"}),
        relation_stems=("repair",),
        subject_concepts=frozenset({"mechanic_ash"}),
        object_concepts=frozenset({"river_pump", "event:repair"}),
    ),
    _KnownFact(
        fact_id="courier_and_mechanic_are_distinct",
        parent_relations=frozenset({"related_to"}),
        required_evidence=frozenset({"ev-03"}),
        allowed_evidence=frozenset({"ev-02", "ev-03"}),
        relation_stems=("distinct", "separat", "different", "disambigu"),
        subject_concepts=frozenset({"courier_lio"}),
        object_concepts=frozenset({"mechanic_ash"}),
        symmetric=True,
    ),
    _KnownFact(
        fact_id="delivery_precedes_opening",
        parent_relations=frozenset({"precedes"}),
        required_evidence=frozenset({"ev-04"}),
        allowed_evidence=frozenset({"ev-04"}),
        relation_stems=("preced", "before", "chain", "order"),
        subject_concepts=frozenset({"event:delivery"}),
        object_concepts=frozenset({"event:opening"}),
        story_point=2,
    ),
    _KnownFact(
        fact_id="credential_delivery_chain",
        parent_relations=frozenset({"precedes"}),
        required_evidence=frozenset({"ev-04"}),
        allowed_evidence=frozenset({"ev-04"}),
        relation_stems=("preced", "before", "chain", "order"),
        required_role_concepts=frozenset(
            {"event:delivery", "event:opening", "courier_lio", "seal"}
        ),
        story_point=2,
    ),
    _KnownFact(
        fact_id="marked_seal_enabled_opening",
        parent_relations=frozenset({"causes"}),
        required_evidence=frozenset({"ev-04", "ev-06"}),
        allowed_evidence=frozenset({"ev-04", "ev-06"}),
        relation_stems=("caus", "enabl", "permit", "credential", "allow"),
        subject_concepts=frozenset({"seal", "mark"}),
        object_concepts=frozenset({"event:opening", "floodgate"}),
        story_point=2,
    ),
    _KnownFact(
        fact_id="mark_matches_floodgate_key",
        parent_relations=frozenset({"related_to"}),
        required_evidence=frozenset({"ev-06"}),
        allowed_evidence=frozenset({"ev-06"}),
        relation_stems=("match", "correspond", "credential"),
        subject_concepts=frozenset({"mark", "seal"}),
        object_concepts=frozenset({"floodgate_key"}),
    ),
    _KnownFact(
        fact_id="bridge_safe_content_reported_by_mara",
        parent_relations=frozenset({"has_status"}),
        required_evidence=frozenset({"ev-05"}),
        allowed_evidence=frozenset({"ev-05"}),
        relation_stems=("safe", "status"),
        subject_concepts=frozenset({"bridge"}),
        object_concepts=frozenset({"safe_state"}),
        holder_concept="mara",
        attitude=EpistemicAttitude.REPORTED,
        commitment=NarrativeCommitment.HOLDER_ATTRIBUTED,
    ),
    _KnownFact(
        fact_id="mara_made_bridge_report",
        parent_relations=frozenset({"reports"}),
        required_evidence=frozenset({"ev-05"}),
        allowed_evidence=frozenset({"ev-05"}),
        relation_stems=("report",),
        subject_concepts=frozenset({"mara"}),
        object_concepts=frozenset({"bridge", "event:report"}),
    ),
    _KnownFact(
        fact_id="bell_and_market_activity",
        parent_relations=frozenset({"related_to"}),
        required_evidence=frozenset({"ev-07"}),
        allowed_evidence=frozenset({"ev-07"}),
        relation_stems=("ring", "rang", "trade", "market"),
        subject_concepts=frozenset({"bell"}),
        object_concepts=frozenset({"vendors"}),
    ),
)

_EXPECTED_EVIDENCE_TEXT: Mapping[str, str] = {
    "ev-01": "At dawn, courier Lio arrived at North Gate carrying a copper seal.",
    "ev-02": (
        "The guard called the courier 'Ash'; a roster lists Lio and Ash with the same copper seal."
    ),
    "ev-03": (
        "Separately, mechanic Ash repaired the river pump while courier Lio remained at North Gate."
    ),
    "ev-04": (
        "After Lio delivered the seal, the council opened the floodgate; water reached "
        "the farms before dusk."
    ),
    "ev-05": "Mara reported that the bridge was safe, but she had not inspected it.",
    "ev-06": (
        "A single scratched mark on the seal matched the floodgate key; without that "
        "mark, guards would have refused it."
    ),
    "ev-07": "The town bell rang hourly while market vendors traded in the square.",
}

_ALLOWED_COMPOSITE_ENTITY_CONCEPTS = {
    frozenset({"seal", "mark"}),
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(value.casefold()))


def _contains_stem(value: str, stems: Sequence[str]) -> bool:
    tokens = _tokens(value)
    return any(any(token.startswith(stem) for token in tokens) for stem in stems)


def _assessment(
    record_id: str,
    record_kind: str,
    status: SemanticSupportStatus,
    evidence_ids: Sequence[str],
    reason: str,
    fact_id: str | None = None,
) -> SemanticGroundingAssessment:
    return SemanticGroundingAssessment(
        record_id=record_id,
        record_kind=record_kind,
        status=status,
        evidence_ids=tuple(evidence_ids),
        matched_fact_id=fact_id,
        reason=reason,
    )


def _validate_fixture_evidence(evidence: Sequence[ModelVisibleEvidenceRecord]) -> None:
    by_id = {item.evidence_id: item for item in evidence}
    if set(by_id) != set(_EXPECTED_EVIDENCE_TEXT):
        raise ValueError("Phase-1 semantic oracle requires the exact seven evidence IDs")
    mismatches = [
        evidence_id
        for evidence_id, text in _EXPECTED_EVIDENCE_TEXT.items()
        if by_id[evidence_id].text != text
    ]
    if mismatches:
        raise ValueError("Phase-1 semantic oracle evidence text changed: " + ", ".join(mismatches))
    observed_mentions = {
        mention.candidate_id for record in evidence for mention in record.mention_candidates
    }
    if observed_mentions != set(_MENTION_CONCEPTS):
        raise ValueError("Phase-1 semantic oracle mention inventory changed")


def _story_time_supported(
    *,
    kind: TemporalKind,
    point: int | None,
    start: int | None,
    end: int | None,
    expected_point: int | None,
) -> bool:
    if expected_point is None:
        return kind in {TemporalKind.UNKNOWN, TemporalKind.NOT_APPLICABLE}
    if kind is TemporalKind.POINT:
        return point == expected_point
    if kind is TemporalKind.INTERVAL:
        return (start is None or start <= expected_point) and (end is None or expected_point <= end)
    return False


def _scope_supported(
    assertion: QualifiedAssertion,
    fact: _KnownFact,
) -> tuple[bool, str]:
    scope = assertion.temporal_scope
    expected_position = max(int(item.split("-")[1]) for item in assertion.evidence_ids)
    if scope.discourse_position.ordering_key != (expected_position, 0, 0):
        return False, "discourse position does not equal the latest cited evidence"
    if scope.revelation_position.revelation_order != expected_position:
        return False, "revelation position does not equal the latest cited evidence"
    if not _story_time_supported(
        kind=scope.story_time.kind,
        point=scope.story_time.point,
        start=scope.story_time.start,
        end=scope.story_time.end,
        expected_point=fact.story_point,
    ):
        return False, "story time is not supported by the evidence clock"
    if not _story_time_supported(
        kind=scope.validity_time.kind,
        point=scope.validity_time.point,
        start=scope.validity_time.start,
        end=scope.validity_time.end,
        expected_point=fact.story_point,
    ):
        return False, "validity time is not supported by the evidence clock"
    return True, "qualified clocks match known-answer evidence"


def _event_concept(
    *,
    label: str,
    description: str,
    reification_reason: str,
    evidence_ids: Sequence[str],
) -> tuple[str | None, int | None]:
    cited = frozenset(evidence_ids)
    matches = [
        (concept, point)
        for concept, (required, stems, point) in _EVENT_SPECS.items()
        if required.issubset(cited) and _contains_stem(label, stems)
    ]
    if not matches:
        semantic_text = " ".join((description, reification_reason))
        matches = [
            (concept, point)
            for concept, (required, stems, point) in _EVENT_SPECS.items()
            if required.issubset(cited) and _contains_stem(semantic_text, stems)
        ]
    if len(matches) != 1:
        return None, None
    return matches[0]


def _referent_matches(concepts: frozenset[str], alternatives: frozenset[str]) -> bool:
    return bool(concepts.intersection(alternatives))


def _assertion_fact(
    assertion: QualifiedAssertion,
    *,
    predicate_parent: str,
    predicate_text: str,
    referent_concepts: Mapping[str, frozenset[str]],
) -> tuple[_KnownFact | None, SemanticSupportStatus, str]:
    cited = frozenset(assertion.evidence_ids)
    subject = (
        frozenset()
        if assertion.subject_id is None
        else referent_concepts.get(assertion.subject_id, frozenset())
    )
    object_ = (
        frozenset()
        if assertion.object_id is None
        else referent_concepts.get(assertion.object_id, frozenset())
    )
    if assertion.direction == "inverse":
        subject, object_ = object_, subject
    role_concepts = frozenset(
        concept
        for role in assertion.roles
        for concept in referent_concepts.get(role.object_id, frozenset())
    )
    for fact in _KNOWN_FACTS:
        if predicate_parent not in fact.parent_relations:
            continue
        if not fact.required_evidence.issubset(cited) or not cited.issubset(fact.allowed_evidence):
            continue
        if not _contains_stem(predicate_text, fact.relation_stems):
            continue
        if fact.required_role_concepts:
            if not fact.required_role_concepts.issubset(role_concepts):
                continue
        else:
            direct = _referent_matches(subject, fact.subject_concepts) and _referent_matches(
                object_, fact.object_concepts
            )
            reverse = (
                fact.symmetric
                and _referent_matches(subject, fact.object_concepts)
                and _referent_matches(object_, fact.subject_concepts)
            )
            if not (direct or reverse):
                continue
        if (
            fact.holder_concept is not None
            and assertion.narrative_commitment is NarrativeCommitment.WORLD_COMMITTED
            and assertion.epistemic_scope is None
        ):
            return (
                fact,
                SemanticSupportStatus.UNSUPPORTED,
                ("holder-attributed report was promoted to world truth"),
            )
        if assertion.narrative_commitment is not fact.commitment:
            return fact, SemanticSupportStatus.UNSUPPORTED, "narrative commitment changed"
        epistemic = assertion.epistemic_scope
        if fact.holder_concept is None:
            if epistemic is not None:
                return fact, SemanticSupportStatus.UNSUPPORTED, "unexpected holder attribution"
        else:
            if epistemic is None:
                return fact, SemanticSupportStatus.UNSUPPORTED, "reported content lost its holder"
            holder = referent_concepts.get(epistemic.holder_id, frozenset())
            if fact.holder_concept not in holder or epistemic.attitude is not fact.attitude:
                return fact, SemanticSupportStatus.UNSUPPORTED, "wrong holder or attitude"
            if not _story_time_supported(
                kind=epistemic.holder_relative_time.kind,
                point=epistemic.holder_relative_time.point,
                start=epistemic.holder_relative_time.start,
                end=epistemic.holder_relative_time.end,
                expected_point=None,
            ):
                return (
                    fact,
                    SemanticSupportStatus.UNSUPPORTED,
                    ("holder-relative time invents a story coordinate"),
                )
        scope_ok, scope_reason = _scope_supported(assertion, fact)
        if not scope_ok:
            return fact, SemanticSupportStatus.UNSUPPORTED, scope_reason
        return fact, SemanticSupportStatus.SUPPORTED, scope_reason

    bridge = "bridge" in subject or "bridge" in object_
    safe = "safe_state" in subject or "safe_state" in object_
    if bridge and safe and assertion.narrative_commitment is NarrativeCommitment.WORLD_COMMITTED:
        return (
            None,
            SemanticSupportStatus.UNSUPPORTED,
            "Mara's uninspected safety report was promoted to world truth",
        )
    return None, SemanticSupportStatus.UNKNOWN, "no known-answer fact supports this assertion"


def audit_acceptance_semantic_grounding(
    *,
    draft: OntologyDraft,
    evidence: Sequence[ModelVisibleEvidenceRecord],
) -> AcceptanceGroundingAudit:
    """Assess every factual draft record against the sealed pilot known answers."""

    _validate_fixture_evidence(evidence)
    assessments: list[SemanticGroundingAssessment] = []
    entity_base: dict[str, SemanticGroundingAssessment] = {}
    event_base: dict[str, SemanticGroundingAssessment] = {}
    referents: dict[str, frozenset[str]] = {}

    for entity in draft.instance_graph.entities:
        concepts = frozenset(
            _MENTION_CONCEPTS[candidate_id]
            for candidate_id in entity.supported_mention_candidate_ids
            if candidate_id in _MENTION_CONCEPTS
        )
        referents[entity.entity_id] = concepts
        if len(concepts) > 1 and concepts not in _ALLOWED_COMPOSITE_ENTITY_CONCEPTS:
            status = SemanticSupportStatus.UNSUPPORTED
            reason = "entity merges distinct known-answer concepts"
        elif not concepts or len(concepts) != len(
            {
                _MENTION_CONCEPTS.get(candidate_id)
                for candidate_id in entity.supported_mention_candidate_ids
            }
        ):
            status = SemanticSupportStatus.UNKNOWN
            reason = "entity has an anchor outside the known-answer inventory"
        else:
            known_points = {
                point
                for evidence_id, point in (("ev-01", 1), ("ev-04", 2))
                if evidence_id in entity.evidence_ids
            }
            if entity.temporal_state.kind is TemporalKind.POINT:
                time_ok = entity.temporal_state.point in known_points
            elif entity.temporal_state.kind is TemporalKind.INTERVAL and known_points:
                time_ok = (
                    entity.temporal_state.start is None
                    or entity.temporal_state.start <= min(known_points)
                ) and (
                    entity.temporal_state.end is None
                    or max(known_points) <= entity.temporal_state.end
                )
            else:
                time_ok = entity.temporal_state.kind in {
                    TemporalKind.UNKNOWN,
                    TemporalKind.NOT_APPLICABLE,
                }
            status = (
                SemanticSupportStatus.SUPPORTED if time_ok else SemanticSupportStatus.UNSUPPORTED
            )
            reason = (
                "mention anchors and story state are supported"
                if time_ok
                else "entity temporal state uses a discourse position as story time"
            )
        entity_base[entity.entity_id] = _assessment(
            entity.entity_id,
            "entity",
            status,
            entity.evidence_ids,
            reason,
        )

    for event in draft.instance_graph.events:
        concept, expected_point = _event_concept(
            label=event.label,
            description=event.description,
            reification_reason=event.reification_reason,
            evidence_ids=event.evidence_ids,
        )
        if concept is None:
            status = SemanticSupportStatus.UNKNOWN
            reason = "event label does not resolve to one evidence-backed event"
            referents[event.event_id] = frozenset()
        else:
            referents[event.event_id] = frozenset({concept})
            time_ok = _story_time_supported(
                kind=event.occurrence_time.kind,
                point=event.occurrence_time.point,
                start=event.occurrence_time.start,
                end=event.occurrence_time.end,
                expected_point=expected_point,
            )
            status = (
                SemanticSupportStatus.SUPPORTED if time_ok else SemanticSupportStatus.UNSUPPORTED
            )
            reason = (
                "event and occurrence time match known evidence"
                if time_ok
                else "event occurrence uses an unsupported story coordinate"
            )
        event_base[event.event_id] = _assessment(
            event.event_id,
            "event",
            status,
            event.evidence_ids,
            reason,
        )

    predicates = {item.predicate_id: item for item in draft.local_schema.predicates}
    assertion_assessments: dict[str, SemanticGroundingAssessment] = {}
    assertion_facts: dict[str, str] = {}
    for assertion in draft.instance_graph.assertions:
        predicate = predicates[assertion.predicate_id]
        fact, status, reason = _assertion_fact(
            assertion,
            predicate_parent=predicate.parent_upper_relation,
            predicate_text=" ".join((predicate.label, predicate.definition)),
            referent_concepts=referents,
        )
        assessment = _assessment(
            assertion.assertion_id,
            "qualified_assertion",
            status,
            assertion.evidence_ids,
            reason,
            None if fact is None else fact.fact_id,
        )
        assertion_assessments[assertion.assertion_id] = assessment
        if fact is not None:
            assertion_facts[assertion.assertion_id] = fact.fact_id

    assertion_by_id = {item.assertion_id: item for item in draft.instance_graph.assertions}
    for entity in draft.instance_graph.entities:
        base = entity_base[entity.entity_id]
        supported_descriptions = [
            assertion_id
            for assertion_id in entity.description_assertion_ids
            if assertion_assessments[assertion_id].status is SemanticSupportStatus.SUPPORTED
        ]
        if base.status is SemanticSupportStatus.SUPPORTED and not supported_descriptions:
            base = _assessment(
                entity.entity_id,
                "entity",
                SemanticSupportStatus.UNSUPPORTED,
                entity.evidence_ids,
                "rich description has no semantically supported assertion",
            )
        elif supported_descriptions:
            linked_evidence = {
                evidence_id
                for assertion_id in supported_descriptions
                for evidence_id in assertion_by_id[assertion_id].evidence_ids
            }
            if not set(entity.evidence_ids).intersection(linked_evidence):
                base = _assessment(
                    entity.entity_id,
                    "entity",
                    SemanticSupportStatus.UNSUPPORTED,
                    entity.evidence_ids,
                    "description assertion does not share entity evidence",
                )
        assessments.append(base)

    event_type_ids = {item.contextual_type_id for item in draft.instance_graph.events}
    type_by_id = {item.type_id: item for item in draft.local_schema.contextual_types}
    for event in draft.instance_graph.events:
        base = event_base[event.event_id]
        supported_descriptions = [
            assertion_id
            for assertion_id in event.description_assertion_ids
            if assertion_assessments[assertion_id].status is SemanticSupportStatus.SUPPORTED
        ]
        if base.status is SemanticSupportStatus.SUPPORTED and not supported_descriptions:
            base = _assessment(
                event.event_id,
                "event",
                SemanticSupportStatus.UNSUPPORTED,
                event.evidence_ids,
                "rich event description has no semantically supported assertion",
            )
        event_type = type_by_id[event.contextual_type_id]
        if event_type.parent_upper_type != "event":
            base = _assessment(
                event.event_id,
                "event",
                SemanticSupportStatus.UNSUPPORTED,
                event.evidence_ids,
                "event is typed by a non-event upper type",
            )
        assessments.append(base)

    assessments.extend(assertion_assessments.values())
    supported_assertion_ids = {
        item.record_id
        for item in assertion_assessments.values()
        if item.status is SemanticSupportStatus.SUPPORTED
    }

    for proposition in draft.instance_graph.proposition_contents:
        linked = [
            assertion
            for assertion in draft.instance_graph.assertions
            if assertion.proposition_content_id == proposition.proposition_content_id
        ]
        supported = [
            assertion for assertion in linked if assertion.assertion_id in supported_assertion_ids
        ]
        if len(supported) == 1:
            fact_id = assertion_facts.get(supported[0].assertion_id)
            status = SemanticSupportStatus.SUPPORTED
            reason = "proposition is preserved by one supported holder-attributed assertion"
            referents[proposition.proposition_content_id] = frozenset({f"proposition:{fact_id}"})
        elif linked:
            fact_id = None
            status = SemanticSupportStatus.UNSUPPORTED
            reason = "proposition is linked only to unsupported attributed assertions"
        else:
            fact_id = None
            status = SemanticSupportStatus.UNKNOWN
            reason = "orphan proposition content has no holder-attributed assertion"
        assessments.append(
            _assessment(
                proposition.proposition_content_id,
                "proposition_content",
                status,
                proposition.evidence_ids,
                reason,
                fact_id,
            )
        )

    assertions_by_predicate: dict[str, list[SemanticGroundingAssessment]] = defaultdict(list)
    for assertion in draft.instance_graph.assertions:
        assertions_by_predicate[assertion.predicate_id].append(
            assertion_assessments[assertion.assertion_id]
        )
    for predicate in draft.local_schema.predicates:
        uses = assertions_by_predicate[predicate.predicate_id]
        status = (
            SemanticSupportStatus.SUPPORTED
            if uses and all(item.status is SemanticSupportStatus.SUPPORTED for item in uses)
            else SemanticSupportStatus.UNKNOWN
            if not uses
            else SemanticSupportStatus.UNSUPPORTED
        )
        assessments.append(
            _assessment(
                predicate.predicate_id,
                "local_predicate",
                status,
                predicate.evidence_ids,
                "predicate is grounded by supported assertions"
                if status is SemanticSupportStatus.SUPPORTED
                else "predicate lacks exclusively supported assertion use",
            )
        )

    supported_node_ids = {
        item.record_id
        for item in assessments
        if item.record_kind in {"entity", "event"}
        and item.status is SemanticSupportStatus.SUPPORTED
    }
    entity_type_ids = {item.contextual_type_id for item in draft.instance_graph.entities}
    for contextual_type in draft.local_schema.contextual_types:
        typed_nodes = [
            item
            for item in (*draft.instance_graph.entities, *draft.instance_graph.events)
            if item.contextual_type_id == contextual_type.type_id
        ]
        typed_supported = bool(typed_nodes) and all(
            (item.entity_id if hasattr(item, "entity_id") else item.event_id) in supported_node_ids
            for item in typed_nodes
        )
        parent_ok = not (
            contextual_type.type_id in event_type_ids and contextual_type.type_id in entity_type_ids
        ) and not (
            contextual_type.type_id in event_type_ids
            and contextual_type.parent_upper_type != "event"
        )
        status = (
            SemanticSupportStatus.SUPPORTED
            if typed_supported and parent_ok
            else SemanticSupportStatus.UNSUPPORTED
        )
        assessments.append(
            _assessment(
                contextual_type.type_id,
                "contextual_type",
                status,
                contextual_type.evidence_ids,
                "contextual type has supported, upper-compatible instances"
                if status is SemanticSupportStatus.SUPPORTED
                else "contextual type mixes incompatible or unsupported instances",
            )
        )

    supported_ids = {
        item.record_id for item in assessments if item.status is SemanticSupportStatus.SUPPORTED
    }
    if all(
        item.status is SemanticSupportStatus.SUPPORTED
        for item in assessments
        if item.record_kind in {"contextual_type", "local_predicate"}
    ):
        supported_ids.add(draft.local_schema.schema_id)
    mention_ids = set(_MENTION_CONCEPTS)
    event_candidate_ids = {
        candidate.candidate_id for record in evidence for candidate in record.event_candidates
    }
    for decision in draft.decisions:
        operator = decision.operator.value
        status = SemanticSupportStatus.SUPPORTED
        reason = "operator prerequisites are supported by known-answer anchors"
        input_concepts = {
            _MENTION_CONCEPTS[item]
            for item in decision.input_object_ids
            if item in _MENTION_CONCEPTS
        }
        if operator == "merge" and len(input_concepts) != 1:
            status = SemanticSupportStatus.UNSUPPORTED
            reason = "merge inputs are not one known entity"
        elif operator == "split" and len(input_concepts) < 2:
            status = SemanticSupportStatus.UNSUPPORTED
            reason = "split inputs do not contain distinct known entities"
        elif operator == "event_reification" and not (
            set(decision.input_object_ids).intersection(event_candidate_ids)
            and set(decision.created_object_ids).intersection(supported_ids)
        ):
            status = SemanticSupportStatus.UNKNOWN
            reason = "event reification is not tied to supported candidates and events"
        elif operator == "schema_relation" and not (
            draft.local_schema.schema_id in decision.created_object_ids
            or set(decision.created_object_ids).intersection(predicates)
        ):
            status = SemanticSupportStatus.UNKNOWN
            reason = "schema decision does not identify the supported local schema"
        elif operator == "contextual_type" and not set(decision.created_object_ids).intersection(
            type_by_id
        ):
            status = SemanticSupportStatus.UNKNOWN
            reason = "type decision does not identify a supported contextual type"
        elif operator == "temporal_qualification" and not set(
            decision.created_object_ids
        ).intersection(supported_assertion_ids):
            status = SemanticSupportStatus.UNKNOWN
            reason = "temporal decision does not identify a supported assertion"
        elif operator == "epistemic_qualification" and not set(
            decision.created_object_ids
        ).intersection(
            supported_assertion_ids
            | {item.proposition_content_id for item in draft.instance_graph.proposition_contents}
        ):
            status = SemanticSupportStatus.UNKNOWN
            reason = "epistemic decision does not identify supported attributed content"
        elif operator == "rare_preservation" and (
            "ev-06" not in decision.evidence_ids
            or not set(decision.created_object_ids).intersection(supported_assertion_ids)
        ):
            status = SemanticSupportStatus.UNSUPPORTED
            reason = "rare guard does not preserve the known pivotal credential fact"
        elif operator == "abstraction" and draft.local_schema.schema_id not in set(
            decision.created_object_ids
        ):
            status = SemanticSupportStatus.UNKNOWN
            reason = "abstraction decision does not identify the constructed schema"
        elif operator in {"selection", "compression", "supported_description"}:
            unknown_inputs = set(decision.input_object_ids) - (
                supported_ids | mention_ids | event_candidate_ids
            )
            if unknown_inputs:
                status = SemanticSupportStatus.UNKNOWN
                reason = "selection-only decision refers to semantically unsupported objects"
        assessments.append(
            _assessment(
                decision.decision_id,
                "ontology_decision",
                status,
                decision.evidence_ids,
                reason,
            )
        )

    return AcceptanceGroundingAudit(tuple(assessments))


__all__ = [
    "AcceptanceGroundingAudit",
    "SemanticGroundingAssessment",
    "SemanticSupportStatus",
    "audit_acceptance_semantic_grounding",
]
