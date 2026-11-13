"""Query-blind evidence assembly and model-visible boundary enforcement.

This module deliberately stops before ontology construction.  It assembles a
sealed, horizon-bounded evidence snapshot from defeasible candidates and packs
all admissible evidence for a comparison unit.  It never resolves entities,
creates predicates, reifies events, or assigns query relevance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from numbers import Integral
from typing import Any, Literal, Self

from pydantic import model_validator

from story_projection_onto.contracts import (
    DiscoursePosition,
    Document,
    EventCandidate,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceSpan,
    ImmutableRecord,
    LegacyModelVisibleEvidenceRecord,
    MentionCandidate,
    ModelVisibleEvidencePacket,
    ModelVisibleEvidenceRecord,
    NonEmptyText,
    NonNegativeInt,
    Passage,
    ProvenanceReference,
    RelationPhraseCandidate,
    ReleaseClass,
    RetrievalMethod,
    RevelationPosition,
    RightsClass,
    Sha256Digest,
    SpoilerHorizon,
    TemporalClue,
    canonical_json,
    canonical_sha256,
    to_model_visible_evidence,
    to_model_visible_packet,
)
from story_projection_onto.temporal import discourse_is_within_horizon

DEFAULT_MAX_EVIDENCE_TOKENS = 6_144


class EvidenceBoundaryCategory(StrEnum):
    """Scientific boundary represented by a rejected field or object."""

    FINAL_ENTITY = "final_entity"
    FINAL_PREDICATE = "final_predicate"
    REIFIED_EVENT = "reified_event"
    QUALIFIED_TRUTH_ASSERTION = "qualified_truth_assertion"
    QUERY_DERIVED = "query_derived"
    SCORER_GOLD = "scorer_gold"
    HIDDEN_ONTOLOGY_CHANNEL = "hidden_ontology_channel"
    UNSUPPORTED_MODEL = "unsupported_model"


class EvidenceBoundaryFinding(ImmutableRecord):
    category: EvidenceBoundaryCategory
    path: NonEmptyText
    field_name: NonEmptyText
    message: NonEmptyText


class EvidenceBoundaryError(ValueError):
    """Raised when constructed or scorer-only semantics enter evidence."""

    def __init__(self, findings: Sequence[EvidenceBoundaryFinding]) -> None:
        self.findings = tuple(findings)
        summary = "; ".join(
            f"{item.category.value} at {item.path} ({item.field_name})" for item in self.findings
        )
        super().__init__(f"pre-query evidence boundary rejected: {summary}")


class EvidenceIntegrityError(ValueError):
    """Raised when evidence lineage, references, or release rules disagree."""


class EvidencePackingError(ValueError):
    """Raised when complete all-admissible packing cannot be performed."""


class EvidenceEqualityError(ValueError):
    """Raised when conditions do not receive identical frozen evidence."""


class EvidenceBoundaryAudit(ImmutableRecord):
    namespace: NonEmptyText
    audited_payload_hash: Sha256Digest
    inspected_value_count: NonNegativeInt
    inspected_field_count: NonNegativeInt
    accepted: Literal[True] = True


class RightsReleaseAudit(ImmutableRecord):
    corpus_id: NonEmptyText
    target_release_class: ReleaseClass
    document_count: NonNegativeInt
    passage_count: NonNegativeInt
    evidence_count: NonNegativeInt
    contains_restricted_source: bool
    accepted: Literal[True] = True


class HorizonRejectionReason(StrEnum):
    DISCOURSE_AFTER_SPOILER_HORIZON = "discourse_after_spoiler_horizon"


class HorizonRejection(ImmutableRecord):
    """Query-blind record of evidence withheld by the registered horizon."""

    evidence_id: NonEmptyText
    evidence_discourse_position: DiscoursePosition
    horizon_id: NonEmptyText
    maximum_discourse_position: DiscoursePosition
    reason: Literal[HorizonRejectionReason.DISCOURSE_AFTER_SPOILER_HORIZON] = (
        HorizonRejectionReason.DISCOURSE_AFTER_SPOILER_HORIZON
    )

    @model_validator(mode="after")
    def position_is_actually_after_horizon(self) -> Self:
        if (
            self.evidence_discourse_position.ordering_key
            <= self.maximum_discourse_position.ordering_key
        ):
            raise ValueError("a horizon rejection must identify evidence after the horizon")
        return self


class EvidenceSnapshotAssembly(ImmutableRecord):
    """A sealed snapshot plus its single-copy evidence and rejection ledger."""

    snapshot: EvidenceSnapshot
    admissible_evidence: tuple[EvidenceRecord, ...]
    horizon_rejections: tuple[HorizonRejection, ...] = ()
    boundary_audit_hash: Sha256Digest
    rights_release_audit_hash: Sha256Digest

    @model_validator(mode="after")
    def assembly_matches_snapshot(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.admissible_evidence)
        if evidence_ids != self.snapshot.eligible_evidence_ids:
            raise ValueError("assembled evidence must exactly match the snapshot order")
        rejected_ids = tuple(item.evidence_id for item in self.horizon_rejections)
        if len(rejected_ids) != len(set(rejected_ids)):
            raise ValueError("horizon rejection IDs must be unique")
        if set(evidence_ids).intersection(rejected_ids):
            raise ValueError("admissible and horizon-rejected evidence must be disjoint")
        if any(
            not discourse_is_within_horizon(
                item.discourse_position,
                self.snapshot.horizon,
            )
            for item in self.admissible_evidence
        ):
            raise ValueError("assembled evidence exceeds the snapshot horizon")
        if any(
            item.horizon_id != self.snapshot.horizon.horizon_id
            or item.maximum_discourse_position != self.snapshot.horizon.max_discourse_position
            for item in self.horizon_rejections
        ):
            raise ValueError("horizon rejection ledger does not match the snapshot horizon")
        if self.snapshot.release_class is ReleaseClass.PUBLIC and any(
            item.release_class is ReleaseClass.RESTRICTED for item in self.admissible_evidence
        ):
            raise ValueError("a public snapshot cannot assemble restricted evidence")
        return self


class ModelVisibleEvidenceSnapshot(ImmutableRecord):
    """Exact preconstruction allowlist; runner IDs and rejection logs stay hidden."""

    snapshot_hash: Sha256Digest
    evidence: tuple[ModelVisibleEvidenceRecord, ...]
    ordered_evidence_ids: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def visible_snapshot_order_matches(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if not evidence_ids:
            raise ValueError("model-visible snapshot evidence cannot be empty")
        if evidence_ids != self.ordered_evidence_ids:
            raise ValueError("model-visible snapshot evidence order is inconsistent")
        return self


_ALLOWED_EVIDENCE_MODELS = (
    DiscoursePosition,
    Document,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceSpan,
    EventCandidate,
    MentionCandidate,
    LegacyModelVisibleEvidenceRecord,
    ModelVisibleEvidencePacket,
    ModelVisibleEvidenceRecord,
    Passage,
    ProvenanceReference,
    RelationPhraseCandidate,
    RevelationPosition,
    SpoilerHorizon,
    TemporalClue,
    EvidenceSnapshotAssembly,
    HorizonRejection,
    ModelVisibleEvidenceSnapshot,
)

_BOUNDARY_KEY_EXCEPTIONS = frozenset(
    {
        "alias_candidate_ids",
        "event_candidate_ids",
        "event_candidates",
        "participant_mention_candidate_ids",
        "provisional_type",
        "relation_phrase_candidates",
        "schema_version",
        "target_candidate_ids",
        "temporal_clues",
    }
)

_FINAL_ENTITY_KEYS = frozenset(
    {
        "aliases",
        "canonical_entity_id",
        "canonical_mentions",
        "contextual_type",
        "entity",
        "entity_id",
        "entity_ids",
        "entity_partition",
        "entity_type",
        "final_entity_id",
        "mention_to_entity",
        "resolved_coreference",
    }
)
_FINAL_PREDICATE_KEYS = frozenset(
    {
        "canonical_relation",
        "local_predicate",
        "normalized_predicate",
        "normalized_relation",
        "predicate",
        "predicate_id",
        "predicate_ids",
        "predicates",
        "relation_definition",
        "relation_definitions",
    }
)
_REIFIED_EVENT_KEYS = frozenset(
    {
        "event",
        "event_graph",
        "event_id",
        "event_ids",
        "event_ontology",
        "event_reification",
        "events",
        "reified_event",
        "reified_events",
    }
)
_QUALIFIED_ASSERTION_KEYS = frozenset(
    {
        "assertion",
        "assertion_id",
        "assertion_ids",
        "assertions",
        "epistemic_scope",
        "holder_status",
        "narrative_commitment",
        "proposition_content",
        "qualified_assertion",
        "qualified_assertions",
        "temporal_scope",
        "truth_assertion",
        "validity_time",
    }
)
_QUERY_DERIVED_KEYS = frozenset(
    {
        "context",
        "context_id",
        "contextual_relevance",
        "lens",
        "query",
        "query_context",
        "query_relevance",
        "relevance",
        "relevance_score",
        "target",
        "why_matters",
    }
)
_SCORER_GOLD_KEYS = frozenset(
    {
        "community_gold",
        "community_label",
        "condition",
        "condition_id",
        "contrast_id",
        "contrastive_pair",
        "difficulty",
        "expected_effect",
        "experiment_id",
        "gold",
        "gold_id",
        "gold_projection",
        "pair_id",
        "pivotal",
        "rare",
        "rare_pivotal",
        "rarity_label",
        "scorer_metadata",
        "split",
        "world_id",
    }
)
_HIDDEN_ONTOLOGY_KEYS = frozenset(
    {
        "constructed_graph",
        "constructed_ontology",
        "hidden_graph",
        "hidden_ontology",
        "instance_graph",
        "local_context_schema",
        "local_schema",
        "ontology",
        "ontology_projection",
        "prebuilt_graph",
        "preontology",
        "sealed_graph",
        "side_channel",
    }
)

_TEXT_FIELDS_NOT_INTERPRETED_AS_METADATA = frozenset(
    {
        "edition",
        "extraction_method",
        "label",
        "locator",
        "normalized_expression",
        "restricted_text_handle",
        "surface",
        "surface_phrase",
        "text",
        "trigger_surface",
    }
)


def _normalize_key(key: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _classify_forbidden_key(key: str) -> EvidenceBoundaryCategory | None:
    normalized = _normalize_key(key)
    if normalized in _BOUNDARY_KEY_EXCEPTIONS:
        return None
    if normalized in _FINAL_ENTITY_KEYS:
        return EvidenceBoundaryCategory.FINAL_ENTITY
    if normalized in _FINAL_PREDICATE_KEYS:
        return EvidenceBoundaryCategory.FINAL_PREDICATE
    if normalized in _REIFIED_EVENT_KEYS:
        return EvidenceBoundaryCategory.REIFIED_EVENT
    if normalized in _QUALIFIED_ASSERTION_KEYS:
        return EvidenceBoundaryCategory.QUALIFIED_TRUTH_ASSERTION
    if normalized in _QUERY_DERIVED_KEYS:
        return EvidenceBoundaryCategory.QUERY_DERIVED
    if normalized in _SCORER_GOLD_KEYS:
        return EvidenceBoundaryCategory.SCORER_GOLD
    if normalized in _HIDDEN_ONTOLOGY_KEYS:
        return EvidenceBoundaryCategory.HIDDEN_ONTOLOGY_CHANNEL

    tokens = set(normalized.split("_"))
    is_candidate = "candidate" in tokens or "candidates" in tokens
    if "gold" in tokens or "scorer" in tokens:
        return EvidenceBoundaryCategory.SCORER_GOLD
    if "query" in tokens or "relevance" in tokens:
        return EvidenceBoundaryCategory.QUERY_DERIVED
    if "predicate" in tokens:
        return EvidenceBoundaryCategory.FINAL_PREDICATE
    if "assertion" in tokens or "proposition" in tokens or "commitment" in tokens:
        return EvidenceBoundaryCategory.QUALIFIED_TRUTH_ASSERTION
    if "reified" in tokens or "reification" in tokens:
        return EvidenceBoundaryCategory.REIFIED_EVENT
    if "ontology" in tokens or "preontology" in tokens or "graph" in tokens:
        return EvidenceBoundaryCategory.HIDDEN_ONTOLOGY_CHANNEL
    if "schema" in tokens and normalized != "schema_version":
        return EvidenceBoundaryCategory.HIDDEN_ONTOLOGY_CHANNEL
    if "entity" in tokens and not is_candidate:
        return EvidenceBoundaryCategory.FINAL_ENTITY
    if "event" in tokens and not is_candidate:
        return EvidenceBoundaryCategory.REIFIED_EVENT
    return None


def audit_prequery_evidence_boundary(
    value: Any,
    *,
    namespace: str = "prequery.evidence",
) -> EvidenceBoundaryAudit:
    """Audit an evidence-side value and reject every semantic escape channel.

    Candidate records are intentionally allowed.  Constructed semantic models
    are rejected by type, while mappings are recursively checked using
    normalized field names.  JSON-shaped strings are inspected unless the field
    is an actual text/surface field, preventing an opaque metadata field from
    carrying a serialized ontology.
    """

    findings: list[EvidenceBoundaryFinding] = []
    inspected_value_count = 0
    inspected_field_count = 0

    def add_finding(
        category: EvidenceBoundaryCategory,
        path: str,
        field_name: str,
        message: str,
    ) -> None:
        findings.append(
            EvidenceBoundaryFinding(
                category=category,
                path=path or "$",
                field_name=field_name,
                message=message,
            )
        )

    def visit(item: Any, path: str, parent_key: str | None = None) -> None:
        nonlocal inspected_field_count, inspected_value_count
        inspected_value_count += 1

        if isinstance(item, ImmutableRecord):
            if not isinstance(item, _ALLOWED_EVIDENCE_MODELS):
                add_finding(
                    EvidenceBoundaryCategory.UNSUPPORTED_MODEL,
                    path,
                    item.__class__.__name__,
                    "only evidence-side immutable record types are allowed pre-query",
                )
            fields = {name: getattr(item, name) for name in item.__class__.model_fields}
            visit(fields, path, parent_key)
            return

        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                if not isinstance(raw_key, str):
                    raise TypeError("pre-query evidence mappings require string keys")
                inspected_field_count += 1
                child_path = f"{path}.{raw_key}" if path else raw_key
                category = _classify_forbidden_key(raw_key)
                if category is not None:
                    add_finding(
                        category,
                        child_path,
                        raw_key,
                        "field carries information prohibited from query-blind evidence",
                    )
                visit(child, child_path, raw_key)
            return

        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", parent_key)
            return

        if isinstance(item, str) and parent_key is not None:
            normalized_parent = _normalize_key(parent_key)
            stripped = item.lstrip()
            if (
                normalized_parent not in _TEXT_FIELDS_NOT_INTERPRETED_AS_METADATA
                and stripped.startswith(("{", "["))
            ):
                try:
                    decoded = json.loads(item)
                except json.JSONDecodeError:
                    return
                visit(decoded, f"{path}<json>", parent_key)

    visit(value, namespace)
    if findings:
        raise EvidenceBoundaryError(findings)
    return EvidenceBoundaryAudit(
        namespace=namespace,
        audited_payload_hash=canonical_sha256(value),
        inspected_value_count=inspected_value_count,
        inspected_field_count=inspected_field_count,
    )


def enforce_rights_and_release(
    *,
    corpus_id: str,
    documents: Iterable[Document],
    passages: Iterable[Passage],
    evidence_records: Iterable[EvidenceRecord],
    target_release_class: ReleaseClass,
) -> RightsReleaseAudit:
    """Validate rights/release monotonicity over the source lineage.

    Descendants may be made more restrictive, but a record derived from a
    restricted ancestor may never be marked public.
    """

    document_items = tuple(sorted(documents, key=lambda item: item.document_id))
    passage_items = tuple(sorted(passages, key=lambda item: (item.passage_order, item.passage_id)))
    evidence_items = tuple(sorted(evidence_records, key=_evidence_order_key))
    document_by_id = {item.document_id: item for item in document_items}
    passage_by_id = {item.passage_id: item for item in passage_items}

    if len(document_by_id) != len(document_items):
        raise EvidenceIntegrityError("document IDs must be unique")
    if len(passage_by_id) != len(passage_items):
        raise EvidenceIntegrityError("passage IDs must be unique")
    if len({item.evidence_id for item in evidence_items}) != len(evidence_items):
        raise EvidenceIntegrityError("evidence IDs must be unique")
    if not document_items or not passage_items or not evidence_items:
        raise EvidenceIntegrityError(
            "snapshot construction requires documents, passages, and evidence"
        )

    for document in document_items:
        if document.corpus_id != corpus_id:
            raise EvidenceIntegrityError(
                f"document {document.document_id!r} belongs to another corpus"
            )

    for passage in passage_items:
        document = document_by_id.get(passage.document_id)
        if document is None:
            raise EvidenceIntegrityError(
                f"passage {passage.passage_id!r} references an unknown document"
            )
        if (
            document.release_class is ReleaseClass.RESTRICTED
            and passage.release_class is ReleaseClass.PUBLIC
        ):
            raise EvidenceIntegrityError(
                f"passage {passage.passage_id!r} cannot publicize a restricted document"
            )

    for record in evidence_items:
        passage = passage_by_id.get(record.passage_id)
        if passage is None:
            raise EvidenceIntegrityError(
                f"evidence {record.evidence_id!r} references an unknown passage"
            )
        if (
            passage.release_class is ReleaseClass.RESTRICTED
            and record.release_class is ReleaseClass.PUBLIC
        ):
            raise EvidenceIntegrityError(
                f"evidence {record.evidence_id!r} cannot publicize a restricted passage"
            )

    restricted = any(
        document.rights_class is RightsClass.RESTRICTED_COPYRIGHTED
        or document.release_class is ReleaseClass.RESTRICTED
        for document in document_items
    ) or any(
        item.release_class is ReleaseClass.RESTRICTED for item in (*passage_items, *evidence_items)
    )
    if target_release_class is ReleaseClass.PUBLIC and restricted:
        raise EvidenceIntegrityError(
            "a public snapshot cannot include restricted or copyrighted source lineage"
        )

    return RightsReleaseAudit(
        corpus_id=corpus_id,
        target_release_class=target_release_class,
        document_count=len(document_items),
        passage_count=len(passage_items),
        evidence_count=len(evidence_items),
        contains_restricted_source=restricted,
    )


def _assert_candidate_integrity(records: Sequence[EvidenceRecord]) -> None:
    mention_ids: set[str] = set()
    event_ids: set[str] = set()
    relation_ids: set[str] = set()
    clue_ids: set[str] = set()

    for record in records:
        for candidate in record.mention_candidates:
            if candidate.candidate_id in mention_ids:
                raise EvidenceIntegrityError(
                    f"duplicate mention candidate ID {candidate.candidate_id!r}"
                )
            mention_ids.add(candidate.candidate_id)
        for candidate in record.event_candidates:
            if candidate.candidate_id in event_ids:
                raise EvidenceIntegrityError(
                    f"duplicate event candidate ID {candidate.candidate_id!r}"
                )
            event_ids.add(candidate.candidate_id)
        for candidate in record.relation_phrase_candidates:
            if candidate.candidate_id in relation_ids:
                raise EvidenceIntegrityError(
                    f"duplicate relation candidate ID {candidate.candidate_id!r}"
                )
            relation_ids.add(candidate.candidate_id)
        for candidate in record.temporal_clues:
            if candidate.clue_id in clue_ids:
                raise EvidenceIntegrityError(f"duplicate temporal clue ID {candidate.clue_id!r}")
            clue_ids.add(candidate.clue_id)

    all_candidate_ids = mention_ids | event_ids | relation_ids | clue_ids
    total_candidate_count = len(mention_ids) + len(event_ids) + len(relation_ids) + len(clue_ids)
    if len(all_candidate_ids) != total_candidate_count:
        raise EvidenceIntegrityError("candidate IDs must be unique across candidate kinds")

    for record in records:
        for mention in record.mention_candidates:
            references = set(mention.alias_candidate_ids) | set(mention.coreference_scores)
            unknown = references - mention_ids
            if unknown:
                raise EvidenceIntegrityError(
                    f"mention {mention.candidate_id!r} references unknown mention candidates: "
                    f"{sorted(unknown)}"
                )
        for event in record.event_candidates:
            unknown = set(event.participant_mention_candidate_ids) - mention_ids
            if unknown:
                raise EvidenceIntegrityError(
                    f"event candidate {event.candidate_id!r} references unknown mentions: "
                    f"{sorted(unknown)}"
                )
        for relation in record.relation_phrase_candidates:
            endpoints = {relation.subject_mention_candidate_id}
            if relation.object_mention_candidate_id is not None:
                endpoints.add(relation.object_mention_candidate_id)
            unknown = endpoints - mention_ids
            if unknown:
                raise EvidenceIntegrityError(
                    f"relation candidate {relation.candidate_id!r} references unknown mentions: "
                    f"{sorted(unknown)}"
                )
        for clue in record.temporal_clues:
            unknown = set(clue.target_candidate_ids) - all_candidate_ids
            if unknown:
                raise EvidenceIntegrityError(
                    f"temporal clue {clue.clue_id!r} references unknown candidates: "
                    f"{sorted(unknown)}"
                )


def _evidence_order_key(record: EvidenceRecord) -> tuple[int, int, int, str]:
    return (*record.discourse_position.ordering_key, record.evidence_id)


def build_evidence_snapshot(
    *,
    snapshot_id: str,
    corpus_id: str,
    world_or_window_id: str,
    horizon: SpoilerHorizon,
    documents: Iterable[Document],
    passages: Iterable[Passage],
    evidence_records: Iterable[EvidenceRecord],
    index_config_hash: str,
    created_at: datetime,
    sealed_at: datetime,
    release_class: ReleaseClass,
) -> EvidenceSnapshotAssembly:
    """Build a deterministic query-blind snapshot and horizon rejection log."""

    document_items = tuple(sorted(documents, key=lambda item: item.document_id))
    passage_items = tuple(sorted(passages, key=lambda item: (item.passage_order, item.passage_id)))
    evidence_items = tuple(sorted(evidence_records, key=_evidence_order_key))
    boundary_audit = audit_prequery_evidence_boundary(
        {
            "documents": document_items,
            "passages": passage_items,
            "evidence": evidence_items,
            "horizon": horizon,
        },
        namespace="prequery.index",
    )
    rights_audit = enforce_rights_and_release(
        corpus_id=corpus_id,
        documents=document_items,
        passages=passage_items,
        evidence_records=evidence_items,
        target_release_class=release_class,
    )
    _assert_candidate_integrity(evidence_items)

    ordered_evidence = evidence_items
    admissible = tuple(
        item
        for item in ordered_evidence
        if discourse_is_within_horizon(item.discourse_position, horizon)
    )
    if not admissible:
        raise EvidenceIntegrityError("the registered horizon admits no evidence")

    # Run reference integrity again after filtering.  This blocks even an opaque
    # candidate ID from pointing through the spoiler horizon.
    try:
        _assert_candidate_integrity(admissible)
    except EvidenceIntegrityError as error:
        raise EvidenceIntegrityError(
            f"horizon filtering left a candidate reference outside the snapshot: {error}"
        ) from error

    rejections = tuple(
        HorizonRejection(
            evidence_id=item.evidence_id,
            evidence_discourse_position=item.discourse_position,
            horizon_id=horizon.horizon_id,
            maximum_discourse_position=horizon.max_discourse_position,
        )
        for item in ordered_evidence
        if not discourse_is_within_horizon(item.discourse_position, horizon)
    )
    snapshot = EvidenceSnapshot(
        snapshot_id=snapshot_id,
        corpus_id=corpus_id,
        world_or_window_id=world_or_window_id,
        horizon=horizon,
        eligible_evidence_ids=tuple(item.evidence_id for item in admissible),
        index_config_hash=index_config_hash,
        created_at=created_at,
        sealed_at=sealed_at,
        release_class=release_class,
    )
    return EvidenceSnapshotAssembly(
        snapshot=snapshot,
        admissible_evidence=admissible,
        horizon_rejections=rejections,
        boundary_audit_hash=boundary_audit.content_hash,
        rights_release_audit_hash=rights_audit.content_hash,
    )


def to_model_visible_snapshot(
    assembly: EvidenceSnapshotAssembly,
) -> ModelVisibleEvidenceSnapshot:
    """Project a snapshot through an explicit model-visible allowlist."""

    audit_prequery_evidence_boundary(assembly, namespace="prequery.snapshot")
    visible = ModelVisibleEvidenceSnapshot(
        snapshot_hash=assembly.snapshot.content_hash,
        evidence=tuple(
            to_model_visible_evidence(record) for record in assembly.admissible_evidence
        ),
        ordered_evidence_ids=assembly.snapshot.eligible_evidence_ids,
    )
    audit_prequery_evidence_boundary(visible, namespace="model_visible.snapshot")
    return visible


def serialize_model_visible_snapshot(assembly: EvidenceSnapshotAssembly) -> str:
    """Return canonical JSON containing only the C1 evidence allowlist."""

    visible = to_model_visible_snapshot(assembly)
    serialized = canonical_json(visible)
    # A strict typed round trip protects this function if serializers are later
    # refactored to manipulate dictionaries.
    ModelVisibleEvidenceSnapshot.model_validate(json.loads(serialized))
    return serialized


def build_all_admissible_packet(
    assembly: EvidenceSnapshotAssembly,
    *,
    packet_id: str,
    created_at: datetime,
    token_counter: Callable[[str], int],
    max_evidence_tokens: int = DEFAULT_MAX_EVIDENCE_TOKENS,
) -> EvidencePacket:
    """Pack every admissible record in frozen order, or reject without truncation."""

    if max_evidence_tokens <= 0:
        raise EvidencePackingError("max_evidence_tokens must be positive")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise EvidencePackingError("packet created_at must be timezone-aware")
    if created_at < assembly.snapshot.sealed_at:
        raise EvidencePackingError("an evidence packet cannot predate its sealed snapshot")

    visible = to_model_visible_snapshot(assembly)
    countable_payload = canonical_json(
        {
            "evidence": visible.evidence,
            "ordered_evidence_ids": visible.ordered_evidence_ids,
        }
    )
    counted = token_counter(countable_payload)
    if isinstance(counted, bool) or not isinstance(counted, Integral) or counted < 0:
        raise EvidencePackingError("token_counter must return a non-negative integer")
    token_count = int(counted)
    if token_count > max_evidence_tokens:
        raise EvidencePackingError(
            "complete all-admissible evidence does not fit: "
            f"{token_count} tokens exceeds {max_evidence_tokens}; no evidence was truncated"
        )

    evidence_ids = assembly.snapshot.eligible_evidence_ids
    packet = EvidencePacket(
        packet_id=packet_id,
        snapshot_hash=assembly.snapshot.content_hash,
        evidence=assembly.admissible_evidence,
        ordered_evidence_ids=evidence_ids,
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        ranks={evidence_id: rank for rank, evidence_id in enumerate(evidence_ids, start=1)},
        scores={},
        token_count=token_count,
        horizon_rejections=tuple(item.evidence_id for item in assembly.horizon_rejections),
        created_at=created_at,
        release_class=assembly.snapshot.release_class,
    )
    assert_packet_matches_snapshot(packet, assembly)
    return packet


def assert_packet_matches_snapshot(
    packet: EvidencePacket,
    assembly: EvidenceSnapshotAssembly,
) -> None:
    """Block packet substitution, omission, reordering, or horizon leakage."""

    if packet.snapshot_hash != assembly.snapshot.content_hash:
        raise EvidenceIntegrityError("packet references a different evidence snapshot")
    if packet.retrieval_method is not RetrievalMethod.ALL_ADMISSIBLE:
        raise EvidenceIntegrityError("primary snapshot packets must use all_admissible retrieval")
    if packet.ordered_evidence_ids != assembly.snapshot.eligible_evidence_ids:
        raise EvidenceIntegrityError("packet omits or reorders sealed admissible evidence")
    if packet.evidence != assembly.admissible_evidence:
        raise EvidenceIntegrityError("packet evidence content differs from the sealed snapshot")
    expected_rejections = tuple(item.evidence_id for item in assembly.horizon_rejections)
    if packet.horizon_rejections != expected_rejections:
        raise EvidenceIntegrityError("packet horizon rejection ledger differs from the snapshot")
    if packet.release_class is not assembly.snapshot.release_class:
        raise EvidenceIntegrityError("packet release class differs from the snapshot")
    audit_prequery_evidence_boundary(packet, namespace="packet.evidence")


def serialize_model_visible_packet(packet: EvidencePacket) -> str:
    """Return canonical JSON containing only the query-time evidence allowlist."""

    audit_prequery_evidence_boundary(packet, namespace="packet.evidence")
    visible = to_model_visible_packet(packet)
    audit_prequery_evidence_boundary(visible, namespace="model_visible.packet")
    serialized = canonical_json(visible)
    ModelVisibleEvidencePacket.model_validate(json.loads(serialized))
    return serialized


def serialize_model_visible_packet_bytes(packet: EvidencePacket) -> bytes:
    """UTF-8 byte form used for exact prompt/artifact hashing."""

    return serialize_model_visible_packet(packet).encode("utf-8")


def evidence_equality_hash(packet: EvidencePacket) -> str:
    """Hash all condition-relevant evidence while ignoring run-local envelope data.

    ``packet_id`` and ``created_at`` are intentionally absent so independently
    materialized envelopes compare equal.  Evidence contents/order, snapshot,
    retrieval metadata, horizon exclusions, token count, and release policy are
    included.
    """

    visible = to_model_visible_packet(packet)
    payload = {
        "snapshot_hash": packet.snapshot_hash,
        "evidence": visible.evidence,
        "ordered_evidence_ids": packet.ordered_evidence_ids,
        "retrieval_method": packet.retrieval_method,
        "ranks": packet.ranks,
        "scores": packet.scores,
        "token_count": packet.token_count,
        "horizon_rejections": packet.horizon_rejections,
        "release_class": packet.release_class,
    }
    audit_prequery_evidence_boundary(payload, namespace="evidence_equality")
    return canonical_sha256(payload)


def assert_evidence_packet_equality(
    packets_by_condition: Mapping[str, EvidencePacket],
) -> str:
    """Return the common equality hash or reject a fairness violation."""

    if not packets_by_condition:
        raise EvidenceEqualityError("at least one packet is required for equality audit")
    hashes = {
        condition: evidence_equality_hash(packet)
        for condition, packet in packets_by_condition.items()
    }
    if len(set(hashes.values())) != 1:
        details = ", ".join(f"{condition}={digest}" for condition, digest in sorted(hashes.items()))
        raise EvidenceEqualityError(f"condition evidence differs: {details}")
    return next(iter(hashes.values()))
