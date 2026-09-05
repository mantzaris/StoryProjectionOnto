from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from story_projection_onto.contracts import (
    AbstractionLevel,
    DiscoursePosition,
    Document,
    EventCandidate,
    EvidencePacket,
    EvidenceRecord,
    LocalContextSchema,
    MentionCandidate,
    Passage,
    ProvenanceReference,
    ReleaseClass,
    RevelationPosition,
    RightsClass,
    SpoilerHorizon,
    to_model_visible_evidence,
)
from story_projection_onto.evidence import (
    EvidenceBoundaryCategory,
    EvidenceBoundaryError,
    EvidenceEqualityError,
    EvidenceIntegrityError,
    EvidencePackingError,
    HorizonRejectionReason,
    assert_evidence_packet_equality,
    assert_packet_matches_snapshot,
    audit_prequery_evidence_boundary,
    build_all_admissible_packet,
    build_evidence_snapshot,
    evidence_equality_hash,
    serialize_model_visible_packet,
    serialize_model_visible_snapshot,
)

UTC = UTC
CREATED_AT = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
SEALED_AT = CREATED_AT + timedelta(seconds=1)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def document(
    *,
    release_class: ReleaseClass = ReleaseClass.PUBLIC,
    rights_class: RightsClass = RightsClass.PUBLIC_SYNTHETIC,
) -> Document:
    return Document(
        document_id="document-1",
        corpus_id="corpus-1",
        edition="synthetic fixture v1",
        restricted_text_handle="fixture://document-1",
        source_text_hash=digest("complete synthetic source"),
        rights_class=rights_class,
        release_class=release_class,
    )


def passage(
    order: int,
    *,
    release_class: ReleaseClass = ReleaseClass.PUBLIC,
) -> Passage:
    return Passage(
        passage_id=f"passage-{order}",
        document_id="document-1",
        passage_order=order,
        restricted_text_handle=f"fixture://document-1/{order}",
        source_text_hash=digest(f"passage {order}"),
        discourse_position=DiscoursePosition(passage_order=order),
        release_class=release_class,
    )


def evidence(
    order: int,
    *,
    release_class: ReleaseClass = ReleaseClass.PUBLIC,
    mention_candidates: tuple[MentionCandidate, ...] = (),
    event_candidates: tuple[EventCandidate, ...] = (),
) -> EvidenceRecord:
    evidence_id = f"evidence-{order}"
    text = f"Witness {order} records a provisional observation."
    return EvidenceRecord(
        evidence_id=evidence_id,
        passage_id=f"passage-{order}",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=order),
        mention_candidates=mention_candidates,
        event_candidates=event_candidates,
        provenance=ProvenanceReference(
            provenance_id=f"provenance-{order}",
            evidence_id=evidence_id,
            extraction_method="hand-authored fixture",
            locator=f"fixture:{order}",
            source_artifact_hash=digest(f"passage {order}"),
            confidence=1.0,
        ),
        confidence=1.0,
        release_class=release_class,
    )


def horizon(maximum: int = 2) -> SpoilerHorizon:
    return SpoilerHorizon(
        horizon_id=f"horizon-{maximum}",
        max_discourse_position=DiscoursePosition(passage_order=maximum),
        max_revelation_position=RevelationPosition(revelation_order=maximum),
    )


def snapshot_assembly(
    *,
    records: tuple[EvidenceRecord, ...] | None = None,
    passage_records: tuple[Passage, ...] | None = None,
    maximum: int = 2,
    source_document: Document | None = None,
    release_class: ReleaseClass = ReleaseClass.PUBLIC,
):
    records = records or (evidence(3), evidence(1), evidence(2))
    passage_records = passage_records or (passage(3), passage(1), passage(2))
    return build_evidence_snapshot(
        snapshot_id="snapshot-1",
        corpus_id="corpus-1",
        world_or_window_id="development-world-1",
        horizon=horizon(maximum),
        documents=(source_document or document(),),
        passages=passage_records,
        evidence_records=records,
        index_config_hash=digest("query-blind-index-config-v1"),
        created_at=CREATED_AT,
        sealed_at=SEALED_AT,
        release_class=release_class,
    )


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        (
            {"candidate": {"canonicalEntityId": "entity-final"}},
            EvidenceBoundaryCategory.FINAL_ENTITY,
        ),
        (
            {"surface_data": {"predicate_id": "secret-predicate"}},
            EvidenceBoundaryCategory.FINAL_PREDICATE,
        ),
        (
            {"metadata": {"reified_events": ["event-1"]}},
            EvidenceBoundaryCategory.REIFIED_EVENT,
        ),
        (
            {"metadata": {"qualified_assertions": ["assertion-1"]}},
            EvidenceBoundaryCategory.QUALIFIED_TRUTH_ASSERTION,
        ),
        (
            {"metadata": {"query_relevance": 0.99}},
            EvidenceBoundaryCategory.QUERY_DERIVED,
        ),
        (
            {"metadata": {"gold_id": "gold-secret"}},
            EvidenceBoundaryCategory.SCORER_GOLD,
        ),
        (
            {"metadata": {"hidden_ontology": {"nodes": []}}},
            EvidenceBoundaryCategory.HIDDEN_ONTOLOGY_CHANNEL,
        ),
        (
            {"opaque_payload": '{"instance_graph":{"nodes":[]}}'},
            EvidenceBoundaryCategory.HIDDEN_ONTOLOGY_CHANNEL,
        ),
    ],
)
def test_prequery_boundary_rejects_semantic_and_gold_channels(
    payload: dict[str, object],
    category: EvidenceBoundaryCategory,
) -> None:
    with pytest.raises(EvidenceBoundaryError) as caught:
        audit_prequery_evidence_boundary(payload)
    assert category in {finding.category for finding in caught.value.findings}


def test_boundary_accepts_defeasible_candidates_and_does_not_parse_story_text() -> None:
    mention = MentionCandidate(
        candidate_id="mention-1",
        evidence_id="evidence-1",
        start_char=0,
        end_char=7,
        surface="Witness",
        surface_hash=digest("Witness"),
        provisional_type="person-like",
        coreference_scores={},
    )
    record = evidence(1, mention_candidates=(mention,))
    audit = audit_prequery_evidence_boundary(record)
    assert audit.accepted
    assert audit.inspected_field_count > 0

    # Copyrighted or synthetic evidence can itself contain JSON-like prose; text
    # is content, not a metadata side channel.
    audit_prequery_evidence_boundary({"text": '{"entity_id":"words appearing in source evidence"}'})


@pytest.mark.parametrize(
    ("start_char", "end_char", "message"),
    (
        (1, 8, "do not resolve"),
        (0, 10_000, "exceed"),
    ),
)
def test_evidence_record_rejects_mentions_not_bound_to_its_exact_text_slice(
    start_char: int,
    end_char: int,
    message: str,
) -> None:
    mention = MentionCandidate(
        candidate_id="mention-bad-offset",
        evidence_id="evidence-1",
        start_char=start_char,
        end_char=end_char,
        surface="Witness",
        surface_hash=digest("Witness"),
    )

    with pytest.raises(ValueError, match=message):
        evidence(1, mention_candidates=(mention,))


def test_evidence_record_rejects_unresolved_event_participant_mentions() -> None:
    text = "Witness 1 records a provisional observation."
    event = EventCandidate(
        candidate_id="event-1",
        evidence_id="evidence-1",
        trigger_start_char=text.index("records"),
        trigger_end_char=text.index("records") + len("records"),
        trigger_surface="records",
        participant_mention_candidate_ids=("missing-mention",),
        confidence=0.9,
    )

    with pytest.raises(ValueError, match="must resolve within"):
        evidence(1, event_candidates=(event,))


def test_boundary_rejects_a_constructed_model_even_under_an_innocent_key() -> None:
    constructed_schema = LocalContextSchema(
        schema_id="schema-1",
        contextual_types=(),
        predicates=(),
        abstraction=AbstractionLevel.ACTOR,
    )
    with pytest.raises(EvidenceBoundaryError) as caught:
        audit_prequery_evidence_boundary({"metadata": constructed_schema})
    assert EvidenceBoundaryCategory.UNSUPPORTED_MODEL in {
        finding.category for finding in caught.value.findings
    }


def test_snapshot_is_deterministic_all_admissible_and_logs_horizon_rejections() -> None:
    first = snapshot_assembly()
    second = snapshot_assembly(
        records=(evidence(2), evidence(3), evidence(1)),
        passage_records=(passage(2), passage(3), passage(1)),
    )

    assert first.snapshot.content_hash == second.snapshot.content_hash
    assert first.content_hash == second.content_hash
    assert first.snapshot.eligible_evidence_ids == ("evidence-1", "evidence-2")
    assert tuple(item.evidence_id for item in first.admissible_evidence) == (
        "evidence-1",
        "evidence-2",
    )
    assert len(first.horizon_rejections) == 1
    rejection = first.horizon_rejections[0]
    assert rejection.evidence_id == "evidence-3"
    assert rejection.reason is HorizonRejectionReason.DISCOURSE_AFTER_SPOILER_HORIZON
    assert rejection.evidence_discourse_position.passage_order == 3


def test_snapshot_fails_if_all_evidence_is_after_the_registered_horizon() -> None:
    with pytest.raises(EvidenceIntegrityError, match="admits no evidence"):
        snapshot_assembly(
            records=(evidence(2),),
            passage_records=(passage(2),),
            maximum=1,
        )


def test_horizon_filter_blocks_candidate_references_to_future_evidence() -> None:
    first_mention = MentionCandidate(
        candidate_id="mention-1",
        evidence_id="evidence-1",
        start_char=0,
        end_char=7,
        surface="Witness",
        surface_hash=digest("Witness"),
        provisional_type="person-like",
        alias_candidate_ids=("mention-3",),
    )
    future_mention = MentionCandidate(
        candidate_id="mention-3",
        evidence_id="evidence-3",
        start_char=0,
        end_char=7,
        surface="Witness",
        surface_hash=digest("Witness"),
        provisional_type="person-like",
    )
    with pytest.raises(EvidenceIntegrityError, match="outside the snapshot"):
        snapshot_assembly(
            records=(
                evidence(1, mention_candidates=(first_mention,)),
                evidence(3, mention_candidates=(future_mention,)),
            ),
            passage_records=(passage(1), passage(3)),
            maximum=2,
        )


def test_rights_and_release_are_monotonic_across_source_lineage() -> None:
    restricted_document = document(
        release_class=ReleaseClass.RESTRICTED,
        rights_class=RightsClass.RESTRICTED_COPYRIGHTED,
    )
    with pytest.raises(EvidenceIntegrityError, match="cannot publicize a restricted document"):
        snapshot_assembly(
            records=(evidence(1),),
            passage_records=(passage(1),),
            maximum=1,
            source_document=restricted_document,
        )

    with pytest.raises(EvidenceIntegrityError, match="public snapshot"):
        snapshot_assembly(
            records=(evidence(1, release_class=ReleaseClass.RESTRICTED),),
            passage_records=(passage(1, release_class=ReleaseClass.RESTRICTED),),
            maximum=1,
            source_document=restricted_document,
            release_class=ReleaseClass.PUBLIC,
        )

    restricted = snapshot_assembly(
        records=(evidence(1, release_class=ReleaseClass.RESTRICTED),),
        passage_records=(passage(1, release_class=ReleaseClass.RESTRICTED),),
        maximum=1,
        source_document=restricted_document,
        release_class=ReleaseClass.RESTRICTED,
    )
    assert restricted.snapshot.release_class is ReleaseClass.RESTRICTED


def test_all_admissible_packet_has_frozen_order_ranks_and_rejection_log() -> None:
    assembly = snapshot_assembly()
    packet = build_all_admissible_packet(
        assembly,
        packet_id="packet-1",
        created_at=SEALED_AT + timedelta(seconds=1),
        token_counter=lambda payload: len(payload.encode("utf-8")),
        max_evidence_tokens=100_000,
    )

    assert packet.retrieval_method.value == "all_admissible"
    assert packet.ordered_evidence_ids == assembly.snapshot.eligible_evidence_ids
    assert packet.evidence == assembly.admissible_evidence
    assert packet.ranks == {"evidence-1": 1, "evidence-2": 2}
    assert packet.scores == {}
    assert packet.horizon_rejections == ("evidence-3",)
    assert packet.token_count > 0


def test_all_admissible_packet_refuses_to_truncate_or_predate_snapshot() -> None:
    assembly = snapshot_assembly()
    with pytest.raises(EvidencePackingError, match="no evidence was truncated"):
        build_all_admissible_packet(
            assembly,
            packet_id="packet-too-large",
            created_at=SEALED_AT,
            token_counter=lambda _payload: 7,
            max_evidence_tokens=6,
        )
    with pytest.raises(EvidencePackingError, match="cannot predate"):
        build_all_admissible_packet(
            assembly,
            packet_id="packet-before-seal",
            created_at=CREATED_AT,
            token_counter=lambda _payload: 1,
        )


def test_model_visible_serialization_is_canonical_and_strictly_allowlisted() -> None:
    assembly = snapshot_assembly()
    snapshot_payload = json.loads(serialize_model_visible_snapshot(assembly))
    assert set(snapshot_payload) == {
        "content_hash",
        "evidence",
        "ordered_evidence_ids",
        "schema_version",
        "snapshot_hash",
    }
    assert "world_or_window_id" not in snapshot_payload
    assert "horizon_rejections" not in snapshot_payload

    packet = build_all_admissible_packet(
        assembly,
        packet_id="packet-1",
        created_at=SEALED_AT + timedelta(seconds=1),
        token_counter=lambda _payload: 10,
    )
    serialized = serialize_model_visible_packet(packet)
    assert serialized == serialize_model_visible_packet(packet)
    packet_payload = json.loads(serialized)
    assert set(packet_payload) == {
        "content_hash",
        "evidence",
        "ordered_evidence_ids",
        "packet_hash",
        "retrieval_method",
        "schema_version",
    }
    assert not {
        "packet_id",
        "snapshot_hash",
        "ranks",
        "scores",
        "token_count",
        "horizon_rejections",
        "created_at",
        "release_class",
    }.intersection(packet_payload)
    for visible_record in packet_payload["evidence"]:
        assert {
            "passage_id",
            "text_hash",
            "provenance",
            "confidence",
        }.issubset(visible_record)
        assert "release_class" not in visible_record
        source = next(
            item for item in packet.evidence if item.evidence_id == visible_record["evidence_id"]
        )
        assert visible_record["passage_id"] == source.passage_id
        assert visible_record["text_hash"] == source.text_hash
        assert visible_record["confidence"] == source.confidence
        assert visible_record["provenance"] == source.provenance.model_dump(mode="json")
    audit_prequery_evidence_boundary(packet_payload, namespace="test.model_visible")


def test_live_model_visible_evidence_rejects_missing_source_artifact_hash() -> None:
    source = evidence(1)
    unbound = EvidenceRecord(
        evidence_id=source.evidence_id,
        passage_id=source.passage_id,
        text=source.text,
        text_hash=source.text_hash,
        discourse_position=source.discourse_position,
        mention_candidates=source.mention_candidates,
        event_candidates=source.event_candidates,
        relation_phrase_candidates=source.relation_phrase_candidates,
        temporal_clues=source.temporal_clues,
        provenance=ProvenanceReference(
            provenance_id=source.provenance.provenance_id,
            evidence_id=source.evidence_id,
            extraction_method=source.provenance.extraction_method,
            locator=source.provenance.locator,
            source_artifact_hash=None,
            confidence=source.provenance.confidence,
        ),
        confidence=source.confidence,
        release_class=source.release_class,
    )

    with pytest.raises(ValueError, match="requires a source artifact hash"):
        to_model_visible_evidence(unbound)


def _reenvelope_packet(
    packet: EvidencePacket,
    *,
    packet_id: str,
    created_at: datetime,
    token_count: int | None = None,
) -> EvidencePacket:
    payload = packet.model_dump(mode="python", exclude={"content_hash"})
    payload["packet_id"] = packet_id
    payload["created_at"] = created_at
    if token_count is not None:
        payload["token_count"] = token_count
    return EvidencePacket.model_validate(payload)


def test_evidence_equality_hash_ignores_envelope_but_detects_input_change() -> None:
    assembly = snapshot_assembly()
    first = build_all_admissible_packet(
        assembly,
        packet_id="packet-c0",
        created_at=SEALED_AT + timedelta(seconds=1),
        token_counter=lambda _payload: 10,
    )
    second = _reenvelope_packet(
        first,
        packet_id="packet-c2",
        created_at=SEALED_AT + timedelta(seconds=20),
    )
    assert first.content_hash != second.content_hash
    assert evidence_equality_hash(first) == evidence_equality_hash(second)
    common_hash = assert_evidence_packet_equality({"C0": first, "C2": second})
    assert common_hash == evidence_equality_hash(first)

    changed = _reenvelope_packet(
        first,
        packet_id="packet-changed",
        created_at=SEALED_AT + timedelta(seconds=30),
        token_count=first.token_count + 1,
    )
    with pytest.raises(EvidenceEqualityError, match="condition evidence differs"):
        assert_evidence_packet_equality({"C0": first, "C2": changed})


def test_packet_snapshot_audit_rejects_reordering_even_when_packet_is_valid() -> None:
    assembly = snapshot_assembly()
    original = build_all_admissible_packet(
        assembly,
        packet_id="packet-1",
        created_at=SEALED_AT + timedelta(seconds=1),
        token_counter=lambda _payload: 10,
    )
    payload = original.model_dump(mode="python", exclude={"content_hash"})
    payload["packet_id"] = "packet-reordered"
    payload["evidence"] = tuple(reversed(original.evidence))
    payload["ordered_evidence_ids"] = tuple(reversed(original.ordered_evidence_ids))
    payload["ranks"] = {"evidence-2": 1, "evidence-1": 2}
    reordered = EvidencePacket.model_validate(payload)
    with pytest.raises(EvidenceIntegrityError, match="omits or reorders"):
        assert_packet_matches_snapshot(reordered, assembly)
