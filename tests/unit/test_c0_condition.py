from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from story_projection_onto.conditions.base import ProduceInputs, RunConditionConfig
from story_projection_onto.conditions.c0 import (
    ClassicalEntityKind,
    ClassicalPreBuilder,
    ClassicalRuleConfig,
    RuleCandidateBackend,
    SpacyCandidateBackend,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    ConditionName,
    DiscoursePosition,
    Document,
    EventCandidate,
    EvidencePacket,
    EvidenceRecord,
    MentionCandidate,
    OutputBudgets,
    Passage,
    ProvenanceReference,
    QueryContext,
    RelationPhraseCandidate,
    ReleaseClass,
    RetrievalMethod,
    RevelationPosition,
    RightsClass,
    SpoilerHorizon,
    StoryTime,
    TemporalClue,
    TemporalKind,
    UpperOntology,
)
from story_projection_onto.evidence import build_evidence_snapshot

BASE = datetime(2026, 9, 3, 10, 0, tzinfo=UTC)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mention(
    evidence_id: str,
    candidate_id: str,
    text: str,
    surface: str,
    provisional_type: str,
    *,
    alias_ids: tuple[str, ...] = (),
) -> MentionCandidate:
    start = text.index(surface)
    return MentionCandidate(
        candidate_id=candidate_id,
        evidence_id=evidence_id,
        start_char=start,
        end_char=start + len(surface),
        surface=surface,
        surface_hash=digest(surface),
        provisional_type=provisional_type,
        alias_candidate_ids=alias_ids,
    )


def evidence_fixture() -> tuple[EvidenceRecord, ...]:
    first_text = "On day 2, Captain Mira reported Rowan to Harbor Guild."
    second_text = "On day 3, Rowan joined Harbor Guild and attacked Taren."
    third_text = "From day 4 to day 6, Lady Mira supported Taren."
    first_mentions = (
        mention("evidence-1", "m-mira-1", first_text, "Captain Mira", "person"),
        mention("evidence-1", "m-rowan-1", first_text, "Rowan", "person", alias_ids=("m-rowan-2",)),
        mention(
            "evidence-1",
            "m-guild-1",
            first_text,
            "Harbor Guild",
            "collective",
            alias_ids=("m-guild-2",),
        ),
    )
    second_mentions = (
        mention(
            "evidence-2", "m-rowan-2", second_text, "Rowan", "person", alias_ids=("m-rowan-1",)
        ),
        mention(
            "evidence-2",
            "m-guild-2",
            second_text,
            "Harbor Guild",
            "collective",
            alias_ids=("m-guild-1",),
        ),
        mention("evidence-2", "m-taren-2", second_text, "Taren", "person"),
    )
    third_mentions = (
        mention(
            "evidence-3", "m-mira-3", third_text, "Lady Mira", "person", alias_ids=("m-mira-1",)
        ),
        mention("evidence-3", "m-taren-3", third_text, "Taren", "person", alias_ids=("m-taren-2",)),
    )
    rows = []
    specifications = (
        (
            "evidence-1",
            first_text,
            first_mentions,
            (
                RelationPhraseCandidate(
                    candidate_id="r-report",
                    evidence_id="evidence-1",
                    subject_mention_candidate_id="m-mira-1",
                    object_mention_candidate_id="m-rowan-1",
                    surface_phrase="reported",
                    confidence=0.96,
                ),
            ),
            (),
        ),
        (
            "evidence-2",
            second_text,
            second_mentions,
            (
                RelationPhraseCandidate(
                    candidate_id="r-member",
                    evidence_id="evidence-2",
                    subject_mention_candidate_id="m-rowan-2",
                    object_mention_candidate_id="m-guild-2",
                    surface_phrase="joined",
                    confidence=0.95,
                ),
            ),
            (
                EventCandidate(
                    candidate_id="event-attack",
                    evidence_id="evidence-2",
                    trigger_start_char=second_text.index("attacked"),
                    trigger_end_char=second_text.index("attacked") + len("attacked"),
                    trigger_surface="attacked",
                    participant_mention_candidate_ids=("m-rowan-2", "m-taren-2"),
                    confidence=0.93,
                ),
            ),
        ),
        (
            "evidence-3",
            third_text,
            third_mentions,
            (
                RelationPhraseCandidate(
                    candidate_id="r-support",
                    evidence_id="evidence-3",
                    subject_mention_candidate_id="m-mira-3",
                    object_mention_candidate_id="m-taren-3",
                    surface_phrase="supported",
                    confidence=0.94,
                ),
            ),
            (),
        ),
    )
    for order, (evidence_id, text, mentions, relations, events) in enumerate(specifications, 1):
        rows.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                passage_id=f"passage-{order}",
                text=text,
                text_hash=digest(text),
                discourse_position=DiscoursePosition(passage_order=order),
                mention_candidates=mentions,
                event_candidates=events,
                relation_phrase_candidates=relations,
                temporal_clues=(
                    TemporalClue(
                        clue_id=f"time-{order}",
                        evidence_id=evidence_id,
                        normalized_expression=(
                            "day 2"
                            if order == 1
                            else "day 3"
                            if order == 2
                            else "day 4 through day 6"
                        ),
                        target_candidate_ids=tuple(
                            item.candidate_id for item in (*relations, *events)
                        ),
                        confidence=1.0,
                    ),
                ),
                provenance=ProvenanceReference(
                    provenance_id=f"provenance-{order}",
                    evidence_id=evidence_id,
                    extraction_method="hand-authored classical fixture",
                    locator=f"fixture:{order}",
                    source_artifact_hash=digest(text),
                    confidence=1.0,
                ),
                confidence=1.0,
                release_class=ReleaseClass.PUBLIC,
            )
        )
    return tuple(rows)


def upper() -> UpperOntology:
    return UpperOntology(
        ontology_id="upper-v1",
        primitive_types=("entity", "event", "proposition"),
        primitive_relations=("related_to",),
        temporal_terms=("story_time", "validity_time", "discourse", "revelation"),
        epistemic_terms=("known", "believed", "reported", "denied"),
        revision="v1",
    )


def semantic_budgets() -> OutputBudgets:
    return OutputBudgets(
        node_budget=20,
        assertion_budget=30,
        display_node_budget=20,
        display_assertion_budget=30,
    )


def snapshot_and_packet():
    evidence = evidence_fixture()
    horizon = SpoilerHorizon(
        horizon_id="horizon-3",
        max_discourse_position=DiscoursePosition(passage_order=3),
        max_revelation_position=RevelationPosition(revelation_order=3),
    )
    document = Document(
        document_id="document-1",
        corpus_id="synthetic",
        edition="fixture-v1",
        restricted_text_handle="fixture://document",
        source_text_hash=digest("fixture-document"),
        rights_class=RightsClass.PUBLIC_SYNTHETIC,
        release_class=ReleaseClass.PUBLIC,
    )
    passages = tuple(
        Passage(
            passage_id=f"passage-{index}",
            document_id="document-1",
            passage_order=index,
            restricted_text_handle=f"fixture://passage/{index}",
            source_text_hash=digest(record.text),
            discourse_position=record.discourse_position,
            release_class=ReleaseClass.PUBLIC,
        )
        for index, record in enumerate(evidence, 1)
    )
    assembly = build_evidence_snapshot(
        snapshot_id="snapshot-c0",
        corpus_id="synthetic",
        world_or_window_id="development-c0",
        horizon=horizon,
        documents=(document,),
        passages=passages,
        evidence_records=evidence,
        index_config_hash=digest("index-v1"),
        created_at=BASE,
        sealed_at=BASE + timedelta(minutes=1),
        release_class=ReleaseClass.PUBLIC,
    )
    packet = EvidencePacket(
        packet_id="packet-c0",
        snapshot_hash=assembly.snapshot.content_hash,
        evidence=evidence,
        ordered_evidence_ids=assembly.snapshot.eligible_evidence_ids,
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        token_count=80,
        created_at=BASE + timedelta(hours=2),
        release_class=ReleaseClass.PUBLIC,
    )
    return assembly.snapshot, evidence, packet


def test_c0_query_blind_prebuild_covers_entity_relation_event_time_and_epistemic() -> None:
    snapshot, evidence, _ = snapshot_and_packet()
    builder = ClassicalPreBuilder()
    preparation = builder.prepare(
        snapshot=snapshot,
        evidence=evidence,
        upper_ontology=upper(),
        preconstruction_budgets=semantic_budgets(),
        constructed_at=BASE + timedelta(minutes=2),
        sealed_at=BASE + timedelta(minutes=3),
    )
    preontology = preparation.sealed_preontology
    assert preontology is not None
    graph = preontology.draft.instance_graph

    # Title stripping and explicit alias candidates merge both Mira mentions.
    mira = next(item for item in graph.entities if "Mira" in item.label)
    assert set(mira.supported_mention_candidate_ids) == {"m-mira-1", "m-mira-3"}
    assert graph.events
    assert any(item.predicate_id == "c0-predicate-member_of" for item in graph.assertions)
    report = next(item for item in graph.assertions if item.predicate_id == "c0-predicate-reported")
    assert report.narrative_commitment.value == "holder_attributed"
    assert report.epistemic_scope is not None
    assert not any(
        item.predicate_id == report.predicate_id
        and item.subject_id == report.subject_id
        and item.object_id == report.object_id
        and item.narrative_commitment.value == "world_committed"
        for item in graph.assertions
    )
    assert any(
        item.temporal_scope.story_time.kind in {TemporalKind.POINT, TemporalKind.INTERVAL}
        for item in graph.assertions
    )
    assert all(
        decision.decided_at <= preontology.construction_seal.sealed_at
        for decision in preontology.draft.decisions
    )


def test_c0_query_projection_selects_only_sealed_ids_under_equal_final_budgets() -> None:
    snapshot, evidence, packet = snapshot_and_packet()
    upper_ontology = upper()
    builder = ClassicalPreBuilder()
    preparation = builder.prepare(
        snapshot=snapshot,
        evidence=evidence,
        upper_ontology=upper_ontology,
        preconstruction_budgets=semantic_budgets(),
        constructed_at=BASE + timedelta(minutes=2),
        sealed_at=BASE + timedelta(minutes=3),
    )
    final_budgets = OutputBudgets(
        node_budget=5,
        assertion_budget=6,
        display_node_budget=5,
        display_assertion_budget=6,
    )
    context = QueryContext(
        context_id="context-c0",
        wording="Who participated in the attack on day three?",
        lens="event participation and conflict",
        target="attack",
        story_scope=StoryTime(kind=TemporalKind.POINT, point=3, label="day three"),
        spoiler_horizon=snapshot.horizon,
        abstraction=AbstractionLevel.EVENT_ROLE,
        budgets=final_budgets,
        revealed_at=BASE + timedelta(hours=2),
    )
    config = RunConditionConfig(
        config_id="c0-run-config",
        condition=ConditionName.C0_CLASSICAL_PRE,
        budgets=final_budgets,
        maximum_input_tokens=10_240,
        maximum_output_tokens=2_048,
        validator_hash=digest("validator-v1"),
        upper_ontology_hash=upper_ontology.content_hash,
    )
    inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        upper_ontology=upper_ontology,
        run_config=config,
    )
    attempt = builder.produce(inputs)
    projection = attempt.projection
    assert projection is not None
    assert attempt.outcome.value == "succeeded"
    assert {item.operator.value for item in projection.decisions} == {"selection"}
    projected_ids = {
        *(item.entity_id for item in projection.instance_graph.entities),
        *(item.event_id for item in projection.instance_graph.events),
        *(item.assertion_id for item in projection.instance_graph.assertions),
    }
    assert projected_ids.issubset(
        preparation.sealed_preontology.construction_seal.sealed_object_ids
    )
    assert projection.budget_accounting.nodes_used <= final_budgets.node_budget
    assert projection.budget_accounting.assertions_used <= final_budgets.assertion_budget


def test_c0_preconstruction_refuses_incomplete_or_post_horizon_evidence() -> None:
    snapshot, evidence, _ = snapshot_and_packet()
    builder = ClassicalPreBuilder()
    with pytest.raises(ValueError, match="complete ordered snapshot evidence"):
        builder.prepare(
            snapshot=snapshot,
            evidence=evidence[:-1],
            upper_ontology=upper(),
            preconstruction_budgets=semantic_budgets(),
            constructed_at=BASE + timedelta(minutes=2),
            sealed_at=BASE + timedelta(minutes=3),
        )


def test_regex_fallback_is_deterministic_without_downloaded_spacy_model() -> None:
    text = "On day 5, Mira warned Rowan."
    record = EvidenceRecord(
        evidence_id="fallback-evidence",
        passage_id="fallback-passage",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        provenance=ProvenanceReference(
            provenance_id="fallback-provenance",
            evidence_id="fallback-evidence",
            extraction_method="fixture",
            locator="fixture:fallback",
            confidence=1.0,
        ),
        confidence=1.0,
        release_class=ReleaseClass.PUBLIC,
    )
    backend = RuleCandidateBackend()
    first = backend.analyze(record, ClassicalRuleConfig())
    second = backend.analyze(record, ClassicalRuleConfig())
    assert first == second
    assert {item.surface for item in first.mentions} == {"Mira", "Rowan"}
    assert first.events and first.temporal_expressions


class FakeSpacyToken:
    def __init__(
        self,
        *,
        text: str,
        index: int,
        start_char: int,
        lemma: str,
        dependency: str,
        part_of_speech: str,
    ) -> None:
        self.text = text
        self.i = index
        self.idx = start_char
        self.lemma_ = lemma
        self.dep_ = dependency
        self.pos_ = part_of_speech
        self.head: FakeSpacyToken = self
        self._children: list[FakeSpacyToken] = []

    @property
    def children(self) -> tuple[FakeSpacyToken, ...]:
        return tuple(self._children)


class FakeSpacyEntity:
    def __init__(self, *, text: str, start_char: int, label: str) -> None:
        self.text = text
        self.start_char = start_char
        self.end_char = start_char + len(text)
        self.label_ = label


class FakeSpacyDoc:
    def __init__(
        self,
        tokens: tuple[FakeSpacyToken, ...],
        entities: tuple[FakeSpacyEntity, ...],
    ) -> None:
        self._tokens = tokens
        self.ents = entities

    def __iter__(self):
        return iter(self._tokens)


class FakeSpacyLanguage:
    pipe_names = ("tok2vec", "ner", "parser")

    def __init__(self, *, text: str, doc: FakeSpacyDoc) -> None:
        self._text = text
        self._doc = doc

    def __call__(self, text: str) -> FakeSpacyDoc:
        assert text == self._text
        return self._doc


def fake_token(
    text: str,
    surface: str,
    index: int,
    dependency: str,
    part_of_speech: str,
    *,
    lemma: str | None = None,
    after: int = 0,
) -> FakeSpacyToken:
    return FakeSpacyToken(
        text=surface,
        index=index,
        start_char=text.index(surface, after),
        lemma=lemma or surface.casefold(),
        dependency=dependency,
        part_of_speech=part_of_speech,
    )


def attach(child: FakeSpacyToken, head: FakeSpacyToken) -> None:
    child.head = head
    head._children.append(child)


def test_spacy_dependencies_fill_missing_relations_and_events_without_replacing_indexed() -> None:
    text = "Rowan was warned by Mira before Taren joined Harbor Guild."
    mira = mention("spacy-evidence", "m-mira", text, "Mira", "person")
    rowan = mention("spacy-evidence", "m-rowan", text, "Rowan", "person")
    taren = mention("spacy-evidence", "m-taren", text, "Taren", "person")
    guild = mention("spacy-evidence", "m-guild", text, "Harbor Guild", "collective")
    record = EvidenceRecord(
        evidence_id="spacy-evidence",
        passage_id="spacy-passage",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        mention_candidates=(mira, rowan, taren, guild),
        event_candidates=(
            EventCandidate(
                candidate_id="event-indexed-warn",
                evidence_id="spacy-evidence",
                trigger_start_char=text.index("warned"),
                trigger_end_char=text.index("warned") + len("warned"),
                trigger_surface="warned",
                participant_mention_candidate_ids=("m-mira", "m-rowan"),
                confidence=0.97,
            ),
        ),
        relation_phrase_candidates=(
            RelationPhraseCandidate(
                candidate_id="relation-indexed-warn",
                evidence_id="spacy-evidence",
                subject_mention_candidate_id="m-mira",
                object_mention_candidate_id="m-rowan",
                surface_phrase="warned",
                confidence=0.96,
            ),
        ),
        provenance=ProvenanceReference(
            provenance_id="spacy-provenance",
            evidence_id="spacy-evidence",
            extraction_method="fixture",
            locator="fixture:spacy",
            confidence=1.0,
        ),
        confidence=1.0,
        release_class=ReleaseClass.PUBLIC,
    )

    rowan_token = fake_token(text, "Rowan", 0, "nsubjpass", "PROPN")
    warned = fake_token(text, "warned", 2, "ROOT", "VERB", lemma="warn")
    by_token = fake_token(text, "by", 3, "agent", "ADP")
    mira_token = fake_token(text, "Mira", 4, "pobj", "PROPN")
    taren_token = fake_token(text, "Taren", 6, "nsubj", "PROPN")
    joined = fake_token(text, "joined", 7, "advcl", "VERB", lemma="join")
    harbor_token = fake_token(text, "Harbor", 8, "compound", "PROPN")
    guild_token = fake_token(text, "Guild", 9, "dobj", "PROPN")
    attach(rowan_token, warned)
    attach(by_token, warned)
    attach(mira_token, by_token)
    attach(taren_token, joined)
    attach(guild_token, joined)
    attach(harbor_token, guild_token)
    doc = FakeSpacyDoc(
        (
            rowan_token,
            warned,
            by_token,
            mira_token,
            taren_token,
            joined,
            harbor_token,
            guild_token,
        ),
        (),
    )
    backend = SpacyCandidateBackend(FakeSpacyLanguage(text=text, doc=doc))

    analysis = backend.analyze(record, ClassicalRuleConfig())

    assert analysis.dependency_backend == backend.backend_name
    assert "relation-indexed-warn" in {item.relation_id for item in analysis.relations}
    warned_relations = [item for item in analysis.relations if item.predicate == "warned"]
    assert len(warned_relations) == 1
    assert warned_relations[0].subject_mention_id == "m-mira"
    assert warned_relations[0].object_mention_id == "m-rowan"
    joined_relation = next(item for item in analysis.relations if item.predicate == "member_of")
    assert joined_relation.subject_mention_id == "m-taren"
    assert joined_relation.object_mention_id == "m-guild"
    assert joined_relation.relation_id.startswith("c0-spacy-relation-")
    assert "event-indexed-warn" in {item.event_candidate_id for item in analysis.events}
    warned_events = [item for item in analysis.events if item.trigger == "warned"]
    assert len(warned_events) == 1
    assert warned_events[0].participant_mention_ids == ("m-mira", "m-rowan")
    joined_event = next(item for item in analysis.events if item.trigger == "joined")
    assert joined_event.participant_mention_ids == ("m-taren", "m-guild")
    assert joined_event.event_candidate_id.startswith("c0-spacy-event-")
    assert backend.analyze(record, ClassicalRuleConfig()) == analysis


def test_spacy_ner_and_dependency_arguments_ground_a_missing_place_participant() -> None:
    text = "On day 5, Mira arrived at the harbor."
    record = EvidenceRecord(
        evidence_id="spacy-ner-evidence",
        passage_id="spacy-ner-passage",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        provenance=ProvenanceReference(
            provenance_id="spacy-ner-provenance",
            evidence_id="spacy-ner-evidence",
            extraction_method="fixture",
            locator="fixture:spacy-ner",
            confidence=1.0,
        ),
        confidence=1.0,
        release_class=ReleaseClass.PUBLIC,
    )

    on_token = fake_token(text, "On", 0, "prep", "ADP")
    day_token = fake_token(text, "day", 1, "pobj", "NOUN")
    mira_token = fake_token(text, "Mira", 3, "nsubj", "PROPN")
    arrived = fake_token(text, "arrived", 4, "ROOT", "VERB", lemma="arrive")
    at_token = fake_token(text, "at", 5, "prep", "ADP")
    the_token = fake_token(text, "the", 6, "det", "DET")
    harbor_token = fake_token(text, "harbor", 7, "pobj", "NOUN")
    attach(on_token, arrived)
    attach(day_token, on_token)
    attach(mira_token, arrived)
    attach(at_token, arrived)
    attach(harbor_token, at_token)
    attach(the_token, harbor_token)
    harbor_entity = FakeSpacyEntity(
        text="the harbor",
        start_char=text.index("the harbor"),
        label="LOC",
    )
    date_entity = FakeSpacyEntity(
        text="day 5",
        start_char=text.index("day 5"),
        label="DATE",
    )
    doc = FakeSpacyDoc(
        (on_token, day_token, mira_token, arrived, at_token, the_token, harbor_token),
        (date_entity, harbor_entity),
    )
    backend = SpacyCandidateBackend(FakeSpacyLanguage(text=text, doc=doc))

    analysis = backend.analyze(record, ClassicalRuleConfig())

    harbor = next(item for item in analysis.mentions if item.surface == "the harbor")
    assert harbor.entity_kind is ClassicalEntityKind.PLACE
    assert not any("day" in item.surface.casefold() for item in analysis.mentions)
    arrived_events = [item for item in analysis.events if item.trigger == "arrived"]
    assert len(arrived_events) == 1
    participants = arrived_events[0].participant_mention_ids
    assert harbor.mention_id in participants
    assert (
        next(item for item in analysis.mentions if item.surface == "Mira").mention_id
        in participants
    )
