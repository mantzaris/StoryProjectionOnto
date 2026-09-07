from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from story_projection_onto.conditions.base import (
    ConditionPreparation,
    ProduceInputs,
    RunConditionConfig,
    preontology_semantic_hash,
)
from story_projection_onto.conditions.c0 import (
    ClassicalEntityKind,
    ClassicalPreBuilder,
    ClassicalProjectionWeights,
    ClassicalRuleConfig,
    RuleCandidateBackend,
    SpacyCandidateBackend,
    load_classical_rule_config,
    project_sealed_c0,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    ConditionName,
    DiscoursePosition,
    Document,
    EventCandidate,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    InstanceGraph,
    MentionCandidate,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    Passage,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    ProvenanceReference,
    QueryAccessEvent,
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
    canonical_sha256,
    projection_validation_target_hash,
    to_model_visible_query,
)
from story_projection_onto.evidence import build_evidence_snapshot

BASE = datetime(2026, 9, 3, 10, 0, tzinfo=UTC)


@pytest.mark.parametrize("left,right", [("Gate 1", "Gate 2"), ("Turn 17", "Turn 18"), ("R2", "R3")])
def test_entity_identity_normalization_preserves_distinguishing_numbers(left, right):
    from story_projection_onto.conditions.c0 import _normalized_surface

    assert _normalized_surface(left, frozenset()) != _normalized_surface(right, frozenset())
    assert _normalized_surface("Dr. " + left, frozenset({"dr"})) == left.casefold()


def test_prequery_identity_keeps_two_numbered_places_separate():
    text = "Mira visited Gate 1. Taro visited Gate 2."
    record = EvidenceRecord(
        evidence_id="ev-numbers",
        passage_id="p-numbers",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        mention_candidates=tuple(
            mention("ev-numbers", f"m-{i}", text, name, "place")
            for i, name in enumerate(("Gate 1", "Gate 2"))
        ),
        provenance=ProvenanceReference(
            provenance_id="prov-numbers",
            evidence_id="ev-numbers",
            extraction_method="fixture",
            locator="fixture:numbers",
            confidence=1,
        ),
        confidence=1,
        release_class=ReleaseClass.PUBLIC,
    )
    builder = ClassicalPreBuilder()
    draft = builder._construct_draft(
        evidence_by_id={record.evidence_id: record},
        analyses=(builder.candidate_backend.analyze(record, builder.config),),
        upper_ontology=upper(),
        budgets=semantic_budgets(),
        constructed_at=BASE,
    )
    groups = [set(e.supported_mention_candidate_ids) for e in draft.instance_graph.entities]
    assert any("m-0" in group for group in groups)
    assert any("m-1" in group for group in groups)
    assert not any({"m-0", "m-1"} <= group for group in groups)


def test_occurrence_order_is_preserved_as_relation_not_third_event():
    text = "Launch occurred before Landing at story step 3."
    record = EvidenceRecord(
        evidence_id="ev-order",
        passage_id="p-order",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        mention_candidates=tuple(
            mention("ev-order", f"m-order-{i}", text, name, "event")
            for i, name in enumerate(("Launch", "Landing"))
        ),
        relation_phrase_candidates=(
            RelationPhraseCandidate(
                candidate_id="r-order",
                evidence_id="ev-order",
                subject_mention_candidate_id="m-order-0",
                object_mention_candidate_id="m-order-1",
                surface_phrase="occurred before",
                confidence=1,
            ),
        ),
        event_candidates=(
            EventCandidate(
                candidate_id="ec-order",
                evidence_id="ev-order",
                trigger_surface="occurred before",
                trigger_start_char=7,
                trigger_end_char=22,
                participant_mention_candidate_ids=("m-order-0", "m-order-1"),
                confidence=1,
            ),
        ),
        provenance=ProvenanceReference(
            provenance_id="prov-order",
            evidence_id="ev-order",
            extraction_method="fixture",
            locator="fixture:order",
            confidence=1,
        ),
        confidence=1,
        release_class=ReleaseClass.PUBLIC,
    )
    builder = ClassicalPreBuilder()
    draft = builder._construct_draft(
        evidence_by_id={record.evidence_id: record},
        analyses=(builder.candidate_backend.analyze(record, builder.config),),
        upper_ontology=upper(),
        budgets=semantic_budgets(),
        constructed_at=BASE,
    )
    assert not draft.instance_graph.events
    assert any(
        a.predicate_id == "c0-predicate-occurred_before" for a in draft.instance_graph.assertions
    )


@pytest.mark.parametrize(
    "verb,expected", [("reported", "reported"), ("denied", "denied"), ("believed", "believed")]
)
def test_embedded_claim_keeps_named_holder_without_promoting_to_world_truth(verb, expected):
    from story_projection_onto.contracts import NarrativeCommitment

    text = f"At story step 5, Mira {verb} that Taro plans to leave Harbor Guild."
    candidates = tuple(
        mention("ev-claim", f"m-{i}", text, name, kind)
        for i, (name, kind) in enumerate(
            (("Mira", "person"), ("Taro", "person"), ("Harbor Guild", "collective"))
        )
    )
    relation = RelationPhraseCandidate(
        candidate_id="r-content",
        evidence_id="ev-claim",
        subject_mention_candidate_id="m-1",
        object_mention_candidate_id="m-2",
        surface_phrase="plans to leave",
        confidence=0.95,
    )
    record = EvidenceRecord(
        evidence_id="ev-claim",
        passage_id="p-claim",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        mention_candidates=candidates,
        relation_phrase_candidates=(relation,),
        provenance=ProvenanceReference(
            provenance_id="prov-claim",
            evidence_id="ev-claim",
            extraction_method="fixture",
            locator="fixture:claim",
            confidence=1,
        ),
        confidence=1,
        release_class=ReleaseClass.PUBLIC,
    )
    builder = ClassicalPreBuilder()
    draft = builder._construct_draft(
        evidence_by_id={record.evidence_id: record},
        analyses=(builder.candidate_backend.analyze(record, builder.config),),
        upper_ontology=upper(),
        budgets=semantic_budgets(),
        constructed_at=BASE,
    )
    assertion = next(
        a
        for a in draft.instance_graph.assertions
        if a.predicate_id == "c0-predicate-plans_to_leave"
    )
    assert assertion.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED
    assert assertion.epistemic_scope.attitude.value == expected
    holder = next(
        e
        for e in draft.instance_graph.entities
        if e.entity_id == assertion.epistemic_scope.holder_id
    )
    assert "m-0" in holder.supported_mention_candidate_ids
    assert assertion.proposition_content_id


def test_predicate_normalization_is_general_and_validity_does_not_invent_end():
    from story_projection_onto.conditions.c0 import _normalize_predicate, _story_and_validity

    assert _normalize_predicate("served as", ClassicalRuleConfig()) == "holds_office"
    assert _normalize_predicate("revealed", ClassicalRuleConfig()) == "revealed"
    assert _normalize_predicate("led", ClassicalRuleConfig()) == "led"
    record = evidence_fixture()[0]
    raw = strip_content_hashes(record.model_dump())
    raw["text"] = "At story step 1, Mira belongs to Harbor Guild."
    raw["text_hash"] = digest(raw["text"])
    raw["mention_candidates"] = []
    raw["relation_phrase_candidates"] = []
    raw["event_candidates"] = []
    raw["temporal_clues"] = []
    evidence = EvidenceRecord.model_validate(raw)
    _, validity = _story_and_validity(evidence, (), state_like=True)
    assert validity.kind is TemporalKind.UNKNOWN
    assert validity.start is None and validity.end is None


def test_prequery_reification_requires_shared_candidate_anchor():
    snapshot, evidence, _ = snapshot_and_packet()
    evidence = tuple(
        EvidenceRecord.model_validate(
            strip_content_hashes(row.model_dump()) | {"event_candidates": []}
        )
        for row in evidence
    )
    builder = ClassicalPreBuilder()
    assert any(builder.candidate_backend.analyze(row, builder.config).events for row in evidence)
    prepared = builder.prepare(
        snapshot=snapshot,
        evidence=evidence,
        upper_ontology=upper(),
        preconstruction_budgets=semantic_budgets(),
        constructed_at=BASE + timedelta(minutes=2),
        sealed_at=BASE + timedelta(minutes=3),
    )
    assert not prepared.sealed_preontology.draft.instance_graph.events
    assert prepared.sealed_preontology.draft.instance_graph.assertions


def test_parser_scratch_alias_is_not_published_as_a_frozen_mention_reference():
    from story_projection_onto.conditions.c0 import RuleCandidateBackend

    class ScratchAliasBackend:
        def analyze(self, evidence, config):
            base = RuleCandidateBackend().analyze(evidence, config)
            original = base.mentions[0]
            extra = type(original).model_validate(
                original.model_dump(exclude={"content_hash"})
                | {"mention_id": f"scratch-{evidence.evidence_id}"}
            )
            return type(base).model_validate(
                base.model_dump(exclude={"content_hash"}) | {"mentions": (*base.mentions, extra)}
            )

    snapshot, evidence, _ = snapshot_and_packet()
    prepared = ClassicalPreBuilder(candidate_backend=ScratchAliasBackend()).prepare(
        snapshot=snapshot,
        evidence=evidence,
        upper_ontology=upper(),
        preconstruction_budgets=semantic_budgets(),
        constructed_at=BASE + timedelta(minutes=2),
        sealed_at=BASE + timedelta(minutes=3),
    )
    draft = prepared.sealed_preontology.draft
    frozen = {m.candidate_id for e in evidence for m in e.mention_candidates}
    assert draft.instance_graph.entities and draft.instance_graph.assertions
    assert all(
        set(entity.supported_mention_candidate_ids) <= frozen
        for entity in draft.instance_graph.entities
    )
    assert not any(
        identifier.startswith("scratch-")
        for decision in draft.decisions
        for identifier in decision.input_object_ids
    )


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_content_hashes(value):
    if isinstance(value, dict):
        return {
            key: strip_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, list):
        return [strip_content_hashes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_content_hashes(item) for item in value)
    return value


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


def test_frozen_c0_config_loads_exact_spacy_backend_and_projection_weights() -> None:
    config = load_classical_rule_config()
    assert config.backend == "spacy-ner-dependency-plus-deterministic-rules-v2"
    assert config.spacy_model_package == "en_core_web_sm"
    assert config.spacy_model_version == "3.8.0"
    assert config.query_blind is True
    assert config.fixed_query_time_weights == ClassicalProjectionWeights(
        lexical_context=1.0,
        time_compatibility=0.2,
        confidence_support=0.1,
        low_frequency_support=0.12,
        typed_path_continuity=0.15,
    )
    assert len(config.prohibited_query_time_operations) == 6


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


def test_c0_cpu_diagnostic_parses_development_story_step_and_validity_forms() -> None:
    # This is a deterministic parser diagnostic, not the registered C0
    # competence gate (which is produced only after the complete development
    # block has terminal ITT rows).
    text = (
        "At story step 1, Mira served as Harbor Warden for Harbor Guild. "
        "This relation held from story step 1 through story step 4."
    )
    mira = mention("story-step-evidence", "m-story-mira", text, "Mira", "person")
    guild = mention(
        "story-step-evidence",
        "m-story-guild",
        text,
        "Harbor Guild",
        "collective",
    )
    relation = RelationPhraseCandidate(
        candidate_id="r-story-office",
        evidence_id="story-step-evidence",
        subject_mention_candidate_id=mira.candidate_id,
        object_mention_candidate_id=guild.candidate_id,
        surface_phrase="served as",
        confidence=0.96,
    )
    record = EvidenceRecord(
        evidence_id="story-step-evidence",
        passage_id="story-step-passage",
        text=text,
        text_hash=digest(text),
        discourse_position=DiscoursePosition(passage_order=1),
        mention_candidates=(mira, guild),
        relation_phrase_candidates=(relation,),
        temporal_clues=(
            TemporalClue(
                clue_id="story-step-clue",
                evidence_id="story-step-evidence",
                normalized_expression="story-step-1",
                target_candidate_ids=(relation.candidate_id,),
                confidence=0.99,
            ),
        ),
        provenance=ProvenanceReference(
            provenance_id="story-step-provenance",
            evidence_id="story-step-evidence",
            extraction_method="fixture",
            locator="fixture:story-step",
            confidence=1.0,
        ),
        confidence=1.0,
        release_class=ReleaseClass.PUBLIC,
    )
    horizon = SpoilerHorizon(
        horizon_id="story-step-horizon",
        max_discourse_position=DiscoursePosition(passage_order=1),
        max_revelation_position=RevelationPosition(revelation_order=1),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id="story-step-snapshot",
        corpus_id="synthetic",
        world_or_window_id="story-step-world",
        horizon=horizon,
        eligible_evidence_ids=(record.evidence_id,),
        index_config_hash=digest("story-step-index"),
        created_at=BASE,
        sealed_at=BASE + timedelta(minutes=1),
        release_class=ReleaseClass.PUBLIC,
    )

    preparation = ClassicalPreBuilder().prepare(
        snapshot=snapshot,
        evidence=(record,),
        upper_ontology=upper(),
        preconstruction_budgets=semantic_budgets(),
        constructed_at=BASE + timedelta(minutes=2),
        sealed_at=BASE + timedelta(minutes=3),
    )

    assert preparation.sealed_preontology is not None
    assertion = next(
        item
        for item in preparation.sealed_preontology.draft.instance_graph.assertions
        if item.predicate_id == "c0-predicate-holds_office"
    )
    assert assertion.temporal_scope.story_time == StoryTime(
        kind=TemporalKind.POINT,
        point=1,
        label="story step 1",
    )
    assert assertion.temporal_scope.validity_time.kind is TemporalKind.INTERVAL
    assert assertion.temporal_scope.validity_time.start == 1
    assert assertion.temporal_scope.validity_time.end == 4


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
        scored_schema_hash=canonical_sha256(
            OntologyProjection.model_json_schema(mode="validation")
        ),
        validator_hash=digest("validator-v1"),
        upper_ontology_hash=upper_ontology.content_hash,
    )
    assert preparation.sealed_preontology is not None
    barrier = PrequeryBarrier(
        barrier_id="barrier-c0-unit",
        execution_id="c0-unit-execution",
        execution_manifest_hash=digest("c0-execution-manifest"),
        neutral_evidence_artifact_hashes=(digest("c0-neutral-evidence"),),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id=snapshot.world_or_window_id,
                condition=ConditionName.C0_CLASSICAL_PRE,
                snapshot_hash=snapshot.content_hash,
                preparation_hash=preparation.content_hash,
                lineage_artifact_hash=(
                    preparation.sealed_preontology.construction_seal.content_hash
                ),
                completed_at=preparation.completed_at,
            ),
        ),
        sealed_at=context.revealed_at - timedelta(microseconds=1),
    )
    query_access = QueryAccessEvent(
        access_event_id="access-context-c0",
        execution_id=barrier.execution_id,
        query_context_hash=context.content_hash,
        model_visible_query_hash=to_model_visible_query(context).content_hash,
        snapshot_hash=snapshot.content_hash,
        stage_manifest_hash=digest("c0-query-stage"),
        query_artifact_hash=digest("c0-query-artifact"),
        prequery_barrier_hash=barrier.content_hash,
        packet_hash=packet.content_hash,
        registered_revealed_at=context.revealed_at,
        accessed_at=context.revealed_at,
    )
    inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        query_access=query_access,
        prequery_barrier=barrier,
        query_processing_started_at=context.revealed_at + timedelta(microseconds=1),
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


def test_c0_projection_recursively_retains_sealed_epistemic_holder_and_binds_target() -> None:
    snapshot, evidence, packet = snapshot_and_packet()
    upper_ontology = upper()
    original = ClassicalPreBuilder().prepare(
        snapshot=snapshot,
        evidence=evidence,
        upper_ontology=upper_ontology,
        preconstruction_budgets=semantic_budgets(),
        constructed_at=BASE + timedelta(minutes=2),
        sealed_at=BASE + timedelta(minutes=3),
    )
    assert original.sealed_preontology is not None
    source = original.sealed_preontology
    graph_payload = strip_content_hashes(source.draft.instance_graph.model_dump(mode="python"))
    assertions = graph_payload["assertions"]
    report = next(item for item in assertions if item["predicate_id"] == "c0-predicate-reported")
    endpoints = {report["subject_id"], report["object_id"]}
    holder = next(item for item in graph_payload["entities"] if item["entity_id"] not in endpoints)
    assert report["epistemic_scope"] is not None
    report["epistemic_scope"]["holder_id"] = holder["entity_id"]
    required_nodes = endpoints | {holder["entity_id"]}
    for entity_payload in graph_payload["entities"]:
        if entity_payload["entity_id"] in required_nodes:
            entity_payload["description_assertion_ids"] = [report["assertion_id"]]
    modified_graph = InstanceGraph.model_validate(graph_payload)
    draft_payload = strip_content_hashes(source.draft.model_dump(mode="python"))
    draft_payload["instance_graph"] = modified_graph
    modified_draft = OntologyDraft.model_validate(draft_payload)
    seal_payload = strip_content_hashes(source.construction_seal.model_dump(mode="python"))
    seal_payload["ontology_hash"] = preontology_semantic_hash(
        upper_ontology,
        modified_draft,
    )
    modified_seal = source.construction_seal.__class__.model_validate(seal_payload)
    source_payload = strip_content_hashes(source.model_dump(mode="python"))
    source_payload.update(draft=modified_draft, construction_seal=modified_seal)
    modified_source = source.__class__.model_validate(source_payload)
    preparation = ConditionPreparation(
        preparation_id="c0-holder-preparation",
        condition=ConditionName.C0_CLASSICAL_PRE,
        snapshot_hash=snapshot.content_hash,
        completed_at=modified_seal.sealed_at,
        sealed_preontology=modified_source,
    )

    final_budgets = OutputBudgets(
        node_budget=3,
        assertion_budget=1,
        display_node_budget=3,
        display_assertion_budget=1,
    )
    context = QueryContext(
        context_id="context-c0-holder",
        wording="Who reported Rowan?",
        lens="reported claims",
        target="reported Rowan",
        story_scope=StoryTime(kind=TemporalKind.POINT, point=2, label="day two"),
        spoiler_horizon=snapshot.horizon,
        abstraction=AbstractionLevel.EVENT_ROLE,
        budgets=final_budgets,
        revealed_at=BASE + timedelta(hours=2),
    )
    config = RunConditionConfig(
        config_id="c0-holder-config",
        condition=ConditionName.C0_CLASSICAL_PRE,
        budgets=final_budgets,
        maximum_input_tokens=10_240,
        maximum_output_tokens=2_048,
        scored_schema_hash=canonical_sha256(
            OntologyProjection.model_json_schema(mode="validation")
        ),
        validator_hash=digest("validator-v1"),
        upper_ontology_hash=upper_ontology.content_hash,
    )
    barrier = PrequeryBarrier(
        barrier_id="barrier-c0-holder",
        execution_id="c0-holder-execution",
        execution_manifest_hash=digest("c0-holder-manifest"),
        neutral_evidence_artifact_hashes=(digest("c0-holder-neutral"),),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id=snapshot.world_or_window_id,
                condition=ConditionName.C0_CLASSICAL_PRE,
                snapshot_hash=snapshot.content_hash,
                preparation_hash=preparation.content_hash,
                lineage_artifact_hash=modified_seal.content_hash,
                completed_at=preparation.completed_at,
            ),
        ),
        sealed_at=context.revealed_at - timedelta(microseconds=1),
    )
    access = QueryAccessEvent(
        access_event_id="access-c0-holder",
        execution_id=barrier.execution_id,
        query_context_hash=context.content_hash,
        model_visible_query_hash=to_model_visible_query(context).content_hash,
        snapshot_hash=snapshot.content_hash,
        stage_manifest_hash=digest("c0-holder-stage"),
        query_artifact_hash=digest("c0-holder-query"),
        prequery_barrier_hash=barrier.content_hash,
        packet_hash=packet.content_hash,
        registered_revealed_at=context.revealed_at,
        accessed_at=context.revealed_at,
    )
    projection = project_sealed_c0(
        modified_source,
        ProduceInputs(
            preparation=preparation,
            snapshot=snapshot,
            packet=packet,
            context=context,
            query_access=access,
            prequery_barrier=barrier,
            query_processing_started_at=context.revealed_at + timedelta(microseconds=1),
            upper_ontology=upper_ontology,
            run_config=config,
        ),
    )

    assert {item.assertion_id for item in projection.instance_graph.assertions} == {
        report["assertion_id"]
    }
    assert holder["entity_id"] in {item.entity_id for item in projection.instance_graph.entities}
    expected_target = projection_validation_target_hash(
        condition=projection.condition,
        snapshot_hash=projection.snapshot_hash,
        packet_hash=projection.packet_hash,
        context_hash=projection.context_hash,
        upper_ontology=projection.upper_ontology,
        local_schema=projection.local_schema,
        instance_graph=projection.instance_graph,
        decisions=projection.decisions,
        omissions=projection.omissions,
        budget_accounting=projection.budget_accounting,
        budgets=projection.budgets,
    )
    assert {item.target_id for item in projection.validation_records} == {expected_target}
    assert any(
        diagnostic.startswith("structural_report_sha256:")
        for diagnostic in projection.validation_records[0].diagnostics
    )

    tampered_payload = strip_content_hashes(projection.model_dump(mode="python"))
    tampered_payload["decisions"][0]["rationale"] = (
        "A changed selection rationale must invalidate the structural target binding."
    )
    with pytest.raises(
        ValueError,
        match="projection structural validation targets different final semantics",
    ):
        OntologyProjection.model_validate(tampered_payload)


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
