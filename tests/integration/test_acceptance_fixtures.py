from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from story_projection_onto.conditions.base import (
    preontology_semantic_hash,
    sealed_semantic_ids,
)
from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    FIXED_SELECT_ALLOWED_OPERATORS,
    ConditionName,
    ConstructionCertificate,
    ConstructionOperator,
    ConstructionRequest,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    NarrativeCommitment,
    OntologyDraft,
    PreconstructionRequest,
    PreQueryInventory,
    ProvenanceReference,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    StoryTime,
    TemporalKind,
    ValidityTime,
    canonical_sha256,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    FixedSelectCapabilityError,
    PackingReport,
    RepairLineageMetadata,
    allowed_capabilities_for,
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
    semantic_fingerprints_from_draft,
)
from story_projection_onto.validate import (
    BoundaryValidationReport,
    GroundingSupportStatus,
    ValidationCode,
    validate_draft_evidence_grounding,
    validate_repair_preservation,
    validate_single_repair_lineage,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "phase1"
QUERY_REVEAL = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

PROMPTS = {
    ConditionName.C1_LLM_PRE: ROOT / "prompts" / "c1_pre" / "prompt_v1.md",
    ConditionName.C2_LLM_QUERY: ROOT / "prompts" / "c2_query" / "prompt_v1.md",
    ConditionName.A_FIXED_SELECT: ROOT / "prompts" / "fixed_select" / "prompt_v1.md",
}
REPAIR_PROMPT = ROOT / "prompts" / "repair" / "prompt_v1.md"

EXPECTED_PROMPT_HASHES = {
    ConditionName.C1_LLM_PRE: "0d6e54e8c3d1b80d31248b9f099a0a433e0a2b0cb9b2f4f70a2a6d547810b03e",
    ConditionName.C2_LLM_QUERY: "f85a326675fe792d30789e3d4341e2e23bffb4649f350c07fa271914770b0e10",
    ConditionName.A_FIXED_SELECT: (
        "1cafbb5130f1aa2694af764cdd6c7d034287926d519d7b8d157339307765fd5b"
    ),
}
EXPECTED_REPAIR_PROMPT_HASH = "6b7430d00daf1cd1b71a10d97d09c18d989444e9ef030ef33a3b16b2a1a4084c"


def load_json(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def strip_content_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, list):
        return [strip_content_hashes(child) for child in value]
    return value


def semantic_ids(draft: OntologyDraft) -> set[str]:
    schema = draft.local_schema
    graph = draft.instance_graph
    return {
        schema.schema_id,
        *(item.type_id for item in schema.contextual_types),
        *(item.predicate_id for item in schema.predicates),
        *(item.entity_id for item in graph.entities),
        *(item.event_id for item in graph.events),
        *(item.proposition_content_id for item in graph.proposition_contents),
        *(item.assertion_id for item in graph.assertions),
    }


def cited_records(draft: OntologyDraft) -> Iterable[tuple[str, tuple[str, ...]]]:
    for item in draft.local_schema.contextual_types:
        yield item.type_id, item.evidence_ids
    for item in draft.local_schema.predicates:
        yield item.predicate_id, item.evidence_ids
    for item in draft.instance_graph.entities:
        yield item.entity_id, item.evidence_ids
    for item in draft.instance_graph.events:
        yield item.event_id, item.evidence_ids
    for item in draft.instance_graph.proposition_contents:
        yield item.proposition_content_id, item.evidence_ids
    for item in draft.instance_graph.assertions:
        yield item.assertion_id, item.evidence_ids
    for item in draft.decisions:
        yield item.decision_id, item.evidence_ids


def assert_draft_referential_integrity(draft: OntologyDraft) -> None:
    graph = draft.instance_graph
    type_ids = {item.type_id for item in draft.local_schema.contextual_types}
    predicate_ids = {item.predicate_id for item in draft.local_schema.predicates}
    object_ids = {
        *(item.entity_id for item in graph.entities),
        *(item.event_id for item in graph.events),
        *(item.proposition_content_id for item in graph.proposition_contents),
    }
    assertion_ids = {item.assertion_id for item in graph.assertions}

    assert all(item.contextual_type_id in type_ids for item in graph.entities)
    assert all(item.contextual_type_id in type_ids for item in graph.events)
    assert all(
        set(item.domain_type_ids).union(item.range_type_ids).issubset(type_ids)
        for item in draft.local_schema.predicates
    )
    assert all(
        set(item.description_assertion_ids).issubset(assertion_ids)
        for item in (*graph.entities, *graph.events)
    )
    for proposition in graph.proposition_contents:
        assert proposition.predicate_id in predicate_ids
        assert {proposition.subject_id, proposition.object_id}.issubset(object_ids)
    for assertion in graph.assertions:
        assert assertion.predicate_id in predicate_ids
        endpoint_ids = {
            value for value in (assertion.subject_id, assertion.object_id) if value is not None
        }
        endpoint_ids.update(role.object_id for role in assertion.roles)
        assert endpoint_ids.issubset(object_ids)
        if assertion.proposition_content_id is not None:
            assert assertion.proposition_content_id in object_ids


def full_validation_inputs(
    request: ConstructionRequest,
) -> tuple[EvidenceSnapshot, EvidencePacket, QueryContext]:
    full_records = []
    for visible in request.packet.evidence:
        provenance = ProvenanceReference(
            provenance_id=f"validator-{visible.evidence_id}",
            evidence_id=visible.evidence_id,
            extraction_method="hand-authored acceptance fixture",
            locator=f"phase1:{visible.evidence_id}",
            source_artifact_hash=digest(f"phase1-source:{visible.evidence_id}"),
            confidence=1.0,
        )
        full_records.append(
            EvidenceRecord(
                evidence_id=visible.evidence_id,
                passage_id=f"passage-{visible.evidence_id}",
                text=visible.text,
                text_hash=hashlib.sha256(visible.text.encode("utf-8")).hexdigest(),
                discourse_position=visible.discourse_position,
                mention_candidates=visible.mention_candidates,
                event_candidates=visible.event_candidates,
                relation_phrase_candidates=visible.relation_phrase_candidates,
                temporal_clues=visible.temporal_clues,
                provenance=provenance,
                confidence=1.0,
                release_class=ReleaseClass.PUBLIC,
            )
        )

    snapshot = EvidenceSnapshot(
        snapshot_id="phase1-validation-snapshot",
        corpus_id="phase1-public-synthetic",
        world_or_window_id="phase1-harbor",
        horizon=request.context.spoiler_horizon,
        eligible_evidence_ids=tuple(item.evidence_id for item in full_records),
        index_config_hash="c" * 64,
        created_at=datetime(2026, 9, 3, 11, 30, tzinfo=UTC),
        sealed_at=datetime(2026, 9, 3, 11, 35, tzinfo=UTC),
        release_class=ReleaseClass.PUBLIC,
    )
    packet = EvidencePacket(
        packet_id="phase1-validation-packet",
        snapshot_hash=snapshot.content_hash,
        evidence=tuple(full_records),
        ordered_evidence_ids=tuple(item.evidence_id for item in full_records),
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        token_count=900,
        horizon_rejections=("ev-08",),
        created_at=QUERY_REVEAL,
        release_class=ReleaseClass.PUBLIC,
    )
    visible_context = request.context
    context = QueryContext(
        context_id="phase1-context",
        wording=visible_context.wording,
        lens=visible_context.lens,
        target=visible_context.target,
        story_scope=visible_context.story_scope,
        spoiler_horizon=visible_context.spoiler_horizon,
        viewpoint=visible_context.viewpoint,
        abstraction=visible_context.abstraction,
        budgets=visible_context.budgets,
        revealed_at=QUERY_REVEAL,
    )
    return snapshot, packet, context


def test_versioned_prompts_are_non_thinking_and_hash_bound_to_requests() -> None:
    c1_request = PreconstructionRequest.model_validate(load_json("c1_pre_request.json"))
    c2_request = ConstructionRequest.model_validate(load_json("c2_query_request.json"))
    fixed_request = ConstructionRequest.model_validate(load_json("fixed_select_request.json"))
    requests = {
        ConditionName.C1_LLM_PRE: c1_request,
        ConditionName.C2_LLM_QUERY: c2_request,
        ConditionName.A_FIXED_SELECT: fixed_request,
    }

    for condition, path in PROMPTS.items():
        prompt = path.read_text(encoding="utf-8")
        assert "v1.0.0" in prompt.splitlines()[0]
        assert "Thinking mode is disabled" in prompt
        assert "Return exactly one JSON object" in prompt
        assert sha256_file(path) == EXPECTED_PROMPT_HASHES[condition]
        assert requests[condition].runtime.prompt_hash == EXPECTED_PROMPT_HASHES[condition]

    c1_prompt = PROMPTS[ConditionName.C1_LLM_PRE].read_text(encoding="utf-8")
    assert "query has not been revealed" in c1_prompt
    assert "Never infer a query" in c1_prompt
    c2_prompt = PROMPTS[ConditionName.C2_LLM_QUERY].read_text(encoding="utf-8")
    for required_phrase in (
        "active construction",
        "merge",
        "remain split",
        "local contextual schema",
        "reify events",
        "holder-level epistemic status",
        "low-frequency",
        "why_matters",
    ):
        assert required_phrase in c2_prompt
    fixed_prompt = PROMPTS[ConditionName.A_FIXED_SELECT].read_text(encoding="utf-8")
    assert "complete same-seed" in fixed_prompt
    assert "forbidden and mechanically audited" in fixed_prompt
    assert "creating any entity" in fixed_prompt
    repair_prompt = REPAIR_PROMPT.read_text(encoding="utf-8")
    assert sha256_file(REPAIR_PROMPT) == EXPECTED_REPAIR_PROMPT_HASH
    assert "fact-free list of validator diagnostics" in repair_prompt
    assert "Diagnostics identify defects; they never prescribe replacement facts" in repair_prompt
    assert "repair attempt 1 of 1" in repair_prompt


def test_model_visible_requests_are_gold_free_query_timed_and_evidence_equal() -> None:
    c1_raw = load_json("c1_pre_request.json")
    c2_raw = load_json("c2_query_request.json")
    fixed_raw = load_json("fixed_select_request.json")
    forbidden_keys = {
        "benchmark_split",
        "community_id",
        "condition_result",
        "expected_effect",
        "gold",
        "gold_id",
        "is_pivotal",
        "is_rare",
        "pair_id",
        "world_id",
    }
    for raw in (c1_raw, c2_raw, fixed_raw):
        assert forbidden_keys.isdisjoint(walk_keys(raw))

    c1 = PreconstructionRequest.model_validate(c1_raw)
    c2 = ConstructionRequest.model_validate(c2_raw)
    fixed = ConstructionRequest.model_validate(fixed_raw)

    assert "context" not in c1_raw
    assert "packet" not in c1_raw
    assert c1.requested_at < QUERY_REVEAL < c2.requested_at
    assert c1.evidence == c2.packet.evidence == fixed.packet.evidence
    assert c2.packet == fixed.packet
    assert c2.context == fixed.context
    assert c1.snapshot_hash == c2.snapshot_hash == fixed.snapshot_hash
    assert c1.upper_ontology == c2.upper_ontology == fixed.upper_ontology
    assert c1.budgets == c2.budgets == fixed.budgets
    assert (
        max(item.discourse_position.ordering_key for item in c2.packet.evidence)
        <= c2.context.spoiler_horizon.max_discourse_position.ordering_key
    )
    assert c2.fixed_ontology is None
    assert fixed.fixed_ontology is not None
    assert fixed.fixed_ontology.construction_seal.sealed_at < QUERY_REVEAL


def test_capability_manifests_make_fixed_select_construction_impossible() -> None:
    c1 = CapabilityManifest.for_condition(ConditionName.C1_LLM_PRE)
    c2 = CapabilityManifest.for_condition(ConditionName.C2_LLM_QUERY)
    fixed = CapabilityManifest.for_condition(ConditionName.A_FIXED_SELECT)
    fixed_request = ConstructionRequest.model_validate(load_json("fixed_select_request.json"))

    assert set(c1.allowed) == allowed_capabilities_for(ConditionName.C1_LLM_PRE)
    assert set(c2.allowed) == set(ConstructionOperator)
    assert set(fixed.allowed) == FIXED_SELECT_ALLOWED_OPERATORS
    assert set(fixed.forbidden) == CONSTRUCTIVE_OPERATORS
    assert set(fixed.allowed).isdisjoint(fixed.forbidden)
    assert not fixed_request.capabilities.create_entities
    assert not fixed_request.capabilities.merge_split
    assert not fixed_request.capabilities.create_schema_predicates
    assert not fixed_request.capabilities.reify_events
    assert not fixed_request.capabilities.change_abstraction
    assert not fixed_request.capabilities.add_temporal_qualification
    assert not fixed_request.capabilities.add_epistemic_qualification


def test_c1_and_c2_drafts_exercise_registered_construction_operators() -> None:
    c1 = OntologyDraft.model_validate(load_json("c1_pre_output.json"))
    c1_secondary = OntologyDraft.model_validate(load_json("c1_pre_output_2.json"))
    c2 = OntologyDraft.model_validate(load_json("c2_query_output.json"))
    c2_secondary = OntologyDraft.model_validate(load_json("c2_query_output_2.json"))
    c1_operators = {item.operator for item in c1.decisions}
    c1_secondary_operators = {item.operator for item in c1_secondary.decisions}
    c2_operators = {item.operator for item in c2.decisions}
    c2_secondary_operators = {item.operator for item in c2_secondary.decisions}

    # The pilot call set, rather than every individual 2,048-token response, is the
    # registered unit of operator coverage.  Both references remain independently
    # coherent while their union exercises every constructive operation.
    assert c1_operators.issubset(allowed_capabilities_for(ConditionName.C1_LLM_PRE))
    assert c1_secondary_operators.issubset(allowed_capabilities_for(ConditionName.C1_LLM_PRE))
    assert c2_operators.issubset(allowed_capabilities_for(ConditionName.C2_LLM_QUERY))
    assert c2_secondary_operators.issubset(allowed_capabilities_for(ConditionName.C2_LLM_QUERY))
    assert c1_operators | c1_secondary_operators == CONSTRUCTIVE_OPERATORS
    assert c2_operators | c2_secondary_operators == CONSTRUCTIVE_OPERATORS
    assert c1_operators | c2_operators == CONSTRUCTIVE_OPERATORS
    assert c1_operators.isdisjoint(c2_operators)
    assert all(
        item.decided_at < QUERY_REVEAL for draft in (c1, c1_secondary) for item in draft.decisions
    )
    assert all(
        item.decided_at > QUERY_REVEAL for draft in (c2, c2_secondary) for item in draft.decisions
    )

    for draft in (c1, c1_secondary, c2, c2_secondary):
        assert_draft_referential_integrity(draft)
        draft_ids = semantic_ids(draft)
        for item in draft.decisions:
            assert item.operator in CONSTRUCTIVE_OPERATORS
            assert item.created_object_ids
            assert set(item.created_object_ids).issubset(draft_ids)

    c1_ids = semantic_ids(c1)
    event_decision = next(
        item for item in c1.decisions if item.operator is ConstructionOperator.EVENT_REIFICATION
    )
    assert set(event_decision.created_object_ids) == {
        "event-c1-delivery",
        "event-c1-opening",
    }
    assert set(event_decision.created_object_ids).issubset(c1_ids)
    c2_event_decision = next(
        item
        for item in c2_secondary.decisions
        if item.operator is ConstructionOperator.EVENT_REIFICATION
    )
    assert set(c2_event_decision.created_object_ids) == {
        "event-c2b-delivery",
        "event-c2b-opening",
    }
    assert ConstructionOperator.MERGE in c1_operators
    assert ConstructionOperator.SPLIT in c2_operators
    assert c2.local_schema.abstraction.value == "collective_causal_chain"
    assert all(
        item.abstraction.value == "collective_causal_chain" for item in c2.instance_graph.entities
    )


def test_reference_identity_event_rare_temporal_and_epistemic_content_is_substantive() -> None:
    request = ConstructionRequest.model_validate(load_json("c2_query_request.json"))
    c1 = OntologyDraft.model_validate(load_json("c1_pre_output.json"))
    draft = OntologyDraft.model_validate(load_json("c2_query_output.json"))
    c2_secondary = OntologyDraft.model_validate(load_json("c2_query_output_2.json"))
    evidence_ids = set(request.packet.ordered_evidence_ids)
    assertions = {item.assertion_id: item for item in draft.instance_graph.assertions}
    c1_assertions = {item.assertion_id: item for item in c1.instance_graph.assertions}
    entities = {item.entity_id: item for item in draft.instance_graph.entities}

    assert "m-ash-courier" in entities["ent-c2-lio"].supported_mention_candidate_ids
    assert entities["ent-c2-ash-mechanic"].supported_mention_candidate_ids == ("m-ash-mechanic",)
    assert entities["ent-c2-lio"].entity_id != entities["ent-c2-ash-mechanic"].entity_id
    assert not draft.instance_graph.events
    assert {item.event_id for item in c2_secondary.instance_graph.events} == {
        "event-c2b-delivery",
        "event-c2b-opening",
    }

    rare_decision = next(
        item for item in c1.decisions if item.operator is ConstructionOperator.RARE_PRESERVATION
    )
    assert rare_decision.evidence_ids == ("ev-06",)
    assert "ev-06" in c1_assertions["assert-c1-enabled"].evidence_ids
    assert c1_assertions["assert-c1-enabled"].why_matters_evidence_ids == (
        "ev-04",
        "ev-06",
    )

    report = assertions["assert-c2-report"]
    proposition = next(
        item
        for item in draft.instance_graph.proposition_contents
        if item.proposition_content_id == "prop-c2-bridge-safe"
    )
    assert report.narrative_commitment is NarrativeCommitment.HOLDER_ATTRIBUTED
    assert report.epistemic_scope is not None
    assert report.epistemic_scope.holder_id == "ent-c2-mara"
    assert report.epistemic_scope.attitude.value == "reported"
    assert (
        proposition.predicate_id,
        proposition.subject_id,
        proposition.object_id,
    ) == (report.predicate_id, report.subject_id, report.object_id)
    assert (proposition.subject_id, proposition.object_id) == (
        "ent-c2-bridge",
        "ent-c2-safe",
    )
    assert report.epistemic_scope.holder_id not in {
        proposition.subject_id,
        proposition.object_id,
    }
    safe = entities["ent-c2-safe"]
    assert safe.supported_mention_candidate_ids == ("m-safe-05",)
    assert report.assertion_id in safe.description_assertion_ids
    assert not any(
        item.proposition_content_id == "prop-c2-bridge-safe"
        and item.narrative_commitment is NarrativeCommitment.WORLD_COMMITTED
        for item in draft.instance_graph.assertions
    )
    assert not c1.instance_graph.proposition_contents
    assert isinstance(report.temporal_scope.story_time, StoryTime)
    assert isinstance(report.temporal_scope.validity_time, ValidityTime)
    assert report.temporal_scope.story_time.kind is TemporalKind.UNKNOWN
    assert report.temporal_scope.validity_time.kind is TemporalKind.UNKNOWN
    assert report.epistemic_scope.holder_relative_time.kind is TemporalKind.UNKNOWN
    assert report.temporal_scope.discourse_position.passage_order == 5
    assert report.temporal_scope.revelation_position.revelation_order == 5
    assert request.context.spoiler_horizon.max_discourse_position.passage_order == 7

    for constructed, prefix in ((c1, "c1"), (c2_secondary, "c2b")):
        constructed_types = {
            item.type_id: item for item in constructed.local_schema.contextual_types
        }
        assert constructed_types[f"t-{prefix}"].parent_upper_type == "entity"
        assert constructed_types[f"t-{prefix}-event"].parent_upper_type == "event"
        assert {item.contextual_type_id for item in constructed.instance_graph.entities} == {
            f"t-{prefix}"
        }
        assert {item.contextual_type_id for item in constructed.instance_graph.events} == {
            f"t-{prefix}-event"
        }
        constructed_assertions = {
            item.assertion_id: item for item in constructed.instance_graph.assertions
        }
        chain = constructed_assertions[f"assert-{prefix}-chain"]
        enabled = constructed_assertions[f"assert-{prefix}-enabled"]
        assert chain.temporal_scope.story_time.kind is TemporalKind.POINT
        assert chain.temporal_scope.story_time.point == 2
        assert chain.temporal_scope.validity_time.point == 2
        assert chain.temporal_scope.discourse_position.passage_order == 4
        assert chain.temporal_scope.revelation_position.revelation_order == 4
        assert enabled.temporal_scope.story_time.point == 2
        assert enabled.temporal_scope.validity_time.point == 2
        assert enabled.temporal_scope.discourse_position.passage_order == 6
        assert enabled.temporal_scope.revelation_position.revelation_order == 6
        assert {item.occurrence_time.point for item in constructed.instance_graph.events} == {2}

    c1_secondary = OntologyDraft.model_validate(load_json("c1_pre_output_2.json"))
    for identity_draft, prefix in ((draft, "c2"), (c1_secondary, "c1b")):
        identity_type_parents = {
            item.parent_upper_type for item in identity_draft.local_schema.contextual_types
        }
        assert identity_type_parents == {"entity"}
        identity_assertions = {
            item.assertion_id: item for item in identity_draft.instance_graph.assertions
        }
        distinct = identity_assertions[f"assert-{prefix}-distinct"]
        attributed = identity_assertions[f"assert-{prefix}-report"]
        assert distinct.temporal_scope.story_time.kind is TemporalKind.UNKNOWN
        assert distinct.temporal_scope.validity_time.kind is TemporalKind.UNKNOWN
        assert distinct.temporal_scope.discourse_position.passage_order == 3
        assert attributed.temporal_scope.story_time.kind is TemporalKind.UNKNOWN
        assert attributed.temporal_scope.validity_time.kind is TemporalKind.UNKNOWN
        assert attributed.temporal_scope.discourse_position.passage_order == 5

    c1_predicates = {item.predicate_id: item for item in c1.local_schema.predicates}
    c2_predicates = {item.predicate_id: item for item in draft.local_schema.predicates}
    assert c1_predicates["p-c1-chain"].parent_upper_relation == "precedes"
    assert c1_predicates["p-c1-credential-enabled"].parent_upper_relation == "causes"
    assert c2_predicates["p-c2-status"].parent_upper_relation == "has_status"
    assert all(
        role.role != "separate_mechanic"
        for assertion_item in (
            *c1.instance_graph.assertions,
            *c2_secondary.instance_graph.assertions,
        )
        for role in assertion_item.roles
    )

    for _, citations in cited_records(draft):
        assert set(citations).issubset(evidence_ids)
    for assertion in draft.instance_graph.assertions:
        assert assertion.why_matters.strip()
        assert set(assertion.why_matters_evidence_ids).issubset(assertion.evidence_ids)
        assert all(item.evidence_id in assertion.evidence_ids for item in assertion.provenance)


def test_c2_empty_inventory_and_certificate_bind_all_post_reveal_decisions() -> None:
    request = ConstructionRequest.model_validate(load_json("c2_query_request.json"))
    draft = OntologyDraft.model_validate(load_json("c2_query_output.json"))
    inventory = PreQueryInventory(
        inventory_id="phase1-c2-empty-inventory",
        condition=ConditionName.C2_LLM_QUERY,
        snapshot_hash=request.snapshot_hash,
        recorded_at=datetime(2026, 9, 3, 11, 59, tzinfo=UTC),
    )
    certificate = ConstructionCertificate(
        certificate_id="phase1-c2-certificate",
        condition=ConditionName.C2_LLM_QUERY,
        snapshot_hash=request.snapshot_hash,
        packet_hash=request.packet.packet_hash,
        query_context_hash=request.context.content_hash,
        query_access_event_hash=digest("phase1-query-access"),
        stage_manifest_hash=digest("phase1-query-stage"),
        prequery_barrier_hash=digest("phase1-prequery-barrier"),
        generation_lineage_hash=digest("phase1-generation-lineage"),
        raw_output_artifact_hash=digest("phase1-raw-output"),
        normalized_draft_hash=draft.content_hash,
        validation_bundle_hash=digest("phase1-validation-bundle"),
        query_revealed_at=QUERY_REVEAL,
        completed_at=datetime(2026, 9, 3, 12, 1, tzinfo=UTC),
        decisions=draft.decisions,
        pre_query_inventory_hash=inventory.content_hash,
    )

    assert not inventory.ontology_refs
    assert not inventory.entity_ids
    assert not inventory.event_ids
    assert any(item.operator in CONSTRUCTIVE_OPERATORS for item in certificate.decisions)
    assert all(item.decided_at > certificate.query_revealed_at for item in certificate.decisions)


def test_fixed_select_draft_is_a_semantically_unchanged_sealed_selection() -> None:
    request = ConstructionRequest.model_validate(load_json("fixed_select_request.json"))
    source_draft = OntologyDraft.model_validate(load_json("c1_pre_output.json"))
    selected_draft = OntologyDraft.model_validate(load_json("fixed_select_output.json"))
    assert request.fixed_ontology is not None
    sealed = sealed_inventory_from_fixed_ontology(
        request.fixed_ontology,
        seed_block=1,
        source_draft=source_draft,
    )

    assert request.fixed_ontology.local_schema == source_draft.local_schema
    assert request.fixed_ontology.instance_graph == source_draft.instance_graph
    assert request.fixed_ontology.construction_seal.ontology_hash == preontology_semantic_hash(
        request.upper_ontology, source_draft
    )
    assert request.fixed_ontology.construction_seal.sealed_object_ids == sealed_semantic_ids(
        source_draft
    )
    assert sealed.objects == semantic_fingerprints_from_draft(source_draft)
    enforce_fixed_select_draft(selected_draft, sealed=sealed, seed_block=1)
    assert {item.operator for item in selected_draft.decisions} == FIXED_SELECT_ALLOWED_OPERATORS
    assert all(
        not item.created_object_ids and not item.removed_object_ids
        for item in selected_draft.decisions
    )
    assert len(sealed.objects) == len(request.fixed_ontology.construction_seal.sealed_object_ids)

    mutated_payload = strip_content_hashes(selected_draft.model_dump(mode="json"))
    mutated_payload["instance_graph"]["entities"][0]["contextual_role"] = (
        "a newly invented semantic role"
    )
    mutated = OntologyDraft.model_validate(mutated_payload)
    with pytest.raises(FixedSelectCapabilityError, match="semantic content changed"):
        enforce_fixed_select_draft(mutated, sealed=sealed, seed_block=1)


def test_complete_packing_reports_cover_each_path_without_truncation() -> None:
    reports = tuple(
        PackingReport.model_validate(item) for item in load_json("packing_reports.json")
    )
    by_condition = {item.condition: item for item in reports}

    assert set(by_condition) == {
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
        ConditionName.A_FIXED_SELECT,
    }
    assert by_condition[ConditionName.C1_LLM_PRE].complete_evidence_snapshot is True
    assert by_condition[ConditionName.C2_LLM_QUERY].complete_evidence_packet is True
    fixed = by_condition[ConditionName.A_FIXED_SELECT]
    assert fixed.complete_evidence_packet is True
    assert fixed.complete_sealed_ontology is True
    assert "sealed_ontology" in fixed.required_section_names
    for report in reports:
        assert report.truncation_applied is False
        assert not report.omitted_section_names
        assert report.input_token_count == sum(item.token_count for item in report.sections)
        prompt_section = next(item for item in report.sections if item.name == "system_prompt")
        assert prompt_section.section_content_hash == EXPECTED_PROMPT_HASHES[report.condition]


def test_invalid_repair_case_is_fact_free_and_fails_packet_and_horizon_gates() -> None:
    raw = load_json("invalid_repair_case.json")
    request = ConstructionRequest.model_validate(load_json("c2_query_request.json"))
    invalid_draft = OntologyDraft.model_validate(raw["base_draft"])
    corrected_draft = OntologyDraft.model_validate(raw["corrected_draft"])
    recorded_report = BoundaryValidationReport.model_validate(raw["validation_report"])
    lineage = RepairLineageMetadata.model_validate(raw["repair_lineage"])
    snapshot, packet, context = full_validation_inputs(request)
    support = {
        (record_id, evidence_id): GroundingSupportStatus.SUPPORTED
        for record_id, citations in cited_records(invalid_draft)
        for evidence_id in citations
    }
    actual_report = validate_draft_evidence_grounding(
        snapshot=snapshot,
        packet=packet,
        context=context,
        draft=invalid_draft,
        support_assessments=support,
    )
    corrected_support = {
        (record_id, evidence_id): GroundingSupportStatus.SUPPORTED
        for record_id, citations in cited_records(corrected_draft)
        for evidence_id in citations
    }
    corrected_report = validate_draft_evidence_grounding(
        snapshot=snapshot,
        packet=packet,
        context=context,
        draft=corrected_draft,
        support_assessments=corrected_support,
    )
    corrected_payload = strip_content_hashes(corrected_draft.model_dump(mode="json"))
    indexed_by_id = {item.evidence_id: item for item in packet.evidence}
    for assertion in corrected_payload["instance_graph"]["assertions"]:
        for provenance in assertion["provenance"]:
            indexed = indexed_by_id[provenance["evidence_id"]]
            provenance["locator"] = indexed.provenance.locator
            provenance["source_artifact_hash"] = indexed.provenance.source_artifact_hash
            provenance["confidence"] = min(indexed.confidence, indexed.provenance.confidence)
    source_bound_corrected = OntologyDraft.model_validate(corrected_payload)
    source_bound_corrected_report = validate_draft_evidence_grounding(
        snapshot=snapshot,
        packet=packet,
        context=context,
        draft=source_bound_corrected,
        support_assessments=corrected_support,
    )

    expected_codes = {
        ValidationCode.EVIDENCE_OUTSIDE_SNAPSHOT,
        ValidationCode.EVIDENCE_OUTSIDE_PACKET,
        ValidationCode.HORIZON_REJECTED_EVIDENCE,
        ValidationCode.REVELATION_HORIZON_LEAK,
    }
    assert not recorded_report.accepted
    assert len(invalid_draft.instance_graph.assertions) == 1
    assert {item.path for item in recorded_report.diagnostics} == {"instance_graph.assertions"}
    assert not actual_report.accepted
    assert not corrected_report.accepted
    assert {item.code for item in corrected_report.diagnostics} == {
        ValidationCode.PROVENANCE_MISMATCH
    }
    assert source_bound_corrected_report.accepted
    assert expected_codes.issubset({item.code for item in actual_report.diagnostics})
    assert expected_codes == {item.code for item in recorded_report.diagnostics}
    assert lineage.base_attempt_id == raw["base_attempt_id"]
    assert lineage.repair_attempt_id == raw["repair_attempt_id"]
    assert lineage.semantic_request_hash == canonical_sha256(load_json("c2_query_request.json"))
    assert lineage.base_output_hash == canonical_sha256(raw["base_draft"])
    assert lineage.validation_record_hash == canonical_sha256(raw["validation_report"])
    assert validate_single_repair_lineage(
        (lineage,), known_attempt_ids=(raw["base_attempt_id"],)
    ).accepted
    assert validate_repair_preservation(
        base_draft=raw["base_draft"],
        repaired_draft=raw["corrected_draft"],
        diagnosed_paths=tuple(item.path for item in recorded_report.diagnostics),
    ).accepted

    diagnostic_payload = raw["validation_report"]["diagnostics"]
    forbidden_repair_keys = {
        "gold",
        "proposed_fact",
        "replacement",
        "replacement_value",
        "suggested_value",
    }
    assert forbidden_repair_keys.isdisjoint(walk_keys(diagnostic_payload))
    assert all("set it to" not in item["message"].casefold() for item in diagnostic_payload)
    bad_assertion = next(
        item
        for item in invalid_draft.instance_graph.assertions
        if item.assertion_id == "assert-c2-enabled"
    )
    assert "ev-08" in bad_assertion.evidence_ids
    assert "ev-08" not in request.packet.ordered_evidence_ids
    assert "ev-08" not in snapshot.eligible_evidence_ids
    assert bad_assertion.temporal_scope.revelation_position.revelation_order > (
        request.context.spoiler_horizon.max_revelation_position.revelation_order
    )
    corrected_assertion = corrected_draft.instance_graph.assertions[0]
    assert bad_assertion.temporal_scope.story_time.point == 2
    assert bad_assertion.temporal_scope.validity_time.point == 2
    assert bad_assertion.temporal_scope.discourse_position.passage_order == 6
    assert corrected_assertion.temporal_scope.story_time.point == 2
    assert corrected_assertion.temporal_scope.validity_time.point == 2
    assert corrected_assertion.temporal_scope.revelation_position.revelation_order == 6
