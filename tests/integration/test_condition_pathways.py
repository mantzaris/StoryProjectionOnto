from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from story_projection_onto.conditions.base import (
    ConditionPreparation,
    ProduceInputs,
    RunConditionConfig,
)
from story_projection_onto.conditions.c1 import (
    LLMPreCondition,
    build_c1_preconstruction_request,
    seal_c1_preconstruction,
)
from story_projection_onto.conditions.c2 import (
    build_c2_construction_request,
    failed_c2_attempt,
    finalize_c2_draft,
    prepare_empty_c2_inventory,
)
from story_projection_onto.conditions.fixed_select import (
    build_fixed_select_request,
    finalize_fixed_select_draft,
    prepare_fixed_selection,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCapabilities,
    ConstructionOperator,
    ConstructionRequest,
    DiscoursePosition,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    OntologyDraft,
    OntologyProjection,
    OutputBudgets,
    PreconstructionRequest,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    ProvenanceReference,
    QueryAccessEvent,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RunOutcome,
    SpoilerHorizon,
    ValidatedGeneration,
    ValidationRecord,
    canonical_sha256,
    normalize_generation_metadata,
    projection_validation_target_hash,
    runtime_structural_acceptance_record,
    to_model_visible_context_for_condition,
    to_model_visible_query,
)
from story_projection_onto.llm import FixedSelectCapabilityError
from story_projection_onto.validate import (
    ValidationCode,
    validate_draft_structure,
    validate_repair_preservation,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "phase1"
SNAPSHOT_SEALED = datetime(2026, 9, 3, 11, 35, tzinfo=UTC)
QUERY_REVEAL = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def strip_content_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, (list, tuple)):
        return [strip_content_hashes(item) for item in value]
    return value


def full_inputs():
    fixture_request = PreconstructionRequest.model_validate(load_json("c1_pre_request.json"))
    records = tuple(
        EvidenceRecord(
            evidence_id=item.evidence_id,
            passage_id=f"passage-{index}",
            text=item.text,
            text_hash=digest(item.text),
            discourse_position=item.discourse_position,
            mention_candidates=item.mention_candidates,
            event_candidates=item.event_candidates,
            relation_phrase_candidates=item.relation_phrase_candidates,
            temporal_clues=item.temporal_clues,
            provenance=ProvenanceReference(
                provenance_id=f"provenance-{index}",
                evidence_id=item.evidence_id,
                extraction_method="phase-one fixture",
                locator=f"fixture:{index}",
                source_artifact_hash=digest(item.text),
                confidence=1.0,
            ),
            confidence=1.0,
            release_class=ReleaseClass.PUBLIC,
        )
        for index, item in enumerate(fixture_request.evidence, 1)
    )
    visible_query_request = load_json("c2_query_request.json")
    visible_context = visible_query_request["context"]
    context = QueryContext(
        context_id="phase1-context",
        **visible_context,
        revealed_at=QUERY_REVEAL,
    )
    snapshot = EvidenceSnapshot(
        snapshot_id="phase1-condition-snapshot",
        corpus_id="phase1-public-synthetic",
        world_or_window_id="phase1-harbor",
        horizon=context.spoiler_horizon,
        eligible_evidence_ids=tuple(item.evidence_id for item in records),
        index_config_hash=digest("index-v1"),
        created_at=SNAPSHOT_SEALED - timedelta(minutes=5),
        sealed_at=SNAPSHOT_SEALED,
        release_class=ReleaseClass.PUBLIC,
    )
    packet = EvidencePacket(
        packet_id="phase1-condition-packet",
        snapshot_hash=snapshot.content_hash,
        evidence=records,
        ordered_evidence_ids=snapshot.eligible_evidence_ids,
        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
        token_count=1_500,
        created_at=QUERY_REVEAL,
        release_class=ReleaseClass.PUBLIC,
    )
    return fixture_request, records, snapshot, packet, context


def source_bound_draft(
    name: str,
    evidence: tuple[EvidenceRecord, ...],
) -> OntologyDraft:
    payload = strip_content_hashes(load_json(name))
    indexed_by_id = {item.evidence_id: item for item in evidence}
    for assertion in payload["instance_graph"]["assertions"]:
        for provenance in assertion["provenance"]:
            indexed = indexed_by_id[provenance["evidence_id"]]
            provenance["locator"] = indexed.provenance.locator
            provenance["source_artifact_hash"] = indexed.provenance.source_artifact_hash
            provenance["confidence"] = min(indexed.confidence, indexed.provenance.confidence)
    return OntologyDraft.model_validate(payload)


def run_config(
    condition: ConditionName,
    *,
    budgets,
    upper_hash: str,
    seed_block: int | None,
) -> RunConditionConfig:
    is_c0 = condition is ConditionName.C0_CLASSICAL_PRE
    runtime = None
    if not is_c0:
        fixture_name = {
            ConditionName.C1_LLM_PRE: "c1_pre_request.json",
            ConditionName.C2_LLM_QUERY: "c2_query_request.json",
            ConditionName.A_FIXED_SELECT: "fixed_select_request.json",
            ConditionName.A_NO_CONTEXT: "c2_query_request.json",
            ConditionName.A_NO_TEMPORAL_EPISTEMIC: "c2_query_request.json",
            ConditionName.A_NO_RARE_GUARD: "c2_query_request.json",
        }[condition]
        runtime = load_json(fixture_name)["runtime"]
    return RunConditionConfig(
        config_id=f"config-{condition.value}",
        condition=condition,
        budgets=budgets,
        maximum_input_tokens=10_240,
        maximum_output_tokens=2_048,
        seed_block=seed_block,
        source_c1_seed_block=(seed_block if condition is ConditionName.A_FIXED_SELECT else None),
        model_stack_hash=None if is_c0 else digest("one-pinned-model-stack"),
        decoding_manifest_hash=None if is_c0 else runtime["decoding_config_hash"],
        decoding_family_hash=None if is_c0 else digest("registered-first-pass-family"),
        seed_manifest_hash=None if is_c0 else digest("one-seed-manifest"),
        resolved_seed=None if is_c0 else 1_234_567,
        prompt_hash=None if is_c0 else runtime["prompt_hash"],
        output_schema_hash=None if is_c0 else runtime["output_schema_hash"],
        scored_schema_hash=canonical_sha256(
            OntologyProjection.model_json_schema(mode="validation")
        ),
        capability_manifest_hash=None if is_c0 else digest(f"capability-{condition.value}"),
        validator_hash=digest("validator-v1"),
        upper_ontology_hash=upper_hash,
    )


def accepted_validation(target: str, at: datetime) -> ValidationRecord:
    return runtime_structural_acceptance_record(
        validation_id=f"validation-{target}",
        target_id=target,
        validated_at=at,
    )


def query_timing(
    preparation: ConditionPreparation,
    snapshot: EvidenceSnapshot,
    packet: EvidencePacket,
    context: QueryContext,
    *,
    seed_block: int | None,
    accessed_at: datetime | None = None,
) -> dict[str, object]:
    physical_access = accessed_at or context.revealed_at
    lineage_artifact = (
        preparation.sealed_preontology.construction_seal
        if preparation.sealed_preontology is not None
        else preparation.empty_inventory
        if preparation.empty_inventory is not None
        else preparation.fixed_selection
    )
    assert lineage_artifact is not None
    barrier = PrequeryBarrier(
        barrier_id=f"barrier-{context.context_id}-{preparation.condition.value}",
        execution_id="phase1-condition-pathways",
        execution_manifest_hash=digest("phase1-condition-manifest"),
        neutral_evidence_artifact_hashes=(digest("phase1-neutral-evidence"),),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id=snapshot.world_or_window_id,
                condition=preparation.condition,
                seed_block=seed_block,
                snapshot_hash=snapshot.content_hash,
                preparation_hash=preparation.content_hash,
                lineage_artifact_hash=lineage_artifact.content_hash,
                completed_at=preparation.completed_at,
            ),
        ),
        sealed_at=physical_access - timedelta(microseconds=1),
    )
    access = QueryAccessEvent(
        access_event_id=f"access-{context.context_id}",
        execution_id="phase1-condition-pathways",
        query_context_hash=context.content_hash,
        model_visible_query_hash=to_model_visible_query(context).content_hash,
        snapshot_hash=snapshot.content_hash,
        stage_manifest_hash=digest(f"stage-{context.context_id}"),
        query_artifact_hash=digest(f"query-{context.context_id}"),
        prequery_barrier_hash=barrier.content_hash,
        packet_hash=packet.content_hash,
        registered_revealed_at=context.revealed_at,
        accessed_at=physical_access,
    )
    return {
        "query_access": access,
        "prequery_barrier": barrier,
        "query_processing_started_at": physical_access + timedelta(microseconds=1),
    }


def trusted_generation(
    *,
    condition: ConditionName,
    request: PreconstructionRequest | ConstructionRequest,
    raw_draft: OntologyDraft,
    config: RunConditionConfig,
    stage_manifest_hash: str,
    horizon: SpoilerHorizon,
    query_access: QueryAccessEvent | None = None,
    barrier: PrequeryBarrier | None = None,
) -> ValidatedGeneration:
    request_hash = request.content_hash
    request_time = request.requested_at
    decision_time = request_time + timedelta(seconds=1)
    completed_at = request_time + timedelta(seconds=2)
    validated_at = request_time + timedelta(seconds=3)
    normalized = normalize_generation_metadata(
        raw_draft,
        decision_recorded_at=decision_time,
        input_tokens=raw_draft.budget_accounting.input_tokens,
        output_tokens=raw_draft.budget_accounting.output_tokens,
    )
    validation = accepted_validation(normalized.content_hash, validated_at)
    evidence = (
        request.evidence
        if isinstance(request, PreconstructionRequest)
        else request.packet.evidence
    )
    structure = validate_draft_structure(
        draft=normalized,
        upper_ontology=request.upper_ontology,
        evidence=evidence,
        horizon=horizon,
        budgets=request.budgets,
        capabilities=request.capabilities,
    )
    assert structure.accepted
    assert config.model_stack_hash is not None
    assert config.decoding_manifest_hash is not None
    assert config.decoding_family_hash is not None
    assert config.seed_manifest_hash is not None
    assert config.resolved_seed is not None
    assert config.prompt_hash is not None
    assert config.output_schema_hash is not None
    assert config.capability_manifest_hash is not None
    return ValidatedGeneration(
        generation_id=f"generation-{condition.value}",
        condition=condition,
        request_hash=request_hash,
        raw_output_artifact_hash=digest(f"raw-{condition.value}"),
        raw_parsed_draft=raw_draft,
        draft=normalized,
        normalized_draft_hash=normalized.content_hash,
        stage_manifest_hash=stage_manifest_hash,
        query_access_event_hash=(query_access.content_hash if query_access else None),
        prequery_barrier_hash=(barrier.content_hash if barrier else None),
        packing_report_hash=digest(f"packing-{condition.value}"),
        capability_manifest_hash=config.capability_manifest_hash,
        seed_manifest_hash=config.seed_manifest_hash,
        prompt_hash=config.prompt_hash,
        output_schema_hash=config.output_schema_hash,
        decoding_manifest_hash=config.decoding_manifest_hash,
        validator_hash=config.validator_hash,
        model_stack_hash=config.model_stack_hash,
        seed=config.resolved_seed,
        input_tokens=normalized.budget_accounting.input_tokens,
        output_tokens=normalized.budget_accounting.output_tokens,
        generation_started_at=request_time,
        generation_completed_at=completed_at,
        decision_recorded_at=decision_time,
        validation_records=(validation,),
        validator_report_hashes=(structure.content_hash,),
        validated_at=validated_at,
    )


def test_final_projection_underdeclared_display_count_is_a_repair_trigger() -> None:
    request, records, snapshot, *_ = full_inputs()
    payload = strip_content_hashes(
        source_bound_draft("c1_pre_output.json", records).model_dump(mode="python")
    )
    payload["budget_accounting"]["display_nodes_used"] = 0
    draft = OntologyDraft.model_validate(payload)

    report = validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=records,
        horizon=snapshot.horizon,
        budgets=request.budgets,
        capabilities=ConstructionCapabilities.active_construction(),
    )

    assert report.validation_status == "rejected"
    assert any(
        item.code is ValidationCode.BUDGET_ACCOUNTING_MISMATCH
        and item.path == "budget_accounting.display_nodes_used"
        and "must equal" in item.message
        for item in report.diagnostics
    )
    repaired_payload = strip_content_hashes(draft.model_dump(mode="python"))
    repaired_payload["budget_accounting"]["display_nodes_used"] = min(
        len(draft.instance_graph.entities) + len(draft.instance_graph.events),
        request.budgets.display_node_budget,
    )
    repair_guard = validate_repair_preservation(
        base_draft=strip_content_hashes(draft.model_dump(mode="python")),
        repaired_draft=repaired_payload,
        diagnosed_paths=tuple(item.path for item in report.diagnostics),
    )
    assert repair_guard.accepted


def test_final_projection_infeasible_display_budget_is_a_repair_trigger() -> None:
    request, records, snapshot, *_ = full_inputs()
    payload = strip_content_hashes(
        source_bound_draft("c1_pre_output.json", records).model_dump(mode="python")
    )
    payload["budget_accounting"].update(
        display_nodes_used=1,
        display_assertions_used=1,
    )
    draft = OntologyDraft.model_validate(payload)
    impossible_budgets = OutputBudgets(
        node_budget=request.budgets.node_budget,
        assertion_budget=request.budgets.assertion_budget,
        display_node_budget=1,
        display_assertion_budget=1,
        repair_attempt_budget=request.budgets.repair_attempt_budget,
    )

    report = validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=records,
        horizon=snapshot.horizon,
        budgets=impossible_budgets,
        capabilities=ConstructionCapabilities.active_construction(),
    )

    assert report.validation_status == "rejected"
    assert any(
        item.code is ValidationCode.BUDGET_ACCOUNTING_MISMATCH
        and "endpoint closure" in item.message
        for item in report.diagnostics
    )


def test_comprehensive_c1_prebuild_is_not_constrained_by_final_display_feasibility() -> None:
    request, records, snapshot, *_ = full_inputs()
    payload = strip_content_hashes(
        source_bound_draft("c1_pre_output.json", records).model_dump(mode="python")
    )
    payload["budget_accounting"].update(
        display_nodes_used=1,
        display_assertions_used=1,
    )
    draft = OntologyDraft.model_validate(payload)
    prebuild_budgets = OutputBudgets(
        node_budget=request.budgets.node_budget,
        assertion_budget=request.budgets.assertion_budget,
        display_node_budget=1,
        display_assertion_budget=1,
        repair_attempt_budget=request.budgets.repair_attempt_budget,
    )

    report = validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=records,
        horizon=snapshot.horizon,
        budgets=prebuild_budgets,
        capabilities=ConstructionCapabilities.prequery_construction(),
    )

    assert report.accepted


def c1_preparation():
    fixture_request, records, snapshot, packet, context = full_inputs()
    request = build_c1_preconstruction_request(
        snapshot_hash=snapshot.content_hash,
        snapshot_sealed_at=snapshot.sealed_at,
        sealed_horizon=snapshot.horizon,
        ordered_snapshot_evidence_ids=snapshot.eligible_evidence_ids,
        evidence=records,
        upper_ontology=fixture_request.upper_ontology,
        preconstruction_budgets=fixture_request.budgets,
        runtime=fixture_request.runtime,
        requested_at=datetime(2026, 9, 3, 11, 40, tzinfo=UTC),
    )
    draft = source_bound_draft("c1_pre_output.json", records)
    config = run_config(
        ConditionName.C1_LLM_PRE,
        budgets=context.budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    generation = trusted_generation(
        condition=ConditionName.C1_LLM_PRE,
        request=request,
        raw_draft=draft,
        config=config,
        stage_manifest_hash=digest("phase1-prequery-stage"),
        horizon=snapshot.horizon,
    )
    preparation = seal_c1_preconstruction(
        request=request,
        generation=generation,
        seed_block=1,
        sealed_at=datetime(2026, 9, 3, 11, 47, tzinfo=UTC),
    )
    return fixture_request, request, generation.draft, preparation, snapshot, packet, context


def test_c1_request_is_query_blind_and_one_seal_is_reused_across_contexts() -> None:
    fixture_request, request, _, preparation, snapshot, packet, context = c1_preparation()
    serialized = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    for forbidden in (
        "context_id",
        "query_context",
        "pair_id",
        "gold",
        "expected_effect",
        "revealed_at",
    ):
        assert forbidden not in serialized
    assert tuple(item.evidence_id for item in request.evidence) == snapshot.eligible_evidence_ids
    assert request.sealed_horizon == snapshot.horizon
    assert "sealed_horizon" in request.model_dump(mode="json")
    assert request.capabilities.select_existing is False
    assert request.capabilities.compress_existing is False

    config = run_config(
        ConditionName.C1_LLM_PRE,
        budgets=context.budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    first_inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        **query_timing(preparation, snapshot, packet, context, seed_block=1),
        upper_ontology=fixture_request.upper_ontology,
        run_config=config,
    )
    first = LLMPreCondition().produce(first_inputs)

    second_context_payload = context.model_dump(exclude={"content_hash"})
    second_context_payload["context_id"] = "phase1-context-second"
    second_context_payload["wording"] = "Which reported beliefs concern the bridge?"
    second_context_payload["lens"] = "knowledge and belief"
    second_context_payload["target"] = "reported bridge claim"
    second_context = QueryContext(**second_context_payload)
    second = LLMPreCondition().produce(
        ProduceInputs(
            preparation=preparation,
            snapshot=snapshot,
            packet=packet,
            context=second_context,
            **query_timing(preparation, snapshot, packet, second_context, seed_block=1),
            upper_ontology=fixture_request.upper_ontology,
            run_config=config,
        )
    )
    assert first.projection is not None and second.projection is not None
    assert (
        first.projection.construction_seal.content_hash
        == second.projection.construction_seal.content_hash
    )
    assert all(
        item.operator is ConstructionOperator.SELECTION
        for item in (*first.projection.decisions, *second.projection.decisions)
    )


def test_c1_request_rejects_evidence_past_its_model_visible_sealed_horizon() -> None:
    fixture_request, records, snapshot, *_ = full_inputs()
    truncated_horizon = SpoilerHorizon(
        horizon_id="phase1-truncated-horizon",
        max_discourse_position=DiscoursePosition(passage_order=6),
        max_revelation_position=snapshot.horizon.max_revelation_position,
    )

    with pytest.raises(ValidationError, match="evidence exceeds its sealed horizon: ev-07"):
        build_c1_preconstruction_request(
            snapshot_hash=snapshot.content_hash,
            snapshot_sealed_at=snapshot.sealed_at,
            sealed_horizon=truncated_horizon,
            ordered_snapshot_evidence_ids=snapshot.eligible_evidence_ids,
            evidence=records,
            upper_ontology=fixture_request.upper_ontology,
            preconstruction_budgets=fixture_request.budgets,
            runtime=fixture_request.runtime,
            requested_at=datetime(2026, 9, 3, 11, 40, tzinfo=UTC),
        )


def test_c1_projection_closes_all_qualified_and_temporal_dependencies() -> None:
    fixture_request, records, snapshot, packet, base_context = full_inputs()
    request = build_c1_preconstruction_request(
        snapshot_hash=snapshot.content_hash,
        snapshot_sealed_at=snapshot.sealed_at,
        sealed_horizon=snapshot.horizon,
        ordered_snapshot_evidence_ids=snapshot.eligible_evidence_ids,
        evidence=records,
        upper_ontology=fixture_request.upper_ontology,
        preconstruction_budgets=fixture_request.budgets,
        runtime=fixture_request.runtime,
        requested_at=datetime(2026, 9, 3, 11, 40, tzinfo=UTC),
    )
    draft_payload = strip_content_hashes(
        source_bound_draft("c1_pre_output.json", records).model_dump(mode="python")
    )
    graph_payload = draft_payload["instance_graph"]
    enabled = next(
        item
        for item in graph_payload["assertions"]
        if item["assertion_id"] == "assert-c1-enabled"
    )
    enabled["proposition_content_id"] = "prop-c1-enabled"
    enabled["narrative_commitment"] = "holder_attributed"
    enabled["temporal_scope"]["story_time"] = {
        "kind": "relative",
        "anchor_id": "ent-c1-lio",
        "relation": "after",
    }
    enabled["temporal_scope"]["validity_time"] = {
        "kind": "partial_order",
        "partial_order": [
            {
                "left_id": "event-c1-delivery",
                "right_id": "event-c1-opening",
                "relation": "before",
            }
        ],
    }
    enabled["epistemic_scope"] = {
        "holder_id": "ent-c1-lio",
        "attitude": "believed",
        "proposition_content_id": "prop-c1-enabled",
        "holder_relative_time": {
            "kind": "relative",
            "anchor_id": "event-c1-delivery",
            "relation": "after",
        },
        "evidence_ids": ["ev-04"],
    }
    graph_payload["proposition_contents"] = [
        {
            "proposition_content_id": "prop-c1-enabled",
            "predicate_id": enabled["predicate_id"],
            "subject_id": enabled["subject_id"],
            "object_id": enabled["object_id"],
            "temporal_content": {
                "story_time": {
                    "kind": "relative",
                    "anchor_id": "event-c1-delivery",
                    "relation": "after",
                },
                "validity_time": {
                    "kind": "partial_order",
                    "partial_order": [
                        {
                            "left_id": "ent-c1-lio",
                            "right_id": "event-c1-opening",
                            "relation": "before",
                        }
                    ],
                },
                "discourse_position": enabled["temporal_scope"]["discourse_position"],
                "revelation_position": enabled["temporal_scope"]["revelation_position"],
            },
            "evidence_ids": enabled["evidence_ids"],
        }
    ]
    for entity in graph_payload["entities"]:
        if entity["entity_id"] in {"ent-c1-lio", "ent-c1-seal"}:
            entity["description_assertion_ids"] = ["assert-c1-enabled"]
    opening = next(
        item
        for item in graph_payload["events"]
        if item["event_id"] == "event-c1-opening"
    )
    opening["description_assertion_ids"] = ["assert-c1-enabled"]
    raw_draft = OntologyDraft.model_validate(draft_payload)

    final_budgets = OutputBudgets(
        node_budget=4,
        assertion_budget=2,
        display_node_budget=4,
        display_assertion_budget=2,
    )
    context = QueryContext(
        context_id="phase1-context-c1-qualified-closure",
        wording="Why did the rare mark enable the opening?",
        lens="holder-attributed causal explanation",
        target="rare mark enabled opening",
        story_scope=base_context.story_scope,
        spoiler_horizon=base_context.spoiler_horizon,
        viewpoint=base_context.viewpoint,
        abstraction=base_context.abstraction,
        budgets=final_budgets,
        revealed_at=base_context.revealed_at,
    )
    config = run_config(
        ConditionName.C1_LLM_PRE,
        budgets=final_budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    generation = trusted_generation(
        condition=ConditionName.C1_LLM_PRE,
        request=request,
        raw_draft=raw_draft,
        config=config,
        stage_manifest_hash=digest("phase1-qualified-prequery-stage"),
        horizon=snapshot.horizon,
    )
    preparation = seal_c1_preconstruction(
        request=request,
        generation=generation,
        seed_block=1,
        sealed_at=datetime(2026, 9, 3, 11, 47, tzinfo=UTC),
    )
    inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        **query_timing(preparation, snapshot, packet, context, seed_block=1),
        upper_ontology=fixture_request.upper_ontology,
        run_config=config,
    )
    attempt = LLMPreCondition().produce(inputs)
    assert attempt.projection is not None
    projection = attempt.projection

    assert {item.assertion_id for item in projection.instance_graph.assertions} == {
        "assert-c1-chain",
        "assert-c1-enabled",
    }
    assert {item.entity_id for item in projection.instance_graph.entities} == {
        "ent-c1-lio",
        "ent-c1-seal",
    }
    assert {item.event_id for item in projection.instance_graph.events} == {
        "event-c1-delivery",
        "event-c1-opening",
    }
    assert {
        item.proposition_content_id
        for item in projection.instance_graph.proposition_contents
    } == {"prop-c1-enabled"}
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


def test_c2_has_empty_prequery_inventory_then_active_postquery_construction() -> None:
    fixture_request, _, _, _, snapshot, packet, context = c1_preparation()
    preparation = prepare_empty_c2_inventory(
        snapshot,
        recorded_at=datetime(2026, 9, 3, 11, 48, tzinfo=UTC),
    )
    config = run_config(
        ConditionName.C2_LLM_QUERY,
        budgets=context.budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        **query_timing(preparation, snapshot, packet, context, seed_block=1),
        upper_ontology=fixture_request.upper_ontology,
        run_config=config,
    )
    runtime = load_json("c2_query_request.json")["runtime"]
    request = build_c2_construction_request(
        inputs,
        runtime=fixture_request.runtime.__class__.model_validate(runtime),
        requested_at=QUERY_REVEAL + timedelta(seconds=1),
    )
    assert request.fixed_ontology is None
    assert preparation.empty_inventory is not None
    assert not preparation.empty_inventory.ontology_refs
    assert not preparation.empty_inventory.entity_ids
    raw_draft = source_bound_draft("c2_query_output.json", tuple(packet.evidence))
    assert any(
        item.operator.value not in {"selection", "compression"}
        for item in raw_draft.decisions
    )
    generation = trusted_generation(
        condition=ConditionName.C2_LLM_QUERY,
        request=request,
        raw_draft=raw_draft,
        config=config,
        stage_manifest_hash=inputs.query_access.stage_manifest_hash,
        horizon=inputs.snapshot.horizon,
        query_access=inputs.query_access,
        barrier=inputs.prequery_barrier,
    )
    attempt = finalize_c2_draft(
        inputs,
        request=request,
        generation=generation,
    )
    assert attempt.projection is not None
    assert attempt.projection.construction_seal is None
    certificate = attempt.projection.construction_certificate
    assert certificate is not None
    assert certificate.pre_query_inventory_hash == preparation.empty_inventory.content_hash
    assert all(item.decided_at >= QUERY_REVEAL for item in certificate.decisions)


def test_no_context_request_hides_query_semantics_and_accepts_validated_draft() -> None:
    fixture_request, _, snapshot, packet, context = full_inputs()
    condition = ConditionName.A_NO_CONTEXT
    preparation = prepare_empty_c2_inventory(
        snapshot,
        recorded_at=datetime(2026, 9, 3, 11, 48, tzinfo=UTC),
        condition=condition,
    )
    config = run_config(
        condition,
        budgets=context.budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        **query_timing(preparation, snapshot, packet, context, seed_block=1),
        upper_ontology=fixture_request.upper_ontology,
        run_config=config,
    )
    runtime_payload = load_json("c2_query_request.json")["runtime"]
    runtime = fixture_request.runtime.__class__.model_validate(runtime_payload)
    request = build_c2_construction_request(
        inputs,
        runtime=runtime,
        requested_at=QUERY_REVEAL + timedelta(seconds=1),
    )
    assert request.context == to_model_visible_context_for_condition(context, condition)
    visible_context = request.context.model_dump(
        mode="json", exclude={"schema_version", "content_hash"}
    )
    assert set(visible_context) == {"generic_request", "budgets"}

    invalid_payload = strip_content_hashes(request.model_dump(mode="python"))
    invalid_payload["context"] = strip_content_hashes(
        to_model_visible_query(context).model_dump(mode="python")
    )
    with pytest.raises(ValidationError, match="query-free generic context"):
        ConstructionRequest.model_validate(invalid_payload)

    raw_draft = source_bound_draft("c2_query_output.json", tuple(packet.evidence))
    generation = trusted_generation(
        condition=condition,
        request=request,
        raw_draft=raw_draft,
        config=config,
        stage_manifest_hash=inputs.query_access.stage_manifest_hash,
        horizon=inputs.snapshot.horizon,
        query_access=inputs.query_access,
        barrier=inputs.prequery_barrier,
    )
    attempt = finalize_c2_draft(inputs, request=request, generation=generation)
    assert attempt.outcome is RunOutcome.SUCCEEDED
    assert attempt.projection is not None


def test_fixed_select_receives_complete_same_seed_c1_and_rejects_mutation() -> None:
    fixture_request, _, source_draft, c1_ready, snapshot, packet, context = c1_preparation()
    fixed_ready = prepare_fixed_selection(
        c1_ready,
        seed_block=1,
        prepared_at=datetime(2026, 9, 3, 11, 48, tzinfo=UTC),
    )
    config = run_config(
        ConditionName.A_FIXED_SELECT,
        budgets=context.budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    inputs = ProduceInputs(
        preparation=fixed_ready,
        snapshot=snapshot,
        packet=packet,
        context=context,
        **query_timing(fixed_ready, snapshot, packet, context, seed_block=1),
        upper_ontology=fixture_request.upper_ontology,
        run_config=config,
    )
    runtime = fixture_request.runtime.__class__.model_validate(
        load_json("fixed_select_request.json")["runtime"]
    )
    request = build_fixed_select_request(
        inputs,
        runtime=runtime,
        requested_at=QUERY_REVEAL + timedelta(seconds=1),
    )
    assert request.fixed_ontology is not None
    assert request.fixed_ontology.local_schema == source_draft.local_schema
    assert request.fixed_ontology.instance_graph == source_draft.instance_graph
    selected = source_bound_draft("fixed_select_output.json", tuple(packet.evidence))
    generation = trusted_generation(
        condition=ConditionName.A_FIXED_SELECT,
        request=request,
        raw_draft=selected,
        config=config,
        stage_manifest_hash=inputs.query_access.stage_manifest_hash,
        horizon=inputs.snapshot.horizon,
        query_access=inputs.query_access,
        barrier=inputs.prequery_barrier,
    )
    attempt = finalize_fixed_select_draft(
        inputs,
        request=request,
        generation=generation,
    )
    assert attempt.projection is not None
    assert all(
        item.operator.value in {"selection", "compression", "supported_description"}
        for item in attempt.projection.decisions
    )

    mutated_payload = strip_content_hashes(selected.model_dump(mode="json"))
    mutated_payload["instance_graph"]["entities"][0]["contextual_role"] = "invented role"
    mutated = OntologyDraft.model_validate(mutated_payload)
    with pytest.raises(FixedSelectCapabilityError, match="semantic content changed"):
        finalize_fixed_select_draft(
            inputs,
            request=request,
            generation=trusted_generation(
                condition=ConditionName.A_FIXED_SELECT,
                request=request,
                raw_draft=mutated,
                config=config,
                stage_manifest_hash=inputs.query_access.stage_manifest_hash,
                horizon=inputs.snapshot.horizon,
                query_access=inputs.query_access,
                barrier=inputs.prequery_barrier,
            ),
        )


def test_fixed_select_cross_seed_preparation_is_mechanically_rejected() -> None:
    *_, c1_ready, _, _, _ = c1_preparation()
    with pytest.raises(ValidationError, match="same seed block"):
        prepare_fixed_selection(
            c1_ready,
            seed_block=2,
            prepared_at=datetime(2026, 9, 3, 11, 48, tzinfo=UTC),
        )


def test_produce_input_rejects_early_packet_and_budget_asymmetry() -> None:
    fixture_request, _, _, preparation, snapshot, packet, context = c1_preparation()
    config = run_config(
        ConditionName.C1_LLM_PRE,
        budgets=context.budgets,
        upper_hash=fixture_request.upper_ontology.content_hash,
        seed_block=1,
    )
    packet_values = packet.model_dump(exclude={"content_hash"})
    packet_values["created_at"] = QUERY_REVEAL - timedelta(seconds=1)
    early_packet = EvidencePacket(**packet_values)
    with pytest.raises(ValidationError, match="must not predate query reveal"):
        ProduceInputs(
            preparation=preparation,
            snapshot=snapshot,
            packet=early_packet,
            context=context,
            **query_timing(preparation, snapshot, packet, context, seed_block=1),
            upper_ontology=fixture_request.upper_ontology,
            run_config=config,
        )

    changed_context = context.model_dump(exclude={"content_hash"})
    changed_context["budgets"] = {
        "node_budget": 10,
        "assertion_budget": 10,
        "display_node_budget": 10,
        "display_assertion_budget": 10,
        "repair_attempt_budget": 1,
    }
    changed_query_context = QueryContext(**changed_context)
    with pytest.raises(ValidationError, match="budgets differ"):
        ProduceInputs(
            preparation=preparation,
            snapshot=snapshot,
            packet=packet,
            context=changed_query_context,
            **query_timing(
                preparation,
                snapshot,
                packet,
                changed_query_context,
                seed_block=1,
            ),
            upper_ontology=fixture_request.upper_ontology,
            run_config=config,
        )


def test_invalid_timeout_attempt_remains_intention_to_treat() -> None:
    fixture_request, _, _, _, snapshot, packet, context = c1_preparation()
    preparation = prepare_empty_c2_inventory(
        snapshot,
        recorded_at=datetime(2026, 9, 3, 11, 48, tzinfo=UTC),
    )
    inputs = ProduceInputs(
        preparation=preparation,
        snapshot=snapshot,
        packet=packet,
        context=context,
        **query_timing(preparation, snapshot, packet, context, seed_block=1),
        upper_ontology=fixture_request.upper_ontology,
        run_config=run_config(
            ConditionName.C2_LLM_QUERY,
            budgets=context.budgets,
            upper_hash=fixture_request.upper_ontology.content_hash,
            seed_block=1,
        ),
    )
    timeout = failed_c2_attempt(
        inputs,
        outcome=RunOutcome.TIMED_OUT,
        failure_code="standard_watchdog_timeout",
    )
    assert timeout.included_in_intention_to_treat
    assert timeout.semantic_score_for_nonempty_gold(None) == 0.0
