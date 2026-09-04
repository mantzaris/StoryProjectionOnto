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
    CommitmentCheckStatus,
    ConditionName,
    ConstructionOperator,
    ConstructionRequest,
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceSupportStatus,
    OntologyDraft,
    OntologyProjection,
    PreconstructionRequest,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    ProvenanceReference,
    QueryAccessEvent,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RunOutcome,
    TemporalDeterminationStatus,
    ValidatedGeneration,
    ValidationRecord,
    ValidationStatus,
    canonical_sha256,
    normalize_generation_metadata,
    to_model_visible_context_for_condition,
    to_model_visible_query,
)
from story_projection_onto.llm import FixedSelectCapabilityError
from story_projection_onto.validate import validate_draft_structure

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
    if isinstance(value, list):
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
    return ValidationRecord(
        validation_id=f"validation-{target}",
        target_id=target,
        validation_status=ValidationStatus.ACCEPTED,
        evidence_support_status=EvidenceSupportStatus.SUPPORTED,
        temporal_status=TemporalDeterminationStatus.VALID,
        commitment_status=CommitmentCheckStatus.VALID,
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


def c1_preparation():
    fixture_request, records, snapshot, packet, context = full_inputs()
    request = build_c1_preconstruction_request(
        snapshot_hash=snapshot.content_hash,
        snapshot_sealed_at=snapshot.sealed_at,
        ordered_snapshot_evidence_ids=snapshot.eligible_evidence_ids,
        evidence=records,
        upper_ontology=fixture_request.upper_ontology,
        preconstruction_budgets=fixture_request.budgets,
        runtime=fixture_request.runtime,
        requested_at=datetime(2026, 9, 3, 11, 40, tzinfo=UTC),
    )
    draft = OntologyDraft.model_validate(load_json("c1_pre_output.json"))
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
    raw_draft = OntologyDraft.model_validate(load_json("c2_query_output.json"))
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

    raw_draft = OntologyDraft.model_validate(load_json("c2_query_output.json"))
    generation = trusted_generation(
        condition=condition,
        request=request,
        raw_draft=raw_draft,
        config=config,
        stage_manifest_hash=inputs.query_access.stage_manifest_hash,
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
    selected = OntologyDraft.model_validate(load_json("fixed_select_output.json"))
    generation = trusted_generation(
        condition=ConditionName.A_FIXED_SELECT,
        request=request,
        raw_draft=selected,
        config=config,
        stage_manifest_hash=inputs.query_access.stage_manifest_hash,
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
