from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from story_projection_onto.conditions.base import ProduceInputs, RunConditionConfig
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
    EvidencePacket,
    EvidenceRecord,
    EvidenceSnapshot,
    EvidenceSupportStatus,
    OntologyDraft,
    PreconstructionRequest,
    ProvenanceReference,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RunOutcome,
    TemporalDeterminationStatus,
    ValidationRecord,
    ValidationStatus,
)
from story_projection_onto.llm import FixedSelectCapabilityError

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
    return RunConditionConfig(
        config_id=f"config-{condition.value}",
        condition=condition,
        budgets=budgets,
        maximum_input_tokens=10_240,
        maximum_output_tokens=2_048,
        seed_block=seed_block,
        source_c1_seed_block=(seed_block if condition is ConditionName.A_FIXED_SELECT else None),
        model_stack_hash=None if is_c0 else digest("one-pinned-model-stack"),
        decoding_family_hash=None if is_c0 else digest("one-decoding-family"),
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
    preparation = seal_c1_preconstruction(
        request=request,
        draft=draft,
        seed_block=1,
        constructed_at=datetime(2026, 9, 3, 11, 46, tzinfo=UTC),
        sealed_at=datetime(2026, 9, 3, 11, 47, tzinfo=UTC),
    )
    return fixture_request, request, draft, preparation, snapshot, packet, context


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
        upper_ontology=fixture_request.upper_ontology,
        run_config=config,
    )
    first = LLMPreCondition().produce(first_inputs)

    second_context_payload = context.model_dump(exclude={"content_hash"})
    second_context_payload["context_id"] = "phase1-context-second"
    second_context_payload["wording"] = "Which reported beliefs concern the bridge?"
    second_context_payload["lens"] = "knowledge and belief"
    second_context_payload["target"] = "reported bridge claim"
    second = LLMPreCondition().produce(
        ProduceInputs(
            preparation=preparation,
            snapshot=snapshot,
            packet=packet,
            context=QueryContext(**second_context_payload),
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
    draft = OntologyDraft.model_validate(load_json("c2_query_output.json"))
    assert any(item.operator.value not in {"selection", "compression"} for item in draft.decisions)
    attempt = finalize_c2_draft(
        inputs,
        request=request,
        draft=draft,
        validation_records=(accepted_validation("c2-draft", QUERY_REVEAL + timedelta(seconds=50)),),
        completed_at=QUERY_REVEAL + timedelta(minutes=1),
        raw_output_hash=digest("real-parsed-c2-output-fixture"),
    )
    assert attempt.projection is not None
    assert attempt.projection.construction_seal is None
    certificate = attempt.projection.construction_certificate
    assert certificate is not None
    assert certificate.pre_query_inventory_hash == preparation.empty_inventory.content_hash
    assert all(item.decided_at >= QUERY_REVEAL for item in certificate.decisions)


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
    attempt = finalize_fixed_select_draft(
        inputs,
        request=request,
        draft=selected,
        validation_records=(
            accepted_validation("fixed-draft", QUERY_REVEAL + timedelta(seconds=50)),
        ),
        completed_at=QUERY_REVEAL + timedelta(minutes=1),
        raw_output_hash=digest("real-parsed-fixed-output-fixture"),
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
            draft=mutated,
            validation_records=(
                accepted_validation("bad-fixed", QUERY_REVEAL + timedelta(seconds=50)),
            ),
            completed_at=QUERY_REVEAL + timedelta(minutes=1),
            raw_output_hash=digest("mutated-fixed-output"),
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
    with pytest.raises(ValidationError, match="budgets differ"):
        ProduceInputs(
            preparation=preparation,
            snapshot=snapshot,
            packet=packet,
            context=QueryContext(**changed_context),
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
