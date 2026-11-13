from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionOperator,
    ConstructionRequest,
    OntologyDraft,
)
from story_projection_onto.llm import (
    FixedSelectOutputAudit,
    ProposedOperation,
    RepairLineageMetadata,
    SealedOntologyInventory,
    SemanticFingerprint,
)
from story_projection_onto.validate import (
    ConstructionLineageAudit,
    EvidenceBoundaryRecord,
    EvidenceCitationAssessment,
    FactualClauseGrounding,
    GroundedFactualRecord,
    GroundingSupportStatus,
    SpoilerHorizonBoundary,
    ValidationCode,
    validate_construction_lineage,
    validate_draft_structure,
    validate_evidence_grounding,
    validate_fixed_select,
    validate_repair_preservation,
    validate_single_repair_lineage,
)

ROOT = Path(__file__).resolve().parents[2]
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
REVEAL = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _without_hashes(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, list):
        return [_without_hashes(child) for child in value]
    return value


def _phase1_c2_boundary(payload: dict[str, object]):
    request = ConstructionRequest.model_validate_json(
        (ROOT / "tests/fixtures/phase1/c2_query_request.json").read_text(encoding="utf-8")
    )
    draft = OntologyDraft.model_validate(_without_hashes(payload))
    return validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=request.packet.evidence,
        horizon=request.context.spoiler_horizon,
        budgets=request.budgets,
        capabilities=request.capabilities,
    )


def valid_evidence_report():
    return validate_evidence_grounding(
        snapshot_eligible_evidence_ids=("e-1", "e-2"),
        packet_ordered_evidence_ids=("e-1", "e-2"),
        packet_records=(
            EvidenceBoundaryRecord(
                evidence_id="e-1",
                discourse_position=2,
                revelation_position=2,
            ),
            EvidenceBoundaryRecord(
                evidence_id="e-2",
                discourse_position=4,
                revelation_position=3,
            ),
        ),
        horizon_rejected_evidence_ids=("e-3",),
        horizon=SpoilerHorizonBoundary(
            maximum_discourse_position=4,
            maximum_revelation_position=3,
        ),
        factual_records=(
            GroundedFactualRecord(
                record_id="assertion-1",
                citations=(
                    EvidenceCitationAssessment(
                        evidence_id="e-1",
                        support_status=GroundingSupportStatus.SUPPORTED,
                    ),
                ),
                factual_clauses=(
                    FactualClauseGrounding(
                        clause_id="assertion-1:why-matters",
                        evidence_ids=("e-1",),
                    ),
                ),
                proposition_revelation_position=3,
            ),
        ),
    )


def test_evidence_packet_horizon_and_grounding_accept_valid_fixture() -> None:
    report = valid_evidence_report()
    assert report.accepted
    assert report.diagnostics == ()


def test_evidence_validation_rejects_outside_packet_horizon_and_unsupported_claims() -> None:
    report = validate_evidence_grounding(
        snapshot_eligible_evidence_ids=("e-1", "e-2"),
        packet_ordered_evidence_ids=("e-1",),
        packet_records=(EvidenceBoundaryRecord(evidence_id="e-1", discourse_position=11),),
        horizon_rejected_evidence_ids=("e-2",),
        horizon=SpoilerHorizonBoundary(
            maximum_discourse_position=10,
            maximum_revelation_position=10,
        ),
        factual_records=(
            GroundedFactualRecord(
                record_id="assertion-1",
                citations=(
                    EvidenceCitationAssessment(
                        evidence_id="e-2",
                        support_status=GroundingSupportStatus.UNSUPPORTED,
                    ),
                ),
                factual_clauses=(
                    FactualClauseGrounding(
                        clause_id="assertion-1:why-matters",
                        evidence_ids=(),
                    ),
                ),
                proposition_revelation_position=12,
            ),
        ),
    )
    codes = {diagnostic.code for diagnostic in report.diagnostics}

    assert not report.accepted
    assert ValidationCode.DISCOURSE_HORIZON_LEAK in codes
    assert ValidationCode.REVELATION_HORIZON_LEAK in codes
    assert ValidationCode.EVIDENCE_OUTSIDE_PACKET in codes
    assert ValidationCode.HORIZON_REJECTED_EVIDENCE in codes
    assert ValidationCode.UNSUPPORTED_GROUNDING in codes
    assert ValidationCode.UNGROUNDED_FACTUAL_CLAUSE in codes


def test_unknown_citation_is_outside_both_snapshot_and_packet() -> None:
    report = validate_evidence_grounding(
        snapshot_eligible_evidence_ids=("e-1",),
        packet_ordered_evidence_ids=("e-1",),
        packet_records=(EvidenceBoundaryRecord(evidence_id="e-1", discourse_position=1),),
        horizon_rejected_evidence_ids=(),
        horizon=SpoilerHorizonBoundary(maximum_discourse_position=10),
        factual_records=(
            GroundedFactualRecord(
                record_id="assertion-unknown-citation",
                citations=(
                    EvidenceCitationAssessment(
                        evidence_id="e-unknown",
                        support_status=GroundingSupportStatus.SUPPORTED,
                    ),
                ),
            ),
        ),
    )

    diagnostics_by_code = {item.code: item for item in report.diagnostics}
    assert ValidationCode.EVIDENCE_OUTSIDE_SNAPSHOT in diagnostics_by_code
    assert ValidationCode.EVIDENCE_OUTSIDE_PACKET in diagnostics_by_code
    assert diagnostics_by_code[ValidationCode.EVIDENCE_OUTSIDE_SNAPSHOT].path == (
        "factual_records.0.citations"
    )
    assert diagnostics_by_code[ValidationCode.EVIDENCE_OUTSIDE_PACKET].path == (
        "factual_records.0.citations"
    )
    assert diagnostics_by_code[ValidationCode.EVIDENCE_OUTSIDE_SNAPSHOT].related_ids == (
        "e-unknown",
    )


def test_absent_revelation_horizon_is_not_conflated_with_discourse_order() -> None:
    report = validate_evidence_grounding(
        snapshot_eligible_evidence_ids=("e-1",),
        packet_ordered_evidence_ids=("e-1",),
        packet_records=(
            EvidenceBoundaryRecord(
                evidence_id="e-1",
                discourse_position=2,
                revelation_position=99,
            ),
        ),
        horizon_rejected_evidence_ids=(),
        horizon=SpoilerHorizonBoundary(maximum_discourse_position=2),
        factual_records=(),
    )
    codes = {diagnostic.code for diagnostic in report.diagnostics}

    assert ValidationCode.REVELATION_HORIZON_LEAK not in codes


def test_discourse_and_revelation_horizons_are_enforced_independently() -> None:
    discourse_only = validate_evidence_grounding(
        snapshot_eligible_evidence_ids=("e-1",),
        packet_ordered_evidence_ids=("e-1",),
        packet_records=(
            EvidenceBoundaryRecord(
                evidence_id="e-1",
                discourse_position=11,
                revelation_position=2,
            ),
        ),
        horizon_rejected_evidence_ids=(),
        horizon=SpoilerHorizonBoundary(
            maximum_discourse_position=10,
            maximum_revelation_position=20,
        ),
        factual_records=(),
    )
    discourse_codes = {item.code for item in discourse_only.diagnostics}
    assert ValidationCode.DISCOURSE_HORIZON_LEAK in discourse_codes
    assert ValidationCode.REVELATION_HORIZON_LEAK not in discourse_codes

    revelation_only = validate_evidence_grounding(
        snapshot_eligible_evidence_ids=("e-1",),
        packet_ordered_evidence_ids=("e-1",),
        packet_records=(
            EvidenceBoundaryRecord(
                evidence_id="e-1",
                discourse_position=2,
                revelation_position=2,
            ),
        ),
        horizon_rejected_evidence_ids=(),
        horizon=SpoilerHorizonBoundary(
            maximum_discourse_position=10,
            maximum_revelation_position=10,
        ),
        factual_records=(
            GroundedFactualRecord(
                record_id="assertion-1",
                citations=(
                    EvidenceCitationAssessment(
                        evidence_id="e-1",
                        support_status=GroundingSupportStatus.SUPPORTED,
                    ),
                ),
                proposition_revelation_position=11,
            ),
        ),
    )
    revelation_codes = {item.code for item in revelation_only.diagnostics}
    assert ValidationCode.REVELATION_HORIZON_LEAK in revelation_codes
    assert ValidationCode.DISCOURSE_HORIZON_LEAK not in revelation_codes


def test_draft_boundary_rejects_assertion_and_proposition_horizon_leaks() -> None:
    payload = json.loads(
        (ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text(encoding="utf-8")
    )
    graph = payload["instance_graph"]
    assertion_scope = graph["assertions"][0]["temporal_scope"]
    proposition_scope = graph["proposition_contents"][0]["temporal_content"]
    for scope in (assertion_scope, proposition_scope):
        scope["discourse_position"] = {"passage_order": 999_999}
        scope["revelation_position"] = {"revelation_order": 999_999}

    report = _phase1_c2_boundary(payload)
    paths_by_code = {
        code: tuple(item.path for item in report.diagnostics if item.code is code)
        for code in (
            ValidationCode.DISCOURSE_HORIZON_LEAK,
            ValidationCode.REVELATION_HORIZON_LEAK,
        )
    }

    assert not report.accepted
    for paths in paths_by_code.values():
        assert any(path.startswith("assertions.") for path in paths)
        assert any(path.startswith("proposition_contents.") for path in paths)


def test_draft_boundary_rejects_explicit_invalid_temporal_values() -> None:
    payload = json.loads(
        (ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text(encoding="utf-8")
    )
    graph = payload["instance_graph"]
    graph["entities"][0]["temporal_state"] = {
        "kind": "invalid",
        "reason": "explicit invalid sentinel",
    }
    graph["proposition_contents"][0]["temporal_content"]["validity_time"] = {
        "kind": "invalid",
        "reason": "explicit invalid sentinel",
    }
    graph["assertions"][0]["temporal_scope"]["story_time"] = {
        "kind": "invalid",
        "reason": "explicit invalid sentinel",
    }

    report = _phase1_c2_boundary(payload)
    invalid = tuple(
        item
        for item in report.diagnostics
        if item.code is ValidationCode.TEMPORAL_VALUE_INVALID
    )

    assert not report.accepted
    assert len(invalid) == 3
    assert {item.related_ids[0] for item in invalid} == {
        graph["entities"][0]["entity_id"],
        graph["proposition_contents"][0]["proposition_content_id"],
        graph["assertions"][0]["assertion_id"],
    }


def test_c1_lineage_requires_every_construction_decision_and_seal_before_reveal() -> None:
    valid = ConstructionLineageAudit(
        condition=ConditionName.C1_LLM_PRE,
        query_revealed_at=REVEAL,
        completed_at=REVEAL - timedelta(minutes=2),
        prequery_seal_completed_at=REVEAL - timedelta(minutes=1),
        construction_decision_timestamps=(REVEAL - timedelta(minutes=3),),
        sealed_semantic_ids=("entity-1",),
        output_semantic_ids=("entity-1",),
        seed_block=1,
    )
    assert validate_construction_lineage(valid).accepted

    invalid = ConstructionLineageAudit(
        condition=ConditionName.C1_LLM_PRE,
        query_revealed_at=REVEAL,
        completed_at=REVEAL + timedelta(minutes=2),
        prequery_seal_completed_at=REVEAL,
        construction_decision_timestamps=(REVEAL + timedelta(seconds=1),),
        sealed_semantic_ids=("entity-1",),
        output_semantic_ids=("entity-1", "entity-new"),
        seed_block=1,
    )
    codes = {item.code for item in validate_construction_lineage(invalid).diagnostics}
    assert ValidationCode.PREQUERY_SEAL_AFTER_REVEAL in codes
    assert ValidationCode.PREQUERY_DECISION_AFTER_REVEAL in codes
    assert ValidationCode.UNSEALED_LINEAGE_ID in codes


def test_c2_requires_empty_prequery_inventory_and_post_reveal_construction() -> None:
    valid = ConstructionLineageAudit(
        condition=ConditionName.C2_LLM_QUERY,
        query_revealed_at=REVEAL,
        request_created_at=REVEAL,
        completed_at=REVEAL + timedelta(minutes=1),
        prequery_inventory_recorded_at=REVEAL - timedelta(seconds=1),
        construction_decision_timestamps=(REVEAL + timedelta(seconds=1),),
        output_semantic_ids=("entity-local",),
        seed_block=1,
    )
    assert validate_construction_lineage(valid).accepted

    invalid = ConstructionLineageAudit(
        condition=ConditionName.C2_LLM_QUERY,
        query_revealed_at=REVEAL,
        request_created_at=REVEAL - timedelta(seconds=1),
        completed_at=REVEAL + timedelta(minutes=1),
        prequery_inventory_recorded_at=REVEAL - timedelta(seconds=1),
        prequery_inventory_ids=("hidden-ontology",),
        construction_decision_timestamps=(REVEAL - timedelta(microseconds=1),),
        seed_block=1,
    )
    codes = {item.code for item in validate_construction_lineage(invalid).diagnostics}
    assert ValidationCode.NONEMPTY_C2_PREQUERY_INVENTORY in codes
    assert ValidationCode.QUERY_REQUEST_BEFORE_REVEAL in codes
    assert ValidationCode.QUERY_DECISION_BEFORE_REVEAL in codes


def test_fixed_select_lineage_requires_same_seed_sealed_ids_and_no_construction() -> None:
    invalid = ConstructionLineageAudit(
        condition=ConditionName.A_FIXED_SELECT,
        query_revealed_at=REVEAL,
        request_created_at=REVEAL,
        completed_at=REVEAL + timedelta(minutes=1),
        prequery_seal_completed_at=REVEAL - timedelta(seconds=1),
        construction_decision_timestamps=(REVEAL + timedelta(seconds=1),),
        selection_decision_timestamps=(REVEAL + timedelta(seconds=1),),
        sealed_semantic_ids=("entity-1",),
        output_semantic_ids=("entity-new",),
        seed_block=2,
        sealed_seed_block=1,
    )
    codes = {item.code for item in validate_construction_lineage(invalid).diagnostics}
    assert ValidationCode.FIXED_SELECT_CAPABILITY in codes
    assert ValidationCode.UNSEALED_LINEAGE_ID in codes
    assert ValidationCode.CROSS_SEED_LINEAGE in codes


def test_fixed_select_wrapper_returns_diagnostics_instead_of_mutating_output() -> None:
    sealed = SealedOntologyInventory(
        seal_hash=HASH_A,
        seed_block=1,
        objects=(
            SemanticFingerprint(
                semantic_id="entity-1",
                semantic_kind="entity",
                semantic_hash=HASH_B,
            ),
        ),
    )
    output = FixedSelectOutputAudit(
        source_seal_hash=HASH_A,
        seed_block=1,
        selected_objects=(),
        operations=(
            ProposedOperation(
                operation_id="merge-1",
                capability=ConstructionOperator.MERGE,
                referenced_ids=("entity-1",),
                created_ids=("entity-merged",),
                decided_at=REVEAL,
            ),
        ),
    )
    report = validate_fixed_select(output, sealed)

    assert not report.accepted
    assert {item.code for item in report.diagnostics} == {ValidationCode.FIXED_SELECT_CAPABILITY}


def repair(attempt_suffix: str) -> RepairLineageMetadata:
    return RepairLineageMetadata(
        root_attempt_id="attempt-0",
        base_attempt_id="attempt-0",
        repair_attempt_id=f"attempt-{attempt_suffix}",
        semantic_request_hash=HASH_A,
        base_output_hash=HASH_B,
        validation_record_hash=HASH_C,
        diagnostic_codes=("invalid_evidence_id",),
    )


def test_only_one_repair_is_permitted_and_parent_must_exist() -> None:
    assert validate_single_repair_lineage((repair("1"),), known_attempt_ids=("attempt-0",)).accepted

    report = validate_single_repair_lineage(
        (repair("1"), repair("2")), known_attempt_ids=("different-attempt",)
    )
    codes = {item.code for item in report.diagnostics}
    assert ValidationCode.MULTIPLE_REPAIRS in codes
    assert ValidationCode.INVALID_REPAIR_PARENT in codes


def test_repair_preservation_allows_only_diagnosed_keyed_record_path() -> None:
    base = {
        "instance_graph": {
            "assertions": [
                {"assertion_id": "a-1", "evidence_ids": ["e-bad"], "confidence": 0.8},
                {"assertion_id": "a-2", "evidence_ids": ["e-2"], "confidence": 0.9},
            ]
        }
    }
    repaired = {
        "instance_graph": {
            "assertions": [
                {"assertion_id": "a-1", "evidence_ids": ["e-1"], "confidence": 0.8},
                {"assertion_id": "a-2", "evidence_ids": ["e-2"], "confidence": 0.9},
            ]
        }
    }
    allowed = ("instance_graph.assertions.a-1.evidence_ids",)
    assert validate_repair_preservation(
        base_draft=base,
        repaired_draft=repaired,
        diagnosed_paths=allowed,
    ).accepted

    repaired["instance_graph"]["assertions"][1]["confidence"] = 0.1
    report = validate_repair_preservation(
        base_draft=base,
        repaired_draft=repaired,
        diagnosed_paths=allowed,
    )
    assert not report.accepted
    assert report.diagnostics[0].code is ValidationCode.REPAIR_MUTATION_OUTSIDE_DIAGNOSTIC
