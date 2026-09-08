"""CPU-only semantic clarification and bounded repair; no altered model records."""

import copy
from pathlib import Path

import pytest

from story_projection_onto.semantic_generation import build_clarified_small_request
from story_projection_onto.semantic_identifiers import build_typed_small_request
from tests.unit.test_capacity_diagnostic_controller import driver
from tests.unit.test_fallback_acceptance import FakeTokenizer, fallback_tokenizer_manifest
from tests.unit.test_semantic_generation import small_wire, source_fixture
from tests.unit.test_typed_small_live_path import assess

ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = (ROOT / "prompts/diagnostics/semantic_instruction_v2.md").read_text()


def requests():
    fixture = source_fixture()
    fixture = fixture.model_copy(update={"evidence": fixture.evidence[:1]})
    tokenizer, manifest = FakeTokenizer(), fallback_tokenizer_manifest()
    old = build_typed_small_request(fixture, tokenizer, manifest)
    new = build_clarified_small_request(fixture, tokenizer, manifest, INSTRUCTION)
    return old, new


def test_revision_changes_only_general_instruction_not_working_representation():
    old, new = requests()
    a, b = old.wire_payload(), new.wire_payload()
    assert a.pop("messages")[1:] == b.pop("messages")[1:]
    assert a == b
    assert "forward read predicate(subject, object)" in new.messages[0].content
    assert "inverse reads predicate(object, subject)" in new.messages[0].content
    assert "No particular one is mandatory" in new.messages[0].content
    assert "kind=unknown" in new.messages[0].content
    for forbidden in ("Lio", "North Gate", "copper seal", "pump", "point 1", "ev-01"):
        assert forbidden not in INSTRUCTION
    assert old.output_schema == new.output_schema
    assert old.opaque_reference_aliases == new.opaque_reference_aliases


def test_unknown_time_is_explicit_and_not_an_empty_or_not_applicable_substitute():
    _, request = requests()
    alternatives = request.output_schema["$defs"]["SemanticTime"]["anyOf"]
    unknown = next(x for x in alternatives if x["properties"]["kind"]["const"] == "unknown")
    assert "reason" in unknown["required"] and "point" not in unknown["properties"]
    result = assess(small_wire())
    assert result["evidence_supported_temporal_bounds"]
    assert result["endpoint_support"] == "supported"
    # Legacy sealed oracle remains unchanged pending explicit scope/clock decision.
    assert not result["legacy_audit_complete"] and not result["all_checks_pass"]


@pytest.mark.parametrize(
    "operator", ["include_exclude", "rare_preservation", "supported_description"]
)
def test_non_substantive_operators_never_satisfy_construction_requirement(operator):
    wire = copy.deepcopy(small_wire())
    wire["decisions"][0]["operator"] = operator
    result = assess(wire)
    assert result["required_nonempty_structure"] and result["structural_valid"]
    assert not result["substantive_operation_reported"]
    assert not result["all_checks_pass"]


def test_fact_free_contract_failure_can_be_repaired_without_oracle_feedback():
    _, request = requests()
    failure = {
        "stage": "schema_or_structural_validation",
        "message": "semantic grounding audit failed: DO_NOT_SEND_ORACLE_FACT",
    }
    diagnostics = [
        {"code": "substantive_decision_missing", "path": "decisions"},
        {"code": "event_type_incompatible", "path": "instance_graph.events.nV1.contextual_type_id"},
    ]
    repaired = driver.prepare_structural_semantic_retry(
        request, failure, FakeTokenizer(), contract_diagnostics=diagnostics
    )
    assert repaired is not None and repaired.request_hash != request.request_hash
    assert repaired.messages[1:] == request.messages[1:]
    assert repaired.output_schema == request.output_schema
    assert "DO_NOT_SEND_ORACLE_FACT" not in repaired.messages[0].content
    assert driver.prepare_structural_semantic_retry(request, failure, FakeTokenizer()) is None


def test_scientific_text_alone_cannot_be_used_as_repair_feedback():
    assert (
        driver.prepare_structural_semantic_retry(
            None,
            {"stage": "scientific_capability_validation", "message": "expected endpoint: secret"},
            None,
        )
        is None
    )


@pytest.mark.parametrize(
    "diagnostics",
    [
        [{"code": "expected_endpoint", "path": "decisions"}],
        [{"code": "substantive_decision_missing", "path": "decisions", "answer": "secret"}],
        [{"code": "event_type_incompatible", "path": "expected answer is secret"}],
        [{"code": "event_type_incompatible", "path": "decisions"}],
    ],
)
def test_repair_rejects_unrecognized_or_answer_bearing_feedback(diagnostics):
    assert (
        driver.prepare_structural_semantic_retry(
            None,
            {"stage": "schema_or_structural_validation", "message": "secret"},
            None,
            contract_diagnostics=diagnostics,
        )
        is None
    )


def test_two_preselected_examples_then_one_evidence_supported_repair():
    pending = ("semantic-first-repair", "failed-first-attempt")
    assert driver.next_small_semantic_step("semantic-first", pending) == ("semantic-second", None)
    assert driver.next_small_semantic_step("semantic-second", pending) == pending
    assert driver.next_small_semantic_step("semantic-first-repair", pending) == (None, None)
    assert driver.next_small_semantic_step("semantic-second", None) == (None, None)
    # CPU preparation has not unlocked/reset the exhausted previous session.
    with pytest.raises(ValueError, match="start/attempt limit"):
        driver.semantic_session_admit(6025.436171, 1, 1, starting=True, seconds=100)
