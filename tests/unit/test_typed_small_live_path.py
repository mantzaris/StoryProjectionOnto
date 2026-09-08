"""CPU validation of the actual opt-in small diagnostic path, no GPU calls."""

import copy

from story_projection_onto.scorer_only.small_diagnostic_checks import component_audit
from story_projection_onto.semantic_generation import reconstruct
from story_projection_onto.semantic_identifiers import build_typed_small_request
from tests.unit.test_fallback_acceptance import FakeTokenizer, fallback_tokenizer_manifest
from tests.unit.test_semantic_generation import execution, small_wire, source_fixture


def test_typed_builder_keeps_whole_evidence_and_binds_decision_target_namespace():
    fixture = source_fixture()
    fixture = fixture.model_copy(update={"evidence": fixture.evidence[:1]})
    request = build_typed_small_request(fixture, FakeTokenizer(), fallback_tokenizer_manifest())
    payload = request.wire_payload()
    assert payload["stream"] is True
    assert payload["guided_json"] == request.output_schema
    assert "nD" in request.messages[0].content and "nA" in request.messages[0].content
    import json

    assert json.loads(request.messages[1].content)["evidence_snapshot"] == [
        e.model_dump(mode="json") for e in fixture.evidence
    ]
    targets = request.output_schema["$defs"]["OntologyDecision"]["properties"]["created_object_ids"]
    assert "nA" in str(targets) and "nE" in str(targets) and "nD" not in str(targets)


def assess(wire):
    fixture = source_fixture()
    draft = reconstruct(
        wire,
        evidence=fixture.evidence,
        upper=fixture.upper_ontology,
        execution=execution(),
        small=True,
    ).draft
    return component_audit(
        draft, fixture.model_copy(update={"evidence": fixture.evidence[:1]}), fixture.evidence
    )


def test_endpoint_support_does_not_depend_on_hidden_numeric_clock():
    result = assess(small_wire())
    assert result["reference_integrity"] and result["endpoint_correctness"]
    assert result["evidence_supported_temporal_bounds"]
    assert not result["legacy_audit_complete"] and not result["all_checks_pass"]
    assert any("story time" in a["reason"] for a in result["legacy_assessments"])


def test_invented_point_is_reported_even_if_legacy_oracle_passes():
    wire = small_wire()
    scope = wire["instance_graph"]["assertions"][0]["temporal_scope"]
    for name in ("story_time", "validity_time"):
        scope[name] = {"kind": "point", "point": 1}
    result = assess(wire)
    assert result["legacy_audit_complete"]
    assert not result["evidence_supported_temporal_bounds"] and not result["all_checks_pass"]
    assert len(result["temporal_support_issues"]) == 2


def test_wrong_endpoint_is_distinct_from_reference_integrity():
    wire = small_wire()
    assertion = wire["instance_graph"]["assertions"][0]
    assertion["subject_id"], assertion["object_id"] = (
        assertion["object_id"],
        assertion["subject_id"],
    )
    result = assess(wire)
    assert result["reference_integrity"] and not result["endpoint_correctness"]


def test_supported_description_is_not_substantive_construction():
    wire = copy.deepcopy(small_wire())
    wire["decisions"][0]["operator"] = "supported_description"
    result = assess(wire)
    assert not result["structural_valid"] and not result["construction_decision_grounding"]
