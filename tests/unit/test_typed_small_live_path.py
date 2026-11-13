"""CPU validation of the actual opt-in small diagnostic path, no GPU calls."""

import copy
import json
from pathlib import Path

import pytest

from story_projection_onto.scorer_only.small_diagnostic_checks import component_audit
from story_projection_onto.semantic_generation import reconstruct
from story_projection_onto.semantic_identifiers import (
    build_typed_small_request,
    identifier_audit,
    reconstruct_typed,
)
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
    assert result["reference_integrity"] and result["endpoint_support"] == "supported"
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
    assert result["reference_integrity"] and result["endpoint_support"] == "unknown"


def test_supported_description_is_not_substantive_construction():
    wire = copy.deepcopy(small_wire())
    wire["decisions"][0]["operator"] = "supported_description"
    result = assess(wire)
    assert result["structural_valid"] and result["required_nonempty_structure"]
    assert not result["substantive_operation_reported"]
    assert not result["construction_decision_grounding"]


def test_retained_typed_live_output_is_reference_valid_but_remains_scientifically_failed():
    root = Path(__file__).resolve().parents[2]
    run = root / (
        "artifacts/restricted/typed-session-backup.jXb6mu/artifacts/restricted/"
        "small-typed-semantic-validation-20260908/run-20260908T025837478254"
    )
    if not run.exists():
        pytest.skip("restricted actual response is not part of the public checkout")
    from story_projection_onto.contracts import PreconstructionRequest, canonical_sha256

    path = run / "small-typed-semantic-validation-20260908-diagnostic-1/decoded.json"
    wire = json.loads(path.read_bytes())
    assert (
        canonical_sha256(wire) == "d1e9ec0e58c619d3e75482c85818db1152b64aab9506ca6a2b20b52cebdfdc86"
    )
    original = copy.deepcopy(wire)
    fixture = PreconstructionRequest.model_validate(
        json.loads((run / "fixture-semantic-first.json").read_bytes())
    )
    audit = identifier_audit(
        wire, ["event-candidate-arrival", "m-lio-01", "m-gate-01", "m-seal-01"]
    )
    assert audit["normalizable"] and not audit["collisions"]
    assert len(audit["declarations"]) == 11
    adapted = reconstruct_typed(
        wire,
        evidence=fixture.evidence,
        upper=fixture.upper_ontology,
        execution=execution(),
        small=True,
    )
    result = component_audit(adapted.draft, fixture, source_fixture().evidence)
    assert result["reference_integrity"] and result["event_connectivity"]
    assert result["endpoint_support"] == "unknown" and not result["all_checks_pass"]
    assert result["structural_valid"] and result["required_nonempty_structure"]
    assert not result["construction_decision_grounding"]
    assert len(result["temporal_support_issues"]) == 10
    assert wire == original
