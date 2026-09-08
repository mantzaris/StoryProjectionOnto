"""CPU-only proposed adapter tests, including immutable retained model failure."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from story_projection_onto.semantic_generation import reconstruct, semantic_schema
from story_projection_onto.semantic_identifiers import (
    PREFIXES,
    identifier_audit,
    normalize_identifiers,
    reconstruct_typed,
    typed_identifier_schema,
)
from tests.unit.test_semantic_generation import authored_wire, execution, small_wire, source_fixture


@pytest.mark.parametrize("filename", ["c1_pre_output.json", "c2_query_output.json"])
def test_proposed_typed_adapter_copies_complete_authored_science(filename):
    wire, _ = authored_wire(filename)
    fixture = source_fixture()
    audit = identifier_audit(wire)
    counts = dict.fromkeys(PREFIXES, 0)
    renamed = {}
    for d in audit["declarations"]:
        counts[d["kind"]] += 1
        renamed[d["id"]] = PREFIXES[d["kind"]] + str(counts[d["kind"]])

    def typed(v):
        if isinstance(v, dict):
            return {k: typed(x) for k, x in v.items()}
        if isinstance(v, list):
            return [typed(x) for x in v]
        return renamed.get(v, v) if isinstance(v, str) else v

    generated = typed(wire)  # Authored fixture only, never model response repair.
    original = copy.deepcopy(generated)
    adapted = reconstruct_typed(
        generated, evidence=fixture.evidence, upper=fixture.upper_ontology, execution=execution()
    )
    assert generated == original
    assert (
        adapted.provenance["identifier_translation"]["source_hash"]
        == (adapted.provenance["generated_semantic_hash"])
    )
    assert len(adapted.draft.instance_graph.assertions) == len(wire["instance_graph"]["assertions"])
    assert adapted.provenance["scientific_validation"] == "not performed by adapter"


def test_cross_namespace_collision_is_resolved_only_by_typed_references():
    wire = small_wire()
    # Test-only authored collision: type ID equals entity ID, no untyped decision
    # reference to that ID. Preserve both records and every chosen endpoint.
    old = wire["local_schema"]["contextual_types"][0]["type_id"]
    new = wire["instance_graph"]["entities"][0]["entity_id"]
    wire["local_schema"]["contextual_types"][0]["type_id"] = new
    wire["instance_graph"]["entities"][0]["contextual_type_id"] = new
    for p in wire["local_schema"]["predicates"]:
        for field in ("domain_type_ids", "range_type_ids"):
            p[field] = [new if x == old else x for x in p[field]]
    original = copy.deepcopy(wire)
    result, receipt = normalize_identifiers(wire, ["relation-candidate-arrived-at"])
    assert wire == original
    assert len(receipt["canonical_ids_by_record_path"]) == len(
        set(receipt["canonical_ids_by_record_path"].values())
    )
    graph = result["instance_graph"]
    assert graph["entities"][0]["entity_id"] != graph["entities"][0]["contextual_type_id"]
    assert graph["assertions"][0]["subject_id"] == graph["entities"][0]["entity_id"]
    fixture = source_fixture()
    adapted = reconstruct(
        result,
        evidence=fixture.evidence,
        upper=fixture.upper_ontology,
        execution=execution(),
        small=True,
    )
    assert len(adapted.draft.instance_graph.entities) == 2
    # Only declared IDs/references changed; inverse administrative translation
    # recovers the exact original payload, including descriptions/confidences.
    reverse = {
        receipt["canonical_ids_by_record_path"][d["path"]]: d["id"] for d in receipt["declarations"]
    }

    def undo(v):
        if isinstance(v, dict):
            return {k: undo(x) for k, x in v.items()}
        if isinstance(v, list):
            return [undo(x) for x in v]
        return reverse.get(v, v) if isinstance(v, str) else v

    assert undo(result) == original


@pytest.mark.parametrize("identical", [True, False])
def test_same_namespace_duplicates_never_discarded_or_selected(identical):
    wire = small_wire()
    wire["instance_graph"]["entities"].append(copy.deepcopy(wire["instance_graph"]["entities"][0]))
    if not identical:
        wire["instance_graph"]["entities"][-1]["label"] = "a different authored record"
    with pytest.raises(ValueError, match="semantic choice"):
        normalize_identifiers(wire)


@pytest.mark.parametrize("filename", ["c1_pre_output.json", "c2_query_output.json"])
def test_authored_full_semantics_survive_id_translation(filename):
    wire, _ = authored_wire(filename)
    fixture = source_fixture()
    supplied = [m.candidate_id for e in fixture.evidence for m in e.mention_candidates]
    supplied += [m.candidate_id for e in fixture.evidence for m in e.event_candidates]
    result, receipt = normalize_identifiers(wire, supplied)
    assert receipt["normalizable"]
    draft = reconstruct(
        result, evidence=fixture.evidence, upper=fixture.upper_ontology, execution=execution()
    ).draft
    assert len(draft.instance_graph.assertions) == len(wire["instance_graph"]["assertions"])


def test_typed_schema_requires_distinct_kinds_and_preserves_content_fields():
    fixture = source_fixture()
    old = semantic_schema(fixture.evidence, fixture.upper_ontology, small=True)
    schema = typed_identifier_schema(old)
    Draft202012Validator.check_schema(schema)
    assert old != schema
    defs = schema["$defs"]
    entity = defs["Entity"]["properties"]
    assert Draft202012Validator(entity["entity_id"]).is_valid("nE1")
    assert not Draft202012Validator(entity["entity_id"]).is_valid("nA1")
    assert not Draft202012Validator(entity["contextual_type_id"]).is_valid("nE1")
    assert entity["description"] == old["$defs"]["Entity"]["properties"]["description"]
    role = defs["RoleBinding"]["properties"]["role"]
    assert Draft202012Validator(role).is_valid("r" * 192)
    assert not Draft202012Validator(role).is_valid("r" * 193)
    assert not Draft202012Validator(role).is_valid("")
    for branch in defs["QualifiedAssertion"]["anyOf"]:
        assert Draft202012Validator(branch["properties"]["assertion_id"]).is_valid("nA1")
        assert not Draft202012Validator(branch["properties"]["assertion_id"]).is_valid("nE1")


def test_retained_first_response_has_three_unresolvable_decision_targets():
    root = Path(__file__).resolve().parents[2]
    paths = list(
        (root / "artifacts/restricted").glob(
            "semantic-session-backup.*/artifacts/restricted/small-semantic-interface-validation-*"
            "/run-*/semantic-diagnostic-1/decoded.json"
        )
    )
    if not paths:
        pytest.skip("restricted retained response is not in a public checkout")
    wire = json.loads(paths[0].read_bytes())
    original = copy.deepcopy(wire)
    audit = identifier_audit(
        wire, ["event-candidate-arrival", "m-lio-01", "m-gate-01", "m-seal-01"]
    )
    assert not any(c["same_namespace"] for c in audit["collisions"])
    assert [r["path"] for r in audit["references"] if r["resolution"] == "ambiguous"] == [
        "/decisions/0/created_object_ids/0",
        "/decisions/0/created_object_ids/1",
        "/decisions/0/created_object_ids/2",
    ]
    with pytest.raises(ValueError, match="semantic choice"):
        normalize_identifiers(wire)
    assert wire == original
