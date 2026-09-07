import copy
import json
from pathlib import Path

import pytest

from story_projection_onto.contracts import OntologyDraft, canonical_json_schema
from story_projection_onto.output_wire import (
    OutputWireError,
    RecordTupleCodec,
    pack_input_tables,
    reference_aliases,
    sealed_record_copies,
    translate_references,
    unpack_input_tables,
)

ROOT = Path(__file__).resolve().parents[2]


def test_opaque_references_are_lossless_without_rewriting_prose():
    value = {
        "evidence_id": "opaque-evidence-001",
        "label": "opaque-evidence-001",
        "source_artifact_hash": "a" * 64,
        "story_time": {"start": 1},
    }
    mapping = reference_aliases(value)
    encoded = translate_references(value, mapping, decode=False)
    assert encoded["label"] == value["label"]
    assert encoded["evidence_id"].startswith("I")
    assert translate_references(encoded, mapping, decode=True) == value
    with pytest.raises(OutputWireError, match="unknown"):
        translate_references({"evidence_id": "I999"}, mapping, decode=True)


def test_sealed_copy_is_exact_and_cannot_override_semantics():
    draft = json.loads((ROOT / "tests/fixtures/phase1/c1_pre_output.json").read_bytes())
    copies = sealed_record_copies({"sealed_ontology": draft})
    codec = RecordTupleCodec(canonical_json_schema(OntologyDraft), copies)
    wire = codec.encode(draft)
    assert "sealed_id" in json.dumps(wire)
    assert codec.decode(wire) == draft
    assert OntologyDraft.model_validate(codec.decode(wire)) == OntologyDraft.model_validate(draft)
    schema = {"$ref": "#/$defs/Entity"}
    with pytest.raises(OutputWireError, match="overridden"):
        codec._convert({"sealed_id": "missing", "confidence": 1}, schema, decode=True)
    with pytest.raises(OutputWireError, match="unknown"):
        codec._convert({"sealed_id": "missing"}, schema, decode=True)


def test_reference_enum_translation_keeps_the_exact_closed_set():
    schema = {
        "type": "object",
        "properties": {"entity_id": {"type": "string", "enum": ["long-entity-id-001"]}},
        "required": ["entity_id"],
    }
    mapping = {"I0": "long-entity-id-001"}
    codec = RecordTupleCodec(schema, opaque_aliases=mapping)
    assert codec.decode(codec.encode({"entity_id": "I0"})) == {"entity_id": "I0"}
    with pytest.raises(OutputWireError, match="enum"):
        codec.encode({"entity_id": "I1"})
    assert schema["properties"]["entity_id"]["enum"] == ["long-entity-id-001"]


@pytest.mark.parametrize(
    "name",
    [
        "c1_pre_output",
        "c1_pre_output_2",
        "c2_query_output",
        "c2_query_output_2",
        "fixed_select_output",
    ],
)
def test_lossless_canonical_hash_and_references(name):
    original = json.loads((ROOT / f"tests/fixtures/phase1/{name}.json").read_bytes())
    codec = RecordTupleCodec(canonical_json_schema(OntologyDraft))
    decoded = codec.decode(codec.encode(original))
    assert OntologyDraft.model_validate(decoded) == OntologyDraft.model_validate(original)
    assert original == decoded


def test_reject_missing_unknown_and_wrong_type():
    codec = RecordTupleCodec(canonical_json_schema(OntologyDraft))
    original = json.loads((ROOT / "tests/fixtures/phase1/c1_pre_output.json").read_bytes())
    wire = codec.encode(original)
    for bad in ({}, {"draft": wire["draft"], "extra": 1}, {"draft": wire["draft"][:-1]}):
        with pytest.raises(OutputWireError):
            codec.decode(bad)
    malformed = copy.deepcopy(wire)
    malformed["draft"][-1]["invented_entity"] = "x"
    with pytest.raises(OutputWireError):
        codec.decode(malformed)


def test_preserve_invalid_semantics_for_unchanged_validator():
    codec = RecordTupleCodec(canonical_json_schema(OntologyDraft))
    original = json.loads((ROOT / "tests/fixtures/phase1/c1_pre_output.json").read_bytes())
    original["instance_graph"]["entities"][0]["confidence"] = 9
    decoded = codec.decode(codec.encode(original))
    assert decoded["instance_graph"]["entities"][0]["confidence"] == 9
    with pytest.raises(ValueError):
        OntologyDraft.model_validate(decoded)


def test_no_semantic_fields_removed_from_wire_schema():
    codec = RecordTupleCodec(canonical_json_schema(OntologyDraft))
    schema = codec.wire_schema()
    assert "QualifiedAssertion" in schema["$defs"]
    assert "epistemic_scope" in json.dumps(schema)
    assert "source_artifact_hash" in json.dumps(schema)
    assert "why_matters" in codec.legend()


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        ["o", 0],
        {"s": ["a", 2]},
        [-1, -0.25, None, True, {"number": -9}],
        {
            "evidence": [
                {"id": "e1", "text": "long repeated evidence"},
                {"id": "e2", "text": "long repeated evidence"},
            ],
            "time": {"unknown": None, "interval": [1, 5]},
            "belief": False,
        },
    ],
)
def test_input_tables_preserve_all_values_and_marker_literals(value):
    assert unpack_input_tables(pack_input_tables(value)) == value


@pytest.mark.parametrize(
    "bad",
    [
        {"names": [], "keys": [], "strings": [], "value": -1},
        {"names": ["a"], "keys": [[0, 0]], "strings": [], "value": [0, 1, 2]},
        {"names": ["a"], "keys": [[0]], "strings": [], "value": [0]},
        {"names": [], "keys": [], "strings": [], "value": [True]},
    ],
)
def test_input_tables_reject_corruption(bad):
    with pytest.raises(OutputWireError):
        unpack_input_tables(bad)
