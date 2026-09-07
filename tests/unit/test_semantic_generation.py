"""Authored CPU examples: neither observed model results nor scorer targets."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from story_projection_onto.contracts import ConditionName, OntologyDraft, PreconstructionRequest
from story_projection_onto.semantic_generation import (
    ExecutionFacts,
    build_small_request,
    reconstruct,
    schema_guide,
    semantic_schema,
)

ROOT = Path(__file__).resolve().parents[2]
RETAINED = (
    ROOT
    / "artifacts/restricted/representation-backup.maHyBy/artifacts/restricted"
    / "small-representation-comparison-20260907/run-20260907T193455715813"
)
if not RETAINED.exists():
    RETAINED = (
        ROOT
        / "artifacts/restricted/small-representation-comparison-20260907"
        / "run-20260907T193455715813"
    )


def source_fixture():
    # Only public hand-authored development input, not any gold.
    from story_projection_onto.phase1_legacy_provenance import Phase1LegacyEvidenceProvenanceBridge

    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    resolved = bridge.resolve_call(
        call_id="c1-01",
        condition=ConditionName.C1_LLM_PRE,
        request_fixture="tests/fixtures/phase1/c1_pre_request.json",
    )
    raw = json.loads((ROOT / "tests/fixtures/phase1/c1_pre_request.json").read_text())
    raw["evidence"] = [e.model_dump(mode="json") for e in resolved.evidence]
    raw["sealed_horizon"] = {
        "horizon_id": "cpu-fixture-horizon",
        "max_discourse_position": {"passage_order": 7},
        "max_revelation_position": {"revelation_order": 7},
    }
    return PreconstructionRequest.model_validate(raw)


def execution():
    start = datetime(2026, 9, 7, tzinfo=UTC)
    return ExecutionFacts("a" * 64, "b" * 64, start, start + timedelta(seconds=3), 100, 200)


def authored_wire(filename="c1_pre_output.json"):
    """TEST-ONLY representation of existing authored canonical fixtures.

    It never accepts or repairs model output. Explicit absence/empty values come
    from the authored canonical record. Rename all local IDs bijectively.
    """
    raw = json.loads((ROOT / "tests/fixtures/phase1" / filename).read_text())
    canonical = OntologyDraft.model_validate(raw).model_dump(mode="json")
    local_ids = set()

    def collect(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if (
                    k
                    in {
                        "entity_id",
                        "event_id",
                        "assertion_id",
                        "predicate_id",
                        "type_id",
                        "decision_id",
                        "proposition_content_id",
                        "schema_id",
                    }
                    and x
                ):
                    local_ids.add(x)
                collect(x)
        elif isinstance(v, list):
            for x in v:
                collect(x)

    collect(canonical)
    ids = {k: f"n{i}" for i, k in enumerate(sorted(local_ids))}

    def clean(v):
        if isinstance(v, list):
            return [clean(x) for x in v]
        if not isinstance(v, dict):
            return ids.get(v, v) if isinstance(v, str) else v
        out = {k: clean(x) for k, x in v.items() if k not in {"content_hash", "schema_version"}}
        if "kind" in out:
            out = {k: x for k, x in out.items() if x is not None and x != []}
        return out

    renamed = clean(canonical)
    wire = copy.deepcopy(renamed)
    wire.pop("budget_accounting")
    for d in wire["decisions"]:
        d.pop("decided_at")
    for kind in ("assertions", "proposition_contents"):
        for a in wire["instance_graph"][kind]:
            a["form"] = "nary" if a["roles"] else "binary"
            for field in ("subject_id", "object_id") if a["roles"] else ("roles",):
                a.pop(field)
    for a in wire["instance_graph"]["assertions"]:
        a["provenance"] = [
            {k: p[k] for k in ("evidence_id", "confidence")} for p in a["provenance"]
        ]
    return wire, renamed


def small_wire():
    """Separate hand-authored capacity/example fixture, NEVER in the request."""
    time = {"kind": "unknown", "label": "dawn", "reason": "No numeric coordinate stated."}
    scope = {
        "story_time": time,
        "validity_time": {"kind": "unknown", "reason": "Duration unspecified."},
        "discourse_position": {"passage_order": 1, "sentence_order": 0, "token_order": 0},
        "revelation_position": {"revelation_order": 1, "label": None},
    }
    entities = []
    for identity, label, mention, typ in (
        ("n1", "Lio", "m-lio-01", "nt1"),
        ("n2", "North Gate", "m-gate-01", "nt2"),
    ):
        entities.append(
            {
                "entity_id": identity,
                "label": label,
                "supported_mention_candidate_ids": [mention],
                "aliases": [],
                "contextual_type_id": typ,
                "contextual_role": "arrival participant",
                "abstraction": "actor",
                "temporal_state": time,
                "uncertainty": "known",
                "confidence": 0.9,
                "evidence_ids": ["ev-01"],
                "description": label,
                "description_assertion_ids": ["na1"],
            }
        )
    return {
        "contextual_interpretation": "An observed arrival, without an invented duration.",
        "local_schema": {
            "schema_id": "ns",
            "abstraction": "actor",
            "contextual_types": [
                {
                    "type_id": typ,
                    "label": label,
                    "definition": label,
                    "parent_upper_type": parent,
                    "abstraction": "actor",
                    "evidence_ids": ["ev-01"],
                }
                for typ, label, parent in (("nt1", "Courier", "person"), ("nt2", "Gate", "place"))
            ],
            "predicates": [
                {
                    "predicate_id": "np1",
                    "label": "arrived at",
                    "definition": "An actor's observed arrival at a place.",
                    "arity": 2,
                    "domain_type_ids": ["nt1"],
                    "range_type_ids": ["nt2"],
                    "role_names": [],
                    "parent_upper_relation": "located_at",
                    "evidence_ids": ["ev-01"],
                }
            ],
        },
        "instance_graph": {
            "entities": entities,
            "events": [],
            "proposition_contents": [],
            "assertions": [
                {
                    "form": "binary",
                    "assertion_id": "na1",
                    "predicate_id": "np1",
                    "subject_id": "n1",
                    "object_id": "n2",
                    "proposition_content_id": None,
                    "direction": "forward",
                    "temporal_scope": scope,
                    "epistemic_scope": None,
                    "narrative_commitment": "world_committed",
                    "confidence": 0.9,
                    "evidence_ids": ["ev-01"],
                    "provenance": [{"evidence_id": "ev-01", "confidence": 0.9}],
                    "contextual_relevance": 1,
                    "why_matters": "Lio arrived at North Gate.",
                    "why_matters_evidence_ids": ["ev-01"],
                }
            ],
        },
        "decisions": [
            {
                "decision_id": "nd1",
                "operator": "schema_relation",
                "evidence_ids": ["ev-01"],
                "rationale": "Represent the explicit arrival relation.",
                "input_object_ids": ["relation-candidate-arrived-at"],
                "created_object_ids": ["np1"],
                "removed_object_ids": [],
            }
        ],
        "omissions": [],
        "uncertainty_and_abstentions": ["Intrinsic duration unspecified."],
    }


@pytest.mark.parametrize("name", ["c1_pre_output.json", "c1_pre_output_2.json"])
def test_every_authored_scientific_field_reconstructs(name):
    f = source_fixture()
    wire, expected = authored_wire(name)
    result = reconstruct(wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution())
    actual = result.draft.model_dump(mode="json")

    def science(v):
        if isinstance(v, list):
            return [science(x) for x in v]
        if not isinstance(v, dict):
            return v
        if "provenance_id" in v:
            return {k: v[k] for k in ("evidence_id", "confidence")}
        return {
            k: science(x)
            for k, x in v.items()
            if k not in {"content_hash", "schema_version", "decided_at", "budget_accounting"}
            and x is not None
            and x != []
        }

    assert science(actual) == science(expected)
    assert result.draft.budget_accounting.output_tokens == 200
    assert all(d.decided_at == execution().generation_completed_at for d in result.draft.decisions)
    assert result.provenance["scientific_validation"] == "not performed by adapter"
    assert (
        reconstruct(wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution())
        == result
    )


@pytest.mark.parametrize(
    "time",
    [
        {"kind": "point", "point": 3},
        {"kind": "interval", "start": 2, "end": 4},
        {"kind": "interval", "start": 2},
        {"kind": "interval", "end": 4},
        {"kind": "relative", "anchor_id": "n2", "relation": "before"},
        {
            "kind": "partial_order",
            "partial_order": [{"left_id": "n1", "right_id": "n2", "relation": "before"}],
        },
        {"kind": "unknown", "reason": "Not given"},
        {"kind": "not_applicable"},
        {"kind": "horizon_withheld", "reason": "Beyond horizon"},
        {"kind": "invalid", "reason": "Contradictory source"},
    ],
)
def test_all_temporal_alternatives_are_lossless(time):
    f = source_fixture()
    wire = small_wire()
    wire["instance_graph"]["assertions"][0]["temporal_scope"]["validity_time"] = time
    actual = reconstruct(
        wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution(), small=True
    ).draft
    value = actual.instance_graph.assertions[0].temporal_scope.validity_time.model_dump(mode="json")
    for key, expected in time.items():
        if key != "partial_order":
            assert value[key] == expected
    if "partial_order" in time:
        assert (
            actual.instance_graph.assertions[0]
            .temporal_scope.validity_time.partial_order[0]
            .relation
            == "before"
        )


@pytest.mark.parametrize(
    "change",
    [
        "endpoint",
        "citation",
        "upper",
        "time_mix",
        "time_missing",
        "empty",
        "admin",
        "confidence",
        "wrong_local_id",
    ],
)
def test_schema_catches_observed_and_critical_missing_fields(change):
    f = source_fixture()
    wire = small_wire()
    a = wire["instance_graph"]["assertions"][0]
    if change == "endpoint":
        a.pop("object_id")
    elif change == "citation":
        a["evidence_ids"] = ["invented"]
    elif change == "upper":
        wire["local_schema"]["predicates"][0]["parent_upper_relation"] = "carries"
    elif change == "time_mix":
        a["temporal_scope"]["story_time"] = {"kind": "point", "point": 0, "partial_order": []}
    elif change == "time_missing":
        a["temporal_scope"]["story_time"] = {"kind": "point", "label": "dawn"}
    elif change == "empty":
        wire["instance_graph"].update(entities=[], assertions=[])
    elif change == "admin":
        wire["decisions"][0]["decided_at"] = ","
    elif change == "confidence":
        a.pop("confidence")
    elif change == "wrong_local_id":
        a["object_id"] = "ev-01"
    original = copy.deepcopy(wire)
    with pytest.raises(ValidationError):
        reconstruct(
            wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution(), small=True
        )
    assert wire == original  # never fill missing semantics or repair model response


def test_general_abstention_stays_representable():
    f = source_fixture()
    wire = small_wire()
    wire["instance_graph"] = dict(entities=[], events=[], assertions=[], proposition_contents=[])
    wire["decisions"] = []
    wire["local_schema"].update(contextual_types=[], predicates=[])
    assert reconstruct(wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution())
    with pytest.raises(ValidationError):
        reconstruct(
            wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution(), small=True
        )


def test_post_generation_checks_still_reject_semantic_reference_error():
    from story_projection_onto.validate import validate_draft_structure

    f = source_fixture()
    wire = small_wire()
    wire["instance_graph"]["assertions"][0]["object_id"] = "nMissing"
    out = reconstruct(
        wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution()
    ).draft
    with pytest.raises(ValueError):
        validate_draft_structure(
            draft=out,
            evidence=f.evidence,
            upper_ontology=f.upper_ontology,
            horizon=f.sealed_horizon,
            budgets=f.budgets,
            capabilities=f.capabilities,
        ).raise_for_errors()


def test_guide_describes_actual_alternatives_and_metadata_boundary():
    f = source_fixture()
    guide = schema_guide(semantic_schema(f.evidence, f.upper_ontology, small=True))
    for field in (
        "subject_id",
        "object_id",
        "roles",
        "holder_relative_time",
        "parent_upper_relation",
        "proposition_content_id",
        "confidence",
        "'unknown'",
        "'not_applicable'",
    ):
        assert field in guide
    for field in ("decided_at", "schema_version", "content_hash", "budget_accounting"):
        assert field not in guide


def test_exact_retained_responses_are_never_repaired():
    if not RETAINED.exists():
        pytest.skip("restricted retained A/B/C records not on this machine")
    f = PreconstructionRequest.model_validate_json(
        (RETAINED / "small-evidence-only-request.json").read_bytes()
    )
    schema = semantic_schema(f.evidence, f.upper_ontology, small=True)
    for n in (1, 2, 3):
        path = RETAINED / f"representation-diagnostic-{n}/decoded.json"
        before = path.read_bytes()
        value = json.loads(before)
        assert list(Draft202012Validator(schema).iter_errors(value))
        assert path.read_bytes() == before
    a = json.loads((RETAINED / "representation-diagnostic-1/decoded.json").read_bytes())
    b = json.loads((RETAINED / "representation-diagnostic-2/decoded.json").read_bytes())
    for value in (a, b):
        for assertion in value["instance_graph"]["assertions"]:
            assert (
                not assertion.get("subject_id")
                and not assertion.get("object_id")
                and not assertion.get("roles")
            )
            for key in ("story_time", "validity_time"):
                time = assertion["temporal_scope"][key]
                assert list(
                    Draft202012Validator(
                        {"$defs": schema["$defs"], "$ref": "#/$defs/SemanticTime"}
                    ).iter_errors(time)
                )


def test_small_request_uses_existing_transport_and_exact_packing():
    from dataclasses import replace

    from tests.unit.test_fallback_acceptance import FakeTokenizer, fallback_tokenizer_manifest

    f = source_fixture()
    f = f.model_copy(update={"evidence": f.evidence[:1]})
    request = build_small_request(f, FakeTokenizer(), fallback_tokenizer_manifest())
    assert request.stream_response and request.canonical_output_schema is None
    assert request.wire_payload()["guided_json"] == semantic_schema(
        f.evidence, f.upper_ontology, small=True
    )
    assert json.loads(request.messages[1].content)["evidence_snapshot"] == [
        e.model_dump(mode="json") for e in f.evidence
    ]
    assert request.decoding.maximum_model_tokens == 12288
    with pytest.raises(ValueError):
        replace(request, rendered_input_token_count=99999)


def test_actual_client_decoder_delivers_semantics_to_adapter():
    from story_projection_onto.gpu_runtime import VLLMGuidedJSONClient
    from tests.unit.test_fallback_acceptance import FakeTokenizer, fallback_tokenizer_manifest

    f = source_fixture()
    f = f.model_copy(update={"evidence": f.evidence[:1]})
    request = build_small_request(f, FakeTokenizer(), fallback_tokenizer_manifest())
    wire = small_wire()
    envelope = {
        "choices": [{"message": {"content": json.dumps(wire)}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": request.rendered_input_token_count,
            "completion_tokens": 200,
            "total_tokens": request.rendered_input_token_count + 200,
        },
    }
    event = {
        "id": "cpu-fixture",
        "choices": [
            {
                "index": 0,
                "delta": {"content": envelope["choices"][0]["message"]["content"]},
                "finish_reason": "stop",
            }
        ],
        "usage": envelope["usage"],
    }
    body = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
    result = VLLMGuidedJSONClient._decode_generation_response(
        request, 200, body, {"content-type": "text/event-stream"}
    )
    assert result.parsed_object == wire
    out = reconstruct(
        result.parsed_object,
        evidence=f.evidence,
        upper=f.upper_ontology,
        execution=execution(),
        small=True,
    )
    assert (
        out.draft.instance_graph.assertions[0].object_id
        == wire["instance_graph"]["assertions"][0]["object_id"]
    )


def test_canonical_holder_commitment_remains_a_post_generation_gate():
    f = source_fixture()
    wire, _ = authored_wire("c1_pre_output_2.json")
    a = next(a for a in wire["instance_graph"]["assertions"] if a["epistemic_scope"])
    a["narrative_commitment"] = "world_committed"
    Draft202012Validator(semantic_schema(f.evidence, f.upper_ontology)).validate(wire)
    with pytest.raises(ValueError, match="promoted to world truth"):
        reconstruct(wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution())


def test_unordered_intrinsic_bounds_are_not_accepted_by_structural_gate():
    from story_projection_onto.validate import validate_draft_structure

    f = source_fixture()
    wire = small_wire()
    wire["instance_graph"]["assertions"][0]["temporal_scope"]["validity_time"] = {
        "kind": "interval",
        "start": 4,
        "end": 2,
    }
    draft = reconstruct(
        wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution()
    ).draft
    report = validate_draft_structure(
        draft=draft,
        evidence=f.evidence,
        upper_ontology=f.upper_ontology,
        horizon=f.sealed_horizon,
        budgets=f.budgets,
        capabilities=f.capabilities,
    )
    assert any(d.code.value == "temporal_interval_invalid" for d in report.diagnostics)


def test_schema_field_inventory_cannot_silently_drop_science():
    """A canonical contract change forces an explicit adapter/schema decision."""
    f = source_fixture()
    actual = semantic_schema(f.evidence, f.upper_ontology)
    canonical = OntologyDraft.model_json_schema()
    removed = {
        "BudgetAccounting": set(canonical["$defs"]["BudgetAccounting"]["properties"]),
        "OntologyDecision": {"decided_at"},
        "ProvenanceReference": {
            "provenance_id",
            "locator",
            "source_artifact_hash",
            "extraction_method",
        },
    }
    temporal = {"StoryTime", "ValidityTime", "HolderRelativeTime"}
    for name, node in canonical["$defs"].items():
        if "properties" not in node or name in temporal or name == "BudgetAccounting":
            continue
        expected = (
            set(node["properties"]) - {"schema_version", "content_hash"} - removed.get(name, set())
        )
        replacement = actual["$defs"][name]
        branches = replacement.get("anyOf", [replacement])
        observed = set().union(*(set(branch["properties"]) for branch in branches)) - {"form"}
        assert observed == expected, name


def test_runtime_never_substitutes_provenance_confidence():
    f = source_fixture()
    wire = small_wire()
    wire["instance_graph"]["assertions"][0]["provenance"][0]["confidence"] = 0.43
    draft = reconstruct(
        wire, evidence=f.evidence, upper=f.upper_ontology, execution=execution()
    ).draft
    p = draft.instance_graph.assertions[0].provenance[0]
    assert p.confidence == 0.43
    assert p.locator == f.evidence[0].provenance.locator
    assert p.source_artifact_hash == f.evidence[0].provenance.source_artifact_hash
    assert draft.instance_graph.assertions[0].confidence == 0.9


def test_controller_adapter_reuses_scientific_gate_and_binds_runtime_facts(monkeypatch):
    from types import SimpleNamespace

    from tests.unit.test_capacity_diagnostic_controller import driver

    source = source_fixture()
    calls = []
    facts = execution()
    result = SimpleNamespace(
        parsed_object=small_wire(),
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=200,
        request_hash=facts.request_hash,
        response_sha256=facts.response_hash,
        diagnostic_journal=SimpleNamespace(validate=lambda stage, operation: operation()),
    )

    def shared_gate(draft, received_result, fixture, oracle):
        calls.append((draft, received_result, fixture, oracle))
        return {"scientific_valid": False, "test_spy_only": True}

    monkeypatch.setattr(driver, "_validate_small_canonical_draft", shared_gate)
    out = driver.validate_named_semantic_diagnostic(result, source, "scorer-only-spy", facts)
    assert out["scientific_valid"] is False
    assert calls[0][1] is result and calls[0][2] is source
    assert calls[0][0].budget_accounting.input_tokens == 100
    result.response_sha256 = "c" * 64
    with pytest.raises(ValueError, match="bound response identity"):
        driver.validate_named_semantic_diagnostic(result, source, "scorer-only-spy", facts)
    assert len(calls) == 1
