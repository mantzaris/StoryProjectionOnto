"""Authored positive/negative controls, scorer-only; never model request material."""

import copy
import json
from pathlib import Path

import pytest

from story_projection_onto.scorer_only.small_diagnostic_checks import reconciled_component_audit
from story_projection_onto.semantic_generation import reconstruct
from tests.unit.test_semantic_generation import execution, small_wire, source_fixture


def binary_wire():
    wire = small_wire()
    for n in wire["instance_graph"]["entities"]:
        n["temporal_state"] = {"kind": "unknown", "reason": "Intrinsic duration unspecified."}
    return wire


def event_wire(*, nary=False):
    wire = binary_wire()
    graph = wire["instance_graph"]
    wire["local_schema"]["contextual_types"].append(
        {
            "type_id": "nt3",
            "label": "Arrival",
            "definition": "Arrival event",
            "parent_upper_type": "event",
            "abstraction": "event_role",
            "evidence_ids": ["ev-01"],
        }
    )
    template = wire["local_schema"]["predicates"][0]
    template.update(
        label="arrival agent",
        definition="Person acting as agent in an arrival event.",
        parent_upper_relation="participates_in",
        role_names=["agent", "event"],
        range_type_ids=["nt3"],
    )
    graph["events"] = [
        {
            "event_id": "nv1",
            "label": "Arrival",
            "contextual_type_id": "nt3",
            "occurrence_time": {"kind": "unknown", "label": "dawn", "reason": "No numeric clock."},
            "reification_reason": "Represent the arrival participants with separate roles.",
            "uncertainty": "known",
            "confidence": 0.9,
            "evidence_ids": ["ev-01"],
            "description": "Lio arrived at North Gate.",
            "description_assertion_ids": ["na1", "na2"],
        }
    ]
    graph["assertions"][0]["object_id"] = "nv1"
    location = copy.deepcopy(template)
    location.update(
        predicate_id="np2",
        label="arrival location",
        definition="Place serving as location of an arrival event.",
        role_names=["location", "event"],
        domain_type_ids=["nt2"],
    )
    wire["local_schema"]["predicates"].append(location)
    second = copy.deepcopy(graph["assertions"][0])
    second.update(assertion_id="na2", predicate_id="np2", subject_id="n2")
    graph["assertions"].append(second)
    graph["entities"][1]["description_assertion_ids"] = ["na2"]
    wire["decisions"][0]["created_object_ids"] = ["np1", "np2"]
    wire["decisions"].append(
        {
            "decision_id": "nd2",
            "operator": "event_reification",
            "evidence_ids": ["ev-01"],
            "rationale": "Reify the arrival to represent participant roles.",
            "input_object_ids": ["event-candidate-arrival"],
            "created_object_ids": ["nv1"],
            "removed_object_ids": [],
        }
    )
    if nary:
        template.update(
            arity=3,
            role_names=["agent", "location", "event"],
            domain_type_ids=[],
            range_type_ids=[],
        )
        template["definition"] = "Arrival event with explicit agent and location roles."
        wire["local_schema"]["predicates"] = [template]
        a = graph["assertions"][0]
        a.pop("subject_id")
        a.pop("object_id")
        a.update(
            form="nary",
            roles=[
                {"role": role, "object_id": ref, "evidence_ids": ["ev-01"]}
                for role, ref in (("agent", "n1"), ("location", "n2"), ("event", "nv1"))
            ],
        )
        graph["assertions"] = [a]
        for node in [*graph["entities"], *graph["events"]]:
            node["description_assertion_ids"] = ["na1"]
        wire["decisions"][0]["created_object_ids"] = ["np1"]
    return wire


def assess(wire):
    fixture = source_fixture()
    draft = reconstruct(
        wire,
        evidence=fixture.evidence,
        upper=fixture.upper_ontology,
        execution=execution(),
        small=True,
    ).draft
    return reconciled_component_audit(
        draft, fixture.model_copy(update={"evidence": fixture.evidence[:1]}), fixture.evidence
    )


def second_binary_wire():
    wire = binary_wire()
    for record in wire["local_schema"]["contextual_types"]:
        record["evidence_ids"] = ["ev-03"]
    wire["local_schema"]["contextual_types"][0].update(label="Mechanic", definition="Mechanic")
    wire["local_schema"]["contextual_types"][1].update(
        label="Pump", definition="Pump", parent_upper_type="entity"
    )
    wire["local_schema"]["predicates"][0].update(
        label="repaired",
        definition="A person repaired an object.",
        parent_upper_relation="related_to",
        evidence_ids=["ev-03"],
    )
    for n, label, mention in zip(
        wire["instance_graph"]["entities"],
        ("Ash", "river pump"),
        ("m-ash-mechanic", "m-pump-03"),
        strict=True,
    ):
        n.update(
            label=label,
            description=label,
            supported_mention_candidate_ids=[mention],
            evidence_ids=["ev-03"],
            contextual_role="repair participant",
        )
    a = wire["instance_graph"]["assertions"][0]
    a.update(
        evidence_ids=["ev-03"],
        provenance=[{"evidence_id": "ev-03", "confidence": 0.9}],
        why_matters="Mechanic Ash repaired the river pump.",
        why_matters_evidence_ids=["ev-03"],
    )
    a["temporal_scope"].update(
        story_time={"kind": "unknown", "reason": "No clock supplied."},
        discourse_position={"passage_order": 3, "sentence_order": 0, "token_order": 0},
        revelation_position={"revelation_order": 3, "label": None},
    )
    wire["decisions"][0].update(
        evidence_ids=["ev-03"],
        input_object_ids=["event-candidate-repair"],
        rationale="Represent the repair relation.",
    )
    wire["contextual_interpretation"] = "Explicit repair with unknown duration."
    return wire


def test_second_preselected_source_has_a_positive_control():
    fixture = source_fixture()
    evidence = tuple(e for e in fixture.evidence if e.evidence_id == "ev-03")
    draft = reconstruct(
        second_binary_wire(),
        evidence=evidence,
        upper=fixture.upper_ontology,
        execution=execution(),
        small=True,
    ).draft
    result = reconciled_component_audit(
        draft, fixture.model_copy(update={"evidence": evidence}), fixture.evidence
    )
    assert result["all_checks_pass"], result


def test_real_controller_adapter_uses_reconciled_gate_without_changing_legacy_route():
    from types import SimpleNamespace

    from tests.unit.test_capacity_diagnostic_controller import driver

    source = source_fixture()
    small = source.model_copy(update={"evidence": source.evidence[:1]})
    facts = execution()
    result = SimpleNamespace(
        parsed_object=binary_wire(),
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=200,
        request_hash=facts.request_hash,
        response_sha256=facts.response_hash,
        diagnostic_journal=SimpleNamespace(validate=lambda stage, operation: operation()),
    )
    checked = driver.validate_named_semantic_diagnostic(
        result, small, source.evidence, facts, reconciled=True
    )
    assert checked["scientific_valid"] and checked["grounding"]["all_supported"]
    with pytest.raises(ValueError, match="semantic grounding audit"):
        driver.validate_named_semantic_diagnostic(result, small, source.evidence, facts)


def test_unknown_definition_or_extra_claim_is_not_accepted_on_relation_keyword_alone():
    wire = binary_wire()
    wire["local_schema"]["predicates"][0]["definition"] += (
        " This causes the destruction of the city."
    )
    result = assess(wire)
    assert result["reconciled_science"]["components"]["schema_description_support"] == "unresolved"
    assert not result["all_checks_pass"]


@pytest.mark.parametrize("wire", [binary_wire, event_wire, lambda: event_wire(nary=True)])
def test_supported_binary_and_event_roles_are_equivalent(wire):
    result = assess(wire())
    assert result["all_checks_pass"], result
    facts = {
        atom
        for a in result["reconciled_science"]["assertion_checks"].values()
        for atom in a["endpoints_roles"]["atoms"]
    }
    assert facts == {"arrival.agent", "arrival.location"}
    assert result["reconciled_science"]["components"]["intrinsic_validity"] == "supported"


@pytest.mark.parametrize(
    "mutation",
    ["reverse", "event_destination", "wrong_carrier", "wrong_role", "contradictory_definition"],
)
def test_wrong_bindings_never_receive_positive_grounding(mutation):
    wire = event_wire() if mutation in {"wrong_role", "contradictory_definition"} else binary_wire()
    a = wire["instance_graph"]["assertions"][0]
    if mutation == "reverse":
        a["subject_id"], a["object_id"] = a["object_id"], a["subject_id"]
    elif mutation == "event_destination":
        wire = event_wire()
        wire["local_schema"]["predicates"][0].update(
            label="arrived at",
            definition="The arrival of a person at a place.",
            role_names=["arriver", "arrival_location"],
        )
    elif mutation == "wrong_carrier":
        # Carrier role assigned to an anchored place, not silently repaired.
        wire["local_schema"]["predicates"][0].update(
            label="carrying",
            definition="Carrier carrying an object.",
            parent_upper_relation="related_to",
            role_names=["carrier", "theme"],
        )
        a["subject_id"] = "n2"
    elif mutation == "wrong_role":
        wire["local_schema"]["predicates"][0]["role_names"] = ["location", "event"]
    else:
        wire["local_schema"]["predicates"][0]["definition"] = "The arrival of a person at a place."
    result = assess(wire)
    assert not result["all_checks_pass"]
    assert result["reconciled_science"]["components"]["endpoint_roles"] == "rejected"


@pytest.mark.parametrize("kind", ["point", "not_applicable"])
def test_unsupported_intrinsic_precision_and_na_are_not_unknown(kind):
    wire = binary_wire()
    wire["instance_graph"]["assertions"][0]["temporal_scope"]["validity_time"] = {
        "kind": kind,
        **({"point": 1} if kind == "point" else {}),
    }
    result = assess(wire)
    assert result["reconciled_science"]["components"]["intrinsic_validity"] == "rejected"
    assert not result["all_checks_pass"]


@pytest.mark.parametrize(
    "operator,targets",
    [
        ("supported_description", ["np1"]),
        ("event_reification", ["np1"]),
        ("schema_relation", ["n1"]),
    ],
)
def test_decision_must_report_actual_substantive_target(operator, targets):
    wire = binary_wire()
    wire["decisions"][0].update(operator=operator, created_object_ids=targets)
    result = assess(wire)
    assert result["reconciled_science"]["components"]["construction_decisions"] == "rejected"
    assert not result["all_checks_pass"]


def test_unknown_description_is_not_positive_support_or_false_by_default():
    wire = binary_wire()
    wire["instance_graph"]["entities"][0]["description"] = "Lio is an immortal king."
    result = assess(wire)
    assert result["reconciled_science"]["components"]["description_support"] == "unresolved"
    assert not result["all_checks_pass"]


def test_unknown_predicate_does_not_make_supported_prose_false():
    wire = binary_wire()
    wire["local_schema"]["predicates"][0].update(
        label="associated with", definition="An unspecified association."
    )
    result = assess(wire)["reconciled_science"]
    assert result["components"]["endpoint_roles"] == "unresolved"
    assert result["description_checks"]["n1"]["text"]["status"] == "supported"
    assert result["description_checks"]["n1"]["assertion_mapping"]["status"] == "unresolved"


def test_retained_graph_stays_failed_without_modification():
    from story_projection_onto.contracts import OntologyDraft, PreconstructionRequest
    from story_projection_onto.scorer_only.small_semantic_rules import audit_small_semantics

    source = Path(
        "artifacts/restricted/typed-session-backup.jXb6mu/artifacts/restricted/small-typed-semantic-validation-20260908/run-20260908T025837478254"
    )
    if not source.exists():
        pytest.skip("restricted retained output not released")
    file = source / "small-typed-semantic-validation-20260908-diagnostic-1/canonical.json"
    before = file.read_bytes()
    fixture = PreconstructionRequest.model_validate_json(
        (source / "fixture-semantic-first.json").read_bytes()
    )
    result = audit_small_semantics(OntologyDraft.model_validate_json(before), fixture)
    assert result["overall_status"] == "rejected"
    assert result["components"]["endpoint_roles"] == "rejected"
    assert result["components"]["intrinsic_validity"] == "rejected"
    assert result["components"]["construction_decisions"] == "rejected"
    assert file.read_bytes() == before
    # Re-evaluation has not rewritten the original recorded failure.
    assert not json.loads((source / "terminal.json").read_bytes())["outcomes"][0]["accepted"]
