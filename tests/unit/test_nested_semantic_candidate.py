"""Authored codec controls, NOT model outcomes or model-visible answers."""

import copy

import pytest
from jsonschema import ValidationError

from story_projection_onto.nested_semantic_candidate import (
    BINDING_FIELDS,
    collapse_candidate,
    expand_candidate,
    reconstruct_candidate,
)
from tests.unit.test_semantic_generation import execution, source_fixture
from tests.unit.test_small_retry_path import demanding_capacity_wire, typed_authored
from tests.unit.test_small_semantic_reconciliation import (
    binary_wire,
    event_wire,
    second_binary_wire,
)


def authored_candidate(wire):
    """TEST ONLY: explicit authored conversion; never applied to failed responses."""
    value = typed_authored(copy.deepcopy(wire))
    graph = value["instance_graph"]
    props = {p["proposition_content_id"]: p for p in graph.pop("proposition_contents")}
    graph["unasserted_contents"] = []
    for a in graph["assertions"]:
        pid = a.pop("proposition_content_id")
        a["content"] = (
            {k: v for k, v in props[pid].items() if k != "proposition_content_id"}
            if pid
            else {
                **{k: copy.deepcopy(a[k]) for k in BINDING_FIELDS if k in a},
                "temporal_content": copy.deepcopy(a["temporal_scope"]),
                "evidence_ids": copy.deepcopy(a["evidence_ids"]),
            }
        )
        for k in BINDING_FIELDS:
            a.pop(k, None)
        a["content_identity"] = {"kind": "shared", "key": "c" + pid[2:]} if pid else None
        a["temporal_scope"] = "content"
        if a["epistemic_scope"] is not None:
            a["epistemic_scope"].pop("proposition_content_id")
    return value


def round_trip(candidate, *, small=True):
    f = source_fixture()
    expanded = expand_candidate(candidate, evidence=f.evidence, upper=f.upper_ontology, small=small)
    assert collapse_candidate(expanded) == candidate
    adapted = reconstruct_candidate(
        candidate, evidence=f.evidence, upper=f.upper_ontology, execution=execution(), small=small
    )
    assert not adapted.provenance["production_adoption_authorized"]
    return expanded, adapted


@pytest.mark.parametrize("wire", [binary_wire(), event_wire(nary=True), second_binary_wire()])
def test_binary_nary_direct_round_trip_without_proposition_or_attribution(wire):
    candidate = authored_candidate(wire)
    original = copy.deepcopy(candidate)
    expanded, _ = round_trip(candidate)
    assert expanded.semantic_payload["instance_graph"]["proposition_contents"] == []
    assert candidate == original


@pytest.mark.parametrize("attitude", ["known", "believed", "reported", "denied", "uncertain"])
def test_all_attributed_forms_preserve_bindings_holder_time_evidence_and_judgments(attitude):
    c = authored_candidate(demanding_capacity_wire(attributed=True))
    a = c["instance_graph"]["assertions"][0]
    a["epistemic_scope"]["attitude"] = attitude
    a["temporal_scope"] = copy.deepcopy(a["content"]["temporal_content"])
    a["temporal_scope"]["validity_time"] = {"kind": "unknown", "reason": "Different assertion time"}
    expanded, adapted = round_trip(c)
    ca = expanded.semantic_payload["instance_graph"]["assertions"][0]
    prop = expanded.semantic_payload["instance_graph"]["proposition_contents"][0]
    assert ca["predicate_id"] == prop["predicate_id"] == a["content"]["predicate_id"]
    assert ca["epistemic_scope"]["attitude"] == attitude
    assert ca["temporal_scope"] != prop["temporal_content"]
    assert ca["confidence"] == a["confidence"]
    assert ca["provenance"] == a["provenance"]
    assert adapted.provenance["scientific_acceptance_claimed"] is False


@pytest.mark.parametrize(
    "time",
    [
        {"kind": "point", "point": 2},
        {"kind": "interval", "start": 1, "end": 3},
        {"kind": "interval", "start": 1},
        {"kind": "interval", "end": 3},
        {"kind": "unknown", "reason": "Undetermined"},
        {"kind": "not_applicable"},
        {"kind": "horizon_withheld", "reason": "Not disclosed"},
        {"kind": "invalid", "reason": "Conflicting evidence"},
        {"kind": "relative", "anchor_id": "nE1", "relation": "before"},
        {
            "kind": "partial_order",
            "partial_order": [{"left_id": "nE1", "relation": "before", "right_id": "nE2"}],
        },
    ],
)
def test_temporal_shapes_are_retained_not_inferred_or_scientifically_endorsed(time):
    c = authored_candidate(binary_wire())
    c["instance_graph"]["assertions"][0]["content"]["temporal_content"]["validity_time"] = time
    round_trip(c)


def test_explicit_shared_identity_only_no_equality_based_merging():
    c = authored_candidate(demanding_capacity_wire(attributed=True))
    a, b, _ = c["instance_graph"]["assertions"]
    b["content"] = copy.deepcopy(a["content"])
    b["content_identity"] = copy.deepcopy(a["content_identity"])
    expanded, _ = round_trip(c)
    assert len(expanded.semantic_payload["instance_graph"]["proposition_contents"]) == 2
    a["content_identity"] = b["content_identity"] = {"kind": "distinct"}
    expanded, _ = round_trip(c)
    assert len(expanded.semantic_payload["instance_graph"]["proposition_contents"]) == 3


def test_conflicting_explicit_identity_and_missing_address_fail_without_guessing():
    c = authored_candidate(demanding_capacity_wire(attributed=True))
    a, b, _ = c["instance_graph"]["assertions"]
    b["content_identity"] = a["content_identity"]
    with pytest.raises(ValueError, match="conflicting bodies"):
        round_trip(c)
    b["content_identity"] = {"kind": "distinct"}
    c["decisions"][0]["created_object_ids"] = [{"content_of_assertion": "nA999"}]
    with pytest.raises(ValueError, match="no declared"):
        round_trip(c)


def test_unasserted_content_and_explicit_decision_addresses_round_trip():
    c = authored_candidate(demanding_capacity_wire(attributed=True))
    body = copy.deepcopy(c["instance_graph"]["assertions"][0]["content"])
    c["instance_graph"]["unasserted_contents"] = [{"key": "c9", "content": body}]
    c["decisions"][0]["created_object_ids"] = [
        {"content_of_assertion": "nA1"},
        {"unasserted_content": "c9"},
    ]
    expanded, _ = round_trip(c)
    assert len(expanded.semantic_payload["instance_graph"]["proposition_contents"]) == 4


def test_relative_content_address_is_reversible_without_choosing_time():
    c = authored_candidate(demanding_capacity_wire(attributed=True))
    a = c["instance_graph"]["assertions"][0]
    a["content"]["temporal_content"]["story_time"] = {
        "kind": "relative",
        "anchor_id": {"content_of_assertion": "nA2"},
        "relation": "before",
    }
    round_trip(c)


def test_attributed_nary_content_lifts_roles_and_citations_exactly():
    c = authored_candidate(event_wire(nary=True))
    a = c["instance_graph"]["assertions"][0]
    a.update(
        epistemic_scope={
            "holder_id": "nE1",
            "attitude": "believed",
            "holder_relative_time": {"kind": "unknown", "reason": "Unspecified"},
            "evidence_ids": ["ev-01"],
        },
        content_identity={"kind": "distinct"},
        narrative_commitment="holder_attributed",
    )
    expanded, _ = round_trip(c)
    assert (
        expanded.semantic_payload["instance_graph"]["proposition_contents"][0]["roles"]
        == a["content"]["roles"]
    )
    # Structural capacity only: direct narration does not support this belief.


@pytest.mark.parametrize("defect", ["unsupported_time", "invented_holder", "reversed_roles"])
def test_reconstruction_does_not_accept_semantic_errors(defect):
    from story_projection_onto.scorer_only.small_diagnostic_checks import reconciled_component_audit

    c = authored_candidate(binary_wire())
    a = c["instance_graph"]["assertions"][0]
    if defect == "unsupported_time":
        a["content"]["temporal_content"]["validity_time"] = {"kind": "point", "point": 1}
    elif defect == "invented_holder":
        a["epistemic_scope"] = {
            "holder_id": "nE1",
            "attitude": "known",
            "holder_relative_time": {"kind": "unknown", "reason": "Unspecified"},
            "evidence_ids": ["ev-01"],
        }
        a["content_identity"] = {"kind": "distinct"}
        a["narrative_commitment"] = "holder_attributed"
    else:
        a["content"]["subject_id"], a["content"]["object_id"] = (
            a["content"]["object_id"],
            a["content"]["subject_id"],
        )
    _, adapted = round_trip(c)
    f = source_fixture()
    result = reconciled_component_audit(
        adapted.draft, f.model_copy(update={"evidence": f.evidence[:1]}), f.evidence
    )
    assert not result["all_checks_pass"]


def test_missing_binding_unknown_evidence_and_wrong_destination_fail():
    c = authored_candidate(binary_wire())
    for path, value in [("subject_id", "nR1"), ("evidence_ids", ["ev-unknown"])]:
        mutated = copy.deepcopy(c)
        mutated["instance_graph"]["assertions"][0]["content"][path] = value
        with pytest.raises(ValidationError):
            round_trip(mutated)
    del c["instance_graph"]["assertions"][0]["content"]["object_id"]
    with pytest.raises(ValidationError):
        round_trip(c)


def test_no_loss_of_direct_content_time_and_no_implicit_attribution():
    c = authored_candidate(binary_wire())
    a = c["instance_graph"]["assertions"][0]
    a["temporal_scope"] = copy.deepcopy(a["content"]["temporal_content"])
    a["temporal_scope"]["validity_time"] = {"kind": "point", "point": 1}
    with pytest.raises(ValueError, match="direct content must preserve"):
        round_trip(c)


def test_small_budget_is_combined_and_general_allows_abstention():
    c = authored_candidate(demanding_capacity_wire())
    c["instance_graph"]["entities"].append(copy.deepcopy(c["instance_graph"]["entities"][0]))
    with pytest.raises(ValidationError):
        round_trip(c)
    c = authored_candidate(binary_wire())
    c["instance_graph"]["entities"] = []
    c["instance_graph"]["assertions"] = []
    c["decisions"] = []
    c["uncertainty_and_abstentions"] = ["No supported answer."]
    round_trip(c, small=False)


def test_restricted_candidate_package_has_exact_bound_request_and_separate_controls():
    import hashlib
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    root = Path("artifacts/restricted/nested-semantic-candidate-cpu-v1")
    if not root.exists():
        pytest.skip("restricted CPU review package is intentionally not public")
    for record in json.loads((root / "MANIFEST.json").read_bytes())["files"]:
        data = (root / record["path"]).read_bytes()
        assert len(data) == record["size"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
    request = json.loads((root / "model-facing/request.json").read_bytes())
    packing = json.loads((root / "CAPACITY_AND_ROUND_TRIP.json").read_bytes())
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    sections = json.loads(request["messages"][1]["content"])
    assert set(sections) == {
        "evidence_snapshot",
        "upper_ontology",
        "sealed_horizon",
        "budgets",
        "capabilities",
    }
    assert request["guided_json"] == json.loads(
        (root / "model-facing/generation.schema.json").read_bytes()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "artifacts/restricted/pinned-tokenizer-cpu", local_files_only=True, trust_remote_code=False
    )
    actual = len(
        tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )
    assert actual == packing["template_inclusive_input_tokens"] == 6102
    assert actual + request["max_tokens"] == 12246
    assert request["max_tokens"] == 6144
    assert not packing["production_adoption_authorized"]
