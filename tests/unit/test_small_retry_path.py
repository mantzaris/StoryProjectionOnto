"""CPU replays of real failures plus authored controls; no service/GPU access."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from story_projection_onto.contracts import PreconstructionRequest, canonical_sha256
from story_projection_onto.semantic_generation import (
    build_clarified_small_request,
    reconstruct,
    repair_epistemic_schema,
    semantic_schema,
)
from story_projection_onto.semantic_identifiers import IdentifierResolutionError, identifier_audit
from tests.unit.test_capacity_diagnostic_controller import driver
from tests.unit.test_fallback_acceptance import FakeTokenizer, fallback_tokenizer_manifest
from tests.unit.test_semantic_generation import execution, source_fixture
from tests.unit.test_small_semantic_reconciliation import (
    assess,
    binary_wire,
    event_wire,
    second_binary_wire,
)

RETAINED = Path(
    "artifacts/restricted/reconciled-session-backup.gZFFyU/artifacts/restricted/"
    "small-reconciled-semantic-validation-20260908/run-20260908T043820389279"
)


def retained_examples(tokenizer=None, manifest=None):
    if not RETAINED.exists():
        pytest.skip("restricted retained outputs intentionally not publicly released")
    tokenizer = tokenizer or FakeTokenizer()
    manifest = manifest or fallback_tokenizer_manifest()
    for i, kind in enumerate(("semantic-first", "semantic-second"), 1):
        fixture = PreconstructionRequest.model_validate_json(
            (RETAINED / f"fixture-{kind}.json").read_bytes()
        )
        request = replace(
            build_clarified_small_request(
                fixture,
                tokenizer,
                manifest,
                Path("prompts/diagnostics/semantic_instruction_v2.md").read_text(),
            ),
            request_id=kind,
        )
        attempt = RETAINED / f"small-reconciled-semantic-validation-20260908-diagnostic-{i}"
        yield kind, fixture, request, json.loads((attempt / "decoded.json").read_bytes()), attempt


def test_actual_missing_ids_reach_compact_repair_not_size_refusal():
    for _, fixture, request, previous, attempt in retained_examples():
        unchanged = canonical_sha256(previous)
        audits = []
        with pytest.raises(IdentifierResolutionError) as caught:
            driver.reconstruct_small_response(
                previous, fixture, execution(), record_audit=audits.append
            )
        failure = driver.semantic_failure_record(caught.value, "canonical_schema_validation")
        assert failure["full_identifier_audit"] == audits[0]
        old = json.loads((attempt / "failure.json").read_bytes())
        assert len(old["message"]) > 13000
        # Both legacy full-dump failure records and new typed exceptions take
        # the production formatter path, recomputing defects from original JSON.
        for error in (old, failure):
            preparation = {}
            retry = driver.prepare_structural_semantic_retry(
                request, error, FakeTokenizer(), previous_response=previous, preparation=preparation
            )
            assert retry is not None and preparation["stop_reason"] is None
            assert retry.messages[1] == request.messages[1]
            assert json.loads(retry.messages[-2].content) == previous
            feedback = json.loads(retry.messages[-1].content)
            observed_paths = {
                p
                for d in feedback["diagnostics"]
                if d["category"] == "undeclared_reference"
                for p in d["paths"]
            }
            assert observed_paths == {
                r["path"] for r in audits[0]["references"] if r["resolution"] == "unknown"
            }
            assert len(observed_paths) == 6
            assert "declarations" not in retry.messages[-1].content
            assert retry.wire_payload()["guided_json"] == retry.output_schema
            assert retry.decoding.maximum_output_tokens == 3584
            assert not retry.packing.truncation_applied
        assert canonical_sha256(previous) == unchanged
        assert not json.loads((attempt / "outcome.json").read_bytes())["accepted"]


def test_controller_simulated_transport_runs_both_then_one_repair_and_protects_shutdown():
    examples = list(retained_examples())
    requests = {x[0]: x[2] for x in examples}
    data = {x[0]: (x[1], x[3]) for x in examples}
    kind, pending, calls, shutdown = "semantic-first", None, [], []
    clock = 400.0
    try:
        while kind is not None:
            base_kind = kind.removesuffix("-repair")
            fixture, previous = data[base_kind]
            # Simulated transport returns the exact retained JSON, including on
            # the repair. This is NOT an authored successful model response.
            calls.append(kind)
            clock += 60
            try:
                driver.reconstruct_small_response(
                    previous, fixture, execution(), record_audit=lambda a: None
                )
            except IdentifierResolutionError as exc:
                failure = driver.semantic_failure_record(exc, "canonical_schema_validation")
            preparation = {}
            step = driver.advance_small_semantic(
                kind=kind,
                attempt_id=f"cpu-{len(calls)}",
                completed=len(calls),
                pending_repair=pending,
                request=requests[kind],
                failure=failure,
                previous_response=previous,
                tokenizer=FakeTokenizer(),
                contract_diagnostics=(),
                healthy=True,
                now_monotonic=clock,
                whole_deadline=1095,
                preparation=preparation,
            )
            if step.get("prepared_request") is not None:
                requests[step["pending"][0]] = step["prepared_request"]
            kind, pending = step["next"], step["pending"]
    finally:
        shutdown.append(clock)
    assert calls == ["semantic-first", "semantic-second", "semantic-first-repair"]
    assert step["stop_reason"] == "call_limit_reached"
    assert shutdown[0] + 60 < 1095


@pytest.mark.parametrize(
    "healthy,clock,completed,reason",
    [
        (False, 400, 1, "unsafe_service_state"),
        (True, 840, 2, "insufficient_time_with_shutdown_reserve"),
        (True, 400, 3, "call_limit_reached"),
    ],
)
def test_controller_explicit_stop_boundaries(healthy, clock, completed, reason):
    result = driver.advance_small_semantic(
        kind="semantic-second",
        attempt_id="cpu",
        completed=completed,
        pending_repair=None,
        request=None,
        failure=None,
        previous_response=None,
        tokenizer=None,
        contract_diagnostics=(),
        healthy=healthy,
        now_monotonic=clock,
        whole_deadline=1095,
        preparation={},
    )
    assert result["next"] is None and result["stop_reason"] == reason


def test_unavailable_repair_is_explicit_but_does_not_skip_second_baseline():
    for kind, expected in [("semantic-first", "semantic-second"), ("semantic-second", None)]:
        preparation = {}
        result = driver.advance_small_semantic(
            kind=kind,
            attempt_id="cpu",
            completed=1 if kind == "semantic-first" else 2,
            pending_repair=None,
            request=None,
            failure={"stage": "scientific_capability_validation", "message": "unresolved"},
            previous_response=None,
            tokenizer=None,
            contract_diagnostics=(),
            healthy=True,
            now_monotonic=400,
            whole_deadline=1095,
            preparation=preparation,
        )
        assert result["next"] == expected
        assert preparation["stop_reason"] == "no_actionable_contract_diagnostic"
        if expected is None:
            assert result["stop_reason"] == "no_actionable_contract_diagnostic"


def test_packing_failure_is_explicit_and_does_not_truncate_feedback():
    class Oversized(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return list(range(13000))

    _, _, request, previous, attempt = next(retained_examples())
    preparation = {}
    assert (
        driver.prepare_structural_semantic_retry(
            request,
            json.loads((attempt / "failure.json").read_bytes()),
            Oversized(),
            previous_response=previous,
            preparation=preparation,
        )
        is None
    )
    assert preparation["stop_reason"] == "repair_packing_failed"
    assert preparation["feedback"]["diagnostics"] and not preparation["truncation_applied"]


def attributed_wire():
    wire = binary_wire()
    a = wire["instance_graph"]["assertions"][0]
    a["proposition_content_id"] = "ncontent"
    a["epistemic_scope"] = dict(
        holder_id="n1",
        attitude="believed",
        proposition_content_id="ncontent",
        holder_relative_time={"kind": "unknown", "reason": "Unspecified."},
        evidence_ids=["ev-01"],
    )
    a["narrative_commitment"] = "holder_attributed"
    wire["instance_graph"]["proposition_contents"] = [
        dict(
            proposition_content_id="ncontent",
            predicate_id=a["predicate_id"],
            form="binary",
            subject_id=a["subject_id"],
            object_id=a["object_id"],
            temporal_content=copy.deepcopy(a["temporal_scope"]),
            evidence_ids=["ev-01"],
        )
    ]
    return wire


def demanding_capacity_wire(*, attributed=False):
    """Authored capacity stress only, never a model result or model-facing answer.

    Four nodes, three meaningful binary/event-role assertions. The attributed
    variant stresses three content records; attribution is deliberately NOT
    claimed grounded in the original one-passage evidence.
    """
    wire = event_wire()
    graph = wire["instance_graph"]
    artifact = copy.deepcopy(graph["entities"][1])
    artifact.update(
        entity_id="n3",
        label="Copper seal",
        description="Copper seal",
        supported_mention_candidate_ids=["m-seal-01"],
        contextual_type_id="nt4",
        description_assertion_ids=["na3"],
    )
    graph["entities"].append(artifact)
    typ = copy.deepcopy(wire["local_schema"]["contextual_types"][1])
    typ.update(type_id="nt4", label="Artifact", definition="Artifact", parent_upper_type="artifact")
    wire["local_schema"]["contextual_types"].append(typ)
    pred = copy.deepcopy(wire["local_schema"]["predicates"][0])
    pred.update(
        predicate_id="np3",
        label="carries",
        definition="A person carries an object.",
        role_names=["carrier", "theme"],
        parent_upper_relation="related_to",
        range_type_ids=["nt4"],
    )
    wire["local_schema"]["predicates"].append(pred)
    a = copy.deepcopy(graph["assertions"][0])
    a.update(
        assertion_id="na3",
        predicate_id="np3",
        subject_id="n1",
        object_id="n3",
        why_matters="Lio carries a copper seal.",
    )
    graph["assertions"].append(a)
    wire["decisions"][0]["created_object_ids"].append("np3")
    if attributed:
        for i, assertion in enumerate(graph["assertions"], 1):
            content_id = f"ncontent{i}"
            assertion.update(
                proposition_content_id=content_id,
                narrative_commitment="holder_attributed",
                epistemic_scope=dict(
                    holder_id="n1",
                    attitude="believed",
                    proposition_content_id=content_id,
                    holder_relative_time={"kind": "unknown", "reason": "Unspecified."},
                    evidence_ids=["ev-01"],
                ),
            )
            graph["proposition_contents"].append(
                dict(
                    proposition_content_id=content_id,
                    predicate_id=assertion["predicate_id"],
                    form="binary",
                    subject_id=assertion["subject_id"],
                    object_id=assertion["object_id"],
                    temporal_content=copy.deepcopy(assertion["temporal_scope"]),
                    evidence_ids=["ev-01"],
                )
            )
    return wire


def typed_authored(wire):
    """Administrative rename for authored fixtures only, never failed responses."""
    from story_projection_onto.semantic_identifiers import PREFIXES

    counts = dict.fromkeys(PREFIXES, 0)
    names = {}
    for declaration in identifier_audit(wire)["declarations"]:
        counts[declaration["kind"]] += 1
        names[declaration["id"]] = PREFIXES[declaration["kind"]] + str(counts[declaration["kind"]])

    def visit(v):
        if isinstance(v, dict):
            return {k: visit(x) for k, x in v.items()}
        if isinstance(v, list):
            return [visit(x) for x in v]
        return names.get(v, v) if isinstance(v, str) else v

    return visit(wire)


def test_repair_schema_catches_commitment_without_forcing_attribution_or_precision():
    fixture = source_fixture()
    old = semantic_schema(fixture.evidence, fixture.upper_ontology, small=True)
    schema = repair_epistemic_schema(old)
    validator = Draft202012Validator(schema)
    for wire in (binary_wire(), attributed_wire()):
        assert validator.is_valid(wire)
    wire = attributed_wire()
    wire["instance_graph"]["assertions"][0]["narrative_commitment"] = "world_committed"
    assert Draft202012Validator(old).is_valid(wire)
    assert not validator.is_valid(wire)
    wire = binary_wire()
    wire["instance_graph"]["assertions"][0]["proposition_content_id"] = "nmissing"
    assert not validator.is_valid(wire)


@pytest.mark.parametrize("attitude", ["known", "believed", "reported", "denied", "uncertain"])
def test_all_attitudes_require_declaration_after_grammar(attitude):
    fixture = source_fixture()
    schema = repair_epistemic_schema(
        semantic_schema(fixture.evidence, fixture.upper_ontology, small=True)
    )
    wire = attributed_wire()
    wire["instance_graph"]["assertions"][0]["epistemic_scope"]["attitude"] = attitude
    Draft202012Validator(schema).validate(wire)
    wire["instance_graph"]["proposition_contents"] = []
    # This dynamic referential constraint is deliberately not claimed encoded.
    Draft202012Validator(schema).validate(wire)
    from story_projection_onto.semantic_identifiers import normalize_identifiers

    with pytest.raises(IdentifierResolutionError):
        normalize_identifiers(wire, ["relation-candidate-arrived-at"])


def test_candidate_token_reallocation_cannot_route_an_ordinary_request():
    from story_projection_onto.gpu_runtime import RuntimeConfigurationError

    _, _, request, previous, attempt = next(retained_examples())
    retry = driver.prepare_structural_semantic_retry(
        request,
        json.loads((attempt / "failure.json").read_bytes()),
        FakeTokenizer(),
        previous_response=previous,
        preparation={},
    )
    assert retry is not None
    with pytest.raises(RuntimeConfigurationError, match="small diagnostic retry candidate only"):
        replace(retry, request_id="ordinary-c1-request")


def test_pinned_capacity_and_complete_adapter_for_demanding_authored_outputs():
    from transformers import AutoTokenizer

    from story_projection_onto.semantic_identifiers import (
        reconstruct_typed,
        typed_identifier_schema,
    )

    path = Path("artifacts/restricted/pinned-tokenizer-cpu")
    if not path.exists():
        pytest.skip("pinned tokenizer intentionally not distributed with tests")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    fixture = source_fixture()
    evidence = fixture.evidence[:1]
    schema = repair_epistemic_schema(
        typed_identifier_schema(semantic_schema(evidence, fixture.upper_ontology, small=True))
    )
    for attributed in (False, True):
        wire = typed_authored(demanding_capacity_wire(attributed=attributed))
        Draft202012Validator(schema).validate(wire)
        result = reconstruct_typed(
            wire, evidence=evidence, upper=fixture.upper_ontology, execution=execution(), small=True
        )
        assert len(result.draft.instance_graph.assertions) == 3
        assert len(result.draft.instance_graph.proposition_contents) == (3 if attributed else 0)
        for text in (json.dumps(wire, separators=(",", ":")), json.dumps(wire)):
            assert len(tokenizer.encode(text, add_special_tokens=False)) < 3584
        if not attributed:
            assert assess(demanding_capacity_wire())["all_checks_pass"]
        else:
            assert not assess(demanding_capacity_wire(attributed=True))["all_checks_pass"]


def test_invented_holder_with_complete_content_still_fails_science():
    result = assess(attributed_wire())
    assert result["reference_integrity"]
    assert (
        result["reconciled_science"]["components"]["discourse_revelation_epistemic"] == "rejected"
    )
    assert not result["all_checks_pass"]


@pytest.mark.parametrize(
    "parent,passes", [("entity", True), ("artifact", True), ("person", False), ("event", False)]
)
def test_supported_pump_artifact_compatibility_does_not_allow_wrong_types(parent, passes):
    source = source_fixture()
    evidence = tuple(e for e in source.evidence if e.evidence_id == "ev-03")
    fixture = source.model_copy(update={"evidence": evidence})
    wire = second_binary_wire()
    wire["local_schema"]["contextual_types"][1]["parent_upper_type"] = parent
    draft = reconstruct(
        wire, evidence=evidence, upper=source.upper_ontology, execution=execution(), small=True
    ).draft
    from story_projection_onto.scorer_only.small_diagnostic_checks import reconciled_component_audit

    result = reconciled_component_audit(draft, fixture, source.evidence)
    assert result["all_checks_pass"] is passes


def test_person_is_not_artifact_and_role_direction_remains_required():
    wire = binary_wire()
    wire["local_schema"]["contextual_types"][0]["parent_upper_type"] = "artifact"
    assert not assess(wire)["all_checks_pass"]
    wire = binary_wire()
    wire["instance_graph"]["assertions"][0]["direction"] = "inverse"
    assert assess(wire)["reconciled_science"]["components"]["endpoint_roles"] == "rejected"


def test_seal_artifact_is_allowed_as_carried_object_not_carrier():
    wire = binary_wire()
    wire["local_schema"]["contextual_types"][1].update(
        label="Seal", definition="Artifact", parent_upper_type="artifact"
    )
    wire["local_schema"]["predicates"][0].update(
        label="carries",
        definition="A person carries an object.",
        role_names=["carrier", "theme"],
        parent_upper_relation="related_to",
    )
    wire["instance_graph"]["entities"][1].update(
        label="Copper seal",
        description="Copper seal",
        supported_mention_candidate_ids=["m-seal-01"],
    )
    a = wire["instance_graph"]["assertions"][0]
    a["why_matters"] = "Lio carries a copper seal."
    assert assess(wire)["all_checks_pass"]
    a["subject_id"], a["object_id"] = a["object_id"], a["subject_id"]
    assert assess(wire)["reconciled_science"]["components"]["endpoint_roles"] == "rejected"


def test_identifier_diagnostic_deduplicates_paths_without_hiding_any():
    audit = identifier_audit(attributed_wire(), ["relation-candidate-arrived-at"])
    audit["references"].extend(copy.deepcopy(audit["references"]))
    from story_projection_onto.semantic_identifiers import identifier_diagnostics

    missing = {
        "path": "/instance_graph/assertions/0/proposition_content_id",
        "id": "nP9",
        "resolution": "unknown",
    }
    audit["references"].extend([missing, missing])
    diagnostic = identifier_diagnostics(audit)
    assert len(diagnostic) == 1 and len(diagnostic[0]["paths"]) == 1
