from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.contracts import (  # noqa: E402
    ConstructionRequest,
    OntologyDraft,
    PreconstructionRequest,
)
from story_projection_onto.phase1_acceptance import (  # noqa: E402
    _condition_output_schema,
    phase1_acceptance_calls,
)
from story_projection_onto.scorer_only.acceptance_grounding import (  # noqa: E402
    SemanticSupportStatus,
    audit_acceptance_semantic_grounding,
)

FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "phase1"


def load(name: str) -> dict[str, object]:
    value = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def audit(output_name: str, request_name: str, *, preconstruction: bool):
    draft = OntologyDraft.model_validate(load(output_name))
    if preconstruction:
        request = PreconstructionRequest.model_validate(load(request_name))
        evidence = request.evidence
    else:
        request = ConstructionRequest.model_validate(load(request_name))
        evidence = request.packet.evidence
    return audit_acceptance_semantic_grounding(draft=draft, evidence=evidence)


@pytest.mark.parametrize(
    ("output_name", "request_name", "preconstruction"),
    (
        ("c1_pre_output.json", "c1_pre_request.json", True),
        ("c1_pre_output_2.json", "c1_pre_request.json", True),
        ("c2_query_output.json", "c2_query_request.json", False),
        ("c2_query_output_2.json", "c2_query_request.json", False),
        ("fixed_select_output.json", "fixed_select_request.json", False),
    ),
)
def test_known_answer_outputs_are_semantically_grounded(
    output_name: str,
    request_name: str,
    preconstruction: bool,
) -> None:
    result = audit(output_name, request_name, preconstruction=preconstruction)
    assert result.complete
    assert result.grounding_precision == 1
    assert result.unsupported_count == result.unknown_count == 0


def test_citation_presence_does_not_validate_a_false_story_clock() -> None:
    mutated = load("c1_pre_output.json")
    assertions = mutated["instance_graph"]["assertions"]
    assertions[0]["temporal_scope"]["story_time"] = {
        "kind": "point",
        "point": 4,
        "label": "passage four",
    }
    draft = OntologyDraft.model_validate(mutated)
    request = PreconstructionRequest.model_validate(load("c1_pre_request.json"))
    result = audit_acceptance_semantic_grounding(draft=draft, evidence=request.evidence)
    target = next(item for item in result.assessments if item.record_id == "assert-c1-chain")
    assert target.status is SemanticSupportStatus.UNSUPPORTED
    assert not result.complete


def test_report_citations_cannot_promote_bridge_safety_to_world_truth() -> None:
    mutated = load("c2_query_output.json")
    assertion = mutated["instance_graph"]["assertions"][1]
    assertion["epistemic_scope"] = None
    assertion["proposition_content_id"] = None
    assertion["narrative_commitment"] = "world_committed"
    draft = OntologyDraft.model_validate(mutated)
    request = ConstructionRequest.model_validate(load("c2_query_request.json"))
    result = audit_acceptance_semantic_grounding(
        draft=draft,
        evidence=request.packet.evidence,
    )
    target = next(item for item in result.assessments if item.record_id == "assert-c2-report")
    assert target.status is SemanticSupportStatus.UNSUPPORTED
    assert "world truth" in target.reason


def _empty_enums(value: object) -> list[str]:
    paths: list[str] = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, Mapping):
            if item.get("enum") == []:
                paths.append(path)
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return paths


def test_fixed_select_schema_forbids_invention_when_sealed_category_is_empty() -> None:
    fixed_call = next(call for call in phase1_acceptance_calls() if call.call_id == "fixed-01")
    base_schema = json.loads(
        (REPOSITORY_ROOT / "schemas/jsonschema/ontology_draft.schema.json").read_text(
            encoding="utf-8"
        )
    )
    fixture = load("fixed_select_request.json")
    assert fixture["fixed_ontology"]["instance_graph"]["proposition_contents"] == []
    constrained = _condition_output_schema(base_schema, call=fixed_call, fixture=fixture)
    definitions = constrained["$defs"]
    graph_properties = definitions["InstanceGraph"]["properties"]
    assertion_properties = definitions["QualifiedAssertion"]["properties"]
    assert graph_properties["proposition_contents"]["maxItems"] == 0
    assert assertion_properties["proposition_content_id"] == {
        "const": None,
        "type": "null",
    }
    assert assertion_properties["epistemic_scope"] == {"const": None, "type": "null"}
    assert _empty_enums(constrained) == []


def test_scorer_oracle_markers_are_absent_from_model_visible_requests() -> None:
    forbidden = (
        "phase1-known-answer-v1",
        "bridge_safe_content_reported_by_mara",
        "marked_seal_enabled_opening",
        "scorer_only_oracle_revision",
    )
    for path in (
        FIXTURE_ROOT / "c1_pre_request.json",
        FIXTURE_ROOT / "c2_query_request.json",
        FIXTURE_ROOT / "fixed_select_request.json",
        REPOSITORY_ROOT / "prompts" / "c1_pre" / "prompt_v1.md",
        REPOSITORY_ROOT / "prompts" / "c2_query" / "prompt_v1.md",
        REPOSITORY_ROOT / "prompts" / "fixed_select" / "prompt_v1.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert all(marker not in text for marker in forbidden)
