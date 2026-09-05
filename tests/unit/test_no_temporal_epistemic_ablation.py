from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionCapabilities,
    ConstructionRequest,
    NoTemporalEpistemicOntologyDraft,
    OntologyDraft,
    TemporalKind,
    canonical_json,
)
from story_projection_onto.llm import (
    ABLATION_QUALIFICATION_REASON,
    base_condition_output_schema,
    condition_output_model,
    normalize_no_temporal_epistemic_draft,
    render_condition_system_prompt,
)
from story_projection_onto.validate import (
    ValidationCode,
    validate_draft_structure,
)

ROOT = Path(__file__).resolve().parents[2]


def _raw_ablation_payload() -> dict[str, object]:
    payload = json.loads(
        (ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text(encoding="utf-8")
    )
    graph = payload["instance_graph"]
    graph.pop("proposition_contents")
    for entity in graph["entities"]:
        entity.pop("temporal_state")
    for event in graph["events"]:
        event.pop("occurrence_time")
    for assertion in graph["assertions"]:
        for field in (
            "proposition_content_id",
            "temporal_scope",
            "epistemic_scope",
            "narrative_commitment",
        ):
            assertion.pop(field, None)
    payload["decisions"] = []
    return payload


def _request() -> ConstructionRequest:
    return ConstructionRequest.model_validate_json(
        (ROOT / "tests/fixtures/phase1/c2_query_request.json").read_text(encoding="utf-8")
    )


def _strip_content_hashes(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, list):
        return [_strip_content_hashes(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_strip_content_hashes(child) for child in value)
    return value


def test_ablation_uses_a_distinct_schema_without_qualification_fields() -> None:
    assert (
        condition_output_model(ConditionName.A_NO_TEMPORAL_EPISTEMIC)
        is NoTemporalEpistemicOntologyDraft
    )
    assert condition_output_model(ConditionName.C2_LLM_QUERY) is OntologyDraft

    schema = base_condition_output_schema(ConditionName.A_NO_TEMPORAL_EPISTEMIC)
    serialized = canonical_json(schema)
    for forbidden_field in (
        "temporal_state",
        "occurrence_time",
        "temporal_scope",
        "temporal_content",
        "epistemic_scope",
        "proposition_content_id",
        "proposition_contents",
        "narrative_commitment",
        "holder_relative_time",
    ):
        assert f'"{forbidden_field}"' not in serialized
    accounting = schema["$defs"]["BudgetAccounting"]["properties"]
    assert accounting["input_tokens"] == {"const": 0, "type": "integer"}
    assert accounting["output_tokens"] == {"const": 0, "type": "integer"}


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        (
            "entities",
            "temporal_state",
            {"kind": "unknown", "reason": "model supplied time"},
        ),
        (
            "assertions",
            "epistemic_scope",
            {
                "holder_id": "entity-c2-alia",
                "attitude": "reported",
                "proposition_content_id": "prop-illicit",
                "holder_relative_time": {
                    "kind": "unknown",
                    "reason": "model supplied holder time",
                },
                "evidence_ids": ["ev-04"],
            },
        ),
    ),
)
def test_ablation_raw_parser_mechanically_rejects_removed_fields(
    section: str,
    field: str,
    value: object,
) -> None:
    payload = _raw_ablation_payload()
    payload["instance_graph"][section][0][field] = value
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        NoTemporalEpistemicOntologyDraft.model_validate(payload)


def test_ablation_normalization_adds_only_explicit_missing_sentinels() -> None:
    raw = NoTemporalEpistemicOntologyDraft.model_validate(_raw_ablation_payload())
    normalized = normalize_no_temporal_epistemic_draft(raw)

    assert normalized.instance_graph.proposition_contents == ()
    assert all(
        entity.temporal_state.kind is TemporalKind.UNKNOWN
        and entity.temporal_state.reason == ABLATION_QUALIFICATION_REASON
        for entity in normalized.instance_graph.entities
    )
    assert all(
        assertion.temporal_scope.story_time.kind is TemporalKind.UNKNOWN
        and assertion.temporal_scope.story_time.reason == ABLATION_QUALIFICATION_REASON
        and assertion.temporal_scope.validity_time.kind is TemporalKind.UNKNOWN
        and assertion.epistemic_scope is None
        and assertion.proposition_content_id is None
        and assertion.narrative_commitment.value == "unknown"
        for assertion in normalized.instance_graph.assertions
    )

    request = _request()
    report = validate_draft_structure(
        draft=normalized,
        upper_ontology=request.upper_ontology,
        evidence=request.packet.evidence,
        horizon=request.context.spoiler_horizon,
        budgets=request.budgets,
        capabilities=ConstructionCapabilities.active_without_temporal_epistemic(),
    )
    assert ValidationCode.ABLATION_QUALIFICATION_PRESENT not in {
        diagnostic.code for diagnostic in report.diagnostics
    }


def test_ablation_validator_rejects_a_supplied_temporal_value_after_normalization() -> None:
    raw = NoTemporalEpistemicOntologyDraft.model_validate(_raw_ablation_payload())
    normalized = normalize_no_temporal_epistemic_draft(raw)
    payload = _strip_content_hashes(normalized.model_dump(mode="python"))
    entity = payload["instance_graph"]["entities"][0]
    entity["temporal_state"] = {"kind": "point", "point": 7}
    tampered = OntologyDraft.model_validate(payload)
    request = _request()

    report = validate_draft_structure(
        draft=tampered,
        upper_ontology=request.upper_ontology,
        evidence=request.packet.evidence,
        horizon=request.context.spoiler_horizon,
        budgets=request.budgets,
        capabilities=ConstructionCapabilities.active_without_temporal_epistemic(),
    )
    assert ValidationCode.ABLATION_QUALIFICATION_PRESENT in {
        diagnostic.code for diagnostic in report.diagnostics
    }


def test_ablation_schema_is_not_an_in_place_mutation_of_c2_schema() -> None:
    c2_schema = base_condition_output_schema(ConditionName.C2_LLM_QUERY)
    before = copy.deepcopy(c2_schema)
    ablation_schema = base_condition_output_schema(
        ConditionName.A_NO_TEMPORAL_EPISTEMIC
    )
    assert c2_schema == before
    assert c2_schema != ablation_schema


def test_prompt_overlays_change_only_the_registered_capability_switch() -> None:
    c2 = render_condition_system_prompt(ROOT, ConditionName.C2_LLM_QUERY)
    assert render_condition_system_prompt(ROOT, ConditionName.A_NO_CONTEXT) == c2

    no_rare = render_condition_system_prompt(ROOT, ConditionName.A_NO_RARE_GUARD)
    assert "inspect every low-frequency item" in c2
    assert "inspect every low-frequency item" not in no_rare
    assert "rare-evidence\n    preservation" in c2
    assert "rare-evidence\n    preservation" not in no_rare
    restored = no_rare.replace(
        "9. ground every type",
        (
            "8. inspect every low-frequency item in the packet for answer necessity, state "
            "change,\n   identity consequences, temporal consequences, or causal reach. "
            "Preserve a one-off\n   fact when it is pivotal; never use frequency alone to prune "
            "it;\n9. ground every type"
        ),
    ).replace(
        "qualification, compression",
        "qualification, rare-evidence\n    preservation, compression",
    )
    assert restored == c2

    no_qualification = render_condition_system_prompt(
        ROOT, ConditionName.A_NO_TEMPORAL_EPISTEMIC
    )
    assert "removes temporal and epistemic construction" in no_qualification
    assert "Do not emit entity temporal state" in no_qualification
