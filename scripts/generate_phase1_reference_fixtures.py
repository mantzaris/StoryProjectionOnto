#!/usr/bin/env python3
"""Generate and token-audit the bounded Phase-1 reference outputs.

This script intentionally loads only the pinned tokenizer.  It never imports or
loads model weights, initializes CUDA, or contacts a network service.  Run it in
the frozen RunPod environment, for example::

    .venv/bin/python scripts/generate_phase1_reference_fixtures.py \
      --tokenizer-snapshot .cache/shared/hub/models--Qwen--Qwen3-14B-AWQ/\
snapshots/1a6fe1ecf891437a270cce11ad54d796c4f56ce0

The generated receipt binds each exact, key-sorted compact JSON payload to the
immutable tokenizer revision and tokenizer.json digest used for its token count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from story_projection_onto.conditions.base import (
    preontology_semantic_hash,
    sealed_semantic_ids,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    OntologyDraft,
    PreconstructionRequest,
    canonical_sha256,
)
from story_projection_onto.gpu_runtime import capture_tokenizer_manifest
from story_projection_onto.llm import DecodingPass, RepairLineageMetadata
from story_projection_onto.phase1_acceptance import (
    build_acceptance_request,
    phase1_acceptance_calls,
)
from story_projection_onto.validate import BoundaryValidationReport

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "phase1"
MODEL_MANIFEST = ROOT / "artifacts" / "public" / "manifests" / "model_snapshot.json"
MODEL_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
MODEL_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"
FULL_OUTPUT_CAP = 2_048
REPAIR_OUTPUT_CAP = 1_536


def compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def story_day(value: int) -> dict[str, object]:
    return {"kind": "point", "point": value, "label": f"day {value}"}


def unknown_story_time() -> dict[str, object]:
    return {"kind": "unknown", "reason": "Evidence gives no story-world coordinate."}


def unknown_validity_time() -> dict[str, object]:
    return {"kind": "unknown", "reason": "Evidence gives no validity interval."}


def temporal(discourse_position: int, *, world_day: int | None = None) -> dict[str, object]:
    return {
        "story_time": story_day(world_day) if world_day is not None else unknown_story_time(),
        "validity_time": (
            story_day(world_day) if world_day is not None else unknown_validity_time()
        ),
        "discourse_position": {"passage_order": discourse_position},
        "revelation_position": {"revelation_order": discourse_position},
    }


def provenance(prefix: str, evidence_id: str) -> dict[str, object]:
    return {
        "provenance_id": f"v-{prefix}-{evidence_id[3:]}",
        "evidence_id": evidence_id,
        "extraction_method": "hand fixture",
        "locator": evidence_id,
        "confidence": 1,
    }


def contextual_type(prefix: str, abstraction: str) -> dict[str, object]:
    return {
        "type_id": f"t-{prefix}",
        "label": "Role",
        "definition": "A context-specific narrative role.",
        "parent_upper_type": "entity",
        "abstraction": abstraction,
        "evidence_ids": ["ev-04"],
    }


def event_contextual_type(prefix: str, abstraction: str) -> dict[str, object]:
    return {
        "type_id": f"t-{prefix}-event",
        "label": "Narrative event",
        "definition": "A reified occurrence in the local event chain.",
        "parent_upper_type": "event",
        "abstraction": abstraction,
        "evidence_ids": ["ev-04"],
    }


def predicate(
    prefix: str,
    name: str,
    label: str,
    definition: str,
    arity: int,
    evidence_ids: list[str],
) -> dict[str, object]:
    parent_upper_relation = {
        "chain": "precedes",
        "credential-enabled": "causes",
        "status": "has_status",
    }.get(name, "related_to")
    return {
        "predicate_id": f"p-{prefix}-{name}",
        "label": label,
        "definition": definition,
        "arity": arity,
        "parent_upper_relation": parent_upper_relation,
        "evidence_ids": evidence_ids,
    }


def entity(
    prefix: str,
    name: str,
    label: str,
    mentions: list[str],
    role: str,
    abstraction: str,
    world_day: int | None,
    evidence_ids: list[str],
    description: str,
    support: list[str],
) -> dict[str, object]:
    return {
        "entity_id": f"ent-{prefix}-{name}",
        "label": label,
        "supported_mention_candidate_ids": mentions,
        "contextual_type_id": f"t-{prefix}",
        "contextual_role": role,
        "abstraction": abstraction,
        "temporal_state": (story_day(world_day) if world_day is not None else unknown_story_time()),
        "uncertainty": "known",
        "confidence": 1,
        "evidence_ids": evidence_ids,
        "description": description,
        "description_assertion_ids": support,
    }


def event(
    prefix: str,
    name: str,
    label: str,
    abstraction: str,
    description: str,
    support: list[str],
) -> dict[str, object]:
    return {
        "event_id": f"event-{prefix}-{name}",
        "label": label,
        "contextual_type_id": f"t-{prefix}-event",
        "occurrence_time": story_day(2),
        "reification_reason": "Needed for temporal roles.",
        "uncertainty": "known",
        "confidence": 1,
        "evidence_ids": ["ev-04"],
        "description": description,
        "description_assertion_ids": support,
    }


def assertion(
    prefix: str,
    name: str,
    predicate_name: str,
    when: int,
    evidence_ids: list[str],
    why: str,
    why_evidence_ids: list[str],
    *,
    subject: str | None = None,
    object_: str | None = None,
    roles: list[dict[str, object]] | None = None,
    proposition_content_id: str | None = None,
    epistemic_scope: dict[str, object] | None = None,
    commitment: str = "world_committed",
    world_day: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "assertion_id": f"assert-{prefix}-{name}",
        "predicate_id": f"p-{prefix}-{predicate_name}",
        "temporal_scope": temporal(when, world_day=world_day),
        "narrative_commitment": commitment,
        "confidence": 1,
        "evidence_ids": evidence_ids,
        "provenance": [provenance(f"{prefix}-{name}", item) for item in evidence_ids],
        "contextual_relevance": 1,
        "why_matters": why,
        "why_matters_evidence_ids": why_evidence_ids,
    }
    if subject is not None:
        value["subject_id"] = subject
        value["object_id"] = object_
    else:
        value["roles"] = roles
    if proposition_content_id is not None:
        value["proposition_content_id"] = proposition_content_id
        value["epistemic_scope"] = epistemic_scope
    return value


def decision(
    prefix: str,
    name: str,
    operator: str,
    when: str,
    evidence_ids: list[str],
    inputs: list[str],
    created: list[str],
    rationale: str,
) -> dict[str, object]:
    value: dict[str, object] = {
        "decision_id": f"d-{prefix}-{name}",
        "operator": operator,
        "evidence_ids": evidence_ids,
        "rationale": rationale,
        "decided_at": when,
    }
    if inputs:
        value["input_object_ids"] = inputs
    if created:
        value["created_object_ids"] = created
    return value


def build_full_draft(prefix: str) -> dict[str, object]:
    c1 = prefix == "c1"
    abstraction = "event_role" if c1 else "collective_causal_chain"
    chain_id = f"assert-{prefix}-chain"
    distinct_id = f"assert-{prefix}-distinct"
    enabled_id = f"assert-{prefix}-enabled"
    report_id = f"assert-{prefix}-report"
    delivery_id = f"event-{prefix}-delivery"
    opening_id = f"event-{prefix}-opening"
    lio_id = f"ent-{prefix}-lio"
    ash_id = f"ent-{prefix}-ash-mechanic"
    seal_id = f"ent-{prefix}-seal"
    mara_id = f"ent-{prefix}-mara"
    bridge_id = f"ent-{prefix}-bridge"
    safe_id = f"ent-{prefix}-safe"
    proposition_id = f"prop-{prefix}-bridge-safe"
    schema_id = f"s-{prefix}"
    type_id = f"t-{prefix}"

    predicates = [
        predicate(
            prefix,
            "chain",
            "credential chain",
            "Delivery precedes opening.",
            4,
            ["ev-04"],
        ),
        predicate(
            prefix,
            "credential-enabled",
            "enabled",
            "A rare mark enabled opening.",
            2,
            ["ev-04", "ev-06"],
        ),
        predicate(
            prefix,
            "status",
            "reported status",
            "Safety content attributed by a holder.",
            2,
            ["ev-05"],
        ),
    ]
    entities = [
        entity(
            prefix,
            "lio",
            "Lio",
            ["m-lio-04", "m-ash-courier"],
            "credential courier",
            abstraction,
            2,
            ["ev-02", "ev-04"],
            "Lio delivered the marked seal.",
            [chain_id],
        ),
        entity(
            prefix,
            "ash-mechanic",
            "Ash the mechanic",
            ["m-ash-mechanic"],
            "separate mechanic",
            abstraction,
            None,
            ["ev-03"],
            "A mechanic distinct from Lio.",
            [chain_id],
        ),
        entity(
            prefix,
            "seal",
            "Marked seal",
            ["m-seal-04", "m-mark-06"],
            "rare credential",
            abstraction,
            2,
            ["ev-04", "ev-06"],
            "Its rare mark enabled opening.",
            [chain_id, enabled_id],
        ),
        entity(
            prefix,
            "mara",
            "Mara",
            ["m-mara-05"],
            "report holder",
            abstraction,
            None,
            ["ev-05"],
            "Holder of the bridge report.",
            [report_id],
        ),
        entity(
            prefix,
            "bridge",
            "Bridge",
            ["m-bridge-05"],
            "reported infrastructure",
            abstraction,
            None,
            ["ev-05"],
            "Safe only inside Mara's report.",
            [report_id],
        ),
        entity(
            prefix,
            "safe",
            "Safe state",
            ["m-safe-05"],
            "reported state",
            abstraction,
            None,
            ["ev-05"],
            "Safety state inside Mara's report.",
            [report_id],
        ),
    ]
    events = [
        event(
            prefix,
            "delivery",
            "Seal delivery",
            abstraction,
            "Delivery preceded opening.",
            [chain_id],
        ),
        event(
            prefix,
            "opening",
            "Floodgate opening",
            abstraction,
            "Opening required the rare mark.",
            [chain_id, enabled_id],
        ),
    ]
    proposition = {
        "proposition_content_id": proposition_id,
        "predicate_id": f"p-{prefix}-status",
        "subject_id": bridge_id,
        "object_id": safe_id,
        "temporal_content": temporal(5),
        "evidence_ids": ["ev-05"],
    }
    chain_evidence_ids = ["ev-04"] if c1 else ["ev-03", "ev-04"]
    chain_roles = [
        {"role": "delivery", "object_id": delivery_id, "evidence_ids": ["ev-04"]},
        {"role": "opening", "object_id": opening_id, "evidence_ids": ["ev-04"]},
        {"role": "courier", "object_id": lio_id, "evidence_ids": ["ev-04"]},
        {"role": "seal", "object_id": seal_id, "evidence_ids": ["ev-04"]},
    ]
    predicates[0]["role_names"] = ["delivery", "opening", "courier", "seal"]
    assertions = [
        assertion(
            prefix,
            "chain",
            "chain",
            4,
            chain_evidence_ids,
            "Shows roles and temporal order.",
            ["ev-04"],
            roles=chain_roles,
            world_day=2,
        ),
        assertion(
            prefix,
            "enabled",
            "credential-enabled",
            6,
            ["ev-04", "ev-06"],
            "The rare mark explains opening.",
            ["ev-04", "ev-06"],
            subject=seal_id,
            object_=opening_id,
            world_day=2,
        ),
        assertion(
            prefix,
            "report",
            "status",
            5,
            ["ev-05"],
            "Does not assert bridge safety.",
            ["ev-05"],
            subject=bridge_id,
            object_=safe_id,
            proposition_content_id=proposition_id,
            epistemic_scope={
                "holder_id": mara_id,
                "attitude": "reported",
                "proposition_content_id": proposition_id,
                "holder_relative_time": {
                    "kind": "unknown",
                    "reason": "Evidence gives no holder-relative time.",
                },
                "evidence_ids": ["ev-05"],
            },
            commitment="holder_attributed",
        ),
    ]

    if c1:
        predicates = [item for item in predicates if item["predicate_id"] != "p-c1-status"]
        entities = [item for item in entities if item["entity_id"] in {lio_id, seal_id}]
        assertions = [item for item in assertions if item["assertion_id"] in {chain_id, enabled_id}]
        decisions = [
            decision(
                prefix,
                "merge",
                "merge",
                "2026-09-03T11:45:01Z",
                ["ev-02", "ev-04"],
                ["m-ash-courier", "m-lio-04"],
                [lio_id],
                "Merge the courier alias.",
            ),
            decision(
                prefix,
                "schema",
                "schema_relation",
                "2026-09-03T11:45:02Z",
                ["ev-04"],
                ["event-candidate-delivery"],
                [schema_id],
                "Create the local relation schema.",
            ),
            decision(
                prefix,
                "events",
                "event_reification",
                "2026-09-03T11:45:03Z",
                ["ev-04"],
                ["event-candidate-delivery", "event-candidate-opening"],
                [delivery_id, opening_id],
                "Reify both ordered events.",
            ),
            decision(
                prefix,
                "temporal",
                "temporal_qualification",
                "2026-09-03T11:45:04Z",
                ["ev-04"],
                [delivery_id, opening_id],
                [chain_id],
                "Qualify delivery before opening.",
            ),
            decision(
                prefix,
                "rare",
                "rare_preservation",
                "2026-09-03T11:45:05Z",
                ["ev-06"],
                ["m-mark-06"],
                [enabled_id],
                "Retain the rare enabling mark.",
            ),
        ]
        interpretation = "Query-blind identity, ordered events, and rare credential evidence."
    else:
        predicates = [
            item
            for item in predicates
            if item["predicate_id"] not in {"p-c2-distinct", "p-c2-credential-enabled"}
        ]
        assertions = [
            item for item in assertions if item["assertion_id"] not in {distinct_id, enabled_id}
        ]
        next(item for item in entities if item["entity_id"] == seal_id)[
            "description_assertion_ids"
        ] = [chain_id]
        c2_lio = next(item for item in entities if item["entity_id"] == lio_id)
        c2_lio["supported_mention_candidate_ids"] = ["m-ash-courier"]
        c2_lio["evidence_ids"] = ["ev-02"]
        c2_lio["temporal_state"] = unknown_story_time()
        c2_seal = next(item for item in entities if item["entity_id"] == seal_id)
        c2_seal["evidence_ids"] = ["ev-04"]
        c2_seal["supported_mention_candidate_ids"] = ["m-seal-04"]
        c2_seal["description"] = "Credential in the ordered chain."
        next(item for item in events if item["event_id"] == opening_id)[
            "description_assertion_ids"
        ] = [chain_id]
        decisions = [
            decision(
                prefix,
                "include",
                "include_exclude",
                "2026-09-03T12:00:11Z",
                ["ev-04"],
                [],
                [chain_id],
                "Include the credential chain.",
            ),
            decision(
                prefix,
                "split",
                "split",
                "2026-09-03T12:00:12Z",
                ["ev-02", "ev-03"],
                ["m-ash-courier", "m-ash-mechanic"],
                [lio_id, ash_id],
                "Separate courier and mechanic.",
            ),
            decision(
                prefix,
                "type",
                "contextual_type",
                "2026-09-03T12:00:13Z",
                ["ev-04"],
                [],
                [type_id],
                "Create context-specific roles.",
            ),
            decision(
                prefix,
                "abstraction",
                "abstraction",
                "2026-09-03T12:00:14Z",
                ["ev-04"],
                [],
                [schema_id],
                "Use causal-chain abstraction.",
            ),
            decision(
                prefix,
                "epistemic",
                "epistemic_qualification",
                "2026-09-03T12:00:15Z",
                ["ev-05"],
                [],
                [proposition_id, report_id],
                "Keep the claim holder-attributed.",
            ),
        ]
        interpretation = "The query needs split identities, ordered events, and Mara's report."

    return {
        "contextual_interpretation": interpretation,
        "local_schema": {
            "schema_id": schema_id,
            "contextual_types": [
                contextual_type(prefix, abstraction),
                event_contextual_type(prefix, abstraction),
            ],
            "predicates": predicates,
            "abstraction": abstraction,
        },
        "instance_graph": {
            "entities": entities,
            "events": events,
            "proposition_contents": [] if c1 else [proposition],
            "assertions": assertions,
        },
        "decisions": decisions,
        "budget_accounting": {
            "nodes_used": 4 if c1 else 8,
            "assertions_used": 2,
            "display_nodes_used": 4 if c1 else 8,
            "display_assertions_used": 2,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


def objects_by_id(items: list[dict[str, object]], id_key: str) -> dict[str, dict[str, object]]:
    return {str(item[id_key]): item for item in items}


def build_fixed_output(c1: dict[str, object]) -> dict[str, object]:
    schema = c1["local_schema"]
    graph = c1["instance_graph"]
    assert isinstance(schema, dict) and isinstance(graph, dict)
    predicates = objects_by_id(schema["predicates"], "predicate_id")
    entities = objects_by_id(graph["entities"], "entity_id")
    events = objects_by_id(graph["events"], "event_id")
    assertions = objects_by_id(graph["assertions"], "assertion_id")
    selected_ids = [
        "s-c1",
        "t-c1",
        "t-c1-event",
        "p-c1-chain",
        "p-c1-credential-enabled",
        "ent-c1-lio",
        "ent-c1-seal",
        "event-c1-delivery",
        "event-c1-opening",
        "assert-c1-chain",
        "assert-c1-enabled",
    ]
    return {
        "contextual_interpretation": "Selection-only credential-chain view of sealed C1.",
        "local_schema": {
            "schema_id": schema["schema_id"],
            "contextual_types": deepcopy(schema["contextual_types"]),
            "predicates": [
                deepcopy(predicates["p-c1-chain"]),
                deepcopy(predicates["p-c1-credential-enabled"]),
            ],
            "abstraction": schema["abstraction"],
        },
        "instance_graph": {
            "entities": [
                deepcopy(entities["ent-c1-lio"]),
                deepcopy(entities["ent-c1-seal"]),
            ],
            "events": [
                deepcopy(events["event-c1-delivery"]),
                deepcopy(events["event-c1-opening"]),
            ],
            "assertions": [
                deepcopy(assertions["assert-c1-chain"]),
                deepcopy(assertions["assert-c1-enabled"]),
            ],
        },
        "decisions": [
            decision(
                "fixed",
                "select",
                "selection",
                "2026-09-03T12:00:21Z",
                ["ev-04", "ev-06"],
                selected_ids,
                [],
                "Select a closed subset of sealed C1 objects.",
            ),
            decision(
                "fixed",
                "compress",
                "compression",
                "2026-09-03T12:00:22Z",
                ["ev-04", "ev-06"],
                ["assert-c1-chain", "assert-c1-enabled"],
                [],
                "Use the sealed assertions as the compact display.",
            ),
            decision(
                "fixed",
                "describe",
                "supported_description",
                "2026-09-03T12:00:23Z",
                ["ev-04", "ev-06"],
                ["ent-c1-lio", "ent-c1-seal", "event-c1-delivery", "event-c1-opening"],
                [],
                "Retain only descriptions already supported in C1.",
            ),
        ],
        "budget_accounting": {
            "nodes_used": 4,
            "assertions_used": 2,
            "display_nodes_used": 4,
            "display_assertions_used": 2,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


def build_repair_draft(*, invalid: bool) -> dict[str, object]:
    evidence_ids = ["ev-08"] if invalid else ["ev-04", "ev-06"]
    revelation = 8 if invalid else 6
    scope = temporal(6, world_day=2)
    scope["revelation_position"] = {"revelation_order": revelation}
    assertion_value = {
        "assertion_id": "assert-c2-enabled",
        "predicate_id": "pred-repair-enabled",
        "subject_id": "ent-repair-seal",
        "object_id": "ent-repair-gate",
        "temporal_scope": scope,
        "narrative_commitment": "world_committed",
        "confidence": 1,
        "evidence_ids": evidence_ids,
        "provenance": [provenance("repair-enabled", item) for item in evidence_ids],
        "contextual_relevance": 1,
        "why_matters": "The rare mark explains opening.",
        "why_matters_evidence_ids": evidence_ids,
    }
    return {
        "contextual_interpretation": "A rare seal mark enables the gate opening.",
        "local_schema": {
            "schema_id": "schema-repair",
            "contextual_types": [
                {
                    "type_id": "type-repair-object",
                    "label": "Operational object",
                    "definition": "An object in the credential chain.",
                    "parent_upper_type": "entity",
                    "abstraction": "collective_causal_chain",
                    "evidence_ids": ["ev-06"],
                }
            ],
            "predicates": [
                {
                    "predicate_id": "pred-repair-enabled",
                    "label": "enabled",
                    "definition": "A rare credential feature enabled operation.",
                    "arity": 2,
                    "parent_upper_relation": "causes",
                    "evidence_ids": ["ev-06"],
                }
            ],
            "abstraction": "collective_causal_chain",
        },
        "instance_graph": {
            "entities": [
                {
                    "entity_id": "ent-repair-seal",
                    "label": "Marked seal",
                    "supported_mention_candidate_ids": ["m-seal-04", "m-mark-06"],
                    "contextual_type_id": "type-repair-object",
                    "contextual_role": "rare credential",
                    "abstraction": "collective_causal_chain",
                    "temporal_state": story_day(2),
                    "uncertainty": "known",
                    "confidence": 1,
                    "evidence_ids": ["ev-04", "ev-06"],
                    "description": "Its mark enables operation.",
                    "description_assertion_ids": ["assert-c2-enabled"],
                },
                {
                    "entity_id": "ent-repair-gate",
                    "label": "Gate",
                    "supported_mention_candidate_ids": ["m-floodgate-04"],
                    "contextual_type_id": "type-repair-object",
                    "contextual_role": "enabled infrastructure",
                    "abstraction": "collective_causal_chain",
                    "temporal_state": story_day(2),
                    "uncertainty": "known",
                    "confidence": 1,
                    "evidence_ids": ["ev-04", "ev-06"],
                    "description": "The credential enables its opening.",
                    "description_assertion_ids": ["assert-c2-enabled"],
                },
            ],
            "events": [],
            "assertions": [assertion_value],
        },
        "decisions": [
            {
                "decision_id": "decision-repair-rare",
                "operator": "rare_preservation",
                "evidence_ids": ["ev-06"],
                "rationale": "Retain the one-off enabling mark.",
                "decided_at": "2026-09-03T12:00:15Z",
                "input_object_ids": ["m-mark-06"],
                "created_object_ids": ["assert-c2-enabled"],
            }
        ],
        "budget_accounting": {
            "nodes_used": 2,
            "assertions_used": 1,
            "display_nodes_used": 2,
            "display_assertions_used": 1,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


def replace_identifier_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_identifier_prefix(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_identifier_prefix(child, old, new) for child in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def set_abstraction(value: dict[str, object], abstraction: str) -> None:
    schema = value["local_schema"]
    graph = value["instance_graph"]
    assert isinstance(schema, dict) and isinstance(graph, dict)
    schema["abstraction"] = abstraction
    for contextual_type_value in schema["contextual_types"]:
        contextual_type_value["abstraction"] = abstraction
    for entity_value in graph["entities"]:
        entity_value["abstraction"] = abstraction


def build_secondary_reference(
    source: dict[str, object],
    *,
    old_prefix: str,
    new_prefix: str,
    abstraction: str,
    interpretation: str,
    decision_time_prefix: str,
) -> dict[str, object]:
    result = replace_identifier_prefix(deepcopy(source), old_prefix, new_prefix)
    assert isinstance(result, dict)
    result["contextual_interpretation"] = interpretation
    set_abstraction(result, abstraction)
    for index, decision_value in enumerate(result["decisions"], start=1):
        decision_value["decided_at"] = f"{decision_time_prefix}{index:02d}Z"
    return result


def prune_to_identity_report(value: dict[str, object], *, prefix: str) -> None:
    schema = value["local_schema"]
    graph = value["instance_graph"]
    assert isinstance(schema, dict) and isinstance(graph, dict)
    predicates = objects_by_id(schema["predicates"], "predicate_id")
    entities = objects_by_id(graph["entities"], "entity_id")
    assertions = objects_by_id(graph["assertions"], "assertion_id")
    distinct_id = f"assert-{prefix}-distinct"

    schema["predicates"] = [
        predicate(
            prefix,
            "distinct",
            "distinct",
            "Separate identity assemblies.",
            2,
            ["ev-02", "ev-03"],
        ),
        predicates[f"p-{prefix}-status"],
    ]
    schema["contextual_types"] = [
        item for item in schema["contextual_types"] if item["type_id"] == f"t-{prefix}"
    ]
    graph["entities"] = [
        entities[f"ent-{prefix}-lio"],
        entities[f"ent-{prefix}-ash-mechanic"],
        entities[f"ent-{prefix}-mara"],
        entities[f"ent-{prefix}-bridge"],
        entities[f"ent-{prefix}-safe"],
    ]
    graph["events"] = []
    graph["assertions"] = [
        assertion(
            prefix,
            "distinct",
            "distinct",
            3,
            ["ev-02", "ev-03"],
            "Prevents identity collapse.",
            ["ev-02"],
            subject=f"ent-{prefix}-ash-mechanic",
            object_=f"ent-{prefix}-lio",
        ),
        assertions[f"assert-{prefix}-report"],
    ]
    for entity_id, description in (
        (f"ent-{prefix}-lio", "Courier identity distinct from the mechanic."),
        (f"ent-{prefix}-ash-mechanic", "Mechanic identity distinct from the courier."),
    ):
        entities[entity_id]["description"] = description
        entities[entity_id]["description_assertion_ids"] = [distinct_id]
    schema["contextual_types"][0]["evidence_ids"] = ["ev-02", "ev-05"]

    decisions = objects_by_id(value["decisions"], "decision_id")
    decisions[f"d-{prefix}-include"]["evidence_ids"] = ["ev-05"]
    decisions[f"d-{prefix}-include"]["created_object_ids"] = [f"assert-{prefix}-report"]
    decisions[f"d-{prefix}-include"]["rationale"] = "Include the attributed report."
    value["budget_accounting"] = {
        "nodes_used": 5,
        "assertions_used": 2,
        "display_nodes_used": 5,
        "display_assertions_used": 2,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def build_validation_report() -> dict[str, object]:
    # The bounded repair probe contains exactly one assertion.  Diagnose that
    # collection as a unit because the invalid citation is repeated across the
    # assertion's evidence, provenance, and why-matters support, and the repair
    # must also lower its revelation coordinate.  Every schema, node, decision,
    # and accounting field remains outside this fact-free repair boundary.
    path = "instance_graph.assertions"
    return {
        "validation_status": "rejected",
        "diagnostics": [
            {
                "code": "evidence_outside_snapshot",
                "path": path,
                "message": "A citation is absent from the sealed snapshot.",
                "related_ids": ["assert-c2-enabled", "ev-08"],
            },
            {
                "code": "evidence_outside_packet",
                "path": path,
                "message": "A citation is absent from the frozen packet.",
                "related_ids": ["assert-c2-enabled", "ev-08"],
            },
            {
                "code": "horizon_rejected_evidence",
                "path": path,
                "message": "A citation is recorded as horizon-withheld.",
                "related_ids": ["assert-c2-enabled", "ev-08"],
            },
            {
                "code": "revelation_horizon_leak",
                "path": path,
                "message": "The revelation coordinate exceeds the horizon.",
                "related_ids": ["assert-c2-enabled"],
            },
        ],
    }


def token_count(tokenizer: Any, value: object) -> int:
    return len(tokenizer.encode(compact_json(value), add_special_tokens=False))


def ensure_runtime_tokenizer_hash(request: dict[str, object], tokenizer_sha256: str) -> None:
    runtime = request["runtime"]
    assert isinstance(runtime, dict)
    runtime["tokenizer_hash"] = tokenizer_sha256


def ensure_runtime_prompt_hash(request: dict[str, object], prompt_path: Path) -> None:
    runtime = request["runtime"]
    assert isinstance(runtime, dict)
    runtime["prompt_hash"] = hashlib.sha256(prompt_path.read_bytes()).hexdigest()


def build_fixed_request(
    base_request: dict[str, object],
    c1_request: PreconstructionRequest,
    c1_raw: dict[str, object],
) -> dict[str, object]:
    result = deepcopy(base_request)
    c1_draft = OntologyDraft.model_validate(c1_raw)
    result["fixed_ontology"] = {
        "construction_seal": {
            "seal_id": "seal-c1-harbor-seed-1",
            "condition": "C1",
            "snapshot_hash": c1_request.snapshot_hash,
            "ontology_hash": preontology_semantic_hash(c1_request.upper_ontology, c1_draft),
            "constructed_at": "2026-09-03T11:46:00Z",
            "sealed_at": "2026-09-03T11:47:00Z",
            "sealed_object_ids": list(sealed_semantic_ids(c1_draft)),
        },
        "upper_ontology": deepcopy(base_request["upper_ontology"]),
        "local_schema": deepcopy(c1_raw["local_schema"]),
        "instance_graph": deepcopy(c1_raw["instance_graph"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if manifest["repository"] != MODEL_REPOSITORY or manifest["revision"] != MODEL_REVISION:
        raise RuntimeError("model manifest does not identify the registered tokenizer revision")
    tokenizer_entry = next(item for item in manifest["files"] if item["path"] == "tokenizer.json")
    tokenizer_path = args.tokenizer_snapshot / "tokenizer.json"
    actual_tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if actual_tokenizer_hash != tokenizer_entry["sha256"]:
        raise RuntimeError("pinned tokenizer.json does not match the public snapshot manifest")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_snapshot,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_manifest = capture_tokenizer_manifest(args.tokenizer_snapshot)
    c1_raw = build_full_draft("c1")
    c2_raw = build_full_draft("c2")
    c1_secondary_raw = build_secondary_reference(
        c2_raw,
        old_prefix="c2",
        new_prefix="c1b",
        abstraction="event_role",
        interpretation="Query-blind split identities and a holder-attributed report.",
        decision_time_prefix="2026-09-03T11:45:",
    )
    prune_to_identity_report(c1_secondary_raw, prefix="c1b")
    prune_to_identity_report(c2_raw, prefix="c2")
    c2_raw["contextual_interpretation"] = (
        "The query needs separate courier and mechanic identities and Mara's report."
    )
    c2_secondary_raw = build_secondary_reference(
        c1_raw,
        old_prefix="c1",
        new_prefix="c2b",
        abstraction="collective_causal_chain",
        interpretation="The query needs merged aliases, ordered events, and the rare mark.",
        decision_time_prefix="2026-09-03T12:00:",
    )
    fixed_raw = build_fixed_output(c1_raw)
    invalid_raw = build_repair_draft(invalid=True)
    corrected_raw = build_repair_draft(invalid=False)

    OntologyDraft.model_validate(c1_raw)
    OntologyDraft.model_validate(c1_secondary_raw)
    OntologyDraft.model_validate(c2_raw)
    OntologyDraft.model_validate(c2_secondary_raw)
    OntologyDraft.model_validate(fixed_raw)
    OntologyDraft.model_validate(invalid_raw)
    OntologyDraft.model_validate(corrected_raw)

    c1_request_raw = json.loads((FIXTURES / "c1_pre_request.json").read_text(encoding="utf-8"))
    c2_request_raw = json.loads((FIXTURES / "c2_query_request.json").read_text(encoding="utf-8"))
    fixed_request_raw = json.loads(
        (FIXTURES / "fixed_select_request.json").read_text(encoding="utf-8")
    )
    for request in (c1_request_raw, c2_request_raw, fixed_request_raw):
        ensure_runtime_tokenizer_hash(request, tokenizer_manifest.manifest_sha256)
    ensure_runtime_prompt_hash(c1_request_raw, ROOT / "prompts/c1_pre/prompt_v1.md")
    ensure_runtime_prompt_hash(c2_request_raw, ROOT / "prompts/c2_query/prompt_v1.md")
    ensure_runtime_prompt_hash(
        fixed_request_raw,
        ROOT / "prompts/fixed_select/prompt_v1.md",
    )
    c1_request = PreconstructionRequest.model_validate(c1_request_raw)
    ConstructionRequest.model_validate(c2_request_raw)
    fixed_request_raw = build_fixed_request(fixed_request_raw, c1_request, c1_raw)
    ConstructionRequest.model_validate(fixed_request_raw)

    validation_report = build_validation_report()
    BoundaryValidationReport.model_validate(validation_report)
    repair_lineage = {
        "root_attempt_id": "phase1-c2-invalid-base",
        "base_attempt_id": "phase1-c2-invalid-base",
        "repair_attempt_id": "phase1-c2-repair-1",
        "repair_number": 1,
        "semantic_request_hash": canonical_sha256(c2_request_raw),
        "base_output_hash": canonical_sha256(invalid_raw),
        "validation_record_hash": canonical_sha256(validation_report),
        "diagnostic_codes": [item["code"] for item in validation_report["diagnostics"]],
        "allowed_changes": ["validation_diagnostics", "maximum_output_tokens"],
    }
    RepairLineageMetadata.model_validate(repair_lineage)
    repair_case = {
        "prompt_path": "prompts/repair/prompt_v1.md",
        "base_attempt_id": "phase1-c2-invalid-base",
        "repair_attempt_id": "phase1-c2-repair-1",
        "base_draft": invalid_raw,
        "validation_report": validation_report,
        "repair_lineage": repair_lineage,
        "corrected_draft": corrected_raw,
    }

    values = {
        "c1_pre_output.json": (c1_raw, FULL_OUTPUT_CAP),
        "c1_pre_output_2.json": (c1_secondary_raw, FULL_OUTPUT_CAP),
        "c2_query_output.json": (c2_raw, FULL_OUTPUT_CAP),
        "c2_query_output_2.json": (c2_secondary_raw, FULL_OUTPUT_CAP),
        "fixed_select_output.json": (fixed_raw, FULL_OUTPUT_CAP),
        "invalid_repair_case.json#base_draft": (invalid_raw, REPAIR_OUTPUT_CAP),
        "invalid_repair_case.json#corrected_draft": (corrected_raw, REPAIR_OUTPUT_CAP),
    }
    receipt_outputs: dict[str, object] = {}
    for name, (value, cap) in values.items():
        encoded = compact_json(value).encode("utf-8")
        count = token_count(tokenizer, value)
        if count > cap:
            raise RuntimeError(f"{name} uses {count} tokens, above its {cap}-token cap")
        receipt_outputs[name] = {
            "compact_json_sha256": hashlib.sha256(encoded).hexdigest(),
            "compact_json_bytes": len(encoded),
            "exact_token_count": count,
            "maximum_output_tokens": cap,
        }
    receipt = {
        "schema_version": "1.0.0",
        "serialization": "UTF-8 JSON; ensure_ascii=false; allow_nan=false; sorted keys; no spaces",
        "tokenizer_repository": MODEL_REPOSITORY,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_json_sha256": actual_tokenizer_hash,
        "tokenizer_manifest": tokenizer_manifest.public_manifest(),
        "add_special_tokens": False,
        "model_weights_loaded": False,
        "gpu_used": False,
        "outputs": receipt_outputs,
    }

    write_json(FIXTURES / "c1_pre_request.json", c1_request_raw)
    write_json(FIXTURES / "c2_query_request.json", c2_request_raw)
    write_json(FIXTURES / "fixed_select_request.json", fixed_request_raw)
    write_json(FIXTURES / "c1_pre_output.json", c1_raw)
    write_json(FIXTURES / "c1_pre_output_2.json", c1_secondary_raw)
    write_json(FIXTURES / "c2_query_output.json", c2_raw)
    write_json(FIXTURES / "c2_query_output_2.json", c2_secondary_raw)
    write_json(FIXTURES / "fixed_select_output.json", fixed_raw)
    write_json(FIXTURES / "invalid_repair_case.json", repair_case)
    write_json(FIXTURES / "output_token_caps.json", receipt)

    representative_calls = {}
    for call in phase1_acceptance_calls():
        if call.decoding_pass is not DecodingPass.FIRST_PASS:
            continue
        representative_calls.setdefault(call.condition, call)
    packing_reports = [
        build_acceptance_request(
            root=ROOT,
            call=representative_calls[condition],
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
        ).packing.model_dump(mode="json")
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
        )
    ]
    write_json(FIXTURES / "packing_reports.json", packing_reports)

    for name, record in receipt_outputs.items():
        print(f"{name}: {record['exact_token_count']} / {record['maximum_output_tokens']} tokens")


if __name__ == "__main__":
    main()
