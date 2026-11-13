"""CPU-only candidate: inline content with explicit, model-chosen content identity.

Not imported by any condition or GPU controller. No semantic extraction, equality
inference, default attribution, automatic deduplication or production activation.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from .contracts import EvidenceRecord, UpperOntology, canonical_sha256
from .semantic_generation import (
    AdaptedSemanticDraft,
    ExecutionFacts,
    _array,
    _object,
    _ref,
    repair_epistemic_schema,
    semantic_schema,
)
from .semantic_identifiers import reconstruct_typed, typed_identifier_schema

REVISION = "nested-content-cpu-candidate-v1"
KEY = {"type": "string", "pattern": r"^c[1-9][0-9]{0,3}$"}
ASSERTION_ID = {"type": "string", "pattern": r"^nA[1-9][0-9]{0,3}$"}
BINDING_FIELDS = ("form", "predicate_id", "subject_id", "object_id", "roles")


def candidate_schema(
    evidence: Sequence[EvidenceRecord], upper: UpperOntology, *, small: bool = False
) -> dict:
    """Reuse every ordinary canonical scientific field and temporal alternative."""
    schema = repair_epistemic_schema(typed_identifier_schema(semantic_schema(evidence, upper)))
    defs = schema["$defs"]
    # This candidate is qualified generation, not the separately registered
    # NoTemporalEpistemic ablation. No unqualified branch may bypass its fields.
    schema["properties"]["instance_graph"] = _ref("InstanceGraph")
    content = copy.deepcopy(defs["PropositionContent"])
    for branch in content["anyOf"]:
        branch["properties"].pop("proposition_content_id")
        branch["required"].remove("proposition_content_id")
    defs["InlineContent"] = content
    defs["ContentIdentity"] = {
        "anyOf": [
            _object({"kind": {"const": "distinct"}}),
            _object({"kind": {"const": "shared"}, "key": copy.deepcopy(KEY)}),
        ]
    }
    scope = defs["EpistemicScope"]
    scope["properties"].pop("proposition_content_id")
    scope["required"].remove("proposition_content_id")
    # Binding shape is authored ONCE in content. The old binary/nary branches
    # differ only there, so retain one absent/present-scope pair.
    assertions = []
    for branch in defs["QualifiedAssertion"]["anyOf"][:2]:
        props = copy.deepcopy(branch["properties"])
        for key in (*BINDING_FIELDS, "proposition_content_id"):
            props.pop(key, None)
        attributed = props["epistemic_scope"] != {"type": "null"}
        props["content"] = _ref("InlineContent")
        props["content_identity"] = _ref("ContentIdentity") if attributed else {"type": "null"}
        props["temporal_scope"] = {"anyOf": [{"const": "content"}, _ref("TemporalScope")]}
        assertions.append(_object(props))
    defs["QualifiedAssertion"] = {"anyOf": assertions}
    defs.pop("PropositionContent")
    graph = defs["InstanceGraph"]
    graph["properties"].pop("proposition_contents")
    graph["required"].remove("proposition_contents")
    graph["properties"]["unasserted_contents"] = _array(
        _object(
            {
                "key": copy.deepcopy(KEY),
                "content": _ref("InlineContent"),
            }
        )
    )
    graph["required"].append("unasserted_contents")
    # Existing typed destinations stay unchanged, except nP references are now
    # explicit addresses into a model-authored nested/standalone content record.
    address = {
        "anyOf": [
            _object({"content_of_assertion": copy.deepcopy(ASSERTION_ID)}),
            _object({"unasserted_content": copy.deepcopy(KEY)}),
        ]
    }

    def references(node):
        if not isinstance(node, dict):
            return
        pattern = node.get("pattern")
        if isinstance(pattern, str) and "nP" in pattern:
            old = copy.deepcopy(node)
            old["pattern"] = pattern.replace("|nP", "").replace("nP|", "")
            node.clear()
            node["anyOf"] = [old, copy.deepcopy(address)]
            return
        for value in node.values():
            if isinstance(value, list):
                for item in value:
                    references(item)
            elif isinstance(value, dict):
                references(value)

    references(schema)
    if small:
        graph["properties"]["assertions"].update(minItems=1, maxItems=3)
        defs["InstanceGraph"] = {"anyOf": []}
        for entities in range(5):
            branch = copy.deepcopy(graph)
            branch["properties"]["entities"].update(minItems=entities, maxItems=entities)
            branch["properties"]["events"].update(
                minItems=max(0, 2 - entities), maxItems=4 - entities
            )
            defs["InstanceGraph"]["anyOf"].append(branch)
        schema["properties"]["decisions"]["minItems"] = 1
        for field in ("contextual_types", "predicates"):
            defs["LocalContextSchema"]["properties"][field]["minItems"] = 1
    # Remove unreachable old ablation definitions from the field guide, not
    # scientific fields reachable from this qualified generation contract.
    used: set[str] = set()

    def collect(node):
        if isinstance(node, list):
            for v in node:
                collect(v)
        elif isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name not in used:
                    used.add(name)
                    collect(defs[name])
            for k, v in node.items():
                if k != "$defs":
                    collect(v)

    collect(schema)
    schema["$defs"] = {k: v for k, v in defs.items() if k in used}
    Draft202012Validator.check_schema(schema)
    return schema


@dataclass(frozen=True)
class ExpandedContent:
    semantic_payload: Mapping[str, Any]
    representation_receipt: Mapping[str, Any]


def expand_candidate(
    generated: Mapping[str, Any],
    *,
    evidence: Sequence[EvidenceRecord],
    upper: UpperOntology,
    small: bool = False,
) -> ExpandedContent:
    """Mechanical lifting. Equal text alone NEVER establishes shared identity."""
    Draft202012Validator(candidate_schema(evidence, upper, small=small)).validate(generated)
    value = copy.deepcopy(dict(generated))
    graph = value["instance_graph"]
    contents: list[dict] = []
    shared: dict[str, str] = {}
    by_assertion: dict[str, str] = {}
    receipt: dict[str, Any] = {
        "revision": REVISION,
        "assertions": [],
        "standalone": [],
        "reference_addresses": [],
        "source_semantic_hash": canonical_sha256(generated),
        "derived_ids_are_administrative": True,
    }

    def lift(body, key=None):
        if key is not None and key in shared:
            pid = shared[key]
            old = next(p for p in contents if p["proposition_content_id"] == pid)
            if {k: v for k, v in old.items() if k != "proposition_content_id"} != body:
                raise ValueError(
                    "explicit shared content key has conflicting bodies; no semantic merge"
                )
            return pid
        pid = "nP" + str(len(contents) + 1)
        contents.append({"proposition_content_id": pid, **copy.deepcopy(body)})
        if key is not None:
            shared[key] = pid
        return pid

    standalone = graph.pop("unasserted_contents")
    if len({p["key"] for p in standalone}) != len(standalone):
        raise ValueError("duplicate standalone content declaration")
    standalone_ids = {}
    for item in standalone:
        pid = lift(item["content"], item["key"])
        standalone_ids[item["key"]] = pid
        receipt["standalone"].append({"key": item["key"], "proposition_id": pid})
    if len({a["assertion_id"] for a in graph["assertions"]}) != len(graph["assertions"]):
        raise ValueError("duplicate assertion ID prevents unambiguous content address")
    for assertion in graph["assertions"]:
        body = assertion.pop("content")
        identity = assertion.pop("content_identity")
        aid = assertion["assertion_id"]
        scope_from_content = assertion["temporal_scope"] == "content"
        if scope_from_content:
            assertion["temporal_scope"] = copy.deepcopy(body["temporal_content"])
        if not set(body["evidence_ids"]) <= set(assertion["evidence_ids"]):
            raise ValueError("content evidence must be part of assertion evidence")
        assertion.update({k: copy.deepcopy(body[k]) for k in BINDING_FIELDS if k in body})
        if assertion["epistemic_scope"] is None:
            # With no canonical content record, unequal time/citations would be
            # silently lost. Reject rather than infer what the model intended.
            if (
                body["temporal_content"] != assertion["temporal_scope"]
                or body["evidence_ids"] != assertion["evidence_ids"]
            ):
                raise ValueError("direct content must preserve assertion time/evidence exactly")
            pid = None
        else:
            pid = lift(body, identity["key"] if identity["kind"] == "shared" else None)
            assertion["epistemic_scope"]["proposition_content_id"] = pid
            by_assertion[aid] = pid
        assertion["proposition_content_id"] = pid
        receipt["assertions"].append(
            {
                "assertion_id": aid,
                "proposition_id": pid,
                "content_identity": identity,
                "temporal_scope_from_content": scope_from_content,
            }
        )
    graph["proposition_contents"] = contents

    def addresses(obj, path=()):
        if isinstance(obj, list):
            return [addresses(v, (*path, i)) for i, v in enumerate(obj)]
        if isinstance(obj, dict):
            table = (
                by_assertion
                if set(obj) == {"content_of_assertion"}
                else standalone_ids
                if set(obj) == {"unasserted_content"}
                else None
            )
            if table is not None:
                key = next(iter(obj.values()))
                if key not in table:
                    raise ValueError(
                        "content address has no declared attributed/standalone destination"
                    )
                receipt["reference_addresses"].append({"path": list(path), "address": obj})
                return table[key]
            return {k: addresses(v, (*path, k)) for k, v in obj.items()}
        return obj

    value = addresses(value)
    receipt["expanded_semantic_hash"] = canonical_sha256(value)
    return ExpandedContent(value, receipt)


def collapse_candidate(expanded: ExpandedContent) -> dict:
    """Inverse from expanded semantic records plus representation-only receipt.

    Receipt stores original address/identity syntax, not endpoints, time, evidence
    or any scientific content. Those must all be recovered from expanded records.
    """
    value = copy.deepcopy(dict(expanded.semantic_payload))
    receipt = expanded.representation_receipt
    if canonical_sha256(value) != receipt["expanded_semantic_hash"]:
        raise ValueError("expanded payload changed; cannot claim a lossless round trip")
    for item in receipt["reference_addresses"]:
        target = value
        for part in item["path"][:-1]:
            target = target[part]
        target[item["path"][-1]] = copy.deepcopy(item["address"])
    graph = value["instance_graph"]
    contents = {p["proposition_content_id"]: p for p in graph.pop("proposition_contents")}

    def body(pid):
        return {
            k: copy.deepcopy(v) for k, v in contents[pid].items() if k != "proposition_content_id"
        }

    graph["unasserted_contents"] = [
        {"key": item["key"], "content": body(item["proposition_id"])}
        for item in receipt["standalone"]
    ]
    for assertion, item in zip(graph["assertions"], receipt["assertions"], strict=True):
        pid = assertion.pop("proposition_content_id")
        content = (
            body(pid)
            if pid
            else {
                **{k: copy.deepcopy(assertion[k]) for k in BINDING_FIELDS if k in assertion},
                "temporal_content": copy.deepcopy(assertion["temporal_scope"]),
                "evidence_ids": copy.deepcopy(assertion["evidence_ids"]),
            }
        )
        for key in BINDING_FIELDS:
            assertion.pop(key, None)
        if assertion["epistemic_scope"] is not None:
            assertion["epistemic_scope"].pop("proposition_content_id")
        assertion.update(content=content, content_identity=item["content_identity"])
        if item["temporal_scope_from_content"]:
            assertion["temporal_scope"] = "content"
    if canonical_sha256(value) != receipt["source_semantic_hash"]:
        raise ValueError("candidate round trip lost model-authored content")
    return value


def reconstruct_candidate(
    generated: Mapping[str, Any],
    *,
    evidence: Sequence[EvidenceRecord],
    upper: UpperOntology,
    execution: ExecutionFacts,
    small: bool = False,
) -> AdaptedSemanticDraft:
    expanded = expand_candidate(generated, evidence=evidence, upper=upper, small=small)
    if collapse_candidate(expanded) != generated:
        raise ValueError("candidate semantic round-trip failed")
    adapted = reconstruct_typed(
        expanded.semantic_payload, evidence=evidence, upper=upper, execution=execution, small=small
    )
    return AdaptedSemanticDraft(
        adapted.draft,
        {
            **adapted.provenance,
            "candidate_revision": REVISION,
            "candidate_semantic_hash": canonical_sha256(generated),
            "nested_content_translation": expanded.representation_receipt,
            "production_adoption_authorized": False,
            "scientific_acceptance_claimed": False,
        },
    )


def build_candidate_request(fixture, tokenizer, tokenizer_manifest, instruction: str):
    """Complete CPU-packed request; no execution entrypoint or scheduler mutation."""
    from .gpu_runtime import ChatMessage
    from .representation_diagnostic import _repack
    from .semantic_generation import build_small_request, schema_guide

    base = build_small_request(fixture, tokenizer, tokenizer_manifest)
    schema = candidate_schema(fixture.evidence, fixture.upper_ontology, small=True)
    return _repack(
        base,
        (
            ChatMessage(role="system", content=instruction.strip() + "\n\n" + schema_guide(schema)),
            *base.messages[1:],
        ),
        schema,
        tokenizer,
        label="nested-content-cpu-v1",
    )
