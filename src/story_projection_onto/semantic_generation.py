"""Diagnostic-only named semantic contract; no ordinary condition is rerouted.

The JSON schema is derived from canonical fields with an explicit, small set of
representation changes. It deliberately does not treat guided decoding as
referential, canonical, or scientific validation. No scorer imports are allowed.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

from story_projection_onto.contracts import (
    SCHEMA_VERSION,
    EvidenceRecord,
    OntologyDraft,
    UpperOntology,
    canonical_sha256,
)

INTERFACE_REVISION = "named-semantic-diagnostic-v1"
REPAIR_CONTRACT_REVISION = "small-epistemic-contract-repair-v4"
ADMIN_FIELDS = frozenset({"content_hash", "schema_version"})
LOCAL_ID_PATTERN = r"^n[A-Za-z0-9_-]{1,63}$"


def _object(properties: dict, required: Sequence[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties if required is None else required),
        "additionalProperties": False,
    }


def _ref(name: str) -> dict:
    return {"$ref": f"#/$defs/{name}"}


def _array(items: dict, minimum: int = 0) -> dict:
    return {"type": "array", "items": items, "minItems": minimum}


def _strip_annotations(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_annotations(v) for v in value]
    if not isinstance(value, dict):
        return value
    return {
        k: (
            {name: _strip_annotations(child) for name, child in v.items()}
            if k in {"properties", "$defs"}
            else _strip_annotations(v)
        )
        for k, v in value.items()
        if k not in {"title", "description", "default"}
    }


def semantic_schema(
    evidence: Sequence[EvidenceRecord], upper: UpperOntology, *, small: bool = False
) -> dict:
    """Bind supplied references while leaving the newly constructed graph unknown.

    All non-variant scientific fields are explicit, including nullable attitudes
    and empty lists. Defaults cannot silently manufacture a scientific choice.
    Unknown time remains a first-class alternative, not a made-up numeric bound.
    """
    schema = _strip_annotations(OntologyDraft.model_json_schema())
    definitions = schema["$defs"]
    for record in (schema, *definitions.values()):
        if "properties" not in record:
            continue
        for name in ADMIN_FIELDS:
            record["properties"].pop(name, None)
        record["required"] = list(record["properties"])
    schema["properties"].pop("budget_accounting")
    schema["required"].remove("budget_accounting")
    del definitions["BudgetAccounting"]
    for name, field in (("OntologyDecision", "decided_at"),):
        definitions[name]["properties"].pop(field)
        definitions[name]["required"].remove(field)

    # Citation-specific confidence is a model judgment. Locator/hash/ID/method
    # are execution/source bookkeeping, reconstructed only from frozen sources.
    definitions["ProvenanceReference"] = _object(
        {
            "evidence_id": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        }
    )
    local_id = {"type": "string", "pattern": LOCAL_ID_PATTERN}
    evidence_id = {"type": "string", "enum": sorted(e.evidence_id for e in evidence)}
    if not evidence_id["enum"]:
        raise ValueError("diagnostic contract requires a supplied evidence vocabulary")
    mentions = sorted(m.candidate_id for e in evidence for m in e.mention_candidates)
    candidates = sorted(
        {
            value
            for e in evidence
            for value in (
                *(m.candidate_id for m in e.mention_candidates),
                *(m.candidate_id for m in e.event_candidates),
                *(m.candidate_id for m in e.relation_phrase_candidates),
                *(m.clue_id for m in e.temporal_clues),
            )
        }
    )
    if any(
        re.fullmatch(LOCAL_ID_PATTERN, x)
        for x in (
            *evidence_id["enum"],
            *candidates,
            *upper.primitive_types,
            *upper.primitive_relations,
        )
    ):
        raise ValueError("supplied vocabulary collides with new local-ID namespace")

    for record in definitions.values():
        for field, node in record.get("properties", {}).items():
            if field in {"evidence_id", "evidence_ids", "why_matters_evidence_ids"}:
                record["properties"][field] = (
                    copy.deepcopy(evidence_id)
                    if field == "evidence_id"
                    else _array(copy.deepcopy(evidence_id), 1)
                )
            elif field == "supported_mention_candidate_ids":
                record["properties"][field] = (
                    _array({"type": "string", "enum": mentions}, 1)
                    if mentions
                    else {"type": "array", "maxItems": 0}
                )
            elif field == "parent_upper_type":
                record["properties"][field] = {
                    "enum": list(upper.primitive_types),
                    "type": "string",
                }
            elif field == "parent_upper_relation":
                record["properties"][field] = {
                    "enum": list(upper.primitive_relations),
                    "type": "string",
                }
            elif field == "input_object_ids":
                record["properties"][field] = _array(
                    {
                        "anyOf": [
                            copy.deepcopy(local_id),
                            {"type": "string", "enum": candidates},
                        ]
                    }
                    if candidates
                    else copy.deepcopy(local_id)
                )
            elif field.endswith("_ids") and field != "role_names":
                record["properties"][field] = _array(
                    copy.deepcopy(local_id), 1 if field == "description_assertion_ids" else 0
                )
            elif field.endswith("_id"):
                nullable = any(v.get("type") == "null" for v in node.get("anyOf", []))
                record["properties"][field] = (
                    {"anyOf": [copy.deepcopy(local_id), {"type": "null"}]}
                    if nullable
                    else copy.deepcopy(local_id)
                )

    # Coordinate/relative/partial-order branches cannot coexist. Labels/reasons
    # stay optional annotations; missing numeric precision must use unknown.
    common = {"label": {"type": "string"}, "reason": {"type": "string"}}

    def time_branch(kind: str, fields: dict, required: Sequence[str] = ()) -> dict:
        return _object(
            {"kind": {"const": kind}, **copy.deepcopy(common), **fields},
            ("kind", *required),
        )

    integer = {"type": "integer"}
    times = [
        time_branch("point", {"point": integer}, ("point",)),
        time_branch("interval", {"start": integer, "end": integer}, ("start", "end")),
        time_branch("interval", {"start": integer}, ("start",)),
        time_branch("interval", {"end": integer}, ("end",)),
        time_branch(
            "relative",
            {"anchor_id": local_id, "relation": _ref("AllenRelation")},
            ("anchor_id", "relation"),
        ),
        time_branch(
            "partial_order",
            {"partial_order": _array(_ref("PartialOrderConstraint"), 1)},
            ("partial_order",),
        ),
        *(time_branch(k, {}, ("reason",)) for k in ("unknown", "horizon_withheld", "invalid")),
        time_branch("not_applicable", {}),
    ]
    definitions["SemanticTime"] = {"anyOf": times}
    for name in ("StoryTime", "ValidityTime", "HolderRelativeTime"):
        definitions[name] = _ref("SemanticTime")
    definitions.pop("TemporalKind", None)

    for name in ("QualifiedAssertion", "PropositionContent"):
        props = definitions[name]["properties"]
        common_props = {
            k: v for k, v in props.items() if k not in {"subject_id", "object_id", "roles"}
        }
        definitions[name] = {
            "anyOf": [
                _object(
                    {
                        **copy.deepcopy(common_props),
                        "form": {"const": "binary"},
                        "subject_id": copy.deepcopy(local_id),
                        "object_id": copy.deepcopy(local_id),
                    }
                ),
                _object(
                    {
                        **copy.deepcopy(common_props),
                        "form": {"const": "nary"},
                        "roles": _array(_ref("RoleBinding"), 2),
                    }
                ),
            ]
        }
    definitions["QualifiedAssertion"]["anyOf"][0]["properties"]["provenance"]["minItems"] = 1
    definitions["QualifiedAssertion"]["anyOf"][1]["properties"]["provenance"]["minItems"] = 1
    if small:
        schema["properties"]["decisions"]["minItems"] = 1
        for field in ("contextual_types", "predicates"):
            definitions["LocalContextSchema"]["properties"][field]["minItems"] = 1
        graph = definitions["InstanceGraph"]
        graph["properties"]["assertions"].update(minItems=1, maxItems=3)
        alternatives = []
        for entities_min, events_min in ((2, 0), (1, 1), (0, 2)):
            branch = copy.deepcopy(graph)
            branch["properties"]["entities"].update(minItems=entities_min, maxItems=4)
            branch["properties"]["events"].update(minItems=events_min, maxItems=4)
            alternatives.append(branch)
        definitions["InstanceGraph"] = {"anyOf": alternatives}
    Draft202012Validator.check_schema(schema)
    return schema


def build_clarified_small_request(fixture, tokenizer, tokenizer_manifest, instruction: str):
    """CPU-prepared instruction revision; existing codec/schema/runtime unchanged.

    Opt-in only. The frozen v1/typed request builder remains reproducible. The
    instruction contains no evidence-specific answers and cannot change the
    complete evidence, template, sampling settings, or output allowance.
    """
    from story_projection_onto.gpu_runtime import ChatMessage
    from story_projection_onto.representation_diagnostic import _repack
    from story_projection_onto.semantic_identifiers import (
        IDENTIFIER_INSTRUCTION,
        build_typed_small_request,
    )

    if instruction.count("{{TYPED_IDENTIFIERS}}") != 1:
        raise ValueError("exactly one unchanged typed-ID contract placeholder required")
    base = build_typed_small_request(fixture, tokenizer, tokenizer_manifest)
    marker = "The following complete field guide"
    guide = marker + base.messages[0].content.split(marker, 1)[1]
    text = instruction.replace("{{TYPED_IDENTIFIERS}}", IDENTIFIER_INSTRUCTION).strip()
    return _repack(
        base,
        (ChatMessage(role="system", content=text + "\n\n" + guide), *base.messages[1:]),
        base.output_schema,
        tokenizer,
        label="semantic-instruction-v2",
    )


def build_small_request(fixture, tokenizer, tokenizer_manifest):
    """CPU packing into the existing GuidedJSONRequest/streaming client pathway.

    No execution hook, model loading, or admission exception is introduced here.
    Complete evidence records remain intact; the guide describes schema syntax,
    not a possible answer. The only task is the already declared small diagnostic.
    """
    from dataclasses import asdict

    from story_projection_onto.contracts import ConditionName, canonical_json
    from story_projection_onto.gpu_runtime import ChatMessage, GuidedJSONRequest
    from story_projection_onto.llm import DecodingManifest, PackingReport, PackingSection

    if fixture.condition is not ConditionName.C1_LLM_PRE or len(fixture.evidence) != 1:
        raise ValueError("this diagnostic builder requires the complete one-passage C1 task")
    schema = semantic_schema(fixture.evidence, fixture.upper_ontology, small=True)
    instruction = (
        "Construct a minimal meaningful ontology from the complete development evidence. "
        "No expected answer is supplied. Return 2 to 4 distinct supported entity/event nodes, "
        "1 to 3 qualified assertions, their local types/predicates and at least one supported "
        "nonselection construction decision. Candidates are defeasible hints. "
        "Return readable named-field compact JSON only, no markdown or reasoning outside fields. "
        "Generate scientific decisions only: NEVER timestamps, hashes, schema versions "
        "or token counts. "
        "New local IDs start with n, are distinct across nodes, assertions, propositions, types, "
        "predicates and decisions, and must be reused exactly in references. Evidence and mention "
        "IDs are supplied, not new IDs. Upper parents must be supplied upper terms. "
        "Each assertion/proposition has form binary with subject_id and object_id, OR form nary "
        "with at least two explicit roles (role, object_id, evidence_ids). Never omit bindings. "
        "A local predicate declares its arity, domain/range types and role_names; "
        "bindings must agree. Choose types, relation meanings, event boundaries, abstraction "
        "and descriptions from evidence. Entity descriptions/labels and event descriptions "
        "must be justified by the cited assertion IDs. "
        "why_matters_evidence_ids must support every factual clause of why_matters. "
        "Confidence, contextual_relevance, provenance confidence and omission confidence are YOUR "
        "evidence-grounded judgments in [0,1], not bookkeeping. Provenance lists exactly one "
        "evidence_id and your support confidence for each cited source; source confidence "
        "ceilings apply. "
        "Story time describes occurrence; intrinsic validity describes how long a relation holds. "
        "Observation is not onset; query visibility is not intrinsic duration. A point requires an "
        "evidence-supported integer coordinate; a label alone cannot supply it. Without "
        "a supported "
        "numeric coordinate use unknown with reason/label, or a supported relative/partial order. "
        "An interval has one or two supported bounds; omit unsupported bounds. Partial-order and "
        "relative anchors reference created local objects, not invented hidden dates. "
        "Never mix time "
        "forms. Unknown means applicable but undetermined; not_applicable means the time concept "
        "does not apply; horizon_withheld and invalid remain explicit failure/withheld states. "
        "discourse_position gives supplied passage/sentence/token order; revelation_position gives "
        "proposition disclosure order, not story time. Do not infer causality from precedence. "
        "A belief/report/denial requires holder_id, attitude, holder_relative_time, evidence_ids, "
        "and a separately declared nonasserted proposition_content_id. Assertion and epistemic "
        "proposition IDs must agree. With epistemic_scope, never mark world_committed; "
        "global truth "
        "requires a separate supported assertion. With no attitude use epistemic_scope=null "
        "and "
        "proposition_content_id=null. Preserve contested/unknown commitment when warranted. "
        "Decisions name supplied candidate/local input IDs and created/removed local object IDs, "
        "with evidence and rationale. Use omissions and uncertainty_and_abstentions "
        "for unsupported "
        "or underdetermined content; they do not excuse an empty graph for this small task. "
        "The following complete field guide matches the supplied grammar. "
        "It is syntax, not an answer.\n" + schema_guide(schema)
    )
    sections = {
        "evidence_snapshot": [e.model_dump(mode="json") for e in fixture.evidence],
        "upper_ontology": fixture.upper_ontology.model_dump(mode="json"),
        "sealed_horizon": fixture.sealed_horizon.model_dump(mode="json"),
        "budgets": fixture.budgets.model_dump(mode="json"),
        "capabilities": fixture.capabilities.model_dump(mode="json"),
    }
    messages = (
        ChatMessage(role="system", content=instruction),
        ChatMessage(role="user", content=canonical_json(sections)),
    )
    count = len(
        tokenizer.apply_chat_template(
            [asdict(m) for m in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    decoding = DecodingManifest.first_pass(
        seed=0,
        eos_token_id=tokenizer_manifest.eos_token_id,
        end_of_turn_token_ids=tokenizer_manifest.end_of_turn_token_ids,
        chat_template_hash=tokenizer_manifest.chat_template_sha256,
        output_schema_hash=canonical_sha256(schema),
        structured_decoder="vllm-0.10.2-xgrammar-no-fallback",
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_input_tokens=6144,
        maximum_output_tokens=6144,
    )
    parts = (
        PackingSection(
            name="system_prompt", section_content_hash=canonical_sha256(instruction), token_count=0
        ),
        PackingSection(
            name="output_schema", section_content_hash=canonical_sha256(schema), token_count=0
        ),
        *(
            PackingSection(name=k, section_content_hash=canonical_sha256(v), token_count=0)
            for k, v in sections.items()
        ),
        PackingSection(
            name="complete_template_inclusive_request",
            token_count=count,
            section_content_hash=canonical_sha256([asdict(m) for m in messages]),
        ),
    )
    packing = PackingReport.build(
        condition=fixture.condition,
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_model_tokens=12288,
        maximum_input_tokens=6144,
        reserved_output_tokens=6144,
        sections=parts,
        required_section_names=tuple(p.name for p in parts),
        complete_evidence_snapshot=True,
        complete_evidence_packet=None,
        complete_sealed_ontology=None,
    )
    return GuidedJSONRequest(
        request_id="semantic-interface-small-v1",
        model_name="qwen3-8b-awq-fallback",
        condition=fixture.condition,
        messages=messages,
        output_schema=schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=count,
        stream_response=True,
    )


def repair_epistemic_schema(schema: Mapping[str, Any]) -> dict:
    """Diagnostic retry only: encode existing canonical scope/commitment branches.

    The field vocabulary and all temporal/role alternatives remain unchanged.
    ID equality, declaration existence and evidence support still require CPU
    validation; JSON Schema cannot express those dynamic graph joins here.
    """
    value = copy.deepcopy(schema)
    branches = []
    for shape in value["$defs"]["QualifiedAssertion"]["anyOf"]:
        for attributed in (False, True):
            branch = copy.deepcopy(shape)
            props = branch["properties"]
            if attributed:
                props["epistemic_scope"] = _ref("EpistemicScope")
                props["proposition_content_id"] = next(
                    a for a in props["proposition_content_id"]["anyOf"] if a.get("type") != "null"
                )
                commitments = ["holder_attributed", "contested", "unknown"]
            else:
                props["epistemic_scope"] = {"type": "null"}
                props["proposition_content_id"] = {"type": "null"}
                commitments = ["world_committed", "contested", "unknown"]
            props["narrative_commitment"] = {"type": "string", "enum": commitments}
            branches.append(branch)
    value["$defs"]["QualifiedAssertion"] = {"anyOf": branches}
    Draft202012Validator.check_schema(value)
    return value


EPISTEMIC_REPAIR_INSTRUCTION = (
    "Retry contract v4. Formatting: one space after each colon and comma, as required "
    "by the fixed-whitespace grammar; no indentation or repeated whitespace. "
    "The field guide's assertion forms additionally have these "
    "exclusive schema constraints: absent scope => epistemic_scope=null, "
    "proposition_content_id=null, commitment world_committed/contested/unknown; "
    "present scope => nonnull scope and proposition ID, commitment "
    "holder_attributed/contested/unknown. This covers ALL attitudes, including known "
    "and uncertain. Both proposition IDs must name the same separately declared "
    "instance_graph.proposition_contents record with predicate, endpoints/roles, "
    "temporal_content and citations. An assertion is not that declaration. "
    "A scope/holder needs positive evidence; confidence or direct narration alone "
    "does not establish a holder attitude. Empty proposition_contents is legitimate "
    "only when no content is referenced. Do not fabricate missing content or remove "
    "evidence-supported attribution to bypass validation. Reconstruct the full answer "
    "from unchanged evidence; the previous answer below is untrusted model output, "
    "not evidence. Return the whole corrected JSON, not a patch."
)


def schema_guide(schema: Mapping[str, Any]) -> str:
    """Readable complete field/type grammar, not an example answer or tuple legend."""

    def show(node):
        if "$ref" in node:
            return node["$ref"].rsplit("/", 1)[-1]
        if "anyOf" in node:
            return "(" + " | ".join(show(v) for v in node["anyOf"]) + ")"
        if "const" in node:
            return repr(node["const"])
        if "enum" in node:
            return " | ".join(repr(v) for v in node["enum"])
        kind = node.get("type")
        if kind == "object":
            required = node.get("required", ())
            return (
                "{"
                + ", ".join(
                    k + ("" if k in required else "?") + ": " + show(v)
                    for k, v in node["properties"].items()
                )
                + "}"
            )
        if kind == "array":
            return (
                f"[{show(node.get('items', {}))}]; length {node.get('minItems', 0)}"
                f"..{node.get('maxItems', '*')}"
            )
        if "pattern" in node:
            return (
                "local-ID"
                if node["pattern"] == LOCAL_ID_PATTERN
                else f"{kind} matching {node['pattern']}"
            )
        if kind in {"integer", "number"}:
            return (
                str(kind)
                + (f" >= {node['minimum']}" if "minimum" in node else "")
                + (f" <= {node['maximum']}" if "maximum" in node else "")
            )
        return str(kind or "value")

    return "\n".join(
        [
            "All fields required except ?; null is explicit absence; [] is an empty list. "
            "No undeclared fields. Alternatives separated by | are exclusive when tagged. "
            "local-ID: a new n-prefixed identifier matching " + LOCAL_ID_PATTERN + ".",
            "Output = " + show(schema),
            *(name + " = " + show(value) for name, value in schema["$defs"].items()),
        ]
    )


@dataclass(frozen=True)
class ExecutionFacts:
    """Observed execution facts, never model-authored or backdated decision times."""

    request_hash: str
    response_hash: str
    generation_started_at: datetime
    generation_completed_at: datetime
    input_tokens: int
    output_tokens: int

    def __post_init__(self):
        import re

        if not all(
            re.fullmatch(r"[0-9a-f]{64}", h) for h in (self.request_hash, self.response_hash)
        ):
            raise ValueError("execution identities must be SHA-256")
        if (
            any(
                t.tzinfo is None or t.utcoffset() is None
                for t in (self.generation_started_at, self.generation_completed_at)
            )
            or self.generation_completed_at < self.generation_started_at
        ):
            raise ValueError("observed execution interval must be aware and ordered")
        if any(type(x) is not int or x < 0 for x in (self.input_tokens, self.output_tokens)):
            raise ValueError("measured tokens must be nonnegative integers")


@dataclass(frozen=True)
class AdaptedSemanticDraft:
    draft: OntologyDraft
    provenance: Mapping[str, Any]


def reconstruct(
    generated: Mapping[str, Any],
    *,
    evidence: Sequence[EvidenceRecord],
    upper: UpperOntology,
    execution: ExecutionFacts,
    small: bool = False,
) -> AdaptedSemanticDraft:
    """Validate representation first; copy all semantics exactly or reject.

    This is NOT scientific acceptance. The unchanged structural, temporal,
    capability, timing, and grounding validators still run on the resulting draft.
    """
    schema = semantic_schema(evidence, upper, small=small)
    Draft202012Validator(schema).validate(generated)
    value = copy.deepcopy(dict(generated))
    semantic_hash = canonical_sha256(generated)
    derived = []

    def add(target: dict, key: str, val: Any, path: str, source: str):
        if key in target:
            raise ValueError("runtime cannot overwrite generated content")
        target[key] = val
        derived.append({"path": path + "/" + key, "source": source, "value": val})

    for i, decision in enumerate(value["decisions"]):
        add(
            decision,
            "decided_at",
            execution.generation_completed_at.isoformat(),
            f"/decisions/{i}",
            "observed generation completion (not claimed internal decision instant)",
        )
    source_by_id = {e.evidence_id: e for e in evidence}
    graph = value["instance_graph"]
    for collection in ("assertions", "proposition_contents"):
        for obj in graph[collection]:
            obj.pop("form")  # purely representational; bindings already explicit
    for i, assertion in enumerate(graph["assertions"]):
        for j, citation in enumerate(assertion["provenance"]):
            source = source_by_id[citation["evidence_id"]].provenance
            path = f"/instance_graph/assertions/{i}/provenance/{j}"
            for key, val, origin in (
                (
                    "provenance_id",
                    f"prov-{semantic_hash}-{i}-{j}",
                    "payload hash and array indices",
                ),
                ("extraction_method", INTERFACE_REVISION, "runtime adapter revision"),
                ("locator", source.locator, "exact frozen cited-source provenance.locator"),
                (
                    "source_artifact_hash",
                    source.source_artifact_hash,
                    "exact frozen cited-source provenance.source_artifact_hash",
                ),
            ):
                add(citation, key, val, path, origin)
    nodes = len(graph["entities"]) + len(graph["events"])
    assertions = len(graph["assertions"])
    add(
        value,
        "budget_accounting",
        {
            "nodes_used": nodes,
            "assertions_used": assertions,
            "display_nodes_used": nodes,
            "display_assertions_used": assertions,
            "input_tokens": execution.input_tokens,
            "output_tokens": execution.output_tokens,
        },
        "",
        "exact graph counts; diagnostic displays all objects; observed usage, no clipping",
    )
    draft = OntologyDraft.model_validate(value)
    return AdaptedSemanticDraft(
        draft=draft,
        provenance={
            "interface_revision": INTERFACE_REVISION,
            "diagnostic_only": True,
            "generated_semantic_hash": semantic_hash,
            "generation_schema_hash": canonical_sha256(schema),
            "canonical_draft_hash": draft.content_hash,
            "request_hash": execution.request_hash,
            "response_hash": execution.response_hash,
            "generation_interval": [
                execution.generation_started_at.isoformat(),
                execution.generation_completed_at.isoformat(),
            ],
            "source_evidence_hashes": {e.evidence_id: e.content_hash for e in evidence},
            "runtime_fields": derived,
            "record_metadata": {
                "schema_version": SCHEMA_VERSION,
                "content_hash": "canonical SHA-256 at each immutable record",
            },
            "canonical_absence_defaults": (
                "inactive temporal fields and the unchosen assertion form; "
                "null/empty only; no scientific values filled"
            ),
            "removed_representation_tags": "assertion/proposition form (binary or nary)",
            "normalization": "canonical record string-edge whitespace normalization only",
            "scientific_validation": "not performed by adapter",
        },
    )
