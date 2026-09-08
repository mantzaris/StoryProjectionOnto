"""Three explicit development stages; lossless assembly, no semantic extraction."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator

from .contracts import ConditionName, EvidenceRecord, canonical_sha256
from .development_adapter import DevelopmentConstructionConfiguration
from .development_demo import (
    aliases,
    backend_generation_schema,
    budget_schema,
    compact_schema,
    evidence_text,
    read,
    reference_translation,
    sources,
    vocabulary_guide,
    without_admin,
)
from .gpu_runtime import ChatMessage, GuidedJSONRequest
from .llm import DecodingManifest, DecodingPass, PackingReport, PackingSection
from .nested_semantic_candidate import candidate_schema
from .semantic_generation import _object

PROTOCOL = "staged-development-bounded-aux-v3"
DESCRIPTION_FIELDS = ("description", "description_assertion_ids")
WHY_FIELDS = ("why_matters", "why_matters_evidence_ids")


def policy(root=None):
    cfg = read((root or Path.cwd()) / "configs/study/staged_development.json")
    exact = {
        "historical_actual_seconds": 6716.108081,
        "amendment_baseline_actual_seconds": 8983.249948,
        "maximum_service_starts": 8,
        "maximum_attempts": 19,
        "maximum_global_seconds": 10316.108081,
        "maximum_additional_seconds": 3600,
    }
    if any(cfg[k] != v for k, v in exact.items()):
        raise ValueError("staged authority differs from explicit amendment")

    def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
        if not math.isfinite(actual) or actual < 8983.249948 or starts < 6 or attempts < 11:
            raise ValueError("historical staged accounting missing")
        if starts + int(starting) > 8 or attempts + int(generating) > 19:
            raise ValueError("staged start/reservation limit")
        end = actual + seconds + 60
        if end > 10316.108081 - 5 or end > 33660 or end >= 36000:
            raise TimeoutError("staged allowance must protect shutdown")
        return {
            "actual": actual,
            "envelope_end": end,
            "development_only": True,
            "complete_forecast_exception": True,
            "ordinary_study_authorized": False,
        }

    return SimpleNamespace(
        BASELINE=6716.108081,
        BLOCK_ID=cfg["amendment_id"],
        ALLOWANCE=3600,
        STARTS=8,
        ATTEMPTS=19,
        GENERATION=180,
        admit=admit,
        config=cfg,
    )


def prune(schema):
    """Remove unreachable definitions, never reachable semantic fields."""
    schema = copy.deepcopy(schema)
    used = set()

    def visit(n):
        if isinstance(n, dict):
            if "$ref" in n:
                name = n["$ref"].split("/")[-1]
                if name not in used:
                    used.add(name)
                    visit(schema["$defs"][name])
            for k, v in n.items():
                if k != "$defs":
                    visit(v)
        elif isinstance(n, list):
            for v in n:
                visit(v)

    visit(schema)
    schema["$defs"] = {k: v for k, v in schema["$defs"].items() if k in used}
    return schema


def inventory(a, b=None):
    groups = {"nS": [a["local_schema"]["schema_id"]]}
    for prefix, rows, key in (
        ("nT", a["local_schema"]["contextual_types"], "type_id"),
        ("nR", a["local_schema"]["predicates"], "predicate_id"),
        ("nE", a["entities"], "entity_id"),
        ("nV", a["events"], "event_id"),
    ):
        groups[prefix] = [r[key] for r in rows]
    if b is not None:
        groups["nA"] = [r["assertion_id"] for r in b["assertions"]]
    flat = [x for v in groups.values() for x in v]
    if len(flat) != len(set(flat)):
        raise ValueError("duplicate declared local identifier")
    return groups


def allowed(values):
    return {"enum": values} if values else {"not": {}}


def stage_schema(evidence, upper, budgets, name, a=None, b=None):
    full = budget_schema(candidate_schema(evidence, upper), budgets)
    defs = full["$defs"]
    root = full["properties"]
    if name == "A":
        for record in ("Entity", "Event"):
            for key in DESCRIPTION_FIELDS:
                defs[record]["properties"].pop(key)
                defs[record]["required"].remove(key)
        graph = copy.deepcopy(defs["InstanceGraph"])
        for branch in graph["anyOf"]:
            for key in ("assertions", "unasserted_contents"):
                branch["properties"].pop(key)
                branch["required"].remove(key)
            branch["properties"].update(
                {
                    "local_schema": root["local_schema"],
                    "contextual_interpretation": root["contextual_interpretation"],
                }
            )
            branch["required"] += ["local_schema", "contextual_interpretation"]
        schema = graph
    elif name == "B":
        if a is None:
            raise ValueError("B requires actual A")
        ids = inventory(a)
        for branch in defs["QualifiedAssertion"]["anyOf"]:
            for key in WHY_FIELDS:
                branch["properties"].pop(key)
                branch["required"].remove(key)
        graph = defs["InstanceGraph"]["anyOf"][0]["properties"]
        schema = _object({k: graph[k] for k in ("assertions", "unasserted_contents")})

        # Only generated destinations are constrained. New assertion/content IDs remain authored.
        def bind(n, key=""):
            if isinstance(n, dict):
                if key in {"subject_id", "object_id", "holder_id"} and n.get("type") == "string":
                    n.clear()
                    n.update(allowed(ids["nE"] + ids["nV"]))
                elif key == "predicate_id" and n.get("type") == "string":
                    n.clear()
                    n.update(allowed(ids["nR"]))
                for k, v in list(n.items()):
                    bind(v, k)
            elif isinstance(n, list):
                for v in n:
                    bind(v, key)

        bind(defs)
    elif name == "C":
        if a is None or b is None:
            raise ValueError("C requires actual A and B")
        ids = inventory(a, b)
        all_ids = [x for v in ids.values() for x in v]
        # Exact per-record objects avoid speculative IDs and duplicate annotation coverage.
        nodes = {}
        for group, field, prefix in (("entities", "entity_id", "nE"), ("events", "event_id", "nV")):
            for r in a[group]:
                node = defs["Entity" if prefix == "nE" else "Event"]["properties"]
                prop = {k: copy.deepcopy(node[k]) for k in DESCRIPTION_FIELDS}
                prop["description_assertion_ids"]["items"] = allowed(ids["nA"])
                prop["description_assertion_ids"]["maxItems"] = len(ids["nA"])
                nodes[r[field]] = _object(prop)
        assertions = {}
        template = defs["QualifiedAssertion"]["anyOf"][0]["properties"]
        for r in b["assertions"]:
            p = {k: copy.deepcopy(template[k]) for k in WHY_FIELDS}
            p["why_matters_evidence_ids"]["items"] = allowed(r["evidence_ids"])
            p["why_matters_evidence_ids"]["maxItems"] = len(r["evidence_ids"])
            assertions[r["assertion_id"]] = _object(p)
        content_addresses = [
            {"content_of_assertion": r["assertion_id"]}
            for r in b["assertions"]
            if r["epistemic_scope"] is not None
        ]
        content_addresses += [{"unasserted_content": r["key"]} for r in b["unasserted_contents"]]
        candidates = [m.candidate_id for e in evidence for m in e.mention_candidates]
        candidates += [v.candidate_id for e in evidence for v in e.event_candidates]
        candidates += [r.candidate_id for e in evidence for r in e.relation_phrase_candidates]
        candidates += [t.clue_id for e in evidence for t in e.temporal_clues]
        targets = {
            "merge": ids["nE"],
            "split": ids["nE"],
            "event_reification": ids["nV"],
            "contextual_type": ids["nT"],
            "schema_relation": ids["nS"] + ids["nR"],
            "abstraction": ids["nS"] + ids["nT"] + ids["nR"] + ids["nE"] + ids["nV"],
            "temporal_qualification": ids["nA"],
            "epistemic_qualification": ids["nA"] + content_addresses,
        }
        decisions = []
        for branch in defs["OntologyDecision"]["anyOf"]:
            props = branch["properties"]
            op = props["operator"]["const"]
            values = targets.get(op, all_ids + content_addresses)
            if op in ("selection", "compression", "supported_description"):
                values = []
            if not values and props["created_object_ids"].get("minItems", 0):
                continue
            props["created_object_ids"].update(items=allowed(values), maxItems=len(values))
            props["input_object_ids"].update(
                items=allowed(all_ids + candidates + content_addresses),
                maxItems=len(all_ids + candidates + content_addresses),
            )
            props["removed_object_ids"].update(items=allowed(candidates), maxItems=len(candidates))
            decisions.append(branch)
        defs["OntologyDecision"] = {"anyOf": decisions}
        schema = _object(
            {
                "node_descriptions": _object(nodes),
                "assertion_descriptions": _object(assertions),
                "decisions": root["decisions"],
                "omissions": root["omissions"],
                "uncertainty_and_abstentions": root["uncertainty_and_abstentions"],
            }
        )
    else:
        raise ValueError("unknown stage")
    schema["$defs"] = defs
    return prune(schema)


def auxiliary_bounds(schema, limits):
    """Explicit exploratory ceilings, not altered registered object budgets.

    Bounds constrain authoring, never trim generated records. Required fields,
    meanings and all evidence in the input remain intact. Uniqueness remains a
    post-check because the pinned request validator rejects uniqueItems.
    """
    schema = copy.deepcopy(schema)

    def visit(node, key=""):
        if isinstance(node, dict):
            if node.get("type") == "array":
                cap = None
                if key in limits:
                    cap = limits[key]
                elif key.endswith("_ids") or key in {"provenance", "role_names"}:
                    cap = limits["reference_list"]
                if cap is not None:
                    node["maxItems"] = min(cap, node.get("maxItems", cap))
            if (
                node.get("type") == "string"
                and not any(k in node for k in ("enum", "const", "pattern"))
                and not key.endswith(("_id", "_ids"))
            ):
                cap = (
                    limits["interpretation_characters"]
                    if key == "contextual_interpretation"
                    else limits["label_characters"]
                    if key in {"label", "aliases", "role_names", "contextual_role"}
                    else limits["prose_characters"]
                )
                node["maxLength"] = min(cap, node.get("maxLength", cap))
            for k, v in node.items():
                if k not in {"enum", "const"}:
                    visit(v, key if k in {"items", "anyOf", "oneOf"} else k)
        elif isinstance(node, list):
            for item in node:
                visit(item, key)

    visit(schema)
    return schema


def split_fixture(value):
    """Authored-fixture inverse for CPU round trips; never used to create model requests."""
    a = {k: copy.deepcopy(value[k]) for k in ("local_schema", "contextual_interpretation")}
    c = {
        k: copy.deepcopy(value[k])
        for k in ("decisions", "omissions", "uncertainty_and_abstentions")
    }
    c.update(node_descriptions={}, assertion_descriptions={})
    for group, field in (("entities", "entity_id"), ("events", "event_id")):
        a[group] = copy.deepcopy(value["instance_graph"][group])
        for r in a[group]:
            c["node_descriptions"][r[field]] = {k: r.pop(k) for k in DESCRIPTION_FIELDS}
    b = {
        k: copy.deepcopy(value["instance_graph"][k]) for k in ("assertions", "unasserted_contents")
    }
    for r in b["assertions"]:
        c["assertion_descriptions"][r["assertion_id"]] = {k: r.pop(k) for k in WHY_FIELDS}
    return a, b, c


def stage_c_guide(schema, a, b):
    """Factor repeated per-record annotations in the HUMAN guide only.

    The effective grammar still requires every exact generated ID. No previous
    records or source text are removed, summarized, or inferred.
    """
    targets = {}
    for branch in schema["$defs"]["OntologyDecision"]["anyOf"]:
        props = branch["properties"]
        targets[props["operator"]["const"]] = props["created_object_ids"]["items"].get("enum", [])
    return (
        "Return exactly {node_descriptions, assertion_descriptions, decisions, omissions, "
        "uncertainty_and_abstentions}. node_descriptions is an object keyed by EVERY actual "
        "entity/event ID in A, once each. Each value is {description: supported string <=180 "
        "characters, description_assertion_ids: array of <=4 actual assertion IDs involving "
        "that node}. assertion_descriptions is keyed by EVERY assertion ID in B, once each. "
        "Each value is {why_matters: supported string <=180 characters, "
        "why_matters_evidence_ids: <=4 evidence IDs drawn from that assertion's citations}. "
        "No extra keys. Empty maps when there are no corresponding records.\n"
        "decisions is an array <=8 of {decision_id: new nD followed by 1..4 digits starting "
        "nonzero, operator: one of the operation names below, evidence_ids: <=4 supplied "
        "evidence IDs, rationale: supported string <=180 characters, input_object_ids: <=4 "
        "actual graph/schema IDs or supplied candidate/clue IDs, created_object_ids: <=4 "
        "actual destinations allowed for that operation below, removed_object_ids: <=4 "
        "supplied candidate IDs}. Every field is required; arrays may be empty except "
        "evidence_ids and substantive operators' created_object_ids require at least one. "
        "Typed IDs: nS schema, nT type, nR predicate, nE entity, nV event, nA assertion. "
        "A proposition destination is {content_of_assertion: actual attributed assertion ID} "
        "or {unasserted_content: actual standalone content key}, not a guessed ID. "
        "Allowed creation destinations by operation (empty means created_object_ids=[]):\n"
        + json.dumps(targets, separators=(",", ":"))
        + "\nomissions is an array <=4 of {evidence_id: supplied evidence ID, reason: string "
        "<=180 characters, confidence: number 0..1}. uncertainty_and_abstentions is an "
        "array <=4 of strings <=180 characters. All reference arrays contain distinct IDs. "
        "No timestamps, token counts or administrative hashes are model-authored."
    )


def assemble(a, b, c):
    ids = inventory(a, b)
    if set(c["node_descriptions"]) != set(ids["nE"] + ids["nV"]):
        raise ValueError("Stage C node coverage differs from A")
    if set(c["assertion_descriptions"]) != set(ids["nA"]):
        raise ValueError("Stage C assertion coverage differs from B")
    graph = {
        "entities": copy.deepcopy(a["entities"]),
        "events": copy.deepcopy(a["events"]),
        **copy.deepcopy(b),
    }
    for group, field in (
        ("entities", "entity_id"),
        ("events", "event_id"),
        ("assertions", "assertion_id"),
    ):
        keys = WHY_FIELDS if group == "assertions" else DESCRIPTION_FIELDS
        annotations = c["assertion_descriptions" if group == "assertions" else "node_descriptions"]
        for record in graph[group]:
            addition = annotations[record[field]]
            if set(addition) != set(keys) or set(record) & set(addition):
                raise ValueError("Stage C tried to overwrite semantic content")
            record.update(copy.deepcopy(addition))
    return {
        "local_schema": copy.deepcopy(a["local_schema"]),
        "contextual_interpretation": a["contextual_interpretation"],
        "instance_graph": graph,
        **{
            k: copy.deepcopy(c[k])
            for k in ("decisions", "omissions", "uncertainty_and_abstentions")
        },
    }


def prepare(root, kind, stage, tokenizer, manifest, *, prior=None, feedback=None, previous=None):
    prior = prior or {}
    _, unit, neutral = sources(root)
    evidence = tuple(EvidenceRecord.model_validate(e) for e in neutral["evidence"])
    mapping = aliases(evidence)
    config = DevelopmentConstructionConfiguration.load(
        root / "configs/study/development_construction.json"
    )
    budgets = (
        config.preconstruction_budgets
        if kind == "c1"
        else config.projection_budgets_by_unit[unit["unit_id"]]
    )
    # Restore readable aliases only for schema construction, not generated prose.
    reverse = {v: k for k, v in mapping.items()}
    a = reference_translation(prior.get("A"), reverse)
    b = reference_translation(prior.get("B"), reverse)
    cfg = read(root / "configs/study/staged_development.json")
    full = auxiliary_bounds(
        compact_schema(
            reference_translation(
                stage_schema(evidence, config.upper_ontology, budgets, stage, a, b),
                mapping,
                schema=True,
            )
        ),
        cfg["auxiliary_limits"],
    )
    intro = (
        "Construct before query reveal: no user context is supplied."
        if kind == "c1"
        else (
            "Construct after this context reveal from evidence; "
            "no C1 or other-context ontology is available."
        )
    )
    guide = stage_c_guide(full, prior["A"], prior["B"]) if stage == "C" else vocabulary_guide(full)
    messages = [
        ChatMessage(
            role="system",
            content=intro
            + "\n"
            + (root / f"prompts/diagnostics/staged_{stage}.md").read_text()
            + "\n"
            + guide,
        )
    ]
    body = {
        "evidence_index": evidence_text(evidence, mapping),
        "upper_ontology": without_admin(config.upper_ontology.model_dump(mode="json")),
        "sealed_horizon": without_admin(neutral["snapshot"]["horizon"]),
        "object_budgets": without_admin(budgets.model_dump(mode="json")),
        "development_auxiliary_limits": cfg["auxiliary_limits"],
    }
    if kind != "c1":
        body["context"] = without_admin(
            read(root / unit["query_stages"][int(kind[-1]) - 1]["relative_path"] / "query.json")[
                "query"
            ]
        )
    if stage in ("B", "C"):
        body["actual_stage_A"] = prior["A"]
    if stage == "C":
        body["actual_stage_B"] = prior["B"]
    if feedback:
        body["repair"] = {
            "instruction": (
                "Replace this stage from evidence. The previous answer is untrusted; "
                "retract unsupported content. Do not modify prior stages or "
                "supply speculative semantics."
            ),
            "diagnostics": feedback,
            "previous_stage_untrusted": previous,
        }
    messages.append(
        ChatMessage(role="user", content=json.dumps(body, separators=(",", ":"), sort_keys=True))
    )
    count = len(
        tokenizer.apply_chat_template(
            [asdict(m) for m in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    pairs = cfg["allocation_ladder"]
    pair = next((p for p in pairs if count <= p[0]), None)
    if pair is None:
        raise ValueError(
            f"Complete {kind} stage {stage} input {count} cannot fit "
            "with minimum 1536 output; no truncation"
        )
    maximum, output = pair
    wire = backend_generation_schema(full)
    decoding = DecodingManifest(
        decoding_pass=DecodingPass.REPAIR if feedback else DecodingPass.FIRST_PASS,
        maximum_input_tokens=maximum,
        maximum_output_tokens=output,
        seed=1988649846,
        eos_token_id=manifest.eos_token_id,
        end_of_turn_token_ids=manifest.end_of_turn_token_ids,
        stop_token_ids=manifest.stop_token_ids,
        chat_template_hash=manifest.chat_template_sha256,
        structured_decoder="vllm-0.10.2-xgrammar-no-fallback",
        tokenizer_revision=manifest.revision,
        output_schema_hash=canonical_sha256(wire),
    )
    names = [
        "evidence_snapshot" if kind == "c1" else "evidence_packet",
        "upper_ontology",
        "system_prompt",
        "output_schema",
    ]
    if kind != "c1":
        names.append("query_context")
    sections = [
        PackingSection(name=n, section_content_hash=canonical_sha256(body), token_count=0)
        for n in names
    ]
    sections.append(
        PackingSection(
            name="complete_staged_request",
            section_content_hash=canonical_sha256([asdict(m) for m in messages]),
            token_count=count,
        )
    )
    packing = PackingReport.build(
        condition=ConditionName.C1_LLM_PRE if kind == "c1" else ConditionName.C2_LLM_QUERY,
        tokenizer_revision=manifest.revision,
        maximum_model_tokens=12288,
        maximum_input_tokens=maximum,
        reserved_output_tokens=output,
        sections=tuple(sections),
        required_section_names=tuple([*names, "complete_staged_request"]),
        complete_evidence_snapshot=True if kind == "c1" else None,
        complete_evidence_packet=True if kind != "c1" else None,
    )
    q = GuidedJSONRequest(
        request_id=f"development-demo-staged-{kind}-{stage}" + ("-repair" if feedback else ""),
        model_name="qwen3-8b-awq-fallback",
        condition=packing.condition,
        messages=tuple(messages),
        output_schema=wire,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=count,
        stream_response=True,
    )
    return q, evidence, mapping, budgets


def repair_owner(stage, feedback):
    """Route only explicit field ownership, never infer a missing semantic choice."""
    if stage != "C":
        return stage
    owners = set()
    for item in feedback:
        if item["category"] in {"unsupported_attribution", "unsupported_intrinsic_precision"}:
            owners.add("B")
            continue
        for path in item.get("paths", []):
            if any(k in path for k in ("description", "why_matters", "decisions", "omissions")):
                owners.add("C")
            elif "assertions" in path or "propositions" in path:
                owners.add("B")
            elif "local_schema" in path or "entities" in path or "events" in path:
                owners.add("A")
            else:
                return None
    # More than one owning stage requires more than the one permitted semantic
    # repair, not an adapter patch or a knowingly irrelevant Stage C retry.
    return next(iter(owners)) if len(owners) == 1 else None


def prefix_duplicate_feedback(text):
    """Only provable list-uniqueness defects in COMPLETE received A members.

    Does not reconstruct incomplete JSON, infer semantic identity, or insert
    reference answers. Full bytes remain in restricted transport lineage.
    """
    paths = []
    duplicate_count = 0
    for field in ("entities", "events"):
        match = re.search(r'"' + field + r'"\s*:\s*\[', text)
        if not match:
            continue
        rest = text[match.end() :].lstrip()
        index = 0
        while rest.startswith("{"):
            try:
                record, end = json.JSONDecoder().raw_decode(rest)
            except json.JSONDecodeError:
                break
            for key, values in record.items():
                if (
                    key.endswith("_ids")
                    and isinstance(values, list)
                    and all(isinstance(v, str) for v in values)
                ):
                    count = Counter(values)
                    excess = sum(n - 1 for n in count.values())
                    if excess:
                        paths.append(f"/{field}/{index}/{key}")
                        duplicate_count += excess
            rest = rest[end:].lstrip()
            if not rest.startswith(","):
                break
            rest = rest[1:].lstrip()
            index += 1
    if not paths:
        return []
    return [
        {
            "category": "duplicate_reference_list",
            "paths": paths,
            "generated_duplicate_occurrences": duplicate_count,
            "constraint": "Each reference list must contain unique IDs. The previous incomplete "
            "Stage A repeated references at these paths. Regenerate Stage A from "
            "the complete evidence, retaining only references you judge support "
            "each modeled identity. Do not copy the entire index into each record. "
            "Keep all required fields and scientific content; budgets are unchanged. "
            "This is a replacement, not a request to complete the truncated suffix.",
        }
    ]


def validate_stage(name, value, request, *, mechanical=False):
    from .development_demo import development_validation_schema

    # Removing ONLY uniqueness for mechanical inspection must happen before
    # evaluating alternatives. An anyOf parent can wrap a uniqueness-only leaf.
    # Strict post-validation retains every uniqueness check and historical fail.
    schema = development_validation_schema(request.output_schema)
    if mechanical:
        schema = backend_generation_schema(schema)
    Draft202012Validator(schema).validate(value)
    if name == "A":
        ids = inventory(value)
        types = set(ids["nT"])
        for r in value["entities"] + value["events"]:
            if r["contextual_type_id"] not in types:
                raise ValueError("A object type is undeclared")
        for r in value["local_schema"]["predicates"]:
            if not set(r["domain_type_ids"] + r["range_type_ids"]) <= types:
                raise ValueError("A predicate type is undeclared")
    if name == "B":
        ids = [r["assertion_id"] for r in value["assertions"]]
        if len(ids) != len(set(ids)):
            raise ValueError("B assertion IDs must be unique")


def mechanical_graph_status(value):
    """Inspection eligibility only; no canonical/scientific invariant relaxed.

    Check explicit local destinations without changing a single authored value.
    Semantic type compatibility and temporal truth are NOT identity resolution.
    """
    errors = []
    try:
        graph = value["instance_graph"]
        a = {"local_schema": value["local_schema"], **{k: graph[k] for k in ("entities", "events")}}
        ids = inventory(a, graph)
        nodes = set(ids["nE"] + ids["nV"])
        predicates = set(ids["nR"])
        for i, assertion in enumerate(graph["assertions"]):
            content = assertion["content"]
            if content["predicate_id"] not in predicates:
                errors.append(f"assertions/{i}/content/predicate_id: undeclared")
            endpoints = (
                [content["subject_id"], content["object_id"]]
                if content["form"] == "binary"
                else [role["object_id"] for role in content["roles"]]
            )
            for endpoint in endpoints:
                if endpoint not in nodes:
                    errors.append(f"assertions/{i}/content: undeclared endpoint {endpoint}")
            scope = assertion["epistemic_scope"]
            if scope and scope["holder_id"] not in nodes:
                errors.append(f"assertions/{i}/epistemic_scope: undeclared holder")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return {
        "mechanically_usable": not errors,
        "structural_blockers": errors,
        "policy": "exploratory-inspection-v1; no claim of canonical or scientific acceptance",
    }


def verify_auxiliary_grammar(compiler, request, authored_a):
    """Exercise the effective backend grammar, not only standalone compilation."""
    import xgrammar

    compiled = compiler.compile_json_schema(json.dumps(request.output_schema), any_whitespace=False)

    # XGrammar serializes object keys in schema order. Preserve fixture values
    # but order their keys identically before testing capacity constraints.
    def ordered(value, schema):
        while "$ref" in schema or "anyOf" in schema:
            if "$ref" in schema:
                schema = request.output_schema["$defs"][schema["$ref"].split("/")[-1]]
            else:
                schema = next(
                    b
                    for b in schema["anyOf"]
                    if Draft202012Validator(
                        {**b, "$defs": request.output_schema["$defs"]}
                    ).is_valid(value)
                )
        if isinstance(value, dict):
            return {k: ordered(value[k], s) for k, s in schema["properties"].items() if k in value}
        if isinstance(value, list):
            return [ordered(v, schema["items"]) for v in value]
        return value

    authored_a = ordered(authored_a, request.output_schema)
    controls = {"valid_authored": copy.deepcopy(authored_a)}
    for field, count in (("contextual_types", 7), ("predicates", 9)):
        bad = copy.deepcopy(authored_a)
        bad["local_schema"][field] = [copy.deepcopy(bad["local_schema"][field][0])] * count
        controls[field + "_overflow"] = bad
    bad = copy.deepcopy(authored_a)
    bad["entities"][0]["evidence_ids"] = ["e1"] * 5
    controls["reference_overflow"] = bad
    bad = copy.deepcopy(authored_a)
    bad["local_schema"]["predicates"][0]["definition"] = "x" * 181
    controls["prose_overflow"] = bad
    result = {
        name: xgrammar.GrammarMatcher(compiled).accept_string(json.dumps(v))
        for name, v in controls.items()
    }
    if result != {name: name == "valid_authored" for name in controls}:
        matcher = xgrammar.GrammarMatcher(compiled)
        serialized = json.dumps(authored_a)
        first_rejection = None
        for i, char in enumerate(serialized):
            if not matcher.accept_string(char):
                first_rejection = {"offset": i, "around": serialized[max(0, i - 60) : i + 60]}
                break
        raise ValueError(
            f"Effective auxiliary grammar controls failed: {result}; {first_rejection}"
        )
    return {"accepted_by_effective_grammar": result, "authored_cpu_controls_only": True}
