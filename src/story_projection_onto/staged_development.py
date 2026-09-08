"""Three explicit development stages; lossless assembly, no semantic extraction."""

from __future__ import annotations

import copy
import json
import math
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

PROTOCOL = "staged-development-v1"
DESCRIPTION_FIELDS = ("description", "description_assertion_ids")
WHY_FIELDS = ("why_matters", "why_matters_evidence_ids")


def policy(root=None):
    cfg = read((root or Path.cwd()) / "configs/study/staged_development.json")
    exact = {
        "historical_actual_seconds": 6716.108081,
        "amendment_baseline_actual_seconds": 8200.425164,
        "maximum_service_starts": 6,
        "maximum_attempts": 22,
        "maximum_global_seconds": 10316.108081,
        "maximum_additional_seconds": 3600,
    }
    if any(cfg[k] != v for k, v in exact.items()):
        raise ValueError("staged authority differs from explicit amendment")

    def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
        if not math.isfinite(actual) or actual < 8200.425164 or starts < 4 or attempts < 7:
            raise ValueError("historical staged accounting missing")
        if starts + int(starting) > 6 or attempts + int(generating) > 22:
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
        STARTS=6,
        ATTEMPTS=22,
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
    full = compact_schema(
        reference_translation(
            stage_schema(evidence, config.upper_ontology, budgets, stage, a, b),
            mapping,
            schema=True,
        )
    )
    intro = (
        "Construct before query reveal: no user context is supplied."
        if kind == "c1"
        else (
            "Construct after this context reveal from evidence; "
            "no C1 or other-context ontology is available."
        )
    )
    guide = vocabulary_guide(full)
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
    pairs = read(root / "configs/study/staged_development.json")["allocation_ladder"]
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


def validate_stage(name, value, request):
    from .development_demo import development_validation_schema

    Draft202012Validator(development_validation_schema(request.output_schema)).validate(value)
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
