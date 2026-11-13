"""Authorized exploratory workload: named nested semantics, no production adoption.

Request preparation imports no scorer and never receives a reference ontology.
The existing capacity controller/guardian owns GPU execution and shutdown.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

from .contracts import ConditionName, EvidenceRecord, canonical_sha256
from .development_adapter import DevelopmentConstructionConfiguration
from .gpu_runtime import ChatMessage, GuidedJSONRequest
from .llm import DecodingManifest, DecodingPass, PackingReport, PackingSection
from .nested_semantic_candidate import candidate_schema, reconstruct_candidate

CONFIG = "configs/study/preliminary_development_demo.json"


def budget_schema(schema, budgets):
    """Encode existing ceilings/operation destinations; no semantic budget reduction.

    Mixed-kind count constraints remain canonical checks where XGrammar cannot
    encode them. Schema/type/proposition counts have no registered ceiling.
    """
    schema = copy.deepcopy(schema)
    defs = schema["$defs"]
    graph = defs["InstanceGraph"]["properties"]
    for field in ("entities", "events"):
        graph[field]["maxItems"] = budgets.node_budget
    graph["assertions"]["maxItems"] = budgets.assertion_budget

    def arrays(node, key=""):
        if isinstance(node, list):
            for item in node:
                arrays(item, key)
        elif isinstance(node, dict):
            if node.get("type") == "array":
                item = node.get("items", {})
                if key.endswith("_ids"):
                    node["uniqueItems"] = True  # also post-validated if decoder ignores it
                    if item.get("enum"):
                        node["maxItems"] = len(item["enum"])
                    elif key == "description_assertion_ids":
                        node["maxItems"] = budgets.assertion_budget
            for k, v in node.items():
                arrays(v, k)

    arrays(schema)
    # Exact aggregate entity+event ceiling, without imposing a partition choice.
    original_graph = defs["InstanceGraph"]
    alternatives = []
    for entities in range(budgets.node_budget + 1):
        branch = copy.deepcopy(original_graph)
        branch["properties"]["entities"].update(minItems=entities, maxItems=entities)
        branch["properties"]["events"]["maxItems"] = budgets.node_budget - entities
        alternatives.append(branch)
    defs["InstanceGraph"] = {
        "anyOf": alternatives,
        "$comment": f"development_joint_node_budget={budgets.node_budget}",
    }
    original = defs["OntologyDecision"]
    targets = {
        "merge": ("nE", budgets.node_budget),
        "split": ("nE", budgets.node_budget),
        "event_reification": ("nV", budgets.node_budget),
        "contextual_type": ("nT", None),
        "schema_relation": ("nS|nR", None),
        "temporal_qualification": ("nA", budgets.assertion_budget),
    }
    variants = []
    for operator in defs["ConstructionOperator"]["enum"]:
        branch = copy.deepcopy(original)
        p = branch["properties"]
        p["operator"] = {"const": operator}
        created = p["created_object_ids"]
        if operator in targets:
            pattern, maximum = targets[operator]
            created["items"] = {"type": "string", "pattern": rf"^(?:{pattern})[1-9][0-9]{{0,3}}$"}
            created["minItems"] = 1
            if maximum is not None:
                created["maxItems"] = maximum
        elif operator in {"selection", "compression", "supported_description"}:
            created["maxItems"] = 0
        variants.append(branch)
    defs["OntologyDecision"] = {"anyOf": variants}
    return schema


def backend_generation_schema(schema):
    """Remove vLLM-unsupported uniqueness only; retain it in post-validation."""
    if isinstance(schema, dict):
        return {k: backend_generation_schema(v) for k, v in schema.items() if k != "uniqueItems"}
    if isinstance(schema, list):
        return [backend_generation_schema(v) for v in schema]
    return schema


def development_validation_schema(schema):
    """Restore the unchanged reference-list uniqueness contract after decoding."""
    value = copy.deepcopy(schema)

    def visit(node, key=""):
        if isinstance(node, dict):
            if node.get("type") == "array" and key.endswith("_ids"):
                node["uniqueItems"] = True
            for k, v in node.items():
                visit(v, k)
        elif isinstance(node, list):
            for item in node:
                visit(item, key)

    visit(value)
    return value


def read(path):
    return json.loads(Path(path).read_bytes())


def sources(root):
    manifest = read(root / "configs/study/development_call_manifest.json")
    unit = next(x for x in manifest["units"] if x["unit_id"] == "dev-unit-01")
    neutral = read(root / unit["neutral_evidence_stage"]["relative_path"] / "neutral_evidence.json")
    return manifest, unit, neutral


def aliases(evidence):
    result = {}
    for i, e in enumerate(evidence, 1):
        result[e.evidence_id] = f"e{i}"
    for prefix, field in (
        ("m", "mention_candidates"),
        ("v", "event_candidates"),
        ("r", "relation_phrase_candidates"),
        ("t", "temporal_clues"),
    ):
        for e in evidence:
            for item in getattr(e, field):
                identifier = getattr(item, "candidate_id", getattr(item, "clue_id", None))
                if identifier not in result:
                    result[identifier] = (
                        f"{prefix}{1 + sum(v.startswith(prefix) for v in result.values())}"
                    )
    return result


def reference_translation(value, mapping, *, schema=False, key=""):
    """Translate only declared reference fields/enums, never descriptive prose."""
    if isinstance(value, dict):
        return {
            k: reference_translation(v, mapping, schema=schema, key=k) for k, v in value.items()
        }
    if isinstance(value, list):
        return [reference_translation(v, mapping, schema=schema, key=key) for v in value]
    if isinstance(value, str) and (key.endswith(("_id", "_ids")) or (schema and key == "enum")):
        return mapping.get(value, value)
    return value


def without_admin(value):
    if isinstance(value, dict):
        return {
            k: without_admin(v)
            for k, v in value.items()
            if k not in {"content_hash", "schema_version"}
        }
    if isinstance(value, list):
        return [without_admin(v) for v in value]
    return value


def evidence_text(evidence, mapping):
    """Complete source prose + readable, query-blind index tables, no graph/gold."""
    lines = [
        "Evidence index: all candidates are provisional, not ontology assertions.",
        "Each record gives id, discourse, source/provenance confidence and full prose.",
        "M columns: candidate | surface | provisional type | aliases | coreference scores.",
        "V columns: candidate | trigger | participants | confidence.",
        "R columns: candidate | phrase | subject candidate | object candidate | confidence.",
        "T columns: clue | expression | relation | targets | confidence.",
        "Empty means absent/empty in the source index. "
        "Offsets/hashes/locators are retained by runtime.",
    ]

    def ref(v):
        if isinstance(v, (list, tuple)):
            return ",".join(mapping.get(x, x) for x in v)
        return mapping.get(v, v) if v is not None else ""

    for e in evidence:
        pos = e.discourse_position
        lines += [
            f"{mapping[e.evidence_id]} discourse={pos.passage_order},"
            f"{pos.sentence_order},{pos.token_order} "
            f"confidence={e.confidence},{e.provenance.confidence}",
            e.text,
        ]
        for m in e.mention_candidates:
            scores = {ref(k): v for k, v in m.coreference_scores.items()}
            lines.append(
                "M "
                + " | ".join(
                    map(
                        str,
                        (
                            ref(m.candidate_id),
                            m.surface,
                            m.provisional_type or "",
                            ref(m.alias_candidate_ids),
                            json.dumps(scores) if scores else "",
                        ),
                    )
                )
            )
        for v in e.event_candidates:
            lines.append(
                "V "
                + " | ".join(
                    map(
                        str,
                        (
                            ref(v.candidate_id),
                            v.trigger_surface,
                            ref(v.participant_mention_candidate_ids),
                            v.confidence,
                        ),
                    )
                )
            )
        for r in e.relation_phrase_candidates:
            lines.append(
                "R "
                + " | ".join(
                    map(
                        str,
                        (
                            ref(r.candidate_id),
                            r.surface_phrase,
                            ref(r.subject_mention_candidate_id),
                            ref(r.object_mention_candidate_id),
                            r.confidence,
                        ),
                    )
                )
            )
        for t in e.temporal_clues:
            lines.append(
                "T "
                + " | ".join(
                    map(
                        str,
                        (
                            ref(t.clue_id),
                            t.normalized_expression,
                            t.relation.value if t.relation else "",
                            ref(t.target_candidate_ids),
                            t.confidence,
                        ),
                    )
                )
            )
    return "\n".join(lines)


def compact_schema(schema):
    """Factor repeated enum vocabularies into definitions; same JSON language."""
    value = copy.deepcopy(schema)
    extra = {}
    seen = {}

    def visit(node):
        if isinstance(node, list):
            for x in node:
                visit(x)
        elif isinstance(node, dict):
            if "enum" in node and len(node["enum"]) > 10:
                identity = json.dumps(node, sort_keys=True)
                if identity not in seen:
                    name = f"SuppliedVocabulary{len(seen) + 1}"
                    seen[identity] = name
                    extra[name] = copy.deepcopy(node)
                node.clear()
                node["$ref"] = "#/$defs/" + seen[identity]
            else:
                for x in node.values():
                    visit(x)

    visit(value)
    value["$defs"].update(extra)
    return value


def vocabulary_guide(schema):
    """Name supplied index vocabularies instead of repeating every ID in prose.

    The effective grammar retains its exact enum. The prompt says every reference
    must occur in the supplied index and names the permitted record kinds.
    """
    value = copy.deepcopy(schema)
    legends = []
    kinds = {
        "e": "evidence heading",
        "m": "M mention row",
        "v": "V event-candidate row",
        "r": "R phrase row",
        "t": "T temporal-clue row",
    }
    for name, definition in value["$defs"].items():
        if name.startswith("SuppliedVocabulary") and all(
            re.fullmatch(r"[emvrt][1-9][0-9]*", v) for v in definition["enum"]
        ):
            prefixes = sorted({v[0] for v in definition["enum"]})
            legends.append(
                name
                + " must be an existing supplied ID from: "
                + ", ".join(kinds[p] for p in prefixes)
                + ". No invented supplied references."
            )
            definition.clear()
            definition.update(type="string")
    return field_guide(value) + "\n" + "\n".join(legends)


def field_guide(schema):
    """Complete named-field guide; factor repeated alternatives, not semantics."""
    identifiers = {}
    defs = schema["$defs"]
    named_shapes = {}
    if "Entity" in defs:
        named_shapes["Citations"] = defs["Entity"]["properties"]["evidence_ids"]
    if "OntologyDecision" in defs:
        decision = defs["OntologyDecision"]["anyOf"][0]["properties"]
        named_shapes.update(
            ObjectTarget=decision["removed_object_ids"]["items"],
            InputTarget=decision["input_object_ids"]["items"],
        )

    def show(n, defining=None):
        for name, shape in named_shapes.items():
            if name != defining and n == shape:
                return name
        if "$ref" in n:
            return n["$ref"].rsplit("/", 1)[-1]
        if "anyOf" in n:
            branches = n["anyOf"]
            if all(b.get("type") == "object" for b in branches):
                common = {
                    k: v
                    for k, v in branches[0]["properties"].items()
                    if all(b["properties"].get(k) == v for b in branches[1:])
                    and all(k in b.get("required", []) for b in branches)
                }
                if common:
                    rest = []
                    for b in branches:
                        c = copy.deepcopy(b)
                        c["properties"] = {
                            k: v for k, v in c["properties"].items() if k not in common
                        }
                        rest.append(c)
                    return (
                        show({"type": "object", "properties": common, "required": list(common)})
                        + " + ("
                        + " | ".join(show(b) for b in rest)
                        + ")"
                    )
            return "(" + " | ".join(show(b) for b in branches) + ")"
        if "const" in n:
            return repr(n["const"])
        if "enum" in n:
            return "|".join(map(repr, n["enum"]))
        kind = n.get("type")
        if kind == "object":
            return (
                "{"
                + ", ".join(
                    k + ("" if k in n.get("required", []) else "?") + ":" + show(v)
                    for k, v in n["properties"].items()
                )
                + "}"
            )
        if kind == "array":
            if n.get("maxItems") == 0:
                return "[]"
            return (
                "["
                + show(n["items"])
                + "]"
                + (
                    f"({n.get('minItems', 0)}..{n['maxItems']})"
                    if "maxItems" in n
                    else ("(1+)" if n.get("minItems") == 1 else "")
                )
                + (" unique" if n.get("uniqueItems") else "")
            )
        if "pattern" in n:
            pattern = n["pattern"]
            if pattern not in identifiers:
                identifiers[pattern] = "ID" + str(len(identifiers) + 1)
            return identifiers[pattern]
        if kind in {"integer", "number"}:
            return (
                kind
                + (f">={n['minimum']}" if "minimum" in n else "")
                + (f"<={n['maximum']}" if "maximum" in n else "")
            )
        return str(kind)

    root_view = schema
    root_note = ""
    if schema.get("$comment", "").startswith("development_joint_node_budget="):
        limit = int(schema["$comment"].split("=")[1])
        root_view = copy.deepcopy(schema["anyOf"][0])
        for field in ("entities", "events"):
            root_view["properties"][field].update(minItems=0, maxItems=limit)
        root_note = f"; entities+events length<={limit}"
    rows = [
        "All fields required except ?; null=absence; []=empty; + combines fields; "
        "| selects an alternative. No extra fields.",
        "Output=" + show(root_view) + root_note,
    ]
    for name, n in schema["$defs"].items():
        if name == "InstanceGraph" and n.get("$comment", "").startswith(
            "development_joint_node_budget="
        ):
            limit = int(n["$comment"].split("=")[1])
            guide_graph = copy.deepcopy(n["anyOf"][0])
            for field in ("entities", "events"):
                guide_graph["properties"][field].update(minItems=0, maxItems=limit)
            rows.append(name + "=" + show(guide_graph) + f"; entities+events length<={limit}")
        else:
            rows.append(name + "=" + show(n))
    rows += [name + "=" + show(n, defining=name) for name, n in named_shapes.items()]
    rows += [name + " matches " + pattern for pattern, name in identifiers.items()]
    return "\n".join(rows)


def prepare_request(root, kind, tokenizer, manifest, *, previous=None, feedback=None, sealed=None):
    _, unit, neutral = sources(root)
    evidence = tuple(EvidenceRecord.model_validate(e) for e in neutral["evidence"])
    mapping = aliases(evidence)
    config = DevelopmentConstructionConfiguration.load(
        root / "configs/study/development_construction.json"
    )
    condition = ConditionName.C1_LLM_PRE if kind == "c1" else ConditionName.C2_LLM_QUERY
    budgets = (
        config.preconstruction_budgets
        if kind == "c1"
        else config.projection_budgets_by_unit[unit["unit_id"]]
    )
    schema = compact_schema(
        reference_translation(
            budget_schema(candidate_schema(evidence, config.upper_ontology), budgets),
            mapping,
            schema=True,
        )
    )
    guide = vocabulary_guide(schema)
    schema = backend_generation_schema(schema)
    protocol = read(root / CONFIG)
    instruction = (root / protocol["instruction_path"]).read_text()
    intro = (
        "Construct a query-blind ontology covering consequential supported content "
        "across this complete evidence. No user context has been revealed to this condition."
        if kind == "c1"
        else "Construct a contextual ontology directly from this evidence and the supplied "
        "user context. Your pre-query ontology is empty. Context can change identities, "
        "grouping, types, relations, events and abstraction."
    )
    body = {
        "evidence_index": evidence_text(evidence, mapping),
        "upper_ontology": without_admin(config.upper_ontology.model_dump(mode="json")),
        "sealed_horizon": without_admin(neutral["snapshot"]["horizon"]),
        "object_budgets": without_admin(budgets.model_dump(mode="json")),
    }
    if kind != "c1":
        ordinal = int(kind[-1])
        body["context"] = reference_translation(
            without_admin(
                read(root / unit["query_stages"][ordinal - 1]["relative_path"] / "query.json")[
                    "query"
                ]
            ),
            mapping,
        )
    messages = [
        ChatMessage(
            role="system",
            content=intro + "\n" + instruction + "\n" + guide,
        ),
        ChatMessage(
            role="user", content=json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        ),
    ]
    if previous is not None:
        # Evidence-first repair: original remains complete in restricted lineage.
        messages += [
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "repair": (
                            "Previous output is untrusted, not evidence, and retained outside "
                            "this request. Reconstruct a complete replacement from unchanged "
                            "evidence. Reconsider and retract unsupported claims. Do not invent "
                            "missing semantics. Address these demonstrated defects; unresolved "
                            "assessment is not a positive verdict."
                        ),
                        "diagnostics": feedback,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        ]
    count = len(
        tokenizer.apply_chat_template(
            [asdict(m) for m in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    max_input, output = protocol[
        "repair_input_output_tokens" if previous is not None else "base_input_output_tokens"
    ]
    if max_input + output != protocol["context_limit"]:
        raise ValueError("development token policy must fit the unchanged context")
    decoding = DecodingManifest(
        decoding_pass=DecodingPass.FIRST_PASS if previous is None else DecodingPass.REPAIR,
        maximum_input_tokens=max_input,
        maximum_output_tokens=output,
        seed=1988649846,
        eos_token_id=manifest.eos_token_id,
        end_of_turn_token_ids=manifest.end_of_turn_token_ids,
        stop_token_ids=manifest.stop_token_ids,
        chat_template_hash=manifest.chat_template_sha256,
        structured_decoder="vllm-0.10.2-xgrammar-no-fallback",
        tokenizer_revision=manifest.revision,
        output_schema_hash=canonical_sha256(schema),
    )
    if count > max_input:
        raise ValueError(f"complete {kind} input {count} exceeds {max_input}; no truncation")
    parts = {
        "evidence_snapshot" if kind == "c1" else "evidence_packet": body["evidence_index"],
        "output_schema": schema,
        "system_prompt": messages[0].content,
        "upper_ontology": body["upper_ontology"],
    }
    if kind != "c1":
        parts["query_context"] = body["context"]
    packing = PackingReport.build(
        condition=condition,
        tokenizer_revision=manifest.revision,
        maximum_model_tokens=protocol["context_limit"],
        maximum_input_tokens=max_input,
        reserved_output_tokens=output,
        sections=(
            *tuple(
                PackingSection(name=k, section_content_hash=canonical_sha256(v), token_count=0)
                for k, v in parts.items()
            ),
            PackingSection(
                name="complete_development_request",
                section_content_hash=canonical_sha256([asdict(m) for m in messages]),
                token_count=count,
            ),
        ),
        required_section_names=(*parts, "complete_development_request"),
        complete_evidence_snapshot=True if kind == "c1" else None,
        complete_evidence_packet=True if kind != "c1" else None,
    )
    request = GuidedJSONRequest(
        request_id="development-demo-" + kind + ("-repair" if previous else ""),
        model_name="qwen3-8b-awq-fallback",
        condition=condition,
        messages=tuple(messages),
        output_schema=schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=count,
        stream_response=True,
    )
    return request, evidence, mapping, budgets


def adapt_output(parsed, evidence, mapping, upper, execution):
    translated = reference_translation(parsed, {v: k for k, v in mapping.items()})
    return reconstruct_candidate(translated, evidence=evidence, upper=upper, execution=execution)


def nontransmitted_reservations(block):
    """Explicit, hash-bound reconciliation; never infer non-transmission from silence."""
    path = block / "reservation-reconciliations.json"
    if not path.exists():
        return {}
    records = read(path)
    for attempt, item in records.items():
        reservation = read(block / item["reservation_file"])
        if (
            reservation["attempt_id"] != attempt
            or canonical_sha256(reservation) != item["reservation_hash"]
            or item["transmitted"] is not False
            or item["preserve_reservation_count"] is not True
        ):
            raise ValueError("invalid non-transmitted reservation reconciliation")
    return records


def pending_work(block):
    """Resume this fixed five-base workload without silently repeating a base."""
    history = [(p, read(p)) for p in sorted(block.glob("run-*/*/outcome.json"))]
    observed = {o["attempt_id"] for _, o in history}
    nontransmitted = nontransmitted_reservations(block)
    for reservation in block.glob("attempt-*.json"):
        if read(reservation)["attempt_id"] not in observed | set(nontransmitted):
            raise ValueError("Reserved attempt has no terminal outcome; reconcile before resume")
    feedback_path = block / "prepared-parent-feedback.json"
    prepared = read(feedback_path) if feedback_path.exists() else {}
    queue = []
    accepted_c1_path = None
    for kind in ("c1", "c2-q1", "c2-q2", "fixed-q1", "fixed-q2"):
        old = [(p, o) for p, o in history if o["kind"] == kind]
        if not old:
            queue.append((kind, None, None))
            continue
        if kind == "c1":
            accepted_c1_path = next((p for p, o in reversed(old) if o["scientific_accepted"]), None)
        if any(o["repair_parent"] or o["scientific_accepted"] for _, o in old):
            continue
        if len(old) != 1:
            raise ValueError("Multiple base records must not be guessed into a repair lineage")
        _, parent = old[0]
        candidate = prepared.get(parent["attempt_id"])
        if candidate is None:
            continue
        if candidate["parent_request_hash"] != parent["request_hash"] or candidate[
            "parent_response_hash"
        ] != parent["transport_metadata"].get("response_sha256"):
            raise ValueError("Prepared repair is not bound to its actual parent response")
        queue.append((kind, parent["attempt_id"], candidate["diagnostics"]))
    return queue, accepted_c1_path


def phase_policy(root=None):
    import math
    from types import SimpleNamespace

    cfg = read((root or Path.cwd()) / CONFIG)

    def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
        if not math.isfinite(actual) or actual < 6716.108081 or seconds < 0:
            raise ValueError("invalid or reset historical allocation")
        if (
            starts + int(starting) > cfg["maximum_service_starts"]
            or attempts + int(generating) > 12
        ):
            raise ValueError("development start/attempt authority exhausted")
        end = actual + seconds + 60
        if end > 10316.108081 - 5 or end > 33660 or end >= 36000:
            raise TimeoutError("development allocation must preserve shutdown and guard")
        return {
            "actual": actual,
            "envelope_end": end,
            "development_only": True,
            "complete_forecast_exception": True,
            "ordinary_study_authorized": False,
        }

    expected = {
        "historical_actual_seconds": 6716.108081,
        "maximum_additional_seconds": 3600,
        "maximum_global_seconds": 10316.108081,
        "maximum_service_starts": 4,
        "maximum_attempts": 12,
        "maximum_repairs_per_base": 1,
    }
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError("development bounds differ from explicit user amendment")
    return SimpleNamespace(
        BASELINE=6716.108081,
        BLOCK_ID=cfg["amendment_id"],
        ALLOWANCE=3600,
        STARTS=cfg["maximum_service_starts"],
        ATTEMPTS=12,
        GENERATION=300,
        admit=admit,
        config=cfg,
    )
