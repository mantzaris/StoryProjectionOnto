#!/usr/bin/env python3
"""Read a retained SMALL development response; write a fresh restricted CPU audit.

No model inference, no scorer input and no reconstruction of ambiguous records.
Candidate request is a proposal, not authorization or activation of a protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.semantic_generation import schema_guide
from story_projection_onto.semantic_identifiers import (
    IDENTIFIER_INSTRUCTION,
    identifier_audit,
    normalize_identifiers,
    typed_identifier_schema,
)


def run(source: Path, output: Path, tokenizer_path: Path):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("fresh restricted output required")
    names = (
        "semantic-diagnostic-1/decoded.json",
        "fixture-semantic-first.json",
        "prepared-semantic-first.json",
        "prepared-semantic-first-repair.json",
    )
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    wire, fixture, first, retry = [json.loads((source / name).read_bytes()) for name in names]
    evidence = fixture["evidence"]
    supplied = [
        obj.get("candidate_id", obj.get("clue_id"))
        for e in evidence
        for field in (
            "mention_candidates",
            "event_candidates",
            "relation_phrase_candidates",
            "temporal_clues",
        )
        for obj in e[field]
    ]
    audit = identifier_audit(wire, supplied)
    try:
        normalize_identifiers(wire, supplied)
    except ValueError as error:
        normalization = {"accepted": False, "reason": error.args[0], "canonical_draft": None}
    else:
        raise ValueError("this audit expects the retained ambiguous failure, not a repaired graph")
    delta = {k: {"before": first[k], "after": retry[k]} for k in first if first[k] != retry[k]}
    assert set(delta) == {"messages"}
    prompt_append = retry["messages"][0]["content"][len(first["messages"][0]["content"]) :]
    assert retry["messages"][0]["content"].startswith(first["messages"][0]["content"])
    assert retry["messages"][1:] == first["messages"][1:]

    # All invariant model/evidence/settings retained; only typed-ID contract and
    # its readable guide replace the old global-counter instructions.
    candidate = copy.deepcopy(first)
    candidate["guided_json"] = typed_identifier_schema(first["guided_json"])
    instruction = first["messages"][0]["content"].split("The following complete field guide")[0]
    begin = instruction.index("New local IDs start with n")
    end = instruction.index("Evidence", begin)
    instruction = instruction[:begin] + IDENTIFIER_INSTRUCTION + " " + instruction[end:]
    candidate["messages"][0]["content"] = (
        instruction + "The following complete field guide matches the supplied grammar. "
        "It is syntax, not an answer.\n" + schema_guide(candidate["guided_json"])
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokens = len(
        tokenizer.apply_chat_template(
            candidate["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )
    if tokens > 6144 or tokens + candidate["max_tokens"] > 12288:
        raise ValueError("candidate does not fit unchanged template-inclusive context")
    capacity = dict(
        input_tokens=tokens,
        output_allowance=candidate["max_tokens"],
        total_context=12288,
        template_inclusive=True,
        model_result=False,
        generation_reliability="unmeasured",
        request_hash=canonical_sha256(candidate),
        tokenizer_files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tokenizer_path.glob("*.json")
        },
    )
    output.mkdir(parents=True, mode=0o700)
    write_json_atomic(
        {
            "sources": hashes,
            "audit": audit,
            "reprocessing": normalization,
            "retry_prompt_append": prompt_append,
            "schema_changed_on_retry": False,
            "candidate_capacity": capacity,
        },
        output / "audit.json",
    )
    write_json_atomic(candidate, output / "candidate-request.json")
    write_json_atomic(candidate["guided_json"], output / "candidate-schema.json")
    (output / "CANDIDATE_REQUEST.md").write_text(
        "# Proposed typed-ID small request — CPU only, NOT launched\n\n"
        + json.dumps(capacity, indent=2)
        + "\n\n## System prompt\n\n"
        + candidate["messages"][0]["content"]
        + "\n\n## Supplied user input\n\n```json\n"
        + json.dumps(json.loads(candidate["messages"][1]["content"]), indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    lines = [
        "# Retained first named-semantic graph — failed, immutable\n",
        "Manual diagnostic findings are NOT scientific acceptance. No corrected graph "
        "was constructed: three decision targets remain ambiguous.\n",
        "## Supplied evidence\n",
    ]
    lines += [f"- {e['evidence_id']}: {e['text']}" for e in evidence]
    lines += [
        "\nSupplied neutral time clue: `day-1-dawn`; no explicit numeric coordinate "
        "mapping or intrinsic duration. Candidate references and exact input are in "
        "CANDIDATE_REQUEST.md; only its system identifier policy differs.\n",
        "## Every generated record and incoming local reference\n",
        "Paths are JSON pointers. `?` marks an ambiguous reference; it is listed for "
        "both possible targets, not resolved twice.\n",
        "| ID | Record type/path | Incoming references |",
        "|---|---|---|",
    ]
    for d in audit["declarations"]:
        refs = [
            r["path"] + (" ?" if r["resolution"] == "ambiguous" else "")
            for r in audit["references"]
            if d["path"] in r["candidates"]
        ]
        lines.append(
            f"| {d['id']} | {d['kind']} `{d['path']}` | " + "<br>".join(refs or ["none"]) + " |"
        )
    lines += [
        "\nAll collisions are cross-record-type. None is a duplicate within a type, "
        "and no entire record is an exact duplicate. The two `carried` predicates "
        "have duplicate semantic payloads except their different IDs; neither is merged. "
        "Schema/decision n000, entity/assertion n001-n003, event/type/predicate n004, "
        "type/predicate n005-n006. n007 is unique.\n",
        "Typed endpoint, type, predicate and description-support fields resolve their "
        "targets. The supported_description decision's created_object_ids n001-n003 "
        "are untyped and each could name an entity OR assertion. Its rationale does "
        "not authorize choosing. Canonical graph-global uniqueness rejects this graph "
        "correctly under the v1 prompt; the plan itself does not require one namespace "
        "for schema/types/predicates/decisions.\n",
        "## Generated graph\n",
        "| Node | Label | Type | Description | Support assertion |",
        "|---|---|---|---|---|",
    ]
    graph = wire["instance_graph"]
    for n in [*graph["entities"], *graph["events"]]:
        lines.append(
            f"| {n.get('entity_id', n.get('event_id'))} | {n['label']} | "
            f"{n['contextual_type_id']} | {n['description']} | "
            + ", ".join(n["description_assertion_ids"])
            + " |"
        )
    predicates = {p["predicate_id"]: p["label"] for p in wire["local_schema"]["predicates"]}
    labels = {
        n.get("entity_id", n.get("event_id")): n["label"]
        for n in [*graph["entities"], *graph["events"]]
    }
    lines += [
        "\n| Assertion | Subject | Predicate | Object | Story/validity | Evidence |",
        "|---|---|---|---|---|---|",
    ]
    for a in graph["assertions"]:
        lines.append(
            f"| {a['assertion_id']} | {labels[a['subject_id']]} ({a['subject_id']}) | "
            f"{predicates[a['predicate_id']]} ({a['predicate_id']}) | "
            f"{labels[a['object_id']]} ({a['object_id']}) | "
            + json.dumps({k: a["temporal_scope"][k] for k in ("story_time", "validity_time")})
            + " | "
            + ", ".join(a["evidence_ids"])
            + " |"
        )
    lines += [
        "\n## Independent remaining findings (manual, not formal acceptance)\n",
        "1. Assertion n001 (Lio arrived_at North Gate) and n002 (Lio carried seal) "
        "have text-supported endpoints and relation meanings. n003 says North Gate "
        "carried seal: unsupported, and contradicts its own why_matters naming Lio. "
        "Its predicate domain lists actor/object types, not North Gate's location type.\n",
        "2. The carried predicates' located_at upper parent is at best an imprecise "
        "specialization: an actor carrying an artifact is not that actor being located "
        "at the artifact. The upper vocabulary has related_to available. This is a "
        "manual schema-meaning issue, not an unknown upper-reference failure.\n",
        "3. North Gate's description support n002 has only Lio/seal endpoints. The "
        "arrival event's support n001 has only Lio/Gate endpoints. Both violate the "
        "existing description-footprint rule independently of ID renaming. Seal's "
        "support includes it but makes the wrong carrier assertion.\n",
        "4. Arrival is explicitly evidenced, but the generated event has no incident "
        "role/assertion. No event_reification or other substantive construction "
        "operator was recorded; the sole operator is supported_description. That does "
        "not meet the prompt's nonselection construction task, despite minItems=1 "
        "passing. The candidate ID-only schema intentionally does not claim to fix "
        "this semantic failure.\n",
        "5. All assertions use story point 1 / validity point 1 labeled dawn. The "
        "supplied clue day-1-dawn does not define the required integer coordinate axis; "
        "point 1 is therefore not established by the request. More decisively, neither "
        "onset nor endpoint of carrying/location is given. A singleton intrinsic "
        "validity interval cannot be inferred from observing arrival at dawn. Unknown "
        "validity remains available. No reported attitude is present: null epistemic "
        "scope/proposition and world_committed content are appropriate in form.\n",
        "6. Every citation is ev-01 and each mention/candidate reference is supplied. "
        "Provenance confidence 1.0 does not exceed source 1.0. Model judgments .96 are "
        "retained, not certified as calibrated. Discourse (1,0,0) and revelation 1 "
        "agree with supplied order. Those passing checks do not cure unsupported content.\n",
        "## What changed before the repetitive second response\n\n```text\n"
        + prompt_append
        + "\n```\n",
        "Only this system-message suffix changed. Same evidence, schema, model, "
        "sampling and 6,144 output allowance. The second response repeatedly declared "
        "auxiliary propositions, reached 6,144 tokens and ended length/truncated. The "
        "schema permits unbounded proposition_contents and does not require their use. "
        "The warning plausibly added bookkeeping burden, but a single uncontrolled "
        "retry cannot establish causation, decoder failure or a model capability limit.\n",
        "## Exact original semantic payload (not modified)\n\n```json\n"
        + json.dumps(wire, indent=2)
        + "\n```\n",
        "## Source hashes\n\n```json\n" + json.dumps(hashes, indent=2) + "\n```\n",
    ]
    (output / "FAILED_GRAPH.md").write_text("\n".join(lines), encoding="utf-8")
    assert hashes == {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
    }
    print(json.dumps({"output": str(output), "normalization": normalization, "capacity": capacity}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    args = p.parse_args()
    run(args.source, args.output, args.tokenizer)
