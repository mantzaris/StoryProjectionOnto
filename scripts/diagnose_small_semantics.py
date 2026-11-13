#!/usr/bin/env python3
"""CPU-only assertion diagnosis and TWO general-instruction request preparations.

Never starts a service. Never modifies a retained request, response or ledger.
The revised observation report is not a new model result or acceptance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from story_projection_onto.contracts import OntologyDraft, PreconstructionRequest, canonical_sha256
from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.semantic_generation import build_clarified_small_request


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source, output, tokenizer_path, instruction_path):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("new ignored restricted output directory required")
    terminal = read(source / "terminal.json")
    outcome = terminal["outcomes"][0]
    attempt = source / outcome["attempt_id"]
    sources = [
        source / "terminal.json",
        attempt / "decoded.json",
        attempt / "canonical.json",
        attempt / "adapter-provenance.json",
        attempt / "component-checks.json",
        source / "fixture-semantic-first.json",
        source / "fixture-semantic-second.json",
        source / "prepared-semantic-first.json",
        source / "prepared-semantic-second.json",
    ]
    original_hashes = {str(p): sha(p) for p in sources}
    original = read(attempt / "decoded.json")
    if canonical_sha256(original) != outcome["response"]["parsed_object_sha256"]:
        raise ValueError("retained raw semantic content hash differs")
    manifest = TokenizerManifest(
        **read(source / "rendered-semantic-first.json")["tokenizer_manifest"]
    )
    for filename, digest in manifest.tokenizer_file_sha256:
        if sha(tokenizer_path / filename) != digest:
            raise ValueError("local tokenizer file differs from live pinned manifest: " + filename)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    instruction = instruction_path.read_text()
    # Prepare both examples before examining any scorer observation. The second
    # was selected/frozen in the preceding session and never received a model call.
    prepared, fixtures = {}, {}
    for label in ("first", "second"):
        fixture = PreconstructionRequest.model_validate(
            read(source / f"fixture-semantic-{label}.json")
        )
        request = build_clarified_small_request(fixture, tokenizer, manifest, instruction)
        old = read(source / f"prepared-semantic-{label}.json")
        new = request.wire_payload()
        if {k: v for k, v in old.items() if k != "messages"} != {
            k: v for k, v in new.items() if k != "messages"
        } or old["messages"][1:] != new["messages"][1:]:
            raise ValueError("revision changed more than the general semantic instruction")
        fixtures[label], prepared[label] = fixture, request

    from story_projection_onto.phase1_legacy_provenance import Phase1LegacyEvidenceProvenanceBridge
    from story_projection_onto.scorer_only.small_diagnostic_checks import component_audit

    root = Path.cwd()
    oracle = (
        Phase1LegacyEvidenceProvenanceBridge.load(root)
        .resolve_call(
            call_id="c1-01",
            condition=fixtures["first"].condition,
            request_fixture="tests/fixtures/phase1/c1_pre_request.json",
        )
        .evidence
    )
    observation = component_audit(
        OntologyDraft.model_validate(read(attempt / "canonical.json")), fixtures["first"], oracle
    )
    translations = read(attempt / "adapter-provenance.json")["identifier_translation"]
    assessment_by_id = {a["record_id"]: a for a in observation["legacy_assessments"]}
    graph = original["instance_graph"]
    nodes = {n.get("entity_id", n.get("event_id")): n for n in graph["entities"] + graph["events"]}
    predicates = {p["predicate_id"]: p for p in original["local_schema"]["predicates"]}
    evidence = {e.evidence_id: e.model_dump(mode="json") for e in fixtures["first"].evidence}
    rows, pages = (
        [],
        [
            "# Assertion-level diagnosis — retained model response, CPU observations v2\n",
            "Original output is unchanged and remains failed. This is not a new generation, "
            "independent review or retrospective acceptance. Manual classification follows the "
            "machine-extracted records below in SEMANTIC_DIAGNOSIS.md.\n",
        ],
    )
    for index, assertion in enumerate(graph["assertions"]):
        canonical_id = translations["canonical_ids_by_record_path"][
            f"/instance_graph/assertions/{index}"
        ]
        predicate = predicates[assertion["predicate_id"]]
        row = dict(
            assertion=assertion,
            predicate=predicate,
            subject=nodes.get(assertion.get("subject_id")),
            object=nodes.get(assertion.get("object_id")),
            supplied_evidence=[evidence[e] for e in assertion["evidence_ids"]],
            node_descriptions=[
                n
                for n in nodes.values()
                if assertion["assertion_id"] in n["description_assertion_ids"]
            ],
            unchanged_legacy_rejection=assessment_by_id[canonical_id],
        )
        rows.append(row)
        pages += [f"## {assertion['assertion_id']}\n", "### Exact supplied evidence\n"]
        pages += [f"- `{e}`: {evidence[e]['text']}" for e in assertion["evidence_ids"]]
        pages += [
            "\n### Generated arguments and predicate\n",
            f"Subject: `{assertion.get('subject_id')}` ({row['subject']['label']}); "
            f"object: `{assertion.get('object_id')}` ({row['object']['label']}); "
            f"direction: `{assertion['direction']}`; roles: `{assertion.get('roles', [])}`.\n",
            "```json\n" + json.dumps(predicate, indent=2) + "\n```\n",
            "### Generated descriptions\n",
            "why_matters: " + assertion["why_matters"] + "\n",
        ]
        pages += [
            f"- `{n.get('entity_id', n.get('event_id'))}`: {n['description']}"
            for n in row["node_descriptions"]
        ]
        pages += [
            "\n### Exact temporal / epistemic qualifications\n",
            "```json\n"
            + json.dumps(
                {
                    k: assertion[k]
                    for k in ("temporal_scope", "epistemic_scope", "narrative_commitment")
                },
                indent=2,
            )
            + "\n```\n",
            "### Check that rejected it\n",
            "`audit_acceptance_semantic_grounding` / `_assertion_fact`: "
            + assessment_by_id[canonical_id]["status"]
            + " — "
            + assessment_by_id[canonical_id]["reason"]
            + ".\n",
            "No known witness match is not proof of falsehood. Domain/role meaning and "
            "unsupported temporal precision are diagnosed separately.\n",
        ]
    output.mkdir(parents=True, mode=0o700)
    for label, request in prepared.items():
        write_json_atomic(request.wire_payload(), output / f"request-{label}.json")
        write_json_atomic(request.packing.model_dump(mode="json"), output / f"packing-{label}.json")
    write_json_atomic(
        dict(
            revision="small-semantic-offline-diagnosis-v2",
            gpu_seconds=0,
            model_result=False,
            scientific_acceptance=False,
            original_sources=original_hashes,
            checker_source_sha256=sha(
                root / "src/story_projection_onto/scorer_only/small_diagnostic_checks.py"
            ),
            instruction_sha256=sha(instruction_path),
            tokenizer_manifest=asdict(manifest),
            observations=observation,
            assertions=rows,
            requests={
                k: dict(
                    request_hash=q.request_hash,
                    input_tokens=q.rendered_input_token_count,
                    output_allowance=q.decoding.maximum_output_tokens,
                    schema_hash=canonical_sha256(q.output_schema),
                    unchanged_schema=True,
                    preselected_evidence_ids=[e.evidence_id for e in fixtures[k].evidence],
                )
                for k, q in prepared.items()
            },
            second_example_selected_before_prior_live_output=True,
            original_failure_preserved=True,
        ),
        output / "offline-diagnosis.json",
    )
    (output / "ASSERTION_RECORDS.md").write_text("\n".join(pages))
    request_pages = [
        "# Prepared small requests — NOT launched\n",
        "Same pinned model, template, "
        "schema, codec, evidence and token allowances; general instruction revised.\n",
    ]
    for label, request in prepared.items():
        request_pages += [
            f"## {label}\n",
            request.messages[0].content,
            "\n### Exact supplied user input\n\n```json\n"
            + json.dumps(json.loads(request.messages[1].content), indent=2)
            + "\n```\n",
        ]
    (output / "REQUESTS.md").write_text("\n".join(request_pages))
    if original_hashes != {str(p): sha(p) for p in sources}:
        raise ValueError("retained source changed during CPU diagnosis")
    print(
        json.dumps(
            dict(
                output=str(output),
                sources_unchanged=True,
                gpu_seconds=0,
                requests={k: q.rendered_input_token_count for k, q in prepared.items()},
                scientific_acceptance=observation["all_checks_pass"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.tokenizer, args.instruction)
