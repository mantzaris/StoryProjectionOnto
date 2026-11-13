#!/usr/bin/env python3
"""Restricted CPU contract/grammar/tokenizer verification. Cannot start a service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from jsonschema import Draft202012Validator

from story_projection_onto.contracts import PreconstructionRequest, canonical_sha256
from story_projection_onto.gpu_runtime import capture_tokenizer_manifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.semantic_generation import (
    build_small_request,
    reconstruct,
    semantic_schema,
)


def json_errors(value, schema):
    return [
        {
            "path": "/" + "/".join(map(str, e.absolute_path)),
            "validator": e.validator,
            "message": e.message,
        }
        for e in Draft202012Validator(schema).iter_errors(value)
    ]


def schema_order(value, node, definitions):
    """Authored capacity fixtures ONLY: follow XGrammar's schema property order."""
    if "$ref" in node:
        return schema_order(value, definitions[node["$ref"].rsplit("/", 1)[-1]], definitions)
    if "anyOf" in node:
        for choice in node["anyOf"]:
            if not json_errors(value, {"$defs": definitions, **choice}):
                return schema_order(value, choice, definitions)
        raise ValueError("authored fixture has no schema alternative")
    if isinstance(value, dict):
        return {
            k: schema_order(value[k], v, definitions)
            for k, v in node["properties"].items()
            if k in value
        }
    if isinstance(value, list):
        return [schema_order(v, node["items"], definitions) for v in value]
    return value


def replay(run, schema):
    """Replay complete retained content, unchanged. Subtree checks are not repairs."""
    frozen = json.loads((run / "comparison-requests.json").read_bytes())
    from story_projection_onto.output_wire import RecordTupleCodec, translate_references

    # Read canonical decoded records and verify unchanged-byte hashes; C also has
    # its actual wire JSON retained in the earlier verified response summary.
    results = []
    for number, label in enumerate("ABC", 1):
        path = run / f"representation-diagnostic-{number}/decoded.json"
        original = path.read_bytes()
        obj = json.loads(original)
        assertions = obj["instance_graph"]["assertions"]
        row = {
            "variant": label,
            "retained_decoded_sha256": hashlib.sha256(original).hexdigest(),
            "unchanged_full_response_schema_errors": json_errors(obj, schema),
            "subtree_checks_are_not_acceptance_or_adaptation": True,
            "assertions_missing_binary_and_nary_bindings": [
                a["assertion_id"]
                for a in assertions
                if not (a.get("subject_id") and a.get("object_id")) and not a.get("roles")
            ],
            "temporal_subtree_errors": [],
            "upper_parent_subtree_errors": [],
        }
        for a in assertions:
            for field in ("story_time", "validity_time"):
                errors = json_errors(
                    a["temporal_scope"][field],
                    {"$defs": schema["$defs"], "$ref": "#/$defs/SemanticTime"},
                )
                row["temporal_subtree_errors"].append(
                    {"assertion": a["assertion_id"], "field": field, "errors": errors}
                )
        for p in obj["local_schema"]["predicates"]:
            parent = schema["$defs"]["LocalPredicateDefinition"]["properties"][
                "parent_upper_relation"
            ]
            row["upper_parent_subtree_errors"].append(
                {
                    "predicate": p["predicate_id"],
                    "errors": json_errors(p.get("parent_upper_relation"), parent),
                }
            )
        graph = obj["instance_graph"]
        row["small_structure"] = {
            "node_records": len(graph["entities"]) + len(graph["events"]),
            "assertions": len(assertions),
            "decisions": len(obj["decisions"]),
        }
        # Reassemble every original SSE stream; no output rewriting or filling.
        import zstandard

        from story_projection_onto.streaming_chat import ChatSSE

        journals = list((run / "http").glob(frozen[label]["request_hash"][:16] + "-*/events.jsonl"))
        if len(journals) != 1:
            raise ValueError("response journal identity is ambiguous")
        parser = ChatSSE()
        journal = journals[0]
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "response_fragment":
                continue
            artifact = event["artifact"]
            data = zstandard.ZstdDecompressor().decompress(
                (journal.parent / "fragments" / artifact["relative_path"]).read_bytes()
            )
            if hashlib.sha256(data).hexdigest() != artifact["content_hash"]:
                raise ValueError("response fragment hash mismatch")
            parser.feed(data, 0)
        text = parser.envelope()["choices"][0]["message"]["content"]
        wire = json.loads(text)
        if label == "C":
            decoded = RecordTupleCodec(
                frozen["C"]["canonical_schema"], opaque_aliases=frozen["C"]["reference_map"]
            ).decode(wire)
            assert translate_references(decoded, frozen["C"]["reference_map"], decode=True) == obj
        else:
            assert wire == obj
        row["actual_wire_content"] = text
        row["actual_wire_content_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        row["actual_wire_new_schema_errors"] = json_errors(wire, schema)
        assert path.read_bytes() == original
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or "restricted" not in args.output.parts:
        raise ValueError("fresh restricted output directory required")
    args.output.mkdir(parents=True, mode=0o700)
    os.umask(0o077)
    started = time.monotonic()
    # Authored test fixtures are deliberately outside the installed runtime package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import xgrammar as xgr
    from transformers import AutoTokenizer

    from tests.unit.test_semantic_generation import (
        authored_wire,
        execution,
        small_wire,
        source_fixture,
    )

    if version("xgrammar") != "0.1.23":
        raise ValueError("requires pinned CPU grammar compiler")
    revision = "4da05a8edb55c6046cce958586c33b61da07bb79"
    manifest = capture_tokenizer_manifest(
        args.snapshot, repository="Qwen/Qwen3-8B-AWQ", revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot, local_files_only=True, trust_remote_code=False
    )
    fixture = PreconstructionRequest.model_validate_json(
        (args.run / "small-evidence-only-request.json").read_bytes()
    )
    request = build_small_request(fixture, tokenizer, manifest)
    schema = request.output_schema
    write_json_atomic(request.wire_payload(), args.output / "request.json")
    write_json_atomic(schema, args.output / "generation.schema.json")
    write_json_atomic(
        {
            "decoding": request.decoding.model_dump(mode="json"),
            "packing": request.packing.model_dump(mode="json"),
            "request_hash": request.request_hash,
            "tokenizer": manifest.public_manifest(),
        },
        args.output / "request-binding.json",
    )
    rendered = tokenizer.apply_chat_template(
        [asdict(m) for m in request.messages],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    write_json_atomic(
        {"text": rendered, "tokens": request.rendered_input_token_count},
        args.output / "rendered-chat.json",
    )
    # Generated human-readable artifact, not a manually transcribed request.
    readable = "# Diagnostic only: exact request and grammar\n\nNo GPU execution authorized.\n\n"
    for message in request.messages:
        readable += f"## {message.role}\n\n{message.content}\n\n"
    readable += (
        "## Exact matching JSON schema\n\n```json\n" + json.dumps(schema, indent=2) + "\n```\n"
    )
    readable += (
        "\n## Exact non-message HTTP fields\n\n```json\n"
        + json.dumps(
            {
                k: v
                for k, v in request.wire_payload().items()
                if k not in {"messages", "guided_json"}
            },
            indent=2,
        )
        + "\n```\n"
    )
    with (args.output / "MODEL_REQUEST_AND_SCHEMA.md").open("x", encoding="utf-8") as stream:
        stream.write(readable)
        stream.flush()
        os.fsync(stream.fileno())

    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer), max_threads=8, cache_enabled=False
    )
    compile_start = time.monotonic()
    # Both actual pinned server stages: pre-validation conversion then compilation
    # with the existing no-arbitrary-whitespace setting. No model/weights loaded.
    xgr.Grammar.from_json_schema(schema)
    compiled = compiler.compile_json_schema(schema, any_whitespace=False)
    compile_seconds = time.monotonic() - compile_start

    def accepts(text, grammar=compiled):
        matcher = xgr.GrammarMatcher(grammar)
        return matcher.accept_string(text) and matcher.accept_token(manifest.eos_token_id)

    capacities = []
    for label, wire, source, small in (
        ("small-two-node-authored", small_wire(), fixture, True),
        ("authored-events-nary", authored_wire()[0], source_fixture(), False),
        (
            "authored-holder-proposition",
            authored_wire("c1_pre_output_2.json")[0],
            source_fixture(),
            False,
        ),
    ):
        current_schema = semantic_schema(source.evidence, source.upper_ontology, small=small)
        reconstruct(
            wire,
            evidence=source.evidence,
            upper=source.upper_ontology,
            execution=execution(),
            small=small,
        )
        ordered = schema_order(wire, current_schema, current_schema["$defs"])
        text = json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))
        grammar = (
            compiled
            if small
            else compiler.compile_json_schema(current_schema, any_whitespace=False)
        )
        capacity = {
            "fixture": label,
            "authored_not_model_output": True,
            "completion_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "completion_termination_allowance": 1,
            "characters": len(text),
            "whitespace_characters": sum(c.isspace() for c in text),
            "grammar_accepts_complete_fixture": accepts(text, grammar),
        }
        capacities.append(capacity)
        write_json_atomic(ordered, args.output / f"{label}.json")
    probes = {}
    time_grammar = compiler.compile_json_schema(
        {"$defs": schema["$defs"], "$ref": "#/$defs/SemanticTime"}, any_whitespace=False
    )
    for label, value in {
        "point_valid": {"kind": "point", "point": 1},
        "unknown_valid": {"kind": "unknown", "reason": "Not stated"},
        "point_without_coordinate_rejected": {"kind": "point", "label": "dawn"},
        "point_plus_partial_order_rejected": {"kind": "point", "point": 0, "partial_order": []},
    }.items():
        probes[label] = accepts(json.dumps(value), time_grammar)
    structural_probes = {}
    for label, node, values in (
        (
            "new_local_id",
            {"type": "string", "pattern": "^n[A-Za-z0-9_-]{1,63}$"},
            (("nActor", True), ("ev-01", False), ("n" + "a" * 64, False)),
        ),
        (
            "evidence_id",
            schema["$defs"]["ProvenanceReference"]["properties"]["evidence_id"],
            (("ev-01", True), ("invented", False)),
        ),
        (
            "upper_relation",
            schema["$defs"]["LocalPredicateDefinition"]["properties"]["parent_upper_relation"],
            (("located_at", True), ("carries", False)),
        ),
    ):
        grammar = compiler.compile_json_schema(node, any_whitespace=False)
        structural_probes[label] = [
            {
                "value": value,
                "expected_accept": expected,
                "actual_accept": accepts(json.dumps(value), grammar),
            }
            for value, expected in values
        ]
    ordered_small = schema_order(small_wire(), schema, schema["$defs"])
    import copy

    for label in ("missing_endpoint", "mixed_time", "empty_graph"):
        mutant = copy.deepcopy(ordered_small)
        if label == "missing_endpoint":
            mutant["instance_graph"]["assertions"][0].pop("object_id")
        elif label == "mixed_time":
            mutant["instance_graph"]["assertions"][0]["temporal_scope"]["story_time"] = {
                "kind": "point",
                "point": 0,
                "partial_order": [],
            }
        else:
            mutant["instance_graph"].update(entities=[], events=[], assertions=[])
        structural_probes[label] = accepts(json.dumps(mutant), compiled)
    root = Path(__file__).resolve().parents[1]
    summary = {
        "diagnostic_only": True,
        "gpu_seconds": 0,
        "service_started": False,
        "versions": {n: version(n) for n in ("xgrammar", "transformers", "vllm")},
        "request_hash": request.request_hash,
        "schema_hash": canonical_sha256(schema),
        "template_inclusive_input_tokens": request.rendered_input_token_count,
        "output_allowance": 6144,
        "total_context": 12288,
        "packing_margin": 12288 - request.rendered_input_token_count - 6144,
        "grammar_compile_seconds": compile_seconds,
        "authored_capacity_fixtures": capacities,
        "grammar_time_probes": probes,
        "grammar_structural_probes": structural_probes,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in (
                "src/story_projection_onto/semantic_generation.py",
                "scripts/verify_semantic_interface_cpu.py",
                "tests/unit/test_semantic_generation.py",
                "scripts/run_capacity_diagnostics.py",
            )
        },
        "replay": replay(args.run, schema),
        "cpu_wall_seconds": time.monotonic() - started,
        "limits": (
            "capacity checks, not model reliability or full-study packing/throughput evidence"
        ),
    }
    write_json_atomic(summary, args.output / "verification.json")
    write_json_atomic(
        {
            "files": [
                {
                    "path": p.name,
                    "size": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(args.output.iterdir())
                if p.is_file()
            ]
        },
        args.output / "manifest.json",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "replay"}, indent=2))


if __name__ == "__main__":
    main()
