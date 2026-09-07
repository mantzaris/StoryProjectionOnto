#!/usr/bin/env python3
"""CPU-only capacity measurements; authored fixtures are never model results.

Only Phase-1 authored fixtures and explicitly registered DEVELOPMENT evidence
are opened. No held-out gold or model weights are read. Output is restricted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from story_projection_onto.contracts import OntologyDraft, canonical_json, canonical_json_schema
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.output_wire import (
    RecordTupleCodec,
    capacity_messages,
    pack_capacity_candidate,
)


def counts(value, tokenizer):
    compact = canonical_json(value)
    buckets = {"keys": [], "identifiers": [], "descriptions": [], "other_values": []}
    descriptions = {
        "description",
        "definition",
        "rationale",
        "why_matters",
        "contextual_interpretation",
        "label",
        "contextual_role",
        "reason",
    }

    def walk(item, field=""):
        if isinstance(item, dict):
            for key, child in item.items():
                buckets["keys"].append(key)
                walk(child, key)
        elif isinstance(item, list):
            for child in item:
                walk(child, field)
        elif isinstance(item, str):
            group = (
                "descriptions"
                if field in descriptions
                else "identifiers"
                if field.endswith(("_id", "_ids", "_hash"))
                else "other_values"
            )
            buckets[group].append(item)

    walk(value)

    def encode(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    codec = RecordTupleCodec(canonical_json_schema(OntologyDraft))
    wire = codec.encode(value)
    assert OntologyDraft.model_validate(codec.decode(wire)) == OntologyDraft.model_validate(value)
    # Capacity counterfactual: new local IDs can be short, and existing source
    # aliases are already supported by the production evidence codec. Save the
    # exact bijection separately; this is not an inference or an activated policy.
    ids = set(buckets["identifiers"])
    aliases = {item: f"x{index}" for index, item in enumerate(sorted(ids))}

    def rename(item):
        if isinstance(item, dict):
            return {key: rename(child) for key, child in item.items()}
        if isinstance(item, list):
            return [rename(child) for child in item]
        return aliases.get(item, item) if isinstance(item, str) else item

    renamed = rename(wire)

    return {
        "compact_tokens": encode(compact),
        "pretty_tokens": encode(json.dumps(value, indent=2)),
        "bytes": len(compact.encode()),
        "sha256": hashlib.sha256(compact.encode()).hexdigest(),
        "record_tuple_tokens": encode(canonical_json(wire)),
        "short_id_tuple_tokens_capacity_counterfactual": encode(canonical_json(renamed)),
        "short_id_dictionary_tokens_if_transmitted": encode(canonical_json(aliases)),
        "tuple_legend_tokens": encode(codec.legend()),
        "isolated_string_tokens_not_additive": {
            key: sum(encode(text) for text in values) for key, values in buckets.items()
        },
        "identifier_occurrences": len(buckets["identifiers"]),
        "unique_identifiers": len(set(buckets["identifiers"])),
    }


def demanding_fixture(value, copies):
    """Disjoint copies of an authored graph; explicit mechanical capacity fixture."""
    original = copy.deepcopy(value)
    graph = original["instance_graph"]
    ids = set()

    def collect(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if (
                    key.endswith("_id")
                    and isinstance(child, str)
                    and not key.startswith(("evidence", "source"))
                ):
                    ids.add(child)
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(original)

    def rename(item, index):
        if isinstance(item, dict):
            return {key: rename(child, index) for key, child in item.items()}
        if isinstance(item, list):
            return [rename(child, index) for child in item]
        return f"{item}-copy{index}" if isinstance(item, str) and item in ids else item

    result = copy.deepcopy(original)
    for key in graph:
        if isinstance(graph[key], list):
            result["instance_graph"][key] = [
                row for index in range(copies) for row in rename(graph[key], index)
            ]
    for key in ("contextual_types", "predicates"):
        if key in original["local_schema"]:
            result["local_schema"][key] = [
                row
                for index in range(copies)
                for row in rename(original["local_schema"][key], index)
            ]
    result["decisions"] = [row for i in range(copies) for row in rename(original["decisions"], i)]
    budget = result["budget_accounting"]
    for key in ("nodes_used", "assertions_used", "display_nodes_used", "display_assertions_used"):
        budget[key] *= copies
    # No requirement that an artificial stress fixture represents real evidence.
    # Pydantic validation protects its canonical shape; scientific scores absent.
    OntologyDraft.model_validate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v10-root", type=Path, required=True)
    parser.add_argument("--reuse-c0-capacity", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    if "restricted" not in args.output.parts or args.output.exists():
        raise ValueError("new restricted output required")
    if args.snapshot.name != "4da05a8edb55c6046cce958586c33b61da07bb79":
        raise ValueError("pinned fallback tokenizer required")
    cpu_started = datetime.now(UTC)
    job_metadata = {
        "pid": os.getpid(),
        "started_at": cpu_started.isoformat(),
        "seed": None,
        "random_sampling": False,
        "gpu_allocated_seconds": 0,
        "source_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in (
                "scripts/assess_output_capacity.py",
                "src/story_projection_onto/output_wire.py",
                "src/story_projection_onto/gpu_runtime.py",
                "configs/study/output_capacity_recovery.json",
            )
        },
    }
    print(json.dumps({"cpu_job": job_metadata}), flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    result = {
        "kind": "cpu_capacity_not_model_results",
        "maximum_model_tokens": 12288,
        "tokenizer_revision": args.snapshot.name,
        "rows": [],
        "cpu_job": job_metadata,
    }
    from story_projection_onto.fallback_acceptance import (
        _verify_second_recovery_decoder_compiles,
        build_fallback_acceptance_request,
        fallback_pilot_calls,
    )
    from story_projection_onto.gpu_runtime import capture_tokenizer_manifest
    from story_projection_onto.model_gate import FallbackModelPolicy
    from story_projection_onto.phase1_legacy_provenance import Phase1LegacyEvidenceProvenanceBridge

    policy = FallbackModelPolicy.from_mapping(
        json.loads((root / "configs/study/fallback_model.json").read_bytes())
    )
    tm = capture_tokenizer_manifest(
        args.snapshot, repository=policy.repository, revision=policy.revision
    )
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(root)
    result["production_requests"] = []
    for call in fallback_pilot_calls(policy):
        request = build_fallback_acceptance_request(
            root=root,
            call=call,
            tokenizer=tokenizer,
            tokenizer_manifest=tm,
            legacy_provenance_bridge=bridge,
        )
        result["production_requests"].append(
            {
                "call": call.call_id,
                "request_hash": request.request_hash,
                "input_tokens_including_template": request.rendered_input_token_count,
                "output_allowance": request.decoding.maximum_output_tokens,
                "remaining_context_capacity": 12288 - request.rendered_input_token_count,
                "structured_output": "guided_json",
                "thinking": False,
            }
        )
        try:
            candidate = pack_capacity_candidate(request, tokenizer)
            _verify_second_recovery_decoder_compiles(candidate.output_schema)
            result["production_requests"][-1]["candidate"] = {
                "input_tokens": candidate.rendered_input_token_count,
                "output_tokens": candidate.decoding.maximum_output_tokens,
                "fits": True,
                "request_hash": candidate.request_hash,
                "xgrammar_compilation_passed": True,
            }
        except ValueError as exc:
            result["production_requests"][-1]["candidate"] = {"fits": False, "failure": str(exc)}
        if call.condition.value == "A-FixedSelect":
            sections = json.loads(request.messages[1].content)
            authored = json.loads((root / "tests/fixtures/phase1/c1_pre_output.json").read_bytes())
            stress = demanding_fixture(authored, 5)
            # Capacity only, not a valid paired-condition invocation: substitute
            # an intact larger authored graph and retain the packet completely.
            fixed = sections["sealed_ontology"]
            fixed["local_schema"] = stress["local_schema"]
            fixed["instance_graph"] = stress["instance_graph"]
            # Rebuild the production closed-ID schema from the larger authored
            # sealed inventory. This is still a capacity-only fixture, not a
            # scientifically valid paired invocation or a changed validator.
            from story_projection_onto.phase1_acceptance import _condition_output_schema

            ids = [fixed["local_schema"]["schema_id"]]
            for collection, field in (
                ("contextual_types", "type_id"),
                ("predicates", "predicate_id"),
            ):
                ids.extend(row[field] for row in fixed["local_schema"][collection])
            for collection, field in (
                ("entities", "entity_id"),
                ("events", "event_id"),
                ("assertions", "assertion_id"),
                ("proposition_contents", "proposition_content_id"),
            ):
                ids.extend(row[field] for row in fixed["instance_graph"][collection])
            fixed["construction_seal"]["sealed_object_ids"] = ids
            stress_schema = _condition_output_schema(
                canonical_json_schema(OntologyDraft),
                call=call.acceptance_call(),
                fixture={"fixed_ontology": fixed, "packet": sections["evidence_packet"]},
            )
            stress_request = SimpleNamespace(
                condition=request.condition, messages=request.messages, output_schema=stress_schema
            )
            messages, aliases, copies = capacity_messages(stress_request, sections)
            codec = RecordTupleCodec(stress_schema, copies, aliases)
            from story_projection_onto.output_wire import translate_references

            selected = copy.deepcopy(stress)
            selection = copy.deepcopy(
                json.loads((root / "tests/fixtures/phase1/fixed_select_output.json").read_bytes())[
                    "decisions"
                ][0]
            )
            selection["operator"] = "selection"
            selection["created_object_ids"] = []
            selection["removed_object_ids"] = []
            selection["input_object_ids"] = [
                record[id_field]
                for kind, id_field in (
                    ("entities", "entity_id"),
                    ("events", "event_id"),
                    ("assertions", "assertion_id"),
                )
                for record in selected["instance_graph"][kind]
            ]
            selection["rationale"] = (
                "Select the complete supplied graph for this authored capacity check."
            )
            selected["decisions"] = [selection]
            aliased_stress = translate_references(selected, aliases, decode=False)
            wire = codec.encode(aliased_stress)
            restored = translate_references(codec.decode(wire), aliases, decode=True)
            assert OntologyDraft.model_validate(restored) == OntologyDraft.model_validate(selected)
            output_count = len(tokenizer.encode(canonical_json(wire), add_special_tokens=False))
            n = len(
                tokenizer.apply_chat_template(
                    [asdict(m) for m in messages],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            result["intact_c1_fixedselect_stress"] = {
                "authored_capacity_only_not_model_result": True,
                "nodes": 20,
                "assertions": 10,
                "input_tokens_including_template": n,
                "reserved_output_tokens": 3072,
                "complete_authored_output_tokens": output_count,
                "total_tokens": n + 3072,
                "fits": n + 3072 <= 12288 and output_count <= 3072,
                "evidence_or_graph_truncated": False,
                "full_canonical_roundtrip_verified": True,
            }
    for name in (
        "c1_pre_output.json",
        "c1_pre_output_2.json",
        "c2_query_output.json",
        "c2_query_output_2.json",
        "fixed_select_output.json",
    ):
        value = json.loads((root / "tests/fixtures/phase1" / name).read_bytes())
        OntologyDraft.model_validate(value)
        result["rows"].append({"fixture": name, **counts(value, tokenizer)})
        if name in {"c1_pre_output.json", "c2_query_output.json"}:
            for copies in (3, 5) if name.startswith("c1") else (3, 4):
                stress = demanding_fixture(value, copies)
                result["rows"].append(
                    {
                        "fixture": f"authored-stress-{copies}x-{name}",
                        "nodes": stress["budget_accounting"]["nodes_used"],
                        **counts(stress, tokenizer),
                    }
                )
    import zstandard

    journals = list(
        (args.v10_root / "artifacts/restricted/http_diagnostics/").glob(
            "fallback-qwen3-8b-awq-development-v10/*/events.jsonl"
        )
    )
    if len(journals) != 1:
        raise ValueError("exact V10 diagnostic journal required")
    raw = bytearray()
    for line in journals[0].read_bytes().splitlines():
        event = json.loads(line)
        if event["event"] == "response_fragment":
            record = event["artifact"]
            fragment = zstandard.ZstdDecompressor().decompress(
                (journals[0].parent / "fragments" / record["relative_path"]).read_bytes()
            )
            assert hashlib.sha256(fragment).hexdigest() == record["content_hash"]
            raw.extend(fragment)
    assert (
        hashlib.sha256(raw).hexdigest()
        == "2255e9b240a2566a4e82f4bb6b8b41a7de9ece78ce7f634b96bf49c3ba7e3715"
    )
    envelope = json.loads(raw)
    content = envelope["choices"][0]["message"]["content"]
    result["v10"] = {
        "usage": envelope["usage"],
        "finish_reason": envelope["choices"][0]["finish_reason"],
        "content_characters": len(content),
        "whitespace_characters": sum(c.isspace() for c in content),
        "reasoning_field": envelope["choices"][0]["message"].get("reasoning_content"),
        "reasoning_markers_in_content": bool(re.search(r"</?think>", content)),
        "content_tokens": len(tokenizer.encode(content, add_special_tokens=False)),
    }
    if args.reuse_c0_capacity:
        previous = json.loads(args.reuse_c0_capacity.read_bytes())
        result["rows"].extend(
            r for r in previous["rows"] if "production-c0-capacity" in r["fixture"]
        )
        result["c0_backend"] = previous["c0_backend"]
        result["c0_capacity_reused_from_sha256"] = hashlib.sha256(
            args.reuse_c0_capacity.read_bytes()
        ).hexdigest()
        result["c0_cpu_seconds_reused_not_measured_again"] = previous["c0_cpu_seconds"]
        result["cpu_job"]["ended_at"] = datetime.now(UTC).isoformat()
        write_json_atomic(result, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
        return

    from story_projection_onto.conditions.c0 import load_production_classical_builder
    from story_projection_onto.development_adapter import DevelopmentConstructionConfiguration
    from story_projection_onto.development_continuation import load_development_prequery_evidence
    from story_projection_onto.development_runtime import load_development_call_manifest

    start = time.monotonic()
    builder, backend = load_production_classical_builder()
    manifest = load_development_call_manifest(root)
    neutral, _, _ = load_development_prequery_evidence(root, manifest)
    config = DevelopmentConstructionConfiguration.load()
    for unit, artifact in neutral.items():
        now = max(datetime.now(UTC), artifact.snapshot.sealed_at + timedelta(seconds=1))
        try:
            preparation = builder.prepare(
                snapshot=artifact.snapshot,
                evidence=artifact.evidence,
                upper_ontology=config.upper_ontology,
                preconstruction_budgets=config.preconstruction_budgets,
                constructed_at=now,
                sealed_at=now + timedelta(seconds=1),
            )
            draft = preparation.sealed_preontology.draft
            value = draft.model_dump(mode="json", exclude_defaults=True)

            # Drop only derivable record envelope hashes/versions for the capacity
            # fixture, not semantics; canonical validation computes them again.
            def strip(item):
                if isinstance(item, dict):
                    return {
                        k: strip(v)
                        for k, v in item.items()
                        if k not in {"content_hash", "schema_version"}
                    }
                if isinstance(item, list):
                    return [strip(v) for v in item]
                return item

            value = strip(value)
            assert OntologyDraft.model_validate(value).content_hash == draft.content_hash
            write_json_atomic(value, args.output.parent / f"{unit}.c0-capacity-draft.json")
            result["rows"].append(
                {
                    "fixture": unit + "-production-c0-capacity-only",
                    "nodes": len(draft.instance_graph.entities) + len(draft.instance_graph.events),
                    "assertions": len(draft.instance_graph.assertions),
                    **counts(value, tokenizer),
                }
            )
        except Exception as exc:
            result["rows"].append({"fixture": unit, "c0_failure": f"{type(exc).__name__}: {exc}"})
    result["c0_cpu_seconds"] = time.monotonic() - start
    result["c0_backend"] = backend.model_dump(mode="json")
    write_json_atomic(result, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
