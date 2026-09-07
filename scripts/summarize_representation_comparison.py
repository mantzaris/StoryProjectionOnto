#!/usr/bin/env python3
"""Read immutable restricted diagnostic evidence; never repair or accept outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonschema
import zstandard

from story_projection_onto.contracts import OntologyDraft
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.output_wire import RecordTupleCodec, translate_references
from story_projection_onto.streaming_chat import ChatSSE


def summarize_semantic(run: Path, output: Path, readable: Path):
    """Render actual small semantic outputs, retaining every stage distinction."""
    if any("restricted" not in p.parts or p.exists() for p in (output, readable)):
        raise ValueError("fresh restricted summary and readable destinations required")
    terminal = json.loads((run / "terminal.json").read_bytes())
    rows = []
    pages = ["# Small semantic-interface diagnostics (not production acceptance)\n"]
    for outcome in terminal["outcomes"]:
        label = outcome["diagnostic_kind"]
        request = json.loads((run / f"prepared-{label}.json").read_bytes())
        base = "semantic-first" if label.startswith("semantic-first") else "semantic-second"
        fixture = json.loads((run / f"fixture-{base}.json").read_bytes())
        journals = list((run / "http").glob(outcome["request_hash"][:16] + "-*/events.jsonl"))
        if len(journals) != 1:
            raise ValueError("diagnostic journal must resolve uniquely")
        journal = journals[0]
        parser, raw = ChatSSE(), b""
        for line in journal.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "response_fragment":
                continue
            artifact = event["artifact"]
            fragment = zstandard.ZstdDecompressor().decompress(
                (journal.parent / "fragments" / artifact["relative_path"]).read_bytes()
            )
            if hashlib.sha256(fragment).hexdigest() != artifact["content_hash"]:
                raise ValueError("stream fragment hash mismatch")
            raw += fragment
            parser.feed(fragment, 0)
        metadata = outcome["transport_metadata"]
        if (
            metadata.get("response_sha256")
            and hashlib.sha256(raw).hexdigest() != metadata["response_sha256"]
        ):
            raise ValueError("complete response hash mismatch")
        row = {
            "attempt_id": outcome["attempt_id"],
            "task": label,
            "request_hash": outcome["request_hash"],
            "input_tokens": outcome["template_inclusive_input_tokens"],
            "output_allowance": outcome["output_allowance"],
            "generation_seconds": outcome["generation_wall_seconds"],
            "response_complete": metadata.get("response_complete", False),
            "generation_schema_valid": outcome["generation_schema_valid"],
            "canonical_reconstruction": outcome["canonical_schema_valid"],
            "reference_cross_field_valid": outcome["reference_cross_field_valid"],
            "scientific_task_accepted": outcome["accepted"],
            "failure": outcome["failure"],
            "first_event_seconds": metadata.get("stream_first_event_seconds"),
            "first_content_seconds": metadata.get("stream_first_content_seconds"),
            "evidence": fixture["evidence"],
            "transport": metadata,
        }
        try:
            envelope = parser.envelope()
            content = envelope["choices"][0]["message"]["content"]
            row.update(
                model_content=content,
                finish_reason=parser.finish_reason,
                usage=parser.usage,
                whitespace_characters=sum(c.isspace() for c in content),
            )
            row["syntax_valid"] = False
            wire = json.loads(content)
            row["syntax_valid"] = True
            errors = list(jsonschema.Draft202012Validator(request["guided_json"]).iter_errors(wire))
            row["schema_replay_errors"] = [e.message for e in errors]
            graph = wire["instance_graph"]
            nodes = graph["entities"] + graph["events"]
            ids = [n.get("entity_id", n.get("event_id")) for n in nodes]
            row.update(
                node_count=len(nodes),
                assertion_count=len(graph["assertions"]),
                decision_count=len(wire["decisions"]),
                required_structure=2 <= len(nodes) <= 4
                and len(ids) == len(set(ids))
                and 1 <= len(graph["assertions"]) <= 3
                and bool(wire["decisions"]),
            )
        except (ValueError, KeyError) as error:
            row["replay_failure"] = str(error)
        rows.append(row)
        pages.append(f"## {outcome['attempt_id']} — {label}\n\n### Supplied evidence\n")
        pages.extend(f"- `{e['evidence_id']}`: {e['text']}\n" for e in fixture["evidence"])
        pages.append(
            "\n### Observed validation outcome\n\n```json\n"
            + json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"model_content", "evidence", "transport"}
                },
                indent=2,
            )
            + "\n```\n"
        )
        pages.append(
            "\n### Exact model-generated semantic graph\n\n```json\n"
            + (
                json.dumps(json.loads(row["model_content"]), indent=2)
                if row.get("syntax_valid")
                else row["model_content"]
            )
            + "\n```\n"
            if "model_content" in row
            else "\nNo complete decoded graph.\n"
        )
    forecast = terminal["remaining_forecast"]
    resources = terminal["resource_samples"]
    value = {
        "rows": rows,
        "actual_global_seconds": terminal["actual_allocated_seconds"],
        "new_session_seconds": terminal["additional_block_seconds"],
        "unused_session_allowance_seconds": 1100 - terminal["additional_block_seconds"],
        "remaining_mandatory_proxy_seconds": forecast["remaining_forecast_seconds"],
        "complete_forecast_proxy_seconds": terminal["actual_allocated_seconds"]
        + forecast["remaining_forecast_seconds"],
        "forecast_uses_terminal_actual_not_historical_template_baseline": True,
        "small_timings_not_production_p95": True,
        "peak_sampled_resources": {
            k: max((r[k] for r in resources), default=None)
            for k in ("gpu_vram_bytes", "process_ram_bytes", "project_storage_bytes")
        },
        "open_allocations": terminal["open_allocations"],
        "open_service_journals": terminal["open_service_journals"],
    }
    write_json_atomic(value, output)
    with readable.open("x") as stream:
        stream.write("\n".join(pages))
    print(json.dumps({k: v for k, v in value.items() if k != "rows"}))


def summarize(run: Path, output: Path):
    if "restricted" not in output.parts or output.exists():
        raise ValueError("new restricted output required")
    frozen = json.loads((run / "comparison-requests.json").read_bytes())
    terminal = json.loads((run / "terminal.json").read_bytes())
    rows = []
    for outcome in terminal["outcomes"]:
        label = outcome["diagnostic_kind"]
        request = json.loads((run / f"prepared-{label}.json").read_bytes())
        journals = list((run / "http").glob(outcome["request_hash"][:16] + "-*/events.jsonl"))
        if len(journals) != 1:
            raise ValueError("response journal must resolve uniquely")
        journal = journals[0]
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        parser = ChatSSE()
        response = b""
        for event in events:
            if event["event"] == "response_fragment":
                artifact = event["artifact"]
                fragment = zstandard.ZstdDecompressor().decompress(
                    (journal.parent / "fragments" / artifact["relative_path"]).read_bytes()
                )
                if hashlib.sha256(fragment).hexdigest() != artifact["content_hash"]:
                    raise ValueError("retained stream fragment hash mismatch")
                response += fragment
                parser.feed(fragment, 0)
        if hashlib.sha256(response).hexdigest() != outcome["transport_metadata"]["response_sha256"]:
            raise ValueError("full retained HTTP response hash mismatch")
        envelope = parser.envelope()
        text = envelope["choices"][0]["message"]["content"]
        wire = json.loads(text)
        row = {
            "variant": label,
            "request_hash": outcome["request_hash"],
            "response_complete": True,
            "json_syntax_valid": True,
            "model_content": text,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "characters": len(text),
            "whitespace_characters": sum(c.isspace() for c in text),
            "usage": parser.usage,
            "finish_reason": parser.finish_reason,
            "first_event_seconds": outcome["transport_metadata"]["stream_first_event_seconds"],
            "first_content_seconds": outcome["transport_metadata"]["stream_first_content_seconds"],
            "generation_event_seconds": outcome["generation_event_seconds"],
            "client_wall_seconds": outcome["generation_wall_seconds"],
            "accepted": outcome["accepted"],
            "scientific_validation_reached": False,
        }
        schema = (
            request.get("guided_json")
            or json.loads((run / "prepared-B.json").read_bytes())["guided_json"]
        )
        try:
            jsonschema.validate(wire, schema)
            row["effective_json_schema_valid"] = True
        except jsonschema.ValidationError as error:
            row.update(effective_json_schema_valid=False, schema_error=error.message)
        canonical = wire
        if label == "C":
            canonical = RecordTupleCodec(
                frozen[label]["canonical_schema"], opaque_aliases=frozen[label]["reference_map"]
            ).decode(wire)
            canonical = translate_references(canonical, frozen[label]["reference_map"], decode=True)
        row["canonical_reconstruction"] = (
            "lossless tuple decode" if label == "C" else "named fields directly; no tuple decoding"
        )
        try:
            OntologyDraft.model_validate(canonical)
            row["canonical_schema_valid"] = True
        except ValueError as error:
            row.update(canonical_schema_valid=False, canonical_errors=str(error))
        graph = canonical["instance_graph"]
        nodes = graph["entities"] + graph["events"]
        ids = [n.get("entity_id", n.get("event_id")) for n in nodes]
        assertions = graph["assertions"]
        row.update(
            node_records=len(nodes),
            distinct_node_ids=len(set(ids)),
            assertions=len(assertions),
            decisions=len(canonical["decisions"]),
            node_labels=[n["label"] for n in nodes],
            asserted_text=[a["why_matters"] for a in assertions],
            canonical=canonical,
        )
        row["count_bounds_satisfied"] = 2 <= len(nodes) <= 4 and 1 <= len(assertions) <= 3
        row["assertions_without_endpoints_or_roles"] = sum(
            not (a.get("subject_id") and a.get("object_id")) and not a.get("roles")
            for a in assertions
        )
        row["required_nonempty_structure"] = (
            row["count_bounds_satisfied"]
            and len(ids) == len(set(ids))
            and not row["assertions_without_endpoints_or_roles"]
            and bool(canonical["decisions"])
        )
        rows.append(row)
    resources = terminal["resource_samples"]
    forecast = terminal["remaining_forecast"]
    pending = forecast.get("pending_acceptance_resume_service_seconds")
    if pending is None:
        # Earlier terminal reports had the inventory subtotal only. Preserve
        # those records, and make the already-registered extra envelope explicit.
        pending = next(
            r["forecast_p95_seconds"]
            for r in forecast["rows"]
            if r["call_class"] == "gpu_session_start"
        )
        inventory = forecast["remaining_forecast_seconds"]
    else:
        inventory = forecast["inventory_remaining_seconds"]
    value = {
        "rows": rows,
        "actual_global_seconds": terminal["actual_allocated_seconds"],
        "new_session_seconds": terminal["additional_block_seconds"],
        "new_session_unused_allowance_seconds": 1200 - terminal["additional_block_seconds"],
        "inventory_remaining_proxy_seconds": inventory,
        "pending_acceptance_resume_service_seconds": pending,
        "remaining_mandatory_proxy_seconds": inventory + pending,
        "complete_forecast_proxy_seconds": terminal["actual_allocated_seconds"]
        + inventory
        + pending,
        "successful_output_p95_established": False,
        "peak_sampled_resources": {
            key: max(r[key] for r in resources)
            for key in ("gpu_vram_bytes", "process_ram_bytes", "project_storage_bytes")
        },
        "resource_violations": [r["violations"] for r in resources if r["violations"]],
        "open_allocations": terminal["open_allocations"],
        "open_service_journals": terminal["open_service_journals"],
    }
    write_json_atomic(value, output)
    print(json.dumps({k: v for k, v in value.items() if k != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--readable", type=Path)
    args = parser.parse_args()
    if args.semantic:
        if args.readable is None:
            parser.error("--semantic requires --readable")
        summarize_semantic(args.run, args.output, args.readable)
    else:
        summarize(args.run, args.output)
