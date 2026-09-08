#!/usr/bin/env python3
"""Restricted actual-response -> retry replay. No service or inference entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

from jsonschema import Draft202012Validator

from story_projection_onto.contracts import PreconstructionRequest, canonical_sha256
from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.semantic_generation import ExecutionFacts, build_clarified_small_request
from story_projection_onto.semantic_identifiers import IdentifierResolutionError


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capacity_checks(output, tokenizer_path):
    """Additional authored checks only; never model-visible or source responses."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transformers import AutoTokenizer

    from scripts.verify_semantic_interface_cpu import schema_order
    from tests.unit.test_small_retry_path import demanding_capacity_wire, typed_authored

    if (output / "CAPACITY.json").exists():
        raise ValueError("capacity predecessor must be preserved")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    schema = read(output / "semantic-first/generation.schema.json")
    rows, strings = [], []
    for attributed in (False, True):
        wire = typed_authored(demanding_capacity_wire(attributed=attributed))
        Draft202012Validator(schema).validate(wire)
        ordered = schema_order(wire, schema, schema["$defs"])
        compact = json.dumps(ordered, separators=(",", ":"))
        standard = json.dumps(ordered)
        # Installed XGrammar's fixed-whitespace default requires single-space
        # separators. Compact/no-space JSON is NOT its accepted output language.
        strings.append(standard)
        tokens = [
            len(tokenizer.encode(text, add_special_tokens=False)) for text in (compact, standard)
        ]
        assert max(tokens) < 3584
        rows.append(
            {
                "authored_fixture": "attributed_capacity_only" if attributed else "direct",
                "nodes": 4,
                "assertions": 3,
                "propositions": 3 if attributed else 0,
                "compact_tokens": tokens[0],
                "standard_spaced_tokens": tokens[1],
                "allowance": 3584,
                "scientific_acceptance_claimed": False,
            }
        )
        write_json_atomic(
            wire, output / ("authored-attributed.json" if attributed else "authored-direct.json")
        )
    write_json_atomic(
        {"rows": rows, "capacity_only_not_model_reliability": True}, output / "CAPACITY.json"
    )
    # JSON-schema property order is preserved INSIDE these strings for the
    # pinned grammar's accept_string check, not confused with JSON equivalence.
    write_json_atomic(
        {
            "schemas": [schema, read(output / "semantic-second/generation.schema.json")],
            "good_json": strings,
            "bad_commitment_json": strings[1].replace('"holder_attributed"', '"world_committed"'),
        },
        output / "decoder-controls.json",
    )
    print(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-only", action="store_true")
    args = parser.parse_args()
    if args.capacity_only:
        capacity_checks(args.output, args.tokenizer)
        return
    if args.output.exists() or "restricted" not in args.output.parts:
        raise ValueError("fresh restricted output required; predecessors cannot be overwritten")
    args.output.mkdir(parents=True, mode=0o700)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "capacity_driver", root / "scripts/run_capacity_diagnostics.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    manifest = TokenizerManifest(
        **read(args.source / "rendered-semantic-first.json")["tokenizer_manifest"]
    )
    for filename, digest in manifest.tokenizer_file_sha256:
        assert sha(args.tokenizer / filename) == digest, filename
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    terminal = read(args.source / "terminal.json")
    ledger_path = args.source.parent.parent / "phase1_acceptance.sqlite"
    ledger = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    hashes = {"terminal.json": sha(args.source / "terminal.json")}
    rows = []
    for number, kind in enumerate(("semantic-first", "semantic-second"), 1):
        fixture = PreconstructionRequest.model_validate_json(
            (args.source / f"fixture-{kind}.json").read_bytes()
        )
        original_outcome = terminal["outcomes"][number - 1]
        attempt = args.source / original_outcome["attempt_id"]
        previous = read(attempt / "decoded.json")
        for name in ("decoded.json", "failure.json", "outcome.json"):
            hashes[str((attempt / name).relative_to(args.source))] = sha(attempt / name)
        request = replace(
            build_clarified_small_request(
                fixture,
                tokenizer,
                manifest,
                (root / "prompts/diagnostics/semantic_instruction_v2.md").read_text(),
            ),
            request_id=kind,
        )
        assert request.request_hash == original_outcome["request_hash"]
        assert canonical_sha256(previous) == original_outcome["response"]["parsed_object_sha256"]
        destination = args.output / kind
        destination.mkdir(mode=0o700)
        # Original execution facts are used only to attempt unchanged offline
        # reconstruction. They are not recorded as a new execution or success.
        from datetime import datetime

        started, ended = ledger.execute(
            "SELECT started_at,ended_at FROM gpu_events WHERE event_id=?",
            (original_outcome["attempt_id"],),
        ).fetchone()
        facts = ExecutionFacts(
            request_hash=request.request_hash,
            response_hash=original_outcome["response"]["response_sha256"],
            generation_started_at=datetime.fromisoformat(started),
            generation_completed_at=datetime.fromisoformat(ended),
            input_tokens=original_outcome["template_inclusive_input_tokens"],
            output_tokens=original_outcome["response"]["completion_tokens"],
        )
        try:
            driver.reconstruct_small_response(
                previous,
                fixture,
                facts,
                record_audit=lambda value, destination=destination: write_json_atomic(
                    value, destination / "full-identifier-audit.json"
                ),
            )
        except IdentifierResolutionError as exc:
            failure = driver.semantic_failure_record(exc, "canonical_schema_validation")
        else:
            raise AssertionError("retained failure unexpectedly reconstructed")
        write_json_atomic(failure, destination / "offline-failure-v3.json")
        preparation = {}
        retry = driver.prepare_structural_semantic_retry(
            request, failure, tokenizer, previous_response=previous, preparation=preparation
        )
        if retry is None:
            write_json_atomic(preparation, destination / "preparation-failure.json")
            raise AssertionError(preparation.get("stop_reason"))
        write_json_atomic(preparation, destination / "preparation.json")
        write_json_atomic(preparation["feedback"], destination / "feedback.json")
        write_json_atomic(retry.wire_payload(), destination / "request.json")
        write_json_atomic(retry.output_schema, destination / "generation.schema.json")
        write_json_atomic(retry.packing.model_dump(mode="json"), destination / "packing.json")
        write_json_atomic(retry.decoding.model_dump(mode="json"), destination / "decoding.json")
        messages = [asdict(m) for m in retry.messages]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        actual = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        assert len(actual) == retry.rendered_input_token_count
        message_tokens = [
            len(tokenizer.encode(m["content"], add_special_tokens=False)) for m in messages
        ]
        write_json_atomic(
            {"text": rendered, "tokens": len(actual)}, destination / "rendered-chat.json"
        )
        grammar_failures = [
            {"path": "/" + "/".join(map(str, e.absolute_path)), "validator": e.validator}
            for e in Draft202012Validator(retry.output_schema).iter_errors(previous)
        ]
        row = {
            "example": kind,
            "original_accepted": False,
            "new_model_execution": False,
            "historical_failure_message_chars": len(read(attempt / "failure.json")["message"]),
            "compact_feedback_chars": len(retry.messages[-1].content),
            "original_request_hash": request.request_hash,
            "repair_request_hash": retry.request_hash,
            "original_response_hash": canonical_sha256(previous),
            "input_tokens": len(actual),
            "reserved_output_tokens": retry.decoding.maximum_output_tokens,
            "total_reserved_tokens": len(actual) + retry.decoding.maximum_output_tokens,
            "context_limit": 12288,
            "context_headroom": 12288 - len(actual) - retry.decoding.maximum_output_tokens,
            "message_content_tokens": dict(
                zip(
                    ("system", "complete_evidence", "previous_response", "feedback"),
                    message_tokens,
                    strict=True,
                )
            ),
            "template_special_token_delta": len(actual) - sum(message_tokens),
            "truncation_applied": retry.packing.truncation_applied,
            "unchanged_output_revised_schema_errors": grammar_failures,
            "unchanged_output_canonical_status": "failed",
            "scientific_assessment": "not_reached",
        }
        rows.append(row)
        readable = "# CPU-prepared repair; NOT executed\n\n"
        readable += (
            "Complete unchanged evidence and prior model JSON are included. No scorer answers.\n\n"
        )
        readable += (
            "## Exact compact feedback\n\n```json\n"
            + json.dumps(preparation["feedback"], indent=2)
            + "\n```\n\n"
        )
        for i, message in enumerate(messages):
            readable += f"## Message {i + 1}: {message['role']}\n\n{message['content']}\n\n"
        readable += (
            "## Exact constrained schema\n\n```json\n"
            + json.dumps(retry.output_schema, indent=2)
            + "\n```\n"
        )
        # Artifact generation, not manual source editing. Atomic output helper.
        temporary = destination / "REQUEST_AND_FEEDBACK.md.partial"
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(readable)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination / "REQUEST_AND_FEEDBACK.md")
    ledger.close()
    for rel, digest in hashes.items():
        assert sha(args.source / rel) == digest
    summary = {
        "revision": "actual-failure-to-retry-cpu-v4",
        "gpu_seconds_added": 0,
        "historical_allocation_unchanged_seconds": 6321.388643,
        "original_file_hashes_preserved": hashes,
        "tokenizer_manifest": manifest.public_manifest(),
        "rows": rows,
    }
    write_json_atomic(summary, args.output / "CPU_REPLAY.json")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prepared_requests": len(rows),
                "tokens": [r["total_reserved_tokens"] for r in rows],
                "gpu_seconds_added": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
