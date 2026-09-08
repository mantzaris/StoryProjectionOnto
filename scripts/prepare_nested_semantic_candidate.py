#!/usr/bin/env python3
"""CPU-only restricted review package. No service or generation entry point."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from story_projection_onto.contracts import PreconstructionRequest, canonical_sha256
from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.ledger_verify import derive_call_status
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.nested_semantic_candidate import (
    REVISION,
    build_candidate_request,
    collapse_candidate,
    expand_candidate,
    reconstruct_candidate,
)
from tests.unit.test_nested_semantic_candidate import authored_candidate
from tests.unit.test_semantic_generation import execution
from tests.unit.test_small_retry_path import demanding_capacity_wire
from tests.unit.test_small_semantic_reconciliation import second_binary_wire

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "artifacts/restricted/parent-repair-session-backup.vaJysn"
RUN = BACKUP / "artifacts/restricted/small-parent-linked-repairs-20260908/run-20260908T122728136382"
OUTPUT = ROOT / "artifacts/restricted/nested-semantic-candidate-cpu-v1"


def emit(path, value):
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f"immutable candidate artifact already differs: {path.name}")
        return
    write_json_atomic(value, path)


def prepare():
    fixture = PreconstructionRequest.model_validate_json(
        (RUN / "fixture-semantic-second.json").read_bytes()
    )
    manifest = TokenizerManifest(
        **json.loads((RUN / "rendered-semantic-first.json").read_bytes())["tokenizer_manifest"]
    )
    tokenizer_path = ROOT / "artifacts/restricted/pinned-tokenizer-cpu"
    for filename, digest in manifest.tokenizer_file_sha256:
        assert hashlib.sha256((tokenizer_path / filename).read_bytes()).hexdigest() == digest
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=False
    )
    assert (
        hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        == manifest.chat_template_sha256
    )
    assert tokenizer.eos_token_id == manifest.eos_token_id
    instruction = (ROOT / "prompts/diagnostics/nested_content_candidate_v1.md").read_text()
    request = build_candidate_request(fixture, tokenizer, manifest, instruction)
    messages = [asdict(m) for m in request.messages]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    assert len(token_ids) == request.rendered_input_token_count
    assert len(tokenizer.encode(rendered, add_special_tokens=False)) == len(token_ids)
    assert len(token_ids) + request.decoding.maximum_output_tokens <= 12288
    assert not request.packing.truncation_applied
    wire = request.wire_payload()
    assert wire["messages"] == messages
    assert wire["guided_json"] == request.output_schema
    emit(OUTPUT / "model-facing/request.json", wire)
    emit(OUTPUT / "model-facing/generation.schema.json", request.output_schema)
    emit(OUTPUT / "model-facing/rendered-chat.json", {"text": rendered, "token_ids": token_ids})
    emit(OUTPUT / "model-facing/fixture.json", fixture.model_dump(mode="json"))
    # Full human-readable request is JSON: messages carry every semantic
    # instruction and intact evidence; grammar is the exact effective object.
    emit(
        OUTPUT / "model-facing/READABLE_REQUEST.json",
        {
            "notice": (
                "Complete CPU candidate only; no expected output, scorer answers "
                "or prior model response."
            ),
            "messages": messages,
            "effective_schema": request.output_schema,
            "decoding": request.decoding.model_dump(mode="json"),
        },
    )
    authored = authored_candidate(second_binary_wire())
    expanded = expand_candidate(
        authored, evidence=fixture.evidence, upper=fixture.upper_ontology, small=True
    )
    assert collapse_candidate(expanded) == authored
    adapted = reconstruct_candidate(
        authored,
        evidence=fixture.evidence,
        upper=fixture.upper_ontology,
        execution=execution(),
        small=True,
    )
    emit(OUTPUT / "authored-controls/output-NOT-MODEL-VISIBLE.json", authored)
    emit(OUTPUT / "authored-controls/expanded.json", dict(expanded.semantic_payload))
    emit(
        OUTPUT / "authored-controls/representation-receipt.json",
        dict(expanded.representation_receipt),
    )
    emit(OUTPUT / "authored-controls/canonical.json", adapted.draft.model_dump(mode="json"))
    emit(
        OUTPUT / "authored-controls/provenance.json",
        {
            "execution_facts": (
                "Synthetic CPU test placeholders, not measured inference or allocation."
            ),
            **adapted.provenance,
        },
    )
    controls = {}
    for name, obj in {
        "ev03_two_node_direct": authored,
        "four_node_three_direct_capacity_only": authored_candidate(demanding_capacity_wire()),
        "four_node_attributed_capacity_NOT_grounded_in_direct_evidence": authored_candidate(
            demanding_capacity_wire(attributed=True)
        ),
    }.items():
        encoded = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
        controls[name] = len(tokenizer.encode(encoded, add_special_tokens=False))
        emit(OUTPUT / f"authored-controls/{name}.json", obj)
    assert max(controls.values()) <= request.decoding.maximum_output_tokens
    packing = {
        "revision": REVISION,
        "request_hash": request.request_hash,
        "schema_hash": canonical_sha256(request.output_schema),
        "tokenizer_manifest": manifest.public_manifest(),
        "template_inclusive_input_tokens": len(token_ids),
        "reserved_output_tokens": request.decoding.maximum_output_tokens,
        "maximum_input_tokens": request.decoding.maximum_input_tokens,
        "maximum_context_tokens": 12288,
        "total_reserved": len(token_ids) + request.decoding.maximum_output_tokens,
        "authored_output_capacity_tokens": controls,
        "wire_schema_is_decoder_configuration_not_an_extra_chat_message": True,
        "packing_report": request.packing.model_dump(mode="json"),
        "round_trip_exact": True,
        "authored_fixture_canonical_reconstruction": True,
        "generation_reliability_proven": False,
        "gpu_calls": 0,
        "live_backend_grammar_compilation_tested": False,
        "production_adoption_authorized": False,
    }
    emit(OUTPUT / "CAPACITY_AND_ROUND_TRIP.json", packing)
    ledger = BACKUP / "artifacts/restricted/phase1_acceptance.sqlite"
    ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        calls = [dict(r) for r in connection.execute("SELECT * FROM model_calls")]
        events = {r["event_id"]: dict(r) for r in connection.execute("SELECT * FROM gpu_events")}
        failures = [dict(r) for r in connection.execute("SELECT * FROM failures")]
        validations = [dict(r) for r in connection.execute("SELECT * FROM validations")]
        rows = []
        for call in calls:
            event = events.get(call["gpu_event_id"])
            if event is None or event["succeeded"] == call["successful"]:
                continue
            related_failures = [r for r in failures if r["attempt_id"] == call["attempt_id"]]
            rows.append(
                {
                    **derive_call_status(call, event, failures, validations, diagnostic_only=True),
                    "source_model_call_hash": canonical_sha256(call),
                    "source_gpu_event_hash": canonical_sha256(event),
                    "source_failure_hashes": [canonical_sha256(r) for r in related_failures],
                }
            )
        total = (
            connection.execute("SELECT SUM(allocated_microseconds) FROM gpu_events").fetchone()[0]
            + connection.execute(
                "SELECT SUM(overhead_microseconds) FROM gpu_service_sessions"
            ).fetchone()[0]
        )
    assert len(rows) == 9 and all(r["scientific_status"] == "rejected" for r in rows)
    assert total == 6716108081
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == ledger_hash
    emit(
        OUTPUT / "LEDGER_STATUS_CLARIFICATION.json",
        {
            "revision": "execution-versus-science-derived-v1",
            "source_ledger_sha256": ledger_hash,
            "historical_rows_modified": False,
            "strict_historical_verifier_mismatches_waived": False,
            "actual_allocated_microseconds": total,
            "derived_rows": rows,
        },
    )
    paths = sorted(p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    emit(
        OUTPUT / "MANIFEST.json",
        {
            "revision": REVISION,
            "release_class": "restricted",
            "files": [
                {
                    "path": str(p.relative_to(OUTPUT)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "size": p.stat().st_size,
                }
                for p in paths
            ],
        },
    )
    print(
        json.dumps(
            {
                "input": len(token_ids),
                "reserved_output": request.decoding.maximum_output_tokens,
                "authored_capacity": controls,
                "derived_status_rows": len(rows),
                "actual_seconds": total / 1e6,
            }
        )
    )


if __name__ == "__main__":
    prepare()
