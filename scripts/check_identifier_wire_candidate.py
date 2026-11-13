#!/usr/bin/env python3
"""CPU-only pinned grammar check for an unactivated identifier-format repair."""

import argparse
import json
import time
from pathlib import Path

from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.output_wire import RecordTupleCodec, bounded_identifier_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or "restricted" not in args.output.parts:
        raise ValueError("new restricted output required")
    started = time.monotonic()
    from importlib.metadata import version

    import xgrammar
    from transformers import AutoTokenizer

    assert version("xgrammar") == "0.1.23"
    revision = "4da05a8edb55c6046cce958586c33b61da07bb79"
    snapshot = Path(".cache/shared/hub/models--Qwen--Qwen3-8B-AWQ/snapshots") / revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo.from_huggingface(tokenizer))
    schema = bounded_identifier_schema(
        json.loads(Path("schemas/jsonschema/ontology_draft.schema.json").read_bytes())
    )
    compiler.compile_json_schema(
        json.dumps(RecordTupleCodec(schema).wire_schema()), any_whitespace=False
    )
    id_schema = schema["$defs"]["QualifiedAssertion"]["properties"]["assertion_id"]
    compiled = compiler.compile_json_schema(json.dumps(id_schema), any_whitespace=False)
    outcomes = {
        value: xgrammar.GrammarMatcher(compiled).accept_string(value)
        for value in ('"n-assertion-01"', '"0, 0, 0, ', '"' + "n" * 97 + '"')
    }
    assert list(outcomes.values()) == [True, False, False]
    result = {
        "kind": "cpu_capacity_not_model_result",
        "tokenizer_revision": revision,
        "xgrammar": version("xgrammar"),
        "whole_wire_schema_compiled": True,
        "prefix_outcomes": outcomes,
        "candidate_activated": False,
        "gpu_allocated_seconds": 0,
        "cpu_wall_seconds": time.monotonic() - started,
    }
    write_json_atomic(result, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
