#!/usr/bin/env python3
"""Validate or materialize externally authored Phase 2 review records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.independent_review_runtime import (
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_COMPLETION_ROOT,
    ReviewCompletionError,
    materialize_review_completion,
    prepare_review_completion,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Reproduce the independent-review gate without generating any reviewer decision."
        )
    )
    command.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    command.add_argument("--response", type=Path, required=True)
    command.add_argument("--adjudication", type=Path, required=True)
    command.add_argument("--amendments", type=Path)
    command.add_argument("--output-root", type=Path, default=DEFAULT_COMPLETION_ROOT)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    try:
        completion = prepare_review_completion(
            benchmark_root=options.benchmark_root,
            response_path=options.response,
            adjudication_path=options.adjudication,
            amendment_path=options.amendments,
        )
        state = "validated"
        if options.materialize:
            state = materialize_review_completion(completion, options.output_root)
        summary = {
            "state": state,
            "writes_performed": bool(options.materialize and state == "created"),
            "completion_manifest_hash": completion.manifest.content_hash,
            "final_seal_hash": completion.final_seal.content_hash,
            "review_item_count": 72,
            "reviewed_projection_count": 9,
            "held_out_launch_authorized": True,
        }
        if options.dry_run:
            summary["planned_relative_files"] = sorted(
                [
                    "response.json",
                    "adjudication.json",
                    "final_seal.json",
                    "completion_manifest.json",
                    *[
                        entry.artifact_file
                        for entry in completion.manifest.reviewed_artifacts
                    ],
                ]
            )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except ReviewCompletionError as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
