#!/usr/bin/env python3
"""Render the sealed blind-review package into a restricted human worksheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.scorer_only.independent_review_handoff import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PACKAGE_PATH,
    DEFAULT_RESPONSE_SCHEMA_PATH,
    IndependentReviewHandoffError,
    build_independent_review_handoff,
    materialize_independent_review_handoff,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--package", type=Path, default=DEFAULT_PACKAGE_PATH)
    command.add_argument(
        "--response-schema",
        type=Path,
        default=DEFAULT_RESPONSE_SCHEMA_PATH,
    )
    command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    try:
        handoff = build_independent_review_handoff(
            package_path=options.package,
            response_schema_path=options.response_schema,
        )
        state = "validated"
        if options.materialize:
            state = materialize_independent_review_handoff(
                handoff,
                output_root=options.output_root,
            )
        print(
            json.dumps(
                {
                    "state": state,
                    "writes_performed": state == "created",
                    "handoff_manifest_hash": handoff.manifest.content_hash,
                    "package_hash": handoff.manifest.package_hash,
                    "condition_blind": True,
                    "contains_method_outputs": False,
                    "contains_reviewer_judgments": False,
                    "world_count": 3,
                    "projection_count": 9,
                    "review_item_count": 72,
                    "files": [
                        {
                            "relative_path": item.relative_path,
                            "size_bytes": item.size_bytes,
                            "file_sha256": item.file_sha256,
                        }
                        for item in handoff.manifest.files
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except (IndependentReviewHandoffError, ValueError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
