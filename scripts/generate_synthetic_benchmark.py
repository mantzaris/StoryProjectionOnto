#!/usr/bin/env python3
"""Materialize or verify the frozen public synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.benchmark import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    materialize_benchmark,
    verify_materialized_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing artifacts without attempting to materialize missing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_materialized_benchmark(args.output_root, args.config)
        print(json.dumps({"status": "verified", "output_root": str(args.output_root)}))
        return
    manifest = materialize_benchmark(args.output_root, args.config)
    print(
        json.dumps(
            {
                "status": "materialized_and_verified",
                "output_root": str(args.output_root),
                "benchmark_manifest_hash": manifest.content_hash,
                "development_worlds": manifest.development_world_count,
                "held_out_worlds": manifest.held_out_world_count,
                "held_out_contexts": manifest.held_out_context_count,
                "review_lifecycle_state": manifest.review_lifecycle_state,
                "review_complete": manifest.review_complete,
                "held_out_launch_authorized": manifest.held_out_launch_authorized,
                "review_package_hash": manifest.review_package_hash,
                "draft_seal_hash": manifest.draft_seal_hash,
                "final_reviewed_seal_hash": manifest.final_reviewed_seal_hash,
                "condition_outputs_generated": manifest.condition_outputs_generated,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
