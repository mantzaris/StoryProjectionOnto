"""Small, side-effect-explicit command line interface for reproducibility checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from story_projection_onto import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="story-projection-onto")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("version", help="print the installed package version")

    synthetic = subcommands.add_parser(
        "verify-synthetic",
        help="verify the materialized synthetic benchmark without modifying it",
    )
    synthetic.add_argument("--project-root", type=Path, default=Path.cwd())

    development = subcommands.add_parser(
        "validate-development-plan",
        help="validate and summarize the registered 24-call development plan",
    )
    development.add_argument("--project-root", type=Path, default=Path.cwd())
    development.add_argument("--plan", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Dispatch read-only package checks; GPU execution remains in guarded scripts."""

    options = _parser().parse_args(arguments)
    if options.command == "version":
        print(__version__)
        return 0

    root = options.project_root.resolve(strict=True)
    if options.command == "verify-synthetic":
        # Keep scorer-bearing compiler imports out of every other CLI pathway.
        from story_projection_onto.synthetic_benchmark import verify_materialized_benchmark

        output_root = root / "data/synthetic"
        config = root / "configs/study/synthetic_benchmark.json"
        verify_materialized_benchmark(output_root, config)
        print(
            json.dumps(
                {"output_root": str(output_root), "status": "verified"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0

    if options.command == "validate-development-plan":
        from story_projection_onto.development_runtime import (
            canonical_public_summary,
            load_development_call_manifest,
        )

        plan = None if options.plan is None else options.plan.resolve(strict=True)
        manifest = load_development_call_manifest(root, plan)
        print(canonical_public_summary(manifest))
        return 0

    raise AssertionError(f"unhandled command: {options.command}")


__all__ = ["main"]
