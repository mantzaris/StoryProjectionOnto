#!/usr/bin/env python3
"""Create a strict local/RunPod source-tree association artifact."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.manifest import (  # noqa: E402
    build_source_association,
    write_json_atomic,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--remote-manifest", type=Path, required=True)
    parser.add_argument(
        "--branch",
        default="implementation/query-dependent-temporal-ontology",
    )
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--revision-label", required=True)
    parser.add_argument(
        "--recorded-at",
        required=True,
        help="Explicit timezone-aware ISO-8601 timestamp",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    try:
        recorded_at = datetime.fromisoformat(options.recorded_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--recorded-at must be a valid ISO-8601 timestamp") from exc
    destination = options.output.resolve()
    local = options.local_manifest.resolve(strict=True)
    remote = options.remote_manifest.resolve(strict=True)
    if local.parent != destination.parent or remote.parent != destination.parent:
        raise SystemExit("both manifests and the association must share one directory")
    association = build_source_association(
        local_manifest_path=options.local_manifest,
        remote_manifest_path=options.remote_manifest,
        branch=options.branch,
        git_commit=options.git_commit,
        revision_label=options.revision_label,
        recorded_at=recorded_at,
    )
    write_json_atomic(association, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
