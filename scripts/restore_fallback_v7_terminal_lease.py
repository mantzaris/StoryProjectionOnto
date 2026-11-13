#!/usr/bin/env python3
"""Restore the exact terminal fallback-v7 vLLM lease and write its receipt."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.fallback_v7_lease_repair import (  # noqa: E402
    restore_fallback_v7_terminal_lease,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--incident", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--shared-cache", type=Path, required=True)
    parser.add_argument("--verified-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    restore_fallback_v7_terminal_lease(
        project_root=options.project_root,
        incident_path=options.incident,
        snapshot_path=options.snapshot,
        shared_cache=options.shared_cache,
        verified_snapshot_manifest_path=options.verified_snapshot_manifest,
        ledger_path=options.ledger,
        output_path=options.output,
        port=options.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
