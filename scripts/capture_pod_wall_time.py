#!/usr/bin/env python3
"""Capture a public-safe current-container wall-time sample."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.gpu_runtime import atomic_write_public_json  # noqa: E402
from story_projection_onto.pod_runtime import capture_pod_wall_time  # noqa: E402


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    atomic_write_public_json(
        options.output,
        capture_pod_wall_time(proc_root=options.proc_root.resolve(strict=True)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
