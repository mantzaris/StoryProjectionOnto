#!/usr/bin/env python3
"""Build the allowlist-only, copyright-safe public artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_projection_onto.public_release import build_public_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/public_bundle_inputs.json"),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("artifacts/public/release/StoryProjectionOnto-public"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_public_bundle(args.source_root, args.manifest, args.bundle_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
