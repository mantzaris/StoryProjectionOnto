#!/usr/bin/env python3
"""Build or verify the immutable-table conference results report."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_projection_onto.reporting import (
    build_results_report,
    verify_report_build,
    write_reproducibility_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/results_manifest.json"),
        help="Immutable result manifest; table paths are relative to its directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Report output directory (defaults to the manifest directory).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/study/reporting.json"),
        help="Frozen Phase-7 reporting and qualitative-selection policy.",
    )
    parser.add_argument("--verify", action="store_true", help="Verify without rewriting outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        verify_report_build(args.manifest, args.policy, args.output_root)
    else:
        build_results_report(args.manifest, args.policy, args.output_root)
        write_reproducibility_report(args.manifest, args.policy, args.output_root)
        verify_report_build(args.manifest, args.policy, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
