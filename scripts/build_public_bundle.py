#!/usr/bin/env python3
"""Build the allowlist-only, copyright-safe public artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_projection_onto.public_release import build_public_bundle
from story_projection_onto.report_ingestion import verify_ingestion_from_files
from story_projection_onto.reporting import verify_report_build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/public_bundle_inputs.json"),
    )
    parser.add_argument(
        "--report-manifest",
        type=Path,
        default=Path("reports/results_manifest.json"),
    )
    parser.add_argument(
        "--report-policy",
        type=Path,
        default=Path("configs/study/reporting.json"),
    )
    parser.add_argument(
        "--ingestion-manifest",
        type=Path,
        default=Path("configs/study/report_ingestion.json"),
    )
    parser.add_argument(
        "--ingestion-receipt",
        type=Path,
        default=Path("reports/report_ingestion_receipt.json"),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("artifacts/public/release/StoryProjectionOnto-public"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_ingestion_from_files(
        args.source_root / args.ingestion_manifest,
        args.source_root / args.ingestion_receipt,
        source_root=args.source_root,
        table_root=args.source_root / "reports",
    )
    verify_report_build(
        args.source_root / args.report_manifest,
        args.source_root / args.report_policy,
        args.source_root / "reports",
    )
    build_public_bundle(
        args.source_root,
        args.source_root / args.manifest,
        args.bundle_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
