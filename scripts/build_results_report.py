#!/usr/bin/env python3
"""Build or verify the immutable-table conference results report."""

from __future__ import annotations

import argparse
from pathlib import Path

from story_projection_onto.report_ingestion import verify_ingestion_from_files
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
        "--ingestion-manifest",
        type=Path,
        default=Path("configs/study/report_ingestion.json"),
        help="Frozen predecessor/table registry used to replay ingestion.",
    )
    parser.add_argument(
        "--ingestion-receipt",
        type=Path,
        default=Path("reports/report_ingestion_receipt.json"),
    )
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--table-root", type=Path, default=Path("reports"))
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
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else args.source_root / args.manifest
    )
    ingestion_manifest_path = (
        args.ingestion_manifest
        if args.ingestion_manifest.is_absolute()
        else args.source_root / args.ingestion_manifest
    )
    ingestion_receipt_path = (
        args.ingestion_receipt
        if args.ingestion_receipt.is_absolute()
        else args.source_root / args.ingestion_receipt
    )
    table_root = (
        args.table_root if args.table_root.is_absolute() else args.source_root / args.table_root
    )
    policy_path = args.policy if args.policy.is_absolute() else args.source_root / args.policy
    verify_ingestion_from_files(
        ingestion_manifest_path,
        ingestion_receipt_path,
        source_root=args.source_root,
        table_root=table_root,
    )
    if args.verify:
        verify_report_build(manifest_path, policy_path, args.output_root)
    else:
        build_results_report(manifest_path, policy_path, args.output_root)
        write_reproducibility_report(manifest_path, policy_path, args.output_root)
        verify_report_build(manifest_path, policy_path, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
