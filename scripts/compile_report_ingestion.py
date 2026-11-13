#!/usr/bin/env python3
"""Compile or verify the fail-closed reporting-ingestion receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.report_ingestion import (
    ReportingIngestionError,
    verify_ingestion_from_files,
    write_ingestion_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/study/report_ingestion.json"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("reports/report_ingestion_receipt.json"),
    )
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        default=None,
        help="Explicit hash-bound historical source overrides for --verify only.",
    )
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--table-root", type=Path, default=Path("reports"))
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else args.source_root / args.manifest
    )
    receipt_path = args.receipt if args.receipt.is_absolute() else args.source_root / args.receipt
    source_snapshot_path = (
        None
        if args.source_snapshot is None
        else (
            args.source_snapshot
            if args.source_snapshot.is_absolute()
            else args.source_root / args.source_snapshot
        )
    )
    table_root = (
        args.table_root if args.table_root.is_absolute() else args.source_root / args.table_root
    )
    try:
        if source_snapshot_path is not None and not args.verify:
            raise ReportingIngestionError(
                "a historical source snapshot is permitted only for receipt verification"
            )
        if args.verify:
            receipt = verify_ingestion_from_files(
                manifest_path,
                receipt_path,
                source_root=args.source_root,
                table_root=table_root,
                source_snapshot_manifest_path=source_snapshot_path,
            )
            state = "verified"
        else:
            receipt = write_ingestion_receipt(
                manifest_path,
                receipt_path,
                source_root=args.source_root,
                table_root=table_root,
            )
            state = "compiled"
    except (ReportingIngestionError, OSError, ValueError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "state": state,
                "receipt_sha256": receipt.receipt_sha256,
                "predecessor_count": len(receipt.predecessors),
                "registered_table_count": len(receipt.tables),
                "available_table_count": sum(
                    item.status.value == "complete" for item in receipt.tables
                ),
                "gpu_service_time_sources": len(receipt.gpu_service_time_artifact_ids),
                "runpod_wall_time_sources": len(receipt.runpod_wall_time_artifact_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
