#!/usr/bin/env python3
"""Verify a candidate synthetic benchmark/schema lineage refresh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.benchmark_refresh_guard import (
    EXACT_LINEAGE_PROJECTION_RULE,
    RefreshVerificationError,
    verify_benchmark_schema_refresh,
    write_refresh_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-benchmark-root", type=Path, required=True)
    parser.add_argument("--candidate-benchmark-root", type=Path, required=True)
    parser.add_argument("--old-schema-root", type=Path, required=True)
    parser.add_argument("--candidate-schema-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-source-root",
        type=Path,
        required=True,
        help="source tree whose four compiler dependencies the candidate seal must bind",
    )
    parser.add_argument("--candidate-benchmark-replay-root", type=Path)
    parser.add_argument("--candidate-schema-replay-root", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if (arguments.candidate_benchmark_replay_root is None) != (
        arguments.candidate_schema_replay_root is None
    ):
        raise SystemExit("candidate benchmark and schema replay roots must be supplied together")
    try:
        receipt = verify_benchmark_schema_refresh(
            old_benchmark_root=arguments.old_benchmark_root,
            candidate_benchmark_root=arguments.candidate_benchmark_root,
            old_schema_root=arguments.old_schema_root,
            candidate_schema_root=arguments.candidate_schema_root,
            candidate_source_root=arguments.candidate_source_root,
            candidate_benchmark_replay_root=arguments.candidate_benchmark_replay_root,
            candidate_schema_replay_root=arguments.candidate_schema_replay_root,
            candidate_projection_rule=EXACT_LINEAGE_PROJECTION_RULE,
        )
        write_refresh_receipt(receipt, arguments.receipt)
    except RefreshVerificationError as exc:
        raise SystemExit(f"refresh verification failed: {exc}") from exc
    print(
        json.dumps(
            {
                "status": receipt.status,
                "receipt": str(arguments.receipt),
                "receipt_sha256": receipt.receipt_sha256,
                "difference_count": len(receipt.differences),
                "candidate_replay_checked": receipt.candidate_replay_checked,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
