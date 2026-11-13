#!/usr/bin/env python3
"""Compile or replay the immutable Phase 7 failure/resource accounting tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.final_accounting import (  # noqa: E402
    FinalAccountingError,
    compile_final_accounting,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Self-hashed final accounting recipe below --source-root.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Real, symlink-free root containing every recipe input.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/public/phase7/accounting"),
        help="Directory for immutable content-addressed outputs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompile in memory and require all exact outputs; write nothing.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    source_root = options.source_root.absolute()
    recipe = options.recipe
    if not recipe.is_absolute():
        recipe = source_root / recipe
    output_root = options.output_root
    if not output_root.is_absolute():
        output_root = source_root / output_root
    try:
        outputs = compile_final_accounting(
            recipe_path=recipe,
            source_root=source_root,
            output_root=output_root,
            verify_only=options.verify,
        )
    except FinalAccountingError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        return 2
    payload = {
        "status": "verified" if options.verify else "compiled",
        "failure_table": outputs.failure_table_path.name,
        "resource_table": outputs.resource_table_path.name,
        "receipt": outputs.receipt_path.name,
        "receipt_manifest_sha256": outputs.receipt["manifest_sha256"],
        "total_allocated_gpu_microseconds": outputs.receipt["reconciliation"][
            "total_allocated_gpu_microseconds"
        ],
        "all_gpu_services_stopped": outputs.receipt["reconciliation"]["all_gpu_services_stopped"],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
