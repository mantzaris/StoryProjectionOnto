#!/usr/bin/env python3
"""Derive the immutable Phase 7 accounting recipe and adapter receipts."""

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
    materialize_final_accounting_recipe,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-recipe",
        type=Path,
        required=True,
        help="Self-hashed routing-only source recipe below --source-root.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Real root containing frozen ledgers, CAS, and native manifests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/restricted/phase7/final-accounting-materialized"),
        help="Child directory for content-addressed recipes and adapter receipts.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-derive and require all immutable outputs without writing.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    source_root = options.source_root.absolute()
    source_recipe = options.source_recipe
    if not source_recipe.is_absolute():
        source_recipe = source_root / source_recipe
    output_root = options.output_root
    if not output_root.is_absolute():
        output_root = source_root / output_root
    try:
        outputs = materialize_final_accounting_recipe(
            source_recipe_path=source_recipe,
            source_root=source_root,
            output_root=output_root,
            verify_only=options.verify,
        )
    except FinalAccountingError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "verified" if options.verify else "materialized",
                "recipe": outputs.recipe_path.name,
                "recipe_manifest_sha256": outputs.recipe.manifest_sha256,
                "normalized_receipt_count": len(outputs.receipt_paths),
                "executed_inference_slots": sum(
                    item.executed for item in outputs.recipe.call_slots
                ),
                "executed_service_slots": sum(
                    item.executed for item in outputs.recipe.service_slots
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
