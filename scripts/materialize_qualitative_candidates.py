#!/usr/bin/env python3
"""Apply frozen qualitative rules and create comparable fixed-grid PNG composites."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from story_projection_onto.scorer_only.qualitative_materializer import (
    QualitativeMaterializationError,
    materialize_qualitative_candidates,
    prepare_qualitative_materialization,
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reporting-policy",
        type=Path,
        default=Path("configs/study/reporting.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(arguments)


def _under(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    inputs = {
        "restricted_root": options.restricted_root,
        "source_manifest_path": _under(
            options.restricted_root, options.source_manifest
        ),
        "reporting_policy_path": options.reporting_policy,
    }
    try:
        if options.validate_only:
            candidates, composites, receipt, files = prepare_qualitative_materialization(
                **inputs
            )
            result = {
                "state": "validated",
                "candidate_count": len(candidates.candidates),
                "composite_count": len(composites.figures),
                "selected_count": len(receipt.selected_candidate_ids),
                "receipt_hash": receipt.content_hash,
                "planned_file_count": len(files),
                "writes_performed": False,
            }
        else:
            output, state, receipt = materialize_qualitative_candidates(
                **inputs,
                output_root=_under(options.restricted_root, options.output_root),
            )
            result = {
                "state": state,
                "selected_count": len(receipt.selected_candidate_ids),
                "receipt_hash": receipt.content_hash,
                "output_leaf": output.name,
                "writes_performed": state == "created",
            }
        print(json.dumps(result, sort_keys=True))
        return 0
    except QualitativeMaterializationError as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
