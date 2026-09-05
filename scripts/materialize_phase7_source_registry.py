#!/usr/bin/env python3
"""Materialize or verify a self-hashed Phase 7 source registry recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.phase7_compiler import Phase7CompilationError
from story_projection_onto.phase7_registry import (
    verify_phase7_source_registry,
    write_phase7_source_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path("configs/study/phase7_compiler.json"),
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    source_root = options.source_root.resolve(strict=True)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else source_root / path

    try:
        if options.verify:
            registry = verify_phase7_source_registry(
                source_root=source_root,
                configuration_path=resolve(options.configuration),
                recipe_path=resolve(options.recipe),
                registry_path=resolve(options.output),
            )
            state = "verified"
        else:
            registry = write_phase7_source_registry(
                source_root=source_root,
                configuration_path=resolve(options.configuration),
                recipe_path=resolve(options.recipe),
                output_path=resolve(options.output),
            )
            state = "materialized"
    except (OSError, ValueError, Phase7CompilationError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "state": state,
                "registry_id": registry.registry_id,
                "manifest_sha256": registry.manifest_sha256,
                "artifact_count": sum(
                    len(predecessor.artifacts) for predecessor in registry.predecessors
                ),
                "complete_table_count": sum(
                    table.status.value == "complete" for table in registry.tables
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
