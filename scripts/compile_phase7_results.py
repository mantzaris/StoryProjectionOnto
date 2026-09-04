#!/usr/bin/env python3
"""Compile or byte-reproduce the canonical Phase 7 conference artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.phase7_compiler import (
    Phase7CompilationError,
    compile_phase7_results,
    verify_phase7_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path("configs/study/phase7_compiler.json"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Self-hashed immutable source registry produced after predecessor closure.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    parser.add_argument("--verify", action="store_true", help="Regenerate and compare only.")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    source_root = options.source_root.resolve(strict=True)
    configuration = (
        options.configuration
        if options.configuration.is_absolute()
        else source_root / options.configuration
    )
    registry = (
        options.registry if options.registry.is_absolute() else source_root / options.registry
    )
    output = (
        options.output_root
        if options.output_root.is_absolute()
        else source_root / options.output_root
    )
    try:
        if options.verify:
            result = verify_phase7_results(
                source_root=source_root,
                configuration_path=configuration,
                registry_path=registry,
                output_root=output,
            )
            action = "verified"
        else:
            result = compile_phase7_results(
                source_root=source_root,
                configuration_path=configuration,
                registry_path=registry,
                output_root=output,
            )
            action = "compiled"
    except (OSError, ValueError, Phase7CompilationError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "state": action,
                "study_status": result.study_status.value,
                "source_registry_sha256": result.registry_sha256,
                "compiler_configuration_sha256": result.configuration_sha256,
                "build_token": result.build_token,
                "immutable_output_count": len(result.output_hashes),
                "current_pointer_sha256": result.current_pointer_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
