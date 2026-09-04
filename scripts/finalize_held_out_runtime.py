#!/usr/bin/env python3
"""Materialize the restricted post-development held-out runtime binding."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from story_projection_onto.held_out_binding import (
    DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH,
    materialize_held_out_runtime_binding,
    persist_restricted_error,
)
from story_projection_onto.held_out_factory import _canonical_restricted_root
from story_projection_onto.held_out_primary import DEFAULT_HELD_OUT_CONTROL_PATH


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repository", type=Path, default=Path("."))
    command.add_argument("--fallback-result", type=Path, required=True)
    command.add_argument("--source-association", type=Path, required=True)
    command.add_argument("--control", type=Path, default=DEFAULT_HELD_OUT_CONTROL_PATH)
    command.add_argument(
        "--development-result",
        type=Path,
        default=Path(
            "artifacts/restricted/fallback-development/development_execution_result.json"
        ),
    )
    command.add_argument(
        "--selected-model-freeze",
        type=Path,
        default=Path("artifacts/restricted/fallback-development/selected_model_freeze.json"),
    )
    command.add_argument(
        "--binding",
        type=Path,
        default=DEFAULT_HELD_OUT_RUNTIME_BINDING_PATH,
    )
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--restricted-root", type=Path, required=True)
    command.add_argument(
        "--created-at",
        required=True,
        help="Externally recorded timezone-aware ISO-8601 completion timestamp",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    try:
        created_at = datetime.fromisoformat(options.created_at.replace("Z", "+00:00"))
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError
    except ValueError:
        print(
            json.dumps(
                {"state": "blocked", "error": "--created-at must include a UTC offset"},
                sort_keys=True,
            )
        )
        return 2
    repository: Path | None = None
    restricted_root: Path | None = None
    try:
        repository = options.repository.resolve()
        restricted_root = _canonical_restricted_root(
            repository,
            options.restricted_root,
        )

        def runtime_path(path: Path) -> Path:
            return path if path.is_absolute() else repository / path

        binding = materialize_held_out_runtime_binding(
            repository=repository,
            fallback_result_path=runtime_path(options.fallback_result),
            source_association_path=runtime_path(options.source_association),
            control_configuration_path=runtime_path(options.control),
            development_result_path=runtime_path(options.development_result),
            selected_model_freeze_path=runtime_path(options.selected_model_freeze),
            binding_path=runtime_path(options.binding),
            ledger_path=runtime_path(options.ledger),
            artifact_root=runtime_path(options.artifact_root),
            restricted_root=restricted_root,
            created_at=created_at,
        )
    except BaseException as error:
        if restricted_root is not None:
            persist_restricted_error(
                restricted_root=restricted_root,
                namespace="held_out_finalizer",
                error=error,
            )
        if isinstance(error, KeyboardInterrupt):
            state = "interrupted"
            error_code = "held_out_finalizer_interrupted"
            return_code = 130
        elif isinstance(error, SystemExit):
            state = "terminated"
            error_code = "held_out_finalizer_system_exit"
            return_code = error.code if isinstance(error.code, int) else 2
        else:
            state = "blocked"
            error_code = "held_out_finalizer_blocked"
            return_code = 2
        print(
            json.dumps(
                {
                    "state": state,
                    "error_code": error_code,
                    "error": "Held-out finalization is blocked; inspect restricted logs",
                },
                sort_keys=True,
            )
        )
        return return_code
    print(
        json.dumps(
            {
                "state": "held_out_runtime_bound",
                "binding_hash": binding.content_hash,
                "development_result_hash": binding.development_execution_result_hash,
                "source_tree_hash": binding.source_tree_sha256,
                "held_out_review_still_required": binding.held_out_review_still_required,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
