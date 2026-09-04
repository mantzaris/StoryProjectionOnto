#!/usr/bin/env python3
"""Create read-only fallback activation and cache-replacement certificates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from story_projection_onto.gpu_runtime import atomic_write_public_json
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    discover_cached_model_repositories,
    fallback_activation_certificate,
    fallback_cache_replacement_receipt,
    migrate_legacy_fallback_artifacts,
)


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} root must be an object")
    return dict(cast(Mapping[str, object], value))


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/study/fallback_model.json"),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--primary-result", type=Path, required=True)
    activate.add_argument("--shared-cache", type=Path, required=True)
    activate.add_argument("--output", type=Path, required=True)

    receipt = subparsers.add_parser("replacement-receipt")
    receipt.add_argument("--activation-certificate", type=Path, required=True)
    receipt.add_argument("--shared-cache", type=Path, required=True)
    receipt.add_argument("--output", type=Path, required=True)

    migrate = subparsers.add_parser("migrate-legacy")
    migrate.add_argument("--primary-result", type=Path, required=True)
    migrate.add_argument("--legacy-activation", type=Path, required=True)
    migrate.add_argument("--operational-replacement-receipt", type=Path, required=True)
    migrate.add_argument("--shared-cache", type=Path, required=True)
    migrate.add_argument("--activation-output", type=Path, required=True)
    migrate.add_argument("--runtime-receipt-output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    policy = FallbackModelPolicy.load(options.policy)
    if options.operation == "activate":
        result = fallback_activation_certificate(
            policy=policy,
            primary_result=_object(options.primary_result),
            cached_model_repositories=discover_cached_model_repositories(
                options.shared_cache
            ),
        )
    elif options.operation == "replacement-receipt":
        result = fallback_cache_replacement_receipt(
            policy=policy,
            activation_certificate=_object(options.activation_certificate),
            shared_cache=options.shared_cache,
        )
        atomic_write_public_json(options.output, result)
        return 0
    else:
        activation, receipt = migrate_legacy_fallback_artifacts(
            policy=policy,
            primary_result=_object(options.primary_result),
            legacy_activation=_object(options.legacy_activation),
            operational_replacement_receipt=_object(
                options.operational_replacement_receipt
            ),
            shared_cache=options.shared_cache,
        )
        atomic_write_public_json(options.activation_output, activation)
        atomic_write_public_json(options.runtime_receipt_output, receipt)
        return 0
    atomic_write_public_json(options.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
