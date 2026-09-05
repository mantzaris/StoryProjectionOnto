#!/usr/bin/env python3
"""Build the Phase 7 routing recipe from explicitly named immutable evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.final_accounting import (  # noqa: E402
    FinalAccountingError,
    LedgerSourceRoute,
    NativeSourceRole,
    NativeSourceRoute,
    SelfHashField,
    SourceFileRoute,
    build_final_accounting_source_recipe,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accounting-id", required=True)
    parser.add_argument("--base-call-inventory", type=Path, required=True)
    parser.add_argument(
        "--native",
        action="append",
        nargs=2,
        metavar=("PRODUCER_ROLE", "PATH"),
        required=True,
        help="Repeat for each exact native producer artifact.",
    )
    parser.add_argument(
        "--amendment",
        action="append",
        type=Path,
        default=[],
        help="Repeat for each self-hashed authorization amendment.",
    )
    parser.add_argument(
        "--ledger",
        action="append",
        nargs=6,
        metavar=("LEDGER_ID", "LINEAGE_ID", "SEQUENCE", "PARENT_OR_DASH", "DB", "CAS"),
        required=True,
        help="Repeat for each cumulative ledger snapshot and its CAS.",
    )
    parser.add_argument(
        "--wall-time-receipt",
        action="append",
        type=Path,
        required=True,
        help="Repeat for each receipt emitted by capture_pod_wall_time.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Child directory for the content-addressed source recipe.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild in memory and require the exact existing recipe without writes.",
    )
    return parser.parse_args(arguments)


def _relative_path(root: Path, path: Path, *, label: str) -> str:
    candidate = path if path.is_absolute() else root / path
    lexical = Path(os.path.abspath(candidate))
    try:
        return lexical.relative_to(root).as_posix()
    except ValueError as exc:
        raise FinalAccountingError(f"{label} must remain inside --source-root") from exc


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    root = Path(os.path.abspath(options.source_root))
    try:
        native_values = sorted(
            (
                NativeSourceRole(role),
                _relative_path(root, Path(path), label=f"native source {role}"),
            )
            for role, path in options.native
        )
        native_routes = tuple(
            NativeSourceRoute(
                artifact_id=f"native-{ordinal:03d}-{role.value}",
                relative_path=relative_path,
                self_hash_field=(
                    SelfHashField.MANIFEST
                    if role is NativeSourceRole.PHASE1_ACCEPTANCE_RESULT
                    else SelfHashField.CONTENT
                ),
                producer_role=role,
            )
            for ordinal, (role, relative_path) in enumerate(native_values, start=1)
        )
        amendment_paths = sorted(
            _relative_path(root, path, label="authorization amendment")
            for path in options.amendment
        )
        amendment_routes = tuple(
            SourceFileRoute(
                artifact_id=f"authorization-amendment-{ordinal:03d}",
                relative_path=relative_path,
                self_hash_field=SelfHashField.MANIFEST,
            )
            for ordinal, relative_path in enumerate(amendment_paths, start=1)
        )
        ledger_values: list[tuple[str, str, int, str | None, str, str]] = []
        for ledger_id, lineage_id, raw_sequence, parent, ledger_path, cas_path in options.ledger:
            try:
                sequence = int(raw_sequence)
            except ValueError as exc:
                raise FinalAccountingError("ledger sequence must be an integer") from exc
            ledger_values.append(
                (
                    ledger_id,
                    lineage_id,
                    sequence,
                    None if parent == "-" else parent,
                    _relative_path(root, Path(ledger_path), label=f"ledger {ledger_id}"),
                    _relative_path(root, Path(cas_path), label=f"CAS {ledger_id}"),
                )
            )
        ledger_routes = tuple(
            LedgerSourceRoute(
                ledger_id=ledger_id,
                lineage_id=lineage_id,
                sequence=sequence,
                parent_ledger_id=parent,
                ledger_relative_path=ledger_path,
                cas_relative_path=cas_path,
            )
            for ledger_id, lineage_id, sequence, parent, ledger_path, cas_path in sorted(
                ledger_values, key=lambda item: (item[1], item[2], item[0])
            )
        )
        wall_paths = sorted(
            _relative_path(root, path, label="wall-time receipt")
            for path in options.wall_time_receipt
        )
        wall_routes = tuple(
            SourceFileRoute(
                artifact_id=f"wall-time-{ordinal:03d}",
                relative_path=relative_path,
                self_hash_field=SelfHashField.MANIFEST,
            )
            for ordinal, relative_path in enumerate(wall_paths, start=1)
        )
        output_root = options.output_root
        if not output_root.is_absolute():
            output_root = root / output_root
        outputs = build_final_accounting_source_recipe(
            accounting_id=options.accounting_id,
            source_root=root,
            output_root=output_root,
            base_call_inventory=SourceFileRoute(
                artifact_id="base-call-inventory",
                relative_path=_relative_path(
                    root,
                    options.base_call_inventory,
                    label="base call inventory",
                ),
            ),
            native_source_artifacts=native_routes,
            authorization_amendments=amendment_routes,
            ledger_sources=ledger_routes,
            wall_time_receipts=wall_routes,
            verify_only=options.verify,
        )
    except (FinalAccountingError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "verified" if options.verify else "built",
                "recipe": outputs.recipe_path.name,
                "recipe_manifest_sha256": outputs.recipe.manifest_sha256,
                "freeze_time": outputs.recipe.compiled_at_utc.isoformat().replace("+00:00", "Z"),
                "native_source_count": len(outputs.recipe.native_source_artifacts),
                "ledger_source_count": len(outputs.recipe.ledger_sources),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
