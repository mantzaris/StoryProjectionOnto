#!/usr/bin/env python3
"""Build the public, zero-GPU fallback-v6 control-plane incident record."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.fallback_v6_control_plane_incident import (  # noqa: E402
    FALLBACK_V6_RUN_ID,
    build_fallback_v6_control_plane_incident,
    write_fallback_v6_control_plane_incident,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=FALLBACK_V6_RUN_ID)
    parser.add_argument("--source-association", type=Path, required=True)
    parser.add_argument("--authorization-overlay", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--restricted-run-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--ledger-before", type=Path, required=True)
    parser.add_argument("--ledger-after", type=Path, required=True)
    parser.add_argument(
        "--audited-at",
        required=True,
        help="Explicit timezone-aware ISO-8601 audit timestamp",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    try:
        audited_at = datetime.fromisoformat(options.audited_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit("--audited-at must be a valid ISO-8601 timestamp") from exc
    incident = build_fallback_v6_control_plane_incident(
        run_id=options.run_id,
        source_association_path=options.source_association,
        authorization_overlay_path=options.authorization_overlay,
        preflight_path=options.preflight,
        restricted_run_root=options.restricted_run_root,
        result_path=options.result,
        ledger_before_path=options.ledger_before,
        ledger_after_path=options.ledger_after,
        audited_at=audited_at,
    )
    write_fallback_v6_control_plane_incident(options.output, incident)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
