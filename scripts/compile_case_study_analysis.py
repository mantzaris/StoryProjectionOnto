#!/usr/bin/env python3
"""Compile the reviewed Phase-6 narrative table without exposing protected inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from story_projection_onto.case_study_analysis import (
    compile_case_study_narrative_analysis,
)


def _aware_datetime(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return timestamp


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--restricted-index-manifest", type=Path, required=True)
    parser.add_argument("--terminal-resume", type=Path, required=True)
    parser.add_argument("--review-template", type=Path, required=True)
    parser.add_argument("--completed-review", type=Path, required=True)
    parser.add_argument("--public-alias-manifest", type=Path, required=True)
    parser.add_argument("--restricted-table", type=Path, required=True)
    parser.add_argument("--protected-canary-manifest", type=Path)
    parser.add_argument("--protected-corpus", type=Path)
    parser.add_argument("--public-table", type=Path)
    parser.add_argument("--restricted-receipt", type=Path, required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--compiled-at", type=_aware_datetime, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    receipt = compile_case_study_narrative_analysis(
        restricted_root=options.restricted_root,
        public_root=options.public_root,
        plan_path=options.plan,
        restricted_index_manifest_path=options.restricted_index_manifest,
        terminal_resume_path=options.terminal_resume,
        review_template_path=options.review_template,
        completed_review_path=options.completed_review,
        public_alias_manifest_path=options.public_alias_manifest,
        restricted_table_path=options.restricted_table,
        protected_canary_manifest_path=options.protected_canary_manifest,
        protected_corpus_path=options.protected_corpus,
        public_table_path=options.public_table,
        restricted_receipt_path=options.restricted_receipt,
        analysis_id=options.analysis_id,
        compiled_at=options.compiled_at,
    )
    print(
        json.dumps(
            {
                "analysis_hash": receipt.content_hash,
                "public_row_count": receipt.public_row_count,
                "public_table_file_sha256": receipt.public_table_file_sha256,
                "state": receipt.publication_status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
