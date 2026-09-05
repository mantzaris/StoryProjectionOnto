#!/usr/bin/env python3
"""Materialize or replay the canonical public Phase 5 feedback table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_projection_onto.phase5_execution import DEFAULT_PHASE5_OUTPUT_ROOT
from story_projection_onto.scorer_only.phase5_feedback import DEFAULT_PHASE5_SCORING_ROOT
from story_projection_onto.scorer_only.phase5_report import (
    DEFAULT_PHASE5_REPORT_ROOT,
    Phase5FeedbackReportError,
    materialize_phase5_feedback_table,
    prepare_phase5_feedback_table,
    verify_phase5_feedback_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--feedback-journal-root",
        type=Path,
        default=DEFAULT_PHASE5_OUTPUT_ROOT,
    )
    parser.add_argument("--scoring-root", type=Path, default=DEFAULT_PHASE5_SCORING_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PHASE5_REPORT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    try:
        repository = options.repository.resolve(strict=True)

        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else repository / path

        journal_root = resolve(options.feedback_journal_root)
        scoring_root = resolve(options.scoring_root)
        output_root = resolve(options.output_root)
        if options.verify:
            receipt = verify_phase5_feedback_table(
                feedback_journal_root=journal_root,
                scoring_root=scoring_root,
                output_root=output_root,
            )
            state = "verified"
            writes_performed = False
        else:
            _rows, table_bytes, receipt = prepare_phase5_feedback_table(
                feedback_journal_root=journal_root,
                scoring_root=scoring_root,
            )
            writes_performed = materialize_phase5_feedback_table(
                output_root=output_root,
                table_bytes=table_bytes,
                receipt=receipt,
            )
            state = "materialized"
        print(
            json.dumps(
                {
                    "state": state,
                    "writes_performed": writes_performed,
                    "receipt_hash": receipt.content_hash,
                    "table_file_sha256": receipt.table_file_sha256,
                    "table_row_count": receipt.table_row_count,
                    "scripted_condition_score_count": (
                        receipt.scripted_condition_score_count
                    ),
                    "researcher_trace_count": receipt.researcher_trace_count,
                    "researcher_trace_gold_fields": (
                        receipt.researcher_trace_gold_fields
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, Phase5FeedbackReportError) as error:
        print(
            json.dumps(
                {
                    "state": "blocked",
                    "error_code": "phase5_feedback_table_blocked",
                    "error": str(error),
                    "writes_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
