#!/usr/bin/env python3
"""Prepare, attest, or verify the final report PDF visual-inspection record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from story_projection_onto.report_visual_inspection import (
    VisualCriterion,
    prepare_report_rasters,
    record_visual_inspection,
    verify_visual_inspection,
)
from story_projection_onto.reporting import ReportingError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, default=Path("reports/results_manifest.json"))
    prepare.add_argument("--policy", type=Path, default=Path("configs/study/reporting.json"))
    prepare.add_argument("--pdf", type=Path, default=Path("reports/RESULTS_REPORT.pdf"))
    prepare.add_argument("--output-root", type=Path, default=Path("reports/visual_inspection"))

    attest = subparsers.add_parser("attest")
    attest.add_argument("--raster-manifest", type=Path, required=True)
    attest.add_argument(
        "--assessment",
        type=Path,
        required=True,
        help=(
            "JSON with reviewer_kind, reviewed_at_utc, inspected_pages, and all six "
            "criterion/status/note records. It must be authored after viewing the rasters."
        ),
    )
    attest.add_argument(
        "--output",
        type=Path,
        default=Path("reports/visual_inspection/report_visual_inspection.json"),
    )

    verify = subparsers.add_parser("verify")
    verify.add_argument("--raster-manifest", type=Path, required=True)
    verify.add_argument("--pdf", type=Path, default=Path("reports/RESULTS_REPORT.pdf"))
    verify.add_argument("--receipt", type=Path)
    verify.add_argument("--require-accepted", action="store_true")
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    try:
        if options.action == "prepare":
            manifest = prepare_report_rasters(
                result_manifest_path=options.manifest,
                policy_path=options.policy,
                pdf_path=options.pdf,
                output_root=options.output_root,
            )
            payload = {"state": "prepared_pending_visual_review", "raster_manifest": str(manifest)}
        elif options.action == "attest":
            assessment = json.loads(options.assessment.read_text(encoding="utf-8"))
            receipt = record_visual_inspection(
                raster_manifest_path=options.raster_manifest,
                output_path=options.output,
                reviewer_kind=assessment["reviewer_kind"],
                reviewed_at_utc=datetime.fromisoformat(
                    assessment["reviewed_at_utc"].replace("Z", "+00:00")
                ),
                inspected_pages=tuple(assessment["inspected_pages"]),
                criteria=tuple(
                    VisualCriterion.model_validate(item) for item in assessment["criteria"]
                ),
            )
            payload = {
                "state": "visual_review_recorded",
                "status": receipt.status,
                "receipt_sha256": receipt.receipt_sha256,
            }
        else:
            raster, receipt = verify_visual_inspection(
                raster_manifest_path=options.raster_manifest,
                pdf_path=options.pdf,
                receipt_path=options.receipt,
                require_accepted=options.require_accepted,
            )
            payload = {
                "state": "verified",
                "page_count": raster.page_count,
                "representative_pages": list(raster.representative_pages),
                "visual_review_status": receipt.status if receipt else "pending",
            }
    except (KeyError, OSError, ValueError, json.JSONDecodeError, ReportingError) as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
