#!/usr/bin/env python3
"""Preflight or run the isolated, CPU-only registered Phase 4 scorer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from story_projection_onto.scorer_only.phase4_analysis import (  # noqa: E402
    Phase4AnalysisError,
    preflight_phase4_analysis,
    run_phase4_analysis,
)


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path("configs/study/phase4_analysis.json"),
    )
    parser.add_argument("--held-out-root", type=Path, required=True)
    parser.add_argument("--scorer-bridge", type=Path, required=True)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-association", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--completed-at",
        help="Required with --run; timezone-aware ISO-8601 completion timestamp",
    )
    return parser.parse_args(arguments)


def _path(repository: Path, value: Path) -> Path:
    return value if value.is_absolute() else repository / value


def _timestamp(value: str | None) -> datetime:
    if value is None:
        raise Phase4AnalysisError("--run requires --completed-at")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Phase4AnalysisError("--completed-at must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Phase4AnalysisError("--completed-at must include a timezone")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    try:
        repository = options.repository.resolve(strict=True)
        values = {
            name: _path(repository, getattr(options, name))
            for name in (
                "configuration",
                "held_out_root",
                "scorer_bridge",
                "combined_root",
                "review_root",
                "benchmark_root",
                "ledger",
                "artifact_root",
                "source_association",
                "geometry_root",
                "output_root",
            )
        }
        shared = {
            "repository": repository,
            "configuration_path": values["configuration"],
            "held_out_root": values["held_out_root"],
            "scorer_bridge_path": values["scorer_bridge"],
            "combined_root": values["combined_root"],
            "review_root": values["review_root"],
            "benchmark_root": values["benchmark_root"],
            "ledger_path": values["ledger"],
            "artifact_root": values["artifact_root"],
            "source_association_path": values["source_association"],
            "geometry_root": values["geometry_root"],
        }
        if options.validate_only:
            if options.completed_at is not None:
                raise Phase4AnalysisError("--completed-at is only valid with --run")
            result = preflight_phase4_analysis(**shared)
            print(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "kind": "phase4_scorer_preflight",
                        "analysis_configuration_hash": result.analysis_configuration_hash,
                        "metric_configuration_hash": result.metric_configuration_hash,
                        "held_out_execution_hash": result.held_out_execution_hash,
                        "held_out_bridge_hash": result.held_out_bridge_hash,
                        "combined_execution_hash": result.combined_execution_hash,
                        "review_completion_hash": result.review_completion_hash,
                        "primary_cell_count": result.primary_cell_count,
                        "combined_cell_count": result.combined_cell_count,
                        "successful_projection_count": result.successful_projection_count,
                        "missing_geometry_count": len(result.missing_geometry_hashes),
                        "ready": result.ready,
                        "gpu_executed": False,
                        "analysis_outputs_written": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result.ready else 2
        result = run_phase4_analysis(
            **shared,
            output_root=values["output_root"],
            completed_at=_timestamp(options.completed_at),
        )
        print(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "kind": "phase4_registered_analysis_complete",
                    "run_id": result.run_id,
                    "analysis_index_hash": result.content_hash,
                    "table_manifest_hash": result.table_manifest_hash,
                    "registered_analysis_hash": result.registered_analysis_hash,
                    "report_gate_status_hash": result.report_gate_status_hash,
                    "primary_score_count": result.primary_score_count,
                    "combined_ordinary_score_count": result.combined_ordinary_score_count,
                    "contrast_score_count": result.contrast_score_count,
                    "cross_seed_community_count": result.cross_seed_community_count,
                    "paraphrase_pair_count": result.paraphrase_pair_count,
                    "ablation_pair_count": result.ablation_pair_count,
                    "registered_world_count": result.registered_world_count,
                    "gpu_executed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, Phase4AnalysisError) as error:
        print(f"phase4 scorer refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
