#!/usr/bin/env python3
"""Prepare or finalize restricted condition-blind post-run review artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from story_projection_onto.scorer_only.blinded_postrun_review import (
    BlindedReviewError,
    materialize_community_review_finalization,
    materialize_community_review_package,
    materialize_community_review_source,
    materialize_error_review_finalization,
    materialize_error_review_package,
    materialize_held_out_failure_source,
    prepare_community_review_finalization,
    prepare_community_review_package,
    prepare_community_review_source,
    prepare_error_review_finalization,
    prepare_error_review_package,
    prepare_held_out_failure_source,
)


def _restricted_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _repository_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be valid ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include an explicit UTC offset")
    return parsed


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)

    error_source = subcommands.add_parser("error-source")
    _shared(error_source)
    error_source.add_argument("--repository", type=Path, default=Path("."))
    error_source.add_argument(
        "--phase4-configuration",
        type=Path,
        default=Path("configs/study/phase4_analysis.json"),
    )
    error_source.add_argument("--phase4-output-root", type=Path, required=True)
    error_source.add_argument("--held-out-root", type=Path, required=True)
    error_source.add_argument("--selected-at", type=_aware_datetime, required=True)
    error_source.set_defaults(handler=_error_source)

    error_prepare = subcommands.add_parser("error-prepare")
    _shared(error_prepare)
    error_prepare.add_argument("--source-manifest", type=Path, required=True)
    error_prepare.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/study/held_out_error_review_taxonomy.json"),
    )
    error_prepare.set_defaults(handler=_error_prepare)

    error_complete = subcommands.add_parser("error-complete")
    _shared(error_complete)
    error_complete.add_argument("--package-root", type=Path, required=True)
    error_complete.add_argument("--completion", type=Path, required=True)
    error_complete.add_argument("--adjudication", type=Path, required=True)
    error_complete.set_defaults(handler=_error_complete)

    community_source = subcommands.add_parser("community-source")
    _shared(community_source)
    community_source.add_argument("--repository", type=Path, default=Path("."))
    community_source.add_argument(
        "--phase4-configuration",
        type=Path,
        default=Path("configs/study/phase4_analysis.json"),
    )
    community_source.add_argument("--phase4-output-root", type=Path, required=True)
    community_source.add_argument("--selected-at", type=_aware_datetime, required=True)
    community_source.set_defaults(handler=_community_source)

    community_prepare = subcommands.add_parser("community-prepare")
    _shared(community_prepare)
    community_prepare.add_argument("--source-manifest", type=Path, required=True)
    community_prepare.add_argument(
        "--rubric-template",
        type=Path,
        default=Path("configs/study/community_review_template.json"),
    )
    community_prepare.set_defaults(handler=_community_prepare)

    community_complete = subcommands.add_parser("community-complete")
    _shared(community_complete)
    community_complete.add_argument("--package-root", type=Path, required=True)
    community_complete.add_argument("--completion", type=Path, required=True)
    community_complete.set_defaults(handler=_community_complete)
    return command


def _error_source(options: argparse.Namespace) -> dict[str, Any]:
    repository = options.repository.resolve(strict=True)
    inputs = {
        "repository": repository,
        "phase4_configuration_path": _repository_path(
            repository, options.phase4_configuration
        ),
        "phase4_output_root": _repository_path(repository, options.phase4_output_root),
        "held_out_root": _restricted_path(options.restricted_root, options.held_out_root),
        "selected_at": options.selected_at,
    }
    if options.validate_only:
        source, files = prepare_held_out_failure_source(**inputs)
        return {
            "state": "validated",
            "source_manifest_hash": source.content_hash,
            "eligible_failure_count": source.eligible_failure_count,
            "primary_receipt_count": source.held_out_output_count,
            "failure_signal_policy_hash": source.failure_signal_policy_hash,
            "planned_file_count": len(files),
            "condition_blind": True,
            "writes_performed": False,
        }
    output, state, source = materialize_held_out_failure_source(
        **inputs,
        restricted_root=options.restricted_root,
        output_root=_restricted_path(options.restricted_root, options.output_root),
    )
    return {
        "state": state,
        "source_manifest_hash": source.content_hash,
        "eligible_failure_count": source.eligible_failure_count,
        "primary_receipt_count": source.held_out_output_count,
        "failure_signal_policy_hash": source.failure_signal_policy_hash,
        "condition_blind": True,
        "output_leaf": output.name,
        "writes_performed": state == "created",
    }


def _error_prepare(options: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "restricted_root": options.restricted_root,
        "source_manifest_path": _restricted_path(
            options.restricted_root, options.source_manifest
        ),
        "taxonomy_path": options.taxonomy,
    }
    if options.validate_only:
        package, _, files = prepare_error_review_package(**inputs)
        return {
            "state": "validated",
            "package_hash": package.content_hash,
            "failure_count": len(package.items),
            "planned_file_count": len(files),
            "condition_blind": True,
            "writes_performed": False,
        }
    output, state, package = materialize_error_review_package(
        **inputs,
        output_root=_restricted_path(options.restricted_root, options.output_root),
    )
    return {
        "state": state,
        "package_hash": package.content_hash,
        "failure_count": len(package.items),
        "condition_blind": True,
        "output_leaf": output.name,
        "writes_performed": state == "created",
    }


def _error_complete(options: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "restricted_root": options.restricted_root,
        "package_root": _restricted_path(options.restricted_root, options.package_root),
        "completion_path": _restricted_path(options.restricted_root, options.completion),
        "adjudication_path": _restricted_path(
            options.restricted_root, options.adjudication
        ),
    }
    if options.validate_only:
        finalization, files = prepare_error_review_finalization(**inputs)
        return {
            "state": "validated",
            "finalization_hash": finalization.content_hash,
            "reviewed_failure_count": finalization.reviewed_failure_count,
            "planned_file_count": len(files),
            "independent_unit": "world",
            "writes_performed": False,
        }
    output, state, finalization = materialize_error_review_finalization(
        **inputs,
        output_root=_restricted_path(options.restricted_root, options.output_root),
    )
    return {
        "state": state,
        "finalization_hash": finalization.content_hash,
        "reviewed_failure_count": finalization.reviewed_failure_count,
        "independent_unit": "world",
        "output_leaf": output.name,
        "writes_performed": state == "created",
    }


def _community_prepare(options: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "restricted_root": options.restricted_root,
        "source_manifest_path": _restricted_path(
            options.restricted_root, options.source_manifest
        ),
        "rubric_template_path": options.rubric_template,
    }
    if options.validate_only:
        package, _, files = prepare_community_review_package(**inputs)
        return {
            "state": "validated",
            "package_hash": package.content_hash,
            "partition_count": len(package.items),
            "planned_file_count": len(files),
            "condition_blind": True,
            "writes_performed": False,
        }
    output, state, package = materialize_community_review_package(
        **inputs,
        output_root=_restricted_path(options.restricted_root, options.output_root),
    )
    return {
        "state": state,
        "package_hash": package.content_hash,
        "partition_count": len(package.items),
        "condition_blind": True,
        "output_leaf": output.name,
        "writes_performed": state == "created",
    }


def _community_source(options: argparse.Namespace) -> dict[str, Any]:
    repository = options.repository.resolve(strict=True)
    inputs = {
        "repository": repository,
        "phase4_configuration_path": _repository_path(
            repository, options.phase4_configuration
        ),
        "phase4_output_root": _repository_path(repository, options.phase4_output_root),
        "selected_at": options.selected_at,
    }
    if options.validate_only:
        source, files = prepare_community_review_source(**inputs)
        return {
            "state": "validated",
            "source_manifest_hash": source.content_hash,
            "partition_count": source.eligible_partition_count,
            "selected_cell_count": source.selected_cell_count,
            "selection_rule_hash": source.selection_rule_hash,
            "planned_file_count": len(files),
            "condition_blind": True,
            "writes_performed": False,
        }
    output, state, source = materialize_community_review_source(
        **inputs,
        restricted_root=options.restricted_root,
        output_root=_restricted_path(options.restricted_root, options.output_root),
    )
    return {
        "state": state,
        "source_manifest_hash": source.content_hash,
        "partition_count": source.eligible_partition_count,
        "selected_cell_count": source.selected_cell_count,
        "selection_rule_hash": source.selection_rule_hash,
        "condition_blind": True,
        "output_leaf": output.name,
        "writes_performed": state == "created",
    }


def _community_complete(options: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "restricted_root": options.restricted_root,
        "package_root": _restricted_path(options.restricted_root, options.package_root),
        "completion_path": _restricted_path(options.restricted_root, options.completion),
    }
    if options.validate_only:
        finalization, files = prepare_community_review_finalization(**inputs)
        return {
            "state": "validated",
            "finalization_hash": finalization.content_hash,
            "reviewed_partition_count": finalization.reviewed_partition_count,
            "planned_file_count": len(files),
            "independent_unit": "world",
            "writes_performed": False,
        }
    output, state, finalization = materialize_community_review_finalization(
        **inputs,
        output_root=_restricted_path(options.restricted_root, options.output_root),
    )
    return {
        "state": state,
        "finalization_hash": finalization.content_hash,
        "reviewed_partition_count": finalization.reviewed_partition_count,
        "independent_unit": "world",
        "output_leaf": output.name,
        "writes_performed": state == "created",
    }


def main(arguments: Sequence[str] | None = None) -> int:
    options = parser().parse_args(arguments)
    handler: Callable[[argparse.Namespace], dict[str, Any]] = options.handler
    try:
        print(json.dumps(handler(options), sort_keys=True))
        return 0
    except BlindedReviewError as error:
        print(json.dumps({"state": "blocked", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
