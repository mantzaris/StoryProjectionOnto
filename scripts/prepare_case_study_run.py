#!/usr/bin/env python3
"""Compile and audit the restricted bounded first-novel execution substrate.

This command never searches for a corpus, opens a model service, or prints a
restricted path, query, packet, passage, model output, or reviewer judgment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from story_projection_onto.case_study_runtime import (
    CaseStudyRuntimePolicy,
    audit_case_study_resume,
    case_study_runtime_contract_hashes,
    compile_case_review_input_template,
    compile_case_study_execution_plan,
    initialize_case_study_resume,
    load_attested_restricted_case_study,
    load_attested_selected_model_freeze,
    load_case_study_admission_attestation,
    load_case_study_execution_plan,
    load_case_study_resume_manifest,
    write_case_resume_manifest,
    write_restricted_case_record,
)


def _aware_datetime(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return timestamp


def _restricted_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    schemas = subcommands.add_parser("contract-hashes")
    schemas.set_defaults(handler=_contract_hashes)

    compile_plan = subcommands.add_parser("compile-plan")
    compile_plan.add_argument("--repository-root", type=Path, required=True)
    compile_plan.add_argument("--restricted-root", type=Path, required=True)
    compile_plan.add_argument("--index", type=Path, required=True)
    compile_plan.add_argument("--index-manifest", type=Path, required=True)
    compile_plan.add_argument("--preregistration", type=Path, required=True)
    compile_plan.add_argument("--input-attestation", type=Path, required=True)
    compile_plan.add_argument("--admission-attestation", type=Path, required=True)
    compile_plan.add_argument("--selected-model-freeze", type=Path, required=True)
    compile_plan.add_argument("--policy", type=Path, required=True)
    compile_plan.add_argument("--output", type=Path, required=True)
    compile_plan.add_argument("--compiled-at", type=_aware_datetime, required=True)
    compile_plan.set_defaults(handler=_compile_plan)

    review = subcommands.add_parser("review-template")
    _restricted_input_arguments(review)
    review.add_argument("--output", type=Path, required=True)
    review.set_defaults(handler=_review_template)

    initialize = subcommands.add_parser("initialize-resume")
    _restricted_input_arguments(initialize)
    initialize.add_argument("--output-directory", type=Path, required=True)
    initialize.add_argument("--initialized-at", type=_aware_datetime, required=True)
    initialize.set_defaults(handler=_initialize_resume)

    status = subcommands.add_parser("resume-status")
    _restricted_input_arguments(status)
    status.add_argument("--resume", type=Path, required=True)
    status.set_defaults(handler=_resume_status)
    return parser.parse_args(arguments)


def _contract_hashes(_: argparse.Namespace) -> int:
    print(json.dumps(case_study_runtime_contract_hashes(), indent=2, sort_keys=True))
    return 0


def _compile_plan(options: argparse.Namespace) -> int:
    loaded = load_attested_restricted_case_study(
        restricted_root=options.restricted_root,
        index_path=options.index,
        manifest_path=options.index_manifest,
        preregistration_path=options.preregistration,
        attestation_path=options.input_attestation,
    )
    admission = load_case_study_admission_attestation(
        options.admission_attestation,
        restricted_root=options.restricted_root,
    )
    selected_model_freeze = load_attested_selected_model_freeze(
        restricted_root=options.restricted_root,
        selected_model_freeze_path=options.selected_model_freeze,
        admission=admission,
    )
    policy = CaseStudyRuntimePolicy.load(options.policy)
    plan = compile_case_study_execution_plan(
        repository_root=options.repository_root,
        policy=policy,
        loaded=loaded,
        admission=admission,
        selected_model_freeze=selected_model_freeze,
        compiled_at=options.compiled_at,
    )
    write_restricted_case_record(
        plan,
        options.output,
        restricted_root=options.restricted_root,
    )
    print(
        json.dumps(
            {
                "bounded_context_count": len(plan.bounded_context_ids),
                "c0_projection_count": 8,
                "c1_preconstruction_count": 4,
                "c2_bounded_construction_count": 8,
                "gpu_call_slot_count": len(plan.gpu_call_slots),
                "operational_full_index_count": 1,
                "operational_result_is_causal": False,
                "plan_hash": plan.content_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _review_template(options: argparse.Namespace) -> int:
    plan = load_case_study_execution_plan(
        options.plan,
        restricted_root=options.restricted_root,
    )
    template = compile_case_review_input_template(plan)
    write_restricted_case_record(
        template,
        options.output,
        restricted_root=options.restricted_root,
    )
    print(
        json.dumps(
            {
                "detailed_matching_unit_count": 4,
                "reviewer_judgment_count": 0,
                "review_unit_count": 8,
                "template_hash": template.content_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _initialize_resume(options: argparse.Namespace) -> int:
    plan = load_case_study_execution_plan(
        options.plan,
        restricted_root=options.restricted_root,
    )
    resume = initialize_case_study_resume(plan, initialized_at=options.initialized_at)
    write_case_resume_manifest(
        resume,
        options.output_directory,
        restricted_root=options.restricted_root,
    )
    print(
        json.dumps(
            {
                "next_stage": "prequery",
                "resume_hash": resume.content_hash,
                "sequence_number": resume.sequence_number,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resume_status(options: argparse.Namespace) -> int:
    plan = load_case_study_execution_plan(
        options.plan,
        restricted_root=options.restricted_root,
    )
    resume = load_case_study_resume_manifest(
        options.resume,
        restricted_root=options.restricted_root,
    )
    status = audit_case_study_resume(plan, resume)
    print(
        json.dumps(
            {
                "complete": status.complete,
                "completed_output_count": status.completed_output_count,
                "completed_prequery_job_count": status.completed_prequery_job_count,
                "failed_output_count": status.failed_output_count,
                "interrupted_output_count": status.interrupted_output_count,
                "invalid_output_count": status.invalid_output_count,
                "next_stage": status.next_stage,
                "non_succeeded_output_count": status.non_succeeded_output_count,
                "query_access_count": status.query_access_count,
                "repaired_attempt_count": status.repaired_attempt_count,
                "resume_manifest_hash": status.resume_manifest_hash,
                "timed_out_output_count": status.timed_out_output_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    return options.handler(options)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
