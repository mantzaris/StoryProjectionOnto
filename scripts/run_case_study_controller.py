#!/usr/bin/env python3
"""Validate the production Phase-6 controller and concrete GPU-adapter boundary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from story_projection_onto.case_study_execution import (
    CASE_BASE_WATCHDOG_SECONDS,
    CASE_REQUIRED_NEXT_REPAIR_SECONDS,
    HARD_GPU_LIMIT_SECONDS,
    SCHEDULED_GPU_LIMIT_SECONDS,
)
from story_projection_onto.case_study_gpu import (
    CASE_GPU_ADAPTER_REVISION,
    build_production_case_study_gpu_adapter,
)
from story_projection_onto.case_study_runtime import (
    audit_case_study_resume,
    load_case_study_execution_plan,
    load_case_study_resume_manifest,
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="fail closed here; lifecycle inputs must use the production factory API",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    plan = load_case_study_execution_plan(
        options.plan,
        restricted_root=options.restricted_root,
    )
    if options.execute:
        raise RuntimeError(
            "Phase-6 CLI execution requires lawful corpus/admission artifacts and an "
            "owned selected-model service; invoke the registered "
            "story_projection_onto.case_study_gpu:"
            "build_production_case_study_gpu_adapter factory from the lifecycle owner"
        )
    if not callable(build_production_case_study_gpu_adapter):  # pragma: no cover
        raise RuntimeError("registered case GPU adapter factory is not callable")
    result: dict[str, object] = {
        "adapter_factory_available": True,
        "adapter_factory_revision": CASE_GPU_ADAPTER_REVISION,
        "adapter_factory_symbol": (
            "story_projection_onto.case_study_gpu:build_production_case_study_gpu_adapter"
        ),
        "case_base_watchdog_seconds": CASE_BASE_WATCHDOG_SECONDS,
        "case_gpu_call_count": len(plan.gpu_call_slots),
        "controller_contract_valid": True,
        "execute_enabled": False,
        "standalone_cli_gpu_execution_enabled": False,
        "execution_plan_hash": plan.content_hash,
        "hard_gpu_limit_seconds": HARD_GPU_LIMIT_SECONDS,
        "required_next_repair_seconds": CASE_REQUIRED_NEXT_REPAIR_SECONDS,
        "scheduled_gpu_limit_seconds": SCHEDULED_GPU_LIMIT_SECONDS,
    }
    if options.resume is not None:
        resume = load_case_study_resume_manifest(
            options.resume,
            restricted_root=options.restricted_root,
        )
        status = audit_case_study_resume(plan, resume)
        result.update(
            {
                "resume_complete": status.complete,
                "resume_hash": status.resume_manifest_hash,
                "resume_next_stage": status.next_stage,
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
