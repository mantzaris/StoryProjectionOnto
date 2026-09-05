#!/usr/bin/env python3
"""Validate or execute the restricted, single-load Phase-6 controller."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from story_projection_onto.case_study_execution import (
    CASE_BASE_WATCHDOG_SECONDS,
    CASE_REQUIRED_NEXT_REPAIR_SECONDS,
    CASE_SERVICE_START_WATCHDOG_SECONDS,
    HARD_GPU_LIMIT_SECONDS,
    SCHEDULED_GPU_LIMIT_SECONDS,
)
from story_projection_onto.case_study_factory import (
    CASE_FACTORY_REVISION,
    CASE_PRODUCTION_FACTORY,
    CaseStudyFactoryError,
    CaseStudyProductionBundle,
    preflight_frozen_production_case_study_bundle,
)
from story_projection_onto.case_study_runtime import (
    CaseStudyRuntimePolicy,
    audit_case_study_resume,
    load_case_study_execution_plan,
    load_case_study_resume_manifest,
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--restricted-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run/resume the exact 4/8/1 block with one lifecycle-owned vLLM service",
    )
    parser.add_argument("--adapter-factory")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--input-attestation", type=Path)
    parser.add_argument("--admission-attestation", type=Path)
    parser.add_argument("--semantic-gate-bundle", type=Path)
    parser.add_argument("--selected-model-freeze", type=Path)
    parser.add_argument("--admission-evidence-bundle", type=Path)
    parser.add_argument("--admission-evidence-bundle-reference", type=Path)
    parser.add_argument(
        "--construction",
        type=Path,
        default=Path("configs/study/development_construction.json"),
    )
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--staging-transition-directory", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--quota-root", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--shared-cache", type=Path)
    parser.add_argument("--verified-model-manifest", type=Path)
    parser.add_argument("--source-association", type=Path)
    parser.add_argument("--expected-predecessor-ledger-sha256")
    parser.add_argument("--source-revision")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(arguments)


def _production_factory(reference: str) -> Callable[..., CaseStudyProductionBundle]:
    if reference != CASE_PRODUCTION_FACTORY:
        raise CaseStudyFactoryError("case adapter factory differs from the frozen implementation")
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise CaseStudyFactoryError("case adapter factory reference is invalid")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise CaseStudyFactoryError("case adapter factory is not callable")
    return factory


def _require_execute_options(
    options: argparse.Namespace,
    *,
    operation: str = "--execute",
) -> dict[str, Any]:
    names = (
        "index",
        "index_manifest",
        "preregistration",
        "input_attestation",
        "admission_attestation",
        "semantic_gate_bundle",
        "selected_model_freeze",
        "admission_evidence_bundle",
        "admission_evidence_bundle_reference",
        "ledger",
        "artifact_root",
        "staging_transition_directory",
        "runtime_root",
        "quota_root",
        "snapshot",
        "shared_cache",
        "verified_model_manifest",
        "source_association",
        "expected_predecessor_ledger_sha256",
        "source_revision",
    )
    missing = tuple(name for name in names if getattr(options, name) is None)
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise CaseStudyFactoryError(f"{operation} requires: {flags}")
    return {name: getattr(options, name) for name in names}


def _redacted_error(error: BaseException, options: argparse.Namespace) -> str:
    del options
    message = str(error)
    # The only detailed public error is the parser-generated inventory of absent
    # flags. Pydantic validation messages may embed input values (including novel
    # prose), so all other diagnostics stay in restricted logs/CAS artifacts.
    if not (
        isinstance(error, CaseStudyFactoryError)
        and message.startswith(("--execute requires:", "--validate-only requires:"))
    ):
        return "case command blocked; diagnostics retained in restricted storage"
    return message


def _status_payload(
    *,
    plan: Any,
    policy: CaseStudyRuntimePolicy,
    options: argparse.Namespace,
) -> dict[str, object]:
    result: dict[str, object] = {
        "adapter_factory_available": callable(
            _production_factory(policy.production_adapter_factory)
        ),
        "adapter_factory_revision": CASE_FACTORY_REVISION,
        "adapter_factory_symbol": policy.production_adapter_factory,
        "case_base_watchdog_seconds": CASE_BASE_WATCHDOG_SECONDS,
        "case_gpu_call_count": len(plan.gpu_call_slots),
        "controller_contract_valid": True,
        "execute_enabled": False,
        "execution_ready": False,
        "execution_plan_hash": plan.content_hash,
        "hard_gpu_limit_seconds": HARD_GPU_LIMIT_SECONDS,
        "required_next_repair_seconds": CASE_REQUIRED_NEXT_REPAIR_SECONDS,
        "scheduled_gpu_limit_seconds": SCHEDULED_GPU_LIMIT_SECONDS,
        "service_start_watchdog_seconds": CASE_SERVICE_START_WATCHDOG_SECONDS,
        "state": "contract_only",
        "validation_scope": "contract_only",
        "writes_performed": False,
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
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    bundle: CaseStudyProductionBundle | None = None
    exit_code = 0
    payload: dict[str, object]
    try:
        repository = options.repository.resolve(strict=True)
        plan = load_case_study_execution_plan(
            options.plan,
            restricted_root=options.restricted_root,
        )
        policy = CaseStudyRuntimePolicy.load(repository / "configs/case_study/runtime.json")
        factory_reference = options.adapter_factory or policy.production_adapter_factory
        factory = _production_factory(factory_reference)
        if options.validate_only:
            values = _require_execute_options(options, operation="--validate-only")
            construction = options.construction
            if not construction.is_absolute():
                construction = repository / construction
            preflight = preflight_frozen_production_case_study_bundle(
                repository=repository,
                restricted_root=options.restricted_root,
                plan_path=options.plan,
                index_path=values["index"],
                index_manifest_path=values["index_manifest"],
                preregistration_path=values["preregistration"],
                input_attestation_path=values["input_attestation"],
                admission_attestation_path=values["admission_attestation"],
                semantic_gate_bundle_path=values["semantic_gate_bundle"],
                selected_model_freeze_path=values["selected_model_freeze"],
                admission_evidence_bundle_path=values["admission_evidence_bundle"],
                admission_evidence_bundle_reference_path=(
                    values["admission_evidence_bundle_reference"]
                ),
                construction_path=construction,
                ledger_path=values["ledger"],
                artifact_root=values["artifact_root"],
                staging_transition_directory=values["staging_transition_directory"],
                runtime_root=values["runtime_root"],
                quota_root=values["quota_root"],
                snapshot_path=values["snapshot"],
                shared_cache=values["shared_cache"],
                verified_model_manifest_path=values["verified_model_manifest"],
                source_association_path=values["source_association"],
                expected_predecessor_ledger_sha256=values[
                    "expected_predecessor_ledger_sha256"
                ],
                source_revision=values["source_revision"],
                port=options.port,
            )
            payload = _status_payload(plan=plan, policy=policy, options=options)
            payload.update(asdict(preflight))
            payload.update(
                {
                    "execute_enabled": True,
                    "execution_ready": True,
                    "state": "ready",
                    "validation_scope": "full_production_preflight",
                    "writes_performed": False,
                }
            )
        elif not options.execute:
            payload = _status_payload(plan=plan, policy=policy, options=options)
        else:
            values = _require_execute_options(options)
            construction = options.construction
            if not construction.is_absolute():
                construction = repository / construction
            bundle = factory(
                repository=repository,
                restricted_root=options.restricted_root,
                plan_path=options.plan,
                index_path=values["index"],
                index_manifest_path=values["index_manifest"],
                preregistration_path=values["preregistration"],
                input_attestation_path=values["input_attestation"],
                admission_attestation_path=values["admission_attestation"],
                semantic_gate_bundle_path=values["semantic_gate_bundle"],
                selected_model_freeze_path=values["selected_model_freeze"],
                admission_evidence_bundle_path=values["admission_evidence_bundle"],
                admission_evidence_bundle_reference_path=(
                    values["admission_evidence_bundle_reference"]
                ),
                construction_path=construction,
                ledger_path=values["ledger"],
                artifact_root=values["artifact_root"],
                staging_transition_directory=values["staging_transition_directory"],
                runtime_root=values["runtime_root"],
                quota_root=values["quota_root"],
                snapshot_path=values["snapshot"],
                shared_cache=values["shared_cache"],
                verified_model_manifest_path=values["verified_model_manifest"],
                source_association_path=values["source_association"],
                expected_predecessor_ledger_sha256=values[
                    "expected_predecessor_ledger_sha256"
                ],
                source_revision=values["source_revision"],
                port=options.port,
            )
            result = bundle.controller.run(bundle.admission_reference)
            result_reference = bundle.persist_result(result)
            payload = {
                "allocated_gpu_seconds_after_case": result.allocated_gpu_seconds_after_case,
                "allocated_gpu_seconds_before_case": result.allocated_gpu_seconds_before_case,
                "completed_output_count": result.status.completed_output_count,
                "execution_plan_hash": result.execution_plan_hash,
                "model_load_count": result.model_load_count,
                "model_service_stopped": result.model_service_stopped,
                "non_succeeded_output_count": result.status.non_succeeded_output_count,
                "repair_attempt_count": result.status.repaired_attempt_count,
                "result_artifact_hash": result_reference.artifact_hash,
                "result_hash": result.content_hash,
                "state": "complete",
            }
    except Exception as error:
        payload = {
            "error": _redacted_error(error, options),
            "error_type": type(error).__name__,
            "state": "blocked",
        }
        exit_code = 2
    finally:
        if bundle is not None:
            try:
                bundle.close()
            except Exception as error:
                payload = {
                    "error": _redacted_error(error, options),
                    "error_type": type(error).__name__,
                    "state": "blocked_during_shutdown",
                }
                exit_code = 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
