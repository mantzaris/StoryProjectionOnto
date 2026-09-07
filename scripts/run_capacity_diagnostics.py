#!/usr/bin/env python3
"""One existing, user-authorized feasibility block. Not an acceptance certificate.

The parent is a preloaded CPU guardian; only its child can start the metered
service. Both bind the same source/configuration. Reservations never disappear.
All new output is restricted. This driver cannot launch ordinary study calls.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    ConstructionSeal,
    FixedOntologyInput,
    OntologyDraft,
    UpperOntology,
    canonical_sha256,
)
from story_projection_onto.experiment import AllocatedGPUMeter, ResourceLimits
from story_projection_onto.gpu_runtime import (
    ResourceSampler,
    ResourceWatchdog,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    capture_gpu_hardware_identity,
    capture_tokenizer_manifest,
)
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.output_capacity_gate import (
    BASELINE_SECONDS,
    BLOCK_SECONDS,
    SHUTDOWN_SECONDS,
    CapacityRecoveryState,
    capacity_forecast,
)
from story_projection_onto.output_wire import pack_capacity_candidate
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    FailureKind,
    Ledger,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
    StoragePreflight,
)

BLOCK_ID = "output-capacity-recovery-v1"


def now():
    return datetime.now(UTC).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def immutable(path, value):
    if path.exists():
        raise FileExistsError(path)
    write_json_atomic(value, path)


def source_binding(root):
    paths = sorted(
        set(root.glob("src/story_projection_onto/**/*.py"))
        | set(root.glob("configs/study/*.json"))
        | set(root.glob("prompts/**/*.md"))
        | {root / "scripts/run_capacity_diagnostics.py"}
    )
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def stage_deadline(*, started, now_monotonic, prior_block_seconds, stage_seconds):
    """Leave a full shutdown reserve before the whole block ends, including gaps."""
    whole = started + BLOCK_SECONDS - prior_block_seconds - 1
    limit = min(now_monotonic + stage_seconds, whole - SHUTDOWN_SECONDS)
    if limit <= now_monotonic:
        raise TimeoutError("block has no allocation left before shutdown reserve")
    return limit, whole


def setup(root, run):
    limits = ResourceLimits.load(root / "configs/study/resource_limits.json")
    ledger = Ledger(root / "artifacts/restricted/phase1_acceptance.sqlite")
    # Both processes call this BEFORE the service starts. Never recover live
    # allocation journals by constructing a new meter during inference.
    meter = AllocatedGPUMeter.from_limits(ledger, limits)
    snapshot_manifest = read(root / "artifacts/public/manifests/model_snapshot_fallback.json")
    config = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=root
        / ".cache/shared/hub/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79",
        shared_cache=root / ".cache/shared",
        model_configuration_path=root / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=snapshot_manifest["manifest_sha256"],
    )
    sampler = ResourceSampler(limits=limits, storage=StoragePreflight(root), ledger=ledger)
    service = VLLMService(
        configuration=config,
        meter=meter,
        client=VLLMGuidedJSONClient(config.base_url, diagnostic_root=run / "http"),
        log_path=run / "vllm.log",
        startup_resource_sampler=sampler,
    )
    return ledger, sampler, service


def count_reservations(block, kind):
    return len(list(block.glob(f"{kind}-*.json")))


def actual_fixed(root, call, base_request, c1, tokenizer):
    from story_projection_onto.conditions.base import preontology_semantic_hash, sealed_semantic_ids
    from story_projection_onto.phase1_acceptance import _condition_output_schema, _request_sections

    raw = read(root / call.request_fixture)
    upper = UpperOntology.model_validate(raw["upper_ontology"])
    seal = ConstructionSeal(
        seal_id="capacity-actual-c1-seal",
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=raw["snapshot_hash"],
        ontology_hash=preontology_semantic_hash(upper, c1),
        constructed_at=datetime.now(UTC),
        sealed_at=datetime.now(UTC),
        sealed_object_ids=sealed_semantic_ids(c1),
    )
    fixed = FixedOntologyInput(
        construction_seal=seal,
        upper_ontology=upper,
        local_schema=c1.local_schema,
        instance_graph=c1.instance_graph,
    )
    raw.pop("content_hash", None)
    raw.update(fixed_ontology=fixed.model_dump(mode="json"), requested_at=now())
    fixture = ConstructionRequest.model_validate(raw)
    sections = _request_sections(call.acceptance_call(), fixture.model_dump(mode="json"))
    original_sections = json.loads(base_request.messages[1].content)
    for name in ("legacy_evidence_provenance", "fallback_capability_probe"):
        if name in original_sections:
            sections[name] = original_sections[name]
    schema = _condition_output_schema(
        read(root / "schemas/jsonschema/ontology_draft.schema.json"),
        call=call.acceptance_call(),
        fixture=fixture.model_dump(mode="json"),
    )
    return pack_capacity_candidate(
        base_request, tokenizer, sections_override=sections, schema_override=schema
    ), fixture


def controller(root, block, run, *, prepare_only=False):
    cpu_started = time.monotonic()
    from transformers import AutoTokenizer

    from story_projection_onto.fallback_acceptance import (
        _require_call_operator_coverage,
        _verify_second_recovery_decoder_compiles,
        build_fallback_acceptance_request,
        fallback_pilot_calls,
    )
    from story_projection_onto.model_gate import FallbackModelPolicy
    from story_projection_onto.phase1_acceptance import validate_acceptance_generation
    from story_projection_onto.phase1_legacy_provenance import Phase1LegacyEvidenceProvenanceBridge

    ledger, sampler, service = setup(root, run)
    binding = read(run / "binding.json")
    if (
        binding["source"] != source_binding(root)
        or binding["configuration"] != service.configuration.configuration_hash
    ):
        raise ValueError("deployed source/configuration differs between controller and guardian")
    policy = FallbackModelPolicy.load(root / "configs/study/fallback_model.json")
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(root)
    tokenizer = AutoTokenizer.from_pretrained(
        service.configuration.snapshot_path, local_files_only=True
    )
    tokenizer_manifest = capture_tokenizer_manifest(
        service.configuration.snapshot_path, repository=policy.repository, revision=policy.revision
    )
    calls = fallback_pilot_calls(policy)
    first = pack_capacity_candidate(
        build_fallback_acceptance_request(
            root=root,
            call=calls[0],
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            legacy_provenance_bridge=bridge,
        ),
        tokenizer,
    )
    _verify_second_recovery_decoder_compiles(first.output_schema)
    inventory = read(root / "artifacts/restricted/v10_validation/terminal-verification.json")[
        "remaining_inventory_rows"
    ]
    forecast = capacity_forecast(inventory)
    immutable(
        run / "cpu-preparation.json",
        {
            "seconds": time.monotonic() - cpu_started,
            "first_request_hash": first.request_hash,
            "template_inclusive_tokens": first.rendered_input_token_count,
            "forecast": forecast,
            "source_binding": canonical_sha256(binding),
            "completed_at": now(),
        },
    )
    if prepare_only:
        print(json.dumps({"cpu_preparation_passed": True, "gpu_allocated": False}), flush=True)
        return
    # Live hardware/storage checks stay here, after invariant preparation.
    hardware = capture_gpu_hardware_identity(root / "artifacts/public/manifests/environment.json")
    immutable(run / "hardware.json", asdict(hardware))
    sampler.sample(sample_id=run.name + "-preallocation", root_pid=os.getpid())
    starts = count_reservations(block, "start")
    attempts = count_reservations(block, "attempt")
    actual = service.meter.actual_allocated_gpu_seconds
    admission = CapacityRecoveryState(actual, starts, attempts).admit(
        remaining_mandatory_seconds=forecast["remaining_forecast_seconds"],
        stage_seconds=300,
        starting_service=True,
        complete_packing=True,
        feasibility_diagnostic_exception=True,
    )
    started = time.monotonic()
    prior = actual - BASELINE_SECONDS
    session = f"capacity-start-{starts + 1}"
    event = session + "-load"
    state = {
        "pid": os.getpid(),
        "session": session,
        "event": event,
        "binding_hash": canonical_sha256(binding),
        "started_monotonic": started,
        "prior_block_seconds": prior,
    }

    def stage(name, seconds):
        deadline, whole = stage_deadline(
            started=started,
            now_monotonic=time.monotonic(),
            prior_block_seconds=prior,
            stage_seconds=seconds,
        )
        state.update(
            stage=name,
            deadline_monotonic=deadline,
            whole_deadline_monotonic=whole,
            recorded_at=now(),
        )
        write_json_atomic(state, run / "state.json")
        return deadline - time.monotonic()

    immutable(block / f"start-{starts + 1:02d}.json", state | {"admission": admission})
    outcomes = []
    artifact_store = ArtifactStore(BlobStore(root / "artifacts/blobs/phase1_acceptance"), ledger)
    accepted_c1 = None
    try:
        cap = stage("startup", 300)
        service.start(
            session_id=session,
            event_id=event,
            watchdog_seconds=cap,
            remaining_required_seconds=SHUTDOWN_SECONDS,
        )
        stage("live_checks", 120)
        service.start_periodic_resource_watchdog(
            ResourceWatchdog(
                sampler=sampler,
                root_pid=service.pid,
                sample_prefix=run.name + "-periodic",
                interval_seconds=2,
                allocation_guard=service.require_hard_stop_margin,
                on_failure=lambda _: service.request_emergency_stop(),
            )
        )
        sampler.sample(sample_id=run.name + "-ready", root_pid=service.pid)
        for index in (0, 1, 3):
            call = calls[index]
            fixed_fixture = None
            if index == 0:
                request = first
            else:
                stage("request_preparation", 120)
                if index == 3:
                    call = replace(call, seed_block=calls[0].seed_block)
                base = build_fallback_acceptance_request(
                    root=root,
                    call=call,
                    tokenizer=tokenizer,
                    tokenizer_manifest=tokenizer_manifest,
                    legacy_provenance_bridge=bridge,
                )
                if index == 3:
                    if accepted_c1 is None:
                        raise ValueError("FixedSelect requires actual accepted C1")
                    request, fixed_fixture = actual_fixed(root, call, base, accepted_c1, tokenizer)
                else:
                    request = pack_capacity_candidate(base, tokenizer)
                _verify_second_recovery_decoder_compiles(request.output_schema)
            stage("pre_generation_checks", 120)
            attempts = count_reservations(block, "attempt")
            receipt = CapacityRecoveryState(
                service.actual_allocated_service_seconds,
                count_reservations(block, "start"),
                attempts,
            ).admit(
                remaining_mandatory_seconds=forecast["remaining_forecast_seconds"],
                stage_seconds=call.watchdog_seconds,
                diagnostic_generation=True,
                complete_packing=True,
                feasibility_diagnostic_exception=True,
            )
            attempt_id = f"capacity-diagnostic-{attempts + 1}"
            attempt_root = run / attempt_id
            attempt_root.mkdir(mode=0o700)
            request_value = request.wire_payload()
            immutable(attempt_root / "request.json", request_value)
            immutable(attempt_root / "packing.json", request.packing.model_dump(mode="json"))
            if fixed_fixture is not None:
                immutable(
                    attempt_root / "actual-fixed-request.json",
                    fixed_fixture.model_dump(mode="json"),
                )
            immutable(
                block / f"attempt-{attempts + 1:02d}.json",
                {
                    "attempt_id": attempt_id,
                    "request_hash": request.request_hash,
                    "condition": call.condition.value,
                    "seed": call.seed_block,
                    "run": run.name,
                    "admission": receipt,
                    "reserved_at": now(),
                },
            )
            job = ledger.create_or_resume_job(
                {"block": BLOCK_ID, "attempt": attempt_id, "request_hash": request.request_hash},
                release_class=ReleaseClass.RESTRICTED,
            )
            ledger.record_attempt(
                attempt_id=attempt_id,
                job_id=job.job_id,
                attempt_kind=AttemptKind.BASE,
                input_hash=request.request_hash,
                config_hash=request.decoding.content_hash,
                seed=call.seed_block,
            )
            result = None
            failure = None
            validation = None
            cap = stage("generation", call.watchdog_seconds)
            tic = time.monotonic()
            failure_stage = "client"
            try:
                result = service.generate(
                    request,
                    event_id=attempt_id,
                    watchdog_seconds=cap,
                    job_id=job.job_id,
                    attempt_id=attempt_id,
                    remaining_required_seconds=SHUTDOWN_SECONDS,
                    accounting_details={
                        "block_id": BLOCK_ID,
                        "feasibility_diagnostic_only": True,
                        "complete_forecast_exception": True,
                    },
                )
                generation_seconds = time.monotonic() - tic
                stage("validation", 120)
                immutable(attempt_root / "decoded.json", result.parsed_object)
                failure_stage = "schema_or_structural_validation"
                validation = validate_acceptance_generation(
                    root=root,
                    call=call.acceptance_call(),
                    parsed_object=result.parsed_object,
                    authoritative_prompt_tokens=result.prompt_tokens,
                    authoritative_completion_tokens=result.completion_tokens,
                    legacy_provenance_bridge=bridge,
                    diagnostic_journal=result.diagnostic_journal,
                    actual_fixed_request=fixed_fixture,
                    actual_c1_draft=accepted_c1 if index == 3 else None,
                )
                failure_stage = "scientific_capability_validation"
                validation["operator_behavior"] = _require_call_operator_coverage(
                    call, validation, result.parsed_object, root=root
                )
                if result.finish_reason == "length":
                    raise ValueError(
                        "length-terminated output is not successful capacity acceptance"
                    )
                immutable(attempt_root / "validation.json", validation)
                if index == 0:
                    accepted_c1 = OntologyDraft.model_validate(result.parsed_object)
                    immutable(
                        attempt_root / "accepted-c1.json", accepted_c1.model_dump(mode="json")
                    )
            except Exception as exc:
                generation_seconds = time.monotonic() - tic
                failure = {
                    "stage": failure_stage,
                    "exception_type": type(exc).__name__,
                    "exception_chain": traceback.format_exc(),
                    "message": str(exc),
                }
                immutable(attempt_root / "failure.json", failure)
            # Every raw response is already fsynced by the HTTP journal before
            # parsing, including failures with no GenerationResult.
            response_artifact = None
            if result is not None:
                response_artifact = artifact_store.put_bytes(
                    result.raw_response,
                    media_type="application/json",
                    release_class=ReleaseClass.RESTRICTED,
                )
            events = ledger.gpu_events_with_prefix(attempt_id)
            seconds = sum(float(e.allocated_seconds) for e in events)
            record = {
                "attempt_id": attempt_id,
                "condition": call.condition.value,
                "request_hash": request.request_hash,
                "configuration_hash": service.configuration.configuration_hash,
                "template_inclusive_input_tokens": request.rendered_input_token_count,
                "output_allowance": request.decoding.maximum_output_tokens,
                "response": None if result is None else result.public_manifest(),
                "generation_event_seconds": seconds,
                "generation_and_validation_seconds": time.monotonic() - tic,
                "generation_wall_seconds": generation_seconds,
                "accepted": failure is None,
                "failure": failure,
                "validation": validation,
                "completed_at": now(),
            }
            immutable(attempt_root / "outcome.json", record)
            ledger.record_model_call(
                model_call_id=attempt_id,
                job_id=job.job_id,
                attempt_id=attempt_id,
                gpu_event_id=attempt_id,
                backend=ModelBackend.VLLM_GPU,
                call_role=ModelCallRole.PILOT,
                retry_class=RetryClass.BASE,
                model_manifest_hash=service.configuration.configuration_hash,
                decoding_manifest_hash=request.decoding.content_hash,
                request_hash=request.request_hash,
                response_artifact_hash=None
                if response_artifact is None
                else response_artifact.content_hash,
                construction_unit_hash=canonical_sha256({"fixture": call.request_fixture}),
                served_context_count=1,
                prompt_tokens=0 if result is None else result.prompt_tokens,
                completion_tokens=0 if result is None else result.completion_tokens,
                allocated_gpu_seconds=seconds,
                successful=failure is None,
            )
            if failure:
                ledger.record_failure(
                    attempt_id=attempt_id,
                    failure_kind=FailureKind.INVALID_OUTPUT,
                    message=failure["message"],
                    details=failure,
                )
            outcomes.append(record)
            print(
                json.dumps(
                    {
                        "diagnostic": attempt_id,
                        "accepted": failure is None,
                        "condition": call.condition.value,
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )
            if failure:
                # Evidence-based repairs can resume this SAME block with its
                # persistent counters. Never retry semantics by raising tokens.
                break
    finally:
        state.update(
            stage="shutdown",
            deadline_monotonic=min(
                time.monotonic() + SHUTDOWN_SECONDS, started + BLOCK_SECONDS - prior - 1
            ),
            recorded_at=now(),
        )
        write_json_atomic(state, run / "state.json")
        service.shutdown(shutdown_seconds=max(1, state["deadline_monotonic"] - time.monotonic()))
        immutable(
            run / "terminal.json",
            {
                "outcomes": outcomes,
                "ended_at": now(),
                "actual_allocated_seconds": service.meter.actual_allocated_gpu_seconds,
                "additional_block_seconds": service.meter.actual_allocated_gpu_seconds
                - BASELINE_SECONDS,
                "complete_acceptance": False,
                "remaining_forecast": forecast,
                "resource_samples": [asdict(s) for s in sampler.samples],
                "open_allocations": len(ledger.unresolved_gpu_allocations()),
                "open_service_journals": len(ledger.unresolved_gpu_service_journals()),
            },
        )


def guardian(root, *, prepare_only=False):
    block = root / "artifacts/restricted" / BLOCK_ID
    block.mkdir(exist_ok=True, mode=0o700)
    with (block / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = block / ("run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
        run.mkdir(mode=0o700)
        ledger, _, service = setup(root, run)
        if ledger.unresolved_gpu_allocations() or ledger.unresolved_gpu_service_journals():
            raise ValueError("existing allocation requires reconciliation; do not duplicate")
        if count_reservations(block, "start") >= 2 or count_reservations(block, "attempt") >= 3:
            raise ValueError("existing block counters exhausted")
        authorization = read(root / "configs/study/output_capacity_recovery.json")
        if authorization["activation_scope"] != "bounded_feasibility_diagnostics_only":
            raise ValueError("missing narrow diagnostic authorization")
        expected = {
            "amendment_id": BLOCK_ID,
            "maximum_service_starts": 2,
            "maximum_diagnostic_generation_attempts": 3,
            "maximum_additional_allocated_seconds": 1200,
            "historical_actual_allocated_seconds": BASELINE_SECONDS,
            "scheduled_seconds": 33660,
            "strict_hard_seconds": 36000,
            "shutdown_reserve_seconds": SHUTDOWN_SECONDS,
        }
        if any(authorization.get(key) != value for key, value in expected.items()):
            raise ValueError("authorization differs from tested block constants")
        binding = {
            "source": source_binding(root),
            "configuration": service.configuration.configuration_hash,
            "authorization": authorization,
            "guardian_pid": os.getpid(),
            "created_at": now(),
        }
        immutable(run / "binding.json", binding)
        command = [sys.executable, __file__, "--controller", str(run)]
        if prepare_only:
            command.append("--prepare-only")
        child = subprocess.Popen(command, cwd=root)
        cpu_deadline = time.monotonic() + 300
        reason = None
        state = None
        while child.poll() is None:
            if (run / "terminal.json").exists():
                child.wait(timeout=10)
                break
            if (run / "state.json").exists():
                state = read(run / "state.json")
                if state["pid"] != child.pid or state["binding_hash"] != canonical_sha256(binding):
                    reason = "controller identity mismatch"
                elif time.monotonic() >= state["deadline_monotonic"]:
                    reason = "stage_or_whole_block_deadline: " + state["stage"]
            elif time.monotonic() > cpu_deadline:
                reason = "CPU preparation timeout before allocation"
            if reason:
                # Popen owns this exact unreaped child; PID reuse is impossible.
                child.kill()
                child.wait(timeout=5)
                break
            time.sleep(0.25)
        if not (run / "terminal.json").exists():
            if state is None and (run / "state.json").exists():
                state = read(run / "state.json")
            if state is not None:
                adopted = service.resume_live_service_lease(
                    expected_session_id=state["session"],
                    expected_event_id=state["event"],
                    cleanup_only=True,
                    watchdog_seconds=60,
                )
                if adopted:
                    service.shutdown(shutdown_seconds=50)
                else:
                    service.recover_stale_service_lease()
            immutable(
                run / "guardian-terminal.json",
                {
                    "reason": reason or "controller_exited",
                    "exit_code": child.returncode,
                    "ended_at": now(),
                    "actual_allocated_seconds": service.meter.actual_allocated_gpu_seconds,
                    "open_allocations": len(ledger.unresolved_gpu_allocations()),
                    "open_service_journals": len(ledger.unresolved_gpu_service_journals()),
                },
            )
        print(
            json.dumps({"run": str(run.relative_to(root)), "exit_code": child.returncode}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.controller:
        controller(root, args.controller.parent, args.controller, prepare_only=args.prepare_only)
    elif args.prepare_only:
        guardian(root, prepare_only=True)
    elif args.execute:
        guardian(root)
    else:
        parser.error("--execute is required; no service is started by importing")
