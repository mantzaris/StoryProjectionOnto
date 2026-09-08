"""One small ladder workload on the existing metered service and guardian."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict

from .contracts import canonical_sha256
from .gpu_runtime import ResourceWatchdog, capture_gpu_hardware_identity, capture_tokenizer_manifest
from .manifest import write_json_atomic
from .simple_ladder import CONFIG, cases, parse_text, policy, prepare
from .store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    FailureKind,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
)


def next_case(outcomes):
    """No adaptive examples: poor basic extraction replaces the harder ladder with one control."""
    from .scorer_only.simple_ladder import direct_gate

    if len(outcomes) < 2:
        return str(len(outcomes) + 1)
    if not direct_gate(outcomes):
        return "control" if len(outcomes) == 2 else None
    return str(len(outcomes) + 1) if len(outcomes) < 8 else None


def execute_workload(root, block, run, *, prepare_only=False):
    from transformers import AutoTokenizer

    from scripts.run_capacity_diagnostics import (
        count_reservations,
        immutable,
        now,
        setup,
        source_binding,
        stage_deadline,
    )

    from .scorer_only.simple_ladder import RULE_VERSION, SYNONYMS, evaluate, references, toy_extract

    cfg = policy()
    ledger, sampler, service = setup(root, run)
    sampler.prepare()
    binding = json.loads((run / "binding.json").read_text())
    if (
        binding["source"] != source_binding(root)
        or binding["configuration"] != service.configuration.configuration_hash
    ):
        raise ValueError("ladder source/configuration differs from guardian")
    tokenizer = AutoTokenizer.from_pretrained(
        service.configuration.snapshot_path, local_files_only=True
    )
    manifest = capture_tokenizer_manifest(
        service.configuration.snapshot_path,
        repository="Qwen/Qwen3-8B-AWQ",
        revision="4da05a8edb55c6046cce958586c33b61da07bb79",
    )
    inputs = cases()
    requests = {k: prepare(c, tokenizer, manifest) for k, c in inputs.items()}
    frozen = {}
    for k, q in requests.items():
        # Effective server request validation, without initializing CUDA/model.
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

        ChatCompletionRequest.model_validate(q.wire_payload())
        assert "guided_json" not in q.wire_payload()
        immutable(run / f"request-{k}.json", q.wire_payload())
        rendered = tokenizer.apply_chat_template(
            [asdict(m) for m in q.messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        immutable(
            run / f"rendered-{k}.json", {"text": rendered, "tokenizer_manifest": asdict(manifest)}
        )
        frozen[k] = {
            "request_hash": q.request_hash,
            "input_tokens": q.rendered_input_token_count,
            "reserved_output_tokens": q.decoding.maximum_output_tokens,
            "maximum_context": 12288,
        }
    immutable(run / "frozen-cases.json", inputs)
    immutable(run / "reference-answers-not-transmitted.json", references())
    immutable(
        run / "frozen-scoring.json",
        {
            "version": RULE_VERSION,
            "synonyms": SYNONYMS,
            "progression": "combined direct precision and recall >= 0.8; otherwise one plain control then stop",
            "matching": "case/whitespace and declared relation synonyms only; duplicate predictions penalized; all facts in denominator",
            "office_alternatives": "predeclared office-mediated, person-direct (case7), or combined; best F1 then recall then listed order",
            "qualification": "underlying relationships and complete qualified facts scored separately",
        },
    )
    immutable(
        run / "toy-baseline.json",
        {
            k: {
                "output": toy_extract(inputs[k]["evidence"]),
                "evaluation": evaluate(k, toy_extract(inputs[k]["evidence"])),
            }
            for k in ("1", "2")
        },
    )
    immutable(run / "packing.json", frozen)
    actual = service.meter.actual_allocated_gpu_seconds
    prior = actual - cfg.BASELINE
    if prepare_only:
        sampler.close_probes()
        print(
            json.dumps(
                {
                    "preparation_passed": True,
                    "gpu_started": False,
                    "actual_seconds": actual,
                    "packing": frozen,
                }
            ),
            flush=True,
        )
        return
    # Compare to an explicit pre-allocation frozen receipt, not an output-dependent request.
    freeze_path = block / "approved-request-hashes.json"
    if not freeze_path.exists() or json.loads(freeze_path.read_text()) != frozen:
        raise ValueError("CPU-prepared request hashes/counts must be frozen before allocation")
    immutable(
        run / "hardware.json",
        asdict(capture_gpu_hardware_identity(root / "artifacts/public/manifests/environment.json")),
    )
    sampler.sample(sample_id=run.name + "-preallocation", root_pid=os.getpid())
    admission = cfg.admit(
        actual,
        count_reservations(block, "start"),
        count_reservations(block, "attempt"),
        starting=True,
        seconds=cfg.ALLOWANCE - prior - 65,
    )
    started = time.monotonic()
    session = cfg.BLOCK_ID + "-start-1"
    state = {
        "pid": os.getpid(),
        "session": session,
        "event": session + "-load",
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
            comparison=True,
            semantic="simple-ladder",
        )
        state.update(
            stage=name,
            deadline_monotonic=deadline,
            whole_deadline_monotonic=whole,
            recorded_at=now(),
        )
        write_json_atomic(state, run / "state.json")
        return deadline - time.monotonic()

    stage("startup", CONFIG["startup_seconds"])
    immutable(block / "start-01.json", state | {"admission": admission})
    outcomes = []
    stop_reason = "completed"
    artifact_store = ArtifactStore(BlobStore(root / "artifacts/blobs/phase1_acceptance"), ledger)
    try:
        service.start(
            session_id=session,
            event_id=state["event"],
            watchdog_seconds=state["deadline_monotonic"] - time.monotonic(),
            remaining_required_seconds=60,
        )
        stage("live_checks", 15)
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
        while (case_id := next_case(outcomes)) is not None:
            # A full 45s request, 5s failure drain, 10s validation, 60s shutdown.
            if state["whole_deadline_monotonic"] - time.monotonic() < 120:
                stop_reason = "protected_shutdown_insufficient_next_request_time"
                break
            n = count_reservations(block, "attempt")
            cfg.admit(service.actual_allocated_service_seconds, 1, n, generating=True, seconds=60)
            q = requests[case_id]
            attempt = cfg.BLOCK_ID + f"-attempt-{n + 1}"
            immutable(
                block / f"attempt-{n + 1:02d}.json",
                {
                    "attempt_id": attempt,
                    "case_id": case_id,
                    "request_hash": q.request_hash,
                    "reserved_at": now(),
                    "run": run.name,
                },
            )
            job = ledger.create_or_resume_job(
                {
                    "diagnostic_ladder": cfg.BLOCK_ID,
                    "case_id": case_id,
                    "request_hash": q.request_hash,
                    "registered_result": False,
                },
                release_class=ReleaseClass.RESTRICTED,
            )
            ledger.record_attempt(
                attempt_id=attempt,
                job_id=job.job_id,
                attempt_kind=AttemptKind.BASE,
                input_hash=q.request_hash,
                config_hash=q.decoding.content_hash,
                seed=q.decoding.seed,
            )
            result, failure = None, None
            tick = time.monotonic()
            stage("generation_and_exception_drain", 50)
            try:
                result = service.generate(
                    q,
                    event_id=attempt,
                    watchdog_seconds=45,
                    job_id=job.job_id,
                    attempt_id=attempt,
                    remaining_required_seconds=60,
                    accounting_details={
                        "block_id": cfg.BLOCK_ID,
                        "simple_ladder": True,
                        "registered_result": False,
                        "complete_forecast_exception": True,
                    },
                )
            except Exception as exc:
                failure = {"message": str(exc), "exception_chain": traceback.format_exc()}
            elapsed = time.monotonic() - tick
            stage("validation_and_bookkeeping", 10)
            text = result.parsed_object["diagnostic_text"] if result else ""
            parsed, normalization, parse_error = parse_text(text)
            if case_id == "control":
                parsed, parse_error = None, None
            evaluation = evaluate(case_id, parsed) if case_id != "control" else None
            response = (
                artifact_store.put_bytes(
                    result.raw_response,
                    media_type="text/event-stream",
                    release_class=ReleaseClass.RESTRICTED,
                )
                if result
                else None
            )
            events = ledger.gpu_events_with_prefix(attempt)
            if events:
                ledger.record_model_call(
                    model_call_id=attempt,
                    job_id=job.job_id,
                    attempt_id=attempt,
                    gpu_event_id=attempt,
                    backend=ModelBackend.VLLM_GPU,
                    call_role=ModelCallRole.PILOT,
                    retry_class=RetryClass.BASE,
                    model_manifest_hash=service.configuration.configuration_hash,
                    decoding_manifest_hash=q.decoding.content_hash,
                    request_hash=q.request_hash,
                    response_artifact_hash=response.content_hash if response else None,
                    construction_unit_hash=canonical_sha256(inputs[case_id]),
                    served_context_count=1,
                    prompt_tokens=result.prompt_tokens if result else 0,
                    completion_tokens=result.completion_tokens if result else 0,
                    allocated_gpu_seconds=sum(e.allocated_seconds for e in events),
                    successful=result is not None,
                )
            if failure or parse_error:
                ledger.record_failure(
                    attempt_id=attempt,
                    failure_kind=FailureKind.INVALID_OUTPUT if result else FailureKind.SERVICE,
                    message=(failure or {}).get("message", parse_error or "unavailable"),
                    details=failure or {"parse_error": parse_error},
                )
            outcome = {
                "case_id": case_id,
                "attempt_id": attempt,
                "request_hash": q.request_hash,
                "raw_text": text,
                "parsed": parsed,
                "normalization": normalization,
                "parse_error": parse_error,
                "evaluation": evaluation,
                "failure": failure,
                "request_seconds": elapsed,
                "response": result.public_manifest() if result else None,
                "registered_result": False,
                "transport_complete": result is not None,
                "completed_at": now(),
            }
            immutable(run / f"outcome-{case_id}.json", outcome)
            outcomes.append(outcome)
            print(
                json.dumps(
                    {"case": case_id, "response": outcome["response"], "evaluation": evaluation}
                ),
                flush=True,
            )
            if failure:
                stop_reason = "transport_failure_or_unsafe_service"
                break
        if outcomes and outcomes[-1]["case_id"] == "control":
            stop_reason = "direct_progression_failed_plain_language_control_completed"
    except Exception:
        stop_reason = "controller_exception"
        immutable(run / "controller-failure.json", {"exception_chain": traceback.format_exc()})
        raise
    finally:
        state.update(
            stage="shutdown",
            deadline_monotonic=min(time.monotonic() + 60, state["whole_deadline_monotonic"]),
            recorded_at=now(),
        )
        write_json_atomic(state, run / "state.json")
        service.shutdown(shutdown_seconds=max(1, state["deadline_monotonic"] - time.monotonic()))
        immutable(
            run / "terminal.json",
            {
                "outcomes": outcomes,
                "stop_reason": stop_reason,
                "ended_at": now(),
                "actual_allocated_seconds": service.meter.actual_allocated_gpu_seconds,
                "new_allocated_seconds": service.meter.actual_allocated_gpu_seconds - cfg.BASELINE,
                "open_allocations": len(ledger.unresolved_gpu_allocations()),
                "open_service_journals": len(ledger.unresolved_gpu_service_journals()),
                "registered_acceptance": False,
                "resource_samples": [s.public_manifest() for s in sampler.samples],
            },
        )
        sampler.close_probes()
