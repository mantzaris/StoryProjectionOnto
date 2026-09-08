"""Development workload dispatched by the existing controller and guardian."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime

from jsonschema import Draft202012Validator

from .contracts import ConditionName, ConstructionSeal, canonical_json, canonical_sha256
from .development_adapter import DevelopmentConstructionConfiguration
from .development_demo import (
    adapt_output,
    development_validation_schema,
    nontransmitted_reservations,
    pending_work,
    phase_policy,
    prepare_request,
    read,
    sources,
)
from .gpu_runtime import (
    ResourceWatchdog,
    RuntimeConfigurationError,
    capture_gpu_hardware_identity,
    capture_tokenizer_manifest,
)
from .manifest import write_json_atomic
from .semantic_generation import ExecutionFacts
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


def validate_server_schema(schema):
    """Run the pinned server's actual request-validator before allocation."""
    from vllm.sampling_params import GuidedDecodingParams, SamplingParams
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar

    validate_xgrammar_grammar(SamplingParams(guided_decoding=GuidedDecodingParams(json=schema)))


def execute_workload(root, block, run, *, prepare_only=False):
    # No new guardian/process supervisor: these are the same tested lifecycle
    # helpers and stage-state contract used by the preceding bounded sessions.
    from transformers import AutoTokenizer

    from scripts.run_capacity_diagnostics import (
        count_reservations,
        diagnostic_metadata,
        generation_watchdog,
        immutable,
        now,
        setup,
        source_binding,
        stage_deadline,
    )

    from .conditions.base import preontology_semantic_hash, sealed_semantic_ids
    from .development_demo_fixed import adapt_fixed, prepare_fixed
    from .model_gate import FallbackModelPolicy
    from .scorer_only.development_demo_assessment import (
        assess,
        compact_feedback,
        exception_feedback,
        source_feedback,
    )

    cfg = phase_policy(root)
    ledger, sampler, service = setup(root, run)
    sampler.prepare()
    binding = read(run / "binding.json")
    if (
        binding["source"] != source_binding(root)
        or binding["configuration"] != service.configuration.configuration_hash
    ):
        raise ValueError("source/configuration differs from guardian binding")
    model = FallbackModelPolicy.load(root / "configs/study/fallback_model.json")
    tokenizer = AutoTokenizer.from_pretrained(
        service.configuration.snapshot_path, local_files_only=True
    )
    token_manifest = capture_tokenizer_manifest(
        service.configuration.snapshot_path, repository=model.repository, revision=model.revision
    )
    construction = DevelopmentConstructionConfiguration.load(
        root / "configs/study/development_construction.json"
    )
    _, unit, neutral = sources(root)
    variants = {
        kind: prepare_request(root, kind, tokenizer, token_manifest)
        for kind in ("c1", "c2-q1", "c2-q2")
    }
    queue, accepted_path = pending_work(block)
    nontransmitted = nontransmitted_reservations(block)
    for attempt in nontransmitted:
        if ledger.gpu_events_with_prefix(attempt):
            raise ValueError("reconciled reservation has a GPU event or model call")
        try:
            ledger.get_model_call(attempt)
        except KeyError:
            pass
        else:
            raise ValueError("reconciled reservation has a model call")
    prepared_retries = {
        parent: prepare_request(
            root, kind, tokenizer, token_manifest, previous={"parent": parent}, feedback=feedback
        )
        for kind, parent, feedback in queue
        if parent and not kind.startswith("fixed")
    }
    import xgrammar

    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo.from_huggingface(tokenizer))
    packing = {}
    for kind, (q, _, _mapping, _budgets) in variants.items():
        validate_server_schema(q.output_schema)
        compiler.compile_json_schema(canonical_json(q.output_schema), any_whitespace=False)
        immutable(run / f"prepared-{kind}.json", q.wire_payload())
        immutable(run / f"packing-{kind}.json", q.packing.model_dump(mode="json"))
        immutable(
            run / f"rendered-{kind}.json",
            {
                "text": tokenizer.apply_chat_template(
                    [asdict(m) for m in q.messages],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "tokenizer_manifest": asdict(token_manifest),
                "request_hash": q.request_hash,
                "input_tokens": q.rendered_input_token_count,
            },
        )
        packing[kind] = {
            "request_hash": q.request_hash,
            "input_tokens": q.rendered_input_token_count,
            "output_allowance": q.decoding.maximum_output_tokens,
        }
    for parent, (q, _, _, _) in prepared_retries.items():
        if (
            read(block / "prepared-parent-feedback.json")[parent]["prepared_request_hash"]
            != q.request_hash
        ):
            raise ValueError("Prepared parent repair request changed before allocation")
        validate_server_schema(q.output_schema)
        compiler.compile_json_schema(canonical_json(q.output_schema), any_whitespace=False)
        immutable(run / f"prepared-repair-{parent}.json", q.wire_payload())
        packing[parent] = {
            "request_hash": q.request_hash,
            "input_tokens": q.rendered_input_token_count,
            "output_allowance": q.decoding.maximum_output_tokens,
        }
    immutable(run / "pending-work.json", {"queue": queue, "prepared_before_allocation": True})
    immutable(
        run / "selection.json",
        {
            "unit": unit["unit_id"],
            "ordinals": [1, 2],
            "snapshot_hash": neutral["snapshot"]["content_hash"],
            "source_evidence": neutral,
            "selected_before_generation": True,
            "packing": packing,
            "alias_map": variants["c1"][2],
            "authoritative_amendment": cfg.config,
        },
    )
    prior = service.meter.actual_allocated_gpu_seconds - cfg.BASELINE
    if prior < 0:
        raise ValueError("authoritative ledger omitted historical allocation")
    inventory = read(root / "artifacts/restricted/v10_validation/terminal-verification.json")[
        "remaining_inventory_rows"
    ]
    immutable(
        run / "remaining-registered-inventory.json",
        {
            "rows": inventory,
            "ordinary_admission_claimed": False,
            "historical_actual": service.meter.actual_allocated_gpu_seconds,
        },
    )
    if prepare_only:
        sampler.close_probes()
        print(
            json.dumps(
                {"development_preparation_passed": True, "packing": packing, "gpu_started": False}
            ),
            flush=True,
        )
        return
    hardware = capture_gpu_hardware_identity(root / "artifacts/public/manifests/environment.json")
    immutable(run / "hardware.json", asdict(hardware))
    sampler.sample(sample_id=run.name + "-preallocation", root_pid=os.getpid())
    starts = count_reservations(block, "start")
    admission = cfg.admit(
        service.meter.actual_allocated_gpu_seconds,
        starts,
        count_reservations(block, "attempt"),
        starting=True,
        seconds=cfg.ALLOWANCE - prior - 65,
    )
    started = time.monotonic()
    session = f"{cfg.BLOCK_ID}-start-{starts + 1}"
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
            comparison=True,
            semantic="development",
        )
        state.update(
            stage=name,
            deadline_monotonic=deadline,
            whole_deadline_monotonic=whole,
            recorded_at=now(),
        )
        write_json_atomic(state, run / "state.json")
        return deadline - time.monotonic()

    stage("startup", cfg.config["startup_seconds"])
    immutable(block / f"start-{starts + 1:02d}.json", state | {"admission": admission})
    outcomes = []
    stop_reason = "completed"
    accepted_c1 = None
    accepted_seal = None
    if accepted_path:
        from .contracts import OntologyDraft

        accepted_c1 = OntologyDraft.model_validate(read(accepted_path.parent / "canonical.json"))
        accepted_seal = ConstructionSeal.model_validate(
            read(accepted_path.parent.parent / "c1-seal.json")
        )
    artifact_store = ArtifactStore(BlobStore(root / "artifacts/blobs/phase1_acceptance"), ledger)
    try:
        service.start(
            session_id=session,
            event_id=event,
            watchdog_seconds=state["deadline_monotonic"] - time.monotonic(),
            remaining_required_seconds=60,
        )
        stage("live_checks", 20)
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
        while queue:
            kind, parent, feedback = queue.pop(0)
            fixed = kind.startswith("fixed")
            if fixed and accepted_c1 is None:
                immutable(
                    run / f"{kind}-blocked.json",
                    {
                        "reason": (
                            "No scientifically accepted sealed C1 ontology; no authored substitute"
                        )
                    },
                )
                continue
            attempts = count_reservations(block, "attempt")
            if attempts >= 12:
                stop_reason = "attempt_limit"
                break
            if state["whole_deadline_monotonic"] - time.monotonic() < 125:
                stop_reason = "protected_shutdown_deadline"
                break
            stage("request_preparation", 60)
            try:
                if fixed:
                    _, evidence, mapping, _ = variants["c1"]
                    q, evidence, mapping, budgets = prepare_fixed(
                        root,
                        kind,
                        tokenizer,
                        token_manifest,
                        c1=accepted_c1,
                        seal=accepted_seal,
                        evidence=evidence,
                        mapping=mapping,
                        config=construction,
                        feedback=feedback,
                    )
                else:
                    q, evidence, mapping, budgets = (
                        variants[kind]
                        if parent is None
                        else prepared_retries.get(parent)
                        or prepare_request(
                            root,
                            kind,
                            tokenizer,
                            token_manifest,
                            previous={"parent": parent},
                            feedback=feedback,
                        )
                    )
                compiler.compile_json_schema(canonical_json(q.output_schema), any_whitespace=False)
            except Exception as exc:
                immutable(
                    run / f"{kind}-{'repair' if parent else 'base'}-packing-failure.json",
                    {
                        "parent": parent,
                        "exception": str(exc),
                        "full_exception": traceback.format_exc(),
                        "generation_attempt_consumed": False,
                    },
                )
                continue
            remaining = state["whole_deadline_monotonic"] - time.monotonic() - 60
            generation_limit = min(300, max(1, remaining - 65))
            cfg.admit(
                service.actual_allocated_service_seconds,
                count_reservations(block, "start"),
                attempts,
                generating=True,
                seconds=generation_limit + 60,
            )
            attempt_id = f"{cfg.BLOCK_ID}-attempt-{attempts + 1:02d}"
            attempt_root = run / attempt_id
            attempt_root.mkdir()
            immutable(attempt_root / "request.json", q.wire_payload())
            immutable(attempt_root / "packing.json", q.packing.model_dump(mode="json"))
            immutable(
                block / f"attempt-{attempts + 1:02d}.json",
                {
                    "attempt_id": attempt_id,
                    "request_hash": q.request_hash,
                    "parent": parent,
                    "kind": kind,
                    "reserved_at": now(),
                    "run": run.name,
                },
            )
            job = (
                ledger.get_job(ledger.attempt_lineage(parent)[-1].job_id)
                if parent
                else ledger.create_or_resume_job(
                    {
                        "development_demo": cfg.BLOCK_ID,
                        "kind": kind,
                        "source_hash": neutral["snapshot"]["content_hash"],
                    },
                    release_class=ReleaseClass.RESTRICTED,
                )
            )
            replacement_parent = (
                next(
                    (a for a, r in nontransmitted.items() if r["original_base_parent"] == parent),
                    None,
                )
                if parent
                else None
            )
            ledger.record_attempt(
                attempt_id=attempt_id,
                job_id=job.job_id,
                attempt_kind=AttemptKind.RETRY
                if replacement_parent
                else AttemptKind.REPAIR
                if parent
                else AttemptKind.BASE,
                parent_attempt_id=replacement_parent or parent,
                input_hash=q.request_hash,
                config_hash=q.decoding.content_hash,
                seed=q.decoding.seed,
            )
            result = None
            draft = None
            assessment = None
            defects = []
            failure = None
            generation_seconds = None
            started_at = datetime.now(UTC)
            tic = time.monotonic()
            try:
                cap = generation_watchdog(
                    stage("generation_and_exception_drain", generation_limit), generation_limit
                )
                result = service.generate(
                    q,
                    event_id=attempt_id,
                    watchdog_seconds=cap,
                    repair=parent is not None,
                    job_id=job.job_id,
                    attempt_id=attempt_id,
                    remaining_required_seconds=60,
                    accounting_details={
                        "block_id": cfg.BLOCK_ID,
                        "development_only": True,
                        "complete_forecast_exception": True,
                    },
                )
                generation_seconds = time.monotonic() - tic
                completed_at = datetime.now(UTC)
                stage("validation", 60)
                immutable(attempt_root / "decoded.json", result.parsed_object)
                defects = [] if fixed else source_feedback(result.parsed_object, evidence, mapping)
                schema_errors = list(
                    Draft202012Validator(
                        q.output_schema if fixed else development_validation_schema(q.output_schema)
                    ).iter_errors(result.parsed_object)
                )
                immutable(
                    attempt_root / "schema-errors.json",
                    [{"path": list(e.absolute_path), "message": e.message} for e in schema_errors],
                )
                for e in schema_errors:
                    defects.append(
                        {
                            "category": "generation_schema",
                            "path": "/" + "/".join(map(str, e.absolute_path)),
                            "constraint": "Must satisfy the supplied schema at this path: "
                            + e.validator,
                        }
                    )
                if result.finish_reason != "stop":
                    raise ValueError(f"incomplete generation finish_reason={result.finish_reason}")
                if schema_errors:
                    raise ValueError("generation schema failed; see full errors")
                facts = ExecutionFacts(
                    q.request_hash,
                    result.response_sha256,
                    started_at,
                    completed_at,
                    result.prompt_tokens,
                    result.completion_tokens,
                )
                adapted = (
                    adapt_fixed(
                        result.parsed_object,
                        c1=accepted_c1,
                        seal=accepted_seal,
                        upper=construction.upper_ontology,
                        execution=facts,
                    )
                    if fixed
                    else adapt_output(
                        result.parsed_object, evidence, mapping, construction.upper_ontology, facts
                    )
                )
                draft = adapted.draft
                immutable(attempt_root / "canonical.json", draft.model_dump(mode="json"))
                immutable(attempt_root / "adapter-provenance.json", adapted.provenance)
                assessment = assess(
                    draft,
                    evidence=evidence,
                    upper=construction.upper_ontology,
                    horizon=neutral["snapshot"]["horizon"],
                    budgets=budgets,
                    root=root,
                    ordinal=None if kind == "c1" else int(kind[-1]),
                    fixed=fixed,
                )
                immutable(attempt_root / "assessment.json", assessment)
                if defects or not assessment["scientific_accepted"]:
                    failure = {
                        "stage": "scientific_or_structural_validation",
                        "message": (
                            "Rejected or unresolved development output; see source and component "
                            "checks"
                        ),
                    }
            except Exception as exc:
                if result is None:
                    generation_seconds = time.monotonic() - tic
                failure = {
                    "stage": "pre_generation_service_contract"
                    if isinstance(exc, RuntimeConfigurationError) and result is None
                    else "transport_or_decoding"
                    if result is None
                    else "canonical_or_assessment",
                    "message": str(exc),
                    "exception_chain": traceback.format_exc(),
                }
                if result is not None and draft is None:
                    defects.extend(exception_feedback(exc))
            metadata = diagnostic_metadata(run / "http", q.request_hash)
            if failure and result is None and metadata.get("failure_stage"):
                failure["stage"] = metadata["failure_stage"]
            # Preserve the original error BEFORE any fallible bookkeeping.
            # A pre-event service rejection is not a model call and must never
            # fabricate a GPU event merely to satisfy the model-call table.
            if failure:
                immutable(attempt_root / "failure.json", failure)
            if result is not None:
                response = artifact_store.put_bytes(
                    result.raw_response,
                    media_type="text/event-stream",
                    release_class=ReleaseClass.RESTRICTED,
                )
            else:
                response = None
            events = ledger.gpu_events_with_prefix(attempt_id)
            seconds = sum(e.allocated_seconds for e in events)
            if events:
                ledger.record_model_call(
                    model_call_id=attempt_id,
                    job_id=job.job_id,
                    attempt_id=attempt_id,
                    gpu_event_id=attempt_id,
                    backend=ModelBackend.VLLM_GPU,
                    call_role=ModelCallRole.PILOT,
                    retry_class=RetryClass.STANDARD if parent else RetryClass.BASE,
                    model_manifest_hash=service.configuration.configuration_hash,
                    decoding_manifest_hash=q.decoding.content_hash,
                    request_hash=q.request_hash,
                    response_artifact_hash=response.content_hash if response else None,
                    construction_unit_hash=canonical_sha256(
                        {"kind": kind, "snapshot": neutral["snapshot"]["content_hash"]}
                    ),
                    served_context_count=1,
                    prompt_tokens=result.prompt_tokens
                    if result
                    else metadata.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=result.completion_tokens
                    if result
                    else metadata.get("usage", {}).get("completion_tokens", 0),
                    allocated_gpu_seconds=seconds,
                    successful=bool(events) and all(e.succeeded is True for e in events),
                )
            if failure:
                ledger.record_failure(
                    attempt_id=attempt_id,
                    failure_kind=FailureKind.INVALID_OUTPUT if events else FailureKind.SERVICE,
                    message=failure["message"],
                    details=failure,
                )
            structure = assessment["structure"] if assessment else None
            retry_feedback = compact_feedback(defects, structure)
            immutable(attempt_root / "repair-feedback.json", retry_feedback)
            outcome = {
                "attempt_id": attempt_id,
                "kind": kind,
                "condition": q.condition.value,
                "repair_parent": parent,
                "replaces_nontransmitted_reservation": replacement_parent,
                "development_request_revision": cfg.config["repair_policy_revision"],
                "request_hash": q.request_hash,
                "response": result.public_manifest() if result else None,
                "transport_metadata": metadata,
                "generation_seconds": generation_seconds,
                "allocated_generation_seconds": seconds,
                "canonical_valid": draft is not None,
                "scientific_accepted": failure is None,
                "source_defects": defects,
                "failure": failure,
                "assessment": assessment,
                "completed_at": now(),
            }
            immutable(attempt_root / "outcome.json", outcome)
            outcomes.append(outcome)
            print(
                json.dumps(
                    {
                        "attempt": attempt_id,
                        "kind": kind,
                        "canonical": draft is not None,
                        "accepted": failure is None,
                        "seconds": seconds,
                    }
                ),
                flush=True,
            )
            if result is None and not (
                metadata.get("http_status") == 200
                and metadata.get("response_complete")
                and metadata.get("stream_done")
            ):
                stop_reason = "unsafe_or_unresolved_transport"
                break
            if failure is None and kind == "c1":
                accepted_c1 = draft
                accepted_seal = ConstructionSeal(
                    seal_id=cfg.BLOCK_ID + "-c1-seal",
                    condition=ConditionName.C1_LLM_PRE,
                    snapshot_hash=neutral["snapshot"]["content_hash"],
                    ontology_hash=preontology_semantic_hash(construction.upper_ontology, draft),
                    constructed_at=completed_at,
                    sealed_at=datetime.now(UTC),
                    sealed_object_ids=sealed_semantic_ids(draft),
                )
                immutable(run / "c1-seal.json", accepted_seal.model_dump(mode="json"))
            if failure and parent is None and retry_feedback:
                queue.insert(0, (kind, attempt_id, retry_feedback))
        if accepted_c1 is None:
            immutable(
                run / "fixed-select-status.json",
                {
                    "status": "blocked",
                    "queries": [1, 2],
                    "reason": (
                        "No scientifically accepted sealed C1 ontology; no authored substitute"
                    ),
                },
            )
    except Exception as exc:
        stop_reason = "workload_exception"
        immutable(
            run / "controller-failure.json",
            {"message": str(exc), "exception_chain": traceback.format_exc()},
        )
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
                "additional_phase_seconds": service.meter.actual_allocated_gpu_seconds
                - cfg.BASELINE,
                "open_allocations": len(ledger.unresolved_gpu_allocations()),
                "open_service_journals": len(ledger.unresolved_gpu_service_journals()),
                "complete_acceptance": False,
                "resource_samples": [s.public_manifest() for s in sampler.samples],
            },
        )
        sampler.close_probes()
