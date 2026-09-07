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

from story_projection_onto import representation_diagnostic as comparison_policy
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    ConstructionSeal,
    FixedOntologyInput,
    OntologyDraft,
    PreconstructionRequest,
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
    DIAGNOSTIC_SECONDS,
    GUARD_SECONDS,
    LIVE_CHECK_SECONDS,
    MAX_ATTEMPTS,
    MAX_STARTS,
    SHUTDOWN_SECONDS,
    SMALL_START_BASELINE_SECONDS,
    SMALL_START_SECONDS,
    STARTUP_SECONDS,
    VALIDATION_SECONDS,
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
EXCEPTION_DRAIN_SECONDS = 5
SMALL_REQUEST_HASH = "cde6c6b6eedefab00aa46b7a01833998ca0b291ad8ff1daf55e574ad23681e7a"
SEMANTIC_SESSION = {
    "block_id": "small-semantic-interface-validation-20260907",
    "historical_actual_seconds": 5363.502630,
    "maximum_new_starts": 1,
    "maximum_new_attempts": 3,
    "maximum_additional_seconds": 1100,
    "global_maximum_seconds": 6463.502630,
    "startup_seconds": 360,
    "live_checks_seconds": 15,
    "generation_seconds": 180,
    "validation_seconds": 30,
    "shutdown_seconds": 60,
    "guard_seconds": 5,
    "second_evidence_id": "ev-03",
    "ordinary_execution_authorized": False,
}


def semantic_session_admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    import math

    settings = SEMANTIC_SESSION
    if not all(math.isfinite(x) for x in (actual, seconds)) or (
        actual < settings["historical_actual_seconds"] or seconds < 0
    ):
        raise ValueError("invalid semantic-session allocation")
    if starts + int(starting) > 1 or attempts + int(generating) > 3:
        raise ValueError("semantic-session start/attempt limit")
    end = actual + seconds + settings["shutdown_seconds"]
    if end > settings["global_maximum_seconds"] - 5 or end > 33660 or end >= 36000:
        raise ValueError("semantic-session deadline must preserve shutdown and global limits")
    return {
        "actual_seconds": actual,
        "envelope_end_seconds": end,
        "baseline_actual_seconds": settings["historical_actual_seconds"],
        "complete_forecast_exception": True,
        "ordinary_execution_authorized": False,
    }


def diagnostic_policy(semantic):
    if not semantic:
        return comparison_policy
    from types import SimpleNamespace

    return SimpleNamespace(
        BASELINE=SEMANTIC_SESSION["historical_actual_seconds"],
        BLOCK_ID=SEMANTIC_SESSION["block_id"],
        ALLOWANCE=1100,
        STARTS=1,
        ATTEMPTS=3,
        GENERATION=180,
        admit=semantic_session_admit,
    )


SMALL_SUCCESS_CRITERIA = {
    "scope": "one development evidence record; no expected answer supplied",
    "transport": "HTTP 200; SSE final choice, usage, DONE; finish_reason stop",
    "decoding": "complete JSON and deterministic lossless canonical reconstruction",
    "schema": "unchanged OntologyDraft Pydantic and production structural validation",
    "structure": "2-4 supported entity/event nodes, 1-3 assertions, required local "
    "types/predicates, at least one supported construction decision",
    "science": "unchanged development semantic-grounding audit; valid references, "
    "story/validity/discourse/revelation, descriptions and provenance; "
    "no missing semantics invented",
    "claim_boundary": "small production-path diagnostic only; "
    "not C1 acceptance or study feasibility",
}


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


def stage_deadline(
    *, started, now_monotonic, prior_block_seconds, stage_seconds, comparison=False, semantic=False
):
    """Leave a full shutdown reserve before the whole block ends, including gaps."""
    whole = (
        started
        + (
            diagnostic_policy(semantic).ALLOWANCE - prior_block_seconds
            if comparison or semantic
            else min(BLOCK_SECONDS - prior_block_seconds, SMALL_START_SECONDS)
        )
        - GUARD_SECONDS
    )
    limit = min(now_monotonic + stage_seconds, whole - (60 if semantic else SHUTDOWN_SECONDS))
    if limit <= now_monotonic:
        raise TimeoutError("block has no allocation left before shutdown reserve")
    return limit, whole


def generation_watchdog(stage_remaining, registered_watchdog):
    """Client expires before the guardian so its failure can reach the ledger.

    The extra CPU bookkeeping margin is admitted inside the SAME whole block,
    not added to the registered inference watchdog or shutdown reserve.
    """
    cap = min(registered_watchdog, stage_remaining - EXCEPTION_DRAIN_SECONDS)
    if cap <= 0:
        raise TimeoutError("no diagnostic allocation before exception/shutdown drain")
    return cap


def setup(root, run):
    limits = ResourceLimits.load(root / "configs/study/resource_limits.json")
    ledger = Ledger(root / "artifacts/restricted/phase1_acceptance.sqlite")
    # Both processes call this BEFORE the service starts. Never recover live
    # allocation journals by constructing a new meter during inference.
    meter = AllocatedGPUMeter.from_limits(ledger, limits)
    snapshot_manifest = read(root / "artifacts/public/manifests/model_snapshot_fallback.json")
    config = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=root
        / ".cache/shared/hub/models--Qwen--Qwen3-8B-AWQ/snapshots"
        / "4da05a8edb55c6046cce958586c33b61da07bb79",
        shared_cache=root / ".cache/shared",
        model_configuration_path=root / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=snapshot_manifest["manifest_sha256"],
    )
    config = replace(config, guided_decoding_disable_any_whitespace=True)
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


def diagnostic_metadata(http_root, request_hash):
    journals = sorted(
        http_root.glob(request_hash[:16] + "-*/events.jsonl"), key=lambda p: p.stat().st_mtime_ns
    )
    if not journals:
        return {}
    rows = [json.loads(line) for line in journals[-1].read_text().splitlines()]
    metadata = {"journal": str(journals[-1]), "usage": {}}
    for row in rows:
        if row["event"] == "completion_metadata":
            metadata.update(usage=row["usage"], finish_reason=row["finish_reason"])
        elif row["event"] == "response_headers":
            metadata["http_status"] = row["http_status"]
        elif row["event"] == "response_complete":
            metadata.update(response_complete=True, response_sha256=row["response_sha256"])
        elif row["event"] == "failure":
            metadata["failure_stage"] = row["stage"]
        elif row["event"] == "model_content_json_complete":
            metadata["model_content_json_complete"] = True
        elif row["event"] == "canonical_reconstruction_passed":
            metadata["canonical_reconstruction_passed"] = True
        elif row["event"] in {"stream_first_event", "stream_first_content"}:
            metadata[row["event"] + "_seconds"] = row["elapsed_seconds"]
        elif row["event"] == "stream_usage":
            metadata["usage"] = row["usage"]
        elif row["event"] == "stream_finish":
            metadata["finish_reason"] = row["finish_reason"]
        elif row["event"] == "stream_done":
            metadata["stream_done"] = True
    return metadata


def seal_actual_c1(root, c1):
    from story_projection_onto.conditions.base import preontology_semantic_hash, sealed_semantic_ids

    raw = read(root / "tests/fixtures/phase1/c1_pre_request.json")
    upper = UpperOntology.model_validate(raw["upper_ontology"])
    return ConstructionSeal(
        seal_id="capacity-actual-c1-seal",
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=raw["snapshot_hash"],
        ontology_hash=preontology_semantic_hash(upper, c1),
        constructed_at=datetime.now(UTC),
        sealed_at=datetime.now(UTC),
        sealed_object_ids=sealed_semantic_ids(c1),
    )


def actual_fixed(root, call, base_request, c1, tokenizer, *, seal):
    from story_projection_onto.conditions.base import preontology_semantic_hash, sealed_semantic_ids
    from story_projection_onto.phase1_acceptance import _condition_output_schema, _request_sections

    raw = read(root / call.request_fixture)
    upper = UpperOntology.model_validate(raw["upper_ontology"])
    if (
        seal.snapshot_hash != raw["snapshot_hash"]
        or seal.ontology_hash != preontology_semantic_hash(upper, c1)
        or seal.sealed_object_ids != sealed_semantic_ids(c1)
    ):
        raise ValueError("actual C1 seal differs from its accepted complete source")
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


def prepare_small_diagnostic(root, run, call, bridge, tokenizer, tokenizer_manifest):
    """One complete evidence record, never an authored answer or acceptance proxy."""
    from story_projection_onto.phase1_acceptance import build_acceptance_request

    resolved = bridge.resolve_call(
        call_id=call.call_id, condition=call.condition, request_fixture=call.request_fixture
    )
    raw = read(root / call.request_fixture)
    raw.pop("content_hash", None)
    evidence = resolved.evidence[:1]
    raw.update(
        request_id="capacity-small-evidence-only",
        evidence=[e.model_dump(mode="json") for e in evidence],
        snapshot_hash=canonical_sha256([e.model_dump(mode="json") for e in evidence]),
        budgets={
            "node_budget": 4,
            "assertion_budget": 3,
            "display_node_budget": 4,
            "display_assertion_budget": 3,
            "repair_attempt_budget": 0,
        },
        sealed_horizon={
            "horizon_id": "capacity-small-horizon",
            "max_discourse_position": {"passage_order": 1},
            "max_revelation_position": {"revelation_order": 1},
        },
    )
    fixture = PreconstructionRequest.model_validate(raw)
    path = run / "small-evidence-only-request.json"
    immutable(path, fixture.model_dump(mode="json"))
    small_call = replace(
        call,
        call_id="capacity-small-c1",
        request_fixture=str(path.relative_to(root)),
        watchdog_seconds=120,
    )
    request = build_acceptance_request(
        root=root,
        call=small_call.acceptance_call(),
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        model_name="qwen3-8b-awq-fallback",
        model_revision=tokenizer_manifest.tokenizer_revision,
        additional_sections={
            "diagnostic_scope": "This complete one-passage development snapshot is a minimal "
            "generation diagnostic, not full acceptance. Construct a small meaningful ontology "
            "with 2 to 4 supported nodes and 1 to 3 qualified assertions, the local "
            "types/predicate it needs, and an explicit supported construction decision. "
            "No expected graph is supplied."
        },
    )
    return (
        small_call,
        replace(pack_capacity_candidate(request, tokenizer), stream_response=True),
        fixture,
        resolved.evidence,
    )


def validate_small_diagnostic(result, fixture, oracle_evidence):
    """Same production structural validator and existing development support audit.

    The oracle requires its complete frozen evidence identity; only the single
    supplied record is admissible to the structural validator. Neither oracle
    nor authored graphs are rendered into the request.
    """
    draft = OntologyDraft.model_validate(result.parsed_object)
    if draft.budget_accounting.input_tokens != 0 or draft.budget_accounting.output_tokens != 0:
        raise ValueError("small diagnostic must preserve raw token sentinels")
    return _validate_small_canonical_draft(draft, result, fixture, oracle_evidence)


def validate_named_semantic_diagnostic(result, fixture, oracle_evidence, execution):
    """Diagnostic adapter entry point only; no scheduler/counter/GPU route activated.

    Retain the identical scientific checks, after runtime metadata reconstruction.
    The old production parser above still requires its original raw sentinels.
    """
    from story_projection_onto.semantic_generation import reconstruct

    if (
        result.finish_reason != "stop"
        or execution.input_tokens != result.prompt_tokens
        or execution.output_tokens != result.completion_tokens
        or execution.request_hash != result.request_hash
        or execution.response_hash != result.response_sha256
    ):
        raise ValueError("semantic diagnostic requires bound response identity and observed usage")
    adapted = result.diagnostic_journal.validate(
        "schema_validation",
        lambda: reconstruct(
            result.parsed_object,
            evidence=fixture.evidence,
            upper=fixture.upper_ontology,
            execution=execution,
            small=True,
        ),
    )
    verdict = _validate_small_canonical_draft(adapted.draft, result, fixture, oracle_evidence)
    return {
        **verdict,
        "adapter_provenance": adapted.provenance,
        "canonical_draft": adapted.draft.model_dump(mode="json"),
    }


def _validate_small_canonical_draft(draft, result, fixture, oracle_evidence):
    from story_projection_onto.scorer_only.acceptance_grounding import (
        audit_acceptance_semantic_grounding,
    )
    from story_projection_onto.validate import validate_draft_structure

    structural = validate_draft_structure(
        draft=draft,
        upper_ontology=fixture.upper_ontology,
        evidence=fixture.evidence,
        horizon=fixture.sealed_horizon,
        budgets=fixture.budgets,
        capabilities=fixture.capabilities,
    )
    result.diagnostic_journal.validate("schema_validation", structural.raise_for_errors)
    graph = draft.instance_graph
    if len(graph.entities) + len(graph.events) < 2 or not graph.assertions or not draft.decisions:
        raise ValueError("small diagnostic omitted its meaningful ontology structure")
    audit = audit_acceptance_semantic_grounding(draft=draft, evidence=oracle_evidence)
    result.diagnostic_journal.validate("scientific_validation", audit.raise_for_failure)
    return {
        "small_diagnostic_only": True,
        "schema_valid": True,
        "structural_valid": True,
        "scientific_valid": True,
        "grounding": audit.public_manifest(),
    }


def next_diagnostic(*, completed, accepted_c1):
    # Fourth-start authority is exactly one SMALL call even if it succeeds.
    return "small" if completed == 0 else None


def prepare_structural_semantic_retry(request, failure, tokenizer):
    """One diagnostic clarification of observed syntax/cross-field errors only.

    Never feed the scorer's expected facts back to the model. Full evidence is
    unchanged. An overlarge error or packing failure ends this repair branch.
    """
    if failure["stage"] not in {"canonical_schema_validation", "schema_or_structural_validation"}:
        return None
    message = failure["message"]
    if "semantic grounding audit" in message or len(message) > 4000:
        return None
    from story_projection_onto.gpu_runtime import ChatMessage
    from story_projection_onto.representation_diagnostic import _repack

    system = request.messages[0].content + (
        "\nStructural repair revision semantic-structural-retry-v1. The previous attempt "
        "failed the following check(s). Reconstruct from the same complete evidence, "
        "correcting these contract violations. No expected facts or graph are supplied. "
        "Do not fill missing semantics with unsupported guesses.\n" + message
    )
    try:
        return _repack(
            request,
            (ChatMessage(role="system", content=system), *request.messages[1:]),
            request.output_schema,
            tokenizer,
            label="semantic-structural-repair",
        )
    except ValueError:
        return None


def controller(root, block, run, *, prepare_only=False, comparison=False, semantic=False):
    policy_mode = diagnostic_policy(semantic)
    comparison = comparison or semantic
    shutdown_seconds = 60 if semantic else SHUTDOWN_SECONDS
    validation_seconds = 30 if semantic else VALIDATION_SECONDS
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
    sampler.prepare()
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
    small_call, small, small_fixture, small_oracle = prepare_small_diagnostic(
        root, run, calls[0], bridge, tokenizer, tokenizer_manifest
    )
    first = small
    if first.request_hash != SMALL_REQUEST_HASH:
        raise ValueError("small diagnostic request changed from the authorized exact request")
    semantic_fixtures = {}
    if semantic:
        from story_projection_onto.semantic_generation import build_small_request

        # Freeze both sources before allocation; no output-dependent selection.
        second_evidence = next(e for e in small_oracle if e.evidence_id == "ev-03")
        second_raw = small_fixture.model_dump(mode="json", exclude={"content_hash"})
        second_raw.update(
            request_id="semantic-small-second",
            evidence=[second_evidence.model_dump(mode="json")],
            snapshot_hash=canonical_sha256([second_evidence.model_dump(mode="json")]),
            sealed_horizon={
                "horizon_id": "semantic-second-horizon",
                "max_discourse_position": {"passage_order": 3},
                "max_revelation_position": {"revelation_order": 3},
            },
        )
        semantic_fixtures = {
            "semantic-first": small_fixture,
            "semantic-second": PreconstructionRequest.model_validate(second_raw),
        }
        variants = {
            k: replace(build_small_request(f, tokenizer, tokenizer_manifest), request_id=k)
            for k, f in semantic_fixtures.items()
        }
        if variants["semantic-first"].request_hash != (
            "bf083c095b1655da8bfb060f98e017c22e7554ef443e59c3bf22cbd5dc266ac4"
        ):
            raise ValueError("first semantic request differs from CPU-frozen interface")
        for label, fixture in semantic_fixtures.items():
            immutable(run / f"fixture-{label}.json", fixture.model_dump(mode="json"))
    else:
        variants = (
            comparison_policy.prepare_comparison(small, small_fixture, tokenizer)
            if comparison
            else {"small": small}
        )
    if comparison:
        if (
            not semantic
            and {label: req.request_hash for label, req in variants.items()}
            != comparison_policy.FROZEN_REQUESTS
        ):
            raise ValueError("prepared comparison differs from the CPU-frozen A/B/C requests")
        immutable(
            run / "comparison-requests.json",
            {
                label: {
                    "request_hash": req.request_hash,
                    "schema_hash": canonical_sha256(req.output_schema),
                    "template_inclusive_tokens": req.rendered_input_token_count,
                    "output_allowance": req.decoding.maximum_output_tokens,
                    "canonical_schema": req.canonical_output_schema,
                    "reference_map": req.opaque_reference_aliases,
                }
                for label, req in variants.items()
            },
        )
    _verify_second_recovery_decoder_compiles(first.output_schema)
    import xgrammar

    compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo.from_huggingface(tokenizer))
    grammar = compiler.compile_json_schema(json.dumps(first.output_schema), any_whitespace=False)
    matcher = xgrammar.GrammarMatcher(grammar)
    # Exercise the exact compiled decoder, not just an instruction in the prompt.
    if matcher.accept_string("{\n\n\n\n\n\n"):
        raise ValueError("restricted decoder still admits the observed whitespace loop")
    identifier_grammar = compiler.compile_json_schema(
        json.dumps(first.output_schema["$defs"]["QualifiedAssertion"]["prefixItems"][0]),
        any_whitespace=False,
    )
    if xgrammar.GrammarMatcher(identifier_grammar).accept_string('"0, 0, 0, '):
        raise ValueError("identifier grammar still admits the observed repetition")
    if not xgrammar.GrammarMatcher(identifier_grammar).accept_string('"n-assertion/01"'):
        raise ValueError("identifier grammar rejects legitimate scoped IDs")
    for label, request in variants.items():
        compiler.compile_json_schema(json.dumps(request.output_schema), any_whitespace=False)
        immutable(run / f"prepared-{label}.json", request.wire_payload())
        rendered = tokenizer.apply_chat_template(
            [asdict(m) for m in request.messages],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        immutable(
            run / f"rendered-{label}.json",
            {
                "text": rendered,
                "token_count": request.rendered_input_token_count,
                "request_hash": request.request_hash,
                "tokenizer_manifest": asdict(tokenizer_manifest),
            },
        )
    inventory = read(root / "artifacts/restricted/v10_validation/terminal-verification.json")[
        "remaining_inventory_rows"
    ]
    # The phase inventory's five future science loads do not include the still
    # pending acceptance/restart/resume service after this diagnostic stops.
    load_proxy = next(
        row["forecast_p95_seconds"] for row in inventory if row["call_class"] == "gpu_session_start"
    )
    forecast = capacity_forecast(
        inventory,
        pending_acceptance_resume_seconds=load_proxy,
        actual_allocated_seconds=service.meter.actual_allocated_gpu_seconds,
        additional_diagnostic_allowance=policy_mode.ALLOWANCE if comparison else BLOCK_SECONDS,
    )
    immutable(run / "small-success-criteria.json", SMALL_SUCCESS_CRITERIA)
    # Authored development fixture, never supplied to the model. This measures
    # representation capacity only, not a valid answer to the small snapshot.
    from story_projection_onto.output_wire import RecordTupleCodec, translate_references

    authored = read(root / "tests/fixtures/phase1/c1_pre_output.json")
    codec = RecordTupleCodec(
        first.canonical_output_schema, first.sealed_record_copies, first.opaque_reference_aliases
    )
    capacity_wire = codec.encode(
        translate_references(authored, first.opaque_reference_aliases, decode=False)
    )
    capacity_text = json.dumps(capacity_wire, separators=(",", ":"))
    capacity_tokens = len(tokenizer.encode(capacity_text, add_special_tokens=False))
    if capacity_tokens >= first.decoding.maximum_output_tokens:
        raise ValueError("authored development representation exceeds small output capacity")
    if comparison:
        capacities = {}
        for label, req in variants.items():
            sample = (
                authored
                if label in {"A", "B"}
                else read(
                    root
                    / "artifacts/restricted/semantic-interface-cpu-v3/small-two-node-authored.json"
                )
                if semantic
                else RecordTupleCodec(
                    req.canonical_output_schema,
                    req.sealed_record_copies,
                    req.opaque_reference_aliases,
                ).encode(translate_references(authored, req.opaque_reference_aliases, decode=False))
            )
            tokens = len(
                tokenizer.encode(
                    json.dumps(sample, separators=(",", ":")), add_special_tokens=False
                )
            )
            if tokens >= req.decoding.maximum_output_tokens:
                raise ValueError("complete authored output does not fit comparison allowance")
            capacities[label] = {
                "authored_output_tokens": tokens,
                "output_allowance": req.decoding.maximum_output_tokens,
                "not_model_output_or_reliability_proof": True,
            }
        immutable(run / "comparison-capacity.json", capacities)
    immutable(
        run / "small-capacity-check.json",
        {
            "kind": "authored_development_capacity_check_not_model_output_or_small_answer",
            "fixture_sha256": canonical_sha256(authored),
            "nodes": 4,
            "assertions": 2,
            "construction_decisions": 5,
            "compact_wire_tokens": capacity_tokens,
            "output_allowance": first.decoding.maximum_output_tokens,
            "template_inclusive_input_tokens": first.rendered_input_token_count,
            "total_reserved_tokens": first.rendered_input_token_count
            + first.decoding.maximum_output_tokens,
            "total_context_limit": 12288,
            "not_a_worst_case_bound": True,
        },
    )
    immutable(
        run / "cpu-preparation.json",
        {
            "seconds": time.monotonic() - cpu_started,
            "first_request_hash": (
                variants["semantic-first"].request_hash if semantic else first.request_hash
            ),
            "template_inclusive_tokens": (
                variants["semantic-first"].rendered_input_token_count
                if semantic
                else first.rendered_input_token_count
            ),
            "forecast": forecast,
            "source_binding": canonical_sha256(binding),
            "completed_at": now(),
        },
    )
    if prepare_only:
        sampler.close_probes()
        print(json.dumps({"cpu_preparation_passed": True, "gpu_allocated": False}), flush=True)
        return
    # Live hardware/storage checks stay here, after invariant preparation.
    hardware = capture_gpu_hardware_identity(root / "artifacts/public/manifests/environment.json")
    immutable(run / "hardware.json", asdict(hardware))
    sampler.sample(sample_id=run.name + "-preallocation", root_pid=os.getpid())
    starts = count_reservations(block, "start")
    attempts = count_reservations(block, "attempt")
    actual = service.meter.actual_allocated_gpu_seconds
    admission = (
        policy_mode.admit(
            actual,
            starts,
            attempts,
            starting=True,
            seconds=policy_mode.ALLOWANCE - shutdown_seconds - GUARD_SECONDS,
        )
        if comparison
        else CapacityRecoveryState(actual, starts, attempts).admit(
            remaining_mandatory_seconds=forecast["remaining_forecast_seconds"],
            # Full envelope with protected five-second guard; shutdown counted once.
            stage_seconds=SMALL_START_SECONDS - SHUTDOWN_SECONDS - GUARD_SECONDS,
            starting_service=True,
            complete_packing=True,
            feasibility_diagnostic_exception=True,
        )
    )
    started = time.monotonic()
    prior = actual - (policy_mode.BASELINE if comparison else BASELINE_SECONDS)
    session = (
        f"semantic-interface-start-{starts + 1}"
        if semantic
        else f"representation-start-{starts + 1}"
        if comparison
        else f"capacity-start-{starts + 1}"
    )
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
            comparison=comparison,
            semantic=semantic,
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
    accepted_c1_seal = None
    semantic_next = "semantic-first"
    semantic_retry_parent = None
    try:
        cap = stage("startup", STARTUP_SECONDS)
        service.start(
            session_id=session,
            event_id=event,
            watchdog_seconds=cap,
            remaining_required_seconds=shutdown_seconds,
        )
        stage("live_checks", LIVE_CHECK_SECONDS)
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
        while True:
            kind = (
                semantic_next
                if semantic
                else (list(variants)[len(outcomes)] if len(outcomes) < len(variants) else None)
                if comparison
                else next_diagnostic(completed=len(outcomes), accepted_c1=accepted_c1 is not None)
            )
            if kind is None:
                break
            if count_reservations(block, "attempt") >= (
                policy_mode.ATTEMPTS if comparison else MAX_ATTEMPTS
            ):
                print(
                    json.dumps({"diagnostic_limit_reached": True, "not_complete_acceptance": True}),
                    flush=True,
                )
                break
            index = -1 if comparison else {"c1": 0, "c2": 1, "fixed": 3, "small": -1}[kind]
            call = (
                replace(small_call, watchdog_seconds=policy_mode.GENERATION)
                if comparison
                else small_call
                if kind == "small"
                else calls[index]
            )
            fixed_fixture = None
            if comparison:
                request = variants[kind]
            elif index == 0:
                request = first
            elif kind == "small":
                request = small
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
                    request, fixed_fixture = actual_fixed(
                        root, call, base, accepted_c1, tokenizer, seal=accepted_c1_seal
                    )
                else:
                    if "sealed_ontology" in json.loads(base.messages[1].content):
                        raise ValueError("C2 diagnostic contains a preconstructed ontology")
                    request = pack_capacity_candidate(base, tokenizer)
                _verify_second_recovery_decoder_compiles(request.output_schema)
            if (not comparison and kind != "small") or not request.stream_response:
                raise ValueError("fifth start is restricted to one streamed small diagnostic")
            attempts = count_reservations(block, "attempt")
            receipt = (
                policy_mode.admit(
                    service.actual_allocated_service_seconds,
                    count_reservations(block, "start"),
                    attempts,
                    generating=True,
                    seconds=policy_mode.GENERATION + validation_seconds,
                )
                if comparison
                else CapacityRecoveryState(
                    service.actual_allocated_service_seconds,
                    count_reservations(block, "start"),
                    attempts,
                ).admit(
                    remaining_mandatory_seconds=forecast["remaining_forecast_seconds"],
                    stage_seconds=DIAGNOSTIC_SECONDS + VALIDATION_SECONDS,
                    diagnostic_generation=True,
                    complete_packing=True,
                    feasibility_diagnostic_exception=True,
                )
            )
            attempt_id = (
                f"semantic-diagnostic-{attempts + 1}"
                if semantic
                else f"representation-diagnostic-{attempts + 1}"
                if comparison
                else f"capacity-diagnostic-{attempts + 1}"
            )
            attempt_root = run / attempt_id
            attempt_root.mkdir(mode=0o700)
            request_value = request.wire_payload()
            immutable(attempt_root / "request.json", request_value)
            immutable(
                attempt_root / "rendered-chat.json",
                {
                    "text": tokenizer.apply_chat_template(
                        request_value["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                    "input_tokens": request.rendered_input_token_count,
                },
            )
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
                    "repair_parent": semantic_retry_parent,
                },
            )
            job = ledger.create_or_resume_job(
                (
                    {
                        "block": block.name,
                        "semantic_task_hash": semantic_fixtures[kind].content_hash,
                    }
                    if semantic
                    else {
                        "block": block.name,
                        "attempt": attempt_id,
                        "request_hash": request.request_hash,
                    }
                ),
                release_class=ReleaseClass.RESTRICTED,
            )
            ledger.record_attempt(
                attempt_id=attempt_id,
                job_id=job.job_id,
                attempt_kind=AttemptKind.REPAIR if semantic_retry_parent else AttemptKind.BASE,
                parent_attempt_id=semantic_retry_parent,
                input_hash=request.request_hash,
                config_hash=request.decoding.content_hash,
                seed=call.seed_block,
            )
            result = None
            failure = None
            validation = None
            schema_valid = False
            generation_schema_valid = None
            cap = generation_watchdog(
                stage(
                    "generation_and_exception_drain",
                    policy_mode.GENERATION if comparison else DIAGNOSTIC_SECONDS,
                ),
                call.watchdog_seconds,
            )
            tic = time.monotonic()
            generation_started_at = datetime.now(UTC)
            failure_stage = "client"
            try:
                result = service.generate(
                    request,
                    event_id=attempt_id,
                    watchdog_seconds=cap,
                    job_id=job.job_id,
                    attempt_id=attempt_id,
                    remaining_required_seconds=shutdown_seconds,
                    accounting_details={
                        "block_id": block.name,
                        "feasibility_diagnostic_only": True,
                        "complete_forecast_exception": True,
                    },
                )
                generation_seconds = time.monotonic() - tic
                generation_completed_at = datetime.now(UTC)
                stage("validation", validation_seconds)
                immutable(attempt_root / "decoded.json", result.parsed_object)
                failure_stage = "canonical_schema_validation"
                adapted = None
                if semantic:
                    from jsonschema import Draft202012Validator

                    from story_projection_onto.semantic_generation import (
                        ExecutionFacts,
                        reconstruct,
                    )

                    generation_schema_valid = False
                    Draft202012Validator(request.output_schema).validate(result.parsed_object)
                    generation_schema_valid = True
                    execution = ExecutionFacts(
                        result.request_hash,
                        result.response_sha256,
                        generation_started_at,
                        generation_completed_at,
                        result.prompt_tokens,
                        result.completion_tokens,
                    )
                    adapted = reconstruct(
                        result.parsed_object,
                        evidence=semantic_fixtures[kind].evidence,
                        upper=semantic_fixtures[kind].upper_ontology,
                        execution=execution,
                        small=True,
                    )
                    immutable(
                        attempt_root / "canonical.json", adapted.draft.model_dump(mode="json")
                    )
                    immutable(attempt_root / "adapter-provenance.json", adapted.provenance)
                else:
                    OntologyDraft.model_validate(result.parsed_object)
                schema_valid = True
                failure_stage = "schema_or_structural_validation"
                validation = (
                    validate_named_semantic_diagnostic(
                        result, semantic_fixtures[kind], small_oracle, execution
                    )
                    if semantic
                    else validate_small_diagnostic(result, small_fixture, small_oracle)
                    if kind == "small" or comparison
                    else validate_acceptance_generation(
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
                )
                failure_stage = "scientific_capability_validation"
                if kind != "small" and not comparison:
                    validation["operator_behavior"] = _require_call_operator_coverage(
                        call, validation, result.parsed_object, root=root
                    )
                if result.finish_reason != "stop":
                    raise ValueError(
                        "only stop-terminated output passes the diagnostic completion criterion"
                    )
                immutable(attempt_root / "validation.json", validation)
                if index == 0:
                    accepted_c1 = OntologyDraft.model_validate(result.parsed_object)
                    # Seal before opening any C2/FixedSelect query fixture, not
                    # retrospectively at FixedSelect request preparation.
                    accepted_c1_seal = seal_actual_c1(root, accepted_c1)
                    immutable(
                        attempt_root / "c1-seal.json", accepted_c1_seal.model_dump(mode="json")
                    )
                    immutable(
                        attempt_root / "accepted-c1.json", accepted_c1.model_dump(mode="json")
                    )
            except Exception as exc:
                if result is None:
                    generation_seconds = time.monotonic() - tic
                stage("validation_and_failure_drain", validation_seconds)
                failure = {
                    "stage": failure_stage,
                    "exception_type": type(exc).__name__,
                    "exception_chain": traceback.format_exc(),
                    "message": str(exc),
                }
                immutable(attempt_root / "failure.json", failure)
            transport_metadata = diagnostic_metadata(run / "http", request.request_hash)
            # Every raw response is already fsynced by the HTTP journal before
            # parsing, including failures with no GenerationResult.
            response_artifact = None
            if result is not None:
                response_artifact = artifact_store.put_bytes(
                    result.raw_response,
                    media_type="text/event-stream"
                    if request.stream_response
                    else "application/json",
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
                "transport_metadata": transport_metadata,
                "generation_event_seconds": seconds,
                "generation_and_validation_seconds": time.monotonic() - tic,
                "generation_wall_seconds": generation_seconds,
                "accepted": failure is None,
                "diagnostic_kind": kind,
                "canonical_schema_valid": schema_valid,
                "generation_schema_valid": generation_schema_valid,
                "reference_cross_field_valid": (
                    failure is None
                    or transport_metadata.get("failure_stage") == "scientific_validation"
                )
                if semantic and schema_valid
                else None,
                "scientifically_valid": failure is None,
                "production_form": kind != "small" and not comparison,
                "repair_parent": semantic_retry_parent,
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
                prompt_tokens=transport_metadata.get("usage", {}).get("prompt_tokens", 0)
                if result is None
                else result.prompt_tokens,
                completion_tokens=transport_metadata.get("usage", {}).get("completion_tokens", 0)
                if result is None
                else result.completion_tokens,
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
            if failure and transport_metadata.get("failure_stage") in {"transport", "http"}:
                # A failed transport may leave an in-flight generation. Do not
                # overlap it or keep using an unhealthy service.
                break
            if semantic:
                if failure is None:
                    semantic_next = "semantic-second" if kind.startswith("semantic-first") else None
                    semantic_retry_parent = None
                else:
                    # No retry from scorer oracle diagnostics. Only a concrete
                    # structural failure can support a prompt clarification.
                    if (
                        transport_metadata.get("failure_stage") == "scientific_validation"
                        or failure["stage"] == "client"
                        or attempts + 1 >= 3
                        or semantic_retry_parent is not None
                    ):
                        semantic_next = None
                    else:
                        stage("bounded_structural_repair_preparation", 90)
                        repaired = prepare_structural_semantic_retry(request, failure, tokenizer)
                        if repaired is None:
                            semantic_next = None
                        else:
                            semantic_next = kind + "-repair"
                            variants[semantic_next] = repaired
                            semantic_fixtures[semantic_next] = semantic_fixtures[kind]
                            semantic_retry_parent = attempt_id
                            compiler.compile_json_schema(
                                json.dumps(repaired.output_schema), any_whitespace=False
                            )
                            immutable(
                                run / f"prepared-{semantic_next}.json", repaired.wire_payload()
                            )
    finally:
        state.update(
            stage="shutdown",
            deadline_monotonic=min(
                time.monotonic() + shutdown_seconds, state["whole_deadline_monotonic"]
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
                - (policy_mode.BASELINE if comparison else BASELINE_SECONDS),
                "complete_acceptance": False,
                "remaining_forecast": forecast,
                "resource_samples": [s.public_manifest() for s in sampler.samples],
                "open_allocations": len(ledger.unresolved_gpu_allocations()),
                "open_service_journals": len(ledger.unresolved_gpu_service_journals()),
            },
        )


def guardian(root, *, prepare_only=False, comparison=False, semantic=False):
    policy_mode = diagnostic_policy(semantic)
    comparison = comparison or semantic
    block = root / "artifacts/restricted" / (policy_mode.BLOCK_ID if comparison else BLOCK_ID)
    block.mkdir(exist_ok=True, mode=0o700)
    with (block / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = block / ("run-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))
        run.mkdir(mode=0o700)
        ledger, _, service = setup(root, run)
        if ledger.unresolved_gpu_allocations() or ledger.unresolved_gpu_service_journals():
            raise ValueError("existing allocation requires reconciliation; do not duplicate")
        if not prepare_only and (
            count_reservations(block, "start") >= (policy_mode.STARTS if comparison else MAX_STARTS)
            or count_reservations(block, "attempt")
            >= (policy_mode.ATTEMPTS if comparison else MAX_ATTEMPTS)
        ):
            raise ValueError("existing block counters exhausted")
        if (
            not prepare_only
            and not comparison
            and (
                count_reservations(block, "start") != 4 or count_reservations(block, "attempt") != 3
            )
        ):
            raise ValueError(
                "fifth-start-only authorization requires four historical starts and three attempts"
            )
        authorization = read(root / "configs/study/output_capacity_recovery.json")
        if authorization["activation_scope"] != "bounded_feasibility_diagnostics_only":
            raise ValueError("missing narrow diagnostic authorization")
        expected = {
            "amendment_id": BLOCK_ID,
            "maximum_service_starts": MAX_STARTS,
            "maximum_diagnostic_generation_attempts": MAX_ATTEMPTS,
            "maximum_additional_allocated_seconds": BLOCK_SECONDS,
            "historical_actual_allocated_seconds": BASELINE_SECONDS,
            "scheduled_seconds": 33660,
            "strict_hard_seconds": 36000,
            "shutdown_reserve_seconds": SHUTDOWN_SECONDS,
        }
        if any(authorization.get(key) != value for key, value in expected.items()):
            raise ValueError("authorization differs from tested block constants")
        narrow = authorization["small_start_only"]
        if (
            narrow["maximum_allocated_seconds"] != SMALL_START_SECONDS
            or narrow["baseline_actual_seconds"] != SMALL_START_BASELINE_SECONDS
            or narrow["maximum_new_attempts"] != 1
            or narrow["permitted_task"] != "small_evidence_only"
            or narrow["startup_seconds"] != STARTUP_SECONDS
            or narrow["live_checks_seconds"] != LIVE_CHECK_SECONDS
            or narrow["diagnostic_seconds"] != DIAGNOSTIC_SECONDS
            or narrow["validation_seconds"] != VALIDATION_SECONDS
            or narrow["shutdown_seconds"] != SHUTDOWN_SECONDS
            or narrow["guard_seconds"] != GUARD_SECONDS
        ):
            raise ValueError("missing exact small-only authorization and stage caps")
        if comparison:
            # Separate explicit new allowance. Prior block remains immutable and
            # actual global ledger must include its terminal consumption.
            current = authorization[
                "semantic_interface_validation" if semantic else "representation_comparison"
            ]
            expected_comparison = (
                SEMANTIC_SESSION
                if semantic
                else {
                    "block_id": comparison_policy.BLOCK_ID,
                    "historical_actual_seconds": comparison_policy.BASELINE,
                    "maximum_new_starts": 1,
                    "maximum_new_attempts": 4,
                    "maximum_additional_seconds": 1200,
                    "permitted_variants": ["A", "B", "C"],
                    "fourth_call_requires_evidence_supported_repair": True,
                    "ordinary_execution_authorized": False,
                }
            )
            if current != expected_comparison:
                raise ValueError("comparison authorization differs from tested limits")
        binding = {
            "source": source_binding(root),
            "configuration": service.configuration.configuration_hash,
            "authorization": authorization,
            "guardian_pid": os.getpid(),
            "created_at": now(),
            "local_commit": os.environ.get("STORYPROJECTION_LOCAL_COMMIT", "CPU-check-only"),
        }
        immutable(run / "binding.json", binding)
        command = [sys.executable, __file__, "--controller", str(run)]
        if prepare_only:
            command.append("--prepare-only")
        if semantic:
            command.append("--semantic")
        elif comparison:
            command.append("--comparison")
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
                remaining = max(
                    1,
                    state.get("whole_deadline_monotonic", time.monotonic() + 60)
                    - time.monotonic()
                    - 1,
                )
                adopted = service.resume_live_service_lease(
                    expected_session_id=state["session"],
                    expected_event_id=state["event"],
                    cleanup_only=True,
                    watchdog_seconds=min(10, remaining / 3),
                )
                if adopted:
                    service.shutdown(shutdown_seconds=max(1, min(45, remaining - 10)))
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
        # After the controller/guardian has terminally reconciled shutdown, a
        # bounded CPU census does not hold the allocated service or its deadline.
        if (
            not prepare_only
            and not ledger.unresolved_gpu_allocations()
            and not ledger.unresolved_gpu_service_journals()
        ):
            checkpoint_sampler = ResourceSampler(
                limits=ResourceLimits.load(root / "configs/study/resource_limits.json"),
                storage=StoragePreflight(root),
            )
            try:
                checkpoint_sampler.prepare(timeout_seconds=120)
                checkpoint = checkpoint_sampler._storage_observation
            except Exception as error:
                checkpoint = {
                    "valid": False,
                    "error_type": type(error).__name__,
                    "exception_chain": traceback.format_exc(),
                }
            finally:
                checkpoint_sampler.close_probes()
            immutable(run / "storage-terminal-checkpoint.json", checkpoint)
        print(
            json.dumps({"run": str(run.relative_to(root)), "exit_code": child.returncode}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.controller:
        controller(
            root,
            args.controller.parent,
            args.controller,
            prepare_only=args.prepare_only,
            comparison=args.comparison,
            semantic=args.semantic,
        )
    elif args.prepare_only:
        guardian(root, prepare_only=True, comparison=args.comparison, semantic=args.semantic)
    elif args.execute:
        guardian(root, comparison=args.comparison, semantic=args.semantic)
    else:
        parser.error("--execute is required; no service is started by importing")
