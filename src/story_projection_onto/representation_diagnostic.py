"""CPU preparation for a frozen A/B/C representation comparison, not acceptance.

A and B use ordinary named canonical records and the same JSON object schema.
C uses the production lossless tuple codec. Semantic fields and validators are
identical; only representation/decoding differs. No answer is constructed here.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

from story_projection_onto.contracts import canonical_json, canonical_sha256
from story_projection_onto.gpu_runtime import ChatMessage, GuidedJSONRequest
from story_projection_onto.llm import DecodingManifest, PackingReport, PackingSection
from story_projection_onto.output_wire import (
    RecordTupleCodec,
    bounded_identifier_schema,
    pack_capacity_candidate,
    translate_references,
)

BASELINE = 4890.322639
ALLOWANCE = 1200
STARTS = 1
ATTEMPTS = 4
BLOCK_ID = "small-representation-comparison-20260907"
STARTUP = 360
GENERATION = 180
SHUTDOWN = 45
GUARD = 5
FROZEN_REQUESTS = {
    "A": "1ef4fc8e36c78458406897e3d654617459d9aa1f6af52ad1968112979586224e",
    "B": "49b1a1e7b8bb81f846c24fb8becc7736c8563aa4278dd926a908dab4da3e857b",
    "C": "c051b02ddecc4f389172f08a54f1e53764a2c6f5e68765072980137cbac4bb38",
}


def admit(actual, starts, attempts, *, starting=False, generating=False, seconds):
    """Only this new explicit allowance; earlier block/ledger is never reset."""
    import math

    if not all(math.isfinite(x) for x in (actual, seconds)) or actual < BASELINE or seconds < 0:
        raise ValueError("invalid comparison allocation")
    if starts + int(starting) > STARTS or attempts + int(generating) > ATTEMPTS:
        raise ValueError("comparison start/attempt limit")
    end = actual + seconds + SHUTDOWN
    if end > BASELINE + ALLOWANCE - GUARD or end > 33660 or end >= 35999:
        raise ValueError("comparison deadline must retain shutdown and global limits")
    return {
        "baseline_actual_seconds": BASELINE,
        "actual_seconds": actual,
        "envelope_end_seconds": end,
        "complete_forecast_exception": True,
        "ordinary_execution_authorized": False,
    }


def _repack(base, messages, schema, tokenizer, *, label, unconstrained=False):
    count = len(
        tokenizer.apply_chat_template(
            [asdict(m) for m in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    decoding = DecodingManifest.model_validate(
        base.decoding.model_dump(exclude={"content_hash"})
        | {
            "maximum_input_tokens": 6144,
            "maximum_output_tokens": 6144,
            "output_schema_hash": canonical_sha256(schema),
        }
    )
    packing = PackingReport.build(
        condition=base.condition,
        tokenizer_revision=decoding.tokenizer_revision,
        maximum_model_tokens=12288,
        maximum_input_tokens=6144,
        reserved_output_tokens=6144,
        sections=(
            *tuple(
                PackingSection(
                    name=name,
                    section_content_hash=canonical_sha256(
                        schema
                        if name == "output_schema"
                        else messages[0].content
                        if name == "system_prompt"
                        else json.loads(messages[1].content)[name]
                    ),
                    token_count=0,
                )
                for name in (
                    "evidence_snapshot",
                    "output_schema",
                    "system_prompt",
                    "upper_ontology",
                )
            ),
            PackingSection(
                name="complete_diagnostic_request",
                section_content_hash=canonical_sha256([asdict(m) for m in messages]),
                token_count=count,
            ),
        ),
        required_section_names=(
            "evidence_snapshot",
            "output_schema",
            "system_prompt",
            "upper_ontology",
            "complete_diagnostic_request",
        ),
        complete_evidence_snapshot=True,
        complete_evidence_packet=None,
        complete_sealed_ontology=None,
    )
    return GuidedJSONRequest(
        request_id="representation-diagnostic-" + label,
        model_name=base.model_name,
        condition=base.condition,
        messages=messages,
        output_schema=schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=count,
        stream_response=True,
        unconstrained_diagnostic=unconstrained,
    )


def prepare_comparison(small, fixture, tokenizer):
    """Reconstruct ONLY the complete supplied request, never its expected output."""
    # The original production packer used standard JSON for this small input.
    sections = translate_references(
        json.loads(small.messages[1].content), small.opaque_reference_aliases, decode=True
    )
    schema = copy.deepcopy(small.canonical_output_schema)
    # Backend normalization removed format=date-time. Make its semantics explicit
    # in BOTH representations; use a supported string pattern in both grammars.
    schema["$defs"]["OntologyDecision"]["properties"]["decided_at"]["pattern"] = (
        r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
    )
    evidence_ids = [e.evidence_id for e in fixture.evidence]
    mentions = [m.candidate_id for e in fixture.evidence for m in e.mention_candidates]
    instruction = (
        "Construct a minimal meaningful ontology from the complete development evidence below. "
        "No expected ontology is supplied. Return 2 to 4 supported entity/event nodes, "
        "1 to 3 evidence-grounded qualified assertions, their local types and predicates, "
        "and at least one supported construction decision. All descriptions and why_matters "
        "must cite supported assertions/evidence. Candidates are defeasible hints. "
        "Keep story time, intrinsic validity, discourse and revelation distinct. Unknown "
        "duration is unknown, not an interval invented from the observation. Do not infer "
        "causation from co-occurrence. Preserve belief/report holders where applicable. "
        "Use only supplied evidence IDs in evidence_ids (including decision evidence_ids): "
        + json.dumps(evidence_ids)
        + ". Mention candidate IDs are "
        + json.dumps(mentions)
        + ". "
        "Event candidates, passage IDs and hashes are NOT evidence IDs. Create short distinct "
        "n-prefixed local IDs (e.g. n1, n2); references to local objects use the exact IDs "
        "you create. Never create an I-plus-digits ID: that namespace is supplied handles only. "
        "decided_at is administrative UTC date-time, not story time: YYYY-MM-DDTHH:MM:SSZ. "
        "For this diagnostic use the supplied request timestamp "
        + fixture.requested_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        + ". "
        "Budget input_tokens and output_tokens are zero sentinels; count actual graph/display "
        "objects exactly. Empty graphs do not satisfy this task even if generic schema allows "
        "abstention. Return compact JSON only, without markdown, analysis or comments."
    )
    # Named records retain the exact canonical semantic fields, including optional
    # n-ary roles/events/epistemics. No smaller semantic task or expected answer.
    named_messages = (
        ChatMessage(
            role="system",
            content=instruction + "\n" + RecordTupleCodec(schema).legend(named_fields=True),
        ),
        ChatMessage(role="user", content=canonical_json(sections)),
    )
    a = _repack(small, named_messages, schema, tokenizer, label="A", unconstrained=True)
    b = _repack(small, named_messages, schema, tokenizer, label="B")
    # Reuse actual production codec and grammar, with the same task/semantics.
    base = _repack(
        small,
        (ChatMessage(role="system", content=instruction), named_messages[1]),
        schema,
        tokenizer,
        label="C",
    )
    c = replace(pack_capacity_candidate(base, tokenizer), stream_response=True)
    assert a.messages == b.messages and a.output_schema == b.output_schema
    assert (
        translate_references(
            json.loads(c.messages[1].content), c.opaque_reference_aliases, decode=True
        )
        == sections
    )
    assert c.canonical_output_schema == bounded_identifier_schema(schema)
    return {"A": a, "B": b, "C": c}
