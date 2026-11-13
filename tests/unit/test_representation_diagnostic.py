import json
from dataclasses import replace
from pathlib import Path

import pytest

from story_projection_onto import representation_diagnostic as policy
from story_projection_onto.contracts import OntologyDraft, PreconstructionRequest
from story_projection_onto.gpu_runtime import VLLMGuidedJSONClient
from story_projection_onto.output_wire import RecordTupleCodec, pack_capacity_candidate
from tests.integration.test_http_response_diagnostics import representative as old_fixture
from tests.unit.test_capacity_diagnostic_controller import driver
from tests.unit.test_fallback_acceptance import FakeTokenizer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def variants():
    _, _, base, envelope = old_fixture.__wrapped__()
    fixture = PreconstructionRequest.model_validate_json(
        (ROOT / "tests/fixtures/phase1/c1_pre_request.json").read_bytes()
    )
    return policy.prepare_comparison(
        pack_capacity_candidate(base, FakeTokenizer()), fixture, FakeTokenizer()
    ), envelope


def test_frozen_comparison_changes_only_A_decoding_and_C_representation(variants):
    requests, _ = variants
    a, b, c = (requests[k] for k in "ABC")
    ap, bp = a.wire_payload(), b.wire_payload()
    assert "guided_json" not in ap
    assert bp.pop("guided_json") == a.output_schema
    assert ap == bp
    assert a.messages == b.messages
    for request in requests.values():
        assert request.decoding.maximum_output_tokens == 6144
        assert request.decoding.maximum_model_tokens == 12288
        assert request.stream_response
        assert "YYYY-MM-DDTHH:MM:SSZ" in request.messages[0].content
        assert "NOT evidence IDs" in request.messages[0].content
    assert c.canonical_output_schema is not None
    assert a.canonical_output_schema is None


def test_named_output_reaches_unchanged_canonical_validator(variants):
    requests, envelope = variants
    for label in "AB":
        request = replace(requests[label], stream_response=False)
        envelope["usage"]["prompt_tokens"] = request.rendered_input_token_count
        response = VLLMGuidedJSONClient._decode_generation_response(
            request, 200, json.dumps(envelope).encode(), {}
        )
        assert OntologyDraft.model_validate(response.parsed_object)
        invalid = dict(response.parsed_object)
        invalid["decisions"] = [dict(invalid["decisions"][0], decided_at=",")]
        with pytest.raises(ValueError):
            OntologyDraft.model_validate(invalid)


def test_date_format_constraint_and_codec_are_consistent(variants):
    import jsonschema

    requests, envelope = variants
    b = requests["B"]
    canonical = json.loads(envelope["choices"][0]["message"]["content"])
    jsonschema.validate(canonical, b.output_schema)
    invalid = dict(canonical, decisions=[dict(canonical["decisions"][0], decided_at=",")])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, b.output_schema)
    codec = RecordTupleCodec(requests["C"].canonical_output_schema)
    assert codec.decode(codec.encode(canonical)) == canonical


def test_new_allowance_never_resets_history_and_caps_shutdown():
    assert policy.admit(policy.BASELINE, 0, 0, starting=True, seconds=1150)
    for kwargs in (
        dict(actual=policy.BASELINE - 1, starts=0, attempts=0, seconds=0),
        dict(actual=policy.BASELINE, starts=1, attempts=0, starting=True, seconds=0),
        dict(actual=policy.BASELINE, starts=1, attempts=4, generating=True, seconds=0),
        dict(actual=policy.BASELINE + 1100, starts=1, attempts=2, seconds=51),
    ):
        with pytest.raises(ValueError):
            policy.admit(**kwargs)
    limit, whole = driver.stage_deadline(
        started=100, now_monotonic=1200, prior_block_seconds=0, stage_seconds=180, comparison=True
    )
    assert whole == 1295 and limit == 1250


def test_unconstrained_flag_cannot_silently_enable_production(variants):
    request = variants[0]["A"]
    with pytest.raises(ValueError):
        replace(request, request_id="fallback-c1-01")
