"""Real local CPU HTTP/SSE pathway for both ordinary named JSON variants."""

import json
from dataclasses import replace

import pytest

from story_projection_onto.contracts import OntologyDraft
from story_projection_onto.gpu_runtime import VLLMGuidedJSONClient
from tests.integration.test_streaming_http_diagnostics import delta, event, sse_server
from tests.unit.test_representation_diagnostic import variants as _variants


@pytest.fixture
def variants():
    return _variants.__wrapped__()


@pytest.mark.parametrize("label", ["A", "B"])
def test_real_named_stream_retains_evidence_and_reaches_validation(tmp_path, variants, label):
    request = replace(variants[0][label], stream_response=True)
    content = variants[1]["choices"][0]["message"]["content"]
    parts = [
        delta(""),
        delta(content[:30]),
        delta(content[30:]),
        delta("", "stop"),
        event(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": request.rendered_input_token_count,
                    "completion_tokens": 50,
                },
            }
        ),
        event("[DONE]"),
    ]
    with sse_server(parts) as (url, observed, *_):
        result = VLLMGuidedJSONClient(url, diagnostic_root=tmp_path / "restricted").generate(
            request, watchdog_seconds=3
        )
    assert ("guided_json" in observed[0]) == (label == "B")
    assert OntologyDraft.model_validate(result.parsed_object)
    records = [
        json.loads(line)
        for path in (tmp_path / "restricted").rglob("events.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(r["event"] == "stream_done" for r in records)
    assert any(r["event"] == "model_content_json_complete" for r in records)
