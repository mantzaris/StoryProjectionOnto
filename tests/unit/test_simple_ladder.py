"""Focused CPU ladder controls; fixtures are never model-success claims."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from story_projection_onto.scorer_only.simple_ladder import (
    direct_gate,
    evaluate,
    references,
    toy_extract,
)
from story_projection_onto.simple_ladder import BASELINE, admit, cases, parse_text, prepare
from tests.unit.test_development_demo import pinned as pinned_fixture


@pytest.fixture(scope="module")
def pinned():
    return pinned_fixture.__wrapped__()


@pytest.mark.parametrize("case_id", [str(i) for i in range(1, 9)])
def test_frozen_references_and_actual_capacity(pinned, case_id):
    tokenizer, manifest = pinned
    q = prepare(cases()[case_id], tokenizer, manifest)
    assert q.rendered_input_token_count < (300 if case_id in ("1", "2") else 600)
    assert q.rendered_input_token_count + q.decoding.maximum_output_tokens <= 12288
    assert not {"guided_json", "response_format", "output_schema"} & set(q.wire_payload())
    assert q.canonical_output_schema is None
    assert "reference" not in q.messages[1].content.lower()
    for alt in references()[case_id]:
        authored = {"facts": alt}
        # This is only capacity: every authored complete alternative must fit.
        tokens = len(tokenizer.encode(json.dumps(authored), add_special_tokens=False))
        assert tokens < q.decoding.maximum_output_tokens
        assert evaluate(case_id, authored)["full"]["f1"] == 1


def test_no_semantic_completion_and_only_exact_fence():
    assert parse_text('```json\n{"facts": []}\n```')[1] == "exact_surrounding_markdown_fence"
    assert parse_text('{"facts": [{"subject": "unfinished')[0] is None
    assert parse_text('Here is JSON: {"facts": []}')[0] is None


def test_wrong_bindings_duplicate_omissions_and_unresolved_stay_in_denominator():
    good = references()["1"][0]
    predictions = copy.deepcopy(good)
    predictions += [
        good[0],
        {**good[0], "subject": "Tomas"},
        {**good[0], "relation": "mysterious paraphrase"},
    ]
    result = evaluate("1", {"facts": predictions})
    assert result["full"]["precision"] == 0.5
    assert result["full"]["predicted"] == 6
    assert len(result["unresolved"]) == 1
    assert evaluate("1", {"facts": good[:1]})["full"]["recall"] == 1 / 3


@pytest.mark.parametrize(
    "case_id,field,value", [("5", "valid_until", 99), ("6", "holder", "Someone else")]
)
def test_qualification_errors_do_not_erase_underlying_measurement(case_id, field, value):
    predictions = copy.deepcopy(references()[case_id][0])
    predictions[0][field] = value
    outcome = evaluate(case_id, {"facts": predictions})
    assert outcome["underlying"]["f1"] == 1
    assert outcome["full"]["f1"] == 0.5


def test_toy_baseline_parses_not_lookup():
    assert toy_extract({"Z9": "Newperson owns a sextant."})["facts"] == [
        dict(subject="Newperson", relation="owns", object="sextant", evidence_id="Z9")
    ]


def test_absent_qualifications_are_not_literal_sentinels():
    for field, value in (("holder", "ABSENT"), ("attitude", None), ("valid_from", "ABSENT")):
        facts = copy.deepcopy(references()["1"][0])
        facts[0][field] = value
        assert evaluate("1", {"facts": facts})["full"]["true_positive"] == 2
    from scripts.report_simple_ladder import graph

    assert "No mechanically" in graph([{"subject": {}, "object": []}])


def test_progression_and_bounds():
    from story_projection_onto.simple_ladder_execution import next_case

    fail = [{"case_id": k, "evaluation": evaluate(k, {"facts": []})} for k in ("1", "2")]
    assert not direct_gate(fail) and next_case(fail) == "control"
    assert next_case([*fail, {"case_id": "control"}]) is None
    good = [
        {"case_id": k, "evaluation": evaluate(k, {"facts": references()[k][0]})} for k in ("1", "2")
    ]
    assert direct_gate(good) and next_case(good) == "3"
    admit(BASELINE, 0, 0, starting=True, seconds=509.607992)
    for kwargs in [
        dict(starts=1, attempts=0, starting=True),
        dict(starts=1, attempts=8, generating=True),
    ]:
        with pytest.raises(ValueError):
            admit(BASELINE, seconds=1, **kwargs)
    with pytest.raises(ValueError):
        admit(BASELINE + 500, 1, 1, seconds=60)


@pytest.mark.parametrize("basics_pass", [True, False])
def test_actual_workload_service_guard_and_stream_transport(
    tmp_path, monkeypatch, pinned, basics_pass
):
    import sys

    import scripts.run_capacity_diagnostics as controller
    import story_projection_onto.simple_ladder_execution as workload
    from story_projection_onto.gpu_runtime import VLLMGuidedJSONClient, VLLMService
    from story_projection_onto.manifest import write_json_atomic

    tokenizer, manifest = pinned
    block = tmp_path / "artifacts/restricted/block"
    run = block / "run"
    run.mkdir(parents=True)
    ledger, sampler, service = MagicMock(), MagicMock(), MagicMock()
    ledger.create_or_resume_job.return_value = SimpleNamespace(job_id="cpu-only")
    ledger.gpu_events_with_prefix.return_value = [SimpleNamespace(allocated_seconds=1)]
    ledger.unresolved_gpu_allocations.return_value = []
    ledger.unresolved_gpu_service_journals.return_value = []
    sampler.samples = []
    service.pid = 12345
    service.configuration = SimpleNamespace(
        configuration_hash="a" * 64,
        snapshot_path=Path("artifacts/restricted/pinned-tokenizer-cpu"),
        served_model_name="qwen3-8b-awq-fallback",
    )
    service.meter.actual_allocated_gpu_seconds = BASELINE
    service.actual_allocated_service_seconds = BASELINE
    seen = []

    def transport(url, payload, timeout):
        payload = json.loads(payload)
        seen.append(payload)
        assert "guided_json" not in payload
        n = str(len(seen))
        text = (
            json.dumps({"facts": references()[n][0] if basics_pass else []})
            if n != "3" or basics_pass
            else "Mira carries a lantern. Tomas owns the lantern. Mira is in the courtyard."
        )
        events = [
            dict(id="cpu", choices=[dict(index=0, delta=dict(content=text), finish_reason=None)]),
            dict(
                id="cpu",
                choices=[dict(index=0, delta={}, finish_reason="stop")],
                usage=dict(prompt_tokens=100, completion_tokens=50),
            ),
            "[DONE]",
        ]
        body = "".join(
            "data: " + (x if isinstance(x, str) else json.dumps(x)) + "\n\n" for x in events
        ).encode()
        return 200, body, {"content-type": "text/event-stream"}

    service.client = VLLMGuidedJSONClient(
        "http://127.0.0.1:8000", transport=transport, diagnostic_root=run / "http"
    )
    service.generate.side_effect = lambda q, **kw: VLLMService.generate(service, q, **kw)
    monkeypatch.setattr(controller, "setup", lambda *_: (ledger, sampler, service))
    monkeypatch.setattr(controller, "source_binding", lambda *_: {"cpu": "test"})
    monkeypatch.setattr(workload, "capture_tokenizer_manifest", lambda *a, **k: manifest)

    @dataclass
    class Hardware:
        kind: str = "CPU fixture only"

    monkeypatch.setattr(workload, "capture_gpu_hardware_identity", lambda *_: Hardware())
    monkeypatch.setattr(
        workload,
        "ArtifactStore",
        lambda *_: SimpleNamespace(
            put_bytes=lambda *a, **k: SimpleNamespace(content_hash="c" * 64)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.protocol",
        SimpleNamespace(ChatCompletionRequest=SimpleNamespace(model_validate=lambda p: None)),
    )
    write_json_atomic({"source": {"cpu": "test"}, "configuration": "a" * 64}, run / "binding.json")
    frozen = {
        k: {
            "request_hash": q.request_hash,
            "input_tokens": q.rendered_input_token_count,
            "reserved_output_tokens": q.decoding.maximum_output_tokens,
            "maximum_context": 12288,
        }
        for k, q in ((k, prepare(c, tokenizer, manifest)) for k, c in cases().items())
    }
    write_json_atomic(frozen, block / "approved-request-hashes.json")
    workload.execute_workload(tmp_path, block, run)
    terminal = json.loads((run / "terminal.json").read_text())
    assert len(seen) == (8 if basics_pass else 3)
    assert service.require_service_capacity.call_count == len(seen)
    assert service.meter.inference.call_count == len(seen)
    assert service.shutdown.call_count == 1
    assert terminal["open_allocations"] == terminal["open_service_journals"] == 0
    assert all(o["transport_complete"] for o in terminal["outcomes"])
    from scripts.report_simple_ladder import render

    rows = render(run, tmp_path / "reports")
    assert len(rows) == len(seen)
    assert (tmp_path / "reports/figures/simple_synthetic_comparison.html").exists()


def test_actual_ladder_scores_and_report_consistency(tmp_path):
    import csv
    import hashlib

    from scripts.report_simple_ladder import render

    run = Path("artifacts/restricted/simple-ladder-backup.GwP5td/run-20260908T222604858308")
    if not run.exists():
        pytest.skip("restricted real outputs unavailable")
    terminal = json.loads((run / "terminal.json").read_text())
    expected = [1, 1, 1, 1, 1, 0, 0, 0.5]
    for outcome, f1 in zip(terminal["outcomes"], expected, strict=True):
        k = outcome["case_id"]
        assert evaluate(k, outcome["parsed"]) == outcome["evaluation"]
        assert outcome["evaluation"]["full"]["f1"] == f1
        packing = json.loads((run / "packing.json").read_text())[k]
        assert outcome["response"]["prompt_tokens"] == packing["input_tokens"]
        assert outcome["response"]["completion_tokens"] < packing["reserved_output_tokens"]
    rows = render(run, tmp_path)
    saved = list(csv.DictReader((tmp_path / "tables/simple_synthetic_results.csv").open()))
    for row, record in zip(rows, saved, strict=True):
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert float(record[key]) == value
    manifest = json.loads((tmp_path / "tables/simple_synthetic_manifest.json").read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == digest
    assert terminal["open_allocations"] == terminal["open_service_journals"] == 0
