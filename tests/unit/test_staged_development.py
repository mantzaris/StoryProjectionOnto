"""Focused staged-development assembly and real service-guard controls (CPU only)."""

import copy

import pytest
from jsonschema import Draft202012Validator, ValidationError

from story_projection_onto.development_demo import read
from story_projection_onto.nested_semantic_candidate import reconstruct_candidate
from story_projection_onto.staged_development import (
    assemble,
    policy,
    prepare,
    repair_owner,
    split_fixture,
    validate_stage,
)
from tests.unit.test_development_demo import ROOT, authored_full_size_capacity
from tests.unit.test_development_demo import pinned as source_pinned
from tests.unit.test_semantic_generation import execution, source_fixture


@pytest.fixture(scope="module")
def pinned():
    return source_pinned.__wrapped__()


def test_lossless_complete_assembly_and_canonical_roundtrip():
    v = authored_full_size_capacity()
    a, b, c = split_fixture(v)
    assert assemble(a, b, c) == v
    f = source_fixture()
    reconstruct_candidate(
        assemble(a, b, c), evidence=f.evidence, upper=f.upper_ontology, execution=execution()
    )
    c["node_descriptions"][a["entities"][0]["entity_id"]]["contextual_type_id"] = "nT99"
    with pytest.raises(ValueError, match="overwrite semantic"):
        assemble(a, b, c)


@pytest.mark.parametrize("stage", list("ABC"))
def test_exact_stage_packing_grammar_and_reference_controls(stage, pinned):
    import json
    from dataclasses import asdict

    from story_projection_onto.staged_development_fixtures import capacity_fixture

    t, m = pinned
    a, b, c = split_fixture(capacity_fixture(ROOT))
    for kind in ("c1", "c2-q1", "c2-q2"):
        q, e, _, _ = prepare(ROOT, kind, stage, t, m, prior={"A": a, "B": b})
        v = {"A": a, "B": b, "C": c}[stage]
        validate_stage(stage, v, q)
        assert q.rendered_input_token_count == len(
            t.apply_chat_template(
                [asdict(x) for x in q.messages],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        assert q.rendered_input_token_count + q.decoding.maximum_output_tokens <= 12288
        assert (
            len(t.encode(json.dumps(v, separators=(",", ":")), add_special_tokens=False))
            <= q.decoding.maximum_output_tokens
        )
        body = json.loads(q.messages[-1].content)
        assert ("context" in body) == (kind != "c1")
        assert all(x.text in body["evidence_index"] for x in e)
        assert ("actual_stage_A" in body) == (stage in "BC")
        assert ("actual_stage_B" in body) == (stage == "C")
        if stage == "B":
            wrong = copy.deepcopy(b)
            wrong["assertions"][0]["content"]["subject_id"] = "nE999"
            with pytest.raises(ValidationError):
                validate_stage(stage, wrong, q)
        if stage == "C":
            wrong = copy.deepcopy(c)
            wrong["decisions"][0]["created_object_ids"] = ["nR999"]
            with pytest.raises(ValidationError):
                validate_stage(stage, wrong, q)
            wrong = copy.deepcopy(c)
            wrong["node_descriptions"].pop("nE1")
            with pytest.raises(ValidationError):
                validate_stage(stage, wrong, q)


def test_authority_shutdown_and_stage_repair_owner():
    p = policy(ROOT)
    p.admit(8983.249948, 6, 11, starting=True, seconds=1267.858133)
    with pytest.raises(TimeoutError):
        p.admit(10300, 7, 14, generating=True, seconds=1)
    with pytest.raises(ValueError):
        p.admit(8983.249948, 8, 11, starting=True, seconds=1)
    with pytest.raises(ValueError):
        p.admit(8983.249948, 6, 19, generating=True, seconds=1)
    assert repair_owner("C", [{"category": "unsupported_attribution"}]) == "B"
    assert repair_owner("C", [{"category": "unknown", "paths": ["/"]}]) is None


def test_retained_prefix_repair_is_specific_complete_and_packable(pinned):
    import json

    from scripts.report_preliminary_development import streamed_content
    from story_projection_onto.staged_development import prefix_duplicate_feedback

    r = (
        ROOT
        / "artifacts/restricted/staged-development-backup.q30cWu"
        / "nested-development-demonstration-20260908/run-20260908T190755327081"
    )
    if not r.exists():
        pytest.skip("Restricted actual response not available")
    t, m = pinned
    for p in sorted(r.glob("*/outcome.json")):
        o = read(p)
        raw = streamed_content(r, o["request_hash"])
        feedback = prefix_duplicate_feedback(raw)
        assert bool(feedback) == (o["kind"] == "c2-q2")
        if feedback:
            assert len(feedback[0]["paths"]) == 9
            assert feedback[0]["generated_duplicate_occurrences"] == 141
            q, e, _, _ = prepare(ROOT, o["kind"], "A", t, m, feedback=feedback)
            body = json.loads(q.messages[-1].content)
            assert all(x.text in body["evidence_index"] for x in e)
            assert body["repair"]["diagnostics"] == feedback
            assert body["repair"]["previous_stage_untrusted"] is None
            assert q.rendered_input_token_count + q.decoding.maximum_output_tokens <= 12288
            assert q.decoding.maximum_output_tokens == 6144
            old = read(p.parent / "request.json")
            assert q.output_schema != old["guided_json"]  # authorized bounded-v3 amendment


@pytest.mark.parametrize("pre_event", [False, True])
def test_staged_controller_real_guard_transport_and_independent_contexts(
    tmp_path, monkeypatch, pinned, pre_event
):
    """Run the actual workload/queue with simulated transport; no model service."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import scripts.run_capacity_diagnostics as controller
    import story_projection_onto.development_demo_execution as workload
    from story_projection_onto.gpu_runtime import GenerationResult, VLLMService
    from story_projection_onto.manifest import write_json_atomic

    _, m = pinned
    block = tmp_path / "block"
    run = block / "run"
    run.mkdir(parents=True)
    ledger = MagicMock()
    ledger.create_or_resume_job.return_value = SimpleNamespace(job_id="cpu-only")
    ledger.get_job.return_value = SimpleNamespace(job_id="cpu-only")
    ledger.attempt_lineage.return_value = [SimpleNamespace(job_id="cpu-only")]
    ledger.gpu_events_with_prefix.return_value = [
        SimpleNamespace(allocated_seconds=1, succeeded=True)
    ]
    ledger.register_artifact.side_effect = lambda pending: pending
    ledger.unresolved_gpu_allocations.return_value = []
    ledger.unresolved_gpu_service_journals.return_value = []
    sampler = MagicMock()
    sampler.samples = []
    service = MagicMock()
    service.pid = 12345
    service.configuration = SimpleNamespace(
        configuration_hash="a" * 64,
        snapshot_path=ROOT / "artifacts/restricted/pinned-tokenizer-cpu",
        served_model_name="qwen3-8b-awq-fallback",
    )
    service.meter.actual_allocated_gpu_seconds = 8983.249948
    service.actual_allocated_service_seconds = 8983.249948
    from story_projection_onto.staged_development import split_fixture
    from story_projection_onto.staged_development_fixtures import capacity_fixture

    a, b, c = split_fixture(capacity_fixture(ROOT))
    seen = []

    def response(q, **kwargs):
        seen.append(q.request_id)
        stage = q.request_id.split("-")[-1]
        stage = q.request_id.split("-")[-2] if stage == "repair" else stage
        payload = {"A": a, "B": b, "C": c}[stage]
        if q.request_id == "development-demo-staged-c2-q1-B":
            payload = {"broken": True}
        return GenerationResult(
            request_id=q.request_id,
            request_hash=q.request_hash,
            response_sha256="b" * 64,
            parsed_object=payload,
            raw_response=b"{}",
            prompt_tokens=q.rendered_input_token_count,
            completion_tokens=8,
            finish_reason="stop",
        )

    service.client.generate.side_effect = response
    service.generate.side_effect = lambda q, **kwargs: VLLMService.generate(service, q, **kwargs)
    if pre_event:
        from story_projection_onto.gpu_runtime import RuntimeConfigurationError

        service.require_service_capacity.side_effect = RuntimeConfigurationError(
            "CPU control: pre-event rejection"
        )
        ledger.gpu_events_with_prefix.return_value = []
    monkeypatch.setattr(controller, "setup", lambda *_: (ledger, sampler, service))
    monkeypatch.setattr(controller, "source_binding", lambda *_: {"cpu": "control"})
    monkeypatch.setattr(workload, "capture_tokenizer_manifest", lambda *a, **k: m)
    monkeypatch.setattr(workload, "validate_server_schema", Draft202012Validator.check_schema)
    monkeypatch.setattr(
        "story_projection_onto.staged_development.verify_auxiliary_grammar",
        lambda *a: {"local_simulation": True, "remote_backend_required": True},
    )
    from dataclasses import dataclass

    @dataclass
    class Hardware:
        label: str = "CPU simulation"

    monkeypatch.setattr(workload, "capture_gpu_hardware_identity", lambda *_: Hardware())
    monkeypatch.setitem(
        sys.modules,
        "xgrammar",
        SimpleNamespace(
            GrammarCompiler=lambda *_: SimpleNamespace(compile_json_schema=lambda *a, **k: None),
            TokenizerInfo=SimpleNamespace(from_huggingface=lambda *_: None),
        ),
    )
    original = workload.read
    monkeypatch.setattr(
        workload,
        "read",
        lambda p: (
            {"remaining_inventory_rows": []}
            if str(p).endswith("terminal-verification.json")
            else original(p)
        ),
    )
    write_json_atomic(
        {"source": {"cpu": "control"}, "configuration": "a" * 64}, run / "binding.json"
    )
    monkeypatch.setattr(
        workload,
        "ArtifactStore",
        lambda *_: SimpleNamespace(
            put_bytes=lambda *a, **k: SimpleNamespace(content_hash="c" * 64)
        ),
    )
    for n in range(1, 7):
        write_json_atomic({"historical": True}, block / f"start-{n:02d}.json")
    for n in range(1, 12):
        write_json_atomic({"historical": True}, block / f"attempt-{n:02d}.json")
    workload.execute_workload(ROOT, block, run, staged=True)
    outcomes = read(run / "terminal.json")["outcomes"]
    if pre_event:
        assert len(outcomes) == 1
        assert outcomes[0]["failure"]["stage"] == "pre_generation_service_contract"
        assert len(list(run.glob("*/failure.json"))) == 1
        ledger.record_model_call.assert_not_called()
        service.client.generate.assert_not_called()
        service.shutdown.assert_called_once()
        return
    assert [x["kind"] for x in outcomes] == ["c2-q1"] * 4 + ["c2-q2"] * 3
    assert [x["construction_stage"] for x in outcomes] == list("ABBCABC")
    assert sum(x["canonical_valid"] for x in outcomes) == 2
    assert outcomes[3]["stage_dependencies"]["B"] == outcomes[2]["attempt_id"]
    assert outcomes[2]["repair_parent"] == outcomes[1]["attempt_id"]
    assert all(not x["scientific_accepted"] for x in outcomes)

    service.shutdown.assert_called_once()
    assert len(list(block.glob("attempt-*.json"))) == 18
    assert service.client.generate.call_count == 7
    assert service.meter.repair.call_count == 1
    assert read(run / "terminal.json")["stop_reason"] == "completed"
    assert outcomes[-1]["mechanically_usable"]
    assert outcomes[-1]["assessment"]["contextual_metrics"] is not None


def test_auxiliary_limits_reject_growth_without_rewriting_output(pinned):
    from story_projection_onto.staged_development_fixtures import capacity_fixture

    t, m = pinned
    a, _, _ = split_fixture(capacity_fixture(ROOT))
    q, *_ = prepare(ROOT, "c2-q1", "A", t, m)
    for field, count in (("contextual_types", 7), ("predicates", 9)):
        wrong = copy.deepcopy(a)
        wrong["local_schema"][field] = [copy.deepcopy(a["local_schema"][field][0])] * count
        with pytest.raises(ValidationError):
            validate_stage("A", wrong, q)
    wrong = copy.deepcopy(a)
    wrong["entities"][0]["evidence_ids"] = ["e1"] * 5
    with pytest.raises(ValidationError):
        validate_stage("A", wrong, q)
    assert "uniqueItems" not in str(q.output_schema)


def test_mechanical_status_does_not_certify_wrong_semantics():
    from story_projection_onto.staged_development import mechanical_graph_status

    v = authored_full_size_capacity()
    original = copy.deepcopy(v)
    assert mechanical_graph_status(v)["mechanically_usable"]
    # A semantically incompatible but declared endpoint is still inspectable.
    v["instance_graph"]["assertions"][0]["content"]["subject_id"] = "nV1"
    assert mechanical_graph_status(v)["mechanically_usable"]
    v["instance_graph"]["assertions"][0]["content"]["subject_id"] = "nE999"
    assert not mechanical_graph_status(v)["mechanically_usable"]
    assert original != v
