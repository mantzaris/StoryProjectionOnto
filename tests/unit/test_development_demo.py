"""Focused development-amendment controls; authored CPU data is not model output."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from story_projection_onto.contracts import ConditionName, ConstructionSeal, OntologyDraft
from story_projection_onto.development_demo import (
    DevelopmentConstructionConfiguration,
    EvidenceRecord,
    aliases,
    compact_schema,
    phase_policy,
    prepare_request,
    read,
    reference_translation,
    sources,
)
from story_projection_onto.development_demo_fixed import GROUPS, adapt_fixed, prepare_fixed, records
from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.scorer_only.development_demo_assessment import (
    assess,
    compact_feedback,
    source_feedback,
)
from tests.unit.test_semantic_generation import execution

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pinned():
    from transformers import AutoTokenizer

    path = ROOT / "artifacts/restricted/pinned-tokenizer-cpu"
    if not path.exists():
        pytest.skip("Restricted local tokenizer not available")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    metadata = next(
        (ROOT / "artifacts/restricted/parent-repair-session-backup.vaJysn").glob(
            "**/rendered-semantic-first.json"
        )
    )
    return tokenizer, TokenizerManifest(**read(metadata)["tokenizer_manifest"])


@pytest.mark.parametrize("kind", ["c1", "c2-q1", "c2-q2"])
def test_exact_full_request_and_feedback_packing(pinned, kind):
    t, m = pinned
    q, e, mp, _ = prepare_request(ROOT, kind, t, m)
    assert len(e) == 25 and len(mp) == 159
    assert q.rendered_input_token_count == len(
        t.apply_chat_template(
            [asdict(x) for x in q.messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    assert q.rendered_input_token_count + q.decoding.maximum_output_tokens <= 12288
    assert ('"context":' in q.messages[1].content) == (kind != "c1")
    assert "intact_c1_graph" not in q.messages[1].content
    for record in e:
        assert record.text in q.messages[1].content
    repair, *_ = prepare_request(
        ROOT,
        kind,
        t,
        m,
        previous={"parent": "x"},
        feedback=[
            {
                "category": "unsupported_attribution",
                "path": "/instance_graph/assertions/0/epistemic_scope",
                "constraint": "Cited direct narration does not support this holder attitude",
            }
        ],
    )
    assert repair.decoding.maximum_output_tokens == 5120
    assert repair.rendered_input_token_count + 5120 <= 12288
    assert "retract unsupported" in repair.messages[-1].content
    assert repair.messages[:2] == q.messages
    Draft202012Validator.check_schema(q.output_schema)


def test_reference_aliases_are_lossless_and_never_translate_prose():
    _, _, n = sources(ROOT)
    e = tuple(EvidenceRecord.model_validate(x) for x in n["evidence"])
    mp = aliases(e)
    src = e[0].evidence_id
    v = {"evidence_ids": [src], "description": src, "nested": {"subject_id": "nE1"}}
    assert reference_translation(reference_translation(v, mp), {v: k for k, v in mp.items()}) == v
    assert reference_translation(v, mp)["description"] == src


def test_factored_enum_preserves_json_language():
    schema = {
        "$defs": {},
        "type": "object",
        "properties": {"x": {"enum": [f"e{i}" for i in range(25)]}},
        "required": ["x"],
    }
    compact = compact_schema(schema)
    for value in ["e1", "e24", "e99", None]:
        assert Draft202012Validator(schema).is_valid({"x": value}) == Draft202012Validator(
            compact
        ).is_valid({"x": value})


@pytest.mark.parametrize(
    "actual,starts,attempts,seconds",
    [(6716.108080, 0, 0, 1), (6716.108081, 4, 0, 1), (10300, 0, 0, 1), (6716.108081, 0, 12, 1)],
)
def test_phase_bounds_preserve_history_and_shutdown(actual, starts, attempts, seconds):
    with pytest.raises((ValueError, TimeoutError)):
        phase_policy(ROOT).admit(
            actual, starts, attempts, starting=True, generating=True, seconds=seconds
        )


def test_whole_deadline_overrides_stages():
    from scripts.run_capacity_diagnostics import stage_deadline

    deadline, whole = stage_deadline(
        started=100,
        now_monotonic=101,
        prior_block_seconds=3500,
        stage_seconds=420,
        semantic="development",
    )
    assert whole == 195 and deadline == 135


def test_only_explicit_development_request_can_use_candidate_allocation(pinned):
    t, m = pinned
    q, *_ = prepare_request(ROOT, "c1", t, m)
    with pytest.raises(ValueError):
        replace(q, request_id="ordinary-c1")


def source_draft():
    obj = read(
        ROOT / "artifacts/restricted/c0-calibration-identity-events-v9/dev-unit-01.preparation.json"
    )
    return OntologyDraft.model_validate(obj["sealed_preontology"]["draft"])


def test_c0_unchanged_artifact_and_existing_scorer_path():
    _, _, n = sources(ROOT)
    e = tuple(EvidenceRecord.model_validate(x) for x in n["evidence"])
    cfg = DevelopmentConstructionConfiguration.load(
        ROOT / "configs/study/development_construction.json"
    )
    a = assess(
        source_draft(),
        evidence=e,
        upper=cfg.upper_ontology,
        horizon=n["snapshot"]["horizon"],
        budgets=cfg.preconstruction_budgets,
        root=ROOT,
        ordinal=1,
    )
    assert a["source_direct_matching"]["metric"]["true_positive_count"] == 21
    assert a["source_direct_matching"]["metric"]["predicted_count"] == 25
    assert a["contextual_metrics"] is not None


def test_source_only_feedback_rejects_unsupported_time_and_attribution():
    _, _, n = sources(ROOT)
    e = tuple(EvidenceRecord.model_validate(x) for x in n["evidence"])
    mp = aliases(e)
    value = {
        "instance_graph": {
            "assertions": [
                {
                    "evidence_ids": ["e1"],
                    "content": {
                        "temporal_content": {"validity_time": {"kind": "point", "point": 1}}
                    },
                    "temporal_scope": "content",
                    "epistemic_scope": {
                        "holder_id": "nE1",
                        "attitude": "known",
                        "evidence_ids": ["e1"],
                    },
                }
            ]
        }
    }
    defects = source_feedback(value, e, mp)
    assert {d["category"] for d in defects} == {
        "unsupported_attribution",
        "unsupported_intrinsic_precision",
    }
    a = value["instance_graph"]["assertions"][0]
    a["epistemic_scope"] = None
    a["content"]["temporal_content"]["validity_time"] = {
        "kind": "unknown",
        "reason": "No interval stated",
    }
    assert source_feedback(value, e, mp) == []
    feedback = compact_feedback(defects + defects)
    assert len(feedback) == 2 and feedback[0]["generated_value"]["holder_id"] == "nE1"


def test_fixed_exact_records_and_unknown_id_rejected(pinned):
    from story_projection_onto.conditions.base import preontology_semantic_hash, sealed_semantic_ids

    c1 = source_draft()  # AUTHORED CPU mechanical control, never passed as model C1.
    cfg = DevelopmentConstructionConfiguration.load(
        ROOT / "configs/study/development_construction.json"
    )
    facts = execution()
    _, _, n = sources(ROOT)
    seal = ConstructionSeal(
        seal_id="cpu-only-control",
        condition=ConditionName.C1_LLM_PRE,
        snapshot_hash=n["snapshot"]["content_hash"],
        ontology_hash=preontology_semantic_hash(cfg.upper_ontology, c1),
        constructed_at=facts.generation_completed_at,
        sealed_at=facts.generation_completed_at,
        sealed_object_ids=sealed_semantic_ids(c1),
    )
    value = {
        "selected_" + k: [getattr(r, GROUPS[k]) for r in rows] for k, rows in records(c1).items()
    }
    value.update(contextual_interpretation="CPU selection control", uncertainty_and_abstentions=[])
    d = adapt_fixed(value, c1=c1, seal=seal, upper=cfg.upper_ontology, execution=facts).draft
    assert d.instance_graph == c1.instance_graph
    value["selected_entities"].append("invented")
    with pytest.raises(ValueError, match="unknown"):
        adapt_fixed(value, c1=c1, seal=seal, upper=cfg.upper_ontology, execution=facts)
    t, m = pinned
    e = tuple(EvidenceRecord.model_validate(x) for x in n["evidence"])
    with pytest.raises(ValueError, match="Intact C1"):
        prepare_fixed(
            ROOT, "fixed-q1", t, m, c1=c1, seal=seal, evidence=e, mapping=aliases(e), config=cfg
        )


@pytest.mark.parametrize("pre_event", [False, True])
def test_controller_scientific_failure_does_not_block_c2_or_shutdown(
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
    service.meter.actual_allocated_gpu_seconds = 6716.108081
    service.actual_allocated_service_seconds = 6716.108081
    service.client.generate.side_effect = lambda q, **kwargs: GenerationResult(
        request_id=q.request_id,
        request_hash=q.request_hash,
        response_sha256="b" * 64,
        parsed_object={"broken": True},
        raw_response=b"{}",
        prompt_tokens=q.rendered_input_token_count,
        completion_tokens=8,
        finish_reason="stop",
    )
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
    workload.execute_workload(ROOT, block, run)
    outcomes = read(run / "terminal.json")["outcomes"]
    if pre_event:
        assert len(outcomes) == 1
        assert outcomes[0]["failure"]["stage"] == "pre_generation_service_contract"
        assert len(list(run.glob("*/failure.json"))) == 1
        ledger.record_model_call.assert_not_called()
        service.client.generate.assert_not_called()
        service.shutdown.assert_called_once()
        return
    assert [x["kind"] for x in outcomes] == ["c1", "c1", "c2-q1", "c2-q1", "c2-q2", "c2-q2"]
    assert all(not x["scientific_accepted"] for x in outcomes)
    assert all(x["repair_parent"] is not None for x in outcomes[1::2])
    service.shutdown.assert_called_once()
    assert len(list(block.glob("attempt-*.json"))) == 6
    assert service.client.generate.call_count == 6
    assert service.meter.repair.call_count == 3


def test_actual_service_repair_flag_guard_reproduces_pre_event_failure(pinned):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from story_projection_onto.gpu_runtime import RuntimeConfigurationError, VLLMService

    t, m = pinned
    q, *_ = prepare_request(ROOT, "c1", t, m, previous={"parent": "failed"}, feedback=[])
    service = SimpleNamespace(
        require_ready=lambda: None,
        require_service_capacity=lambda *a, **k: None,
        configuration=SimpleNamespace(served_model_name=q.model_name),
        meter=MagicMock(),
        client=MagicMock(),
    )
    with pytest.raises(RuntimeConfigurationError, match="repair event kind differs"):
        VLLMService.generate(service, q, event_id="cpu-control", watchdog_seconds=300)
    service.client.generate.assert_not_called()
    service.meter.repair.assert_not_called()
    VLLMService.generate(service, q, event_id="cpu-control", watchdog_seconds=300, repair=True)
    service.meter.repair.assert_called_once()
    service.client.generate.assert_called_once()


def test_report_pipeline_actual_c0_and_explicit_unattempted_llm(tmp_path):
    from scripts.report_preliminary_development import build, complete_record_fragments
    from story_projection_onto.manifest import write_json_atomic

    run = tmp_path / "run"
    run.mkdir()
    write_json_atomic(
        {
            "actual_allocated_seconds": 6716.108081,
            "open_allocations": 0,
            "open_service_journals": 0,
            "stop_reason": "CPU report control; no GPU",
        },
        run / "terminal.json",
    )
    result = build(ROOT, run, tmp_path / "reports")
    assert len(result["rows"]) == 4
    report = (tmp_path / "reports/PRELIMINARY_DEVELOPMENT_RESULTS.md").read_text()
    assert "not proof" in report and "No GPU output was scientifically accepted" in report
    table = (tmp_path / "reports/tables/preliminary_development_results.json").read_text()
    assert "score.ctx_" not in table and "assertion_target_ids" not in table
    fragment = complete_record_fragments(
        '{"instance_graph":{"entities":[{"entity_id":"nE1","label":"actual '
        'complete fragment"},{"entity_id":"nE2","label":"cut'
    )
    assert len(fragment["instance_graph"]["entities"]) == 1
    assert fragment["partial_records_only"]


def test_resume_only_parent_bound_repairs_never_repeats_a_base(tmp_path):
    from story_projection_onto.development_demo import pending_work
    from story_projection_onto.manifest import write_json_atomic

    block = tmp_path
    run = block / "run-original"
    for i, kind in enumerate(("c1", "c2-q1", "c2-q2"), 1):
        d = run / f"attempt-{i}"
        d.mkdir(parents=True)
        o = {
            "kind": kind,
            "attempt_id": f"a{i}",
            "repair_parent": None,
            "scientific_accepted": False,
            "request_hash": str(i) * 64,
            "transport_metadata": {"response_sha256": "f" * 64},
        }
        write_json_atomic(o, d / "outcome.json")
        write_json_atomic({"attempt_id": f"a{i}"}, block / f"attempt-{i:02d}.json")
    feedback = {
        f"a{i}": {
            "parent_request_hash": str(i) * 64,
            "parent_response_hash": "f" * 64,
            "diagnostics": [{"category": "actual_defect"}],
        }
        for i in (1, 2, 3)
    }
    write_json_atomic(feedback, block / "prepared-parent-feedback.json")
    queue, _ = pending_work(block)
    assert [q[1] for q in queue[:3]] == ["a1", "a2", "a3"]
    assert not any(q[0].startswith(("c1", "c2")) and q[1] is None for q in queue)
    feedback["a1"]["parent_response_hash"] = "0" * 64
    write_json_atomic(feedback, block / "prepared-parent-feedback.json")
    with pytest.raises(ValueError, match="bound"):
        pending_work(block)


def test_persisted_feedback_key_order_does_not_change_wire_request(pinned):
    import json

    t, m = pinned
    feedback = [
        {
            "path": "/decisions",
            "category": "construction_reporting",
            "constraint": "Only report actual operations",
        }
    ]
    a, *_ = prepare_request(ROOT, "c2-q1", t, m, previous={"parent": "a"}, feedback=feedback)
    b, *_ = prepare_request(
        ROOT,
        "c2-q1",
        t,
        m,
        previous={"parent": "a"},
        feedback=json.loads(json.dumps(feedback, sort_keys=True)),
    )
    assert a.request_hash == b.request_hash and a.wire_payload() == b.wire_payload()


def test_existing_budgets_bound_generation_without_new_type_ceiling(pinned):
    import copy
    import json

    from story_projection_onto.development_demo import (
        backend_generation_schema,
        budget_schema,
        development_validation_schema,
        field_guide,
    )
    from story_projection_onto.nested_semantic_candidate import candidate_schema

    _, _, neutral = sources(ROOT)
    evidence = tuple(EvidenceRecord.model_validate(e) for e in neutral["evidence"])
    cfg = DevelopmentConstructionConfiguration.load(
        ROOT / "configs/study/development_construction.json"
    )
    b = cfg.projection_budgets_by_unit["dev-unit-01"]
    s = budget_schema(candidate_schema(evidence, cfg.upper_ontology), b)
    assert "uniqueItems" not in json.dumps(backend_generation_schema(s))
    assert development_validation_schema(backend_generation_schema(s)) == s
    for count, branch in enumerate(s["$defs"]["InstanceGraph"]["anyOf"]):
        graph = branch["properties"]
        assert graph["entities"]["maxItems"] == graph["entities"]["minItems"] == count
        assert graph["events"]["maxItems"] + count == b.node_budget
        assert graph["assertions"]["maxItems"] == b.assertion_budget
    choices = {
        x["properties"]["operator"]["const"]: x for x in s["$defs"]["OntologyDecision"]["anyOf"]
    }
    merge = choices["merge"]["properties"]["created_object_ids"]
    validator = Draft202012Validator(merge)
    assert validator.is_valid([f"nE{i}" for i in range(1, 11)])
    assert not validator.is_valid([f"nE{i}" for i in range(1, 12)])
    assert not validator.is_valid(["nV1"])
    assert not validator.is_valid(["nE1", "nE1"])
    assert choices["selection"]["properties"]["created_object_ids"]["maxItems"] == 0
    assert "maxItems" not in choices["contextual_type"]["properties"]["created_object_ids"]
    unchanged = copy.deepcopy(s)
    guide = field_guide(s)
    assert s == unchanged and "content_identity" in guide and "Citations=" in guide


def test_nontransmitted_reservation_preserved_without_blocking_one_repair(tmp_path):
    from story_projection_onto.contracts import canonical_sha256
    from story_projection_onto.development_demo import pending_work
    from story_projection_onto.manifest import write_json_atomic

    base = {
        "kind": "c1",
        "attempt_id": "a1",
        "repair_parent": None,
        "scientific_accepted": False,
        "request_hash": "1" * 64,
        "transport_metadata": {"response_sha256": "2" * 64},
    }
    write_json_atomic(base, tmp_path / "run-first/a1/outcome.json")
    reservation = {"attempt_id": "a2", "parent": "a1"}
    write_json_atomic(reservation, tmp_path / "attempt-02.json")
    with pytest.raises(ValueError, match="reconcile"):
        pending_work(tmp_path)
    write_json_atomic(
        {
            "a2": {
                "reservation_file": "attempt-02.json",
                "reservation_hash": canonical_sha256(reservation),
                "transmitted": False,
                "preserve_reservation_count": True,
                "original_base_parent": "a1",
            }
        },
        tmp_path / "reservation-reconciliations.json",
    )
    write_json_atomic(
        {
            "a1": {
                "parent_request_hash": "1" * 64,
                "parent_response_hash": "2" * 64,
                "diagnostics": [{"category": "actual_defect"}],
            }
        },
        tmp_path / "prepared-parent-feedback.json",
    )
    queue, _ = pending_work(tmp_path)
    assert queue[0][0:2] == ("c1", "a1")
    assert read(tmp_path / "attempt-02.json") == reservation
    repaired = {**base, "attempt_id": "a3", "repair_parent": "a1"}
    write_json_atomic(repaired, tmp_path / "run-next/a3/outcome.json")
    queue, _ = pending_work(tmp_path)
    assert not any(k == "c1" for k, _, _ in queue)


def authored_full_size_capacity():
    """Ten-node authored stress, never supplied to a model or scored as a result."""
    import copy
    import re

    from tests.unit.test_nested_semantic_candidate import authored_candidate
    from tests.unit.test_small_retry_path import demanding_capacity_wire
    from tests.unit.test_small_semantic_reconciliation import binary_wire

    parts = []
    for i, wire in enumerate([demanding_capacity_wire(), demanding_capacity_wire(), binary_wire()]):

        def visit(v, offset=i * 100):
            if isinstance(v, dict):
                return {k: visit(x) for k, x in v.items()}
            if isinstance(v, list):
                return [visit(x) for x in v]
            if isinstance(v, str) and re.fullmatch(r"n[STREVAD][0-9]+", v):
                return v[:2] + str(int(v[2:]) + offset)
            return v

        parts.append(visit(authored_candidate(wire)))
    result = copy.deepcopy(parts[0])
    result["contextual_interpretation"] = (
        "Authored ten-node packing stress only, not an expected graph or model result."
    )
    for p in parts[1:]:
        for key in ("entities", "events", "assertions", "unasserted_contents"):
            result["instance_graph"][key] += p["instance_graph"][key]
        for key in ("contextual_types", "predicates"):
            result["local_schema"][key] += p["local_schema"][key]
        result["decisions"] += p["decisions"]
    return result


def test_full_size_authored_serialization_capacity_and_lossless_adapter(pinned):
    import json

    from story_projection_onto.development_demo import budget_schema
    from story_projection_onto.nested_semantic_candidate import (
        candidate_schema,
        collapse_candidate,
        expand_candidate,
        reconstruct_candidate,
    )
    from tests.unit.test_semantic_generation import source_fixture

    t, _ = pinned
    value = authored_full_size_capacity()
    fixture = source_fixture()
    cfg = DevelopmentConstructionConfiguration.load(
        ROOT / "configs/study/development_construction.json"
    )
    schema = budget_schema(
        candidate_schema(fixture.evidence, fixture.upper_ontology),
        cfg.projection_budgets_by_unit["dev-unit-01"],
    )
    Draft202012Validator(schema).validate(value)
    expanded = expand_candidate(value, evidence=fixture.evidence, upper=fixture.upper_ontology)
    assert collapse_candidate(expanded) == value
    reconstruct_candidate(
        value, evidence=fixture.evidence, upper=fixture.upper_ontology, execution=execution()
    )
    assert len(value["instance_graph"]["entities"]) + len(value["instance_graph"]["events"]) == 10
    assert len(value["instance_graph"]["assertions"]) == 7
    for separators in [(",", ":"), (", ", ": ")]:
        assert (
            len(t.encode(json.dumps(value, separators=separators), add_special_tokens=False))
            <= 5120
        )
