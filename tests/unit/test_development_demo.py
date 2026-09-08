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
    assert q.rendered_input_token_count + 4096 <= 12288
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
    assert repair.rendered_input_token_count + 3072 <= 12288
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
    [(6716.108080, 0, 0, 1), (6716.108081, 2, 0, 1), (10300, 0, 0, 1), (6716.108081, 0, 12, 1)],
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


def test_controller_scientific_failure_does_not_block_c2_or_shutdown(tmp_path, monkeypatch, pinned):
    """Run the actual workload/queue with simulated transport; no model service."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import scripts.run_capacity_diagnostics as controller
    import story_projection_onto.development_demo_execution as workload
    from story_projection_onto.gpu_runtime import GenerationResult
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
    )
    service.meter.actual_allocated_gpu_seconds = 6716.108081
    service.actual_allocated_service_seconds = 6716.108081
    service.generate.side_effect = lambda q, **kwargs: GenerationResult(
        request_id=q.request_id,
        request_hash=q.request_hash,
        response_sha256="b" * 64,
        parsed_object={"broken": True},
        raw_response=b"{}",
        prompt_tokens=q.rendered_input_token_count,
        completion_tokens=8,
        finish_reason="stop",
    )
    monkeypatch.setattr(controller, "setup", lambda *_: (ledger, sampler, service))
    monkeypatch.setattr(controller, "source_binding", lambda *_: {"cpu": "control"})
    monkeypatch.setattr(workload, "capture_tokenizer_manifest", lambda *a, **k: m)
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
    assert [x["kind"] for x in outcomes] == ["c1", "c1", "c2-q1", "c2-q1", "c2-q2", "c2-q2"]
    assert all(not x["scientific_accepted"] for x in outcomes)
    assert all(x["repair_parent"] is not None for x in outcomes[1::2])
    service.shutdown.assert_called_once()
    assert len(list(block.glob("attempt-*.json"))) == 6


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
        '{"instance_graph":{"entities":[{"entity_id":"nE1","label":"actual complete fragment"},{"entity_id":"nE2","label":"cut'
    )
    assert len(fragment["instance_graph"]["entities"]) == 1
    assert fragment["partial_records_only"]
