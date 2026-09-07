import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "capacity_diagnostics", ROOT / "scripts/run_capacity_diagnostics.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_stage_caps_preserve_whole_deadline_and_shutdown():
    deadline, whole = driver.stage_deadline(
        started=100, now_monotonic=200, prior_block_seconds=400, stage_seconds=300
    )
    assert deadline == 500
    assert whole == 899
    deadline, whole = driver.stage_deadline(
        started=100, now_monotonic=800, prior_block_seconds=400, stage_seconds=300
    )
    assert deadline == whole - 60 == 839
    with pytest.raises(TimeoutError):
        driver.stage_deadline(
            started=100, now_monotonic=840, prior_block_seconds=400, stage_seconds=240
        )


def test_reservations_survive_new_run_name(tmp_path):
    driver.immutable(tmp_path / "start-01.json", {"run": "old"})
    driver.immutable(tmp_path / "attempt-01.json", {"run": "old"})
    assert driver.count_reservations(tmp_path, "start") == 1
    assert driver.count_reservations(tmp_path, "attempt") == 1
    with pytest.raises(FileExistsError):
        driver.immutable(tmp_path / "start-01.json", {"run": "new"})


def test_controller_is_bound_to_existing_block_not_new_authority():
    assert driver.BLOCK_ID == "output-capacity-recovery-v1"
    assert driver.BASELINE_SECONDS == 3227.826324
    assert driver.BLOCK_SECONDS == 1200


def test_guardian_kills_owned_controller_and_adopts_only_for_cleanup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = driver.read(ROOT / "configs/study/output_capacity_recovery.json")
    config_path = tmp_path / "configs/study/output_capacity_recovery.json"
    config_path.parent.mkdir(parents=True)
    driver.immutable(config_path, config)
    (tmp_path / "artifacts/restricted").mkdir(parents=True)
    observed = []
    service = SimpleNamespace(
        configuration=SimpleNamespace(configuration_hash="configuration"),
        meter=SimpleNamespace(actual_allocated_gpu_seconds=driver.BASELINE_SECONDS),
        resume_live_service_lease=lambda **kw: observed.append(("adopt", kw)) or True,
        shutdown=lambda **kw: observed.append(("shutdown", kw)),
    )
    ledger = SimpleNamespace(
        unresolved_gpu_allocations=lambda: [], unresolved_gpu_service_journals=lambda: []
    )
    monkeypatch.setattr(driver, "setup", lambda *a: (ledger, None, service))
    monkeypatch.setattr(driver, "source_binding", lambda *a: {})

    class Child:
        pid = 123
        returncode = None

        def __init__(self, command, **kwargs):
            run = Path(command[3])
            binding = driver.read(run / "binding.json")
            driver.immutable(
                run / "state.json",
                {
                    "pid": self.pid,
                    "binding_hash": driver.canonical_sha256(binding),
                    "deadline_monotonic": -1,
                    "stage": "generation",
                    "session": "capacity-start-1",
                    "event": "capacity-start-1-load",
                },
            )

        def poll(self):
            return self.returncode

        def kill(self):
            observed.append(("kill", self.pid))
            self.returncode = -9

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(driver.subprocess, "Popen", Child)
    driver.guardian(tmp_path)
    assert [item[0] for item in observed] == ["kill", "adopt", "shutdown"]
    assert observed[1][1]["cleanup_only"] is True
    assert observed[1][1]["expected_event_id"] == "capacity-start-1-load"


def test_actual_fixed_requires_accepted_source_and_is_exact():
    # Test the canonical seal construction independently of a GPU/tokenizer.
    from types import SimpleNamespace

    from story_projection_onto.conditions.base import sealed_semantic_ids
    from story_projection_onto.contracts import OntologyDraft
    from story_projection_onto.llm import sealed_inventory_from_fixed_ontology

    raw = driver.read(ROOT / "tests/fixtures/phase1/c1_pre_output.json")
    draft = OntologyDraft.model_validate(raw)
    import story_projection_onto.phase1_acceptance as acceptance

    call = next(c for c in acceptance.phase1_acceptance_calls() if c.call_id == "fixed-01")
    wrapper = SimpleNamespace(request_fixture=call.request_fixture, acceptance_call=lambda: call)
    base = SimpleNamespace(messages=[None, SimpleNamespace(content="{}")])
    original = driver.pack_capacity_candidate
    try:
        driver.pack_capacity_candidate = lambda *a, **kw: kw
        seal = driver.seal_actual_c1(ROOT, draft)
        packed, fixture = driver.actual_fixed(ROOT, wrapper, base, draft, None, seal=seal)
    finally:
        driver.pack_capacity_candidate = original
    assert fixture.fixed_ontology.instance_graph == draft.instance_graph
    assert fixture.requested_at >= seal.sealed_at
    assert set(fixture.fixed_ontology.construction_seal.sealed_object_ids) == set(
        sealed_semantic_ids(draft)
    )
    sealed_inventory_from_fixed_ontology(fixture.fixed_ontology, seed_block=0, source_draft=draft)
    assert packed["sections_override"]["sealed_ontology"][
        "instance_graph"
    ] == draft.instance_graph.model_dump(mode="json")
