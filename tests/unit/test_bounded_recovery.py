from __future__ import annotations

import copy
import json
from itertools import chain
from pathlib import Path

import pytest

from story_projection_onto import bounded_recovery as recovery
from story_projection_onto import fallback_acceptance as fallback
from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.store import ReadOnlyLedger

ROOT = Path(__file__).resolve().parents[2]


def gate(tmp_path):
    clock = [1000.0]
    return recovery.RecoveryGate(tmp_path / "checkpoint", "a" * 64, clock=lambda: clock[0]), clock


@pytest.mark.parametrize("stage", ["startup", "live_checks", "c1_retry", "validation_drain"])
def test_each_stage_deadline_fails_closed(tmp_path, stage):
    controller, clock = gate(tmp_path)
    stages = ["startup", "live_checks", "c1_retry", "validation_drain"]
    for current in stages[: stages.index(stage) + 1]:
        controller.transition(current)
        if current != stage:
            clock[0] += 1
    clock[0] += recovery.CAPS[stage] - 1.01
    assert controller.deadline_reason() is None
    clock[0] += 0.02
    assert controller.deadline_reason() == f"recovery_{stage}_deadline"
    with pytest.raises(recovery.RecoveryDeadlineExceeded):
        controller.transition(
            "accepted" if stage == "validation_drain" else stages[stages.index(stage) + 1],
            details={"c1_valid": True, "fresh_admission": True},
        )
    controller.begin_shutdown()
    assert controller.records()[-1]["stage"] == "shutdown"


def test_whole_deadline_cannot_reset_by_transition(tmp_path):
    controller, clock = gate(tmp_path)
    for stage, elapsed in (
        ("startup", 298),
        ("live_checks", 118),
        ("c1_retry", 238),
        ("validation_drain", 118),
    ):
        controller.transition(stage)
        clock[0] += elapsed
    assert clock[0] == 1772
    controller.begin_shutdown()
    clock[0] = 1840
    assert controller.deadline_reason() == "recovery_shutdown_deadline"
    assert controller.records()[0]["monotonic"] == 1000


def test_shutdown_has_early_guardian_takeover_margin(tmp_path):
    controller, clock = gate(tmp_path)
    controller.transition("startup")
    controller.begin_shutdown()
    clock[0] += 9
    assert controller.deadline_reason() == "recovery_shutdown_deadline"
    # 51 seconds still remain to complete bounded identity-aware shutdown.
    assert recovery.CAPS["shutdown"] - (clock[0] - 1000) == 51


@pytest.mark.parametrize("details", [None, {}, {"c1_valid": True}, {"fresh_admission": True}])
def test_c1_success_alone_never_releases_recovery(tmp_path, details):
    controller, _ = gate(tmp_path)
    for stage in ("startup", "live_checks", "c1_retry", "validation_drain"):
        controller.transition(stage)
    with pytest.raises(ValueError, match="requires C1"):
        controller.transition("accepted", details=details)


def test_success_releases_only_recovery_envelope_not_shutdown_cap(tmp_path):
    controller, clock = gate(tmp_path)
    for stage in ("startup", "live_checks", "c1_retry", "validation_drain"):
        controller.transition(stage)
    controller.transition("accepted", details={"c1_valid": True, "fresh_admission": True})
    clock[0] += 2000
    assert controller.deadline_reason() is None
    controller.begin_shutdown()
    assert controller.deadline_reason() is None
    clock[0] += 10
    assert controller.deadline_reason() == "recovery_shutdown_deadline"
    controller.transition("stopped")
    assert controller.deadline_reason() is None
    with pytest.raises(ValueError, match="illegal"):
        controller.transition("startup")


def test_stage_tamper_and_boot_reset_fail_closed(tmp_path):
    controller, _ = gate(tmp_path)
    controller.transition("startup")
    path = next(controller.root.glob("*.json"))
    value = json.loads(path.read_bytes())
    value["boot_id"] = "another-boot"
    value["manifest_sha256"] = canonical_sha256(
        {k: v for k, v in value.items() if k != "manifest_sha256"}
    )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identity"):
        controller.deadline_reason()


def test_amendment_and_all_in_forecast_preserve_every_prior_second():
    auth = recovery.load_authorization(
        ROOT / "configs/study/bounded_recovery_v10.json", root=ROOT, run_id=recovery.RUN_ID
    )
    assert auth["original_nine_hour_target_met"] is False
    value = recovery.prelaunch_forecast(ROOT, recovery.PRIOR_ACTUAL)
    assert value["all_in_seconds"] == pytest.approx(33523.58320194846)
    assert value["scheduled_reserve_seconds"] == pytest.approx(136.41679805154)
    assert value["earmark_subtraction_seconds"] == 240
    assert value["recovery_envelope_seconds"] == sum(recovery.CAPS.values()) == 840
    assert (
        sum(
            row["remaining_count"]
            for row in value["mandatory_rows"]
            if row["call_class"] == "gpu_session_start"
        )
        == 5
    )
    assert (
        value["mandatory_rows"]
        == recovery.prior_result(ROOT)["post_fallback_full_manifest_forecast"]["rows"]
    )
    with pytest.raises(RuntimeError, match="admission failed"):
        recovery.prelaunch_forecast(ROOT, recovery.PRIOR_ACTUAL + 137)


def test_historical_event_and_service_rows_and_frozen_science_are_unchanged():
    recovery.validate_frozen_science(ROOT)
    with ReadOnlyLedger(
        ROOT
        / "artifacts/restricted/recovery_validation/v9_terminal_20260906T2313Z"
        / "phase1_acceptance.sqlite"
    ) as ledger:
        recovery.validate_history(ROOT, ledger, before_start=True)


def test_v10_cannot_use_exhausted_v9_authorization():
    options = fallback.parse_arguments(
        [
            "--output",
            "result.json",
            "--run-id",
            recovery.RUN_ID,
            "--controller-stage",
            "orchestrate",
            *chain.from_iterable(
                (
                    [f"--{name}", "placeholder"]
                    for name in (
                        "primary-result",
                        "activation-certificate",
                        "cache-replacement-receipt",
                        "snapshot",
                        "shared-cache",
                        "verified-model-manifest",
                        "source-association",
                        "ledger",
                        "artifact-root",
                        "checkpoint",
                        "quota-root",
                    )
                ),
            ),
        ]
    )
    with pytest.raises(SystemExit, match="bounded-recovery"):
        fallback._require_execution_arguments(options)
    options.bounded_recovery_authorization = ROOT / "configs/study/bounded_recovery_v10.json"
    fallback._require_execution_arguments(options)
    command = fallback._internal_controller_command(
        options, stage="run", output=ROOT / "result.json", guard=ROOT / "guard.json"
    )
    assert "--bounded-recovery-authorization" in command
    changed = copy.deepcopy(options)
    changed.resume_orchestrator = True
    with pytest.raises(SystemExit, match="one fresh start"):
        fallback._require_execution_arguments(changed)


def test_v10_controller_contains_required_validation_and_restart_paths():
    import inspect

    run = inspect.getsource(fallback.FallbackAcceptanceRunner.run)
    assert run.index('transition("c1_retry")') < run.index("self.service.run_fallback_test(")
    assert run.index('transition("validation_drain")') < run.index(
        "validate_acceptance_generation("
    )
    assert run.index("validate_acceptance_generation(") < run.index('"accepted",')
    assert 'call.call_id == "fallback-c1-01"' in run
    assert "resume_from_checkpoint" in run and "stage_one_pid == os.getpid()" in run
    guardian = inspect.getsource(fallback._run_guardian)
    assert "deadline_reason()" in guardian and "_revoke_controller_authority(" in guardian


@pytest.mark.parametrize("failure", ["transport", "schema", "scientific", "none", "admission"])
def test_v10_real_runner_stops_after_failed_c1_without_repair(tmp_path, monkeypatch, failure):
    """Real runner + immutable copied history; only model/service are CPU fixtures."""
    import shutil

    from story_projection_onto.gpu_runtime import RuntimeTransportError
    from story_projection_onto.store import Ledger
    from tests.unit import test_fallback_acceptance as fixtures

    shutil.copy2(
        ROOT
        / "artifacts/restricted/recovery_validation/v9_terminal_20260906T2313Z"
        / "phase1_acceptance.sqlite",
        tmp_path / "ledger.sqlite",
    )
    auth = recovery.load_authorization(
        ROOT / "configs/study/bounded_recovery_v10.json", root=ROOT, run_id=recovery.RUN_ID
    )
    outputs = fixtures._fallback_outputs(trigger_repair=failure == "scientific")
    if failure == "schema":
        outputs["fallback-c1-01"] = {}
    live = {}
    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        configuration = fixtures._fallback_launch_configuration(tmp_path)
        first = fixtures.FakeFallbackService(configuration, ledger, outputs, live)
        second = fixtures.FakeFallbackService(configuration, ledger, outputs, live)

        def configured(service):
            runner = fixtures._runner(tmp_path=tmp_path, ledger=ledger, service=service)
            runner.run_id = recovery.RUN_ID
            runner.bounded_recovery_authorization = auth
            runner.service_start_watchdog_seconds = 300
            # The fixture tokenizer is deliberately not the pinned tokenizer;
            # byte identity is separately checked by the real remote preflight.
            monkeypatch.setattr(
                type(runner), "_validate_second_recovery_request_binding", lambda self: None
            )
            return runner

        prepared = configured(first)
        monkeypatch.setattr(fixtures.os, "getpid", lambda: 41001)
        prepared.prepare_controller_restart()
        executed = configured(second)
        monkeypatch.setattr(fixtures.os, "getpid", lambda: 41002)
        if failure == "transport":

            def failed(*args, **kwargs):
                raise RuntimeTransportError("CPU fixture transport failure")

            monkeypatch.setattr(type(second), "run_fallback_test", failed)
        if failure == "admission":
            second.actual_allocated_service_seconds = 34000
            with pytest.raises(RuntimeError, match="fresh amended admission"):
                executed.run()
            assert second.shutdown_count > 0 and live["running"] is False
            assert not any(r["stage"] == "accepted" for r in executed._recovery_gate().records())
            return
        result = executed.run()
        assert result["completed_base_call_count"] == (4 if failure == "none" else 0)
        assert result["repair_attempt_count"] == 0
        assert result["phase1_gate_passed"] is (failure == "none")
        assert result["vllm_service_stopped"] is True
        assert first.start_count == 1 and second.start_count == 0
        assert second.resume_count == 1
        assert executed._recovery_gate().records()[-1]["stage"] == "stopped"
        assert any(r["stage"] == "accepted" for r in executed._recovery_gate().records()) is (
            failure == "none"
        )
