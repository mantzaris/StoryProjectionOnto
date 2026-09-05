from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import story_projection_onto.development_continuation as continuation_module
import story_projection_onto.development_execution as execution_module
from story_projection_onto.contracts import ConditionName
from story_projection_onto.development_continuation import (
    ProductionDevelopmentContinuationAdopter,
)
from story_projection_onto.development_execution import ProductionDevelopmentCallExecutor
from story_projection_onto.development_runtime import (
    DEVELOPMENT_UNIT_IDS,
    DevelopmentCallKind,
)

ROOT = Path(__file__).resolve().parents[2]


class _TickingClock:
    def __init__(self, start: datetime) -> None:
        self.value = start

    def __call__(self) -> datetime:
        self.value += timedelta(milliseconds=1)
        return self.value


def test_fresh_development_query_processing_starts_after_physical_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed_at = datetime(2035, 1, 1, tzinfo=UTC)
    access = SimpleNamespace(accessed_at=accessed_at)
    opening = SimpleNamespace(access_event=access, context=object())
    materialized = SimpleNamespace(packet=object(), event=object())
    preparation = object()
    snapshot = object()
    upper_ontology = object()
    config = object()
    stage = SimpleNamespace(staging_manifest_hash="stage-hash")
    call = SimpleNamespace(
        query_stage=stage,
        kind=DevelopmentCallKind.C2_CONSTRUCTION,
        unit_id="dev-unit-01",
        condition=ConditionName.C2_LLM_QUERY,
    )
    envelope = SimpleNamespace(
        query_access_event=access,
        prequery_barrier_hash="barrier-hash",
    )
    adapter = SimpleNamespace(
        audited_opening=lambda _stage: opening,
        persisted_prequery_barrier=lambda _hash: object(),
    )
    repository = SimpleNamespace(
        preparations={(call.unit_id, call.condition): preparation},
        neutral_by_unit={call.unit_id: SimpleNamespace(snapshot=snapshot)},
        construction=SimpleNamespace(upper_ontology=upper_ontology),
    )
    captured: dict[str, object] = {}
    produced = object()

    def capture_inputs(**kwargs: object) -> object:
        captured.update(kwargs)
        return produced

    monkeypatch.setattr(execution_module, "ProduceInputs", capture_inputs)
    monkeypatch.setattr(
        ProductionDevelopmentCallExecutor,
        "_query_materialization",
        lambda _self, _adapter, _call: materialized,
    )
    executor = ProductionDevelopmentCallExecutor(
        repository=repository,
        clock=_TickingClock(accessed_at),
    )

    inputs, returned_materialization = executor._produce_inputs(
        adapter, call, envelope, config
    )

    assert inputs is produced
    assert returned_materialization is materialized
    assert captured["query_processing_started_at"] > accessed_at


def test_continuation_cpu_projection_starts_after_physical_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed_at = datetime(2035, 1, 1, tzinfo=UTC)
    stage = SimpleNamespace(staging_manifest_hash="stage-hash")
    call = SimpleNamespace(
        query_stage=stage,
        kind=DevelopmentCallKind.C2_CONSTRUCTION,
        unit_id=DEVELOPMENT_UNIT_IDS[0],
    )
    opening = SimpleNamespace(
        access_event=SimpleNamespace(accessed_at=accessed_at),
        context=object(),
    )
    materialized = SimpleNamespace(packet=object(), event=object())
    repository = SimpleNamespace(
        manifest=SimpleNamespace(calls=(call,)),
        materializations={stage.staging_manifest_hash: materialized},
        preparations={
            (DEVELOPMENT_UNIT_IDS[0], ConditionName.C0_CLASSICAL_PRE): object()
        },
        neutral_by_unit={
            DEVELOPMENT_UNIT_IDS[0]: SimpleNamespace(snapshot=object())
        },
        construction=SimpleNamespace(upper_ontology=object()),
    )
    adapter = SimpleNamespace(
        audited_opening=lambda _stage: opening,
        persisted_prequery_barrier=lambda _hash: object(),
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        '{"prequery_barrier":{"content_hash":"barrier-hash"}}',
        encoding="utf-8",
    )
    adopter = ProductionDevelopmentContinuationAdopter(
        root=ROOT,
        service=object(),
        artifacts=object(),
        tokenizer=object(),
        tokenizer_manifest=object(),
        launcher_configuration_hash="launcher",
        model_snapshot_manifest_hash="model",
        source_revision="test",
        checkpoint_path=checkpoint_path,
        adapter_state_path=tmp_path / "adapter-state.json",
        preparation_pointer_path=tmp_path / "preparations.json",
        assessment_manifest_path=tmp_path / "assessment.json",
        assessment_factory=lambda *_args, **_kwargs: object(),
        clock=_TickingClock(accessed_at),
    )
    adopter._repository = repository
    adopter._adapter = adapter
    adopter._classical_builder = object()
    captured: dict[str, object] = {}

    class _CapturedInputs(RuntimeError):
        pass

    def capture_inputs(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _CapturedInputs

    monkeypatch.setattr(continuation_module, "ProduceInputs", capture_inputs)
    monkeypatch.setattr(
        ProductionDevelopmentContinuationAdopter,
        "_cpu_run_config",
        lambda _self, **_kwargs: object(),
    )

    with pytest.raises(_CapturedInputs):
        adopter._cpu_projection_receipts()

    assert captured["query_processing_started_at"] > accessed_at
