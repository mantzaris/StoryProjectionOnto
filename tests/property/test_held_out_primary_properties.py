from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import story_projection_onto.held_out_primary as held_out
from story_projection_onto.held_out_primary import (
    HeldOutCallManifest,
    load_held_out_control_configuration,
)

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies")

ROOT = Path(__file__).resolve().parents[2]


@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=0, max_value=167))
def test_removing_any_registered_call_breaks_exact_inventory(index: int) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    payload = manifest.model_dump(mode="python", exclude={"content_hash"})
    calls = list(payload["calls"])
    calls.pop(index)
    payload["calls"] = calls
    with pytest.raises(ValidationError, match="168 model-call slots"):
        HeldOutCallManifest.model_validate(payload)


@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=96, max_value=167))
def test_fixed_select_cannot_rebind_another_seed_c1(index: int) -> None:
    configuration = load_held_out_control_configuration(ROOT)
    manifest = held_out._derive_call_manifest(ROOT, configuration)
    payload = manifest.model_dump(mode="python", exclude={"content_hash"})
    calls = list(payload["calls"])
    target = dict(calls[index])
    target.pop("content_hash", None)
    target["source_c1_call_id"] = "test-c1-wrong-unit-s1"
    calls[index] = target
    payload["calls"] = calls
    with pytest.raises(ValidationError, match="same-world same-seed"):
        HeldOutCallManifest.model_validate(payload)
