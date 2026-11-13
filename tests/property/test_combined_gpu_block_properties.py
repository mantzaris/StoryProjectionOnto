from __future__ import annotations

import pytest
from pydantic import ValidationError

from story_projection_onto.combined_gpu_block import (
    CombinedBlockError,
    CombinedCallClass,
    CombinedCallManifest,
    ConfigurationDelta,
    OneSwitchFingerprint,
    assert_one_switch_only,
)
from tests.unit.test_combined_gpu_block import _fingerprint, combined_fixture, digest

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = pytest.importorskip("hypothesis.strategies")


@pytest.mark.property
@settings(max_examples=16, deadline=None)
@given(st.integers(min_value=0, max_value=48))
def test_removing_any_combined_call_breaks_exact_49_inventory(index: int) -> None:
    *_, manifest = combined_fixture()
    payload = manifest.model_dump(mode="python", exclude={"content_hash"})
    calls = list(payload["calls"])
    calls.pop(index)
    payload["calls"] = calls
    with pytest.raises(ValidationError, match="49 base calls"):
        CombinedCallManifest.model_validate(payload)


@pytest.mark.property
@settings(max_examples=16, deadline=None)
@given(
    st.sampled_from(
        [
            "model_manifest_hash",
            "tokenizer_hash",
            "packet_hash",
            "ordered_evidence_hash",
            "horizon_hash",
            "upper_ontology_hash",
            "budgets_hash",
            "seed_manifest_hash",
            "decoding_family_hash",
            "repair_policy_hash",
            "validator_hash",
        ]
    )
)
def test_no_context_one_switch_rejects_every_extra_constant_change(field: str) -> None:
    baseline = _fingerprint()
    payload = baseline.model_dump(mode="python", exclude={"content_hash"})
    payload["context_payload_mode"] = "generic"
    payload[field] = digest(f"mutated-{field}")
    ablated = OneSwitchFingerprint.model_validate(payload)
    delta = ConfigurationDelta(
        switch_name="context_payload_mode",
        baseline_value="structured",
        ablated_value="generic",
    )
    with pytest.raises(CombinedBlockError, match="one-switch"):
        assert_one_switch_only(baseline, ablated, delta)


@pytest.mark.property
@settings(max_examples=12, deadline=None)
@given(st.integers(min_value=21, max_value=32))
def test_each_no_context_payload_omits_all_structured_query_fields(index: int) -> None:
    *_, manifest = combined_fixture()
    call = manifest.calls[index]
    assert call.call_class is CombinedCallClass.ABLATION_NO_CONTEXT
    payload = call.model_visible_context().model_dump(mode="json")
    assert not {
        "wording",
        "lens",
        "target",
        "story_scope",
        "spoiler_horizon",
        "viewpoint",
        "abstraction",
    }.intersection(payload)
