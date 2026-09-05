from __future__ import annotations

import json
from pathlib import Path

import pytest

from story_projection_onto.phase1_acceptance import (
    _condition_output_schema,
    phase1_acceptance_calls,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def test_exact_failed_c1_surface_compiles_with_installed_xgrammar() -> None:
    """Exercise the CPU-only converter used by the pinned vLLM 0.10.2 backend."""

    xgrammar = pytest.importorskip("xgrammar")
    call = next(item for item in phase1_acceptance_calls() if item.call_id == "c1-01")
    schema = json.loads(
        (ROOT / "schemas/jsonschema/ontology_draft.schema.json").read_text(
            encoding="utf-8"
        )
    )
    fixture = json.loads(
        (ROOT / "tests/fixtures/phase1/c1_pre_request.json").read_text(
            encoding="utf-8"
        )
    )
    compatible = _condition_output_schema(schema, call=call, fixture=fixture)

    # This is intentionally a direct xgrammar call.  Importing vLLM's protocol
    # modules initializes CUDA in the pinned environment and is not CPU-only.
    xgrammar.Grammar.from_json_schema(
        json.dumps(compatible, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def test_exact_fixed_select_surface_compiles_with_installed_xgrammar() -> None:
    """Compile the mechanically restricted schema used by the acceptance gate."""

    xgrammar = pytest.importorskip("xgrammar")
    call = next(item for item in phase1_acceptance_calls() if item.call_id == "fixed-01")
    schema = json.loads(
        (ROOT / "schemas/jsonschema/ontology_draft.schema.json").read_text(
            encoding="utf-8"
        )
    )
    fixture = json.loads(
        (ROOT / "tests/fixtures/phase1/fixed_select_request.json").read_text(
            encoding="utf-8"
        )
    )
    compatible = _condition_output_schema(schema, call=call, fixture=fixture)

    xgrammar.Grammar.from_json_schema(
        json.dumps(compatible, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
