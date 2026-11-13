from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from story_projection_onto.contracts import OntologyDraft

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "phase1"
MODEL_MANIFEST = ROOT / "artifacts" / "public" / "manifests" / "model_snapshot.json"
PINNED_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
PINNED_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"
PINNED_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
EXPECTED_EXACT_COUNTS = {
    "c1_pre_output.json": 1639,
    "c1_pre_output_2.json": 1798,
    "c2_query_output.json": 1771,
    "c2_query_output_2.json": 1701,
    "fixed_select_output.json": 1513,
    "invalid_repair_case.json#base_draft": 694,
    "invalid_repair_case.json#corrected_draft": 744,
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def receipt_payload(name: str) -> object:
    file_name, separator, selector = name.partition("#")
    value = load_json(FIXTURES / file_name)
    if separator:
        return value[selector]
    return value


def test_exact_pinned_tokenizer_receipt_binds_every_compact_output_below_cap() -> None:
    receipt = load_json(FIXTURES / "output_token_caps.json")
    model_manifest = load_json(MODEL_MANIFEST)
    tokenizer_record = next(
        item for item in model_manifest["files"] if item["path"] == "tokenizer.json"
    )

    assert receipt["tokenizer_repository"] == model_manifest["repository"] == PINNED_REPOSITORY
    assert receipt["tokenizer_revision"] == model_manifest["revision"] == PINNED_REVISION
    assert receipt["tokenizer_json_sha256"] == tokenizer_record["sha256"] == PINNED_TOKENIZER_SHA256
    tokenizer_manifest = receipt["tokenizer_manifest"]
    assert tokenizer_manifest["repository"] == PINNED_REPOSITORY
    assert tokenizer_manifest["revision"] == PINNED_REVISION
    assert tokenizer_manifest["local_files_only"] is True
    assert tokenizer_manifest["trust_remote_code"] is False
    assert tokenizer_manifest["enable_thinking"] is False
    assert receipt["add_special_tokens"] is False
    assert receipt["model_weights_loaded"] is False
    assert receipt["gpu_used"] is False
    assert receipt["outputs"].keys() == EXPECTED_EXACT_COUNTS.keys()

    for name, expected_count in EXPECTED_EXACT_COUNTS.items():
        value = receipt_payload(name)
        draft = OntologyDraft.model_validate(value)
        compact = compact_json(value).encode("utf-8")
        record = receipt["outputs"][name]
        assert draft.budget_accounting.input_tokens == 0
        assert draft.budget_accounting.output_tokens == 0
        assert len(compact) == record["compact_json_bytes"]
        assert hashlib.sha256(compact).hexdigest() == record["compact_json_sha256"]
        assert record["exact_token_count"] == expected_count
        assert expected_count <= record["maximum_output_tokens"]
        assert record["maximum_output_tokens"] == (
            1536 if name.startswith("invalid_repair_case.json#") else 2048
        )

    for request_name in (
        "c1_pre_request.json",
        "c2_query_request.json",
        "fixed_select_request.json",
    ):
        request = load_json(FIXTURES / request_name)
        assert request["runtime"]["tokenizer_hash"] == tokenizer_manifest["manifest_sha256"]


def test_pinned_tokenizer_recounts_receipt_when_snapshot_is_explicitly_available() -> None:
    """Independent live recount; the ordinary CPU suite never downloads a tokenizer."""

    snapshot_value = os.environ.get("STORY_PROJECTION_TOKENIZER_SNAPSHOT")
    if snapshot_value is None:
        pytest.skip("set STORY_PROJECTION_TOKENIZER_SNAPSHOT to the pinned local snapshot")

    snapshot = Path(snapshot_value)
    tokenizer_path = snapshot / "tokenizer.json"
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == PINNED_TOKENIZER_SHA256

    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=True,
    )
    receipt = load_json(FIXTURES / "output_token_caps.json")
    for name, record in receipt["outputs"].items():
        exact_count = len(
            tokenizer.encode(compact_json(receipt_payload(name)), add_special_tokens=False)
        )
        assert exact_count == record["exact_token_count"]
