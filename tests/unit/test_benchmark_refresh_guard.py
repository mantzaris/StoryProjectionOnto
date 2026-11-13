from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from story_projection_onto.benchmark_refresh_guard import (
    EXACT_LINEAGE_PROJECTION_RULE,
    LEGACY_PROJECTION_RULE,
    RefreshVerificationError,
    _canonical_sha256,
    _normalize_benchmark_document,
    _validate_model_evidence_against_neutral,
    verify_benchmark_schema_refresh,
    write_refresh_receipt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "synthetic"
SCHEMA_ROOT = PROJECT_ROOT / "schemas" / "jsonschema"


def _canonical_file_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _refresh_content_hashes(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _refresh_content_hashes(child)
        if "content_hash" in value:
            value["content_hash"] = _canonical_sha256(value)
    elif isinstance(value, list):
        for child in value:
            _refresh_content_hashes(child)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_file_bytes(value))


def _update_generated_file_binding(candidate_root: Path, relative_path: str) -> None:
    target = candidate_root / relative_path
    manifest_path = candidate_root / "manifests" / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item for item in manifest["generated_files"] if item["relative_path"] == relative_path
    )
    entry["byte_count"] = target.stat().st_size
    entry["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    _refresh_content_hashes(entry)
    _refresh_content_hashes(manifest)
    _write_json(manifest_path, manifest)


class BenchmarkRefreshGuardTests(unittest.TestCase):
    def test_identity_tree_produces_self_hashed_replay_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_benchmark = root / "candidate-benchmark"
            replay_benchmark = root / "replay-benchmark"
            candidate_schema = root / "candidate-schema"
            replay_schema = root / "replay-schema"
            shutil.copytree(BENCHMARK_ROOT, candidate_benchmark)
            shutil.copytree(BENCHMARK_ROOT, replay_benchmark)
            shutil.copytree(SCHEMA_ROOT, candidate_schema)
            shutil.copytree(SCHEMA_ROOT, replay_schema)

            receipt = verify_benchmark_schema_refresh(
                old_benchmark_root=BENCHMARK_ROOT,
                candidate_benchmark_root=candidate_benchmark,
                old_schema_root=SCHEMA_ROOT,
                candidate_schema_root=candidate_schema,
                candidate_benchmark_replay_root=replay_benchmark,
                candidate_schema_replay_root=replay_schema,
                candidate_projection_rule=EXACT_LINEAGE_PROJECTION_RULE,
            )

            self.assertTrue(receipt.candidate_replay_checked)
            self.assertEqual(receipt.benchmark_file_count, 259)
            self.assertEqual(receipt.review_projection_count, 9)
            self.assertEqual(receipt.review_item_count, 72)
            self.assertEqual(receipt.differences, ())
            receipt.verify_self_hash()
            output = root / "receipt.json"
            write_refresh_receipt(receipt, output)
            first = output.read_bytes()
            write_refresh_receipt(receipt, output)
            self.assertEqual(output.read_bytes(), first)

    def test_registered_scorer_world_must_remain_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_benchmark = root / "candidate-benchmark"
            candidate_schema = root / "candidate-schema"
            shutil.copytree(BENCHMARK_ROOT, candidate_benchmark)
            shutil.copytree(SCHEMA_ROOT, candidate_schema)
            scorer_path = candidate_benchmark / "scorer_only/development/syn-dev-01.json"
            scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
            scorer["scorer_namespace"] = "changed"
            _refresh_content_hashes(scorer)
            _write_json(scorer_path, scorer)
            _update_generated_file_binding(
                candidate_benchmark, "scorer_only/development/syn-dev-01.json"
            )

            with self.assertRaisesRegex(
                RefreshVerificationError, "scientific artifact is not byte-identical"
            ):
                verify_benchmark_schema_refresh(
                    old_benchmark_root=BENCHMARK_ROOT,
                    candidate_benchmark_root=candidate_benchmark,
                    old_schema_root=SCHEMA_ROOT,
                    candidate_schema_root=candidate_schema,
                    candidate_projection_rule=LEGACY_PROJECTION_RULE,
                )

    def test_query_semantics_fail_outside_exact_lineage_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_benchmark = root / "candidate-benchmark"
            candidate_schema = root / "candidate-schema"
            shutil.copytree(BENCHMARK_ROOT, candidate_benchmark)
            shutil.copytree(SCHEMA_ROOT, candidate_schema)
            relative = (
                "model_visible/query_stages/reveal_1ed7999f54103553f109/query.json"
            )
            query_path = candidate_benchmark / relative
            query = json.loads(query_path.read_text(encoding="utf-8"))
            query["query"]["wording"] = "A scientifically different question"
            _refresh_content_hashes(query)
            _write_json(query_path, query)
            _update_generated_file_binding(candidate_benchmark, relative)

            with self.assertRaisesRegex(
                RefreshVerificationError, "changed outside the evidence-lineage allowlist"
            ):
                verify_benchmark_schema_refresh(
                    old_benchmark_root=BENCHMARK_ROOT,
                    candidate_benchmark_root=candidate_benchmark,
                    old_schema_root=SCHEMA_ROOT,
                    candidate_schema_root=candidate_schema,
                    candidate_projection_rule=LEGACY_PROJECTION_RULE,
                )

    def test_modern_model_evidence_must_exactly_project_neutral_lineage(self) -> None:
        text = "Rendered neutral passage."
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        neutral_record = {
            "schema_version": "1.0.0",
            "content_hash": "ignored",
            "evidence_id": "ev_1",
            "passage_id": "psg_1",
            "text": text,
            "text_hash": text_hash,
            "discourse_position": {"passage_order": 0},
            "mention_candidates": [],
            "event_candidates": [],
            "relation_phrase_candidates": [],
            "temporal_clues": [],
            "provenance": {
                "provenance_id": "prv_1",
                "evidence_id": "ev_1",
                "extraction_method": "synthetic",
                "locator": "synthetic:1",
                "source_artifact_hash": text_hash,
                "confidence": 1.0,
            },
            "confidence": 1.0,
            "release_class": "public",
        }
        model_record = {
            key: value
            for key, value in neutral_record.items()
            if key not in {"release_class", "content_hash"}
        }
        neutral = {"evidence": [neutral_record]}
        model = {"evidence": [model_record]}
        _validate_model_evidence_against_neutral(
            model, neutral, exact_lineage=True, label="test"
        )
        model["evidence"][0]["text_hash"] = "0" * 64
        with self.assertRaisesRegex(RefreshVerificationError, "exact neutral-evidence projection"):
            _validate_model_evidence_against_neutral(
                model, neutral, exact_lineage=True, label="test"
            )

    def test_neutral_normalizer_does_not_hide_locator_changes(self) -> None:
        old = {
            "content_hash": "a" * 64,
            "evidence": [
                {
                    "text": "same",
                    "provenance": {
                        "locator": "synthetic:one",
                        "source_artifact_hash": None,
                    },
                }
            ],
        }
        candidate = json.loads(json.dumps(old))
        candidate["evidence"][0]["provenance"]["source_artifact_hash"] = "b" * 64
        self.assertEqual(
            _normalize_benchmark_document(
                "condition_inputs/neutral_evidence/neutral_x/neutral_evidence.json", old
            ),
            _normalize_benchmark_document(
                "condition_inputs/neutral_evidence/neutral_x/neutral_evidence.json", candidate
            ),
        )
        candidate["evidence"][0]["provenance"]["locator"] = "synthetic:two"
        self.assertNotEqual(
            _normalize_benchmark_document(
                "condition_inputs/neutral_evidence/neutral_x/neutral_evidence.json", old
            ),
            _normalize_benchmark_document(
                "condition_inputs/neutral_evidence/neutral_x/neutral_evidence.json", candidate
            ),
        )


if __name__ == "__main__":
    unittest.main()
