from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from story_projection_onto.contracts import ConditionName, RunOutcome
from story_projection_onto.development_artifacts import (
    DevelopmentCallAuditReceipt,
    DevelopmentCPUProjectionReceipt,
    LogicalCASReference,
    OpaqueJSONReference,
)
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.scorer_only.development_assessment import (
    _CORE_RUNTIME_SOURCE_PATHS,
    DevelopmentAssessmentInputManifest,
    DevelopmentAssessmentIntegrityError,
    DevelopmentScientificAssessmentProvider,
    RuntimeSourceBinding,
    _assert_gold_free_payload,
    _CASReader,
    _context_independent_assertion_slot,
    _sha256_file,
    _stage_objects,
    _verify_runtime_sources,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    Ledger,
    ReleaseClass,
)

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def digest(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def reference(value: object) -> LogicalCASReference:
    return LogicalCASReference(
        logical_content_hash=digest(("logical", value)),
        artifact_hash=digest(("artifact", value)),
        object_kind="test_record",
    )


def opaque_reference(value: object) -> OpaqueJSONReference:
    return OpaqueJSONReference(
        logical_content_hash=digest(("logical", value)),
        artifact_hash=digest(("artifact", value)),
        object_kind="runtime_source_manifest",
    )


def runtime_bindings() -> tuple[RuntimeSourceBinding, ...]:
    core = tuple(
        RuntimeSourceBinding(
            relative_path=path,
            sha256=digest(path),
            role="core_runtime",
        )
        for path in sorted(_CORE_RUNTIME_SOURCE_PATHS)
    )
    return (
        *core,
        RuntimeSourceBinding(
            relative_path="src/story_projection_onto/development_adapter.py",
            sha256=digest("adapter"),
            role="development_service_adapter",
        ),
    )


def assessment_manifest() -> DevelopmentAssessmentInputManifest:
    return DevelopmentAssessmentInputManifest(
        call_manifest_hash=digest("calls"),
        prequery_inputs_hash=digest("prequery"),
        source_tree_hash=digest("tree"),
        source_revision="test-revision",
        benchmark_manifest_file_sha256=digest("benchmark"),
        assessment_bundle_artifact_hash=digest("assessment-bundle"),
        runtime_source_manifest=opaque_reference("source-manifest"),
        prequery_preparation_artifacts=tuple(reference(("prep", index)) for index in range(11)),
        call_receipt_artifact_hashes=tuple(
            digest(("call-receipt", index)) for index in range(24)
        ),
        service_result_artifact_hashes=tuple(
            digest(("service-result", index)) for index in range(24)
        ),
        cpu_projection_receipt_artifact_hashes=tuple(
            digest(("cpu", index)) for index in range(24)
        ),
        packing_preflight_artifact_hash=digest("packing-preflight"),
        runtime_sources=runtime_bindings(),
    )


def test_assessment_manifest_requires_every_immutable_output() -> None:
    payload = assessment_manifest().model_dump(mode="python", exclude={"content_hash"})
    payload["service_result_artifact_hashes"] = payload[
        "service_result_artifact_hashes"
    ][:-1]
    with pytest.raises(ValueError, match="24 service-result"):
        DevelopmentAssessmentInputManifest.model_validate(payload)

    payload = assessment_manifest().model_dump(mode="python", exclude={"content_hash"})
    payload["cpu_projection_receipt_artifact_hashes"] = payload[
        "cpu_projection_receipt_artifact_hashes"
    ][:-1]
    with pytest.raises(ValueError, match="12 C0 and 12 C1"):
        DevelopmentAssessmentInputManifest.model_validate(payload)

    payload = assessment_manifest().model_dump(mode="python", exclude={"content_hash"})
    payload["call_receipt_artifact_hashes"] = payload[
        "call_receipt_artifact_hashes"
    ][:-1]
    with pytest.raises(ValueError, match="24 call-receipt"):
        DevelopmentAssessmentInputManifest.model_validate(payload)


def test_call_receipt_rejects_validated_failure() -> None:
    with pytest.raises(ValueError, match=r"nonsuccessful.*validated"):
        DevelopmentCallAuditReceipt(
            receipt_id="receipt-failed",
            ordinal=1,
            call_id="call-failed",
            condition=ConditionName.C1_LLM_PRE,
            outcome=RunOutcome.FAILED,
            request_started=False,
            service_identity=reference("identity"),
            validated_generation=reference("generation"),
            created_at=NOW,
        )


def test_cpu_receipt_requires_unique_validation_rows() -> None:
    with pytest.raises(ValueError, match="unique"):
        DevelopmentCPUProjectionReceipt(
            receipt_id="cpu-receipt",
            condition=ConditionName.C0_CLASSICAL_PRE,
            unit_id="dev-unit-01",
            query_ordinal=1,
            job_id="job-1",
            attempt_id="attempt-1",
            ledger_projection_id="projection-ledger-1",
            ledger_validation_ids=("validation-1", "validation-1"),
            condition_attempt=reference("attempt"),
            projection=reference("projection"),
            run_condition_config=reference("config"),
            comparison_input_manifest=reference("comparison"),
            evidence_packet=reference("packet"),
            packet_materialization_event_hash=digest("materialization"),
            query_context=reference("context"),
            created_at=NOW,
        )


def test_cas_reader_checks_both_artifact_and_logical_hashes(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    try:
        blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
        artifacts = ArtifactStore(blobs, ledger)
        value = reference("persisted-value")
        artifact = artifacts.put_bytes(
            (value.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.test-record+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=NOW,
        )
        stored = LogicalCASReference(
            logical_content_hash=value.content_hash,
            artifact_hash=artifact.content_hash,
            object_kind="test_record",
        )
        reader = _CASReader(ledger, blobs)
        assert reader.logical(stored, LogicalCASReference) == value

        changed = stored.model_copy(
            update={"logical_content_hash": digest("wrong-logical-hash")}
        )
        with pytest.raises(DevelopmentAssessmentIntegrityError, match="logical hash"):
            reader.logical(changed, LogicalCASReference)
    finally:
        ledger.close()


def test_runtime_source_audit_rejects_tamper_and_scorer_import(tmp_path: Path) -> None:
    source = tmp_path / "src/story_projection_onto/adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text("from story_projection_onto.contracts import ConditionName\n")
    binding = RuntimeSourceBinding(
        relative_path="src/story_projection_onto/adapter.py",
        sha256=_sha256_file(source),
        role="development_service_adapter",
    )
    assert _verify_runtime_sources(tmp_path, (binding,))

    source.write_text("from story_projection_onto.scorer_only import hidden\n")
    with pytest.raises(DevelopmentAssessmentIntegrityError, match="binding changed"):
        _verify_runtime_sources(tmp_path, (binding,))

    rebound = binding.model_copy(update={"sha256": _sha256_file(source)})
    with pytest.raises(DevelopmentAssessmentIntegrityError, match="imports scorer"):
        _verify_runtime_sources(tmp_path, (rebound,))


@pytest.mark.parametrize(
    "payload",
    [
        {"gold_projection": {"answer": "hidden"}},
        {"innocent": "load scorer_only/development/syn-dev-01.json"},
    ],
)
def test_model_payload_gold_firewall_rejects_keys_and_values(payload: object) -> None:
    with pytest.raises(DevelopmentAssessmentIntegrityError, match="scorer"):
        _assert_gold_free_payload(payload)


def test_stage_audit_uses_embedded_semantic_hash_and_detects_content_tamper(
    tmp_path: Path,
) -> None:
    call = load_development_call_manifest(ROOT).calls[0]
    staged, reveal = _stage_objects(ROOT, call, query_revealed=False)
    evidence_path = ROOT / call.prequery_stage.relative_path / "evidence.json"
    assert reveal is None
    assert staged.content_hash == call.prequery_stage.evidence_artifact_hash
    assert _sha256_file(evidence_path) != staged.content_hash

    copied = tmp_path / call.prequery_stage.relative_path
    shutil.copytree(evidence_path.parent, copied)
    payload = json.loads((copied / "evidence.json").read_text(encoding="utf-8"))
    payload["evidence"][0]["text"] += " altered"
    (copied / "evidence.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevelopmentAssessmentIntegrityError, match="invalid"):
        _stage_objects(tmp_path, call, query_revealed=False)


def test_union_gold_slot_is_context_independent_and_fails_closed() -> None:
    left = "score.ctx_a.assertion.fact.007"
    right = "score.ctx_b.assertion.fact.007"
    assert _context_independent_assertion_slot(left) == "fact.007"
    assert _context_independent_assertion_slot(right) == "fact.007"
    with pytest.raises(DevelopmentAssessmentIntegrityError, match="stable semantic slot"):
        _context_independent_assertion_slot("unregistered-id")


def test_provider_does_not_read_gold_before_exact_24_rows(tmp_path: Path) -> None:
    provider = DevelopmentScientificAssessmentProvider(
        root=ROOT,
        ledger=cast(Any, None),
        blobs=cast(Any, None),
        prequery_inputs=cast(Any, None),
        assessment_manifest_path=tmp_path / "must-not-be-read.json",
        assessment_manifest_file_sha256=digest("absent"),
    )
    with pytest.raises(DevelopmentAssessmentIntegrityError, match="exact 24"):
        provider(load_development_call_manifest(ROOT), ())
