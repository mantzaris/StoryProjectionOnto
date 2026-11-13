from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    OntologyDraft,
    RunOutcome,
)
from story_projection_onto.development_artifacts import (
    DevelopmentCallAuditReceipt,
    DevelopmentCPUProjectionReceipt,
    LogicalCASReference,
    OpaqueJSONReference,
)
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.metrics.alignment import (
    AlignmentPlan,
    AnchorKind,
    AssertionAlignmentTarget,
    GroundingStatus,
    NodeAlignmentTarget,
    NodeKind,
    PermissibleAssertionAlternative,
    prediction_records_from_components,
    score_alignment,
)
from story_projection_onto.metrics.common import revalidated_copy
from story_projection_onto.scorer_only.development_assessment import (
    _CORE_RUNTIME_SOURCE_PATHS,
    DevelopmentAssessmentInputManifest,
    DevelopmentAssessmentIntegrityError,
    DevelopmentScientificAssessmentProvider,
    RuntimeSourceBinding,
    _assert_gold_free_payload,
    _CASReader,
    _context_independent_assertion_slot,
    _prediction_bundle,
    _sha256_file,
    _stage_objects,
    _validate_ledger_validations,
    _verify_runtime_sources,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    Ledger,
    ReleaseClass,
)
from story_projection_onto.store import (
    CommitmentCheckStatus as LedgerCommitmentCheckStatus,
)
from story_projection_onto.store import (
    EvidenceSupportStatus as LedgerEvidenceSupportStatus,
)
from story_projection_onto.store import (
    SemanticAssessmentScope as LedgerSemanticAssessmentScope,
)
from story_projection_onto.store import (
    TemporalValidationStatus as LedgerTemporalValidationStatus,
)
from story_projection_onto.store import ValidationStatus as LedgerValidationStatus
from story_projection_onto.validate import validate_draft_structure

ROOT = Path(__file__).resolve().parents[2]


def test_development_scorer_resolves_anonymized_evidence_without_gold_id_leak():
    from story_projection_onto.development_continuation import load_development_prequery_evidence
    from story_projection_onto.scorer_only.development_assessment import (
        resolve_development_world_id,
    )

    manifest = load_development_call_manifest(ROOT)
    neutral, visible, _ = load_development_prequery_evidence(ROOT, manifest)
    resolved = []
    for unit, artifact in neutral.items():
        world_id = resolve_development_world_id(
            ROOT, visible[unit].content_hash, artifact.content_hash
        )
        assert world_id != artifact.snapshot.world_or_window_id
        resolved.append(world_id)
    assert set(resolved) == {"syn-dev-01", "syn-dev-02", "syn-dev-03", "syn-dev-04"}


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
        call_receipt_artifact_hashes=tuple(digest(("call-receipt", index)) for index in range(24)),
        service_result_artifact_hashes=tuple(
            digest(("service-result", index)) for index in range(24)
        ),
        cpu_projection_receipt_artifact_hashes=tuple(digest(("cpu", index)) for index in range(24)),
        packing_preflight_artifact_hash=digest("packing-preflight"),
        runtime_sources=runtime_bindings(),
    )


def test_assessment_manifest_requires_every_immutable_output() -> None:
    payload = assessment_manifest().model_dump(mode="python", exclude={"content_hash"})
    payload["service_result_artifact_hashes"] = payload["service_result_artifact_hashes"][:-1]
    with pytest.raises(ValueError, match="24 service-result"):
        DevelopmentAssessmentInputManifest.model_validate(payload)

    payload = assessment_manifest().model_dump(mode="python", exclude={"content_hash"})
    payload["cpu_projection_receipt_artifact_hashes"] = payload[
        "cpu_projection_receipt_artifact_hashes"
    ][:-1]
    with pytest.raises(ValueError, match="12 C0 and 12 C1"):
        DevelopmentAssessmentInputManifest.model_validate(payload)

    payload = assessment_manifest().model_dump(mode="python", exclude={"content_hash"})
    payload["call_receipt_artifact_hashes"] = payload["call_receipt_artifact_hashes"][:-1]
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

        changed = stored.model_copy(update={"logical_content_hash": digest("wrong-logical-hash")})
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


@pytest.mark.parametrize(
    ("successful", "validation_status"),
    [
        (True, LedgerValidationStatus.ACCEPTED),
        (False, LedgerValidationStatus.REJECTED),
    ],
)
def test_development_ledger_integrity_requires_structural_only_status_tuple(
    successful: bool,
    validation_status: LedgerValidationStatus,
) -> None:
    record = SimpleNamespace(
        job_id="job-1",
        attempt_id="attempt-1",
        input_artifact_hash=digest("raw"),
        validator_manifest_hash=digest("validator"),
        validation_status=validation_status,
        evidence_support_status=LedgerEvidenceSupportStatus.NOT_APPLICABLE,
        temporal_status=LedgerTemporalValidationStatus.NOT_APPLICABLE,
        commitment_status=LedgerCommitmentCheckStatus.NOT_APPLICABLE,
        semantic_assessment_scope=(
            LedgerSemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
        ),
    )
    ledger = SimpleNamespace(get_validation=lambda _validation_id: record)
    receipt = SimpleNamespace(
        ledger_validation_ids=("validation-1",),
        job_id=record.job_id,
        attempt_id=record.attempt_id,
    )

    _validate_ledger_validations(
        cast(Any, ledger),
        cast(Any, receipt),
        raw_response_hash=record.input_artifact_hash,
        validator_hash=record.validator_manifest_hash,
        successful=successful,
    )

    record.evidence_support_status = LedgerEvidenceSupportStatus.SUPPORTED
    with pytest.raises(
        DevelopmentAssessmentIntegrityError,
        match="structural-only append-only validation verdict",
    ):
        _validate_ledger_validations(
            cast(Any, ledger),
            cast(Any, receipt),
            raw_response_hash=record.input_artifact_hash,
            validator_hash=record.validator_manifest_hash,
            successful=successful,
        )

    record.evidence_support_status = LedgerEvidenceSupportStatus.NOT_APPLICABLE
    record.semantic_assessment_scope = LedgerSemanticAssessmentScope.LEGACY_UNSPECIFIED
    with pytest.raises(
        DevelopmentAssessmentIntegrityError,
        match="structural-only append-only validation verdict",
    ):
        _validate_ledger_validations(
            cast(Any, ledger),
            cast(Any, receipt),
            raw_response_hash=record.input_artifact_hash,
            validator_hash=record.validator_manifest_hash,
            successful=successful,
        )


def test_scorer_marks_structurally_valid_false_assertion_unsupported_and_strict_miss() -> None:
    request_payload = json.loads(
        (ROOT / "tests/fixtures/phase1/c2_query_request.json").read_text(encoding="utf-8")
    )
    for index, evidence in enumerate(request_payload["packet"]["evidence"], start=1):
        text_hash = hashlib.sha256(evidence["text"].encode("utf-8")).hexdigest()
        evidence.update(
            {
                "passage_id": f"development-assessment-passage-{index}",
                "text_hash": text_hash,
                "confidence": 1.0,
                "provenance": {
                    "provenance_id": f"development-assessment-provenance-{index}",
                    "evidence_id": evidence["evidence_id"],
                    "extraction_method": "hand-authored synthetic test lineage",
                    "locator": f"development-assessment:{index}",
                    "source_artifact_hash": text_hash,
                    "confidence": 1.0,
                },
            }
        )
    request = ConstructionRequest.model_validate(request_payload)
    indexed_by_id = {item.evidence_id: item for item in request.packet.evidence}
    source_payload = json.loads(
        (ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text(encoding="utf-8")
    )
    for assertion in source_payload["instance_graph"]["assertions"]:
        for provenance in assertion["provenance"]:
            indexed = indexed_by_id[provenance["evidence_id"]]
            provenance["locator"] = indexed.provenance.locator
            provenance["source_artifact_hash"] = indexed.provenance.source_artifact_hash
            provenance["confidence"] = min(indexed.confidence, indexed.provenance.confidence)
    source = OntologyDraft.model_validate(source_payload)
    valid_evidence_ids = frozenset(item.evidence_id for item in request.packet.evidence)
    node_targets = tuple(
        NodeAlignmentTarget(
            target_id=f"gold-{entity.entity_id}",
            kind=NodeKind.ENTITY,
            anchor_kind=AnchorKind.MENTION,
            permissible_anchor_sets=(tuple(sorted(entity.supported_mention_candidate_ids)),),
        )
        for entity in source.instance_graph.entities
    ) + tuple(
        NodeAlignmentTarget(
            target_id=f"gold-{event.event_id}",
            kind=NodeKind.EVENT,
            anchor_kind=AnchorKind.EVIDENCE,
            permissible_anchor_sets=(tuple(sorted(event.evidence_ids)),),
        )
        for event in source.instance_graph.events
    )
    nodes_only_plan = AlignmentPlan(
        matcher_revision="semantic-separation-regression-v1",
        source_gold_hash="a" * 64,
        source_alternative_set_hash="b" * 64,
        node_targets=node_targets,
        assertion_targets=(),
    )
    _, source_predictions, _ = prediction_records_from_components(
        local_schema=source.local_schema,
        instance_graph=source.instance_graph,
        predicate_aliases=None,
        valid_evidence_ids=valid_evidence_ids,
        grounding_by_assertion_id={
            item.assertion_id: GroundingStatus.SUPPORTED
            for item in source.instance_graph.assertions
        },
        plan=nodes_only_plan,
    )
    supported_source = source_predictions[0]
    plan = AlignmentPlan(
        matcher_revision=nodes_only_plan.matcher_revision,
        source_gold_hash=nodes_only_plan.source_gold_hash,
        source_alternative_set_hash=nodes_only_plan.source_alternative_set_hash,
        node_targets=node_targets,
        assertion_targets=(
            AssertionAlignmentTarget(
                target_id="gold-assertion",
                alternatives=(
                    PermissibleAssertionAlternative(
                        alternative_id="registered-primary",
                        signature=supported_source.signature,
                        supporting_evidence_ids=supported_source.evidence_ids,
                    ),
                ),
                essential_temporal=True,
            ),
        ),
    )

    false_assertion = revalidated_copy(
        source.instance_graph.assertions[0],
        predicate_id="p-c2-status",
    )
    false_graph = revalidated_copy(
        source.instance_graph,
        assertions=(false_assertion, *source.instance_graph.assertions[1:]),
    )
    structurally_valid_false_draft = revalidated_copy(source, instance_graph=false_graph)
    structural = validate_draft_structure(
        draft=structurally_valid_false_draft,
        upper_ontology=request.upper_ontology,
        evidence=request.packet.evidence,
        horizon=request.context.spoiler_horizon,
        budgets=request.budgets,
        capabilities=request.capabilities,
    )
    assert structural.accepted

    nodes, assertions, supported_ids = _prediction_bundle(
        local_schema=structurally_valid_false_draft.local_schema,
        instance_graph=structurally_valid_false_draft.instance_graph,
        plan=plan,
        valid_evidence_ids=valid_evidence_ids,
    )
    score = score_alignment(
        plan=plan,
        predicted_nodes=nodes,
        predicted_assertions=assertions,
    )
    by_id = {item.prediction_id: item for item in assertions}

    assert false_assertion.assertion_id not in supported_ids
    assert by_id[false_assertion.assertion_id].grounding_status is GroundingStatus.UNSUPPORTED
    assert score.strict_assertion_matches == ()
    assert score.strict_assertion_score.true_positive_count == 0
    assert score.strict_assertion_score.f1 == 0.0
