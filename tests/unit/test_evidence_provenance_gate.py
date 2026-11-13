from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from story_projection_onto.contracts import (
    DiscoursePosition,
    EvidenceRecord,
    LegacyModelVisibleEvidenceRecord,
    ModelVisibleEvidenceRecord,
    OntologyDraft,
    PreconstructionRequest,
    ProvenanceReference,
    ReleaseClass,
    SpoilerHorizon,
    to_model_visible_evidence,
)
from story_projection_onto.development_adapter import (
    DevelopmentAdapterIntegrityError,
    decode_development_semantic_request,
    encode_development_semantic_request,
)
from story_projection_onto.validate import ValidationCode, validate_draft_structure

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase1"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _without_content_hashes(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, list):
        return [_without_content_hashes(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_without_content_hashes(child) for child in value)
    return value


def _legacy_request() -> PreconstructionRequest:
    return PreconstructionRequest.model_validate_json(
        (FIXTURES / "c1_pre_request.json").read_text(encoding="utf-8")
    )


def _horizon() -> SpoilerHorizon:
    return SpoilerHorizon(
        horizon_id="provenance-gate-horizon",
        max_discourse_position=DiscoursePosition(passage_order=100),
    )


def _modern_evidence(
    request: PreconstructionRequest,
) -> tuple[ModelVisibleEvidenceRecord, ...]:
    records: list[ModelVisibleEvidenceRecord] = []
    for index, legacy in enumerate(request.evidence, start=1):
        provenance = ProvenanceReference(
            provenance_id=f"indexed-{legacy.evidence_id}",
            evidence_id=legacy.evidence_id,
            extraction_method="query-blind fixture indexing",
            locator=f"fixture://phase1/{legacy.evidence_id}",
            source_artifact_hash=_digest(f"source:{legacy.evidence_id}"),
            confidence=0.8,
        )
        records.append(
            to_model_visible_evidence(
                EvidenceRecord(
                    evidence_id=legacy.evidence_id,
                    passage_id=f"passage-{index}",
                    text=legacy.text,
                    text_hash=_digest(legacy.text),
                    discourse_position=legacy.discourse_position,
                    mention_candidates=legacy.mention_candidates,
                    event_candidates=legacy.event_candidates,
                    relation_phrase_candidates=legacy.relation_phrase_candidates,
                    temporal_clues=legacy.temporal_clues,
                    provenance=provenance,
                    confidence=0.9,
                    release_class=ReleaseClass.PUBLIC,
                )
            )
        )
    return tuple(records)


def _source_bound_draft(
    evidence: tuple[ModelVisibleEvidenceRecord, ...],
) -> OntologyDraft:
    raw = _without_content_hashes(
        json.loads((FIXTURES / "c1_pre_output.json").read_text(encoding="utf-8"))
    )
    indexed = {item.evidence_id: item for item in evidence}
    for assertion in raw["instance_graph"]["assertions"]:
        for emitted in assertion["provenance"]:
            source = indexed[emitted["evidence_id"]]
            emitted["locator"] = source.provenance.locator
            emitted["source_artifact_hash"] = source.provenance.source_artifact_hash
            emitted["confidence"] = min(source.confidence, source.provenance.confidence)
    return OntologyDraft.model_validate(raw)


def _report(draft: OntologyDraft, evidence: tuple[ModelVisibleEvidenceRecord, ...]):
    request = _legacy_request()
    return validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=evidence,
        horizon=_horizon(),
        budgets=request.budgets,
        capabilities=request.capabilities,
    )


def test_live_model_wire_carries_and_losslessly_decodes_indexed_grounding() -> None:
    legacy = _legacy_request()
    evidence = _modern_evidence(legacy)
    payload = legacy.model_dump(mode="python", exclude={"content_hash"})
    payload["sealed_horizon"] = _horizon()
    payload["evidence"] = evidence
    request = PreconstructionRequest.model_validate(payload)

    encoded = encode_development_semantic_request(request)
    grounding = encoded.sections["evidence_grounding"]
    assert isinstance(grounding, dict)
    first_alias = encoded.alias_manifest.source_to_alias[evidence[0].evidence_id]
    first = grounding[first_alias]
    assert first["passage_id"] == evidence[0].passage_id
    assert first["text_hash"] == evidence[0].text_hash
    assert first["record_confidence"] == evidence[0].confidence
    assert first["provenance"]["locator"] == evidence[0].provenance.locator
    assert (
        first["provenance"]["source_artifact_hash"]
        == evidence[0].provenance.source_artifact_hash
    )
    assert decode_development_semantic_request(encoded) == request


def test_legacy_request_remains_parseable_but_cannot_enter_production_validation_or_wire() -> None:
    request = _legacy_request()
    assert all(isinstance(item, LegacyModelVisibleEvidenceRecord) for item in request.evidence)
    draft = OntologyDraft.model_validate_json(
        (FIXTURES / "c1_pre_output.json").read_text(encoding="utf-8")
    )

    report = validate_draft_structure(
        draft=draft,
        upper_ontology=request.upper_ontology,
        evidence=request.evidence,
        horizon=_horizon(),
        budgets=request.budgets,
        capabilities=request.capabilities,
    )
    assert not report.accepted
    assert any(
        item.code is ValidationCode.PROVENANCE_MISMATCH
        and item.path.startswith("evidence.")
        for item in report.diagnostics
    )
    with pytest.raises(DevelopmentAdapterIntegrityError, match="legacy evidence"):
        encode_development_semantic_request(request)


def test_live_c1_request_contract_rejects_legacy_or_mixed_evidence() -> None:
    legacy = _legacy_request()
    payload = legacy.model_dump(mode="python", exclude={"content_hash"})
    payload["sealed_horizon"] = _horizon()
    with pytest.raises(ValueError, match="exact model-visible evidence lineage"):
        PreconstructionRequest.model_validate(payload)

    modern = _modern_evidence(legacy)
    payload["evidence"] = (modern[0], *legacy.evidence[1:])
    with pytest.raises(ValueError, match="exact model-visible evidence lineage"):
        PreconstructionRequest.model_validate(payload)


def test_assertion_specific_provenance_identity_is_allowed_when_source_is_exact() -> None:
    evidence = _modern_evidence(_legacy_request())
    draft = _source_bound_draft(evidence)
    payload = _without_content_hashes(draft.model_dump(mode="python"))
    emitted = payload["instance_graph"]["assertions"][0]["provenance"][0]
    emitted["provenance_id"] = "assertion-specific-provenance"
    emitted["extraction_method"] = "assertion-specific grounded construction"
    changed = OntologyDraft.model_validate(payload)

    assert _report(changed, evidence).accepted


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("locator", "fixture://tampered-source"),
        ("source_artifact_hash", "f" * 64),
        ("confidence", 0.81),
    ),
)
def test_structural_validation_rejects_tampered_or_overconfident_provenance(
    field: str,
    value: object,
) -> None:
    evidence = _modern_evidence(_legacy_request())
    draft = _source_bound_draft(evidence)
    payload = _without_content_hashes(draft.model_dump(mode="python"))
    payload["instance_graph"]["assertions"][0]["provenance"][0][field] = value
    changed = OntologyDraft.model_validate(payload)

    report = _report(changed, evidence)
    assert not report.accepted
    assert any(item.code is ValidationCode.PROVENANCE_MISMATCH for item in report.diagnostics)


@pytest.mark.parametrize(
    ("evidence_ids", "expected_code"),
    (
        ([], ValidationCode.MISSING_GROUNDING),
        (["evidence-outside-packet"], ValidationCode.EVIDENCE_OUTSIDE_PACKET),
    ),
)
def test_production_local_schema_requires_nonempty_valid_evidence(
    evidence_ids: list[str],
    expected_code: ValidationCode,
) -> None:
    evidence = _modern_evidence(_legacy_request())
    draft = _source_bound_draft(evidence)
    payload = copy.deepcopy(_without_content_hashes(draft.model_dump(mode="python")))
    payload["local_schema"]["contextual_types"][0]["evidence_ids"] = evidence_ids
    changed = OntologyDraft.model_validate(payload)

    report = _report(changed, evidence)
    assert not report.accepted
    assert any(item.code is expected_code for item in report.diagnostics)
