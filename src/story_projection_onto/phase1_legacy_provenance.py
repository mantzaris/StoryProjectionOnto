"""Authenticated provenance bridge for the immutable Phase-1 v3 fixtures.

The three scientific acceptance requests were frozen before model-visible evidence
records carried their query-blind source lineage.  This module does not make that
legacy shape generally admissible.  It recognizes only the byte-exact registered
Phase-1 files and calls, verifies their common evidence value record by record, and
materializes the missing administrative lineage for the strict production validator.

The bridge is deliberately explicit at every call site.  Loading a legacy request,
calling the production validator, or building any other model request never activates
it implicitly.
"""

from __future__ import annotations

import copy
import hashlib
import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .contracts import (
    ConditionName,
    ModelVisibleEvidenceRecord,
    OntologyDraft,
    ProvenanceReference,
    assert_evidence_boundary,
    canonical_sha256,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]

CERTIFICATE_RELATIVE_PATH = "configs/study/phase1_legacy_evidence_provenance.json"
BRIDGE_REVISION = "phase1/immutable-v3-evidence-provenance/v1"

_REGISTERED_FIXTURE_HASHES = {
    "tests/fixtures/phase1/c1_pre_request.json": (
        "98ad2955f2311277ae2be45a2df8552f3d90cd7ebe607934f6163171e568d666"
    ),
    "tests/fixtures/phase1/c2_query_request.json": (
        "46a95f154c607fe0bcbeda1206f96590a12be08ed72fc44e699872fe97f0565d"
    ),
    "tests/fixtures/phase1/fixed_select_request.json": (
        "8690f2e1d316066b4abbe534187a7d01b314e0857c7463195a04c9517c8d8ad5"
    ),
    "tests/fixtures/phase1/invalid_repair_case.json": (
        "0f7e2a80c6881ab86828c72691f902890196c98bd915607e655510e9372e3276"
    ),
    "tests/fixtures/phase1/c1_pre_output.json": (
        "41dd9e2f6b3739205779d944d0d003bcb81f372da1f68d6305ed7b7a31622593"
    ),
}

_REGISTERED_EVIDENCE_POINTERS = {
    "tests/fixtures/phase1/c1_pre_request.json": "/evidence",
    "tests/fixtures/phase1/c2_query_request.json": "/packet/evidence",
    "tests/fixtures/phase1/fixed_select_request.json": "/packet/evidence",
    "tests/fixtures/phase1/invalid_repair_case.json": None,
    "tests/fixtures/phase1/c1_pre_output.json": None,
}

_REGISTERED_CALLS = {
    "c1-01": (
        ConditionName.C1_LLM_PRE,
        "tests/fixtures/phase1/c1_pre_request.json",
        "tests/fixtures/phase1/c1_pre_request.json",
    ),
    "c1-02": (
        ConditionName.C1_LLM_PRE,
        "tests/fixtures/phase1/c1_pre_request.json",
        "tests/fixtures/phase1/c1_pre_request.json",
    ),
    "c2-01": (
        ConditionName.C2_LLM_QUERY,
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
    ),
    "c2-02": (
        ConditionName.C2_LLM_QUERY,
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
    ),
    "c2-03": (
        ConditionName.C2_LLM_QUERY,
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
    ),
    "fixed-01": (
        ConditionName.A_FIXED_SELECT,
        "tests/fixtures/phase1/fixed_select_request.json",
        "tests/fixtures/phase1/fixed_select_request.json",
    ),
    "fixed-02": (
        ConditionName.A_FIXED_SELECT,
        "tests/fixtures/phase1/fixed_select_request.json",
        "tests/fixtures/phase1/fixed_select_request.json",
    ),
    "repair-01": (
        ConditionName.C2_LLM_QUERY,
        "tests/fixtures/phase1/invalid_repair_case.json",
        "tests/fixtures/phase1/c2_query_request.json",
    ),
    "fallback-c1-01": (
        ConditionName.C1_LLM_PRE,
        "tests/fixtures/phase1/c1_pre_request.json",
        "tests/fixtures/phase1/c1_pre_request.json",
    ),
    "fallback-c2-01": (
        ConditionName.C2_LLM_QUERY,
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
    ),
    "fallback-c2-02": (
        ConditionName.C2_LLM_QUERY,
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
    ),
    "fallback-fixed-01": (
        ConditionName.A_FIXED_SELECT,
        "tests/fixtures/phase1/fixed_select_request.json",
        "tests/fixtures/phase1/fixed_select_request.json",
    ),
}

_REGISTERED_EVIDENCE = (
    (
        "ev-01",
        "phase1-harbor-passage-01",
        "6d58b129db85ad0fb6a7b49aaed53e4798704755f0c21d3c5dd903c6e10d60e9",
        "bce73eeff552f84b5fdbf342c0d73d0206ee31faa56dcd70ce24d1aafc422c61",
    ),
    (
        "ev-02",
        "phase1-harbor-passage-02",
        "a23e92c5a9efe9c7cf4bbc86b33b905aae63e4913e71e0a23cc6fef8491e414d",
        "6fcc067500756ed8452d89a313224de0d7728cd5672566ffc8ba18f93b7ac253",
    ),
    (
        "ev-03",
        "phase1-harbor-passage-03",
        "d8bae6145787bd76b4643b100295f8c066b2561d5a34f930b237cee3870faf34",
        "3264bad5993b2b36d3bf5ccdcee58f82e9f49d39528dc53150ea91b09c500181",
    ),
    (
        "ev-04",
        "phase1-harbor-passage-04",
        "8287068978462551ab3c8196ce6334031578350566537b02f13a5868111be10a",
        "dabac5b09e12bdb3032db7465e3841bd40f11efdf7169343a47782bb3183879f",
    ),
    (
        "ev-05",
        "phase1-harbor-passage-05",
        "0849174fe99888ac7c3bfbd7a0caeb0d1f3b4394d3d3e743cdd5caafaf0bd92c",
        "f8621df86450f6db7833156f190ec0d2ea3eb1c90bab1151ec63d1d584dc1ece",
    ),
    (
        "ev-06",
        "phase1-harbor-passage-06",
        "54ba2bd861d1d2c60e5eb2baac772e696639dd1a18697a053db6badced91a547",
        "d9c7815b48e2666add758fde65d68e663c0faa8b038b42b7587e0693a9af0861",
    ),
    (
        "ev-07",
        "phase1-harbor-passage-07",
        "538ee94ca6b229a655800647732e3490ba8ac9ed9f1d9fb1268a3257408fbd8c",
        "366f004f6af7730687e2811c010c6ebfb3d9ce1038e2c6074a3d54e510834938",
    ),
)


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Phase1LegacyFixtureBinding(_FrozenRecord):
    relative_path: str = Field(min_length=1)
    file_sha256: Sha256
    evidence_pointer: Literal["/evidence", "/packet/evidence"] | None
    evidence_payload_sha256: Sha256 | None

    @model_validator(mode="after")
    def evidence_pointer_and_hash_agree(self) -> Self:
        if (self.evidence_pointer is None) != (self.evidence_payload_sha256 is None):
            raise ValueError("legacy fixture evidence pointer/hash presence differs")
        return self


class Phase1LegacyCallBinding(_FrozenRecord):
    call_id: str = Field(min_length=1)
    condition: ConditionName
    request_fixture: str = Field(min_length=1)
    semantic_fixture: str = Field(min_length=1)


class Phase1LegacySourceMetadata(_FrozenRecord):
    provenance_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    source_artifact_hash: Sha256
    provenance_confidence: UnitInterval


class Phase1LegacyEvidenceBinding(_FrozenRecord):
    sequence: int = Field(ge=1)
    evidence_id: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    legacy_record_sha256: Sha256
    text_sha256: Sha256
    record_confidence: UnitInterval
    source: Phase1LegacySourceMetadata


class Phase1LegacyEvidenceProvenanceCertificate(_FrozenRecord):
    """Typed, self-hashed authority for the one frozen acceptance exception."""

    schema_version: Literal["1.0.0"]
    kind: Literal["phase1_legacy_evidence_provenance_bridge"]
    bridge_revision: Literal[BRIDGE_REVISION]
    protocol: Literal["immutable-v3-phase1-provenance-sidecar-v1"]
    scope: Literal["exact_phase1_acceptance_fixtures_only"]
    source_snapshot_sha256: Sha256
    shared_legacy_evidence_sha256: Sha256
    fixture_bindings: tuple[Phase1LegacyFixtureBinding, ...]
    call_bindings: tuple[Phase1LegacyCallBinding, ...]
    evidence_bindings: tuple[Phase1LegacyEvidenceBinding, ...]
    added_fields: tuple[
        Literal[
            "passage_id",
            "text_hash",
            "record_confidence",
            "provenance.source_artifact_hash",
        ],
        ...,
    ]
    confidence_semantics: Literal[
        "byte-verified synthetic record/source linkage, not proposition truth"
    ]
    query_blind: Literal[True]
    gold_free: Literal[True]
    fixture_bytes_modified: Literal[False]
    text_candidates_order_budgets_changed: Literal[False]
    cpu_semantic_inference_performed: Literal[False]
    production_reuse_forbidden: Literal[True]
    development_reuse_forbidden: Literal[True]
    held_out_reuse_forbidden: Literal[True]
    case_study_reuse_forbidden: Literal[True]
    fixed_raw_semantic_comparison_precedes_effective_binding: Literal[True]
    derivation_rules: tuple[str, ...]
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def validate_registered_authority(self) -> Self:
        immutable = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_sha256(immutable):
            raise ValueError("legacy provenance certificate hash does not match its contents")
        if self.bridge_revision != BRIDGE_REVISION:
            raise ValueError("legacy provenance certificate uses an unknown bridge revision")
        fixtures = {
            item.relative_path: (
                item.file_sha256,
                item.evidence_pointer,
                item.evidence_payload_sha256,
            )
            for item in self.fixture_bindings
        }
        expected_fixtures = {
            path: (
                digest,
                _REGISTERED_EVIDENCE_POINTERS[path],
                (
                    self.shared_legacy_evidence_sha256
                    if _REGISTERED_EVIDENCE_POINTERS[path] is not None
                    else None
                ),
            )
            for path, digest in _REGISTERED_FIXTURE_HASHES.items()
        }
        if fixtures != expected_fixtures:
            raise ValueError("legacy provenance certificate fixture authority changed")
        if len(self.fixture_bindings) != len(fixtures):
            raise ValueError("legacy provenance certificate repeats a fixture")
        if tuple(item.relative_path for item in self.fixture_bindings) != tuple(
            _REGISTERED_FIXTURE_HASHES
        ):
            raise ValueError("legacy provenance certificate fixture order changed")
        calls = {
            item.call_id: (item.condition, item.request_fixture, item.semantic_fixture)
            for item in self.call_bindings
        }
        if calls != _REGISTERED_CALLS or len(calls) != len(self.call_bindings):
            raise ValueError("legacy provenance certificate call scope changed")
        if tuple(item.call_id for item in self.call_bindings) != tuple(_REGISTERED_CALLS):
            raise ValueError("legacy provenance certificate call order changed")
        expected_evidence = tuple(
            (evidence_id, passage_id, record_hash, text_hash)
            for evidence_id, passage_id, record_hash, text_hash in _REGISTERED_EVIDENCE
        )
        observed_evidence = tuple(
            (
                item.evidence_id,
                item.passage_id,
                item.legacy_record_sha256,
                item.text_sha256,
            )
            for item in self.evidence_bindings
        )
        if observed_evidence != expected_evidence:
            raise ValueError("legacy provenance certificate evidence authority changed")
        if tuple(item.sequence for item in self.evidence_bindings) != tuple(
            range(1, len(self.evidence_bindings) + 1)
        ):
            raise ValueError("legacy provenance evidence sequence is not exact")
        for item in self.evidence_bindings:
            if (
                item.record_confidence != 1.0
                or item.source.provenance_id != f"phase1-indexed-{item.evidence_id}"
                or item.source.evidence_id != item.evidence_id
                or item.source.extraction_method
                != "immutable-v3-byte-bound-fixture"
                or item.source.locator != item.evidence_id
                or item.source.source_artifact_hash
                != self.shared_legacy_evidence_sha256
                or item.source.provenance_confidence != 1.0
            ):
                raise ValueError("legacy provenance deterministic source metadata changed")
        expected_rules = (
            "passage_id=phase1-harbor-passage-{one-based evidence sequence:02d}",
            "text_sha256=sha256(UTF-8 exact legacy evidence text)",
            "legacy_record_sha256=canonical_sha256(exact legacy evidence object)",
            "source_artifact_hash=shared_legacy_evidence_sha256",
            "locator=evidence_id",
            "record_confidence=provenance_confidence=1.0",
            "controller may add only a missing assertion provenance source_artifact_hash",
        )
        if self.derivation_rules != expected_rules:
            raise ValueError("legacy provenance certificate derivation rules changed")
        if self.added_fields != (
            "passage_id",
            "text_hash",
            "record_confidence",
            "provenance.source_artifact_hash",
        ):
            raise ValueError("legacy provenance certificate added-field scope changed")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedPhase1LegacyEvidence:
    call_id: str
    semantic_fixture: str
    semantic_fixture_sha256: str
    legacy_evidence_sha256: str
    resolved_evidence_sha256: str
    evidence: tuple[ModelVisibleEvidenceRecord, ...]


@dataclass(frozen=True, slots=True)
class EffectivePhase1Draft:
    raw_draft: OntologyDraft
    effective_draft: OntologyDraft
    raw_draft_sha256: str
    effective_draft_sha256: str
    inserted_source_hash_paths: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_registered_file(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != relative_path
    ):
        raise ValueError("legacy provenance bridge received an unsafe path")
    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("legacy provenance bridge refuses symlinked inputs")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("legacy provenance bridge input escapes the project root") from exc
    if not stat.S_ISREG(resolved.stat(follow_symlinks=False).st_mode):
        raise ValueError("legacy provenance bridge input is not a regular file")
    return resolved


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("legacy provenance bridge input must be a JSON object")
    return cast(dict[str, object], value)


def _legacy_evidence_from_request(
    request: Mapping[str, object], relative_path: str
) -> tuple[Mapping[str, object], ...]:
    if relative_path.endswith("c1_pre_request.json"):
        value = request.get("evidence")
    else:
        packet = request.get("packet")
        value = packet.get("evidence") if isinstance(packet, Mapping) else None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not all(isinstance(item, Mapping) for item in value)
    ):
        raise ValueError("registered legacy request has no exact evidence sequence")
    return tuple(cast(Sequence[Mapping[str, object]], value))


def _without_content_hashes(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _without_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_content_hashes(child) for child in value]
    return value


def _without_source_artifact_hashes(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _without_source_artifact_hashes(child)
            for key, child in value.items()
            if key != "source_artifact_hash"
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_source_artifact_hashes(child) for child in value]
    return value


@dataclass(frozen=True, slots=True)
class Phase1LegacyEvidenceProvenanceBridge:
    """Verified, acceptance-only resolver for the immutable-v3 evidence value."""

    root: Path
    certificate_path: Path
    certificate_file_sha256: str
    certificate: Phase1LegacyEvidenceProvenanceCertificate

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        certificate_relative_path: str = CERTIFICATE_RELATIVE_PATH,
    ) -> Self:
        resolved_root = root.resolve(strict=True)
        certificate_path = _safe_registered_file(resolved_root, certificate_relative_path)
        certificate = Phase1LegacyEvidenceProvenanceCertificate.model_validate(
            _load_json_object(certificate_path)
        )
        bridge = cls(
            root=resolved_root,
            certificate_path=certificate_path,
            certificate_file_sha256=_file_sha256(certificate_path),
            certificate=certificate,
        )
        bridge._verify_all_registered_files()
        return bridge

    @property
    def manifest_sha256(self) -> str:
        return self.certificate.manifest_sha256

    def _verified_fixture(self, relative_path: str) -> dict[str, object]:
        expected_hash = _REGISTERED_FIXTURE_HASHES.get(relative_path)
        if expected_hash is None:
            raise ValueError("legacy provenance bridge fixture is outside its frozen scope")
        path = _safe_registered_file(self.root, relative_path)
        if _file_sha256(path) != expected_hash:
            raise ValueError("registered immutable-v3 fixture bytes changed")
        return _load_json_object(path)

    def _verify_all_registered_files(self) -> None:
        for relative_path in _REGISTERED_FIXTURE_HASHES:
            self._verified_fixture(relative_path)
        semantic_paths = (
            "tests/fixtures/phase1/c1_pre_request.json",
            "tests/fixtures/phase1/c2_query_request.json",
            "tests/fixtures/phase1/fixed_select_request.json",
        )
        evidence_values = [
            _legacy_evidence_from_request(self._verified_fixture(path), path)
            for path in semantic_paths
        ]
        if not all(value == evidence_values[0] for value in evidence_values[1:]):
            raise ValueError("registered immutable-v3 requests no longer share exact evidence")
        evidence = evidence_values[0]
        if canonical_sha256(evidence) != self.certificate.shared_legacy_evidence_sha256:
            raise ValueError("legacy provenance certificate evidence-value hash changed")
        snapshots = {self._snapshot_hash(self._verified_fixture(path)) for path in semantic_paths}
        if snapshots != {self.certificate.source_snapshot_sha256}:
            raise ValueError("legacy provenance certificate snapshot binding changed")
        self._resolve_records(evidence)

    @staticmethod
    def _snapshot_hash(request: Mapping[str, object]) -> object:
        return request.get("snapshot_hash")

    def _resolve_records(
        self, evidence: Sequence[Mapping[str, object]]
    ) -> tuple[ModelVisibleEvidenceRecord, ...]:
        if len(evidence) != len(self.certificate.evidence_bindings):
            raise ValueError("legacy evidence record count changed")
        resolved: list[ModelVisibleEvidenceRecord] = []
        for raw, binding in zip(evidence, self.certificate.evidence_bindings, strict=True):
            raw_value = dict(raw)
            text = raw_value.get("text")
            if (
                raw_value.get("evidence_id") != binding.evidence_id
                or canonical_sha256(raw_value) != binding.legacy_record_sha256
                or not isinstance(text, str)
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != binding.text_sha256
            ):
                raise ValueError("registered immutable-v3 evidence record changed")
            source = binding.source
            raw_value.update(
                {
                    "passage_id": binding.passage_id,
                    "text_hash": binding.text_sha256,
                    "confidence": binding.record_confidence,
                    "provenance": ProvenanceReference(
                        provenance_id=source.provenance_id,
                        evidence_id=binding.evidence_id,
                        extraction_method=source.extraction_method,
                        locator=source.locator,
                        source_artifact_hash=source.source_artifact_hash,
                        confidence=source.provenance_confidence,
                    ),
                }
            )
            record = ModelVisibleEvidenceRecord.model_validate(raw_value)
            assert_evidence_boundary(record)
            resolved.append(record)
        return tuple(resolved)

    def resolve_call(
        self,
        *,
        call_id: str,
        condition: ConditionName,
        request_fixture: str,
    ) -> ResolvedPhase1LegacyEvidence:
        binding = next(
            (item for item in self.certificate.call_bindings if item.call_id == call_id),
            None,
        )
        expected = _REGISTERED_CALLS.get(call_id)
        observed = None
        if binding is not None:
            observed = (binding.condition, binding.request_fixture, binding.semantic_fixture)
        if expected is None or observed != expected:
            raise ValueError("legacy provenance bridge call is outside its frozen scope")
        if condition is not binding.condition or request_fixture != binding.request_fixture:
            raise ValueError("legacy provenance bridge call identity changed")
        # Verify the actual request fixture even when the repair call obtains its
        # scientific evidence from the independently frozen C2 request.
        self._verified_fixture(binding.request_fixture)
        semantic = self._verified_fixture(binding.semantic_fixture)
        raw_evidence = _legacy_evidence_from_request(semantic, binding.semantic_fixture)
        evidence = self._resolve_records(raw_evidence)
        return ResolvedPhase1LegacyEvidence(
            call_id=call_id,
            semantic_fixture=binding.semantic_fixture,
            semantic_fixture_sha256=_REGISTERED_FIXTURE_HASHES[binding.semantic_fixture],
            legacy_evidence_sha256=canonical_sha256(raw_evidence),
            resolved_evidence_sha256=canonical_sha256(evidence),
            evidence=evidence,
        )

    def model_visible_provenance_section(
        self,
        *,
        call_id: str,
        condition: ConditionName,
        request_fixture: str,
    ) -> dict[str, object]:
        resolved = self.resolve_call(
            call_id=call_id,
            condition=condition,
            request_fixture=request_fixture,
        )
        rows = []
        for record in resolved.evidence:
            provenance = record.provenance.model_dump(
                mode="json", exclude={"content_hash", "schema_version"}
            )
            effective_source_hash = provenance["source_artifact_hash"]
            if condition is ConditionName.A_FIXED_SELECT:
                # The registered sealed C1 ontology predates the field.  FixedSelect
                # must preserve that raw null before the controller creates a separate
                # validation-only view; otherwise it would mutate a pre-query object.
                provenance["source_artifact_hash"] = None
            row = {
                "evidence_id": record.evidence_id,
                "passage_id": record.passage_id,
                "text_hash": record.text_hash,
                "record_confidence": record.confidence,
                "provenance": provenance,
            }
            if condition is ConditionName.A_FIXED_SELECT:
                row["effective_validation_source_artifact_hash"] = effective_source_hash
            rows.append(row)
        output_rule = (
            "Preserve the sealed ontology's raw provenance exactly, including its null "
            "source_artifact_hash. The controller may add only that missing hash in a "
            "separate post-selection validation view."
            if condition is ConditionName.A_FIXED_SELECT
            else (
                "For every cited evidence ID, copy its exact locator and "
                "source_artifact_hash; emitted confidence must not exceed either "
                "indexed confidence bound."
            )
        )
        section = {
            "bridge_revision": BRIDGE_REVISION,
            "certificate_sha256": self.manifest_sha256,
            "source_fixture_sha256": resolved.semantic_fixture_sha256,
            "legacy_evidence_sha256": resolved.legacy_evidence_sha256,
            "resolved_evidence_sha256": resolved.resolved_evidence_sha256,
            "records": rows,
            "output_rule": output_rule,
        }
        assert_evidence_boundary(section)
        return section

    def effective_draft(
        self,
        *,
        raw_draft: OntologyDraft,
        resolved: ResolvedPhase1LegacyEvidence,
        purpose: Literal["fixed_select", "historical_reference"],
    ) -> EffectivePhase1Draft:
        """Add only absent source hashes; never repair any emitted semantic value."""

        if purpose == "fixed_select" and not resolved.call_id.startswith(
            ("fixed-", "fallback-fixed-")
        ):
            raise ValueError("legacy effective view is not authorized for a non-FixedSelect call")
        raw_value = cast(
            dict[str, object],
            _without_content_hashes(raw_draft.model_dump(mode="python")),
        )
        if purpose == "historical_reference":
            historical = OntologyDraft.model_validate(
                self._verified_fixture("tests/fixtures/phase1/c1_pre_output.json")
            )
            historical_value = _without_content_hashes(historical.model_dump(mode="python"))
            if _without_source_artifact_hashes(
                raw_value
            ) != _without_source_artifact_hashes(historical_value):
                raise ValueError(
                    "legacy effective view historical mode requires the exact C1 reference output"
                )
        effective_value = copy.deepcopy(raw_value)
        graph = cast(dict[str, object], effective_value["instance_graph"])
        assertions = cast(list[dict[str, object]], graph["assertions"])
        source_hashes = {
            item.evidence_id: cast(str, item.provenance.source_artifact_hash)
            for item in resolved.evidence
        }
        inserted: list[str] = []
        for assertion_index, assertion in enumerate(assertions):
            provenance = cast(list[dict[str, object]], assertion["provenance"])
            cited = assertion.get("evidence_ids")
            cited_ids = tuple(cast(Sequence[str], cited))
            observed_ids = tuple(cast(str, item.get("evidence_id")) for item in provenance)
            provenance_ids = tuple(cast(str, item.get("provenance_id")) for item in provenance)
            if (
                len(observed_ids) != len(set(observed_ids))
                or set(observed_ids) != set(cited_ids)
                or len(provenance_ids) != len(set(provenance_ids))
            ):
                raise ValueError("legacy effective view requires exact raw provenance cardinality")
            for provenance_index, item in enumerate(provenance):
                evidence_id = item.get("evidence_id")
                if (
                    not isinstance(evidence_id, str)
                    or evidence_id not in source_hashes
                    or item.get("locator") != evidence_id
                    or not isinstance(item.get("confidence"), (int, float))
                    or cast(float, item["confidence"]) > 1.0
                ):
                    raise ValueError("legacy effective view rejects invalid raw provenance")
                observed_source_hash = item.get("source_artifact_hash")
                if observed_source_hash is not None and observed_source_hash != source_hashes[
                    evidence_id
                ]:
                    raise ValueError("legacy effective view rejects a conflicting source hash")
                if (
                    observed_source_hash is None
                ):
                    item["source_artifact_hash"] = source_hashes[evidence_id]
                    inserted.append(
                        "instance_graph.assertions."
                        f"{assertion_index}.provenance.{provenance_index}."
                        "source_artifact_hash"
                    )
        effective = OntologyDraft.model_validate(effective_value)
        return EffectivePhase1Draft(
            raw_draft=raw_draft,
            effective_draft=effective,
            raw_draft_sha256=raw_draft.content_hash,
            effective_draft_sha256=effective.content_hash,
            inserted_source_hash_paths=tuple(inserted),
        )


__all__ = [
    "BRIDGE_REVISION",
    "CERTIFICATE_RELATIVE_PATH",
    "EffectivePhase1Draft",
    "Phase1LegacyEvidenceBinding",
    "Phase1LegacyEvidenceProvenanceBridge",
    "Phase1LegacyEvidenceProvenanceCertificate",
    "Phase1LegacyFixtureBinding",
    "Phase1LegacySourceMetadata",
    "ResolvedPhase1LegacyEvidence",
]
