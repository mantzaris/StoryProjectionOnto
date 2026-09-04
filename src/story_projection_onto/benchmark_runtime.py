"""Process-minimal loading for sealed synthetic benchmark requests.

Only this module and the neutral contracts module are needed in a model worker.
The offline compiler writes one flat directory per runtime job. A pre-query
directory contains evidence only; a query-time directory contains the same
evidence object plus exactly one revealed query. Scorer routing and gold are
never named by a runtime manifest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    EvidenceRecord,
    EvidenceSnapshot,
    ImmutableRecord,
    ModelVisibleEvidenceRecord,
    ModelVisibleQueryContext,
    Sha256Digest,
    canonical_sha256,
    to_model_visible_evidence,
)


class GoldFirewallError(PermissionError):
    """Raised when a model-side operation can reach non-staged material."""


class RuntimeStageKind(StrEnum):
    NEUTRAL_EVIDENCE = "neutral_evidence"
    PREQUERY_EVIDENCE = "prequery_evidence"
    QUERY_REVEALED = "query_revealed"


class ModelEligibleWorldArtifact(ImmutableRecord):
    """Opaque evidence-only artifact; deliberately incapable of carrying a query."""

    artifact_id: str
    snapshot: EvidenceSnapshot
    evidence: tuple[ModelVisibleEvidenceRecord, ...]

    @model_validator(mode="after")
    def snapshot_matches_evidence(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != self.snapshot.eligible_evidence_ids:
            raise ValueError("model evidence order must equal the sealed snapshot")
        return self


class NeutralEvidenceArtifact(ImmutableRecord):
    """Query-blind full evidence for CPU conditions and deterministic validators."""

    artifact_id: str
    snapshot: EvidenceSnapshot
    evidence: tuple[EvidenceRecord, ...]

    @model_validator(mode="after")
    def snapshot_and_release_match_evidence(self) -> Self:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if evidence_ids != self.snapshot.eligible_evidence_ids:
            raise ValueError("neutral evidence order must equal the sealed snapshot")
        if any(item.release_class is not self.snapshot.release_class for item in self.evidence):
            raise ValueError("neutral evidence and snapshot release classes differ")
        return self


class EvidenceProjectionEquivalenceCertificate(ImmutableRecord):
    """Hash-bound proof that full evidence projects to one model-visible artifact."""

    certificate_id: str
    neutral_evidence_artifact_hash: Sha256Digest
    model_visible_evidence_artifact_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    ordered_full_evidence_hash: Sha256Digest
    ordered_model_visible_evidence_hash: Sha256Digest
    evidence_count: int = Field(gt=0)
    projection_rule: Literal["contracts.to_model_visible_evidence/v1"] = (
        "contracts.to_model_visible_evidence/v1"
    )


class QueryRevealArtifact(ImmutableRecord):
    """Exactly one allowlisted query revealed after its evidence seal."""

    reveal_id: str
    evidence_artifact_hash: Sha256Digest
    query: ModelVisibleQueryContext
    revealed_at: AwareDatetime


class ModelEligibleParaphraseArtifact(ImmutableRecord):
    contexts: tuple[ModelVisibleQueryContext, ...]

    @model_validator(mode="after")
    def exactly_twelve(self) -> Self:
        if len(self.contexts) != 12:
            raise ValueError("the paraphrase artifact must contain twelve contexts")
        return self


class RuntimeStagingManifest(ImmutableRecord):
    """Complete allowlist for one physically bounded model-worker directory."""

    stage_id: str
    stage_kind: RuntimeStageKind
    artifact_hashes: tuple[Sha256Digest, ...]
    file_names: tuple[str, ...]
    manifest_file_name: Literal["manifest.json"] = "manifest.json"

    @model_validator(mode="after")
    def exact_stage_shape(self) -> Self:
        if self.stage_kind is RuntimeStageKind.NEUTRAL_EVIDENCE:
            expected_names = ("neutral_evidence.json", "equivalence.json")
        elif self.stage_kind is RuntimeStageKind.PREQUERY_EVIDENCE:
            expected_names = ("evidence.json",)
        else:
            expected_names = ("evidence.json", "query.json")
        if self.file_names != expected_names:
            raise ValueError(
                f"{self.stage_kind.value} stage must contain exactly {expected_names!r}"
            )
        if len(self.artifact_hashes) != len(self.file_names):
            raise ValueError("runtime staging hashes and file names must align")
        return self


_FORBIDDEN_KEYS = frozenset(
    {
        "world_id",
        "split",
        "pair_id",
        "contrast_pair",
        "expected_effect",
        "expected_signature",
        "gold",
        "gold_projection_id",
        "scorer_namespace",
        "rare",
        "pivotal",
        "community",
        "review_status",
        "difficulty",
    }
)
_FORBIDDEN_VALUE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"syn[-_.]test",
        r"syn[-_.]dev",
        r"held[-_ ]?out",
        r"\bdevelopment\b",
        r"scorer[-_ ]?only",
        r"gold[-_.:]",
        r"expected[-_ ]?effect",
        r"contrast[-_ ]?pair",
    )
)


def scan_model_payload(value: Any, path: str = "") -> None:
    """Reject scorer metadata in both keys and string values."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            child_path = f"{path}.{key}" if path else str(key)
            if key_text in _FORBIDDEN_KEYS or any(
                marker in key_text for marker in ("scorer", "expected_effect", "gold_")
            ):
                raise GoldFirewallError(
                    f"model payload contains scorer/experiment key {key!r} at {child_path}"
                )
            scan_model_payload(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            scan_model_payload(child, f"{path}[{index}]")
    elif isinstance(value, str):
        for pattern in _FORBIDDEN_VALUE_PATTERNS:
            if pattern.search(value):
                raise GoldFirewallError(
                    f"model payload contains split/scorer value at {path}: {value!r}"
                )


def preconstruction_request_payload(artifact: ModelEligibleWorldArtifact) -> dict[str, Any]:
    """Build the evidence-only body permitted before query reveal."""

    payload: dict[str, Any] = {
        "snapshot_hash": artifact.snapshot.content_hash,
        "evidence": [item.model_dump(mode="json") for item in artifact.evidence],
    }
    scan_model_payload(payload)
    return payload


def verify_neutral_evidence_projection(
    neutral: NeutralEvidenceArtifact,
    model_visible: ModelEligibleWorldArtifact,
    certificate: EvidenceProjectionEquivalenceCertificate,
) -> None:
    """Recompute the only permitted full-to-model evidence projection."""

    projected_evidence = tuple(to_model_visible_evidence(item) for item in neutral.evidence)
    expected_model = ModelEligibleWorldArtifact(
        artifact_id=model_visible.artifact_id,
        snapshot=neutral.snapshot,
        evidence=projected_evidence,
    )
    expected = (
        neutral.content_hash,
        model_visible.content_hash,
        neutral.snapshot.content_hash,
        canonical_sha256(neutral.evidence),
        canonical_sha256(projected_evidence),
        len(neutral.evidence),
    )
    observed = (
        certificate.neutral_evidence_artifact_hash,
        certificate.model_visible_evidence_artifact_hash,
        certificate.snapshot_hash,
        certificate.ordered_full_evidence_hash,
        certificate.ordered_model_visible_evidence_hash,
        certificate.evidence_count,
    )
    if expected_model != model_visible or observed != expected:
        raise GoldFirewallError(
            "neutral evidence does not project exactly to the staged model-visible artifact"
        )


def build_evidence_projection_equivalence_certificate(
    neutral: NeutralEvidenceArtifact,
    model_visible: ModelEligibleWorldArtifact,
    *,
    certificate_id: str,
) -> EvidenceProjectionEquivalenceCertificate:
    """Construct and immediately verify a deterministic projection certificate."""

    certificate = EvidenceProjectionEquivalenceCertificate(
        certificate_id=certificate_id,
        neutral_evidence_artifact_hash=neutral.content_hash,
        model_visible_evidence_artifact_hash=model_visible.content_hash,
        snapshot_hash=neutral.snapshot.content_hash,
        ordered_full_evidence_hash=canonical_sha256(neutral.evidence),
        ordered_model_visible_evidence_hash=canonical_sha256(
            tuple(to_model_visible_evidence(item) for item in neutral.evidence)
        ),
        evidence_count=len(neutral.evidence),
    )
    verify_neutral_evidence_projection(neutral, model_visible, certificate)
    return certificate


def model_request_payload(
    artifact: ModelEligibleWorldArtifact,
    reveal: QueryRevealArtifact,
) -> dict[str, Any]:
    """Build the exact single-query request body consumed after reveal."""

    if reveal.evidence_artifact_hash != artifact.content_hash:
        raise GoldFirewallError("query reveal names a different evidence artifact")
    if reveal.query.spoiler_horizon != artifact.snapshot.horizon:
        raise GoldFirewallError("query reveal horizon differs from its sealed evidence")
    payload = preconstruction_request_payload(artifact)
    payload["query"] = reveal.query.model_dump(mode="json")
    scan_model_payload(payload)
    return payload


def _resolve_flat_runtime_file(path: Path, staging_root: Path) -> Path:
    if staging_root.is_symlink():
        raise GoldFirewallError("runtime staging root cannot be a symlink")
    root = staging_root.resolve(strict=True)
    if path.is_symlink():
        raise GoldFirewallError("runtime artifact cannot be a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise GoldFirewallError("model artifact is outside the runtime staging root") from error
    if len(relative.parts) != 1 or relative.suffix != ".json":
        raise GoldFirewallError("runtime loader permits only a flat staged JSON object")
    return resolved


def _assert_exact_stage_directory(staging_root: Path, manifest: RuntimeStagingManifest) -> None:
    root = staging_root.resolve(strict=True)
    expected = {*manifest.file_names, manifest.manifest_file_name}
    actual = {item.name for item in root.iterdir()}
    if actual != expected or any(item.is_dir() or item.is_symlink() for item in root.iterdir()):
        raise GoldFirewallError(
            f"runtime stage is not exact; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    manifest_path = root / manifest.manifest_file_name
    parsed = RuntimeStagingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if parsed != manifest:
        raise GoldFirewallError("runtime manifest bytes disagree with the supplied manifest")


def _verified_bytes(
    staging_root: Path,
    manifest: RuntimeStagingManifest,
    file_name: str,
) -> bytes:
    safe = _resolve_flat_runtime_file(staging_root / file_name, staging_root)
    try:
        index = manifest.file_names.index(file_name)
    except ValueError as error:
        raise GoldFirewallError("runtime artifact is absent from the staging allowlist") from error
    payload = safe.read_bytes()
    embedded_hash = json.loads(payload)["content_hash"]
    if embedded_hash != manifest.artifact_hashes[index]:
        raise GoldFirewallError("runtime artifact hash disagrees with staging manifest")
    return payload


def load_staged_world(
    path: Path,
    staging_root: Path,
    manifest: RuntimeStagingManifest,
) -> ModelEligibleWorldArtifact:
    """Load one evidence-only pre-query stage."""

    if manifest.stage_kind is not RuntimeStageKind.PREQUERY_EVIDENCE:
        raise GoldFirewallError("evidence-only loader rejects a query-revealed stage")
    if Path(path).name != "evidence.json":
        raise GoldFirewallError("pre-query loader accepts only evidence.json")
    _assert_exact_stage_directory(staging_root, manifest)
    artifact = ModelEligibleWorldArtifact.model_validate_json(
        _verified_bytes(staging_root, manifest, "evidence.json")
    )
    scan_model_payload(artifact.model_dump(mode="json"))
    preconstruction_request_payload(artifact)
    return artifact


def load_staged_neutral_evidence(
    staging_root: Path,
    condition_input_root: Path,
    manifest: RuntimeStagingManifest,
    model_visible: ModelEligibleWorldArtifact,
) -> tuple[NeutralEvidenceArtifact, EvidenceProjectionEquivalenceCertificate]:
    """Load full query-blind evidence without importing compiler or scorer code."""

    if manifest.stage_kind is not RuntimeStageKind.NEUTRAL_EVIDENCE:
        raise GoldFirewallError("neutral evidence loader requires a neutral evidence stage")
    if condition_input_root.is_symlink() or staging_root.is_symlink():
        raise GoldFirewallError("neutral condition-input paths cannot be symlinks")
    neutral_root = condition_input_root.resolve(strict=True)
    if neutral_root.name != "neutral_evidence" or neutral_root.parent.name != "condition_inputs":
        raise GoldFirewallError(
            "neutral evidence must remain below condition_inputs/neutral_evidence"
        )
    resolved_stage = staging_root.resolve(strict=True)
    try:
        relative_stage = resolved_stage.relative_to(neutral_root)
    except ValueError as error:
        raise GoldFirewallError(
            "neutral evidence stage escaped its condition-input root"
        ) from error
    if len(relative_stage.parts) != 1 or not relative_stage.parts[0].startswith("neutral_"):
        raise GoldFirewallError("neutral evidence input must be one opaque isolated stage")
    _assert_exact_stage_directory(staging_root, manifest)
    neutral = NeutralEvidenceArtifact.model_validate_json(
        _verified_bytes(staging_root, manifest, "neutral_evidence.json")
    )
    certificate = EvidenceProjectionEquivalenceCertificate.model_validate_json(
        _verified_bytes(staging_root, manifest, "equivalence.json")
    )
    scan_model_payload(neutral.model_dump(mode="json"))
    verify_neutral_evidence_projection(neutral, model_visible, certificate)
    return neutral, certificate


def load_staged_query(
    staging_root: Path,
    manifest: RuntimeStagingManifest,
) -> tuple[ModelEligibleWorldArtifact, QueryRevealArtifact]:
    """Load a stage containing exactly one evidence object and one revealed query."""

    if manifest.stage_kind is not RuntimeStageKind.QUERY_REVEALED:
        raise GoldFirewallError("query loader requires a query-revealed stage")
    _assert_exact_stage_directory(staging_root, manifest)
    evidence = ModelEligibleWorldArtifact.model_validate_json(
        _verified_bytes(staging_root, manifest, "evidence.json")
    )
    reveal = QueryRevealArtifact.model_validate_json(
        _verified_bytes(staging_root, manifest, "query.json")
    )
    scan_model_payload(evidence.model_dump(mode="json"))
    scan_model_payload(reveal.model_dump(mode="json"))
    model_request_payload(evidence, reveal)
    return evidence, reveal


def assert_runtime_manifest_matches(
    artifacts: Sequence[ImmutableRecord], manifest: RuntimeStagingManifest
) -> None:
    if tuple(item.content_hash for item in artifacts) != manifest.artifact_hashes:
        raise GoldFirewallError("runtime manifest artifact ordering or hashes changed")
    if canonical_sha256(tuple(manifest.file_names)) == canonical_sha256(()) and artifacts:
        raise GoldFirewallError("nonempty runtime staging cannot use an empty file list")
