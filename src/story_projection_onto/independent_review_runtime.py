"""Fail-closed completion of the external synthetic-gold review lifecycle.

This module is scorer-side only.  It never creates review judgments and it does
not make scorer records model-visible.  It validates externally authored review
and adjudication records, derives reviewed artifacts, and atomically materializes
the lineage outside the immutable benchmark tree.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.contracts import (
    GoldAdjudicationStatus,
    GoldAlternativeSet,
    GoldContextualProjection,
    GoldReviewStatus,
    Identifier,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
)
from story_projection_onto.synthetic_benchmark import (
    AdjudicationDisposition,
    BlindIndependentReviewPackage,
    FinalReviewedSeal,
    HeldOutDraftSeal,
    IndependentReviewGateError,
    IndependentReviewResponse,
    ReviewAdjudication,
    ReviewedProjectionArtifact,
    ReviewLifecycleError,
    ReviewProjectionBinding,
    ReviewProjectionBindingManifest,
    ScorerWorldArtifact,
    bind_review_response,
    finalize_reviewed_seal,
    require_independent_review_complete,
    reviewed_artifact_from_original,
    reviewed_semantic_hash,
    validate_adjudication,
)

DEFAULT_BENCHMARK_ROOT = Path("data/synthetic")
DEFAULT_COMPLETION_ROOT = Path(
    "artifacts/restricted/scorer_only/independent_review"
)
DEFAULT_PUBLIC_REVIEWED_GOLD_ROOT = Path(
    "artifacts/public/scorer_only/final_reviewed_gold"
)


class ReviewCompletionError(RuntimeError):
    """An external review cannot be safely validated or materialized."""


class MethodologicalAmendmentRequired(ReviewCompletionError):
    """The registered workflow cannot resolve an EXCLUDE decision."""


class AmendedProjectionSubmission(ImmutableRecord):
    """Explicit replacement semantics supplied by the adjudication process."""

    blind_projection_id: Identifier
    source_gold_projection_hash: Sha256Digest
    source_alternative_set_hash: Sha256Digest
    final_semantic_hash: Sha256Digest
    gold_projection: GoldContextualProjection
    alternatives: GoldAlternativeSet

    @model_validator(mode="after")
    def projection_ids_match(self) -> Self:
        if self.gold_projection.gold_projection_id != self.alternatives.gold_projection_id:
            raise ValueError("amended gold and alternative set name different projections")
        return self


class ReviewAmendmentBundle(ImmutableRecord):
    """Separately supplied replacements for projections adjudicated AMEND."""

    package_hash: Sha256Digest
    response_hash: Sha256Digest
    adjudication_hash: Sha256Digest
    submitted_at: AwareDatetime
    submissions: tuple[AmendedProjectionSubmission, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def unique_projection_ids(self) -> Self:
        ids = [item.blind_projection_id for item in self.submissions]
        if len(ids) != len(set(ids)):
            raise ValueError("amendment submissions must be unique by blind projection")
        return self


class ReviewedArtifactManifestEntry(ImmutableRecord):
    blind_projection_id: Identifier
    world_id: Identifier
    query_id: Identifier
    artifact_file: Identifier
    artifact_hash: Sha256Digest
    source_gold_projection_hash: Sha256Digest
    source_alternative_set_hash: Sha256Digest
    final_semantic_hash: Sha256Digest


class IndependentReviewCompletionManifest(ImmutableRecord):
    """Self-hashed, path-portable proof of a fully reproduced completion."""

    manifest_id: Identifier
    lifecycle_state: Literal["reviewed_adjudicated_frozen"] = (
        "reviewed_adjudicated_frozen"
    )
    package_hash: Sha256Digest
    binding_manifest_hash: Sha256Digest
    draft_seal_hash: Sha256Digest
    response_hash: Sha256Digest
    adjudication_hash: Sha256Digest
    final_seal_hash: Sha256Digest
    response_file: Literal["response.json"] = "response.json"
    adjudication_file: Literal["adjudication.json"] = "adjudication.json"
    final_seal_file: Literal["final_seal.json"] = "final_seal.json"
    reviewed_artifacts: tuple[ReviewedArtifactManifestEntry, ...]
    review_item_count: Literal[72] = 72
    reviewed_projection_count: Literal[9] = 9
    held_out_launch_authorized: Literal[True] = True
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def exact_artifact_inventory(self) -> Self:
        if len(self.reviewed_artifacts) != 9:
            raise ValueError("completion manifest must bind exactly nine reviewed artifacts")
        if len({item.blind_projection_id for item in self.reviewed_artifacts}) != 9:
            raise ValueError("completion manifest projection IDs must be unique")
        expected = {
            f"reviewed/{item.blind_projection_id}.json"
            for item in self.reviewed_artifacts
        }
        actual = {item.artifact_file for item in self.reviewed_artifacts}
        if actual != expected:
            raise ValueError("reviewed artifact filenames must derive from blind IDs")
        for name in actual:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
                raise ValueError("completion manifest contains a non-portable logical path")
        return self


class PublicReviewedArtifactManifestEntry(ImmutableRecord):
    """Public file and semantic lineage for one finalized reviewed projection."""

    blind_projection_id: Identifier
    world_id: Identifier
    query_id: Identifier
    artifact_file: Identifier
    artifact_file_sha256: Sha256Digest
    artifact_hash: Sha256Digest
    source_gold_projection_hash: Sha256Digest
    source_alternative_set_hash: Sha256Digest
    source_semantic_hash: Sha256Digest
    final_semantic_hash: Sha256Digest
    gold_projection_hash: Sha256Digest
    alternative_set_hash: Sha256Digest
    amended: bool

    @model_validator(mode="after")
    def path_and_amendment_are_derived(self) -> Self:
        if self.artifact_file != f"reviewed/{self.blind_projection_id}.json":
            raise ValueError("public reviewed artifact path must derive from its blind ID")
        if self.amended != (self.source_semantic_hash != self.final_semantic_hash):
            raise ValueError("public reviewed artifact amendment flag is not derived")
        return self


class PublicReviewedGoldPublicationManifest(ImmutableRecord):
    """Public-safe proof that the released gold is the completed reviewed gold."""

    manifest_id: Identifier
    lifecycle_state: Literal["final_reviewed_gold_publication"] = (
        "final_reviewed_gold_publication"
    )
    source_completion_manifest_hash: Sha256Digest
    source_completion_manifest_file_sha256: Sha256Digest
    draft_seal_hash: Sha256Digest
    package_hash: Sha256Digest
    binding_manifest_hash: Sha256Digest
    final_seal_hash: Sha256Digest
    final_seal_file: Literal["final_seal.json"] = "final_seal.json"
    final_seal_file_sha256: Sha256Digest
    reviewed_artifacts: tuple[PublicReviewedArtifactManifestEntry, ...]
    review_item_count: Literal[72] = 72
    reviewed_projection_count: Literal[9] = 9
    amendment_count: int = Field(ge=0, le=9)
    held_out_launch_authorized: Literal[True] = True
    condition_outputs_generated_before_review: Literal[False] = False
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC

    @model_validator(mode="after")
    def exact_public_inventory(self) -> Self:
        if len(self.reviewed_artifacts) != 9:
            raise ValueError("public reviewed-gold manifest requires exactly nine artifacts")
        identifiers = tuple(item.blind_projection_id for item in self.reviewed_artifacts)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("public reviewed-gold artifacts must be unique and sorted")
        if self.amendment_count != sum(item.amended for item in self.reviewed_artifacts):
            raise ValueError("public reviewed-gold amendment count is not derived")
        return self


@dataclass(frozen=True)
class ReviewCompletion:
    package: BlindIndependentReviewPackage
    bindings: ReviewProjectionBindingManifest
    draft: HeldOutDraftSeal
    response: IndependentReviewResponse
    adjudication: ReviewAdjudication
    reviewed_artifacts: tuple[ReviewedProjectionArtifact, ...]
    final_seal: FinalReviewedSeal
    manifest: IndependentReviewCompletionManifest


@dataclass(frozen=True)
class PublicReviewedGoldPublication:
    manifest: PublicReviewedGoldPublicationManifest
    reviewed_artifacts: tuple[ReviewedProjectionArtifact, ...]
    final_seal: FinalReviewedSeal


def _assert_no_symlink(path: Path, *, must_exist: bool) -> None:
    candidate = path.absolute()
    existing = candidate if candidate.exists() or candidate.is_symlink() else candidate.parent
    while True:
        if existing.is_symlink():
            raise ReviewCompletionError(f"symlinked path is forbidden: {path}")
        if existing.parent == existing:
            break
        existing = existing.parent
    if must_exist:
        if not path.exists():
            raise ReviewCompletionError(f"required review input is missing: {path}")
        if not path.is_file():
            raise ReviewCompletionError(f"review input is not a regular file: {path}")


def _read_model(path: Path, model_type: type[ImmutableRecord]) -> ImmutableRecord:
    _assert_no_symlink(path, must_exist=True)
    try:
        raw = path.read_bytes()
        if len(raw) > 100 * 1024 * 1024:
            raise ReviewCompletionError(f"review input exceeds 100 MiB: {path}")
        return model_type.model_validate_json(raw)
    except ReviewCompletionError:
        raise
    except Exception as error:
        raise ReviewCompletionError(f"invalid {model_type.__name__} at {path}: {error}") from error


def _without_hashes(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_hashes(child) for child in value]
    return value


def _load_frozen_inputs(
    benchmark_root: Path,
) -> tuple[
    BlindIndependentReviewPackage,
    ReviewProjectionBindingManifest,
    HeldOutDraftSeal,
    dict[str, tuple[GoldContextualProjection, GoldAlternativeSet]],
]:
    benchmark_root = benchmark_root.absolute()
    _assert_no_symlink(benchmark_root, must_exist=False)
    package = _read_model(
        benchmark_root / "scorer_only/review/blind_review_package.json",
        BlindIndependentReviewPackage,
    )
    bindings = _read_model(
        benchmark_root / "scorer_only/review/scorer_bindings.json",
        ReviewProjectionBindingManifest,
    )
    draft = _read_model(
        benchmark_root / "scorer_only/held_out/draft_seal.json", HeldOutDraftSeal
    )
    assert isinstance(package, BlindIndependentReviewPackage)
    assert isinstance(bindings, ReviewProjectionBindingManifest)
    assert isinstance(draft, HeldOutDraftSeal)
    if draft.review_package_hash != package.content_hash:
        raise ReviewCompletionError("draft seal does not bind the exact review package")
    if draft.review_binding_manifest_hash != bindings.content_hash:
        raise ReviewCompletionError("draft seal does not bind the exact scorer bindings")
    if bindings.package_hash != package.content_hash:
        raise ReviewCompletionError("scorer bindings do not bind the exact review package")

    originals: dict[str, tuple[GoldContextualProjection, GoldAlternativeSet]] = {}
    scorer_cache: dict[str, ScorerWorldArtifact] = {}
    for binding in bindings.entries:
        scorer = scorer_cache.get(binding.world_id)
        if scorer is None:
            loaded = _read_model(
                benchmark_root / f"scorer_only/held_out/{binding.world_id}.json",
                ScorerWorldArtifact,
            )
            assert isinstance(loaded, ScorerWorldArtifact)
            scorer = loaded
            scorer_cache[binding.world_id] = scorer
        gold = next(
            (item for item in scorer.gold_projections if item.query_id == binding.query_id), None
        )
        if gold is None:
            raise ReviewCompletionError("binding query is absent from its sealed scorer world")
        alternatives = next(
            (
                item
                for item in scorer.alternatives
                if item.gold_projection_id == gold.gold_projection_id
            ),
            None,
        )
        if alternatives is None:
            raise ReviewCompletionError("binding alternative set is absent from scorer world")
        if (
            gold.content_hash != binding.source_gold_projection_hash
            or alternatives.content_hash != binding.source_alternative_set_hash
            or reviewed_semantic_hash(gold, alternatives) != binding.source_semantic_hash
            or gold.world_id != binding.world_id
        ):
            raise ReviewCompletionError("sealed scorer source differs from its review binding")
        originals[binding.blind_projection_id] = (gold, alternatives)
    return package, bindings, draft, originals


def _reviewed_amendment(
    *,
    binding: ReviewProjectionBinding,
    response: IndependentReviewResponse,
    adjudication: ReviewAdjudication,
    submission: AmendedProjectionSubmission,
    source_gold: GoldContextualProjection,
    source_alternatives: GoldAlternativeSet,
) -> ReviewedProjectionArtifact:
    if (
        submission.blind_projection_id != binding.blind_projection_id
        or submission.source_gold_projection_hash != binding.source_gold_projection_hash
        or submission.source_alternative_set_hash != binding.source_alternative_set_hash
        or submission.gold_projection.world_id != binding.world_id
        or submission.gold_projection.query_id != binding.query_id
        or submission.gold_projection.gold_projection_id != source_gold.gold_projection_id
        or submission.alternatives.alternative_set_id
        != source_alternatives.alternative_set_id
    ):
        raise ReviewCompletionError("amendment identifiers or source hashes changed")
    gold_values = _without_hashes(submission.gold_projection.model_dump(mode="python"))
    assert isinstance(gold_values, dict)
    gold_values.update(
        review_status=GoldReviewStatus.DISAGREEMENT_LOGGED.value,
        adjudication_status=GoldAdjudicationStatus.ADJUDICATED.value,
        independent_review_record_hash=response.content_hash,
        adjudication_record_hash=adjudication.content_hash,
    )
    alt_values = _without_hashes(submission.alternatives.model_dump(mode="python"))
    assert isinstance(alt_values, dict)
    alt_values.update(
        review_status=GoldReviewStatus.DISAGREEMENT_LOGGED.value,
        adjudication_status=GoldAdjudicationStatus.ADJUDICATED.value,
    )
    amended_gold = GoldContextualProjection.model_validate(gold_values)
    amended_alt = GoldAlternativeSet.model_validate(alt_values)
    if amended_gold.gold_projection_id != amended_alt.gold_projection_id:
        raise ReviewCompletionError("amended projection and alternatives disagree on gold ID")
    final_hash = reviewed_semantic_hash(amended_gold, amended_alt)
    if final_hash != submission.final_semantic_hash:
        raise ReviewCompletionError("submitted final semantic hash does not reproduce")
    if final_hash == binding.source_semantic_hash:
        raise ReviewCompletionError("AMEND must supply semantics distinct from the sealed source")
    return ReviewedProjectionArtifact(
        blind_projection_id=binding.blind_projection_id,
        source_gold_projection_hash=binding.source_gold_projection_hash,
        source_alternative_set_hash=binding.source_alternative_set_hash,
        response_hash=response.content_hash,
        adjudication_hash=adjudication.content_hash,
        final_semantic_hash=final_hash,
        gold_projection=amended_gold,
        alternatives=amended_alt,
    )


def prepare_review_completion(
    *,
    benchmark_root: Path,
    response_path: Path,
    adjudication_path: Path,
    amendment_path: Path | None = None,
) -> ReviewCompletion:
    """Validate complete lineage and construct, but do not write, completion records."""

    if response_path.absolute() == adjudication_path.absolute():
        raise ReviewCompletionError("review response and adjudication must be separate records")
    package, bindings, draft, originals = _load_frozen_inputs(benchmark_root)
    response = _read_model(response_path, IndependentReviewResponse)
    adjudication = _read_model(adjudication_path, ReviewAdjudication)
    assert isinstance(response, IndependentReviewResponse)
    assert isinstance(adjudication, ReviewAdjudication)
    try:
        bind_review_response(package, response)
        validate_adjudication(package, response, adjudication)
    except ReviewLifecycleError as error:
        raise ReviewCompletionError(str(error)) from error
    if adjudication.adjudicated_at < response.reviewed_at:
        raise ReviewCompletionError("adjudication timestamp precedes external review")

    amendment_items = {
        item.review_item_id: item
        for item in adjudication.items
        if item.disposition is AdjudicationDisposition.AMEND
    }
    if any(
        item.disposition is AdjudicationDisposition.EXCLUDE for item in adjudication.items
    ):
        raise MethodologicalAmendmentRequired(
            "EXCLUDE requires an explicit methodological amendment; held-out launch remains blocked"
        )
    bundle: ReviewAmendmentBundle | None = None
    if amendment_path is not None:
        loaded_bundle = _read_model(amendment_path, ReviewAmendmentBundle)
        assert isinstance(loaded_bundle, ReviewAmendmentBundle)
        bundle = loaded_bundle
        if (
            bundle.package_hash != package.content_hash
            or bundle.response_hash != response.content_hash
            or bundle.adjudication_hash != adjudication.content_hash
        ):
            raise ReviewCompletionError("amendment bundle lineage does not match review records")
        if bundle.submitted_at < adjudication.adjudicated_at:
            raise ReviewCompletionError("amendment bundle timestamp precedes adjudication")

    package_projection_items = {
        projection.blind_projection_id: {item.review_item_id for item in projection.review_items}
        for world in package.worlds
        for projection in world.projections
    }
    amended_projection_ids = {
        blind_id
        for blind_id, item_ids in package_projection_items.items()
        if set(amendment_items).intersection(item_ids)
    }
    submissions = (
        {item.blind_projection_id: item for item in bundle.submissions} if bundle else {}
    )
    if set(submissions) != amended_projection_ids:
        raise ReviewCompletionError(
            "amendment bundle must cover exactly every projection adjudicated AMEND"
        )

    artifacts: list[ReviewedProjectionArtifact] = []
    for binding in bindings.entries:
        gold, alternatives = originals[binding.blind_projection_id]
        submission = submissions.get(binding.blind_projection_id)
        if submission is None:
            artifact = reviewed_artifact_from_original(
                binding=binding,
                package=package,
                response=response,
                adjudication=adjudication,
                gold=gold,
                alternatives=alternatives,
            )
        else:
            artifact = _reviewed_amendment(
                binding=binding,
                response=response,
                adjudication=adjudication,
                submission=submission,
                source_gold=gold,
                source_alternatives=alternatives,
            )
            expected_hashes = {
                item.amended_artifact_hash
                for item_id, item in amendment_items.items()
                if item_id in package_projection_items[binding.blind_projection_id]
            }
            if expected_hashes != {artifact.final_semantic_hash}:
                raise ReviewCompletionError(
                    "all AMEND decisions for a projection must bind its final semantic hash"
                )
        artifacts.append(artifact)

    try:
        final_seal = finalize_reviewed_seal(
            draft, package, bindings, response, adjudication, artifacts
        )
        require_independent_review_complete(
            draft, package, bindings, response, adjudication, artifacts, final_seal
        )
    except (ReviewLifecycleError, IndependentReviewGateError) as error:
        raise ReviewCompletionError(str(error)) from error
    artifact_entries = tuple(
        ReviewedArtifactManifestEntry(
            blind_projection_id=artifact.blind_projection_id,
            world_id=binding.world_id,
            query_id=binding.query_id,
            artifact_file=f"reviewed/{artifact.blind_projection_id}.json",
            artifact_hash=artifact.content_hash,
            source_gold_projection_hash=artifact.source_gold_projection_hash,
            source_alternative_set_hash=artifact.source_alternative_set_hash,
            final_semantic_hash=artifact.final_semantic_hash,
        )
        for binding, artifact in zip(bindings.entries, artifacts, strict=True)
    )
    manifest = IndependentReviewCompletionManifest(
        manifest_id=f"review-completion-{final_seal.content_hash[:24]}",
        package_hash=package.content_hash,
        binding_manifest_hash=bindings.content_hash,
        draft_seal_hash=draft.content_hash,
        response_hash=response.content_hash,
        adjudication_hash=adjudication.content_hash,
        final_seal_hash=final_seal.content_hash,
        reviewed_artifacts=artifact_entries,
    )
    return ReviewCompletion(
        package=package,
        bindings=bindings,
        draft=draft,
        response=response,
        adjudication=adjudication,
        reviewed_artifacts=tuple(artifacts),
        final_seal=final_seal,
        manifest=manifest,
    )


def _completion_payloads(completion: ReviewCompletion) -> dict[str, bytes]:
    payloads = {
        "response.json": completion.response.to_canonical_json().encode() + b"\n",
        "adjudication.json": completion.adjudication.to_canonical_json().encode() + b"\n",
        "final_seal.json": completion.final_seal.to_canonical_json().encode() + b"\n",
        "completion_manifest.json": completion.manifest.to_canonical_json().encode() + b"\n",
    }
    payloads.update(
        {
            f"reviewed/{item.blind_projection_id}.json": (
                item.to_canonical_json().encode() + b"\n"
            )
            for item in completion.reviewed_artifacts
        }
    )
    return payloads


def _existing_files(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReviewCompletionError(f"symlink in completion tree is forbidden: {path}")
        if path.is_file():
            names.add(path.relative_to(root).as_posix())
        elif path.is_dir() and path != root / "reviewed":
            raise ReviewCompletionError(f"unexpected directory in completion tree: {path}")
    return names


def _assert_restricted_permissions(root: Path, relative_files: Sequence[str]) -> None:
    """Reject persisted scorer gold that is readable outside its owning account."""

    expected_modes = {root: 0o700, root / "reviewed": 0o700}
    expected_modes.update({root / relative: 0o600 for relative in relative_files})
    for path, expected_mode in expected_modes.items():
        try:
            observed_mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        except OSError as error:
            raise ReviewCompletionError(
                f"cannot inspect restricted review permissions: {path}: {error}"
            ) from error
        if observed_mode != expected_mode:
            raise ReviewCompletionError(
                "restricted review permission drift: "
                f"{path} has {observed_mode:#05o}, expected {expected_mode:#05o}"
            )


def materialize_review_completion(
    completion: ReviewCompletion, output_root: Path = DEFAULT_COMPLETION_ROOT
) -> Literal["created", "already_exact"]:
    """Atomically write a new tree, or accept only a byte-identical complete tree."""

    output_root = output_root.absolute()
    _assert_no_symlink(output_root, must_exist=False)
    payloads = _completion_payloads(completion)
    if output_root.exists():
        if not output_root.is_dir():
            raise ReviewCompletionError("completion output exists but is not a directory")
        actual = _existing_files(output_root)
        if actual != set(payloads):
            raise ReviewCompletionError("existing completion tree is partial or unexpected")
        for relative, expected in payloads.items():
            if (output_root / relative).read_bytes() != expected:
                raise ReviewCompletionError(f"existing completion artifact drift: {relative}")
        _assert_restricted_permissions(output_root, tuple(payloads))
        return "already_exact"

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / (
        f".{output_root.name}.staging-{completion.manifest.content_hash[:16]}"
    )
    if staging.exists() or staging.is_symlink():
        raise ReviewCompletionError(f"unexpected staging path exists: {staging}")
    staging.mkdir(mode=0o700)
    (staging / "reviewed").mkdir(mode=0o700)
    try:
        for relative, payload in sorted(payloads.items()):
            destination = staging / relative
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(staging, output_root)
    except Exception:
        # Preserve staging evidence on any failure; never silently delete or overwrite it.
        raise
    _assert_restricted_permissions(output_root, tuple(payloads))
    return "created"


def load_completed_review(
    *, benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    output_root: Path = DEFAULT_COMPLETION_ROOT,
) -> ReviewCompletion:
    """Reproduce the entire review gate from persisted records, never a boolean flag."""

    output_root = output_root.absolute()
    _assert_no_symlink(output_root, must_exist=False)
    if not output_root.is_dir():
        raise IndependentReviewGateError("held-out launch blocked: review completion is missing")
    manifest = _read_model(
        output_root / "completion_manifest.json", IndependentReviewCompletionManifest
    )
    response = _read_model(output_root / "response.json", IndependentReviewResponse)
    adjudication = _read_model(output_root / "adjudication.json", ReviewAdjudication)
    final_seal = _read_model(output_root / "final_seal.json", FinalReviewedSeal)
    assert isinstance(manifest, IndependentReviewCompletionManifest)
    assert isinstance(response, IndependentReviewResponse)
    assert isinstance(adjudication, ReviewAdjudication)
    assert isinstance(final_seal, FinalReviewedSeal)
    artifacts = tuple(
        _read_model(output_root / entry.artifact_file, ReviewedProjectionArtifact)
        for entry in manifest.reviewed_artifacts
    )
    assert all(isinstance(item, ReviewedProjectionArtifact) for item in artifacts)
    package, bindings, draft, _originals = _load_frozen_inputs(benchmark_root)
    expected_files = set(_completion_payloads(ReviewCompletion(
        package=package,
        bindings=bindings,
        draft=draft,
        response=response,
        adjudication=adjudication,
        reviewed_artifacts=artifacts,
        final_seal=final_seal,
        manifest=manifest,
    )))
    if _existing_files(output_root) != expected_files:
        raise IndependentReviewGateError("completion tree inventory is partial or unexpected")
    try:
        _assert_restricted_permissions(output_root, tuple(expected_files))
    except ReviewCompletionError as error:
        raise IndependentReviewGateError(
            f"held-out launch blocked: {error}"
        ) from error
    try:
        require_independent_review_complete(
            draft, package, bindings, response, adjudication, artifacts, final_seal
        )
    except IndependentReviewGateError:
        raise
    by_blind = {item.blind_projection_id: item for item in artifacts}
    expected_entries = tuple(
        ReviewedArtifactManifestEntry(
            blind_projection_id=binding.blind_projection_id,
            world_id=binding.world_id,
            query_id=binding.query_id,
            artifact_file=f"reviewed/{binding.blind_projection_id}.json",
            artifact_hash=by_blind[binding.blind_projection_id].content_hash,
            source_gold_projection_hash=binding.source_gold_projection_hash,
            source_alternative_set_hash=binding.source_alternative_set_hash,
            final_semantic_hash=by_blind[binding.blind_projection_id].final_semantic_hash,
        )
        for binding in bindings.entries
    )
    expected_manifest = IndependentReviewCompletionManifest(
        manifest_id=f"review-completion-{final_seal.content_hash[:24]}",
        package_hash=package.content_hash,
        binding_manifest_hash=bindings.content_hash,
        draft_seal_hash=draft.content_hash,
        response_hash=response.content_hash,
        adjudication_hash=adjudication.content_hash,
        final_seal_hash=final_seal.content_hash,
        reviewed_artifacts=expected_entries,
    )
    if expected_manifest.content_hash != manifest.content_hash:
        raise IndependentReviewGateError(
            "held-out launch blocked: completion manifest cannot be reproduced"
        )
    return ReviewCompletion(
        package=package,
        bindings=bindings,
        draft=draft,
        response=response,
        adjudication=adjudication,
        reviewed_artifacts=artifacts,
        final_seal=final_seal,
        manifest=manifest,
    )


def prepare_public_reviewed_gold_publication(
    completion: ReviewCompletion,
) -> PublicReviewedGoldPublication:
    """Derive the public finalized-gold tree from a fully reproduced completion."""

    try:
        require_independent_review_complete(
            completion.draft,
            completion.package,
            completion.bindings,
            completion.response,
            completion.adjudication,
            completion.reviewed_artifacts,
            completion.final_seal,
        )
    except IndependentReviewGateError as error:
        raise ReviewCompletionError(
            f"cannot publish unreproduced reviewed gold: {error}"
        ) from error
    if (
        completion.manifest.final_seal_hash != completion.final_seal.content_hash
        or completion.manifest.package_hash != completion.package.content_hash
        or completion.manifest.binding_manifest_hash != completion.bindings.content_hash
        or completion.manifest.draft_seal_hash != completion.draft.content_hash
        or completion.manifest.response_hash != completion.response.content_hash
        or completion.manifest.adjudication_hash != completion.adjudication.content_hash
    ):
        raise ReviewCompletionError(
            "cannot publish reviewed gold from a mismatched completion manifest"
        )

    artifacts_by_blind = {
        item.blind_projection_id: item for item in completion.reviewed_artifacts
    }
    completion_entries = {
        item.blind_projection_id: item for item in completion.manifest.reviewed_artifacts
    }
    if len(artifacts_by_blind) != 9 or set(artifacts_by_blind) != set(completion_entries):
        raise ReviewCompletionError(
            "public reviewed-gold publication requires the exact completion inventory"
        )
    public_entries: list[PublicReviewedArtifactManifestEntry] = []
    ordered_artifacts: list[ReviewedProjectionArtifact] = []
    for binding in sorted(
        completion.bindings.entries, key=lambda item: item.blind_projection_id
    ):
        artifact = artifacts_by_blind[binding.blind_projection_id]
        completion_entry = completion_entries[binding.blind_projection_id]
        artifact_payload = artifact.to_canonical_json().encode("utf-8") + b"\n"
        if (
            completion_entry.world_id != binding.world_id
            or completion_entry.query_id != binding.query_id
            or completion_entry.artifact_hash != artifact.content_hash
            or completion_entry.source_gold_projection_hash
            != binding.source_gold_projection_hash
            or completion_entry.source_alternative_set_hash
            != binding.source_alternative_set_hash
            or completion_entry.final_semantic_hash != artifact.final_semantic_hash
        ):
            raise ReviewCompletionError(
                "public reviewed-gold entry differs from its completion lineage"
            )
        public_entries.append(
            PublicReviewedArtifactManifestEntry(
                blind_projection_id=binding.blind_projection_id,
                world_id=binding.world_id,
                query_id=binding.query_id,
                artifact_file=f"reviewed/{binding.blind_projection_id}.json",
                artifact_file_sha256=hashlib.sha256(artifact_payload).hexdigest(),
                artifact_hash=artifact.content_hash,
                source_gold_projection_hash=binding.source_gold_projection_hash,
                source_alternative_set_hash=binding.source_alternative_set_hash,
                source_semantic_hash=binding.source_semantic_hash,
                final_semantic_hash=artifact.final_semantic_hash,
                gold_projection_hash=artifact.gold_projection.content_hash,
                alternative_set_hash=artifact.alternatives.content_hash,
                amended=artifact.final_semantic_hash != binding.source_semantic_hash,
            )
        )
        ordered_artifacts.append(artifact)

    completion_manifest_payload = (
        completion.manifest.to_canonical_json().encode("utf-8") + b"\n"
    )
    final_seal_payload = completion.final_seal.to_canonical_json().encode("utf-8") + b"\n"
    manifest = PublicReviewedGoldPublicationManifest(
        manifest_id=f"public-reviewed-gold-{completion.final_seal.content_hash[:24]}",
        source_completion_manifest_hash=completion.manifest.content_hash,
        source_completion_manifest_file_sha256=hashlib.sha256(
            completion_manifest_payload
        ).hexdigest(),
        draft_seal_hash=completion.draft.content_hash,
        package_hash=completion.package.content_hash,
        binding_manifest_hash=completion.bindings.content_hash,
        final_seal_hash=completion.final_seal.content_hash,
        final_seal_file_sha256=hashlib.sha256(final_seal_payload).hexdigest(),
        reviewed_artifacts=tuple(public_entries),
        amendment_count=sum(item.amended for item in public_entries),
    )
    return PublicReviewedGoldPublication(
        manifest=manifest,
        reviewed_artifacts=tuple(ordered_artifacts),
        final_seal=completion.final_seal,
    )


def _public_reviewed_gold_payloads(
    publication: PublicReviewedGoldPublication,
) -> dict[str, bytes]:
    artifacts_by_blind = {
        item.blind_projection_id: item for item in publication.reviewed_artifacts
    }
    if len(artifacts_by_blind) != 9:
        raise ReviewCompletionError("public reviewed-gold payload inventory is incomplete")
    payloads = {
        "publication_manifest.json": (
            publication.manifest.to_canonical_json().encode("utf-8") + b"\n"
        ),
        "final_seal.json": publication.final_seal.to_canonical_json().encode("utf-8")
        + b"\n",
    }
    for entry in publication.manifest.reviewed_artifacts:
        artifact = artifacts_by_blind.get(entry.blind_projection_id)
        if artifact is None:
            raise ReviewCompletionError(
                "public reviewed-gold manifest names an absent reviewed artifact"
            )
        payload = artifact.to_canonical_json().encode("utf-8") + b"\n"
        if (
            artifact.content_hash != entry.artifact_hash
            or artifact.gold_projection.content_hash != entry.gold_projection_hash
            or artifact.alternatives.content_hash != entry.alternative_set_hash
            or hashlib.sha256(payload).hexdigest() != entry.artifact_file_sha256
        ):
            raise ReviewCompletionError(
                "public reviewed-gold artifact bytes differ from their manifest"
            )
        payloads[entry.artifact_file] = payload
    if (
        publication.final_seal.content_hash != publication.manifest.final_seal_hash
        or hashlib.sha256(payloads["final_seal.json"]).hexdigest()
        != publication.manifest.final_seal_file_sha256
    ):
        raise ReviewCompletionError("public reviewed-gold final seal differs from its manifest")
    return payloads


def _scan_public_reviewed_gold_tree(root: Path, payloads: Mapping[str, bytes]) -> None:
    from story_projection_onto.public_release import PublicEntry, scan_public_entries

    scan_public_entries(
        root,
        tuple(
            PublicEntry(
                source_relative_path=relative,
                bundle_relative_path=relative,
                sha256=hashlib.sha256(payload).hexdigest(),
                release_class=ReleaseClass.PUBLIC.value,
            )
            for relative, payload in sorted(payloads.items())
        ),
    )


def materialize_public_reviewed_gold(
    publication: PublicReviewedGoldPublication,
    output_root: Path = DEFAULT_PUBLIC_REVIEWED_GOLD_ROOT,
) -> Literal["created", "already_exact"]:
    """Write an append-only, scanner-validated public finalized-gold tree."""

    output_root = output_root.absolute()
    _assert_no_symlink(output_root, must_exist=False)
    payloads = _public_reviewed_gold_payloads(publication)
    if output_root.exists():
        if not output_root.is_dir():
            raise ReviewCompletionError(
                "public reviewed-gold output exists but is not a directory"
            )
        if _existing_files(output_root) != set(payloads):
            raise ReviewCompletionError(
                "existing public reviewed-gold tree is partial or unexpected"
            )
        for relative, expected in payloads.items():
            if (output_root / relative).read_bytes() != expected:
                raise ReviewCompletionError(
                    f"existing public reviewed-gold artifact drift: {relative}"
                )
        _scan_public_reviewed_gold_tree(output_root, payloads)
        return "already_exact"

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / (
        f".{output_root.name}.staging-{publication.manifest.content_hash[:16]}"
    )
    if staging.exists() or staging.is_symlink():
        raise ReviewCompletionError(
            f"unexpected public reviewed-gold staging path exists: {staging}"
        )
    staging.mkdir(mode=0o755)
    (staging / "reviewed").mkdir(mode=0o755)
    for relative, payload in sorted(payloads.items()):
        destination = staging / relative
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    _scan_public_reviewed_gold_tree(staging, payloads)
    os.rename(staging, output_root)
    directory = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return "created"


def load_public_reviewed_gold_publication(
    *,
    completion: ReviewCompletion,
    output_root: Path = DEFAULT_PUBLIC_REVIEWED_GOLD_ROOT,
) -> PublicReviewedGoldPublication:
    """Reproduce a public finalized-gold tree from its restricted completion."""

    output_root = output_root.absolute()
    _assert_no_symlink(output_root, must_exist=False)
    if not output_root.is_dir():
        raise ReviewCompletionError("public reviewed-gold publication is missing")
    expected = prepare_public_reviewed_gold_publication(completion)
    payloads = _public_reviewed_gold_payloads(expected)
    if _existing_files(output_root) != set(payloads):
        raise ReviewCompletionError(
            "public reviewed-gold publication inventory is partial or unexpected"
        )
    for relative, expected_payload in payloads.items():
        path = output_root / relative
        _assert_no_symlink(path, must_exist=True)
        if path.read_bytes() != expected_payload:
            raise ReviewCompletionError(
                f"public reviewed-gold publication drift: {relative}"
            )
    _scan_public_reviewed_gold_tree(output_root, payloads)
    return expected


def require_materialized_independent_review_complete(
    *, benchmark_root: Path = DEFAULT_BENCHMARK_ROOT, output_root: Path = DEFAULT_COMPLETION_ROOT
) -> None:
    """Held-out gate backed by full hash and semantic reproduction."""

    try:
        load_completed_review(benchmark_root=benchmark_root, output_root=output_root)
    except IndependentReviewGateError:
        raise
    except ReviewCompletionError as error:
        raise IndependentReviewGateError(
            f"held-out launch blocked: {error}"
        ) from error


__all__ = [
    "DEFAULT_BENCHMARK_ROOT",
    "DEFAULT_COMPLETION_ROOT",
    "DEFAULT_PUBLIC_REVIEWED_GOLD_ROOT",
    "AmendedProjectionSubmission",
    "IndependentReviewCompletionManifest",
    "MethodologicalAmendmentRequired",
    "PublicReviewedArtifactManifestEntry",
    "PublicReviewedGoldPublication",
    "PublicReviewedGoldPublicationManifest",
    "ReviewAmendmentBundle",
    "ReviewCompletion",
    "ReviewCompletionError",
    "load_completed_review",
    "load_public_reviewed_gold_publication",
    "materialize_public_reviewed_gold",
    "materialize_review_completion",
    "prepare_public_reviewed_gold_publication",
    "prepare_review_completion",
    "require_materialized_independent_review_complete",
]
