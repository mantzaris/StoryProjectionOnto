from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from story_projection_onto.contracts import GoldAlternativeSet, GoldContextualProjection
from story_projection_onto.independent_review_runtime import (
    AmendedProjectionSubmission,
    MethodologicalAmendmentRequired,
    ReviewAmendmentBundle,
    ReviewCompletionError,
    load_completed_review,
    load_public_reviewed_gold_publication,
    materialize_public_reviewed_gold,
    materialize_review_completion,
    prepare_public_reviewed_gold_publication,
    prepare_review_completion,
    require_materialized_independent_review_complete,
)
from story_projection_onto.synthetic_benchmark import (
    AdjudicationDisposition,
    BlindIndependentReviewPackage,
    IndependentReviewGateError,
    IndependentReviewResponse,
    IndependentReviewResponseItem,
    ReviewAdjudication,
    ReviewAdjudicationItem,
    ReviewDisposition,
    ReviewProjectionBindingManifest,
    reviewed_artifact_from_original,
    reviewed_semantic_hash,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "data/synthetic"


def _load(path: Path, model_type):
    return model_type.model_validate_json(path.read_bytes())


def _write(path: Path, record) -> Path:
    path.write_text(record.to_canonical_json() + "\n", encoding="utf-8")
    return path


def _without_hashes(value):
    if isinstance(value, dict):
        return {key: _without_hashes(item) for key, item in value.items() if key != "content_hash"}
    if isinstance(value, list):
        return [_without_hashes(item) for item in value]
    return value


def _package() -> BlindIndependentReviewPackage:
    return _load(
        BENCHMARK / "scorer_only/review/blind_review_package.json",
        BlindIndependentReviewPackage,
    )


def _external_records(
    tmp_path: Path,
    *,
    exceptional: ReviewDisposition | None = None,
    adjudication_disposition: AdjudicationDisposition = AdjudicationDisposition.RETAIN,
    amended_hash: str | None = None,
):
    package = _package()
    prompts = [
        item
        for world in package.worlds
        for projection in world.projections
        for item in projection.review_items
    ]
    exceptional_id = prompts[0].review_item_id if exceptional else None
    response = IndependentReviewResponse(
        package_hash=package.content_hash,
        reviewer_pseudonym="test-fixture-external-reviewer",
        reviewed_at="2026-01-02T00:00:00Z",
        items=tuple(
            IndependentReviewResponseItem(
                review_item_id=item.review_item_id,
                blind_projection_id=item.blind_projection_id,
                criterion=item.criterion,
                disposition=(
                    exceptional
                    if item.review_item_id == exceptional_id
                    else ReviewDisposition.AGREE
                ),
                notes="TEST FIXTURE ONLY; not a scientific review",
            )
            for item in prompts
        ),
    )
    adjudication = ReviewAdjudication(
        package_hash=package.content_hash,
        response_hash=response.content_hash,
        adjudicator_pseudonym="test-fixture-adjudicator",
        adjudicated_at=response.reviewed_at + timedelta(hours=1),
        items=(
            (
                ReviewAdjudicationItem(
                    review_item_id=exceptional_id,
                    disposition=adjudication_disposition,
                    rationale="TEST FIXTURE ONLY; not scientific adjudication",
                    amended_artifact_hash=amended_hash,
                ),
            )
            if exceptional_id
            else ()
        ),
    )
    return (
        package,
        response,
        adjudication,
        _write(tmp_path / "response.json", response),
        _write(tmp_path / "adjudication.json", adjudication),
    )


def test_missing_review_completion_blocks_held_out(tmp_path: Path) -> None:
    with pytest.raises(IndependentReviewGateError, match="completion is missing"):
        require_materialized_independent_review_complete(
            benchmark_root=BENCHMARK, output_root=tmp_path / "missing"
        )


def test_all_agree_fixture_materializes_idempotently_and_reproduces_seal(
    tmp_path: Path,
) -> None:
    _package_record, _response, _adjudication, response_path, adjudication_path = (
        _external_records(tmp_path)
    )
    completion = prepare_review_completion(
        benchmark_root=BENCHMARK,
        response_path=response_path,
        adjudication_path=adjudication_path,
    )
    assert len(completion.reviewed_artifacts) == 9
    assert completion.manifest.review_item_count == 72
    output = tmp_path / "restricted-completion"
    assert materialize_review_completion(completion, output) == "created"
    assert materialize_review_completion(completion, output) == "already_exact"
    loaded = load_completed_review(benchmark_root=BENCHMARK, output_root=output)
    assert loaded.final_seal.content_hash == completion.final_seal.content_hash
    assert loaded.manifest.content_hash == completion.manifest.content_hash
    manifest_text = (output / "completion_manifest.json").read_text()
    assert "test-fixture-external-reviewer" not in manifest_text
    assert "TEST FIXTURE" not in manifest_text

    publication = prepare_public_reviewed_gold_publication(loaded)
    public_output = tmp_path / "public-reviewed-gold"
    assert materialize_public_reviewed_gold(publication, public_output) == "created"
    assert materialize_public_reviewed_gold(publication, public_output) == "already_exact"
    reproduced = load_public_reviewed_gold_publication(
        completion=loaded,
        output_root=public_output,
    )
    assert reproduced.manifest.content_hash == publication.manifest.content_hash
    assert reproduced.final_seal.content_hash == loaded.final_seal.content_hash
    assert len(reproduced.reviewed_artifacts) == 9
    published_paths = {
        path.relative_to(public_output).as_posix()
        for path in public_output.rglob("*")
    }
    assert published_paths == {
        "final_seal.json",
        "publication_manifest.json",
        "reviewed",
        *{
            f"reviewed/{item.blind_projection_id}.json"
            for item in reproduced.reviewed_artifacts
        },
    }
    public_bytes = b"".join(
        path.read_bytes() for path in public_output.rglob("*") if path.is_file()
    )
    assert b"test-fixture-external-reviewer" not in public_bytes
    assert b"TEST FIXTURE" not in public_bytes


def test_disagreement_retain_preserves_sealed_semantics(tmp_path: Path) -> None:
    _p, _r, _a, response_path, adjudication_path = _external_records(
        tmp_path, exceptional=ReviewDisposition.DISAGREE
    )
    completion = prepare_review_completion(
        benchmark_root=BENCHMARK,
        response_path=response_path,
        adjudication_path=adjudication_path,
    )
    binding = completion.bindings.entries[0]
    artifact = completion.reviewed_artifacts[0]
    assert artifact.final_semantic_hash == binding.source_semantic_hash
    assert artifact.gold_projection.review_status.value == "disagreement_logged"
    assert artifact.gold_projection.adjudication_status.value == "adjudicated"


def test_amendment_requires_explicit_reproducible_semantics(tmp_path: Path) -> None:
    package = _package()
    bindings = _load(
        BENCHMARK / "scorer_only/review/scorer_bindings.json",
        ReviewProjectionBindingManifest,
    )
    binding = bindings.entries[0]
    scorer = json.loads(
        (BENCHMARK / f"scorer_only/held_out/{binding.world_id}.json").read_text()
    )
    gold_raw = next(
        item for item in scorer["gold_projections"] if item["query_id"] == binding.query_id
    )
    alt_raw = next(
        item
        for item in scorer["alternatives"]
        if item["gold_projection_id"] == gold_raw["gold_projection_id"]
    )
    amended_alt_raw = _without_hashes(alt_raw)
    amended_alt_raw["matching_rule"] = amended_alt_raw["matching_rule"] + "; adjudicated amendment"
    amended_gold = GoldContextualProjection.model_validate(_without_hashes(gold_raw))
    amended_alt = GoldAlternativeSet.model_validate(amended_alt_raw)
    semantic_hash = reviewed_semantic_hash(amended_gold, amended_alt)
    _p, response, adjudication, response_path, adjudication_path = _external_records(
        tmp_path,
        exceptional=ReviewDisposition.DISAGREE,
        adjudication_disposition=AdjudicationDisposition.AMEND,
        amended_hash=semantic_hash,
    )
    bundle = ReviewAmendmentBundle(
        package_hash=package.content_hash,
        response_hash=response.content_hash,
        adjudication_hash=adjudication.content_hash,
        submitted_at=adjudication.adjudicated_at + timedelta(hours=1),
        submissions=(
            AmendedProjectionSubmission(
                blind_projection_id=binding.blind_projection_id,
                source_gold_projection_hash=binding.source_gold_projection_hash,
                source_alternative_set_hash=binding.source_alternative_set_hash,
                final_semantic_hash=semantic_hash,
                gold_projection=amended_gold,
                alternatives=amended_alt,
            ),
        ),
    )
    completion = prepare_review_completion(
        benchmark_root=BENCHMARK,
        response_path=response_path,
        adjudication_path=adjudication_path,
        amendment_path=_write(tmp_path / "amendments.json", bundle),
    )
    assert completion.reviewed_artifacts[0].final_semantic_hash == semantic_hash
    assert semantic_hash != binding.source_semantic_hash

    restricted_output = tmp_path / "restricted-completion"
    materialize_review_completion(completion, restricted_output)
    loaded = load_completed_review(
        benchmark_root=BENCHMARK,
        output_root=restricted_output,
    )
    publication = prepare_public_reviewed_gold_publication(loaded)
    assert publication.manifest.amendment_count == 1
    amended_entry = next(
        item
        for item in publication.manifest.reviewed_artifacts
        if item.blind_projection_id == binding.blind_projection_id
    )
    assert amended_entry.amended is True
    assert amended_entry.final_semantic_hash == semantic_hash
    public_output = tmp_path / "public-reviewed-gold"
    materialize_public_reviewed_gold(publication, public_output)
    load_public_reviewed_gold_publication(completion=loaded, output_root=public_output)

    stale_original = reviewed_artifact_from_original(
        binding=binding,
        package=package,
        response=response,
        adjudication=adjudication,
        gold=GoldContextualProjection.model_validate(gold_raw),
        alternatives=GoldAlternativeSet.model_validate(alt_raw),
    )
    assert stale_original.final_semantic_hash == binding.source_semantic_hash
    (public_output / amended_entry.artifact_file).write_text(
        stale_original.to_canonical_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ReviewCompletionError, match="publication drift"):
        load_public_reviewed_gold_publication(
            completion=loaded,
            output_root=public_output,
        )


def test_exclude_requires_methodological_amendment(tmp_path: Path) -> None:
    _p, _r, _a, response_path, adjudication_path = _external_records(
        tmp_path,
        exceptional=ReviewDisposition.UNCERTAIN,
        adjudication_disposition=AdjudicationDisposition.EXCLUDE,
    )
    with pytest.raises(MethodologicalAmendmentRequired, match="methodological amendment"):
        prepare_review_completion(
            benchmark_root=BENCHMARK,
            response_path=response_path,
            adjudication_path=adjudication_path,
        )


def test_hash_id_symlink_partial_and_tamper_fail_closed(tmp_path: Path) -> None:
    package, response, adjudication, response_path, adjudication_path = _external_records(tmp_path)
    wrong_response = IndependentReviewResponse.model_validate(
        {
            **_without_hashes(response.model_dump(mode="python")),
            "package_hash": "0" * 64,
        }
    )
    with pytest.raises(ReviewCompletionError, match="package hash"):
        prepare_review_completion(
            benchmark_root=BENCHMARK,
            response_path=_write(tmp_path / "wrong.json", wrong_response),
            adjudication_path=adjudication_path,
        )

    completion = prepare_review_completion(
        benchmark_root=BENCHMARK,
        response_path=response_path,
        adjudication_path=adjudication_path,
    )
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "response.json").write_text("{}")
    with pytest.raises(ReviewCompletionError, match="partial or unexpected"):
        materialize_review_completion(completion, partial)

    output = tmp_path / "complete"
    materialize_review_completion(completion, output)
    (output / "final_seal.json").write_text("{}")
    with pytest.raises(ReviewCompletionError, match="artifact drift"):
        materialize_review_completion(completion, output)

    symlink = tmp_path / "linked-response.json"
    symlink.symlink_to(response_path)
    with pytest.raises(ReviewCompletionError, match="symlinked path"):
        prepare_review_completion(
            benchmark_root=BENCHMARK,
            response_path=symlink,
            adjudication_path=adjudication_path,
        )
    assert package.content_hash == response.package_hash
    assert adjudication.response_hash == response.content_hash


def test_restricted_completion_permission_drift_fails_closed(tmp_path: Path) -> None:
    _package_record, _response, _adjudication, response_path, adjudication_path = (
        _external_records(tmp_path)
    )
    completion = prepare_review_completion(
        benchmark_root=BENCHMARK,
        response_path=response_path,
        adjudication_path=adjudication_path,
    )
    output = tmp_path / "restricted-completion"
    assert materialize_review_completion(completion, output) == "created"
    response_output = output / "response.json"
    response_output.chmod(0o644)

    with pytest.raises(ReviewCompletionError, match="permission drift"):
        materialize_review_completion(completion, output)
    with pytest.raises(IndependentReviewGateError, match="permission drift"):
        load_completed_review(benchmark_root=BENCHMARK, output_root=output)
