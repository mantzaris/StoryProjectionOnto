from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import (
    AbstractionLevel,
    DiscoursePosition,
    OutputBudgets,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
)
from story_projection_onto.novel_case import (
    CasePreregistrationError,
    CaseStudyPreregistration,
    CaseStudyReviewBundle,
    CaseUnitReview,
    CaseWindowRegistration,
    DetailedCaseMatching,
    LawfulNovelSourceAuthorization,
    LawfulSourceRequiredError,
    NovelIndexError,
    NovelSegmentationConfig,
    OperationalRetrievalRegistration,
    PublicNarrativeIllustration,
    PublicReleaseViolation,
    RestrictedNovelIndexManifest,
    build_public_case_artifact,
    build_restricted_novel_index,
    materialize_window_evidence,
    retrieve_operational_packet,
    scan_public_case_artifact,
    validate_case_preregistration,
    validate_case_review_bundle,
    verify_restricted_novel_index,
    write_restricted_manifest,
)

T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _source_text() -> str:
    return """CHAPTER I

The amber courier crossed the bridge before dawn carrying a sealed token.

The watch recorded the courier and denied knowing the token's origin.

CHAPTER II

At noon the council reported that the amber token changed the planned meeting.

Another witness challenged that report and the matter remained uncertain.

CHAPTER III

The courier returned after dusk while the council debated the earlier report.
"""


def _config() -> NovelSegmentationConfig:
    return NovelSegmentationConfig(
        corpus_id="restricted-first-novel",
        edition_label="lawful fixture edition",
        chapter_heading_pattern=r"^CHAPTER [IVX]+$",
        passage_target_characters=80,
        passage_max_characters=140,
        operational_top_k=3,
        operational_max_evidence_tokens=500,
    )


def _build(tmp_path: Path) -> tuple[Path, Path, RestrictedNovelIndexManifest]:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    source = restricted / "source.txt"
    source.write_text(_source_text(), encoding="utf-8", newline="")
    index = restricted / "novel.sqlite"
    authorization = LawfulNovelSourceAuthorization(
        source_path=source,
        restricted_root=restricted,
        lawful_copy_attested=True,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    manifest = build_restricted_novel_index(
        authorization,
        _config(),
        destination=index,
        built_at=T0,
    )
    return source, index, manifest


def _horizon(order: int = 20) -> SpoilerHorizon:
    return SpoilerHorizon(
        horizon_id=f"case-horizon-{order}",
        max_discourse_position=DiscoursePosition(passage_order=order),
    )


def _context(identifier: str, horizon: SpoilerHorizon) -> QueryContext:
    return QueryContext(
        context_id=identifier,
        wording=f"Examine the registered lens for {identifier}",
        lens="belief and temporal consequence",
        target="the registered contextual relationship",
        story_scope=StoryTime(kind=TemporalKind.UNKNOWN, reason="to be reviewed from evidence"),
        spoiler_horizon=horizon,
        abstraction=AbstractionLevel.EVENT_ROLE,
        budgets=OutputBudgets(
            node_budget=10,
            assertion_budget=20,
            display_node_budget=10,
            display_assertion_budget=20,
        ),
        revealed_at=T0 + timedelta(hours=1),
    )


def _window(number: int, passage_ids: tuple[str, ...], chapter: int) -> CaseWindowRegistration:
    horizon = _horizon()
    return CaseWindowRegistration(
        window_id=f"case-window-{number}",
        chapter_ordinals=(chapter,),
        passage_ids=passage_ids,
        fixed_horizon=horizon,
        contexts=(
            _context(f"case-context-{number}-a", horizon),
            _context(f"case-context-{number}-b", horizon),
        ),
        contrast_dimensions=("belief_revelation",),
        selection_basis="registered evidence and narrative criterion before model output",
    )


def _preregistration(manifest: RestrictedNovelIndexManifest) -> CaseStudyPreregistration:
    windows = (
        _window(1, ("passage-000001",), 1),
        _window(2, ("passage-000002",), 1),
        _window(3, ("passage-000003",), 2),
        _window(4, ("passage-000004",), 2),
    )
    operational = OperationalRetrievalRegistration(
        operational_query_id="case-operational-1",
        context=_context("case-operational-context", _horizon()),
        frozen_fts_query="amber token report",
        top_k=3,
        max_evidence_tokens=500,
    )
    return CaseStudyPreregistration(
        preregistration_id="first-novel-preregistration",
        restricted_index_manifest_hash=manifest.content_hash,
        windows=windows,
        operational_retrieval=operational,
        detailed_review_context_ids=(
            "case-context-1-a",
            "case-context-2-a",
            "case-context-3-a",
            "case-context-4-a",
        ),
        primary_seed=7103,
        frozen_at=T0 + timedelta(minutes=1),
    )


def test_source_access_requires_explicit_lawful_authority(tmp_path: Path) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    source = restricted / "source.txt"
    source.write_text("CHAPTER I\n\nFixture.", encoding="utf-8")
    index = restricted / "index.sqlite"

    with pytest.raises(LawfulSourceRequiredError, match="attestation"):
        build_restricted_novel_index(
            LawfulNovelSourceAuthorization(source, restricted, lawful_copy_attested=False),
            _config(),
            destination=index,
            built_at=T0,
        )

    outside = tmp_path / "outside.txt"
    outside.write_text("CHAPTER I\n\nFixture.", encoding="utf-8")
    with pytest.raises(LawfulSourceRequiredError, match="inside"):
        build_restricted_novel_index(
            LawfulNovelSourceAuthorization(outside, restricted, lawful_copy_attested=True),
            _config(),
            destination=index,
            built_at=T0,
        )
    assert not index.exists()


def test_builds_one_restricted_query_blind_external_content_index(tmp_path: Path) -> None:
    source, index, manifest = _build(tmp_path)

    assert source.read_text(encoding="utf-8") == _source_text()
    assert manifest.query_blind
    assert manifest.release_class is ReleaseClass.RESTRICTED
    assert manifest.chapter_count == 3
    assert manifest.passage_count >= 5
    assert str(source) not in manifest.to_canonical_json()
    assert str(index) not in manifest.to_canonical_json()
    assert sorted(path.name for path in index.parent.iterdir()) == ["novel.sqlite", "source.txt"]

    verify_restricted_novel_index(index, manifest)
    connection = sqlite3.connect(index)
    try:
        fts_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='passage_fts'"
        ).fetchone()[0]
        assert "content='passages'" in fts_sql
        assert connection.execute("SELECT count(*) FROM passage_fts").fetchone()[0] == (
            manifest.passage_count
        )
        keys = {row[0] for row in connection.execute("SELECT key FROM metadata")}
        assert not {"query", "lens", "target", "ontology"}.intersection(keys)
    finally:
        connection.close()


def test_index_refuses_overwrite_and_detects_manifest_tampering(tmp_path: Path) -> None:
    source, index, manifest = _build(tmp_path)
    authorization = LawfulNovelSourceAuthorization(
        source, source.parent, lawful_copy_attested=True
    )
    with pytest.raises(NovelIndexError, match="overwrite"):
        build_restricted_novel_index(
            authorization, _config(), destination=index, built_at=T0
        )

    tampered_payload = manifest.model_dump(mode="json")
    tampered_payload["index_sha256"] = "0" * 64
    tampered_payload["content_hash"] = ""
    tampered = RestrictedNovelIndexManifest.model_validate(tampered_payload)
    with pytest.raises(NovelIndexError, match="hash"):
        verify_restricted_novel_index(index, tampered)

    with pytest.raises(NovelIndexError, match="inside"):
        write_restricted_manifest(
            manifest,
            tmp_path / "public-manifest.json",
            restricted_root=source.parent,
        )
    restricted_manifest = source.parent / "index-manifest.json"
    write_restricted_manifest(
        manifest,
        restricted_manifest,
        restricted_root=source.parent,
    )
    assert RestrictedNovelIndexManifest.model_validate_json(
        restricted_manifest.read_text(encoding="utf-8")
    ) == manifest


def test_preregistration_requires_four_windows_eight_same_horizon_contexts(
    tmp_path: Path,
) -> None:
    _, index, manifest = _build(tmp_path)
    preregistration = _preregistration(manifest)
    validate_case_preregistration(preregistration, manifest, index_path=index)
    assert len(preregistration.windows) == 4
    assert sum(len(window.contexts) for window in preregistration.windows) == 8
    assert preregistration.operational_retrieval.operational_only
    assert not preregistration.operational_retrieval.causal_comparison_eligible
    assert preregistration.c1_window_preconstruction_count == 4
    assert preregistration.primary_seed_count == 1

    bad_window = preregistration.windows[0].model_copy(
        update={
            "contexts": (
                preregistration.windows[0].contexts[0],
                _context("wrong-horizon", _horizon(1)),
            ),
            "content_hash": "",
        }
    )
    with pytest.raises(ValidationError, match="exact fixed horizon"):
        CaseWindowRegistration.model_validate(bad_window.model_dump())


def test_materializes_bounded_all_admissible_window_only_in_memory(tmp_path: Path) -> None:
    _, index, manifest = _build(tmp_path)
    window = _preregistration(manifest).windows[0]
    before = {path.name for path in index.parent.iterdir()}
    result = materialize_window_evidence(
        index_path=index,
        manifest=manifest,
        window=window,
        created_at=T0 + timedelta(minutes=2),
        sealed_at=T0 + timedelta(minutes=3),
        packet_created_at=T0 + timedelta(minutes=4),
        token_counter=lambda value: len(value.split()),
    )

    assert result.packet.retrieval_method is RetrievalMethod.ALL_ADMISSIBLE
    assert result.packet.release_class is ReleaseClass.RESTRICTED
    assert result.packet.ordered_evidence_ids == ("evidence-000001",)
    assert result.snapshot_assembly.snapshot.horizon == window.fixed_horizon
    assert {path.name for path in index.parent.iterdir()} == before


def test_operational_bm25_packet_is_separate_bounded_and_prose_free_in_receipt(
    tmp_path: Path,
) -> None:
    _, index, manifest = _build(tmp_path)
    registration = _preregistration(manifest).operational_retrieval
    result = retrieve_operational_packet(
        index_path=index,
        manifest=manifest,
        registration=registration,
        query_accessed_at=T0 + timedelta(minutes=2),
        packet_created_at=T0 + timedelta(minutes=3),
        token_counter=lambda value: len(value.split()),
    )

    assert result.packet.retrieval_method is RetrievalMethod.SQLITE_FTS5_BM25
    assert result.packet.release_class is ReleaseClass.RESTRICTED
    assert len(result.packet.evidence) <= registration.top_k
    assert result.receipt.operational_only
    assert not result.receipt.causal_comparison_eligible
    receipt_json = result.receipt.to_canonical_json()
    assert "amber courier crossed" not in receipt_json
    assert "text" not in json.loads(receipt_json)
    assert result.receipt.packet_hash == result.packet.content_hash


def test_public_export_uses_opaque_coarse_locators_and_blocks_verbatim_prose(
    tmp_path: Path,
) -> None:
    _, index, manifest = _build(tmp_path)
    safe = PublicNarrativeIllustration(
        illustration_id="illustration-1",
        window_id="case-window-1",
        chapter_ordinals=(1,),
        opaque_evidence_ids=("evidence-000001",),
        high_level_paraphrase="A delivery changes how a later deliberation is interpreted.",
        interpretation_scope="qualitative narrative transfer illustration",
    )
    artifact = build_public_case_artifact(
        manifest=manifest,
        illustrations=(safe,),
        restricted_index_path=index,
    )
    payload = artifact.to_canonical_json()
    assert "start_char" not in payload
    assert "end_char" not in payload
    assert str(index) not in payload
    assert str(index.parent) not in payload

    unsafe = PublicNarrativeIllustration(
        illustration_id="illustration-unsafe",
        window_id="case-window-1",
        chapter_ordinals=(1,),
        opaque_evidence_ids=("evidence-000001",),
        high_level_paraphrase=(
            "The amber courier crossed the bridge before dawn carrying a sealed token."
        ),
        interpretation_scope="unsafe exact reproduction",
    )
    unsafe_artifact = build_public_case_artifact(
        manifest=manifest,
        illustrations=(unsafe,),
        restricted_index_path=None,
    )
    with pytest.raises(PublicReleaseViolation, match="verbatim source overlap"):
        scan_public_case_artifact(unsafe_artifact, restricted_index_path=index)


def test_review_bundle_requires_all_eight_and_exact_four_detailed_reviews(
    tmp_path: Path,
) -> None:
    _, _, manifest = _build(tmp_path)
    preregistration = _preregistration(manifest)
    detailed_ids = set(preregistration.detailed_review_context_ids)
    context_ids = [
        context.context_id
        for window in preregistration.windows
        for context in window.contexts
    ]
    reviews = tuple(
        CaseUnitReview(
            review_id=f"review-{context_id}",
            context_id=context_id,
            reviewer_id="case-reviewer-1",
            evidence_support="supported",
            contextual_relevance="supported",
            temporal_integrity="uncertain",
            rare_pivotal_preservation="not_applicable",
            high_level_organization="supported",
            detailed_matching=(
                DetailedCaseMatching(
                    matched_assertion_count=2,
                    unmatched_assertion_count=1,
                    matched_event_count=1,
                    unmatched_event_count=0,
                    matching_notes="restricted structured review without quoted prose",
                )
                if context_id in detailed_ids
                else None
            ),
            reviewed_at=T0 + timedelta(hours=2),
        )
        for context_id in context_ids
    )
    bundle = CaseStudyReviewBundle(
        preregistration_hash=preregistration.content_hash,
        reviews=reviews,
    )
    validate_case_review_bundle(bundle, preregistration)

    with pytest.raises(ValidationError, match="8 items"):
        CaseStudyReviewBundle(
            preregistration_hash=preregistration.content_hash,
            reviews=(*reviews, reviews[0]),
        )

    duplicate_reviews = (*reviews[:-1], reviews[0])
    with pytest.raises(ValidationError, match="eight distinct contexts"):
        CaseStudyReviewBundle(
            preregistration_hash=preregistration.content_hash,
            reviews=duplicate_reviews,
        )

    wrong_reviews = list(reviews)
    wrong_reviews[0] = wrong_reviews[0].model_copy(
        update={"detailed_matching": None, "content_hash": ""}
    )
    wrong_bundle = CaseStudyReviewBundle(
        preregistration_hash=preregistration.content_hash,
        reviews=tuple(wrong_reviews),
    )
    with pytest.raises(CasePreregistrationError, match="exactly four"):
        validate_case_review_bundle(wrong_bundle, preregistration)
