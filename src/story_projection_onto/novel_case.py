"""Restricted first-novel indexing and preregistration substrate.

This module deliberately does not discover a corpus.  A caller must provide one
exact source path, an enclosing restricted root, and an affirmative lawful-copy
attestation.  Protected prose is held only by the source and the restricted
external-content FTS index; it is never written to ordinary run artifacts or
public exports.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import model_validator

from story_projection_onto.contracts import (
    ConditionName,
    Document,
    EvidencePacket,
    EvidenceRecord,
    Identifier,
    ImmutableRecord,
    NonEmptyText,
    NonNegativeInt,
    OutputBudgets,
    Passage,
    PositiveInt,
    ProvenanceReference,
    QueryContext,
    ReleaseClass,
    RetrievalMethod,
    RightsClass,
    Sha256Digest,
    SpoilerHorizon,
    canonical_json,
    canonical_sha256,
    to_model_visible_evidence,
)
from story_projection_onto.evidence import (
    EvidenceSnapshotAssembly,
    build_all_admissible_packet,
    build_evidence_snapshot,
)


class NovelCaseError(RuntimeError):
    """Base class for fail-closed case-study errors."""


class LawfulSourceRequiredError(NovelCaseError):
    """Raised when explicit lawful-source authority is absent or invalid."""


class NovelIndexError(NovelCaseError):
    """Raised when the restricted index cannot be built or verified safely."""


class CasePreregistrationError(NovelCaseError):
    """Raised when the bounded four-window/eight-context design is incomplete."""


class PublicReleaseViolation(NovelCaseError):
    """Raised when a proposed public case artifact may expose protected material."""


@dataclass(frozen=True, slots=True)
class LawfulNovelSourceAuthorization:
    """Ephemeral source authority; intentionally not an artifact contract.

    Paths are excluded from ``repr`` and from all returned manifests so a public
    logging layer cannot expose a personal or restricted filesystem location.
    """

    source_path: Path = field(repr=False)
    restricted_root: Path = field(repr=False)
    lawful_copy_attested: bool = False
    expected_source_sha256: str | None = field(default=None, repr=False)

    def resolve(self) -> tuple[Path, Path]:
        if not self.lawful_copy_attested:
            raise LawfulSourceRequiredError("lawful-copy attestation is required")
        if not self.source_path.is_absolute() or not self.restricted_root.is_absolute():
            raise LawfulSourceRequiredError("source and restricted-root paths must be explicit")
        if self.source_path.is_symlink():
            raise LawfulSourceRequiredError("the restricted source cannot be a symbolic link")
        try:
            root = self.restricted_root.resolve(strict=True)
            source = self.source_path.resolve(strict=True)
        except OSError as error:
            raise LawfulSourceRequiredError(
                "the configured restricted source is unavailable"
            ) from error
        if not source.is_file():
            raise LawfulSourceRequiredError("the configured restricted source is not a file")
        if not source.is_relative_to(root):
            raise LawfulSourceRequiredError(
                "the source must be inside the explicit restricted root"
            )
        if self.expected_source_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_source_sha256
        ):
            raise LawfulSourceRequiredError("expected source hash must be lowercase SHA-256")
        return source, root


class NovelSegmentationConfig(ImmutableRecord):
    """Frozen query-blind segmentation and retrieval configuration."""

    corpus_id: Identifier
    edition_label: NonEmptyText
    encoding: Literal["utf-8"] = "utf-8"
    unicode_normalization: Literal["NFC"] = "NFC"
    chapter_heading_pattern: NonEmptyText
    passage_target_characters: PositiveInt = 1_200
    passage_max_characters: PositiveInt = 1_800
    fts_tokenizer: Literal["unicode61 remove_diacritics 2"] = (
        "unicode61 remove_diacritics 2"
    )
    operational_top_k: PositiveInt = 12
    operational_max_evidence_tokens: PositiveInt = 6_144

    @model_validator(mode="after")
    def validate_segmentation(self) -> Self:
        if self.passage_target_characters > self.passage_max_characters:
            raise ValueError("passage target cannot exceed its hard maximum")
        try:
            re.compile(self.chapter_heading_pattern, flags=re.MULTILINE)
        except re.error as error:
            raise ValueError("chapter_heading_pattern is not a valid regular expression") from error
        return self


class RestrictedNovelIndexManifest(ImmutableRecord):
    """Path-free manifest for one restricted, query-blind FTS index."""

    manifest_id: Identifier
    corpus_id: Identifier
    edition_label: NonEmptyText
    source_sha256: Sha256Digest
    normalized_source_sha256: Sha256Digest
    index_sha256: Sha256Digest
    index_size_bytes: PositiveInt
    index_config_hash: Sha256Digest
    chapter_count: PositiveInt
    passage_count: PositiveInt
    normalized_character_count: PositiveInt
    built_at: datetime
    query_blind: Literal[True] = True
    rights_class: Literal[RightsClass.RESTRICTED_COPYRIGHTED] = (
        RightsClass.RESTRICTED_COPYRIGHTED
    )
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED


_CONTRAST_DIMENSIONS = frozenset(
    {
        "identity",
        "allegiance",
        "event_participation",
        "causality",
        "temporal_state",
        "belief_revelation",
    }
)


class CaseWindowRegistration(ImmutableRecord):
    """One output-blind bounded window with two same-horizon contexts."""

    window_id: Identifier
    chapter_ordinals: tuple[PositiveInt, ...]
    passage_ids: tuple[Identifier, ...]
    fixed_horizon: SpoilerHorizon
    contexts: tuple[QueryContext, QueryContext]
    contrast_dimensions: tuple[NonEmptyText, ...]
    selection_basis: NonEmptyText
    selected_before_condition_outputs: Literal[True] = True

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if not self.chapter_ordinals or tuple(sorted(set(self.chapter_ordinals))) != (
            self.chapter_ordinals
        ):
            raise ValueError("chapter ordinals must be nonempty, unique, and sorted")
        if not self.passage_ids or len(self.passage_ids) != len(set(self.passage_ids)):
            raise ValueError("a window needs unique bounded passage IDs")
        if self.contexts[0].context_id == self.contexts[1].context_id:
            raise ValueError("the two window contexts must have distinct IDs")
        if any(context.spoiler_horizon != self.fixed_horizon for context in self.contexts):
            raise ValueError("both window contexts must use the exact fixed horizon")
        if not self.contrast_dimensions or not set(self.contrast_dimensions).issubset(
            _CONTRAST_DIMENSIONS
        ):
            raise ValueError("window contrast dimensions are missing or unsupported")
        return self


class OperationalRetrievalRegistration(ImmutableRecord):
    """The sole full-index retrieval demonstration, excluded from causal contrasts."""

    operational_query_id: Identifier
    context: QueryContext
    frozen_fts_query: NonEmptyText
    top_k: PositiveInt
    max_evidence_tokens: PositiveInt
    condition: Literal[ConditionName.C2_LLM_QUERY] = ConditionName.C2_LLM_QUERY
    operational_only: Literal[True] = True
    causal_comparison_eligible: Literal[False] = False


class CaseStudyPreregistration(ImmutableRecord):
    """Exact four-window/eight-context design plus one operational query."""

    preregistration_id: Identifier
    restricted_index_manifest_hash: Sha256Digest
    windows: tuple[
        CaseWindowRegistration,
        CaseWindowRegistration,
        CaseWindowRegistration,
        CaseWindowRegistration,
    ]
    operational_retrieval: OperationalRetrievalRegistration
    detailed_review_context_ids: tuple[Identifier, Identifier, Identifier, Identifier]
    primary_seed: NonNegativeInt
    c1_window_preconstruction_count: Literal[4] = 4
    bounded_context_count: Literal[8] = 8
    primary_seed_count: Literal[1] = 1
    comparison_conditions: tuple[ConditionName, ConditionName, ConditionName] = (
        ConditionName.C0_CLASSICAL_PRE,
        ConditionName.C1_LLM_PRE,
        ConditionName.C2_LLM_QUERY,
    )
    frozen_at: datetime
    condition_outputs_inspected: Literal[False] = False
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def validate_complete_design(self) -> Self:
        window_ids = tuple(window.window_id for window in self.windows)
        if len(set(window_ids)) != 4:
            raise ValueError("the case study requires four distinct windows")
        context_ids = tuple(
            context.context_id for window in self.windows for context in window.contexts
        )
        if len(context_ids) != 8 or len(set(context_ids)) != 8:
            raise ValueError("the case study requires eight distinct bounded contexts")
        if self.operational_retrieval.context.context_id in context_ids:
            raise ValueError("the operational query must be separate from the eight comparisons")
        if len(set(self.detailed_review_context_ids)) != 4 or not set(
            self.detailed_review_context_ids
        ).issubset(context_ids):
            raise ValueError("exactly four bounded contexts require detailed matching review")
        if self.comparison_conditions != (
            ConditionName.C0_CLASSICAL_PRE,
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
        ):
            raise ValueError("bounded windows must compare C0, C1, and C2 in fixed order")
        return self


_REVIEW_JUDGMENTS = frozenset({"supported", "unsupported", "uncertain", "not_applicable"})


class DetailedCaseMatching(ImmutableRecord):
    matched_assertion_count: NonNegativeInt
    unmatched_assertion_count: NonNegativeInt
    matched_event_count: NonNegativeInt
    unmatched_event_count: NonNegativeInt
    permissible_alternative_ids: tuple[Identifier, ...] = ()
    matching_notes: NonEmptyText


class CaseUnitReview(ImmutableRecord):
    """Restricted review of one bounded query without source prose fields."""

    review_id: Identifier
    context_id: Identifier
    reviewer_id: Identifier
    evidence_support: NonEmptyText
    contextual_relevance: NonEmptyText
    temporal_integrity: NonEmptyText
    rare_pivotal_preservation: NonEmptyText
    high_level_organization: NonEmptyText
    detailed_matching: DetailedCaseMatching | None = None
    reviewed_at: datetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def judgments_are_registered(self) -> Self:
        values = (
            self.evidence_support,
            self.contextual_relevance,
            self.temporal_integrity,
            self.rare_pivotal_preservation,
            self.high_level_organization,
        )
        if not set(values).issubset(_REVIEW_JUDGMENTS):
            raise ValueError("case review judgments must use the frozen categorical vocabulary")
        return self


class CaseStudyReviewBundle(ImmutableRecord):
    preregistration_hash: Sha256Digest
    reviews: tuple[
        CaseUnitReview,
        CaseUnitReview,
        CaseUnitReview,
        CaseUnitReview,
        CaseUnitReview,
        CaseUnitReview,
        CaseUnitReview,
        CaseUnitReview,
    ]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def reviews_cover_registered_units(self) -> Self:
        context_ids = tuple(review.context_id for review in self.reviews)
        if len(context_ids) != 8 or len(set(context_ids)) != 8:
            raise ValueError("case review bundle must cover eight distinct contexts")
        return self


def validate_case_review_bundle(
    bundle: CaseStudyReviewBundle, preregistration: CaseStudyPreregistration
) -> None:
    """Require all eight broad reviews and the four preregistered detailed reviews."""

    if bundle.preregistration_hash != preregistration.content_hash:
        raise CasePreregistrationError("case review bundle references another preregistration")
    expected = {
        context.context_id for window in preregistration.windows for context in window.contexts
    }
    review_by_context = {review.context_id: review for review in bundle.reviews}
    if set(review_by_context) != expected:
        raise CasePreregistrationError("case reviews do not cover the eight bounded contexts")
    detailed = {
        context_id
        for context_id, review in review_by_context.items()
        if review.detailed_matching is not None
    }
    if detailed != set(preregistration.detailed_review_context_ids):
        raise CasePreregistrationError(
            "detailed assertion/event matching must cover exactly four preregistered contexts"
        )


class OperationalRetrievalReceipt(ImmutableRecord):
    """Safe metadata receipt; protected packet text is intentionally absent."""

    receipt_id: Identifier
    operational_query_id: Identifier
    restricted_index_manifest_hash: Sha256Digest
    packet_hash: Sha256Digest
    ordered_evidence_ids: tuple[Identifier, ...]
    omitted_evidence_ids: tuple[Identifier, ...]
    ranks: dict[Identifier, PositiveInt]
    scores: dict[Identifier, float]
    eligible_before_top_k: NonNegativeInt
    horizon_rejected_match_count: NonNegativeInt
    omitted_by_top_k_or_token_cap: NonNegativeInt
    query_accessed_at: datetime
    packet_created_at: datetime
    retrieval_method: Literal[RetrievalMethod.SQLITE_FTS5_BM25] = (
        RetrievalMethod.SQLITE_FTS5_BM25
    )
    operational_only: Literal[True] = True
    causal_comparison_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_retrieval_accounting(self) -> Self:
        evidence_ids = self.ordered_evidence_ids
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("operational receipt requires unique selected evidence IDs")
        if set(self.omitted_evidence_ids).intersection(evidence_ids):
            raise ValueError("selected and omitted evidence IDs must be disjoint")
        if len(self.omitted_evidence_ids) != len(set(self.omitted_evidence_ids)):
            raise ValueError("omitted evidence IDs must be unique")
        if set(self.ranks) != set(evidence_ids) or tuple(
            self.ranks[evidence_id] for evidence_id in evidence_ids
        ) != tuple(range(1, len(evidence_ids) + 1)):
            raise ValueError("operational ranks must exactly follow selected evidence order")
        if set(self.scores) != set(evidence_ids):
            raise ValueError("operational scores must cover exactly the selected evidence")
        if self.omitted_by_top_k_or_token_cap != len(self.omitted_evidence_ids):
            raise ValueError("operational omitted count must match omitted evidence IDs")
        if self.eligible_before_top_k != len(evidence_ids) + len(self.omitted_evidence_ids):
            raise ValueError("operational eligible count does not reconcile")
        if self.packet_created_at < self.query_accessed_at:
            raise ValueError("operational packet cannot predate query access")
        return self


@dataclass(frozen=True, slots=True)
class OperationalRetrievalResult:
    """In-memory protected packet paired with its prose-free receipt."""

    packet: EvidencePacket = field(repr=False)
    receipt: OperationalRetrievalReceipt


class PublicNovelIndexSummary(ImmutableRecord):
    manifest_id: Identifier
    corpus_id: Identifier
    edition_label: NonEmptyText
    source_sha256: Sha256Digest
    normalized_source_sha256: Sha256Digest
    index_config_hash: Sha256Digest
    chapter_count: PositiveInt
    passage_count: PositiveInt
    query_blind: Literal[True] = True
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC


class PublicNarrativeIllustration(ImmutableRecord):
    """Manually reviewed high-level paraphrase with only coarse/opaque anchors."""

    illustration_id: Identifier
    window_id: Identifier
    chapter_ordinals: tuple[PositiveInt, ...]
    opaque_evidence_ids: tuple[Identifier, ...]
    high_level_paraphrase: NonEmptyText
    interpretation_scope: NonEmptyText
    reviewed_for_public_release: Literal[True] = True
    contains_verbatim_source_text: Literal[False] = False
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC

    @model_validator(mode="after")
    def require_opaque_coarse_anchors(self) -> Self:
        if not self.chapter_ordinals:
            raise ValueError("public illustration requires a chapter-level locator")
        if not self.opaque_evidence_ids or any(
            not evidence_id.startswith("evidence-") for evidence_id in self.opaque_evidence_ids
        ):
            raise ValueError("public illustration evidence locators must use opaque IDs")
        if len(self.opaque_evidence_ids) != len(set(self.opaque_evidence_ids)):
            raise ValueError("public illustration evidence IDs must be unique")
        return self


class PublicCaseArtifact(ImmutableRecord):
    index_summary: PublicNovelIndexSummary
    illustrations: tuple[PublicNarrativeIllustration, ...]
    operational_receipt: OperationalRetrievalReceipt | None = None
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC


@dataclass(frozen=True, slots=True)
class WindowEvidenceBundle:
    """Protected in-memory window evidence; callers must not publicize it."""

    snapshot_assembly: EvidenceSnapshotAssembly = field(repr=False)
    packet: EvidencePacket = field(repr=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source(raw: bytes, encoding: str) -> str:
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as error:
        raise NovelIndexError("restricted source is not valid UTF-8") from error
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if text.startswith("\ufeff"):
        text = text[1:]
    if not text.strip():
        raise NovelIndexError("restricted source is empty")
    if "\x00" in text:
        raise NovelIndexError("restricted source contains NUL characters")
    return text


@dataclass(frozen=True, slots=True)
class _ChapterSlice:
    order: int
    heading: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _PassageSlice:
    passage_id: str
    chapter_id: str
    chapter_order: int
    passage_order: int
    start: int
    end: int
    text: str


def _chapter_slices(text: str, pattern: str) -> tuple[_ChapterSlice, ...]:
    matches = tuple(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise NovelIndexError("chapter heading pattern matched no chapters")
    slices: list[_ChapterSlice] = []
    if text[: matches[0].start()].strip():
        slices.append(
            _ChapterSlice(
                order=1,
                heading="",
                start=0,
                end=matches[0].start(),
            )
        )
    order_offset = len(slices)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        slices.append(
            _ChapterSlice(
                order=index + 1 + order_offset,
                heading=match.group(0).strip(),
                start=match.start(),
                end=end,
            )
        )
    return tuple(slices)


def _bounded_text_chunks(text: str, target: int, maximum: int) -> Iterator[tuple[int, int]]:
    paragraphs = tuple(
        match for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL)
    )
    pending_start: int | None = None
    pending_end: int | None = None
    for paragraph in paragraphs:
        start, end = paragraph.span()
        if end - start > maximum:
            if pending_start is not None and pending_end is not None:
                yield pending_start, pending_end
                pending_start = pending_end = None
            cursor = start
            while cursor < end:
                proposed = min(cursor + maximum, end)
                if proposed < end:
                    boundary = text.rfind(" ", cursor + 1, proposed + 1)
                    if boundary <= cursor:
                        boundary = proposed
                else:
                    boundary = end
                yield cursor, boundary
                cursor = boundary
                while cursor < end and text[cursor].isspace():
                    cursor += 1
            continue
        if pending_start is None:
            pending_start, pending_end = start, end
            continue
        assert pending_end is not None
        if end - pending_start <= maximum and pending_end - pending_start < target:
            pending_end = end
        else:
            yield pending_start, pending_end
            pending_start, pending_end = start, end
    if pending_start is not None and pending_end is not None:
        yield pending_start, pending_end


def _passage_slices(
    text: str,
    chapters: Sequence[_ChapterSlice],
    config: NovelSegmentationConfig,
) -> tuple[_PassageSlice, ...]:
    passages: list[_PassageSlice] = []
    passage_order = 0
    for chapter in chapters:
        chapter_text = text[chapter.start : chapter.end]
        for local_start, local_end in _bounded_text_chunks(
            chapter_text,
            config.passage_target_characters,
            config.passage_max_characters,
        ):
            passage_text = chapter_text[local_start:local_end].strip()
            if not passage_text:
                continue
            passage_order += 1
            absolute_start = chapter.start + local_start
            absolute_end = absolute_start + len(passage_text)
            passages.append(
                _PassageSlice(
                    passage_id=f"passage-{passage_order:06d}",
                    chapter_id=f"chapter-{chapter.order:04d}",
                    chapter_order=chapter.order,
                    passage_order=passage_order,
                    start=absolute_start,
                    end=absolute_end,
                    text=passage_text,
                )
            )
    if not passages:
        raise NovelIndexError("segmentation produced no passages")
    return tuple(passages)


def _initialize_index(connection: sqlite3.Connection, tokenizer: str) -> None:
    try:
        connection.executescript(
            f"""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT;
            CREATE TABLE chapters (
                chapter_id TEXT PRIMARY KEY,
                chapter_order INTEGER NOT NULL UNIQUE,
                heading_hash TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL
            ) STRICT;
            CREATE TABLE passages (
                rowid INTEGER PRIMARY KEY,
                passage_id TEXT NOT NULL UNIQUE,
                chapter_id TEXT NOT NULL REFERENCES chapters(chapter_id),
                chapter_order INTEGER NOT NULL,
                passage_order INTEGER NOT NULL UNIQUE,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                text TEXT NOT NULL,
                text_hash TEXT NOT NULL
            ) STRICT;
            CREATE VIRTUAL TABLE passage_fts USING fts5(
                text,
                content='passages',
                content_rowid='rowid',
                tokenize='{tokenizer}'
            );
            """
        )
    except sqlite3.OperationalError as error:
        raise NovelIndexError(
            "SQLite FTS5 is unavailable or index schema creation failed"
        ) from error


def _assert_restricted_destination(destination: Path, restricted_root: Path) -> Path:
    if not destination.is_absolute():
        raise NovelIndexError("restricted index destination must be explicit")
    resolved_parent = destination.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(restricted_root):
        raise NovelIndexError("restricted index must remain inside the restricted root")
    if destination.exists() or destination.is_symlink():
        raise NovelIndexError("refusing to overwrite an existing restricted index")
    return resolved_parent / destination.name


def build_restricted_novel_index(
    authorization: LawfulNovelSourceAuthorization,
    config: NovelSegmentationConfig,
    *,
    destination: Path,
    built_at: datetime,
) -> RestrictedNovelIndexManifest:
    """Build one query-blind external-content FTS index atomically.

    No normalized text file, packet, snippet, or offset map is emitted.  The one
    SQLite database is restricted and stores passage text once at the SQL level;
    FTS uses that table as its external content source.
    """

    if built_at.tzinfo is None or built_at.utcoffset() is None:
        raise NovelIndexError("built_at must be timezone-aware")
    source, restricted_root = authorization.resolve()
    destination = _assert_restricted_destination(destination, restricted_root)
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if (
        authorization.expected_source_sha256 is not None
        and source_hash != authorization.expected_source_sha256
    ):
        raise LawfulSourceRequiredError("configured source hash does not match the lawful copy")
    normalized = _normalize_source(raw, config.encoding)
    normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    chapters = _chapter_slices(normalized, config.chapter_heading_pattern)
    passages = _passage_slices(normalized, chapters, config)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary)
        try:
            _initialize_index(connection, config.fts_tokenizer)
            metadata = {
                "schema_version": "1.0.0",
                "corpus_id": config.corpus_id,
                "edition_label": config.edition_label,
                "source_sha256": source_hash,
                "normalized_source_sha256": normalized_hash,
                "index_config_hash": config.content_hash,
                "query_blind": "true",
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", sorted(metadata.items())
            )
            connection.executemany(
                "INSERT INTO chapters VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        f"chapter-{chapter.order:04d}",
                        chapter.order,
                        hashlib.sha256(chapter.heading.encode("utf-8")).hexdigest(),
                        chapter.start,
                        chapter.end,
                    )
                    for chapter in chapters
                ),
            )
            connection.executemany(
                """INSERT INTO passages(
                    passage_id, chapter_id, chapter_order, passage_order,
                    start_char, end_char, text, text_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    (
                        passage.passage_id,
                        passage.chapter_id,
                        passage.chapter_order,
                        passage.passage_order,
                        passage.start,
                        passage.end,
                        passage.text,
                        hashlib.sha256(passage.text.encode("utf-8")).hexdigest(),
                    )
                    for passage in passages
                ),
            )
            connection.execute("INSERT INTO passage_fts(passage_fts) VALUES ('rebuild')")
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise NovelIndexError("restricted index failed SQLite integrity_check")
            fts_count = connection.execute("SELECT count(*) FROM passage_fts").fetchone()[0]
            if fts_count != len(passages):
                raise NovelIndexError("FTS index does not cover every segmented passage")
        finally:
            connection.close()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    index_hash = _sha256_file(destination)
    return RestrictedNovelIndexManifest(
        manifest_id=f"novel-index-{index_hash[:16]}",
        corpus_id=config.corpus_id,
        edition_label=config.edition_label,
        source_sha256=source_hash,
        normalized_source_sha256=normalized_hash,
        index_sha256=index_hash,
        index_size_bytes=destination.stat().st_size,
        index_config_hash=config.content_hash,
        chapter_count=len(chapters),
        passage_count=len(passages),
        normalized_character_count=len(normalized),
        built_at=built_at,
    )


def verify_restricted_novel_index(
    index_path: Path, manifest: RestrictedNovelIndexManifest
) -> None:
    """Verify path-free manifest hashes and the query-blind restricted schema."""

    if not index_path.is_file() or index_path.is_symlink():
        raise NovelIndexError("configured restricted index is unavailable")
    if _sha256_file(index_path) != manifest.index_sha256:
        raise NovelIndexError("restricted index hash does not match its manifest")
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise NovelIndexError("restricted index failed integrity_check")
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        expected = {
            "corpus_id": manifest.corpus_id,
            "edition_label": manifest.edition_label,
            "source_sha256": manifest.source_sha256,
            "normalized_source_sha256": manifest.normalized_source_sha256,
            "index_config_hash": manifest.index_config_hash,
            "query_blind": "true",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise NovelIndexError("restricted index metadata does not match its manifest")
        chapter_count = connection.execute("SELECT count(*) FROM chapters").fetchone()[0]
        passage_count = connection.execute("SELECT count(*) FROM passages").fetchone()[0]
        if (chapter_count, passage_count) != (manifest.chapter_count, manifest.passage_count):
            raise NovelIndexError("restricted index row counts do not match its manifest")
    finally:
        connection.close()


def validate_case_preregistration(
    preregistration: CaseStudyPreregistration,
    manifest: RestrictedNovelIndexManifest,
    *,
    index_path: Path | None = None,
) -> None:
    """Bind four windows and eight contexts to one immutable restricted index."""

    if preregistration.restricted_index_manifest_hash != manifest.content_hash:
        raise CasePreregistrationError("preregistration references another restricted index")
    if preregistration.frozen_at < manifest.built_at:
        raise CasePreregistrationError("preregistration cannot predate the index")
    if index_path is None:
        return
    verify_restricted_novel_index(index_path, manifest)
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        rows = {
            row[0]: (row[1], row[2])
            for row in connection.execute(
                "SELECT passage_id, chapter_order, passage_order FROM passages"
            )
        }
    finally:
        connection.close()
    for window in preregistration.windows:
        unknown = set(window.passage_ids) - set(rows)
        if unknown:
            raise CasePreregistrationError("a window references unknown passage IDs")
        realized_chapters = tuple(sorted({rows[item][0] for item in window.passage_ids}))
        if realized_chapters != window.chapter_ordinals:
            raise CasePreregistrationError("window chapter locators do not match passage IDs")
        max_order = max(rows[item][1] for item in window.passage_ids)
        if max_order > window.fixed_horizon.max_discourse_position.passage_order:
            raise CasePreregistrationError("window includes evidence beyond its fixed horizon")


def _restricted_records(
    connection: sqlite3.Connection,
    passage_ids: Sequence[str],
    *,
    source_artifact_hash: str,
) -> tuple[EvidenceRecord, ...]:
    if not passage_ids:
        raise NovelIndexError("evidence materialization requires passage IDs")
    placeholders = ",".join("?" for _ in passage_ids)
    rows = connection.execute(
        f"""SELECT passage_id, passage_order, text, text_hash
            FROM passages WHERE passage_id IN ({placeholders})
            ORDER BY passage_order""",
        tuple(passage_ids),
    ).fetchall()
    if len(rows) != len(set(passage_ids)):
        raise NovelIndexError("requested evidence contains unknown passage IDs")
    records: list[EvidenceRecord] = []
    for passage_id, passage_order, text, text_hash in rows:
        evidence_id = passage_id.replace("passage-", "evidence-", 1)
        records.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                passage_id=passage_id,
                text=text,
                text_hash=text_hash,
                discourse_position={"passage_order": passage_order},
                provenance=ProvenanceReference(
                    provenance_id=f"provenance-{evidence_id}",
                    evidence_id=evidence_id,
                    extraction_method="query-blind passage segmentation",
                    locator=f"restricted://first-novel/{passage_id}",
                    source_artifact_hash=source_artifact_hash,
                    confidence=1.0,
                ),
                confidence=1.0,
                release_class=ReleaseClass.RESTRICTED,
            )
        )
    return tuple(records)


def materialize_window_evidence(
    *,
    index_path: Path,
    manifest: RestrictedNovelIndexManifest,
    window: CaseWindowRegistration,
    created_at: datetime,
    sealed_at: datetime,
    packet_created_at: datetime,
    token_counter: Callable[[str], int],
) -> WindowEvidenceBundle:
    """Materialize one all-admissible protected packet without persisting prose."""

    verify_restricted_novel_index(index_path, manifest)
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        records = _restricted_records(
            connection,
            window.passage_ids,
            source_artifact_hash=manifest.source_sha256,
        )
        passage_rows = connection.execute(
            """SELECT passage_id, passage_order, text_hash FROM passages
               WHERE passage_id IN ({}) ORDER BY passage_order""".format(
                ",".join("?" for _ in window.passage_ids)
            ),
            window.passage_ids,
        ).fetchall()
    finally:
        connection.close()
    document = Document(
        document_id="first-novel",
        corpus_id=manifest.corpus_id,
        edition=manifest.edition_label,
        restricted_text_handle="restricted://first-novel/source",
        source_text_hash=manifest.source_sha256,
        rights_class=RightsClass.RESTRICTED_COPYRIGHTED,
        release_class=ReleaseClass.RESTRICTED,
    )
    passages = tuple(
        Passage(
            passage_id=passage_id,
            document_id=document.document_id,
            passage_order=passage_order,
            restricted_text_handle=f"restricted://first-novel/{passage_id}",
            source_text_hash=text_hash,
            discourse_position={"passage_order": passage_order},
            release_class=ReleaseClass.RESTRICTED,
        )
        for passage_id, passage_order, text_hash in passage_rows
    )
    assembly = build_evidence_snapshot(
        snapshot_id=f"snapshot-{window.window_id}",
        corpus_id=manifest.corpus_id,
        world_or_window_id=window.window_id,
        horizon=window.fixed_horizon,
        documents=(document,),
        passages=passages,
        evidence_records=records,
        index_config_hash=manifest.index_config_hash,
        created_at=created_at,
        sealed_at=sealed_at,
        release_class=ReleaseClass.RESTRICTED,
    )
    packet = build_all_admissible_packet(
        assembly,
        packet_id=f"packet-{window.window_id}",
        created_at=packet_created_at,
        token_counter=token_counter,
    )
    return WindowEvidenceBundle(snapshot_assembly=assembly, packet=packet)


def _fts_expression(query: str) -> str:
    terms = tuple(dict.fromkeys(re.findall(r"[\w]+", unicodedata.normalize("NFC", query))))
    if not terms:
        raise NovelIndexError("operational FTS query contains no searchable terms")
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def retrieve_operational_packet(
    *,
    index_path: Path,
    manifest: RestrictedNovelIndexManifest,
    registration: OperationalRetrievalRegistration,
    query_accessed_at: datetime,
    packet_created_at: datetime,
    token_counter: Callable[[str], int],
) -> OperationalRetrievalResult:
    """Run the one bounded full-index BM25 demonstration after query access."""

    if query_accessed_at.tzinfo is None or packet_created_at.tzinfo is None:
        raise NovelIndexError("operational retrieval timestamps must be timezone-aware")
    if packet_created_at < query_accessed_at:
        raise NovelIndexError("operational packet cannot predate query access")
    verify_restricted_novel_index(index_path, manifest)
    horizon_order = registration.context.spoiler_horizon.max_discourse_position.passage_order
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        expression = _fts_expression(registration.frozen_fts_query)
        ranked_matches = connection.execute(
            """SELECT passages.passage_id, passages.passage_order,
                      bm25(passage_fts) AS score
               FROM passage_fts JOIN passages ON passages.rowid = passage_fts.rowid
               WHERE passage_fts MATCH ? AND passages.passage_order <= ?
               ORDER BY score ASC, passages.passage_order ASC""",
            (expression, horizon_order),
        ).fetchall()
        horizon_rejected_match_count = connection.execute(
            """SELECT count(*) FROM passage_fts
               JOIN passages ON passages.rowid = passage_fts.rowid
               WHERE passage_fts MATCH ? AND passages.passage_order > ?""",
            (expression, horizon_order),
        ).fetchone()[0]
        candidate_matches = ranked_matches[: registration.top_k]
        candidate_ids = tuple(row[0] for row in candidate_matches)
        if not candidate_ids:
            raise NovelIndexError("operational FTS retrieval returned no evidence")
        candidate_evidence = _restricted_records(
            connection,
            candidate_ids,
            source_artifact_hash=manifest.source_sha256,
        )
    finally:
        connection.close()

    evidence_by_passage = {item.passage_id: item for item in candidate_evidence}
    selected: list[tuple[EvidenceRecord, float]] = []
    selected_token_count = 0
    for passage_id, _, score in candidate_matches:
        record = evidence_by_passage[passage_id]
        prospective = [*[item[0] for item in selected], record]
        count = token_counter(
            canonical_json(tuple(to_model_visible_evidence(item) for item in prospective))
        )
        if count > registration.max_evidence_tokens:
            break
        selected.append((record, float(score)))
        selected_token_count = count
    if not selected:
        raise NovelIndexError("top operational passage cannot fit the frozen token cap")
    ordered_evidence = tuple(item[0] for item in selected)
    evidence_ids = tuple(item.evidence_id for item in ordered_evidence)
    scores = {
        item.evidence_id: score
        for item, (_, score) in zip(ordered_evidence, selected, strict=True)
    }
    ranks = {evidence_id: rank for rank, evidence_id in enumerate(evidence_ids, start=1)}
    omitted_evidence_ids = tuple(
        passage_id.replace("passage-", "evidence-", 1)
        for passage_id, _, _ in ranked_matches[len(selected) :]
    )
    packet = EvidencePacket(
        packet_id=f"packet-{registration.operational_query_id}",
        snapshot_hash=manifest.content_hash,
        evidence=ordered_evidence,
        ordered_evidence_ids=evidence_ids,
        retrieval_method=RetrievalMethod.SQLITE_FTS5_BM25,
        ranks=ranks,
        scores=scores,
        token_count=selected_token_count,
        horizon_rejections=(),
        created_at=packet_created_at,
        release_class=ReleaseClass.RESTRICTED,
    )
    receipt = OperationalRetrievalReceipt(
        receipt_id=f"receipt-{registration.operational_query_id}",
        operational_query_id=registration.operational_query_id,
        restricted_index_manifest_hash=manifest.content_hash,
        packet_hash=packet.content_hash,
        ordered_evidence_ids=evidence_ids,
        omitted_evidence_ids=omitted_evidence_ids,
        ranks=ranks,
        scores=scores,
        eligible_before_top_k=len(ranked_matches),
        horizon_rejected_match_count=horizon_rejected_match_count,
        omitted_by_top_k_or_token_cap=len(omitted_evidence_ids),
        query_accessed_at=query_accessed_at,
        packet_created_at=packet_created_at,
    )
    return OperationalRetrievalResult(packet=packet, receipt=receipt)


_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "source_path",
        "restricted_root",
        "restricted_text_handle",
        "text",
        "raw_text",
        "passage_text",
        "snippet",
        "quote",
        "start_char",
        "end_char",
        "offset",
        "offsets",
        "index_path",
        "fts_path",
    }
)


def _iter_public_strings(value: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, ImmutableRecord):
        yield from _iter_public_strings(value.model_dump(mode="python"), path)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in _FORBIDDEN_PUBLIC_KEYS:
                raise PublicReleaseViolation(
                    f"public case artifact contains forbidden field {key!r}"
                )
            yield from _iter_public_strings(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _iter_public_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def scan_public_case_artifact(
    value: PublicCaseArtifact,
    *,
    restricted_index_path: Path | None = None,
    minimum_verbatim_words: int = 8,
) -> None:
    """Reject restricted fields, paths, and reconstructive source overlap."""

    if minimum_verbatim_words < 6:
        raise ValueError("minimum_verbatim_words cannot be less than six")
    strings = tuple(_iter_public_strings(value))
    for path, item in strings:
        if item.startswith(("/", "~", "file://", "restricted://")):
            raise PublicReleaseViolation(
                f"public case artifact contains a private locator at {path}"
            )
        if "SQLite format 3" in item or "CREATE VIRTUAL TABLE passage_fts" in item:
            raise PublicReleaseViolation(
                f"public case artifact contains reconstructive index data at {path}"
            )
    if restricted_index_path is None:
        return
    connection = sqlite3.connect(f"file:{restricted_index_path}?mode=ro", uri=True)
    try:
        source_ngrams: set[str] = set()
        for (text,) in connection.execute("SELECT text FROM passages"):
            words = re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())
            source_ngrams.update(
                " ".join(words[index : index + minimum_verbatim_words])
                for index in range(len(words) - minimum_verbatim_words + 1)
            )
    finally:
        connection.close()
    for path, item in strings:
        words = re.findall(r"\w+", unicodedata.normalize("NFC", item).casefold())
        for index in range(len(words) - minimum_verbatim_words + 1):
            phrase = " ".join(words[index : index + minimum_verbatim_words])
            if phrase in source_ngrams:
                raise PublicReleaseViolation(
                    f"public case artifact contains verbatim source overlap at {path}"
                )


def build_public_case_artifact(
    *,
    manifest: RestrictedNovelIndexManifest,
    illustrations: Sequence[PublicNarrativeIllustration],
    operational_receipt: OperationalRetrievalReceipt | None = None,
    restricted_index_path: Path | None = None,
) -> PublicCaseArtifact:
    """Create and scan the only supported public first-novel export shape."""

    artifact = PublicCaseArtifact(
        index_summary=PublicNovelIndexSummary(
            manifest_id=manifest.manifest_id,
            corpus_id=manifest.corpus_id,
            edition_label=manifest.edition_label,
            source_sha256=manifest.source_sha256,
            normalized_source_sha256=manifest.normalized_source_sha256,
            index_config_hash=manifest.index_config_hash,
            chapter_count=manifest.chapter_count,
            passage_count=manifest.passage_count,
        ),
        illustrations=tuple(illustrations),
        operational_receipt=operational_receipt,
    )
    scan_public_case_artifact(artifact, restricted_index_path=restricted_index_path)
    return artifact


def load_segmentation_config(path: Path) -> NovelSegmentationConfig:
    return NovelSegmentationConfig.model_validate_json(path.read_text(encoding="utf-8"))


def write_restricted_manifest(
    manifest: RestrictedNovelIndexManifest,
    destination: Path,
    *,
    restricted_root: Path,
) -> None:
    """Atomically write a restricted manifest beside the restricted index."""

    try:
        root = restricted_root.resolve(strict=True)
        parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise NovelIndexError("restricted manifest parent is unavailable") from error
    if not destination.is_absolute() or not parent.is_relative_to(root):
        raise NovelIndexError("restricted manifest must remain inside the restricted root")
    if destination.exists() or destination.is_symlink():
        raise NovelIndexError("refusing to overwrite an existing restricted manifest")
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def public_contract_hashes() -> dict[str, str]:
    """Hash case-study JSON Schemas without creating a corpus or annotations."""

    contracts = (
        NovelSegmentationConfig,
        CaseStudyPreregistration,
        PublicCaseArtifact,
        OperationalRetrievalReceipt,
    )
    return {
        contract.__name__: canonical_sha256(contract.model_json_schema(mode="validation"))
        for contract in contracts
    }


def common_case_budgets(preregistration: CaseStudyPreregistration) -> tuple[OutputBudgets, ...]:
    """Expose the eight fixed output budgets for runner fairness checks."""

    return tuple(
        context.budgets
        for window in preregistration.windows
        for context in window.contexts
    )
