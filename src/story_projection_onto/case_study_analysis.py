"""Copyright-safe descriptive analysis for the bounded first-novel study.

This module cannot create reviewer judgments.  It accepts only a complete human
review that has been validated against all terminal ITT receipts, then emits the
canonical ``novel_case`` CSV consumed by Phase 7.  Public rows contain counts,
scores, registered opaque identifiers, and fixed interpretation labels only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.case_study_runtime import (
    CaseDetailedMatchSummary,
    CaseReviewVerdict,
    CaseStudyCompletedReview,
    CaseStudyExecutionPlan,
    CaseStudyResumeError,
    CaseStudyResumeManifest,
    CaseStudyReviewInputTemplate,
    load_case_study_execution_plan,
    load_case_study_resume_manifest,
    load_completed_case_study_review,
    validate_completed_case_study_review,
    write_restricted_case_record,
)
from story_projection_onto.contracts import (
    Identifier,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
    canonical_sha256,
)
from story_projection_onto.novel_case import RestrictedNovelIndexManifest
from story_projection_onto.public_release import (
    PublicReleaseError,
    load_protected_prose_canaries,
    scan_public_bytes,
)

NOVEL_CASE_COLUMNS = (
    "scope",
    "window_id",
    "context_id",
    "condition",
    "metric",
    "value",
    "status",
    "interpretation",
)
OPERATIONAL_WINDOW_IDENTIFIER = "complete-query-blind-index"


class CaseStudyNarrativeAnalysisReceipt(ImmutableRecord):
    analysis_id: Identifier
    execution_plan_hash: Sha256Digest
    restricted_index_manifest_hash: Sha256Digest
    corpus_source_sha256: Sha256Digest
    terminal_resume_manifest_hash: Sha256Digest
    completed_review_hash: Sha256Digest
    public_alias_manifest_hash: Sha256Digest
    restricted_table_file_sha256: Sha256Digest
    publication_status: Literal["restricted_pending_canaries", "public_scan_passed"]
    public_table_file_sha256: Sha256Digest | None = None
    protected_canary_manifest_hash: Sha256Digest | None = None
    release_scan_receipt_hash: Sha256Digest | None = None
    public_row_inventory_hash: Sha256Digest
    public_row_count: int = Field(gt=0)
    bounded_context_count: Literal[8] = 8
    bounded_output_count: Literal[24] = 24
    operational_output_count: Literal[1] = 1
    causal_and_operational_rows_kept_separate: Literal[True] = True
    protected_prose_exported: Literal[False] = False
    reconstructive_offsets_exported: Literal[False] = False
    population_inference_performed: Literal[False] = False
    compiled_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def publication_requires_a_real_canary_scan(self) -> CaseStudyNarrativeAnalysisReceipt:
        public_fields = (
            self.public_table_file_sha256,
            self.protected_canary_manifest_hash,
            self.release_scan_receipt_hash,
        )
        if self.publication_status == "public_scan_passed":
            if any(value is None for value in public_fields):
                raise ValueError("public narrative table lacks its protected-prose scan lineage")
        elif any(value is not None for value in public_fields):
            raise ValueError("pending narrative output cannot claim public scan artifacts")
        return self


class CaseStudyPublicAliasManifest(ImmutableRecord):
    """Restricted deterministic map from plan IDs to stable public aliases."""

    manifest_id: Identifier
    execution_plan_hash: Sha256Digest
    source_identifier_inventory_hash: Sha256Digest
    window_aliases: Mapping[Identifier, Identifier]
    context_aliases: Mapping[Identifier, Identifier]
    derivation_method: Literal["sorted_execution_plan_identifiers_v1"] = (
        "sorted_execution_plan_identifiers_v1"
    )
    condition_outputs_used: Literal[False] = False
    derived_at: AwareDatetime
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def aliases_are_opaque_and_unique(self) -> CaseStudyPublicAliasManifest:
        if len(self.window_aliases) != 5 or len(self.context_aliases) != 9:
            raise ValueError("public alias freeze requires five windows and nine contexts")
        aliases = tuple(self.window_aliases.values()) + tuple(self.context_aliases.values())
        if len(aliases) != len(set(aliases)):
            raise ValueError("public case aliases must be globally unique")
        if any(alias in self.window_aliases or alias in self.context_aliases for alias in aliases):
            raise ValueError("a public alias must not reproduce a restricted identifier")
        expected_windows = {f"novel-window-{index:02d}" for index in range(1, 6)}
        expected_contexts = {f"novel-context-{index:02d}" for index in range(1, 10)}
        if set(self.window_aliases.values()) != expected_windows:
            raise ValueError("window aliases differ from the frozen opaque inventory")
        if set(self.context_aliases.values()) != expected_contexts:
            raise ValueError("context aliases differ from the frozen opaque inventory")
        return self


def _real_directory(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    probe = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        probe /= component
        if probe.is_symlink():
            raise CaseStudyResumeError(f"{label} cannot traverse a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise CaseStudyResumeError(f"{label} is unavailable") from error
    if resolved != lexical or not resolved.is_dir():
        raise CaseStudyResumeError(f"{label} must be one real directory")
    return resolved


def _contained_output(root: Path, path: Path, *, label: str) -> Path:
    root = _real_directory(root, label=f"{label} root")
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise CaseStudyResumeError(f"{label} must remain inside its declared root") from error
    if relative == Path("."):
        raise CaseStudyResumeError(f"{label} must name a file below its root")
    probe = root
    for component in relative.parts:
        probe /= component
        if probe.is_symlink():
            raise CaseStudyResumeError(f"{label} cannot traverse a symbolic link")
    lexical.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if lexical.parent.resolve(strict=True) != lexical.parent:
        raise CaseStudyResumeError(f"{label} parent is not a stable real path")
    if lexical.exists() and (lexical.is_symlink() or not lexical.is_file()):
        raise CaseStudyResumeError(f"{label} must be a regular non-symlink file")
    return lexical


def _publish_once(path: Path, payload: bytes, *, label: str) -> None:
    if path.is_symlink():
        raise CaseStudyResumeError(f"append-only {label} cannot be a symbolic link")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise CaseStudyResumeError(f"append-only {label} changed")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise CaseStudyResumeError(f"concurrent append-only {label} changed") from None
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _score(verdict: CaseReviewVerdict) -> str:
    return {
        CaseReviewVerdict.MEETS: "1",
        CaseReviewVerdict.PARTIALLY_MEETS: "0.5",
        CaseReviewVerdict.DOES_NOT_MEET: "0",
        CaseReviewVerdict.NOT_ASSESSABLE: "",
    }[verdict]


def _deterministic_aliases(
    identifiers: set[str], *, prefix: Literal["novel-window", "novel-context"]
) -> dict[str, str]:
    """Remove post-output alias discretion by ordering the frozen private IDs."""

    return {
        identifier: f"{prefix}-{ordinal:02d}"
        for ordinal, identifier in enumerate(sorted(identifiers), 1)
    }


def derive_case_study_public_alias_manifest(
    plan: CaseStudyExecutionPlan,
    *,
    derived_at: datetime,
) -> CaseStudyPublicAliasManifest:
    """Derive aliases only from identifiers already sealed in the case plan."""

    if derived_at.tzinfo is None or derived_at.utcoffset() is None:
        raise CaseStudyResumeError("alias derivation time must be timezone-aware")
    window_ids = {window.window_id for window in plan.windows} | {
        OPERATIONAL_WINDOW_IDENTIFIER
    }
    context_ids = {
        context_id for window in plan.windows for context_id in window.context_ids
    } | {plan.operational.context_id}
    identifier_inventory_hash = canonical_sha256(
        {
            "execution_plan_hash": plan.content_hash,
            "window_ids": tuple(sorted(window_ids)),
            "context_ids": tuple(sorted(context_ids)),
            "operational_window_identifier": OPERATIONAL_WINDOW_IDENTIFIER,
        }
    )
    return CaseStudyPublicAliasManifest(
        manifest_id=f"case-public-aliases-{plan.content_hash[:20]}",
        execution_plan_hash=plan.content_hash,
        source_identifier_inventory_hash=identifier_inventory_hash,
        window_aliases=_deterministic_aliases(window_ids, prefix="novel-window"),
        context_aliases=_deterministic_aliases(context_ids, prefix="novel-context"),
        derived_at=derived_at,
    )


def _ratio(numerator: int, denominator: int) -> tuple[str, str]:
    if denominator == 0:
        return "", "not_assessable"
    return f"{numerator / denominator:.6f}".rstrip("0").rstrip("."), "complete"


def _matching_rows(
    *,
    window_id: str,
    context_id: str,
    matching: CaseDetailedMatchSummary,
) -> list[dict[str, str]]:
    bases = (
        (
            "assertion",
            matching.matched_assertion_count,
            matching.output_assertion_count,
            matching.reference_assertion_count,
        ),
        (
            "event",
            matching.matched_event_count,
            matching.output_event_count,
            matching.reference_event_count,
        ),
    )
    rows: list[dict[str, str]] = []
    for label, matches, output_count, reference_count in bases:
        precision, precision_status = _ratio(matches, output_count)
        recall, recall_status = _ratio(matches, reference_count)
        denominator = output_count + reference_count
        f1, f1_status = _ratio(2 * matches, denominator)
        for metric, value, status in (
            (f"detailed_{label}_precision", precision, precision_status),
            (f"detailed_{label}_recall", recall, recall_status),
            (f"detailed_{label}_f1", f1, f1_status),
        ):
            rows.append(
                {
                    "scope": "bounded_same_evidence",
                    "window_id": window_id,
                    "context_id": context_id,
                    "condition": matching.condition.value,
                    "metric": metric,
                    "value": value,
                    "status": status,
                    "interpretation": "descriptive_review_matching_no_population_inference",
                }
            )
    return rows


def _canonical_csv(rows: list[dict[str, str]]) -> bytes:
    ordered = sorted(
        rows,
        key=lambda row: tuple(row[column] for column in NOVEL_CASE_COLUMNS[:5]),
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(NOVEL_CASE_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(ordered)
    return buffer.getvalue().encode("utf-8")


def compile_case_study_narrative_analysis(
    *,
    restricted_root: Path,
    public_root: Path,
    plan_path: Path,
    restricted_index_manifest_path: Path,
    terminal_resume_path: Path,
    review_template_path: Path,
    completed_review_path: Path,
    public_alias_manifest_path: Path,
    restricted_table_path: Path,
    protected_canary_manifest_path: Path | None,
    protected_corpus_path: Path | None,
    public_table_path: Path | None,
    restricted_receipt_path: Path,
    analysis_id: str,
    compiled_at: datetime,
) -> CaseStudyNarrativeAnalysisReceipt:
    """Validate completed review lineage and emit a copyright-safe canonical table."""

    if compiled_at.tzinfo is None or compiled_at.utcoffset() is None:
        raise CaseStudyResumeError("narrative analysis time must be timezone-aware")
    plan = load_case_study_execution_plan(plan_path, restricted_root=restricted_root)
    index_manifest_file = _contained_existing_restricted_file(
        restricted_root,
        restricted_index_manifest_path,
        label="restricted novel index manifest",
    )
    try:
        index_manifest = RestrictedNovelIndexManifest.model_validate_json(
            index_manifest_file.read_bytes()
        )
    except Exception as error:
        raise CaseStudyResumeError("invalid restricted novel index manifest") from error
    if (
        index_manifest.content_hash != plan.restricted_index_manifest_hash
        or hashlib.sha256(index_manifest_file.read_bytes()).hexdigest()
        != plan.restricted_index_manifest_file_sha256
    ):
        raise CaseStudyResumeError("narrative analysis uses another restricted novel index")
    resume: CaseStudyResumeManifest = load_case_study_resume_manifest(
        terminal_resume_path, restricted_root=restricted_root
    )
    template_path = Path(os.path.abspath(review_template_path))
    template_file = _contained_existing_restricted_file(
        restricted_root, template_path, label="case review template"
    )
    template = CaseStudyReviewInputTemplate.model_validate_json(template_file.read_bytes())
    review: CaseStudyCompletedReview = load_completed_case_study_review(
        completed_review_path, restricted_root=restricted_root
    )
    validate_completed_case_study_review(
        plan=plan,
        resume=resume,
        template=template,
        review=review,
    )
    if compiled_at <= review.completed_at:
        raise CaseStudyResumeError("narrative analysis must follow completed human review")
    operational_access = next(
        item for item in resume.query_access_receipts if item.operational_only
    )
    if operational_access.window_id != OPERATIONAL_WINDOW_IDENTIFIER:
        raise CaseStudyResumeError("operational output uses an unknown window identifier")
    expected_windows = {window.window_id for window in plan.windows} | {
        OPERATIONAL_WINDOW_IDENTIFIER
    }
    expected_contexts = {
        context_id for window in plan.windows for context_id in window.context_ids
    } | {plan.operational.context_id}
    expected_aliases = derive_case_study_public_alias_manifest(
        plan,
        derived_at=compiled_at,
    )
    alias_path = _contained_output(
        restricted_root,
        public_alias_manifest_path,
        label="case public alias manifest",
    )
    if alias_path.exists():
        try:
            aliases = CaseStudyPublicAliasManifest.model_validate_json(
                alias_path.read_bytes()
            )
        except Exception as error:
            raise CaseStudyResumeError("invalid case public alias manifest") from error
        expected_payload = expected_aliases.model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        expected_payload["derived_at"] = aliases.derived_at
        expected_static = CaseStudyPublicAliasManifest.model_validate(expected_payload)
        if aliases != expected_static or aliases.derived_at < plan.compiled_at:
            raise CaseStudyResumeError(
                "public aliases are not the deterministic execution-plan mapping"
            )
    else:
        aliases = expected_aliases
        write_restricted_case_record(
            aliases,
            alias_path,
            restricted_root=restricted_root,
        )
    receipts = {item.projection_job_id: item for item in resume.output_receipts}
    rows: list[dict[str, str]] = []
    for unit in review.units:
        for judgment in unit.primary.judgments:
            receipt = receipts[judgment.projection_job_id]
            rows.append(
                {
                    "scope": "bounded_same_evidence",
                    "window_id": aliases.window_aliases[unit.window_id],
                    "context_id": aliases.context_aliases[unit.context_id],
                    "condition": judgment.condition.value,
                    "metric": f"review_{judgment.dimension.value}",
                    "value": _score(judgment.verdict),
                    "status": (
                        "complete"
                        if judgment.verdict is not CaseReviewVerdict.NOT_ASSESSABLE
                        else receipt.terminal_outcome.value
                    ),
                    "interpretation": "descriptive_human_review_no_population_inference",
                }
            )
        for matching in unit.primary.detailed_matching:
            rows.extend(
                _matching_rows(
                    window_id=aliases.window_aliases[unit.window_id],
                    context_id=aliases.context_aliases[unit.context_id],
                    matching=matching,
                )
            )
        if unit.secondary is not None:
            primary = {
                (item.projection_job_id, item.dimension): item
                for item in unit.primary.judgments
            }
            for judgment in unit.secondary.judgments:
                peer = primary[(judgment.projection_job_id, judgment.dimension)]
                rows.append(
                    {
                        "scope": "bounded_same_evidence",
                        "window_id": aliases.window_aliases[unit.window_id],
                        "context_id": aliases.context_aliases[unit.context_id],
                        "condition": judgment.condition.value,
                        "metric": f"second_reader_agreement_{judgment.dimension.value}",
                        "value": "1" if judgment.verdict is peer.verdict else "0",
                        "status": "complete",
                        "interpretation": "descriptive_disagreement_preserved",
                    }
                )

    operational_output = next(item for item in resume.output_receipts if item.operational_only)
    retrieval = operational_access.operational_retrieval_receipt
    if retrieval is None:
        raise CaseStudyResumeError("operational result lacks its retrieval receipt")
    operational_metrics = (
        ("output_succeeded", int(operational_output.terminal_outcome.value == "succeeded")),
        ("retrieved_evidence_count", len(retrieval.ordered_evidence_ids)),
        ("omitted_by_cap_count", retrieval.omitted_by_top_k_or_token_cap),
        ("horizon_rejected_match_count", retrieval.horizon_rejected_match_count),
    )
    for metric, value in operational_metrics:
        rows.append(
            {
                "scope": "full_index_operational_noncausal",
                "window_id": aliases.window_aliases[operational_access.window_id],
                "context_id": aliases.context_aliases[plan.operational.context_id],
                "condition": "C2",
                "metric": metric,
                "value": str(value),
                "status": operational_output.terminal_outcome.value,
                "interpretation": "operational_retrieval_not_same_evidence_causal_comparison",
            }
        )

    payload = _canonical_csv(rows)
    for restricted_identifier in (*expected_windows, *expected_contexts):
        if restricted_identifier.encode("utf-8") in payload:
            raise CaseStudyResumeError("restricted narrative identifier leaked into public rows")
    restricted_table = _contained_output(
        restricted_root,
        restricted_table_path,
        label="restricted novel-case table",
    )
    _publish_once(restricted_table, payload, label="restricted novel-case table")
    table_sha256 = hashlib.sha256(payload).hexdigest()
    publication_status: Literal["restricted_pending_canaries", "public_scan_passed"]
    public_table_sha256: str | None = None
    public_table_target: Path | None = None
    canary_manifest_hash: str | None = None
    scan_receipt_hash: str | None = None
    if protected_canary_manifest_path is None:
        publication_status = "restricted_pending_canaries"
        if protected_corpus_path is not None:
            raise CaseStudyResumeError(
                "protected corpus verification requires a canary manifest"
            )
        if public_table_path is not None and public_table_path.exists():
            raise CaseStudyResumeError("pending narrative analysis cannot reuse a public target")
    else:
        if public_table_path is None:
            raise CaseStudyResumeError("a public narrative target is required for canary scanning")
        if protected_corpus_path is None:
            raise CaseStudyResumeError(
                "public narrative output requires the exact protected corpus"
            )
        try:
            canary_manifest, canaries = load_protected_prose_canaries(
                restricted_root,
                protected_canary_manifest_path,
            )
            corpus_file = _contained_existing_restricted_file(
                restricted_root,
                protected_corpus_path,
                label="protected narrative corpus",
            )
            corpus_payload = corpus_file.read_bytes()
            if (
                hashlib.sha256(corpus_payload).hexdigest()
                != index_manifest.source_sha256
                or canary_manifest.corpus_hash != index_manifest.source_sha256
                or any(canary not in corpus_payload for canary in canaries)
            ):
                raise CaseStudyResumeError(
                    "protected-prose canaries are not derived from the indexed corpus"
                )
            public_table = _contained_output(
                public_root,
                public_table_path,
                label="novel-case table",
            )
            public_relative = public_table.relative_to(
                _real_directory(public_root, label="public root")
            ).as_posix()
            scan_public_bytes(
                payload,
                relative_path=public_relative,
                forbidden_canaries=canaries,
            )
        except PublicReleaseError as error:
            raise CaseStudyResumeError(
                "narrative table failed protected-prose/private-path release scan"
            ) from error
        canary_manifest_hash = canary_manifest.content_hash
        scan_receipt_hash = canonical_sha256(
            {
                "canary_manifest_hash": canary_manifest_hash,
                "corpus_source_sha256": index_manifest.source_sha256,
                "payload_sha256": table_sha256,
                "public_relative_path": public_relative,
                "scanner": "story_projection_onto.public_release.scan_public_bytes",
            }
        )
        public_table_target = public_table
        public_table_sha256 = table_sha256
        publication_status = "public_scan_passed"
    receipt = CaseStudyNarrativeAnalysisReceipt(
        analysis_id=analysis_id,
        execution_plan_hash=plan.content_hash,
        restricted_index_manifest_hash=index_manifest.content_hash,
        corpus_source_sha256=index_manifest.source_sha256,
        terminal_resume_manifest_hash=resume.content_hash,
        completed_review_hash=review.content_hash,
        public_alias_manifest_hash=aliases.content_hash,
        restricted_table_file_sha256=table_sha256,
        publication_status=publication_status,
        public_table_file_sha256=public_table_sha256,
        protected_canary_manifest_hash=canary_manifest_hash,
        release_scan_receipt_hash=scan_receipt_hash,
        public_row_inventory_hash=canonical_sha256(
            sorted(rows, key=lambda row: tuple(row[column] for column in NOVEL_CASE_COLUMNS))
        ),
        public_row_count=len(rows),
        compiled_at=compiled_at,
    )
    receipt_path = _contained_output(
        _real_directory(restricted_root, label="restricted root"),
        restricted_receipt_path,
        label="narrative-analysis receipt",
    )
    if receipt_path.exists():
        observed = CaseStudyNarrativeAnalysisReceipt.model_validate_json(
            receipt_path.read_bytes()
        )
        if observed != receipt:
            raise CaseStudyResumeError("append-only narrative-analysis receipt changed")
    else:
        write_restricted_case_record(
            receipt,
            receipt_path,
            restricted_root=restricted_root,
        )
    if public_table_target is not None:
        _publish_once(public_table_target, payload, label="novel-case table")
    return receipt


def _contained_existing_restricted_file(
    root: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    root = _real_directory(root, label="restricted root")
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise CaseStudyResumeError(f"{label} must remain inside the restricted root") from error
    probe = root
    for component in relative.parts:
        probe /= component
        if probe.is_symlink():
            raise CaseStudyResumeError(f"{label} cannot traverse a symbolic link")
    if not lexical.is_file() or lexical.resolve(strict=True) != lexical:
        raise CaseStudyResumeError(f"{label} must be one regular real file")
    return lexical


__all__ = [
    "NOVEL_CASE_COLUMNS",
    "CaseStudyNarrativeAnalysisReceipt",
    "CaseStudyPublicAliasManifest",
    "compile_case_study_narrative_analysis",
    "derive_case_study_public_alias_manifest",
]
