from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import scripts.compile_case_study_analysis as analysis_cli
import story_projection_onto.case_study_analysis as analysis_module
from story_projection_onto.case_study_analysis import (
    CaseStudyNarrativeAnalysisReceipt,
    CaseStudyPublicAliasManifest,
    _deterministic_aliases,
)
from story_projection_onto.contracts import RunOutcome
from story_projection_onto.novel_case import RestrictedNovelIndexManifest
from story_projection_onto.public_release import (
    ProtectedProseCanary,
    ProtectedProseCanaryManifest,
)

HASH = "a" * 64
T0 = datetime(2026, 9, 4, tzinfo=UTC)


def _aliases() -> CaseStudyPublicAliasManifest:
    return CaseStudyPublicAliasManifest(
        manifest_id="case-public-aliases",
        execution_plan_hash=HASH,
        source_identifier_inventory_hash=HASH,
        window_aliases={
            f"restricted-window-{index}": f"novel-window-{index:02d}"
            for index in range(1, 6)
        },
        context_aliases={
            f"restricted-context-{index}": f"novel-context-{index:02d}"
            for index in range(1, 10)
        },
        derived_at=T0,
    )


def test_public_alias_manifest_is_opaque_exact_and_restricted() -> None:
    aliases = _aliases()
    assert len(aliases.window_aliases) == 5
    assert len(aliases.context_aliases) == 9
    unsafe = aliases.model_dump(mode="json", exclude={"content_hash"})
    unsafe["window_aliases"]["restricted-window-1"] = "restricted-window-1"
    with pytest.raises(ValidationError, match="must not reproduce"):
        CaseStudyPublicAliasManifest.model_validate(unsafe)


def test_public_alias_assignment_is_deterministic_over_private_ids() -> None:
    assert _deterministic_aliases(
        {"window-z", "window-a"}, prefix="novel-window"
    ) == {
        "window-a": "novel-window-01",
        "window-z": "novel-window-02",
    }


def test_public_receipt_requires_real_canary_scan_lineage() -> None:
    common = {
        "analysis_id": "case-analysis",
        "execution_plan_hash": HASH,
        "restricted_index_manifest_hash": HASH,
        "corpus_source_sha256": HASH,
        "terminal_resume_manifest_hash": HASH,
        "completed_review_hash": HASH,
        "public_alias_manifest_hash": HASH,
        "restricted_table_file_sha256": HASH,
        "public_row_inventory_hash": HASH,
        "public_row_count": 1,
        "compiled_at": T0,
    }
    pending = CaseStudyNarrativeAnalysisReceipt(
        **common,
        publication_status="restricted_pending_canaries",
    )
    assert pending.public_table_file_sha256 is None
    with pytest.raises(ValidationError, match="protected-prose scan lineage"):
        CaseStudyNarrativeAnalysisReceipt(
            **common,
            publication_status="public_scan_passed",
            public_table_file_sha256=HASH,
        )


def _cli_arguments() -> list[str]:
    return [
        "--restricted-root",
        "/restricted",
        "--public-root",
        "/public",
        "--plan",
        "/restricted/plan.json",
        "--restricted-index-manifest",
        "/restricted/index-manifest.json",
        "--terminal-resume",
        "/restricted/resume.json",
        "--review-template",
        "/restricted/template.json",
        "--completed-review",
        "/restricted/review.json",
        "--public-alias-manifest",
        "/restricted/aliases.json",
        "--restricted-table",
        "/restricted/novel_case.csv",
        "--restricted-receipt",
        "/restricted/receipt.json",
        "--analysis-id",
        "case-analysis",
        "--compiled-at",
        "2026-09-04T00:00:00Z",
    ]


def test_analysis_cli_keeps_publication_pending_without_canaries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt = CaseStudyNarrativeAnalysisReceipt(
        analysis_id="case-analysis",
        execution_plan_hash=HASH,
        restricted_index_manifest_hash=HASH,
        corpus_source_sha256=HASH,
        terminal_resume_manifest_hash=HASH,
        completed_review_hash=HASH,
        public_alias_manifest_hash=HASH,
        restricted_table_file_sha256=HASH,
        publication_status="restricted_pending_canaries",
        public_row_inventory_hash=HASH,
        public_row_count=1,
        compiled_at=T0,
    )
    captured: dict[str, object] = {}

    def fake_compile(**values: object) -> CaseStudyNarrativeAnalysisReceipt:
        captured.update(values)
        return receipt

    monkeypatch.setattr(analysis_cli, "compile_case_study_narrative_analysis", fake_compile)
    assert analysis_cli.main(_cli_arguments()) == 0
    assert captured["protected_canary_manifest_path"] is None
    assert captured["protected_corpus_path"] is None
    assert captured["public_table_path"] is None
    assert '"state": "restricted_pending_canaries"' in capsys.readouterr().out


def test_narrative_compiler_derives_aliases_and_scans_the_exact_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restricted = tmp_path / "restricted"
    public = tmp_path / "public"
    restricted.mkdir()
    public.mkdir()
    corpus = restricted / "lawful-novel.txt"
    canary = b"EXACT PRIVATE PROSE CANARY"
    corpus.write_bytes(b"prefix\n" + canary + b"\nsuffix\n")
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    index_manifest = RestrictedNovelIndexManifest(
        manifest_id="novel-index-manifest",
        corpus_id="first-novel",
        edition_label="lawful private edition",
        source_sha256=corpus_hash,
        normalized_source_sha256="b" * 64,
        index_sha256="c" * 64,
        index_size_bytes=1,
        index_config_hash="d" * 64,
        chapter_count=1,
        passage_count=1,
        normalized_character_count=1,
        built_at=T0,
    )
    index_path = restricted / "index-manifest.json"
    index_path.write_text(index_manifest.to_canonical_json() + "\n", encoding="utf-8")
    canary_record = ProtectedProseCanary(
        canary_id="corpus-canary-1",
        payload_base64=base64.b64encode(canary).decode("ascii"),
        payload_sha256=hashlib.sha256(canary).hexdigest(),
    )
    canary_manifest = ProtectedProseCanaryManifest(
        manifest_id="protected-canaries-v1",
        corpus_hash=corpus_hash,
        canaries=(canary_record,),
    )
    canary_path = restricted / "canaries.json"
    canary_path.write_text(canary_manifest.to_canonical_json() + "\n", encoding="utf-8")

    plan = SimpleNamespace(
        content_hash=HASH,
        restricted_index_manifest_hash=index_manifest.content_hash,
        restricted_index_manifest_file_sha256=hashlib.sha256(
            index_path.read_bytes()
        ).hexdigest(),
        windows=tuple(
            SimpleNamespace(
                window_id=f"private-window-{index}",
                context_ids=(
                    f"private-context-{index}-a",
                    f"private-context-{index}-b",
                ),
            )
            for index in range(1, 5)
        ),
        operational=SimpleNamespace(context_id="private-operational-context"),
        compiled_at=T0,
    )
    retrieval = SimpleNamespace(
        ordered_evidence_ids=("opaque-evidence-1",),
        omitted_by_top_k_or_token_cap=0,
        horizon_rejected_match_count=0,
    )
    access = SimpleNamespace(
        operational_only=True,
        window_id=analysis_module.OPERATIONAL_WINDOW_IDENTIFIER,
        operational_retrieval_receipt=retrieval,
    )
    output = SimpleNamespace(
        operational_only=True,
        projection_job_id="operational-output",
        terminal_outcome=RunOutcome.SUCCEEDED,
    )
    resume = SimpleNamespace(
        content_hash="e" * 64,
        query_access_receipts=(access,),
        output_receipts=(output,),
    )
    review = SimpleNamespace(
        content_hash="f" * 64,
        completed_at=T0,
        units=(),
    )
    template_path = restricted / "template.json"
    template_path.write_text("{}\n", encoding="utf-8")
    review_path = restricted / "review.json"
    review_path.write_text("{}\n", encoding="utf-8")
    plan_path = restricted / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    resume_path = restricted / "resume.json"
    resume_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        analysis_module,
        "load_case_study_execution_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        analysis_module,
        "load_case_study_resume_manifest",
        lambda *_args, **_kwargs: resume,
    )
    monkeypatch.setattr(
        analysis_module,
        "load_completed_case_study_review",
        lambda *_args, **_kwargs: review,
    )
    monkeypatch.setattr(
        analysis_module,
        "validate_completed_case_study_review",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        analysis_module,
        "CaseStudyReviewInputTemplate",
        SimpleNamespace(model_validate_json=lambda _payload: SimpleNamespace()),
    )

    alias_path = restricted / "public-aliases.json"
    receipt_path = restricted / "analysis-receipt.json"
    public_table = public / "tables" / "novel_case.csv"
    receipt = analysis_module.compile_case_study_narrative_analysis(
        restricted_root=restricted,
        public_root=public,
        plan_path=plan_path,
        restricted_index_manifest_path=index_path,
        terminal_resume_path=resume_path,
        review_template_path=template_path,
        completed_review_path=review_path,
        public_alias_manifest_path=alias_path,
        restricted_table_path=restricted / "novel_case.csv",
        protected_canary_manifest_path=canary_path,
        protected_corpus_path=corpus,
        public_table_path=public_table,
        restricted_receipt_path=receipt_path,
        analysis_id="case-analysis",
        compiled_at=T0.replace(hour=1),
    )

    aliases = CaseStudyPublicAliasManifest.model_validate_json(alias_path.read_bytes())
    assert aliases.condition_outputs_used is False
    assert aliases.derivation_method == "sorted_execution_plan_identifiers_v1"
    assert receipt.publication_status == "public_scan_passed"
    assert receipt.protected_canary_manifest_hash == canary_manifest.content_hash
    assert receipt_path.is_file()
    assert public_table.is_file()
    assert canary not in public_table.read_bytes()
