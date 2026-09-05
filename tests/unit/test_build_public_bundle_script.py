from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest

import scripts.build_public_bundle as bundle_cli
from story_projection_onto.case_study_analysis import CaseStudyNarrativeAnalysisReceipt
from story_projection_onto.novel_case import RestrictedNovelIndexManifest
from story_projection_onto.public_release import (
    NarrativeReleaseLineage,
    Phase7ReleaseLineage,
    ProtectedProseCanary,
    ProtectedProseCanaryManifest,
    PublicEntry,
    PublicReleaseError,
)
from story_projection_onto.reporting import ReportStatus, canonical_sha256

HASH = "a" * 64
T0 = datetime(2026, 9, 4, tzinfo=UTC)


def _lineage() -> Phase7ReleaseLineage:
    return Phase7ReleaseLineage(
        build_token="0123456789abcdef",
        source_registry_sha256=HASH,
        compiler_configuration_sha256=HASH,
        current_pointer_file_sha256=HASH,
        current_pointer_manifest_sha256=HASH,
        compilation_manifest_file_sha256=HASH,
        compilation_manifest_sha256=HASH,
        immutable_output_inventory_sha256=HASH,
        result_manifest_file_sha256=HASH,
        report_pdf_file_sha256=HASH,
        public_bundle_input_file_sha256=HASH,
        public_entry_inventory_sha256=HASH,
    )


def _write_manifest(path: Path, payload: dict[str, object]) -> dict[str, object]:
    value = dict(payload)
    value["manifest_sha256"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return value


def _write_phase7_release_tree(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    source_root = tmp_path / "source"
    reports = source_root / "reports"
    reports.mkdir(parents=True)
    token = "0123456789abcdef"
    registry_hash = "1" * 64
    configuration_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "configuration_id": "test-production-configuration",
    }
    configuration_payload["configuration_sha256"] = canonical_sha256(
        configuration_payload
    )
    configuration = source_root / "phase7-configuration.json"
    configuration.write_text(
        json.dumps(configuration_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    target_payloads = {
        f"REPRODUCIBILITY.{token}.md": b"# Reproducibility\n",
        f"RESULTS_REPORT.{token}.md": b"# Results\n",
        f"RESULTS_REPORT.{token}.pdf": b"%PDF-1.7\n%%EOF\n",
        f"result_figure_manifest.{token}.json": b"{}\n",
        f"results_manifest.{token}.json": b"{}\n",
    }
    for relative, payload in target_payloads.items():
        (reports / relative).write_bytes(payload)

    entries = []
    for relative, payload in sorted(target_payloads.items()):
        entries.append(
            {
                "source_relative_path": f"reports/{relative}",
                "bundle_relative_path": f"reports/{relative}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "release_class": "public",
            }
        )
    public_relative = f"public_bundle_inputs.{token}.json"
    public_payload = _write_manifest(
        reports / public_relative,
        {
            "schema_version": "1.0.0",
            "bundle_status": "incomplete",
            "status_reason": "test build",
            "source_registry_sha256": registry_hash,
            "entries": sorted(entries, key=lambda item: item["bundle_relative_path"]),
            "intentional_exclusions": ["restricted inputs"],
        },
    )
    target_payloads[public_relative] = (reports / public_relative).read_bytes()

    aliases = bundle_cli._expected_phase7_aliases(token)
    for alias, target in aliases.items():
        (reports / alias).write_bytes((reports / target).read_bytes())
    immutable_outputs = [
        {
            "relative_path": relative,
            "sha256": hashlib.sha256((reports / relative).read_bytes()).hexdigest(),
        }
        for relative in sorted(target_payloads)
    ]
    compilation_relative = f"phase7_compilation.{token}.json"
    compilation = _write_manifest(
        reports / compilation_relative,
        {
            "schema_version": "1.0.0",
            "kind": "phase7_compilation_manifest",
            "source_registry_sha256": registry_hash,
            "compiler_configuration_sha256": configuration_payload[
                "configuration_sha256"
            ],
            "study_status": "incomplete",
            "build_token": token,
            "immutable_outputs": immutable_outputs,
            "aliases": aliases,
            "manual_numeric_transcription_permitted": False,
        },
    )
    pointer = _write_manifest(
        reports / f"phase7_current.{token}.json",
        {
            "schema_version": "1.0.0",
            "kind": "phase7_current_pointer",
            "source_registry_sha256": registry_hash,
            "compiler_configuration_sha256": configuration_payload[
                "configuration_sha256"
            ],
            "build_token": token,
            "study_status": "incomplete",
            "compilation_manifest_relative_path": compilation_relative,
            "compilation_manifest_file_sha256": hashlib.sha256(
                (reports / compilation_relative).read_bytes()
            ).hexdigest(),
            "result_manifest_relative_path": f"results_manifest.{token}.json",
            "result_manifest_file_sha256": hashlib.sha256(
                target_payloads[f"results_manifest.{token}.json"]
            ).hexdigest(),
            "report_pdf_relative_path": f"RESULTS_REPORT.{token}.pdf",
            "report_pdf_file_sha256": hashlib.sha256(
                target_payloads[f"RESULTS_REPORT.{token}.pdf"]
            ).hexdigest(),
            "public_bundle_input_relative_path": public_relative,
            "public_bundle_input_file_sha256": hashlib.sha256(
                (reports / public_relative).read_bytes()
            ).hexdigest(),
        },
    )
    (reports / "phase7_current.json").write_bytes(
        (reports / f"phase7_current.{token}.json").read_bytes()
    )
    return source_root, {
        "configuration": configuration,
        "current": reports / "phase7_current.json",
        "public": reports / "public_bundle_inputs.json",
        "public_token": reports / public_relative,
        "result": reports / "results_manifest.json",
        "pointer": reports / f"phase7_current.{token}.json",
        "compilation": reports / compilation_relative,
        "public_manifest_hash": Path(str(public_payload["manifest_sha256"])),
        "pointer_manifest_hash": Path(str(pointer["manifest_sha256"])),
        "compilation_manifest_hash": Path(str(compilation["manifest_sha256"])),
    }


def test_phase7_release_lineage_accepts_exact_compiler_chain(tmp_path: Path) -> None:
    source_root, paths = _write_phase7_release_tree(tmp_path)
    lineage = bundle_cli._verify_phase7_release_lineage(
        source_root=source_root,
        current_pointer_path=paths["current"],
        configuration_path=paths["configuration"],
        report_manifest_path=paths["result"],
        public_manifest_path=paths["public"],
    )
    assert lineage.current_pointer_manifest_sha256 == paths[
        "pointer_manifest_hash"
    ].name
    assert lineage.compilation_manifest_sha256 == paths[
        "compilation_manifest_hash"
    ].name
    assert lineage.public_bundle_input_file_sha256 == hashlib.sha256(
        paths["public"].read_bytes()
    ).hexdigest()


def test_phase7_release_lineage_rejects_self_hashed_incomplete_allowlist(
    tmp_path: Path,
) -> None:
    source_root, paths = _write_phase7_release_tree(tmp_path)
    payload = json.loads(paths["public"].read_text(encoding="utf-8"))
    payload["entries"] = payload["entries"][:-1]
    payload.pop("manifest_sha256")
    _write_manifest(paths["public"], payload)
    paths["public_token"].write_bytes(paths["public"].read_bytes())
    with pytest.raises(PublicReleaseError, match=r"hash mismatch|does not bind"):
        bundle_cli._verify_phase7_release_lineage(
            source_root=source_root,
            current_pointer_path=paths["current"],
            configuration_path=paths["configuration"],
            report_manifest_path=paths["result"],
            public_manifest_path=paths["public"],
        )


def test_phase7_release_lineage_rejects_stale_cross_build_alias(tmp_path: Path) -> None:
    source_root, paths = _write_phase7_release_tree(tmp_path)
    paths["result"].write_text('{"stale":"another build"}\n', encoding="utf-8")
    with pytest.raises(PublicReleaseError, match="alias differs"):
        bundle_cli._verify_phase7_release_lineage(
            source_root=source_root,
            current_pointer_path=paths["current"],
            configuration_path=paths["configuration"],
            report_manifest_path=paths["result"],
            public_manifest_path=paths["public"],
        )


def _status_manifest(
    *,
    study_status: ReportStatus = ReportStatus.INCOMPLETE,
    phase_six_status: ReportStatus = ReportStatus.INCOMPLETE,
    with_narrative_table: bool = False,
) -> SimpleNamespace:
    tables = (
        (
            SimpleNamespace(
                table_id="novel_case",
                status=ReportStatus.COMPLETE,
                relative_path="tables/novel_case.csv",
                sha256=HASH,
            ),
        )
        if with_narrative_table
        else ()
    )
    return SimpleNamespace(
        study_status=study_status,
        phases=(SimpleNamespace(phase_id="phase_6", status=phase_six_status),),
        tables=tables,
    )


def test_narrative_detection_uses_typed_status_not_filenames() -> None:
    assert bundle_cli._narrative_release_required(
        _status_manifest(study_status=ReportStatus.COMPLETE),
        bundle_status=ReportStatus.COMPLETE,
        receipt_supplied=False,
    )
    assert bundle_cli._narrative_release_required(
        _status_manifest(with_narrative_table=True),
        bundle_status=ReportStatus.INCOMPLETE,
        receipt_supplied=False,
    )
    assert not bundle_cli._narrative_release_required(
        _status_manifest(),
        bundle_status=ReportStatus.INCOMPLETE,
        receipt_supplied=False,
    )


def test_bundle_and_result_status_must_match() -> None:
    with pytest.raises(PublicReleaseError, match="status differs"):
        bundle_cli._narrative_release_required(
            _status_manifest(),
            bundle_status=ReportStatus.COMPLETE,
            receipt_supplied=False,
        )


def test_complete_release_cannot_bypass_lineage_with_neutral_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = Namespace(
        source_root=tmp_path,
        manifest=Path("neutral-inputs.json"),
        report_manifest=Path("neutral-results.json"),
        phase7_current=Path("neutral-current.json"),
        phase7_configuration=Path("neutral-phase7-configuration.json"),
        phase7_registry=Path("neutral-phase7-registry.json"),
        report_policy=Path("neutral-policy.json"),
        ingestion_manifest=Path("neutral-ingestion.json"),
        ingestion_receipt=Path("neutral-ingestion-receipt.json"),
        bundle_root=tmp_path / "neutral-release",
        visual_raster_manifest=None,
        visual_inspection_receipt=None,
        restricted_root=None,
        protected_prose_canary_manifest=None,
        narrative_analysis_receipt=None,
        restricted_index_manifest=None,
        protected_corpus=None,
    )
    monkeypatch.setattr(bundle_cli, "parse_args", lambda: arguments)
    monkeypatch.setattr(bundle_cli, "_safe_source_file", lambda *_, **__: tmp_path)
    monkeypatch.setattr(
        bundle_cli,
        "_reproduce_phase7_results",
        lambda **_: SimpleNamespace(
            registry_sha256=HASH,
            configuration_sha256=HASH,
            build_token="0123456789abcdef",
            current_pointer_sha256=HASH,
        ),
    )
    monkeypatch.setattr(bundle_cli, "_verify_phase7_release_lineage", lambda **_: _lineage())
    monkeypatch.setattr(bundle_cli, "verify_ingestion_from_files", lambda *_, **__: None)
    monkeypatch.setattr(bundle_cli, "verify_report_build", lambda *_, **__: None)
    monkeypatch.setattr(
        bundle_cli,
        "load_result_manifest",
        lambda _: _status_manifest(study_status=ReportStatus.COMPLETE),
    )
    monkeypatch.setattr(bundle_cli, "load_public_entries", lambda _: ())
    monkeypatch.setattr(bundle_cli, "_bundle_status", lambda _: ReportStatus.COMPLETE)
    with pytest.raises(PublicReleaseError, match="narrative-inclusive or final"):
        bundle_cli.main()


def test_extracted_pdf_text_is_scanned_even_when_encoded_bytes_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = b"PROTECTED EXTRACTED PROSE"
    pdf = tmp_path / "neutral-paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nencoded-stream-without-plain-text\n%%EOF\n")
    entry = PublicEntry(
        source_relative_path=pdf.name,
        bundle_relative_path="paper.pdf",
        sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        release_class="public",
    )
    monkeypatch.setattr(bundle_cli, "_extract_pdf_text", lambda _: canary)
    with pytest.raises(PublicReleaseError, match="protected-text canary"):
        bundle_cli._scan_allowlisted_pdf_text(
            tmp_path,
            (entry,),
            forbidden_canaries=(canary,),
        )


def test_pdf_extraction_uses_bounded_pdftotext_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n%%EOF\n")
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        observed["command"] = command
        observed.update(kwargs)
        output = cast(BinaryIO, kwargs["stdout"])
        output.write(b"decoded text\n")
        return subprocess.CompletedProcess(command, 0, None, b"")

    monkeypatch.setattr(bundle_cli.shutil, "which", lambda _: "/usr/bin/pdftotext")
    monkeypatch.setattr(bundle_cli.subprocess, "run", fake_run)
    assert bundle_cli._extract_pdf_text(pdf) == b"decoded text\n"
    assert observed["command"] == ["/usr/bin/pdftotext", str(pdf), "-"]
    assert observed["stderr"] is subprocess.PIPE
    assert observed["timeout"] == 120


def _narrative_lineage(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    ProtectedProseCanaryManifest,
    tuple[bytes, ...],
    SimpleNamespace,
]:
    source = tmp_path / "source"
    report_tables = source / "reports" / "tables"
    report_tables.mkdir(parents=True)
    public_table = report_tables / "novel_case.csv"
    public_table.write_text("scope,value\nbounded,1\n", encoding="utf-8")
    table_hash = hashlib.sha256(public_table.read_bytes()).hexdigest()

    restricted = tmp_path / "restricted"
    restricted.mkdir()
    canary = b"EXACT PROTECTED CORPUS CANARY"
    corpus = restricted / "novel.txt"
    corpus.write_bytes(b"opening\n" + canary + b"\nending\n")
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
    (restricted / "canaries.json").write_text(
        canary_manifest.to_canonical_json() + "\n", encoding="utf-8"
    )
    scan_hash = canonical_sha256(
        {
            "canary_manifest_hash": canary_manifest.content_hash,
            "corpus_source_sha256": corpus_hash,
            "payload_sha256": table_hash,
            "public_relative_path": "tables/novel_case.csv",
            "scanner": "story_projection_onto.public_release.scan_public_bytes",
        }
    )
    receipt = CaseStudyNarrativeAnalysisReceipt(
        analysis_id="case-analysis",
        execution_plan_hash=HASH,
        restricted_index_manifest_hash=index_manifest.content_hash,
        corpus_source_sha256=corpus_hash,
        terminal_resume_manifest_hash=HASH,
        completed_review_hash=HASH,
        public_alias_manifest_hash=HASH,
        restricted_table_file_sha256=table_hash,
        publication_status="public_scan_passed",
        public_table_file_sha256=table_hash,
        protected_canary_manifest_hash=canary_manifest.content_hash,
        release_scan_receipt_hash=scan_hash,
        public_row_inventory_hash=HASH,
        public_row_count=1,
        compiled_at=T0,
    )
    receipt_path = restricted / "narrative-receipt.json"
    receipt_path.write_text(receipt.to_canonical_json() + "\n", encoding="utf-8")
    result_manifest = SimpleNamespace(
        tables=(
            SimpleNamespace(
                table_id="novel_case",
                status=ReportStatus.COMPLETE,
                relative_path="tables/novel_case.csv",
                sha256=table_hash,
            ),
        )
    )
    return (
        source,
        restricted,
        receipt_path,
        index_path,
        canary_manifest,
        (canary,),
        result_manifest,
    )


def test_narrative_lineage_binds_corpus_index_canaries_and_report_table(
    tmp_path: Path,
) -> None:
    source, restricted, receipt, index, manifest, canaries, results = _narrative_lineage(tmp_path)
    loaded = bundle_cli._verify_narrative_release_lineage(
        source_root=source,
        result_manifest=results,
        restricted_root=restricted,
        narrative_analysis_receipt_path=receipt,
        restricted_index_manifest_path=index,
        protected_corpus_path=restricted / "novel.txt",
        protected_canary_manifest_path=restricted / "canaries.json",
        canary_manifest=manifest,
        canaries=canaries,
    )
    assert isinstance(loaded, NarrativeReleaseLineage)
    assert loaded.publication_status == "public_scan_passed"
    assert loaded.analysis_receipt_file_sha256 == hashlib.sha256(
        receipt.read_bytes()
    ).hexdigest()
    assert loaded.index_manifest_sha256 == RestrictedNovelIndexManifest.model_validate_json(
        index.read_bytes()
    ).content_hash
    assert loaded.protected_canary_manifest_sha256 == manifest.content_hash
    assert loaded.corpus_sha256 == manifest.corpus_hash
    assert "restricted" not in json.dumps(loaded.manifest_payload())


def test_narrative_lineage_rejects_corpus_substitution(tmp_path: Path) -> None:
    source, restricted, receipt, index, manifest, canaries, results = _narrative_lineage(tmp_path)
    (restricted / "novel.txt").write_bytes(b"a different corpus with no registered canary")
    with pytest.raises(PublicReleaseError, match="lineage differs"):
        bundle_cli._verify_narrative_release_lineage(
            source_root=source,
            result_manifest=results,
            restricted_root=restricted,
            narrative_analysis_receipt_path=receipt,
            restricted_index_manifest_path=index,
            protected_corpus_path=restricted / "novel.txt",
            protected_canary_manifest_path=restricted / "canaries.json",
            canary_manifest=manifest,
            canaries=canaries,
        )


def test_narrative_lineage_rejects_stale_table_scan_receipt(tmp_path: Path) -> None:
    source, restricted, receipt_path, index, manifest, canaries, results = _narrative_lineage(
        tmp_path
    )
    receipt = CaseStudyNarrativeAnalysisReceipt.model_validate_json(receipt_path.read_bytes())
    stale_payload = receipt.model_dump(mode="json", exclude={"content_hash"})
    stale_payload["release_scan_receipt_hash"] = "f" * 64
    stale = CaseStudyNarrativeAnalysisReceipt.model_validate(stale_payload)
    receipt_path.write_text(stale.to_canonical_json() + "\n", encoding="utf-8")
    with pytest.raises(PublicReleaseError, match="release-scan receipt is stale"):
        bundle_cli._verify_narrative_release_lineage(
            source_root=source,
            result_manifest=results,
            restricted_root=restricted,
            narrative_analysis_receipt_path=receipt_path,
            restricted_index_manifest_path=index,
            protected_corpus_path=restricted / "novel.txt",
            protected_canary_manifest_path=restricted / "canaries.json",
            canary_manifest=manifest,
            canaries=canaries,
        )
