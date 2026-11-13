from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import zstandard

import story_projection_onto.public_release as public_release
from story_projection_onto.public_release import (
    NarrativeReleaseLineage,
    Phase7ReleaseLineage,
    ProtectedProseCanary,
    ProtectedProseCanaryManifest,
    PublicEntry,
    PublicReleaseError,
    VisualReleaseLineage,
    build_public_bundle,
    load_protected_prose_canaries,
    scan_public_entries,
)
from story_projection_onto.reporting import canonical_manifest_payload, canonical_sha256


def _entry(path: Path, relative: str, *, release_class: str = "public") -> PublicEntry:
    return PublicEntry(
        source_relative_path=relative,
        bundle_relative_path=relative,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        release_class=release_class,
    )


def _write_bundle_manifest(path: Path, entries: list[PublicEntry]) -> None:
    payload = canonical_manifest_payload(
        {
            "schema_version": "1.0.0",
            "entries": [asdict(entry) for entry in entries],
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _visual_lineage(*, pdf_sha256: str = "9" * 64) -> VisualReleaseLineage:
    return VisualReleaseLineage(
        raster_manifest_file_sha256="a" * 64,
        raster_manifest_sha256="b" * 64,
        inspection_receipt_file_sha256="c" * 64,
        inspection_receipt_sha256="d" * 64,
        source_pdf_sha256=pdf_sha256,
    )


def _narrative_lineage(
    *, canary_sha256: str, public_table_sha256: str = "8" * 64
) -> NarrativeReleaseLineage:
    return NarrativeReleaseLineage(
        analysis_receipt_file_sha256="1" * 64,
        analysis_receipt_sha256="2" * 64,
        index_manifest_file_sha256="3" * 64,
        index_manifest_sha256="4" * 64,
        protected_canary_manifest_file_sha256="5" * 64,
        protected_canary_manifest_sha256=canary_sha256,
        corpus_sha256="6" * 64,
        release_scan_receipt_sha256="7" * 64,
        public_table_file_sha256=public_table_sha256,
    )


@pytest.mark.parametrize(
    "relative",
    [
        ".local_data/novel.txt",
        "artifacts/restricted/index.json",
        "models/weights.safetensors",
        "case/index.sqlite",
    ],
)
def test_public_path_boundary_rejects_restricted_and_model_artifacts(
    tmp_path: Path,
    relative: str,
) -> None:
    source = tmp_path / "safe.json"
    source.write_text("{}\n")
    entry = PublicEntry(relative, relative, hashlib.sha256(b"{}\n").hexdigest(), "public")
    with pytest.raises(PublicReleaseError):
        scan_public_entries(tmp_path, (entry,))


def test_public_scan_rejects_restricted_records_and_reconstructive_case_fields(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted.json"
    restricted.write_text('{"release_class":"restricted"}\n')
    with pytest.raises(PublicReleaseError, match="restricted release"):
        scan_public_entries(tmp_path, (_entry(restricted, "restricted.json"),))

    rights = tmp_path / "rights.json"
    rights.write_text('{"rights_class":"restricted_copyrighted"}\n')
    with pytest.raises(PublicReleaseError, match="restricted rights"):
        scan_public_entries(tmp_path, (_entry(rights, "rights.json"),))

    case_dir = tmp_path / "reports"
    case_dir.mkdir()
    case = case_dir / "novel_case.json"
    case.write_text('{"chapter_id":"opaque-1","char_start":42}\n')
    with pytest.raises(PublicReleaseError, match="reconstructive"):
        scan_public_entries(tmp_path, (_entry(case, "reports/novel_case.json"),))


def test_public_scan_accepts_opaque_case_locator_and_high_level_paraphrase(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "reports"
    case_dir.mkdir()
    case = case_dir / "novel_case.json"
    case.write_text(
        '{"chapter_id":"chapter-opaque-1","evidence_id":"ev-opaque-1",'
        '"paraphrase":"A leader revises an allegiance."}\n'
    )
    records = scan_public_entries(tmp_path, (_entry(case, "reports/novel_case.json"),))
    assert records[0]["release_class"] == "public"


def test_compressed_canary_is_decompressed_and_rejected(tmp_path: Path) -> None:
    canary = b"COPYRIGHT-CANARY-EXCERPT"
    path = tmp_path / "attempts.jsonl.zst"
    compressor = zstandard.ZstdCompressor()
    path.write_bytes(compressor.compress(b'{"value":"' + canary + b'"}\n'))
    with pytest.raises(PublicReleaseError, match="protected-text canary"):
        scan_public_entries(
            tmp_path,
            (_entry(path, "attempts.jsonl.zst"),),
            forbidden_canaries=(canary,),
        )


def test_bundle_is_allowlist_only_and_deterministic(tmp_path: Path) -> None:
    public = tmp_path / "result.csv"
    public.write_text("metric,value\nstrict_f1,incomplete\n")
    private = tmp_path / "private.txt"
    private.write_text("not allowlisted")
    entry = _entry(public, "result.csv")
    manifest = tmp_path / "bundle_inputs.json"
    _write_bundle_manifest(manifest, [entry])
    bundle = tmp_path / "release"
    lineage = Phase7ReleaseLineage(
        build_token="0123456789abcdef",
        source_registry_sha256="1" * 64,
        compiler_configuration_sha256="2" * 64,
        current_pointer_file_sha256="3" * 64,
        current_pointer_manifest_sha256="4" * 64,
        compilation_manifest_file_sha256="5" * 64,
        compilation_manifest_sha256="6" * 64,
        immutable_output_inventory_sha256="7" * 64,
        result_manifest_file_sha256="8" * 64,
        report_pdf_file_sha256="9" * 64,
        public_bundle_input_file_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        public_entry_inventory_sha256=canonical_sha256([asdict(entry)]),
    )
    result = build_public_bundle(tmp_path, manifest, bundle, phase7_lineage=lineage)
    assert result["entries"][0]["bundle_relative_path"] == "result.csv"
    assert result["phase7_release_lineage"] == lineage.manifest_payload()
    assert (bundle / "result.csv").is_file()
    assert not (bundle / "private.txt").exists()
    assert bundle.with_suffix(".zip").is_file()
    second_bundle = tmp_path / "release-again"
    build_public_bundle(tmp_path, manifest, second_bundle, phase7_lineage=lineage)
    assert bundle.with_suffix(".zip").read_bytes() == second_bundle.with_suffix(".zip").read_bytes()


def test_complete_or_compiler_bundle_requires_exact_phase7_lineage(tmp_path: Path) -> None:
    public = tmp_path / "result.csv"
    public.write_text("metric,value\nstrict_f1,incomplete\n")
    entry = _entry(public, "result.csv")
    manifest = tmp_path / "bundle_inputs.json"
    payload = canonical_manifest_payload(
        {
            "schema_version": "1.0.0",
            "bundle_status": "complete",
            "source_registry_sha256": "1" * 64,
            "entries": [asdict(entry)],
        }
    )
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(PublicReleaseError, match="requires Phase-7 lineage"):
        build_public_bundle(tmp_path, manifest, tmp_path / "release")

    exact = Phase7ReleaseLineage(
        build_token="0123456789abcdef",
        source_registry_sha256="1" * 64,
        compiler_configuration_sha256="2" * 64,
        current_pointer_file_sha256="3" * 64,
        current_pointer_manifest_sha256="4" * 64,
        compilation_manifest_file_sha256="5" * 64,
        compilation_manifest_sha256="6" * 64,
        immutable_output_inventory_sha256="7" * 64,
        result_manifest_file_sha256="8" * 64,
        report_pdf_file_sha256="9" * 64,
        public_bundle_input_file_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        public_entry_inventory_sha256=canonical_sha256([asdict(entry)]),
    )
    with pytest.raises(PublicReleaseError, match="visual-inspection lineage"):
        build_public_bundle(
            tmp_path,
            manifest,
            tmp_path / "release-without-inspection",
            phase7_lineage=exact,
        )

    result = build_public_bundle(
        tmp_path,
        manifest,
        tmp_path / "complete-release",
        phase7_lineage=exact,
        visual_release_lineage=_visual_lineage(),
    )
    assert result["visual_release_lineage"] == _visual_lineage().manifest_payload()

    wrong = Phase7ReleaseLineage(
        build_token="0123456789abcdef",
        source_registry_sha256="1" * 64,
        compiler_configuration_sha256="2" * 64,
        current_pointer_file_sha256="3" * 64,
        current_pointer_manifest_sha256="4" * 64,
        compilation_manifest_file_sha256="5" * 64,
        compilation_manifest_sha256="6" * 64,
        immutable_output_inventory_sha256="7" * 64,
        result_manifest_file_sha256="8" * 64,
        report_pdf_file_sha256="9" * 64,
        public_bundle_input_file_sha256="a" * 64,
        public_entry_inventory_sha256="b" * 64,
    )
    with pytest.raises(PublicReleaseError, match="lineage differs"):
        build_public_bundle(
            tmp_path,
            manifest,
            tmp_path / "wrong-release",
            phase7_lineage=wrong,
        )


def test_reproduction_source_extensions_and_case_named_code_are_public_safe(
    tmp_path: Path,
) -> None:
    sources = {
        "src/story_projection_onto/case_study_analysis.py": "# public source\n",
        "scripts/capture.mjs": "export const value = 1;\n",
        "scripts/replay.sh": "#!/bin/sh\nexit 0\n",
        "ui/CYTOSCAPE_LICENSE": "Public license notice.\n",
        "uv.lock": "version = 1\n",
    }
    entries = []
    for relative, payload in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        entries.append(_entry(path, relative))
    manifest = tmp_path / "bundle_inputs.json"
    _write_bundle_manifest(manifest, entries)
    result = build_public_bundle(tmp_path, manifest, tmp_path / "release")
    assert result["case_study_publication"] is False
    assert {item["bundle_relative_path"] for item in result["entries"]} == set(sources)


def test_public_scan_rejects_private_paths_credentials_and_canary(tmp_path: Path) -> None:
    canary = b"fictional copyrighted canary"
    for index, payload in enumerate(
        (
            b'{"path":"/ho' + b'me/researcher/.ssh/id_ed25519"}\n',
            b'{"path":"/work' + b'space/StoryProjectionOnto/.cache/model"}\n',
            b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----\n",
            b'{"text":"' + canary + b'"}\n',
        )
    ):
        path = tmp_path / f"unsafe-{index}.json"
        path.write_bytes(payload)
        with pytest.raises(PublicReleaseError):
            scan_public_entries(
                tmp_path,
                (_entry(path, path.name),),
                forbidden_canaries=(canary,),
            )


def test_restricted_release_class_is_rejected_before_copy(tmp_path: Path) -> None:
    path = tmp_path / "aggregate.csv"
    path.write_text("metric,value\nscore,0.5\n")
    with pytest.raises(PublicReleaseError, match="non-public"):
        scan_public_entries(tmp_path, (_entry(path, path.name, release_class="restricted"),))


def test_case_bundle_requires_and_applies_restricted_canary_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    case_directory = source_root / "reports"
    case_directory.mkdir(parents=True)
    case = case_directory / "novel_case.json"
    case.write_text('{"paraphrase":"A safe high-level summary."}\n')
    entry = _entry(case, "reports/novel_case.json")
    bundle_manifest = source_root / "bundle.json"
    _write_bundle_manifest(bundle_manifest, [entry])
    with pytest.raises(PublicReleaseError, match="requires a validated"):
        build_public_bundle(source_root, bundle_manifest, tmp_path / "blocked")

    restricted = tmp_path / "restricted"
    restricted.mkdir()
    canary = b"PROTECTED-PROSE-CANARY-001"
    record = ProtectedProseCanary(
        canary_id="canary-001",
        payload_base64=base64.b64encode(canary).decode("ascii"),
        payload_sha256=hashlib.sha256(canary).hexdigest(),
    )
    canary_manifest = ProtectedProseCanaryManifest(
        manifest_id="protected-canaries-v1",
        corpus_hash="a" * 64,
        canaries=(record,),
    )
    canary_path = restricted / "canaries.json"
    canary_path.write_text(canary_manifest.to_canonical_json() + "\n")
    loaded, values = load_protected_prose_canaries(restricted, canary_path)
    with pytest.raises(PublicReleaseError, match="narrative release lineage"):
        build_public_bundle(
            source_root,
            bundle_manifest,
            tmp_path / "blocked-without-narrative-lineage",
            forbidden_canaries=values,
            protected_canary_manifest_hash=loaded.content_hash,
        )
    with pytest.raises(PublicReleaseError, match="differs from protected canary"):
        build_public_bundle(
            source_root,
            bundle_manifest,
            tmp_path / "blocked-with-stale-narrative-lineage",
            forbidden_canaries=values,
            protected_canary_manifest_hash=loaded.content_hash,
            narrative_release_lineage=_narrative_lineage(
                canary_sha256="f" * 64,
                public_table_sha256=entry.sha256,
            ),
        )
    with pytest.raises(PublicReleaseError, match="allowlisted public table"):
        build_public_bundle(
            source_root,
            bundle_manifest,
            tmp_path / "blocked-with-stale-table-lineage",
            forbidden_canaries=values,
            protected_canary_manifest_hash=loaded.content_hash,
            narrative_release_lineage=_narrative_lineage(
                canary_sha256=loaded.content_hash
            ),
        )
    narrative_lineage = _narrative_lineage(
        canary_sha256=loaded.content_hash,
        public_table_sha256=entry.sha256,
    )
    result = build_public_bundle(
        source_root,
        bundle_manifest,
        tmp_path / "release",
        forbidden_canaries=values,
        protected_canary_manifest_hash=loaded.content_hash,
        narrative_release_lineage=narrative_lineage,
    )
    assert result["case_study_publication"] is True
    assert result["protected_canary_manifest_hash"] == loaded.content_hash
    assert result["narrative_release_lineage"] == narrative_lineage.manifest_payload()
    assert str(restricted) not in json.dumps(result)
    assert canary.decode("ascii") not in json.dumps(result)
    case.write_bytes(b'{"paraphrase":"' + canary + b'"}\n')
    leaking_entry = _entry(case, "reports/novel_case.json")
    _write_bundle_manifest(bundle_manifest, [leaking_entry])
    with pytest.raises(PublicReleaseError, match="protected-text canary"):
        build_public_bundle(
            source_root,
            bundle_manifest,
            tmp_path / "leaking-release",
            forbidden_canaries=values,
            protected_canary_manifest_hash=loaded.content_hash,
            narrative_release_lineage=_narrative_lineage(
                canary_sha256=loaded.content_hash,
                public_table_sha256=leaking_entry.sha256,
            ),
        )


def test_public_bundle_rejects_tampered_attestation_bindings(tmp_path: Path) -> None:
    public = tmp_path / "result.csv"
    public.write_text("metric,value\nstrict_f1,incomplete\n")
    entry = _entry(public, "result.csv")
    manifest = tmp_path / "bundle_inputs.json"
    payload = canonical_manifest_payload(
        {
            "schema_version": "1.0.0",
            "bundle_status": "complete",
            "source_registry_sha256": "1" * 64,
            "entries": [asdict(entry)],
        }
    )
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lineage = Phase7ReleaseLineage(
        build_token="0123456789abcdef",
        source_registry_sha256="1" * 64,
        compiler_configuration_sha256="2" * 64,
        current_pointer_file_sha256="3" * 64,
        current_pointer_manifest_sha256="4" * 64,
        compilation_manifest_file_sha256="5" * 64,
        compilation_manifest_sha256="6" * 64,
        immutable_output_inventory_sha256="7" * 64,
        result_manifest_file_sha256="8" * 64,
        report_pdf_file_sha256="9" * 64,
        public_bundle_input_file_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        public_entry_inventory_sha256=canonical_sha256([asdict(entry)]),
    )
    with pytest.raises(PublicReleaseError, match="differs from Phase-7 report PDF"):
        build_public_bundle(
            tmp_path,
            manifest,
            tmp_path / "tampered-visual",
            phase7_lineage=lineage,
            visual_release_lineage=_visual_lineage(pdf_sha256="e" * 64),
        )

    with pytest.raises(ValueError, match="accepted inspection"):
        VisualReleaseLineage(
            raster_manifest_file_sha256="a" * 64,
            raster_manifest_sha256="b" * 64,
            inspection_receipt_file_sha256="c" * 64,
            inspection_receipt_sha256="d" * 64,
            source_pdf_sha256="9" * 64,
            inspection_status="rejected",
        )


def test_public_scan_rejects_intermediate_source_symlink(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    outside = tmp_path / "outside"
    source_root.mkdir()
    outside.mkdir()
    payload = outside / "result.json"
    payload.write_text("{}\n")
    (source_root / "nested").symlink_to(outside, target_is_directory=True)
    entry = PublicEntry(
        source_relative_path="nested/result.json",
        bundle_relative_path="result.json",
        sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
        release_class="public",
    )
    with pytest.raises(PublicReleaseError, match="symlinked"):
        scan_public_entries(source_root, (entry,))


def test_oversized_staging_is_rejected_before_copy(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "aggregate.csv"
    path.write_text("metric,value\nscore,0.5\n")
    monkeypatch.setattr(public_release, "PUBLIC_BUNDLE_LIMIT_BYTES", 1)
    with pytest.raises(PublicReleaseError, match="2 GB"):
        scan_public_entries(tmp_path, (_entry(path, path.name),))
