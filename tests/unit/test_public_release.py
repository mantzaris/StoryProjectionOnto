from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import zstandard

import story_projection_onto.public_release as public_release
from story_projection_onto.public_release import (
    PublicEntry,
    PublicReleaseError,
    build_public_bundle,
    scan_public_entries,
)
from story_projection_onto.reporting import canonical_manifest_payload


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
    result = build_public_bundle(tmp_path, manifest, bundle)
    assert result["entries"][0]["bundle_relative_path"] == "result.csv"
    assert (bundle / "result.csv").is_file()
    assert not (bundle / "private.txt").exists()
    assert bundle.with_suffix(".zip").is_file()
    second_bundle = tmp_path / "release-again"
    build_public_bundle(tmp_path, manifest, second_bundle)
    assert bundle.with_suffix(".zip").read_bytes() == second_bundle.with_suffix(".zip").read_bytes()


def test_public_scan_rejects_private_paths_credentials_and_canary(tmp_path: Path) -> None:
    canary = b"fictional copyrighted canary"
    for index, payload in enumerate(
        (
            b'{"path":"/home/researcher/.ssh/id_ed25519"}\n',
            b'{"path":"/workspace/StoryProjectionOnto/.cache/model"}\n',
            b"-----BEGIN OPENSSH PRIVATE KEY-----\n",
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


def test_oversized_staging_is_rejected_before_copy(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "aggregate.csv"
    path.write_text("metric,value\nscore,0.5\n")
    monkeypatch.setattr(public_release, "PUBLIC_BUNDLE_LIMIT_BYTES", 1)
    with pytest.raises(PublicReleaseError, match="2 GB"):
        scan_public_entries(tmp_path, (_entry(path, path.name),))
