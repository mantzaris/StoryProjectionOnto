from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from story_projection_onto.manifest import (
    build_source_association,
    build_source_manifest,
    load_source_manifest,
    write_manifest_atomic,
)


def test_manifest_is_order_independent_and_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "b.py").write_text("b = 2\n", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")

    first = build_source_manifest(tmp_path, "revision")
    second = build_source_manifest(tmp_path, "revision")
    assert first == second
    assert [item.path for item in first.files] == ["src/a.py", "src/b.py"]

    (tmp_path / "src" / "a.py").write_text("a = 3\n", encoding="utf-8")
    changed = build_source_manifest(tmp_path, "revision")
    assert changed.tree_sha256 != first.tree_sha256


def test_manifest_excludes_private_and_cache_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "src" / "legacy.pyc").write_bytes(b"direct bytecode")
    (tmp_path / "src" / "optimized.pyo").write_bytes(b"direct optimized bytecode")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "module.pyc").write_bytes(b"private")
    (tmp_path / ".local_data").mkdir()
    (tmp_path / ".local_data" / "novel.txt").write_text("restricted", encoding="utf-8")

    manifest = build_source_manifest(tmp_path, "revision")
    paths = {item.path for item in manifest.files}
    assert paths == {"src/module.py"}


def test_manifest_includes_authoritative_plans_and_execution_documentation(
    tmp_path: Path,
) -> None:
    (tmp_path / "plan_notes").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "plan_notes" / "method.md").write_text("method\n", encoding="utf-8")
    (tmp_path / "docs" / "run.md").write_text("run\n", encoding="utf-8")

    manifest = build_source_manifest(tmp_path, "revision")

    assert {item.path for item in manifest.files} == {
        "docs/run.md",
        "plan_notes/method.md",
    }


def test_atomic_manifest_writer_round_trips(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("research\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path, "abc123")
    destination = tmp_path / "output" / "source_manifest.json"

    write_manifest_atomic(manifest, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == manifest.to_dict()
    assert not list(destination.parent.glob("*.tmp"))


def test_source_association_requires_independent_byte_identical_manifests(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("research\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path, "abc123+freeze")
    local = tmp_path / "source.local.json"
    remote = tmp_path / "source.remote.json"
    write_manifest_atomic(manifest, local)
    write_manifest_atomic(manifest, remote)

    association = build_source_association(
        local_manifest_path=local,
        remote_manifest_path=remote,
        branch="implementation/query-dependent-temporal-ontology",
        git_commit="a" * 40,
        revision_label="abc123+freeze",
        recorded_at=datetime(2026, 9, 4, tzinfo=UTC),
    )

    assert association["local_tree_sha256"] == manifest.tree_sha256
    assert association["remote_tree_sha256"] == manifest.tree_sha256
    assert association["local_manifest_file_sha256"] == association["remote_manifest_file_sha256"]
    assert association["recorded_at"] == "2026-09-04T00:00:00Z"

    remote.write_text(remote.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="byte-identical"):
        build_source_association(
            local_manifest_path=local,
            remote_manifest_path=remote,
            branch="implementation/query-dependent-temporal-ontology",
            git_commit="a" * 40,
            revision_label="abc123+freeze",
            recorded_at=datetime(2026, 9, 4, tzinfo=UTC),
        )

    remote.unlink()
    remote.symlink_to(local)
    with pytest.raises(ValueError, match="symlinked"):
        build_source_association(
            local_manifest_path=local,
            remote_manifest_path=remote,
            branch="implementation/query-dependent-temporal-ontology",
            git_commit="a" * 40,
            revision_label="abc123+freeze",
            recorded_at=datetime(2026, 9, 4, tzinfo=UTC),
        )


def test_source_manifest_loader_rejects_forged_tree_digest(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("research\n", encoding="utf-8")
    path = tmp_path / "source.json"
    write_manifest_atomic(build_source_manifest(tmp_path, "revision"), path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["tree_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="tree hash"):
        load_source_manifest(path)
