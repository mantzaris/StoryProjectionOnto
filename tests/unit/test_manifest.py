from __future__ import annotations

import json
from pathlib import Path

from story_projection_onto.manifest import build_source_manifest, write_manifest_atomic


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
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "module.pyc").write_bytes(b"private")
    (tmp_path / ".local_data").mkdir()
    (tmp_path / ".local_data" / "novel.txt").write_text("restricted", encoding="utf-8")

    manifest = build_source_manifest(tmp_path, "revision")
    paths = {item.path for item in manifest.files}
    assert paths == {"src/module.py"}


def test_atomic_manifest_writer_round_trips(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("research\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path, "abc123")
    destination = tmp_path / "output" / "source_manifest.json"

    write_manifest_atomic(manifest, destination)

    assert json.loads(destination.read_text(encoding="utf-8")) == manifest.to_dict()
    assert not list(destination.parent.glob("*.tmp"))
