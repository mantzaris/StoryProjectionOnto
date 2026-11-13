#!/usr/bin/env python3
"""Remove only verified-regenerable Python caches inside a bounded project root."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def allocated_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    candidates = (path,) if path.is_file() else path.rglob("*")
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        stat = candidate.stat()
        total += stat.st_blocks * 512 if stat.st_blocks else stat.st_size
    return total


def require_descendant(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    resolved.relative_to(root)
    return resolved


def prune(project_root: Path) -> dict[str, object]:
    root = project_root.resolve(strict=True)
    virtual_environment = require_descendant(root, root / ".venv")
    cache_root = require_descendant(root, root / ".cache")
    targets: list[Path] = []
    for name in ("pip", "uv"):
        candidate = require_descendant(root, cache_root / name)
        if candidate.exists():
            targets.append(candidate)
    if virtual_environment.is_dir():
        targets.extend(
            require_descendant(root, candidate)
            for candidate in virtual_environment.rglob("__pycache__")
            if candidate.is_dir() and not candidate.is_symlink()
        )

    # Deepest directories first prevents a selected parent from invalidating a child path.
    unique_targets = sorted(set(targets), key=lambda item: len(item.parts), reverse=True)
    before = sum(allocated_bytes(path) for path in unique_targets)
    removed: list[str] = []
    for target in unique_targets:
        if not target.exists():
            continue
        target.relative_to(root)
        shutil.rmtree(target)
        removed.append(target.relative_to(root).as_posix())
    return {
        "schema_version": "1.0.0",
        "removed_path_count": len(removed),
        "allocated_bytes_reclaimed": before,
        "removed_roots": [name for name in (".cache/pip", ".cache/uv") if name in removed],
        "bytecode_cache_count": sum(name.endswith("__pycache__") for name in removed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    options = parser.parse_args()
    print(json.dumps(prune(options.project_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
