"""Deterministic source-tree manifests for local/remote run association."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0.0"
DEFAULT_SOURCE_ROOTS = (
    "configs",
    "prompts",
    "schemas",
    "scripts",
    "src",
    "tests",
    "ui",
)
DEFAULT_SOURCE_FILES = (".gitignore", "LICENSE", "README.md", "pyproject.toml", "uv.lock")
EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".hypothesis",
    ".local_data",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "blobs",
    "quarantine",
    "restricted",
    "tmp",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One immutable file entry."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Content-only manifest; timestamps and machine names are intentionally absent."""

    schema_version: str
    revision: str
    files: tuple[SourceFile, ...]
    tree_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "files": [asdict(item) for item in self.files],
            "tree_sha256": self.tree_sha256,
        }


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield bounded implementation files in normalized lexical order."""

    candidates: list[Path] = []
    for name in DEFAULT_SOURCE_FILES:
        candidate = root / name
        if candidate.is_file():
            candidates.append(candidate)
    for name in DEFAULT_SOURCE_ROOTS:
        directory = root / name
        if not directory.is_dir():
            continue
        for candidate in directory.rglob("*"):
            if not candidate.is_file() or candidate.is_symlink():
                continue
            relative = candidate.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            candidates.append(candidate)
    yield from sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())


def build_source_manifest(root: Path, revision: str) -> SourceManifest:
    """Hash the exact bounded source tree without reading Git metadata."""

    resolved_root = root.resolve(strict=True)
    files = tuple(
        SourceFile(
            path=path.relative_to(resolved_root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in iter_source_files(resolved_root)
    )
    if not files:
        raise ValueError("source manifest cannot be empty")
    tree_payload = {
        "schema_version": SCHEMA_VERSION,
        "revision": revision,
        "files": [asdict(item) for item in files],
    }
    return SourceManifest(
        schema_version=SCHEMA_VERSION,
        revision=revision,
        files=files,
        tree_sha256=hashlib.sha256(_canonical_json(tree_payload)).hexdigest(),
    )


def write_manifest_atomic(manifest: SourceManifest, destination: Path) -> None:
    """Write a verified manifest without exposing partial output."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_args(arguments)
    manifest = build_source_manifest(options.root, options.revision)
    if options.output is None:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        write_manifest_atomic(manifest, options.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
