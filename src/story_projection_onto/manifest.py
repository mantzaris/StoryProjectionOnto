"""Deterministic source-tree manifests for local/remote run association."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


def load_source_manifest(path: Path) -> dict[str, object]:
    """Load and independently verify one source manifest.

    Association generation must not trust a copied manifest's declared tree
    digest.  Recomputing it here catches partial transfers and hand-edited file
    inventories before either tree can authorize a scientific run.
    """

    if path.is_symlink():
        raise ValueError("source manifest must be one regular, non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("source manifest must be one regular, non-symlink file")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read source manifest") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "revision",
        "files",
        "tree_sha256",
    }:
        raise ValueError("source manifest has an unexpected shape")
    revision = value.get("revision")
    files = value.get("files")
    tree_sha256 = value.get("tree_sha256")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(revision, str)
        or not revision
        or not isinstance(files, list)
        or not isinstance(tree_sha256, str)
    ):
        raise ValueError("source manifest header is invalid")
    normalized: list[dict[str, object]] = []
    paths: list[str] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("source manifest file entry is invalid")
        relative = entry.get("path")
        size_bytes = entry.get("size_bytes")
        sha256 = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("source manifest file entry is unsafe")
        paths.append(relative)
        normalized.append(cast(dict[str, object], entry))
    if not normalized or paths != sorted(set(paths)):
        raise ValueError("source manifest file inventory is empty, duplicate, or unsorted")
    expected_tree = _sha256_object(
        {
            "schema_version": SCHEMA_VERSION,
            "revision": revision,
            "files": normalized,
        }
    )
    if tree_sha256 != expected_tree:
        raise ValueError("source manifest tree hash does not match its inventory")
    return cast(dict[str, object], value)


def build_source_association(
    *,
    local_manifest_path: Path,
    remote_manifest_path: Path,
    branch: str,
    git_commit: str,
    revision_label: str,
    recorded_at: datetime,
) -> dict[str, object]:
    """Bind independently produced, byte-identical local and remote manifests."""

    if branch != "implementation/query-dependent-temporal-ontology":
        raise ValueError("source association requires the registered implementation branch")
    if (
        len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("source association git commit must be lowercase full SHA-1")
    if not revision_label or revision_label.strip() != revision_label:
        raise ValueError("source association revision label must be nonempty and stripped")
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("source association timestamp must be timezone-aware")
    if local_manifest_path.is_symlink() or remote_manifest_path.is_symlink():
        raise ValueError("source association refuses symlinked manifests")
    local_path = local_manifest_path.resolve(strict=True)
    remote_path = remote_manifest_path.resolve(strict=True)
    if local_path.name != local_manifest_path.name or remote_path.name != remote_manifest_path.name:
        raise ValueError("source manifest arguments must use safe basename paths")
    if local_path == remote_path:
        raise ValueError("local and remote manifests must be independently materialized files")
    local = load_source_manifest(local_path)
    remote = load_source_manifest(remote_path)
    if local != remote or local_path.read_bytes() != remote_path.read_bytes():
        raise ValueError("local and remote source manifests are not byte-identical")
    if local.get("revision") != revision_label:
        raise ValueError("source manifest revision differs from the association label")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "local_remote_source_tree_association",
        "branch": branch,
        "git_commit": git_commit,
        "revision_label": revision_label,
        "recorded_at": recorded_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "local_manifest": local_path.name,
        "local_manifest_file_sha256": _sha256_file(local_path),
        "remote_manifest": remote_path.name,
        "remote_manifest_file_sha256": _sha256_file(remote_path),
        "local_tree_sha256": local["tree_sha256"],
        "remote_tree_sha256": remote["tree_sha256"],
        "byte_identity_checks": [
            "Local and remote manifest JSON bytes are identical.",
            "Both declared source-tree hashes were independently recomputed "
            "from their inventories.",
        ],
    }
    return {**payload, "manifest_sha256": _sha256_object(payload)}


def write_json_atomic(value: object, destination: Path) -> None:
    """Atomically write one canonical, human-readable JSON artifact."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


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

    write_json_atomic(manifest.to_dict(), destination)


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
