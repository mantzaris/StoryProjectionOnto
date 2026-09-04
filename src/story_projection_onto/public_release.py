"""Allowlist-only public artifact bundling and restricted-content scanning."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from story_projection_onto.reporting import canonical_manifest_payload, canonical_sha256, write_json

try:
    import zstandard
except ImportError:  # pragma: no cover - base project dependency
    zstandard = None

PUBLIC_BUNDLE_LIMIT_BYTES = 2_000_000_000
MAX_SCANNED_DECOMPRESSED_BYTES = 128 * 1024 * 1024
ALLOWED_SUFFIXES = frozenset(
    {
        ".css",
        ".csv",
        ".html",
        ".js",
        ".json",
        ".md",
        ".pdf",
        ".png",
        ".py",
        ".svg",
        ".toml",
        ".txt",
        ".zst",
    }
)
PROHIBITED_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".epub",
        ".gguf",
        ".index",
        ".mobi",
        ".pt",
        ".safetensors",
        ".sqlite",
        ".sqlite3",
    }
)
PROHIBITED_COMPONENTS = frozenset(
    {
        ".cache",
        ".git",
        ".local_data",
        ".ssh",
        "blobs",
        "model_cache",
        "quarantine",
        "restricted",
    }
)
_PRIVATE_HOME_PATTERN = rb"(?:/" + rb"home/[^/\x00\s]+|/" + rb"root)/(?:\.ssh|[^\x00\r\n]{0,200})"
_WORKSPACE_PATTERN = rb"/work" + rb"space/[A-Za-z0-9_.-]+"

SENSITIVE_TEXT_PATTERNS = (
    re.compile(rb"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    re.compile(_PRIVATE_HOME_PATTERN, re.IGNORECASE),
    re.compile(_WORKSPACE_PATTERN, re.IGNORECASE),
    re.compile(rb"[A-Za-z]:\\Users\\[^\x00\r\n]+", re.IGNORECASE),
    re.compile(rb"ssh\s+(?:-[^\r\n ]+\s+)*[^\s@]+@[^\s]+", re.IGNORECASE),
    re.compile(rb'"(?:restricted|source|corpus|model_cache)_path"\s*:\s*"[^\"]+"', re.IGNORECASE),
)
RECONSTRUCTIVE_CASE_KEYS = frozenset(
    {
        "byte_end",
        "byte_start",
        "char_end",
        "char_start",
        "detailed_offsets",
        "normalized_text",
        "raw_prose",
        "raw_text",
        "restricted_text_handle",
        "snippet",
        "source_path",
        "text",
    }
)


class PublicReleaseError(RuntimeError):
    """A public artifact violates the release boundary."""


@dataclass(frozen=True, slots=True)
class PublicEntry:
    source_relative_path: str
    bundle_relative_path: str
    sha256: str
    release_class: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise PublicReleaseError(f"backslash is prohibited in public path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise PublicReleaseError(f"unsafe public path: {value}")
    lowered = {item.lower() for item in path.parts}
    blocked = sorted(lowered & PROHIBITED_COMPONENTS)
    if blocked:
        raise PublicReleaseError(f"prohibited public path component: {blocked[0]}")
    suffix = path.suffix.lower()
    extensionless_notice = not suffix and path.name.upper() in {"LICENSE", "NOTICE"}
    if suffix in PROHIBITED_SUFFIXES or (
        suffix not in ALLOWED_SUFFIXES and not extensionless_notice
    ):
        raise PublicReleaseError(f"prohibited or unknown public artifact suffix: {suffix}")
    return path


def _safe_source(root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path)
    candidate = root.joinpath(*relative.parts)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as exc:
        raise PublicReleaseError(f"missing public artifact: {relative_path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PublicReleaseError(f"symlink prohibited in public bundle: {relative_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise PublicReleaseError(f"public artifact is not a regular file: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()):
        raise PublicReleaseError(f"public artifact escapes source root: {relative_path}")
    return resolved


def scan_public_bytes(
    payload: bytes,
    *,
    relative_path: str,
    forbidden_canaries: Sequence[bytes] = (),
) -> None:
    """Reject credentials, private paths, and exact protected-text canaries."""

    _validate_relative_path(relative_path)
    for pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.search(payload):
            raise PublicReleaseError(f"sensitive content in public artifact: {relative_path}")
    for canary in forbidden_canaries:
        if len(canary) < 8:
            raise PublicReleaseError("release canaries must be at least eight bytes")
        if canary in payload:
            raise PublicReleaseError(f"protected-text canary in public artifact: {relative_path}")


def _iter_json_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _iter_json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_keys(child)


def _walk_json(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _scan_structured_release(value: Any, relative_path: str) -> None:
    for key, child in _walk_json(value):
        lowered_key = key.lower()
        lowered_value = child.lower() if isinstance(child, str) else None
        if lowered_key == "release_class" and lowered_value == "restricted":
            raise PublicReleaseError(
                f"restricted release record in public artifact: {relative_path}"
            )
        if lowered_key == "rights_class" and lowered_value in {
            "restricted",
            "restricted_copyrighted",
        }:
            raise PublicReleaseError(
                f"restricted rights record in public artifact: {relative_path}"
            )
        if lowered_key in {"private_key", "ssh_private_key", "credential", "access_token"}:
            raise PublicReleaseError(f"credential field in public artifact: {relative_path}")


def _parse_structured_payload(payload: bytes, relative_path: str) -> Any | None:
    logical_path = relative_path[:-4] if relative_path.lower().endswith(".zst") else relative_path
    suffix = PurePosixPath(logical_path).suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicReleaseError(f"invalid public JSON: {relative_path}") from exc
    if suffix == ".jsonl":
        records = []
        try:
            for line in payload.decode("utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicReleaseError(f"invalid public JSONL: {relative_path}") from exc
        return records
    return None


def _decompress_zstd(path: Path, relative_path: str) -> bytes:
    if zstandard is None:
        raise PublicReleaseError("zstd public artifacts require the pinned zstandard dependency")
    output = bytearray()
    try:
        with (
            path.open("rb") as compressed,
            zstandard.ZstdDecompressor().stream_reader(compressed) as reader,
        ):
            while chunk := reader.read(1024 * 1024):
                output.extend(chunk)
                if len(output) > MAX_SCANNED_DECOMPRESSED_BYTES:
                    raise PublicReleaseError(
                        f"decompressed public artifact exceeds scan cap: {relative_path}"
                    )
    except zstandard.ZstdError as exc:
        raise PublicReleaseError(f"invalid zstd public artifact: {relative_path}") from exc
    return bytes(output)


def _scan_case_artifact(payload: bytes, relative_path: str) -> None:
    """Allow only paraphrases and opaque/chapter-level locators in public case output."""

    lowered_path = relative_path.lower()
    data_bearing = lowered_path.startswith(("artifacts/", "reports/"))
    case_bearing = "novel" in lowered_path or "case" in lowered_path
    if not (data_bearing and case_bearing):
        return
    logical_path = relative_path[:-4] if relative_path.lower().endswith(".zst") else relative_path
    suffix = PurePosixPath(logical_path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        value = _parse_structured_payload(payload, relative_path)
        keys = {item.lower() for item in _iter_json_keys(value)}
    elif suffix == ".csv":
        first_line = payload.splitlines()[0].decode("utf-8") if payload else ""
        keys = {item.strip().lower() for item in first_line.split(",")}
    else:
        return
    violations = sorted(
        key for key in keys if key in RECONSTRUCTIVE_CASE_KEYS or "offset" in key
    )
    if violations:
        raise PublicReleaseError(
            "reconstructive case-study field in public artifact: " + violations[0]
        )


def _scan_file_payload(
    path: Path,
    relative_path: str,
    *,
    forbidden_canaries: Sequence[bytes],
) -> None:
    if path.suffix.lower() == ".zst":
        payload = _decompress_zstd(path, relative_path)
    else:
        if path.stat().st_size > MAX_SCANNED_DECOMPRESSED_BYTES:
            raise PublicReleaseError(
                f"individual public artifact exceeds scan cap: {relative_path}"
            )
        payload = path.read_bytes()
    scan_public_bytes(
        payload,
        relative_path=relative_path,
        forbidden_canaries=forbidden_canaries,
    )
    structured = _parse_structured_payload(payload, relative_path)
    if structured is not None:
        _scan_structured_release(structured, relative_path)
    _scan_case_artifact(payload, relative_path)


def load_public_entries(manifest_path: Path) -> tuple[PublicEntry, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied_hash = payload.pop("manifest_sha256", None)
    if supplied_hash != canonical_sha256(payload):
        raise PublicReleaseError("public bundle input manifest self-hash mismatch")
    if payload.get("schema_version") != "1.0.0":
        raise PublicReleaseError("unsupported public bundle manifest schema")
    entries = tuple(PublicEntry(**item) for item in payload.get("entries", ()))
    if not entries:
        raise PublicReleaseError("public bundle allowlist is empty")
    targets = [item.bundle_relative_path for item in entries]
    if len(targets) != len(set(targets)):
        raise PublicReleaseError("duplicate public bundle target")
    return entries


def scan_public_entries(
    source_root: Path,
    entries: Sequence[PublicEntry],
    *,
    forbidden_canaries: Sequence[bytes] = (),
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    total = 0
    for entry in entries:
        if entry.release_class != "public":
            raise PublicReleaseError(f"non-public release class: {entry.source_relative_path}")
        _validate_relative_path(entry.bundle_relative_path)
        path = _safe_source(source_root, entry.source_relative_path)
        actual_hash = _file_sha256(path)
        if actual_hash != entry.sha256:
            raise PublicReleaseError(f"public source hash mismatch: {entry.source_relative_path}")
        size = path.stat().st_size
        total += size
        if total > PUBLIC_BUNDLE_LIMIT_BYTES:
            raise PublicReleaseError("public bundle exceeds the 2 GB conference limit")
        _scan_file_payload(
            path,
            entry.bundle_relative_path,
            forbidden_canaries=forbidden_canaries,
        )
        records.append(
            {
                "source_relative_path": entry.source_relative_path,
                "bundle_relative_path": entry.bundle_relative_path,
                "sha256": actual_hash,
                "size_bytes": size,
                "release_class": entry.release_class,
            }
        )
    return tuple(records)


def _copy_verified(source: Path, target: Path, expected_hash: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        raise PublicReleaseError(f"stale public bundle temporary: {temporary.name}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if _file_sha256(temporary) != expected_hash:
        raise PublicReleaseError(f"copied public artifact hash mismatch: {target.name}")
    temporary.replace(target)


def build_public_bundle(
    source_root: Path,
    manifest_path: Path,
    bundle_root: Path,
    *,
    forbidden_canaries: Sequence[bytes] = (),
) -> dict[str, Any]:
    """Create a new allowlisted directory and deterministic ZIP archive.

    The destination must not exist. This prevents a rebuild from silently retaining a
    file that has been removed from the allowlist.
    """

    entries = load_public_entries(manifest_path)
    records = scan_public_entries(
        source_root,
        entries,
        forbidden_canaries=forbidden_canaries,
    )
    if bundle_root.exists():
        raise PublicReleaseError("public bundle destination already exists")
    bundle_root.mkdir(parents=True)
    for entry in entries:
        source = _safe_source(source_root, entry.source_relative_path)
        target_path = _validate_relative_path(entry.bundle_relative_path)
        target = bundle_root.joinpath(*target_path.parts)
        _copy_verified(source, target, entry.sha256)
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "verified_public_bundle",
        "source_manifest_sha256": _file_sha256(manifest_path),
        "entries": list(records),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "excluded_classes": [
            "copyrighted_prose",
            "detailed_reconstructive_offsets",
            "fts_indexes",
            "model_weights",
            "raw_restricted_prompts",
            "restricted_paths",
        ],
    }
    bundle_manifest = canonical_manifest_payload(payload)
    write_json(bundle_root / "PUBLIC_BUNDLE_MANIFEST.json", bundle_manifest)
    scan_public_bytes(
        (bundle_root / "PUBLIC_BUNDLE_MANIFEST.json").read_bytes(),
        relative_path="PUBLIC_BUNDLE_MANIFEST.json",
        forbidden_canaries=forbidden_canaries,
    )
    archive_path = bundle_root.with_suffix(".zip")
    if archive_path.exists():
        raise PublicReleaseError("public bundle archive destination already exists")
    with zipfile.ZipFile(
        archive_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(bundle_root).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return bundle_manifest


def verified_bundle_paths(bundle_root: Path) -> Iterable[Path]:
    """Yield only regular non-symlink bundle files for release inspection."""

    for path in sorted(bundle_root.rglob("*")):
        if path.is_symlink():
            raise PublicReleaseError(f"symlink found in completed public bundle: {path}")
        if path.is_file():
            yield path
