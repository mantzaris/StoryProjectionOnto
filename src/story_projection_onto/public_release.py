"""Allowlist-only public artifact bundling and restricted-content scanning."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from story_projection_onto.contracts import ImmutableRecord, ReleaseClass, Sha256Digest
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
        ".lock",
        ".md",
        ".mjs",
        ".pdf",
        ".png",
        ".py",
        ".sh",
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


@dataclass(frozen=True, slots=True)
class Phase7ReleaseLineage:
    """Exact production-compiler chain authenticated before public bundling."""

    build_token: str
    source_registry_sha256: str
    compiler_configuration_sha256: str
    current_pointer_file_sha256: str
    current_pointer_manifest_sha256: str
    compilation_manifest_file_sha256: str
    compilation_manifest_sha256: str
    immutable_output_inventory_sha256: str
    result_manifest_file_sha256: str
    report_pdf_file_sha256: str
    public_bundle_input_file_sha256: str
    public_entry_inventory_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.build_token, str)
            or len(self.build_token) != 16
            or any(character not in "0123456789abcdef" for character in self.build_token)
        ):
            raise ValueError("Phase-7 release build token must be 16 lowercase hex digits")
        for name, value in self.manifest_payload().items():
            if name == "build_token":
                continue
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Phase-7 release digest is invalid: {name}")

    def manifest_payload(self) -> dict[str, str]:
        return {
            "build_token": self.build_token,
            "source_registry_sha256": self.source_registry_sha256,
            "compiler_configuration_sha256": self.compiler_configuration_sha256,
            "current_pointer_file_sha256": self.current_pointer_file_sha256,
            "current_pointer_manifest_sha256": self.current_pointer_manifest_sha256,
            "compilation_manifest_file_sha256": self.compilation_manifest_file_sha256,
            "compilation_manifest_sha256": self.compilation_manifest_sha256,
            "immutable_output_inventory_sha256": self.immutable_output_inventory_sha256,
            "result_manifest_file_sha256": self.result_manifest_file_sha256,
            "report_pdf_file_sha256": self.report_pdf_file_sha256,
            "public_bundle_input_file_sha256": self.public_bundle_input_file_sha256,
            "public_entry_inventory_sha256": self.public_entry_inventory_sha256,
        }


@dataclass(frozen=True, slots=True)
class VisualReleaseLineage:
    """Path-free hashes for an accepted inspection of the released report PDF."""

    raster_manifest_file_sha256: str
    raster_manifest_sha256: str
    inspection_receipt_file_sha256: str
    inspection_receipt_sha256: str
    source_pdf_sha256: str
    inspection_status: str = "accepted"

    def __post_init__(self) -> None:
        for name, value in self.manifest_payload().items():
            if name == "inspection_status":
                if value != "accepted":
                    raise ValueError("visual release lineage must record accepted inspection")
                continue
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"visual release digest is invalid: {name}")

    def manifest_payload(self) -> dict[str, str]:
        return {
            "raster_manifest_file_sha256": self.raster_manifest_file_sha256,
            "raster_manifest_sha256": self.raster_manifest_sha256,
            "inspection_receipt_file_sha256": self.inspection_receipt_file_sha256,
            "inspection_receipt_sha256": self.inspection_receipt_sha256,
            "source_pdf_sha256": self.source_pdf_sha256,
            "inspection_status": self.inspection_status,
        }


@dataclass(frozen=True, slots=True)
class NarrativeReleaseLineage:
    """Path-free hashes for the restricted corpus-to-public-table release chain."""

    analysis_receipt_file_sha256: str
    analysis_receipt_sha256: str
    index_manifest_file_sha256: str
    index_manifest_sha256: str
    protected_canary_manifest_file_sha256: str
    protected_canary_manifest_sha256: str
    corpus_sha256: str
    release_scan_receipt_sha256: str
    public_table_file_sha256: str
    publication_status: str = "public_scan_passed"

    def __post_init__(self) -> None:
        for name, value in self.manifest_payload().items():
            if name == "publication_status":
                if value != "public_scan_passed":
                    raise ValueError("narrative release lineage must record a passed public scan")
                continue
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"narrative release digest is invalid: {name}")

    def manifest_payload(self) -> dict[str, str]:
        return {
            "analysis_receipt_file_sha256": self.analysis_receipt_file_sha256,
            "analysis_receipt_sha256": self.analysis_receipt_sha256,
            "index_manifest_file_sha256": self.index_manifest_file_sha256,
            "index_manifest_sha256": self.index_manifest_sha256,
            "protected_canary_manifest_file_sha256": (
                self.protected_canary_manifest_file_sha256
            ),
            "protected_canary_manifest_sha256": self.protected_canary_manifest_sha256,
            "corpus_sha256": self.corpus_sha256,
            "release_scan_receipt_sha256": self.release_scan_receipt_sha256,
            "public_table_file_sha256": self.public_table_file_sha256,
            "publication_status": self.publication_status,
        }


class ProtectedProseCanary(ImmutableRecord):
    """One restricted exact-byte canary; public manifests expose only its hash."""

    canary_id: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"),
    ]
    payload_base64: Annotated[str, StringConstraints(min_length=12, max_length=8192)] = Field(
        repr=False
    )
    payload_sha256: Sha256Digest

    @model_validator(mode="after")
    def payload_is_canonical_and_bound(self) -> Self:
        try:
            decoded = base64.b64decode(self.payload_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("protected prose canary is not canonical base64") from error
        if base64.b64encode(decoded).decode("ascii") != self.payload_base64:
            raise ValueError("protected prose canary base64 is noncanonical")
        if not 8 <= len(decoded) <= 4096:
            raise ValueError("protected prose canary must contain 8..4096 bytes")
        if hashlib.sha256(decoded).hexdigest() != self.payload_sha256:
            raise ValueError("protected prose canary payload hash mismatch")
        return self

    def decoded(self) -> bytes:
        return base64.b64decode(self.payload_base64, validate=True)


class ProtectedProseCanaryManifest(ImmutableRecord):
    """Restricted scanner input required whenever case-study artifacts are published."""

    manifest_id: Annotated[
        str,
        StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"),
    ]
    corpus_hash: Sha256Digest
    canaries: tuple[ProtectedProseCanary, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def inventory_is_nonempty_and_unique(self) -> Self:
        identifiers = [item.canary_id for item in self.canaries]
        payload_hashes = [item.payload_sha256 for item in self.canaries]
        if not identifiers:
            raise ValueError("protected prose canary manifest must not be empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("protected prose canary IDs must be unique")
        if len(payload_hashes) != len(set(payload_hashes)):
            raise ValueError("protected prose canary payloads must be unique")
        return self


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
    upper_name = path.name.upper()
    extensionless_notice = not suffix and (
        upper_name in {"LICENSE", "NOTICE"} or upper_name.endswith("_LICENSE")
    )
    if suffix in PROHIBITED_SUFFIXES or (
        suffix not in ALLOWED_SUFFIXES and not extensionless_notice
    ):
        raise PublicReleaseError(f"prohibited or unknown public artifact suffix: {suffix}")
    return path


def _assert_no_symlink_chain(path: Path) -> None:
    current = path.absolute()
    while True:
        if current.is_symlink():
            raise PublicReleaseError(f"symlinked public-release path is prohibited: {path}")
        if current.parent == current:
            return
        current = current.parent


def _safe_source(root: Path, relative_path: str) -> Path:
    relative = _validate_relative_path(relative_path)
    _assert_no_symlink_chain(root)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError("public source root does not exist") from error
    if not resolved_root.is_dir():
        raise PublicReleaseError("public source root is not a directory")
    candidate = root.absolute().joinpath(*relative.parts)
    _assert_no_symlink_chain(candidate)
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as exc:
        raise PublicReleaseError(f"missing public artifact: {relative_path}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PublicReleaseError(f"symlink prohibited in public bundle: {relative_path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise PublicReleaseError(f"public artifact is not a regular file: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise PublicReleaseError(f"public artifact escapes source root: {relative_path}")
    return resolved


def _safe_explicit_file(root: Path, path: Path, *, label: str) -> Path:
    _assert_no_symlink_chain(root)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"{label} root does not exist") from error
    candidate = path.absolute()
    if not candidate.is_relative_to(root.absolute()):
        raise PublicReleaseError(f"{label} is outside its explicit root")
    _assert_no_symlink_chain(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"missing {label}") from error
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise PublicReleaseError(f"{label} escapes its explicit root")
    return resolved


def load_protected_prose_canaries(
    restricted_root: Path,
    manifest_path: Path,
) -> tuple[ProtectedProseCanaryManifest, tuple[bytes, ...]]:
    """Load exact canaries only from a symlink-free explicit restricted root."""

    source = _safe_explicit_file(
        restricted_root,
        manifest_path,
        label="protected prose canary manifest",
    )
    try:
        if source.stat().st_size > 1024 * 1024:
            raise PublicReleaseError("protected prose canary manifest exceeds 1 MiB")
        manifest = ProtectedProseCanaryManifest.model_validate_json(source.read_bytes())
    except PublicReleaseError:
        raise
    except Exception as error:
        raise PublicReleaseError(f"invalid protected prose canary manifest: {error}") from error
    return manifest, tuple(item.decoded() for item in manifest.canaries)


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


def _load_public_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicReleaseError("invalid public bundle input manifest") from error
    if not isinstance(payload, dict):
        raise PublicReleaseError("public bundle input manifest must be one JSON object")
    supplied_hash = payload.get("manifest_sha256")
    immutable = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if supplied_hash != canonical_sha256(immutable):
        raise PublicReleaseError("public bundle input manifest self-hash mismatch")
    if payload.get("schema_version") != "1.0.0":
        raise PublicReleaseError("unsupported public bundle manifest schema")
    return payload


def load_public_entries(manifest_path: Path) -> tuple[PublicEntry, ...]:
    payload = _load_public_manifest_payload(manifest_path)
    entries = tuple(PublicEntry(**item) for item in payload.get("entries", ()))
    if not entries:
        raise PublicReleaseError("public bundle allowlist is empty")
    targets = [item.bundle_relative_path for item in entries]
    if len(targets) != len(set(targets)):
        raise PublicReleaseError("duplicate public bundle target")
    return entries


def _contains_case_study_publication(entries: Sequence[PublicEntry]) -> bool:
    for entry in entries:
        paths = (entry.source_relative_path, entry.bundle_relative_path)
        if not any(
            PurePosixPath(value).parts
            and PurePosixPath(value).parts[0].lower() in {"artifacts", "reports"}
            for value in paths
        ):
            continue
        components = {
            component.lower()
            for value in paths
            for component in PurePosixPath(value).parts
        }
        if any(
            re.search(r"(?:^|[_-])(?:case|novel)(?:[_-]|$)", item)
            for item in components
        ):
            return True
    return False


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
    protected_canary_manifest_hash: Sha256Digest | None = None,
    phase7_lineage: Phase7ReleaseLineage | None = None,
    visual_release_lineage: VisualReleaseLineage | None = None,
    narrative_release_lineage: NarrativeReleaseLineage | None = None,
) -> dict[str, Any]:
    """Create a new allowlisted directory and deterministic ZIP archive.

    The destination must not exist. This prevents a rebuild from silently retaining a
    file that has been removed from the allowlist.
    """

    safe_manifest = _safe_explicit_file(
        source_root,
        manifest_path,
        label="public bundle input manifest",
    )
    manifest_payload = _load_public_manifest_payload(safe_manifest)
    entries = load_public_entries(safe_manifest)
    compiler_bound = "source_registry_sha256" in manifest_payload
    complete_release = manifest_payload.get("bundle_status") == "complete"
    if (compiler_bound or complete_release) and phase7_lineage is None:
        raise PublicReleaseError(
            "compiler-generated or complete public bundle requires Phase-7 lineage"
        )
    if phase7_lineage is not None:
        entry_inventory_hash = canonical_sha256(
            [
                {
                    "source_relative_path": item.source_relative_path,
                    "bundle_relative_path": item.bundle_relative_path,
                    "sha256": item.sha256,
                    "release_class": item.release_class,
                }
                for item in entries
            ]
        )
        if (
            phase7_lineage.public_bundle_input_file_sha256 != _file_sha256(safe_manifest)
            or phase7_lineage.public_entry_inventory_sha256 != entry_inventory_hash
            or (
                compiler_bound
                and manifest_payload["source_registry_sha256"]
                != phase7_lineage.source_registry_sha256
            )
        ):
            raise PublicReleaseError("Phase-7 release lineage differs from public input")
    case_study_publication = _contains_case_study_publication(entries)
    if complete_release and visual_release_lineage is None:
        raise PublicReleaseError(
            "complete public bundle requires accepted visual-inspection lineage"
        )
    if case_study_publication and (
        not forbidden_canaries or protected_canary_manifest_hash is None
    ):
        raise PublicReleaseError(
            "case-study publication requires a validated protected-prose canary manifest"
        )
    if protected_canary_manifest_hash is not None and not forbidden_canaries:
        raise PublicReleaseError("protected canary manifest hash supplied without canaries")
    if case_study_publication and narrative_release_lineage is None:
        raise PublicReleaseError(
            "case-study publication requires complete narrative release lineage"
        )
    if not case_study_publication and narrative_release_lineage is not None:
        raise PublicReleaseError(
            "narrative release lineage supplied without a case-study publication"
        )
    if (
        narrative_release_lineage is not None
        and narrative_release_lineage.protected_canary_manifest_sha256
        != protected_canary_manifest_hash
    ):
        raise PublicReleaseError(
            "narrative release lineage differs from protected canary manifest"
        )
    if (
        narrative_release_lineage is not None
        and narrative_release_lineage.public_table_file_sha256
        not in {entry.sha256 for entry in entries}
    ):
        raise PublicReleaseError(
            "narrative release lineage does not bind an allowlisted public table"
        )
    if (
        visual_release_lineage is not None
        and phase7_lineage is not None
        and visual_release_lineage.source_pdf_sha256
        != phase7_lineage.report_pdf_file_sha256
    ):
        raise PublicReleaseError("visual release lineage differs from Phase-7 report PDF")
    records = scan_public_entries(
        source_root,
        entries,
        forbidden_canaries=forbidden_canaries,
    )
    _assert_no_symlink_chain(bundle_root)
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
        "source_manifest_sha256": _file_sha256(safe_manifest),
        "phase7_release_lineage": (
            None if phase7_lineage is None else phase7_lineage.manifest_payload()
        ),
        "visual_release_lineage": (
            None
            if visual_release_lineage is None
            else visual_release_lineage.manifest_payload()
        ),
        "narrative_release_lineage": (
            None
            if narrative_release_lineage is None
            else narrative_release_lineage.manifest_payload()
        ),
        "case_study_publication": case_study_publication,
        "protected_canary_manifest_hash": protected_canary_manifest_hash,
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
