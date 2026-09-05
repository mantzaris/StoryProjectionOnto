"""Fail-closed proof for a synthetic benchmark evidence-lineage refresh.

The benchmark is frozen scientific material.  Regenerating it after an
implementation-level provenance correction is therefore not an ordinary build:
the old and candidate trees must have identical scientific semantics, while a
small, explicit set of content-addressed lineage fields is allowed to change.

This module is deliberately CPU-only and depends only on the Python standard
library.  It does not import the benchmark compiler (or scorer contracts), so a
verification cannot silently inherit the implementation that produced the
candidate being checked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

SCHEMA_VERSION = "1.0.0"
POLICY_REVISION = "synthetic-evidence-lineage-refresh/v1"
LEGACY_PROJECTION_RULE = "contracts.to_model_visible_evidence/v1"
EXACT_LINEAGE_PROJECTION_RULE = "contracts.to_model_visible_evidence/v2-exact-lineage"

EXPECTED_BENCHMARK_FILE_COUNT = 259
EXPECTED_GENERATED_NAMESPACE_COUNTS = {
    "condition_input": 48,
    "manifest": 2,
    "model_visible": 176,
    "scorer_only": 32,
}
EXPECTED_SCHEMA_FILE_COUNT = 23
EXPECTED_REVIEW_WORLD_COUNT = 3
EXPECTED_REVIEW_PROJECTION_COUNT = 9
EXPECTED_REVIEW_ITEM_COUNT = 72

_BENCHMARK_MANIFEST = "manifests/benchmark_manifest.json"
_ROUTING = "scorer_only/routing/model_artifacts.json"
_BLIND_PACKAGE = "scorer_only/review/blind_review_package.json"
_REVIEW_BINDINGS = "scorer_only/review/scorer_bindings.json"
_DRAFT_SEAL = "scorer_only/held_out/draft_seal.json"

_EXACT_SCIENTIFIC_FILES = frozenset(
    {
        "README.md",
        "manifests/seed_manifest.json",
        "scorer_only/audits/scientific_integrity.json",
        "scorer_only/mutations/expectations.json",
        "scorer_only/provenance/rejected_candidate.json",
        "scorer_only/review/selection_manifest.json",
        "scorer_only/sampling/a_no_context_selection.json",
        "scorer_only/sampling/eligibility.json",
        "scorer_only/sampling/held_out_query_allocation.json",
        "scorer_only/sampling/paraphrase_selection.json",
        *{
            f"scorer_only/development/syn-dev-{index:02d}.json"
            for index in range(1, 5)
        },
        *{
            f"scorer_only/held_out/syn-test-{index:02d}.json"
            for index in range(1, 13)
        },
    }
)

_BENCHMARK_REVIEW_SCHEMA_FILES = frozenset(
    {
        "scorer_only/review/adjudication.schema.json",
        "scorer_only/review/final_reviewed_seal.schema.json",
        "scorer_only/review/reviewed_projection.schema.json",
        "scorer_only/review/reviewer_response.schema.json",
    }
)

_LINEAGE_SCHEMA_FILES = frozenset(
    {
        "construction_request.schema.json",
        "model_visible_evidence_packet.schema.json",
        "model_visible_evidence_record.schema.json",
        "preconstruction_request.schema.json",
    }
)

_SOURCE_DEPENDENCY_PATHS = {
    "benchmark_runtime.py": "src/story_projection_onto/benchmark_runtime.py",
    "contracts.py": "src/story_projection_onto/contracts.py",
    "metrics/alignment.py": "src/story_projection_onto/metrics/alignment.py",
    "synthetic_benchmark.py": "src/story_projection_onto/synthetic_benchmark.py",
}


class RefreshVerificationError(RuntimeError):
    """Raised when a candidate refresh is not exactly the registered refresh."""


@dataclass(frozen=True, slots=True)
class FileDigest:
    """One regular file in an immutable tree inventory."""

    relative_path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class JsonPointerDifference:
    """One leaf-level old/candidate JSON difference."""

    surface: Literal["benchmark", "schema"]
    relative_path: str
    pointer: str
    operation: Literal["add", "remove", "replace"]
    classification: str
    old_value: Any
    candidate_value: Any


@dataclass(frozen=True, slots=True)
class BenchmarkRefreshReceipt:
    """Canonical, self-hashed proof returned by a successful verification."""

    schema_version: str
    kind: str
    policy_revision: str
    status: Literal["verified"]
    candidate_projection_rule: str
    benchmark_file_count: int
    generated_namespace_counts: Mapping[str, int]
    schema_file_count: int
    old_benchmark_inventory: tuple[FileDigest, ...]
    candidate_benchmark_inventory: tuple[FileDigest, ...]
    old_schema_inventory: tuple[FileDigest, ...]
    candidate_schema_inventory: tuple[FileDigest, ...]
    old_benchmark_tree_sha256: str
    candidate_benchmark_tree_sha256: str
    old_schema_tree_sha256: str
    candidate_schema_tree_sha256: str
    candidate_replay_checked: bool
    exact_scientific_file_count: int
    neutral_artifact_count: int
    model_visible_artifact_file_count: int
    query_reveal_count: int
    review_world_count: int
    review_projection_count: int
    review_item_count: int
    differences: tuple[JsonPointerDifference, ...]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation, including the self-hash."""

        return cast(dict[str, Any], asdict(self))

    def verify_self_hash(self) -> None:
        """Recompute and verify the receipt's canonical checksum."""

        value = self.to_dict()
        supplied = value.pop("receipt_sha256")
        expected = _canonical_sha256(value, strip_content_hashes=False)
        if supplied != expected:
            raise RefreshVerificationError("refresh receipt self-hash mismatch")


@dataclass(frozen=True, slots=True)
class _CandidateSummary:
    neutral_artifact_count: int
    model_visible_artifact_file_count: int
    query_reveal_count: int
    review_world_count: int
    review_projection_count: int
    review_item_count: int


def _json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RefreshVerificationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise RefreshVerificationError(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except RefreshVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RefreshVerificationError(f"cannot parse canonical JSON file: {path}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RefreshVerificationError("value is not canonical-JSON serializable") from exc


def _json_ready(value: Any) -> Any:
    if isinstance(value, FileDigest | JsonPointerDifference):
        return {key: _json_ready(child) for key, child in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(child) for child in value]
    return value


def _without_content_hashes(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_content_hashes(child)
            for key, child in value.items()
            if key != "content_hash"
        }
    if isinstance(value, list):
        return [_without_content_hashes(child) for child in value]
    return value


def _canonical_sha256(value: Any, *, strip_content_hashes: bool = True) -> str:
    normalized = _without_content_hashes(value) if strip_content_hashes else value
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RefreshVerificationError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _safe_root(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise RefreshVerificationError(f"{label} root cannot be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RefreshVerificationError(f"{label} root does not exist") from exc
    if not resolved.is_dir():
        raise RefreshVerificationError(f"{label} root is not a directory")
    return resolved


def _inventory(path: Path, *, label: str) -> tuple[FileDigest, ...]:
    root = _safe_root(path, label=label)
    records: list[FileDigest] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directory_names:
            child = base / name
            if child.is_symlink():
                raise RefreshVerificationError(f"{label} contains a symlinked directory: {child}")
        for name in file_names:
            child = base / name
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise RefreshVerificationError(f"cannot stat {label} file: {child}") from exc
            if not stat.S_ISREG(mode) or child.is_symlink():
                raise RefreshVerificationError(f"{label} contains a non-regular file: {child}")
            relative = child.relative_to(root).as_posix()
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RefreshVerificationError(f"unsafe {label} inventory path: {relative}")
            records.append(
                FileDigest(
                    relative_path=relative,
                    byte_count=child.stat().st_size,
                    sha256=_file_sha256(child),
                )
            )
    records.sort(key=lambda item: item.relative_path)
    paths = [item.relative_path for item in records]
    if not records or paths != sorted(set(paths)):
        raise RefreshVerificationError(f"{label} inventory is empty, duplicate, or unsorted")
    return tuple(records)


def _inventory_hash(records: Sequence[FileDigest]) -> str:
    return _canonical_sha256([asdict(item) for item in records], strip_content_hashes=False)


def _inventory_map(records: Iterable[FileDigest]) -> dict[str, FileDigest]:
    return {item.relative_path: item for item in records}


def _require_same_paths(
    old: Sequence[FileDigest],
    candidate: Sequence[FileDigest],
    *,
    label: str,
) -> tuple[str, ...]:
    old_paths = {item.relative_path for item in old}
    candidate_paths = {item.relative_path for item in candidate}
    if old_paths != candidate_paths:
        raise RefreshVerificationError(
            f"{label} file-set drift; missing={sorted(old_paths - candidate_paths)}, "
            f"extra={sorted(candidate_paths - old_paths)}"
        )
    return tuple(sorted(old_paths))


def _validate_recursive_content_hashes(
    value: Any, *, relative_path: str, pointer: str = ""
) -> None:
    if isinstance(value, Mapping):
        supplied = value.get("content_hash")
        if supplied is not None:
            _require_sha256(supplied, label=f"{relative_path}{pointer}/content_hash")
            expected = _canonical_sha256(value)
            if supplied != expected:
                raise RefreshVerificationError(
                    f"invalid content_hash at {relative_path}{pointer or '/'}"
                )
        for key, child in value.items():
            _validate_recursive_content_hashes(
                child,
                relative_path=relative_path,
                pointer=f"{pointer}/{_escape_pointer(str(key))}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_recursive_content_hashes(
                child,
                relative_path=relative_path,
                pointer=f"{pointer}/{index}",
            )


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _expanded_difference(
    *,
    surface: Literal["benchmark", "schema"],
    relative_path: str,
    pointer: str,
    operation: Literal["add", "remove"],
    old_value: Any,
    candidate_value: Any,
) -> list[JsonPointerDifference]:
    value = candidate_value if operation == "add" else old_value
    if isinstance(value, Mapping):
        if not value:
            return [
                JsonPointerDifference(
                    surface,
                    relative_path,
                    pointer or "",
                    operation,
                    _classify_difference(relative_path, pointer),
                    old_value,
                    candidate_value,
                )
            ]
        result: list[JsonPointerDifference] = []
        for key in sorted(value):
            child_pointer = f"{pointer}/{_escape_pointer(str(key))}"
            result.extend(
                _expanded_difference(
                    surface=surface,
                    relative_path=relative_path,
                    pointer=child_pointer,
                    operation=operation,
                    old_value=None if operation == "add" else value[key],
                    candidate_value=value[key] if operation == "add" else None,
                )
            )
        return result
    if isinstance(value, list):
        if not value:
            return [
                JsonPointerDifference(
                    surface,
                    relative_path,
                    pointer or "",
                    operation,
                    _classify_difference(relative_path, pointer),
                    old_value,
                    candidate_value,
                )
            ]
        result = []
        for index, child in enumerate(value):
            child_pointer = f"{pointer}/{index}"
            result.extend(
                _expanded_difference(
                    surface=surface,
                    relative_path=relative_path,
                    pointer=child_pointer,
                    operation=operation,
                    old_value=None if operation == "add" else child,
                    candidate_value=child if operation == "add" else None,
                )
            )
        return result
    return [
        JsonPointerDifference(
            surface,
            relative_path,
            pointer or "",
            operation,
            _classify_difference(relative_path, pointer),
            old_value,
            candidate_value,
        )
    ]


def _json_differences(
    old: Any,
    candidate: Any,
    *,
    surface: Literal["benchmark", "schema"],
    relative_path: str,
    pointer: str = "",
) -> list[JsonPointerDifference]:
    if isinstance(old, Mapping) and isinstance(candidate, Mapping):
        result: list[JsonPointerDifference] = []
        for key in sorted(set(old) | set(candidate)):
            child_pointer = f"{pointer}/{_escape_pointer(str(key))}"
            if key not in old:
                result.extend(
                    _expanded_difference(
                        surface=surface,
                        relative_path=relative_path,
                        pointer=child_pointer,
                        operation="add",
                        old_value=None,
                        candidate_value=candidate[key],
                    )
                )
            elif key not in candidate:
                result.extend(
                    _expanded_difference(
                        surface=surface,
                        relative_path=relative_path,
                        pointer=child_pointer,
                        operation="remove",
                        old_value=old[key],
                        candidate_value=None,
                    )
                )
            else:
                result.extend(
                    _json_differences(
                        old[key],
                        candidate[key],
                        surface=surface,
                        relative_path=relative_path,
                        pointer=child_pointer,
                    )
                )
        return result
    if isinstance(old, list) and isinstance(candidate, list):
        result = []
        common = min(len(old), len(candidate))
        for index in range(common):
            result.extend(
                _json_differences(
                    old[index],
                    candidate[index],
                    surface=surface,
                    relative_path=relative_path,
                    pointer=f"{pointer}/{index}",
                )
            )
        for index in range(common, len(old)):
            result.extend(
                _expanded_difference(
                    surface=surface,
                    relative_path=relative_path,
                    pointer=f"{pointer}/{index}",
                    operation="remove",
                    old_value=old[index],
                    candidate_value=None,
                )
            )
        for index in range(common, len(candidate)):
            result.extend(
                _expanded_difference(
                    surface=surface,
                    relative_path=relative_path,
                    pointer=f"{pointer}/{index}",
                    operation="add",
                    old_value=None,
                    candidate_value=candidate[index],
                )
            )
        return result
    if type(old) is type(candidate) and old == candidate:
        return []
    return [
        JsonPointerDifference(
            surface,
            relative_path,
            pointer or "",
            "replace",
            _classify_difference(relative_path, pointer),
            old,
            candidate,
        )
    ]


def _classify_difference(relative_path: str, pointer: str) -> str:
    if pointer.endswith("/content_hash"):
        return "derived_content_hash"
    if pointer.endswith("/source_artifact_hash"):
        return "evidence_source_hash"
    if any(pointer.endswith(f"/{field}") for field in ("passage_id", "text_hash", "confidence")):
        return "model_visible_evidence_lineage"
    if "/provenance/" in pointer or pointer.endswith("/provenance"):
        return "model_visible_evidence_lineage"
    if relative_path.endswith(".schema.json") or relative_path == "schema_manifest.json":
        return "generated_schema_lineage"
    if pointer.endswith("_hash") or "_hashes/" in pointer:
        return "derived_artifact_hash"
    if pointer.endswith("_id"):
        return "derived_artifact_identifier"
    if pointer.endswith("/projection_rule"):
        return "evidence_projection_protocol"
    return "allowed_lineage_container"


def _remove_required_name(schema: dict[str, Any], name: str) -> None:
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [item for item in required if item != name]


def _normalize_adjudication_schema(value: Any) -> Any:
    normalized = copy.deepcopy(value)
    if not isinstance(normalized, dict):
        return normalized
    properties = normalized.get("properties")
    if isinstance(properties, dict):
        properties.pop("adjudicator_pseudonym", None)
    _remove_required_name(normalized, "adjudicator_pseudonym")
    return normalized


def _validate_adjudication_schema_upgrade(value: Any) -> None:
    schema = _require_mapping(value, label="review adjudication schema")
    properties = _require_mapping(
        schema.get("properties"), label="review adjudication schema properties"
    )
    expected = {
        "maxLength": 192,
        "minLength": 1,
        "pattern": "^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
        "title": "Adjudicator Pseudonym",
        "type": "string",
    }
    required = _require_list(schema.get("required"), label="review adjudication required")
    if properties.get("adjudicator_pseudonym") != expected or required.count(
        "adjudicator_pseudonym"
    ) != 1:
        raise RefreshVerificationError(
            "review adjudication schema lacks the exact independent-adjudicator identity field"
        )


def _normalize_benchmark_manifest(value: Any) -> Any:
    normalized = _without_content_hashes(copy.deepcopy(value))
    if not isinstance(normalized, dict):
        return normalized
    for key in (
        "compiler_dependency_hashes",
        "draft_seal_hash",
        "generator_source_hash",
        "review_binding_manifest_hash",
        "review_package_hash",
        "runtime_source_hash",
    ):
        normalized.pop(key, None)
    generated = normalized.get("generated_files")
    if isinstance(generated, list):
        for item in generated:
            if isinstance(item, dict):
                item.pop("byte_count", None)
                item.pop("sha256", None)
    return normalized


def _normalize_benchmark_document(relative_path: str, value: Any) -> Any:
    # In a JSON Schema, ``content_hash`` is a property name rather than a
    # materialized record checksum and must never be erased generically.
    if relative_path.endswith(".schema.json"):
        if relative_path == "scorer_only/review/adjudication.schema.json":
            return _normalize_adjudication_schema(value)
        return copy.deepcopy(value)
    normalized = _without_content_hashes(copy.deepcopy(value))
    if "/neutral_evidence/" in relative_path and relative_path.endswith(
        "/neutral_evidence.json"
    ):
        if isinstance(normalized, dict) and isinstance(normalized.get("evidence"), list):
            for record in normalized["evidence"]:
                if isinstance(record, dict) and isinstance(record.get("provenance"), dict):
                    record["provenance"].pop("source_artifact_hash", None)
        return normalized
    if relative_path.startswith("model_visible/") and relative_path.endswith("/evidence.json"):
        if isinstance(normalized, dict) and isinstance(normalized.get("evidence"), list):
            for record in normalized["evidence"]:
                if isinstance(record, dict):
                    for key in ("passage_id", "text_hash", "provenance", "confidence"):
                        record.pop(key, None)
        return normalized
    if relative_path.startswith("model_visible/query_stages/") and relative_path.endswith(
        "/query.json"
    ):
        if isinstance(normalized, dict):
            normalized.pop("evidence_artifact_hash", None)
        return normalized
    if relative_path.endswith("/manifest.json") and (
        relative_path.startswith("model_visible/")
        or relative_path.startswith("condition_inputs/neutral_evidence/")
    ):
        if isinstance(normalized, dict):
            normalized.pop("artifact_hashes", None)
            normalized.pop("stage_id", None)
        return normalized
    if relative_path.endswith("/equivalence.json"):
        if isinstance(normalized, dict):
            for key in (
                "certificate_id",
                "model_visible_evidence_artifact_hash",
                "neutral_evidence_artifact_hash",
                "ordered_full_evidence_hash",
                "ordered_model_visible_evidence_hash",
                "projection_rule",
            ):
                normalized.pop(key, None)
        return normalized
    if relative_path == _ROUTING:
        if isinstance(normalized, list):
            for entry in normalized:
                if isinstance(entry, dict):
                    for key in (
                        "artifact_hash",
                        "evidence_equivalence_certificate_hash",
                        "neutral_evidence_artifact_hash",
                        "query_reveal_hashes",
                    ):
                        entry.pop(key, None)
        return normalized
    if relative_path == _BLIND_PACKAGE:
        if isinstance(normalized, dict) and isinstance(normalized.get("worlds"), list):
            for world in normalized["worlds"]:
                if not isinstance(world, dict) or not isinstance(world.get("evidence"), list):
                    continue
                for record in world["evidence"]:
                    if isinstance(record, dict):
                        for key in ("passage_id", "text_hash", "provenance", "confidence"):
                            record.pop(key, None)
        return normalized
    if relative_path == _REVIEW_BINDINGS:
        if isinstance(normalized, dict):
            normalized.pop("package_hash", None)
            entries = normalized.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        entry.pop("blind_projection_hash", None)
        return normalized
    if relative_path == _DRAFT_SEAL:
        if isinstance(normalized, dict):
            for key in (
                "compiler_dependency_hashes",
                "generator_source_hash",
                "review_binding_manifest_hash",
                "review_package_hash",
                "runtime_source_hash",
                "seal_id",
            ):
                normalized.pop(key, None)
            entries = normalized.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        entry.pop("model_eligible_artifact_hash", None)
        return normalized
    if relative_path == _BENCHMARK_MANIFEST:
        return _normalize_benchmark_manifest(value)
    return normalized


def _raise_semantic_difference(
    *,
    label: str,
    relative_path: str,
    old: Any,
    candidate: Any,
) -> None:
    differences = _json_differences(
        old,
        candidate,
        surface="benchmark",
        relative_path=relative_path,
    )
    preview = ", ".join(item.pointer or "/" for item in differences[:8])
    raise RefreshVerificationError(
        f"{label} changed outside the evidence-lineage allowlist in {relative_path}: {preview}"
    )


def _opaque(prefix: str, *parts: object) -> str:
    material = "\0".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(material).hexdigest()[:20]}"


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RefreshVerificationError(f"{label} must be a JSON object")
    return value


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RefreshVerificationError(f"{label} must be a JSON array")
    return value


def _validate_generated_inventory(
    root: Path,
    inventory: Sequence[FileDigest],
) -> Mapping[str, int]:
    manifest = _require_mapping(
        _load_json(root / _BENCHMARK_MANIFEST), label="benchmark manifest"
    )
    _validate_recursive_content_hashes(manifest, relative_path=_BENCHMARK_MANIFEST)
    generated = _require_list(manifest.get("generated_files"), label="generated_files")
    actual = _inventory_map(inventory)
    expected_paths = set(actual) - {_BENCHMARK_MANIFEST}
    observed_paths: list[str] = []
    namespaces: Counter[str] = Counter()
    for index, raw in enumerate(generated):
        entry = _require_mapping(raw, label=f"generated_files[{index}]")
        if set(entry) != {
            "byte_count",
            "content_hash",
            "namespace",
            "relative_path",
            "release_class",
            "schema_version",
            "sha256",
        }:
            raise RefreshVerificationError("generated-file manifest entry shape changed")
        relative = entry.get("relative_path")
        if not isinstance(relative, str) or relative not in expected_paths:
            raise RefreshVerificationError("generated-file manifest names an unknown path")
        observed_paths.append(relative)
        record = actual[relative]
        if (
            entry.get("byte_count") != record.byte_count
            or entry.get("sha256") != record.sha256
            or entry.get("release_class") != "public"
            or entry.get("schema_version") != SCHEMA_VERSION
        ):
            raise RefreshVerificationError(
                f"generated-file manifest does not bind exact bytes: {relative}"
            )
        namespace = entry.get("namespace")
        if namespace not in EXPECTED_GENERATED_NAMESPACE_COUNTS:
            raise RefreshVerificationError(f"unknown generated namespace for {relative}")
        expected_namespace = (
            "condition_input"
            if relative.startswith("condition_inputs/")
            else "model_visible"
            if relative.startswith("model_visible/")
            else "scorer_only"
            if relative.startswith("scorer_only/")
            else "manifest"
        )
        if namespace != expected_namespace:
            raise RefreshVerificationError(f"generated namespace/path mismatch: {relative}")
        namespaces[namespace] += 1
    if observed_paths != sorted(expected_paths):
        raise RefreshVerificationError(
            "benchmark generated-file inventory is incomplete or unsorted"
        )
    if dict(sorted(namespaces.items())) != EXPECTED_GENERATED_NAMESPACE_COUNTS:
        raise RefreshVerificationError(
            f"generated namespace counts changed: {dict(sorted(namespaces.items()))}"
        )
    return dict(sorted(namespaces.items()))


def _validate_neutral_record(record: Any, *, exact_lineage: bool, label: str) -> None:
    value = _require_mapping(record, label=label)
    evidence_id = value.get("evidence_id")
    text = value.get("text")
    passage_id = value.get("passage_id")
    text_hash = value.get("text_hash")
    confidence = value.get("confidence")
    provenance = _require_mapping(value.get("provenance"), label=f"{label}.provenance")
    if not all(isinstance(item, str) and item for item in (evidence_id, text, passage_id)):
        raise RefreshVerificationError(f"{label} has invalid evidence identity or text")
    expected_text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash != expected_text_hash:
        raise RefreshVerificationError(f"{label} text_hash does not bind exact UTF-8 text")
    if confidence != 1.0 or provenance.get("confidence") != 1.0:
        raise RefreshVerificationError(f"{label} changed the synthetic confidence ceiling")
    if provenance.get("evidence_id") != evidence_id:
        raise RefreshVerificationError(f"{label} provenance names another evidence record")
    source_hash = provenance.get("source_artifact_hash")
    if exact_lineage and source_hash != expected_text_hash:
        raise RefreshVerificationError(
            f"{label} source_artifact_hash does not bind exact rendered passage bytes"
        )
    if not exact_lineage and source_hash is not None:
        raise RefreshVerificationError(f"legacy {label} unexpectedly contains exact source lineage")


def _project_neutral_record(record: Mapping[str, Any], *, exact_lineage: bool) -> dict[str, Any]:
    projected = cast(dict[str, Any], _without_content_hashes(copy.deepcopy(record)))
    projected.pop("release_class", None)
    if not exact_lineage:
        for key in ("passage_id", "text_hash", "provenance", "confidence"):
            projected.pop(key, None)
    return projected


def _validate_model_evidence_against_neutral(
    model_document: Any,
    neutral_document: Any,
    *,
    exact_lineage: bool,
    label: str,
) -> None:
    model = _require_mapping(model_document, label=label)
    neutral = _require_mapping(neutral_document, label=f"{label} neutral source")
    model_records = _require_list(model.get("evidence"), label=f"{label}.evidence")
    neutral_records = _require_list(neutral.get("evidence"), label=f"{label}.neutral_evidence")
    if len(model_records) != len(neutral_records):
        raise RefreshVerificationError(f"{label} changed evidence cardinality")
    for index, (model_record, neutral_record) in enumerate(
        zip(model_records, neutral_records, strict=True)
    ):
        neutral_mapping = _require_mapping(
            neutral_record, label=f"{label}.neutral_evidence[{index}]"
        )
        expected = _project_neutral_record(neutral_mapping, exact_lineage=exact_lineage)
        observed = _without_content_hashes(model_record)
        if observed != expected:
            raise RefreshVerificationError(
                f"{label}.evidence[{index}] is not the exact neutral-evidence projection"
            )


def _validate_stage_manifest(
    value: Any,
    *,
    stage_kind: str,
    artifact_hashes: Sequence[str],
    file_names: Sequence[str],
    expected_stage_id: str,
    label: str,
) -> None:
    manifest = _require_mapping(value, label=label)
    if (
        manifest.get("stage_kind") != stage_kind
        or manifest.get("artifact_hashes") != list(artifact_hashes)
        or manifest.get("file_names") != list(file_names)
        or manifest.get("manifest_file_name") != "manifest.json"
        or manifest.get("stage_id") != expected_stage_id
    ):
        raise RefreshVerificationError(f"{label} does not bind its exact stage contents")


def _validate_review_package_lineage(
    package: Mapping[str, Any],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    *,
    exact_lineage: bool,
) -> tuple[int, int, int, dict[str, str]]:
    worlds = _require_list(package.get("worlds"), label="blind review worlds")
    if (
        len(worlds) != EXPECTED_REVIEW_WORLD_COUNT
        or package.get("condition_blind") is not True
        or package.get("contains_method_outputs") is not False
        or package.get("selection_rule") != "one-seeded-world-per-difficulty-stratum"
    ):
        raise RefreshVerificationError("blind review package changed its registered blind design")
    projection_count = 0
    review_item_count = 0
    projection_hashes: dict[str, str] = {}
    for world_index, raw_world in enumerate(worlds):
        world = _require_mapping(raw_world, label=f"review world {world_index}")
        evidence = _require_list(world.get("evidence"), label="review-world evidence")
        for evidence_index, raw_record in enumerate(evidence):
            record = _require_mapping(raw_record, label="review evidence record")
            evidence_id = record.get("evidence_id")
            neutral = evidence_by_id.get(cast(str, evidence_id))
            if neutral is None:
                raise RefreshVerificationError(
                    "review package evidence is absent from neutral index"
                )
            expected = _project_neutral_record(neutral, exact_lineage=exact_lineage)
            if _without_content_hashes(record) != expected:
                raise RefreshVerificationError(
                    f"review evidence {world_index}/{evidence_index} changed beyond exact lineage"
                )
        projections = _require_list(world.get("projections"), label="review projections")
        projection_count += len(projections)
        for projection in projections:
            projection_mapping = _require_mapping(projection, label="review projection")
            projection_id = projection_mapping.get("blind_projection_id")
            projection_hash = projection_mapping.get("content_hash")
            if not isinstance(projection_id, str) or projection_id in projection_hashes:
                raise RefreshVerificationError("review projection IDs are invalid or duplicate")
            projection_hashes[projection_id] = _require_sha256(
                projection_hash, label="blind projection hash"
            )
            items = _require_list(projection_mapping.get("review_items"), label="review items")
            review_item_count += len(items)
            for assertion in _require_list(
                projection_mapping.get("proposed_qualified_assertions"),
                label="review proposed qualified assertions",
            ):
                assertion_mapping = _require_mapping(assertion, label="review assertion")
                for provenance in _require_list(
                    assertion_mapping.get("provenance"), label="review assertion provenance"
                ):
                    provenance_mapping = _require_mapping(
                        provenance, label="review assertion provenance entry"
                    )
                    evidence_id = provenance_mapping.get("evidence_id")
                    neutral = evidence_by_id.get(cast(str, evidence_id))
                    if neutral is None:
                        raise RefreshVerificationError(
                            "review assertion provenance cites absent neutral evidence"
                        )
                    neutral_provenance = _require_mapping(
                        neutral.get("provenance"), label="neutral provenance"
                    )
                    expected_provenance = cast(
                        dict[str, Any], _without_content_hashes(neutral_provenance)
                    )
                    # The condition-blinding transform deliberately removes record
                    # envelope fields from proposal dictionaries.  Proposed gold
                    # remains the byte-frozen scorer representation: the new source
                    # digest belongs to neutral/model-visible evidence, not a post-
                    # freeze rewrite of the known-answer ontology.
                    expected_provenance.pop("schema_version", None)
                    expected_provenance["source_artifact_hash"] = None
                    if _without_content_hashes(provenance_mapping) != expected_provenance:
                        raise RefreshVerificationError(
                            "review assertion provenance differs from its neutral source"
                        )
    if (
        projection_count != EXPECTED_REVIEW_PROJECTION_COUNT
        or review_item_count != EXPECTED_REVIEW_ITEM_COUNT
    ):
        raise RefreshVerificationError(
            "blind review package changed projection or rubric-item cardinality"
        )
    return len(worlds), projection_count, review_item_count, projection_hashes


def _validate_source_bindings(
    draft: Mapping[str, Any], manifest: Mapping[str, Any], candidate_source_root: Path
) -> None:
    root = _safe_root(candidate_source_root, label="candidate source")
    expected: dict[str, str] = {}
    for label, relative in _SOURCE_DEPENDENCY_PATHS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RefreshVerificationError(f"candidate source dependency is unsafe: {relative}")
        expected[label] = _file_sha256(path)
    if draft.get("compiler_dependency_hashes") != expected:
        raise RefreshVerificationError("draft seal compiler hashes do not bind candidate source")
    if manifest.get("compiler_dependency_hashes") != expected:
        raise RefreshVerificationError("benchmark manifest compiler hashes do not bind source")
    if (
        draft.get("generator_source_hash") != expected["synthetic_benchmark.py"]
        or manifest.get("generator_source_hash") != expected["synthetic_benchmark.py"]
        or draft.get("runtime_source_hash") != expected["benchmark_runtime.py"]
        or manifest.get("runtime_source_hash") != expected["benchmark_runtime.py"]
    ):
        raise RefreshVerificationError("generator/runtime source hashes are inconsistent")


def _validate_candidate_tree(
    root: Path,
    inventory: Sequence[FileDigest],
    *,
    candidate_projection_rule: str,
    candidate_source_root: Path | None,
) -> _CandidateSummary:
    exact_lineage = candidate_projection_rule == EXACT_LINEAGE_PROJECTION_RULE
    if candidate_projection_rule not in {LEGACY_PROJECTION_RULE, EXACT_LINEAGE_PROJECTION_RULE}:
        raise RefreshVerificationError("unsupported candidate evidence projection rule")

    json_values: dict[str, Any] = {}
    for record in inventory:
        if not record.relative_path.endswith(".json"):
            continue
        value = _load_json(root / record.relative_path)
        json_values[record.relative_path] = value
        if not record.relative_path.endswith(".schema.json"):
            _validate_recursive_content_hashes(value, relative_path=record.relative_path)

    routing = _require_list(json_values[_ROUTING], label="model routing")
    if len(routing) != 16:
        raise RefreshVerificationError("model routing must contain exactly sixteen worlds")
    world_ids: list[str] = []
    all_query_paths: list[str] = []
    all_prequery_paths: list[str] = []
    all_neutral_paths: list[str] = []
    neutral_evidence_by_id: dict[str, Mapping[str, Any]] = {}

    for index, raw_entry in enumerate(routing):
        entry = _require_mapping(raw_entry, label=f"routing[{index}]")
        world_id = entry.get("world_id")
        split = entry.get("split")
        expected_split = "development" if isinstance(world_id, str) and world_id.startswith(
            "syn-dev-"
        ) else "held_out"
        if not isinstance(world_id, str) or split != expected_split:
            raise RefreshVerificationError("routing world/split binding changed")
        world_ids.append(world_id)
        neutral_relative = entry.get("relative_neutral_evidence_stage_path")
        prequery_relative = entry.get("relative_prequery_stage_path")
        query_relatives = entry.get("relative_query_stage_paths")
        if (
            not isinstance(neutral_relative, str)
            or not neutral_relative.startswith("condition_inputs/neutral_evidence/neutral_")
            or not isinstance(prequery_relative, str)
            or not prequery_relative.startswith("model_visible/prequery_stages/artifact_")
            or not isinstance(query_relatives, list)
            or len(query_relatives) != 3
            or any(
                not isinstance(item, str)
                or not item.startswith("model_visible/query_stages/reveal_")
                for item in query_relatives
            )
        ):
            raise RefreshVerificationError("routing contains an unsafe or malformed stage path")
        all_neutral_paths.append(neutral_relative)
        all_prequery_paths.append(prequery_relative)
        all_query_paths.extend(cast(list[str], query_relatives))

        neutral = _require_mapping(
            json_values[f"{neutral_relative}/neutral_evidence.json"],
            label=f"neutral artifact for {world_id}",
        )
        equivalence = _require_mapping(
            json_values[f"{neutral_relative}/equivalence.json"],
            label=f"equivalence certificate for {world_id}",
        )
        artifact = _require_mapping(
            json_values[f"{prequery_relative}/evidence.json"],
            label=f"model-visible artifact for {world_id}",
        )
        neutral_hash = _require_sha256(neutral.get("content_hash"), label="neutral artifact hash")
        artifact_hash = _require_sha256(artifact.get("content_hash"), label="model artifact hash")
        equivalence_hash = _require_sha256(
            equivalence.get("content_hash"), label="equivalence certificate hash"
        )
        if (
            neutral.get("artifact_id") != Path(neutral_relative).name
            or artifact.get("artifact_id") != Path(prequery_relative).name
            or entry.get("artifact_id") != artifact.get("artifact_id")
            or entry.get("neutral_evidence_artifact_hash") != neutral_hash
            or entry.get("artifact_hash") != artifact_hash
            or entry.get("evidence_equivalence_certificate_hash") != equivalence_hash
        ):
            raise RefreshVerificationError("routing does not bind its neutral/model artifacts")
        neutral_snapshot = _require_mapping(neutral.get("snapshot"), label="neutral snapshot")
        artifact_snapshot = _require_mapping(artifact.get("snapshot"), label="model snapshot")
        neutral_records = _require_list(neutral.get("evidence"), label="neutral evidence")
        for evidence_index, record in enumerate(neutral_records):
            _validate_neutral_record(
                record,
                exact_lineage=exact_lineage,
                label=f"{world_id}.evidence[{evidence_index}]",
            )
            mapping = _require_mapping(record, label="neutral evidence record")
            evidence_id = mapping.get("evidence_id")
            if not isinstance(evidence_id, str) or evidence_id in neutral_evidence_by_id:
                raise RefreshVerificationError("neutral evidence IDs are invalid or non-unique")
            neutral_evidence_by_id[evidence_id] = mapping
        _validate_model_evidence_against_neutral(
            artifact,
            neutral,
            exact_lineage=exact_lineage,
            label=f"{world_id}.prequery",
        )
        if (
            equivalence.get("projection_rule") != candidate_projection_rule
            or equivalence.get("neutral_evidence_artifact_hash") != neutral_hash
            or equivalence.get("model_visible_evidence_artifact_hash") != artifact_hash
            or equivalence.get("snapshot_hash") != neutral_snapshot.get("content_hash")
            or neutral_snapshot != artifact_snapshot
            or equivalence.get("ordered_full_evidence_hash")
            != _canonical_sha256(neutral_records)
            or equivalence.get("ordered_model_visible_evidence_hash")
            != _canonical_sha256(artifact.get("evidence"))
            or equivalence.get("evidence_count") != len(neutral_records)
            or equivalence.get("certificate_id")
            != _opaque("equivalence", neutral_hash, artifact_hash)
        ):
            raise RefreshVerificationError("evidence-equivalence certificate is not reproducible")
        _validate_stage_manifest(
            json_values[f"{neutral_relative}/manifest.json"],
            stage_kind="neutral_evidence",
            artifact_hashes=(neutral_hash, equivalence_hash),
            file_names=("neutral_evidence.json", "equivalence.json"),
            expected_stage_id=_opaque("neutralstage", neutral_hash),
            label=f"neutral stage for {world_id}",
        )
        _validate_stage_manifest(
            json_values[f"{prequery_relative}/manifest.json"],
            stage_kind="prequery_evidence",
            artifact_hashes=(artifact_hash,),
            file_names=("evidence.json",),
            expected_stage_id=_opaque("stage", artifact_hash, "prequery"),
            label=f"prequery stage for {world_id}",
        )
        reveal_hashes: list[str] = []
        for query_relative in cast(list[str], query_relatives):
            query_evidence_path = root / query_relative / "evidence.json"
            prequery_evidence_path = root / prequery_relative / "evidence.json"
            if query_evidence_path.read_bytes() != prequery_evidence_path.read_bytes():
                raise RefreshVerificationError(
                    "query stage does not reuse exact sealed evidence bytes"
                )
            reveal = _require_mapping(
                json_values[f"{query_relative}/query.json"], label="query reveal"
            )
            reveal_hash = _require_sha256(reveal.get("content_hash"), label="query reveal hash")
            reveal_hashes.append(reveal_hash)
            if (
                reveal.get("reveal_id") != Path(query_relative).name
                or reveal.get("evidence_artifact_hash") != artifact_hash
            ):
                raise RefreshVerificationError("query reveal is not bound to sealed evidence")
            _validate_stage_manifest(
                json_values[f"{query_relative}/manifest.json"],
                stage_kind="query_revealed",
                artifact_hashes=(artifact_hash, reveal_hash),
                file_names=("evidence.json", "query.json"),
                expected_stage_id=_opaque("stage", reveal_hash, "query"),
                label=f"query stage {query_relative}",
            )
        if entry.get("query_reveal_hashes") != reveal_hashes:
            raise RefreshVerificationError("routing query order/hash bindings changed")

    if len(world_ids) != len(set(world_ids)):
        raise RefreshVerificationError("routing worlds are duplicate")
    if len(set(all_neutral_paths)) != 16 or len(set(all_prequery_paths)) != 16:
        raise RefreshVerificationError(
            "routing does not assign one unique prequery stage per world"
        )
    if len(set(all_query_paths)) != 48:
        raise RefreshVerificationError("routing does not assign forty-eight unique query stages")

    package = _require_mapping(json_values[_BLIND_PACKAGE], label="blind review package")
    review_world_count, projection_count, item_count, projection_hashes = (
        _validate_review_package_lineage(
            package,
            neutral_evidence_by_id,
            exact_lineage=exact_lineage,
        )
    )
    bindings = _require_mapping(json_values[_REVIEW_BINDINGS], label="review bindings")
    binding_entries = _require_list(bindings.get("entries"), label="review binding entries")
    if (
        len(binding_entries) != EXPECTED_REVIEW_PROJECTION_COUNT
        or bindings.get("package_hash") != package.get("content_hash")
    ):
        raise RefreshVerificationError("review binding manifest names another package")
    for raw_binding in binding_entries:
        binding = _require_mapping(raw_binding, label="review binding")
        projection_id = binding.get("blind_projection_id")
        if projection_hashes.get(cast(str, projection_id)) != binding.get("blind_projection_hash"):
            raise RefreshVerificationError("review binding names another blind projection")

    draft = _require_mapping(json_values[_DRAFT_SEAL], label="held-out draft seal")
    manifest = _require_mapping(json_values[_BENCHMARK_MANIFEST], label="benchmark manifest")
    if (
        draft.get("review_package_hash") != package.get("content_hash")
        or draft.get("review_binding_manifest_hash") != bindings.get("content_hash")
        or manifest.get("review_package_hash") != package.get("content_hash")
        or manifest.get("review_binding_manifest_hash") != bindings.get("content_hash")
        or manifest.get("draft_seal_hash") != draft.get("content_hash")
        or draft.get("seal_id")
        != _opaque("draftseal", draft.get("configuration_hash"), package.get("content_hash"))
    ):
        raise RefreshVerificationError("draft seal/review/manifest lineage is inconsistent")
    if candidate_source_root is not None:
        _validate_source_bindings(draft, manifest, candidate_source_root)

    return _CandidateSummary(
        neutral_artifact_count=len(all_neutral_paths),
        model_visible_artifact_file_count=len(all_prequery_paths) + len(all_query_paths),
        query_reveal_count=len(all_query_paths),
        review_world_count=review_world_count,
        review_projection_count=projection_count,
        review_item_count=item_count,
    )


def _schema_manifest(root: Path, inventory: Sequence[FileDigest]) -> Mapping[str, Any]:
    manifest = _require_mapping(_load_json(root / "schema_manifest.json"), label="schema manifest")
    if set(manifest) != {
        "generator",
        "manifest_hash",
        "schema_version",
        "schemas",
        "scorer_only_contracts_included",
    }:
        raise RefreshVerificationError("schema manifest shape changed")
    supplied = _require_sha256(manifest.get("manifest_hash"), label="schema manifest hash")
    immutable = dict(manifest)
    immutable.pop("manifest_hash")
    if supplied != _canonical_sha256(immutable, strip_content_hashes=False):
        raise RefreshVerificationError("schema manifest self-hash mismatch")
    if (
        manifest.get("generator") != "scripts/generate_schemas.py"
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("scorer_only_contracts_included") is not False
    ):
        raise RefreshVerificationError("schema manifest header changed")
    actual = _inventory_map(inventory)
    entries = _require_list(manifest.get("schemas"), label="schema manifest entries")
    observed: list[str] = []
    contracts: list[str] = []
    for raw in entries:
        entry = _require_mapping(raw, label="schema manifest entry")
        if set(entry) != {"bytes", "contract", "file", "sha256", "surface"}:
            raise RefreshVerificationError("schema manifest entry shape changed")
        relative = entry.get("file")
        contract = entry.get("contract")
        if not isinstance(relative, str) or relative not in actual:
            raise RefreshVerificationError("schema manifest names an absent schema")
        if not isinstance(contract, str):
            raise RefreshVerificationError("schema manifest contract is invalid")
        record = actual[relative]
        if entry.get("bytes") != record.byte_count or entry.get("sha256") != record.sha256:
            raise RefreshVerificationError(f"schema manifest does not bind exact bytes: {relative}")
        if entry.get("surface") not in {"model_interaction", "public_contract"}:
            raise RefreshVerificationError("schema manifest surface changed")
        observed.append(relative)
        contracts.append(contract)
    if set(observed) != set(actual) - {"schema_manifest.json"} or len(observed) != len(
        set(observed)
    ):
        raise RefreshVerificationError("schema manifest file inventory is incomplete or duplicate")
    if contracts != sorted(set(contracts)):
        raise RefreshVerificationError("schema manifest contracts are duplicate or unsorted")
    return manifest


def _collapse_schema_union(value: Any) -> Any:
    """Erase only the strict/legacy evidence union added by this refresh."""

    if isinstance(value, list):
        return [_collapse_schema_union(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {str(key): _collapse_schema_union(child) for key, child in value.items()}
    reference = normalized.get("$ref")
    if reference == "#/$defs/LegacyModelVisibleEvidenceRecord":
        normalized["$ref"] = "#/$defs/ModelVisibleEvidenceRecord"
    any_of = normalized.get("anyOf")
    if isinstance(any_of, list):
        references = {
            item.get("$ref")
            for item in any_of
            if isinstance(item, dict) and set(item) == {"$ref"}
        }
        if len(any_of) == 2 and references in (
            {
                "#/$defs/ModelVisibleEvidenceRecord",
                "#/$defs/LegacyModelVisibleEvidenceRecord",
            },
            {"#/$defs/ModelVisibleEvidenceRecord"},
        ):
            normalized.pop("anyOf")
            normalized["$ref"] = "#/$defs/ModelVisibleEvidenceRecord"
    definitions = normalized.get("$defs")
    if isinstance(definitions, dict) and "LegacyModelVisibleEvidenceRecord" in definitions:
        legacy = cast(dict[str, Any], definitions.pop("LegacyModelVisibleEvidenceRecord"))
        legacy = copy.deepcopy(legacy)
        legacy["title"] = "ModelVisibleEvidenceRecord"
        definitions["ModelVisibleEvidenceRecord"] = legacy
    return normalized


def _referenced_definitions(schema: Mapping[str, Any]) -> set[str]:
    references: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                references.add(reference.removeprefix("#/$defs/"))
            for key, child in value.items():
                if key != "$defs":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    definitions = schema.get("$defs")
    if isinstance(definitions, Mapping):
        pending = list(references)
        while pending:
            name = pending.pop()
            definition = definitions.get(name)
            before = set(references)
            visit(definition)
            pending.extend(sorted(references - before))
    return references


def _prune_unreferenced_definitions(schema: dict[str, Any]) -> None:
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return
    referenced = _referenced_definitions(schema)
    for name in tuple(definitions):
        if name not in referenced:
            definitions.pop(name)


def _normalize_lineage_schema(relative_path: str, value: Any) -> Any:
    normalized = _collapse_schema_union(copy.deepcopy(value))
    if not isinstance(normalized, dict):
        return normalized
    if relative_path == "model_visible_evidence_record.schema.json":
        normalized.pop("description", None)
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            for key in ("passage_id", "text_hash", "provenance", "confidence"):
                properties.pop(key, None)
                _remove_required_name(normalized, key)
    definitions = normalized.get("$defs")
    if isinstance(definitions, dict):
        evidence_definition = definitions.get("ModelVisibleEvidenceRecord")
        if isinstance(evidence_definition, dict):
            evidence_definition.pop("description", None)
            properties = evidence_definition.get("properties")
            # After union collapse this is the legacy definition.  This branch also
            # handles a direct strict definition defensively without widening any
            # other schema path.
            if isinstance(properties, dict) and "passage_id" in properties:
                for key in ("passage_id", "text_hash", "provenance", "confidence"):
                    properties.pop(key, None)
                    _remove_required_name(evidence_definition, key)
    if relative_path == "preconstruction_request.schema.json":
        properties = normalized.get("properties")
        if isinstance(properties, dict):
            properties.pop("sealed_horizon", None)
        _remove_required_name(normalized, "sealed_horizon")
    _prune_unreferenced_definitions(normalized)
    return normalized


def _validate_candidate_lineage_schemas(root: Path) -> None:
    strict = _require_mapping(
        _load_json(root / "model_visible_evidence_record.schema.json"),
        label="model-visible evidence schema",
    )
    properties = _require_mapping(strict.get("properties"), label="evidence schema properties")
    required = _require_list(strict.get("required"), label="evidence schema required")
    lineage_fields = {"passage_id", "text_hash", "provenance", "confidence"}
    if not lineage_fields.issubset(properties) or not lineage_fields.issubset(required):
        raise RefreshVerificationError("strict model-visible evidence schema lacks exact lineage")
    text_hash = _require_mapping(properties["text_hash"], label="text_hash schema")
    confidence = _require_mapping(properties["confidence"], label="confidence schema")
    if (
        text_hash.get("pattern") != "^[0-9a-f]{64}$"
        or confidence.get("minimum") != 0.0
        or confidence.get("maximum") != 1.0
        or strict.get("additionalProperties") is not False
    ):
        raise RefreshVerificationError("strict evidence lineage schema constraints changed")
    preconstruction = _require_mapping(
        _load_json(root / "preconstruction_request.schema.json"),
        label="preconstruction request schema",
    )
    preconstruction_properties = _require_mapping(
        preconstruction.get("properties"), label="preconstruction properties"
    )
    if "sealed_horizon" not in preconstruction_properties:
        raise RefreshVerificationError("preconstruction schema lacks its sealed horizon")


def _compare_schema_roots(
    old_root: Path,
    candidate_root: Path,
    old_inventory: Sequence[FileDigest],
    candidate_inventory: Sequence[FileDigest],
    *,
    require_lineage_upgrade: bool,
) -> tuple[JsonPointerDifference, ...]:
    old_manifest = _schema_manifest(old_root, old_inventory)
    candidate_manifest = _schema_manifest(candidate_root, candidate_inventory)
    old_entries = {
        cast(str, item["file"]): {
            "contract": item["contract"],
            "surface": item["surface"],
        }
        for item in cast(list[dict[str, Any]], old_manifest["schemas"])
    }
    candidate_entries = {
        cast(str, item["file"]): {
            "contract": item["contract"],
            "surface": item["surface"],
        }
        for item in cast(list[dict[str, Any]], candidate_manifest["schemas"])
    }
    if old_entries != candidate_entries:
        raise RefreshVerificationError("schema contract/file/surface inventory changed")
    differences: list[JsonPointerDifference] = []
    old_map = _inventory_map(old_inventory)
    candidate_map = _inventory_map(candidate_inventory)
    for relative in sorted(old_map):
        if old_map[relative].sha256 == candidate_map[relative].sha256:
            continue
        old_value = _load_json(old_root / relative)
        candidate_value = _load_json(candidate_root / relative)
        differences.extend(
            _json_differences(
                old_value,
                candidate_value,
                surface="schema",
                relative_path=relative,
            )
        )
        if relative == "schema_manifest.json":
            old_normalized = copy.deepcopy(old_value)
            candidate_normalized = copy.deepcopy(candidate_value)
            for manifest in (old_normalized, candidate_normalized):
                manifest.pop("manifest_hash", None)
                for entry in manifest.get("schemas", []):
                    if entry.get("file") in _LINEAGE_SCHEMA_FILES:
                        entry.pop("bytes", None)
                        entry.pop("sha256", None)
            if old_normalized != candidate_normalized:
                raise RefreshVerificationError("schema manifest changed outside lineage schemas")
            continue
        if relative not in _LINEAGE_SCHEMA_FILES:
            raise RefreshVerificationError(
                f"generated schema changed outside evidence-lineage surface: {relative}"
            )
        if _normalize_lineage_schema(relative, old_value) != _normalize_lineage_schema(
            relative, candidate_value
        ):
            raise RefreshVerificationError(
                f"generated schema has a non-lineage semantic change: {relative}"
            )
    if require_lineage_upgrade:
        _validate_candidate_lineage_schemas(candidate_root)
    return tuple(differences)


def _require_replay_identity(
    candidate: Sequence[FileDigest], replay: Sequence[FileDigest], *, label: str
) -> None:
    if tuple(candidate) != tuple(replay):
        candidate_map = _inventory_map(candidate)
        replay_map = _inventory_map(replay)
        changed = sorted(
            path
            for path in set(candidate_map) | set(replay_map)
            if candidate_map.get(path) != replay_map.get(path)
        )
        raise RefreshVerificationError(
            f"independent candidate {label} regeneration is not byte-identical: {changed[:12]}"
        )


def verify_benchmark_schema_refresh(
    *,
    old_benchmark_root: Path,
    candidate_benchmark_root: Path,
    old_schema_root: Path,
    candidate_schema_root: Path,
    candidate_source_root: Path | None = None,
    candidate_benchmark_replay_root: Path | None = None,
    candidate_schema_replay_root: Path | None = None,
    candidate_projection_rule: str = EXACT_LINEAGE_PROJECTION_RULE,
) -> BenchmarkRefreshReceipt:
    """Verify one complete benchmark/schema refresh and return its proof.

    Replay roots are optional at the API level so callers can perform a bounded
    diagnostic first.  Supplying one requires supplying both, and a publication
    or recovery checkpoint should accept only a receipt whose
    ``candidate_replay_checked`` field is true.
    """

    old_benchmark = _safe_root(old_benchmark_root, label="old benchmark")
    candidate_benchmark = _safe_root(candidate_benchmark_root, label="candidate benchmark")
    old_schema = _safe_root(old_schema_root, label="old schema")
    candidate_schema = _safe_root(candidate_schema_root, label="candidate schema")
    old_benchmark_inventory = _inventory(old_benchmark, label="old benchmark")
    candidate_benchmark_inventory = _inventory(candidate_benchmark, label="candidate benchmark")
    old_schema_inventory = _inventory(old_schema, label="old schema")
    candidate_schema_inventory = _inventory(candidate_schema, label="candidate schema")

    benchmark_paths = _require_same_paths(
        old_benchmark_inventory, candidate_benchmark_inventory, label="benchmark"
    )
    _require_same_paths(old_schema_inventory, candidate_schema_inventory, label="schema")
    if len(benchmark_paths) != EXPECTED_BENCHMARK_FILE_COUNT:
        raise RefreshVerificationError(
            f"benchmark file count changed: expected {EXPECTED_BENCHMARK_FILE_COUNT}, "
            f"observed {len(benchmark_paths)}"
        )
    if len(old_schema_inventory) != EXPECTED_SCHEMA_FILE_COUNT:
        raise RefreshVerificationError(
            f"schema file count changed: expected {EXPECTED_SCHEMA_FILE_COUNT}, "
            f"observed {len(old_schema_inventory)}"
        )

    old_namespace_counts = _validate_generated_inventory(
        old_benchmark, old_benchmark_inventory
    )
    candidate_namespace_counts = _validate_generated_inventory(
        candidate_benchmark, candidate_benchmark_inventory
    )
    if old_namespace_counts != candidate_namespace_counts:
        raise RefreshVerificationError("generated namespace inventory changed")

    old_map = _inventory_map(old_benchmark_inventory)
    candidate_map = _inventory_map(candidate_benchmark_inventory)
    missing_exact = sorted(_EXACT_SCIENTIFIC_FILES - set(benchmark_paths))
    if missing_exact:
        raise RefreshVerificationError(
            f"registered exact scientific files are absent: {missing_exact}"
        )
    for relative in sorted(_EXACT_SCIENTIFIC_FILES):
        if old_map[relative] != candidate_map[relative]:
            raise RefreshVerificationError(
                f"registered scientific artifact is not byte-identical: {relative}"
            )

    benchmark_differences: list[JsonPointerDifference] = []
    for relative in benchmark_paths:
        if old_map[relative].sha256 == candidate_map[relative].sha256:
            continue
        if relative in _EXACT_SCIENTIFIC_FILES:
            raise AssertionError("exact-file difference escaped the byte gate")
        if not relative.endswith(".json"):
            raise RefreshVerificationError(f"non-JSON benchmark artifact changed: {relative}")
        old_value = _load_json(old_benchmark / relative)
        candidate_value = _load_json(candidate_benchmark / relative)
        benchmark_differences.extend(
            _json_differences(
                old_value,
                candidate_value,
                surface="benchmark",
                relative_path=relative,
            )
        )
        old_normalized = _normalize_benchmark_document(relative, old_value)
        candidate_normalized = _normalize_benchmark_document(relative, candidate_value)
        if old_normalized != candidate_normalized:
            _raise_semantic_difference(
                label="benchmark artifact",
                relative_path=relative,
                old=old_normalized,
                candidate=candidate_normalized,
            )

    summary = _validate_candidate_tree(
        candidate_benchmark,
        candidate_benchmark_inventory,
        candidate_projection_rule=candidate_projection_rule,
        candidate_source_root=candidate_source_root,
    )
    if candidate_projection_rule == EXACT_LINEAGE_PROJECTION_RULE:
        _validate_adjudication_schema_upgrade(
            _load_json(candidate_benchmark / "scorer_only/review/adjudication.schema.json")
        )
    schema_differences = _compare_schema_roots(
        old_schema,
        candidate_schema,
        old_schema_inventory,
        candidate_schema_inventory,
        require_lineage_upgrade=candidate_projection_rule == EXACT_LINEAGE_PROJECTION_RULE,
    )

    replay_checked = (candidate_benchmark_replay_root is not None) or (
        candidate_schema_replay_root is not None
    )
    if (candidate_benchmark_replay_root is None) != (candidate_schema_replay_root is None):
        raise RefreshVerificationError(
            "benchmark and schema replay roots must be supplied together"
        )
    if candidate_benchmark_replay_root is not None and candidate_schema_replay_root is not None:
        benchmark_replay = _inventory(
            candidate_benchmark_replay_root, label="candidate benchmark replay"
        )
        schema_replay = _inventory(candidate_schema_replay_root, label="candidate schema replay")
        _require_replay_identity(
            candidate_benchmark_inventory, benchmark_replay, label="benchmark"
        )
        _require_replay_identity(candidate_schema_inventory, schema_replay, label="schema")

    all_differences = tuple(
        sorted(
            (*benchmark_differences, *schema_differences),
            key=lambda item: (
                item.surface,
                item.relative_path,
                item.pointer,
                item.operation,
            ),
        )
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synthetic_benchmark_schema_refresh_equivalence_receipt",
        "policy_revision": POLICY_REVISION,
        "status": "verified",
        "candidate_projection_rule": candidate_projection_rule,
        "benchmark_file_count": len(candidate_benchmark_inventory),
        "generated_namespace_counts": candidate_namespace_counts,
        "schema_file_count": len(candidate_schema_inventory),
        "old_benchmark_inventory": tuple(old_benchmark_inventory),
        "candidate_benchmark_inventory": tuple(candidate_benchmark_inventory),
        "old_schema_inventory": tuple(old_schema_inventory),
        "candidate_schema_inventory": tuple(candidate_schema_inventory),
        "old_benchmark_tree_sha256": _inventory_hash(old_benchmark_inventory),
        "candidate_benchmark_tree_sha256": _inventory_hash(candidate_benchmark_inventory),
        "old_schema_tree_sha256": _inventory_hash(old_schema_inventory),
        "candidate_schema_tree_sha256": _inventory_hash(candidate_schema_inventory),
        "candidate_replay_checked": replay_checked,
        "exact_scientific_file_count": len(_EXACT_SCIENTIFIC_FILES),
        "neutral_artifact_count": summary.neutral_artifact_count,
        "model_visible_artifact_file_count": summary.model_visible_artifact_file_count,
        "query_reveal_count": summary.query_reveal_count,
        "review_world_count": summary.review_world_count,
        "review_projection_count": summary.review_projection_count,
        "review_item_count": summary.review_item_count,
        "differences": tuple(all_differences),
    }
    receipt_hash = _canonical_sha256(
        json.loads(_canonical_bytes(payload)), strip_content_hashes=False
    )
    receipt = BenchmarkRefreshReceipt(
        **payload,
        receipt_sha256=receipt_hash,
    )
    receipt.verify_self_hash()
    return receipt


def write_refresh_receipt(receipt: BenchmarkRefreshReceipt, destination: Path) -> None:
    """Atomically write one canonical, human-readable receipt without overwriting."""

    receipt.verify_self_hash()
    destination = Path(destination)
    if destination.is_symlink():
        raise RefreshVerificationError("refresh receipt destination cannot be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            receipt.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.read_bytes() != rendered:
            raise RefreshVerificationError("refusing to overwrite a different refresh receipt")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
