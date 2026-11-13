#!/usr/bin/env python3
"""Build the allowlist-only, copyright-safe public artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from story_projection_onto.case_study_analysis import CaseStudyNarrativeAnalysisReceipt
from story_projection_onto.novel_case import RestrictedNovelIndexManifest
from story_projection_onto.public_release import (
    NarrativeReleaseLineage,
    Phase7ReleaseLineage,
    ProtectedProseCanaryManifest,
    PublicEntry,
    PublicReleaseError,
    VisualReleaseLineage,
    build_public_bundle,
    load_protected_prose_canaries,
    load_public_entries,
    scan_public_bytes,
    scan_public_entries,
)
from story_projection_onto.report_ingestion import verify_ingestion_from_files
from story_projection_onto.report_visual_inspection import verify_visual_inspection
from story_projection_onto.reporting import (
    ReportingError,
    ReportStatus,
    ResultManifest,
    canonical_sha256,
    load_result_manifest,
    verify_report_build,
)

_PDF_TEXT_SCAN_LIMIT_BYTES = 128 * 1024 * 1024
_SHA256_LENGTH = 64

_PHASE7_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "source_registry_sha256",
        "compiler_configuration_sha256",
        "build_token",
        "study_status",
        "compilation_manifest_relative_path",
        "compilation_manifest_file_sha256",
        "result_manifest_relative_path",
        "result_manifest_file_sha256",
        "report_pdf_relative_path",
        "report_pdf_file_sha256",
        "public_bundle_input_relative_path",
        "public_bundle_input_file_sha256",
        "manifest_sha256",
    }
)
_PHASE7_COMPILATION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "source_registry_sha256",
        "compiler_configuration_sha256",
        "study_status",
        "build_token",
        "immutable_outputs",
        "aliases",
        "manual_numeric_transcription_permitted",
        "manifest_sha256",
    }
)
_PHASE7_PUBLIC_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_status",
        "status_reason",
        "source_registry_sha256",
        "entries",
        "intentional_exclusions",
        "manifest_sha256",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/public_bundle_inputs.json"),
    )
    parser.add_argument(
        "--report-manifest",
        type=Path,
        default=Path("reports/results_manifest.json"),
    )
    parser.add_argument(
        "--phase7-current",
        type=Path,
        default=Path("reports/phase7_current.json"),
        help="Self-hashed current pointer emitted by the production Phase-7 compiler.",
    )
    parser.add_argument(
        "--phase7-configuration",
        type=Path,
        default=Path("configs/study/phase7_compiler.json"),
        help="Frozen self-hashed production compiler configuration.",
    )
    parser.add_argument(
        "--phase7-registry",
        type=Path,
        default=Path("artifacts/restricted/phase7/source_registry.json"),
        help="Restricted immutable source registry used for full compiler reproduction.",
    )
    parser.add_argument(
        "--report-policy",
        type=Path,
        default=Path("configs/study/reporting.json"),
    )
    parser.add_argument(
        "--ingestion-manifest",
        type=Path,
        default=Path("configs/study/report_ingestion.json"),
    )
    parser.add_argument(
        "--ingestion-receipt",
        type=Path,
        default=Path("reports/report_ingestion_receipt.json"),
    )
    parser.add_argument(
        "--ingestion-source-snapshot",
        type=Path,
        default=None,
        help="Explicit hash-bound historical source overrides for report replay.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("artifacts/public/release/StoryProjectionOnto-public"),
    )
    parser.add_argument(
        "--visual-raster-manifest",
        type=Path,
        default=None,
        help="Raster manifest prepared by scripts/inspect_results_pdf.py.",
    )
    parser.add_argument(
        "--visual-inspection-receipt",
        type=Path,
        default=None,
        help="Explicit human/agent visual-review receipt for a complete release.",
    )
    parser.add_argument(
        "--restricted-root",
        type=Path,
        default=None,
        help="Explicit restricted root containing the protected-prose canary manifest.",
    )
    parser.add_argument(
        "--protected-prose-canary-manifest",
        type=Path,
        default=None,
        help="Restricted self-hashed exact-byte canary manifest; required for case output.",
    )
    parser.add_argument(
        "--narrative-analysis-receipt",
        type=Path,
        default=None,
        help="Restricted successful Phase-6 narrative-analysis receipt.",
    )
    parser.add_argument(
        "--restricted-index-manifest",
        type=Path,
        default=None,
        help="Restricted query-blind novel-index manifest bound by the analysis receipt.",
    )
    parser.add_argument(
        "--protected-corpus",
        type=Path,
        default=None,
        help="Exact protected corpus used by the query-blind index and canary manifest.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_source_file(source_root: Path, path: Path, *, label: str) -> Path:
    """Resolve one explicit regular source file without following any symlink."""

    try:
        root = source_root.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError("public source root is unavailable") from error
    candidate = path if path.is_absolute() else source_root / path
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise PublicReleaseError(f"{label} is outside the public source root") from error
    cursor = root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise PublicReleaseError(f"{label} cannot traverse a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"missing {label}") from error
    if resolved != lexical or not resolved.is_file() or not resolved.is_relative_to(root):
        raise PublicReleaseError(f"{label} is not one stable source file")
    return resolved


def _safe_phase7_output(report_root: Path, relative_path: object, *, label: str) -> Path:
    if not isinstance(relative_path, str) or "\\" in relative_path:
        raise PublicReleaseError(f"invalid {label} path")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PublicReleaseError(f"unsafe {label} path")
    candidate = report_root.joinpath(*relative.parts)
    root = report_root.resolve(strict=True)
    lexical = Path(os.path.abspath(candidate))
    try:
        path_relative = lexical.relative_to(root)
    except ValueError as error:
        raise PublicReleaseError(f"{label} escapes the Phase-7 output root") from error
    cursor = root
    for component in path_relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise PublicReleaseError(f"{label} cannot traverse a symbolic link")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"missing {label}") from error
    if resolved != lexical or not resolved.is_file() or not resolved.is_relative_to(root):
        raise PublicReleaseError(f"{label} is not one stable Phase-7 output")
    return resolved


def _load_self_hashed_json(
    path: Path,
    *,
    hash_field: str,
    expected_fields: frozenset[str] | None,
    label: str,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise PublicReleaseError(f"{label} must be UTF-8 without BOM")
        payload = json.loads(raw.decode("utf-8"))
    except PublicReleaseError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicReleaseError(f"invalid {label}") from error
    if not isinstance(payload, dict):
        raise PublicReleaseError(f"{label} must be one JSON object")
    if expected_fields is not None and frozenset(payload) != expected_fields:
        raise PublicReleaseError(f"{label} field inventory differs from compiler output")
    supplied_hash = payload.get(hash_field)
    immutable = {key: value for key, value in payload.items() if key != hash_field}
    if not _is_sha256(supplied_hash) or supplied_hash != canonical_sha256(immutable):
        raise PublicReleaseError(f"{label} canonical self-hash mismatch")
    return payload


def _require_identical_file(left: Path, right: Path, *, label: str) -> None:
    if _file_sha256(left) != _file_sha256(right):
        raise PublicReleaseError(f"{label} alias differs from its immutable Phase-7 output")


def _expected_phase7_aliases(token: str) -> dict[str, str]:
    return {
        "REPRODUCIBILITY.md": f"REPRODUCIBILITY.{token}.md",
        "RESULTS_REPORT.md": f"RESULTS_REPORT.{token}.md",
        "RESULTS_REPORT.pdf": f"RESULTS_REPORT.{token}.pdf",
        "public_bundle_inputs.json": f"public_bundle_inputs.{token}.json",
        "result_figure_manifest.json": f"result_figure_manifest.{token}.json",
        "results_manifest.json": f"results_manifest.{token}.json",
    }


def _verify_compiler_generated_entries(
    *,
    source_root: Path,
    report_root: Path,
    public_manifest: Mapping[str, Any],
    immutable_outputs: Mapping[str, str],
    public_manifest_relative_path: str,
) -> tuple[PublicEntry, ...]:
    try:
        output_prefix = report_root.relative_to(source_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise PublicReleaseError("Phase-7 output root escapes the public source root") from error
    try:
        entries = tuple(PublicEntry(**item) for item in public_manifest["entries"])
    except (KeyError, TypeError, ValueError) as error:
        raise PublicReleaseError("invalid compiler public-entry inventory") from error
    targets = [item.bundle_relative_path for item in entries]
    if targets != sorted(targets) or len(targets) != len(set(targets)):
        raise PublicReleaseError("compiler public-entry inventory is not sorted and unique")
    by_target = {item.bundle_relative_path: item for item in entries}
    generated_outputs = {
        relative: digest
        for relative, digest in immutable_outputs.items()
        if not relative.startswith("report_ingestion_manifest.")
        and relative != public_manifest_relative_path
    }
    for relative, digest in generated_outputs.items():
        bundle_relative = (
            f"reports/{Path(relative).name}"
            if "/" not in relative
            else f"reports/{relative}"
        )
        expected_source = f"{output_prefix}/{relative}" if output_prefix != "." else relative
        entry = by_target.get(bundle_relative)
        if entry != PublicEntry(
            source_relative_path=expected_source,
            bundle_relative_path=bundle_relative,
            sha256=digest,
            release_class="public",
        ):
            raise PublicReleaseError(
                f"compiler public allowlist omits or changes output: {relative}"
            )
    return entries


def _verify_phase7_release_lineage(
    *,
    source_root: Path,
    current_pointer_path: Path,
    configuration_path: Path,
    report_manifest_path: Path,
    public_manifest_path: Path,
) -> Phase7ReleaseLineage:
    """Authenticate the exact compiler build before any release bytes are copied."""

    source_root = source_root.resolve(strict=True)
    current_path = _safe_source_file(
        source_root, current_pointer_path, label="Phase-7 current pointer"
    )
    report_root = current_path.parent
    pointer = _load_self_hashed_json(
        current_path,
        hash_field="manifest_sha256",
        expected_fields=_PHASE7_POINTER_FIELDS,
        label="Phase-7 current pointer",
    )
    if pointer["schema_version"] != "1.0.0" or pointer["kind"] != "phase7_current_pointer":
        raise PublicReleaseError("unsupported Phase-7 current pointer")
    token = pointer["build_token"]
    if (
        not isinstance(token, str)
        or len(token) != 16
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise PublicReleaseError("invalid Phase-7 build token")
    expected_paths = {
        "compilation_manifest_relative_path": f"phase7_compilation.{token}.json",
        "result_manifest_relative_path": f"results_manifest.{token}.json",
        "report_pdf_relative_path": f"RESULTS_REPORT.{token}.pdf",
        "public_bundle_input_relative_path": f"public_bundle_inputs.{token}.json",
    }
    if any(pointer[field] != expected for field, expected in expected_paths.items()):
        raise PublicReleaseError("Phase-7 current pointer path inventory changed")
    for field in (
        "source_registry_sha256",
        "compiler_configuration_sha256",
        "compilation_manifest_file_sha256",
        "result_manifest_file_sha256",
        "report_pdf_file_sha256",
        "public_bundle_input_file_sha256",
        "manifest_sha256",
    ):
        if not _is_sha256(pointer[field]):
            raise PublicReleaseError(f"invalid Phase-7 pointer digest: {field}")

    configuration_file = _safe_source_file(
        source_root, configuration_path, label="Phase-7 compiler configuration"
    )
    configuration = _load_self_hashed_json(
        configuration_file,
        hash_field="configuration_sha256",
        expected_fields=None,
        label="Phase-7 compiler configuration",
    )
    if configuration["configuration_sha256"] != pointer["compiler_configuration_sha256"]:
        raise PublicReleaseError("Phase-7 pointer names another compiler configuration")

    compilation_path = _safe_phase7_output(
        report_root,
        pointer["compilation_manifest_relative_path"],
        label="Phase-7 compilation manifest",
    )
    if _file_sha256(compilation_path) != pointer["compilation_manifest_file_sha256"]:
        raise PublicReleaseError("Phase-7 compilation manifest physical hash mismatch")
    compilation = _load_self_hashed_json(
        compilation_path,
        hash_field="manifest_sha256",
        expected_fields=_PHASE7_COMPILATION_FIELDS,
        label="Phase-7 compilation manifest",
    )
    common_fields = (
        "source_registry_sha256",
        "compiler_configuration_sha256",
        "study_status",
        "build_token",
    )
    if any(compilation[field] != pointer[field] for field in common_fields):
        raise PublicReleaseError("Phase-7 pointer and compilation manifest differ")
    if (
        compilation["schema_version"] != "1.0.0"
        or compilation["kind"] != "phase7_compilation_manifest"
        or compilation["manual_numeric_transcription_permitted"] is not False
        or compilation["aliases"] != _expected_phase7_aliases(token)
    ):
        raise PublicReleaseError("unsupported Phase-7 compilation manifest")

    output_rows = compilation["immutable_outputs"]
    if not isinstance(output_rows, list) or not output_rows:
        raise PublicReleaseError("Phase-7 immutable output inventory is empty")
    immutable_outputs: dict[str, str] = {}
    observed_paths: list[str] = []
    for row in output_rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "sha256"}:
            raise PublicReleaseError("invalid Phase-7 immutable output row")
        relative = row["relative_path"]
        digest = row["sha256"]
        if not isinstance(relative, str) or relative in immutable_outputs or not _is_sha256(digest):
            raise PublicReleaseError("invalid or duplicate Phase-7 immutable output")
        output_path = _safe_phase7_output(
            report_root, relative, label="Phase-7 immutable output"
        )
        if _file_sha256(output_path) != digest:
            raise PublicReleaseError(f"Phase-7 immutable output hash mismatch: {relative}")
        immutable_outputs[relative] = digest
        observed_paths.append(relative)
    if observed_paths != sorted(observed_paths):
        raise PublicReleaseError("Phase-7 immutable output inventory is not sorted")
    missing_alias_targets = sorted(
        set(_expected_phase7_aliases(token).values()) - set(immutable_outputs)
    )
    if missing_alias_targets:
        raise PublicReleaseError(
            "Phase-7 immutable output inventory omits an alias target: "
            + missing_alias_targets[0]
        )

    pointer_hash_fields = {
        pointer["result_manifest_relative_path"]: pointer["result_manifest_file_sha256"],
        pointer["report_pdf_relative_path"]: pointer["report_pdf_file_sha256"],
        pointer["public_bundle_input_relative_path"]: pointer[
            "public_bundle_input_file_sha256"
        ],
    }
    if any(immutable_outputs.get(path) != digest for path, digest in pointer_hash_fields.items()):
        raise PublicReleaseError("Phase-7 pointer does not bind its immutable outputs")

    token_pointer = _safe_phase7_output(
        report_root, f"phase7_current.{token}.json", label="immutable Phase-7 pointer"
    )
    _require_identical_file(current_path, token_pointer, label="Phase-7 current pointer")
    for alias, target in _expected_phase7_aliases(token).items():
        alias_path = _safe_phase7_output(report_root, alias, label="Phase-7 report alias")
        target_path = _safe_phase7_output(
            report_root, target, label="immutable Phase-7 alias target"
        )
        _require_identical_file(alias_path, target_path, label=f"Phase-7 {alias}")

    report_manifest = _safe_source_file(
        source_root, report_manifest_path, label="result-manifest alias"
    )
    public_manifest_file = _safe_source_file(
        source_root, public_manifest_path, label="public-bundle input alias"
    )
    expected_report_alias = _safe_phase7_output(
        report_root, "results_manifest.json", label="Phase-7 result-manifest alias"
    )
    expected_public_alias = _safe_phase7_output(
        report_root, "public_bundle_inputs.json", label="Phase-7 public-input alias"
    )
    _require_identical_file(report_manifest, expected_report_alias, label="result manifest")
    _require_identical_file(public_manifest_file, expected_public_alias, label="public input")

    public_manifest = _load_self_hashed_json(
        public_manifest_file,
        hash_field="manifest_sha256",
        expected_fields=_PHASE7_PUBLIC_INPUT_FIELDS,
        label="Phase-7 public-bundle input manifest",
    )
    if (
        public_manifest["schema_version"] != "1.0.0"
        or public_manifest["source_registry_sha256"] != pointer["source_registry_sha256"]
        or public_manifest["bundle_status"] != pointer["study_status"]
    ):
        raise PublicReleaseError("Phase-7 public input differs from its compiler build")
    entries = _verify_compiler_generated_entries(
        source_root=source_root,
        report_root=report_root,
        public_manifest=public_manifest,
        immutable_outputs=immutable_outputs,
        public_manifest_relative_path=pointer["public_bundle_input_relative_path"],
    )
    return Phase7ReleaseLineage(
        build_token=token,
        source_registry_sha256=pointer["source_registry_sha256"],
        compiler_configuration_sha256=pointer["compiler_configuration_sha256"],
        current_pointer_file_sha256=_file_sha256(current_path),
        current_pointer_manifest_sha256=pointer["manifest_sha256"],
        compilation_manifest_file_sha256=_file_sha256(compilation_path),
        compilation_manifest_sha256=compilation["manifest_sha256"],
        immutable_output_inventory_sha256=canonical_sha256(output_rows),
        result_manifest_file_sha256=pointer["result_manifest_file_sha256"],
        report_pdf_file_sha256=pointer["report_pdf_file_sha256"],
        public_bundle_input_file_sha256=pointer["public_bundle_input_file_sha256"],
        public_entry_inventory_sha256=canonical_sha256(
            [
                {
                    "source_relative_path": item.source_relative_path,
                    "bundle_relative_path": item.bundle_relative_path,
                    "sha256": item.sha256,
                    "release_class": item.release_class,
                }
                for item in entries
            ]
        ),
    )


def _reproduce_phase7_results(
    *,
    source_root: Path,
    configuration_path: Path,
    registry_path: Path,
    output_root: Path,
) -> Any:
    """Import the analysis-heavy compiler only for the production replay gate."""

    from story_projection_onto.phase7_compiler import (
        Phase7CompilationError,
        verify_phase7_results,
    )

    try:
        return verify_phase7_results(
            source_root=source_root,
            configuration_path=configuration_path,
            registry_path=registry_path,
            output_root=output_root,
        )
    except Phase7CompilationError as error:
        raise PublicReleaseError(
            "public release requires byte-reproduction of the exact Phase-7 build"
        ) from error


def _safe_restricted_file(root: Path, path: Path, *, label: str) -> Path:
    """Resolve one explicitly named restricted input without following symlinks."""

    if not root.is_absolute():
        raise PublicReleaseError("restricted root must be an explicit absolute path")
    lexical_root = Path(os.path.abspath(root))
    lexical_path = Path(os.path.abspath(path if path.is_absolute() else root / path))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise PublicReleaseError(f"{label} is outside the restricted root") from error
    cursor = lexical_root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise PublicReleaseError(f"{label} cannot traverse a symbolic link")
    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved_path = lexical_path.resolve(strict=True)
    except OSError as error:
        raise PublicReleaseError(f"{label} is unavailable") from error
    if (
        resolved_root != lexical_root
        or not resolved_root.is_dir()
        or resolved_path != lexical_path
        or not resolved_path.is_file()
        or not resolved_path.is_relative_to(resolved_root)
    ):
        raise PublicReleaseError(f"{label} is not one stable restricted file")
    return resolved_path


def _bundle_status(manifest_path: Path) -> ReportStatus:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ReportStatus(payload["bundle_status"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise PublicReleaseError(
            "public bundle input manifest lacks a valid explicit bundle_status"
        ) from error


def _narrative_release_required(
    result_manifest: ResultManifest,
    *,
    bundle_status: ReportStatus,
    receipt_supplied: bool,
) -> bool:
    """Use typed scientific status, never public filenames, to identify narrative output."""

    if bundle_status is not result_manifest.study_status:
        raise PublicReleaseError(
            "public bundle status differs from the verified result-manifest status"
        )
    narrative_tables = [table for table in result_manifest.tables if table.table_id == "novel_case"]
    if len(narrative_tables) > 1:
        raise PublicReleaseError("result manifest contains duplicate narrative tables")
    phase_six = [phase for phase in result_manifest.phases if phase.phase_id == "phase_6"]
    if len(phase_six) != 1:
        raise PublicReleaseError("result manifest lacks exactly one Phase-6 status")
    return bool(
        receipt_supplied
        or narrative_tables
        or phase_six[0].status is ReportStatus.COMPLETE
        or result_manifest.study_status is ReportStatus.COMPLETE
        or bundle_status is ReportStatus.COMPLETE
    )


def _load_restricted_record(
    model: type[Any],
    root: Path,
    path: Path,
    *,
    label: str,
) -> Any:
    source = _safe_restricted_file(root, path, label=label)
    try:
        if source.stat().st_size > 4 * 1024 * 1024:
            raise PublicReleaseError(f"{label} exceeds the 4 MiB metadata limit")
        return model.model_validate_json(source.read_bytes())
    except PublicReleaseError:
        raise
    except Exception as error:
        raise PublicReleaseError(f"invalid {label}: {error}") from error


def _verify_narrative_release_lineage(
    *,
    source_root: Path,
    result_manifest: ResultManifest,
    restricted_root: Path,
    narrative_analysis_receipt_path: Path,
    restricted_index_manifest_path: Path,
    protected_corpus_path: Path,
    protected_canary_manifest_path: Path,
    canary_manifest: ProtectedProseCanaryManifest,
    canaries: tuple[bytes, ...],
) -> NarrativeReleaseLineage:
    """Revalidate the complete corpus -> index -> canary -> public-table chain."""

    receipt_source = _safe_restricted_file(
        restricted_root,
        narrative_analysis_receipt_path,
        label="narrative-analysis receipt",
    )
    receipt = _load_restricted_record(
        CaseStudyNarrativeAnalysisReceipt,
        restricted_root,
        receipt_source,
        label="narrative-analysis receipt",
    )
    index_source = _safe_restricted_file(
        restricted_root,
        restricted_index_manifest_path,
        label="restricted novel-index manifest",
    )
    index_manifest = _load_restricted_record(
        RestrictedNovelIndexManifest,
        restricted_root,
        index_source,
        label="restricted novel-index manifest",
    )
    canary_source = _safe_restricted_file(
        restricted_root,
        protected_canary_manifest_path,
        label="protected prose canary manifest",
    )
    corpus = _safe_restricted_file(
        restricted_root,
        protected_corpus_path,
        label="protected narrative corpus",
    )
    corpus_payload = corpus.read_bytes()
    corpus_sha256 = hashlib.sha256(corpus_payload).hexdigest()
    if receipt.publication_status != "public_scan_passed":
        raise PublicReleaseError("narrative analysis has not passed its publication scan")
    if any(
        (
            receipt.restricted_index_manifest_hash != index_manifest.content_hash,
            receipt.corpus_source_sha256 != index_manifest.source_sha256,
            corpus_sha256 != index_manifest.source_sha256,
            canary_manifest.corpus_hash != index_manifest.source_sha256,
            receipt.protected_canary_manifest_hash != canary_manifest.content_hash,
        )
    ):
        raise PublicReleaseError("narrative receipt, index, corpus, or canary lineage differs")
    if not canaries or any(canary not in corpus_payload for canary in canaries):
        raise PublicReleaseError("protected-prose canaries are not present in the indexed corpus")

    narrative_tables = [table for table in result_manifest.tables if table.table_id == "novel_case"]
    if len(narrative_tables) != 1:
        raise PublicReleaseError(
            "narrative release requires exactly one typed novel_case report table"
        )
    narrative_table = narrative_tables[0]
    if narrative_table.status is not ReportStatus.COMPLETE:
        raise PublicReleaseError("narrative release requires a complete novel_case table")
    public_table_path = source_root / "reports" / narrative_table.relative_path
    if any(
        (
            receipt.public_table_file_sha256 != narrative_table.sha256,
            receipt.restricted_table_file_sha256 != narrative_table.sha256,
            _file_sha256(public_table_path) != narrative_table.sha256,
        )
    ):
        raise PublicReleaseError("narrative-analysis receipt does not bind the report table")
    expected_scan_receipt_hash = canonical_sha256(
        {
            "canary_manifest_hash": canary_manifest.content_hash,
            "corpus_source_sha256": index_manifest.source_sha256,
            "payload_sha256": narrative_table.sha256,
            "public_relative_path": narrative_table.relative_path,
            "scanner": "story_projection_onto.public_release.scan_public_bytes",
        }
    )
    if receipt.release_scan_receipt_hash != expected_scan_receipt_hash:
        raise PublicReleaseError("narrative table release-scan receipt is stale")
    assert receipt.protected_canary_manifest_hash is not None
    assert receipt.release_scan_receipt_hash is not None
    assert receipt.public_table_file_sha256 is not None
    return NarrativeReleaseLineage(
        analysis_receipt_file_sha256=_file_sha256(receipt_source),
        analysis_receipt_sha256=receipt.content_hash,
        index_manifest_file_sha256=_file_sha256(index_source),
        index_manifest_sha256=index_manifest.content_hash,
        protected_canary_manifest_file_sha256=_file_sha256(canary_source),
        protected_canary_manifest_sha256=receipt.protected_canary_manifest_hash,
        corpus_sha256=corpus_sha256,
        release_scan_receipt_sha256=receipt.release_scan_receipt_hash,
        public_table_file_sha256=receipt.public_table_file_sha256,
    )


def _extract_pdf_text(path: Path) -> bytes:
    extractor = shutil.which("pdftotext")
    if extractor is None:
        raise PublicReleaseError(
            "public PDF scanning requires the already-registered pdftotext extractor"
        )
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(
                [extractor, str(path), "-"],
                check=False,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            output.seek(0)
            payload = output.read(_PDF_TEXT_SCAN_LIMIT_BYTES + 1)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PublicReleaseError("public PDF text extraction failed") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise PublicReleaseError(f"public PDF text extraction failed: {detail}")
    if len(payload) > _PDF_TEXT_SCAN_LIMIT_BYTES:
        raise PublicReleaseError("extracted public PDF text exceeds the 128 MiB scan limit")
    return payload


def _scan_allowlisted_pdf_text(
    source_root: Path,
    entries: tuple[PublicEntry, ...],
    *,
    forbidden_canaries: tuple[bytes, ...],
) -> None:
    """Scan decoded PDF text in addition to the encoded PDF bytes."""

    resolved_root = source_root.resolve(strict=True)
    for entry in entries:
        if Path(entry.bundle_relative_path).suffix.lower() != ".pdf":
            continue
        source = (resolved_root / entry.source_relative_path).resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(resolved_root):
            raise PublicReleaseError("allowlisted PDF escapes the public source root")
        payload = _extract_pdf_text(source)
        scan_public_bytes(
            payload,
            relative_path=entry.bundle_relative_path,
            forbidden_canaries=forbidden_canaries,
        )


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    configuration_path = _safe_source_file(
        source_root,
        args.phase7_configuration,
        label="Phase-7 compiler configuration",
    )
    registry_path = _safe_source_file(
        source_root,
        args.phase7_registry,
        label="Phase-7 source registry",
    )
    current_pointer_path = _safe_source_file(
        source_root,
        args.phase7_current,
        label="Phase-7 current pointer",
    )
    reproduced = _reproduce_phase7_results(
        source_root=source_root,
        configuration_path=configuration_path,
        registry_path=registry_path,
        output_root=current_pointer_path.parent,
    )
    phase7_lineage = _verify_phase7_release_lineage(
        source_root=source_root,
        current_pointer_path=current_pointer_path,
        configuration_path=configuration_path,
        report_manifest_path=args.report_manifest,
        public_manifest_path=args.manifest,
    )
    if (
        reproduced.registry_sha256 != phase7_lineage.source_registry_sha256
        or reproduced.configuration_sha256
        != phase7_lineage.compiler_configuration_sha256
        or reproduced.build_token != phase7_lineage.build_token
        or reproduced.current_pointer_sha256
        != phase7_lineage.current_pointer_manifest_sha256
    ):
        raise PublicReleaseError("reproduced Phase-7 build differs from its release lineage")
    canary_arguments = (
        args.restricted_root is not None,
        args.protected_prose_canary_manifest is not None,
    )
    if any(canary_arguments) and not all(canary_arguments):
        raise PublicReleaseError(
            "restricted root and protected-prose canary manifest must be supplied together"
        )
    canaries: tuple[bytes, ...] = ()
    canary_manifest_hash = None
    canary_manifest = None
    canary_path = None
    if all(canary_arguments):
        assert args.restricted_root is not None
        assert args.protected_prose_canary_manifest is not None
        canary_path = args.protected_prose_canary_manifest
        if not canary_path.is_absolute():
            canary_path = args.restricted_root / canary_path
        canary_manifest, canaries = load_protected_prose_canaries(
            args.restricted_root,
            canary_path,
        )
        canary_manifest_hash = canary_manifest.content_hash
    verify_ingestion_from_files(
        args.source_root / args.ingestion_manifest,
        args.source_root / args.ingestion_receipt,
        source_root=args.source_root,
        table_root=args.source_root / "reports",
        source_snapshot_manifest_path=(
            None
            if args.ingestion_source_snapshot is None
            else (
                args.ingestion_source_snapshot
                if args.ingestion_source_snapshot.is_absolute()
                else args.source_root / args.ingestion_source_snapshot
            )
        ),
    )
    verify_report_build(
        args.source_root / args.report_manifest,
        args.source_root / args.report_policy,
        args.source_root / "reports",
    )
    result_manifest = load_result_manifest(args.source_root / args.report_manifest)
    public_manifest_path = args.source_root / args.manifest
    entries = load_public_entries(public_manifest_path)
    status = _bundle_status(public_manifest_path)
    narrative_specific_arguments = (
        args.narrative_analysis_receipt is not None,
        args.restricted_index_manifest is not None,
        args.protected_corpus is not None,
    )
    narrative_required = _narrative_release_required(
        result_manifest,
        bundle_status=status,
        receipt_supplied=any(narrative_specific_arguments),
    )
    if narrative_required and not all((*canary_arguments, *narrative_specific_arguments)):
        raise PublicReleaseError(
            "a narrative-inclusive or final release requires its analysis receipt, "
            "restricted index, exact corpus, and protected-prose canary lineage"
        )
    if any(narrative_specific_arguments) and not all(narrative_specific_arguments):
        raise PublicReleaseError(
            "narrative analysis receipt, restricted index manifest, and protected corpus "
            "must be supplied together"
        )
    narrative_release_lineage = None
    if narrative_required:
        assert args.restricted_root is not None
        assert args.narrative_analysis_receipt is not None
        assert args.restricted_index_manifest is not None
        assert args.protected_corpus is not None
        assert canary_manifest is not None
        assert canary_path is not None
        narrative_release_lineage = _verify_narrative_release_lineage(
            source_root=args.source_root,
            result_manifest=result_manifest,
            restricted_root=args.restricted_root,
            narrative_analysis_receipt_path=args.narrative_analysis_receipt,
            restricted_index_manifest_path=args.restricted_index_manifest,
            protected_corpus_path=args.protected_corpus,
            protected_canary_manifest_path=canary_path,
            canary_manifest=canary_manifest,
            canaries=canaries,
        )
    visual_inputs = (
        args.visual_raster_manifest is not None,
        args.visual_inspection_receipt is not None,
    )
    if any(visual_inputs) and not all(visual_inputs):
        raise ReportingError(
            "visual raster manifest and inspection receipt must be supplied together"
        )
    if result_manifest.study_status is ReportStatus.COMPLETE and not all(visual_inputs):
        raise ReportingError(
            "a complete public release requires an accepted PDF visual-inspection receipt"
        )
    visual_release_lineage = None
    if all(visual_inputs):
        assert args.visual_raster_manifest is not None
        assert args.visual_inspection_receipt is not None
        raster_manifest_path = args.source_root / args.visual_raster_manifest
        inspection_receipt_path = args.source_root / args.visual_inspection_receipt
        raster, receipt = verify_visual_inspection(
            raster_manifest_path=raster_manifest_path,
            pdf_path=args.source_root / "reports/RESULTS_REPORT.pdf",
            receipt_path=inspection_receipt_path,
            require_accepted=result_manifest.study_status is ReportStatus.COMPLETE,
        )
        if receipt is not None and receipt.status == "accepted":
            visual_release_lineage = VisualReleaseLineage(
                raster_manifest_file_sha256=_file_sha256(raster_manifest_path),
                raster_manifest_sha256=raster.manifest_sha256,
                inspection_receipt_file_sha256=_file_sha256(inspection_receipt_path),
                inspection_receipt_sha256=receipt.receipt_sha256,
                source_pdf_sha256=raster.source_pdf_sha256,
            )
    scan_public_entries(
        args.source_root,
        entries,
        forbidden_canaries=canaries,
    )
    _scan_allowlisted_pdf_text(
        args.source_root,
        entries,
        forbidden_canaries=canaries,
    )
    build_public_bundle(
        args.source_root,
        args.source_root / args.manifest,
        args.bundle_root,
        forbidden_canaries=canaries,
        protected_canary_manifest_hash=canary_manifest_hash,
        phase7_lineage=phase7_lineage,
        visual_release_lineage=visual_release_lineage,
        narrative_release_lineage=narrative_release_lineage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
