"""Rasterize the results PDF and record a distinct visual-review attestation.

Automated checks can prove hashes, page counts, raster dimensions, and the presence of
registered section/figure text.  They cannot establish readability or visual quality.
An accepted receipt therefore exists only after a researcher or agent explicitly
records inspection of every deterministically selected representative page.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from story_projection_onto.reporting import ReportingError, build_document, canonical_sha256

VISUAL_INSPECTION_SCHEMA_VERSION = "1.0.0"
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=500)]
_CRITERIA = frozenset(
    {
        "clipping",
        "figure_presence",
        "graph_label_readability",
        "mathematical_notation",
        "page_breaks",
        "references_and_provenance",
    }
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RasterPage(FrozenModel):
    page_number: int = Field(ge=1)
    relative_path: RelativePath
    file_sha256: Sha256Digest
    width_pixels: int = Field(ge=100)
    height_pixels: int = Field(ge=100)
    size_bytes: int = Field(ge=1000)


class StructuralCheck(FrozenModel):
    check_id: Literal[
        "pdf_header",
        "page_count",
        "section_text_inventory",
        "figure_text_inventory",
        "table_text_inventory",
        "raster_dimensions",
    ]
    status: Literal["passed"]
    detail: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class ReportRasterManifest(FrozenModel):
    schema_version: Literal["1.0.0"] = VISUAL_INSPECTION_SCHEMA_VERSION
    kind: Literal["report_raster_manifest"] = "report_raster_manifest"
    source_result_manifest_sha256: Sha256Digest
    source_pdf_sha256: Sha256Digest
    source_pdf_name: Literal["RESULTS_REPORT.pdf"] = "RESULTS_REPORT.pdf"
    renderer: Literal["pdftoppm"] = "pdftoppm"
    renderer_version: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    raster_prepared_at_utc: AwareDatetime
    dpi: Literal[144] = 144
    page_count: int = Field(ge=1)
    representative_pages: tuple[int, ...]
    pages: tuple[RasterPage, ...]
    automated_checks: tuple[StructuralCheck, ...]
    visual_review_status: Literal["pending"] = "pending"
    visual_review_notice: Literal[
        "Automated structural checks are not a visual-quality acceptance."
    ] = "Automated structural checks are not a visual-quality acceptance."
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def inventory_is_exact(self) -> Self:
        if (
            self.raster_prepared_at_utc.utcoffset() is None
            or self.raster_prepared_at_utc.utcoffset().total_seconds() != 0
        ):
            raise ValueError("raster preparation timestamp must be normalized to UTC")
        numbers = tuple(item.page_number for item in self.pages)
        if numbers != tuple(range(1, self.page_count + 1)):
            raise ValueError("raster manifest must contain every PDF page in order")
        if self.representative_pages != tuple(sorted(set(self.representative_pages))):
            raise ValueError("representative page numbers must be sorted and unique")
        if not self.representative_pages or not set(self.representative_pages).issubset(numbers):
            raise ValueError("representative pages must resolve to rasterized PDF pages")
        check_ids = {item.check_id for item in self.automated_checks}
        if check_ids != {
            "pdf_header",
            "page_count",
            "section_text_inventory",
            "figure_text_inventory",
            "table_text_inventory",
            "raster_dimensions",
        }:
            raise ValueError("raster manifest lacks an automated structural check")
        return self


class VisualCriterion(FrozenModel):
    criterion: Literal[
        "clipping",
        "figure_presence",
        "graph_label_readability",
        "mathematical_notation",
        "page_breaks",
        "references_and_provenance",
    ]
    status: Literal["passed", "failed"]
    note: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class VisualInspectionReceipt(FrozenModel):
    schema_version: Literal["1.0.0"] = VISUAL_INSPECTION_SCHEMA_VERSION
    kind: Literal["report_visual_inspection"] = "report_visual_inspection"
    source_raster_manifest_sha256: Sha256Digest
    source_raster_manifest_file_sha256: Sha256Digest
    source_pdf_sha256: Sha256Digest
    reviewer_kind: Literal["researcher", "agent"]
    reviewed_at_utc: AwareDatetime
    inspected_pages: tuple[int, ...]
    criteria: tuple[VisualCriterion, ...]
    status: Literal["accepted", "rejected"]
    receipt_sha256: Sha256Digest

    @model_validator(mode="after")
    def criteria_agree_with_status(self) -> Self:
        names = [item.criterion for item in self.criteria]
        if len(names) != len(set(names)) or set(names) != _CRITERIA:
            raise ValueError("visual receipt requires every review criterion exactly once")
        if self.inspected_pages != tuple(sorted(set(self.inspected_pages))):
            raise ValueError("inspected pages must be sorted and unique")
        if (
            self.reviewed_at_utc.utcoffset() is None
            or self.reviewed_at_utc.utcoffset().total_seconds() != 0
        ):
            raise ValueError("visual review timestamp must be normalized to UTC")
        all_passed = all(item.status == "passed" for item in self.criteria)
        if (self.status == "accepted") != all_passed:
            raise ValueError("visual receipt status must agree with criterion findings")
        return self


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_symlink_ancestry(path: Path, *, label: str) -> None:
    current = Path(os.path.abspath(path))
    while True:
        if current.is_symlink():
            raise ReportingError(f"{label} has a symlinked ancestor: {path}")
        if current.parent == current:
            return
        current = current.parent


def _real_visual_root(path: Path, *, create: bool) -> Path:
    if ".." in path.parts:
        raise ReportingError(f"visual-inspection root contains parent traversal: {path}")
    lexical = Path(os.path.abspath(path))
    _assert_no_symlink_ancestry(lexical, label="visual-inspection root")
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
        _assert_no_symlink_ancestry(lexical, label="visual-inspection root")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ReportingError(f"missing visual-inspection root: {path}") from error
    if resolved != lexical or not resolved.is_dir():
        raise ReportingError("visual-inspection root must be one real directory")
    return resolved


def _safe_visual_descendant(
    root: Path,
    candidate: Path,
    *,
    label: str,
    require_file: bool,
) -> Path:
    """Reject lexical/resolved escape and every symlink below a visual root."""

    root = _real_visual_root(root, create=False)
    if ".." in candidate.parts:
        raise ReportingError(f"{label} contains parent traversal: {candidate}")
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ReportingError(
            f"{label} must remain inside the visual-inspection root: {candidate}"
        ) from error
    if relative == Path("."):
        raise ReportingError(f"{label} must name a file below the visual-inspection root")
    probe = root
    for component in relative.parts:
        probe /= component
        if probe.is_symlink():
            raise ReportingError(f"{label} has a symlinked ancestor: {candidate}")
    try:
        resolved = lexical.resolve(strict=require_file)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        state = "missing" if require_file else "unsafe"
        raise ReportingError(f"{state} {label}: {candidate}") from error
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise ReportingError(f"{label} must be one regular file: {candidate}")
    return lexical


def _safe_visual_file(path: Path, *, label: str) -> Path:
    if ".." in path.parts:
        raise ReportingError(f"{label} contains parent traversal: {path}")
    lexical = Path(os.path.abspath(path))
    _assert_no_symlink_ancestry(lexical, label=label)
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ReportingError(f"missing {label}: {path}") from error
    if resolved != lexical or not resolved.is_file():
        raise ReportingError(f"{label} must be one real regular file: {path}")
    return resolved


def _self_hashed_json(path: Path, field: str, label: str) -> dict[str, object]:
    path = _safe_visual_file(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportingError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ReportingError(f"{label} must be one JSON object")
    supplied = payload.get(field)
    immutable = {key: value for key, value in payload.items() if key != field}
    if supplied != canonical_sha256(immutable):
        raise ReportingError(f"{label} canonical self-hash mismatch")
    return payload


def load_raster_manifest(path: Path) -> ReportRasterManifest:
    return ReportRasterManifest.model_validate(
        _self_hashed_json(path, "manifest_sha256", "report raster manifest")
    )


def load_visual_inspection_receipt(path: Path) -> VisualInspectionReceipt:
    return VisualInspectionReceipt.model_validate(
        _self_hashed_json(path, "receipt_sha256", "visual inspection receipt")
    )


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ReportingError(f"invalid raster PNG: {path.name}")
    return struct.unpack(">II", header[16:24])


def _run(command: list[str], *, label: str) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except OSError as error:
        raise ReportingError(f"cannot execute {label}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise ReportingError(f"{label} failed: {detail}")
    return result


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _ordered_poppler_rasters(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    def page_number(path: Path) -> int:
        match = re.fullmatch(r"page-(\d+)\.png", path.name)
        if match is None:
            raise ReportingError(f"unexpected Poppler raster name: {path.name}")
        return int(match.group(1))

    ordered = tuple(sorted(paths, key=page_number))
    numbers = tuple(page_number(path) for path in ordered)
    if numbers != tuple(range(1, len(ordered) + 1)):
        raise ReportingError("Poppler raster page numbering is not contiguous")
    return ordered


def _representatives(
    page_texts: tuple[str, ...], required_titles: tuple[str, ...]
) -> tuple[int, ...]:
    page_count = len(page_texts)
    selected = {1, page_count}
    for fraction in (0.25, 0.5, 0.75):
        selected.add(1 + round((page_count - 1) * fraction))
    important = (
        "Primary C2 versus C1 results",
        "Rare-pivotal preservation",
        "Feedback and interface demonstrations",
        "Bounded first-novel transfer study",
        "Easy-to-interpret qualitative illustrations",
        *required_titles,
    )
    normalized_pages = tuple(_normalized_text(item) for item in page_texts)
    for title in important:
        normalized = _normalized_text(title)
        for index, text in enumerate(normalized_pages, start=1):
            if normalized in text:
                selected.add(index)
                break
    return tuple(sorted(selected))


def _write_append_only(path: Path, payload: bytes) -> None:
    _assert_no_symlink_ancestry(path, label="visual-inspection output")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_ancestry(path, label="visual-inspection output")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ReportingError(f"append-only visual-inspection artifact drift: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise ReportingError(
                    f"concurrent append-only visual-inspection drift: {path}"
                ) from None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_report_rasters(
    *,
    result_manifest_path: Path,
    policy_path: Path,
    pdf_path: Path,
    output_root: Path,
) -> Path:
    """Rasterize every page and emit a pending, self-hashed structural manifest."""

    output_root = _real_visual_root(output_root, create=True)
    document = build_document(result_manifest_path, policy_path)
    pdf_path = _safe_visual_file(pdf_path, label="results PDF")
    if not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise ReportingError("RESULTS_REPORT.pdf is missing or invalid")
    for executable in ("pdfinfo", "pdftoppm", "pdftotext"):
        if shutil.which(executable) is None:
            raise ReportingError(f"PDF visual inspection requires {executable}")
    pdf_hash = _file_sha256(pdf_path)
    token = pdf_hash[:16]
    info = _run(["pdfinfo", str(pdf_path)], label="pdfinfo").stdout.decode(
        "utf-8", errors="replace"
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", info, flags=re.MULTILINE)
    if match is None or int(match.group(1)) < 1:
        raise ReportingError("pdfinfo did not report a positive page count")
    page_count = int(match.group(1))
    renderer_version = _run(["pdftoppm", "-v"], label="pdftoppm -v").stderr.decode(
        "utf-8", errors="replace"
    ).strip().splitlines()[0]
    with tempfile.TemporaryDirectory(prefix="story-projection-pdf-raster-") as temporary:
        prefix = Path(temporary) / "page"
        _run(
            ["pdftoppm", "-png", "-r", "144", str(pdf_path), str(prefix)],
            label="pdftoppm rasterization",
        )
        generated = _ordered_poppler_rasters(
            tuple(Path(temporary).glob("page-*.png"))
        )
        if len(generated) != page_count:
            raise ReportingError("rasterized page count differs from pdfinfo")
        page_records: list[RasterPage] = []
        for number, source in enumerate(generated, start=1):
            relative = PurePosixPath(token) / "pages" / f"page-{number:04d}.png"
            target = _safe_visual_descendant(
                output_root,
                output_root.joinpath(*relative.parts),
                label="report raster",
                require_file=False,
            )
            payload = source.read_bytes()
            _write_append_only(target, payload)
            width, height = _png_dimensions(target)
            page_records.append(
                RasterPage(
                    page_number=number,
                    relative_path=relative.as_posix(),
                    file_sha256=hashlib.sha256(payload).hexdigest(),
                    width_pixels=width,
                    height_pixels=height,
                    size_bytes=len(payload),
                )
            )
    full_text = _run(["pdftotext", str(pdf_path), "-"], label="pdftotext").stdout.decode(
        "utf-8", errors="replace"
    )
    normalized = _normalized_text(full_text)
    section_titles = tuple(item.spec.title for item in document.sections)
    figure_titles = tuple(
        item.caption or item.title for item in document.manifest.figures
    )
    table_titles = tuple(item.spec.description for item in document.tables.values())
    for label, values in (
        ("section", section_titles),
        ("figure", figure_titles),
        ("table", table_titles),
    ):
        missing = [item for item in values if _normalized_text(item) not in normalized]
        if missing:
            raise ReportingError(
                f"report PDF omits registered {label} text: {missing[0]}"
            )
    page_texts = tuple(
        _run(
            ["pdftotext", "-f", str(number), "-l", str(number), str(pdf_path), "-"],
            label=f"pdftotext page {number}",
        ).stdout.decode("utf-8", errors="replace")
        for number in range(1, page_count + 1)
    )
    representatives = _representatives(page_texts, figure_titles)
    checks = (
        StructuralCheck(check_id="pdf_header", status="passed", detail="PDF header is valid."),
        StructuralCheck(
            check_id="page_count",
            status="passed",
            detail=f"pdfinfo and raster output agree on {page_count} pages.",
        ),
        StructuralCheck(
            check_id="section_text_inventory",
            status="passed",
            detail=f"All {len(section_titles)} registered section titles are extractable.",
        ),
        StructuralCheck(
            check_id="figure_text_inventory",
            status="passed",
            detail=f"All {len(figure_titles)} registered figure titles are extractable.",
        ),
        StructuralCheck(
            check_id="table_text_inventory",
            status="passed",
            detail=f"All {len(table_titles)} rendered table titles are extractable.",
        ),
        StructuralCheck(
            check_id="raster_dimensions",
            status="passed",
            detail="Every PDF page raster is a nontrivial PNG with recorded dimensions.",
        ),
    )
    raw: dict[str, object] = {
        "schema_version": VISUAL_INSPECTION_SCHEMA_VERSION,
        "kind": "report_raster_manifest",
        "source_result_manifest_sha256": document.manifest.manifest_sha256,
        "source_pdf_sha256": pdf_hash,
        "source_pdf_name": "RESULTS_REPORT.pdf",
        "renderer": "pdftoppm",
        "renderer_version": renderer_version,
        "raster_prepared_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dpi": 144,
        "page_count": page_count,
        "representative_pages": list(representatives),
        "pages": [item.model_dump(mode="json") for item in page_records],
        "automated_checks": [item.model_dump(mode="json") for item in checks],
        "visual_review_status": "pending",
        "visual_review_notice": (
            "Automated structural checks are not a visual-quality acceptance."
        ),
    }
    raw["manifest_sha256"] = canonical_sha256(raw)
    manifest = ReportRasterManifest.model_validate(raw)
    manifest_path = _safe_visual_descendant(
        output_root,
        output_root / f"report_raster_manifest.{token}.json",
        label="report raster manifest",
        require_file=False,
    )
    _write_append_only(
        manifest_path, (manifest.model_dump_json(indent=2) + "\n").encode("utf-8")
    )
    return manifest_path


def record_visual_inspection(
    *,
    raster_manifest_path: Path,
    output_path: Path,
    reviewer_kind: Literal["researcher", "agent"],
    reviewed_at_utc: AwareDatetime,
    inspected_pages: tuple[int, ...],
    criteria: tuple[VisualCriterion, ...],
) -> VisualInspectionReceipt:
    """Record an explicit post-view inspection; this function never infers acceptance."""

    raster_manifest_path = _safe_visual_file(
        raster_manifest_path, label="report raster manifest"
    )
    visual_root = _real_visual_root(raster_manifest_path.parent, create=False)
    output_path = _safe_visual_descendant(
        visual_root,
        output_path,
        label="visual inspection receipt",
        require_file=False,
    )
    raster = load_raster_manifest(raster_manifest_path)
    if reviewed_at_utc.utcoffset() is None:
        raise ReportingError("visual review timestamp must include a UTC offset")
    normalized_reviewed_at = reviewed_at_utc.astimezone(UTC)
    if normalized_reviewed_at <= raster.raster_prepared_at_utc:
        raise ReportingError("visual review timestamp must be after raster preparation")
    inspected = tuple(sorted(set(inspected_pages)))
    if not set(raster.representative_pages).issubset(inspected):
        raise ReportingError("visual inspection omitted a representative page")
    if not set(inspected).issubset(range(1, raster.page_count + 1)):
        raise ReportingError("visual inspection names a nonexistent page")
    status = "accepted" if all(item.status == "passed" for item in criteria) else "rejected"
    raw: dict[str, object] = {
        "schema_version": VISUAL_INSPECTION_SCHEMA_VERSION,
        "kind": "report_visual_inspection",
        "source_raster_manifest_sha256": raster.manifest_sha256,
        "source_raster_manifest_file_sha256": _file_sha256(raster_manifest_path),
        "source_pdf_sha256": raster.source_pdf_sha256,
        "reviewer_kind": reviewer_kind,
        "reviewed_at_utc": normalized_reviewed_at
        .isoformat()
        .replace("+00:00", "Z"),
        "inspected_pages": list(inspected),
        "criteria": [item.model_dump(mode="json") for item in criteria],
        "status": status,
    }
    raw["receipt_sha256"] = canonical_sha256(raw)
    receipt = VisualInspectionReceipt.model_validate(raw)
    _write_append_only(
        output_path, (receipt.model_dump_json(indent=2) + "\n").encode("utf-8")
    )
    return receipt


def verify_visual_inspection(
    *,
    raster_manifest_path: Path,
    pdf_path: Path,
    receipt_path: Path | None = None,
    require_accepted: bool = False,
) -> tuple[ReportRasterManifest, VisualInspectionReceipt | None]:
    raster_manifest_path = _safe_visual_file(
        raster_manifest_path, label="report raster manifest"
    )
    pdf_path = _safe_visual_file(pdf_path, label="results PDF")
    raster = load_raster_manifest(raster_manifest_path)
    if _file_sha256(pdf_path) != raster.source_pdf_sha256:
        raise ReportingError("visual-inspection raster manifest names another PDF")
    root = _real_visual_root(raster_manifest_path.parent, create=False)
    for page in raster.pages:
        relative = PurePosixPath(page.relative_path)
        if "\\" in page.relative_path or relative.is_absolute() or ".." in relative.parts:
            raise ReportingError("unsafe report raster path")
        path = _safe_visual_descendant(
            root,
            root.joinpath(*relative.parts),
            label=f"report raster page {page.page_number}",
            require_file=True,
        )
        if _file_sha256(path) != page.file_sha256 or path.stat().st_size != page.size_bytes:
            raise ReportingError(f"report raster page changed: {page.page_number}")
        if _png_dimensions(path) != (page.width_pixels, page.height_pixels):
            raise ReportingError(f"report raster dimensions changed: {page.page_number}")
    receipt = None
    if receipt_path is not None:
        receipt_path = _safe_visual_descendant(
            root,
            receipt_path,
            label="visual inspection receipt",
            require_file=True,
        )
        receipt = load_visual_inspection_receipt(receipt_path)
        if (
            receipt.source_raster_manifest_sha256 != raster.manifest_sha256
            or receipt.source_raster_manifest_file_sha256 != _file_sha256(raster_manifest_path)
            or receipt.source_pdf_sha256 != raster.source_pdf_sha256
        ):
            raise ReportingError("visual inspection receipt has stale report lineage")
        if not set(raster.representative_pages).issubset(receipt.inspected_pages):
            raise ReportingError("visual inspection receipt omits a representative page")
        if receipt.reviewed_at_utc <= raster.raster_prepared_at_utc:
            raise ReportingError("visual inspection predates raster preparation")
    if require_accepted and (receipt is None or receipt.status != "accepted"):
        raise ReportingError("an accepted human/agent PDF visual inspection is required")
    return raster, receipt


__all__ = [
    "RasterPage",
    "ReportRasterManifest",
    "StructuralCheck",
    "VisualCriterion",
    "VisualInspectionReceipt",
    "load_raster_manifest",
    "load_visual_inspection_receipt",
    "prepare_report_rasters",
    "record_visual_inspection",
    "verify_visual_inspection",
]
