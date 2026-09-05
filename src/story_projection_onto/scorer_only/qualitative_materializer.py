"""Reproducible qualitative-candidate selection and fixed-grid PNG production."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from story_projection_onto.contracts import (
    Identifier,
    ImmutableRecord,
    ReleaseClass,
    Sha256Digest,
)
from story_projection_onto.phase7_compiler import (
    QualitativeCandidate,
    QualitativeCandidateSet,
    select_qualitative_candidates,
)
from story_projection_onto.reporting import canonical_sha256
from story_projection_onto.scorer_only.blinded_postrun_review import (
    BlindedReviewError,
    _contained_path,
    _file_sha256_bytes,
    _materialize_bundle,
    _model_bytes,
    _read_limited,
    _relative_path,
    _safe_config_file,
)

RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=500)]
ConditionLabel = Literal[
    "c0_classical_pre",
    "c1_llm_pre",
    "c2_llm_query",
    "a_fixed_select",
]
_CONDITION_ORDER = {
    "c0_classical_pre": 0,
    "c1_llm_pre": 1,
    "c2_llm_query": 2,
    "a_fixed_select": 3,
}
_CONDITION_SHORT = {
    "c0_classical_pre": "C0 ClassicalPre",
    "c1_llm_pre": "C1 LLMPre",
    "c2_llm_query": "C2 LLMQuery",
    "a_fixed_select": "A-FixedSelect",
}


class QualitativeMaterializationError(BlindedReviewError):
    """A qualitative source cannot produce the frozen comparable artifacts."""


class QualitativePanelSource(ImmutableRecord):
    panel_artifact_id: Identifier
    candidate_id: Identifier
    context_id: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    condition: ConditionLabel
    relative_path: RelativePath
    file_sha256: Sha256Digest
    width_pixels: int = Field(ge=64, le=4096)
    height_pixels: int = Field(ge=64, le=4096)
    fixed_anchor_layout_hash: Sha256Digest
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC


class QualitativeMaterializationSource(ImmutableRecord):
    manifest_id: Identifier
    frozen_reporting_policy_sha256: Sha256Digest
    copyright_release_attestation_hash: Sha256Digest
    candidates: tuple[QualitativeCandidate, ...]
    panels: tuple[QualitativePanelSource, ...]
    release_class: Literal[ReleaseClass.RESTRICTED] = ReleaseClass.RESTRICTED

    @model_validator(mode="after")
    def inventory_is_exact_and_comparable(self) -> Self:
        candidate_ids = [item.candidate_id for item in self.candidates]
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("qualitative candidate IDs must be nonempty and unique")
        panel_keys = [
            (item.candidate_id, item.context_id, item.condition) for item in self.panels
        ]
        panel_ids = [item.panel_artifact_id for item in self.panels]
        if len(panel_keys) != len(set(panel_keys)) or len(panel_ids) != len(set(panel_ids)):
            raise ValueError("qualitative panel identities must be unique")
        by_candidate: dict[str, list[QualitativePanelSource]] = defaultdict(list)
        for panel in self.panels:
            by_candidate[panel.candidate_id].append(panel)
        if set(by_candidate) != set(candidate_ids):
            raise ValueError("panels must cover exactly the qualitative candidates")
        for candidate in self.candidates:
            panels = by_candidate[candidate.candidate_id]
            required = {
                (row.context_id, row.condition) for row in candidate.display_rows
            }
            required_conditions = (
                {"c0_classical_pre", "c1_llm_pre", "c2_llm_query"}
                if candidate.evidence_split == "case_study"
                else set(_CONDITION_ORDER)
            )
            for context_id in candidate.context_ids:
                conditions = {
                    row.condition
                    for row in candidate.display_rows
                    if row.context_id == context_id
                }
                if conditions != required_conditions:
                    raise ValueError(
                        "candidate displays differ from the registered condition inventory"
                    )
            actual = {(panel.context_id, panel.condition) for panel in panels}
            if actual != required:
                raise ValueError("panels must cover every display row exactly once")
            dimensions = {(panel.width_pixels, panel.height_pixels) for panel in panels}
            if len(dimensions) != 1:
                raise ValueError("all panels in a candidate must use identical dimensions")
            for context_id in candidate.context_ids:
                anchors = {
                    panel.fixed_anchor_layout_hash
                    for panel in panels
                    if panel.context_id == context_id
                }
                if len(anchors) != 1:
                    raise ValueError(
                        "condition panels within a context require one fixed-anchor layout"
                    )
            required_lineage = {
                candidate.composite_figure_artifact_id,
                *(panel.panel_artifact_id for panel in panels),
            }
            if not required_lineage.issubset(candidate.source_artifact_ids):
                raise ValueError("candidate lineage omits a panel or composite artifact")
        return self


class CompositeFigureRecord(ImmutableRecord):
    candidate_id: Identifier
    artifact_id: Identifier
    relative_path: RelativePath
    file_sha256: Sha256Digest
    width_pixels: int = Field(gt=0)
    height_pixels: int = Field(gt=0)
    context_order: tuple[str, ...]
    condition_order: tuple[ConditionLabel, ...]
    panel_artifact_ids: tuple[Identifier, ...]
    panel_hashes: tuple[Sha256Digest, ...]


class QualitativeCompositeManifest(ImmutableRecord):
    manifest_id: Identifier
    source_manifest_hash: Sha256Digest
    candidate_set_hash: Sha256Digest
    selected_candidate_ids: tuple[Identifier, ...]
    figures: tuple[CompositeFigureRecord, ...]
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC


class QualitativeMaterializationReceipt(ImmutableRecord):
    receipt_id: Identifier
    source_manifest_hash: Sha256Digest
    candidate_set_hash: Sha256Digest
    composite_manifest_hash: Sha256Digest
    selected_candidate_ids: tuple[Identifier, ...]
    selected_example_ids: tuple[Identifier, ...]
    selection_rule_hashes: tuple[Sha256Digest, ...]
    complete: Literal[True] = True
    release_class: Literal[ReleaseClass.PUBLIC] = ReleaseClass.PUBLIC


def _load_source(
    restricted_root: Path,
    source_manifest_path: Path,
) -> QualitativeMaterializationSource:
    source_path = _contained_path(
        restricted_root,
        source_manifest_path,
        must_exist=True,
        regular_file=True,
    )
    try:
        return QualitativeMaterializationSource.model_validate_json(_read_limited(source_path))
    except BlindedReviewError:
        raise
    except Exception as error:
        raise QualitativeMaterializationError(
            f"invalid qualitative materialization source: {error}"
        ) from error


def _candidate_set_bytes(value: QualitativeCandidateSet) -> bytes:
    return (
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_panel(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    panel: QualitativePanelSource,
):
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - study dependency gate
        raise QualitativeMaterializationError(
            "qualitative PNG production requires the study Pillow dependency"
        ) from error
    relative = _relative_path(panel.relative_path)
    path = _contained_path(
        restricted_root,
        source_manifest_path.parent.joinpath(*relative.parts),
        must_exist=True,
        regular_file=True,
    )
    payload = _read_limited(path)
    if _file_sha256_bytes(payload) != panel.file_sha256:
        raise QualitativeMaterializationError("qualitative panel hash mismatch")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except Exception as error:
        raise QualitativeMaterializationError("qualitative panel is not a valid PNG") from error
    if image.format != "PNG" or image.size != (panel.width_pixels, panel.height_pixels):
        raise QualitativeMaterializationError("qualitative panel PNG identity drift")
    return image.convert("RGB")


def _render_candidate(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    candidate: QualitativeCandidate,
    panels: tuple[QualitativePanelSource, ...],
) -> tuple[bytes, int, int, tuple[QualitativePanelSource, ...]]:
    from PIL import Image, ImageDraw, ImageFont

    ordered = tuple(
        sorted(
            panels,
            key=lambda item: (
                candidate.context_ids.index(item.context_id),
                _CONDITION_ORDER[item.condition],
            ),
        )
    )
    panel_width = ordered[0].width_pixels
    panel_height = ordered[0].height_pixels
    context_order = candidate.context_ids
    conditions = tuple(
        condition
        for condition in _CONDITION_ORDER
        if any(panel.condition == condition for panel in ordered)
    )
    header_height = 30
    row_label_height = 24
    gap = 8
    margin = 12
    canvas_width = margin * 2 + len(conditions) * panel_width + (len(conditions) - 1) * gap
    canvas_height = (
        margin * 2
        + header_height
        + len(context_order) * (row_label_height + panel_height)
        + (len(context_order) - 1) * gap
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, condition in enumerate(conditions):
        x = margin + column * (panel_width + gap)
        draw.rectangle((x, margin, x + panel_width - 1, margin + header_height - 1), fill="#e8edf5")
        draw.text((x + 6, margin + 8), _CONDITION_SHORT[condition], fill="#111827", font=font)
    by_key = {(item.context_id, item.condition): item for item in ordered}
    for row, context_id in enumerate(context_order):
        y = margin + header_height + row * (row_label_height + panel_height + gap)
        draw.text((margin, y + 5), f"Context {row + 1}: {context_id}", fill="#111827", font=font)
        panel_y = y + row_label_height
        for column, condition in enumerate(conditions):
            panel = by_key[(context_id, condition)]
            image = _load_panel(
                restricted_root=restricted_root,
                source_manifest_path=source_manifest_path,
                panel=panel,
            )
            x = margin + column * (panel_width + gap)
            canvas.paste(image, (x, panel_y))
            draw.rectangle(
                (x, panel_y, x + panel_width - 1, panel_y + panel_height - 1),
                outline="#374151",
                width=1,
            )
    output = io.BytesIO()
    canvas.save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue(), canvas_width, canvas_height, ordered


def prepare_qualitative_materialization(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    reporting_policy_path: Path,
) -> tuple[
    QualitativeCandidateSet,
    QualitativeCompositeManifest,
    QualitativeMaterializationReceipt,
    dict[str, bytes],
]:
    source = _load_source(restricted_root, source_manifest_path)
    policy_path = _safe_config_file(reporting_policy_path)
    candidate_payload = {
        "schema_version": "1.0.0",
        "candidate_set_id": f"qualitative-candidates-{source.content_hash[:20]}",
        "frozen_reporting_policy_sha256": source.frozen_reporting_policy_sha256,
        "candidates": [item.model_dump(mode="json") for item in source.candidates],
        "copyright_release_attestation_hash": source.copyright_release_attestation_hash,
    }
    candidate_payload["content_hash"] = canonical_sha256(candidate_payload)
    candidate_set = QualitativeCandidateSet.model_validate(candidate_payload)
    try:
        selected = select_qualitative_candidates(candidate_set, policy_path=policy_path)
    except Exception as error:
        raise QualitativeMaterializationError(
            f"qualitative source cannot satisfy the frozen selection rules: {error}"
        ) from error
    panels_by_candidate: dict[str, list[QualitativePanelSource]] = defaultdict(list)
    for panel in source.panels:
        panels_by_candidate[panel.candidate_id].append(panel)
    figure_records: list[CompositeFigureRecord] = []
    files: dict[str, bytes] = {}
    for candidate in source.candidates:
        payload, width, height, ordered = _render_candidate(
            restricted_root=restricted_root,
            source_manifest_path=source_manifest_path,
            candidate=candidate,
            panels=tuple(panels_by_candidate[candidate.candidate_id]),
        )
        relative = f"figures/{candidate.composite_figure_artifact_id}.png"
        files[relative] = payload
        figure_records.append(
            CompositeFigureRecord(
                candidate_id=candidate.candidate_id,
                artifact_id=candidate.composite_figure_artifact_id,
                relative_path=relative,
                file_sha256=hashlib.sha256(payload).hexdigest(),
                width_pixels=width,
                height_pixels=height,
                context_order=candidate.context_ids,
                condition_order=tuple(
                    condition
                    for condition in _CONDITION_ORDER
                    if any(item.condition == condition for item in ordered)
                ),
                panel_artifact_ids=tuple(item.panel_artifact_id for item in ordered),
                panel_hashes=tuple(item.file_sha256 for item in ordered),
            )
        )
    selected_candidate_ids = tuple(item.candidate_id for _, item, _ in selected)
    composite_manifest = QualitativeCompositeManifest(
        manifest_id=f"qualitative-composites-{candidate_set.content_hash[:20]}",
        source_manifest_hash=source.content_hash,
        candidate_set_hash=candidate_set.content_hash,
        selected_candidate_ids=selected_candidate_ids,
        figures=tuple(sorted(figure_records, key=lambda item: item.candidate_id)),
    )
    receipt = QualitativeMaterializationReceipt(
        receipt_id=f"qualitative-receipt-{composite_manifest.content_hash[:20]}",
        source_manifest_hash=source.content_hash,
        candidate_set_hash=candidate_set.content_hash,
        composite_manifest_hash=composite_manifest.content_hash,
        selected_candidate_ids=selected_candidate_ids,
        selected_example_ids=tuple(example.value for example, _, _ in selected),
        selection_rule_hashes=tuple(rule_hash for _, _, rule_hash in selected),
    )
    files.update(
        {
            "qualitative_candidate_set.json": _candidate_set_bytes(candidate_set),
            "qualitative_composite_manifest.json": _model_bytes(composite_manifest),
            "materialization_receipt.json": _model_bytes(receipt),
        }
    )
    return candidate_set, composite_manifest, receipt, files


def materialize_qualitative_candidates(
    *,
    restricted_root: Path,
    source_manifest_path: Path,
    reporting_policy_path: Path,
    output_root: Path,
) -> tuple[Path, Literal["created", "verified"], QualitativeMaterializationReceipt]:
    _, _, receipt, files = prepare_qualitative_materialization(
        restricted_root=restricted_root,
        source_manifest_path=source_manifest_path,
        reporting_policy_path=reporting_policy_path,
    )
    path, state = _materialize_bundle(
        restricted_root=restricted_root,
        output_root=output_root,
        identity_hash=receipt.content_hash,
        files=files,
    )
    return path, state, receipt


__all__ = [
    "CompositeFigureRecord",
    "QualitativeCompositeManifest",
    "QualitativeMaterializationError",
    "QualitativeMaterializationReceipt",
    "QualitativeMaterializationSource",
    "QualitativePanelSource",
    "materialize_qualitative_candidates",
    "prepare_qualitative_materialization",
]
