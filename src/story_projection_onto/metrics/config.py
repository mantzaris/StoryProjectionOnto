"""Frozen, condition-blind configuration for the registered metric panel."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from story_projection_onto.contracts import ImmutableRecord, canonical_sha256
from story_projection_onto.metrics.entropy import OTHER_RELATION_BIN

LEIDEN_LIBRARY_SEED_MODULUS = 2_147_483_647


class RendererMetricConfiguration(ImmutableRecord):
    """Identity of the renderer/layout used for geometry-dependent measures."""

    visualization_configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout_name: str = Field(min_length=1)
    coordinate_rule: str = Field(min_length=1)
    layout_seed: int = Field(ge=0)
    seed_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout_seed_entry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_name: str = Field(min_length=1)
    progressive_disclosure: Literal[True] = True
    font_family: str = Field(min_length=1)
    base_font_px: int = Field(gt=0)
    viewport_width: int = Field(gt=0)
    viewport_height: int = Field(gt=0)

    @property
    def layout_config_hash(self) -> str:
        return canonical_sha256(
            {
                "algorithm": "preset",
                "coordinate_rule": self.coordinate_rule,
                "seed": self.layout_seed,
                "seed_manifest_hash": self.seed_manifest_hash,
                "layout_seed_entry_hash": self.layout_seed_entry_hash,
            }
        )

    @property
    def style_config_hash(self) -> str:
        return canonical_sha256(
            {
                "style": self.style_name,
                "progressive_disclosure": self.progressive_disclosure,
            }
        )

    @property
    def font_config_hash(self) -> str:
        return canonical_sha256(
            {"family": self.font_family, "base_px": self.base_font_px}
        )

    @property
    def viewport_hash(self) -> str:
        """Hash the complete fixed metric viewport, not only its dimensions."""

        return canonical_sha256(
            {
                "schema_version": "1.0.0",
                "center_x": 0.0,
                "center_y": 0.0,
                "zoom": 1.0,
                "width": self.viewport_width,
                "height": self.viewport_height,
            }
        )


class StudyMetricConfiguration(ImmutableRecord):
    """One frozen configuration used symmetrically for every condition."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    configuration_id: Literal["conference-metrics-v1"] = "conference-metrics-v1"
    metric_formula_revision: Literal["conference-metric-formulas-v3"]
    condition_blind: Literal[True] = True
    frozen_before_held_out_scoring: Literal[True] = True
    seed_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_relation_vocabulary: tuple[str, ...]
    upper_relation_mapping: tuple[tuple[str, str], ...]
    leiden_base_resolution: float = Field(gt=0.0)
    leiden_half_resolution: float = Field(gt=0.0)
    leiden_double_resolution: float = Field(gt=0.0)
    leiden_seed_entry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    leiden_source_seed: int = Field(ge=0, lt=2**63)
    leiden_seed_mapping: Literal["modulo-2147483647-v1"]
    leiden_seed: int = Field(ge=0, lt=LEIDEN_LIBRARY_SEED_MODULUS)
    renderer: RendererMetricConfiguration
    von_neumann_entropy: Literal["omitted"] = "omitted"
    von_neumann_omission_reason: str = Field(min_length=1)
    community_review_rubric_revision: str = Field(min_length=1)
    simplification_claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def frozen_values_are_self_consistent(self) -> Self:
        vocabulary = self.canonical_relation_vocabulary
        if len(vocabulary) != len(set(vocabulary)) or OTHER_RELATION_BIN not in vocabulary:
            raise ValueError("canonical relation vocabulary must be unique and include OTHER")
        if any(not item or item.strip() != item for item in vocabulary):
            raise ValueError("canonical relation vocabulary entries must be nonempty and stripped")
        mapping = dict(self.upper_relation_mapping)
        if len(mapping) != len(self.upper_relation_mapping):
            raise ValueError("upper relation mappings must have unique source relations")
        if any(target not in vocabulary for target in mapping.values()):
            raise ValueError("upper relation mappings must target the frozen vocabulary")
        if not math.isclose(
            self.leiden_half_resolution,
            self.leiden_base_resolution * 0.5,
            abs_tol=1e-15,
        ) or not math.isclose(
            self.leiden_double_resolution,
            self.leiden_base_resolution * 2.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Leiden sensitivity resolutions must be exactly half/base/double")
        if self.renderer.seed_manifest_hash != self.seed_manifest_hash:
            raise ValueError("renderer and Leiden metrics must use the same seed manifest")
        if self.leiden_seed != self.leiden_source_seed % LEIDEN_LIBRARY_SEED_MODULUS:
            raise ValueError(
                "Leiden library seed must be the documented signed-32-bit-safe mapping"
            )
        return self

    @property
    def upper_relation_map(self) -> dict[str, str]:
        return dict(self.upper_relation_mapping)

    @property
    def metric_version_hash(self) -> str:
        """Bind result rows to both frozen settings and formula implementation revision."""

        return canonical_sha256(
            {
                "configuration_hash": self.content_hash,
                "formula_revision": self.metric_formula_revision,
            }
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        visualization_path: str | Path | None = None,
        seed_manifest_path: str | Path | None = None,
    ) -> StudyMetricConfiguration:
        metric_path = Path(path)
        configuration = cls.model_validate(
            json.loads(metric_path.read_text(encoding="utf-8"))
        )
        renderer_path = (
            Path(visualization_path)
            if visualization_path is not None
            else metric_path.with_name("visualization.json")
        )
        if not renderer_path.is_file():
            raise ValueError(
                "metric configuration requires its frozen visualization configuration"
            )
        raw = json.loads(renderer_path.read_text(encoding="utf-8"))
        renderer = configuration.renderer
        expected_hash = canonical_sha256(raw)
        if renderer.visualization_configuration_hash != expected_hash:
            raise ValueError(
                "metric renderer binding differs from the visualization configuration hash"
            )
        try:
            viewport = raw["viewport"]
            expected_fields = {
                "layout_name": raw["layout_name"],
                "coordinate_rule": raw["coordinate_rule"],
                "layout_seed": raw["layout_seed"],
                "seed_manifest_hash": raw["seed_manifest_hash"],
                "layout_seed_entry_hash": raw["layout_seed_entry_hash"],
                "style_name": raw["style_name"],
                "progressive_disclosure": raw["progressive_disclosure"],
                "font_family": raw["font_family"],
                "base_font_px": raw["font_base_px"],
                "viewport_width": viewport["width"],
                "viewport_height": viewport["height"],
            }
        except (KeyError, TypeError) as error:
            raise ValueError("visualization configuration lacks renderer metric fields") from error
        actual_fields = {
            name: getattr(renderer, name) for name in expected_fields
        }
        if actual_fields != expected_fields:
            raise ValueError(
                "metric renderer fields differ from the frozen visualization configuration"
            )

        manifest_path = (
            Path(seed_manifest_path)
            if seed_manifest_path is not None
            else metric_path.parents[2] / "data/synthetic/manifests/seed_manifest.json"
        )
        if not manifest_path.is_file():
            raise ValueError("metric configuration requires the frozen seed manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = canonical_sha256(manifest)
        if configuration.seed_manifest_hash != manifest_hash:
            raise ValueError("metric seed binding differs from the frozen seed manifest hash")
        leiden_entries = tuple(
            item for item in manifest.get("entries", ()) if item.get("purpose") == "leiden"
        )
        if len(leiden_entries) != 1:
            raise ValueError("seed manifest must contain exactly one Leiden entry")
        leiden_entry = leiden_entries[0]
        if (
            canonical_sha256(leiden_entry) != configuration.leiden_seed_entry_hash
            or leiden_entry.get("content_hash") != configuration.leiden_seed_entry_hash
        ):
            raise ValueError("metric Leiden entry hash differs from the seed manifest")
        if leiden_entry.get("seed") != configuration.leiden_source_seed:
            raise ValueError("metric Leiden source seed differs from the seed manifest")
        return configuration


class CommunityReviewEntry(ImmutableRecord):
    blinded_output_id: str = Field(min_length=1)
    semantic_coherence: int = Field(ge=1, le=5)
    interpretability: int = Field(ge=1, le=5)
    evidence_support: int = Field(ge=1, le=5)
    reviewer_note: str = ""


class CommunityReviewTemplate(ImmutableRecord):
    schema_version: Literal["1.0.0"] = "1.0.0"
    template_id: Literal["condition-blind-community-review-v1"]
    condition_blind: Literal[True] = True
    rubric_revision: Literal["community-coherence-rubric-v1"]
    allowed_scores: tuple[Literal[1, 2, 3, 4, 5], ...]
    dimensions: tuple[
        Literal["semantic_coherence", "interpretability", "evidence_support"], ...
    ]
    reviews: tuple[CommunityReviewEntry, ...] = ()

    @model_validator(mode="after")
    def template_is_frozen_and_condition_blind(self) -> Self:
        if self.allowed_scores != (1, 2, 3, 4, 5):
            raise ValueError("community review scores must use the frozen five-point scale")
        if self.dimensions != (
            "semantic_coherence",
            "interpretability",
            "evidence_support",
        ):
            raise ValueError("community review dimensions differ from the frozen rubric")
        return self

    @classmethod
    def load(cls, path: str | Path) -> CommunityReviewTemplate:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT_STUDY_METRIC_CONFIG_PATH = Path("configs/study/metrics.json")
DEFAULT_COMMUNITY_REVIEW_TEMPLATE_PATH = Path("configs/study/community_review_template.json")


__all__ = [
    "DEFAULT_COMMUNITY_REVIEW_TEMPLATE_PATH",
    "DEFAULT_STUDY_METRIC_CONFIG_PATH",
    "LEIDEN_LIBRARY_SEED_MODULUS",
    "CommunityReviewEntry",
    "CommunityReviewTemplate",
    "RendererMetricConfiguration",
    "StudyMetricConfiguration",
]
