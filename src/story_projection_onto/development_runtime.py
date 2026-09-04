"""Fail-closed control plane for the registered 24-call development block.

This module deliberately does not know how to start, stop, or load a model.  A
caller may inject an already-running, already-metered service through the
``InjectedLiveDevelopmentService`` protocol.  That separation lets the fallback
acceptance controller retain sole ownership of the third and final
acceptance/development model load.

Only opaque model-visible paths and their separately staged neutral evidence
inputs are accepted.  Scorer/gold paths and held-out markers are rejected before
any staged file is opened.  Query JSON is not read while the plan or pre-query
barrier is constructed; an injected audited loader opens each query only after
the C1 prefix has reached terminal ITT state and a query barrier has been
checkpointed.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import (
    RuntimeStageKind,
    RuntimeStagingManifest,
    load_staged_neutral_evidence,
    load_staged_world,
)
from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    OutputBudgets,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    QueryAccessEvent,
    RunOutcome,
    Sha256Digest,
    canonical_json,
)

DEFAULT_DEVELOPMENT_PLAN = Path("configs/study/development_call_manifest.json")
DEVELOPMENT_CALL_COUNT = 24
DEVELOPMENT_UNIT_IDS = (
    "dev-unit-01",
    "dev-unit-02",
    "dev-unit-03",
    "dev-unit-04",
)
DEVELOPMENT_CLASS_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "development_c1": 4,
        "development_c2": 12,
        "development_fixed_select": 4,
        "development_ablation": 3,
        "development_repair": 1,
    }
)
DEVELOPMENT_CLASS_ADMISSION_P95_SECONDS: Mapping[str, int] = MappingProxyType(
    {
        "development_c1": 180,
        "development_c2": 120,
        "development_fixed_select": 90,
        "development_ablation": 120,
        "development_repair": 90,
    }
)
DEVELOPMENT_CLASS_WATCHDOGS: Mapping[str, int] = MappingProxyType(
    {
        "development_c1": 240,
        "development_c2": 150,
        "development_fixed_select": 90,
        "development_ablation": 150,
        "development_repair": 90,
    }
)

# These are the immutable manifest-file hashes of exactly four development
# evidence stages and their twelve query stages.  The compiler currently emits
# split routing only inside scorer_only; this explicit public allowlist avoids
# importing that namespace into a model or condition process.
REGISTERED_DEVELOPMENT_STAGE_MANIFEST_SHA256S = frozenset(
    {
        "c56b8f9114ed37b33bca4ae04df9a7f57335c78790449f9febf3863540998fb8",
        "dead8380aa8310cfe90eb9e5b4aae3a18df8d5deaef7af1b10bd1a0a4b29c603",
        "10812280ffc8459a2beebd9d09e40612dd4049cc9ea5309c9df88366291260e9",
        "63a58c77a136f6f702bd0f9a11b9164606e9255e183109a68b6752057714db59",
        "c1672bf928f5115dd2e78f60e1b3047fa9cabfa9db73cdd1bab3399219e4e67d",
        "14602f5cfad1f1b045834b0e4790d2e53994b4f250c3b736000aaccff28af8ec",
        "aa80a10dc1ed4f87075ad18ca9255d9ddf43def315ca0956f999d774d4813b21",
        "bdb748dae4caf117c489a94e4c5d498b7961b6fee04b79076618243cff0fd689",
        "dbc1ef5c7d2b7b4cacf544bfa1f5837b120ee6d2cdeef74a037566f4b53463d7",
        "b589a2290e26eec18249367c32f02001cafd9c2806f19683a43fd1c31d467e47",
        "a1803fa1c077ba296baf33eb60603008e3db3029a8e589f38ad26e9298df0bd9",
        "104c4cd09d3e996ebd894f7cf3ed5714b1487805f292394d535f4903fe0d6b93",
        "857aa42dc178930ea59dfb64e05becbbc23f0614f52c19e4256cef416640fe6d",
        "f50f3dfe62a70a86480e555428a4192d5a3a9a2e005948911a5daaad93e943f9",
        "8c14f52f4c636194315ccc5a9c03ec97fe7cc110e3a30f7005a28cf205830d6f",
        "1920a8725b613ce89fb5e8eda44730e63929aca3ce5b0725476ba5c621a9ada9",
    }
)
REGISTERED_DEVELOPMENT_NEUTRAL_STAGE_MANIFEST_SHA256S = frozenset(
    {
        "4717cd307566f926438bccb5f2d1e71184f94303d95074978a79ce9c2e6a3a66",
        "f1dec98dd03e7d2f531a473b4675cc1e2ca8a58fb14146a855dddb617fa8fa48",
        "30b521b30e97c7e78eaee373ab81b75063b714dca4799f8fa832c9c224b157c1",
        "661a137c9d87cf7c984440591ced7fe4be6f1f507080d186f96a26afea053726",
    }
)

_FORBIDDEN_RUNTIME_MARKERS = (
    "scorer_only",
    "scorer-only",
    "held_out",
    "held-out",
    "syn-test",
    "gold_projection",
    "expected_effect",
)
_TERMINAL_OUTCOMES = frozenset(
    {
        RunOutcome.SUCCEEDED,
        RunOutcome.INVALID,
        RunOutcome.FAILED,
        RunOutcome.TIMED_OUT,
        RunOutcome.INTERRUPTED,
    }
)


class DevelopmentIntegrityError(ValueError):
    """A frozen route, boundary, dependency, or result was inconsistent."""


class DevelopmentAdmissionError(RuntimeError):
    """The remaining mandatory work cannot be admitted inside the schedule."""


class DevelopmentResumeError(RuntimeError):
    """A checkpoint cannot safely resume against the injected live service."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_safe_runtime_text(name: str, value: str) -> None:
    lowered = value.casefold().replace("\\", "/")
    found = tuple(marker for marker in _FORBIDDEN_RUNTIME_MARKERS if marker in lowered)
    if found:
        raise DevelopmentIntegrityError(
            f"{name} contains a scorer/held-out marker: {', '.join(found)}"
        )


def _assert_safe_relative_path(value: str) -> None:
    _assert_safe_runtime_text("relative_path", value)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise DevelopmentIntegrityError("development stage path must be a bounded relative path")
    required_prefix = ("data", "synthetic", "model_visible")
    if path.parts[: len(required_prefix)] != required_prefix:
        raise DevelopmentIntegrityError(
            "development stage must remain below data/synthetic/model_visible"
        )


def _assert_safe_neutral_relative_path(value: str) -> None:
    _assert_safe_runtime_text("neutral_relative_path", value)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise DevelopmentIntegrityError(
            "neutral evidence stage path must be a bounded relative path"
        )
    required_prefix = ("data", "synthetic", "condition_inputs", "neutral_evidence")
    if path.parts[: len(required_prefix)] != required_prefix or len(path.parts) != 5:
        raise DevelopmentIntegrityError(
            "neutral evidence stage must remain below its isolated condition-input root"
        )


class StageSource(ImmutableRecord):
    relative_path: str = Field(min_length=1)
    manifest_file_sha256: Sha256Digest

    @model_validator(mode="after")
    def public_model_stage_only(self) -> Self:
        _assert_safe_relative_path(self.relative_path)
        if self.manifest_file_sha256 not in REGISTERED_DEVELOPMENT_STAGE_MANIFEST_SHA256S:
            raise ValueError("stage is absent from the frozen development allowlist")
        return self


class NeutralStageSource(ImmutableRecord):
    relative_path: str = Field(min_length=1)
    manifest_file_sha256: Sha256Digest

    @model_validator(mode="after")
    def public_neutral_stage_only(self) -> Self:
        _assert_safe_neutral_relative_path(self.relative_path)
        if self.manifest_file_sha256 not in REGISTERED_DEVELOPMENT_NEUTRAL_STAGE_MANIFEST_SHA256S:
            raise ValueError("neutral stage is absent from the frozen development allowlist")
        return self


class DevelopmentUnitSource(ImmutableRecord):
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    neutral_evidence_stage: NeutralStageSource
    prequery_stage: StageSource
    query_stages: tuple[StageSource, StageSource, StageSource]
    fixed_select_query_ordinal: Literal[1, 2, 3]

    @model_validator(mode="after")
    def stages_are_unique(self) -> Self:
        hashes = (
            self.prequery_stage.manifest_file_sha256,
            *(item.manifest_file_sha256 for item in self.query_stages),
        )
        if len(set(hashes)) != 4:
            raise ValueError("one development unit requires four distinct stage manifests")
        return self


class AblationProbeSource(ImmutableRecord):
    call_id: str = Field(pattern=r"^dev-ablation-[a-z0-9-]+$")
    condition: Literal[
        ConditionName.A_NO_CONTEXT,
        ConditionName.A_NO_TEMPORAL_EPISTEMIC,
        ConditionName.A_NO_RARE_GUARD,
    ]
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    query_ordinal: Literal[1, 2, 3]
    parent_call_id: str = Field(pattern=r"^dev-c2-u0[1-4]-q0[1-3]$")
    switch_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    baseline_value: str = Field(min_length=1)
    probe_value: str = Field(min_length=1)

    @model_validator(mode="after")
    def exactly_one_real_switch(self) -> Self:
        if self.baseline_value == self.probe_value:
            raise ValueError("an ablation probe must change its sole registered switch")
        return self


class RepairProbeSource(ImmutableRecord):
    call_id: Literal["dev-repair-probe-u04-q02"]
    unit_id: Literal["dev-unit-04"]
    query_ordinal: Literal[2]
    parent_call_id: Literal["dev-c2-u04-q02"]
    diagnostic_fixture_id: str = Field(min_length=1)
    fault_injection: Literal["deterministic_unknown_reference_after_preserving_raw_parent"]


class DevelopmentPlanSource(ImmutableRecord):
    plan_id: Literal["phase3-development-block-v1"]
    split: Literal["development"]
    benchmark_manifest_path: str
    benchmark_manifest_file_sha256: Sha256Digest
    seed_manifest_path: str
    seed_manifest_file_sha256: Sha256Digest
    gpu_call_inventory_path: str
    gpu_call_inventory_file_sha256: Sha256Digest
    seed_block: Literal[1]
    frozen_llm_block_seed: Literal[3864250958737859446]
    vllm_seed: Literal[1988649846]
    seed_mapping: Literal["low-31-bits-of-frozen-llm-block-1-v1"]
    units: tuple[
        DevelopmentUnitSource,
        DevelopmentUnitSource,
        DevelopmentUnitSource,
        DevelopmentUnitSource,
    ]
    ablation_probes: tuple[AblationProbeSource, AblationProbeSource, AblationProbeSource]
    repair_probe: RepairProbeSource
    registered_call_ids: tuple[str, ...]

    @model_validator(mode="after")
    def exact_registered_source(self) -> Self:
        for name, value in (
            ("benchmark_manifest_path", self.benchmark_manifest_path),
            ("seed_manifest_path", self.seed_manifest_path),
            ("gpu_call_inventory_path", self.gpu_call_inventory_path),
        ):
            _assert_safe_runtime_text(name, value)
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a bounded relative path")
        if tuple(item.unit_id for item in self.units) != DEVELOPMENT_UNIT_IDS:
            raise ValueError("development units must be the frozen four-unit sequence")
        ablation_conditions = tuple(item.condition for item in self.ablation_probes)
        if ablation_conditions != (
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        ):
            raise ValueError("development ablation probes must use the registered three switches")
        if len(self.registered_call_ids) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("development call-ID inventory must contain exactly 24 entries")
        if len(set(self.registered_call_ids)) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("development call IDs must be unique")
        expected_seed = self.frozen_llm_block_seed & (2**31 - 1)
        if self.vllm_seed != expected_seed:
            raise ValueError("vLLM seed does not match the frozen block-seed mapping")
        return self


class DevelopmentCallKind(StrEnum):
    C1_PRECONSTRUCTION = "c1_preconstruction"
    C2_CONSTRUCTION = "c2_construction"
    FIXED_SELECTION = "fixed_selection"
    ABLATION_PROBE = "ablation_probe"
    REPAIR_PROBE = "repair_probe"


class ConfigurationDelta(ImmutableRecord):
    switch_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    baseline_value: str = Field(min_length=1)
    probe_value: str = Field(min_length=1)
    changed_field_count: Literal[1] = 1

    @model_validator(mode="after")
    def value_changes(self) -> Self:
        if self.baseline_value == self.probe_value:
            raise ValueError("one-switch delta must change its value")
        return self


class StageReference(ImmutableRecord):
    relative_path: str
    manifest_file_sha256: Sha256Digest
    staging_manifest_hash: Sha256Digest
    stage_id: str = Field(min_length=1)
    stage_kind: RuntimeStageKind
    evidence_artifact_hash: Sha256Digest
    query_artifact_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def stage_shape_is_exact(self) -> Self:
        _assert_safe_relative_path(self.relative_path)
        if self.stage_kind is RuntimeStageKind.PREQUERY_EVIDENCE:
            if self.query_artifact_hash is not None:
                raise ValueError("pre-query stage cannot bind a query artifact")
        elif self.query_artifact_hash is None:
            raise ValueError("query stage must bind exactly one query artifact")
        return self


class NeutralStageReference(ImmutableRecord):
    """Verified query-blind full evidence and its model-projection certificate."""

    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    relative_path: str
    manifest_file_sha256: Sha256Digest
    staging_manifest_hash: Sha256Digest
    stage_id: str = Field(pattern=r"^neutralstage_[0-9a-f]{20}$")
    stage_kind: Literal[RuntimeStageKind.NEUTRAL_EVIDENCE]
    neutral_evidence_artifact_hash: Sha256Digest
    equivalence_certificate_hash: Sha256Digest
    snapshot_hash: Sha256Digest
    runtime_unit_id: str = Field(pattern=r"^unit_[0-9a-f]{20}$")
    model_visible_evidence_artifact_hash: Sha256Digest

    @model_validator(mode="after")
    def isolated_neutral_stage(self) -> Self:
        _assert_safe_neutral_relative_path(self.relative_path)
        return self


class DevelopmentCallSpec(ImmutableRecord):
    ordinal: int = Field(ge=1, le=DEVELOPMENT_CALL_COUNT)
    call_id: str = Field(min_length=1)
    call_class: Literal[
        "development_c1",
        "development_c2",
        "development_fixed_select",
        "development_ablation",
        "development_repair",
    ]
    kind: DevelopmentCallKind
    condition: ConditionName
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    seed_block: Literal[1]
    frozen_llm_block_seed: Literal[3864250958737859446]
    vllm_seed: Literal[1988649846]
    admission_p95_seconds: Literal[90, 120, 180]
    watchdog_seconds: Literal[90, 150, 240]
    prequery_stage: StageReference
    query_stage: StageReference | None = None
    source_c1_call_id: str | None = None
    parent_call_id: str | None = None
    configuration_delta: ConfigurationDelta | None = None
    diagnostic_fixture_id: str | None = None
    fault_injection: str | None = None

    @model_validator(mode="after")
    def condition_shape_matches_kind(self) -> Self:
        if self.admission_p95_seconds != DEVELOPMENT_CLASS_ADMISSION_P95_SECONDS[self.call_class]:
            raise ValueError("call admission p95 differs from the frozen inventory")
        if self.watchdog_seconds != DEVELOPMENT_CLASS_WATCHDOGS[self.call_class]:
            raise ValueError("call watchdog differs from the registered class ceiling")
        if self.prequery_stage.stage_kind is not RuntimeStageKind.PREQUERY_EVIDENCE:
            raise ValueError("every call must bind its evidence-only stage")
        query_required = self.kind is not DevelopmentCallKind.C1_PRECONSTRUCTION
        if query_required != (self.query_stage is not None):
            raise ValueError("only C1 development calls omit a query stage")
        if self.query_stage is not None:
            if self.query_stage.stage_kind is not RuntimeStageKind.QUERY_REVEALED:
                raise ValueError("query-time call must bind a query-revealed stage")
            if (
                self.query_stage.evidence_artifact_hash
                != self.prequery_stage.evidence_artifact_hash
            ):
                raise ValueError("query-time call stage differs from its sealed evidence stage")

        if self.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
            expected = ("development_c1", ConditionName.C1_LLM_PRE)
        elif self.kind is DevelopmentCallKind.C2_CONSTRUCTION:
            expected = ("development_c2", ConditionName.C2_LLM_QUERY)
        elif self.kind is DevelopmentCallKind.FIXED_SELECTION:
            expected = ("development_fixed_select", ConditionName.A_FIXED_SELECT)
        elif self.kind is DevelopmentCallKind.ABLATION_PROBE:
            expected = ("development_ablation", self.condition)
            if self.condition not in {
                ConditionName.A_NO_CONTEXT,
                ConditionName.A_NO_TEMPORAL_EPISTEMIC,
                ConditionName.A_NO_RARE_GUARD,
            }:
                raise ValueError("ablation call uses a non-ablation condition")
        else:
            expected = ("development_repair", ConditionName.C2_LLM_QUERY)
        if (self.call_class, self.condition) != expected:
            raise ValueError("development call class, kind, and condition disagree")

        if self.kind is DevelopmentCallKind.FIXED_SELECTION:
            if self.source_c1_call_id is None or self.parent_call_id is not None:
                raise ValueError("FixedSelect requires exactly its same-unit C1 source")
        elif self.source_c1_call_id is not None:
            raise ValueError("only FixedSelect may cite a C1 source call")

        parent_required = self.kind in {
            DevelopmentCallKind.ABLATION_PROBE,
            DevelopmentCallKind.REPAIR_PROBE,
        }
        if parent_required != (self.parent_call_id is not None):
            raise ValueError("only ablation/repair probes require a C2 parent")
        if (self.kind is DevelopmentCallKind.ABLATION_PROBE) != (
            self.configuration_delta is not None
        ):
            raise ValueError("exactly ablation probes carry a one-switch delta")
        repair_metadata = self.diagnostic_fixture_id is not None or self.fault_injection is not None
        if (self.kind is DevelopmentCallKind.REPAIR_PROBE) != repair_metadata:
            raise ValueError("repair-probe metadata must occur only on the repair call")
        if self.kind is DevelopmentCallKind.REPAIR_PROBE and (
            self.diagnostic_fixture_id is None or self.fault_injection is None
        ):
            raise ValueError("repair probe requires both fixture and fault-injection identifiers")
        return self


class DevelopmentCallManifest(ImmutableRecord):
    plan_id: Literal["phase3-development-block-v1"]
    source_plan_hash: Sha256Digest
    benchmark_manifest_file_sha256: Sha256Digest
    seed_manifest_file_sha256: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    seed_block: Literal[1]
    frozen_llm_block_seed: Literal[3864250958737859446]
    vllm_seed: Literal[1988649846]
    neutral_evidence_stages: tuple[
        NeutralStageReference,
        NeutralStageReference,
        NeutralStageReference,
        NeutralStageReference,
    ]
    calls: tuple[DevelopmentCallSpec, ...]

    @model_validator(mode="after")
    def exact_24_call_inventory(self) -> Self:
        if len(self.calls) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("development manifest must contain exactly 24 calls")
        if tuple(item.ordinal for item in self.calls) != tuple(range(1, 25)):
            raise ValueError("development calls must use the exact contiguous ordering")
        if len({item.call_id for item in self.calls}) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("development manifest call IDs must be unique")
        if tuple(item.unit_id for item in self.neutral_evidence_stages) != DEVELOPMENT_UNIT_IDS:
            raise ValueError("development manifest must bind four ordered neutral stages")
        if len(
            {item.neutral_evidence_artifact_hash for item in self.neutral_evidence_stages}
        ) != len(DEVELOPMENT_UNIT_IDS):
            raise ValueError("each development unit requires distinct neutral evidence")
        observed = Counter(item.call_class for item in self.calls)
        if observed != Counter(DEVELOPMENT_CLASS_COUNTS):
            raise ValueError(
                "development call-class inventory differs from the registered 24 calls"
            )

        c1_by_unit = {
            item.unit_id: item
            for item in self.calls
            if item.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
        }
        c2_by_id = {
            item.call_id: item
            for item in self.calls
            if item.kind is DevelopmentCallKind.C2_CONSTRUCTION
        }
        if tuple(sorted(c1_by_unit)) != DEVELOPMENT_UNIT_IDS:
            raise ValueError("each development unit must have exactly one C1 preconstruction")
        for neutral in self.neutral_evidence_stages:
            if (
                c1_by_unit[neutral.unit_id].prequery_stage.evidence_artifact_hash
                != neutral.model_visible_evidence_artifact_hash
            ):
                raise ValueError("neutral projection certificate names another model-visible stage")
        c2_pairs = {(item.unit_id, item.query_stage.content_hash) for item in c2_by_id.values()}
        if len(c2_pairs) != 12:
            raise ValueError("C2 must cover every one of twelve development query stages")

        for item in self.calls:
            if item.source_c1_call_id is not None:
                source = c1_by_unit.get(item.unit_id)
                if source is None or source.call_id != item.source_c1_call_id:
                    raise ValueError("FixedSelect source is not the same-unit C1 call")
            if item.parent_call_id is not None:
                parent = c2_by_id.get(item.parent_call_id)
                if parent is None or parent.unit_id != item.unit_id:
                    raise ValueError("probe parent must be a same-unit registered C2 call")
                if parent.query_stage != item.query_stage:
                    raise ValueError("probe and C2 parent must use the exact same query stage")
                if parent.vllm_seed != item.vllm_seed:
                    raise ValueError("probe and parent must use the same paired seed")
        return self


def _safe_resolve(root: Path, relative_path: str) -> Path:
    _assert_safe_relative_path(relative_path)
    repository = root.resolve(strict=True)
    model_root = (repository / "data/synthetic/model_visible").resolve(strict=True)
    candidate = repository / relative_path
    if candidate.is_symlink():
        raise DevelopmentIntegrityError("development stage cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(model_root)
    except ValueError as error:
        raise DevelopmentIntegrityError(
            "development stage escaped the model-visible root"
        ) from error
    return resolved


def _safe_resolve_neutral(root: Path, relative_path: str) -> Path:
    _assert_safe_neutral_relative_path(relative_path)
    repository = root.resolve(strict=True)
    neutral_root = (repository / "data/synthetic/condition_inputs/neutral_evidence").resolve(
        strict=True
    )
    candidate = repository / relative_path
    if candidate.is_symlink():
        raise DevelopmentIntegrityError("neutral evidence stage cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(neutral_root)
    except ValueError as error:
        raise DevelopmentIntegrityError(
            "neutral evidence stage escaped its condition-input root"
        ) from error
    if len(relative.parts) != 1:
        raise DevelopmentIntegrityError("neutral evidence stage must be one isolated directory")
    return resolved


def _load_stage(
    root: Path,
    source: StageSource,
    *,
    expected_kind: RuntimeStageKind,
) -> StageReference:
    directory = _safe_resolve(root, source.relative_path)
    if not directory.is_dir():
        raise DevelopmentIntegrityError("development stage is not a directory")
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or _file_sha256(manifest_path) != source.manifest_file_sha256:
        raise DevelopmentIntegrityError("development stage manifest bytes changed")
    manifest = RuntimeStagingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.stage_kind is not expected_kind:
        raise DevelopmentIntegrityError("development stage kind differs from its frozen role")
    expected_names = (
        {"manifest.json", "evidence.json"}
        if expected_kind is RuntimeStageKind.PREQUERY_EVIDENCE
        else {"manifest.json", "evidence.json", "query.json"}
    )
    actual_names = {item.name for item in directory.iterdir()}
    if actual_names != expected_names or any(
        item.is_dir() or item.is_symlink() for item in directory.iterdir()
    ):
        raise DevelopmentIntegrityError("development stage directory is not the exact flat sandbox")
    query_hash = (
        manifest.artifact_hashes[1] if expected_kind is RuntimeStageKind.QUERY_REVEALED else None
    )
    return StageReference(
        relative_path=source.relative_path,
        manifest_file_sha256=source.manifest_file_sha256,
        staging_manifest_hash=manifest.content_hash,
        stage_id=manifest.stage_id,
        stage_kind=manifest.stage_kind,
        evidence_artifact_hash=manifest.artifact_hashes[0],
        query_artifact_hash=query_hash,
    )


def _load_neutral_stage(
    root: Path,
    unit_id: str,
    source: NeutralStageSource,
    prequery_source: StageSource,
) -> NeutralStageReference:
    directory = _safe_resolve_neutral(root, source.relative_path)
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or _file_sha256(manifest_path) != source.manifest_file_sha256:
        raise DevelopmentIntegrityError("neutral stage manifest bytes changed")
    manifest = RuntimeStagingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.stage_kind is not RuntimeStageKind.NEUTRAL_EVIDENCE:
        raise DevelopmentIntegrityError("neutral stage kind differs from its frozen role")

    prequery_directory = _safe_resolve(root, prequery_source.relative_path)
    prequery_manifest = RuntimeStagingManifest.model_validate_json(
        (prequery_directory / "manifest.json").read_text(encoding="utf-8")
    )
    model_visible = load_staged_world(
        prequery_directory / "evidence.json",
        prequery_directory,
        prequery_manifest,
    )
    neutral, certificate = load_staged_neutral_evidence(
        directory,
        root / "data/synthetic/condition_inputs/neutral_evidence",
        manifest,
        model_visible,
    )
    return NeutralStageReference(
        unit_id=unit_id,
        relative_path=source.relative_path,
        manifest_file_sha256=source.manifest_file_sha256,
        staging_manifest_hash=manifest.content_hash,
        stage_id=manifest.stage_id,
        stage_kind=manifest.stage_kind,
        neutral_evidence_artifact_hash=neutral.content_hash,
        equivalence_certificate_hash=certificate.content_hash,
        snapshot_hash=neutral.snapshot.content_hash,
        runtime_unit_id=neutral.snapshot.world_or_window_id,
        model_visible_evidence_artifact_hash=model_visible.content_hash,
    )


def _call(
    *,
    ordinal: int,
    call_id: str,
    call_class: str,
    kind: DevelopmentCallKind,
    condition: ConditionName,
    unit: DevelopmentUnitSource,
    prequery: StageReference,
    query: StageReference | None,
    source: DevelopmentPlanSource,
    source_c1_call_id: str | None = None,
    parent_call_id: str | None = None,
    delta: ConfigurationDelta | None = None,
    diagnostic_fixture_id: str | None = None,
    fault_injection: str | None = None,
) -> DevelopmentCallSpec:
    return DevelopmentCallSpec(
        ordinal=ordinal,
        call_id=call_id,
        call_class=call_class,
        kind=kind,
        condition=condition,
        unit_id=unit.unit_id,
        seed_block=source.seed_block,
        frozen_llm_block_seed=source.frozen_llm_block_seed,
        vllm_seed=source.vllm_seed,
        watchdog_seconds=DEVELOPMENT_CLASS_WATCHDOGS[call_class],
        admission_p95_seconds=DEVELOPMENT_CLASS_ADMISSION_P95_SECONDS[call_class],
        prequery_stage=prequery,
        query_stage=query,
        source_c1_call_id=source_c1_call_id,
        parent_call_id=parent_call_id,
        configuration_delta=delta,
        diagnostic_fixture_id=diagnostic_fixture_id,
        fault_injection=fault_injection,
    )


def load_development_call_manifest(
    root: Path,
    plan_path: Path | None = None,
) -> DevelopmentCallManifest:
    """Load and resolve the frozen development plan without reading query JSON."""

    repository = root.resolve(strict=True)
    source_path = plan_path or repository / DEFAULT_DEVELOPMENT_PLAN
    _assert_safe_runtime_text("development_plan_path", source_path.as_posix())
    if source_path.is_symlink():
        raise DevelopmentIntegrityError("development plan cannot be a symlink")
    source = DevelopmentPlanSource.model_validate_json(source_path.read_text(encoding="utf-8"))
    for relative, expected in (
        (source.benchmark_manifest_path, source.benchmark_manifest_file_sha256),
        (source.seed_manifest_path, source.seed_manifest_file_sha256),
        (source.gpu_call_inventory_path, source.gpu_call_inventory_file_sha256),
    ):
        candidate = repository / relative
        if candidate.is_symlink() or _file_sha256(candidate.resolve(strict=True)) != expected:
            raise DevelopmentIntegrityError("frozen benchmark or seed manifest bytes changed")

    resolved: dict[
        str,
        tuple[NeutralStageReference, StageReference, tuple[StageReference, ...]],
    ] = {}
    for unit in source.units:
        prequery = _load_stage(
            repository,
            unit.prequery_stage,
            expected_kind=RuntimeStageKind.PREQUERY_EVIDENCE,
        )
        queries = tuple(
            _load_stage(repository, item, expected_kind=RuntimeStageKind.QUERY_REVEALED)
            for item in unit.query_stages
        )
        if any(item.evidence_artifact_hash != prequery.evidence_artifact_hash for item in queries):
            raise DevelopmentIntegrityError(
                "query stages do not share their unit's sealed evidence"
            )
        neutral = _load_neutral_stage(
            repository,
            unit.unit_id,
            unit.neutral_evidence_stage,
            unit.prequery_stage,
        )
        if neutral.model_visible_evidence_artifact_hash != prequery.evidence_artifact_hash:
            raise DevelopmentIntegrityError(
                "neutral evidence does not project to the unit's prequery evidence"
            )
        resolved[unit.unit_id] = (neutral, prequery, queries)

    configured_stage_hashes = tuple(
        stage.manifest_file_sha256
        for unit in source.units
        for stage in (unit.prequery_stage, *unit.query_stages)
    )
    if (
        len(configured_stage_hashes) != len(REGISTERED_DEVELOPMENT_STAGE_MANIFEST_SHA256S)
        or len(set(configured_stage_hashes)) != len(configured_stage_hashes)
        or set(configured_stage_hashes) != REGISTERED_DEVELOPMENT_STAGE_MANIFEST_SHA256S
    ):
        raise DevelopmentIntegrityError(
            "configured development stages differ from the exact frozen 4+12 allowlist"
        )
    configured_neutral_hashes = tuple(
        unit.neutral_evidence_stage.manifest_file_sha256 for unit in source.units
    )
    if (
        len(set(configured_neutral_hashes)) != len(DEVELOPMENT_UNIT_IDS)
        or set(configured_neutral_hashes) != REGISTERED_DEVELOPMENT_NEUTRAL_STAGE_MANIFEST_SHA256S
    ):
        raise DevelopmentIntegrityError(
            "configured neutral stages differ from the exact frozen four-unit allowlist"
        )

    calls: list[DevelopmentCallSpec] = []
    ordinal = 1
    for unit in source.units:
        _, prequery, _ = resolved[unit.unit_id]
        suffix = unit.unit_id.removeprefix("dev-unit-")
        calls.append(
            _call(
                ordinal=ordinal,
                call_id=f"dev-c1-u{suffix}",
                call_class="development_c1",
                kind=DevelopmentCallKind.C1_PRECONSTRUCTION,
                condition=ConditionName.C1_LLM_PRE,
                unit=unit,
                prequery=prequery,
                query=None,
                source=source,
            )
        )
        ordinal += 1
    for unit in source.units:
        _, prequery, queries = resolved[unit.unit_id]
        suffix = unit.unit_id.removeprefix("dev-unit-")
        for query_ordinal, query in enumerate(queries, start=1):
            calls.append(
                _call(
                    ordinal=ordinal,
                    call_id=f"dev-c2-u{suffix}-q{query_ordinal:02d}",
                    call_class="development_c2",
                    kind=DevelopmentCallKind.C2_CONSTRUCTION,
                    condition=ConditionName.C2_LLM_QUERY,
                    unit=unit,
                    prequery=prequery,
                    query=query,
                    source=source,
                )
            )
            ordinal += 1
    for unit in source.units:
        _, prequery, queries = resolved[unit.unit_id]
        suffix = unit.unit_id.removeprefix("dev-unit-")
        query_ordinal = unit.fixed_select_query_ordinal
        calls.append(
            _call(
                ordinal=ordinal,
                call_id=f"dev-fixed-u{suffix}-q{query_ordinal:02d}",
                call_class="development_fixed_select",
                kind=DevelopmentCallKind.FIXED_SELECTION,
                condition=ConditionName.A_FIXED_SELECT,
                unit=unit,
                prequery=prequery,
                query=queries[query_ordinal - 1],
                source=source,
                source_c1_call_id=f"dev-c1-u{suffix}",
            )
        )
        ordinal += 1
    units_by_id = {item.unit_id: item for item in source.units}
    for probe in source.ablation_probes:
        unit = units_by_id[probe.unit_id]
        _, prequery, queries = resolved[unit.unit_id]
        calls.append(
            _call(
                ordinal=ordinal,
                call_id=probe.call_id,
                call_class="development_ablation",
                kind=DevelopmentCallKind.ABLATION_PROBE,
                condition=probe.condition,
                unit=unit,
                prequery=prequery,
                query=queries[probe.query_ordinal - 1],
                source=source,
                parent_call_id=probe.parent_call_id,
                delta=ConfigurationDelta(
                    switch_name=probe.switch_name,
                    baseline_value=probe.baseline_value,
                    probe_value=probe.probe_value,
                ),
            )
        )
        ordinal += 1
    repair = source.repair_probe
    unit = units_by_id[repair.unit_id]
    _, prequery, queries = resolved[unit.unit_id]
    calls.append(
        _call(
            ordinal=ordinal,
            call_id=repair.call_id,
            call_class="development_repair",
            kind=DevelopmentCallKind.REPAIR_PROBE,
            condition=ConditionName.C2_LLM_QUERY,
            unit=unit,
            prequery=prequery,
            query=queries[repair.query_ordinal - 1],
            source=source,
            parent_call_id=repair.parent_call_id,
            diagnostic_fixture_id=repair.diagnostic_fixture_id,
            fault_injection=repair.fault_injection,
        )
    )
    if tuple(item.call_id for item in calls) != source.registered_call_ids:
        raise DevelopmentIntegrityError("expanded call sequence differs from its frozen registry")
    return DevelopmentCallManifest(
        plan_id=source.plan_id,
        source_plan_hash=source.content_hash,
        benchmark_manifest_file_sha256=source.benchmark_manifest_file_sha256,
        seed_manifest_file_sha256=source.seed_manifest_file_sha256,
        gpu_call_inventory_file_sha256=source.gpu_call_inventory_file_sha256,
        seed_block=source.seed_block,
        frozen_llm_block_seed=source.frozen_llm_block_seed,
        vllm_seed=source.vllm_seed,
        neutral_evidence_stages=tuple(resolved[unit_id][0] for unit_id in DEVELOPMENT_UNIT_IDS),
        calls=tuple(calls),
    )


class UnitPrequeryBinding(ImmutableRecord):
    """Gold-free runner inputs that must exist before the query stage is opened."""

    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    runtime_unit_id: str = Field(pattern=r"^unit_[0-9a-f]{20}$")
    snapshot_hash: Sha256Digest
    staged_model_visible_evidence_hash: Sha256Digest
    neutral_full_evidence_artifact_hash: Sha256Digest
    evidence_equivalence_certificate_hash: Sha256Digest
    preexisting_preparation_bindings: tuple[PrequeryPreparationBinding, ...]
    completed_at: AwareDatetime
    materialized_from_scorer_namespace: Literal[False] = False
    c2_inventory_verified_empty: Literal[True] = True

    @model_validator(mode="after")
    def exact_preexisting_preparations(self) -> Self:
        expected_conditions: dict[str, tuple[ConditionName, ...]] = {
            "dev-unit-01": (
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_NO_CONTEXT,
            ),
            "dev-unit-02": (
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ),
            "dev-unit-03": (
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_NO_RARE_GUARD,
            ),
            "dev-unit-04": (
                ConditionName.C0_CLASSICAL_PRE,
                ConditionName.C2_LLM_QUERY,
            ),
        }
        observed = tuple(item.condition for item in self.preexisting_preparation_bindings)
        if observed != expected_conditions[self.unit_id]:
            raise ValueError("unit does not carry its exact C0/C2/ablation prequery preparations")
        for item in self.preexisting_preparation_bindings:
            if item.unit_id != self.runtime_unit_id or item.snapshot_hash != self.snapshot_hash:
                raise ValueError("prequery preparation belongs to another unit or snapshot")
            expected_seed = None if item.condition is ConditionName.C0_CLASSICAL_PRE else 1
            if item.seed_block != expected_seed:
                raise ValueError("prequery preparation uses a different frozen seed block")
            if item.completed_at > self.completed_at:
                raise ValueError("unit completion cannot predate one of its preparations")
        return self


class FixedSchemaDerivationPlan(ImmutableRecord):
    """Query-blind recipe for an exact grammar that depends on a future C1 seal."""

    derivation_algorithm: Literal["fixed-select-sealed-inventory-schema-v1"] = (
        "fixed-select-sealed-inventory-schema-v1"
    )
    ordinal: Literal[17, 18, 19, 20]
    call_id: str = Field(pattern=r"^dev-fixed-u0[1-4]-q03$")
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    source_c1_call_id: str = Field(pattern=r"^dev-c1-u0[1-4]$")
    seed_block: Literal[1] = 1
    resolved_seed: Literal[1988649846] = 1988649846
    budgets: OutputBudgets
    maximum_input_tokens: Literal[10240] = 10240
    maximum_output_tokens: Literal[2048] = 2048
    repair_attempt_budget: Literal[1] = 1
    model_stack_hash: Sha256Digest
    decoding_family_hash: Sha256Digest
    seed_manifest_hash: Sha256Digest
    prompt_hash: Sha256Digest
    base_output_schema_hash: Sha256Digest
    scored_schema_hash: Sha256Digest
    capability_manifest_hash: Sha256Digest
    validator_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    prequery_evidence_artifact_hash: Sha256Digest
    query_stage_manifest_hash: Sha256Digest

    @model_validator(mode="after")
    def same_unit_and_exact_budget(self) -> Self:
        unit_suffix = self.unit_id.removeprefix("dev-unit-")
        if (
            unit_suffix not in self.call_id
            or unit_suffix not in self.source_c1_call_id
            or self.repair_attempt_budget != self.budgets.repair_attempt_budget
        ):
            raise ValueError("FixedSelect derivation plan changed unit or budget")
        return self


class FixedSchemaDerivationReceipt(ImmutableRecord):
    """Exact post-C1, pre-query materialization of one frozen derivation plan."""

    plan_hash: Sha256Digest
    call_id: str = Field(pattern=r"^dev-fixed-u0[1-4]-q03$")
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    source_c1_construction_seal_hash: Sha256Digest
    source_c1_preparation_hash: Sha256Digest
    fixed_ontology_hash: Sha256Digest
    evidence_alias_bijection_hash: Sha256Digest
    derived_output_schema_hash: Sha256Digest
    derived_decoding_manifest_hash: Sha256Digest
    exact_run_condition_config_hash: Sha256Digest
    exact_run_condition_config_artifact_hash: Sha256Digest
    derived_at: AwareDatetime


class DevelopmentPrequeryInputs(ImmutableRecord):
    data_contract_kind: Literal["gold_free_neutral_evidence_v1"] = "gold_free_neutral_evidence_v1"
    source_tree_hash: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    upper_ontology_hash: Sha256Digest
    prompt_family_hash: Sha256Digest
    output_schema_hash: Sha256Digest
    validator_hash: Sha256Digest
    run_condition_config_hashes: tuple[Sha256Digest | None, ...]
    fixed_schema_derivation_plans: tuple[
        FixedSchemaDerivationPlan,
        FixedSchemaDerivationPlan,
        FixedSchemaDerivationPlan,
        FixedSchemaDerivationPlan,
    ]
    unit_bindings: tuple[
        UnitPrequeryBinding,
        UnitPrequeryBinding,
        UnitPrequeryBinding,
        UnitPrequeryBinding,
    ]

    @model_validator(mode="after")
    def exactly_four_gold_free_units(self) -> Self:
        if tuple(item.unit_id for item in self.unit_bindings) != DEVELOPMENT_UNIT_IDS:
            raise ValueError("pre-query bindings must use the exact four development units")
        if len({item.neutral_full_evidence_artifact_hash for item in self.unit_bindings}) != 4:
            raise ValueError("each development unit requires its own neutral evidence artifact")
        if len({item.snapshot_hash for item in self.unit_bindings}) != 4:
            raise ValueError("each development unit requires its own evidence snapshot")
        if len(self.run_condition_config_hashes) != DEVELOPMENT_CALL_COUNT:
            raise ValueError("prequery inputs must bind one RunConditionConfig per call")
        if any(
            value is not None
            for value in self.run_condition_config_hashes[16:20]
        ) or any(
            value is None
            for value in (
                *self.run_condition_config_hashes[:16],
                *self.run_condition_config_hashes[20:],
            )
        ):
            raise ValueError(
                "only FixedSelect exact run configurations may be deferred until C1 seals"
            )
        if tuple(item.ordinal for item in self.fixed_schema_derivation_plans) != (
            17,
            18,
            19,
            20,
        ):
            raise ValueError("prequery inputs require four ordered FixedSelect derivation plans")
        return self

    def binding_for(self, unit_id: str) -> UnitPrequeryBinding:
        for binding in self.unit_bindings:
            if binding.unit_id == unit_id:
                return binding
        raise KeyError(unit_id)

    def run_config_hash_for(self, ordinal: int) -> Sha256Digest:
        if not 1 <= ordinal <= DEVELOPMENT_CALL_COUNT:
            raise IndexError(ordinal)
        value = self.run_condition_config_hashes[ordinal - 1]
        if value is None:
            raise KeyError(
                "FixedSelect exact run config is materialized only after its C1 seal"
            )
        return value

    def fixed_schema_plan_for(self, ordinal: int) -> FixedSchemaDerivationPlan:
        for plan in self.fixed_schema_derivation_plans:
            if plan.ordinal == ordinal:
                return plan
        raise KeyError(ordinal)


class LiveServiceIdentity(ImmutableRecord):
    """Stable identity of a model service loaded by an external lifecycle owner."""

    owner_run_id: str = Field(min_length=1)
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    gpu_session_event_id: str = Field(min_length=1)
    launcher_configuration_hash: Sha256Digest
    model_snapshot_hash: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    source_execution_hash: Sha256Digest
    already_running: Literal[True] = True
    model_load_count: Literal[1] = 1
    runner_lifecycle_authority: Literal[False] = False


class DevelopmentExecutionManifest(ImmutableRecord):
    """Hash root binding the exact calls, inputs, service, and forecast receipt."""

    execution_id: str = Field(min_length=1)
    call_manifest_hash: Sha256Digest
    source_plan_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    forecast_control_hash: Sha256Digest


class DevelopmentPreconstructionBarrier(ImmutableRecord):
    """Controller-only prefix seal created before the four C1 model calls."""

    barrier_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    execution_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    source_plan_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    unit_binding_hashes: tuple[Sha256Digest, Sha256Digest, Sha256Digest, Sha256Digest]
    recorded_at: AwareDatetime
    query_stage_open_count: Literal[0] = 0


class C1SealReceipt(ImmutableRecord):
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    call_id: str = Field(pattern=r"^dev-c1-u0[1-4]$")
    itt_record_hash: Sha256Digest
    outcome: RunOutcome
    construction_seal_hash: Sha256Digest | None = None
    preparation_bindings: tuple[PrequeryPreparationBinding, ...] = ()
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def successful_c1_has_seal(self) -> Self:
        if self.outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("C1 seal receipt requires a terminal outcome")
        if (self.outcome is RunOutcome.SUCCEEDED) != (self.construction_seal_hash is not None):
            raise ValueError("only successful C1 calls may contribute a construction seal")
        if self.outcome is RunOutcome.SUCCEEDED:
            if tuple(item.condition for item in self.preparation_bindings) != (
                ConditionName.C1_LLM_PRE,
                ConditionName.A_FIXED_SELECT,
            ):
                raise ValueError("successful C1 must bind its C1 and same-seed Fixed preparation")
            if any(item.seed_block != 1 for item in self.preparation_bindings):
                raise ValueError("C1/Fixed preparations use another unit or seed")
        elif self.preparation_bindings:
            raise ValueError("failed C1 cannot claim condition preparations")
        return self


class CallExecutionEnvelope(ImmutableRecord):
    execution_id: str = Field(min_length=1)
    execution_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    expected_run_condition_config_hash: Sha256Digest | None = None
    fixed_schema_derivation_plan_hash: Sha256Digest | None = None
    preconstruction_barrier_hash: Sha256Digest
    preconstruction_barrier_recorded_at: AwareDatetime
    prequery_barrier_hash: Sha256Digest | None = None
    query_access_event: QueryAccessEvent | None = None
    unit_prequery_binding_hash: Sha256Digest
    source_c1_seal_hash: Sha256Digest | None = None
    source_c1_output_artifact_hash: Sha256Digest | None = None
    parent_itt_record_hash: Sha256Digest | None = None
    parent_output_artifact_hash: Sha256Digest | None = None
    service_identity_hash: Sha256Digest

    @model_validator(mode="after")
    def exact_or_deferred_run_configuration(self) -> Self:
        if (self.expected_run_condition_config_hash is None) == (
            self.fixed_schema_derivation_plan_hash is None
        ):
            raise ValueError(
                "execution envelope requires exactly one exact config or Fixed derivation plan"
            )
        return self


class ServiceCallResult(ImmutableRecord):
    """Ledger/CAS receipt returned by an injected, already-metered service adapter."""

    call_id: str = Field(min_length=1)
    outcome: RunOutcome
    request_started: bool
    request_hash: Sha256Digest | None = None
    response_artifact_hash: Sha256Digest | None = None
    validated_generation_hash: Sha256Digest | None = None
    validation_record_hash: Sha256Digest | None = None
    condition_attempt_hash: Sha256Digest | None = None
    condition_preparation_hash: Sha256Digest | None = None
    ledger_receipt_hash: Sha256Digest | None = None
    gpu_event_id: str | None = None
    service_identity_hash: Sha256Digest
    run_condition_config_hash: Sha256Digest
    fixed_schema_derivation: FixedSchemaDerivationReceipt | None = None
    query_access_event_hash: Sha256Digest | None = None
    construction_seal_hash: Sha256Digest | None = None
    prequery_preparation_bindings: tuple[PrequeryPreparationBinding, ...] = ()
    repair_parent_raw_output_hash: Sha256Digest | None = None
    allocated_gpu_seconds: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    failure_code: str | None = None
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def terminal_and_accounted(self) -> Self:
        if self.outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("development service result requires a terminal outcome")
        for name, value in (("allocated_gpu_seconds", self.allocated_gpu_seconds),):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self.request_started:
            if any(
                (
                    self.request_hash,
                    self.response_artifact_hash,
                    self.ledger_receipt_hash,
                    self.gpu_event_id,
                )
            ):
                raise ValueError("a request-not-started result cannot claim GPU/CAS accounting")
            if self.allocated_gpu_seconds != 0 or self.prompt_tokens or self.completion_tokens:
                raise ValueError("a request-not-started result cannot claim model work")
        else:
            if (
                self.request_hash is None
                or self.ledger_receipt_hash is None
                or self.gpu_event_id is None
            ):
                raise ValueError(
                    "a started request requires request, ledger, and GPU-event lineage"
                )
        if self.outcome is RunOutcome.SUCCEEDED:
            required = (
                self.response_artifact_hash,
                self.validated_generation_hash,
                self.validation_record_hash,
            )
            if not self.request_started or any(item is None for item in required):
                raise ValueError("successful call requires stored output and validation")
            if (self.condition_attempt_hash is None) == (self.condition_preparation_hash is None):
                raise ValueError("success requires exactly one attempt or preparation artifact")
            if self.failure_code is not None:
                raise ValueError("successful call cannot carry a failure code")
        elif not self.failure_code:
            raise ValueError("nonsuccessful call requires a failure code")
        if self.fixed_schema_derivation is not None and (
            self.fixed_schema_derivation.exact_run_condition_config_hash
            != self.run_condition_config_hash
        ):
            raise ValueError("FixedSelect derivation and exact run config hashes differ")
        fixed_started = self.call_id.startswith("dev-fixed-") and self.request_started
        if fixed_started != (self.fixed_schema_derivation is not None):
            raise ValueError(
                "exactly FixedSelect service results require a schema derivation"
            )
        if self.outcome is not RunOutcome.SUCCEEDED and (
            self.condition_preparation_hash is not None or self.prequery_preparation_bindings
        ):
            raise ValueError("nonsuccessful call cannot claim completed preparations")
        return self


@runtime_checkable
class InjectedLiveDevelopmentService(Protocol):
    """Narrow protocol intentionally lacking start/load/shutdown operations."""

    def identity(self) -> LiveServiceIdentity:
        """Return the immutable identity of the already-running service."""

    def allocated_gpu_seconds(self) -> float:
        """Return the cumulative monotonic allocated-service counter."""

    def open_query(
        self,
        stage: StageReference,
        barrier: PrequeryBarrier,
    ) -> QueryAccessEvent:
        """Audit, reconstruct, and persist one full-context query before returning."""

    def execute_call(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult:
        """Execute and meter one already-admitted request."""

    def recover_call(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult | None:
        """Recover a durable receipt; never issue a duplicate request."""


class RequestStartState(StrEnum):
    NOT_STARTED = "not_started"
    STARTED = "started"
    UNKNOWN_AFTER_INTERRUPTION = "unknown_after_interruption"


class DevelopmentITTRecord(ImmutableRecord):
    ordinal: int = Field(ge=1, le=24)
    call_id: str = Field(min_length=1)
    call_class: str = Field(min_length=1)
    condition: ConditionName
    unit_id: str = Field(pattern=r"^dev-unit-0[1-4]$")
    outcome: RunOutcome
    request_start_state: RequestStartState
    service_result_hash: Sha256Digest | None = None
    response_artifact_hash: Sha256Digest | None = None
    validated_generation_hash: Sha256Digest | None = None
    validation_record_hash: Sha256Digest | None = None
    condition_attempt_hash: Sha256Digest | None = None
    condition_preparation_hash: Sha256Digest | None = None
    run_condition_config_hash: Sha256Digest | None = None
    fixed_schema_derivation: FixedSchemaDerivationReceipt | None = None
    ledger_receipt_hash: Sha256Digest | None = None
    gpu_event_id: str | None = None
    query_access_event_hash: Sha256Digest | None = None
    construction_seal_hash: Sha256Digest | None = None
    prequery_preparation_bindings: tuple[PrequeryPreparationBinding, ...] = ()
    repair_parent_raw_output_hash: Sha256Digest | None = None
    allocated_gpu_seconds: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    failure_code: str | None = None
    completed_at: AwareDatetime
    included_in_intention_to_treat: Literal[True] = True

    @model_validator(mode="after")
    def terminal_itt_row(self) -> Self:
        if self.outcome not in _TERMINAL_OUTCOMES:
            raise ValueError("ITT rows require terminal outcomes")
        if self.request_start_state is RequestStartState.NOT_STARTED:
            if self.allocated_gpu_seconds or self.gpu_event_id or self.service_result_hash:
                raise ValueError("unstarted ITT row cannot claim a model-call receipt")
        elif self.request_start_state is RequestStartState.STARTED and (
            self.service_result_hash is None
            or self.ledger_receipt_hash is None
            or self.run_condition_config_hash is None
        ):
            raise ValueError("started ITT row requires durable result and ledger receipts")
        if self.outcome is RunOutcome.SUCCEEDED:
            if self.call_class == "development_c1":
                if (
                    self.condition_preparation_hash is None
                    or self.condition_attempt_hash is not None
                ):
                    raise ValueError("successful C1 row requires only a preparation artifact")
                if len(self.prequery_preparation_bindings) != 2:
                    raise ValueError("successful C1 row requires C1 and Fixed barrier bindings")
            elif self.condition_attempt_hash is None or self.condition_preparation_hash is not None:
                raise ValueError("successful query-time row requires only an attempt artifact")
        if self.outcome is not RunOutcome.SUCCEEDED and not self.failure_code:
            raise ValueError("nonsuccessful ITT row requires a failure code")
        if self.fixed_schema_derivation is not None and (
            self.fixed_schema_derivation.exact_run_condition_config_hash
            != self.run_condition_config_hash
        ):
            raise ValueError("ITT FixedSelect derivation and exact config hashes differ")
        fixed_started = (
            self.call_id.startswith("dev-fixed-")
            and self.request_start_state is RequestStartState.STARTED
        )
        if fixed_started != (self.fixed_schema_derivation is not None):
            raise ValueError("exactly FixedSelect ITT rows require a schema derivation")
        return self


class DevelopmentPhase(StrEnum):
    PRECONSTRUCTION_BARRIER_COMMITTED = "preconstruction_barrier_committed"
    C1_PRECONSTRUCTION = "c1_preconstruction"
    PREQUERY_BARRIER_COMMITTED = "prequery_barrier_committed"
    QUERY_EXECUTION = "query_execution"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"


class DevelopmentCheckpoint(ImmutableRecord):
    execution_id: str = Field(min_length=1)
    execution_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    source_plan_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    preconstruction_barrier: DevelopmentPreconstructionBarrier
    prequery_barrier: PrequeryBarrier | None = None
    c1_seal_receipts: tuple[C1SealReceipt, ...] = ()
    phase: DevelopmentPhase
    active_call_id: str | None = None
    itt_records: tuple[DevelopmentITTRecord, ...] = ()
    query_access_events: tuple[QueryAccessEvent, ...] = ()
    admission_failure_code: str | None = None
    allocated_gpu_seconds_at_start: float = Field(ge=0.0)
    observed_allocated_gpu_seconds: float = Field(ge=0.0)
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def state_is_prefix_and_monotonic(self) -> Self:
        ordinals = tuple(item.ordinal for item in self.itt_records)
        if ordinals != tuple(range(1, len(ordinals) + 1)):
            raise ValueError("checkpoint ITT rows must be one contiguous call prefix")
        call_ids = tuple(item.call_id for item in self.itt_records)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("checkpoint ITT call IDs must be unique")
        if self.active_call_id in set(call_ids):
            raise ValueError("checkpoint active call is already terminal")
        if self.observed_allocated_gpu_seconds < self.allocated_gpu_seconds_at_start:
            raise ValueError("checkpoint GPU allocation counter regressed")
        if (
            self.preconstruction_barrier.execution_id != self.execution_id
            or self.preconstruction_barrier.execution_manifest_hash != self.execution_manifest_hash
            or self.preconstruction_barrier.call_manifest_hash != self.call_manifest_hash
            or self.preconstruction_barrier.source_plan_hash != self.source_plan_hash
            or self.preconstruction_barrier.service_identity_hash != self.service_identity_hash
            or self.preconstruction_barrier.prequery_inputs_hash != self.prequery_inputs_hash
        ):
            raise ValueError("preconstruction barrier differs from checkpoint lineage")
        query_phases = {
            DevelopmentPhase.PREQUERY_BARRIER_COMMITTED,
            DevelopmentPhase.QUERY_EXECUTION,
            DevelopmentPhase.FINALIZING,
            DevelopmentPhase.COMPLETED,
            DevelopmentPhase.COMPLETED_WITH_FAILURES,
        }
        if (self.phase in query_phases) != (self.prequery_barrier is not None):
            raise ValueError("canonical prequery barrier presence disagrees with phase")
        if len(self.c1_seal_receipts) not in {0, 4}:
            raise ValueError("checkpoint must carry zero or four ordered C1 seal receipts")
        if (
            self.c1_seal_receipts
            and tuple(item.unit_id for item in self.c1_seal_receipts) != DEVELOPMENT_UNIT_IDS
        ):
            raise ValueError("C1 seal receipts must follow the registered unit order")
        if self.prequery_barrier is not None:
            if len(self.itt_records) < 4 or len(self.c1_seal_receipts) != 4:
                raise ValueError("canonical prequery barrier requires four terminal C1 rows")
            if (
                self.prequery_barrier.execution_id != self.execution_id
                or self.prequery_barrier.execution_manifest_hash != self.execution_manifest_hash
            ):
                raise ValueError("canonical prequery barrier differs from execution lineage")
            if tuple(item.itt_record_hash for item in self.c1_seal_receipts) != tuple(
                item.content_hash for item in self.itt_records[:4]
            ):
                raise ValueError("C1 receipts do not bind the first four ITT rows")
        receipt_stages = tuple(item.stage_manifest_hash for item in self.query_access_events)
        if len(receipt_stages) != len(set(receipt_stages)):
            raise ValueError("a checkpoint may contain only one access event per query stage")
        if self.prequery_barrier is None and self.query_access_events:
            raise ValueError("query access cannot precede the query barrier")
        if self.prequery_barrier is not None and any(
            item.prequery_barrier_hash != self.prequery_barrier.content_hash
            or item.execution_id != self.execution_id
            or item.accessed_at <= self.prequery_barrier.sealed_at
            for item in self.query_access_events
        ):
            raise ValueError("query-access event differs from its checkpoint barrier")
        return self


class ForecastControl(ImmutableRecord):
    forecast_receipt_hash: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    post_development_mandatory_forecast_seconds: float = Field(ge=0.0)
    complete_post_development_manifest_included: Literal[True] = True
    scheduled_limit_seconds: float = Field(default=9 * 3600, gt=0.0)
    hard_limit_seconds: float = Field(default=10 * 3600, gt=0.0)

    @model_validator(mode="after")
    def scheduled_precedes_hard(self) -> Self:
        if self.scheduled_limit_seconds >= self.hard_limit_seconds:
            raise ValueError("scheduled GPU limit must remain below the hard stop")
        return self


class DevelopmentScientificAssessment(ImmutableRecord):
    """Frozen development-gate measurements supplied by the semantic evaluator."""

    c0_explicit_family_coverage: float = Field(ge=0.0, le=1.0)
    c0_direct_assertion_precision: float = Field(ge=0.0, le=1.0)
    c0_direct_assertion_recall: float = Field(ge=0.0, le=1.0)
    c0_valid_evidence_reference_rate: float = Field(ge=0.0, le=1.0)
    c1_schema_valid: bool
    c1_all_construction_operators_exercised: bool
    c1_valid_evidence_id_rate: float = Field(ge=0.0, le=1.0)
    c1_grounding_precision: float = Field(ge=0.0, le=1.0)
    c1_union_gold_recall_after_seal: float = Field(ge=0.0, le=1.0)
    c2_construction_operator_present: bool
    programmed_horizon_leak_count: int = Field(ge=0)
    evidence_packets_equal: bool
    c1_query_blindness_verified: bool
    c2_empty_prequery_inventories_verified: bool
    fixed_complete_graph_packing_verified: bool
    fixed_constructive_operations_rejected: bool
    ablation_one_switch_verified: bool
    gold_firewall_verified: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.c0_explicit_family_coverage >= 1.0,
                self.c0_direct_assertion_precision >= 0.85,
                self.c0_direct_assertion_recall >= 0.70,
                self.c0_valid_evidence_reference_rate == 1.0,
                self.c1_schema_valid,
                self.c1_all_construction_operators_exercised,
                self.c1_valid_evidence_id_rate == 1.0,
                self.c1_grounding_precision >= 0.95,
                self.c1_union_gold_recall_after_seal >= 0.75,
                self.c2_construction_operator_present,
                self.programmed_horizon_leak_count == 0,
                self.evidence_packets_equal,
                self.c1_query_blindness_verified,
                self.c2_empty_prequery_inventories_verified,
                self.fixed_complete_graph_packing_verified,
                self.fixed_constructive_operations_rejected,
                self.ablation_one_switch_verified,
                self.gold_firewall_verified,
            )
        )


class TimingClassSummary(ImmutableRecord):
    call_class: str
    sample_count: int = Field(ge=0)
    p50_seconds: float | None = Field(default=None, ge=0.0)
    p95_seconds: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def percentiles_follow_sample_count(self) -> Self:
        both_absent = self.p50_seconds is None and self.p95_seconds is None
        if (self.sample_count == 0) != both_absent:
            raise ValueError("timing percentiles must be present exactly when samples exist")
        return self


class DevelopmentForecastResult(ImmutableRecord):
    forecast_receipt_hash: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    timing_by_call_class: tuple[TimingClassSummary, ...]
    actual_allocated_seconds_before: float = Field(ge=0.0)
    development_allocated_seconds: float = Field(ge=0.0)
    actual_allocated_seconds_after: float = Field(ge=0.0)
    post_development_mandatory_forecast_seconds: float = Field(ge=0.0)
    actual_plus_remaining_seconds: float = Field(ge=0.0)
    scheduled_limit_seconds: float = Field(gt=0.0)
    hard_limit_seconds: float = Field(gt=0.0)
    scheduled_admitted: bool
    below_hard_stop: bool

    @model_validator(mode="after")
    def arithmetic_reconciles(self) -> Self:
        if not math.isclose(
            self.development_allocated_seconds,
            self.actual_allocated_seconds_after - self.actual_allocated_seconds_before,
            abs_tol=1e-6,
        ):
            raise ValueError("development allocation does not reconcile with cumulative counters")
        expected = (
            self.actual_allocated_seconds_after + self.post_development_mandatory_forecast_seconds
        )
        if not math.isclose(self.actual_plus_remaining_seconds, expected, abs_tol=1e-6):
            raise ValueError("forecast total does not equal actual plus remaining")
        if self.scheduled_admitted != (expected <= self.scheduled_limit_seconds):
            raise ValueError("scheduled forecast admission flag is inconsistent")
        if self.below_hard_stop != (self.actual_allocated_seconds_after < self.hard_limit_seconds):
            raise ValueError("hard-stop status is inconsistent")
        return self


class DevelopmentGateResult(ImmutableRecord):
    exact_24_call_manifest: bool
    every_planned_call_has_itt_row: bool
    every_planned_request_started: bool
    every_planned_call_succeeded: bool
    twelve_query_access_events: bool
    same_live_service_identity: bool
    no_model_load_or_service_start_by_runner: Literal[True] = True
    forecast_admitted: bool
    scientific_assessment_hash: Sha256Digest | None = None
    scientific_thresholds_passed: bool
    passed: bool

    @model_validator(mode="after")
    def gate_is_conjunction(self) -> Self:
        expected = all(
            (
                self.exact_24_call_manifest,
                self.every_planned_call_has_itt_row,
                self.every_planned_request_started,
                self.every_planned_call_succeeded,
                self.twelve_query_access_events,
                self.same_live_service_identity,
                self.no_model_load_or_service_start_by_runner,
                self.forecast_admitted,
                self.scientific_thresholds_passed,
                self.scientific_assessment_hash is not None,
            )
        )
        if self.passed != expected:
            raise ValueError("development gate must equal its complete blocking conjunction")
        return self


class DevelopmentExecutionResult(ImmutableRecord):
    execution_id: str
    execution_manifest_hash: Sha256Digest
    call_manifest_hash: Sha256Digest
    source_plan_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    prequery_inputs_hash: Sha256Digest
    preconstruction_barrier_hash: Sha256Digest
    prequery_barrier_hash: Sha256Digest | None
    phase: DevelopmentPhase
    itt_records: tuple[DevelopmentITTRecord, ...]
    query_access_events: tuple[QueryAccessEvent, ...]
    admission_failure_code: str | None = None
    forecast: DevelopmentForecastResult
    gate: DevelopmentGateResult
    service_start_count_by_runner: Literal[0] = 0
    model_load_count_by_runner: Literal[0] = 0
    service_shutdown_by_runner: Literal[False] = False
    service_returned_live_to_owner: Literal[True] = True
    neutral_full_evidence_contract_supplied: Literal[True] = True


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("nearest-rank percentile requires observations")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _replace_record(record: DevelopmentCheckpoint, **updates: object) -> DevelopmentCheckpoint:
    payload = record.model_dump(mode="python", exclude={"content_hash"})
    payload.update(updates)
    return DevelopmentCheckpoint.model_validate(payload)


def _replace_itt_record(
    record: DevelopmentITTRecord,
    **updates: object,
) -> DevelopmentITTRecord:
    """Revalidate an immutable ITT update and recompute its content hash."""

    payload = record.model_dump(mode="python", exclude={"content_hash"})
    payload.update(updates)
    return DevelopmentITTRecord.model_validate(payload)


def _atomic_checkpoint(path: Path, checkpoint: DevelopmentCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(checkpoint.to_canonical_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_checkpoint(path: Path) -> DevelopmentCheckpoint:
    if path.is_symlink():
        raise DevelopmentResumeError("development checkpoint cannot be a symlink")
    return DevelopmentCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))


ScientificAssessmentProvider = Callable[
    [DevelopmentCallManifest, tuple[DevelopmentITTRecord, ...]],
    DevelopmentScientificAssessment | None,
]


class DevelopmentRunner:
    """Execute the exact development manifest against one injected live service."""

    def __init__(
        self,
        *,
        execution_id: str,
        manifest: DevelopmentCallManifest,
        prequery_inputs: DevelopmentPrequeryInputs,
        service: InjectedLiveDevelopmentService,
        checkpoint_path: Path,
        forecast_control: ForecastControl,
        assessment_provider: ScientificAssessmentProvider | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        _assert_safe_runtime_text("execution_id", execution_id)
        if not execution_id:
            raise ValueError("development execution_id must be nonempty")
        self.execution_id = execution_id
        self.manifest = manifest
        self.prequery_inputs = prequery_inputs
        self.service = service
        self.checkpoint_path = checkpoint_path
        _assert_safe_runtime_text("checkpoint_path", checkpoint_path.as_posix())
        self.forecast_control = forecast_control
        self.assessment_provider = assessment_provider
        self.clock = clock or (lambda: datetime.now(UTC))
        self._validate_prequery_inputs()
        if (
            self.forecast_control.gpu_call_inventory_file_sha256
            != self.manifest.gpu_call_inventory_file_sha256
        ):
            raise DevelopmentIntegrityError(
                "forecast control differs from the frozen GPU call inventory"
            )

    def _validate_prequery_inputs(self) -> None:
        for unit_id in DEVELOPMENT_UNIT_IDS:
            binding = self.prequery_inputs.binding_for(unit_id)
            neutral = next(
                item for item in self.manifest.neutral_evidence_stages if item.unit_id == unit_id
            )
            stages = {
                call.prequery_stage.evidence_artifact_hash
                for call in self.manifest.calls
                if call.unit_id == unit_id
            }
            if stages != {binding.staged_model_visible_evidence_hash}:
                raise DevelopmentIntegrityError(
                    "prequery evidence binding differs from the frozen model-visible stage"
                )
            expected_neutral_binding = (
                neutral.runtime_unit_id,
                neutral.snapshot_hash,
                neutral.model_visible_evidence_artifact_hash,
                neutral.neutral_evidence_artifact_hash,
                neutral.equivalence_certificate_hash,
            )
            observed_neutral_binding = (
                binding.runtime_unit_id,
                binding.snapshot_hash,
                binding.staged_model_visible_evidence_hash,
                binding.neutral_full_evidence_artifact_hash,
                binding.evidence_equivalence_certificate_hash,
            )
            if observed_neutral_binding != expected_neutral_binding:
                raise DevelopmentIntegrityError(
                    "prequery input differs from its verified neutral evidence certificate"
                )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("development runner clock must return an aware datetime")
        return value

    def _strictly_after(self, threshold: datetime) -> datetime:
        """Read a real/injected clock until it is strictly after a sealed event.

        A bounded retry accommodates coarse wall-clock resolution without ever
        inventing, backdating, or arithmetically advancing a scientific timestamp.
        """

        for _ in range(100):
            candidate = self._now()
            if candidate > threshold:
                return candidate
        raise DevelopmentIntegrityError(
            "clock did not advance strictly beyond the sealed predecessor timestamp"
        )

    def _actual_allocated_seconds(self) -> float:
        value = float(self.service.allocated_gpu_seconds())
        if not math.isfinite(value) or value < 0:
            raise DevelopmentIntegrityError("live service returned an invalid GPU counter")
        return value

    def _service_identity(self) -> LiveServiceIdentity:
        identity = self.service.identity()
        if identity.selected_model_freeze_hash != self.prequery_inputs.selected_model_freeze_hash:
            raise DevelopmentIntegrityError(
                "injected service differs from the selected-model freeze"
            )
        if identity.source_execution_hash != self.prequery_inputs.source_tree_hash:
            raise DevelopmentIntegrityError("injected service differs from the source-tree binding")
        if not identity.already_running or identity.runner_lifecycle_authority:
            raise DevelopmentIntegrityError("development runner requires a borrowed live service")
        return identity

    def _execution_manifest(
        self,
        identity: LiveServiceIdentity,
    ) -> DevelopmentExecutionManifest:
        return DevelopmentExecutionManifest(
            execution_id=self.execution_id,
            call_manifest_hash=self.manifest.content_hash,
            source_plan_hash=self.manifest.source_plan_hash,
            prequery_inputs_hash=self.prequery_inputs.content_hash,
            service_identity_hash=identity.content_hash,
            forecast_control_hash=self.forecast_control.content_hash,
        )

    def _new_checkpoint(self, identity: LiveServiceIdentity) -> DevelopmentCheckpoint:
        execution_manifest = self._execution_manifest(identity)
        latest_preparation = max(
            binding.completed_at for binding in self.prequery_inputs.unit_bindings
        )
        recorded_at = self._strictly_after(latest_preparation)
        barrier = DevelopmentPreconstructionBarrier(
            barrier_id=f"{self.execution_id}-preconstruction-barrier",
            execution_id=self.execution_id,
            execution_manifest_hash=execution_manifest.content_hash,
            call_manifest_hash=self.manifest.content_hash,
            source_plan_hash=self.manifest.source_plan_hash,
            service_identity_hash=identity.content_hash,
            prequery_inputs_hash=self.prequery_inputs.content_hash,
            unit_binding_hashes=tuple(
                item.content_hash for item in self.prequery_inputs.unit_bindings
            ),
            recorded_at=recorded_at,
        )
        actual = self._actual_allocated_seconds()
        checkpoint = DevelopmentCheckpoint(
            execution_id=self.execution_id,
            execution_manifest_hash=execution_manifest.content_hash,
            call_manifest_hash=self.manifest.content_hash,
            source_plan_hash=self.manifest.source_plan_hash,
            service_identity_hash=identity.content_hash,
            prequery_inputs_hash=self.prequery_inputs.content_hash,
            preconstruction_barrier=barrier,
            phase=DevelopmentPhase.PRECONSTRUCTION_BARRIER_COMMITTED,
            allocated_gpu_seconds_at_start=actual,
            observed_allocated_gpu_seconds=actual,
            updated_at=recorded_at,
        )
        _atomic_checkpoint(self.checkpoint_path, checkpoint)
        return checkpoint

    def _load_or_create_checkpoint(
        self,
        identity: LiveServiceIdentity,
    ) -> DevelopmentCheckpoint:
        if not self.checkpoint_path.exists():
            return self._new_checkpoint(identity)
        checkpoint = _read_checkpoint(self.checkpoint_path)
        execution_manifest = self._execution_manifest(identity)
        expected = (
            self.execution_id,
            execution_manifest.content_hash,
            self.manifest.content_hash,
            self.manifest.source_plan_hash,
            identity.content_hash,
            self.prequery_inputs.content_hash,
        )
        observed = (
            checkpoint.execution_id,
            checkpoint.execution_manifest_hash,
            checkpoint.call_manifest_hash,
            checkpoint.source_plan_hash,
            checkpoint.service_identity_hash,
            checkpoint.prequery_inputs_hash,
        )
        if observed != expected:
            raise DevelopmentResumeError(
                "development checkpoint differs from the execution, manifest, inputs, or service"
            )
        expected_prefix = tuple(
            item.call_id for item in self.manifest.calls[: len(checkpoint.itt_records)]
        )
        if tuple(item.call_id for item in checkpoint.itt_records) != expected_prefix:
            raise DevelopmentResumeError("development checkpoint is not the frozen call prefix")
        actual = self._actual_allocated_seconds()
        if actual < checkpoint.observed_allocated_gpu_seconds:
            raise DevelopmentResumeError("allocated GPU counter regressed across controller resume")
        return checkpoint

    def _save(
        self,
        checkpoint: DevelopmentCheckpoint,
        **updates: object,
    ) -> DevelopmentCheckpoint:
        updates.setdefault("updated_at", self._now())
        updated = _replace_record(checkpoint, **updates)
        _atomic_checkpoint(self.checkpoint_path, updated)
        return updated

    def _commit_prequery_barrier(
        self,
        checkpoint: DevelopmentCheckpoint,
    ) -> DevelopmentCheckpoint:
        if checkpoint.prequery_barrier is not None:
            return checkpoint
        if len(checkpoint.itt_records) < 4:
            raise DevelopmentIntegrityError("prequery barrier cannot precede four C1 ITT rows")
        if checkpoint.query_access_events:
            raise DevelopmentIntegrityError("prequery barrier cannot follow any query access")
        c1_rows = checkpoint.itt_records[:4]
        if any(
            call.kind is not DevelopmentCallKind.C1_PRECONSTRUCTION
            for call in self.manifest.calls[:4]
        ):
            raise DevelopmentIntegrityError("development manifest does not begin with C1")
        receipts = tuple(
            C1SealReceipt(
                unit_id=row.unit_id,
                call_id=row.call_id,
                itt_record_hash=row.content_hash,
                outcome=row.outcome,
                construction_seal_hash=row.construction_seal_hash,
                preparation_bindings=row.prequery_preparation_bindings,
                completed_at=row.completed_at,
            )
            for row in c1_rows
        )
        preparation_bindings = tuple(
            binding
            for unit in self.prequery_inputs.unit_bindings
            for binding in unit.preexisting_preparation_bindings
        ) + tuple(binding for receipt in receipts for binding in receipt.preparation_bindings)
        latest_prequery_event = max(
            (
                *(row.completed_at for row in c1_rows),
                *(item.completed_at for item in preparation_bindings),
            )
        )
        sealed_at = self._strictly_after(latest_prequery_event)
        barrier = PrequeryBarrier(
            barrier_id=f"{self.execution_id}-prequery-barrier",
            execution_id=self.execution_id,
            execution_manifest_hash=checkpoint.execution_manifest_hash,
            neutral_evidence_artifact_hashes=tuple(
                item.staged_model_visible_evidence_hash
                for item in self.prequery_inputs.unit_bindings
            ),
            preparation_bindings=preparation_bindings,
            sealed_at=sealed_at,
        )
        return self._save(
            checkpoint,
            prequery_barrier=barrier,
            c1_seal_receipts=receipts,
            phase=DevelopmentPhase.PREQUERY_BARRIER_COMMITTED,
        )

    @staticmethod
    def _access_receipt_for(
        checkpoint: DevelopmentCheckpoint,
        stage: StageReference,
    ) -> QueryAccessEvent | None:
        for event in checkpoint.query_access_events:
            if event.stage_manifest_hash == stage.staging_manifest_hash:
                return event
        return None

    def _ensure_query_access(
        self,
        checkpoint: DevelopmentCheckpoint,
        call: DevelopmentCallSpec,
    ) -> tuple[DevelopmentCheckpoint, QueryAccessEvent]:
        stage = call.query_stage
        barrier = checkpoint.prequery_barrier
        if stage is None or barrier is None or stage.query_artifact_hash is None:
            raise DevelopmentIntegrityError("query-time call lacks its committed query barrier")
        existing = self._access_receipt_for(checkpoint, stage)
        if existing is not None:
            return checkpoint, existing
        event = self.service.open_query(stage, barrier)
        unit_binding = self.prequery_inputs.binding_for(call.unit_id)
        if (
            event.execution_id != self.execution_id
            or event.stage_manifest_hash != stage.staging_manifest_hash
            or event.query_artifact_hash != stage.query_artifact_hash
            or event.snapshot_hash != unit_binding.snapshot_hash
            or event.prequery_barrier_hash != barrier.content_hash
        ):
            raise DevelopmentIntegrityError("query-access event differs from its frozen stage")
        if event.accessed_at <= barrier.sealed_at:
            raise DevelopmentIntegrityError(
                "physical query access did not follow the query barrier"
            )
        checkpoint = self._save(
            checkpoint,
            query_access_events=(*checkpoint.query_access_events, event),
        )
        return checkpoint, event

    @staticmethod
    def _row_by_call(
        checkpoint: DevelopmentCheckpoint,
        call_id: str,
    ) -> DevelopmentITTRecord | None:
        return next((item for item in checkpoint.itt_records if item.call_id == call_id), None)

    def _dependency_failure_code(
        self,
        checkpoint: DevelopmentCheckpoint,
        call: DevelopmentCallSpec,
    ) -> str | None:
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            source = self._row_by_call(checkpoint, call.source_c1_call_id or "")
            if (
                source is None
                or source.outcome is not RunOutcome.SUCCEEDED
                or source.construction_seal_hash is None
            ):
                return "same_unit_c1_seal_unavailable"
        if call.kind is DevelopmentCallKind.REPAIR_PROBE:
            parent = self._row_by_call(checkpoint, call.parent_call_id or "")
            if (
                parent is None
                or parent.outcome is not RunOutcome.SUCCEEDED
                or parent.response_artifact_hash is None
            ):
                return "repair_probe_parent_output_unavailable"
        return None

    def _envelope(
        self,
        checkpoint: DevelopmentCheckpoint,
        call: DevelopmentCallSpec,
        access: QueryAccessEvent | None,
    ) -> CallExecutionEnvelope:
        source_seal = None
        source_output = None
        if call.source_c1_call_id is not None:
            row = self._row_by_call(checkpoint, call.source_c1_call_id)
            source_seal = None if row is None else row.construction_seal_hash
            source_output = None if row is None else row.response_artifact_hash
        parent_hash = None
        parent_output = None
        if call.parent_call_id is not None:
            row = self._row_by_call(checkpoint, call.parent_call_id)
            parent_hash = None if row is None else row.content_hash
            parent_output = None if row is None else row.response_artifact_hash
        fixed_plan = (
            self.prequery_inputs.fixed_schema_plan_for(call.ordinal)
            if call.kind is DevelopmentCallKind.FIXED_SELECTION
            else None
        )
        return CallExecutionEnvelope(
            execution_id=self.execution_id,
            execution_manifest_hash=checkpoint.execution_manifest_hash,
            call_manifest_hash=self.manifest.content_hash,
            prequery_inputs_hash=self.prequery_inputs.content_hash,
            expected_run_condition_config_hash=(
                None
                if fixed_plan is not None
                else self.prequery_inputs.run_config_hash_for(call.ordinal)
            ),
            fixed_schema_derivation_plan_hash=(
                None if fixed_plan is None else fixed_plan.content_hash
            ),
            preconstruction_barrier_hash=checkpoint.preconstruction_barrier.content_hash,
            preconstruction_barrier_recorded_at=(checkpoint.preconstruction_barrier.recorded_at),
            prequery_barrier_hash=(
                None
                if call.query_stage is None or checkpoint.prequery_barrier is None
                else checkpoint.prequery_barrier.content_hash
            ),
            query_access_event=access,
            unit_prequery_binding_hash=self.prequery_inputs.binding_for(call.unit_id).content_hash,
            source_c1_seal_hash=source_seal,
            source_c1_output_artifact_hash=source_output,
            parent_itt_record_hash=parent_hash,
            parent_output_artifact_hash=parent_output,
            service_identity_hash=checkpoint.service_identity_hash,
        )

    def _validate_service_result(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
        result: ServiceCallResult,
    ) -> None:
        if result.call_id != call.call_id:
            raise DevelopmentIntegrityError("service returned a result for another call")
        if result.service_identity_hash != envelope.service_identity_hash:
            raise DevelopmentIntegrityError("service result cites another model process")
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            plan = self.prequery_inputs.fixed_schema_plan_for(call.ordinal)
            derivation = result.fixed_schema_derivation
            access = envelope.query_access_event
            if (
                envelope.expected_run_condition_config_hash is not None
                or envelope.fixed_schema_derivation_plan_hash != plan.content_hash
                or derivation is None
                or derivation.plan_hash != plan.content_hash
                or derivation.call_id != call.call_id
                or derivation.unit_id != call.unit_id
                or derivation.source_c1_construction_seal_hash
                != envelope.source_c1_seal_hash
                or access is None
                or derivation.derived_at >= access.accessed_at
            ):
                raise DevelopmentIntegrityError(
                    "FixedSelect exact config was not deterministically derived pre-query"
                )
        elif (
            result.run_condition_config_hash
            != envelope.expected_run_condition_config_hash
            or envelope.fixed_schema_derivation_plan_hash is not None
            or result.fixed_schema_derivation is not None
        ):
            raise DevelopmentIntegrityError("service result used another RunConditionConfig")
        if not result.request_started:
            raise DevelopmentIntegrityError("execute_call returned without starting its request")
        if result.completed_at <= envelope.preconstruction_barrier_recorded_at:
            raise DevelopmentIntegrityError(
                "model-call completion predates its preconstruction seal"
            )
        access = envelope.query_access_event
        if call.query_stage is None:
            if (
                envelope.prequery_barrier_hash is not None
                or access is not None
                or result.query_access_event_hash is not None
            ):
                raise DevelopmentIntegrityError("C1 result cannot claim query access")
        elif (
            envelope.prequery_barrier_hash is None
            or access is None
            or result.query_access_event_hash != access.content_hash
        ):
            raise DevelopmentIntegrityError("query-time result lacks its exact access event")
        elif result.completed_at < access.accessed_at:
            raise DevelopmentIntegrityError("query-time result completed before query access")
        if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION:
            if (result.outcome is RunOutcome.SUCCEEDED) != (
                result.construction_seal_hash is not None
            ):
                raise DevelopmentIntegrityError("only successful C1 calls may return a seal")
            if result.outcome is RunOutcome.SUCCEEDED:
                binding = self.prequery_inputs.binding_for(call.unit_id)
                preparations = result.prequery_preparation_bindings
                if tuple(item.condition for item in preparations) != (
                    ConditionName.C1_LLM_PRE,
                    ConditionName.A_FIXED_SELECT,
                ):
                    raise DevelopmentIntegrityError(
                        "successful C1 result lacks exact C1/Fixed preparation bindings"
                    )
                if any(
                    item.unit_id != binding.runtime_unit_id
                    or item.snapshot_hash != binding.snapshot_hash
                    or item.seed_block != call.seed_block
                    or item.completed_at > result.completed_at
                    for item in preparations
                ):
                    raise DevelopmentIntegrityError(
                        "C1/Fixed preparation binding differs from its unit, seed, or timing"
                    )
                if (
                    result.condition_preparation_hash != preparations[0].preparation_hash
                    or result.construction_seal_hash != preparations[0].lineage_artifact_hash
                ):
                    raise DevelopmentIntegrityError(
                        "C1 result preparation/seal hashes differ from its barrier binding"
                    )
        elif result.construction_seal_hash is not None:
            raise DevelopmentIntegrityError("only C1 calls may create a pre-query seal")
        elif result.prequery_preparation_bindings:
            raise DevelopmentIntegrityError("query-time result cannot add prequery preparations")
        if call.kind is DevelopmentCallKind.REPAIR_PROBE:
            if result.repair_parent_raw_output_hash != envelope.parent_output_artifact_hash:
                raise DevelopmentIntegrityError(
                    "repair result differs from its preserved raw parent"
                )
        elif result.repair_parent_raw_output_hash is not None:
            raise DevelopmentIntegrityError("only the repair probe may cite a raw repair parent")

    @staticmethod
    def _itt_from_result(
        call: DevelopmentCallSpec,
        result: ServiceCallResult,
    ) -> DevelopmentITTRecord:
        return DevelopmentITTRecord(
            ordinal=call.ordinal,
            call_id=call.call_id,
            call_class=call.call_class,
            condition=call.condition,
            unit_id=call.unit_id,
            outcome=result.outcome,
            request_start_state=RequestStartState.STARTED,
            service_result_hash=result.content_hash,
            response_artifact_hash=result.response_artifact_hash,
            validated_generation_hash=result.validated_generation_hash,
            validation_record_hash=result.validation_record_hash,
            condition_attempt_hash=result.condition_attempt_hash,
            condition_preparation_hash=result.condition_preparation_hash,
            run_condition_config_hash=result.run_condition_config_hash,
            fixed_schema_derivation=result.fixed_schema_derivation,
            ledger_receipt_hash=result.ledger_receipt_hash,
            gpu_event_id=result.gpu_event_id,
            query_access_event_hash=result.query_access_event_hash,
            construction_seal_hash=result.construction_seal_hash,
            prequery_preparation_bindings=result.prequery_preparation_bindings,
            repair_parent_raw_output_hash=result.repair_parent_raw_output_hash,
            allocated_gpu_seconds=result.allocated_gpu_seconds,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            failure_code=result.failure_code,
            completed_at=result.completed_at,
        )

    def _unstarted_itt(
        self,
        call: DevelopmentCallSpec,
        failure_code: str,
    ) -> DevelopmentITTRecord:
        return DevelopmentITTRecord(
            ordinal=call.ordinal,
            call_id=call.call_id,
            call_class=call.call_class,
            condition=call.condition,
            unit_id=call.unit_id,
            outcome=RunOutcome.FAILED,
            request_start_state=RequestStartState.NOT_STARTED,
            allocated_gpu_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            failure_code=failure_code,
            completed_at=self._now(),
        )

    def _interrupted_itt(self, call: DevelopmentCallSpec) -> DevelopmentITTRecord:
        return DevelopmentITTRecord(
            ordinal=call.ordinal,
            call_id=call.call_id,
            call_class=call.call_class,
            condition=call.condition,
            unit_id=call.unit_id,
            outcome=RunOutcome.INTERRUPTED,
            request_start_state=RequestStartState.UNKNOWN_AFTER_INTERRUPTION,
            allocated_gpu_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            failure_code="controller_interrupted_without_durable_call_receipt",
            completed_at=self._now(),
        )

    def _append_itt(
        self,
        checkpoint: DevelopmentCheckpoint,
        row: DevelopmentITTRecord,
    ) -> DevelopmentCheckpoint:
        expected = self.manifest.calls[len(checkpoint.itt_records)]
        if (row.ordinal, row.call_id) != (expected.ordinal, expected.call_id):
            raise DevelopmentIntegrityError("ITT append differs from the next frozen call")
        actual = self._actual_allocated_seconds()
        if actual < checkpoint.observed_allocated_gpu_seconds:
            raise DevelopmentIntegrityError("GPU allocation counter regressed during development")
        return self._save(
            checkpoint,
            active_call_id=None,
            itt_records=(*checkpoint.itt_records, row),
            observed_allocated_gpu_seconds=actual,
        )

    def _recover_active_call(
        self,
        checkpoint: DevelopmentCheckpoint,
    ) -> DevelopmentCheckpoint:
        active = checkpoint.active_call_id
        if active is None:
            return checkpoint
        if len(checkpoint.itt_records) >= len(self.manifest.calls):
            raise DevelopmentResumeError("terminal checkpoint still names an active call")
        call = self.manifest.calls[len(checkpoint.itt_records)]
        if call.call_id != active:
            raise DevelopmentResumeError("active checkpoint call is not the next frozen call")
        access = None
        if call.query_stage is not None:
            access = self._access_receipt_for(checkpoint, call.query_stage)
            if access is None:
                raise DevelopmentResumeError("active query call lacks its persisted access receipt")
        envelope = self._envelope(checkpoint, call, access)
        recovered = self.service.recover_call(call, envelope)
        if recovered is None:
            return self._append_itt(checkpoint, self._interrupted_itt(call))
        self._validate_service_result(call, envelope, recovered)
        return self._append_itt(checkpoint, self._itt_from_result(call, recovered))

    def _rehydrate_completed_prefix(
        self,
        checkpoint: DevelopmentCheckpoint,
    ) -> DevelopmentCheckpoint:
        """Rebuild in-memory state solely from audited events and durable call receipts.

        A controller restart creates a fresh adapter/executor object even though the
        model process remains live.  Checkpoint rows alone are therefore insufficient:
        later FixedSelect and repair calls depend on C1 preparations, derived schemas,
        and packet materializations.  This method validates every persisted prefix row
        against the exact service receipt and asks the adapter to reconstruct those
        dependencies without reissuing inference.
        """

        barrier = checkpoint.prequery_barrier
        stage_by_hash = {
            call.query_stage.staging_manifest_hash: call.query_stage
            for call in self.manifest.calls
            if call.query_stage is not None
        }
        if checkpoint.query_access_events and barrier is None:
            raise DevelopmentResumeError("persisted query events lack a prequery barrier")
        for persisted in checkpoint.query_access_events:
            stage = stage_by_hash.get(persisted.stage_manifest_hash)
            if stage is None:
                raise DevelopmentResumeError(
                    "checkpoint contains an unregistered query-access stage"
                )
            opened = self.service.open_query(stage, cast(PrequeryBarrier, barrier))
            if opened != persisted:
                raise DevelopmentResumeError(
                    "reopened query event differs from the persisted checkpoint"
                )

        for call, row in zip(
            self.manifest.calls[: len(checkpoint.itt_records)],
            checkpoint.itt_records,
            strict=True,
        ):
            if row.request_start_state is not RequestStartState.STARTED:
                continue
            access = None
            if call.query_stage is not None:
                access = self._access_receipt_for(checkpoint, call.query_stage)
                if access is None:
                    raise DevelopmentResumeError(
                        "completed query call lacks its persisted access event"
                    )
            envelope = self._envelope(checkpoint, call, access)
            recovered = self.service.recover_call(call, envelope)
            if recovered is None:
                raise DevelopmentResumeError(
                    "completed checkpoint call lacks a durable service receipt"
                )
            self._validate_service_result(call, envelope, recovered)
            if self._itt_from_result(call, recovered) != row:
                raise DevelopmentResumeError(
                    "durable service receipt differs from its checkpoint ITT row"
                )
        return checkpoint

    def _remaining_admission_p95_seconds(self, checkpoint: DevelopmentCheckpoint) -> float:
        return float(
            sum(
                call.admission_p95_seconds
                for call in self.manifest.calls[len(checkpoint.itt_records) :]
            )
        )

    def _admission_failure_code(self, checkpoint: DevelopmentCheckpoint) -> str | None:
        actual = self._actual_allocated_seconds()
        if actual >= self.forecast_control.hard_limit_seconds:
            return "hard_gpu_stop_reached"
        required = (
            actual
            + self._remaining_admission_p95_seconds(checkpoint)
            + self.forecast_control.post_development_mandatory_forecast_seconds
        )
        if required > self.forecast_control.scheduled_limit_seconds:
            return "mandatory_manifest_forecast_not_admitted"
        return None

    def _close_remaining_without_requests(
        self,
        checkpoint: DevelopmentCheckpoint,
        failure_code: str,
    ) -> DevelopmentCheckpoint:
        checkpoint = self._save(checkpoint, admission_failure_code=failure_code)
        while len(checkpoint.itt_records) < len(self.manifest.calls):
            call = self.manifest.calls[len(checkpoint.itt_records)]
            checkpoint = self._append_itt(
                checkpoint,
                self._unstarted_itt(call, failure_code),
            )
        return checkpoint

    def run(self) -> DevelopmentExecutionResult:
        """Run or safely resume the exact prefix; never start or stop the service."""

        identity = self._service_identity()
        checkpoint = self._load_or_create_checkpoint(identity)
        if checkpoint.phase in {
            DevelopmentPhase.COMPLETED,
            DevelopmentPhase.COMPLETED_WITH_FAILURES,
        }:
            self._rehydrate_completed_prefix(checkpoint)
            return self._result(checkpoint, identity)
        checkpoint = self._rehydrate_completed_prefix(checkpoint)
        checkpoint = self._recover_active_call(checkpoint)

        while len(checkpoint.itt_records) < len(self.manifest.calls):
            if len(checkpoint.itt_records) >= 4 and checkpoint.prequery_barrier is None:
                checkpoint = self._commit_prequery_barrier(checkpoint)
            call = self.manifest.calls[len(checkpoint.itt_records)]
            phase = (
                DevelopmentPhase.C1_PRECONSTRUCTION
                if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
                else DevelopmentPhase.QUERY_EXECUTION
            )
            checkpoint = self._save(checkpoint, phase=phase)

            admission_failure = self._admission_failure_code(checkpoint)
            if admission_failure is not None:
                checkpoint = self._close_remaining_without_requests(
                    checkpoint,
                    admission_failure,
                )
                break
            dependency_failure = self._dependency_failure_code(checkpoint, call)
            if dependency_failure is not None:
                checkpoint = self._append_itt(
                    checkpoint,
                    self._unstarted_itt(call, dependency_failure),
                )
                continue

            access = None
            if call.query_stage is not None:
                checkpoint, access = self._ensure_query_access(checkpoint, call)
            envelope = self._envelope(checkpoint, call, access)
            checkpoint = self._save(checkpoint, active_call_id=call.call_id)
            try:
                result = self.service.execute_call(call, envelope)
            except TimeoutError:
                checkpoint = self._append_itt(
                    checkpoint,
                    _replace_itt_record(
                        self._interrupted_itt(call),
                        outcome=RunOutcome.TIMED_OUT,
                        failure_code="service_timeout_without_durable_call_receipt",
                    ),
                )
                continue
            except Exception:
                checkpoint = self._append_itt(
                    checkpoint,
                    _replace_itt_record(
                        self._interrupted_itt(call),
                        outcome=RunOutcome.FAILED,
                        failure_code="service_failure_without_durable_call_receipt",
                    ),
                )
                continue
            self._validate_service_result(call, envelope, result)
            checkpoint = self._append_itt(
                checkpoint,
                self._itt_from_result(call, result),
            )

        if checkpoint.prequery_barrier is None and len(checkpoint.itt_records) >= 4:
            checkpoint = self._commit_prequery_barrier(checkpoint)
        checkpoint = self._save(checkpoint, phase=DevelopmentPhase.FINALIZING)
        final_phase = (
            DevelopmentPhase.COMPLETED
            if all(item.outcome is RunOutcome.SUCCEEDED for item in checkpoint.itt_records)
            else DevelopmentPhase.COMPLETED_WITH_FAILURES
        )
        checkpoint = self._save(checkpoint, phase=final_phase)
        return self._result(checkpoint, identity)

    def _timing_summaries(
        self,
        rows: Sequence[DevelopmentITTRecord],
    ) -> tuple[TimingClassSummary, ...]:
        summaries: list[TimingClassSummary] = []
        for call_class in DEVELOPMENT_CLASS_COUNTS:
            values = [
                item.allocated_gpu_seconds
                for item in rows
                if item.call_class == call_class
                and item.request_start_state is RequestStartState.STARTED
            ]
            summaries.append(
                TimingClassSummary(
                    call_class=call_class,
                    sample_count=len(values),
                    p50_seconds=None if not values else _nearest_rank(values, 0.50),
                    p95_seconds=None if not values else _nearest_rank(values, 0.95),
                )
            )
        return tuple(summaries)

    def _result(
        self,
        checkpoint: DevelopmentCheckpoint,
        initial_identity: LiveServiceIdentity,
    ) -> DevelopmentExecutionResult:
        current_identity = self._service_identity()
        same_identity = current_identity == initial_identity
        actual_after = self._actual_allocated_seconds()
        if actual_after < checkpoint.allocated_gpu_seconds_at_start:
            raise DevelopmentIntegrityError("final allocated GPU counter regressed")
        remaining = self.forecast_control.post_development_mandatory_forecast_seconds
        forecast = DevelopmentForecastResult(
            forecast_receipt_hash=self.forecast_control.forecast_receipt_hash,
            gpu_call_inventory_file_sha256=(self.forecast_control.gpu_call_inventory_file_sha256),
            timing_by_call_class=self._timing_summaries(checkpoint.itt_records),
            actual_allocated_seconds_before=checkpoint.allocated_gpu_seconds_at_start,
            development_allocated_seconds=(
                actual_after - checkpoint.allocated_gpu_seconds_at_start
            ),
            actual_allocated_seconds_after=actual_after,
            post_development_mandatory_forecast_seconds=remaining,
            actual_plus_remaining_seconds=actual_after + remaining,
            scheduled_limit_seconds=self.forecast_control.scheduled_limit_seconds,
            hard_limit_seconds=self.forecast_control.hard_limit_seconds,
            scheduled_admitted=(
                actual_after + remaining <= self.forecast_control.scheduled_limit_seconds
            ),
            below_hard_stop=actual_after < self.forecast_control.hard_limit_seconds,
        )
        assessment = (
            None
            if self.assessment_provider is None
            else self.assessment_provider(self.manifest, checkpoint.itt_records)
        )
        exact_itt = len(checkpoint.itt_records) == DEVELOPMENT_CALL_COUNT and tuple(
            item.call_id for item in checkpoint.itt_records
        ) == tuple(item.call_id for item in self.manifest.calls)
        every_started = exact_itt and all(
            item.request_start_state is RequestStartState.STARTED for item in checkpoint.itt_records
        )
        every_succeeded = exact_itt and all(
            item.outcome is RunOutcome.SUCCEEDED for item in checkpoint.itt_records
        )
        access_complete = (
            len(checkpoint.query_access_events) == 12
            and len({item.stage_manifest_hash for item in checkpoint.query_access_events}) == 12
        )
        scientific_passed = assessment is not None and assessment.passed
        gate_values = {
            "exact_24_call_manifest": len(self.manifest.calls) == DEVELOPMENT_CALL_COUNT,
            "every_planned_call_has_itt_row": exact_itt,
            "every_planned_request_started": every_started,
            "every_planned_call_succeeded": every_succeeded,
            "twelve_query_access_events": access_complete,
            "same_live_service_identity": same_identity,
            "forecast_admitted": (
                checkpoint.admission_failure_code is None
                and forecast.scheduled_admitted
                and forecast.below_hard_stop
            ),
            "scientific_assessment_hash": None if assessment is None else assessment.content_hash,
            "scientific_thresholds_passed": scientific_passed,
        }
        gate = DevelopmentGateResult(
            **gate_values,
            passed=all(
                (
                    *(
                        bool(value)
                        for key, value in gate_values.items()
                        if key != "scientific_assessment_hash"
                    ),
                    gate_values["scientific_assessment_hash"] is not None,
                )
            ),
        )
        return DevelopmentExecutionResult(
            execution_id=self.execution_id,
            execution_manifest_hash=checkpoint.execution_manifest_hash,
            call_manifest_hash=self.manifest.content_hash,
            source_plan_hash=self.manifest.source_plan_hash,
            service_identity_hash=initial_identity.content_hash,
            prequery_inputs_hash=self.prequery_inputs.content_hash,
            preconstruction_barrier_hash=checkpoint.preconstruction_barrier.content_hash,
            prequery_barrier_hash=(
                None
                if checkpoint.prequery_barrier is None
                else checkpoint.prequery_barrier.content_hash
            ),
            phase=checkpoint.phase,
            itt_records=checkpoint.itt_records,
            query_access_events=checkpoint.query_access_events,
            admission_failure_code=checkpoint.admission_failure_code,
            forecast=forecast,
            gate=gate,
        )


def manifest_public_summary(manifest: DevelopmentCallManifest) -> dict[str, Any]:
    """Return a compact public plan with no query text or scorer routing."""

    return {
        "schema_version": manifest.schema_version,
        "kind": "phase3_development_call_manifest",
        "plan_id": manifest.plan_id,
        "manifest_sha256": manifest.content_hash,
        "source_plan_sha256": manifest.source_plan_hash,
        "seed_block": manifest.seed_block,
        "vllm_seed": manifest.vllm_seed,
        "call_count": len(manifest.calls),
        "call_class_counts": dict(
            sorted(Counter(item.call_class for item in manifest.calls).items())
        ),
        "neutral_evidence_stages": [
            {
                "unit_id": item.unit_id,
                "stage_manifest_sha256": item.staging_manifest_hash,
                "neutral_evidence_sha256": item.neutral_evidence_artifact_hash,
                "equivalence_certificate_sha256": item.equivalence_certificate_hash,
                "model_visible_evidence_sha256": (item.model_visible_evidence_artifact_hash),
            }
            for item in manifest.neutral_evidence_stages
        ],
        "calls": [
            {
                "ordinal": item.ordinal,
                "call_id": item.call_id,
                "call_class": item.call_class,
                "condition": item.condition.value,
                "unit_id": item.unit_id,
                "stage_manifest_sha256": (
                    item.prequery_stage.staging_manifest_hash
                    if item.query_stage is None
                    else item.query_stage.staging_manifest_hash
                ),
                "source_c1_call_id": item.source_c1_call_id,
                "parent_call_id": item.parent_call_id,
                "configuration_delta_hash": (
                    None
                    if item.configuration_delta is None
                    else item.configuration_delta.content_hash
                ),
                "watchdog_seconds": item.watchdog_seconds,
                "admission_p95_seconds": item.admission_p95_seconds,
            }
            for item in manifest.calls
        ],
        "model_service_start_permitted": False,
        "model_load_permitted": False,
        "model_service_shutdown_permitted": False,
        "query_json_read_during_manifest_load": False,
    }


def canonical_public_summary(manifest: DevelopmentCallManifest) -> str:
    return canonical_json(manifest_public_summary(manifest))
