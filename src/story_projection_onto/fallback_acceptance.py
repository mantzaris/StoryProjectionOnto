"""Certified, reserve-charged Phase-1 fallback micro-pilot.

Importing this module is CPU-only.  Its CLI defaults to a static public plan and
requires ``--execute`` plus activation, cache-replacement, snapshot, ledger, and
resource evidence before it may start the single pinned fallback service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self, cast, runtime_checkable

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    ConditionName,
    ImmutableRecord,
    RunOutcome,
    Sha256Digest,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.development_runtime import (
    DevelopmentCallManifest,
    DevelopmentCheckpoint,
    DevelopmentExecutionResult,
    DevelopmentPhase,
    DevelopmentPrequeryInputs,
    ForecastControl,
    InjectedLiveDevelopmentService,
    LiveServiceIdentity,
    RequestStartState,
)
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    GPUCallInventory,
    ResourceLimits,
    StorageAllocationPlan,
    TimingObservation,
    forecast_gpu_schedule,
    summarize_call_class_timings,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    ChatMessage,
    GenerationResult,
    GPUHardwareIdentity,
    GuidedJSONRequest,
    ResourceSampler,
    ResourceWatchdog,
    RuntimeResourceLimitExceeded,
    RuntimeStackManifest,
    ServiceState,
    ServiceUptime,
    TokenizerManifest,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    atomic_write_public_json,
    capture_gpu_hardware_identity,
    capture_runtime_stack,
    capture_tokenizer_manifest,
    public_runtime_manifest,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    DecodingManifest,
    FixedSelectCapabilityError,
    PackingReport,
    PackingSection,
)
from story_projection_onto.manifest import build_source_manifest
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    validate_fallback_activation_certificate,
    validate_fallback_cache_replacement_receipt,
    validate_fallback_snapshot_manifest,
)
from story_projection_onto.phase1_acceptance import (
    SCHEMA_VERSION,
    AcceptanceCall,
    AcceptanceStatus,
    PackingTokenizer,
    build_acceptance_request,
    validate_acceptance_generation,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    FailureKind,
    GpuEventKind,
    GpuSummary,
    Ledger,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
    StorageBudget,
    StorageBudgetExceeded,
    StoragePreflight,
)
from story_projection_onto.validate import (
    BoundaryValidationError,
    validate_repair_preservation,
)

NORMAL_ACCEPTANCE_CLASSES = (
    "acceptance_c1",
    "acceptance_c2",
    "acceptance_fixed_select",
    "acceptance_repair",
)
FALLBACK_IMPLEMENTATION_FILES = (
    "configs/study/decoding.json",
    "configs/study/development_call_manifest.json",
    "configs/study/development_construction.json",
    "configs/study/fallback_model.json",
    "configs/study/gpu_call_inventory.json",
    "configs/study/model.json",
    "configs/study/resource_limits.json",
    "configs/study/storage_phase_allocations.json",
    "prompts/c1_pre/prompt_v1.md",
    "prompts/c2_query/prompt_v1.md",
    "prompts/fixed_select/prompt_v1.md",
    "prompts/ablations/no_temporal_epistemic_v1.md",
    "prompts/repair/prompt_v1.md",
    "schemas/jsonschema/ontology_draft.schema.json",
    "scripts/authorize_fallback_model.py",
    "scripts/run_development_block.py",
    "scripts/run_fallback_gpu_acceptance.py",
    "scripts/verify_model_snapshot.py",
    "src/story_projection_onto/benchmark_runtime.py",
    "src/story_projection_onto/conditions/base.py",
    "src/story_projection_onto/conditions/c0.py",
    "src/story_projection_onto/conditions/c1.py",
    "src/story_projection_onto/conditions/c2.py",
    "src/story_projection_onto/conditions/fixed_select.py",
    "src/story_projection_onto/contracts.py",
    "src/story_projection_onto/development_adapter.py",
    "src/story_projection_onto/development_artifacts.py",
    "src/story_projection_onto/development_assessment_bridge.py",
    "src/story_projection_onto/development_continuation.py",
    "src/story_projection_onto/development_execution.py",
    "src/story_projection_onto/development_runtime.py",
    "src/story_projection_onto/evidence.py",
    "src/story_projection_onto/experiment.py",
    "src/story_projection_onto/fallback_acceptance.py",
    "src/story_projection_onto/gpu_runtime.py",
    "src/story_projection_onto/llm.py",
    "src/story_projection_onto/manifest.py",
    "src/story_projection_onto/model_gate.py",
    "src/story_projection_onto/phase1_acceptance.py",
    "src/story_projection_onto/query_runtime.py",
    "src/story_projection_onto/scorer_only/acceptance_grounding.py",
    "src/story_projection_onto/scorer_only/development_assessment.py",
    "src/story_projection_onto/store.py",
    "src/story_projection_onto/validate.py",
)
REPAIR_TRIGGER_RULE = (
    "first base output with model-visible schema, boundary, capability, budget, "
    "evidence-ID, grounding-presence, or horizon diagnostics; scorer-only semantic "
    "grounding never enters the repair prompt; no second repair"
)
FORBIDDEN_ADAPTER_LIFECYCLE_MEMBERS = (
    "close",
    "emergency_stop",
    "kill",
    "load",
    "restart",
    "shutdown",
    "start",
    "stop",
    "terminate",
)


class DevelopmentAdopterRegistration(ImmutableRecord):
    """Static, hash-bound identity for the synchronous development adopter.

    This registration is available before the fallback model is loaded.  It
    authenticates the exact 24-call manifest and the code that will construct
    freeze-bound development inputs, while deliberately granting no model
    lifecycle authority.
    """

    protocol: Literal["fallback-live-development-adopter-v1"] = (
        "fallback-live-development-adopter-v1"
    )
    adopter_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    implementation_sha256: Sha256Digest
    source_sha256: Sha256Digest
    development_call_manifest_sha256: Sha256Digest
    prequery_input_builder_sha256: Sha256Digest
    expected_development_call_count: Literal[24] = 24
    synchronous_adoption: Literal[True] = True
    lifecycle_authority: Literal[False] = False
    query_blind_preparation: Literal[True] = True


class DevelopmentContinuationBootstrap(ImmutableRecord):
    """Freeze-bound input persisted before any adopter preparation runs."""

    bootstrap_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    owner_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    fallback_execution_hash: Sha256Digest
    accepted_fallback_result_hash: Sha256Digest
    micro_pilot_acceptance_receipt: dict[str, object]
    micro_pilot_acceptance_receipt_sha256: Sha256Digest
    adopter_registration_hash: Sha256Digest
    selected_model_freeze: dict[str, object]
    selected_model_freeze_hash: Sha256Digest
    source_tree_hash: Sha256Digest
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    gpu_session_event_id: str = Field(min_length=1)
    launcher_configuration_hash: Sha256Digest
    model_snapshot_manifest_hash: Sha256Digest
    service_checkpoint_sha256: Sha256Digest
    allocated_gpu_seconds_before_preparation: float = Field(ge=0.0)
    expected_development_call_count: Literal[24] = 24
    model_load_count: Literal[1] = 1
    controller_process_restart_observed: Literal[True] = True
    model_process_restart_observed: Literal[False] = False
    adopter_lifecycle_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_bound_manifests_and_counter(self) -> Self:
        for name, manifest, expected in (
            (
                "micro-pilot acceptance receipt",
                self.micro_pilot_acceptance_receipt,
                self.micro_pilot_acceptance_receipt_sha256,
            ),
            (
                "selected-model freeze",
                self.selected_model_freeze,
                self.selected_model_freeze_hash,
            ),
        ):
            immutable = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            if (
                manifest.get("manifest_sha256") != expected
                or canonical_sha256(immutable) != expected
            ):
                raise ValueError(f"development bootstrap carries an invalid {name}")
        receipt = self.micro_pilot_acceptance_receipt
        if (
            receipt.get("accepted_result_manifest_sha256")
            != self.accepted_fallback_result_hash
            or receipt.get("execution_hash") != self.fallback_execution_hash
            or receipt.get("source_tree_sha256") != self.source_tree_hash
            or self.selected_model_freeze.get(
                "micro_pilot_acceptance_receipt_sha256"
            )
            != self.micro_pilot_acceptance_receipt_sha256
        ):
            raise ValueError("development bootstrap lineage is internally inconsistent")
        if not math.isfinite(self.allocated_gpu_seconds_before_preparation):
            raise ValueError("development bootstrap GPU counter must be finite")
        return self


@dataclass(frozen=True, slots=True)
class PreparedDevelopmentContinuation:
    """Query-blind objects built after freeze and before the callback starts."""

    call_manifest: DevelopmentCallManifest
    prequery_inputs: DevelopmentPrequeryInputs
    forecast_control: ForecastControl
    execution_id: str
    execution_manifest_hash: Sha256Digest
    checkpoint_path: Path
    service_adapter: InjectedLiveDevelopmentService
    query_access_event_count: Literal[0] = 0


class DevelopmentContinuationHandoff(ImmutableRecord):
    """Authenticated input offered while the accepted model process is live."""

    handoff_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    owner_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    bootstrap_hash: Sha256Digest
    fallback_execution_hash: Sha256Digest
    accepted_fallback_result_hash: Sha256Digest
    source_execution_hash: Sha256Digest
    micro_pilot_acceptance_receipt_sha256: Sha256Digest
    adopter_registration_hash: Sha256Digest
    selected_model_freeze: dict[str, object]
    selected_model_freeze_hash: Sha256Digest
    live_service_identity: LiveServiceIdentity
    live_service_identity_hash: Sha256Digest
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    gpu_session_event_id: str = Field(min_length=1)
    launcher_configuration_hash: Sha256Digest
    model_snapshot_manifest_hash: Sha256Digest
    service_checkpoint_sha256: Sha256Digest
    allocated_gpu_seconds_before: float = Field(ge=0.0)
    development_call_manifest_hash: Sha256Digest
    development_source_plan_hash: Sha256Digest
    development_execution_id: str = Field(min_length=1)
    development_execution_manifest_hash: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    development_prequery_inputs_hash: Sha256Digest
    development_checkpoint_path: str = Field(min_length=1)
    development_checkpoint_sha256_before: Sha256Digest | None = None
    forecast_receipt_hash: Sha256Digest
    post_development_mandatory_forecast_seconds: float = Field(ge=0.0)
    complete_post_development_manifest_included: Literal[True] = True
    query_access_event_count_before_adoption: Literal[0] = 0
    expected_development_call_count: Literal[24] = 24
    model_load_count: Literal[1] = 1
    controller_process_restart_observed: Literal[True] = True
    model_process_restart_observed: Literal[False] = False
    adopter_lifecycle_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_freeze_and_counter(self) -> Self:
        immutable = {
            key: value
            for key, value in self.selected_model_freeze.items()
            if key != "manifest_sha256"
        }
        if (
            self.selected_model_freeze.get("manifest_sha256")
            != self.selected_model_freeze_hash
            or canonical_sha256(immutable) != self.selected_model_freeze_hash
        ):
            raise ValueError("development handoff carries an invalid selected-model freeze")
        if not math.isfinite(self.allocated_gpu_seconds_before):
            raise ValueError("development handoff GPU counter must be finite")
        if self.live_service_identity.content_hash != self.live_service_identity_hash:
            raise ValueError("development handoff service identity hash is invalid")
        expected_identity = {
            "owner_run_id": self.owner_run_id,
            "service_pid": self.service_pid,
            "service_start_ticks": self.service_start_ticks,
            "gpu_session_event_id": self.gpu_session_event_id,
            "launcher_configuration_hash": self.launcher_configuration_hash,
            "model_snapshot_hash": self.model_snapshot_manifest_hash,
            "selected_model_freeze_hash": self.selected_model_freeze_hash,
            "source_execution_hash": self.source_execution_hash,
        }
        if any(
            getattr(self.live_service_identity, key) != value
            for key, value in expected_identity.items()
        ):
            raise ValueError("development handoff duplicates another service identity")
        if not math.isfinite(self.post_development_mandatory_forecast_seconds):
            raise ValueError("development handoff forecast must be finite")
        return self


class DevelopmentContinuationReceipt(ImmutableRecord):
    """Proof that the adopter used and returned the same still-live service."""

    handoff_hash: Sha256Digest
    bootstrap_hash: Sha256Digest
    adopter_registration_hash: Sha256Digest
    fallback_execution_hash: Sha256Digest
    source_execution_hash: Sha256Digest
    selected_model_freeze_hash: Sha256Digest
    live_service_identity_hash_before: Sha256Digest
    live_service_identity_hash_after: Sha256Digest
    service_pid: int = Field(gt=0)
    service_start_ticks: int = Field(gt=0)
    gpu_session_event_id: str = Field(min_length=1)
    launcher_configuration_hash: Sha256Digest
    model_snapshot_manifest_hash: Sha256Digest
    development_execution_id: str = Field(min_length=1)
    development_execution_manifest_hash: Sha256Digest
    development_call_manifest_hash: Sha256Digest
    development_source_plan_hash: Sha256Digest
    gpu_call_inventory_file_sha256: Sha256Digest
    development_prequery_inputs_hash: Sha256Digest
    development_result_hash: Sha256Digest
    development_checkpoint_sha256_after: Sha256Digest
    forecast_receipt_hash: Sha256Digest
    terminal_call_count: Literal[24] = 24
    every_call_has_intention_to_treat_row: Literal[True] = True
    every_planned_request_started: bool
    every_planned_call_succeeded: bool
    twelve_query_access_events: bool
    same_live_service_identity: Literal[True] = True
    development_gate_passed: bool
    development_forecast_admitted: bool
    scientific_thresholds_passed: bool
    failure_artifact_hash: Sha256Digest | None = None
    service_returned_live_to_owner: Literal[True] = True
    model_load_count_by_adopter: Literal[0] = 0
    service_start_count_by_adopter: Literal[0] = 0
    service_shutdown_count_by_adopter: Literal[0] = 0
    allocated_gpu_seconds_after: float = Field(ge=0.0)
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def allocated_counter_is_finite(self) -> Self:
        if not math.isfinite(self.allocated_gpu_seconds_after):
            raise ValueError("development receipt GPU counter must be finite")
        expected_gate = all(
            (
                self.every_planned_request_started,
                self.every_planned_call_succeeded,
                self.twelve_query_access_events,
                self.same_live_service_identity,
                self.development_forecast_admitted,
                self.scientific_thresholds_passed,
            )
        )
        if self.development_gate_passed != expected_gate:
            raise ValueError("development receipt gate differs from its frozen conjunction")
        if (
            self.live_service_identity_hash_before
            != self.live_service_identity_hash_after
        ):
            raise ValueError("development receipt changed live-service identity")
        if self.development_gate_passed and self.failure_artifact_hash is not None:
            raise ValueError("passed development receipt cannot cite a failure artifact")
        return self


@runtime_checkable
class DevelopmentContinuationAdopter(Protocol):
    """Authenticated two-step adopter; neither step receives lifecycle methods."""

    def registration(self) -> DevelopmentAdopterRegistration:
        """Return the immutable code, input-builder, and 24-call identity."""

    def prepare(
        self,
        bootstrap: DevelopmentContinuationBootstrap,
    ) -> PreparedDevelopmentContinuation:
        """Build exact query-blind inputs and an already-live narrow adapter."""

    def adopt_and_run(
        self,
        handoff: DevelopmentContinuationHandoff,
        service: InjectedLiveDevelopmentService,
    ) -> DevelopmentExecutionResult:
        """Run synchronously and return the typed development execution result."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root must be an object")
    return cast(dict[str, object], value)


def validate_source_association(
    path: Path,
    *,
    source_root: Path | None = None,
) -> Mapping[str, object]:
    """Validate an association and optionally rehash its exact local source tree.

    Structural validation is useful when inspecting historical provenance.  GPU
    execution always supplies ``source_root``; that mode also validates the
    linked local manifest file byte-for-byte and independently rebuilds its file
    inventory/tree hash from the current checkout.  A stale but self-consistent
    association therefore cannot authorize changed code.
    """

    association = _load_object(path)
    supplied = association.get("manifest_sha256")
    immutable = {key: value for key, value in association.items() if key != "manifest_sha256"}
    if not isinstance(supplied, str) or supplied != canonical_sha256(immutable):
        raise ValueError("source association manifest hash does not match its contents")
    local_tree = association.get("local_tree_sha256")
    remote_tree = association.get("remote_tree_sha256")
    if (
        association.get("kind") != "local_remote_source_tree_association"
        or not isinstance(local_tree, str)
        or len(local_tree) != 64
        or local_tree != remote_tree
        or association.get("branch") != "implementation/query-dependent-temporal-ontology"
    ):
        raise ValueError("source association does not bind one identical local/remote tree")
    if source_root is not None:
        revision = association.get("revision_label")
        manifest_name = association.get("local_manifest")
        manifest_file_hash = association.get("local_manifest_file_sha256")
        if (
            not isinstance(revision, str)
            or not revision
            or not isinstance(manifest_name, str)
            or Path(manifest_name).name != manifest_name
            or not isinstance(manifest_file_hash, str)
        ):
            raise ValueError("source association lacks one safe local manifest binding")
        local_manifest_path = path.resolve(strict=True).parent / manifest_name
        if not local_manifest_path.is_file() or local_manifest_path.is_symlink():
            raise ValueError("source association local manifest is unavailable")
        if _file_sha256(local_manifest_path) != manifest_file_hash:
            raise ValueError("source association local manifest file hash changed")
        local_manifest = _load_object(local_manifest_path)
        rebuilt = build_source_manifest(source_root, revision).to_dict()
        if local_manifest != rebuilt:
            raise ValueError("current source tree differs from its associated local manifest")
        if rebuilt.get("tree_sha256") != local_tree:
            raise ValueError("rebuilt source-tree hash differs from its association")
    return association


def pre_fallback_gpu_accounting_baseline(
    primary_result: Mapping[str, object],
) -> dict[str, object]:
    """Reproduce the immutable cumulative accounting claimed by primary rejection."""

    runtime = primary_result.get("runtime")
    accounting = runtime.get("gpu_accounting") if isinstance(runtime, Mapping) else None
    if not isinstance(accounting, Mapping):
        raise ValueError("primary result lacks cumulative GPU accounting")
    expected_by_kind = accounting.get("by_kind_microseconds")
    if not isinstance(expected_by_kind, Mapping) or not all(
        isinstance(name, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for name, value in expected_by_kind.items()
    ):
        raise ValueError("primary result has invalid per-kind GPU accounting")
    scalar_names = (
        "total_allocated_microseconds",
        "event_count",
        "service_session_count",
    )
    if any(
        not isinstance(accounting.get(name), int)
        or isinstance(accounting.get(name), bool)
        or cast(int, accounting[name]) < 0
        for name in scalar_names
    ):
        raise ValueError("primary result has invalid cumulative GPU accounting")
    if cast(int, accounting["total_allocated_microseconds"]) != sum(
        cast(int, value) for value in expected_by_kind.values()
    ):
        raise ValueError("primary GPU accounting total does not reconcile by kind")
    expected: dict[str, object] = {
        "total_allocated_microseconds": accounting.get("total_allocated_microseconds"),
        "event_count": accounting.get("event_count"),
        "service_session_count": accounting.get("service_session_count"),
        "by_kind_microseconds": dict(sorted(expected_by_kind.items())),
    }
    primary_manifest_hash = primary_result.get("manifest_sha256")
    if not isinstance(primary_manifest_hash, str):
        raise ValueError("primary result lacks its immutable manifest hash")
    payload: dict[str, object] = {
        "kind": "pre_fallback_gpu_accounting_baseline",
        "primary_result_manifest_sha256": primary_manifest_hash,
        **expected,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def validate_pre_fallback_gpu_accounting(
    primary_result: Mapping[str, object],
    observed: GpuSummary,
) -> dict[str, object]:
    """Require the fallback ledger to continue the exact rejected-primary ledger.

    This check is performed immediately before the only permitted fallback service
    start.  It prevents a fresh SQLite file from silently resetting spent GPU time
    while still accepting the primary rejection certificate.
    """

    baseline = pre_fallback_gpu_accounting_baseline(primary_result)
    expected = {
        name: baseline[name]
        for name in (
            "total_allocated_microseconds",
            "event_count",
            "service_session_count",
            "by_kind_microseconds",
        )
    }
    actual: dict[str, object] = {
        "total_allocated_microseconds": observed.total_allocated_microseconds,
        "event_count": observed.event_count,
        "service_session_count": observed.service_session_count,
        "by_kind_microseconds": {
            kind.value: microseconds for kind, microseconds in observed.by_kind_microseconds
        },
    }
    if expected != actual:
        raise RuntimeError(
            "fallback ledger does not reproduce the exact rejected-primary GPU accounting"
        )
    return baseline


def _private_atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class FallbackCallSpec:
    call_id: str
    form: str
    reserve_call_class: str
    watchdog_seconds: int
    condition: ConditionName
    seed_block: int
    request_fixture: str
    prompt_file: str
    forecast_call_class: str

    def acceptance_call(self) -> AcceptanceCall:
        return AcceptanceCall(
            call_id=self.call_id,
            call_class=self.forecast_call_class,
            condition=self.condition,
            seed_block=self.seed_block,
            request_fixture=self.request_fixture,
            prompt_file=self.prompt_file,
            watchdog_seconds={
                "acceptance_c1": 180,
                "acceptance_c2": 120,
                "acceptance_fixed_select": 90,
            }[self.forecast_call_class],
        )

    @property
    def retry_class(self) -> RetryClass:
        return {
            "reserve_long": RetryClass.LONG,
            "reserve_standard": RetryClass.STANDARD,
            "reserve_short": RetryClass.SHORT,
        }[self.reserve_call_class]

    @property
    def required_constructive_operators(self) -> tuple[str, ...]:
        all_operators = tuple(sorted(operator.value for operator in CONSTRUCTIVE_OPERATORS))
        if self.condition is ConditionName.C1_LLM_PRE:
            # The fallback protocol has one C1 call rather than the normal block's
            # two C1 calls.  Keep this bounded call a substantive, richly grounded
            # probe that fits the registered 2,048-token output cap.  The integrated
            # 24-call development gate remains responsible for the preregistered
            # *complete* C1 operator inventory, grounding, and union-gold recall.
            return (
                "event_reification",
                "merge",
                "rare_preservation",
                "schema_relation",
                "temporal_qualification",
            )
        if self.call_id == "fallback-c2-01":
            return (
                "abstraction",
                "contextual_type",
                "epistemic_qualification",
                "include_exclude",
                "split",
            )
        if self.call_id == "fallback-c2-02":
            return tuple(operator for operator in all_operators if operator not in {
                "abstraction",
                "contextual_type",
                "epistemic_qualification",
                "include_exclude",
                "split",
            })
        return ()

    def public_manifest(self, root: Path) -> dict[str, object]:
        request_call = self.acceptance_call()
        request_manifest = request_call.public_manifest(root)
        request_manifest.update(
            {
                "form": self.form,
                "reserve_call_class": self.reserve_call_class,
                "watchdog_seconds": self.watchdog_seconds,
                "forecast_proxy_call_class": self.forecast_call_class,
                "required_constructive_operators": list(
                    self.required_constructive_operators
                ),
            }
        )
        return request_manifest


def fallback_pilot_calls(policy: FallbackModelPolicy) -> tuple[FallbackCallSpec, ...]:
    frozen = (
        (
            "fallback-c1-01",
            "C1",
            "reserve_long",
            240,
            ConditionName.C1_LLM_PRE,
            0,
            "tests/fixtures/phase1/c1_pre_request.json",
            "prompts/c1_pre/prompt_v1.md",
            "acceptance_c1",
        ),
        (
            "fallback-c2-01",
            "C2",
            "reserve_standard",
            150,
            ConditionName.C2_LLM_QUERY,
            0,
            "tests/fixtures/phase1/c2_query_request.json",
            "prompts/c2_query/prompt_v1.md",
            "acceptance_c2",
        ),
        (
            "fallback-c2-02",
            "C2",
            "reserve_standard",
            150,
            ConditionName.C2_LLM_QUERY,
            1,
            "tests/fixtures/phase1/c2_query_request.json",
            "prompts/c2_query/prompt_v1.md",
            "acceptance_c2",
        ),
        (
            "fallback-fixed-01",
            "A-FixedSelect",
            "reserve_short",
            90,
            ConditionName.A_FIXED_SELECT,
            1,
            "tests/fixtures/phase1/fixed_select_request.json",
            "prompts/fixed_select/prompt_v1.md",
            "acceptance_fixed_select",
        ),
    )
    policy_rows = tuple(
        (call.call_id, call.form, call.reserve_tier, call.watchdog_seconds)
        for call in policy.calls
    )
    if policy_rows != tuple(row[:4] for row in frozen):
        raise ValueError("fallback call order or identity differs from the executable plan")
    return tuple(FallbackCallSpec(*row) for row in frozen)


def fallback_plan_manifest(root: Path) -> dict[str, object]:
    """Return the complete CPU-only executable plan for the permitted fallback."""

    root = root.resolve(strict=True)
    policy_path = root / "configs/study/fallback_model.json"
    policy = FallbackModelPolicy.load(policy_path)
    calls = fallback_pilot_calls(policy)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "phase1_fallback_micro_pilot_plan",
        "model_candidate": "fallback",
        "repository": policy.repository,
        "revision": policy.revision,
        "fallback_policy_sha256": policy.content_hash,
        "base_call_count": 4,
        "maximum_repair_call_count": 1,
        "maximum_inference_attempt_count": 5,
        "maximum_inference_seconds": 720,
        "service_start_event_count": 1,
        "process_restart_event_count": 0,
        "controller_checkpoint_handoff_required": True,
        "controller_invocation_count": 2,
        "controller_process_restart_required": True,
        "supervising_orchestrator_required": True,
        "internal_controller_stages_require_live_guard": True,
        "normal_exit_requires_verified_shutdown": True,
        "pre_fallback_ledger_must_exactly_reproduce_primary_result": True,
        "fresh_gpu_ledger_forbidden": True,
        "model_process_pid_must_remain_unchanged": True,
        "stage_one_executes_inference": False,
        "development_continuation": {
            "same_live_model_service_required": True,
            "next_registered_call_count": 24,
            "micro_pilot_c1_operator_scope": "representative_grounded_subset",
            "complete_c1_operator_inventory_gate": (
                "DevelopmentScientificAssessment."
                "c1_all_construction_operators_exercised"
            ),
            "authenticated_adopter_must_be_integrated_before_gpu_execution": True,
            "adopter_protocol": "fallback-live-development-adopter-v1",
            "adopter_registration_binds": [
                "implementation_sha256",
                "source_sha256",
                "development_call_manifest_sha256",
                "prequery_input_builder_sha256",
            ],
            "query_blind_preparation_persisted_before_callback": True,
            "callback_receives_narrow_live_service_adapter": True,
            "callback_receives_lifecycle_methods": False,
            "callback_returns": "canonical DevelopmentExecutionResult",
            "fallback_owner_derives_continuation_receipt": True,
            "preparation_binds": [
                "development_call_manifest",
                "development_source_plan",
                "gpu_call_inventory",
                "development_prequery_inputs",
                "development_checkpoint",
                "forecast_receipt",
                "complete_post_development_forecast",
                "live_service_identity",
            ],
            "selected_model_freeze_bootstrapped_after_micro_pilot": True,
            "selected_model_freeze_bootstrap_requires_development_completion": False,
            "adopter_receipt_requires_same_pid_and_start_ticks": True,
            "adopter_model_load_start_shutdown_counts": [0, 0, 0],
            "fallback_owner_performs_final_shutdown": True,
            "standalone_runner_leaves_orphan": False,
            "standalone_cli_gpu_execution_enabled": True,
            "standalone_registered_factory": (
                "story_projection_onto.development_continuation:"
                "create_production_development_adopter"
            ),
            "standalone_phase1_gate_requires_completed_development": True,
        },
        "runtime_feasibility_freeze": {
            "enforce_eager": True,
            "applies_to_all_selected_model_llm_calls": True,
            "may_not_change_after_micro_pilot": True,
            "timing_and_vram_must_be_measured_with_this_setting": True,
            "throughput_tradeoff_must_be_reported": True,
        },
        "normal_acceptance_block": {
            "executed": False,
            "superseded_classes": list(NORMAL_ACCEPTANCE_CLASSES),
            "reason": "the frozen fallback micro-pilot replaces the rejected primary block",
        },
        "repair": {
            "trigger_rule": REPAIR_TRIGGER_RULE,
            "reserve_call_class": "reserve_short",
            "watchdog_seconds": 90,
            "maximum_calls": 1,
        },
        "calls": [call.public_manifest(root) for call in calls],
        "implementation_files": [
            {"path": relative, "sha256": _file_sha256(root / relative)}
            for relative in FALLBACK_IMPLEMENTATION_FILES
        ],
        "executes_gpu": False,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def build_fallback_acceptance_request(
    *,
    root: Path,
    call: FallbackCallSpec,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
) -> GuidedJSONRequest:
    """Build one frozen fallback request with its model-visible capability probe."""

    capability_probe = {
        "probe_kind": "registered_constructive_operator_capability",
        "required_operators": list(call.required_constructive_operators),
        "require_one_explicit_decision_per_operator": bool(
            call.required_constructive_operators
        ),
        "facts_or_gold_labels_supplied": False,
        "coverage_scope": (
            "bounded_micro_pilot_subset"
            if call.condition is ConditionName.C1_LLM_PRE
            else "bounded_micro_pilot_partition"
        ),
        "complete_c1_inventory_gate": (
            "integrated_24_call_development_scientific_assessment"
            if call.condition is ConditionName.C1_LLM_PRE
            else None
        ),
    }
    return build_acceptance_request(
        root=root,
        call=call.acceptance_call(),
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        model_name=FALLBACK_SERVED_MODEL_NAME,
        model_revision=FALLBACK_MODEL_REVISION,
        additional_sections={"fallback_capability_probe": capability_probe},
    )


class FallbackCapabilityCoverageError(ValueError):
    """A valid graph omitted a frozen model-visible capability probe operator."""

    def __init__(self, missing_operators: Sequence[str]) -> None:
        self.missing_operators = tuple(sorted(missing_operators))
        super().__init__(
            "fallback capability probe lacks required constructive operators: "
            + ", ".join(self.missing_operators)
        )


class FallbackCapabilityBehaviorError(ValueError):
    """Operator labels were present without their registered graph behavior."""

    def __init__(self, invalid_operators: Sequence[str]) -> None:
        self.invalid_operators = tuple(sorted(invalid_operators))
        super().__init__(
            "fallback capability labels lack required graph behavior: "
            + ", ".join(self.invalid_operators)
        )


def _require_call_operator_coverage(
    call: FallbackCallSpec,
    audit: Mapping[str, object],
    parsed_object: Mapping[str, object],
    *,
    root: Path,
) -> dict[str, object]:
    observed = set(cast(Sequence[str], audit["constructive_operators"]))
    missing = set(call.required_constructive_operators) - observed
    if missing:
        raise FallbackCapabilityCoverageError(sorted(missing))
    if not call.required_constructive_operators:
        return {"required": [], "behaviorally_valid": True, "operators": {}}

    graph = cast(Mapping[str, object], parsed_object["instance_graph"])
    schema = cast(Mapping[str, object], parsed_object["local_schema"])
    entities = {
        cast(str, item["entity_id"]): cast(Mapping[str, object], item)
        for item in cast(Sequence[Mapping[str, object]], graph["entities"])
    }
    events = {
        cast(str, item["event_id"]): cast(Mapping[str, object], item)
        for item in cast(Sequence[Mapping[str, object]], graph["events"])
    }
    propositions = {
        cast(str, item["proposition_content_id"]): cast(Mapping[str, object], item)
        for item in cast(Sequence[Mapping[str, object]], graph["proposition_contents"])
    }
    assertions = {
        cast(str, item["assertion_id"]): cast(Mapping[str, object], item)
        for item in cast(Sequence[Mapping[str, object]], graph["assertions"])
    }
    contextual_types = {
        cast(str, item["type_id"]): cast(Mapping[str, object], item)
        for item in cast(Sequence[Mapping[str, object]], schema["contextual_types"])
    }
    predicates = {
        cast(str, item["predicate_id"]): cast(Mapping[str, object], item)
        for item in cast(Sequence[Mapping[str, object]], schema["predicates"])
    }
    decisions = cast(Sequence[Mapping[str, object]], parsed_object["decisions"])
    by_operator: dict[str, list[Mapping[str, object]]] = {}
    for decision in decisions:
        by_operator.setdefault(cast(str, decision["operator"]), []).append(decision)

    request = json.loads((root / call.request_fixture).read_text(encoding="utf-8"))
    evidence = request.get("evidence") or request["packet"]["evidence"]
    one_off_ids = {
        cast(str, item["evidence_id"])
        for item in cast(Sequence[Mapping[str, object]], evidence)
        if any(
            marker in cast(str, item["text"]).casefold()
            for marker in ("a single ", "one-off", "only time", "once")
        )
    }

    def created(decision: Mapping[str, object]) -> set[str]:
        return set(cast(Sequence[str], decision.get("created_object_ids", ())))

    def inputs(decision: Mapping[str, object]) -> set[str]:
        return set(cast(Sequence[str], decision.get("input_object_ids", ())))

    def evidence_ids(value: Mapping[str, object]) -> set[str]:
        return set(cast(Sequence[str], value.get("evidence_ids", ())))

    def nontrivial_time(assertion: Mapping[str, object]) -> bool:
        temporal = cast(Mapping[str, object], assertion.get("temporal_scope", {}))
        story = cast(Mapping[str, object], temporal.get("story_time", {}))
        validity = cast(Mapping[str, object], temporal.get("validity_time", {}))
        discourse = cast(Mapping[str, object], temporal.get("discourse_position", {}))
        revelation = cast(Mapping[str, object], temporal.get("revelation_position", {}))
        return (
            story.get("kind") != "unknown"
            or validity.get("kind") != "unknown"
            or cast(int, discourse.get("passage_order", 0)) > 0
            or cast(int, revelation.get("revelation_order", 0)) > 0
        )

    checks: dict[str, object] = {}
    for operator in call.required_constructive_operators:
        candidates = by_operator.get(operator, [])
        valid_decisions: list[str] = []
        for decision in candidates:
            targets = created(decision)
            valid = False
            if operator == "include_exclude":
                valid = bool(targets.intersection({*entities, *events, *assertions}))
            elif operator == "merge":
                valid = len(inputs(decision)) >= 2 and any(
                    len(
                        set(
                            cast(
                                Sequence[str],
                                entities[target].get("supported_mention_candidate_ids", ()),
                            )
                        )
                    )
                    >= 2
                    for target in targets.intersection(entities)
                )
            elif operator == "split":
                valid = len(targets.intersection(entities)) >= 2
            elif operator == "contextual_type":
                created_types = targets.intersection(contextual_types)
                valid = bool(created_types) and any(
                    item.get("contextual_type_id") in created_types
                    for item in (*entities.values(), *events.values())
                )
            elif operator == "schema_relation":
                valid = bool(targets.intersection(predicates)) or (
                    schema.get("schema_id") in targets and bool(predicates)
                )
            elif operator == "event_reification":
                valid = bool(targets.intersection(events))
            elif operator == "abstraction":
                abstraction = schema.get("abstraction")
                valid = isinstance(abstraction, str) and bool(abstraction) and (
                    schema.get("schema_id") in targets
                    or any(
                        item.get("abstraction") == abstraction
                        for target, item in {**entities, **contextual_types}.items()
                        if target in targets
                    )
                )
            elif operator == "temporal_qualification":
                valid = any(
                    nontrivial_time(assertions[target])
                    for target in targets.intersection(assertions)
                )
            elif operator == "epistemic_qualification":
                valid = any(
                    isinstance(assertions[target].get("epistemic_scope"), Mapping)
                    and cast(Mapping[str, object], assertions[target]["epistemic_scope"]).get(
                        "holder_id"
                    )
                    in entities
                    and cast(Mapping[str, object], assertions[target]["epistemic_scope"]).get(
                        "proposition_content_id"
                    )
                    in propositions
                    for target in targets.intersection(assertions)
                )
            elif operator == "rare_preservation":
                valid = bool(evidence_ids(decision).intersection(one_off_ids)) and any(
                    evidence_ids(
                        ({**entities, **events, **assertions, **propositions})[target]
                    ).intersection(one_off_ids)
                    for target in targets.intersection(
                        {*entities, *events, *assertions, *propositions}
                    )
                )
            if valid:
                valid_decisions.append(cast(str, decision["decision_id"]))
        checks[operator] = {
            "valid": bool(valid_decisions),
            "decision_ids": sorted(valid_decisions),
        }
    invalid = [
        operator
        for operator, result in checks.items()
        if cast(Mapping[str, object], result)["valid"] is not True
    ]
    if invalid:
        raise FallbackCapabilityBehaviorError(invalid)
    return {
        "required": list(call.required_constructive_operators),
        "behaviorally_valid": True,
        "operators": checks,
        "one_off_evidence_ids": sorted(one_off_ids),
    }


def _request_public_metadata(request: GuidedJSONRequest) -> dict[str, object]:
    return {
        "request_hash": request.request_hash,
        "prompt_hash": request.prompt_hash,
        "rendered_input_token_count": request.rendered_input_token_count,
        "decoding_manifest": request.decoding.model_dump(mode="json"),
        "packing_report": request.packing.model_dump(mode="json", by_alias=True),
        "capability_manifest": CapabilityManifest.for_condition(
            request.condition
        ).model_dump(mode="json"),
        "condition_output_schema_sha256": canonical_sha256(request.output_schema),
    }


def _fact_free_repair_diagnostics(exc: Exception) -> tuple[dict[str, object], ...]:
    """Map only model-visible validator failures to non-prescriptive diagnostics."""

    if isinstance(exc, FallbackCapabilityCoverageError):
        return (
            {
                "code": "missing_constructive_operator",
                "path": "decisions",
                "message": (
                    "the registered capability probe did not demonstrate every "
                    "required constructive operator"
                ),
                "related_ids": list(exc.missing_operators),
            },
        )
    if isinstance(exc, FallbackCapabilityBehaviorError):
        return (
            {
                "code": "invalid_constructive_operator_behavior",
                "path": "decisions",
                "message": (
                    "operator labels did not point to graph changes demonstrating the "
                    "registered capability behavior"
                ),
                "related_ids": list(exc.invalid_operators),
            },
        )
    if isinstance(exc, ValidationError):
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        for error in exc.errors(include_url=False, include_context=False, include_input=False):
            location_parts = tuple(str(part) for part in error.get("loc", ()))
            location = location_parts[0] if location_parts else "ontology_draft"
            if location in seen:
                continue
            seen.add(location)
            rows.append(
                {
                    "code": "schema_validation",
                    "path": location,
                    "message": "field violates the frozen OntologyDraft schema",
                    "related_ids": [],
                }
            )
        return tuple(rows)
    if isinstance(exc, BoundaryValidationError):
        return tuple(
            {
                "code": diagnostic.code.value,
                "path": diagnostic.path,
                "message": diagnostic.message,
                "related_ids": list(diagnostic.related_ids),
            }
            for diagnostic in exc.report.diagnostics
        )
    if isinstance(exc, FixedSelectCapabilityError):
        return (
            {
                "code": "fixed_selection_violation",
                "path": "instance_graph",
                "message": "output changed or invented sealed semantic content",
                "related_ids": [],
            },
            {
                "code": "fixed_selection_violation",
                "path": "decisions",
                "message": "output exceeded the selection-only capability boundary",
                "related_ids": [],
            },
        )
    message = str(exc)
    if message.startswith("semantic grounding audit failed:"):
        # Scorer-only known-answer assessments are never model-visible.
        return ()
    mappings: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("generated decisions exceed", "capability_violation", ("decisions",)),
        (
            "crosses the fixed spoiler horizon",
            "spoiler_horizon_violation",
            ("instance_graph.assertions",),
        ),
        ("decision predates", "query_timing_violation", ("decisions",)),
        (
            "fixed acceptance request",
            "fixed_selection_violation",
            ("local_schema", "instance_graph", "decisions"),
        ),
        (
            "selection-only",
            "fixed_selection_violation",
            ("local_schema", "instance_graph", "decisions"),
        ),
        (
            "outside the complete frozen input",
            "invalid_evidence_id",
            ("local_schema", "instance_graph", "decisions", "omissions"),
        ),
        (
            "lacks required evidence/description grounding",
            "missing_grounding_reference",
            ("local_schema", "instance_graph", "decisions"),
        ),
        ("has no construction operation", "missing_construction", ("decisions",)),
    )
    for fragment, code, paths in mappings:
        if fragment in message:
            return tuple(
                {
                    "code": code,
                    "path": path,
                    "message": "field group failed a model-visible mechanical validator",
                    "related_ids": [],
                }
                for path in paths
            )
    return ()


def _packing_sections(
    *,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    prompt: str,
    output_schema_hash: str,
    sections: Mapping[str, object],
    rendered_count: int,
) -> tuple[PackingSection, ...]:
    packed = [
        PackingSection(
            name="output_schema",
            section_content_hash=output_schema_hash,
            token_count=0,
        )
    ]
    for name, value in {"system_prompt": prompt, **sections}.items():
        text = value if isinstance(value, str) else canonical_json(value)
        packed.append(
            PackingSection(
                name=name,
                section_content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                token_count=len(tokenizer.encode(text, add_special_tokens=False)),
            )
        )
    independently_encoded = sum(section.token_count for section in packed)
    if rendered_count < independently_encoded:
        raise ValueError("repair sections exceed the exact rendered prompt token count")
    if rendered_count > independently_encoded:
        packed.append(
            PackingSection(
                name="chat_protocol",
                section_content_hash=tokenizer_manifest.nonthinking_probe_sha256,
                token_count=rendered_count - independently_encoded,
            )
        )
    return tuple(packed)


def build_fallback_repair_request(
    *,
    root: Path,
    base_call: AcceptanceCall,
    base_request: GuidedJSONRequest,
    invalid_draft: Mapping[str, object],
    diagnostics: Sequence[Mapping[str, object]],
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
) -> GuidedJSONRequest:
    """Build the one repair from the original complete request and safe diagnostics."""

    if not diagnostics:
        raise ValueError("fallback repair requires model-visible diagnostics")
    if base_request.model_name != FALLBACK_SERVED_MODEL_NAME:
        raise ValueError("fallback repair base request targets a different model")
    original_sections = json.loads(base_request.messages[-1].content)
    if not isinstance(original_sections, dict):
        raise ValueError("fallback base request does not contain one semantic object")
    sections: dict[str, object] = {
        **original_sections,
        "invalid_draft": dict(invalid_draft),
        "validation_diagnostics": [dict(item) for item in diagnostics],
    }
    prompt = (root / "prompts/repair/prompt_v1.md").read_text(encoding="utf-8")
    output_schema = copy.deepcopy(dict(base_request.output_schema))
    output_schema_hash = canonical_sha256(output_schema)
    decoding = DecodingManifest.repair(
        seed=base_call.seed_block,
        eos_token_id=tokenizer_manifest.eos_token_id,
        end_of_turn_token_ids=tokenizer_manifest.end_of_turn_token_ids,
        chat_template_hash=tokenizer_manifest.chat_template_sha256,
        output_schema_hash=output_schema_hash,
        structured_decoder="vllm-0.10.2-xgrammar-no-fallback",
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
    )
    messages = (
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=canonical_json(sections)),
    )
    rendered = tokenizer.apply_chat_template(
        [asdict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes, bytearray)):
        raise ValueError("tokenizer did not return repair prompt token IDs")
    packed = _packing_sections(
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        prompt=prompt,
        output_schema_hash=output_schema_hash,
        sections=sections,
        rendered_count=len(rendered),
    )
    packing = PackingReport.build(
        condition=base_call.condition,
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_model_tokens=12_288,
        maximum_input_tokens=10_752,
        reserved_output_tokens=1_536,
        sections=packed,
        required_section_names=(
            *base_request.packing.required_section_names,
            "invalid_draft",
            "validation_diagnostics",
        ),
        complete_evidence_snapshot=(
            True if base_call.condition is ConditionName.C1_LLM_PRE else None
        ),
        complete_evidence_packet=(
            None if base_call.condition is ConditionName.C1_LLM_PRE else True
        ),
        complete_sealed_ontology=(
            True if base_call.condition is ConditionName.A_FIXED_SELECT else None
        ),
    )
    return GuidedJSONRequest(
        request_id=f"{base_call.call_id}-repair-01",
        model_name=FALLBACK_SERVED_MODEL_NAME,
        condition=base_call.condition,
        messages=messages,
        output_schema=output_schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=len(rendered),
    )


def _validate_generation_envelope(
    generated: GenerationResult,
    request: GuidedJSONRequest,
) -> None:
    if generated.finish_reason != "stop":
        raise ValueError("fallback generation did not finish at a stop token")
    if generated.prompt_tokens != request.rendered_input_token_count:
        raise ValueError("vLLM prompt-token count differs from complete fallback packing")
    if generated.completion_tokens > request.decoding.maximum_output_tokens:
        raise ValueError("vLLM fallback completion exceeded its declared output cap")


def _parsed_vllm_object(payload: bytes) -> dict[str, object]:
    try:
        response = json.loads(payload)
        parsed = json.loads(response["choices"][0]["message"]["content"])
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored fallback response is not resumable guided JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("stored fallback guided JSON response is not an object")
    return cast(dict[str, object], parsed)


def _stored_generation(payload: bytes, request: GuidedJSONRequest) -> GenerationResult:
    """Reconstruct and validate resumable public response metadata."""

    try:
        response = json.loads(payload)
        choice = response["choices"][0]
        parsed = json.loads(choice["message"]["content"])
        usage = response["usage"]
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        finish_reason = choice.get("finish_reason")
        service_request_id = response.get("id")
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored fallback response is not resumable guided JSON") from exc
    if (
        not isinstance(parsed, dict)
        or isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or (finish_reason is not None and not isinstance(finish_reason, str))
        or (service_request_id is not None and not isinstance(service_request_id, str))
    ):
        raise ValueError("stored fallback response metadata has an invalid type")
    return GenerationResult(
        request_id=request.request_id,
        request_hash=request.request_hash,
        response_sha256=hashlib.sha256(payload).hexdigest(),
        parsed_object=parsed,
        raw_response=payload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        service_request_id=service_request_id,
    )


def _base_construction_unit_hash(call: FallbackCallSpec) -> str:
    return canonical_sha256(
        {"fixture": call.request_fixture, "seed_block": call.seed_block}
    )


def _repair_construction_unit_hash(call: FallbackCallSpec, parent_attempt_id: str) -> str:
    return canonical_sha256(
        {
            "fixture": call.request_fixture,
            "seed_block": call.seed_block,
            "repair_parent": parent_attempt_id,
        }
    )


def _event_for(ledger: Ledger, event_id: str):
    matches = tuple(
        event for event in ledger.gpu_events_with_prefix(event_id) if event.event_id == event_id
    )
    if len(matches) > 1:
        raise RuntimeError("GPU event identifier is not unique")
    return None if not matches else matches[0]


def _failure_kind(exc: Exception) -> FailureKind:
    if isinstance(exc, TimeoutError):
        return FailureKind.TIMEOUT
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).casefold():
        return FailureKind.OUT_OF_MEMORY
    return FailureKind.SERVICE


def _resource_gate(ledger: Ledger, limits: ResourceLimits) -> dict[str, object]:
    samples = ledger.resource_samples()
    violations = [
        sample.sample_id
        for sample in samples
        if (
            sample.process_ram_bytes >= limits.maximum_process_ram_bytes
            or sample.gpu_vram_bytes >= limits.maximum_peak_vram_bytes
            or sample.project_storage_bytes > limits.maximum_project_occupied_bytes
            or sample.cpu_worker_count > limits.maximum_cpu_workers
        )
    ]
    failed_storage = [sample.sample_id for sample in ledger.storage_samples() if not sample.allowed]
    return {
        "sample_count": len(samples),
        "storage_preflight_count": len(ledger.storage_samples()),
        "resource_violation_sample_ids": violations,
        "failed_storage_sample_ids": failed_storage,
        "peak_process_ram_bytes": max((sample.process_ram_bytes for sample in samples), default=0),
        "peak_gpu_vram_bytes": max((sample.gpu_vram_bytes for sample in samples), default=0),
        "peak_project_storage_bytes": max(
            (sample.project_storage_bytes for sample in samples), default=0
        ),
        "accepted": not violations and not failed_storage,
    }


def _reserve_receipts(
    ledger: Ledger,
    checkpoint_receipts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_id: dict[str, dict[str, object]] = {}
    for receipt in checkpoint_receipts:
        reservation_id = receipt.get("reservation_id")
        if not isinstance(reservation_id, str):
            raise ValueError("fallback checkpoint has an invalid reserve reservation")
        by_id[reservation_id] = dict(receipt)
    for event in ledger.gpu_events():
        details = json.loads(event.details_json)
        if not isinstance(details, Mapping):
            continue
        reservation_id = details.get("reserve_reservation_id")
        reserve_class = details.get("reserve_call_class")
        if not isinstance(reservation_id, str) or not isinstance(reserve_class, str):
            continue
        observed = {
            "reservation_id": reservation_id,
            "reserve_call_class": reserve_class,
            "watchdog_seconds": {
                "reserve_long": 240,
                "reserve_standard": 150,
                "reserve_short": 90,
            }.get(reserve_class),
        }
        prior = by_id.get(reservation_id)
        if prior is not None and any(
            prior.get(key) != value for key, value in observed.items() if key != "reservation_id"
        ):
            raise RuntimeError("reserve receipt differs between checkpoint and GPU ledger")
        by_id[reservation_id] = {**(prior or {}), **observed, "gpu_event_id": event.event_id}
    receipts = tuple(by_id[key] for key in sorted(by_id))
    counts = Counter(cast(str, row["reserve_call_class"]) for row in receipts)
    maxima = {"reserve_long": 4, "reserve_standard": 8, "reserve_short": 4}
    if any(counts[name] > maximum for name, maximum in maxima.items()):
        raise RuntimeError("cumulative reserve consumption exceeds a frozen tier")
    return receipts


def _selected_model_freeze(
    *,
    activation_certificate: Mapping[str, object],
    replacement_receipt: Mapping[str, object],
    snapshot_manifest: Mapping[str, object],
    launcher: VLLMLaunchConfiguration,
    tokenizer: TokenizerManifest,
    acceptance_receipt: Mapping[str, object],
    source_association: Mapping[str, object],
) -> dict[str, object]:
    if launcher.model_candidate != "fallback" or (
        launcher.repository,
        launcher.revision,
        launcher.served_model_name,
    ) != (
        FALLBACK_MODEL_REPOSITORY,
        FALLBACK_MODEL_REVISION,
        FALLBACK_SERVED_MODEL_NAME,
    ):
        raise ValueError("selected-model freeze rejects a mixed or non-fallback launcher")
    if launcher.enforce_eager is not True:
        raise ValueError("selected-model freeze requires the measured eager runtime")
    if (tokenizer.repository, tokenizer.revision) != (
        FALLBACK_MODEL_REPOSITORY,
        FALLBACK_MODEL_REVISION,
    ):
        raise ValueError("selected-model freeze rejects a mixed tokenizer candidate")
    if (
        snapshot_manifest.get("repository"),
        snapshot_manifest.get("revision"),
    ) != (FALLBACK_MODEL_REPOSITORY, FALLBACK_MODEL_REVISION):
        raise ValueError("selected-model freeze rejects a mixed snapshot candidate")
    if acceptance_receipt.get("micro_pilot_passed") is not True:
        raise ValueError("selected-model freeze requires an accepted fallback micro-pilot")
    if source_association.get("manifest_sha256") != acceptance_receipt.get(
        "source_association_manifest_sha256"
    ):
        raise ValueError("selected-model freeze source association changed after acceptance")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "selected_llm_model_freeze",
        "model_candidate": "fallback",
        "repository": FALLBACK_MODEL_REPOSITORY,
        "revision": FALLBACK_MODEL_REVISION,
        "served_model_name": FALLBACK_SERVED_MODEL_NAME,
        "activation_certificate_sha256": activation_certificate["manifest_sha256"],
        "cache_replacement_receipt_sha256": replacement_receipt["manifest_sha256"],
        "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "launcher_configuration_sha256": launcher.configuration_hash,
        "tokenizer_manifest_sha256": tokenizer.manifest_sha256,
        "micro_pilot_acceptance_receipt_sha256": acceptance_receipt[
            "manifest_sha256"
        ],
        "accepted_micro_pilot_result_sha256": acceptance_receipt[
            "accepted_result_manifest_sha256"
        ],
        "accepted_request_family_hash": acceptance_receipt[
            "request_family_hash"
        ],
        "accepted_operator_gate_sha256": acceptance_receipt[
            "operator_gate_sha256"
        ],
        "accepted_grounding_horizon_gate_sha256": acceptance_receipt[
            "grounding_horizon_gate_sha256"
        ],
        "runtime_stack_manifest_sha256": acceptance_receipt[
            "runtime_stack_manifest_sha256"
        ],
        "gpu_hardware_manifest_sha256": acceptance_receipt[
            "gpu_hardware_manifest_sha256"
        ],
        "source_association_manifest_sha256": source_association[
            "manifest_sha256"
        ],
        "source_tree_sha256": source_association["local_tree_sha256"],
        "runtime_feasibility_setting": {"enforce_eager": True},
        "scope": "registered_development_llm_conditions",
        "applies_symmetrically_to": ["C1", "C2", "A-FixedSelect", "LLM_ablations"],
        "held_out_execution_allowed": False,
        "held_out_requires_separate_development_and_review_gates": True,
        "mixed_model_candidates_forbidden": True,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _micro_pilot_acceptance_receipt(
    result: Mapping[str, object],
    *,
    source_association: Mapping[str, object],
) -> dict[str, object]:
    """Freeze only measured, accepted evidence needed to bootstrap development."""

    if result.get("micro_pilot_passed") is not True:
        raise ValueError("development bootstrap requires a passed fallback micro-pilot")
    call_bindings: list[dict[str, object]] = []
    for untyped in cast(Sequence[Mapping[str, object]], result.get("calls", [])):
        decoding = cast(Mapping[str, object], untyped.get("decoding_manifest", {}))
        capability = cast(Mapping[str, object], untyped.get("capability_manifest", {}))
        audit = cast(Mapping[str, object], untyped.get("mechanical_audit", {}))
        call_bindings.append(
            {
                "call_id": untyped.get("call_id"),
                "status": untyped.get("status"),
                "request_hash": untyped.get("request_hash"),
                "response_artifact_hash": untyped.get("response_artifact_hash"),
                "prompt_hash": untyped.get("prompt_hash"),
                "decoding_manifest_hash": decoding.get("content_hash"),
                "output_schema_hash": untyped.get("condition_output_schema_sha256"),
                "capability_manifest_hash": capability.get("content_hash"),
                "mechanical_audit_hash": canonical_sha256(audit),
                "accepted_via_repair": untyped.get("accepted_via_repair", False),
            }
        )
    if len(call_bindings) not in {4, 5}:
        raise ValueError(
            "fallback acceptance receipt requires four bases and at most one repair"
        )
    execution_identity = cast(Mapping[str, object], result["execution_identity"])
    request_family_hash = canonical_sha256(call_bindings)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "phase1_fallback_micro_pilot_acceptance_receipt",
        "micro_pilot_passed": True,
        "accepted_result_manifest_sha256": result["manifest_sha256"],
        "execution_hash": result["execution_hash"],
        "fallback_plan_manifest_sha256": result["fallback_plan_manifest_sha256"],
        "source_association_manifest_sha256": source_association["manifest_sha256"],
        "source_tree_sha256": source_association["local_tree_sha256"],
        "launcher_configuration_sha256": execution_identity[
            "launcher_configuration_sha256"
        ],
        "tokenizer_manifest_sha256": execution_identity["tokenizer_manifest_sha256"],
        "runtime_stack_manifest_sha256": execution_identity[
            "runtime_stack_manifest_sha256"
        ],
        "gpu_hardware_manifest_sha256": execution_identity[
            "gpu_hardware_manifest_sha256"
        ],
        "request_family_hash": request_family_hash,
        "request_bindings": call_bindings,
        "operator_gate_sha256": canonical_sha256(result["operator_coverage_gate"]),
        "grounding_horizon_gate_sha256": canonical_sha256(
            {"passed": result["grounding_horizon_gate"]}
        ),
        "timing_gate_sha256": canonical_sha256(result["timing_gate"]),
        "resource_gate_sha256": canonical_sha256(result["cumulative_resource_gate"]),
        "forecast_sha256": canonical_sha256(
            result["post_fallback_full_manifest_forecast"]
        ),
        "controller_resume_gate_sha256": canonical_sha256(
            result["controller_resume_gate"]
        ),
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


@dataclass(slots=True)
class FallbackAcceptanceRunner:
    root: Path
    run_id: str
    service: VLLMService
    ledger: Ledger
    artifacts: ArtifactStore
    resource_sampler: ResourceSampler
    tokenizer: PackingTokenizer
    tokenizer_manifest: TokenizerManifest
    checkpoint_path: Path
    activation_certificate: Mapping[str, object]
    replacement_receipt: Mapping[str, object]
    snapshot_manifest: Mapping[str, object]
    source_association: Mapping[str, object]
    pre_fallback_gpu_accounting: Mapping[str, object] | None = None
    development_adopter: DevelopmentContinuationAdopter | None = None
    runtime_stack: RuntimeStackManifest | None = None
    gpu_hardware: GPUHardwareIdentity | None = None
    _last_shutdown_uptime: ServiceUptime | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or len(self.run_id) > 96
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in self.run_id
            )
        ):
            raise ValueError("run_id must be a lowercase public-safe identifier")
        self.root = self.root.resolve(strict=True)
        if self.service.configuration.model_candidate != "fallback":
            raise ValueError("fallback runner cannot use the primary model")

    def _development_registration(
        self,
        *,
        required: bool,
    ) -> DevelopmentAdopterRegistration | None:
        adopter = self.development_adopter
        if adopter is None:
            if required:
                raise RuntimeError(
                    "fallback GPU execution requires a registered live development adopter"
                )
            return None
        if not isinstance(adopter, DevelopmentContinuationAdopter):
            raise TypeError("development adopter does not implement the frozen protocol")
        registration = adopter.registration()
        if not isinstance(registration, DevelopmentAdopterRegistration):
            raise TypeError("development adopter returned an untyped registration")
        try:
            registration = DevelopmentAdopterRegistration.model_validate(
                registration.model_dump(mode="python")
            )
        except ValidationError as exc:
            raise ValueError("development adopter registration is not canonical") from exc
        if registration.expected_development_call_count != 24:
            raise ValueError("development adopter does not bind the exact 24-call block")
        return registration

    def _execution_identity(self, plan_hash: str) -> dict[str, object]:
        registration = self._development_registration(required=False)
        return {
            "fallback_plan_manifest_sha256": plan_hash,
            "activation_certificate_sha256": self.activation_certificate["manifest_sha256"],
            "cache_replacement_receipt_sha256": self.replacement_receipt["manifest_sha256"],
            "snapshot_manifest_sha256": self.snapshot_manifest["manifest_sha256"],
            "source_association_manifest_sha256": self.source_association[
                "manifest_sha256"
            ],
            "source_tree_sha256": self.source_association["local_tree_sha256"],
            "pre_fallback_gpu_accounting_sha256": (
                None
                if self.pre_fallback_gpu_accounting is None
                else self.pre_fallback_gpu_accounting["manifest_sha256"]
            ),
            "launcher_configuration_sha256": self.service.configuration.configuration_hash,
            "tokenizer_manifest_sha256": self.tokenizer_manifest.manifest_sha256,
            "runtime_stack_manifest_sha256": (
                None
                if self.runtime_stack is None
                else self.runtime_stack.public_manifest()["manifest_sha256"]
            ),
            "gpu_hardware_manifest_sha256": (
                None
                if self.gpu_hardware is None
                else self.gpu_hardware.public_manifest()["manifest_sha256"]
            ),
            "development_adopter_registration_hash": (
                None if registration is None else registration.content_hash
            ),
        }

    def _initial_state(self, execution_hash: str) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "completed_call_ids": [],
            "accepted_outputs": {},
            "reserve_consumption": [],
            "repair_parent_call_id": None,
            "service_start_attempted": False,
            "controller_handoff_complete": False,
            "stage_one_controller_pid": None,
            "stage_two_controller_pid": None,
            "handoff_service_pid": None,
            "controller_process_restart_observed": False,
            "development_continuation_ready": False,
            "development_continuation_completed": False,
            "development_adopter_registration_hash": None,
            "micro_pilot_acceptance_receipt": None,
            "accepted_micro_pilot_result": None,
            "selected_model_freeze": None,
            "development_continuation_bootstrap": None,
            "development_continuation_bootstrap_hash": None,
            "development_preparation": None,
            "development_handoff_hash": None,
            "development_handoff": None,
            "development_continuation_receipt": None,
            "development_execution_result": None,
            "development_handoff_controller_pid": None,
            "development_handoff_service_pid": None,
            "live_allocated_seconds_at_development_handoff": None,
            "live_allocated_seconds_at_controller_handoff": None,
            "orphan_cleanup_completed": False,
            "active_call_id": None,
            "failed_call_id": None,
            "resume_sequence": 0,
        }

    def _load_state(
        self,
        execution_hash: str,
        calls: Sequence[FallbackCallSpec],
    ) -> dict[str, object]:
        if not self.checkpoint_path.exists():
            state = self._initial_state(execution_hash)
            _private_atomic_json(self.checkpoint_path, state)
            return state
        state = _load_object(self.checkpoint_path)
        if state.get("run_id") != self.run_id or state.get("execution_hash") != execution_hash:
            raise ValueError("fallback checkpoint does not match its immutable execution")
        completed = state.get("completed_call_ids")
        if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
            raise ValueError("fallback checkpoint completed-call list is invalid")
        if completed != [call.call_id for call in calls[: len(completed)]]:
            raise ValueError("fallback checkpoint is not an ordered call prefix")
        if state.get("failed_call_id") is not None or state.get("active_call_id") is not None:
            raise RuntimeError("fallback checkpoint is terminal after a failed/interrupted attempt")
        return state

    def prepare_controller_restart(self) -> dict[str, object]:
        """Stage one: start exactly once, checkpoint, and leave the model live."""

        registration = self._development_registration(required=True)
        assert registration is not None
        policy = FallbackModelPolicy.load(self.root / "configs/study/fallback_model.json")
        calls = fallback_pilot_calls(policy)
        plan = fallback_plan_manifest(self.root)
        plan_hash = cast(str, plan["manifest_sha256"])
        identity = self._execution_identity(plan_hash)
        execution_hash = canonical_sha256(identity)
        limits = ResourceLimits.load(self.root / "configs/study/resource_limits.json")
        if _resource_gate(self.ledger, limits)["accepted"] is not True:
            raise RuntimeError("cumulative resource or storage ledger already violates a hard gate")
        state = self._load_state(execution_hash, calls)
        if (
            state["service_start_attempted"] is True
            or state["controller_handoff_complete"] is True
            or state["completed_call_ids"]
        ):
            raise RuntimeError("fallback controller-restart preparation is not repeatable")
        state["resume_sequence"] = cast(int, state["resume_sequence"]) + 1
        state["service_start_attempted"] = True
        state["stage_one_controller_pid"] = os.getpid()
        state["development_adopter_registration_hash"] = registration.content_hash
        self._save(state)
        service_checkpoint = self.checkpoint_path.with_name(
            self.checkpoint_path.name + ".service"
        )
        try:
            self.service.start(
                session_id=self.run_id,
                event_id=f"{self.run_id}-service-start-001",
                watchdog_seconds=180,
                remaining_required_seconds=sum(call.watchdog_seconds for call in calls) + 90,
            )
            self.resource_sampler.sample(
                sample_id=f"{self.run_id}-stage-one-after-load",
                root_pid=self.service.pid,
                gpu_event_id=f"{self.run_id}-service-start-001",
            )
            self.service.write_resume_checkpoint(service_checkpoint)
            checkpoint = _load_object(service_checkpoint)
            if checkpoint.get("controller_pid") != os.getpid():
                raise RuntimeError("service checkpoint did not bind the stage-one controller")
            state["handoff_service_pid"] = self.service.pid
            state["controller_handoff_complete"] = True
            state["live_allocated_seconds_at_controller_handoff"] = max(
                self.ledger.gpu_summary().total_allocated_seconds,
                float(getattr(self.service, "actual_allocated_service_seconds", 0.0)),
            )
            self._save(state)
            self.service.detach_for_controller_restart(service_checkpoint)
        except BaseException:
            self._mark_terminal(state, "controller-restart-prepare")
            self.service.shutdown()
            raise
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_controller_restart_handoff",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "execution_identity": identity,
            "fallback_plan_manifest_sha256": plan_hash,
            "controller_stage": "prepare_complete",
            "controller_restart_handoff_pending": True,
            "model_service_left_live_for_controller_restart": True,
            "physical_service_live": True,
            "vllm_service_stopped": False,
            "live_allocated_seconds_at_controller_handoff": state[
                "live_allocated_seconds_at_controller_handoff"
            ],
            "model_process_restart": False,
            "controller_process_restart": False,
            "completed_base_call_count": 0,
            "repair_attempt_count": 0,
            "reserve_consumption": [],
            "micro_pilot_passed": False,
            "phase1_gate_passed": False,
            "gate_passed": False,
            "next_required_stage": "run_under_a_different_controller_pid",
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}

    def cleanup_orphan(self) -> dict[str, object]:
        """Adopt the exact checkpointed service and terminate it without inference."""

        plan = fallback_plan_manifest(self.root)
        plan_hash = cast(str, plan["manifest_sha256"])
        identity = self._execution_identity(plan_hash)
        execution_hash = canonical_sha256(identity)
        if not self.checkpoint_path.exists():
            raise RuntimeError("orphan cleanup requires a runner checkpoint")
        state = _load_object(self.checkpoint_path)
        if (
            state.get("run_id") != self.run_id
            or state.get("execution_hash") != execution_hash
        ):
            raise ValueError("orphan cleanup checkpoint identifies another execution")
        checkpoint = self.checkpoint_path.with_name(self.checkpoint_path.name + ".service")
        adopted = False
        uptime = None
        try:
            adopted = self.service.resume_from_checkpoint(
                checkpoint,
                allow_same_controller_cleanup=True,
            )
        finally:
            # Even malformed/stale checkpoint failures must exercise the
            # endpoint-aware shutdown path before they escape this controller.
            uptime = self.service.shutdown()
            if uptime is not None:
                self._last_shutdown_uptime = uptime
        state["orphan_cleanup_completed"] = True
        state["active_call_id"] = None
        state["failed_call_id"] = "orphan-cleanup-requested"
        self._save(state)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_orphan_cleanup_result",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "checkpoint_service_adopted": adopted,
            "physical_shutdown_verified": self.service.state is ServiceState.STOPPED,
            "inference_executed": False,
            "runtime": public_runtime_manifest(
                launcher=self.service.configuration,
                tokenizer=self.tokenizer_manifest,
                runtime_stack=self.runtime_stack,
                gpu_hardware=self.gpu_hardware,
                resource_samples=self.resource_sampler.samples,
                uptime=uptime,
                ledger=self.ledger,
            ),
            "micro_pilot_passed": False,
            "phase1_gate_passed": False,
            "gate_passed": False,
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}

    def ensure_service_stopped_after_failure(self) -> ServiceUptime | None:
        """Shut down an owned service or adopt-clean a detached handoff."""

        service_checkpoint = self.checkpoint_path.with_name(
            self.checkpoint_path.name + ".service"
        )
        state: dict[str, object] | None = None
        if self.checkpoint_path.exists():
            state = _load_object(self.checkpoint_path)
        detached_handoff = (
            state is not None
            and state.get("controller_handoff_complete") is True
            and state.get("orphan_cleanup_completed") is not True
            and service_checkpoint.exists()
            and self.service.state is ServiceState.STOPPED
        )
        if detached_handoff:
            try:
                self.service.resume_from_checkpoint(
                    service_checkpoint,
                    allow_same_controller_cleanup=True,
                )
            finally:
                # If adoption initialized even a partial handle, shutdown must
                # still verify both process-tree and endpoint absence.
                uptime = self.service.shutdown()
                if uptime is not None:
                    self._last_shutdown_uptime = uptime
            assert state is not None
            state["orphan_cleanup_completed"] = True
            state["active_call_id"] = None
            state["failed_call_id"] = state.get("failed_call_id") or "failure-cleanup"
            self._save(state)
            return uptime

        uptime = self.service.shutdown()
        if uptime is not None:
            self._last_shutdown_uptime = uptime
        elif self._last_shutdown_uptime is not None:
            uptime = self._last_shutdown_uptime
        return uptime

    def _save(self, state: Mapping[str, object]) -> None:
        _private_atomic_json(self.checkpoint_path, state)

    def _reserve(
        self,
        state: dict[str, object],
        *,
        call_id: str,
        reserve_call_class: str,
        watchdog_seconds: int,
    ) -> dict[str, object]:
        reservation_id = f"{self.run_id}:{call_id}"
        rows = cast(list[dict[str, object]], state["reserve_consumption"])
        if any(row.get("reservation_id") == reservation_id for row in rows):
            raise RuntimeError("a fallback reserve reservation cannot be attempted twice")
        receipt: dict[str, object] = {
            "reservation_id": reservation_id,
            "call_id": call_id,
            "reserve_call_class": reserve_call_class,
            "watchdog_seconds": watchdog_seconds,
        }
        rows.append(receipt)
        state["active_call_id"] = call_id
        self._save(state)
        return receipt

    def _mark_terminal(self, state: dict[str, object], call_id: str) -> None:
        state["active_call_id"] = None
        state["failed_call_id"] = call_id
        self._save(state)

    def _record_failed_transport(
        self,
        *,
        call: FallbackCallSpec,
        request: GuidedJSONRequest,
        job_id: str,
        attempt_id: str,
        event_id: str,
        exc: Exception,
        retry_class: RetryClass,
        call_role: ModelCallRole,
    ) -> float:
        event = _event_for(self.ledger, event_id)
        allocated = 0.0 if event is None else event.allocated_seconds
        if event is not None:
            self.ledger.record_model_call(
                model_call_id=attempt_id.removesuffix("-attempt"),
                job_id=job_id,
                attempt_id=attempt_id,
                gpu_event_id=event_id,
                backend=ModelBackend.VLLM_GPU,
                call_role=call_role,
                retry_class=retry_class,
                model_manifest_hash=self.service.configuration.configuration_hash,
                decoding_manifest_hash=request.decoding.content_hash,
                request_hash=request.request_hash,
                response_artifact_hash=None,
                construction_unit_hash=_base_construction_unit_hash(call),
                served_context_count=1,
                prompt_tokens=0,
                completion_tokens=0,
                allocated_gpu_seconds=allocated,
                successful=False,
            )
        self.ledger.record_failure(
            attempt_id=attempt_id,
            failure_kind=_failure_kind(exc),
            message="Fallback micro-pilot model call failed",
            details={"exception_type": type(exc).__name__, "call_id": call.call_id},
        )
        return allocated

    def _resume_completed(
        self,
        *,
        state: Mapping[str, object],
        call: FallbackCallSpec,
    ) -> tuple[
        dict[str, object],
        dict[str, object],
        tuple[TimingObservation, ...],
    ]:
        accepted = cast(Mapping[str, Mapping[str, object]], state["accepted_outputs"])[
            call.call_id
        ]
        base_request = build_fallback_acceptance_request(
            root=self.root,
            call=call,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
        )
        base_call = call.acceptance_call()
        base_attempt_id = f"{self.run_id}-{call.call_id}-attempt"
        base_model_call = self.ledger.get_model_call(f"{self.run_id}-{call.call_id}")
        repaired = accepted.get("repaired") is True
        base_generation = self._validate_resumed_model_call(
            model_call_id=base_model_call.model_call_id,
            request=base_request,
            expected_attempt_id=base_attempt_id,
            expected_attempt_kind=AttemptKind.BASE,
            expected_parent_attempt_id=None,
            expected_call_role=ModelCallRole.PILOT,
            expected_retry_class=call.retry_class,
            expected_event_kind=GpuEventKind.FALLBACK_TEST,
            expected_reserve_class=call.reserve_call_class,
            expected_reservation_id=f"{self.run_id}:{call.call_id}",
            expected_construction_unit_hash=_base_construction_unit_hash(call),
            expected_success=not repaired,
        )
        timings = [
            TimingObservation(
                call_class=call.forecast_call_class,
                allocated_seconds=base_model_call.allocated_gpu_microseconds / 1_000_000,
            )
        ]
        result: dict[str, object]
        if repaired:
            if accepted.get("invalid_base_artifact_hash") != base_model_call.response_artifact_hash:
                raise RuntimeError("resumed repair changed its invalid base artifact")
            diagnostics = cast(Sequence[Mapping[str, object]], accepted["diagnostics"])
            try:
                base_audit = validate_acceptance_generation(
                    root=self.root,
                    call=base_call,
                    parsed_object=base_generation.parsed_object,
                    authoritative_prompt_tokens=base_generation.prompt_tokens,
                    authoritative_completion_tokens=base_generation.completion_tokens,
                )
                base_audit["operator_behavior"] = _require_call_operator_coverage(
                    call,
                    base_audit,
                    base_generation.parsed_object,
                    root=self.root,
                )
            except Exception as exc:
                regenerated_diagnostics = _fact_free_repair_diagnostics(exc)
            else:
                raise RuntimeError("resumed repair has a mechanically valid base output")
            if [dict(item) for item in diagnostics] != [
                dict(item) for item in regenerated_diagnostics
            ]:
                raise RuntimeError("resumed repair diagnostics changed")
            repair_request = build_fallback_repair_request(
                root=self.root,
                base_call=base_call,
                base_request=base_request,
                invalid_draft=base_generation.parsed_object,
                diagnostics=diagnostics,
                tokenizer=self.tokenizer,
                tokenizer_manifest=self.tokenizer_manifest,
            )
            repair_id = f"{call.call_id}-repair-01"
            repair_model_call_id = f"{self.run_id}-{repair_id}"
            if accepted.get("model_call_id") != repair_model_call_id:
                raise RuntimeError("resumed repair model-call identity changed")
            repair_model_call = self.ledger.get_model_call(repair_model_call_id)
            generation = self._validate_resumed_model_call(
                model_call_id=repair_model_call_id,
                request=repair_request,
                expected_attempt_id=f"{repair_model_call_id}-attempt",
                expected_attempt_kind=AttemptKind.REPAIR,
                expected_parent_attempt_id=base_attempt_id,
                expected_call_role=ModelCallRole.REPAIR,
                expected_retry_class=RetryClass.SHORT,
                expected_event_kind=GpuEventKind.REPAIR,
                expected_reserve_class="reserve_short",
                expected_reservation_id=f"{self.run_id}:{repair_id}",
                expected_construction_unit_hash=_repair_construction_unit_hash(
                    call,
                    base_attempt_id,
                ),
                expected_success=True,
            )
            audit = validate_acceptance_generation(
                root=self.root,
                call=base_call,
                parsed_object=generation.parsed_object,
                authoritative_prompt_tokens=generation.prompt_tokens,
                authoritative_completion_tokens=generation.completion_tokens,
            )
            audit["operator_behavior"] = _require_call_operator_coverage(
                call,
                audit,
                generation.parsed_object,
                root=self.root,
            )
            preservation = validate_repair_preservation(
                base_draft=base_generation.parsed_object,
                repaired_draft=generation.parsed_object,
                diagnosed_paths=tuple(cast(str, item["path"]) for item in diagnostics),
            )
            preservation.raise_for_errors()
            timings.append(
                TimingObservation(
                    call_class="acceptance_repair",
                    allocated_seconds=(
                        repair_model_call.allocated_gpu_microseconds / 1_000_000
                    ),
                )
            )
            result = {
                "call_id": call.call_id,
                "status": AcceptanceStatus.RESUMED.value,
                **generation.public_manifest(),
                **_request_public_metadata(repair_request),
                "base_attempt": {
                    **base_generation.public_manifest(),
                    **_request_public_metadata(base_request),
                    "response_artifact_hash": base_model_call.response_artifact_hash,
                },
                "accepted_via_repair": True,
                "response_artifact_hash": repair_model_call.response_artifact_hash,
                "invalid_base_artifact_hash": base_model_call.response_artifact_hash,
                "reserve_call_class": "reserve_short",
                "repair_number": 1,
                "diagnostic_count": len(diagnostics),
                "scorer_only_diagnostics_exposed": False,
                "mechanical_audit": audit,
            }
        else:
            if accepted.get("model_call_id") != base_model_call.model_call_id:
                raise RuntimeError("resumed base model-call identity changed")
            generation = base_generation
            audit = validate_acceptance_generation(
                root=self.root,
                call=base_call,
                parsed_object=generation.parsed_object,
                authoritative_prompt_tokens=generation.prompt_tokens,
                authoritative_completion_tokens=generation.completion_tokens,
            )
            audit["operator_behavior"] = _require_call_operator_coverage(
                call,
                audit,
                generation.parsed_object,
                root=self.root,
            )
            result = {
                "call_id": call.call_id,
                "status": AcceptanceStatus.RESUMED.value,
                **generation.public_manifest(),
                **_request_public_metadata(base_request),
                "accepted_via_repair": False,
                "response_artifact_hash": base_model_call.response_artifact_hash,
                "reserve_call_class": call.reserve_call_class,
                "mechanical_audit": audit,
            }
        return result, audit, tuple(timings)

    def _validate_resumed_model_call(
        self,
        *,
        model_call_id: str,
        request: GuidedJSONRequest,
        expected_attempt_id: str,
        expected_attempt_kind: AttemptKind,
        expected_parent_attempt_id: str | None,
        expected_call_role: ModelCallRole,
        expected_retry_class: RetryClass,
        expected_event_kind: GpuEventKind,
        expected_reserve_class: str,
        expected_reservation_id: str,
        expected_construction_unit_hash: str,
        expected_success: bool,
    ) -> GenerationResult:
        model_call = self.ledger.get_model_call(model_call_id)
        expected_metadata = {
            "attempt_id": expected_attempt_id,
            "backend": ModelBackend.VLLM_GPU,
            "call_role": expected_call_role,
            "retry_class": expected_retry_class,
            "model_manifest_hash": self.service.configuration.configuration_hash,
            "decoding_manifest_hash": request.decoding.content_hash,
            "request_hash": request.request_hash,
            "construction_unit_hash": expected_construction_unit_hash,
            "served_context_count": 1,
            "successful": expected_success,
        }
        mismatches = [
            name
            for name, expected in expected_metadata.items()
            if getattr(model_call, name) != expected
        ]
        if model_call.response_artifact_hash is None:
            mismatches.append("response_artifact_hash")
        lineage = self.ledger.attempt_lineage(expected_attempt_id)
        attempt = lineage[-1]
        if (
            attempt.attempt_kind is not expected_attempt_kind
            or attempt.parent_attempt_id != expected_parent_attempt_id
            or attempt.input_hash != request.request_hash
            or attempt.config_hash != request.decoding.content_hash
            or attempt.seed != request.decoding.seed
            or attempt.job_id != model_call.job_id
            or len(lineage) != (1 if expected_parent_attempt_id is None else 2)
        ):
            mismatches.append("attempt_lineage")
        event = None if model_call.gpu_event_id is None else _event_for(
            self.ledger,
            model_call.gpu_event_id,
        )
        if (
            event is None
            or event.event_kind is not expected_event_kind
            or event.job_id != model_call.job_id
            or event.attempt_id != expected_attempt_id
        ):
            mismatches.append("gpu_event_lineage")
        else:
            details = json.loads(event.details_json)
            if (
                details.get("request_id") != request.request_id
                or details.get("request_hash") != request.request_hash
                or details.get("reserve_call_class") != expected_reserve_class
                or details.get("reserve_reservation_id") != expected_reservation_id
            ):
                mismatches.append("gpu_event_request_or_reserve_identity")
        if mismatches:
            raise RuntimeError(
                "resumed fallback model call metadata changed: " + ", ".join(mismatches)
            )
        assert model_call.response_artifact_hash is not None
        stored = self.artifacts.blobs.read_bytes(
            self.ledger.get_artifact(model_call.response_artifact_hash)
        )
        generation = _stored_generation(stored, request)
        _validate_generation_envelope(generation, request)
        if (
            generation.response_sha256 != model_call.response_artifact_hash
            or generation.prompt_tokens != model_call.prompt_tokens
            or generation.completion_tokens != model_call.completion_tokens
        ):
            raise RuntimeError("resumed fallback response tokens or artifact hash changed")
        return generation

    def _run_development_continuation(
        self,
        *,
        provisional_result: Mapping[str, object],
        state: dict[str, object],
        execution_hash: str,
        registration: DevelopmentAdopterRegistration,
    ) -> DevelopmentContinuationReceipt:
        """Prepare query-blind inputs, then offer the exact live service adapter."""

        if self.service.state is not ServiceState.READY:
            raise RuntimeError("development handoff requires the accepted service to be live")
        service_pid = self.service.pid
        service_checkpoint_path = self.checkpoint_path.with_name(
            self.checkpoint_path.name + ".service"
        )
        checkpoint = _load_object(service_checkpoint_path)
        start_ticks = checkpoint.get("process_start_ticks")
        gpu_session_event_id = checkpoint.get("accounting_session_id")
        if (
            checkpoint.get("pid") != service_pid
            or checkpoint.get("configuration_hash")
            != self.service.configuration.configuration_hash
            or isinstance(start_ticks, bool)
            or not isinstance(start_ticks, int)
            or start_ticks <= 0
            or not isinstance(gpu_session_event_id, str)
            or not gpu_session_event_id
        ):
            raise RuntimeError("development handoff service checkpoint identity is invalid")

        acceptance_receipt = _micro_pilot_acceptance_receipt(
            provisional_result,
            source_association=self.source_association,
        )
        selected_freeze = _selected_model_freeze(
            activation_certificate=self.activation_certificate,
            replacement_receipt=self.replacement_receipt,
            snapshot_manifest=self.snapshot_manifest,
            launcher=self.service.configuration,
            tokenizer=self.tokenizer_manifest,
            acceptance_receipt=acceptance_receipt,
            source_association=self.source_association,
        )
        allocated_before = max(
            self.ledger.gpu_summary().total_allocated_seconds,
            float(getattr(self.service, "actual_allocated_service_seconds", 0.0)),
        )
        bootstrap = DevelopmentContinuationBootstrap(
            bootstrap_id=f"{self.run_id}-development-bootstrap",
            owner_run_id=self.run_id,
            fallback_execution_hash=execution_hash,
            accepted_fallback_result_hash=cast(
                str,
                provisional_result["manifest_sha256"],
            ),
            micro_pilot_acceptance_receipt=acceptance_receipt,
            micro_pilot_acceptance_receipt_sha256=cast(
                str,
                acceptance_receipt["manifest_sha256"],
            ),
            adopter_registration_hash=registration.content_hash,
            selected_model_freeze=selected_freeze,
            selected_model_freeze_hash=cast(str, selected_freeze["manifest_sha256"]),
            source_tree_hash=cast(str, self.source_association["local_tree_sha256"]),
            service_pid=service_pid,
            service_start_ticks=start_ticks,
            gpu_session_event_id=gpu_session_event_id,
            launcher_configuration_hash=self.service.configuration.configuration_hash,
            model_snapshot_manifest_hash=cast(
                str,
                self.snapshot_manifest["manifest_sha256"],
            ),
            service_checkpoint_sha256=_file_sha256(service_checkpoint_path),
            allocated_gpu_seconds_before_preparation=allocated_before,
        )
        # The accepted result and selected-model freeze are durable before any
        # adopter code prepares development inputs.  This removes any possible
        # dependency of the freeze on development outcomes.
        state["micro_pilot_acceptance_receipt"] = acceptance_receipt
        state["accepted_micro_pilot_result"] = dict(provisional_result)
        state["selected_model_freeze"] = selected_freeze
        state["development_continuation_bootstrap"] = bootstrap.model_dump(mode="json")
        state["development_continuation_bootstrap_hash"] = bootstrap.content_hash
        self._save(state)

        adopter = self.development_adopter
        assert adopter is not None
        if adopter.registration() != registration:
            raise RuntimeError("development adopter registration changed before preparation")
        prepared = adopter.prepare(bootstrap)
        if not isinstance(prepared, PreparedDevelopmentContinuation):
            raise TypeError("development adopter returned an untyped preparation")
        if not isinstance(prepared.call_manifest, DevelopmentCallManifest):
            raise TypeError("development preparation lacks a typed call manifest")
        if not isinstance(prepared.prequery_inputs, DevelopmentPrequeryInputs):
            raise TypeError("development preparation lacks typed prequery inputs")
        if not isinstance(prepared.forecast_control, ForecastControl):
            raise TypeError("development preparation lacks typed forecast control")
        try:
            DevelopmentCallManifest.model_validate(
                prepared.call_manifest.model_dump(mode="python")
            )
            DevelopmentPrequeryInputs.model_validate(
                prepared.prequery_inputs.model_dump(mode="python")
            )
            ForecastControl.model_validate(
                prepared.forecast_control.model_dump(mode="python")
            )
        except ValidationError as exc:
            raise RuntimeError("development preparation is not canonical") from exc
        if prepared.query_access_event_count != 0:
            raise RuntimeError("development preparation opened a query before adoption")
        if (
            not prepared.execution_id
            or len(prepared.execution_id) > 96
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in prepared.execution_id
            )
        ):
            raise RuntimeError("development preparation has an unsafe execution ID")
        if prepared.call_manifest.content_hash != registration.development_call_manifest_sha256:
            raise RuntimeError("development preparation changed the registered call manifest")
        if (
            prepared.prequery_inputs.selected_model_freeze_hash
            != bootstrap.selected_model_freeze_hash
            or prepared.prequery_inputs.source_tree_hash != bootstrap.source_tree_hash
        ):
            raise RuntimeError(
                "development prequery inputs differ from the selected freeze or source tree"
            )
        if (
            prepared.call_manifest.gpu_call_inventory_file_sha256
            != prepared.forecast_control.gpu_call_inventory_file_sha256
        ):
            raise RuntimeError("development forecast changed the registered GPU inventory")
        if not prepared.forecast_control.complete_post_development_manifest_included:
            raise RuntimeError("development forecast omits post-development mandatory work")

        development_checkpoint = prepared.checkpoint_path.resolve(strict=False)
        if prepared.checkpoint_path.is_symlink():
            raise RuntimeError("development checkpoint cannot be a symlink")
        if not development_checkpoint.parent.is_dir():
            raise RuntimeError("development checkpoint parent does not exist")
        allowed_checkpoint_roots = (
            self.root,
            self.checkpoint_path.parent.resolve(strict=True),
        )
        if not any(
            development_checkpoint.is_relative_to(allowed_root)
            for allowed_root in allowed_checkpoint_roots
        ):
            raise RuntimeError("development checkpoint is outside controlled run roots")
        checkpoint_before = (
            _file_sha256(development_checkpoint)
            if development_checkpoint.is_file()
            else None
        )
        adapter = prepared.service_adapter
        if not isinstance(adapter, InjectedLiveDevelopmentService):
            raise TypeError("development preparation lacks the narrow live-service adapter")
        exposed_lifecycle_members = tuple(
            name
            for name in FORBIDDEN_ADAPTER_LIFECYCLE_MEMBERS
            if hasattr(adapter, name)
        )
        if exposed_lifecycle_members:
            raise RuntimeError(
                "development adapter exposes forbidden lifecycle members: "
                + ", ".join(exposed_lifecycle_members)
            )
        identity_before = adapter.identity()
        if not isinstance(identity_before, LiveServiceIdentity):
            raise TypeError("development adapter returned an untyped service identity")
        expected_identity = LiveServiceIdentity(
            owner_run_id=self.run_id,
            service_pid=service_pid,
            service_start_ticks=start_ticks,
            gpu_session_event_id=gpu_session_event_id,
            launcher_configuration_hash=self.service.configuration.configuration_hash,
            model_snapshot_hash=cast(str, self.snapshot_manifest["manifest_sha256"]),
            selected_model_freeze_hash=bootstrap.selected_model_freeze_hash,
            source_execution_hash=prepared.prequery_inputs.source_tree_hash,
        )
        if identity_before != expected_identity:
            raise RuntimeError("development adapter identifies another live model service")
        adapter_allocated_before = adapter.allocated_gpu_seconds()
        if (
            not math.isfinite(adapter_allocated_before)
            or adapter_allocated_before < 0
            or not math.isclose(
                adapter_allocated_before,
                allocated_before,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise RuntimeError("development adapter starts from another GPU counter")

        handoff = DevelopmentContinuationHandoff(
            handoff_id=f"{self.run_id}-development-handoff",
            owner_run_id=self.run_id,
            bootstrap_hash=bootstrap.content_hash,
            fallback_execution_hash=execution_hash,
            accepted_fallback_result_hash=bootstrap.accepted_fallback_result_hash,
            source_execution_hash=expected_identity.source_execution_hash,
            micro_pilot_acceptance_receipt_sha256=cast(
                str,
                acceptance_receipt["manifest_sha256"],
            ),
            adopter_registration_hash=registration.content_hash,
            selected_model_freeze=selected_freeze,
            selected_model_freeze_hash=cast(str, selected_freeze["manifest_sha256"]),
            live_service_identity=expected_identity,
            live_service_identity_hash=expected_identity.content_hash,
            service_pid=service_pid,
            service_start_ticks=start_ticks,
            gpu_session_event_id=gpu_session_event_id,
            launcher_configuration_hash=self.service.configuration.configuration_hash,
            model_snapshot_manifest_hash=cast(
                str,
                self.snapshot_manifest["manifest_sha256"],
            ),
            service_checkpoint_sha256=_file_sha256(service_checkpoint_path),
            allocated_gpu_seconds_before=allocated_before,
            development_call_manifest_hash=prepared.call_manifest.content_hash,
            development_source_plan_hash=prepared.call_manifest.source_plan_hash,
            development_execution_id=prepared.execution_id,
            development_execution_manifest_hash=prepared.execution_manifest_hash,
            gpu_call_inventory_file_sha256=(
                prepared.call_manifest.gpu_call_inventory_file_sha256
            ),
            development_prequery_inputs_hash=prepared.prequery_inputs.content_hash,
            development_checkpoint_path=development_checkpoint.as_posix(),
            development_checkpoint_sha256_before=checkpoint_before,
            forecast_receipt_hash=prepared.forecast_control.forecast_receipt_hash,
            post_development_mandatory_forecast_seconds=(
                prepared.forecast_control.post_development_mandatory_forecast_seconds
            ),
        )
        state["development_continuation_ready"] = True
        state["development_adopter_registration_hash"] = registration.content_hash
        state["development_preparation"] = {
            "call_manifest_hash": prepared.call_manifest.content_hash,
            "source_plan_hash": prepared.call_manifest.source_plan_hash,
            "execution_id": prepared.execution_id,
            "execution_manifest_hash": prepared.execution_manifest_hash,
            "gpu_call_inventory_file_sha256": (
                prepared.call_manifest.gpu_call_inventory_file_sha256
            ),
            "prequery_inputs_hash": prepared.prequery_inputs.content_hash,
            "checkpoint_path": development_checkpoint.as_posix(),
            "checkpoint_sha256_before": checkpoint_before,
            "forecast_receipt_hash": prepared.forecast_control.forecast_receipt_hash,
            "post_development_mandatory_forecast_seconds": (
                prepared.forecast_control.post_development_mandatory_forecast_seconds
            ),
            "service_identity_hash": identity_before.content_hash,
            "query_access_event_count": 0,
        }
        state["development_handoff_hash"] = handoff.content_hash
        state["development_handoff"] = handoff.model_dump(mode="json")
        state["development_handoff_controller_pid"] = os.getpid()
        state["development_handoff_service_pid"] = service_pid
        state["live_allocated_seconds_at_development_handoff"] = allocated_before
        self._save(state)

        if adopter.registration() != registration:
            raise RuntimeError("development adopter registration changed before handoff")
        development_result = adopter.adopt_and_run(handoff, adapter)
        if not isinstance(development_result, DevelopmentExecutionResult):
            raise TypeError("development adopter returned an untyped execution result")
        try:
            development_result = DevelopmentExecutionResult.model_validate(
                development_result.model_dump(mode="python")
            )
        except ValidationError as exc:
            raise RuntimeError(
                "development adopter returned a noncanonical execution result"
            ) from exc
        identity_after = adapter.identity()
        if not isinstance(identity_after, LiveServiceIdentity):
            raise TypeError("development adapter returned an untyped final identity")
        adapter_allocated_after = adapter.allocated_gpu_seconds()
        allocated_after = max(
            self.ledger.gpu_summary().total_allocated_seconds,
            float(getattr(self.service, "actual_allocated_service_seconds", 0.0)),
        )
        checkpoint_after = (
            _file_sha256(development_checkpoint)
            if development_checkpoint.is_file()
            and not development_checkpoint.is_symlink()
            else None
        )
        exact_itt = (
            len(development_result.itt_records) == 24
            and tuple(
                (
                    row.ordinal,
                    row.call_id,
                    row.call_class,
                    row.condition,
                    row.unit_id,
                )
                for row in development_result.itt_records
            )
            == tuple(
                (
                    call.ordinal,
                    call.call_id,
                    call.call_class,
                    call.condition,
                    call.unit_id,
                )
                for call in prepared.call_manifest.calls
            )
        )
        every_started = exact_itt and all(
            row.request_start_state is RequestStartState.STARTED
            for row in development_result.itt_records
        )
        every_succeeded = exact_itt and all(
            row.outcome is RunOutcome.SUCCEEDED
            for row in development_result.itt_records
        )
        observed_access_by_stage = {
            event.stage_manifest_hash: event
            for event in development_result.query_access_events
        }
        expected_access_by_stage = {
            call.query_stage.staging_manifest_hash: (
                call.query_stage.query_artifact_hash,
                prepared.prequery_inputs.binding_for(call.unit_id).snapshot_hash,
            )
            for call in prepared.call_manifest.calls
            if call.query_stage is not None
        }
        twelve_access_events = (
            len(development_result.query_access_events) == 12
            and set(observed_access_by_stage) == set(expected_access_by_stage)
            and development_result.prequery_barrier_hash is not None
            and all(
                event.execution_id == development_result.execution_id
                and event.prequery_barrier_hash
                == development_result.prequery_barrier_hash
                and (
                    event.query_artifact_hash,
                    event.snapshot_hash,
                )
                == expected_access_by_stage[event.stage_manifest_hash]
                for event in development_result.query_access_events
            )
        )
        itt_query_access_lineage = exact_itt and all(
            (
                row.query_access_event_hash is None
                if call.query_stage is None
                else row.query_access_event_hash
                == (
                    None
                    if observed_access_by_stage.get(
                        call.query_stage.staging_manifest_hash
                    )
                    is None
                    else observed_access_by_stage[
                    call.query_stage.staging_manifest_hash
                    ].content_hash
                )
            )
            for call, row in zip(
                prepared.call_manifest.calls,
                development_result.itt_records,
                strict=True,
            )
        )
        forecast_admitted = (
            development_result.admission_failure_code is None
            and development_result.forecast.scheduled_admitted
            and development_result.forecast.below_hard_stop
        )
        expected_gate = {
            "exact_24_call_manifest": len(prepared.call_manifest.calls) == 24,
            "every_planned_call_has_itt_row": exact_itt,
            "every_planned_request_started": every_started,
            "every_planned_call_succeeded": every_succeeded,
            "twelve_query_access_events": twelve_access_events,
            "same_live_service_identity": identity_after == identity_before,
            "forecast_admitted": forecast_admitted,
        }
        mismatches: list[str] = []
        expected_result_lineage = {
            "execution_id": handoff.development_execution_id,
            "execution_manifest_hash": handoff.development_execution_manifest_hash,
            "call_manifest_hash": handoff.development_call_manifest_hash,
            "source_plan_hash": handoff.development_source_plan_hash,
            "service_identity_hash": handoff.live_service_identity_hash,
            "prequery_inputs_hash": handoff.development_prequery_inputs_hash,
        }
        mismatches.extend(
            name
            for name, expected in expected_result_lineage.items()
            if getattr(development_result, name) != expected
        )
        if not itt_query_access_lineage:
            mismatches.append("itt_query_access_lineage")
        mismatches.extend(
            f"gate.{name}"
            for name, expected in expected_gate.items()
            if getattr(development_result.gate, name) != expected
        )
        if (
            development_result.forecast.forecast_receipt_hash
            != handoff.forecast_receipt_hash
            or development_result.forecast.gpu_call_inventory_file_sha256
            != handoff.gpu_call_inventory_file_sha256
            or not math.isclose(
                development_result.forecast.post_development_mandatory_forecast_seconds,
                handoff.post_development_mandatory_forecast_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            mismatches.append("development_forecast_lineage")
        if development_result.phase not in {
            DevelopmentPhase.COMPLETED,
            DevelopmentPhase.COMPLETED_WITH_FAILURES,
        }:
            mismatches.append("development_terminal_phase")
        if (
            development_result.gate.passed
            and development_result.phase is not DevelopmentPhase.COMPLETED
        ):
            mismatches.append("passed_development_phase")
        if adopter.registration() != registration:
            mismatches.append("adopter_registration_after")
        if identity_after != identity_before:
            mismatches.append("service_adapter_identity_after")
        if self.service.state is not ServiceState.READY or self.service.pid != service_pid:
            mismatches.append("live_service_return")
        if (
            not math.isfinite(adapter_allocated_after)
            or not math.isclose(
                adapter_allocated_after,
                allocated_after,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or adapter_allocated_after < adapter_allocated_before
            or checkpoint_after is None
            or not math.isclose(
                development_result.forecast.actual_allocated_seconds_before,
                handoff.allocated_gpu_seconds_before,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                development_result.forecast.actual_allocated_seconds_after,
                allocated_after,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            mismatches.append("allocated_gpu_seconds_after")
        if mismatches:
            raise RuntimeError(
                "development continuation receipt changed live-service identity: "
                + ", ".join(mismatches)
            )
        if checkpoint_after is None:
            raise RuntimeError("development continuation lacks its durable checkpoint")
        receipt = DevelopmentContinuationReceipt(
            handoff_hash=handoff.content_hash,
            bootstrap_hash=bootstrap.content_hash,
            adopter_registration_hash=registration.content_hash,
            fallback_execution_hash=execution_hash,
            source_execution_hash=expected_identity.source_execution_hash,
            selected_model_freeze_hash=handoff.selected_model_freeze_hash,
            live_service_identity_hash_before=identity_before.content_hash,
            live_service_identity_hash_after=identity_after.content_hash,
            service_pid=service_pid,
            service_start_ticks=start_ticks,
            gpu_session_event_id=gpu_session_event_id,
            launcher_configuration_hash=handoff.launcher_configuration_hash,
            model_snapshot_manifest_hash=handoff.model_snapshot_manifest_hash,
            development_execution_id=development_result.execution_id,
            development_execution_manifest_hash=(
                development_result.execution_manifest_hash
            ),
            development_call_manifest_hash=development_result.call_manifest_hash,
            development_source_plan_hash=development_result.source_plan_hash,
            gpu_call_inventory_file_sha256=(
                development_result.forecast.gpu_call_inventory_file_sha256
            ),
            development_prequery_inputs_hash=development_result.prequery_inputs_hash,
            development_result_hash=development_result.content_hash,
            development_checkpoint_sha256_after=checkpoint_after,
            forecast_receipt_hash=development_result.forecast.forecast_receipt_hash,
            every_planned_request_started=every_started,
            every_planned_call_succeeded=every_succeeded,
            twelve_query_access_events=twelve_access_events,
            development_gate_passed=development_result.gate.passed,
            development_forecast_admitted=forecast_admitted,
            scientific_thresholds_passed=(
                development_result.gate.scientific_thresholds_passed
            ),
            allocated_gpu_seconds_after=allocated_after,
            completed_at=datetime.now(UTC),
        )
        state["development_continuation_completed"] = True
        state["development_execution_result"] = development_result.model_dump(mode="json")
        state["development_continuation_receipt"] = receipt.model_dump(mode="json")
        state["live_allocated_seconds_at_development_handoff"] = allocated_after
        self._save(state)
        return receipt

    def run(self) -> dict[str, object]:
        registration = self._development_registration(required=True)
        assert registration is not None
        policy = FallbackModelPolicy.load(self.root / "configs/study/fallback_model.json")
        calls = fallback_pilot_calls(policy)
        plan = fallback_plan_manifest(self.root)
        plan_manifest_hash = cast(str, plan["manifest_sha256"])
        execution_identity = self._execution_identity(plan_manifest_hash)
        execution_hash = canonical_sha256(execution_identity)
        limits = ResourceLimits.load(self.root / "configs/study/resource_limits.json")
        cumulative_before = _resource_gate(self.ledger, limits)
        if cumulative_before["accepted"] is not True:
            raise RuntimeError("cumulative resource or storage ledger already violates a hard gate")
        state = self._load_state(execution_hash, calls)
        if state.get("development_adopter_registration_hash") != registration.content_hash:
            raise RuntimeError("development adopter registration changed across controllers")
        state["resume_sequence"] = cast(int, state["resume_sequence"]) + 1
        self._save(state)

        service_checkpoint = self.checkpoint_path.with_name(
            self.checkpoint_path.name + ".service"
        )
        stage_one_pid = state.get("stage_one_controller_pid")
        if (
            state["service_start_attempted"] is not True
            or state["controller_handoff_complete"] is not True
            or not isinstance(stage_one_pid, int)
            or isinstance(stage_one_pid, bool)
            or not service_checkpoint.exists()
        ):
            raise RuntimeError(
                "fallback calls require a completed controller-restart preparation stage"
            )
        if stage_one_pid == os.getpid():
            raise RuntimeError("fallback stage two must use a different controller process")
        resumed_live_service = False
        try:
            resumed_live_service = self.service.resume_from_checkpoint(service_checkpoint)
            if not resumed_live_service or self.service.state is not ServiceState.READY:
                raise RuntimeError("fallback service checkpoint could not adopt the live process")
            if self.service.pid != state.get("handoff_service_pid"):
                raise RuntimeError("controller restart changed the live model-service PID")
            state["stage_two_controller_pid"] = os.getpid()
            state["controller_process_restart_observed"] = True
            self._save(state)
        except BaseException:
            self.service.shutdown()
            raise

        results: list[dict[str, object]] = []
        successful_audits: list[tuple[FallbackCallSpec, dict[str, object]]] = []
        timing_observations: list[TimingObservation] = []
        repair_attempt_count = 1 if state["repair_parent_call_id"] is not None else 0
        completed = cast(list[str], state["completed_call_ids"])
        uptime = None
        try:
            resource_watchdog = ResourceWatchdog(
                sampler=self.resource_sampler,
                root_pid=self.service.pid,
                sample_prefix=f"{self.run_id}-periodic-{state['resume_sequence']:03d}",
                allocation_guard=getattr(self.service, "require_hard_stop_margin", None),
                on_failure=lambda _: self.service.emergency_stop(),
            )
            resource_watchdog.start()
        except BaseException:
            self.service.shutdown()
            raise
        try:
            self.resource_sampler.sample(
                sample_id=f"{self.run_id}-after-load-{state['resume_sequence']:03d}",
                root_pid=self.service.pid,
                gpu_event_id=f"{self.run_id}-service-start-001",
            )
            for call_index, call in enumerate(calls):
                if call.call_id in completed:
                    result, audit, resumed_timings = self._resume_completed(
                        state=state,
                        call=call,
                    )
                    results.append(result)
                    successful_audits.append((call, audit))
                    timing_observations.extend(resumed_timings)
                    continue
                base_call = call.acceptance_call()
                request = build_fallback_acceptance_request(
                    root=self.root,
                    call=call,
                    tokenizer=self.tokenizer,
                    tokenizer_manifest=self.tokenizer_manifest,
                )
                remaining = sum(later.watchdog_seconds for later in calls[call_index + 1 :]) + (
                    0 if state["repair_parent_call_id"] is not None else 90
                )
                job = self.ledger.create_or_resume_job(
                    {"run_id": self.run_id, "call_id": call.call_id, "plan_hash": execution_hash},
                    release_class=ReleaseClass.PUBLIC,
                )
                attempt_id = f"{self.run_id}-{call.call_id}-attempt"
                self.ledger.record_attempt(
                    attempt_id=attempt_id,
                    job_id=job.job_id,
                    attempt_kind=AttemptKind.BASE,
                    input_hash=request.request_hash,
                    config_hash=request.decoding.content_hash,
                    seed=request.decoding.seed,
                )
                reserve = self._reserve(
                    state,
                    call_id=call.call_id,
                    reserve_call_class=call.reserve_call_class,
                    watchdog_seconds=call.watchdog_seconds,
                )
                event_id = f"{self.run_id}-{call.call_id}-gpu"
                try:
                    generated = self.service.run_fallback_test(
                        request,
                        event_id=event_id,
                        watchdog_seconds=call.watchdog_seconds,
                        reserve_call_class=call.reserve_call_class,
                        reserve_reservation_id=cast(str, reserve["reservation_id"]),
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        remaining_required_seconds=remaining,
                    )
                except Exception as exc:
                    allocated = self._record_failed_transport(
                        call=call,
                        request=request,
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        event_id=event_id,
                        exc=exc,
                        retry_class=call.retry_class,
                        call_role=ModelCallRole.PILOT,
                    )
                    timing_observations.append(
                        TimingObservation(
                            call_class=call.forecast_call_class,
                            allocated_seconds=allocated,
                        )
                    )
                    results.append(
                        {
                            "call_id": call.call_id,
                            "status": AcceptanceStatus.FAILED.value,
                            "exception_type": type(exc).__name__,
                        }
                    )
                    self._mark_terminal(state, call.call_id)
                    break

                event = _event_for(self.ledger, event_id)
                if event is None:
                    raise RuntimeError("fallback model call completed without a GPU event")
                allocated = event.allocated_seconds
                timing_observations.append(
                    TimingObservation(
                        call_class=call.forecast_call_class,
                        allocated_seconds=allocated,
                    )
                )
                artifact = self.artifacts.put_bytes(
                    generated.raw_response,
                    media_type="application/json",
                    release_class=ReleaseClass.PUBLIC,
                )
                common_call = {
                    "model_call_id": f"{self.run_id}-{call.call_id}",
                    "job_id": job.job_id,
                    "attempt_id": attempt_id,
                    "gpu_event_id": event_id,
                    "backend": ModelBackend.VLLM_GPU,
                    "call_role": ModelCallRole.PILOT,
                    "retry_class": call.retry_class,
                    "model_manifest_hash": self.service.configuration.configuration_hash,
                    "decoding_manifest_hash": request.decoding.content_hash,
                    "request_hash": request.request_hash,
                    "response_artifact_hash": artifact.content_hash,
                    "construction_unit_hash": _base_construction_unit_hash(call),
                    "served_context_count": 1,
                    "prompt_tokens": generated.prompt_tokens,
                    "completion_tokens": generated.completion_tokens,
                    "allocated_gpu_seconds": allocated,
                }
                try:
                    _validate_generation_envelope(generated, request)
                    audit = validate_acceptance_generation(
                        root=self.root,
                        call=base_call,
                        parsed_object=generated.parsed_object,
                        authoritative_prompt_tokens=generated.prompt_tokens,
                        authoritative_completion_tokens=generated.completion_tokens,
                    )
                    audit["operator_behavior"] = _require_call_operator_coverage(
                        call,
                        audit,
                        generated.parsed_object,
                        root=self.root,
                    )
                except Exception as validation_error:
                    self.ledger.record_model_call(**common_call, successful=False)
                    self.ledger.record_failure(
                        attempt_id=attempt_id,
                        failure_kind=FailureKind.INVALID_OUTPUT,
                        message="Fallback base output failed mechanical validation",
                        details={
                            "exception_type": type(validation_error).__name__,
                            "call_id": call.call_id,
                        },
                        artifact_hash=artifact.content_hash,
                    )
                    diagnostics = _fact_free_repair_diagnostics(validation_error)
                    can_repair = (
                        bool(diagnostics)
                        and state["repair_parent_call_id"] is None
                        and repair_attempt_count == 0
                    )
                    results.append(
                        {
                            "call_id": call.call_id,
                            "status": AcceptanceStatus.FAILED.value,
                            **generated.public_manifest(),
                            **_request_public_metadata(request),
                            "exception_type": type(validation_error).__name__,
                            "response_artifact_hash": artifact.content_hash,
                            "reserve_call_class": call.reserve_call_class,
                            "repair_diagnostics_hash": canonical_sha256(diagnostics),
                            "repair_triggered": can_repair,
                        }
                    )
                    if not can_repair:
                        self._mark_terminal(state, call.call_id)
                        break
                    state["repair_parent_call_id"] = call.call_id
                    repair_attempt_count += 1
                    repair_result = self._run_repair(
                        state=state,
                        call=call,
                        base_call=base_call,
                        base_request=request,
                        invalid_generated=generated,
                        invalid_artifact_hash=artifact.content_hash,
                        diagnostics=diagnostics,
                        job_id=job.job_id,
                        parent_attempt_id=attempt_id,
                        remaining_required_seconds=sum(
                            later.watchdog_seconds for later in calls[call_index + 1 :]
                        ),
                    )
                    results.append(repair_result[0])
                    timing_observations.append(
                        TimingObservation(
                            call_class="acceptance_repair",
                            allocated_seconds=repair_result[2],
                        )
                    )
                    if repair_result[1] is None:
                        self._mark_terminal(state, f"{call.call_id}-repair-01")
                        break
                    audit = repair_result[1]
                    accepted_model_call_id = cast(str, repair_result[0]["model_call_id"])
                    cast(dict[str, object], state["accepted_outputs"])[call.call_id] = {
                        "model_call_id": accepted_model_call_id,
                        "repaired": True,
                        "invalid_base_artifact_hash": artifact.content_hash,
                        "diagnostics": [dict(item) for item in diagnostics],
                    }
                else:
                    self.ledger.record_model_call(**common_call, successful=True)
                    cast(dict[str, object], state["accepted_outputs"])[call.call_id] = {
                        "model_call_id": f"{self.run_id}-{call.call_id}",
                        "repaired": False,
                    }
                    results.append(
                        {
                            "call_id": call.call_id,
                            "status": AcceptanceStatus.COMPLETED.value,
                            **generated.public_manifest(),
                            **_request_public_metadata(request),
                            "response_artifact_hash": artifact.content_hash,
                            "reserve_call_class": call.reserve_call_class,
                            "mechanical_audit": audit,
                            "accepted_via_repair": False,
                        }
                    )
                completed.append(call.call_id)
                successful_audits.append((call, audit))
                state["active_call_id"] = None
                self._save(state)
                self.resource_sampler.sample(
                    sample_id=f"{self.run_id}-{call.call_id}-resources",
                    root_pid=self.service.pid,
                    gpu_event_id=(
                        f"{self.run_id}-{call.call_id}-repair-01-gpu"
                        if cast(Mapping[str, object], state["accepted_outputs"])[call.call_id].get(
                            "repaired"
                        )
                        else event_id
                    ),
                    job_id=job.job_id,
                )
        except RuntimeResourceLimitExceeded:
            self._mark_terminal(state, "resource-sample")
            raise
        finally:
            try:
                resource_watchdog.stop(raise_failure=False)
                # Do not mask a pending fallback exception by invoking downstream
                # work.  On a clean micro-pilot return, evaluate the complete gate
                # while the exact service is still live, bootstrap the development-
                # only model freeze, and synchronously invoke the pre-registered
                # adopter.  The adopter must return the same live service.
                if sys.exc_info()[0] is None:
                    provisional = self._result(
                        policy=policy,
                        calls=calls,
                        plan_manifest_hash=plan_manifest_hash,
                        execution_identity=execution_identity,
                        execution_hash=execution_hash,
                        state=state,
                        results=results,
                        successful_audits=successful_audits,
                        timing_observations=timing_observations,
                        repair_attempt_count=repair_attempt_count,
                        resource_watchdog=resource_watchdog,
                        uptime=None,
                        resumed_live_service=resumed_live_service,
                        limits=limits,
                    )
                    if provisional["micro_pilot_passed"] is True:
                        self._run_development_continuation(
                            provisional_result=provisional,
                            state=state,
                            execution_hash=execution_hash,
                            registration=registration,
                        )
            finally:
                # Lifecycle authority never crosses the handoff.  Whether the
                # adopter succeeds, fails, raises, or returns a bad receipt, the
                # fallback owner reconciles and stops the one model process.
                uptime = self.service.shutdown()
                self._last_shutdown_uptime = uptime

        return self._result(
            policy=policy,
            calls=calls,
            plan_manifest_hash=plan_manifest_hash,
            execution_identity=execution_identity,
            execution_hash=execution_hash,
            state=state,
            results=results,
            successful_audits=successful_audits,
            timing_observations=timing_observations,
            repair_attempt_count=repair_attempt_count,
            resource_watchdog=resource_watchdog,
            uptime=uptime,
            resumed_live_service=resumed_live_service,
            limits=limits,
        )

    def _run_repair(
        self,
        *,
        state: dict[str, object],
        call: FallbackCallSpec,
        base_call: AcceptanceCall,
        base_request: GuidedJSONRequest,
        invalid_generated: GenerationResult,
        invalid_artifact_hash: str,
        diagnostics: Sequence[Mapping[str, object]],
        job_id: str,
        parent_attempt_id: str,
        remaining_required_seconds: float,
    ) -> tuple[dict[str, object], dict[str, object] | None, float]:
        repair_id = f"{call.call_id}-repair-01"
        request = build_fallback_repair_request(
            root=self.root,
            base_call=base_call,
            base_request=base_request,
            invalid_draft=invalid_generated.parsed_object,
            diagnostics=diagnostics,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
        )
        attempt_id = f"{self.run_id}-{repair_id}-attempt"
        self.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=parent_attempt_id,
            input_hash=request.request_hash,
            config_hash=request.decoding.content_hash,
            seed=request.decoding.seed,
        )
        reserve = self._reserve(
            state,
            call_id=repair_id,
            reserve_call_class="reserve_short",
            watchdog_seconds=90,
        )
        event_id = f"{self.run_id}-{repair_id}-gpu"
        try:
            generated = self.service.generate(
                request,
                event_id=event_id,
                watchdog_seconds=90,
                repair=True,
                job_id=job_id,
                attempt_id=attempt_id,
                remaining_required_seconds=remaining_required_seconds,
                accounting_details={
                    "reserve_call_class": "reserve_short",
                    "reserve_reservation_id": reserve["reservation_id"],
                    "fallback_repair_parent_call_id": call.call_id,
                },
            )
        except Exception as exc:
            allocated = self._record_failed_transport(
                call=call,
                request=request,
                job_id=job_id,
                attempt_id=attempt_id,
                event_id=event_id,
                exc=exc,
                retry_class=RetryClass.SHORT,
                call_role=ModelCallRole.REPAIR,
            )
            return (
                {
                    "call_id": repair_id,
                    "model_call_id": f"{self.run_id}-{repair_id}",
                    "status": AcceptanceStatus.FAILED.value,
                    "exception_type": type(exc).__name__,
                    "repair_number": 1,
                },
                None,
                allocated,
            )
        event = _event_for(self.ledger, event_id)
        if event is None:
            raise RuntimeError("fallback repair completed without a GPU event")
        allocated = event.allocated_seconds
        artifact = self.artifacts.put_bytes(
            generated.raw_response,
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
        )
        common_call = {
            "model_call_id": f"{self.run_id}-{repair_id}",
            "job_id": job_id,
            "attempt_id": attempt_id,
            "gpu_event_id": event_id,
            "backend": ModelBackend.VLLM_GPU,
            "call_role": ModelCallRole.REPAIR,
            "retry_class": RetryClass.SHORT,
            "model_manifest_hash": self.service.configuration.configuration_hash,
            "decoding_manifest_hash": request.decoding.content_hash,
            "request_hash": request.request_hash,
            "response_artifact_hash": artifact.content_hash,
            "construction_unit_hash": _repair_construction_unit_hash(
                call,
                parent_attempt_id,
            ),
            "served_context_count": 1,
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "allocated_gpu_seconds": allocated,
        }
        try:
            _validate_generation_envelope(generated, request)
            audit = validate_acceptance_generation(
                root=self.root,
                call=base_call,
                parsed_object=generated.parsed_object,
                authoritative_prompt_tokens=generated.prompt_tokens,
                authoritative_completion_tokens=generated.completion_tokens,
            )
            audit["operator_behavior"] = _require_call_operator_coverage(
                call,
                audit,
                generated.parsed_object,
                root=self.root,
            )
            preservation = validate_repair_preservation(
                base_draft=invalid_generated.parsed_object,
                repaired_draft=generated.parsed_object,
                diagnosed_paths=tuple(cast(str, item["path"]) for item in diagnostics),
            )
            preservation.raise_for_errors()
        except Exception as exc:
            self.ledger.record_model_call(**common_call, successful=False)
            self.ledger.record_failure(
                attempt_id=attempt_id,
                failure_kind=FailureKind.INVALID_OUTPUT,
                message="Fallback repair failed complete mechanical validation",
                details={"exception_type": type(exc).__name__, "call_id": repair_id},
                artifact_hash=artifact.content_hash,
            )
            return (
                {
                    "call_id": repair_id,
                    "model_call_id": f"{self.run_id}-{repair_id}",
                    "status": AcceptanceStatus.FAILED.value,
                    "exception_type": type(exc).__name__,
                    "response_artifact_hash": artifact.content_hash,
                    "repair_number": 1,
                },
                None,
                allocated,
            )
        self.ledger.record_model_call(**common_call, successful=True)
        return (
            {
                "call_id": repair_id,
                "model_call_id": f"{self.run_id}-{repair_id}",
                "status": AcceptanceStatus.COMPLETED.value,
                **generated.public_manifest(),
                **_request_public_metadata(request),
                "response_artifact_hash": artifact.content_hash,
                "invalid_base_artifact_hash": invalid_artifact_hash,
                "reserve_call_class": "reserve_short",
                "mechanical_audit": audit,
                "repair_number": 1,
                "diagnostic_count": len(diagnostics),
                "scorer_only_diagnostics_exposed": False,
            },
            audit,
            allocated,
        )

    def failure_result(
        self,
        exc: BaseException,
        *,
        uptime: ServiceUptime | None,
        cleanup_failure: BaseException | None = None,
    ) -> dict[str, object]:
        """Build a truthful public result after fail-safe service shutdown."""

        plan = fallback_plan_manifest(self.root)
        plan_hash = cast(str, plan["manifest_sha256"])
        identity = self._execution_identity(plan_hash)
        execution_hash = canonical_sha256(identity)
        state: Mapping[str, object] = {}
        if self.checkpoint_path.exists():
            candidate = _load_object(self.checkpoint_path)
            if (
                candidate.get("run_id") == self.run_id
                and candidate.get("execution_hash") == execution_hash
            ):
                state = candidate
        completed = state.get("completed_call_ids", [])
        completed_ids = (
            list(cast(Sequence[str], completed))
            if isinstance(completed, list)
            and all(isinstance(item, str) for item in completed)
            else []
        )
        raw_receipts = state.get("reserve_consumption", [])
        checkpoint_receipts = (
            [dict(item) for item in cast(Sequence[Mapping[str, object]], raw_receipts)]
            if isinstance(raw_receipts, list)
            and all(isinstance(item, Mapping) for item in raw_receipts)
            else []
        )
        try:
            receipts = list(_reserve_receipts(self.ledger, checkpoint_receipts))
        except Exception:
            receipts = checkpoint_receipts
        limits = ResourceLimits.load(self.root / "configs/study/resource_limits.json")
        inventory = GPUCallInventory.load(
            self.root / "configs/study/gpu_call_inventory.json"
        )
        reference = forecast_gpu_schedule(inventory, limits=limits)
        consumed = Counter(
            cast(str, row.get("reserve_call_class")) for row in receipts
        )
        for name in NORMAL_ACCEPTANCE_CLASSES:
            consumed[name] = inventory.call_class(name).count
        lifecycle_kinds = {
            GpuEventKind.GPU_SESSION_START.value,
            GpuEventKind.RESTART.value,
        }
        consumed["gpu_session_start"] = sum(
            event.event_kind in {GpuEventKind.GPU_SESSION_START, GpuEventKind.RESTART}
            or json.loads(event.details_json).get("intended_event_kind") in lifecycle_kinds
            for event in self.ledger.gpu_events()
        )
        raw_development_receipt = state.get("development_continuation_receipt")
        try:
            development_receipt = (
                None
                if raw_development_receipt is None
                else DevelopmentContinuationReceipt.model_validate(
                    raw_development_receipt
                )
            )
        except ValidationError:
            development_receipt = None
        if (
            state.get("development_continuation_completed") is True
            and development_receipt is not None
        ):
            for name in (
                "development_c1",
                "development_c2",
                "development_fixed_select",
                "development_ablation",
                "development_repair",
            ):
                consumed[name] = inventory.call_class(name).count
        remaining = sum(
            max(0, row.count - consumed[row.call_class]) * row.forecast_p95_seconds
            for row in reference.rows
        )
        actual = max(
            self.ledger.gpu_summary().total_allocated_seconds,
            float(state.get("live_allocated_seconds_at_controller_handoff") or 0.0),
            float(state.get("live_allocated_seconds_at_development_handoff") or 0.0),
            float(getattr(self.service, "actual_allocated_service_seconds", 0.0)),
        )
        physical_shutdown_verified = (
            cleanup_failure is None and self.service.state is ServiceState.STOPPED
        )
        raw_selected_freeze = state.get("selected_model_freeze")
        selected_freeze = (
            dict(raw_selected_freeze)
            if isinstance(raw_selected_freeze, Mapping)
            and raw_selected_freeze.get("manifest_sha256")
            == canonical_sha256(
                {
                    key: value
                    for key, value in raw_selected_freeze.items()
                    if key != "manifest_sha256"
                }
            )
            else None
        )
        continuation_ready = state.get("development_continuation_ready") is True
        continuation_completed = (
            state.get("development_continuation_completed") is True
        )
        raw_accepted_micro_pilot = state.get("accepted_micro_pilot_result")
        raw_acceptance_receipt = state.get("micro_pilot_acceptance_receipt")
        micro_pilot_accepted_before_failure = (
            isinstance(raw_accepted_micro_pilot, Mapping)
            and isinstance(raw_acceptance_receipt, Mapping)
            and raw_accepted_micro_pilot.get("micro_pilot_passed") is True
            and raw_accepted_micro_pilot.get("manifest_sha256")
            == canonical_sha256(
                {
                    key: value
                    for key, value in raw_accepted_micro_pilot.items()
                    if key != "manifest_sha256"
                }
            )
            and raw_acceptance_receipt.get("accepted_result_manifest_sha256")
            == raw_accepted_micro_pilot.get("manifest_sha256")
        )
        partial_development_checkpoint: dict[str, object] | None = None
        raw_preparation = state.get("development_preparation")
        if isinstance(raw_preparation, Mapping):
            raw_checkpoint_path = raw_preparation.get("checkpoint_path")
            if isinstance(raw_checkpoint_path, str):
                checkpoint_candidate = Path(raw_checkpoint_path)
                allowed_roots = (
                    self.root,
                    self.checkpoint_path.parent.resolve(strict=True),
                )
                if (
                    checkpoint_candidate.is_absolute()
                    and not checkpoint_candidate.is_symlink()
                    and checkpoint_candidate.is_file()
                    and any(
                        checkpoint_candidate.is_relative_to(allowed_root)
                        for allowed_root in allowed_roots
                    )
                ):
                    checkpoint_hash = _file_sha256(checkpoint_candidate)
                    try:
                        parsed_checkpoint = DevelopmentCheckpoint.model_validate_json(
                            checkpoint_candidate.read_text(encoding="utf-8")
                        )
                    except (OSError, UnicodeError, ValidationError, ValueError):
                        partial_development_checkpoint = {
                            "checkpoint_sha256": checkpoint_hash,
                            "canonical_checkpoint_valid": False,
                            "resume_authorized_after_owner_shutdown": False,
                        }
                    else:
                        partial_development_checkpoint = {
                            "checkpoint_sha256": checkpoint_hash,
                            "canonical_checkpoint_valid": True,
                            "phase": parsed_checkpoint.phase.value,
                            "terminal_itt_row_count": len(parsed_checkpoint.itt_records),
                            "query_access_event_count": len(
                                parsed_checkpoint.query_access_events
                            ),
                            "active_call_id": parsed_checkpoint.active_call_id,
                            "service_identity_hash": (
                                parsed_checkpoint.service_identity_hash
                            ),
                            "resume_authorized_after_owner_shutdown": False,
                        }
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_micro_pilot_result",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "execution_identity": identity,
            "fallback_plan_manifest_sha256": plan_hash,
            "normal_acceptance_block_executed": False,
            "completed_base_call_count": len(completed_ids),
            "completed_call_ids": completed_ids,
            "repair_attempt_count": (
                1 if state.get("repair_parent_call_id") is not None else 0
            ),
            "reserve_consumption": receipts,
            "active_call_id_at_failure": state.get("active_call_id"),
            "failed_call_id": state.get("failed_call_id"),
            "failure_type": type(exc).__name__,
            "failure_message": "fallback execution failed; inspect private controller logs",
            "cleanup_failure_type": (
                None if cleanup_failure is None else type(cleanup_failure).__name__
            ),
            "micro_pilot_passed": micro_pilot_accepted_before_failure,
            "failure_stage": (
                "development_continuation"
                if micro_pilot_accepted_before_failure
                else "fallback_micro_pilot"
            ),
            "phase1_gate_passed": False,
            "gate_passed": False,
            "actual_plus_remaining_forecast": {
                "actual_allocated_seconds": actual,
                "remaining_forecast_seconds": remaining,
                "actual_plus_remaining_seconds": actual + remaining,
                "scheduled_limit_seconds": limits.scheduled_gpu_seconds,
                "admitted": actual + remaining <= limits.scheduled_gpu_seconds,
                "normal_acceptance_rows_superseded_not_executed": True,
            },
            "runtime": public_runtime_manifest(
                launcher=self.service.configuration,
                tokenizer=self.tokenizer_manifest,
                runtime_stack=self.runtime_stack,
                gpu_hardware=self.gpu_hardware,
                resource_samples=self.resource_sampler.samples,
                uptime=uptime,
                ledger=self.ledger,
            ),
            "vllm_service_stopped": physical_shutdown_verified,
            "physical_service_live": False if physical_shutdown_verified else None,
            "physical_service_state_unverified": not physical_shutdown_verified,
            "development_handoff_pending": (
                continuation_ready
                and not continuation_completed
                and not physical_shutdown_verified
            ),
            "development_continuation_interrupted": (
                continuation_ready and not continuation_completed
            ),
            "partial_development_checkpoint": partial_development_checkpoint,
            "development_continuation_integration_pending": not continuation_ready,
            "micro_pilot_acceptance_receipt": state.get(
                "micro_pilot_acceptance_receipt"
            ),
            "accepted_micro_pilot_result": state.get("accepted_micro_pilot_result"),
            "development_continuation_bootstrap": state.get(
                "development_continuation_bootstrap"
            ),
            "development_preparation": state.get("development_preparation"),
            "development_handoff": state.get("development_handoff"),
            "development_execution_result": state.get(
                "development_execution_result"
            ),
            "development_continuation_receipt": (
                None
                if development_receipt is None
                else development_receipt.model_dump(mode="json")
            ),
            "selected_model_freeze": selected_freeze,
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}

    def _result(
        self,
        *,
        policy: FallbackModelPolicy,
        calls: Sequence[FallbackCallSpec],
        plan_manifest_hash: str,
        execution_identity: Mapping[str, object],
        execution_hash: str,
        state: Mapping[str, object],
        results: Sequence[Mapping[str, object]],
        successful_audits: Sequence[tuple[FallbackCallSpec, Mapping[str, object]]],
        timing_observations: list[TimingObservation],
        repair_attempt_count: int,
        resource_watchdog: ResourceWatchdog,
        uptime: ServiceUptime | None,
        resumed_live_service: bool,
        limits: ResourceLimits,
    ) -> dict[str, object]:
        lifecycle_events = tuple(
            event
            for event in self.ledger.gpu_events_with_prefix(f"{self.run_id}-")
            if event.event_kind in {GpuEventKind.GPU_SESSION_START, GpuEventKind.RESTART}
            or json.loads(event.details_json).get("intended_event_kind")
            in {GpuEventKind.GPU_SESSION_START.value, GpuEventKind.RESTART.value}
        )
        service_starts = tuple(
            event
            for event in lifecycle_events
            if event.event_kind is GpuEventKind.GPU_SESSION_START
            or json.loads(event.details_json).get("intended_event_kind")
            == GpuEventKind.GPU_SESSION_START.value
        )
        restarts = tuple(
            event
            for event in lifecycle_events
            if event.event_kind is GpuEventKind.RESTART
            or json.loads(event.details_json).get("intended_event_kind")
            == GpuEventKind.RESTART.value
        )
        overhead = float(getattr(uptime, "unclassified_service_seconds", 0.0))
        complete_timing_observations = list(timing_observations)
        if service_starts:
            complete_timing_observations.append(
                TimingObservation(
                    call_class="gpu_session_start",
                    allocated_seconds=service_starts[0].allocated_seconds + overhead,
                )
            )
        timing = summarize_call_class_timings(complete_timing_observations)
        timing_by_name = {item.call_class: item for item in timing}
        expected_timing_counts = {
            "acceptance_c1": 1,
            "acceptance_c2": 2,
            "acceptance_fixed_select": 1,
        }
        timing_gate = {
            "expected_base_sample_counts": expected_timing_counts,
            "observed_call_classes": sorted(timing_by_name),
            "base_sample_counts_exact": all(
                name in timing_by_name and timing_by_name[name].sample_count == count
                for name, count in expected_timing_counts.items()
            ),
            "repair_sample_count": (
                0
                if "acceptance_repair" not in timing_by_name
                else timing_by_name["acceptance_repair"].sample_count
            ),
            "repair_sample_count_valid": (
                repair_attempt_count <= 1
                and (
                    0
                    if "acceptance_repair" not in timing_by_name
                    else timing_by_name["acceptance_repair"].sample_count
                )
                == repair_attempt_count
            ),
            "service_start_sample_count": len(service_starts),
            "service_start_sample_count_exact": len(service_starts) == 1,
            "physical_restart_event_count": len(restarts),
            "physical_restart_forbidden_by_one_load_policy": len(restarts) == 0,
            "service_overhead_seconds_included": overhead,
        }

        inventory = GPUCallInventory.load(self.root / "configs/study/gpu_call_inventory.json")
        forecast = forecast_gpu_schedule(
            inventory,
            complete_timing_observations,
            limits=limits,
        )
        receipts = _reserve_receipts(
            self.ledger,
            cast(Sequence[Mapping[str, object]], state["reserve_consumption"]),
        )
        consumed = Counter(cast(str, row["reserve_call_class"]) for row in receipts)
        # The fallback is a protocol-defined substitute, not a second pilot.
        # Supersede every unused normal acceptance row without representing it
        # as an executed call or successful model output.
        for name in NORMAL_ACCEPTANCE_CLASSES:
            consumed[name] = inventory.call_class(name).count
        lifecycle_kinds = {
            GpuEventKind.GPU_SESSION_START.value,
            GpuEventKind.RESTART.value,
        }
        consumed["gpu_session_start"] = sum(
            event.event_kind in {GpuEventKind.GPU_SESSION_START, GpuEventKind.RESTART}
            or json.loads(event.details_json).get("intended_event_kind") in lifecycle_kinds
            for event in self.ledger.gpu_events()
        )
        raw_development_receipt = state.get("development_continuation_receipt")
        development_receipt = (
            None
            if raw_development_receipt is None
            else DevelopmentContinuationReceipt.model_validate(raw_development_receipt)
        )
        if state.get("development_continuation_completed") is True:
            if development_receipt is None:
                raise RuntimeError(
                    "completed development continuation lacks its immutable receipt"
                )
            for name in (
                "development_c1",
                "development_c2",
                "development_fixed_select",
                "development_ablation",
                "development_repair",
            ):
                consumed[name] = inventory.call_class(name).count
        post_fallback_rows: list[dict[str, object]] = []
        remaining_forecast = 0.0
        for row in forecast.rows:
            superseded = row.call_class in NORMAL_ACCEPTANCE_CLASSES
            consumed_count = row.count if superseded else min(
                row.count,
                consumed[row.call_class],
            )
            remaining_count = max(0, row.count - consumed_count)
            remaining_seconds = remaining_count * row.forecast_p95_seconds
            remaining_forecast += remaining_seconds
            post_fallback_rows.append(
                {
                    **asdict(row),
                    "superseded_without_execution": superseded,
                    "consumed_count": consumed_count,
                    "remaining_count": remaining_count,
                    "remaining_forecast_seconds": remaining_seconds,
                }
            )
        actual = max(
            self.ledger.gpu_summary().total_allocated_seconds,
            float(
                state.get("live_allocated_seconds_at_development_handoff")
                or getattr(
                    self.service,
                    "actual_allocated_service_seconds",
                    0.0,
                )
            ),
        )
        continuation = {
            "actual_allocated_seconds": actual,
            "remaining_forecast_seconds": remaining_forecast,
            "actual_plus_remaining_seconds": actual + remaining_forecast,
            "scheduled_limit_seconds": limits.scheduled_gpu_seconds,
            "normal_acceptance_rows_superseded_not_executed": {
                name: inventory.call_class(name).count for name in NORMAL_ACCEPTANCE_CLASSES
            },
            "consumed_gpu_session_start_slots": consumed["gpu_session_start"],
            "consumed_reserve_slots": {
                name: consumed[name]
                for name in ("reserve_long", "reserve_standard", "reserve_short")
            },
            "admitted": actual + remaining_forecast <= limits.scheduled_gpu_seconds,
        }

        required_operators = {operator.value for operator in CONSTRUCTIVE_OPERATORS}
        c1_pilot_required = {
            operator
            for call in calls
            if call.condition is ConditionName.C1_LLM_PRE
            for operator in call.required_constructive_operators
        }
        c2_pilot_required = {
            operator
            for call in calls
            if call.condition is ConditionName.C2_LLM_QUERY
            for operator in call.required_constructive_operators
        }
        c1_operators = {
            operator
            for call, audit in successful_audits
            if call.condition is ConditionName.C1_LLM_PRE
            for operator in cast(Sequence[str], audit["constructive_operators"])
        }
        c2_operators = {
            operator
            for call, audit in successful_audits
            if call.condition is ConditionName.C2_LLM_QUERY
            for operator in cast(Sequence[str], audit["constructive_operators"])
        }
        operator_gate = {
            "global_constructive_operator_inventory": sorted(required_operators),
            "c1_pilot_required": sorted(c1_pilot_required),
            "c2_pilot_required": sorted(c2_pilot_required),
            "c1_observed": sorted(c1_operators),
            "c2_observed": sorted(c2_operators),
            "c1_pilot_complete": c1_pilot_required.issubset(c1_operators),
            "c1_global_complete_in_micro_pilot": required_operators.issubset(c1_operators),
            "c1_global_completion_gate": (
                "integrated_development_scientific_assessment."
                "c1_all_construction_operators_exercised"
            ),
            "c2_complete": c2_pilot_required.issubset(c2_operators),
            "c1_behaviorally_valid": all(
                cast(Mapping[str, object], audit.get("operator_behavior", {})).get(
                    "behaviorally_valid"
                )
                is True
                for call, audit in successful_audits
                if call.condition is ConditionName.C1_LLM_PRE
            ),
            "c2_behaviorally_valid": all(
                cast(Mapping[str, object], audit.get("operator_behavior", {})).get(
                    "behaviorally_valid"
                )
                is True
                for call, audit in successful_audits
                if call.condition is ConditionName.C2_LLM_QUERY
            ),
        }
        grounding_horizon_gate = all(
            audit["grounding_complete"] is True
            and audit["evidence_ids_valid"] is True
            and audit["horizon_leak_count"] == 0
            for _, audit in successful_audits
        )
        resource_gate = _resource_gate(self.ledger, limits)
        controller_resume_gate = {
            "checkpoint_handoff_completed": state["controller_handoff_complete"] is True,
            "live_service_adopted": resumed_live_service,
            "second_model_load_performed": False,
            "controller_process_restart": (
                state["controller_process_restart_observed"] is True
            ),
            "model_process_restart": False,
            "service_pid_unchanged": state.get("handoff_service_pid") is not None,
            "stage_controller_pids_distinct": (
                state.get("stage_one_controller_pid")
                != state.get("stage_two_controller_pid")
            ),
            "passed": (
                state["controller_handoff_complete"] is True
                and resumed_live_service
                and state["controller_process_restart_observed"] is True
                and state.get("stage_one_controller_pid")
                != state.get("stage_two_controller_pid")
            ),
        }
        physical_restart_gate = {
            "required_restart_kind": "controller_process_restart_with_live_model_adoption",
            "controller_process_restart": controller_resume_gate[
                "controller_process_restart"
            ],
            "model_process_restart": False,
            "second_model_load_performed": False,
            "passed": controller_resume_gate["passed"],
        }
        completed = cast(Sequence[str], state["completed_call_ids"])
        micro_pilot_passed = (
            len(completed) == len(calls)
            and state["failed_call_id"] is None
            and resource_watchdog.failure is None
            and resource_gate["accepted"] is True
            and continuation["admitted"] is True
            and operator_gate["c1_pilot_complete"] is True
            and operator_gate["c2_complete"] is True
            and operator_gate["c1_behaviorally_valid"] is True
            and operator_gate["c2_behaviorally_valid"] is True
            and grounding_horizon_gate
            and timing_gate["base_sample_counts_exact"] is True
            and timing_gate["repair_sample_count_valid"] is True
            and timing_gate["service_start_sample_count_exact"] is True
            and timing_gate["physical_restart_forbidden_by_one_load_policy"] is True
            and controller_resume_gate["passed"] is True
        )
        registration = self._development_registration(required=False)
        ready = state.get("development_continuation_ready") is True
        continuation_completed = state.get("development_continuation_completed") is True
        same_service_pid = (
            state.get("development_handoff_service_pid")
            == state.get("handoff_service_pid")
        )
        service_is_live = self.service.state is ServiceState.READY
        owner_shutdown_verified = (
            self.service.state is ServiceState.STOPPED and uptime is not None
        )
        receipt_matches_handoff = (
            development_receipt is not None
            and development_receipt.handoff_hash == state.get("development_handoff_hash")
            and development_receipt.fallback_execution_hash == execution_hash
            and development_receipt.service_pid == state.get("handoff_service_pid")
            and development_receipt.adopter_registration_hash
            == state.get("development_adopter_registration_hash")
        )
        development_continuation_gate = {
            "ready": ready,
            "completed": continuation_completed,
            "same_model_service_pid": (
                same_service_pid
            ),
            "configuration_sha256": self.service.configuration.configuration_hash,
            "source_execution_hash": execution_hash,
            "next_workload": "registered_24_call_development_block",
            "requires_new_model_load": False,
            "integrated_adopter_available": registration is not None,
            "adopter_registration_hash": (
                None if registration is None else registration.content_hash
            ),
            "receipt_matches_handoff": receipt_matches_handoff,
            "terminal_intention_to_treat_call_count": (
                None
                if development_receipt is None
                else development_receipt.terminal_call_count
            ),
            "service_returned_live_to_owner": (
                None
                if development_receipt is None
                else development_receipt.service_returned_live_to_owner
            ),
            "development_gate_passed": (
                None
                if development_receipt is None
                else development_receipt.development_gate_passed
            ),
            "development_forecast_admitted": (
                None
                if development_receipt is None
                else development_receipt.development_forecast_admitted
            ),
            "scientific_thresholds_passed": (
                None
                if development_receipt is None
                else development_receipt.scientific_thresholds_passed
            ),
            "owner_final_shutdown_verified": owner_shutdown_verified,
            "standalone_runner_fail_closed": True,
            "standalone_runner_shutdown_verified": owner_shutdown_verified,
            "integration_hook_state_fields": [
                "development_continuation_ready",
                "development_handoff_controller_pid",
                "development_handoff_service_pid",
                "live_allocated_seconds_at_development_handoff",
            ],
            "passed": (
                ready
                and continuation_completed
                and same_service_pid
                and registration is not None
                and receipt_matches_handoff
                and development_receipt is not None
                and development_receipt.development_gate_passed
                and development_receipt.development_forecast_admitted
                and development_receipt.scientific_thresholds_passed
                and owner_shutdown_verified
            ),
        }
        phase1_gate_passed = (
            micro_pilot_passed
            and physical_restart_gate["passed"] is True
            and development_continuation_gate["passed"] is True
        )
        selected_freeze = state.get("selected_model_freeze")
        if selected_freeze is not None:
            if not isinstance(selected_freeze, Mapping):
                raise RuntimeError("checkpoint selected-model freeze is not an object")
            immutable_freeze = {
                key: value
                for key, value in selected_freeze.items()
                if key != "manifest_sha256"
            }
            if selected_freeze.get("manifest_sha256") != canonical_sha256(
                immutable_freeze
            ):
                raise RuntimeError("checkpoint selected-model freeze hash changed")
        if phase1_gate_passed and selected_freeze is None:
            raise RuntimeError("passed fallback handoff lacks a selected-model freeze")
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_micro_pilot_result",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "fallback_plan_manifest_sha256": plan_manifest_hash,
            "execution_identity": dict(execution_identity),
            "normal_acceptance_block_executed": False,
            "base_call_count": len(calls),
            "completed_base_call_count": len(completed),
            "repair_attempt_count": repair_attempt_count,
            "micro_pilot_passed": micro_pilot_passed,
            "phase1_gate_passed": phase1_gate_passed,
            "gate_passed": phase1_gate_passed,
            "calls": list(results),
            "reserve_consumption": list(receipts),
            "runtime": public_runtime_manifest(
                launcher=self.service.configuration,
                tokenizer=self.tokenizer_manifest,
                runtime_stack=self.runtime_stack,
                gpu_hardware=self.gpu_hardware,
                resource_samples=self.resource_sampler.samples,
                uptime=uptime,
                ledger=self.ledger,
            ),
            "resumed_live_service": resumed_live_service,
            "vllm_service_stopped": owner_shutdown_verified,
            "model_service_live_for_development_continuation": (
                service_is_live and ready and not continuation_completed
            ),
            "development_handoff_pending": (
                ready and not continuation_completed
            ),
            "development_continuation_integration_pending": (
                micro_pilot_passed
                and not ready
            ),
            "physical_service_live": service_is_live,
            "resource_watchdog": {
                "sample_count": resource_watchdog.sample_count,
                "failure_type": (
                    None
                    if resource_watchdog.failure is None
                    else type(resource_watchdog.failure).__name__
                ),
            },
            "cumulative_resource_gate": resource_gate,
            "controller_resume_gate": controller_resume_gate,
            "physical_restart_gate": physical_restart_gate,
            "development_continuation_gate": development_continuation_gate,
            "timing_by_call_class": [asdict(item) for item in timing],
            "timing_gate": timing_gate,
            "post_fallback_full_manifest_forecast": {
                "rows": post_fallback_rows,
                "actual_allocated_seconds": actual,
                "remaining_forecast_seconds": remaining_forecast,
                "total_seconds": actual + remaining_forecast,
                "total_hours": (actual + remaining_forecast) / 3600,
                "scheduled_limit_seconds": forecast.scheduled_limit_seconds,
                "preferred_limit_seconds": forecast.preferred_limit_seconds,
                "hard_limit_seconds": forecast.hard_limit_seconds,
                "admitted": continuation["admitted"],
                "normal_acceptance_rows_counted": False,
                "reference_inventory_total_seconds_before_supersession": (
                    forecast.total_seconds
                ),
            },
            "actual_plus_remaining_forecast": continuation,
            "operator_coverage_gate": operator_gate,
            "grounding_horizon_gate": grounding_horizon_gate,
            "micro_pilot_acceptance_receipt": state.get(
                "micro_pilot_acceptance_receipt"
            ),
            "accepted_micro_pilot_result": state.get("accepted_micro_pilot_result"),
            "development_continuation_bootstrap": state.get(
                "development_continuation_bootstrap"
            ),
            "development_preparation": state.get("development_preparation"),
            "development_handoff": state.get("development_handoff"),
            "selected_model_freeze": selected_freeze,
            "development_execution_result": state.get(
                "development_execution_result"
            ),
            "development_continuation_receipt": (
                None
                if development_receipt is None
                else development_receipt.model_dump(mode="json")
            ),
            "runtime_feasibility_tradeoff": {
                "enforce_eager": True,
                "compile_latency_avoided": True,
                "possible_throughput_reduction": True,
                "measured_forecast_uses_same_setting": True,
            },
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _controller_execution_arguments(options: argparse.Namespace) -> dict[str, object]:
    """Return the exact private argument identity shared by both controllers."""

    return {
        "project_root": str(options.project_root.resolve()),
        "run_id": options.run_id,
        "primary_result": str(options.primary_result.resolve()),
        "activation_certificate": str(options.activation_certificate.resolve()),
        "cache_replacement_receipt": str(
            options.cache_replacement_receipt.resolve()
        ),
        "snapshot": str(options.snapshot.resolve()),
        "shared_cache": str(options.shared_cache.resolve()),
        "verified_model_manifest": str(options.verified_model_manifest.resolve()),
        "source_association": str(options.source_association.resolve()),
        "ledger": str(options.ledger.resolve()),
        "artifact_root": str(options.artifact_root.resolve()),
        "checkpoint": str(options.checkpoint.resolve()),
        "quota_root": str(options.quota_root.resolve()),
        "port": options.port,
    }


def _expected_orchestrator_guard(options: argparse.Namespace) -> Path:
    return options.checkpoint.resolve().with_name(
        options.checkpoint.name + ".orchestrator-guard.json"
    )


def _validate_orchestrator_guard(options: argparse.Namespace) -> Mapping[str, object]:
    supplied = options.orchestrator_guard
    expected = _expected_orchestrator_guard(options)
    if supplied is None or supplied.resolve() != expected:
        raise RuntimeError(
            "internal controller stages require the exact active orchestrator guard"
        )
    guard = _load_object(expected)
    immutable = {key: value for key, value in guard.items() if key != "manifest_sha256"}
    if guard.get("manifest_sha256") != canonical_sha256(immutable):
        raise ValueError("orchestrator guard hash does not match its contents")
    orchestrator_pid = guard.get("orchestrator_pid")
    if (
        guard.get("kind") != "fallback_controller_orchestrator_guard"
        or guard.get("state") != "active"
        or guard.get("run_id") != options.run_id
        or guard.get("execution_arguments_sha256")
        != canonical_sha256(_controller_execution_arguments(options))
        or isinstance(orchestrator_pid, bool)
        or not isinstance(orchestrator_pid, int)
        or orchestrator_pid <= 0
        or orchestrator_pid == os.getpid()
    ):
        raise ValueError("orchestrator guard does not authorize this controller stage")
    try:
        os.kill(orchestrator_pid, 0)
    except OSError as exc:
        raise RuntimeError("controller orchestrator is no longer alive") from exc
    return guard


def _internal_controller_command(
    options: argparse.Namespace,
    *,
    stage: str,
    output: Path,
    guard: Path,
) -> tuple[str, ...]:
    arguments = _controller_execution_arguments(options)
    command = [
        sys.executable,
        str(
            Path(cast(str, arguments["project_root"]))
            / "scripts/run_fallback_gpu_acceptance.py"
        ),
        "--execute",
        "--controller-stage",
        stage,
        "--project-root",
        cast(str, arguments["project_root"]),
        "--output",
        str(output.resolve()),
        "--orchestrator-guard",
        str(guard),
    ]
    for name in (
        "run_id",
        "primary_result",
        "activation_certificate",
        "cache_replacement_receipt",
        "snapshot",
        "shared_cache",
        "verified_model_manifest",
        "source_association",
        "ledger",
        "artifact_root",
        "checkpoint",
        "quota_root",
    ):
        command.extend((f"--{name.replace('_', '-')}", cast(str, arguments[name])))
    command.extend(("--port", str(arguments["port"])))
    return tuple(command)


def _orchestrate_controller_processes(options: argparse.Namespace) -> int:
    """Supervise both controller invocations and guarantee normal-path cleanup."""

    guard_path = _expected_orchestrator_guard(options)
    handoff_output = options.output.resolve().with_name(
        options.output.name + ".controller-handoff.json"
    )
    cleanup_output = options.output.resolve().with_name(
        options.output.name + ".orphan-cleanup.json"
    )
    identity = _controller_execution_arguments(options)
    guard_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_controller_orchestrator_guard",
        "state": "active",
        "run_id": options.run_id,
        "orchestrator_pid": os.getpid(),
        "execution_arguments_sha256": canonical_sha256(identity),
    }
    _private_atomic_json(
        guard_path,
        {**guard_payload, "manifest_sha256": canonical_sha256(guard_payload)},
    )
    prepare_return_code: int | None = None
    run_return_code: int | None = None
    cleanup_return_code: int | None = None
    physical_shutdown_verified = False
    raised: BaseException | None = None
    try:
        prepared = subprocess.run(
            _internal_controller_command(
                options,
                stage="prepare",
                output=handoff_output,
                guard=guard_path,
            ),
            check=False,
        )
        prepare_return_code = prepared.returncode
        if prepare_return_code != 0:
            raise RuntimeError("fallback prepare controller failed")
        handoff = _load_object(handoff_output)
        if (
            handoff.get("run_id") != options.run_id
            or handoff.get("model_service_left_live_for_controller_restart") is not True
            or handoff.get("physical_service_live") is not True
            or handoff.get("vllm_service_stopped") is not False
        ):
            raise RuntimeError("fallback prepare controller emitted an invalid handoff")
        executed = subprocess.run(
            _internal_controller_command(
                options,
                stage="run",
                output=options.output,
                guard=guard_path,
            ),
            check=False,
        )
        run_return_code = executed.returncode
        if options.output.is_file():
            result = _load_object(options.output)
            physical_shutdown_verified = (
                result.get("vllm_service_stopped") is True
                and result.get("physical_service_live") is False
            )
        if not physical_shutdown_verified:
            raise RuntimeError(
                "fallback run controller did not publish a verified service shutdown"
            )
        return run_return_code
    except BaseException as exc:
        raised = exc
        raise
    finally:
        service_checkpoint = options.checkpoint.with_name(
            options.checkpoint.name + ".service"
        )
        if not physical_shutdown_verified and service_checkpoint.exists():
            cleanup = subprocess.run(
                _internal_controller_command(
                    options,
                    stage="cleanup",
                    output=cleanup_output,
                    guard=guard_path,
                ),
                check=False,
            )
            cleanup_return_code = cleanup.returncode
            if cleanup_output.is_file():
                cleanup_result = _load_object(cleanup_output)
                physical_shutdown_verified = (
                    cleanup.returncode == 0
                    and cleanup_result.get("physical_shutdown_verified") is True
                )
            if not physical_shutdown_verified and raised is None:
                raise RuntimeError("fallback orchestrator could not verify orphan cleanup")
        closed_payload = {
            **guard_payload,
            "state": "closed",
            "prepare_return_code": prepare_return_code,
            "run_return_code": run_return_code,
            "cleanup_return_code": cleanup_return_code,
            "physical_shutdown_verified": physical_shutdown_verified,
        }
        _private_atomic_json(
            guard_path,
            {
                **closed_payload,
                "manifest_sha256": canonical_sha256(closed_payload),
            },
        )


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--controller-stage",
        choices=("orchestrate", "prepare", "run", "cleanup"),
    )
    parser.add_argument("--orchestrator-guard", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-id")
    parser.add_argument("--primary-result", type=Path)
    parser.add_argument("--activation-certificate", type=Path)
    parser.add_argument("--cache-replacement-receipt", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--shared-cache", type=Path)
    parser.add_argument("--verified-model-manifest", type=Path)
    parser.add_argument("--source-association", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quota-root", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(arguments)


def _require_execution_arguments(options: argparse.Namespace) -> None:
    required = (
        "run_id",
        "controller_stage",
        "primary_result",
        "activation_certificate",
        "cache_replacement_receipt",
        "snapshot",
        "shared_cache",
        "verified_model_manifest",
        "source_association",
        "ledger",
        "artifact_root",
        "checkpoint",
        "quota_root",
    )
    missing = [name for name in required if getattr(options, name) is None]
    if missing:
        raise SystemExit(
            "--execute requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )


def main(
    arguments: Sequence[str] | None = None,
    *,
    development_adopter: DevelopmentContinuationAdopter | None = None,
) -> int:
    options = parse_arguments(arguments)
    root = options.project_root.resolve(strict=True)
    if not options.execute:
        atomic_write_public_json(options.output, fallback_plan_manifest(root))
        return 0
    _require_execution_arguments(options)
    if options.controller_stage == "orchestrate":
        # Import the registered production factory before the orchestrator may
        # request the third model load.  Internal controller processes perform
        # the actual construction against their adopted service handle.
        from story_projection_onto.development_continuation import (
            create_production_development_adopter,
        )

        del create_production_development_adopter
        return _orchestrate_controller_processes(options)
    _validate_orchestrator_guard(options)
    policy_path = root / "configs/study/fallback_model.json"
    policy = FallbackModelPolicy.load(policy_path)
    primary_result = _load_object(options.primary_result)
    pre_fallback_accounting = pre_fallback_gpu_accounting_baseline(primary_result)
    activation = validate_fallback_activation_certificate(
        policy=policy,
        certificate=_load_object(options.activation_certificate),
        primary_result=primary_result,
    )
    replacement = validate_fallback_cache_replacement_receipt(
        policy=policy,
        activation_certificate=activation,
        receipt=_load_object(options.cache_replacement_receipt),
        shared_cache=options.shared_cache,
    )
    snapshot_manifest = validate_fallback_snapshot_manifest(
        options.verified_model_manifest,
        policy=policy,
        policy_path=policy_path,
    )
    source_association = validate_source_association(
        options.source_association,
        source_root=root,
    )
    configuration = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=options.snapshot,
        shared_cache=options.shared_cache,
        model_configuration_path=root / "configs/study/model.json",
        model_candidate="fallback",
        verified_snapshot_manifest_sha256=cast(str, snapshot_manifest["manifest_sha256"]),
        port=options.port,
    )
    limits = ResourceLimits.load(root / "configs/study/resource_limits.json")
    storage_plan = StorageAllocationPlan.load(root / "configs/study/storage_phase_allocations.json")
    phase_one = storage_plan.reservation_for("phase_1")
    storage = StoragePreflight(
        options.quota_root,
        controlled_paths=(
            root,
            options.shared_cache,
            options.ledger.parent,
            options.artifact_root,
            options.checkpoint.parent,
            options.output.parent,
        ),
        budget=StorageBudget(
            total_allocation_bytes=limits.maximum_project_allocation_bytes,
            max_occupied_bytes=limits.maximum_project_occupied_bytes,
            min_headroom_bytes=limits.minimum_storage_headroom_bytes,
        ),
    )
    preflight = storage.check(**phase_one.preflight_arguments())
    result: dict[str, object]
    with Ledger(options.ledger) as ledger:
        ledger.record_storage_sample(preflight, phase=f"phase1_fallback:{options.run_id}")
        if not preflight.allowed:
            raise StorageBudgetExceeded(preflight)
        if options.controller_stage == "prepare":
            observed_baseline = validate_pre_fallback_gpu_accounting(
                primary_result,
                ledger.gpu_summary(),
            )
            if observed_baseline != pre_fallback_accounting:
                raise RuntimeError("pre-fallback GPU accounting baseline changed")
        snapshot_manifest = validate_fallback_snapshot_manifest(
            options.verified_model_manifest,
            policy=policy,
            policy_path=policy_path,
            snapshot_path=options.snapshot,
            shared_cache=options.shared_cache,
        )
        tokenizer_manifest = capture_tokenizer_manifest(
            options.snapshot,
            repository=FALLBACK_MODEL_REPOSITORY,
            revision=FALLBACK_MODEL_REVISION,
        )
        runtime_stack = capture_runtime_stack()
        gpu_hardware = capture_gpu_hardware_identity(
            root / "artifacts/public/manifests/environment.json"
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(options.snapshot),
            local_files_only=True,
            trust_remote_code=False,
            revision=FALLBACK_MODEL_REVISION,
        )
        meter = AllocatedGPUMeter.from_limits(ledger, limits)
        client = VLLMGuidedJSONClient(configuration.base_url)
        sampler = ResourceSampler(limits=limits, storage=storage, ledger=ledger)
        service = VLLMService(
            configuration=configuration,
            client=client,
            meter=meter,
            log_path=options.checkpoint.parent / f"{options.run_id}.vllm.log",
            startup_resource_sampler=sampler,
            preflight_endpoint_check=lambda: client.health(0.25),
            readiness_check=lambda: client.ready(
                2.0, model_name=FALLBACK_SERVED_MODEL_NAME
            ),
        )
        artifact_store = ArtifactStore(BlobStore(options.artifact_root), ledger)
        if development_adopter is None:
            from story_projection_onto.development_assessment_bridge import (
                build_post_run_development_assessment_provider,
            )
            from story_projection_onto.development_continuation import (
                create_production_development_adopter,
            )

            development_adopter = create_production_development_adopter(
                root=root,
                service=service,
                artifacts=artifact_store,
                tokenizer=cast(PackingTokenizer, tokenizer),
                tokenizer_manifest=tokenizer_manifest,
                launcher_configuration_hash=configuration.configuration_hash,
                model_snapshot_manifest_hash=cast(
                    str, snapshot_manifest["manifest_sha256"]
                ),
                source_association=source_association,
                checkpoint_path=options.checkpoint,
                assessment_factory=build_post_run_development_assessment_provider,
            )
        runner = FallbackAcceptanceRunner(
            root=root,
            run_id=options.run_id,
            service=service,
            ledger=ledger,
            artifacts=artifact_store,
            resource_sampler=sampler,
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            checkpoint_path=options.checkpoint,
            activation_certificate=activation,
            replacement_receipt=replacement,
            snapshot_manifest=snapshot_manifest,
            source_association=source_association,
            pre_fallback_gpu_accounting=pre_fallback_accounting,
            development_adopter=development_adopter,
            runtime_stack=runtime_stack,
            gpu_hardware=gpu_hardware,
        )
        try:
            if options.controller_stage == "prepare":
                result = runner.prepare_controller_restart()
            elif options.controller_stage == "cleanup":
                result = runner.cleanup_orphan()
            else:
                result = runner.run()
        except BaseException as exc:
            uptime = None
            cleanup_failure = None
            try:
                uptime = runner.ensure_service_stopped_after_failure()
            except BaseException as cleanup_exc:
                cleanup_failure = cleanup_exc
            result = runner.failure_result(
                exc,
                uptime=uptime,
                cleanup_failure=cleanup_failure,
            )
        try:
            atomic_write_public_json(options.output, result)
        except BaseException:
            # A successful prepare deliberately leaves the model process live.
            # If its public handoff receipt cannot be persisted, immediately
            # adopt the exact private checkpoint and verify physical shutdown.
            if (
                options.controller_stage == "prepare"
                and result.get("model_service_left_live_for_controller_restart") is True
            ):
                runner.cleanup_orphan()
            raise
    if options.controller_stage == "prepare":
        return 0 if result.get("next_required_stage") is not None else 2
    if options.controller_stage == "cleanup":
        return 0 if result.get("physical_shutdown_verified") is True else 2
    return 0 if result.get("phase1_gate_passed") is True else 2


__all__ = [
    "FALLBACK_IMPLEMENTATION_FILES",
    "FORBIDDEN_ADAPTER_LIFECYCLE_MEMBERS",
    "NORMAL_ACCEPTANCE_CLASSES",
    "REPAIR_TRIGGER_RULE",
    "DevelopmentAdopterRegistration",
    "DevelopmentContinuationAdopter",
    "DevelopmentContinuationBootstrap",
    "DevelopmentContinuationHandoff",
    "DevelopmentContinuationReceipt",
    "FallbackAcceptanceRunner",
    "FallbackCallSpec",
    "PreparedDevelopmentContinuation",
    "build_fallback_repair_request",
    "fallback_pilot_calls",
    "fallback_plan_manifest",
    "main",
    "parse_arguments",
    "pre_fallback_gpu_accounting_baseline",
    "validate_pre_fallback_gpu_accounting",
    "validate_source_association",
]
