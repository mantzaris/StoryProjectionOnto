"""Certified, reserve-charged Phase-1 fallback micro-pilot.

Importing this module is CPU-only.  Its CLI defaults to a static public plan and
requires ``--execute`` plus activation, cache-replacement, snapshot, ledger, and
resource evidence before it may start the single pinned fallback service.  One
additional start is possible only through a hash-bound, predecessor-specific
methodological-amendment certificate.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, Self, cast, runtime_checkable

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    RUNTIME_STRUCTURAL_ONLY_DIAGNOSTIC,
    CommitmentCheckStatus,
    ConditionName,
    EvidenceSupportStatus,
    ImmutableRecord,
    RunOutcome,
    Sha256Digest,
    TemporalDeterminationStatus,
    ValidationStatus,
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
    DEFAULT_SHUTDOWN_SECONDS,
    DURABLE_EXEC_GATE_PROTOCOL,
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    PINNED_RUNTIME_VERSION,
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
    restricted_transport_failure_details,
)
from story_projection_onto.llm import (
    VLLM_XGRAMMAR_IGNORED_STRING_KEYWORDS,
    CapabilityManifest,
    DecodingManifest,
    FixedSelectCapabilityError,
    PackingReport,
    PackingSection,
    base_condition_output_schema,
)
from story_projection_onto.manifest import build_source_manifest, load_source_manifest
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
from story_projection_onto.phase1_legacy_provenance import (
    BRIDGE_REVISION,
    CERTIFICATE_RELATIVE_PATH,
    Phase1LegacyEvidenceProvenanceBridge,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    FailureKind,
    GpuEventKind,
    GpuServiceJournalState,
    GpuServiceSession,
    GpuSummary,
    JobState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    ReadOnlyLedger,
    ReleaseClass,
    RetryClass,
    SemanticAssessmentScope,
    StorageBudget,
    StorageBudgetExceeded,
    StoragePreflight,
    StorageReport,
    TemporalValidationStatus,
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

FALLBACK_GUARDIAN_POLL_SECONDS = 1.0
FALLBACK_GUARDIAN_READY_TIMEOUT_SECONDS = 30.0
FALLBACK_ORCHESTRATOR_RESUME_GRACE_SECONDS = 300.0
FALLBACK_CONTROL_GROUP_TERM_SECONDS = 2.0
FALLBACK_CONTROL_GROUP_KILL_SECONDS = 2.0
DEFAULT_FALLBACK_STARTUP_WATCHDOG_SECONDS = 180
AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS = 300
SECOND_RECOVERY_V3_RUN_ID = "fallback-qwen3-8b-awq-development-v3"
SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256 = (
    "6a76fae9bb970fdf11a9ae37cf785910cf6c270146ca7b752a6aedb678de3efd"
)
SECOND_RECOVERY_V3_RESULT_FILE_SHA256 = (
    "9279cdcb048be6bf44da6570d26df9ffd048d839a3bb9f5d76a328f6ea62108a"
)
SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256 = (
    "2e33bcca745dfd5e85b02e5f0f1039444cb082cd7bb88ed252677414f83e249a"
)
SECOND_RECOVERY_V3_INCIDENT_FILE_SHA256 = (
    "fd42f197b88bef61ad5696280fd6b4a72f256fd3b0991a069771c5074c9b50e7"
)
SECOND_RECOVERY_V3_REQUEST_SHA256 = (
    "1cc73c5525e096a4df830892f37cdc8062899363a0b75835bb2f04b3a14a0d44"
)
SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256 = (
    "0b605bf4ed55dfe3bc7e14f988f9df002a7a295b8515be1274e774c4b03bf883"
)
SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256 = (
    "9ef82782c03d9915c081e89cf554a531ef3d1fba42836c522d6ebf97cdce0f28"
)
SECOND_RECOVERY_V3_FAILED_CALL_MICROSECONDS = 852_878
SECOND_RECOVERY_RETRY_CALL_ID = "fallback-c1-01"
SECOND_RECOVERY_RETRY_RESERVE_CLASS = "reserve_long"
SECOND_RECOVERY_RETRY_WATCHDOG_SECONDS = 240
SECOND_RECOVERY_OVERLAY_KIND = "phase1_fallback_second_recovery_overlay"
SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH = (
    "src/story_projection_onto/phase1_legacy_provenance.py"
)
SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH = (
    "tests/unit/test_phase1_legacy_provenance.py"
)
SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME = "legacy_evidence_provenance"
SECOND_RECOVERY_V3_C1_REQUIRED_OPERATORS = (
    "event_reification",
    "merge",
    "rare_preservation",
    "schema_relation",
    "temporal_qualification",
)
SECOND_RECOVERY_V3_C0_IMPLEMENTATION_SHA256 = (
    "bb7ec5f00df86ec24eedeaea6b90a1bda473511d0ab9f334cecda9f1970f27c7"
)
SECOND_RECOVERY_V3_C1_IMPLEMENTATION_SHA256 = (
    "828ab6e093d2baf9bf9147ad4202ba53c2bac9af3977be6cc3c216e5e294e55f"
)
SECOND_RECOVERY_C0_IMPLEMENTATION_PATH = "src/story_projection_onto/conditions/c0.py"
SECOND_RECOVERY_C0_REGRESSION_TEST_PATH = "tests/unit/test_c0_condition.py"
SECOND_RECOVERY_C1_IMPLEMENTATION_PATH = "src/story_projection_onto/conditions/c1.py"
SECOND_RECOVERY_C1_CONDITION_PATHWAY_TEST_PATH = "tests/integration/test_condition_pathways.py"
SECOND_RECOVERY_C1_DEVELOPMENT_ASSESSMENT_TEST_PATH = "tests/unit/test_development_assessment.py"
SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256 = (
    "a0bd4f61dff5df8fc668d4d185ec760c280af2c3a9848066223dfea7baf62151"
)
SECOND_RECOVERY_V4_C1_CONDITION_PATHWAY_TEST_SHA256 = (
    "74ca579ad8dc3c51e77387c31d8811f47eecfc33af17a9541bef53c19d7dd281"
)
SECOND_RECOVERY_V4_C1_DEVELOPMENT_ASSESSMENT_TEST_SHA256 = (
    "a7efd8462a73d4aef8a6ff43f4dbfc10ea8a26a931a4930ab76f15d026536611"
)
SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256 = (
    "cf907e89e6ab241ea3708fcf9b0ff083cc96a5052c61e9230906c0ad69863f8d"
)
SECOND_RECOVERY_POST_TWO_FIX_C0_REGRESSION_TEST_SHA256 = (
    "9fc632407a8eda7a7fcb8c6d3994ca204d290daefb71089b1d6f160fd36e0816"
)
SECOND_RECOVERY_UNCHANGED_ONTOLOGY_DRAFT_SCHEMA_SHA256 = (
    "270d0f9acab96a7ebf1edc0606da9494799633146d293992649836283c0d5210"
)
SECOND_RECOVERY_V3_SOURCE_MANIFEST_PATH = (
    "artifacts/public/manifests/source_tree_fallback_development_v5.local.json"
)
SECOND_RECOVERY_V3_SOURCE_MANIFEST_FILE_SHA256 = (
    "47d249905194deca92eae9372e0392ee74d209323179be45dee825092f4fed0c"
)
SECOND_RECOVERY_V3_SOURCE_TREE_SHA256 = (
    "81edb1ba173458970e93b903efa27981a96fb3008a8b4ec221f5810f4c515b20"
)
SECOND_RECOVERY_V3_SOURCE_REVISION = (
    "8a77a3aafa0acc834a6563b0015b5fdff29151f8+fallback-development-freeze"
)
SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS = (
    "configs/study/decoding.json",
    "configs/study/development_construction.json",
    "configs/study/fallback_model.json",
    "prompts/ablations/no_temporal_epistemic_v1.md",
    "prompts/c1_pre/prompt_v1.md",
    "prompts/c2_query/prompt_v1.md",
    "prompts/fixed_select/prompt_v1.md",
    "prompts/repair/prompt_v1.md",
    "schemas/jsonschema/model_visible_query_context.schema.json",
    "schemas/jsonschema/ontology_draft.schema.json",
    "schemas/jsonschema/query_context.schema.json",
    "tests/fixtures/phase1/c1_pre_request.json",
    "tests/fixtures/phase1/c2_query_request.json",
    "tests/fixtures/phase1/fixed_select_request.json",
)
SECOND_RECOVERY_FROZEN_PROMPT_SET_SHA256 = (
    "57fc25af62597b70a63b9ffe157922f1948d5f3333279237bf8c7abb430637fe"
)
SECOND_RECOVERY_FROZEN_MODEL_INPUT_FIXTURES_SHA256 = (
    "3ad86a78ffad6384eec5690be44c239ee4cf22a79f33d628bd78af1017a09216"
)
SECOND_RECOVERY_FROZEN_EVIDENCE_VALUES_SHA256 = (
    "6daaab141b3ce91b88f88a219481c00856842f74048b6f384fcaa7ba71096c98"
)
SECOND_RECOVERY_FROZEN_BUDGET_VALUES_SHA256 = (
    "be648dfaa52360852e3c4386c685b49f567d3d5a915dc72303b0556898920424"
)
SECOND_RECOVERY_FROZEN_DECODING_VALUES_SHA256 = (
    "f3f8a84eec29894ab8a4efaa6a2c1db176b36eea14e4a56d18e52b0a80361f83"
)
SECOND_RECOVERY_VALIDATE_IMPLEMENTATION_PATH = "src/story_projection_onto/validate.py"
SECOND_RECOVERY_CONTRACTS_IMPLEMENTATION_PATH = "src/story_projection_onto/contracts.py"
SECOND_RECOVERY_CONTRACTS_REGRESSION_TEST_PATH = "tests/unit/test_contracts.py"
SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_SOURCE_PATHS = (
    "schemas/jsonschema/ontology_projection.schema.json",
    "schemas/jsonschema/schema_manifest.json",
    "schemas/jsonschema/validated_generation.schema.json",
    "src/story_projection_onto/case_study_gpu.py",
    "src/story_projection_onto/combined_gpu_factory.py",
    "src/story_projection_onto/conditions/c0.py",
    "src/story_projection_onto/conditions/c1.py",
    "src/story_projection_onto/contracts.py",
    "src/story_projection_onto/development_continuation.py",
    "src/story_projection_onto/development_execution.py",
    "src/story_projection_onto/fallback_acceptance.py",
    "src/story_projection_onto/held_out_binding.py",
    "src/story_projection_onto/held_out_c0.py",
    "src/story_projection_onto/held_out_execution.py",
    "src/story_projection_onto/ledger_verify.py",
    "src/story_projection_onto/phase1_acceptance.py",
    "src/story_projection_onto/phase5_execution.py",
    "src/story_projection_onto/scorer_only/development_assessment.py",
    "src/story_projection_onto/store.py",
)
SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_TEST_PATHS = (
    "tests/integration/test_condition_pathways.py",
    "tests/unit/test_c0_condition.py",
    "tests/unit/test_case_study_gpu.py",
    "tests/unit/test_combined_gpu_factory.py",
    "tests/unit/test_contracts.py",
    "tests/unit/test_development_assessment.py",
    "tests/unit/test_development_continuation.py",
    "tests/unit/test_fallback_acceptance.py",
    "tests/unit/test_held_out_c0.py",
    "tests/unit/test_held_out_execution.py",
    "tests/unit/test_ledger_verify.py",
    "tests/unit/test_phase4_metric_pipeline.py",
    "tests/unit/test_phase5_execution.py",
    "tests/unit/test_store.py",
)
SECOND_RECOVERY_INTEGRITY_DISPLAY_SOURCE_PATHS = (
    "src/story_projection_onto/display_selection.py",
    "src/story_projection_onto/feedback_runtime.py",
    "src/story_projection_onto/phase5_capture.py",
    "src/story_projection_onto/phase5_production.py",
    "src/story_projection_onto/phase5_sources.py",
    "src/story_projection_onto/renderer_geometry.py",
    "src/story_projection_onto/scorer_only/geometry_sources.py",
    "src/story_projection_onto/scorer_only/phase4_analysis.py",
    "src/story_projection_onto/ui.py",
    "src/story_projection_onto/validate.py",
    "ui/app.js",
)
SECOND_RECOVERY_INTEGRITY_DISPLAY_TEST_PATHS = (
    "tests/integration/test_condition_pathways.py",
    "tests/integration/test_phase5_static_interface.py",
    "tests/integration/test_renderer_geometry_capture.py",
    "tests/integration/test_ui_smoke.py",
    "tests/unit/test_phase4_analysis_orchestration.py",
    "tests/unit/test_phase5_capture.py",
    "tests/unit/test_phase5_execution.py",
    "tests/unit/test_phase5_feedback_runtime.py",
    "tests/unit/test_phase5_production.py",
    "tests/unit/test_phase5_sources.py",
    "tests/unit/test_renderer_geometry.py",
    "tests/unit/test_ui.py",
)
SECOND_RECOVERY_INTEGRITY_GROUNDING_SOURCE_PATHS = (
    "src/story_projection_onto/metrics/__init__.py",
    "src/story_projection_onto/metrics/adapters.py",
    "src/story_projection_onto/metrics/alignment.py",
    "src/story_projection_onto/metrics/pipeline.py",
    "src/story_projection_onto/scorer_only/development_assessment.py",
    "src/story_projection_onto/scorer_only/phase4_analysis.py",
    "src/story_projection_onto/scorer_only/phase5_feedback.py",
)
SECOND_RECOVERY_INTEGRITY_GROUNDING_TEST_PATHS = (
    "tests/unit/test_development_assessment.py",
    "tests/unit/test_metrics_alignment.py",
    "tests/unit/test_phase4_analysis_orchestration.py",
    "tests/unit/test_phase4_metric_pipeline.py",
    "tests/unit/test_phase5_feedback_scoring.py",
    "tests/unit/test_phase5_report.py",
)
# The lifecycle/accounting set is deliberately centralized here. The overlay
# builder and validator hash every member; changing this exact set changes both
# source-plan identity and the typed correction record.
SECOND_RECOVERY_INTEGRITY_LIFECYCLE_SOURCE_PATHS = (
    "src/story_projection_onto/case_study_gpu.py",
    "src/story_projection_onto/combined_gpu_factory.py",
    "src/story_projection_onto/combined_gpu_production.py",
    "src/story_projection_onto/development_continuation.py",
    "src/story_projection_onto/development_execution.py",
    "src/story_projection_onto/fallback_acceptance.py",
    "src/story_projection_onto/held_out_binding.py",
    "src/story_projection_onto/held_out_c0.py",
    "src/story_projection_onto/held_out_execution.py",
    "src/story_projection_onto/ledger_verify.py",
    "src/story_projection_onto/manifest.py",
    "src/story_projection_onto/metrics/pipeline.py",
    "src/story_projection_onto/phase1_acceptance.py",
    "src/story_projection_onto/phase5_execution.py",
    "src/story_projection_onto/phase5_production.py",
    "src/story_projection_onto/scorer_only/development_assessment.py",
    "src/story_projection_onto/scorer_only/phase4_analysis.py",
    "src/story_projection_onto/store.py",
)
SECOND_RECOVERY_INTEGRITY_LIFECYCLE_TEST_PATHS = (
    "tests/integration/test_held_out_controller.py",
    "tests/unit/test_case_study_execution.py",
    "tests/unit/test_case_study_factory.py",
    "tests/unit/test_case_study_gpu.py",
    "tests/unit/test_combined_gpu_factory.py",
    "tests/unit/test_combined_gpu_production.py",
    "tests/unit/test_development_assessment.py",
    "tests/unit/test_development_continuation.py",
    "tests/unit/test_fallback_acceptance.py",
    "tests/unit/test_held_out_c0.py",
    "tests/unit/test_held_out_execution.py",
    "tests/unit/test_ledger_verify.py",
    "tests/unit/test_manifest.py",
    "tests/unit/test_phase4_analysis_orchestration.py",
    "tests/unit/test_phase4_metric_pipeline.py",
    "tests/unit/test_phase5_execution.py",
    "tests/unit/test_phase5_production.py",
    "tests/unit/test_read_only_store.py",
    "tests/unit/test_store.py",
)
SECOND_RECOVERY_INTEGRITY_DOCUMENTATION_PATHS = (
    "docs/FALLBACK_SECOND_RECOVERY.md",
    "docs/PHASE4_SCORING_AND_ANALYSIS.md",
    "docs/PHASE5_EXECUTION.md",
)
_FALLBACK_CORE_IMPLEMENTATION_FILES = (
    "configs/study/decoding.json",
    "configs/study/development_call_manifest.json",
    "configs/study/development_construction.json",
    "configs/study/fallback_model.json",
    "configs/study/fallback_service_retry_amendment.json",
    "configs/study/gpu_call_inventory.json",
    "configs/study/model.json",
    CERTIFICATE_RELATIVE_PATH,
    "configs/study/resource_limits.json",
    "configs/study/storage_phase_allocations.json",
    "docs/FALLBACK_SECOND_RECOVERY.md",
    "docs/PHASE4_SCORING_AND_ANALYSIS.md",
    "docs/PHASE5_EXECUTION.md",
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
    "src/story_projection_onto/combined_gpu_factory.py",
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
    "src/story_projection_onto/display_selection.py",
    "src/story_projection_onto/evidence.py",
    "src/story_projection_onto/experiment.py",
    "src/story_projection_onto/fallback_acceptance.py",
    "src/story_projection_onto/gpu_runtime.py",
    "src/story_projection_onto/held_out_execution.py",
    "src/story_projection_onto/ledger_verify.py",
    "src/story_projection_onto/llm.py",
    "src/story_projection_onto/manifest.py",
    "src/story_projection_onto/model_gate.py",
    "src/story_projection_onto/metrics/__init__.py",
    "src/story_projection_onto/metrics/adapters.py",
    "src/story_projection_onto/metrics/alignment.py",
    "src/story_projection_onto/metrics/pipeline.py",
    "src/story_projection_onto/phase1_acceptance.py",
    SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH,
    "src/story_projection_onto/query_runtime.py",
    "src/story_projection_onto/scorer_only/acceptance_grounding.py",
    "src/story_projection_onto/scorer_only/development_assessment.py",
    "src/story_projection_onto/scorer_only/phase4_analysis.py",
    "src/story_projection_onto/scorer_only/phase5_feedback.py",
    "src/story_projection_onto/renderer_geometry.py",
    "src/story_projection_onto/store.py",
    "src/story_projection_onto/ui.py",
    "src/story_projection_onto/validate.py",
    "tests/integration/test_condition_pathways.py",
    "tests/unit/test_c0_condition.py",
    "tests/unit/test_combined_gpu_production.py",
    "tests/unit/test_contracts.py",
    "tests/unit/test_development_assessment.py",
    "tests/unit/test_held_out_execution.py",
    "tests/unit/test_ledger_verify.py",
    "tests/unit/test_metrics_alignment.py",
    "tests/unit/test_phase4_analysis_orchestration.py",
    "tests/unit/test_phase4_metric_pipeline.py",
    "tests/unit/test_phase5_feedback_scoring.py",
    "tests/unit/test_phase5_report.py",
    SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH,
    "tests/unit/test_read_only_store.py",
    "tests/unit/test_renderer_geometry.py",
    "tests/unit/test_store.py",
    "tests/unit/test_ui.py",
    "ui/app.js",
)
FALLBACK_IMPLEMENTATION_FILES = tuple(
    sorted(
        {
            *_FALLBACK_CORE_IMPLEMENTATION_FILES,
            *SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_SOURCE_PATHS,
            *SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_TEST_PATHS,
            *SECOND_RECOVERY_INTEGRITY_DISPLAY_SOURCE_PATHS,
            *SECOND_RECOVERY_INTEGRITY_DISPLAY_TEST_PATHS,
            *SECOND_RECOVERY_INTEGRITY_GROUNDING_SOURCE_PATHS,
            *SECOND_RECOVERY_INTEGRITY_GROUNDING_TEST_PATHS,
            *SECOND_RECOVERY_INTEGRITY_LIFECYCLE_SOURCE_PATHS,
            *SECOND_RECOVERY_INTEGRITY_LIFECYCLE_TEST_PATHS,
            *SECOND_RECOVERY_INTEGRITY_DOCUMENTATION_PATHS,
        }
    )
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


class _StrictOverlayRecord(BaseModel):
    """Frozen, extra-forbid component of the second recovery overlay."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SecondRecoveryAuthorization(_StrictOverlayRecord):
    status: Literal["proposed", "authorized"]
    basis: str = Field(min_length=1)
    authorized_by: Literal["user"] | None = None
    recorded_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def require_coherent_authorization(self) -> Self:
        pending = "pending" in self.basis.casefold()
        if self.status == "authorized":
            if self.authorized_by != "user" or self.recorded_at is None or pending:
                raise ValueError("authorized recovery requires a dated, non-pending user basis")
        elif self.authorized_by is not None or self.recorded_at is not None:
            raise ValueError("a proposed recovery cannot name an authorizer or approval time")
        return self


class SecondRecoveryPlanHashes(_StrictOverlayRecord):
    methodological: Sha256Digest
    implementation: Sha256Digest


class SecondRecoveryModelBinding(_StrictOverlayRecord):
    repository: Literal["Qwen/Qwen3-8B-AWQ"]
    revision: Literal["4da05a8edb55c6046cce958586c33b61da07bb79"]
    served_model_name: Literal["qwen3-8b-awq-fallback"]


class SecondRecoveryGpuAccounting(_StrictOverlayRecord):
    total_allocated_microseconds: int = Field(ge=0, strict=True)
    event_count: int = Field(ge=0, strict=True)
    service_session_count: int = Field(ge=0, strict=True)
    by_kind_microseconds: dict[str, int]

    @model_validator(mode="after")
    def reconcile_total(self) -> Self:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.by_kind_microseconds.values()
        ):
            raise ValueError("GPU accounting kinds must contain nonnegative integers")
        if self.total_allocated_microseconds != sum(self.by_kind_microseconds.values()):
            raise ValueError("GPU accounting does not reconcile by kind")
        return self


class SecondRecoveryPredecessorBinding(_StrictOverlayRecord):
    run_id: Literal["fallback-qwen3-8b-awq-development-v3"]
    result_file_sha256: Sha256Digest
    result_manifest_sha256: Sha256Digest
    incident_file_sha256: Sha256Digest
    incident_manifest_sha256: Sha256Digest
    prior_retry_amendment_file_sha256: Sha256Digest
    prior_retry_amendment_manifest_sha256: Sha256Digest
    failed_call_id: Literal["fallback-c1-01"]
    failed_request_sha256: Sha256Digest
    failed_decoder_schema_sha256: Sha256Digest
    failure_type: Literal["RuntimeTransportError"]
    inference_call_reached_generation: Literal[False]
    service_shutdown_verified: Literal[True]
    consumed_recovery_service_starts: Literal[1]
    consumed_reserve_class: Literal["reserve_long"]
    consumed_reserve_slots: Literal[1]


class SecondRecoveryDecoderBinding(_StrictOverlayRecord):
    protocol: Literal["vllm-0.10.2-xgrammar-ignored-string-keywords-v1"]
    vllm_version: Literal["0.10.2"]
    xgrammar_version: Literal["0.1.23"]
    structured_decoder: Literal["vllm-0.10.2-xgrammar-no-fallback"]
    canonical_validation_schema_file_sha256: Sha256Digest
    compatibility_implementation_file_sha256: Sha256Digest
    phase1_schema_builder_file_sha256: Sha256Digest
    regression_test_file_sha256: Sha256Digest
    compatible_decoder_schema_sha256: Sha256Digest
    retry_request_sha256: Sha256Digest
    failed_v3_decoder_schema_reconstructed_sha256: Sha256Digest
    failed_v3_request_reconstructed_sha256: Sha256Digest
    failed_v3_request_exactly_reconstructed: Literal[True]
    retry_scientific_inputs_exactly_reconstructed: Literal[True]
    retry_wire_delta_scope: Literal[
        "guided_schema_schema_derived_runtime_hashes_and_hash_bound_provenance_only"
    ]
    stripped_string_keywords: tuple[str, ...]
    ontology_draft_schema_and_pydantic_validation_unchanged: Literal[True]
    cpu_xgrammar_compilation_required_before_gpu: Literal[True]

    @model_validator(mode="after")
    def require_exact_ignored_keyword_set(self) -> Self:
        if self.stripped_string_keywords != tuple(sorted(VLLM_XGRAMMAR_IGNORED_STRING_KEYWORDS)):
            raise ValueError("decoder overlay changed the exact ignored keyword set")
        return self


class SecondRecoveryEvidenceBridgeBinding(_StrictOverlayRecord):
    """Exact authority and wire binding for the immutable-v3 provenance bridge."""

    protocol: Literal["immutable-v3-phase1-provenance-sidecar-v1"]
    bridge_revision: Literal["phase1/immutable-v3-evidence-provenance/v1"]
    scope: Literal["exact_phase1_acceptance_fixtures_only"]
    certificate_file: Literal["configs/study/phase1_legacy_evidence_provenance.json"]
    certificate_file_sha256: Sha256Digest
    certificate_manifest_sha256: Sha256Digest
    implementation_file: Literal[
        "src/story_projection_onto/phase1_legacy_provenance.py"
    ]
    implementation_file_sha256: Sha256Digest
    regression_test_file: Literal["tests/unit/test_phase1_legacy_provenance.py"]
    regression_test_file_sha256: Sha256Digest
    model_visible_section_name: Literal["legacy_evidence_provenance"]
    retry_call_id: Literal["fallback-c1-01"]
    retry_section_content_sha256: Sha256Digest
    retry_legacy_evidence_sha256: Sha256Digest
    retry_resolved_evidence_sha256: Sha256Digest
    source_critical_validation_fields: tuple[
        Literal["evidence_id", "locator", "source_artifact_hash", "confidence_ceiling"],
        ...,
    ]
    immutable_v3_fixture_bytes_changed: Literal[False]
    cpu_semantic_inference_performed: Literal[False]
    fixed_raw_semantic_comparison_precedes_effective_binding: Literal[True]
    fixed_prompt_preserves_raw_null_source_artifact_hashes: Literal[True]
    repairs_preserve_exact_model_visible_section: Literal[True]

    @model_validator(mode="after")
    def require_exact_source_critical_fields(self) -> Self:
        if self.source_critical_validation_fields != (
            "evidence_id",
            "locator",
            "source_artifact_hash",
            "confidence_ceiling",
        ):
            raise ValueError("legacy evidence bridge changed its source-critical match scope")
        return self


class SecondRecoverySourceBinding(_StrictOverlayRecord):
    predecessor_association_manifest_sha256: Sha256Digest
    predecessor_tree_sha256: Sha256Digest
    current_association_file_sha256: Sha256Digest
    current_association_manifest_sha256: Sha256Digest
    current_tree_sha256: Sha256Digest


class SecondRecoveryDelta(_StrictOverlayRecord):
    additional_fallback_service_loads: Literal[1]
    recovery_service_start_watchdog_seconds: Literal[300]
    authorized_retry_inference_attempts: Literal[1]
    retry_call_id: Literal["fallback-c1-01"]
    retry_attempt_kind: Literal["retry"]
    retry_reserve_call_class: Literal["reserve_long"]
    retry_watchdog_seconds: Literal[240]
    additional_unreserved_inference_attempts: Literal[0]
    original_accounting_events: Literal[286]
    prior_effective_accounting_events: Literal[287]
    amended_effective_accounting_events: Literal[288]
    original_maximum_inference_attempts: Literal[278]
    amended_maximum_inference_attempts: Literal[278]
    prior_consumed_reserve_long_slots: Literal[1]
    projected_consumed_reserve_long_slots_after_retry: Literal[2]
    registered_reserve_long_slot_count: Literal[4]


class SecondRecoveryForecast(_StrictOverlayRecord):
    prior_actual_allocated_seconds: float = Field(ge=0)
    successful_service_start_p95_seconds: float = Field(gt=0)
    corrected_remaining_mandatory_forecast_seconds: float = Field(ge=0)
    corrected_forecast_rows_sha256: Sha256Digest
    service_start_watchdog_seconds: Literal[300.0]
    additional_service_allocation_forecast_seconds: float = Field(gt=0)
    additional_service_allocation_forecast_basis: Literal[
        "max(recovery_startup_watchdog,observed_successful_service_start_p95)"
    ]
    retry_inference_ceiling_seconds: Literal[240.0]
    retry_inference_already_in_remaining_forecast: Literal[True]
    actual_plus_remaining_and_service_start_seconds: float = Field(ge=0)
    scheduled_limit_seconds: float = Field(gt=0)
    scheduled_reserve_seconds: float
    protected_shutdown_seconds: float = Field(ge=0)
    hard_limit_seconds: float = Field(gt=0)
    hard_contingency_after_start_and_shutdown_seconds: float
    admitted: Literal[True]


class SecondRecoveryC0PreDataCorrection(_StrictOverlayRecord):
    """Hash-bound disclosure of the C0 correction made before any study output."""

    classification: Literal["pre_data_implementation_correction"]
    condition: Literal["C0"]
    implementation_file: Literal["src/story_projection_onto/conditions/c0.py"]
    predecessor_implementation_file_sha256: Sha256Digest
    current_implementation_file_sha256: Sha256Digest
    regression_test_file: Literal["tests/unit/test_c0_condition.py"]
    regression_test_file_sha256: Sha256Digest
    change_ids: tuple[
        Literal["parse_story_step_point_and_validity_interval"],
        Literal["admit_supported_served_as_predicate"],
    ]
    predecessor_completed_base_call_count: Literal[0]
    predecessor_development_call_count: Literal[0]
    predecessor_development_output_count: Literal[0]
    retry_request_affected: Literal[False]
    authoritative_plan_rewritten: Literal[False]


class SecondRecoverySemanticValidationUnchangedControls(_StrictOverlayRecord):
    """Scientific inputs and constructive behavior untouched by the correction."""

    prompts: Literal[True]
    evidence: Literal[True]
    model_visible_payload_byte_identity_to_v3_claimed: Literal[False]
    model_visible_delta_limited_to_hash_bound_provenance: Literal[True]
    llm_ontology_draft_schema: Literal[True]
    construction_semantics: Literal[True]
    selection_semantics: Literal[True]
    object_and_display_budgets: Literal[True]
    seeds: Literal[True]


class SecondRecoverySemanticValidationCorrection(_StrictOverlayRecord):
    """Exact post-v3 correction of C1's non-semantic runtime status metadata."""

    classification: Literal["post_v3_integrity_correction"]
    condition: Literal["C1"]
    implementation_file: Literal["src/story_projection_onto/conditions/c1.py"]
    predecessor_implementation_file_sha256: Sha256Digest
    current_implementation_file_sha256: Sha256Digest
    condition_pathway_regression_file: Literal["tests/integration/test_condition_pathways.py"]
    condition_pathway_regression_file_sha256: Sha256Digest
    development_assessment_regression_file: Literal["tests/unit/test_development_assessment.py"]
    development_assessment_regression_file_sha256: Sha256Digest
    change_ids: tuple[Literal["semantic-validation-correction-c1-structural-status-v1"]]
    predecessor_accepted_output_count: Literal[0]
    predecessor_completed_base_call_count: Literal[0]
    predecessor_development_call_count: Literal[0]
    predecessor_development_output_count: Literal[0]
    retry_request_affected: Literal[False]
    changed_surface: Literal["post_generation_validation_metadata_only"]
    validation_status: Literal["accepted"]
    evidence_support_status: Literal["not_applicable"]
    temporal_status: Literal["not_applicable"]
    commitment_status: Literal["not_applicable"]
    standardized_diagnostic: Literal[
        "runtime structural, citation-membership, temporal-shape, and commitment-shape "
        "validation accepted; semantic evidence support, evidence-aligned temporal "
        "correctness, and narrative commitment correctness were not assessed"
    ]
    unchanged_controls: SecondRecoverySemanticValidationUnchangedControls
    authoritative_plan_rewritten: Literal[False]


class SecondRecoveryFrozenInputFileComparison(_StrictOverlayRecord):
    """One current input source compared to the immutable v3 source inventory."""

    path: str = Field(min_length=1)
    predecessor_sha256: Sha256Digest
    current_sha256: Sha256Digest
    byte_identical: Literal[True]

    @model_validator(mode="after")
    def require_actual_byte_identity(self) -> Self:
        if self.predecessor_sha256 != self.current_sha256:
            raise ValueError("frozen input comparison is not byte-identical")
        return self


class SecondRecoveryFrozenInputControls(_StrictOverlayRecord):
    """Immutable-v3 comparison for the fallback block's scientific inputs.

    This record deliberately does not claim whole-wire-request identity: the
    registered decoder-compatibility transform changes guided-schema bytes and
    the hash-bound provenance bridge adds one source-only request section.
    """

    classification: Literal["immutable_v3_fallback_input_comparison"]
    scope: Literal["fallback_micro_pilot_inputs_and_development_configuration"]
    predecessor_source_manifest_file: Literal[
        "artifacts/public/manifests/source_tree_fallback_development_v5.local.json"
    ]
    predecessor_source_manifest_file_sha256: Sha256Digest
    predecessor_source_tree_sha256: Sha256Digest
    predecessor_source_revision: Literal[
        "8a77a3aafa0acc834a6563b0015b5fdff29151f8+fallback-development-freeze"
    ]
    byte_identical_files: tuple[SecondRecoveryFrozenInputFileComparison, ...]
    prompt_file_set_sha256: Sha256Digest
    model_input_fixture_values_sha256: Sha256Digest
    evidence_values_sha256: Sha256Digest
    output_budget_values_sha256: Sha256Digest
    decoding_parameter_values_sha256: Sha256Digest
    ontology_draft_schema_file_sha256: Sha256Digest
    fallback_call_seed_values: tuple[Literal[0], Literal[0], Literal[1], Literal[1]]
    whole_wire_payload_byte_identity_to_failed_v3_claimed: Literal[False]
    wire_payload_difference: Literal[
        "registered_decoder_compatibility_and_hash_bound_provenance_bridge"
    ]

    @model_validator(mode="after")
    def require_exact_sorted_input_inventory(self) -> Self:
        paths = tuple(item.path for item in self.byte_identical_files)
        if paths != tuple(sorted(SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS)):
            raise ValueError("frozen input comparison does not contain the exact inventory")
        return self


class SecondRecoveryProjectionDependencyUnchangedControls(_StrictOverlayRecord):
    """Exact frozen-input record used by the projection correction."""

    frozen_input_controls_sha256: Sha256Digest
    frozen_input_source_bytes_changed: Literal[False]
    registered_decoding_budget_and_seed_values_changed: Literal[False]
    whole_wire_payload_byte_identity_to_failed_v3_claimed: Literal[False]


class SecondRecoveryProjectionDependencyCorrection(_StrictOverlayRecord):
    """Hash-bound post-v3/pre-v4 correction before any accepted model output.

    The predecessor hashes name the exact historical C0 two-fix layer and C1
    semantic-status layer. Current hashes bind the dependency-closed selectors
    and their final-payload validation contract without pretending that the
    query-time selection implementation remained byte-identical.
    """

    classification: Literal["pre_data_projection_dependency_integrity_correction"]
    correction_id: Literal[
        "second-recovery-predata-projection-dependency-integrity-v1"
    ]
    c0_implementation_file: Literal["src/story_projection_onto/conditions/c0.py"]
    predecessor_c0_implementation_file_sha256: Sha256Digest
    current_c0_implementation_file_sha256: Sha256Digest
    c1_implementation_file: Literal["src/story_projection_onto/conditions/c1.py"]
    predecessor_c1_implementation_file_sha256: Sha256Digest
    current_c1_implementation_file_sha256: Sha256Digest
    validate_implementation_file: Literal["src/story_projection_onto/validate.py"]
    validate_implementation_file_sha256: Sha256Digest
    contracts_implementation_file: Literal["src/story_projection_onto/contracts.py"]
    contracts_implementation_file_sha256: Sha256Digest
    c0_regression_test_file: Literal["tests/unit/test_c0_condition.py"]
    c0_regression_test_file_sha256: Sha256Digest
    c1_regression_test_file: Literal["tests/integration/test_condition_pathways.py"]
    c1_regression_test_file_sha256: Sha256Digest
    contracts_regression_test_file: Literal["tests/unit/test_contracts.py"]
    contracts_regression_test_file_sha256: Sha256Digest
    change_ids: tuple[
        Literal["recursive-projection-dependency-closure-v1"],
        Literal["final-selection-structural-revalidation-v1"],
        Literal["projection-semantic-validation-target-v1"],
        Literal["repair-validation-lineage-pairing-v1"],
    ]
    predecessor_accepted_output_count: Literal[0]
    predecessor_base_output_count: Literal[0]
    predecessor_completed_base_call_count: Literal[0]
    predecessor_development_call_count: Literal[0]
    predecessor_development_output_count: Literal[0]
    correction_model_output_count: Literal[0]
    correction_gpu_call_count: Literal[0]
    retry_request_affected: Literal[False]
    prequery_construction_behavior_changed: Literal[False]
    query_time_selection_ranking_changed: Literal[False]
    query_time_selection_feasibility_changed: Literal[True]
    final_projection_validation_targeting_changed: Literal[True]
    final_projection_validation_target_domain: Literal[
        "ontology-projection-structural-validation-target-v1"
    ]
    final_projection_validation_target_excludes_cyclic_fields: Literal[True]
    repair_parent_hash_and_attempt_required_together: Literal[True]
    construction_and_selection_implementation_byte_identical: Literal[False]
    future_projection_outputs_may_change: Literal[True]
    unchanged_controls: SecondRecoveryProjectionDependencyUnchangedControls
    authoritative_plan_rewritten: Literal[False]

    @model_validator(mode="after")
    def require_distinct_corrected_condition_implementations(self) -> Self:
        if (
            self.current_c0_implementation_file_sha256
            == self.predecessor_c0_implementation_file_sha256
            or self.current_c1_implementation_file_sha256
            == self.predecessor_c1_implementation_file_sha256
        ):
            raise ValueError("projection correction must bind distinct predecessor/current bytes")
        return self


class SecondRecoveryIntegrityFileBinding(_StrictOverlayRecord):
    """One exact regular repository file used by a pre-data correction."""

    path: str = Field(min_length=1)
    sha256: Sha256Digest

    @model_validator(mode="after")
    def require_safe_relative_path(self) -> Self:
        path = Path(self.path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("integrity file binding requires a safe relative path")
        return self


class SecondRecoveryIntegritySurfaceFiles(_StrictOverlayRecord):
    """Exact source and regression inventory for one corrected surface."""

    sources: tuple[SecondRecoveryIntegrityFileBinding, ...]
    regression_tests: tuple[SecondRecoveryIntegrityFileBinding, ...]

    @model_validator(mode="after")
    def require_nonempty_sorted_unique_inventories(self) -> Self:
        for label, records in (
            ("source", self.sources),
            ("regression", self.regression_tests),
        ):
            paths = tuple(record.path for record in records)
            if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                raise ValueError(f"{label} integrity files must be nonempty, sorted, and unique")
        return self


class SecondRecoveryConcurrentIntegrityDisclosure(_StrictOverlayRecord):
    """Hash-bound disclosure for other required zero-output integrity repairs."""

    classification: Literal["pre_data_concurrent_integrity_corrections"]
    change_ids: tuple[
        Literal["runtime-semantic-assessment-scope-v1"],
        Literal["registered-display-qualified-dependency-closure-v2"],
        Literal["qualification-aware-scorer-grounding-v1"],
        Literal["append-only-terminal-lifecycle-accounting-v1"],
    ]
    semantic_assessment_scope: SecondRecoveryIntegritySurfaceFiles
    registered_display_feasibility: SecondRecoveryIntegritySurfaceFiles
    scorer_grounding: SecondRecoveryIntegritySurfaceFiles
    lifecycle_accounting: SecondRecoveryIntegritySurfaceFiles
    documentation: tuple[SecondRecoveryIntegrityFileBinding, ...]
    predecessor_accepted_output_count: Literal[0]
    predecessor_base_output_count: Literal[0]
    predecessor_completed_base_call_count: Literal[0]
    predecessor_development_call_count: Literal[0]
    predecessor_development_output_count: Literal[0]
    correction_model_output_count: Literal[0]
    correction_gpu_call_count: Literal[0]
    retry_request_affected: Literal[False]
    frozen_input_controls_sha256: Sha256Digest
    whole_wire_payload_byte_identity_to_failed_v3_claimed: Literal[False]
    semantic_assessment_contract_changed: Literal[True]
    c1_prequery_construction_acceptance_changed: Literal[False]
    final_projection_display_feasibility_changed: Literal[True]
    scorer_grounding_analysis_changed: Literal[True]
    lifecycle_accounting_integrity_changed: Literal[True]
    construction_and_selection_implementation_byte_identical: Literal[False]
    future_analysis_or_display_outputs_may_change: Literal[True]
    authoritative_plan_rewritten: Literal[False]

    @model_validator(mode="after")
    def require_sorted_unique_documentation(self) -> Self:
        paths = tuple(record.path for record in self.documentation)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("integrity documentation files must be nonempty, sorted, and unique")
        return self


class SecondFallbackRecoveryOverlay(_StrictOverlayRecord):
    """Strict proposed-or-authorized overlay for the one v3 transport retry."""

    schema_version: Literal["1.2.0"] = "1.2.0"
    kind: Literal["phase1_fallback_second_recovery_overlay"]
    authorization: SecondRecoveryAuthorization
    authorized_recovery_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    authoritative_plan_file_sha256: SecondRecoveryPlanHashes
    base_gpu_call_inventory_file_sha256: Sha256Digest
    fallback_policy_file_sha256: Sha256Digest
    fallback_activation_manifest_sha256: Sha256Digest
    primary_rejection_manifest_sha256: Sha256Digest
    model: SecondRecoveryModelBinding
    predecessor: SecondRecoveryPredecessorBinding
    cumulative_gpu_accounting: SecondRecoveryGpuAccounting
    decoder_compatibility: SecondRecoveryDecoderBinding
    evidence_provenance_bridge: SecondRecoveryEvidenceBridgeBinding
    source: SecondRecoverySourceBinding
    amendment: SecondRecoveryDelta
    corrected_forecast: SecondRecoveryForecast
    c0_pre_data_correction: SecondRecoveryC0PreDataCorrection
    semantic_validation_correction: SecondRecoverySemanticValidationCorrection
    frozen_input_controls: SecondRecoveryFrozenInputControls
    projection_dependency_correction: SecondRecoveryProjectionDependencyCorrection
    concurrent_integrity_disclosure: SecondRecoveryConcurrentIntegrityDisclosure
    unchanged_scientific_controls: dict[str, bool]
    scope: str = Field(min_length=1)
    authoritative_plans_rewritten: Literal[False]
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def verify_manifest_identity(self) -> Self:
        immutable = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != canonical_sha256(immutable):
            raise ValueError("second recovery overlay manifest hash changed")
        return self


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


class DevelopmentStorageAdmissionReceipt(ImmutableRecord):
    """Exact registered Phase-3 reservation persisted before development starts."""

    admission_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    owner_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    phase: Literal["phase_3"] = "phase_3"
    ledger_phase: str = Field(min_length=1)
    storage_plan_file_sha256: Sha256Digest
    storage_sample_id: Sha256Digest
    sampled_total_allocation_bytes: int = Field(ge=0)
    sampled_max_occupied_bytes: int = Field(ge=0)
    sampled_min_headroom_bytes: int = Field(ge=0)
    declared_growth_bytes: int = Field(ge=0)
    largest_atomic_temporary_bytes: int = Field(ge=0)
    quarantine_allowance_bytes: int = Field(ge=0)
    release_staging_bytes: int = Field(ge=0)
    current_occupied_bytes: int = Field(ge=0)
    additional_reserved_bytes: int = Field(ge=0)
    projected_occupied_bytes: int = Field(ge=0)
    filesystem_free_bytes: int = Field(ge=0)
    projected_allocation_free_bytes: int
    projected_filesystem_free_bytes: int
    effective_projected_headroom_bytes: int
    allowed: bool
    violations: tuple[str, ...]
    sampled_at: AwareDatetime

    @model_validator(mode="after")
    def validate_storage_arithmetic(self) -> Self:
        if (
            self.sampled_max_occupied_bytes + self.sampled_min_headroom_bytes
            > self.sampled_total_allocation_bytes
        ):
            raise ValueError("development storage admission carries an invalid sampled budget")
        expected_additional = (
            self.declared_growth_bytes
            + self.largest_atomic_temporary_bytes
            + self.quarantine_allowance_bytes
            + self.release_staging_bytes
        )
        if self.additional_reserved_bytes != expected_additional:
            raise ValueError("development storage admission changed its exact reservation")
        if self.projected_occupied_bytes != (
            self.current_occupied_bytes + self.additional_reserved_bytes
        ):
            raise ValueError("development storage admission has inconsistent occupancy")
        if self.projected_allocation_free_bytes != (
            self.sampled_total_allocation_bytes - self.projected_occupied_bytes
        ):
            raise ValueError("development storage admission has inconsistent allocation headroom")
        if self.projected_filesystem_free_bytes != (
            self.filesystem_free_bytes - self.additional_reserved_bytes
        ):
            raise ValueError("development storage admission has inconsistent filesystem headroom")
        if self.effective_projected_headroom_bytes != min(
            self.projected_allocation_free_bytes,
            self.projected_filesystem_free_bytes,
        ):
            raise ValueError("development storage admission has inconsistent effective headroom")
        expected_violations: list[str] = []
        if self.projected_occupied_bytes > self.sampled_max_occupied_bytes:
            expected_violations.append("projected_occupancy_exceeds_limit")
        if self.filesystem_free_bytes < self.sampled_min_headroom_bytes:
            expected_violations.append("actual_filesystem_headroom_below_minimum")
        if self.projected_allocation_free_bytes < self.sampled_min_headroom_bytes:
            expected_violations.append("projected_allocation_headroom_below_minimum")
        if self.projected_filesystem_free_bytes < self.sampled_min_headroom_bytes:
            expected_violations.append("projected_filesystem_headroom_below_minimum")
        if self.violations != tuple(expected_violations) or self.allowed != (
            not expected_violations
        ):
            raise ValueError("development storage admission status differs from sampled budget")
        return self


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
    development_storage_admission_hash: Sha256Digest
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
            immutable = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            if (
                manifest.get("manifest_sha256") != expected
                or canonical_sha256(immutable) != expected
            ):
                raise ValueError(f"development bootstrap carries an invalid {name}")
        receipt = self.micro_pilot_acceptance_receipt
        if (
            receipt.get("accepted_result_manifest_sha256") != self.accepted_fallback_result_hash
            or receipt.get("execution_hash") != self.fallback_execution_hash
            or receipt.get("source_tree_sha256") != self.source_tree_hash
            or self.selected_model_freeze.get("micro_pilot_acceptance_receipt_sha256")
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
    development_storage_admission_hash: Sha256Digest
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
            self.selected_model_freeze.get("manifest_sha256") != self.selected_model_freeze_hash
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
    development_storage_admission_hash: Sha256Digest
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
        if self.live_service_identity_hash_before != self.live_service_identity_hash_after:
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


def _gpu_summary_payload(summary: GpuSummary) -> dict[str, object]:
    return {
        "total_allocated_microseconds": summary.total_allocated_microseconds,
        "event_count": summary.event_count,
        "service_session_count": summary.service_session_count,
        "by_kind_microseconds": {
            kind.value: microseconds for kind, microseconds in summary.by_kind_microseconds
        },
    }


def _validated_manifest_object(path: Path, *, expected_kind: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{expected_kind} must be one regular non-symlink file")
    value = _load_object(resolved)
    supplied = value.get("manifest_sha256")
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if (
        value.get("kind") != expected_kind
        or not isinstance(supplied, str)
        or supplied != canonical_sha256(immutable)
    ):
        raise ValueError(f"{expected_kind} manifest identity is invalid")
    return value


def validate_fallback_service_retry_amendment(
    *,
    root: Path,
    amendment_path: Path,
    prior_failure_path: Path,
    run_id: str,
    policy: FallbackModelPolicy,
    activation_certificate: Mapping[str, object],
    primary_result: Mapping[str, object],
    limits: ResourceLimits,
    observed: GpuSummary | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the one user-authorized recovery start and its exact predecessor.

    The base plan remains untouched.  This overlay adds one service-allocation
    event but zero inference attempts.  It is intentionally unusable for any
    other run, predecessor, model, watchdog, or cumulative-ledger state.
    """

    amendment = _validated_manifest_object(
        amendment_path,
        expected_kind="phase1_fallback_service_retry_amendment",
    )
    prior = _validated_manifest_object(
        prior_failure_path,
        expected_kind="phase1_fallback_micro_pilot_result",
    )
    required_root_fields = {
        "schema_version",
        "kind",
        "authorization_status",
        "authorization_basis",
        "recorded_at",
        "authorized_recovery_run_id",
        "authoritative_plan_file_sha256",
        "base_gpu_call_inventory_file_sha256",
        "fallback_policy_file_sha256",
        "fallback_activation_manifest_sha256",
        "primary_rejection_manifest_sha256",
        "model",
        "prior_failure",
        "amendment",
        "forecast",
        "unchanged_scientific_controls",
        "scope",
        "authoritative_plans_rewritten",
        "manifest_sha256",
    }
    if set(amendment) != required_root_fields:
        raise ValueError("fallback retry amendment fields differ from the authorized overlay")
    if (
        amendment.get("authorization_status") != "authorized"
        or amendment.get("authorized_recovery_run_id") != run_id
        or amendment.get("authoritative_plans_rewritten") is not False
        or not isinstance(amendment.get("authorization_basis"), str)
        or not cast(str, amendment["authorization_basis"]).strip()
        or not isinstance(amendment.get("scope"), str)
        or not cast(str, amendment["scope"]).strip()
    ):
        raise ValueError("fallback retry amendment lacks exact user authorization")

    declared_plans = amendment.get("authoritative_plan_file_sha256")
    expected_plans = {
        "methodological": _file_sha256(
            root / "plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
        ),
        "implementation": _file_sha256(
            root / "plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
        ),
    }
    if declared_plans != expected_plans:
        raise ValueError("fallback retry amendment does not bind both authoritative plans")

    inventory_path = root / "configs/study/gpu_call_inventory.json"
    policy_path = root / "configs/study/fallback_model.json"
    inventory = GPUCallInventory.load(inventory_path)
    if (
        amendment.get("base_gpu_call_inventory_file_sha256") != _file_sha256(inventory_path)
        or amendment.get("fallback_policy_file_sha256") != _file_sha256(policy_path)
        or inventory.session_start_count != 8
        or inventory.accounting_events != 286
        or inventory.maximum_inference_attempts != 278
        or policy.service_start_events != 1
    ):
        raise ValueError("fallback retry amendment base call inventory changed")
    if (
        amendment.get("fallback_activation_manifest_sha256")
        != activation_certificate.get("manifest_sha256")
        or activation_certificate.get("reserved_service_start_events") != 1
        or amendment.get("primary_rejection_manifest_sha256")
        != primary_result.get("manifest_sha256")
    ):
        raise ValueError("fallback retry amendment activation lineage changed")
    if amendment.get("model") != {
        "repository": policy.repository,
        "revision": policy.revision,
        "served_model_name": policy.served_model_name,
    }:
        raise ValueError("fallback retry amendment changed the selected model")

    prior_declaration = amendment.get("prior_failure")
    if not isinstance(prior_declaration, Mapping):
        raise ValueError("fallback retry amendment lacks its prior failure declaration")
    prior_runtime = prior.get("runtime")
    prior_accounting = (
        prior_runtime.get("gpu_accounting") if isinstance(prior_runtime, Mapping) else None
    )
    if not isinstance(prior_accounting, Mapping):
        raise ValueError("fallback retry predecessor lacks cumulative GPU accounting")
    expected_prior_declaration = {
        "run_id": prior.get("run_id"),
        "artifact_file_sha256": _file_sha256(prior_failure_path),
        "artifact_manifest_sha256": prior.get("manifest_sha256"),
        "failure_stage": prior.get("failure_stage"),
        "failure_type": prior.get("failure_type"),
        "failed_call_id": prior.get("failed_call_id"),
        "completed_model_calls": prior.get("completed_base_call_count"),
        "service_shutdown_verified": (
            prior.get("vllm_service_stopped") is True
            and prior.get("physical_service_live") is False
        ),
        "cumulative_gpu_microseconds": prior_accounting.get("total_allocated_microseconds"),
        "gpu_event_count": prior_accounting.get("event_count"),
        "gpu_service_session_count": prior_accounting.get("service_session_count"),
    }
    if prior_declaration != expected_prior_declaration:
        raise ValueError("fallback retry amendment does not bind the exact failed predecessor")
    if (
        prior.get("failure_stage") != "fallback_micro_pilot"
        or prior.get("failure_type") != "RuntimeWatchdogTimeout"
        or prior.get("failed_call_id") != "controller-restart-prepare"
        or prior.get("completed_base_call_count") != 0
        or prior.get("completed_call_ids") != []
        or prior.get("vllm_service_stopped") is not True
        or prior.get("physical_service_live") is not False
    ):
        raise ValueError("fallback retry predecessor is not the authorized startup timeout")

    primary_baseline = pre_fallback_gpu_accounting_baseline(primary_result)
    primary_by_kind = cast(Mapping[str, int], primary_baseline["by_kind_microseconds"])
    prior_by_kind = prior_accounting.get("by_kind_microseconds")
    if not isinstance(prior_by_kind, Mapping):
        raise ValueError("fallback retry predecessor has invalid per-kind accounting")
    added_timeout = cast(int, prior_accounting["total_allocated_microseconds"]) - cast(
        int, primary_baseline["total_allocated_microseconds"]
    )
    if (
        added_timeout <= 0
        or prior_accounting.get("event_count") != cast(int, primary_baseline["event_count"]) + 1
        or prior_accounting.get("service_session_count")
        != cast(int, primary_baseline["service_session_count"]) + 1
        or prior_by_kind.get("failure", 0) != primary_by_kind.get("failure", 0)
        or prior_by_kind.get("timeout", 0) != primary_by_kind.get("timeout", 0) + added_timeout
        or set(prior_by_kind) - {"failure", "timeout"}
    ):
        raise ValueError("fallback retry predecessor is not one additional timeout event")
    if observed is not None and _gpu_summary_payload(observed) != dict(prior_accounting):
        raise RuntimeError(
            "fallback retry ledger does not exactly match its authorized failed predecessor"
        )

    delta = amendment.get("amendment")
    expected_delta = {
        "additional_fallback_service_loads": 1,
        "original_total_allocation_events": inventory.session_start_count,
        "amended_total_allocation_events": inventory.session_start_count + 1,
        "recovery_service_start_watchdog_seconds": (AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
        "additional_inference_attempts": 0,
        "maximum_inference_attempts_unchanged": inventory.maximum_inference_attempts,
    }
    if delta != expected_delta:
        raise ValueError("fallback retry amendment delta is not the one authorized switch")
    controls = amendment.get("unchanged_scientific_controls")
    expected_control_names = {
        "model_snapshot",
        "runtime_stack",
        "no_cpu_weight_offload",
        "generation_concurrency_one",
        "prompts",
        "schemas",
        "evidence",
        "decoding",
        "seeds",
        "inference_call_inventory",
        "condition_semantics",
        "validation_and_repair_policy",
    }
    if (
        not isinstance(controls, Mapping)
        or set(controls) != expected_control_names
        or any(value is not True for value in controls.values())
    ):
        raise ValueError("fallback retry amendment weakens a scientific control")

    prior_forecast = prior.get("actual_plus_remaining_forecast")
    declared_forecast = amendment.get("forecast")
    if not isinstance(prior_forecast, Mapping) or not isinstance(declared_forecast, Mapping):
        raise ValueError("fallback retry amendment lacks a complete forecast")
    prior_actual = float(cast(float, prior_forecast["actual_allocated_seconds"]))
    remaining = float(cast(float, prior_forecast["remaining_forecast_seconds"]))
    amended_ceiling = prior_actual + remaining + AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
    expected_forecast = {
        "prior_actual_allocated_seconds": prior_actual,
        "remaining_mandatory_forecast_seconds": remaining,
        "recovery_service_start_ceiling_seconds": float(AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
        "amended_actual_plus_remaining_ceiling_seconds": amended_ceiling,
        "scheduled_limit_seconds": float(limits.scheduled_gpu_seconds),
        "hard_limit_seconds": float(limits.hard_gpu_seconds),
        "admitted": (
            amended_ceiling <= limits.scheduled_gpu_seconds
            and amended_ceiling < limits.hard_gpu_seconds
        ),
    }
    if declared_forecast != expected_forecast or expected_forecast["admitted"] is not True:
        raise ValueError("fallback retry amendment does not fit the GPU schedule")
    return amendment, prior


def _second_recovery_corrected_forecast(
    *,
    predecessor: Mapping[str, object],
    inventory: GPUCallInventory,
    limits: ResourceLimits,
) -> dict[str, object]:
    """Rebuild the v3 forecast without treating its failed transport as latency."""

    raw_timings = predecessor.get("timing_by_call_class")
    if not isinstance(raw_timings, list) or not all(
        isinstance(item, Mapping) for item in raw_timings
    ):
        raise ValueError("v3 predecessor lacks its timing summaries")
    timings = {
        cast(str, item.get("call_class")): item
        for item in cast(Sequence[Mapping[str, object]], raw_timings)
    }
    if set(timings) != {"acceptance_c1", "gpu_session_start"}:
        raise ValueError("v3 predecessor timing classes changed")
    failed_transport = timings["acceptance_c1"]
    if (
        failed_transport.get("sample_count") != 1
        or float(cast(float, failed_transport.get("maximum_seconds")))
        != SECOND_RECOVERY_V3_FAILED_CALL_MICROSECONDS / 1_000_000
    ):
        raise ValueError("v3 failed-transport timing identity changed")
    successful_start = timings["gpu_session_start"]
    if successful_start.get("sample_count") != 1:
        raise ValueError("v3 service-start timing identity changed")
    service_start_seconds = float(cast(float, successful_start.get("maximum_seconds")))
    if not math.isfinite(service_start_seconds) or service_start_seconds <= 0:
        raise ValueError("v3 service-start timing is not positive and finite")

    # The service start completed successfully and remains a valid timing input.
    # The 0.852878-second C1 row never reached generation and is deliberately absent.
    forecast = forecast_gpu_schedule(
        inventory,
        (TimingObservation("gpu_session_start", service_start_seconds),),
        limits=limits,
    )
    runtime = predecessor.get("runtime")
    accounting = runtime.get("gpu_accounting") if isinstance(runtime, Mapping) else None
    if not isinstance(accounting, Mapping):
        raise ValueError("v3 predecessor lacks cumulative GPU accounting")
    service_session_count = accounting.get("service_session_count")
    if service_session_count != 4:
        raise ValueError("v3 predecessor service-session count changed")
    base_service_starts = cast(int, service_session_count) - 1
    reserve_rows = predecessor.get("reserve_consumption")
    if not isinstance(reserve_rows, list) or reserve_rows != [
        {
            "call_id": SECOND_RECOVERY_RETRY_CALL_ID,
            "gpu_event_id": (f"{SECOND_RECOVERY_V3_RUN_ID}-{SECOND_RECOVERY_RETRY_CALL_ID}-gpu"),
            "reservation_id": (f"{SECOND_RECOVERY_V3_RUN_ID}:{SECOND_RECOVERY_RETRY_CALL_ID}"),
            "reserve_call_class": SECOND_RECOVERY_RETRY_RESERVE_CLASS,
            "watchdog_seconds": SECOND_RECOVERY_RETRY_WATCHDOG_SECONDS,
        }
    ]:
        raise ValueError("v3 predecessor reserve consumption changed")

    consumed = Counter({SECOND_RECOVERY_RETRY_RESERVE_CLASS: 1})
    for name in NORMAL_ACCEPTANCE_CLASSES:
        consumed[name] = inventory.call_class(name).count
    consumed["gpu_session_start"] = base_service_starts
    corrected_rows: list[dict[str, object]] = []
    remaining = 0.0
    for row in forecast.rows:
        superseded = row.call_class in NORMAL_ACCEPTANCE_CLASSES
        consumed_count = row.count if superseded else min(row.count, consumed[row.call_class])
        remaining_count = max(0, row.count - consumed_count)
        remaining_seconds = remaining_count * row.forecast_p95_seconds
        remaining += remaining_seconds
        corrected_rows.append(
            {
                **asdict(row),
                "superseded_without_execution": superseded,
                "consumed_count": consumed_count,
                "remaining_count": remaining_count,
                "remaining_forecast_seconds": remaining_seconds,
            }
        )
    actual = cast(int, accounting["total_allocated_microseconds"]) / 1_000_000
    service_allocation_forecast = max(
        float(AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
        service_start_seconds,
    )
    projected = actual + service_allocation_forecast + remaining
    protected_shutdown = float(2 * DEFAULT_SHUTDOWN_SECONDS)
    hard_projection = projected + protected_shutdown
    return {
        "prior_actual_allocated_seconds": actual,
        "successful_service_start_p95_seconds": service_start_seconds,
        "corrected_remaining_mandatory_forecast_seconds": remaining,
        "corrected_forecast_rows_sha256": canonical_sha256(corrected_rows),
        "service_start_watchdog_seconds": float(AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
        "additional_service_allocation_forecast_seconds": (service_allocation_forecast),
        "additional_service_allocation_forecast_basis": (
            "max(recovery_startup_watchdog,observed_successful_service_start_p95)"
        ),
        "retry_inference_ceiling_seconds": float(SECOND_RECOVERY_RETRY_WATCHDOG_SECONDS),
        "retry_inference_already_in_remaining_forecast": True,
        "actual_plus_remaining_and_service_start_seconds": projected,
        "scheduled_limit_seconds": float(limits.scheduled_gpu_seconds),
        "scheduled_reserve_seconds": float(limits.scheduled_gpu_seconds) - projected,
        "protected_shutdown_seconds": protected_shutdown,
        "hard_limit_seconds": float(limits.hard_gpu_seconds),
        "hard_contingency_after_start_and_shutdown_seconds": (
            float(limits.hard_gpu_seconds) - hard_projection
        ),
        "admitted": (
            projected <= limits.scheduled_gpu_seconds and hard_projection < limits.hard_gpu_seconds
        ),
    }


def _second_recovery_c0_pre_data_correction(
    *,
    root: Path,
    predecessor: Mapping[str, object],
    incident: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact disclosure for the C0 correction made after terminal v3.

    The correction is admissible only because v3 produced neither an accepted
    fallback base call nor a development output. This historical layer always
    names the reconstructed post-two-fix bytes; later corrections must never
    make it dynamically claim a newer C0 implementation.
    """

    del root
    terminal = incident.get("terminal_state")
    no_development_output = (
        predecessor.get("development_execution_result") is None
        and predecessor.get("development_continuation_receipt") is None
        and predecessor.get("development_handoff") is None
        and predecessor.get("development_preparation") is None
    )
    if (
        predecessor.get("completed_base_call_count") != 0
        or not isinstance(terminal, Mapping)
        or terminal.get("development_call_count") != 0
        or not no_development_output
    ):
        raise ValueError("C0 correction is not demonstrably pre-data")
    return {
        "classification": "pre_data_implementation_correction",
        "condition": ConditionName.C0_CLASSICAL_PRE.value,
        "implementation_file": SECOND_RECOVERY_C0_IMPLEMENTATION_PATH,
        "predecessor_implementation_file_sha256": (SECOND_RECOVERY_V3_C0_IMPLEMENTATION_SHA256),
        "current_implementation_file_sha256": (
            SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256
        ),
        "regression_test_file": SECOND_RECOVERY_C0_REGRESSION_TEST_PATH,
        "regression_test_file_sha256": (
            SECOND_RECOVERY_POST_TWO_FIX_C0_REGRESSION_TEST_SHA256
        ),
        "change_ids": [
            "parse_story_step_point_and_validity_interval",
            "admit_supported_served_as_predicate",
        ],
        "predecessor_completed_base_call_count": 0,
        "predecessor_development_call_count": 0,
        "predecessor_development_output_count": 0,
        "retry_request_affected": False,
        "authoritative_plan_rewritten": False,
    }


def _second_recovery_semantic_validation_correction(
    *,
    root: Path,
    predecessor: Mapping[str, object],
    incident: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact disclosure for C1's post-v3 runtime-status correction."""

    del root
    terminal = incident.get("terminal_state")
    no_development_output = (
        predecessor.get("development_execution_result") is None
        and predecessor.get("development_continuation_receipt") is None
        and predecessor.get("development_handoff") is None
        and predecessor.get("development_preparation") is None
    )
    if (
        predecessor.get("completed_base_call_count") != 0
        or not isinstance(terminal, Mapping)
        or terminal.get("accepted_output_count") != 0
        or terminal.get("completed_base_call_count") != 0
        or terminal.get("development_call_count") != 0
        or not no_development_output
    ):
        raise ValueError("C1 semantic-validation correction lacks zero-output predecessor proof")
    return {
        "classification": "post_v3_integrity_correction",
        "condition": ConditionName.C1_LLM_PRE.value,
        "implementation_file": SECOND_RECOVERY_C1_IMPLEMENTATION_PATH,
        "predecessor_implementation_file_sha256": (SECOND_RECOVERY_V3_C1_IMPLEMENTATION_SHA256),
        "current_implementation_file_sha256": (SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256),
        "condition_pathway_regression_file": (SECOND_RECOVERY_C1_CONDITION_PATHWAY_TEST_PATH),
        "condition_pathway_regression_file_sha256": (
            SECOND_RECOVERY_V4_C1_CONDITION_PATHWAY_TEST_SHA256
        ),
        "development_assessment_regression_file": (
            SECOND_RECOVERY_C1_DEVELOPMENT_ASSESSMENT_TEST_PATH
        ),
        "development_assessment_regression_file_sha256": (
            SECOND_RECOVERY_V4_C1_DEVELOPMENT_ASSESSMENT_TEST_SHA256
        ),
        "change_ids": ["semantic-validation-correction-c1-structural-status-v1"],
        "predecessor_accepted_output_count": 0,
        "predecessor_completed_base_call_count": 0,
        "predecessor_development_call_count": 0,
        "predecessor_development_output_count": 0,
        "retry_request_affected": False,
        "changed_surface": "post_generation_validation_metadata_only",
        "validation_status": ValidationStatus.ACCEPTED.value,
        "evidence_support_status": EvidenceSupportStatus.NOT_APPLICABLE.value,
        "temporal_status": TemporalDeterminationStatus.NOT_APPLICABLE.value,
        "commitment_status": CommitmentCheckStatus.NOT_APPLICABLE.value,
        "standardized_diagnostic": RUNTIME_STRUCTURAL_ONLY_DIAGNOSTIC,
        "unchanged_controls": {
            "prompts": True,
            "evidence": True,
            "model_visible_payload_byte_identity_to_v3_claimed": False,
            "model_visible_delta_limited_to_hash_bound_provenance": True,
            "llm_ontology_draft_schema": True,
            "construction_semantics": True,
            "selection_semantics": True,
            "object_and_display_budgets": True,
            "seeds": True,
        },
        "authoritative_plan_rewritten": False,
    }


def _second_recovery_zero_output_proof(
    *,
    predecessor: Mapping[str, object],
    incident: Mapping[str, object],
) -> dict[str, int]:
    """Require the immutable v3 boundary to precede every scientific output."""

    terminal = incident.get("terminal_state")
    no_development_output = (
        predecessor.get("development_execution_result") is None
        and predecessor.get("development_continuation_receipt") is None
        and predecessor.get("development_handoff") is None
        and predecessor.get("development_preparation") is None
    )
    if (
        predecessor.get("completed_base_call_count") != 0
        or not isinstance(terminal, Mapping)
        or terminal.get("accepted_output_count") != 0
        or terminal.get("completed_base_call_count") != 0
        or terminal.get("development_call_count") != 0
        or not no_development_output
    ):
        raise ValueError("pre-data integrity correction lacks zero-output predecessor proof")
    return {
        "predecessor_accepted_output_count": 0,
        "predecessor_base_output_count": 0,
        "predecessor_completed_base_call_count": 0,
        "predecessor_development_call_count": 0,
        "predecessor_development_output_count": 0,
        "correction_model_output_count": 0,
        "correction_gpu_call_count": 0,
    }


def _second_recovery_bound_file(root: Path, relative: str) -> dict[str, str]:
    """Hash one exact regular file while rejecting escapes and symlink ancestry."""

    root = root.resolve(strict=True)
    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise ValueError("second recovery integrity path is not safely relative")
    candidate = root
    for part in relative_path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("second recovery integrity path contains a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("second recovery integrity path escaped the source root") from exc
    if not resolved.is_file():
        raise ValueError("second recovery integrity binding requires a regular file")
    return {"path": relative, "sha256": _file_sha256(resolved)}


def _require_phase1_legacy_provenance_bridge(
    root: Path,
    bridge: Phase1LegacyEvidenceProvenanceBridge,
) -> Phase1LegacyEvidenceProvenanceBridge:
    """Require one already verified bridge rooted in this exact source tree."""

    resolved_root = root.resolve(strict=True)
    raw_certificate = resolved_root / CERTIFICATE_RELATIVE_PATH
    expected_certificate = raw_certificate.resolve(strict=True)
    if (
        raw_certificate.is_symlink()
        or bridge.root != resolved_root
        or bridge.certificate_path != expected_certificate
        or _file_sha256(expected_certificate) != bridge.certificate_file_sha256
        or bridge.certificate.bridge_revision != BRIDGE_REVISION
        or bridge.certificate.manifest_sha256 != bridge.manifest_sha256
    ):
        raise ValueError("fallback legacy provenance bridge does not bind this source tree")
    return bridge


def _legacy_provenance_bridge_inventory(
    root: Path,
    bridge: Phase1LegacyEvidenceProvenanceBridge,
) -> dict[str, object]:
    """Bind the certificate, resolver implementation, and focused regression test."""

    bridge = _require_phase1_legacy_provenance_bridge(root, bridge)
    certificate = bridge.certificate
    implementation = _second_recovery_bound_file(
        root,
        SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH,
    )
    regression = _second_recovery_bound_file(
        root,
        SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH,
    )
    return {
        "protocol": certificate.protocol,
        "bridge_revision": certificate.bridge_revision,
        "scope": certificate.scope,
        "certificate_file": CERTIFICATE_RELATIVE_PATH,
        "certificate_file_sha256": bridge.certificate_file_sha256,
        "certificate_manifest_sha256": bridge.manifest_sha256,
        "implementation_file": SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH,
        "implementation_file_sha256": implementation["sha256"],
        "regression_test_file": SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH,
        "regression_test_file_sha256": regression["sha256"],
        "model_visible_section_name": SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME,
        "immutable_v3_fixture_bytes_changed": certificate.fixture_bytes_modified,
        "cpu_semantic_inference_performed": certificate.cpu_semantic_inference_performed,
        "fixed_raw_semantic_comparison_precedes_effective_binding": (
            certificate.fixed_raw_semantic_comparison_precedes_effective_binding
        ),
        "fixed_prompt_preserves_raw_null_source_artifact_hashes": True,
        "repairs_preserve_exact_model_visible_section": True,
    }


def _second_recovery_evidence_bridge_binding(
    *,
    root: Path,
    bridge: Phase1LegacyEvidenceProvenanceBridge,
    retry_request: GuidedJSONRequest,
) -> dict[str, object]:
    """Verify and describe the provenance section carried by the v4 retry."""

    bridge = _require_phase1_legacy_provenance_bridge(root, bridge)
    resolved = bridge.resolve_call(
        call_id=SECOND_RECOVERY_RETRY_CALL_ID,
        condition=ConditionName.C1_LLM_PRE,
        request_fixture="tests/fixtures/phase1/c1_pre_request.json",
    )
    expected_section = bridge.model_visible_provenance_section(
        call_id=SECOND_RECOVERY_RETRY_CALL_ID,
        condition=ConditionName.C1_LLM_PRE,
        request_fixture="tests/fixtures/phase1/c1_pre_request.json",
    )
    try:
        user_payload = json.loads(retry_request.messages[-1].content)
    except (IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("fallback retry lacks its model-visible provenance section") from exc
    packing_sections = {
        section.name: section for section in retry_request.packing.sections
    }
    section_sha256 = hashlib.sha256(
        canonical_json(expected_section).encode("utf-8")
    ).hexdigest()
    packed = packing_sections.get(SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME)
    if (
        not isinstance(user_payload, Mapping)
        or user_payload.get(SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME)
        != expected_section
        or packed is None
        or packed.section_content_hash != section_sha256
        or retry_request.packing.required_section_names.count(
            SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
        )
        != 1
    ):
        raise ValueError("fallback retry provenance section or packing binding changed")
    return {
        **_legacy_provenance_bridge_inventory(root, bridge),
        "retry_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
        "retry_section_content_sha256": section_sha256,
        "retry_legacy_evidence_sha256": resolved.legacy_evidence_sha256,
        "retry_resolved_evidence_sha256": resolved.resolved_evidence_sha256,
        "source_critical_validation_fields": [
            "evidence_id",
            "locator",
            "source_artifact_hash",
            "confidence_ceiling",
        ],
    }


def _second_recovery_integrity_surface(
    root: Path,
    *,
    sources: Sequence[str],
    regression_tests: Sequence[str],
) -> dict[str, object]:
    return {
        "sources": [
            _second_recovery_bound_file(root, relative)
            for relative in sorted(sources)
        ],
        "regression_tests": [
            _second_recovery_bound_file(root, relative)
            for relative in sorted(regression_tests)
        ],
    }


def _second_recovery_frozen_input_controls(root: Path) -> dict[str, object]:
    """Compare the fallback block's exact inputs to its immutable v3 inventory."""

    manifest_binding = _second_recovery_bound_file(
        root,
        SECOND_RECOVERY_V3_SOURCE_MANIFEST_PATH,
    )
    if manifest_binding["sha256"] != SECOND_RECOVERY_V3_SOURCE_MANIFEST_FILE_SHA256:
        raise ValueError("immutable v3 source manifest bytes changed")
    manifest = load_source_manifest(root / SECOND_RECOVERY_V3_SOURCE_MANIFEST_PATH)
    if (
        manifest.get("tree_sha256") != SECOND_RECOVERY_V3_SOURCE_TREE_SHA256
        or manifest.get("revision") != SECOND_RECOVERY_V3_SOURCE_REVISION
    ):
        raise ValueError("immutable v3 source manifest identity changed")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, Sequence):
        raise ValueError("immutable v3 source manifest lacks its file inventory")
    predecessor_by_path: dict[str, str] = {}
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            raise ValueError("immutable v3 source manifest contains an invalid file record")
        relative = raw_file.get("path")
        sha256 = raw_file.get("sha256")
        if isinstance(relative, str) and isinstance(sha256, str):
            predecessor_by_path[relative] = sha256

    comparisons: list[dict[str, object]] = []
    for relative in sorted(SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS):
        predecessor_sha256 = predecessor_by_path.get(relative)
        if predecessor_sha256 is None:
            raise ValueError(f"immutable v3 source manifest omits frozen input {relative}")
        current = _second_recovery_bound_file(root, relative)
        if current["sha256"] != predecessor_sha256:
            raise ValueError(f"frozen fallback input differs from v3: {relative}")
        comparisons.append(
            {
                "path": relative,
                "predecessor_sha256": predecessor_sha256,
                "current_sha256": current["sha256"],
                "byte_identical": True,
            }
        )

    def load_object(relative: str) -> dict[str, object]:
        try:
            value = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load frozen fallback input {relative}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"frozen fallback input is not an object: {relative}")
        return cast(dict[str, object], value)

    prompt_paths = tuple(
        relative
        for relative in SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS
        if relative.startswith("prompts/")
    )
    prompt_set_sha256 = canonical_sha256(
        {
            relative: predecessor_by_path[relative]
            for relative in sorted(prompt_paths)
        }
    )
    fixture_paths = (
        "tests/fixtures/phase1/c1_pre_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/fixed_select_request.json",
    )
    fixtures = {relative: load_object(relative) for relative in fixture_paths}
    evidence_values: dict[str, object] = {}
    budget_values: dict[str, object] = {}
    for relative, fixture in fixtures.items():
        evidence = fixture.get("evidence")
        if evidence is None:
            packet = fixture.get("packet")
            evidence = packet.get("evidence") if isinstance(packet, Mapping) else None
        budgets = fixture.get("budgets")
        if evidence is None or not isinstance(budgets, Mapping):
            raise ValueError(f"frozen fallback fixture lacks evidence or budgets: {relative}")
        evidence_values[relative] = evidence
        budget_values[relative] = dict(budgets)
    decoding_values = load_object("configs/study/decoding.json")
    seed_values = tuple(
        call.seed_block
        for call in fallback_pilot_calls(
            FallbackModelPolicy.load(root / "configs/study/fallback_model.json")
        )
    )
    value_hashes = {
        "prompt_file_set_sha256": prompt_set_sha256,
        "model_input_fixture_values_sha256": canonical_sha256(fixtures),
        "evidence_values_sha256": canonical_sha256(evidence_values),
        "output_budget_values_sha256": canonical_sha256(budget_values),
        "decoding_parameter_values_sha256": canonical_sha256(decoding_values),
        "ontology_draft_schema_file_sha256": predecessor_by_path[
            "schemas/jsonschema/ontology_draft.schema.json"
        ],
    }
    expected_hashes = {
        "prompt_file_set_sha256": SECOND_RECOVERY_FROZEN_PROMPT_SET_SHA256,
        "model_input_fixture_values_sha256": (
            SECOND_RECOVERY_FROZEN_MODEL_INPUT_FIXTURES_SHA256
        ),
        "evidence_values_sha256": SECOND_RECOVERY_FROZEN_EVIDENCE_VALUES_SHA256,
        "output_budget_values_sha256": SECOND_RECOVERY_FROZEN_BUDGET_VALUES_SHA256,
        "decoding_parameter_values_sha256": (
            SECOND_RECOVERY_FROZEN_DECODING_VALUES_SHA256
        ),
        "ontology_draft_schema_file_sha256": (
            SECOND_RECOVERY_UNCHANGED_ONTOLOGY_DRAFT_SCHEMA_SHA256
        ),
    }
    if value_hashes != expected_hashes or seed_values != (0, 0, 1, 1):
        raise ValueError("frozen fallback scientific values differ from immutable v3")
    return {
        "classification": "immutable_v3_fallback_input_comparison",
        "scope": "fallback_micro_pilot_inputs_and_development_configuration",
        "predecessor_source_manifest_file": SECOND_RECOVERY_V3_SOURCE_MANIFEST_PATH,
        "predecessor_source_manifest_file_sha256": manifest_binding["sha256"],
        "predecessor_source_tree_sha256": SECOND_RECOVERY_V3_SOURCE_TREE_SHA256,
        "predecessor_source_revision": SECOND_RECOVERY_V3_SOURCE_REVISION,
        "byte_identical_files": comparisons,
        **value_hashes,
        "fallback_call_seed_values": list(seed_values),
        "whole_wire_payload_byte_identity_to_failed_v3_claimed": False,
        "wire_payload_difference": (
            "registered_decoder_compatibility_and_hash_bound_provenance_bridge"
        ),
    }


def _second_recovery_validate_retry_request_inputs(
    *,
    root: Path,
    predecessor: Mapping[str, object],
    policy: FallbackModelPolicy,
    retry_request: GuidedJSONRequest,
    compatible_schema: Mapping[str, object],
    legacy_provenance_bridge: Phase1LegacyEvidenceProvenanceBridge,
) -> dict[str, object]:
    """Require the retry wire semantics to derive from the immutable v3 inputs.

    The failed v3 wire hash cannot remain identical because the registered
    decoder-compatibility transform changes the guided-schema bytes, and the v4
    request adds one hash-bound, source-only provenance section.  This check
    reconstructs the historical v3 wire first, then checks that the current wire
    differs only by those two declared layers. It prevents a freshly built
    overlay from silently rebaselining a changed prompt, evidence item, budget, seed, model,
    or decoding value through its dynamic ``retry_request_sha256`` field.
    """

    legacy_provenance_bridge = _require_phase1_legacy_provenance_bridge(
        root,
        legacy_provenance_bridge,
    )
    call = fallback_pilot_calls(policy)[0]
    if (
        call.call_id != SECOND_RECOVERY_RETRY_CALL_ID
        or call.condition is not ConditionName.C1_LLM_PRE
        or call.seed_block != 0
        or call.request_fixture != "tests/fixtures/phase1/c1_pre_request.json"
        or call.prompt_file != "prompts/c1_pre/prompt_v1.md"
        or call.required_constructive_operators
        != SECOND_RECOVERY_V3_C1_REQUIRED_OPERATORS
    ):
        raise ValueError("second recovery retry call no longer has its frozen C1 identity")

    try:
        fixture = json.loads((root / call.request_fixture).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load the frozen second-recovery C1 fixture") from exc
    if not isinstance(fixture, dict):
        raise ValueError("frozen second-recovery C1 fixture is not an object")
    try:
        prompt = (root / call.prompt_file).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("cannot load the frozen second-recovery C1 prompt") from exc

    predecessor_runtime = predecessor.get("runtime")
    predecessor_tokenizer = (
        predecessor_runtime.get("tokenizer")
        if isinstance(predecessor_runtime, Mapping)
        else None
    )
    execution_identity = predecessor.get("execution_identity")
    if not isinstance(predecessor_tokenizer, Mapping) or not isinstance(
        execution_identity, Mapping
    ):
        raise ValueError("v3 predecessor lacks its exact tokenizer identity")
    tokenizer_payload = {
        key: copy.deepcopy(value)
        for key, value in predecessor_tokenizer.items()
        if key != "manifest_sha256"
    }
    tokenizer_manifest_sha256 = predecessor_tokenizer.get("manifest_sha256")
    if (
        canonical_sha256(tokenizer_payload) != tokenizer_manifest_sha256
        or execution_identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
    ):
        raise ValueError("v3 predecessor tokenizer manifest does not authenticate itself")
    eos_token_id = predecessor_tokenizer.get("eos_token_id")
    end_of_turn_token_ids = predecessor_tokenizer.get("end_of_turn_token_ids")
    tokenizer_revision = predecessor_tokenizer.get("tokenizer_revision")
    chat_template_sha256 = predecessor_tokenizer.get("chat_template_sha256")
    nonthinking_probe_sha256 = predecessor_tokenizer.get("nonthinking_probe_sha256")
    if (
        isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not isinstance(end_of_turn_token_ids, Sequence)
        or isinstance(end_of_turn_token_ids, (str, bytes, bytearray))
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in end_of_turn_token_ids
        )
        or not isinstance(tokenizer_revision, str)
        or not isinstance(chat_template_sha256, str)
        or not isinstance(nonthinking_probe_sha256, str)
        or not isinstance(tokenizer_manifest_sha256, str)
    ):
        raise ValueError("v3 predecessor tokenizer manifest has invalid fields")

    canonical_schema = copy.deepcopy(
        json.loads(
            (root / "schemas/jsonschema/ontology_draft.schema.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if not isinstance(canonical_schema, dict):
        raise ValueError("canonical OntologyDraft schema is not an object")
    definitions = canonical_schema.get("$defs")
    budget_accounting = (
        definitions.get("BudgetAccounting") if isinstance(definitions, Mapping) else None
    )
    budget_properties = (
        budget_accounting.get("properties")
        if isinstance(budget_accounting, Mapping)
        else None
    )
    if not isinstance(budget_properties, dict):
        raise ValueError("canonical OntologyDraft schema lacks BudgetAccounting")
    budget_properties["input_tokens"] = {"const": 0, "type": "integer"}
    budget_properties["output_tokens"] = {"const": 0, "type": "integer"}
    v3_decoder_schema_sha256 = canonical_sha256(canonical_schema)
    if v3_decoder_schema_sha256 != SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256:
        raise ValueError("failed v3 decoder schema cannot be exactly reconstructed")

    def decoding_for_schema(schema_sha256: str) -> DecodingManifest:
        return DecodingManifest.first_pass(
            seed=call.seed_block,
            eos_token_id=eos_token_id,
            end_of_turn_token_ids=cast(Sequence[int], end_of_turn_token_ids),
            chat_template_hash=chat_template_sha256,
            output_schema_hash=schema_sha256,
            structured_decoder="vllm-0.10.2-xgrammar-no-fallback",
            tokenizer_revision=tokenizer_revision,
            maximum_input_tokens=10_240,
            maximum_output_tokens=2_048,
        )

    def runtime_for_decoding(
        schema_sha256: str,
        decoding: DecodingManifest,
    ) -> dict[str, object]:
        return {
            "model_id": FALLBACK_SERVED_MODEL_NAME,
            "model_revision": FALLBACK_MODEL_REVISION,
            "tokenizer_hash": tokenizer_manifest_sha256,
            "runtime_version": PINNED_RUNTIME_VERSION,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "output_schema_hash": schema_sha256,
            "decoding_config_hash": decoding.content_hash,
        }

    capability_probe = {
        "probe_kind": "registered_constructive_operator_capability",
        "required_operators": list(SECOND_RECOVERY_V3_C1_REQUIRED_OPERATORS),
        "require_one_explicit_decision_per_operator": True,
        "facts_or_gold_labels_supplied": False,
        "coverage_scope": "bounded_micro_pilot_subset",
        "complete_c1_inventory_gate": (
            "integrated_24_call_development_scientific_assessment"
        ),
    }
    def sections_for_runtime(
        runtime: Mapping[str, object],
        *,
        legacy_provenance: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        request_envelope = {
            key: copy.deepcopy(value)
            for key, value in fixture.items()
            if key not in {"upper_ontology", "evidence", "runtime"}
        }
        request_envelope["runtime"] = dict(runtime)
        sections = {
            "upper_ontology": copy.deepcopy(fixture.get("upper_ontology")),
            "evidence_snapshot": copy.deepcopy(fixture.get("evidence")),
            "request_envelope": request_envelope,
            "fallback_capability_probe": capability_probe,
        }
        if legacy_provenance is not None:
            sections[SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME] = copy.deepcopy(
                dict(legacy_provenance)
            )
        return sections

    def request_wire(
        *,
        schema: Mapping[str, object],
        decoding: DecodingManifest,
        sections: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "model": FALLBACK_SERVED_MODEL_NAME,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": canonical_json(sections)},
            ],
            "guided_json": schema,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "top_k": decoding.top_k,
            "min_p": decoding.min_p,
            "presence_penalty": decoding.presence_penalty,
            "frequency_penalty": decoding.frequency_penalty,
            "repetition_penalty": decoding.repetition_penalty,
            "n": decoding.n,
            "best_of": decoding.best_of,
            "use_beam_search": decoding.beam_search,
            "ignore_eos": decoding.ignore_eos,
            "max_tokens": decoding.maximum_output_tokens,
            "seed": decoding.seed,
            "stop_token_ids": list(decoding.stop_token_ids),
            "stream": False,
        }

    v3_decoding = decoding_for_schema(v3_decoder_schema_sha256)
    v3_sections = sections_for_runtime(
        runtime_for_decoding(v3_decoder_schema_sha256, v3_decoding)
    )
    v3_request_sha256 = canonical_sha256(
        request_wire(
            schema=canonical_schema,
            decoding=v3_decoding,
            sections=v3_sections,
        )
    )
    if v3_request_sha256 != SECOND_RECOVERY_V3_REQUEST_SHA256:
        raise ValueError("failed v3 retry request cannot be exactly reconstructed")

    output_schema_sha256 = canonical_sha256(compatible_schema)
    expected_decoding = decoding_for_schema(output_schema_sha256)
    expected_legacy_provenance = legacy_provenance_bridge.model_visible_provenance_section(
        call_id=call.call_id,
        condition=call.condition,
        request_fixture=call.request_fixture,
    )
    expected_sections = sections_for_runtime(
        runtime_for_decoding(output_schema_sha256, expected_decoding),
        legacy_provenance=expected_legacy_provenance,
    )
    expected_messages = (
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=canonical_json(expected_sections)),
    )
    if (
        retry_request.request_id != call.call_id
        or retry_request.model_name != FALLBACK_SERVED_MODEL_NAME
        or retry_request.condition is not ConditionName.C1_LLM_PRE
        or retry_request.messages != expected_messages
        or canonical_sha256(retry_request.output_schema) != output_schema_sha256
        or retry_request.decoding.model_dump(mode="json")
        != expected_decoding.model_dump(mode="json")
    ):
        raise ValueError(
            "second recovery retry request differs from its frozen v3 scientific inputs"
        )

    expected_section_hashes = {
        "output_schema": output_schema_sha256,
        "system_prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        **{
            name: hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
            for name, value in expected_sections.items()
        },
        "chat_protocol": nonthinking_probe_sha256,
    }
    packing_sections = {section.name: section for section in retry_request.packing.sections}
    if set(packing_sections) != set(expected_section_hashes) or any(
        packing_sections[name].section_content_hash != expected_sha256
        for name, expected_sha256 in expected_section_hashes.items()
    ):
        raise ValueError("second recovery retry packing does not bind its exact inputs")
    output_schema_section = packing_sections["output_schema"]
    if (
        output_schema_section.token_count != 0
        or retry_request.packing.required_section_names
        != (
            "system_prompt",
            "output_schema",
            "upper_ontology",
            "evidence_snapshot",
            SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME,
            "fallback_capability_probe",
        )
        or retry_request.packing.tokenizer_revision != tokenizer_revision
        or retry_request.packing.maximum_model_tokens != 12_288
        or retry_request.packing.maximum_input_tokens != 10_240
        or retry_request.packing.reserved_output_tokens != 2_048
        or retry_request.packing.complete_evidence_snapshot is not True
        or retry_request.packing.complete_evidence_packet is not None
        or retry_request.packing.complete_sealed_ontology is not None
    ):
        raise ValueError("second recovery retry packing changed its frozen C1 contract")
    return {
        "failed_v3_decoder_schema_reconstructed_sha256": v3_decoder_schema_sha256,
        "failed_v3_request_reconstructed_sha256": v3_request_sha256,
        "failed_v3_request_exactly_reconstructed": True,
        "retry_scientific_inputs_exactly_reconstructed": True,
        "retry_wire_delta_scope": (
            "guided_schema_schema_derived_runtime_hashes_and_hash_bound_provenance_only"
        ),
    }


def _second_recovery_projection_dependency_correction(
    *,
    root: Path,
    predecessor: Mapping[str, object],
    incident: Mapping[str, object],
) -> dict[str, object]:
    """Bind dependency closure and final structural validation to current bytes."""

    zero_output = _second_recovery_zero_output_proof(
        predecessor=predecessor,
        incident=incident,
    )
    frozen_inputs_sha256 = canonical_sha256(_second_recovery_frozen_input_controls(root))
    return {
        "classification": "pre_data_projection_dependency_integrity_correction",
        "correction_id": "second-recovery-predata-projection-dependency-integrity-v1",
        "c0_implementation_file": SECOND_RECOVERY_C0_IMPLEMENTATION_PATH,
        "predecessor_c0_implementation_file_sha256": (
            SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256
        ),
        "current_c0_implementation_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_C0_IMPLEMENTATION_PATH,
        )["sha256"],
        "c1_implementation_file": SECOND_RECOVERY_C1_IMPLEMENTATION_PATH,
        "predecessor_c1_implementation_file_sha256": (
            SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256
        ),
        "current_c1_implementation_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_C1_IMPLEMENTATION_PATH,
        )["sha256"],
        "validate_implementation_file": SECOND_RECOVERY_VALIDATE_IMPLEMENTATION_PATH,
        "validate_implementation_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_VALIDATE_IMPLEMENTATION_PATH,
        )["sha256"],
        "contracts_implementation_file": SECOND_RECOVERY_CONTRACTS_IMPLEMENTATION_PATH,
        "contracts_implementation_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_CONTRACTS_IMPLEMENTATION_PATH,
        )["sha256"],
        "c0_regression_test_file": SECOND_RECOVERY_C0_REGRESSION_TEST_PATH,
        "c0_regression_test_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_C0_REGRESSION_TEST_PATH,
        )["sha256"],
        "c1_regression_test_file": SECOND_RECOVERY_C1_CONDITION_PATHWAY_TEST_PATH,
        "c1_regression_test_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_C1_CONDITION_PATHWAY_TEST_PATH,
        )["sha256"],
        "contracts_regression_test_file": SECOND_RECOVERY_CONTRACTS_REGRESSION_TEST_PATH,
        "contracts_regression_test_file_sha256": _second_recovery_bound_file(
            root,
            SECOND_RECOVERY_CONTRACTS_REGRESSION_TEST_PATH,
        )["sha256"],
        "change_ids": [
            "recursive-projection-dependency-closure-v1",
            "final-selection-structural-revalidation-v1",
            "projection-semantic-validation-target-v1",
            "repair-validation-lineage-pairing-v1",
        ],
        **zero_output,
        "retry_request_affected": False,
        "prequery_construction_behavior_changed": False,
        "query_time_selection_ranking_changed": False,
        "query_time_selection_feasibility_changed": True,
        "final_projection_validation_targeting_changed": True,
        "final_projection_validation_target_domain": (
            "ontology-projection-structural-validation-target-v1"
        ),
        "final_projection_validation_target_excludes_cyclic_fields": True,
        "repair_parent_hash_and_attempt_required_together": True,
        "construction_and_selection_implementation_byte_identical": False,
        "future_projection_outputs_may_change": True,
        "unchanged_controls": {
            "frozen_input_controls_sha256": frozen_inputs_sha256,
            "frozen_input_source_bytes_changed": False,
            "registered_decoding_budget_and_seed_values_changed": False,
            "whole_wire_payload_byte_identity_to_failed_v3_claimed": False,
        },
        "authoritative_plan_rewritten": False,
    }


def _second_recovery_concurrent_integrity_disclosure(
    *,
    root: Path,
    predecessor: Mapping[str, object],
    incident: Mapping[str, object],
) -> dict[str, object]:
    """Bind required semantic, display, scorer, and lifecycle corrections."""

    frozen_inputs = _second_recovery_frozen_input_controls(root)
    frozen_inputs_sha256 = canonical_sha256(frozen_inputs)
    zero_output = _second_recovery_zero_output_proof(
        predecessor=predecessor,
        incident=incident,
    )
    return {
        "classification": "pre_data_concurrent_integrity_corrections",
        "change_ids": [
            "runtime-semantic-assessment-scope-v1",
            "registered-display-qualified-dependency-closure-v2",
            "qualification-aware-scorer-grounding-v1",
            "append-only-terminal-lifecycle-accounting-v1",
        ],
        "semantic_assessment_scope": _second_recovery_integrity_surface(
            root,
            sources=SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_SOURCE_PATHS,
            regression_tests=SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_TEST_PATHS,
        ),
        "registered_display_feasibility": _second_recovery_integrity_surface(
            root,
            sources=SECOND_RECOVERY_INTEGRITY_DISPLAY_SOURCE_PATHS,
            regression_tests=SECOND_RECOVERY_INTEGRITY_DISPLAY_TEST_PATHS,
        ),
        "scorer_grounding": _second_recovery_integrity_surface(
            root,
            sources=SECOND_RECOVERY_INTEGRITY_GROUNDING_SOURCE_PATHS,
            regression_tests=SECOND_RECOVERY_INTEGRITY_GROUNDING_TEST_PATHS,
        ),
        "lifecycle_accounting": _second_recovery_integrity_surface(
            root,
            sources=SECOND_RECOVERY_INTEGRITY_LIFECYCLE_SOURCE_PATHS,
            regression_tests=SECOND_RECOVERY_INTEGRITY_LIFECYCLE_TEST_PATHS,
        ),
        "documentation": [
            _second_recovery_bound_file(root, relative)
            for relative in sorted(SECOND_RECOVERY_INTEGRITY_DOCUMENTATION_PATHS)
        ],
        **zero_output,
        "retry_request_affected": False,
        "frozen_input_controls_sha256": frozen_inputs_sha256,
        "whole_wire_payload_byte_identity_to_failed_v3_claimed": False,
        "semantic_assessment_contract_changed": True,
        "c1_prequery_construction_acceptance_changed": False,
        "final_projection_display_feasibility_changed": True,
        "scorer_grounding_analysis_changed": True,
        "lifecycle_accounting_integrity_changed": True,
        "construction_and_selection_implementation_byte_identical": False,
        "future_analysis_or_display_outputs_may_change": True,
        "authoritative_plan_rewritten": False,
    }


def _verify_second_recovery_decoder_compiles(schema: Mapping[str, object]) -> None:
    """Run the pinned CPU-only XGrammar conversion without importing vLLM."""

    from importlib.metadata import version

    if version("xgrammar") != "0.1.23":
        raise RuntimeError("second recovery requires the pinned xgrammar 0.1.23")
    import xgrammar

    xgrammar.Grammar.from_json_schema(
        json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def validate_second_fallback_recovery_overlay(
    *,
    root: Path,
    overlay_path: Path,
    v3_result_path: Path,
    v3_incident_path: Path,
    prior_retry_amendment_path: Path,
    prior_retry_failure_path: Path,
    run_id: str,
    policy: FallbackModelPolicy,
    activation_certificate: Mapping[str, object],
    primary_result: Mapping[str, object],
    limits: ResourceLimits,
    source_association: Mapping[str, object],
    source_association_path: Path,
    retry_request: GuidedJSONRequest,
    legacy_provenance_bridge: Phase1LegacyEvidenceProvenanceBridge,
    observed: GpuSummary | None = None,
    require_authorized: bool = True,
    verify_decoder_compilation: bool = False,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate the one possible recovery after the exact terminal v3 incident.

    This deliberately chains through the original amendment validator instead
    of reinterpreting or replacing it.  ``require_authorized=False`` exists only
    for CPU-only proposal inspection; execution callers must retain the default.
    """

    legacy_provenance_bridge = _require_phase1_legacy_provenance_bridge(
        root,
        legacy_provenance_bridge,
    )
    prior_amendment, _ = validate_fallback_service_retry_amendment(
        root=root,
        amendment_path=prior_retry_amendment_path,
        prior_failure_path=prior_retry_failure_path,
        run_id=SECOND_RECOVERY_V3_RUN_ID,
        policy=policy,
        activation_certificate=activation_certificate,
        primary_result=primary_result,
        limits=limits,
    )
    if prior_amendment.get("manifest_sha256") != SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256:
        raise ValueError("second recovery does not chain through the exact v3 amendment")

    predecessor = _validated_manifest_object(
        v3_result_path,
        expected_kind="phase1_fallback_micro_pilot_result",
    )
    incident = _validated_manifest_object(
        v3_incident_path,
        expected_kind="fallback_gpu_acceptance_incident_association",
    )
    if (
        predecessor.get("manifest_sha256") != SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256
        or _file_sha256(v3_result_path) != SECOND_RECOVERY_V3_RESULT_FILE_SHA256
        or incident.get("manifest_sha256") != SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256
        or _file_sha256(v3_incident_path) != SECOND_RECOVERY_V3_INCIDENT_FILE_SHA256
    ):
        raise ValueError("second recovery predecessor is not the exact terminal v3 incident")

    provenance = incident.get("provenance")
    failed_result_provenance = (
        provenance.get("failed_result") if isinstance(provenance, Mapping) else None
    )
    prior_amendment_provenance = (
        provenance.get("retry_amendment") if isinstance(provenance, Mapping) else None
    )
    diagnosis = incident.get("diagnosis")
    cpu_reproduction = diagnosis.get("cpu_reproduction") if isinstance(diagnosis, Mapping) else None
    terminal = incident.get("terminal_state")
    if (
        incident.get("run_id") != SECOND_RECOVERY_V3_RUN_ID
        or not isinstance(failed_result_provenance, Mapping)
        or failed_result_provenance.get("file_sha256") != SECOND_RECOVERY_V3_RESULT_FILE_SHA256
        or failed_result_provenance.get("manifest_sha256")
        != SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256
        or not isinstance(prior_amendment_provenance, Mapping)
        or prior_amendment_provenance.get("file_sha256") != _file_sha256(prior_retry_amendment_path)
        or prior_amendment_provenance.get("manifest_sha256")
        != SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256
        or not isinstance(diagnosis, Mapping)
        or diagnosis.get("classification") != "decoder_schema_rejected_before_generation"
        or diagnosis.get("failed_call_id") != SECOND_RECOVERY_RETRY_CALL_ID
        or diagnosis.get("exception_type") != "RuntimeTransportError"
        or diagnosis.get("inference_call_reached_generation") is not False
        or not isinstance(cpu_reproduction, Mapping)
        or cpu_reproduction.get("request_hash") != SECOND_RECOVERY_V3_REQUEST_SHA256
        or cpu_reproduction.get("schema_sha256") != SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256
        or not isinstance(terminal, Mapping)
        or terminal.get("retry_authorization_consumed") is not True
        or terminal.get("vllm_service_stopped") is not True
        or terminal.get("physical_service_live") is not False
    ):
        raise ValueError("v3 incident no longer certifies its terminal decoder failure")

    calls = predecessor.get("calls")
    runtime = predecessor.get("runtime")
    predecessor_accounting = runtime.get("gpu_accounting") if isinstance(runtime, Mapping) else None
    if (
        predecessor.get("run_id") != SECOND_RECOVERY_V3_RUN_ID
        or predecessor.get("fallback_service_retry_amendment_sha256")
        != SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256
        or predecessor.get("completed_base_call_count") != 0
        or predecessor.get("micro_pilot_passed") is not False
        or predecessor.get("gate_passed") is not False
        or predecessor.get("vllm_service_stopped") is not True
        or predecessor.get("physical_service_live") is not False
        or predecessor.get("recovery_service_start_events_consumed") != 1
        or calls
        != [
            {
                "call_id": SECOND_RECOVERY_RETRY_CALL_ID,
                "exception_type": "RuntimeTransportError",
                "status": "failed",
            }
        ]
        or not isinstance(predecessor_accounting, Mapping)
    ):
        raise ValueError("v3 result is not the exact pre-generation terminal failure")
    accounting = SecondRecoveryGpuAccounting.model_validate(predecessor_accounting)
    accounting_payload = accounting.model_dump(mode="json")
    incident_accounting = incident.get("accounting")
    if (
        not isinstance(incident_accounting, Mapping)
        or incident_accounting.get("cumulative_gpu_microseconds")
        != accounting.total_allocated_microseconds
        or incident_accounting.get("gpu_event_count") != accounting.event_count
        or incident_accounting.get("gpu_service_session_count") != accounting.service_session_count
        or incident_accounting.get("failed_call_microseconds")
        != SECOND_RECOVERY_V3_FAILED_CALL_MICROSECONDS
    ):
        raise ValueError("v3 incident and result GPU accounting differ")
    if observed is not None and _gpu_summary_payload(observed) != accounting_payload:
        raise RuntimeError("second recovery ledger differs from the terminal v3 ledger")

    try:
        overlay = SecondFallbackRecoveryOverlay.model_validate(
            _validated_manifest_object(
                overlay_path,
                expected_kind=SECOND_RECOVERY_OVERLAY_KIND,
            )
        )
    except ValidationError as exc:
        raise ValueError("second recovery overlay contract is invalid") from exc
    if overlay.authorized_recovery_run_id != run_id or run_id == SECOND_RECOVERY_V3_RUN_ID:
        raise ValueError("second recovery overlay authorizes another or non-new run ID")
    if require_authorized and overlay.authorization.status != "authorized":
        raise PermissionError("second recovery overlay remains proposed and cannot execute")

    expected_plans = {
        "methodological": _file_sha256(
            root / "plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
        ),
        "implementation": _file_sha256(
            root / "plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
        ),
    }
    inventory_path = root / "configs/study/gpu_call_inventory.json"
    policy_path = root / "configs/study/fallback_model.json"
    inventory = GPUCallInventory.load(inventory_path)
    if (
        overlay.authoritative_plan_file_sha256.model_dump(mode="json") != expected_plans
        or overlay.base_gpu_call_inventory_file_sha256 != _file_sha256(inventory_path)
        or overlay.fallback_policy_file_sha256 != _file_sha256(policy_path)
        or overlay.fallback_activation_manifest_sha256
        != activation_certificate.get("manifest_sha256")
        or overlay.primary_rejection_manifest_sha256 != primary_result.get("manifest_sha256")
        or overlay.model.model_dump(mode="json")
        != {
            "repository": policy.repository,
            "revision": policy.revision,
            "served_model_name": policy.served_model_name,
        }
        or inventory.accounting_events != 286
        or inventory.maximum_inference_attempts != 278
        or inventory.session_start_count != 8
    ):
        raise ValueError("second recovery changed a frozen plan, model, or inventory")

    predecessor_source = cast(Mapping[str, object], provenance.get("source_tree"))
    expected_predecessor_binding = {
        "run_id": SECOND_RECOVERY_V3_RUN_ID,
        "result_file_sha256": SECOND_RECOVERY_V3_RESULT_FILE_SHA256,
        "result_manifest_sha256": SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256,
        "incident_file_sha256": SECOND_RECOVERY_V3_INCIDENT_FILE_SHA256,
        "incident_manifest_sha256": SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256,
        "prior_retry_amendment_file_sha256": _file_sha256(prior_retry_amendment_path),
        "prior_retry_amendment_manifest_sha256": (SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256),
        "failed_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
        "failed_request_sha256": SECOND_RECOVERY_V3_REQUEST_SHA256,
        "failed_decoder_schema_sha256": SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256,
        "failure_type": "RuntimeTransportError",
        "inference_call_reached_generation": False,
        "service_shutdown_verified": True,
        "consumed_recovery_service_starts": 1,
        "consumed_reserve_class": SECOND_RECOVERY_RETRY_RESERVE_CLASS,
        "consumed_reserve_slots": 1,
    }
    if overlay.predecessor.model_dump(mode="json") != expected_predecessor_binding:
        raise ValueError("second recovery overlay changed its v3 predecessor binding")
    if overlay.cumulative_gpu_accounting.model_dump(mode="json") != accounting_payload:
        raise ValueError("second recovery overlay changed cumulative GPU accounting")

    compatible_schema = base_condition_output_schema(ConditionName.C1_LLM_PRE)
    if retry_request.request_id != SECOND_RECOVERY_RETRY_CALL_ID:
        raise ValueError("second recovery request is not the failed v3 C1 call")
    retry_input_proof = _second_recovery_validate_retry_request_inputs(
        root=root,
        predecessor=predecessor,
        policy=policy,
        retry_request=retry_request,
        compatible_schema=compatible_schema,
        legacy_provenance_bridge=legacy_provenance_bridge,
    )
    expected_decoder = {
        "protocol": "vllm-0.10.2-xgrammar-ignored-string-keywords-v1",
        "vllm_version": "0.10.2",
        "xgrammar_version": "0.1.23",
        "structured_decoder": "vllm-0.10.2-xgrammar-no-fallback",
        "canonical_validation_schema_file_sha256": _file_sha256(
            root / "schemas/jsonschema/ontology_draft.schema.json"
        ),
        "compatibility_implementation_file_sha256": _file_sha256(
            root / "src/story_projection_onto/llm.py"
        ),
        "phase1_schema_builder_file_sha256": _file_sha256(
            root / "src/story_projection_onto/phase1_acceptance.py"
        ),
        "regression_test_file_sha256": _file_sha256(
            root / "tests/integration/test_xgrammar_decoder_compatibility.py"
        ),
        "compatible_decoder_schema_sha256": canonical_sha256(compatible_schema),
        "retry_request_sha256": retry_request.request_hash,
        **retry_input_proof,
        "stripped_string_keywords": sorted(VLLM_XGRAMMAR_IGNORED_STRING_KEYWORDS),
        "ontology_draft_schema_and_pydantic_validation_unchanged": True,
        "cpu_xgrammar_compilation_required_before_gpu": True,
    }
    if (
        retry_request.decoding.structured_decoder != expected_decoder["structured_decoder"]
        or canonical_sha256(retry_request.output_schema)
        != expected_decoder["compatible_decoder_schema_sha256"]
        or overlay.decoder_compatibility.model_dump(mode="json") != expected_decoder
    ):
        raise ValueError("second recovery decoder or exact retry request changed")
    if verify_decoder_compilation:
        _verify_second_recovery_decoder_compiles(compatible_schema)

    expected_evidence_bridge = _second_recovery_evidence_bridge_binding(
        root=root,
        bridge=legacy_provenance_bridge,
        retry_request=retry_request,
    )
    if (
        overlay.evidence_provenance_bridge.model_dump(mode="json")
        != expected_evidence_bridge
    ):
        raise ValueError("second recovery evidence provenance bridge binding changed")

    expected_source = {
        "predecessor_association_manifest_sha256": predecessor_source.get(
            "association_manifest_sha256"
        ),
        "predecessor_tree_sha256": predecessor_source.get("tree_sha256"),
        "current_association_file_sha256": _file_sha256(source_association_path),
        "current_association_manifest_sha256": source_association.get("manifest_sha256"),
        "current_tree_sha256": source_association.get("local_tree_sha256"),
    }
    if overlay.source.model_dump(mode="json") != expected_source:
        raise ValueError("second recovery source association changed")

    expected_delta = {
        "additional_fallback_service_loads": 1,
        "recovery_service_start_watchdog_seconds": (AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
        "authorized_retry_inference_attempts": 1,
        "retry_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
        "retry_attempt_kind": AttemptKind.RETRY.value,
        "retry_reserve_call_class": SECOND_RECOVERY_RETRY_RESERVE_CLASS,
        "retry_watchdog_seconds": SECOND_RECOVERY_RETRY_WATCHDOG_SECONDS,
        "additional_unreserved_inference_attempts": 0,
        "original_accounting_events": inventory.accounting_events,
        "prior_effective_accounting_events": inventory.accounting_events + 1,
        "amended_effective_accounting_events": inventory.accounting_events + 2,
        "original_maximum_inference_attempts": inventory.maximum_inference_attempts,
        "amended_maximum_inference_attempts": inventory.maximum_inference_attempts,
        "prior_consumed_reserve_long_slots": 1,
        "projected_consumed_reserve_long_slots_after_retry": 2,
        "registered_reserve_long_slot_count": inventory.call_class(
            SECOND_RECOVERY_RETRY_RESERVE_CLASS
        ).count,
    }
    if overlay.amendment.model_dump(mode="json") != expected_delta:
        raise ValueError("second recovery delta is not exactly one reserve-charged retry")

    expected_c0_correction = _second_recovery_c0_pre_data_correction(
        root=root,
        predecessor=predecessor,
        incident=incident,
    )
    if overlay.c0_pre_data_correction.model_dump(mode="json") != expected_c0_correction:
        raise ValueError("second recovery misstates the pre-data C0 correction")

    expected_semantic_validation_correction = _second_recovery_semantic_validation_correction(
        root=root,
        predecessor=predecessor,
        incident=incident,
    )
    if (
        overlay.semantic_validation_correction.model_dump(mode="json")
        != expected_semantic_validation_correction
    ):
        raise ValueError("second recovery misstates the C1 semantic-validation correction")

    expected_frozen_input_controls = _second_recovery_frozen_input_controls(root)
    if (
        overlay.frozen_input_controls.model_dump(mode="json")
        != expected_frozen_input_controls
    ):
        raise ValueError("second recovery misstates its immutable-v3 input comparison")

    expected_projection_dependency_correction = (
        _second_recovery_projection_dependency_correction(
            root=root,
            predecessor=predecessor,
            incident=incident,
        )
    )
    if (
        overlay.projection_dependency_correction.model_dump(mode="json")
        != expected_projection_dependency_correction
    ):
        raise ValueError("second recovery misstates the projection-dependency correction")

    expected_concurrent_integrity_disclosure = (
        _second_recovery_concurrent_integrity_disclosure(
            root=root,
            predecessor=predecessor,
            incident=incident,
        )
    )
    if (
        overlay.concurrent_integrity_disclosure.model_dump(mode="json")
        != expected_concurrent_integrity_disclosure
    ):
        raise ValueError("second recovery misstates the concurrent integrity disclosure")

    expected_controls = {
        "model_snapshot": True,
        "runtime_stack": True,
        "no_cpu_weight_offload": True,
        "generation_concurrency_one": True,
        "c1_prequery_llm_construction_semantics": True,
        "c2_query_dependent_construction_semantics": True,
        "a_fixed_select_mechanical_selection_semantics": True,
        "a_fixed_select_raw_seal_checked_before_effective_provenance_binding": True,
        "post_generation_validation_required": True,
        "single_bounded_repair_limit": True,
        "repair_model_prompt_fact_free_diagnostics": True,
        "immutable_v3_fallback_input_comparison_required": True,
    }
    if (
        overlay.unchanged_scientific_controls != expected_controls
        or overlay.authoritative_plans_rewritten is not False
    ):
        raise ValueError("second recovery weakens an unchanged scientific control")

    expected_forecast = _second_recovery_corrected_forecast(
        predecessor=predecessor,
        inventory=inventory,
        limits=limits,
    )
    if (
        overlay.corrected_forecast.model_dump(mode="json") != expected_forecast
        or expected_forecast["admitted"] is not True
    ):
        raise ValueError("second recovery lacks the corrected admitted GPU forecast")
    return overlay.model_dump(mode="json"), predecessor, incident


def _second_recovery_authorization(
    *,
    status: Literal["proposed", "authorized"],
    basis: str | None,
    authorized_at: datetime | None,
) -> dict[str, object]:
    """Normalize explicit authorization inputs without inferring approval."""

    if status == "proposed":
        if authorized_at is not None:
            raise ValueError("a proposed recovery cannot have an authorization time")
        raw: dict[str, object] = {
            "status": "proposed",
            "basis": basis or "Pending explicit user authorization after final source freeze.",
            "authorized_by": None,
            "recorded_at": None,
        }
    else:
        if basis is None or not basis.strip():
            raise ValueError("authorized recovery construction requires an explicit basis")
        if authorized_at is None or authorized_at.tzinfo is None:
            raise ValueError("authorized recovery construction requires an aware timestamp")
        raw = {
            "status": "authorized",
            "basis": basis,
            "authorized_by": "user",
            "recorded_at": authorized_at,
        }
    return SecondRecoveryAuthorization.model_validate(raw).model_dump(mode="json")


def _assert_no_symlink_ancestry(path: Path) -> None:
    """Reject a symlink at any existing component of an absolute path."""

    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError("second recovery restricted path ancestry cannot contain a symlink")


def _restricted_append_only_path(path: Path, *, restricted_root: Path) -> Path:
    """Resolve one non-symlink output below an explicitly restricted root.

    Containment is established before any output directory is created.  Both the
    restricted root and output ancestry are checked again after creation so an
    intermediate symlink cannot redirect a nominally private artifact into a
    public namespace.
    """

    root_candidate = Path(os.path.abspath(os.fspath(restricted_root)))
    if "restricted" not in {part.casefold() for part in root_candidate.parts}:
        raise ValueError("second recovery output root must be explicitly restricted")
    _assert_no_symlink_ancestry(root_candidate)
    root_candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_no_symlink_ancestry(root_candidate)
    if not root_candidate.is_dir():
        raise ValueError("second recovery restricted root is not a directory")
    resolved_root = root_candidate.resolve(strict=True)
    output_candidate = Path(os.path.abspath(os.fspath(path)))
    if output_candidate.suffix != ".json":
        raise ValueError("second recovery output must be a JSON file")
    try:
        relative_output = output_candidate.relative_to(root_candidate)
    except ValueError as exc:
        raise ValueError("second recovery output escaped its restricted root") from exc
    if not relative_output.parts:
        raise ValueError("second recovery output must be below its restricted root")
    _assert_no_symlink_ancestry(output_candidate.parent)
    output_candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _assert_no_symlink_ancestry(output_candidate.parent)
    resolved_parent = output_candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("second recovery output escaped its restricted root") from exc
    resolved = resolved_parent / output_candidate.name
    if resolved.is_symlink():
        raise ValueError("second recovery output cannot be a symlink")
    return resolved


def _write_restricted_append_only_json(
    path: Path,
    value: Mapping[str, object],
) -> None:
    """Atomically create immutable private JSON; exact replay is idempotent."""

    payload = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError("append-only second recovery output already differs")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise FileExistsError(
                    "concurrent append-only second recovery output differs"
                ) from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def build_second_fallback_recovery_overlay(
    *,
    root: Path,
    output_path: Path,
    restricted_output_root: Path,
    v3_result_path: Path,
    v3_incident_path: Path,
    prior_retry_amendment_path: Path,
    prior_retry_failure_path: Path,
    run_id: str,
    policy: FallbackModelPolicy,
    activation_certificate: Mapping[str, object],
    primary_result: Mapping[str, object],
    limits: ResourceLimits,
    source_association: Mapping[str, object],
    source_association_path: Path,
    retry_request: GuidedJSONRequest,
    legacy_provenance_bridge: Phase1LegacyEvidenceProvenanceBridge,
    observed: GpuSummary,
    authorization_status: Literal["proposed", "authorized"] = "proposed",
    authorization_basis: str | None = None,
    authorized_at: datetime | None = None,
    verify_decoder_compilation: bool = True,
) -> dict[str, object]:
    """Deterministically build and append one CPU-only recovery overlay.

    The default is an inert proposal.  Authorized construction requires the
    caller to supply both the real user basis and an aware timestamp.  This
    function never imports vLLM and never constructs or starts a model service.
    """

    root = root.resolve(strict=True)
    legacy_provenance_bridge = _require_phase1_legacy_provenance_bridge(
        root,
        legacy_provenance_bridge,
    )
    output_path = _restricted_append_only_path(
        output_path,
        restricted_root=restricted_output_root,
    )
    validated_source = validate_source_association(
        source_association_path,
        source_root=root,
    )
    if dict(validated_source) != dict(source_association):
        raise ValueError("second recovery builder source association changed")

    prior_amendment, _ = validate_fallback_service_retry_amendment(
        root=root,
        amendment_path=prior_retry_amendment_path,
        prior_failure_path=prior_retry_failure_path,
        run_id=SECOND_RECOVERY_V3_RUN_ID,
        policy=policy,
        activation_certificate=activation_certificate,
        primary_result=primary_result,
        limits=limits,
    )
    predecessor = _validated_manifest_object(
        v3_result_path,
        expected_kind="phase1_fallback_micro_pilot_result",
    )
    incident = _validated_manifest_object(
        v3_incident_path,
        expected_kind="fallback_gpu_acceptance_incident_association",
    )
    runtime = predecessor.get("runtime")
    accounting = runtime.get("gpu_accounting") if isinstance(runtime, Mapping) else None
    provenance = incident.get("provenance")
    predecessor_source = provenance.get("source_tree") if isinstance(provenance, Mapping) else None
    if not isinstance(accounting, Mapping) or not isinstance(predecessor_source, Mapping):
        raise ValueError("second recovery predecessor lacks accounting or source binding")

    inventory_path = root / "configs/study/gpu_call_inventory.json"
    policy_path = root / "configs/study/fallback_model.json"
    inventory = GPUCallInventory.load(inventory_path)
    compatible_schema = base_condition_output_schema(ConditionName.C1_LLM_PRE)
    retry_input_proof = _second_recovery_validate_retry_request_inputs(
        root=root,
        predecessor=predecessor,
        policy=policy,
        retry_request=retry_request,
        compatible_schema=compatible_schema,
        legacy_provenance_bridge=legacy_provenance_bridge,
    )
    authorization = _second_recovery_authorization(
        status=authorization_status,
        basis=authorization_basis,
        authorized_at=authorized_at,
    )
    payload: dict[str, object] = {
        "schema_version": "1.2.0",
        "kind": SECOND_RECOVERY_OVERLAY_KIND,
        "authorization": authorization,
        "authorized_recovery_run_id": run_id,
        "authoritative_plan_file_sha256": {
            "methodological": _file_sha256(
                root / "plan_notes/METHODOLOGICAL_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
            ),
            "implementation": _file_sha256(
                root / "plan_notes/IMPLEMENTATION_PLAN_QUERY_DEPENDENT_TEMPORAL_ONTOLOGY.md"
            ),
        },
        "base_gpu_call_inventory_file_sha256": _file_sha256(inventory_path),
        "fallback_policy_file_sha256": _file_sha256(policy_path),
        "fallback_activation_manifest_sha256": activation_certificate.get("manifest_sha256"),
        "primary_rejection_manifest_sha256": primary_result.get("manifest_sha256"),
        "model": {
            "repository": policy.repository,
            "revision": policy.revision,
            "served_model_name": policy.served_model_name,
        },
        "predecessor": {
            "run_id": SECOND_RECOVERY_V3_RUN_ID,
            "result_file_sha256": _file_sha256(v3_result_path),
            "result_manifest_sha256": predecessor.get("manifest_sha256"),
            "incident_file_sha256": _file_sha256(v3_incident_path),
            "incident_manifest_sha256": incident.get("manifest_sha256"),
            "prior_retry_amendment_file_sha256": _file_sha256(prior_retry_amendment_path),
            "prior_retry_amendment_manifest_sha256": prior_amendment.get("manifest_sha256"),
            "failed_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
            "failed_request_sha256": SECOND_RECOVERY_V3_REQUEST_SHA256,
            "failed_decoder_schema_sha256": (SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256),
            "failure_type": "RuntimeTransportError",
            "inference_call_reached_generation": False,
            "service_shutdown_verified": True,
            "consumed_recovery_service_starts": 1,
            "consumed_reserve_class": SECOND_RECOVERY_RETRY_RESERVE_CLASS,
            "consumed_reserve_slots": 1,
        },
        "cumulative_gpu_accounting": dict(accounting),
        "decoder_compatibility": {
            "protocol": "vllm-0.10.2-xgrammar-ignored-string-keywords-v1",
            "vllm_version": "0.10.2",
            "xgrammar_version": "0.1.23",
            "structured_decoder": "vllm-0.10.2-xgrammar-no-fallback",
            "canonical_validation_schema_file_sha256": _file_sha256(
                root / "schemas/jsonschema/ontology_draft.schema.json"
            ),
            "compatibility_implementation_file_sha256": _file_sha256(
                root / "src/story_projection_onto/llm.py"
            ),
            "phase1_schema_builder_file_sha256": _file_sha256(
                root / "src/story_projection_onto/phase1_acceptance.py"
            ),
            "regression_test_file_sha256": _file_sha256(
                root / "tests/integration/test_xgrammar_decoder_compatibility.py"
            ),
            "compatible_decoder_schema_sha256": canonical_sha256(compatible_schema),
            "retry_request_sha256": retry_request.request_hash,
            **retry_input_proof,
            "stripped_string_keywords": sorted(VLLM_XGRAMMAR_IGNORED_STRING_KEYWORDS),
            "ontology_draft_schema_and_pydantic_validation_unchanged": True,
            "cpu_xgrammar_compilation_required_before_gpu": True,
        },
        "evidence_provenance_bridge": _second_recovery_evidence_bridge_binding(
            root=root,
            bridge=legacy_provenance_bridge,
            retry_request=retry_request,
        ),
        "source": {
            "predecessor_association_manifest_sha256": predecessor_source.get(
                "association_manifest_sha256"
            ),
            "predecessor_tree_sha256": predecessor_source.get("tree_sha256"),
            "current_association_file_sha256": _file_sha256(source_association_path),
            "current_association_manifest_sha256": source_association.get("manifest_sha256"),
            "current_tree_sha256": source_association.get("local_tree_sha256"),
        },
        "amendment": {
            "additional_fallback_service_loads": 1,
            "recovery_service_start_watchdog_seconds": (AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
            "authorized_retry_inference_attempts": 1,
            "retry_call_id": SECOND_RECOVERY_RETRY_CALL_ID,
            "retry_attempt_kind": AttemptKind.RETRY.value,
            "retry_reserve_call_class": SECOND_RECOVERY_RETRY_RESERVE_CLASS,
            "retry_watchdog_seconds": SECOND_RECOVERY_RETRY_WATCHDOG_SECONDS,
            "additional_unreserved_inference_attempts": 0,
            "original_accounting_events": inventory.accounting_events,
            "prior_effective_accounting_events": inventory.accounting_events + 1,
            "amended_effective_accounting_events": inventory.accounting_events + 2,
            "original_maximum_inference_attempts": (inventory.maximum_inference_attempts),
            "amended_maximum_inference_attempts": inventory.maximum_inference_attempts,
            "prior_consumed_reserve_long_slots": 1,
            "projected_consumed_reserve_long_slots_after_retry": 2,
            "registered_reserve_long_slot_count": inventory.call_class(
                SECOND_RECOVERY_RETRY_RESERVE_CLASS
            ).count,
        },
        "corrected_forecast": _second_recovery_corrected_forecast(
            predecessor=predecessor,
            inventory=inventory,
            limits=limits,
        ),
        "c0_pre_data_correction": _second_recovery_c0_pre_data_correction(
            root=root,
            predecessor=predecessor,
            incident=incident,
        ),
        "semantic_validation_correction": (
            _second_recovery_semantic_validation_correction(
                root=root,
                predecessor=predecessor,
                incident=incident,
            )
        ),
        "frozen_input_controls": _second_recovery_frozen_input_controls(root),
        "projection_dependency_correction": (
            _second_recovery_projection_dependency_correction(
                root=root,
                predecessor=predecessor,
                incident=incident,
            )
        ),
        "concurrent_integrity_disclosure": (
            _second_recovery_concurrent_integrity_disclosure(
                root=root,
                predecessor=predecessor,
                incident=incident,
            )
        ),
        "unchanged_scientific_controls": {
            "model_snapshot": True,
            "runtime_stack": True,
            "no_cpu_weight_offload": True,
            "generation_concurrency_one": True,
            "c1_prequery_llm_construction_semantics": True,
            "c2_query_dependent_construction_semantics": True,
            "a_fixed_select_mechanical_selection_semantics": True,
            "a_fixed_select_raw_seal_checked_before_effective_provenance_binding": True,
            "post_generation_validation_required": True,
            "single_bounded_repair_limit": True,
            "repair_model_prompt_fact_free_diagnostics": True,
            "immutable_v3_fallback_input_comparison_required": True,
        },
        "scope": (
            "One reserve-long retry of the exact v3 fallback C1 transport failure "
            "and one additional recovery service start; no other retry is authorized."
        ),
        "authoritative_plans_rewritten": False,
    }
    candidate = SecondFallbackRecoveryOverlay.model_validate(
        {**payload, "manifest_sha256": canonical_sha256(payload)}
    ).model_dump(mode="json")

    with tempfile.TemporaryDirectory(
        dir=output_path.parent,
        prefix=".second-recovery-validation-",
    ) as temporary_directory:
        validation_path = Path(temporary_directory) / "candidate.json"
        validation_path.write_text(canonical_json(candidate) + "\n", encoding="utf-8")
        validated, _, _ = validate_second_fallback_recovery_overlay(
            root=root,
            overlay_path=validation_path,
            v3_result_path=v3_result_path,
            v3_incident_path=v3_incident_path,
            prior_retry_amendment_path=prior_retry_amendment_path,
            prior_retry_failure_path=prior_retry_failure_path,
            run_id=run_id,
            policy=policy,
            activation_certificate=activation_certificate,
            primary_result=primary_result,
            limits=limits,
            source_association=source_association,
            source_association_path=source_association_path,
            retry_request=retry_request,
            legacy_provenance_bridge=legacy_provenance_bridge,
            observed=observed,
            require_authorized=authorization_status == "authorized",
            verify_decoder_compilation=verify_decoder_compilation,
        )
    _write_restricted_append_only_json(output_path, validated)
    return validated


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
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_append_only_json(
    path: Path,
    value: Mapping[str, object],
    *,
    allow_exact_replay: bool = False,
) -> None:
    """Create one fsynced operational receipt without replacing prior bytes."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("append-only orchestration path cannot be a symlink")
    payload = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if allow_exact_replay and path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"append-only orchestration receipt already exists: {path.name}")
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_hashed_object(path: Path, *, expected_kind: str) -> dict[str, object]:
    value = _load_object(path)
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("kind") != expected_kind or value.get("manifest_sha256") != canonical_sha256(
        immutable
    ):
        raise ValueError(f"{expected_kind} manifest hash does not match its contents")
    return value


def _public_development_state(
    value: object,
    *,
    restricted_fields: frozenset[str],
) -> dict[str, object] | None:
    """Redact operational paths while retaining a hash-bound public summary."""

    if not isinstance(value, Mapping):
        return None
    public = dict(value)
    record_hash = public.pop("content_hash", None)
    withheld = sorted(name for name in restricted_fields if name in public)
    for name in withheld:
        public.pop(name)
    if record_hash is not None:
        public["restricted_record_sha256"] = record_hash
    public["restricted_fields_withheld"] = withheld
    return public


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
            return tuple(
                operator
                for operator in all_operators
                if operator
                not in {
                    "abstraction",
                    "contextual_type",
                    "epistemic_qualification",
                    "include_exclude",
                    "split",
                }
            )
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
                "required_constructive_operators": list(self.required_constructive_operators),
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
        (call.call_id, call.form, call.reserve_tier, call.watchdog_seconds) for call in policy.calls
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
    legacy_provenance_bridge = Phase1LegacyEvidenceProvenanceBridge.load(root)
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
        "append_only_orchestration_invocation_required": True,
        "independent_lease_guardian_required_before_service_start": True,
        "guardian_binds_hard_stop_and_exact_execution_arguments": True,
        "durable_preexec_service_identity_gate_required": True,
        "fresh_controller_outputs_must_be_absent": True,
        "resume_requires_exact_invocation_guard_and_stage_bindings": True,
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
                "DevelopmentScientificAssessment.c1_all_construction_operators_exercised"
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
        "legacy_evidence_provenance_bridge": {
            **_legacy_provenance_bridge_inventory(root, legacy_provenance_bridge),
            "source_critical_validation_fields": [
                "evidence_id",
                "locator",
                "source_artifact_hash",
                "confidence_ceiling",
            ],
            "required_for_base_resume_and_repair_validation": True,
        },
        "second_recovery_overlay_support": {
            "authorization_artifact_bundled": False,
            "execution_requires_explicit_dated_user_authorization": True,
            "predecessor_run_id": SECOND_RECOVERY_V3_RUN_ID,
            "predecessor_result_manifest_sha256": (SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256),
            "predecessor_incident_manifest_sha256": (SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256),
            "additional_service_start_events": 1,
            "recovery_startup_watchdog_seconds": (AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS),
            "additional_service_allocation_forecast_basis": (
                "max(recovery_startup_watchdog,observed_successful_service_start_p95)"
            ),
            "authorized_retry_inference_attempts": 1,
            "retry_attempt_kind": AttemptKind.RETRY.value,
            "retry_reserve_call_class": SECOND_RECOVERY_RETRY_RESERVE_CLASS,
            "additional_unreserved_inference_attempts": 0,
            "maximum_inference_attempts_unchanged": True,
            "proposed_overlay_validation_is_cpu_only": True,
            "deterministic_cpu_only_builder_available": True,
            "builder_output_policy": "restricted_append_only_exact_replay",
            "overlay_schema_version": "1.2.0",
            "evidence_provenance_bridge_binding_required": True,
            "retry_wire_delta_scope": (
                "guided_schema_schema_derived_runtime_hashes_and_"
                "hash_bound_provenance_only"
            ),
            "historical_c0_two_fix_layer_is_constant_bound": True,
            "historical_c1_semantic_status_layer_is_constant_bound": True,
            "immutable_v3_fallback_input_comparison_required": True,
            "whole_wire_payload_byte_identity_to_failed_v3_claimed": False,
            "projection_dependency_correction_required": True,
            "concurrent_integrity_disclosure_required": True,
            "predecessor_accepted_output_count": 0,
            "predecessor_base_output_count": 0,
            "predecessor_development_output_count": 0,
            "integrity_correction_model_output_count": 0,
            "integrity_correction_gpu_call_count": 0,
            "c1_prequery_construction_acceptance_unchanged": True,
            "final_projection_selection_feasibility_changed_before_outputs": True,
            "final_projection_validation_targeting_changed_before_outputs": True,
            "future_projection_and_analysis_outputs_may_change": True,
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
    legacy_provenance_bridge: Phase1LegacyEvidenceProvenanceBridge,
) -> GuidedJSONRequest:
    """Build one frozen fallback request with capability and provenance sections."""

    legacy_provenance_bridge = _require_phase1_legacy_provenance_bridge(
        root,
        legacy_provenance_bridge,
    )

    capability_probe = {
        "probe_kind": "registered_constructive_operator_capability",
        "required_operators": list(call.required_constructive_operators),
        "require_one_explicit_decision_per_operator": bool(call.required_constructive_operators),
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
        legacy_provenance_bridge=legacy_provenance_bridge,
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
                valid = (
                    isinstance(abstraction, str)
                    and bool(abstraction)
                    and (
                        schema.get("schema_id") in targets
                        or any(
                            item.get("abstraction") == abstraction
                            for target, item in {**entities, **contextual_types}.items()
                            if target in targets
                        )
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
    provenance_sections = tuple(
        section
        for section in request.packing.sections
        if section.name == SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
    )
    if (
        len(provenance_sections) != 1
        or request.packing.required_section_names.count(
            SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
        )
        != 1
    ):
        raise ValueError("fallback request metadata lacks one required provenance section")
    return {
        "request_hash": request.request_hash,
        "prompt_hash": request.prompt_hash,
        "rendered_input_token_count": request.rendered_input_token_count,
        "decoding_manifest": request.decoding.model_dump(mode="json"),
        "packing_report": request.packing.model_dump(mode="json", by_alias=True),
        "capability_manifest": CapabilityManifest.for_condition(request.condition).model_dump(
            mode="json"
        ),
        "condition_output_schema_sha256": canonical_sha256(request.output_schema),
        "legacy_evidence_provenance_section_sha256": (
            provenance_sections[0].section_content_hash
        ),
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
    legacy_provenance_bridge: Phase1LegacyEvidenceProvenanceBridge,
) -> GuidedJSONRequest:
    """Build the one repair from the original complete request and safe diagnostics."""

    legacy_provenance_bridge = _require_phase1_legacy_provenance_bridge(
        root,
        legacy_provenance_bridge,
    )
    if not diagnostics:
        raise ValueError("fallback repair requires model-visible diagnostics")
    if base_request.model_name != FALLBACK_SERVED_MODEL_NAME:
        raise ValueError("fallback repair base request targets a different model")
    original_sections = json.loads(base_request.messages[-1].content)
    if not isinstance(original_sections, dict):
        raise ValueError("fallback base request does not contain one semantic object")
    legacy_section = original_sections.get(SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME)
    expected_legacy_section = legacy_provenance_bridge.model_visible_provenance_section(
        call_id=base_call.call_id,
        condition=base_call.condition,
        request_fixture=base_call.request_fixture,
    )
    base_packing_sections = {
        section.name: section for section in base_request.packing.sections
    }
    packed_legacy_section = base_packing_sections.get(
        SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
    )
    expected_legacy_section_sha256 = (
        hashlib.sha256(canonical_json(legacy_section).encode("utf-8")).hexdigest()
        if isinstance(legacy_section, Mapping)
        else None
    )
    if (
        not isinstance(legacy_section, Mapping)
        or legacy_section != expected_legacy_section
        or packed_legacy_section is None
        or packed_legacy_section.section_content_hash != expected_legacy_section_sha256
        or base_request.packing.required_section_names.count(
            SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
        )
        != 1
    ):
        raise ValueError("fallback repair base request lacks its exact provenance bridge")
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
    repaired_packing_sections = {section.name: section for section in packing.sections}
    repaired_legacy_section = repaired_packing_sections.get(
        SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME
    )
    if (
        sections.get(SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME) != legacy_section
        or repaired_legacy_section is None
        or repaired_legacy_section.section_content_hash
        != packed_legacy_section.section_content_hash
    ):
        raise ValueError("fallback repair changed its provenance bridge section")
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
    return canonical_sha256({"fixture": call.request_fixture, "seed_block": call.seed_block})


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
        "micro_pilot_acceptance_receipt_sha256": acceptance_receipt["manifest_sha256"],
        "accepted_micro_pilot_result_sha256": acceptance_receipt["accepted_result_manifest_sha256"],
        "accepted_request_family_hash": acceptance_receipt["request_family_hash"],
        "accepted_operator_gate_sha256": acceptance_receipt["operator_gate_sha256"],
        "accepted_grounding_horizon_gate_sha256": acceptance_receipt[
            "grounding_horizon_gate_sha256"
        ],
        "runtime_stack_manifest_sha256": acceptance_receipt["runtime_stack_manifest_sha256"],
        "gpu_hardware_manifest_sha256": acceptance_receipt["gpu_hardware_manifest_sha256"],
        "source_association_manifest_sha256": source_association["manifest_sha256"],
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
        raise ValueError("fallback acceptance receipt requires four bases and at most one repair")
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
        "launcher_configuration_sha256": execution_identity["launcher_configuration_sha256"],
        "tokenizer_manifest_sha256": execution_identity["tokenizer_manifest_sha256"],
        "runtime_stack_manifest_sha256": execution_identity["runtime_stack_manifest_sha256"],
        "gpu_hardware_manifest_sha256": execution_identity["gpu_hardware_manifest_sha256"],
        "request_family_hash": request_family_hash,
        "request_bindings": call_bindings,
        "operator_gate_sha256": canonical_sha256(result["operator_coverage_gate"]),
        "grounding_horizon_gate_sha256": canonical_sha256(
            {"passed": result["grounding_horizon_gate"]}
        ),
        "timing_gate_sha256": canonical_sha256(result["timing_gate"]),
        "resource_gate_sha256": canonical_sha256(result["cumulative_resource_gate"]),
        "forecast_sha256": canonical_sha256(result["post_fallback_full_manifest_forecast"]),
        "controller_resume_gate_sha256": canonical_sha256(result["controller_resume_gate"]),
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


@dataclass(slots=True)
class FallbackAcceptanceRunner:
    root: Path
    legacy_provenance_bridge: Phase1LegacyEvidenceProvenanceBridge
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
    retry_amendment: Mapping[str, object] | None = None
    prior_fallback_failure: Mapping[str, object] | None = None
    second_recovery_overlay: Mapping[str, object] | None = None
    second_recovery_v3_result: Mapping[str, object] | None = None
    second_recovery_v3_incident: Mapping[str, object] | None = None
    service_start_watchdog_seconds: int = DEFAULT_FALLBACK_STARTUP_WATCHDOG_SECONDS
    development_adopter: DevelopmentContinuationAdopter | None = None
    runtime_stack: RuntimeStackManifest | None = None
    gpu_hardware: GPUHardwareIdentity | None = None
    _last_shutdown_uptime: ServiceUptime | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_service_adoption_failure_type: str | None = field(
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
        self.legacy_provenance_bridge = _require_phase1_legacy_provenance_bridge(
            self.root,
            self.legacy_provenance_bridge,
        )
        if self.service.configuration.model_candidate != "fallback":
            raise ValueError("fallback runner cannot use the primary model")
        amendment_present = self.retry_amendment is not None
        if amendment_present != (self.prior_fallback_failure is not None):
            raise ValueError("fallback retry amendment and predecessor must be supplied together")
        second_recovery_values = (
            self.second_recovery_overlay,
            self.second_recovery_v3_result,
            self.second_recovery_v3_incident,
        )
        second_recovery_present = all(value is not None for value in second_recovery_values)
        if any(value is not None for value in second_recovery_values) != second_recovery_present:
            raise ValueError("second recovery overlay and both v3 records are inseparable")
        if second_recovery_present and not amendment_present:
            raise ValueError("second recovery must retain the original v3 amendment chain")
        expected_watchdog = (
            AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
            if amendment_present
            else DEFAULT_FALLBACK_STARTUP_WATCHDOG_SECONDS
        )
        if self.service_start_watchdog_seconds != expected_watchdog:
            raise ValueError("fallback startup watchdog differs from its authorization")
        if (
            amendment_present
            and self.retry_amendment is not None
            and self.retry_amendment.get("authorized_recovery_run_id")
            != (SECOND_RECOVERY_V3_RUN_ID if second_recovery_present else self.run_id)
        ):
            raise ValueError("fallback retry amendment authorizes another run ID")
        if second_recovery_present:
            assert self.second_recovery_overlay is not None
            assert self.second_recovery_v3_result is not None
            assert self.second_recovery_v3_incident is not None
            authorization = self.second_recovery_overlay.get("authorization")
            if (
                not isinstance(authorization, Mapping)
                or authorization.get("status") != "authorized"
                or self.second_recovery_overlay.get("authorized_recovery_run_id") != self.run_id
                or self.second_recovery_v3_result.get("manifest_sha256")
                != SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256
                or self.second_recovery_v3_incident.get("manifest_sha256")
                != SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256
            ):
                raise ValueError("second fallback recovery is not explicitly authorized")

    @property
    def retry_amendment_hash(self) -> str | None:
        if self.retry_amendment is None:
            return None
        return cast(str, self.retry_amendment["manifest_sha256"])

    @property
    def prior_fallback_failure_hash(self) -> str | None:
        if self.prior_fallback_failure is None:
            return None
        return cast(str, self.prior_fallback_failure["manifest_sha256"])

    @property
    def second_recovery_overlay_hash(self) -> str | None:
        if self.second_recovery_overlay is None:
            return None
        return cast(str, self.second_recovery_overlay["manifest_sha256"])

    @property
    def second_recovery_v3_result_hash(self) -> str | None:
        if self.second_recovery_v3_result is None:
            return None
        return cast(str, self.second_recovery_v3_result["manifest_sha256"])

    @property
    def second_recovery_v3_incident_hash(self) -> str | None:
        if self.second_recovery_v3_incident is None:
            return None
        return cast(str, self.second_recovery_v3_incident["manifest_sha256"])

    def _recovery_service_start_event_ids(self) -> tuple[str, ...]:
        if self.second_recovery_overlay is not None:
            return (
                f"{SECOND_RECOVERY_V3_RUN_ID}-service-start-001",
                f"{self.run_id}-service-start-001",
            )
        if self.retry_amendment is not None:
            return (f"{self.run_id}-service-start-001",)
        return ()

    def _base_inventory_consumed_service_starts(self) -> tuple[int, int]:
        lifecycle_kinds = {
            GpuEventKind.GPU_SESSION_START.value,
            GpuEventKind.RESTART.value,
        }
        lifecycle_events = tuple(
            event
            for event in self.ledger.gpu_events()
            if event.event_kind in {GpuEventKind.GPU_SESSION_START, GpuEventKind.RESTART}
            or json.loads(event.details_json).get("intended_event_kind") in lifecycle_kinds
        )
        recovery_event_ids = self._recovery_service_start_event_ids()
        counts = Counter(event.event_id for event in lifecycle_events)
        if any(counts[event_id] > 1 for event_id in recovery_event_ids):
            raise RuntimeError("GPU ledger contains excess fallback recovery service starts")
        recovery_count = sum(counts[event_id] for event_id in recovery_event_ids)
        return len(lifecycle_events) - recovery_count, recovery_count

    def _remaining_mandatory_forecast_seconds(
        self,
        state: Mapping[str, object],
        *,
        exclude_upcoming_base_service_start: bool = False,
    ) -> float:
        """Return every still-required registered second after the next GPU action."""

        limits = ResourceLimits.load(self.root / "configs/study/resource_limits.json")
        inventory = GPUCallInventory.load(self.root / "configs/study/gpu_call_inventory.json")
        reference = forecast_gpu_schedule(inventory, limits=limits)
        raw_receipts = state.get("reserve_consumption", [])
        if not isinstance(raw_receipts, list) or not all(
            isinstance(item, Mapping) for item in raw_receipts
        ):
            raise RuntimeError("fallback checkpoint reserve inventory is invalid")
        receipts = _reserve_receipts(
            self.ledger,
            cast(Sequence[Mapping[str, object]], raw_receipts),
        )
        consumed = Counter(cast(str, receipt["reserve_call_class"]) for receipt in receipts)
        for name in NORMAL_ACCEPTANCE_CLASSES:
            consumed[name] = inventory.call_class(name).count
        base_service_starts, _ = self._base_inventory_consumed_service_starts()
        consumed["gpu_session_start"] = base_service_starts
        if exclude_upcoming_base_service_start:
            consumed["gpu_session_start"] += 1
        if state.get("development_continuation_completed") is True:
            for name in (
                "development_c1",
                "development_c2",
                "development_fixed_select",
                "development_ablation",
                "development_repair",
            ):
                consumed[name] = inventory.call_class(name).count
        return sum(
            max(0, row.count - consumed[row.call_class]) * row.forecast_p95_seconds
            for row in reference.rows
        )

    def _effective_inventory_manifest(self) -> dict[str, object]:
        inventory_path = self.root / "configs/study/gpu_call_inventory.json"
        inventory = GPUCallInventory.load(inventory_path)
        recovery_count = (
            2
            if self.second_recovery_overlay is not None
            else 1
            if self.retry_amendment is not None
            else 0
        )
        authorized_retry_count = 1 if self.second_recovery_overlay is not None else 0
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "effective_gpu_call_inventory",
            "base_inventory_file_sha256": _file_sha256(inventory_path),
            "base_accounting_events": inventory.accounting_events,
            "base_inference_attempts": inventory.maximum_inference_attempts,
            "base_service_start_events": inventory.session_start_count,
            "recovery_service_start_events": recovery_count,
            "effective_accounting_events": inventory.accounting_events + recovery_count,
            "effective_inference_attempts": inventory.maximum_inference_attempts,
            "effective_service_start_events": inventory.session_start_count + recovery_count,
            "recovery_service_start_watchdog_seconds": (
                self.service_start_watchdog_seconds if recovery_count else None
            ),
            "retry_amendment_sha256": self.retry_amendment_hash,
            "second_recovery_overlay_sha256": self.second_recovery_overlay_hash,
            "authorized_retry_inference_attempts": authorized_retry_count,
            "additional_unreserved_inference_attempts": 0,
            "retry_reserve_call_class": (
                SECOND_RECOVERY_RETRY_RESERVE_CLASS if authorized_retry_count else None
            ),
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}

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
            "source_association_manifest_sha256": self.source_association["manifest_sha256"],
            "source_tree_sha256": self.source_association["local_tree_sha256"],
            "pre_fallback_gpu_accounting_sha256": (
                None
                if self.pre_fallback_gpu_accounting is None
                else self.pre_fallback_gpu_accounting["manifest_sha256"]
            ),
            "fallback_service_retry_amendment_sha256": self.retry_amendment_hash,
            "prior_fallback_failure_sha256": self.prior_fallback_failure_hash,
            "second_fallback_recovery_overlay_sha256": (self.second_recovery_overlay_hash),
            "second_recovery_v3_result_sha256": self.second_recovery_v3_result_hash,
            "second_recovery_v3_incident_sha256": (self.second_recovery_v3_incident_hash),
            "service_start_watchdog_seconds": self.service_start_watchdog_seconds,
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
            "development_storage_admissions": [],
            "development_storage_admission_hash": None,
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
            "service_adoption_failure_type": None,
            "active_call_id": None,
            "active_attempt": None,
            "failed_call_id": None,
            "interrupted_call_terminalization": None,
            "interrupted_call_lifecycle_finalized": False,
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
        if (
            state.get("failed_call_id") is not None
            or state.get("active_call_id") is not None
            or state.get("active_attempt") is not None
        ):
            raise RuntimeError("fallback checkpoint is terminal after a failed/interrupted attempt")
        return state

    def _second_recovery_retry_lineage(self) -> tuple[str, str] | None:
        """Verify and return the exact failed v3 job/attempt reused by the retry."""

        if self.second_recovery_overlay is None:
            return None
        parent_model_call_id = f"{SECOND_RECOVERY_V3_RUN_ID}-{SECOND_RECOVERY_RETRY_CALL_ID}"
        parent_attempt_id = f"{parent_model_call_id}-attempt"
        parent = self.ledger.get_model_call(parent_model_call_id)
        if not self.ledger.is_frozen_legacy_fallback_c1_retry_prebuild(parent.job_id):
            raise RuntimeError("second recovery v3 job classification changed in the ledger")
        attempt = self.ledger.attempt_lineage(parent_attempt_id)[-1]
        event = _event_for(
            self.ledger,
            f"{SECOND_RECOVERY_V3_RUN_ID}-{SECOND_RECOVERY_RETRY_CALL_ID}-gpu",
        )
        if event is None:
            raise RuntimeError("second recovery lacks the exact failed v3 GPU event")
        details = json.loads(event.details_json)
        failures = self.ledger.failures_for_lineage(parent_attempt_id)
        if (
            parent.attempt_id != parent_attempt_id
            or parent.successful
            or parent.request_hash != SECOND_RECOVERY_V3_REQUEST_SHA256
            or parent.response_artifact_hash is not None
            or parent.prompt_tokens != 0
            or parent.completion_tokens != 0
            or parent.allocated_gpu_microseconds != SECOND_RECOVERY_V3_FAILED_CALL_MICROSECONDS
            or attempt.attempt_kind is not AttemptKind.BASE
            or attempt.parent_attempt_id is not None
            or event.succeeded is not False
            or event.allocated_microseconds != SECOND_RECOVERY_V3_FAILED_CALL_MICROSECONDS
            or not isinstance(details, Mapping)
            or details.get("reserve_call_class") != SECOND_RECOVERY_RETRY_RESERVE_CLASS
            or details.get("reserve_reservation_id")
            != f"{SECOND_RECOVERY_V3_RUN_ID}:{SECOND_RECOVERY_RETRY_CALL_ID}"
            or len(failures) != 1
            or failures[0].failure_kind is not FailureKind.SERVICE
        ):
            raise RuntimeError("second recovery v3 retry lineage changed in the ledger")
        return parent.job_id, parent_attempt_id

    def _validate_second_recovery_request_binding(self) -> None:
        if self.second_recovery_overlay is None:
            return
        policy = FallbackModelPolicy.load(self.root / "configs/study/fallback_model.json")
        retry_call = fallback_pilot_calls(policy)[0]
        if retry_call.call_id != SECOND_RECOVERY_RETRY_CALL_ID:
            raise RuntimeError("fallback plan no longer begins with the authorized retry")
        request = build_fallback_acceptance_request(
            root=self.root,
            call=retry_call,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            legacy_provenance_bridge=self.legacy_provenance_bridge,
        )
        decoder = self.second_recovery_overlay.get("decoder_compatibility")
        evidence_bridge = self.second_recovery_overlay.get("evidence_provenance_bridge")
        expected_evidence_bridge = _second_recovery_evidence_bridge_binding(
            root=self.root,
            bridge=self.legacy_provenance_bridge,
            retry_request=request,
        )
        if (
            not isinstance(decoder, Mapping)
            or not isinstance(evidence_bridge, Mapping)
            or request.request_hash != decoder.get("retry_request_sha256")
            or canonical_sha256(request.output_schema)
            != decoder.get("compatible_decoder_schema_sha256")
            or dict(evidence_bridge) != expected_evidence_bridge
        ):
            raise RuntimeError("second recovery retry request differs from its authorization")

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
        self._validate_second_recovery_request_binding()
        self._second_recovery_retry_lineage()
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
        service_checkpoint = self.checkpoint_path.with_name(self.checkpoint_path.name + ".service")
        try:
            self.service.start(
                session_id=self.run_id,
                event_id=f"{self.run_id}-service-start-001",
                watchdog_seconds=self.service_start_watchdog_seconds,
                remaining_required_seconds=self._remaining_mandatory_forecast_seconds(
                    state,
                    exclude_upcoming_base_service_start=(self.retry_amendment is None),
                ),
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

    @property
    def _service_checkpoint_path(self) -> Path:
        return self.checkpoint_path.with_name(self.checkpoint_path.name + ".service")

    @property
    def _service_event_id(self) -> str:
        return f"{self.run_id}-service-start-001"

    def _validate_terminal_service_accounting(self) -> GpuServiceSession | None:
        """Require exact terminal accounting when this run opened a service journal."""

        unresolved_allocations = self.ledger.unresolved_gpu_allocations()
        if unresolved_allocations:
            raise RuntimeError(
                "fallback cleanup left unresolved GPU allocation accounting: "
                + ", ".join(item.allocation_id for item in unresolved_allocations)
            )
        latest = self.ledger.latest_gpu_service_journal(self._service_event_id)
        record = self.ledger.get_gpu_service_session(self._service_event_id)
        if latest is None:
            if record is not None:
                raise RuntimeError("fallback service session exists without its durable journal")
            matching_events = tuple(
                event
                for event in self.ledger.gpu_events()
                if event.event_id == self._service_event_id
            )
            if matching_events:
                lease = self.service.read_authoritative_service_lease()
                event = matching_events[0]
                try:
                    event_details = json.loads(event.details_json)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "PID-less pre-exec allocation has invalid recovery details"
                    ) from exc
                if (
                    len(matching_events) != 1
                    or event.event_kind is not GpuEventKind.FAILURE
                    or event.succeeded is not False
                    or not isinstance(event_details, Mapping)
                    or event_details.get("recovered_from_open_journal") is not True
                    or event_details.get("intended_event_kind")
                    != GpuEventKind.GPU_SESSION_START.value
                    or lease is None
                    or lease.get("lease_state") != "stopped_verified"
                    or lease.get("launch_protocol") != DURABLE_EXEC_GATE_PROTOCOL
                    or lease.get("session_id") != self.run_id
                    or lease.get("accounting_session_id") != self._service_event_id
                    or lease.get("configuration_hash")
                    != self.service.configuration.configuration_hash
                    or lease.get("service_pid") is not None
                    or lease.get("process_start_ticks") is not None
                    or lease.get("launch_supervisor_command_sha256") is not None
                    or self.ledger.unresolved_gpu_allocations()
                ):
                    raise RuntimeError(
                        "fallback service event exists without terminal session accounting"
                    )
            # The runtime opens its service journal before spawning vLLM.  No
            # journal plus an exact stopped pre-exec gate proves that target exec
            # was never released. Its recovered allocation event conservatively
            # accounts the controller-loss interval without inventing a service
            # session for a model process that never existed.
            return None
        if (
            latest.service_session_id != self._service_event_id
            or latest.session_id != self.run_id
            or latest.configuration_hash != self.service.configuration.configuration_hash
        ):
            raise RuntimeError("fallback service journal identity changed during cleanup")
        if (
            latest.state not in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
            or record is None
            or record.service_session_id != self._service_event_id
            or record.session_id != self.run_id
        ):
            raise RuntimeError("fallback service cleanup lacks terminal GPU accounting")
        if any(
            item.service_session_id == self._service_event_id
            for item in self.ledger.unresolved_gpu_service_journals()
        ):
            raise RuntimeError("fallback service journal remains unresolved after cleanup")
        return record

    def _stop_or_reconcile_service(self) -> tuple[ServiceUptime | None, bool]:
        """Stop/adopt one exact service or conservatively close its stale journal."""

        self._last_service_adoption_failure_type = None
        adopted = False
        uptime: ServiceUptime | None = None
        if self.service.state is not ServiceState.STOPPED:
            uptime = self.service.shutdown()
        else:
            try:
                # The atomic lease is the authoritative crash boundary and can
                # identify token-bound descendants after a service leader dies.
                # A checkpoint is only a compatibility fallback once the lease
                # has proved that no detached process or group remains.
                resume_lease = getattr(self.service, "resume_live_service_lease", None)
                if callable(resume_lease):
                    adopted = bool(
                        resume_lease(
                            expected_session_id=self.run_id,
                            expected_event_id=self._service_event_id,
                            watchdog_seconds=self.service_start_watchdog_seconds,
                            cleanup_only=True,
                        )
                    )
                if not adopted and self._service_checkpoint_path.exists():
                    adopted = self.service.resume_from_checkpoint(
                        self._service_checkpoint_path,
                        allow_same_controller_cleanup=True,
                    )
                if adopted:
                    uptime = self.service.shutdown()
                else:
                    recover_lease = getattr(self.service, "recover_stale_service_lease", None)
                    if not callable(recover_lease):
                        raise RuntimeError(
                            "fallback cleanup cannot prove a detached service is absent"
                        )
                    recover_lease()
            except BaseException as adoption_error:
                self._last_service_adoption_failure_type = type(adoption_error).__name__
                try:
                    emergency_uptime = self.service.shutdown()
                    if emergency_uptime is not None:
                        uptime = emergency_uptime
                except BaseException as shutdown_error:
                    raise RuntimeError(
                        "fallback service adoption failed and physical shutdown was not verified"
                    ) from shutdown_error
                if emergency_uptime is None:
                    raise adoption_error
        terminal = self._validate_terminal_service_accounting()
        if terminal is not None and uptime is None:
            self._last_shutdown_uptime = ServiceUptime(
                session_id=terminal.session_id,
                started_at=datetime.fromisoformat(terminal.started_at),
                ended_at=datetime.fromisoformat(terminal.ended_at),
                service_seconds=terminal.service_seconds,
                allocated_event_seconds=terminal.service_seconds - terminal.overhead_seconds,
            )
            uptime = self._last_shutdown_uptime
        elif uptime is not None:
            self._last_shutdown_uptime = uptime
        if self.service.state is not ServiceState.STOPPED:
            raise RuntimeError("fallback service did not enter the verified stopped state")
        return uptime, adopted

    def recover_controller_restart_preparation(self) -> dict[str, object]:
        """Complete a crash-interrupted first handoff without a second model load."""

        registration = self._development_registration(required=True)
        assert registration is not None
        calls = fallback_pilot_calls(
            FallbackModelPolicy.load(self.root / "configs/study/fallback_model.json")
        )
        plan_hash = cast(str, fallback_plan_manifest(self.root)["manifest_sha256"])
        identity = self._execution_identity(plan_hash)
        execution_hash = canonical_sha256(identity)
        state = self._load_state(execution_hash, calls)
        if (
            state["service_start_attempted"] is not True
            or state["controller_handoff_complete"] is True
            or state["completed_call_ids"]
        ):
            raise RuntimeError("fallback preparation recovery has no interrupted handoff")
        adopted = False
        try:
            if self._service_checkpoint_path.exists():
                adopted = self.service.resume_from_checkpoint(self._service_checkpoint_path)
            if not adopted:
                adopted = self.service.resume_live_service_lease(
                    expected_session_id=self.run_id,
                    expected_event_id=self._service_event_id,
                    watchdog_seconds=self.service_start_watchdog_seconds,
                )
            if not adopted or self.service.state is not ServiceState.READY:
                self.service.recover_stale_service_lease()
                self._validate_terminal_service_accounting()
                self._mark_terminal(state, "controller-restart-prepare-interrupted")
                raise RuntimeError(
                    "interrupted fallback preparation terminated before a resumable handoff"
                )
            self.service.write_resume_checkpoint(self._service_checkpoint_path)
            checkpoint = _load_object(self._service_checkpoint_path)
            if checkpoint.get("controller_pid") != os.getpid():
                raise RuntimeError("recovered service checkpoint did not bind this controller")
            state["handoff_service_pid"] = self.service.pid
            state["controller_handoff_complete"] = True
            state["live_allocated_seconds_at_controller_handoff"] = max(
                self.ledger.gpu_summary().total_allocated_seconds,
                float(getattr(self.service, "actual_allocated_service_seconds", 0.0)),
            )
            self._save(state)
            self.service.detach_for_controller_restart(self._service_checkpoint_path)
        except BaseException:
            if adopted and self.service.state is not ServiceState.STOPPED:
                self.service.shutdown()
            raise
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_controller_restart_handoff",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "execution_identity": identity,
            "fallback_plan_manifest_sha256": plan_hash,
            "controller_stage": "prepare_recovered",
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
        if state.get("run_id") != self.run_id or state.get("execution_hash") != execution_hash:
            raise ValueError("orphan cleanup checkpoint identifies another execution")
        uptime, adopted = self._stop_or_reconcile_service()
        interrupted_call_id = state.get("active_call_id")
        if interrupted_call_id is not None:
            self._terminalize_interrupted_call_lifecycle(state)
        state["orphan_cleanup_completed"] = True
        state["service_adoption_failure_type"] = self._last_service_adoption_failure_type
        state["active_call_id"] = None
        state["active_attempt"] = None
        state["failed_call_id"] = (
            state.get("failed_call_id") or interrupted_call_id or "orphan-cleanup-requested"
        )
        self._save(state)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_orphan_cleanup_result",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "checkpoint_service_adopted": adopted,
            "checkpoint_service_adoption_failed": (
                self._last_service_adoption_failure_type is not None
            ),
            "checkpoint_service_adoption_failure_type": (self._last_service_adoption_failure_type),
            "physical_shutdown_verified": (
                self.service.state is ServiceState.STOPPED
                and not self.ledger.unresolved_gpu_allocations()
                and not self.ledger.unresolved_gpu_service_journals()
            ),
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

        state: dict[str, object] | None = None
        if self.checkpoint_path.exists():
            state = _load_object(self.checkpoint_path)
        service_may_have_started = (
            state is not None
            and state.get("service_start_attempted") is True
            and state.get("orphan_cleanup_completed") is not True
        )
        if service_may_have_started:
            uptime, _adopted = self._stop_or_reconcile_service()
            assert state is not None
            interrupted_call_id = state.get("active_call_id")
            if interrupted_call_id is not None:
                self._terminalize_interrupted_call_lifecycle(state)
            state["orphan_cleanup_completed"] = True
            state["service_adoption_failure_type"] = self._last_service_adoption_failure_type
            state["active_call_id"] = None
            state["active_attempt"] = None
            state["failed_call_id"] = (
                state.get("failed_call_id") or interrupted_call_id or "failure-cleanup"
            )
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

    @staticmethod
    def _storage_sample_timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _development_storage_budget(self) -> StorageBudget:
        """Return the sampled budget after enforcing the registered study envelope."""

        storage = self.resource_sampler.storage
        if storage is None:
            raise RuntimeError("development continuation lacks its storage preflight")
        budget = storage.budget
        limits = ResourceLimits.load(self.root / "configs/study/resource_limits.json")
        if (
            budget.total_allocation_bytes > limits.maximum_project_allocation_bytes
            or budget.max_occupied_bytes > limits.maximum_project_occupied_bytes
            or budget.min_headroom_bytes < limits.minimum_storage_headroom_bytes
        ):
            raise RuntimeError(
                "development storage preflight is more permissive than registered limits"
            )
        return budget

    def _verify_development_storage_admissions(
        self,
        state: Mapping[str, object],
        *,
        required: bool,
    ) -> tuple[DevelopmentStorageAdmissionReceipt, ...]:
        raw_receipts = state.get("development_storage_admissions")
        if not isinstance(raw_receipts, list):
            raise RuntimeError("development storage-admission checkpoint inventory is invalid")
        receipts = tuple(
            DevelopmentStorageAdmissionReceipt.model_validate(item) for item in raw_receipts
        )
        ledger_phase = f"phase3_development:{self.run_id}"
        rows = tuple(
            row
            for row in self.ledger.storage_samples_with_phase_prefix(ledger_phase)
            if row.phase == ledger_phase
        )
        if required and not receipts:
            raise RuntimeError("development continuation lacks its Phase-3 storage admission")
        if not receipts:
            if state.get("development_storage_admission_hash") is not None or rows:
                raise RuntimeError(
                    "development storage-admission ledger/hash lacks its checkpoint receipt"
                )
            return ()

        plan_path = self.root / "configs/study/storage_phase_allocations.json"
        plan = StorageAllocationPlan.load(plan_path)
        reservation = plan.reservation_for("phase_3")
        reservation_arguments = reservation.preflight_arguments()
        plan_file_hash = _file_sha256(plan_path)
        sampled_budget = self._development_storage_budget()
        if tuple(receipt.storage_sample_id for receipt in receipts) != tuple(
            row.sample_id for row in rows
        ):
            raise RuntimeError(
                "development storage-admission checkpoint is not the complete ordered ledger "
                "history"
            )
        previous_sampled_at: datetime | None = None
        for index, (receipt, row) in enumerate(zip(receipts, rows, strict=True), start=1):
            if (
                receipt.admission_id != f"{self.run_id}-phase3-storage-{index:03d}"
                or receipt.owner_run_id != self.run_id
                or receipt.ledger_phase != ledger_phase
                or receipt.storage_plan_file_sha256 != plan_file_hash
                or any(
                    getattr(receipt, name) != value
                    for name, value in reservation_arguments.items()
                )
                or receipt.sampled_total_allocation_bytes
                != sampled_budget.total_allocation_bytes
                or receipt.sampled_max_occupied_bytes != sampled_budget.max_occupied_bytes
                or receipt.sampled_min_headroom_bytes != sampled_budget.min_headroom_bytes
                or receipt.projected_allocation_free_bytes
                != receipt.sampled_total_allocation_bytes - receipt.projected_occupied_bytes
                or row.phase != receipt.ledger_phase
                or row.sampled_at != self._storage_sample_timestamp(receipt.sampled_at)
                or row.current_occupied_bytes != receipt.current_occupied_bytes
                or row.additional_reserved_bytes != receipt.additional_reserved_bytes
                or row.projected_occupied_bytes != receipt.projected_occupied_bytes
                or row.filesystem_free_bytes != receipt.filesystem_free_bytes
                or row.effective_projected_headroom_bytes
                != receipt.effective_projected_headroom_bytes
                or row.allowed != receipt.allowed
                or row.violations != receipt.violations
            ):
                raise RuntimeError(
                    "development storage admission differs from its registered plan or ledger row"
                )
            if previous_sampled_at is not None and receipt.sampled_at <= previous_sampled_at:
                raise RuntimeError("development storage-admission history is not chronological")
            previous_sampled_at = receipt.sampled_at
        if state.get("development_storage_admission_hash") != receipts[-1].content_hash:
            raise RuntimeError("development storage-admission checkpoint points to another receipt")
        if required and not receipts[-1].allowed:
            raise RuntimeError(
                "completed development continuation cites a rejected storage admission"
            )
        return receipts

    def _reconcile_development_storage_admissions(
        self,
        state: dict[str, object],
    ) -> tuple[DevelopmentStorageAdmissionReceipt, ...]:
        """Recover only a ledger-first checkpoint gap, preserving every observation."""

        raw_receipts = state.get("development_storage_admissions")
        if raw_receipts is None:
            raw_receipts = []
            state["development_storage_admissions"] = raw_receipts
        if not isinstance(raw_receipts, list):
            raise RuntimeError("development storage-admission checkpoint inventory is invalid")
        receipts = tuple(
            DevelopmentStorageAdmissionReceipt.model_validate(item) for item in raw_receipts
        )
        ledger_phase = f"phase3_development:{self.run_id}"
        rows = tuple(
            row
            for row in self.ledger.storage_samples_with_phase_prefix(ledger_phase)
            if row.phase == ledger_phase
        )
        checkpoint_sample_ids = tuple(receipt.storage_sample_id for receipt in receipts)
        ledger_sample_ids = tuple(row.sample_id for row in rows)
        if (
            len(checkpoint_sample_ids) > len(ledger_sample_ids)
            or checkpoint_sample_ids != ledger_sample_ids[: len(checkpoint_sample_ids)]
        ):
            raise RuntimeError(
                "development storage-admission checkpoint is not an ordered ledger prefix"
            )

        if len(receipts) < len(rows):
            plan_path = self.root / "configs/study/storage_phase_allocations.json"
            reservation = StorageAllocationPlan.load(plan_path).reservation_for("phase_3")
            sampled_budget = self._development_storage_budget()
            plan_file_hash = _file_sha256(plan_path)
            for index, row in enumerate(rows[len(receipts) :], start=len(receipts) + 1):
                sampled_at = datetime.fromisoformat(row.sampled_at.replace("Z", "+00:00"))
                recovered = DevelopmentStorageAdmissionReceipt(
                    admission_id=f"{self.run_id}-phase3-storage-{index:03d}",
                    owner_run_id=self.run_id,
                    ledger_phase=ledger_phase,
                    storage_plan_file_sha256=plan_file_hash,
                    storage_sample_id=row.sample_id,
                    sampled_total_allocation_bytes=(
                        sampled_budget.total_allocation_bytes
                    ),
                    sampled_max_occupied_bytes=sampled_budget.max_occupied_bytes,
                    sampled_min_headroom_bytes=sampled_budget.min_headroom_bytes,
                    **reservation.preflight_arguments(),
                    current_occupied_bytes=row.current_occupied_bytes,
                    additional_reserved_bytes=row.additional_reserved_bytes,
                    projected_occupied_bytes=row.projected_occupied_bytes,
                    filesystem_free_bytes=row.filesystem_free_bytes,
                    projected_allocation_free_bytes=(
                        sampled_budget.total_allocation_bytes - row.projected_occupied_bytes
                    ),
                    projected_filesystem_free_bytes=(
                        row.filesystem_free_bytes - row.additional_reserved_bytes
                    ),
                    effective_projected_headroom_bytes=(
                        row.effective_projected_headroom_bytes
                    ),
                    allowed=row.allowed,
                    violations=row.violations,
                    sampled_at=sampled_at,
                )
                raw_receipts.append(recovered.model_dump(mode="json"))
            state["development_storage_admission_hash"] = cast(
                Mapping[str, object], raw_receipts[-1]
            )["content_hash"]
            self._save(state)
        return self._verify_development_storage_admissions(state, required=False)

    def _record_development_storage_admission(
        self,
        state: dict[str, object],
    ) -> tuple[DevelopmentStorageAdmissionReceipt, StorageReport]:
        prior_receipts = self._reconcile_development_storage_admissions(state)
        plan_path = self.root / "configs/study/storage_phase_allocations.json"
        plan = StorageAllocationPlan.load(plan_path)
        reservation = plan.reservation_for("phase_3")
        storage = self.resource_sampler.storage
        if storage is None:
            raise RuntimeError("development continuation lacks its storage preflight")
        sampled_budget = self._development_storage_budget()
        report = storage.check(**reservation.preflight_arguments())
        if report.budget != sampled_budget:
            raise RuntimeError("development storage report changed its sampled budget")
        sampled_at = datetime.now(UTC)
        if prior_receipts and sampled_at <= prior_receipts[-1].sampled_at:
            sampled_at = prior_receipts[-1].sampled_at + timedelta(microseconds=1)
        ledger_phase = f"phase3_development:{self.run_id}"
        sample_id = self.ledger.record_storage_sample(
            report,
            phase=ledger_phase,
            sampled_at=sampled_at,
        )
        receipt = DevelopmentStorageAdmissionReceipt(
            admission_id=(
                f"{self.run_id}-phase3-storage-"
                f"{len(self.ledger.storage_samples_with_phase_prefix(ledger_phase)):03d}"
            ),
            owner_run_id=self.run_id,
            ledger_phase=ledger_phase,
            storage_plan_file_sha256=_file_sha256(plan_path),
            storage_sample_id=sample_id,
            sampled_total_allocation_bytes=report.budget.total_allocation_bytes,
            sampled_max_occupied_bytes=report.budget.max_occupied_bytes,
            sampled_min_headroom_bytes=report.budget.min_headroom_bytes,
            **reservation.preflight_arguments(),
            current_occupied_bytes=report.current_occupied_bytes,
            additional_reserved_bytes=report.additional_reserved_bytes,
            projected_occupied_bytes=report.projected_occupied_bytes,
            filesystem_free_bytes=report.filesystem_free_bytes,
            projected_allocation_free_bytes=report.projected_allocation_free_bytes,
            projected_filesystem_free_bytes=report.projected_filesystem_free_bytes,
            effective_projected_headroom_bytes=report.effective_projected_headroom_bytes,
            allowed=report.allowed,
            violations=report.violations,
            sampled_at=sampled_at,
        )
        raw_receipts = state.get("development_storage_admissions")
        if not isinstance(raw_receipts, list):
            raise RuntimeError("development storage-admission checkpoint inventory is invalid")
        raw_receipts.append(receipt.model_dump(mode="json"))
        state["development_storage_admission_hash"] = receipt.content_hash
        self._save(state)
        self._verify_development_storage_admissions(state, required=False)
        return receipt, report

    def _resume_completed_development_continuation(
        self,
        *,
        state: Mapping[str, object],
        execution_hash: str,
        registration: DevelopmentAdopterRegistration,
    ) -> DevelopmentContinuationReceipt:
        storage_receipts = self._verify_development_storage_admissions(state, required=True)
        raw_result = state.get("development_execution_result")
        raw_receipt = state.get("development_continuation_receipt")
        if raw_result is None or raw_receipt is None:
            raise RuntimeError("completed development continuation lacks immutable result lineage")
        result = DevelopmentExecutionResult.model_validate(raw_result)
        receipt = DevelopmentContinuationReceipt.model_validate(raw_receipt)
        if (
            receipt.fallback_execution_hash != execution_hash
            or receipt.adopter_registration_hash != registration.content_hash
            or receipt.development_result_hash != result.content_hash
            or receipt.development_storage_admission_hash
            != storage_receipts[-1].content_hash
            or receipt.handoff_hash != state.get("development_handoff_hash")
        ):
            raise RuntimeError("completed development continuation replay changed its lineage")
        return receipt

    @staticmethod
    def _call_role(call: FallbackCallSpec) -> ModelCallRole:
        return {
            ConditionName.C1_LLM_PRE: ModelCallRole.PREBUILD,
            ConditionName.C2_LLM_QUERY: ModelCallRole.QUERY_TIME,
            ConditionName.A_FIXED_SELECT: ModelCallRole.FIXED_SELECT,
        }[call.condition]

    def _job_identity(
        self,
        *,
        call: FallbackCallSpec,
        execution_hash: str,
    ) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "call_id": call.call_id,
            "plan_hash": execution_hash,
            "condition": call.condition.value,
            "lifecycle_kind": (
                "query_blind_prebuild"
                if call.condition is ConditionName.C1_LLM_PRE
                else "query_time_generation"
            ),
        }

    def _advance_one(
        self,
        *,
        job_id: str,
        call: FallbackCallSpec,
        state: JobState,
        occurred_at: datetime,
    ) -> None:
        observed = tuple(item.to_state for item in self.ledger.transitions(job_id))
        repair_path = JobState.REPAIRED in observed or state is JobState.REPAIRED
        query_blind = call.condition is ConditionName.C1_LLM_PRE
        canonical_path = (
            (JobState.PLANNED, JobState.PREQUERY_SEALED)
            + (() if query_blind else (JobState.QUERY_REVEALED,))
            + (JobState.GENERATED,)
            + ((JobState.REPAIRED,) if repair_path else ())
            + (JobState.VALIDATED, JobState.FINALIZED)
        )
        if observed != canonical_path[: len(observed)]:
            raise RuntimeError("fallback job has a noncanonical lifecycle prefix")
        try:
            target_index = canonical_path.index(state)
        except ValueError as error:
            raise RuntimeError("fallback job requested an unsupported lifecycle state") from error
        if target_index < len(observed):
            return
        if target_index != len(observed):
            raise RuntimeError("fallback job lifecycle would skip a required state")
        self.ledger.transition_job(
            job_id,
            state,
            occurred_at=occurred_at,
            frozen_legacy_fallback_c1_retry=(
                query_blind
                and state is JobState.GENERATED
                and self.second_recovery_overlay is not None
            ),
        )

    def _prepare_job_lifecycle(
        self,
        *,
        job_id: str,
        call: FallbackCallSpec,
        prequery_sealed_at: datetime,
    ) -> None:
        self._advance_one(
            job_id=job_id,
            call=call,
            state=JobState.PREQUERY_SEALED,
            occurred_at=prequery_sealed_at,
        )
        if call.condition is ConditionName.C1_LLM_PRE:
            return
        query_revealed_at = max(
            datetime.now(UTC),
            prequery_sealed_at + timedelta(microseconds=1),
        )
        self._advance_one(
            job_id=job_id,
            call=call,
            state=JobState.QUERY_REVEALED,
            occurred_at=query_revealed_at,
        )

    def _record_structural_validation(
        self,
        *,
        job_id: str,
        attempt_id: str,
        input_artifact_hash: str,
        validator_manifest_hash: str,
        accepted: bool,
        diagnostics_artifact_hash: str | None = None,
        parent_validation_id: str | None = None,
        repair: bool = False,
        created_at: datetime | None = None,
    ) -> str:
        validation_id = f"{attempt_id}-validation"
        self.ledger.record_validation(
            validation_id=validation_id,
            job_id=job_id,
            attempt_id=attempt_id,
            input_artifact_hash=input_artifact_hash,
            validator_manifest_hash=validator_manifest_hash,
            validation_status=(
                ValidationStatus.ACCEPTED if accepted else ValidationStatus.REJECTED
            ),
            evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
            temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
            commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
            semantic_assessment_scope=(
                SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            ),
            diagnostics_artifact_hash=diagnostics_artifact_hash,
            parent_validation_id=parent_validation_id,
            repair_attempt_id=attempt_id if repair else None,
            created_at=created_at,
        )
        return validation_id

    def _finish_job_lifecycle(
        self,
        *,
        job_id: str,
        call: FallbackCallSpec,
        occurred_at: datetime,
    ) -> None:
        self._advance_one(
            job_id=job_id,
            call=call,
            state=JobState.VALIDATED,
            occurred_at=occurred_at,
        )
        self._advance_one(
            job_id=job_id,
            call=call,
            state=JobState.FINALIZED,
            occurred_at=occurred_at,
        )

    def _reserve(
        self,
        state: dict[str, object],
        *,
        call_id: str,
        reserve_call_class: str,
        watchdog_seconds: int,
        job_id: str,
        attempt_id: str,
        attempt_kind: AttemptKind,
        parent_attempt_id: str | None,
        input_hash: str,
        config_hash: str,
        seed: int,
        attempt_created_at: datetime,
    ) -> dict[str, object]:
        existing_active_call = state.get("active_call_id")
        replacing_failed_base_with_repair = (
            isinstance(existing_active_call, str)
            and call_id == f"{existing_active_call}-repair-01"
            and state.get("repair_parent_call_id") == existing_active_call
        )
        if (
            (existing_active_call is not None or state.get("active_attempt") is not None)
            and not replacing_failed_base_with_repair
        ):
            raise RuntimeError("another fallback attempt is already active")
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
        state["active_attempt"] = {
            "call_id": call_id,
            "job_id": job_id,
            "attempt_id": attempt_id,
            "attempt_kind": attempt_kind.value,
            "parent_attempt_id": parent_attempt_id,
            "input_hash": input_hash,
            "config_hash": config_hash,
            "seed": seed,
            "created_at": attempt_created_at.isoformat(),
        }
        self._save(state)
        return receipt

    def _mark_terminal(self, state: dict[str, object], call_id: str) -> None:
        state["active_call_id"] = None
        state["active_attempt"] = None
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
        construction_unit_hash: str,
        validator_manifest_hash: str,
        parent_validation_id: str | None = None,
        repair: bool = False,
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
                construction_unit_hash=construction_unit_hash,
                served_context_count=1,
                prompt_tokens=0,
                completion_tokens=0,
                allocated_gpu_seconds=allocated,
                successful=False,
            )
        terminal_at = datetime.now(UTC)
        public_diagnostics = self.artifacts.put_bytes(
            (
                canonical_json(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "diagnostic_kind": "fallback_transport_failure",
                        "call_id": call.call_id,
                        "request_hash": request.request_hash,
                        "exception_type": type(exc).__name__,
                        "repair": repair,
                    }
                )
                + "\n"
            ).encode("utf-8"),
            media_type=(
                "application/vnd.story-projection."
                "fallback-failure-diagnostics+json"
            ),
            release_class=ReleaseClass.PUBLIC,
            created_at=terminal_at,
        )
        failure_details: dict[str, object] = {
            "exception_type": type(exc).__name__,
            "call_id": call.call_id,
        }
        restricted_diagnostics_artifact_hash = None
        transport_details = restricted_transport_failure_details(exc)
        if transport_details is not None:
            diagnostics = self.artifacts.put_bytes(
                (
                    canonical_json(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "diagnostic_kind": "vllm_http_transport",
                            "call_id": call.call_id,
                            "request_hash": request.request_hash,
                            "exception_type": type(exc).__name__,
                            **transport_details,
                        }
                    )
                    + "\n"
                ).encode("utf-8"),
                media_type=(
                    "application/vnd.story-projection.restricted-transport-diagnostics+json"
                ),
                release_class=ReleaseClass.RESTRICTED,
            )
            restricted_diagnostics_artifact_hash = diagnostics.content_hash
            failure_details["restricted_diagnostics_artifact_hash"] = (
                restricted_diagnostics_artifact_hash
            )
        self._advance_one(
            job_id=job_id,
            call=call,
            state=JobState.REPAIRED if repair else JobState.GENERATED,
            occurred_at=terminal_at,
        )
        self._record_structural_validation(
            job_id=job_id,
            attempt_id=attempt_id,
            input_artifact_hash=public_diagnostics.content_hash,
            validator_manifest_hash=validator_manifest_hash,
            accepted=False,
            diagnostics_artifact_hash=public_diagnostics.content_hash,
            parent_validation_id=parent_validation_id,
            repair=repair,
            created_at=terminal_at,
        )
        self.ledger.record_failure(
            attempt_id=attempt_id,
            failure_kind=_failure_kind(exc),
            message="Fallback micro-pilot model call failed",
            details=failure_details,
            artifact_hash=public_diagnostics.content_hash,
            occurred_at=terminal_at,
        )
        self._finish_job_lifecycle(
            job_id=job_id,
            call=call,
            occurred_at=terminal_at,
        )
        return allocated

    def _terminalize_interrupted_call_lifecycle(
        self,
        state: dict[str, object],
    ) -> dict[str, object] | None:
        """Append the terminal ledger lineage for a controller-interrupted call.

        The active-attempt intent is checkpointed before its attempt row, closing
        the process-death gap at that boundary. Cleanup and guardian processes
        can therefore recover the exact attempt after controller death. A
        deterministic receipt is checkpointed before any new CAS or ledger
        write, making a second terminalizer an exact replay. A durable accepted
        validation is preserved, but the missing scientific result commit is
        still recorded as an ITT interruption.
        """

        active_call_id = state.get("active_call_id")
        if active_call_id is None:
            return None
        if not isinstance(active_call_id, str) or not active_call_id:
            raise ValueError("fallback checkpoint active call identifier is invalid")

        repair_suffix = "-repair-01"
        repair = active_call_id.endswith(repair_suffix)
        base_call_id = (
            active_call_id.removesuffix(repair_suffix) if repair else active_call_id
        )
        policy = FallbackModelPolicy.load(self.root / "configs/study/fallback_model.json")
        call_by_id = {call.call_id: call for call in fallback_pilot_calls(policy)}
        try:
            call = call_by_id[base_call_id]
        except KeyError as error:
            raise ValueError(
                "fallback checkpoint active call is outside the registered pilot"
            ) from error
        if repair and state.get("repair_parent_call_id") != base_call_id:
            raise ValueError("fallback repair checkpoint lacks its exact parent call")

        execution_hash = state.get("execution_hash")
        if not isinstance(execution_hash, str) or len(execution_hash) != 64:
            raise ValueError("fallback checkpoint execution hash is invalid")
        second_recovery_retry = (
            self.second_recovery_overlay is not None
            and call.call_id == SECOND_RECOVERY_RETRY_CALL_ID
        )
        if second_recovery_retry:
            lineage = self._second_recovery_retry_lineage()
            if lineage is None:
                raise RuntimeError("interrupted fallback retry lacks its frozen v3 job")
            expected_job_id = lineage[0]
            expected_parent_attempt_id = (
                f"{SECOND_RECOVERY_V3_RUN_ID}-{SECOND_RECOVERY_RETRY_CALL_ID}-attempt"
            )
            expected_base_kind = AttemptKind.RETRY
        else:
            expected_job_id = canonical_sha256(
                self._job_identity(call=call, execution_hash=execution_hash)
            )
            expected_parent_attempt_id = None
            expected_base_kind = AttemptKind.BASE
        if repair:
            expected_parent_attempt_id = f"{self.run_id}-{base_call_id}-attempt"
            expected_attempt_kind = AttemptKind.REPAIR
        else:
            expected_attempt_kind = expected_base_kind

        attempt_id = f"{self.run_id}-{active_call_id}-attempt"
        active_attempt_value = state.get("active_attempt")
        active_attempt = (
            dict(active_attempt_value)
            if isinstance(active_attempt_value, Mapping)
            else None
        )
        if active_attempt_value is not None and active_attempt is None:
            raise ValueError("fallback checkpoint active-attempt intent is invalid")
        try:
            attempt = self.ledger.attempt_lineage(attempt_id)[-1]
        except KeyError:
            if active_attempt is None:
                raise RuntimeError(
                    "interrupted fallback call lacks both an attempt and durable intent"
                ) from None
            if (
                active_attempt.get("call_id") != active_call_id
                or active_attempt.get("attempt_id") != attempt_id
                or active_attempt.get("job_id") != expected_job_id
                or active_attempt.get("attempt_kind") != expected_attempt_kind.value
                or active_attempt.get("parent_attempt_id") != expected_parent_attempt_id
            ):
                raise ValueError(
                    "fallback checkpoint attempt intent changed identity"
                ) from None
            attempt = self.ledger.record_attempt(
                attempt_id=attempt_id,
                job_id=cast(str, active_attempt["job_id"]),
                attempt_kind=AttemptKind(cast(str, active_attempt["attempt_kind"])),
                parent_attempt_id=cast(
                    str | None,
                    active_attempt.get("parent_attempt_id"),
                ),
                input_hash=cast(str, active_attempt["input_hash"]),
                config_hash=cast(str, active_attempt["config_hash"]),
                seed=cast(int, active_attempt["seed"]),
                created_at=cast(str, active_attempt["created_at"]),
            )
        if attempt.attempt_id != attempt_id:
            raise RuntimeError("interrupted fallback attempt lookup changed identity")
        if active_attempt is not None:
            expected_intent = {
                "call_id": active_call_id,
                "job_id": attempt.job_id,
                "attempt_id": attempt.attempt_id,
                "attempt_kind": attempt.attempt_kind.value,
                "parent_attempt_id": attempt.parent_attempt_id,
                "input_hash": attempt.input_hash,
                "config_hash": attempt.config_hash,
                "seed": attempt.seed,
            }
            observed_intent = {
                key: value for key, value in active_attempt.items() if key != "created_at"
            }
            try:
                intent_created_at = datetime.fromisoformat(
                    cast(str, active_attempt["created_at"])
                )
                attempt_created_at = datetime.fromisoformat(attempt.created_at)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("fallback checkpoint attempt intent time is invalid") from error
            if (
                observed_intent != expected_intent
                or intent_created_at != attempt_created_at
            ):
                raise ValueError(
                    "fallback checkpoint attempt intent differs from its ledger row"
                )
        if attempt.job_id != expected_job_id:
            raise ValueError("interrupted fallback attempt belongs to another job")
        job = self.ledger.get_job(attempt.job_id)

        if second_recovery_retry:
            if lineage[0] != job.job_id:
                raise RuntimeError("interrupted fallback retry changed its frozen v3 job")
        else:
            try:
                job_identity = json.loads(job.identity_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("interrupted fallback job identity is invalid") from error
            if job_identity != self._job_identity(call=call, execution_hash=execution_hash):
                raise ValueError("interrupted fallback attempt belongs to another job")
        if (
            attempt.attempt_kind is not expected_attempt_kind
            or attempt.parent_attempt_id != expected_parent_attempt_id
        ):
            raise ValueError("interrupted fallback attempt lineage changed")

        validation_id = f"{attempt_id}-validation"
        try:
            validation = self.ledger.get_validation(validation_id)
        except KeyError:
            validation = None
        try:
            model_call = self.ledger.get_model_call(attempt_id.removesuffix("-attempt"))
        except KeyError:
            model_call = None
        own_failures = tuple(
            failure
            for failure in self.ledger.failures_for_lineage(attempt_id)
            if failure.attempt_id == attempt_id
        )
        if len(own_failures) > 1:
            raise RuntimeError("interrupted fallback attempt has multiple failures")
        accepted_validation = (
            validation is not None
            and validation.validation_status.value == ValidationStatus.ACCEPTED.value
        )
        if accepted_validation and (
            model_call is None
            or not model_call.successful
            or model_call.response_artifact_hash is None
            or validation is None
            or validation.input_artifact_hash != model_call.response_artifact_hash
        ):
            raise RuntimeError(
                "accepted interrupted fallback attempt lacks exact output lineage"
            )

        receipt_value = state.get("interrupted_call_terminalization")
        if receipt_value is None:
            timestamps = [
                datetime.fromisoformat(self.ledger.transitions(job.job_id)[-1].occurred_at),
                datetime.fromisoformat(attempt.created_at),
                datetime.now(UTC),
            ]
            if model_call is not None:
                timestamps.append(datetime.fromisoformat(model_call.created_at))
            if validation is not None:
                timestamps.append(datetime.fromisoformat(validation.created_at))
            if own_failures:
                timestamps.append(datetime.fromisoformat(own_failures[0].occurred_at))
            terminalized_at = max(timestamps) + timedelta(microseconds=1)
            diagnostic_payload: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "diagnostic_kind": "fallback_controller_interruption",
                "run_id": self.run_id,
                "call_id": active_call_id,
                "base_call_id": base_call_id,
                "condition": call.condition.value,
                "job_id": job.job_id,
                "attempt_id": attempt_id,
                "attempt_kind": attempt.attempt_kind.value,
                "request_hash": attempt.input_hash,
                "response_artifact_durable": (
                    model_call is not None
                    and model_call.response_artifact_hash is not None
                ),
                "accepted_validation_durable": accepted_validation,
                "scientific_result_pointer_durable": False,
                "terminalized_at": terminalized_at.isoformat(),
            }
            diagnostic_bytes = (canonical_json(diagnostic_payload) + "\n").encode("utf-8")
            receipt_payload: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "kind": "fallback_interrupted_call_terminalization",
                "run_id": self.run_id,
                "call_id": active_call_id,
                "base_call_id": base_call_id,
                "condition": call.condition.value,
                "job_id": job.job_id,
                "attempt_id": attempt_id,
                "attempt_kind": attempt.attempt_kind.value,
                "repair": repair,
                "accepted_validation_preserved": accepted_validation,
                "terminalized_at": terminalized_at.isoformat(),
                "diagnostic_artifact_hash": hashlib.sha256(diagnostic_bytes).hexdigest(),
            }
            receipt = {
                **receipt_payload,
                "content_hash": canonical_sha256(receipt_payload),
            }
            state["interrupted_call_terminalization"] = receipt
            self._save(state)
        else:
            if not isinstance(receipt_value, Mapping):
                raise ValueError("interrupted-call terminalization receipt is invalid")
            receipt = dict(receipt_value)
            receipt_payload = {
                key: value for key, value in receipt.items() if key != "content_hash"
            }
            if (
                receipt.get("content_hash") != canonical_sha256(receipt_payload)
                or receipt.get("run_id") != self.run_id
                or receipt.get("call_id") != active_call_id
                or receipt.get("base_call_id") != base_call_id
                or receipt.get("condition") != call.condition.value
                or receipt.get("job_id") != job.job_id
                or receipt.get("attempt_id") != attempt_id
                or receipt.get("attempt_kind") != attempt.attempt_kind.value
                or receipt.get("repair") is not repair
                or receipt.get("accepted_validation_preserved") is not accepted_validation
            ):
                raise ValueError("interrupted-call terminalization receipt changed identity")
            terminalized_at = datetime.fromisoformat(cast(str, receipt["terminalized_at"]))
            diagnostic_payload = {
                "schema_version": SCHEMA_VERSION,
                "diagnostic_kind": "fallback_controller_interruption",
                "run_id": self.run_id,
                "call_id": active_call_id,
                "base_call_id": base_call_id,
                "condition": call.condition.value,
                "job_id": job.job_id,
                "attempt_id": attempt_id,
                "attempt_kind": attempt.attempt_kind.value,
                "request_hash": attempt.input_hash,
                "response_artifact_durable": (
                    model_call is not None
                    and model_call.response_artifact_hash is not None
                ),
                "accepted_validation_durable": accepted_validation,
                "scientific_result_pointer_durable": False,
                "terminalized_at": terminalized_at.isoformat(),
            }
            diagnostic_bytes = (canonical_json(diagnostic_payload) + "\n").encode("utf-8")
            if receipt.get("diagnostic_artifact_hash") != hashlib.sha256(
                diagnostic_bytes
            ).hexdigest():
                raise ValueError("interrupted-call diagnostic binding changed")

        diagnostic_artifact = self.artifacts.put_bytes(
            diagnostic_bytes,
            media_type=(
                "application/vnd.story-projection."
                "fallback-controller-interruption+json"
            ),
            release_class=ReleaseClass.PUBLIC,
            created_at=terminalized_at,
        )
        if diagnostic_artifact.content_hash != receipt["diagnostic_artifact_hash"]:
            raise RuntimeError("interrupted-call diagnostic CAS hash changed")

        event_id = f"{self.run_id}-{active_call_id}-gpu"
        event = _event_for(self.ledger, event_id)
        if model_call is None and event is not None:
            if event.succeeded is None:
                raise RuntimeError("interrupted fallback GPU event is not terminal")
            self.ledger.record_model_call(
                model_call_id=attempt_id.removesuffix("-attempt"),
                job_id=job.job_id,
                attempt_id=attempt_id,
                gpu_event_id=event_id,
                backend=ModelBackend.VLLM_GPU,
                call_role=(ModelCallRole.REPAIR if repair else self._call_role(call)),
                retry_class=(RetryClass.SHORT if repair else call.retry_class),
                model_manifest_hash=self.service.configuration.configuration_hash,
                decoding_manifest_hash=attempt.config_hash,
                request_hash=attempt.input_hash,
                response_artifact_hash=None,
                construction_unit_hash=(
                    _repair_construction_unit_hash(
                        call,
                        cast(str, attempt.parent_attempt_id),
                    )
                    if repair
                    else _base_construction_unit_hash(call)
                ),
                served_context_count=1,
                prompt_tokens=0,
                completion_tokens=0,
                allocated_gpu_seconds=event.allocated_seconds,
                successful=event.succeeded,
                created_at=terminalized_at,
            )
        elif model_call is not None and (
            model_call.job_id != job.job_id or model_call.attempt_id != attempt_id
        ):
            raise ValueError("interrupted fallback model-call lineage changed")

        milestone = JobState.REPAIRED if repair else JobState.GENERATED
        self._advance_one(
            job_id=job.job_id,
            call=call,
            state=milestone,
            occurred_at=terminalized_at,
        )
        if validation is None:
            parent_validation_id = (
                f"{self.run_id}-{base_call_id}-attempt-validation" if repair else None
            )
            self._record_structural_validation(
                job_id=job.job_id,
                attempt_id=attempt_id,
                input_artifact_hash=(
                    model_call.response_artifact_hash
                    if model_call is not None
                    and model_call.response_artifact_hash is not None
                    else diagnostic_artifact.content_hash
                ),
                validator_manifest_hash=execution_hash,
                accepted=False,
                diagnostics_artifact_hash=diagnostic_artifact.content_hash,
                parent_validation_id=parent_validation_id,
                repair=repair,
                created_at=terminalized_at,
            )
        elif (
            validation.validation_status.value != ValidationStatus.REJECTED.value
            and validation.validation_status.value != ValidationStatus.ACCEPTED.value
        ):
            raise RuntimeError("interrupted fallback validation has an invalid status")
        if not own_failures:
            self.ledger.record_failure(
                attempt_id=attempt_id,
                failure_kind=FailureKind.INTERRUPTED,
                message="Fallback controller was interrupted before durable result commit",
                details={
                    "call_id": active_call_id,
                    "base_call_id": base_call_id,
                    "accepted_validation_preserved": accepted_validation,
                },
                artifact_hash=diagnostic_artifact.content_hash,
                occurred_at=terminalized_at,
            )
        self._finish_job_lifecycle(
            job_id=job.job_id,
            call=call,
            occurred_at=terminalized_at,
        )
        state["interrupted_call_lifecycle_finalized"] = True
        self._save(state)
        return cast(dict[str, object], receipt)

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
        accepted = cast(Mapping[str, Mapping[str, object]], state["accepted_outputs"])[call.call_id]
        base_request = build_fallback_acceptance_request(
            root=self.root,
            call=call,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            legacy_provenance_bridge=self.legacy_provenance_bridge,
        )
        base_call = call.acceptance_call()
        base_attempt_id = f"{self.run_id}-{call.call_id}-attempt"
        base_model_call = self.ledger.get_model_call(f"{self.run_id}-{call.call_id}")
        repaired = accepted.get("repaired") is True
        base_generation = self._validate_resumed_model_call(
            model_call_id=base_model_call.model_call_id,
            request=base_request,
            expected_attempt_id=base_attempt_id,
            expected_attempt_kind=(
                AttemptKind.RETRY
                if self.second_recovery_overlay is not None
                and call.call_id == SECOND_RECOVERY_RETRY_CALL_ID
                else AttemptKind.BASE
            ),
            expected_parent_attempt_id=(
                f"{SECOND_RECOVERY_V3_RUN_ID}-{SECOND_RECOVERY_RETRY_CALL_ID}-attempt"
                if self.second_recovery_overlay is not None
                and call.call_id == SECOND_RECOVERY_RETRY_CALL_ID
                else None
            ),
            expected_call_role=self._call_role(call),
            expected_retry_class=call.retry_class,
            expected_event_kind=GpuEventKind.FALLBACK_TEST,
            expected_reserve_class=call.reserve_call_class,
            expected_reservation_id=f"{self.run_id}:{call.call_id}",
            expected_construction_unit_hash=_base_construction_unit_hash(call),
            # ``successful`` records transport/model execution, not the later
            # structural assessment. A repair parent therefore remains a
            # successful call with a rejected validation record.
            expected_success=True,
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
                    legacy_provenance_bridge=self.legacy_provenance_bridge,
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
                legacy_provenance_bridge=self.legacy_provenance_bridge,
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
                legacy_provenance_bridge=self.legacy_provenance_bridge,
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
                    allocated_seconds=(repair_model_call.allocated_gpu_microseconds / 1_000_000),
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
                legacy_provenance_bridge=self.legacy_provenance_bridge,
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
        self._verify_completed_job_persistence(
            state=state,
            call=call,
            repaired=repaired,
        )
        return result, audit, tuple(timings)

    def _verify_completed_job_persistence(
        self,
        *,
        state: Mapping[str, object],
        call: FallbackCallSpec,
        repaired: bool,
    ) -> None:
        """Fail closed unless a resumed accepted output has exact terminal lineage."""

        base_attempt_id = f"{self.run_id}-{call.call_id}-attempt"
        base_call = self.ledger.get_model_call(f"{self.run_id}-{call.call_id}")
        job = self.ledger.get_job(base_call.job_id)
        legacy_retry = (
            self.second_recovery_overlay is not None
            and call.call_id == SECOND_RECOVERY_RETRY_CALL_ID
        )
        if legacy_retry:
            if not self.ledger.is_frozen_legacy_fallback_c1_retry_prebuild(job.job_id):
                raise RuntimeError("resumed fallback legacy C1 job classification changed")
        else:
            execution_hash = state.get("execution_hash")
            if not isinstance(execution_hash, str):
                raise RuntimeError("fallback checkpoint omitted its execution hash")
            expected_identity = self._job_identity(
                call=call,
                execution_hash=execution_hash,
            )
            if json.loads(job.identity_json) != expected_identity:
                raise RuntimeError("resumed fallback job identity changed")

        expected_states = (
            (JobState.PLANNED, JobState.PREQUERY_SEALED)
            + (
                ()
                if call.condition is ConditionName.C1_LLM_PRE
                else (JobState.QUERY_REVEALED,)
            )
            + (JobState.GENERATED,)
            + ((JobState.REPAIRED,) if repaired else ())
            + (JobState.VALIDATED, JobState.FINALIZED)
        )
        transitions = self.ledger.transitions(job.job_id)
        if tuple(item.to_state for item in transitions) != expected_states:
            raise RuntimeError("resumed fallback terminal lifecycle changed")

        base_validation = self.ledger.get_validation(f"{base_attempt_id}-validation")
        base_artifact_hash = base_call.response_artifact_hash
        if base_artifact_hash is None:
            raise RuntimeError("resumed fallback base output artifact is absent")
        expected_base_status = (
            ValidationStatus.REJECTED if repaired else ValidationStatus.ACCEPTED
        )
        validation_common = (
            base_validation.job_id == job.job_id
            and base_validation.attempt_id == base_attempt_id
            and base_validation.input_artifact_hash == base_artifact_hash
            and base_validation.validator_manifest_hash == state.get("execution_hash")
            and base_validation.validation_status.value == expected_base_status.value
            and base_validation.evidence_support_status.value == "not_applicable"
            and base_validation.temporal_status
            is TemporalValidationStatus.NOT_APPLICABLE
            and base_validation.commitment_status.value == "not_applicable"
            and base_validation.semantic_assessment_scope
            is SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
            and base_validation.parent_validation_id is None
            and base_validation.repair_attempt_id is None
        )
        if not validation_common:
            raise RuntimeError("resumed fallback base validation lineage changed")
        base_failures = tuple(
            failure
            for failure in self.ledger.failures_for_lineage(base_attempt_id)
            if failure.attempt_id == base_attempt_id
        )
        if repaired:
            if (
                len(base_failures) != 1
                or base_failures[0].failure_kind is not FailureKind.INVALID_OUTPUT
                or base_failures[0].artifact_hash != base_artifact_hash
            ):
                raise RuntimeError("resumed fallback repair base failure changed")
            repair_attempt_id = f"{self.run_id}-{call.call_id}-repair-01-attempt"
            repair_call = self.ledger.get_model_call(
                f"{self.run_id}-{call.call_id}-repair-01"
            )
            repair_validation = self.ledger.get_validation(
                f"{repair_attempt_id}-validation"
            )
            if (
                repair_call.job_id != job.job_id
                or repair_call.response_artifact_hash is None
                or repair_validation.job_id != job.job_id
                or repair_validation.attempt_id != repair_attempt_id
                or repair_validation.input_artifact_hash
                != repair_call.response_artifact_hash
                or repair_validation.validator_manifest_hash != state.get("execution_hash")
                or repair_validation.validation_status.value
                != ValidationStatus.ACCEPTED.value
                or repair_validation.evidence_support_status.value != "not_applicable"
                or repair_validation.temporal_status
                is not TemporalValidationStatus.NOT_APPLICABLE
                or repair_validation.commitment_status.value != "not_applicable"
                or repair_validation.semantic_assessment_scope
                is not SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
                or repair_validation.parent_validation_id
                != base_validation.validation_id
                or repair_validation.repair_attempt_id != repair_attempt_id
            ):
                raise RuntimeError("resumed fallback repair validation lineage changed")
            repair_failures = tuple(
                failure
                for failure in self.ledger.failures_for_lineage(repair_attempt_id)
                if failure.attempt_id == repair_attempt_id
            )
            if repair_failures:
                raise RuntimeError("resumed accepted fallback repair carries a failure")
        elif base_failures:
            raise RuntimeError("resumed accepted fallback base carries a failure")

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
        parent_lineage = (
            ()
            if expected_parent_attempt_id is None
            else self.ledger.attempt_lineage(expected_parent_attempt_id)
        )
        if (
            attempt.attempt_kind is not expected_attempt_kind
            or attempt.parent_attempt_id != expected_parent_attempt_id
            or attempt.input_hash != request.request_hash
            or attempt.config_hash != request.decoding.content_hash
            or attempt.seed != request.decoding.seed
            or attempt.job_id != model_call.job_id
            or tuple(lineage[:-1]) != tuple(parent_lineage)
        ):
            mismatches.append("attempt_lineage")
        event = (
            None
            if model_call.gpu_event_id is None
            else _event_for(
                self.ledger,
                model_call.gpu_event_id,
            )
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

        if state.get("development_continuation_completed") is True:
            return self._resume_completed_development_continuation(
                state=state,
                execution_hash=execution_hash,
                registration=registration,
            )
        if self.service.state is not ServiceState.READY:
            raise RuntimeError("development handoff requires the accepted service to be live")
        storage_admission, storage_report = self._record_development_storage_admission(state)
        if not storage_report.allowed:
            raise StorageBudgetExceeded(storage_report)
        service_pid = self.service.pid
        service_checkpoint_path = self.checkpoint_path.with_name(
            self.checkpoint_path.name + ".service"
        )
        checkpoint = _load_object(service_checkpoint_path)
        start_ticks = checkpoint.get("process_start_ticks")
        gpu_session_event_id = checkpoint.get("accounting_session_id")
        if (
            checkpoint.get("pid") != service_pid
            or checkpoint.get("configuration_hash") != self.service.configuration.configuration_hash
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
            development_storage_admission_hash=storage_admission.content_hash,
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
            DevelopmentCallManifest.model_validate(prepared.call_manifest.model_dump(mode="python"))
            DevelopmentPrequeryInputs.model_validate(
                prepared.prequery_inputs.model_dump(mode="python")
            )
            ForecastControl.model_validate(prepared.forecast_control.model_dump(mode="python"))
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
            _file_sha256(development_checkpoint) if development_checkpoint.is_file() else None
        )
        adapter = prepared.service_adapter
        if not isinstance(adapter, InjectedLiveDevelopmentService):
            raise TypeError("development preparation lacks the narrow live-service adapter")
        exposed_lifecycle_members = tuple(
            name for name in FORBIDDEN_ADAPTER_LIFECYCLE_MEMBERS if hasattr(adapter, name)
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
            development_storage_admission_hash=storage_admission.content_hash,
            allocated_gpu_seconds_before=allocated_before,
            development_call_manifest_hash=prepared.call_manifest.content_hash,
            development_source_plan_hash=prepared.call_manifest.source_plan_hash,
            development_execution_id=prepared.execution_id,
            development_execution_manifest_hash=prepared.execution_manifest_hash,
            gpu_call_inventory_file_sha256=(prepared.call_manifest.gpu_call_inventory_file_sha256),
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
            if development_checkpoint.is_file() and not development_checkpoint.is_symlink()
            else None
        )
        exact_itt = len(development_result.itt_records) == 24 and tuple(
            (
                row.ordinal,
                row.call_id,
                row.call_class,
                row.condition,
                row.unit_id,
            )
            for row in development_result.itt_records
        ) == tuple(
            (
                call.ordinal,
                call.call_id,
                call.call_class,
                call.condition,
                call.unit_id,
            )
            for call in prepared.call_manifest.calls
        )
        every_started = exact_itt and all(
            row.request_start_state is RequestStartState.STARTED
            for row in development_result.itt_records
        )
        every_succeeded = exact_itt and all(
            row.outcome is RunOutcome.SUCCEEDED for row in development_result.itt_records
        )
        observed_access_by_stage = {
            event.stage_manifest_hash: event for event in development_result.query_access_events
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
                and event.prequery_barrier_hash == development_result.prequery_barrier_hash
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
                    if observed_access_by_stage.get(call.query_stage.staging_manifest_hash) is None
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
            development_result.forecast.forecast_receipt_hash != handoff.forecast_receipt_hash
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
            development_storage_admission_hash=(
                handoff.development_storage_admission_hash
            ),
            development_execution_id=development_result.execution_id,
            development_execution_manifest_hash=(development_result.execution_manifest_hash),
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
            scientific_thresholds_passed=(development_result.gate.scientific_thresholds_passed),
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

        service_checkpoint = self.checkpoint_path.with_name(self.checkpoint_path.name + ".service")
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

        second_retry_lineage = self._second_recovery_retry_lineage()
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
            for call in calls:
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
                is_second_recovery_retry = (
                    second_retry_lineage is not None
                    and call.call_id == SECOND_RECOVERY_RETRY_CALL_ID
                )
                prequery_sealed_at = datetime.now(UTC)
                job = (
                    self.ledger.get_job(second_retry_lineage[0])
                    if is_second_recovery_retry and second_retry_lineage is not None
                    else self.ledger.create_or_resume_job(
                        self._job_identity(call=call, execution_hash=execution_hash),
                        release_class=ReleaseClass.PUBLIC,
                        created_at=prequery_sealed_at,
                    )
                )
                self._prepare_job_lifecycle(
                    job_id=job.job_id,
                    call=call,
                    prequery_sealed_at=prequery_sealed_at,
                )
                # Query-time requests are constructed only after the durable reveal
                # transition above. C1 takes the direct query-blind branch instead.
                request = build_fallback_acceptance_request(
                    root=self.root,
                    call=call,
                    tokenizer=self.tokenizer,
                    tokenizer_manifest=self.tokenizer_manifest,
                    legacy_provenance_bridge=self.legacy_provenance_bridge,
                )
                attempt_id = f"{self.run_id}-{call.call_id}-attempt"
                attempt_kind = (
                    AttemptKind.RETRY if is_second_recovery_retry else AttemptKind.BASE
                )
                parent_attempt_id = (
                    second_retry_lineage[1]
                    if is_second_recovery_retry and second_retry_lineage is not None
                    else None
                )
                attempt_created_at = datetime.now(UTC)
                reserve = self._reserve(
                    state,
                    call_id=call.call_id,
                    reserve_call_class=call.reserve_call_class,
                    watchdog_seconds=call.watchdog_seconds,
                    job_id=job.job_id,
                    attempt_id=attempt_id,
                    attempt_kind=attempt_kind,
                    parent_attempt_id=parent_attempt_id,
                    input_hash=request.request_hash,
                    config_hash=request.decoding.content_hash,
                    seed=request.decoding.seed,
                    attempt_created_at=attempt_created_at,
                )
                self.ledger.record_attempt(
                    attempt_id=attempt_id,
                    job_id=job.job_id,
                    attempt_kind=attempt_kind,
                    parent_attempt_id=parent_attempt_id,
                    input_hash=request.request_hash,
                    config_hash=request.decoding.content_hash,
                    seed=request.decoding.seed,
                    created_at=attempt_created_at,
                )
                remaining = self._remaining_mandatory_forecast_seconds(state)
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
                    self._record_failed_transport(
                        call=call,
                        request=request,
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        event_id=event_id,
                        exc=exc,
                        retry_class=call.retry_class,
                        call_role=self._call_role(call),
                        construction_unit_hash=_base_construction_unit_hash(call),
                        validator_manifest_hash=execution_hash,
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
                    "call_role": self._call_role(call),
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
                        legacy_provenance_bridge=self.legacy_provenance_bridge,
                    )
                    audit["operator_behavior"] = _require_call_operator_coverage(
                        call,
                        audit,
                        generated.parsed_object,
                        root=self.root,
                    )
                except Exception as validation_error:
                    # The model call completed and its GPU event succeeded; the
                    # separate validation/failure records below capture that the
                    # returned ontology draft was structurally inadmissible.
                    self.ledger.record_model_call(**common_call, successful=True)
                    terminal_at = datetime.now(UTC)
                    self._advance_one(
                        job_id=job.job_id,
                        call=call,
                        state=JobState.GENERATED,
                        occurred_at=terminal_at,
                    )
                    base_validation_id = self._record_structural_validation(
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        input_artifact_hash=artifact.content_hash,
                        validator_manifest_hash=execution_hash,
                        accepted=False,
                        diagnostics_artifact_hash=artifact.content_hash,
                        created_at=terminal_at,
                    )
                    self.ledger.record_failure(
                        attempt_id=attempt_id,
                        failure_kind=FailureKind.INVALID_OUTPUT,
                        message="Fallback base output failed mechanical validation",
                        details={
                            "exception_type": type(validation_error).__name__,
                            "call_id": call.call_id,
                        },
                        artifact_hash=artifact.content_hash,
                        occurred_at=terminal_at,
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
                        self._finish_job_lifecycle(
                            job_id=job.job_id,
                            call=call,
                            occurred_at=terminal_at,
                        )
                        self._mark_terminal(state, call.call_id)
                        break
                    state["repair_parent_call_id"] = call.call_id
                    repair_attempt_count += 1
                    (
                        repair_public_result,
                        repaired_audit,
                        repair_allocated_seconds,
                        repair_transport_succeeded,
                    ) = self._run_repair(
                        state=state,
                        call=call,
                        base_call=base_call,
                        base_request=request,
                        invalid_generated=generated,
                        invalid_artifact_hash=artifact.content_hash,
                        diagnostics=diagnostics,
                        job_id=job.job_id,
                        parent_attempt_id=attempt_id,
                        parent_validation_id=base_validation_id,
                        validator_manifest_hash=execution_hash,
                    )
                    results.append(repair_public_result)
                    if repair_transport_succeeded:
                        timing_observations.append(
                            TimingObservation(
                                call_class="acceptance_repair",
                                allocated_seconds=repair_allocated_seconds,
                            )
                        )
                    if repaired_audit is None:
                        self._mark_terminal(state, f"{call.call_id}-repair-01")
                        break
                    audit = repaired_audit
                    accepted_model_call_id = cast(str, repair_public_result["model_call_id"])
                    cast(dict[str, object], state["accepted_outputs"])[call.call_id] = {
                        "model_call_id": accepted_model_call_id,
                        "repaired": True,
                        "invalid_base_artifact_hash": artifact.content_hash,
                        "diagnostics": [dict(item) for item in diagnostics],
                    }
                else:
                    self.ledger.record_model_call(**common_call, successful=True)
                    terminal_at = datetime.now(UTC)
                    self._advance_one(
                        job_id=job.job_id,
                        call=call,
                        state=JobState.GENERATED,
                        occurred_at=terminal_at,
                    )
                    self._record_structural_validation(
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        input_artifact_hash=artifact.content_hash,
                        validator_manifest_hash=execution_hash,
                        accepted=True,
                        created_at=terminal_at,
                    )
                    self._finish_job_lifecycle(
                        job_id=job.job_id,
                        call=call,
                        occurred_at=terminal_at,
                    )
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
                state["active_attempt"] = None
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
                    self._reconcile_development_storage_admissions(state)
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
        parent_validation_id: str,
        validator_manifest_hash: str,
    ) -> tuple[dict[str, object], dict[str, object] | None, float, bool]:
        repair_id = f"{call.call_id}-repair-01"
        request = build_fallback_repair_request(
            root=self.root,
            base_call=base_call,
            base_request=base_request,
            invalid_draft=invalid_generated.parsed_object,
            diagnostics=diagnostics,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            legacy_provenance_bridge=self.legacy_provenance_bridge,
        )
        attempt_id = f"{self.run_id}-{repair_id}-attempt"
        attempt_created_at = datetime.now(UTC)
        reserve = self._reserve(
            state,
            call_id=repair_id,
            reserve_call_class="reserve_short",
            watchdog_seconds=90,
            job_id=job_id,
            attempt_id=attempt_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=parent_attempt_id,
            input_hash=request.request_hash,
            config_hash=request.decoding.content_hash,
            seed=request.decoding.seed,
            attempt_created_at=attempt_created_at,
        )
        self.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job_id,
            attempt_kind=AttemptKind.REPAIR,
            parent_attempt_id=parent_attempt_id,
            input_hash=request.request_hash,
            config_hash=request.decoding.content_hash,
            seed=request.decoding.seed,
            created_at=attempt_created_at,
        )
        remaining_required_seconds = self._remaining_mandatory_forecast_seconds(state)
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
                construction_unit_hash=_repair_construction_unit_hash(
                    call,
                    parent_attempt_id,
                ),
                validator_manifest_hash=validator_manifest_hash,
                parent_validation_id=parent_validation_id,
                repair=True,
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
                False,
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
                legacy_provenance_bridge=self.legacy_provenance_bridge,
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
            # Transport/model execution succeeded even though the resulting
            # repair failed subsequent structural validation.
            self.ledger.record_model_call(**common_call, successful=True)
            terminal_at = datetime.now(UTC)
            self._advance_one(
                job_id=job_id,
                call=call,
                state=JobState.REPAIRED,
                occurred_at=terminal_at,
            )
            self._record_structural_validation(
                job_id=job_id,
                attempt_id=attempt_id,
                input_artifact_hash=artifact.content_hash,
                validator_manifest_hash=validator_manifest_hash,
                accepted=False,
                diagnostics_artifact_hash=artifact.content_hash,
                parent_validation_id=parent_validation_id,
                repair=True,
                created_at=terminal_at,
            )
            self.ledger.record_failure(
                attempt_id=attempt_id,
                failure_kind=FailureKind.INVALID_OUTPUT,
                message="Fallback repair failed complete mechanical validation",
                details={"exception_type": type(exc).__name__, "call_id": repair_id},
                artifact_hash=artifact.content_hash,
                occurred_at=terminal_at,
            )
            self._finish_job_lifecycle(
                job_id=job_id,
                call=call,
                occurred_at=terminal_at,
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
                True,
            )
        self.ledger.record_model_call(**common_call, successful=True)
        terminal_at = datetime.now(UTC)
        self._advance_one(
            job_id=job_id,
            call=call,
            state=JobState.REPAIRED,
            occurred_at=terminal_at,
        )
        self._record_structural_validation(
            job_id=job_id,
            attempt_id=attempt_id,
            input_artifact_hash=artifact.content_hash,
            validator_manifest_hash=validator_manifest_hash,
            accepted=True,
            parent_validation_id=parent_validation_id,
            repair=True,
            created_at=terminal_at,
        )
        self._finish_job_lifecycle(
            job_id=job_id,
            call=call,
            occurred_at=terminal_at,
        )
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
            True,
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
            if isinstance(completed, list) and all(isinstance(item, str) for item in completed)
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
        inventory = GPUCallInventory.load(self.root / "configs/study/gpu_call_inventory.json")
        reference = forecast_gpu_schedule(inventory, limits=limits)
        consumed = Counter(cast(str, row.get("reserve_call_class")) for row in receipts)
        for name in NORMAL_ACCEPTANCE_CLASSES:
            consumed[name] = inventory.call_class(name).count
        base_service_starts, recovery_service_starts = (
            self._base_inventory_consumed_service_starts()
        )
        consumed["gpu_session_start"] = base_service_starts
        raw_development_receipt = state.get("development_continuation_receipt")
        try:
            development_receipt = (
                None
                if raw_development_receipt is None
                else DevelopmentContinuationReceipt.model_validate(raw_development_receipt)
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
            cleanup_failure is None
            and self.service.state is ServiceState.STOPPED
            and not self.ledger.unresolved_gpu_allocations()
            and not self.ledger.unresolved_gpu_service_journals()
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
        continuation_completed = state.get("development_continuation_completed") is True
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
                            "query_access_event_count": len(parsed_checkpoint.query_access_events),
                            "active_call_id": parsed_checkpoint.active_call_id,
                            "service_identity_hash": (parsed_checkpoint.service_identity_hash),
                            "resume_authorized_after_owner_shutdown": False,
                        }
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_micro_pilot_result",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "execution_identity": identity,
            "fallback_plan_manifest_sha256": plan_hash,
            "fallback_service_retry_amendment_sha256": self.retry_amendment_hash,
            "prior_fallback_failure_sha256": self.prior_fallback_failure_hash,
            "second_fallback_recovery_overlay_sha256": (self.second_recovery_overlay_hash),
            "second_recovery_v3_result_sha256": self.second_recovery_v3_result_hash,
            "second_recovery_v3_incident_sha256": (self.second_recovery_v3_incident_hash),
            "effective_gpu_call_inventory": self._effective_inventory_manifest(),
            "recovery_service_start_events_consumed": recovery_service_starts,
            "normal_acceptance_block_executed": False,
            "completed_base_call_count": len(completed_ids),
            "completed_call_ids": completed_ids,
            "repair_attempt_count": (1 if state.get("repair_parent_call_id") is not None else 0),
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
                continuation_ready and not continuation_completed and not physical_shutdown_verified
            ),
            "development_continuation_interrupted": (
                continuation_ready and not continuation_completed
            ),
            "partial_development_checkpoint": partial_development_checkpoint,
            "development_continuation_integration_pending": not continuation_ready,
            "micro_pilot_acceptance_receipt": state.get("micro_pilot_acceptance_receipt"),
            "accepted_micro_pilot_result": state.get("accepted_micro_pilot_result"),
            "development_continuation_bootstrap": state.get("development_continuation_bootstrap"),
            "development_storage_admission": (
                cast(list[object], state.get("development_storage_admissions", []))[-1]
                if state.get("development_storage_admissions")
                else None
            ),
            "development_preparation": _public_development_state(
                state.get("development_preparation"),
                restricted_fields=frozenset({"checkpoint_path"}),
            ),
            "development_handoff": _public_development_state(
                state.get("development_handoff"),
                restricted_fields=frozenset({"development_checkpoint_path"}),
            ),
            "development_execution_result": state.get("development_execution_result"),
            "development_continuation_receipt": (
                None if development_receipt is None else development_receipt.model_dump(mode="json")
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
        completed_repair_transport_count = sum(
            1
            for result in results
            if str(result.get("call_id", "")).endswith("-repair-01")
            and isinstance(result.get("response_artifact_hash"), str)
        )
        observed_repair_sample_count = (
            0
            if "acceptance_repair" not in timing_by_name
            else timing_by_name["acceptance_repair"].sample_count
        )
        timing_gate = {
            "expected_base_sample_counts": expected_timing_counts,
            "observed_call_classes": sorted(timing_by_name),
            "base_sample_counts_exact": all(
                name in timing_by_name and timing_by_name[name].sample_count == count
                for name, count in expected_timing_counts.items()
            ),
            "repair_sample_count": observed_repair_sample_count,
            "completed_repair_transport_count": completed_repair_transport_count,
            "repair_sample_count_valid": (
                repair_attempt_count <= 1
                and observed_repair_sample_count == completed_repair_transport_count
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
        base_service_starts, recovery_service_starts = (
            self._base_inventory_consumed_service_starts()
        )
        consumed["gpu_session_start"] = base_service_starts
        raw_development_receipt = state.get("development_continuation_receipt")
        development_receipt = (
            None
            if raw_development_receipt is None
            else DevelopmentContinuationReceipt.model_validate(raw_development_receipt)
        )
        continuation_completed = state.get("development_continuation_completed") is True
        storage_receipts = self._verify_development_storage_admissions(
            state,
            required=continuation_completed,
        )
        storage_admission = storage_receipts[-1] if storage_receipts else None
        if continuation_completed:
            if development_receipt is None:
                raise RuntimeError("completed development continuation lacks its immutable receipt")
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
            consumed_count = (
                row.count
                if superseded
                else min(
                    row.count,
                    consumed[row.call_class],
                )
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
            "consumed_recovery_service_start_slots": recovery_service_starts,
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
            "controller_process_restart": (state["controller_process_restart_observed"] is True),
            "model_process_restart": False,
            "service_pid_unchanged": state.get("handoff_service_pid") is not None,
            "stage_controller_pids_distinct": (
                state.get("stage_one_controller_pid") != state.get("stage_two_controller_pid")
            ),
            "passed": (
                state["controller_handoff_complete"] is True
                and resumed_live_service
                and state["controller_process_restart_observed"] is True
                and state.get("stage_one_controller_pid") != state.get("stage_two_controller_pid")
            ),
        }
        physical_restart_gate = {
            "required_restart_kind": "controller_process_restart_with_live_model_adoption",
            "controller_process_restart": controller_resume_gate["controller_process_restart"],
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
        same_service_pid = state.get("development_handoff_service_pid") == state.get(
            "handoff_service_pid"
        )
        service_is_live = self.service.state is ServiceState.READY
        owner_shutdown_verified = (
            self.service.state is ServiceState.STOPPED
            and uptime is not None
            and not self.ledger.unresolved_gpu_allocations()
            and not self.ledger.unresolved_gpu_service_journals()
        )
        receipt_matches_handoff = (
            development_receipt is not None
            and development_receipt.handoff_hash == state.get("development_handoff_hash")
            and development_receipt.fallback_execution_hash == execution_hash
            and development_receipt.service_pid == state.get("handoff_service_pid")
            and development_receipt.adopter_registration_hash
            == state.get("development_adopter_registration_hash")
            and storage_admission is not None
            and development_receipt.development_storage_admission_hash
            == storage_admission.content_hash
        )
        development_continuation_gate = {
            "ready": ready,
            "completed": continuation_completed,
            "same_model_service_pid": (same_service_pid),
            "configuration_sha256": self.service.configuration.configuration_hash,
            "source_execution_hash": execution_hash,
            "next_workload": "registered_24_call_development_block",
            "requires_new_model_load": False,
            "integrated_adopter_available": registration is not None,
            "adopter_registration_hash": (
                None if registration is None else registration.content_hash
            ),
            "receipt_matches_handoff": receipt_matches_handoff,
            "phase3_storage_admitted": (
                None if storage_admission is None else storage_admission.allowed
            ),
            "phase3_storage_admission_hash": (
                None if storage_admission is None else storage_admission.content_hash
            ),
            "terminal_intention_to_treat_call_count": (
                None if development_receipt is None else development_receipt.terminal_call_count
            ),
            "service_returned_live_to_owner": (
                None
                if development_receipt is None
                else development_receipt.service_returned_live_to_owner
            ),
            "development_gate_passed": (
                None if development_receipt is None else development_receipt.development_gate_passed
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
                and storage_admission is not None
                and storage_admission.allowed
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
                key: value for key, value in selected_freeze.items() if key != "manifest_sha256"
            }
            if selected_freeze.get("manifest_sha256") != canonical_sha256(immutable_freeze):
                raise RuntimeError("checkpoint selected-model freeze hash changed")
        if phase1_gate_passed and selected_freeze is None:
            raise RuntimeError("passed fallback handoff lacks a selected-model freeze")
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_fallback_micro_pilot_result",
            "run_id": self.run_id,
            "execution_hash": execution_hash,
            "fallback_plan_manifest_sha256": plan_manifest_hash,
            "fallback_service_retry_amendment_sha256": self.retry_amendment_hash,
            "prior_fallback_failure_sha256": self.prior_fallback_failure_hash,
            "second_fallback_recovery_overlay_sha256": (self.second_recovery_overlay_hash),
            "second_recovery_v3_result_sha256": self.second_recovery_v3_result_hash,
            "second_recovery_v3_incident_sha256": (self.second_recovery_v3_incident_hash),
            "effective_gpu_call_inventory": self._effective_inventory_manifest(),
            "recovery_service_start_events_consumed": recovery_service_starts,
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
            "development_handoff_pending": (ready and not continuation_completed),
            "development_continuation_integration_pending": (micro_pilot_passed and not ready),
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
                "reference_inventory_total_seconds_before_supersession": (forecast.total_seconds),
            },
            "actual_plus_remaining_forecast": continuation,
            "operator_coverage_gate": operator_gate,
            "grounding_horizon_gate": grounding_horizon_gate,
            "micro_pilot_acceptance_receipt": state.get("micro_pilot_acceptance_receipt"),
            "accepted_micro_pilot_result": state.get("accepted_micro_pilot_result"),
            "development_continuation_bootstrap": state.get("development_continuation_bootstrap"),
            "development_storage_admission": (
                None if storage_admission is None else storage_admission.model_dump(mode="json")
            ),
            "development_preparation": _public_development_state(
                state.get("development_preparation"),
                restricted_fields=frozenset({"checkpoint_path"}),
            ),
            "development_handoff": _public_development_state(
                state.get("development_handoff"),
                restricted_fields=frozenset({"development_checkpoint_path"}),
            ),
            "selected_model_freeze": selected_freeze,
            "development_execution_result": state.get("development_execution_result"),
            "development_continuation_receipt": (
                None if development_receipt is None else development_receipt.model_dump(mode="json")
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

    identity: dict[str, object] = {
        "project_root": str(options.project_root.resolve()),
        "run_id": options.run_id,
        "primary_result": str(options.primary_result.resolve()),
        "activation_certificate": str(options.activation_certificate.resolve()),
        "cache_replacement_receipt": str(options.cache_replacement_receipt.resolve()),
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
    if options.retry_amendment is not None:
        identity["retry_amendment"] = str(options.retry_amendment.resolve())
        identity["prior_fallback_failure"] = str(options.prior_fallback_failure.resolve())
    if options.second_recovery_overlay is not None:
        identity["second_recovery_overlay"] = str(options.second_recovery_overlay.resolve())
        identity["second_recovery_v3_result"] = str(options.second_recovery_v3_result.resolve())
        identity["second_recovery_v3_incident"] = str(options.second_recovery_v3_incident.resolve())
    return identity


@dataclass(frozen=True, slots=True)
class _FallbackOrchestrationPaths:
    checkpoint: Path
    service_checkpoint: Path
    invocation: Path
    guardian_ticket: Path
    guardian_lock: Path
    guardian_log: Path
    guardian_result: Path
    terminal_request: Path
    controller_takeover: Path
    handoff_output: Path
    cleanup_output: Path
    result_output: Path


def _orchestration_paths(options: argparse.Namespace) -> _FallbackOrchestrationPaths:
    checkpoint = options.checkpoint.resolve()
    result = options.output.resolve()
    return _FallbackOrchestrationPaths(
        checkpoint=checkpoint,
        service_checkpoint=checkpoint.with_name(checkpoint.name + ".service"),
        invocation=checkpoint.with_name(checkpoint.name + ".orchestrator-invocation.json"),
        guardian_ticket=checkpoint.with_name(checkpoint.name + ".guardian-ticket.json"),
        guardian_lock=checkpoint.with_name(checkpoint.name + ".guardian.lock"),
        guardian_log=checkpoint.with_name(checkpoint.name + ".guardian.log"),
        guardian_result=checkpoint.with_name(checkpoint.name + ".guardian-result.json"),
        terminal_request=checkpoint.with_name(checkpoint.name + ".guardian-terminal-request.json"),
        controller_takeover=checkpoint.with_name(
            checkpoint.name + ".guardian-controller-takeover.json"
        ),
        handoff_output=result.with_name(result.name + ".controller-handoff.json"),
        cleanup_output=result.with_name(result.name + ".orphan-cleanup.json"),
        result_output=result,
    )


def _guard_path(options: argparse.Namespace, sequence: int) -> Path:
    if sequence <= 0 or sequence > 999_999:
        raise ValueError("orchestrator invocation sequence is out of range")
    checkpoint = options.checkpoint.resolve()
    return checkpoint.with_name(f"{checkpoint.name}.orchestrator-guard-{sequence:06d}.json")


def _guard_paths(options: argparse.Namespace) -> tuple[Path, ...]:
    checkpoint = options.checkpoint.resolve()
    prefix = checkpoint.name + ".orchestrator-guard-"
    candidates: list[tuple[int, Path]] = []
    if checkpoint.parent.exists():
        for item in checkpoint.parent.iterdir():
            if not item.name.startswith(prefix):
                continue
            suffix = item.name.removeprefix(prefix)
            if len(suffix) != 11 or not suffix.endswith(".json") or not suffix[:6].isdigit():
                raise RuntimeError("orchestrator guard inventory contains an invalid entry")
            if item.is_symlink() or not item.is_file():
                raise RuntimeError("orchestrator guard must be a regular non-symlink file")
            candidates.append((int(suffix[:6]), item.resolve()))
    candidates.sort()
    if [sequence for sequence, _ in candidates] != list(range(1, len(candidates) + 1)):
        raise RuntimeError("orchestrator guard sequence is not contiguous")
    paths = tuple(path for _, path in candidates)
    expected_close_names = {_guard_close_path(path).name for path in paths}
    if checkpoint.parent.exists():
        close_prefix = "." + prefix
        for item in checkpoint.parent.iterdir():
            if item.name.startswith(close_prefix) and item.name.endswith(".json.closed.json"):
                if item.name not in expected_close_names:
                    raise RuntimeError("orchestrator close inventory has no matching guard")
                if item.is_symlink() or not item.is_file():
                    raise RuntimeError("orchestrator close receipt must be a regular file")
    return paths


def _load_guard_chain(options: argparse.Namespace) -> tuple[dict[str, object], ...]:
    previous_hash: str | None = None
    guards: list[dict[str, object]] = []
    for sequence, path in enumerate(_guard_paths(options), start=1):
        guard = _load_hashed_object(
            path,
            expected_kind="fallback_controller_orchestrator_guard",
        )
        if (
            guard.get("run_id") != options.run_id
            or guard.get("sequence") != sequence
            or guard.get("previous_guard_sha256") != previous_hash
        ):
            raise ValueError("orchestrator guard chain changed or was reordered")
        close_path = _guard_close_path(path)
        if close_path.exists() or close_path.is_symlink():
            closed = _load_hashed_object(
                close_path,
                expected_kind="fallback_controller_orchestrator_closed",
            )
            if (
                closed.get("run_id") != options.run_id
                or closed.get("orchestrator_guard_sha256") != guard.get("manifest_sha256")
                or not isinstance(closed.get("physical_shutdown_verified"), bool)
            ):
                raise ValueError("orchestrator close receipt is not bound to its exact guard")
        previous_hash = cast(str, guard["manifest_sha256"])
        guards.append(guard)
    return tuple(guards)


def _expected_orchestrator_guard(options: argparse.Namespace) -> Path:
    """Return the first append-only guard path used by a fresh invocation."""

    return _guard_path(options, 1)


def _guard_close_path(guard: Path) -> Path:
    return guard.with_name("." + guard.name + ".closed.json")


def _controller_receipt_path(options: argparse.Namespace, sequence: int) -> Path:
    if sequence <= 0 or sequence > 999_999:
        raise ValueError("internal controller sequence is out of range")
    checkpoint = options.checkpoint.resolve()
    return checkpoint.with_name(f"{checkpoint.name}.internal-controller-{sequence:06d}.json")


def _controller_receipt_paths(options: argparse.Namespace) -> tuple[Path, ...]:
    checkpoint = options.checkpoint.resolve()
    prefix = checkpoint.name + ".internal-controller-"
    candidates: list[tuple[int, Path]] = []
    if checkpoint.parent.exists():
        for item in checkpoint.parent.iterdir():
            if not item.name.startswith(prefix):
                continue
            suffix = item.name.removeprefix(prefix)
            if len(suffix) != 11 or not suffix.endswith(".json") or not suffix[:6].isdigit():
                raise RuntimeError(
                    "internal controller receipt inventory contains an invalid entry"
                )
            if item.is_symlink() or not item.is_file():
                raise RuntimeError("internal controller receipt must be a regular non-symlink file")
            candidates.append((int(suffix[:6]), item.resolve()))
    candidates.sort()
    if [sequence for sequence, _ in candidates] != list(range(1, len(candidates) + 1)):
        raise RuntimeError("internal controller receipt sequence is not contiguous")
    return tuple(path for _, path in candidates)


def _raw_command_sha256(command: Sequence[str]) -> str:
    encoded = b"".join(os.fsencode(argument) + b"\0" for argument in command)
    return hashlib.sha256(encoded).hexdigest()


def _process_identity(pid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, str]:
    if isinstance(pid, bool) or pid <= 0:
        raise ValueError("process identity requires a positive PID")
    stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    close = stat.rfind(")")
    if close < 0:
        raise RuntimeError("process stat has no command terminator")
    fields = stat[close + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError("process stat lacks start ticks")
    start_ticks = int(fields[19])
    command_sha256 = hashlib.sha256((proc_root / str(pid) / "cmdline").read_bytes()).hexdigest()
    return start_ticks, command_sha256


def _process_is_zombie(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
    stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    close = stat.rfind(")")
    if close < 0:
        raise RuntimeError("process stat has no command terminator")
    fields = stat[close + 2 :].split()
    if not fields:
        raise RuntimeError("process stat lacks state")
    return fields[0] == "Z"


def _exact_bound_process_is_live(
    identity: Mapping[str, object],
    *,
    prefix: str,
) -> bool:
    pid = identity.get(f"{prefix}_pid")
    start_ticks = identity.get(f"{prefix}_start_ticks")
    command_sha256 = identity.get(f"{prefix}_command_sha256")
    process_group_id = identity.get(f"{prefix}_process_group_id")
    session_id = identity.get(f"{prefix}_session_id")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
        or not isinstance(command_sha256, str)
        or len(command_sha256) != 64
        or any(character not in "0123456789abcdef" for character in command_sha256)
        or isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id <= 0
        or isinstance(session_id, bool)
        or not isinstance(session_id, int)
        or session_id <= 0
    ):
        raise ValueError(f"{prefix} has an invalid bound process identity")
    try:
        os.kill(pid, 0)
        if _process_is_zombie(pid):
            return False
        observed_ticks, observed_command = _process_identity(pid)
        observed_group = os.getpgid(pid)
        observed_session = os.getsid(pid)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        observed_ticks == start_ticks
        and observed_command == command_sha256
        and observed_group == process_group_id
        and observed_session == session_id
    )


def _guard_process_is_live(guard: Mapping[str, object]) -> bool:
    if "orchestrator_process_group_id" in guard or "orchestrator_session_id" in guard:
        return _exact_bound_process_is_live(guard, prefix="orchestrator")
    pid = guard.get("orchestrator_pid")
    start_ticks = guard.get("orchestrator_start_ticks")
    command_sha256 = guard.get("orchestrator_command_sha256")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
        or not isinstance(command_sha256, str)
        or len(command_sha256) != 64
        or any(character not in "0123456789abcdef" for character in command_sha256)
    ):
        raise ValueError("orchestrator guard has an invalid process identity")
    try:
        os.kill(pid, 0)
        observed_ticks, observed_command = _process_identity(pid)
    except (OSError, RuntimeError, ValueError):
        return False
    return observed_ticks == start_ticks and observed_command == command_sha256


def _guardian_controller_command(
    options: argparse.Namespace,
    *,
    output: Path,
    ticket: Path,
) -> tuple[str, ...]:
    command = list(
        _internal_controller_command(
            options,
            stage="guardian",
            output=output,
            guard=None,
        )
    )
    command.extend(("--guardian-ticket", str(ticket.resolve())))
    return tuple(command)


def _guardian_ticket_for_invocation(
    options: argparse.Namespace,
    *,
    paths: _FallbackOrchestrationPaths,
    invocation: Mapping[str, object],
) -> dict[str, object]:
    guardian_command = _guardian_controller_command(
        options,
        output=paths.guardian_result,
        ticket=paths.guardian_ticket,
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_service_guardian_ticket",
        "run_id": options.run_id,
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "execution_arguments_sha256": invocation["execution_arguments_sha256"],
        "service_session_id": options.run_id,
        "service_event_id": f"{options.run_id}-service-start-001",
        "hard_stop_at": invocation["hard_stop_at"],
        "resume_grace_seconds": invocation["resume_grace_seconds"],
        "guardian_command_sha256": canonical_sha256(guardian_command),
        "created_at": invocation["created_at"],
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _new_orchestration_identity(
    options: argparse.Namespace,
    paths: _FallbackOrchestrationPaths,
) -> tuple[dict[str, object], dict[str, object]]:
    existing_targets = (
        paths.checkpoint,
        paths.service_checkpoint,
        paths.invocation,
        paths.guardian_ticket,
        paths.guardian_lock,
        paths.guardian_log,
        paths.guardian_result,
        paths.terminal_request,
        paths.controller_takeover,
        paths.handoff_output,
        paths.cleanup_output,
        paths.result_output,
        paths.checkpoint.parent / f"{options.run_id}.vllm.log",
    )
    if (
        any(path.exists() or path.is_symlink() for path in existing_targets)
        or _guard_paths(options)
        or _controller_receipt_paths(options)
    ):
        raise RuntimeError("fresh fallback orchestration requires absent durable state and outputs")
    arguments = _controller_execution_arguments(options)
    arguments_hash = canonical_sha256(arguments)
    limits = ResourceLimits.load(options.project_root / "configs/study/resource_limits.json")
    with ReadOnlyLedger(options.ledger) as ledger:
        actual = ledger.gpu_summary().total_allocated_seconds
    protected_seconds = (
        2 * DEFAULT_SHUTDOWN_SECONDS
        + FALLBACK_GUARDIAN_POLL_SECONDS
        + FALLBACK_CONTROL_GROUP_TERM_SECONDS
        + FALLBACK_CONTROL_GROUP_KILL_SECONDS
        + 5.0
    )
    remaining_until_hard_stop = limits.hard_gpu_seconds - actual - protected_seconds
    if remaining_until_hard_stop <= 0:
        raise RuntimeError("fallback guardian has no protected hard-stop interval")
    created_at = datetime.now(UTC)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_controller_orchestration_invocation",
        "run_id": options.run_id,
        "execution_arguments_sha256": arguments_hash,
        "result_output": str(paths.result_output),
        "handoff_output": str(paths.handoff_output),
        "cleanup_output": str(paths.cleanup_output),
        "checkpoint": str(paths.checkpoint),
        "service_session_id": options.run_id,
        "service_event_id": f"{options.run_id}-service-start-001",
        "invocation_nonce": secrets.token_hex(32),
        "gpu_seconds_before_invocation": actual,
        "protected_shutdown_seconds": protected_seconds,
        "hard_stop_at": (created_at + timedelta(seconds=remaining_until_hard_stop)).isoformat(),
        "resume_grace_seconds": FALLBACK_ORCHESTRATOR_RESUME_GRACE_SECONDS,
        "created_at": created_at.isoformat(),
    }
    invocation = {**payload, "manifest_sha256": canonical_sha256(payload)}
    _write_append_only_json(paths.invocation, invocation)
    ticket = _guardian_ticket_for_invocation(
        options,
        paths=paths,
        invocation=invocation,
    )
    _write_append_only_json(paths.guardian_ticket, ticket)
    return invocation, ticket


def _load_orchestration_identity(
    options: argparse.Namespace,
    paths: _FallbackOrchestrationPaths,
) -> tuple[dict[str, object], dict[str, object]]:
    invocation = _load_hashed_object(
        paths.invocation,
        expected_kind="fallback_controller_orchestration_invocation",
    )
    arguments_hash = canonical_sha256(_controller_execution_arguments(options))
    invocation_identity_valid = (
        invocation.get("run_id") == options.run_id
        and invocation.get("execution_arguments_sha256") == arguments_hash
        and invocation.get("result_output") == str(paths.result_output)
        and invocation.get("handoff_output") == str(paths.handoff_output)
        and invocation.get("cleanup_output") == str(paths.cleanup_output)
        and invocation.get("checkpoint") == str(paths.checkpoint)
    )
    if not invocation_identity_valid:
        raise ValueError("fallback orchestration invocation identity changed on resume")
    if not paths.guardian_ticket.exists():
        if (
            not options.resume_orchestrator
            or paths.guardian_ticket.is_symlink()
            or _guard_paths(options)
            or _controller_receipt_paths(options)
            or _guardian_ready_paths(paths)
            or any(
                path.exists() or path.is_symlink()
                for path in (
                    paths.checkpoint,
                    paths.service_checkpoint,
                    paths.guardian_result,
                    paths.terminal_request,
                    paths.controller_takeover,
                    paths.handoff_output,
                    paths.cleanup_output,
                    paths.result_output,
                )
            )
        ):
            raise RuntimeError(
                "missing guardian ticket cannot be repaired after controller state exists"
            )
        repaired_ticket = _guardian_ticket_for_invocation(
            options,
            paths=paths,
            invocation=invocation,
        )
        _write_append_only_json(paths.guardian_ticket, repaired_ticket)
    ticket = _load_hashed_object(
        paths.guardian_ticket,
        expected_kind="fallback_service_guardian_ticket",
    )
    if (
        ticket.get("run_id") != options.run_id
        or ticket.get("execution_arguments_sha256") != arguments_hash
        or ticket.get("orchestration_invocation_sha256") != invocation.get("manifest_sha256")
        or ticket.get("service_session_id") != options.run_id
        or ticket.get("service_event_id") != f"{options.run_id}-service-start-001"
        or ticket.get("hard_stop_at") != invocation.get("hard_stop_at")
        or ticket.get("resume_grace_seconds") != invocation.get("resume_grace_seconds")
    ):
        raise ValueError("fallback orchestration invocation identity changed on resume")
    if canonical_sha256(
        _guardian_controller_command(
            options,
            output=paths.guardian_result,
            ticket=paths.guardian_ticket,
        )
    ) != ticket.get("guardian_command_sha256"):
        raise ValueError("fallback guardian command changed after authorization")
    return invocation, ticket


def _create_orchestrator_guard(
    options: argparse.Namespace,
    *,
    invocation: Mapping[str, object],
    ticket: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    prior_paths = _guard_paths(options)
    prior_guards = _load_guard_chain(options)
    previous_hash = None
    if prior_paths:
        prior = prior_guards[-1]
        if any(
            item.get("orchestration_invocation_sha256") != invocation["manifest_sha256"]
            or item.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
            or item.get("execution_arguments_sha256") != invocation["execution_arguments_sha256"]
            for item in prior_guards
        ):
            raise ValueError("prior orchestrator guard chain belongs to another invocation")
        if _guard_process_is_live(prior):
            raise RuntimeError("a live fallback orchestrator already owns this invocation")
        prior_group = prior.get("orchestrator_process_group_id")
        if isinstance(prior_group, bool) or not isinstance(prior_group, int) or prior_group <= 0:
            raise ValueError("prior orchestrator guard lacks its dedicated process group")
        if _control_process_group_alive(prior_group):
            raise RuntimeError(
                "the prior orchestrator control group is still live; resume must wait"
            )
        if any(
            _exact_bound_process_is_live(receipt, prefix="controller")
            for receipt in _load_controller_receipts(
                options,
                invocation=invocation,
            )
        ):
            raise RuntimeError(
                "an internal fallback controller is still live; resume must wait for it"
            )
        previous_hash = prior["manifest_sha256"]
    sequence = len(prior_paths) + 1
    path = _guard_path(options, sequence)
    orchestrator_pid = os.getpid()
    process_group_id = os.getpgrp()
    if process_group_id != orchestrator_pid:
        raise RuntimeError(
            "fallback orchestrator requires its dedicated process group; "
            "use scripts/run_fallback_gpu_acceptance.py"
        )
    start_ticks, command_sha256 = _process_identity(os.getpid())
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_controller_orchestrator_guard",
        "state": "active",
        "run_id": options.run_id,
        "sequence": sequence,
        "previous_guard_sha256": previous_hash,
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "guardian_ticket_sha256": ticket["manifest_sha256"],
        "orchestrator_pid": orchestrator_pid,
        "orchestrator_start_ticks": start_ticks,
        "orchestrator_command_sha256": command_sha256,
        "orchestrator_process_group_id": process_group_id,
        "orchestrator_session_id": os.getsid(0),
        "execution_arguments_sha256": invocation["execution_arguments_sha256"],
        "started_at": datetime.now(UTC).isoformat(),
    }
    guard = {**payload, "manifest_sha256": canonical_sha256(payload)}
    _write_append_only_json(path, guard)
    return path, guard


def _validate_orchestrator_guard(options: argparse.Namespace) -> Mapping[str, object]:
    supplied = options.orchestrator_guard
    guards = _guard_paths(options)
    if supplied is None or not guards or supplied.resolve() != guards[-1]:
        raise RuntimeError("internal controller stage lacks the latest append-only guard")
    guard_chain = _load_guard_chain(options)
    guard = guard_chain[-1]
    paths = _orchestration_paths(options)
    if paths.controller_takeover.exists() or paths.controller_takeover.is_symlink():
        raise RuntimeError(
            "fallback guardian revoked controller launch authority; resume is forbidden"
        )
    invocation, ticket = _load_orchestration_identity(options, paths)
    orchestrator_pid = guard.get("orchestrator_pid")
    if (
        any(
            item.get("orchestration_invocation_sha256") != invocation["manifest_sha256"]
            or item.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
            or item.get("execution_arguments_sha256") != invocation["execution_arguments_sha256"]
            for item in guard_chain
        )
        or guard.get("state") != "active"
        or guard.get("run_id") != options.run_id
        or guard.get("sequence") != len(guard_chain)
        or guard.get("execution_arguments_sha256") != invocation["execution_arguments_sha256"]
        or guard.get("orchestration_invocation_sha256") != invocation["manifest_sha256"]
        or guard.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
        or _guard_close_path(guards[-1]).exists()
        or isinstance(orchestrator_pid, bool)
        or not isinstance(orchestrator_pid, int)
        or orchestrator_pid <= 0
        or orchestrator_pid == os.getpid()
        or not _guard_process_is_live(guard)
    ):
        raise ValueError("orchestrator guard does not authorize this controller stage")
    return guard


def _internal_controller_command(
    options: argparse.Namespace,
    *,
    stage: str,
    output: Path,
    guard: Path | None,
) -> tuple[str, ...]:
    arguments = _controller_execution_arguments(options)
    command = [
        sys.executable,
        str(Path(cast(str, arguments["project_root"])) / "scripts/run_fallback_gpu_acceptance.py"),
        "--execute",
        "--controller-stage",
        stage,
        "--project-root",
        cast(str, arguments["project_root"]),
        "--output",
        str(output.resolve()),
    ]
    if guard is not None:
        command.extend(("--orchestrator-guard", str(guard.resolve())))
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
    for name in (
        "retry_amendment",
        "prior_fallback_failure",
        "second_recovery_overlay",
        "second_recovery_v3_result",
        "second_recovery_v3_incident",
    ):
        if name in arguments:
            command.extend((f"--{name.replace('_', '-')}", cast(str, arguments[name])))
    command.extend(("--port", str(arguments["port"])))
    return tuple(command)


def _load_controller_receipts(
    options: argparse.Namespace,
    *,
    invocation: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    guards = _load_guard_chain(options)
    guard_by_hash = {cast(str, guard["manifest_sha256"]): guard for guard in guards}
    previous_hash: str | None = None
    receipts: list[dict[str, object]] = []
    paths = _orchestration_paths(options)
    expected_outputs = {
        "prepare": paths.handoff_output,
        "recover-prepare": paths.handoff_output,
        "run": paths.result_output,
        "cleanup": paths.cleanup_output,
    }
    for sequence, path in enumerate(_controller_receipt_paths(options), start=1):
        receipt = _load_hashed_object(
            path,
            expected_kind="fallback_internal_controller_receipt",
        )
        stage = receipt.get("controller_stage")
        bound_guard = guard_by_hash.get(cast(str, receipt.get("orchestrator_guard_sha256")))
        if not isinstance(stage, str) or stage not in expected_outputs:
            raise ValueError("internal controller receipt has an invalid stage")
        if bound_guard is None:
            raise ValueError("internal controller receipt has no bound orchestrator guard")
        expected_command = _internal_controller_command(
            options,
            stage=stage,
            output=expected_outputs[stage],
            guard=_guard_path(options, cast(int, bound_guard["sequence"])),
        )
        if (
            receipt.get("run_id") != options.run_id
            or receipt.get("sequence") != sequence
            or receipt.get("previous_controller_receipt_sha256") != previous_hash
            or receipt.get("orchestration_invocation_sha256") != invocation.get("manifest_sha256")
            or receipt.get("execution_arguments_sha256")
            != invocation.get("execution_arguments_sha256")
            or receipt.get("controller_output") != str(expected_outputs[stage].resolve())
            or receipt.get("controller_command_sha256") != _raw_command_sha256(expected_command)
            or receipt.get("controller_process_group_id")
            != bound_guard.get("orchestrator_process_group_id")
            or receipt.get("controller_session_id") != bound_guard.get("orchestrator_session_id")
        ):
            raise ValueError("internal controller receipt changed its bound identity")
        previous_hash = cast(str, receipt["manifest_sha256"])
        receipts.append(receipt)
    return tuple(receipts)


def _register_internal_controller(
    options: argparse.Namespace,
    *,
    guard: Mapping[str, object],
    invocation: Mapping[str, object],
) -> dict[str, object]:
    paths = _orchestration_paths(options)
    if paths.controller_takeover.exists() or paths.controller_takeover.is_symlink():
        raise RuntimeError("guardian takeover forbids a new internal controller")
    prior = _load_controller_receipts(options, invocation=invocation)
    stage = cast(str, options.controller_stage)
    expected_outputs = {
        "prepare": paths.handoff_output,
        "recover-prepare": paths.handoff_output,
        "run": paths.result_output,
        "cleanup": paths.cleanup_output,
    }
    if stage not in expected_outputs:
        raise RuntimeError("only scientific internal controllers may register")
    expected_command = _internal_controller_command(
        options,
        stage=stage,
        output=expected_outputs[stage],
        guard=cast(Path, options.orchestrator_guard),
    )
    pid = os.getpid()
    start_ticks, observed_command_sha256 = _process_identity(pid)
    expected_command_sha256 = _raw_command_sha256(expected_command)
    if observed_command_sha256 != expected_command_sha256:
        raise RuntimeError("internal controller process command differs from authorization")
    process_group_id = os.getpgrp()
    session_id = os.getsid(0)
    if process_group_id != guard.get("orchestrator_process_group_id") or session_id != guard.get(
        "orchestrator_session_id"
    ):
        raise RuntimeError("internal controller escaped its bound orchestrator group")
    sequence = len(prior) + 1
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_internal_controller_receipt",
        "run_id": options.run_id,
        "sequence": sequence,
        "previous_controller_receipt_sha256": (None if not prior else prior[-1]["manifest_sha256"]),
        "controller_stage": stage,
        "controller_output": str(expected_outputs[stage].resolve()),
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "orchestrator_guard_sha256": guard["manifest_sha256"],
        "execution_arguments_sha256": invocation["execution_arguments_sha256"],
        "controller_pid": pid,
        "controller_start_ticks": start_ticks,
        "controller_command_sha256": observed_command_sha256,
        "controller_process_group_id": process_group_id,
        "controller_session_id": session_id,
        "registered_at": datetime.now(UTC).isoformat(),
    }
    receipt = {**payload, "manifest_sha256": canonical_sha256(payload)}
    _write_append_only_json(_controller_receipt_path(options, sequence), receipt)
    # Close the race in which a guardian revokes authority while this process is
    # publishing its identity.  The receipt remains as exact kill authority.
    if paths.controller_takeover.exists() or paths.controller_takeover.is_symlink():
        raise RuntimeError("guardian takeover began during controller registration")
    return receipt


def _bind_controller_stage_result(
    result: Mapping[str, object],
    *,
    stage: str,
    guard: Mapping[str, object],
    invocation: Mapping[str, object],
    controller_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    original = {key: value for key, value in result.items() if key != "manifest_sha256"}
    if result.get("manifest_sha256") != canonical_sha256(original):
        raise ValueError("controller result was not canonical before orchestration binding")
    if result.get("run_id") != invocation.get("run_id"):
        raise ValueError("controller result belongs to another fallback run")
    payload = {
        **original,
        "orchestration_controller_stage": stage,
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "orchestrator_guard_sha256": guard["manifest_sha256"],
        "execution_arguments_sha256": invocation["execution_arguments_sha256"],
        "controller_process_receipt_sha256": (
            None if controller_receipt is None else controller_receipt["manifest_sha256"]
        ),
        "unbound_controller_result_sha256": result["manifest_sha256"],
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _validate_controller_stage_result(
    path: Path,
    *,
    options: argparse.Namespace,
    invocation: Mapping[str, object],
    expected_stage: str,
) -> dict[str, object]:
    expected_kind = "phase1_fallback_micro_pilot_result"
    if expected_stage in {"prepare", "recover-prepare"}:
        expected_kind = "phase1_fallback_controller_restart_handoff"
    elif expected_stage == "cleanup":
        expected_kind = "phase1_fallback_orphan_cleanup_result"
    result = _load_hashed_object(path, expected_kind=expected_kind)
    guards = _load_guard_chain(options)
    guard_hashes = {guard["manifest_sha256"] for guard in guards}
    controller_receipts = _load_controller_receipts(
        options,
        invocation=invocation,
    )
    controller_by_hash = {receipt["manifest_sha256"]: receipt for receipt in controller_receipts}
    controller_receipt = controller_by_hash.get(result.get("controller_process_receipt_sha256"))
    if (
        result.get("run_id") != options.run_id
        or result.get("orchestration_controller_stage") != expected_stage
        or result.get("orchestration_invocation_sha256") != invocation.get("manifest_sha256")
        or result.get("execution_arguments_sha256") != invocation.get("execution_arguments_sha256")
        or result.get("orchestrator_guard_sha256") not in guard_hashes
        or (
            controller_receipts
            and (
                controller_receipt is None
                or controller_receipt.get("controller_stage") != expected_stage
                or controller_receipt.get("orchestrator_guard_sha256")
                != result.get("orchestrator_guard_sha256")
            )
        )
    ):
        raise ValueError("controller stage result is not bound to this orchestration")
    unbound_hash = result.get("unbound_controller_result_sha256")
    rebound = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "manifest_sha256",
            "orchestration_controller_stage",
            "orchestration_invocation_sha256",
            "orchestrator_guard_sha256",
            "execution_arguments_sha256",
            "controller_process_receipt_sha256",
            "unbound_controller_result_sha256",
        }
    }
    if unbound_hash != canonical_sha256(rebound):
        raise ValueError("controller stage result changed after its inner result was bound")
    checkpoint_path = Path(options.checkpoint)
    if checkpoint_path.is_file():
        checkpoint = _load_object(checkpoint_path)
        if result.get("execution_hash") != checkpoint.get("execution_hash"):
            raise ValueError("controller stage result changed its runner execution hash")
    if expected_stage in {"prepare", "recover-prepare"} and (
        result.get("model_service_left_live_for_controller_restart") is not True
        or result.get("physical_service_live") is not True
        or result.get("vllm_service_stopped") is not False
    ):
        raise ValueError("controller handoff does not certify the one live service")
    if expected_stage == "cleanup" and result.get("physical_shutdown_verified") is not True:
        raise ValueError("cleanup result does not certify physical shutdown")
    if expected_stage == "run" and (
        result.get("vllm_service_stopped") is not True
        or result.get("physical_service_live") is not False
    ):
        raise ValueError("run result does not certify physical shutdown")
    return result


def _guardian_ready_paths(paths: _FallbackOrchestrationPaths) -> tuple[Path, ...]:
    prefix = paths.checkpoint.name + ".guardian-ready-"
    candidates: list[tuple[int, Path]] = []
    if paths.checkpoint.parent.exists():
        for item in paths.checkpoint.parent.iterdir():
            if not item.name.startswith(prefix):
                continue
            suffix = item.name.removeprefix(prefix)
            if len(suffix) != 11 or not suffix.endswith(".json") or not suffix[:6].isdigit():
                raise RuntimeError("guardian-ready inventory contains an invalid entry")
            if item.is_symlink() or not item.is_file():
                raise RuntimeError("guardian-ready receipt must be a regular file")
            candidates.append((int(suffix[:6]), item.resolve()))
    candidates.sort()
    if [sequence for sequence, _ in candidates] != list(range(1, len(candidates) + 1)):
        raise RuntimeError("guardian-ready sequence is not contiguous")
    return tuple(path for _, path in candidates)


def _guardian_ready_path(paths: _FallbackOrchestrationPaths, sequence: int) -> Path:
    return paths.checkpoint.with_name(f"{paths.checkpoint.name}.guardian-ready-{sequence:06d}.json")


def _live_guardian_receipt(
    paths: _FallbackOrchestrationPaths,
    *,
    invocation: Mapping[str, object],
    ticket: Mapping[str, object],
) -> dict[str, object] | None:
    for ready_path in reversed(_guardian_ready_paths(paths)):
        ready = _load_hashed_object(
            ready_path,
            expected_kind="fallback_service_guardian_ready",
        )
        pid = ready.get("guardian_pid")
        start_ticks = ready.get("guardian_start_ticks")
        command_sha256 = ready.get("guardian_process_command_sha256")
        if (
            ready.get("run_id") != invocation.get("run_id")
            or ready.get("orchestration_invocation_sha256") != invocation.get("manifest_sha256")
            or ready.get("guardian_ticket_sha256") != ticket.get("manifest_sha256")
            or ready.get("guardian_command_sha256") != ticket.get("guardian_command_sha256")
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or isinstance(start_ticks, bool)
            or not isinstance(start_ticks, int)
            or not isinstance(command_sha256, str)
        ):
            raise ValueError("guardian-ready receipt changed its invocation identity")
        try:
            os.kill(pid, 0)
            observed_ticks, observed_command = _process_identity(pid)
        except (OSError, RuntimeError, ValueError):
            continue
        if observed_ticks == start_ticks and observed_command == command_sha256:
            return ready
    return None


def _launch_guardian_process(
    options: argparse.Namespace,
    *,
    paths: _FallbackOrchestrationPaths,
) -> subprocess.Popen[bytes]:
    command = _guardian_controller_command(
        options,
        output=paths.guardian_result,
        ticket=paths.guardian_ticket,
    )
    paths.guardian_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(paths.guardian_log, flags, 0o600)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        os.close(descriptor)


def _ensure_guardian_running(
    options: argparse.Namespace,
    *,
    paths: _FallbackOrchestrationPaths,
    invocation: Mapping[str, object],
    ticket: Mapping[str, object],
) -> Mapping[str, object]:
    existing = _live_guardian_receipt(paths, invocation=invocation, ticket=ticket)
    if existing is not None:
        return existing
    process = _launch_guardian_process(options, paths=paths)
    deadline = time.monotonic() + FALLBACK_GUARDIAN_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        ready = _live_guardian_receipt(paths, invocation=invocation, ticket=ticket)
        if ready is not None:
            return ready
        return_code = process.poll()
        if return_code is not None:
            # A replacement may lose the guardian lock to the still-valid first
            # guardian. Recheck its receipt before treating this as a failure.
            ready = _live_guardian_receipt(paths, invocation=invocation, ticket=ticket)
            if ready is not None:
                return ready
            raise RuntimeError(f"fallback guardian exited before readiness ({return_code})")
        time.sleep(0.05)
    raise RuntimeError("fallback guardian did not publish readiness before its watchdog")


def _validate_guardian_ticket(options: argparse.Namespace) -> dict[str, object]:
    paths = _orchestration_paths(options)
    supplied = options.guardian_ticket
    if supplied is None or supplied.resolve() != paths.guardian_ticket:
        raise RuntimeError("guardian stage requires the exact immutable ticket")
    invocation, ticket = _load_orchestration_identity(options, paths)
    if ticket.get("orchestration_invocation_sha256") != invocation.get(
        "manifest_sha256"
    ) or canonical_sha256(
        _guardian_controller_command(
            options,
            output=paths.guardian_result,
            ticket=paths.guardian_ticket,
        )
    ) != ticket.get("guardian_command_sha256"):
        raise ValueError("guardian ticket does not bind this invocation command")
    return ticket


def _control_process_group_alive(process_group_id: int) -> bool:
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                close = stat.rfind(")")
                fields = stat[close + 2 :].split()
                state = fields[0]
                observed_group = int(fields[2])
            except (OSError, RuntimeError, ValueError, IndexError):
                continue
            if observed_group == process_group_id and state != "Z":
                return True
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_bound_control_groups(
    options: argparse.Namespace,
    *,
    invocation: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Terminate only dedicated groups with a live, exact persisted identity."""

    guards = _load_guard_chain(options)
    receipts = _load_controller_receipts(options, invocation=invocation)
    receipts_by_guard: dict[str, list[Mapping[str, object]]] = {}
    for receipt in receipts:
        receipts_by_guard.setdefault(cast(str, receipt["orchestrator_guard_sha256"]), []).append(
            receipt
        )
    outcomes: list[dict[str, object]] = []
    for guard in guards:
        guard_hash = cast(str, guard["manifest_sha256"])
        process_group_id = guard.get("orchestrator_process_group_id")
        orchestrator_pid = guard.get("orchestrator_pid")
        if (
            isinstance(process_group_id, bool)
            or not isinstance(process_group_id, int)
            or isinstance(orchestrator_pid, bool)
            or not isinstance(orchestrator_pid, int)
            or process_group_id != orchestrator_pid
        ):
            raise ValueError("guardian takeover requires a dedicated bound orchestrator group")
        identities: tuple[tuple[Mapping[str, object], str], ...] = (
            (guard, "orchestrator"),
            *tuple((receipt, "controller") for receipt in receipts_by_guard.get(guard_hash, [])),
        )
        live_before = tuple(
            prefix
            for identity, prefix in identities
            if _exact_bound_process_is_live(identity, prefix=prefix)
        )
        outcome: dict[str, object] = {
            "orchestrator_guard_sha256": guard_hash,
            "process_group_id": process_group_id,
            "exact_live_identity_count_before_signal": len(live_before),
            "sigterm_sent": False,
            "sigkill_sent": False,
            "process_group_absent": not _control_process_group_alive(process_group_id),
            "numeric_group_live_without_bound_identity": False,
            "bound_controller_allocation_absent": False,
        }
        if not live_before:
            # A reused numeric PGID is not authority to signal unrelated work.
            # Exact PID/start/argv/session absence is the relevant proof for an
            # old guard whose dedicated group no longer contains a bound owner.
            outcome["numeric_group_live_without_bound_identity"] = _control_process_group_alive(
                process_group_id
            )
            outcome["bound_controller_allocation_absent"] = True
            outcomes.append(outcome)
            continue

        # Reverify immediately before group signalling so PID reuse or a changed
        # argv cannot turn a stale receipt into authority over another process.
        if not any(
            _exact_bound_process_is_live(identity, prefix=prefix) for identity, prefix in identities
        ):
            outcome["process_group_absent"] = not _control_process_group_alive(process_group_id)
            if outcome["process_group_absent"] is not True:
                outcome["numeric_group_live_without_bound_identity"] = True
            outcome["bound_controller_allocation_absent"] = True
            outcomes.append(outcome)
            continue
        try:
            os.killpg(process_group_id, signal.SIGTERM)
            outcome["sigterm_sent"] = True
        except ProcessLookupError:
            pass
        term_deadline = time.monotonic() + FALLBACK_CONTROL_GROUP_TERM_SECONDS
        while _control_process_group_alive(process_group_id) and time.monotonic() < term_deadline:
            time.sleep(0.02)
        if _control_process_group_alive(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
                outcome["sigkill_sent"] = True
            except ProcessLookupError:
                pass
        kill_deadline = time.monotonic() + FALLBACK_CONTROL_GROUP_KILL_SECONDS
        while _control_process_group_alive(process_group_id) and time.monotonic() < kill_deadline:
            time.sleep(0.02)
        outcome["process_group_absent"] = not _control_process_group_alive(process_group_id)
        outcome["bound_controller_allocation_absent"] = outcome["process_group_absent"]
        outcomes.append(outcome)
    return tuple(outcomes)


def _revoke_controller_authority(
    options: argparse.Namespace,
    *,
    paths: _FallbackOrchestrationPaths,
    invocation: Mapping[str, object],
    ticket: Mapping[str, object],
    trigger: str,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if trigger not in {"hard_stop_deadline", "orchestrator_lost"}:
        raise ValueError("controller takeover requires a terminal guardian trigger")
    if paths.controller_takeover.exists() or paths.controller_takeover.is_symlink():
        takeover = _load_hashed_object(
            paths.controller_takeover,
            expected_kind="fallback_guardian_controller_takeover",
        )
        if (
            takeover.get("run_id") != options.run_id
            or takeover.get("orchestration_invocation_sha256") != invocation.get("manifest_sha256")
            or takeover.get("guardian_ticket_sha256") != ticket.get("manifest_sha256")
            or takeover.get("trigger") not in {"hard_stop_deadline", "orchestrator_lost"}
        ):
            raise ValueError("guardian takeover receipt changed invocation identity")
    else:
        guards = _load_guard_chain(options)
        controller_receipts = _load_controller_receipts(
            options,
            invocation=invocation,
        )
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "fallback_guardian_controller_takeover",
            "run_id": options.run_id,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "execution_arguments_sha256": invocation["execution_arguments_sha256"],
            "trigger": trigger,
            "latest_orchestrator_guard_sha256": (
                None if not guards else guards[-1]["manifest_sha256"]
            ),
            "controller_receipt_sha256s_before_takeover": [
                receipt["manifest_sha256"] for receipt in controller_receipts
            ],
            "controller_launch_authority_revoked_at": datetime.now(UTC).isoformat(),
        }
        takeover = {**payload, "manifest_sha256": canonical_sha256(payload)}
        _write_append_only_json(paths.controller_takeover, takeover)
    outcomes = _terminate_bound_control_groups(
        options,
        invocation=invocation,
    )
    if any(outcome["bound_controller_allocation_absent"] is not True for outcome in outcomes):
        raise RuntimeError("guardian could not prove every signalled controller group absent")
    return takeover, outcomes


def _guardian_terminalize(
    runner: FallbackAcceptanceRunner,
    *,
    invocation: Mapping[str, object],
    ticket: Mapping[str, object],
    trigger: str,
    terminal_request_sha256: str | None,
    controller_takeover_sha256: str | None,
    control_group_outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    state: dict[str, object] | None = None
    if runner.checkpoint_path.is_file():
        state = _load_object(runner.checkpoint_path)
        if state.get("run_id") != runner.run_id:
            raise ValueError("guardian runner checkpoint belongs to another run")
    uptime: ServiceUptime | None = None
    adopted = False
    exact_service_journal = runner.ledger.latest_gpu_service_journal(runner._service_event_id)
    authoritative_lease = runner.service.read_authoritative_service_lease()
    exact_service_lease = (
        authoritative_lease is not None
        and authoritative_lease.get("session_id") == runner.run_id
        and authoritative_lease.get("accounting_session_id") == runner._service_event_id
        and authoritative_lease.get("configuration_hash")
        == runner.service.configuration.configuration_hash
    )
    service_may_have_started = (
        (state is not None and state.get("service_start_attempted") is True)
        or exact_service_journal is not None
        or exact_service_lease
    )
    if service_may_have_started:
        uptime, adopted = runner._stop_or_reconcile_service()
        if state is not None:
            state["service_adoption_failure_type"] = runner._last_service_adoption_failure_type
    terminal = runner._validate_terminal_service_accounting()
    unresolved = runner.ledger.unresolved_gpu_service_journals()
    physical_shutdown_verified = (
        runner.service.state is ServiceState.STOPPED
        and not unresolved
        and not runner.ledger.unresolved_gpu_allocations()
    )
    if not physical_shutdown_verified:
        raise RuntimeError("guardian could not verify terminal fallback service state")
    if state is not None:
        interrupted_call_id = state.get("active_call_id")
        if interrupted_call_id is not None:
            runner._terminalize_interrupted_call_lifecycle(state)
        if trigger != "terminal_request" or interrupted_call_id is not None:
            state["orphan_cleanup_completed"] = True
            state["active_call_id"] = None
            state["active_attempt"] = None
            state["failed_call_id"] = (
                state.get("failed_call_id") or interrupted_call_id or f"guardian-{trigger}"
            )
            runner._save(state)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_service_guardian_result",
        "run_id": runner.run_id,
        "orchestration_invocation_sha256": invocation["manifest_sha256"],
        "guardian_ticket_sha256": ticket["manifest_sha256"],
        "execution_arguments_sha256": invocation["execution_arguments_sha256"],
        "trigger": trigger,
        "terminal_request_sha256": terminal_request_sha256,
        "controller_takeover_sha256": controller_takeover_sha256,
        "control_group_outcomes": [dict(item) for item in control_group_outcomes],
        "checkpoint_service_adopted": adopted,
        "checkpoint_service_adoption_failed": (
            runner._last_service_adoption_failure_type is not None
        ),
        "checkpoint_service_adoption_failure_type": (runner._last_service_adoption_failure_type),
        "service_session_accounting_sha256": (
            None if terminal is None else canonical_sha256(asdict(terminal))
        ),
        "accounted_service_seconds": (None if terminal is None else terminal.service_seconds),
        "uptime_recovered": uptime is not None,
        "actual_allocated_gpu_seconds": runner.ledger.gpu_summary().total_allocated_seconds,
        "physical_shutdown_verified": physical_shutdown_verified,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _run_guardian(
    options: argparse.Namespace,
    *,
    runner: FallbackAcceptanceRunner,
    ticket: Mapping[str, object],
) -> dict[str, object]:
    paths = _orchestration_paths(options)
    invocation, expected_ticket = _load_orchestration_identity(options, paths)
    if ticket.get("manifest_sha256") != expected_ticket.get("manifest_sha256"):
        raise ValueError("guardian received another immutable ticket")
    descriptor = os.open(paths.guardian_lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another exact fallback guardian is already active") from exc
        start_ticks, command_sha256 = _process_identity(os.getpid())
        ready_sequence = len(_guardian_ready_paths(paths)) + 1
        ready_payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "fallback_service_guardian_ready",
            "run_id": options.run_id,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "guardian_command_sha256": ticket["guardian_command_sha256"],
            "guardian_pid": os.getpid(),
            "guardian_start_ticks": start_ticks,
            "guardian_process_command_sha256": command_sha256,
            "ready_at": datetime.now(UTC).isoformat(),
        }
        ready = {**ready_payload, "manifest_sha256": canonical_sha256(ready_payload)}
        _write_append_only_json(_guardian_ready_path(paths, ready_sequence), ready)
        hard_stop_at = datetime.fromisoformat(cast(str, ticket["hard_stop_at"]))
        resume_grace = float(ticket["resume_grace_seconds"])
        dead_since: datetime | None = None
        trigger: str | None = None
        terminal_request_hash: str | None = None
        while trigger is None:
            now = datetime.now(UTC)
            if paths.controller_takeover.exists() or paths.controller_takeover.is_symlink():
                takeover = _load_hashed_object(
                    paths.controller_takeover,
                    expected_kind="fallback_guardian_controller_takeover",
                )
                if (
                    takeover.get("run_id") != options.run_id
                    or takeover.get("orchestration_invocation_sha256")
                    != invocation["manifest_sha256"]
                    or takeover.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
                    or takeover.get("trigger") not in {"hard_stop_deadline", "orchestrator_lost"}
                ):
                    raise ValueError("guardian observed an invalid prior controller takeover")
                trigger = cast(str, takeover["trigger"])
                break
            if paths.terminal_request.exists():
                request = _load_hashed_object(
                    paths.terminal_request,
                    expected_kind="fallback_guardian_terminal_request",
                )
                if (
                    request.get("run_id") != options.run_id
                    or request.get("orchestration_invocation_sha256")
                    != invocation["manifest_sha256"]
                    or request.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
                ):
                    raise ValueError("guardian terminal request changed invocation identity")
                trigger = "terminal_request"
                terminal_request_hash = cast(str, request["manifest_sha256"])
                break
            guard_paths = _guard_paths(options)
            guard_chain = _load_guard_chain(options)
            latest_guard = None if not guard_paths else guard_chain[-1]
            if latest_guard is not None and (
                latest_guard.get("orchestration_invocation_sha256") != invocation["manifest_sha256"]
                or latest_guard.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
            ):
                raise ValueError("guardian observed an unbound orchestrator guard")
            if latest_guard is not None and _guard_process_is_live(latest_guard):
                dead_since = None
            elif dead_since is None:
                dead_since = now
            if now >= hard_stop_at:
                trigger = "hard_stop_deadline"
            elif dead_since is not None and (now - dead_since).total_seconds() >= resume_grace:
                trigger = "orchestrator_lost"
            if trigger is None:
                time.sleep(FALLBACK_GUARDIAN_POLL_SECONDS)
        takeover_hash: str | None = None
        control_group_outcomes: tuple[dict[str, object], ...] = ()
        if trigger in {"hard_stop_deadline", "orchestrator_lost"}:
            takeover, control_group_outcomes = _revoke_controller_authority(
                options,
                paths=paths,
                invocation=invocation,
                ticket=ticket,
                trigger=trigger,
            )
            takeover_hash = cast(str, takeover["manifest_sha256"])
        result = _guardian_terminalize(
            runner,
            invocation=invocation,
            ticket=ticket,
            trigger=cast(str, trigger),
            terminal_request_sha256=terminal_request_hash,
            controller_takeover_sha256=takeover_hash,
            control_group_outcomes=control_group_outcomes,
        )
        _write_append_only_json(paths.guardian_result, result)
        return result
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _request_guardian_terminal_verification(
    options: argparse.Namespace,
    *,
    paths: _FallbackOrchestrationPaths,
    invocation: Mapping[str, object],
    ticket: Mapping[str, object],
    guard: Mapping[str, object],
    controller_result_sha256: str | None,
) -> dict[str, object]:
    if paths.terminal_request.exists():
        request = _load_hashed_object(
            paths.terminal_request,
            expected_kind="fallback_guardian_terminal_request",
        )
        if (
            request.get("run_id") != options.run_id
            or request.get("orchestration_invocation_sha256") != invocation["manifest_sha256"]
            or request.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
            or request.get("controller_result_sha256") != controller_result_sha256
        ):
            raise ValueError("existing guardian terminal request changed its binding")
    else:
        request_payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "fallback_guardian_terminal_request",
            "run_id": options.run_id,
            "orchestration_invocation_sha256": invocation["manifest_sha256"],
            "guardian_ticket_sha256": ticket["manifest_sha256"],
            "orchestrator_guard_sha256": guard["manifest_sha256"],
            "controller_result_sha256": controller_result_sha256,
            "requested_at": datetime.now(UTC).isoformat(),
        }
        request = {**request_payload, "manifest_sha256": canonical_sha256(request_payload)}
        _write_append_only_json(paths.terminal_request, request)
    deadline = (
        time.monotonic() + FALLBACK_GUARDIAN_READY_TIMEOUT_SECONDS + 2 * (DEFAULT_SHUTDOWN_SECONDS)
    )
    while time.monotonic() < deadline:
        if paths.guardian_result.is_file():
            result = _load_hashed_object(
                paths.guardian_result,
                expected_kind="fallback_service_guardian_result",
            )
            if (
                result.get("run_id") != options.run_id
                or result.get("orchestration_invocation_sha256") != invocation["manifest_sha256"]
                or result.get("guardian_ticket_sha256") != ticket["manifest_sha256"]
                or result.get("terminal_request_sha256") != request["manifest_sha256"]
                or result.get("physical_shutdown_verified") is not True
            ):
                raise ValueError("guardian result is not exact terminal proof for this request")
            return result
        time.sleep(0.05)
    raise RuntimeError("guardian did not publish terminal service verification")


def _close_orchestrator_guard(
    guard_path: Path,
    guard: Mapping[str, object],
    *,
    prepare_return_code: int | None,
    run_return_code: int | None,
    cleanup_return_code: int | None,
    physical_shutdown_verified: bool,
    guardian_result_sha256: str | None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_controller_orchestrator_closed",
        "run_id": guard["run_id"],
        "orchestrator_guard_sha256": guard["manifest_sha256"],
        "prepare_return_code": prepare_return_code,
        "run_return_code": run_return_code,
        "cleanup_return_code": cleanup_return_code,
        "physical_shutdown_verified": physical_shutdown_verified,
        "guardian_result_sha256": guardian_result_sha256,
        "closed_at": datetime.now(UTC).isoformat(),
    }
    closed = {**payload, "manifest_sha256": canonical_sha256(payload)}
    _write_append_only_json(_guard_close_path(guard_path), closed)


def _orchestrate_controller_processes(options: argparse.Namespace) -> int:
    """Supervise controllers under one append-only identity and independent guardian."""

    paths = _orchestration_paths(options)
    if options.resume_orchestrator:
        invocation, ticket = _load_orchestration_identity(options, paths)
        if paths.controller_takeover.exists() or paths.controller_takeover.is_symlink():
            raise RuntimeError(
                "fallback guardian takeover already began; scientific resume is forbidden"
            )
        if paths.terminal_request.exists():
            raise RuntimeError(
                "fallback guardian terminalization already began; scientific resume is forbidden"
            )
        if paths.guardian_result.exists():
            prior_guardian = _load_hashed_object(
                paths.guardian_result,
                expected_kind="fallback_service_guardian_result",
            )
            raise RuntimeError(
                "fallback orchestration is already terminal and cannot resume: "
                f"guardian_verified={prior_guardian.get('physical_shutdown_verified')!r}"
            )
    else:
        invocation, ticket = _new_orchestration_identity(options, paths)
    guard_path, guard = _create_orchestrator_guard(
        options,
        invocation=invocation,
        ticket=ticket,
    )
    _ensure_guardian_running(
        options,
        paths=paths,
        invocation=invocation,
        ticket=ticket,
    )
    prepare_return_code: int | None = None
    run_return_code: int | None = None
    cleanup_return_code: int | None = None
    physical_shutdown_verified = False
    guardian_result_hash: str | None = None
    controller_result_hash: str | None = None
    raised: BaseException | None = None
    try:
        if options.resume_orchestrator and paths.cleanup_output.exists():
            cleanup_result = _validate_controller_stage_result(
                paths.cleanup_output,
                options=options,
                invocation=invocation,
                expected_stage="cleanup",
            )
            controller_result_hash = cast(str, cleanup_result["manifest_sha256"])
            raise RuntimeError("fallback cleanup already began; scientific resume is forbidden")
        if options.resume_orchestrator and paths.handoff_output.exists():
            prior_handoff = _load_object(paths.handoff_output)
            prior_stage = prior_handoff.get("orchestration_controller_stage")
            if prior_stage not in {"prepare", "recover-prepare"}:
                raise ValueError("fallback handoff has an invalid controller stage")
            _validate_controller_stage_result(
                paths.handoff_output,
                options=options,
                invocation=invocation,
                expected_stage=cast(str, prior_stage),
            )
        if paths.result_output.exists():
            if not options.resume_orchestrator:
                raise FileExistsError("fresh fallback result output already exists")
            result = _validate_controller_stage_result(
                paths.result_output,
                options=options,
                invocation=invocation,
                expected_stage="run",
            )
            run_return_code = 0 if result.get("phase1_gate_passed") is True else 2
            controller_result_hash = cast(str, result["manifest_sha256"])
        else:
            preparation_stage: str | None = None
            if not paths.checkpoint.exists():
                preparation_stage = "prepare"
            else:
                state = _load_object(paths.checkpoint)
                if state.get("run_id") != options.run_id:
                    raise ValueError("fallback resume checkpoint belongs to another run")
                if (
                    state.get("active_call_id") is not None
                    or state.get("failed_call_id") is not None
                ):
                    raise RuntimeError(
                        "interrupted fallback call is terminal under intention-to-treat"
                    )
                if state.get("service_start_attempted") is not True:
                    preparation_stage = "prepare"
                elif state.get("controller_handoff_complete") is not True:
                    preparation_stage = "recover-prepare"
            if preparation_stage is not None:
                if paths.handoff_output.exists() or paths.handoff_output.is_symlink():
                    raise FileExistsError(
                        "fallback handoff output exists before its selected controller stage"
                    )
                prepared = subprocess.run(
                    _internal_controller_command(
                        options,
                        stage=preparation_stage,
                        output=paths.handoff_output,
                        guard=guard_path,
                    ),
                    check=False,
                )
                prepare_return_code = prepared.returncode
                if prepare_return_code != 0:
                    raise RuntimeError("fallback prepare controller failed")
                _validate_controller_stage_result(
                    paths.handoff_output,
                    options=options,
                    invocation=invocation,
                    expected_stage=preparation_stage,
                )
            executed = subprocess.run(
                _internal_controller_command(
                    options,
                    stage="run",
                    output=paths.result_output,
                    guard=guard_path,
                ),
                check=False,
            )
            run_return_code = executed.returncode
            result = _validate_controller_stage_result(
                paths.result_output,
                options=options,
                invocation=invocation,
                expected_stage="run",
            )
            expected_return_code = 0 if result.get("phase1_gate_passed") is True else 2
            if run_return_code != expected_return_code:
                raise RuntimeError("fallback run return code disagrees with its bound result")
            controller_result_hash = cast(str, result["manifest_sha256"])
        guardian_result = _request_guardian_terminal_verification(
            options,
            paths=paths,
            invocation=invocation,
            ticket=ticket,
            guard=guard,
            controller_result_sha256=controller_result_hash,
        )
        guardian_result_hash = cast(str, guardian_result["manifest_sha256"])
        physical_shutdown_verified = True
        return run_return_code
    except BaseException as exc:
        raised = exc
        raise
    finally:
        if not physical_shutdown_verified:
            try:
                if paths.cleanup_output.exists():
                    if not options.resume_orchestrator:
                        raise FileExistsError("fresh fallback cleanup output already exists")
                    cleanup_result = _validate_controller_stage_result(
                        paths.cleanup_output,
                        options=options,
                        invocation=invocation,
                        expected_stage="cleanup",
                    )
                    cleanup_return_code = 0
                else:
                    cleanup = subprocess.run(
                        _internal_controller_command(
                            options,
                            stage="cleanup",
                            output=paths.cleanup_output,
                            guard=guard_path,
                        ),
                        check=False,
                    )
                    cleanup_return_code = cleanup.returncode
                    cleanup_result = _validate_controller_stage_result(
                        paths.cleanup_output,
                        options=options,
                        invocation=invocation,
                        expected_stage="cleanup",
                    )
                    if cleanup.returncode != 0:
                        raise RuntimeError("fallback cleanup controller returned failure")
                controller_result_hash = cast(str, cleanup_result["manifest_sha256"])
            except BaseException:
                # The independent guardian remains responsible for exact lease
                # adoption/reconciliation even if a cleanup controller cannot
                # publish its receipt.
                cleanup_return_code = 2 if cleanup_return_code is None else cleanup_return_code
            try:
                _ensure_guardian_running(
                    options,
                    paths=paths,
                    invocation=invocation,
                    ticket=ticket,
                )
                guardian_result = _request_guardian_terminal_verification(
                    options,
                    paths=paths,
                    invocation=invocation,
                    ticket=ticket,
                    guard=guard,
                    controller_result_sha256=controller_result_hash,
                )
                guardian_result_hash = cast(str, guardian_result["manifest_sha256"])
                physical_shutdown_verified = True
            except BaseException:
                physical_shutdown_verified = False
            if not physical_shutdown_verified and raised is None:
                raise RuntimeError("fallback orchestrator could not verify orphan cleanup")
        _close_orchestrator_guard(
            guard_path,
            guard,
            prepare_return_code=prepare_return_code,
            run_return_code=run_return_code,
            cleanup_return_code=cleanup_return_code,
            physical_shutdown_verified=physical_shutdown_verified,
            guardian_result_sha256=guardian_result_hash,
        )


def _orchestrator_status(options: argparse.Namespace) -> dict[str, object]:
    if options.run_id is None or options.checkpoint is None:
        raise SystemExit("--status requires --run-id and --checkpoint")
    paths = _orchestration_paths(options)
    invocation = (
        None
        if not paths.invocation.is_file()
        else _load_hashed_object(
            paths.invocation,
            expected_kind="fallback_controller_orchestration_invocation",
        )
    )
    if invocation is not None:
        _require_execution_arguments(
            options,
            operation="status with a durable orchestration invocation",
        )
        expected_arguments_hash = canonical_sha256(_controller_execution_arguments(options))
        if (
            invocation.get("run_id") != options.run_id
            or invocation.get("execution_arguments_sha256") != expected_arguments_hash
            or invocation.get("result_output") != str(paths.result_output)
            or invocation.get("handoff_output") != str(paths.handoff_output)
            or invocation.get("cleanup_output") != str(paths.cleanup_output)
            or invocation.get("checkpoint") != str(paths.checkpoint)
        ):
            raise ValueError("status arguments differ from the durable orchestration identity")
    guard_paths = _guard_paths(options)
    guards = _load_guard_chain(options)
    latest = None if not guards else guards[-1]
    controller_receipts = (
        () if invocation is None else _load_controller_receipts(options, invocation=invocation)
    )
    live_controller_count = sum(
        _exact_bound_process_is_live(receipt, prefix="controller")
        for receipt in controller_receipts
    )
    latest_group = None if latest is None else latest.get("orchestrator_process_group_id")
    latest_control_group_live = (
        False
        if isinstance(latest_group, bool) or not isinstance(latest_group, int)
        else _control_process_group_alive(latest_group)
    )
    checkpoint = None if not paths.checkpoint.is_file() else _load_object(paths.checkpoint)
    guardian_result = (
        None
        if not paths.guardian_result.is_file()
        else _load_hashed_object(
            paths.guardian_result,
            expected_kind="fallback_service_guardian_result",
        )
    )
    ledger_seconds = None
    unresolved_services = None
    unresolved_allocations = None
    if options.ledger is not None and options.ledger.is_file():
        with ReadOnlyLedger(options.ledger) as ledger:
            ledger_seconds = ledger.gpu_summary().total_allocated_seconds
            unresolved_services = len(ledger.unresolved_gpu_service_journals())
            unresolved_allocations = len(ledger.unresolved_gpu_allocations())
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fallback_controller_orchestration_status",
        "run_id": options.run_id,
        "invocation_present": invocation is not None,
        "invocation_sha256": (None if invocation is None else invocation["manifest_sha256"]),
        "hard_stop_at": None if invocation is None else invocation["hard_stop_at"],
        "orchestrator_invocation_count": len(guards),
        "latest_orchestrator_guard_sha256": (None if latest is None else latest["manifest_sha256"]),
        "latest_orchestrator_live": (False if latest is None else _guard_process_is_live(latest)),
        "latest_control_group_live": latest_control_group_live,
        "latest_orchestrator_closed": (
            False if not guard_paths else _guard_close_path(guard_paths[-1]).is_file()
        ),
        "internal_controller_receipt_count": len(controller_receipts),
        "live_internal_controller_count": live_controller_count,
        "controller_launch_authority_revoked": paths.controller_takeover.is_file(),
        "guardian_ready_receipt_count": len(_guardian_ready_paths(paths)),
        "guardian_terminal": guardian_result is not None,
        "guardian_physical_shutdown_verified": (
            None if guardian_result is None else guardian_result["physical_shutdown_verified"]
        ),
        "checkpoint_present": checkpoint is not None,
        "service_start_attempted": (
            None if checkpoint is None else checkpoint.get("service_start_attempted")
        ),
        "controller_handoff_complete": (
            None if checkpoint is None else checkpoint.get("controller_handoff_complete")
        ),
        "active_call_id": None if checkpoint is None else checkpoint.get("active_call_id"),
        "failed_call_id": None if checkpoint is None else checkpoint.get("failed_call_id"),
        "result_present": paths.result_output.is_file(),
        "handoff_present": paths.handoff_output.is_file(),
        "cleanup_present": paths.cleanup_output.is_file(),
        "actual_allocated_gpu_seconds": ledger_seconds,
        "unresolved_gpu_allocation_count": unresolved_allocations,
        "unresolved_gpu_service_count": unresolved_services,
        "resume_allowed": (
            invocation is not None
            and guardian_result is None
            and not paths.terminal_request.exists()
            and not paths.controller_takeover.exists()
            and (latest is None or not _guard_process_is_live(latest))
            and not latest_control_group_live
            and live_controller_count == 0
            and (
                checkpoint is None
                or (
                    checkpoint.get("active_call_id") is None
                    and checkpoint.get("failed_call_id") is None
                )
            )
        ),
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def _aware_datetime_argument(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--build-second-recovery-overlay", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument(
        "--controller-stage",
        choices=("orchestrate", "prepare", "recover-prepare", "run", "cleanup", "guardian"),
    )
    parser.add_argument("--orchestrator-guard", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--guardian-ticket", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-orchestrator", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--primary-result", type=Path)
    parser.add_argument("--activation-certificate", type=Path)
    parser.add_argument("--cache-replacement-receipt", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--shared-cache", type=Path)
    parser.add_argument("--verified-model-manifest", type=Path)
    parser.add_argument("--source-association", type=Path)
    parser.add_argument("--retry-amendment", type=Path)
    parser.add_argument("--prior-fallback-failure", type=Path)
    parser.add_argument("--second-recovery-overlay", type=Path)
    parser.add_argument("--second-recovery-v3-result", type=Path)
    parser.add_argument("--second-recovery-v3-incident", type=Path)
    parser.add_argument("--restricted-output-root", type=Path)
    parser.add_argument(
        "--second-recovery-authorization-status",
        choices=("proposed", "authorized"),
        default="proposed",
    )
    parser.add_argument("--second-recovery-authorization-basis")
    parser.add_argument(
        "--second-recovery-authorized-at",
        type=_aware_datetime_argument,
    )
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quota-root", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(arguments)


def establish_fallback_orchestrator_process_group(
    arguments: Sequence[str] | None = None,
) -> None:
    """Place only the public orchestrator entry in a dedicated control group.

    Internal controllers deliberately inherit this group, while the guardian and
    vLLM each start their own sessions.  That lets the guardian revoke all
    controller launch authority without ever signalling an unrelated shell or
    the separately bound model-service group.
    """

    options = parse_arguments(arguments)
    if not options.execute or options.controller_stage != "orchestrate":
        return
    pid = os.getpid()
    if os.getpgrp() != pid:
        try:
            os.setpgid(0, 0)
        except OSError as exc:
            raise RuntimeError(
                "cannot establish the fallback orchestrator's dedicated process group"
            ) from exc
    if os.getpgrp() != pid:
        raise RuntimeError(
            "fallback orchestrator did not become its dedicated process-group leader"
        )


def _require_second_recovery_builder_arguments(options: argparse.Namespace) -> None:
    required = (
        "run_id",
        "primary_result",
        "activation_certificate",
        "snapshot",
        "source_association",
        "retry_amendment",
        "prior_fallback_failure",
        "second_recovery_v3_result",
        "second_recovery_v3_incident",
        "restricted_output_root",
        "ledger",
    )
    missing = [name for name in required if getattr(options, name) is None]
    if missing:
        raise SystemExit(
            "second recovery overlay builder requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    authorized = options.second_recovery_authorization_status == "authorized"
    if authorized and (
        options.second_recovery_authorization_basis is None
        or options.second_recovery_authorized_at is None
    ):
        raise SystemExit(
            "authorized overlay construction requires "
            "--second-recovery-authorization-basis and "
            "--second-recovery-authorized-at"
        )
    if not authorized and options.second_recovery_authorized_at is not None:
        raise SystemExit("a proposed overlay cannot have an authorization time")


def _build_second_recovery_overlay_from_cli(
    options: argparse.Namespace,
    *,
    root: Path,
) -> dict[str, object]:
    """CPU-only CLI bridge; intentionally has no service or CUDA construction."""

    _require_second_recovery_builder_arguments(options)
    policy = FallbackModelPolicy.load(root / "configs/study/fallback_model.json")
    legacy_provenance_bridge = Phase1LegacyEvidenceProvenanceBridge.load(root)
    primary_result = _load_object(options.primary_result)
    activation = validate_fallback_activation_certificate(
        policy=policy,
        certificate=_load_object(options.activation_certificate),
        primary_result=primary_result,
    )
    source_association = validate_source_association(
        options.source_association,
        source_root=root,
    )
    tokenizer_manifest = capture_tokenizer_manifest(
        options.snapshot,
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(options.snapshot),
        local_files_only=True,
        trust_remote_code=False,
        revision=FALLBACK_MODEL_REVISION,
    )
    request = build_fallback_acceptance_request(
        root=root,
        call=fallback_pilot_calls(policy)[0],
        tokenizer=cast(PackingTokenizer, tokenizer),
        tokenizer_manifest=tokenizer_manifest,
        legacy_provenance_bridge=legacy_provenance_bridge,
    )
    with ReadOnlyLedger(options.ledger) as ledger:
        observed = ledger.gpu_summary()
    return build_second_fallback_recovery_overlay(
        root=root,
        output_path=options.output,
        restricted_output_root=options.restricted_output_root,
        v3_result_path=options.second_recovery_v3_result,
        v3_incident_path=options.second_recovery_v3_incident,
        prior_retry_amendment_path=options.retry_amendment,
        prior_retry_failure_path=options.prior_fallback_failure,
        run_id=options.run_id,
        policy=policy,
        activation_certificate=activation,
        primary_result=primary_result,
        limits=ResourceLimits.load(root / "configs/study/resource_limits.json"),
        source_association=source_association,
        source_association_path=options.source_association,
        retry_request=request,
        legacy_provenance_bridge=legacy_provenance_bridge,
        observed=observed,
        authorization_status=options.second_recovery_authorization_status,
        authorization_basis=options.second_recovery_authorization_basis,
        authorized_at=options.second_recovery_authorized_at,
        verify_decoder_compilation=True,
    )


def _require_execution_arguments(
    options: argparse.Namespace,
    *,
    operation: str = "execution preflight",
) -> None:
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
            f"{operation} requires: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if (options.retry_amendment is None) != (options.prior_fallback_failure is None):
        raise SystemExit("--retry-amendment and --prior-fallback-failure must be supplied together")
    second_values = (
        options.second_recovery_overlay,
        options.second_recovery_v3_result,
        options.second_recovery_v3_incident,
    )
    if any(value is not None for value in second_values) and not all(
        value is not None for value in second_values
    ):
        raise SystemExit(
            "second recovery requires its overlay, v3 result, and v3 incident together"
        )
    if options.second_recovery_overlay is not None and options.retry_amendment is None:
        raise SystemExit("second recovery requires the original v3 amendment lineage")
    if options.resume_orchestrator and options.controller_stage != "orchestrate":
        raise SystemExit("--resume-orchestrator requires --controller-stage orchestrate")
    if options.guardian_ticket is not None and options.controller_stage != "guardian":
        raise SystemExit("--guardian-ticket is reserved for the guardian controller")
    if options.controller_stage == "guardian" and options.guardian_ticket is None:
        raise SystemExit("guardian controller requires --guardian-ticket")


def _validate_execution_preflight(
    options: argparse.Namespace,
    *,
    root: Path,
) -> dict[str, object]:
    """Validate the exact prospective run without allocating the GPU."""

    if options.controller_stage != "orchestrate":
        raise SystemExit("--validate-only requires --controller-stage orchestrate")
    policy_path = root / "configs/study/fallback_model.json"
    policy = FallbackModelPolicy.load(policy_path)
    legacy_provenance_bridge = Phase1LegacyEvidenceProvenanceBridge.load(root)
    primary_result = _load_object(options.primary_result)
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
        snapshot_path=options.snapshot,
        shared_cache=options.shared_cache,
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
    retry_request: GuidedJSONRequest | None = None
    if options.second_recovery_overlay is not None:
        from transformers import AutoTokenizer

        tokenizer_manifest = capture_tokenizer_manifest(
            options.snapshot,
            repository=FALLBACK_MODEL_REPOSITORY,
            revision=FALLBACK_MODEL_REVISION,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(options.snapshot),
            local_files_only=True,
            trust_remote_code=False,
            revision=FALLBACK_MODEL_REVISION,
        )
        retry_request = build_fallback_acceptance_request(
            root=root,
            call=fallback_pilot_calls(policy)[0],
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            legacy_provenance_bridge=legacy_provenance_bridge,
        )
    limits = ResourceLimits.load(root / "configs/study/resource_limits.json")
    storage_plan = StorageAllocationPlan.load(root / "configs/study/storage_phase_allocations.json")
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
    storage_report = storage.check(**storage_plan.reservation_for("phase_1").preflight_arguments())
    if not storage_report.allowed:
        raise StorageBudgetExceeded(storage_report)
    orchestration_paths = _orchestration_paths(options)
    checkpoint_paths = (
        orchestration_paths.checkpoint,
        orchestration_paths.service_checkpoint,
        orchestration_paths.invocation,
        orchestration_paths.guardian_ticket,
        orchestration_paths.guardian_result,
        orchestration_paths.terminal_request,
        orchestration_paths.controller_takeover,
    )
    if (
        any(path.exists() or path.is_symlink() for path in checkpoint_paths)
        or _guard_paths(options)
        or _controller_receipt_paths(options)
        or _guardian_ready_paths(orchestration_paths)
    ):
        raise RuntimeError("fallback recovery run already has durable controller state")

    inventory = GPUCallInventory.load(root / "configs/study/gpu_call_inventory.json")
    reference = forecast_gpu_schedule(inventory, limits=limits)
    with ReadOnlyLedger(options.ledger) as ledger:
        observed = ledger.gpu_summary()
        second_overlay: dict[str, object] | None = None
        second_predecessor: dict[str, object] | None = None
        second_incident: dict[str, object] | None = None
        if options.retry_amendment is None:
            validate_pre_fallback_gpu_accounting(primary_result, observed)
            next_watchdog = float(DEFAULT_FALLBACK_STARTUP_WATCHDOG_SECONDS)
            next_service_allocation_forecast = next_watchdog
            remaining = sum(
                max(
                    0,
                    row.count
                    - (
                        observed.service_session_count + 1
                        if row.call_class == "gpu_session_start"
                        else row.count
                        if row.call_class in NORMAL_ACCEPTANCE_CLASSES
                        else 0
                    ),
                )
                * row.forecast_p95_seconds
                for row in reference.rows
            )
            amendment_hash = None
            predecessor_hash = None
        elif options.second_recovery_overlay is not None:
            assert retry_request is not None
            second_overlay, second_predecessor, second_incident = (
                validate_second_fallback_recovery_overlay(
                    root=root,
                    overlay_path=options.second_recovery_overlay,
                    v3_result_path=cast(Path, options.second_recovery_v3_result),
                    v3_incident_path=cast(Path, options.second_recovery_v3_incident),
                    prior_retry_amendment_path=options.retry_amendment,
                    prior_retry_failure_path=cast(Path, options.prior_fallback_failure),
                    run_id=options.run_id,
                    policy=policy,
                    activation_certificate=activation,
                    primary_result=primary_result,
                    limits=limits,
                    source_association=source_association,
                    source_association_path=options.source_association,
                    retry_request=retry_request,
                    legacy_provenance_bridge=legacy_provenance_bridge,
                    observed=observed,
                    require_authorized=False,
                    verify_decoder_compilation=True,
                )
            )
            forecast = cast(Mapping[str, object], second_overlay["corrected_forecast"])
            next_watchdog = float(AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS)
            next_service_allocation_forecast = float(
                cast(
                    float,
                    forecast["additional_service_allocation_forecast_seconds"],
                )
            )
            remaining = float(
                cast(
                    float,
                    forecast["corrected_remaining_mandatory_forecast_seconds"],
                )
            )
            amendment_hash = SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256
            original_predecessor = _validated_manifest_object(
                cast(Path, options.prior_fallback_failure),
                expected_kind="phase1_fallback_micro_pilot_result",
            )
            predecessor_hash = cast(str, original_predecessor["manifest_sha256"])
        else:
            amendment, predecessor = validate_fallback_service_retry_amendment(
                root=root,
                amendment_path=options.retry_amendment,
                prior_failure_path=cast(Path, options.prior_fallback_failure),
                run_id=options.run_id,
                policy=policy,
                activation_certificate=activation,
                primary_result=primary_result,
                limits=limits,
                observed=observed,
            )
            forecast = cast(Mapping[str, object], amendment["forecast"])
            next_watchdog = float(AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS)
            next_service_allocation_forecast = next_watchdog
            remaining = float(cast(float, forecast["remaining_mandatory_forecast_seconds"]))
            amendment_hash = amendment["manifest_sha256"]
            predecessor_hash = predecessor["manifest_sha256"]
        actual = observed.total_allocated_seconds

    projected = actual + next_service_allocation_forecast + remaining
    protected_hard_projection = projected + 2 * DEFAULT_SHUTDOWN_SECONDS
    if projected > limits.scheduled_gpu_seconds:
        raise RuntimeError("fallback recovery no longer fits the scheduled GPU envelope")
    if protected_hard_projection >= limits.hard_gpu_seconds:
        raise RuntimeError("fallback recovery lacks its protected hard-stop margin")
    authorization = (
        None
        if second_overlay is None
        else cast(Mapping[str, object], second_overlay["authorization"])
    )
    execution_authorized = (
        True if authorization is None else authorization.get("status") == "authorized"
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "phase1_fallback_execution_preflight",
        "run_id": options.run_id,
        "model": {
            "repository": policy.repository,
            "revision": policy.revision,
            "served_model_name": policy.served_model_name,
        },
        "launcher_configuration_sha256": configuration.configuration_hash,
        "activation_certificate_sha256": activation["manifest_sha256"],
        "cache_replacement_receipt_sha256": replacement["manifest_sha256"],
        "snapshot_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "source_association_sha256": source_association["manifest_sha256"],
        "source_tree_sha256": source_association["local_tree_sha256"],
        "retry_amendment_sha256": amendment_hash,
        "prior_fallback_failure_sha256": predecessor_hash,
        "second_recovery_overlay_sha256": (
            None if second_overlay is None else second_overlay["manifest_sha256"]
        ),
        "second_recovery_v3_result_sha256": (
            None if second_predecessor is None else second_predecessor["manifest_sha256"]
        ),
        "second_recovery_v3_incident_sha256": (
            None if second_incident is None else second_incident["manifest_sha256"]
        ),
        "authorization_status": (None if authorization is None else authorization.get("status")),
        "authorization_basis": (None if authorization is None else authorization.get("basis")),
        "execution_authorized": execution_authorized,
        "authorization_gaps": (
            [] if execution_authorized else ["explicit dated user authorization remains pending"]
        ),
        "gpu_accounting_before_start": _gpu_summary_payload(observed),
        "next_watchdog_seconds": next_watchdog,
        "next_service_allocation_forecast_seconds": (next_service_allocation_forecast),
        "remaining_mandatory_forecast_seconds": remaining,
        "actual_plus_next_and_remaining_seconds": projected,
        "scheduled_limit_seconds": float(limits.scheduled_gpu_seconds),
        "scheduled_reserve_seconds": float(limits.scheduled_gpu_seconds) - projected,
        "protected_shutdown_seconds": float(2 * DEFAULT_SHUTDOWN_SECONDS),
        "hard_limit_seconds": float(limits.hard_gpu_seconds),
        "hard_contingency_after_next_and_shutdown_seconds": (
            float(limits.hard_gpu_seconds) - protected_hard_projection
        ),
        "effective_accounting_events": inventory.accounting_events
        + (2 if second_overlay is not None else 1 if amendment_hash is not None else 0),
        "maximum_inference_attempts": inventory.maximum_inference_attempts,
        "authorized_retry_inference_attempts": (1 if second_overlay is not None else 0),
        "additional_unreserved_inference_attempts": 0,
        "checkpoint_absent": True,
        "storage": {
            "current_occupied_bytes": storage_report.current_occupied_bytes,
            "projected_occupied_bytes": storage_report.projected_occupied_bytes,
            "effective_projected_headroom_bytes": (
                storage_report.effective_projected_headroom_bytes
            ),
            "allowed": storage_report.allowed,
        },
        "gpu_allocation_performed": False,
        "model_process_started": False,
        "passed": execution_authorized,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


def main(
    arguments: Sequence[str] | None = None,
    *,
    development_adopter: DevelopmentContinuationAdopter | None = None,
) -> int:
    options = parse_arguments(arguments)
    if options.status:
        print(canonical_json(_orchestrator_status(options)))
        return 0
    root = options.project_root.resolve(strict=True)
    if options.build_second_recovery_overlay:
        _build_second_recovery_overlay_from_cli(options, root=root)
        return 0
    if not options.execute and not options.validate_only:
        atomic_write_public_json(options.output, fallback_plan_manifest(root))
        return 0
    _require_execution_arguments(options)
    if options.validate_only:
        validation = _validate_execution_preflight(options, root=root)
        # A validation receipt is part of the immutable run record.  Refuse an
        # existing path at the final link operation so a repeated operator
        # command cannot replace the evidence for an earlier admission check.
        _write_append_only_json(options.output.resolve(), validation)
        return 0 if validation.get("passed") is True else 2
    if options.controller_stage == "orchestrate":
        # Import the registered production factory before the orchestrator may
        # request the third model load.  Internal controller processes perform
        # the actual construction against their adopted service handle.
        from story_projection_onto.development_continuation import (
            create_production_development_adopter,
        )

        del create_production_development_adopter
        return _orchestrate_controller_processes(options)
    orchestrator_guard: Mapping[str, object] | None = None
    guardian_ticket: Mapping[str, object] | None = None
    controller_receipt: Mapping[str, object] | None = None
    if options.controller_stage == "guardian":
        guardian_ticket = _validate_guardian_ticket(options)
    else:
        orchestrator_guard = _validate_orchestrator_guard(options)
        invocation, _ticket = _load_orchestration_identity(
            options,
            _orchestration_paths(options),
        )
        controller_receipt = _register_internal_controller(
            options,
            guard=orchestrator_guard,
            invocation=invocation,
        )
    policy_path = root / "configs/study/fallback_model.json"
    policy = FallbackModelPolicy.load(policy_path)
    legacy_provenance_bridge = Phase1LegacyEvidenceProvenanceBridge.load(root)
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
    result: dict[str, object]
    with Ledger(options.ledger) as ledger:
        ledger.record_storage_sample(preflight, phase=f"phase1_fallback:{options.run_id}")
        if not preflight.allowed:
            raise StorageBudgetExceeded(preflight)
        retry_amendment: dict[str, object] | None = None
        prior_fallback_failure: dict[str, object] | None = None
        second_recovery_overlay: dict[str, object] | None = None
        second_recovery_v3_result: dict[str, object] | None = None
        second_recovery_v3_incident: dict[str, object] | None = None
        service_start_watchdog_seconds = DEFAULT_FALLBACK_STARTUP_WATCHDOG_SECONDS
        recovery_service_start_event_ids: tuple[str, ...] = ()
        if options.second_recovery_overlay is not None:
            retry_request = build_fallback_acceptance_request(
                root=root,
                call=fallback_pilot_calls(policy)[0],
                tokenizer=cast(PackingTokenizer, tokenizer),
                tokenizer_manifest=tokenizer_manifest,
                legacy_provenance_bridge=legacy_provenance_bridge,
            )
            (
                second_recovery_overlay,
                second_recovery_v3_result,
                second_recovery_v3_incident,
            ) = validate_second_fallback_recovery_overlay(
                root=root,
                overlay_path=options.second_recovery_overlay,
                v3_result_path=cast(Path, options.second_recovery_v3_result),
                v3_incident_path=cast(Path, options.second_recovery_v3_incident),
                prior_retry_amendment_path=cast(Path, options.retry_amendment),
                prior_retry_failure_path=cast(Path, options.prior_fallback_failure),
                run_id=options.run_id,
                policy=policy,
                activation_certificate=activation,
                primary_result=primary_result,
                limits=limits,
                source_association=source_association,
                source_association_path=options.source_association,
                retry_request=retry_request,
                legacy_provenance_bridge=legacy_provenance_bridge,
                observed=(ledger.gpu_summary() if options.controller_stage == "prepare" else None),
                require_authorized=True,
                verify_decoder_compilation=True,
            )
            retry_amendment = _validated_manifest_object(
                cast(Path, options.retry_amendment),
                expected_kind="phase1_fallback_service_retry_amendment",
            )
            prior_fallback_failure = _validated_manifest_object(
                cast(Path, options.prior_fallback_failure),
                expected_kind="phase1_fallback_micro_pilot_result",
            )
            service_start_watchdog_seconds = AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
            recovery_service_start_event_ids = (
                f"{SECOND_RECOVERY_V3_RUN_ID}-service-start-001",
                f"{options.run_id}-service-start-001",
            )
        elif options.retry_amendment is not None:
            retry_amendment, prior_fallback_failure = validate_fallback_service_retry_amendment(
                root=root,
                amendment_path=options.retry_amendment,
                prior_failure_path=options.prior_fallback_failure,
                run_id=options.run_id,
                policy=policy,
                activation_certificate=activation,
                primary_result=primary_result,
                limits=limits,
                observed=(ledger.gpu_summary() if options.controller_stage == "prepare" else None),
            )
            service_start_watchdog_seconds = AMENDED_FALLBACK_STARTUP_WATCHDOG_SECONDS
            recovery_service_start_event_ids = (f"{options.run_id}-service-start-001",)
        elif options.controller_stage == "prepare":
            observed_baseline = validate_pre_fallback_gpu_accounting(
                primary_result,
                ledger.gpu_summary(),
            )
            if observed_baseline != pre_fallback_accounting:
                raise RuntimeError("pre-fallback GPU accounting baseline changed")
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
            readiness_check=lambda: client.ready(2.0, model_name=FALLBACK_SERVED_MODEL_NAME),
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
                model_snapshot_manifest_hash=cast(str, snapshot_manifest["manifest_sha256"]),
                source_association=source_association,
                checkpoint_path=options.checkpoint,
                assessment_factory=build_post_run_development_assessment_provider,
                retry_amendment_sha256=(
                    None
                    if retry_amendment is None
                    else cast(str, retry_amendment["manifest_sha256"])
                ),
                second_recovery_overlay_sha256=(
                    None
                    if second_recovery_overlay is None
                    else cast(str, second_recovery_overlay["manifest_sha256"])
                ),
                recovery_service_start_event_ids=recovery_service_start_event_ids,
            )
        runner = FallbackAcceptanceRunner(
            root=root,
            legacy_provenance_bridge=legacy_provenance_bridge,
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
            retry_amendment=retry_amendment,
            prior_fallback_failure=prior_fallback_failure,
            second_recovery_overlay=second_recovery_overlay,
            second_recovery_v3_result=second_recovery_v3_result,
            second_recovery_v3_incident=second_recovery_v3_incident,
            service_start_watchdog_seconds=service_start_watchdog_seconds,
            development_adopter=development_adopter,
            runtime_stack=runtime_stack,
            gpu_hardware=gpu_hardware,
        )
        if options.controller_stage == "guardian":
            assert guardian_ticket is not None
            result = _run_guardian(options, runner=runner, ticket=guardian_ticket)
        else:
            try:
                if options.controller_stage == "prepare":
                    result = runner.prepare_controller_restart()
                elif options.controller_stage == "recover-prepare":
                    result = runner.recover_controller_restart_preparation()
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
            assert orchestrator_guard is not None
            invocation, _ticket = _load_orchestration_identity(
                options,
                _orchestration_paths(options),
            )
            result = _bind_controller_stage_result(
                result,
                stage=cast(str, options.controller_stage),
                guard=orchestrator_guard,
                invocation=invocation,
                controller_receipt=controller_receipt,
            )
            try:
                _write_append_only_json(options.output.resolve(), result)
            except BaseException:
                # A successful prepare deliberately leaves the model process live.
                # If its public handoff receipt cannot be persisted, immediately
                # adopt the exact private checkpoint and verify physical shutdown.
                if (
                    options.controller_stage in {"prepare", "recover-prepare"}
                    and result.get("model_service_left_live_for_controller_restart") is True
                ):
                    runner.cleanup_orphan()
                raise
    if options.controller_stage in {"prepare", "recover-prepare"}:
        return 0 if result.get("next_required_stage") is not None else 2
    if options.controller_stage == "cleanup":
        return 0 if result.get("physical_shutdown_verified") is True else 2
    if options.controller_stage == "guardian":
        return 0 if result.get("physical_shutdown_verified") is True else 2
    return 0 if result.get("phase1_gate_passed") is True else 2


__all__ = [
    "FALLBACK_IMPLEMENTATION_FILES",
    "FORBIDDEN_ADAPTER_LIFECYCLE_MEMBERS",
    "NORMAL_ACCEPTANCE_CLASSES",
    "REPAIR_TRIGGER_RULE",
    "SECOND_RECOVERY_C0_IMPLEMENTATION_PATH",
    "SECOND_RECOVERY_C0_REGRESSION_TEST_PATH",
    "SECOND_RECOVERY_C1_CONDITION_PATHWAY_TEST_PATH",
    "SECOND_RECOVERY_C1_DEVELOPMENT_ASSESSMENT_TEST_PATH",
    "SECOND_RECOVERY_C1_IMPLEMENTATION_PATH",
    "SECOND_RECOVERY_CONTRACTS_IMPLEMENTATION_PATH",
    "SECOND_RECOVERY_CONTRACTS_REGRESSION_TEST_PATH",
    "SECOND_RECOVERY_EVIDENCE_BRIDGE_IMPLEMENTATION_PATH",
    "SECOND_RECOVERY_EVIDENCE_BRIDGE_REGRESSION_TEST_PATH",
    "SECOND_RECOVERY_EVIDENCE_BRIDGE_SECTION_NAME",
    "SECOND_RECOVERY_FROZEN_BUDGET_VALUES_SHA256",
    "SECOND_RECOVERY_FROZEN_DECODING_VALUES_SHA256",
    "SECOND_RECOVERY_FROZEN_EVIDENCE_VALUES_SHA256",
    "SECOND_RECOVERY_FROZEN_INPUT_SOURCE_PATHS",
    "SECOND_RECOVERY_FROZEN_MODEL_INPUT_FIXTURES_SHA256",
    "SECOND_RECOVERY_FROZEN_PROMPT_SET_SHA256",
    "SECOND_RECOVERY_INTEGRITY_DISPLAY_SOURCE_PATHS",
    "SECOND_RECOVERY_INTEGRITY_DISPLAY_TEST_PATHS",
    "SECOND_RECOVERY_INTEGRITY_DOCUMENTATION_PATHS",
    "SECOND_RECOVERY_INTEGRITY_GROUNDING_SOURCE_PATHS",
    "SECOND_RECOVERY_INTEGRITY_GROUNDING_TEST_PATHS",
    "SECOND_RECOVERY_INTEGRITY_LIFECYCLE_SOURCE_PATHS",
    "SECOND_RECOVERY_INTEGRITY_LIFECYCLE_TEST_PATHS",
    "SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_SOURCE_PATHS",
    "SECOND_RECOVERY_INTEGRITY_SEMANTIC_SCOPE_TEST_PATHS",
    "SECOND_RECOVERY_OVERLAY_KIND",
    "SECOND_RECOVERY_POST_TWO_FIX_C0_IMPLEMENTATION_SHA256",
    "SECOND_RECOVERY_POST_TWO_FIX_C0_REGRESSION_TEST_SHA256",
    "SECOND_RECOVERY_RETRY_CALL_ID",
    "SECOND_RECOVERY_UNCHANGED_ONTOLOGY_DRAFT_SCHEMA_SHA256",
    "SECOND_RECOVERY_V3_C0_IMPLEMENTATION_SHA256",
    "SECOND_RECOVERY_V3_C1_IMPLEMENTATION_SHA256",
    "SECOND_RECOVERY_V3_DECODER_SCHEMA_SHA256",
    "SECOND_RECOVERY_V3_FAILED_CALL_MICROSECONDS",
    "SECOND_RECOVERY_V3_INCIDENT_FILE_SHA256",
    "SECOND_RECOVERY_V3_INCIDENT_MANIFEST_SHA256",
    "SECOND_RECOVERY_V3_REQUEST_SHA256",
    "SECOND_RECOVERY_V3_RESULT_FILE_SHA256",
    "SECOND_RECOVERY_V3_RESULT_MANIFEST_SHA256",
    "SECOND_RECOVERY_V3_RETRY_AMENDMENT_SHA256",
    "SECOND_RECOVERY_V3_RUN_ID",
    "SECOND_RECOVERY_V3_SOURCE_MANIFEST_FILE_SHA256",
    "SECOND_RECOVERY_V3_SOURCE_MANIFEST_PATH",
    "SECOND_RECOVERY_V3_SOURCE_REVISION",
    "SECOND_RECOVERY_V3_SOURCE_TREE_SHA256",
    "SECOND_RECOVERY_V4_C1_CONDITION_PATHWAY_TEST_SHA256",
    "SECOND_RECOVERY_V4_C1_DEVELOPMENT_ASSESSMENT_TEST_SHA256",
    "SECOND_RECOVERY_V4_C1_IMPLEMENTATION_SHA256",
    "SECOND_RECOVERY_VALIDATE_IMPLEMENTATION_PATH",
    "DevelopmentAdopterRegistration",
    "DevelopmentContinuationAdopter",
    "DevelopmentContinuationBootstrap",
    "DevelopmentContinuationHandoff",
    "DevelopmentContinuationReceipt",
    "FallbackAcceptanceRunner",
    "FallbackCallSpec",
    "PreparedDevelopmentContinuation",
    "SecondFallbackRecoveryOverlay",
    "SecondRecoveryC0PreDataCorrection",
    "SecondRecoveryConcurrentIntegrityDisclosure",
    "SecondRecoveryEvidenceBridgeBinding",
    "SecondRecoveryFrozenInputControls",
    "SecondRecoveryFrozenInputFileComparison",
    "SecondRecoveryIntegrityFileBinding",
    "SecondRecoveryIntegritySurfaceFiles",
    "SecondRecoveryProjectionDependencyCorrection",
    "SecondRecoveryProjectionDependencyUnchangedControls",
    "SecondRecoverySemanticValidationCorrection",
    "SecondRecoverySemanticValidationUnchangedControls",
    "build_fallback_repair_request",
    "build_second_fallback_recovery_overlay",
    "establish_fallback_orchestrator_process_group",
    "fallback_pilot_calls",
    "fallback_plan_manifest",
    "main",
    "parse_arguments",
    "pre_fallback_gpu_accounting_baseline",
    "validate_pre_fallback_gpu_accounting",
    "validate_second_fallback_recovery_overlay",
    "validate_source_association",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the operational CLI
    raise SystemExit(main())
