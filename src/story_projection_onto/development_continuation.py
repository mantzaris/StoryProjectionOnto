"""Production fallback-to-development continuation on one live model service.

The fallback controller owns the vLLM lifecycle.  This module performs all
query-blind CPU preparation, constructs the narrow lifecycle-free adapter, and
runs the exact registered development block synchronously.  It cannot start,
load, restart, or stop a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from story_projection_onto.benchmark_runtime import (
    EvidenceProjectionEquivalenceCertificate,
    ModelEligibleWorldArtifact,
    NeutralEvidenceArtifact,
    RuntimeStagingManifest,
    load_staged_neutral_evidence,
    load_staged_world,
)
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ComparisonInputManifest,
    ConditionPreparation,
    ProduceInputs,
    RunConditionConfig,
)
from story_projection_onto.conditions.c0 import (
    ClassicalPreBuilder,
    load_production_classical_builder,
)
from story_projection_onto.conditions.c1 import (
    LLMPreCondition,
    build_c1_preconstruction_request,
)
from story_projection_onto.conditions.c2 import prepare_empty_c2_inventory
from story_projection_onto.contracts import (
    ConditionName,
    ImmutableRecord,
    PrequeryPreparationBinding,
    ReleaseClass,
    RunOutcome,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    AdapterDurableState,
    DevelopmentConstructionConfiguration,
    MeteredGenerationService,
    PackingTokenizer,
    ProductionDevelopmentServiceAdapter,
    build_development_guided_request,
    development_packing_equivalence_hash,
    development_request_runtime,
    development_runtime_identifiers,
    encode_development_semantic_request,
    persist_logical_record,
    persist_opaque_json,
)
from story_projection_onto.development_artifacts import (
    FALLBACK_V9_RECOVERY_SERVICE_START_EVENT_IDS,
    HISTORICAL_SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
    SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
    DevelopmentAssessmentBundle,
    DevelopmentCPUProjectionReceipt,
    DevelopmentForecastInventoryRow,
    DevelopmentForecastReceipt,
    DevelopmentPackedRequestReceipt,
    DevelopmentPackingPreflight,
    DevelopmentPreparationIndex,
    LogicalCASReference,
    OpaqueJSONReference,
)
from story_projection_onto.development_execution import (
    DevelopmentExecutionRepository,
    ProductionDevelopmentCallExecutor,
)
from story_projection_onto.development_runtime import (
    DEVELOPMENT_CALL_COUNT,
    DEVELOPMENT_UNIT_IDS,
    DevelopmentCallKind,
    DevelopmentCallManifest,
    DevelopmentExecutionManifest,
    DevelopmentExecutionResult,
    DevelopmentITTRecord,
    DevelopmentPrequeryInputs,
    DevelopmentRunner,
    DevelopmentScientificAssessment,
    FixedSchemaDerivationPlan,
    ForecastControl,
    InjectedLiveDevelopmentService,
    LiveServiceIdentity,
    UnitPrequeryBinding,
    load_development_call_manifest,
)
from story_projection_onto.fallback_acceptance import (
    DevelopmentAdopterRegistration,
    DevelopmentContinuationBootstrap,
    DevelopmentContinuationHandoff,
    PreparedDevelopmentContinuation,
)
from story_projection_onto.gpu_runtime import TokenizerManifest
from story_projection_onto.llm import (
    CapabilityManifest,
    base_condition_output_schema,
    render_condition_system_prompt,
)
from story_projection_onto.manifest import SourceManifest, build_source_manifest
from story_projection_onto.query_runtime import AuditedBenchmarkRuntime
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    CommitmentCheckStatus,
    EvidenceSupportStatus,
    GpuEventKind,
    InputKind,
    JobState,
    Ledger,
    SemanticAssessmentScope,
    TemporalValidationStatus,
    ValidationStatus,
)

DEFAULT_GPU_CALL_INVENTORY = Path("configs/study/gpu_call_inventory.json")
DEFAULT_SEED_MANIFEST = Path("data/synthetic/manifests/seed_manifest.json")
_NORMAL_ACCEPTANCE_CLASSES = frozenset(
    {
        "acceptance_c1",
        "acceptance_c2",
        "acceptance_fixed_select",
        "acceptance_repair",
    }
)
_DEVELOPMENT_CLASSES = frozenset(
    {
        "development_c1",
        "development_c2",
        "development_fixed_select",
        "development_ablation",
        "development_repair",
    }
)


class DevelopmentContinuationError(RuntimeError):
    """A frozen continuation input or durable artifact was inconsistent."""


def _validate_recovery_service_start_binding(
    *,
    retry_amendment_sha256: str | None,
    second_recovery_overlay_sha256: str | None,
    recovery_service_start_event_ids: Sequence[str],
) -> None:
    """Keep ordinary and second-recovery lifecycle overlays disjoint."""

    identifiers = tuple(recovery_service_start_event_ids)
    hashes = (retry_amendment_sha256, second_recovery_overlay_sha256)
    if any(
        value is not None
        and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        )
        for value in hashes
    ):
        raise DevelopmentContinuationError("GPU recovery overlay hash is invalid")
    if len(set(identifiers)) != len(identifiers):
        raise DevelopmentContinuationError("GPU recovery service IDs must be unique")
    if second_recovery_overlay_sha256 is not None:
        if (
            retry_amendment_sha256 is None
            or identifiers
            not in {
                HISTORICAL_SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
                SECOND_FALLBACK_RECOVERY_SERVICE_START_EVENT_IDS,
                FALLBACK_V9_RECOVERY_SERVICE_START_EVENT_IDS,
            }
        ):
            raise DevelopmentContinuationError(
                "second recovery requires an exact ordered v3+v7[/v8][/v9] service lineage"
            )
        return
    if len(identifiers) > 1 or bool(identifiers) != bool(retry_amendment_sha256):
        raise DevelopmentContinuationError(
            "ordinary recovery permits at most one amendment-bound service ID"
        )


class PostRunAssessmentFactory(Protocol):
    """Late-bound scorer bridge; it must not read scorer data when constructed."""

    def __call__(
        self,
        *,
        root: Path,
        ledger: Ledger,
        blobs: BlobStore,
        prequery_inputs: DevelopmentPrequeryInputs,
        assessment_bundle_artifact_hash: str,
        assessment_manifest_path: Path,
    ) -> Callable[
        [DevelopmentCallManifest, tuple[DevelopmentITTRecord, ...]],
        DevelopmentScientificAssessment,
    ]: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pointer(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise DevelopmentContinuationError("preparation pointer cannot be a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(payload))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_utc(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise DevelopmentContinuationError("development clock must be timezone-aware")
    return value.astimezone(UTC)


def _strictly_after(clock: Callable[[], datetime], threshold: datetime) -> datetime:
    for _ in range(100):
        value = _strict_utc(clock)
        if value > threshold:
            return value
    raise DevelopmentContinuationError("development clock did not advance")


def _logical_from_cas(
    artifacts: ArtifactStore,
    reference: LogicalCASReference,
    model: type[ImmutableRecord],
) -> ImmutableRecord:
    record = artifacts.ledger.get_artifact(reference.artifact_hash)
    value = model.model_validate_json(artifacts.blobs.read_bytes(record))
    if value.content_hash != reference.logical_content_hash:
        raise DevelopmentContinuationError("logical CAS reference changed")
    return value


def load_development_prequery_evidence(
    root: Path,
    manifest: DevelopmentCallManifest,
) -> tuple[
    dict[str, NeutralEvidenceArtifact],
    dict[str, ModelEligibleWorldArtifact],
    dict[str, EvidenceProjectionEquivalenceCertificate],
]:
    neutral_by_unit: dict[str, NeutralEvidenceArtifact] = {}
    visible_by_unit: dict[str, ModelEligibleWorldArtifact] = {}
    certificates: dict[str, EvidenceProjectionEquivalenceCertificate] = {}
    for reference in manifest.neutral_evidence_stages:
        unit_id = reference.unit_id
        prequery = next(
            call.prequery_stage for call in manifest.calls if call.unit_id == unit_id
        )
        prequery_root = (root / prequery.relative_path).resolve(strict=True)
        prequery_manifest = RuntimeStagingManifest.model_validate_json(
            (prequery_root / "manifest.json").read_text(encoding="utf-8")
        )
        visible = load_staged_world(
            prequery_root / "evidence.json", prequery_root, prequery_manifest
        )
        neutral_root = (root / reference.relative_path).resolve(strict=True)
        neutral_manifest = RuntimeStagingManifest.model_validate_json(
            (neutral_root / "manifest.json").read_text(encoding="utf-8")
        )
        neutral, certificate = load_staged_neutral_evidence(
            neutral_root,
            root / "data/synthetic/condition_inputs/neutral_evidence",
            neutral_manifest,
            visible,
        )
        expected = (
            reference.neutral_evidence_artifact_hash,
            reference.model_visible_evidence_artifact_hash,
            reference.equivalence_certificate_hash,
            reference.snapshot_hash,
            reference.runtime_unit_id,
        )
        observed = (
            neutral.content_hash,
            visible.content_hash,
            certificate.content_hash,
            neutral.snapshot.content_hash,
            neutral.snapshot.world_or_window_id,
        )
        if observed != expected:
            raise DevelopmentContinuationError(
                "prequery evidence differs from the registered neutral stage"
            )
        neutral_by_unit[unit_id] = neutral
        visible_by_unit[unit_id] = visible
        certificates[unit_id] = certificate
    return neutral_by_unit, visible_by_unit, certificates


def development_seed_manifest_hash(
    root: Path, manifest: DevelopmentCallManifest
) -> str:
    path = root / DEFAULT_SEED_MANIFEST
    if path.is_symlink() or _sha256_file(path) != manifest.seed_manifest_file_sha256:
        raise DevelopmentContinuationError("development seed manifest bytes changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content_hash = value.get("content_hash") if isinstance(value, dict) else None
    if not isinstance(content_hash, str) or len(content_hash) != 64:
        raise DevelopmentContinuationError("development seed manifest is invalid")
    return content_hash


def development_validator_hash(root: Path) -> str:
    return canonical_sha256(
        {
            "validator_protocol": "development-boundary-validator-v1",
            "validate_source_sha256": _sha256_file(
                root / "src/story_projection_onto/validate.py"
            ),
            "scored_projection_schema_hash": SCORED_PROJECTION_SCHEMA_HASH,
        }
    )


def _prompt_family_hash(root: Path) -> str:
    bindings = {
        condition.value: hashlib.sha256(
            render_condition_system_prompt(root, condition).encode("utf-8")
        ).hexdigest()
        for condition in (
            ConditionName.C1_LLM_PRE,
            ConditionName.C2_LLM_QUERY,
            ConditionName.A_FIXED_SELECT,
            ConditionName.A_NO_CONTEXT,
            ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ConditionName.A_NO_RARE_GUARD,
        )
    }
    bindings["repair"] = hashlib.sha256(
        (root / "prompts/repair/prompt_v1.md").read_bytes()
    ).hexdigest()
    return canonical_sha256(bindings)


def build_development_run_configurations(
    *,
    root: Path,
    manifest: DevelopmentCallManifest,
    construction: DevelopmentConstructionConfiguration,
    tokenizer_manifest: TokenizerManifest,
    model_stack_hash: str,
    seed_manifest_hash: str,
    validator_hash: str,
) -> tuple[dict[int, RunConditionConfig], dict[int, FixedSchemaDerivationPlan]]:
    """Build every non-Fixed config and four query-blind Fixed recipes."""

    configs: dict[int, RunConditionConfig] = {}
    plans: dict[int, FixedSchemaDerivationPlan] = {}
    baseline_runtime = development_request_runtime(
        root=root,
        condition=ConditionName.C2_LLM_QUERY,
        tokenizer_manifest=tokenizer_manifest,
        seed=manifest.vllm_seed,
    )
    fixed_prompt_hash = hashlib.sha256(
        render_condition_system_prompt(root, ConditionName.A_FIXED_SELECT).encode(
            "utf-8"
        )
    ).hexdigest()
    fixed_capability = CapabilityManifest.for_condition(ConditionName.A_FIXED_SELECT)
    fixed_base_schema_hash = canonical_sha256(
        base_condition_output_schema(ConditionName.A_FIXED_SELECT)
    )
    for call in manifest.calls:
        budgets = (
            construction.preconstruction_budgets
            if call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
            else construction.projection_budgets_by_unit[call.unit_id]
        )
        if call.kind is DevelopmentCallKind.FIXED_SELECTION:
            assert call.source_c1_call_id is not None and call.query_stage is not None
            plans[call.ordinal] = FixedSchemaDerivationPlan(
                ordinal=call.ordinal,
                call_id=call.call_id,
                unit_id=call.unit_id,
                source_c1_call_id=call.source_c1_call_id,
                budgets=budgets,
                model_stack_hash=model_stack_hash,
                decoding_family_hash=baseline_runtime.decoding_manifest.comparison_family_hash,
                seed_manifest_hash=seed_manifest_hash,
                prompt_hash=fixed_prompt_hash,
                base_output_schema_hash=fixed_base_schema_hash,
                scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
                capability_manifest_hash=fixed_capability.content_hash,
                validator_hash=validator_hash,
                upper_ontology_hash=construction.upper_ontology.content_hash,
                prequery_evidence_artifact_hash=call.prequery_stage.evidence_artifact_hash,
                query_stage_manifest_hash=call.query_stage.staging_manifest_hash,
            )
            continue
        repair = call.kind is DevelopmentCallKind.REPAIR_PROBE
        runtime = development_request_runtime(
            root=root,
            condition=call.condition,
            tokenizer_manifest=tokenizer_manifest,
            seed=call.vllm_seed,
            repair=repair,
        )
        maximum_input = (
            construction.repair_input_tokens
            if repair
            else construction.first_pass_input_tokens
        )
        maximum_output = (
            construction.repair_output_tokens
            if repair
            else construction.first_pass_output_tokens
        )
        configs[call.ordinal] = RunConditionConfig(
            config_id=f"run-config-{call.call_id}",
            condition=call.condition,
            budgets=budgets,
            maximum_input_tokens=maximum_input,
            maximum_output_tokens=maximum_output,
            repair_attempt_budget=budgets.repair_attempt_budget,
            seed_block=call.seed_block,
            model_stack_hash=model_stack_hash,
            decoding_manifest_hash=runtime.decoding_manifest.content_hash,
            decoding_family_hash=runtime.decoding_manifest.comparison_family_hash,
            seed_manifest_hash=seed_manifest_hash,
            resolved_seed=call.vllm_seed,
            prompt_hash=runtime.prompt_hash,
            output_schema_hash=runtime.output_schema_hash,
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            capability_manifest_hash=runtime.capability_manifest.content_hash,
            validator_hash=validator_hash,
            upper_ontology_hash=construction.upper_ontology.content_hash,
        )
    if set(configs) != set(range(1, 25)) - {17, 18, 19, 20}:
        raise DevelopmentContinuationError("non-Fixed run configuration inventory changed")
    if set(plans) != {17, 18, 19, 20}:
        raise DevelopmentContinuationError("Fixed derivation-plan inventory changed")
    return configs, plans


def build_development_forecast_receipt(
    *,
    root: Path,
    ledger: Ledger,
    service: MeteredGenerationService,
    manifest: DevelopmentCallManifest,
    clock: Callable[[], datetime],
    retry_amendment_sha256: str | None = None,
    second_recovery_overlay_sha256: str | None = None,
    recovery_service_start_event_ids: Sequence[str] = (),
    scheduled_limit_seconds: float = 9 * 3600,
    hard_limit_seconds: float = 10 * 3600,
) -> DevelopmentForecastReceipt:
    """Derive a cumulative forecast without double-counting superseded calls."""

    path = root / DEFAULT_GPU_CALL_INVENTORY
    if path.is_symlink() or _sha256_file(path) != manifest.gpu_call_inventory_file_sha256:
        raise DevelopmentContinuationError("GPU call inventory bytes changed")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    rows = parsed.get("classes") if isinstance(parsed, dict) else None
    if not isinstance(rows, list):
        raise DevelopmentContinuationError("GPU call inventory is invalid")
    ordered_recovery_ids = tuple(recovery_service_start_event_ids)
    recovery_ids = frozenset(ordered_recovery_ids)
    _validate_recovery_service_start_binding(
        retry_amendment_sha256=retry_amendment_sha256,
        second_recovery_overlay_sha256=second_recovery_overlay_sha256,
        recovery_service_start_event_ids=ordered_recovery_ids,
    )
    event_counts: Counter[str] = Counter()
    reserve_ids: set[str] = set()
    observed_recovery_ids: set[str] = set()
    for event in ledger.gpu_events():
        details = json.loads(event.details_json)
        service_start = (
            event.event_kind is GpuEventKind.GPU_SESSION_START
            or details.get("intended_event_kind") == GpuEventKind.GPU_SESSION_START.value
        )
        if service_start and event.event_id in recovery_ids:
            observed_recovery_ids.add(event.event_id)
        elif service_start:
            event_counts["gpu_session_start"] += 1
        call_class = details.get("call_class")
        if isinstance(call_class, str) and call_class != "gpu_session_start":
            event_counts[call_class] += 1
        reserve_class = details.get("reserve_call_class")
        reservation_id = details.get("reserve_reservation_id")
        if isinstance(reserve_class, str) and isinstance(reservation_id, str):
            identity = f"{reserve_class}:{reservation_id}"
            if identity not in reserve_ids:
                reserve_ids.add(identity)
                event_counts[reserve_class] += 1
    if observed_recovery_ids != recovery_ids:
        raise DevelopmentContinuationError(
            "GPU recovery overlay does not match the cumulative ledger"
        )
    development_counts = Counter(call.call_class for call in manifest.calls)
    forecast_rows: list[DevelopmentForecastInventoryRow] = []
    development_seconds = 0.0
    for untyped in rows:
        if not isinstance(untyped, dict):
            raise DevelopmentContinuationError("GPU inventory row is invalid")
        name = untyped.get("name")
        count = untyped.get("count")
        p95 = untyped.get("provisional_p95_seconds")
        if (
            not isinstance(name, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or isinstance(p95, bool)
            or not isinstance(p95, (int, float))
        ):
            raise DevelopmentContinuationError("GPU inventory row types changed")
        if name in _NORMAL_ACCEPTANCE_CLASSES:
            consumed = count
            development = 0
        elif name in _DEVELOPMENT_CLASSES:
            consumed = 0
            development = development_counts[name]
            if development != count:
                raise DevelopmentContinuationError(
                    "development call inventory differs from the registered manifest"
                )
            development_seconds += development * float(p95)
        else:
            consumed = min(count, event_counts[name])
            development = 0
        remaining = count - consumed - development
        forecast_rows.append(
            DevelopmentForecastInventoryRow(
                call_class=name,
                registered_count=count,
                consumed_before_development=consumed,
                development_call_count=development,
                remaining_after_development=remaining,
                provisional_p95_seconds=float(p95),
                remaining_forecast_seconds=remaining * float(p95),
            )
        )
    actual = max(
        ledger.gpu_summary().total_allocated_seconds,
        float(service.actual_allocated_service_seconds),
    )
    if not math.isfinite(actual):
        raise DevelopmentContinuationError("cumulative GPU allocation is invalid")
    post = sum(item.remaining_forecast_seconds for item in forecast_rows)
    total = actual + development_seconds + post
    return DevelopmentForecastReceipt(
        receipt_id="development-continuation-cumulative-forecast-v1",
        gpu_call_inventory_file_sha256=manifest.gpu_call_inventory_file_sha256,
        actual_allocated_seconds_before_development=actual,
        development_forecast_seconds=development_seconds,
        post_development_mandatory_forecast_seconds=post,
        total_forecast_seconds=total,
        scheduled_limit_seconds=scheduled_limit_seconds,
        hard_limit_seconds=hard_limit_seconds,
        inventory_rows=tuple(forecast_rows),
        retry_amendment_sha256=retry_amendment_sha256,
        second_recovery_overlay_sha256=second_recovery_overlay_sha256,
        recovery_service_start_event_ids=ordered_recovery_ids,
        authorized_additional_service_start_events=len(recovery_ids),
        effective_accounting_events=sum(item.registered_count for item in forecast_rows)
        + len(recovery_ids),
        effective_inference_attempts=sum(
            item.registered_count
            for item in forecast_rows
            if item.call_class != "gpu_session_start"
        ),
        admitted=total <= scheduled_limit_seconds,
        created_at=_strict_utc(clock),
    )


def build_c1_packing_preflight(
    *,
    root: Path,
    manifest: DevelopmentCallManifest,
    construction: DevelopmentConstructionConfiguration,
    neutral_by_unit: Mapping[str, NeutralEvidenceArtifact],
    run_configs: Mapping[int, RunConditionConfig],
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    artifacts: ArtifactStore,
    clock: Callable[[], datetime],
) -> tuple[DevelopmentPackingPreflight, str]:
    """Tokenize and persist the exact four complete query-blind C1 requests."""

    receipts: list[DevelopmentPackedRequestReceipt] = []
    for call in manifest.calls[:4]:
        if call.kind is not DevelopmentCallKind.C1_PRECONSTRUCTION:
            raise DevelopmentContinuationError("development manifest does not begin with C1")
        neutral = neutral_by_unit[call.unit_id]
        config = run_configs[call.ordinal]
        requested_at = _strictly_after(clock, neutral.snapshot.sealed_at)
        semantic = build_c1_preconstruction_request(
            snapshot_hash=neutral.snapshot.content_hash,
            snapshot_sealed_at=neutral.snapshot.sealed_at,
            sealed_horizon=neutral.snapshot.horizon,
            ordered_snapshot_evidence_ids=neutral.snapshot.eligible_evidence_ids,
            evidence=neutral.evidence,
            upper_ontology=construction.upper_ontology,
            preconstruction_budgets=config.budgets,
            runtime=development_runtime_identifiers(
                root=root,
                condition=ConditionName.C1_LLM_PRE,
                tokenizer_manifest=tokenizer_manifest,
                seed=call.vllm_seed,
            ),
            requested_at=requested_at,
        )
        guided = build_development_guided_request(
            root=root,
            call_id=call.call_id,
            semantic_request=semantic,
            tokenizer=tokenizer,
            tokenizer_manifest=tokenizer_manifest,
            seed=call.vllm_seed,
        )
        alias = encode_development_semantic_request(semantic).alias_manifest
        semantic_ref = persist_logical_record(
            artifacts,
            semantic,
            object_kind="preconstruction_request",
            created_at=requested_at,
        )
        alias_ref = persist_logical_record(
            artifacts,
            alias,
            object_kind="model_wire_alias_manifest",
            created_at=requested_at,
        )
        wire_ref = persist_opaque_json(
            artifacts,
            guided.wire_payload(),
            object_kind="rendered_model_request",
            created_at=requested_at,
        )
        packing_ref = persist_logical_record(
            artifacts,
            guided.packing,
            object_kind="packing_report",
            created_at=requested_at,
        )
        receipts.append(
            DevelopmentPackedRequestReceipt(
                call_id=call.call_id,
                condition=call.condition,
                semantic_request=semantic_ref,
                packing_equivalence_hash=development_packing_equivalence_hash(
                    semantic
                ),
                alias_manifest=alias_ref,
                rendered_model_request=wire_ref,
                packing_report=packing_ref,
                rendered_input_token_count=guided.rendered_input_token_count,
                maximum_input_tokens=config.maximum_input_tokens,
            )
        )
    preflight = DevelopmentPackingPreflight(
        preflight_id="development-c1-complete-packing-v1",
        tokenizer_manifest_hash=tokenizer_manifest.manifest_sha256,
        call_manifest_hash=manifest.content_hash,
        receipts=tuple(receipts),
        worst_rendered_input_tokens=max(
            item.rendered_input_token_count for item in receipts
        ),
        completed_at=_strict_utc(clock),
    )
    artifact = artifacts.put_bytes(
        (preflight.to_canonical_json() + "\n").encode("utf-8"),
        media_type="application/vnd.story-projection.development-packing-preflight+json",
        release_class=ReleaseClass.PUBLIC,
        created_at=preflight.completed_at,
    )
    return preflight, artifact.content_hash


def _failing_assessment() -> DevelopmentScientificAssessment:
    return DevelopmentScientificAssessment(
        c0_explicit_family_coverage=0.0,
        c0_direct_assertion_precision=0.0,
        c0_direct_assertion_recall=0.0,
        c0_valid_evidence_reference_rate=0.0,
        c1_schema_valid=False,
        c1_all_construction_operators_exercised=False,
        c1_valid_evidence_id_rate=0.0,
        c1_grounding_precision=0.0,
        c1_union_gold_recall_after_seal=0.0,
        c2_construction_operator_present=False,
        programmed_horizon_leak_count=1,
        evidence_packets_equal=False,
        c1_query_blindness_verified=False,
        c2_empty_prequery_inventories_verified=False,
        fixed_complete_graph_packing_verified=False,
        fixed_constructive_operations_rejected=False,
        ablation_one_switch_verified=False,
        gold_firewall_verified=False,
    )


@dataclass(slots=True)
class ProductionDevelopmentContinuationAdopter:
    """Authenticated adopter used by both fallback controller subprocesses."""

    root: Path
    service: MeteredGenerationService = field(repr=False)
    artifacts: ArtifactStore
    tokenizer: PackingTokenizer = field(repr=False)
    tokenizer_manifest: TokenizerManifest
    launcher_configuration_hash: str
    model_snapshot_manifest_hash: str
    source_revision: str
    checkpoint_path: Path
    adapter_state_path: Path
    preparation_pointer_path: Path
    assessment_manifest_path: Path
    assessment_factory: PostRunAssessmentFactory
    retry_amendment_sha256: str | None = None
    second_recovery_overlay_sha256: str | None = None
    recovery_service_start_event_ids: tuple[str, ...] = ()
    construction_path: Path | None = None
    classical_builder_loader: Callable[[Path], tuple[ClassicalPreBuilder, object]] = (
        load_production_classical_builder
    )
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _prepared: PreparedDevelopmentContinuation | None = field(
        default=None, init=False, repr=False
    )
    _repository: DevelopmentExecutionRepository | None = field(
        default=None, init=False, repr=False
    )
    _adapter: ProductionDevelopmentServiceAdapter | None = field(
        default=None, init=False, repr=False
    )
    _runtime_source_reference: OpaqueJSONReference | None = field(
        default=None, init=False, repr=False
    )
    _classical_builder: ClassicalPreBuilder | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.root = self.root.resolve(strict=True)
        self.checkpoint_path = self.checkpoint_path.resolve(strict=False)
        self.adapter_state_path = self.adapter_state_path.resolve(strict=False)
        self.preparation_pointer_path = self.preparation_pointer_path.resolve(
            strict=False
        )
        self.assessment_manifest_path = self.assessment_manifest_path.resolve(
            strict=False
        )
        if self.construction_path is None:
            self.construction_path = self.root / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
        else:
            self.construction_path = self.construction_path.resolve(strict=True)
        _validate_recovery_service_start_binding(
            retry_amendment_sha256=self.retry_amendment_sha256,
            second_recovery_overlay_sha256=self.second_recovery_overlay_sha256,
            recovery_service_start_event_ids=self.recovery_service_start_event_ids,
        )

    def registration(self) -> DevelopmentAdopterRegistration:
        manifest = load_development_call_manifest(self.root)
        source = self.root / "src/story_projection_onto/development_continuation.py"
        # The authoritative association is the complete bounded source-tree
        # manifest, not a hand-maintained subset that can silently miss a newly
        # imported semantic dependency.
        implementation_hash = build_source_manifest(
            self.root, self.source_revision
        ).tree_sha256
        builder_hash = canonical_sha256(
            {
                "implementation": implementation_hash,
                "construction_configuration": _sha256_file(
                    cast(Path, self.construction_path)
                ),
                "development_plan": _sha256_file(
                    self.root / "configs/study/development_call_manifest.json"
                ),
                "gpu_inventory": manifest.gpu_call_inventory_file_sha256,
                "retry_amendment_sha256": self.retry_amendment_sha256,
                "second_recovery_overlay_sha256": (
                    self.second_recovery_overlay_sha256
                ),
                "recovery_service_start_event_ids": list(
                    self.recovery_service_start_event_ids
                ),
            }
        )
        return DevelopmentAdopterRegistration(
            adopter_id="production-development-continuation-v1",
            implementation_sha256=implementation_hash,
            source_sha256=_sha256_file(source),
            development_call_manifest_sha256=manifest.content_hash,
            prequery_input_builder_sha256=builder_hash,
        )

    def _identity(self, bootstrap: DevelopmentContinuationBootstrap) -> LiveServiceIdentity:
        return LiveServiceIdentity(
            owner_run_id=bootstrap.owner_run_id,
            service_pid=bootstrap.service_pid,
            service_start_ticks=bootstrap.service_start_ticks,
            gpu_session_event_id=bootstrap.gpu_session_event_id,
            launcher_configuration_hash=bootstrap.launcher_configuration_hash,
            model_snapshot_hash=bootstrap.model_snapshot_manifest_hash,
            selected_model_freeze_hash=bootstrap.selected_model_freeze_hash,
            source_execution_hash=bootstrap.source_tree_hash,
        )

    def _register_study_and_snapshots(
        self,
        *,
        execution_id: str,
        manifest: DevelopmentCallManifest,
        construction: DevelopmentConstructionConfiguration,
        source_tree_hash: str,
        neutral_by_unit: Mapping[str, NeutralEvidenceArtifact],
    ) -> None:
        ledger = self.artifacts.ledger
        created_at = min(item.snapshot.created_at for item in neutral_by_unit.values())
        ledger.register_study(
            study_id=execution_id,
            protocol_hash=manifest.source_plan_hash,
            code_manifest_hash=source_tree_hash,
            configuration_hash=construction.content_hash,
            release_class=ReleaseClass.PUBLIC,
            created_at=created_at,
        )
        for neutral in neutral_by_unit.values():
            snapshot = neutral.snapshot
            reference = persist_logical_record(
                self.artifacts,
                snapshot,
                object_kind="evidence_snapshot",
                created_at=snapshot.created_at,
            )
            input_id = f"{snapshot.snapshot_id}-input"
            ledger.register_input(
                input_id=input_id,
                study_id=execution_id,
                input_kind=InputKind.EVIDENCE_SNAPSHOT,
                content_hash=snapshot.content_hash,
                artifact_hash=reference.artifact_hash,
                release_class=ReleaseClass.PUBLIC,
                created_at=snapshot.created_at,
            )
            ledger.register_evidence_snapshot(
                snapshot_id=snapshot.snapshot_id,
                input_id=input_id,
                horizon_hash=snapshot.horizon.content_hash,
                evidence_manifest_hash=canonical_sha256(
                    tuple(item.content_hash for item in neutral.evidence)
                ),
                index_configuration_hash=snapshot.index_config_hash,
                prequery_seal_hash=snapshot.content_hash,
                eligible_evidence_count=len(snapshot.eligible_evidence_ids),
                created_at=snapshot.sealed_at,
            )

    def _build_preparations(
        self,
        *,
        manifest: DevelopmentCallManifest,
        construction: DevelopmentConstructionConfiguration,
        neutral_by_unit: Mapping[str, NeutralEvidenceArtifact],
    ) -> tuple[
        dict[tuple[str, ConditionName], ConditionPreparation],
        tuple[LogicalCASReference, ...],
        tuple[UnitPrequeryBinding, ...],
    ]:
        builder, _backend = self.classical_builder_loader(
            self.root / "configs/study/c0_rules.json"
        )
        self._classical_builder = builder
        preparations: dict[tuple[str, ConditionName], ConditionPreparation] = {}
        references: list[LogicalCASReference] = []
        bindings: list[UnitPrequeryBinding] = []
        active_by_unit = {
            "dev-unit-01": (ConditionName.C2_LLM_QUERY, ConditionName.A_NO_CONTEXT),
            "dev-unit-02": (
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_NO_TEMPORAL_EPISTEMIC,
            ),
            "dev-unit-03": (
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_NO_RARE_GUARD,
            ),
            "dev-unit-04": (ConditionName.C2_LLM_QUERY,),
        }
        for unit_id in DEVELOPMENT_UNIT_IDS:
            neutral = neutral_by_unit[unit_id]
            constructed_at = _strictly_after(self.clock, neutral.snapshot.sealed_at)
            sealed_at = _strictly_after(self.clock, constructed_at)
            c0 = builder.prepare(
                snapshot=neutral.snapshot,
                evidence=neutral.evidence,
                upper_ontology=construction.upper_ontology,
                preconstruction_budgets=construction.preconstruction_budgets,
                constructed_at=constructed_at,
                sealed_at=sealed_at,
            )
            unit_preparations = [c0]
            latest = c0.completed_at
            for condition in active_by_unit[unit_id]:
                completed = _strictly_after(self.clock, latest)
                preparation = prepare_empty_c2_inventory(
                    neutral.snapshot,
                    recorded_at=completed,
                    condition=condition,
                )
                unit_preparations.append(preparation)
                latest = completed
            prequery_bindings: list[PrequeryPreparationBinding] = []
            for preparation in unit_preparations:
                lineage = (
                    preparation.sealed_preontology.construction_seal
                    if preparation.sealed_preontology is not None
                    else preparation.empty_inventory
                )
                if lineage is None:
                    raise DevelopmentContinuationError(
                        "prequery preparation lacks query-blind lineage"
                    )
                reference = persist_logical_record(
                    self.artifacts,
                    preparation,
                    object_kind="condition_preparation",
                    created_at=preparation.completed_at,
                )
                preparations[(unit_id, preparation.condition)] = preparation
                references.append(reference)
                prequery_bindings.append(
                    PrequeryPreparationBinding(
                        unit_id=neutral.snapshot.world_or_window_id,
                        condition=preparation.condition,
                        seed_block=(
                            None
                            if preparation.condition is ConditionName.C0_CLASSICAL_PRE
                            else 1
                        ),
                        snapshot_hash=neutral.snapshot.content_hash,
                        preparation_hash=preparation.content_hash,
                        lineage_artifact_hash=lineage.content_hash,
                        completed_at=preparation.completed_at,
                    )
                )
            bindings.append(
                UnitPrequeryBinding(
                    unit_id=unit_id,
                    runtime_unit_id=neutral.snapshot.world_or_window_id,
                    snapshot_hash=neutral.snapshot.content_hash,
                    staged_model_visible_evidence_hash=next(
                        call.prequery_stage.evidence_artifact_hash
                        for call in manifest.calls
                        if call.unit_id == unit_id
                    ),
                    neutral_full_evidence_artifact_hash=neutral.content_hash,
                    evidence_equivalence_certificate_hash=next(
                        item.equivalence_certificate_hash
                        for item in manifest.neutral_evidence_stages
                        if item.unit_id == unit_id
                    ),
                    preexisting_preparation_bindings=tuple(prequery_bindings),
                    completed_at=latest,
                )
            )
        return preparations, tuple(references), tuple(bindings)

    def _persist_configurations(
        self, configs: Mapping[int, RunConditionConfig]
    ) -> tuple[LogicalCASReference, ...]:
        created_at = _strict_utc(self.clock)
        return tuple(
            persist_logical_record(
                self.artifacts,
                configs[ordinal],
                object_kind="run_condition_config",
                created_at=created_at,
            )
            for ordinal in sorted(configs)
        )

    def _read_logical(
        self,
        reference: LogicalCASReference,
        model: type[ImmutableRecord],
    ) -> ImmutableRecord:
        record = self.artifacts.ledger.get_artifact(reference.artifact_hash)
        value = model.model_validate_json(self.artifacts.blobs.read_bytes(record))
        if value.content_hash != reference.logical_content_hash:
            raise DevelopmentContinuationError("logical CAS reference changed")
        return value

    def _read_opaque(self, reference: OpaqueJSONReference) -> dict[str, object]:
        record = self.artifacts.ledger.get_artifact(reference.artifact_hash)
        try:
            value = json.loads(self.artifacts.blobs.read_bytes(record))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DevelopmentContinuationError("opaque CAS object is not JSON") from exc
        if (
            not isinstance(value, dict)
            or canonical_sha256(value) != reference.logical_content_hash
        ):
            raise DevelopmentContinuationError("opaque CAS reference changed")
        return value

    def _install_runtime_objects(
        self,
        *,
        bootstrap: DevelopmentContinuationBootstrap,
        manifest: DevelopmentCallManifest,
        construction: DevelopmentConstructionConfiguration,
        neutral: Mapping[str, NeutralEvidenceArtifact],
        visible: Mapping[str, ModelEligibleWorldArtifact],
        preparations: dict[tuple[str, ConditionName], ConditionPreparation],
        preparation_refs: tuple[LogicalCASReference, ...],
        run_configs: dict[int, RunConditionConfig],
        fixed_plans: Mapping[int, FixedSchemaDerivationPlan],
        seed_hash: str,
        validator_hash: str,
        prequery_inputs: DevelopmentPrequeryInputs,
        forecast_control: ForecastControl,
        preflight_artifact_hash: str,
        source_ref: OpaqueJSONReference,
    ) -> PreparedDevelopmentContinuation:
        execution_id = f"{bootstrap.owner_run_id}-development-v1"
        identity = self._identity(bootstrap)
        execution_manifest = DevelopmentExecutionManifest(
            execution_id=execution_id,
            call_manifest_hash=manifest.content_hash,
            source_plan_hash=manifest.source_plan_hash,
            prequery_inputs_hash=prequery_inputs.content_hash,
            service_identity_hash=identity.content_hash,
            forecast_control_hash=forecast_control.content_hash,
        )
        repository = DevelopmentExecutionRepository(
            root=self.root,
            execution_id=execution_id,
            manifest=manifest,
            construction=construction,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            source_tree_hash=bootstrap.source_tree_hash,
            selected_model_freeze_hash=bootstrap.selected_model_freeze_hash,
            model_manifest_hash=self.launcher_configuration_hash,
            seed_manifest_hash=seed_hash,
            validator_hash=validator_hash,
            neutral_by_unit=neutral,
            model_visible_by_unit=visible,
            preparations=preparations,
            preparation_references=preparation_refs,
            run_configs=run_configs,
            fixed_schema_plans=fixed_plans,
            packing_preflight_artifact_hash=preflight_artifact_hash,
            post_development_forecast_seconds=(
                forecast_control.post_development_mandatory_forecast_seconds
            ),
        )
        adapter = ProductionDevelopmentServiceAdapter(
            live_identity=identity,
            _service=self.service,
            artifacts=self.artifacts,
            state_path=self.adapter_state_path,
            execution_manifest_hash=execution_manifest.content_hash,
            repository_root=self.root,
            query_runtime=AuditedBenchmarkRuntime(
                ledger=self.artifacts.ledger,
                blobs=self.artifacts.blobs,
                clock=self.clock,
            ),
            prequery_evidence_by_hash={item.content_hash: item for item in visible.values()},
            clock=self.clock,
        )
        executor = ProductionDevelopmentCallExecutor(
            repository=repository,
            clock=self.clock,
        )
        adapter._executor = executor
        state = adapter._state()
        adapter._write_state(
            state,
            packing_preflight_artifact_hash=preflight_artifact_hash,
        )
        prepared = PreparedDevelopmentContinuation(
            call_manifest=manifest,
            prequery_inputs=prequery_inputs,
            forecast_control=forecast_control,
            execution_id=execution_id,
            execution_manifest_hash=execution_manifest.content_hash,
            checkpoint_path=self.checkpoint_path,
            service_adapter=cast(InjectedLiveDevelopmentService, adapter),
        )
        self._prepared = prepared
        self._repository = repository
        self._adapter = adapter
        self._runtime_source_reference = source_ref
        return prepared

    def _recover_preparation(
        self,
        *,
        bootstrap: DevelopmentContinuationBootstrap,
        source_manifest: SourceManifest,
        manifest: DevelopmentCallManifest,
        construction: DevelopmentConstructionConfiguration,
        neutral: Mapping[str, NeutralEvidenceArtifact],
        visible: Mapping[str, ModelEligibleWorldArtifact],
    ) -> PreparedDevelopmentContinuation | None:
        """Recover an exact query-blind CAS root without rebuilding timestamps."""

        path = self.preparation_pointer_path
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise DevelopmentContinuationError("preparation pointer is unsafe")
        try:
            pointer = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DevelopmentContinuationError("preparation pointer is not JSON") from exc
        if not isinstance(pointer, dict):
            raise DevelopmentContinuationError("preparation pointer root is invalid")
        immutable = {key: value for key, value in pointer.items() if key != "manifest_sha256"}
        if (
            set(pointer)
            != {
                "schema_version",
                "bootstrap_hash",
                "preparation_index_artifact_hash",
                "manifest_sha256",
            }
            or pointer.get("schema_version") != "1.0.0"
            or pointer.get("bootstrap_hash") != bootstrap.content_hash
            or pointer.get("manifest_sha256") != canonical_sha256(immutable)
        ):
            raise DevelopmentContinuationError("preparation pointer lineage changed")
        artifact_hash = pointer.get("preparation_index_artifact_hash")
        if not isinstance(artifact_hash, str):
            raise DevelopmentContinuationError("preparation pointer lacks its CAS root")
        index_record = self.artifacts.ledger.get_artifact(artifact_hash)
        index = DevelopmentPreparationIndex.model_validate_json(
            self.artifacts.blobs.read_bytes(index_record, allow_restricted=True)
        )
        execution_id = f"{bootstrap.owner_run_id}-development-v1"
        if (
            index.bootstrap_hash != bootstrap.content_hash
            or index.execution_id != execution_id
            or index.call_manifest_hash != manifest.content_hash
            or Path(index.checkpoint_path).resolve(strict=False) != self.checkpoint_path
            or Path(index.adapter_state_path).resolve(strict=False) != self.adapter_state_path
        ):
            raise DevelopmentContinuationError("preparation index belongs to another run")
        if self._read_opaque(index.runtime_source_manifest) != source_manifest.to_dict():
            raise DevelopmentContinuationError("preparation source manifest changed")
        stored_construction = cast(
            DevelopmentConstructionConfiguration,
            self._read_logical(
                index.construction_configuration,
                DevelopmentConstructionConfiguration,
            ),
        )
        if stored_construction != construction:
            raise DevelopmentContinuationError("preparation construction config changed")
        prequery_inputs = cast(
            DevelopmentPrequeryInputs,
            self._read_logical(index.prequery_inputs, DevelopmentPrequeryInputs),
        )
        forecast_control = cast(
            ForecastControl,
            self._read_logical(index.forecast_control, ForecastControl),
        )
        forecast_receipt = cast(
            DevelopmentForecastReceipt,
            self._read_logical(index.forecast_receipt, DevelopmentForecastReceipt),
        )
        if (
            not forecast_receipt.admitted
            or forecast_control.forecast_receipt_hash != forecast_receipt.content_hash
            or forecast_control.post_development_mandatory_forecast_seconds
            != forecast_receipt.post_development_mandatory_forecast_seconds
            or forecast_control.gpu_call_inventory_file_sha256
            != manifest.gpu_call_inventory_file_sha256
        ):
            raise DevelopmentContinuationError("recovered development forecast changed")
        seed_hash = development_seed_manifest_hash(self.root, manifest)
        validator_hash = development_validator_hash(self.root)
        expected_configs, expected_plans = build_development_run_configurations(
            root=self.root,
            manifest=manifest,
            construction=construction,
            tokenizer_manifest=self.tokenizer_manifest,
            model_stack_hash=self.launcher_configuration_hash,
            seed_manifest_hash=seed_hash,
            validator_hash=validator_hash,
        )
        recovered_configs: dict[int, RunConditionConfig] = {}
        hash_to_ordinal = {
            value.content_hash: ordinal for ordinal, value in expected_configs.items()
        }
        for reference in index.run_condition_config_artifacts:
            config = cast(
                RunConditionConfig,
                self._read_logical(reference, RunConditionConfig),
            )
            try:
                ordinal = hash_to_ordinal[config.content_hash]
            except KeyError as exc:
                raise DevelopmentContinuationError(
                    "recovered RunConditionConfig is not registered"
                ) from exc
            recovered_configs[ordinal] = config
        if recovered_configs != expected_configs:
            raise DevelopmentContinuationError("recovered RunConditionConfig set changed")
        if tuple(prequery_inputs.fixed_schema_derivation_plans) != tuple(
            expected_plans[ordinal] for ordinal in (17, 18, 19, 20)
        ):
            raise DevelopmentContinuationError("recovered FixedSelect plan set changed")
        expected_hashes = tuple(
            None
            if call.kind is DevelopmentCallKind.FIXED_SELECTION
            else expected_configs[call.ordinal].content_hash
            for call in manifest.calls
        )
        if (
            prequery_inputs.run_condition_config_hashes != expected_hashes
            or prequery_inputs.source_tree_hash != bootstrap.source_tree_hash
            or prequery_inputs.selected_model_freeze_hash
            != bootstrap.selected_model_freeze_hash
            or prequery_inputs.upper_ontology_hash
            != construction.upper_ontology.content_hash
        ):
            raise DevelopmentContinuationError("recovered prequery input bindings changed")
        preparations: dict[tuple[str, ConditionName], ConditionPreparation] = {}
        snapshot_units = {
            value.snapshot.content_hash: unit_id for unit_id, value in neutral.items()
        }
        for reference in index.prequery_preparation_artifacts:
            preparation = cast(
                ConditionPreparation,
                self._read_logical(reference, ConditionPreparation),
            )
            try:
                unit_id = snapshot_units[preparation.snapshot_hash]
            except KeyError as exc:
                raise DevelopmentContinuationError(
                    "recovered preparation belongs to another snapshot"
                ) from exc
            key = (unit_id, preparation.condition)
            if key in preparations:
                raise DevelopmentContinuationError("duplicate recovered preparation")
            preparations[key] = preparation
        bound_hashes = {
            item.preparation_hash
            for unit in prequery_inputs.unit_bindings
            for item in unit.preexisting_preparation_bindings
        }
        if {item.content_hash for item in preparations.values()} != bound_hashes:
            raise DevelopmentContinuationError("recovered preparation bindings changed")
        preflight_record = self.artifacts.ledger.get_artifact(
            index.packing_preflight_artifact_hash
        )
        preflight = DevelopmentPackingPreflight.model_validate_json(
            self.artifacts.blobs.read_bytes(preflight_record)
        )
        if (
            preflight.call_manifest_hash != manifest.content_hash
            or preflight.tokenizer_manifest_hash
            != self.tokenizer_manifest.manifest_sha256
        ):
            raise DevelopmentContinuationError("recovered packing preflight changed")
        identity = self._identity(bootstrap)
        execution_manifest = DevelopmentExecutionManifest(
            execution_id=execution_id,
            call_manifest_hash=manifest.content_hash,
            source_plan_hash=manifest.source_plan_hash,
            prequery_inputs_hash=prequery_inputs.content_hash,
            service_identity_hash=identity.content_hash,
            forecast_control_hash=forecast_control.content_hash,
        )
        if execution_manifest.content_hash != index.execution_manifest_hash:
            raise DevelopmentContinuationError("recovered execution manifest changed")
        builder, _backend = self.classical_builder_loader(
            self.root / "configs/study/c0_rules.json"
        )
        self._classical_builder = builder
        return self._install_runtime_objects(
            bootstrap=bootstrap,
            manifest=manifest,
            construction=construction,
            neutral=neutral,
            visible=visible,
            preparations=preparations,
            preparation_refs=index.prequery_preparation_artifacts,
            run_configs=recovered_configs,
            fixed_plans=expected_plans,
            seed_hash=seed_hash,
            validator_hash=validator_hash,
            prequery_inputs=prequery_inputs,
            forecast_control=forecast_control,
            preflight_artifact_hash=index.packing_preflight_artifact_hash,
            source_ref=index.runtime_source_manifest,
        )

    def prepare(
        self, bootstrap: DevelopmentContinuationBootstrap
    ) -> PreparedDevelopmentContinuation:
        registration = self.registration()
        if bootstrap.adopter_registration_hash != registration.content_hash:
            raise DevelopmentContinuationError("bootstrap names another adopter")
        if (
            bootstrap.launcher_configuration_hash != self.launcher_configuration_hash
            or bootstrap.model_snapshot_manifest_hash
            != self.model_snapshot_manifest_hash
        ):
            raise DevelopmentContinuationError("bootstrap names another model stack")
        source_manifest = build_source_manifest(self.root, self.source_revision)
        if source_manifest.tree_sha256 != bootstrap.source_tree_hash:
            raise DevelopmentContinuationError(
                "current source tree differs from the fallback-selected freeze"
            )
        manifest = load_development_call_manifest(self.root)
        construction = DevelopmentConstructionConfiguration.load(
            cast(Path, self.construction_path)
        )
        neutral, visible, _certificates = load_development_prequery_evidence(
            self.root, manifest
        )
        recovered = self._recover_preparation(
            bootstrap=bootstrap,
            source_manifest=source_manifest,
            manifest=manifest,
            construction=construction,
            neutral=neutral,
            visible=visible,
        )
        if recovered is not None:
            return recovered
        if self.adapter_state_path.exists():
            if self.adapter_state_path.is_symlink():
                raise DevelopmentContinuationError("adapter state is unsafe")
            adapter_state = AdapterDurableState.model_validate_json(
                self.adapter_state_path.read_text(encoding="utf-8")
            )
            if adapter_state.query_access_events:
                raise DevelopmentContinuationError(
                    "query-blind preparation cannot begin after query access"
                )
            raise DevelopmentContinuationError(
                "query-blind preparation root is missing after adapter state creation"
            )
        execution_id = f"{bootstrap.owner_run_id}-development-v1"
        self._register_study_and_snapshots(
            execution_id=execution_id,
            manifest=manifest,
            construction=construction,
            source_tree_hash=bootstrap.source_tree_hash,
            neutral_by_unit=neutral,
        )
        preparations, preparation_refs, unit_bindings = self._build_preparations(
            manifest=manifest,
            construction=construction,
            neutral_by_unit=neutral,
        )
        seed_hash = development_seed_manifest_hash(self.root, manifest)
        validator_hash = development_validator_hash(self.root)
        run_configs, fixed_plans = build_development_run_configurations(
            root=self.root,
            manifest=manifest,
            construction=construction,
            tokenizer_manifest=self.tokenizer_manifest,
            model_stack_hash=self.launcher_configuration_hash,
            seed_manifest_hash=seed_hash,
            validator_hash=validator_hash,
        )
        config_refs = self._persist_configurations(run_configs)
        prequery_inputs = DevelopmentPrequeryInputs(
            source_tree_hash=bootstrap.source_tree_hash,
            selected_model_freeze_hash=bootstrap.selected_model_freeze_hash,
            upper_ontology_hash=construction.upper_ontology.content_hash,
            prompt_family_hash=_prompt_family_hash(self.root),
            output_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            validator_hash=validator_hash,
            run_condition_config_hashes=tuple(
                None
                if call.kind is DevelopmentCallKind.FIXED_SELECTION
                else run_configs[call.ordinal].content_hash
                for call in manifest.calls
            ),
            fixed_schema_derivation_plans=tuple(
                fixed_plans[ordinal] for ordinal in (17, 18, 19, 20)
            ),
            unit_bindings=unit_bindings,
        )
        preflight, preflight_artifact_hash = build_c1_packing_preflight(
            root=self.root,
            manifest=manifest,
            construction=construction,
            neutral_by_unit=neutral,
            run_configs=run_configs,
            tokenizer=self.tokenizer,
            tokenizer_manifest=self.tokenizer_manifest,
            artifacts=self.artifacts,
            clock=self.clock,
        )
        del preflight
        source_ref = persist_opaque_json(
            self.artifacts,
            source_manifest.to_dict(),
            object_kind="runtime_source_manifest",
            created_at=_strict_utc(self.clock),
        )
        forecast_receipt = build_development_forecast_receipt(
            root=self.root,
            ledger=self.artifacts.ledger,
            service=self.service,
            manifest=manifest,
            clock=self.clock,
            retry_amendment_sha256=self.retry_amendment_sha256,
            second_recovery_overlay_sha256=self.second_recovery_overlay_sha256,
            recovery_service_start_event_ids=self.recovery_service_start_event_ids,
        )
        if not forecast_receipt.admitted:
            raise DevelopmentContinuationError(
                "complete mandatory GPU manifest cannot fit below nine hours"
            )
        forecast_ref = persist_logical_record(
            self.artifacts,
            forecast_receipt,
            object_kind="development_forecast_receipt",
            created_at=forecast_receipt.created_at,
        )
        forecast_control = ForecastControl(
            forecast_receipt_hash=forecast_receipt.content_hash,
            gpu_call_inventory_file_sha256=manifest.gpu_call_inventory_file_sha256,
            post_development_mandatory_forecast_seconds=(
                forecast_receipt.post_development_mandatory_forecast_seconds
            ),
            scheduled_limit_seconds=forecast_receipt.scheduled_limit_seconds,
            hard_limit_seconds=forecast_receipt.hard_limit_seconds,
        )
        prequery_ref = persist_logical_record(
            self.artifacts,
            prequery_inputs,
            object_kind="development_prequery_inputs",
            created_at=_strict_utc(self.clock),
        )
        forecast_control_ref = persist_logical_record(
            self.artifacts,
            forecast_control,
            object_kind="forecast_control",
            created_at=_strict_utc(self.clock),
        )
        construction_ref = persist_logical_record(
            self.artifacts,
            construction,
            object_kind="development_construction_configuration",
            created_at=_strict_utc(self.clock),
        )
        identity = self._identity(bootstrap)
        execution_manifest = DevelopmentExecutionManifest(
            execution_id=execution_id,
            call_manifest_hash=manifest.content_hash,
            source_plan_hash=manifest.source_plan_hash,
            prequery_inputs_hash=prequery_inputs.content_hash,
            service_identity_hash=identity.content_hash,
            forecast_control_hash=forecast_control.content_hash,
        )
        index = DevelopmentPreparationIndex(
            index_id=f"{execution_id}-query-blind-preparation",
            bootstrap_hash=bootstrap.content_hash,
            execution_id=execution_id,
            execution_manifest_hash=execution_manifest.content_hash,
            call_manifest_hash=manifest.content_hash,
            prequery_inputs=prequery_ref,
            forecast_control=forecast_control_ref,
            forecast_receipt=forecast_ref,
            runtime_source_manifest=source_ref,
            construction_configuration=construction_ref,
            prequery_preparation_artifacts=preparation_refs,
            run_condition_config_artifacts=config_refs,
            packing_preflight_artifact_hash=preflight_artifact_hash,
            checkpoint_path=self.checkpoint_path.as_posix(),
            adapter_state_path=self.adapter_state_path.as_posix(),
            created_at=_strict_utc(self.clock),
        )
        index_artifact = self.artifacts.put_bytes(
            (index.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.development-preparation-index+json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=index.created_at,
        )
        pointer_payload = {
            "schema_version": "1.0.0",
            "bootstrap_hash": bootstrap.content_hash,
            "preparation_index_artifact_hash": index_artifact.content_hash,
        }
        _atomic_pointer(
            self.preparation_pointer_path,
            {**pointer_payload, "manifest_sha256": canonical_sha256(pointer_payload)},
        )
        return self._install_runtime_objects(
            bootstrap=bootstrap,
            manifest=manifest,
            construction=construction,
            neutral=neutral,
            visible=visible,
            preparations=preparations,
            preparation_refs=preparation_refs,
            run_configs=run_configs,
            fixed_plans=fixed_plans,
            seed_hash=seed_hash,
            validator_hash=validator_hash,
            prequery_inputs=prequery_inputs,
            forecast_control=forecast_control,
            preflight_artifact_hash=preflight_artifact_hash,
            source_ref=source_ref,
        )

    def _cpu_run_config(
        self,
        *,
        condition: ConditionName,
        unit_id: str,
        query_ordinal: int,
    ) -> RunConditionConfig:
        repository = cast(DevelopmentExecutionRepository, self._repository)
        c2 = next(
            call
            for call in repository.manifest.calls
            if call.unit_id == unit_id
            and call.kind is DevelopmentCallKind.C2_CONSTRUCTION
            and call.call_id.endswith(f"q{query_ordinal:02d}")
        )
        baseline = repository.run_configs[c2.ordinal]
        if condition is ConditionName.C0_CLASSICAL_PRE:
            return RunConditionConfig(
                config_id=f"cpu-c0-{unit_id}-q{query_ordinal:02d}",
                condition=condition,
                budgets=baseline.budgets,
                maximum_input_tokens=baseline.maximum_input_tokens,
                maximum_output_tokens=baseline.maximum_output_tokens,
                repair_attempt_budget=baseline.repair_attempt_budget,
                scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
                validator_hash=baseline.validator_hash,
                upper_ontology_hash=baseline.upper_ontology_hash,
            )
        c1 = repository.run_configs[
            next(
                call.ordinal
                for call in repository.manifest.calls
                if call.unit_id == unit_id
                and call.kind is DevelopmentCallKind.C1_PRECONSTRUCTION
            )
        ]
        return RunConditionConfig(
            config_id=f"cpu-c1-{unit_id}-q{query_ordinal:02d}",
            condition=condition,
            budgets=baseline.budgets,
            maximum_input_tokens=baseline.maximum_input_tokens,
            maximum_output_tokens=baseline.maximum_output_tokens,
            repair_attempt_budget=baseline.repair_attempt_budget,
            seed_block=c1.seed_block,
            model_stack_hash=c1.model_stack_hash,
            decoding_manifest_hash=c1.decoding_manifest_hash,
            decoding_family_hash=c1.decoding_family_hash,
            seed_manifest_hash=c1.seed_manifest_hash,
            resolved_seed=c1.resolved_seed,
            prompt_hash=c1.prompt_hash,
            output_schema_hash=c1.output_schema_hash,
            scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
            capability_manifest_hash=c1.capability_manifest_hash,
            validator_hash=c1.validator_hash,
            upper_ontology_hash=c1.upper_ontology_hash,
        )

    def _cpu_projection_receipts(self) -> tuple[str, ...]:
        repository = cast(DevelopmentExecutionRepository, self._repository)
        adapter = cast(ProductionDevelopmentServiceAdapter, self._adapter)
        if self._classical_builder is None:
            raise DevelopmentContinuationError("C0 production builder is unavailable")
        checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        barrier_hash = checkpoint.get("prequery_barrier", {}).get("content_hash")
        if not isinstance(barrier_hash, str):
            raise DevelopmentContinuationError("CPU projections lack the query barrier")
        barrier = adapter.persisted_prequery_barrier(barrier_hash)
        receipt_hashes: list[str] = []
        for condition in (ConditionName.C0_CLASSICAL_PRE, ConditionName.C1_LLM_PRE):
            for unit_id in DEVELOPMENT_UNIT_IDS:
                query_calls = tuple(
                    call
                    for call in repository.manifest.calls
                    if call.unit_id == unit_id
                    and call.kind is DevelopmentCallKind.C2_CONSTRUCTION
                )
                for query_ordinal, call in enumerate(query_calls, start=1):
                    assert call.query_stage is not None
                    opening = adapter.audited_opening(call.query_stage)
                    materialized = repository.materializations.get(
                        call.query_stage.staging_manifest_hash
                    )
                    if materialized is None:
                        raise DevelopmentContinuationError(
                            "CPU projection lacks an audited packet materialization"
                        )
                    preparation = repository.preparations[(unit_id, condition)]
                    config = self._cpu_run_config(
                        condition=condition,
                        unit_id=unit_id,
                        query_ordinal=query_ordinal,
                    )
                    inputs = ProduceInputs(
                        preparation=preparation,
                        snapshot=repository.neutral_by_unit[unit_id].snapshot,
                        packet=materialized.packet,
                        context=opening.context,
                        query_access=opening.access_event,
                        prequery_barrier=barrier,
                        query_processing_started_at=_strictly_after(
                            self.clock, opening.access_event.accessed_at
                        ),
                        packet_materialization=materialized.event,
                        upper_ontology=repository.construction.upper_ontology,
                        run_config=config,
                    )
                    attempt = (
                        self._classical_builder.produce(inputs)
                        if condition is ConditionName.C0_CLASSICAL_PRE
                        else LLMPreCondition().produce(inputs)
                    )
                    projection = attempt.projection
                    if attempt.outcome is not RunOutcome.SUCCEEDED or projection is None:
                        raise DevelopmentContinuationError(
                            "deterministic CPU projection did not succeed"
                        )
                    created_at = _strict_utc(self.clock)
                    attempt_ref = persist_logical_record(
                        self.artifacts,
                        attempt,
                        object_kind="condition_attempt",
                        created_at=created_at,
                    )
                    projection_ref = persist_logical_record(
                        self.artifacts,
                        projection,
                        object_kind="ontology_projection",
                        created_at=created_at,
                    )
                    config_ref = persist_logical_record(
                        self.artifacts,
                        config,
                        object_kind="run_condition_config",
                        created_at=created_at,
                    )
                    comparison = ComparisonInputManifest.from_inputs(inputs)
                    comparison_ref = persist_logical_record(
                        self.artifacts,
                        comparison,
                        object_kind="comparison_input_manifest",
                        created_at=created_at,
                    )
                    packet_ref = persist_logical_record(
                        self.artifacts,
                        materialized.packet,
                        object_kind="evidence_packet",
                        created_at=created_at,
                    )
                    context_ref = persist_logical_record(
                        self.artifacts,
                        opening.context,
                        object_kind="query_context",
                        created_at=created_at,
                    )
                    job = self.artifacts.ledger.create_or_resume_job(
                        {
                            "execution_id": repository.execution_id,
                            "condition": condition.value,
                            "unit_id": unit_id,
                            "query_ordinal": query_ordinal,
                            "kind": "cpu_projection",
                            "lifecycle_kind": "query_time_projection",
                            "query_access_event_hash": opening.access_event.content_hash,
                        },
                        release_class=ReleaseClass.PUBLIC,
                        created_at=barrier.sealed_at,
                    )
                    self.artifacts.ledger.link_job_to_study(
                        study_id=repository.execution_id,
                        job_id=job.job_id,
                        created_at=created_at,
                    )
                    self.artifacts.ledger.advance_job_lifecycle(
                        job.job_id,
                        (
                            (JobState.PREQUERY_SEALED, barrier.sealed_at),
                            (JobState.QUERY_REVEALED, opening.access_event.accessed_at),
                            (JobState.GENERATED, created_at),
                        ),
                    )
                    attempt_id = (
                        f"{repository.execution_id}-cpu-{condition.value}-"
                        f"{unit_id}-q{query_ordinal:02d}"
                    )
                    self.artifacts.ledger.record_attempt(
                        attempt_id=attempt_id,
                        job_id=job.job_id,
                        attempt_kind=AttemptKind.BASE,
                        input_hash=comparison.content_hash,
                        config_hash=config.content_hash,
                        seed=0 if config.seed_block is None else cast(int, config.resolved_seed),
                        created_at=created_at,
                    )
                    validation_id = f"{attempt_id}-validation"
                    self.artifacts.ledger.record_validation(
                        validation_id=validation_id,
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        input_artifact_hash=projection_ref.artifact_hash,
                        validator_manifest_hash=config.validator_hash,
                        validation_status=ValidationStatus.ACCEPTED,
                        evidence_support_status=EvidenceSupportStatus.NOT_APPLICABLE,
                        temporal_status=TemporalValidationStatus.NOT_APPLICABLE,
                        commitment_status=CommitmentCheckStatus.NOT_APPLICABLE,
                        semantic_assessment_scope=(
                            SemanticAssessmentScope.RUNTIME_STRUCTURAL_ONLY_NOT_ASSESSED
                        ),
                        created_at=created_at,
                    )
                    ledger_projection_id = f"{attempt_id}-projection"
                    seal = projection.construction_seal
                    if seal is None:
                        raise DevelopmentContinuationError(
                            "CPU projection lacks its prequery construction seal"
                        )
                    self.artifacts.ledger.record_projection(
                        projection_id=ledger_projection_id,
                        job_id=job.job_id,
                        validation_id=validation_id,
                        snapshot_id=inputs.snapshot.snapshot_id,
                        packet_input_id=inputs.packet.packet_id,
                        condition_id=condition.value,
                        context_hash=inputs.context.content_hash,
                        upper_ontology_hash=inputs.upper_ontology.content_hash,
                        construction_certificate_hash=seal.content_hash,
                        projection_artifact_hash=projection_ref.artifact_hash,
                        projection_semantic_hash=projection.content_hash,
                        release_class=ReleaseClass.PUBLIC,
                        finalized_at=created_at,
                    )
                    self.artifacts.ledger.advance_job_lifecycle(
                        job.job_id,
                        (
                            (JobState.PREQUERY_SEALED, barrier.sealed_at),
                            (JobState.QUERY_REVEALED, opening.access_event.accessed_at),
                            (JobState.GENERATED, created_at),
                            (JobState.VALIDATED, created_at),
                            (JobState.FINALIZED, created_at),
                        ),
                    )
                    receipt = DevelopmentCPUProjectionReceipt(
                        receipt_id=f"{attempt_id}-receipt",
                        condition=condition,
                        unit_id=unit_id,
                        query_ordinal=cast(Any, query_ordinal),
                        job_id=job.job_id,
                        attempt_id=attempt_id,
                        ledger_projection_id=ledger_projection_id,
                        ledger_validation_ids=(validation_id,),
                        condition_attempt=attempt_ref,
                        projection=projection_ref,
                        run_condition_config=config_ref,
                        comparison_input_manifest=comparison_ref,
                        evidence_packet=packet_ref,
                        packet_materialization_event_hash=materialized.event.content_hash,
                        query_context=context_ref,
                        created_at=created_at,
                    )
                    artifact = self.artifacts.put_bytes(
                        (receipt.to_canonical_json() + "\n").encode("utf-8"),
                        media_type=(
                            "application/vnd.story-projection."
                            "development-cpu-projection-receipt+json"
                        ),
                        release_class=ReleaseClass.PUBLIC,
                        created_at=created_at,
                    )
                    receipt_hashes.append(artifact.content_hash)
        if len(receipt_hashes) != DEVELOPMENT_CALL_COUNT:
            raise DevelopmentContinuationError("CPU projection inventory is incomplete")
        return tuple(receipt_hashes)

    def _assessment_provider(
        self,
        manifest: DevelopmentCallManifest,
        rows: tuple[DevelopmentITTRecord, ...],
    ) -> DevelopmentScientificAssessment:
        if (
            len(rows) != DEVELOPMENT_CALL_COUNT
            or tuple(row.call_id for row in rows)
            != tuple(call.call_id for call in manifest.calls)
            or any(row.outcome is not RunOutcome.SUCCEEDED for row in rows)
        ):
            return _failing_assessment()
        repository = cast(DevelopmentExecutionRepository, self._repository)
        adapter = cast(ProductionDevelopmentServiceAdapter, self._adapter)
        state = adapter._state()
        call_receipts = tuple(state.completed_receipts[call.call_id] for call in manifest.calls)
        service_results = tuple(state.completed_results[call.call_id] for call in manifest.calls)
        source_ref = cast(OpaqueJSONReference, self._runtime_source_reference)
        existing_bundle_hash = state.assessment_bundle_artifact_hash
        if existing_bundle_hash is not None:
            record = self.artifacts.ledger.get_artifact(existing_bundle_hash)
            bundle = DevelopmentAssessmentBundle.model_validate_json(
                self.artifacts.blobs.read_bytes(record)
            )
            expected = (
                bundle.call_manifest_hash,
                bundle.prequery_inputs_hash,
                bundle.source_tree_hash,
                bundle.runtime_source_manifest,
                bundle.benchmark_manifest_file_sha256,
                bundle.prequery_preparation_artifacts,
                bundle.call_receipt_artifact_hashes,
                bundle.service_result_artifact_hashes,
                bundle.packing_preflight_artifact_hash,
            )
            observed = (
                manifest.content_hash,
                cast(
                    PreparedDevelopmentContinuation, self._prepared
                ).prequery_inputs.content_hash,
                repository.source_tree_hash,
                source_ref,
                manifest.benchmark_manifest_file_sha256,
                repository.preparation_references,
                call_receipts,
                service_results,
                repository.packing_preflight_artifact_hash,
            )
            if expected != observed:
                raise DevelopmentContinuationError(
                    "persisted assessment bundle differs from the completed execution"
                )
            provider = self.assessment_factory(
                root=self.root,
                ledger=self.artifacts.ledger,
                blobs=self.artifacts.blobs,
                prequery_inputs=cast(
                    PreparedDevelopmentContinuation, self._prepared
                ).prequery_inputs,
                assessment_bundle_artifact_hash=existing_bundle_hash,
                assessment_manifest_path=self.assessment_manifest_path,
            )
            return provider(manifest, rows)
        cpu_receipts = self._cpu_projection_receipts()
        bundle = DevelopmentAssessmentBundle(
            bundle_id=f"{repository.execution_id}-assessment-bundle",
            call_manifest_hash=manifest.content_hash,
            prequery_inputs_hash=cast(
                PreparedDevelopmentContinuation, self._prepared
            ).prequery_inputs.content_hash,
            source_tree_hash=repository.source_tree_hash,
            runtime_source_manifest=source_ref,
            benchmark_manifest_file_sha256=manifest.benchmark_manifest_file_sha256,
            prequery_preparation_artifacts=repository.preparation_references,
            call_receipt_artifact_hashes=call_receipts,
            service_result_artifact_hashes=service_results,
            cpu_projection_receipt_artifact_hashes=cpu_receipts,
            packing_preflight_artifact_hash=repository.packing_preflight_artifact_hash,
            created_at=_strict_utc(self.clock),
        )
        artifact = self.artifacts.put_bytes(
            (bundle.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.development-assessment-bundle+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=bundle.created_at,
        )
        adapter._write_state(
            state,
            assessment_bundle_artifact_hash=artifact.content_hash,
        )
        provider = self.assessment_factory(
            root=self.root,
            ledger=self.artifacts.ledger,
            blobs=self.artifacts.blobs,
            prequery_inputs=cast(
                PreparedDevelopmentContinuation, self._prepared
            ).prequery_inputs,
            assessment_bundle_artifact_hash=artifact.content_hash,
            assessment_manifest_path=self.assessment_manifest_path,
        )
        return provider(manifest, rows)

    def adopt_and_run(
        self,
        handoff: DevelopmentContinuationHandoff,
        service: InjectedLiveDevelopmentService,
    ) -> DevelopmentExecutionResult:
        prepared = self._prepared
        adapter = self._adapter
        if prepared is None or adapter is None or self._repository is None:
            raise DevelopmentContinuationError("development continuation was not prepared")
        expected = (
            handoff.development_call_manifest_hash,
            handoff.development_source_plan_hash,
            handoff.development_execution_id,
            handoff.development_execution_manifest_hash,
            handoff.development_prequery_inputs_hash,
            handoff.live_service_identity,
        )
        observed = (
            prepared.call_manifest.content_hash,
            prepared.call_manifest.source_plan_hash,
            prepared.execution_id,
            prepared.execution_manifest_hash,
            prepared.prequery_inputs.content_hash,
            adapter.identity(),
        )
        if observed != expected or service is not adapter:
            raise DevelopmentContinuationError("handoff differs from prepared continuation")
        runner = DevelopmentRunner(
            execution_id=prepared.execution_id,
            manifest=prepared.call_manifest,
            prequery_inputs=prepared.prequery_inputs,
            service=service,
            checkpoint_path=prepared.checkpoint_path,
            forecast_control=prepared.forecast_control,
            assessment_provider=self._assessment_provider,
            clock=self.clock,
        )
        return runner.run()


def create_production_development_adopter(
    *,
    root: Path,
    service: MeteredGenerationService,
    artifacts: ArtifactStore,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    launcher_configuration_hash: str,
    model_snapshot_manifest_hash: str,
    source_association: Mapping[str, object],
    checkpoint_path: Path,
    assessment_factory: PostRunAssessmentFactory,
    retry_amendment_sha256: str | None = None,
    second_recovery_overlay_sha256: str | None = None,
    recovery_service_start_event_ids: tuple[str, ...] = (),
) -> ProductionDevelopmentContinuationAdopter:
    """Create the registered adopter without starting or touching the model."""

    revision = source_association.get("revision_label")
    tree_hash = source_association.get("local_tree_sha256")
    if not isinstance(revision, str) or not revision:
        raise DevelopmentContinuationError("source association lacks its revision label")
    if not isinstance(tree_hash, str) or len(tree_hash) != 64:
        raise DevelopmentContinuationError("source association lacks its tree hash")
    run_root = checkpoint_path.resolve(strict=False).parent
    return ProductionDevelopmentContinuationAdopter(
        root=root,
        service=service,
        artifacts=artifacts,
        tokenizer=tokenizer,
        tokenizer_manifest=tokenizer_manifest,
        launcher_configuration_hash=launcher_configuration_hash,
        model_snapshot_manifest_hash=model_snapshot_manifest_hash,
        source_revision=revision,
        checkpoint_path=run_root / "development.checkpoint.json",
        adapter_state_path=run_root / "development.adapter-state.json",
        preparation_pointer_path=run_root / "development.preparation.json",
        assessment_manifest_path=run_root / "development.assessment-input.json",
        assessment_factory=assessment_factory,
        retry_amendment_sha256=retry_amendment_sha256,
        second_recovery_overlay_sha256=second_recovery_overlay_sha256,
        recovery_service_start_event_ids=recovery_service_start_event_ids,
    )


__all__ = [
    "DevelopmentContinuationError",
    "PostRunAssessmentFactory",
    "ProductionDevelopmentContinuationAdopter",
    "build_c1_packing_preflight",
    "build_development_forecast_receipt",
    "build_development_run_configurations",
    "create_production_development_adopter",
    "development_seed_manifest_hash",
    "development_validator_hash",
    "load_development_prequery_evidence",
]
