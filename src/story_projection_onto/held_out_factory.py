"""Concrete, fail-closed production factory for the reviewed held-out run."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from story_projection_onto.benchmark_runtime import (
    EvidenceProjectionEquivalenceCertificate,
    GoldFirewallError,
    ModelEligibleWorldArtifact,
    NeutralEvidenceArtifact,
    RuntimeStagingManifest,
    preconstruction_request_payload,
    scan_model_payload,
    verify_neutral_evidence_projection,
)
from story_projection_onto.conditions.base import SCORED_PROJECTION_SCHEMA_HASH
from story_projection_onto.contracts import ConditionName, canonical_sha256
from story_projection_onto.development_adapter import (
    DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
    DevelopmentConstructionConfiguration,
    PackingTokenizer,
)
from story_projection_onto.development_continuation import development_validator_hash
from story_projection_onto.development_runtime import (
    DevelopmentExecutionResult,
    DevelopmentPrequeryInputs,
)
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    ResourceLimits,
    StorageAllocationPlan,
)
from story_projection_onto.fallback_acceptance import validate_source_association
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    ResourceSampler,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    capture_tokenizer_manifest,
)
from story_projection_onto.held_out_binding import (
    DevelopmentPredecessorLedgerBinding,
    capture_development_predecessor_ledger,
    verify_predecessor_gpu_event_chain,
)
from story_projection_onto.held_out_c0 import build_production_held_out_c0_adapter
from story_projection_onto.held_out_execution import (
    FrozenHeldOutSemanticExecutor,
    HeldOutSemanticState,
)
from story_projection_onto.held_out_primary import (
    HeldOutControlConfiguration,
    HeldOutControlError,
    ReviewedHeldOutPlan,
    _read_repository_stage,
)
from story_projection_onto.held_out_production import (
    HeldOutScheduleState,
    ProductionHeldOutBundle,
    SequentialVLLMConditionActivator,
    create_production_held_out_bundle,
)
from story_projection_onto.llm import render_condition_system_prompt
from story_projection_onto.model_gate import (
    FallbackModelPolicy,
    validate_fallback_snapshot_manifest,
)
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Ledger,
    ReleaseClass,
    StorageBudget,
    StoragePreflight,
    StorageReport,
)


class HeldOutFactoryError(RuntimeError):
    """A concrete held-out dependency differs from the accepted freeze."""


def _registered_phase_three_storage_preflight(
    repository: Path,
    storage: StoragePreflight,
) -> StorageReport:
    plan = StorageAllocationPlan.load(
        repository / "configs/study/storage_phase_allocations.json"
    )
    reservation = plan.reservation_for("phase_3")
    return storage.check(**reservation.preflight_arguments())


def _require_registered_phase_three_storage_preflight(
    repository: Path,
    storage: StoragePreflight,
    ledger: Ledger,
) -> StorageReport:
    report = _registered_phase_three_storage_preflight(repository, storage)
    ledger.record_storage_sample(
        report,
        phase="phase_3:held_out_primary:factory",
    )
    if not report.allowed:
        raise HeldOutFactoryError("held-out storage preflight failed")
    return report


_PREDECESSOR_REPLAY_DYNAMIC_FIELDS = {
    "predecessor_total_allocated_microseconds",
    "predecessor_gpu_summary_hash",
    "predecessor_gpu_service_session_count",
    "predecessor_gpu_service_session_rows",
    "predecessor_gpu_service_sessions_hash",
    "predecessor_gpu_service_journal_rows",
    "predecessor_gpu_service_journal_hash",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_existing_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HeldOutFactoryError(f"{label} must be an existing regular file")
    return path.resolve(strict=True)


def _canonical_restricted_root(repository: Path, candidate: Path) -> Path:
    """Bind private held-out state to the repository's sole restricted root."""

    repository = repository.resolve(strict=True)
    absolute = candidate if candidate.is_absolute() else repository / candidate
    if ".." in absolute.parts:
        raise HeldOutFactoryError("restricted root contains parent traversal")
    expected = repository / "artifacts" / "restricted"
    try:
        lexical = absolute.absolute().relative_to(repository)
    except ValueError as error:
        raise HeldOutFactoryError(
            "restricted root is not the canonical repository root"
        ) from error
    if lexical != Path("artifacts/restricted"):
        raise HeldOutFactoryError("restricted root is not the canonical repository root")
    probe = repository
    for component in lexical.parts:
        probe /= component
        if probe.is_symlink():
            raise HeldOutFactoryError("canonical restricted root has a symlinked ancestor")
    if not expected.is_dir():
        raise HeldOutFactoryError("canonical restricted root must be an existing directory")
    resolved = expected.resolve(strict=True)
    if resolved != expected:
        raise HeldOutFactoryError("canonical restricted root resolved outside its fixed path")
    return resolved


def _restricted_descendant(
    restricted_root: Path,
    candidate: Path,
    *,
    label: str,
) -> Path:
    """Resolve a private path while rejecting every symlink below its explicit root."""

    if restricted_root.is_symlink() or not restricted_root.is_dir():
        raise HeldOutFactoryError("restricted root must be an existing real directory")
    root = restricted_root.resolve(strict=True)
    absolute = candidate if candidate.is_absolute() else root / candidate
    resolved = absolute.resolve(strict=False)
    if ".." in absolute.parts:
        raise HeldOutFactoryError(f"{label} contains parent traversal")
    try:
        relative = resolved.relative_to(root)
        lexical_relative = absolute.absolute().relative_to(root)
    except ValueError as error:
        raise HeldOutFactoryError(f"{label} lies outside the restricted root") from error
    if relative == Path(".") or lexical_relative == Path("."):
        raise HeldOutFactoryError(f"{label} must be below the restricted root")
    probe = root
    for part in lexical_relative.parts:
        probe /= part
        if probe.is_symlink():
            raise HeldOutFactoryError(f"{label} has a symlinked ancestor")
    return resolved


def _selected_freeze(path: Path) -> Mapping[str, object]:
    resolved = _safe_existing_file(path, label="selected-model freeze")
    try:
        outer = json.loads(resolved.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HeldOutFactoryError("selected-model freeze is not valid JSON") from error
    if not isinstance(outer, Mapping):
        raise HeldOutFactoryError("selected-model freeze root must be an object")
    nested = outer.get("selected_model_freeze")
    value = nested if isinstance(nested, Mapping) else outer
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if value.get("manifest_sha256") != canonical_sha256(immutable):
        raise HeldOutFactoryError("selected-model freeze self-hash changed")
    required_hashes = (
        "snapshot_manifest_sha256",
        "launcher_configuration_sha256",
        "tokenizer_manifest_sha256",
        "source_association_manifest_sha256",
        "source_tree_sha256",
    )
    if any(
        not isinstance(value.get(name), str) or len(cast(str, value[name])) != 64
        for name in required_hashes
    ):
        raise HeldOutFactoryError("selected-model freeze lacks runtime lineage")
    if (
        value.get("kind") != "selected_llm_model_freeze"
        or value.get("model_candidate") != "fallback"
        or value.get("repository") != FALLBACK_MODEL_REPOSITORY
        or value.get("revision") != FALLBACK_MODEL_REVISION
        or value.get("served_model_name") != FALLBACK_SERVED_MODEL_NAME
        or value.get("mixed_model_candidates_forbidden") is not True
        or value.get("held_out_requires_separate_development_and_review_gates") is not True
        or value.get("applies_symmetrically_to") != ["C1", "C2", "A-FixedSelect", "LLM_ablations"]
    ):
        raise HeldOutFactoryError("selected-model freeze is not the symmetric fallback stack")
    return value


def _prompt_family_hash(root: Path) -> str:
    bindings = {
        condition.value: hashlib.sha256(
            render_condition_system_prompt(root, condition).encode()
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


def _load_development_result(
    repository: Path,
    configuration: HeldOutControlConfiguration,
) -> DevelopmentExecutionResult:
    if configuration.development_execution_result_file_sha256 == "PENDING":
        raise HeldOutFactoryError("passing development result is not frozen")
    path = _safe_existing_file(
        repository / configuration.development_execution_result_path,
        label="development result",
    )
    if _file_sha256(path) != configuration.development_execution_result_file_sha256:
        raise HeldOutFactoryError("development result file hash changed")
    result = DevelopmentExecutionResult.model_validate_json(path.read_bytes())
    if not result.gate.passed:
        raise HeldOutFactoryError("development predecessor did not pass")
    return result


def _load_prequery_inputs(
    artifacts: ArtifactStore,
    artifact_hash: str,
    expected_logical_hash: str,
) -> DevelopmentPrequeryInputs:
    record = artifacts.ledger.get_artifact(artifact_hash)
    value = DevelopmentPrequeryInputs.model_validate_json(
        artifacts.blobs.read_bytes(
            record,
            allow_restricted=record.release_class.value == "restricted",
        )
    )
    if value.content_hash != expected_logical_hash:
        raise HeldOutFactoryError("development prequery CAS root changed")
    return value


def _load_held_out_neutral_worlds(
    repository: Path,
    reviewed_plan: ReviewedHeldOutPlan,
) -> dict[str, Any]:
    loaded = {}
    for unit in reviewed_plan.call_manifest.units:
        try:
            visible_names, visible_files = _read_repository_stage(
                repository,
                unit.prequery_stage.relative_path,
                read_names=("manifest.json", "evidence.json"),
            )
            manifest = RuntimeStagingManifest.model_validate_json(
                visible_files["manifest.json"]
            )
            if (
                manifest.stage_kind.value != "prequery_evidence"
                or visible_names
                != frozenset({*manifest.file_names, manifest.manifest_file_name})
            ):
                raise HeldOutFactoryError(
                    "held-out prequery stage differs from its exact allowlist"
                )
            visible_payload = json.loads(visible_files["evidence.json"])
            if (
                not isinstance(visible_payload, dict)
                or visible_payload.get("content_hash") != manifest.artifact_hashes[0]
            ):
                raise HeldOutFactoryError(
                    "held-out prequery evidence differs from its manifest"
                )
            visible = ModelEligibleWorldArtifact.model_validate(visible_payload)
            scan_model_payload(visible.model_dump(mode="json"))
            preconstruction_request_payload(visible)

            stage_name = Path(unit.prequery_stage.relative_path).name
            suffix = stage_name.removeprefix("artifact_")
            if not suffix or stage_name != f"artifact_{suffix}":
                raise HeldOutFactoryError(
                    "held-out prequery stage lacks an opaque suffix"
                )
            neutral_relative = (
                "data/synthetic/condition_inputs/neutral_evidence/"
                f"neutral_{suffix}"
            )
            neutral_names, neutral_files = _read_repository_stage(
                repository,
                neutral_relative,
                read_names=(
                    "manifest.json",
                    "neutral_evidence.json",
                    "equivalence.json",
                ),
            )
            neutral_manifest = RuntimeStagingManifest.model_validate_json(
                neutral_files["manifest.json"]
            )
            if neutral_names != frozenset(
                {*neutral_manifest.file_names, neutral_manifest.manifest_file_name}
            ):
                raise HeldOutFactoryError(
                    "neutral held-out stage differs from its exact allowlist"
                )
            neutral_payload = json.loads(neutral_files["neutral_evidence.json"])
            certificate_payload = json.loads(neutral_files["equivalence.json"])
            if (
                not isinstance(neutral_payload, dict)
                or not isinstance(certificate_payload, dict)
                or neutral_payload.get("content_hash")
                != neutral_manifest.artifact_hashes[0]
                or certificate_payload.get("content_hash")
                != neutral_manifest.artifact_hashes[1]
            ):
                raise HeldOutFactoryError(
                    "neutral held-out artifacts differ from their manifest"
                )
            neutral = NeutralEvidenceArtifact.model_validate(neutral_payload)
            certificate = EvidenceProjectionEquivalenceCertificate.model_validate(
                certificate_payload
            )
            scan_model_payload(neutral.model_dump(mode="json"))
            verify_neutral_evidence_projection(neutral, visible, certificate)
        except HeldOutFactoryError:
            raise
        except (HeldOutControlError, GoldFirewallError, OSError, ValueError) as error:
            raise HeldOutFactoryError(
                "held-out evidence stage failed containment or integrity validation"
            ) from error
        if (
            neutral.snapshot.content_hash != unit.prequery_stage.snapshot_hash
            or certificate.model_visible_evidence_artifact_hash
            != unit.prequery_stage.evidence_artifact_hash
        ):
            raise HeldOutFactoryError("neutral/model-visible held-out evidence changed")
        loaded[unit.unit_id] = neutral
    if len(loaded) != 12:
        raise HeldOutFactoryError("held-out factory requires exactly twelve neutral worlds")
    return loaded


def _require_bound_successor_state(
    *,
    runtime_root: Path,
    reviewed_plan: ReviewedHeldOutPlan,
    development_result: DevelopmentExecutionResult,
    predecessor: DevelopmentPredecessorLedgerBinding,
) -> None:
    """Permit post-bound GPU rows only for this exact durable held-out resume."""

    schedule_path = runtime_root / "schedule-state.json"
    semantic_path = runtime_root / "semantic-state.json"
    try:
        schedule = HeldOutScheduleState.model_validate_json(
            _safe_existing_file(
                schedule_path,
                label="held-out successor schedule state",
            ).read_bytes()
        )
        semantic = HeldOutSemanticState.model_validate_json(
            _safe_existing_file(
                semantic_path,
                label="held-out successor semantic state",
            ).read_bytes()
        )
    except (OSError, ValueError) as error:
        raise HeldOutFactoryError(
            "GPU rows after the predecessor lack a typed held-out resume state"
        ) from error
    if (
        schedule.call_manifest_hash != reviewed_plan.call_manifest.content_hash
        or schedule.development_execution_result_hash != development_result.content_hash
        or schedule.development_predecessor_ledger_hash != predecessor.content_hash
        or semantic.call_manifest_hash != reviewed_plan.call_manifest.content_hash
        or not schedule.activation_intents
    ):
        raise HeldOutFactoryError(
            "GPU successor rows are not bound to this held-out execution"
        )


def create_frozen_production_held_out_bundle(
    *,
    repository: Path,
    reviewed_plan: ReviewedHeldOutPlan,
    configuration: HeldOutControlConfiguration,
    snapshot_path: Path,
    shared_cache: Path,
    verified_model_manifest_path: Path,
    selected_model_freeze_path: Path,
    source_association_path: Path,
    development_prequery_inputs_artifact_hash: str,
    ledger_path: Path,
    artifact_root: Path,
    runtime_root: Path,
    restricted_root: Path,
    quota_root: Path,
    predecessor_ledger_binding: DevelopmentPredecessorLedgerBinding | None = None,
    port: int = 8000,
) -> ProductionHeldOutBundle:
    """Bind accepted development to three sequential, independently loaded services.

    Construction performs CPU-only verification.  A vLLM process is created only
    when the reviewed controller activates the corresponding condition session.
    """

    repository = repository.resolve(strict=True)
    if configuration.production_adapter_factory == "PENDING":
        raise HeldOutFactoryError("production adapter factory is not frozen")
    if reviewed_plan.call_manifest.development_execution_result_hash == "PENDING":
        raise HeldOutFactoryError("held-out plan lacks an accepted development predecessor")
    restricted_root = _canonical_restricted_root(repository, restricted_root)
    development_result = _load_development_result(repository, configuration)
    if (
        development_result.content_hash
        != reviewed_plan.call_manifest.development_execution_result_hash
    ):
        raise HeldOutFactoryError("reviewed plan binds another development result")

    limits = ResourceLimits.load(repository / "configs/study/resource_limits.json")
    if (
        limits.scheduled_gpu_seconds != configuration.scheduled_gpu_seconds_limit
        or limits.hard_gpu_seconds != configuration.hard_gpu_seconds_limit
    ):
        raise HeldOutFactoryError("held-out GPU limits differ from global resource limits")
    quota_root = quota_root.resolve(strict=True)
    for label, candidate in (
        ("repository", repository),
        ("shared cache", shared_cache),
        ("ledger directory", ledger_path.parent),
        ("artifact root", artifact_root),
        ("runtime root", runtime_root),
        ("restricted root", restricted_root),
    ):
        try:
            candidate.resolve(strict=False).relative_to(quota_root)
        except ValueError as error:
            raise HeldOutFactoryError(f"{label} lies outside the controlled quota root") from error
    ledger_path = _restricted_descendant(
        restricted_root,
        ledger_path,
        label="GPU ledger",
    )
    artifact_root = _restricted_descendant(
        restricted_root,
        artifact_root,
        label="CAS root",
    )
    runtime_root = _restricted_descendant(
        restricted_root,
        runtime_root,
        label="runtime root",
    )
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(runtime_root, 0o700)
    ledger = Ledger(ledger_path)
    try:
        artifacts = ArtifactStore(BlobStore(artifact_root), ledger)
        storage = StoragePreflight(
            quota_root,
            controlled_paths=(
                repository,
                shared_cache,
                ledger_path.parent,
                artifact_root,
                runtime_root,
            ),
            budget=StorageBudget(
                total_allocation_bytes=limits.maximum_project_allocation_bytes,
                max_occupied_bytes=limits.maximum_project_occupied_bytes,
                min_headroom_bytes=limits.minimum_storage_headroom_bytes,
            ),
        )
        _require_registered_phase_three_storage_preflight(
            repository,
            storage,
            ledger,
        )
        if predecessor_ledger_binding is None:
            raise HeldOutFactoryError("exact development predecessor ledger binding is missing")
        if (
            predecessor_ledger_binding.development_execution_id
            != development_result.execution_id
            or predecessor_ledger_binding.development_execution_result_hash
            != development_result.content_hash
        ):
            raise HeldOutFactoryError(
                "development predecessor ledger binding names another result"
            )
        try:
            has_gpu_successors = verify_predecessor_gpu_event_chain(
                ledger,
                predecessor_ledger_binding,
            )
            replayed_predecessor = capture_development_predecessor_ledger(
                artifacts=artifacts,
                development=development_result,
                predecessor_gpu_event_count=(
                    predecessor_ledger_binding.predecessor_gpu_event_count
                ),
            )
        except (HeldOutControlError, OSError, ValueError, KeyError) as error:
            raise HeldOutFactoryError(
                "development predecessor ledger lineage cannot be replayed"
            ) from error
        if replayed_predecessor.model_dump(
            mode="python",
            exclude=_PREDECESSOR_REPLAY_DYNAMIC_FIELDS,
        ) != predecessor_ledger_binding.model_dump(
            mode="python",
            exclude=_PREDECESSOR_REPLAY_DYNAMIC_FIELDS,
        ):
            raise HeldOutFactoryError(
                "ledger does not contain the exact development predecessor chain"
            )
        if has_gpu_successors:
            _require_bound_successor_state(
                runtime_root=runtime_root,
                reviewed_plan=reviewed_plan,
                development_result=development_result,
                predecessor=predecessor_ledger_binding,
            )

        prequery = _load_prequery_inputs(
            artifacts,
            development_prequery_inputs_artifact_hash,
            development_result.prequery_inputs_hash,
        )
        freeze = _selected_freeze(selected_model_freeze_path)
        freeze_hash = cast(str, freeze["manifest_sha256"])
        if prequery.selected_model_freeze_hash != freeze_hash:
            raise HeldOutFactoryError("development inputs bind another selected model freeze")
        association = validate_source_association(
            source_association_path,
            source_root=repository,
        )
        if (
            association.get("manifest_sha256") != freeze["source_association_manifest_sha256"]
            or association.get("local_tree_sha256") != freeze["source_tree_sha256"]
            or prequery.source_tree_hash != freeze["source_tree_sha256"]
        ):
            raise HeldOutFactoryError("source association differs from accepted development")

        policy_path = repository / "configs/study/fallback_model.json"
        policy = FallbackModelPolicy.load(policy_path)
        snapshot_manifest = validate_fallback_snapshot_manifest(
            verified_model_manifest_path,
            policy=policy,
            policy_path=policy_path,
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
        )
        snapshot_hash = cast(str, snapshot_manifest["manifest_sha256"])
        if snapshot_hash != freeze["snapshot_manifest_sha256"]:
            raise HeldOutFactoryError("model snapshot differs from selected freeze")
        launcher = VLLMLaunchConfiguration.from_model_configuration(
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            model_configuration_path=repository / "configs/study/model.json",
            model_candidate="fallback",
            verified_snapshot_manifest_sha256=snapshot_hash,
            port=port,
        )
        if launcher.configuration_hash != freeze["launcher_configuration_sha256"]:
            raise HeldOutFactoryError("vLLM launcher differs from selected freeze")
        tokenizer_manifest = capture_tokenizer_manifest(
            snapshot_path,
            repository=FALLBACK_MODEL_REPOSITORY,
            revision=FALLBACK_MODEL_REVISION,
        )
        if tokenizer_manifest.manifest_sha256 != freeze["tokenizer_manifest_sha256"]:
            raise HeldOutFactoryError("tokenizer differs from selected freeze")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
            revision=FALLBACK_MODEL_REVISION,
        )
        construction = DevelopmentConstructionConfiguration.load(
            repository / DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG
        )
        seed_value = json.loads(
            (repository / configuration.seed_manifest_path).read_text(encoding="utf-8")
        )
        seed_hash = seed_value.get("content_hash") if isinstance(seed_value, dict) else None
        validator_hash = development_validator_hash(repository)
        if (
            construction.upper_ontology.content_hash != prequery.upper_ontology_hash
            or _prompt_family_hash(repository) != prequery.prompt_family_hash
            or prequery.output_schema_hash != SCORED_PROJECTION_SCHEMA_HASH
            or validator_hash != prequery.validator_hash
            or not isinstance(seed_hash, str)
        ):
            raise HeldOutFactoryError("development semantic stack changed")
        neutral = _load_held_out_neutral_worlds(repository, reviewed_plan)
        ledger.register_study(
            study_id=reviewed_plan.call_manifest.manifest_id,
            protocol_hash=reviewed_plan.call_manifest.content_hash,
            code_manifest_hash=cast(str, association["local_tree_sha256"]),
            configuration_hash=configuration.content_hash,
            release_class=ReleaseClass.RESTRICTED,
            created_at=datetime.now(UTC),
        )
        meter = AllocatedGPUMeter.from_limits(ledger, limits)
        sampler = ResourceSampler(limits=limits, storage=storage, ledger=ledger)

        def service_factory(condition: ConditionName) -> VLLMService:
            client = VLLMGuidedJSONClient(launcher.base_url)
            return VLLMService(
                configuration=launcher,
                client=client,
                meter=meter,
                log_path=runtime_root / f"{condition.value}.vllm.log",
                startup_resource_sampler=sampler,
                preflight_endpoint_check=lambda: client.health(0.25),
                readiness_check=lambda: client.ready(
                    2.0,
                    model_name=FALLBACK_SERVED_MODEL_NAME,
                ),
            )

        owner = reviewed_plan.call_manifest.manifest_id
        activators = {
            condition: SequentialVLLMConditionActivator(
                condition=condition,
                service_factory=lambda condition=condition: service_factory(condition),
                checkpoint_path=runtime_root / f"{condition.value}.service-checkpoint.json",
                owner_run_id=owner,
                launcher_configuration_hash=launcher.configuration_hash,
                model_snapshot_hash=snapshot_hash,
                selected_model_freeze_hash=freeze_hash,
                source_execution_hash=cast(str, association["local_tree_sha256"]),
                service_start_watchdog_seconds=(
                    configuration.service_start_watchdog_seconds
                ),
            )
            for condition in (
                ConditionName.C1_LLM_PRE,
                ConditionName.C2_LLM_QUERY,
                ConditionName.A_FIXED_SELECT,
            )
        }
        semantic = FrozenHeldOutSemanticExecutor(
            root=repository,
            call_manifest=reviewed_plan.call_manifest,
            construction=construction,
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            model_stack_hash=launcher.configuration_hash,
            seed_manifest_hash=seed_hash,
            validator_hash=validator_hash,
            selected_model_freeze_hash=freeze_hash,
            source_tree_hash=cast(str, association["local_tree_sha256"]),
            artifacts=artifacts,
            neutral_by_unit=neutral,
            state_path=runtime_root / "semantic-state.json",
        )
        cpu = build_production_held_out_c0_adapter(
            repository=repository,
            call_manifest=reviewed_plan.call_manifest,
            configuration=configuration,
            artifacts=artifacts,
            state_directory=runtime_root / "c0-state",
            c1_projection_delegate=semantic,
        )
        bundle = create_production_held_out_bundle(
            repository=repository,
            reviewed_plan=reviewed_plan,
            configuration=configuration,
            development_result=development_result,
            development_predecessor_ledger_hash=(
                predecessor_ledger_binding.content_hash
            ),
            artifacts=artifacts,
            tokenizer=cast(PackingTokenizer, tokenizer),
            neutral_by_model_visible_hash={
                unit.prequery_stage.evidence_artifact_hash: neutral[unit.unit_id]
                for unit in reviewed_plan.call_manifest.units
            },
            state_path=runtime_root / "schedule-state.json",
            semantic_executor=semantic,
            activators=activators,
            cpu=cpu,
        )
        bundle.close_callback = ledger.close
        return bundle
    except BaseException:
        ledger.close()
        raise


__all__ = [
    "HeldOutFactoryError",
    "_canonical_restricted_root",
    "_restricted_descendant",
    "create_frozen_production_held_out_bundle",
]
