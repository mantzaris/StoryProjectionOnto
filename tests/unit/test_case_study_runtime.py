from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.case_study_runtime import (
    CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES,
    AttestedSelectedModelFreeze,
    CaseOutputReceipt,
    CasePrequeryBarrierReceipt,
    CasePrequeryKind,
    CasePrequeryReceipt,
    CaseQueryAccessReceipt,
    CaseStudyAdmissionAttestation,
    CaseStudyExecutionPlan,
    CaseStudyInputAttestation,
    CaseStudyInputError,
    CaseStudyPlanError,
    CaseStudyResumeError,
    CaseStudyResumeManifest,
    CaseStudyRuntimePolicy,
    audit_case_study_resume,
    build_public_case_study_progress,
    compile_case_review_input_template,
    compile_case_study_execution_plan,
    initialize_case_study_resume,
    load_attested_restricted_case_study,
    load_attested_selected_model_freeze,
    validate_resume_successor,
    write_case_resume_manifest,
    write_restricted_case_record,
)
from story_projection_onto.contracts import (
    AbstractionLevel,
    ConditionName,
    ConstructionSeal,
    DiscoursePosition,
    OutputBudgets,
    PreQueryInventory,
    QueryContext,
    RetrievalMethod,
    RunOutcome,
    SpoilerHorizon,
    StoryTime,
    TemporalKind,
    canonical_sha256,
)
from story_projection_onto.novel_case import (
    CaseStudyPreregistration,
    CaseWindowRegistration,
    LawfulNovelSourceAuthorization,
    OperationalRetrievalReceipt,
    OperationalRetrievalRegistration,
    RestrictedNovelIndexManifest,
    build_restricted_novel_index,
    load_segmentation_config,
)

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def test_semantic_native_inventory_requires_complete_community_review() -> None:
    assert {
        "blinded_community_rubric_template",
        "blinded_community_source_manifest",
        "blinded_community_package",
        "blinded_community_rejoin",
        "blinded_community_completion",
        "blinded_community_finalization",
        "blinded_community_table",
    }.issubset(CASE_SEMANTIC_NATIVE_ARTIFACT_ROLES)


def _artificial_source() -> str:
    return """CHAPTER I

An invented amber courier crosses a painted bridge in this artificial fixture.

CHAPTER II

An invented council disputes a wholly synthetic report about a glass token.

CHAPTER III

An invented navigator revises a fictional route after a clock sounds.

CHAPTER IV

An invented archivist records a made-up denial in a paper ledger.
"""


def _horizon(number: int) -> SpoilerHorizon:
    return SpoilerHorizon(
        horizon_id=f"case-horizon-{number}",
        max_discourse_position=DiscoursePosition(passage_order=number),
    )


def _context(identifier: str, horizon: SpoilerHorizon) -> QueryContext:
    return QueryContext(
        context_id=identifier,
        wording=f"Inspect the artificial relation for {identifier}",
        lens="synthetic belief and temporal consequence",
        target="the artificial contextual relation",
        story_scope=StoryTime(kind=TemporalKind.UNKNOWN, reason="fixture scope"),
        spoiler_horizon=horizon,
        abstraction=AbstractionLevel.EVENT_ROLE,
        budgets=OutputBudgets(
            node_budget=10,
            assertion_budget=20,
            display_node_budget=10,
            display_assertion_budget=20,
        ),
        revealed_at=T0 + timedelta(minutes=30),
    )


def _window(number: int) -> CaseWindowRegistration:
    horizon = _horizon(number)
    return CaseWindowRegistration(
        window_id=f"case-window-{number}",
        chapter_ordinals=(number,),
        passage_ids=(f"passage-{number:06d}",),
        fixed_horizon=horizon,
        contexts=(
            _context(f"case-context-{number}-a", horizon),
            _context(f"case-context-{number}-b", horizon),
        ),
        contrast_dimensions=("belief_revelation",),
        selection_basis="artificial narrative criterion fixed before any condition output",
    )


def _preregistration(manifest: RestrictedNovelIndexManifest) -> CaseStudyPreregistration:
    horizon = _horizon(4)
    return CaseStudyPreregistration(
        preregistration_id="artificial-first-novel-preregistration",
        restricted_index_manifest_hash=manifest.content_hash,
        windows=(_window(1), _window(2), _window(3), _window(4)),
        operational_retrieval=OperationalRetrievalRegistration(
            operational_query_id="case-operational-query",
            context=_context("case-operational-context", horizon),
            frozen_fts_query="invented artificial token",
            top_k=12,
            max_evidence_tokens=6144,
        ),
        detailed_review_context_ids=(
            "case-context-1-a",
            "case-context-2-a",
            "case-context-3-a",
            "case-context-4-a",
        ),
        primary_seed=1_988_649_846,
        frozen_at=T0 + timedelta(minutes=1),
    )


def _write_json(path: Path, record: object) -> None:
    payload = (
        record.model_dump(mode="json")  # type: ignore[union-attr]
        if hasattr(record, "model_dump")
        else record
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _Fixture:
    restricted_root: Path
    index_path: Path
    manifest_path: Path
    preregistration_path: Path
    attestation_path: Path
    selected_model_freeze_path: Path
    admission: CaseStudyAdmissionAttestation
    selected_model_freeze: AttestedSelectedModelFreeze
    plan: CaseStudyExecutionPlan


def _selected_model_freeze_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "selected_llm_model_freeze",
        "model_candidate": "fallback",
        "repository": "Qwen/Qwen3-8B-AWQ",
        "revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
        "served_model_name": "qwen3-8b-awq-fallback",
        "activation_certificate_sha256": HASH_A,
        "cache_replacement_receipt_sha256": HASH_A,
        "snapshot_manifest_sha256": HASH_A,
        "launcher_configuration_sha256": HASH_A,
        "tokenizer_manifest_sha256": HASH_A,
        "micro_pilot_acceptance_receipt_sha256": HASH_A,
        "accepted_micro_pilot_result_sha256": HASH_A,
        "accepted_request_family_hash": HASH_A,
        "accepted_operator_gate_sha256": HASH_A,
        "accepted_grounding_horizon_gate_sha256": HASH_A,
        "runtime_stack_manifest_sha256": HASH_A,
        "gpu_hardware_manifest_sha256": HASH_A,
        "source_association_manifest_sha256": HASH_A,
        "source_tree_sha256": HASH_A,
        "runtime_feasibility_setting": {"enforce_eager": True},
        "scope": "artificial-fixture-only",
        "held_out_execution_allowed": False,
        "held_out_requires_separate_development_and_review_gates": True,
        "mixed_model_candidates_forbidden": True,
        "applies_symmetrically_to": ["C1", "C2", "A-FixedSelect", "LLM_ablations"],
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


@pytest.fixture
def case_fixture(tmp_path: Path) -> _Fixture:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    source = restricted / "artificial-source.txt"
    source.write_text(_artificial_source(), encoding="utf-8", newline="")
    index = restricted / "artificial-index.sqlite"
    config = load_segmentation_config(ROOT / "configs/case_study/indexing.json")
    manifest = build_restricted_novel_index(
        LawfulNovelSourceAuthorization(
            source_path=source,
            restricted_root=restricted,
            lawful_copy_attested=True,
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
        config,
        destination=index,
        built_at=T0,
    )
    assert manifest.passage_count == 4
    manifest_path = restricted / "index-manifest.json"
    _write_json(manifest_path, manifest)
    preregistration = _preregistration(manifest)
    preregistration_path = restricted / "preregistration.json"
    _write_json(preregistration_path, preregistration)
    attestation = CaseStudyInputAttestation(
        attestation_id="artificial-input-attestation",
        restricted_index_sha256=manifest.index_sha256,
        restricted_index_manifest_file_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        restricted_index_manifest_hash=manifest.content_hash,
        preregistration_file_sha256=hashlib.sha256(preregistration_path.read_bytes()).hexdigest(),
        preregistration_hash=preregistration.content_hash,
        attested_by="fixture-researcher",
        attested_at=T0 + timedelta(minutes=2),
    )
    attestation_path = restricted / "input-attestation.json"
    _write_json(attestation_path, attestation)
    loaded = load_attested_restricted_case_study(
        restricted_root=restricted,
        index_path=index,
        manifest_path=manifest_path,
        preregistration_path=preregistration_path,
        attestation_path=attestation_path,
    )
    selected_model_freeze_path = restricted / "selected-model-freeze.json"
    selected_freeze_payload = _selected_model_freeze_payload()
    _write_json(selected_model_freeze_path, selected_freeze_payload)
    admission = CaseStudyAdmissionAttestation(
        admission_id="artificial-admission",
        synthetic_run_closure_hash=HASH_A,
        timing_lineage_audit_hash=HASH_A,
        gold_firewall_audit_hash=HASH_A,
        registered_metric_regeneration_hash=HASH_A,
        blinded_error_review_hash=HASH_A,
        storage_preflight_hash=HASH_A,
        gpu_schedule_admission_hash=HASH_A,
        public_release_scan_hash=HASH_A,
        semantic_gate_bundle_hash=HASH_A,
        held_out_results_closure_hash=HASH_A,
        cumulative_ledger_sha256=HASH_A,
        gpu_event_inventory_hash=HASH_A,
        selected_model_freeze_file_sha256=hashlib.sha256(
            selected_model_freeze_path.read_bytes()
        ).hexdigest(),
        selected_model_freeze_hash=str(selected_freeze_payload["manifest_sha256"]),
        upper_ontology_hash=HASH_A,
        validator_hash=HASH_A,
        c1_capability_manifest_hash=HASH_A,
        c2_capability_manifest_hash=HASH_A,
        attested_by="fixture-researcher",
        attested_at=T0 + timedelta(minutes=3),
    )
    selected_model_freeze = load_attested_selected_model_freeze(
        restricted_root=restricted,
        selected_model_freeze_path=selected_model_freeze_path,
        admission=admission,
    )
    plan = compile_case_study_execution_plan(
        repository_root=ROOT,
        policy=CaseStudyRuntimePolicy.load(ROOT / "configs/case_study/runtime.json"),
        loaded=loaded,
        admission=admission,
        selected_model_freeze=selected_model_freeze,
        compiled_at=T0 + timedelta(minutes=4),
    )
    return _Fixture(
        restricted_root=restricted,
        index_path=index,
        manifest_path=manifest_path,
        preregistration_path=preregistration_path,
        attestation_path=attestation_path,
        selected_model_freeze_path=selected_model_freeze_path,
        admission=admission,
        selected_model_freeze=selected_model_freeze,
        plan=plan,
    )


def test_compiler_freezes_exact_4_8_1_calls_seed_and_safe_storage(
    case_fixture: _Fixture,
) -> None:
    plan = case_fixture.plan
    roles = [slot.role.value for slot in plan.gpu_call_slots]
    assert len(plan.windows) == 4
    assert len(plan.bounded_context_ids) == 8
    assert plan.bounded_projection_job_count == 24
    assert roles.count("c1_window_preconstruction") == 4
    assert roles.count("c2_bounded_construction") == 8
    assert roles.count("c2_full_index_operational") == 1
    assert tuple(slot.call_id for slot in plan.gpu_call_slots[:4]) == tuple(
        window.c1_preconstruction_call_id for window in plan.windows
    )
    assert tuple(slot.call_id for slot in plan.gpu_call_slots[4:12]) == tuple(
        call_id for window in plan.windows for call_id in window.c2_construction_call_ids
    )
    assert plan.gpu_call_slots[-1].call_id == plan.operational.c2_call_id
    assert {slot.resolved_vllm_seed for slot in plan.gpu_call_slots} == {1_988_649_846}
    assert plan.model_runtime.selected_model_candidate == "fallback"
    assert plan.model_runtime.model_repository == "Qwen/Qwen3-8B-AWQ"
    assert plan.model_runtime.model_revision == ("4da05a8edb55c6046cce958586c33b61da07bb79")
    assert (
        plan.model_runtime.selected_model_freeze_hash
        == case_fixture.selected_model_freeze.manifest_sha256
    )
    assert all(slot.backend == "vllm_gpu" for slot in plan.gpu_call_slots)
    assert plan.c0_runtime.query_blind_preconstruction
    assert plan.c0_runtime.deterministic_fixed_projection
    assert plan.operational.retrieval_method is RetrievalMethod.SQLITE_FTS5_BM25
    assert plan.operational.operational_only
    assert not plan.operational.causal_comparison_eligible
    assert plan.artifact_storage.raw_source_copy_count == 0
    assert plan.artifact_storage.normalized_source_copy_count == 0
    assert plan.artifact_storage.bounded_packet_storage == ("in_memory_with_hash_receipt_only")
    serialized = plan.to_canonical_json()
    assert str(case_fixture.restricted_root) not in serialized
    assert "invented amber courier" not in serialized
    assert "frozen_fts_query" not in serialized


def test_exact_attestation_detects_tamper_and_rejects_symlink(
    case_fixture: _Fixture,
) -> None:
    case_fixture.preregistration_path.write_text(
        case_fixture.preregistration_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(CaseStudyInputError, match="attestation mismatch"):
        load_attested_restricted_case_study(
            restricted_root=case_fixture.restricted_root,
            index_path=case_fixture.index_path,
            manifest_path=case_fixture.manifest_path,
            preregistration_path=case_fixture.preregistration_path,
            attestation_path=case_fixture.attestation_path,
        )
    alias = case_fixture.restricted_root / "attestation-alias.json"
    alias.symlink_to(case_fixture.attestation_path)
    with pytest.raises(CaseStudyInputError, match="symbolic-link"):
        load_attested_restricted_case_study(
            restricted_root=case_fixture.restricted_root,
            index_path=case_fixture.index_path,
            manifest_path=case_fixture.manifest_path,
            preregistration_path=case_fixture.preregistration_path,
            attestation_path=alias,
        )


def _admission_for_freeze(
    admission: CaseStudyAdmissionAttestation,
    path: Path,
    payload: dict[str, object],
) -> CaseStudyAdmissionAttestation:
    values = admission.model_dump(mode="python")
    values.update(
        {
            "selected_model_freeze_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_model_freeze_hash": payload["manifest_sha256"],
            "content_hash": "",
        }
    )
    return CaseStudyAdmissionAttestation.model_validate(values)


def test_selected_model_freeze_requires_exact_file_candidate_and_revision(
    case_fixture: _Fixture,
) -> None:
    case_fixture.selected_model_freeze_path.write_text(
        case_fixture.selected_model_freeze_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(CaseStudyInputError, match="content/file hash"):
        load_attested_selected_model_freeze(
            restricted_root=case_fixture.restricted_root,
            selected_model_freeze_path=case_fixture.selected_model_freeze_path,
            admission=case_fixture.admission,
        )

    with pytest.raises(CaseStudyInputError, match="unavailable"):
        load_attested_selected_model_freeze(
            restricted_root=case_fixture.restricted_root,
            selected_model_freeze_path=case_fixture.restricted_root / "missing-freeze.json",
            admission=case_fixture.admission,
        )

    wrong_candidate = _selected_model_freeze_payload()
    wrong_candidate["model_candidate"] = "primary"
    wrong_candidate["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in wrong_candidate.items() if key != "manifest_sha256"}
    )
    wrong_candidate_path = case_fixture.restricted_root / "wrong-candidate.json"
    _write_json(wrong_candidate_path, wrong_candidate)
    wrong_candidate_admission = _admission_for_freeze(
        case_fixture.admission,
        wrong_candidate_path,
        wrong_candidate,
    )
    with pytest.raises(CaseStudyInputError, match="symmetric fallback"):
        load_attested_selected_model_freeze(
            restricted_root=case_fixture.restricted_root,
            selected_model_freeze_path=wrong_candidate_path,
            admission=wrong_candidate_admission,
        )

    wrong_revision = _selected_model_freeze_payload()
    wrong_revision["revision"] = "0" * 40
    wrong_revision["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in wrong_revision.items() if key != "manifest_sha256"}
    )
    wrong_revision_path = case_fixture.restricted_root / "wrong-revision.json"
    _write_json(wrong_revision_path, wrong_revision)
    wrong_revision_admission = _admission_for_freeze(
        case_fixture.admission,
        wrong_revision_path,
        wrong_revision,
    )
    loaded_wrong_revision = load_attested_selected_model_freeze(
        restricted_root=case_fixture.restricted_root,
        selected_model_freeze_path=wrong_revision_path,
        admission=wrong_revision_admission,
    )
    loaded_case = load_attested_restricted_case_study(
        restricted_root=case_fixture.restricted_root,
        index_path=case_fixture.index_path,
        manifest_path=case_fixture.manifest_path,
        preregistration_path=case_fixture.preregistration_path,
        attestation_path=case_fixture.attestation_path,
    )
    with pytest.raises(CaseStudyPlanError, match="permitted fallback"):
        compile_case_study_execution_plan(
            repository_root=ROOT,
            policy=CaseStudyRuntimePolicy.load(ROOT / "configs/case_study/runtime.json"),
            loaded=loaded_case,
            admission=wrong_revision_admission,
            selected_model_freeze=loaded_wrong_revision,
            compiled_at=T0 + timedelta(minutes=4),
        )


def _snapshot_hash(number: int) -> str:
    return canonical_sha256({"artificial_snapshot": number})


def _artifact_hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _prequery_receipts(plan: CaseStudyExecutionPlan) -> tuple[CasePrequeryReceipt, ...]:
    receipts: list[CasePrequeryReceipt] = []
    for number, window in enumerate(plan.windows, start=1):
        snapshot_hash = _snapshot_hash(number)
        c0_seal = ConstructionSeal(
            seal_id=f"seal-c0-{number}",
            condition=ConditionName.C0_CLASSICAL_PRE,
            snapshot_hash=snapshot_hash,
            ontology_hash=_artifact_hash(f"c0-ontology-{number}"),
            constructed_at=T0 + timedelta(minutes=6),
            sealed_at=T0 + timedelta(minutes=7),
            sealed_object_ids=(f"c0-artificial-object-{number}",),
        )
        receipts.append(
            CasePrequeryReceipt(
                receipt_id=f"pre-c0-{number}",
                execution_plan_hash=plan.content_hash,
                preparation_job_id=window.c0_preparation_job_id,
                kind=CasePrequeryKind.C0_WINDOW_PRECONSTRUCTION,
                window_id=window.window_id,
                condition=ConditionName.C0_CLASSICAL_PRE,
                evidence_binding_hash=window.evidence_binding_hash,
                snapshot_hash=snapshot_hash,
                backend="spacy-ner-dependency-plus-deterministic-rules-v2",
                terminal_outcome=RunOutcome.SUCCEEDED,
                started_at=T0 + timedelta(minutes=5),
                completed_at=T0 + timedelta(minutes=8),
                construction_seal=c0_seal,
            )
        )
        c1_seal = ConstructionSeal(
            seal_id=f"seal-c1-{number}",
            condition=ConditionName.C1_LLM_PRE,
            snapshot_hash=snapshot_hash,
            ontology_hash=_artifact_hash(f"c1-ontology-{number}"),
            constructed_at=T0 + timedelta(minutes=9),
            sealed_at=T0 + timedelta(minutes=10),
            sealed_object_ids=(f"c1-artificial-object-{number}",),
        )
        receipts.append(
            CasePrequeryReceipt(
                receipt_id=f"pre-c1-{number}",
                execution_plan_hash=plan.content_hash,
                preparation_job_id=window.c1_preconstruction_call_id,
                kind=CasePrequeryKind.C1_WINDOW_PRECONSTRUCTION,
                window_id=window.window_id,
                condition=ConditionName.C1_LLM_PRE,
                evidence_binding_hash=window.evidence_binding_hash,
                snapshot_hash=snapshot_hash,
                backend="vllm_gpu",
                terminal_outcome=RunOutcome.SUCCEEDED,
                started_at=T0 + timedelta(minutes=8),
                completed_at=T0 + timedelta(minutes=11),
                base_attempt_artifact_hash=_artifact_hash(f"c1-attempt-{number}"),
                construction_seal=c1_seal,
            )
        )
        inventory = PreQueryInventory(
            inventory_id=f"inventory-c2-{number}",
            snapshot_hash=snapshot_hash,
            recorded_at=T0 + timedelta(minutes=12),
        )
        receipts.append(
            CasePrequeryReceipt(
                receipt_id=f"pre-c2-{number}",
                execution_plan_hash=plan.content_hash,
                preparation_job_id=window.c2_empty_inventory_job_id,
                kind=CasePrequeryKind.C2_BOUNDED_EMPTY_INVENTORY,
                window_id=window.window_id,
                condition=ConditionName.C2_LLM_QUERY,
                evidence_binding_hash=window.evidence_binding_hash,
                snapshot_hash=snapshot_hash,
                backend="controller",
                terminal_outcome=RunOutcome.SUCCEEDED,
                started_at=T0 + timedelta(minutes=12),
                completed_at=T0 + timedelta(minutes=13),
                empty_inventory=inventory,
            )
        )
    operational_snapshot = plan.restricted_index_manifest_hash
    inventory = PreQueryInventory(
        inventory_id="inventory-c2-operational",
        snapshot_hash=operational_snapshot,
        recorded_at=T0 + timedelta(minutes=14),
    )
    receipts.append(
        CasePrequeryReceipt(
            receipt_id="pre-c2-operational",
            execution_plan_hash=plan.content_hash,
            preparation_job_id=plan.operational.c2_empty_inventory_job_id,
            kind=CasePrequeryKind.C2_OPERATIONAL_EMPTY_INVENTORY,
            window_id="complete-query-blind-index",
            condition=ConditionName.C2_LLM_QUERY,
            evidence_binding_hash=plan.operational.full_index_binding_hash,
            snapshot_hash=operational_snapshot,
            backend="controller",
            terminal_outcome=RunOutcome.SUCCEEDED,
            started_at=T0 + timedelta(minutes=14),
            completed_at=T0 + timedelta(minutes=15),
            empty_inventory=inventory,
        )
    )
    return tuple(receipts)


def _barrier(
    plan: CaseStudyExecutionPlan,
    receipts: tuple[CasePrequeryReceipt, ...],
) -> CasePrequeryBarrierReceipt:
    return CasePrequeryBarrierReceipt(
        barrier_id="case-prequery-barrier",
        execution_plan_hash=plan.content_hash,
        prequery_receipt_hashes=tuple(item.content_hash for item in receipts),
        sealed_at=T0 + timedelta(minutes=20),
    )


def _bounded_accesses(
    plan: CaseStudyExecutionPlan,
    barrier: CasePrequeryBarrierReceipt,
) -> tuple[CaseQueryAccessReceipt, ...]:
    receipts: list[CaseQueryAccessReceipt] = []
    for number, window in enumerate(plan.windows, start=1):
        snapshot_hash = _snapshot_hash(number)
        packet_hash = _artifact_hash(f"bounded-packet-{number}")
        evidence_ids_hash = _artifact_hash(f"bounded-evidence-ids-{number}")
        for offset, (context_id, context_hash) in enumerate(
            zip(window.context_ids, window.context_hashes, strict=True)
        ):
            receipts.append(
                CaseQueryAccessReceipt(
                    access_id=f"access-{context_id}",
                    execution_plan_hash=plan.content_hash,
                    context_id=context_id,
                    context_hash=context_hash,
                    window_id=window.window_id,
                    prequery_barrier_hash=barrier.content_hash,
                    evidence_binding_hash=window.evidence_binding_hash,
                    packet_equality_group_id=window.packet_equality_group_id,
                    snapshot_hash=snapshot_hash,
                    packet_hash=packet_hash,
                    ordered_evidence_ids_hash=evidence_ids_hash,
                    evidence_count=1,
                    retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
                    registered_revealed_at=T0 + timedelta(minutes=30),
                    accessed_at=T0 + timedelta(minutes=31, seconds=offset),
                    packet_materialized_at=T0 + timedelta(minutes=19),
                    operational_only=False,
                    causal_comparison_eligible=True,
                )
            )
    return tuple(receipts)


def _failed_c2_output(
    plan: CaseStudyExecutionPlan,
    preparation: CasePrequeryReceipt,
    access: CaseQueryAccessReceipt,
    *,
    job_id: str,
    operational: bool = False,
) -> CaseOutputReceipt:
    base_hash = _artifact_hash(f"base-{job_id}")
    assert preparation.empty_inventory is not None
    return CaseOutputReceipt(
        receipt_id=f"output-{job_id}",
        execution_plan_hash=plan.content_hash,
        projection_job_id=job_id,
        condition=ConditionName.C2_LLM_QUERY,
        context_id=access.context_id,
        window_id=access.window_id,
        backend="vllm_gpu",
        query_access_receipt_hash=access.content_hash,
        preparation_receipt_hash=preparation.content_hash,
        evidence_binding_hash=access.evidence_binding_hash,
        packet_equality_group_id=access.packet_equality_group_id,
        snapshot_hash=access.snapshot_hash,
        packet_hash=access.packet_hash,
        pre_query_inventory_hash=preparation.empty_inventory.content_hash,
        base_attempt_artifact_hash=base_hash,
        repair_attempt_count=1,
        repair_attempt_artifact_hash=_artifact_hash(f"repair-{job_id}"),
        repair_parent_artifact_hash=base_hash,
        repair_reserve_class="reserve_standard",
        terminal_outcome=RunOutcome.INVALID,
        failure_lineage_hash=_artifact_hash(f"failure-{job_id}"),
        query_accessed_at=access.accessed_at,
        completed_at=access.accessed_at + timedelta(minutes=2),
        operational_only=operational,
        causal_comparison_eligible=not operational,
    )


def test_resume_audit_preserves_barrier_equality_itt_and_repair_lineage(
    case_fixture: _Fixture,
) -> None:
    plan = case_fixture.plan
    initial = initialize_case_study_resume(
        plan,
        initialized_at=T0 + timedelta(minutes=5),
    )
    prequery = _prequery_receipts(plan)
    barrier = _barrier(plan, prequery)
    accesses = _bounded_accesses(plan, barrier)
    preparation = next(
        item
        for item in prequery
        if item.preparation_job_id == plan.windows[0].c2_empty_inventory_job_id
    )
    failed = _failed_c2_output(
        plan,
        preparation,
        accesses[0],
        job_id=plan.windows[0].c2_construction_call_ids[0],
    )
    resume = CaseStudyResumeManifest(
        resume_id="resume-artificial-0001",
        sequence_number=1,
        parent_resume_hash=initial.content_hash,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=prequery,
        prequery_barrier=barrier,
        query_access_receipts=accesses,
        output_receipts=(failed,),
        updated_at=T0 + timedelta(minutes=40),
    )
    validate_resume_successor(initial, resume)
    status = audit_case_study_resume(plan, resume)
    assert status.completed_prequery_job_count == 13
    assert status.query_access_count == 8
    assert status.completed_output_count == 1
    assert status.non_succeeded_output_count == 1
    assert status.invalid_output_count == 1
    assert status.repaired_attempt_count == 1
    assert status.next_stage == "query_outputs"
    assert not status.complete
    public = build_public_case_study_progress(status)
    assert public.registered_full_index_operational_count == 1
    assert "packet" not in public.to_canonical_json()
    assert str(case_fixture.restricted_root) not in public.to_canonical_json()


def test_resume_rejects_unequal_bounded_packet_and_cpu_c2(
    case_fixture: _Fixture,
) -> None:
    plan = case_fixture.plan
    prequery = _prequery_receipts(plan)
    barrier = _barrier(plan, prequery)
    accesses = list(_bounded_accesses(plan, barrier))
    accesses[1] = accesses[1].model_copy(update={"packet_hash": HASH_B, "content_hash": ""})
    resume = CaseStudyResumeManifest(
        resume_id="resume-artificial-unequal",
        sequence_number=1,
        parent_resume_hash=HASH_A,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=prequery,
        prequery_barrier=barrier,
        query_access_receipts=tuple(accesses),
        updated_at=T0 + timedelta(minutes=40),
    )
    with pytest.raises(CaseStudyResumeError, match="unequal evidence"):
        audit_case_study_resume(plan, resume)

    preparation = next(
        item
        for item in prequery
        if item.preparation_job_id == plan.windows[0].c2_empty_inventory_job_id
    )
    valid = _failed_c2_output(
        plan,
        preparation,
        accesses[0],
        job_id=plan.windows[0].c2_construction_call_ids[0],
    )
    payload = valid.model_dump(mode="python")
    payload["backend"] = "spacy-ner-dependency-plus-deterministic-rules-v2"
    payload["content_hash"] = ""
    with pytest.raises(ValidationError, match="GPU LLM"):
        CaseOutputReceipt.model_validate(payload)


def test_resume_rejects_late_barrier_and_stale_seal_after_failed_prebuild(
    case_fixture: _Fixture,
) -> None:
    plan = case_fixture.plan
    prequery = _prequery_receipts(plan)
    late_barrier = CasePrequeryBarrierReceipt(
        barrier_id="case-late-prequery-barrier",
        execution_plan_hash=plan.content_hash,
        prequery_receipt_hashes=tuple(item.content_hash for item in prequery),
        sealed_at=T0 + timedelta(minutes=30),
    )
    late_accesses = _bounded_accesses(plan, late_barrier)
    late_resume = CaseStudyResumeManifest(
        resume_id="resume-artificial-late-barrier",
        sequence_number=1,
        parent_resume_hash=HASH_A,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=prequery,
        prequery_barrier=late_barrier,
        query_access_receipts=(late_accesses[0],),
        updated_at=T0 + timedelta(minutes=40),
    )
    with pytest.raises(CaseStudyResumeError, match="reveal must strictly follow"):
        audit_case_study_resume(plan, late_resume)

    original_c0 = prequery[0]
    failed_payload = original_c0.model_dump(mode="python")
    failed_payload.update(
        {
            "terminal_outcome": RunOutcome.FAILED,
            "construction_seal": None,
            "failure_lineage_hash": _artifact_hash("c0-prebuild-failure"),
            "content_hash": "",
        }
    )
    failed_c0 = CasePrequeryReceipt.model_validate(failed_payload)
    failed_prequery = (failed_c0, *prequery[1:])
    barrier = _barrier(plan, failed_prequery)
    access = _bounded_accesses(plan, barrier)[0]
    assert original_c0.construction_seal is not None
    stale_output = CaseOutputReceipt(
        receipt_id="output-stale-c0",
        execution_plan_hash=plan.content_hash,
        projection_job_id=plan.windows[0].c0_projection_job_ids[0],
        condition=ConditionName.C0_CLASSICAL_PRE,
        context_id=access.context_id,
        window_id=access.window_id,
        backend="spacy-ner-dependency-plus-deterministic-rules-v2",
        query_access_receipt_hash=access.content_hash,
        preparation_receipt_hash=failed_c0.content_hash,
        evidence_binding_hash=access.evidence_binding_hash,
        packet_equality_group_id=access.packet_equality_group_id,
        snapshot_hash=access.snapshot_hash,
        packet_hash=access.packet_hash,
        source_construction_seal_hash=original_c0.construction_seal.content_hash,
        terminal_outcome=RunOutcome.FAILED,
        failure_lineage_hash=_artifact_hash("c0-output-failure"),
        query_accessed_at=access.accessed_at,
        completed_at=access.accessed_at + timedelta(minutes=1),
        operational_only=False,
        causal_comparison_eligible=True,
    )
    stale_resume = CaseStudyResumeManifest(
        resume_id="resume-artificial-stale-c0",
        sequence_number=1,
        parent_resume_hash=HASH_A,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=failed_prequery,
        prequery_barrier=barrier,
        query_access_receipts=(access,),
        output_receipts=(stale_output,),
        updated_at=T0 + timedelta(minutes=40),
    )
    with pytest.raises(CaseStudyResumeError, match="stale source seal"):
        audit_case_study_resume(plan, stale_resume)


def test_operational_access_and_output_are_explicitly_noncausal(
    case_fixture: _Fixture,
) -> None:
    plan = case_fixture.plan
    prequery = _prequery_receipts(plan)
    barrier = _barrier(plan, prequery)
    preparation = prequery[-1]
    access = CaseQueryAccessReceipt(
        access_id="access-operational",
        execution_plan_hash=plan.content_hash,
        context_id=plan.operational.context_id,
        context_hash=plan.operational.context_hash,
        window_id="complete-query-blind-index",
        prequery_barrier_hash=barrier.content_hash,
        evidence_binding_hash=plan.operational.full_index_binding_hash,
        packet_equality_group_id="case-full-index-operational-packet",
        snapshot_hash=plan.restricted_index_manifest_hash,
        packet_hash=_artifact_hash("operational-packet"),
        ordered_evidence_ids_hash=canonical_sha256(("evidence-000001", "evidence-000002")),
        evidence_count=2,
        retrieval_method=RetrievalMethod.SQLITE_FTS5_BM25,
        registered_revealed_at=T0 + timedelta(minutes=30),
        accessed_at=T0 + timedelta(minutes=31),
        packet_materialized_at=T0 + timedelta(minutes=32),
        operational_retrieval_receipt=OperationalRetrievalReceipt(
            receipt_id="operational-retrieval-receipt",
            operational_query_id=plan.operational.operational_query_id,
            restricted_index_manifest_hash=plan.restricted_index_manifest_hash,
            packet_hash=_artifact_hash("operational-packet"),
            ordered_evidence_ids=("evidence-000001", "evidence-000002"),
            omitted_evidence_ids=("evidence-000003",),
            ranks={"evidence-000001": 1, "evidence-000002": 2},
            scores={"evidence-000001": -2.0, "evidence-000002": -1.0},
            eligible_before_top_k=3,
            horizon_rejected_match_count=0,
            omitted_by_top_k_or_token_cap=1,
            query_accessed_at=T0 + timedelta(minutes=31),
            packet_created_at=T0 + timedelta(minutes=32),
        ),
        operational_only=True,
        causal_comparison_eligible=False,
    )
    output = _failed_c2_output(
        plan,
        preparation,
        access,
        job_id=plan.operational.c2_call_id,
        operational=True,
    )
    resume = CaseStudyResumeManifest(
        resume_id="resume-artificial-operational",
        sequence_number=1,
        parent_resume_hash=HASH_A,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=prequery,
        prequery_barrier=barrier,
        query_access_receipts=(access,),
        output_receipts=(output,),
        updated_at=T0 + timedelta(minutes=40),
    )
    status = audit_case_study_resume(plan, resume)
    assert status.query_access_count == 1
    assert output.operational_only
    assert not output.causal_comparison_eligible

    bad = access.model_copy(
        update={
            "packet_materialized_at": access.accessed_at - timedelta(seconds=1),
            "content_hash": "",
        }
    )
    with pytest.raises(ValidationError, match="cannot predate"):
        CaseQueryAccessReceipt.model_validate(bad.model_dump())


def test_append_only_resume_detects_rewrite(case_fixture: _Fixture) -> None:
    plan = case_fixture.plan
    initial = initialize_case_study_resume(
        plan,
        initialized_at=T0 + timedelta(minutes=5),
    )
    first_receipt = _prequery_receipts(plan)[0]
    first = CaseStudyResumeManifest(
        resume_id="resume-artificial-0001",
        sequence_number=1,
        parent_resume_hash=initial.content_hash,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=(first_receipt,),
        updated_at=T0 + timedelta(minutes=9),
    )
    validate_resume_successor(initial, first)
    rewritten = first_receipt.model_copy(
        update={"failure_lineage_hash": HASH_A, "content_hash": ""}
    )
    # Use another valid C0 failure rather than mutating an internally invalid record.
    payload = rewritten.model_dump(mode="python")
    payload.update(
        {
            "terminal_outcome": RunOutcome.FAILED,
            "construction_seal": None,
            "content_hash": "",
        }
    )
    rewritten = CasePrequeryReceipt.model_validate(payload)
    second = CaseStudyResumeManifest(
        resume_id="resume-artificial-0002",
        sequence_number=2,
        parent_resume_hash=first.content_hash,
        execution_plan_hash=plan.content_hash,
        prequery_receipts=(rewritten,),
        updated_at=T0 + timedelta(minutes=10),
    )
    with pytest.raises(CaseStudyResumeError, match="rewrote"):
        validate_resume_successor(first, second)


def test_blank_review_template_has_eight_broad_and_four_detailed_units(
    case_fixture: _Fixture,
) -> None:
    template = compile_case_review_input_template(case_fixture.plan)
    assert len(template.units) == 8
    assert sum(unit.detailed_matching is not None for unit in template.units) == 4
    assert all(len(unit.dimensions) == 5 for unit in template.units)
    assert all(unit.judgment_state == "unpopulated" for unit in template.units)
    assert all(unit.reviewer_id is None and not unit.judgments for unit in template.units)
    assert all(len(unit.required_output_job_ids) == 3 for unit in template.units)
    payload = template.to_canonical_json()
    assert 'supported"' not in payload
    assert "invented amber courier" not in payload


def test_restricted_writers_refuse_overwrite_escape_and_symlink(
    case_fixture: _Fixture,
    tmp_path: Path,
) -> None:
    initial = initialize_case_study_resume(
        case_fixture.plan,
        initialized_at=T0 + timedelta(minutes=5),
    )
    states = case_fixture.restricted_root / "resume"
    states.mkdir()
    written = write_case_resume_manifest(
        initial,
        states,
        restricted_root=case_fixture.restricted_root,
    )
    assert written.name == f"{initial.content_hash}.json"
    assert (
        write_case_resume_manifest(
            initial,
            states,
            restricted_root=case_fixture.restricted_root,
        )
        == written
    )
    changed = initial.model_copy(
        update={"resume_id": "resume-artificial-changed", "content_hash": ""}
    )
    with pytest.raises(CaseStudyResumeError, match="overwrite"):
        write_restricted_case_record(
            changed,
            written,
            restricted_root=case_fixture.restricted_root,
        )
    with pytest.raises(CaseStudyResumeError, match="inside"):
        write_restricted_case_record(
            initial,
            tmp_path / "escaped.json",
            restricted_root=case_fixture.restricted_root,
        )
    real_directory = case_fixture.restricted_root / "real"
    real_directory.mkdir()
    alias = case_fixture.restricted_root / "alias"
    alias.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(CaseStudyResumeError, match="symbolic link"):
        write_restricted_case_record(
            initial,
            alias / "record.json",
            restricted_root=case_fixture.restricted_root,
        )
