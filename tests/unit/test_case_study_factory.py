from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import story_projection_onto.case_study_factory as factory_module
from story_projection_onto.case_study_execution import validate_case_admission_evidence
from story_projection_onto.case_study_factory import (
    CASE_PRODUCTION_FACTORY,
    CaseStudyFactoryError,
    compile_case_study_semantic_admission_bundle,
    create_frozen_production_case_study_bundle,
    preflight_frozen_production_case_study_bundle,
    stage_case_admission_evidence,
)
from story_projection_onto.case_study_runtime import (
    CaseBlindedErrorReviewGate,
    CaseGoldFirewallGate,
    CaseGpuScheduleGate,
    CaseMetricRegenerationGate,
    CaseNativeGateArtifactReference,
    CasePublicReleaseScanGate,
    CaseStoragePreflightGate,
    CaseStudyAdmissionAttestation,
    CaseStudySemanticAdmissionBundle,
    CaseSyntheticClosureGate,
    CaseTimingLineageGate,
)
from story_projection_onto.case_study_transition import (
    CaseStudyTransitionError,
    case_admission_transition_paths,
    verify_case_admission_staging_transition,
)
from story_projection_onto.contracts import ReleaseClass, canonical_json
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    GpuEventKind,
    Ledger,
)

T0 = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
HASH = "a" * 64
PLAN_HASH = "b" * 64
GATE_NAMES = (
    "synthetic_run_closure_hash",
    "timing_lineage_audit_hash",
    "gold_firewall_audit_hash",
    "registered_metric_regeneration_hash",
    "blinded_error_review_hash",
    "storage_preflight_hash",
    "gpu_schedule_admission_hash",
    "public_release_scan_hash",
)


def _shared_blobs(repository: Path) -> Path:
    storage_plan = repository / "configs" / "study" / "storage_phase_allocations.json"
    if not storage_plan.exists():
        storage_plan.parent.mkdir(parents=True, exist_ok=True)
        storage_plan.write_bytes(
            (
                Path(__file__).resolve().parents[2]
                / "configs"
                / "study"
                / "storage_phase_allocations.json"
            ).read_bytes()
        )
    root = repository / "artifacts" / "blobs" / "study"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _patch_execution_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    restricted_root: Path,
    admission: CaseStudyAdmissionAttestation,
    semantic_bundle: CaseStudySemanticAdmissionBundle,
) -> tuple[Path, SimpleNamespace]:
    """Supply only the three plan bindings exercised by staging.

    Full plan construction is intentionally outside these transition tests; the
    transition itself, ledger, and CAS remain real.
    """

    plan_path = restricted_root / "execution-plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    plan = SimpleNamespace(
        execution_id="case-study-transition-test",
        content_hash=PLAN_HASH,
        admission_attestation_hash=admission.content_hash,
        semantic_gate_bundle_hash=semantic_bundle.content_hash,
    )
    monkeypatch.setattr(
        factory_module,
        "load_case_study_execution_plan",
        lambda *_args, **_kwargs: plan,
    )
    return plan_path, plan


def test_semantic_replay_requires_canonical_phase4_siblings(tmp_path: Path) -> None:
    root = tmp_path / "phase4"
    assert factory_module._canonical_phase4_output_root(
        index_path=root / "analysis_index.json",
        table_manifest_path=root / "table_manifest.json",
    ) == root

    with pytest.raises(CaseStudyFactoryError, match="canonical sibling paths"):
        factory_module._canonical_phase4_output_root(
            index_path=root / "copied-analysis-index.json",
            table_manifest_path=root / "table_manifest.json",
        )


def test_semantic_replay_requires_canonical_community_review_bundles(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    rubric = repository / "configs/study/community_review_template.json"
    source = repository / "restricted/community-source/source_manifest.json"
    package = repository / "restricted/community-package/reviewer/review_package.json"
    rejoin = repository / "restricted/community-package/scorer_only/rejoin_map.json"
    completion = repository / "restricted/community-final/completion.json"
    finalization = repository / "restricted/community-final/finalization.json"
    table = repository / "restricted/community-final/community_blind_review.csv"

    assert factory_module._canonical_community_review_roots(
        evidence_root=repository,
        rubric_template_path=rubric,
        source_manifest_path=source,
        package_path=package,
        rejoin_path=rejoin,
        completion_path=completion,
        finalization_path=finalization,
        table_path=table,
    ) == (source.parent, package.parent.parent, finalization.parent)

    with pytest.raises(CaseStudyFactoryError, match="canonical sibling paths"):
        factory_module._canonical_community_review_roots(
            evidence_root=repository,
            rubric_template_path=rubric,
            source_manifest_path=source,
            package_path=package,
            rejoin_path=rejoin.with_name("copied-rejoin.json"),
            completion_path=completion,
            finalization_path=finalization,
            table_path=table,
        )


def test_semantic_replay_requires_exact_failure_source_file_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "failure-source"
    (root / "panels").mkdir(parents=True)
    (root / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "panels" / "failure-1.json").write_text("{}\n", encoding="utf-8")
    expected = frozenset({"source_manifest.json", "panels/failure-1.json"})
    assert factory_module._exact_regular_file_inventory(
        root,
        label="failure-source fixture",
    ) == expected

    (root / "shadow.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CaseStudyFactoryError, match="inventory changed"):
        factory_module._require_exact_regular_file_inventory(
            root,
            expected=expected,
            label="failure-source fixture",
        )


@pytest.mark.parametrize("inverted_pair", range(4))
def test_semantic_replay_rejects_inverted_community_review_chronology(
    inverted_pair: int,
) -> None:
    chronology = [T0 + timedelta(minutes=index) for index in range(5)]
    chronology[inverted_pair + 1] = chronology[inverted_pair] - timedelta(seconds=1)
    with pytest.raises(CaseStudyFactoryError, match="community-review chronology"):
        factory_module._validate_community_review_chronology(
            analysis_completed_at=chronology[0],
            source_selected_at=chronology[1],
            reviewer_completed_at=chronology[2],
            binding_gate_completed_at=chronology[3],
            semantic_bundle_frozen_at=chronology[4],
        )


def test_semantic_replay_rejects_community_panel_identity_leaks() -> None:
    factory_module._assert_neutral_community_panel_payload(
        b'{"condition_blind":true,"nodes":[]}',
        forbidden_identity_values=("private-world",),
    )
    with pytest.raises(CaseStudyFactoryError, match="rejoin field"):
        factory_module._assert_neutral_community_panel_payload(
            b'{"condition":"C2LLMQuery"}',
            forbidden_identity_values=("private-world",),
        )
    for forbidden_key in ("gold", "gold_hash", "expected_effect"):
        with pytest.raises(CaseStudyFactoryError, match="rejoin field"):
            factory_module._assert_neutral_community_panel_payload(
                json.dumps({"display": {forbidden_key: "hidden"}}).encode("utf-8"),
                forbidden_identity_values=("private-world",),
            )
    with pytest.raises(CaseStudyFactoryError, match="rejoin identity"):
        factory_module._assert_neutral_community_panel_payload(
            b'{"node_id":"private-world"}',
            forbidden_identity_values=("private-world",),
        )
    with pytest.raises(CaseStudyFactoryError, match="rejoin identity"):
        factory_module._assert_neutral_community_panel_payload(
            b'{"label":"A prefix PRIVATE-WORLD suffix"}',
            forbidden_identity_values=("private-world",),
        )


def test_semantic_replay_requires_content_addressed_review_roots(tmp_path: Path) -> None:
    expected = "a" * 64
    factory_module._require_content_addressed_root(
        tmp_path / expected,
        expected_hash=expected,
        label="fixture",
    )
    with pytest.raises(CaseStudyFactoryError, match="content hash"):
        factory_module._require_content_addressed_root(
            tmp_path / "human-readable-copy",
            expected_hash=expected,
            label="fixture",
        )


@pytest.mark.parametrize("inverted_pair", range(5))
def test_semantic_replay_rejects_inverted_error_review_chronology(
    inverted_pair: int,
) -> None:
    chronology = [T0 + timedelta(minutes=index) for index in range(6)]
    chronology[inverted_pair + 1] = chronology[inverted_pair] - timedelta(seconds=1)
    with pytest.raises(CaseStudyFactoryError, match="chronology is not monotonic"):
        factory_module._validate_error_review_chronology(
            analysis_completed_at=chronology[0],
            source_selected_at=chronology[1],
            reviewer_completed_at=chronology[2],
            adjudicated_at=chronology[3],
            review_gate_completed_at=chronology[4],
            semantic_bundle_frozen_at=chronology[5],
        )


def _admission_and_gates(
    restricted_root: Path,
) -> tuple[Path, Path, CaseStudyAdmissionAttestation, dict[str, Path]]:
    common = {
        "study_id": "artificial-study",
        "source_tree_sha256": HASH,
        "selected_model_freeze_hash": HASH,
        "completed_at": T0 - timedelta(minutes=2),
    }
    closure = CaseSyntheticClosureGate(
        **common,
        held_out_execution_plan_hash=HASH,
        held_out_results_closure_hash=HASH,
        synthetic_benchmark_hash=HASH,
        output_receipt_inventory_hash=HASH,
    )
    gates = {
        "synthetic_run_closure_hash": closure,
        "timing_lineage_audit_hash": CaseTimingLineageGate(
            **common,
            held_out_results_closure_hash=HASH,
            cumulative_ledger_sha256=HASH,
            gpu_event_inventory_hash=HASH,
            actual_allocated_gpu_seconds=1.0,
        ),
        "gold_firewall_audit_hash": CaseGoldFirewallGate(
            **common,
            held_out_execution_plan_hash=HASH,
            held_out_results_closure_hash=HASH,
            synthetic_benchmark_hash=HASH,
            model_visible_payload_inventory_hash=HASH,
            scorer_only_inventory_hash=HASH,
        ),
        "registered_metric_regeneration_hash": CaseMetricRegenerationGate(
            **common,
            held_out_results_closure_hash=HASH,
            synthetic_benchmark_hash=HASH,
            registered_analysis_manifest_hash=HASH,
            canonical_table_manifest_hash=HASH,
        ),
        "blinded_error_review_hash": CaseBlindedErrorReviewGate(
            **common,
            held_out_results_closure_hash=HASH,
            output_receipt_inventory_hash=HASH,
            frozen_selection_manifest_hash=HASH,
            condition_alias_manifest_hash=HASH,
            completed_response_hash=HASH,
            adjudication_hash=HASH,
            reviewed_unit_count=3,
        ),
        "storage_preflight_hash": CaseStoragePreflightGate(
            **common,
            cumulative_ledger_sha256=HASH,
            gpu_event_inventory_hash=HASH,
            occupied_bytes=1_000,
            projected_occupied_bytes_after_case=2_000,
            filesystem_free_bytes=6_000_000_000,
        ),
        "gpu_schedule_admission_hash": CaseGpuScheduleGate(
            **common,
            cumulative_ledger_sha256=HASH,
            gpu_event_inventory_hash=HASH,
            actual_allocated_gpu_seconds=1.0,
            projected_scheduled_gpu_seconds=2_851.0,
            projected_hard_gpu_seconds=2_851.0,
        ),
        "public_release_scan_hash": CasePublicReleaseScanGate(
            **common,
            canonical_table_manifest_hash=HASH,
            public_candidate_manifest_hash=HASH,
            release_scan_receipt_hash=HASH,
        ),
    }
    paths: dict[str, Path] = {}
    for name, value in gates.items():
        path = restricted_root / f"{name}.json"
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        paths[name] = path
    semantic_path = restricted_root / "semantic-gates.json"
    dummy = restricted_root / "native-dummy.json"
    dummy.write_text('{"dummy":true}\n', encoding="utf-8")
    dummy_sha = __import__("hashlib").sha256(dummy.read_bytes()).hexdigest()
    native = tuple(
        CaseNativeGateArtifactReference(
            role=role,
            relative_path=dummy.name,
            file_sha256=dummy_sha,
            logical_content_hash=hashlib.sha256(
                canonical_json({"dummy": True}).encode("utf-8")
            ).hexdigest(),
        )
        for role in sorted(factory_module._NATIVE_GATE_ROLES)
    )
    semantic = CaseStudySemanticAdmissionBundle(
        bundle_id="artificial-semantic-gates",
        synthetic_run_closure=gates["synthetic_run_closure_hash"],
        timing_lineage_audit=gates["timing_lineage_audit_hash"],
        gold_firewall_audit=gates["gold_firewall_audit_hash"],
        registered_metric_regeneration=gates["registered_metric_regeneration_hash"],
        blinded_error_review=gates["blinded_error_review_hash"],
        storage_preflight=gates["storage_preflight_hash"],
        gpu_schedule_admission=gates["gpu_schedule_admission_hash"],
        public_release_scan=gates["public_release_scan_hash"],
        native_artifacts=native,
        frozen_at=T0 - timedelta(minutes=1),
    )
    semantic_path.write_text(canonical_json(semantic) + "\n", encoding="utf-8")
    admission = CaseStudyAdmissionAttestation(
        admission_id="artificial-case-admission",
        **{name: gates[name].content_hash for name in GATE_NAMES},
        semantic_gate_bundle_hash=semantic.content_hash,
        held_out_results_closure_hash=HASH,
        cumulative_ledger_sha256=HASH,
        gpu_event_inventory_hash=HASH,
        selected_model_freeze_file_sha256=HASH,
        selected_model_freeze_hash=HASH,
        upper_ontology_hash=HASH,
        validator_hash=HASH,
        c1_capability_manifest_hash=HASH,
        c2_capability_manifest_hash=HASH,
        attested_by="artificial fixture",
        attested_at=T0,
    )
    path = restricted_root / "admission.json"
    path.write_text(canonical_json(admission) + "\n", encoding="utf-8")
    return path, semantic_path, admission, paths


def test_stage_case_admission_evidence_is_restricted_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, admission, gate_paths = _admission_and_gates(restricted)
    semantic = CaseStudySemanticAdmissionBundle.model_validate_json(
        semantic_path.read_bytes()
    )
    plan_path, plan = _patch_execution_plan(
        monkeypatch,
        restricted_root=restricted,
        admission=admission,
        semantic_bundle=semantic,
    )
    ledger_path = restricted / "study.sqlite"
    artifact_root = _shared_blobs(tmp_path)
    ledger = Ledger(ledger_path)
    ledger.record_gpu_event(
        event_id="artificial-predecessor",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1.0,
        started_at=T0 - timedelta(seconds=1),
        ended_at=T0,
        succeeded=True,
        details={"artificial_fixture": True},
    )
    ledger.close()
    replays: list[dict[str, object]] = []
    monkeypatch.setattr(
        factory_module,
        "replay_case_study_semantic_admission",
        lambda **values: replays.append(values),
    )
    bundle_path = restricted / "staged" / "bundle.json"
    reference_path = restricted / "staged" / "reference.json"
    transition_directory = restricted / "staged" / "transition"

    bundle, reference, transition_receipt = stage_case_admission_evidence(
        restricted_root=restricted,
        semantic_evidence_root=tmp_path,
        execution_plan_path=plan_path,
        admission_attestation_path=admission_path,
        semantic_gate_bundle_path=semantic_path,
        gate_paths=gate_paths,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        transition_directory=transition_directory,
        bundle_output_path=bundle_path,
        reference_output_path=reference_path,
        staged_at=T0 + timedelta(minutes=1),
    )

    assert bundle.admission_attestation_hash == admission.content_hash
    assert len(bundle.evidence) == 8
    assert bundle_path.is_file()
    assert reference_path.is_file()
    assert reference.object_kind == "case_admission_evidence_bundle"
    assert CASE_PRODUCTION_FACTORY.endswith("create_frozen_production_case_study_bundle")
    assert len(replays) == 1
    assert replays[0]["ledger_path"] == ledger_path
    assert replays[0]["evidence_root"] == tmp_path
    assert replays[0]["blob_root"] == artifact_root
    transition_paths = case_admission_transition_paths(
        restricted_root=restricted,
        transition_directory=transition_directory,
    )
    assert transition_paths.h0_ledger.is_file()
    assert transition_paths.h1_ledger.is_file()
    assert transition_receipt.execution_plan_hash == plan.content_hash
    assert transition_receipt.evidence_bundle_hash == bundle.content_hash
    assert len(transition_receipt.added_artifacts) == 9
    verified = verify_case_admission_staging_transition(
        restricted_root=restricted,
        ledger_path=ledger_path,
        cas_root=artifact_root,
        transition_directory=transition_directory,
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission.content_hash,
        semantic_bundle_hash=semantic.content_hash,
        evidence_bundle_hash=bundle.content_hash,
        evidence_bundle_artifact_hash=reference.artifact_hash,
        evidence_bundle_reference_hash=reference.content_hash,
    )
    assert verified.current_matches_archived_h1
    assert verified.intent.h0_ledger.file_sha256 != verified.archived_h1_ledger_sha256
    assert verified.archived_h1_ledger_sha256 == transition_receipt.h1_ledger.file_sha256
    ledger = Ledger(ledger_path)
    artifacts = ArtifactStore(
        BlobStore(artifact_root, compression=Compression.ZSTD),
        ledger,
    )
    validate_case_admission_evidence(
        admission=admission,
        bundle=bundle,
        artifacts=artifacts,
    )
    ledger.close()

    replayed, replayed_reference, replayed_transition_receipt = (
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=plan_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=artifact_root,
            transition_directory=transition_directory,
            bundle_output_path=bundle_path,
            reference_output_path=reference_path,
            staged_at=T0 + timedelta(minutes=1),
        )
    )
    assert replayed == bundle
    assert replayed_reference == reference
    assert replayed_transition_receipt == transition_receipt
    assert len(replays) == 2
    assert replays[1]["ledger_path"] == transition_paths.h0_ledger
    assert replays[1]["cumulative_ledger_override"] == transition_paths.h0_ledger


def test_execution_semantic_replay_uses_archived_h0_and_binds_proven_h1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, admission, gate_paths = _admission_and_gates(
        restricted
    )
    semantic = CaseStudySemanticAdmissionBundle.model_validate_json(
        semantic_path.read_bytes()
    )
    plan_path, plan = _patch_execution_plan(
        monkeypatch,
        restricted_root=restricted,
        admission=admission,
        semantic_bundle=semantic,
    )
    ledger_path = restricted / "study.sqlite"
    with Ledger(ledger_path) as ledger:
        ledger.record_gpu_event(
            event_id="artificial-predecessor",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=1.0,
            started_at=T0 - timedelta(seconds=1),
            ended_at=T0,
            succeeded=True,
            details={"artificial_fixture": True},
        )
    blob_root = _shared_blobs(tmp_path)
    transition_directory = restricted / "staged" / "transition"
    replays: list[dict[str, object]] = []
    monkeypatch.setattr(
        factory_module,
        "replay_case_study_semantic_admission",
        lambda **values: replays.append(values),
    )
    evidence_bundle, evidence_reference, transition_receipt = (
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=plan_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=blob_root,
            transition_directory=transition_directory,
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )
    )
    paths = case_admission_transition_paths(
        restricted_root=restricted,
        transition_directory=transition_directory,
    )
    replays.clear()

    with pytest.raises(CaseStudyFactoryError, match="archived staging H1"):
        factory_module._replay_execution_semantic_admission(
            plan=plan,
            admission=admission,
            bundle=semantic,
            evidence_bundle=evidence_bundle,
            evidence_bundle_reference=evidence_reference,
            repository=tmp_path,
            restricted_root=restricted,
            ledger_path=ledger_path,
            blob_root=blob_root,
            transition_directory=transition_directory,
            expected_predecessor_ledger_sha256=HASH,
            require_current_h1=True,
        )
    assert replays == []

    verified = factory_module._replay_execution_semantic_admission(
        plan=plan,
        admission=admission,
        bundle=semantic,
        evidence_bundle=evidence_bundle,
        evidence_bundle_reference=evidence_reference,
        repository=tmp_path,
        restricted_root=restricted,
        ledger_path=ledger_path,
        blob_root=blob_root,
        transition_directory=transition_directory,
        expected_predecessor_ledger_sha256=transition_receipt.h1_ledger.file_sha256,
        require_current_h1=True,
    )
    assert verified.current_matches_archived_h1
    assert replays == [
        {
            "bundle": semantic,
            "evidence_root": tmp_path,
            "ledger_path": paths.h0_ledger,
            "blob_root": blob_root,
            "cumulative_ledger_override": paths.h0_ledger,
        }
    ]


def test_stage_retry_rejects_tampered_archived_h0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, admission, gate_paths = _admission_and_gates(
        restricted
    )
    semantic = CaseStudySemanticAdmissionBundle.model_validate_json(
        semantic_path.read_bytes()
    )
    plan_path, _plan = _patch_execution_plan(
        monkeypatch,
        restricted_root=restricted,
        admission=admission,
        semantic_bundle=semantic,
    )
    ledger_path = restricted / "study.sqlite"
    with Ledger(ledger_path) as ledger:
        ledger.record_gpu_event(
            event_id="artificial-predecessor",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=1.0,
            started_at=T0 - timedelta(seconds=1),
            ended_at=T0,
            succeeded=True,
        )
    blob_root = _shared_blobs(tmp_path)
    transition_directory = restricted / "staged" / "transition"
    monkeypatch.setattr(
        factory_module,
        "replay_case_study_semantic_admission",
        lambda **_: None,
    )
    staging_arguments = {
        "restricted_root": restricted,
        "semantic_evidence_root": tmp_path,
        "execution_plan_path": plan_path,
        "admission_attestation_path": admission_path,
        "semantic_gate_bundle_path": semantic_path,
        "gate_paths": gate_paths,
        "ledger_path": ledger_path,
        "artifact_root": blob_root,
        "transition_directory": transition_directory,
        "bundle_output_path": restricted / "staged" / "bundle.json",
        "reference_output_path": restricted / "staged" / "reference.json",
        "staged_at": T0 + timedelta(minutes=1),
    }
    stage_case_admission_evidence(**staging_arguments)
    transition_paths = case_admission_transition_paths(
        restricted_root=restricted,
        transition_directory=transition_directory,
    )
    transition_paths.h0_ledger.write_bytes(
        transition_paths.h0_ledger.read_bytes() + b"tampered"
    )

    with pytest.raises(CaseStudyTransitionError, match="archived H0 ledger hash changed"):
        stage_case_admission_evidence(**staging_arguments)


def test_interrupted_bootstrap_rejects_unexpected_post_h1_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, admission, gate_paths = _admission_and_gates(
        restricted
    )
    semantic = CaseStudySemanticAdmissionBundle.model_validate_json(
        semantic_path.read_bytes()
    )
    plan_path, plan = _patch_execution_plan(
        monkeypatch,
        restricted_root=restricted,
        admission=admission,
        semantic_bundle=semantic,
    )
    ledger_path = restricted / "study.sqlite"
    with Ledger(ledger_path) as ledger:
        ledger.record_gpu_event(
            event_id="artificial-predecessor",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=1.0,
            started_at=T0 - timedelta(seconds=1),
            ended_at=T0,
            succeeded=True,
        )
    blob_root = _shared_blobs(tmp_path)
    transition_directory = restricted / "staged" / "transition"
    monkeypatch.setattr(
        factory_module,
        "replay_case_study_semantic_admission",
        lambda **_: None,
    )
    evidence_bundle, evidence_reference, transition_receipt = (
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=plan_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=blob_root,
            transition_directory=transition_directory,
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )
    )
    staged_transition = verify_case_admission_staging_transition(
        restricted_root=restricted,
        ledger_path=ledger_path,
        cas_root=blob_root,
        transition_directory=transition_directory,
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission.content_hash,
        semantic_bundle_hash=semantic.content_hash,
        evidence_bundle_hash=evidence_bundle.content_hash,
        evidence_bundle_artifact_hash=evidence_reference.artifact_hash,
        evidence_bundle_reference_hash=evidence_reference.content_hash,
    )
    source_tree_sha256 = "c" * 64
    construction = SimpleNamespace(
        content_hash="d" * 64,
        source_file_sha256="e" * 64,
    )
    runtime_root = restricted / "runtime"
    runtime_root.mkdir()
    bootstrap_intent = factory_module.CaseAdmissionBootstrapIntent(
        execution_plan_hash=plan.content_hash,
        admission_attestation_hash=admission.content_hash,
        evidence_bundle_hash=evidence_bundle.content_hash,
        evidence_bundle_reference_hash=evidence_reference.content_hash,
        construction_configuration_hash=construction.content_hash,
        construction_configuration_file_sha256=construction.source_file_sha256,
        source_revision="fixture-revision",
        source_tree_sha256=source_tree_sha256,
        staging_transition_receipt_hash=transition_receipt.content_hash,
        predecessor_ledger_sha256=transition_receipt.h1_ledger.file_sha256,
        prior_gpu_event_inventory_hash=transition_receipt.gpu_inventory_hash,
        allocated_gpu_seconds_before_case=(
            transition_receipt.allocated_gpu_microseconds / 1_000_000
        ),
        admitted_at=T0 + timedelta(minutes=2),
    )
    (runtime_root / "case-admission-bootstrap-intent.json").write_text(
        canonical_json(bootstrap_intent) + "\n",
        encoding="utf-8",
    )
    source_manifest = SimpleNamespace(
        tree_sha256=source_tree_sha256,
        to_dict=lambda: {
            "revision": "fixture-revision",
            "tree_sha256": source_tree_sha256,
        },
    )
    monkeypatch.setattr(
        factory_module,
        "build_source_manifest",
        lambda *_args, **_kwargs: source_manifest,
    )
    with Ledger(ledger_path) as ledger:
        ArtifactStore(BlobStore(blob_root), ledger).put_bytes(
            b'{"unexpected":"post-h1"}',
            media_type="application/json",
            release_class=ReleaseClass.RESTRICTED,
            created_at=T0 + timedelta(minutes=2),
        )

    with pytest.raises(
        CaseStudyTransitionError,
        match="artifact rows are not the declared post-H1 prefix",
    ):
        factory_module._verify_interrupted_bootstrap_prefix(
            repository=tmp_path,
            restricted_root=restricted,
            runtime_root=runtime_root,
            transition_directory=transition_directory,
            ledger_path=ledger_path,
            artifact_root=blob_root,
            plan=plan,
            admission_attestation=admission,
            evidence_bundle=evidence_bundle,
            evidence_bundle_reference=evidence_reference,
            construction=construction,
            source_revision="fixture-revision",
            source_tree_sha256=source_tree_sha256,
            staging_transition=staged_transition,
        )


def test_stage_case_admission_evidence_rejects_tampered_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, admission, gate_paths = _admission_and_gates(restricted)
    semantic = CaseStudySemanticAdmissionBundle.model_validate_json(
        semantic_path.read_bytes()
    )
    plan_path, _plan = _patch_execution_plan(
        monkeypatch,
        restricted_root=restricted,
        admission=admission,
        semantic_bundle=semantic,
    )
    gate_paths["gold_firewall_audit_hash"].write_text(
        '{"gate":"different","passed":true}\n',
        encoding="utf-8",
    )
    ledger_path = restricted / "study.sqlite"
    ledger = Ledger(ledger_path)
    ledger.record_gpu_event(
        event_id="artificial-predecessor",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1.0,
        started_at=T0 - timedelta(seconds=1),
        ended_at=T0,
        succeeded=True,
        details={"artificial_fixture": True},
    )
    ledger.close()
    monkeypatch.setattr(factory_module, "replay_case_study_semantic_admission", lambda **_: None)

    with pytest.raises(CaseStudyFactoryError, match="differs from its attestation"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=plan_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=_shared_blobs(tmp_path),
            transition_directory=restricted / "staged" / "transition",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )


def test_semantic_bundle_rejects_cross_lineage_drift(tmp_path: Path) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    _admission_path, _semantic_path, _admission, gate_paths = _admission_and_gates(restricted)
    review = CaseBlindedErrorReviewGate.model_validate_json(
        gate_paths["blinded_error_review_hash"].read_bytes()
    ).model_copy(update={"output_receipt_inventory_hash": "b" * 64, "content_hash": ""})
    gate_paths["blinded_error_review_hash"].write_text(
        canonical_json(review) + "\n", encoding="utf-8"
    )
    with pytest.raises(CaseStudyFactoryError, match="exact native replay artifact inventory"):
        compile_case_study_semantic_admission_bundle(
            restricted_root=restricted,
            evidence_root=restricted,
            gate_paths=gate_paths,
            native_artifact_paths={},
            ledger_path=restricted / "missing.sqlite",
            blob_root=_shared_blobs(tmp_path),
            output_path=restricted / "drifted-semantic-gates.json",
            bundle_id="drifted-semantic-gates",
            frozen_at=T0,
        )


def test_stage_case_admission_evidence_does_not_create_a_fresh_ledger(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, _admission, gate_paths = _admission_and_gates(restricted)
    ledger_path = restricted / "missing.sqlite"

    with pytest.raises(CaseStudyFactoryError, match="existing study ledger"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=admission_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=ledger_path,
            artifact_root=_shared_blobs(tmp_path),
            transition_directory=restricted / "staged" / "transition",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )

    assert not ledger_path.exists()


def test_stage_requires_restricted_ledger_and_shared_cas_realpaths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    admission_path, semantic_path, _admission, gate_paths = _admission_and_gates(restricted)
    outside_ledger = tmp_path / "outside.sqlite"
    ledger = Ledger(outside_ledger)
    ledger.record_gpu_event(
        event_id="artificial-predecessor",
        event_kind=GpuEventKind.INFERENCE,
        allocated_seconds=1.0,
        started_at=T0 - timedelta(seconds=1),
        ended_at=T0,
        succeeded=True,
    )
    ledger.close()
    monkeypatch.setattr(factory_module, "replay_case_study_semantic_admission", lambda **_: None)
    with pytest.raises(CaseStudyFactoryError, match="inside the restricted root"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=admission_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=outside_ledger,
            artifact_root=_shared_blobs(tmp_path),
            transition_directory=restricted / "staged" / "transition",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )

    inside_ledger = restricted / "study.sqlite"
    inside_ledger.write_bytes(outside_ledger.read_bytes())
    outside_blobs = tmp_path / "outside-blobs"
    outside_blobs.mkdir()
    blob_parent = tmp_path / "artifacts" / "blobs"
    blob_parent.mkdir(parents=True, exist_ok=True)
    (blob_parent / "linked").symlink_to(outside_blobs, target_is_directory=True)
    with pytest.raises(CaseStudyFactoryError, match="symbolic link"):
        stage_case_admission_evidence(
            restricted_root=restricted,
            semantic_evidence_root=tmp_path,
            execution_plan_path=admission_path,
            admission_attestation_path=admission_path,
            semantic_gate_bundle_path=semantic_path,
            gate_paths=gate_paths,
            ledger_path=inside_ledger,
            artifact_root=blob_parent / "linked",
            transition_directory=restricted / "staged" / "transition",
            bundle_output_path=restricted / "staged" / "bundle.json",
            reference_output_path=restricted / "staged" / "reference.json",
            staged_at=T0 + timedelta(minutes=1),
        )


def test_full_production_factory_rejects_external_ledger_before_model_construction(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    placeholder = restricted / "placeholder.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    outside_ledger = tmp_path / "outside.sqlite"
    outside_ledger.write_bytes(b"not opened")

    with pytest.raises(CaseStudyFactoryError, match="inside the restricted root"):
        create_frozen_production_case_study_bundle(
            repository=Path(__file__).resolve().parents[2],
            restricted_root=restricted,
            plan_path=placeholder,
            index_path=placeholder,
            index_manifest_path=placeholder,
            preregistration_path=placeholder,
            input_attestation_path=placeholder,
            admission_attestation_path=placeholder,
            semantic_gate_bundle_path=placeholder,
            selected_model_freeze_path=placeholder,
            admission_evidence_bundle_path=placeholder,
            admission_evidence_bundle_reference_path=placeholder,
            construction_path=placeholder,
            ledger_path=outside_ledger,
            artifact_root=restricted / "blobs",
            staging_transition_directory=restricted / "staging-transition",
            runtime_root=restricted / "runtime",
            quota_root=tmp_path,
            snapshot_path=tmp_path,
            shared_cache=tmp_path,
            verified_model_manifest_path=placeholder,
            source_association_path=placeholder,
            expected_predecessor_ledger_sha256="a" * 64,
            source_revision="artificial-source",
        )


def test_full_production_preflight_is_zero_write_and_rejects_held_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    restricted = repository / "restricted"
    artifact_root = repository / "artifacts" / "blobs" / "study"
    runtime_root = restricted / "runtime"
    transition_directory = restricted / "transition"
    shared_cache = repository / "cache"
    snapshot = shared_cache / "models--fixture" / "snapshots" / ("f" * 40)
    construction_path = repository / "configs" / "study" / "development_construction.json"
    for directory in (
        restricted,
        artifact_root,
        runtime_root,
        transition_directory,
        snapshot,
        construction_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    runtime_root.chmod(0o700)
    construction_path.write_text('{"fixture":true}\n', encoding="utf-8")
    placeholder = restricted / "input.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    ledger_path = restricted / "study.sqlite3"
    ledger_path.write_bytes(b"read-only-ledger-fixture")

    references = tuple(
        factory_module.CaseAdmissionEvidenceReference(
            name=name,
            logical_content_hash=str(index + 1) * 64,
            artifact_hash=hashlib.sha256(name.encode("utf-8")).hexdigest(),
        )
        for index, name in enumerate(GATE_NAMES)
    )
    evidence = factory_module.CaseAdmissionEvidenceBundle(
        bundle_id="preflight-fixture",
        admission_attestation_hash="9" * 64,
        evidence=references,
        frozen_at=T0,
    )
    evidence_reference = factory_module.CaseArtifactReference(
        logical_content_hash=evidence.content_hash,
        artifact_hash="e" * 64,
        object_kind="case_admission_evidence_bundle",
    )
    upper_hash = "8" * 64
    plan = SimpleNamespace(
        content_hash=PLAN_HASH,
        input_attestation_hash="1" * 64,
        admission_attestation_hash="9" * 64,
        semantic_gate_bundle_hash="2" * 64,
        runtime_policy_hash="3" * 64,
        model_runtime=SimpleNamespace(
            selected_model_freeze_hash="4" * 64,
            upper_ontology_hash=upper_hash,
        ),
        gpu_call_slots=(
            SimpleNamespace(call_id="call-1", watchdog_seconds=10),
            SimpleNamespace(call_id="call-2", watchdog_seconds=20),
        ),
    )
    loaded = SimpleNamespace(attestation=SimpleNamespace(content_hash="1" * 64))
    admission = SimpleNamespace(content_hash="9" * 64, upper_ontology_hash=upper_hash)
    semantic = SimpleNamespace(content_hash="2" * 64)
    selected_freeze = SimpleNamespace(manifest_sha256="4" * 64)
    policy = SimpleNamespace(
        content_hash="3" * 64,
        production_adapter_factory=CASE_PRODUCTION_FACTORY,
        service_start_watchdog_seconds=300,
    )
    construction = SimpleNamespace(
        source_file_sha256=hashlib.sha256(construction_path.read_bytes()).hexdigest(),
        upper_ontology=SimpleNamespace(content_hash=upper_hash),
    )
    transition = SimpleNamespace(
        receipt=SimpleNamespace(content_hash="5" * 64),
        current_ledger_sha256="6" * 64,
        current_allocated_gpu_microseconds=12_000_000,
    )

    monkeypatch.setattr(factory_module, "load_case_study_execution_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        factory_module,
        "load_attested_restricted_case_study",
        lambda **_k: loaded,
    )
    monkeypatch.setattr(
        factory_module,
        "load_case_study_admission_attestation",
        lambda *_a, **_k: admission,
    )
    monkeypatch.setattr(
        factory_module,
        "load_case_study_semantic_admission_bundle",
        lambda *_a, **_k: semantic,
    )
    monkeypatch.setattr(factory_module, "validate_case_study_semantic_admission", lambda *_a: None)
    monkeypatch.setattr(
        factory_module,
        "_load_record",
        lambda _path, model, **_k: (
            evidence
            if model is factory_module.CaseAdmissionEvidenceBundle
            else evidence_reference
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "load_attested_selected_model_freeze",
        lambda **_k: selected_freeze,
    )
    monkeypatch.setattr(factory_module.CaseStudyRuntimePolicy, "load", lambda *_a: policy)
    monkeypatch.setattr(
        factory_module,
        "validate_source_association",
        lambda *_a, **_k: {
            "revision_label": "fixture-source",
            "local_tree_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        factory_module.DevelopmentConstructionConfiguration,
        "load",
        lambda *_a: construction,
    )
    monkeypatch.setattr(
        factory_module,
        "_replay_execution_semantic_admission",
        lambda **_k: transition,
    )
    monkeypatch.setattr(
        factory_module,
        "_verify_interrupted_bootstrap_prefix",
        lambda **_k: None,
    )

    class _ReadOnlyLedger:
        def gpu_summary(self) -> SimpleNamespace:
            return SimpleNamespace(total_allocated_seconds=12.0)

        def unresolved_gpu_allocations(self) -> tuple[object, ...]:
            return ()

        def unresolved_gpu_service_journals(self) -> tuple[object, ...]:
            return ()

    class _ReadOnlyArtifacts:
        ledger = _ReadOnlyLedger()

        def __enter__(self) -> _ReadOnlyArtifacts:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    read_only_artifacts = _ReadOnlyArtifacts()
    monkeypatch.setattr(
        factory_module.ReadOnlyArtifactStore,
        "from_paths",
        lambda **_k: read_only_artifacts,
    )
    monkeypatch.setattr(factory_module, "validate_case_admission_evidence", lambda **_k: None)
    monkeypatch.setattr(
        factory_module,
        "_read_reference",
        lambda *_a, **_k: canonical_json(evidence).encode("utf-8"),
    )
    limits = SimpleNamespace(
        scheduled_gpu_seconds=32_400,
        hard_gpu_seconds=36_000,
        maximum_project_allocation_bytes=30 * 1024**3,
        maximum_project_occupied_bytes=25 * 1024**3,
        minimum_storage_headroom_bytes=5 * 1024**3,
    )
    monkeypatch.setattr(factory_module.ResourceLimits, "load", lambda *_a: limits)
    reservation = SimpleNamespace(preflight_arguments=lambda: {})
    allocation_plan = SimpleNamespace(reservation_for=lambda _phase: reservation)
    monkeypatch.setattr(
        factory_module.StorageAllocationPlan,
        "load",
        lambda *_a: allocation_plan,
    )
    tokenizer_closed: list[bool] = []
    tokenizer = SimpleNamespace(close=lambda: tokenizer_closed.append(True))
    launcher = SimpleNamespace(configuration_hash="a" * 64)
    monkeypatch.setattr(
        factory_module,
        "_verify_runtime_model",
        lambda **_k: (launcher, tokenizer, SimpleNamespace(), "b" * 64),
    )

    def tree_snapshot() -> tuple[tuple[str, int, str], ...]:
        return tuple(
            (
                path.relative_to(repository).as_posix(),
                path.stat().st_mode,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(repository.rglob("*"))
            if path.is_file()
        )

    before = tree_snapshot()
    result = preflight_frozen_production_case_study_bundle(
        repository=repository,
        restricted_root=restricted,
        plan_path=placeholder,
        index_path=placeholder,
        index_manifest_path=placeholder,
        preregistration_path=placeholder,
        input_attestation_path=placeholder,
        admission_attestation_path=placeholder,
        semantic_gate_bundle_path=placeholder,
        selected_model_freeze_path=placeholder,
        admission_evidence_bundle_path=placeholder,
        admission_evidence_bundle_reference_path=placeholder,
        construction_path=construction_path,
        ledger_path=ledger_path,
        artifact_root=artifact_root,
        staging_transition_directory=transition_directory,
        runtime_root=runtime_root,
        quota_root=repository,
        snapshot_path=snapshot,
        shared_cache=shared_cache,
        verified_model_manifest_path=placeholder,
        source_association_path=placeholder,
        expected_predecessor_ledger_sha256="c" * 64,
        source_revision="fixture-source",
    )
    assert result.execution_ready
    assert result.writes_performed is False
    assert result.actual_allocated_gpu_seconds == 12.0
    assert result.remaining_required_gpu_seconds == 570.0
    assert tokenizer_closed == [True]
    assert tree_snapshot() == before
    assert not factory_module._runtime_lock_path(
        runtime_root,
        ledger_path=ledger_path,
    ).exists()
    assert not (shared_cache / factory_module.SERVICE_LOCK_FILENAME).exists()

    lock = factory_module._acquire_runtime_lock(runtime_root, ledger_path=ledger_path)
    try:
        with pytest.raises(CaseStudyFactoryError, match="held by another controller"):
            preflight_frozen_production_case_study_bundle(
                repository=repository,
                restricted_root=restricted,
                plan_path=placeholder,
                index_path=placeholder,
                index_manifest_path=placeholder,
                preregistration_path=placeholder,
                input_attestation_path=placeholder,
                admission_attestation_path=placeholder,
                semantic_gate_bundle_path=placeholder,
                selected_model_freeze_path=placeholder,
                admission_evidence_bundle_path=placeholder,
                admission_evidence_bundle_reference_path=placeholder,
                construction_path=construction_path,
                ledger_path=ledger_path,
                artifact_root=artifact_root,
                staging_transition_directory=transition_directory,
                runtime_root=runtime_root,
                quota_root=repository,
                snapshot_path=snapshot,
                shared_cache=shared_cache,
                verified_model_manifest_path=placeholder,
                source_association_path=placeholder,
                expected_predecessor_ledger_sha256="c" * 64,
                source_revision="fixture-source",
            )
    finally:
        factory_module._release_runtime_lock(lock)


def test_production_factory_releases_runtime_lock_after_preledger_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    restricted = repository / "artifacts" / "restricted"
    artifact_root = repository / "artifacts" / "blobs" / "study"
    construction_path = repository / "configs" / "study" / "development_construction.json"
    runtime_root = restricted / "runtime"
    transition_directory = restricted / "transition"
    shared_cache = repository / "cache"
    snapshot = shared_cache / "snapshot"
    for directory in (
        restricted,
        artifact_root,
        construction_path.parent,
        runtime_root,
        transition_directory,
        snapshot,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    construction_path.write_text("{}\n", encoding="utf-8")
    placeholder = restricted / "placeholder.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    ledger_path = restricted / "study.sqlite3"
    ledger_path.write_bytes(b"preledger-fixture")

    admission = SimpleNamespace(content_hash="1" * 64)
    semantic = SimpleNamespace(content_hash="2" * 64)
    freeze = SimpleNamespace(manifest_sha256="3" * 64)
    loaded = SimpleNamespace(attestation=SimpleNamespace(content_hash="4" * 64))
    policy = SimpleNamespace(
        content_hash="5" * 64,
        production_adapter_factory=CASE_PRODUCTION_FACTORY,
        service_start_watchdog_seconds=300,
    )
    plan = SimpleNamespace(
        input_attestation_hash=loaded.attestation.content_hash,
        admission_attestation_hash=admission.content_hash,
        semantic_gate_bundle_hash=semantic.content_hash,
        model_runtime=SimpleNamespace(selected_model_freeze_hash=freeze.manifest_sha256),
        runtime_policy_hash=policy.content_hash,
    )
    evidence = SimpleNamespace(content_hash="6" * 64)
    evidence_reference = SimpleNamespace(content_hash="7" * 64)
    construction = SimpleNamespace(content_hash="8" * 64)

    monkeypatch.setattr(factory_module, "load_case_study_execution_plan", lambda *_a, **_k: plan)
    monkeypatch.setattr(
        factory_module,
        "load_attested_restricted_case_study",
        lambda **_k: loaded,
    )
    monkeypatch.setattr(
        factory_module,
        "load_case_study_admission_attestation",
        lambda *_a, **_k: admission,
    )
    monkeypatch.setattr(
        factory_module,
        "load_case_study_semantic_admission_bundle",
        lambda *_a, **_k: semantic,
    )
    monkeypatch.setattr(factory_module, "validate_case_study_semantic_admission", lambda *_a: None)
    monkeypatch.setattr(
        factory_module,
        "_load_record",
        lambda _path, model, **_k: (
            evidence
            if model.__name__ == "CaseAdmissionEvidenceBundle"
            else evidence_reference
        ),
    )
    monkeypatch.setattr(
        factory_module,
        "load_attested_selected_model_freeze",
        lambda **_k: freeze,
    )
    monkeypatch.setattr(factory_module.CaseStudyRuntimePolicy, "load", lambda *_a: policy)
    monkeypatch.setattr(
        factory_module,
        "validate_source_association",
        lambda *_a, **_k: {
            "revision_label": "fixture-revision",
            "local_tree_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(
        factory_module.DevelopmentConstructionConfiguration,
        "load",
        lambda *_a: construction,
    )
    monkeypatch.setattr(
        factory_module,
        "_replay_execution_semantic_admission",
        lambda **_k: (_ for _ in ()).throw(CaseStudyFactoryError("forced post-lock failure")),
    )

    retained_errors: list[BaseException] = []
    try:
        create_frozen_production_case_study_bundle(
            repository=repository,
            restricted_root=restricted,
            plan_path=placeholder,
            index_path=placeholder,
            index_manifest_path=placeholder,
            preregistration_path=placeholder,
            input_attestation_path=placeholder,
            admission_attestation_path=placeholder,
            semantic_gate_bundle_path=placeholder,
            selected_model_freeze_path=placeholder,
            admission_evidence_bundle_path=placeholder,
            admission_evidence_bundle_reference_path=placeholder,
            construction_path=construction_path,
            ledger_path=ledger_path,
            artifact_root=artifact_root,
            staging_transition_directory=transition_directory,
            runtime_root=runtime_root,
            quota_root=repository,
            snapshot_path=snapshot,
            shared_cache=shared_cache,
            verified_model_manifest_path=placeholder,
            source_association_path=placeholder,
            expected_predecessor_ledger_sha256="a" * 64,
            source_revision="fixture-revision",
        )
    except CaseStudyFactoryError as error:
        retained_errors.append(error)
    assert retained_errors and str(retained_errors[0]) == "forced post-lock failure"

    # Retaining the traceback used to retain the locked stream as a frame local.
    # A same-process retry must nevertheless acquire the exact lock immediately.
    retried_lock = factory_module._acquire_runtime_lock(
        runtime_root,
        ledger_path=ledger_path,
    )
    retried_lock.close()


def test_controller_lock_excludes_another_runtime_root_for_the_same_ledger(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    first_runtime = restricted / "runtime-a"
    second_runtime = restricted / "runtime-b"
    first_runtime.mkdir(parents=True)
    second_runtime.mkdir()
    ledger_path = restricted / "study.sqlite3"
    ledger_path.write_bytes(b"stable lock identity")

    first = factory_module._acquire_runtime_lock(
        first_runtime,
        ledger_path=ledger_path,
    )
    try:
        with pytest.raises(CaseStudyFactoryError, match="another case-study controller"):
            factory_module._acquire_runtime_lock(
                second_runtime,
                ledger_path=ledger_path,
            )
    finally:
        factory_module._release_runtime_lock(first)

    successor = factory_module._acquire_runtime_lock(
        second_runtime,
        ledger_path=ledger_path,
    )
    factory_module._release_runtime_lock(successor)


def test_runtime_lock_release_closes_descriptor_when_explicit_unlock_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = SimpleNamespace(
        closed=False,
        fileno=lambda: 19,
    )

    def close() -> None:
        stream.closed = True

    stream.close = close
    monkeypatch.setattr(
        factory_module.fcntl,
        "flock",
        lambda *_args: (_ for _ in ()).throw(OSError("forced unlock failure")),
    )

    with pytest.raises(OSError, match="forced unlock failure"):
        factory_module._release_runtime_lock(stream)
    assert stream.closed


def test_bundle_close_releases_runtime_lock_when_ledger_close_fails(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime_lock = factory_module._acquire_runtime_lock(runtime_root)

    class _FailingLedger:
        def close(self) -> None:
            raise RuntimeError("forced ledger close failure")

    bundle = factory_module.CaseStudyProductionBundle(
        controller=SimpleNamespace(),
        admission_reference=SimpleNamespace(),
        artifacts=SimpleNamespace(),
        ledger=_FailingLedger(),
        service=SimpleNamespace(state=factory_module.ServiceState.STOPPED),
        gpu=SimpleNamespace(),
        runtime_root=runtime_root,
        storage_report=SimpleNamespace(),
        _runtime_lock=runtime_lock,
    )
    with pytest.raises(RuntimeError, match="forced ledger close failure"):
        bundle.close()
    assert bundle._closed

    retried_lock = factory_module._acquire_runtime_lock(runtime_root)
    retried_lock.close()
