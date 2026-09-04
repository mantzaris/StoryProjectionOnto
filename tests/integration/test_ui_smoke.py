from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import pytest

from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    EvidenceRecord,
    ReleaseClass,
)
from story_projection_onto.ui import (
    FeedbackReplayExpectation,
    LocalUiRepository,
    RevisionCallKind,
    RevisionCallRecord,
    RevisionExecutionResult,
    build_visualization_bundle,
    compare_visualizations,
    create_app,
    resolve_feedback_for_condition,
    verify_feedback_replay,
)
from tests.unit.test_ui import NOW, context, packet, projection


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_local_api_projection_evidence_filter_and_honest_revision_gate() -> None:
    fastapi = pytest.importorskip("fastapi")
    assert fastapi is not None
    testclient = pytest.importorskip("fastapi.testclient")
    query_context = context()
    evidence_packet = packet()
    ontology_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        query_context,
        evidence_packet,
    )
    bundle = build_visualization_bundle(
        ontology_projection,
        query_context,
        evidence_packet,
        include_public_evidence_text=True,
    )
    repository = LocalUiRepository((bundle,))
    ui_directory = Path(__file__).resolve().parents[2] / "ui"
    app = create_app(repository, static_directory=ui_directory)
    client = testclient.TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["usability_claims"] is False
    assert health.json()["revision_actions"] == [
        "REFINE_CONTEXT",
        "REQUEST_MERGE_SPLIT",
    ]
    assert health.json()["revision_seed_decimal"] is None
    assert health.json()["cytoscape"]["verified"] is True
    assert (
        health.json()["cytoscape"]["sha256"]
        == "1bb5340e549511e111b31e5684872c949ad33d40ea5dba0ad8e7d90c62c7b3b9"
    )

    listing = client.get("/api/projections")
    assert listing.status_code == 200
    assert listing.json()[0]["projection_id"] == ontology_projection.projection_id
    loaded = client.get(f"/api/projections/{ontology_projection.projection_id}")
    assert loaded.status_code == 200
    assert loaded.json()["projection_hash"] == ontology_projection.content_hash

    evidence = client.get("/api/evidence/ev-a")
    assert evidence.status_code == 200
    assert evidence.json()["public_text"] == "Ari carried the key."

    filtered = client.post(
        f"/api/projections/{ontology_projection.projection_id}/filter",
        json={
            "temporal_filter": {
                "story_scope": {"kind": "point", "point": 1},
                "spoiler_horizon": None,
                "epistemic_holder_id": None,
            }
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["state"]["semantic_hash"] == ontology_projection.content_hash

    revision = client.post(
        "/api/revisions",
        json={
            "before_projection_id": ontology_projection.projection_id,
            "action": "REFINE_CONTEXT",
            "anchors": [
                {
                    "evidence_ids": ["ev-a"],
                    "mention_candidate_ids": ["m-a"],
                    "requested_semantic_signature": "focus on first-signal responsibility",
                }
            ],
            "rationale": "Show the causal lens at the first signal.",
            "sequence": 1,
            "seed": 0,
            "lens": "first-signal responsibility",
            "grouped_mention_candidate_ids": [],
        },
    )
    assert revision.status_code == 503
    assert "no regeneration was claimed" in revision.json()["detail"]


def test_static_page_has_required_rich_fields_and_no_cytoscape_stub() -> None:
    root = Path(__file__).resolve().parents[2] / "ui"
    html = (root / "index.html").read_text(encoding="utf-8")
    app_js = (root / "app.js").read_text(encoding="utf-8")
    loader = (root / "cytoscape-loader.js").read_text(encoding="utf-8")
    assert "REFINE_CONTEXT" in html
    assert "REQUEST_MERGE_SPLIT" in html
    assert "Story point" in html
    assert "Max revelation" in html
    assert "Why it matters" in app_js
    assert "epistemic_attitude" in app_js
    assert "evidence-link" in app_js
    assert "function cytoscape" not in loader


def test_injected_fixture_runner_exercises_typed_c2_rebuild_diff_and_replay() -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    query_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
    )
    after_projection = projection(
        ConditionName.C2_LLM_QUERY,
        query_context,
        evidence_packet,
        merged=True,
        decision_offset_seconds=65,
    )
    before_bundle = build_visualization_bundle(
        before_projection,
        query_context,
        evidence_packet,
    )
    after_bundle = build_visualization_bundle(
        after_projection,
        query_context,
        evidence_packet,
    )
    completed_at = NOW + timedelta(minutes=2)

    def fixture_runner(instruction, supplied_before_bundle, seed):
        assert supplied_before_bundle.content_hash == before_bundle.content_hash
        resolution = resolve_feedback_for_condition(
            instruction=instruction,
            packet=evidence_packet,
            receiving_condition=ConditionName.C2_LLM_QUERY,
            before_projection=before_projection,
            after_projection=after_projection,
            resolver_hash=digest("fixture-resolver-v1"),
            seed=seed,
            resolved_at=completed_at,
        )
        diff = compare_visualizations(before_bundle, after_bundle)
        expectation = FeedbackReplayExpectation(
            instruction_hash=instruction.content_hash,
            resolution_hash=resolution.content_hash,
            before_bundle_hash=before_bundle.content_hash,
            after_bundle_hash=after_bundle.content_hash,
            diff_hash=diff.content_hash,
        )
        replay = verify_feedback_replay(
            expectation=expectation,
            instruction=instruction,
            resolution=resolution,
            before_bundle=before_bundle,
            after_bundle=after_bundle,
            diff=diff,
            checked_at=completed_at,
        )
        return RevisionExecutionResult(
            instruction=instruction,
            resolution=resolution,
            before_bundle_hash=before_bundle.content_hash,
            call=RevisionCallRecord(
                call_artifact_hash=digest("fixture-ui-model-call"),
                call_id="fixture-ui-model-call",
                condition=ConditionName.C2_LLM_QUERY,
                kind=RevisionCallKind.GPU_RECONSTRUCT,
                instruction_hash=instruction.content_hash,
                before_projection_hash=before_projection.content_hash,
                after_projection_hash=after_projection.content_hash,
                seed=seed,
                allocated_gpu_seconds=0.005,
                started_at=NOW + timedelta(minutes=1, milliseconds=1),
                completed_at=NOW + timedelta(minutes=1, milliseconds=6),
            ),
            after_bundle=after_bundle,
            diff=diff,
            latency_seconds=0.01,
            replay=replay,
        )

    root = Path(__file__).resolve().parents[2] / "ui"
    app = create_app(
        LocalUiRepository((before_bundle,)),
        revision_runner=fixture_runner,
        revision_seed=0,
        static_directory=root,
        clock=lambda: NOW + timedelta(minutes=1),
    )
    client = testclient.TestClient(app)
    response = client.post(
        "/api/revisions",
        json={
            "before_projection_id": before_projection.projection_id,
            "action": "REQUEST_MERGE_SPLIT",
            "anchors": [
                {
                    "evidence_ids": ["ev-a", "ev-b"],
                    "mention_candidate_ids": ["m-a", "m-b"],
                    "requested_semantic_signature": "merge m-a and m-b",
                }
            ],
            "rationale": "Test only: exercise the injected parser-boundary fixture.",
            "sequence": 1,
            "seed": 0,
            "merge_split_operation": "merge",
            "grouped_mention_candidate_ids": [["m-a"], ["m-b"]],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resolution"]["status"] == "resolved"
    assert body["replay"]["replay_hash_success"] is True
    loaded_after = client.get(f"/api/projections/{after_projection.projection_id}")
    assert loaded_after.status_code == 200
    diff = client.get(
        f"/api/diffs/{before_projection.projection_id}/{after_projection.projection_id}"
    )
    assert diff.status_code == 200
    assert any(item["kind"] == "merged" for item in diff.json()["changes"])

    health = client.get("/api/health")
    assert health.json()["revision_seed_decimal"] == "0"


def test_local_api_returns_explicit_c0_capability_limit_without_output() -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    query_context = context()
    evidence_packet = packet()
    before_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        query_context,
        evidence_packet,
    )
    before_bundle = build_visualization_bundle(
        before_projection,
        query_context,
        evidence_packet,
    )

    def limited_runner(instruction, supplied_before_bundle, seed):
        resolution = resolve_feedback_for_condition(
            instruction=instruction,
            packet=evidence_packet,
            receiving_condition=ConditionName.C0_CLASSICAL_PRE,
            before_projection=before_projection,
            after_projection=None,
            resolver_hash=digest("fixture-resolver-v1"),
            seed=seed,
            resolved_at=NOW + timedelta(minutes=2),
        )
        return RevisionExecutionResult(
            instruction=instruction,
            resolution=resolution,
            before_bundle_hash=supplied_before_bundle.content_hash,
            latency_seconds=0.001,
        )

    root = Path(__file__).resolve().parents[2] / "ui"
    client = testclient.TestClient(
        create_app(
            LocalUiRepository((before_bundle,)),
            revision_runner=limited_runner,
            revision_seed=0,
            static_directory=root,
            clock=lambda: NOW + timedelta(minutes=1),
        )
    )
    response = client.post(
        "/api/revisions",
        json={
            "before_projection_id": before_projection.projection_id,
            "action": "REQUEST_MERGE_SPLIT",
            "anchors": [
                {
                    "evidence_ids": ["ev-a", "ev-b"],
                    "mention_candidate_ids": ["m-a", "m-b"],
                    "requested_semantic_signature": "merge m-a and m-b",
                }
            ],
            "rationale": "Exercise the explicit sealed-condition capability result.",
            "sequence": 1,
            "seed": 0,
            "merge_split_operation": "merge",
            "grouped_mention_candidate_ids": [["m-a"], ["m-b"]],
        },
    )
    assert response.status_code == 200
    assert response.json()["resolution"]["status"] == "capability_limited"
    assert response.json()["after_bundle"] is None
    assert response.json()["call"] is None


def test_restricted_bundle_requires_explicit_local_authorization() -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    query_context = context()
    public_packet = packet()
    restricted_records = tuple(
        EvidenceRecord(
            **{
                **item.model_dump(mode="python", exclude={"content_hash"}),
                "release_class": ReleaseClass.RESTRICTED,
            }
        )
        for item in public_packet.evidence
    )
    restricted_packet = EvidencePacket(
        **{
            **public_packet.model_dump(
                mode="python",
                exclude={"content_hash", "evidence"},
            ),
            "evidence": restricted_records,
            "release_class": ReleaseClass.RESTRICTED,
        }
    )
    ontology_projection = projection(
        ConditionName.C0_CLASSICAL_PRE,
        query_context,
        restricted_packet,
    )
    bundle = build_visualization_bundle(
        ontology_projection,
        query_context,
        restricted_packet,
    )
    assert bundle.release_class is ReleaseClass.RESTRICTED
    root = Path(__file__).resolve().parents[2] / "ui"

    denied = testclient.TestClient(create_app(LocalUiRepository((bundle,)), static_directory=root))
    assert denied.get("/api/projections").json() == []
    assert denied.get(f"/api/projections/{ontology_projection.projection_id}").status_code == 403
    assert denied.get("/api/evidence/ev-a").status_code == 403

    authorized = testclient.TestClient(
        create_app(
            LocalUiRepository((bundle,)),
            static_directory=root,
            allow_restricted_evidence_metadata=True,
        )
    )
    response = authorized.get(f"/api/projections/{ontology_projection.projection_id}")
    assert response.status_code == 200
    assert response.json()["evidence_metadata"][0]["public_text"] is None
