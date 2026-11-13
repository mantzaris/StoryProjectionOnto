from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from story_projection_onto.phase7_compiler import (
    C2IntentionToTreatScore,
    QualitativeCandidate,
    QualitativeDisplayRow,
    QualitativeExample,
)
from story_projection_onto.scorer_only.qualitative_materializer import (
    QualitativeMaterializationSource,
    QualitativePanelSource,
    materialize_qualitative_candidates,
    prepare_qualitative_materialization,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "configs/study/reporting.json"
HASH = "a" * 64


def _display(context: str, condition: str) -> QualitativeDisplayRow:
    return QualitativeDisplayRow(
        context_id=context,
        condition=condition,
        node_summary="Readable context-dependent entities.",
        assertion_summary="A grounded directed and qualified assertion.",
        temporal_sequence="The first event precedes the second event.",
        why_matters="The distinction changes the contextual answer.",
        opaque_evidence_ids=("evidence-opaque-1",),
        projection_hash=HASH,
    )


def _candidate(
    candidate_id: str,
    example: QualitativeExample,
    *,
    split: str,
    world: str,
    contexts: tuple[str, ...],
    query: str,
    stratum: str,
    window: int | None = None,
    rare: int = 0,
    support: bool = False,
    temporal: bool = False,
    holder: bool = False,
    scores: tuple[C2IntentionToTreatScore, ...] = (),
) -> QualitativeCandidate:
    conditions = (
        "c0_classical_pre",
        "c1_llm_pre",
        "c2_llm_query",
    )
    if split != "case_study":
        conditions = (*conditions, "a_fixed_select")
    panel_ids = tuple(
        f"panel-{candidate_id}-{context}-{condition}"
        for context in contexts
        for condition in conditions
    )
    composite_id = f"composite-{candidate_id}"
    return QualitativeCandidate(
        candidate_id=candidate_id,
        eligible_examples=(example,),
        evidence_split=split,
        world_id=world,
        context_ids=contexts,
        query_ordinal=query,
        stratum=stratum,
        window_ordinal=window,
        rare_pivotal_denominator=rare,
        complete_support_path=support,
        temporal_epistemic_eligible=temporal,
        holder_attributed=holder,
        c2_itt_scores=scores,
        reviewed_failure_id=(
            f"reviewed-failure-{candidate_id}"
            if example is QualitativeExample.COUNTEREXAMPLE
            else None
        ),
        display_rows=tuple(
            _display(context, condition)
            for context in contexts
            for condition in conditions
        ),
        composite_figure_artifact_id=composite_id,
        source_artifact_ids=(composite_id, *panel_ids),
        paraphrase_only=True,
        opaque_evidence_only=True,
        contains_verbatim_copyrighted_text=False,
    )


def _candidates() -> tuple[QualitativeCandidate, ...]:
    return (
        _candidate(
            "tutorial",
            QualitativeExample.DEVELOPMENT_TUTORIAL,
            split="development",
            world="syn-dev-01",
            contexts=("ctx-dev-a", "ctx-dev-b"),
            query="A+B",
            stratum="easy",
        ),
        _candidate(
            "rare",
            QualitativeExample.RARE_PIVOTAL,
            split="development",
            world="syn-dev-02",
            contexts=("ctx-rare",),
            query="A",
            stratum="medium",
            rare=1,
            support=True,
        ),
        _candidate(
            "temporal",
            QualitativeExample.TEMPORAL_EPISTEMIC,
            split="development",
            world="syn-dev-03",
            contexts=("ctx-temporal",),
            query="A",
            stratum="medium",
            temporal=True,
            holder=True,
        ),
        _candidate(
            "heldout",
            QualitativeExample.HELD_OUT_ILLUSTRATION,
            split="held_out",
            world="syn-test-03",
            contexts=("ctx_556b0577875afcad7419",),
            query="A",
            stratum="easy",
        ),
        _candidate(
            "counterexample",
            QualitativeExample.COUNTEREXAMPLE,
            split="held_out",
            world="syn-test-12",
            contexts=("ctx-hard",),
            query="B",
            stratum="hard",
            scores=(
                C2IntentionToTreatScore(
                    seed_block=1,
                    outcome="succeeded",
                    strict_qualified_assertion_f1=0.25,
                ),
                C2IntentionToTreatScore(
                    seed_block=2,
                    outcome="invalid",
                    strict_qualified_assertion_f1=0.0,
                ),
            ),
        ),
        _candidate(
            "narrative",
            QualitativeExample.NARRATIVE_ILLUSTRATION,
            split="case_study",
            world="window-01",
            contexts=("case-context-a", "case-context-b"),
            query="not_applicable",
            stratum="case_study",
            window=1,
        ),
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


def test_qualitative_materializer_applies_frozen_rules_and_builds_composites(
    tmp_path: Path,
) -> None:
    restricted = tmp_path / "restricted"
    source_directory = restricted / "source"
    panel_directory = source_directory / "panels"
    panel_directory.mkdir(parents=True)
    candidates = _candidates()
    panels: list[QualitativePanelSource] = []
    for candidate_index, candidate in enumerate(candidates):
        for row_index, row in enumerate(candidate.display_rows):
            artifact_id = f"panel-{candidate.candidate_id}-{row.context_id}-{row.condition}"
            relative = f"panels/{artifact_id}.png"
            payload = _png_bytes((30 + candidate_index * 20, 40 + row_index * 5, 90))
            (source_directory / relative).write_bytes(payload)
            panels.append(
                QualitativePanelSource(
                    panel_artifact_id=artifact_id,
                    candidate_id=candidate.candidate_id,
                    context_id=row.context_id,
                    condition=row.condition,
                    relative_path=relative,
                    file_sha256=hashlib.sha256(payload).hexdigest(),
                    width_pixels=64,
                    height_pixels=64,
                    fixed_anchor_layout_hash=hashlib.sha256(
                        f"{candidate.candidate_id}/{row.context_id}".encode()
                    ).hexdigest(),
                )
            )
    policy_hash = json.loads(POLICY.read_text())["policy_sha256"]
    source = QualitativeMaterializationSource(
        manifest_id="qualitative-materials-v1",
        frozen_reporting_policy_sha256=policy_hash,
        copyright_release_attestation_hash="b" * 64,
        candidates=candidates,
        panels=tuple(panels),
    )
    source_path = source_directory / "source.json"
    source_path.write_text(source.to_canonical_json() + "\n")

    candidate_set, composites, receipt, files = prepare_qualitative_materialization(
        restricted_root=restricted,
        source_manifest_path=source_path,
        reporting_policy_path=POLICY,
    )
    assert len(candidate_set.candidates) == 6
    assert len(composites.figures) == 6
    assert receipt.selected_candidate_ids == (
        "tutorial",
        "rare",
        "temporal",
        "heldout",
        "counterexample",
        "narrative",
    )
    assert len(receipt.selection_rule_hashes) == 6
    assert all(files[item.relative_path].startswith(b"\x89PNG") for item in composites.figures)

    output, state, materialized = materialize_qualitative_candidates(
        restricted_root=restricted,
        source_manifest_path=source_path,
        reporting_policy_path=POLICY,
        output_root=restricted / "materialized",
    )
    assert state == "created"
    assert materialized == receipt
    assert (output / "qualitative_candidate_set.json").is_file()
    replay, replay_state, _ = materialize_qualitative_candidates(
        restricted_root=restricted,
        source_manifest_path=source_path,
        reporting_policy_path=POLICY,
        output_root=restricted / "materialized",
    )
    assert replay == output
    assert replay_state == "verified"
