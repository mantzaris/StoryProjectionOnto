#!/usr/bin/env python3
"""Run and score the production CPU baseline on the four development worlds.

No model output is synthesized. Query opening follows durable C0 seals. Gold
is imported only AFTER all twelve CPU projections have been persisted. This is
a standalone calibration record, not a substitute for the integrated 24-call gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from story_projection_onto.benchmark_runtime import RuntimeStagingManifest
from story_projection_onto.conditions.base import (
    SCORED_PROJECTION_SCHEMA_HASH,
    ProduceInputs,
    RunConditionConfig,
)
from story_projection_onto.conditions.c0 import load_production_classical_builder
from story_projection_onto.contracts import (
    ConditionName,
    EvidencePacket,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    RetrievalMethod,
    canonical_sha256,
)
from story_projection_onto.development_adapter import DevelopmentConstructionConfiguration
from story_projection_onto.development_continuation import (
    development_validator_hash,
    load_development_prequery_evidence,
)
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.manifest import write_json_atomic
from story_projection_onto.query_runtime import AuditedBenchmarkRuntime
from story_projection_onto.store import BlobStore, Ledger, ReleaseClass


def rebind(record, **updates):
    return type(record).model_validate(record.model_dump(exclude={"content_hash"}) | updates)


def score_existing(root: Path, output: Path):
    """Reuse already persisted CPU projections; never re-run construction."""
    from story_projection_onto.conditions.base import ConditionAttemptRecord
    from story_projection_onto.scorer_only.development_assessment import (
        DevelopmentScientificAssessmentProvider,
        resolve_development_world_id,
    )
    from story_projection_onto.store import ReadOnlyLedger
    from story_projection_onto.synthetic_benchmark import ScorerWorldArtifact

    tic = time.monotonic()
    manifest = load_development_call_manifest(root)
    neutral, visible, _ = load_development_prequery_evidence(root, manifest)
    scorers, projections = {}, []
    for unit, artifact in neutral.items():
        world_id = resolve_development_world_id(
            root, visible[unit].content_hash, artifact.content_hash
        )
        scorers[unit] = ScorerWorldArtifact.model_validate_json(
            (root / f"data/synthetic/scorer_only/development/{world_id}.json").read_bytes()
        )
        for ordinal in range(1, 4):
            attempt = ConditionAttemptRecord.model_validate_json(
                (output / f"{unit}.q{ordinal}.json").read_bytes()
            )
            if attempt.projection is None:
                raise ValueError("persisted C0 attempt is invalid")
            projections.append(
                SimpleNamespace(
                    receipt=SimpleNamespace(
                        condition=ConditionName.C0_CLASSICAL_PRE,
                        unit_id=unit,
                        query_ordinal=ordinal,
                    ),
                    packet=SimpleNamespace(
                        ordered_evidence_ids=artifact.snapshot.eligible_evidence_ids
                    ),
                    projection=attempt.projection,
                )
            )
    with ReadOnlyLedger(output / "cpu.sqlite") as ledger:
        assert ledger.gpu_summary().total_allocated_seconds == 0
    assessor = SimpleNamespace(
        _alignment_plan=DevelopmentScientificAssessmentProvider._alignment_plan
    )
    coverage, precision, recall, evidence = DevelopmentScientificAssessmentProvider._assess_c0(
        assessor, projections, scorers
    )
    result = {
        "kind": "rescored_preserved_production_c0_development",
        "preconstructions": 4,
        "projections": len(projections),
        "explicit_family_coverage": coverage,
        "direct_assertion_precision": precision,
        "direct_assertion_recall": recall,
        "valid_evidence_reference_rate": evidence,
        "gpu_allocated_seconds": 0,
        "scoring_cpu_seconds": time.monotonic() - tic,
        "passes": coverage == 1 and precision >= 0.85 and recall >= 0.70 and evidence == 1,
        "thresholds": {"precision": 0.85, "recall": 0.70, "coverage": 1, "evidence": 1},
        "generation_repeated": False,
        "integrated_development_gate_substituted": False,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.json")
        },
    }
    if (output / "calibration.json").exists():
        raise FileExistsError("calibration result is immutable")
    write_json_atomic(result, output / "calibration.json")
    print(json.dumps(result))


def run(root: Path, output: Path, snapshot: Path):
    if output.exists() or "restricted" not in output.parts:
        raise ValueError("new restricted calibration directory required")
    output.mkdir(parents=True, mode=0o700)
    started = datetime.now(UTC)
    tic = time.monotonic()
    manifest = load_development_call_manifest(root)
    neutral, visible, _ = load_development_prequery_evidence(root, manifest)
    config = DevelopmentConstructionConfiguration.load()
    builder, backend = load_production_classical_builder()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    preparations = {}
    for unit, artifact in neutral.items():
        now = datetime.now(UTC)
        prepared = builder.prepare(
            snapshot=artifact.snapshot,
            evidence=artifact.evidence,
            upper_ontology=config.upper_ontology,
            preconstruction_budgets=config.preconstruction_budgets,
            constructed_at=now,
            sealed_at=now,
        )
        # Bind the actual completed CPU seal time, after construction returns.
        sealed = datetime.now(UTC)
        ontology = prepared.sealed_preontology
        ontology = rebind(
            ontology, construction_seal=rebind(ontology.construction_seal, sealed_at=sealed)
        )
        prepared = rebind(prepared, sealed_preontology=ontology, completed_at=sealed)
        preparations[unit] = prepared
        write_json_atomic(prepared.model_dump(mode="json"), output / f"{unit}.preparation.json")
    bindings = tuple(
        PrequeryPreparationBinding(
            unit_id=neutral[unit].snapshot.world_or_window_id,
            condition=ConditionName.C0_CLASSICAL_PRE,
            snapshot_hash=prepared.snapshot_hash,
            preparation_hash=prepared.content_hash,
            lineage_artifact_hash=prepared.sealed_preontology.construction_seal.content_hash,
            completed_at=prepared.completed_at,
        )
        for unit, prepared in preparations.items()
    )
    barrier = PrequeryBarrier(
        barrier_id="c0-development-calibration-barrier",
        execution_id=output.name,
        execution_manifest_hash=manifest.content_hash,
        neutral_evidence_artifact_hashes=tuple(item.content_hash for item in visible.values()),
        preparation_bindings=bindings,
        sealed_at=datetime.now(UTC),
    )
    projections = []
    with Ledger(output / "cpu.sqlite") as ledger:
        runtime = AuditedBenchmarkRuntime(ledger=ledger, blobs=BlobStore(output / "blobs"))
        # The benchmark's evidence is PUBLIC-eligible synthetic data. All files
        # still reside in ignored restricted calibration storage; no publication.
        runtime.persist_prequery_barrier(barrier, release_class=ReleaseClass.PUBLIC)
        for source in json.loads(
            (root / "configs/study/development_call_manifest.json").read_bytes()
        )["units"]:
            unit = source["unit_id"]
            artifact = neutral[unit]
            for ordinal, stage in enumerate(source["query_stages"], 1):
                directory = root / stage["relative_path"]
                staged_manifest = RuntimeStagingManifest.model_validate_json(
                    (directory / "manifest.json").read_bytes()
                )
                opening = runtime.open_query(
                    staging_root=None,
                    manifest=staged_manifest,
                    barrier=barrier,
                    execution_id=barrier.execution_id,
                    execution_manifest_hash=manifest.content_hash,
                    access_event_id=f"{unit}-query-{ordinal}",
                    repository=root,
                    stage_relative_path=stage["relative_path"],
                    manifest_file_sha256=stage["manifest_file_sha256"],
                )

                def materializer(
                    evidence, reveal, *, unit=unit, ordinal=ordinal, artifact=artifact
                ):
                    return EvidencePacket(
                        packet_id=f"{unit}-packet-{ordinal}",
                        snapshot_hash=artifact.snapshot.content_hash,
                        evidence=artifact.evidence,
                        ordered_evidence_ids=artifact.snapshot.eligible_evidence_ids,
                        retrieval_method=RetrievalMethod.ALL_ADMISSIBLE,
                        token_count=sum(
                            len(tokenizer.encode(row.text, add_special_tokens=False))
                            for row in artifact.evidence
                        ),
                        created_at=datetime.now(UTC),
                        release_class=artifact.snapshot.release_class,
                    )

                packet = runtime.materialize_packet(
                    opening,
                    materialization_event_id=f"{unit}-materialize-{ordinal}",
                    retrieval_config_hash=canonical_sha256(config.retrieval_policy),
                    materializer=materializer,
                )
                context = opening.context
                run_config = RunConditionConfig(
                    config_id=f"{unit}-calibration",
                    condition=ConditionName.C0_CLASSICAL_PRE,
                    budgets=config.projection_budgets_by_unit[unit],
                    maximum_input_tokens=10240,
                    maximum_output_tokens=2048,
                    scored_schema_hash=SCORED_PROJECTION_SCHEMA_HASH,
                    validator_hash=development_validator_hash(root),
                    upper_ontology_hash=config.upper_ontology.content_hash,
                )
                inputs = ProduceInputs(
                    preparation=preparations[unit],
                    snapshot=artifact.snapshot,
                    packet=packet.packet,
                    context=context,
                    query_access=opening.access_event,
                    prequery_barrier=barrier,
                    packet_materialization=packet.event,
                    query_processing_started_at=datetime.now(UTC),
                    upper_ontology=config.upper_ontology,
                    run_config=run_config,
                )
                try:
                    attempt = builder.produce(inputs)
                except Exception as error:
                    report = getattr(error, "validation_report", None)
                    if report is not None:
                        write_json_atomic(
                            report.model_dump(mode="json"),
                            output / f"{unit}.q{ordinal}.failure.json",
                        )
                    raise
                write_json_atomic(
                    attempt.model_dump(mode="json"), output / f"{unit}.q{ordinal}.json"
                )
                if attempt.projection is None:
                    raise ValueError(f"C0 projection failed: {unit}/{ordinal}")
                projections.append(
                    SimpleNamespace(
                        receipt=SimpleNamespace(
                            condition=ConditionName.C0_CLASSICAL_PRE,
                            unit_id=unit,
                            query_ordinal=ordinal,
                        ),
                        packet=packet.packet,
                        projection=attempt.projection,
                    )
                )
        assert ledger.gpu_summary().total_allocated_seconds == 0
    # No query or scorer access occurred during the preconstruction loop.
    from story_projection_onto.scorer_only.development_assessment import (
        DevelopmentScientificAssessmentProvider,
        resolve_development_world_id,
    )
    from story_projection_onto.synthetic_benchmark import BenchmarkSplit, ScorerWorldArtifact

    scorers = {}
    for unit, artifact in neutral.items():
        world_id = resolve_development_world_id(
            root, visible[unit].content_hash, artifact.content_hash
        )
        path = root / "data/synthetic/scorer_only/development" / f"{world_id}.json"
        scorer = ScorerWorldArtifact.model_validate_json(path.read_bytes())
        if scorer.world_spec.split is not BenchmarkSplit.DEVELOPMENT:
            raise ValueError("only development scorer records are allowed")
        scorers[unit] = scorer
    # Reuse the exact existing pure C0 aggregation implementation; this adapter
    # supplies only its static alignment function, never a pass/fail judgment.
    assessor = SimpleNamespace(
        _alignment_plan=DevelopmentScientificAssessmentProvider._alignment_plan
    )
    coverage, precision, recall, evidence = DevelopmentScientificAssessmentProvider._assess_c0(
        assessor, projections, scorers
    )
    result = {
        "kind": "production_c0_development_calibration",
        "backend": backend.model_dump(mode="json"),
        "started_at": started.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "cpu_wall_seconds": time.monotonic() - tic,
        "gpu_allocated_seconds": 0,
        "preconstructions": len(preparations),
        "projections": len(projections),
        "explicit_family_coverage": coverage,
        "direct_assertion_precision": precision,
        "direct_assertion_recall": recall,
        "valid_evidence_reference_rate": evidence,
        "thresholds": {"precision": 0.85, "recall": 0.70, "coverage": 1, "evidence": 1},
        "passes": precision >= 0.85 and recall >= 0.70 and coverage == 1 and evidence == 1,
        "integrated_development_gate_substituted": False,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.json")
        },
    }
    write_json_atomic(result, output / "calibration.json")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--score-existing", action="store_true")
    args = parser.parse_args()
    if args.score_existing:
        score_existing(Path.cwd(), args.output)
    else:
        run(Path.cwd(), args.output, args.snapshot)
