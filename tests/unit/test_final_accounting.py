from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from story_projection_onto.contracts import canonical_sha256
from story_projection_onto.development_runtime import load_development_call_manifest
from story_projection_onto.experiment import (
    REGISTERED_CALL_CLASSES,
    SESSION_START_CLASS,
)
from story_projection_onto.final_accounting import (
    FAILURE_ACCOUNTING_COLUMNS,
    RESOURCE_ACCOUNTING_COLUMNS,
    FinalAccountingError,
    LedgerSourceRoute,
    NativeSourceRole,
    NativeSourceRoute,
    SelfHashField,
    SourceFileRoute,
    _native_source_index,
    _service_event_belongs_to_session,
    _validate_native_source_payload,
    _verify_self_hash,
    build_final_accounting_source_recipe,
    cas_inventory_sha256,
    compile_final_accounting,
    materialize_final_accounting_recipe,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    Compression,
    GpuEventKind,
    GpuServiceJournalState,
    Ledger,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
    StoragePreflight,
)

T0 = "2026-09-05T00:00:00Z"
T1 = "2026-09-05T00:00:01Z"
T2 = "2026-09-05T00:00:02Z"
T3 = "2026-09-05T00:00:03Z"
T5 = "2026-09-05T00:00:05Z"
T6 = "2026-09-05T00:00:06Z"
T7 = "2026-09-05T00:00:07Z"
T8 = "2026-09-05T00:00:08Z"
T9 = "2026-09-05T00:00:09Z"
HASH_A = "a" * 64
HASH_B = "b" * 64


def test_successful_service_start_event_may_straddle_sampled_session_boundary() -> None:
    session = {
        "service_session_id": "service-start-001",
        "session_id": "acceptance-run-v3",
        "started_at": T1,
        "ended_at": T5,
    }
    event = {
        "event_id": "service-start-001",
        "attempt_id": None,
        "event_kind": GpuEventKind.GPU_SESSION_START.value,
        "started_at": "2026-09-05T00:00:00.973000Z",
        "ended_at": T2,
        "details_json": json.dumps(
            {
                "intended_event_kind": SESSION_START_CLASS,
                "session_id": "acceptance-run-v3",
            }
        ),
    }

    assert _service_event_belongs_to_session(session, event)
    assert not _service_event_belongs_to_session(
        session,
        {**event, "event_id": "unrelated-service-start"},
    )


def _write_self_hashed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["manifest_sha256"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _successful_fallback_result(*, with_repair: bool) -> dict[str, Any]:
    call_rows = [
        {"call_id": "fallback-c1-01", "status": "completed"},
        {
            "call_id": "fallback-c2-01",
            "status": "failed" if with_repair else "completed",
        },
        {"call_id": "fallback-c2-02", "status": "completed"},
        {"call_id": "fallback-fixed-01", "status": "completed"},
    ]
    reserve_rows = [
        {
            "call_id": "fallback-c1-01",
            "reservation_id": "fallback-v5:fallback-c1-01",
            "reserve_call_class": "reserve_long",
            "watchdog_seconds": 240,
        },
        {
            "call_id": "fallback-c2-01",
            "reservation_id": "fallback-v5:fallback-c2-01",
            "reserve_call_class": "reserve_standard",
            "watchdog_seconds": 150,
        },
        {
            "call_id": "fallback-c2-02",
            "reservation_id": "fallback-v5:fallback-c2-02",
            "reserve_call_class": "reserve_standard",
            "watchdog_seconds": 150,
        },
        {
            "call_id": "fallback-fixed-01",
            "reservation_id": "fallback-v5:fallback-fixed-01",
            "reserve_call_class": "reserve_short",
            "watchdog_seconds": 90,
        },
    ]
    if with_repair:
        call_rows.insert(
            2,
            {"call_id": "fallback-c2-01-repair-01", "status": "completed"},
        )
        reserve_rows.insert(
            2,
            {
                "call_id": "fallback-c2-01-repair-01",
                "reservation_id": "fallback-v5:fallback-c2-01-repair-01",
                "reserve_call_class": "reserve_short",
                "watchdog_seconds": 90,
            },
        )
    return {
        "schema_version": "1.0.0",
        "kind": "phase1_fallback_micro_pilot_result",
        "run_id": "fallback-qwen3-8b-awq-development-v5",
        "gate_passed": True,
        "base_call_count": 4,
        "completed_base_call_count": 4,
        "repair_attempt_count": int(with_repair),
        "calls": call_rows,
        "reserve_consumption": reserve_rows,
        "development_execution_result": {
            "kind": "development_execution_result",
            "execution_id": "development-fallback-v5",
            "content_hash": "d" * 64,
        },
    }


def _phase(call_class: str) -> str:
    if call_class.startswith(("acceptance_", "development_", "reserve_")):
        return "phase_1"
    if call_class.startswith(("scripted_", "researcher_")):
        return "phase_5"
    if call_class.startswith("case_"):
        return "phase_6"
    return "phase_3"


def _condition(call_class: str) -> str:
    if call_class.startswith("reserve_"):
        return "reserve_repair"
    if "fixed_select" in call_class:
        return "a_fixed_select"
    if "c1" in call_class:
        return "c1_llm_pre"
    if "c2" in call_class:
        return "c2_llm_query"
    if call_class == "ablation_no_context":
        return "a_no_context"
    if call_class == "ablation_no_temporal_epistemic":
        return "a_no_temporal_epistemic"
    if call_class == "ablation_no_rare_guard":
        return "a_no_rare_guard"
    return "study_control"


def _registered_slots(job_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call_class, (count, _watchdog) in REGISTERED_CALL_CLASSES.items():
        if call_class == SESSION_START_CLASS:
            continue
        for ordinal in range(1, count + 1):
            slot_id = f"{call_class}-{ordinal:03d}"
            row: dict[str, Any] = {
                "slot_id": slot_id,
                "call_class": call_class,
                "ordinal": ordinal,
                "phase": _phase(call_class),
                "run_id": f"planned-{call_class}",
                "call_id": slot_id,
                "condition": _condition(call_class),
                "included_in_itt": call_class.startswith(("test_", "paraphrase_", "ablation_")),
                "ledger_id": None,
                "job_id": None,
                "attempt_id": None,
                "model_call_id": None,
                "gpu_event_ids": [],
            }
            if call_class == "acceptance_c1" and ordinal == 1:
                row.update(
                    {
                        "ledger_id": "ledger-final",
                        "job_id": job_id,
                        "attempt_id": "attempt-one",
                        "model_call_id": "model-call-one",
                        "gpu_event_ids": ["inference-one"],
                        "run_id": "final-accounting-test",
                    }
                )
            rows.append(row)
    return rows


def _fixture(
    root: Path,
    *,
    amendment_interleaved: bool = False,
    record_storage: bool = True,
    record_resources: bool = True,
    reserve_c1: bool = False,
    cover_all_services: bool = True,
    pair_resource_storage: bool = True,
    unclassified_attempt: bool = False,
    conflicting_amendment_identity: bool = False,
    short_presampler_failure: bool = False,
) -> tuple[Path, Path]:
    config = root / "configs" / "gpu_call_inventory.json"
    config.parent.mkdir(parents=True)
    shutil.copy2(Path("configs/study/gpu_call_inventory.json"), config)
    ledger_path = root / "frozen" / "study.sqlite"
    cas_root = root / "frozen" / "cas"
    ledger_path.parent.mkdir(parents=True)
    storage_ids: list[str] = []
    with Ledger(ledger_path) as ledger:
        artifacts = ArtifactStore(BlobStore(cas_root, compression=Compression.GZIP), ledger)
        job = ledger.create_or_resume_job(
            {"run": "final-accounting-test"},
            release_class=ReleaseClass.PUBLIC,
            created_at=T0,
        )
        attempt = ledger.record_attempt(
            attempt_id="attempt-one",
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=HASH_A,
            config_hash=HASH_B,
            seed=1,
            created_at=T1,
        )
        if unclassified_attempt:
            unknown_job = ledger.create_or_resume_job(
                {"run": "unclassified-scientific-attempt"},
                release_class=ReleaseClass.PUBLIC,
                created_at=T1,
            )
            ledger.record_attempt(
                attempt_id="attempt-unclassified",
                job_id=unknown_job.job_id,
                attempt_kind=AttemptKind.BASE,
                input_hash=HASH_A,
                config_hash=HASH_B,
                seed=2,
                created_at=T1,
            )
        request = artifacts.put_bytes(
            b'{"request":"one"}\n',
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T1,
        )
        response = artifacts.put_bytes(
            b'{"response":"one"}\n',
            media_type="application/x-ndjson",
            release_class=ReleaseClass.PUBLIC,
            created_at=T3,
        )
        ledger.record_gpu_service_observation(
            service_session_id="service-one",
            state=GpuServiceJournalState.OPENED,
            session_id="server-one",
            configuration_hash=HASH_A,
            service_started_at=T0,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T0,
        )
        ledger.record_gpu_event(
            event_id="service-load-one",
            event_kind=GpuEventKind.MODEL_LOAD,
            allocated_seconds=2,
            started_at=T0,
            ended_at=T2,
            succeeded=True,
            details={"call_class": "gpu_session_start"},
        )
        event = ledger.record_gpu_event(
            event_id="inference-one",
            event_kind=GpuEventKind.INFERENCE,
            allocated_seconds=1,
            started_at=T2,
            ended_at=T3,
            succeeded=True,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            details=(
                {
                    "reserve_call_class": "reserve_long",
                    "request_id": "fallback-c1-01",
                    "intended_event_kind": "fallback_test",
                }
                if reserve_c1
                else {
                    "call_class": "acceptance_c1",
                    "call_id": "acceptance_c1-001",
                }
            ),
        )
        ledger.record_model_call(
            model_call_id="model-call-one",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            gpu_event_id=event.event_id,
            backend=ModelBackend.VLLM_GPU,
            call_role=ModelCallRole.PREBUILD,
            retry_class=RetryClass.BASE,
            model_manifest_hash=HASH_A,
            decoding_manifest_hash=HASH_B,
            request_hash=request.content_hash,
            response_artifact_hash=response.content_hash,
            construction_unit_hash=HASH_A,
            served_context_count=3,
            prompt_tokens=11,
            completion_tokens=7,
            allocated_gpu_seconds=1,
            successful=True,
            created_at=T3,
        )
        ledger.record_gpu_service_observation(
            service_session_id="service-one",
            state=GpuServiceJournalState.PROCESS_STOPPED,
            session_id="server-one",
            configuration_hash=HASH_A,
            service_started_at=T0,
            elapsed_seconds=5,
            ledger_allocated_seconds_before_session=0,
            hard_limit_seconds=36_000,
            observed_at=T5,
        )
        ledger.close_gpu_service_journal(
            service_session_id="service-one",
            session_id="server-one",
            service_seconds=5,
            classified_event_seconds=3,
            started_at=T0,
            ended_at=T5,
            details={"accounting": "mixed-events-and-overhead"},
        )
        if amendment_interleaved:
            ledger.record_gpu_service_observation(
                service_session_id="service-amendment-v3",
                state=GpuServiceJournalState.OPENED,
                session_id="authorized-recovery-v3",
                configuration_hash=HASH_A,
                service_started_at=T6,
                elapsed_seconds=0,
                ledger_allocated_seconds_before_session=5,
                hard_limit_seconds=36_000,
                observed_at=T6,
            )
            ledger.record_gpu_service_observation(
                service_session_id="service-amendment-v3",
                state=GpuServiceJournalState.PROCESS_STOPPED,
                session_id="authorized-recovery-v3",
                configuration_hash=HASH_A,
                service_started_at=T6,
                elapsed_seconds=1,
                ledger_allocated_seconds_before_session=5,
                hard_limit_seconds=36_000,
                observed_at=T7,
            )
            ledger.close_gpu_service_journal(
                service_session_id="service-amendment-v3",
                session_id="authorized-recovery-v3",
                service_seconds=1,
                classified_event_seconds=0,
                started_at=T6,
                ended_at=T7,
                details={
                    "authorized_recovery_run_id": (
                        "unauthorized-recovery"
                        if conflicting_amendment_identity
                        else "authorized-recovery-v3"
                    )
                },
            )
            ledger.record_gpu_service_observation(
                service_session_id="service-base-two",
                state=GpuServiceJournalState.OPENED,
                session_id="server-base-two",
                configuration_hash=HASH_A,
                service_started_at=T8,
                elapsed_seconds=0,
                ledger_allocated_seconds_before_session=6,
                hard_limit_seconds=36_000,
                observed_at=T8,
            )
            ledger.record_gpu_service_observation(
                service_session_id="service-base-two",
                state=GpuServiceJournalState.PROCESS_STOPPED,
                session_id="server-base-two",
                configuration_hash=HASH_A,
                service_started_at=T8,
                elapsed_seconds=1,
                ledger_allocated_seconds_before_session=6,
                hard_limit_seconds=36_000,
                observed_at=T9,
            )
            ledger.close_gpu_service_journal(
                service_session_id="service-base-two",
                session_id="server-base-two",
                service_seconds=1,
                classified_event_seconds=0,
                started_at=T8,
                ended_at=T9,
                details={
                    "accounting": ("diagnostic-authorized-recovery-v3-suffix-is-not-an-identity")
                },
            )
        if short_presampler_failure:
            short_start = "2026-09-05T00:00:06.000000Z"
            short_end = "2026-09-05T00:00:06.060000Z"
            short_service = "short-presampler-service-start-001"
            ledger.record_gpu_service_observation(
                service_session_id=short_service,
                state=GpuServiceJournalState.OPENED,
                session_id="short-presampler-run",
                configuration_hash=HASH_A,
                service_started_at=short_start,
                elapsed_seconds=0,
                ledger_allocated_seconds_before_session=5,
                hard_limit_seconds=36_000,
                observed_at=short_start,
            )
            ledger.record_gpu_event(
                event_id=short_service,
                event_kind=GpuEventKind.FAILURE,
                allocated_seconds=0.09633,
                started_at="2026-09-05T00:00:05.990000Z",
                ended_at="2026-09-05T00:00:06.086330Z",
                succeeded=False,
                details={
                    "intended_event_kind": "gpu_session_start",
                    "session_id": "short-presampler-run",
                    "exception_type": "TypeError",
                },
            )
            ledger.record_gpu_service_observation(
                service_session_id=short_service,
                state=GpuServiceJournalState.PROCESS_STOPPED,
                session_id="short-presampler-run",
                configuration_hash=HASH_A,
                service_started_at=short_start,
                elapsed_seconds=0.09633,
                ledger_allocated_seconds_before_session=5,
                hard_limit_seconds=36_000,
                observed_at=short_end,
            )
            ledger.close_gpu_service_journal(
                service_session_id=short_service,
                session_id="short-presampler-run",
                service_seconds=0.09633,
                classified_event_seconds=0.09633,
                started_at=short_start,
                ended_at=short_end,
                details={"accounting": "pre-sampler launch failure"},
            )
        storage = StoragePreflight(root).check(
            current_occupied_bytes=1_000,
            declared_growth_bytes=0,
            filesystem_free_bytes=10_000_000_000,
        )
        storage_id = "absent-storage-sample"
        if record_storage:
            storage_id = ledger.record_storage_sample(storage, phase="phase_7", sampled_at=T5)
            storage_ids.append(storage_id)
        if record_resources:
            ledger.record_resource_sample(
                sample_id="resource-one",
                job_id=job.job_id,
                gpu_event_id="inference-one",
                process_ram_bytes=100,
                system_available_ram_bytes=1_000,
                gpu_vram_bytes=200,
                project_storage_bytes=1_000,
                cpu_worker_count=1,
                sampled_at=T3,
            )
            if record_storage and pair_resource_storage:
                storage_ids.append(
                    ledger.record_storage_sample(
                        storage,
                        phase="resource_sample:resource-one",
                        sampled_at=T3,
                    )
                )
            if amendment_interleaved and cover_all_services:
                for sample_id, sampled_at in (
                    ("resource-amendment", T6),
                    ("resource-base-two", T8),
                ):
                    ledger.record_resource_sample(
                        sample_id=sample_id,
                        job_id=None,
                        gpu_event_id=None,
                        process_ram_bytes=100,
                        system_available_ram_bytes=1_000,
                        gpu_vram_bytes=200,
                        project_storage_bytes=1_000,
                        cpu_worker_count=1,
                        sampled_at=sampled_at,
                    )
                    if record_storage and pair_resource_storage:
                        storage_ids.append(
                            ledger.record_storage_sample(
                                storage,
                                phase=f"resource_sample:{sample_id}",
                                sampled_at=sampled_at,
                            )
                        )

    call_slots = _registered_slots(job.job_id)
    service_slots = [
        {
            "slot_id": f"service-base-{ordinal:03d}",
            "source": "base_inventory",
            "ordinal": ordinal,
            "phase": "phase_1",
            "run_id": "server-one" if ordinal == 1 else f"planned-service-base-{ordinal:03d}",
            "call_id": "service-one" if ordinal == 1 else f"service-base-{ordinal:03d}",
            "amendment_artifact_id": None,
            "ledger_id": "ledger-final" if ordinal == 1 else None,
            "service_session_id": "service-one" if ordinal == 1 else None,
            "gpu_event_ids": ["service-load-one"] if ordinal == 1 else [],
        }
        for ordinal in range(1, 9)
    ]
    receipts_root = root / "receipts"
    receipts: list[dict[str, Any]] = []

    native_path = root / "native" / "acceptance_manifest.json"
    _write_self_hashed(
        native_path,
        {
            "schema_version": "1.0.0",
            "kind": (
                "phase1_fallback_micro_pilot_result"
                if reserve_c1
                else "phase1_gpu_acceptance_result"
            ),
            "run_id": "final-accounting-test",
            "gate_passed": not reserve_c1,
            "base_call_count": 4 if reserve_c1 else None,
            "completed_base_call_count": 0 if reserve_c1 else None,
            "repair_attempt_count": 0 if reserve_c1 else None,
            "calls": [
                (
                    {
                        "call_id": "fallback-c1-01",
                        "status": "failed",
                        "reserve_call_class": "reserve_long",
                        "forecast_proxy_call_class": "acceptance_c1",
                        "condition": "C1",
                        "model_call_id": "model-call-one",
                    }
                    if reserve_c1
                    else {
                        "call_id": "acceptance-one",
                        "call_class": "acceptance_c1",
                        "model_call_id": "model-call-one",
                    }
                )
            ],
            "reserve_consumption": (
                [
                    {
                        "call_id": "fallback-c1-01",
                        "gpu_event_id": "inference-one",
                        "reservation_id": "fallback-c1-reservation",
                        "reserve_call_class": "reserve_long",
                        "watchdog_seconds": 240,
                    }
                ]
                if reserve_c1
                else []
            ),
        },
    )
    wall_path = receipts_root / "wall.json"
    _write_self_hashed(
        wall_path,
        {
            "schema_version": "1.0.0",
            "kind": "runpod_container_wall_time_sample",
            "scope": "current_container_pid_1_lifetime",
            "container_pid": 1,
            "container_pid_1_started_at": T0,
            "sampled_at": T9 if amendment_interleaved or short_presampler_failure else T5,
            "elapsed_microseconds": (
                9_000_000 if amendment_interleaved or short_presampler_failure else 5_000_000
            ),
            "measurement_method": "linux_proc_boot_epoch_plus_pid1_start_ticks",
            "scientific_gpu_accounting": False,
            "billing_time_equivalence_claimed": False,
            "caveat": "test lower bound",
        },
    )
    amendment_routes: list[dict[str, Any]] = []
    if amendment_interleaved:
        amendment_path = root / "configs" / "service_amendment.json"
        _write_self_hashed(
            amendment_path,
            {
                "kind": "phase1_fallback_service_retry_amendment",
                "authorization_status": "authorized",
                "authorized_recovery_run_id": "authorized-recovery-v3",
                "base_gpu_call_inventory_file_sha256": _file_hash(config),
                "amendment": {
                    "additional_fallback_service_loads": 1,
                    "additional_unreserved_inference_attempts": 0,
                    "amended_maximum_inference_attempts": 278,
                },
            },
        )
        amendment_routes.append(
            {
                "artifact_id": "service-amendment",
                "relative_path": amendment_path.relative_to(root).as_posix(),
                "self_hash_field": "manifest_sha256",
            }
        )
    source_recipe_path = root / "source_recipe.json"
    source_recipe = _write_self_hashed(
        source_recipe_path,
        {
            "schema_version": "1.0.0",
            "kind": "final_phase7_accounting_source_recipe",
            "accounting_id": "accounting-test",
            "compiled_at_utc": T9 if amendment_interleaved or short_presampler_failure else T5,
            "base_call_inventory": {
                "artifact_id": "base-inventory",
                "relative_path": config.relative_to(root).as_posix(),
                "self_hash_field": None,
            },
            "native_source_artifacts": [
                {
                    "artifact_id": "native-phase",
                    "relative_path": native_path.relative_to(root).as_posix(),
                    "self_hash_field": "manifest_sha256",
                    "producer_role": "phase1_acceptance_result",
                }
            ],
            "authorization_amendments": amendment_routes,
            "ledger_sources": [
                {
                    "ledger_id": "ledger-final",
                    "lineage_id": "study",
                    "sequence": 0,
                    "parent_ledger_id": None,
                    "ledger_relative_path": ledger_path.relative_to(root).as_posix(),
                    "cas_relative_path": cas_root.relative_to(root).as_posix(),
                }
            ],
            "wall_time_receipts": [
                {
                    "artifact_id": "wall",
                    "relative_path": wall_path.relative_to(root).as_posix(),
                    "self_hash_field": "manifest_sha256",
                }
            ],
        },
    )
    native_file_hashes = sorted({_file_hash(source_recipe_path), _file_hash(native_path)})
    ledger_file_hashes = [_file_hash(ledger_path)]

    def receipt(name: str, payload: dict[str, Any]) -> None:
        path = receipts_root / f"{name}.json"
        payload.update(
            {
                "generated_by": "story_projection_onto.final_accounting.materializer/v1",
                "source_materialization_recipe_sha256": source_recipe["manifest_sha256"],
                "source_native_file_sha256": native_file_hashes,
                "source_ledger_file_sha256": ledger_file_hashes,
                "source_authorization_file_sha256": [],
                "base_call_inventory_file_sha256": _file_hash(config),
            }
        )
        _write_self_hashed(path, payload)
        receipts.append(
            {
                "artifact_id": name,
                "relative_path": path.relative_to(root).as_posix(),
                "file_sha256": _file_hash(path),
                "self_hash_field": "manifest_sha256",
                "receipt_kind": payload["kind"],
            }
        )

    receipt(
        "execution",
        {
            "kind": "final_execution_inventory_receipt",
            "call_bindings": [
                {
                    "slot_id": "acceptance_c1-001",
                    "ledger_id": "ledger-final",
                    "job_id": job.job_id,
                    "attempt_id": "attempt-one",
                    "model_call_id": "model-call-one",
                    "gpu_event_ids": ["inference-one"],
                }
            ],
        },
    )
    all_slot_ids = [row["slot_id"] for row in call_slots]
    itt_ids = [row["slot_id"] for row in call_slots if row["included_in_itt"]]
    terminal_outcomes = [
        {
            "slot_id": row["slot_id"],
            "outcome": (
                "success"
                if row["slot_id"] == "acceptance_c1-001"
                else "not_used"
                if row["call_class"].startswith("reserve_")
                else "incomplete"
            ),
        }
        for row in call_slots
    ]
    receipt(
        "results",
        {
            "kind": "final_result_inventory_receipt",
            "registered_slot_ids": all_slot_ids,
            "itt_slot_ids": itt_ids,
            "non_itt_slot_ids": [item for item in all_slot_ids if item not in set(itt_ids)],
            "terminal_outcomes": terminal_outcomes,
        },
    )
    receipt(
        "repairs",
        {
            "kind": "final_repair_inventory_receipt",
            "repair_bindings": [],
        },
    )
    receipt(
        "services",
        {
            "kind": "final_service_inventory_receipt",
            "registered_service_slot_ids": [row["slot_id"] for row in service_slots],
            "service_bindings": [
                {
                    "slot_id": "service-base-001",
                    "ledger_id": "ledger-final",
                    "service_session_id": "service-one",
                    "gpu_event_ids": ["service-load-one"],
                }
            ],
            "all_services_stopped": True,
        },
    )
    receipt(
        "tokens",
        {
            "kind": "final_token_accounting_receipt",
            "ledger_id": "ledger-final",
            "model_call_ids": ["model-call-one"],
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
    )
    receipt(
        "storage",
        {
            "kind": "final_storage_accounting_receipt",
            "ledger_id": "ledger-final",
            "storage_sample_ids": sorted(storage_ids),
            "resource_sample_ids": ["resource-one"] if record_resources else [],
            "uncovered_short_failed_service_session_ids": [],
            "resource_sampling_coverage_complete": True,
            "peak_project_storage_bytes": 1_000,
            "peak_projected_storage_bytes": 1_000,
            "minimum_effective_headroom_bytes": storage.effective_projected_headroom_bytes,
            "all_storage_samples_allowed": True,
        },
    )
    receipt(
        "exclusions",
        {
            "kind": "final_attempt_exclusion_receipt",
            "exclusions": [],
        },
    )
    receipts.append(
        {
            "artifact_id": "wall",
            "relative_path": wall_path.relative_to(root).as_posix(),
            "file_sha256": _file_hash(wall_path),
            "self_hash_field": "manifest_sha256",
            "receipt_kind": "runpod_container_wall_time_sample",
        }
    )

    recipe_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "final_phase7_accounting_recipe",
        "accounting_id": "accounting-test",
        "compiled_at_utc": T5,
        "source_materialization_recipe_sha256": source_recipe["manifest_sha256"],
        "base_call_inventory": {
            "artifact_id": "base-inventory",
            "relative_path": config.relative_to(root).as_posix(),
            "file_sha256": _file_hash(config),
            "self_hash_field": None,
        },
        "native_source_artifacts": [
            {
                "artifact_id": "fa-materialization-source-recipe",
                "relative_path": source_recipe_path.relative_to(root).as_posix(),
                "file_sha256": _file_hash(source_recipe_path),
                "self_hash_field": "manifest_sha256",
            },
            {
                "artifact_id": "native-phase",
                "relative_path": native_path.relative_to(root).as_posix(),
                "file_sha256": _file_hash(native_path),
                "self_hash_field": "manifest_sha256",
            },
        ],
        "authorization_amendments": [],
        "ledger_snapshots": [
            {
                "ledger_id": "ledger-final",
                "lineage_id": "study",
                "sequence": 0,
                "parent_ledger_id": None,
                "ledger_relative_path": ledger_path.relative_to(root).as_posix(),
                "ledger_file_sha256": _file_hash(ledger_path),
                "cas_relative_path": cas_root.relative_to(root).as_posix(),
                "cas_inventory_sha256": cas_inventory_sha256(cas_root),
            }
        ],
        "receipts": receipts,
        "call_slots": call_slots,
        "service_slots": service_slots,
        "attempt_exclusions": [],
    }
    recipe_path = root / "recipe.json"
    _write_self_hashed(recipe_path, recipe_payload)
    return recipe_path, root / "output"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_compiler_reconciles_registered_inventory_and_service_overhead(tmp_path: Path) -> None:
    recipe, output = _fixture(tmp_path)

    compiled = compile_final_accounting(
        recipe_path=recipe,
        source_root=tmp_path,
        output_root=output,
    )

    assert tuple(_read_csv(compiled.failure_table_path)[0]) == FAILURE_ACCOUNTING_COLUMNS
    failures = _read_csv(compiled.failure_table_path)
    assert len(failures) == 286
    inference = next(row for row in failures if row["call_id"] == "acceptance_c1-001")
    service = next(row for row in failures if row["call_id"] == "service-one")
    assert inference["outcome"] == "success"
    assert inference["outcome_scope"] == "execution_and_recorded_failure_not_scientific_acceptance"
    assert inference["scientific_status"] == "unresolved"
    assert inference["allocated_gpu_seconds"] == "1.000000"
    assert service["outcome"] == "success"
    assert service["scientific_status"] == "not_applicable"
    assert service["allocated_gpu_seconds"] == "4.000000"
    resources = _read_csv(compiled.resource_table_path)
    assert tuple(resources[0]) == RESOURCE_ACCOUNTING_COLUMNS
    actual = next(
        row
        for row in resources
        if row["scope"] == "study"
        and row["metric"] == "actual_allocated_gpu_time"
        and row["unit"] == "seconds"
    )
    overhead = next(
        row
        for row in resources
        if row["scope"] == "gpu_event_kind" and row["metric"] == "service_overhead"
    )
    assert actual["value"] == "5.000000"
    assert overhead["value"] == "2.000000"
    assert compiled.receipt["reconciliation"]["total_allocated_gpu_microseconds"] == 5_000_000
    assert compiled.receipt["reconciliation"]["service_overhead_microseconds"] == 2_000_000
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    replay = compile_final_accounting(
        recipe_path=recipe,
        source_root=tmp_path,
        output_root=output,
        verify_only=True,
    )
    assert replay.receipt == compiled.receipt
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_materializer_derives_receipts_and_recipe_then_replays(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    materialized_root = tmp_path / "materialized"

    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=materialized_root,
    )
    compiled = compile_final_accounting(
        recipe_path=materialized.recipe_path,
        source_root=tmp_path,
        output_root=tmp_path / "materialized-tables",
    )

    assert len(materialized.recipe.call_slots) == 278
    assert sum(item.executed for item in materialized.recipe.call_slots) == 1
    assert len(materialized.receipt_paths) == 7
    assert compiled.receipt["reconciliation"]["total_allocated_gpu_microseconds"] == 5_000_000
    before = {path.name: path.read_bytes() for path in materialized_root.iterdir()}
    replay = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=materialized_root,
        verify_only=True,
    )
    assert replay.recipe == materialized.recipe
    assert {path.name: path.read_bytes() for path in materialized_root.iterdir()} == before


def test_materializer_and_compiler_reject_native_source_tampering(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )
    native = tmp_path / "native" / "acceptance_manifest.json"
    payload = json.loads(native.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["extra"] = "tampered-but-self-hashed"
    _write_self_hashed(native, payload)

    with pytest.raises(FinalAccountingError, match="missing immutable output"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
            verify_only=True,
        )
    with pytest.raises(FinalAccountingError, match="physical SHA-256 mismatch"):
        compile_final_accounting(
            recipe_path=materialized.recipe_path,
            source_root=tmp_path,
            output_root=tmp_path / "tables",
        )


def test_materializer_verify_never_repairs_a_publish_partial(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )
    target = materialized.receipt_paths[0]
    partial = target.parent / f".{target.name}.{'a' * 32}.partial"
    os.link(target, partial)

    with pytest.raises(FinalAccountingError, match="singly-linked regular file"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
            verify_only=True,
        )
    assert partial.exists()
    assert target.stat().st_nlink == 2


def test_materializer_partitions_interleaved_amendment_session_by_identity(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path, amendment_interleaved=True)

    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )

    executed_base = [
        item
        for item in materialized.recipe.service_slots
        if item.source == "base_inventory" and item.executed
    ]
    executed_amendment = [
        item
        for item in materialized.recipe.service_slots
        if item.source == "authorized_amendment" and item.executed
    ]
    assert [item.service_session_id for item in executed_base] == [
        "service-one",
        "service-base-two",
    ]
    assert [item.service_session_id for item in executed_amendment] == ["service-amendment-v3"]


def test_materializer_rejects_uncovered_gpu_service(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(
        tmp_path,
        amendment_interleaved=True,
        cover_all_services=False,
    )

    with pytest.raises(FinalAccountingError, match="lacks resource-sample coverage"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_short_presampler_failure_is_explicit_incomplete_resource_evidence(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(
        tmp_path,
        short_presampler_failure=True,
    )
    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )
    compiled = compile_final_accounting(
        recipe_path=materialized.recipe_path,
        source_root=tmp_path,
        output_root=tmp_path / "tables",
    )

    resources = _read_csv(compiled.resource_table_path)
    by_metric = {row["metric"]: row for row in resources if row["scope"] == "study"}
    assert by_metric["resource_sampling_coverage_complete"] == {
        "scope": "study",
        "metric": "resource_sampling_coverage_complete",
        "value": "false",
        "unit": "boolean",
        "status": "incomplete_evidence",
        "source_note": (
            "one sample inside every allocated service except enumerated sub-second "
            "pre-sampler failures"
        ),
    }
    assert by_metric["peak_gpu_vram"]["status"] == "observed_lower_bound"
    assert compiled.receipt["reconciliation"]["uncovered_short_failed_gpu_service_ids"] == [
        "ledger-final:short-presampler-service-start-001"
    ]


def test_materializer_rejects_conflicting_service_amendment_identity(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(
        tmp_path,
        amendment_interleaved=True,
        conflicting_amendment_identity=True,
    )

    with pytest.raises(FinalAccountingError, match="unauthorized recovery run identity"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_materializer_rejects_resource_sample_without_storage_pair(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(
        tmp_path,
        pair_resource_storage=False,
    )

    with pytest.raises(FinalAccountingError, match="lacks its paired storage sample"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_materializer_preserves_underlying_condition_for_c1_reserve(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path, reserve_c1=True)

    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )

    executed = [item for item in materialized.recipe.call_slots if item.executed]
    assert len(executed) == 1
    assert executed[0].call_class == "reserve_long"
    assert executed[0].condition == "c1_llm_pre"
    assert executed[0].phase == "phase_1"


def test_content_self_hash_accepts_real_nested_immutable_manifest() -> None:
    manifest = load_development_call_manifest(Path.cwd())
    payload = manifest.model_dump(mode="json")
    pending: list[object] = [payload]
    content_hash_key_count = 0
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            content_hash_key_count += int("content_hash" in current)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)

    assert content_hash_key_count > 1
    _verify_self_hash(payload, SelfHashField.CONTENT, label="nested development manifest")


@pytest.mark.parametrize("with_repair", [False, True])
def test_phase1_native_source_accepts_complete_fallback_result(
    with_repair: bool,
) -> None:
    payload = _successful_fallback_result(with_repair=with_repair)
    route = NativeSourceRoute(
        artifact_id="accepted-fallback-v5",
        relative_path="native/fallback-v5.json",
        self_hash_field=SelfHashField.MANIFEST,
        producer_role=NativeSourceRole.PHASE1_ACCEPTANCE_RESULT,
    )

    _validate_native_source_payload(route, payload)
    roles, classes, repair_reserves, conditions = _native_source_index(((route, payload),))

    phase1_strings = roles[NativeSourceRole.PHASE1_ACCEPTANCE_RESULT]
    assert {
        "fallback-c1-01",
        "fallback-c2-01",
        "fallback-c2-02",
        "fallback-fixed-01",
    } <= phase1_strings
    assert "development-fallback-v5" not in phase1_strings
    assert classes["fallback-c1-01"] == frozenset({"acceptance_c1"})
    assert classes["fallback-c2-01"] == frozenset({"acceptance_c2"})
    assert classes["fallback-fixed-01"] == frozenset({"acceptance_fixed_select"})
    assert repair_reserves["fallback-c1-01"] == frozenset({"reserve_long"})
    assert conditions["fallback-c1-01"] == frozenset({"c1_llm_pre"})
    if with_repair:
        assert repair_reserves["fallback-c2-01-repair-01"] == frozenset({"reserve_short"})


def test_phase1_native_source_rejects_changed_fallback_call_inventory() -> None:
    payload = _successful_fallback_result(with_repair=False)
    payload["calls"][2]["call_id"] = "fallback-unregistered-01"
    route = NativeSourceRoute(
        artifact_id="changed-fallback-v5",
        relative_path="native/fallback-v5.json",
        self_hash_field=SelfHashField.MANIFEST,
        producer_role=NativeSourceRole.PHASE1_ACCEPTANCE_RESULT,
    )

    with pytest.raises(FinalAccountingError, match="unregistered call"):
        _validate_native_source_payload(route, payload)


def test_source_recipe_builder_derives_freeze_and_replays_without_writes(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    output = tmp_path / "built-sources"
    arguments = {
        "accounting_id": "accounting-test",
        "source_root": tmp_path,
        "output_root": output,
        "base_call_inventory": SourceFileRoute(
            artifact_id="base-call-inventory",
            relative_path="configs/gpu_call_inventory.json",
        ),
        "native_source_artifacts": (
            NativeSourceRoute(
                artifact_id="native-phase1",
                relative_path="native/acceptance_manifest.json",
                self_hash_field=SelfHashField.MANIFEST,
                producer_role=NativeSourceRole.PHASE1_ACCEPTANCE_RESULT,
            ),
        ),
        "authorization_amendments": (),
        "ledger_sources": (
            LedgerSourceRoute(
                ledger_id="ledger-final",
                lineage_id="study",
                sequence=0,
                ledger_relative_path="frozen/study.sqlite",
                cas_relative_path="frozen/cas",
            ),
        ),
        "wall_time_receipts": (
            SourceFileRoute(
                artifact_id="wall-time-001",
                relative_path="receipts/wall.json",
                self_hash_field=SelfHashField.MANIFEST,
            ),
        ),
    }

    built = build_final_accounting_source_recipe(**arguments)
    assert built.recipe.compiled_at_utc.isoformat() == "2026-09-05T00:00:05+00:00"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    replay = build_final_accounting_source_recipe(**arguments, verify_only=True)
    assert replay.recipe == built.recipe
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before

    materialized = materialize_final_accounting_recipe(
        source_recipe_path=built.recipe_path,
        source_root=tmp_path,
        output_root=tmp_path / "materialized-from-builder",
    )
    assert materialized.recipe.source_materialization_recipe_sha256 == (
        built.recipe.manifest_sha256
    )


def test_materializer_rejects_untyped_native_source_even_when_rehashed(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    native = tmp_path / "native" / "acceptance_manifest.json"
    payload = json.loads(native.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["kind"] = "operator_authored_call_annotations"
    _write_self_hashed(native, payload)

    with pytest.raises(FinalAccountingError, match="unrecognized producer kind"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_materializer_rejects_unclassified_scientific_attempt(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path, unclassified_attempt=True)

    with pytest.raises(FinalAccountingError, match="no registered call-class source"):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_materializer_requires_storage_evidence_in_each_terminal_ledger(
    tmp_path: Path,
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    other_ledger = tmp_path / "frozen" / "other.sqlite"
    other_cas = tmp_path / "frozen" / "other-cas"
    other_cas.mkdir()
    with Ledger(other_ledger):
        pass
    source = tmp_path / "source_recipe.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["ledger_sources"].append(
        {
            "ledger_id": "ledger-empty",
            "lineage_id": "empty-lineage",
            "sequence": 0,
            "parent_ledger_id": None,
            "ledger_relative_path": other_ledger.relative_to(tmp_path).as_posix(),
            "cas_relative_path": other_cas.relative_to(tmp_path).as_posix(),
        }
    )
    _write_self_hashed(source, payload)

    with pytest.raises(FinalAccountingError, match=r"ledger-empty.*no project-wide storage"):
        materialize_final_accounting_recipe(
            source_recipe_path=source,
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ("storage_samples", "no project-wide storage sample evidence"),
        ("resource_samples", "without any resource sample evidence"),
    ],
)
def test_materializer_rejects_vacuous_resource_compliance(
    tmp_path: Path, table: str, message: str
) -> None:
    _manual_recipe, _manual_output = _fixture(
        tmp_path,
        record_storage=table != "storage_samples",
        record_resources=table != "resource_samples",
    )

    with pytest.raises(FinalAccountingError, match=message):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_compiler_rejects_receipt_tampering(tmp_path: Path) -> None:
    recipe, output = _fixture(tmp_path)
    token_receipt = tmp_path / "receipts" / "tokens.json"
    payload = json.loads(token_receipt.read_text(encoding="utf-8"))
    payload["prompt_tokens"] = 12
    token_receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FinalAccountingError, match="physical SHA-256 mismatch"):
        compile_final_accounting(
            recipe_path=recipe,
            source_root=tmp_path,
            output_root=output,
        )


def test_compiler_rejects_rehashed_missing_registered_slot(tmp_path: Path) -> None:
    recipe, output = _fixture(tmp_path)
    payload = json.loads(recipe.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["call_slots"].pop()
    _write_self_hashed(recipe, payload)

    with pytest.raises(FinalAccountingError, match="exact registered 278"):
        compile_final_accounting(
            recipe_path=recipe,
            source_root=tmp_path,
            output_root=output,
        )


def test_compiler_rejects_rehashed_itt_receipt_drift(tmp_path: Path) -> None:
    recipe, output = _fixture(tmp_path)
    result_path = tmp_path / "receipts" / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("manifest_sha256")
    moved = result["itt_slot_ids"].pop()
    result["non_itt_slot_ids"].append(moved)
    _write_self_hashed(result_path, result)
    recipe_payload = json.loads(recipe.read_text(encoding="utf-8"))
    recipe_payload.pop("manifest_sha256")
    result_ref = next(
        item for item in recipe_payload["receipts"] if item["artifact_id"] == "results"
    )
    result_ref["file_sha256"] = _file_hash(result_path)
    _write_self_hashed(recipe, recipe_payload)

    with pytest.raises(FinalAccountingError, match="ITT membership"):
        compile_final_accounting(
            recipe_path=recipe,
            source_root=tmp_path,
            output_root=output,
        )


def test_compiler_rejects_rehashed_amendment_ordinal_drift(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path, amendment_interleaved=True)
    materialized = materialize_final_accounting_recipe(
        source_recipe_path=tmp_path / "source_recipe.json",
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )
    payload = json.loads(materialized.recipe_path.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    amendment = next(
        item for item in payload["service_slots"] if item["source"] == "authorized_amendment"
    )
    amendment["ordinal"] = 2
    _write_self_hashed(materialized.recipe_path, payload)

    with pytest.raises(FinalAccountingError, match="ordinals differ"):
        compile_final_accounting(
            recipe_path=materialized.recipe_path,
            source_root=tmp_path,
            output_root=tmp_path / "tables",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scope", "operator-chosen-scope", "unsupported accounting claim"),
        ("measurement_method", "wall-clock-guess", "unsupported accounting claim"),
        ("elapsed_microseconds", 4_000_000, "elapsed duration is inconsistent"),
        ("sampled_at", T3, "elapsed duration is inconsistent"),
    ],
)
def test_materializer_rejects_wall_time_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    wall = tmp_path / "receipts" / "wall.json"
    payload = json.loads(wall.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload[field] = value
    _write_self_hashed(wall, payload)

    with pytest.raises(FinalAccountingError, match=message):
        materialize_final_accounting_recipe(
            source_recipe_path=tmp_path / "source_recipe.json",
            source_root=tmp_path,
            output_root=tmp_path / "materialized",
        )


def test_wall_time_deduplicates_same_container_start(tmp_path: Path) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    duplicate = tmp_path / "receipts" / "wall-earlier.json"
    _write_self_hashed(
        duplicate,
        {
            "schema_version": "1.0.0",
            "kind": "runpod_container_wall_time_sample",
            "scope": "current_container_pid_1_lifetime",
            "container_pid": 1,
            "container_pid_1_started_at": T0,
            "sampled_at": T3,
            "elapsed_microseconds": 3_000_000,
            "measurement_method": "linux_proc_boot_epoch_plus_pid1_start_ticks",
            "scientific_gpu_accounting": False,
            "billing_time_equivalence_claimed": False,
            "caveat": "earlier sample from the same test container",
        },
    )
    source = tmp_path / "source_recipe.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("manifest_sha256")
    payload["wall_time_receipts"].append(
        {
            "artifact_id": "wall-earlier",
            "relative_path": duplicate.relative_to(tmp_path).as_posix(),
            "self_hash_field": "manifest_sha256",
        }
    )
    _write_self_hashed(source, payload)
    materialized = materialize_final_accounting_recipe(
        source_recipe_path=source,
        source_root=tmp_path,
        output_root=tmp_path / "materialized",
    )
    compiled = compile_final_accounting(
        recipe_path=materialized.recipe_path,
        source_root=tmp_path,
        output_root=tmp_path / "tables",
    )

    assert (
        compiled.receipt["reconciliation"]["runpod_container_wall_time_lower_bound_microseconds"]
        == 5_000_000
    )


def test_cli_compiles_and_verify_mode_is_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe, output = _fixture(tmp_path)
    script_path = Path("scripts/compile_final_accounting.py")
    spec = importlib.util.spec_from_file_location("compile_final_accounting_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.main(
            [
                "--source-root",
                str(tmp_path),
                "--recipe",
                str(recipe),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "compiled"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert (
        module.main(
            [
                "--source-root",
                str(tmp_path),
                "--recipe",
                str(recipe),
                "--output-root",
                str(output),
                "--verify",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_materializer_cli_materializes_and_verify_mode_is_read_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    script_path = Path("scripts/materialize_final_accounting.py")
    spec = importlib.util.spec_from_file_location(
        "materialize_final_accounting_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "materialized"
    arguments = [
        "--source-root",
        str(tmp_path),
        "--source-recipe",
        str(tmp_path / "source_recipe.json"),
        "--output-root",
        str(output),
    ]

    assert module.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "materialized"
    assert result["normalized_receipt_count"] == 7
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert module.main([*arguments, "--verify"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_source_recipe_builder_cli_is_deterministic_and_verify_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manual_recipe, _manual_output = _fixture(tmp_path)
    script_path = Path("scripts/build_final_accounting_source_recipe.py")
    spec = importlib.util.spec_from_file_location(
        "build_final_accounting_source_recipe_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "built-sources-cli"
    arguments = [
        "--source-root",
        str(tmp_path),
        "--accounting-id",
        "accounting-test",
        "--base-call-inventory",
        "configs/gpu_call_inventory.json",
        "--native",
        "phase1_acceptance_result",
        "native/acceptance_manifest.json",
        "--ledger",
        "ledger-final",
        "study",
        "0",
        "-",
        "frozen/study.sqlite",
        "frozen/cas",
        "--wall-time-receipt",
        "receipts/wall.json",
        "--output-root",
        str(output),
    ]

    assert module.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "built"
    assert result["freeze_time"] == T5
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert module.main([*arguments, "--verify"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_materializer_cli_does_not_resolve_away_symlinked_source_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    _manual_recipe, _manual_output = _fixture(real_root)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    script_path = Path("scripts/materialize_final_accounting.py")
    spec = importlib.util.spec_from_file_location(
        "materialize_final_accounting_symlink_script", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert (
        module.main(
            [
                "--source-root",
                str(linked_root),
                "--source-recipe",
                "source_recipe.json",
                "--output-root",
                "materialized",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "invalid"
    assert "symbolic-link ancestor" in result["error"]
    assert not (real_root / "materialized").exists()
