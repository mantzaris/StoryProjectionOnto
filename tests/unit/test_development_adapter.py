from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from story_projection_onto.benchmark_runtime import (
    ModelEligibleWorldArtifact,
    QueryRevealArtifact,
    RuntimeStagingManifest,
    load_staged_neutral_evidence,
    load_staged_world,
)
from story_projection_onto.conditions.c1 import build_c1_preconstruction_request
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    FixedOntologyInput,
    ModelVisibleEvidencePacket,
    ModelVisibleGenericConstructionContext,
    PreconstructionRequest,
    PrequeryBarrier,
    PrequeryPreparationBinding,
    RunOutcome,
)
from story_projection_onto.development_adapter import (
    AdapterDurableState,
    DevelopmentAdapterIntegrityError,
    DevelopmentConstructionConfiguration,
    EncodedSemanticRequest,
    ModelWireAliasManifest,
    ProductionDevelopmentServiceAdapter,
    WireAliasEntry,
    build_development_guided_request,
    decode_development_semantic_request,
    development_request_runtime,
    development_runtime_identifiers,
    encode_development_semantic_request,
    restore_model_output_source_aliases,
)
from story_projection_onto.development_artifacts import (
    DevelopmentCallAuditReceipt,
    DevelopmentRepairDiagnostic,
    DevelopmentRepairProbeInput,
    DevelopmentTreatmentSwitches,
    LogicalCASReference,
    OpaqueJSONReference,
)
from story_projection_onto.development_runtime import (
    CallExecutionEnvelope,
    LiveServiceIdentity,
    ServiceCallResult,
    load_development_call_manifest,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REPOSITORY,
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    TokenizerManifest,
)
from story_projection_onto.query_runtime import AuditedBenchmarkRuntime
from story_projection_onto.store import (
    ArtifactStore,
    BlobStore,
    Compression,
    Ledger,
    ReleaseClass,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = ROOT / "configs/study/development_construction.json"
NO_CONTEXT_BASE_REQUEST = ROOT / "tests/fixtures/phase1/c2_query_request.json"
FORBIDDEN_LIFECYCLE_MEMBERS = {
    "close",
    "emergency_stop",
    "kill",
    "load",
    "restart",
    "shutdown",
    "start",
    "stop",
    "terminate",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _logical(kind: str) -> LogicalCASReference:
    return LogicalCASReference(
        logical_content_hash=_digest(f"{kind}:logical"),
        artifact_hash=_digest(f"{kind}:artifact"),
        object_kind=kind,
    )


def _opaque(kind: str) -> OpaqueJSONReference:
    return OpaqueJSONReference(
        logical_content_hash=_digest(f"{kind}:logical"),
        artifact_hash=_digest(f"{kind}:artifact"),
        object_kind=kind,
    )


def _started_receipt(
    *,
    condition: ConditionName = ConditionName.C2_LLM_QUERY,
    outcome: RunOutcome = RunOutcome.SUCCEEDED,
) -> DevelopmentCallAuditReceipt:
    query_time = condition is not ConditionName.C1_LLM_PRE
    fixed = condition is ConditionName.A_FIXED_SELECT
    succeeded = outcome is RunOutcome.SUCCEEDED
    return DevelopmentCallAuditReceipt(
        receipt_id="development-call-receipt-01",
        ordinal=5 if query_time else 1,
        call_id="dev-c2-u01-q01" if query_time else "dev-c1-u01",
        condition=condition,
        outcome=outcome,
        request_started=True,
        service_identity=_logical("service_identity"),
        call_execution_envelope=_logical("call_execution_envelope"),
        job_id="job-development-call-01",
        attempt_id="attempt-development-call-01",
        model_call_id="model-development-call-01",
        rendered_model_request=_opaque("rendered_model_request"),
        semantic_request=_logical("semantic_request"),
        wire_alias_manifest=_logical("wire_alias_manifest"),
        raw_response_artifact_hash=(
            _digest("raw-response") if succeeded else None
        ),
        validated_generation=(
            _logical("validated_generation") if succeeded else None
        ),
        condition_result=_logical("condition_result") if succeeded else None,
        run_condition_config=_logical("run_condition_config"),
        packing_report=_logical("packing_report"),
        capability_manifest=_logical("capability_manifest"),
        comparison_input_manifest=(
            _logical("comparison_input_manifest") if query_time else None
        ),
        evidence_packet=_logical("evidence_packet") if query_time else None,
        packet_materialization_event_hash=(
            _digest("packet-materialization-event") if query_time else None
        ),
        query_context=_logical("query_context") if query_time else None,
        treatment_switches=_logical("treatment_switches"),
        ledger_validation_ids=("validation-boundary", "validation-grounding"),
        fixed_sealed_inventory=(
            _logical("fixed_sealed_inventory") if fixed else None
        ),
        fixed_forbidden_probe=(
            _logical("fixed_forbidden_probe") if fixed else None
        ),
        fixed_schema_derivation=(
            _logical("fixed_schema_derivation") if fixed else None
        ),
        created_at=datetime(2026, 9, 4, 4, tzinfo=UTC),
    )


class _FakeTokenizer:
    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        conversation: object,
        **kwargs: object,
    ) -> list[int]:
        assert kwargs["enable_thinking"] is False
        assert isinstance(conversation, (list, tuple))
        token_count = sum(len(str(row["content"]).split()) for row in conversation)
        return list(range(token_count + 8))


class _FakeMeteredGenerationService:
    @property
    def actual_allocated_service_seconds(self) -> float:
        return 0.0

    def generate(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("durable recovery must never reissue model inference")


class _SequenceClock:
    def __init__(self, values: Iterator[datetime]) -> None:
        self._values = values

    def __call__(self) -> datetime:
        return next(self._values)


def _tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository=FALLBACK_MODEL_REPOSITORY,
        revision=FALLBACK_MODEL_REVISION,
        tokenizer_class="tests.FakeTokenizer",
        tokenizer_revision=FALLBACK_MODEL_REVISION,
        tokenizer_file_sha256=(("tokenizer.json", _digest("tokenizer")),),
        eos_token_id=7,
        end_of_turn_token_ids=(8,),
        stop_token_ids=(7, 8),
        chat_template_sha256=_digest("template"),
        nonthinking_probe_sha256=_digest("probe"),
        nonthinking_probe_token_count=10,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def _live_service_identity() -> LiveServiceIdentity:
    return LiveServiceIdentity(
        owner_run_id="development-owner",
        service_pid=1234,
        service_start_ticks=5678,
        gpu_session_event_id="gpu-session-development",
        launcher_configuration_hash=_digest("launcher"),
        model_snapshot_hash=_digest("model-snapshot"),
        selected_model_freeze_hash=_digest("model-freeze"),
        source_execution_hash=_digest("source-execution"),
    )


def _development_c1_requests() -> tuple[PreconstructionRequest, ...]:
    call_manifest = load_development_call_manifest(ROOT)
    configuration = DevelopmentConstructionConfiguration.load(CONFIGURATION_PATH)
    tokenizer_manifest = _tokenizer_manifest()
    neutral_root = ROOT / "data/synthetic/condition_inputs/neutral_evidence"
    requests: list[PreconstructionRequest] = []

    for call, neutral_reference in zip(
        call_manifest.calls[:4],
        call_manifest.neutral_evidence_stages,
        strict=True,
    ):
        prequery_root = ROOT / call.prequery_stage.relative_path
        prequery_manifest = RuntimeStagingManifest.model_validate_json(
            (prequery_root / "manifest.json").read_text(encoding="utf-8")
        )
        model_visible = load_staged_world(
            prequery_root / "evidence.json",
            prequery_root,
            prequery_manifest,
        )
        neutral_stage_root = ROOT / neutral_reference.relative_path
        neutral_manifest = RuntimeStagingManifest.model_validate_json(
            (neutral_stage_root / "manifest.json").read_text(encoding="utf-8")
        )
        neutral, _ = load_staged_neutral_evidence(
            neutral_stage_root,
            neutral_root,
            neutral_manifest,
            model_visible,
        )
        runtime = development_runtime_identifiers(
            root=ROOT,
            condition=ConditionName.C1_LLM_PRE,
            tokenizer_manifest=tokenizer_manifest,
            seed=call.vllm_seed,
        )
        requests.append(
            build_c1_preconstruction_request(
                snapshot_hash=neutral.snapshot.content_hash,
                snapshot_sealed_at=neutral.snapshot.sealed_at,
                sealed_horizon=neutral.snapshot.horizon,
                ordered_snapshot_evidence_ids=neutral.snapshot.eligible_evidence_ids,
                evidence=neutral.evidence,
                upper_ontology=configuration.upper_ontology,
                preconstruction_budgets=configuration.preconstruction_budgets,
                runtime=runtime,
                requested_at=neutral.snapshot.sealed_at,
            )
        )
    return tuple(requests)


def _no_context_request(
    tokenizer_manifest: TokenizerManifest,
) -> tuple[ConstructionRequest, dict[str, object]]:
    original = json.loads(NO_CONTEXT_BASE_REQUEST.read_text(encoding="utf-8"))
    for index, evidence in enumerate(original["packet"]["evidence"], start=1):
        text_hash = hashlib.sha256(evidence["text"].encode("utf-8")).hexdigest()
        evidence.update(
            {
                "passage_id": f"development-test-passage-{index}",
                "text_hash": text_hash,
                "confidence": 1.0,
                "provenance": {
                    "provenance_id": f"development-test-provenance-{index}",
                    "evidence_id": evidence["evidence_id"],
                    "extraction_method": "hand-authored synthetic test lineage",
                    "locator": f"development-test:{index}",
                    "source_artifact_hash": text_hash,
                    "confidence": 1.0,
                },
            }
        )
    original_context = dict(original["context"])
    runtime = development_request_runtime(
        root=ROOT,
        condition=ConditionName.A_NO_CONTEXT,
        tokenizer_manifest=tokenizer_manifest,
        seed=1_988_649_846,
    )
    original["condition"] = ConditionName.A_NO_CONTEXT.value
    original["request_id"] = "development-no-context-wire-test"
    original["context"] = ModelVisibleGenericConstructionContext(
        budgets=original_context["budgets"],
    ).model_dump(mode="json", exclude={"content_hash"})
    original["runtime"] = {
        "model_id": FALLBACK_SERVED_MODEL_NAME,
        "model_revision": FALLBACK_MODEL_REVISION,
        "tokenizer_hash": tokenizer_manifest.manifest_sha256,
        "runtime_version": "vllm-0.10.2",
        "prompt_hash": runtime.prompt_hash,
        "output_schema_hash": runtime.output_schema_hash,
        "decoding_config_hash": runtime.decoding_manifest.content_hash,
    }
    return ConstructionRequest.model_validate(original), original_context


def _fixed_request(tokenizer_manifest: TokenizerManifest) -> ConstructionRequest:
    raw = json.loads(
        (ROOT / "tests/fixtures/phase1/fixed_select_request.json").read_text(
            encoding="utf-8"
        )
    )
    for index, evidence in enumerate(raw["packet"]["evidence"], start=1):
        text_hash = hashlib.sha256(evidence["text"].encode("utf-8")).hexdigest()
        evidence.update(
            {
                "passage_id": f"development-test-passage-{index}",
                "text_hash": text_hash,
                "confidence": 1.0,
                "provenance": {
                    "provenance_id": f"development-test-provenance-{index}",
                    "evidence_id": evidence["evidence_id"],
                    "extraction_method": "hand-authored synthetic test lineage",
                    "locator": f"development-test:{index}",
                    "source_artifact_hash": text_hash,
                    "confidence": 1.0,
                },
            }
        )
    fixed = FixedOntologyInput.model_validate(raw["fixed_ontology"])
    packet = ModelVisibleEvidencePacket.model_validate(raw["packet"])
    runtime = development_request_runtime(
        root=ROOT,
        condition=ConditionName.A_FIXED_SELECT,
        tokenizer_manifest=tokenizer_manifest,
        seed=1_988_649_846,
        fixed_ontology=fixed,
        fixed_evidence=packet.evidence,
    )
    raw["runtime"] = {
        "model_id": FALLBACK_SERVED_MODEL_NAME,
        "model_revision": FALLBACK_MODEL_REVISION,
        "tokenizer_hash": tokenizer_manifest.manifest_sha256,
        "runtime_version": "vllm-0.10.2",
        "prompt_hash": runtime.prompt_hash,
        "output_schema_hash": runtime.output_schema_hash,
        "decoding_config_hash": runtime.decoding_manifest.content_hash,
    }
    return ConstructionRequest.model_validate(raw)


def _repair_request_and_input(
    tokenizer_manifest: TokenizerManifest,
) -> tuple[ConstructionRequest, DevelopmentRepairProbeInput]:
    raw = json.loads(NO_CONTEXT_BASE_REQUEST.read_text(encoding="utf-8"))
    for index, evidence in enumerate(raw["packet"]["evidence"], start=1):
        text_hash = hashlib.sha256(evidence["text"].encode("utf-8")).hexdigest()
        evidence.update(
            {
                "passage_id": f"development-test-passage-{index}",
                "text_hash": text_hash,
                "confidence": 1.0,
                "provenance": {
                    "provenance_id": f"development-test-provenance-{index}",
                    "evidence_id": evidence["evidence_id"],
                    "extraction_method": "hand-authored synthetic test lineage",
                    "locator": f"development-test:{index}",
                    "source_artifact_hash": text_hash,
                    "confidence": 1.0,
                },
            }
        )
    runtime = development_request_runtime(
        root=ROOT,
        condition=ConditionName.C2_LLM_QUERY,
        tokenizer_manifest=tokenizer_manifest,
        seed=1_988_649_846,
        repair=True,
    )
    raw["request_id"] = "development-repair-wire-test"
    raw["runtime"] = {
        "model_id": FALLBACK_SERVED_MODEL_NAME,
        "model_revision": FALLBACK_MODEL_REVISION,
        "tokenizer_hash": tokenizer_manifest.manifest_sha256,
        "runtime_version": "vllm-0.10.2",
        "prompt_hash": runtime.prompt_hash,
        "output_schema_hash": runtime.output_schema_hash,
        "decoding_config_hash": runtime.decoding_manifest.content_hash,
    }
    request = ConstructionRequest.model_validate(raw)
    invalid = json.loads(
        (ROOT / "tests/fixtures/phase1/c2_query_output.json").read_text(
            encoding="utf-8"
        )
    )
    entity_id = invalid["instance_graph"]["entities"][0]["entity_id"]
    invalid["instance_graph"]["entities"][0]["contextual_type_id"] = (
        "unknown-development-type"
    )
    repair_input = DevelopmentRepairProbeInput(
        parent_call_id="dev-c2-u04-q02",
        parent_semantic_request_hash=_digest("parent-semantic"),
        repair_semantic_request_hash=request.content_hash,
        parent_raw_output_hash=_digest("parent-raw"),
        diagnostic_fixture_id="development-schema-reference-repair-probe-v1",
        fault_injection=(
            "deterministic_unknown_reference_after_preserving_raw_parent"
        ),
        invalid_draft=invalid,
        diagnostics=(
            DevelopmentRepairDiagnostic(
                code="unknown_contextual_type_reference",
                path=(
                    f"instance_graph.entities.{entity_id}.contextual_type_id"
                ),
                message=(
                    "Replace only the unknown contextual_type_id with an identifier "
                    "declared in local_schema.contextual_types."
                ),
            ),
        ),
        created_at=datetime(2026, 9, 4, 4, tzinfo=UTC),
    )
    return request, repair_input


def test_call_audit_receipt_round_trip_is_canonical_and_hash_stable() -> None:
    receipt = _started_receipt()
    restored = DevelopmentCallAuditReceipt.model_validate_json(
        receipt.to_canonical_json()
    )

    assert restored == receipt
    assert restored.content_hash == receipt.content_hash
    assert restored.to_canonical_json() == receipt.to_canonical_json()


def test_call_audit_receipt_rejects_incomplete_or_misclassified_lineage() -> None:
    successful = _started_receipt()
    with pytest.raises(ValidationError, match="raw and validated outputs"):
        successful.model_copy(
            update={"raw_response_artifact_hash": None},
        ).__class__.model_validate(
            successful.model_dump(
                mode="python",
                exclude={"content_hash", "raw_response_artifact_hash"},
            )
        )

    c1 = _started_receipt(condition=ConditionName.C1_LLM_PRE)
    with pytest.raises(ValidationError, match="cannot contain query artifacts"):
        DevelopmentCallAuditReceipt.model_validate(
            {
                **c1.model_dump(mode="python", exclude={"content_hash"}),
                "query_context": _logical("forbidden_query_context"),
            }
        )

    failed_fixed = _started_receipt(
        condition=ConditionName.A_FIXED_SELECT,
        outcome=RunOutcome.TIMED_OUT,
    )
    assert failed_fixed.raw_response_artifact_hash is None
    assert failed_fixed.validated_generation is None
    assert failed_fixed.fixed_sealed_inventory is not None
    assert failed_fixed.fixed_forbidden_probe is not None
    assert failed_fixed.fixed_schema_derivation is not None

    with pytest.raises(ValidationError, match="only FixedSelect"):
        DevelopmentCallAuditReceipt.model_validate(
            {
                **successful.model_dump(mode="python", exclude={"content_hash"}),
                "fixed_sealed_inventory": _logical("forbidden_fixed_inventory"),
            }
        )

    with pytest.raises(ValidationError, match="packet/context/fairness"):
        DevelopmentCallAuditReceipt.model_validate(
            successful.model_dump(
                mode="python",
                exclude={"content_hash", "packet_materialization_event_hash"},
            )
        )

    with pytest.raises(ValidationError, match="cannot contain query artifacts"):
        DevelopmentCallAuditReceipt.model_validate(
            {
                **c1.model_dump(mode="python", exclude={"content_hash"}),
                "packet_materialization_event_hash": _digest(
                    "forbidden-packet-materialization"
                ),
            }
        )


def test_treatment_switch_hash_changes_for_exact_one_switch() -> None:
    baseline = DevelopmentTreatmentSwitches()
    no_context = DevelopmentTreatmentSwitches(context_payload_mode="generic")

    assert baseline.content_hash != no_context.content_hash
    changed = {
        key
        for key, value in baseline.model_dump(mode="json").items()
        if no_context.model_dump(mode="json")[key] != value
    }
    assert changed == {"context_payload_mode", "content_hash"}


def test_construction_configuration_is_typed_and_binds_exact_source_bytes(
    tmp_path: Path,
) -> None:
    baseline = DevelopmentConstructionConfiguration.load(CONFIGURATION_PATH)
    raw = CONFIGURATION_PATH.read_bytes()

    copied_path = tmp_path / "copied-development-construction.json"
    copied_path.write_bytes(raw)
    copied = DevelopmentConstructionConfiguration.load(copied_path)
    assert copied == baseline
    assert copied.source_file_sha256 == hashlib.sha256(raw).hexdigest()

    mutated_payload = json.loads(raw)
    mutated_payload["preconstruction_budgets"]["node_budget"] += 1
    mutated_path = tmp_path / "mutated-development-construction.json"
    mutated_path.write_text(json.dumps(mutated_payload), encoding="utf-8")
    mutated = DevelopmentConstructionConfiguration.load(mutated_path)
    assert mutated.source_file_sha256 != baseline.source_file_sha256
    assert mutated.content_hash != baseline.content_hash
    with pytest.raises(DevelopmentAdapterIntegrityError, match="bytes changed"):
        DevelopmentConstructionConfiguration.load(
            mutated_path,
            expected_sha256=baseline.source_file_sha256,
        )

    invalid_payload = json.loads(raw)
    invalid_payload["first_pass_input_tokens"] = 10_239
    invalid_path = tmp_path / "invalid-development-construction.json"
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(DevelopmentAdapterIntegrityError, match="failed validation"):
        DevelopmentConstructionConfiguration.load(invalid_path)


def test_no_context_guided_wire_contains_no_original_query_semantics() -> None:
    tokenizer_manifest = _tokenizer_manifest()
    semantic_request, original_context = _no_context_request(tokenizer_manifest)

    guided = build_development_guided_request(
        root=ROOT,
        call_id="dev-ablation-no-context-u01-q01",
        semantic_request=semantic_request,
        tokenizer=_FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        seed=1_988_649_846,
    )
    wire = json.dumps(guided.wire_payload(), ensure_ascii=False, sort_keys=True)
    user_payload = json.loads(guided.messages[-1].content)

    assert set(user_payload["query_context"]) == {"budgets", "request"}
    forbidden_query_context_fields = {
        "wording",
        "lens",
        "target",
        "story_scope",
        "spoiler_horizon",
        "viewpoint",
        "abstraction",
    }
    assert forbidden_query_context_fields.isdisjoint(user_payload["query_context"])
    for original_value in (
        original_context["wording"],
        original_context["lens"],
        original_context["target"],
    ):
        assert str(original_value) not in wire


def test_all_four_staged_c1_requests_have_exact_typed_codec_round_trip() -> None:
    requests = _development_c1_requests()

    assert len(requests) == 4
    assert len({request.snapshot_hash for request in requests}) == 4
    for request in requests:
        encoded = encode_development_semantic_request(request)
        decoded = decode_development_semantic_request(encoded)

        assert isinstance(decoded, PreconstructionRequest)
        assert decoded == request
        assert decoded.to_canonical_json() == request.to_canonical_json()
        assert encoded.alias_manifest.semantic_request_hash == request.content_hash


def test_codec_rejects_semantic_mutation_and_unregistered_or_duplicate_aliases() -> None:
    request = _development_c1_requests()[0]
    encoded = encode_development_semantic_request(request)

    mutated_sections = copy.deepcopy(dict(encoded.sections))
    evidence_rows = mutated_sections["evidence_snapshot"]
    assert isinstance(evidence_rows, list)
    first_row = evidence_rows[0]
    assert isinstance(first_row, list)
    first_row[1] = f"{first_row[1]} [tampered]"
    with pytest.raises(
        DevelopmentAdapterIntegrityError,
        match="compact evidence failed typed lossless decoding",
    ):
        decode_development_semantic_request(
            EncodedSemanticRequest(
                sections=mutated_sections,
                alias_manifest=encoded.alias_manifest,
            )
        )

    unknown_alias_sections = copy.deepcopy(dict(encoded.sections))
    unknown_rows = unknown_alias_sections["evidence_snapshot"]
    assert isinstance(unknown_rows, list)
    assert isinstance(unknown_rows[0], list)
    unknown_rows[0][0] = "srcE9999"
    with pytest.raises(DevelopmentAdapterIntegrityError, match="unregistered source alias"):
        decode_development_semantic_request(
            EncodedSemanticRequest(
                sections=unknown_alias_sections,
                alias_manifest=encoded.alias_manifest,
            )
        )

    entries = list(encoded.alias_manifest.entries)
    duplicate_payload = entries[1].model_dump(mode="python", exclude={"content_hash"})
    duplicate_payload["alias"] = entries[0].alias
    entries[1] = WireAliasEntry.model_validate(duplicate_payload)
    with pytest.raises(ValidationError, match="bijective"):
        ModelWireAliasManifest(
            semantic_request_hash=request.content_hash,
            entries=tuple(entries),
        )


def test_codec_documents_surface_hash_derivation_and_shared_src_e_namespace() -> None:
    request = _development_c1_requests()[0]
    encoded = encode_development_semantic_request(request)
    wire_manifest = encoded.sections["wire_encoding"]
    assert isinstance(wire_manifest, dict)
    omitted = wire_manifest["omitted_derivable_fields"]
    assert isinstance(omitted, dict)
    assert omitted["surface_hash"] == (
        "sha256(UTF-8 surface) is recomputed by the typed decoder"
    )

    evidence_rows = encoded.sections["evidence_snapshot"]
    assert isinstance(evidence_rows, list)
    compact_mentions = [
        mention
        for row in evidence_rows
        for mention in row[3]
    ]
    assert compact_mentions
    assert all(len(mention) == 7 for mention in compact_mentions)
    decoded = decode_development_semantic_request(encoded)
    decoded_mentions = [
        mention
        for evidence in decoded.evidence
        for mention in evidence.mention_candidates
    ]
    assert all(
        mention.surface_hash
        == hashlib.sha256(mention.surface.encode("utf-8")).hexdigest()
        for mention in decoded_mentions
    )

    shared_entries = [
        entry
        for entry in encoded.alias_manifest.entries
        if entry.alias.startswith("srcE")
    ]
    assert {entry.source_kind for entry in shared_entries} == {"evidence", "event"}
    assert len({entry.alias for entry in shared_entries}) == len(shared_entries)
    assert all(
        entry.source_kind in {"evidence", "event"}
        for entry in shared_entries
    )
    shared_description = wire_manifest["shared_alias_namespace"]
    assert isinstance(shared_description, dict)
    assert shared_description["members"] == ["evidence", "event"]
    assert "source_kind" in str(shared_description["disambiguation"])


def test_output_alias_restoration_is_field_aware_and_preserves_authored_text() -> None:
    encoded = encode_development_semantic_request(_development_c1_requests()[0])
    evidence_entry = next(
        item
        for item in encoded.alias_manifest.entries
        if item.source_kind == "evidence"
    )
    mention_entry = next(
        item
        for item in encoded.alias_manifest.entries
        if item.source_kind == "mention"
    )
    payload = {
        "label": evidence_entry.alias,
        "description": f"literal {mention_entry.alias} remains authored prose",
        "rationale": evidence_entry.alias,
        "why_matters": mention_entry.alias,
        "evidence_ids": [evidence_entry.alias],
        "supported_mention_candidate_ids": [mention_entry.alias],
    }

    restored = restore_model_output_source_aliases(payload, encoded.alias_manifest)

    assert restored["label"] == evidence_entry.alias
    assert restored["description"] == payload["description"]
    assert restored["rationale"] == evidence_entry.alias
    assert restored["why_matters"] == mention_entry.alias
    assert restored["evidence_ids"] == [evidence_entry.source_id]
    assert restored["supported_mention_candidate_ids"] == [mention_entry.source_id]
    with pytest.raises(
        DevelopmentAdapterIntegrityError,
        match="reserved source alias",
    ):
        restore_model_output_source_aliases(
            {"entity_id": mention_entry.alias},
            encoded.alias_manifest,
        )


def test_fixed_select_guided_schema_binds_complete_sealed_inventory_and_aliases() -> None:
    tokenizer_manifest = _tokenizer_manifest()
    semantic_request = _fixed_request(tokenizer_manifest)
    guided = build_development_guided_request(
        root=ROOT,
        call_id="dev-fixed-select-schema-test",
        semantic_request=semantic_request,
        tokenizer=_FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        seed=1_988_649_846,
    )
    definitions = guided.output_schema["$defs"]

    assert definitions["ConstructionOperator"]["enum"] == [
        "selection",
        "compression",
        "supported_description",
    ]
    decision = definitions["OntologyDecision"]["properties"]
    assert decision["created_object_ids"]["maxItems"] == 0
    assert decision["removed_object_ids"]["maxItems"] == 0
    assert set(decision["input_object_ids"]["items"]["enum"]) == set(
        semantic_request.fixed_ontology.construction_seal.sealed_object_ids
    )
    assert set(definitions["Entity"]["properties"]["entity_id"]["enum"]) == {
        item.entity_id for item in semantic_request.fixed_ontology.instance_graph.entities
    }
    assert "novel-entity" not in definitions["Entity"]["properties"]["entity_id"]["enum"]

    encoded = encode_development_semantic_request(semantic_request)
    expected_evidence_aliases = {
        encoded.alias_manifest.source_to_alias[item]
        for item in semantic_request.packet.ordered_evidence_ids
    }
    evidence_schema = definitions["QualifiedAssertion"]["properties"][
        "evidence_ids"
    ]["items"]
    assert set(evidence_schema["enum"]) == expected_evidence_aliases
    assert not set(semantic_request.packet.ordered_evidence_ids).intersection(
        evidence_schema["enum"]
    )


def test_repair_guided_request_binds_parent_fault_and_fact_free_diagnostic() -> None:
    tokenizer_manifest = _tokenizer_manifest()
    semantic_request, repair_input = _repair_request_and_input(tokenizer_manifest)

    guided = build_development_guided_request(
        root=ROOT,
        call_id="dev-repair-probe-u04-q02",
        semantic_request=semantic_request,
        tokenizer=_FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest,
        seed=1_988_649_846,
        repair=True,
        repair_probe_input=repair_input,
    )
    payload = json.loads(guided.messages[-1].content)

    assert payload["invalid_draft"] == repair_input.invalid_draft
    assert payload["validation_diagnostics"] == [
        item.model_dump(mode="json", exclude={"schema_version", "content_hash"})
        for item in repair_input.diagnostics
    ]
    assert guided.messages[0].content == (
        ROOT / "prompts/repair/prompt_v1.md"
    ).read_text(encoding="utf-8")
    assert guided.decoding.maximum_input_tokens == 10_752
    assert guided.decoding.maximum_output_tokens == 1_536
    assert guided.packing.required_section_names[-2:] == (
        "invalid_draft",
        "validation_diagnostics",
    )
    assert guided.packing.complete_evidence_packet is True
    with pytest.raises(
        DevelopmentAdapterIntegrityError,
        match="complete deterministic repair-probe input",
    ):
        build_development_guided_request(
            root=ROOT,
            call_id="dev-repair-probe-u04-q02",
            semantic_request=semantic_request,
            tokenizer=_FakeTokenizer(),
            tokenizer_manifest=tokenizer_manifest,
            seed=1_988_649_846,
            repair=True,
        )


def test_production_adapter_public_surface_has_no_lifecycle_authority() -> None:
    exposed = set(dir(ProductionDevelopmentServiceAdapter))

    assert FORBIDDEN_LIFECYCLE_MEMBERS.isdisjoint(exposed)
    assert "service" not in exposed, (
        "the lifecycle-capable raw service must remain private behind the narrow adapter"
    )


@pytest.mark.parametrize("query_mutation", ["tampered", "unavailable"])
def test_query_opening_restart_rehydrates_cas_without_rereading_query_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query_mutation: str,
) -> None:
    call = load_development_call_manifest(ROOT).calls[4]
    stage = call.query_stage
    assert stage is not None
    repository = tmp_path / "repository"
    staged_root = repository / stage.relative_path
    staged_root.parent.mkdir(parents=True)
    shutil.copytree(ROOT / stage.relative_path, staged_root)
    stage_manifest = RuntimeStagingManifest.model_validate_json(
        (staged_root / "manifest.json").read_text(encoding="utf-8")
    )
    evidence = ModelEligibleWorldArtifact.model_validate_json(
        (staged_root / "evidence.json").read_text(encoding="utf-8")
    )
    reveal = QueryRevealArtifact.model_validate_json(
        (staged_root / "query.json").read_text(encoding="utf-8")
    )
    preparation_completed_at = reveal.revealed_at - timedelta(seconds=3)
    barrier_sealed_at = reveal.revealed_at - timedelta(seconds=2)
    barrier_persisted_at = reveal.revealed_at - timedelta(seconds=1)
    accessed_at = reveal.revealed_at + timedelta(seconds=1)
    execution_manifest_hash = _digest("query-restart-execution-manifest")
    barrier = PrequeryBarrier(
        barrier_id="development-query-restart-barrier",
        execution_id="development-query-restart",
        execution_manifest_hash=execution_manifest_hash,
        neutral_evidence_artifact_hashes=(evidence.content_hash,),
        preparation_bindings=(
            PrequeryPreparationBinding(
                unit_id=call.unit_id,
                condition=ConditionName.C0_CLASSICAL_PRE,
                snapshot_hash=evidence.snapshot.content_hash,
                preparation_hash=_digest("query-restart-preparation"),
                lineage_artifact_hash=_digest("query-restart-lineage"),
                completed_at=preparation_completed_at,
            ),
        ),
        sealed_at=barrier_sealed_at,
    )

    ledger = Ledger(tmp_path / "query-restart.sqlite3")
    blobs = BlobStore(tmp_path / "query-restart-blobs", compression=Compression.GZIP)
    artifacts = ArtifactStore(blobs, ledger)
    first_runtime = AuditedBenchmarkRuntime(
        ledger=ledger,
        blobs=blobs,
        clock=_SequenceClock(iter((barrier_persisted_at, accessed_at))),
    )
    adapter_arguments = {
        "live_identity": _live_service_identity(),
        "_service": _FakeMeteredGenerationService(),
        "artifacts": artifacts,
        "state_path": tmp_path / "query-restart-adapter-state.json",
        "execution_manifest_hash": execution_manifest_hash,
        "repository_root": repository,
        "prequery_evidence_by_hash": {stage.evidence_artifact_hash: evidence},
    }
    try:
        first_adapter = ProductionDevelopmentServiceAdapter(
            **adapter_arguments,
            query_runtime=first_runtime,
        )
        committed = first_adapter.open_query(stage, barrier)
        assert committed.stage_manifest_hash == stage_manifest.content_hash
        assert (
            ledger.get_query_access_by_event_id(committed.access_event_id).access_event_hash
            == committed.content_hash
        )

        query_path = staged_root / "query.json"
        if query_mutation == "tampered":
            query_path.write_bytes(b'{"tampered":true}\n')
        else:
            query_path.rename(staged_root / "query.unavailable")

        physical_query_reads = 0
        original_read_bytes = Path.read_bytes

        def reject_physical_query_read(path: Path) -> bytes:
            nonlocal physical_query_reads
            if path.name == "query.json":
                physical_query_reads += 1
                raise AssertionError("committed query must be rehydrated from CAS")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", reject_physical_query_read)
        recovery_runtime = AuditedBenchmarkRuntime(
            ledger=ledger,
            blobs=blobs,
            clock=lambda: barrier_persisted_at,
        )
        recovered_adapter = ProductionDevelopmentServiceAdapter(
            **adapter_arguments,
            query_runtime=recovery_runtime,
        )
        recovered = recovered_adapter.open_query(stage, barrier)
        assert recovered == committed
        assert physical_query_reads == 0

        changed_stage_payload = stage.model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        changed_stage_payload["query_artifact_hash"] = _digest(
            "tampered-stage-query-artifact"
        )
        changed_stage = type(stage).model_validate(changed_stage_payload)
        changed_stage_adapter = ProductionDevelopmentServiceAdapter(
            **adapter_arguments,
            query_runtime=recovery_runtime,
        )
        with pytest.raises(
            DevelopmentAdapterIntegrityError,
            match="persisted query access differs",
        ):
            changed_stage_adapter.open_query(changed_stage, barrier)

        changed_barrier_payload = barrier.model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        changed_barrier_payload["barrier_id"] = (
            "development-query-restart-barrier-tampered"
        )
        changed_barrier = PrequeryBarrier.model_validate(changed_barrier_payload)
        changed_barrier_adapter = ProductionDevelopmentServiceAdapter(
            **adapter_arguments,
            query_runtime=recovery_runtime,
        )
        with pytest.raises(
            DevelopmentAdapterIntegrityError,
            match="persisted query access differs",
        ):
            changed_barrier_adapter.open_query(stage, changed_barrier)
        assert physical_query_reads == 0
    finally:
        ledger.close()


def test_durable_receipt_index_recovers_without_reissuing_generation(
    tmp_path: Path,
) -> None:
    manifest = load_development_call_manifest(ROOT)
    call = manifest.calls[4]
    identity = _live_service_identity()
    execution_manifest_hash = _digest("execution-manifest")
    run_configuration_hash = _digest("run-configuration")
    completed_at = datetime(2026, 9, 4, 5, tzinfo=UTC)
    envelope = CallExecutionEnvelope(
        execution_id="development-execution",
        execution_manifest_hash=execution_manifest_hash,
        call_manifest_hash=manifest.content_hash,
        prequery_inputs_hash=_digest("prequery-inputs"),
        expected_run_condition_config_hash=run_configuration_hash,
        preconstruction_barrier_hash=_digest("preconstruction-barrier"),
        preconstruction_barrier_recorded_at=completed_at,
        unit_prequery_binding_hash=_digest("unit-prequery-binding"),
        service_identity_hash=identity.content_hash,
    )

    ledger = Ledger(tmp_path / "ledger.sqlite3")
    blobs = BlobStore(tmp_path / "blobs", compression=Compression.GZIP)
    artifacts = ArtifactStore(blobs, ledger)
    try:
        identity_artifact = artifacts.put_bytes(
            (identity.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.service-identity+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=completed_at,
        )
        envelope_artifact = artifacts.put_bytes(
            (envelope.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.call-envelope+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=completed_at,
        )
        receipt_payload = _started_receipt().model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        receipt_payload.update(
            {
                "call_id": call.call_id,
                "ordinal": call.ordinal,
                "condition": call.condition,
                "service_identity": LogicalCASReference(
                    logical_content_hash=identity.content_hash,
                    artifact_hash=identity_artifact.content_hash,
                    object_kind="service_identity",
                ),
                "call_execution_envelope": LogicalCASReference(
                    logical_content_hash=envelope.content_hash,
                    artifact_hash=envelope_artifact.content_hash,
                    object_kind="call_execution_envelope",
                ),
            }
        )
        receipt = DevelopmentCallAuditReceipt.model_validate(receipt_payload)
        receipt_artifact = artifacts.put_bytes(
            (receipt.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.development-call-receipt+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=completed_at,
        )
        result = ServiceCallResult(
            call_id=call.call_id,
            outcome=RunOutcome.SUCCEEDED,
            request_started=True,
            request_hash=_digest("rendered_model_request:logical"),
            response_artifact_hash=_digest("raw-response"),
            validated_generation_hash=_digest("validated_generation:logical"),
            validation_record_hash=_digest("validation-record"),
            condition_attempt_hash=_digest("condition_result:logical"),
            ledger_receipt_hash=receipt_artifact.content_hash,
            gpu_event_id="gpu-development-call-05",
            service_identity_hash=identity.content_hash,
            run_condition_config_hash=run_configuration_hash,
            allocated_gpu_seconds=2.5,
            prompt_tokens=100,
            completion_tokens=20,
            completed_at=completed_at,
        )
        result_artifact = artifacts.put_bytes(
            (result.to_canonical_json() + "\n").encode("utf-8"),
            media_type="application/vnd.story-projection.service-call-result+json",
            release_class=ReleaseClass.PUBLIC,
            created_at=completed_at,
        )
        state = AdapterDurableState(
            execution_manifest_hash=execution_manifest_hash,
            service_identity_hash=identity.content_hash,
            completed_receipts={call.call_id: receipt_artifact.content_hash},
            completed_results={call.call_id: result_artifact.content_hash},
            updated_at=completed_at,
        )
        state_path = tmp_path / "adapter-state.json"
        state_path.write_text(state.to_canonical_json() + "\n", encoding="utf-8")

        adapter = ProductionDevelopmentServiceAdapter(
            live_identity=identity,
            _service=_FakeMeteredGenerationService(),
            artifacts=artifacts,
            state_path=state_path,
            execution_manifest_hash=execution_manifest_hash,
        )
        assert adapter.recover_call(call, envelope) == result
        assert adapter.recover_call(manifest.calls[5], envelope) is None

        changed_envelope_payload = envelope.model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        changed_envelope_payload["execution_manifest_hash"] = _digest(
            "another-execution"
        )
        changed_envelope = CallExecutionEnvelope.model_validate(
            changed_envelope_payload
        )
        with pytest.raises(
            DevelopmentAdapterIntegrityError,
            match="changed execution",
        ):
            adapter.recover_call(call, changed_envelope)

        lineage_change_payload = envelope.model_dump(
            mode="python",
            exclude={"content_hash"},
        )
        lineage_change_payload["prequery_inputs_hash"] = _digest(
            "changed-prequery-inputs"
        )
        lineage_change = CallExecutionEnvelope.model_validate(
            lineage_change_payload
        )
        with pytest.raises(
            DevelopmentAdapterIntegrityError,
            match="bound inputs changed",
        ):
            adapter.recover_call(call, lineage_change)
    finally:
        ledger.close()


def test_durable_state_rejects_mismatched_receipt_and_result_indexes() -> None:
    with pytest.raises(ValidationError, match="indexes differ"):
        AdapterDurableState(
            execution_manifest_hash=_digest("execution-manifest"),
            service_identity_hash=_digest("service-identity"),
            completed_receipts={"call-01": _digest("receipt")},
            completed_results={},
            updated_at=datetime(2026, 9, 4, 5, tzinfo=UTC),
        )
