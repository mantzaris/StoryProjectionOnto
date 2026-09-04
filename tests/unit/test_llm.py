from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    FIXED_SELECT_ALLOWED_OPERATORS,
    ConditionName,
    ConstructionOperator,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    DecodingManifest,
    FixedSelectCapabilityError,
    FixedSelectOutputAudit,
    GPURequestMetadata,
    PackingReport,
    PackingSection,
    ProposedOperation,
    RepairLineageMetadata,
    RequestTimeoutClass,
    SealedOntologyInventory,
    SeedManifest,
    SemanticFingerprint,
    allowed_capabilities_for,
    enforce_fixed_select_output,
    make_gpu_request_metadata,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
GIT_REVISION = "d" * 40
TOKENIZER_REVISION = "tokenizer@immutable-revision"
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def decoding(seed: int, *, repair: bool = False) -> DecodingManifest:
    constructor = DecodingManifest.repair if repair else DecodingManifest.first_pass
    return constructor(
        seed=seed,
        eos_token_id=1,
        end_of_turn_token_ids=(2,),
        chat_template_hash=HASH_A,
        output_schema_hash=HASH_B,
        structured_decoder="vllm-json-schema@1",
        tokenizer_revision=TOKENIZER_REVISION,
    )


def section(name: str, tokens: int = 10, hash_value: str = HASH_A) -> PackingSection:
    return PackingSection(
        name=name,
        section_content_hash=hash_value,
        token_count=tokens,
    )


def packing(condition: ConditionName) -> PackingReport:
    common = [
        section("system_prompt"),
        section("output_schema"),
        section("upper_ontology"),
    ]
    if condition is ConditionName.C1_LLM_PRE:
        sections = [*common, section("evidence_snapshot")]
        return PackingReport.build(
            condition=condition,
            tokenizer_revision=TOKENIZER_REVISION,
            maximum_model_tokens=12_288,
            maximum_input_tokens=10_240,
            reserved_output_tokens=2_048,
            sections=sections,
            required_section_names=[item.name for item in sections],
            complete_evidence_snapshot=True,
        )
    sections = [*common, section("query_context"), section("evidence_packet")]
    if condition is ConditionName.A_FIXED_SELECT:
        sections.append(section("sealed_ontology"))
    return PackingReport.build(
        condition=condition,
        tokenizer_revision=TOKENIZER_REVISION,
        maximum_model_tokens=12_288,
        maximum_input_tokens=10_240,
        reserved_output_tokens=2_048,
        sections=sections,
        required_section_names=[item.name for item in sections],
        complete_evidence_packet=True,
        complete_sealed_ontology=True if condition is ConditionName.A_FIXED_SELECT else None,
    )


def test_decoding_manifest_records_every_registered_decision() -> None:
    manifest = decoding(41)
    payload = manifest.model_dump(mode="json")

    assert {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "n",
        "best_of",
        "beam_search",
        "thinking_mode",
        "ignore_eos",
        "maximum_model_tokens",
        "maximum_input_tokens",
        "maximum_output_tokens",
        "seed",
        "eos_token_id",
        "end_of_turn_token_ids",
        "stop_token_ids",
        "chat_template_hash",
        "output_schema_hash",
        "structured_decoder",
        "tokenizer_revision",
    }.issubset(payload)
    assert manifest.stop_token_ids == (1, 2)
    assert manifest.maximum_input_tokens == 10_240
    assert manifest.maximum_output_tokens == 2_048
    assert len(manifest.content_hash) == 64


def test_decoding_comparison_family_excludes_only_schema_and_paired_seed() -> None:
    first = decoding(41)
    another_seed = decoding(42)
    another_schema = DecodingManifest.first_pass(
        seed=41,
        eos_token_id=1,
        end_of_turn_token_ids=(2,),
        chat_template_hash=HASH_A,
        output_schema_hash=HASH_C,
        structured_decoder="vllm-json-schema@1",
        tokenizer_revision=TOKENIZER_REVISION,
    )

    assert len({first.content_hash, another_seed.content_hash, another_schema.content_hash}) == 3
    assert first.comparison_family_hash == another_seed.comparison_family_hash
    assert first.comparison_family_hash == another_schema.comparison_family_hash

    repair = decoding(41, repair=True)
    assert first.comparison_family_hash != repair.comparison_family_hash


def test_repair_decoding_changes_only_declared_pass_and_caps() -> None:
    first = decoding(41)
    repair = decoding(41, repair=True)
    excluded = {
        "content_hash",
        "decoding_pass",
        "maximum_input_tokens",
        "maximum_output_tokens",
    }

    assert first.model_dump(exclude=excluded) == repair.model_dump(exclude=excluded)
    assert (repair.maximum_input_tokens, repair.maximum_output_tokens) == (10_752, 1_536)


def test_decoding_rejects_unregistered_or_incomplete_values() -> None:
    values = decoding(41).model_dump(exclude={"content_hash"})
    values["top_p"] = 0.9
    with pytest.raises(ValidationError, match="top_p must remain"):
        DecodingManifest(**values)

    values = decoding(41).model_dump(exclude={"content_hash"})
    values["stop_token_ids"] = (1,)
    with pytest.raises(ValidationError, match="end-of-turn"):
        DecodingManifest(**values)


def test_seed_manifest_is_deterministic_domain_separated_and_self_verifying() -> None:
    first = SeedManifest.from_root(8675309, seed_block=1, scope=("world-01",))
    repeated = SeedManifest.from_root(8675309, seed_block=1, scope=("world-01",))
    other_block = SeedManifest.from_root(8675309, seed_block=2, scope=("world-01",))

    assert first == repeated
    assert first.content_hash == repeated.content_hash
    assert first.llm_seed != first.layout_seed
    assert first.llm_seed != other_block.llm_seed
    tampered = first.model_dump(exclude={"content_hash"})
    tampered["llm_seed"] += 1
    with pytest.raises(ValidationError, match="llm_seed"):
        SeedManifest(**tampered)


def test_condition_capabilities_are_exact_and_fixed_select_is_selection_only() -> None:
    c1 = CapabilityManifest.for_condition(ConditionName.C1_LLM_PRE)
    c2 = CapabilityManifest.for_condition(ConditionName.C2_LLM_QUERY)
    fixed = CapabilityManifest.for_condition(ConditionName.A_FIXED_SELECT)

    assert set(fixed.allowed) == FIXED_SELECT_ALLOWED_OPERATORS
    assert set(fixed.allowed).isdisjoint(CONSTRUCTIVE_OPERATORS)
    assert ConstructionOperator.MERGE in c1.allowed
    assert ConstructionOperator.MERGE in c2.allowed
    assert allowed_capabilities_for(ConditionName.A_FIXED_SELECT) == set(fixed.allowed)

    forged = fixed.model_dump(exclude={"content_hash"})
    forged["allowed"] = (*fixed.allowed, ConstructionOperator.MERGE)
    forged["forbidden"] = tuple(
        item for item in fixed.forbidden if item is not ConstructionOperator.MERGE
    )
    with pytest.raises(ValidationError, match="allowlist differs"):
        CapabilityManifest(**forged)


def test_packing_report_proves_complete_fixed_input_without_truncation() -> None:
    report = packing(ConditionName.A_FIXED_SELECT)

    assert report.complete_evidence_packet is True
    assert report.complete_sealed_ontology is True
    assert report.truncation_applied is False
    assert report.omitted_section_names == ()
    assert report.input_token_count == sum(item.token_count for item in report.sections)


def test_packing_report_rejects_missing_sections_and_oversize_content() -> None:
    valid = packing(ConditionName.A_FIXED_SELECT)
    values = valid.model_dump(exclude={"content_hash"})
    values["sections"] = tuple(item for item in valid.sections if item.name != "sealed_ontology")
    values["input_token_count"] -= 10
    with pytest.raises(ValidationError, match="missing"):
        PackingReport(**values)

    values = packing(ConditionName.C2_LLM_QUERY).model_dump(exclude={"content_hash"})
    values["maximum_input_tokens"] = values["input_token_count"] - 1
    with pytest.raises(ValidationError, match="truncation is forbidden"):
        PackingReport(**values)


def test_c1_packing_is_query_blind_and_c2_cannot_pack_hidden_ontology() -> None:
    c1 = packing(ConditionName.C1_LLM_PRE)
    values = c1.model_dump(exclude={"content_hash"})
    values["sections"] = (*c1.sections, section("query_context"))
    values["input_token_count"] += 10
    with pytest.raises(ValidationError, match="query-blind"):
        PackingReport(**values)

    c2 = packing(ConditionName.C2_LLM_QUERY)
    values = c2.model_dump(exclude={"content_hash"})
    values["complete_sealed_ontology"] = True
    with pytest.raises(ValidationError, match="must not receive"):
        PackingReport(**values)


def test_fixed_select_accepts_only_unchanged_sealed_objects() -> None:
    original = SemanticFingerprint(
        semantic_id="assertion-1",
        semantic_kind="assertion",
        semantic_hash=HASH_A,
    )
    sealed = SealedOntologyInventory(
        seal_hash=HASH_B,
        seed_block=1,
        objects=(original,),
    )
    output = FixedSelectOutputAudit(
        source_seal_hash=HASH_B,
        seed_block=1,
        selected_objects=(original,),
        operations=(
            ProposedOperation(
                operation_id="select-1",
                capability=ConstructionOperator.SELECTION,
                referenced_ids=("assertion-1",),
                decided_at=NOW,
            ),
        ),
    )

    assert enforce_fixed_select_output(output, sealed) is None


@pytest.mark.parametrize(
    ("selected", "operation", "match"),
    [
        (
            SemanticFingerprint(
                semantic_id="assertion-new",
                semantic_kind="assertion",
                semantic_hash=HASH_A,
            ),
            ConstructionOperator.SELECTION,
            "new semantic ID",
        ),
        (
            SemanticFingerprint(
                semantic_id="assertion-1",
                semantic_kind="assertion",
                semantic_hash=HASH_C,
            ),
            ConstructionOperator.SELECTION,
            "semantic content changed",
        ),
        (
            SemanticFingerprint(
                semantic_id="assertion-1",
                semantic_kind="assertion",
                semantic_hash=HASH_A,
            ),
            ConstructionOperator.MERGE,
            "constructive operation",
        ),
    ],
)
def test_fixed_select_rejects_new_ids_mutations_and_construction(
    selected: SemanticFingerprint,
    operation: ConstructionOperator,
    match: str,
) -> None:
    sealed = SealedOntologyInventory(
        seal_hash=HASH_B,
        seed_block=1,
        objects=(
            SemanticFingerprint(
                semantic_id="assertion-1",
                semantic_kind="assertion",
                semantic_hash=HASH_A,
            ),
        ),
    )
    output = FixedSelectOutputAudit(
        source_seal_hash=HASH_B,
        seed_block=1,
        selected_objects=(selected,),
        operations=(
            ProposedOperation(
                operation_id="operation-1",
                capability=operation,
                referenced_ids=(selected.semantic_id,),
                created_ids=("assertion-new",) if operation is ConstructionOperator.MERGE else (),
                decided_at=NOW,
            ),
        ),
    )

    with pytest.raises(FixedSelectCapabilityError, match=match):
        enforce_fixed_select_output(output, sealed)


def repair_lineage() -> RepairLineageMetadata:
    return RepairLineageMetadata(
        root_attempt_id="attempt-0",
        base_attempt_id="attempt-0",
        repair_attempt_id="attempt-1",
        semantic_request_hash=HASH_A,
        base_output_hash=HASH_B,
        validation_record_hash=HASH_C,
        diagnostic_codes=("invalid_evidence_id",),
    )


def test_repair_lineage_is_exactly_one_direct_semantics_preserving_attempt() -> None:
    lineage = repair_lineage()
    assert lineage.repair_number == 1
    assert set(lineage.allowed_changes) == {
        "validation_diagnostics",
        "maximum_output_tokens",
    }

    values = lineage.model_dump(exclude={"content_hash"})
    values["root_attempt_id"] = "older-attempt"
    with pytest.raises(ValidationError, match="directly from the base"):
        RepairLineageMetadata(**values)


def test_gpu_request_metadata_binds_complete_manifests_without_calling_a_model() -> None:
    seeds = SeedManifest.from_root(123, seed_block=1, scope=("world-01", "context-a"))
    decode = decoding(seeds.llm_seed)
    capabilities = CapabilityManifest.for_condition(ConditionName.C2_LLM_QUERY)
    pack = packing(ConditionName.C2_LLM_QUERY)

    metadata = make_gpu_request_metadata(
        decoding=decode,
        seeds=seeds,
        capabilities=capabilities,
        packing=pack,
        request_id="request-1",
        attempt_id="attempt-0",
        condition=ConditionName.C2_LLM_QUERY,
        model_repository="Qwen/Qwen3-14B-AWQ",
        model_revision=GIT_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        runtime_version="vllm-test-version",
        pytorch_version="pytorch-test-version",
        cuda_version="cuda-test-version",
        driver_version="driver-test-version",
        hardware_manifest_hash=HASH_A,
        code_revision=GIT_REVISION,
        prompt_hash=HASH_A,
        output_schema_hash=HASH_B,
        semantic_request_hash=HASH_C,
        evidence_packet_hash=HASH_A,
        seed_block=1,
        requested_at=NOW,
        query_revealed_at=NOW,
        timeout_class=RequestTimeoutClass.STANDARD,
        watchdog_seconds=120,
        allocated_gpu_seconds_before_request=0.0,
    )

    assert isinstance(metadata, GPURequestMetadata)
    assert metadata.backend == "vllm_gpu"
    assert metadata.cpu_offload_gb == 0
    assert metadata.request_concurrency == 1
    assert metadata.decoding_manifest_hash == decode.content_hash
    assert metadata.seed_manifest_hash == seeds.content_hash


def test_gpu_request_metadata_rejects_hidden_c2_ontology_and_unmetered_repair() -> None:
    common = dict(
        request_id="request-1",
        attempt_id="attempt-0",
        condition=ConditionName.C2_LLM_QUERY,
        model_repository="Qwen/Qwen3-14B-AWQ",
        model_revision=GIT_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        runtime_version="vllm-test-version",
        pytorch_version="pytorch-test-version",
        cuda_version="cuda-test-version",
        driver_version="driver-test-version",
        hardware_manifest_hash=HASH_A,
        code_revision=GIT_REVISION,
        prompt_hash=HASH_A,
        output_schema_hash=HASH_B,
        semantic_request_hash=HASH_C,
        evidence_packet_hash=HASH_A,
        decoding_manifest_hash=HASH_A,
        seed_manifest_hash=HASH_A,
        capability_manifest_hash=HASH_A,
        packing_report_hash=HASH_A,
        seed_block=1,
        requested_at=NOW,
        query_revealed_at=NOW,
        timeout_class=RequestTimeoutClass.STANDARD,
        watchdog_seconds=120,
        allocated_gpu_seconds_before_request=0.0,
    )
    with pytest.raises(ValidationError, match="hidden prebuilt ontology"):
        GPURequestMetadata(**common, decoding_pass="first_pass", sealed_ontology_hash=HASH_B)
    with pytest.raises(ValidationError, match="repair lineage"):
        GPURequestMetadata(**common, decoding_pass="repair")
