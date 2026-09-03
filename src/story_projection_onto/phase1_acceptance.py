"""Exact eight-call Phase-1 GPU acceptance block.

Importing this module is CPU-only.  The command-line entry point defaults to a
public-safe dry run; an operator must pass ``--execute`` to start vLLM or submit
any GPU request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from story_projection_onto.contracts import (
    CONSTRUCTIVE_OPERATORS,
    ConditionName,
    ConstructionRequest,
    OntologyDraft,
    PreconstructionRequest,
    canonical_json,
    canonical_sha256,
)
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    GPUCallInventory,
    ResourceLimits,
    TimingObservation,
    forecast_gpu_schedule,
    summarize_call_class_timings,
)
from story_projection_onto.gpu_runtime import (
    PINNED_MODEL_REVISION,
    SERVED_MODEL_NAME,
    ChatMessage,
    GenerationResult,
    GuidedJSONRequest,
    ResourceSampler,
    ResourceWatchdog,
    RuntimeStackManifest,
    ServiceState,
    TokenizerManifest,
    VLLMGuidedJSONClient,
    VLLMLaunchConfiguration,
    VLLMService,
    atomic_write_public_json,
    capture_runtime_stack,
    capture_tokenizer_manifest,
    public_runtime_manifest,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    DecodingManifest,
    DecodingPass,
    PackingReport,
    PackingSection,
    enforce_fixed_select_draft,
    sealed_inventory_from_fixed_ontology,
)
from story_projection_onto.store import (
    ArtifactStore,
    AttemptKind,
    BlobStore,
    FailureKind,
    Ledger,
    ModelBackend,
    ModelCallRole,
    ReleaseClass,
    RetryClass,
    StorageBudget,
    StoragePreflight,
)

SCHEMA_VERSION = "1.0.0"
EXPECTED_ACCEPTANCE_COUNTS = {
    "acceptance_c1": 2,
    "acceptance_c2": 3,
    "acceptance_fixed_select": 2,
    "acceptance_repair": 1,
}


class AcceptanceStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    RESUMED = "resumed"


@dataclass(frozen=True, slots=True)
class AcceptanceCall:
    call_id: str
    call_class: str
    condition: ConditionName
    seed_block: int
    request_fixture: str
    prompt_file: str
    watchdog_seconds: int
    decoding_pass: DecodingPass = DecodingPass.FIRST_PASS
    parent_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.call_class not in EXPECTED_ACCEPTANCE_COUNTS:
            raise ValueError(f"unregistered acceptance call class {self.call_class!r}")
        if self.watchdog_seconds not in {90, 120, 180}:
            raise ValueError("acceptance watchdog must be 90, 120, or 180 seconds")
        if (self.decoding_pass is DecodingPass.REPAIR) != (self.parent_call_id is not None):
            raise ValueError("only the repair acceptance call has a parent")

    def public_manifest(self, root: Path) -> dict[str, object]:
        fixture = root / self.request_fixture
        prompt = root / self.prompt_file
        base_schema = _load_json_object(root / "schemas/jsonschema/ontology_draft.schema.json")
        condition_schema = _condition_output_schema(
            base_schema,
            call=self,
            fixture=_load_json_object(fixture),
        )
        return {
            "call_id": self.call_id,
            "call_class": self.call_class,
            "condition": self.condition.value,
            "seed_block": self.seed_block,
            "request_fixture": self.request_fixture,
            "request_fixture_sha256": _file_hash(fixture),
            "prompt_file": self.prompt_file,
            "prompt_sha256": _file_hash(prompt),
            "condition_output_schema_sha256": canonical_sha256(condition_schema),
            "watchdog_seconds": self.watchdog_seconds,
            "decoding_pass": self.decoding_pass.value,
            "parent_call_id": self.parent_call_id,
        }


def phase1_acceptance_calls() -> tuple[AcceptanceCall, ...]:
    """Return the frozen 2 C1 / 3 C2 / 2 Fixed / 1 repair order."""

    return (
        AcceptanceCall(
            call_id="c1-01",
            call_class="acceptance_c1",
            condition=ConditionName.C1_LLM_PRE,
            seed_block=0,
            request_fixture="tests/fixtures/phase1/c1_pre_request.json",
            prompt_file="prompts/c1_pre/prompt_v1.md",
            watchdog_seconds=180,
        ),
        AcceptanceCall(
            call_id="c1-02",
            call_class="acceptance_c1",
            condition=ConditionName.C1_LLM_PRE,
            seed_block=1,
            request_fixture="tests/fixtures/phase1/c1_pre_request.json",
            prompt_file="prompts/c1_pre/prompt_v1.md",
            watchdog_seconds=180,
        ),
        AcceptanceCall(
            call_id="c2-01",
            call_class="acceptance_c2",
            condition=ConditionName.C2_LLM_QUERY,
            seed_block=0,
            request_fixture="tests/fixtures/phase1/c2_query_request.json",
            prompt_file="prompts/c2_query/prompt_v1.md",
            watchdog_seconds=120,
        ),
        AcceptanceCall(
            call_id="c2-02",
            call_class="acceptance_c2",
            condition=ConditionName.C2_LLM_QUERY,
            seed_block=1,
            request_fixture="tests/fixtures/phase1/c2_query_request.json",
            prompt_file="prompts/c2_query/prompt_v1.md",
            watchdog_seconds=120,
        ),
        AcceptanceCall(
            call_id="c2-03",
            call_class="acceptance_c2",
            condition=ConditionName.C2_LLM_QUERY,
            seed_block=2,
            request_fixture="tests/fixtures/phase1/c2_query_request.json",
            prompt_file="prompts/c2_query/prompt_v1.md",
            watchdog_seconds=120,
        ),
        AcceptanceCall(
            call_id="fixed-01",
            call_class="acceptance_fixed_select",
            condition=ConditionName.A_FIXED_SELECT,
            seed_block=1,
            request_fixture="tests/fixtures/phase1/fixed_select_request.json",
            prompt_file="prompts/fixed_select/prompt_v1.md",
            watchdog_seconds=90,
        ),
        AcceptanceCall(
            call_id="fixed-02",
            call_class="acceptance_fixed_select",
            condition=ConditionName.A_FIXED_SELECT,
            seed_block=1,
            request_fixture="tests/fixtures/phase1/fixed_select_request.json",
            prompt_file="prompts/fixed_select/prompt_v1.md",
            watchdog_seconds=90,
        ),
        AcceptanceCall(
            call_id="repair-01",
            call_class="acceptance_repair",
            condition=ConditionName.C2_LLM_QUERY,
            seed_block=0,
            request_fixture="tests/fixtures/phase1/invalid_repair_case.json",
            prompt_file="prompts/repair/prompt_v1.md",
            watchdog_seconds=90,
            decoding_pass=DecodingPass.REPAIR,
            parent_call_id="repair-base-fixture",
        ),
    )


def validate_acceptance_calls(calls: Sequence[AcceptanceCall]) -> None:
    ids = tuple(call.call_id for call in calls)
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Phase-1 acceptance must contain exactly eight unique calls")
    actual = {
        name: sum(call.call_class == name for call in calls) for name in EXPECTED_ACCEPTANCE_COUNTS
    }
    if actual != EXPECTED_ACCEPTANCE_COUNTS:
        raise ValueError(f"acceptance call counts differ from protocol: {actual}")
    for call in calls:
        if call.decoding_pass is DecodingPass.REPAIR:
            if call.parent_call_id != "repair-base-fixture":
                raise ValueError("repair must descend from the independent invalid fixture")
        elif call.parent_call_id is not None:
            raise ValueError("a non-repair acceptance call cannot have a parent")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def acceptance_plan_manifest(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    calls = phase1_acceptance_calls()
    validate_acceptance_calls(calls)
    schema_file = root / "schemas/jsonschema/ontology_draft.schema.json"
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "phase1_gpu_acceptance_plan",
        "model_revision": PINNED_MODEL_REVISION,
        "call_count": len(calls),
        "class_counts": EXPECTED_ACCEPTANCE_COUNTS,
        "output_schema_file": "schemas/jsonschema/ontology_draft.schema.json",
        "output_schema_sha256": _file_hash(schema_file),
        "calls": [call.public_manifest(root) for call in calls],
        "executes_gpu": False,
    }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


class PackingTokenizer(Protocol):
    def encode(self, text: str, **kwargs: object) -> Sequence[int]: ...

    def apply_chat_template(self, conversation: object, **kwargs: object) -> object: ...


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return cast(dict[str, object], value)


def validate_verified_model_manifest(
    path: Path,
    *,
    snapshot_path: Path | None = None,
    shared_cache: Path | None = None,
    model_configuration_path: Path | None = None,
) -> dict[str, object]:
    """Validate the signed inventory and, when supplied, rehash the exact snapshot."""

    manifest = _load_json_object(path)
    supplied_hash = manifest.get("manifest_sha256")
    immutable = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    calculated_hash = hashlib.sha256(canonical_json(immutable).encode("utf-8")).hexdigest()
    if supplied_hash != calculated_hash:
        raise ValueError("verified model manifest hash does not match its contents")
    expected = {
        "repository": "Qwen/Qwen3-14B-AWQ",
        "revision": PINNED_MODEL_REVISION,
        "license": "Apache-2.0",
        "quantization": "awq",
        "single_repository_in_shared_cache": True,
        "single_snapshot_in_shared_cache": True,
        "incomplete_file_count": 0,
    }
    mismatches = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatches:
        raise ValueError("verified model manifest differs at: " + ", ".join(mismatches))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("verified model manifest has no file hash inventory")
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("verified model file entry must be an object")
        relative_text = entry.get("path")
        digest = entry.get("sha256")
        size_bytes = entry.get("size_bytes")
        if (
            not isinstance(relative_text, str)
            or PurePosixPath(relative_text).is_absolute()
            or ".." in PurePosixPath(relative_text).parts
        ):
            raise ValueError("verified model file entry has an unsafe relative path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("verified model file entry has no SHA-256")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("verified model file entry has no nonnegative size")
    if manifest.get("file_count") != len(files):
        raise ValueError("verified model file count differs from its inventory")
    expected_total = sum(cast(int, entry["size_bytes"]) for entry in files)
    if manifest.get("total_bytes") != expected_total:
        raise ValueError("verified model byte total differs from its inventory")
    if model_configuration_path is not None and manifest.get(
        "model_configuration_sha256"
    ) != _file_hash(model_configuration_path):
        raise ValueError("verified model manifest targets a different model configuration")
    if (snapshot_path is None) != (shared_cache is None):
        raise ValueError("snapshot_path and shared_cache must be supplied together")
    if snapshot_path is not None and shared_cache is not None:
        snapshot = snapshot_path.resolve(strict=True)
        cache = shared_cache.resolve(strict=True)
        try:
            snapshot.relative_to(cache)
        except ValueError as exc:
            raise ValueError("verified snapshot is outside the one shared cache") from exc
        expected_paths = {cast(str, entry["path"]) for entry in files}
        actual_paths = {
            candidate.relative_to(snapshot).as_posix()
            for candidate in snapshot.rglob("*")
            if candidate.is_file()
        }
        if actual_paths != expected_paths:
            raise ValueError("snapshot file set differs from verified model manifest")
        for entry in files:
            candidate = snapshot / cast(str, entry["path"])
            try:
                candidate.resolve(strict=True).relative_to(cache)
            except ValueError as exc:
                raise ValueError("snapshot file resolves outside the shared cache") from exc
            if candidate.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"snapshot size changed for {entry['path']}")
            if _file_hash(candidate) != entry["sha256"]:
                raise ValueError(f"snapshot hash changed for {entry['path']}")
        if any(candidate.is_file() for candidate in cache.rglob("*.incomplete")):
            raise ValueError("shared model cache still contains an incomplete file")
    return manifest


def _string_enum(values: Sequence[str]) -> dict[str, object]:
    unique = sorted(set(values))
    if not unique:
        raise ValueError("a frozen identifier enum cannot be empty")
    return {"enum": unique, "type": "string"}


def _condition_output_schema(
    base_schema: Mapping[str, object],
    *,
    call: AcceptanceCall,
    fixture: Mapping[str, object],
) -> dict[str, object]:
    """Derive the selection-only grammar; C1/C2 retain the common schema."""

    schema = cast(dict[str, object], copy.deepcopy(dict(base_schema)))
    if call.condition is not ConditionName.A_FIXED_SELECT:
        return schema
    definitions = cast(dict[str, dict[str, object]], schema.get("$defs"))
    fixed = cast(Mapping[str, object], fixture.get("fixed_ontology"))
    packet = cast(Mapping[str, object], fixture.get("packet"))
    local_schema = cast(Mapping[str, object], fixed.get("local_schema"))
    graph = cast(Mapping[str, object], fixed.get("instance_graph"))
    seal = cast(Mapping[str, object], fixed.get("construction_seal"))
    if not all((definitions, fixed, packet, local_schema, graph, seal)):
        raise ValueError("A-FixedSelect fixture lacks its complete sealed inputs")

    type_ids = [cast(str, row["type_id"]) for row in local_schema["contextual_types"]]
    predicate_ids = [cast(str, row["predicate_id"]) for row in local_schema["predicates"]]
    entity_ids = [cast(str, row["entity_id"]) for row in graph["entities"]]
    event_ids = [cast(str, row["event_id"]) for row in graph["events"]]
    proposition_ids = [
        cast(str, row["proposition_content_id"]) for row in graph["proposition_contents"]
    ]
    assertion_ids = [cast(str, row["assertion_id"]) for row in graph["assertions"]]
    object_ids = [cast(str, value) for value in seal["sealed_object_ids"]]
    evidence_ids = [cast(str, row["evidence_id"]) for row in packet["evidence"]]
    referent_ids = [*entity_ids, *event_ids, *proposition_ids]

    definitions["ConstructionOperator"]["enum"] = [
        operator.value for operator in CapabilityManifest.for_condition(call.condition).allowed
    ]
    decision_properties = cast(
        dict[str, dict[str, object]], definitions["OntologyDecision"]["properties"]
    )
    for name in ("created_object_ids", "removed_object_ids"):
        decision_properties[name]["maxItems"] = 0
    decision_properties["input_object_ids"]["items"] = _string_enum(object_ids)

    identifier_constraints: tuple[tuple[str, str, Sequence[str]], ...] = (
        ("LocalContextSchema", "schema_id", [cast(str, local_schema["schema_id"])]),
        ("LocalTypeDefinition", "type_id", type_ids),
        ("LocalPredicateDefinition", "predicate_id", predicate_ids),
        ("LocalPredicateDefinition", "domain_type_ids", type_ids),
        ("LocalPredicateDefinition", "range_type_ids", type_ids),
        ("Entity", "entity_id", entity_ids),
        ("Entity", "contextual_type_id", type_ids),
        ("Entity", "description_assertion_ids", assertion_ids),
        ("Event", "event_id", event_ids),
        ("Event", "contextual_type_id", type_ids),
        ("Event", "description_assertion_ids", assertion_ids),
        ("PropositionContent", "proposition_content_id", proposition_ids),
        ("PropositionContent", "predicate_id", predicate_ids),
        ("QualifiedAssertion", "assertion_id", assertion_ids),
        ("QualifiedAssertion", "predicate_id", predicate_ids),
        ("QualifiedAssertion", "proposition_content_id", proposition_ids),
        ("RoleBinding", "object_id", referent_ids),
        ("EpistemicScope", "holder_id", entity_ids),
        ("EpistemicScope", "proposition_content_id", proposition_ids),
    )
    for definition_name, property_name, allowed in identifier_constraints:
        properties = cast(dict[str, dict[str, object]], definitions[definition_name]["properties"])
        original = properties[property_name]
        constraint = _string_enum(allowed)
        if original.get("type") == "array":
            original["items"] = constraint
        elif "anyOf" in original:
            original["anyOf"] = [constraint, {"type": "null"}]
        else:
            properties[property_name] = constraint
    for definition_name in (
        "Entity",
        "Event",
        "LocalPredicateDefinition",
        "LocalTypeDefinition",
        "OntologyDecision",
        "PropositionContent",
        "QualifiedAssertion",
        "RoleBinding",
    ):
        properties = cast(dict[str, dict[str, object]], definitions[definition_name]["properties"])
        if "evidence_ids" in properties:
            properties["evidence_ids"]["items"] = _string_enum(evidence_ids)
    cast(dict[str, object], definitions["OmissionRecord"]["properties"])["evidence_id"] = (
        _string_enum(evidence_ids)
    )
    return schema


def _request_sections(
    call: AcceptanceCall,
    fixture: Mapping[str, object],
) -> dict[str, object]:
    if call.decoding_pass is DecodingPass.REPAIR:
        # A repair receives the complete original semantic request as well as the
        # invalid draft and exact fact-free diagnostics. Runtime IDs, hashes, and
        # timestamps are already immutably bound by the call/request manifests;
        # rendering those redundant administrative fields would not add semantic
        # information and would consume the repair-only token reserve.
        root = cast(Path, fixture["__project_root"])
        original = _load_json_object(root / "tests/fixtures/phase1/c2_query_request.json")
        repair = {key: value for key, value in fixture.items() if key != "__project_root"}
        return {
            "upper_ontology": original["upper_ontology"],
            "query_context": original["context"],
            "evidence_packet": original["packet"],
            "semantic_controls": {
                "condition": original["condition"],
                "budgets": original["budgets"],
                "capabilities": original["capabilities"],
            },
            "invalid_draft": repair["base_draft"],
            "validation_diagnostics": repair["validation_report"],
        }
    common = {
        "upper_ontology": fixture["upper_ontology"],
    }
    if call.condition is ConditionName.C1_LLM_PRE:
        return {
            **common,
            "evidence_snapshot": fixture["evidence"],
            "request_envelope": {
                key: value
                for key, value in fixture.items()
                if key not in {"upper_ontology", "evidence"}
            },
        }
    result = {
        **common,
        "query_context": fixture["context"],
        "evidence_packet": fixture["packet"],
        "request_envelope": {
            key: value
            for key, value in fixture.items()
            if key not in {"upper_ontology", "context", "packet", "fixed_ontology"}
        },
    }
    if call.condition is ConditionName.A_FIXED_SELECT:
        result["sealed_ontology"] = fixture["fixed_ontology"]
    return result


def build_acceptance_request(
    *,
    root: Path,
    call: AcceptanceCall,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
) -> GuidedJSONRequest:
    """Pack a fixture without truncation and bind exact tokenizer/decoding metadata."""

    root = root.resolve(strict=True)
    fixture = _load_json_object(root / call.request_fixture)
    if call.decoding_pass is DecodingPass.REPAIR:
        fixture["__project_root"] = root
    output_schema = _condition_output_schema(
        _load_json_object(root / "schemas/jsonschema/ontology_draft.schema.json"),
        call=call,
        fixture=fixture,
    )
    prompt = (root / call.prompt_file).read_text(encoding="utf-8")
    sections = _request_sections(call, fixture)
    user_content = canonical_json(sections)
    messages = (
        ChatMessage(role="system", content=prompt),
        ChatMessage(role="user", content=user_content),
    )
    rendered = tokenizer.apply_chat_template(
        [asdict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes, bytearray)):
        raise ValueError("tokenizer did not return rendered prompt token IDs")
    rendered_count = len(rendered)
    output_schema_hash = canonical_sha256(output_schema)
    named_text = {"system_prompt": prompt, **sections}
    # vLLM receives the complete schema through its guided_json request field,
    # not as duplicated chat text. Bind the full canonical schema hash and mark
    # it complete while honestly recording that it consumes zero prompt tokens.
    packing_sections: list[PackingSection] = [
        PackingSection(
            name="output_schema",
            section_content_hash=output_schema_hash,
            token_count=0,
        )
    ]
    for name, value in named_text.items():
        text = value if isinstance(value, str) else canonical_json(value)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        packing_sections.append(
            PackingSection(
                name=name,
                section_content_hash=hashlib.sha256(text.encode()).hexdigest(),
                token_count=len(encoded),
            )
        )
    # Individually encoding sections is deliberately conservative.  Record any
    # positive chat-wrapper delta explicitly; the exact rendered count is also
    # bound into GuidedJSONRequest and independently checked against the cap.
    individually_encoded = sum(section.token_count for section in packing_sections)
    if rendered_count < individually_encoded:
        raise ValueError(
            "independently encoded prompt sections exceed exact rendered token count"
        )
    if rendered_count > individually_encoded:
        packing_sections.append(
            PackingSection(
                name="chat_protocol",
                section_content_hash=tokenizer_manifest.nonthinking_probe_sha256,
                token_count=rendered_count - individually_encoded,
            )
        )
    maximum_input_tokens, maximum_output_tokens = {
        DecodingPass.FIRST_PASS: (10_240, 2_048),
        DecodingPass.REPAIR: (10_752, 1_536),
    }[call.decoding_pass]
    constructor = (
        DecodingManifest.first_pass
        if call.decoding_pass is DecodingPass.FIRST_PASS
        else DecodingManifest.repair
    )
    decoding = constructor(
        seed=call.seed_block,
        eos_token_id=tokenizer_manifest.eos_token_id,
        end_of_turn_token_ids=tokenizer_manifest.end_of_turn_token_ids,
        chat_template_hash=tokenizer_manifest.chat_template_sha256,
        output_schema_hash=output_schema_hash,
        structured_decoder="vllm-0.10.2-guided_json",
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_input_tokens=maximum_input_tokens,
        maximum_output_tokens=maximum_output_tokens,
    )
    required = {
        ConditionName.C1_LLM_PRE: (
            "system_prompt",
            "output_schema",
            "upper_ontology",
            "evidence_snapshot",
        ),
        ConditionName.C2_LLM_QUERY: (
            "system_prompt",
            "output_schema",
            "upper_ontology",
            "query_context",
            "evidence_packet",
        ),
        ConditionName.A_FIXED_SELECT: (
            "system_prompt",
            "output_schema",
            "upper_ontology",
            "query_context",
            "evidence_packet",
            "sealed_ontology",
        ),
    }[call.condition]
    packing = PackingReport.build(
        condition=call.condition,
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_model_tokens=12_288,
        maximum_input_tokens=maximum_input_tokens,
        reserved_output_tokens=maximum_output_tokens,
        sections=packing_sections,
        required_section_names=required,
        complete_evidence_snapshot=True if call.condition is ConditionName.C1_LLM_PRE else None,
        complete_evidence_packet=None if call.condition is ConditionName.C1_LLM_PRE else True,
        complete_sealed_ontology=(True if call.condition is ConditionName.A_FIXED_SELECT else None),
    )
    return GuidedJSONRequest(
        request_id=call.call_id,
        model_name=SERVED_MODEL_NAME,
        condition=call.condition,
        messages=messages,
        output_schema=output_schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=rendered_count,
    )


def _cited_evidence_ids(value: object) -> set[str]:
    cited: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "evidence_id" and isinstance(child, str):
                cited.add(child)
            elif key.endswith("evidence_ids") and isinstance(child, Sequence):
                cited.update(item for item in child if isinstance(item, str))
            else:
                cited.update(_cited_evidence_ids(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            cited.update(_cited_evidence_ids(child))
    return cited


def validate_acceptance_generation(
    *,
    root: Path,
    call: AcceptanceCall,
    parsed_object: Mapping[str, object],
) -> dict[str, object]:
    """Mechanically audit schema, budget, evidence, and condition capabilities."""

    draft = OntologyDraft.model_validate(parsed_object)
    if call.decoding_pass is DecodingPass.REPAIR:
        request_raw = _load_json_object(root / "tests/fixtures/phase1/c2_query_request.json")
    else:
        request_raw = _load_json_object(root / call.request_fixture)
    capabilities = CapabilityManifest.for_condition(call.condition)
    forbidden = [
        decision.operator.value
        for decision in draft.decisions
        if decision.operator not in capabilities.allowed
    ]
    if forbidden:
        raise ValueError("generated decisions exceed condition capability allowlist")

    if call.condition is ConditionName.C1_LLM_PRE:
        request = PreconstructionRequest.model_validate(request_raw)
        evidence_ids = {item.evidence_id for item in request.evidence}
        budgets = request.budgets
        horizon_leaks: list[str] = []
    else:
        request = ConstructionRequest.model_validate(request_raw)
        evidence_ids = {item.evidence_id for item in request.packet.evidence}
        budgets = request.budgets
        maximum_discourse = request.context.spoiler_horizon.max_discourse_position.ordering_key
        maximum_revelation = request.context.spoiler_horizon.max_revelation_position
        maximum_revelation_order = (
            None if maximum_revelation is None else maximum_revelation.revelation_order
        )
        horizon_leaks = [
            assertion.assertion_id
            for assertion in draft.instance_graph.assertions
            if (
                assertion.temporal_scope.discourse_position.ordering_key > maximum_discourse
                or (
                    maximum_revelation_order is not None
                    and assertion.temporal_scope.revelation_position.revelation_order
                    > maximum_revelation_order
                )
            )
        ]
        if horizon_leaks:
            raise ValueError("generated draft crosses the fixed spoiler horizon")
        if call.decoding_pass is DecodingPass.FIRST_PASS and any(
            decision.decided_at < request.requested_at for decision in draft.decisions
        ):
            raise ValueError("query-time construction decision predates request/reveal")
        if call.condition is ConditionName.A_FIXED_SELECT:
            if request.fixed_ontology is None:
                raise ValueError("fixed acceptance request omitted its sealed ontology")
            source_draft = OntologyDraft.model_validate(
                _load_json_object(root / "tests/fixtures/phase1/c1_pre_output.json")
            )
            sealed = sealed_inventory_from_fixed_ontology(
                request.fixed_ontology,
                seed_block=call.seed_block,
                source_draft=source_draft,
            )
            enforce_fixed_select_draft(draft, sealed=sealed, seed_block=call.seed_block)
    draft.budget_accounting.validate_against(budgets)
    cited = _cited_evidence_ids(draft.model_dump(mode="json"))
    unknown = sorted(cited - evidence_ids)
    if unknown:
        raise ValueError("generated draft cites evidence outside the complete frozen input")
    grounded_records = (
        *draft.local_schema.contextual_types,
        *draft.local_schema.predicates,
        *draft.instance_graph.entities,
        *draft.instance_graph.events,
        *draft.instance_graph.proposition_contents,
        *draft.instance_graph.assertions,
        *draft.decisions,
    )
    missing_grounding = [
        getattr(record, "content_hash", type(record).__name__)
        for record in grounded_records
        if not record.evidence_ids
    ]
    unsupported_descriptions = [
        item.entity_id
        for item in draft.instance_graph.entities
        if not item.description_assertion_ids
    ] + [
        item.event_id for item in draft.instance_graph.events if not item.description_assertion_ids
    ]
    missing_why_support = [
        assertion.assertion_id
        for assertion in draft.instance_graph.assertions
        if not assertion.why_matters_evidence_ids
    ]
    if missing_grounding or unsupported_descriptions or missing_why_support:
        raise ValueError("generated draft lacks required evidence/description grounding")
    constructive = sorted(
        {
            decision.operator.value
            for decision in draft.decisions
            if decision.operator in CONSTRUCTIVE_OPERATORS
        }
    )
    if (
        call.condition in {ConditionName.C1_LLM_PRE, ConditionName.C2_LLM_QUERY}
        and not constructive
    ):
        raise ValueError("active construction acceptance output has no construction operation")
    graph = draft.instance_graph
    return {
        "draft_hash": draft.content_hash,
        "schema_valid": True,
        "budget_valid": True,
        "evidence_ids_valid": True,
        "evidence_citation_count": len(cited),
        "grounded_record_count": len(grounded_records),
        "grounding_complete": True,
        "horizon_leak_count": len(horizon_leaks),
        "capability_valid": True,
        "entity_count": len(graph.entities),
        "event_count": len(graph.events),
        "assertion_count": len(graph.assertions),
        "decision_count": len(draft.decisions),
        "constructive_operators": constructive,
    }


def _private_atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parsed_object_from_vllm_response(payload: bytes) -> dict[str, object]:
    try:
        response = json.loads(payload)
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored vLLM response cannot be resumed as guided JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("stored guided JSON response is not an object")
    return cast(dict[str, object], parsed)


def _request_public_metadata(request: GuidedJSONRequest) -> dict[str, object]:
    """Expose complete reproducibility metadata without prompt/evidence contents."""

    return {
        "request_hash": request.request_hash,
        "prompt_hash": request.prompt_hash,
        "rendered_input_token_count": request.rendered_input_token_count,
        "decoding_manifest": request.decoding.model_dump(mode="json"),
        "packing_report": request.packing.model_dump(mode="json", by_alias=True),
        "capability_manifest": CapabilityManifest.for_condition(request.condition).model_dump(
            mode="json"
        ),
        "condition_output_schema_sha256": canonical_sha256(request.output_schema),
    }


@dataclass(slots=True)
class AcceptanceRunner:
    root: Path
    run_id: str
    service: VLLMService
    ledger: Ledger
    artifacts: ArtifactStore
    resource_sampler: ResourceSampler
    tokenizer: PackingTokenizer
    tokenizer_manifest: TokenizerManifest
    checkpoint_path: Path
    runtime_stack: RuntimeStackManifest | None = None

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or len(self.run_id) > 96
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in self.run_id
            )
        ):
            raise ValueError("run_id must be a lowercase public-safe identifier")
        self.root = self.root.resolve(strict=True)

    def _checkpoint(
        self,
        completed: Sequence[str],
        plan_hash: str,
        *,
        failed_call_id: str | None = None,
        resume_sequence: int = 0,
    ) -> None:
        _private_atomic_json(
            self.checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "plan_hash": plan_hash,
                "completed_call_ids": list(completed),
                "failed_call_id": failed_call_id,
                "resume_sequence": resume_sequence,
            },
        )

    def _resume_state(self, plan_hash: str) -> tuple[list[str], int]:
        if not self.checkpoint_path.exists():
            return [], 0
        value = _load_json_object(self.checkpoint_path)
        if value.get("run_id") != self.run_id or value.get("plan_hash") != plan_hash:
            raise ValueError("acceptance checkpoint does not match run identity and plan")
        completed = value.get("completed_call_ids")
        if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
            raise ValueError("acceptance checkpoint has invalid completed-call list")
        failed = value.get("failed_call_id")
        if failed is not None:
            raise RuntimeError(
                f"acceptance run is terminal after {failed!r}; use a declared reserve/new run ID"
            )
        expected_prefix = [call.call_id for call in phase1_acceptance_calls()[: len(completed)]]
        if completed != expected_prefix:
            raise ValueError("acceptance checkpoint is not a valid ordered plan prefix")
        resume_sequence = value.get("resume_sequence", 0)
        if (
            isinstance(resume_sequence, bool)
            or not isinstance(resume_sequence, int)
            or resume_sequence < 0
        ):
            raise ValueError("acceptance checkpoint has invalid resume sequence")
        return cast(list[str], completed), resume_sequence

    def _ensure_independent_repair_base(self, plan_hash: str) -> tuple[str, str]:
        """Ledger the hand-authored invalid base without pretending it was a GPU call."""

        fixture = _load_json_object(self.root / "tests/fixtures/phase1/invalid_repair_case.json")
        base_draft = fixture["base_draft"]
        base_payload = (canonical_json(base_draft) + "\n").encode("utf-8")
        artifact = self.artifacts.put_bytes(
            base_payload,
            media_type="application/json",
            release_class=ReleaseClass.PUBLIC,
        )
        job = self.ledger.create_or_resume_job(
            {
                "run_id": self.run_id,
                "call_id": "repair-base-fixture",
                "plan_hash": plan_hash,
            },
            release_class=ReleaseClass.PUBLIC,
        )
        attempt_id = f"{self.run_id}-repair-base-fixture-attempt"
        self.ledger.record_attempt(
            attempt_id=attempt_id,
            job_id=job.job_id,
            attempt_kind=AttemptKind.BASE,
            input_hash=canonical_sha256(base_draft),
            config_hash=plan_hash,
            seed=0,
        )
        self.ledger.record_model_call(
            model_call_id=f"{self.run_id}-repair-base-fixture",
            job_id=job.job_id,
            attempt_id=attempt_id,
            gpu_event_id=None,
            backend=ModelBackend.HAND_AUTHORED_FIXTURE,
            call_role=ModelCallRole.PILOT,
            retry_class=RetryClass.BASE,
            model_manifest_hash=self.service.configuration.configuration_hash,
            decoding_manifest_hash=plan_hash,
            request_hash=canonical_sha256(fixture["repair_lineage"]),
            response_artifact_hash=artifact.content_hash,
            construction_unit_hash=canonical_sha256({"fixture": "invalid_repair_case.json"}),
            served_context_count=1,
            prompt_tokens=0,
            completion_tokens=0,
            allocated_gpu_seconds=0,
            successful=False,
        )
        self.ledger.record_failure(
            attempt_id=attempt_id,
            failure_kind=FailureKind.INVALID_OUTPUT,
            message="Hand-authored invalid fixture for the bounded repair probe",
            details={"diagnostics_are_fact_free": True},
            artifact_hash=artifact.content_hash,
        )
        return job.job_id, attempt_id

    def run(self) -> dict[str, object]:
        calls = phase1_acceptance_calls()
        validate_acceptance_calls(calls)
        plan = acceptance_plan_manifest(self.root)
        plan_hash = cast(str, plan["manifest_sha256"])
        completed, previous_resume_sequence = self._resume_state(plan_hash)
        resume_sequence = previous_resume_sequence + 1
        self._checkpoint(completed, plan_hash, resume_sequence=resume_sequence)
        results: list[dict[str, object]] = []
        timing_observations: list[TimingObservation] = []
        successful_audits: list[tuple[AcceptanceCall, dict[str, object]]] = []
        call_jobs: dict[str, str] = {}
        remaining_call_ceiling = sum(call.watchdog_seconds for call in calls)
        service_checkpoint = self.checkpoint_path.with_name(self.checkpoint_path.name + ".service")
        resumed_live_service = False
        if service_checkpoint.exists() and hasattr(self.service, "resume_from_checkpoint"):
            resumed_live_service = self.service.resume_from_checkpoint(service_checkpoint)
        service_event_id: str | None = None
        if self.service.state is ServiceState.STOPPED:
            service_event_id = f"{self.run_id}-service-start-{resume_sequence:03d}"
            before_start = self.ledger.gpu_summary().total_allocated_microseconds
            self.service.start(
                session_id=self.run_id,
                event_id=service_event_id,
                watchdog_seconds=180,
                remaining_required_seconds=remaining_call_ceiling + 180,
            )
            after_start = self.ledger.gpu_summary().total_allocated_microseconds
            timing_observations.append(
                TimingObservation(
                    call_class="gpu_session_start",
                    allocated_seconds=(after_start - before_start) / 1_000_000,
                )
            )
        restart_event_id = f"{self.run_id}-restart-{resume_sequence:03d}"
        before_restart = self.ledger.gpu_summary().total_allocated_microseconds
        self.service.restart(
            event_id=restart_event_id,
            watchdog_seconds=180,
            remaining_required_seconds=remaining_call_ceiling,
        )
        after_restart = self.ledger.gpu_summary().total_allocated_microseconds
        timing_observations.append(
            TimingObservation(
                call_class="gpu_session_start",
                allocated_seconds=(after_restart - before_restart) / 1_000_000,
            )
        )
        self.service = self.service.handoff_resume(service_checkpoint)
        uptime = None
        resource_watchdog = ResourceWatchdog(
            sampler=self.resource_sampler,
            root_pid=self.service.pid,
            sample_prefix=f"{self.run_id}-periodic-{resume_sequence:03d}",
            allocation_guard=getattr(self.service, "require_hard_stop_margin", None),
            on_failure=lambda _: self.service.emergency_stop(),
        )
        resource_watchdog.start()
        try:
            self.resource_sampler.sample(
                sample_id=f"{self.run_id}-after-load-{resume_sequence:03d}",
                root_pid=self.service.pid,
                gpu_event_id=restart_event_id,
            )
            for call_index, call in enumerate(calls):
                request = build_acceptance_request(
                    root=self.root,
                    call=call,
                    tokenizer=self.tokenizer,
                    tokenizer_manifest=self.tokenizer_manifest,
                )
                remaining_required_seconds = sum(
                    later.watchdog_seconds for later in calls[call_index + 1 :]
                )
                if call.call_id in completed:
                    model_call = self.ledger.get_model_call(f"{self.run_id}-{call.call_id}")
                    call_jobs[call.call_id] = model_call.job_id
                    if model_call.response_artifact_hash is None:
                        raise RuntimeError("resumed model call has no response artifact")
                    stored = self.artifacts.blobs.read_bytes(
                        self.ledger.get_artifact(model_call.response_artifact_hash)
                    )
                    audit = validate_acceptance_generation(
                        root=self.root,
                        call=call,
                        parsed_object=_parsed_object_from_vllm_response(stored),
                    )
                    successful_audits.append((call, audit))
                    timing_observations.append(
                        TimingObservation(
                            call_class=call.call_class,
                            allocated_seconds=model_call.allocated_gpu_microseconds / 1_000_000,
                        )
                    )
                    results.append(
                        {
                            "call_id": call.call_id,
                            "status": AcceptanceStatus.RESUMED.value,
                            **_request_public_metadata(request),
                            "response_artifact_hash": model_call.response_artifact_hash,
                            "prompt_tokens": model_call.prompt_tokens,
                            "completion_tokens": model_call.completion_tokens,
                            "mechanical_audit": audit,
                            "repair_lineage": (
                                None
                                if call.decoding_pass is DecodingPass.FIRST_PASS
                                else {
                                    "model_visible_base_attempt_id": "phase1-c2-invalid-base",
                                    "ledger_parent_attempt_id": self.ledger.attempt_lineage(
                                        model_call.attempt_id
                                    )[0].attempt_id,
                                    "independent_from_acceptance_c2_calls": True,
                                    "repair_number": 1,
                                }
                            ),
                        }
                    )
                    continue
                parent_attempt: str | None = None
                if call.decoding_pass is DecodingPass.REPAIR:
                    parent_job, parent_attempt = self._ensure_independent_repair_base(plan_hash)
                else:
                    parent_job = call_jobs.get(call.parent_call_id or "")
                identity = {
                    "run_id": self.run_id,
                    "call_id": call.call_id if parent_job is None else call.parent_call_id,
                    "plan_hash": plan_hash,
                }
                job = (
                    self.ledger.get_job(parent_job)
                    if parent_job is not None
                    else self.ledger.create_or_resume_job(
                        identity,
                        release_class=ReleaseClass.PUBLIC,
                    )
                )
                call_jobs[call.call_id] = job.job_id
                attempt_id = f"{self.run_id}-{call.call_id}-attempt"
                try:
                    self.ledger.attempt_lineage(attempt_id)
                except KeyError:
                    pass
                else:
                    self._checkpoint(
                        completed,
                        plan_hash,
                        failed_call_id=call.call_id,
                        resume_sequence=resume_sequence,
                    )
                    raise RuntimeError(
                        "unfinished acceptance attempt requires a declared reserve successor"
                    )
                self.ledger.record_attempt(
                    attempt_id=attempt_id,
                    job_id=job.job_id,
                    attempt_kind=(
                        AttemptKind.BASE
                        if call.decoding_pass is DecodingPass.FIRST_PASS
                        else AttemptKind.REPAIR
                    ),
                    parent_attempt_id=parent_attempt,
                    input_hash=request.request_hash,
                    config_hash=request.decoding.content_hash,
                    seed=request.decoding.seed,
                )
                event_id = f"{self.run_id}-{call.call_id}-gpu"
                call_role = (
                    ModelCallRole.REPAIR
                    if call.decoding_pass is DecodingPass.REPAIR
                    else ModelCallRole.PILOT
                )
                retry_class = (
                    RetryClass.SHORT
                    if call.decoding_pass is DecodingPass.REPAIR
                    else RetryClass.BASE
                )
                construction_unit_hash = canonical_sha256(
                    {"fixture": call.request_fixture, "seed_block": call.seed_block}
                )
                before_summary = self.ledger.gpu_summary()
                before = before_summary.total_allocated_microseconds
                try:
                    if call.call_id in {"c1-01", "c1-02"}:
                        lifecycle_operation = (
                            self.service.run_warmup
                            if call.call_id == "c1-01"
                            else self.service.run_schema_probe
                        )
                        generated = cast(
                            GenerationResult,
                            lifecycle_operation(
                                partial(
                                    self.service.client.generate,
                                    request,
                                    watchdog_seconds=call.watchdog_seconds,
                                ),
                                event_id=event_id,
                                watchdog_seconds=call.watchdog_seconds,
                                job_id=job.job_id,
                                attempt_id=attempt_id,
                                remaining_required_seconds=remaining_required_seconds,
                            ),
                        )
                    else:
                        generated = self.service.generate(
                            request,
                            event_id=event_id,
                            watchdog_seconds=call.watchdog_seconds,
                            repair=call.decoding_pass is DecodingPass.REPAIR,
                            job_id=job.job_id,
                            attempt_id=attempt_id,
                            remaining_required_seconds=remaining_required_seconds,
                        )
                except Exception as exc:
                    after_summary = self.ledger.gpu_summary()
                    if after_summary.event_count > before_summary.event_count:
                        failed_allocated_seconds = (
                            after_summary.total_allocated_microseconds
                            - before_summary.total_allocated_microseconds
                        ) / 1_000_000
                        timing_observations.append(
                            TimingObservation(
                                call_class=call.call_class,
                                allocated_seconds=failed_allocated_seconds,
                            )
                        )
                        self.ledger.record_model_call(
                            model_call_id=f"{self.run_id}-{call.call_id}",
                            job_id=job.job_id,
                            attempt_id=attempt_id,
                            gpu_event_id=event_id,
                            backend=ModelBackend.VLLM_GPU,
                            call_role=call_role,
                            retry_class=retry_class,
                            model_manifest_hash=self.service.configuration.configuration_hash,
                            decoding_manifest_hash=request.decoding.content_hash,
                            request_hash=request.request_hash,
                            response_artifact_hash=None,
                            construction_unit_hash=construction_unit_hash,
                            served_context_count=1,
                            prompt_tokens=0,
                            completion_tokens=0,
                            allocated_gpu_seconds=failed_allocated_seconds,
                            successful=False,
                        )
                    lowered_error = str(exc).casefold()
                    self.ledger.record_failure(
                        attempt_id=attempt_id,
                        failure_kind=(
                            FailureKind.TIMEOUT
                            if isinstance(exc, TimeoutError)
                            else (
                                FailureKind.OUT_OF_MEMORY
                                if isinstance(exc, MemoryError) or "out of memory" in lowered_error
                                else FailureKind.SERVICE
                            )
                        ),
                        message="Phase-1 acceptance call failed",
                        details={"exception_type": type(exc).__name__, "call_id": call.call_id},
                    )
                    results.append(
                        {
                            "call_id": call.call_id,
                            "status": AcceptanceStatus.FAILED.value,
                            "exception_type": type(exc).__name__,
                        }
                    )
                    self._checkpoint(
                        completed,
                        plan_hash,
                        failed_call_id=call.call_id,
                        resume_sequence=resume_sequence,
                    )
                    break
                after = self.ledger.gpu_summary().total_allocated_microseconds
                timing_observations.append(
                    TimingObservation(
                        call_class=call.call_class,
                        allocated_seconds=(after - before) / 1_000_000,
                    )
                )
                artifact = self.artifacts.put_bytes(
                    generated.raw_response,
                    media_type="application/json",
                    release_class=ReleaseClass.PUBLIC,
                )
                common_model_call = {
                    "model_call_id": f"{self.run_id}-{call.call_id}",
                    "job_id": job.job_id,
                    "attempt_id": attempt_id,
                    "gpu_event_id": event_id,
                    "backend": ModelBackend.VLLM_GPU,
                    "call_role": call_role,
                    "retry_class": retry_class,
                    "model_manifest_hash": self.service.configuration.configuration_hash,
                    "decoding_manifest_hash": request.decoding.content_hash,
                    "request_hash": request.request_hash,
                    "response_artifact_hash": artifact.content_hash,
                    "construction_unit_hash": construction_unit_hash,
                    "served_context_count": 1,
                    "prompt_tokens": generated.prompt_tokens,
                    "completion_tokens": generated.completion_tokens,
                    "allocated_gpu_seconds": (after - before) / 1_000_000,
                }
                try:
                    if generated.finish_reason != "stop":
                        raise ValueError("acceptance generation did not finish at a stop token")
                    if generated.prompt_tokens != request.rendered_input_token_count:
                        raise ValueError(
                            "vLLM prompt-token count differs from the frozen complete packing"
                        )
                    if generated.completion_tokens > request.decoding.maximum_output_tokens:
                        raise ValueError("vLLM completion exceeded its declared output cap")
                    audit = validate_acceptance_generation(
                        root=self.root,
                        call=call,
                        parsed_object=generated.parsed_object,
                    )
                except Exception as exc:
                    self.ledger.record_model_call(**common_model_call, successful=False)
                    self.ledger.record_failure(
                        attempt_id=attempt_id,
                        failure_kind=FailureKind.INVALID_OUTPUT,
                        message="Phase-1 generated output failed mechanical validation",
                        details={"exception_type": type(exc).__name__, "call_id": call.call_id},
                        artifact_hash=artifact.content_hash,
                    )
                    results.append(
                        {
                            "call_id": call.call_id,
                            "status": AcceptanceStatus.FAILED.value,
                            "exception_type": type(exc).__name__,
                            "response_artifact_hash": artifact.content_hash,
                            "allocated_gpu_microseconds": after - before,
                        }
                    )
                    self._checkpoint(
                        completed,
                        plan_hash,
                        failed_call_id=call.call_id,
                        resume_sequence=resume_sequence,
                    )
                    break
                self.ledger.record_model_call(
                    **common_model_call,
                    successful=True,
                )
                completed.append(call.call_id)
                successful_audits.append((call, audit))
                self._checkpoint(
                    completed,
                    plan_hash,
                    resume_sequence=resume_sequence,
                )
                self.resource_sampler.sample(
                    sample_id=f"{self.run_id}-{call.call_id}-resources",
                    root_pid=self.service.pid,
                    gpu_event_id=event_id,
                    job_id=job.job_id,
                )
                results.append(
                    {
                        "call_id": call.call_id,
                        "status": AcceptanceStatus.COMPLETED.value,
                        **generated.public_manifest(),
                        **_request_public_metadata(request),
                        "response_artifact_hash": artifact.content_hash,
                        "allocated_gpu_microseconds": after - before,
                        "mechanical_audit": audit,
                        "repair_lineage": (
                            None
                            if call.decoding_pass is DecodingPass.FIRST_PASS
                            else {
                                "model_visible_base_attempt_id": "phase1-c2-invalid-base",
                                "ledger_parent_attempt_id": parent_attempt,
                                "independent_from_acceptance_c2_calls": True,
                                "repair_number": 1,
                            }
                        ),
                    }
                )
        finally:
            resource_watchdog.stop(raise_failure=False)
            uptime = self.service.shutdown()
        resource_failure = resource_watchdog.failure
        if resource_failure is not None:
            self._checkpoint(
                completed,
                plan_hash,
                failed_call_id="resource-watchdog",
                resume_sequence=resume_sequence,
            )
        timing = summarize_call_class_timings(timing_observations)
        timing_by_name = {item.call_class: item for item in timing}
        timing_gate = {
            "required_call_classes": ["gpu_session_start", *EXPECTED_ACCEPTANCE_COUNTS],
            "observed_call_classes": sorted(timing_by_name),
            "all_classes_observed": {
                "gpu_session_start",
                *EXPECTED_ACCEPTANCE_COUNTS,
            }.issubset(timing_by_name),
            "acceptance_sample_counts_complete": all(
                timing_by_name.get(name) is not None
                and timing_by_name[name].sample_count == expected_count
                for name, expected_count in EXPECTED_ACCEPTANCE_COUNTS.items()
            ),
            "model_load_sample_count": (
                0
                if timing_by_name.get("gpu_session_start") is None
                else timing_by_name["gpu_session_start"].sample_count
            ),
        }
        inventory = GPUCallInventory.load(self.root / "configs/study/gpu_call_inventory.json")
        resource_limits = ResourceLimits.load(self.root / "configs/study/resource_limits.json")
        forecast = forecast_gpu_schedule(
            inventory,
            timing_observations,
            limits=resource_limits,
        )
        required_operators = {operator.value for operator in CONSTRUCTIVE_OPERATORS}
        c1_operators = {
            operator
            for call, audit in successful_audits
            if call.condition is ConditionName.C1_LLM_PRE
            for operator in cast(list[str], audit["constructive_operators"])
        }
        c2_operators = {
            operator
            for call, audit in successful_audits
            if call.condition is ConditionName.C2_LLM_QUERY
            and call.decoding_pass is DecodingPass.FIRST_PASS
            for operator in cast(list[str], audit["constructive_operators"])
        }
        operator_gate = {
            "required": sorted(required_operators),
            "c1_observed": sorted(c1_operators),
            "c2_observed": sorted(c2_operators),
            "c1_complete": required_operators.issubset(c1_operators),
            "c2_complete": required_operators.issubset(c2_operators),
        }
        audit_gate = all(
            audit["horizon_leak_count"] == 0
            and audit["grounding_complete"] is True
            and audit["evidence_ids_valid"] is True
            for _, audit in successful_audits
        )
        gate_passed = (
            len(completed) == len(calls)
            and resource_failure is None
            and forecast.admitted
            and operator_gate["c1_complete"] is True
            and operator_gate["c2_complete"] is True
            and audit_gate
            and timing_gate["all_classes_observed"] is True
            and timing_gate["acceptance_sample_counts_complete"] is True
        )
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase1_gpu_acceptance_result",
            "run_id": self.run_id,
            "plan_hash": plan_hash,
            "call_count": len(calls),
            "completed_call_count": len(completed),
            "gate_passed": gate_passed,
            "calls": results,
            "runtime": public_runtime_manifest(
                launcher=self.service.configuration,
                tokenizer=self.tokenizer_manifest,
                runtime_stack=self.runtime_stack,
                resource_samples=self.resource_sampler.samples,
                uptime=uptime,
                ledger=self.ledger,
            ),
            "resumed_live_service": resumed_live_service,
            "vllm_service_stopped": self.service.state.value == "stopped",
            "resource_watchdog": {
                "sample_count": resource_watchdog.sample_count,
                "failure_type": (
                    None if resource_failure is None else type(resource_failure).__name__
                ),
            },
            "timing_by_call_class": [asdict(item) for item in timing],
            "timing_gate": timing_gate,
            "full_manifest_forecast": {
                "rows": [asdict(row) for row in forecast.rows],
                "total_seconds": forecast.total_seconds,
                "total_hours": forecast.total_hours,
                "scheduled_limit_seconds": forecast.scheduled_limit_seconds,
                "preferred_limit_seconds": forecast.preferred_limit_seconds,
                "hard_limit_seconds": forecast.hard_limit_seconds,
                "admission_signal": forecast.admission_signal.value,
                "admitted": forecast.admitted,
                "within_preferred_margin": forecast.within_preferred_margin,
            },
            "operator_coverage_gate": operator_gate,
            "grounding_horizon_gate": audit_gate,
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--shared-cache", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quota-root", type=Path)
    parser.add_argument("--verified-model-manifest", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(arguments)


def _require_execution_arguments(options: argparse.Namespace) -> None:
    names = (
        "run_id",
        "snapshot",
        "shared_cache",
        "ledger",
        "artifact_root",
        "checkpoint",
        "quota_root",
        "verified_model_manifest",
    )
    missing = [name for name in names if getattr(options, name) is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit("--execute requires: " + flags)


def main(arguments: Sequence[str] | None = None) -> int:
    options = parse_arguments(arguments)
    root = options.project_root.resolve(strict=True)
    if not options.execute:
        atomic_write_public_json(options.output, acceptance_plan_manifest(root))
        return 0
    _require_execution_arguments(options)
    verified_model = validate_verified_model_manifest(options.verified_model_manifest)
    configuration = VLLMLaunchConfiguration.from_model_configuration(
        snapshot_path=options.snapshot,
        shared_cache=options.shared_cache,
        model_configuration_path=root / "configs/study/model.json",
        verified_snapshot_manifest_sha256=cast(str, verified_model["manifest_sha256"]),
        port=options.port,
    )
    limits = ResourceLimits.load(root / "configs/study/resource_limits.json")
    budget = StorageBudget(
        total_allocation_bytes=limits.maximum_project_allocation_bytes,
        max_occupied_bytes=limits.maximum_project_occupied_bytes,
        min_headroom_bytes=limits.minimum_storage_headroom_bytes,
    )
    controlled_paths = (
        root,
        options.shared_cache,
        options.ledger.parent,
        options.artifact_root,
        options.checkpoint.parent,
        options.output.parent,
    )
    storage = StoragePreflight(
        options.quota_root,
        controlled_paths=controlled_paths,
        budget=budget,
    )
    storage.require()
    verified_model = validate_verified_model_manifest(
        options.verified_model_manifest,
        snapshot_path=options.snapshot,
        shared_cache=options.shared_cache,
        model_configuration_path=root / "configs/study/model.json",
    )
    if verified_model["manifest_sha256"] != configuration.verified_snapshot_manifest_sha256:
        raise RuntimeError("launcher and reverified snapshot manifests differ")
    tokenizer_manifest = capture_tokenizer_manifest(options.snapshot)
    runtime_stack = capture_runtime_stack()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(options.snapshot),
        local_files_only=True,
        trust_remote_code=False,
        revision=PINNED_MODEL_REVISION,
    )
    with Ledger(options.ledger) as ledger:
        meter = AllocatedGPUMeter.from_limits(ledger, limits)
        client = VLLMGuidedJSONClient(configuration.base_url)
        resource_sampler = ResourceSampler(limits=limits, storage=storage, ledger=ledger)
        service = VLLMService(
            configuration=configuration,
            client=client,
            meter=meter,
            log_path=options.checkpoint.parent / f"{options.run_id}.vllm.log",
            startup_resource_sampler=resource_sampler,
            preflight_endpoint_check=lambda: client.health(0.25),
            readiness_check=lambda: client.ready(2.0),
        )
        runner = AcceptanceRunner(
            root=root,
            run_id=options.run_id,
            service=service,
            ledger=ledger,
            artifacts=ArtifactStore(BlobStore(options.artifact_root), ledger),
            resource_sampler=resource_sampler,
            tokenizer=cast(PackingTokenizer, tokenizer),
            tokenizer_manifest=tokenizer_manifest,
            checkpoint_path=options.checkpoint,
            runtime_stack=runtime_stack,
        )
        try:
            result = runner.run()
        except Exception as exc:
            # ``run`` transfers the live process handle into ``runner.service``
            # during its checkpoint/resume exercise. Always stop and report the
            # current owner rather than the pre-handoff controller.
            active_service = runner.service
            uptime = active_service.shutdown()
            failure_payload: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "kind": "phase1_gpu_acceptance_result",
                "run_id": options.run_id,
                "gate_passed": False,
                "completed_call_count": 0,
                "failure_type": type(exc).__name__,
                "runtime": public_runtime_manifest(
                    launcher=active_service.configuration,
                    tokenizer=tokenizer_manifest,
                    runtime_stack=runtime_stack,
                    resource_samples=resource_sampler.samples,
                    uptime=uptime,
                    ledger=ledger,
                ),
                "vllm_service_stopped": active_service.state.value == "stopped",
            }
            result = {
                **failure_payload,
                "manifest_sha256": canonical_sha256(failure_payload),
            }
    atomic_write_public_json(options.output, result)
    return 0 if result["gate_passed"] is True else 2


__all__ = [
    "EXPECTED_ACCEPTANCE_COUNTS",
    "SCHEMA_VERSION",
    "AcceptanceCall",
    "AcceptanceRunner",
    "AcceptanceStatus",
    "acceptance_plan_manifest",
    "build_acceptance_request",
    "main",
    "parse_arguments",
    "phase1_acceptance_calls",
    "validate_acceptance_calls",
    "validate_acceptance_generation",
    "validate_verified_model_manifest",
]
