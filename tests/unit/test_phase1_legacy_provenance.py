from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from story_projection_onto.contracts import ConditionName, OntologyDraft, canonical_sha256
from story_projection_onto.gpu_runtime import PINNED_MODEL_REVISION, TokenizerManifest
from story_projection_onto.phase1_acceptance import (
    build_acceptance_request,
    phase1_acceptance_calls,
    validate_acceptance_generation,
)
from story_projection_onto.phase1_legacy_provenance import (
    CERTIFICATE_RELATIVE_PATH,
    Phase1LegacyEvidenceProvenanceBridge,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "phase1"
SHARED_EVIDENCE_HASH = "a20b09f014e86dbb5ef490beb35134b75a1ad306536de6d0d07e4570de8f1b73"


class FakeTokenizer:
    def encode(self, text: str, **kwargs: object) -> list[int]:
        del kwargs
        return list(range(len(text.split())))

    def apply_chat_template(self, conversation: object, **kwargs: object) -> list[int]:
        assert kwargs["enable_thinking"] is False
        assert isinstance(conversation, (list, tuple))
        rendered = " chat_turn ".join(
            (
                "chat_start",
                *(str(item["content"]) for item in conversation),
                "assistant_start",
                "chat_end",
            )
        )
        return list(range(len(rendered.split())))


def tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        schema_version="1.0.0",
        repository="Qwen/Qwen3-14B-AWQ",
        revision=PINNED_MODEL_REVISION,
        tokenizer_class="tests.FakeTokenizer",
        tokenizer_revision=PINNED_MODEL_REVISION,
        tokenizer_file_sha256=(("tokenizer.json", "a" * 64),),
        eos_token_id=7,
        end_of_turn_token_ids=(8,),
        stop_token_ids=(7, 8),
        chat_template_sha256="a" * 64,
        nonthinking_probe_sha256="b" * 64,
        nonthinking_probe_token_count=10,
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


def load_output(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def source_bound_output(
    name: str,
    *,
    bridge: Phase1LegacyEvidenceProvenanceBridge,
    call_index: int,
) -> dict[str, object]:
    value = load_output(name)
    call = phase1_acceptance_calls()[call_index]
    resolved = bridge.resolve_call(
        call_id=call.call_id,
        condition=call.condition,
        request_fixture=call.request_fixture,
    )
    source_hashes = {
        item.evidence_id: item.provenance.source_artifact_hash for item in resolved.evidence
    }
    graph = value["instance_graph"]
    assert isinstance(graph, dict)
    assertions = graph["assertions"]
    assert isinstance(assertions, list)
    for assertion in assertions:
        for provenance in assertion["provenance"]:
            provenance["source_artifact_hash"] = source_hashes[provenance["evidence_id"]]
    return value


def copy_bridge_inputs(destination: Path) -> None:
    relative_paths = (
        CERTIFICATE_RELATIVE_PATH,
        "tests/fixtures/phase1/c1_pre_request.json",
        "tests/fixtures/phase1/c2_query_request.json",
        "tests/fixtures/phase1/fixed_select_request.json",
        "tests/fixtures/phase1/invalid_repair_case.json",
        "tests/fixtures/phase1/c1_pre_output.json",
    )
    for relative in relative_paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)


def test_certificate_recomputes_exact_fixture_evidence_record_and_text_hashes() -> None:
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FIXTURES.glob("*.json")
    }
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)

    assert bridge.manifest_sha256 == (
        "4be84e3dbb09d276bea6237da140c35c09ab73714bb8976035a8d38c3885a820"
    )
    assert bridge.certificate.shared_legacy_evidence_sha256 == SHARED_EVIDENCE_HASH
    assert len(bridge.certificate.evidence_bindings) == 7
    assert bridge.certificate.query_blind is True
    assert bridge.certificate.gold_free is True
    assert bridge.certificate.fixture_bytes_modified is False
    assert bridge.certificate.cpu_semantic_inference_performed is False
    assert bridge.certificate.production_reuse_forbidden is True
    assert bridge.certificate.development_reuse_forbidden is True
    assert bridge.certificate.held_out_reuse_forbidden is True
    assert bridge.certificate.case_study_reuse_forbidden is True
    for call in phase1_acceptance_calls():
        resolved = bridge.resolve_call(
            call_id=call.call_id,
            condition=call.condition,
            request_fixture=call.request_fixture,
        )
        assert resolved.legacy_evidence_sha256 == SHARED_EVIDENCE_HASH
        assert len(resolved.evidence) == 7
        assert all(
            item.provenance.source_artifact_hash == SHARED_EVIDENCE_HASH
            for item in resolved.evidence
        )
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FIXTURES.glob("*.json")
    }
    assert after == before


@pytest.mark.parametrize(
    ("call_id", "condition", "fixture"),
    (
        ("fallback-c1-01", ConditionName.C1_LLM_PRE, "c1_pre_request.json"),
        ("fallback-c2-01", ConditionName.C2_LLM_QUERY, "c2_query_request.json"),
        ("fallback-c2-02", ConditionName.C2_LLM_QUERY, "c2_query_request.json"),
        ("fallback-fixed-01", ConditionName.A_FIXED_SELECT, "fixed_select_request.json"),
    ),
)
def test_certificate_scope_includes_only_registered_fallback_calls(
    call_id: str,
    condition: ConditionName,
    fixture: str,
) -> None:
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    resolved = bridge.resolve_call(
        call_id=call_id,
        condition=condition,
        request_fixture=f"tests/fixtures/phase1/{fixture}",
    )
    assert resolved.call_id == call_id
    with pytest.raises(ValueError, match="outside its frozen scope"):
        bridge.resolve_call(
            call_id=f"unregistered-{call_id}",
            condition=condition,
            request_fixture=f"tests/fixtures/phase1/{fixture}",
        )


def test_bridge_rejects_self_consistent_certificate_metadata_tampering(tmp_path: Path) -> None:
    copy_bridge_inputs(tmp_path)
    certificate_path = tmp_path / CERTIFICATE_RELATIVE_PATH
    value = json.loads(certificate_path.read_text(encoding="utf-8"))
    value["evidence_bindings"][0]["source"]["locator"] = "changed-locator"
    immutable = {key: item for key, item in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = canonical_sha256(immutable)
    certificate_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic source metadata"):
        Phase1LegacyEvidenceProvenanceBridge.load(tmp_path)


def test_bridge_rejects_fixture_mutation_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe path"):
        Phase1LegacyEvidenceProvenanceBridge.load(
            ROOT,
            certificate_relative_path="../phase1_legacy_evidence_provenance.json",
        )
    copy_bridge_inputs(tmp_path)
    c1 = tmp_path / "tests/fixtures/phase1/c1_pre_request.json"
    c1.write_bytes(c1.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fixture bytes changed"):
        Phase1LegacyEvidenceProvenanceBridge.load(tmp_path)

    symlink_root = tmp_path / "symlink-case"
    copy_bridge_inputs(symlink_root)
    symlink = symlink_root / "tests/fixtures/phase1/c1_pre_request.json"
    symlink.unlink()
    symlink.symlink_to(ROOT / "tests/fixtures/phase1/c1_pre_request.json")
    with pytest.raises(ValueError, match="refuses symlinked"):
        Phase1LegacyEvidenceProvenanceBridge.load(symlink_root)


def test_wire_request_adds_counted_boundary_checked_provenance_without_fixture_mutation() -> None:
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    fixture_hashes = {
        binding.relative_path: hashlib.sha256(
            (ROOT / binding.relative_path).read_bytes()
        ).hexdigest()
        for binding in bridge.certificate.fixture_bindings
    }
    c2_call = phase1_acceptance_calls()[2]
    repair_call = phase1_acceptance_calls()[7]
    c2_request = build_acceptance_request(
        root=ROOT,
        call=c2_call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest(),
        legacy_provenance_bridge=bridge,
    )
    repair_request = build_acceptance_request(
        root=ROOT,
        call=repair_call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest(),
        legacy_provenance_bridge=bridge,
    )
    c2_payload = json.loads(c2_request.messages[1].content)
    repair_payload = json.loads(repair_request.messages[1].content)
    section = c2_payload["legacy_evidence_provenance"]
    assert section == repair_payload["legacy_evidence_provenance"]
    assert section["certificate_sha256"] == bridge.manifest_sha256
    assert section["records"][0]["provenance"]["source_artifact_hash"] == (
        SHARED_EVIDENCE_HASH
    )
    assert any(
        item.name == "legacy_evidence_provenance" for item in c2_request.packing.sections
    )
    assert c2_request.rendered_input_token_count <= c2_request.decoding.maximum_input_tokens
    assert repair_request.rendered_input_token_count <= repair_request.decoding.maximum_input_tokens
    assert fixture_hashes == {
        binding.relative_path: hashlib.sha256(
            (ROOT / binding.relative_path).read_bytes()
        ).hexdigest()
        for binding in bridge.certificate.fixture_bindings
    }


def test_live_c1_and_c2_require_emitted_source_hashes_but_accept_source_critical_values() -> None:
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    for call_index, output_name in ((0, "c1_pre_output.json"), (2, "c2_query_output.json")):
        call = phase1_acceptance_calls()[call_index]
        with pytest.raises(ValueError, match="source locator/hash"):
            validate_acceptance_generation(
                root=ROOT,
                call=call,
                parsed_object=load_output(output_name),
                legacy_provenance_bridge=bridge,
            )
        output = source_bound_output(output_name, bridge=bridge, call_index=call_index)
        first = output["instance_graph"]["assertions"][0]["provenance"][0]
        first["provenance_id"] = "assertion-specific-id"
        first["extraction_method"] = "assertion-specific extraction"
        first["confidence"] = 0.5
        audit = validate_acceptance_generation(
            root=ROOT,
            call=call,
            parsed_object=output,
            legacy_provenance_bridge=bridge,
        )
        assert audit["raw_draft_sha256"] == audit["effective_validation_draft_sha256"]
        assert audit["effective_source_hash_insertions"] == []
        assert audit["legacy_evidence_provenance_certificate_sha256"] == (
            bridge.manifest_sha256
        )


def test_legacy_evidence_without_explicit_bridge_still_fails_strict_validation() -> None:
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    output = source_bound_output("c2_query_output.json", bridge=bridge, call_index=2)
    with pytest.raises(ValueError, match="production evidence lacks"):
        validate_acceptance_generation(
            root=ROOT,
            call=phase1_acceptance_calls()[2],
            parsed_object=output,
        )


def test_fixed_select_is_checked_raw_then_gets_only_missing_source_hashes() -> None:
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    call = phase1_acceptance_calls()[5]
    request = build_acceptance_request(
        root=ROOT,
        call=call,
        tokenizer=FakeTokenizer(),
        tokenizer_manifest=tokenizer_manifest(),
        legacy_provenance_bridge=bridge,
    )
    section = json.loads(request.messages[1].content)["legacy_evidence_provenance"]
    assert section["records"][0]["provenance"]["source_artifact_hash"] is None
    assert section["records"][0]["effective_validation_source_artifact_hash"] == (
        SHARED_EVIDENCE_HASH
    )
    assert "Preserve the sealed ontology's raw provenance exactly" in section["output_rule"]
    raw = load_output("fixed_select_output.json")
    raw_hash = OntologyDraft.model_validate(raw).content_hash
    audit = validate_acceptance_generation(
        root=ROOT,
        call=call,
        parsed_object=raw,
        legacy_provenance_bridge=bridge,
    )
    assert audit["raw_draft_sha256"] == raw_hash
    assert audit["effective_validation_draft_sha256"] != raw_hash
    assert len(audit["effective_source_hash_insertions"]) == 3

    changed = copy.deepcopy(raw)
    changed["instance_graph"]["assertions"][0]["provenance"][0]["locator"] = "changed"
    with pytest.raises(ValueError):
        validate_acceptance_generation(
            root=ROOT,
            call=call,
            parsed_object=changed,
            legacy_provenance_bridge=bridge,
        )

    nonnull = copy.deepcopy(raw)
    nonnull["instance_graph"]["assertions"][0]["provenance"][0][
        "source_artifact_hash"
    ] = SHARED_EVIDENCE_HASH
    with pytest.raises(ValueError):
        validate_acceptance_generation(
            root=ROOT,
            call=call,
            parsed_object=nonnull,
            legacy_provenance_bridge=bridge,
        )


def test_historical_reference_helper_rejects_conflicts_and_changes_only_missing_hashes() -> None:
    bridge = Phase1LegacyEvidenceProvenanceBridge.load(ROOT)
    call = phase1_acceptance_calls()[0]
    resolved = bridge.resolve_call(
        call_id=call.call_id,
        condition=call.condition,
        request_fixture=call.request_fixture,
    )
    raw = OntologyDraft.model_validate(load_output("c1_pre_output.json"))
    effective = bridge.effective_draft(
        raw_draft=raw,
        resolved=resolved,
        purpose="historical_reference",
    )
    assert effective.raw_draft_sha256 == raw.content_hash
    assert effective.effective_draft_sha256 != raw.content_hash
    assert effective.inserted_source_hash_paths
    assert all(
        provenance.source_artifact_hash == SHARED_EVIDENCE_HASH
        for assertion in effective.effective_draft.instance_graph.assertions
        for provenance in assertion.provenance
    )

    conflicting = load_output("c1_pre_output.json")
    conflicting["instance_graph"]["assertions"][0]["provenance"][0][
        "source_artifact_hash"
    ] = "f" * 64
    with pytest.raises(ValueError, match="conflicting source hash"):
        bridge.effective_draft(
            raw_draft=OntologyDraft.model_validate(conflicting),
            resolved=resolved,
            purpose="historical_reference",
        )
