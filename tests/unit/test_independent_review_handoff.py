from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
from pathlib import Path

import pytest

from scripts.render_independent_review_handoff import main as render_handoff_main
from story_projection_onto.scorer_only.independent_review_handoff import (
    MANIFEST_FILE,
    PACKET_FILE,
    RESPONSE_TEMPLATE_FILE,
    IndependentReviewHandoffError,
    IndependentReviewHandoffManifest,
    build_independent_review_handoff,
    materialize_independent_review_handoff,
)
from story_projection_onto.synthetic_benchmark import (
    BlindIndependentReviewPackage,
    ReviewCriterion,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PATH = ROOT / "data/synthetic/scorer_only/review/blind_review_package.json"
RESPONSE_SCHEMA_PATH = ROOT / "data/synthetic/scorer_only/review/reviewer_response.schema.json"


def _package() -> BlindIndependentReviewPackage:
    return BlindIndependentReviewPackage.model_validate_json(PACKAGE_PATH.read_bytes())


def _tsv_rows(value: bytes) -> list[dict[str, str]]:
    data_lines = [line for line in value.decode("utf-8").splitlines() if not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(data_lines)), delimiter="\t"))


def test_checked_in_package_renders_all_nine_projections_and_exact_questions() -> None:
    package = _package()
    handoff = build_independent_review_handoff(
        package_path=PACKAGE_PATH,
        response_schema_path=RESPONSE_SCHEMA_PATH,
    )
    repeated = build_independent_review_handoff(
        package_path=PACKAGE_PATH,
        response_schema_path=RESPONSE_SCHEMA_PATH,
    )

    assert handoff == repeated
    assert handoff.packet_markdown == repeated.packet_markdown
    assert handoff.response_template_tsv == repeated.response_template_tsv
    assert handoff.manifest.package_hash == package.content_hash
    assert handoff.manifest.world_count == 3
    assert handoff.manifest.projection_count == 9
    assert handoff.manifest.review_item_count == 72
    assert handoff.manifest.condition_blind is True
    assert handoff.manifest.contains_method_outputs is False
    assert handoff.manifest.contains_reviewer_judgments is False
    assert handoff.manifest.response_dispositions_prefilled is False

    packet = handoff.packet_markdown.decode("utf-8")
    assert packet.count("## Blind world ") == 3
    assert packet.count("### Projection ") == 9
    assert packet.count("#### Review questions") == 9
    assert packet.count("##### Proposed qualified assertions") == 9
    assert packet.count("##### Proposed identity partitions") == 9
    assert packet.count("##### Proposed event objects") == 9
    assert packet.count("##### Rare-pivotal support path") == 9
    assert packet.count("##### Proposed communities") == 9
    assert packet.count("##### Proposed contrast behavior") == 9
    assert packet.count("##### Proposed permissible alternatives") == 9
    assert "proposed gold interpretation to review" in packet
    assert "Holder-relative epistemic status" in packet
    assert "Validity time" in packet
    assert "Revelation position" in packet

    for world in package.worlds:
        assert world.blind_world_id in packet
        for evidence in world.evidence:
            assert str(evidence["evidence_id"]) in packet
            assert str(evidence["text"]) in packet
        for projection in world.projections:
            assert projection.blind_projection_id in packet
            for item in projection.review_items:
                assert packet.count(item.review_item_id) == 1
                assert item.prompt in packet

    condition_output_markers = (
        "syn-test",
        "syn-dev",
        "c0_classical_pre",
        "c1_llm_pre",
        "c2_llm_query",
        "a_fixed_select",
        "a-fixedselect",
    )
    rendered = (handoff.packet_markdown + handoff.response_template_tsv).decode("utf-8").casefold()
    assert all(marker not in rendered for marker in condition_output_markers)


def test_response_worksheet_is_blank_complete_and_directly_bound_to_package() -> None:
    package = _package()
    handoff = build_independent_review_handoff(
        package_path=PACKAGE_PATH,
        response_schema_path=RESPONSE_SCHEMA_PATH,
    )
    template = handoff.response_template_tsv.decode("utf-8")
    rows = _tsv_rows(handoff.response_template_tsv)

    assert f"# package_id\t{package.package_id}" in template
    assert f"# package_hash\t{package.content_hash}" in template
    assert "# reviewer_pseudonym\t\n" in template
    assert "# reviewed_at_utc\t\n" in template
    assert len(rows) == 72
    assert {row["world_number"] for row in rows} == {"1", "2", "3"}
    assert {row["projection_number"] for row in rows} == {str(value) for value in range(1, 10)}
    assert {row["context_label"] for row in rows} == {"A", "B", "C"}
    assert {row["criterion"] for row in rows} == {criterion.value for criterion in ReviewCriterion}
    assert len({row["blind_projection_id"] for row in rows}) == 9
    assert len({row["review_item_id"] for row in rows}) == 72
    assert all(row["disposition"] == "" for row in rows)
    assert all(row["notes_and_evidence_ids"] == "" for row in rows)

    expected_items = {
        (
            item.review_item_id,
            projection.blind_projection_id,
            item.criterion.value,
            item.prompt,
        )
        for world in package.worlds
        for projection in world.projections
        for item in projection.review_items
    }
    actual_items = {
        (
            row["review_item_id"],
            row["blind_projection_id"],
            row["criterion"],
            row["question"],
        )
        for row in rows
    }
    assert actual_items == expected_items


def test_manifest_binds_exact_rendered_bytes_and_response_schema() -> None:
    handoff = build_independent_review_handoff(
        package_path=PACKAGE_PATH,
        response_schema_path=RESPONSE_SCHEMA_PATH,
    )
    by_name = {item.relative_path: item for item in handoff.manifest.files}
    assert by_name[PACKET_FILE].size_bytes == len(handoff.packet_markdown)
    assert by_name[PACKET_FILE].file_sha256 == hashlib.sha256(handoff.packet_markdown).hexdigest()
    assert by_name[RESPONSE_TEMPLATE_FILE].size_bytes == len(handoff.response_template_tsv)
    assert (
        by_name[RESPONSE_TEMPLATE_FILE].file_sha256
        == hashlib.sha256(handoff.response_template_tsv).hexdigest()
    )
    assert (
        handoff.manifest.package_file_sha256
        == hashlib.sha256(PACKAGE_PATH.read_bytes()).hexdigest()
    )
    assert (
        handoff.manifest.response_schema_file_sha256
        == hashlib.sha256(RESPONSE_SCHEMA_PATH.read_bytes()).hexdigest()
    )
    assert (
        IndependentReviewHandoffManifest.model_validate_json(handoff.file_bytes()[MANIFEST_FILE])
        == handoff.manifest
    )


def test_materialization_is_restricted_append_only_and_exact(tmp_path: Path) -> None:
    handoff = build_independent_review_handoff(
        package_path=PACKAGE_PATH,
        response_schema_path=RESPONSE_SCHEMA_PATH,
    )
    unrestricted = tmp_path / "public" / "handoff"
    with pytest.raises(IndependentReviewHandoffError, match="restricted"):
        materialize_independent_review_handoff(handoff, unrestricted)

    output = tmp_path / "restricted" / "scorer_only" / "handoff"
    assert materialize_independent_review_handoff(handoff, output) == "created"
    assert materialize_independent_review_handoff(handoff, output) == "already_exact"
    assert {item.name for item in output.iterdir()} == {
        PACKET_FILE,
        RESPONSE_TEMPLATE_FILE,
        MANIFEST_FILE,
    }
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    (output / PACKET_FILE).write_text("changed", encoding="utf-8")
    with pytest.raises(IndependentReviewHandoffError, match="differs"):
        materialize_independent_review_handoff(handoff, output)

    partial = tmp_path / "restricted" / "scorer_only" / "partial"
    partial.mkdir()
    (partial / PACKET_FILE).write_bytes(handoff.packet_markdown)
    with pytest.raises(IndependentReviewHandoffError, match="partial"):
        materialize_independent_review_handoff(handoff, partial)


def test_renderer_rejects_a_different_response_schema(tmp_path: Path) -> None:
    changed_schema = tmp_path / "reviewer_response.schema.json"
    changed_schema.write_text("{}\n", encoding="utf-8")
    with pytest.raises(IndependentReviewHandoffError, match="typed response contract"):
        build_independent_review_handoff(
            package_path=PACKAGE_PATH,
            response_schema_path=changed_schema,
        )


def test_cli_dry_run_and_materialization_never_claim_review_completion(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "restricted" / "scorer_only" / "cli-handoff"
    common = [
        "--package",
        str(PACKAGE_PATH),
        "--response-schema",
        str(RESPONSE_SCHEMA_PATH),
        "--output-root",
        str(output),
    ]
    assert render_handoff_main([*common, "--dry-run"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["state"] == "validated"
    assert dry["writes_performed"] is False
    assert dry["contains_reviewer_judgments"] is False
    assert not output.exists()

    assert render_handoff_main([*common, "--materialize"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["state"] == "created"
    assert created["writes_performed"] is True
    assert created["review_item_count"] == 72
    assert "held_out_launch_authorized" not in created

    assert render_handoff_main([*common, "--materialize"]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["state"] == "already_exact"
    assert replay["writes_performed"] is False
