"""Materialize a Phase 7 source registry without hand-copying hashes or row counts.

The recipe names already-frozen artifacts and declarative table filters.  This module
computes physical/logical hashes and CSV cardinalities from those immutable bytes,
then validates the exact registry contract used by :mod:`phase7_compiler`.  It never
discovers study outputs and never computes or edits scientific values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints, model_validator

from story_projection_onto.contracts import (
    canonical_sha256 as canonical_record_sha256,
)
from story_projection_onto.phase7_compiler import (
    CommunityReviewArtifactBinding,
    ErrorReviewArtifactBinding,
    ExactRowFilter,
    Phase7CompilationError,
    Phase7CompilerConfiguration,
    Phase7SourceRegistry,
    Phase7TableBinding,
    PhaseInputState,
    QualitativeProducerArtifactBinding,
    SupplementalMetricSourceBinding,
    _inspect_supplemental_metric_source_csv,
    _validate_final_accounting_table_selection,
    _verify_phase4_table_manifest_binding,
    _verify_registry_contracts,
    load_phase7_source_registry,
)
from story_projection_onto.report_ingestion import (
    MeasurementDomain,
    PredecessorState,
    SourceArtifactSpec,
    SourceFamily,
)
from story_projection_onto.reporting import (
    REGISTERED_COMPLETE_TABLE_IDS,
    ReportStatus,
    canonical_sha256,
)

REGISTRY_RECIPE_SCHEMA_VERSION = "1.0.0"
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
RelativePath = Annotated[str, StringConstraints(min_length=1, max_length=500)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceArtifactRecipe(FrozenModel):
    artifact_id: Identifier
    family: SourceFamily
    relative_path: RelativePath
    media_type: Literal[
        "application/json",
        "application/jsonl",
        "application/x-sqlite3",
        "application/pdf",
        "image/png",
        "text/csv",
    ]
    release_class: Literal["public", "restricted"]
    logical_hash_field: Literal[
        "none", "content_hash", "manifest_sha256", "policy_sha256"
    ] = "none"
    logical_hash_mode: Literal[
        "none", "declared", "canonical_without_field", "immutable_record"
    ] = "none"
    required_json_fields: tuple[Identifier, ...] = ()
    measurement_domains: tuple[MeasurementDomain, ...] = ()

    @model_validator(mode="after")
    def logical_contract_is_coherent(self) -> Self:
        absent = self.logical_hash_field == "none"
        if absent != (self.logical_hash_mode == "none"):
            raise ValueError("logical hash field and mode must either both be absent or present")
        if (
            self.logical_hash_mode == "immutable_record"
            and self.logical_hash_field != "content_hash"
        ):
            raise ValueError("immutable-record hashing requires a content_hash field")
        if self.media_type != "application/json" and (
            not absent or self.required_json_fields
        ):
            raise ValueError("only JSON artifacts may declare logical JSON contracts")
        return self


class PredecessorRecipe(FrozenModel):
    family: SourceFamily
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    artifacts: tuple[SourceArtifactRecipe, ...] = ()

    @model_validator(mode="after")
    def family_and_status_are_coherent(self) -> Self:
        if self.status is ReportStatus.NOT_APPLICABLE:
            raise ValueError("all registered predecessor families are applicable")
        if self.status is ReportStatus.COMPLETE and not self.artifacts:
            raise ValueError("a complete predecessor requires immutable artifacts")
        if any(item.family is not self.family for item in self.artifacts):
            raise ValueError("predecessor recipe contains an artifact from another family")
        return self


class SupplementalMetricSourceRecipe(FrozenModel):
    source_role: Identifier
    source_artifact_id: Identifier
    table_manifest_artifact_id: Identifier


class TableBindingRecipe(FrozenModel):
    table_id: Identifier
    status: ReportStatus
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    scope: Literal["interim", "final"]
    source_artifact_ids: tuple[Identifier, ...] = ()
    source_table_artifact_id: Identifier | None = None
    producer_manifest_artifact_id: Identifier | None = None
    source_receipt_artifact_id: Identifier | None = None
    row_filters: tuple[ExactRowFilter, ...] = ()
    singleton_join_artifact_id: Identifier | None = None
    supplemental_metric_sources: tuple[SupplementalMetricSourceRecipe, ...] = ()

    @model_validator(mode="after")
    def availability_is_coherent(self) -> Self:
        generated = self.table_id in {"study_status", "qualitative_examples"}
        if self.status is ReportStatus.NOT_APPLICABLE:
            raise ValueError("all registered tables are applicable")
        if self.status is not ReportStatus.COMPLETE and any(
            (
                self.source_artifact_ids,
                self.source_table_artifact_id,
                self.producer_manifest_artifact_id,
                self.source_receipt_artifact_id,
                self.row_filters,
                self.singleton_join_artifact_id,
                self.supplemental_metric_sources,
            )
        ):
            raise ValueError("an unavailable table recipe cannot claim source material")
        if (
            self.status is ReportStatus.COMPLETE
            and self.table_id != "study_status"
            and not self.source_artifact_ids
        ):
            raise ValueError("an available scientific table requires source lineage")
        if (
            self.status is ReportStatus.COMPLETE
            and not generated
            and self.source_table_artifact_id is None
        ):
            raise ValueError("an available canonical table requires a source CSV")
        if generated and any(
            (
                self.source_table_artifact_id,
                self.producer_manifest_artifact_id,
                self.source_receipt_artifact_id,
                self.row_filters,
                self.singleton_join_artifact_id,
                self.supplemental_metric_sources,
            )
        ):
            raise ValueError("generated tables cannot name a source CSV or filters")
        if self.source_receipt_artifact_id is not None and (
            self.source_receipt_artifact_id not in self.source_artifact_ids
        ):
            raise ValueError("the source receipt must be part of table lineage")
        if self.producer_manifest_artifact_id is not None and (
            self.producer_manifest_artifact_id not in self.source_artifact_ids
        ):
            raise ValueError("the producer manifest must be part of table lineage")
        roles = tuple(item.source_role for item in self.supplemental_metric_sources)
        if roles != tuple(sorted(set(roles))):
            raise ValueError("supplemental metric recipe roles must be sorted and unique")
        return self


class Phase7RegistryRecipe(FrozenModel):
    schema_version: Literal["1.0.0"] = REGISTRY_RECIPE_SCHEMA_VERSION
    registry_id: Identifier
    compiled_at_utc: AwareDatetime
    study_id: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    status_reason: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    code_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    dirty_worktree: bool
    model_repository: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None
    model_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")] | None
    predecessors: tuple[PredecessorRecipe, ...]
    phases: tuple[PhaseInputState, ...]
    tables: tuple[TableBindingRecipe, ...]
    qualitative_candidate_set_artifact_id: Identifier | None = None
    qualitative_producer_artifacts: QualitativeProducerArtifactBinding | None = None
    error_review_artifacts: ErrorReviewArtifactBinding | None = None
    community_review_artifacts: CommunityReviewArtifactBinding | None = None
    independent_review_completion_manifest_artifact_id: Identifier | None = None
    public_reviewed_gold_manifest_artifact_id: Identifier | None = None
    public_artifact_ids: tuple[Identifier, ...] = ()
    recipe_sha256: Sha256Digest

    @model_validator(mode="after")
    def exact_inventory(self) -> Self:
        families = [item.family for item in self.predecessors]
        if len(families) != len(set(families)) or set(families) != set(SourceFamily):
            raise ValueError("registry recipe requires each predecessor family exactly once")
        phase_ids = [getattr(item, "phase_id", None) for item in self.phases]
        if len(phase_ids) != len(set(phase_ids)) or set(phase_ids) != {
            f"phase_{number}" for number in range(1, 8)
        }:
            raise ValueError("registry recipe requires all seven phases")
        table_ids = [item.table_id for item in self.tables]
        if len(table_ids) != len(set(table_ids)) or set(table_ids) != set(
            REGISTERED_COMPLETE_TABLE_IDS
        ):
            raise ValueError("registry recipe requires every registered table")
        return self


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _real_source_root(root: Path) -> Path:
    """Return one existing directory whose full ancestry contains no symlink."""

    lexical = Path(os.path.abspath(root))
    current = lexical
    while True:
        if current.is_symlink():
            raise Phase7CompilationError(
                f"symlinked registry root ancestry is prohibited: {root}"
            )
        if current.parent == current:
            break
        current = current.parent
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise Phase7CompilationError(f"missing registry source root: {root}") from error
    if resolved != lexical or not resolved.is_dir():
        raise Phase7CompilationError(f"registry source root must be a real directory: {root}")
    return resolved


def _safe_registry_path(
    root: Path,
    candidate: Path,
    *,
    label: str,
    require_file: bool,
) -> Path:
    """Enforce lexical and resolved containment plus a symlink-free ancestry."""

    root = _real_source_root(root)
    if ".." in candidate.parts:
        raise Phase7CompilationError(f"{label} contains parent traversal: {candidate}")
    absolute = candidate if candidate.is_absolute() else root / candidate
    lexical = Path(os.path.abspath(absolute))
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise Phase7CompilationError(
            f"{label} must remain inside the registry source root: {candidate}"
        ) from error
    if relative == Path("."):
        raise Phase7CompilationError(f"{label} must name a file below the source root")
    probe = root
    for component in relative.parts:
        probe /= component
        if probe.is_symlink():
            raise Phase7CompilationError(f"{label} has a symlinked ancestor: {candidate}")
    try:
        resolved = lexical.resolve(strict=require_file)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        state = "missing" if require_file else "unsafe"
        raise Phase7CompilationError(f"{state} {label}: {candidate}") from error
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise Phase7CompilationError(f"{label} must be one regular file: {candidate}")
    return lexical


def _safe_source(root: Path, relative_path: str) -> Path:
    if "\\" in relative_path:
        raise Phase7CompilationError(f"backslash is prohibited in source path: {relative_path}")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise Phase7CompilationError(f"unsafe registry source path: {relative_path}")
    return _safe_registry_path(
        root,
        root.joinpath(*relative.parts),
        label="registry source",
        require_file=True,
    )


def _load_self_hashed_recipe(path: Path) -> Phase7RegistryRecipe:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase7CompilationError(f"invalid Phase 7 registry recipe: {error}") from error
    if not isinstance(payload, dict):
        raise Phase7CompilationError("Phase 7 registry recipe must be one JSON object")
    supplied = payload.get("recipe_sha256")
    immutable = {key: value for key, value in payload.items() if key != "recipe_sha256"}
    if supplied != canonical_sha256(immutable):
        raise Phase7CompilationError("Phase 7 registry recipe canonical self-hash mismatch")
    return Phase7RegistryRecipe.model_validate(payload)


def _materialize_artifact(root: Path, recipe: SourceArtifactRecipe) -> SourceArtifactSpec:
    path = _safe_source(root, recipe.relative_path)
    logical_hash: str | None = None
    if recipe.media_type == "application/json" and (
        recipe.logical_hash_field != "none" or recipe.required_json_fields
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Phase7CompilationError(
                f"invalid predecessor JSON: {recipe.artifact_id}"
            ) from error
        if not isinstance(payload, dict):
            raise Phase7CompilationError(
                f"predecessor JSON is not an object: {recipe.artifact_id}"
            )
        missing = set(recipe.required_json_fields) - set(payload)
        if missing:
            raise Phase7CompilationError(
                f"predecessor {recipe.artifact_id} lacks fields: {', '.join(sorted(missing))}"
            )
        if recipe.logical_hash_field != "none":
            supplied = payload.get(recipe.logical_hash_field)
            if not isinstance(supplied, str) or len(supplied) != 64:
                raise Phase7CompilationError(
                    f"predecessor logical hash is absent: {recipe.artifact_id}"
                )
            logical_hash = supplied
            if recipe.logical_hash_mode == "canonical_without_field":
                immutable = {
                    key: value
                    for key, value in payload.items()
                    if key != recipe.logical_hash_field
                }
                if canonical_sha256(immutable) != supplied:
                    raise Phase7CompilationError(
                        f"predecessor canonical hash is invalid: {recipe.artifact_id}"
                    )
            elif (
                recipe.logical_hash_mode == "immutable_record"
                and canonical_record_sha256(payload) != supplied
            ):
                raise Phase7CompilationError(
                    f"predecessor immutable-record hash is invalid: {recipe.artifact_id}"
                )
    return SourceArtifactSpec(
        **recipe.model_dump(mode="python"),
        file_sha256=_file_sha256(path),
        logical_hash=logical_hash,
    )


def _canonical_csv_counts(
    path: Path, filters: tuple[ExactRowFilter, ...]
) -> tuple[int, int, tuple[str, ...]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise Phase7CompilationError(f"source table is not canonical UTF-8/LF CSV: {path}")
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise Phase7CompilationError(f"source table is not UTF-8: {path}") from error
    columns = tuple(reader.fieldnames or ())
    rows = tuple(dict(row) for row in reader)
    if not columns or len(columns) != len(set(columns)):
        raise Phase7CompilationError(f"source table has missing/duplicate columns: {path}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise Phase7CompilationError(f"source table has a ragged row: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    if buffer.getvalue().encode("utf-8") != raw:
        raise Phase7CompilationError(f"source table serialization is not canonical: {path}")
    unknown = {item.column for item in filters} - set(columns)
    if unknown:
        raise Phase7CompilationError(
            "source table lacks row-filter columns: " + ", ".join(sorted(unknown))
        )
    selected = sum(
        all(item.accepts(row[item.column]) for item in filters) for row in rows
    )
    return len(rows), selected, columns


def materialize_phase7_source_registry(
    *,
    source_root: Path,
    configuration_path: Path,
    recipe_path: Path,
) -> Phase7SourceRegistry:
    """Build one validated registry entirely from recipe paths and source bytes."""

    source_root = _real_source_root(source_root)
    recipe_path = _safe_registry_path(
        source_root,
        recipe_path,
        label="Phase 7 registry recipe",
        require_file=True,
    )
    configuration = Phase7CompilerConfiguration.load(
        configuration_path, source_root=source_root
    )
    recipe = _load_self_hashed_recipe(recipe_path)
    materialized_predecessors: list[PredecessorState] = []
    for predecessor in recipe.predecessors:
        materialized_predecessors.append(
            PredecessorState(
                family=predecessor.family,
                status=predecessor.status,
                reason=predecessor.reason,
                artifacts=tuple(
                    _materialize_artifact(source_root, item)
                    for item in predecessor.artifacts
                ),
            )
        )
    artifacts = {
        item.artifact_id: item
        for predecessor in materialized_predecessors
        for item in predecessor.artifacts
    }
    contracts = {item.table_id: item for item in configuration.table_contracts}
    bindings: list[Phase7TableBinding] = []
    for table in recipe.tables:
        _validate_final_accounting_table_selection(
            table_id=table.table_id,
            row_filters=table.row_filters,
            producer_manifest_artifact_id=table.producer_manifest_artifact_id,
            singleton_join_artifact_id=table.singleton_join_artifact_id,
            singleton_join_row_count=None,
            supplemental_metric_sources=table.supplemental_metric_sources,
        )
        source_rows: int | None = None
        output_rows: int | None = None
        singleton_rows: Literal[1] | None = None
        supplemental_bindings: list[SupplementalMetricSourceBinding] = []
        if table.status is ReportStatus.COMPLETE and table.source_table_artifact_id:
            source = artifacts.get(table.source_table_artifact_id)
            if source is None or source.media_type != "text/csv":
                raise Phase7CompilationError(
                    f"table source does not resolve to a CSV: {table.table_id}"
                )
            source_rows, output_rows, columns = _canonical_csv_counts(
                _safe_source(source_root, source.relative_path), table.row_filters
            )
            missing = set(contracts[table.table_id].required_columns) - set(columns)
            if table.singleton_join_artifact_id:
                singleton = artifacts.get(table.singleton_join_artifact_id)
                if singleton is not None and singleton.media_type == "text/csv":
                    _, _, singleton_columns = _canonical_csv_counts(
                        _safe_source(source_root, singleton.relative_path), ()
                    )
                    missing -= set(singleton_columns)
            if missing:
                raise Phase7CompilationError(
                    f"source table {table.table_id} lacks columns: "
                    + ", ".join(sorted(missing))
                )
        if table.status is ReportStatus.COMPLETE and table.singleton_join_artifact_id:
            singleton = artifacts.get(table.singleton_join_artifact_id)
            if singleton is None or singleton.media_type != "text/csv":
                raise Phase7CompilationError(
                    f"singleton source does not resolve to a CSV: {table.table_id}"
                )
            singleton_count, _, _ = _canonical_csv_counts(
                _safe_source(source_root, singleton.relative_path), ()
            )
            if singleton_count != 1:
                raise Phase7CompilationError(
                    f"singleton table must contain exactly one row: {table.table_id}"
                )
            singleton_rows = 1
        contract_sources = {
            item.source_role: item
            for item in contracts[table.table_id].supplemental_metric_sources
        }
        for supplemental in table.supplemental_metric_sources:
            contract = contract_sources.get(supplemental.source_role)
            source = artifacts.get(supplemental.source_artifact_id)
            manifest = artifacts.get(supplemental.table_manifest_artifact_id)
            if contract is None:
                raise Phase7CompilationError(
                    f"unknown supplemental metric source role: {supplemental.source_role}"
                )
            if source is None or source.media_type != "text/csv":
                raise Phase7CompilationError(
                    "supplemental metric source does not resolve to CSV: "
                    f"{supplemental.source_role}"
                )
            if manifest is None or manifest.media_type != "application/json":
                raise Phase7CompilationError(
                    "supplemental metric table manifest does not resolve to JSON: "
                    f"{supplemental.source_role}"
                )
            total, selected, metric_counts, columns, metric_version_hash = (
                _inspect_supplemental_metric_source_csv(
                    _safe_source(source_root, source.relative_path),
                    expected_hash=source.file_sha256,
                    contract=contract,
                )
            )
            _verify_phase4_table_manifest_binding(
                _safe_source(source_root, manifest.relative_path),
                expected_manifest_hash=manifest.file_sha256,
                source=source,
                source_columns=columns,
                source_row_count=total,
                producer_table_id=contract.producer_table_id,
                metric_version_hash=metric_version_hash,
            )
            supplemental_bindings.append(
                SupplementalMetricSourceBinding(
                    source_role=supplemental.source_role,
                    source_artifact_id=supplemental.source_artifact_id,
                    table_manifest_artifact_id=supplemental.table_manifest_artifact_id,
                    source_row_count=total,
                    selected_row_count=selected,
                    metric_row_counts=metric_counts,
                )
            )
        bindings.append(
            Phase7TableBinding(
                **table.model_dump(
                    mode="python", exclude={"supplemental_metric_sources"}
                ),
                source_row_count=source_rows,
                output_row_count=output_rows,
                singleton_join_row_count=singleton_rows,
                supplemental_metric_sources=tuple(
                    sorted(supplemental_bindings, key=lambda item: item.source_role)
                ),
            )
        )
    payload: dict[str, Any] = {
        "schema_version": REGISTRY_RECIPE_SCHEMA_VERSION,
        "registry_id": recipe.registry_id,
        "compiled_at_utc": recipe.compiled_at_utc.isoformat().replace("+00:00", "Z"),
        "study_id": recipe.study_id,
        "status_reason": recipe.status_reason,
        "code_revision": recipe.code_revision,
        "dirty_worktree": recipe.dirty_worktree,
        "model_repository": recipe.model_repository,
        "model_revision": recipe.model_revision,
        "predecessors": [item.model_dump(mode="json") for item in materialized_predecessors],
        "phases": [item.model_dump(mode="json") for item in recipe.phases],
        "tables": [item.model_dump(mode="json") for item in bindings],
        "qualitative_candidate_set_artifact_id": (
            recipe.qualitative_candidate_set_artifact_id
        ),
        "qualitative_producer_artifacts": (
            recipe.qualitative_producer_artifacts.model_dump(mode="json")
            if recipe.qualitative_producer_artifacts is not None
            else None
        ),
        "error_review_artifacts": (
            recipe.error_review_artifacts.model_dump(mode="json")
            if recipe.error_review_artifacts is not None
            else None
        ),
        "community_review_artifacts": (
            recipe.community_review_artifacts.model_dump(mode="json")
            if recipe.community_review_artifacts is not None
            else None
        ),
        "independent_review_completion_manifest_artifact_id": (
            recipe.independent_review_completion_manifest_artifact_id
        ),
        "public_reviewed_gold_manifest_artifact_id": (
            recipe.public_reviewed_gold_manifest_artifact_id
        ),
        "public_artifact_ids": list(recipe.public_artifact_ids),
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    registry = Phase7SourceRegistry.model_validate(payload)
    _verify_registry_contracts(registry, configuration, source_root=source_root)
    return registry


def write_phase7_source_registry(
    *,
    source_root: Path,
    configuration_path: Path,
    recipe_path: Path,
    output_path: Path,
) -> Phase7SourceRegistry:
    source_root = _real_source_root(source_root)
    output_path = _safe_registry_path(
        source_root,
        output_path,
        label="Phase 7 registry output",
        require_file=False,
    )
    registry = materialize_phase7_source_registry(
        source_root=source_root,
        configuration_path=configuration_path,
        recipe_path=recipe_path,
    )
    payload = (registry.model_dump_json(indent=2) + "\n").encode("utf-8")
    if output_path.exists():
        if not output_path.is_file() or output_path.read_bytes() != payload:
            raise Phase7CompilationError("append-only Phase 7 source registry changed")
        return registry
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = _safe_registry_path(
        source_root,
        output_path,
        label="Phase 7 registry output",
        require_file=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".partial", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            if not output_path.is_file() or output_path.read_bytes() != payload:
                raise Phase7CompilationError(
                    "concurrent append-only Phase 7 source registry drift"
                ) from None
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return registry


def verify_phase7_source_registry(
    *,
    source_root: Path,
    configuration_path: Path,
    recipe_path: Path,
    registry_path: Path,
) -> Phase7SourceRegistry:
    source_root = _real_source_root(source_root)
    registry_path = _safe_registry_path(
        source_root,
        registry_path,
        label="Phase 7 source registry",
        require_file=True,
    )
    expected = materialize_phase7_source_registry(
        source_root=source_root,
        configuration_path=configuration_path,
        recipe_path=recipe_path,
    )
    observed = load_phase7_source_registry(registry_path)
    if observed != expected:
        raise Phase7CompilationError(
            "Phase 7 source registry does not reproduce from its recipe and artifacts"
        )
    return observed


__all__ = [
    "CommunityReviewArtifactBinding",
    "ErrorReviewArtifactBinding",
    "Phase7RegistryRecipe",
    "PredecessorRecipe",
    "QualitativeProducerArtifactBinding",
    "SourceArtifactRecipe",
    "TableBindingRecipe",
    "materialize_phase7_source_registry",
    "verify_phase7_source_registry",
    "write_phase7_source_registry",
]
