"""Production, lifecycle-free adapter for the frozen development GPU block.

This module contains only gold-free runtime logic.  It binds the tracked
construction configuration, packs complete semantic requests, persists dual
logical/CAS lineage, and exposes the narrow service API consumed by
``DevelopmentRunner``.  Starting or stopping vLLM remains the fallback
controller's exclusive responsibility.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from story_projection_onto.benchmark_runtime import (
    ModelEligibleWorldArtifact,
    QueryRevealArtifact,
    RuntimeStagingManifest,
)
from story_projection_onto.contracts import (
    ConditionName,
    ConstructionRequest,
    FixedOntologyInput,
    ImmutableRecord,
    LegacyModelVisibleEvidenceRecord,
    ModelVisibleEvidenceInput,
    ModelVisibleEvidenceRecord,
    OutputBudgets,
    PreconstructionRequest,
    PrequeryBarrier,
    QueryAccessEvent,
    QueryContext,
    RuntimeIdentifiers,
    Sha256Digest,
    UpperOntology,
    canonical_json,
    canonical_sha256,
    to_model_visible_query,
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
    DevelopmentCallSpec,
    FixedSchemaDerivationReceipt,
    LiveServiceIdentity,
    ServiceCallResult,
    StageReference,
)
from story_projection_onto.gpu_runtime import (
    FALLBACK_MODEL_REVISION,
    FALLBACK_SERVED_MODEL_NAME,
    ChatMessage,
    GenerationResult,
    GuidedJSONRequest,
    TokenizerManifest,
)
from story_projection_onto.llm import (
    CapabilityManifest,
    DecodingManifest,
    PackingReport,
    PackingSection,
    base_condition_output_schema,
    render_condition_system_prompt,
)
from story_projection_onto.query_runtime import AuditedBenchmarkRuntime, AuditedQueryOpening
from story_projection_onto.store import ArtifactStore, QueryAccessRecord
from story_projection_onto.store import ReleaseClass as LedgerReleaseClass

DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG = Path(
    "configs/study/development_construction.json"
)
STRUCTURED_DECODER = "vllm-0.10.2-xgrammar-no-fallback"
PINNED_RUNTIME_VERSION = "vllm-0.10.2"
MODEL_WIRE_ENCODING_ID = "lossless-model-wire-v1"
_WIRE_FIELDS = (
    "abstraction",
    "alias_candidate_ids",
    "anchor_id",
    "candidate_id",
    "confidence",
    "coreference_scores",
    "discourse_position",
    "end_char",
    "event_candidates",
    "evidence",
    "evidence_id",
    "generic_request",
    "lens",
    "mention_candidates",
    "normalized_expression",
    "object_mention_candidate_id",
    "partial_order",
    "passage_order",
    "participant_mention_candidate_ids",
    "provisional_type",
    "relation",
    "relation_phrase_candidates",
    "sentence_order",
    "start_char",
    "subject_mention_candidate_id",
    "surface",
    "surface_hash",
    "surface_phrase",
    "target_candidate_ids",
    "temporal_clues",
    "text",
    "token_order",
    "trigger_end_char",
    "trigger_start_char",
    "trigger_surface",
    "wording",
)
# Globally unique reserved keys keep the codec reversible even where unrelated
# record types use ordinary names such as ``events``, ``start``, or ``aliases``.
_WIRE_ALIASES: Mapping[str, str] = {
    name: f"@{index:02x}" for index, name in enumerate(_WIRE_FIELDS)
} | {"generic_request": "request"}


class DevelopmentAdapterIntegrityError(RuntimeError):
    """A frozen adapter input, durable receipt, or service result changed."""


class WireAliasEntry(ImmutableRecord):
    alias: str = Field(pattern=r"^src[EMRT][0-9]{4}$")
    source_id: str = Field(min_length=1)
    source_kind: Literal["evidence", "mention", "event", "relation", "temporal"]


class ModelWireAliasManifest(ImmutableRecord):
    encoding: Literal["lossless-model-wire-v1"] = MODEL_WIRE_ENCODING_ID
    semantic_request_hash: Sha256Digest
    entries: tuple[WireAliasEntry, ...]

    @model_validator(mode="after")
    def aliases_and_sources_are_bijective(self) -> Self:
        aliases = tuple(item.alias for item in self.entries)
        sources = tuple(item.source_id for item in self.entries)
        if len(aliases) != len(set(aliases)) or len(sources) != len(set(sources)):
            raise ValueError("model wire source aliases must be bijective")
        return self

    @property
    def source_to_alias(self) -> Mapping[str, str]:
        return {item.source_id: item.alias for item in self.entries}

    @property
    def alias_to_source(self) -> Mapping[str, str]:
        return {item.alias: item.source_id for item in self.entries}


@dataclass(frozen=True, slots=True)
class EncodedSemanticRequest:
    sections: Mapping[str, object]
    alias_manifest: ModelWireAliasManifest


def _source_alias_manifest(
    semantic_request: PreconstructionRequest | ConstructionRequest,
) -> ModelWireAliasManifest:
    evidence = (
        semantic_request.evidence
        if isinstance(semantic_request, PreconstructionRequest)
        else semantic_request.packet.evidence
    )
    entries: list[WireAliasEntry] = []
    counters = {kind: 0 for kind in "EMRT"}

    def add(source_id: str, kind_code: str, kind: str) -> None:
        counters[kind_code] += 1
        entries.append(
            WireAliasEntry(
                alias=f"src{kind_code}{counters[kind_code]:04d}",
                source_id=source_id,
                source_kind=kind,
            )
        )

    for record in evidence:
        add(record.evidence_id, "E", "evidence")
        for item in record.mention_candidates:
            add(item.candidate_id, "M", "mention")
        for item in record.event_candidates:
            add(item.candidate_id, "E", "event")
        for item in record.relation_phrase_candidates:
            add(item.candidate_id, "R", "relation")
        for item in record.temporal_clues:
            add(item.clue_id, "T", "temporal")
    return ModelWireAliasManifest(
        semantic_request_hash=semantic_request.content_hash,
        entries=tuple(entries),
    )


def _source_aliases_for_evidence(
    evidence: Sequence[ModelVisibleEvidenceRecord],
) -> Mapping[str, str]:
    """Derive the same deterministic aliases without needing a request hash."""

    counters = {kind: 0 for kind in "EMRT"}
    result: dict[str, str] = {}

    def add(source_id: str, kind_code: str) -> None:
        if source_id in result:
            raise DevelopmentAdapterIntegrityError(
                "model-visible source identifiers must be globally unique"
            )
        counters[kind_code] += 1
        result[source_id] = f"src{kind_code}{counters[kind_code]:04d}"

    for record in evidence:
        add(record.evidence_id, "E")
        for item in record.mention_candidates:
            add(item.candidate_id, "M")
        for item in record.event_candidates:
            add(item.candidate_id, "E")
        for item in record.relation_phrase_candidates:
            add(item.candidate_id, "R")
        for item in record.temporal_clues:
            add(item.clue_id, "T")
    return result


def model_wire_source_alias_bijection(
    evidence: Sequence[ModelVisibleEvidenceRecord],
) -> Mapping[str, str]:
    """Expose the deterministic evidence/candidate alias derivation for audits."""

    return dict(_source_aliases_for_evidence(evidence))


def _alias(source_id: str | None, aliases: Mapping[str, str]) -> str | None:
    if source_id is None:
        return None
    try:
        return aliases[source_id]
    except KeyError as exc:
        raise DevelopmentAdapterIntegrityError(
            f"wire codec encountered an unregistered source ID: {source_id}"
        ) from exc


def _compact_evidence_records(
    evidence: Sequence[object],
    aliases: Mapping[str, str],
) -> list[object]:
    records: list[object] = []
    for untyped in evidence:
        record = cast(Any, untyped)
        discourse = record.discourse_position
        mentions = [
            [
                _alias(item.candidate_id, aliases),
                item.start_char,
                item.end_char,
                item.surface,
                item.provisional_type,
                [_alias(value, aliases) for value in item.alias_candidate_ids],
                {
                    cast(str, _alias(key, aliases)): score
                    for key, score in sorted(item.coreference_scores.items())
                },
            ]
            for item in record.mention_candidates
        ]
        events = [
            [
                _alias(item.candidate_id, aliases),
                item.trigger_start_char,
                item.trigger_end_char,
                item.trigger_surface,
                [_alias(value, aliases) for value in item.participant_mention_candidate_ids],
                item.confidence,
            ]
            for item in record.event_candidates
        ]
        relations = [
            [
                _alias(item.candidate_id, aliases),
                _alias(item.subject_mention_candidate_id, aliases),
                _alias(item.object_mention_candidate_id, aliases),
                item.surface_phrase,
                item.confidence,
            ]
            for item in record.relation_phrase_candidates
        ]
        temporal = [
            [
                _alias(item.clue_id, aliases),
                item.normalized_expression,
                [_alias(value, aliases) for value in item.target_candidate_ids],
                None if item.relation is None else item.relation.value,
                item.confidence,
            ]
            for item in record.temporal_clues
        ]
        records.append(
            [
                _alias(record.evidence_id, aliases),
                record.text,
                [
                    discourse.passage_order,
                    discourse.sentence_order,
                    discourse.token_order,
                ],
                mentions,
                events,
                relations,
                temporal,
            ]
        )
    return records


def _compact_evidence_grounding(
    evidence: Sequence[object],
    aliases: Mapping[str, str],
) -> dict[str, object]:
    """Expose immutable indexed lineage separately from the seven-field text rows.

    Keeping the positional candidate codec unchanged preserves its audited meaning.
    New live requests additionally carry this exact, losslessly decoded mapping so a
    model can copy the authoritative provenance object instead of guessing it.
    """

    grounding: dict[str, object] = {}
    for untyped in evidence:
        if not isinstance(untyped, ModelVisibleEvidenceRecord):
            raise DevelopmentAdapterIntegrityError(
                "legacy evidence without indexed grounding lineage cannot enter a model wire"
            )
        record = untyped
        evidence_alias = _alias(record.evidence_id, aliases)
        if evidence_alias is None:
            raise DevelopmentAdapterIntegrityError(
                "model-visible evidence lacks its registered source alias"
            )
        provenance = record.provenance.model_dump(
            mode="json",
            exclude={"content_hash", "schema_version"},
        )
        provenance["evidence_id"] = evidence_alias
        grounding[evidence_alias] = {
            "passage_id": record.passage_id,
            "text_hash": record.text_hash,
            "record_confidence": record.confidence,
            "provenance": provenance,
        }
    return grounding


def _compact_wire_value(
    value: object,
    *,
    parent_evidence_id: str | None = None,
) -> object:
    """Remove only derivable metadata/defaults and apply a declared key dictionary."""

    if isinstance(value, Mapping):
        inherited_evidence_id = value.get("evidence_id")
        current_evidence_id = (
            inherited_evidence_id if isinstance(inherited_evidence_id, str) else parent_evidence_id
        )
        result: dict[str, object] = {}
        for key, child in value.items():
            if key in {"content_hash", "schema_version"}:
                continue
            if key == "evidence_id" and parent_evidence_id == child:
                continue
            if child is None or child == [] or child == {}:
                continue
            alias = _WIRE_ALIASES.get(key, key)
            result[alias] = _compact_wire_value(
                child,
                parent_evidence_id=current_evidence_id,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _compact_wire_value(item, parent_evidence_id=parent_evidence_id)
            for item in value
        ]
    return value


def model_wire_encoding_manifest(
    alias_manifest: ModelWireAliasManifest | None = None,
) -> dict[str, object]:
    """Self-describe the reversible administrative compaction sent to the model."""

    return {
        "encoding": MODEL_WIRE_ENCODING_ID,
        "aliases": dict(sorted(_WIRE_ALIASES.items())),
        "omitted_derivable_fields": {
            "content_hash": "canonical record hash is recomputed by the typed decoder",
            "schema_version": "typed schema default is reapplied by the decoder",
            "surface_hash": "sha256(UTF-8 surface) is recomputed by the typed decoder",
        },
        "omitted_defaults": ["null", "empty_array", "empty_object"],
        "child_evidence_id": "inherit enclosing evidence record evidence_id",
        "source_alias_layout": {
            "evidence": "srcE####",
            "mention": "srcM####",
            "event": "srcE####",
            "relation": "srcR####",
            "temporal": "srcT####",
        },
        "shared_alias_namespace": {
            "prefix": "srcE",
            "members": ["evidence", "event"],
            "disambiguation": "the persisted bijection records source_kind for every alias",
        },
        "model_output_rule": "cite source aliases; controller restores registered IDs",
        "evidence_grounding_lineage": {
            "join_key": "evidence source alias",
            "fields": [
                "passage_id",
                "text_hash",
                "record_confidence",
                "provenance",
            ],
            "provenance_output_rule": (
                "preserve each cited source locator and source_artifact_hash exactly; "
                "provenance confidence must not exceed record_confidence or indexed "
                "provenance confidence"
            ),
        },
        "alias_manifest_hash": (
            None if alias_manifest is None else alias_manifest.content_hash
        ),
        "semantic_evidence_truncation": False,
    }


def _source_id(alias: object, aliases: Mapping[str, str]) -> str:
    if not isinstance(alias, str) or alias not in aliases:
        raise DevelopmentAdapterIntegrityError(
            "wire decoder encountered an unregistered source alias"
        )
    return aliases[alias]


def decode_compact_evidence_records(
    records: Sequence[object],
    alias_manifest: ModelWireAliasManifest,
    grounding: Mapping[str, object] | None = None,
) -> tuple[ModelVisibleEvidenceInput, ...]:
    """Invert the positional evidence codec and re-run every typed invariant."""

    aliases = alias_manifest.alias_to_source
    grounding_by_evidence_id: dict[str, dict[str, object]] = {}
    if grounding is not None:
        try:
            for evidence_alias, untyped_grounding in grounding.items():
                evidence_id = _source_id(evidence_alias, aliases)
                if not isinstance(untyped_grounding, Mapping):
                    raise DevelopmentAdapterIntegrityError(
                        "compact evidence grounding entry must be an object"
                    )
                expected_keys = {
                    "passage_id",
                    "text_hash",
                    "record_confidence",
                    "provenance",
                }
                if set(untyped_grounding) != expected_keys:
                    raise DevelopmentAdapterIntegrityError(
                        "compact evidence grounding entry has an invalid field set"
                    )
                untyped_provenance = untyped_grounding["provenance"]
                if not isinstance(untyped_provenance, Mapping):
                    raise DevelopmentAdapterIntegrityError(
                        "compact evidence provenance must be an object"
                    )
                provenance = dict(untyped_provenance)
                provenance_evidence_alias = provenance.get("evidence_id")
                if provenance_evidence_alias != evidence_alias:
                    raise DevelopmentAdapterIntegrityError(
                        "compact provenance must name its containing evidence alias"
                    )
                provenance["evidence_id"] = _source_id(
                    provenance_evidence_alias,
                    aliases,
                )
                if evidence_id in grounding_by_evidence_id:
                    raise DevelopmentAdapterIntegrityError(
                        "compact evidence grounding IDs must be unique"
                    )
                grounding_by_evidence_id[evidence_id] = {
                    "passage_id": untyped_grounding["passage_id"],
                    "text_hash": untyped_grounding["text_hash"],
                    "confidence": untyped_grounding["record_confidence"],
                    "provenance": provenance,
                }
        except DevelopmentAdapterIntegrityError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise DevelopmentAdapterIntegrityError(
                "compact evidence grounding failed lossless decoding"
            ) from exc
    decoded: list[ModelVisibleEvidenceInput] = []
    decoded_evidence_ids: set[str] = set()
    try:
        for untyped in records:
            row = cast(Sequence[object], untyped)
            if len(row) != 7:
                raise DevelopmentAdapterIntegrityError(
                    "compact evidence row must contain exactly seven fields"
                )
            evidence_id = _source_id(row[0], aliases)
            decoded_evidence_ids.add(evidence_id)
            discourse = cast(Sequence[object], row[2])
            if len(discourse) != 3:
                raise DevelopmentAdapterIntegrityError(
                    "compact discourse position must contain exactly three fields"
                )
            mentions = []
            for untyped_mention in cast(Sequence[object], row[3]):
                mention = cast(Sequence[object], untyped_mention)
                if len(mention) != 7:
                    raise DevelopmentAdapterIntegrityError(
                        "compact mention must contain exactly seven fields"
                    )
                surface = cast(str, mention[3])
                coreference = cast(Mapping[str, object], mention[6])
                mentions.append(
                    {
                        "candidate_id": _source_id(mention[0], aliases),
                        "evidence_id": evidence_id,
                        "start_char": mention[1],
                        "end_char": mention[2],
                        "surface": surface,
                        "surface_hash": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
                        "provisional_type": mention[4],
                        "alias_candidate_ids": [
                            _source_id(item, aliases)
                            for item in cast(Sequence[object], mention[5])
                        ],
                        "coreference_scores": {
                            _source_id(key, aliases): value
                            for key, value in coreference.items()
                        },
                    }
                )
            events = []
            for untyped_event in cast(Sequence[object], row[4]):
                event = cast(Sequence[object], untyped_event)
                if len(event) != 6:
                    raise DevelopmentAdapterIntegrityError(
                        "compact event must contain exactly six fields"
                    )
                events.append(
                    {
                        "candidate_id": _source_id(event[0], aliases),
                        "evidence_id": evidence_id,
                        "trigger_start_char": event[1],
                        "trigger_end_char": event[2],
                        "trigger_surface": event[3],
                        "participant_mention_candidate_ids": [
                            _source_id(item, aliases)
                            for item in cast(Sequence[object], event[4])
                        ],
                        "confidence": event[5],
                    }
                )
            relations = []
            for untyped_relation in cast(Sequence[object], row[5]):
                relation = cast(Sequence[object], untyped_relation)
                if len(relation) != 5:
                    raise DevelopmentAdapterIntegrityError(
                        "compact relation must contain exactly five fields"
                    )
                object_alias = relation[2]
                relations.append(
                    {
                        "candidate_id": _source_id(relation[0], aliases),
                        "evidence_id": evidence_id,
                        "subject_mention_candidate_id": _source_id(
                            relation[1], aliases
                        ),
                        "object_mention_candidate_id": (
                            None
                            if object_alias is None
                            else _source_id(object_alias, aliases)
                        ),
                        "surface_phrase": relation[3],
                        "confidence": relation[4],
                    }
                )
            temporal = []
            for untyped_temporal in cast(Sequence[object], row[6]):
                clue = cast(Sequence[object], untyped_temporal)
                if len(clue) != 5:
                    raise DevelopmentAdapterIntegrityError(
                        "compact temporal clue must contain exactly five fields"
                    )
                temporal.append(
                    {
                        "clue_id": _source_id(clue[0], aliases),
                        "evidence_id": evidence_id,
                        "normalized_expression": clue[1],
                        "target_candidate_ids": [
                            _source_id(item, aliases)
                            for item in cast(Sequence[object], clue[2])
                        ],
                        "relation": clue[3],
                        "confidence": clue[4],
                    }
                )
            evidence_type = (
                ModelVisibleEvidenceRecord
                if grounding is not None
                else LegacyModelVisibleEvidenceRecord
            )
            decoded.append(
                evidence_type.model_validate(
                    {
                        "evidence_id": evidence_id,
                        "text": row[1],
                        "discourse_position": {
                            "passage_order": discourse[0],
                            "sentence_order": discourse[1],
                            "token_order": discourse[2],
                        },
                        "mention_candidates": mentions,
                        "event_candidates": events,
                        "relation_phrase_candidates": relations,
                        "temporal_clues": temporal,
                        **grounding_by_evidence_id.get(evidence_id, {}),
                    }
                )
            )
    except DevelopmentAdapterIntegrityError:
        raise
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise DevelopmentAdapterIntegrityError(
            "compact evidence failed typed lossless decoding"
        ) from exc
    if grounding is not None and set(grounding_by_evidence_id) != decoded_evidence_ids:
        raise DevelopmentAdapterIntegrityError(
            "compact evidence grounding must cover exactly the positional evidence rows"
        )
    return tuple(decoded)


def _expand_compact_wire_value(value: object) -> object:
    """Invert the schema-key dictionary used outside positional evidence rows."""

    inverse = {
        compact: full
        for full, compact in _WIRE_ALIASES.items()
        if full != compact and compact != "aliases"
    }
    if isinstance(value, Mapping):
        return {
            inverse.get(str(key), str(key)): _expand_compact_wire_value(child)
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_expand_compact_wire_value(item) for item in value]
    return value


_SOURCE_REFERENCE_FIELDS = frozenset(
    {
        "evidence_id",
        "evidence_ids",
        "input_object_ids",
        "supported_mention_candidate_ids",
        "why_matters_evidence_ids",
    }
)


def _rewrite_source_references(
    value: object,
    replacements: Mapping[str, str],
) -> object:
    """Rewrite only audited source-reference fields, never authored prose."""

    def rewrite(item: object, field_name: str | None = None) -> object:
        if isinstance(item, Mapping):
            return {
                str(key): rewrite(child, str(key))
                for key, child in item.items()
            }
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return [rewrite(child, field_name) for child in item]
        if (
            field_name in _SOURCE_REFERENCE_FIELDS
            and isinstance(item, str)
            and item in replacements
        ):
            return replacements[item]
        return item

    return rewrite(value)


def decode_development_semantic_request(
    encoded: EncodedSemanticRequest,
) -> PreconstructionRequest | ConstructionRequest:
    """Reconstruct the complete typed request represented by ``lossless-model-wire-v1``.

    This decoder is a controller audit, not a model-facing semantic operation.  It
    proves that positional evidence and administrative/default omission are
    reversible before any request is admitted to the live service.
    """

    sections = encoded.sections
    envelope_value = sections.get("request_envelope")
    evidence_value = sections.get("evidence_snapshot", sections.get("evidence_packet"))
    grounding_value = sections.get("evidence_grounding")
    if not isinstance(envelope_value, Mapping) or evidence_value is None:
        raise DevelopmentAdapterIntegrityError("lossless wire lacks its request envelope")
    if grounding_value is not None and not isinstance(grounding_value, Mapping):
        raise DevelopmentAdapterIntegrityError(
            "lossless wire evidence grounding must be an object"
        )
    envelope = cast(dict[str, object], _expand_compact_wire_value(envelope_value))
    try:
        condition = ConditionName(cast(str, envelope["condition"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DevelopmentAdapterIntegrityError(
            "lossless wire request envelope lacks a valid condition"
        ) from exc
    upper_value = sections.get("upper_ontology")
    if not isinstance(upper_value, Mapping):
        raise DevelopmentAdapterIntegrityError("lossless wire lacks the upper ontology")
    upper = _expand_compact_wire_value(upper_value)
    if condition is ConditionName.C1_LLM_PRE:
        if not isinstance(evidence_value, Sequence) or isinstance(
            evidence_value, (str, bytes, bytearray)
        ):
            raise DevelopmentAdapterIntegrityError("C1 wire lacks positional evidence")
        request = PreconstructionRequest.model_validate(
            {
                **envelope,
                "upper_ontology": upper,
                "evidence": [
                    item.model_dump(mode="json", exclude={"content_hash"})
                    for item in decode_compact_evidence_records(
                        evidence_value,
                        encoded.alias_manifest,
                        grounding_value,
                    )
                ],
            }
        )
    else:
        if not isinstance(evidence_value, Mapping):
            raise DevelopmentAdapterIntegrityError(
                "query-time wire lacks its complete evidence packet"
            )
        evidence_rows = evidence_value.get("evidence")
        if not isinstance(evidence_rows, Sequence) or isinstance(
            evidence_rows, (str, bytes, bytearray)
        ):
            raise DevelopmentAdapterIntegrityError(
                "query-time wire packet lacks positional evidence"
            )
        source_ids = encoded.alias_manifest.alias_to_source
        ordered = evidence_value.get("ordered_evidence_ids")
        if not isinstance(ordered, Sequence) or isinstance(
            ordered, (str, bytes, bytearray)
        ):
            raise DevelopmentAdapterIntegrityError(
                "query-time wire packet lacks its complete evidence order"
            )
        context_value = sections.get("query_context")
        if not isinstance(context_value, Mapping):
            raise DevelopmentAdapterIntegrityError("query-time wire lacks its context")
        packet = {
            "packet_hash": evidence_value.get("packet_hash"),
            "retrieval_method": evidence_value.get("retrieval_method"),
            "ordered_evidence_ids": [_source_id(item, source_ids) for item in ordered],
            "evidence": [
                item.model_dump(mode="json", exclude={"content_hash"})
                for item in decode_compact_evidence_records(
                    evidence_rows,
                    encoded.alias_manifest,
                    grounding_value,
                )
            ],
        }
        payload: dict[str, object] = {
            **envelope,
            "upper_ontology": upper,
            "context": _expand_compact_wire_value(context_value),
            "packet": packet,
        }
        fixed = sections.get("sealed_ontology")
        if fixed is not None:
            expanded_fixed = _expand_compact_wire_value(fixed)
            payload["fixed_ontology"] = _rewrite_source_references(
                expanded_fixed,
                encoded.alias_manifest.alias_to_source,
            )
        request = ConstructionRequest.model_validate(payload)
    if request.content_hash != encoded.alias_manifest.semantic_request_hash:
        raise DevelopmentAdapterIntegrityError(
            "lossless wire round trip changed the semantic request"
        )
    return request


class PackingTokenizer(Protocol):
    def apply_chat_template(self, conversation: object, **kwargs: object) -> object: ...

    def encode(self, text: str, **kwargs: object) -> Sequence[int]: ...


class MeteredGenerationService(Protocol):
    """Private dependency; never expose this lifecycle-capable object to the runner."""

    @property
    def actual_allocated_service_seconds(self) -> float: ...

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        repair: bool,
        job_id: str,
        attempt_id: str,
        remaining_required_seconds: float = 0.0,
        accounting_details: Mapping[str, object] | None = None,
    ) -> GenerationResult: ...


class DevelopmentCallExecutor(Protocol):
    """Gold-free semantic executor injected only by the production adopter."""

    def execute(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult: ...

    def rehydrate_completed_call(
        self,
        adapter: ProductionDevelopmentServiceAdapter,
        call: DevelopmentCallSpec,
        receipt: DevelopmentCallAuditReceipt,
        result: ServiceCallResult,
    ) -> None: ...


class DevelopmentConstructionConfiguration(ImmutableRecord):
    """Tracked query-blind ontology, budget, packing, and retrieval choices."""

    configuration_id: Literal["development-construction-v1"]
    source_file_sha256: Sha256Digest
    upper_ontology: UpperOntology
    preconstruction_budgets: OutputBudgets
    projection_budgets_by_unit: Mapping[str, OutputBudgets]
    retrieval_policy: Literal["complete_admissible_snapshot_v1"]
    first_pass_input_tokens: Literal[10240]
    first_pass_output_tokens: Literal[2048]
    repair_input_tokens: Literal[10752]
    repair_output_tokens: Literal[1536]
    maximum_model_tokens: Literal[12288]

    @model_validator(mode="after")
    def exact_development_units_and_resource_caps(self) -> Self:
        expected_units = {
            "dev-unit-01",
            "dev-unit-02",
            "dev-unit-03",
            "dev-unit-04",
        }
        if set(self.projection_budgets_by_unit) != expected_units:
            raise ValueError("development construction config must bind exactly four units")
        if self.first_pass_input_tokens + self.first_pass_output_tokens > self.maximum_model_tokens:
            raise ValueError("first-pass token allocation exceeds model context")
        if self.repair_input_tokens + self.repair_output_tokens > self.maximum_model_tokens:
            raise ValueError("repair token allocation exceeds model context")
        return self

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG,
        *,
        expected_sha256: str | None = None,
    ) -> DevelopmentConstructionConfiguration:
        if path.is_symlink() or not path.is_file():
            raise DevelopmentAdapterIntegrityError(
                "development construction configuration must be a regular file"
            )
        raw = path.read_bytes()
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise DevelopmentAdapterIntegrityError(
                "development construction configuration bytes changed"
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DevelopmentAdapterIntegrityError(
                "development construction configuration is not JSON"
            ) from exc
        if not isinstance(value, dict):
            raise DevelopmentAdapterIntegrityError(
                "development construction configuration root must be an object"
            )
        if value.pop("schema_version", None) != "1.0.0":
            raise DevelopmentAdapterIntegrityError(
                "development construction configuration schema changed"
            )
        value["source_file_sha256"] = observed_sha256
        try:
            return cls.model_validate(value)
        except ValueError as exc:
            raise DevelopmentAdapterIntegrityError(
                "development construction configuration failed validation"
            ) from exc


class DevelopmentRequestRuntime(ImmutableRecord):
    """Precomputed non-circular prompt/schema/decoding bindings."""

    condition: ConditionName
    prompt_hash: Sha256Digest
    output_schema_hash: Sha256Digest
    decoding_manifest: DecodingManifest
    capability_manifest: CapabilityManifest


def development_request_runtime(
    *,
    root: Path,
    condition: ConditionName,
    tokenizer_manifest: TokenizerManifest,
    seed: int,
    repair: bool = False,
    fixed_ontology: FixedOntologyInput | None = None,
    fixed_evidence: Sequence[ModelVisibleEvidenceRecord] | None = None,
) -> DevelopmentRequestRuntime:
    prompt = (
        (root / "prompts/repair/prompt_v1.md").read_text(encoding="utf-8")
        if repair
        else render_condition_system_prompt(root, condition)
    )
    schema = base_condition_output_schema(condition)
    if condition is ConditionName.A_FIXED_SELECT:
        if fixed_ontology is None or fixed_evidence is None:
            raise DevelopmentAdapterIntegrityError(
                "FixedSelect runtime requires its complete sealed ontology and packet"
            )
        schema = _fixed_select_output_schema(
            schema,
            fixed_ontology=fixed_ontology,
            evidence_aliases=_source_aliases_for_evidence(fixed_evidence),
            evidence_source_ids=tuple(item.evidence_id for item in fixed_evidence),
        )
    elif fixed_ontology is not None or fixed_evidence is not None:
        raise DevelopmentAdapterIntegrityError(
            "only FixedSelect may parameterize a sealed-inventory grammar"
        )
    schema_hash = canonical_sha256(schema)
    constructor = DecodingManifest.repair if repair else DecodingManifest.first_pass
    decoding = constructor(
        seed=seed,
        eos_token_id=tokenizer_manifest.eos_token_id,
        end_of_turn_token_ids=tokenizer_manifest.end_of_turn_token_ids,
        chat_template_hash=tokenizer_manifest.chat_template_sha256,
        output_schema_hash=schema_hash,
        structured_decoder=STRUCTURED_DECODER,
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
    )
    return DevelopmentRequestRuntime(
        condition=condition,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        output_schema_hash=schema_hash,
        decoding_manifest=decoding,
        capability_manifest=CapabilityManifest.for_condition(condition),
    )


def development_runtime_identifiers(
    *,
    root: Path,
    condition: ConditionName,
    tokenizer_manifest: TokenizerManifest,
    model_name: str = FALLBACK_SERVED_MODEL_NAME,
    model_revision: str = FALLBACK_MODEL_REVISION,
    seed: int,
    repair: bool = False,
    fixed_ontology: FixedOntologyInput | None = None,
    fixed_evidence: Sequence[ModelVisibleEvidenceRecord] | None = None,
) -> RuntimeIdentifiers:
    """Build the exact identifiers needed before the semantic request exists."""

    runtime = development_request_runtime(
        root=root,
        condition=condition,
        tokenizer_manifest=tokenizer_manifest,
        seed=seed,
        repair=repair,
        fixed_ontology=fixed_ontology,
        fixed_evidence=fixed_evidence,
    )
    return RuntimeIdentifiers(
        model_id=model_name,
        model_revision=model_revision,
        tokenizer_hash=tokenizer_manifest.manifest_sha256,
        runtime_version=PINNED_RUNTIME_VERSION,
        prompt_hash=runtime.prompt_hash,
        output_schema_hash=runtime.output_schema_hash,
        decoding_config_hash=runtime.decoding_manifest.content_hash,
    )


def _string_enum(values: Sequence[str]) -> dict[str, object]:
    unique = sorted(set(values))
    if not unique:
        raise DevelopmentAdapterIntegrityError(
            "a FixedSelect identifier enum cannot be empty"
        )
    return {"enum": unique, "type": "string"}


def _fixed_select_output_schema(
    base: Mapping[str, object],
    *,
    fixed_ontology: FixedOntologyInput,
    evidence_aliases: Mapping[str, str],
    evidence_source_ids: Sequence[str],
) -> dict[str, object]:
    """Bind constrained decoding to the exact complete sealed C1 inventory.

    Evidence and mention references use the deterministic aliases present on the
    model wire.  Every constructed ontology identifier remains the exact sealed
    identifier.  CPU validation still rechecks the same inventory after aliases
    are restored; this grammar is a first, independent mechanical boundary.
    """

    schema = copy.deepcopy(dict(base))
    definitions = cast(dict[str, dict[str, Any]], schema.get("$defs"))
    local_schema = fixed_ontology.local_schema
    graph = fixed_ontology.instance_graph
    seal = fixed_ontology.construction_seal
    if not definitions:
        raise DevelopmentAdapterIntegrityError("FixedSelect schema lacks definitions")

    type_ids = [item.type_id for item in local_schema.contextual_types]
    predicate_ids = [item.predicate_id for item in local_schema.predicates]
    entity_ids = [item.entity_id for item in graph.entities]
    event_ids = [item.event_id for item in graph.events]
    proposition_ids = [
        item.proposition_content_id for item in graph.proposition_contents
    ]
    assertion_ids = [item.assertion_id for item in graph.assertions]
    referent_ids = [*entity_ids, *event_ids, *proposition_ids]
    role_names = [
        role
        for predicate in local_schema.predicates
        for role in predicate.role_names
    ]
    evidence_ids = [
        evidence_aliases[source_id] for source_id in evidence_source_ids
    ]
    mention_aliases = [
        alias for alias in evidence_aliases.values() if alias.startswith("srcM")
    ]

    definitions["ConstructionOperator"]["enum"] = [
        "selection",
        "compression",
        "supported_description",
    ]
    decision_properties = cast(
        dict[str, dict[str, Any]], definitions["OntologyDecision"]["properties"]
    )
    for name in ("created_object_ids", "removed_object_ids"):
        decision_properties[name]["maxItems"] = 0
    decision_properties["input_object_ids"]["items"] = _string_enum(
        seal.sealed_object_ids
    )

    if not proposition_ids:
        graph_properties = cast(
            dict[str, dict[str, Any]], definitions["InstanceGraph"]["properties"]
        )
        graph_properties["proposition_contents"]["maxItems"] = 0
        assertion_properties = cast(
            dict[str, dict[str, Any]], definitions["QualifiedAssertion"]["properties"]
        )
        assertion_properties["proposition_content_id"] = {
            "const": None,
            "type": "null",
        }
        assertion_properties["epistemic_scope"] = {"const": None, "type": "null"}

    constraints: tuple[tuple[str, str, Sequence[str]], ...] = (
        ("LocalContextSchema", "schema_id", (local_schema.schema_id,)),
        ("LocalTypeDefinition", "type_id", type_ids),
        ("LocalPredicateDefinition", "predicate_id", predicate_ids),
        ("LocalPredicateDefinition", "domain_type_ids", type_ids),
        ("LocalPredicateDefinition", "range_type_ids", type_ids),
        ("Entity", "entity_id", entity_ids),
        ("Entity", "supported_mention_candidate_ids", mention_aliases),
        ("Entity", "contextual_type_id", type_ids),
        ("Entity", "description_assertion_ids", assertion_ids),
        ("Event", "event_id", event_ids),
        ("Event", "contextual_type_id", type_ids),
        ("Event", "description_assertion_ids", assertion_ids),
        ("PropositionContent", "proposition_content_id", proposition_ids),
        ("PropositionContent", "predicate_id", predicate_ids),
        ("PropositionContent", "subject_id", referent_ids),
        ("PropositionContent", "object_id", referent_ids),
        ("QualifiedAssertion", "assertion_id", assertion_ids),
        ("QualifiedAssertion", "predicate_id", predicate_ids),
        ("QualifiedAssertion", "proposition_content_id", proposition_ids),
        ("QualifiedAssertion", "subject_id", referent_ids),
        ("QualifiedAssertion", "object_id", referent_ids),
        ("RoleBinding", "object_id", referent_ids),
        ("EpistemicScope", "holder_id", entity_ids),
        ("EpistemicScope", "proposition_content_id", proposition_ids),
    )
    if role_names:
        constraints = (*constraints, ("RoleBinding", "role", role_names))
    for definition_name, property_name, allowed in constraints:
        properties = cast(
            dict[str, dict[str, Any]], definitions[definition_name]["properties"]
        )
        if not allowed and definition_name == "PropositionContent":
            continue
        if (
            not allowed
            and definition_name in {"QualifiedAssertion", "EpistemicScope"}
            and property_name == "proposition_content_id"
        ):
            properties[property_name] = {"const": None, "type": "null"}
            continue
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
        "ProvenanceReference",
        "QualifiedAssertion",
        "RoleBinding",
        "EpistemicScope",
    ):
        properties = cast(
            dict[str, dict[str, Any]], definitions[definition_name]["properties"]
        )
        if "evidence_ids" in properties:
            properties["evidence_ids"]["items"] = _string_enum(evidence_ids)
        if "evidence_id" in properties:
            properties["evidence_id"] = _string_enum(evidence_ids)
    cast(dict[str, Any], definitions["OmissionRecord"]["properties"])[
        "evidence_id"
    ] = _string_enum(evidence_ids)
    return schema


def development_output_schema_for_request(
    semantic_request: PreconstructionRequest | ConstructionRequest,
) -> dict[str, object]:
    """Return the exact guided grammar bound to one semantic request."""

    schema = base_condition_output_schema(semantic_request.condition)
    if semantic_request.condition is not ConditionName.A_FIXED_SELECT:
        return schema
    if (
        not isinstance(semantic_request, ConstructionRequest)
        or semantic_request.fixed_ontology is None
    ):
        raise DevelopmentAdapterIntegrityError(
            "FixedSelect output schema requires its complete sealed ontology"
        )
    aliases = _source_alias_manifest(semantic_request).source_to_alias
    return _fixed_select_output_schema(
        schema,
        fixed_ontology=semantic_request.fixed_ontology,
        evidence_aliases=aliases,
        evidence_source_ids=semantic_request.packet.ordered_evidence_ids,
    )


def _request_sections(
    semantic_request: PreconstructionRequest | ConstructionRequest,
) -> EncodedSemanticRequest:
    alias_manifest = _source_alias_manifest(semantic_request)
    aliases = alias_manifest.source_to_alias
    if isinstance(semantic_request, PreconstructionRequest):
        sections = {
            "wire_encoding": model_wire_encoding_manifest(alias_manifest),
            "upper_ontology": _compact_wire_value(
                semantic_request.upper_ontology.model_dump(mode="json")
            ),
            "evidence_snapshot": _compact_evidence_records(
                semantic_request.evidence,
                aliases,
            ),
            "evidence_grounding": _compact_evidence_grounding(
                semantic_request.evidence,
                aliases,
            ),
            "request_envelope": _compact_wire_value(
                semantic_request.model_dump(
                    mode="json",
                    exclude={"upper_ontology", "evidence"},
                )
            ),
        }
        encoded = EncodedSemanticRequest(
            sections=sections,
            alias_manifest=alias_manifest,
        )
        if decode_development_semantic_request(encoded) != semantic_request:
            raise DevelopmentAdapterIntegrityError(
                "lossless C1 wire failed exact typed round-trip equality"
            )
        return encoded
    sections: dict[str, object] = {
        "wire_encoding": model_wire_encoding_manifest(alias_manifest),
        "upper_ontology": _compact_wire_value(
            semantic_request.upper_ontology.model_dump(mode="json")
        ),
        "query_context": _compact_wire_value(semantic_request.context.model_dump(mode="json")),
        "evidence_packet": {
            "packet_hash": semantic_request.packet.packet_hash,
            "retrieval_method": semantic_request.packet.retrieval_method.value,
            "ordered_evidence_ids": [
                _alias(item, aliases) for item in semantic_request.packet.ordered_evidence_ids
            ],
            "evidence": _compact_evidence_records(
                semantic_request.packet.evidence,
                aliases,
            ),
        },
        "evidence_grounding": _compact_evidence_grounding(
            semantic_request.packet.evidence,
            aliases,
        ),
        "request_envelope": _compact_wire_value(
            semantic_request.model_dump(
                mode="json",
                exclude={"upper_ontology", "context", "packet", "fixed_ontology"},
            )
        ),
    }
    if semantic_request.condition is ConditionName.A_FIXED_SELECT:
        if semantic_request.fixed_ontology is None:
            raise DevelopmentAdapterIntegrityError("FixedSelect request lacks sealed ontology")
        sections["sealed_ontology"] = _compact_wire_value(
            _rewrite_source_references(
                semantic_request.fixed_ontology.model_dump(mode="json"),
                aliases,
            )
        )
    encoded = EncodedSemanticRequest(
        sections=sections,
        alias_manifest=alias_manifest,
    )
    if decode_development_semantic_request(encoded) != semantic_request:
        raise DevelopmentAdapterIntegrityError(
            "lossless query-time wire failed exact typed round-trip equality"
        )
    return encoded


def encode_development_semantic_request(
    semantic_request: PreconstructionRequest | ConstructionRequest,
) -> EncodedSemanticRequest:
    """Return the compact wire sections and separately persistable alias bijection."""

    return _request_sections(semantic_request)


def development_packing_equivalence_hash(
    semantic_request: PreconstructionRequest | ConstructionRequest,
) -> Sha256Digest:
    """Hash model-visible semantics while excluding the trusted request timestamp.

    Query-blind C1 packing is performed before the runner creates its controller
    barrier, so the final ``requested_at`` value cannot yet exist.  The timestamp
    is provenance, not model semantics.  This hash proves the preflight and live
    request differ at most in that field; the live request is token-counted again
    immediately before inference.
    """

    payload = semantic_request.model_dump(
        mode="json",
        exclude={"content_hash", "requested_at"},
    )
    return canonical_sha256(payload)


def restore_model_output_source_aliases(
    parsed_object: Mapping[str, object],
    alias_manifest: ModelWireAliasManifest,
) -> dict[str, object]:
    """Mechanically restore source IDs without changing model-authored semantics."""

    defining_id_fields = {
        "assertion_id",
        "decision_id",
        "entity_id",
        "event_id",
        "predicate_id",
        "proposition_content_id",
        "schema_id",
        "type_id",
    }
    aliases = alias_manifest.alias_to_source

    def restore(value: object, key: str | None = None) -> object:
        if isinstance(value, Mapping):
            restored: dict[str, object] = {}
            for child_key, child in value.items():
                if child_key in defining_id_fields and isinstance(child, str) and child in aliases:
                    raise DevelopmentAdapterIntegrityError(
                        "model output reused a reserved source alias as a constructed ID"
                    )
                restored[child_key] = restore(child, child_key)
            return restored
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [restore(item, key) for item in value]
        if key in _SOURCE_REFERENCE_FIELDS and isinstance(value, str) and value in aliases:
            return aliases[value]
        return value

    return cast(dict[str, object], restore(parsed_object))


def build_development_repair_probe_input(
    *,
    parent_call_id: str,
    parent_semantic_request: ConstructionRequest,
    repair_semantic_request: ConstructionRequest,
    parent_raw_output_hash: Sha256Digest,
    parent_raw_output: Mapping[str, object],
    created_at: datetime,
) -> DevelopmentRepairProbeInput:
    """Apply the sole frozen development repair fault without adding semantics."""

    if parent_call_id != "dev-c2-u04-q02":
        raise DevelopmentAdapterIntegrityError(
            "repair probe parent is not the registered development C2 call"
        )
    invariant_parent = parent_semantic_request.model_dump(
        mode="python",
        exclude={"content_hash", "request_id", "requested_at", "runtime"},
    )
    invariant_repair = repair_semantic_request.model_dump(
        mode="python",
        exclude={"content_hash", "request_id", "requested_at", "runtime"},
    )
    if (
        parent_semantic_request.condition is not ConditionName.C2_LLM_QUERY
        or repair_semantic_request.condition is not ConditionName.C2_LLM_QUERY
        or invariant_parent != invariant_repair
    ):
        raise DevelopmentAdapterIntegrityError(
            "repair semantic request changed the registered parent inputs"
        )
    invalid = copy.deepcopy(dict(parent_raw_output))
    graph = invalid.get("instance_graph")
    if not isinstance(graph, dict):
        raise DevelopmentAdapterIntegrityError("repair parent lacks an instance graph")
    chosen_kind = None
    chosen_id = None
    for kind, id_field in (("entities", "entity_id"), ("events", "event_id")):
        records = graph.get(kind)
        if isinstance(records, list) and records and isinstance(records[0], dict):
            candidate = records[0].get(id_field)
            if isinstance(candidate, str) and candidate:
                records[0]["contextual_type_id"] = "unknown-development-type"
                chosen_kind = kind
                chosen_id = candidate
                break
    if chosen_kind is None or chosen_id is None:
        raise DevelopmentAdapterIntegrityError(
            "repair parent has no entity/event target for the frozen fault"
        )
    diagnostic = DevelopmentRepairDiagnostic(
        code="unknown_contextual_type_reference",
        path=f"instance_graph.{chosen_kind}.{chosen_id}.contextual_type_id",
        message=(
            "Replace only the unknown contextual_type_id with an identifier "
            "declared in local_schema.contextual_types."
        ),
    )
    return DevelopmentRepairProbeInput(
        parent_call_id=parent_call_id,
        parent_semantic_request_hash=parent_semantic_request.content_hash,
        repair_semantic_request_hash=repair_semantic_request.content_hash,
        parent_raw_output_hash=parent_raw_output_hash,
        diagnostic_fixture_id="development-schema-reference-repair-probe-v1",
        fault_injection=(
            "deterministic_unknown_reference_after_preserving_raw_parent"
        ),
        invalid_draft=invalid,
        diagnostics=(diagnostic,),
        created_at=created_at,
    )


def build_development_guided_request(
    *,
    root: Path,
    call_id: str,
    semantic_request: PreconstructionRequest | ConstructionRequest,
    tokenizer: PackingTokenizer,
    tokenizer_manifest: TokenizerManifest,
    model_name: str = FALLBACK_SERVED_MODEL_NAME,
    model_revision: str = FALLBACK_MODEL_REVISION,
    seed: int,
    repair: bool = False,
    repair_probe_input: DevelopmentRepairProbeInput | None = None,
) -> GuidedJSONRequest:
    """Pack one complete semantic request without truncation or hidden fields."""

    condition = semantic_request.condition
    if repair != (repair_probe_input is not None):
        raise DevelopmentAdapterIntegrityError(
            "repair decoding requires the complete deterministic repair-probe input"
        )
    if tokenizer_manifest.tokenizer_revision != model_revision:
        raise DevelopmentAdapterIntegrityError("tokenizer and model revisions differ")
    runtime = development_request_runtime(
        root=root,
        condition=condition,
        tokenizer_manifest=tokenizer_manifest,
        seed=seed,
        repair=repair,
        fixed_ontology=(
            semantic_request.fixed_ontology
            if condition is ConditionName.A_FIXED_SELECT
            else None
        ),
        fixed_evidence=(
            semantic_request.packet.evidence
            if condition is ConditionName.A_FIXED_SELECT
            else None
        ),
    )
    expected_runtime = semantic_request.runtime
    observed = (
        expected_runtime.model_id,
        expected_runtime.model_revision,
        expected_runtime.tokenizer_hash,
        expected_runtime.runtime_version,
        expected_runtime.prompt_hash,
        expected_runtime.output_schema_hash,
        expected_runtime.decoding_config_hash,
    )
    required = (
        model_name,
        model_revision,
        tokenizer_manifest.manifest_sha256,
        PINNED_RUNTIME_VERSION,
        runtime.prompt_hash,
        runtime.output_schema_hash,
        runtime.decoding_manifest.content_hash,
    )
    if observed != required:
        raise DevelopmentAdapterIntegrityError(
            "semantic request runtime differs from frozen request stack"
        )
    prompt = (
        (root / "prompts/repair/prompt_v1.md").read_text(encoding="utf-8")
        if repair
        else render_condition_system_prompt(root, condition)
    )
    schema = development_output_schema_for_request(semantic_request)
    encoded = encode_development_semantic_request(semantic_request)
    sections = dict(encoded.sections)
    if repair_probe_input is not None:
        if repair_probe_input.repair_semantic_request_hash != semantic_request.content_hash:
            raise DevelopmentAdapterIntegrityError(
                "repair-probe input belongs to another semantic request"
            )
        sections["invalid_draft"] = dict(repair_probe_input.invalid_draft)
        sections["validation_diagnostics"] = [
            item.model_dump(mode="json", exclude={"schema_version", "content_hash"})
            for item in repair_probe_input.diagnostics
        ]
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
        raise DevelopmentAdapterIntegrityError("tokenizer did not return token IDs")
    rendered_count = len(rendered)
    section_values: dict[str, object] = {"system_prompt": prompt, **sections}
    packing_sections: list[PackingSection] = [
        PackingSection(
            name="output_schema",
            section_content_hash=canonical_sha256(schema),
            token_count=0,
        )
    ]
    for name, value in section_values.items():
        text = value if isinstance(value, str) else canonical_json(value)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        packing_sections.append(
            PackingSection(
                name=name,
                section_content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                token_count=len(tokens),
            )
        )
    individually_encoded = sum(item.token_count for item in packing_sections)
    if rendered_count < individually_encoded:
        raise DevelopmentAdapterIntegrityError(
            "independent section token counts exceed rendered prompt"
        )
    if rendered_count > individually_encoded:
        packing_sections.append(
            PackingSection(
                name="chat_protocol",
                section_content_hash=tokenizer_manifest.nonthinking_probe_sha256,
                token_count=rendered_count - individually_encoded,
            )
        )
    required_names = ("output_schema", *section_values)
    packing = PackingReport.build(
        condition=condition,
        tokenizer_revision=tokenizer_manifest.tokenizer_revision,
        maximum_model_tokens=runtime.decoding_manifest.maximum_model_tokens,
        maximum_input_tokens=runtime.decoding_manifest.maximum_input_tokens,
        reserved_output_tokens=runtime.decoding_manifest.maximum_output_tokens,
        sections=packing_sections,
        required_section_names=required_names,
        complete_evidence_snapshot=True if condition is ConditionName.C1_LLM_PRE else None,
        complete_evidence_packet=None if condition is ConditionName.C1_LLM_PRE else True,
        complete_sealed_ontology=True if condition is ConditionName.A_FIXED_SELECT else None,
    )
    return GuidedJSONRequest(
        request_id=call_id,
        model_name=model_name,
        condition=condition,
        messages=messages,
        output_schema=schema,
        decoding=runtime.decoding_manifest,
        packing=packing,
        rendered_input_token_count=rendered_count,
    )


class AdapterDurableState(ImmutableRecord):
    """Crash-safe index; only committed receipts are recoverable."""

    execution_manifest_hash: Sha256Digest
    service_identity_hash: Sha256Digest
    completed_receipts: Mapping[str, Sha256Digest] = Field(default_factory=dict)
    completed_results: Mapping[str, Sha256Digest] = Field(default_factory=dict)
    query_access_events: Mapping[Sha256Digest, Sha256Digest] = Field(default_factory=dict)
    packet_materialization_events: Mapping[Sha256Digest, Sha256Digest] = Field(
        default_factory=dict
    )
    fixed_schema_derivations: Mapping[str, Sha256Digest] = Field(default_factory=dict)
    packing_preflight_artifact_hash: Sha256Digest | None = None
    assessment_bundle_artifact_hash: Sha256Digest | None = None
    active_call_id: str | None = None
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def active_call_is_not_completed(self) -> Self:
        if set(self.completed_receipts) != set(self.completed_results):
            raise ValueError("adapter receipt/result indexes differ")
        if self.active_call_id in self.completed_receipts:
            raise ValueError("completed adapter call cannot remain active")
        return self


def _atomic_state(path: Path, state: AdapterDurableState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(state.to_canonical_json())
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class ProductionDevelopmentServiceAdapter:
    """Narrow same-process adapter; it intentionally defines no lifecycle method."""

    live_identity: LiveServiceIdentity
    _service: MeteredGenerationService = field(repr=False)
    artifacts: ArtifactStore
    state_path: Path
    execution_manifest_hash: Sha256Digest
    repository_root: Path | None = None
    query_runtime: AuditedBenchmarkRuntime | None = None
    prequery_evidence_by_hash: Mapping[
        Sha256Digest, ModelEligibleWorldArtifact
    ] = field(default_factory=dict, repr=False)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _executor: DevelopmentCallExecutor | None = field(
        default=None,
        repr=False,
    )
    _query_openings: dict[str, AuditedQueryOpening] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.state_path = self.state_path.resolve(strict=False)
        if self.repository_root is not None:
            self.repository_root = self.repository_root.resolve(strict=True)
        if self.state_path.is_symlink():
            raise DevelopmentAdapterIntegrityError("adapter state cannot be a symlink")
        if self.state_path.exists():
            state = self._read_state()
            if (
                state.execution_manifest_hash != self.execution_manifest_hash
                or state.service_identity_hash != self.live_identity.content_hash
            ):
                raise DevelopmentAdapterIntegrityError("adapter state belongs to another execution")

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise DevelopmentAdapterIntegrityError("adapter clock must be timezone-aware")
        return value.astimezone(UTC)

    def _read_state(self) -> AdapterDurableState:
        if not self.state_path.is_file() or self.state_path.is_symlink():
            raise DevelopmentAdapterIntegrityError("adapter state is missing or unsafe")
        return AdapterDurableState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def _state(self) -> AdapterDurableState:
        if self.state_path.exists():
            return self._read_state()
        return AdapterDurableState(
            execution_manifest_hash=self.execution_manifest_hash,
            service_identity_hash=self.live_identity.content_hash,
            updated_at=self._now(),
        )

    def _write_state(self, state: AdapterDurableState, **updates: object) -> None:
        payload = state.model_dump(mode="python", exclude={"content_hash"})
        payload.update(updates)
        payload["updated_at"] = self._now()
        _atomic_state(self.state_path, AdapterDurableState.model_validate(payload))

    def _remember_query_opening(
        self,
        stage: StageReference,
        opening: AuditedQueryOpening,
    ) -> None:
        state = self._state()
        persisted = dict(state.query_access_events)
        existing = persisted.get(stage.staging_manifest_hash)
        if existing is not None and existing != opening.access_event.content_hash:
            raise DevelopmentAdapterIntegrityError(
                "query stage already maps to another durable access event"
            )
        persisted[stage.staging_manifest_hash] = opening.access_event.content_hash
        self._write_state(state, query_access_events=persisted)

    def identity(self) -> LiveServiceIdentity:
        return self.live_identity

    def allocated_gpu_seconds(self) -> float:
        observed = max(
            float(self._service.actual_allocated_service_seconds),
            self.artifacts.ledger.gpu_summary().total_allocated_seconds,
        )
        if not math.isfinite(observed) or observed < 0:
            raise DevelopmentAdapterIntegrityError("allocated GPU counter is invalid")
        return observed

    def open_query(
        self,
        stage: StageReference,
        barrier: PrequeryBarrier,
    ) -> QueryAccessEvent:
        if self.repository_root is None or self.query_runtime is None:
            raise DevelopmentAdapterIntegrityError(
                "audited query opener was not supplied; query access remains fail-closed"
            )
        access_event_id = (
            f"{barrier.execution_id}-query-{stage.stage_id.removeprefix('stage_')}"
        )
        existing = self._query_openings.get(stage.staging_manifest_hash)
        if existing is not None:
            if existing.access_event.prequery_barrier_hash != barrier.content_hash:
                raise DevelopmentAdapterIntegrityError(
                    "query stage was opened under another prequery barrier"
                )
            return existing.access_event
        self.query_runtime.persist_prequery_barrier(
            barrier,
            release_class=LedgerReleaseClass.PUBLIC,
        )
        try:
            persisted = self.artifacts.ledger.get_query_access_by_event_id(access_event_id)
        except KeyError:
            persisted = None
        if persisted is not None:
            opening = self._rehydrate_query_opening(
                stage=stage,
                barrier=barrier,
                access_event_id=access_event_id,
                persistence=persisted,
            )
            self._query_openings[stage.staging_manifest_hash] = opening
            self._remember_query_opening(stage, opening)
            return opening.access_event
        stage_root = (self.repository_root / stage.relative_path).resolve(strict=True)
        try:
            stage_root.relative_to(self.repository_root)
        except ValueError as exc:
            raise DevelopmentAdapterIntegrityError("query stage escaped repository") from exc
        manifest_path = stage_root / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DevelopmentAdapterIntegrityError("query stage manifest is missing or unsafe")
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != stage.manifest_file_sha256:
            raise DevelopmentAdapterIntegrityError("query stage manifest bytes changed")
        manifest = RuntimeStagingManifest.model_validate_json(manifest_bytes)
        if manifest.content_hash != stage.staging_manifest_hash:
            raise DevelopmentAdapterIntegrityError("query stage logical manifest changed")
        opening = self.query_runtime.open_query(
            staging_root=stage_root,
            manifest=manifest,
            barrier=barrier,
            execution_id=barrier.execution_id,
            execution_manifest_hash=barrier.execution_manifest_hash,
            access_event_id=access_event_id,
        )
        if opening.access_event.query_artifact_hash != stage.query_artifact_hash:
            raise DevelopmentAdapterIntegrityError(
                "audited query opening differs from frozen stage"
            )
        self._query_openings[stage.staging_manifest_hash] = opening
        self._remember_query_opening(stage, opening)
        return opening.access_event

    def _rehydrate_query_opening(
        self,
        *,
        stage: StageReference,
        barrier: PrequeryBarrier,
        access_event_id: str,
        persistence: QueryAccessRecord,
    ) -> AuditedQueryOpening:
        """Recover an already-committed query without touching ``query.json`` again."""

        # Keep this exact-field validation local so a self-consistent stale CAS
        # object cannot be adopted under another execution or stage.
        record = persistence
        expected = (
            access_event_id,
            barrier.execution_id,
            stage.staging_manifest_hash,
            stage.query_artifact_hash,
            barrier.content_hash,
        )
        observed = (
            record.access_event_id,
            record.execution_id,
            record.stage_manifest_hash,
            record.query_artifact_hash,
            record.prequery_barrier_hash,
        )
        if observed != expected:
            raise DevelopmentAdapterIntegrityError(
                "persisted query access differs from the requested execution stage"
            )
        evidence = self.prequery_evidence_by_hash.get(stage.evidence_artifact_hash)
        if evidence is None:
            raise DevelopmentAdapterIntegrityError(
                "persisted query access cannot be recovered without sealed prequery evidence"
            )
        query_record = self.artifacts.ledger.get_artifact(
            record.query_payload_artifact_hash
        )
        access_record = self.artifacts.ledger.get_artifact(
            record.access_event_artifact_hash
        )
        query_bytes = self.artifacts.blobs.read_bytes(query_record)
        access_bytes = self.artifacts.blobs.read_bytes(access_record)
        reveal = QueryRevealArtifact.model_validate_json(query_bytes)
        access = QueryAccessEvent.model_validate_json(access_bytes)
        context = QueryContext(
            context_id=f"ctx_{reveal.reveal_id.removeprefix('reveal_')}",
            revealed_at=reveal.revealed_at,
            **reveal.query.model_dump(
                mode="python",
                exclude={"schema_version", "content_hash"},
            ),
        )
        if (
            access.content_hash != record.access_event_hash
            or access.access_event_id != access_event_id
            or access.execution_id != barrier.execution_id
            or access.stage_manifest_hash != stage.staging_manifest_hash
            or access.query_artifact_hash != reveal.content_hash
            or access.query_artifact_hash != stage.query_artifact_hash
            or access.prequery_barrier_hash != barrier.content_hash
            or access.snapshot_hash != evidence.snapshot.content_hash
            or access.model_visible_query_hash != reveal.query.content_hash
            or access.query_context_hash != context.content_hash
            or to_model_visible_query(context) != reveal.query
        ):
            raise DevelopmentAdapterIntegrityError(
                "persisted query CAS lineage failed exact rehydration"
            )
        return AuditedQueryOpening(
            evidence=evidence,
            reveal=reveal,
            context=context,
            access_event=access,
            persistence=record,
        )

    def execute_call(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult:
        """Delegate to the hash-bound executor or fail before model inference."""

        if self._executor is None:
            raise DevelopmentAdapterIntegrityError(
                "development semantic input repository is not configured"
            )
        if envelope.service_identity_hash != self.live_identity.content_hash:
            raise DevelopmentAdapterIntegrityError(
                "development execution envelope names another live service"
            )
        return self._executor.execute(self, call, envelope)

    def audited_opening(self, stage: StageReference) -> AuditedQueryOpening:
        """Return only an opening already persisted through :meth:`open_query`."""

        try:
            return self._query_openings[stage.staging_manifest_hash]
        except KeyError as exc:
            raise DevelopmentAdapterIntegrityError(
                "query-time call lacks a prior audited query opening"
            ) from exc

    def persisted_prequery_barrier(self, barrier_hash: str) -> PrequeryBarrier:
        """Rehydrate one exact barrier already registered by the audited opener."""

        record = self.artifacts.ledger.get_prequery_barrier(barrier_hash)
        artifact = self.artifacts.ledger.get_artifact(record.barrier_artifact_hash)
        barrier = PrequeryBarrier.model_validate_json(
            self.artifacts.blobs.read_bytes(artifact)
        )
        if barrier.content_hash != barrier_hash:
            raise DevelopmentAdapterIntegrityError(
                "persisted prequery barrier logical hash changed"
            )
        return barrier

    def recover_call(
        self,
        call: DevelopmentCallSpec,
        envelope: CallExecutionEnvelope,
    ) -> ServiceCallResult | None:
        state = self._state()
        artifact_hash = state.completed_receipts.get(call.call_id)
        if artifact_hash is None:
            return None
        if envelope.execution_manifest_hash != state.execution_manifest_hash:
            raise DevelopmentAdapterIntegrityError("recovery envelope changed execution")
        record = self.artifacts.ledger.get_artifact(artifact_hash)
        payload = self.artifacts.blobs.read_bytes(record)
        receipt = DevelopmentCallAuditReceipt.model_validate_json(payload)
        identity_record = self.artifacts.ledger.get_artifact(
            receipt.service_identity.artifact_hash
        )
        persisted_identity = LiveServiceIdentity.model_validate_json(
            self.artifacts.blobs.read_bytes(identity_record)
        )
        envelope_reference = receipt.call_execution_envelope
        if envelope_reference is None:
            raise DevelopmentAdapterIntegrityError("started receipt lacks execution envelope")
        envelope_record = self.artifacts.ledger.get_artifact(envelope_reference.artifact_hash)
        persisted_envelope = CallExecutionEnvelope.model_validate_json(
            self.artifacts.blobs.read_bytes(envelope_record)
        )
        if (
            receipt.call_id != call.call_id
            or receipt.ordinal != call.ordinal
            or receipt.condition is not call.condition
            or receipt.service_identity.logical_content_hash
            != self.live_identity.content_hash
            or persisted_identity != self.live_identity
            or envelope_reference.logical_content_hash != envelope.content_hash
            or persisted_envelope != envelope
        ):
            raise DevelopmentAdapterIntegrityError("receipt index or bound inputs changed")
        result_artifact_hash = state.completed_results[call.call_id]
        result_record = self.artifacts.ledger.get_artifact(result_artifact_hash)
        result = ServiceCallResult.model_validate_json(
            self.artifacts.blobs.read_bytes(result_record)
        )
        if self._executor is not None:
            self._executor.rehydrate_completed_call(self, call, receipt, result)
        expected_config_hash = envelope.expected_run_condition_config_hash
        if envelope.fixed_schema_derivation_plan_hash is not None:
            derivation = result.fixed_schema_derivation
            derivation_reference = receipt.fixed_schema_derivation
            if (
                expected_config_hash is not None
                or derivation is None
                or derivation_reference is None
                or derivation.plan_hash
                != envelope.fixed_schema_derivation_plan_hash
                or derivation_reference.logical_content_hash
                != derivation.content_hash
            ):
                raise DevelopmentAdapterIntegrityError(
                    "recovered FixedSelect derivation changed"
                )
            derivation_record = self.artifacts.ledger.get_artifact(
                derivation_reference.artifact_hash
            )
            persisted_derivation = FixedSchemaDerivationReceipt.model_validate_json(
                self.artifacts.blobs.read_bytes(derivation_record)
            )
            if persisted_derivation != derivation:
                raise DevelopmentAdapterIntegrityError(
                    "recovered FixedSelect derivation CAS object changed"
                )
            expected_config_hash = derivation.exact_run_condition_config_hash
        if (
            result.call_id != call.call_id
            or result.ledger_receipt_hash != artifact_hash
            or result.service_identity_hash != self.live_identity.content_hash
            or result.run_condition_config_hash != expected_config_hash
            or result.outcome is not receipt.outcome
            or result.request_started is not receipt.request_started
            or result.request_hash != receipt.rendered_model_request.logical_content_hash
            or result.response_artifact_hash != receipt.raw_response_artifact_hash
        ):
            raise DevelopmentAdapterIntegrityError("recovered result lineage changed")
        if receipt.validated_generation is not None and (
            result.validated_generation_hash
            != receipt.validated_generation.logical_content_hash
        ):
            raise DevelopmentAdapterIntegrityError("recovered validated generation changed")
        if receipt.condition_result is not None:
            expected_condition_hash = (
                result.condition_preparation_hash
                if call.condition is ConditionName.C1_LLM_PRE
                else result.condition_attempt_hash
            )
            if expected_condition_hash != receipt.condition_result.logical_content_hash:
                raise DevelopmentAdapterIntegrityError("recovered condition result changed")
        return result


def persist_logical_record(
    artifacts: ArtifactStore,
    value: ImmutableRecord,
    *,
    object_kind: str,
    created_at: datetime,
) -> LogicalCASReference:
    artifact = artifacts.put_bytes(
        (value.to_canonical_json() + "\n").encode("utf-8"),
        media_type=f"application/vnd.story-projection.{object_kind}+json",
        release_class=LedgerReleaseClass.PUBLIC,
        created_at=created_at,
    )
    return LogicalCASReference(
        logical_content_hash=value.content_hash,
        artifact_hash=artifact.content_hash,
        object_kind=object_kind,
    )


def persist_opaque_json(
    artifacts: ArtifactStore,
    value: Mapping[str, object],
    *,
    object_kind: str,
    created_at: datetime,
) -> OpaqueJSONReference:
    artifact = artifacts.put_bytes(
        (canonical_json(value) + "\n").encode("utf-8"),
        media_type=f"application/vnd.story-projection.{object_kind}+json",
        release_class=LedgerReleaseClass.PUBLIC,
        created_at=created_at,
    )
    return OpaqueJSONReference(
        logical_content_hash=canonical_sha256(value),
        artifact_hash=artifact.content_hash,
        object_kind=object_kind,
    )


def treatment_switches_for(condition: ConditionName) -> DevelopmentTreatmentSwitches:
    if condition is ConditionName.A_NO_CONTEXT:
        return DevelopmentTreatmentSwitches(context_payload_mode="generic")
    if condition is ConditionName.A_NO_TEMPORAL_EPISTEMIC:
        return DevelopmentTreatmentSwitches(temporal_epistemic_fields="disabled")
    if condition is ConditionName.A_NO_RARE_GUARD:
        return DevelopmentTreatmentSwitches(rare_guard="disabled")
    return DevelopmentTreatmentSwitches()


__all__ = [
    "DEFAULT_DEVELOPMENT_CONSTRUCTION_CONFIG",
    "AdapterDurableState",
    "DevelopmentAdapterIntegrityError",
    "DevelopmentCallExecutor",
    "DevelopmentConstructionConfiguration",
    "DevelopmentRequestRuntime",
    "MeteredGenerationService",
    "PackingTokenizer",
    "ProductionDevelopmentServiceAdapter",
    "build_development_guided_request",
    "build_development_repair_probe_input",
    "decode_compact_evidence_records",
    "decode_development_semantic_request",
    "development_output_schema_for_request",
    "development_packing_equivalence_hash",
    "development_request_runtime",
    "development_runtime_identifiers",
    "model_wire_source_alias_bijection",
    "persist_logical_record",
    "persist_opaque_json",
    "treatment_switches_for",
]
