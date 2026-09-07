"""Lossless schema-directed record tuples, with unchanged semantic validation.

Only record field names and derivable immutable-record envelopes are omitted.
Required values remain mandatory. Optional values remain explicit in a trailing
object; their absence uses exactly the existing canonical contract defaults.
No evidence, IDs, text, times, qualifications, or graph objects are inferred.
This codec is not activated in production until its packing gate passes.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

ADMIN = frozenset({"content_hash", "schema_version"})


def is_reference_field(name: str) -> bool:
    return name.endswith(("_id", "_ids", "_hash", "_hashes"))


def reference_aliases(value: Any) -> dict[str, str]:
    """Assign reversible handles to opaque IDs, never to prose or semantic text."""
    references: set[str] = set()
    literals: set[str] = set()

    def visit(item, field=""):
        if isinstance(item, dict):
            for key, child in item.items():
                literals.add(key)
                visit(child, key)
        elif isinstance(item, list):
            for child in item:
                visit(child, field)
        elif isinstance(item, str):
            literals.add(item)
            if is_reference_field(field) and len(item) > 12:
                references.add(item)

    visit(value)
    result = {f"I{index}": text for index, text in enumerate(sorted(references))}
    if set(result) & literals:
        raise OutputWireError("reserved opaque-reference handle collides with input")
    return result


def translate_references(value: Any, mapping: Mapping[str, str], *, decode: bool) -> Any:
    """Translate supplied references only. All non-reference values are unchanged."""
    if len(set(mapping.values())) != len(mapping):
        raise OutputWireError("opaque references are not bijective")
    lookup = dict(mapping) if decode else {v: k for k, v in mapping.items()}

    def visit(item, field=""):
        if isinstance(item, dict):
            result = {lookup.get(k, k): visit(v, k) for k, v in item.items()}
            if len(result) != len(item):
                raise OutputWireError("opaque reference translation collides with object key")
            return result
        if isinstance(item, list):
            return [visit(child, field) for child in item]
        if isinstance(item, str) and is_reference_field(field):
            if decode and item.startswith("I") and item[1:].isdigit() and item not in lookup:
                raise OutputWireError("unknown supplied opaque-reference handle")
            return lookup.get(item, item)
        return copy.deepcopy(item)

    return visit(value)


TABLE_INSTRUCTIONS = (
    "Input lossless-json-tables-v3: names contains object field names; keys contains "
    "ordered lists of indices into names. "
    "strings contains shared literal strings. In value, [i,...] is an object "
    "using keys[i], [null,...] is an array, and negative integer -i-1 means "
    'strings[i]. A literal negative number is wrapped as {"number":n}. '
    "Other scalars are literal. Decode these tables to "
    "read the complete request. Nothing is omitted, summarized, or inferred."
)


def pack_input_tables(value: Any) -> dict[str, Any]:
    """Losslessly deduplicate request syntax and repeated long string values."""
    shapes: set[tuple[str, ...]] = set()
    strings: Counter[str] = Counter()

    def visit(item):
        if isinstance(item, dict):
            shapes.add(tuple(sorted(item)))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            strings[item] += 1

    visit(value)
    keys = sorted(shapes)
    texts = sorted(text for text, count in strings.items() if count > 1 and len(text) >= 4)
    names = sorted({key for shape in keys for key in shape})
    name_index = {name: index for index, name in enumerate(names)}
    key_index = {shape: index for index, shape in enumerate(keys)}
    text_index = {text: index for index, text in enumerate(texts)}

    def encode(item):
        if isinstance(item, dict):
            shape = tuple(sorted(item))
            return [key_index[shape], *(encode(item[key]) for key in shape)]
        if isinstance(item, list):
            return [None, *(encode(child) for child in item)]
        if isinstance(item, str) and item in text_index:
            return -1 - text_index[item]
        if isinstance(item, (int, float)) and not isinstance(item, bool) and item < 0:
            return {"number": item}
        return item

    packed = {
        "names": names,
        "keys": [[name_index[key] for key in shape] for shape in keys],
        "strings": texts,
        "value": encode(value),
    }
    if unpack_input_tables(packed) != value:
        raise OutputWireError("input table round-trip failed")
    return packed


def unpack_input_tables(packed: Mapping[str, Any]) -> Any:
    if set(packed) != {"names", "keys", "strings", "value"}:
        raise OutputWireError("unexpected input table envelope")
    names, raw_keys, strings = packed["names"], packed["keys"], packed["strings"]
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise OutputWireError("invalid names dictionary")
    if not isinstance(raw_keys, list) or any(not isinstance(shape, list) for shape in raw_keys):
        raise OutputWireError("invalid shape dictionary")
    if any(type(i) is not int or not 0 <= i < len(names) for shape in raw_keys for i in shape):
        raise OutputWireError("shape index outside names dictionary")
    keys = [[names[i] for i in shape] for shape in raw_keys]
    if not isinstance(keys, list) or not isinstance(strings, list):
        raise OutputWireError("invalid table dictionaries")
    if any(not isinstance(text, str) for text in strings):
        raise OutputWireError("dictionary contains non-string literal")
    if any(
        not isinstance(shape, list)
        or any(not isinstance(key, str) for key in shape)
        or len(shape) != len(set(shape))
        for shape in keys
    ):
        raise OutputWireError("invalid record key dictionary")

    def index(values, i):
        if type(i) is not int or not 0 <= i < len(values):
            raise OutputWireError("table reference outside dictionary")
        return values[i]

    def decode(item):
        if not isinstance(item, list):
            if isinstance(item, dict):
                if (
                    set(item) == {"number"}
                    and type(item["number"]) in (int, float)
                    and item["number"] < 0
                ):
                    return item["number"]
                raise OutputWireError("invalid numeric literal escape")
            if type(item) is int and item < 0:
                return index(strings, -1 - item)
            return item
        if not item:
            raise OutputWireError("missing table tag")
        if item[0] is None:
            return [decode(child) for child in item[1:]]
        if type(item[0]) is int:
            shape = index(keys, item[0])
            if len(shape) != len(item) - 1:
                raise OutputWireError("record does not match key shape")
            return dict(zip(shape, (decode(child) for child in item[1:]), strict=True))
        raise OutputWireError("unknown input table tag")

    return decode(packed["value"])


class OutputWireError(ValueError):
    pass


COPY_ID_FIELDS = {
    "Entity": "entity_id",
    "Event": "event_id",
    "LocalContextSchema": "schema_id",
    "LocalPredicateDefinition": "predicate_id",
    "LocalTypeDefinition": "type_id",
    "PropositionContent": "proposition_content_id",
    "QualifiedAssertion": "assertion_id",
}


def sealed_record_copies(sections: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index only complete records actually supplied in the sealed C1 input."""
    from story_projection_onto.contracts import OntologyDraft, canonical_json_schema

    definitions = canonical_json_schema(OntologyDraft)["$defs"]
    result: dict[str, dict[str, Any]] = {}

    def visit(item):
        if isinstance(item, dict):
            for name, id_field in COPY_ID_FIELDS.items():
                required = set(definitions[name]["required"]) - ADMIN
                if required <= set(item) and id_field in item:
                    records = result.setdefault(name, {})
                    identity = item[id_field]
                    if identity in records and records[identity] != item:
                        raise OutputWireError("conflicting sealed record identity")
                    records[identity] = copy.deepcopy(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(sections["sealed_ontology"])
    return result


class RecordTupleCodec:
    def __init__(
        self,
        schema: Mapping[str, Any],
        sealed_copies: Mapping | None = None,
        opaque_aliases: Mapping[str, str] | None = None,
    ):
        self.schema = copy.deepcopy(dict(schema))
        inverse = {v: k for k, v in (opaque_aliases or {}).items()}

        def alias_constraints(node):
            if isinstance(node, dict):
                if "enum" in node:
                    node["enum"] = [
                        inverse.get(v, v) if isinstance(v, str) else v for v in node["enum"]
                    ]
                if isinstance(node.get("const"), str):
                    node["const"] = inverse.get(node["const"], node["const"])
                for child in node.values():
                    alias_constraints(child)
            elif isinstance(node, list):
                for child in node:
                    alias_constraints(child)

        alias_constraints(self.schema)
        self.definitions = self.schema.get("$defs", {})
        self.sealed_copies = copy.deepcopy(dict(sealed_copies or {}))

    @staticmethod
    def fields(schema: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(sorted(set(schema.get("required", ())) - ADMIN))

    def resolve(self, schema: Mapping[str, Any]) -> Mapping[str, Any]:
        reference = schema.get("$ref")
        if reference:
            if not reference.startswith("#/$defs/"):
                raise OutputWireError("only local canonical schema references are supported")
            return self.definitions[reference.removeprefix("#/$defs/")]
        return schema

    def transform_schema(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(schema))
        if "$ref" in result:
            return result
        if "properties" in result:
            required = self.fields(result)
            properties = result["properties"]
            optional = {
                key: self.transform_schema(child)
                for key, child in properties.items()
                if key not in required and key not in ADMIN
            }
            result = {
                "type": "array",
                "minItems": len(required) + 1,
                "maxItems": len(required) + 1,
                "prefixItems": [self.transform_schema(properties[key]) for key in required]
                + [{"type": "object", "properties": optional, "additionalProperties": False}],
            }
        else:
            for key in ("items", "additionalProperties"):
                if isinstance(result.get(key), dict):
                    result[key] = self.transform_schema(result[key])
            for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
                if key in result:
                    result[key] = [self.transform_schema(child) for child in result[key]]
        return result

    def wire_schema(self) -> dict[str, Any]:
        root = {key: value for key, value in self.schema.items() if key != "$defs"}
        definitions = {}
        for name, value in self.definitions.items():
            wire = self.transform_schema(value)
            if name in self.sealed_copies:
                wire = {
                    "anyOf": [
                        wire,
                        {
                            "type": "object",
                            "properties": {
                                "sealed_id": {
                                    "type": "string",
                                    "enum": sorted(self.sealed_copies[name]),
                                }
                            },
                            "required": ["sealed_id"],
                            "additionalProperties": False,
                        },
                    ]
                }
            definitions[name] = wire
        return {
            "type": "object",
            "properties": {"draft": self.transform_schema(root)},
            "required": ["draft"],
            "additionalProperties": False,
            "$defs": definitions,
        }

    def legend(self) -> str:
        rows = [
            'Record-tuples-v1: return {"draft": ROOT}. Each record is an array in the '
            "listed field order, followed by an object of optional fields ({} if none). "
            "Keep all required values; optional defaults have their canonical meanings. "
            "Do not emit content_hash or schema_version; the canonical parser derives these. "
            "Use compact JSON without indentation. No fields or semantics may be dropped."
        ]
        for name, schema in [("ROOT", self.schema), *sorted(self.definitions.items())]:
            if "properties" in schema:
                fields = self.fields(schema)
                rows.append(f"{name}=[{','.join(fields)},{{optional canonical fields}}]")
        if self.sealed_copies:
            rows.append(
                "A complete unchanged record from the supplied sealed C1 may instead be "
                'returned as {"sealed_id":"its supplied ID"}. This copies exactly that '
                "record, including every qualification and supported description. "
                "No field overrides or novel IDs are allowed in a sealed reference. "
                "Use a full tuple when a permitted descriptive/relevance edit is needed."
            )
        return "\n".join(rows)

    def _convert(self, value: Any, schema: Mapping[str, Any], *, decode: bool) -> Any:
        name = str(schema.get("$ref", "")).removeprefix("#/$defs/")
        copies = self.sealed_copies.get(name, {})
        if decode and isinstance(value, dict) and "sealed_id" in value:
            if set(value) != {"sealed_id"} or value["sealed_id"] not in copies:
                raise OutputWireError("unknown or overridden sealed record reference")
            return copy.deepcopy(copies[value["sealed_id"]])
        if not decode and copies:
            for identity, record in copies.items():
                if record == value:
                    return {"sealed_id": identity}
        schema = self.resolve(schema)
        choices = schema.get("anyOf", schema.get("oneOf"))
        if choices:
            for choice in choices:
                try:
                    return self._convert(value, choice, decode=decode)
                except OutputWireError:
                    pass
            raise OutputWireError("value matches no canonical alternative")
        if "properties" in schema:
            fields = self.fields(schema)
            properties = schema["properties"]
            if decode:
                if not isinstance(value, list) or len(value) != len(fields) + 1:
                    raise OutputWireError("record tuple length differs from canonical fields")
                optional = value[-1]
                if not isinstance(optional, dict) or set(optional) & (set(fields) | ADMIN):
                    raise OutputWireError("optional record fields duplicate required/admin values")
                value = dict(zip(fields, value[:-1], strict=True)) | optional
            if not isinstance(value, dict):
                raise OutputWireError("canonical record must be an object")
            if set(value) - set(properties) or set(fields) - set(value):
                raise OutputWireError("unknown or missing canonical fields")
            converted = {
                key: self._convert(child, properties[key], decode=decode)
                for key, child in value.items()
                if key not in ADMIN
            }
            if decode:
                return converted
            return [converted[key] for key in fields] + [
                {key: child for key, child in converted.items() if key not in fields}
            ]
        expected = schema.get("type")
        checks = {
            "null": value is None,
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }
        if expected and not checks.get(expected, False):
            raise OutputWireError("wire value has wrong canonical type")
        if "enum" in schema and value not in schema["enum"]:
            raise OutputWireError("wire value outside canonical enum")
        if "const" in schema and value != schema["const"]:
            raise OutputWireError("wire value differs from canonical constant")
        if isinstance(value, list) and "items" in schema:
            return [self._convert(child, schema["items"], decode=decode) for child in value]
        if isinstance(value, dict) and isinstance(schema.get("additionalProperties"), dict):
            return {
                key: self._convert(child, schema["additionalProperties"], decode=decode)
                for key, child in value.items()
            }
        return copy.deepcopy(value)

    def encode(self, canonical: Mapping[str, Any]) -> dict[str, Any]:
        return {"draft": self._convert(dict(canonical), self.schema, decode=False)}

    def decode(self, wire: Mapping[str, Any]) -> dict[str, Any]:
        if set(wire) != {"draft"}:
            raise OutputWireError("wire root requires exactly draft")
        return self._convert(wire["draft"], self.schema, decode=True)


def capacity_messages(request, sections):
    """Build lossless model-visible text and its restricted inverse mapping."""
    from story_projection_onto.contracts import canonical_json, canonical_sha256
    from story_projection_onto.gpu_runtime import ChatMessage

    aliases = reference_aliases(sections)
    encoded = translate_references(sections, aliases, decode=False)
    if translate_references(encoded, aliases, decode=True) != sections:
        raise OutputWireError("opaque-reference input round-trip failed")
    copies = sealed_record_copies(encoded) if request.condition.value == "A-FixedSelect" else {}
    codec = RecordTupleCodec(request.output_schema, copies, aliases)
    messages = (
        ChatMessage(
            role="system",
            content=request.messages[0].content
            + "\n"
            + codec.legend()
            + "\n"
            + TABLE_INSTRUCTIONS
            + "\nI followed by digits is an immutable supplied opaque ID or hash handle. "
            "Preserve these references exactly. For new local IDs use short n-prefixed IDs. "
            "Do not use I-prefixed IDs for new objects. Prose is never an opaque reference. "
            "Opaque reference binding SHA256="
            + canonical_sha256(aliases)
            + "\nSealed copy binding SHA256="
            + canonical_sha256(copies),
        ),
        ChatMessage(role="user", content=canonical_json(pack_input_tables(encoded))),
    )
    return messages, aliases, copies


def pack_capacity_candidate(request, tokenizer, *, sections_override=None, schema_override=None):
    """Repack an existing, complete request under the symmetric candidate policy.

    This has no service/ledger side effects. It refuses packing overflow before
    a request can be returned. Constructive output has 6144 tokens; selection
    output has 3072 with exact sealed-record references, not graph truncation.
    Historical requests keep their original policy and hashes. No input is cut.
    """
    from dataclasses import asdict

    from story_projection_onto.contracts import canonical_sha256
    from story_projection_onto.gpu_runtime import GuidedJSONRequest
    from story_projection_onto.llm import DecodingManifest, PackingReport, PackingSection

    if sections_override is not None or schema_override is not None:
        from types import SimpleNamespace
        # Rebind the complete real FixedSelect input before compact packing. An
        # uncompressed intermediate need not fit; the complete wire input must.
        if request.condition.value != "A-FixedSelect" or sections_override is None or schema_override is None:
            raise OutputWireError("only complete paired FixedSelect overrides are permitted")
        request = SimpleNamespace(**{
            name: getattr(request, name) for name in (
                "request_id", "model_name", "condition", "decoding", "packing", "messages")
        }, output_schema=schema_override)
    actual_sections = sections_override if sections_override is not None else json.loads(request.messages[1].content)
    messages, aliases, copies = capacity_messages(request, actual_sections)
    codec = RecordTupleCodec(request.output_schema, copies, aliases)
    schema = codec.wire_schema()
    output_limit = 3072 if request.condition.value == "A-FixedSelect" else 6144
    input_limit = 12288 - output_limit
    count = len(
        tokenizer.apply_chat_template(
            [asdict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if count > input_limit:
        raise OutputWireError(
            f"complete candidate input {count} + output {output_limit} = {count + output_limit} "
            "exceeds unchanged total context 12288; no truncation"
        )
    decoding = DecodingManifest.model_validate(
        request.decoding.model_dump(exclude={"content_hash"})
        | {
            "maximum_input_tokens": input_limit,
            "maximum_output_tokens": output_limit,
            "output_schema_hash": canonical_sha256(schema),
        }
    )
    # Exact original section identities survive in the validated reversible
    # table. Count the actual rendered envelope once, not once per shared value.
    sections = tuple(
        PackingSection(
            name=s.name,
            section_content_hash=(
                canonical_sha256(schema) if s.name == "output_schema" else
                canonical_sha256(actual_sections[s.name]) if s.name in actual_sections else
                s.section_content_hash
            ),
            token_count=0,
        )
        for s in request.packing.sections
    )
    sections = (
        *sections,
        PackingSection(
            name="output_wire_legend",
            section_content_hash=canonical_sha256([asdict(m) for m in messages]),
            token_count=count,
        ),
    )
    old = request.packing
    packing = PackingReport.build(
        condition=request.condition,
        tokenizer_revision=decoding.tokenizer_revision,
        maximum_model_tokens=12288,
        maximum_input_tokens=input_limit,
        reserved_output_tokens=output_limit,
        sections=sections,
        required_section_names=(*old.required_section_names, "output_wire_legend"),
        complete_evidence_snapshot=old.complete_evidence_snapshot,
        complete_evidence_packet=old.complete_evidence_packet,
        complete_sealed_ontology=old.complete_sealed_ontology,
    )
    return GuidedJSONRequest(
        request_id=request.request_id,
        model_name=request.model_name,
        condition=request.condition,
        messages=messages,
        output_schema=schema,
        decoding=decoding,
        packing=packing,
        rendered_input_token_count=count,
        canonical_output_schema=request.output_schema,
        opaque_reference_aliases=aliases,
        sealed_record_copies=copies,
    )
