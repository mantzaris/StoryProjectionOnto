"""Strict one-fallback policy for the Phase-1 model feasibility gate.

This module never downloads or deletes a model.  It validates the only permitted
fallback candidate and emits an activation certificate only from a persisted,
failed primary acceptance result whose service is verified stopped.  Operational
cache replacement is a separate, explicitly invoked step so a dry plan cannot
mutate the shared cache.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self, cast

from story_projection_onto.contracts import canonical_sha256

REGISTERED_FALLBACK_REPOSITORY = "Qwen/Qwen3-8B-AWQ"
REGISTERED_FALLBACK_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
REGISTERED_FALLBACK_ALIAS = "qwen3-8b-awq-fallback"
REGISTERED_PRIMARY_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
REGISTERED_PRIMARY_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"


@dataclass(frozen=True, slots=True)
class FallbackPilotCall:
    call_id: str
    form: Literal["C1", "C2", "A-FixedSelect"]
    reserve_tier: Literal["reserve_long", "reserve_standard", "reserve_short"]
    watchdog_seconds: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = {"call_id", "form", "reserve_tier", "watchdog_seconds"}
        if set(value) != expected:
            raise ValueError("fallback pilot call fields differ from the frozen policy")
        return cls(
            call_id=cast(str, value["call_id"]),
            form=cast(Literal["C1", "C2", "A-FixedSelect"], value["form"]),
            reserve_tier=cast(
                Literal["reserve_long", "reserve_standard", "reserve_short"],
                value["reserve_tier"],
            ),
            watchdog_seconds=cast(int, value["watchdog_seconds"]),
        )

    def __post_init__(self) -> None:
        if not self.call_id or self.call_id.strip() != self.call_id:
            raise ValueError("fallback call_id must be nonempty and stripped")
        allowed_forms = {"C1", "C2", "A-FixedSelect"}
        allowed_tiers = {"reserve_long", "reserve_standard", "reserve_short"}
        if self.form not in allowed_forms or self.reserve_tier not in allowed_tiers:
            raise ValueError("fallback call form or reserve tier is unregistered")
        tier_watchdogs = {
            "reserve_long": 240,
            "reserve_standard": 150,
            "reserve_short": 90,
        }
        if isinstance(self.watchdog_seconds, bool) or (
            self.watchdog_seconds != tier_watchdogs[self.reserve_tier]
        ):
            raise ValueError("fallback call watchdog differs from its reserve tier")


@dataclass(frozen=True, slots=True)
class FallbackRepairPolicy:
    maximum_calls: int
    reserve_tier: Literal["reserve_short"]
    watchdog_seconds: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if set(value) != {"maximum_calls", "reserve_tier", "watchdog_seconds"}:
            raise ValueError("fallback repair fields differ from the frozen policy")
        return cls(
            maximum_calls=cast(int, value["maximum_calls"]),
            reserve_tier=cast(Literal["reserve_short"], value["reserve_tier"]),
            watchdog_seconds=cast(int, value["watchdog_seconds"]),
        )

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_calls, bool)
            or self.maximum_calls != 1
            or self.reserve_tier != "reserve_short"
            or isinstance(self.watchdog_seconds, bool)
            or self.watchdog_seconds != 90
        ):
            raise ValueError("fallback permits at most one 90-second short repair")


@dataclass(frozen=True, slots=True)
class FallbackModelPolicy:
    schema_version: str
    activation: str
    repository: str
    revision: str
    served_model_name: str
    license: str
    quantization: str
    model_search_allowed: bool
    maximum_simultaneous_model_snapshots: int
    delete_verified_rejected_primary_before_download: bool
    service_start_events: int
    calls: tuple[FallbackPilotCall, ...]
    repair: FallbackRepairPolicy

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        expected = {
            "schema_version",
            "activation",
            "repository",
            "revision",
            "served_model_name",
            "license",
            "quantization",
            "model_search_allowed",
            "maximum_simultaneous_model_snapshots",
            "delete_verified_rejected_primary_before_download",
            "micro_pilot",
        }
        if set(value) != expected:
            raise ValueError("fallback model policy fields differ from the frozen protocol")
        raw_pilot = value["micro_pilot"]
        if not isinstance(raw_pilot, Mapping) or set(raw_pilot) != {
            "service_start_events",
            "calls",
            "repair",
        }:
            raise ValueError("fallback micro-pilot fields differ from the frozen protocol")
        raw_calls = raw_pilot["calls"]
        raw_repair = raw_pilot["repair"]
        if not isinstance(raw_calls, list) or not all(
            isinstance(item, Mapping) for item in raw_calls
        ):
            raise ValueError("fallback calls must be an array of objects")
        if not isinstance(raw_repair, Mapping):
            raise ValueError("fallback repair policy must be an object")
        return cls(
            schema_version=cast(str, value["schema_version"]),
            activation=cast(str, value["activation"]),
            repository=cast(str, value["repository"]),
            revision=cast(str, value["revision"]),
            served_model_name=cast(str, value["served_model_name"]),
            license=cast(str, value["license"]),
            quantization=cast(str, value["quantization"]),
            model_search_allowed=cast(bool, value["model_search_allowed"]),
            maximum_simultaneous_model_snapshots=cast(
                int, value["maximum_simultaneous_model_snapshots"]
            ),
            delete_verified_rejected_primary_before_download=cast(
                bool, value["delete_verified_rejected_primary_before_download"]
            ),
            service_start_events=cast(int, raw_pilot["service_start_events"]),
            calls=tuple(FallbackPilotCall.from_mapping(item) for item in raw_calls),
            repair=FallbackRepairPolicy.from_mapping(raw_repair),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load fallback model policy: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("fallback model policy root must be an object")
        return cls.from_mapping(raw)

    def __post_init__(self) -> None:
        exact = {
            "schema_version": "1.0.0",
            "activation": "only_after_primary_phase1_rejection",
            "repository": REGISTERED_FALLBACK_REPOSITORY,
            "revision": REGISTERED_FALLBACK_REVISION,
            "served_model_name": REGISTERED_FALLBACK_ALIAS,
            "license": "Apache-2.0",
            "quantization": "awq",
            "model_search_allowed": False,
            "maximum_simultaneous_model_snapshots": 1,
            "delete_verified_rejected_primary_before_download": True,
            "service_start_events": 1,
        }
        mismatches = [name for name, expected in exact.items() if getattr(self, name) != expected]
        if mismatches:
            raise ValueError("fallback policy differs at: " + ", ".join(mismatches))
        if len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        ):
            raise ValueError("fallback revision must be an immutable 40-character commit")
        identifiers = tuple(item.call_id for item in self.calls)
        if len(identifiers) != 4 or len(set(identifiers)) != 4:
            raise ValueError("fallback micro-pilot requires four unique base calls")
        forms = Counter(item.form for item in self.calls)
        tiers = Counter(item.reserve_tier for item in self.calls)
        if forms != {"C1": 1, "C2": 2, "A-FixedSelect": 1}:
            raise ValueError("fallback call-form counts differ from the protocol")
        if tiers != {"reserve_long": 1, "reserve_standard": 2, "reserve_short": 1}:
            raise ValueError("fallback reserve consumption differs from the protocol")

    @property
    def maximum_inference_seconds(self) -> int:
        return sum(item.watchdog_seconds for item in self.calls) + (
            self.repair.maximum_calls * self.repair.watchdog_seconds
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(asdict(self))


def fallback_activation_certificate(
    *,
    policy: FallbackModelPolicy,
    primary_result: Mapping[str, object],
    cached_model_repositories: tuple[str, ...],
) -> Mapping[str, object]:
    """Authorize only the declared fallback after an immutable failed primary result."""

    if primary_result.get("kind") != "phase1_gpu_acceptance_result":
        raise ValueError("fallback activation requires a Phase-1 acceptance result")
    if primary_result.get("gate_passed") is not False:
        raise ValueError("fallback cannot activate unless the primary gate rejected")
    if primary_result.get("vllm_service_stopped") is not True:
        raise ValueError("fallback cannot activate while the primary service may be live")
    primary_hash = primary_result.get("manifest_sha256")
    if not isinstance(primary_hash, str) or len(primary_hash) != 64:
        raise ValueError("primary rejection result lacks an immutable manifest hash")
    immutable_primary = {
        key: value for key, value in primary_result.items() if key != "manifest_sha256"
    }
    if primary_hash != canonical_sha256(immutable_primary):
        raise ValueError("primary rejection result manifest hash does not match its contents")
    runtime = primary_result.get("runtime")
    launcher = runtime.get("launcher") if isinstance(runtime, Mapping) else None
    if not isinstance(launcher, Mapping):
        raise ValueError("primary rejection result lacks launcher identity")
    if (
        launcher.get("repository") != REGISTERED_PRIMARY_REPOSITORY
        or launcher.get("revision") != REGISTERED_PRIMARY_REVISION
        or launcher.get("model_candidate") != "primary"
    ):
        raise ValueError("fallback activation source is not the registered primary model")
    expected_cache = (REGISTERED_PRIMARY_REPOSITORY,)
    if cached_model_repositories != expected_cache:
        raise ValueError(
            "shared cache must contain exactly the rejected primary before replacement"
        )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "fallback_activation_certificate",
        "authorized": True,
        "primary_rejection_manifest_sha256": primary_hash,
        "fallback_policy_sha256": policy.content_hash,
        "fallback_repository": policy.repository,
        "fallback_revision": policy.revision,
        "required_cache_replacement": True,
        "maximum_inference_seconds": policy.maximum_inference_seconds,
        "reserved_service_start_events": policy.service_start_events,
    }
    return MappingProxyType({**payload, "manifest_sha256": canonical_sha256(payload)})


__all__ = [
    "REGISTERED_FALLBACK_ALIAS",
    "REGISTERED_FALLBACK_REPOSITORY",
    "REGISTERED_FALLBACK_REVISION",
    "REGISTERED_PRIMARY_REPOSITORY",
    "REGISTERED_PRIMARY_REVISION",
    "FallbackModelPolicy",
    "FallbackPilotCall",
    "FallbackRepairPolicy",
    "fallback_activation_certificate",
]
