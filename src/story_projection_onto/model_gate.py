"""Strict one-fallback policy for the Phase-1 model feasibility gate.

This module never downloads or deletes a model.  It validates the only permitted
fallback candidate and emits an activation certificate only from a persisted,
failed primary acceptance result whose service is verified stopped.  Operational
cache replacement is a separate, explicitly invoked step so a dry plan cannot
mutate the shared cache.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Self, cast

from story_projection_onto.contracts import canonical_sha256

REGISTERED_FALLBACK_REPOSITORY = "Qwen/Qwen3-8B-AWQ"
REGISTERED_FALLBACK_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
REGISTERED_FALLBACK_ALIAS = "qwen3-8b-awq-fallback"
REGISTERED_PRIMARY_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
REGISTERED_PRIMARY_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root must be an object")
    return cast(dict[str, object], value)


def discover_cached_model_repositories(shared_cache: Path) -> tuple[str, ...]:
    """Return the exact known repository identities in the one shared cache.

    This is deliberately a read-only activation-time check.  It does not accept
    caller assertions about cache contents and it never downloads or removes a
    snapshot.
    """

    cache = shared_cache.resolve(strict=True)
    hub = cache / "hub"
    if not hub.is_dir():
        raise ValueError("shared cache has no Hugging Face hub directory")
    known_directories = {
        "models--Qwen--Qwen3-14B-AWQ": REGISTERED_PRIMARY_REPOSITORY,
        "models--Qwen--Qwen3-8B-AWQ": REGISTERED_FALLBACK_REPOSITORY,
    }
    repositories: list[str] = []
    for candidate in sorted(hub.glob("models--*"), key=lambda path: path.name):
        if not candidate.is_dir() or candidate.is_symlink():
            raise ValueError("model cache repository entries must be real directories")
        try:
            repositories.append(known_directories[candidate.name])
        except KeyError as exc:
            raise ValueError(
                f"shared cache contains unregistered model directory {candidate.name}"
            ) from exc
    return tuple(repositories)


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
        expected_identifiers = (
            "fallback-c1-01",
            "fallback-c2-01",
            "fallback-c2-02",
            "fallback-fixed-01",
        )
        if identifiers != expected_identifiers:
            raise ValueError("fallback micro-pilot call identities/order differ from protocol")
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
    legacy_activation_manifest_sha256: str | None = None,
    operational_replacement_receipt_manifest_sha256: str | None = None,
) -> Mapping[str, object]:
    """Authorize only the declared fallback after an immutable failed primary result."""

    if primary_result.get("kind") != "phase1_gpu_acceptance_result":
        raise ValueError("fallback activation requires a Phase-1 acceptance result")
    if primary_result.get("gate_passed") is not False:
        raise ValueError("fallback cannot activate unless the primary gate rejected")
    if primary_result.get("vllm_service_stopped") is not True:
        raise ValueError("fallback cannot activate while the primary service may be live")
    failure_classification = classify_primary_failure_for_fallback(primary_result)
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
        "primary_failure_classification": failure_classification,
        "fallback_policy_sha256": policy.content_hash,
        "fallback_repository": policy.repository,
        "fallback_revision": policy.revision,
        "fallback_served_model_name": policy.served_model_name,
        "required_cache_replacement": True,
        "maximum_inference_seconds": policy.maximum_inference_seconds,
        "reserved_service_start_events": policy.service_start_events,
    }
    provenance_hashes = (
        legacy_activation_manifest_sha256,
        operational_replacement_receipt_manifest_sha256,
    )
    if any(value is not None for value in provenance_hashes):
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in provenance_hashes
        ):
            raise ValueError("derived activation provenance hashes must be lowercase SHA-256")
        payload.update(
            {
                "legacy_activation_manifest_sha256": legacy_activation_manifest_sha256,
                "operational_replacement_receipt_manifest_sha256": (
                    operational_replacement_receipt_manifest_sha256
                ),
            }
        )
    return MappingProxyType({**payload, "manifest_sha256": canonical_sha256(payload)})


def classify_primary_failure_for_fallback(primary_result: Mapping[str, object]) -> str:
    """Return one protocol-eligible reason, rejecting controller/programming faults."""

    explicit = primary_result.get("fallback_eligible_failure_class")
    if explicit in {"latency", "resource", "packing", "structured_output"}:
        without_override = dict(primary_result)
        without_override.pop("fallback_eligible_failure_class", None)
        objective = classify_primary_failure_for_fallback(without_override)
        if objective != explicit:
            raise ValueError(
                "explicit fallback failure class lacks matching objective evidence"
            )
        return cast(str, explicit)
    if explicit is not None:
        raise ValueError("explicit fallback failure class is not protocol eligible")
    failure_type = primary_result.get("failure_type")
    if failure_type in {"RuntimeWatchdogTimeout", "GpuWatchdogExceeded"}:
        return "latency"
    if failure_type in {
        "RuntimeResourceLimitExceeded",
        "StorageBudgetExceeded",
        "GpuBudgetExceeded",
    }:
        return "resource"
    forecast = primary_result.get("full_manifest_forecast")
    continuation = primary_result.get("actual_plus_remaining_forecast")
    if (
        (isinstance(forecast, Mapping) and forecast.get("admitted") is False)
        or (
            isinstance(continuation, Mapping)
            and continuation.get("admitted") is False
        )
    ):
        return "latency"
    resource_watchdog = primary_result.get("resource_watchdog")
    if isinstance(resource_watchdog, Mapping) and resource_watchdog.get("failure_type") in {
        "RuntimeResourceLimitExceeded",
        "StorageBudgetExceeded",
    }:
        return "resource"
    packing_gate = primary_result.get("packing_gate")
    if isinstance(packing_gate, Mapping) and packing_gate.get("passed") is False:
        return "packing"
    calls = primary_result.get("calls")
    if isinstance(calls, list) and any(
        isinstance(call, Mapping)
        and call.get("status") == "failed"
        and isinstance(call.get("response_artifact_hash"), str)
        for call in calls
    ):
        # An immutable generated response existed and subsequently failed the
        # schema/capability/grounding validator.  This excludes startup and
        # controller failures that never produced model output.
        return "structured_output"
    operator_gate = primary_result.get("operator_coverage_gate")
    grounding_gate = primary_result.get("grounding_horizon_gate")
    if (
        (
            isinstance(operator_gate, Mapping)
            and (
                operator_gate.get("c1_complete") is False
                or operator_gate.get("c2_complete") is False
            )
        )
        or grounding_gate is False
    ):
        return "structured_output"
    raise ValueError(
        "primary rejection is not classified as latency, resource, packing, or structured output"
    )


def validate_fallback_activation_certificate(
    *,
    policy: FallbackModelPolicy,
    certificate: Mapping[str, object],
    primary_result: Mapping[str, object],
) -> Mapping[str, object]:
    """Revalidate a persisted authorization before any fallback execution."""

    supplied_hash = certificate.get("manifest_sha256")
    immutable = {key: value for key, value in certificate.items() if key != "manifest_sha256"}
    if not isinstance(supplied_hash, str) or supplied_hash != canonical_sha256(immutable):
        raise ValueError("fallback activation certificate hash does not match its contents")
    required = {
        "schema_version": "1.0.0",
        "kind": "fallback_activation_certificate",
        "authorized": True,
        "primary_rejection_manifest_sha256": primary_result.get("manifest_sha256"),
        "primary_failure_classification": classify_primary_failure_for_fallback(primary_result),
        "fallback_policy_sha256": policy.content_hash,
        "fallback_repository": policy.repository,
        "fallback_revision": policy.revision,
        "fallback_served_model_name": policy.served_model_name,
        "required_cache_replacement": True,
        "maximum_inference_seconds": policy.maximum_inference_seconds,
        "reserved_service_start_events": 1,
    }
    optional_provenance = {
        name: certificate.get(name)
        for name in (
            "legacy_activation_manifest_sha256",
            "operational_replacement_receipt_manifest_sha256",
        )
        if name in certificate
    }
    if optional_provenance:
        if set(optional_provenance) != {
            "legacy_activation_manifest_sha256",
            "operational_replacement_receipt_manifest_sha256",
        } or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in optional_provenance.values()
        ):
            raise ValueError("fallback activation has invalid legacy provenance hashes")
        required.update(optional_provenance)
    if immutable != required:
        raise ValueError("fallback activation certificate differs from the frozen authorization")
    # Re-run all primary-result identity/hash checks without making the obsolete
    # pre-replacement cache assertion part of execution-time validation.
    if primary_result.get("kind") != "phase1_gpu_acceptance_result":
        raise ValueError("fallback activation source is not a Phase-1 acceptance result")
    primary_hash = primary_result.get("manifest_sha256")
    primary_immutable = {
        key: value for key, value in primary_result.items() if key != "manifest_sha256"
    }
    if not isinstance(primary_hash, str) or primary_hash != canonical_sha256(primary_immutable):
        raise ValueError("primary rejection result manifest hash does not match its contents")
    if primary_result.get("gate_passed") is not False:
        raise ValueError("fallback activation source no longer records a rejected gate")
    if primary_result.get("vllm_service_stopped") is not True:
        raise ValueError("fallback activation source does not verify primary shutdown")
    runtime = primary_result.get("runtime")
    launcher = runtime.get("launcher") if isinstance(runtime, Mapping) else None
    if not isinstance(launcher, Mapping) or (
        launcher.get("repository"),
        launcher.get("revision"),
        launcher.get("model_candidate"),
    ) != (
        REGISTERED_PRIMARY_REPOSITORY,
        REGISTERED_PRIMARY_REVISION,
        "primary",
    ):
        raise ValueError("fallback activation source is not the registered primary model")
    return MappingProxyType(dict(certificate))


def fallback_cache_replacement_receipt(
    *,
    policy: FallbackModelPolicy,
    activation_certificate: Mapping[str, object],
    shared_cache: Path,
) -> Mapping[str, object]:
    """Certify the post-replacement cache without deleting or downloading files."""

    activation_hash = activation_certificate.get("manifest_sha256")
    if not isinstance(activation_hash, str) or activation_hash != canonical_sha256(
        {
            key: value
            for key, value in activation_certificate.items()
            if key != "manifest_sha256"
        }
    ):
        raise ValueError("cache replacement requires a valid activation certificate hash")
    repositories = discover_cached_model_repositories(shared_cache)
    if repositories != (policy.repository,):
        raise ValueError("post-replacement cache must contain exactly the fallback repository")
    snapshot = (
        shared_cache.resolve(strict=True)
        / "hub"
        / ("models--" + policy.repository.replace("/", "--"))
        / "snapshots"
        / policy.revision
    )
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise ValueError("post-replacement cache lacks the exact fallback snapshot")
    sibling_snapshots = tuple(path for path in snapshot.parent.iterdir() if path.is_dir())
    if sibling_snapshots != (snapshot,):
        raise ValueError("post-replacement cache contains more than one snapshot")
    primary_directory = (
        shared_cache.resolve(strict=True) / "hub" / "models--Qwen--Qwen3-14B-AWQ"
    )
    if primary_directory.exists():
        raise ValueError("rejected primary repository is still present after replacement")
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": "fallback_cache_replacement_receipt",
        "activation_certificate_sha256": activation_hash,
        "rejected_primary_repository_absent": True,
        "fallback_repository": policy.repository,
        "fallback_revision": policy.revision,
        "single_repository_in_shared_cache": True,
        "single_snapshot_in_shared_cache": True,
    }
    for name in (
        "legacy_activation_manifest_sha256",
        "operational_replacement_receipt_manifest_sha256",
    ):
        if name in activation_certificate:
            payload[name] = activation_certificate[name]
    return MappingProxyType({**payload, "manifest_sha256": canonical_sha256(payload)})


def validate_fallback_cache_replacement_receipt(
    *,
    policy: FallbackModelPolicy,
    activation_certificate: Mapping[str, object],
    receipt: Mapping[str, object],
    shared_cache: Path,
) -> Mapping[str, object]:
    """Recompute and compare the exact post-replacement receipt."""

    observed = fallback_cache_replacement_receipt(
        policy=policy,
        activation_certificate=activation_certificate,
        shared_cache=shared_cache,
    )
    if dict(receipt) != dict(observed):
        raise ValueError("fallback cache replacement receipt differs from current cache state")
    return MappingProxyType(dict(receipt))


def migrate_legacy_fallback_artifacts(
    *,
    policy: FallbackModelPolicy,
    primary_result: Mapping[str, object],
    legacy_activation: Mapping[str, object],
    operational_replacement_receipt: Mapping[str, object],
    shared_cache: Path,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Upgrade already-recorded fallback provenance without rewriting it.

    The original activation and rich operational replacement receipt remain
    immutable provenance.  This routine verifies both before deriving the
    stricter runtime activation and current-cache receipt used by the runner.
    """

    legacy_hash = legacy_activation.get("manifest_sha256")
    legacy_immutable = {
        key: value for key, value in legacy_activation.items() if key != "manifest_sha256"
    }
    if not isinstance(legacy_hash, str) or legacy_hash != canonical_sha256(legacy_immutable):
        raise ValueError("legacy fallback activation hash does not match its contents")
    expected_legacy = {
        "schema_version": "1.0.0",
        "kind": "fallback_activation_certificate",
        "authorized": True,
        "primary_rejection_manifest_sha256": primary_result.get("manifest_sha256"),
        "fallback_policy_sha256": policy.content_hash,
        "fallback_repository": policy.repository,
        "fallback_revision": policy.revision,
        "required_cache_replacement": True,
        "maximum_inference_seconds": policy.maximum_inference_seconds,
        "reserved_service_start_events": policy.service_start_events,
    }
    if legacy_immutable != expected_legacy:
        raise ValueError("legacy fallback activation differs from its original frozen shape")

    receipt_hash = operational_replacement_receipt.get("manifest_sha256")
    receipt_immutable = {
        key: value
        for key, value in operational_replacement_receipt.items()
        if key != "manifest_sha256"
    }
    if not isinstance(receipt_hash, str) or receipt_hash != canonical_sha256(receipt_immutable):
        raise ValueError("operational cache replacement receipt hash does not match")
    if operational_replacement_receipt.get("kind") != "model_cache_replacement_receipt":
        raise ValueError("migration requires the rich operational replacement receipt")
    activation = operational_replacement_receipt.get("activation")
    cache_state = operational_replacement_receipt.get("cache_state")
    primary = operational_replacement_receipt.get("primary")
    fallback = operational_replacement_receipt.get("fallback")
    if not all(
        isinstance(value, Mapping)
        for value in (activation, cache_state, primary, fallback)
    ):
        raise ValueError("operational replacement receipt is missing provenance sections")
    activation = cast(Mapping[str, object], activation)
    cache_state = cast(Mapping[str, object], cache_state)
    primary = cast(Mapping[str, object], primary)
    fallback = cast(Mapping[str, object], fallback)
    if activation.get("manifest_sha256") != legacy_hash:
        raise ValueError("operational replacement receipt targets another activation")
    if (
        primary.get("repository"),
        primary.get("revision"),
        fallback.get("repository"),
        fallback.get("revision"),
    ) != (
        REGISTERED_PRIMARY_REPOSITORY,
        REGISTERED_PRIMARY_REVISION,
        policy.repository,
        policy.revision,
    ):
        raise ValueError("operational replacement receipt has mixed model identities")
    required_cache_claims = {
        "primary_and_fallback_never_coexisted": True,
        "single_shared_cache": True,
        "model_repository_count": 1,
        "repository": policy.repository,
        "fallback_download_incomplete_file_count": 0,
    }
    if any(cache_state.get(name) != value for name, value in required_cache_claims.items()):
        raise ValueError("operational replacement receipt lacks exact cache-safety claims")
    if discover_cached_model_repositories(shared_cache) != (policy.repository,):
        raise ValueError("current cache no longer matches the operational replacement receipt")

    # Recheck eligibility now; a legacy TypeError/controller failure cannot be
    # upgraded merely because a fallback happened to have been downloaded.
    classify_primary_failure_for_fallback(primary_result)
    upgraded_activation = fallback_activation_certificate(
        policy=policy,
        primary_result=primary_result,
        cached_model_repositories=(REGISTERED_PRIMARY_REPOSITORY,),
        legacy_activation_manifest_sha256=cast(str, legacy_hash),
        operational_replacement_receipt_manifest_sha256=cast(str, receipt_hash),
    )
    runtime_receipt = fallback_cache_replacement_receipt(
        policy=policy,
        activation_certificate=upgraded_activation,
        shared_cache=shared_cache,
    )
    return upgraded_activation, runtime_receipt


def validate_fallback_snapshot_manifest(
    path: Path,
    *,
    policy: FallbackModelPolicy,
    policy_path: Path,
    snapshot_path: Path | None = None,
    shared_cache: Path | None = None,
) -> Mapping[str, object]:
    """Validate and optionally rehash the one installed fallback snapshot."""

    manifest = _load_object(path)
    supplied_hash = manifest.get("manifest_sha256")
    immutable = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if not isinstance(supplied_hash, str) or supplied_hash != canonical_sha256(immutable):
        raise ValueError("fallback model manifest hash does not match its contents")
    required = {
        "repository": policy.repository,
        "revision": policy.revision,
        "license": policy.license,
        "quantization": policy.quantization,
        "single_repository_in_shared_cache": True,
        "single_snapshot_in_shared_cache": True,
        "incomplete_file_count": 0,
        "model_configuration_sha256": _file_sha256(policy_path.resolve(strict=True)),
    }
    mismatches = [name for name, expected in required.items() if manifest.get(name) != expected]
    if mismatches:
        raise ValueError("verified fallback manifest differs at: " + ", ".join(mismatches))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("verified fallback manifest has no file hash inventory")
    paths: set[str] = set()
    expected_total = 0
    for entry in files:
        if not isinstance(entry, Mapping):
            raise ValueError("verified fallback file entry must be an object")
        relative = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in paths
        ):
            raise ValueError("verified fallback file entry has an unsafe or duplicate path")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("verified fallback file entry has no SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("verified fallback file entry has no nonnegative size")
        paths.add(relative)
        expected_total += size
    if manifest.get("file_count") != len(files) or manifest.get("total_bytes") != expected_total:
        raise ValueError("verified fallback manifest count or byte total is inconsistent")
    if (snapshot_path is None) != (shared_cache is None):
        raise ValueError("snapshot_path and shared_cache must be supplied together")
    if snapshot_path is None or shared_cache is None:
        return MappingProxyType(manifest)

    snapshot = snapshot_path.resolve(strict=True)
    cache = shared_cache.resolve(strict=True)
    try:
        snapshot.relative_to(cache)
    except ValueError as exc:
        raise ValueError("fallback snapshot is outside the one shared cache") from exc
    expected_repository_directory = "models--" + policy.repository.replace("/", "--")
    if (
        snapshot.name != policy.revision
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != expected_repository_directory
    ):
        raise ValueError("snapshot path does not identify the registered fallback")
    if discover_cached_model_repositories(cache) != (policy.repository,):
        raise ValueError("shared cache must contain only the registered fallback repository")
    discovered_snapshots = tuple(
        candidate.resolve()
        for candidate in snapshot.parent.iterdir()
        if candidate.is_dir() and not candidate.is_symlink()
    )
    if discovered_snapshots != (snapshot,):
        raise ValueError("shared cache must contain exactly one fallback snapshot")
    actual_paths = {
        candidate.relative_to(snapshot).as_posix()
        for candidate in snapshot.rglob("*")
        if candidate.is_file()
    }
    if actual_paths != paths:
        raise ValueError("fallback snapshot file set differs from its verified manifest")
    for entry in files:
        candidate = snapshot / cast(str, entry["path"])
        try:
            candidate.resolve(strict=True).relative_to(cache)
        except ValueError as exc:
            raise ValueError("fallback snapshot file resolves outside the shared cache") from exc
        if candidate.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"fallback snapshot size changed for {entry['path']}")
        if _file_sha256(candidate) != entry["sha256"]:
            raise ValueError(f"fallback snapshot hash changed for {entry['path']}")
    model_config = _load_object(snapshot / "config.json")
    quantization = model_config.get("quantization_config")
    if not isinstance(quantization, Mapping) or (
        str(quantization.get("quant_method", "")).casefold() != "awq"
    ):
        raise ValueError("fallback model config does not declare AWQ quantization")
    if any(candidate.is_file() for candidate in cache.rglob("*.incomplete")):
        raise ValueError("shared model cache still contains an incomplete file")
    return MappingProxyType(manifest)


__all__ = [
    "REGISTERED_FALLBACK_ALIAS",
    "REGISTERED_FALLBACK_REPOSITORY",
    "REGISTERED_FALLBACK_REVISION",
    "REGISTERED_PRIMARY_REPOSITORY",
    "REGISTERED_PRIMARY_REVISION",
    "FallbackModelPolicy",
    "FallbackPilotCall",
    "FallbackRepairPolicy",
    "classify_primary_failure_for_fallback",
    "discover_cached_model_repositories",
    "fallback_activation_certificate",
    "fallback_cache_replacement_receipt",
    "migrate_legacy_fallback_artifacts",
    "validate_fallback_activation_certificate",
    "validate_fallback_cache_replacement_receipt",
    "validate_fallback_snapshot_manifest",
]
