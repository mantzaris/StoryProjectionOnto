"""Frozen call inventory, forecasting, cost attribution, and GPU metering.

This module deliberately contains no model client.  It is the runner-side
control plane which decides whether a request may start and records every
mutually-exclusive allocated GPU interval in :mod:`story_projection_onto.store`.
The scientific inventory is validated against the authoritative 278-attempt,
eight-start manifest rather than accepting a conveniently smaller run.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Self

from story_projection_onto.store import (
    GpuAllocationJournalState,
    GpuBudgetExceeded,
    GpuEvent,
    GpuEventKind,
    GpuServiceJournalRecord,
    GpuServiceJournalState,
    GpuServiceSession,
    Ledger,
)

REGISTERED_INFERENCE_ATTEMPTS = 278
REGISTERED_SESSION_STARTS = 8
REGISTERED_ACCOUNTING_EVENTS = 286
REGISTERED_PROVISIONAL_SECONDS = 31_229
REGISTERED_SCHEDULED_SECONDS = 9 * 60 * 60
REGISTERED_PREFERRED_SECONDS = 8.25 * 60 * 60
REGISTERED_HARD_SECONDS = 10 * 60 * 60
SESSION_START_CLASS = "gpu_session_start"
REGISTERED_STORAGE_PHASES = tuple(f"phase_{index}" for index in range(1, 8))
REGISTERED_CALL_CLASSES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "gpu_session_start": (8, 180),
        "acceptance_c1": (2, 180),
        "acceptance_c2": (3, 120),
        "acceptance_fixed_select": (2, 90),
        "acceptance_repair": (1, 90),
        "development_c1": (4, 180),
        "development_c2": (12, 120),
        "development_fixed_select": (4, 90),
        "development_ablation": (3, 120),
        "development_repair": (1, 90),
        "test_c1": (24, 180),
        "test_c2": (72, 95),
        "test_fixed_select": (72, 72),
        "paraphrase_c2": (12, 95),
        "scripted_feedback_c2": (6, 95),
        "researcher_trace_c2": (3, 95),
        "ablation_no_context": (12, 95),
        "ablation_no_temporal_epistemic": (8, 95),
        "ablation_no_rare_guard": (8, 95),
        "case_c1": (4, 240),
        "case_c2": (8, 150),
        "case_full_index_c2": (1, 150),
        "reserve_long": (4, 240),
        "reserve_standard": (8, 150),
        "reserve_short": (4, 90),
    }
)


class InventoryValidationError(ValueError):
    """The call inventory does not match its own totals or the frozen plan."""


class ForecastAdmissionError(GpuBudgetExceeded):
    """A request cannot be admitted inside the scheduled GPU envelope."""


class GpuAccountingRegression(RuntimeError):
    """An append-only ledger's observed allocated total moved backwards."""


class GpuWatchdogExceeded(GpuBudgetExceeded):
    """A metered interval exceeded the maximum admitted before it started."""


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InventoryValidationError(f"{name} must be a positive integer")
    return value


def _nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a nonnegative finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return numeric


@dataclass(frozen=True)
class CallClassPlan:
    """One immutable row in the registered GPU call manifest."""

    name: str
    count: int
    provisional_p95_seconds: int

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise InventoryValidationError("call-class name must be nonempty and stripped")
        _positive_int(f"{self.name}.count", self.count)
        _positive_int(f"{self.name}.provisional_p95_seconds", self.provisional_p95_seconds)

    @property
    def provisional_seconds(self) -> int:
        return self.count * self.provisional_p95_seconds


@dataclass(frozen=True)
class GPUCallInventory:
    """Typed, self-reconciling representation of ``gpu_call_inventory.json``."""

    schema_version: str
    accounting_events: int
    maximum_inference_attempts: int
    classes: tuple[CallClassPlan, ...]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise InventoryValidationError("schema_version must be nonempty")
        _positive_int("accounting_events", self.accounting_events)
        _positive_int("maximum_inference_attempts", self.maximum_inference_attempts)
        if not self.classes:
            raise InventoryValidationError("call inventory must contain classes")
        names = tuple(item.name for item in self.classes)
        if len(set(names)) != len(names):
            raise InventoryValidationError("call-class names must be unique")
        if self.computed_accounting_events != self.accounting_events:
            raise InventoryValidationError(
                "accounting_events does not equal the sum of call-class counts"
            )
        if self.computed_inference_attempts != self.maximum_inference_attempts:
            raise InventoryValidationError(
                "maximum_inference_attempts excludes exactly the session-start row"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        enforce_registered_plan: bool = True,
    ) -> Self:
        raw_classes = value.get("classes")
        if not isinstance(raw_classes, list):
            raise InventoryValidationError("classes must be a JSON array")
        classes: list[CallClassPlan] = []
        for index, raw in enumerate(raw_classes):
            if not isinstance(raw, Mapping):
                raise InventoryValidationError(f"classes[{index}] must be an object")
            name = raw.get("name")
            if not isinstance(name, str):
                raise InventoryValidationError(f"classes[{index}].name must be a string")
            classes.append(
                CallClassPlan(
                    name=name,
                    count=_positive_int(f"classes[{index}].count", raw.get("count")),
                    provisional_p95_seconds=_positive_int(
                        f"classes[{index}].provisional_p95_seconds",
                        raw.get("provisional_p95_seconds"),
                    ),
                )
            )
        schema_version = value.get("schema_version")
        if not isinstance(schema_version, str):
            raise InventoryValidationError("schema_version must be a string")
        inventory = cls(
            schema_version=schema_version,
            accounting_events=_positive_int("accounting_events", value.get("accounting_events")),
            maximum_inference_attempts=_positive_int(
                "maximum_inference_attempts", value.get("maximum_inference_attempts")
            ),
            classes=tuple(classes),
        )
        if enforce_registered_plan:
            inventory.validate_registered_plan()
        return inventory

    @classmethod
    def load(cls, path: str | Path, *, enforce_registered_plan: bool = True) -> Self:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InventoryValidationError(f"cannot load GPU call inventory: {exc}") from exc
        if not isinstance(value, Mapping):
            raise InventoryValidationError("GPU call inventory root must be an object")
        return cls.from_mapping(value, enforce_registered_plan=enforce_registered_plan)

    @property
    def by_name(self) -> Mapping[str, CallClassPlan]:
        return {item.name: item for item in self.classes}

    @property
    def session_start_count(self) -> int:
        row = self.by_name.get(SESSION_START_CLASS)
        return 0 if row is None else row.count

    @property
    def computed_accounting_events(self) -> int:
        return sum(item.count for item in self.classes)

    @property
    def computed_inference_attempts(self) -> int:
        return sum(item.count for item in self.classes if item.name != SESSION_START_CLASS)

    @property
    def provisional_planned_seconds(self) -> int:
        return sum(item.provisional_seconds for item in self.classes)

    def call_class(self, name: str) -> CallClassPlan:
        try:
            return self.by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown registered call class {name!r}") from exc

    def validate_registered_plan(self) -> None:
        expected_reserves = {
            "reserve_long": (4, 240),
            "reserve_standard": (8, 150),
            "reserve_short": (4, 90),
        }
        actual_reserves = {
            name: (self.call_class(name).count, self.call_class(name).provisional_p95_seconds)
            for name in expected_reserves
            if name in self.by_name
        }
        failures: list[str] = []
        actual_classes = {
            item.name: (item.count, item.provisional_p95_seconds) for item in self.classes
        }
        if actual_classes != REGISTERED_CALL_CLASSES:
            failures.append("call classes/counts/provisional ceilings differ from the frozen plan")
        if self.maximum_inference_attempts != REGISTERED_INFERENCE_ATTEMPTS:
            failures.append(f"inference attempts must be {REGISTERED_INFERENCE_ATTEMPTS}")
        if self.session_start_count != REGISTERED_SESSION_STARTS:
            failures.append(f"session starts must be {REGISTERED_SESSION_STARTS}")
        if self.accounting_events != REGISTERED_ACCOUNTING_EVENTS:
            failures.append(f"accounting events must be {REGISTERED_ACCOUNTING_EVENTS}")
        if self.provisional_planned_seconds != REGISTERED_PROVISIONAL_SECONDS:
            failures.append(f"provisional seconds must be {REGISTERED_PROVISIONAL_SECONDS}")
        if actual_reserves != expected_reserves:
            failures.append("reserve tiers must remain fixed at long=4, standard=8, short=4")
        if failures:
            raise InventoryValidationError("; ".join(failures))


# Mixed-case alias retained for callers following the rest of the package's ``Gpu*`` names.
GpuCallInventory = GPUCallInventory


@dataclass(frozen=True)
class ResourceLimits:
    """Typed resource-limit configuration used by admission and metering."""

    schema_version: str
    maximum_cpu_workers: int
    maximum_process_ram_bytes: int
    maximum_peak_vram_bytes: int
    maximum_project_occupied_bytes: int
    minimum_storage_headroom_bytes: int
    maximum_project_allocation_bytes: int
    scheduled_gpu_seconds: int
    preferred_forecast_gpu_seconds: int
    hard_gpu_seconds: int
    model_cpu_offload_allowed: bool
    generation_concurrency: int

    def __post_init__(self) -> None:
        integer_fields = (
            "maximum_cpu_workers",
            "maximum_process_ram_bytes",
            "maximum_peak_vram_bytes",
            "maximum_project_occupied_bytes",
            "minimum_storage_headroom_bytes",
            "maximum_project_allocation_bytes",
            "scheduled_gpu_seconds",
            "preferred_forecast_gpu_seconds",
            "hard_gpu_seconds",
            "generation_concurrency",
        )
        for name in integer_fields:
            _positive_int(name, getattr(self, name))
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be nonempty")
        if not (
            self.preferred_forecast_gpu_seconds
            <= self.scheduled_gpu_seconds
            < self.hard_gpu_seconds
        ):
            raise ValueError("GPU limits must satisfy preferred <= scheduled < hard")
        if self.maximum_project_occupied_bytes + self.minimum_storage_headroom_bytes > (
            self.maximum_project_allocation_bytes
        ):
            raise ValueError("occupied storage plus headroom exceeds the allocation")
        if not isinstance(self.model_cpu_offload_allowed, bool):
            raise ValueError("model_cpu_offload_allowed must be a boolean")
        if self.model_cpu_offload_allowed:
            raise ValueError("CPU model-weight offload is forbidden")
        if self.generation_concurrency != 1:
            raise ValueError("generation concurrency must remain one")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        field_names = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        missing = [name for name in field_names if name not in value]
        if missing:
            raise ValueError(f"resource limits missing fields: {', '.join(missing)}")
        unknown = sorted(set(value) - set(field_names))
        if unknown:
            raise ValueError(f"resource limits contain unknown fields: {', '.join(unknown)}")
        return cls(**{name: value[name] for name in field_names})  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> Self:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load resource limits: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("resource-limit root must be an object")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class PhaseStorageReservation:
    """Concurrent writable-space reservation applied at one phase boundary."""

    declared_growth_bytes: int
    largest_atomic_temporary_bytes: int
    quarantine_allowance_bytes: int
    release_staging_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        field_names = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        missing = [name for name in field_names if name not in value]
        if missing:
            raise ValueError("storage reservation missing fields: " + ", ".join(missing))
        unknown = sorted(set(value) - set(field_names))
        if unknown:
            raise ValueError("storage reservation contains unknown fields: " + ", ".join(unknown))
        return cls(**{name: value[name] for name in field_names})  # type: ignore[arg-type]

    @property
    def additional_reserved_bytes(self) -> int:
        return sum(getattr(self, name) for name in self.__dataclass_fields__)  # type: ignore[attr-defined]

    def preflight_arguments(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__  # type: ignore[attr-defined]
        }


@dataclass(frozen=True)
class StorageAllocationPlan:
    """Typed seven-phase storage reservations loaded from tracked configuration."""

    schema_version: str
    reservations: Mapping[str, PhaseStorageReservation]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("storage allocation schema_version must be nonempty")
        actual = tuple(sorted(self.reservations))
        expected = tuple(sorted(REGISTERED_STORAGE_PHASES))
        if actual != expected:
            raise ValueError("storage allocation phases must be exactly: " + ", ".join(expected))
        if not all(
            isinstance(item, PhaseStorageReservation) for item in self.reservations.values()
        ):
            raise ValueError("storage allocation contains an untyped reservation")
        object.__setattr__(self, "reservations", MappingProxyType(dict(self.reservations)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        unknown = sorted(set(value) - {"schema_version", "reservations"})
        if unknown:
            raise ValueError(
                "storage allocation plan contains unknown fields: " + ", ".join(unknown)
            )
        schema_version = value.get("schema_version")
        raw_reservations = value.get("reservations")
        if not isinstance(schema_version, str):
            raise ValueError("storage allocation schema_version must be a string")
        if not isinstance(raw_reservations, Mapping):
            raise ValueError("storage allocation reservations must be an object")
        reservations: dict[str, PhaseStorageReservation] = {}
        for phase, raw_reservation in raw_reservations.items():
            if not isinstance(phase, str) or not isinstance(raw_reservation, Mapping):
                raise ValueError("each storage phase must map to a reservation object")
            reservations[phase] = PhaseStorageReservation.from_mapping(raw_reservation)
        return cls(schema_version=schema_version, reservations=reservations)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load storage allocation plan: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("storage allocation plan root must be an object")
        return cls.from_mapping(value)

    def reservation_for(self, phase: str) -> PhaseStorageReservation:
        try:
            return self.reservations[phase]
        except KeyError as exc:
            raise KeyError(f"unknown storage phase {phase!r}") from exc


def load_gpu_call_inventory(path: str | Path) -> GPUCallInventory:
    return GPUCallInventory.load(path)


def load_resource_limits(path: str | Path) -> ResourceLimits:
    return ResourceLimits.load(path)


@dataclass(frozen=True)
class TimingObservation:
    call_class: str
    allocated_seconds: float

    def __post_init__(self) -> None:
        if not self.call_class:
            raise ValueError("call_class must be nonempty")
        _nonnegative_finite("allocated_seconds", self.allocated_seconds)


@dataclass(frozen=True)
class CallClassTiming:
    call_class: str
    sample_count: int
    p50_seconds: float
    p95_seconds: float
    maximum_seconds: float


def nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    """Return the registered nearest-rank percentile (never an interpolated quantile)."""

    if not values:
        raise ValueError("nearest-rank percentile requires at least one observation")
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    ordered = sorted(_nonnegative_finite("timing observation", value) for value in values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def summarize_call_class_timings(
    observations: Iterable[TimingObservation] | Mapping[str, Sequence[float]],
) -> tuple[CallClassTiming, ...]:
    grouped: dict[str, list[float]] = defaultdict(list)
    if isinstance(observations, Mapping):
        for call_class, values in observations.items():
            if not call_class:
                raise ValueError("call_class must be nonempty")
            grouped[call_class].extend(
                _nonnegative_finite("allocated_seconds", value) for value in values
            )
    else:
        for observation in observations:
            grouped[observation.call_class].append(observation.allocated_seconds)
    summaries = []
    for call_class in sorted(grouped):
        values = grouped[call_class]
        if not values:
            continue
        summaries.append(
            CallClassTiming(
                call_class=call_class,
                sample_count=len(values),
                p50_seconds=nearest_rank_percentile(values, 0.50),
                p95_seconds=nearest_rank_percentile(values, 0.95),
                maximum_seconds=max(values),
            )
        )
    return tuple(summaries)


class ForecastSource(StrEnum):
    MEASURED = "measured"
    CONSERVATIVE_PROXY = "conservative_proxy"
    PROVISIONAL = "provisional"
    REGISTERED_RESERVE = "registered_reserve"


class AdmissionSignal(StrEnum):
    PREFERRED = "preferred_margin"
    ADMITTED = "admitted_above_preferred"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ForecastRow:
    call_class: str
    count: int
    provisional_p95_seconds: float
    p50_seconds: float | None
    measured_p95_seconds: float | None
    forecast_p95_seconds: float
    forecast_seconds: float
    source: ForecastSource
    proxy_call_class: str | None = None


@dataclass(frozen=True)
class GPUForecast:
    rows: tuple[ForecastRow, ...]
    total_seconds: float
    scheduled_limit_seconds: float
    preferred_limit_seconds: float
    hard_limit_seconds: float

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600

    @property
    def admitted(self) -> bool:
        return self.total_seconds <= self.scheduled_limit_seconds

    @property
    def within_preferred_margin(self) -> bool:
        return self.total_seconds <= self.preferred_limit_seconds

    @property
    def admission_signal(self) -> AdmissionSignal:
        if not self.admitted:
            return AdmissionSignal.REJECTED
        if self.within_preferred_margin:
            return AdmissionSignal.PREFERRED
        return AdmissionSignal.ADMITTED

    @property
    def scheduled_slack_seconds(self) -> float:
        return self.scheduled_limit_seconds - self.total_seconds

    @property
    def preferred_slack_seconds(self) -> float:
        return self.preferred_limit_seconds - self.total_seconds

    def require_admission(self) -> None:
        if not self.admitted:
            raise ForecastAdmissionError(
                f"forecast {self.total_seconds:.6f}s exceeds scheduled "
                f"limit {self.scheduled_limit_seconds:.6f}s"
            )


def _proxy_family(name: str) -> str:
    if name == SESSION_START_CLASS:
        return "session_start"
    if name.startswith("reserve_"):
        return "reserve"
    if "fixed_select" in name:
        return "fixed_select"
    if "repair" in name:
        return "repair"
    if "c1" in name:
        return "c1"
    if "c2" in name or "ablation" in name:
        return "c2"
    return name


def forecast_gpu_schedule(
    inventory: GPUCallInventory,
    observations: Iterable[TimingObservation] | Mapping[str, Sequence[float]] = (),
    *,
    limits: ResourceLimits | None = None,
    minimum_direct_p95_samples: int = 20,
) -> GPUForecast:
    """Reforecast every row using measured p95 or a conservative longer proxy.

    With fewer than 20 observations, nearest-rank p95 is necessarily at or near
    the observed maximum.  The forecast additionally considers observed rows in
    the same semantic family whose registered workload ceiling is at least as
    long.  Reserve rows are never reallocated or shortened by pilot timing.
    """

    if minimum_direct_p95_samples <= 0:
        raise ValueError("minimum_direct_p95_samples must be positive")
    timings = {item.call_class: item for item in summarize_call_class_timings(observations)}
    unknown = sorted(set(timings) - set(inventory.by_name))
    if unknown:
        raise ValueError(f"timings contain unregistered call classes: {', '.join(unknown)}")

    rows: list[ForecastRow] = []
    for plan in inventory.classes:
        own = timings.get(plan.name)
        source = ForecastSource.PROVISIONAL
        proxy_name: str | None = None
        if plan.name.startswith("reserve_"):
            effective = float(plan.provisional_p95_seconds)
            source = ForecastSource.REGISTERED_RESERVE
        elif own is not None and own.sample_count >= minimum_direct_p95_samples:
            effective = own.p95_seconds
            source = ForecastSource.MEASURED
        else:
            # A sub-20 sample p95 is at/near its sample maximum.  Do not let a
            # single unusually fast call silently replace the registered
            # ceiling unless a separate equal-or-longer workload supplies the
            # conservative proxy required by the plan.
            candidates: list[CallClassTiming] = []
            for candidate_name, candidate in timings.items():
                if candidate_name == plan.name:
                    continue
                candidate_plan = inventory.call_class(candidate_name)
                if _proxy_family(candidate_name) != _proxy_family(plan.name):
                    continue
                if candidate_plan.provisional_p95_seconds < plan.provisional_p95_seconds:
                    continue
                candidates.append(candidate)
            if candidates:
                chosen = max(candidates, key=lambda item: (item.p95_seconds, item.call_class))
                effective = max(
                    chosen.p95_seconds,
                    own.p95_seconds if own is not None else 0.0,
                )
                if own is not None and own.p95_seconds > chosen.p95_seconds:
                    source = ForecastSource.MEASURED
                else:
                    proxy_name = chosen.call_class
                    source = ForecastSource.CONSERVATIVE_PROXY
            else:
                effective = max(
                    float(plan.provisional_p95_seconds),
                    own.p95_seconds if own is not None else 0.0,
                )
        rows.append(
            ForecastRow(
                call_class=plan.name,
                count=plan.count,
                provisional_p95_seconds=float(plan.provisional_p95_seconds),
                p50_seconds=None if own is None else own.p50_seconds,
                measured_p95_seconds=None if own is None else own.p95_seconds,
                forecast_p95_seconds=effective,
                forecast_seconds=plan.count * effective,
                source=source,
                proxy_call_class=(
                    proxy_name if source is ForecastSource.CONSERVATIVE_PROXY else None
                ),
            )
        )

    scheduled = REGISTERED_SCHEDULED_SECONDS if limits is None else limits.scheduled_gpu_seconds
    preferred = (
        REGISTERED_PREFERRED_SECONDS if limits is None else limits.preferred_forecast_gpu_seconds
    )
    hard = REGISTERED_HARD_SECONDS if limits is None else limits.hard_gpu_seconds
    return GPUForecast(
        rows=tuple(rows),
        total_seconds=math.fsum(row.forecast_seconds for row in rows),
        scheduled_limit_seconds=float(scheduled),
        preferred_limit_seconds=float(preferred),
        hard_limit_seconds=float(hard),
    )


class ReserveTier(StrEnum):
    LONG = "long"
    STANDARD = "standard"
    SHORT = "short"


RESERVE_CLASS_BY_TIER: Mapping[ReserveTier, str] = {
    ReserveTier.LONG: "reserve_long",
    ReserveTier.STANDARD: "reserve_standard",
    ReserveTier.SHORT: "reserve_short",
}


@dataclass(frozen=True)
class ReserveSlot:
    tier: ReserveTier
    index: int
    reservation_id: str
    call_class: str
    watchdog_seconds: int


@dataclass(frozen=True)
class ReserveState:
    consumed: tuple[ReserveSlot, ...] = ()

    def __post_init__(self) -> None:
        ids = [slot.reservation_id for slot in self.consumed]
        if len(ids) != len(set(ids)):
            raise ValueError("reserve reservation IDs must be globally unique")
        for tier in ReserveTier:
            indexes = [slot.index for slot in self.consumed if slot.tier is tier]
            if indexes != list(range(1, len(indexes) + 1)):
                raise ValueError(f"{tier.value} reserve indexes must be contiguous from one")


class ReserveExhausted(RuntimeError):
    """A fixed reserve tier is exhausted; another tier cannot be borrowed."""


class ReservePool:
    """Fixed 4/8/4 reserve consumption with idempotent resume tokens."""

    def __init__(self, inventory: GPUCallInventory, state: ReserveState | None = None) -> None:
        inventory.validate_registered_plan()
        self._inventory = inventory
        self._state = state or ReserveState()
        for slot in self._state.consumed:
            plan = inventory.call_class(RESERVE_CLASS_BY_TIER[slot.tier])
            if (
                slot.call_class != plan.name
                or slot.watchdog_seconds != plan.provisional_p95_seconds
            ):
                raise ValueError("persisted reserve slot differs from the registered inventory")
            if slot.index > plan.count:
                raise ValueError("persisted reserve state exceeds its registered tier")

    @property
    def state(self) -> ReserveState:
        return self._state

    def remaining(self, tier: ReserveTier) -> int:
        normalized = ReserveTier(tier)
        plan = self._inventory.call_class(RESERVE_CLASS_BY_TIER[normalized])
        used = sum(slot.tier is normalized for slot in self._state.consumed)
        return plan.count - used

    def consume(self, tier: ReserveTier, reservation_id: str) -> ReserveSlot:
        normalized = ReserveTier(tier)
        if not reservation_id:
            raise ValueError("reservation_id must be nonempty")
        for slot in self._state.consumed:
            if slot.reservation_id == reservation_id:
                if slot.tier is not normalized:
                    raise ValueError("a resumed reservation cannot change reserve tier")
                return slot
        plan = self._inventory.call_class(RESERVE_CLASS_BY_TIER[normalized])
        used = plan.count - self.remaining(normalized)
        if used >= plan.count:
            raise ReserveExhausted(
                f"{normalized.value} reserve exhausted; borrowing from other tiers is forbidden"
            )
        slot = ReserveSlot(
            tier=normalized,
            index=used + 1,
            reservation_id=reservation_id,
            call_class=plan.name,
            watchdog_seconds=plan.provisional_p95_seconds,
        )
        self._state = ReserveState((*self._state.consumed, slot))
        return slot


class CostRole(StrEnum):
    PREBUILD = "prebuild"
    QUERY_TIME = "query_time"
    INHERITED_C1_PREBUILD = "inherited_c1_prebuild"


@dataclass(frozen=True)
class CostEntry:
    """One attempt-level cost attribution record (including failed attempts)."""

    condition: str
    block_id: str
    call_class: str
    role: CostRole
    allocated_gpu_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    served_context_count: int = 0
    context_id: str | None = None
    query_ordinal: int | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", CostRole(self.role))
        for name in ("condition", "block_id", "call_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be nonempty")
        _nonnegative_finite("allocated_gpu_seconds", self.allocated_gpu_seconds)
        for name in ("prompt_tokens", "completion_tokens", "served_context_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.query_ordinal is not None and self.query_ordinal <= 0:
            raise ValueError("query_ordinal must be positive when present")
        if self.role is CostRole.PREBUILD and self.query_ordinal is not None:
            raise ValueError("a prebuild cannot have a query ordinal")

    @classmethod
    def from_gpu_event(
        cls,
        event: GpuEvent,
        *,
        condition: str,
        block_id: str,
        call_class: str,
        role: CostRole,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        served_context_count: int = 0,
        context_id: str | None = None,
        query_ordinal: int | None = None,
    ) -> Self:
        return cls(
            condition=condition,
            block_id=block_id,
            call_class=call_class,
            role=role,
            allocated_gpu_seconds=event.allocated_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            served_context_count=served_context_count,
            context_id=context_id,
            query_ordinal=query_ordinal,
            attempt_id=event.attempt_id,
        )


@dataclass(frozen=True)
class BlockCostSummary:
    condition: str
    block_id: str
    own_allocated_gpu_seconds: float
    inherited_c1_prebuild_seconds: float
    prebuild_seconds: float
    query_time_seconds: float
    served_context_count: int
    amortized_seconds_per_context: float | None
    first_query_workload_seconds: float
    complete_workload_seconds: float
    own_prompt_tokens: int
    own_completion_tokens: int
    inherited_prompt_tokens: int
    inherited_completion_tokens: int


def summarize_condition_block_costs(entries: Iterable[CostEntry]) -> tuple[BlockCostSummary, ...]:
    grouped: dict[tuple[str, str], list[CostEntry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.condition, entry.block_id)].append(entry)
    results: list[BlockCostSummary] = []
    for (condition, block_id), group in sorted(grouped.items()):
        inherited = [item for item in group if item.role is CostRole.INHERITED_C1_PREBUILD]
        prebuild = [item for item in group if item.role is CostRole.PREBUILD]
        query = [item for item in group if item.role is CostRole.QUERY_TIME]
        own = [*prebuild, *query]
        inherited_seconds = math.fsum(item.allocated_gpu_seconds for item in inherited)
        prebuild_seconds = math.fsum(item.allocated_gpu_seconds for item in prebuild)
        query_seconds = math.fsum(item.allocated_gpu_seconds for item in query)
        own_seconds = prebuild_seconds + query_seconds

        context_ids = {item.context_id for item in group if item.context_id is not None}
        if context_ids:
            served = len(context_ids)
        elif query:
            served = sum(item.served_context_count for item in query)
        else:
            served = max((item.served_context_count for item in group), default=0)
        complete = inherited_seconds + own_seconds
        if query:
            with_ordinals = [item for item in query if item.query_ordinal is not None]
            if with_ordinals:
                first_ordinal = min(item.query_ordinal for item in with_ordinals)
                first_query = math.fsum(
                    item.allocated_gpu_seconds
                    for item in query
                    if item.query_ordinal == first_ordinal
                )
            else:
                first_query = query[0].allocated_gpu_seconds
            first_workload = inherited_seconds + prebuild_seconds + first_query
        else:
            first_workload = complete
        results.append(
            BlockCostSummary(
                condition=condition,
                block_id=block_id,
                own_allocated_gpu_seconds=own_seconds,
                inherited_c1_prebuild_seconds=inherited_seconds,
                prebuild_seconds=prebuild_seconds,
                query_time_seconds=query_seconds,
                served_context_count=served,
                amortized_seconds_per_context=None if served == 0 else complete / served,
                first_query_workload_seconds=first_workload,
                complete_workload_seconds=complete,
                own_prompt_tokens=sum(item.prompt_tokens for item in own),
                own_completion_tokens=sum(item.completion_tokens for item in own),
                inherited_prompt_tokens=sum(item.prompt_tokens for item in inherited),
                inherited_completion_tokens=sum(item.completion_tokens for item in inherited),
            )
        )
    return tuple(results)


@dataclass
class MeteredGPUAllocation:
    """Mutable outcome handle yielded by :class:`AllocatedGPUMeter`."""

    event_id: str
    intended_kind: GpuEventKind
    maximum_seconds: float
    started_at: datetime
    _outcome_kind: GpuEventKind | None = field(default=None, init=False, repr=False)
    _succeeded: bool | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _details: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def mark_failure(self, **details: object) -> None:
        self._set_outcome(GpuEventKind.FAILURE, False, details)

    def mark_timeout(self, **details: object) -> None:
        self._set_outcome(GpuEventKind.TIMEOUT, False, details)

    def mark_succeeded(self, **details: object) -> None:
        self._set_outcome(self.intended_kind, True, details)

    def _set_outcome(
        self, kind: GpuEventKind, succeeded: bool | None, details: Mapping[str, object]
    ) -> None:
        if self._closed:
            raise RuntimeError("metered GPU allocation is already closed")
        self._outcome_kind = kind
        self._succeeded = succeeded
        self._details.update(details)


class _AllocationContext:
    def __init__(
        self,
        meter: AllocatedGPUMeter,
        *,
        event_id: str,
        event_kind: GpuEventKind,
        maximum_seconds: float,
        remaining_required_seconds: float,
        contingency_unlocked: bool,
        essential_recovery: bool,
        job_id: str | None,
        attempt_id: str | None,
        details: Mapping[str, object] | None,
    ) -> None:
        self.meter = meter
        self.event_id = event_id
        self.event_kind = event_kind
        self.maximum_seconds = maximum_seconds
        self.remaining_required_seconds = remaining_required_seconds
        self.contingency_unlocked = contingency_unlocked
        self.essential_recovery = essential_recovery
        self.job_id = job_id
        self.attempt_id = attempt_id
        self.details = dict(details or {})
        self.handle: MeteredGPUAllocation | None = None
        self.started_monotonic: float | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_failure: BaseException | None = None

    def _heartbeat(self) -> None:
        assert self.started_monotonic is not None
        while not self._heartbeat_stop.wait(self.meter.heartbeat_interval_seconds):
            try:
                elapsed = max(0.0, self.meter._monotonic_clock() - self.started_monotonic)
                self.meter.ledger.record_gpu_allocation_observation(
                    allocation_id=self.event_id,
                    state=GpuAllocationJournalState.HEARTBEAT,
                    intended_event_kind=self.event_kind,
                    elapsed_seconds=elapsed,
                    maximum_seconds=self.maximum_seconds,
                    observed_at=self.meter._wall_clock(),
                    job_id=self.job_id,
                    attempt_id=self.attempt_id,
                )
            except BaseException as exc:
                self._heartbeat_failure = exc
                self._heartbeat_stop.set()
                return

    def __enter__(self) -> MeteredGPUAllocation:
        self.meter.require_capacity(
            self.maximum_seconds,
            remaining_required_seconds=self.remaining_required_seconds,
            contingency_unlocked=self.contingency_unlocked,
            essential_recovery=self.essential_recovery,
        )
        started_at = self.meter._wall_clock()
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("wall clock must return a timezone-aware datetime")
        self.started_monotonic = self.meter._monotonic_clock()
        self.meter.ledger.record_gpu_allocation_observation(
            allocation_id=self.event_id,
            state=GpuAllocationJournalState.OPENED,
            intended_event_kind=self.event_kind,
            elapsed_seconds=0,
            maximum_seconds=self.maximum_seconds,
            observed_at=started_at,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            details=self.details,
        )
        self.handle = MeteredGPUAllocation(
            event_id=self.event_id,
            intended_kind=self.event_kind,
            maximum_seconds=self.maximum_seconds,
            started_at=started_at,
        )
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat,
            name=f"gpu-allocation-heartbeat-{self.event_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()
        return self.handle

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self.handle is None or self.started_monotonic is None:
            return False
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self.meter.heartbeat_interval_seconds + 1.0)
            if self._heartbeat_thread.is_alive():
                self._heartbeat_failure = RuntimeError("GPU allocation heartbeat did not stop")
        ended_monotonic = self.meter._monotonic_clock()
        elapsed = ended_monotonic - self.started_monotonic
        if not math.isfinite(elapsed) or elapsed < 0:
            raise GpuAccountingRegression("monotonic clock moved backwards during GPU allocation")
        ended_at = self.meter._wall_clock()
        outcome = self.handle._outcome_kind or self.event_kind
        succeeded = self.handle._succeeded
        effective_exception = exc if exc is not None else self._heartbeat_failure
        if effective_exception is not None:
            outcome = (
                GpuEventKind.TIMEOUT
                if isinstance(effective_exception, TimeoutError)
                else GpuEventKind.FAILURE
            )
            succeeded = False
            self.details["exception_type"] = type(effective_exception).__name__
        elif succeeded is None:
            succeeded = outcome not in {GpuEventKind.FAILURE, GpuEventKind.TIMEOUT}
        self.details.update(self.handle._details)
        self.details.setdefault("intended_event_kind", self.event_kind.value)
        self.details.setdefault("admitted_maximum_seconds", self.maximum_seconds)
        self.meter._record_completed_interval(
            event_id=self.event_id,
            event_kind=outcome,
            allocated_seconds=elapsed,
            started_at=self.handle.started_at,
            ended_at=ended_at,
            succeeded=succeeded,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            details=self.details,
        )
        self.meter.ledger.record_gpu_allocation_observation(
            allocation_id=self.event_id,
            state=GpuAllocationJournalState.CLOSED,
            intended_event_kind=self.event_kind,
            elapsed_seconds=elapsed,
            maximum_seconds=self.maximum_seconds,
            observed_at=ended_at,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            details={"outcome_event_kind": outcome.value},
        )
        self.handle._closed = True
        if exc is None and self._heartbeat_failure is not None:
            raise self._heartbeat_failure
        if elapsed > self.maximum_seconds and exc is None:
            raise GpuWatchdogExceeded(
                f"GPU interval {self.event_id!r} used {elapsed:.6f}s, above its admitted "
                f"{self.maximum_seconds:.6f}s watchdog"
            )
        return False


class AllocatedGPUMeter:
    """Resume-aware monotonic enforcement around the append-only GPU ledger."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        scheduled_limit_seconds: float = REGISTERED_SCHEDULED_SECONDS,
        hard_limit_seconds: float = REGISTERED_HARD_SECONDS,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self.ledger = ledger
        self.scheduled_limit_seconds = _nonnegative_finite(
            "scheduled_limit_seconds", scheduled_limit_seconds
        )
        self.hard_limit_seconds = _nonnegative_finite("hard_limit_seconds", hard_limit_seconds)
        if self.scheduled_limit_seconds <= 0 or self.hard_limit_seconds <= 0:
            raise ValueError("GPU limits must be positive")
        if self.scheduled_limit_seconds >= self.hard_limit_seconds:
            raise ValueError("scheduled GPU limit must be below the hard limit")
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self.heartbeat_interval_seconds = _nonnegative_finite(
            "heartbeat_interval_seconds", heartbeat_interval_seconds
        )
        if self.heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self._accounting_lock = threading.RLock()
        self.ledger.recover_unclosed_gpu_allocations(recovered_at=self._wall_clock())
        self._last_observed_seconds = ledger.gpu_summary().total_allocated_seconds

    @classmethod
    def from_limits(
        cls,
        ledger: Ledger,
        limits: ResourceLimits,
        **kwargs: object,
    ) -> Self:
        return cls(
            ledger,
            scheduled_limit_seconds=limits.scheduled_gpu_seconds,
            hard_limit_seconds=limits.hard_gpu_seconds,
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def actual_allocated_gpu_seconds(self) -> float:
        with self._accounting_lock:
            observed = self.ledger.gpu_summary().total_allocated_seconds
            if observed + 1e-9 < self._last_observed_seconds:
                raise GpuAccountingRegression(
                    "actual_allocated_gpu_seconds decreased across a resume boundary"
                )
            self._last_observed_seconds = max(self._last_observed_seconds, observed)
            return observed

    def recover_unclosed_allocations(
        self,
        *,
        recovered_at: datetime,
    ) -> tuple[GpuEvent, ...]:
        """Terminalize controller-lost allocation intervals at one stop time.

        A persistent guardian is intentionally constructed before the scientific
        controller starts.  Its meter therefore cannot rely on ``__init__``'s
        startup recovery to see journals opened later by that controller.  The
        service owner calls this method only after physical process absence has
        been established and before reconciling the encompassing service
        interval.  Recovering the classified intervals first lets service
        accounting store only their non-overlapping complement.
        """

        if recovered_at.tzinfo is None or recovered_at.utcoffset() is None:
            raise ValueError("GPU allocation recovery time must be timezone-aware")
        with self._accounting_lock:
            recovered = self.ledger.recover_unclosed_gpu_allocations(recovered_at=recovered_at)
            current = self.ledger.gpu_summary().total_allocated_seconds
            if current + 1e-9 < self._last_observed_seconds:
                raise GpuAccountingRegression(
                    "GPU ledger total decreased after allocation-journal recovery"
                )
            self._last_observed_seconds = max(self._last_observed_seconds, current)
        if current >= self.hard_limit_seconds:
            raise GpuBudgetExceeded(
                "recovered GPU allocation reached/crossed the strict hard-stop boundary"
            )
        return recovered

    def require_capacity(
        self,
        next_maximum_seconds: float,
        *,
        remaining_required_seconds: float = 0,
        contingency_unlocked: bool = False,
        essential_recovery: bool = False,
    ) -> None:
        next_seconds = _nonnegative_finite("next_maximum_seconds", next_maximum_seconds)
        remaining = _nonnegative_finite("remaining_required_seconds", remaining_required_seconds)
        if next_seconds <= 0:
            raise ValueError("next_maximum_seconds must be positive")
        used = self.actual_allocated_gpu_seconds
        # The hard boundary is strict: reaching exactly ten hours is forbidden.
        if used + next_seconds >= self.hard_limit_seconds:
            raise GpuBudgetExceeded(
                f"GPU allocation would reach/cross hard limit: used={used:.6f}s, "
                f"next={next_seconds:.6f}s, hard={self.hard_limit_seconds:.6f}s"
            )
        projected_required = used + next_seconds + remaining
        if projected_required <= self.scheduled_limit_seconds:
            return
        if not (contingency_unlocked and essential_recovery):
            raise ForecastAdmissionError(
                f"used plus next and remaining required work ({projected_required:.6f}s) "
                f"exceeds the scheduled {self.scheduled_limit_seconds:.6f}s envelope"
            )

    def reconcile_service_session(
        self,
        *,
        service_session_id: str,
        session_id: str,
        service_seconds: float,
        classified_event_seconds: float,
        started_at: datetime,
        ended_at: datetime,
        details: Mapping[str, object] | None = None,
    ) -> GpuServiceSession:
        """Durably add only service time not already represented by GPU events."""

        service = _nonnegative_finite("service_seconds", service_seconds)
        classified = _nonnegative_finite("classified_event_seconds", classified_event_seconds)
        journal = self.ledger.latest_gpu_service_journal(service_session_id)
        if journal is None:
            # Backward-compatible path for imported legacy accounting and
            # bounded unit fixtures created before schema version 5.
            record = self.ledger.record_gpu_service_session(
                service_session_id=service_session_id,
                session_id=session_id,
                service_seconds=service,
                classified_event_seconds=classified,
                started_at=started_at,
                ended_at=ended_at,
                details=details,
            )
        else:
            record = self.ledger.close_gpu_service_journal(
                service_session_id=service_session_id,
                session_id=session_id,
                service_seconds=service,
                classified_event_seconds=classified,
                started_at=started_at,
                ended_at=ended_at,
                details=details,
            )
        with self._accounting_lock:
            current = self.ledger.gpu_summary().total_allocated_seconds
            if current + 1e-9 < self._last_observed_seconds:
                raise GpuAccountingRegression(
                    "GPU ledger total decreased after service-session reconciliation"
                )
            self._last_observed_seconds = max(self._last_observed_seconds, current)
        if current >= self.hard_limit_seconds:
            raise GpuBudgetExceeded(
                "actual allocated GPU time reached/crossed the strict hard-stop boundary"
            )
        return record

    def open_service_journal(
        self,
        *,
        service_session_id: str,
        session_id: str,
        configuration_hash: str,
        started_at: datetime,
        ledger_allocated_seconds_before_session: float,
        details: Mapping[str, object] | None = None,
    ) -> GpuServiceJournalRecord:
        """Open durable service accounting before the model process is spawned."""

        baseline = _nonnegative_finite(
            "ledger_allocated_seconds_before_session",
            ledger_allocated_seconds_before_session,
        )
        observed = self.actual_allocated_gpu_seconds
        if abs(observed - baseline) > 1e-6:
            raise GpuAccountingRegression(
                "GPU service baseline differs from the durable allocation ledger"
            )
        return self.ledger.record_gpu_service_observation(
            service_session_id=service_session_id,
            state=GpuServiceJournalState.OPENED,
            session_id=session_id,
            configuration_hash=configuration_hash,
            service_started_at=started_at,
            elapsed_seconds=0,
            ledger_allocated_seconds_before_session=baseline,
            hard_limit_seconds=self.hard_limit_seconds,
            observed_at=started_at,
            details=details,
        )

    def observe_service_journal(
        self,
        *,
        service_session_id: str,
        elapsed_seconds: float,
        observed_at: datetime,
        process_stopped: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> GpuServiceJournalRecord:
        """Heartbeat or mark a verified physical stop for an open service."""

        elapsed = _nonnegative_finite("elapsed_seconds", elapsed_seconds)
        latest = self.ledger.latest_gpu_service_journal(service_session_id)
        if latest is None:
            raise KeyError(f"unknown GPU service journal {service_session_id}")
        state = (
            GpuServiceJournalState.PROCESS_STOPPED
            if process_stopped
            else GpuServiceJournalState.HEARTBEAT
        )
        record = self.ledger.record_gpu_service_observation(
            service_session_id=service_session_id,
            state=state,
            session_id=latest.session_id,
            configuration_hash=latest.configuration_hash,
            service_started_at=latest.service_started_at,
            elapsed_seconds=elapsed,
            ledger_allocated_seconds_before_session=(
                latest.ledger_allocated_microseconds_before_session / 1_000_000
            ),
            hard_limit_seconds=latest.hard_limit_microseconds / 1_000_000,
            observed_at=observed_at,
            details=details,
        )
        estimated_total = latest.ledger_allocated_microseconds_before_session / 1_000_000 + elapsed
        if estimated_total >= self.hard_limit_seconds:
            raise GpuBudgetExceeded(
                "live GPU service reached/crossed the strict hard-stop boundary"
            )
        return record

    def recover_service_journal(
        self,
        *,
        service_session_id: str,
        recovered_at: datetime,
        details: Mapping[str, object] | None = None,
    ) -> GpuServiceSession:
        """Account a stale journal after the caller verifies service absence."""

        record = self.ledger.recover_gpu_service_journal(
            service_session_id=service_session_id,
            recovered_at=recovered_at,
            details=details,
        )
        with self._accounting_lock:
            current = self.ledger.gpu_summary().total_allocated_seconds
            if current + 1e-9 < self._last_observed_seconds:
                raise GpuAccountingRegression(
                    "GPU ledger total decreased after service-journal recovery"
                )
            self._last_observed_seconds = max(self._last_observed_seconds, current)
        if current >= self.hard_limit_seconds:
            raise GpuBudgetExceeded(
                "recovered GPU service reached/crossed the strict hard-stop boundary"
            )
        return record

    def allocation(
        self,
        *,
        event_id: str,
        event_kind: GpuEventKind,
        maximum_seconds: float,
        remaining_required_seconds: float = 0,
        contingency_unlocked: bool = False,
        essential_recovery: bool = False,
        job_id: str | None = None,
        attempt_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> _AllocationContext:
        if not event_id:
            raise ValueError("event_id must be nonempty")
        normalized_kind = GpuEventKind(event_kind)
        if normalized_kind is GpuEventKind.SERVICE_OVERHEAD:
            raise ValueError("service_overhead is session-derived; use reconcile_service_session")
        return _AllocationContext(
            self,
            event_id=event_id,
            event_kind=normalized_kind,
            maximum_seconds=_nonnegative_finite("maximum_seconds", maximum_seconds),
            remaining_required_seconds=_nonnegative_finite(
                "remaining_required_seconds", remaining_required_seconds
            ),
            contingency_unlocked=contingency_unlocked,
            essential_recovery=essential_recovery,
            job_id=job_id,
            attempt_id=attempt_id,
            details=details,
        )

    def model_load(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(
            event_kind=GpuEventKind.MODEL_LOAD,
            **kwargs,  # type: ignore[arg-type]
        )

    def session_start(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(
            event_kind=GpuEventKind.GPU_SESSION_START,
            **kwargs,  # type: ignore[arg-type]
        )

    def warmup(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(event_kind=GpuEventKind.WARM_UP, **kwargs)  # type: ignore[arg-type]

    def schema_probe(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(
            event_kind=GpuEventKind.SCHEMA_PROBE,
            **kwargs,  # type: ignore[arg-type]
        )

    def inference(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(
            event_kind=GpuEventKind.INFERENCE,
            **kwargs,  # type: ignore[arg-type]
        )

    def repair(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(event_kind=GpuEventKind.REPAIR, **kwargs)  # type: ignore[arg-type]

    def restart(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(event_kind=GpuEventKind.RESTART, **kwargs)  # type: ignore[arg-type]

    def failure(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(event_kind=GpuEventKind.FAILURE, **kwargs)  # type: ignore[arg-type]

    def timeout(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(event_kind=GpuEventKind.TIMEOUT, **kwargs)  # type: ignore[arg-type]

    def fallback_test(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(
            event_kind=GpuEventKind.FALLBACK_TEST,
            **kwargs,  # type: ignore[arg-type]
        )

    def rerun(self, **kwargs: object) -> _AllocationContext:
        return self.allocation(event_kind=GpuEventKind.RERUN, **kwargs)  # type: ignore[arg-type]

    def _record_completed_interval(
        self,
        *,
        event_id: str,
        event_kind: GpuEventKind,
        allocated_seconds: float,
        started_at: datetime,
        ended_at: datetime,
        succeeded: bool | None,
        job_id: str | None,
        attempt_id: str | None,
        details: Mapping[str, object],
    ) -> GpuEvent:
        if event_kind is GpuEventKind.SERVICE_OVERHEAD:
            raise ValueError("service_overhead is session-derived; use reconcile_service_session")
        event = self.ledger.record_gpu_event(
            event_id=event_id,
            event_kind=event_kind,
            allocated_seconds=allocated_seconds,
            started_at=started_at,
            ended_at=ended_at,
            succeeded=succeeded,
            job_id=job_id,
            attempt_id=attempt_id,
            details=details,
        )
        with self._accounting_lock:
            current = self.ledger.gpu_summary().total_allocated_seconds
            if current + 1e-9 < self._last_observed_seconds:
                raise GpuAccountingRegression(
                    "GPU ledger total decreased after appending an interval"
                )
            self._last_observed_seconds = max(self._last_observed_seconds, current)
        if current >= self.hard_limit_seconds:
            raise GpuBudgetExceeded(
                "actual allocated GPU time reached/crossed the strict hard-stop boundary"
            )
        return event


# Compatibility alias matching the shorter wording used in runner code and reports.
GPUMeter = AllocatedGPUMeter
