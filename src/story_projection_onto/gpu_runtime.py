"""Controlled local vLLM runtime for the Phase-1 acceptance pilot.

The module deliberately keeps process control, HTTP transport, resource sampling,
and public manifests separate from the semantic LLM contracts in :mod:`llm`.
Nothing here downloads a model: the launcher accepts only the verified, pinned
snapshot already present below the one shared project cache.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Protocol, Self, cast

from story_projection_onto.contracts import ConditionName, canonical_json, canonical_sha256
from story_projection_onto.experiment import (
    AllocatedGPUMeter,
    ForecastAdmissionError,
    ResourceLimits,
)
from story_projection_onto.llm import DecodingManifest, DecodingPass, PackingReport
from story_projection_onto.store import (
    GpuBudgetExceeded,
    GpuServiceJournalState,
    GpuServiceSession,
    Ledger,
    StoragePreflight,
)

SCHEMA_VERSION = "1.0.0"
PINNED_MODEL_REPOSITORY = "Qwen/Qwen3-14B-AWQ"
PINNED_MODEL_REVISION = "1a6fe1ecf891437a270cce11ad54d796c4f56ce0"
PINNED_RUNTIME_VERSION = "0.10.2"
SERVED_MODEL_NAME = "qwen3-14b-awq-pinned"
FALLBACK_MODEL_REPOSITORY = "Qwen/Qwen3-8B-AWQ"
FALLBACK_MODEL_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
FALLBACK_SERVED_MODEL_NAME = "qwen3-8b-awq-fallback"
MAXIMUM_MODEL_LENGTH = 12_288
MAXIMUM_CPU_WORKERS = 8
GPU_MEMORY_UTILIZATION = 0.88
DEFAULT_SERVICE_START_WATCHDOG_SECONDS = 180
DEFAULT_SHUTDOWN_SECONDS = 30
MEBIBYTE = 1024 * 1024
PROC_ROOT = Path("/proc")
SERVICE_LOCK_FILENAME = ".story-projection-onto-vllm.lock"
GUIDED_DECODING_BACKEND = "xgrammar"
RUNTIME_ENVIRONMENT_PASSTHROUGH = (
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
)
MODEL_CANDIDATES = (
    ("primary", PINNED_MODEL_REPOSITORY, PINNED_MODEL_REVISION, SERVED_MODEL_NAME),
    (
        "fallback",
        FALLBACK_MODEL_REPOSITORY,
        FALLBACK_MODEL_REVISION,
        FALLBACK_SERVED_MODEL_NAME,
    ),
)


class RuntimeConfigurationError(ValueError):
    """The launcher or request would depart from the frozen Phase-1 protocol."""


class RuntimeTransportError(RuntimeError):
    """The local vLLM HTTP service returned an invalid response."""


class RuntimeWatchdogTimeout(TimeoutError):
    """A local service operation exceeded its admitted watchdog."""


class RuntimeResourceLimitExceeded(RuntimeError):
    """A sampled hard resource limit was exceeded."""

    def __init__(self, snapshot: ResourceSnapshot) -> None:
        self.snapshot = snapshot
        super().__init__("; ".join(snapshot.violations))


def _model_candidate_name(repository: str, revision: str, served_model_name: str) -> str:
    for (
        candidate_name,
        candidate_repository,
        candidate_revision,
        candidate_alias,
    ) in MODEL_CANDIDATES:
        if (repository, revision, served_model_name) == (
            candidate_repository,
            candidate_revision,
            candidate_alias,
        ):
            return candidate_name
    raise RuntimeConfigurationError(
        "model repository, revision, and served alias must match one allowlisted candidate"
    )


def _model_candidate_identity(candidate_name: str) -> tuple[str, str, str]:
    _require_plain_identifier("model_candidate", candidate_name)
    for name, repository, revision, served_model_name in MODEL_CANDIDATES:
        if name == candidate_name:
            return repository, revision, served_model_name
    raise RuntimeConfigurationError("model_candidate must be 'primary' or 'fallback'")


def _model_candidate_for_tokenizer(repository: str, revision: str) -> str:
    for candidate_name, candidate_repository, candidate_revision, _ in MODEL_CANDIDATES:
        if (repository, revision) == (candidate_repository, candidate_revision):
            return candidate_name
    raise RuntimeConfigurationError(
        "tokenizer repository and revision must match one allowlisted model candidate"
    )


def _is_allowlisted_model_alias(model_name: str) -> bool:
    return any(model_name == candidate_alias for _, _, _, candidate_alias in MODEL_CANDIDATES)


def _model_revision_for_alias(model_name: str) -> str:
    for _, _, revision, candidate_alias in MODEL_CANDIDATES:
        if model_name == candidate_alias:
            return revision
    raise RuntimeConfigurationError("model alias is not an allowlisted candidate")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _path_identity(path: Path) -> str:
    """Return a stable public identifier without publishing an absolute path."""

    return _sha256_bytes(str(path.resolve()).encode("utf-8"))


def _parse_aware_datetime(name: str, value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigurationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeConfigurationError(f"{name} must include an offset")
    return parsed


def _require_plain_identifier(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(char in value for char in "\r\n\0")
    ):
        raise RuntimeConfigurationError(f"{name} must be a nonempty plain identifier")
    return value


def atomic_write_public_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a manifest after rejecting obvious private path fields."""

    for key in value:
        lowered = key.casefold()
        if lowered.endswith("_path") or lowered.endswith("_directory"):
            raise RuntimeConfigurationError(
                f"public manifest may not contain filesystem field {key!r}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            # ``model_gate`` deliberately returns immutable ``MappingProxyType``
            # certificates.  The stdlib JSON encoder only recognizes concrete
            # dicts as mappings, so materialize the public top-level object while
            # preserving its already-canonical contents.
            json.dump(dict(value), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class VLLMLaunchConfiguration:
    """Pinned, local-only vLLM 0.10.2 service configuration."""

    snapshot_path: Path
    shared_cache: Path
    repository: str = PINNED_MODEL_REPOSITORY
    revision: str = PINNED_MODEL_REVISION
    runtime_version: str = PINNED_RUNTIME_VERSION
    served_model_name: str = SERVED_MODEL_NAME
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    maximum_model_length: int = MAXIMUM_MODEL_LENGTH
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    cpu_offload_gb: int = 0
    maximum_sequences: int = 1
    cpu_workers: int = MAXIMUM_CPU_WORKERS
    dtype: str = "half"
    prefix_caching: bool = False
    speculative_decoding: bool = False
    enforce_eager: bool = True
    verified_snapshot_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        snapshot = self.snapshot_path.resolve(strict=True)
        cache = self.shared_cache.resolve(strict=True)
        try:
            snapshot.relative_to(cache)
        except ValueError as exc:
            raise RuntimeConfigurationError(
                "the model snapshot must be inside the one shared cache"
            ) from exc
        expected = {
            "runtime_version": PINNED_RUNTIME_VERSION,
            "host": "127.0.0.1",
            "tensor_parallel_size": 1,
            "maximum_model_length": MAXIMUM_MODEL_LENGTH,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "cpu_offload_gb": 0,
            "maximum_sequences": 1,
            "cpu_workers": MAXIMUM_CPU_WORKERS,
            "dtype": "half",
            "prefix_caching": False,
            "speculative_decoding": False,
            "enforce_eager": True,
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise RuntimeConfigurationError(f"{name} must remain {required!r}")
        _model_candidate_name(self.repository, self.revision, self.served_model_name)
        expected_repository_directory = "models--" + self.repository.replace("/", "--")
        if (
            snapshot.name != self.revision
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != expected_repository_directory
        ):
            raise RuntimeConfigurationError(
                "snapshot path must identify the selected repository and revision"
            )
        if self.verified_snapshot_manifest_sha256 is not None and (
            len(self.verified_snapshot_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.verified_snapshot_manifest_sha256
            )
        ):
            raise RuntimeConfigurationError(
                "verified snapshot manifest hash must be lowercase SHA-256"
            )
        if isinstance(self.port, bool) or not 1 <= self.port <= 65_535:
            raise RuntimeConfigurationError("port must be an integer from 1 through 65535")
        object.__setattr__(self, "snapshot_path", snapshot)
        object.__setattr__(self, "shared_cache", cache)

    @classmethod
    def from_model_configuration(
        cls,
        *,
        snapshot_path: Path,
        shared_cache: Path,
        model_configuration_path: Path,
        model_candidate: str = "primary",
        verified_snapshot_manifest_sha256: str | None = None,
        port: int = 8000,
    ) -> Self:
        raw = json.loads(model_configuration_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise RuntimeConfigurationError("model configuration root must be an object")
        if (raw.get("repository"), raw.get("revision")) != (
            PINNED_MODEL_REPOSITORY,
            PINNED_MODEL_REVISION,
        ):
            raise RuntimeConfigurationError(
                "model configuration must retain the frozen primary candidate identity"
            )
        repository, revision, served_model_name = _model_candidate_identity(model_candidate)
        return cls(
            snapshot_path=snapshot_path,
            shared_cache=shared_cache,
            repository=repository,
            revision=revision,
            runtime_version=cast(str, raw.get("runtime_version")),
            served_model_name=served_model_name,
            tensor_parallel_size=cast(int, raw.get("tensor_parallel_size")),
            maximum_model_length=cast(int, raw.get("max_model_len")),
            gpu_memory_utilization=cast(float, raw.get("gpu_memory_utilization_pilot")),
            cpu_offload_gb=cast(int, raw.get("cpu_offload_gb")),
            maximum_sequences=cast(int, raw.get("request_concurrency")),
            prefix_caching=cast(bool, raw.get("prefix_decoding")),
            speculative_decoding=cast(bool, raw.get("speculative_decoding")),
            enforce_eager=cast(bool, raw.get("enforce_eager")),
            verified_snapshot_manifest_sha256=verified_snapshot_manifest_sha256,
            port=port,
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def model_candidate(self) -> str:
        """Return the immutable allowlisted candidate kind selected by this config."""

        return _model_candidate_name(
            self.repository,
            self.revision,
            self.served_model_name,
        )

    @property
    def configuration_hash(self) -> str:
        return canonical_sha256(self.public_manifest())

    def public_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "repository": self.repository,
            "revision": self.revision,
            "runtime": "vllm",
            "runtime_version": self.runtime_version,
            "model_candidate": self.model_candidate,
            "served_model_name": self.served_model_name,
            "snapshot_location_sha256": _path_identity(self.snapshot_path),
            "shared_cache_location_sha256": _path_identity(self.shared_cache),
            "local_snapshot_only": True,
            "host_scope": "loopback",
            "tensor_parallel_size": self.tensor_parallel_size,
            "maximum_model_length": self.maximum_model_length,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "cpu_offload_gb": self.cpu_offload_gb,
            "maximum_sequences": self.maximum_sequences,
            "cpu_workers": self.cpu_workers,
            "dtype": "float16",
            "prefix_caching": self.prefix_caching,
            "speculative_decoding": self.speculative_decoding,
            "enforce_eager": self.enforce_eager,
            "guided_decoding_backend": GUIDED_DECODING_BACKEND,
            "guided_decoding_fallback": False,
            "runtime_environment_policy": "explicit_os_cuda_allowlist",
            "runtime_environment_passthrough": list(RUNTIME_ENVIRONMENT_PASSTHROUGH),
            "verified_snapshot_manifest_sha256": self.verified_snapshot_manifest_sha256,
        }
        return {**payload, "manifest_sha256": canonical_sha256(payload)}

    def command(self, python_executable: str = sys.executable) -> tuple[str, ...]:
        """Build the explicit vLLM 0.10.2 OpenAI-server command."""

        return (
            python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.snapshot_path),
            "--tokenizer",
            str(self.snapshot_path),
            "--served-model-name",
            self.served_model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "12288",
            "--gpu-memory-utilization",
            "0.88",
            "--cpu-offload-gb",
            "0",
            "--max-num-seqs",
            "1",
            "--dtype",
            "half",
            "--generation-config",
            "vllm",
            "--guided-decoding-backend",
            GUIDED_DECODING_BACKEND,
            "--guided-decoding-disable-fallback",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--no-enable-log-requests",
            "--uvicorn-log-level",
            "warning",
        )

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        inherited = os.environ if base is None else base
        environment = {
            name: inherited[name] for name in RUNTIME_ENVIRONMENT_PASSTHROUGH if name in inherited
        }
        cache = str(self.shared_cache)
        environment.update(
            {
                "HF_HOME": cache,
                "HF_HUB_CACHE": cache,
                "HUGGINGFACE_HUB_CACHE": cache,
                "TRANSFORMERS_CACHE": cache,
                "VLLM_CACHE_ROOT": cache,
                "XDG_CACHE_HOME": cache,
                "TORCH_HOME": cache,
                "TORCHINDUCTOR_CACHE_DIR": str(self.shared_cache / "torchinductor"),
                "TRITON_CACHE_DIR": str(self.shared_cache / "triton"),
                "CUDA_CACHE_PATH": str(self.shared_cache / "cuda"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
                "NUMEXPR_NUM_THREADS": "8",
                "TMPDIR": cache,
            }
        )
        return environment


class TokenizerLike(Protocol):
    eos_token_id: int | Sequence[int] | None
    unk_token_id: int | None
    chat_template: str | None

    def apply_chat_template(self, conversation: object, **kwargs: object) -> object: ...

    def convert_tokens_to_ids(self, tokens: str) -> int | Sequence[int] | None: ...


@dataclass(frozen=True, slots=True)
class RuntimeStackManifest:
    """Observed immutable runtime versions, captured without touching CUDA memory."""

    python_version: str
    vllm_version: str
    transformers_version: str
    torch_version: str
    torch_cuda_version: str

    def public_manifest(self) -> dict[str, object]:
        payload = {"schema_version": SCHEMA_VERSION, **asdict(self)}
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


@dataclass(frozen=True, slots=True)
class GPUHardwareIdentity:
    """Exact live GPU identity linked to the tracked environment manifest."""

    name: str
    memory_mib: int
    driver_version: str
    pci_bus_id: str
    environment_manifest_sha256: str

    def public_manifest(self) -> dict[str, object]:
        payload = {"schema_version": SCHEMA_VERSION, "count": 1, **asdict(self)}
        return {**payload, "manifest_sha256": canonical_sha256(payload)}


def capture_gpu_hardware_identity(
    environment_manifest_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GPUHardwareIdentity:
    """Capture one GPU via ``nvidia-smi`` and require the tracked exact identity."""

    manifest_path = environment_manifest_path.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
        expected = manifest["gpu"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeConfigurationError("environment manifest has no valid GPU identity") from exc
    if not isinstance(expected, Mapping):
        raise RuntimeConfigurationError("environment manifest GPU identity must be an object")
    expected_count = expected.get("count")
    expected_name = expected.get("name")
    expected_memory = expected.get("memory_mib")
    expected_driver = expected.get("driver_version")
    expected_bus = expected.get("pci_bus_id")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != 1
        or not isinstance(expected_name, str)
        or not expected_name
        or isinstance(expected_memory, bool)
        or not isinstance(expected_memory, int)
        or expected_memory <= 0
        or not isinstance(expected_driver, str)
        or not expected_driver
        or not isinstance(expected_bus, str)
        or not expected_bus
    ):
        raise RuntimeConfigurationError(
            "environment manifest must freeze exactly one complete GPU identity"
        )
    try:
        completed = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeConfigurationError("cannot capture live GPU hardware identity") from exc
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise RuntimeConfigurationError("cannot capture live GPU hardware identity")
    rows = [
        tuple(field.strip() for field in row)
        for row in csv.reader(completed.stdout.splitlines())
        if any(field.strip() for field in row)
    ]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeConfigurationError("live environment must expose exactly one GPU")
    name, memory_text, driver_version, pci_bus_id = rows[0]
    try:
        memory_mib = int(memory_text)
    except ValueError as exc:
        raise RuntimeConfigurationError("live GPU memory must be an integer MiB value") from exc
    observed = {
        "name": name,
        "memory_mib": memory_mib,
        "driver_version": driver_version,
        "pci_bus_id": pci_bus_id,
    }
    frozen = {
        "name": expected_name,
        "memory_mib": expected_memory,
        "driver_version": expected_driver,
        "pci_bus_id": expected_bus,
    }
    mismatches = [key for key in frozen if observed[key] != frozen[key]]
    if mismatches:
        raise RuntimeConfigurationError(
            "live GPU differs from environment manifest: " + ", ".join(mismatches)
        )
    return GPUHardwareIdentity(
        name=name,
        memory_mib=memory_mib,
        driver_version=driver_version,
        pci_bus_id=pci_bus_id,
        environment_manifest_sha256=_sha256_bytes(manifest_bytes),
    )


def capture_runtime_stack(
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
    module_importer: Callable[[str], object] = importlib.import_module,
) -> RuntimeStackManifest:
    """Capture and enforce the pod's pinned vLLM/Transformers/Torch stack."""

    torch_module = module_importer("torch")
    torch_version = getattr(torch_module, "__version__", None)
    cuda_namespace = getattr(torch_module, "version", None)
    cuda_version = getattr(cuda_namespace, "cuda", None)
    if not isinstance(torch_version, str) or not isinstance(cuda_version, str):
        raise RuntimeConfigurationError("installed torch does not expose version/CUDA strings")
    observed = RuntimeStackManifest(
        python_version=".".join(str(value) for value in sys.version_info[:3]),
        vllm_version=version_reader("vllm"),
        transformers_version=version_reader("transformers"),
        torch_version=torch_version,
        torch_cuda_version=cuda_version,
    )
    expected = {
        "vllm_version": PINNED_RUNTIME_VERSION,
        "transformers_version": "4.55.2",
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
    }
    mismatches = [name for name, value in expected.items() if getattr(observed, name) != value]
    if mismatches:
        raise RuntimeConfigurationError(
            "installed runtime differs from frozen stack at: " + ", ".join(mismatches)
        )
    return observed


@dataclass(frozen=True, slots=True)
class TokenizerManifest:
    """Public-safe exact tokenizer and non-thinking chat-template capture."""

    schema_version: str
    repository: str
    revision: str
    tokenizer_class: str
    tokenizer_revision: str
    tokenizer_file_sha256: tuple[tuple[str, str], ...]
    eos_token_id: int
    end_of_turn_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    chat_template_sha256: str
    nonthinking_probe_sha256: str
    nonthinking_probe_token_count: int
    local_files_only: bool
    trust_remote_code: bool
    enable_thinking: bool

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(asdict(self))

    def public_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        return {**payload, "manifest_sha256": self.manifest_sha256}


def _one_token_id(tokenizer: TokenizerLike, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeConfigurationError(f"tokenizer does not define exact token {token!r}")
    if tokenizer.unk_token_id is not None and value == tokenizer.unk_token_id:
        raise RuntimeConfigurationError(f"tokenizer mapped {token!r} to its unknown token")
    return value


def capture_tokenizer_manifest(
    snapshot_path: Path,
    *,
    repository: str = PINNED_MODEL_REPOSITORY,
    revision: str = PINNED_MODEL_REVISION,
    tokenizer_loader: Callable[..., TokenizerLike] | None = None,
    end_of_turn_tokens: Sequence[str] = ("<|im_end|>",),
) -> TokenizerManifest:
    """Load only the local pinned tokenizer and hash its exact non-thinking behavior."""

    _model_candidate_for_tokenizer(repository, revision)
    snapshot = snapshot_path.resolve(strict=True)
    if snapshot.name != revision:
        raise RuntimeConfigurationError("tokenizer snapshot directory must equal pinned revision")
    if tokenizer_loader is None:
        from transformers import AutoTokenizer

        tokenizer_loader = cast(Callable[..., TokenizerLike], AutoTokenizer.from_pretrained)
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required_tokenizer_files if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeConfigurationError(
            "pinned snapshot lacks tokenizer files: " + ", ".join(missing)
        )
    optional_tokenizer_files = ("special_tokens_map.json", "added_tokens.json")
    tokenizer_files = tuple(
        name
        for name in (*required_tokenizer_files, *optional_tokenizer_files)
        if (snapshot / name).is_file()
    )
    tokenizer = tokenizer_loader(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        revision=revision,
    )
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeConfigurationError("pinned tokenizer has no raw chat template")
    eos_value = tokenizer.eos_token_id
    if isinstance(eos_value, bool) or not isinstance(eos_value, int) or eos_value < 0:
        raise RuntimeConfigurationError("pinned tokenizer must expose one exact EOS token ID")
    if not end_of_turn_tokens:
        raise RuntimeConfigurationError("at least one end-of-turn token must be captured")
    end_ids = tuple(dict.fromkeys(_one_token_id(tokenizer, token) for token in end_of_turn_tokens))
    if not all(token in template for token in end_of_turn_tokens):
        raise RuntimeConfigurationError("declared end-of-turn token is absent from chat template")
    stop_ids = tuple(dict.fromkeys((eos_value, *end_ids)))
    probe = (
        {"role": "system", "content": "Return one JSON object."},
        {"role": "user", "content": "Acknowledge readiness."},
    )
    rendered = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer.apply_chat_template(
        probe,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeConfigurationError("non-thinking tokenizer probe did not render text")
    if not isinstance(encoded, Sequence) or isinstance(encoded, (str, bytes, bytearray)):
        raise RuntimeConfigurationError("non-thinking tokenizer probe did not return token IDs")
    tokenizer_class = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    return TokenizerManifest(
        schema_version=SCHEMA_VERSION,
        repository=repository,
        revision=revision,
        tokenizer_class=tokenizer_class,
        tokenizer_revision=revision,
        tokenizer_file_sha256=tuple(
            (name, _sha256_bytes((snapshot / name).read_bytes())) for name in tokenizer_files
        ),
        eos_token_id=eos_value,
        end_of_turn_token_ids=end_ids,
        stop_token_ids=stop_ids,
        chat_template_sha256=_sha256_bytes(template.encode("utf-8")),
        nonthinking_probe_sha256=_sha256_bytes(rendered.encode("utf-8")),
        nonthinking_probe_token_count=len(encoded),
        local_files_only=True,
        trust_remote_code=False,
        enable_thinking=False,
    )


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise RuntimeConfigurationError("unsupported chat role")
        if not self.content:
            raise RuntimeConfigurationError("chat message content must be nonempty")


@dataclass(frozen=True, slots=True)
class GuidedJSONRequest:
    """One fully packed request admitted to the loopback vLLM endpoint."""

    request_id: str
    model_name: str
    condition: ConditionName
    messages: tuple[ChatMessage, ...]
    output_schema: Mapping[str, object]
    decoding: DecodingManifest
    packing: PackingReport
    rendered_input_token_count: int

    def __post_init__(self) -> None:
        _require_plain_identifier("request_id", self.request_id)
        _require_plain_identifier("model_name", self.model_name)
        if not _is_allowlisted_model_alias(self.model_name):
            raise RuntimeConfigurationError(
                "request must target one allowlisted served-model alias"
            )
        if not self.messages:
            raise RuntimeConfigurationError("request must include messages")
        if self.condition is not self.packing.condition:
            raise RuntimeConfigurationError("request and packing conditions differ")
        if self.packing.tokenizer_revision != self.decoding.tokenizer_revision:
            raise RuntimeConfigurationError("packing and decoding tokenizer revisions differ")
        if self.decoding.tokenizer_revision != _model_revision_for_alias(self.model_name):
            raise RuntimeConfigurationError(
                "request tokenizer revision differs from its served-model candidate"
            )
        if self.decoding.output_schema_hash != canonical_sha256(self.output_schema):
            raise RuntimeConfigurationError("guided_json schema differs from decoding manifest")
        if self.packing.maximum_model_tokens != self.decoding.maximum_model_tokens:
            raise RuntimeConfigurationError("packing and decoding model-token caps differ")
        if self.packing.maximum_input_tokens != self.decoding.maximum_input_tokens:
            raise RuntimeConfigurationError("packing and decoding input caps differ")
        if self.packing.reserved_output_tokens != self.decoding.maximum_output_tokens:
            raise RuntimeConfigurationError("packing and decoding output caps differ")
        if self.rendered_input_token_count < 0:
            raise RuntimeConfigurationError("rendered_input_token_count must be nonnegative")
        if self.rendered_input_token_count > self.decoding.maximum_input_tokens:
            raise RuntimeConfigurationError("complete rendered prompt exceeds its input cap")
        if self.packing.input_token_count != self.rendered_input_token_count:
            raise RuntimeConfigurationError(
                "packing report token count differs from the exact rendered prompt"
            )
        if self.packing.truncation_applied or self.packing.omitted_section_names:
            raise RuntimeConfigurationError("runtime refuses truncated or omitted packing")

    @property
    def prompt_hash(self) -> str:
        return canonical_sha256([asdict(message) for message in self.messages])

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.wire_payload())

    def wire_payload(self) -> dict[str, object]:
        decoding = self.decoding
        return {
            "model": self.model_name,
            "messages": [asdict(message) for message in self.messages],
            "guided_json": self.output_schema,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "top_k": decoding.top_k,
            "min_p": decoding.min_p,
            "presence_penalty": decoding.presence_penalty,
            "frequency_penalty": decoding.frequency_penalty,
            "repetition_penalty": decoding.repetition_penalty,
            "n": decoding.n,
            "best_of": decoding.best_of,
            "use_beam_search": decoding.beam_search,
            "ignore_eos": decoding.ignore_eos,
            "max_tokens": decoding.maximum_output_tokens,
            "seed": decoding.seed,
            "stop_token_ids": list(decoding.stop_token_ids),
            "stream": False,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    request_hash: str
    response_sha256: str
    parsed_object: Mapping[str, object]
    raw_response: bytes = field(repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    service_request_id: str | None = None

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "response_sha256": self.response_sha256,
            "parsed_object_sha256": canonical_sha256(self.parsed_object),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "service_request_id_sha256": (
                None
                if self.service_request_id is None
                else _sha256_bytes(self.service_request_id.encode("utf-8"))
            ),
        }


Transport = Callable[[str, bytes | None, float], tuple[int, bytes, Mapping[str, str]]]


def _loopback_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise RuntimeConfigurationError("vLLM endpoint must be unauthenticated loopback HTTP")
    return base_url.rstrip("/")


def _urllib_transport(
    url: str,
    payload: bytes | None,
    timeout_seconds: float,
) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        # Preserve the actual service response.  HTTPError is also a URLError,
        # so it must be handled first or a schema/request rejection is
        # misleadingly reported as a connection failure.
        return exc.code, exc.read(), dict(exc.headers.items())
    except TimeoutError as exc:
        raise RuntimeWatchdogTimeout(f"local vLLM request exceeded {timeout_seconds}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RuntimeWatchdogTimeout(f"local vLLM request exceeded {timeout_seconds}s") from exc
        raise RuntimeTransportError(f"cannot reach local vLLM service: {exc.reason}") from exc


class VLLMGuidedJSONClient:
    """Concurrency-one guided-JSON client restricted to a local vLLM service."""

    def __init__(self, base_url: str, *, transport: Transport = _urllib_transport) -> None:
        self.base_url = _loopback_base_url(base_url)
        self._transport = transport
        self._generation_lock = threading.Lock()

    def _transport_with_wall_deadline(
        self,
        url: str,
        payload: bytes | None,
        timeout_seconds: float,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        """Bound the whole transport call, including a trickling response body."""

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise RuntimeConfigurationError("timeout_seconds must be positive and finite")
        completed = threading.Event()
        result: list[tuple[int, bytes, Mapping[str, str]]] = []
        failure: list[BaseException] = []

        def run_transport() -> None:
            try:
                result.append(self._transport(url, payload, timeout_seconds))
            except BaseException as exc:
                failure.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(
            target=run_transport,
            name="vllm-http-transport",
            daemon=True,
        )
        worker.start()
        if not completed.wait(timeout_seconds):
            raise RuntimeWatchdogTimeout(
                f"local vLLM request exceeded {timeout_seconds}s total wall time"
            )
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeTransportError("local vLLM transport completed without one response")
        return result[0]

    def health(self, timeout_seconds: float = 2.0) -> bool:
        try:
            status, _, _ = self._transport_with_wall_deadline(
                f"{self.base_url}/health",
                None,
                timeout_seconds,
            )
        except (RuntimeTransportError, RuntimeWatchdogTimeout):
            return False
        return status == 200

    def serves_model(self, model_name: str, timeout_seconds: float = 2.0) -> bool:
        """Require the loopback endpoint to expose the frozen served-model alias."""

        _require_plain_identifier("model_name", model_name)
        try:
            status, body, _ = self._transport_with_wall_deadline(
                f"{self.base_url}/v1/models",
                None,
                timeout_seconds,
            )
            payload = json.loads(body)
            rows = payload["data"]
        except (
            IndexError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
            RuntimeTransportError,
            RuntimeWatchdogTimeout,
        ):
            return False
        if status != 200 or not isinstance(rows, list):
            return False
        return any(isinstance(row, Mapping) and row.get("id") == model_name for row in rows)

    def endpoint_live(self, timeout_seconds: float = 0.25) -> bool:
        """Conservatively distinguish an absent endpoint from an unresponsive one."""

        try:
            self._transport_with_wall_deadline(
                f"{self.base_url}/health",
                None,
                timeout_seconds,
            )
        except RuntimeWatchdogTimeout:
            # A timeout is not proof that the controlled listener disappeared.
            return True
        except RuntimeTransportError:
            return False
        return True

    def ready(
        self,
        timeout_seconds: float = 2.0,
        *,
        model_name: str = SERVED_MODEL_NAME,
    ) -> bool:
        """Check both generic health and the exact model identity."""

        _require_plain_identifier("model_name", model_name)
        if not _is_allowlisted_model_alias(model_name):
            raise RuntimeConfigurationError(
                "readiness must target one allowlisted served-model alias"
            )
        return self.health(timeout_seconds) and self.serves_model(
            model_name,
            timeout_seconds,
        )

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        watchdog_seconds: float,
    ) -> GenerationResult:
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise RuntimeConfigurationError("watchdog_seconds must be positive and finite")
        payload = canonical_json(request.wire_payload()).encode("utf-8")
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeConfigurationError("concurrent vLLM generation is mechanically forbidden")

        completed = threading.Event()
        result: list[GenerationResult] = []
        failure: list[BaseException] = []

        def run_generation() -> None:
            try:
                status, body, headers = self._transport(
                    f"{self.base_url}/v1/chat/completions",
                    payload,
                    watchdog_seconds,
                )
                result.append(self._decode_generation_response(request, status, body, headers))
            except BaseException as exc:
                failure.append(exc)
            finally:
                self._generation_lock.release()
                completed.set()

        worker = threading.Thread(
            target=run_generation,
            name=f"vllm-generation-{request.request_id}",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            self._generation_lock.release()
            raise
        if not completed.wait(watchdog_seconds):
            raise RuntimeWatchdogTimeout(
                f"local vLLM generation exceeded {watchdog_seconds}s total wall time"
            )
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeTransportError("local vLLM generation completed without one response")
        return result[0]

    @staticmethod
    def _decode_generation_response(
        request: GuidedJSONRequest,
        status: int,
        body: bytes,
        headers: Mapping[str, str],
    ) -> GenerationResult:
        if status != 200:
            raise RuntimeTransportError(f"vLLM returned HTTP {status}")
        try:
            response = json.loads(body)
            choice = response["choices"][0]
            content = choice["message"]["content"]
            parsed_object = json.loads(content)
            usage = response["usage"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeTransportError("vLLM response is not one guided JSON choice") from exc
        if not isinstance(parsed_object, Mapping):
            raise RuntimeTransportError("guided_json response must decode to an object")
        if not isinstance(usage, Mapping):
            raise RuntimeTransportError("vLLM response usage must be an object")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens < 0
        ):
            raise RuntimeTransportError("vLLM usage must contain nonnegative integer token counts")
        service_request_id = response.get("id") or headers.get("x-request-id")
        if service_request_id is not None and not isinstance(service_request_id, str):
            raise RuntimeTransportError("vLLM response request ID must be text")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise RuntimeTransportError("vLLM finish reason must be text")
        return GenerationResult(
            request_id=request.request_id,
            request_hash=request.request_hash,
            response_sha256=_sha256_bytes(body),
            parsed_object=dict(parsed_object),
            raw_response=body,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            service_request_id=service_request_id,
        )


@dataclass(frozen=True, slots=True)
class ProcessTreeUsage:
    pids: frozenset[int]
    rss_bytes: int


def _proc_children(pid: int, proc_root: Path) -> tuple[int, ...]:
    children: set[int] = set()
    task_root = proc_root / str(pid) / "task"
    try:
        tasks = tuple(task_root.iterdir())
    except (FileNotFoundError, PermissionError):
        return ()
    for task in tasks:
        try:
            text = (task / "children").read_text(encoding="ascii").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        children.update(int(value) for value in text.split() if value.isdecimal())
    return tuple(sorted(children))


def sample_process_tree(root_pid: int, *, proc_root: Path = PROC_ROOT) -> ProcessTreeUsage:
    """Conservatively sum resident memory for a Linux process and descendants."""

    if root_pid <= 0:
        raise ValueError("root_pid must be positive")
    pending = [root_pid]
    seen: set[int] = set()
    rss_bytes = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        status_path = proc_root / str(pid) / "status"
        try:
            status = status_path.read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) < 2 or not fields[1].isdecimal():
                    raise RuntimeError(f"invalid VmRSS for process {pid}")
                rss_bytes += int(fields[1]) * 1024
                break
        pending.extend(_proc_children(pid, proc_root))
    return ProcessTreeUsage(pids=frozenset(seen), rss_bytes=rss_bytes)


def sample_system_available_ram(*, proc_root: Path = PROC_ROOT) -> int:
    for line in (proc_root / "meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdecimal():
                return int(fields[1]) * 1024
    raise RuntimeError("/proc/meminfo does not expose MemAvailable")


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def sample_gpu_vram(pids: frozenset[int], *, runner: CommandRunner = subprocess.run) -> int:
    """Sum compute-process VRAM for only the controlled process tree."""

    result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total_mib = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = tuple(part.strip() for part in line.split(","))
        if len(fields) != 2 or not fields[0].isdecimal() or not fields[1].isdecimal():
            raise RuntimeError("unexpected nvidia-smi compute-process row")
        if int(fields[0]) in pids:
            total_mib += int(fields[1])
    return total_mib * MEBIBYTE


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    sample_id: str
    sampled_at: datetime
    root_pid: int
    process_ids: tuple[int, ...]
    process_ram_bytes: int
    system_available_ram_bytes: int
    gpu_vram_bytes: int
    project_storage_bytes: int
    filesystem_free_bytes: int
    cpu_worker_count: int
    violations: tuple[str, ...] = ()

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "sampled_at": self.sampled_at.isoformat(),
            "process_count": len(self.process_ids),
            "process_ram_bytes": self.process_ram_bytes,
            "system_available_ram_bytes": self.system_available_ram_bytes,
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "project_storage_bytes": self.project_storage_bytes,
            "filesystem_free_bytes": self.filesystem_free_bytes,
            "cpu_worker_count": self.cpu_worker_count,
            "violations": list(self.violations),
        }


class ResourceSampler:
    """Hard-limit sampler that appends every observation before raising."""

    def __init__(
        self,
        *,
        limits: ResourceLimits,
        storage: StoragePreflight,
        ledger: Ledger | None = None,
        cpu_worker_count: int = MAXIMUM_CPU_WORKERS,
        process_sampler: Callable[[int], ProcessTreeUsage] = sample_process_tree,
        system_ram_sampler: Callable[[], int] = sample_system_available_ram,
        gpu_sampler: Callable[[frozenset[int]], int] = sample_gpu_vram,
        filesystem_free_sampler: Callable[[], int] | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if cpu_worker_count > limits.maximum_cpu_workers or cpu_worker_count > MAXIMUM_CPU_WORKERS:
            raise RuntimeConfigurationError("CPU worker count exceeds the hard eight-worker cap")
        if cpu_worker_count <= 0:
            raise RuntimeConfigurationError("CPU worker count must be positive")
        self.limits = limits
        self.storage = storage
        self.ledger = ledger
        self.cpu_worker_count = cpu_worker_count
        self._process_sampler = process_sampler
        self._system_ram_sampler = system_ram_sampler
        self._gpu_sampler = gpu_sampler
        self._filesystem_free_sampler = filesystem_free_sampler or (
            lambda: (
                os.statvfs(storage.quota_root).f_bavail * os.statvfs(storage.quota_root).f_frsize
            )
        )
        self._wall_clock = wall_clock
        self._samples: list[ResourceSnapshot] = []

    @property
    def samples(self) -> tuple[ResourceSnapshot, ...]:
        """Return all observations, including the sample that tripped a gate."""

        return tuple(self._samples)

    def sample(
        self,
        *,
        sample_id: str,
        root_pid: int,
        job_id: str | None = None,
        gpu_event_id: str | None = None,
    ) -> ResourceSnapshot:
        _require_plain_identifier("sample_id", sample_id)
        process = self._process_sampler(root_pid)
        project_storage = self.storage.measure_occupied_bytes()
        filesystem_free = self._filesystem_free_sampler()
        values = {
            "process_ram_bytes": process.rss_bytes,
            "system_available_ram_bytes": self._system_ram_sampler(),
            "gpu_vram_bytes": self._gpu_sampler(process.pids),
            "project_storage_bytes": project_storage,
            "filesystem_free_bytes": filesystem_free,
        }
        violations: list[str] = []
        if values["process_ram_bytes"] >= self.limits.maximum_process_ram_bytes:
            violations.append("process_ram_not_below_limit")
        if values["gpu_vram_bytes"] >= self.limits.maximum_peak_vram_bytes:
            violations.append("gpu_vram_not_below_limit")
        if project_storage > self.limits.maximum_project_occupied_bytes:
            violations.append("project_storage_exceeds_limit")
        if filesystem_free < self.limits.minimum_storage_headroom_bytes:
            violations.append("filesystem_headroom_below_minimum")
        if self.cpu_worker_count > self.limits.maximum_cpu_workers:
            violations.append("cpu_worker_count_exceeds_limit")
        sampled_at = self._wall_clock()
        if sampled_at.tzinfo is None or sampled_at.utcoffset() is None:
            raise RuntimeConfigurationError("resource sampler clock must be timezone-aware")
        snapshot = ResourceSnapshot(
            sample_id=sample_id,
            sampled_at=sampled_at,
            root_pid=root_pid,
            process_ids=tuple(sorted(process.pids)),
            process_ram_bytes=values["process_ram_bytes"],
            system_available_ram_bytes=values["system_available_ram_bytes"],
            gpu_vram_bytes=values["gpu_vram_bytes"],
            project_storage_bytes=project_storage,
            filesystem_free_bytes=filesystem_free,
            cpu_worker_count=self.cpu_worker_count,
            violations=tuple(violations),
        )
        self._samples.append(snapshot)
        if self.ledger is not None:
            storage_report = self.storage.check(
                current_occupied_bytes=project_storage,
                filesystem_free_bytes=filesystem_free,
            )
            self.ledger.record_storage_sample(
                storage_report,
                phase=f"resource_sample:{sample_id}",
                sampled_at=sampled_at,
            )
            self.ledger.record_resource_sample(
                sample_id=sample_id,
                job_id=job_id,
                gpu_event_id=gpu_event_id,
                process_ram_bytes=snapshot.process_ram_bytes,
                system_available_ram_bytes=snapshot.system_available_ram_bytes,
                gpu_vram_bytes=snapshot.gpu_vram_bytes,
                project_storage_bytes=snapshot.project_storage_bytes,
                cpu_worker_count=snapshot.cpu_worker_count,
                sampled_at=sampled_at,
            )
        if violations:
            raise RuntimeResourceLimitExceeded(snapshot)
        return snapshot


@dataclass(slots=True)
class ResourceWatchdog:
    """Periodically sample a live service and surface the first hard violation."""

    sampler: ResourceSampler
    root_pid: int
    sample_prefix: str
    interval_seconds: float = 1.0
    job_id: str | None = None
    gpu_event_id: str | None = None
    allocation_guard: Callable[[], None] | None = None
    on_limit: Callable[[ResourceSnapshot], None] | None = None
    on_failure: Callable[[BaseException], None] | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _failure: BaseException | None = field(default=None, init=False, repr=False)
    _sample_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _require_plain_identifier("sample_prefix", self.sample_prefix)
        if self.root_pid <= 0:
            raise RuntimeConfigurationError("resource watchdog root PID must be positive")
        if self.interval_seconds <= 0:
            raise RuntimeConfigurationError("resource watchdog interval must be positive")

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_count += 1
            try:
                if self.allocation_guard is not None:
                    self.allocation_guard()
                self.sampler.sample(
                    sample_id=f"{self.sample_prefix}-{self._sample_count:06d}",
                    root_pid=self.root_pid,
                    job_id=self.job_id,
                    gpu_event_id=self.gpu_event_id,
                )
            except RuntimeResourceLimitExceeded as exc:
                self._failure = exc
                if self.on_limit is not None:
                    self.on_limit(exc.snapshot)
                if self.on_failure is not None:
                    self.on_failure(exc)
                self._stop.set()
                return
            except BaseException as exc:
                self._failure = exc
                if self.on_failure is not None:
                    self.on_failure(exc)
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> Self:
        return self.start()

    def start(self) -> Self:
        if self._thread is not None:
            raise RuntimeError("resource watchdog cannot be started twice")
        self._thread = threading.Thread(
            target=self._run,
            name=f"resource-watchdog-{self.sample_prefix}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, raise_failure: bool = True) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        if self._thread.is_alive() and raise_failure:
            raise RuntimeError("resource watchdog did not stop")
        if self._failure is not None and raise_failure:
            raise self._failure

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.stop(raise_failure=exc is None)
        return False


class ServiceState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServiceUptime:
    session_id: str
    started_at: datetime
    ended_at: datetime
    service_seconds: float
    allocated_event_seconds: float

    @property
    def unclassified_service_seconds(self) -> float:
        return max(0.0, self.service_seconds - self.allocated_event_seconds)

    @property
    def allocated_service_seconds(self) -> float:
        """Scientific allocation including transparent between-event service time."""

        return max(self.service_seconds, self.allocated_event_seconds)

    def public_manifest(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "service_seconds": self.service_seconds,
            "allocated_event_seconds": self.allocated_event_seconds,
            "unclassified_service_seconds": self.unclassified_service_seconds,
            "allocated_service_seconds": self.allocated_service_seconds,
            "unclassified_time_persisted_as_service_overhead": True,
        }


@dataclass(frozen=True, slots=True)
class RecoveredServiceProcessIdentity:
    """Path-free process identity retained after terminal stale recovery."""

    configuration_hash: str
    session_id: str
    accounting_session_id: str
    pid: int
    process_start_ticks: int
    process_command_sha256: str
    service_started_at: datetime


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


PopenFactory = Callable[..., ProcessHandle]


def _signal_controlled_process_group(pid: int, signal_number: int) -> None:
    """Signal only a child that is the leader of its launcher-created session."""

    process_group = os.getpgid(pid)
    if process_group != pid:
        raise RuntimeError("refusing to signal a process outside its controlled vLLM group")
    os.killpg(process_group, signal_number)


def _process_group_alive(process_group: int) -> bool:
    """Return whether any process remains in the launcher-created process group."""

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Lack of permission is positive evidence that the group still exists.
        return True
    return True


def _process_alive(pid: int) -> bool:
    """Return whether a PID still exists, treating permission denial as live."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _available_cpu_affinity() -> set[int]:
    """Return this controller's affinity set using the required current-PID sentinel."""

    return os.sched_getaffinity(0)


def _process_start_ticks(pid: int, proc_root: Path = PROC_ROOT) -> int:
    stat = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    _, separator, suffix = stat.rpartition(")")
    if not separator:
        raise RuntimeError("cannot parse controlled process identity")
    fields = suffix.split()
    # suffix starts at procfs field 3; starttime is field 22.
    if len(fields) <= 19 or not fields[19].isdecimal():
        raise RuntimeError("controlled process has no valid start time")
    return int(fields[19])


def _process_command_sha256(pid: int, proc_root: Path = PROC_ROOT) -> str:
    """Hash the exact argv of a controlled process without persisting its paths."""

    raw = (proc_root / str(pid) / "cmdline").read_bytes()
    arguments = raw.split(b"\0")
    if arguments and arguments[-1] == b"":
        arguments.pop()
    if not arguments or any(not argument for argument in arguments):
        raise RuntimeError("controlled process has no valid command line")
    try:
        decoded = [argument.decode("utf-8") for argument in arguments]
    except UnicodeDecodeError as exc:
        raise RuntimeError("controlled process command line is not UTF-8") from exc
    return canonical_sha256(decoded)


@dataclass(slots=True)
class _AdoptedProcess:
    pid: int
    monotonic_clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 0
        return None

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else self.monotonic_clock() + timeout
        while self.poll() is None:
            if deadline is not None and self.monotonic_clock() >= deadline:
                raise subprocess.TimeoutExpired("adopted-vllm", timeout)
            self.sleep(0.05)
        return 0

    def terminate(self) -> None:
        os.kill(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        os.kill(self.pid, signal.SIGKILL)


@dataclass(slots=True)
class VLLMService:
    """Safe process lifecycle, metered without overlapping allocation intervals."""

    configuration: VLLMLaunchConfiguration
    client: VLLMGuidedJSONClient
    meter: AllocatedGPUMeter
    log_path: Path | None = None
    service_lock_path: Path | None = None
    startup_resource_sampler: ResourceSampler | None = None
    startup_sample_interval_seconds: float = 1.0
    service_heartbeat_interval_seconds: float = 5.0
    preflight_endpoint_check: Callable[[], bool] | None = None
    readiness_check: Callable[[], bool] | None = None
    popen_factory: PopenFactory = subprocess.Popen
    monotonic_clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    sleep: Callable[[float], None] = time.sleep
    process_group_signaler: Callable[[int, int], None] = _signal_controlled_process_group
    process_liveness_check: Callable[[int], bool] = _process_alive
    process_group_liveness_check: Callable[[int], bool] = _process_group_alive
    available_cpu_sampler: Callable[[], set[int]] = _available_cpu_affinity
    affinity_setter: Callable[[int, set[int]], None] = os.sched_setaffinity
    state: ServiceState = field(default=ServiceState.STOPPED, init=False)
    _process: ProcessHandle | None = field(default=None, init=False, repr=False)
    _last_service_pid: int | None = field(default=None, init=False, repr=False)
    _last_process_start_ticks: int | None = field(default=None, init=False, repr=False)
    _last_process_command_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _session_id: str | None = field(default=None, init=False, repr=False)
    _accounting_session_id: str | None = field(default=None, init=False, repr=False)
    _started_monotonic: float | None = field(default=None, init=False, repr=False)
    _started_at: datetime | None = field(default=None, init=False, repr=False)
    _allocated_at_start: float | None = field(default=None, init=False, repr=False)
    _carried_service_seconds: float = field(default=0.0, init=False, repr=False)
    _log_stream: BinaryIO | None = field(default=None, init=False, repr=False)
    _service_lock_stream: BinaryIO | None = field(default=None, init=False, repr=False)
    _prior_service_lease: dict[str, object] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _service_journal_opened: bool = field(default=False, init=False, repr=False)
    _service_heartbeat_stop: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )
    _service_heartbeat_thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _service_heartbeat_failure: BaseException | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _physically_stopped_at: datetime | None = field(default=None, init=False, repr=False)
    _physically_stopped_seconds: float | None = field(default=None, init=False, repr=False)
    _process_stop_journaled: bool = field(default=False, init=False, repr=False)
    _last_recovered_process_identity: RecoveredServiceProcessIdentity | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _process_control_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_path = self.log_path.resolve()
            if self.log_path.exists() and not self.log_path.is_file():
                raise RuntimeConfigurationError("vLLM log path must be a regular file")
        if self.startup_sample_interval_seconds <= 0:
            raise RuntimeConfigurationError("startup sample interval must be positive")
        if (
            not math.isfinite(self.service_heartbeat_interval_seconds)
            or self.service_heartbeat_interval_seconds <= 0
            or self.service_heartbeat_interval_seconds > 5
        ):
            raise RuntimeConfigurationError(
                "service heartbeat interval must be positive and no greater than five seconds"
            )
        lock_path = self.service_lock_path
        if lock_path is None:
            # The frozen configuration's one shared cache is project-scoped and
            # common to every ledger/checkpoint for this model execution.
            lock_path = self.configuration.shared_cache / SERVICE_LOCK_FILENAME
        lock_path = Path(lock_path)
        if lock_path.is_symlink():
            raise RuntimeConfigurationError("vLLM service lock path must not be a symlink")
        expected_lock_path = (self.configuration.shared_cache / SERVICE_LOCK_FILENAME).resolve()
        self.service_lock_path = lock_path.resolve()
        if self.service_lock_path != expected_lock_path:
            raise RuntimeConfigurationError(
                "vLLM service lock must use the project shared-cache lock path"
            )
        if self.service_lock_path.exists() and not self.service_lock_path.is_file():
            raise RuntimeConfigurationError("vLLM service lock path must be a regular file")

    def _write_service_lock_metadata(
        self,
        *,
        lease_state: str,
        service_pid: int | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        stream = self._service_lock_stream
        if stream is None:
            raise RuntimeError("vLLM service lock is not held")
        _require_plain_identifier("lease_state", lease_state)
        updated_at = self.wall_clock()
        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise RuntimeConfigurationError("service lease clock must be timezone-aware")
        if ended_at is not None and (ended_at.tzinfo is None or ended_at.utcoffset() is None):
            raise RuntimeConfigurationError("service lease end must be timezone-aware")
        if self._started_at is not None and (
            self._started_at.tzinfo is None or self._started_at.utcoffset() is None
        ):
            raise RuntimeConfigurationError("service lease start must be timezone-aware")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "configuration_hash": self.configuration.configuration_hash,
            "controller_pid": os.getpid(),
            "lease_state": lease_state,
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "service_pid": service_pid,
            "process_start_ticks": self._last_process_start_ticks,
            "process_command_sha256": self._last_process_command_sha256,
            "service_started_at": (
                None if self._started_at is None else self._started_at.isoformat()
            ),
            "service_ended_at": None if ended_at is None else ended_at.isoformat(),
            "ledger_allocated_seconds_before_session": self._allocated_at_start,
            "observed_service_seconds": (
                None if self._started_at is None else self._elapsed_service_seconds()
            ),
            "updated_at": updated_at.isoformat(),
        }
        metadata = canonical_json(payload).encode("utf-8")
        stream.seek(0)
        stream.truncate()
        stream.write(metadata + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._prior_service_lease = payload

    def _acquire_service_lock(self) -> bool:
        """Acquire the nonblocking project lock; return whether this call acquired it."""

        if self._service_lock_stream is not None:
            return False
        lock_path = cast(Path, self.service_lock_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise RuntimeConfigurationError("vLLM service lock path must be a regular file")
            if lock_stat.st_size > 65_536:
                raise RuntimeConfigurationError("vLLM service lease metadata is too large")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeConfigurationError(
                    "another project controller holds the exclusive vLLM service lock"
                ) from exc
            os.lseek(descriptor, 0, os.SEEK_SET)
            prior_bytes = os.read(descriptor, lock_stat.st_size)
            if prior_bytes:
                try:
                    prior_value = json.loads(prior_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeConfigurationError(
                        "vLLM service lease metadata is invalid"
                    ) from exc
                if not isinstance(prior_value, Mapping):
                    raise RuntimeConfigurationError("vLLM service lease metadata must be an object")
                prior_lease = dict(prior_value)
            else:
                prior_lease = None
            stream = os.fdopen(descriptor, "r+b", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise
        try:
            directory_descriptor = os.open(lock_path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
        self._service_lock_stream = stream
        self._prior_service_lease = prior_lease
        return True

    def _release_service_lock(self) -> None:
        stream = self._service_lock_stream
        self._service_lock_stream = None
        self._prior_service_lease = None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def _open_service_journal(self) -> None:
        if self._service_journal_opened:
            return
        if (
            self._accounting_session_id is None
            or self._session_id is None
            or self._started_at is None
            or self._allocated_at_start is None
        ):
            raise RuntimeError("vLLM service accounting identity is incomplete")
        self.meter.open_service_journal(
            service_session_id=self._accounting_session_id,
            session_id=self._session_id,
            configuration_hash=self.configuration.configuration_hash,
            started_at=self._started_at,
            ledger_allocated_seconds_before_session=self._allocated_at_start,
            details={"model_candidate": self.configuration.model_candidate},
        )
        self._service_journal_opened = True

    def _observe_service_journal(
        self,
        *,
        process_stopped: bool = False,
        observed_at: datetime | None = None,
        elapsed_seconds: float | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if not self._service_journal_opened:
            return
        if self._accounting_session_id is None:
            raise RuntimeError("vLLM service journal has no accounting identity")
        self.meter.observe_service_journal(
            service_session_id=self._accounting_session_id,
            elapsed_seconds=(
                self._elapsed_service_seconds() if elapsed_seconds is None else elapsed_seconds
            ),
            observed_at=self.wall_clock() if observed_at is None else observed_at,
            process_stopped=process_stopped,
            details=details,
        )

    def _adopt_service_journal(self) -> None:
        if (
            self._accounting_session_id is None
            or self._session_id is None
            or self._started_at is None
            or self._allocated_at_start is None
        ):
            raise RuntimeError("resumed vLLM service accounting identity is incomplete")
        latest = self.meter.ledger.latest_gpu_service_journal(self._accounting_session_id)
        if latest is None or latest.state.value not in {"opened", "heartbeat"}:
            raise RuntimeConfigurationError(
                "resumed vLLM service has no matching open service journal"
            )
        journal_started_at = _parse_aware_datetime(
            "service journal start",
            latest.service_started_at,
        )
        if (
            latest.session_id != self._session_id
            or latest.configuration_hash != self.configuration.configuration_hash
            or journal_started_at != self._started_at
            or abs(
                latest.ledger_allocated_microseconds_before_session / 1_000_000
                - self._allocated_at_start
            )
            > 1e-6
        ):
            raise RuntimeConfigurationError(
                "resumed vLLM service journal identity differs from its checkpoint"
            )
        self._carried_service_seconds = max(
            self._carried_service_seconds,
            latest.elapsed_microseconds / 1_000_000,
        )
        self._service_journal_opened = True

    def _validate_resume_lease(
        self,
        service_pid: int,
        *,
        controller_restart_handoff: bool = False,
    ) -> None:
        lease = self._prior_service_lease
        if lease is None:
            raise RuntimeConfigurationError(
                "resumed vLLM service has no matching durable service lease"
            )
        baseline = lease.get("ledger_allocated_seconds_before_session")
        if isinstance(baseline, bool) or not isinstance(baseline, int | float):
            raise RuntimeConfigurationError("resumed vLLM service lease baseline is invalid")
        expected = {
            "configuration_hash": self.configuration.configuration_hash,
            "lease_state": (
                "controller_restart_handoff"
                if controller_restart_handoff
                else "live"
            ),
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "service_pid": service_pid,
            "service_started_at": cast(datetime, self._started_at).isoformat(),
        }
        mismatches = [name for name, value in expected.items() if lease.get(name) != value]
        if (
            self._allocated_at_start is None
            or abs(float(baseline) - self._allocated_at_start) > 1e-6
        ):
            mismatches.append("ledger_allocated_seconds_before_session")
        if mismatches:
            raise RuntimeConfigurationError(
                "resumed vLLM service lease identity differs from its checkpoint: "
                + ", ".join(mismatches)
            )

    def _journal_verified_process_stop(self) -> None:
        if self._process_stop_journaled or not self._service_journal_opened:
            return
        if (
            self._accounting_session_id is None
            or self._physically_stopped_at is None
            or self._physically_stopped_seconds is None
        ):
            raise RuntimeError("verified vLLM process stop identity is incomplete")
        try:
            self._observe_service_journal(
                process_stopped=True,
                observed_at=self._physically_stopped_at,
                elapsed_seconds=self._physically_stopped_seconds,
                details={"process_and_endpoint_absence_verified": True},
            )
        except BaseException:
            # The meter raises a strict hard-limit exception *after* persisting
            # the observation.  Preserve that durable transition and let final
            # reconciliation surface the same hard-boundary failure.
            latest = self.meter.ledger.latest_gpu_service_journal(self._accounting_session_id)
            if (
                latest is None
                or latest.state is not GpuServiceJournalState.PROCESS_STOPPED
                or _parse_aware_datetime("service journal stop", latest.observed_at)
                != self._physically_stopped_at
            ):
                raise
        self._process_stop_journaled = True

    def _service_heartbeat(self) -> None:
        while not self._service_heartbeat_stop.wait(self.service_heartbeat_interval_seconds):
            try:
                self.require_hard_stop_margin()
                self._observe_service_journal(
                    details={
                        "service_pid": None if self._process is None else self._process.pid,
                    }
                )
            except BaseException as exc:
                self._service_heartbeat_failure = exc
                self._service_heartbeat_stop.set()
                process = self._process
                if process is not None and process.poll() is None:
                    with suppress(Exception):
                        self.process_group_signaler(process.pid, signal.SIGKILL)
                return

    def _start_service_heartbeat(self) -> None:
        if not self._service_journal_opened:
            return
        if self._service_heartbeat_thread is not None and self._service_heartbeat_thread.is_alive():
            raise RuntimeError("vLLM service heartbeat is already running")
        if self._service_heartbeat_failure is not None:
            raise RuntimeError("vLLM service heartbeat previously failed") from (
                self._service_heartbeat_failure
            )
        self._service_heartbeat_stop.clear()
        self._service_heartbeat_thread = threading.Thread(
            target=self._service_heartbeat,
            name=f"vllm-service-heartbeat-{self._accounting_session_id}",
            daemon=True,
        )
        self._service_heartbeat_thread.start()

    def _stop_service_heartbeat(self) -> None:
        self._service_heartbeat_stop.set()
        thread = self._service_heartbeat_thread
        if thread is None:
            return
        thread.join(timeout=self.service_heartbeat_interval_seconds + 1.0)
        if thread.is_alive():
            raise RuntimeError("vLLM service heartbeat did not stop")
        self._service_heartbeat_thread = None

    def _raise_service_heartbeat_failure(self) -> None:
        if self._service_heartbeat_failure is not None:
            raise RuntimeError("vLLM service heartbeat failed") from (
                self._service_heartbeat_failure
            )

    @property
    def pid(self) -> int:
        if self._process is None:
            raise RuntimeError("vLLM service has no controlled process")
        return self._process.pid

    @property
    def last_recovered_process_identity(
        self,
    ) -> RecoveredServiceProcessIdentity | None:
        """Return exact lease identity only after successful terminal recovery."""

        return self._last_recovered_process_identity

    def _open_log_stream(self) -> BinaryIO | int:
        if self.log_path is None:
            return subprocess.DEVNULL
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.log_path, flags, 0o600)
        os.chmod(self.log_path, 0o600)
        self._log_stream = os.fdopen(descriptor, "ab", buffering=0)
        return self._log_stream

    def _close_log_stream(self) -> None:
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def _spawn(self) -> None:
        output = self._open_log_stream()
        command = self.configuration.command()
        self._process = self.popen_factory(
            command,
            env=self.configuration.environment(),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._last_service_pid = self._process.pid
        self._last_process_command_sha256 = canonical_sha256(list(command))
        try:
            self._last_process_start_ticks = _process_start_ticks(self._process.pid)
            observed_command_sha256 = _process_command_sha256(self._process.pid)
        except (OSError, RuntimeError) as exc:
            # Unit-test process handles do not have procfs entries.  A real
            # launcher must make the PID-reuse-safe identity durable before it
            # can call the service live or recoverable.
            if self.popen_factory is subprocess.Popen:
                raise RuntimeConfigurationError(
                    "cannot persist the launched vLLM process identity"
                ) from exc
            self._last_process_start_ticks = None
            self._last_process_command_sha256 = None
        else:
            if observed_command_sha256 != self._last_process_command_sha256:
                raise RuntimeConfigurationError(
                    "launched vLLM process command differs from its frozen argv"
                )
        # Persist the only controlled PID immediately after Popen returns.  If
        # later setup fails, stale-journal recovery can prove both PID and
        # process-group absence before terminalizing the allocation.
        self._write_service_lock_metadata(
            lease_state="starting",
            service_pid=self._process.pid,
        )
        allowed_cpus = set(sorted(self.available_cpu_sampler())[: self.configuration.cpu_workers])
        if not allowed_cpus:
            raise RuntimeConfigurationError("vLLM process has no permitted CPU affinity")
        self.affinity_setter(self._process.pid, allowed_cpus)

    def _ready(self, timeout_seconds: float = 4.0) -> bool:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise RuntimeConfigurationError("readiness timeout must be positive and finite")
        if self.readiness_check is None:
            # ``client.ready`` performs two sequential HTTP operations.  Give
            # each half of the total readiness deadline.
            return self.client.ready(
                max(0.001, timeout_seconds / 2),
                model_name=self.configuration.served_model_name,
            )
        completed = threading.Event()
        result: list[bool] = []
        failure: list[BaseException] = []

        def check() -> None:
            try:
                result.append(bool(cast(Callable[[], bool], self.readiness_check)()))
            except BaseException as exc:
                failure.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(
            target=check,
            name="vllm-readiness-check",
            daemon=True,
        )
        worker.start()
        if not completed.wait(timeout_seconds):
            raise RuntimeWatchdogTimeout("vLLM readiness check exceeded its wall deadline")
        if failure:
            raise failure[0]
        if len(result) != 1:
            raise RuntimeTransportError("vLLM readiness check returned no result")
        return result[0]

    def _wait_until_healthy(self, watchdog_seconds: float) -> None:
        deadline = self.monotonic_clock() + watchdog_seconds
        while self.monotonic_clock() < deadline:
            self._raise_service_heartbeat_failure()
            if self._process is None or self._process.poll() is not None:
                self.state = ServiceState.FAILED
                raise RuntimeTransportError("vLLM process exited before becoming healthy")
            remaining = max(0.001, deadline - self.monotonic_clock())
            if self._ready(remaining):
                self.state = ServiceState.READY
                return
            self.sleep(min(0.25, max(0.0, deadline - self.monotonic_clock())))
        self.state = ServiceState.FAILED
        raise RuntimeWatchdogTimeout("vLLM service did not become healthy before watchdog")

    def _start_unmetered(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float,
    ) -> None:
        if self.state is not ServiceState.STOPPED:
            raise RuntimeError("vLLM service can start only from stopped state")
        acquired_here = self._acquire_service_lock()
        prior_state = (
            None
            if self._prior_service_lease is None
            else self._prior_service_lease.get("lease_state")
        )
        if acquired_here and prior_state not in {None, "stopped_verified"}:
            self._release_service_lock()
            raise RuntimeConfigurationError(
                "an unresolved prior vLLM service lease forbids a new launch"
            )
        try:
            endpoint_already_live = self._endpoint_live(0.25)
            if self.preflight_endpoint_check is not None:
                endpoint_already_live = (
                    bool(self.preflight_endpoint_check()) or endpoint_already_live
                )
        except BaseException:
            self._release_service_lock()
            raise
        if endpoint_already_live:
            self._release_service_lock()
            raise RuntimeConfigurationError(
                "configured loopback port already exposes a service; refusing a duplicate launch"
            )
        startup_watchdog: ResourceWatchdog | None = None
        try:
            self.state = ServiceState.STARTING
            self._session_id = session_id
            if self._accounting_session_id is None:
                self._accounting_session_id = event_id
            if self._started_monotonic is None:
                self._started_monotonic = self.monotonic_clock()
            if self._started_at is None:
                started_at = self.wall_clock()
                if started_at.tzinfo is None or started_at.utcoffset() is None:
                    raise RuntimeConfigurationError("service start clock must be timezone-aware")
                self._started_at = started_at
            if self._allocated_at_start is None:
                self._allocated_at_start = self.meter.actual_allocated_gpu_seconds
            self._write_service_lock_metadata(lease_state="starting")
            self._open_service_journal()
            self._start_service_heartbeat()
            self._spawn()
            self._raise_service_heartbeat_failure()
            self._write_service_lock_metadata(lease_state="live", service_pid=self.pid)
            if self.startup_resource_sampler is not None:
                startup_watchdog = ResourceWatchdog(
                    sampler=self.startup_resource_sampler,
                    root_pid=self.pid,
                    sample_prefix=f"{event_id}-startup",
                    interval_seconds=self.startup_sample_interval_seconds,
                    allocation_guard=self.require_hard_stop_margin,
                    on_failure=lambda _: self.emergency_stop(),
                )
                startup_watchdog.start()
            self._wait_until_healthy(watchdog_seconds)
            if startup_watchdog is not None:
                startup_watchdog.stop()
        except BaseException:
            if startup_watchdog is not None:
                startup_watchdog.stop(raise_failure=False)
            self._stop_process(release_lock=False)
            raise

    def start(
        self,
        *,
        session_id: str,
        event_id: str,
        watchdog_seconds: float = DEFAULT_SERVICE_START_WATCHDOG_SECONDS,
        remaining_required_seconds: float = 0,
    ) -> None:
        _require_plain_identifier("session_id", session_id)
        if (
            self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("a prior vLLM service session must be reconciled before a new start")
        with self.meter.session_start(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            details={
                "session_id": session_id,
                "configuration_hash": self.configuration.configuration_hash,
            },
        ):
            self._start_unmetered(
                session_id=session_id,
                event_id=event_id,
                watchdog_seconds=watchdog_seconds,
            )

    def restart(
        self,
        *,
        event_id: str,
        watchdog_seconds: float,
        remaining_required_seconds: float = 0,
    ) -> None:
        if self._session_id is None:
            raise RuntimeError("cannot restart a service without a session identity")
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        session_id = self._session_id
        with self.meter.restart(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            details={
                "session_id": session_id,
                "configuration_hash": self.configuration.configuration_hash,
            },
        ):
            self._stop_process(
                release_lock=False,
                continuing_service=True,
            )
            self._start_unmetered(
                session_id=session_id,
                event_id=event_id,
                watchdog_seconds=watchdog_seconds,
            )

    def write_resume_checkpoint(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
        controller_restart_handoff: bool = False,
    ) -> None:
        """Persist a PID-reuse-safe operational checkpoint without a model path."""

        self.require_ready()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "configuration_hash": self.configuration.configuration_hash,
            "controller_pid": os.getpid(),
            "controller_restart_handoff": controller_restart_handoff,
            "session_id": self._session_id,
            "accounting_session_id": self._accounting_session_id,
            "pid": self.pid,
            "process_start_ticks": _process_start_ticks(self.pid, proc_root),
            "ledger_allocated_seconds_before_session": self._allocated_at_start,
            "session_service_seconds": self._elapsed_service_seconds(),
            "session_started_at": cast(datetime, self._started_at).isoformat(),
            "recorded_at": self.wall_clock().isoformat(),
        }
        _atomic_write_private_json(path, payload)

    def detach_for_controller_restart(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
    ) -> None:
        """Leave one verified vLLM process live for a new controller invocation.

        This is intentionally narrower than a model restart: the service PID and
        loaded weights remain unchanged.  The exact PID/start-ticks/configuration
        checkpoint lets a later controller adopt it without a second load.  Any
        exception before ownership is released leaves this controller capable of
        performing the normal fail-safe shutdown.
        """

        self.require_ready()
        self.write_resume_checkpoint(
            path,
            proc_root=proc_root,
            controller_restart_handoff=True,
        )
        self._observe_service_journal(
            details={
                "controller_restart_handoff": True,
                "controller_pid": os.getpid(),
                "service_pid": self.pid,
            }
        )
        self._stop_service_heartbeat()
        process = self._process
        assert process is not None
        self._write_service_lock_metadata(
            lease_state="controller_restart_handoff",
            service_pid=process.pid,
        )
        self._close_log_stream()
        self._release_service_lock()
        self._process = None
        self._last_service_pid = None
        self._last_process_start_ticks = None
        self._last_process_command_sha256 = None
        self._session_id = None
        self._accounting_session_id = None
        self._started_at = None
        self._started_monotonic = None
        self._allocated_at_start = None
        self._carried_service_seconds = 0.0
        self._service_journal_opened = False
        self.state = ServiceState.STOPPED

    def resume_live_service_lease(
        self,
        *,
        expected_session_id: str,
        expected_event_id: str,
        watchdog_seconds: float = DEFAULT_SERVICE_START_WATCHDOG_SECONDS,
        proc_root: Path = PROC_ROOT,
        adopted_factory: Callable[[int], ProcessHandle] = _AdoptedProcess,
    ) -> bool:
        """Adopt one exact live lease when its controller died before checkpointing.

        This path never launches a process and never opens a second GPU event.  The
        exclusive project lock proves the prior controller released ownership; the
        lease and open service journal then have to agree on configuration, logical
        session, accounting event, start time, and baseline.  PID start ticks and an
        exact argv hash close the PID-reuse and lookalike-process holes.  ``False``
        means the validated process is absent and is deliberately distinct from
        terminal stale-journal reconciliation.
        """

        _require_plain_identifier("expected_session_id", expected_session_id)
        _require_plain_identifier("expected_event_id", expected_event_id)
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise RuntimeConfigurationError(
                "live-lease recovery watchdog must be positive and finite"
            )
        if (
            self.state is not ServiceState.STOPPED
            or self._process is not None
            or self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("live-lease recovery requires a fresh stopped controller")
        self._last_recovered_process_identity = None
        acquired_here = self._acquire_service_lock()
        candidate_pid: int | None = None
        try:
            lease = self._prior_service_lease
            if lease is None or lease.get("lease_state") == "stopped_verified":
                return False
            lease_state = lease.get("lease_state")
            if lease_state not in {
                "starting",
                "live",
                "controller_restart_handoff",
                "accounting_pending",
                "shutdown_unverified",
            }:
                raise RuntimeConfigurationError("live vLLM service lease state is invalid")
            pid = lease.get("service_pid")
            start_ticks = lease.get("process_start_ticks")
            command_sha256 = lease.get("process_command_sha256")
            session_id = lease.get("session_id")
            accounting_session_id = lease.get("accounting_session_id")
            started_at_text = lease.get("service_started_at")
            baseline = lease.get("ledger_allocated_seconds_before_session")
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or isinstance(start_ticks, bool)
                or not isinstance(start_ticks, int)
                or start_ticks <= 0
                or not isinstance(command_sha256, str)
                or len(command_sha256) != 64
                or any(character not in "0123456789abcdef" for character in command_sha256)
                or not isinstance(session_id, str)
                or not isinstance(accounting_session_id, str)
                or not isinstance(started_at_text, str)
                or isinstance(baseline, bool)
                or not isinstance(baseline, int | float)
                or not math.isfinite(float(baseline))
                or baseline < 0
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease lacks an exact recoverable process identity"
                )
            candidate_pid = pid
            expected_command_sha256 = canonical_sha256(
                list(self.configuration.command())
            )
            if (
                lease.get("configuration_hash")
                != self.configuration.configuration_hash
                or session_id != expected_session_id
                or accounting_session_id != expected_event_id
                or command_sha256 != expected_command_sha256
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease differs from the expected activation identity"
                )
            started_at = _parse_aware_datetime(
                "live service lease start",
                started_at_text,
            )
            latest = self.meter.ledger.latest_gpu_service_journal(expected_event_id)
            if latest is None or latest.state not in {
                GpuServiceJournalState.OPENED,
                GpuServiceJournalState.HEARTBEAT,
            }:
                raise RuntimeConfigurationError(
                    "live vLLM lease has no matching open service journal"
                )
            if (
                latest.session_id != expected_session_id
                or latest.configuration_hash != self.configuration.configuration_hash
                or _parse_aware_datetime(
                    "live service journal start",
                    latest.service_started_at,
                )
                != started_at
                or abs(
                    latest.ledger_allocated_microseconds_before_session / 1_000_000
                    - float(baseline)
                )
                > 1e-6
            ):
                raise RuntimeConfigurationError(
                    "live vLLM lease and service journal identities differ"
                )
            try:
                pid_live = self.process_liveness_check(pid)
                process_group_live = self.process_group_liveness_check(pid)
            except BaseException as exc:
                raise RuntimeConfigurationError(
                    "cannot verify the leased vLLM process identity"
                ) from exc
            if not pid_live and not process_group_live:
                if self._endpoint_live(0.25):
                    raise RuntimeConfigurationError(
                        "leased vLLM process is absent but its endpoint remains live"
                    )
                return False
            if not pid_live or not process_group_live:
                raise RuntimeConfigurationError(
                    "leased vLLM PID and process-group liveness disagree"
                )
            if lease_state in {"accounting_pending", "shutdown_unverified"}:
                raise RuntimeConfigurationError(
                    "terminalizing vLLM lease cannot be adopted as a ready service"
                )
            try:
                observed_start_ticks = _process_start_ticks(pid, proc_root)
                observed_command_sha256 = _process_command_sha256(pid, proc_root)
            except (OSError, RuntimeError) as exc:
                raise RuntimeConfigurationError(
                    "cannot inspect the leased vLLM process"
                ) from exc
            if observed_start_ticks != start_ticks:
                raise RuntimeConfigurationError("live vLLM lease PID was reused")
            if observed_command_sha256 != command_sha256:
                raise RuntimeConfigurationError(
                    "live vLLM lease command line changed"
                )
            adopted = adopted_factory(pid)
            if adopted.poll() is not None:
                return False
            resumed_at = self.wall_clock()
            if resumed_at.tzinfo is None or resumed_at.utcoffset() is None:
                raise RuntimeConfigurationError("live-lease recovery clock must be aware")
            wall_service_seconds = (resumed_at - started_at).total_seconds()
            if wall_service_seconds < 0:
                raise RuntimeConfigurationError(
                    "live-lease recovery predates the service session"
                )
            self._process = adopted
            self._last_service_pid = pid
            self._last_process_start_ticks = start_ticks
            self._last_process_command_sha256 = command_sha256
            self._session_id = expected_session_id
            self._accounting_session_id = expected_event_id
            self._started_at = started_at
            self._started_monotonic = self.monotonic_clock()
            self._allocated_at_start = float(baseline)
            self._carried_service_seconds = wall_service_seconds
            self._adopt_service_journal()
            remaining = watchdog_seconds - wall_service_seconds
            if remaining <= 0:
                self.state = ServiceState.FAILED
                raise RuntimeWatchdogTimeout(
                    "live vLLM lease exceeded its original startup watchdog"
                )
            self.state = ServiceState.STARTING
            self._wait_until_healthy(remaining)
            self._write_service_lock_metadata(lease_state="live", service_pid=pid)
            self._observe_service_journal(
                details={
                    "recovered_without_checkpoint": True,
                    "service_pid": pid,
                }
            )
            self._start_service_heartbeat()
            return True
        finally:
            if acquired_here and self._process is None:
                endpoint_absent = False
                process_absent = candidate_pid is None
                with suppress(Exception):
                    endpoint_absent = not self._endpoint_live(0.25)
                if candidate_pid is not None:
                    with suppress(Exception):
                        process_absent = not self.process_liveness_check(
                            candidate_pid
                        ) and not self.process_group_liveness_check(candidate_pid)
                if endpoint_absent and process_absent:
                    self._release_service_lock()
                else:
                    self.state = ServiceState.FAILED

    def resume_from_checkpoint(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
        adopted_factory: Callable[[int], ProcessHandle] = _AdoptedProcess,
        allow_same_controller_cleanup: bool = False,
    ) -> bool:
        """Adopt the exact still-live service; never start or meter a duplicate load."""

        if self.state is not ServiceState.STOPPED or self._process is not None:
            raise RuntimeError("resume requires a stopped controller")
        if not path.exists():
            return False
        acquired_here = self._acquire_service_lock()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise RuntimeConfigurationError("service checkpoint root must be an object")
            if value.get("configuration_hash") != self.configuration.configuration_hash:
                raise RuntimeConfigurationError("service checkpoint configuration changed")
            pid = value.get("pid")
            start_ticks = value.get("process_start_ticks")
            carried_seconds = value.get("session_service_seconds")
            ledger_before_session = value.get("ledger_allocated_seconds_before_session")
            started_at_text = value.get("session_started_at")
            session_id = value.get("session_id")
            accounting_session_id = value.get("accounting_session_id")
            checkpoint_controller_pid = value.get("controller_pid")
            controller_restart_handoff = value.get(
                "controller_restart_handoff",
                False,
            )
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or isinstance(start_ticks, bool)
                or not isinstance(start_ticks, int)
                or isinstance(carried_seconds, bool)
                or not isinstance(carried_seconds, int | float)
                or carried_seconds < 0
                or isinstance(ledger_before_session, bool)
                or not isinstance(ledger_before_session, int | float)
                or ledger_before_session < 0
                or not isinstance(started_at_text, str)
                or not isinstance(session_id, str)
                or not isinstance(accounting_session_id, str)
                or not accounting_session_id
                or isinstance(checkpoint_controller_pid, bool)
                or not isinstance(checkpoint_controller_pid, int)
                or checkpoint_controller_pid <= 0
                or not isinstance(controller_restart_handoff, bool)
            ):
                raise RuntimeConfigurationError("service checkpoint has invalid process identity")
            if (
                controller_restart_handoff
                and checkpoint_controller_pid == os.getpid()
                and not allow_same_controller_cleanup
            ):
                raise RuntimeConfigurationError(
                    "controller-restart handoff requires a different controller PID"
                )
            try:
                started_at = datetime.fromisoformat(started_at_text)
            except ValueError as exc:
                raise RuntimeConfigurationError("checkpoint session start is not ISO-8601") from exc
            if started_at.tzinfo is None or started_at.utcoffset() is None:
                raise RuntimeConfigurationError("checkpoint session start must include an offset")
            try:
                observed_start_ticks = _process_start_ticks(pid, proc_root)
            except (FileNotFoundError, ProcessLookupError):
                return False
            if observed_start_ticks != start_ticks:
                raise RuntimeConfigurationError("service checkpoint PID was reused")
            try:
                command_line = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
            except OSError as exc:
                raise RuntimeConfigurationError("cannot inspect checkpointed service") from exc
            required_parts = {
                b"vllm.entrypoints.openai.api_server",
                str(self.configuration.snapshot_path).encode("utf-8"),
            }
            if not required_parts.issubset(set(command_line)):
                raise RuntimeConfigurationError(
                    "checkpoint process is not the configured vLLM service"
                )
            adopted = adopted_factory(pid)
            if adopted.poll() is not None:
                return False
            resumed_at = self.wall_clock()
            if resumed_at.tzinfo is None or resumed_at.utcoffset() is None:
                raise RuntimeConfigurationError("resume clock must be timezone-aware")
            wall_service_seconds = (resumed_at - started_at).total_seconds()
            if wall_service_seconds < 0:
                raise RuntimeConfigurationError("resume clock predates the checkpointed session")
            self._process = adopted
            self._last_service_pid = pid
            self._last_process_start_ticks = observed_start_ticks
            self._last_process_command_sha256 = _process_command_sha256(pid, proc_root)
            self._session_id = session_id
            self._accounting_session_id = accounting_session_id
            self._started_at = started_at
            self._started_monotonic = self.monotonic_clock()
            self._allocated_at_start = float(ledger_before_session)
            # A service that remained live consumed allocation between checkpoint
            # write and controller recovery. Charge the larger persisted/wall gap.
            self._carried_service_seconds = max(float(carried_seconds), wall_service_seconds)
            self._validate_resume_lease(
                pid,
                controller_restart_handoff=controller_restart_handoff,
            )
            self._adopt_service_journal()
            self._write_service_lock_metadata(lease_state="live", service_pid=pid)
            try:
                ready = self._ready()
            except BaseException as readiness_error:
                # Ownership has already transferred to this controller.  A
                # failed probe must not leave the adopted process detached: it
                # is now safe (and mandatory) to stop it and reconcile the
                # already-adopted service journal before reporting the probe
                # error to the caller.
                self.state = ServiceState.FAILED
                try:
                    self.shutdown()
                except BaseException as cleanup_error:
                    raise RuntimeConfigurationError(
                        "checkpoint adoption failed and the adopted service "
                        "could not be terminally reconciled"
                    ) from cleanup_error
                raise readiness_error
            if not ready:
                self.state = ServiceState.FAILED
                try:
                    self.shutdown()
                except BaseException as cleanup_error:
                    raise RuntimeConfigurationError(
                        "unready adopted service could not be terminally reconciled"
                    ) from cleanup_error
                return False
            self.state = ServiceState.READY
            self._observe_service_journal(details={"resumed_controller": True, "service_pid": pid})
            self._start_service_heartbeat()
            return True
        finally:
            if acquired_here and self._process is None:
                # A malformed/stale checkpoint is not proof that a detached
                # service disappeared. Retain the lease unless endpoint absence
                # can be positively established, so a subsequent start cannot
                # create an unowned duplicate.
                endpoint_absent = False
                with suppress(Exception):
                    endpoint_absent = not self._endpoint_live(0.25)
                if endpoint_absent:
                    self._release_service_lock()
                else:
                    self.state = ServiceState.FAILED

    def handoff_resume(
        self,
        path: Path,
        *,
        proc_root: Path = PROC_ROOT,
    ) -> VLLMService:
        """Exercise checkpoint/resume by transferring the live controlled handle."""

        self.write_resume_checkpoint(path, proc_root=proc_root)
        self._stop_service_heartbeat()
        process = self._process
        assert process is not None
        resumed = VLLMService(
            configuration=self.configuration,
            client=self.client,
            meter=self.meter,
            log_path=self.log_path,
            service_lock_path=self.service_lock_path,
            startup_resource_sampler=self.startup_resource_sampler,
            startup_sample_interval_seconds=self.startup_sample_interval_seconds,
            service_heartbeat_interval_seconds=self.service_heartbeat_interval_seconds,
            preflight_endpoint_check=self.preflight_endpoint_check,
            readiness_check=self.readiness_check,
            popen_factory=self.popen_factory,
            monotonic_clock=self.monotonic_clock,
            wall_clock=self.wall_clock,
            sleep=self.sleep,
            process_group_signaler=self.process_group_signaler,
            process_liveness_check=self.process_liveness_check,
            process_group_liveness_check=self.process_group_liveness_check,
            available_cpu_sampler=self.available_cpu_sampler,
            affinity_setter=self.affinity_setter,
        )
        # Keep one uninterrupted exclusive lock while controller ownership moves.
        resumed._service_lock_stream = self._service_lock_stream
        resumed._prior_service_lease = self._prior_service_lease
        try:
            if not resumed.resume_from_checkpoint(
                path,
                proc_root=proc_root,
                adopted_factory=lambda pid: process,
            ):
                raise RuntimeError("live vLLM service failed its resume health check")
        except BaseException:
            resumed._service_lock_stream = None
            resumed._process = None
            resumed.state = ServiceState.STOPPED
            self._start_service_heartbeat()
            raise
        # Ownership transfers only after every PID/config/health check succeeds.
        self._service_lock_stream = None
        self._prior_service_lease = None
        self._service_journal_opened = False
        resumed._log_stream = self._log_stream
        self._log_stream = None
        self._process = None
        self._last_service_pid = None
        self._last_process_start_ticks = None
        self._last_process_command_sha256 = None
        self._accounting_session_id = None
        self._started_at = None
        self._started_monotonic = None
        self._allocated_at_start = None
        self._carried_service_seconds = 0.0
        self.state = ServiceState.STOPPED
        return resumed

    def run_warmup(
        self,
        operation: Callable[[], object],
        *,
        event_id: str,
        watchdog_seconds: float,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> object:
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        with self.meter.warmup(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
        ):
            try:
                return operation()
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def run_schema_probe(
        self,
        operation: Callable[[], object],
        *,
        event_id: str,
        watchdog_seconds: float,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> object:
        self.require_ready()
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        with self.meter.schema_probe(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
        ):
            try:
                return operation()
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def generate(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        repair: bool = False,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
        accounting_details: Mapping[str, object] | None = None,
    ) -> GenerationResult:
        self.require_ready()
        if request.model_name != self.configuration.served_model_name:
            raise RuntimeConfigurationError(
                "request served-model alias differs from the controlled service"
            )
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        if repair != (request.decoding.decoding_pass is DecodingPass.REPAIR):
            raise RuntimeConfigurationError("repair event kind differs from decoding pass")
        context = self.meter.repair if repair else self.meter.inference
        details = {"request_id": request.request_id, "request_hash": request.request_hash}
        for name, value in (accounting_details or {}).items():
            if name in details:
                raise RuntimeConfigurationError(
                    "additional accounting details cannot replace request identity"
                )
            details[name] = value
        with context(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
            details=details,
        ):
            try:
                return self.client.generate(request, watchdog_seconds=watchdog_seconds)
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def run_fallback_test(
        self,
        request: GuidedJSONRequest,
        *,
        event_id: str,
        watchdog_seconds: float,
        reserve_call_class: str,
        reserve_reservation_id: str,
        job_id: str | None = None,
        attempt_id: str | None = None,
        remaining_required_seconds: float = 0,
    ) -> GenerationResult:
        """Run one first-pass fallback probe charged to its exact reserve tier."""

        expected_watchdog = {
            "reserve_long": 240,
            "reserve_standard": 150,
            "reserve_short": 90,
        }.get(reserve_call_class)
        if expected_watchdog is None or watchdog_seconds != expected_watchdog:
            raise RuntimeConfigurationError(
                "fallback watchdog must match its registered reserve call class"
            )
        _require_plain_identifier("reserve_reservation_id", reserve_reservation_id)
        if request.decoding.decoding_pass is not DecodingPass.FIRST_PASS:
            raise RuntimeConfigurationError("fallback base probe must use first-pass decoding")
        self.require_ready()
        if request.model_name != self.configuration.served_model_name:
            raise RuntimeConfigurationError(
                "request served-model alias differs from the controlled service"
            )
        if self.configuration.model_candidate != "fallback":
            raise RuntimeConfigurationError("fallback probe requires the pinned fallback model")
        self.require_service_capacity(
            watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
        )
        with self.meter.fallback_test(
            event_id=event_id,
            maximum_seconds=watchdog_seconds,
            remaining_required_seconds=remaining_required_seconds,
            job_id=job_id,
            attempt_id=attempt_id,
            details={
                "request_id": request.request_id,
                "request_hash": request.request_hash,
                "reserve_call_class": reserve_call_class,
                "reserve_reservation_id": reserve_reservation_id,
            },
        ):
            try:
                return self.client.generate(request, watchdog_seconds=watchdog_seconds)
            except RuntimeWatchdogTimeout:
                self.emergency_stop()
                raise

    def require_ready(self) -> None:
        self._raise_service_heartbeat_failure()
        if (
            self.state is not ServiceState.READY
            or self._process is None
            or self._process.poll() is not None
        ):
            raise RuntimeError("vLLM service is not ready")

    @property
    def actual_allocated_service_seconds(self) -> float:
        """Resume-monotonic allocation including time between classified events."""

        ledger_seconds = self.meter.actual_allocated_gpu_seconds
        if self._started_monotonic is None or self._allocated_at_start is None:
            return ledger_seconds
        live_seconds = self._elapsed_service_seconds()
        return max(
            ledger_seconds,
            self._allocated_at_start + live_seconds,
        )

    def _elapsed_service_seconds(self) -> float:
        if self._physically_stopped_seconds is not None:
            return self._physically_stopped_seconds
        if self._started_monotonic is None:
            return self._carried_service_seconds
        elapsed = self.monotonic_clock() - self._started_monotonic
        if elapsed < 0:
            raise RuntimeError("monotonic service clock moved backwards")
        return self._carried_service_seconds + elapsed

    def require_service_capacity(
        self,
        next_maximum_seconds: float,
        *,
        remaining_required_seconds: float = 0,
    ) -> None:
        """Apply scheduled/hard gates to actual service allocation, not token time."""

        if next_maximum_seconds <= 0 or remaining_required_seconds < 0:
            raise RuntimeConfigurationError("GPU capacity values must be positive/nonnegative")
        actual = self.actual_allocated_service_seconds
        shutdown_reserve_seconds = 2 * DEFAULT_SHUTDOWN_SECONDS
        if (
            actual + next_maximum_seconds + shutdown_reserve_seconds
            >= self.meter.hard_limit_seconds
        ):
            raise GpuBudgetExceeded("next request would reach the hard GPU-service limit")
        if (
            actual + next_maximum_seconds + remaining_required_seconds
            > self.meter.scheduled_limit_seconds
        ):
            raise ForecastAdmissionError(
                "next request plus remaining work exceeds scheduled GPU-service allocation"
            )

    def require_hard_stop_margin(self) -> None:
        """Leave enough allocation to terminate and, if needed, kill the service."""

        shutdown_reserve_seconds = 2 * DEFAULT_SHUTDOWN_SECONDS
        if (
            self.actual_allocated_service_seconds + shutdown_reserve_seconds
            >= self.meter.hard_limit_seconds
        ):
            raise GpuBudgetExceeded("vLLM service reached its protected hard-stop margin")

    def _endpoint_live(self, timeout_seconds: float) -> bool:
        probe = getattr(self.client, "endpoint_live", None)
        if not callable(probe):
            raise RuntimeError("vLLM client cannot verify endpoint shutdown")
        return bool(probe(timeout_seconds))

    def _shutdown_status(
        self,
        process: ProcessHandle | None,
        process_group: int | None,
        *,
        deadline: float,
    ) -> tuple[bool, bool, bool]:
        leader_live = process is not None and process.poll() is None
        group_live = process_group is not None and self.process_group_liveness_check(process_group)
        remaining = max(0.0, deadline - self.monotonic_clock())
        endpoint_live = self._endpoint_live(max(0.001, min(0.25, remaining)))
        return leader_live, group_live, endpoint_live

    def _stop_process(
        self,
        shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS,
        *,
        release_lock: bool = True,
        continuing_service: bool = False,
        journal_process_stopped: bool = False,
    ) -> None:
        with self._process_control_lock:
            self._stop_process_locked(
                shutdown_seconds,
                release_lock=release_lock,
                continuing_service=continuing_service,
                journal_process_stopped=journal_process_stopped,
            )

    def _stop_process_locked(
        self,
        shutdown_seconds: float,
        *,
        release_lock: bool,
        continuing_service: bool,
        journal_process_stopped: bool,
    ) -> None:
        if not math.isfinite(shutdown_seconds) or shutdown_seconds <= 0:
            raise RuntimeConfigurationError("shutdown_seconds must be positive and finite")
        if continuing_service and journal_process_stopped:
            raise RuntimeConfigurationError(
                "an intermediate restart cannot terminalize the service journal"
            )
        heartbeat_stop_error: BaseException | None = None
        try:
            self._stop_service_heartbeat()
        except BaseException as exc:
            # Physical termination is mandatory even if a stuck persistence
            # thread cannot yet be joined.  In that case the journal remains
            # unresolved and the exclusive lease stays fail-closed.
            heartbeat_stop_error = exc
        process = self._process
        if process is None and self._service_lock_stream is None:
            self._close_log_stream()
            self.state = (
                ServiceState.STOPPED if heartbeat_stop_error is None else ServiceState.FAILED
            )
            if heartbeat_stop_error is not None:
                raise RuntimeError("vLLM service heartbeat could not be stopped") from (
                    heartbeat_stop_error
                )
            return
        process_group = self._last_service_pid if process is None else process.pid
        physical_absence_verified = False
        verified_at: datetime | None = None
        try:
            if process is not None and process.poll() is None:
                with suppress(ProcessLookupError):
                    self.process_group_signaler(process.pid, signal.SIGTERM)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=shutdown_seconds)

            verification_deadline = self.monotonic_clock() + shutdown_seconds
            leader_live, group_live, endpoint_live = self._shutdown_status(
                process,
                process_group,
                deadline=verification_deadline,
            )
            if (leader_live or group_live) and process_group is not None:
                with suppress(ProcessLookupError):
                    self.process_group_signaler(process_group, signal.SIGKILL)
                if process is not None and process.poll() is None:
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(
                            timeout=max(0.0, verification_deadline - self.monotonic_clock())
                        )
                leader_live, group_live, endpoint_live = self._shutdown_status(
                    process,
                    process_group,
                    deadline=verification_deadline,
                )

            while leader_live or group_live or endpoint_live:
                remaining = verification_deadline - self.monotonic_clock()
                if remaining <= 0:
                    self.state = ServiceState.FAILED
                    live_parts = [
                        name
                        for name, live in (
                            ("leader", leader_live),
                            ("process_group", group_live),
                            ("endpoint", endpoint_live),
                        )
                        if live
                    ]
                    raise RuntimeError(
                        "failed to verify vLLM shutdown; still live: " + ", ".join(live_parts)
                    )
                self.sleep(min(0.05, remaining))
                leader_live, group_live, endpoint_live = self._shutdown_status(
                    process,
                    process_group,
                    deadline=verification_deadline,
                )

            physical_absence_verified = True
            verified_at = self.wall_clock()
            if verified_at.tzinfo is None or verified_at.utcoffset() is None:
                raise RuntimeConfigurationError("service stop clock must be timezone-aware")
            if (
                not continuing_service
                and heartbeat_stop_error is None
                and self._physically_stopped_at is None
            ):
                service_seconds = self._elapsed_service_seconds()
                if service_seconds < 0:
                    raise RuntimeError("monotonic service clock moved backwards")
                self._physically_stopped_at = verified_at
                self._physically_stopped_seconds = service_seconds

            if self._service_lock_stream is not None:
                prior_state = (
                    None
                    if self._prior_service_lease is None
                    else self._prior_service_lease.get("lease_state")
                )
                current_session_needs_accounting = (
                    self._started_at is not None
                    or self._started_monotonic is not None
                    or self._session_id is not None
                )
                unresolved_prior_session = prior_state not in {
                    None,
                    "stopped_verified",
                }
                if current_session_needs_accounting:
                    self._write_service_lock_metadata(
                        lease_state="accounting_pending",
                        service_pid=process_group,
                        ended_at=(
                            verified_at if continuing_service else self._physically_stopped_at
                        ),
                    )
                elif not unresolved_prior_session:
                    self._write_service_lock_metadata(
                        lease_state="stopped_verified",
                        service_pid=process_group,
                        ended_at=verified_at,
                    )

            if not continuing_service and journal_process_stopped and heartbeat_stop_error is None:
                self._journal_verified_process_stop()
            self._process = None
            self.state = (
                ServiceState.STOPPED if heartbeat_stop_error is None else ServiceState.FAILED
            )
            if heartbeat_stop_error is not None:
                raise RuntimeError("vLLM service heartbeat could not be stopped") from (
                    heartbeat_stop_error
                )
            if release_lock:
                self._release_service_lock()
        except BaseException:
            self.state = ServiceState.FAILED
            if self._service_lock_stream is not None:
                with suppress(Exception):
                    self._write_service_lock_metadata(
                        lease_state=(
                            "accounting_pending"
                            if physical_absence_verified and self._session_id is not None
                            else "shutdown_unverified"
                        ),
                        service_pid=process_group,
                        ended_at=(
                            self._physically_stopped_at
                            if self._physically_stopped_at is not None
                            else verified_at
                        ),
                    )
            raise
        finally:
            self._close_log_stream()

    def emergency_stop(self) -> None:
        """Stop the controlled process tree while retaining uptime for final accounting."""

        self._stop_process(
            shutdown_seconds=min(5.0, DEFAULT_SHUTDOWN_SECONDS),
            release_lock=False,
            journal_process_stopped=False,
        )

    def shutdown(
        self,
        *,
        shutdown_seconds: float = DEFAULT_SHUTDOWN_SECONDS,
    ) -> ServiceUptime | None:
        if self._started_at is None or self._started_monotonic is None or self._session_id is None:
            self._stop_process(shutdown_seconds)
            return None
        terminalized = False
        try:
            self._stop_process(
                shutdown_seconds,
                release_lock=False,
                journal_process_stopped=True,
            )
            ended_at = self._physically_stopped_at
            service_seconds = self._physically_stopped_seconds
            if ended_at is None or service_seconds is None:
                raise RuntimeError("vLLM service has no verified physical stop")
            if self._allocated_at_start is None:
                raise RuntimeError("live vLLM service has no durable allocation baseline")
            allocated_start = self._allocated_at_start
            allocated_seconds = self.meter.actual_allocated_gpu_seconds - allocated_start
            if allocated_seconds < -1e-9:
                raise RuntimeError("GPU ledger predates the service allocation baseline")
            allocated_seconds = max(0.0, allocated_seconds)
            uptime = ServiceUptime(
                session_id=self._session_id,
                started_at=self._started_at,
                ended_at=ended_at,
                service_seconds=service_seconds,
                allocated_event_seconds=allocated_seconds,
            )
            accounting_session_id = self._accounting_session_id
            if accounting_session_id is None:
                raise RuntimeError("live vLLM service has no accounting session identity")
            if not self._service_journal_opened:
                raise RuntimeError(
                    "vLLM service cannot reconcile without its durable service journal"
                )
            try:
                self.meter.reconcile_service_session(
                    service_session_id=accounting_session_id,
                    session_id=self._session_id,
                    service_seconds=uptime.allocated_service_seconds,
                    classified_event_seconds=allocated_seconds,
                    started_at=self._started_at,
                    ended_at=ended_at,
                    details={
                        "accounting_method": "service_total_minus_classified_events",
                        "configuration_hash": self.configuration.configuration_hash,
                    },
                )
            except BaseException:
                latest = self.meter.ledger.latest_gpu_service_journal(accounting_session_id)
                if (
                    latest is not None
                    and latest.state
                    in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
                    and self.meter.ledger.get_gpu_service_session(accounting_session_id) is not None
                ):
                    self._write_service_lock_metadata(
                        lease_state="stopped_verified",
                        service_pid=self._last_service_pid,
                        ended_at=ended_at,
                    )
                    terminalized = True
                raise
            self._write_service_lock_metadata(
                lease_state="stopped_verified",
                service_pid=self._last_service_pid,
                ended_at=ended_at,
            )
            terminalized = True
            heartbeat_failure = self._service_heartbeat_failure
            if heartbeat_failure is not None:
                raise RuntimeError("vLLM service heartbeat failed") from heartbeat_failure
            return uptime
        finally:
            if terminalized:
                self._session_id = None
                self._started_at = None
                self._started_monotonic = None
                self._allocated_at_start = None
                self._accounting_session_id = None
                self._last_service_pid = None
                self._last_process_start_ticks = None
                self._last_process_command_sha256 = None
                self._carried_service_seconds = 0.0
                self._service_journal_opened = False
                self._service_heartbeat_failure = None
                self._physically_stopped_at = None
                self._physically_stopped_seconds = None
                self._process_stop_journaled = False
                self.state = ServiceState.STOPPED
                self._release_service_lock()
            else:
                self.state = ServiceState.FAILED

    def recover_stale_service_lease(self) -> GpuServiceSession | None:
        """Recover one stale allocation only after proving all service absence."""

        if (
            self.state is not ServiceState.STOPPED
            or self._process is not None
            or self._service_lock_stream is not None
            or self._started_at is not None
            or self._accounting_session_id is not None
        ):
            raise RuntimeError("stale service recovery requires a fresh stopped controller")
        self._last_recovered_process_identity = None
        self._acquire_service_lock()
        terminalized = False
        try:
            lease = self._prior_service_lease
            if lease is None:
                return None
            lease_state = lease.get("lease_state")
            if lease_state not in {
                "starting",
                "live",
                "controller_restart_handoff",
                "accounting_pending",
                "shutdown_unverified",
                "stopped_verified",
            }:
                raise RuntimeConfigurationError("stale vLLM service lease state is invalid")
            service_session_id = lease.get("accounting_session_id")
            session_id = lease.get("session_id")
            service_pid = lease.get("service_pid")
            process_start_ticks = lease.get("process_start_ticks")
            process_command_sha256 = lease.get("process_command_sha256")
            started_at_text = lease.get("service_started_at")
            baseline = lease.get("ledger_allocated_seconds_before_session")
            if (
                not isinstance(service_session_id, str)
                or not service_session_id
                or not isinstance(session_id, str)
                or not session_id
                or isinstance(service_pid, bool)
                or not isinstance(service_pid, int)
                or service_pid <= 0
                or not isinstance(started_at_text, str)
                or isinstance(baseline, bool)
                or not isinstance(baseline, int | float)
                or not math.isfinite(float(baseline))
                or baseline < 0
            ):
                raise RuntimeConfigurationError(
                    "stale vLLM lease lacks an exact recoverable process identity"
                )
            latest = self.meter.ledger.latest_gpu_service_journal(service_session_id)
            if latest is None:
                raise RuntimeConfigurationError("stale vLLM lease has no durable service journal")
            lease_started_at = _parse_aware_datetime(
                "stale service lease start",
                started_at_text,
            )
            recovered_process_identity: RecoveredServiceProcessIdentity | None = None
            if process_start_ticks is not None or process_command_sha256 is not None:
                if (
                    isinstance(process_start_ticks, bool)
                    or not isinstance(process_start_ticks, int)
                    or process_start_ticks <= 0
                    or not isinstance(process_command_sha256, str)
                    or len(process_command_sha256) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in process_command_sha256
                    )
                    or process_command_sha256
                    != canonical_sha256(list(self.configuration.command()))
                ):
                    raise RuntimeConfigurationError(
                        "stale vLLM lease process identity is invalid"
                    )
                recovered_process_identity = RecoveredServiceProcessIdentity(
                    configuration_hash=self.configuration.configuration_hash,
                    session_id=session_id,
                    accounting_session_id=service_session_id,
                    pid=service_pid,
                    process_start_ticks=process_start_ticks,
                    process_command_sha256=process_command_sha256,
                    service_started_at=lease_started_at,
                )
            journal_started_at = _parse_aware_datetime(
                "stale service journal start",
                latest.service_started_at,
            )
            if (
                lease.get("configuration_hash") != self.configuration.configuration_hash
                or latest.configuration_hash != self.configuration.configuration_hash
                or latest.session_id != session_id
                or lease_started_at != journal_started_at
                or abs(
                    latest.ledger_allocated_microseconds_before_session / 1_000_000
                    - float(baseline)
                )
                > 1e-6
            ):
                raise RuntimeConfigurationError(
                    "stale vLLM lease and service journal identities differ"
                )
            if lease_state == "stopped_verified":
                record = self.meter.ledger.get_gpu_service_session(service_session_id)
                ended_at_text = lease.get("service_ended_at")
                if (
                    latest.state
                    not in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
                    or record is None
                    or not isinstance(ended_at_text, str)
                    or record.service_session_id != service_session_id
                    or record.session_id != session_id
                    or _parse_aware_datetime("terminal service start", record.started_at)
                    != lease_started_at
                    or _parse_aware_datetime("terminal service end", record.ended_at)
                    != _parse_aware_datetime("terminal lease end", ended_at_text)
                ):
                    raise RuntimeConfigurationError(
                        "terminal vLLM lease has no matching service accounting record"
                    )
                self._last_recovered_process_identity = recovered_process_identity
                terminalized = True
                return record
            try:
                pid_live = self.process_liveness_check(service_pid)
                process_group_live = self.process_group_liveness_check(service_pid)
                endpoint_live = self._endpoint_live(0.25)
            except BaseException as exc:
                raise RuntimeConfigurationError(
                    "cannot prove stale vLLM process and endpoint absence"
                ) from exc
            if pid_live or process_group_live or endpoint_live:
                live_parts = [
                    name
                    for name, live in (
                        ("pid", pid_live),
                        ("process_group", process_group_live),
                        ("endpoint", endpoint_live),
                    )
                    if live
                ]
                raise RuntimeConfigurationError(
                    "stale vLLM service is still live: " + ", ".join(live_parts)
                )
            recovered_at = self.wall_clock()
            if recovered_at.tzinfo is None or recovered_at.utcoffset() is None:
                raise RuntimeConfigurationError("service recovery clock must be timezone-aware")

            def finalize_lease(record: GpuServiceSession) -> None:
                nonlocal terminalized
                self._session_id = record.session_id
                self._accounting_session_id = record.service_session_id
                self._started_at = _parse_aware_datetime(
                    "recovered service start",
                    record.started_at,
                )
                self._allocated_at_start = float(baseline)
                self._carried_service_seconds = record.service_seconds
                self._physically_stopped_at = _parse_aware_datetime(
                    "recovered service end",
                    record.ended_at,
                )
                self._physically_stopped_seconds = record.service_seconds
                self._last_service_pid = service_pid
                self._last_process_start_ticks = process_start_ticks
                self._last_process_command_sha256 = process_command_sha256
                self._write_service_lock_metadata(
                    lease_state="stopped_verified",
                    service_pid=service_pid,
                    ended_at=self._physically_stopped_at,
                )
                self._last_recovered_process_identity = recovered_process_identity
                terminalized = True

            try:
                record = self.meter.recover_service_journal(
                    service_session_id=service_session_id,
                    recovered_at=recovered_at,
                    details={
                        "pid_absence_verified": True,
                        "process_group_absence_verified": True,
                        "endpoint_absence_verified": True,
                    },
                )
            except BaseException:
                latest = self.meter.ledger.latest_gpu_service_journal(service_session_id)
                record = self.meter.ledger.get_gpu_service_session(service_session_id)
                if (
                    latest is not None
                    and latest.state
                    in {GpuServiceJournalState.CLOSED, GpuServiceJournalState.RECOVERED}
                    and record is not None
                ):
                    finalize_lease(record)
                raise
            finalize_lease(record)
            return record
        finally:
            if terminalized:
                self.state = ServiceState.STOPPED
            self._session_id = None
            self._accounting_session_id = None
            self._started_at = None
            self._started_monotonic = None
            self._allocated_at_start = None
            self._carried_service_seconds = 0.0
            self._physically_stopped_at = None
            self._physically_stopped_seconds = None
            self._process_stop_journaled = False
            self._service_journal_opened = False
            self._service_heartbeat_failure = None
            self._release_service_lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.shutdown()
        return False


def _atomic_write_private_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomic operational state; unlike public manifests this may contain a PID."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
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


def public_runtime_manifest(
    *,
    launcher: VLLMLaunchConfiguration,
    tokenizer: TokenizerManifest,
    runtime_stack: RuntimeStackManifest | None = None,
    gpu_hardware: GPUHardwareIdentity | None = None,
    resource_samples: Sequence[ResourceSnapshot] = (),
    uptime: ServiceUptime | None = None,
    ledger: Ledger | None = None,
) -> dict[str, object]:
    """Compose hashes and accounting only; prompts, outputs, and paths are excluded."""

    if (
        tokenizer.repository != launcher.repository
        or tokenizer.revision != launcher.revision
        or tokenizer.tokenizer_revision != launcher.revision
    ):
        raise RuntimeConfigurationError(
            "runtime tokenizer identity differs from the selected model candidate"
        )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "launcher": launcher.public_manifest(),
        "tokenizer": tokenizer.public_manifest(),
        "runtime_stack": None if runtime_stack is None else runtime_stack.public_manifest(),
        "gpu_hardware": None if gpu_hardware is None else gpu_hardware.public_manifest(),
        "resource_samples": [sample.public_manifest() for sample in resource_samples],
        "service_uptime": None if uptime is None else uptime.public_manifest(),
    }
    if ledger is not None:
        summary = ledger.gpu_summary()
        payload["gpu_accounting"] = {
            "event_count": summary.event_count,
            "service_session_count": summary.service_session_count,
            "total_allocated_microseconds": summary.total_allocated_microseconds,
            "by_kind_microseconds": {
                kind.value: microseconds for kind, microseconds in summary.by_kind_microseconds
            },
        }
    return {**payload, "manifest_sha256": canonical_sha256(payload)}


__all__ = [
    "DEFAULT_SERVICE_START_WATCHDOG_SECONDS",
    "DEFAULT_SHUTDOWN_SECONDS",
    "FALLBACK_MODEL_REPOSITORY",
    "FALLBACK_MODEL_REVISION",
    "FALLBACK_SERVED_MODEL_NAME",
    "GPU_MEMORY_UTILIZATION",
    "GUIDED_DECODING_BACKEND",
    "MAXIMUM_CPU_WORKERS",
    "MAXIMUM_MODEL_LENGTH",
    "MODEL_CANDIDATES",
    "PINNED_MODEL_REPOSITORY",
    "PINNED_MODEL_REVISION",
    "PINNED_RUNTIME_VERSION",
    "PROC_ROOT",
    "SCHEMA_VERSION",
    "SERVED_MODEL_NAME",
    "ChatMessage",
    "GPUHardwareIdentity",
    "GenerationResult",
    "GuidedJSONRequest",
    "ProcessTreeUsage",
    "ResourceSampler",
    "ResourceSnapshot",
    "ResourceWatchdog",
    "RuntimeConfigurationError",
    "RuntimeResourceLimitExceeded",
    "RuntimeStackManifest",
    "RuntimeTransportError",
    "RuntimeWatchdogTimeout",
    "ServiceState",
    "ServiceUptime",
    "TokenizerManifest",
    "VLLMGuidedJSONClient",
    "VLLMLaunchConfiguration",
    "VLLMService",
    "atomic_write_public_json",
    "capture_gpu_hardware_identity",
    "capture_runtime_stack",
    "capture_tokenizer_manifest",
    "public_runtime_manifest",
    "sample_gpu_vram",
    "sample_process_tree",
    "sample_system_available_ram",
]
