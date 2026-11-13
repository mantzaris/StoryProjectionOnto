#!/usr/bin/env python3
"""Capture a safe, exact local environment and hardware manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def nvidia_manifest() -> dict[str, object]:
    fields = "name,memory.total,driver_version,pci.bus_id"
    row = command_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if len(row) != 1:
        raise RuntimeError("the controlled environment must expose exactly one GPU")
    name, memory_mib, driver, bus_id = (item.strip() for item in row[0].split(","))
    return {
        "count": 1,
        "name": name,
        "memory_mib": int(memory_mib),
        "driver_version": driver,
        "pci_bus_id": bus_id,
    }


def capture(project_root: Path) -> tuple[dict[str, object], str]:
    import psutil
    import torch
    import transformers
    import vllm

    resolved_root = project_root.resolve(strict=True)
    lock_path = resolved_root / "uv.lock"
    if not lock_path.is_file():
        raise FileNotFoundError("uv.lock is required before environment capture")
    process = psutil.Process()
    affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
    virtual_memory = psutil.virtual_memory()
    root_disk = shutil.disk_usage("/")
    project_disk = shutil.disk_usage(resolved_root)
    freeze = command_output([sys.executable, "-m", "pip", "freeze", "--all"]) + "\n"
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_prefix": sys.prefix,
            "python_base_prefix": sys.base_prefix,
            "logical_cpu_count": os.cpu_count(),
            "available_cpu_affinity_count": len(affinity) if affinity else None,
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_total_bytes": virtual_memory.total,
        },
        "gpu": nvidia_manifest(),
        "runtime": {
            "uv": command_output(["uv", "--version"]),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_path": str(Path(torch.__file__).resolve()),
            "torch_cuda_available": torch.cuda.is_available(),
            "vllm": vllm.__version__,
            "vllm_path": str(Path(vllm.__file__).resolve()),
            "transformers": transformers.__version__,
            "pydantic": package_version("pydantic"),
        },
        "storage": {
            "container_total_bytes": root_disk.total,
            "container_free_bytes": root_disk.free,
            "project_filesystem_total_bytes": project_disk.total,
            "project_filesystem_free_bytes": project_disk.free,
            "project_controlled_roots": [str(resolved_root)],
            "immutable_container_paths": [
                "/usr/local/lib/python3.12/dist-packages",
                "/opt/nvidia",
            ],
        },
        "locks": {
            "uv_lock_sha256": sha256_file(lock_path),
            "environment_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        },
        "study_limits": {
            "cpu_workers": 8,
            "process_ram_bytes": 25_000_000_000,
            "peak_vram_bytes": 23 * 1024**3,
            "cpu_weight_offload": False,
        },
    }
    return manifest, freeze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    options = parse_args()
    manifest, freeze = capture(options.project_root)
    output = options.output_dir.resolve()
    atomic_write(
        output / "environment.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(output / "environment-freeze.txt", freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
