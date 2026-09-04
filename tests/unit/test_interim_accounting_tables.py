from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path("scripts/build_interim_accounting_tables.py")
    spec = importlib.util.spec_from_file_location("build_interim_accounting_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_interim_accounting_tables_regenerate_exactly(tmp_path: Path) -> None:
    module = _load_script()
    module.build_tables(
        Path("artifacts/public/results/phase1_gpu_acceptance_v1_failed.json"),
        Path("artifacts/public/results/phase1_gpu_acceptance_v2_failed.json"),
        tmp_path,
    )
    for name in ("resource_accounting.csv", "failure_accounting.csv"):
        assert (tmp_path / name).read_bytes() == (Path("reports/tables") / name).read_bytes()
