"""SLURM-WP-3: standalone scc_worker CLI.

Verifies the worker entry point handles the orchestrator-worker
contract correctly:

  - main() returns 0 on a valid work unit
  - The cache file appears in PHASIC_CACHE_DIR after the worker
    runs (the worker's only side effect)
  - main() returns non-zero on missing/malformed input
  - run_worker handles each failure mode with a distinct status
  - Subprocess invocation as ``python -m phasic.scc_worker``
    works end-to-end

Most tests are in-process to keep the suite fast; one
subprocess test exercises the actual CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import phasic.cache as cache
from phasic import distributed_scc, scc_worker
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


def _make_work_unit(tmp_path: Path, scc_idx: int = 1) -> Path:
    """Build a single work unit for toy_b SCC scc_idx."""
    g = build_toy_b()
    scc = g.scc_decomposition().scc_at(scc_idx)
    wu_path = tmp_path / f"wu_scc_{scc_idx}.json"
    distributed_scc.write_work_unit(scc, str(wu_path))
    return wu_path


def test_run_worker_returns_zero_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    wu = _make_work_unit(tmp_path)
    rc = scc_worker.run_worker(str(wu))
    assert rc == 0


def test_run_worker_writes_cache_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    wu = _make_work_unit(tmp_path)
    scc_worker.run_worker(str(wu))

    files = list((cache_dir / "parameterized_reward_compute").glob("scc_*.bin"))
    assert len(files) == 1


def test_run_worker_missing_file(tmp_path):
    rc = scc_worker.run_worker(str(tmp_path / "does_not_exist.json"))
    assert rc == 1  # load failure


def test_run_worker_malformed_json(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")

    rc = scc_worker.run_worker(str(bad))
    assert rc == 1


def test_run_worker_valid_json_but_wrong_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    bad = tmp_path / "wrong.json"
    bad.write_text('{"foo": "bar"}')

    rc = scc_worker.run_worker(str(bad))
    assert rc == 2  # deserialise failure


def test_main_with_args(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    wu = _make_work_unit(tmp_path)
    rc = scc_worker.main([str(wu)])
    assert rc == 0


def test_main_verbose_flag_does_not_break(tmp_path, monkeypatch):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    wu = _make_work_unit(tmp_path)
    rc = scc_worker.main(["-v", str(wu)])
    assert rc == 0


def test_subprocess_invocation_succeeds(tmp_path, monkeypatch):
    """End-to-end: invoke as a subprocess via
    ``python -m phasic.scc_worker``. This mirrors what SLURM
    sbatch will actually do."""
    cache_dir = tmp_path / "cache"
    wu = _make_work_unit(tmp_path)

    env = os.environ.copy()
    env["PHASIC_CACHE_DIR"] = str(cache_dir)
    env.pop("PHASIC_DISABLE_CACHE", None)

    result = subprocess.run(
            [sys.executable, "-m", "phasic.scc_worker", str(wu)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60)
    assert result.returncode == 0, (
            f"stdout: {result.stdout}\nstderr: {result.stderr}")

    files = list((cache_dir / "parameterized_reward_compute").glob("scc_*.bin"))
    assert len(files) == 1


def test_worker_then_orchestrator_full_pipeline(tmp_path, monkeypatch):
    """End-to-end: run a worker per SCC, then have the
    orchestrator do hierarchical compose. All compose calls
    should hit the worker-populated cache."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    # Phase 1: orchestrator writes work units.
    g = build_toy_b()
    scc_decomp = g.scc_decomposition()
    n = len(scc_decomp)
    work_units = []
    for i in range(n):
        scc = scc_decomp.scc_at(i)
        wu_path = tmp_path / f"wu_{i}.json"
        distributed_scc.write_work_unit(scc, str(wu_path))
        work_units.append(str(wu_path))

    # Phase 2: workers (in-process simulation, but using the
    # CLI entry point).
    cache.reset_scc_compose_stats()
    for wu in work_units:
        rc = scc_worker.run_worker(wu)
        assert rc == 0

    worker_stats = cache.scc_compose_stats()
    assert worker_stats["cache_misses"] == n

    # Phase 3: orchestrator runs hierarchical compose. All
    # SCCs should hit.
    cache.reset_scc_compose_stats()
    g_run = build_toy_b()
    g_run.update_weights(THETA)
    result = g_run.expected_waiting_time()
    final_stats = cache.scc_compose_stats()
    assert final_stats["cache_hits"] == n
    assert final_stats["cache_misses"] == 0
    assert result[0] == pytest.approx(1.88, rel=1e-12)
