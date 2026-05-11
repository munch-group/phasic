"""Conflict-raising validate() tests for the new PhasicConfig.

Under the refactored config API, the rules are:
  - Shell-set env vars and configure() arguments must agree.
    Disagreement raises PTDConfigError. No silent override.
  - Conflicting compute requirements (compute='jax-cpu' without
    JAX, compute='jax-gpu' without GPU, etc.) raise.
  - parallel_elimination requires cpu_threads >= 2.
  - high_precision_mode='always' requires MPFR.
  - effective() round-trips through os.environ.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import phasic
from phasic import PhasicConfig
from phasic.config import _phasic_assigned_env, reset_config
from phasic.exceptions import PTDConfigError


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_config()
    _phasic_assigned_env.clear()
    for v in ('OMP_NUM_THREADS', 'PHASIC_HIERAR_ELIMINATION',
              'PHASIC_DISABLE_CACHE', 'PHASIC_CACHE_DIR',
              'PHASIC_MIN_SCC_SIZE_TO_CACHE',
              'PHASIC_MAX_PARALLEL_SCCS', 'PHASIC_FORCE_MPFR',
              'PHASIC_MPFR_BITS', 'PHASIC_CONDITION_THRESHOLD',
              'PHASIC_DISABLE_CONDITION_WARNINGS'):
        monkeypatch.delenv(v, raising=False)
    yield
    reset_config()
    _phasic_assigned_env.clear()


# ---------------------------------------------------------------
# env-var <-> configure() conflicts
# ---------------------------------------------------------------


def test_shell_omp_conflicts_with_configure_cpu_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(PTDConfigError, match="OMP_NUM_THREADS"):
        phasic.configure(cpu_threads=8)


def test_shell_cache_dir_conflicts_with_configure(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "shell"))
    with pytest.raises(PTDConfigError, match="PHASIC_CACHE_DIR"):
        phasic.configure(cache_dir=str(tmp_path / "python"))


def test_shell_max_parallel_conflicts_with_configure(monkeypatch):
    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", "2")
    with pytest.raises(PTDConfigError, match="PHASIC_MAX_PARALLEL_SCCS"):
        phasic.configure(parallel_elimination_max_concurrent=8)


def test_agreeing_env_and_configure_is_ok(monkeypatch):
    """When env and configure() agree exactly, no raise."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    phasic.configure(cpu_threads=4)
    assert os.environ["OMP_NUM_THREADS"] == "4"


# ---------------------------------------------------------------
# compute consistency
# ---------------------------------------------------------------


def test_compute_gpu_raises_when_no_gpu():
    """compute='jax-gpu' on a CPU-only system raises."""
    # On hardware with a GPU this test trivially passes (the
    # error path doesn't trigger). Skip in that case.
    from phasic.config import _get_available_platforms
    if 'gpu' in _get_available_platforms():
        pytest.skip("system has a GPU; conflict path not exercised")
    with pytest.raises(PTDConfigError, match="GPU"):
        phasic.configure(compute='jax-gpu')


def test_compute_unknown_value_raises():
    with pytest.raises(PTDConfigError):
        phasic.configure(compute='not-a-real-mode')


# ---------------------------------------------------------------
# parallel_elimination requires cpu_threads >= 2
# ---------------------------------------------------------------


def test_parallel_elimination_with_cpu_threads_1_raises():
    """parallel_elimination=True is meaningless with one thread."""
    with pytest.raises(PTDConfigError, match="parallel_elimination"):
        phasic.configure(parallel_elimination=True, cpu_threads=1)


# ---------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------


def test_high_precision_bits_below_53_raises():
    with pytest.raises(PTDConfigError, match="high_precision_bits"):
        phasic.configure(high_precision_bits=32)


def test_ill_condition_threshold_too_low_raises():
    with pytest.raises(PTDConfigError, match="ill_condition_threshold"):
        phasic.configure(ill_condition_threshold=0.5)


def test_negative_min_subgraph_raises():
    with pytest.raises(PTDConfigError, match="parallel_elimination_min_subgraph"):
        phasic.configure(parallel_elimination_min_subgraph=-1)


def test_zero_max_concurrent_raises():
    with pytest.raises(PTDConfigError, match="parallel_elimination_max_concurrent"):
        phasic.configure(parallel_elimination_max_concurrent=0)


# ---------------------------------------------------------------
# effective() round-trip
# ---------------------------------------------------------------


def test_effective_returns_three_sections():
    phasic.configure(parallel_elimination=True, cpu_threads=4)
    eff = phasic.get_config().effective()
    assert 'fields' in eff
    assert 'environment' in eff
    assert 'derived' in eff


def test_effective_environment_reflects_writes(tmp_path):
    phasic.configure(cache_dir=str(tmp_path), parallel_elimination=True)
    eff = phasic.get_config().effective()
    assert eff['environment']['PHASIC_CACHE_DIR'] == str(tmp_path)
    assert eff['environment']['PHASIC_HIERAR_ELIMINATION'] == '1'


def test_effective_derived_reports_compute_resolved():
    phasic.configure(compute='cpu')
    eff = phasic.get_config().effective()
    assert eff['derived']['compute_resolved'] == 'cpu'
    reset_config()
    _phasic_assigned_env.clear()
    phasic.configure(compute='jax-cpu')
    eff = phasic.get_config().effective()
    assert eff['derived']['compute_resolved'] == 'jax-cpu'


def test_effective_derived_reports_threads_source():
    phasic.configure(cpu_threads=7)
    eff = phasic.get_config().effective()
    assert eff['derived']['cpu_threads_source'] == 'configure()'


def test_repr_contains_effective():
    phasic.configure(parallel_elimination=True)
    s = repr(phasic.get_config())
    assert 'PhasicConfig' in s
    assert 'parallel_elimination' in s
