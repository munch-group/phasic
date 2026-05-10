"""Tests for SCC + cache controls exposed via phasic.configure().

Verifies that:
  - configure() with the new fields writes the corresponding
    PHASIC_* env vars so the C-side readers pick them up.
  - configure() overrides pre-existing env vars (last setter wins).
  - Leaving a field as None preserves the existing env var (so
    SLURM-style shell overrides survive a configure() call that
    only changes other fields).
  - get_config() reads env vars on first construction so
    configuration is consistent whether set via shell or
    Python.
  - Validation errors for invalid values.
  - End-to-end: set via configure() and observe the C-side
    behaviour through cache_compose_stats.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import phasic
import phasic.cache as cache
from phasic.config import PTDAlgorithmsConfig, reset_config
from phasic.exceptions import PTDConfigError
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture(autouse=True)
def _reset_config_and_env(monkeypatch):
    """Reset the global config and clear PHASIC_* env vars
    before each test."""
    reset_config()
    monkeypatch.delenv("PHASIC_HIERAR_ELIMINATION", raising=False)
    monkeypatch.delenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", raising=False)
    monkeypatch.delenv("PHASIC_MAX_PARALLEL_SCCS", raising=False)
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("PHASIC_CACHE_DIR", raising=False)
    yield
    reset_config()


# ---------------------------------------------------------------
# configure() writes env vars
# ---------------------------------------------------------------


def test_configure_hierar_elimination_sets_env():
    phasic.configure(hierar_elimination=True)
    assert os.environ.get("PHASIC_HIERAR_ELIMINATION") == "1"


def test_configure_hierar_elimination_false_clears_env(monkeypatch):
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    phasic.configure(hierar_elimination=False)
    assert os.environ.get("PHASIC_HIERAR_ELIMINATION") is None


def test_configure_min_scc_size_sets_env():
    phasic.configure(min_scc_size_to_cache=8)
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "8"


def test_configure_min_scc_size_zero():
    phasic.configure(min_scc_size_to_cache=0)
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "0"


def test_configure_max_parallel_sccs_sets_env():
    phasic.configure(max_parallel_sccs=4)
    assert os.environ.get("PHASIC_MAX_PARALLEL_SCCS") == "4"


def test_configure_disable_cache_sets_env():
    phasic.configure(disable_cache=True)
    assert os.environ.get("PHASIC_DISABLE_CACHE") == "1"


def test_configure_cache_dir_sets_env(tmp_path):
    phasic.configure(cache_dir=str(tmp_path))
    assert os.environ.get("PHASIC_CACHE_DIR") == str(tmp_path)


# ---------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------


def test_configure_overrides_existing_env(monkeypatch):
    """configure() with a concrete value wins over a pre-existing
    env var (last setter wins)."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "100")
    phasic.configure(min_scc_size_to_cache=4)
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "4"


def test_configure_none_field_preserves_env(monkeypatch):
    """A field left at None does NOT overwrite a pre-existing env
    var. Useful so SLURM job scripts can set PHASIC_CACHE_DIR
    and a Python configure() call can change other knobs without
    blowing it away."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "16")
    phasic.configure(hierar_elimination=True)  # don't touch min_scc
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "16"


# ---------------------------------------------------------------
# get_config() reads env vars
# ---------------------------------------------------------------


def test_get_config_reads_hierar_elimination_from_env(monkeypatch):
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cfg = phasic.get_config()
    assert cfg.hierar_elimination is True


def test_get_config_reads_min_scc_size_from_env(monkeypatch):
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "8")
    cfg = phasic.get_config()
    assert cfg.min_scc_size_to_cache == 8


def test_get_config_reads_max_parallel_from_env(monkeypatch):
    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", "4")
    cfg = phasic.get_config()
    assert cfg.max_parallel_sccs == 4


def test_get_config_reads_cache_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    cfg = phasic.get_config()
    assert cfg.cache_dir == str(tmp_path)


def test_get_config_handles_malformed_env(monkeypatch):
    """Malformed numeric env vars are silently ignored — the C
    side has its own fallback to sane defaults."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "not_a_number")
    cfg = phasic.get_config()
    assert cfg.min_scc_size_to_cache is None


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------


def test_configure_rejects_negative_min_scc_size():
    with pytest.raises(PTDConfigError, match="min_scc_size_to_cache"):
        phasic.configure(min_scc_size_to_cache=-1)


def test_configure_rejects_zero_max_parallel():
    with pytest.raises(PTDConfigError, match="max_parallel_sccs"):
        phasic.configure(max_parallel_sccs=0)


def test_configure_unknown_field_raises():
    with pytest.raises(PTDConfigError, match="Unknown configuration option"):
        phasic.configure(this_field_does_not_exist=True)


# ---------------------------------------------------------------
# End-to-end: configure() actually changes C-side behaviour
# ---------------------------------------------------------------


def test_configure_threshold_affects_cache_bypass(tmp_path):
    """Setting min_scc_size_to_cache via configure() actually
    changes which SCCs bypass the cache."""
    phasic.configure(
            hierar_elimination=True,
            cache_dir=str(tmp_path),
            min_scc_size_to_cache=1000,  # bypass everything
    )
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    assert s["cache_bypassed"] == 5
    assert s["cache_misses"] == 0


def test_configure_zero_threshold_caches_everything(tmp_path):
    """configure(min_scc_size_to_cache=0) caches every SCC."""
    phasic.configure(
            hierar_elimination=True,
            cache_dir=str(tmp_path),
            min_scc_size_to_cache=0,
    )
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    assert s["cache_bypassed"] == 0
    assert s["cache_misses"] == 5


def test_configure_disable_cache_end_to_end(tmp_path):
    """configure(disable_cache=True) bypasses load and save."""
    phasic.configure(
            hierar_elimination=True,
            cache_dir=str(tmp_path),
            disable_cache=True,
            min_scc_size_to_cache=0,
    )
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    # No cache hits/misses recorded when cache is disabled.
    assert s["cache_hits"] == 0
    assert s["cache_misses"] == 0


def test_configure_correctness_unchanged(tmp_path):
    """Whatever knob settings, the numerical answer is the same."""
    base_result = None
    for cfg_kwargs in [
        {"hierar_elimination": False},
        {"hierar_elimination": True, "cache_dir": str(tmp_path / "a"),
         "min_scc_size_to_cache": 0},
        {"hierar_elimination": True, "cache_dir": str(tmp_path / "b"),
         "min_scc_size_to_cache": 1000},
        {"hierar_elimination": True, "cache_dir": str(tmp_path / "c"),
         "max_parallel_sccs": 1, "min_scc_size_to_cache": 0},
    ]:
        phasic.configure(**cfg_kwargs)
        g = build_toy_b()
        g.update_weights(THETA)
        result = np.asarray(g.expected_waiting_time())
        if base_result is None:
            base_result = result
        else:
            np.testing.assert_allclose(result, base_result, rtol=1e-12)
