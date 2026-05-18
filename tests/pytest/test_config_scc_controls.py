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
from phasic.config import PhasicConfig, reset_config
from phasic.exceptions import PTDConfigError
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


@pytest.fixture(autouse=True)
def _reset_config_and_env(monkeypatch):
    """Reset the global config and clear PHASIC_* env vars before
    each test (and after, since ``phasic.configure`` writes to
    ``os.environ`` directly — bypassing pytest's monkeypatch
    tracking — so leaked env vars would otherwise contaminate
    subsequent tests / modules)."""
    from phasic.config import _phasic_assigned_env

    _PHASIC_ENV_VARS = (
        "PHASIC_HIERAR_ELIMINATION",
        "PHASIC_MIN_SCC_SIZE_TO_CACHE",
        "PHASIC_MAX_PARALLEL_SCCS",
        "PHASIC_REWARD_COMPUTE_CACHE",
        "PHASIC_DISABLE_GRAPH_CACHE",
        "PHASIC_CACHE_DIR",
    )

    reset_config()
    _phasic_assigned_env.clear()
    for v in _PHASIC_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    yield
    reset_config()
    _phasic_assigned_env.clear()
    # Belt-and-braces: scrub env vars that ``phasic.configure(...)``
    # may have written via ``os.environ[...] = ...`` (which pytest's
    # monkeypatch does not track). Without this, the leftovers leak
    # into the next test module and break tests that assume a clean
    # default-config environment.
    for v in _PHASIC_ENV_VARS:
        os.environ.pop(v, None)


# ---------------------------------------------------------------
# configure() writes env vars
# ---------------------------------------------------------------


def test_configure_hierar_elimination_sets_env():
    phasic.configure(parallel_elimination=True)
    assert os.environ.get("PHASIC_HIERAR_ELIMINATION") == "1"


def test_configure_default_with_conflicting_env_does_nothing(monkeypatch):
    """If the shell has set PHASIC_HIERAR_ELIMINATION=1 and the
    user passes the field at its default (False), the env stays
    set — phasic does not write the env for default-valued fields.
    Effectively the shell wins for fields the user didn't touch."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    phasic.configure(parallel_elimination=False)
    # Env still set; that's intentional under the new policy
    # (default-valued fields don't write env vars).
    assert os.environ.get("PHASIC_HIERAR_ELIMINATION") == "1"


def test_configure_min_scc_size_sets_env():
    phasic.configure(parallel_elimination_min_subgraph=8)
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "8"


def test_configure_min_scc_size_zero():
    phasic.configure(parallel_elimination_min_subgraph=0)
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "0"


def test_configure_max_parallel_sccs_sets_env():
    phasic.configure(parallel_elimination_max_concurrent=4)
    assert os.environ.get("PHASIC_MAX_PARALLEL_SCCS") == "4"


def test_configure_disable_graph_cache_sets_env():
    phasic.configure(graph_cache=False)
    assert os.environ.get("PHASIC_DISABLE_GRAPH_CACHE") == "1"


def test_configure_enable_reward_compute_cache_sets_env():
    phasic.configure(reward_compute_cache=True)
    assert os.environ.get("PHASIC_REWARD_COMPUTE_CACHE") == "1"


def test_configure_default_reward_compute_cache_unsets_env():
    # Default policy is OFF; the field at its default removes the env var.
    phasic.configure(reward_compute_cache=False)
    assert os.environ.get("PHASIC_REWARD_COMPUTE_CACHE") is None


def test_configure_cache_dir_sets_env(tmp_path):
    phasic.configure(cache_dir=str(tmp_path))
    assert os.environ.get("PHASIC_CACHE_DIR") == str(tmp_path)


# ---------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------


def test_configure_raises_on_conflict_with_existing_env(monkeypatch):
    """configure() with a value disagreeing with a pre-existing
    env var raises PTDConfigError (refactored policy:
    shell-set env vars are user-authoritative)."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "100")
    with pytest.raises(PTDConfigError, match="PHASIC_MIN_SCC_SIZE_TO_CACHE"):
        phasic.configure(parallel_elimination_min_subgraph=4)


def test_configure_none_field_preserves_env(monkeypatch):
    """A field left at None does NOT overwrite a pre-existing env
    var. Useful so SLURM job scripts can set PHASIC_CACHE_DIR
    and a Python configure() call can change other knobs without
    blowing it away."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "16")
    phasic.configure(parallel_elimination=True)  # don't touch min_scc
    assert os.environ.get("PHASIC_MIN_SCC_SIZE_TO_CACHE") == "16"


# ---------------------------------------------------------------
# get_config() reads env vars
# ---------------------------------------------------------------


def test_get_config_reads_hierar_elimination_from_env(monkeypatch):
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cfg = phasic.get_config()
    assert cfg.parallel_elimination is True


def test_get_config_reads_min_scc_size_from_env(monkeypatch):
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "8")
    cfg = phasic.get_config()
    assert cfg.parallel_elimination_min_subgraph == 8


def test_get_config_reads_max_parallel_from_env(monkeypatch):
    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", "4")
    cfg = phasic.get_config()
    assert cfg.parallel_elimination_max_concurrent == 4


def test_get_config_reads_cache_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    cfg = phasic.get_config()
    assert cfg.cache_dir == str(tmp_path)


def test_get_config_handles_malformed_env(monkeypatch):
    """Malformed numeric env vars are silently ignored — the C
    side has its own fallback to sane defaults."""
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "not_a_number")
    cfg = phasic.get_config()
    assert cfg.parallel_elimination_min_subgraph is None


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------


def test_configure_rejects_negative_min_scc_size():
    with pytest.raises(PTDConfigError, match="parallel_elimination_min_subgraph"):
        phasic.configure(parallel_elimination_min_subgraph=-1)


def test_configure_rejects_zero_max_parallel():
    with pytest.raises(PTDConfigError, match="parallel_elimination_max_concurrent"):
        phasic.configure(parallel_elimination_max_concurrent=0)


def test_configure_unknown_field_raises():
    with pytest.raises(PTDConfigError, match="Unknown configuration option"):
        phasic.configure(this_field_does_not_exist=True)


# ---------------------------------------------------------------
# End-to-end: configure() actually changes C-side behaviour
# ---------------------------------------------------------------


def test_configure_threshold_affects_cache_bypass(tmp_path):
    """Setting parallel_elimination_min_subgraph via configure() actually
    changes which SCCs bypass the cache."""
    phasic.configure(
            parallel_elimination=True,
            cache_dir=str(tmp_path),
            reward_compute_cache=True,  # opt in (default is off)
            parallel_elimination_min_subgraph=1000,  # bypass everything
    )
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    assert s["cache_bypassed"] == 5
    assert s["cache_misses"] == 0


def test_configure_zero_threshold_caches_everything(tmp_path):
    """configure(parallel_elimination_min_subgraph=0) caches every SCC."""
    phasic.configure(
            parallel_elimination=True,
            cache_dir=str(tmp_path),
            reward_compute_cache=True,  # opt in (default is off)
            parallel_elimination_min_subgraph=0,
    )
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    assert s["cache_bypassed"] == 0
    assert s["cache_misses"] == 5


def test_configure_disable_reward_compute_cache_end_to_end(tmp_path):
    """reward_compute_cache=False (the default) bypasses load + save."""
    phasic.configure(
            parallel_elimination=True,
            cache_dir=str(tmp_path),
            reward_compute_cache=False,
            parallel_elimination_min_subgraph=0,
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
        {"parallel_elimination": False},
        {"parallel_elimination": True, "cache_dir": str(tmp_path / "a"),
         "parallel_elimination_min_subgraph": 0},
        {"parallel_elimination": True, "cache_dir": str(tmp_path / "b"),
         "parallel_elimination_min_subgraph": 1000},
        {"parallel_elimination": True, "cache_dir": str(tmp_path / "c"),
         "parallel_elimination_max_concurrent": 1, "parallel_elimination_min_subgraph": 0},
    ]:
        phasic.configure(**cfg_kwargs)
        g = build_toy_b()
        g.update_weights(THETA)
        result = np.asarray(g.expected_waiting_time())
        if base_result is None:
            base_result = result
        else:
            np.testing.assert_allclose(result, base_result, rtol=1e-12)
