"""SLURM-WP-1: PHASIC_CACHE_DIR override.

Both the C-level cache (parameterized_reward_compute, traces) and
the Python helpers must honour PHASIC_CACHE_DIR. When set, all
on-disk cache I/O routes through that directory instead of
``$HOME/.phasic_cache``. This is the foundation for shared-
filesystem caching on SLURM clusters where home directories may
not be the right shared location.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import phasic.cache as cache
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


def test_python_cache_dir_honours_override(tmp_path, monkeypatch):
    """phasic.cache._param_compute_cache_dir() respects
    PHASIC_CACHE_DIR."""
    custom = tmp_path / "shared_cache"
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(custom))

    expected = custom / "parameterized_reward_compute"
    actual = cache._param_compute_cache_dir()
    assert actual == expected


def test_python_cache_dir_default(tmp_path, monkeypatch):
    """Without PHASIC_CACHE_DIR, falls back to $HOME/.phasic_cache."""
    monkeypatch.delenv("PHASIC_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    expected = tmp_path / ".phasic_cache" / "parameterized_reward_compute"
    actual = cache._param_compute_cache_dir()
    assert actual == expected


def test_c_cache_writes_to_override_dir(tmp_path, monkeypatch):
    """Running a hierarchical compose with PHASIC_CACHE_DIR set
    writes cache files into the override directory (not into
    $HOME/.phasic_cache)."""
    custom = tmp_path / "shared_cache"
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(custom))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    # Files should appear in the custom cache, not in $HOME.
    custom_files = list(
        (custom / "parameterized_reward_compute").glob("scc_*.bin"))
    assert len(custom_files) > 0, (
        f"expected cache files in {custom} after compose; found none")


def test_c_cache_does_not_touch_home_when_override_set(tmp_path, monkeypatch):
    """With PHASIC_CACHE_DIR set, $HOME/.phasic_cache is not
    touched during compose."""
    custom = tmp_path / "shared_cache"
    home_cache = Path.home() / ".phasic_cache" / "parameterized_reward_compute"

    # Snapshot existing files in home cache (other tests may have
    # populated it). We only check that NEW files don't appear
    # in $HOME after running with the override.
    before = set(home_cache.glob("scc_*.bin")) if home_cache.exists() else set()

    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(custom))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)
    cache.reset_scc_compose_stats()

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    after = set(home_cache.glob("scc_*.bin")) if home_cache.exists() else set()
    new_files_in_home = after - before
    assert len(new_files_in_home) == 0, (
        f"PHASIC_CACHE_DIR set but new files appeared in $HOME: "
        f"{new_files_in_home}")


def test_cache_hit_across_processes_via_override(tmp_path, monkeypatch):
    """First compose populates the override cache; second compose
    (after stats reset) hits it."""
    custom = tmp_path / "shared_cache"
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(custom))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    cache.reset_scc_compose_stats()
    g1 = build_toy_b(); g1.update_weights(THETA)
    r1 = g1.expected_waiting_time()
    s1 = cache.scc_compose_stats()
    assert s1["cache_misses"] == 5
    assert s1["cache_hits"] == 0

    cache.reset_scc_compose_stats()
    g2 = build_toy_b(); g2.update_weights(THETA)
    r2 = g2.expected_waiting_time()
    s2 = cache.scc_compose_stats()
    assert s2["cache_hits"] == 5
    assert s2["cache_misses"] == 0

    # Numerical results match.
    assert np.allclose(r1, r2)


def test_clear_cache_uses_override(tmp_path, monkeypatch):
    """phasic.cache.clear_param_compute_cache() under
    PHASIC_CACHE_DIR clears the override directory."""
    custom = tmp_path / "shared_cache"
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(custom))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    g = build_toy_b(); g.update_weights(THETA)
    _ = g.expected_waiting_time()

    cache_subdir = custom / "parameterized_reward_compute"
    assert any(cache_subdir.glob("scc_*.bin"))

    n = cache.clear_param_compute_cache()
    assert n > 0
    assert not any(cache_subdir.glob("scc_*.bin"))


def test_cache_info_uses_override(tmp_path, monkeypatch):
    """param_compute_cache_info reports files in override dir."""
    custom = tmp_path / "shared_cache"
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(custom))
    monkeypatch.delenv("PHASIC_DISABLE_CACHE", raising=False)

    g = build_toy_b(); g.update_weights(THETA)
    _ = g.expected_waiting_time()

    info = cache.param_compute_cache_info()
    assert info["n_files"] > 0
    assert str(custom / "parameterized_reward_compute") in info["cache_dir"]


def test_empty_override_falls_back_to_home(tmp_path, monkeypatch):
    """An empty PHASIC_CACHE_DIR=\"\" falls back to default."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", "")
    monkeypatch.setenv("HOME", str(tmp_path))

    expected = tmp_path / ".phasic_cache" / "parameterized_reward_compute"
    actual = cache._param_compute_cache_dir()
    assert actual == expected
