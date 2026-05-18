"""Tests for ``phasic.cache`` cache management helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import phasic
import phasic.cache as cache_mod
from phasic import Graph

GraphBuilder = phasic.parameterized.GraphBuilder


def _build_chain_json(n_states: int) -> str:
    def cb(state, **kw):
        n = state[0]
        if n <= 1:
            return []
        return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]

    g = Graph(cb, ipv=[n_states])
    sd = g.serialize()
    return json.dumps(
        {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sd.items()}
    )


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Run each test against a clean ``~/.phasic_cache`` rooted at a
    pytest tmp_path. Every helper in ``phasic.cache`` resolves the
    cache directory from ``HOME`` at call time, so this isolates the
    test from the user's real cache. Also opts in to the
    reward-compute cache (off by default) since this file's tests
    exercise the cache machinery."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    yield tmp_path


# ---------------------------------------------------------------------------
# param_compute_cache_info
# ---------------------------------------------------------------------------


def test_info_empty_cache(isolated_cache):
    info = cache_mod.param_compute_cache_info()
    assert info["n_files"] == 0
    assert info["total_size"] == 0
    assert info["disabled"] is False
    assert info["cache_dir"] == str(
        isolated_cache / ".phasic_cache" / "parameterized_reward_compute"
    )


def test_info_with_files(isolated_cache):
    # Trigger one compute to populate the cache.
    builder = GraphBuilder(_build_chain_json(20))
    builder.compute_moments(np.array([1.0]), 2)

    info = cache_mod.param_compute_cache_info()
    assert info["n_files"] == 1
    assert info["total_size"] > 0


def test_info_reflects_disable_env(isolated_cache, monkeypatch):
    # Override the isolated_cache fixture's opt-in; the default
    # policy is "disabled".
    monkeypatch.delenv("PHASIC_REWARD_COMPUTE_CACHE", raising=False)
    info = cache_mod.param_compute_cache_info()
    assert info["disabled"] is True


# ---------------------------------------------------------------------------
# clear_param_compute_cache
# ---------------------------------------------------------------------------


def test_clear_empty_cache_returns_zero(isolated_cache):
    assert cache_mod.clear_param_compute_cache() == 0


def test_clear_removes_files(isolated_cache):
    # Populate with two distinct models.
    GraphBuilder(_build_chain_json(15)).compute_moments(np.array([1.0]), 2)
    GraphBuilder(_build_chain_json(20)).compute_moments(np.array([1.0]), 2)
    assert cache_mod.param_compute_cache_info()["n_files"] == 2

    deleted = cache_mod.clear_param_compute_cache()
    assert deleted == 2
    assert cache_mod.param_compute_cache_info()["n_files"] == 0


def test_clear_then_compute_repopulates(isolated_cache):
    """After clearing, the next compute must rebuild the cache from
    scratch and persist it again. Result must match the pre-clear
    result bit-identically (both go through the same elimination
    code path)."""
    structure_json = _build_chain_json(20)
    theta = np.array([1.7])

    b1 = GraphBuilder(structure_json)
    m1 = b1.compute_moments(theta, 2)
    assert cache_mod.param_compute_cache_info()["n_files"] == 1

    cache_mod.clear_param_compute_cache()
    assert cache_mod.param_compute_cache_info()["n_files"] == 0

    b2 = GraphBuilder(structure_json)
    m2 = b2.compute_moments(theta, 2)
    np.testing.assert_allclose(m2, m1, rtol=1e-12)
    assert cache_mod.param_compute_cache_info()["n_files"] == 1


# ---------------------------------------------------------------------------
# clear_all_caches
# ---------------------------------------------------------------------------


def test_clear_all_caches_returns_counts(isolated_cache):
    GraphBuilder(_build_chain_json(15)).compute_moments(np.array([1.0]), 2)
    result = cache_mod.clear_all_caches()
    assert isinstance(result, dict)
    assert "param_compute" in result
    assert "traces" in result
    assert result["param_compute"] == 1
    # No traces were created in this test, so the trace count is 0.
    assert result["traces"] == 0


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_module_is_importable_from_phasic():
    """``import phasic.cache`` must work and surface the documented
    public API."""
    import phasic.cache as c
    expected = {
        "is_cache_disabled",
        "clear_param_compute_cache",
        "param_compute_cache_info",
        "clear_trace_cache",
        "trace_cache_info",
        "clear_all_caches",
    }
    assert expected.issubset(set(dir(c)))


def test_phasic_cache_attribute_present():
    """``phasic.cache.clear_param_compute_cache`` should be reachable
    via attribute access on the top-level package."""
    assert callable(getattr(phasic.cache, "clear_param_compute_cache"))
    assert callable(getattr(phasic.cache, "param_compute_cache_info"))
