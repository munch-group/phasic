"""Tests for the user-facing SCC composer controls:

  - PHASIC_MIN_SCC_SIZE_TO_CACHE: skip the cache for small SCCs.
  - PHASIC_MAX_PARALLEL_SCCS: cap simultaneous per-SCC computes
    within a level (independent of OMP_NUM_THREADS).

Default behaviour (threshold=4) is exercised separately from
the explicit-override path (threshold=0).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import phasic.cache as cache
from toy_model import build_toy_b


THETA = [1.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------
# PHASIC_MIN_SCC_SIZE_TO_CACHE
# ---------------------------------------------------------------


def test_default_threshold_bypasses_small_synth(tmp_path, monkeypatch):
    """Default PHASIC_MIN_SCC_SIZE_TO_CACHE=4: toy_b's smallest
    SCC (synth=3 vertices) is bypassed; the rest are cached."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    monkeypatch.delenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", raising=False)

    cache.reset_scc_compose_stats()
    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    # toy_b has 5 SCCs; SCC 4's synth has 3 vertices -> bypassed.
    assert s["cache_bypassed"] == 1
    assert s["cache_misses"] == 4
    assert s["cache_hits"] == 0


def test_threshold_zero_caches_everything(tmp_path, monkeypatch):
    """PHASIC_MIN_SCC_SIZE_TO_CACHE=0: every SCC participates
    in the cache; bypass count is 0."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    cache.reset_scc_compose_stats()
    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    assert s["cache_bypassed"] == 0
    assert s["cache_misses"] == 5


def test_threshold_high_bypasses_all(tmp_path, monkeypatch):
    """PHASIC_MIN_SCC_SIZE_TO_CACHE=1000: all SCCs bypass the
    cache (toy_b synths are all <1000 vertices)."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "1000")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    cache.reset_scc_compose_stats()
    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    s = cache.scc_compose_stats()
    assert s["cache_bypassed"] == 5
    assert s["cache_misses"] == 0
    assert s["cache_hits"] == 0


def test_threshold_invalid_falls_back_to_default(tmp_path, monkeypatch):
    """A non-numeric or negative value falls back to default 4."""
    for bad in ["abc", "-5", ""]:
        monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
        monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / f"c_{bad or 'empty'}"))
        monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", bad)
        monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

        cache.reset_scc_compose_stats()
        g = build_toy_b()
        g.update_weights(THETA)
        _ = g.expected_waiting_time()
        s = cache.scc_compose_stats()
        # Default 4 -> 1 bypass for toy_b.
        assert s["cache_bypassed"] == 1, (
                f"value {bad!r} did not fall back to default 4")


def test_threshold_correctness_unchanged(tmp_path, monkeypatch):
    """Whatever the threshold value, the numerical result is the
    same — the threshold only controls *whether* we cache, not
    the math."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    results = {}
    for thresh in ["0", "4", "1000"]:
        monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", thresh)
        g = build_toy_b()
        g.update_weights(THETA)
        results[thresh] = np.asarray(g.expected_waiting_time())

    np.testing.assert_allclose(results["0"], results["4"], rtol=1e-12)
    np.testing.assert_allclose(results["0"], results["1000"], rtol=1e-12)


def test_bypassed_synths_do_not_write_cache_files(tmp_path, monkeypatch):
    """When all SCCs are bypassed, no cache files appear on disk."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "1000")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")

    g = build_toy_b()
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    cache_subdir = tmp_path / "parameterized_reward_compute"
    files = list(cache_subdir.glob("scc_*.bin")) if cache_subdir.exists() else []
    assert len(files) == 0


# ---------------------------------------------------------------
# PHASIC_MAX_PARALLEL_SCCS
# ---------------------------------------------------------------


@pytest.mark.parametrize("max_par", ["1", "2", "4"])
def test_max_parallel_correctness_unchanged(max_par, monkeypatch):
    """The cap controls scheduling, not math. Result agrees
    with the unconstrained run."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")
    monkeypatch.delenv("PHASIC_MAX_PARALLEL_SCCS", raising=False)

    g_un = build_toy_b()
    g_un.update_weights(THETA)
    res_un = np.asarray(g_un.expected_waiting_time())

    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", max_par)
    g_cap = build_toy_b()
    g_cap.update_weights(THETA)
    res_cap = np.asarray(g_cap.expected_waiting_time())

    np.testing.assert_allclose(res_un, res_cap, rtol=1e-12, atol=1e-12)


def test_max_parallel_zero_is_unconstrained(monkeypatch):
    """PHASIC_MAX_PARALLEL_SCCS=0 falls back to OpenMP default
    (no cap). Correctness check only — scheduling isn't
    observable from Python."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")
    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", "0")

    g = build_toy_b()
    g.update_weights(THETA)
    r = g.expected_waiting_time()
    assert r[0] == pytest.approx(1.88, rel=1e-12)


def test_max_parallel_invalid_falls_back(monkeypatch):
    """Non-numeric or negative values fall back to no cap."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")
    for bad in ["abc", "-1", ""]:
        monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", bad)
        g = build_toy_b()
        g.update_weights(THETA)
        r = g.expected_waiting_time()
        assert r[0] == pytest.approx(1.88, rel=1e-12), (
                f"value {bad!r} broke compose")


def test_max_parallel_independent_of_omp_num_threads(monkeypatch):
    """Setting PHASIC_MAX_PARALLEL_SCCS=1 caps SCC fan-out to
    1 even when OMP_NUM_THREADS is high. Verified via
    correctness only — actual concurrency is not observable
    from Python — but the path must not deadlock or fail."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", "1")

    g = build_toy_b()
    g.update_weights(THETA)
    r = g.expected_waiting_time()
    assert r[0] == pytest.approx(1.88, rel=1e-12)


# ---------------------------------------------------------------
# Stats key surface
# ---------------------------------------------------------------


def test_cache_bypassed_key_present_after_reset():
    cache.reset_scc_compose_stats()
    s = cache.scc_compose_stats()
    assert "cache_bypassed" in s
    assert s["cache_bypassed"] == 0
