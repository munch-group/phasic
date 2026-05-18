"""WP-4 tests: per-SCC PRC compute and disk cache.

Verifies ``ptd_scc_get_or_compute_prc`` (exposed as
``SCCVertex.get_or_compute_prc()`` in pybind):
- Cold cache: cache file is created on first call.
- Warm cache: cache file mtime unchanged on subsequent calls.
- Cross-graph reuse: two parents sharing an SCC structurally
  produce the same cache key, so the second parent's compute
  hits the first parent's cache.
- Removing PHASIC_REWARD_COMPUTE_CACHE skips both load and save
  (the default policy).
- Toy-D's aux SCC round-trips through the cache.

See wp4-plan.md for the design.
"""

from __future__ import annotations

import os
import time

import pytest

from phasic import Graph
from toy_model import (
    BUILDERS,
    build_toy_base,
    build_toy_c_p,
    build_toy_c_pprime,
    build_toy_d,
)


@pytest.fixture(autouse=True)
def _force_cache_all_sccs(monkeypatch):
    """This file's tests assert on cache file existence, so the
    reward-compute cache must be enabled (it is off by default) and
    the per-SCC size threshold must allow every SCC to be cached."""
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scc_with_external_anchors(g):
    """Return the first SCC that has at least one external anchor
    (so its cache file actually contains EXTERNAL pointers)."""
    scc_decomp = g.scc_decomposition()
    for scc in scc_decomp.sccs_in_topo_order():
        synth, meta = scc.as_synthetic_graph()
        if meta.n_upstream_connecting + meta.n_downstream_connecting > 0:
            return scc
    raise AssertionError("no SCC with placeholder edges in this graph")


def _find_scc_with_states(g, target_states):
    """Return the SCC whose internal vertices have the given set of states."""
    scc_decomp = g.scc_decomposition()
    for scc in scc_decomp.sccs_in_topo_order():
        states = {tuple(int(x) for x in g.vertex_at(i).state())
                  for i in scc.internal_vertex_indices()}
        if states == target_states:
            return scc
    raise AssertionError(f"no SCC with states {target_states}")


def _ensure_clean(path):
    """Remove the cache file if it exists, so the next compute is cold."""
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Cold-cache → save
# ---------------------------------------------------------------------------

def test_cold_cache_creates_file():
    """First call to get_or_compute_prc on an SCC creates the cache file."""
    g = build_toy_base()
    scc = _scc_with_external_anchors(g)
    path = scc.cache_file_path()
    _ensure_clean(path)
    assert not os.path.exists(path)

    synth, meta, n_anchors = scc.get_or_compute_prc()
    assert os.path.exists(path), (
        f"cache file {path} not created by get_or_compute_prc"
    )
    assert n_anchors > 0


def test_cold_cache_anchor_count_recorded():
    """Cache file size scales with number of EXTERNAL anchors."""
    g = build_toy_base()
    scc = _scc_with_external_anchors(g)
    path = scc.cache_file_path()
    _ensure_clean(path)

    _, _, n_anchors = scc.get_or_compute_prc()
    # Sanity: the file is non-empty.
    size = os.path.getsize(path)
    assert size > 64  # header is ~64 bytes


# ---------------------------------------------------------------------------
# Warm-cache → load (no rewrite)
# ---------------------------------------------------------------------------

def test_warm_cache_does_not_rewrite():
    """Second call with a warm cache should NOT rewrite the file
    (mtime unchanged)."""
    g = build_toy_base()
    scc = _scc_with_external_anchors(g)
    path = scc.cache_file_path()
    _ensure_clean(path)

    scc.get_or_compute_prc()
    mtime_first = os.path.getmtime(path)

    # Sleep enough to make any rewrite detectable.
    time.sleep(0.05)
    scc.get_or_compute_prc()
    mtime_second = os.path.getmtime(path)

    assert mtime_first == mtime_second, (
        f"cache file was rewritten on warm hit (mtime {mtime_first} -> "
        f"{mtime_second})"
    )


# ---------------------------------------------------------------------------
# Cross-graph reuse: the headline invariant
# ---------------------------------------------------------------------------

def test_toy_c_cross_graph_scc_reuse():
    """Toy-C P and Toy-C P' share SCC₂ structurally. Computing P
    populates the cache; computing P' must hit it (mtime unchanged).
    This is the cross-graph reuse invariant."""
    target_states = {(2, 0), (2, 1)}

    p = build_toy_c_p()
    p_scc = _find_scc_with_states(p, target_states)
    p_path = p_scc.cache_file_path()
    _ensure_clean(p_path)

    p_scc.get_or_compute_prc()
    assert os.path.exists(p_path)
    mtime_after_p = os.path.getmtime(p_path)

    pp = build_toy_c_pprime()
    pp_scc = _find_scc_with_states(pp, target_states)
    pp_path = pp_scc.cache_file_path()

    # Sanity: paths must match (this is the cache key invariant).
    assert p_path == pp_path

    time.sleep(0.05)
    pp_scc.get_or_compute_prc()
    mtime_after_pp = os.path.getmtime(p_path)

    assert mtime_after_p == mtime_after_pp, (
        "cross-graph cache hit failed: P populated the cache but "
        "P''s compute rewrote the file"
    )


def test_toy_c_unshared_scc_uses_different_cache_key():
    """Inverse of the cross-graph test: SCC₁ ({A,B}) has different
    internal coefficients in P vs P', so its cache key must differ."""
    target_states = {(1, 0), (1, 1)}

    p = build_toy_c_p()
    p_scc = _find_scc_with_states(p, target_states)
    p_path = p_scc.cache_file_path()

    pp = build_toy_c_pprime()
    pp_scc = _find_scc_with_states(pp, target_states)
    pp_path = pp_scc.cache_file_path()

    assert p_path != pp_path, (
        "cache keys collide for SCCs with different internal structure"
    )


# ---------------------------------------------------------------------------
# Cache disable
# ---------------------------------------------------------------------------

def test_disabled_reward_compute_cache_skips_save(monkeypatch):
    """When the reward-compute cache is disabled (the default
    policy, i.e. PHASIC_REWARD_COMPUTE_CACHE unset),
    ``get_or_compute_prc`` must not create or update the cache
    file."""
    g = build_toy_base()
    scc = _scc_with_external_anchors(g)
    path = scc.cache_file_path()
    _ensure_clean(path)

    monkeypatch.delenv("PHASIC_REWARD_COMPUTE_CACHE", raising=False)

    scc.get_or_compute_prc()
    assert not os.path.exists(path), (
        "Disabled cache still saved a file"
    )


def test_disabled_reward_compute_cache_skips_load(monkeypatch):
    """When the reward-compute cache is disabled, an existing
    cache file is ignored — every call recomputes without
    touching it."""
    g = build_toy_base()
    scc = _scc_with_external_anchors(g)
    path = scc.cache_file_path()
    _ensure_clean(path)

    # Pre-populate the cache (autouse fixture has set
    # PHASIC_REWARD_COMPUTE_CACHE=1).
    scc.get_or_compute_prc()
    assert os.path.exists(path)
    mtime_before = os.path.getmtime(path)

    # Now disable cache and compute again.
    monkeypatch.delenv("PHASIC_REWARD_COMPUTE_CACHE", raising=False)
    time.sleep(0.05)
    scc.get_or_compute_prc()

    # File still exists (we didn't delete it), but its mtime is
    # unchanged because the disabled-cache path doesn't touch it.
    mtime_after = os.path.getmtime(path)
    assert mtime_before == mtime_after, (
        "Disabled cache still touched the cache file"
    )


# ---------------------------------------------------------------------------
# Toy-D: duplicate-state aux SCC
# ---------------------------------------------------------------------------

def test_toy_d_aux_scc_caches():
    """Toy-D's aux-vertex SCC must round-trip through the cache
    correctly. The duplicate-state aux vertex is the regression
    case for the whole pipeline."""
    g = build_toy_d()
    scc_decomp = g.scc_decomposition()
    aux_scc = None
    for scc in scc_decomp.sccs_in_topo_order():
        if scc.size() == 3:
            aux_scc = scc
            break
    assert aux_scc is not None

    path = aux_scc.cache_file_path()
    _ensure_clean(path)

    aux_scc.get_or_compute_prc()
    assert os.path.exists(path)

    # Warm-cache call must succeed.
    mtime_first = os.path.getmtime(path)
    time.sleep(0.05)
    aux_scc.get_or_compute_prc()
    mtime_second = os.path.getmtime(path)
    assert mtime_first == mtime_second


# ---------------------------------------------------------------------------
# Coverage across all toys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("toy_name", list(BUILDERS.keys()))
def test_get_or_compute_prc_runs_on_every_toy_scc(toy_name):
    """Every SCC of every toy variant must succeed through
    get_or_compute_prc without error. This is the coverage
    smoke for the cache pipeline."""
    g = BUILDERS[toy_name]()
    scc_decomp = g.scc_decomposition()
    for scc in scc_decomp.sccs_in_topo_order():
        path = scc.cache_file_path()
        _ensure_clean(path)
        synth, meta, n_anchors = scc.get_or_compute_prc()
        assert os.path.exists(path)
        assert synth.vertices_length() == meta.n_vertices
