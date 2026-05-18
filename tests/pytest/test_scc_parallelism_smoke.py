"""Sanity gates for SCC composer parallelism.

These tests are deliberately lightweight — they don't try to
prove a specific speedup, just that the parallel path is wired
up correctly. They protect against three classes of silent
regression that the rest of the test suite missed:

  1. The `#pragma omp parallel for` getting silently dropped
     during preprocessing (OpenMP_C not linked, etc.). Detected
     by cpu/wall ratio > 1 on a multi-SCC workload.
  2. The TLS re-entrancy guard not propagating across OpenMP
     worker threads, causing inner expected_waiting_time calls
     to recurse into compose. Detected by cache_misses /
     cache_hits exceeding the SCC count.
  3. The hierarchical path not actually being taken under
     PHASIC_HIERAR_ELIMINATION=1. Detected by compose_calls
     staying at 0.

Each test uses a graph small enough to run quickly (~100 ms)
but with enough parallel-eliminable SCCs to actually exercise
the OpenMP fan-out.
"""

from __future__ import annotations

import os
import resource
import shutil
import tempfile
import time

import pytest

import phasic
import phasic.cache as cache
from phasic import Graph, Property, StateIndexer


THETA = [2.0, 5.0]


def _cpu_time() -> float:
    """User + system CPU time, RUSAGE_SELF (includes all
    threads of the current process)."""
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


@pytest.fixture
def parallel_friendly_graph(tmp_path, monkeypatch):
    """A two-locus ARG with many parallel-eliminable SCCs.
    Returns the constructed (parameterised) Graph."""
    nr_samples = 6
    indexer = StateIndexer(descendants=[
        Property("loc1", max_value=nr_samples),
        Property("loc2", max_value=nr_samples),
    ])
    initial = [0] * indexer.state_length
    initial[indexer.props_to_index(loc1=1, loc2=1)] = nr_samples

    def two_locus_arg(state, indexer=None):
        transitions = []
        if state.sum() <= 1:
            return transitions
        for i in range(indexer.state_length):
            if state[i] == 0:
                continue
            pi = indexer.index_to_props(i)
            for j in range(i, indexer.state_length):
                if state[j] == 0:
                    continue
                pj = indexer.index_to_props(j)
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                child = state.copy()
                child[i] -= 1
                child[j] -= 1
                loc1 = pi.descendants.loc1 + pj.descendants.loc1
                loc2 = pi.descendants.loc2 + pj.descendants.loc2
                if loc1 <= nr_samples and loc2 <= nr_samples:
                    child[indexer.props_to_index(loc1=loc1, loc2=loc2)] += 1
                    transitions.append([
                        child,
                        [state[i] * (state[j] - same) / (1 + same), 0],
                    ])
            if state[i] > 0 and pi.descendants.loc1 > 0 and pi.descendants.loc2 > 0:
                child = state.copy()
                child[i] -= 1
                child[indexer.props_to_index(loc1=pi.descendants.loc1, loc2=0)] += 1
                child[indexer.props_to_index(loc1=0, loc2=pi.descendants.loc2)] += 1
                transitions.append([child, [0, 1]])
        return transitions

    return Graph(two_locus_arg, ipv=initial, indexer=indexer)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Use a fresh cache dir per test so we exercise the
    elimination path, not load from a previous run's cache."""
    monkeypatch.setenv("PHASIC_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PHASIC_MIN_SCC_SIZE_TO_CACHE", "0")
    monkeypatch.setenv("PHASIC_REWARD_COMPUTE_CACHE", "1")


def test_hierar_env_var_takes_the_parallel_path(parallel_friendly_graph,
                                                 monkeypatch):
    """Under PHASIC_HIERAR_ELIMINATION=1, expected_waiting_time
    triggers exactly one compose_calls bump."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cache.reset_scc_compose_stats()

    g = parallel_friendly_graph
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 1, (
        f"hierarchical path was not taken; compose_calls={stats['compose_calls']}"
    )


def test_cache_misses_equal_scc_count_no_double_counting(
        parallel_friendly_graph, monkeypatch):
    """Each SCC should be cache-miss-and-compute exactly once
    per compose call. Excess counts indicate the TLS re-entrancy
    guard is broken and worker threads are recursing into the
    hierarchical path."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    cache.reset_scc_compose_stats()

    g = parallel_friendly_graph
    g.update_weights(THETA)
    _ = g.expected_waiting_time()

    scc_count = len(g.scc_decomposition())
    stats = cache.scc_compose_stats()
    total = stats["cache_hits"] + stats["cache_misses"] + stats["cache_bypassed"]
    assert total == scc_count, (
        f"cache events ({total}) != SCC count ({scc_count}); "
        f"hits={stats['cache_hits']} misses={stats['cache_misses']} "
        f"bypassed={stats['cache_bypassed']}. Excess events suggest "
        f"the TLS re-entrancy guard is not propagating to OpenMP "
        f"worker threads (each iteration of ptd_compose_scc_one must "
        f"bump ptd_scc_compose_in_progress on its own worker)."
    )


def test_cpu_time_exceeds_wall_time_on_warm_path(
        parallel_friendly_graph, monkeypatch):
    """When the per-SCC PRC cache is warm, the multiply-
    accumulate sweep runs in parallel across SCCs. CPU time
    consumed should noticeably exceed wall time on any
    multi-core machine.

    Skipped on single-core systems and when an explicit cap
    forces serial execution.
    """
    n_cpus = os.cpu_count() or 1
    if n_cpus < 2:
        pytest.skip("single-core machine; nothing to parallelise")
    if os.environ.get("OMP_NUM_THREADS") == "1":
        pytest.skip("OMP_NUM_THREADS=1; parallelism disabled by env")
    if os.environ.get("PHASIC_MAX_PARALLEL_SCCS") == "1":
        pytest.skip("PHASIC_MAX_PARALLEL_SCCS=1; parallelism capped to 1")

    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    g = parallel_friendly_graph
    g.update_weights(THETA)

    # Warm the cache.
    _ = g.expected_waiting_time()

    # Measure a fresh compose run against the warm cache.
    g.update_weights([3.0, 7.0])
    cache.reset_scc_compose_stats()
    cpu0 = _cpu_time()
    wall0 = time.perf_counter()
    _ = g.expected_waiting_time()
    wall = time.perf_counter() - wall0
    cpu = _cpu_time() - cpu0

    # Sanity check: we actually took the hierarchical path.
    stats = cache.scc_compose_stats()
    assert stats["compose_calls"] == 1
    assert stats["cache_hits"] > 0  # warm path should hit

    # The bar is intentionally low — we only want to catch the
    # silent-no-op regression. Even a modest 1.3x ratio is well
    # above what a serial execution would show (~1.0).
    ratio = cpu / wall if wall > 0 else 0.0
    assert ratio >= 1.3, (
        f"cpu/wall ratio {ratio:.2f} is too close to 1.0 — "
        f"elimination appears to be running on a single core "
        f"despite OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"and {n_cpus} CPUs available. Suspect:\n"
        f"  - OpenMP_C not linked into phasic_pybind (check "
        f"CMakeLists target_link_libraries)\n"
        f"  - '#pragma omp parallel for' silently dropped at "
        f"preprocessing\n"
        f"  - PHASIC_MAX_PARALLEL_SCCS cap kicking in\n"
        f"wall={wall:.3f}s cpu={cpu:.3f}s scc_hits={stats['cache_hits']}"
    )


def test_max_parallel_one_keeps_correctness(parallel_friendly_graph,
                                              monkeypatch):
    """Setting PHASIC_MAX_PARALLEL_SCCS=1 keeps the result
    correct (it's a scheduling knob, not a math knob)."""
    monkeypatch.setenv("PHASIC_HIERAR_ELIMINATION", "1")
    g = parallel_friendly_graph
    g.update_weights(THETA)

    monkeypatch.delenv("PHASIC_MAX_PARALLEL_SCCS", raising=False)
    result_par = g.expected_waiting_time()

    monkeypatch.setenv("PHASIC_MAX_PARALLEL_SCCS", "1")
    g2 = parallel_friendly_graph  # same fixture, fresh graph would be cleaner
    # Use a different theta to force a fresh compose.
    g2.update_weights([3.0, 7.0])
    _ = g2.expected_waiting_time()
    g2.update_weights(THETA)
    result_serial = g2.expected_waiting_time()

    import numpy as np
    np.testing.assert_allclose(result_par, result_serial, rtol=1e-12)
