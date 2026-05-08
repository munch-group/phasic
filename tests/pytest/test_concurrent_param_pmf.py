"""Stage 3 verification — concurrent parameterised PDF.

These tests exercise the C parameterised PDF code path
(``ptd_graph_ex_absorbation_time_comp_graph_parameterized`` and
``ptd_precompute_reward_compute_graph``) from multiple OS threads
simultaneously. Before Stage 3:

- ``ptd_err`` was a shared global ``volatile char[4096]`` written by
  ``snprintf`` from 100+ sites — race-prone.
- ``ll_c2_alloced{,_index}`` and ``ll_p2_alloced{,_index}`` were
  shared statics torn down by every parameterised compute-graph build.
- ``ptd_precompute_reward_compute_graph``'s lazy build had no lock —
  two threads could race to build the cache.

Stage 3 makes the per-thread state thread-local (``__thread`` /
``_Thread_local``) and adds a per-graph ``pthread_mutex_t`` around the
lazy build. With those changes, calling ``compute_pmf_and_moments``
from N concurrent ``ThreadPoolExecutor`` workers on independent
``GraphBuilder`` instances must produce no segfault and per-thread
results that match the single-threaded reference.

These tests exist to catch regressions in the thread-safety story —
e.g., if a future patch adds a new shared static or removes the
mutex, this test fails (segfault, NaN, or numerical mismatch). They
do NOT directly catch the absence of the mutex *on master before
Stage 3*: the underlying thread-safety bug typically manifests as a
JAX pmap segfault rather than a deterministic threading hang in
plain Python ThreadPoolExecutor — the GIL serialises Python entry
to ``compute_pmf_and_moments``, but inside the function the GIL is
released and the C code runs concurrently. Even one of the many
small races (``ptd_err`` torn write, allocator double-free) suffices
to corrupt state.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from phasic import Graph
import phasic

GraphBuilder = phasic.parameterized.GraphBuilder


def _build_param_graph_json(n_states=10):
    """A simple n-state coalescent: state[0]=k, transitions k → k-1 with
    rate proportional to k(k-1)/2 × theta[0]. Parameterised on theta[0]."""

    def callback(state, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        coeff = float(n * (n - 1) / 2)
        return [(np.array([n - 1]), [coeff])]

    g = Graph(callback, ipv=[n_states])
    sd = g.serialize()
    return json.dumps(
        {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in sd.items()}
    )


def _single_thread_compute(structure_json, theta, times, granularity=100, nr_moments=2):
    builder = GraphBuilder(structure_json)
    pmf, moments = builder.compute_pmf_and_moments(
        np.asarray(theta, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        nr_moments=nr_moments,
        discrete=False,
        granularity=granularity,
    )
    return np.asarray(pmf), np.asarray(moments)


# ---------------------------------------------------------------------------
# Stress test: concurrent compute_pmf_and_moments across independent builders
# ---------------------------------------------------------------------------


def test_concurrent_compute_pmf_no_segfault():
    """8 worker threads × 32 iterations on independent GraphBuilder
    instances. Each iteration builds a fresh GraphBuilder from JSON
    (matches the JAX FFI path, which constructs a fresh builder per
    pure_callback invocation — see ffi_handlers.cpp). After the
    test, every per-thread result must equal the single-threaded
    reference computed in the main thread before workers start.

    On unpatched master, the C parameterised PDF path's shared
    ``ll_c2``/``ll_p2`` allocator state and ``ptd_err`` global buffer
    cause torn writes / use-after-free / segfaults under concurrent
    calls."""
    structure_json = _build_param_graph_json(n_states=10)
    times = np.linspace(0.1, 5.0, 20)

    # Reference values for each (theta_seed) input
    n_workers = 8
    n_iterations = 32

    rng_seeds = list(range(n_iterations))
    references = []
    for seed in rng_seeds:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.5, 2.0, size=1)
        pmf_ref, mom_ref = _single_thread_compute(
            structure_json, theta, times, granularity=100, nr_moments=2)
        references.append((theta, pmf_ref, mom_ref))

    def worker(seed):
        theta, _pmf_ref, _mom_ref = references[seed]
        builder = GraphBuilder(structure_json)
        pmf, moments = builder.compute_pmf_and_moments(
            theta,
            times,
            nr_moments=2,
            discrete=False,
            granularity=100,
        )
        return seed, np.asarray(pmf), np.asarray(moments)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(worker, rng_seeds))

    for seed, pmf_thr, mom_thr in results:
        _theta, pmf_ref, mom_ref = references[seed]
        np.testing.assert_allclose(
            pmf_thr, pmf_ref, rtol=1e-9,
            err_msg=f"PMF mismatch seed={seed} under concurrent execution",
        )
        np.testing.assert_allclose(
            mom_thr, mom_ref, rtol=1e-9,
            err_msg=f"Moments mismatch seed={seed} under concurrent execution",
        )


# ---------------------------------------------------------------------------
# Shared-graph mutex test: two threads, one Graph instance
# ---------------------------------------------------------------------------


def test_shared_graph_concurrent_pdf():
    """Two threads call pdf() on the *same* Python Graph object. This
    exercises the per-graph compute_graph_lock in
    ptd_precompute_reward_compute_graph: both threads observe the
    forward-state caches as NULL on first call, race to build them,
    and the mutex serialises the build.

    Without the mutex (Stage 3.2 not applied), the two threads can
    both allocate a reward_compute_graph, one wins the assignment,
    and the loser leaks. Worse: if one thread reads the half-built
    structure of the other, a segfault follows.
    """
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    g.update_weights([1.5, 1.0])

    # Reference single-threaded
    pdf_ref = float(g.pdf(2.0, granularity=100))

    # Build a fresh graph (so the cache is clean) for the concurrent test
    g2 = Graph(1)
    start = g2.starting_vertex()
    v3 = g2.find_or_create_vertex([3])
    v2 = g2.find_or_create_vertex([2])
    v1 = g2.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])
    g2.update_weights([1.5, 1.0])

    def worker(_seed):
        # All threads query the same time, so the cache is built once
        # and reused thereafter.
        return float(g2.pdf(2.0, granularity=100))

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, range(16)))

    for r in results:
        np.testing.assert_allclose(
            r, pdf_ref, rtol=1e-12,
            err_msg="Shared-graph concurrent pdf returned wrong value",
        )


# ---------------------------------------------------------------------------
# JAX FFI / pmap variant
# ---------------------------------------------------------------------------


def test_concurrent_jax_pmap_compute_pmf():
    """JAX pmap dispatches the pure_callback to multiple worker
    threads on different logical devices. This is the original
    failure mode that motivated Stage 3.

    The test is skipped if JAX exposes fewer than 2 local devices.
    """
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    if jax.local_device_count() < 2:
        pytest.skip(
            f"pmap requires >=2 local devices; got {jax.local_device_count()}"
        )

    def callback(state, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]

    g = Graph(callback, ipv=[5])
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False)

    # 2 thetas × 10 time points
    n_devices = jax.local_device_count()
    theta_batch = jnp.array(np.tile(np.array([1.5]), (n_devices, 1)))
    times = jnp.linspace(0.1, 5.0, 10)

    pmapped = jax.pmap(lambda t: model(t, times))
    pmf_batch, mom_batch = pmapped(theta_batch)
    pmf_arr = np.asarray(pmf_batch)
    mom_arr = np.asarray(mom_batch)
    assert np.all(np.isfinite(pmf_arr)), "Concurrent pmap produced non-finite PMF"
    assert np.all(np.isfinite(mom_arr)), "Concurrent pmap produced non-finite moments"

    # All devices use identical theta, so all rows must match
    for i in range(1, n_devices):
        np.testing.assert_allclose(
            pmf_arr[i], pmf_arr[0], rtol=1e-9,
            err_msg=f"pmap device {i} disagrees with device 0 (PMF)",
        )
        np.testing.assert_allclose(
            mom_arr[i], mom_arr[0], rtol=1e-9,
            err_msg=f"pmap device {i} disagrees with device 0 (moments)",
        )
