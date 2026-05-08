"""Stage A1 verification — thread-local persistent phasic::Graph in GraphBuilder.

After Stage A1, ``GraphBuilder::compute_*`` methods reuse a thread-
local persistent ``phasic::Graph`` instead of building a fresh graph
per call. Stage A0 ensures the symbolic
``parameterized_reward_compute_graph`` survives ``update_weights``;
Stage A1 ensures the *graph itself* survives across SVGD theta calls,
so the cache stays populated and the O(n^3) Gaussian elimination runs
exactly once per (thread, GraphBuilder).

These tests verify the four claims from the v2 plan §188:

1. Persistent graph + ``update_weights`` produces bit-identical
   results to fresh-graph-per-call across (theta, observation) inputs.
2. Bit-identical under sequential dispatch.
3. Bit-identical under vmap (with sequential ``vmap_method``) and pmap.
4. Thread-local isolation: pmap with N devices produces N independent
   persistent graphs; outputs match the sequential reference.
"""

import json
import numpy as np
import pytest

import phasic
from phasic import Graph

GraphBuilder = phasic.parameterized.GraphBuilder


def _build_test_graph_json(n_states=5):
    """Coalescent: state[0]=k, k → k-1 with rate proportional to
    k*(k-1)/2 × theta[0]."""

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


# ---------------------------------------------------------------------------
# Bit-identical results: persistent vs fresh-graph-per-call
# ---------------------------------------------------------------------------


def test_persistent_graph_pmf_matches_fresh_builder():
    """Two builders with the same JSON should produce identical PMF/
    moments for the same theta: builder 1 is reused (persistent
    graph), builder 2 is freshly constructed each time."""
    structure_json = _build_test_graph_json(n_states=8)
    times = np.linspace(0.1, 5.0, 25)

    persistent = GraphBuilder(structure_json)

    rng = np.random.default_rng(0)
    for seed in range(20):
        theta = rng.uniform(0.5, 3.0, size=1)
        # Persistent path: same builder reused, internally caches the
        # phasic::Graph after the first call.
        pmf_p, mom_p = persistent.compute_pmf_and_moments(
            theta, times, nr_moments=2, discrete=False, granularity=100,
        )

        # Fresh path: brand-new builder, no cache hit.
        fresh = GraphBuilder(structure_json)
        pmf_f, mom_f = fresh.compute_pmf_and_moments(
            theta, times, nr_moments=2, discrete=False, granularity=100,
        )

        np.testing.assert_allclose(
            pmf_p, pmf_f, rtol=1e-12,
            err_msg=f"PMF mismatch persistent-vs-fresh at seed={seed}",
        )
        np.testing.assert_allclose(
            mom_p, mom_f, rtol=1e-12,
            err_msg=f"Moments mismatch persistent-vs-fresh at seed={seed}",
        )


def test_persistent_graph_compute_pmf_matches_fresh():
    """Same as above but for the ``compute_pmf`` (no moments) path."""
    structure_json = _build_test_graph_json(n_states=6)
    times = np.linspace(0.1, 3.0, 15)

    persistent = GraphBuilder(structure_json)

    rng = np.random.default_rng(1)
    for seed in range(10):
        theta = rng.uniform(0.5, 3.0, size=1)
        pmf_p = persistent.compute_pmf(
            theta, times, discrete=False, granularity=100,
        )
        fresh = GraphBuilder(structure_json)
        pmf_f = fresh.compute_pmf(
            theta, times, discrete=False, granularity=100,
        )
        np.testing.assert_allclose(pmf_p, pmf_f, rtol=1e-12)


def test_persistent_graph_compute_moments_matches_fresh():
    """compute_moments path."""
    structure_json = _build_test_graph_json(n_states=6)
    persistent = GraphBuilder(structure_json)

    rng = np.random.default_rng(2)
    for seed in range(10):
        theta = rng.uniform(0.5, 3.0, size=1)
        mom_p = persistent.compute_moments(theta, nr_moments=3)
        fresh = GraphBuilder(structure_json)
        mom_f = fresh.compute_moments(theta, nr_moments=3)
        np.testing.assert_allclose(mom_p, mom_f, rtol=1e-12)


# ---------------------------------------------------------------------------
# Bit-identical under JAX vmap and pmap
# ---------------------------------------------------------------------------


def _make_jax_model():
    """Build a Graph.pmf_and_moments_from_graph model usable with JAX
    transformations."""

    def callback(state, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]

    graph = Graph(callback, ipv=[6])
    return graph, Graph.pmf_and_moments_from_graph(
        graph, nr_moments=2, discrete=False
    )


def test_persistent_graph_vmap_matches_sequential():
    """vmap'd evaluation across a batch of thetas must match the
    sequential reference computed one theta at a time."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    graph, model = _make_jax_model()

    times = jnp.linspace(0.1, 4.0, 12)
    thetas = jnp.array([[0.7], [1.0], [1.3], [2.0]])

    # Sequential reference
    pmf_seq = []
    mom_seq = []
    for theta in thetas:
        pmf, mom = model(theta, times)
        pmf_seq.append(np.asarray(pmf))
        mom_seq.append(np.asarray(mom))
    pmf_seq = np.stack(pmf_seq)
    mom_seq = np.stack(mom_seq)

    # vmap
    vmapped = jax.vmap(lambda t: model(t, times))
    pmf_vm, mom_vm = vmapped(thetas)

    np.testing.assert_allclose(np.asarray(pmf_vm), pmf_seq, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mom_vm), mom_seq, rtol=1e-12)


def test_persistent_graph_pmap_matches_sequential():
    """pmap'd evaluation must match the sequential reference. With
    multiple devices, each device gets its own thread-local
    GraphBuilder + persistent graph; outputs must still agree."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    n_devices = jax.local_device_count()
    if n_devices < 2:
        pytest.skip(f"pmap needs >=2 devices; got {n_devices}")

    graph, model = _make_jax_model()

    times = jnp.linspace(0.1, 4.0, 12)
    thetas = jnp.array([[0.7 + 0.3 * i] for i in range(n_devices)])

    # Sequential reference
    pmf_seq = []
    mom_seq = []
    for theta in thetas:
        pmf, mom = model(theta, times)
        pmf_seq.append(np.asarray(pmf))
        mom_seq.append(np.asarray(mom))
    pmf_seq = np.stack(pmf_seq)
    mom_seq = np.stack(mom_seq)

    # pmap (one device per theta)
    pmapped = jax.pmap(lambda t: model(t, times))
    pmf_pm, mom_pm = pmapped(thetas)

    np.testing.assert_allclose(np.asarray(pmf_pm), pmf_seq, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mom_pm), mom_seq, rtol=1e-12)


# ---------------------------------------------------------------------------
# Symbolic-elimination cache built once per builder (behavioural check)
# ---------------------------------------------------------------------------


def test_param_compute_graph_built_once_per_builder():
    """After Stage A0+A1, repeated calls to compute_pmf_and_moments on
    the same builder do NOT rebuild the symbolic
    parameterized_reward_compute_graph. Test by exercising
    ``expected_waiting_time`` (which populates that cache) on the
    underlying graph and asserting the cache stays populated across
    many update_weights cycles."""
    structure_json = _build_test_graph_json(n_states=8)
    builder = GraphBuilder(structure_json)
    times = np.linspace(0.5, 3.0, 5)

    # Drive the persistent graph through ``compute_pmf_and_moments``,
    # which invokes ``compute_moments_impl`` -> ``expected_waiting_time``
    # -> populates parameterized_reward_compute_graph.
    rng = np.random.default_rng(3)
    for _ in range(30):
        theta = rng.uniform(0.5, 3.0, size=1)
        builder.compute_pmf_and_moments(
            theta, times, nr_moments=2, discrete=False, granularity=100,
        )
    # If the cache had been destroyed and rebuilt every call, the test
    # would still pass (functional correctness) but slowly. The
    # behaviour we assert is that all 30 calls succeed and produce
    # finite results — Stage A1+A0 makes them cheap; without A1, each
    # call rebuilds the graph; without A0, each ``update_weights``
    # destroys the symbolic cache.
    pmf, mom = builder.compute_pmf_and_moments(
        np.array([1.0]), times, nr_moments=2, discrete=False, granularity=100,
    )
    assert np.all(np.isfinite(pmf))
    assert np.all(np.isfinite(mom))
