"""GATE G7 — persistent thread-local graph reuse equivalence (Stage-3 Q7).

Q7 will switch the FFI pmf/moments handler from fresh-build-per-batch-element
(graph_builder_ffi.cpp:436) onto the persistent ``per_thread_graph_cache``
pattern the pybind path already uses (``get_or_init_persistent_graph`` ->
``tl_persistent_graphs``, graph_builder.cpp:377).  These gates pin the numeric
invariant that must survive that flip: persistent-reuse output is bit-identical
to fresh-build output, and the FFI path is bit-identical to the pybind path.

Every result is bit-identical (``==``) today; the xfail template in the DPH gate
is provided commented-out for the Stage-3 engineer to flip only if a real
divergence appears.  A sibling of tests/pytest/test_concurrent_param_pmf.py
(whose comments still cite the Stage-1-deleted ffi_handlers.cpp).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import phasic
from phasic import Graph
from _gate_backend import requires_ffi, assert_ffi_target  # noqa: E402

pytestmark = [pytest.mark.equivalence]

par = phasic.phasic_pybind.parameterized
GraphBuilder = par.GraphBuilder


def _cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    return [(np.array([n - 1]), [float(n * (n - 1) / 2)])]


def _serialize(ipv):
    g = Graph(_cb, ipv=[ipv])
    sd = g.serialize()
    sj = json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in sd.items()})
    return g, sj


# ---------------------------------------------------------------------------
# G7a — Impl P: ONE shared GraphBuilder, N threads, K persistent-reuse calls
# ---------------------------------------------------------------------------
def test_g7a_shared_builder_persistent_reuse_matches_fresh():
    _g, sj = _serialize(ipv=8)              # unique ipv -> unique builder cache key
    times = np.linspace(0.1, 5.0, 16)
    GRAN = 200                              # rate 28*theta<=56 -> gran>=200

    struct = json.loads(sj)
    assert len(struct.get("param_edges", [])) >= 1     # persistent-reuse branch is taken

    seeds = list(range(24))
    refs = {}
    for s in seeds:                          # fresh builder per theta = reference
        th = np.random.default_rng(s).uniform(0.5, 2.0, size=1)
        p, m = GraphBuilder(sj).compute_pmf_and_moments(
            th, times, nr_moments=3, discrete=False, granularity=GRAN)
        refs[s] = (th, np.asarray(p), np.asarray(m))

    shared = GraphBuilder(sj)                # keep alive whole test (address stability)
    assert shared.param_length >= 1          # param_length is an attribute, not a method

    def worker(s):
        th, _, _ = refs[s]
        out = []
        for _ in range(20):                  # repeated -> tl_persistent_graphs HIT
            p, m = shared.compute_pmf_and_moments(
                th, times, nr_moments=3, discrete=False, granularity=GRAN)
            out.append((np.asarray(p), np.asarray(m)))
        return s, out

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, seeds * 2))

    for s, out in results:
        _, pr, mr = refs[s]
        for p, m in out:
            assert np.array_equal(p, pr), f"pmf drift seed={s}"      # bit-identical
            assert np.array_equal(m, mr), f"moments drift seed={s}"


# ---------------------------------------------------------------------------
# G7b — Impl F: forced FFI, alternating theta, per-call == fresh pybind ref
# ---------------------------------------------------------------------------
@requires_ffi
def test_g7b_ffi_persistent_alternating_theta_matches_fresh():
    import jax
    import jax.numpy as jnp
    if not phasic.get_config()._use_ffi:
        pytest.skip("FFI disabled in config")

    g, sj = _serialize(ipv=6)               # distinct ipv from G7a
    times = np.linspace(0.1, 5.0, 12)
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=3, discrete=False,
                                             use_ffi=True)
    jf = jax.jit(lambda t: model(t, jnp.asarray(times)))

    # BACKEND ASSERTION: prove FFI custom-call ran, not pure_callback
    assert_ffi_target(lambda t: model(t, jnp.asarray(times)),
                      jnp.array([1.0]), target="ptd_compute_pmf_and_moments")

    def fresh_ref(v):                        # model FFI uses granularity=0 internally
        return tuple(np.asarray(x) for x in GraphBuilder(sj).compute_pmf_and_moments(
            np.array([v]), times, nr_moments=3, discrete=False, granularity=0))

    thetas = [0.7, 1.7, 0.7, 1.7, 0.7]
    outs = [tuple(np.asarray(x) for x in jf(jnp.array([v]))) for v in thetas]
    for v, (p, m) in zip(thetas, outs):
        pr, mr = fresh_ref(v)
        assert np.array_equal(p, pr)         # bit-identical, granularity matched (0)
        assert np.array_equal(m, mr)
    # repeat-theta drift-free (stale-weight probe for the reused graph)
    assert np.array_equal(outs[0][0], outs[2][0])
    assert np.array_equal(outs[1][0], outs[3][0])


@requires_ffi
def test_g7b_ffi_vmap_reused_graph_numeric_stability():
    import jax
    import jax.numpy as jnp
    if not phasic.get_config()._use_ffi:
        pytest.skip("FFI disabled in config")

    g, sj = _serialize(ipv=7)
    times = jnp.linspace(0.1, 5.0, 12)
    model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, discrete=False,
                                             use_ffi=True)
    batch = jnp.array([[0.6], [1.1], [0.6], [1.7], [1.1]])
    vf = jax.jit(jax.vmap(lambda t: model(t, times)))

    assert_ffi_target(jax.vmap(lambda t: model(t, times)),
                      batch, target="ptd_compute_pmf_and_moments")

    pmf, mom = (np.asarray(x) for x in vf(batch))
    assert np.all(np.isfinite(pmf)) and np.all(np.isfinite(mom))
    assert np.array_equal(pmf[0], pmf[2])              # identical-theta rows identical
    assert np.array_equal(pmf[1], pmf[4])

    def fresh(v):
        return np.asarray(GraphBuilder(sj).compute_pmf_and_moments(
            np.array([v]), np.asarray(times), nr_moments=2, discrete=False,
            granularity=0)[0])
    assert np.array_equal(pmf[0], fresh(0.6))


# ---------------------------------------------------------------------------
# G7c — discrete/DPH persistent reuse (highest latent-divergence risk; today ==)
# ---------------------------------------------------------------------------
def test_g7c_discrete_persistent_reuse_matches_fresh():
    _g, sj = _serialize(ipv=3)              # DPH: rate 3*theta<=1 -> theta<=0.33
    jumps = np.arange(1, 10, dtype=float)
    b = GraphBuilder(sj)

    def fresh_ref(v):
        return tuple(np.asarray(x) for x in GraphBuilder(sj).compute_pmf_and_moments(
            np.array([v]), jumps, nr_moments=2, discrete=True, granularity=0))

    for v in [0.10, 0.30, 0.10, 0.30, 0.10]:
        p, m = b.compute_pmf_and_moments(np.array([v]), jumps, nr_moments=2,
                                         discrete=True, granularity=0)
        pr, mr = fresh_ref(v)
        assert np.array_equal(np.asarray(p), pr)   # bit-identical
        assert np.array_equal(np.asarray(m), mr)
    # If Stage-3 FFI-persistent regresses DPH dph_compute_invalidated re-normalization,
    # flip this to xfail(strict=True, reason="Q7: FFI pmf/moments persistent per-thread
    # graph diverges from fresh-build under DPH re-normalization; unify
    # graph_builder_ffi.cpp:436 onto the per_thread_graph_cache pattern (:877).")
