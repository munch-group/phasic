"""GATE G1 — FFI-vs-pybind bit-identity for PMF and moments (Stage-3 duplication #7).

Pins the JAX XLA-FFI handler path (``ffi_wrappers.compute_{pmf,moments,
pmf_and_moments}_ffi`` -> ``CreateCompute*Handler`` in graph_builder_ffi.cpp)
against the pybind-direct ``GraphBuilder.compute_{pmf,moments,pmf_and_moments}``
path.  Per the refactor map the FFI handlers re-implement the times-loop /
discrete-dispatch orchestration and build a fresh graph per call, while the
pybind path reuses a persistent thread-local graph — so numeric divergence is a
real risk this gate must catch before Stage-3 Q7 unifies them.

Every test proves the FFI side actually lowered to the ``@ptd_compute_*``
custom-call (not the ``pure_callback`` fallback) via ``_gate_backend``.

Q7.1 (FIXED): with ``rewards != None`` the FFI combined handler used to compute
the PMF on the UNTRANSFORMED graph (graph_builder_ffi.cpp ~562/611) and feed real
rewards to the continuous moment path, while the pybind path reward-transforms
first (graph_builder.cpp ~787/825). Both now reward-transform once and read the
PMF and moments from the transformed graph, so the reward tests below are
FFI==pybind gates (continuous: bit-identical; discrete: the ~2-5% moment gap is
closed).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")  # belt-and-suspenders; phasic also sets it

import numpy as np
import pytest

import phasic
from _gate_backend import (  # noqa: E402
    requires_ffi,
    assert_ffi_target,
    assert_pybind_direct,
    coalescent_graph,
    dph_graph,
    structure_str,
    reset_jax_state,  # noqa: F401  (autouse fixture, activated by import)
)
from phasic.ffi_wrappers import (  # noqa: E402
    compute_pmf_ffi,
    compute_moments_ffi,
    compute_pmf_and_moments_ffi,
)

import jax.numpy as jnp  # noqa: E402

pytestmark = [pytest.mark.equivalence, requires_ffi]

GRAN = 100                        # fixed on BOTH sides (FFI default 0, pybind default 100)
THETA_PRIME = np.array([1.3])     # primes the pybind persistent graph via update_weights
THETA = np.array([3.7])           # the theta actually compared
TIMES = np.array([0.5, 1.0, 1.5, 2.0])


def _builder(sj):
    return phasic.parameterized.GraphBuilder(sj)


# --------------------------------------------------------------------------- #
# rewards=None  ->  exact bit-identity (both leaves run the same C kernels)
# --------------------------------------------------------------------------- #
def test_g1_pmf_continuous_bit_identity():
    sj = structure_str(coalescent_graph(4))
    b = _builder(sj)
    b.compute_pmf(THETA_PRIME, TIMES, False, GRAN)          # prime persistent path
    py = np.asarray(b.compute_pmf(THETA, TIMES, False, GRAN))
    assert_pybind_direct(py)
    assert_ffi_target(
        lambda th, t: compute_pmf_ffi(sj, th, t, discrete=False, granularity=GRAN),
        jnp.asarray(THETA), jnp.asarray(TIMES), target="ptd_compute_pmf",
    )
    ffi = np.asarray(compute_pmf_ffi(sj, jnp.asarray(THETA), jnp.asarray(TIMES),
                                     discrete=False, granularity=GRAN))
    np.testing.assert_array_equal(ffi, py)


def test_g1_pmf_discrete_bit_identity():
    sj = structure_str(dph_graph(3, 0.2))
    jumps = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    th = np.array([1.0])
    b = _builder(sj)
    b.compute_pmf(np.array([0.7]), jumps, True, GRAN)       # prime
    py = np.asarray(b.compute_pmf(th, jumps, True, GRAN))
    assert_ffi_target(
        lambda t, j: compute_pmf_ffi(sj, t, j, discrete=True, granularity=GRAN),
        jnp.asarray(th), jnp.asarray(jumps), target="ptd_compute_pmf",
    )
    ffi = np.asarray(compute_pmf_ffi(sj, jnp.asarray(th), jnp.asarray(jumps),
                                     discrete=True, granularity=GRAN))
    np.testing.assert_array_equal(ffi, py)


def test_g1_moments_bit_identity():
    sj = structure_str(coalescent_graph(4))
    b = _builder(sj)
    b.compute_moments(THETA_PRIME, 3)                       # prime
    py = np.asarray(b.compute_moments(THETA, 3))
    assert_ffi_target(
        lambda th: compute_moments_ffi(sj, th, 3),
        jnp.asarray(THETA), target="ptd_compute_moments",
    )
    ffi = np.asarray(compute_moments_ffi(sj, jnp.asarray(THETA), 3))
    np.testing.assert_array_equal(ffi, py)


def test_g1_pmf_and_moments_bit_identity_no_rewards():
    sj = structure_str(coalescent_graph(4))
    b = _builder(sj)
    b.compute_pmf_and_moments(THETA_PRIME, TIMES, 3, False, GRAN)   # prime
    pmf_py, mom_py = b.compute_pmf_and_moments(THETA, TIMES, 3, False, GRAN)
    assert_ffi_target(
        lambda th, t: compute_pmf_and_moments_ffi(sj, th, t, 3, discrete=False,
                                                  granularity=GRAN),
        jnp.asarray(THETA), jnp.asarray(TIMES), target="ptd_compute_pmf_and_moments",
    )
    pmf_f, mom_f = compute_pmf_and_moments_ffi(sj, jnp.asarray(THETA), jnp.asarray(TIMES),
                                               3, discrete=False, granularity=GRAN)
    np.testing.assert_array_equal(np.asarray(pmf_f), np.asarray(pmf_py))
    np.testing.assert_array_equal(np.asarray(mom_f), np.asarray(mom_py))


# --------------------------------------------------------------------------- #
# rewards!=None  ->  Q7.1 fixed: FFI now reward-transforms, matching pybind
# --------------------------------------------------------------------------- #
def test_g1_pmf_and_moments_rewards_pmf_matches_pybind():
    """Was strict-xfail Q7.1: the FFI combined handler computed the PMF on the
    UNTRANSFORMED graph, so a rewards!=None PMF diverged from pybind. The FFI now
    reward-transforms once and reads the PMF from the transformed graph, so the
    two paths run the same C kernel on the same graph -> bit-identical."""
    g = coalescent_graph(4)
    sj = structure_str(g)
    rew = np.linspace(0.5, 2.0, g.vertices_length())
    b = _builder(sj)
    pmf_py, _ = b.compute_pmf_and_moments(THETA, TIMES, 3, False, GRAN, rew)
    pmf_f, _ = compute_pmf_and_moments_ffi(sj, jnp.asarray(THETA), jnp.asarray(TIMES),
                                           3, discrete=False, granularity=GRAN,
                                           rewards=jnp.asarray(rew))
    np.testing.assert_array_equal(np.asarray(pmf_f), np.asarray(pmf_py))


def test_g1_pmf_and_moments_rewards_moments_match_pybind():
    """Both paths now compute reward moments as
    ``compute_moments_impl(reward_transform(g), {})`` -- previously the FFI fed
    real rewards to the CONTINUOUS moment path (expected_waiting_time(rewards)
    then *rewards), which was only ~1 ulp off in the continuous case but is the
    same code now, so this is bit-identity."""
    g = coalescent_graph(4)
    sj = structure_str(g)
    rew = np.linspace(0.5, 2.0, g.vertices_length())
    b = _builder(sj)
    _, mom_py = b.compute_pmf_and_moments(THETA, TIMES, 3, False, GRAN, rew)
    _, mom_f = compute_pmf_and_moments_ffi(sj, jnp.asarray(THETA), jnp.asarray(TIMES),
                                           3, discrete=False, granularity=GRAN,
                                           rewards=jnp.asarray(rew))
    np.testing.assert_array_equal(np.asarray(mom_f), np.asarray(mom_py))


def test_g1_pmf_and_moments_discrete_integer_rewards_match_pybind():
    """Discrete + integer rewards: the fix's real teeth. Before it, the FFI fed
    real-valued rewards to the continuous moment path and THEN applied the
    discrete correction -- mathematically inconsistent, ~2-5% off pybind
    (measured [20, 530, 17870] FFI vs [20, 520, 17000] pybind). Now both do
    ``reward_transform_discrete(int rewards)`` -> empty-reward moments ->
    correction, so PMF and moments both match pybind. No prior discrete-reward
    moment gate existed."""
    g = dph_graph(3, 0.2)
    sj = structure_str(g)
    th = np.array([1.0])
    jumps = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    rew = np.array([1.0, 2.0, 1.0, 2.0][: g.vertices_length()], dtype=np.float64)
    assert len(rew) == g.vertices_length()
    b = _builder(sj)
    b.compute_pmf_and_moments(np.array([0.7]), jumps, 3, True, GRAN, rew)   # prime
    pmf_py, mom_py = b.compute_pmf_and_moments(th, jumps, 3, True, GRAN, rew)
    pmf_f, mom_f = compute_pmf_and_moments_ffi(sj, jnp.asarray(th), jnp.asarray(jumps),
                                               3, discrete=True, granularity=GRAN,
                                               rewards=jnp.asarray(rew))
    np.testing.assert_array_equal(np.asarray(pmf_f), np.asarray(pmf_py))
    np.testing.assert_array_equal(np.asarray(mom_f), np.asarray(mom_py))


def test_g1_discrete_noninteger_rewards_raises_not_silently_wrong():
    """No-silent-fallback: discrete + NON-integer rewards must raise (via
    rewards_to_int_or_throw), not return a plausible number. Before the fix the
    FFI fed the real rewards straight to the continuous moment path and returned
    one."""
    g = dph_graph(3, 0.2)
    sj = structure_str(g)
    th = np.array([1.0])
    jumps = np.array([2.0, 3.0, 4.0])
    rew = np.array([1.0, 1.5, 1.0][: g.vertices_length()], dtype=np.float64)
    with pytest.raises(Exception):
        compute_pmf_and_moments_ffi(sj, jnp.asarray(th), jnp.asarray(jumps),
                                    3, discrete=True, granularity=GRAN,
                                    rewards=jnp.asarray(rew))


# ---------------------------------------------------------------------------
# Discrete PMF is EXACT — not a step-wise approximation
# ---------------------------------------------------------------------------
#
# The continuous PDF is a uniformization approximation whose error scales as
# 1/granularity (~5e-4 at the default). The DISCRETE pmf is not: dph_pmf(jumps)
# takes no granularity and steps the chain exactly. Pin that, because it is a
# property a refactor can silently destroy — and one did: the JIT wrapper for
# pmf_from_cpp(discrete=True) called the CONTINUOUS normalize(), which rescales
# every vertex's outgoing weights to sum to 1. That turns the chain into a
# deterministic walk (P(T = n_phases) = 1, else 0) and, since the rescaling
# divides theta out, makes the gradient identically zero.


def _two_phase_dph(p0, p1):
    """s -> v3 -(p0)-> v2 -(p1)-> v1(absorbing).

    In a DPH an edge weight IS the per-step transition probability and the
    deficit (1 - row sum) is the implicit stay-in-place probability, so with
    p0 == p1 == p the jump count is NegBinomial(2, p).
    """
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0])
    v2.add_edge(v1, [0.0, 1.0])
    return g


def test_g1_discrete_pmf_is_exact_not_discretised():
    # NegBinomial(2, p): P(T = n) = (n-1) p^2 (1-p)^(n-2), n >= 2.
    p = 0.3
    jumps = np.array([2.0, 3.0, 5.0, 10.0, 20.0])
    expected = (jumps - 1) * p ** 2 * (1 - p) ** (jumps - 2)

    model = phasic.Graph.pmf_from_graph(_two_phase_dph(p, p), discrete=True)
    got = np.asarray(model(jnp.asarray([p, p]), jnp.asarray(jumps)))

    # Machine precision — NOT the ~5e-4 the granularity-bound continuous path
    # would give. A wrong-but-plausible answer must not slip through, so this
    # is rtol=1e-12, not "isfinite and > 0".
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


def test_g1_discrete_pmf_actually_depends_on_theta():
    # Guards the normalize() failure mode directly: under it the PMF collapsed
    # to P(T = 2) = 1 and became CONSTANT in theta, so the gradient vanished.
    jumps = jnp.asarray([2.0, 3.0, 5.0])
    model = phasic.Graph.pmf_from_graph(_two_phase_dph(0.3, 0.3), discrete=True)

    a = np.asarray(model(jnp.asarray([0.3, 0.3]), jumps))
    b = np.asarray(model(jnp.asarray([0.7, 0.3]), jumps))

    assert not np.allclose(a, b), "discrete PMF is not responding to theta"
    # and it must not be the degenerate deterministic walk
    assert not np.allclose(a, [1.0, 0.0, 0.0]), "chain collapsed to a deterministic walk"
