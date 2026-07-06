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

Known divergence: with ``rewards != None`` the FFI combined handler computes the
PMF on the UNTRANSFORMED graph (graph_builder_ffi.cpp:492) while the pybind path
uses the reward-transformed graph (graph_builder.cpp:741) -> xfail(strict) Q7.1.
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
# rewards!=None  ->  known divergence (Q7.1) + math-equal-not-bit-equal moments
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason=(
    "Q7.1: FFI ComputePmfAndMomentsFfiImpl computes the combined PMF on the "
    "UNTRANSFORMED graph (graph_builder_ffi.cpp:492) while pybind "
    "compute_pmf_and_moments uses the reward-transformed graph "
    "(graph_builder.cpp:741); rewards!=None PMFs differ. Stage-3 Q7 must unify "
    "which graph the combined PMF uses."))
def test_g1_pmf_and_moments_rewards_pmf_divergence():
    g = coalescent_graph(4)
    sj = structure_str(g)
    rew = np.linspace(0.5, 2.0, g.vertices_length())
    b = _builder(sj)
    pmf_py, _ = b.compute_pmf_and_moments(THETA, TIMES, 3, False, GRAN, rew)
    pmf_f, _ = compute_pmf_and_moments_ffi(sj, jnp.asarray(THETA), jnp.asarray(TIMES),
                                           3, discrete=False, granularity=GRAN,
                                           rewards=jnp.asarray(rew))
    np.testing.assert_array_equal(np.asarray(pmf_f), np.asarray(pmf_py))


def test_g1_pmf_and_moments_rewards_moments_close():
    """FFI reward-moments (expected_waiting_time(rewards) then *rewards) and pybind
    (reward_transform then standard moments) are math-equivalent but ~1 ulp apart.
    Pin agreement to a tight rtol, not bit-identity."""
    g = coalescent_graph(4)
    sj = structure_str(g)
    rew = np.linspace(0.5, 2.0, g.vertices_length())
    b = _builder(sj)
    _, mom_py = b.compute_pmf_and_moments(THETA, TIMES, 3, False, GRAN, rew)
    _, mom_f = compute_pmf_and_moments_ffi(sj, jnp.asarray(THETA), jnp.asarray(TIMES),
                                           3, discrete=False, granularity=GRAN,
                                           rewards=jnp.asarray(rew))
    np.testing.assert_allclose(np.asarray(mom_f), np.asarray(mom_py), rtol=1e-12, atol=0.0)
