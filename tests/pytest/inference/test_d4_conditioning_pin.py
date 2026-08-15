"""Deferred-4 Phase-0 regression pin (CC-2, user-approved 2026-08-15).

Pins the sweep's CLEAN verdict (`b3-d4-sweep-findings.md`) against
future changes to the shared moments core:
1. exact == the FROZEN infinite-precision-oracle values at a benign and
   a mixed-scale point (the oracle: exact rational arithmetic,
   experiments/dr_d4_exact_oracle.py, calibrated at 1e-16-class against
   shipped values, shipped Jacobians, and an independent jax float64
   dense reference);
2. the extreme-ratio point DECLINES with the INFO line (the
   no-silent-fallbacks contract, verified end-to-end by the sweep);
3. everything gate-dependent SKIPS on a non-MPFR build (the gate is
   inert there -- ptd_dbg_tape_needs_mpfr returns 0 unconditionally, so
   a "clean" assertion would be meaningless; plan §2.3 / review F9).
"""
import contextlib
import logging
import os

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402


def _chain2():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 0.5])
    return g


def _have_mpfr() -> bool:
    """Force-probe: with PHASIC_FORCE_MPFR=1 the gate declines every call
    IF MPFR is compiled; on a non-MPFR build the gate is inert."""
    os.environ['PHASIC_FORCE_MPFR'] = '1'
    try:
        g = _chain2()
        g.update_weights([2.0, 1.0])
        return np.asarray(g._moments_grad_theta(3)).size == 0
    finally:
        os.environ.pop('PHASIC_FORCE_MPFR', None)


# Frozen oracle values (exact-rational oracle, dr_d4_exact_oracle.py;
# rationals evaluate to these float64 values exactly -- chain2, K=3).
ORACLE_BENIGN = np.array([  # theta = [2.0, 1.0]
    [-0.16, -0.08],
    [-0.256, -0.128],
    [-0.4608, -0.2304],
])
ORACLE_MIXED = np.array([  # theta = [1.0, 1e-8] -- the FD-defect regime
    [-0.9999999900000001, -0.49999999500000003],
    [-3.999999940000001, -1.9999999700000004],
    [-17.999999640000006, -8.999999820000003],
])


def test_exact_matches_frozen_oracle_benign():
    g = _chain2()
    g.update_weights([2.0, 1.0])
    J = np.asarray(g._moments_grad_theta(3)).reshape(3, 2)
    # sweep measured 0.0..6e-16 against the exact oracle here
    np.testing.assert_allclose(J, ORACLE_BENIGN, rtol=1e-12)


def test_exact_matches_frozen_oracle_mixed_scale():
    """The regime where FD is 5e-8-class wrong (and catastrophically
    wrong at large theta) -- the exact path must stay oracle-grade."""
    g = _chain2()
    g.update_weights([1.0, 1e-8])
    J = np.asarray(g._moments_grad_theta(3)).reshape(3, 2)
    np.testing.assert_allclose(J, ORACLE_MIXED, rtol=1e-12)


@pytest.mark.skipif(not _have_mpfr(),
                    reason="non-MPFR build: the conditioning gate is "
                           "inert; the decline pin would be meaningless")
def test_extreme_ratio_declines_with_info_line():
    """theta=[1, 1e-12]: gate condition ~1e13 > the 1e12 default
    threshold -> decline; through the model the fallback is FD with
    exactly the INFO line (sweep-verified end-to-end). Lifted-gate
    accuracy at this point was 2e-16 (the gate is conservative here --
    documented, not asserted)."""
    class _H(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def emit(self, r):
            self.msgs.append(r.getMessage())

    g = _chain2()
    g.update_weights([1.0, 1e-12])
    assert np.asarray(g._moments_grad_theta(3)).size == 0  # wrapper declines

    m = Graph.pmf_and_moments_from_graph(_chain2(), nr_moments=2,
                                         discrete=False, theta_dim=2)
    lg = logging.getLogger('phasic')
    h = _H(); h.setLevel(logging.INFO)
    prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(m(th, jnp.asarray([0.5, 1.0]))[1])
        )(jnp.asarray([1.0, 1e-12])))
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    assert np.all(np.isfinite(grad))
    assert any('finite differences' in x for x in h.msgs), h.msgs
