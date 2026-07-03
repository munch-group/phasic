"""Regression test: refine=True must not drift FIXED parameters.

The refine loop in log_likelihood(refine=True) and map_estimate_with_
optimization performed full-space gradient ascent (x = x + step*grad) with no
fixed_mask projection. A fixed parameter's likelihood gradient is generally
non-zero, so the "fixed" parameter drifted off its pinned value while
degrees_of_freedom still counted it fixed — corrupting AIC/BIC/LRT for nested
models.

We build a minimal SVGD instance and mock the surrounding seams so the refine
loop runs against a log-posterior whose gradient is non-zero on the fixed
dimension (so an unmasked loop WOULD drift it).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from phasic.svgd import SVGD

# Posterior peaked at (3, 9); gradient at x1=5 is -2*(5-9)=+8 (non-zero),
# so without masking the fixed dim (pinned at 5) would climb toward 9.
def _fake_logpost(x, **kw):
    return -((x[0] - 3.0) ** 2) - ((x[1] - 9.0) ** 2)


def _make_svgd():
    s = SVGD.__new__(SVGD)
    s.is_fitted = True
    s.theta_dim = 2
    s.param_transform = None
    s.nr_moments = 0
    s.sample_moments = None
    s.regularization = 0.0
    s.rewards = None
    # dim 1 is fixed at 5.0; dim 0 learnable, starts at 1.0
    s.fixed_mask = jnp.array([0.0, 1.0])
    s.fixed_values = jnp.array([0.0, 5.0])
    s._log_prob_unified = _fake_logpost
    s.map_estimate_from_particles = lambda unconstrained=False: ([1.0, 5.0], 0.0)
    return s


def test_log_likelihood_refine_keeps_fixed_param_pinned():
    s = _make_svgd()
    captured = {}

    def fake_ll_at(theta, rewards=None):
        captured["theta"] = np.asarray(theta)
        return 0.0

    s._log_likelihood_at = fake_ll_at
    s.log_likelihood(refine=True)

    theta = captured["theta"]
    assert theta[1] == pytest.approx(5.0, abs=1e-6), (
        f"fixed dim drifted to {theta[1]} (should stay pinned at 5.0)"
    )
    # Sanity: the learnable dim actually moved toward its optimum (3.0).
    assert theta[0] > 1.5, "learnable dim should have ascended"


def test_map_estimate_with_optimization_keeps_fixed_param_pinned():
    s = _make_svgd()
    # This method returns the refined theta directly.
    s.map_estimate_from_particles = lambda unconstrained=False: (
        jnp.array([1.0, 5.0]), 0.0,
    )
    refined, _ = s.map_estimate_with_optimization(n_steps=100, step_size=0.05)
    assert refined[1] == pytest.approx(5.0, abs=1e-6), (
        f"fixed dim drifted to {refined[1]} (should stay pinned at 5.0)"
    )
    assert refined[0] == pytest.approx(3.0, abs=0.3), "learnable dim should converge to 3"
