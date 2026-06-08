"""Fix A: the non-daisy FD VJP skips fixed parameters.

Proves both properties at once via jax.grad of the model:
  * the fixed dimension's gradient is exactly 0 (skipped, not computed), and
  * the learnable dimensions are BIT-IDENTICAL to the no-mask gradient
    (value-preserving — the skip only drops work that SVGD discarded anyway).
Covers the standard 1D-reward path and the multivariate (2D-reward) path that
delegates to it (the path the moran-coalescent notebook uses).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import phasic
from phasic import Graph
import jax
import jax.numpy as jnp

TIMES = jnp.linspace(0.1, 4.0, 20)


def g3():
    """start -(1.0)-> [2] -(linear, coeffs [1,1,1])-> [1]; rate = t0+t1+t2."""
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0, 1.0, 1.0])
    return g


def test_fd_skip_zeros_fixed_and_preserves_learnable():
    theta = jnp.array([0.5, 0.6, 0.7])

    m_no = Graph.pmf_and_moments_from_graph(g3(), theta_dim=3)
    m_fix = Graph.pmf_and_moments_from_graph(
        g3(), theta_dim=3, fixed_mask=jnp.array([0.0, 1.0, 0.0]))

    def loss(model, t):
        pmf, _ = model(t, TIMES)
        return jnp.sum(jnp.log(pmf))

    g_no = np.asarray(jax.grad(lambda t: loss(m_no, t))(theta))
    g_fix = np.asarray(jax.grad(lambda t: loss(m_fix, t))(theta))

    assert g_fix[1] == 0.0                          # fixed dim skipped
    assert g_no[1] != 0.0                           # (it was genuinely nonzero)
    np.testing.assert_array_equal(g_fix[[0, 2]], g_no[[0, 2]])  # learnable bit-identical


def test_fd_skip_multivariate_path():
    # The 2D-rewards builder delegates to the 1D model; confirm the skip
    # propagates (this is the notebook's path).
    theta = jnp.array([0.5, 0.6, 0.7])
    n_v = g3().vertices_length()
    rewards2d = jnp.asarray(np.tile(np.arange(1.0, n_v + 1.0), (2, 1)))  # (2 features, n_v)

    m_no = Graph.pmf_and_moments_from_graph_multivariate(g3(), theta_dim=3)
    m_fix = Graph.pmf_and_moments_from_graph_multivariate(
        g3(), theta_dim=3, fixed_mask=jnp.array([1.0, 0.0, 0.0]))

    def loss(model, t):
        pmf, _ = model(t, TIMES, rewards=rewards2d)
        return jnp.nansum(jnp.log(pmf))

    g_no = np.asarray(jax.grad(lambda t: loss(m_no, t))(theta))
    g_fix = np.asarray(jax.grad(lambda t: loss(m_fix, t))(theta))

    assert g_fix[0] == 0.0                           # fixed dim skipped
    np.testing.assert_array_equal(g_fix[[1, 2]], g_no[[1, 2]])  # learnable bit-identical


def test_fd_skip_multiple_fixed_dims():
    # Two fixed dims, one free: both fixed gradients are 0, the free one isn't.
    theta = jnp.array([0.5, 0.6, 0.7])
    m = Graph.pmf_and_moments_from_graph(
        g3(), theta_dim=3, fixed_mask=jnp.array([1.0, 1.0, 0.0]))

    def loss(t):
        pmf, _ = m(t, TIMES)
        return jnp.sum(jnp.log(pmf))

    g = np.asarray(jax.grad(loss)(theta))
    assert g[0] == 0.0 and g[1] == 0.0 and g[2] != 0.0
