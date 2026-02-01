"""Tests for automatic learning rate calibration in SVGD.

Verifies that _calibrate_learning_rate produces sensible defaults and
that SVGD converges when learning_rate=None (no optimizer).
"""

from phasic import Graph, SVGD
from phasic.svgd import _calibrate_learning_rate, SVGDKernel, ConstantStepSize

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_exponential_graph():
    """Exponential(theta) graph: S -> [2] -> [1] with rate theta."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def generate_data(true_theta, n_samples=100, seed=42):
    np.random.seed(seed)
    graph = build_exponential_graph()
    graph.update_parameterized_weights([true_theta])
    return np.array(graph.sample(n_samples))


def gaussian_log_prob(theta, mu=2.0, sigma=1.0):
    """Simple Gaussian log probability for unit testing the calibration."""
    return -0.5 * jnp.sum(((theta - mu) / sigma) ** 2)


def gaussian_log_prob_narrow(theta, mu=2.0, sigma=0.1):
    """Narrow Gaussian — gradients ~100x larger than standard."""
    return -0.5 * jnp.sum(((theta - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Unit tests for _calibrate_learning_rate
# ---------------------------------------------------------------------------

def test_calibrate_produces_finite_positive_lr():
    """Auto-calibration returns a finite, positive learning rate."""
    key = jax.random.PRNGKey(0)
    theta_init = jax.random.normal(key, (20, 2))

    kernel = SVGDKernel(bandwidth='median_per_dim')
    lr = _calibrate_learning_rate(theta_init, gaussian_log_prob, kernel)

    assert np.isfinite(lr), f"lr must be finite, got {lr}"
    assert lr > 0, f"lr must be positive, got {lr}"
    assert lr <= 1.0, f"lr must be <= 1.0, got {lr}"


def test_calibrated_lr_scales_with_gradient_magnitude():
    """Narrower target (larger gradients) should produce smaller lr."""
    key = jax.random.PRNGKey(0)
    theta_init = jax.random.normal(key, (20, 2))

    kernel = SVGDKernel(bandwidth='median_per_dim')
    lr_wide = _calibrate_learning_rate(theta_init, gaussian_log_prob, kernel)
    lr_narrow = _calibrate_learning_rate(theta_init, gaussian_log_prob_narrow, kernel)

    # Narrow sigma=0.1 vs wide sigma=1.0 → gradients ~100x larger → lr ~100x smaller
    ratio = lr_wide / lr_narrow
    assert ratio > 5.0, (
        f"Expected lr_wide >> lr_narrow; "
        f"lr_wide={lr_wide:.6f}, lr_narrow={lr_narrow:.6f}, ratio={ratio:.2f}"
    )


def test_calibrate_with_fixed_mask():
    """Calibration works correctly when some parameters are fixed."""
    key = jax.random.PRNGKey(0)
    theta_init = jax.random.normal(key, (20, 3))

    fixed_mask = jnp.array([0, 1, 0])  # Fix dimension 1
    fixed_values = jnp.array([0.0, 5.0, 0.0])

    kernel = SVGDKernel(bandwidth='median_per_dim')
    lr = _calibrate_learning_rate(
        theta_init, gaussian_log_prob, kernel,
        fixed_mask=fixed_mask, fixed_values=fixed_values,
    )

    assert np.isfinite(lr), f"lr must be finite, got {lr}"
    assert lr > 0, f"lr must be positive, got {lr}"


# ---------------------------------------------------------------------------
# Integration tests via SVGD class
# ---------------------------------------------------------------------------

def test_svgd_auto_lr_convergence():
    """SVGD with auto-lr converges on Exponential(2.0) problem."""
    true_theta = 2.0
    data = generate_data(true_theta, n_samples=100, seed=42)

    graph = build_exponential_graph()
    model = Graph.pmf_and_moments_from_graph(graph, discrete=False, theta_dim=1)

    def prior(phi):
        return -0.5 * jnp.sum(((phi - 0.0) / 10.0) ** 2)

    svgd = SVGD(
        model=model,
        observed_data=data,
        prior=prior,
        theta_dim=1,
        n_particles=30,
        n_iterations=500,
        learning_rate=None,  # Trigger auto-calibration
        seed=42,
        verbose=True,
        parallel='vmap',
    )
    svgd.optimize()

    results = svgd.get_results()
    posterior_mean = float(results['theta_mean'][0])

    # Posterior mean should be within 30% of true value
    assert abs(posterior_mean - true_theta) / true_theta < 0.30, (
        f"Posterior mean {posterior_mean:.3f} too far from true {true_theta}"
    )

    # Check that learning rate was actually calibrated
    assert svgd.learning_rate is not None
    assert svgd.learning_rate > 0
    assert np.isfinite(svgd.learning_rate)


def test_svgd_auto_lr_with_fixed_params():
    """Auto-calibration works correctly with fixed parameters via SVGD class."""
    # 2-param model: fix one, optimize the other
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0, 0.0])  # rate = theta[0]

    model = Graph.pmf_and_moments_from_graph(g, discrete=False, theta_dim=2)
    data = generate_data(2.0, n_samples=50, seed=42)

    def prior(phi):
        return -0.5 * jnp.sum(((phi - 0.0) / 10.0) ** 2)

    svgd = SVGD(
        model=model,
        observed_data=data,
        prior=prior,
        theta_dim=2,
        n_particles=20,
        n_iterations=300,
        learning_rate=None,
        fixed=[(1, 1.0)],  # Fix theta[1] = 1.0
        seed=42,
        verbose=True,
        parallel='vmap',
    )
    svgd.optimize()

    assert svgd.learning_rate is not None
    assert svgd.learning_rate > 0
    assert np.isfinite(svgd.learning_rate)
