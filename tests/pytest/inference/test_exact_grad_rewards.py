"""B3 exact gradient + rewards: Graph.pmf_and_moments_from_graph(...,
exact_moment_grad=True, rewards=...).

Regression test for a bug found via adversarial review while making
exact_moment_grad default to True: the exact reverse-mode moment-vector
Jacobian is computed against a private, NEVER reward-transformed graph clone
(_exact_graph in pmf_and_moments_from_graph), and neither
ptd_moments_grad_theta nor ptd_moments_grad_theta_dph take a rewards argument
at all -- so whenever the forward actually returns reward-transformed
moments (rewards is not None), the "exact" Jacobian silently computed the
gradient of the WRONG (standard, reward=all-ones) moments instead. Confirmed
empirically before the fix: at rewards=[0,2,3,0.5], exact_moment_grad=True
returned [-6.0, -1.25] while the true gradient (FD, and an independent
central-difference of the forward reward-transformed moments) is
[-24.0, -8.25] -- wrong by 75-85%. Fixed by declining the exact path
whenever rewards are provided for this call, falling back to FD (which
already threads rewards through correctly) and logging why (no silent
fallback, matching the weight_mode-out-of-scope case).

Every assertion is anchored to an INDEPENDENT oracle: manual central-
difference of the model's own forward (reward-transformed) moments output,
never the code path under test.
"""
import contextlib
import logging

import numpy as np
import pytest

import phasic
from phasic import Graph

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402


def _build_three_stage():
    """S -> [3] --t0--> [2] --t1--> [1(absorbing)], 4 vertices total (incl.
    start), so a non-trivial (non-uniform) reward vector is meaningful."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0])
    v2.add_edge(v1, [0.0, 1.0])
    return g


def _manual_central_diff_of_forward(model, theta, times, rewards, eps=1e-6):
    """Independent oracle: central-difference the model's OWN forward
    (reward-transformed) moments sum directly, bypassing jax.grad/custom_vjp
    entirely."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        _, mp = model(jnp.asarray(tp), times, rewards)
        _, mm = model(jnp.asarray(tm), times, rewards)
        grad[i] = float(jnp.sum(mp) - jnp.sum(mm)) / (2 * eps)
    return grad


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_phasic_info_logs():
    logger = logging.getLogger('phasic')
    handler = _CollectingHandler()
    handler.setLevel(logging.INFO)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_exact_grad_with_rewards_matches_fd_and_forward_central_diff():
    """exact_moment_grad=True with a non-trivial reward vector must match FD
    (and the independent forward central-diff), not silently return the
    un-reward-transformed gradient."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])

    model_exact = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)
    model_fd = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=False)

    def loss(th, model):
        _, moments = model(th, times, rewards)
        return jnp.sum(moments)

    g_exact = np.asarray(jax.grad(lambda th: loss(th, model_exact))(theta))
    g_fd = np.asarray(jax.grad(lambda th: loss(th, model_fd))(theta))
    g_ref = _manual_central_diff_of_forward(model_fd, theta, times, rewards)

    np.testing.assert_allclose(g_exact, g_fd, rtol=1e-6)
    np.testing.assert_allclose(g_exact, g_ref, rtol=1e-4)


def test_exact_grad_without_rewards_unaffected():
    """The rewards guard must not disable the exact path when no rewards are
    passed at all (the common case) -- omitted and explicit rewards=None
    must both still use exact grad, matching each other exactly."""
    theta = jnp.asarray([1.0])
    times = jnp.asarray([1.0, 2.0])

    def build():
        g = Graph(1)
        s = g.starting_vertex()
        v2 = g.find_or_create_vertex([2])
        v1 = g.find_or_create_vertex([1])
        s.add_edge(v2, 1.0)
        v2.add_edge_parameterized(v1, 0.0, [1.0])
        return g

    model = Graph.pmf_and_moments_from_graph(build(), nr_moments=2, discrete=False, theta_dim=1, exact_moment_grad=True)

    with _capture_phasic_info_logs() as handler:
        g_omitted = jax.grad(lambda th: jnp.sum(model(th, times)[1]))(theta)
        g_none = jax.grad(lambda th: jnp.sum(model(th, times, None)[1]))(theta)
    np.testing.assert_array_equal(np.asarray(g_omitted), np.asarray(g_none))
    messages = [r.getMessage() for r in handler.records]
    assert not any("rewards" in m for m in messages)


def test_exact_grad_with_rewards_logs_why_fd_is_used():
    """No silent fallback: providing rewards with exact_moment_grad=True must
    log why finite differences are used instead."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])
    model = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)

    with _capture_phasic_info_logs() as handler:
        jax.grad(lambda th: jnp.sum(model(th, times, rewards)[1]))(theta)
    messages = [r.getMessage() for r in handler.records]
    assert any("rewards" in m and "finite differences" in m for m in messages)
