"""B3 exact gradient + rewards: Graph.pmf_and_moments_from_graph(...,
exact_moment_grad=True, rewards=...).

HISTORY. The original version of this file was the regression test for a
bug found via adversarial review while making exact_moment_grad default to
True: the exact Jacobian ignored rewards entirely (computed the gradient of
the STANDARD moments -- [-6.0, -1.25] where the true reward-transformed
gradient is [-24.0, -8.25], wrong by 75-85%), and the fix then was to
DECLINE the exact path whenever rewards were provided, falling back to FD
with a log.

CURRENT (Batch A, 2026-08-14, b3-batchA-plan.md). 1-D rewards are threaded
INTO the exact moments adjoint: the C core (ptd_b3_moments_core) re-scales
at every stage of the forward moment chain and applies the matching
adjoint-side VJP, so the exact path now ENGAGES with rewards on continuous
linear- and log-mode models. Two static declines remain, both tested here:
DISCRETE models (the continuous->discrete moment correction is REFUTED
under reward weighting -- permanent, not a feature gap) and 2-D rewards on
this 1-D leaf (the multivariate wrapper's per-feature 1-D slices take the
exact path instead).

Every value assertion is anchored to an INDEPENDENT oracle: manual central-
difference of the model's own forward (reward-transformed) moments output,
never the code path under test. Tolerances are measured actuals (G4-fold
probe, 2026-08-14), not guesses.
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


@pytest.mark.parametrize("nr_moments", [1, 2, 3])
def test_exact_grad_with_rewards_matches_fd_and_forward_central_diff(nr_moments):
    """Batch A REWORK: pre-A both models were FD (this compared a path to
    itself); post-A exact-with-rewards is a genuinely different path and
    this is a REAL exact-vs-FD comparison, at every supported K (the C
    core's per-stage reward re-scaling runs inside the j=1..K loop, so
    K=1 alone could not detect a seed-only-scaling bug -- K>=2 can).
    Measured actuals (G4-fold probe): exact grads are exactly
    K=1: [-2, -0.75], K=2: [-24, -8.25], K=3: [-267, -83.625];
    rel vs FD <= 5.6e-10, rel vs central-diff <= 9.9e-11."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])

    model_exact = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=nr_moments, discrete=False,
        theta_dim=2, exact_moment_grad=True)
    model_fd = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=nr_moments, discrete=False,
        theta_dim=2, exact_moment_grad=False)

    def loss(th, model):
        _, moments = model(th, times, rewards)
        return jnp.sum(moments)

    g_exact = np.asarray(jax.grad(lambda th: loss(th, model_exact))(theta))
    g_fd = np.asarray(jax.grad(lambda th: loss(th, model_fd))(theta))
    g_ref = _manual_central_diff_of_forward(model_fd, theta, times, rewards)

    np.testing.assert_allclose(g_exact, g_fd, rtol=1e-8)
    np.testing.assert_allclose(g_exact, g_ref, rtol=1e-8)


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


def test_exact_grad_with_rewards_engages_silently_continuous(monkeypatch):
    """Batch A REWRITE of test_exact_grad_with_rewards_logs_why_fd_is_used
    (the fate table's break-by-design): 1-D rewards on a CONTINUOUS model
    now use the exact path -- the old decline log must be GONE.

    G4-fold hardening: the original log-absence assertion grepped a string
    this batch DELETED, so it was vacuously true even with the exact path
    monkeypatched to decline every call (G4 probe). Engagement is now
    asserted POSITIVELY via a spy on the C Jacobian entry point (measured:
    exactly 1 call, rewards_len=4, full-size return), and the log filter
    greps ANY "finite differences" message (static declines and per-theta
    declines both contain it; measured zero INFO records on this fixture)."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])
    model = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)

    calls = []
    orig = Graph._moments_grad_theta

    def spy(self, K, *args, **kwargs):
        rw = kwargs.get('rewards')
        result = orig(self, K, *args, **kwargs)
        calls.append((K, 0 if rw is None else len(rw), np.asarray(result).size))
        return result

    monkeypatch.setattr(Graph, "_moments_grad_theta", spy)
    with _capture_phasic_info_logs() as handler:
        g = jax.grad(lambda th: jnp.sum(model(th, times, rewards)[1]))(theta)
    assert np.all(np.isfinite(np.asarray(g)))
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m for m in messages), messages
    # engagement: the exact C Jacobian ran, WITH the rewards, and returned a
    # full-size (K*P = 2*2) Jacobian rather than a decline sentinel
    assert len(calls) >= 1, "exact C Jacobian never called -- FD fallback?"
    assert all(rw_len == 4 and size == 4 for _, rw_len, size in calls), calls


def test_exact_grad_with_rewards_discrete_declines_with_refutation_log():
    """Batch A: DISCRETE models decline rewards+exact with a truthful log
    (the c2d correction is REFUTED under reward weighting)."""
    theta = jnp.asarray([0.3, 0.4])
    jumps = jnp.asarray([1.0, 2.0])
    # the discrete reward transform requires NON-NEGATIVE INTEGER rewards
    # (plan-review finding; fractional values raise in the forward)
    rewards = jnp.asarray([0.0, 2.0, 3.0, 1.0])
    model = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=True, theta_dim=2,
        exact_moment_grad=True)

    with _capture_phasic_info_logs() as handler:
        g = jax.grad(lambda th: jnp.sum(model(th, jumps, rewards)[1]))(theta)
    assert np.all(np.isfinite(np.asarray(g)))
    messages = [r.getMessage() for r in handler.records]
    assert any("DISCRETE" in m and "correction" in m for m in messages), messages


def test_exact_grad_rewards_all_ones_matches_rewardless():
    """all-ones rewards == rewardless, bitwise-class (the strongest cheap
    sensitivity check on both hook lines -- G4-fold discipline from the
    plan review)."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    model = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2,
        exact_moment_grad=True)
    ones = jnp.ones(4)

    g_rw = np.asarray(jax.grad(
        lambda th: jnp.sum(model(th, times, ones)[1]))(theta))
    g_no = np.asarray(jax.grad(
        lambda th: jnp.sum(model(th, times)[1]))(theta))
    # NOTE: the FORWARD with ones-rewards runs the reward-transform (a
    # different C path) so the pmf term could differ at fp; the MOMENTS
    # gradient loss isolates the adjoint -- measured BITWISE identical
    # (G4 probe Q1: max abs diff exactly 0.0), so assert equality per the
    # measured-actuals discipline. This equality doubles as the leaf's
    # de-facto engagement guard: a silent FD fallback differs at ~1e-9
    # here (G4 probe Q2), far above bitwise.
    np.testing.assert_array_equal(g_rw, g_no)


def test_multivariate_per_feature_exact_engages():
    """The Batch-A free side effect (master s3): the multivariate wrapper
    slices 2-D rewards to 1-D per feature, so each feature's exact path
    engages with no decline logs, and the gradient is finite. (CLOSURE
    NOTE, Batch G.2 2026-08-15: the direct 2-D-on-1-D route -- whose
    FORWARD shape-contract defect was ledgered at Batch A's merge review
    -- is now LOUDLY REJECTED by user decision (uniform rejection,
    master s16b item 10 CLOSED); this wrapper is the one blessed 2-D
    route, and its kwarg is covered by
    test_exact_grad_multivariate_kwarg.py.)"""
    theta = jnp.asarray([1.0, 2.0])
    times2d = jnp.asarray([[1.0, 2.0], [1.5, 2.5]]).T  # (n_times, n_features)
    # wrapper slices rewards_arr[j, :] -> rows are FEATURES: (n_features, n_vertices)
    rewards2d = jnp.asarray([[0.0, 2.0, 3.0, 0.5], [1.0, 1.0, 2.0, 1.0]])
    model = Graph.pmf_and_moments_from_graph_multivariate(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2)
    with _capture_phasic_info_logs() as handler:
        g = np.asarray(jax.grad(
            lambda th: jnp.sum(model(th, times2d, rewards2d)[1]))(theta))
    assert np.all(np.isfinite(g))
    # G4-fold: the promised VALUE assertion (plan I4.3 / amendment 4's
    # absolute pin). Measured actuals: exact grad is exactly
    # [-31.0, -11.75]; rel vs the independent central-diff oracle 4.0e-11.
    g_ref = _manual_central_diff_of_forward(model, theta, times2d, rewards2d)
    np.testing.assert_allclose(g, [-31.0, -11.75], rtol=1e-9)
    np.testing.assert_allclose(g, g_ref, rtol=1e-8)
    messages = [r.getMessage() for r in handler.records]
    # measured zero INFO records on this fixture -- any decline is a failure
    assert not any("finite differences" in m for m in messages), messages


def test_exact_grad_rewards_log_mode_matches_central_diff():
    """G4-fold (plan I4.2, closing the review-found evidence gap): rewards
    through the LOG-mode wrapper (_moments_grad_theta_log) -- until the G4
    probes this combination had zero empirical coverage anywhere. Fixture
    needs every edge's full coefficient*theta product positive (log mode:
    w_e = prod_i c_e[i]*theta_i over ALL i). Measured actuals: exact grad
    exactly [-27.0, -13.5], rel vs central-diff 2.5e-11."""
    g = Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.5])
    v2.add_edge(v1, [2.0, 1.0])
    g.weight_mode = 'log'

    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])
    model = Graph.pmf_and_moments_from_graph(
        g, nr_moments=2, discrete=False, theta_dim=2, exact_moment_grad=True)

    with _capture_phasic_info_logs() as handler:
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(model(th, times, rewards)[1]))(theta))
    g_ref = _manual_central_diff_of_forward(model, theta, times, rewards)
    np.testing.assert_allclose(grad, [-27.0, -13.5], rtol=1e-9)
    np.testing.assert_allclose(grad, g_ref, rtol=1e-8)
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m for m in messages), messages


def test_exact_grad_rewards_all_zeros():
    """G4-fold (plan amendment 4): all-zeros rewards -- the degenerate case
    exercising the core's isfinite sweep and zero-scaling hooks. The
    reward-transformed moments are identically 0 for every theta, so the
    gradient is exactly [0, 0] (measured: exact and central-diff both
    bitwise zero, no decline)."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    zeros = jnp.zeros(4)
    model = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2,
        exact_moment_grad=True)
    with _capture_phasic_info_logs() as handler:
        grad = np.asarray(jax.grad(
            lambda th: jnp.sum(model(th, times, zeros)[1]))(theta))
    np.testing.assert_array_equal(grad, [0.0, 0.0])
    messages = [r.getMessage() for r in handler.records]
    assert not any("finite differences" in m for m in messages), messages


def test_exact_grad_rewards_with_fixed_mask():
    """G4-fold (plan I4.6): fixed_mask composes with the rewards-exact path
    -- the fixed dim's gradient is exactly 0 and the free dim is IDENTICAL
    (bitwise) to the unmasked exact gradient (measured: fixed [0.0, -8.25]
    vs unmasked [-24.0, -8.25])."""
    theta = jnp.asarray([1.0, 2.0])
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])
    model_fixed = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2,
        fixed_mask=[True, False], exact_moment_grad=True)
    model_free = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2,
        exact_moment_grad=True)
    g_fixed = np.asarray(jax.grad(
        lambda th: jnp.sum(model_fixed(th, times, rewards)[1]))(theta))
    g_free = np.asarray(jax.grad(
        lambda th: jnp.sum(model_free(th, times, rewards)[1]))(theta))
    assert g_fixed[0] == 0.0
    assert g_fixed[1] == g_free[1]
    np.testing.assert_array_equal(g_free, [-24.0, -8.25])


def test_exact_grad_rewards_vmap_jit_composition():
    """G4-fold (plan I4.7): the rewards-exact path under SVGD's actual
    calling pattern -- vmap(grad) over a particle batch must equal the
    per-particle grads BITWISE (expand_dims callback batching), and
    jit(vmap(grad)) must equal vmap(grad) bitwise (measured: both exactly
    equal, G4 probes P5/P6)."""
    times = jnp.asarray([1.0, 2.0])
    rewards = jnp.asarray([0.0, 2.0, 3.0, 0.5])
    model = Graph.pmf_and_moments_from_graph(
        _build_three_stage(), nr_moments=2, discrete=False, theta_dim=2,
        exact_moment_grad=True)
    loss = lambda th: jnp.sum(model(th, times, rewards)[1])  # noqa: E731
    particles = jnp.asarray([[1.0, 2.0], [1.5, 1.0], [0.7, 2.5]])

    g_vmap = np.asarray(jax.vmap(jax.grad(loss))(particles))
    g_per = np.stack([np.asarray(jax.grad(loss)(p)) for p in particles])
    g_jit = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(particles))

    np.testing.assert_array_equal(g_vmap, g_per)
    np.testing.assert_array_equal(g_jit, g_vmap)
    # anchor the first particle to the known exact value
    np.testing.assert_array_equal(g_vmap[0], [-24.0, -8.25])
