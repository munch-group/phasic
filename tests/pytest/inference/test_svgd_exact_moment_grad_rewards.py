"""Batch A -- svgd rewards leaves x exact_moment_grad (the bundled opt-out).

User decision 2026-08-14: A relaxed R29's 1-D-rewards arm and forwards
exact_moment_grad on the 1-D-rewards svgd leaf, so the default flip
(rewards runs going exact) ships WITH its opt-out. Batch G.2
(2026-08-15, full symmetry) extended the same contract to the
2-D/multivariate leaf (tested below via the sparse route).
"""
import numpy as np
import pytest

import phasic
from phasic import Graph, set_log_level
from phasic.exceptions import SvgdConfigError

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

set_log_level("WARNING")


def _param_graph():
    g = Graph(1)
    s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]
    va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    v[0].add_edge_parameterized(v[1], 0.0, [1.0, 0.0])
    v[1].add_edge_parameterized(v[2], 0.0, [0.0, 1.0])
    v[2].add_edge_parameterized(va, 0.0, [1.0, 1.0])
    return g


TIMES = np.asarray([0.5, 1.0, 1.5, 2.0])


def test_forwarding_discrimination_rewards_leaf(monkeypatch):
    """explicit False must ARRIVE at the callee; None must be ABSENT --
    a regression to not-forwarding would silently resurrect the
    pre-A window (default True with no opt-out)."""
    seen = []
    orig = Graph.pmf_and_moments_from_graph.__func__

    def spy(cls, *a, **kw):
        seen.append(dict(kw))
        return orig(cls, *a, **kw)

    monkeypatch.setattr(Graph, "pmf_and_moments_from_graph",
                        classmethod(spy))
    g = _param_graph()
    rw = np.ones(g.vertices_length())
    g.svgd(TIMES, rewards=rw, exact_moment_grad=False,
           n_iterations=1, n_particles=4)
    assert seen and seen[-1].get("exact_moment_grad") is False
    seen.clear()
    g2 = _param_graph()
    g2.svgd(TIMES, rewards=rw, n_iterations=1, n_particles=4)
    assert seen and "exact_moment_grad" not in seen[-1]


def test_frontdoor_rewards_default_runs_exact(monkeypatch):
    """The E lesson: front-door svgd + rewards, default kwarg -- the
    particle cloud samples its real init distribution; the fit must
    complete finite with the exact path ENGAGING.

    G4-fold hardening (finding M2): the original assertion grepped a log
    string this batch DELETED, so it passed even with the exact path
    monkeypatched to decline every call. Now: (1) a spy on the C Jacobian
    entry point asserts >= n_particles calls with nonempty rewards
    (measured: 13 = 6 particles x 2 iterations + 1 precompile, all 13
    with rewards); (2) the log filter greps the LIVE static-decline
    messages ("rewards" AND "finite differences" -- matches both Batch A
    static declines, not the per-theta conditioning message, which random
    particle inits may legitimately trigger)."""
    import logging

    class _H(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, r):
            self.records.append(r)

    calls = []
    orig = Graph._moments_grad_theta

    def spy(self, K, *args, **kwargs):
        rw_arg = kwargs.get('rewards')
        calls.append(0 if rw_arg is None else len(rw_arg))
        return orig(self, K, *args, **kwargs)

    monkeypatch.setattr(Graph, "_moments_grad_theta", spy)
    g = _param_graph()
    rw = np.ones(g.vertices_length()); rw[1] = 2.0; rw[2] = 0.0
    lg = logging.getLogger("phasic")
    h = _H(); h.setLevel(logging.INFO)
    prev = lg.level
    lg.addHandler(h); lg.setLevel(logging.INFO)
    try:
        fit = g.svgd(TIMES, rewards=rw, n_iterations=2, n_particles=6)
    finally:
        lg.removeHandler(h); lg.setLevel(prev)
    assert np.all(np.isfinite(np.asarray(fit.particles)))
    msgs = [r.getMessage() for r in h.records]
    assert not any("rewards" in m and "finite differences" in m
                   for m in msgs), msgs
    with_rewards = [n for n in calls if n > 0]
    assert len(with_rewards) >= 6, (
        f"exact C Jacobian ran {len(with_rewards)} times with rewards "
        f"(expected >= n_particles=6; measured 13)")


def test_rewards_2d_honored_with_spy(monkeypatch):
    """Batch G.2 FATE FLIP (the 'G.2 message' this test previously
    pinned is retired -- user decision 2026-08-15, full symmetry):
    explicit exact_moment_grad on the 2-D leaf is forwarded to the
    multivariate wrapper. False => ZERO exact C calls; default (None)
    => full-size successes per feature (the success-floor
    discipline)."""
    from phasic.svgd import dense_to_sparse
    calls = []
    orig = Graph._moments_grad_theta

    def spy(self, K, *a, **kw):
        r = orig(self, K, *a, **kw)
        calls.append(np.asarray(r).size)
        return r

    monkeypatch.setattr(Graph, "_moments_grad_theta", spy)
    g = _param_graph()
    n = g.vertices_length()
    rw = np.ones((2, n)); rw[1, 1] = 2.0
    obs = dense_to_sparse(np.asarray([[0.5, 1.0], [1.0, 1.5],
                                      [1.5, 2.0]]))
    fit_off = g.svgd(obs, rewards=rw, exact_moment_grad=False,
                     n_iterations=1, n_particles=4)
    assert np.all(np.isfinite(np.asarray(fit_off.particles)))
    assert len(calls) == 0, f"opt-out leaked {len(calls)} exact calls"

    g2 = _param_graph()
    fit_on = g2.svgd(obs, rewards=rw, n_iterations=1, n_particles=4)
    assert np.all(np.isfinite(np.asarray(fit_on.particles)))
    full = [s for s in calls if s > 0]
    assert len(full) >= 4, (
        f"default: {len(full)} full-size successes (want >= n_particles)")
