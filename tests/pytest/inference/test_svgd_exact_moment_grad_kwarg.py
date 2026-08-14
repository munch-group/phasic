"""
Batch D Tier-1 / D.4: `Graph.svgd(exact_moment_grad=...)`.

Contract (master plan §6-D.4; rule R29 in svgd_config.py; relaxed by
Batch A, 2026-08-14):
- None (default): not forwarded — byte-identical behavior.
- Explicit True/False: forwarded to pmf_and_moments_from_graph on the
  moments leaves — no-rewards (leaf 5) and, since Batch A, 1-D rewards
  (the exact moments adjoint supports 1-D rewards now).
- Explicit value on any other leaf (rewards 2-D, epoch_starts,
  joint-probability): SvgdConfigError (a ValueError) at validation time —
  never a silently inert kwarg. (Builder-level static declines that depend
  on graph properties — the callback mode [formula is covered since
  Batch B, bar its own static declines: discrete×formula and the
  lazy-decoupled theta-dimension class], discrete+rewards — remain
  accepted-but-INFO-logged; R29 polices leaf routing only, the Batch A G4
  disposition.)
"""
from __future__ import annotations

import contextlib
import logging
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, StateIndexer, Property, with_ipv
from phasic.exceptions import SvgdConfigError

jnp = pytest.importorskip("jax.numpy")
jax = pytest.importorskip("jax")

_pkg_dir = phasic._get_package_dir()
_HAS_SOURCES = (_pkg_dir / "src" / "cpp" / "phasiccpp.cpp").exists()
requires_sources = pytest.mark.skipif(
    not _HAS_SOURCES,
    reason="model JIT compilation requires the C/C++ sources on disk",
)


def _param_graph():
    g = phasic.Graph(1)
    s = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v3, 1.0)
    v3.add_edge(v2, [2.0, 3.0])
    v2.add_edge(v1, [1.0, 2.0])
    return g


def _make_joint_graph(discrete):
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=2)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 2

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2,
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            out.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return out

    base = Graph(cb, indexer=indexer)
    return base.joint_prob_graph(
        mutation_rate=1.0, tot_reward_limit=5, discrete=discrete,
    )


TIMES = np.array([0.2, 0.5, 0.9, 1.4])


class TestR29Rejections:
    """Explicit exact_moment_grad on unsupported leaves raises pre-construction."""

    # match on "exact_moment_grad" so these pin R29 specifically — a future
    # earlier rule mentioning the leaf word alone cannot mask an untested
    # R29 (G4 diff-review finding 5).

    def test_rewards_1d_honored(self):
        # Batch A FATE FLIP (user decision 2026-08-14, the bundled
        # opt-out): explicit exact_moment_grad is now HONORED on the
        # 1-D-rewards leaf -- False = FD opt-out, True = the (new)
        # default made explicit. Both must run to completion (the True
        # leg was G4-fold finding MINOR 8: previously only False ran).
        g = _param_graph()
        fit = g.svgd(TIMES, rewards=np.ones(g.vertices_length()),
                     exact_moment_grad=False, n_iterations=1,
                     n_particles=4)
        assert np.all(np.isfinite(np.asarray(fit.particles)))
        opts = fit.effective_options(return_dict=True)
        assert opts['exact_moment_grad']['status'] == 'user'
        assert opts['exact_moment_grad']['value'] is False

        g_true = _param_graph()
        fit_true = g_true.svgd(TIMES,
                               rewards=np.ones(g_true.vertices_length()),
                               exact_moment_grad=True, n_iterations=1,
                               n_particles=4)
        assert np.all(np.isfinite(np.asarray(fit_true.particles)))
        opts_true = fit_true.effective_options(return_dict=True)
        assert opts_true['exact_moment_grad']['status'] == 'user'
        assert opts_true['exact_moment_grad']['value'] is True

    def test_rewards_2d_rejected(self):
        g = _param_graph()
        rw = np.ones((2, g.vertices_length()))
        with pytest.raises(SvgdConfigError,
                           match="exact_moment_grad.*rewards"):
            g.svgd(TIMES, rewards=rw, exact_moment_grad=True)

    def test_joint_prob_rejected(self):
        jpg = _make_joint_graph(discrete=True)
        with pytest.raises(SvgdConfigError,
                           match="exact_moment_grad.*joint-probability"):
            jpg.svgd(np.array([1, 2, 1]), exact_moment_grad=False)

    def test_joint_stop_prob_rejected(self):
        # A joint_stop_prob_graph() classifies as 'joint_stop_prob', NOT
        # 'joint_prob' — the R29 hole the G4 diff review found (finding 1).
        jsp = _make_joint_graph(discrete=False).joint_stop_prob_graph()
        with pytest.raises(SvgdConfigError,
                           match="exact_moment_grad.*joint_stop_prob"):
            jsp.svgd(np.array([1, 2, 1]), exact_moment_grad=False)

    def test_epoch_starts_rejected(self):
        jpg = _make_joint_graph(discrete=False)
        with pytest.raises(SvgdConfigError,
                           match="exact_moment_grad.*epoch_starts"):
            jpg.svgd(np.array([1, 2, 1]), epoch_starts=[0.0, 0.5],
                     exact_moment_grad=False)


def test_direct_svgd_constructor_gets_helpful_error():
    # Passing the kwarg to the low-level SVGD(model=...) path must point the
    # caller at Graph.svgd, not fall through to a bare TypeError (user-approved
    # one-token svgd.py change, G4 diff-review finding 4).
    from phasic.svgd import SVGD
    with pytest.raises(TypeError, match="Graph.svgd"):
        SVGD(model=lambda th, t: t, observed_data=np.array([1.0]),
             theta_dim=1, exact_moment_grad=False)


class _CollectingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture_phasic_info_logs():
    # 'phasic' logger has propagate=False, so caplog never sees it; attach
    # a handler directly (pattern from test_exact_grad_discrete.py).
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


@requires_sources
class TestLeaf5Forwarding:

    def _fit(self, **kw):
        g = _param_graph()
        return g.svgd(TIMES, n_iterations=1, n_particles=4, seed=0,
                      progress=False, **kw)

    def test_explicit_false_reaches_callee(self):
        # Explicit False is the only forwarding-discriminating probe: the
        # callee's construction-time INFO line proves the kwarg arrived.
        with _capture_phasic_info_logs() as handler:
            self._fit(exact_moment_grad=False)
        msgs = [r.getMessage() for r in handler.records]
        assert any("exact_moment_grad=False" in m and "finite differences" in m
                   for m in msgs), msgs

    def test_default_none_fit_runs_and_ledger_default(self):
        svgd = self._fit()
        opts = svgd.effective_options(return_dict=True)
        assert 'exact_moment_grad' in opts
        assert opts['exact_moment_grad']['status'] == 'default'

    def test_explicit_value_surfaces_in_ledger_as_user(self):
        svgd = self._fit(exact_moment_grad=False)
        opts = svgd.effective_options(return_dict=True)
        assert opts['exact_moment_grad']['status'] == 'user'
        assert opts['exact_moment_grad']['value'] is False
