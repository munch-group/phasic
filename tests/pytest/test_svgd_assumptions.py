"""Gate 2 tests: SVGD assumption notices + effective_options() (UI overhaul).

Covers the once-per-call SvgdAssumptionWarning notices emitted when Graph.svgd
forces/assumes an option, the quiet_assumptions kwarg, and the options-ledger
introspection method svgd.effective_options().
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic  # noqa: F401  (enables jax_enable_x64 before any jax array)
from phasic import Graph, StateIndexer, Property, with_ipv, SVGD
from phasic.exceptions import SvgdAssumptionWarning


# --- fixtures ------------------------------------------------------------------

def _make_joint_prob_graph(*, discrete=False):
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=2)],
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = 2

    @with_ipv(ipv)
    def cb(state, indexer=None):
        out = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy(); new[i] -= 1; new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    base = Graph(cb, indexer=indexer)
    return base.joint_prob_graph(
        indexer, mutation_rate=1.0, reward_limit=10, discrete=discrete,
    )


def _joint_observations(jpg, n=200, seed=0):
    jpt = jpg.joint_prob_table()
    p = jpt['prob'] / jpt['prob'].sum()
    rng = np.random.RandomState(seed)
    sample = rng.choice(jpt.index.values, n, p=p.to_numpy())
    return jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()


def _fit_joint(jpg, obs, **kw):
    base = dict(fixed=[(1, 1.0)], n_iterations=2, n_particles=6,
                seed=0, progress=False)
    base.update(kw)
    return jpg.svgd(obs, **base)


def _build_two_stage_graph():
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge(v2, [1.0, 0.0])
    v2.add_edge(v1, [0.0, 1.0])
    return g


# --- notice tests --------------------------------------------------------------

def test_forcing_notices_emitted_by_default():
    jpg = _make_joint_prob_graph(discrete=False)
    obs = _joint_observations(jpg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _fit_joint(jpg, obs, discrete=False)  # explicit False -> overridden
    msgs = [str(ww.message) for ww in w
            if issubclass(ww.category, SvgdAssumptionWarning)]
    joined = "\n".join(msgs)
    assert any('discrete=True' in m and 'overridden' in m for m in msgs), joined
    assert any('joint_index=True' in m for m in msgs), joined
    assert any('probability-Jacobian' in m for m in msgs), joined


def test_quiet_assumptions_silences_notices():
    jpg = _make_joint_prob_graph(discrete=False)
    obs = _joint_observations(jpg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _fit_joint(jpg, obs, discrete=False, quiet_assumptions=True)
    assert not any(issubclass(ww.category, SvgdAssumptionWarning) for ww in w)


def test_standard_path_emits_no_forcing_notices():
    """A plain (non-joint) fit must not emit forcing notices: discrete/theta_dim
    inference is routine and recorded silently."""
    graph = _build_two_stage_graph()
    np.random.seed(0)
    data = (np.random.exponential(1.0 / 10.0, size=100)
            + np.random.exponential(1.0 / 1.0, size=100))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Pass an explicit prior so the DataPrior fallback path can't fire.
        graph.svgd(data, theta_dim=2, prior=lambda t: -0.5 * (t ** 2).sum(),
                   n_iterations=2, n_particles=6, seed=0, progress=False)
    assert not any(issubclass(ww.category, SvgdAssumptionWarning) for ww in w)


# --- effective_options() -------------------------------------------------------

def test_effective_options_provenance():
    jpg = _make_joint_prob_graph(discrete=False)
    obs = _joint_observations(jpg)
    result = _fit_joint(jpg, obs, discrete=False, quiet_assumptions=True)
    d = result.effective_options(return_dict=True)

    assert d['discrete']['status'] == 'forced'
    assert d['discrete']['user_value'] is False
    assert d['discrete']['value'] is True

    assert d['joint_index']['status'] == 'forced'
    assert d['joint_index']['user_value'] is False

    assert d['preconditioner']['status'] == 'inferred'
    assert 'probability-Jacobian' in d['preconditioner']['value']

    assert d['theta_dim']['status'] == 'inferred'
    assert d['theta_dim']['value'] == 2

    # n_iterations was passed (2 != default 1000) -> user; nr_moments default.
    assert d['n_iterations']['status'] == 'user'
    assert d['nr_moments']['status'] == 'default'


def test_effective_options_prints_table(capsys):
    jpg = _make_joint_prob_graph(discrete=False)
    obs = _joint_observations(jpg)
    result = _fit_joint(jpg, obs, quiet_assumptions=True)
    ret = result.effective_options()
    assert ret is None
    out = capsys.readouterr().out
    assert 'SVGD options in effect' in out
    assert 'discrete' in out and 'forced' in out


def test_direct_svgd_has_no_ledger(capsys):
    """The direct SVGD(model=...) path has no resolution layer, so the ledger is
    absent and effective_options() reports that instead of crashing."""
    graph = _build_two_stage_graph()
    model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2, discrete=False)
    np.random.seed(0)
    data = np.random.exponential(0.5, size=50)
    svgd = SVGD(model=model, observed_data=data, theta_dim=2,
                n_particles=4, n_iterations=1, seed=0, progress=False)
    ret = svgd.effective_options(return_dict=True)
    assert ret is None
    assert 'No options ledger' in capsys.readouterr().out
