"""Tests for the ``weight_formula=`` kwargs on ``Graph.update_weights`` and
``Graph.svgd`` (sugar over the ``graph.weight_formula`` property, mirroring the
existing ``callback=`` kwargs).

Pins: parity with the property form, one-shot semantics on update_weights,
save/restore on success+exception, preservation of a prior persistent formula,
mutual exclusion with callback (R22), and that — unlike callback — a formula is
ALLOWED under epoch_starts.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import math
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import Graph, ExpStepSize, StateIndexer, Property, with_ipv
from phasic.exceptions import SvgdConfigError


# --- simple fixtures --------------------------------------------------------
def g_1coeff():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [1.0])
    return g


def g_2coeff():
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, [2.0, 0.5])
    return g


def const_graph(rate):
    g = Graph(1)
    s = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    s.add_edge(v2, 1.0)
    v2.add_edge(v1, float(rate))
    return g


TIMES = np.linspace(0.1, 4.0, 40)
GRAN = 128
DATA = np.random.default_rng(0).exponential(scale=1.0 / (1.3 ** 2), size=200)
LR = ExpStepSize(first_step=0.02, last_step=0.002, tau=40.0)


def _lin_cb(theta, coeffs):
    return float(coeffs[0]) * float(theta[0])


# --- update_weights(weight_formula=...) -------------------------------------
def test_update_weights_kwarg_matches_property_and_reference():
    theta = np.array([0.6, 1.1])
    g1 = g_2coeff()
    g1.update_weights(theta, weight_formula="exp(c0*t0)")
    pdf_kwarg = np.asarray(g1.pdf(TIMES, granularity=GRAN))

    g2 = g_2coeff()
    g2.weight_formula = "exp(c0*t0)"
    g2.update_weights(theta)
    pdf_prop = np.asarray(g2.pdf(TIMES, granularity=GRAN))

    ref = np.asarray(const_graph(math.exp(2.0 * 0.6)).pdf(TIMES, granularity=GRAN))
    np.testing.assert_allclose(pdf_kwarg, pdf_prop, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pdf_kwarg, ref, rtol=0, atol=1e-12)


def test_update_weights_kwarg_is_one_shot():
    theta = np.array([0.6, 1.1])
    g = g_2coeff()
    pre_mode = g._weight_mode
    g.update_weights(theta, weight_formula="exp(c0*t0)")
    # weight_mode unchanged; no persistent formula left behind
    assert g._weight_mode == pre_mode
    assert g.weight_formula is None
    # a subsequent plain update_weights computes LINEAR weights (tape cleared)
    g.update_weights(theta)
    pdf_lin = np.asarray(g.pdf(TIMES, granularity=GRAN))
    ref_lin = np.asarray(
        const_graph(2.0 * 0.6 + 0.5 * 1.1).pdf(TIMES, granularity=GRAN))
    np.testing.assert_allclose(pdf_lin, ref_lin, rtol=0, atol=1e-12)


def test_update_weights_rejects_callback_and_formula_together():
    g = g_1coeff()
    with pytest.raises(ValueError, match="only one of"):
        g.update_weights(np.array([1.0]), callback=_lin_cb,
                         weight_formula="c0*t0")


def test_update_weights_kwarg_bad_formula_raises():
    g = g_1coeff()
    with pytest.raises(Exception):
        g.update_weights(np.array([1.0]), weight_formula="exp(t0,")  # syntax


# --- svgd(weight_formula=...) -----------------------------------------------
def test_svgd_kwarg_matches_property():
    kw = dict(theta_dim=1, n_iterations=25, n_particles=8, seed=42,
              positive_params=True, learning_rate=LR)
    g1 = g_1coeff()
    r1 = g1.svgd(DATA, weight_formula="(c0*t0)**2", **kw)
    g2 = g_1coeff()
    g2.weight_formula = "(c0*t0)**2"
    r2 = g2.svgd(DATA, **kw)
    np.testing.assert_allclose(np.asarray(r1.particles),
                               np.asarray(r2.particles), rtol=0, atol=1e-6)


def test_svgd_kwarg_restores_on_success():
    g = g_1coeff()
    pre_mode = g._weight_mode
    g.svgd(DATA, weight_formula="(c0*t0)**2", theta_dim=1, n_iterations=4,
           n_particles=4, seed=1, positive_params=True)
    assert g._weight_mode == pre_mode
    assert g.weight_formula is None


def test_svgd_kwarg_restores_on_exception():
    g = g_1coeff()
    pre_mode = g._weight_mode
    with pytest.raises(Exception):
        # compiles fine, but yields a negative weight at runtime -> C raises
        g.svgd(DATA, weight_formula="c0*t0 - 1000.0", theta_dim=1,
               n_iterations=2, n_particles=4, seed=1, positive_params=True)
    assert g._weight_mode == pre_mode
    assert g.weight_formula is None


def test_svgd_kwarg_preserves_prior_persistent_formula():
    g = g_1coeff()
    g.weight_formula = "(c0*t0)**2"            # persistent
    g.svgd(DATA, weight_formula="c0*t0", theta_dim=1, n_iterations=4,
           n_particles=4, seed=1, positive_params=True)
    # prior formula + mode restored
    assert g.weight_formula == "(c0*t0)**2"
    assert g._weight_mode == "formula"
    # and the prior live tape is restored: direct pdf uses (c0*t0)**2
    g.update_weights(np.array([1.5]))          # (1.0*1.5)**2 = 2.25
    pdf = np.asarray(g.pdf(TIMES, granularity=GRAN))
    ref = np.asarray(const_graph(2.25).pdf(TIMES, granularity=GRAN))
    np.testing.assert_allclose(pdf, ref, rtol=0, atol=1e-12)


def test_svgd_rejects_callback_and_formula_together():
    g = g_1coeff()
    with pytest.raises(SvgdConfigError, match="mutually exclusive"):
        g.svgd(DATA, callback=_lin_cb, weight_formula="c0*t0", theta_dim=1,
               n_iterations=2, n_particles=4, positive_params=True)


# --- daisy-chain: formula allowed under epoch_starts (callback is NOT) -------
MUT = 1e-4
THETA = 1 / 10_000
REWARD_LIMIT = 5


def _build_daisy_model():
    nr_samples = 4
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=nr_samples)])
    all_pairs = partial(combinations_with_replacement, r=2)

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state, indexer=None):
        out = []
        for i, j in all_pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair = state[i] * (state[j] - same) / (1 + same)
            out.append([new, [pair]])
        return out

    graph = Graph(coalescent_1param, indexer=indexer)
    np.random.seed(17)
    jpg_disc = graph.joint_prob_graph(
        indexer, reward_limit=REWARD_LIMIT, mutation_rate=MUT)
    jpg_disc.update_weights([THETA, MUT])
    jpt = jpg_disc.joint_prob_table()
    p = jpt['prob'] / jpt['prob'].sum()
    sample = np.random.choice(jpt.index.values, 300, p=p.to_numpy())
    obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    return graph, indexer, obs


@pytest.fixture(scope="module")
def daisy_model():
    return _build_daisy_model()


def test_svgd_formula_kwarg_allowed_with_epoch_starts(daisy_model):
    from phasic import LogGaussPrior, Adamelia
    graph, indexer, obs = daisy_model
    jpg = graph.joint_prob_graph(
        indexer, reward_limit=REWARD_LIMIT, mutation_rate=MUT, discrete=False)
    svgd = jpg.svgd(
        obs, weight_formula="c0*t0 + c1*t1", fixed=[(1, MUT)],
        prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5000]),
        n_iterations=3, n_particles=8, seed=0,
        optimizer=Adamelia(learning_rate=0.2), epoch_starts=[0, 0.5])
    assert np.all(np.isfinite(np.asarray(svgd.particles)))
    # one-shot: no persistent formula left on the graph
    assert jpg.weight_formula is None


def test_svgd_callback_kwarg_still_rejected_with_epoch_starts(daisy_model):
    graph, indexer, obs = daisy_model
    jpg = graph.joint_prob_graph(
        indexer, reward_limit=REWARD_LIMIT, mutation_rate=MUT, discrete=False)
    with pytest.raises(SvgdConfigError, match="callback.*not supported.*epoch_starts"):
        jpg.svgd(obs, callback=_lin_cb, fixed=[(1, MUT)],
                 n_iterations=2, n_particles=4, epoch_starts=[0, 0.5])
