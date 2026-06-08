"""Batch 3 (daisy-chain): weight_formula on the daisy-chain / sojourn SVGD
path (epoch_starts=...), where Python callbacks are forbidden by validator
rule R21. This is the highest-value scenario for weight_formula -- the only
way to use a non-inner-product weight on the recorded-trace fast path.

The formula tape is propagated onto the derived joint_stop_prob_graph /
joint_sojourn_graph (so it rides their serialize() into the daisy FFI). We
check it is BOTH honored+correct (a linear-equivalent formula reproduces the
linear posterior) AND actually applied (a scaled formula diverges from linear,
ruling out a silent linear fallback).

Model is the proven coalescent joint-prob fixture from
test_daisy_chain_svgd_kernel.py.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

import phasic
from phasic import (
    Graph, with_ipv, LogGaussPrior, Adamelia, StateIndexer, Property,
)

MUT = 1e-4
THETA = 1 / 10_000
REWARD_LIMIT = 5


def _build_model():
    nr_samples = 4
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=nr_samples)])
    all_pairs = partial(combinations_with_replacement, r=2)

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state, indexer=None):
        transitions = []
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
            transitions.append([new, [pair]])
        return transitions

    graph = Graph(coalescent_1param, indexer=indexer)
    return graph, indexer


def _observations(graph, indexer):
    np.random.seed(17)
    jpg_disc = graph.joint_prob_graph(
        indexer, reward_limit=REWARD_LIMIT, mutation_rate=MUT)
    jpg_disc.update_weights([THETA, MUT])
    jpt = jpg_disc.joint_prob_table()
    p = jpt['prob'] / jpt['prob'].sum()
    sample = np.random.choice(jpt.index.values, 400, p=p.to_numpy())
    return jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()


def _run_daisy_svgd(graph, indexer, observations, formula=None):
    jpg = graph.joint_prob_graph(
        indexer, reward_limit=REWARD_LIMIT, mutation_rate=MUT, discrete=False)
    if formula is not None:
        jpg.weight_formula = formula
    svgd = jpg.svgd(
        observations,
        fixed=[(1, MUT)],
        prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5000]),
        n_iterations=3,
        n_particles=8,
        seed=0,
        optimizer=Adamelia(learning_rate=0.2),
        epoch_starts=[0, 0.5],
    )
    return np.asarray(svgd.particles)


@pytest.fixture(scope="module")
def model_and_obs():
    graph, indexer = _build_model()
    obs = _observations(graph, indexer)
    return graph, indexer, obs


def test_daisy_formula_not_rejected_and_runs(model_and_obs):
    graph, indexer, obs = model_and_obs
    parts = _run_daisy_svgd(graph, indexer, obs, formula="c0*t0 + c1*t1")
    assert parts.shape[0] == 8
    assert np.all(np.isfinite(parts))


def test_daisy_formula_linear_equivalent_matches(model_and_obs):
    # "c0*t0 + c1*t1" IS the linear inner product -> identical posterior.
    graph, indexer, obs = model_and_obs
    lin = _run_daisy_svgd(graph, indexer, obs, formula=None)
    frm = _run_daisy_svgd(graph, indexer, obs, formula="c0*t0 + c1*t1")
    np.testing.assert_allclose(frm, lin, rtol=0, atol=1e-6)


def test_daisy_formula_is_actually_applied(model_and_obs):
    # A scaled formula must change the result -> proves the tape reaches the
    # daisy FFI and is honored (not silently ignored as linear).
    graph, indexer, obs = model_and_obs
    lin = _run_daisy_svgd(graph, indexer, obs, formula=None)
    scaled = _run_daisy_svgd(graph, indexer, obs, formula="2.0*c0*t0 + c1*t1")
    assert not np.allclose(scaled, lin, atol=1e-2), (
        "scaled formula produced the same posterior as linear -> formula was "
        "ignored on the daisy-chain path")


# --- joint_index (non-epoch) path: weight_formula on the joint-prob graph -----
def _run_joint_svgd(graph, indexer, observations, formula=None):
    from phasic import LogGaussPrior, Adamelia
    jpg = graph.joint_prob_graph(
        indexer, reward_limit=REWARD_LIMIT, mutation_rate=MUT, discrete=False)
    if formula is not None:
        jpg.weight_formula = formula
    svgd = jpg.svgd(
        observations, fixed=[(1, MUT)],
        prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5000]),
        n_iterations=3, n_particles=8, seed=0,
        optimizer=Adamelia(learning_rate=0.2),
        preconditioner=None,            # no epoch_starts -> joint_index path
    )
    return np.asarray(svgd.particles)


def test_joint_index_formula_linear_equivalent_matches(model_and_obs):
    graph, indexer, obs = model_and_obs
    lin = _run_joint_svgd(graph, indexer, obs, formula=None)
    frm = _run_joint_svgd(graph, indexer, obs, formula="c0*t0 + c1*t1")
    np.testing.assert_allclose(frm, lin, rtol=0, atol=1e-6)


def test_joint_index_formula_is_actually_applied(model_and_obs):
    graph, indexer, obs = model_and_obs
    lin = _run_joint_svgd(graph, indexer, obs, formula=None)
    scaled = _run_joint_svgd(graph, indexer, obs, formula="2.0*c0*t0 + c1*t1")
    assert not np.allclose(scaled, lin, atol=1e-2), (
        "scaled formula produced the same posterior as linear -> formula was "
        "ignored on the joint_index path")
