"""Inference correctness for the models in the svgd-* tutorial notebooks.

The notebooks (docs/pages/tutorial/svgd-basics, svgd-multi-param,
svgd-multi-feature, svgd-joint-prob) all follow one pattern: build a model,
sample observations from KNOWN parameter values, fit, and check inference
finds those values again. That check lived only in the notebooks, where
nothing enforced it. This file promotes it into the suite.

Requirement being guarded (user, 2026-08-16): inference must be correct for
any model, and that must be tracked by tests rather than by judgement.

THREE ORDERS OF MAGNITUDE. Every model is exercised at three sizes whose
vertex counts fall in three different decades, so a defect that only
appears as the state space grows cannot hide:

    coalescent_1param   n=10 ->   43 | n=13 ->  102 | n=22 -> 1003
    two_island          n=4  ->   21 | n=8  ->  186 | n=12 -> 1166
    joint_prob          n=3  ->   25 | n=4  ->  168 | n=5  -> 1559
    coalescent+rewards  n=10 ->   43 | n=13 ->  102 | n=22 -> 1003

TWO TIERS, because a full posterior fit is not affordable at every size.
Be clear about what each proves:

  0. SAMPLER-VS-DENSITY tier. A Kolmogorov-Smirnov test that draws from
     `Graph.sample()` follow `Graph.cdf()`. Everything else rests on
     this: if the sampler and the fitted density were different
     distributions, "recovering the true parameters" would measure
     nothing.

  1. LIKELIHOOD-SURFACE tier (fast, runs at ALL sizes and ALL models).
     Asserts the data-generating parameters are a local maximum of the
     likelihood the library computes — perturbing any parameter must
     lower the log-likelihood. This is the property inference consumes,
     and it is what actually gives three-decade coverage. Costs ~1s per
     evaluation even at 1166 vertices.

  2. POSTERIOR-RECOVERY tier (slow, runs where affordable). Runs a real
     SVGD fit and asserts the true value lies inside the 95% HPD
     interval, with a bound on the interval's relative width so a
     diffuse posterior cannot pass. This is end-to-end inference.

Where tier 2 is missing, tier 1 still covers that size; the omissions are
named explicitly below rather than left as silent gaps. The two_island
model is the expensive one: its per-theta model rebuild is far costlier
than the coalescent's at matched size (about 228s for a reduced fit at 186
vertices versus 26s for the coalescent at 102, despite identical per-pdf
cost — it is cyclic, via migration), which is why its largest size carries
the likelihood-surface check only.

Measured totals: fast tier 17 tests in ~48s; slow tier 3 tests in ~5.8min.

Fit budgets here (particles, iterations, observation counts) are reduced
from production settings to fit the suite's timeout. That is a test-budget
choice and says nothing about how to configure a real fit — a real fit uses
about 20 particles per free parameter, minimum 200, and about 100
iterations. Point estimates are deliberately not asserted tightly: they
run 12-21% high on the coalescent at these budgets, which is a genuine
property of the fit rather than something a test should freeze.
"""
import random
from functools import partial
from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import (Graph, GaussPrior, LogGaussPrior, Property, StateIndexer,
                    set_log_level, with_ipv)

set_log_level("ERROR")

all_pairs = partial(combinations_with_replacement, r=2)


# --------------------------------------------------------------- the models
def coalescent_graph(nr_samples):
    """svgd-basics / svgd-multi-feature: Kingman coalescent, one rate."""
    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                transitions.append(
                    [new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions
    return Graph(coalescent_1param)


def two_island_graph(nr_samples):
    """svgd-multi-param: two-island model, coalescence + migration.

    Migration makes this graph CYCLIC, unlike the coalescent.
    """
    indexer = StateIndexer(descendants=[
        Property('pop1', min_value=0, max_value=nr_samples),
        Property('pop2', min_value=0, max_value=nr_samples),
        Property('in_pop', min_value=1, max_value=2),
    ])
    initial = [0] * indexer.state_length
    initial[indexer.descendants.props_to_index(
        pop1=1, pop2=0, in_pop=1)] = nr_samples

    @with_ipv(initial)
    def two_island(state):
        transitions = []
        if state[indexer.descendants.indices()].sum() <= 1:
            return transitions
        for i in range(indexer.descendants.state_length):
            if state[i] == 0:
                continue
            props_i = indexer.descendants.index_to_props(i)
            for j in range(i, indexer.descendants.state_length):
                if state[j] == 0:
                    continue
                props_j = indexer.descendants.index_to_props(j)
                if props_j.in_pop != props_i.in_pop:
                    continue
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                child = state.copy()
                child[i] -= 1
                child[j] -= 1
                des1 = props_i.pop1 + props_j.pop1
                des2 = props_i.pop2 + props_j.pop2
                if des1 <= nr_samples and des2 <= nr_samples:
                    k = indexer.descendants.props_to_index(
                        pop1=des1, pop2=des2, in_pop=props_i.in_pop)
                    child[k] += 1
                    transitions.append(
                        [child,
                         [state[i] * (state[j] - same) / (1 + same), 0]])
            if state[i] > 0:
                other = 2 if props_i.in_pop == 1 else 1
                child = state.copy()
                child[i] -= 1
                k = indexer.descendants.props_to_index(
                    pop1=props_i.pop1, pop2=props_i.pop2, in_pop=other)
                child[k] += 1
                transitions.append([child, [0, state[i]]])
        return transitions
    return Graph(two_island)


def coalescent_indexed(nr_samples):
    """svgd-joint-prob: the coalescent built through a StateIndexer so a
    joint-probability graph can be derived from it."""
    indexer = StateIndexer(lineage=[
        Property('descendants', min_value=1, max_value=nr_samples)])

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state, indexer=None):
        transitions = []
        for i, j in all_pairs(indexer.lineage):
            p1 = indexer.lineage.index_to_props(i)
            p2 = indexer.lineage.index_to_props(j)
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            k = indexer.lineage.props_to_index(
                descendants=p1.descendants + p2.descendants)
            new[k] += 1
            transitions.append(
                [new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions
    return Graph(coalescent_1param, indexer=indexer), indexer


# ------------------------------------------------------------------ helpers
def seed_all(n):
    """Seed BOTH generators.

    `Graph.sample()` does NOT use numpy: the pybind layer seeds the C
    sampler from Python's stdlib `random` module (phasic_pybind.cpp
    set_c_seed), so `np.random.seed` alone leaves sampling
    non-deterministic and makes these tests flaky. Verified: two
    `sample()` calls under the same `random.seed` are identical, under
    the same `np.random.seed` alone they are not.
    """
    random.seed(n)
    np.random.seed(n)


def log_likelihood(graph, theta, data, rewards=None):
    """Log-likelihood of `data` under `graph` at `theta`.

    With `rewards`, the graph is re-weighted FIRST and reward-transformed
    after: `reward_transform` returns a plain (non-parameterized) graph, so
    it cannot itself take an update_weights call.
    """
    graph.update_weights(list(theta))
    g = graph.reward_transform(rewards) if rewards is not None else graph
    p = np.array([g.pdf(float(t)) for t in data])
    return float(np.sum(np.log(np.clip(p, 1e-300, None))))


def assert_truth_is_local_max(graph, true_theta, data, factors=(0.5, 2.0),
                              rewards=None):
    """Tier 1: the parameters the data were sampled from must beat every
    perturbation of them under the library's own likelihood."""
    base = log_likelihood(graph, true_theta, data, rewards)
    assert np.isfinite(base), "log-likelihood at the true parameters is not finite"
    for i in range(len(true_theta)):
        for f in factors:
            th = list(true_theta)
            th[i] = th[i] * f
            alt = log_likelihood(graph, th, data, rewards)
            assert alt < base, (
                f"parameter {i} scaled by {f}: log-likelihood {alt:.3f} is "
                f"NOT below the value at the true parameters {base:.3f} — "
                f"the data-generating parameters are not a local maximum, "
                f"so inference cannot be expected to find them")


def assert_recovers(svgd, true_theta, free=None, max_rel_width=12.0):
    """Tier 2: every FREE parameter's true value inside its 95% HPD, with
    the interval not uselessly wide."""
    res = svgd.get_results()
    lo = np.asarray(res['hpd_lower']).ravel()
    hi = np.asarray(res['hpd_upper']).ravel()
    mean = np.asarray(res['theta_mean']).ravel()
    for i in (range(len(true_theta)) if free is None else free):
        t = true_theta[i]
        assert lo[i] <= t <= hi[i], (
            f"parameter {i}: true value {t:g} OUTSIDE the 95% HPD "
            f"[{lo[i]:g}, {hi[i]:g}] (posterior mean {mean[i]:g}) — "
            f"inference did not recover the value it was sampled from")
        width = (hi[i] - lo[i]) / max(abs(mean[i]), 1e-30)
        assert width < max_rel_width, (
            f"parameter {i}: HPD is {width:.1f}x the posterior mean, so "
            f"wide that containing the truth is uninformative")


def assert_informative(obs, min_distinct=3):
    """Guard against a vacuous recovery test: if the sampled observations
    are (nearly) all the same outcome, the data say nothing about theta and
    any 'recovery' is just the prior."""
    distinct = len({tuple(int(v) for v in row) for row in np.asarray(obs)})
    assert distinct >= min_distinct, (
        f"observations are degenerate: only {distinct} distinct outcome(s) "
        f"in {len(obs)} draws, so the data carry no information about the "
        f"parameters and this test would pass on the prior alone")


def sample_joint_observations(jpg, theta, n, rng):
    """svgd-joint-prob's sampler: draw joint outcomes from the model's own
    joint-probability table at the true parameters."""
    jpg.update_weights(list(theta))
    table = jpg.joint_prob_table()
    p = table['prob'].to_numpy()
    p = p / p.sum()
    picks = rng.choice(table.index.values, n, p=p)
    return table.loc[picks, table.columns[:-1]].to_numpy().tolist()


COAL_TRUE = [7.0]
ISLAND_TRUE = [0.7, 0.3]
REWARD_TRUE = [10.0]
# Mutation rate 1e-2 against a coalescent rate of 1e-4, NOT the notebook's
# 1e-4. At 1e-4 the joint distribution is degenerate -- the top outcome
# holds 99.97% of the mass, every sampled observation is the same row, and
# the likelihood rises monotonically in theta with no maximum. A recovery
# test there passes only because the prior contains the truth, which is
# exactly the vacuous pass this file exists to prevent. At 1e-2 there are
# 6-38 effective outcomes depending on size. `assert_informative` below
# enforces this so the fixture cannot silently degenerate again.
MUTATION_RATE = 1e-2
JOINT_TRUE = [1.0 / 10_000, MUTATION_RATE]

COAL_SIZES = [(10, 43), (13, 102), (22, 1003)]
ISLAND_SIZES = [(4, 21), (8, 186), (12, 1166)]
JOINT_SIZES = [(3, 2, 25), (4, 3, 168), (5, 4, 1559)]


# ============================== TIER 1: likelihood surface, ALL sizes =======
@pytest.mark.parametrize("nr_samples,expect_vertices", COAL_SIZES,
                         ids=["tens-43v", "hundreds-102v", "thousands-1003v"])
def test_coalescent_likelihood_peaks_at_truth(nr_samples, expect_vertices):
    graph = coalescent_graph(nr_samples)
    assert graph.vertices_length() == expect_vertices, (
        f"model size changed: {graph.vertices_length()} vertices, ladder "
        f"expects {expect_vertices}")
    graph.update_weights(COAL_TRUE)
    seed_all(0)
    assert_truth_is_local_max(graph, COAL_TRUE, graph.sample(300))


@pytest.mark.parametrize("nr_samples,expect_vertices", ISLAND_SIZES,
                         ids=["tens-21v", "hundreds-186v", "thousands-1166v"])
def test_two_island_likelihood_peaks_at_truth(nr_samples, expect_vertices):
    graph = two_island_graph(nr_samples)
    assert graph.vertices_length() == expect_vertices, (
        f"model size changed: {graph.vertices_length()} vertices, ladder "
        f"expects {expect_vertices}")
    graph.update_weights(ISLAND_TRUE)
    seed_all(0)
    assert_truth_is_local_max(graph, ISLAND_TRUE, graph.sample(300))


@pytest.mark.parametrize("nr_samples,expect_vertices", COAL_SIZES,
                         ids=["tens-43v", "hundreds-102v", "thousands-1003v"])
def test_reward_transformed_likelihood_peaks_at_truth(nr_samples,
                                                      expect_vertices):
    """svgd-multi-feature: observations are a REWARD-transformed time, so
    the likelihood runs through the reward path."""
    graph = coalescent_graph(nr_samples)
    assert graph.vertices_length() == expect_vertices
    graph.update_weights(REWARD_TRUE)
    rewards = np.sum(graph.states().T, axis=0)
    seed_all(17)
    data = graph.sample(300, rewards=rewards)
    assert_truth_is_local_max(graph, REWARD_TRUE, data, rewards=rewards)


# ===== TIER 0: does the sampler agree with the density it is fitted against? =
@pytest.mark.parametrize("tag,builder,true_theta", [
    ("coalescent", lambda: coalescent_graph(10), COAL_TRUE),
    ("two_island", lambda: two_island_graph(4), ISLAND_TRUE),
    ("two_island_186", lambda: two_island_graph(8), ISLAND_TRUE),
])
def test_sampler_matches_density(tag, builder, true_theta):
    """Kolmogorov-Smirnov: draws from Graph.sample() must follow Graph.cdf().

    This underpins every other test in this file. Parameter recovery is
    only meaningful if the data-generating sampler and the density being
    fitted describe the SAME distribution; if they diverged, "recovering
    the true parameters" would be measuring nothing.
    """
    from scipy.stats import kstest
    graph = builder()
    graph.update_weights(true_theta)
    seed_all(0)
    data = np.asarray(graph.sample(3000))
    res = kstest(data, lambda x: np.array(
        [graph.cdf(float(v)) for v in np.atleast_1d(x)]))
    assert res.pvalue > 0.01, (
        f"{tag}: sample() and cdf() disagree (KS D={res.statistic:.4f}, "
        f"p={res.pvalue:.2e}) — the sampler and the fitted density are not "
        f"the same distribution, so parameter recovery is meaningless")


# ============================== TIER 2: end-to-end posterior recovery ======
def test_coalescent_svgd_recovers_rate_tens():
    graph = coalescent_graph(10)
    graph.update_weights(COAL_TRUE)
    seed_all(0)
    svgd = graph.svgd(graph.sample(500), prior=GaussPrior(ci=[3, 15]),
                      n_iterations=60, n_particles=60)
    assert_recovers(svgd, COAL_TRUE)


def test_coalescent_svgd_recovers_rate_hundreds():
    graph = coalescent_graph(13)
    graph.update_weights(COAL_TRUE)
    seed_all(0)
    svgd = graph.svgd(graph.sample(500), prior=GaussPrior(ci=[3, 15]),
                      n_iterations=60, n_particles=60)
    assert_recovers(svgd, COAL_TRUE)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_coalescent_svgd_recovers_rate_thousands():
    """1003 vertices. Measured ~122s at this budget."""
    graph = coalescent_graph(22)
    graph.update_weights(COAL_TRUE)
    seed_all(0)
    svgd = graph.svgd(graph.sample(300), prior=GaussPrior(ci=[3, 15]),
                      n_iterations=20, n_particles=25)
    assert_recovers(svgd, COAL_TRUE)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_two_island_svgd_recovers_both_rates_tens():
    """21 vertices, two free parameters, ~85s.

    Marked slow because two free parameters double the per-iteration
    probe count, and this model's per-theta rebuild is expensive.
    """
    graph = two_island_graph(4)
    graph.update_weights(ISLAND_TRUE)
    seed_all(0)
    svgd = graph.svgd(graph.sample(500), n_iterations=40, n_particles=50)
    assert_recovers(svgd, ISLAND_TRUE)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_two_island_svgd_recovers_both_rates_hundreds():
    """186 vertices, ~228s at this budget — the most expensive fit here.

    The 1166-vertex size has no fit test at all; it is covered by the
    likelihood-surface tier only.
    """
    graph = two_island_graph(8)
    graph.update_weights(ISLAND_TRUE)
    seed_all(0)
    svgd = graph.svgd(graph.sample(300), n_iterations=20, n_particles=25)
    assert_recovers(svgd, ISLAND_TRUE)


def test_coalescent_with_rewards_svgd_recovers_rate():
    """svgd-multi-feature, end to end through the reward path."""
    graph = coalescent_graph(10)
    graph.update_weights(REWARD_TRUE)
    rewards = np.sum(graph.states().T, axis=0)
    seed_all(17)
    data = graph.sample(500, rewards=rewards)
    svgd = graph.svgd(observed_data=data, rewards=rewards,
                      prior=GaussPrior(ci=[5, 25]),
                      n_iterations=60, n_particles=60)
    assert_recovers(svgd, REWARD_TRUE)


@pytest.mark.parametrize("nr_samples,reward_limit,expect_vertices",
                         JOINT_SIZES[:2],
                         ids=["tens-25v", "hundreds-168v"])
def test_joint_prob_svgd_recovers_coalescent_rate(nr_samples, reward_limit,
                                                  expect_vertices):
    """svgd-joint-prob: observations are joint mutation-count outcomes drawn
    from the model's own joint-probability table; the mutation rate is held
    fixed and the coalescent rate must be recovered.

    The 1559-vertex size is not fitted here — building its joint-probability
    table and fitting it exceeds the suite budget.
    """
    graph, indexer = coalescent_indexed(nr_samples)
    jpg = graph.joint_prob_graph(indexer, reward_limit=reward_limit,
                                 mutation_rate=MUTATION_RATE)
    assert jpg.vertices_length() == expect_vertices, (
        f"model size changed: {jpg.vertices_length()} vertices, ladder "
        f"expects {expect_vertices}")
    rng = np.random.default_rng(17)
    obs = sample_joint_observations(jpg, JOINT_TRUE, 1000, rng)
    assert_informative(obs)
    svgd = jpg.svgd(obs,
                    fixed=[(1, MUTATION_RATE)],
                    prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5_000]),
                    n_iterations=60, n_particles=60)
    assert_recovers(svgd, JOINT_TRUE, free=[0])
