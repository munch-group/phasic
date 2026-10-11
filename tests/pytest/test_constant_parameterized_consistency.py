"""Constant vs parameterized models: consistent user-facing behaviour.

Plan: constant-parameterized-consistency-plan.md.

Part 2 (this section): every entry point that takes theta refuses a
constant model with one standard message. A constant model is one built
with scalar rates (add_edge(v, 3.0), or a callback returning floats), or a
derived graph with baked-in weights (reward_transform() output). Before
this, several entry points (svgd, mcmc, method_of_moments, joint_prob_graph
...) silently treated it as a 1-parameter model scaling every rate.
"""
import numpy as np
import pytest

import phasic
from phasic import Graph, with_ipv

jnp = pytest.importorskip("jax.numpy")

GUARD_MSG = "must have parameterized edges"
N = 4


def _coalescent(state, *, as_list):
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
            r = state[i] * (state[j] - same) / (1 + same)
            transitions.append([new, [r] if as_list else r])
    return transitions


@with_ipv([N] + [0] * (N - 1))
def _coal_constant(state):
    return _coalescent(state, as_list=False)


@with_ipv([N] + [0] * (N - 1))
def _coal_param(state):
    return _coalescent(state, as_list=True)


def constant_graph():
    g = Graph(_coal_constant, graph_cache=False)
    assert not g.parameterized()
    return g


def param_graph():
    g = Graph(_coal_param, graph_cache=False)
    assert g.parameterized()
    return g


DATA = np.array([0.3, 0.5, 0.9, 1.2, 0.4])


def _rewards(g):
    r = np.zeros(g.vertices_length())
    r[2] = 1.0
    return r


# entry point -> call(graph). Each must raise the guard error on a constant
# graph BEFORE doing any work.
SITES = {
    "svgd": lambda g: g.svgd(DATA, n_particles=2, n_iterations=1, theta_dim=1,
                             verbose=False),
    "mcmc": lambda g: g.mcmc(DATA, n_samples=2, n_chains=1, burn_in=1,
                             theta_dim=1),
    "Graph.method_of_moments": lambda g: g.method_of_moments(DATA, theta_dim=1),
    "method_of_moments module": lambda g: __import__(
        "phasic.method_of_moments", fromlist=["x"]).method_of_moments(
            g, DATA, theta_dim=1),
    "Graph.probability_matching": lambda g: g.probability_matching(DATA),
    "probability_matching module": lambda g: __import__(
        "phasic.probability_matching", fromlist=["x"]).probability_matching(
            g, DATA),
    "epoch_model": lambda g: g.epoch_model(DATA, [0.0]),
    "daisy_chain_joint_probs": lambda g: g.daisy_chain_joint_probs(
        epoch_thetas=np.ones((1, 1)), epoch_dts=[1.0], initial_ipv=[1.0]),
    "pmf_and_moments_from_graph": lambda g: Graph.pmf_and_moments_from_graph(g),
    "moments_from_graph": lambda g: Graph.moments_from_graph(g),
    "pmf_and_moments_from_graph_multivariate":
        lambda g: Graph.pmf_and_moments_from_graph_multivariate(g),
    "pmf_from_graph_joint_index": lambda g: Graph.pmf_from_graph_joint_index(g),
    "add_epoch": lambda g: g.add_epoch(1.0),
    "joint_prob_graph": lambda g: g.joint_prob_graph(reward_limit=1),
    "joint_stop_prob_graph": lambda g: g.joint_stop_prob_graph(),
    "joint_sojourn_graph": lambda g: g.joint_sojourn_graph(),
    "reward_visit_probability(theta=jax)": lambda g: g.reward_visit_probability(
        _rewards(g), theta=jnp.asarray([1.0])),
    "reward_visit_probability(theta=numpy)": lambda g: g.reward_visit_probability(
        _rewards(g), theta=np.asarray([1.0])),
    "profile(theta)": lambda g: g.profile(theta=[1.0], probe_dyn=False),
    "profile_graph(theta)": lambda g: __import__(
        "phasic.profile", fromlist=["x"]).profile_graph(
            g, theta=[1.0], probe_dyn=False),
    "bffg.path_exit_rates_by_param": lambda g: __import__(
        "phasic.bffg", fromlist=["x"]).path_exit_rates_by_param(g, {}),
    "bffg.bffg_log_prob": lambda g: __import__(
        "phasic.bffg", fromlist=["x"]).bffg_log_prob(
            g, g, np.ones(1), lambda t: t, DATA),
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_constant_graph_refused(site):
    with pytest.raises(ValueError, match=GUARD_MSG) as exc:
        SITES[site](constant_graph())
    assert "needs a parameterized model" in str(exc.value)


@pytest.mark.parametrize("site", sorted(SITES))
def test_parameterized_graph_passes_the_guard(site):
    """The parameterized twin may fail for other reasons (e.g. "requires a
    joint-probability graph"), but never at the guard."""
    try:
        SITES[site](param_graph())
    except Exception as e:  # noqa: BLE001 - only the guard message matters
        assert "needs a parameterized model" not in str(e), site


def test_reward_transform_output_is_refused():
    p = param_graph()
    p.update_weights([1.0])
    rt = p.reward_transform(np.ones(p.vertices_length()))
    assert not rt.parameterized()
    with pytest.raises(ValueError, match=GUARD_MSG):
        rt.svgd(DATA, n_particles=2, n_iterations=1, theta_dim=1, verbose=False)


def test_constant_graph_without_theta_still_works():
    """Entry points used WITHOUT theta keep working on constant graphs."""
    g = constant_graph()
    p = g.reward_visit_probability(_rewards(g))
    assert 0.0 <= float(p) <= 1.0
    prof = g.profile(probe_dyn=False)     # _max_rate used to update_weights
    assert prof is not None
    np.testing.assert_allclose(g.expectation(), 1.5, rtol=1e-12)

