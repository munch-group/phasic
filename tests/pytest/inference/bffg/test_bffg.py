"""
Tests for general BFFG utilities.
"""

from phasic import (
    Graph, with_ipv,
    path_to_rewards, path_exit_rates,
    importance_log_weight_from_rates, importance_weighted_log_likelihood,
)
import numpy as np
import jax.numpy as jnp
import pytest


def build_simple_chain():
    """3-state chain: start -> v1(rate=2) -> v2(rate=3) -> absorb."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 2.0)
    v2.add_edge(v3, 3.0)
    return g


def build_coalescent(n=4):
    """Block coalescent with n samples, rate = 1/N with N=1."""
    @with_ipv([n] + [0] * (n - 1))
    def coal(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2: continue
                if not same and (state[i] < 1 or state[j] < 1): continue
                new = state.copy()
                new[i] -= 1; new[j] -= 1; new[i + j + 1] += 1
                transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions
    g = Graph(coal)
    g.update_weights([1.0])
    return g


class TestPathToRewards:

    def test_1d_rewards(self):
        """1D rewards give a scalar result."""
        g = build_simple_chain()
        path = g.sample_path()
        rewards = np.ones(g.vertices_length())
        result = path_to_rewards(g, path, rewards)
        assert isinstance(result, float)
        assert result > 0

    def test_2d_rewards(self):
        """2D rewards give per-feature results."""
        g = build_coalescent(4)
        path = g.sample_path()
        states = g.states()
        rewards = states.T  # (n_features, n_vertices)
        result = path_to_rewards(g, path, rewards)
        assert result.shape == (rewards.shape[0],)

    def test_none_rewards_uses_states(self):
        """rewards=None uses graph.states().T"""
        g = build_coalescent(4)
        path = g.sample_path()
        result_none = path_to_rewards(g, path)
        result_explicit = path_to_rewards(g, path, rewards=g.states().T)
        np.testing.assert_array_almost_equal(result_none, result_explicit)

    def test_unit_rewards_equals_total_time(self):
        """Unit rewards (all 1s) should give total absorption time."""
        g = build_simple_chain()
        path = g.sample_path()
        rewards = np.ones(g.vertices_length())
        result = path_to_rewards(g, path, rewards)
        # Total time = last entry time (absorbing vertex entry)
        total_time = path['entry_times'][-1]
        np.testing.assert_almost_equal(result, total_time, decimal=10)

    def test_statistical_match_with_sample(self):
        """Mean of path_to_rewards should match mean of graph.sample(rewards=...)."""
        g = build_coalescent(4)
        rewards_1d = np.ones(g.vertices_length())
        n = 2000

        # Via sample_path + path_to_rewards
        path_results = [path_to_rewards(g, g.sample_path(), rewards_1d) for _ in range(n)]

        # Via graph.sample (uses same C code internally)
        sample_results = g.sample(n, rewards=rewards_1d.tolist())

        # Means should be close (both estimate the same expectation)
        assert abs(np.mean(path_results) - np.mean(sample_results)) < 0.15 * np.mean(sample_results)


class TestPathExitRates:

    def test_simple_chain_rates(self):
        """Exit rates for simple chain should be [2.0, 3.0]."""
        g = build_simple_chain()
        path = g.sample_path()
        rates, sojourns = path_exit_rates(g, path)
        # v1 has rate 2.0, v2 has rate 3.0
        np.testing.assert_array_almost_equal(rates, [2.0, 3.0])
        assert len(sojourns) == 2
        assert np.all(sojourns > 0)

    def test_sojourn_times_positive(self):
        """Sojourn times should be positive."""
        g = build_coalescent(5)
        path = g.sample_path()
        rates, sojourns = path_exit_rates(g, path)
        assert np.all(sojourns > 0)
        assert np.all(rates > 0)


class TestImportanceLogWeightFromRates:

    def test_zero_when_equal(self):
        """Weight is 0 when proposal = target."""
        rates = np.array([2.0, 3.0, 1.0])
        sojourns = np.array([0.5, 0.3, 0.8])
        w = importance_log_weight_from_rates(rates, rates, sojourns)
        np.testing.assert_almost_equal(w, 0.0)

    def test_nonzero_when_different(self):
        """Weight is nonzero when rates differ."""
        proposal = np.array([2.0, 3.0])
        target = np.array([4.0, 1.0])
        sojourns = np.array([0.5, 0.3])
        w = importance_log_weight_from_rates(proposal, target, sojourns)
        assert w != 0.0

    def test_formula(self):
        """Verify against manual calculation."""
        proposal = np.array([2.0])
        target = np.array([4.0])
        sojourns = np.array([0.5])
        # log(4/2) - (4-2)*0.5 = log(2) - 1 ≈ -0.3069
        expected = np.log(2.0) - 1.0
        w = importance_log_weight_from_rates(proposal, target, sojourns)
        np.testing.assert_almost_equal(w, expected, decimal=10)


class TestImportanceWeightedLogLikelihood:

    def test_uniform_weights(self):
        """With log_weights=0, result is log-mean of likelihoods."""
        log_liks = jnp.array([-1.0, -2.0, -3.0])
        log_weights = jnp.zeros(3)
        result = float(importance_weighted_log_likelihood(log_liks, log_weights))
        # Should be log(mean(exp(log_liks))) = log((e^-1 + e^-2 + e^-3)/3)
        expected = float(jnp.log(jnp.mean(jnp.exp(log_liks))))
        np.testing.assert_almost_equal(result, expected, decimal=10)

    def test_single_sample(self):
        """With M=1, result is log_lik + log_weight."""
        log_lik = jnp.array([-5.0])
        log_w = jnp.array([2.0])
        result = float(importance_weighted_log_likelihood(log_lik, log_w))
        np.testing.assert_almost_equal(result, -3.0, decimal=10)

    def test_jax_compatible(self):
        """Should work with JAX arrays."""
        import jax
        log_liks = jnp.array([-1.0, -2.0])
        log_weights = jnp.array([0.5, -0.5])
        result = importance_weighted_log_likelihood(log_liks, log_weights)
        assert jnp.isfinite(result)
