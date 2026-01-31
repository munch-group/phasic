"""
Tests for method-of-moments parameter estimation and probability matching.

Covers:
- Single-parameter exponential model
- Two-parameter Erlang-like model
- Overdetermined system (more moments than parameters)
- Automatic nr_moments increase
- Fixed parameters
- Prior construction and SVGD integration
- Multi-feature (2D) observations with 2D rewards
- Custom initial guess
- Verbose/silent modes
- Probability matching for joint probability graphs
"""

import numpy as np
import pytest
from functools import partial
from itertools import combinations_with_replacement

from phasic import (
    Graph, GaussPrior, SVGD, ProbMatchResult, DataPrior,
    with_ipv, StateIndexer, Property,
)
from phasic.method_of_moments import MoMResult, method_of_moments
from phasic.probability_matching import ProbMatchResult as ProbMatchResultDirect

import jax.numpy as jnp

all_pairs = partial(combinations_with_replacement, r=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_exponential_graph():
    """Single-parameter exponential: S -> [2] -> [1] with rate theta[0]."""
    g = Graph(1)
    start = g.starting_vertex()
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v2, 1.0)
    v2.add_edge_parameterized(v1, 0.0, [1.0])
    return g


def _build_two_stage_graph():
    """Two-parameter Erlang-like: S -> [3] --(theta[0])--> [2] --(theta[1])--> [1].

    Total time = Exp(theta[0]) + Exp(theta[1]).
    """
    g = Graph(1)
    start = g.starting_vertex()
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    start.add_edge(v3, 1.0)
    v3.add_edge_parameterized(v2, 0.0, [1.0, 0.0])  # rate = theta[0]
    v2.add_edge_parameterized(v1, 0.0, [0.0, 1.0])  # rate = theta[1]
    return g


# ---------------------------------------------------------------------------
# 1. Single-parameter exponential
# ---------------------------------------------------------------------------

def test_exponential_analytical():
    """For Exp(theta), E[T] = 1/theta.  MoM should recover theta ~= 1/mean."""
    true_rate = 3.0
    np.random.seed(42)
    data = np.random.exponential(1.0 / true_rate, size=500)

    g = _build_exponential_graph()
    result = g.method_of_moments(data, nr_moments=2, verbose=False)

    assert isinstance(result, MoMResult)
    assert result.success
    # Analytical MoM: theta = 1/mean(data)
    np.testing.assert_allclose(result.theta[0], 1.0 / np.mean(data), rtol=0.05)


# ---------------------------------------------------------------------------
# 2. Two-parameter Erlang-like
# ---------------------------------------------------------------------------

def test_two_param_moments_match():
    """Moments at solution should approximate sample moments."""
    true_theta = (5.0, 2.0)
    np.random.seed(42)
    data = (np.random.exponential(1.0 / true_theta[0], size=300)
            + np.random.exponential(1.0 / true_theta[1], size=300))

    g = _build_two_stage_graph()
    result = g.method_of_moments(data, nr_moments=2, verbose=False)

    assert result.success
    # Model moments should be close to sample moments
    np.testing.assert_allclose(
        result.model_moments.ravel(),
        result.sample_moments.ravel(),
        rtol=0.05,
    )


# ---------------------------------------------------------------------------
# 3. Overdetermined (4 moments, 1 parameter)
# ---------------------------------------------------------------------------

def test_overdetermined():
    """More moment equations than parameters should still converge."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=300)

    g = _build_exponential_graph()
    result = g.method_of_moments(data, nr_moments=4, verbose=False)

    assert result.success
    assert result.model_moments.shape == (4,)
    assert result.sample_moments.shape == (4,)


# ---------------------------------------------------------------------------
# 4. Auto nr_moments increase
# ---------------------------------------------------------------------------

def test_auto_nr_moments_increase(capsys):
    """If nr_moments gives fewer equations than free params, it auto-increases."""
    np.random.seed(42)
    data = (np.random.exponential(0.2, size=200)
            + np.random.exponential(0.5, size=200))

    g = _build_two_stage_graph()
    result = g.method_of_moments(data, nr_moments=1, verbose=True)

    captured = capsys.readouterr()
    assert "increased nr_moments" in captured.out
    assert result.success
    # Should have been increased to at least 2
    assert result.model_moments.shape[0] >= 2


def test_default_nr_moments_is_overdetermined():
    """Default nr_moments=None should auto-select an overdetermined system."""
    np.random.seed(42)

    # 1-parameter model: should get at least 4 moments (max(2*1, 4) = 4)
    g1 = _build_exponential_graph()
    data1 = np.random.exponential(0.5, size=200)
    result1 = g1.method_of_moments(data1, verbose=False)
    assert result1.model_moments.shape[0] >= 4, (
        f"Expected >= 4 moments for 1-param model, got {result1.model_moments.shape[0]}"
    )
    assert result1.success

    # 2-parameter model: should get at least 4 moments (max(2*2, 4) = 4)
    g2 = _build_two_stage_graph()
    data2 = (np.random.exponential(0.2, size=200)
             + np.random.exponential(0.5, size=200))
    result2 = g2.method_of_moments(data2, verbose=False)
    n_equations = result2.model_moments.size
    assert n_equations >= 4, (
        f"Expected >= 4 equations for 2-param model, got {n_equations}"
    )


# ---------------------------------------------------------------------------
# 5. Fixed parameters
# ---------------------------------------------------------------------------

def test_fixed_parameter():
    """Fix one parameter and optimise the other."""
    true_theta = (5.0, 2.0)
    np.random.seed(42)
    data = (np.random.exponential(1.0 / true_theta[0], size=300)
            + np.random.exponential(1.0 / true_theta[1], size=300))

    g = _build_two_stage_graph()
    result = g.method_of_moments(
        data, nr_moments=2,
        fixed=[(0, 5.0)],
        verbose=False,
    )

    assert result.success
    # Fixed value should be preserved exactly
    assert result.theta[0] == 5.0
    # Free parameter should be reasonable
    assert result.theta[1] > 0
    # Prior for fixed param should be None
    assert result.prior[0] is None
    assert result.prior[1] is not None


# ---------------------------------------------------------------------------
# 6. Prior construction
# ---------------------------------------------------------------------------

def test_prior_construction():
    """Returned priors should be GaussPrior instances with correct mean/std."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=200)

    g = _build_exponential_graph()
    result = g.method_of_moments(data, nr_moments=2, std_multiplier=3.0, verbose=False)

    assert len(result.prior) == 1
    prior = result.prior[0]
    assert isinstance(prior, GaussPrior)
    assert prior.mu == pytest.approx(result.theta[0], rel=1e-10)
    # std_multiplier = 3.0, so prior sigma should be 3 * asymptotic_std
    assert prior.sigma == pytest.approx(3.0 * result.std[0], rel=1e-10)


def test_std_is_meaningful():
    """Std should reflect sampling uncertainty, not curve-fit residuals.

    For Exp(theta) with n=200, the delta-method std should be of order
    theta / sqrt(n) ≈ 2 / 14 ≈ 0.14, NOT near-zero.
    """
    np.random.seed(42)
    data = np.random.exponential(0.5, size=200)  # true rate = 2.0

    g = _build_exponential_graph()
    result = g.method_of_moments(data, nr_moments=2, verbose=False)

    # std should be statistically meaningful (order 0.01 - 1.0), not tiny
    assert result.std[0] > 0.01, f"std too small: {result.std[0]}"
    assert result.std[0] < 10.0, f"std too large: {result.std[0]}"


def test_std_scales_with_sample_size():
    """Std should decrease roughly as 1/sqrt(n)."""
    g = _build_exponential_graph()

    stds = []
    for n in [200, 2000]:
        np.random.seed(42)
        data = np.random.exponential(0.5, size=n)
        result = g.method_of_moments(data, nr_moments=2, verbose=False)
        stds.append(result.std[0])

    # std(n=2000) should be roughly sqrt(10) ≈ 3.16 times smaller than std(n=200)
    ratio = stds[0] / stds[1]
    assert 1.5 < ratio < 6.0, f"std ratio {ratio:.2f} not consistent with 1/sqrt(n) scaling"


# ---------------------------------------------------------------------------
# 7. Multi-feature (2D data + 2D rewards)
# ---------------------------------------------------------------------------

def test_multivariate_rewards():
    """2D observations with 2D reward matrix should match per-feature moments."""
    true_rate = 3.0
    np.random.seed(42)
    n_samples = 300

    g = _build_exponential_graph()
    n_vertices = g.vertices_length()

    # Create reward vectors — two features with different rewards
    rewards = np.ones((2, n_vertices), dtype=np.float64)
    rewards[0, :] = 1.0  # Feature 0: standard
    rewards[1, :] = 2.0  # Feature 1: doubled rewards

    # Generate data for two features
    base_data = np.random.exponential(1.0 / true_rate, size=n_samples)
    feature_0 = base_data * 1.0
    feature_1 = base_data * 2.0
    observed_data = np.column_stack([feature_0, feature_1])

    result = g.method_of_moments(
        observed_data, nr_moments=2, rewards=rewards, verbose=False,
    )

    assert result.success
    assert result.sample_moments.shape == (2, 2)  # (n_features, nr_moments)
    assert result.model_moments.shape == (2, 2)


# ---------------------------------------------------------------------------
# 8. Integration with SVGD
# ---------------------------------------------------------------------------

def test_svgd_integration():
    """result.prior should work as the prior argument to SVGD."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=100)

    g = _build_exponential_graph()
    mom_result = g.method_of_moments(data, nr_moments=2, verbose=False)

    svgd = g.svgd(
        observed_data=data,
        prior=mom_result.prior,
        n_particles=8,
        n_iterations=3,
        verbose=False,
    )
    assert svgd.particles.shape == (8, 1)


# ---------------------------------------------------------------------------
# 9. Custom initial guess
# ---------------------------------------------------------------------------

def test_custom_theta_init():
    """theta_init should be respected as the starting point."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=200)

    g = _build_exponential_graph()
    result = g.method_of_moments(
        data, nr_moments=2,
        theta_init=np.array([1.5]),
        verbose=False,
    )
    assert result.success
    # Should still converge to the right answer regardless of init
    np.testing.assert_allclose(result.theta[0], 1.0 / np.mean(data), rtol=0.1)


# ---------------------------------------------------------------------------
# 10. Verbose/silent modes
# ---------------------------------------------------------------------------

def test_verbose_false_no_crash():
    """verbose=False should produce no output and not crash."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=100)

    g = _build_exponential_graph()
    result = g.method_of_moments(data, verbose=False)
    assert isinstance(result, MoMResult)


def test_verbose_true_no_crash(capsys):
    """verbose=True should print info and not crash."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=100)

    g = _build_exponential_graph()
    result = g.method_of_moments(data, verbose=True)
    assert isinstance(result, MoMResult)

    captured = capsys.readouterr()
    assert "MoM:" in captured.out


# ---------------------------------------------------------------------------
# Probability matching helpers
# ---------------------------------------------------------------------------

def _build_joint_prob_graph():
    """4-sample coalescent with joint_prob_graph(indexer, tot_reward_limit=2).

    Returns (joint_graph, base_graph) with 1 coalescent-rate param + mutation_rate.
    """
    nr_samples = 4

    indexer = StateIndexer(
        lineage=[
            Property('descendants', min_value=1, max_value=nr_samples),
        ]
    )

    @with_ipv([nr_samples] + [0] * (nr_samples - 1))
    def coalescent_1param(state):
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
            descendants = p1.descendants + p2.descendants
            k = indexer.lineage.props_to_index(descendants=descendants)
            new[k] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions

    base_graph = Graph(coalescent_1param)
    mutation_rate = 1.0
    joint_graph = base_graph.joint_prob_graph(
        indexer, tot_reward_limit=2, mutation_rate=mutation_rate,
    )
    return joint_graph, base_graph, mutation_rate


def _sample_joint_observations(joint_graph, theta, n):
    """Sample n observations from a joint prob graph at given theta."""
    joint_graph.update_weights(theta)
    table = joint_graph.joint_prob_table()
    p = table['prob'].to_numpy()
    p = p / p.sum()
    feature_cols = table.columns[:-1]  # all columns except 'prob'
    sampled_rows = np.random.choice(len(table), size=n, p=p)
    observations = table.iloc[sampled_rows][feature_cols].values.tolist()
    return [tuple(int(x) for x in row) for row in observations]


# ---------------------------------------------------------------------------
# 11. Probability matching — basic convergence
# ---------------------------------------------------------------------------

def test_prob_matching_basic():
    """probability_matching() converges and recovers theta approximately."""
    np.random.seed(42)
    joint_graph, _, mutation_rate = _build_joint_prob_graph()

    true_theta = [7.0, mutation_rate]
    observations = _sample_joint_observations(joint_graph, true_theta, n=2000)

    result = joint_graph.probability_matching(
        observations,
        fixed=[(1, mutation_rate)],
        verbose=False,
    )

    assert isinstance(result, ProbMatchResult)
    assert result.success
    # theta[0] should be in a reasonable range around the true value
    assert 1.0 < result.theta[0] < 30.0, f"theta[0]={result.theta[0]} out of range"
    # Fixed parameter preserved
    assert result.theta[1] == mutation_rate


# ---------------------------------------------------------------------------
# 12. Probability matching — model probs match empirical
# ---------------------------------------------------------------------------

def test_prob_matching_probs_match():
    """Model probs should approximate empirical probs at the solution."""
    np.random.seed(123)
    joint_graph, _, mutation_rate = _build_joint_prob_graph()

    true_theta = [7.0, mutation_rate]
    observations = _sample_joint_observations(joint_graph, true_theta, n=5000)

    result = joint_graph.probability_matching(
        observations,
        fixed=[(1, mutation_rate)],
        verbose=False,
    )

    assert result.success
    # Model probs should be close to empirical probs
    np.testing.assert_allclose(
        result.model_probs, result.empirical_probs, atol=0.05,
    )


# ---------------------------------------------------------------------------
# 13. Probability matching — prior construction
# ---------------------------------------------------------------------------

def test_prob_matching_prior_construction():
    """Returns GaussPrior for free params and None for fixed params."""
    np.random.seed(42)
    joint_graph, _, mutation_rate = _build_joint_prob_graph()

    true_theta = [7.0, mutation_rate]
    observations = _sample_joint_observations(joint_graph, true_theta, n=1000)

    result = joint_graph.probability_matching(
        observations,
        fixed=[(1, mutation_rate)],
        verbose=False,
    )

    assert len(result.prior) == 2
    # theta[0] is free -> GaussPrior
    assert isinstance(result.prior[0], GaussPrior)
    assert result.prior[0].mu == pytest.approx(result.theta[0], rel=1e-10)
    # theta[1] is fixed -> None
    assert result.prior[1] is None


# ---------------------------------------------------------------------------
# 14. Probability matching — SVGD integration
# ---------------------------------------------------------------------------

def test_prob_matching_svgd_integration():
    """The .prior list works with graph.svgd()."""
    np.random.seed(42)
    joint_graph, _, mutation_rate = _build_joint_prob_graph()

    true_theta = [7.0, mutation_rate]
    observations = _sample_joint_observations(joint_graph, true_theta, n=500)

    result = joint_graph.probability_matching(
        observations,
        fixed=[(1, mutation_rate)],
        verbose=False,
    )

    svgd = joint_graph.svgd(
        observations,
        prior=result.prior,
        fixed=[(1, mutation_rate)],
        n_particles=8,
        n_iterations=3,
        verbose=False,
    )
    assert svgd.particles.shape == (8, 2)


# ---------------------------------------------------------------------------
# 15. Probability matching — rejects non-joint graph
# ---------------------------------------------------------------------------

def test_prob_matching_rejects_non_joint_graph():
    """Calling probability_matching on a non-joint graph raises ValueError."""
    g = _build_exponential_graph()

    with pytest.raises(ValueError, match="joint probability graph"):
        g.probability_matching([(0, 0, 0, 0)], verbose=False)


# ---------------------------------------------------------------------------
# 16. Probability matching — verbose output
# ---------------------------------------------------------------------------

def test_prob_matching_verbose(capsys):
    """Verbose output includes useful diagnostic information."""
    np.random.seed(42)
    joint_graph, _, mutation_rate = _build_joint_prob_graph()

    true_theta = [7.0, mutation_rate]
    observations = _sample_joint_observations(joint_graph, true_theta, n=500)

    result = joint_graph.probability_matching(
        observations,
        fixed=[(1, mutation_rate)],
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "ProbMatch:" in captured.out
    assert "theta_dim=" in captured.out


# ---------------------------------------------------------------------------
# 17. DataPrior — standard graph (method_of_moments)
# ---------------------------------------------------------------------------

def test_data_prior_exponential():
    """DataPrior on a standard graph uses method_of_moments and converges."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=300)

    g = _build_exponential_graph()
    dp = DataPrior(g, data, sd=2.0, verbose=False)

    assert dp.method == "method_of_moments"
    assert dp.success
    # Estimate should be close to 1/mean(data) ≈ 2
    np.testing.assert_allclose(dp.theta[0], 1.0 / np.mean(data), rtol=0.15)


# ---------------------------------------------------------------------------
# 18. DataPrior — fixed parameters
# ---------------------------------------------------------------------------

def test_data_prior_fixed():
    """Fixed params yield None priors at fixed indices, GaussPrior at free."""
    np.random.seed(42)
    true_theta = (5.0, 2.0)
    data = (np.random.exponential(1.0 / true_theta[0], size=300)
            + np.random.exponential(1.0 / true_theta[1], size=300))

    g = _build_two_stage_graph()
    dp = DataPrior(g, data, fixed=[(0, 5.0)], verbose=False)

    assert len(dp) == 2
    assert dp[0] is None
    assert isinstance(dp[1], GaussPrior)
    assert dp.theta[0] == 5.0


# ---------------------------------------------------------------------------
# 19. DataPrior — SVGD integration
# ---------------------------------------------------------------------------

def test_data_prior_svgd_integration():
    """DataPrior works as the prior= argument to graph.svgd()."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=100)

    g = _build_exponential_graph()
    dp = DataPrior(g, data, verbose=False)

    svgd = g.svgd(
        observed_data=data,
        prior=dp,
        n_particles=8,
        n_iterations=3,
        verbose=False,
    )
    assert svgd.particles.shape == (8, 1)


# ---------------------------------------------------------------------------
# 20. DataPrior — joint prob graph (probability_matching)
# ---------------------------------------------------------------------------

def test_data_prior_joint_prob():
    """DataPrior on a joint prob graph uses probability_matching."""
    np.random.seed(42)
    joint_graph, _, mutation_rate = _build_joint_prob_graph()

    true_theta = [7.0, mutation_rate]
    observations = _sample_joint_observations(joint_graph, true_theta, n=2000)

    dp = DataPrior(
        joint_graph, observations,
        fixed=[(1, mutation_rate)],
        verbose=False,
    )

    assert dp.method == "probability_matching"
    assert dp.success
    assert dp.theta[1] == mutation_rate
    assert dp[1] is None  # fixed
    assert isinstance(dp[0], GaussPrior)


# ---------------------------------------------------------------------------
# 21. DataPrior — iterable interface
# ---------------------------------------------------------------------------

def test_data_prior_iterable():
    """DataPrior supports len, iteration, and indexing."""
    np.random.seed(42)
    data = (np.random.exponential(0.2, size=200)
            + np.random.exponential(0.5, size=200))

    g = _build_two_stage_graph()
    dp = DataPrior(g, data, verbose=False)

    assert len(dp) == 2
    assert list(dp) == [dp[0], dp[1]]
    for p in dp:
        assert isinstance(p, GaussPrior)


# ---------------------------------------------------------------------------
# 22. DataPrior — repr
# ---------------------------------------------------------------------------

def test_data_prior_repr():
    """repr includes method name and convergence status."""
    np.random.seed(42)
    data = np.random.exponential(0.5, size=200)

    g = _build_exponential_graph()
    dp = DataPrior(g, data, verbose=False)

    r = repr(dp)
    assert "method_of_moments" in r
    assert "converged" in r
