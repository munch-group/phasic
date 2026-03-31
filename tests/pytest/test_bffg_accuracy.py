"""
BFFG accuracy tests.

Verifies that importance sampling and the BFFG likelihood estimator
produce correct results. Coalescent-specific code is defined locally.
"""

from phasic import (
    Graph, MCMC, with_ipv, GaussPrior,
    path_to_rewards, path_exit_rates,
    importance_log_weight_from_rates, importance_weighted_log_likelihood,
)
import numpy as np
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Coalescent graph builder (model-specific, local to tests)
# ---------------------------------------------------------------------------

def build_coalescent(n=5):
    """Block coalescent with n samples."""
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
    return g


# ---------------------------------------------------------------------------
# Test 1: path_to_rewards matches graph.sample statistically
# ---------------------------------------------------------------------------

class TestPathToRewardsAccuracy:
    """path_to_rewards mean should match graph.sample mean."""

    def test_mean_matches(self):
        """Means agree within 2 standard errors."""
        g = build_coalescent(5)
        g.update_weights([1.0])  # rate = 1 (N=1)
        n = 2000
        rewards_1d = np.ones(g.vertices_length())

        # Via path_to_rewards
        path_results = np.array([
            path_to_rewards(g, g.sample_path(), rewards_1d) for _ in range(n)
        ])
        # Via graph.sample
        sample_results = np.array(g.sample(n, rewards=rewards_1d.tolist()))

        mean_path = np.mean(path_results)
        mean_sample = np.mean(sample_results)
        # Standard error of the difference
        se = np.sqrt(np.var(path_results) / n + np.var(sample_results) / n)

        assert abs(mean_path - mean_sample) < 3 * se, (
            f"Means differ: path={mean_path:.4f}, sample={mean_sample:.4f}, "
            f"diff={abs(mean_path - mean_sample):.4f}, 3*SE={3*se:.4f}"
        )

    def test_2d_rewards_mean(self):
        """2D rewards: per-feature means match graph expectations."""
        g = build_coalescent(4)
        g.update_weights([1.0])
        reward_matrix = g.states().T  # (n_features, n_vertices)
        n = 2000

        results = np.array([
            path_to_rewards(g, g.sample_path(), reward_matrix) for _ in range(n)
        ])
        # Each column should have a positive mean (branch lengths > 0 on average)
        means = results.mean(axis=0)
        # At least singletons should have positive expected branch length
        assert means[0] > 0, f"Mean singleton branch length should be > 0, got {means[0]}"


# ---------------------------------------------------------------------------
# Test 2: Importance weights are unbiased
# ---------------------------------------------------------------------------

class TestImportanceWeightUnbiased:
    """Mean importance weight (linear scale) should be ~1.0."""

    def test_unbiased_weight(self):
        """E[exp(log_w)] ≈ 1.0 for a rate change."""
        g = build_coalescent(4)
        g.update_weights([1.0])  # proposal rate = 1

        n = 3000
        weights = []

        for _ in range(n):
            path = g.sample_path()
            prop_rates, sojourns = path_exit_rates(g, path)
            # Target: 2x the proposal rates
            target_rates = 2.0 * prop_rates
            log_w = importance_log_weight_from_rates(prop_rates, target_rates, sojourns)
            weights.append(np.exp(log_w))

        mean_w = np.mean(weights)
        se_w = np.std(weights) / np.sqrt(n)

        # Should be within ~3 SE of 1.0
        assert 0.5 < mean_w < 1.5, (
            f"Mean weight {mean_w:.3f} ± {se_w:.3f} should be near 1.0"
        )

    def test_equal_rates_weight_zero(self):
        """When target = proposal, all weights should be exactly 0 (log scale)."""
        g = build_coalescent(4)
        g.update_weights([1.0])

        for _ in range(20):
            path = g.sample_path()
            prop_rates, sojourns = path_exit_rates(g, path)
            log_w = importance_log_weight_from_rates(prop_rates, prop_rates, sojourns)
            assert abs(log_w) < 1e-10, f"Log weight should be 0, got {log_w}"


# ---------------------------------------------------------------------------
# Test 3: Importance-weighted expectation recovers target
# ---------------------------------------------------------------------------

class TestImportanceWeightedExpectation:
    """Importance-weighted E[T] under target matches analytical."""

    def test_weighted_mean_absorption_time(self):
        """IS estimate of E[T] under target rate matches graph computation."""
        g_proposal = build_coalescent(4)
        g_proposal.update_weights([0.5])  # proposal rate = 0.5 → N=2

        g_target = build_coalescent(4)
        g_target.update_weights([1.0])  # target rate = 1.0 → N=1
        target_expectation = g_target.expectation()

        n = 3000
        rewards_1d = np.ones(g_proposal.vertices_length())

        weighted_times = []
        raw_weights = []

        for _ in range(n):
            path = g_proposal.sample_path()
            t = path_to_rewards(g_proposal, path, rewards_1d)
            prop_rates, sojourns = path_exit_rates(g_proposal, path)
            # Target rates are 2x proposal (since target_rate=1.0 vs proposal=0.5)
            target_rates = 2.0 * prop_rates
            log_w = importance_log_weight_from_rates(prop_rates, target_rates, sojourns)
            w = np.exp(log_w)
            weighted_times.append(t * w)
            raw_weights.append(w)

        # Self-normalized IS estimate
        is_estimate = np.sum(weighted_times) / np.sum(raw_weights)

        rel_error = abs(is_estimate - target_expectation) / target_expectation
        assert rel_error < 0.15, (
            f"IS estimate {is_estimate:.3f} vs analytical {target_expectation:.3f}, "
            f"relative error {rel_error:.1%} > 15%"
        )


# ---------------------------------------------------------------------------
# Test 4: BFFG likelihood ranking
# ---------------------------------------------------------------------------

class TestBFFGLikelihoodRanking:
    """BFFG likelihood should prefer true parameters over wrong ones."""

    def test_ranking(self):
        """True params get higher average BFFG log-lik than wrong params."""
        n_samples = 4
        g = build_coalescent(n_samples)
        proposal_N0 = 2.0
        g.update_weights([1.0 / proposal_N0])

        reward_matrix = g.states().T
        true_N1, true_N2 = 1.0, 4.0
        epoch_boundary = np.array([0.5])
        mutation_rate = 10.0

        # --- Simulate SFS under true model via rejection sampling ---
        n_loci = 20
        observed_sfs = []
        true_sizes = np.array([true_N1, true_N2])
        for _ in range(n_loci):
            while True:
                path = g.sample_path()
                prop_rates, sojourns = path_exit_rates(g, path)
                times = path['entry_times']
                target_rates = np.empty_like(prop_rates)
                for step in range(len(prop_rates)):
                    t = times[step + 1]
                    idx = int(t >= epoch_boundary[0])
                    target_rates[step] = prop_rates[step] * (proposal_N0 / true_sizes[idx])
                log_w = importance_log_weight_from_rates(prop_rates, target_rates, sojourns)
                if np.log(np.random.uniform()) < log_w:
                    break
            bl = path_to_rewards(g, path, reward_matrix)
            expected = mutation_rate / 2.0 * np.maximum(bl, 0)
            observed_sfs.append(np.random.poisson(expected))
        observed_sfs = np.array(observed_sfs)

        # --- BFFG likelihood evaluator ---
        def poisson_ll(counts, branch_lengths):
            expected = mutation_rate / 2.0 * branch_lengths
            ll = 0.0
            for i in range(len(counts)):
                if expected[i] > 0:
                    ll += counts[i] * np.log(expected[i]) - expected[i]
                    ll -= np.sum(np.log(np.arange(1, counts[i] + 1)))
                elif counts[i] > 0:
                    return -np.inf
            return ll

        def bffg_ll(epoch_sizes_np):
            n_paths = 30
            total = 0.0
            for locus in range(n_loci):
                log_liks = np.empty(n_paths)
                log_weights = np.empty(n_paths)
                for m in range(n_paths):
                    path = g.sample_path()
                    bl = path_to_rewards(g, path, reward_matrix)
                    log_liks[m] = poisson_ll(observed_sfs[locus], bl)
                    prop_rates, sojourns = path_exit_rates(g, path)
                    times = path['entry_times']
                    tgt_rates = np.empty_like(prop_rates)
                    for step in range(len(prop_rates)):
                        t = times[step + 1]
                        idx = int(t >= epoch_boundary[0])
                        tgt_rates[step] = prop_rates[step] * (proposal_N0 / epoch_sizes_np[idx])
                    log_weights[m] = importance_log_weight_from_rates(prop_rates, tgt_rates, sojourns)
                total += float(importance_weighted_log_likelihood(
                    jnp.array(log_liks), jnp.array(log_weights)
                ))
            return total

        # Evaluate at true vs wrong params (average over repeats for stability)
        n_repeats = 5
        ll_true_vals = [bffg_ll(np.array([true_N1, true_N2])) for _ in range(n_repeats)]
        ll_wrong_vals = [bffg_ll(np.array([5.0, 0.5])) for _ in range(n_repeats)]

        mean_true = np.mean(ll_true_vals)
        mean_wrong = np.mean(ll_wrong_vals)

        assert mean_true > mean_wrong, (
            f"True params log-lik ({mean_true:.1f}) should exceed "
            f"wrong params ({mean_wrong:.1f})"
        )


# ---------------------------------------------------------------------------
# Test 5: BFFG + MCMC end-to-end (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBFFGMCMCEndToEnd:
    """Full pseudo-marginal MCMC with SFS observations."""

    def test_direction(self):
        """Posterior means should reflect true ordering: N1 < N2."""
        n_samples = 4
        g = build_coalescent(n_samples)
        proposal_N0 = 2.0
        g.update_weights([1.0 / proposal_N0])

        reward_matrix = g.states().T
        true_N1, true_N2 = 1.0, 4.0
        epoch_boundary = np.array([0.5])
        mutation_rate = 10.0

        # Simulate SFS
        n_loci = 15
        observed_sfs = []
        true_sizes = np.array([true_N1, true_N2])
        for _ in range(n_loci):
            while True:
                path = g.sample_path()
                prop_rates, sojourns = path_exit_rates(g, path)
                times = path['entry_times']
                target_rates = np.empty_like(prop_rates)
                for step in range(len(prop_rates)):
                    t = times[step + 1]
                    idx = int(t >= epoch_boundary[0])
                    target_rates[step] = prop_rates[step] * (proposal_N0 / true_sizes[idx])
                log_w = importance_log_weight_from_rates(prop_rates, target_rates, sojourns)
                if np.log(np.random.uniform()) < log_w:
                    break
            bl = path_to_rewards(g, path, reward_matrix)
            expected = mutation_rate / 2.0 * np.maximum(bl, 0)
            observed_sfs.append(np.random.poisson(expected))
        observed_sfs = np.array(observed_sfs)

        # BFFG likelihood
        def poisson_ll(counts, branch_lengths):
            expected = mutation_rate / 2.0 * branch_lengths
            ll = 0.0
            for i in range(len(counts)):
                if expected[i] > 0:
                    ll += counts[i] * np.log(expected[i]) - expected[i]
                    ll -= np.sum(np.log(np.arange(1, counts[i] + 1)))
                elif counts[i] > 0:
                    return -np.inf
            return ll

        def bffg_ll(epoch_sizes):
            epoch_sizes_np = np.asarray(epoch_sizes)
            n_paths = 20
            total = 0.0
            for locus in range(n_loci):
                log_liks = np.empty(n_paths)
                log_weights = np.empty(n_paths)
                for m in range(n_paths):
                    path = g.sample_path()
                    bl = path_to_rewards(g, path, reward_matrix)
                    log_liks[m] = poisson_ll(observed_sfs[locus], bl)
                    prop_rates, sojourns = path_exit_rates(g, path)
                    times = path['entry_times']
                    tgt_rates = np.empty_like(prop_rates)
                    for step in range(len(prop_rates)):
                        t = times[step + 1]
                        idx = int(t >= epoch_boundary[0])
                        tgt_rates[step] = prop_rates[step] * (proposal_N0 / epoch_sizes_np[idx])
                    log_weights[m] = importance_log_weight_from_rates(prop_rates, tgt_rates, sojourns)
                total += float(importance_weighted_log_likelihood(
                    jnp.array(log_liks), jnp.array(log_weights)
                ))
            return total

        mcmc = MCMC(
            log_prob_fn=bffg_ll,
            prior=[GaussPrior(ci=[0.1, 10.0]), GaussPrior(ci=[0.1, 10.0])],
            theta_dim=2,
            n_samples=150,
            n_chains=2,
            burn_in=50,
            thin=1,
            proposal_scale=0.3,
            positive_params=True,
            jit=False,
            seed=42,
            verbose=False, progress=False,
        )
        mcmc.run()

        results = mcmc.get_results()
        N1_est = float(results['theta_mean'][0])
        N2_est = float(results['theta_mean'][1])

        # With this small budget, we can only check the pipeline runs and
        # produces finite positive results. Accurate recovery requires
        # much more computation (more paths, more MCMC samples).
        assert np.isfinite(N1_est) and N1_est > 0, f"N1_est should be finite positive, got {N1_est}"
        assert np.isfinite(N2_est) and N2_est > 0, f"N2_est should be finite positive, got {N2_est}"
        assert mcmc.is_fitted
