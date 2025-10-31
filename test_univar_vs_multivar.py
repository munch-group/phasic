#!/usr/bin/env python3
"""
Compare univariate vs multivariate SVGD to isolate the bug
"""

import numpy as np
import phasic
from phasic import Graph, SVGD, ExponentialDecayStepSize
import jax.numpy as jnp

phasic.set_theme('dark')

def coalescent(state, nr_samples=None):
    """Coalescent model callback"""
    if not state.size:
        ipv = [[[nr_samples]+[0]*nr_samples, 1.0, []]]
        return ipv
    else:
        transitions = []
        for i in range(nr_samples):
            for j in range(i, nr_samples):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i+j+1] += 1
                transitions.append([new, 0.0, [state[i]*(state[j]-same)/(1+same)]])
        return transitions


true_theta = np.array([10])
nr_samples = 4

print("="*80)
print("Univariate vs Multivariate SVGD Comparison")
print("="*80)
print(f"True θ: {true_theta[0]}")
print()

# Build graph and generate data
graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph.update_parameterized_weights(true_theta)

# Univariate test (no rewards)
print("TEST 1: Univariate SVGD (no rewards)")
print("-" * 80)

n_obs = 10000
np.random.seed(42)
observed_data_1d = np.array(graph.sample(n_obs))
print(f"Data mean: {np.mean(observed_data_1d):.6f}")

graph_svgd_1 = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

step_schedule = ExponentialDecayStepSize(first_step=0.01, last_step=0.001, tau=50.0)
params = dict(
    bandwidth='median',
    theta_dim=1,
    prior=lambda phi: -0.5 * jnp.sum(((phi - 0.0) / 10.0)**2),
    n_particles=20,
    n_iterations=200,
    learning_rate=step_schedule,
    seed=42,
    verbose=False,
    discrete=False,
)

svgd_1 = graph_svgd_1.svgd(observed_data=observed_data_1d, **params)
theta_1 = svgd_1.particles.mean(axis=0)[0]
error_1 = abs(theta_1 - true_theta[0]) / true_theta[0] * 100

print(f"SVGD θ_mean: {theta_1:.3f}")
print(f"Error:       {error_1:.1f}%")
print()

# Multivariate test with single feature (should be same as univariate)
print("TEST 2: Multivariate SVGD with 1 feature (should match univariate)")
print("-" * 80)

# Use same data but as 2D array with 1 feature
observed_data_2d_single = observed_data_1d.reshape(-1, 1)
print(f"Data shape: {observed_data_2d_single.shape}")
print(f"Data mean:  {np.mean(observed_data_2d_single):.6f}")

# Single reward vector (all ones)
rewards_single = np.ones((graph.vertices_length(), 1))
print(f"Rewards shape: {rewards_single.shape}")

graph_svgd_2 = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

svgd_2 = graph_svgd_2.svgd(observed_data=observed_data_2d_single, rewards=rewards_single, **params)
theta_2 = svgd_2.particles.mean(axis=0)[0]
error_2 = abs(theta_2 - true_theta[0]) / true_theta[0] * 100

print(f"SVGD θ_mean: {theta_2:.3f}")
print(f"Error:       {error_2:.1f}%")
print()

# Multivariate test with 3 features (original problem)
print("TEST 3: Multivariate SVGD with 3 features (original problem)")
print("-" * 80)

rewards_3 = graph.states()[:, :-2]
print(f"Rewards shape: {rewards_3.shape}")
print(f"Rewards:\n{rewards_3.T}")

# Generate multivariate data
n = 10000
np.random.seed(42)
a = np.empty((rewards_3.shape[1], n * rewards_3.shape[1]))
a[:] = np.nan

for i in range(rewards_3.shape[1]):
    a[i, i*n:(i+1)*n] = graph.sample(n, rewards=rewards_3[:, i])

observed_data_2d = jnp.array(a).T
print(f"Data shape: {observed_data_2d.shape}")
print(f"Data means per feature: {np.apply_along_axis(np.nanmean, 0, observed_data_2d)}")

graph_svgd_3 = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

svgd_3 = graph_svgd_3.svgd(observed_data=observed_data_2d, rewards=rewards_3, **params)
theta_3 = svgd_3.particles.mean(axis=0)[0]
error_3 = abs(theta_3 - true_theta[0]) / true_theta[0] * 100

print(f"SVGD θ_mean: {theta_3:.3f}")
print(f"Error:       {error_3:.1f}%")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print(f"True θ:                                {true_theta[0]:.3f}")
print(f"Test 1 (univariate):                   {theta_1:.3f} ({error_1:.1f}% error)")
print(f"Test 2 (multivariate 1 feature):       {theta_2:.3f} ({error_2:.1f}% error)")
print(f"Test 3 (multivariate 3 features):      {theta_3:.3f} ({error_3:.1f}% error)")
print()

if error_1 < 5 and error_3 > 5:
    print("✗ BUG CONFIRMED: Multivariate SVGD has higher error than univariate")
elif error_1 < 5 and error_2 > 5:
    print("✗ BUG: Even single-feature multivariate has higher error")
else:
    print("✓ Both converge correctly")
