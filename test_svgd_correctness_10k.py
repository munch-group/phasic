#!/usr/bin/env python3
"""
Comprehensive SVGD correctness test with 10,000 observations
Tests both univariate and multivariate cases
"""

import numpy as np
import phasic
from phasic import Graph
import jax.numpy as jnp

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


def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2)"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)


true_theta = 10.0
nr_samples = 4
n_observations = 10000

print("="*80)
print("SVGD Correctness Test with 10,000 Observations")
print("="*80)
print(f"True θ = {true_theta}")
print(f"Number of observations = {n_observations}")
print()

# ============================================================================
# Test 1: Univariate SVGD (no rewards)
# ============================================================================
print("="*80)
print("Test 1: Univariate SVGD (NO reward transformation)")
print("="*80)
print()

# Generate data
print("Generating univariate data...")
graph_data = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph_data.update_parameterized_weights(np.array([true_theta]))
observed_data_univar = graph_data.sample(n_observations)

print(f"  Data mean: {np.mean(observed_data_univar):.6f}")
print(f"  Expected E[T] at θ={true_theta}: {graph_data.expectation():.6f}")
print()

# Run SVGD
print("Running univariate SVGD...")
graph_svgd_univar = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

params = dict(
    bandwidth='median',
    theta_dim=1,
    prior=uninformative_prior,
    n_particles=20,
    n_iterations=200,
    learning_rate=0.01,
    seed=42,
    verbose=False,
    discrete=False,
)

svgd_univar = graph_svgd_univar.svgd(observed_data=observed_data_univar, **params)

theta_mean_univar = svgd_univar.particles.mean(axis=0)[0]
theta_std_univar = svgd_univar.particles.std(axis=0)[0]
error_pct_univar = abs(theta_mean_univar - true_theta) / true_theta * 100

print()
print("Univariate SVGD Results:")
print(f"  True θ:           {true_theta:.3f}")
print(f"  Inferred θ (mean): {theta_mean_univar:.3f}")
print(f"  Std dev:          {theta_std_univar:.3f}")
print(f"  Relative error:   {error_pct_univar:.2f}%")
print()

if error_pct_univar < 5.0:
    print("✅ PASS: Univariate SVGD converged to correct value (error < 5%)")
else:
    print(f"❌ FAIL: Univariate SVGD error too large ({error_pct_univar:.2f}% >= 5%)")

# ============================================================================
# Test 2: Multivariate SVGD (with rewards)
# ============================================================================
print()
print("="*80)
print("Test 2: Multivariate SVGD (WITH reward transformation)")
print("="*80)
print()

# Generate multivariate data
print("Generating multivariate data...")
graph_data_multivar = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph_data_multivar.update_parameterized_weights(np.array([true_theta]))

rewards = graph_data_multivar.states()[:, :-2]
n_features = rewards.shape[1]

print(f"  Number of features: {n_features}")
print(f"  Rewards shape: {rewards.shape}")
print()

# Generate data for each feature
np.random.seed(42)
a = np.empty((n_features, n_observations * n_features))
a[:] = np.nan

for i in range(n_features):
    a[i, i*n_observations:(i+1)*n_observations] = graph_data_multivar.sample(
        n_observations, rewards=rewards[:, i]
    )

observed_data_multivar = jnp.array(a).T

# Compute expected values for each feature
print("Expected values for each feature:")
for i in range(n_features):
    E_X = graph_data_multivar.expectation(rewards=rewards[:, i])
    obs_mean = np.nanmean(observed_data_multivar[:, i])
    print(f"  Feature {i}: E[X] = {E_X:.6f}, observed mean = {obs_mean:.6f}")
print()

# Run multivariate SVGD
print("Running multivariate SVGD...")
graph_svgd_multivar = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

svgd_multivar = graph_svgd_multivar.svgd(
    observed_data=observed_data_multivar,
    rewards=rewards,
    **params
)

theta_mean_multivar = svgd_multivar.particles.mean(axis=0)[0]
theta_std_multivar = svgd_multivar.particles.std(axis=0)[0]
error_pct_multivar = abs(theta_mean_multivar - true_theta) / true_theta * 100

print()
print("Multivariate SVGD Results:")
print(f"  True θ:           {true_theta:.3f}")
print(f"  Inferred θ (mean): {theta_mean_multivar:.3f}")
print(f"  Std dev:          {theta_std_multivar:.3f}")
print(f"  Relative error:   {error_pct_multivar:.2f}%")
print()

if error_pct_multivar < 5.0:
    print("✅ PASS: Multivariate SVGD converged to correct value (error < 5%)")
else:
    print(f"❌ FAIL: Multivariate SVGD error too large ({error_pct_multivar:.2f}% >= 5%)")

# ============================================================================
# Summary
# ============================================================================
print()
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Univariate SVGD:   error = {error_pct_univar:.2f}%  {'✅ PASS' if error_pct_univar < 5.0 else '❌ FAIL'}")
print(f"Multivariate SVGD: error = {error_pct_multivar:.2f}%  {'✅ PASS' if error_pct_multivar < 5.0 else '❌ FAIL'}")
print()

if error_pct_univar < 5.0 and error_pct_multivar < 5.0:
    print("✅ ALL TESTS PASSED")
    exit(0)
else:
    print("❌ SOME TESTS FAILED")
    exit(1)
