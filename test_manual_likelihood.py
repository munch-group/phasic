#!/usr/bin/env python3
"""
Manually compute log-likelihood for θ=10 vs θ=11 to check which is actually better.
If θ=11 has higher likelihood, then SVGD is correct and the data was somehow mislabeled.
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


nr_samples = 4

print("="*80)
print("Manual Log-Likelihood Computation")
print("="*80)
print()

# Generate data at θ=10 (exactly as in the multivariate test)
print("Generating data at θ=10...")
graph_data = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph_data.update_parameterized_weights(np.array([10.0]))

rewards = graph_data.states()[:, :-2]
n = 10000
np.random.seed(42)
a = np.empty((rewards.shape[1], n * rewards.shape[1]))
a[:] = np.nan

for i in range(rewards.shape[1]):
    a[i, i*n:(i+1)*n] = graph_data.sample(n, rewards=rewards[:, i])

observed_data = jnp.array(a).T
print(f"Data shape: {observed_data.shape}")
print(f"Data means: {np.apply_along_axis(np.nanmean, 0, observed_data)}")
print()

# Create multivariate model
model = Graph.pmf_and_moments_from_graph_multivariate(
    graph_data, nr_moments=2, discrete=False, param_length=1
)

# Compute log-likelihood at different θ values
print("Computing log-likelihoods...")
print("-" * 80)

theta_values = np.array([8.0, 9.0, 10.0, 11.0, 12.0])

for theta in theta_values:
    theta_jax = jnp.array([theta])

    # Compute PMF for all observations
    pmf_vals, moments = model(theta_jax, observed_data, rewards=rewards)

    # Compute log-likelihood
    mask = ~jnp.isnan(pmf_vals)
    log_lik = jnp.sum(jnp.where(mask, jnp.log(pmf_vals + 1e-10), 0.0))

    print(f"θ = {theta:5.1f}: log-lik = {float(log_lik):12.2f}")

print()
print("="*80)
print("Analysis")
print("="*80)
print("The θ with HIGHEST log-likelihood is the MLE.")
print("If θ=10 has highest log-lik: data was generated correctly, SVGD has a bug")
print("If θ=11 has highest log-lik: something wrong with data generation or model")
