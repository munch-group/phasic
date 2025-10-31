#!/usr/bin/env python3
"""
Reproduce multivariate SVGD convergence issue:
- Expected: converge to theta = 10
- Actual: converges to theta = 12
"""

import numpy as np
import phasic
from phasic import Graph, SVGD, ExponentialDecayStepSize
import jax.numpy as jnp

phasic.set_theme('dark')

def coalescent(state, nr_samples=None):
    """Coalescent model callback"""
    if not state.size:
        # Starting edge: base_weight=1.0, empty coefficients (non-parameterized)
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


# Parameters
true_theta = np.array([10])  # TRUE VALUE - SVGD should converge to this
nr_samples = 4

print("="*80)
print("Multivariate SVGD Convergence Test")
print("="*80)
print(f"True θ: {true_theta[0]}")
print(f"nr_samples: {nr_samples}")
print()

# Build graph and generate data
print("Building graph and generating data...")
graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph.update_parameterized_weights(true_theta)

# Get rewards (states excluding last 2 columns)
rewards = graph.states()[:, :-2]
print(f"Rewards shape: {rewards.shape}")
print(f"Rewards:\n{rewards.T}")
print()

# Generate multivariate data matching the notebook pattern
n = 10000  # nr_observations
np.random.seed(42)
a = np.empty((rewards.shape[1], n * rewards.shape[1]))
a[:] = np.nan

print("Generating multivariate data...")
for i in range(rewards.shape[1]):
    print(f"  Feature {i}: sampling with reward vector {rewards[:, i]}")
    a[i, i*n:(i+1)*n] = graph.sample(n, rewards=rewards[:, i])

observed_data = jnp.array(a).T
print(f"Observed data shape: {observed_data.shape}")
print(f"Observed data means per feature: {np.apply_along_axis(np.nanmean, 0, observed_data)}")
print()

# Build fresh graph for SVGD (no update_parameterized_weights called)
print("Building fresh graph for SVGD...")
graph_svgd = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

# Setup SVGD parameters
step_schedule = ExponentialDecayStepSize(first_step=0.01, last_step=0.001, tau=50.0)

params = dict(
    bandwidth='median',
    theta_dim=len(true_theta),
    prior=lambda phi: -0.5 * jnp.sum(((phi - 0.0) / 10.0)**2),  # Uninformative prior
    n_particles=20,
    n_iterations=200,
    learning_rate=step_schedule,
    seed=42,
    verbose=False,
    discrete=False,
)

print("Running SVGD with multivariate data and rewards...")
svgd = graph_svgd.svgd(observed_data=observed_data, rewards=rewards, **params)

# Results
theta_mean = svgd.particles.mean(axis=0)[0]
error_pct = abs(theta_mean - true_theta[0]) / true_theta[0] * 100

print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"True θ:      {true_theta[0]:.3f}")
print(f"SVGD θ_mean: {theta_mean:.3f}")
print(f"Error:       {error_pct:.1f}%")
print()

if abs(theta_mean - 12.0) < 0.5:
    print("✗ BUG REPRODUCED: Converges to ~12 instead of 10")
elif abs(theta_mean - 10.0) < 0.5:
    print("✓ CORRECT: Converges to ~10 as expected")
else:
    print(f"? UNEXPECTED: Converges to {theta_mean:.1f}")
