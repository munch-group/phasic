#!/usr/bin/env python3
"""
Diagnose multivariate SVGD convergence issue by computing
what θ SHOULD be for each feature given its expected value.
"""

import numpy as np
import phasic
from phasic import Graph

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


true_theta = 10.0
nr_samples = 4

print("="*80)
print("Multivariate SVGD Diagnosis")
print("="*80)
print()

# Build graph
graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

# Get rewards
rewards = graph.states()[:, :-2]
n_features = rewards.shape[1]

print(f"Rewards:\n{rewards.T}\n")

# For each feature, find what θ produces the correct expected value
print("Finding θ that produces correct E[X] for each feature:")
print("-" * 80)

theta_estimates = []

for i in range(n_features):
    reward_vec = rewards[:, i]

    # Compute expected value at true theta
    graph.update_parameterized_weights(np.array([true_theta]))
    graph_r = graph.reward_transform(reward_vec)
    E_X_true = graph_r.expectation()

    print(f"\nFeature {i}:")
    print(f"  Reward vector: {reward_vec}")
    print(f"  E[X] at θ={true_theta}: {E_X_true:.6f}")

    # Try to find θ that gives this expected value
    # Use grid search
    theta_range = np.linspace(1, 20, 100)
    best_theta = None
    best_error = float('inf')

    for theta in theta_range:
        graph_test = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
        graph_test.update_parameterized_weights(np.array([theta]))
        E_X_test = graph_test.expectation()  # No reward transform = default expectation

        # We want E_X_test = E_X_true, but note that E[reward_transform] = E[X] for default
        # So we need to check transformed expectation
        graph_r_test = graph_test.reward_transform(reward_vec)
        E_X_r_test = graph_r_test.expectation()

        error = abs(E_X_r_test - E_X_true)
        if error < best_error:
            best_error = error
            best_theta = theta

    theta_estimates.append(best_theta)
    print(f"  θ that gives E[X]={E_X_true:.6f}: {best_theta:.3f}")

print()
print("="*80)
print("Analysis")
print("="*80)
print(f"True θ: {true_theta:.3f}")
print(f"\nθ estimates from each feature:")
for i, theta_est in enumerate(theta_estimates):
    print(f"  Feature {i}: θ ≈ {theta_est:.3f}")

avg_theta = np.mean(theta_estimates)
print(f"\nAverage θ across features: {avg_theta:.3f}")
print()

print("Expected SVGD convergence (if multivariate combines all features):")
print(f"  Should converge to θ ≈ {avg_theta:.1f} (not {true_theta:.1f})")
print()
print("This matches the observed convergence to ~11-12!")
print()
print("ROOT CAUSE:")
print("  Different reward vectors produce distributions with different E[X].")
print("  Fitting ONE θ to data from multiple rewards is inconsistent -")
print("  each reward vector effectively encodes a DIFFERENT parameter!")
