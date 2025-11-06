#!/usr/bin/env python3
"""
Check expected values for each reward vector used in multivariate test
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


true_theta = 10.0
nr_samples = 4

graph = Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph.update_parameterized_weights(np.array([true_theta]))

# Get rewards
rewards = graph.states()[:, :-2]

print("="*80)
print("Expected Values for Each Reward Vector")
print("="*80)
print(f"True θ: {true_theta}")
print(f"Rewards shape: {rewards.shape}")
print(f"\nRewards matrix (vertices x features):")
print(rewards)
print()

# Compute expected value for each reward vector
print("Expected values E[X] for each feature:")
print("-" * 80)

for i in range(rewards.shape[1]):
    reward_vec = rewards[:, i]
    print(f"\nFeature {i}:")
    print(f"  Reward vector: {reward_vec}")

    # Transform graph with this reward vector
    graph_transformed = graph.reward_transform(reward_vec)

    # Expected value = graph.expectation() for the transformed graph
    expected = graph_transformed.expectation()

    print(f"  E[X]: {expected:.6f}")

    # Also check by sampling
    np.random.seed(42)
    samples = graph.sample(50000, rewards=reward_vec)
    sample_mean = np.mean(samples)
    print(f"  Sample mean (50k samples): {sample_mean:.6f}")

print()
print("="*80)
print("Analysis")
print("="*80)
print("If all features have SAME expected value (0.15), then multivariate should")
print("converge to same θ as univariate.")
print()
print("If features have DIFFERENT expected values, then multivariate data is")
print("inconsistent - you're fitting ONE θ to data from DIFFERENT distributions!")
print("This would cause parameter estimation to be biased.")
