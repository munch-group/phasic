#!/usr/bin/env python3
"""
Check if PMF values differ for same time but different reward vectors.
This would explain the SVGD convergence issue.
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
print("PMF Check for Different Reward Vectors")
print("="*80)
print()

# Check PMF at time t=0.15 for each reward vector
t = 0.15

print(f"Time t = {t}")
print("-" * 80)

for i in range(rewards.shape[1]):
    reward_vec = rewards[:, i]

    # Compute PMF for this reward vector
    graph_r = graph.reward_transform(reward_vec)
    pmf = graph_r.pdf(t)

    print(f"\nFeature {i}:")
    print(f"  Reward vector: {reward_vec}")
    print(f"  PMF(t={t}): {pmf:.6f}")

print()
print("="*80)
print("Analysis")
print("="*80)
print("If PMF values are DIFFERENT for same time t but different rewards,")
print("then the 'observed_data' in multivariate case is comparing apples to oranges!")
print()
print("The model should compute PMF using the SAME reward vector that was used")
print("to generate that feature's data.")
print()
print("But our current multivariate model applies ALL reward vectors to compute")
print("a 2D PMF matrix. This is CORRECT - each column uses its reward vector.")
print()
print("Let me check if the model is computing PMF correctly...")
