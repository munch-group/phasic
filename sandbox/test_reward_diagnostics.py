"""Diagnostic: Check if reward-transformed graph computes correct moments"""
import phasic
import numpy as np
import jax.numpy as jnp

def coalescent(state, nr_samples=None):
    if not state.size:
        ipv = [[[nr_samples]+[0]*nr_samples, 1, []]]
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

print("=" * 70)
print("REWARD TRANSFORMATION DIAGNOSTIC")
print("=" * 70)

true_theta = np.array([10.0])
nr_samples = 5

# Create parameterized graph
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
graph.update_parameterized_weights(true_theta)

# Get rewards (one column per feature)
rewards = graph.states()[:, :-2]
print(f"\nRewards shape: {rewards.shape}")
print(f"Rewards:\n{rewards}")

# For each feature, compute:
# 1. Expected mean from reward transformation
# 2. Sample data and compute empirical mean
# 3. Compare

n_samples = 10000
np.random.seed(42)

print(f"\n{'=' * 70}")
print(f"Testing {rewards.shape[1]} features with {n_samples} samples each")
print(f"{'=' * 70}\n")

for feature_idx in range(rewards.shape[1]):
    print(f"Feature {feature_idx}:")
    print("-" * 70)

    # Get reward vector for this feature
    reward_vec = rewards[:, feature_idx]
    print(f"  Reward vector: {reward_vec}")

    # Apply reward transformation
    reward_graph = graph.reward_transform(reward_vec)

    # Compute first moment using reward-transformed graph
    # The moment should equal E[R * T] where T is the absorption time
    moments = reward_graph.moments(1)
    expected_mean = moments[0]
    print(f"  Expected mean (from reward_transform): {expected_mean:.6f}")

    # Sample data using this reward vector
    sampled_data = graph.sample(n_samples, rewards=reward_vec)
    empirical_mean = np.mean(sampled_data)
    empirical_std = np.std(sampled_data) / np.sqrt(n_samples)

    print(f"  Empirical mean (from {n_samples} samples): {empirical_mean:.6f} ± {empirical_std:.6f}")

    # Check if they match
    diff = abs(expected_mean - empirical_mean)
    z_score = diff / empirical_std if empirical_std > 0 else float('inf')

    if z_score < 3:  # Within 3 standard errors
        status = "✓ MATCH"
    else:
        status = f"✗ MISMATCH (z={z_score:.1f})"

    print(f"  Difference: {diff:.6f} ({status})")
    print()

print(f"{'=' * 70}")
print("INTERPRETATION:")
print("  If expected != empirical:")
print("    → reward_transform() or moments() is incorrect")
print("  If expected == empirical but SVGD fails:")
print("    → Problem is in how model uses rewards in PMF/PDF computation")
print(f"{'=' * 70}")
