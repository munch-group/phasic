"""Test reward_transform on non-parameterized graph"""
import phasic
import numpy as np

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
                rate = state[i]*(state[j]-same)/(1+same)
                transitions.append([new, rate, []])  # Non-parameterized
        return transitions

print("Testing reward_transform on NON-parameterized graph")
print("=" * 70)

nr_samples = 5

# Create NON-parameterized graph (concrete rates)
graph = phasic.Graph(callback=coalescent, parameterized=False, nr_samples=nr_samples)

# Get some reward vectors
rewards = np.array([0, 5, 3, 1, 2, 0, 1, 0])  # Feature 0 from coalescent
print(f"Reward vector: {rewards}")

# Apply reward transformation
reward_graph = graph.reward_transform(rewards)

# Compute first moment
moments = reward_graph.moments(1)
expected_mean = moments[0]

print(f"Expected mean (from reward_transform): {expected_mean}")

# Sample and compare
n_samples = 10000
np.random.seed(42)
sampled_data = graph.sample(n_samples, rewards=rewards)
empirical_mean = np.mean(sampled_data)

print(f"Empirical mean (from {n_samples} samples): {empirical_mean:.6f}")
print(f"Difference: {abs(expected_mean - empirical_mean):.6f}")

if abs(expected_mean - empirical_mean) < 0.01:
    print("✓ reward_transform works on non-parameterized graph")
else:
    print("✗ reward_transform broken on non-parameterized graph too")

print("\n" + "=" * 70)
print("Testing reward_transform on PARAMETERIZED graph")
print("=" * 70)

def coalescent_param(state, nr_samples=None):
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

# Create parameterized graph
graph_param = phasic.Graph(callback=coalescent_param, parameterized=True, nr_samples=nr_samples)
graph_param.update_parameterized_weights(np.array([1.0]))

print(f"Reward vector: {rewards}")

# Apply reward transformation
reward_graph_param = graph_param.reward_transform(rewards)

# Compute first moment
moments_param = reward_graph_param.moments(1)
expected_mean_param = moments_param[0]

print(f"Expected mean (from reward_transform): {expected_mean_param}")

# Sample and compare
sampled_data_param = graph_param.sample(n_samples, rewards=rewards)
empirical_mean_param = np.mean(sampled_data_param)

print(f"Empirical mean (from {n_samples} samples): {empirical_mean_param:.6f}")
print(f"Difference: {abs(expected_mean_param - empirical_mean_param):.6f}")

if np.isnan(expected_mean_param):
    print("✗ reward_transform returns NaN on parameterized graph")
    print("\n→ This is the root cause of SVGD estimation errors!")
else:
    print("✓ reward_transform works on parameterized graph")
