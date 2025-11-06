"""Test if model_1d works correctly in isolation"""
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

# Create graph and set theta=1
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
theta = np.array([1.0])
graph.update_parameterized_weights(theta)

# Get rewards for feature 0 only
rewards = graph.states()[:, 1:-1]
reward_0 = rewards[:, 0]  # Feature 0: [0, 0, 1, 2, 0, 0]
print(f"reward_0: {reward_0}")

# Sample data for feature 0
nr_observations = 1000
observed_data_0 = jnp.array(graph.sample(nr_observations, rewards=reward_0))
print(f"observed_data_0 shape: {observed_data_0.shape}")
print(f"observed_data_0 mean: {jnp.mean(observed_data_0):.6f}")

# Create 1D model
model_1d = phasic.Graph.pmf_and_moments_from_graph(
    graph, nr_moments=1, discrete=False, use_ffi=False, param_length=1
)

print("\n=== Testing model_1d with reward_0 ===")
pmf, moments = model_1d(theta, observed_data_0, rewards=reward_0)
print(f"pmf shape: {pmf.shape}")
print(f"pmf min/max: {jnp.min(pmf):.6e} / {jnp.max(pmf):.6e}")
print(f"pmf zero count: {jnp.sum(pmf == 0)} / {pmf.size}")
print(f"moments: {moments}")
