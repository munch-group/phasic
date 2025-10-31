"""Test model with and without rewards"""
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

# Sample data WITHOUT rewards (standard coalescent)
nr_observations = 1000
observed_data = jnp.array(graph.sample(nr_observations))
print(f"observed_data shape: {observed_data.shape}")
print(f"observed_data mean: {jnp.mean(observed_data):.6f}")

# Create model WITHOUT FFI
print("=== Creating model with use_ffi=False ===")
model_no_ffi = phasic.Graph.pmf_and_moments_from_graph(
    graph, nr_moments=1, discrete=False, use_ffi=False, param_length=1
)

# Create model WITH FFI
print("=== Creating model with use_ffi=True ===")
model_ffi = phasic.Graph.pmf_and_moments_from_graph(
    graph, nr_moments=1, discrete=False, use_ffi=True, param_length=1
)

print("\n=== Test 1: use_ffi=False, no rewards ===")
pmf, moments = model_no_ffi(theta, observed_data, rewards=None)
print(f"pmf min/max: {jnp.min(pmf):.6e} / {jnp.max(pmf):.6e}")
print(f"pmf zero count: {jnp.sum(pmf == 0)} / {pmf.size}")
print(f"moments: {moments}")

print("\n=== Test 2: use_ffi=True, no rewards ===")
pmf2, moments2 = model_ffi(theta, observed_data, rewards=None)
print(f"pmf min/max: {jnp.min(pmf2):.6e} / {jnp.max(pmf2):.6e}")
print(f"pmf zero count: {jnp.sum(pmf2 == 0)} / {pmf2.size}")
print(f"moments: {moments2}")

print("\n=== Test 3: use_ffi=False, with rewards ===")
rewards = graph.states()[:, 1:-1]
reward_0 = rewards[:, 0]
print(f"reward_0: {reward_0}")
pmf3, moments3 = model_no_ffi(theta, observed_data, rewards=reward_0)
print(f"pmf min/max: {jnp.min(pmf3):.6e} / {jnp.max(pmf3):.6e}")
print(f"pmf zero count: {jnp.sum(pmf3 == 0)} / {pmf3.size}")
print(f"moments: {moments3}")

print("\n=== Test 4: use_ffi=True, with rewards ===")
pmf4, moments4 = model_ffi(theta, observed_data, rewards=reward_0)
print(f"pmf min/max: {jnp.min(pmf4):.6e} / {jnp.max(pmf4):.6e}")
print(f"pmf zero count: {jnp.sum(pmf4 == 0)} / {pmf4.size}")
print(f"moments: {moments4}")
