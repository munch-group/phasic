"""Test exact workflow from multivar_test.py"""
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

# Exact workflow from multivar_test.py
true_theta = np.array([10])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)  # Fresh graph for SVGD

nr_observations = 1000
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)  # Separate graph for sampling
_graph.update_parameterized_weights(true_theta)

rewards = _graph.states()[:, 1:-1]  # Get rewards from _graph
print(f"rewards shape: {rewards.shape}")
print(f"rewards:\n{rewards}")

observed_data = jnp.array([
    _graph.sample(nr_observations, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T  # Shape: (nr_observations, 3)

print(f"\nobserved_data shape: {observed_data.shape}")
print(f"observed_data means: {jnp.mean(observed_data, axis=0)}")

# Create multivariate model (same as svgd() does)
print("\n=== Creating multivariate model (as svgd() does) ===")
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph,  # Using fresh graph!
    nr_moments=1,
    discrete=False,
    use_ffi=False,
    param_length=len(true_theta)
)

# Test model with true_theta
print("\n=== Calling model with true_theta ===")
pmf, moments = model(true_theta, observed_data, rewards=rewards)
print(f"pmf min/max: {jnp.min(pmf):.6e} / {jnp.max(pmf):.6e}")
print(f"pmf zero count: {jnp.sum(pmf == 0)} / {pmf.size}")
print(f"moments:\n{moments}")
