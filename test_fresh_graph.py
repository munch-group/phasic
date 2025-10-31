"""Test with fresh graph (no update_parameterized_weights before model creation)"""
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

# Create graph WITHOUT calling update_parameterized_weights
print("=== Creating fresh graph (no update_parameterized_weights) ===")
graph_fresh = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)

# Create model from fresh graph
model = phasic.Graph.pmf_and_moments_from_graph(
    graph_fresh, nr_moments=1, discrete=False, use_ffi=True, param_length=1
)

# NOW sample data using a SEPARATE graph
graph_for_sampling = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
theta = np.array([1.0])
graph_for_sampling.update_parameterized_weights(theta)

nr_observations = 1000
observed_data = jnp.array(graph_for_sampling.sample(nr_observations))
print(f"observed_data mean: {jnp.mean(observed_data):.6f}")

# Call model with theta
print("\n=== Calling model(theta, observed_data) ===")
pmf, moments = model(theta, observed_data, rewards=None)
print(f"pmf min/max: {jnp.min(pmf):.6e} / {jnp.max(pmf):.6e}")
print(f"pmf zero count: {jnp.sum(pmf == 0)} / {pmf.size}")
print(f"moments: {moments}")
