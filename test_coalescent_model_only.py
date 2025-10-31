"""Test just the coalescent model without SVGD"""

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

print("Step 1: Creating graph...")
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print(f"  ✓ Graph has {graph.vertices_length()} vertices")

print("\nStep 2: Getting rewards...")
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards_raw = _graph.states().T[:-2]
rewards = rewards_raw.T
print(f"  ✓ rewards shape: {rewards.shape} (n_vertices={rewards.shape[0]}, n_features={rewards.shape[1]})")

print("\nStep 3: Creating multivariate model...")
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False
)
print("  ✓ Model created")

print("\nStep 4: Creating test data...")
test_times = jnp.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])  # (2 times, 3 features)
print(f"  test_times shape: {test_times.shape}")

print("\nStep 5: Calling model...")
try:
    pmf, moments = model(true_theta, test_times, rewards=rewards)
    print(f"  ✓ Model call succeeded!")
    print(f"    PMF shape: {pmf.shape}")
    print(f"    Moments shape: {moments.shape}")
except Exception as e:
    print(f"  ✗ Model call failed: {e}")
    import traceback
    traceback.print_exc()
