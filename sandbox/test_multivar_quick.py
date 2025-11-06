"""Quick test of multivar with correct shapes"""

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

print("Creating graph...")
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

nr_observations = 50  # Much smaller for quick test
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)

print("Getting rewards...")
rewards_raw = _graph.states().T[:-2]
rewards = rewards_raw.T  # Transpose to (n_vertices, n_features)
print(f"  rewards shape: {rewards.shape}")

print("Sampling data...")
observed_data_raw = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards_raw])
observed_data = observed_data_raw.T  # Transpose to (n_observations, n_features)
print(f"  observed_data shape: {observed_data.shape}")

print("\nRunning SVGD (quick test: 5 iterations, 8 particles)...")

params = dict(
    observed_data=observed_data,
    bandwidth='median',
    theta_dim=1,
    n_particles=8,
    n_iterations=5,
    seed=42,
    verbose=True,
    rewards=rewards,
    parallel='none',  # Safe config for testing
    jit=False
)

svgd = graph.svgd(**params)

print(f"\n✓ SUCCESS!")
print(f"  True theta: {true_theta}")
print(f"  Posterior mean: {svgd.theta_mean}")
