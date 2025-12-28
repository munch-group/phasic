"""Fixed version of multivar_test.py with correct array shapes"""

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


true_theta = np.array([10.0])  # Make sure it's float!
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

nr_observations = 10000
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)

# Get rewards - this is (n_features, n_vertices) but we need (n_vertices, n_features)
rewards_raw = _graph.states().T
rewards_raw = rewards_raw[:-2]
print(f"rewards_raw shape: {rewards_raw.shape} (n_features, n_vertices)")

# TRANSPOSE rewards to (n_vertices, n_features)
rewards = rewards_raw.T
print(f"rewards (transposed) shape: {rewards.shape} (n_vertices, n_features) ✓")

# Sample data - this creates (n_features, n_observations) but we need (n_observations, n_features)
observed_data_raw = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards_raw])
print(f"observed_data_raw shape: {observed_data_raw.shape} (n_features, n_observations)")

# TRANSPOSE observed_data to (n_observations, n_features)
observed_data = observed_data_raw.T
print(f"observed_data (transposed) shape: {observed_data.shape} (n_observations, n_features) ✓")

def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

n_iterations = 200

print("\nRunning SVGD...")
print(f"  observed_data shape: {observed_data.shape}")
print(f"  rewards shape: {rewards.shape}")

params = dict(
    observed_data=observed_data,  # Transposed!
    bandwidth='median',
    theta_dim=len(true_theta),
    prior=uninformative_prior,
    n_particles=24,
    n_iterations=n_iterations,
    seed=42,
    verbose=True,
    rewards=rewards,  # Transposed!
)

# graph.svgd() will auto-detect 2D rewards and use multivariate model
svgd = graph.svgd(**params)

print(f"\n✓ SVGD completed!")
print(f"  True theta: {true_theta}")
print(f"  Posterior mean: {svgd.theta_mean}")
print(f"  Posterior std: {svgd.theta_std}")
