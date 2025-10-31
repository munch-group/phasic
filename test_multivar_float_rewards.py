"""Test with float rewards instead of int"""

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

print("Creating graphs...")
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)

print("Preparing rewards...")
rewards_raw = _graph.states().T[:-2]
rewards = rewards_raw.T

# KEY FIX: Convert to float64!
rewards = rewards.astype(np.float64)
print(f"rewards shape: {rewards.shape}, dtype: {rewards.dtype}")

print("Sampling data...")
nr_observations = 50
observed_data_raw = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards_raw])
observed_data = observed_data_raw.T
print(f"observed_data shape: {observed_data.shape}, dtype: {observed_data.dtype}")

def uninformative_prior(phi):
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

print("\nRunning SVGD...")
params = dict(
    observed_data=observed_data,
    bandwidth='median',
    theta_dim=1,
    prior=uninformative_prior,
    n_particles=8,
    n_iterations=3,
    seed=42,
    verbose=True,
    regularization=0,
    nr_moments=0,
    rewards=rewards,  # Now float64!
)

try:
    svgd = graph.svgd(**params)
    print("\n✓ SUCCESS!")
    print(f"  True theta: {true_theta}")
    if hasattr(svgd, 'theta_mean'):
        print(f"  Posterior mean: {svgd.theta_mean}")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
