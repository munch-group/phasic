"""
Debug sample moments computation
"""

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


true_theta = np.array([10])
nr_samples = 3

# Create graph and rewards
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

print(f"Rewards shape: {rewards.shape}")
print(f"Rewards:\n{rewards.T}")

# Create observed data (transposed)
n = 1000
a = np.empty((rewards.shape[1], n * rewards.shape[1]))
a[:] = np.nan
for i in range(rewards.shape[1]):
    a[i, i*n:(i+1)*n] = _graph.sample(n, rewards=rewards[:, i])

observed_data = jnp.array(a).T

print(f"\nObserved data shape: {observed_data.shape}")

# Manually compute sample moments per-feature
from phasic.svgd import compute_sample_moments

feature_moments = []
for i in range(observed_data.shape[1]):
    feature_data = observed_data[:, i]
    moments_i = compute_sample_moments(feature_data, nr_moments=2)
    print(f"\nFeature {i}:")
    print(f"  Non-NaN count: {(~jnp.isnan(feature_data)).sum()}")
    print(f"  Mean: {jnp.nanmean(feature_data):.6f}")
    print(f"  Moments: {moments_i}")
    feature_moments.append(moments_i)

sample_moments_agg = jnp.mean(jnp.array(feature_moments), axis=0)
print(f"\nAggregated sample moments: {sample_moments_agg}")

# Now create model and compute model moments at true parameters
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2)
pmf, model_moments = model(true_theta, observed_data, rewards=rewards)

print(f"\nModel moments shape: {model_moments.shape}")
print(f"Model moments:\n{model_moments}")

model_moments_agg = jnp.mean(model_moments, axis=0)
print(f"\nAggregated model moments: {model_moments_agg}")

print(f"\nDifference:")
print(f"  Sample: {sample_moments_agg}")
print(f"  Model:  {model_moments_agg}")
print(f"  Diff:   {model_moments_agg - sample_moments_agg}")
