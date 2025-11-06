"""Verify likelihood at different theta values matches diagnostic"""
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

print("Generating data...")
true_theta = np.array([10.0])
nr_samples = 5
nr_observations = 10000

_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)

rewards = _graph.states()[:, :-2]
print(f"Rewards shape: {rewards.shape}")

# Generate same data as multivar_test.py
np.random.seed(42)
observed_data = jnp.array([
    _graph.sample(nr_observations, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T
print(f"Observed data shape: {observed_data.shape}")

# Create model using the same approach as graph.svgd()
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

# This is what graph.svgd() does internally
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=2, discrete=False, use_ffi=False, param_length=1
)

# Test likelihood at different theta values
print("\nTesting log-likelihoods (using FIRST 100 observations)...")
test_data = observed_data[:100, :]

test_thetas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 709.0]

for theta_val in test_thetas:
    theta = jnp.array([theta_val])
    pmf, _ = model(theta, test_data, rewards=rewards)
    log_lik = jnp.sum(jnp.log(pmf + 1e-10))
    print(f"  theta={theta_val:7.1f}: log-lik={log_lik:12.4f}")
