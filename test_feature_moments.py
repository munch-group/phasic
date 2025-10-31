"""Test if model computes correct expected values for multivariate features"""
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

# Get rewards for 3 features
rewards = graph.states()[:, 1:-1]  # Shape: (6 vertices, 3 features)
print(f"rewards shape: {rewards.shape}")
print(f"rewards:\n{rewards}")

# First, let's check expected_waiting_time directly (this should work)
print("\nDirect expected_waiting_time test:")
for j in range(3):
    reward_j = rewards[:, j]
    ewt = graph.expected_waiting_time(reward_j)
    print(f"  Feature {j}: reward={reward_j}, E[R·T]={ewt[0]:.6f}")

# Create multivariate model with moments=1
model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
    graph, nr_moments=1, discrete=False, use_ffi=False, param_length=1
)

# Sample some actual data to test the model
nr_observations = 1000
observed_data = jnp.array([
    graph.sample(nr_observations, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T  # Shape: (nr_observations, 3)

print(f"\nobserved_data shape: {observed_data.shape}")
print(f"observed_data means: {jnp.mean(observed_data, axis=0)}")

# Compute PMF and moments for theta=1
print("\nCalling model with theta=1, observed_data, rewards...")
try:
    pmf, moments = model(theta, observed_data, rewards=rewards)
    print(f"  SUCCESS: model returned results")
    print(f"  pmf shape: {pmf.shape}, pmf min/max: {jnp.min(pmf):.6e}/{jnp.max(pmf):.6e}")
    print(f"  moments shape: {moments.shape}")
    print(f"  moments:\n{moments}")

    # Check if PMF has issues
    print(f"\n  PMF diagnostics:")
    print(f"    NaN count: {jnp.sum(jnp.isnan(pmf))}")
    print(f"    Inf count: {jnp.sum(jnp.isinf(pmf))}")
    print(f"    Zero count: {jnp.sum(pmf == 0)}")
    print(f"    Negative count: {jnp.sum(pmf < 0)}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print(f"\nExpected E[R·T] from direct computation: [1.0, 0.667, 0.0]")
print(f"User's expected values: [2.0, 1.0, 0.667]")
print(f"\nNOTE: User's values are 2× the computed values - might be different parameterization?")
