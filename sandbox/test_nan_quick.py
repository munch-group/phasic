"""Quick test of NaN support with single feature"""
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

print("Quick NaN test - Feature 0 only")

true_theta = np.array([10.0])
nr_samples = 5
nr_observations = 100  # Small dataset

# Create data generation graph
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

n_features = rewards.shape[1]
feature_idx = 0

# Generate data for feature 0 only
np.random.seed(42)
observed_feature = jnp.array(_graph.sample(nr_observations, rewards=rewards[:, feature_idx]))

# Create observed_data with NaN for other features
observed_data = jnp.full((nr_observations, n_features), jnp.nan)
observed_data = observed_data.at[:, feature_idx].set(observed_feature)

print(f"Observed data shape: {observed_data.shape}")
print(f"Non-NaN count: {jnp.sum(~jnp.isnan(observed_data))}")
print(f"Feature {feature_idx} mean: {jnp.nanmean(observed_data[:, feature_idx]):.4f}")

# Create fresh graph for SVGD
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

# Run SVGD with very few iterations
n_iterations = 50
step_schedule = phasic.ExponentialDecayStepSize(
    first_step=0.001, last_step=0.0001, tau=n_iterations/5
)

def uninformative_prior(phi):
    return -0.5 * jnp.sum(((phi - 0.0) / 10.0)**2)

params = dict(
    observed_data=observed_data,
    bandwidth='median',
    theta_dim=len(true_theta),
    prior=uninformative_prior,
    n_particles=12,  # Fewer particles
    n_iterations=n_iterations,
    learning_rate=step_schedule,
    seed=42,
    verbose=True,
    rewards=rewards,
)

print(f"\nRunning SVGD for {n_iterations} iterations...")
svgd = graph.svgd(**params)

# Get final theta estimate
results = svgd.get_results()
final_theta = results['theta_mean']
final_std = results['theta_std']

print(f"\nResults:")
print(f"  True theta:      {true_theta[0]:.2f}")
print(f"  Estimated theta: {final_theta[0]:.2f} ± {final_std[0]:.2f}")
print(f"  Error:           {abs(final_theta[0] - true_theta[0]):.2f} ({abs(final_theta[0] - true_theta[0])/true_theta[0]*100:.1f}%)")
