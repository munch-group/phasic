"""Test all 4 features with 10,000 observations"""
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

print("Testing ALL 4 features with 10,000 observations")
print("=" * 70)

true_theta = np.array([10.0])
nr_samples = 5
nr_observations = 10000

# Create data
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

print(f"Rewards shape: {rewards.shape}")
print(f"Number of features: {rewards.shape[1]}")

# Sample data with all features (NO NaN)
np.random.seed(42)
observed_data = jnp.array([
    _graph.sample(nr_observations, rewards=rewards[:, i])
    for i in range(rewards.shape[1])
]).T

print(f"Observed data shape: {observed_data.shape}")
print(f"Data means: {jnp.mean(observed_data, axis=0)}")
print(f"Data stds: {jnp.std(observed_data, axis=0)}")

# Create fresh graph for SVGD
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)

n_iterations = 1000
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
    n_particles=24,
    n_iterations=n_iterations,
    learning_rate=step_schedule,
    seed=42,
    verbose=True,
    rewards=rewards,
    regularization=0,  # No regularization
    nr_moments=1,
)

print(f"\nRunning SVGD with 10,000 observations × 4 features = 40,000 data points...")
print(f"Iterations: {n_iterations}")
svgd = graph.svgd(**params)

# Get final theta estimate
results = svgd.get_results()
final_theta = results['theta_mean']
final_std = results['theta_std']

print(f"\n{'=' * 70}")
print("RESULTS")
print(f"{'=' * 70}")
print(f"  True theta:      {true_theta[0]:.2f}")
print(f"  Estimated theta: {final_theta[0]:.2f} ± {final_std[0]:.2f}")
error = final_theta[0] - true_theta[0]
error_pct = abs(error) / true_theta[0] * 100
print(f"  Error:           {error:+.2f} ({error_pct:.1f}%)")

if error_pct < 5:
    print(f"  ✓ EXCELLENT: Error < 5%")
elif error_pct < 10:
    print(f"  ✓ GOOD: Error < 10%")
elif error_pct < 15:
    print(f"  ⚠ ACCEPTABLE: Error < 15%")
else:
    print(f"  ✗ POOR: Error > 15%")

print(f"\n{'=' * 70}")
print("COMPARISON:")
print(f"  Feature 0 alone (10K obs):     7.51 ± 0.65 (24.9% error)")
print(f"  Feature 1 alone (10K obs):    12.97 ± 2.20 (29.7% error)")
print(f"  All 4 features (1K obs):      10.88 ± 1.52 (8.8% error)")
print(f"  All 4 features (10K obs):     {final_theta[0]:.2f} ± {final_std[0]:.2f} ({error_pct:.1f}% error)")
print(f"{'=' * 70}")
