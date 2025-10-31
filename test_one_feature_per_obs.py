"""Test with ONE feature per observation (others NaN) to avoid double-counting"""
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

print("Testing with ONE feature per observation (NaN for others)")
print("=" * 70)

true_theta = np.array([10.0])
nr_samples = 5
n = 2500  # 2500 obs per feature

# Create data generation graph
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards = _graph.states()[:, :-2]

print(f"Rewards shape: {rewards.shape}")
print(f"Number of features: {rewards.shape[1]}")

# Generate data: ONE feature per observation
# Observations 0-2499: Feature 0 (others NaN)
# Observations 2500-4999: Feature 1 (others NaN)
# Observations 5000-7499: Feature 2 (others NaN)
# Observations 7500-9999: Feature 3 (others NaN)

np.random.seed(42)
a = np.empty((rewards.shape[1], n * rewards.shape[1]))
a[:] = np.nan
for i in range(rewards.shape[1]):
    a[i, i*n:(i+1)*n] = _graph.sample(n, rewards=rewards[:, i])
observed_data = jnp.array(a).T

total_obs = n * rewards.shape[1]
print(f"\nData structure:")
print(f"  Total observations: {total_obs}")
print(f"  Observations per feature: {n}")
print(f"  Non-NaN values per observation: 1")
print(f"  Total non-NaN values: {jnp.sum(~jnp.isnan(observed_data))}")

# Verify data structure
print(f"\nObserved data shape: {observed_data.shape}")
print(f"First 5 obs (should have only feature 0):")
print(observed_data[:5])
print(f"\nObs 2500-2505 (should have only feature 1):")
print(observed_data[2500:2505])

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
    regularization=0,
    nr_moments=1,
)

print(f"\nRunning SVGD with {total_obs} observations (1 feature each)...")
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
print(f"  All 4 features, all obs (1K × 4 = 4K points):    10.88 ± 1.52 (8.8%)")
print(f"  All 4 features, all obs (10K × 4 = 40K points):  13.24 ± 1.09 (32.4%)")
print(f"  One feature per obs (10K obs, 10K points):       {final_theta[0]:.2f} ± {final_std[0]:.2f} ({error_pct:.1f}%)")
print(f"{'=' * 70}")
print("\nIf this works better, it confirms the issue:")
print("  → Treating each (obs, feature) pair as independent inflates sample size")
print("  → Solution: Use NaN to mark missing features (one per observation)")
print(f"{'=' * 70}")
