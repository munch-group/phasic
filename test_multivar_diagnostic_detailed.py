"""Detailed diagnostic to find exactly where the segfault occurs"""

import phasic
import numpy as np
import jax.numpy as jnp
import sys

print("=" * 60)
print("STEP 1: Define coalescent function")
print("=" * 60)

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

print("✓ Coalescent function defined")

print("\n" + "=" * 60)
print("STEP 2: Create graphs")
print("=" * 60)

true_theta = np.array([10.0])
nr_samples = 4
print(f"true_theta = {true_theta}, nr_samples = {nr_samples}")

print("Creating first graph...")
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print("✓ First graph created")

print("Creating second graph for sampling...")
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print("✓ Second graph created")

print("Updating parameterized weights...")
_graph.update_parameterized_weights(true_theta)
print("✓ Weights updated")

print("\n" + "=" * 60)
print("STEP 3: Prepare rewards")
print("=" * 60)

print("Getting states...")
states = _graph.states()
print(f"  states shape: {states.shape}")

print("Transposing...")
rewards_raw = states.T
print(f"  rewards_raw shape after .T: {rewards_raw.shape}")

print("Removing last 2 rows...")
rewards_raw = rewards_raw[:-2]
print(f"  rewards_raw shape after [:-2]: {rewards_raw.shape}")

print("Final transpose...")
rewards = rewards_raw.T
print(f"✓ rewards shape: {rewards.shape}")
print(f"  Expected: (n_vertices, n_features)")

print("\n" + "=" * 60)
print("STEP 4: Sample observed data")
print("=" * 60)

nr_observations = 50  # Start with small number
print(f"Sampling {nr_observations} observations for {rewards_raw.shape[0]} features...")

try:
    observed_data_raw = jnp.array([_graph.sample(nr_observations, rewards=r) for r in rewards_raw])
    print(f"✓ Sampled successfully, shape: {observed_data_raw.shape}")

    print("Transposing observed_data...")
    observed_data = observed_data_raw.T
    print(f"✓ observed_data shape: {observed_data.shape}")
    print(f"  Expected: (n_observations, n_features)")
except Exception as e:
    print(f"✗ FAILED during sampling: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 5: Define prior")
print("=" * 60)

def uninformative_prior(phi):
    """Uninformative prior: φ ~ N(0, 10^2) - very wide"""
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

print("✓ Prior defined")

print("\n" + "=" * 60)
print("STEP 6: Prepare SVGD parameters")
print("=" * 60)

n_iterations = 3  # Very small for testing
n_particles = 8   # Very small for testing

params = dict(
    observed_data=observed_data,
    bandwidth='median',
    theta_dim=len(true_theta),
    prior=uninformative_prior,
    n_particles=n_particles,
    n_iterations=n_iterations,
    seed=42,
    verbose=True,
    regularization=0,
    nr_moments=0,
    rewards=rewards,
)

print(f"Parameters:")
for k, v in params.items():
    if isinstance(v, (np.ndarray, jnp.ndarray)):
        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
    elif callable(v):
        print(f"  {k}: <function>")
    else:
        print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("STEP 7: Run SVGD")
print("=" * 60)
print("This is where it might crash...")
sys.stdout.flush()

try:
    print("Calling graph.svgd()...")
    sys.stdout.flush()
    svgd = graph.svgd(**params)
    print("✓ SVGD completed successfully!")
    print(f"  Result type: {type(svgd)}")
    if hasattr(svgd, 'theta_mean'):
        print(f"  theta_mean: {svgd.theta_mean}")
except Exception as e:
    print(f"✗ FAILED during SVGD: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS - No segfault!")
print("=" * 60)
