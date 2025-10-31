"""Test coalescent SVGD with FFI disabled"""
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

print("Creating data...")
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
nr_observations = 50
observed_data = jnp.array(_graph.sample(nr_observations))
print(f"Data shape: {observed_data.shape}")

print("\nCreating model with use_ffi=False...")
model = phasic.Graph.pmf_and_moments_from_graph(
    graph,
    nr_moments=0,
    discrete=False,
    use_ffi=False,  # DISABLE FFI
    param_length=1
)
print("✓ Model created")

print("\nTesting model call...")
test_theta = jnp.array([10.0])
test_times = observed_data[:10]
try:
    pmf, moments = model(test_theta, test_times)
    print(f"✓ Model works! pmf shape: {pmf.shape}")
except Exception as e:
    print(f"✗ Model call failed: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\nRunning SVGD...")
def uninformative_prior(phi):
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)

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
    use_ffi=False,  # DISABLE FFI
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
