"""Step-by-step SVGD creation to find where it crashes"""
import phasic
import numpy as np
import jax.numpy as jnp
import sys

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
sys.stdout.flush()
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print("✓ Graph created")
sys.stdout.flush()

print("\nCreating data...")
sys.stdout.flush()
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
nr_observations = 50
observed_data = jnp.array(_graph.sample(nr_observations))
print(f"✓ Data shape: {observed_data.shape}")
sys.stdout.flush()

print("\nCreating model...")
sys.stdout.flush()
try:
    model = phasic.Graph.pmf_and_moments_from_graph(
        graph,
        nr_moments=0,
        discrete=False,
        param_length=1
    )
    print("✓ Model created")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting model call...")
sys.stdout.flush()
try:
    test_theta = jnp.array([10.0])
    test_times = observed_data[:10]  # Just first 10
    pmf, moments = model(test_theta, test_times)
    print(f"✓ Model callable, pmf shape: {pmf.shape}")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ Model call failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCreating prior...")
sys.stdout.flush()
def uninformative_prior(phi):
    mu = 0.0
    sigma = 10.0
    return -0.5 * jnp.sum(((phi - mu) / sigma)**2)
print("✓ Prior created")
sys.stdout.flush()

print("\nCreating SVGD object...")
sys.stdout.flush()
try:
    svgd_obj = phasic.SVGD(
        model=model,
        observed_data=observed_data,
        theta_dim=1,
        prior=uninformative_prior,
        n_particles=8,
        n_iterations=3,
        seed=42,
        verbose=True,
        bandwidth='median',
        regularization=0,
        nr_moments=0,
    )
    print("✓ SVGD object created")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ SVGD object creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nCalling SVGD.fit()...")
sys.stdout.flush()
try:
    result = svgd_obj.fit()
    print("✓ SVGD.fit() completed!")
    print(f"  Result type: {type(result)}")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ SVGD.fit() failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
