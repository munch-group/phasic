"""Test just the model creation without SVGD"""

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
print("✓ Graph created")

print("\nPreparing rewards...")
_graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
_graph.update_parameterized_weights(true_theta)
rewards_raw = _graph.states().T[:-2]
rewards = rewards_raw.T.astype(np.float64)
print(f"✓ rewards shape: {rewards.shape}, dtype: {rewards.dtype}")

print("\nTesting multivariate model creation...")
try:
    print("Calling Graph.pmf_and_moments_from_graph_multivariate()...")
    model = phasic.Graph.pmf_and_moments_from_graph_multivariate(
        graph,
        nr_moments=0,
        discrete=False,
        use_ffi=False,
        param_length=1
    )
    print("✓ Model created successfully!")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ FAILED during model creation: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\nTesting model evaluation with concrete parameters...")
try:
    test_theta = jnp.array([10.0])
    test_times = jnp.array([[1.0, 2.0, 3.0]] * 10)  # (10, 3) shape for 3 features

    print(f"  test_theta shape: {test_theta.shape}")
    print(f"  test_times shape: {test_times.shape}")
    print(f"  rewards shape: {rewards.shape}")

    print("Calling model function...")
    pmf, moments = model(test_theta, test_times, rewards=rewards)
    print("✓ Model evaluation successful!")
    print(f"  pmf shape: {pmf.shape}")
    print(f"  moments shape: {moments.shape}")
except Exception as e:
    print(f"✗ FAILED during model evaluation: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS - Model works!")
print("="*60)
