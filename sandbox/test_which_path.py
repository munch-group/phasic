"""Diagnostic: Determine which code path is being used"""
import phasic
import numpy as np
import jax.numpy as jnp
import sys

print("=" * 70)
print("DIAGNOSTIC: Which code path is being used?")
print("=" * 70)

print("\n1. Checking configuration...")
config = phasic.get_config()
print(f"   config.ffi = {config.ffi}")
print(f"   config.jit = {config.jit}")
print(f"   config.openmp = {config.openmp}")

print("\n2. Checking FFI availability...")
try:
    from phasic import phasic_pybind as cpp_module
    has_ffi_capsule = hasattr(cpp_module.parameterized, 'get_compute_pmf_ffi_capsule')
    print(f"   FFI capsule available: {has_ffi_capsule}")
except Exception as e:
    print(f"   Error checking FFI: {e}")
    has_ffi_capsule = False

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

print("\n3. Creating graph...")
sys.stdout.flush()
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print("   ✓ Graph created")
sys.stdout.flush()

print("\n4. Creating model with instrumentation...")
sys.stdout.flush()

# Monkey-patch to see which path is used
original_pmf_and_moments = phasic.Graph.pmf_and_moments_from_graph

def instrumented_pmf_and_moments(graph, nr_moments=2, discrete=False, use_ffi=False, param_length=None):
    print(f"   >>> pmf_and_moments_from_graph called with use_ffi={use_ffi}")
    sys.stdout.flush()

    # Call original
    result = original_pmf_and_moments(graph, nr_moments, discrete, use_ffi, param_length)

    # Wrap the result to trace calls
    original_model = result
    def traced_model(theta, times, rewards=None):
        print(f"   >>> Model called: theta shape={theta.shape}, times shape={times.shape}, rewards={'None' if rewards is None else rewards.shape}")
        sys.stdout.flush()
        try:
            return original_model(theta, times, rewards)
        except Exception as e:
            print(f"   >>> Model FAILED: {e}")
            sys.stdout.flush()
            raise

    return traced_model

phasic.Graph.pmf_and_moments_from_graph = instrumented_pmf_and_moments

try:
    model = phasic.Graph.pmf_and_moments_from_graph(graph, nr_moments=0, discrete=False, param_length=1)
    print("   ✓ Model created")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ Model creation FAILED: {e}")
    sys.stdout.flush()
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing model WITHOUT rewards...")
sys.stdout.flush()
try:
    test_theta = jnp.array([10.0])
    test_times = jnp.array([1.0, 2.0, 3.0])
    print(f"   Calling model(theta, times, rewards=None)...")
    sys.stdout.flush()
    pmf, moments = model(test_theta, test_times, rewards=None)
    print(f"   ✓ Model works without rewards: pmf shape={pmf.shape}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ Model FAILED without rewards: {e}")
    sys.stdout.flush()
    import traceback
    traceback.print_exc()

print("\n6. Testing model WITH rewards...")
sys.stdout.flush()
try:
    _graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
    _graph.update_parameterized_weights(true_theta)
    rewards_raw = _graph.states().T[:-2]
    reward_vec = rewards_raw.T[:, 0]  # Just first feature
    print(f"   reward_vec shape: {reward_vec.shape}")
    print(f"   Calling model(theta, times, rewards=reward_vec)...")
    sys.stdout.flush()
    pmf, moments = model(test_theta, test_times, rewards=reward_vec)
    print(f"   ✓ Model works with rewards: pmf shape={pmf.shape}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ Model FAILED with rewards: {e}")
    sys.stdout.flush()
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Diagnostic complete")
print("=" * 70)
