"""Test model creation without FFI"""
import phasic
import numpy as np
import jax.numpy as jnp
import sys

# Disable FFI
config = phasic.get_config()
config.ffi = False
print(f"FFI disabled: config.ffi={config.ffi}")

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

print("\n1. Creating graph...")
sys.stdout.flush()
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print("   ✓ Graph created")
sys.stdout.flush()

print("\n2. Creating model (no FFI)...")
sys.stdout.flush()
model = phasic.Graph.pmf_and_moments_from_graph(graph, nr_moments=0, discrete=False, param_length=1)
print("   ✓ Model created")
sys.stdout.flush()

print("\n3. Testing model call...")
sys.stdout.flush()
test_theta = jnp.array([10.0])
test_times = jnp.array([1.0, 2.0, 3.0])
print(f"   Calling model(theta={test_theta}, times={test_times}, rewards=None)...")
sys.stdout.flush()

try:
    pmf, moments = model(test_theta, test_times, rewards=None)
    print(f"   ✓ Model call succeeded!")
    print(f"   PMF shape: {pmf.shape}, values: {pmf}")
    print(f"   Moments shape: {moments.shape}, values: {moments}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ Model call FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

print("\nSuccess!")
