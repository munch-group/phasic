"""Test GraphBuilder.compute_pmf_and_moments directly (no JAX)"""
import phasic
from phasic import phasic_pybind as cpp
from phasic.ffi_wrappers import _make_json_serializable
import numpy as np
import json
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

print("1. Creating graph...")
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=4)
print(f"   ✓ Graph created with {graph.vertices_length()} vertices")

print("\n2. Serializing graph...")
serialized = graph.serialize(param_length=1)
json_dict = _make_json_serializable(serialized)
json_str = json.dumps(json_dict)
print(f"   ✓ Serialized to JSON ({len(json_str)} bytes)")

print("\n3. Creating GraphBuilder...")
sys.stdout.flush()
builder = cpp.parameterized.GraphBuilder(json_str)
print("   ✓ GraphBuilder created")
sys.stdout.flush()

print("\n4. Calling compute_pmf_and_moments...")
theta = np.array([10.0])
times = np.array([1.0, 2.0, 3.0])
nr_moments = 0  # Test with nr_moments=0 (should work now!)
sys.stdout.flush()

try:
    pmf, moments = builder.compute_pmf_and_moments(
        theta, times,
        nr_moments=nr_moments,
        discrete=False,
        granularity=0,
        rewards=None
    )
    print(f"   ✓ Success!")
    print(f"   PMF: {pmf}")
    print(f"   Moments: {moments}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

print("\n5. Testing with batch...")
theta_batch = np.array([[10.0], [12.0], [8.0]])
print(f"   theta_batch shape: {theta_batch.shape}")
sys.stdout.flush()

try:
    for i, theta_single in enumerate(theta_batch):
        pmf, moments = builder.compute_pmf_and_moments(
            theta_single, times,
            nr_moments=nr_moments,
            discrete=False,
            granularity=0,
            rewards=None
        )
        print(f"   Batch {i}: PMF={pmf}")
    print("   ✓ Batch test success!")
except Exception as e:
    print(f"   ✗ Batch FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed!")
