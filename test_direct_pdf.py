"""Test direct PDF computation on coalescent graph"""
import phasic
import numpy as np
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

print("1. Creating coalescent graph...")
sys.stdout.flush()
true_theta = np.array([10.0])
nr_samples = 4
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=nr_samples)
print(f"   ✓ Graph has {graph.vertices_length()} vertices")
sys.stdout.flush()

print("\n2. Update weights...")
sys.stdout.flush()
graph.update_parameterized_weights(true_theta)
print("   ✓ Weights updated")
sys.stdout.flush()

print("\n3. Test direct PDF (C++)...")
sys.stdout.flush()
try:
    pdf_value = graph.pdf(1.0, granularity=100)
    print(f"   ✓ PDF(1.0) = {pdf_value}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ PDF failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Test PDF on multiple times...")
sys.stdout.flush()
try:
    times = np.array([0.5, 1.0, 1.5, 2.0])
    pdf_values = graph.pdf(times, granularity=100)
    print(f"   ✓ PDF at {len(times)} points: {pdf_values}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ Multiple PDF failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Test GraphBuilder (pybind)...")
sys.stdout.flush()
try:
    from phasic import phasic_pybind as cpp
    from phasic.ffi_wrappers import _make_json_serializable
    import json
    serialized = graph.serialize(param_length=1)
    json_dict = _make_json_serializable(serialized)
    json_str = json.dumps(json_dict)
    builder = cpp.parameterized.GraphBuilder(json_str)

    theta = np.array([10.0])
    times = np.array([1.0, 2.0, 3.0])

    print("   Calling builder.compute_pmf...")
    sys.stdout.flush()
    pmf = builder.compute_pmf(theta, times, discrete=False, granularity=0)
    print(f"   ✓ PMF via GraphBuilder: {pmf}")
    sys.stdout.flush()
except Exception as e:
    print(f"   ✗ GraphBuilder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nSuccess!")
