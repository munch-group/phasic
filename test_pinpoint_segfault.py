"""Pinpoint exact location of segfault"""
import phasic
import numpy as np

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

print("Step 1: Creating parameterized graph...")
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
print("✓ Graph created")

print("\nStep 2: Getting rewards...")
rewards = graph.states()[:, :-2].astype(np.float64)
print(f"✓ Rewards shape: {rewards.shape}")
reward_vec = rewards[:, 0]
print(f"✓ Reward vector: {reward_vec}")

print("\nStep 3: Creating GraphBuilder directly...")
from phasic.phasic_pybind import parameterized
import json
from phasic.ffi_wrappers import _make_json_serializable

serialized = graph.serialize()
print(f"✓ Serialized (param_length={serialized['param_length']})")

structure_json_dict = _make_json_serializable(serialized)
print(f"✓ Made JSON serializable")

# Check for NaN in structure
import math
def check_nan(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_nan(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_nan(v, f"{path}[{i}]")
    elif isinstance(obj, float) and math.isnan(obj):
        print(f"✗ Found NaN at: {path}")
    return

check_nan(structure_json_dict)

structure_json_str = json.dumps(structure_json_dict)
print(f"✓ JSON dumped (length={len(structure_json_str)})")

builder = parameterized.GraphBuilder(structure_json_str)
print(f"✓ GraphBuilder created")

print("\nStep 4: Calling compute_pmf_and_moments WITHOUT rewards...")
theta = np.array([10.0])
times = np.array([0.1, 0.2, 0.3])

try:
    pmf, moments = builder.compute_pmf_and_moments(
        theta, times, nr_moments=1, discrete=False, granularity=0, rewards=None
    )
    print(f"✓ Without rewards: PMF = {pmf}, moments = {moments}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 5: Calling compute_pmf_and_moments WITH rewards...")
print(f"Rewards dtype: {reward_vec.dtype}, shape: {reward_vec.shape}")
try:
    pmf, moments = builder.compute_pmf_and_moments(
        theta, times, nr_moments=1, discrete=False, granularity=0, rewards=reward_vec
    )
    print(f"✓ With rewards: PMF = {pmf}, moments = {moments}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ All tests completed without segfault!")
