"""Minimal test to debug reward_transform segfault"""
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

print("Creating parameterized graph...")
graph = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)

print("Updating weights with theta=[10.0]...")
graph.update_parameterized_weights(np.array([10.0]))

print("Getting rewards...")
rewards = graph.states()[:, :-2]
print(f"Rewards shape: {rewards.shape}")

print("\nTesting reward_transform on parameterized graph...")
reward_vec = rewards[:, 0].astype(np.float64)
print(f"Reward vector: {reward_vec}")

try:
    print("Calling graph.reward_transform...")
    reward_graph = graph.reward_transform(reward_vec)
    print("✓ reward_transform succeeded")

    print("Computing PDF...")
    pdf_value = reward_graph.pdf(0.1)
    print(f"PDF at 0.1: {pdf_value}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nNow testing through GraphBuilder...")
from phasic.phasic_pybind import parameterized
import json
from phasic.ffi_wrappers import _make_json_serializable

# Serialize graph
serialized = graph.serialize()
print(f"Param length: {serialized['param_length']}")

# Create builder
structure_json = json.dumps(_make_json_serializable(serialized))
builder = parameterized.GraphBuilder(structure_json)

print("Testing compute_pmf_and_moments WITHOUT rewards...")
theta = np.array([10.0])
times = np.array([0.1, 0.2, 0.3])
try:
    pmf, moments = builder.compute_pmf_and_moments(
        theta, times, nr_moments=1, discrete=False, granularity=0, rewards=None
    )
    print(f"✓ Without rewards: PMF = {pmf[:3]}, moment = {moments}")
except Exception as e:
    print(f"✗ Error without rewards: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting compute_pmf_and_moments WITH rewards...")
try:
    pmf, moments = builder.compute_pmf_and_moments(
        theta, times, nr_moments=1, discrete=False, granularity=0, rewards=reward_vec
    )
    print(f"✓ With rewards: PMF = {pmf[:3]}, moment = {moments}")
except Exception as e:
    print(f"✗ Error with rewards: {e}")
    import traceback
    traceback.print_exc()
