"""Test if Python-level reward transformation works"""
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
graph_param = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)

print("Updating weights with theta=[10.0]...")
graph_param.update_parameterized_weights(np.array([10.0]))

print("\nGetting rewards...")
rewards = graph_param.states()[:, :-2].astype(np.float64)
print(f"Rewards shape: {rewards.shape}")
reward_vec = rewards[:, 0]
print(f"Reward vector for feature 0: {reward_vec}")

print("\nAttempt 1: reward_transform on parameterized graph")
try:
    graph_transformed = graph_param.reward_transform(reward_vec)
    pdf_val = graph_transformed.pdf(0.1)
    print(f"✓ PDF at 0.1: {pdf_val}")
    if np.isnan(pdf_val):
        print("  ⚠ PDF is NaN - reward_transform doesn't work on parameterized graphs")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nAttempt 2: Create non-parameterized graph first, then transform")
# Create non-parameterized version by building with concrete theta
def coalescent_concrete(state, nr_samples=None):
    theta = 10.0  # Concrete value
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
                rate = theta * state[i]*(state[j]-same)/(1+same)
                transitions.append([new, rate, []])  # Non-parameterized
        return transitions

graph_concrete = phasic.Graph(callback=coalescent_concrete, parameterized=False, nr_samples=5)
print("✓ Created concrete graph")

print("Applying reward_transform...")
graph_transformed = graph_concrete.reward_transform(reward_vec)
print("✓ Transformation succeeded")

print("Computing PDF...")
pdf_val = graph_transformed.pdf(0.1)
print(f"✓ PDF at 0.1: {pdf_val}")

if not np.isnan(pdf_val):
    print("\n✓✓✓ SUCCESS! reward_transform works on concrete (non-parameterized) graphs")
    print("    → Solution: Build concrete graph for each theta, then transform")
else:
    print("\n✗ Still getting NaN")
