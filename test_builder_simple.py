"""Simple test of GraphBuilder with rewards"""
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

print("\nCreating model...")
model = phasic.Graph.pmf_and_moments_from_graph(
    graph, nr_moments=1, discrete=False, use_ffi=False
)

print("\nGetting rewards...")
rewards = graph.states()[:, :-2].astype(np.float64)
print(f"Rewards shape: {rewards.shape}")
reward_vec = rewards[:, 0]  # Feature 0
print(f"Reward vector: {reward_vec}")

print("\nTest 1: Model WITHOUT rewards")
import jax.numpy as jnp
theta = jnp.array([10.0])
times = jnp.array([0.1, 0.2, 0.3])

try:
    pmf, moments = model(theta, times, rewards=None)
    print(f"✓ PMF: {pmf}")
    print(f"✓ Moments: {moments}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest 2: Model WITH rewards")
try:
    pmf, moments = model(theta, times, rewards=reward_vec)
    print(f"✓ PMF: {pmf}")
    print(f"✓ Moments: {moments}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
