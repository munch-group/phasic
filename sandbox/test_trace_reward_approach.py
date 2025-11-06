"""Test using trace + instantiate + reward_transform"""
import phasic
import numpy as np
import jax.numpy as jnp
from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

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

print("Step 1: Create parameterized graph and record trace...")
graph_param = phasic.Graph(callback=coalescent, parameterized=True, nr_samples=5)
trace = record_elimination_trace(graph_param, param_length=1)
print(f"✓ Trace recorded: {len(trace.operations)} operations")

print("\nStep 2: Get rewards...")
rewards = graph_param.states()[:, :-2].astype(np.float64)
reward_vec = rewards[:, 0]
print(f"✓ Reward vector: {reward_vec}")

print("\nStep 3: Instantiate concrete graph from trace...")
theta = np.array([10.0])
graph_concrete = instantiate_from_trace(trace, theta)
print(f"✓ Concrete graph instantiated")

print("\nStep 4: Apply reward_transform to concrete graph...")
graph_transformed = graph_concrete.reward_transform(reward_vec)
print(f"✓ Reward transform succeeded")

print("\nStep 5: Compute PDF on transformed graph...")
pdf_val = graph_transformed.pdf(0.1, granularity=100)
print(f"✓ PDF at 0.1: {pdf_val}")

if not np.isnan(pdf_val) and pdf_val > 0:
    print("\n✓✓✓ SUCCESS! This approach works:")
    print("    1. Record trace from parameterized graph")
    print("    2. Instantiate concrete graph with theta")
    print("    3. Apply reward_transform")
    print("    4. Compute PDF")

    print("\nStep 6: Sample from original graph and compare...")
    n_samples = 10000
    samples = graph_param.sample(n_samples, rewards=reward_vec)
    empirical_mean = np.mean(samples)
    print(f"✓ Empirical mean from samples: {empirical_mean:.6f}")

    # Compute expected mean from transformed graph
    moments = graph_transformed.moments(1)
    expected_mean = moments[0]
    print(f"✓ Expected mean from transformed graph: {expected_mean:.6f}")

    if abs(empirical_mean - expected_mean) < 0.01:
        print("\n✓✓✓ Means match! Reward transformation is correct!")
    else:
        print(f"\n✗ Means don't match (diff: {abs(empirical_mean - expected_mean):.6f})")
else:
    print("\n✗ PDF is still NaN or zero")
