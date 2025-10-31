"""
Test if sampling works with bypassed vertices
"""
import phasic
import numpy as np

def chain_3(state):
    """Chain: 0→1→2→3→absorbing"""
    if not state.size:
        return [([1], 1.0)]
    n = state[0]
    if n == 1:
        return [([2], 1.0)]
    elif n == 2:
        return [([3], 1.0)]
    elif n == 3:
        return [([4], 1.0)]
    else:
        return []

print("Creating graph...")
graph = phasic.Graph(callback=chain_3, parameterized=False)

# Test different reward vectors
test_cases = [
    ("No bypass", [1, 1, 1, 1, 1]),
    ("1 bypass", [1, 1, 0, 1, 1]),
    ("2 bypasses", [1, 1, 0, 0, 1]),
    ("3 bypasses", [1, 0, 0, 0, 1]),
]

for name, rewards in test_cases:
    print(f"\n{'='*60}")
    print(f"{name}: rewards={rewards}")

    # Apply reward_transform
    graph_t = graph.reward_transform(rewards)

    # Try sampling
    try:
        samples = graph_t.sample(100)
        mean = np.mean(samples)
        print(f"  Sampling works: n=100, mean={mean:.4f}")
    except Exception as e:
        print(f"  Sampling FAILED: {e}")

    # Try PDF
    try:
        pdf_val = graph_t.pdf(1.0, granularity=0)
        print(f"  PDF at t=1.0: {pdf_val:.6f}")
    except Exception as e:
        print(f"  PDF FAILED: {e}")
