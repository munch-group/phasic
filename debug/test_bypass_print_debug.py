"""
Add print debugging to see what's happening during bypass
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

print(f"Original graph: {graph.vertices_length()} vertices")
for i in range(graph.vertices_length()):
    print(f"  Vertex {i}: state={graph.vertex_at(i).state()}")

# Test case: 3 consecutive bypasses
rewards = [1, 0, 0, 0, 1]
print(f"\nTesting rewards={rewards}")
print("Expected: bypass vertices 1,2,3 → final graph has vertices 0,4")

graph_t = graph.reward_transform(rewards)

print(f"\nResult: {graph.vertices_length()} → {graph_t.vertices_length()} vertices")
print(f"Expected: 5 → 2 vertices")

# Try sampling
samples = graph_t.sample(10)
print(f"\nSamples: {samples[:5]}")
print(f"Mean: {np.mean(samples)}")

# Try PDF
pdf_val = graph_t.pdf(1.0, granularity=0)
print(f"PDF at t=1.0: {pdf_val}")

if np.mean(samples) == 0 and pdf_val == 0:
    print("\n❌ Graph is completely broken - both sampling and PDF return 0")
else:
    print("\n✓ Graph has some valid structure")
