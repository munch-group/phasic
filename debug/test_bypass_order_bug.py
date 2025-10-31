"""
Demonstrate the vertex bypass order bug with detailed tracing
"""
import phasic
import numpy as np

def chain_3(state):
    """Chain of exactly 3 transient states: 0→1→2→3→absorbing"""
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

print("="*70)
print("Demonstrating vertex bypass order bug")
print("="*70)

graph = phasic.Graph(callback=chain_3, parameterized=False)

print(f"\nOriginal graph: {graph.vertices_length()} vertices")
for i in range(graph.vertices_length()):
    print(f"  Vertex {i}: state={graph.vertex_at(i).state()}")

# Test different bypass patterns
test_cases = [
    ("No bypass", [1, 1, 1, 1, 1]),
    ("Bypass vertex 2", [1, 1, 0, 1, 1]),
    ("Bypass vertices 2,3", [1, 1, 0, 0, 1]),
    ("Bypass vertices 1,2,3", [1, 0, 0, 0, 1]),
]

times = np.linspace(0.01, 20.0, 200)

print(f"\n{'='*70}")
print("Testing different bypass patterns")
print("="*70)

for name, rewards in test_cases:
    rewards_arr = np.array(rewards)

    try:
        graph_transformed = graph.reward_transform(rewards_arr.tolist())

        n_vertices = graph_transformed.vertices_length()
        n_bypassed = len(rewards) - n_vertices

        # Compute PDF
        pdf = [graph_transformed.pdf(t, granularity=0) for t in times]
        integral = np.trapezoid(pdf, times)

        status = "✓" if abs(integral - 1.0) < 0.1 else "❌"

        print(f"\n{name}:")
        print(f"  Rewards: {rewards}")
        print(f"  Vertices: {len(rewards)} → {n_vertices} ({n_bypassed} bypassed)")
        print(f"  PDF integral: {integral:.6f} {status}")

        if abs(integral - 1.0) > 0.1:
            print(f"  ⚠️  Normalization error: {abs(integral - 1.0):.6f}")

    except Exception as e:
        print(f"\n{name}:")
        print(f"  Rewards: {rewards}")
        print(f"  ❌ ERROR: {e}")

print(f"\n{'='*70}")
print("Pattern:")
print("  - Single bypass: OK")
print("  - Multiple consecutive bypasses: BROKEN")
print("  - Bug: Parent-child relationships not updated between bypasses")
print("="*70)
