"""
Test vertex bypass logic in reward_transform
"""
import phasic
import numpy as np

def simple_chain(state):
    """Simple linear chain: 0 → 1 → 2 → absorbing"""
    if not state.size:
        return [([1], 1.0)]  # Start at vertex 1

    n = state[0]
    if n == 1:
        return [([2], 1.0)]
    elif n == 2:
        return [([3], 1.0)]
    else:
        return []  # Absorbing

print("="*70)
print("Test 1: Simple chain with vertex bypass")
print("="*70)

# Create simple chain graph
graph = phasic.Graph(callback=simple_chain, parameterized=False)

print(f"\nOriginal graph:")
print(f"  Vertices: {graph.vertices_length()}")
for i in range(graph.vertices_length()):
    state = graph.vertex_at(i).state()
    print(f"  Vertex {i}: state={state}")

# Test Case 1: All rewards = 1 (no bypass)
print(f"\n{'='*70}")
print("Case 1: All rewards = 1 (no vertices bypassed)")
print("="*70)

rewards_all_1 = np.ones(graph.vertices_length())
graph_all_1 = graph.reward_transform(rewards_all_1.tolist())

print(f"  Original vertices: {graph.vertices_length()}")
print(f"  Transformed vertices: {graph_all_1.vertices_length()}")
print(f"  Expected: same number (no bypass)")

# Compute PDF
times = np.linspace(0.01, 10.0, 100)
pdf_all_1 = [graph_all_1.pdf(t, granularity=0) for t in times]
integral_all_1 = np.trapezoid(pdf_all_1, times)

print(f"  PDF integral: {integral_all_1:.6f} (expected: 1.0)")

# Test Case 2: Middle vertex has reward=0 (should be bypassed)
print(f"\n{'='*70}")
print("Case 2: Middle vertex has reward=0 (bypass vertex 2)")
print("="*70)

# Vertices: [0, 1, 2, 3] → bypass vertex 2
# Expected: 0 → 1 → 3 (skipping 2)
rewards_bypass_middle = np.array([1, 1, 0, 1])  # Bypass vertex 2
graph_bypass = graph.reward_transform(rewards_bypass_middle.tolist())

print(f"  Original vertices: {graph.vertices_length()}")
print(f"  Transformed vertices: {graph_bypass.vertices_length()}")
print(f"  Expected: 3 (vertex 2 removed)")

pdf_bypass = [graph_bypass.pdf(t, granularity=0) for t in times]
integral_bypass = np.trapezoid(pdf_bypass, times)

print(f"  PDF integral: {integral_bypass:.6f} (expected: 1.0)")

if abs(integral_bypass - 1.0) > 0.1:
    print(f"  ❌ PDF not normalized! (off by {abs(integral_bypass - 1.0):.3f})")
else:
    print(f"  ✓ PDF properly normalized")

# Test Case 3: Two vertices bypassed (very sparse)
print(f"\n{'='*70}")
print("Case 3: Two vertices have reward=0 (bypass vertices 1 and 2)")
print("="*70)

rewards_sparse = np.array([1, 0, 0, 1])  # Bypass vertices 1 and 2
graph_sparse = graph.reward_transform(rewards_sparse.tolist())

print(f"  Original vertices: {graph.vertices_length()}")
print(f"  Transformed vertices: {graph_sparse.vertices_length()}")
print(f"  Expected: 2 (vertices 1,2 removed)")

pdf_sparse = [graph_sparse.pdf(t, granularity=0) for t in times]
integral_sparse = np.trapezoid(pdf_sparse, times)

print(f"  PDF integral: {integral_sparse:.6f} (expected: 1.0)")

if abs(integral_sparse - 1.0) > 0.1:
    print(f"  ❌ PDF not normalized! (off by {abs(integral_sparse - 1.0):.3f})")
    print(f"  This confirms the bug in vertex bypass logic!")
else:
    print(f"  ✓ PDF properly normalized")

# Compare PDFs
print(f"\n{'='*70}")
print("Summary")
print("="*70)
print(f"All rewards=1: integral = {integral_all_1:.6f}")
print(f"1 bypass:      integral = {integral_bypass:.6f}")
print(f"2 bypasses:    integral = {integral_sparse:.6f}")

if integral_all_1 > 0.9 and (integral_bypass < 0.9 or integral_sparse < 0.9):
    print(f"\n❌ CONFIRMED: Vertex bypass causes PDF normalization bug!")
    print(f"   More bypasses → worse normalization")
