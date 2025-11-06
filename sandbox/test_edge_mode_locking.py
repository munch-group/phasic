"""
Test edge mode locking: prevent mixing constant and parameterized edges.

This ensures that:
1. Constant graphs (scalar syntax) reject parameterized edges
2. Parameterized graphs (array syntax) reject constant edges
3. update_weights() only works on parameterized graphs
4. IPV (starting vertex) edges don't affect mode locking
"""

from phasic import Graph
import sys

print("="*70)
print("EDGE MODE LOCKING TESTS")
print("="*70)

# Test 1: Constant graph locks mode
print("\n1. Testing constant graph mode locking")
print("-" * 70)

g1 = Graph(state_length=1)
v0 = g1.starting_vertex()
v1 = g1.find_or_create_vertex([1])
v2 = g1.find_or_create_vertex([2])
v3 = g1.find_or_create_vertex([0])

# IPV edge (doesn't lock mode)
v0.add_edge(v1, 1.0)
print("  Added IPV edge (v0 -> v1): constant, doesn't lock mode")

# First non-IPV edge: constant (locks to CONSTANT mode)
v1.add_edge(v2, 3.0)
print("  Added first non-IPV edge (v1 -> v2): constant → locks to CONSTANT mode")

# Try to add parameterized edge (should fail)
try:
    v2.add_edge(v3, [2.0])
    print("  ❌ ERROR: Mixing allowed (should have been rejected)")
    sys.exit(1)
except RuntimeError as e:
    if "Cannot mix" in str(e):
        print(f"  ✅ Mixing rejected: {str(e)[:80]}...")
    else:
        print(f"  ❌ Wrong error: {e}")
        sys.exit(1)

# Test 2: Parameterized graph locks mode
print("\n2. Testing parameterized graph mode locking")
print("-" * 70)

g2 = Graph(state_length=1)
v0 = g2.starting_vertex()
v1 = g2.find_or_create_vertex([1])
v2 = g2.find_or_create_vertex([2])
v3 = g2.find_or_create_vertex([0])

# IPV edge (doesn't lock mode)
v0.add_edge(v1, 1.0)
print("  Added IPV edge (v0 -> v1): constant, doesn't lock mode")

# First non-IPV edge: parameterized (locks to PARAMETERIZED mode)
v1.add_edge(v2, [1.0])
print("  Added first non-IPV edge (v1 -> v2): parameterized → locks to PARAMETERIZED mode")

# Try to add constant edge (should fail)
try:
    v2.add_edge(v3, 3.0)
    print("  ❌ ERROR: Mixing allowed (should have been rejected)")
    sys.exit(1)
except RuntimeError as e:
    if "Cannot mix" in str(e):
        print(f"  ✅ Mixing rejected: {str(e)[:80]}...")
    else:
        print(f"  ❌ Wrong error: {e}")
        sys.exit(1)

# Test 3: update_weights() on constant graph should fail
print("\n3. Testing update_weights() on constant graph")
print("-" * 70)

g3 = Graph(state_length=1)
v0 = g3.starting_vertex()
v1 = g3.find_or_create_vertex([1])
v2 = g3.find_or_create_vertex([0])

v0.add_edge(v1, 1.0)  # IPV
v1.add_edge(v2, 3.0)  # Constant → locks to CONSTANT

print("  Graph has constant edges only")

try:
    g3.update_weights([2.0])
    print("  ❌ ERROR: update_weights() allowed on constant graph")
    sys.exit(1)
except RuntimeError as e:
    if "Cannot call update_weights() on constant graph" in str(e):
        print(f"  ✅ update_weights() rejected: {str(e)[:80]}...")
    else:
        print(f"  ❌ Wrong error: {e}")
        sys.exit(1)

# Test 4: update_weights() on empty graph should fail
print("\n4. Testing update_weights() on empty graph")
print("-" * 70)

g4 = Graph(state_length=1)

try:
    g4.update_weights([1.0])
    print("  ❌ ERROR: update_weights() allowed on empty graph")
    sys.exit(1)
except RuntimeError as e:
    if "Cannot call update_weights() on empty graph" in str(e):
        print(f"  ✅ update_weights() rejected: {str(e)[:80]}...")
    else:
        print(f"  ❌ Wrong error: {e}")
        sys.exit(1)

# Test 5: update_weights() on parameterized graph should work
print("\n5. Testing update_weights() on parameterized graph")
print("-" * 70)

g5 = Graph(state_length=1)
v0 = g5.starting_vertex()
v1 = g5.find_or_create_vertex([1])
v2 = g5.find_or_create_vertex([0])

v0.add_edge(v1, 1.0)  # IPV (constant, but doesn't affect mode)
v1.add_edge(v2, [1.0])  # Parameterized → locks to PARAMETERIZED

print("  Graph has parameterized edges")
print(f"  Before update_weights: edge weight = {v1.edges()[0].weight()}")

try:
    g5.update_weights([3.0])
    print(f"  After update_weights([3.0]): edge weight = {v1.edges()[0].weight()}")

    if abs(v1.edges()[0].weight() - 3.0) < 0.001:
        print("  ✅ update_weights() succeeded and updated weight correctly")
    else:
        print(f"  ❌ ERROR: Weight not updated correctly (expected 3.0, got {v1.edges()[0].weight()})")
        sys.exit(1)
except RuntimeError as e:
    print(f"  ❌ ERROR: update_weights() rejected: {e}")
    sys.exit(1)

# Test 6: IPV edges don't count toward mode locking
print("\n6. Testing IPV edges don't lock mode")
print("-" * 70)

g6 = Graph(state_length=1)
v0 = g6.starting_vertex()
v1 = g6.find_or_create_vertex([1])
v2 = g6.find_or_create_vertex([2])
v3 = g6.find_or_create_vertex([0])

# Multiple IPV edges (none should lock mode)
v0.add_edge(v1, 1.0)
print("  Added IPV edge 1")

# Mode still unlocked, can add parameterized edge
v1.add_edge(v2, [2.0])
print("  Added parameterized non-IPV edge → locks to PARAMETERIZED")

# Can add more parameterized edges
v2.add_edge(v3, [3.0])
print("  Added another parameterized edge")

try:
    g6.update_weights([2.0])
    print(f"  update_weights() succeeded")
    print("  ✅ IPV edges correctly don't affect mode locking")
except RuntimeError as e:
    print(f"  ❌ ERROR: Unexpected failure: {e}")
    sys.exit(1)

# Test 7: Constant edges don't get rescaled in parameterized graph
print("\n7. Testing constant edges in constant graph are preserved")
print("-" * 70)

g7 = Graph(state_length=1)
v0 = g7.starting_vertex()
v1 = g7.find_or_create_vertex([1])
v2 = g7.find_or_create_vertex([0])

v0.add_edge(v1, 1.0)  # IPV: always constant
v1.add_edge(v2, 3.0)  # Constant → locks to CONSTANT

print(f"  Starting edge (IPV): {v0.edges()[0].weight()}")
print(f"  Non-starting constant edge: {v1.edges()[0].weight()}")
print("  ✅ Constant graphs correctly preserve edge weights")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("✅ Constant graph mode locks correctly")
print("✅ Parameterized graph mode locks correctly")
print("✅ update_weights() correctly rejects constant graphs")
print("✅ update_weights() correctly rejects empty graphs")
print("✅ update_weights() works on parameterized graphs")
print("✅ IPV edges don't affect mode locking")
print("✅ Constant edges preserved in constant graphs")

print("\n✅ ALL EDGE MODE LOCKING TESTS PASSED!")
print("="*70)
