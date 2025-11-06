"""Test add_aux_vertex() method"""

import phasic
import numpy as np

print("="*70)
print("TEST: add_aux_vertex() METHOD")
print("="*70)

# Test 1: Non-parameterized graph
print("\nTest 1: Non-parameterized graph with constant rate")
g1 = phasic.Graph(2)
v1 = g1.find_or_create_vertex([1, 0])
v2 = g1.find_or_create_vertex([2, 0])
v1.add_edge(v2, 3.0)  # Lock to constant mode

print(f"  graph.parameterized() = {g1.parameterized()}")
aux1 = v1.add_aux_vertex(2.5)
print(f"  aux vertex state = {list(aux1.state())}")
print(f"  aux vertex edges = {len(aux1.edges())}")

# Verify aux vertex has all-zero state
assert list(aux1.state()) == [0, 0], f"Expected [0, 0], got {list(aux1.state())}"

# Verify aux has 1 edge (to parent)
assert len(aux1.edges()) == 1, f"Expected 1 edge from aux, got {len(aux1.edges())}"

# Verify parent has edge to aux
parent_has_edge_to_aux = False
for edge in v1.edges():
    target_state = list(edge.to().state())
    if target_state == [0, 0]:
        parent_has_edge_to_aux = True
        print(f"  Parent edge to aux: weight={edge.weight()}")
        break

assert parent_has_edge_to_aux, "Parent should have edge to aux vertex"
print("  ✅ Test 1 PASSED")

# Test 2: Parameterized graph
print("\nTest 2: Parameterized graph with coefficient array")
g2 = phasic.Graph(2)
v3 = g2.find_or_create_vertex([1, 0])
v4 = g2.find_or_create_vertex([2, 0])
v3.add_edge(v4, [1.0, 0.0])  # Lock to parameterized mode

print(f"  graph.parameterized() = {g2.parameterized()}")
aux2 = v3.add_aux_vertex([2.0, 1.0])
print(f"  aux vertex state = {list(aux2.state())}")
print(f"  aux vertex edges = {len(aux2.edges())}")

# Verify aux vertex has all-zero state
assert list(aux2.state()) == [0, 0], f"Expected [0, 0], got {list(aux2.state())}"

# Verify aux has 1 edge (to parent)
assert len(aux2.edges()) == 1, f"Expected 1 edge from aux, got {len(aux2.edges())}"

# Verify parent has edge to aux
parent_has_edge_to_aux = False
for edge in v3.edges():
    target_state = list(edge.to().state())
    if target_state == [0, 0]:
        parent_has_edge_to_aux = True
        print(f"  Parent edge to aux: weight={edge.weight()}")
        break

assert parent_has_edge_to_aux, "Parent should have edge to aux vertex"
print("  ✅ Test 2 PASSED")

# Test 3: Error - wrong rate type for parameterized graph
print("\nTest 3: Error handling - scalar for parameterized graph")
try:
    aux3 = v3.add_aux_vertex(3.0)  # Should fail
    print("  ❌ FAILED: Should have raised error")
    assert False, "Should have raised error"
except Exception as e:
    print(f"  ✅ Correctly raised error: {type(e).__name__}")
    print(f"     Message: {str(e)}")

# Test 4: Error - wrong rate type for constant graph
print("\nTest 4: Error handling - array for constant graph")
try:
    aux4 = v1.add_aux_vertex([1.0, 2.0])  # Should fail
    print("  ❌ FAILED: Should have raised error")
    assert False, "Should have raised error"
except Exception as e:
    print(f"  ✅ Correctly raised error: {type(e).__name__}")
    print(f"     Message: {str(e)}")

# Test 5: Multiple auxiliary vertices (each is distinct)
print("\nTest 5: Multiple auxiliary vertices")
g3 = phasic.Graph(3)
v5 = g3.find_or_create_vertex([1, 0, 0])
v5.add_edge(g3.find_or_create_vertex([2, 0, 0]), 1.0)  # Lock to constant

aux_a = v5.add_aux_vertex(2.0)
aux_b = v5.add_aux_vertex(3.0)

print(f"  aux_a state = {list(aux_a.state())}")
print(f"  aux_b state = {list(aux_b.state())}")
print(f"  Same vertex? {aux_a == aux_b}")

# Both should have [0, 0, 0] state
assert list(aux_a.state()) == [0, 0, 0]
assert list(aux_b.state()) == [0, 0, 0]

# They should be DIFFERENT vertices (create_vertex creates new each time)
# Each aux vertex has its own unique parent→aux edge with different rate
assert aux_a != aux_b, "Multiple calls should create separate aux vertices"
print("  ✅ Test 5 PASSED - Separate aux vertices created")

# Test 6: Verify edge weights
print("\nTest 6: Verify edge weights")
g4 = phasic.Graph(1)
v6 = g4.find_or_create_vertex([1])
v6.add_edge(g4.find_or_create_vertex([2]), 5.0)
aux_c = v6.add_aux_vertex(7.0)

# Check aux -> parent edge (should be 1.0)
aux_edge = list(aux_c.edges())[0]
print(f"  Edge from aux to parent: weight={aux_edge.weight()}")
assert abs(aux_edge.weight() - 1.0) < 1e-10, f"Expected weight 1.0, got {aux_edge.weight()}"

# Check parent -> aux edge (should be 7.0)
parent_to_aux_weight = None
for edge in v6.edges():
    if list(edge.to().state()) == [0]:
        parent_to_aux_weight = edge.weight()
        break

print(f"  Edge from parent to aux: weight={parent_to_aux_weight}")
assert parent_to_aux_weight is not None, "Parent should have edge to aux"
assert abs(parent_to_aux_weight - 7.0) < 1e-10, f"Expected weight 7.0, got {parent_to_aux_weight}"
print("  ✅ Test 6 PASSED")

print("\n" + "="*70)
print("ALL TESTS PASSED! ✅")
print("add_aux_vertex() method works correctly!")
print("="*70)
