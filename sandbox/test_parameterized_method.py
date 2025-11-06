"""Test graph.parameterized() method"""

import phasic

# Test 1: Non-parameterized graph
print("Test 1: Non-parameterized graph")
g1 = phasic.Graph(1)
v0 = g1.starting_vertex()
v1 = g1.find_or_create_vertex([1])
v0.add_edge(v1, 2.0)
print(f"  graph.parameterized() = {g1.parameterized()}")
assert g1.parameterized() == False, "Expected False for non-parameterized graph"
print("  ✅ PASSED")

# Test 2: Parameterized graph
print("\nTest 2: Parameterized graph")
g2 = phasic.Graph(1)
v0 = g2.starting_vertex()
v1 = g2.find_or_create_vertex([1])
v2 = g2.find_or_create_vertex([2])
# Add a non-IPV parameterized edge to lock the mode
v1.add_edge_parameterized(v2, 0.0, [2.0, 1.0])
print(f"  graph.parameterized() = {g2.parameterized()}")
assert g2.parameterized() == True, "Expected True for parameterized graph"
print("  ✅ PASSED")

# Test 3: Callback graph
print("\nTest 3: Callback graph")
@phasic.callback([2, 0])
def rabbits(state):
    left, right = state
    transitions = []
    if left:
        transitions.append([[left - 1, right + 1], [1, 0, 0, 0]])
        transitions.append([[0, right], [0, 1, 0, 0]])
    if right:
        transitions.append([[left + 1, right - 1], [1, 0, 0, 0]])
        transitions.append([[left, 0], [0, 0, 1, 0]])
    return transitions

g3 = phasic.Graph(rabbits)
print(f"  graph.parameterized() = {g3.parameterized()}")
# Callback graphs are parameterized
assert g3.parameterized() == True, "Expected True for callback graph"
print("  ✅ PASSED")

print("\n✅ ALL TESTS PASSED: graph.parameterized() works correctly!")
