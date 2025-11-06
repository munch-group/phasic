"""Test that aux→parent edges survive update_weights()"""

import phasic
import numpy as np

print("="*70)
print("TEST: aux→parent edges survive update_weights()")
print("="*70)

# Create parameterized graph with aux vertex
print("\nSetting up parameterized graph with aux vertex...")
g = phasic.Graph(2)
v1 = g.find_or_create_vertex([1, 0])
v2 = g.find_or_create_vertex([2, 0])
v1.add_edge(v2, [1.0, 0.0])  # Lock to parameterized mode

# Add aux vertex
aux = v1.add_aux_vertex([2.0, 1.0])
print(f"Created aux vertex with state: {list(aux.state())}")

# Check aux→parent edge weight before update
aux_edges = list(aux.edges())
print(f"Aux vertex has {len(aux_edges)} edge(s)")
assert len(aux_edges) == 1, "Aux should have exactly 1 edge"

aux_edge = aux_edges[0]
weight_before = aux_edge.weight()
print(f"aux→parent edge weight BEFORE update: {weight_before}")
assert abs(weight_before - 1.0) < 1e-10, f"Expected 1.0, got {weight_before}"

# Update weights with new parameters
print("\nCalling update_weights([2.0, 3.0])...")
g.update_weights([2.0, 3.0])

# Check aux→parent edge weight after update
aux_edges_after = list(aux.edges())
aux_edge_after = aux_edges_after[0]
weight_after = aux_edge_after.weight()
print(f"aux→parent edge weight AFTER update:  {weight_after}")

# THE CRITICAL TEST: aux→parent edge should still be 1.0!
assert abs(weight_after - 1.0) < 1e-10, f"FAILED: aux→parent edge changed from 1.0 to {weight_after}!"

print("\n✅ SUCCESS: aux→parent edge survived update_weights()!")
print("   Weight remained 1.0 (not affected by parameter update)")

# Also check that the parent→aux edge WAS updated
print("\nVerifying parent→aux edge WAS updated...")
parent_to_aux_found = False
for edge in v1.edges():
    target_state = list(edge.to().state())
    if target_state == [0, 0]:
        parent_to_aux_weight = edge.weight()
        print(f"parent→aux edge weight: {parent_to_aux_weight}")
        # Should be 2.0*2.0 + 1.0*3.0 = 4.0 + 3.0 = 7.0
        expected = 2.0*2.0 + 1.0*3.0
        assert abs(parent_to_aux_weight - expected) < 1e-10, \
            f"Expected {expected}, got {parent_to_aux_weight}"
        parent_to_aux_found = True
        break

assert parent_to_aux_found, "Should have found parent→aux edge"
print(f"✅ parent→aux edge correctly updated to {expected}")

print("\n" + "="*70)
print("ALL TESTS PASSED! ✅")
print("="*70)
