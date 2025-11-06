#!/usr/bin/env python3
"""Test the exact notebook sequence that crashes"""

# CELL 125: Create graph
import phasic
import numpy as np

nr_rabbits, flood_left, flood_right = 2, 2, 4

@phasic.callback([nr_rabbits, 0])
def rabbits(state, rabbit_jump=None, flood_left=None, flood_right=None):
    left, right = state
    transitions = []
    if left:
        transitions.append([[left - 1, right + 1], rabbit_jump])
        transitions.append([[0, right], flood_left])
    if right:
        transitions.append([[left + 1, right - 1], rabbit_jump])
        transitions.append([[left, 0], flood_right])
    return transitions

print("CELL 125: Creating graph...")
graph = phasic.Graph(rabbits, rabbit_jump=1, flood_left=2, flood_right=4)
states = graph.states()
print(f"  Graph has {len(states)} vertices")
print("  ✅ Cell 125 done")

# CELL 126: Discretize
print("\nCELL 126: Discretizing...")
carrot_graph, carrot_vertices = graph.discretize(0.7)
carrot_states = carrot_graph.states()
print(f"  Discretized graph has {len(carrot_states)} vertices")
print(f"  Type: {type(carrot_graph)}")
print("  ✅ Cell 126 done")

# CELL 127: Copy and modify
print("\nCELL 127: Copy and modify...")
mdph_carrot_graph = graph.copy()

vlength = mdph_carrot_graph.vertices_length()
carrot_vertices_left = np.repeat(False, vlength*2)
carrot_vertices_right = np.repeat(False, vlength*2)

print(f"  Original vertices: {vlength}")

for i in range(vlength):
    vertex = mdph_carrot_graph.vertex_at(i)
    rabbits_state = vertex.state()

    if rabbits_state[0] > 0:
        obtained_carrot_vertex = mdph_carrot_graph.create_vertex([0])
        obtained_carrot_vertex.add_edge(vertex, 1)
        vertex.add_edge(obtained_carrot_vertex, rabbits_state[0] * 0.1)
        carrot_vertices_left[obtained_carrot_vertex.index()] = True

    if rabbits_state[1] > 0:
        obtained_carrot_vertex = mdph_carrot_graph.create_vertex([0])
        obtained_carrot_vertex.add_edge(vertex, 1)
        vertex.add_edge(obtained_carrot_vertex, rabbits_state[1] * 0.1)
        carrot_vertices_right[obtained_carrot_vertex.index()] = True

print(f"  Final vertices: {mdph_carrot_graph.vertices_length()}")
print("  ✅ Cell 127 done")

# CELL 129: Normalize and compute covariance
print("\nCELL 129: Normalize and compute covariance...")
carrot_vertices_left = carrot_vertices_left[0:mdph_carrot_graph.vertices_length()]
carrot_vertices_right = carrot_vertices_right[0:mdph_carrot_graph.vertices_length()]
mdph_carrot_graph.normalize()
rewards = np.column_stack((carrot_vertices_left, carrot_vertices_right)).astype(int)
cov = mdph_carrot_graph.covariance_discrete(rewards[:,0], rewards[:,1])
print(f"  Covariance: {cov}")
print("  ✅ Cell 129 done")

print("\n" + "="*70)
print("✅ ALL CELLS COMPLETED SUCCESSFULLY!")
print("="*70)
