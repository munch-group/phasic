"""Debug callback pattern to see graph structure"""

import phasic
import numpy as np

print("Creating graph with callback...")

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

graph = phasic.Graph(rabbits)
print(f"Graph created: {graph.vertices_length()} vertices")

# Check starting vertex
start = graph.starting_vertex()
print(f"\nStarting vertex state: {start.state()}")
print(f"Starting vertex has {len(start.edges())} edges")

# List starting vertex edges
print("\nStarting vertex edges:")
for i, edge in enumerate(start.edges()):
    target = edge.to()
    print(f"  Edge {i}: to vertex with state {target.state()}")
    print(f"    Weight: {edge.weight()}")

# Check if starting vertex is in vertices array
print(f"\nChecking if starting vertex is in vertices array...")
start_in_array = False
for i, v in enumerate(graph.vertices()):
    if np.array_equal(v.state(), start.state()):
        print(f"  Found vertex with matching state at index {i}: {list(v.state())}")
        if v == start:
            print(f"    This IS the starting vertex!")
            start_in_array = True
        else:
            print(f"    This is NOT the starting vertex (different object)")

if not start_in_array:
    print(f"  Starting vertex is NOT in vertices array")

# List all vertices
print(f"\nAll vertices:")
for i, v in enumerate(graph.vertices()):
    print(f"  Vertex {i}: state={list(v.state())}, edges={len(v.edges())}")
