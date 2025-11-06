"""Check edge coefficients via C++ API"""

import phasic
import ctypes

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

# Access C graph directly
c_graph = graph.c_graph()
print(f"C graph pointer: {c_graph}")
print(f"Graph param_length: {c_graph.param_length}")
print(f"Graph parameterized: {c_graph.parameterized}")

# Check starting vertex
start = graph.starting_vertex()
print(f"\nStarting vertex edges: {len(start.edges())}")

edge = list(start.edges())[0]
target = edge.to()
print(f"Edge target state: {list(target.state())}")
print(f"Edge weight: {edge.weight()}")

# Try to access coefficients via pybind11
# Check what methods are available
print(f"\nEdge methods: {[m for m in dir(edge) if not m.startswith('_')]}")
