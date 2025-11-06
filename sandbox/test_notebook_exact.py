"""Test exact code from notebook"""

import phasic

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

print(f"✅ Graph created: {graph.vertices_length()} vertices")

# Test clone (the original issue)
cloned = graph.clone()
print(f"✅ Clone successful: {cloned.vertices_length()} vertices")

print("✅ SUCCESS: Exact notebook code works!")
