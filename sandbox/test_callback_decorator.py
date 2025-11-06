"""Test the @phasic.callback decorator pattern from notebook"""

import phasic
import os

print("Testing @phasic.callback decorator pattern...")

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

print("Creating graph from callback...")
try:
    graph = phasic.Graph(rabbits)
    print(f"✅ Graph created successfully: {graph.vertices_length()} vertices")

    print("\nTesting clone...")
    cloned = graph.clone()
    print(f"✅ Clone successful: {cloned.vertices_length()} vertices")

    print("\n✅ SUCCESS: @phasic.callback pattern works!")
    os._exit(0)

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
    os._exit(1)
