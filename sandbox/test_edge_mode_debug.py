"""
Quick debug test for edge mode locking
"""

from phasic import Graph
import sys

g = Graph(state_length=1)
v0 = g.starting_vertex()
v1 = g.find_or_create_vertex([1])
v2 = g.find_or_create_vertex([2])
v3 = g.find_or_create_vertex([0])

# IPV edge
print("Adding IPV edge v0 -> v1 with weight 1.0...")
v0.add_edge(v1, 1.0)
print("  Success")

# First non-IPV constant edge (locks mode)
print("Adding constant edge v1 -> v2 with weight 3.0...")
v1.add_edge(v2, 3.0)
print("  Success - mode should now be locked to CONSTANT")

# Try to add parameterized edge (should fail)
print("Adding parameterized edge v2 -> v3 with coefficients [2.0]...")
try:
    v2.add_edge(v3, [2.0])
    print("  ERROR: No exception was raised!")
    sys.exit(1)
except RuntimeError as e:
    print(f"  RuntimeError caught: {e}")
except Exception as e:
    print(f"  Different exception caught: {type(e).__name__}: {e}")
