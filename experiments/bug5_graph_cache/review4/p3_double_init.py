"""What happens if the C++ base is initialised twice on one Graph (the shape the
current broad `except Exception` fall-through would produce if a step AFTER
super().__init__(cached_graph) raised)."""
import os
assert "scratchpad" in os.environ["HOME"]
import phasic
from phasic import _Graph
g = phasic.Graph(3)
print("first init ok, vertices", g.vertices_length())
try:
    _Graph.__init__(g, 5)
    print("second init: no error; state_length now", g.state_length(), "vertices", g.vertices_length())
except Exception as e:
    print("second init raised", type(e).__name__, str(e)[:120])
