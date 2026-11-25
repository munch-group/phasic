"""reviewAL p4: test-26 premise: a callback with mixed coefficient lengths builds but
serialize() fails (so the 'cannot be cached' note path is reachable)."""
import os
assert os.environ["HOME"].startswith("/private/tmp/")
from phasic import Graph

def cb(state, **kw):
    n = state[0]
    if n <= 1:
        return []
    if n % 2:
        return [[[n - 1], [1.0]]]
    return [[[n - 1], [1.0, 2.0]]]

for ipv in ([5], [4]):
    try:
        g = Graph(cb, ipv=ipv, graph_cache=False)
        print("ipv", ipv, "built", g.vertices_length(), "param_length", g.param_length())
        try:
            g.serialize()
            print("  serialize OK")
        except Exception as e:
            print("  serialize raised", type(e).__name__, str(e)[:120])
    except Exception as e:
        print("ipv", ipv, "build raised", type(e).__name__, str(e)[:120])
