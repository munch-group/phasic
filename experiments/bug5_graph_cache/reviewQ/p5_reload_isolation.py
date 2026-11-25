"""Review Q: test 26 plans an in-process importlib.reload(phasic). Does that
leak into later tests in the same pytest process (class identity)?"""
import importlib
import phasic
G0 = phasic.Graph            # what an earlier-collected test module holds
g = G0(1)
importlib.reload(phasic)
print("phasic.Graph is G0 after reload:", phasic.Graph is G0)
c = g.clone()
print("clone() of a pre-reload graph is instance of G0:", isinstance(c, G0))
h = phasic.Graph(1)
print("new Graph(1) is instance of G0:", isinstance(h, G0))
