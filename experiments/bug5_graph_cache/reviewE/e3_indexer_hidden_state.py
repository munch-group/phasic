"""reviewE e3: under v3.0 rule 8 a StateIndexer is keyed by to_dict() only, and a Property by
(name, min_value, max_value). Show inputs with identical key material that build different
graphs. Builds use graph_cache=False (master), so nothing is written."""
import os, numpy as np
import phasic
from phasic import Graph, StateIndexer, Property
assert os.environ["HOME"].startswith("/private/tmp")

def cb(state, indexer=None, **kw):
    n = state[0]
    if n <= 1: return []
    r = getattr(indexer, "rate", 1.0)          # user attribute stored on the indexer
    return [([n-1], n*(n-1)/2*r)]

a = StateIndexer(lineage=[Property('n', max_value=6)])
b = StateIndexer(lineage=[Property('n', max_value=6)])
b.rate = 5.0                                    # allowed by StateIndexer.__setattr__ (state_indexing.py:1005-1011)
print("to_dict equal:", a.to_dict() == b.to_dict(), "| a == b:", a == b, "| hash equal:", hash(a) == hash(b))
ga = Graph(cb, ipv=[6], indexer=a, graph_cache=False)
gb = Graph(cb, ipv=[6], indexer=b, graph_cache=False)
print("expectation a / b:", ga.expectation(), gb.expectation())

# PropertySet attribute (also mutable, also outside to_dict)
c = StateIndexer(lineage=[Property('n', max_value=6)])
c.lineage.rate = 5.0
def cb2(state, indexer=None, **kw):
    n = state[0]
    if n <= 1: return []
    return [([n-1], n*(n-1)/2*getattr(indexer.lineage, "rate", 1.0))]
print("PropertySet attr: to_dict equal:", a.to_dict() == c.to_dict(),
      Graph(cb2, ipv=[6], indexer=a, graph_cache=False).expectation(),
      Graph(cb2, ipv=[6], indexer=c, graph_cache=False).expectation())

# Property: offset is a field the key drops
p1 = Property('n', max_value=6, min_value=1, offset=0)
p2 = Property('n', max_value=6, min_value=1, offset=3)   # min_value != 0 -> offset not folded, but kept
print("Property key fields equal:", (p1.name, p1.min_value, p1.max_value) == (p2.name, p2.min_value, p2.max_value),
      "| offset", p1.offset, p2.offset, "| p1 == p2:", p1 == p2)

# Property subclass inside an EXACT-class StateIndexer: to_dict identical
class P2(Property):
    def encode_value(self, v): return v
d = StateIndexer(lineage=[P2('n', max_value=6, min_value=1)])
e = StateIndexer(lineage=[Property('n', max_value=6, min_value=1)])
print("Property subclass inside indexer: to_dict equal:", d.to_dict() == e.to_dict(),
      "| type(d) is StateIndexer:", type(d) is StateIndexer,
      "| props_to_index differ:", d.lineage.props_to_index({'n': 3}) if hasattr(d.lineage,'props_to_index') else 'n/a',
      e.lineage.props_to_index({'n': 3}) if hasattr(e.lineage,'props_to_index') else 'n/a')
