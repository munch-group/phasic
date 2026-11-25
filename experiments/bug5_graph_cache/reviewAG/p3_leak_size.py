"""Review AG p3: bytes retained per generated class under the v5.8 registry
(weakref + finalize + snapshot dict(vars(cls))), vs no registry; and whether
indexers survive pickle / deepcopy with registered classes."""
import copy, gc, pickle, tracemalloc, weakref, warnings
import phasic.state_indexing as si
from phasic.state_indexing import StateIndexer, Property

REG = {}
orig = si.make_dataclass


def patched(name, fields, **kw):
    cls = orig(name, fields, **kw)
    k = id(cls)
    REG[k] = (weakref.ref(cls), tuple(f if isinstance(f, str) else f[0] for f in fields),
              dict(vars(cls)))
    weakref.finalize(cls, REG.pop, k, None)
    return cls


def run(n):
    for i in range(n):
        a = StateIndexer(lin=[Property("d", max_value=3)])
        r = StateIndexer(lin_d=[Property("lin_d", max_value=3)])
        j = a + r          # what joint_prob_graph does per call
        j.index_to_props(1)
        StateIndexer.from_dict(j.to_dict())   # the (c) rebuild per key computation
        del a, r, j


for label, mk in (("no registry", orig), ("v5.8 registry", patched)):
    si.make_dataclass = mk
    REG.clear(); gc.collect()
    tracemalloc.start()
    run(50); gc.collect()
    b0 = tracemalloc.get_traced_memory()[0]
    run(1000); gc.collect()
    b1 = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    print(f"{label:15s}: retained {(b1-b0)/1e6:.2f} MB over 1000 iterations "
          f"(~{(b1-b0)/1000/1e3:.1f} kB per joint_prob-style iteration); registry={len(REG)}")

si.make_dataclass = patched
a = StateIndexer(lin=[Property("d", max_value=3)])
for name, fn in (("deepcopy", copy.deepcopy), ("pickle", lambda o: pickle.loads(pickle.dumps(o)))):
    try:
        b = fn(a)
        print(name, ": ok; _result_class identical:", b._result_class is a._result_class)
    except Exception as e:
        print(name, ": fails:", type(e).__name__, str(e)[:100])
