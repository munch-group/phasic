"""Review AG p4: a non-leaking registry variant: WeakSet of generated classes,
snapshot stored ON the class (a GC-collectable cycle), compared excluding its
own key.  Checks: no leak, ordinary indexers pass, swaps refused."""
import gc, tracemalloc, weakref, dataclasses
import phasic.state_indexing as si
from phasic.state_indexing import StateIndexer, Property

_SNAP = "__phasic_generated_snapshot__"
_GEN = weakref.WeakSet()
orig = si.make_dataclass


def patched(name, fields, **kw):
    cls = orig(name, fields, **kw)
    names = tuple(f if isinstance(f, str) else f[0] for f in fields)
    snap = dict(vars(cls))
    type.__setattr__(cls, _SNAP, (names, snap))
    _GEN.add(cls)
    return cls


def ok(cls, fields):
    if type(cls) is not type or cls not in _GEN:
        return "not registered"
    v = vars(cls)
    names, snap = v[_SNAP]
    if names != tuple(fields):
        return "fields"
    if set(v) - {_SNAP} != set(snap):
        return "keyset"
    if any(v[k] is not o for k, o in snap.items()):
        return "entry replaced"
    return "PASS"


si.make_dataclass = patched
ix = StateIndexer(lin=[Property("d", max_value=3)]); ix.lin.index_to_props(1)
print("result_class:", ok(ix._result_class, ["lin"]),
      "| props_dataclass:", ok(ix.lin._props_dataclass, ["d"]))
ix2 = StateIndexer(lin=[Property("d", max_value=3)])
ix2._result_class.__init__ = lambda s, **k: None
print("swapped __init__:", ok(ix2._result_class, ["lin"]))
ix3 = StateIndexer(lin=[Property("d", max_value=3)])
type.__setattr__(ix3._result_class, "__post_init__", lambda s: None)
print("added __post_init__:", ok(ix3._result_class, ["lin"]))
u = dataclasses.make_dataclass("IndexResult", [("lin", object, None)], frozen=True)
print("unregistered:", ok(u, ["lin"]))
del ix, ix2, ix3, u
gc.collect()
tracemalloc.start()
for i in range(50):
    StateIndexer(lin=[Property("d", max_value=3)]).lin.index_to_props(1)
gc.collect(); b0 = tracemalloc.get_traced_memory()[0]
for i in range(1000):
    StateIndexer(lin=[Property("d", max_value=3)]).lin.index_to_props(1)
gc.collect(); b1 = tracemalloc.get_traced_memory()[0]
print(f"retained over 1000 indexers: {(b1-b0)/1e6:.3f} MB; live registered classes: {len(_GEN)}")
