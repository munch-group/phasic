"""Review AI probe 3: simulate the plan's make_dataclass registry (v6.0 §2.2.2
rule 8 (c)) in-process, without editing source, then check
 (1) an ordinary used indexer passes the snapshot identity check;
 (2) copying an index_to_props RESULT (ordinary user action) breaks it;
 (3) dill / joblib-cloudpickle can still dump an indexer whose classes carry
     the snapshot attribute (test 18 feasibility)."""
import copy, weakref, sys
import phasic.state_indexing as si
from phasic.state_indexing import StateIndexer, Property

_real = si.make_dataclass
_GEN = weakref.WeakSet()
def _wrapper(*a, **k):
    cls = _real(*a, **k)
    snap = dict(vars(cls))
    names = tuple(f[0] if isinstance(f, tuple) else f for f in a[1])
    cls.__phasic_snapshot__ = (snap, names)
    _GEN.add(cls)
    return cls
si.make_dataclass = _wrapper

def check(cls):
    if cls not in _GEN:
        return "unregistered"
    snap, _ = cls.__phasic_snapshot__
    cur = {k: v for k, v in vars(cls).items() if k != "__phasic_snapshot__"}
    if set(cur) != set(snap):
        return f"key set differs: +{sorted(set(cur)-set(snap))} -{sorted(set(snap)-set(cur))}"
    bad = [k for k in snap if cur[k] is not snap[k]]
    return "ok" if not bad else f"entries replaced: {bad}"

def classes(ix):
    out = [("IndexResult", ix._result_class)]
    for n, ps in ix._property_sets.items():
        c = getattr(ps, "_props_dataclass", None)
        if c is not None:
            out.append((n, c))
    return out

ix = StateIndexer(lineage=[Property("n", max_value=3), Property("pop", min_value=1, max_value=2)], slots=["epoch"])
r = ix.index_to_props(1)
print("(1) fresh+used:", [(n, check(c)) for n, c in classes(ix)])
labels = {0: r}
copy.deepcopy(labels)            # e.g. user snapshots a state->label map
print("(2) after deepcopy of an index_to_props result:", [(n, check(c)) for n, c in classes(ix)])

ix2 = StateIndexer(lineage=[Property("n", max_value=3)])
ix2.index_to_props(0)
try:
    import dill
    s = dill.dumps(ix2)
    back = dill.loads(s)
    print("(3) dill dumps/loads ok; loaded class registered:", back._result_class in _GEN,
          "same class:", back._result_class is ix2._result_class)
except Exception as e:
    print("(3) dill failed:", type(e).__name__, str(e)[:200])
try:
    from joblib.externals import cloudpickle
    s = cloudpickle.dumps(ix2)
    print("(3) cloudpickle dumps ok, bytes", len(s))
except Exception as e:
    print("(3) cloudpickle failed:", type(e).__name__, str(e)[:200])
