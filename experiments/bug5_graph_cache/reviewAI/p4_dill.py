"""Review AI probe 4: dill round trip of a used StateIndexer, (A) unmodified
master classes, (B) with a registry wrapper that keeps __module__ =
phasic.state_indexing and stores the snapshot on the class (plan v6.0)."""
import sys, weakref
import dill
import phasic.state_indexing as si
from phasic.state_indexing import StateIndexer, Property

def mk():
    ix = StateIndexer(lineage=[Property("n", max_value=3)])
    ix.index_to_props(0)
    return ix

ix = mk()
print("(A) class module:", ix._result_class.__module__)
try:
    b = dill.loads(dill.dumps(ix))
    print("(A) dill ok; same class:", b._result_class is ix._result_class)
except Exception as e:
    print("(A) dill failed:", type(e).__name__, str(e)[:160])

_real = si.make_dataclass
_GEN = weakref.WeakSet()
def _wrapper(*a, **k):
    k.setdefault("module", "phasic.state_indexing")
    cls = _real(*a, **k)
    cls.__phasic_snapshot__ = (dict(vars(cls)), None)
    _GEN.add(cls)
    return cls
si.make_dataclass = _wrapper
ix = mk()
print("(B) class module:", ix._result_class.__module__)
try:
    b = dill.loads(dill.dumps(ix))
    print("(B) dill ok; same class:", b._result_class is ix._result_class,
          "registered:", b._result_class in _GEN)
except Exception as e:
    print("(B) dill failed:", type(e).__name__, str(e)[:160])
