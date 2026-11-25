"""Review AI probe 6: (i) dill on a never-used indexer (only _result_class)
under the plan's snapshot-on-class registry; (ii) the suggested fix: hold the
snapshot in a small object whose __reduce__ yields an empty stub, so dill /
cloudpickle copy a harmless stub (the rebuilt class is still unregistered)."""
import weakref
import dill
import phasic.state_indexing as si
from phasic.state_indexing import StateIndexer, Property

class _Snap:
    __slots__ = ("entries", "fields", "__weakref__")
    def __init__(self, entries, fields):
        self.entries, self.fields = entries, fields
    def __reduce__(self):
        return (_Snap, (None, None))

_real = si.make_dataclass
_GEN = weakref.WeakSet()
MODE = {"fix": False}
def _wrapper(*a, **k):
    k.setdefault("module", "phasic.state_indexing")
    cls = _real(*a, **k)
    snap = dict(vars(cls))
    cls.__phasic_snapshot__ = _Snap(snap, None) if MODE["fix"] else (snap, None)
    _GEN.add(cls)
    return cls
si.make_dataclass = _wrapper

for fix in (False, True):
    MODE["fix"] = fix
    for used in (False, True):
        ix = StateIndexer(lineage=[Property("n", max_value=3)], slots=["e"])
        if used:
            ix.index_to_props(0); ix.lineage.index_to_props(0)
        try:
            b = dill.loads(dill.dumps(ix))
            r = f"ok (loaded class registered: {b._result_class in _GEN})"
        except Exception as e:
            r = f"{type(e).__name__}: {str(e)[:100]}"
        print(f"fix={fix} used={used}: dill -> {r}")
