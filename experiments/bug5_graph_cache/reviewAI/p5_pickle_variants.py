"""Review AI probe 5: isolate WHY dill fails under the plan's registry, and
check joblib-cloudpickle dump -> load in a fresh subprocess.
Variant C: WeakSet only (no snapshot attribute). Variant D: snapshot attr."""
import sys, weakref, subprocess, os
import dill
from joblib.externals import cloudpickle
import phasic.state_indexing as si
from phasic.state_indexing import StateIndexer, Property

_real = si.make_dataclass
_GEN = weakref.WeakSet()
MODE = {"snap": False}
def _wrapper(*a, **k):
    k.setdefault("module", "phasic.state_indexing")
    cls = _real(*a, **k)
    if MODE["snap"]:
        cls.__phasic_snapshot__ = (dict(vars(cls)), None)
    _GEN.add(cls)
    return cls
si.make_dataclass = _wrapper

def mk():
    ix = StateIndexer(lineage=[Property("n", max_value=3)])
    ix.index_to_props(0)
    return ix

for snap in (False, True):
    MODE["snap"] = snap
    ix = mk()
    try:
        dill.loads(dill.dumps(ix)); r = "ok"
    except Exception as e:
        r = f"{type(e).__name__}: {str(e)[:120]}"
    print(f"snapshot_attr={snap}: dill -> {r}")
    try:
        blob = cloudpickle.dumps(ix)
        path = os.path.join(os.environ["SCRATCH"], f"cp_{int(snap)}.pkl")
        open(path, "wb").write(blob)
        code = ("import pickle,sys; ix=pickle.load(open(sys.argv[1],'rb')); "
                "print('loaded; n states', ix.state_length, 'cls', ix._result_class.__name__, "
                "'has snapshot attr', hasattr(ix._result_class,'__phasic_snapshot__'))")
        out = subprocess.run([sys.executable, "-c", code, path], capture_output=True, text=True)
        print(f"snapshot_attr={snap}: cloudpickle load in subprocess -> rc={out.returncode}",
              (out.stdout.strip() or out.stderr.strip().splitlines()[-1])[:200])
        os.remove(path)
    except Exception as e:
        print(f"snapshot_attr={snap}: cloudpickle dump failed {type(e).__name__}: {str(e)[:120]}")
