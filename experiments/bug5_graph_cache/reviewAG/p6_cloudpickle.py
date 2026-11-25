"""Review AG p6: does a StateIndexer survive joblib's (vendored) cloudpickle or
dill, and is its _result_class then the registered class?  (A worker process
would see an unregistered class -> rule 8 refusal.)"""
import pickle
from phasic.state_indexing import StateIndexer, Property

ix = StateIndexer(lin=[Property("d", max_value=3)])
ix.lin.index_to_props(1)
for name in ("joblib.externals.cloudpickle", "dill"):
    try:
        mod = __import__(name, fromlist=["dumps"])
        blob = mod.dumps(ix)
        # loads in the SAME process may resolve back by reference; report both
        ix2 = pickle.loads(blob)
        print(f"{name}: pickles ({len(blob)} B); same _result_class object after "
              f"in-process loads: {ix2._result_class is ix._result_class}")
    except Exception as e:
        print(f"{name}: fails: {type(e).__name__}: {str(e)[:120]}")
