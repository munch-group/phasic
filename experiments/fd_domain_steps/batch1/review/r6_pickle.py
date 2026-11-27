"""Review probe r6: does the new per-model FDStats (holds a threading.Lock)
make a model unpicklable where it was picklable before? (cloudpickle / pickle)"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import phasic
from phasic import Graph, _fd_steps
import _fd_golden_cells as C
try:
    import cloudpickle
except ImportError:
    cloudpickle = None
print("cloudpickle available:", cloudpickle is not None)
for name in ("pfg_linear", "pam_linear", "dc_noexp"):
    if name == "pfg_linear":
        m = Graph.pmf_from_graph(C.beside(), theta_dim=2)
    elif name == "pam_linear":
        m = Graph.pmf_and_moments_from_graph(C.beside(), nr_moments=2, theta_dim=2)
    else:
        jpg, oi = C.joint_graphs()
        m, _, _, _ = jpg._daisy_chain_svgd_model(observed_indices=list(oi), epoch_starts=[0.0, 0.5],
                                                 user_prior=None, user_fixed=None)
    for lib in ([pickle] + ([cloudpickle] if cloudpickle else [])):
        try:
            lib.dumps(m)
            print(name, lib.__name__, "OK")
        except Exception as e:
            print(name, lib.__name__, "FAIL", type(e).__name__, str(e)[:120])
    try:
        pickle.dumps(_fd_steps.FDStats("x"))
        print("FDStats pickle OK")
    except Exception as e:
        print("FDStats pickle FAIL", type(e).__name__, str(e)[:100])
