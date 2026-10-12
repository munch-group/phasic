"""De-risk (discretize-cpp-ssot-plan.md, phase -1 (a), (b) proxy).

Times, on one graph at a time:
  build      - Graph(callback): pybind calls the Python callback once per
               vertex from C++ (proxy for a C++ discretize calling a Python
               rate callback per vertex)
  clone      - C graph clone (lower bound for a C++ rebuild)
  widen      - Python _rebuild_with_wider_layout(extra_coeff_slots=1)
  normalize  - normalize() on a clone
  disc_fixed - discretize(0.1)
  disc_cb    - discretize(lambda s: 0.1 * s.sum())
Usage: python dr_discretize_cpp_costs.py coal:40 island:20 ...
"""
import sys, time, resource
import numpy as np
sys.path.insert(0, 'tests/pytest')
sys.path.insert(0, 'experiments')
from dr_discretize_cpp_sizes import coal
from test_notebook_model_recovery import two_island_graph


def timed(f):
    t = time.perf_counter(); out = f(); return out, time.perf_counter() - t


for spec in sys.argv[1:]:
    fam, n = spec.split(':'); n = int(n)
    g, t_build = timed(lambda: coal(n) if fam == 'coal' else two_island_graph(n))
    V = g.vertices_length()
    E = sum(v.edges_length() for v in g.vertices())
    _, t_clone = timed(lambda: g.clone())
    _, t_widen = timed(lambda: g._rebuild_with_wider_layout(extra_coeff_slots=1))
    c = g.clone()
    _, t_norm = timed(lambda: c.normalize())
    d, t_fixed = timed(lambda: g.discretize(0.1))
    _, t_cb = timed(lambda: g.discretize(lambda s: 0.1 * s.sum()))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"{fam} n={n}: V={V} E={E} | build {t_build:.2f}s ({1e6*t_build/V:.1f} us/v) | "
          f"clone {t_clone:.3f}s | widen {t_widen:.2f}s ({1e6*t_widen/(V+E):.1f} us/(v+e)) | "
          f"normalize {t_norm:.3f}s | disc_fixed {t_fixed:.2f}s | disc_cb {t_cb:.2f}s "
          f"(cb overhead {1e6*(t_cb-t_fixed)/V:.1f} us/v) | aux loop ~{t_fixed - t_widen - t_norm:.2f}s | "
          f"maxrss {rss:.0f} MB", flush=True)
    del g, c, d
