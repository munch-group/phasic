"""Performance gate for the C++ widening primitive (discretize-cpp-ssot-plan.md
phase 1, added by the de-risk findings). Run memory-capped, one at a time:

    python experiments/dr_widen_layout_perf_gate.py

Asserts (1) widening is at least 5x faster than the Python helper measured
in discretize-cpp-derisk-findings.md on the same graphs, and (2) cost per
vertex+edge is roughly constant from the coalescent n=40 to n=50 (linear,
not the O(V*E) of the C clone)."""
import sys, time
sys.path.insert(0, 'tests/pytest')
sys.path.insert(0, 'experiments')
from dr_discretize_cpp_sizes import coal
from test_notebook_model_recovery import two_island_graph

# Python helper, measured before phase 1 (findings table (a)), seconds.
BASELINE = {("coal", 40): 1.82, ("island", 22): 10.80, ("coal", 50): 13.27}

per_elem = {}
ok = True
for (fam, n), base in BASELINE.items():
    g = coal(n) if fam == "coal" else two_island_graph(n)
    V = g.vertices_length()
    E = sum(v.edges_length() for v in g.vertices())
    t = time.perf_counter()
    g._rebuild_with_wider_layout(extra_coeff_slots=1)
    dt = time.perf_counter() - t
    per_elem[(fam, n)] = dt / (V + E)
    faster = base / dt
    good = faster >= 5
    ok &= good
    print(f"{fam} n={n}: V={V} E={E} widen {dt:.3f}s vs Python {base:.2f}s "
          f"({faster:.0f}x) {1e6 * dt / (V + E):.2f} us/(v+e) {'OK' if good else 'TOO SLOW'}",
          flush=True)
ratio = per_elem[("coal", 50)] / per_elem[("coal", 40)]
lin = ratio < 2.0
ok &= lin
print(f"per-element cost ratio coal n=50 / n=40: {ratio:.2f} {'OK (linear)' if lin else 'NOT LINEAR'}")
print("ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
