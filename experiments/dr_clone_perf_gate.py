"""Timing gate for the C ptd_clone_graph fix (2026-10-13): clone must be
linear in vertices + edges. Before the fix it scanned the vertex array per
edge: 2.99 s at 37k vertices, 109.9 s at 204k (discretize-cpp-derisk-findings.md).
Run it through the memory-capped runner (child RLIMIT + parent RSS
watchdog; RLIMIT alone is not enforced on macOS):
    python .git/claude-scratch/runcap.py --mb 6144 -- python experiments/dr_clone_perf_gate.py"""
import sys, time
sys.path.insert(0, 'experiments')
from dr_discretize_cpp_sizes import coal

BEFORE = {40: 2.99, 50: 109.9}
per = {}
ok = True
for n, before in BEFORE.items():
    g = coal(n)
    V = g.vertices_length(); E = sum(v.edges_length() for v in g.vertices())
    t = time.perf_counter(); g.clone(); dt = time.perf_counter() - t
    per[n] = dt / (V + E)
    good = dt < before / 10
    ok &= good
    print(f"coal n={n}: V={V} E={E} clone {dt:.3f}s (before {before}s, {before/dt:.0f}x) "
          f"{1e6*per[n]:.3f} us/(v+e) {'OK' if good else 'TOO SLOW'}", flush=True)
ratio = per[50] / per[40]
ok &= ratio < 2.0
print(f"per-element ratio n=50/n=40: {ratio:.2f} {'OK (linear)' if ratio < 2.0 else 'NOT LINEAR'}")
print("ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
