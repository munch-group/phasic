"""RA2: per-step cost of the native step (ptd_dph_probability_distribution_step)
split into per-edge and per-vertex parts, on graphs < 5000 vertices, plus the
cost of READING stop_probability() through pybind (what a Python/any per-step
vector read-out pays)."""
import sys, time
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed, coalescent_graph, two_island_graph

def bench(name, g, nsteps=3000):
    V = g.vertices_length(); E = sum(len(g.vertex_at(i).edges()) for i in range(V))
    ctx = g.distribution_context(100000)
    t0 = time.perf_counter()
    for _ in range(nsteps): ctx.step()
    dt = (time.perf_counter() - t0) / nsteps
    print(f'{name:28s} V={V:6d} E={E:7d} E/V={E/V:5.2f}: {dt*1e6:8.2f} us/step = {dt*1e9/E:6.2f} ns/edge = {dt*1e9/(E+V):6.2f} ns/(edge+vertex)', flush=True)
    return V, E, dt

rows = []
g, idx = coalescent_indexed(6)
jc = g.joint_prob_graph(idx, discrete=False, reward_limit=3); jc.update_weights([1.0, 1.0])
rows.append(bench('joint n=6 cap3', jc))
jsp = jc.joint_stop_prob_graph(); jsp.update_weights([1.0, 1.0])
rows.append(bench('JSP n=6 cap3', jsp))
for n in (20, 40):
    g = coalescent_graph(n); g.update_weights([1.0]); rows.append(bench(f'coalescent n={n}', g))
for n in (6, 9):
    g = two_island_graph(n); g.update_weights([1.0, 0.5]); rows.append(bench(f'two_island n={n}', g))
A = np.array([[e, v] for v, e, _ in rows], float); y = np.array([d for *_, d in rows])
(ce, cv), *_ = np.linalg.lstsq(A, y, rcond=None)
print(f'least squares: {ce*1e9:.2f} ns/edge + {cv*1e9:.2f} ns/vertex per step')
for V, E in ((20_000, 100_000), (60_000, 400_000), (600_000, 4_000_000)):
    s = ce * E + cv * V
    print(f'  extrapolated V={V} E={E}: {s*1e3:.2f} ms/step; Lambda=1e3 -> {s*1.3e3:.1f} s, 1e4 -> {s*1.03e4:.0f} s, 1e5 -> {s*1.003e5:.0f} s per forward')
