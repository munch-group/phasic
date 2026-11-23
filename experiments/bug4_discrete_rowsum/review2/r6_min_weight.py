"""Review 2 of plan v3: would the new 'negative or non-finite weight' rejection
hit legitimate discrete graphs? Smallest edge weight on the pin cases and the
legitimate graphs used before (after update_weights)."""
import sys, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
import _discretize_pins as pins
def mw(g):
    ws = [float(e.weight()) for i in range(g.vertices_length()) for e in g.vertex_at(i).edges()]
    return min(ws), sum(w < 0 for w in ws), sum(not math.isfinite(w) for w in ws), sum(w == 0 for w in ws), len(ws)
worst = (1.0, None)
for name in sorted(pins.CASES):
    if name in pins.EXPECTED_ERRORS: continue
    build, call, theta = pins.CASES[name]
    d = call(build())
    if theta is not None:
        for t in (theta if isinstance(theta[0], list) else [theta]): d.update_weights(t)
    m = mw(d)
    if m[1] or m[2] or m[0] < worst[0]: worst = (m[0], name)
    if m[1] or m[2]: print(f'pin {name}: min {m[0]!r} negative {m[1]} non-finite {m[2]}')
print('pin cases: smallest weight', worst)
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
g, idx = coalescent_indexed(5); J = g.joint_prob_graph(idx, reward_limit=3)
for th in ([2.0, 0.05], [1e-4, 1e-2], [50.0, 1e-6]):
    J.update_weights(th); print('joint n=5', th, 'min, #neg, #nonfinite, #zero, #edges =', mw(J))
for nm, g0, th in (('coalescent n=10', coalescent_graph(10), [7.0]), ('two-island n=8', two_island_graph(8), [0.7, 0.3]), ('two-island n=8', two_island_graph(8), [123.4, 0.017])):
    g0.update_weights(th)
    for rate in (0.001, 0.2):
        try:
            d = g0.discretize(rate); d = d[0] if isinstance(d, tuple) else d
            print(nm, th, 'discretize', rate, mw(d))
        except Exception as e: print(nm, th, rate, type(e).__name__, str(e)[:90])
