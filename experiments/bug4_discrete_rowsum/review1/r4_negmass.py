"""Review 1, part 4: do LEGITIMATE graphs (rows already divided by their sum, excess <= a few ulp)
yield negative mass today?  If yes, the plan's per-vertex 1/sum scale cannot be what guarantees
non-negativity: these rows ARE w/sum already."""
import sys, warnings, platform, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from phasic import Graph
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
print('machine', platform.machine(), ' long double mantissa bits:', np.finfo(np.longdouble).nmant)
def seqsum(ws):
    s = 0.0
    for w in ws: s += w
    return s
def scan(label, g, steps):
    rs = [seqsum([e.weight() for e in g.vertex_at(i).edges()]) for i in range(1, g.vertices_length()) if g.vertex_at(i).edges()]
    c = g.distribution_context(); mn_sp = 0.0; mn_pmf = 0.0; nneg = 0; tot = 0.0
    for k in range(steps):
        c.step(); sp = np.asarray(c.stop_probability(), dtype=float); mn_sp = min(mn_sp, sp.min()); nneg += int((sp < 0).sum()); mn_pmf = min(mn_pmf, c.pmf()); tot += c.pmf()
    print(f'  {label}: {g.vertices_length()} v, max row excess {max(rs)-1:+.2e}; over {steps} steps: min stop_probability entry {mn_sp:+.3e} ({nneg} negative entries), min pmf {mn_pmf:+.3e}, cdf {c.cdf():.17g}')
# 1. the simplest legit case: single row, already normalised, one ulp over
for ws in ([0.1, 0.2, 0.7000000000000001], [1/3, 1/3, 1/3 + 1.2e-16], [0.3, 0.3, 0.3, 0.1]):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); s.add_edge(a, 1.0)
    for i, w in enumerate(ws): a.add_edge(g.find_or_create_vertex([i + 2]), w)
    g.is_discrete = True
    r = seqsum(ws); r2 = seqsum([w / r for w in ws])
    print(f'  row {ws}: sum-1 = {r-1:+.2e}; after dividing by sum: sum-1 = {r2-1:+.2e}; scale-then-multiply sum(w*(1/sum))-1 = {seqsum([w*(1/r) for w in ws])-1:+.2e}')
    scan('    ', g, 3)
# 2. three probabilities whose renormalised sum still exceeds 1: search
rng = np.random.default_rng(3); cnt = 0; cnt2 = 0; N = 200000
for _ in range(N):
    w = rng.uniform(0, 1, 3).tolist(); r = seqsum(w)
    if seqsum([x / r for x in w]) > 1: cnt += 1
    if seqsum([x * (1 / r) for x in w]) > 1: cnt2 += 1
print(f'  random 3-edge rows: after w/sum, {cnt}/{N} rows still sum > 1; after w*(1/sum), {cnt2}/{N}')
g, idx = coalescent_indexed(5); J = g.joint_prob_graph(idx, reward_limit=3); J.update_weights([2.0, 0.05]); scan('joint_prob n=5 [2.0,0.05]', J, 400)
g0 = two_island_graph(8); g0.update_weights([0.7, 0.3]); d = g0.discretize(0.5); d = d[0] if isinstance(d, tuple) else d; scan('two-island n=8 discretize(0.5)', d, 400)
g0 = coalescent_graph(22); g0.update_weights([0.37]); d = g0.discretize(1.0); d = d[0] if isinstance(d, tuple) else d; scan('coalescent n=22 discretize(1)', d, 400)
