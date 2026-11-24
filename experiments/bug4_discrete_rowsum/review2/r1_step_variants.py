"""Review 2: emulate ptd_dph_probability_distribution_step in float64 (arm64:
long double == double) and compare treatments of rows whose sum is in (1, 1+tol].
A = unchanged; B1 = (old*w)*s ; B2 = old*(w*s) with s = 1/sum for sum>1;
C = cap each outflow at the mass remaining at the source (running residual)."""
import sys, warnings, random
import numpy as np
warnings.filterwarnings('ignore')
print('long double bits:', np.finfo(np.longdouble).nmant)
def emulate(n, edges, variant, steps):
    # edges: list of (from, to, w) grouped by from, vertex 0 = start
    out = [[] for _ in range(n)]
    for f, t, w in edges: out[f].append((t, w))
    has = [len(o) > 0 for o in out]
    scale = [1.0] * n
    for v in range(n):
        s = 0.0
        for _, w in out[v]: s += w
        if s > 1.0: scale[v] = 1.0 / s
    p = [0.0] * n; p[0] = 1.0
    pmfs = []; cdf = 0.0; minmass = 0.0; nneg = 0
    for it in range(steps + 1):
        old = p[:]
        for v in range(n):
            if not has[v]: p[v] = 0.0
        for v in range(n):
            resid = old[v]
            for t, w in out[v]:
                if variant == 'A' or v == 0: add = old[v] * w
                elif variant == 'B1': add = (old[v] * w) * scale[v]
                elif variant == 'B2': add = old[v] * (w * scale[v])
                elif variant == 'C':
                    add = old[v] * w
                    if add > resid: add = resid
                    resid -= add
                p[t] += add; p[v] -= add
        pmf = 0.0
        for v in range(n):
            if not has[v]: pmf += p[v]; p[v] = 0.0
            else:
                if p[v] < 0: nneg += 1; minmass = min(minmass, p[v])
        p[0] = 0.0
        cdf += pmf
        if it > 0: pmfs.append(pmf)
    return np.array(pmfs), cdf, minmass, nneg, sum(p)
def report(label, n, edges, steps):
    ref = None
    for var in ('A', 'B1', 'B2', 'C'):
        pm, cdf, mm, nneg, rest = emulate(n, edges, var, steps)
        if ref is None: ref = pm
        d = np.max(np.abs(pm - ref) / np.maximum(np.abs(ref), 1e-300))
        print(f'  {label} {var}: min pmf {pm.min():+.3e}  #neg pmf {(pm<0).sum()}  min transient mass {mm:+.3e} (#neg {nneg})  cdf-1 {cdf-1:+.3e}  cdf+rest-1 {cdf+rest-1:+.3e}  max rel diff vs A {d:.2e}')
# (i) synthetic: rows normalised by division (what was_dph update_weights does), many edges
rng = random.Random(3)
for nv, deg in ((40, 3), (40, 12), (200, 30)):
    edges = [(0, 1, 1.0)]
    over = 0
    for v in range(1, nv):
        ts = rng.sample(range(1, nv + 1), min(deg, nv))
        ws = [rng.random() ** 3 for _ in ts]
        s = 0.0
        for w in ws: s += w
        ws = [w / s for w in ws]
        s2 = 0.0
        for w in ws: s2 += w
        over += s2 > 1
        edges += [(v, t, w) for t, w in zip(ts, ws)]
    print(f'## random nv={nv} deg={deg}: rows above 1: {over}')
    report('rand', nv + 1, edges, 400)
# (ii) geometric with p = 1 + d, and a two-edge full row
for d in (2.220446049250313e-16, 1e-15, 1e-12, 1e-9):
    print(f'## geometric p = 1 + {d:.3g}')
    report('geom', 3, [(0, 1, 1.0), (1, 2, 1.0 + d)], 6)
# (iii) real graph: joint_prob_graph discrete n=5 theta=[2, .05]
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
g, idx = coalescent_indexed(5)
J = g.joint_prob_graph(idx, reward_limit=3); J.update_weights([2.0, 0.05])
n = J.vertices_length(); edges = []
for i in range(n):
    for e in J.vertex_at(i).edges(): edges.append((i, e.to().index(), e.weight()))
print(f'## real joint_prob_graph n=5: {n} vertices {len(edges)} edges, is_discrete={J.is_discrete}')
report('jpg', n, edges, 300)
# the library itself on this graph today
try:
    pm = np.asarray(J.pdf(np.arange(1, 301)))
    print(f'  library pmf: min {pm.min():+.3e}  #neg {(pm<0).sum()}  sum-1 {pm.sum()-1:+.3e}')
    sp = np.concatenate([np.asarray(J.stop_probability(k)) for k in (1, 2, 5, 20, 100)])
    print(f'  library stop_probability: min {sp.min():+.3e}  #neg {(sp<0).sum()}')
    av = np.asarray(J.accumulated_visits(100)); print(f'  library accumulated_visits(100): min {av.min():+.3e} #neg {(av<0).sum()}')
except Exception as e: print('  library:', type(e).__name__, str(e)[:200])
