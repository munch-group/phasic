"""Review 3: numpy replica of ptd_dph_probability_distribution_step (float64 == long double on arm64).
(1) validate replica bit-for-bit against the library; (2) apply the plan's per-vertex scale and see
whether bit-exact pins (tests/pytest/data/discretize_pins.json: pdf(1..5)) would change;
(3) what assertion tolerances hold after the fix (min pmf, min residual mass, total)."""
import sys, warnings, json
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
import _discretize_pins as pins
from test_notebook_model_recovery import coalescent_indexed, coalescent_graph, two_island_graph
from phasic import Graph
TOL = 1e-9
def extract(g):
    n = g.vertices_length(); E = []; absorbing = np.zeros(n, bool); rows = np.zeros(n)
    for i in range(n):
        es = g.vertex_at(i).edges()
        absorbing[i] = len(es) == 0
        for e in es:
            E.append((i, int(e.to().index()), float(e.weight()))); rows[i] += float(e.weight())   # same summation order as C
    return n, E, absorbing, rows
def run(n, E, absorbing, rows, steps, variant):
    scale = np.ones(n)
    over = (rows > 1.0) & (np.arange(n) > 0)
    if variant != 'none': scale[over] = 1.0 / rows[over]
    W = [(f, t, (w * scale[f]) if variant == 'B' else (w / rows[f] if (variant == 'C' and over[f]) else w)) for f, t, w in E]
    p = np.zeros(n); p[0] = 1.0; out = []; minres = 0.0
    for s in range(steps + 1):
        old = p.copy(); p[absorbing] = 0.0
        for f, t, w in W:
            add = old[f] * w
            if variant == 'A': add = add * scale[f]
            p[t] += add; p[f] -= add
        pmf = p[absorbing].sum() if absorbing.sum() > 1 else float(p[absorbing][0]) if absorbing.any() else 0.0
        # C sums pmf sequentially over absorbing vertices in index order
        pmf = 0.0
        for i in np.nonzero(absorbing)[0]: pmf += p[i]
        nonabs = p[~absorbing]; minres = min(minres, nonabs[1:].min() if len(nonabs) > 1 else 0.0)
        p[absorbing] = 0.0; p[0] = 0.0
        if s > 0: out.append(pmf)
    return np.array(out), minres
print('#### 1. bit-exact pins: does a rounding-level rescale change pdf(1..5)?')
REC = json.load(open(pins.RECORDS))
tot = chg = 0
for name, (build, call, theta) in pins.CASES.items():
    if name in pins.EXPECTED_ERRORS: continue
    try:
        d = call(build())
        if theta is not None:
            th = theta[-1] if isinstance(theta[0], list) else theta
            d.update_weights(th)
        lib = [float(d.pdf(k)) for k in range(1, 6)]
        n, E, ab, rows = extract(d)
        base, _ = run(n, E, ab, rows, 5, 'none')
        ok = [a.hex() for a in map(float, base)] == [a.hex() for a in lib]
        res = {}
        for v in 'ABC':
            got, _ = run(n, E, ab, rows, 5, v); res[v] = int(sum(float(a).hex() != float(b).hex() for a, b in zip(got, base)))
        nover = int(((rows > 1) & (np.arange(n) > 0)).sum())
        tot += 1; chg += any(res.values())
        print(f'  {name:28s} v={n:4d} rows>1: {nover:3d} max-1={rows[1:].max()-1:+.1e} replica==lib bits: {ok}; pdf(1..5) values whose bits change  A:{res["A"]} B:{res["B"]} C:{res["C"]}')
    except Exception as e:
        print(f'  {name}: {type(e).__name__}: {str(e)[:100]}')
print(f'  => {chg} of {tot} pinned cases change bits under at least one variant')
print('#### 2. tolerances after fix on legit graphs (replica), 400 steps')
def rep(label, g, steps=400):
    n, E, ab, rows = extract(g)
    nover = int(((rows > 1) & (np.arange(n) > 0)).sum())
    lib = np.array([float(g.pdf(k)) for k in range(1, 6)])
    for v in ('none', 'A', 'B', 'C'):
        pm, minres = run(n, E, ab, rows, steps, v)
        cdf = np.cumsum(pm)
        print(f'  {label} [{v:4s}] rows>1={nover}/{n} replica==lib(5)={np.array_equal(pm[:5], lib) if v=="none" else "-"} min pmf={pm.min():+.3e} neg pmf={int((pm<0).sum())} min non-absorbing mass={minres:+.3e} sum-1={pm.sum()-1:+.3e} max cdf-1={cdf.max()-1:+.3e} monotone={bool(np.all(np.diff(cdf)>=0))}')
gr, idx = coalescent_indexed(5); J = gr.joint_prob_graph(idx, reward_limit=3); J.update_weights([2.0, 0.05]); rep('joint n=5 rl=3 [2,0.05] (plan test 4)', J)
tab = J.joint_prob_table()['prob'].to_numpy(); print(f'    joint_prob_table today: min={tab.min():.3e} neg={(tab<0).sum()} sum-1={tab.sum()-1:+.3e} n={len(tab)}')
g0 = coalescent_graph(10); g0.update_weights([7.0]); d = g0.discretize(0.001); d = d[0] if isinstance(d, tuple) else d; rep('coalescent n=10 discretize(0.001)', d)
g0 = two_island_graph(8); g0.update_weights([0.7, 0.3]); d = g0.discretize(0.001); d = d[0] if isinstance(d, tuple) else d; rep('two-island n=8 discretize(0.001)', d)
def geom(p):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
for ex in (2.220446049250313e-16, 1e-15, 1e-12, 9e-10):
    rep(f'geometric p=1+{ex:.1e}', geom(1 + ex), 50)
print('#### 3. random multi-edge rows summing to 1+ulp..1+1e-12: can scaled leaving mass still exceed mass present?')
rng = np.random.default_rng(0); worst = {v: 0.0 for v in 'ABC'}; cnt = {v: 0 for v in 'ABC'}; N = 20000
for _ in range(N):
    k = rng.integers(2, 12); w = rng.random(k); w = w / w.sum(); w[0] += rng.choice([2.3e-16, 1e-15, 1e-12])
    s = 0.0
    for x in w: s += x
    if s <= 1: continue
    m = rng.random()
    for v in 'ABC':
        left = m
        for x in w:
            add = m * x * (1 / s) if v == 'A' else (m * (x * (1 / s)) if v == 'B' else m * (x / s))
            left -= add
        if left < 0: cnt[v] += 1; worst[v] = min(worst[v], left / m)
print('  negative residual counts / worst relative residual:', cnt, worst)
