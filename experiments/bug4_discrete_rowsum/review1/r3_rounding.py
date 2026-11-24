"""Review 1, part 3: how large can the rounding excess of LEGITIMATE rows get?
Row sum measured as the C checks do: sequential double accumulation."""
import sys, random, warnings, math
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import phasic
from phasic import Graph
from fractions import Fraction
def seqsum(ws):
    s = 0.0
    for w in ws: s += w
    return s
def rows(g):
    out = []
    for i in range(1, g.vertices_length()):
        es = g.vertex_at(i).edges()
        if es: out.append(seqsum([e.weight() for e in es]))
    return np.array(out)
def rep(label, r):
    r = np.asarray(r)
    print(f'  {label}: {len(r)} rows, max excess = {r.max() - 1:+.3e}, min (sum-1) among rows within 1e-6 of 1 = {(r[np.abs(r-1)<1e-6] - 1).min() if (np.abs(r-1)<1e-6).any() else float("nan"):+.3e}, rows > 1: {(r > 1).sum()}', flush=True)
rng = np.random.default_rng(7)
print('## (a) pure arithmetic: w_i / sum(w) summed sequentially; n edges, magnitudes spread over `dec` decades; worst of 300 draws')
for n in (10, 100, 1000, 5000, 50000, 600000):
    for dec in (0, 6, 16):
        worst = -1; reps = 300 if n <= 5000 else (30 if n <= 50000 else 5)
        for _ in range(reps):
            w = 10.0 ** rng.uniform(-dec, 0, n) if dec else rng.uniform(0.5, 1.5, n)
            tot = seqsum(w.tolist())                      # was_dph renormalisation: sequential total, then divide
            p = (w / tot).tolist()
            for order in (p, sorted(p), sorted(p, reverse=True)):
                worst = max(worst, seqsum(order) - 1)
        print(f'  n={n:>6} decades={dec:>2}: worst excess {worst:+.3e}   (n*eps = {n*2.2e-16:.1e})', flush=True)
print('## (a2) user-normalised rows: numpy pairwise sum then divide (float64), and float32 probabilities')
for n in (3, 10, 100, 1000, 4000):
    w64 = -1; w32 = -1
    for _ in range(300):
        w = rng.uniform(0, 1, n)
        w64 = max(w64, seqsum((w / w.sum()).tolist()) - 1)
        p32 = (w.astype(np.float32) / w.astype(np.float32).sum(dtype=np.float32)).astype(np.float64)
        w32 = max(w32, seqsum(p32.tolist()) - 1)
    print(f'  n={n:>5}: float64 worst excess {w64:+.3e};  float32-normalised worst excess {w32:+.3e}')
print('  decimal-rounded probabilities (rounded to d digits, as from a table/file): worst excess over 2000 draws')
for d in (3, 4, 6, 8):
    for n in (3, 7, 20):
        worst = -1
        for _ in range(2000):
            w = rng.uniform(0, 1, n); p = np.round(w / w.sum(), d)
            worst = max(worst, seqsum(p.tolist()) - 1)
        print(f'    digits={d} n={n:>2}: {worst:+.3e}')
print('## (b) real graph: one vertex with n parameterized out-edges, was_dph renormalisation in C (update_weights)')
for n in (1000, 4500):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); s.add_edge(a, 1.0)
    co = 10.0 ** rng.uniform(-12, 3, n)
    for i in range(n): a.add_edge(g.find_or_create_vertex([i + 2]), [float(co[i]), float(co[::-1][i])])
    g.is_discrete = True; g.set_was_dph(True)
    worst = -1; lo = 1
    for _ in range(200):
        th = (10.0 ** rng.uniform(-8, 8, 2)).tolist(); g.update_weights(th)
        r = seqsum([e.weight() for e in a.edges()]); worst = max(worst, r - 1); lo = min(lo, r - 1)
    print(f'  n={n}: worst excess {worst:+.3e}, most negative {lo:+.3e}')
    try: print('    pdf(1) =', g.pdf(1))
    except Exception as e: print('    pdf:', str(e)[:80])
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
print('## (c) discretize() at tiny / huge rates')
for name, mk, th in (('coalescent n=22', lambda: coalescent_graph(22), [0.37]), ('two-island n=8', lambda: two_island_graph(8), [123.4, 0.017]), ('two-island n=8', lambda: two_island_graph(8), [1e-9, 1e7])):
    for rate in (1e-12, 1e-3, 1.0, 1e9):
        try:
            g0 = mk(); g0.update_weights(th); d = g0.discretize(rate); d = d[0] if isinstance(d, tuple) else d
            rep(f'{name} theta={th} discretize({rate:g})', rows(d))
            d.update_weights(list(th) + [rate] if d.param_length() == len(th) + 1 else th); rep('     after update_weights', rows(d))
        except Exception as e: print(f'  {name} discretize({rate:g}): {type(e).__name__}: {str(e)[:110]}')
print('## (d) reward_transform_discrete output rows (zero-reward elimination multiplies/sums probabilities)')
for name, mk, th, rate in (('coalescent n=14', lambda: coalescent_graph(14), [0.37], 0.01), ('two-island n=7', lambda: two_island_graph(7), [0.7, 0.3], 0.5), ('two-island n=7', lambda: two_island_graph(7), [1e-6, 1e4], 1e-3)):
    try:
        g0 = mk(); g0.update_weights(th); d = g0.discretize(rate); d, base = (d if isinstance(d, tuple) else (d, None))
        nv = d.vertices_length(); worst = -1; nrows = 0; fails = 0
        for k in range(12):
            rw = rng.choice([0, 0, 0, 1, 2, 3], nv) if k % 2 else (rng.uniform(0, 1, nv) < 0.15).astype(int)
            rw[0] = 0
            try:
                tg = d.reward_transform_discrete(np.asarray(rw, dtype=int)); r = rows(tg); nrows += len(r)
                if len(r): worst = max(worst, r.max() - 1)
            except Exception as e:
                fails += 1; last = str(e)[:90]
        print(f'  {name} theta={th} rate={rate}: {nv} vertices, {nrows} transformed rows over 12 reward vectors, worst excess {worst:+.3e}, failures {fails}' + (f' ({last})' if fails else ''), flush=True)
    except Exception as e: print(f'  {name}: {type(e).__name__}: {str(e)[:140]}')
print('## (e) joint_prob_graph(discrete=True) with reward limits, extreme theta')
for n, th, kw in ((5, [2.0, 0.05], dict(reward_limit=3)), (5, [1e-8, 1e6], dict(reward_limit=3)), (6, [1e5, 1e-9], dict(reward_limit=2)), (6, [5.0, 0.01], dict(reward_limit=3, tot_reward_limit=5))):
    try:
        g, idx = coalescent_indexed(n); J = g.joint_prob_graph(idx, **kw); J.update_weights(th); rep(f'n={n} theta={th} {kw} ({J.vertices_length()} v)', rows(J))
    except Exception as e: print(f'  n={n} theta={th} {kw}: {type(e).__name__}: {str(e)[:120]}')
print('## (f) normalize_discrete (aux edge weight 1 - rate): rate + (1 - rate)')
worst = -1
for _ in range(200000):
    r = seqsum(rng.uniform(0, 0.3, 3).tolist()); worst = max(worst, (r + (1 - r)) - 1)
print(f'  worst excess of rate + (1 - rate): {worst:+.3e}')
