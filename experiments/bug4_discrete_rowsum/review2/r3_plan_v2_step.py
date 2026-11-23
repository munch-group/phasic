"""Review 2 of plan v2: float64 replica (arm64: long double == double) of
 A  = today's step (phasic.c:12943-12990),
 P1 = plan v2 section 2.2 as written: probability_at zeroed, inflow accumulated,
      each vertex's capped remainder added when its edge group ends,
 P2 = same but remainders added after all increments,
 I  = in-place capped step: today's statements (to += add; from -= add) with only
      'if (add > remaining) add = remaining; remaining -= add' inserted.
Questions: (1) does A reproduce the library bit for bit; (2) negatives;
(3) how many values change in bits vs A -- in particular the pinned pdf(1..5)
of tests/pytest/test_gate_discretize_pins.py; (4) self-loops."""
import sys, warnings, time, random
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
def extract(g):
    n = g.vertices_length(); out = [[] for _ in range(n)]
    for i in range(n):
        for e in g.vertex_at(i).edges(): out[i].append((e.to().index(), float(e.weight())))
    return n, out
def run(n, out, variant, steps):
    has = [len(o) > 0 for o in out]
    p = [0.0] * n; p[0] = 1.0
    pmfs = []; negmass = 0; minmass = 0.0; caps = 0; stops = []
    for it in range(steps + 1):
        old = p[:]
        if variant in ('A', 'I'):
            for v in range(n):
                if not has[v]: p[v] = 0.0
        else:
            p = [0.0] * n
        rem_all = [0.0] * n
        for v in range(n):
            if not has[v]: continue
            rem = old[v]
            for t, w in out[v]:
                add = old[v] * w
                if variant != 'A':
                    if add > rem: add = rem; caps += 1
                    rem -= add
                p[t] += add
                if variant in ('A', 'I'): p[v] -= add
            if variant == 'P1': p[v] += rem
            rem_all[v] = rem
        if variant == 'P2':
            for v in range(n):
                if has[v]: p[v] += rem_all[v]
        pmf = 0.0
        for v in range(n):
            if not has[v]: pmf += p[v]; p[v] = 0.0
            elif p[v] < 0: negmass += 1; minmass = min(minmass, p[v])
        p[0] = 0.0
        if it > 0: pmfs.append(pmf); stops.append(p[:])
    return pmfs, negmass, minmass, caps, stops
def cmp(label, g, steps, lib=True):
    n, out = extract(g)
    selfloops = sum(1 for v in range(n) for t, _ in out[v] if t == v)
    ref, negA, minA, _, _ = run(n, out, 'A', steps)
    line = f'{label}: {n}v selfloops={selfloops}'
    if lib:
        libp = [float(g.pdf(k)) for k in range(1, steps + 1)]
        line += f' | replica A == library bitwise: {sum(a.hex()==b.hex() for a,b in zip(ref,libp))}/{steps}'
    line += f' | A: neg pmf {sum(x<0 for x in ref)}, neg mass {negA}'
    for var in ('P1', 'P2', 'I'):
        pm, neg, mn, caps, _ = run(n, out, var, steps)
        nd = sum(a.hex() != b.hex() for a, b in zip(pm, ref))
        rel = max((abs(a-b)/abs(b) for a, b in zip(pm, ref) if abs(b) > 1e-14), default=0.0)
        line += f' | {var}: neg pmf {sum(x<0 for x in pm)}, neg mass {neg}, caps fired {caps}, pmf bits changed {nd}/{steps}, max rel(>1e-14) {rel:.1e}, cdf-1 {sum(pm)-1:+.1e}'
    print(line, flush=True)
print('## pinned discretize cases: pdf(1..5) is what the pin file stores')
import _discretize_pins as pins
for name in sorted(pins.CASES):
    if name in pins.EXPECTED_ERRORS: continue
    build, call, theta = pins.CASES[name]
    try:
        d = call(build())
        if theta is not None:
            for t in (theta if isinstance(theta[0], list) else [theta]): d.update_weights(t)
        cmp(f'pin {name}', d, 5)
    except Exception as e:
        print(f'pin {name}: {type(e).__name__}: {str(e)[:120]}')
print('## legitimate graphs, 400 steps')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
g, idx = coalescent_indexed(5); J = g.joint_prob_graph(idx, reward_limit=3); J.update_weights([2.0, 0.05]); cmp('joint n=5', J, 400)
g0 = coalescent_graph(10); g0.update_weights([7.0]); d = g0.discretize(0.001); d = d[0] if isinstance(d, tuple) else d; cmp('coalescent n=10 discretize', d, 400)
g0 = two_island_graph(8); g0.update_weights([0.7, 0.3]); d = g0.discretize(0.001); d = d[0] if isinstance(d, tuple) else d; cmp('two-island n=8 discretize', d, 400)
print('## native graph, rows well below 1 (no cap can fire), with explicit self-loops')
import phasic
from phasic import Graph
rng = random.Random(5)
g = Graph(1); vs = [g.starting_vertex()] + [g.find_or_create_vertex([i]) for i in range(1, 40)]
vs[0].add_edge(vs[1], 0.6); vs[0].add_edge(vs[2], 0.4)
for i in range(1, 39):
    vs[i].add_edge(vs[i], 0.2)                       # self-loop
    for t in rng.sample(range(1, 40), 4):
        if t != i: vs[i].add_edge(vs[t], 0.15 * rng.random())
g.is_discrete = True
cmp('native 40v self-loops', g, 200)
print('## sampler cost today (per-visited-vertex check) vs one whole-graph row-sum pass')
g0 = coalescent_graph(22); g0.update_weights([0.37]); d = g0.discretize(0.001); d = d[0] if isinstance(d, tuple) else d
random.seed(1); np.random.seed(1)
t0 = time.perf_counter(); s = d.sample(2000); t1 = time.perf_counter()
n, out = extract(d); E = sum(len(o) for o in out)
print(f'  coalescent n=22 discretize: {n} vertices {E} edges; sample(2000) {1e6*(t1-t0)/2000:.1f} us per draw, mean jumps {float(np.mean(s)):.0f}')
