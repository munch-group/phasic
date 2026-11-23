"""Review 3 of plan v2: numpy replica of the discrete step with the CAPPED variants of plan section 2.2.
 today : current C statements (validated bit-for-bit against the library below)
 capA  : plan 2.2 literal, remaining added right after the vertex's last edge (p zeroed at step start)
 capB  : plan 2.2, remaining added in a final pass (inflow_total + remaining)
 capM  : minimal alternative: today's statements, only `add` capped by a separately tracked remaining
Checks: (1) which bit-exact pin records (pdf(1..5)) change and by how much; (2) on the three legitimate
graphs over 400 jumps: strict pmf >= 0, strict vertex mass (stop_probability) >= 0, monotone cdf, cdf <= 1+1e-12;
(3) geometric chains of tests 3, 5, 10."""
import sys, warnings, json
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
import _discretize_pins as pins
from test_notebook_model_recovery import coalescent_indexed, coalescent_graph, two_island_graph
from phasic import Graph
def extract(g):
    n = g.vertices_length(); E = [[] for _ in range(n)]; absorbing = np.zeros(n, bool)
    for i in range(n):
        es = g.vertex_at(i).edges(); absorbing[i] = len(es) == 0
        for e in es: E[i].append((int(e.to().index()), float(e.weight())))
    return n, E, absorbing
def run(n, E, absorbing, steps, variant, track=False):
    p = np.zeros(n); p[0] = 1.0; out = []; minmass = 0.0; nneg = 0; caps = 0; masses = []
    ab = np.nonzero(absorbing)[0]
    for s in range(steps + 1):
        old = p.copy()
        if variant in ('today', 'capM'):
            p[absorbing] = 0.0
            for f in range(n):
                rem = old[f]
                for t, w in E[f]:
                    add = old[f] * w
                    if variant == 'capM':
                        if add > rem: add = rem; caps += 1
                        rem -= add
                    p[t] += add; p[f] -= add
        else:
            p[:] = 0.0; rems = np.zeros(n)
            for f in range(n):
                rem = old[f]
                for t, w in E[f]:
                    add = old[f] * w
                    if add > rem: add = rem; caps += 1
                    p[t] += add; rem -= add
                if absorbing[f]: rem = 0.0           # absorbed mass is read out and cleared each step
                if variant == 'capA': p[f] += rem
                else: rems[f] = rem
            if variant == 'capB': p += rems
        pmf = 0.0
        for i in ab: pmf += p[i]
        nonabs = p[~absorbing]
        if s > 0:
            minmass = min(minmass, nonabs.min()); nneg += int((nonabs < 0).sum()); out.append(pmf)
        p[absorbing] = 0.0; p[0] = 0.0
        if track and s > 0: masses.append(p.copy())
    return np.array(out), minmass, nneg, caps, masses
VARS = ('today', 'capA', 'capB', 'capM')
print('#### 1. bit-exact pins: pdf(1..5) under the capped step')
for name, (build, call, theta) in pins.CASES.items():
    if name in pins.EXPECTED_ERRORS: continue
    d = call(build())
    if theta is not None: d.update_weights(theta[-1] if isinstance(theta[0], list) else theta)
    lib = np.array([float(d.pdf(k)) for k in range(1, 6)])
    n, E, ab = extract(d); res = {}
    base = run(n, E, ab, 5, 'today')[0]
    for v in VARS[1:]:
        got, _, _, caps, _ = run(n, E, ab, 5, v)
        ch = int(sum(float(a).hex() != float(b).hex() for a, b in zip(got, base)))
        rel = float(np.max(np.abs(got - base) / np.where(base != 0, np.abs(base), 1)))
        res[v] = f'{ch}/5 rel {rel:.1e} caps {caps}'
    print(f'  {name:26s} replica==lib: {bool(np.array_equal(base, lib))} | ' + ' | '.join(f'{v}: {res[v]}' for v in VARS[1:]))
print('#### 2. legitimate graphs, 400 jumps')
def rep(label, g, steps=400, lib=True):
    n, E, ab = extract(g)
    if lib:
        k = np.arange(1, steps + 1); lp = np.asarray(g.pdf(k)); lc = np.asarray(g.cdf(k))
        sp = np.array([np.asarray(g.stop_probability(int(j))) for j in (1, 2, 5, 10, 50, 100, 200, 400)])
        print(f'  {label} LIBRARY today: neg pmf {int((lp<0).sum())} min {lp.min():+.2e}; cdf monotone {bool(np.all(np.diff(lc)>=0))} max cdf-1 {lc.max()-1:+.2e}; stop_probability at 8 jump counts: neg entries {int((sp<0).sum())} min {sp.min():+.2e}')
    for v in VARS:
        pm, minmass, nneg, caps, _ = run(n, E, ab, steps, v); cdf = np.cumsum(pm)
        extra = f' replica==lib {bool(np.array_equal(pm, lp))}' if (lib and v == 'today') else ''
        print(f'    [{v:5s}] neg pmf {int((pm<0).sum())} min pmf {pm.min():+.2e} | neg vertex masses {nneg} min {minmass:+.2e} | cdf monotone {bool(np.all(np.diff(cdf)>=0))} max cdf-1 {cdf.max()-1:+.2e} final cdf-1 {cdf[-1]-1:+.2e} | caps fired {caps}{extra}')
gr, idx = coalescent_indexed(5); J = gr.joint_prob_graph(idx, reward_limit=3); J.update_weights([2.0, 0.05])
rows = [sum(e.weight() for e in J.vertex_at(i).edges()) for i in range(1, J.vertices_length())]
print('  precondition joint n=5: rows above 1 =', sum(r > 1 for r in rows))
rep('joint n=5 rl=3 [2,0.05]', J)
g0 = coalescent_graph(10); g0.update_weights([7.0]); d = g0.discretize(0.001); rep('coalescent n=10 discretize(0.001)', d)
g0 = two_island_graph(8); g0.update_weights([0.7, 0.3]); d = g0.discretize(0.001); rep('two-island n=8 discretize(0.001)', d)
g0 = coalescent_graph(10); g0.update_weights([7.0]); d = g0.discretize(0.5); rep('coalescent n=10 discretize(0.5)', d, 2000, lib=False)
print('#### 3. geometric chains (tests 3, 5, 10)')
def geom(p, ipv=1.0):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, ipv); a.add_edge(b, p); g.is_discrete = True; return g
for p, ipv in ((1 + 2.220446049250313e-16, 1), (1 + 1e-15, 1), (1 + 1e-12, 1), (1 + 9e-10, 1), (0.5, 1 + 1e-12), (0.5, 1 + 9e-10)):
    n, E, ab = extract(geom(p, ipv))
    for v in ('today', 'capA', 'capM'):
        pm, minmass, nneg, caps, _ = run(n, E, ab, 200, v)
        print(f'  p-1={p-1:+.1e} ipv-1={ipv-1:+.1e} [{v:5s}] pdf(1)={pm[0]!r} min pmf {pm.min():+.1e} max cdf-1 {np.cumsum(pm).max()-1:+.2e} neg masses {nneg}')
k = np.arange(1, 201)
for p in (0.1, 0.3, 0.5, 0.9, 1.0):
    n, E, ab = extract(geom(p)); tp = p * (1 - p) ** (k - 1); tc = 1 - (1 - p) ** k
    print(f'  guard 10 p={p}: ' + ' | '.join(f'{v}: pmf abs {np.abs(run(n,E,ab,200,v)[0]-tp).max():.1e} cdf abs {np.abs(np.cumsum(run(n,E,ab,200,v)[0])-tc).max():.1e}' for v in VARS))
