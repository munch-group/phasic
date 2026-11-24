"""Review 2 of plan v2, follow-up to r3: (a) what the changed pinned values look
like (old -> new) for the cases the plan says must NOT change; (b) a native graph
with rows well below 1: does the plan's rewrite change bits when no cap fires;
(c) are self-loop edges constructible; (d) sampler cost per draw today."""
import sys, warnings, time, random, io, contextlib
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/experiments/bug4_discrete_rowsum/review2')
src = open('/Users/kmt/phasic-bug4/experiments/bug4_discrete_rowsum/review2/r3_plan_v2_step.py').read().split("print('## pinned discretize cases")[0]
exec(src)
import _discretize_pins as pins
import json
REC = json.load(open(pins.RECORDS))
print('## (a) pinned pdf(1..5): record vs replica of today (A) vs plan step (P1) vs in-place cap (I)')
changedP = changedI = total = 0
for name in sorted(pins.CASES):
    if name in pins.EXPECTED_ERRORS: continue
    build, call, theta = pins.CASES[name]
    d = call(build())
    if theta is not None:
        for t in (theta if isinstance(theta[0], list) else [theta]): d.update_weights(t)
    n, out = extract(d)
    rows = []
    for v in range(1, n):
        s = 0.0
        for _, w in out[v]: s += w
        if out[v]: rows.append(s)
    a = run(n, out, 'A', 5)[0]; p = run(n, out, 'P1', 5)[0]; i = run(n, out, 'I', 5)[0]
    rec = REC['cases'][name]['pdf']
    recok = isinstance(rec, list) and all(float(x).hex() == y.hex() for x, y in zip(rec, a))
    total += 1; cp = any(x.hex() != y.hex() for x, y in zip(a, p)); ci = any(x.hex() != y.hex() for x, y in zip(a, i))
    changedP += cp; changedI += ci
    print(f'{name}: rows above 1: {sum(r > 1 for r in rows)}/{len(rows)}; record == replica A: {recok}; P1 changes pin: {cp}; I changes pin: {ci}')
    for k in range(5):
        if a[k].hex() != p[k].hex() or a[k].hex() != i[k].hex():
            print(f'     pdf({k+1}): today {a[k]!r}  plan-step {p[k]!r}  in-place {i[k]!r}')
print(f'## pinned cases whose pdf record changes: plan step {changedP}/{total}, in-place cap {changedI}/{total}')
print('## (b,c) native graph')
from phasic import Graph
g = Graph(1); s = g.starting_vertex(); a_ = g.find_or_create_vertex([1])
try: a_.add_edge(a_, 0.2); print('  self-loop edge accepted')
except Exception as e: print('  self-loop edge refused at construction:', type(e).__name__, str(e)[:80])
try: s.add_edge(a_, -0.5); print('  negative edge weight accepted at construction')
except Exception as e: print('  negative edge weight refused:', type(e).__name__, str(e)[:80])
rng = random.Random(5)
g = Graph(1); vs = [g.starting_vertex()] + [g.find_or_create_vertex([i]) for i in range(1, 41)]
vs[0].add_edge(vs[1], 0.6); vs[0].add_edge(vs[2], 0.4)
for i in range(1, 40):
    for t in rng.sample(range(1, 41), 4):
        if t != i: vs[i].add_edge(vs[t], 0.15 * rng.random())
g.is_discrete = True
cmp('native 41v rows <= 0.6', g, 200)
print('## (d) sampler cost today')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph
g0 = coalescent_graph(22); g0.update_weights([0.37]); d = g0.discretize(0.001); d = d[0] if isinstance(d, tuple) else d
random.seed(1); np.random.seed(1)
t0 = time.perf_counter(); smp = d.sample(2000); t1 = time.perf_counter()
n, out = extract(d); E = sum(len(o) for o in out)
t2 = time.perf_counter()
for _ in range(200): float(d.pdf(1)); d.update_weights([0.37])
t3 = time.perf_counter()
print(f'  coalescent n=22 discretize: {n} vertices {E} edges; sample(2000): {1e6*(t1-t0)/2000:.1f} us per draw, mean jumps {float(np.mean(smp)):.0f}')
