"""Minimum edge weight on legitimate discrete graphs (notebook-style pipelines and all pin cases): any negative / non-finite?"""
import sys, warnings, math, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
import _discretize_pins as pins
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
from phasic import Graph
def scan(g):
    ws = [e.weight() for i in range(g.vertices_length()) for e in g.vertex_at(i).edges()]
    return min(ws), sum(not math.isfinite(w) for w in ws), len(ws)
tot = 0; worst = 0.0; bad = 0
def rep(label, g):
    global tot, worst, bad
    mn, nf, n = scan(g); tot += 1; worst = min(worst, mn); bad += nf
    if mn < 0 or nf: print(f'  NEGATIVE/NONFINITE {label}: min {mn} nonfinite {nf} of {n}')
for name, (build, call, theta) in pins.CASES.items():
    if name in pins.EXPECTED_ERRORS: continue
    d = call(build()); rep(name + ' fresh', d)
    if theta is not None:
        for th in (theta if isinstance(theta[0], list) else [theta]): d.update_weights(th); rep(f'{name} theta={th}', d)
rng = np.random.default_rng(1)
for nm, g0, th in (('coal13', coalescent_graph(13), [7.0]), ('coal22', coalescent_graph(22), [0.37]), ('island8', two_island_graph(8), [0.7, 0.3]), ('island8b', two_island_graph(8), [123.4, 0.017])):
    g0.update_weights(th)
    for rate in (0.001, 0.3, 1, 50.0):
        d = g0.discretize(rate); rep(f'{nm} discretize({rate})', d); d.update_weights(list(th) + [1.0] * (d.param_length() - len(th))); rep(f"{nm} discretize({rate}) updated", d)
        for k in range(3):
            r = rng.integers(0, 3, d.vertices_length()); r[rng.random(len(r)) < 0.5] = 0
            try: rep(f'{nm} discretize({rate}) rtd', d.reward_transform_discrete(r))
            except Exception as e: pass
        try: rep(f'{nm} discretize({rate}) reward_transform(aux)', d.reward_transform(d.rewards))
        except Exception as e: pass
for n, kw in ((3, dict(reward_limit=2, mutation_rate=1e-2)), (4, dict(reward_limit=2)), (5, dict(reward_limit=3)), (4, dict(tot_reward_limit=2, mutation_rate=0.1))):
    g, idx = coalescent_indexed(n); J = g.joint_prob_graph(idx, **kw); rep(f'joint n={n} fresh', J)
    for th in ([1e-9, 1e-2], [1e-4, 1e-2], [2.0, 0.05], [1e4, 1e-2]): J.update_weights(th); rep(f'joint n={n} {th}', J)
print(f'scanned {tot} graphs; most negative weight {worst}; non-finite weights {bad}')
