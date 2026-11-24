"""Review 3: how far above 1 do rows get after reward transforms of legit discrete graphs (notebook-style pipelines)?"""
import sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, two_island_graph, coalescent_indexed
from phasic import Graph
def rows(g):
    out = []
    for i in range(1, g.vertices_length()):
        es = g.vertex_at(i).edges()
        if es: out.append(sum(e.weight() for e in es))
    return np.array(out)
def rep(label, g):
    r = rows(g); print(f'  {label}: v={g.vertices_length()} is_discrete={g.is_discrete} was_dph={g.get_was_dph()} rows={len(r)} max-1={r.max()-1:+.3e} min={r.min():.4f} rows>1={(r>1).sum()} rows>1+1e-12={(r>1+1e-12).sum()}')
rng = np.random.default_rng(3)
print('## discrete.ipynb pipeline: coalescent n=4, discretize(callback 0.3*lineages), reward transforms')
from phasic import with_ipv
def coal(n):
    @with_ipv([n]+[0]*(n-1))
    def coalescent(state):
        tr = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2: continue
                if not same and (state[i] < 1 or state[j] < 1): continue
                new = state.copy(); new[i] -= 1; new[j] -= 1; new[i+j+1] += 1
                tr.append((new, state[i]*(state[j]-same)/(1+same)))
        return tr
    return Graph(coalescent)
for n in (4, 8, 12):
    g = coal(n)
    mg = g.discretize(rate=lambda s: float(sum(s)) * 0.3); rep(f'n={n} discretize(callback)', mg)
    try:
        t1 = mg.reward_transform(mg.rewards); rep(f'n={n} .reward_transform(aux rewards)', t1)
    except Exception as e: print('   rt fail', str(e)[:100])
    rew = g.states().T[:-1]
    for i in (0, len(rew)//2):
        rt = g.reward_transform(rew[i]); drt = rt.discretize(0.3); rep(f'n={n} rt({i}).discretize(0.3)', drt)
        m = drt.reward_transform(drt.rewards); rep(f'n={n} rt({i}).discretize.reward_transform(aux)', m)
        try:
            p = np.asarray(m.pdf(np.arange(0, 400))); print(f'      pmf min={p.min():+.3e} neg={(p<0).sum()} sum-1={p.sum()-1:+.3e}')
        except Exception as e: print('      pdf fail', str(e)[:100])
print('## reward_transform_discrete with random integer rewards (zeros eliminate vertices)')
for name, g0, th in (('coalescent n=13', coalescent_graph(13), [7.0]), ('coalescent n=22', coalescent_graph(22), [0.37]), ('two-island n=8', two_island_graph(8), [0.7, 0.3])):
    g0.update_weights(th); d = g0.discretize(0.01); d = d[0] if isinstance(d, tuple) else d
    rep(name + ' discretize(0.01)', d)
    worst = 0
    for k in range(6):
        r = rng.integers(0, 3, d.vertices_length()); r[rng.random(len(r)) < 0.5] = 0
        try:
            t = d.reward_transform_discrete(r); rr = rows(t); worst = max(worst, rr.max() - 1)
        except Exception as e: print('   fail', str(e)[:120]); break
    print(f'     worst max-1 over 6 random integer reward transforms: {worst:+.3e}')
print('## native DPH with long rows: 1/k probabilities (user-built), row sum rounding')
for k in (3, 10, 49, 1000, 100000):
    s = 0.0
    for _ in range(k): s += 1.0 / k
    print(f'   k={k}: sequential sum of k copies of 1/k - 1 = {s-1:+.3e}')
w = rng.random(200000); w = w / w.sum(); s = 0.0
for x in w: s += x
print(f'   200000 random normalised weights: sequential sum - 1 = {s-1:+.3e}')
