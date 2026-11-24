"""Freshly built discrete graphs BEFORE update_weights: row sums and read-outs (unmodified build)."""
import sys, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed, coalescent_graph
def rows(g):
    out = []
    for i in range(1, g.vertices_length()):
        es = g.vertex_at(i).edges()
        if es: out.append(sum(e.weight() for e in es))
    return np.array(out)
def t(label, f):
    try: v = f(); print(f'    {label}: OK {str(v)[:110]}')
    except BaseException as e: print(f'    {label}: RAISES {type(e).__name__}: {str(e).splitlines()[0][:110]}')
for n, kw in ((2, dict(reward_limit=2)), (3, dict(reward_limit=2)), (3, dict(reward_limit=2, mutation_rate=1e-2)), (5, dict(reward_limit=3))):
    g, idx = coalescent_indexed(n); J = g.joint_prob_graph(idx, **kw); r = rows(J)
    print(f'## fresh joint_prob_graph n={n} {kw}: v={J.vertices_length()} was_dph={J.get_was_dph()} max row={r.max():.6g} rows>1+1e-9: {(r>1+1e-9).sum()}/{len(r)}')
    t('joint_prob_table().prob.sum()', lambda: J.joint_prob_table()['prob'].sum())
    t('expectation()', lambda: J.expectation())
    t('pdf(1..3)', lambda: np.asarray(J.pdf(np.arange(1, 4))))
    t('sample(2)', lambda: J.sample(2))
g0 = coalescent_graph(6); d = g0.discretize(0.1); d = d[0] if isinstance(d, tuple) else d; r = rows(d)
print(f'## fresh coalescent(6).discretize(0.1) (no update_weights): was_dph={d.get_was_dph()} max row={r.max():.6g} rows>1+1e-9: {(r>1+1e-9).sum()}/{len(r)}')
t('pdf(1..3)', lambda: np.asarray(d.pdf(np.arange(1, 4)))); t('expectation()', lambda: d.expectation())
