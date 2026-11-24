"""Review 3, part 2: other read-outs, pmf_from_graph, stale context, test 4 (joint table), joint row sums vs theta, fix replica."""
import random, warnings, sys
import numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
import jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed, JOINT_TRUE, MUTATION_RATE
def geom(p):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
def pgeom():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); return g
def t(label, f):
    try: v = f(); print(f'    {label}: OK {v!r}'[:200]); return v
    except BaseException as e: print(f'    {label}: RAISES {type(e).__name__}: {str(e).splitlines()[0][:90]}')
def rows(g):
    out = []
    for i in range(1, g.vertices_length()):
        es = g.vertex_at(i).edges()
        if es: out.append(sum(e.weight() for e in es))
    return np.array(out)
print('#### B. pmf_from_graph(discrete=True) (no use_ffi kwarg exists)')
T = jnp.asarray([1., 2., 3.])
for ex in (0.0, 1e-12, 1e-5, 1e-4, 2e-4):
    th = jnp.asarray([1.0 + ex]); print(f'## theta = 1 + {ex:g}')
    m = Graph.pmf_from_graph(pgeom(), discrete=True)
    t('pmf', lambda: np.asarray(m(th, T)))
    t('vmap pmf', lambda: np.asarray(jax.vmap(lambda q: m(q, T))(jnp.stack([th, th]))))
    t('grad sum log|pmf[0]|', lambda: np.asarray(jax.grad(lambda q: m(q, T)[0])(th)))
print('#### B2. FD probe crossing: theta exactly 1.0 (valid model), gradient')
m2 = Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True)
for th0 in (1.0, 1 - 5e-8, 1 - 2e-7, 0.999):
    t(f'theta={th0!r} grad pmf[0]', lambda: np.asarray(jax.grad(lambda q: Graph.pmf_from_graph(pgeom(), discrete=True)(q, T)[0])(jnp.asarray([th0]))))
    t(f'theta={th0!r} grad E[N^2] (moments)', lambda: np.asarray(jax.grad(lambda q: m2(q, T)[1][1])(jnp.asarray([th0]))))
print('#### C. stale context: valid weights -> pdf -> update to invalid -> pdf')
g = pgeom(); g.update_weights([0.5]); g.is_discrete = True
t('pdf at 0.5', lambda: np.asarray(g.pdf(np.arange(1,4))))
g.update_weights([1.5]); t('pdf at 1.5 after update', lambda: np.asarray(g.pdf(np.arange(1,4))))
g.update_weights([1+1e-5]); t('pdf at 1+1e-5 after update', lambda: np.asarray(g.pdf(np.arange(1,4))))
print('#### D. joint_prob_graph discrete: flags and row sums vs theta')
for n, rl in ((3, 2), (4, 2), (5, 3)):
    gr, idx = coalescent_indexed(n)
    try: J = gr.joint_prob_graph(idx, reward_limit=rl, mutation_rate=MUTATION_RATE)
    except Exception as e: print('build fail', n, rl, e); continue
    print(f'## n={n} rl={rl} v={J.vertices_length()} is_discrete={J.is_discrete} was_dph={J.get_was_dph()}')
    for th in ([1e-9, 1e-2], [1e-4, 1e-2], [2e-4,1e-2],[5e-5,1e-2],[1.0, 1e-2], [50.0, 1e-2], [1e4, 1e-2], [1e-4+1e-7, 1e-2], [2.0, 0.05]):
        try:
            J.update_weights(th); r = rows(J)
            tab = J.joint_prob_table(); p = tab['prob'].to_numpy()
            print(f'   theta={th}: max row-1={r.max()-1:+.2e} min row-1={r.min()-1:+.2e} rows>1: {(r>1).sum()}/{len(r)}; table min={p.min():.3e} neg={(p<0).sum()} sum-1={p.sum()-1:+.3e}')
        except Exception as e: print(f'   theta={th}: {type(e).__name__}: {str(e)[:100]}')
