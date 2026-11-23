"""Review 1, part 2: parameterized paths, stale contexts, was_dph, self-loops, rate graphs flagged discrete."""
import sys, random, warnings
import numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
import jax, jax.numpy as jnp
def t(label, f):
    try:
        v = f()
        if isinstance(v, (list, tuple)) and len(v) and hasattr(v[0], 'shape'): v = [np.asarray(x, dtype=float).round(8).tolist() for x in v]
        elif hasattr(v, 'shape') or isinstance(v, (list, tuple)): v = np.asarray(v, dtype=float).round(8).tolist()
        print(f'    {label}: {v}')
    except BaseException as e: print(f'    {label}: {type(e).__name__}: {str(e).splitlines()[0][:110]}')
def pgeom(set_discrete=True):          # parameterized native geometric: weight = theta[0]
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); g.is_discrete = set_discrete; return g
jumps = jnp.asarray([1., 2., 3.])
print('## A. parameterized likelihood builders on a native DPH (weight = theta)')
for th in (0.5, 1 + 1e-12, 1 + 1e-5, 1.5):
    print(f'# theta = {th!r}')
    th_ = jnp.asarray([th])
    t('pmf_from_graph(discrete=True)', lambda: Graph.pmf_from_graph(pgeom(), discrete=True)(th_, jumps))
    for ffi in (False, True):
        t(f'pmf_and_moments_from_graph(discrete=True, use_ffi={ffi})', lambda: Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi=ffi)(th_, jumps))
    t('pmf_and_moments + rewards [0,2,0]', lambda: Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True)(th_, jumps, rewards=jnp.asarray([0., 2., 0.])))
    t('moments_from_graph', lambda: Graph.moments_from_graph(pgeom(), nr_moments=2)(th_))
    t('grad pmf_and_moments moments[1]', lambda: jax.grad(lambda x: Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True)(x, jumps)[1][1])(th_))
    def gi():
        g = pgeom(); g.update_weights([th]); return g
    t('update_weights; expectation', lambda: gi().expectation())
    t('update_weights; variance', lambda: gi().variance())
    t('update_weights; pdf', lambda: gi().pdf(np.array([1, 2, 3])))
print('## B. stale contexts')
g = pgeom(); g.update_weights([0.5])
t('accumulated_visits(3) at theta=0.5', lambda: g.accumulated_visits(3))
g.update_weights([1.5])
t('accumulated_visits(5) after update_weights([1.5])  [no weight_version check, phasiccpp.h:1773]', lambda: g.accumulated_visits(5))
t('stop_probability(5) same graph', lambda: g.stop_probability(5))
g = pgeom(); g.update_weights([0.5]); c = g.distribution_context(); c.step()
g.update_weights([1.5]); 
def st():
    out = []
    for _ in range(3): c.step(); out.append(c.pmf())
    return out
t('user-held distribution_context: step x3 after update_weights([1.5])', st)
print('## C. was_dph graphs')
def cont():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, 3.0); return g
d = cont().discretize(10.0); d = d[0] if isinstance(d, tuple) else d
def rows(g): return [(i, [e.weight() for e in g.vertex_at(i).edges()]) for i in range(g.vertices_length())]
print('    discretize(10) rows:', rows(d), 'was_dph', d.get_was_dph(), 'is_discrete', d.is_discrete)
h = cont(); h.is_discrete = True; h.set_was_dph(True)
print('    rate graph + set_was_dph(True), no update_weights: rows', rows(h))
t('  pdf', lambda: h.pdf(np.array([1, 2]))); t('  expectation', lambda: h.expectation()); t('  variance', lambda: h.variance())
h = Graph(1); s = h.starting_vertex(); a = h.find_or_create_vertex([1]); b = h.find_or_create_vertex([2])
s.add_edge(a, 1.0); a.add_edge(b, [3.0]); h.is_discrete = True; h.set_was_dph(True); h.update_weights([1.0])
print('    parameterized rate graph + set_was_dph(True) + update_weights([1]): rows', rows(h), ' <- renormalised to a deterministic walk, silently')
print('## D. rate graph flagged is_discrete by hand (user error)')
for r in (0.3, 3.0):
    h = Graph(1); s = h.starting_vertex(); a = h.find_or_create_vertex([1]); b = h.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, r); h.is_discrete = True
    print(f'# rate {r}')
    t('pdf(1..3)', lambda: h.pdf(np.array([1, 2, 3]))); t('expectation', lambda: h.expectation()); t('variance', lambda: h.variance())
print('## E. self-loops')
def selfloop(stay, go):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, go); a.add_edge(a, stay); g.is_discrete = True; return g
for stay, go in ((0.5, 0.5), (0.6, 0.9)):
    print(f'# a->a {stay}, a->b {go}')
    t('build', lambda: rows(selfloop(stay, go)) and 'ok')
    t('pdf(1..3)', lambda: selfloop(stay, go).pdf(np.array([1, 2, 3])))
    t('expectation', lambda: selfloop(stay, go).expectation())
    t('sample', lambda: (random.seed(1), selfloop(stay, go).sample(4))[1])
print('## F. normalize_discrete then row sums')
g = Graph(1); s = g.starting_vertex(); vs = [g.find_or_create_vertex([i + 1]) for i in range(4)]
s.add_edge(vs[0], 1.0)
for i in range(3): vs[i].add_edge(vs[i + 1], [0.1, 0.3, 0.7][i])
g.is_discrete = True; t('normalize()', lambda: g.normalize()); print('    rows after normalize_discrete:', rows(g))
t('pdf(1..3) after normalize (aux vertices count as steps?)', lambda: g.pdf(np.array([1, 2, 3])))
