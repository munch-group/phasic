"""P0: what the EXISTING context does (ground truth for the plan's 3.1a claims)."""
import numpy as np
from scipy.linalg import expm
from phasic import Graph

def build():
    # states: 1,2,3 transient; 4,5 absorbing. theta=(a,b). defective start + atom
    g = Graph(1); g.set_param_length(2) if hasattr(g,'set_param_length') else None
    s = g.starting_vertex()
    V = {i: g.find_or_create_vertex([i]) for i in range(1, 6)}
    s.add_edge(V[1], 0.6); s.add_edge(V[2], 0.2); s.add_edge(V[4], 0.1)   # sums to 0.9
    V[1].add_edge(V[2], [2.0, 0.0]); V[2].add_edge(V[1], [0.0, 1.5]); V[2].add_edge(V[3], [0.7, 0.0])
    V[3].add_edge(V[1], [0.0, 0.3]); V[3].add_edge(V[4], [1.1, 0.0]); V[1].add_edge(V[5], [0.0, 0.2])
    V[3].add_edge(V[5], [0.05, 0.0])
    return g

def gen(g):
    n = g.vertices_length(); Q = np.zeros((n, n)); 
    vs = list(g.vertices())
    for v in vs:
        if v.index() == 0: continue
        for e in v.edges():
            Q[v.index(), e.to().index()] += e.weight(); Q[v.index(), v.index()] -= e.weight()
    ipv = np.zeros(n)
    for e in vs[0].edges(): ipv[e.to().index()] += e.weight()
    absorbing = np.array([len(v.edges()) == 0 for v in vs])
    return Q, ipv, absorbing

g = build()
g.update_weights([1.0, 1.0])
Q, ipv, ab = gen(g)
print('vertices', g.vertices_length(), 'absorbing', ab, 'ipv', ipv)
ctx = g.distribution_context()
G = 1024
print('at creation: time', ctx.time(), 'cdf', ctx.cdf(), 'pdf', ctx.pdf())
print(' stop', np.array(ctx.stop_probability(), float))
print(' avt ', np.array(ctx.accumulated_visiting_time(), float), ' ipv/G', ipv / G)
a = Q[:, ab].sum(1)  # rate into absorbing
print(' exact density at 0:', ipv @ a)
K = 300
for _ in range(K): ctx.step()
t = ctx.time(); print('after', K, 'steps: time', t, 'K/G', K / G)
for tt, lab in ((K / G, 'K/G'), ((K - 1) / G, '(K-1)/G')):
    p = ipv @ expm(Q * tt)
    print(f' expm at {lab}: pdf {(p*~ab)@a:.12g} cdf {p[ab].sum():.12g}')
print(' ctx pdf', ctx.pdf(), 'cdf', ctx.cdf())
print(' stop at absorbing vertices:', np.array(ctx.stop_probability(), float)[ab])
avt = np.array(ctx.accumulated_visiting_time(), float)
tr = ~ab; tr[0] = False
Qt = Q[np.ix_(tr, tr)]; 
ex = ipv[tr] @ np.linalg.solve(Qt, expm(Qt * t) - np.eye(Qt.shape[0]))
print(' avt ctx', avt[tr], '\n avt exact', ex, '\n diff', (avt[tr] - ex), ' ipv/G', (ipv / G)[tr])

# update_weights raising the rate far above the context's fixed G, then step
g.update_weights([5000.0, 1.0])
ctx.step()
print('after update_weights([5000,1]) + 1 step: stop', np.array(ctx.stop_probability(), float), 'cdf', ctx.cdf(), 'pdf', ctx.pdf(), 'time', ctx.time())
for _ in range(5): ctx.step()
print(' +5 steps: stop', np.array(ctx.stop_probability(), float), 'cdf', ctx.cdf())

# context created BEFORE first update_weights (tutorial cell 6 order)
g2 = build()
try:
    c2 = g2.distribution_context()
    print('ctx before update_weights: pdf', c2.pdf(), 'cdf', c2.cdf(), 'avt', np.array(c2.accumulated_visiting_time(), float))
    g2.update_weights([1.0, 1.0]); 
    print(' after update_weights (no step): pdf', c2.pdf())
except Exception as e:
    print('ERR', e)
# discrete
g3 = build(); g3.update_weights([0.1, 0.1])
try:
    print('add_aux_vertex_constant(0.0):'); list(g3.vertices())[1].add_aux_vertex_constant(0.0)
except Exception as e: print('  raises:', e)
