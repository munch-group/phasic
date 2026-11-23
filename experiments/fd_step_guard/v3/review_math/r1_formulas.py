"""R1: independent check of the Poisson read-out formulas against expm, using
the EXISTING native context iterates (index conventions) on hand-built graphs."""
import sys, math
import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard/v3')
import unif
from phasic import Graph

def build(n_trans, start, edges, n_abs=1):
    """transient states 1..n_trans, absorbing n_trans+1.. ; start: {state: w}; edges: (i,j,w)"""
    g = Graph(1); s = g.starting_vertex()
    V = {i: g.find_or_create_vertex([i]) for i in range(1, n_trans + n_abs + 1)}
    for j, w in start.items(): s.add_edge(V[j], w)
    for i, j, w in edges: V[i].add_edge(V[j], w)
    S = np.zeros((n_trans, n_trans)); alpha = np.zeros(n_trans); atom = 0.0
    for j, w in start.items():
        if j <= n_trans: alpha[j - 1] += w
        else: atom += w
    for i, j, w in edges:
        if i == j: continue
        S[i - 1, i - 1] -= w
        if j <= n_trans: S[i - 1, j - 1] += w
    return g, alpha, S, atom

def exact(alpha, S, atom, t):
    E = expm(S * t); p = alpha @ E
    cdf = atom + alpha.sum() - p.sum()
    pdf = p @ (-S @ np.ones(len(alpha)))
    acc = alpha @ np.linalg.solve(S, E - np.eye(len(alpha)))
    return cdf, pdf, p, acc

def readout(g, G, t, c_tail=10):
    lam = G * t; n = int(lam + c_tail * math.sqrt(lam + 1) + 30)
    ctx = g.distribution_context(int(G))
    A = [ctx.cdf()]; Pk = [np.array(ctx.stop_probability(), float)]
    for k in range(n + 1):
        ctx.step(); A.append(ctx.cdf()); Pk.append(np.array(ctx.stop_probability(), float))
    A = np.array(A); Pk = np.array(Pk)
    k = np.arange(n + 1); w = poisson.pmf(k, lam); sf = poisson.sf(k, lam)
    cdf = w @ A[:n + 1]; pdf = G * (w @ (A[1:n + 2] - A[:n + 1]))
    stop = w @ Pk[:n + 1]; acc = (sf @ Pk[:n + 1]) / G
    return cdf, pdf, stop, acc, n

cases = {
 'hypo(1,3)':       (2, {1: 1.0}, [(1, 2, 1.0), (2, 3, 3.0)], 1),
 'cyclic 2 abs':    (3, {1: 0.5, 2: 0.5}, [(1, 2, 2.0), (2, 1, 1.5), (2, 3, 0.7), (3, 1, 0.3), (3, 4, 1.1), (1, 5, 0.2), (3, 5, 0.05)], 2),
 'mixed 1e3/1e-2':  (3, {1: 1.0}, [(1, 2, 1e3), (2, 1, 5e2), (2, 3, 4e2), (3, 2, 1e-2), (3, 4, 2e-2)], 1),
 'defective+atom':  (2, {1: 0.6, 3: 0.1}, [(1, 2, 1.0), (2, 1, 0.5), (2, 3, 3.0)], 1),
 'self-loop':       (2, {1: 1.0}, [(1, 1, 5.0), (1, 2, 1.0), (2, 3, 3.0)], 1),
 'slow-max-differs':(3, {3: 1.0}, [(3, 2, 0.01), (2, 1, 0.1), (1, 4, 50.0), (1, 3, 25.0)], 1),
}
for name, (nt, start, edges, nabs) in cases.items():
    g, alpha, S, atom = build(nt, start, edges, nabs)
    R = unif.max_rate(g); Rtrue = max(-np.diag(S))
    print(f'## {name}: unif.max_rate={R:g} true max exit rate={Rtrue:g}')
    for t in (0.0, 1e-3, 0.3, 2.0, 40.0):
        ec, ep, es, ea = exact(alpha, S, atom, t)
        for G in sorted({int(math.ceil(R)), int(math.ceil(2 * R)) , int(math.ceil(R)) * 7}):
            if G * t > 2e5: continue
            c, p, st, ac, n = readout(g, G, t)
            st_t = st[1:nt + 1]; ac_t = ac[1:nt + 1]
            rel = lambda a, b: np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-300))
            sh = ''
            try:
                sh = f' | shipped pdf rel {abs(g.pdf(t) - ep) / max(ep, 1e-300):.1e} cdf abs {abs(g.cdf(t) - ec):.1e}'
            except Exception as e:
                sh = ' | shipped ERR ' + str(e)[:50]
            print(f'  t={t:6g} G={G:6d} n={n:6d}: cdf abs {abs(c - ec):.1e} pdf rel {rel(p, ep):.1e} stop rel {rel(st_t, es):.1e} acc rel {rel(ac_t, ea) if t > 0 else abs(ac_t).max():.1e} (stop[0]={st[0]:.1e}, abs-states stop={st[nt+1:].max():.1e}){sh}')
    # prototype with its rescaling trick
    for t in (0.3, 2.0):
        ec, ep, es, ea = exact(alpha, S, atom, t)
        g2, *_ = build(nt, start, [(i, j, w / R) for i, j, w in edges], nabs)
        p, c, n = unif.pdf_cdf(g2, [t * R])
        print(f'  prototype(rescaled) t={t}: pdf rel {abs(p[0] * R - ep) / ep:.1e} cdf abs {abs(c[0] - ec):.1e} steps {n}')
