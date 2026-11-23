"""R2: (a) pdf from cdf differences vs from per-step absorbed mass (ctx.pdf()/G) in the right tail;
(b) high-precision reference (mpmath) for the stiff case to separate expm error from uniformization noise;
(c) deep graph at short time: truncation n = lam + 10 sqrt(lam+1) + 30 vs graph depth;
(d) left truncation window vs dominant terms in the right tail."""
import sys, math
import numpy as np
import mpmath as mp
from scipy.stats import poisson
from scipy.special import gammaln
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard/v3')
import unif
from phasic import Graph
mp.mp.dps = 60

def chain(rates, extra=()):
    g = Graph(1); s = g.starting_vertex()
    vs = [g.find_or_create_vertex([i + 1]) for i in range(len(rates) + 1)]
    s.add_edge(vs[0], 1.0)
    for i, r in enumerate(rates): vs[i].add_edge(vs[i + 1], r)
    for i, j, w in extra: vs[i].add_edge(vs[j], w)
    return g

def iterates(g, G, n):
    ctx = g.distribution_context(int(G))
    A = [ctx.cdf()]; m = []
    for k in range(n + 1):
        ctx.step(); A.append(ctx.cdf()); m.append(ctx.pdf() / G)   # m[k] = absorbed in step k+1
    return np.array(A), np.array(m)

print('## (a) hypo(1,3) right tail: pdf by cdf-differences (prototype) vs by per-step absorbed mass')
g = chain([1.0, 3.0])
for t in (5.0, 10.0, 20.0, 30.0, 40.0, 60.0):
    f = 1.5 * (math.exp(-t) - math.exp(-3 * t))
    G = 4; lam = G * t; n = unif.n_terms(lam)
    A, m = iterates(g, G, n + 1)
    w = unif.pois_w(lam, n)
    p_diff = G * np.dot(w, A[1:n + 2] - A[:n + 1]); p_mass = G * np.dot(w, m[:n + 1])
    pp = unif.pdf_cdf(g, [t], G)[0][0]
    try: sh = abs(g.pdf(t) - f) / f
    except Exception as e: sh = float('nan')
    print(f'  t={t:5g} exact={f:.3e}: prototype relerr {abs(pp - f)/f:.1e} (value {pp:.3e})  cdf-diff {abs(p_diff - f)/f:.1e}  absorbed-mass {abs(p_mass - f)/f:.1e}  shipped {sh:.1e}')

print('## (b) stiff cyclic 3-state (rates 1e3,5e2,4e2,1e-2,2e-2): mpmath reference')
S = mp.matrix([[-1000, 1000, 0], [500, -900, 400], [0, mp.mpf('0.01'), mp.mpf('-0.03')]])
ex = mp.matrix([0, 0, mp.mpf('0.02')])
g = chain([1e3, 4e2, 2e-2], extra=[(1, 0, 5e2), (2, 1, 1e-2)])
for t in (0.3, 2.0, 40.0, 400.0):
    E = mp.expm(S * t, method='taylor'); f = float((mp.matrix([[1, 0, 0]]) * E * ex)[0])
    row = []
    for G in (1000, 1020, 2000):
        lam = G * t; n = unif.n_terms(lam)
        A, m = iterates(g, G, n + 1)
        k = np.arange(n + 1)
        w1 = unif.pois_w(lam, n); w2 = poisson.pmf(k, lam)
        row.append(f'G={G}: lgamma-w {abs(G*np.dot(w1, m[:n+1]) - f)/f:.1e} scipy-w {abs(G*np.dot(w2, m[:n+1]) - f)/f:.1e} cdfdiff {abs(G*np.dot(w1, A[1:n+2]-A[:n+1]) - f)/f:.1e} sum(w1)-1={w1.sum()-1:+.1e}')
    print(f'  t={t:5g} exact={f:.6e} Lambda~{1000*t:g}: ' + ' | '.join(row), flush=True)

print('## (c) Erlang depth K, rate 1, short time: truncation n vs depth (true pdf = t^(K-1) e^-t/(K-1)!)')
for K in (30, 60, 100, 150):
    g = chain([1.0] * K)
    for t in (1e-3, 0.5, 5.0):
        lf = (K - 1) * math.log(t) - t - gammaln(K)
        pp = unif.pdf_cdf(g, [t], 2)[0][0]            # prototype uses G=2 after rescaling to max rate 1
        n = unif.n_terms(2 * t)
        try: sh = g.pdf(t)
        except Exception as e: sh = float('nan')
        print(f'  K={K:3d} t={t:6g}: true log pdf {lf:9.2f} (pdf {math.exp(lf) if lf > -700 else 0:.2e})  prototype {pp:.3e} (n_terms={n})  shipped {sh:.3e}')

print('## (d) right tail, G = 1.02*rate: where are the dominant Poisson terms relative to the +-10 sigma window?')
for K, t in ((3, 300.0), (3, 600.0), (10, 600.0)):
    lam = 1.02 * t
    # absorbed mass at step j of the uniformized Erlang-K with stay prob q=0.02/1.02: negative binomial
    q = 0.02 / 1.02; p = 1 - q
    j = np.arange(K, int(lam + 10 * math.sqrt(lam) + 30) + 2)
    logm = gammaln(j) - gammaln(K) - gammaln(j - K + 1) + K * math.log(p) + (j - K) * math.log(q)
    k = j - 1
    logw = k * math.log(lam) - lam - gammaln(k + 1)
    terms = logw + logm
    tot = np.logaddexp.reduce(terms) + math.log(1.02)
    lf = (K - 1) * math.log(t) - t - gammaln(K)
    lo = lam - 10 * math.sqrt(lam)
    inwin = np.logaddexp.reduce(terms[k >= lo]) + math.log(1.02) if (k >= lo).any() else -np.inf
    print(f'  K={K} t={t}: true log pdf {lf:.3f}; full Poisson sum {tot:.3f}; dominant k = {k[np.argmax(terms)]} vs window [{lo:.0f}, {lam + 10*math.sqrt(lam):.0f}]; sum restricted to window: {inwin:.3f}')
