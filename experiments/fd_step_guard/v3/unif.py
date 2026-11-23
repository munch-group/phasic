"""Prototype of TRUE (Poisson) uniformization on top of the existing native
step iterator: cdf(t) = sum_k Pois(k; G t) A_k, pdf(t) = G sum_k Pois(k; G t)
(A_{k+1} - A_k), with A_k the absorbed mass after k steps of P = I + Q/G.
Exact for any G >= max exit rate (up to the truncated Poisson tail)."""
import math
import numpy as np
from scipy.special import gammaln

def max_rate(g):
    # vertex 0 is the starting vertex: its edges carry the initial
    # distribution, not rates
    return max((sum(e.weight() for e in g.vertex_at(i).edges())
                for i in range(1, g.vertices_length())), default=0.0)

def absorbed_iterates(g, G, n):
    ctx = g.distribution_context(int(G))
    A = np.empty(n + 1)
    A[0] = ctx.cdf()
    for k in range(1, n + 1):
        ctx.step(); A[k] = ctx.cdf()
    return A

def n_terms(lam, tol=1e-14):
    # right truncation point of Poisson(lam): mean + generous tail
    return int(lam + 10.0 * math.sqrt(lam + 1.0) + 30)

def pois_w(lam, n):
    k = np.arange(n + 1)
    if lam == 0:
        w = np.zeros(n + 1); w[0] = 1.0; return w
    return np.exp(k * math.log(lam) - lam - gammaln(k + 1))

def unif_G(g):
    return max(1, int(math.ceil(max_rate(g) * 1.02)))

def pdf_cdf(g, times, G=None):
    """True-uniformization pdf and cdf at each time; returns (pdf, cdf, n_steps)."""
    G = unif_G(g) if G is None else int(G)
    times = np.atleast_1d(np.asarray(times, float))
    n = n_terms(G * times.max())
    A = absorbed_iterates(g, G, n + 1)
    pdf = np.empty(len(times)); cdf = np.empty(len(times))
    for j, t in enumerate(times):
        w = pois_w(G * t, n)
        cdf[j] = np.dot(w, A[:n + 1])
        pdf[j] = G * np.dot(w, A[1:n + 2] - A[:n + 1])
    return pdf, cdf, n + 1
