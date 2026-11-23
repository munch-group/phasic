"""X4: ONE finite-difference rule, a central step in log(theta):
   dF/dlog(theta_i) ~ (F(theta_i*e^s) - F(theta_i*e^-s)) / (2s)
on the true-uniformization forward.  Error = |FD - exact| in units of the
per-observation log-likelihood gradient w.r.t. log(theta) (what an optimiser
in a positive-parameter space consumes).  Closed forms:
  Erlang-k(theta):  log f = k log(theta) + (k-1) log t - theta t - lgamma(k);  d/dlog(theta) = k - theta t
  hypoexp(l1,l2):   f = l1 l2/(l2-l1) (e^{-l1 t} - e^{-l2 t})"""
import sys, math
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard/v3')
from unif import pdf_cdf, max_rate
from phasic import Graph

def chain(rates):
    g = Graph(1); s = g.starting_vertex()
    vs = [g.find_or_create_vertex([i + 1]) for i in range(len(rates) + 1)]
    s.add_edge(vs[0], 1.0)
    for i, r in enumerate(rates): vs[i].add_edge(vs[i + 1], r)
    return g
def logpdf(rates, t):
    sc = max(rates)                      # time rescaling (prototype's integer-G limitation)
    p, _, _ = pdf_cdf(chain([r / sc for r in rates]), [t * sc])
    return math.log(p[0]) + math.log(sc)
def fd_log(build, th, i, t, s):
    up = list(th); dn = list(th); up[i] *= math.exp(s); dn[i] *= math.exp(-s)
    return (logpdf(build(up), t) - logpdf(build(dn), t)) / (2 * s)

S = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
print('columns: |FD - exact| of d log f / d log theta, for log-step s = ' + ' '.join(f'{s:g}' for s in S))
print('## Erlang-k, theta*t = 0.5 (same distribution at every scale)')
for k in (3, 10, 30):
    for th in (1e-9, 1.0, 1e6):
        t = 0.5 / th; ex = k - th * t
        row = ' '.join(f'{abs(fd_log(lambda v: [v[0]] * k, [th], 0, t, s) - ex):8.1e}' for s in S)
        print(f'  k={k:2d} theta={th:6.0e} exact={ex:6.2f}: {row}', flush=True)
print('## Erlang-k in the power-law regime, theta*t = 1e-6 (f ~ theta^k)')
for k in (3, 10, 30):
    for th in (1e-9, 1.0):
        t = 1e-6 / th; ex = k - th * t
        row = ' '.join(f'{abs(fd_log(lambda v: [v[0]] * k, [th], 0, t, s) - ex):8.1e}' for s in S)
        print(f'  k={k:2d} theta={th:6.0e} exact={ex:6.2f}: {row}', flush=True)
print('## hypoexponential at mixed scale: theta = [1, 1e-8]')
def ex_h(l1, l2, t):
    e1, e2 = math.exp(-l1 * t), math.exp(-l2 * t); d = e1 - e2
    g1 = 1 + l1 / (l2 - l1) + l1 * (-t * e1) / d
    g2 = 1 - l2 / (l2 - l1) + l2 * (t * e2) / d
    return g1, g2
for t in (1.0, 1e4, 1e8):
    ex = ex_h(1.0, 1e-8, t)
    for i in (0, 1):
        row = ' '.join(f'{abs(fd_log(lambda v: v, [1.0, 1e-8], i, t, s) - ex[i]):8.1e}' for s in S)
        print(f'  t={t:6.0e} d/dlog(theta_{i}) exact={ex[i]: .6e}: {row}', flush=True)
