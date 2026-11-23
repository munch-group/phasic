"""X3: two-island model (cyclic, 21 vertices) at rate scales 1, 200, 2000
(times scaled inversely, so it is the SAME distribution).  (a) is the
true-uniformization log-likelihood smooth in theta where the shipped one
jumps?  (b) log-space FD gradient vs the scale-1 Richardson reference
(dF/dlog(theta) is invariant under the rescaling)."""
import sys, math
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard/v3')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from unif import pdf_cdf, max_rate
from test_notebook_model_recovery import two_island_graph
g = two_island_graph(4)
T = np.array([0.4, 1.0, 2.0])
def F_new(th, times):
    g.update_weights(list(th)); sc = max_rate(g)
    g.update_weights([x / sc for x in th])
    p, _, n = pdf_cdf(g, times * sc)
    return float(np.sum(np.log(p * sc))), n
def F_old(th, times):
    g.update_weights(list(th)); return float(np.sum(np.log([g.pdf(t) for t in times])))
def dlog(F, th, i, times, s):
    up = np.array(th, float); dn = up.copy(); up[i] *= math.exp(s); dn[i] *= math.exp(-s)
    return (F(up, times) - F(dn, times)) / (2 * s)
Fn = lambda th, times: F_new(th, times)[0]
ref = [(4 * dlog(Fn, [1.0, 0.5], i, T, 5e-3) - dlog(Fn, [1.0, 0.5], i, T, 1e-2)) / 3 for i in (0, 1)]
print(f'reference dF/dlog(theta) at scale 1 (Richardson, s=1e-2/5e-3): {ref[0]:.10f} {ref[1]:.10f}')
for sc in (1.0, 200.0, 2000.0):
    th = np.array([1.0, 0.5]) * sc; times = T / sc
    g.update_weights(list(th)); R = max_rate(g)
    xs = th[0] * (1 + np.linspace(-1e-3, 1e-3, 401))
    fo = np.array([F_old([x, th[1]], times) for x in xs]); fn = np.array([Fn([x, th[1]], times) for x in xs])
    rough = lambda f: np.max(np.abs(np.diff(f, 2)))       # second differences: jumps show up, curvature ~1e-11 here
    n_new = F_new(th, times)[1]; n_old = times.max() * max(1000, int(2 * max(512, R)))
    print(f'scale={sc:6g} maxrate={R:8.4g}: steps shipped {n_old:7.0f} vs new {n_new:4d};  '
          f'F shipped-new = {fo[200]-fn[200]:+.2e};  roughness shipped {rough(fo):.1e} vs new {rough(fn):.1e}')
    for s in (1e-3, 1e-4):
        e = [abs(dlog(Fn, th, i, times, s) - ref[i]) for i in (0, 1)]
        eo = [abs(dlog(F_old, th, i, times, s) - ref[i]) for i in (0, 1)]
        print(f'      log-step s={s:g}: |error| new forward {e[0]:.1e} {e[1]:.1e}   shipped forward {eo[0]:.1e} {eo[1]:.1e}')
