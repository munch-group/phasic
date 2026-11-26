"""P2: default continuous pdf path (pmf_from_graph, Euler + auto granularity).
(a) mixed scale: two-island, migration small; master jax.grad vs prototype
    full / capped log rule vs Richardson reference.
(b) jump regime (scale 200, max rate ~3600): distribution of FD error over 200
    nearby theta for today's step, the capped rule and the full rule, against
    the derivative of the same pdf at FIXED granularity (smooth)."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
from test_notebook_model_recovery import two_island_graph
from common import *
set_log_level("ERROR")

g = two_island_graph(4)
m = Graph.pmf_from_graph(g, theta_dim=2)
TIMES = jnp.array([0.3, 0.8, 1.5, 2.5])
def F(th): return np.asarray(m(jnp.asarray(th), TIMES), dtype=np.float64)
gr = jax.grad(lambda th: jnp.sum(jnp.log(m(th, TIMES))))
print("(a) mixed scale, theta=[coal=1, mig]")
for mig in [1e-3, 1e-5, 1e-7, 1e-8, 1e-9, 1e-11]:
    th = np.array([1.0, mig]); cot = 1.0 / F(th)
    try: gt = np.array(gr(jnp.asarray(th))); e = ''
    except Exception as ex: gt = np.full(2, np.nan); e = type(ex).__name__
    for i in (0, 1):
        ref, sp = richardson_log(F, th, i, cot, s=1e-2)
        line = f"  mig={mig:.0e} slot{i} ref={ref: .8e} (spread {sp:.1e}) master {rel(gt[i], ref):.1e} {e}"
        for s in (1e-3, 1e-4, 1e-5):
            line += f" | s={s:.0e} full {rel(fd_log(F, th, i, cot, s), ref):.1e} cap {rel(fd_log(F, th, i, cot, s, cap=1e-7), ref):.1e}"
        print(line, flush=True)

print("\n(b) jump regime: scale 200, times scaled down")
g6 = two_island_graph(6)
T = [0.002, 0.005, 0.01]
def Fg(th, gran=0):
    g6.update_weights(list(th))
    return np.array([g6.pdf(t, gran) for t in T])
def maxrate(th):
    g6.update_weights(list(th))
    return max(sum(e.weight() for e in g6.vertex_at(i).edges()) for i in range(g6.vertices_length()))
rng = np.random.default_rng(1)
for scale in (200.0,):
    errs = {k: [] for k in ('today', 'cap s=1e-4', 'full s=1e-4', 'full s=1e-3')}
    for k in range(200):
        th = np.array([1.0 * scale, 0.5 * scale]) * (1 + 1e-3 * rng.standard_normal(2))
        G = int(2 * maxrate(th))
        Ff = lambda x: Fg(x, G)
        cot = 1.0 / Fg(th)
        ref = fd_log(Ff, th, 0, cot, 1e-4)   # smooth at fixed G
        F0 = lambda x: Fg(x, 0)
        errs['today'].append(rel(fd_today(F0, th, 0, cot), ref))
        errs['cap s=1e-4'].append(rel(fd_log(F0, th, 0, cot, 1e-4, cap=1e-7), ref))
        errs['full s=1e-4'].append(rel(fd_log(F0, th, 0, cot, 1e-4), ref))
        errs['full s=1e-3'].append(rel(fd_log(F0, th, 0, cot, 1e-3), ref))
    for k, v in errs.items():
        v = np.array(v)
        print(f"  scale={scale:g} {k:12s} median {np.median(v):.1e} 95pct {np.percentile(v,95):.1e} max {v.max():.1e} frac>1% {np.mean(v>1e-2):.3f}")
