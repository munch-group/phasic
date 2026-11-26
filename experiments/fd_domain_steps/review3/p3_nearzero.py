"""P3: parameters whose effect is LINEAR near 0 (finite dF/dtheta at theta->0).
Reference = one-sided 2nd-order theta-space difference with probes theta,
theta+h, theta+2h (all positive), Richardson over h=1e-4,5e-5 (error ~h^2).
Compares master's 1e-7 step (where it doesn't crash), the capped and the full
log rule. Shows how log-coordinate FD's relative error grows as 1/theta."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, set_log_level
from test_notebook_model_recovery import two_island_graph
from common import *
set_log_level("ERROR")
exec(open(os.path.join(os.path.dirname(__file__), 'p1_daisy.py')).read().split("def F(th)")[0])  # builds `model`
def Fd(th):
    p, _ = model(jnp.asarray(th), None); return np.asarray(p, np.float64)
g = two_island_graph(4)
mp = Graph.pmf_from_graph(g, theta_dim=2)
TIMES = jnp.array([0.3, 0.8, 1.5, 2.5])
def Fp(th): return np.asarray(mp(jnp.asarray(th), TIMES), np.float64)

def ref_onesided(F, th, i, cot):
    def D(h):
        t1 = th.copy(); t1[i] += h; t2 = th.copy(); t2[i] += 2 * h
        return float(np.sum(cot * (-3 * F(th) + 4 * F(t1) - F(t2)))) / (2 * h)
    a, b = D(1e-4), D(5e-5)
    return (4 * b - a) / 3, rel(a, b)

for name, F, base, i in [("pdf two-island mig", Fp, [1.0, None], 1),
                         ("daisy epoch-1 coal", Fd, [None, 1e-4, 1e-4, 1e-4], 0),
                         ("daisy epoch-1 coal (ordinary)", Fd, [None, 0.002, 1.5, 0.005], 0)]:
    print(f"\n### {name}")
    for v in [1e-3, 1e-5, 1e-7, 1e-8, 1e-9, 1e-11, 1e-13]:
        th = np.array([v if b is None else b for b in base], float)
        cot = 1.0 / F(th)
        ref, ra = ref_onesided(F, th, i, cot)
        try: mt = rel(fd_today(F, th, i, cot), ref)
        except Exception as e: mt = float('nan')
        line = f"  theta={v:.0e} ref={ref: .9e} (ref h-agree {ra:.1e}) master {mt:.1e}"
        for s in (1e-3, 1e-4, 1e-5):
            line += f" | s={s:.0e} full {rel(fd_log(F, th, i, cot, s), ref):.1e} cap {rel(fd_log(F, th, i, cot, s, cap=1e-7), ref):.1e}"
        print(line, flush=True)
