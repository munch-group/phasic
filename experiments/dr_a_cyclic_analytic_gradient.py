"""DR-A — make-or-break: is an ANALYTIC gradient of a moment feasible on a
CYCLIC graph, and is it mixed-scale-robust where FD fails?

Cyclic CTMC:  start -> A ;  A <-> B (rate th0 each way, the cycle = SCC{A,B}) ;
              B -> absorb (rate th1).  A has NO direct exit.
Closed form (verified):  E[T from A] = 1/th0 + 2/th1
                         dE/dth0 = -1/th0^2 ,  dE/dth1 = -2/th1^2.

DR-A.1  Analytic gradient via the sub-generator linear solve E[T]=1^T(-S)^-1 a
        -- cyclic-correct (the inverse absorbs the cycle) and JAX-differentiable
        (jnp.linalg.solve has an exact VJP). Compare to (i) the closed-form
        oracle, (ii) phasic's native forward, (iii) the production FD gradient,
        at moderate and MIXED scale.
DR-A.2  Confirm the current Tier-1 blocker: record_elimination_trace RAISES on
        the cyclic graph (the self-loop of Gaussian elimination), and note the
        correction the C tape already applies.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import phasic
from phasic import Graph
import jax
import jax.numpy as jnp


# ---- phasic cyclic graph (native forward + production FD gradient) ----
def cyclic_graph():
    g = Graph(1)
    s = g.starting_vertex()
    A = g.find_or_create_vertex([2]); B = g.find_or_create_vertex([1]); _ = g.find_or_create_vertex([0])
    s.add_edge(A, 1.0)
    A.add_edge(B, [1.0, 0.0])   # A -> B  rate th0
    B.add_edge(A, [1.0, 0.0])   # B -> A  rate th0   (the cycle)
    B.add_edge(g.find_or_create_vertex([0]), [0.0, 1.0])   # B -> absorb rate th1
    return g


_model = Graph.pmf_and_moments_from_graph(cyclic_graph(), nr_moments=2, theta_dim=2)
_T = jnp.array([1.0])


def ET_phasic(theta):
    _pmf, m = _model(theta, _T)
    return m[0]


# ---- analytic E[T] via the sub-generator solve (cyclic-correct, differentiable) ----
def ET_analytic(theta):
    th0, th1 = theta[0], theta[1]
    negS = jnp.array([[th0, -th0],           # -S on {A,B}
                      [-th0, th0 + th1]])
    a = jnp.array([1.0, 0.0])                 # start in A
    t = jnp.linalg.solve(negS, jnp.ones(2))   # (-S)^-1 1  = per-state E[T]
    return jnp.dot(a, t)


def oracle(theta):
    return np.array([-1.0 / theta[0] ** 2, -2.0 / theta[1] ** 2])


grad_phasic = jax.grad(ET_phasic)             # production FD
grad_analytic = jax.grad(ET_analytic)         # exact AD through linalg.solve


def relerr(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


print("=" * 82)
print("DR-A.1  analytic (linalg.solve AD) vs FD vs closed-form oracle on a CYCLIC graph")
print("=" * 82)
for th in ([1.0, 1.0], [1.0, 1e-4], [1.0, 1e-8]):
    thj = jnp.array(th)
    print(f"\ntheta = {th}")
    print(f"  forward  E[T]: phasic={float(ET_phasic(thj)):.8g}  analytic={float(ET_analytic(thj)):.8g}  "
          f"closed={1/th[0] + 2/th[1]:.8g}")
    orc = oracle(th)
    ga = np.array(grad_analytic(thj))
    try:
        gf = np.array(grad_phasic(thj))
        gf_s = [f"{x:.6g}" for x in gf]; gf_re = relerr(gf, orc)
    except Exception as e:
        gf_s = [f"CRASH:{type(e).__name__}"] * 2; gf_re = np.array([np.nan, np.nan])
    print(f"  {'grad':>10} {'oracle':>14} {'ANALYTIC(AD)':>16} {'a.relerr':>10} {'FD(prod)':>16} {'fd.relerr':>10}")
    for i in range(2):
        print(f"  {'dth'+str(i):>10} {orc[i]:>14.6g} {ga[i]:>16.6g} {relerr(ga[i], orc[i]):>10.2e} "
              f"{gf_s[i]:>16} {gf_re[i]:>10.2e}")

print()
print("=" * 82)
print("DR-A.2  current Tier-1 blocker: does record_elimination_trace handle the cycle?")
print("=" * 82)
try:
    from phasic.trace_elimination import record_elimination_trace
    g = cyclic_graph()
    try:
        tr = record_elimination_trace(g)
        print("  record_elimination_trace SUCCEEDED (unexpected):", type(tr).__name__)
    except Exception as e:
        print(f"  record_elimination_trace RAISED: {type(e).__name__}: {str(e)[:110]}")
except Exception as e:
    print("  could not import record_elimination_trace:", e)
print("  (The C PRC tape handles the cycle via add_command(parent,parent,1/(1-q));")
print("   the Python trace lacks that self-loop correction -> raises. Engineering gap, not fundamental.)")
