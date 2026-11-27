"""Today's FD gradient with 1-D rewards for a slot in the band (5e-8): the
minus probe (-5e-8) is silently mis-evaluated (a2_state_leak.log), so check the
gradient against a central difference of probes that stay positive, and
whether the call AFTER the gradient raises."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from g1_refusal_grid import build, rewards_for, TIMES

g = build("linear", "cont")
rw = rewards_for(g)
m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=False)
L = lambda th: jnp.sum(jnp.log(m(th, TIMES, rewards=rw)[0]))
th = jnp.asarray([1.0, 5e-8])
try:
    gr = np.asarray(jax.grad(L)(th))
    print("today grad:", gr)
except Exception as e:  # noqa: BLE001
    print("today grad RAISES", str(e)[-150:])
try:
    print("next forward at a valid theta [1, 0.5]:",
          np.asarray(m(jnp.asarray([1.0, 0.5]), TIMES, rewards=rw)[0]))
except Exception as e:  # noqa: BLE001
    print("next forward at [1, 0.5] RAISES:", str(e).replace("\n", " ")[-170:])
Lf = lambda t1: float(L(jnp.asarray([1.0, t1])))
for h in (2e-8, 1e-8, 5e-9):
    print(f"positive-only central difference h={h:g}: {(Lf(5e-8 + h) - Lf(5e-8 - h)) / (2 * h):.6e}")
