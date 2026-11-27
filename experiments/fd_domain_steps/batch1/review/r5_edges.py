"""Review probe r5: edge behaviours.
E1  'unit' slot at exactly 0 on a refusing model (epoch) vs undeclared at 0.
E1b 'unit' slot at exactly 0 on a native-discrete pmf_from_graph.
E2  'fixed' domain slot in the FINAL epoch with exact_final_grad=True: zero?
E3  rvp FDStats are process-global, not per model.
E4  'unit' probes for t within a few ulp of 1 (sigmoid rounding).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import phasic
import jax, jax.numpy as jnp
import numpy as np
from phasic import Graph, _fd_steps
import _fd_golden_cells as C


def tryg(label, fn):
    try:
        g = np.asarray(fn())
        print(f"{label}: OK grad={g}", flush=True)
    except Exception as e:
        print(f"{label}: RAISES {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)


jpg, oi = C.joint_graphs()
kw = dict(observed_indices=list(oi), epoch_starts=np.array([0.0, 0.5]), user_prior=None, user_fixed=None)
th = jnp.asarray([1.0, 0.0, 1.0, 0.5])
for doms in (None, ["positive", "unit", "positive", "unit"], ["positive", "positive", "positive", "positive"]):
    m, _, _, _ = jpg._daisy_chain_svgd_model(param_domains=doms, **kw)
    tryg(f"E1 epoch slot1=0 domains={doms}", lambda: jax.grad(lambda t: jnp.sum(jnp.log(m(t, None)[0])))(th))

k = jnp.asarray([1.0, 2.0, 3.0])
for doms in (None, ["unit", "unit"]):
    m = Graph.pmf_from_graph(C.beside(), discrete=True, theta_dim=2, param_domains=doms)
    tryg(f"E1b dph slot1=0 domains={doms}", lambda: jax.grad(lambda t: jnp.sum(m(t, k)))(jnp.asarray([0.3, 0.0])))

th = jnp.asarray([1.0, 0.5, 1.5, 0.3])
for efg in (False, True):
    m, _, _, _ = jpg._daisy_chain_svgd_model(param_domains=[None, None, None, "fixed"], exact_final_grad=efg, **kw)
    tryg(f"E2 'fixed' final-epoch slot 3, exact_final_grad={efg}",
         lambda: jax.grad(lambda t: jnp.sum(jnp.log(m(t, None)[0])))(th))
    m, _, _, _ = jpg._daisy_chain_svgd_model(param_domains=[None, None, None, "fixed"], exact_final_grad=efg,
                                             exposure_arr=np.linspace(0.5, 2.0, len(oi)), exposure_param_index=1, **kw)
    tryg(f"E2 exposure 'fixed' final-epoch slot 3, exact_final_grad={efg}",
         lambda: jax.grad(lambda t: jnp.sum(jnp.log(m(t, None)[0])))(th))

from phasic import ffi_wrappers as FW
g1, g2 = C.beside("formula"), C.beside("formula")
tgt = jnp.asarray([1, 2], dtype=jnp.int32)
for g in (g1, g2):
    sj = g.serialize(theta_dim=2); ipv = jnp.asarray(g._initial_probability_vector())
    # offset slot tie at 1e-13 to trigger a counted tie
    f = lambda t: FW.compute_reward_visit_probability_ffi(sj, t, tgt, ipv)
    tryg("E3 rvp grad", lambda: jax.grad(f)(jnp.asarray([1.0, 1e-13])))
print("E3 rvp global stats object:", FW._RVP_FD_STATS.site, "tie=", FW._RVP_FD_STATS.tie,
      "zero=", FW._RVP_FD_STATS.zero_difference, "(two distinct graphs, one counter)", flush=True)

info = _fd_steps.BuildInfo(True, False, "capped")
for t in (1 - 1e-12, 1 - 1e-15, 1 - 2.2e-16, np.nextafter(1.0, 0.0)):
    for fam in ("capped", "full"):
        th2 = jnp.asarray([0.5, t])
        tp, tm, use = _fd_steps.probes(th2, 1, th2.at[1].add(1e-7), th2.at[1].add(-1e-7), _fd_steps.UNIT,
                                       info._replace(family=fam))
        print(f"E4 unit t=1-{1-t:.3g} {fam}: up<1? {float(tp[1]) < 1.0} up-dn={float(tp[1]-tm[1]):.3g} use={bool(use)}", flush=True)
