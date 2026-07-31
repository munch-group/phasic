"""Batch-3 MPFR safety gate: the exact moments Jacobian DECLINES (empty ->
FD fallback) when the primal would divert to MPFR (condition > 1e12), so the
double adjoint never silently mismatches the MPFR forward."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, phasic
from phasic import Graph
import jax, jax.numpy as jnp

def two_stage():
    g=Graph(1); s=g.starting_vertex()
    v3=g.find_or_create_vertex([3]);v2=g.find_or_create_vertex([2]);v1=g.find_or_create_vertex([1])
    s.add_edge(v3,1.0);v3.add_edge(v2,[1.0,0.0]);v2.add_edge(v1,[0.0,1.0]); return g

ok=True
# well-conditioned: exact path active
g=two_stage(); g.update_weights([1.0,0.5])
n_ok = len(g._moments_grad_theta(2))
print(f"well-conditioned theta=[1,0.5]: exact J size={n_ok} (expect 4)"); ok &= (n_ok==4)

# ill-conditioned (cond ~1e13 > 1e12): exact declines
gh=two_stage(); gh.update_weights([1.0,1e-13])
n_gate = len(gh._moments_grad_theta(2))
print(f"ill-conditioned theta=[1,1e-13]: exact J size={n_gate} (expect 0 -> FD fallback)"); ok &= (n_gate==0)

# end-to-end: jax.grad with exact enabled stays FINITE at high condition (FD fallback)
model=Graph.pmf_and_moments_from_graph(two_stage(),nr_moments=2,theta_dim=2,exact_moment_grad=True)
grad_hi = np.array(jax.grad(lambda th: model(th,jnp.array([1.0]))[1][0])(jnp.array([1.0,1e-13])))
finite = np.all(np.isfinite(grad_hi))
print(f"jax.grad(m0) at high condition (FD fallback): {grad_hi} finite={finite}"); ok &= finite

# and exact stays exact at benign scale
grad_lo = np.array(jax.grad(lambda th: model(th,jnp.array([1.0]))[1][0])(jnp.array([1.0,1e-4])))
r = np.max(np.abs(grad_lo-np.array([-1,-1e8]))/np.array([1,1e8]))
print(f"jax.grad(m0) at benign scale (exact): relerr={r:.2e}"); ok &= (r<1e-6)

print("ALL PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
