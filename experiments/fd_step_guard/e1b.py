import sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
import sites as S
from phasic import Graph
print('## shipped pmf_and_moments (clamped FD), coal small; reference = d/dtheta via central FD h=theta*1e-3')
for exact in (False, True):
    f = S.site_pmf_and_moments(exact)
    for v in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        th = jnp.array([v, 0.5])
        g = float(jax.grad(f)(th)[0])
        h = v*1e-3; r = (float(f(th.at[0].add(h))) - float(f(th.at[0].add(-h))))/(2*h)
        print(f'  exact_moment_grad={exact} theta={v:g}: grad={g:.6e} ref={r:.6e} relerr={abs(g-r)/abs(r):.2e}')
print('## joint_index near the 1e-7 pole (shipped FD)')
f = S.site_joint_index(True)
for i, base in ((1, [2.0, None]), (0, [None, 0.5])):
    for d in (-1e-3, -1e-6, -1e-9, 0.0, 1e-9, 1e-6, 1e-3):
        v = 1e-7*(1+d)
        th = jnp.array([v if b is None else b for b in base])
        try: g = float(jax.grad(f)(th)[i])
        except Exception as e: g = type(e).__name__
        print(f'  slot {i} theta=1e-7*(1{d:+g}): grad={g}')
print('## joint_index exact vs FD values, mu small (5 mutations in the data)')
J, oi = S.joint(); vi = jnp.asarray(oi)
m = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi)
me = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi, exact_grad=True, exact_grad_decline='fd')
for v in (1e-9, 1e-8, 1e-7, 1e-6):
    th = jnp.array([2.0, v])
    lf = lambda t, mm=m: jnp.sum(jnp.log(mm(t, vi)[0]))
    le = lambda t: jnp.sum(jnp.log(me(t, vi)[0]))
    gf = float(jax.grad(lf)(th)[1]); ge = float(jax.grad(le)(th)[1])
    # high-precision reference: Richardson of central FD on log-lik, h = v*{1e-2,5e-3}
    d = lambda h: (float(lf(th.at[1].add(h))) - float(lf(th.at[1].add(-h))))/(2*h)
    rr = (4*d(v*5e-3) - d(v*1e-2))/3
    print(f'  mu={v:g}: FD={gf:.8e} exact={ge:.8e} loglikRichardson={rr:.8e} 5/mu={5/v:.8e}')
