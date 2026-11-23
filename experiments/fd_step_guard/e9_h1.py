import sys, os, logging
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
import sites as S
from phasic import Graph
logging.disable(logging.WARNING)
print('rule', os.environ.get('PHASIC_FD_RULE', 'new'))
J, oi = S.joint(); vi = jnp.asarray(oi)
me = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi, exact_grad=True, exact_grad_decline='fd')
le = lambda t: jnp.sum(jnp.log(me(t, vi)[0]))
for v in (1e-9, 1e-8, 1e-6, 1e-4):
    th = jnp.array([2.0, v]); ge = float(jax.grad(le)(th)[1])
    d = lambda h: (float(le(th.at[1].add(h))) - float(le(th.at[1].add(-h))))/(2*h)
    rr = (4*d(v*5e-3) - d(v*1e-2))/3
    print(f'  mu={v:g}: exact-path grad={ge:.8e} reference={rr:.8e} relerr={abs(ge-rr)/abs(rr):.1e}')
