"""E8 (prototype): relative error of the SHIPPED jax.grad at small theta.
References: exact final-epoch gradient (daisy efg slot 2,3); Richardson
central FD on the log-likelihood with h = theta*{1e-3,5e-4} (power-law
cases, measured agreement 3e-7 in E1)."""
import sys, os
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import sites as S
from test_notebook_model_recovery import coalescent_indexed
from phasic import Graph
print('rule', os.environ.get('PHASIC_FD_RULE', 'new'), 'c', os.environ.get('PHASIC_FD_REL', '3e-3'))
def rich(f, th, i):
    v = float(th[i]); d = lambda h: (float(f(th.at[i].add(h))) - float(f(th.at[i].add(-h))))/(2*h)
    return (4*d(v*5e-4) - d(v*1e-3))/3
g3, idx3 = coalescent_indexed(3)
J10 = g3.joint_prob_graph(idx3, reward_limit=10, discrete=False)
obs = [[k, 0, 0] for k in (0, 1, 3, 5, 7, 10)] + [[0, k, 0] for k in (2, 6, 10)]
oi = J10._map_joint_observations_to_indices(obs); vi = jnp.asarray(oi)
mji = Graph.pmf_from_graph_joint_index(J10, theta_dim=2, observed_indices=oi)
f_ji = lambda t: jnp.sum(jnp.log(mji(t, vi)[0]))
f_pm = S.site_pmf_and_moments()
md, *_ = J10._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=np.array([0.0, 0.15]), user_prior=None, user_fixed=None)
me, *_ = J10._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=np.array([0.0, 0.15]), user_prior=None, user_fixed=None, exact_final_grad=True)
ll = lambda m: (lambda t: jnp.sum(jnp.log((lambda o: o[0] if isinstance(o, tuple) else o)(m(t)))))
f_d, f_e = ll(md), ll(me)
for v in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
    row = []
    th = jnp.array([2.0, v]); row.append(abs(float(jax.grad(f_ji)(th)[1]) - rich(f_ji, th, 1))/abs(rich(f_ji, th, 1)))
    th = jnp.array([v, 0.5]); row.append(abs(float(jax.grad(f_pm)(th)[0]) - rich(f_pm, th, 0))/abs(rich(f_pm, th, 0)))
    for i in (2, 3):
        th = jnp.array([2.0, 0.5, 2.0, 0.5]).at[i].set(v)
        try: row.append(abs(float(jax.grad(f_d)(th)[i]) - float(jax.grad(f_e)(th)[i]))/abs(float(jax.grad(f_e)(th)[i])))
        except Exception: row.append(np.nan)
    print(f'theta={v:6.0e}  joint_index(mu,k<=10)={row[0]:.1e}  pmf(coal)={row[1]:.1e}  daisy final coal={row[2]:.1e}  daisy final mu(k<=10)={row[3]:.1e}', flush=True)
