"""E2b: power-law worst case: joint graph reward_limit=10, observations with
k up to 10 mutations, mu small. Reference: exact final-epoch gradient (daisy
efg) and Richardson on log-lik with h = mu*{1e-3,5e-4} (joint_index)."""
import sys
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
from phasic import Graph
EPS = 1e-7
g, idx = coalescent_indexed(3)
J = g.joint_prob_graph(idx, reward_limit=10, discrete=False)
print('vertices', J.vertices_length())
obs = [[k, 0, 0] for k in (0, 1, 3, 5, 7, 10)] + [[0, k, 0] for k in (2, 6, 10)]
oi = J._map_joint_observations_to_indices(obs)
vi = jnp.asarray(oi)
mji = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi)
Pji = lambda th: np.asarray(mji(th, vi)[0])
md, *_ = J._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=np.array([0.0, 0.15]),
                                    user_prior=None, user_fixed=None)
me, *_ = J._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=np.array([0.0, 0.15]),
                                    user_prior=None, user_fixed=None, exact_final_grad=True)
def Pd(th):
    o = md(th); return np.asarray(o[0] if isinstance(o, tuple) else o)
def exact_d(th):
    f = lambda t: jnp.sum(jnp.log((lambda o: o[0] if isinstance(o, tuple) else o)(me(t))))
    return np.asarray(jax.grad(f)(th))
def fd(P, th, i, cot, h):
    pp, pm = P(th.at[i].add(h)), P(th.at[i].add(-h))
    return float(np.sum(cot*(pp-pm))/(2*h))
CS = (0.5, 0.1, 3e-2, 1e-2, 3e-3, 1e-3)
for name, P, base, i, exact in (('joint_index', Pji, [2.0, None], 1, None),
                                ('daisy final mu', Pd, [2.0, 0.5, 2.0, None], 3, exact_d),
                                ('daisy epoch1 mu', Pd, [2.0, None, 2.0, 0.5], 1, None)):
    print(f'### {name}')
    print('  theta     ' + ''.join(f'{"c="+format(c,"g"):>10}' for c in CS) + '       cur')
    for v in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        th = jnp.array([v if b is None else b for b in base], dtype=jnp.float64)
        cot = 1.0/P(th)
        if exact is not None:
            ref = float(exact(th)[i])
        else:
            ll = lambda t: float(np.sum(np.log(P(t))))
            d = lambda h: (ll(th.at[i].add(h)) - ll(th.at[i].add(-h)))/(2*h)
            ref = (4*d(v*5e-4) - d(v*1e-3))/3
        row = ''.join(f'{abs(fd(P, th, i, cot, min(EPS, c*v))-ref)/abs(ref):10.1e}' for c in CS)
        try: cur = f'{abs(fd(P, th, i, cot, EPS)-ref)/abs(ref):10.1e}'
        except Exception: cur = '     error'
        print(f'  {v:8.0e} {row}{cur}', flush=True)
