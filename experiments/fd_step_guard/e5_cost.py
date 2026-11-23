"""E5: evaluation cost vs rate magnitude (the 'slow band')."""
import sys, time
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed, two_island_graph
from phasic import Graph
LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
which = sys.argv[1]
def maxrate(g):
    return max(sum(e.weight() for e in g.vertex_at(i).edges()) for i in range(g.vertices_length()))
def sweep(label, fn, scales):
    for s in scales:
        t = time.time()
        try:
            info = fn(s); err = ''
        except Exception as e:
            info = ''; err = type(e).__name__ + ': ' + str(e).split('\n')[0][:120]
        dt = time.time() - t
        print(f'{label} scale={s:8.0e}  {dt:9.3f}s  {info} {err}', flush=True)
        if dt > LIMIT: print(f'{label}: stopping (>{LIMIT}s)'); break
if which.startswith('stopprob'):
    n = int(which[-1]); dt = 0.15
    g, idx = coalescent_indexed(n)
    J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    def f(s):
        J.update_weights([2.0*s, 0.5*s])
        R = maxrate(J)
        p = J.stop_probability(dt)
        return f'maxrate={R:.3g} steps~{dt*max(1000, 2*R):.3g} sum={np.sum(p):.6f}'
    sweep(f'stop_probability joint n={n} ({J.vertices_length()} v, dt={dt})', f, [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
elif which == 'daisy':
    g, idx = coalescent_indexed(6)
    J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    oi = J._map_joint_observations_to_indices([[0,0,0,0,0,0],[1,0,0,0,0,0]])
    m, *_ = J._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=np.array([0.0, 0.15]),
                                      user_prior=None, user_fixed=None)
    m(jnp.array([2.0, 0.5, 2.0, 0.5]))  # compile
    def f(s):
        o = m(jnp.array([2.0*s, 0.5, 2.0, 0.5])); o = o[0] if isinstance(o, tuple) else o
        return f'p={np.asarray(o)}'
    sweep('daisy forward joint n=6 (epoch-1 coal scaled)', f, [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
elif which == 'pmf':
    g = two_island_graph(10)
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
    T = jnp.array([0.5, 1.0, 2.5])
    m(jnp.array([1.0, 0.5]), T)
    def f(s):
        p, mom = m(jnp.array([1.0*s, 0.5*s]), T)
        return f'pmf={np.asarray(p)}'
    sweep('pmf_and_moments island n=10 (482 v), t<=2.5', f, [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
