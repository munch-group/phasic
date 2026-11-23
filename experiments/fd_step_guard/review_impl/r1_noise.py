"""R1: forward round-off noise floor vs graph size (joint-index forward, master).
The relative FD step's error for a smooth parameter is ~ noise / (h |p'/p|).
Measure noise as the residual of p(theta + j*d) about a quadratic fit, j=-10..10,
d tiny (so the true function is ~linear), relative to |p|."""
import sys, time, warnings
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
warnings.filterwarnings('ignore')
from test_notebook_model_recovery import coalescent_indexed
from phasic import Graph

for n in [int(a) for a in sys.argv[1:]]:
    t0 = time.time()
    g, idx = coalescent_indexed(n)
    J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    nv = J.vertices_length()
    obs = [[0] * n, [1] + [0] * (n - 1), [0, 1] + [0] * (n - 2)]
    oi = J._map_joint_observations_to_indices(obs)
    m = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi)
    vi = jnp.asarray(oi)
    f = jax.jit(lambda th: m(th, vi)[0])
    for which, base in ((0, [2.0, 0.5]), (1, [2.0, 0.5]), (1, [2.0, 1e-6])):
        th0 = np.array(base)
        d = th0[which] * 1e-9
        js = np.arange(-10, 11)
        P = np.array([np.asarray(f(jnp.asarray(th0 + np.eye(2)[which] * j * d))) for j in js])
        rel = []
        for k in range(P.shape[1]):
            c = np.polyfit(js, P[:, k], 2)
            r = P[:, k] - np.polyval(c, js)
            rel.append(np.std(r) / abs(P[0, k]))
        # implied rel-step FD error at c=3e-3 for this theta, using measured |p'/p| from fit
        slope = np.array([np.polyfit(js * d, P[:, k], 1)[0] for k in range(P.shape[1])])
        lr = np.abs(slope / P[10]) * th0[which]   # dimensionless elasticity
        print(f'n={n} V={nv} theta={base} param={which}: rel noise {np.max(rel):.1e}; '
              f'elasticity {np.min(lr):.2e}..{np.max(lr):.2e}; '
              f'implied FD err at c=3e-3 ~ {np.max(np.array(rel)/(3e-3*lr)):.1e}', flush=True)
    print(f'  ({time.time()-t0:.1f}s)', flush=True)
