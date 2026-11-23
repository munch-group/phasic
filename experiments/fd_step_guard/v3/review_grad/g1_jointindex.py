"""G1: joint-index (elimination-based) forward noise and relative-step FD error
through the real jax.grad path (prototype worktree, PHASIC_FD_RULE=rel).
Reference: Richardson on hand central differences at relative steps 2e-2/1e-2
(O(h^4)), cross-checked against exact_grad=True where it commits."""
import sys, os, time, warnings
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, os.path.join(os.getcwd(), 'tests/pytest'))
warnings.filterwarnings('ignore')
from test_notebook_model_recovery import coalescent_indexed
import phasic
from phasic import Graph
print('phasic', phasic.__file__, 'proto', hasattr(phasic, '_fd_half_step'), flush=True)

def run(n, RL, base):
    t0 = time.time()
    g, idx = coalescent_indexed(n)
    J = g.joint_prob_graph(idx, reward_limit=RL, discrete=False)
    nv = J.vertices_length()
    obs = [[0]*n, [1]+[0]*(n-1), [0,1]+[0]*(n-2), [RL]+[0]*(n-1), [1,1]+[0]*(n-2), [0,0,1]+[0]*(n-3)]
    try:
        oi = J._map_joint_observations_to_indices(obs)
    except Exception as e:
        print('obs map failed', e); return
    m = Graph.pmf_from_graph_joint_index(J, theta_dim=2, observed_indices=oi)
    vi = jnp.asarray(oi)
    fwd = jax.jit(lambda th: m(th, vi)[0])
    ll = lambda th: jnp.sum(jnp.log(m(th, vi)[0]))
    print(f'## n={n} RL={RL} V={nv} build {time.time()-t0:.1f}s', flush=True)
    for th0 in base:
        th0 = np.array(th0, float)
        p0 = np.asarray(fwd(jnp.asarray(th0)))
        print(f' theta={th0.tolist()} p range {p0.min():.2e}..{p0.max():.2e}')
        for i in range(2):
            e = np.eye(2)[i]
            F = lambda r: np.asarray(fwd(jnp.asarray(th0 * (1 + e * r))))
            js = np.arange(-10, 11)
            P = np.array([F(j * 1e-12) for j in js])
            noise = max(np.std(P[:, k] - np.polyval(np.polyfit(js, P[:, k], 2), js)) / abs(p0[k]) for k in range(P.shape[1]))
            # dlogL/dlog(theta_i) by hand at relative step r
            def dl(r): return np.sum(np.log(F(r)) - np.log(F(-r))) / (2 * r)
            ref = (4 * dl(1e-2) - dl(2e-2)) / 3
            ref2 = (4 * dl(5e-3) - dl(1e-2)) / 3
            row = []
            for c in ('1e-2', '1e-3', '1e-4', '1e-5', '1e-6'):
                os.environ['PHASIC_FD_RULE'] = 'rel'; os.environ['PHASIC_FD_REL'] = c
                gr = float(np.asarray(jax.grad(ll)(jnp.asarray(th0)))[i]) * th0[i]
                row.append(abs(gr - ref2))
            os.environ['PHASIC_FD_RULE'] = 'old'
            gold = float(np.asarray(jax.grad(ll)(jnp.asarray(th0)))[i]) * th0[i]
            print(f'  param {i}: fwd rel noise {noise:.1e}; dlogL/dlog ref {ref2:+.8e} (Richardson self-consistency {abs(ref-ref2):.1e});'
                  f' |err| rel-step 1e-2..1e-6: ' + ' '.join(f'{x:.1e}' for x in row) + f'; today(abs 1e-7): {abs(gold-ref2):.1e}', flush=True)

cases = [(4, 3), (6, 3), (7, 3)] if len(sys.argv) < 2 else [tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]]
for n, RL in cases:
    run(n, RL, [[1.0, 0.5], [1.0, 1e-3], [1e-4, 1e-8], [2e4, 1e-2]])
