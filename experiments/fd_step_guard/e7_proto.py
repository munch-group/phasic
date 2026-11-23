"""E7 (prototype branch): run once with PHASIC_FD_RULE=old and once without.
Saves vmap(grad) at normal and small theta for 4 shipped sites, times it, and
runs an end-to-end epoch fit with one particle near the floor."""
import sys, os, time, random, warnings
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
import sites as S
import phasic
from phasic import Graph, GaussPrior
from phasic.svgd import _inverse_softplus
tag = sys.argv[1]
print('phasic from', phasic.__file__)
rng = np.random.default_rng(0)
out = {}
SITES = {'daisy': (S.site_daisy, 4), 'daisy_exp': (lambda: S.site_daisy(exposure=True), 4),
         'joint_index': (S.site_joint_index, 2), 'pmf_and_moments': (S.site_pmf_and_moments, 2)}
for name, (mk, d) in SITES.items():
    f = mk()
    vg = jax.jit(jax.vmap(jax.grad(f)))
    normal = jnp.asarray(rng.uniform(1e-5, 5.0, size=(32, d)))       # every component >= 1e-5
    small = jnp.asarray(10.0 ** rng.uniform(-9, -5, size=(32, d)))    # every component small
    mixed = normal.at[:, 0].set(small[:, 0])
    for lab, th in (('normal', normal), ('small', small), ('mixed', mixed)):
        try:
            g = np.asarray(vg(th)); out[f'{name}/{lab}'] = g
        except Exception as e:
            out[f'{name}/{lab}'] = np.full((32, d), np.nan); print(name, lab, 'ERROR', str(e).split('\n')[0][:120])
    vg(normal).block_until_ready()
    ts = []
    for _ in range(5):
        t = time.perf_counter(); vg(normal).block_until_ready(); ts.append(time.perf_counter() - t)
    out[f'{name}/time'] = np.array(ts)
    print(f'{name}: vmap(grad) 32 particles median {np.median(ts)*1e3:.1f} ms', flush=True)
np.savez(f'/private/tmp/claude-501/-Users-kmt-phasic/3666ba56-7958-4c29-a8b2-ec58b11551e2/scratchpad/fd/e7_{tag}.npz', **out)
# end-to-end epoch fit, one particle near the floor
random.seed(1); np.random.seed(1)
g, idx = S.coalescent_indexed(4)
J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
obs = [[0,0,0,0]]*40 + [[1,0,0,0]]*20 + [[0,1,0,0]]*10 + [[2,0,0,0]]*8
NP = 12
theta = np.tile([2.0, 0.5, 2.0, 0.5], (NP, 1)); theta[0, 2] = 5e-8; theta[1, 0] = 1e-9
init = np.log(np.expm1(theta))   # exact phi, no 1e-6 floor
print('init theta check', np.asarray(jax.nn.softplus(jnp.asarray(init)))[[0,1],[2,0]])
t = time.time()
try:
    s = J.svgd(obs, epoch_starts=[0.0, 0.15], fixed=[(1, 0.5)],
               prior=[GaussPrior(ci=[0.5, 8], lower=0.0), None]*2, theta_init=init,
               n_particles=NP, n_iterations=40, progress=False, learning_rate=0.01, seed=1)
    P = np.asarray(s.get_results()['particles'])
    print(f'e2e: completed in {time.time()-t:.1f}s; particle0 epoch2 coal {P[0,2]:.3g}, particle1 epoch1 coal {P[1,0]:.3g}; mean {np.asarray(s.theta_mean)}')
except Exception as e:
    print(f'e2e: FAILED after {time.time()-t:.1f}s:', type(e).__name__, str(e).split(chr(10))[0][:200])
