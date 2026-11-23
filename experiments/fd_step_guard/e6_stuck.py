"""E6: SVGD with ONE particle initialised where the forward fails fast
(rate ~1e19 -> 'too large' in the native build): what happens to it and is
the user told?  Epoch path (NaN forward) and pdf path (fail-soft penalty)."""
import sys, random, warnings, logging
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed, coalescent_graph
from phasic import GaussPrior
from phasic.svgd import _inverse_softplus
random.seed(1); np.random.seed(1)
which = sys.argv[1]
NP = 12
if which == 'daisy':
    g, idx = coalescent_indexed(4)
    J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    obs = [[0,0,0,0]]*40 + [[1,0,0,0]]*20 + [[0,1,0,0]]*10 + [[2,0,0,0]]*8
    theta = np.tile([2.0, 0.5, 2.0, 0.5], (NP, 1)); theta[0, 0] = 1e19
    init = np.asarray(_inverse_softplus(jnp.asarray(theta)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        s = J.svgd(obs, epoch_starts=[0.0, 0.15], fixed=[(1, 0.5)],
                   prior=[GaussPrior(ci=[0.5, 8], lower=0.0), None]*2,
                   theta_init=init, n_particles=NP, n_iterations=60, progress=False,
                   learning_rate=0.01, seed=1)
else:
    G = coalescent_graph(5); G.update_weights([2.0]); data = G.sample(200)
    theta = np.full((NP, 1), 2.0); theta[0, 0] = 1e19
    init = np.asarray(_inverse_softplus(jnp.asarray(theta)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        s = G.svgd(data, prior=GaussPrior(ci=[0.5, 8], lower=0.0), theta_init=init,
                   n_particles=NP, n_iterations=60, progress=False, learning_rate=0.01, seed=1)
P = np.asarray(s.get_results()['particles'])
print('final particle 0:', P[0], ' others median:', np.median(P[1:], axis=0))
print('theta_mean:', np.asarray(s.theta_mean), ' theta_std:', np.asarray(s.theta_std))
print('warnings:', sorted(set(type(x.message).__name__ + ': ' + str(x.message)[:100] for x in w)))
