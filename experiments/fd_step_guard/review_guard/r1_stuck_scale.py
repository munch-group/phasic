"""Does a fail-soft particle stay stuck at rates far below 1e19?
Same as E6 pmf branch, particle 0 initialised at theta0 (argv), where the
native pdf already fails-soft (1e9 step cap) for every observed time."""
import sys, random, warnings
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph
from phasic import GaussPrior
from phasic.svgd import _inverse_softplus
random.seed(1); np.random.seed(1)
t0 = float(sys.argv[1]); NIT = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NP = 12
G = coalescent_graph(5); G.update_weights([2.0]); data = G.sample(200)
print('min data time', np.min(data), 'max', np.max(data))
theta = np.full((NP, 1), 2.0); theta[0, 0] = t0
init = np.asarray(_inverse_softplus(jnp.asarray(theta)))
s = G.svgd(data, prior=GaussPrior(ci=[0.5, 8], lower=0.0), theta_init=init,
           n_particles=NP, n_iterations=NIT, progress=False, learning_rate=0.01, seed=1)
P = np.asarray(s.get_results()['particles'])
print('theta0 init', t0, '-> final particle 0:', P[0], ' others median:', np.median(P[1:], axis=0))
