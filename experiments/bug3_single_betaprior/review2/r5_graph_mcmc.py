"""Is the Graph.mcmc default-DataPrior difference (patched vs control) real or run-to-run noise?"""
import sys, os, warnings, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp, random
from phasic import Graph
import phasic.svgd as S
def h(a): return hashlib.md5(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:12]
obs = jnp.array([1.5, 2.0, 2.5, 3.0, 1.0, 0.7, 4.0, 2.2])
def graph2():
    g = Graph(1); v0 = g.starting_vertex(); v1 = g.find_or_create_vertex([1]); v2 = g.find_or_create_vertex([2])
    v0.add_edge(v1, [1.0, 0.0]); v1.add_edge(v2, [0.0, 1.0]); g.update_weights(np.array([1.0, 0.5])); return g
for rep in range(3):
    random.seed(0); np.random.seed(0)
    m = graph2().mcmc(observed_data=obs, theta_dim=2, n_samples=60, n_chains=2, burn_in=20, seed=3, verbose=False)
    pl = m.prior_list
    print(f'PROTO_OFF={os.environ.get("PROTO_OFF")} rep {rep}: init {h(m.theta_init)} chains {h(m._chains_phi)} results {h(m.get_results()["particles"])} '
          f'priors {[(round(float(p.mu), 6), round(float(p.sigma), 6)) for p in pl]} parallel={m.parallel} '
          f'T(0.3)={np.asarray(m.param_transform(jnp.array([0.3, -30.0]))).tolist()}')
