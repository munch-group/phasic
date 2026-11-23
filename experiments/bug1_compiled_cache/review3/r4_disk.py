# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Review 3: is a stale on-disk compiled_svgd_*.pkl really READ on master, and can a
test show it without touching the real home directory (HOME redirected in-process)?"""
import os, sys, random, pickle, pathlib, warnings
FAKE = '/Users/kmt/phasic-bug1/experiments/bug1_compiled_cache/review3/fakehome'
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_graph
from phasic import Graph, GaussPrior, SVGD
os.environ['HOME'] = FAKE          # what monkeypatch.setenv('HOME', tmp_path) does
print('Path.home() ->', pathlib.Path.home())
KW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False, seed=1)
random.seed(3); np.random.seed(3)
g = coalescent_graph(4); g.update_weights([8.0]); d = np.asarray(g.sample(120))
P = GaussPrior(ci=[0.5, 20], lower=0.0)
model = Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)
SVGD._compiled_cache.clear()
s = SVGD(model=model, observed_data=d, theta_dim=1, prior=P, **KW); s.optimize(return_history=False)
good = np.asarray(s.particles).copy()
print('files after an honest fit:', sorted(os.listdir(FAKE + '/.phasic_cache')) if os.path.isdir(FAKE + '/.phasic_cache') else 'no dir')
# plant a poisoned file under the name master would look up, then refit
SVGD._compiled_cache.clear()
s2 = SVGD(model=model, observed_data=d, theta_dim=1, prior=P, **KW)
h = s2._get_cache_key_unified(s2.nr_moments, s2.regularization, None)
path = pathlib.Path(FAKE) / '.phasic_cache' / f'compiled_svgd_{h}.pkl'
path.parent.mkdir(exist_ok=True)
with open(path, 'wb') as f: import jax.numpy as jnp; pickle.dump({'grad': jnp.zeros_like}, f)   # "gradient" = 0
s2.optimize(return_history=False); bad = np.asarray(s2.particles).copy()
print('poisoned file', path.name, '-> particles equal to honest fit:', np.array_equal(good, bad),
      ' mean honest', good.mean().round(4), ' mean poisoned', bad.mean().round(4))
path.unlink()
