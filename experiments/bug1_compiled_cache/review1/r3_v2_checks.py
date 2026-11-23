"""Re-review of plan v2: (a) does precompile=False avoid the class cache? (b) jit=False, precompile=False?
HOME redirected; unmodified code."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['HOME'] = os.path.join(HERE, 'fakehome3'); os.makedirs(os.environ['HOME'], exist_ok=True)
import random, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
from test_notebook_model_recovery import coalescent_graph
from phasic import Graph, GaussPrior, SVGD
KW = dict(n_particles=8, n_iterations=5, learning_rate=0.01, progress=False, seed=1)
random.seed(3); np.random.seed(3)
g = coalescent_graph(4); g.update_weights([8.0]); d = np.asarray(g.sample(100))
model = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=1)
for label, extra in (('defaults', {}), ('precompile=False', dict(precompile=False)),
                     ('jit=False', dict(jit=False)), ('jit=False, precompile=False', dict(jit=False, precompile=False))):
    SVGD._compiled_cache.clear()
    s = SVGD(model=model, observed_data=d, theta_dim=1, prior=GaussPrior(ci=[0.5, 20], lower=0.0), **KW, **extra)
    s.optimize()
    print(f'{label:32s} jit_enabled={s.jit_enabled}  cache entries after fit: {len(SVGD._compiled_cache)}')
