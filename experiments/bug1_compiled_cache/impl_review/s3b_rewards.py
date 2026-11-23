# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Rewards 1-D: is 'B after A on same model' != 'B on fresh model' an order effect or plain run-to-run noise?"""
import os, sys, tempfile, warnings, random
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ['HOME'] = tempfile.mkdtemp(prefix='fakehome_', dir=HERE)
warnings.filterwarnings('ignore'); sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
import numpy as np, jax.numpy as jnp, phasic
from phasic import Graph, GaussPrior, SVGD
from test_notebook_model_recovery import coalescent_graph
print('build:', phasic.__file__, hasattr(SVGD, '_compiled_cache'))
KW = dict(n_particles=10, n_iterations=15, learning_rate=0.01, progress=False, seed=1, theta_dim=1)
wide = lambda: GaussPrior(ci=[0.5, 20], lower=0.0)
def data(theta, n=120, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([theta]); return np.asarray(g.sample(n))
DA, DB = data(8.0), data(1.0, seed=4)
mk = lambda: Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)
RW = jnp.asarray(np.linspace(0.5, 1.5, coalescent_graph(4).vertices_length()))
def fit(m, d, **kw):
    s = SVGD(model=m, observed_data=d, prior=wide(), **{**KW, **kw}); s.optimize(); return np.asarray(s.particles)
d = lambda a, b: f'{float(np.max(np.abs(a-b))):.3e}'
for tag, kw in [('rewards', dict(rewards=RW)), ('no rewards', dict())]:
    f1 = fit(mk(), DB, **kw); f2 = fit(mk(), DB, **kw); f3 = fit(mk(), DB, **kw)
    m = mk(); s1 = fit(m, DB, **kw); s2 = fit(m, DB, **kw)          # same model, same fit twice
    m = mk(); fit(m, DA, **kw); aft = fit(m, DB, **kw)              # B after A
    print(f'{tag:11s} fresh1-fresh2 {d(f1,f2)}  fresh1-fresh3 {d(f1,f3)}  same-model twice {d(s1,s2)}  s1-fresh1 {d(s1,f1)}  B-after-A vs fresh1 {d(aft,f1)}', flush=True)
# direct model evaluation determinism with rewards
m = mk(); th = jnp.asarray([2.0]); t = jnp.asarray(DB[:5])
v = [np.asarray(m(th, t, rewards=RW)[0]) for _ in range(4)]; m2 = mk(); w = np.asarray(m2(th, t, rewards=RW)[0])
print('model(theta,t,rewards) repeat diffs', [d(v[0], x) for x in v[1:]], ' other model object', d(v[0], w))
import shutil; shutil.rmtree(os.environ['HOME'], ignore_errors=True)
