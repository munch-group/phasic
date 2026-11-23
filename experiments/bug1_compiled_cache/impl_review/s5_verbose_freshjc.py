"""Impl review: (a) verbose=True paths still run (dense, regularized, sparse, no-jit); (b) epoch reference on a FRESH joint graph
(the test fixture shares one joint graph between 'shared' and 'fresh' epoch models)."""
import os, sys, tempfile, warnings, random, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ['HOME'] = tempfile.mkdtemp(prefix='fakehome_', dir=HERE)
warnings.filterwarnings('ignore'); sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
import numpy as np, jax.numpy as jnp, phasic
from phasic import Graph, GaussPrior, SVGD
from phasic.svgd import dense_to_sparse
from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
wide = lambda: GaussPrior(ci=[0.5, 20], lower=0.0); narrow = lambda: GaussPrior(ci=[0.01, 0.2], lower=0.0)
KW = dict(n_particles=6, n_iterations=4, learning_rate=0.01, progress=False, seed=1)
random.seed(3); np.random.seed(3)
g = coalescent_graph(4); g.update_weights([4.0]); d = np.asarray(g.sample(60))
mk = lambda: Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)
def run(name, f):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): s = f()
        out = buf.getvalue(); lines = [l for l in out.splitlines() if 'Precompil' in l or 'shape' in l or 'cache' in l.lower() or 'JIT compiled' in l]
        print(f'{name}: OK finite={bool(np.all(np.isfinite(np.asarray(s.particles))))}  |', ' / '.join(x.strip() for x in lines)[:260], flush=True)
    except Exception as e:
        print(f'{name}: ERROR {type(e).__name__}: {str(e)[:200]}', flush=True)
def fit(**kw):
    opt = kw.pop('opt', {}); a = dict(model=mk(), observed_data=d, theta_dim=1, prior=wide()); a.update(KW); a.update(kw)
    s = SVGD(**a); s.optimize(**opt); return s
run('verbose dense', lambda: fit(verbose=True))
run('verbose dense no_history', lambda: fit(verbose=True, opt=dict(return_history=False)))
run('verbose regularized', lambda: fit(verbose=True, regularization=1.0))
run('verbose jit=False precompile=False', lambda: fit(verbose=True, jit=False, precompile=False))
run('verbose precompile=True jit=False', lambda: fit(verbose=True, jit=False, precompile=True))
def chain():
    gg = Graph(1); s = gg.starting_vertex(); v = [gg.find_or_create_vertex([i + 1]) for i in range(3)]; va = gg.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for (a, b), c in zip([(v[0], v[1]), (v[1], v[2]), (v[2], va)], [(1.0, 0.5), (2.0, 0.25), (0.5, 1.5)]): a.add_edge(b, list(c))
    return gg
RW2 = jnp.asarray([[1.0, 0.0, 2.0, 0.5, 1.0], [1.0, 1.0, 2.0, 1.0, 0.5]])
M = np.random.default_rng(0).gamma(2.0, 1.0, size=(30, 2)); M[::4, 0] = np.nan
run('verbose sparse 2-D rewards', lambda: fit(model=Graph.pmf_and_moments_from_graph_multivariate(chain(), nr_moments=2, discrete=False, theta_dim=2),
    observed_data=dense_to_sparse(jnp.asarray(M)), theta_dim=2, prior=[wide(), wide()], rewards=RW2, verbose=True))
# (b)
L = 3; EP = [0.0, 0.4]
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 400)))]
def jc():
    gg, idx = coalescent_indexed(2); return gg.joint_prob_graph(idx, discrete=False, reward_limit=L)
EKW = dict(n_particles=12, n_iterations=25, learning_rate=0.01, progress=False, seed=1)
FULL = [(1, 1.0), (3, 1.0)]; NESTED = [(1, 1.0), (2, 0.5), (3, 1.0)]
pri = lambda mkp=wide: [mkp(), None, mkp(), None]
J = jc(); em = J.epoch_model(rows, epoch_starts=EP)
em.fit(fixed=FULL, prior=pri(wide), **EKW); em.fit(fixed=NESTED, prior=pri(), **EKW)
a = em.fit(fixed=FULL, prior=pri(narrow), **EKW)
b = J.epoch_model(rows, epoch_starts=EP).fit(fixed=FULL, prior=pri(narrow), **EKW)          # fresh model, used joint graph
c = jc().epoch_model(rows, epoch_starts=EP).fit(fixed=FULL, prior=pri(narrow), **EKW)       # fresh model, fresh joint graph
f = lambda x, y: float(np.max(np.abs(np.asarray(x.particles) - np.asarray(y.particles))))
h = lambda x, y: float(np.max(np.abs(np.asarray(x.history_log_prob) - np.asarray(y.history_log_prob))))
print(f'epoch: 3rd fit on used model vs fresh model/used jc: {f(a,b):.3e} (logp {h(a,b):.3e});  vs fresh model/FRESH jc: {f(a,c):.3e} (logp {h(a,c):.3e})')
import shutil; shutil.rmtree(os.environ['HOME'], ignore_errors=True)
