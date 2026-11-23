# NOTE (after the fix): this script documents the UNFIXED code and uses
# SVGD._compiled_cache / removed helpers, which no longer exist; it does not
# run on builds that include the fix.
"""Impl review check 3: second fit after a first on the SAME model object vs the
same fit on a FRESH model object. Runs on either build (no cache manipulation).
HOME is redirected to an EMPTY directory (no .phasic_cache pre-created)."""
import os, sys, shutil, tempfile, warnings, random, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = tempfile.mkdtemp(prefix='fakehome_', dir=HERE)
os.environ['HOME'] = FAKE
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug1/tests/pytest')
import numpy as np
import jax.numpy as jnp
import phasic
from phasic import Graph, GaussPrior, SVGD
from phasic.svgd import dense_to_sparse, ExpRegularization
from test_epoch_normalization import normalised, p_n2_two_epoch
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
print('build:', phasic.__file__, ' has cache:', hasattr(SVGD, '_compiled_cache'))

KW = dict(n_particles=10, n_iterations=15, learning_rate=0.01, progress=False, seed=1)
wide = lambda: GaussPrior(ci=[0.5, 20], lower=0.0)
narrow = lambda: GaussPrior(ci=[0.01, 0.2], lower=0.0)

def data(theta, n=120, seed=3):
    random.seed(seed); np.random.seed(seed)
    g = coalescent_graph(4); g.update_weights([theta]); return np.asarray(g.sample(n))
DA, DB = data(8.0), data(1.0, seed=4)

def P(s):
    p = np.asarray(s.particles); assert np.all(np.isfinite(p)), 'non-finite'; return p

def check(name, mk, first, second):
    try:
        shared = mk()
        p1 = P(first(shared)); p2 = P(second(shared)); pf = P(second(mk()))
        d = float(np.max(np.abs(p2 - pf))); pw = float(np.max(np.abs(p1 - pf)))
        print(f'{name:46s} after-vs-fresh {d:.3e}  power(first-vs-fresh) {pw:.3e}  {"EQUAL" if d == 0 else "DIFFERENT"}', flush=True)
    except Exception as e:
        print(f'{name:46s} ERROR {type(e).__name__}: {str(e)[:300]}', flush=True)
        traceback.print_exc(limit=3)

def fit(m, d, prior, opt=None, **kw):
    s = SVGD(model=m, observed_data=d, prior=prior, **{**dict(theta_dim=1), **KW, **kw}); s.optimize(**(opt or {})); return s

mk1 = lambda: Graph.pmf_and_moments_from_graph(coalescent_graph(4), nr_moments=2, theta_dim=1)

# 1-D rewards
nv = coalescent_graph(4).vertices_length()
RW = jnp.asarray(np.linspace(0.5, 1.5, nv))
check('rewards 1-D, data A->B', mk1, lambda m: fit(m, DA, wide(), rewards=RW), lambda m: fit(m, DB, wide(), rewards=RW))
check('rewards 1-D + regularization', mk1, lambda m: fit(m, DA, wide(), rewards=RW, regularization=1.0),
      lambda m: fit(m, DB, wide(), rewards=RW, regularization=1.0))
# regularization schedule
check('regularization schedule, data A->B', mk1,
      lambda m: fit(m, DA, wide(), regularization=ExpRegularization(1.0, 0.0, 5.0)),
      lambda m: fit(m, DB, wide(), regularization=ExpRegularization(1.0, 0.0, 5.0)))
# parallel none, jit variants
check("parallel='none'", mk1, lambda m: fit(m, DA, wide(), parallel='none'), lambda m: fit(m, DB, wide(), parallel='none'))
check("parallel='none', jit=False", mk1, lambda m: fit(m, DA, wide(), parallel='none', jit=False, n_iterations=5),
      lambda m: fit(m, DB, wide(), parallel='none', jit=False, n_iterations=5))
check('jit=False precompile=False', mk1, lambda m: fit(m, DA, wide(), jit=False, precompile=False, n_iterations=5),
      lambda m: fit(m, DB, wide(), jit=False, precompile=False, n_iterations=5))
check('jit=False precompile=False no_history', mk1,
      lambda m: fit(m, DA, wide(), opt=dict(return_history=False), jit=False, precompile=False, n_iterations=5),
      lambda m: fit(m, DB, wide(), opt=dict(return_history=False), jit=False, precompile=False, n_iterations=5))
# exposure on direct SVGD
EXa = np.linspace(0.5, 2.0, len(DA)); EXb = EXa[::-1].copy()
check('direct exposure, exposure values differ', mk1,
      lambda m: fit(m, DA, wide(), exposure=EXa, exposure_param_index=0),
      lambda m: fit(m, DA, wide(), exposure=EXb, exposure_param_index=0))
# different transform (untruncated prior, positive_params False needs domain care -> skip); fixed values via SVGD
# multivariate dense + sparse
def chain():
    g = Graph(1); s = g.starting_vertex()
    v = [g.find_or_create_vertex([i + 1]) for i in range(3)]; va = g.find_or_create_vertex([9])
    s.add_edge(v[0], 1.0)
    for (a, b), c in zip([(v[0], v[1]), (v[1], v[2]), (v[2], va)], [(1.0, 0.5), (2.0, 0.25), (0.5, 1.5)]):
        a.add_edge(b, list(c))
    return g
RW2 = jnp.asarray([[1.0, 0.0, 2.0, 0.5, 1.0], [1.0, 1.0, 2.0, 1.0, 0.5]])
mkmv = lambda: Graph.pmf_and_moments_from_graph_multivariate(chain(), nr_moments=2, discrete=False, theta_dim=2)
rng = np.random.default_rng(0)
MA = rng.gamma(2.0, 1.0, size=(40, 2)); MB = rng.gamma(2.0, 0.2, size=(40, 2))
pr2 = lambda: [wide(), wide()]
check('multivariate dense 2-D rewards', mkmv, lambda m: fit(m, MA, pr2(), theta_dim=2, rewards=RW2),
      lambda m: fit(m, MB, pr2(), theta_dim=2, rewards=RW2))
mask = rng.random((40, 2)) < 0.3; mask[:, 1] &= ~mask[:, 0]
SA = MA.copy(); SA[mask] = np.nan; SB = MB.copy(); SB[mask] = np.nan
check('sparse observations (same mask), 2-D rewards', mkmv,
      lambda m: fit(m, dense_to_sparse(jnp.asarray(SA)), pr2(), theta_dim=2, rewards=RW2),
      lambda m: fit(m, dense_to_sparse(jnp.asarray(SB)), pr2(), theta_dim=2, rewards=RW2))
# sparse: same number of values, DIFFERENT mask (old key = (len(values), n_features))
mask2 = np.zeros_like(mask); idx = np.argwhere(~mask)[:int(mask.sum())]; mask2[idx[:, 0], idx[:, 1]] = True
SC = MA.copy(); SC[mask2] = np.nan
check('sparse observations, different mask same count', mkmv,
      lambda m: fit(m, dense_to_sparse(jnp.asarray(SA)), pr2(), theta_dim=2, rewards=RW2),
      lambda m: fit(m, dense_to_sparse(jnp.asarray(SC)), pr2(), theta_dim=2, rewards=RW2))
# two different models, equal shapes (id-recycling scenario on the old key)
def two_models():
    m = mk1(); fit(m, DA, wide()); del m
    import gc; gc.collect()
    return fit(mk1(), DB, wide())
try:
    a = P(fit(mk1(), DB, wide())); b = P(two_models())
    print(f'{"different model objects, equal shapes":46s} {float(np.max(np.abs(a-b))):.3e}', flush=True)
except Exception as e:
    print('different models ERROR', e)

# epoch fits
L = 3; EP = [0.0, 0.4]
g, idx = coalescent_indexed(2)
jc = g.joint_prob_graph(idx, discrete=False, reward_limit=L)
p = normalised([p_n2_two_epoch(k, 2.0, 1.0, 6.0, 1.0, EP[1]) for k in range(L + 1)])
rows = [[k, 0] for k in range(L + 1) for _ in range(int(round(p[k] * 200)))]
rowsB = rows[::-1][:len(rows)]
p2 = normalised([p_n2_two_epoch(k, 6.0, 1.0, 1.0, 1.0, EP[1]) for k in range(L + 1)])
rows2 = [[k, 0] for k in range(L + 1) for _ in range(int(round(p2[k] * 200)))]
EKW = dict(n_particles=10, n_iterations=15, learning_rate=0.01, progress=False, seed=1)
FULL = [(1, 1.0), (3, 1.0)]
pri4 = lambda mk=wide: [mk(), None, mk(), None]
ex = np.tile([0.5, 1.0, 2.0], len(rows) // 3 + 1)[:len(rows)]
mkex = lambda: jc.epoch_model(rows, epoch_starts=EP, exposure=ex, exposure_param_index=1)
check('epoch + exposure, prior P1->P2', mkex, lambda em: em.fit(fixed=FULL, prior=pri4(wide), **EKW),
      lambda em: em.fit(fixed=FULL, prior=pri4(narrow), **EKW))
check('epoch + exposure, non-nested fixed', mkex,
      lambda em: em.fit(fixed=[(1, 1.0)], prior=[wide() for _ in range(4)], **EKW),
      lambda em: em.fit(fixed=[(3, 1.0)], prior=[wide() for _ in range(4)], **EKW))
check('epoch + exposure, default prior, fixed values', mkex,
      lambda em: em.fit(fixed=[(1, 1.0), (3, 1.0)], **EKW), lambda em: em.fit(fixed=[(1, 3.0), (3, 3.0)], **EKW))
check('epoch single epoch_starts=[0.0], prior P1->P2', lambda: jc.epoch_model(rows, epoch_starts=[0.0]),
      lambda em: em.fit(fixed=[(1, 1.0)], prior=[wide(), None], **EKW),
      lambda em: em.fit(fixed=[(1, 1.0)], prior=[narrow(), None], **EKW))
# Graph.svgd tied epochs / exposure epochs on ONE graph object, different data
def gs(graph, r, **kw):
    return graph.svgd(r, epoch_starts=EP, fixed=[(1, 1.0)], prior=[wide(), None], **{**EKW, **kw})
class G:  # "model" = the joint graph itself
    pass
mkjc = lambda: coalescent_indexed(2)[0].joint_prob_graph(coalescent_indexed(2)[1], discrete=False, reward_limit=L)
def mkjc():
    gg, ii = coalescent_indexed(2); return gg.joint_prob_graph(ii, discrete=False, reward_limit=L)
check('Graph.svgd tied epochs, data 1->2', mkjc, lambda G_: gs(G_, rows, tied=[(0, [0, 1])]), lambda G_: gs(G_, rows2, tied=[(0, [0, 1])]))
check('Graph.svgd tied epochs + exposure, data 1->2', mkjc,
      lambda G_: gs(G_, rows, tied=[(0, [0, 1])], exposure=ex, exposure_param_index=1),
      lambda G_: gs(G_, rows2, tied=[(0, [0, 1])], exposure=ex[:len(rows2)] if len(rows2) <= len(ex) else np.resize(ex, len(rows2)), exposure_param_index=1))
check('Graph.svgd joint (no epochs), data 1->2', mkjc,
      lambda G_: G_.svgd(rows, fixed=[(1, 1.0)], prior=[wide(), None], **EKW),
      lambda G_: G_.svgd(rows2, fixed=[(1, 1.0)], prior=[wide(), None], **EKW))
print('fake HOME contents:', sorted(os.path.relpath(os.path.join(dp, f), FAKE) for dp, _, fs in os.walk(FAKE) for f in fs)[:20])
print('compiled_svgd files:', [f for dp, _, fs in os.walk(FAKE) for f in fs if 'compiled_svgd' in f])
shutil.rmtree(FAKE, ignore_errors=True)
