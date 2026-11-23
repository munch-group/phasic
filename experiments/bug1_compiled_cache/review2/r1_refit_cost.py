"""Review 2: what does a REFIT on the same model object cost once the cache is
gone?  Protocol per path: s0 = first fit (cold everything); then
  recompile = [clear cache; new SVGD on s0.model; optimize(1 iter)]  x3
  hit       = [same, cache NOT cleared]                               x3
refit cost of the fix = recompile - hit.  Also decompose the recompile into
trace+compile of the unbatched warm-up call, one execution, and the vmapped
compile."""
import sys, time, warnings, random, traceback
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import (coalescent_graph, coalescent_indexed,
    two_island_graph, sample_joint_observations)
from functools import partial
from phasic import Graph, GaussPrior, SVGD
ONLY = sys.argv[1:]  # optional path-name filters
KW = dict(n_particles=8, n_iterations=1, learning_rate=0.01, progress=False, seed=1)

def refit(s0, **over):
    kw = dict(KW); kw.update(over)
    s = SVGD(model=s0.model, observed_data=s0.observed_data, theta_dim=s0.theta_dim,
             prior=s0.prior, fixed=s0.fixed, rewards=s0.rewards, **kw)
    t = time.time(); s.optimize(); return time.time() - t, s

def measure(name, nv, mk, **over):
    if ONLY and not any(o in name for o in ONLY): return
    try:
        t = time.time(); s0 = mk(); cold = time.time() - t
        rec, hit = [], []
        for _ in range(3):
            SVGD._compiled_cache.clear(); rec.append(refit(s0, **over)[0])
        for _ in range(3):
            hit.append(refit(s0, **over)[0])
        # decomposition on a fresh instance
        SVGD._compiled_cache.clear()
        kw = dict(KW); kw.update(over)
        s = SVGD(model=s0.model, observed_data=s0.observed_data, theta_dim=s0.theta_dim,
                 prior=s0.prior, fixed=s0.fixed, rewards=s0.rewards, **kw)
        f = jax.jit(jax.value_and_grad(partial(s._log_prob_unified, nr_moments=s.nr_moments,
                    sample_moments=s.sample_moments, regularization=s.regularization, rewards=s.rewards)))
        th = s.theta_init
        t = time.time(); jax.block_until_ready(f(th[0])); t1 = time.time() - t
        t = time.time(); jax.block_until_ready(f(th[0])); tx = time.time() - t
        dec = f'warmup(single) {t1:.2f}s exec1 {tx:.3f}s'
        if s.parallel_mode == 'vmap':
            t = time.time(); jax.block_until_ready(jax.vmap(f)(th)); t2 = time.time() - t
            t = time.time(); jax.block_until_ready(jax.vmap(f)(th)); tx2 = time.time() - t
            dec += f' vmap-first {t2:.2f}s vmap-exec {tx2:.3f}s'
        r = lambda v: f'{min(v):.2f}-{max(v):.2f}'
        print(f'{name:38s} {nv:5d}v P={s0.theta_dim} par={s.parallel_mode:5s} cold {cold:6.2f}s | recompile {r(rec)}s | hit {r(hit)}s | '
              f'refit cost {min(rec)-min(hit):.2f}..{max(rec)-min(hit):.2f}s | {dec}', flush=True)
    except Exception as e:
        print(f'{name}: FAILED {type(e).__name__}: {str(e)[:300]}', flush=True)
        traceback.print_exc(limit=3)

P1 = GaussPrior(ci=[0.5, 20], lower=0.0)
# A. pdf + moments path, coalescent, three sizes
for n in (10, 22, 26):
    g = coalescent_graph(n); g.update_weights([7.0])
    random.seed(1); np.random.seed(1); d = np.asarray(g.sample(300))
    measure(f'A pdf+moments coalescent n={n}', g.vertices_length(),
            lambda: g.svgd(observed_data=d, prior=P1, **KW))
# A2. two-island (cyclic) two sizes
for n in (4, 8):
    g = two_island_graph(n); g.update_weights([0.7, 0.3])
    random.seed(1); np.random.seed(1); d = np.asarray(g.sample(300))
    measure(f'A2 pdf+moments two-island n={n}', g.vertices_length(),
            lambda: g.svgd(observed_data=d, prior=GaussPrior(ci=[0.1, 5], lower=0.0), **KW))
# B. 1-D rewards
g = coalescent_graph(10); g.update_weights([10.0])
rw = np.sum(g.states().T, axis=0)
random.seed(1); np.random.seed(1); d = g.sample(300, rewards=rw)
measure('B rewards 1-D coalescent n=10', g.vertices_length(),
        lambda: g.svgd(observed_data=d, rewards=rw, prior=P1, **KW))
# C. 2-D rewards (multivariate)
try:
    rw2 = np.asarray(g.states().T, dtype=float)[:4]
    random.seed(1); np.random.seed(1)
    d2 = np.column_stack([np.asarray(g.sample(200, rewards=rw2[j])) for j in range(rw2.shape[0])])
    measure('C rewards 2-D (4 features) n=10', g.vertices_length(),
            lambda: g.svgd(observed_data=d2, rewards=rw2, prior=P1, **KW))
except Exception as e:
    print('C setup failed', type(e).__name__, str(e)[:300])
# D. joint-index, non-epoch
for n, rl in ((4, 3), (6, 3)):
    gi, idx = coalescent_indexed(n)
    J = gi.joint_prob_graph(idx, reward_limit=rl, mutation_rate=1e-2)
    obs = sample_joint_observations(J, [1e-4, 1e-2], 300, np.random.default_rng(17))
    from phasic import LogGaussPrior
    measure(f'D joint-index (discrete) n={n}', J.vertices_length(),
            lambda: J.svgd(obs, fixed=[(1, 1e-2)], prior=LogGaussPrior(ci=[2e-5, 6e-5]), **KW))
# E. epoch model: plain, and with many unique exposures
for n in (4, 6):
    gi, idx = coalescent_indexed(n)
    J = gi.joint_prob_graph(idx, discrete=False, reward_limit=3)
    J.update_weights([2.0, 0.5])
    tab = J.joint_prob_table(); rows = tab[tab.columns[:-1]].to_numpy().tolist()[:40]
    pri = [GaussPrior(ci=[0.5, 20], lower=0.0), None] * 2
    FX = [(1, 0.5), (3, 0.5)]
    def mk_plain():
        em = J.epoch_model(rows, epoch_starts=[0.0, 0.3]); return em.fit(fixed=FX, prior=pri, **KW)
    measure(f'E epoch 2-epoch n={n}', J.vertices_length(), mk_plain)
    expo = np.linspace(0.5, 2.0, len(rows))
    def mk_expo():
        em = J.epoch_model(rows, epoch_starts=[0.0, 0.3], exposure=expo, exposure_param_index=1)
        return em.fit(fixed=FX, prior=pri, **KW)
    measure(f'E2 epoch + 40 unique exposures n={n}', J.vertices_length(), mk_expo)
# F. parallel modes on the small pdf model
g = coalescent_graph(10); g.update_weights([7.0])
random.seed(1); np.random.seed(1); d = np.asarray(g.sample(300))
for par in ('none', 'pmap'):
    measure(f'F pdf+moments n=10 parallel={par}', g.vertices_length(),
            lambda: g.svgd(observed_data=d, prior=P1, parallel=par, **KW), parallel=par)
