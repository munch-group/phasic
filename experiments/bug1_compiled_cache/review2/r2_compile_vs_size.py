"""Review 2: is JIT compile time graph-size dependent?  The FFI call carries
the serialized graph as a static string attribute, so the lowered module
contains it.  Time trace+lower and compile WITHOUT executing, for the
unbatched and the vmapped (8 particles) value_and_grad, against vertex count
and the size of the lowered module text."""
import sys, time, warnings, random, json
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import (coalescent_graph, coalescent_indexed,
    two_island_graph, sample_joint_observations)
from functools import partial
from phasic import Graph, GaussPrior, LogGaussPrior, SVGD
KW = dict(n_particles=8, n_iterations=1, learning_rate=0.01, progress=False, seed=1)

def probe(name, nv, s):
    out = []
    for rep in range(3):
        f = jax.value_and_grad(partial(s._log_prob_unified, nr_moments=s.nr_moments,
              sample_moments=s.sample_moments, regularization=s.regularization, rewards=s.rewards))
        th = s.theta_init
        t = time.time(); lo = jax.jit(f).lower(th[0]); tl1 = time.time() - t
        t = time.time(); lo.compile(); tc1 = time.time() - t
        t = time.time(); lo2 = jax.jit(jax.vmap(jax.jit(f))).lower(th); tl2 = time.time() - t
        t = time.time(); lo2.compile(); tc2 = time.time() - t
        out.append((tl1, tc1, tl2, tc2))
    try:
        mb = len(lo.as_text()) / 1e6
    except Exception as e:
        mb = float('nan')
    a = np.array(out)
    print(f'{name:34s} {nv:5d}v module {mb:7.2f} MB | single: lower {a[:,0].min():.2f}-{a[:,0].max():.2f}s compile {a[:,1].min():.2f}-{a[:,1].max():.2f}s'
          f' | vmap8: lower {a[:,2].min():.2f}-{a[:,2].max():.2f}s compile {a[:,3].min():.2f}-{a[:,3].max():.2f}s'
          f' | total(min) {a.sum(1).min():.2f}s', flush=True)

P1 = GaussPrior(ci=[0.5, 20], lower=0.0)
for n in (10, 22, 26):
    g = coalescent_graph(n); g.update_weights([7.0])
    random.seed(1); np.random.seed(1); d = np.asarray(g.sample(50))
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=1)
    probe(f'pdf+moments coalescent n={n}', g.vertices_length(), SVGD(model=m, observed_data=d, theta_dim=1, prior=P1, **KW))
for n in (4, 8, 12):
    g = two_island_graph(n); g.update_weights([0.7, 0.3])
    random.seed(1); np.random.seed(1); d = np.asarray(g.sample(50))
    m = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2)
    probe(f'pdf+moments two-island n={n}', g.vertices_length(), SVGD(model=m, observed_data=d, theta_dim=2, prior=GaussPrior(ci=[0.1, 5], lower=0.0), **KW))
for n in (4, 6):
    gi, idx = coalescent_indexed(n)
    J = gi.joint_prob_graph(idx, discrete=False, reward_limit=3)
    J.update_weights([2.0, 0.5])
    tab = J.joint_prob_table(); rows = tab[tab.columns[:-1]].to_numpy().tolist()[:40]
    em = J.epoch_model(rows, epoch_starts=[0.0, 0.3])
    pri = [GaussPrior(ci=[0.5, 20], lower=0.0), None] * 2
    s = SVGD(model=em.model, observed_data=em.observed_data, theta_dim=em.theta_dim, prior=pri,
             fixed=[(1, 0.5), (3, 0.5)], **KW)
    probe(f'epoch 2-epoch joint n={n}', J.vertices_length(), s)
