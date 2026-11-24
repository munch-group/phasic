"""Review 1 (diagnosis) part b: Graph.svgd routes, epoch broadcast, exposure, fixed, hdr plot. Unmodified code."""
import warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import phasic
from phasic import Graph, GaussPrior, BetaPrior, LogGaussPrior, SVGD, StateIndexer, Property, with_ipv, Adamelia, set_log_level
from phasic.svgd import _sigmoid
from itertools import combinations_with_replacement
from functools import partial
set_log_level('WARNING')
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy(dim):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(jnp.asarray(times, dtype=float)), jnp.zeros(2)
    return spy, seen
def beta(): return BetaPrior(ci=(0.05, 0.95))
def kind(fn, x=5.0):
    v = float(np.asarray(fn(jnp.array([x]))).ravel()[0])
    return 'sigmoid' if abs(v - 0.99330715) < 1e-6 else ('softplus' if abs(v - 5.0067153) < 1e-6 else f'other({v:.4f})')
def slots(s):
    return [kind(lambda p, i=i: s.param_transform(jnp.full((s.theta_dim,), float(p[0])))[i:i+1]) for i in range(s.theta_dim)]

def two_param_graph():
    g = Graph(1); v0 = g.starting_vertex(); v1 = g.find_or_create_vertex([1]); v2 = g.find_or_create_vertex([2])
    v0.add_edge(v1, [1.0, 0.0]); v1.add_edge(v2, [0.0, 1.0]); return g

print('## G1 Graph.svgd (non-epoch), theta_dim 2')
obs = jnp.array([1.5, 2.0, 2.5, 3.0])
for label, pr, kw in [
    ('single BetaPrior', beta(), {}),
    ('[Beta, Beta]', [beta(), beta()], {}),
    ('single BetaPrior + fixed=[(0, 5.0)]', beta(), dict(fixed=[(0, 5.0)])),
    ('[None, Beta] + fixed=[(0, 5.0)]', [None, beta()], dict(fixed=[(0, 5.0)])),
]:
    try:
        g = two_param_graph(); g.update_weights(np.array([1.0, 0.5]))
        s = g.svgd(observed_data=obs, prior=pr, theta_dim=2, n_iterations=3, n_particles=8, progress=False, **kw)
        P = np.asarray(s.get_results()['particles'])
        print(f'  {label}: model transform per slot {slots(s)}; prior._transform {[kind(q._transform) for q in (pr if isinstance(pr, list) else [pr]) if q is not None]}; '
              f'constrained particles max {P.max(axis=0).round(4).tolist()}')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:300]}')

print('\n## G2 SVGD-level exposure on a BetaPrior slot (spy sees per-observation theta)')
for label, pr in [('[Gauss, Beta], exposure on slot 1', [GaussPrior(ci=[0.5, 5], lower=0.0), beta()])]:
    try:
        spy, seen = make_spy(2)
        s = SVGD(model=spy, observed_data=d, theta_dim=2, prior=pr, n_particles=8, n_iterations=3, progress=False, seed=1,
                 learning_rate=0.05, exposure=np.array([1.0, 10.0, 100.0, 1000.0]), exposure_param_index=1, preconditioner='none')
        s.optimize()
        allv = np.concatenate([a.reshape(-1, 2) for a in seen])
        print(f'  {label}: model saw slot-1 theta in [{allv[:,1].min():.4g}, {allv[:,1].max():.4g}] (a "probability" times alpha); slots {slots(s)}')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:300]}')

print('\n## G3 plot_pairwise_hdr on [Gauss, Beta]: which theta is the model evaluated at for the Beta axis?')
try:
    spy, seen = make_spy(2)
    s = SVGD(model=spy, observed_data=d, theta_dim=2, prior=[GaussPrior(ci=[0.5, 5], lower=0.0), beta()], n_particles=20,
             n_iterations=5, progress=False, seed=1, learning_rate=0.05, preconditioner='none')
    s.optimize(); P = np.asarray(s.get_results()['particles'])
    n0 = len(seen)
    ax = s.plot_pairwise_hdr(return_fig=True, center_map=False, ylim=(0.2, 0.8), xlim=(1.0, 3.0), n=6, pad=0)
    plot_v = np.concatenate([a.reshape(-1, 2) for a in seen[n0:]])
    print(f'  requested Beta-axis range (0.2, 0.8); model evaluated Beta slot in [{plot_v[:,1].min():.4f}, {plot_v[:,1].max():.4f}]; '
          f'sigmoid(inv_softplus(0.2))={float(_sigmoid(jnp.log(jnp.expm1(0.2)))):.4f}, sigmoid(inv_softplus(0.8))={float(_sigmoid(jnp.log(jnp.expm1(0.8)))):.4f}; '
          f'Gauss slot range [{plot_v[:,0].min():.4f}, {plot_v[:,0].max():.4f}] (requested 1..3)')
except Exception as e:
    import traceback; traceback.print_exc()

print('\n## G4 epoch path: a SINGLE BetaPrior is broadcast to a per-slot list (so it already gets sigmoid there)')
try:
    np.random.seed(17); import random; random.seed(17)
    all_pairs = partial(combinations_with_replacement, r=2)
    nr = 4
    indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])
    @with_ipv([nr] + [0] * (nr - 1))
    def coal(state, indexer=None):
        tr = []
        for i, j in all_pairs(range(indexer.lineages.state_length)):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy(); new[i] -= 1; new[j] -= 1; new[min(i + j + 1, state.size - 1)] += 1
            tr.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return tr
    graph = Graph(coal, indexer=indexer)
    mu = 0.3; th = 0.5
    jd = graph.joint_prob_graph(indexer, reward_limit=3, mutation_rate=mu); jd.update_weights([th, mu])
    jpt = jd.joint_prob_table(); p = jpt["prob"] / jpt["prob"].sum()
    sample = np.random.choice(jpt.index.values, 300, p=p.to_numpy())
    observations = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    for label, pr, kw in [
        ('single BetaPrior, 2 epochs', beta(), {}),
        ('single BetaPrior, 2 epochs, tied', beta(), dict(tied=[(0, [0, 1])])),
        ('single LogGaussPrior, 2 epochs (control)', LogGaussPrior(ci=[0.1, 2]), {}),
    ]:
        try:
            jc = graph.joint_prob_graph(indexer, reward_limit=3, mutation_rate=mu, discrete=False)
            s = jc.svgd(observations, fixed=[(1, mu)], prior=pr, n_iterations=2, n_particles=6, optimizer=Adamelia(learning_rate=0.05),
                        epoch_starts=[0, 0.5], progress=False, **kw)
            P = np.asarray(s.get_results()['particles'])
            print(f'  {label}: SVGD.prior_list types {[type(q).__name__ if q is not None else None for q in (s.prior_list or [])]}; model transform per slot {slots(s)}; '
                  f'constrained particle max {P.max(axis=0).round(4).tolist()}')
        except Exception as e:
            print(f'  {label}: {type(e).__name__}: {str(e)[:300]}')
except Exception as e:
    import traceback; traceback.print_exc()
