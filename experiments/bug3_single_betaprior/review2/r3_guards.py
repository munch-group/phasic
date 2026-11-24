"""Guards + targeted attacks.  Run twice: patched, and PROTO_OFF=1 (control);
lines starting 'G ' must be identical between the two runs."""
import sys, os, warnings, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np, jax, jax.numpy as jnp, random
import phasic
from phasic import Graph, GaussPrior, LogGaussPrior, GammaPrior, HalfCauchyPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
import phasic.svgd as S
OFF = bool(os.environ.get('PROTO_OFF'))
print('PROTO_OFF =', OFF)
def h(a): return hashlib.md5(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:12]
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy(dim):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.exp(-times * jnp.sum(theta)) * jnp.sum(theta), jnp.zeros(2) + jnp.sum(theta)
    return spy, seen
def graph2():
    g = Graph(1); v0 = g.starting_vertex(); v1 = g.find_or_create_vertex([1]); v2 = g.find_or_create_vertex([2])
    v0.add_edge(v1, [1.0, 0.0]); v1.add_edge(v2, [0.0, 1.0]); g.update_weights(np.array([1.0, 0.5])); return g
obs = jnp.array([1.5, 2.0, 2.5, 3.0, 1.0, 0.7, 4.0, 2.2])

print('## (a) guards: non-natural priors must be bit-identical patched vs control')
for label, mk, dim, kw in [
    ('single Gauss', lambda: GaussPrior(ci=[0.5, 5], lower=0.0), 2, {}),
    ('single LogGauss', lambda: LogGaussPrior(ci=[0.5, 5]), 2, {}),
    ('[Gamma, HalfCauchy]', lambda: [GammaPrior(ci=[0.5, 5]), HalfCauchyPrior(scale=1.0)], 2, {}),
    ('[None, Gauss] fixed', lambda: [None, GaussPrior(ci=[0.5, 5], lower=0.0)], 2, dict(fixed=[(0, 2.0)])),
    ('no prior', lambda: None, 2, {}),
    ('custom exp, Gauss(None)', lambda: GaussPrior(ci=[0.5, 5]), 1, dict(positive_params=False, param_transform=jnp.exp)),
    ('no transform, Gauss(None)', lambda: GaussPrior(ci=[0.5, 5]), 1, dict(positive_params=False)),
    ('[Beta] list (master already per-dim)', lambda: [BetaPrior(ci=(0.05, 0.95))], 1, {}),
    ('[Gauss, Beta] list', lambda: [GaussPrior(ci=[0.5, 5], lower=0.0), BetaPrior(ci=(0.05, 0.95))], 2, {}),
]:
    try:
        spy, seen = make_spy(dim)
        s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=mk(), n_particles=8, n_iterations=5,
                 progress=False, seed=1, learning_rate=0.05, verbose=False, **kw); s.optimize()
        print(f'G SVGD {label}: particles {h(s.particles)} results {h(s.get_results()["particles"])}')
    except Exception as e:
        print(f'G SVGD {label}: {type(e).__name__}: {str(e)[:120]}')
    try:
        spy, seen = make_spy(dim)
        m = MCMC(model=spy, observed_data=d, theta_dim=dim, prior=mk(), n_samples=60, n_chains=2, burn_in=20,
                 verbose=False, seed=1, **kw); m.run()
        tag = 'G' if 'Beta' not in label else 'X(expected to differ)'
        print(f'{tag} MCMC {label}: chains {h(m._chains_phi)} results {h(m.get_results()["particles"])}')
    except Exception as e:
        print(f'G MCMC {label}: {type(e).__name__}: {str(e)[:120]}')
for label, call in [
    ('Graph.svgd default DataPrior', lambda g: g.svgd(observed_data=obs, theta_dim=2, n_iterations=5, n_particles=8, progress=False, seed=3, verbose=False)),
    ('Graph.mcmc default DataPrior', lambda g: g.mcmc(observed_data=obs, theta_dim=2, n_samples=60, n_chains=2, burn_in=20, seed=3, verbose=False)),
]:
    try:
        random.seed(0); np.random.seed(0)
        r = call(graph2()); res = r.get_results()
        print(f'G {label}: {h(res["particles"])} prior_list types {[type(p).__name__ for p in (r.prior_list or [])]}')
    except Exception as e:
        print(f'G {label}: {type(e).__name__}: {str(e)[:200]}')

print('## (b) plan test 2: single BetaPrior vs list form, same seed -> particles array-equal?')
for dim in (1, 2):
    out = {}
    for form in ('single', 'list'):
        for init in ('sampled', 'given'):
            spy, seen = make_spy(dim)
            pr = BetaPrior(ci=(0.05, 0.95)) if form == 'single' else [BetaPrior(ci=(0.05, 0.95)) for _ in range(dim)]
            kw = {} if init == 'sampled' else dict(theta_init=np.linspace(-1, 1, 8 * dim).reshape(8, dim))
            try:
                s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=pr, n_particles=8, n_iterations=5,
                         progress=False, seed=1, learning_rate=0.05, verbose=False, **kw); s.optimize()
                out[(form, init)] = (np.asarray(s.theta_init), np.asarray(s.particles))
            except Exception as e:
                print(f'  dim {dim} {form} {init}: {type(e).__name__}: {str(e)[:120]}')
    for init in ('sampled', 'given'):
        if ('single', init) in out and ('list', init) in out:
            a, b = out[('single', init)], out[('list', init)]
            print(f'  dim {dim}, theta_init {init}: init equal={np.array_equal(a[0], b[0])}; particles equal={np.array_equal(a[1], b[1])}; max|diff|={np.abs(a[1] - b[1]).max():.3g}')

print('## (c) plot_pairwise_hdr with sigmoid slots: does the model get the grid theta?')
for label, mk in [('single Beta dim2', lambda: BetaPrior(ci=(0.05, 0.95))), ('[Beta, Beta]', lambda: [BetaPrior(ci=(0.05, 0.95)), BetaPrior(ci=(0.05, 0.95))]),
                  ('single Gauss dim2 (control)', lambda: GaussPrior(ci=[0.5, 5], lower=0.0))]:
    try:
        spy, seen = make_spy(2)
        s = SVGD(model=spy, observed_data=d, theta_dim=2, prior=mk(), n_particles=8, n_iterations=5,
                 progress=False, seed=1, learning_rate=0.05, verbose=False); s.optimize()
        P = np.asarray(s.get_results()['particles']); del seen[:]
        s.plot_pairwise_hdr(center_map=False, n=6, xlim=(0.2, 0.8), ylim=(0.2, 0.8))
        a = np.concatenate(seen)
        print(f'  {label}: requested display grid within [0.2, 0.8] (+pad); model evaluated at theta in '
              f'({a.min(axis=0).round(4).tolist()}, {a.max(axis=0).round(4).tolist()})')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:200]}')

print('## (d) prior._transform is stale across fits (transform -> no transform)')
p = GaussPrior(ci=[0.5, 5])
spy, _ = make_spy(1)
SVGD(model=spy, observed_data=d, theta_dim=1, prior=p, n_particles=4, n_iterations=1, progress=False, seed=1, verbose=False, positive_params=False, param_transform=jnp.exp)
spy, _ = make_spy(1)
SVGD(model=spy, observed_data=d, theta_dim=1, prior=p, n_particles=4, n_iterations=1, progress=False, seed=1, verbose=False, positive_params=False)
print('  after fit with param_transform=exp then fit with NO transform: prior._transform is', 'None (reset)' if p._transform is None else 'STILL SET (stale)')

print('## (e) BetaPrior density/sampler are sigmoid-specific regardless of the _transform handed in')
b = BetaPrior(alpha=2.0, beta=5.0); phi = jnp.array([0.3])
b._transform = lambda x: jnp.maximum(jax.nn.softplus(x), 1e-9)
th = float(b._transform(phi)[0]); true = (2 - 1) * np.log(th) + (5 - 1) * np.log1p(-th) + np.log(float(jax.nn.sigmoid(phi[0])))
print(f'  with _transform=softplus: prior(phi)={float(b(phi)):.6f}; correct Beta log-density + softplus log-Jacobian={true:.6f}')
