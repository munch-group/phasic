"""Review 3: minimal versions of plan tests 1-5 on the UNMODIFIED build."""
import random, warnings, sys
import numpy as np
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph
import jax, jax.numpy as jnp
print('phasic from', phasic.__file__)
def geom(p):
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, p); g.is_discrete = True; return g
def pgeom():
    g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, [1.0]); return g
def t(label, f):
    try: v = f(); print(f'    {label}: OK {v!r}'); return ('ok', v)
    except BaseException as e:
        print(f'    {label}: RAISES {type(e).__name__}: {str(e).splitlines()[0][:90]}'); return ('raise', e)
K = np.arange(1, 6)
READS = {
 'pdf': lambda g: np.asarray(g.pdf(K)),
 'pdf_discrete': lambda g: g.pdf_discrete(2),
 'cdf': lambda g: np.asarray(g.cdf(K)),
 'sample': lambda g: (random.seed(1), g.sample(5))[1],
 'sample_discrete?': lambda g: g.sample_discrete(5) if hasattr(g,'sample_discrete') else 'n/a',
 'sample_multivariate': lambda g: g.sample_multivariate(3, np.array([[0.],[1.],[0.]])) if hasattr(g,'sample_multivariate') else 'n/a',
 'rtd.pdf': lambda g: np.asarray(g.reward_transform_discrete(np.array([0,1,0])).pdf(K)),
 'expectation': lambda g: g.expectation(),
 'variance': lambda g: g.variance(),
 'moments': lambda g: g.moments(2),
 'moments(discrete=True)': lambda g: g.moments(2, discrete=True),
 'expected_visits?': lambda g: g.expected_visits() if hasattr(g,'expected_visits') else 'n/a',
 'expected_sojourn_time': lambda g: g.expected_sojourn_time(),
 'expected_waiting_time': lambda g: g.expected_waiting_time(),
 'stop_probability(2)': lambda g: g.stop_probability(2),
 'accumulated_visits? ': lambda g: g.accumulated_visits_discrete(2) if hasattr(g,'accumulated_visits_discrete') else 'n/a',
 'covariance_discrete?': lambda g: g.covariance_discrete(np.array([0.,1,0]),np.array([0.,1,0])) if hasattr(g,'covariance_discrete') else 'n/a',
 'expectation_discrete?': lambda g: g.expectation_discrete(np.array([0,1,0])) if hasattr(g,'expectation_discrete') else 'n/a',
 'variance_discrete?': lambda g: g.variance_discrete(np.array([0,1,0])) if hasattr(g,'variance_discrete') else 'n/a',
}
print('#### TEST 1/2/3: every read-out vs excess')
for ex in (0.0, 2.220446049250313e-16, 1e-15, 1e-12, 1e-9, 2e-9, 1e-6, 1e-5, 1e-4, 2e-4, 0.5):
    p = 1.0 + ex
    print(f'## p = 1 + {ex:.3g}   (p-1 as stored = {p-1:.3e})')
    for name, f in READS.items():
        t(name, lambda: f(geom(p)))
    g = geom(p)
    try:
        pm = np.asarray(g.pdf(np.arange(1, 400))); cd = np.asarray(g.cdf(np.arange(1, 400)))
        print(f'    SUMMARY min pmf={pm.min():.3e} sum-1={pm.sum()-1:.3e} max cdf-1={cd.max()-1:.3e} cdf monotone={bool(np.all(np.diff(cd)>=0))} var={g.variance():.3e}')
    except Exception as e: print('    SUMMARY raises')
print('#### TEST 5: parameterized paths, c*theta = 1+ex')
T = jnp.asarray([1., 2., 3.])
for ex in (0.0, 1e-12, 1e-9, 1e-5, 1e-4, 2e-4, 0.5):
    th = jnp.asarray([1.0 + ex])
    print(f'## theta = 1 + {ex:.3g}')
    for use_ffi in (False, True):
        t(f'pmf_from_graph(discrete, use_ffi={use_ffi})', lambda: np.asarray(Graph.pmf_from_graph(pgeom(), discrete=True, use_ffi=use_ffi)(th, T)))
        t(f'pmf_and_moments(discrete, use_ffi={use_ffi})', lambda: [np.asarray(x) for x in Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi=use_ffi)(th, T)])
        t(f'vmap pmf (use_ffi={use_ffi})', lambda: np.asarray(jax.vmap(lambda q: Graph.pmf_from_graph(pgeom(), discrete=True, use_ffi=use_ffi)(q, T))(jnp.stack([th, th]))))
        t(f'grad pmf_and_moments (use_ffi={use_ffi})', lambda: np.asarray(jax.grad(lambda q: Graph.pmf_and_moments_from_graph(pgeom(), nr_moments=2, discrete=True, use_ffi=use_ffi)(q, T)[1][1])(th)))
