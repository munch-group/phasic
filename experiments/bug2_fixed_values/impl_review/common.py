import sys, os, random, traceback, warnings
sys.path.insert(0, '/Users/kmt/phasic-bug2/tests/pytest')
import numpy as np, jax, jax.numpy as jnp
import phasic
from phasic import Graph, GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
import phasic.svgd as svgd_mod, phasic.mcmc as mcmc_mod
print("phasic from", phasic.__file__)
SEEN = []
def wrap(model):
    def spy(theta, *a, **k):
        jax.debug.callback(lambda t: SEEN.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return model(theta, *a, **k)
    spy.__dict__.update(getattr(model, '__dict__', {}))
    return spy
_orig_s = SVGD.__init__
def _s_init(self, *a, **k):
    if 'model' in k and k['model'] is not None and not os.environ.get('NOSPY'):
        k['model'] = wrap(k['model'])
    elif a and not os.environ.get('NOSPY'):
        a = (wrap(a[0]),) + a[1:]
    return _orig_s(self, *a, **k)
SVGD.__init__ = _s_init
_orig_m = MCMC.__init__
def _m_init(self, *a, **k):
    if k.get('model') is not None and not os.environ.get('NOSPY'):
        k['model'] = wrap(k['model'])
    return _orig_m(self, *a, **k)
MCMC.__init__ = _m_init
def col(slot):
    if not SEEN: return np.array([])
    return np.unique(np.concatenate([s[:, slot] for s in SEEN]))
def scen(name, fn):
    SEEN.clear()
    print(f"\n=== {name}")
    try:
        fn()
    except Exception as e:
        print(f"  RAISED {type(e).__name__}: {str(e)[:400]}")
def gauss(): return GaussPrior(ci=[0.5, 5], lower=0.0)
def chain2():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, 1.0); a.add_edge(c, [1.0, 0.0]); a.add_edge(b, [0.0, 1.0]); b.add_edge(c, [1.0, 0.0])
    return g
REAL = np.array([0.2, 0.5, 0.9, 1.4, 0.3, 0.7, 1.1, 0.6])
def report(s, slots, n=None):
    r = s.get_results()
    P = np.asarray(r['particles'])
    for sl in slots:
        print(f"  slot {sl}: seen={col(sl)[:6]} particles_uniq={np.unique(P[:, sl])[:4]} "
              f"mean={float(np.asarray(r['theta_mean'])[sl])!r} attr_mean={float(np.asarray(s.theta_mean)[sl])!r} "
              f"std={float(np.asarray(r['theta_std'])[sl])!r} hpd=({r['hpd_lower'][sl]!r},{r['hpd_upper'][sl]!r})")
    print("  all finite:", bool(np.all(np.isfinite(P))), " n model calls:", len(SEEN))
