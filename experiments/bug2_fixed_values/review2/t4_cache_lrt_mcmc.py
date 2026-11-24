"""(a) compiled-gradient cache x wrapper: same model object, two fits with different fixed values
(b) model_selection._check_nested_fixed reads fixed_values -> placeholder defeats it
(c) MCMC with the wrapper"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug2/experiments/bug2_fixed_values/review2')
import numpy as np, jax, jax.numpy as jnp
import proto
from phasic import Graph, GaussPrior, BetaPrior, SVGD, MCMC
from phasic import model_selection as ms
d = np.array([0.3, 0.5, 1.0, 1.5, 2.0, 0.8, 1.2, 0.6])
base = Graph.pmf_and_moments_from_graph(proto.graph3(), nr_moments=2, theta_dim=2)
def mk(model, v, patched, **kw):
    if patched: proto.install('plan')
    try:
        s = SVGD(model=model, observed_data=d, theta_dim=2, fixed=[(1, v)], n_particles=6, n_iterations=3,
                 progress=False, seed=1, prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None], **kw)
        s.optimize(); return s
    finally: proto.uninstall()
print('## (a) SAME model object, fit fixed=0.3 then fixed=2.0 (no cache clear between)')
for patched in (False, True):
    proto.clear()
    spy, seen = proto.spy_model(base)
    mk(spy, 0.3, patched); n0 = len(seen)
    s2 = mk(spy, 2.0, patched)
    print(f'   patched={patched}: second fit model saw slot1 = {proto.slot_values(seen[n0:], 1)}; reported mean = {float(np.asarray(s2.get_results()["theta_mean"])[1])}')
print('## (b) LRT nested check: full fixed (1,0.3); nested claims fixed (1,2.0)+... same index different value')
def three():
    g = Graph(1); s = g.starting_vertex()
    a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2]); c = g.find_or_create_vertex([3])
    s.add_edge(a, [0.0, 0.0, 1.0]); a.add_edge(c, [1.0, 0.0, 0.0]); a.add_edge(b, [0.0, 1.0, 0.0]); b.add_edge(c, [1.0, 0.0, 0.0]); return g
base3 = Graph.pmf_and_moments_from_graph(three(), nr_moments=2, theta_dim=3)
for patched in (False, True):
    proto.clear()
    if patched: proto.install('plan')
    try:
        P = lambda: GaussPrior(ci=[0.5, 5], lower=0.0)
        full = SVGD(model=base3, observed_data=d, theta_dim=3, fixed=[(1, 0.3)], n_particles=6, n_iterations=2, progress=False, seed=1, prior=[P(), None, P()])
        nested = SVGD(model=base3, observed_data=d, theta_dim=3, fixed=[(1, 2.0), (2, 1.0)], n_particles=6, n_iterations=2, progress=False, seed=1, prior=[P(), None, None])
        try:
            ms._check_nested_fixed(full, nested, True); print(f'   patched={patched}: check PASSED (values 0.3 vs 2.0 at shared index NOT detected)')
        except Exception as e: print(f'   patched={patched}: check raised: {str(e)[:140]}')
    finally: proto.uninstall()
print('## (c) MCMC')
def mc(label, v, mode='plan', **kw):
    proto.install(mode); proto.clear()
    spy, seen = proto.spy_model(base)
    kw.setdefault('prior', [GaussPrior(ci=[0.5, 5], lower=0.0 if kw.get('positive_params', True) else None), None])
    try:
        for par in ('vmap', None):
            try:
                m = MCMC(model=spy, observed_data=d, theta_dim=2, fixed=[(1, v)], n_chains=2, n_samples=30, burn_in=20, progress=False, verbose=False, seed=1, parallel=par, **kw)
                m.run()
                r = m.get_results()
                print(f'   [{mode}] {label} parallel={par}: saw={proto.slot_values(seen, 1)[:3]} mean={float(np.asarray(r["theta_mean"])[1])!r} std={float(np.asarray(r["theta_std"])[1])!r} hpd=({r["hpd_lower"][1]!r},{r["hpd_upper"][1]!r}) free_finite={bool(np.all(np.isfinite(np.asarray(r["particles"])[:,0])))} shapes={sorted(set(proto.CALLS))}')
                seen.clear()
            except Exception as e:
                print(f'   [{mode}] {label} parallel={par}: RAISED {type(e).__name__}: {str(e)[:200]}')
    finally: proto.uninstall()
for v in (0.0, 1e-12, 0.3, 800.0): mc(f'fixed={v:g}', v)
mc('strict fixed=0.3', 0.3, mode='strict')
mc('exp fixed=2', 2.0, positive_params=False, param_transform=lambda p: jnp.exp(p), prior=[GaussPrior(ci=[0.5, 5]), None])
