from common import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from phasic.svgd import ExpRegularization, ConstantRegularization
KW = dict(n_particles=6, n_iterations=4, learning_rate=0.01, progress=False, seed=1)
def mk(): return Graph.pmf_and_moments_from_graph(chain2(), nr_moments=2, theta_dim=2)
V = 1e-10
def fit(**kw):
    k = dict(model=mk(), observed_data=REAL, theta_dim=2, prior=[gauss(), None], fixed=[(1, V)]); k.update(KW); k.update(kw)
    s = SVGD(**k); return s
def b1():
    s = fit(regularization=1.0); s.optimize(); report(s,[1])
scen("B1 regularization=1.0", b1)
def b2():
    s = fit(regularization=ExpRegularization(first_reg=2.0, last_reg=0.1, tau=2.0)); s.optimize(); report(s,[1])
scen("B2 regularization schedule", b2)
for pc in ['jacobian','fisher','none', None]:
    def b3(pc=pc):
        s = fit(preconditioner=pc); s.optimize(); report(s,[1])
    scen(f"B3 preconditioner={pc!r}", b3)
def b3b():
    s = fit(preconditioner='jacobian', prior=[BetaPrior(ci=(0.05,0.95)), None]); s.optimize(); report(s,[1])
scen("B3b jacobian + beta composite", b3b)
def b3c():
    s = fit(preconditioner='fisher', positive_params=False, param_transform=lambda p: jnp.exp(p), prior=[GaussPrior(ci=[0.5,5]), None]); s.optimize(); report(s,[1])
scen("B3c fisher + custom exp transform", b3c)
def b4():
    s = fit(); s.optimize(return_history=False); report(s,[1]); print("  history:", s.history if not hasattr(s.history,'shape') else s.history.shape)
scen("B4 return_history=False", b4)
def b5():
    s = fit(n_particles=40); s.optimize()
    report(s,[1])
    print("  MAP", s.map_estimate_from_particles(), "MAP unc", s.map_estimate_from_particles(unconstrained=True))
    print("  ll", s.log_likelihood(), "refine", s.log_likelihood(refine=True), "ll(theta)", s.log_likelihood([1.0, V]))
    print("  dof", s.degrees_of_freedom() if callable(getattr(s,'degrees_of_freedom',None)) else None)
    lo, hi = s.hpd_intervals(0.5); print("  hpd50", lo, hi)
    try:
        lo, hi = s.hpd_intervals(0.5, unconstrained=True); print("  hpd50 unc", lo, hi)
    except Exception as e: print("  hpd unc raised", e)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): s.summary()
    print(buf.getvalue())
    for name in ['plot_trace','plot_posterior','plot_pairwise','plot_ci','plot_convergence','plot_hdr','plot_pairwise_hdr','plot_trace_accel']:
        SEEN.clear()
        try:
            getattr(s, name)(); print(f"  {name}: ok; fixed slot seen during plot: {col(1)}")
        except Exception as e:
            print(f"  {name}: RAISED {type(e).__name__}: {str(e)[:200]}")
        plt.close('all')
    for name in ['plot_trace','plot_posterior','plot_pairwise','plot_ci']:
        try:
            getattr(s, name)(unconstrained=True); print(f"  {name}(unconstrained=True): ok")
        except Exception as e:
            print(f"  {name}(unconstrained=True): RAISED {type(e).__name__}: {str(e)[:200]}")
        plt.close('all')
scen("B5 exports/plots, 40 particles, fixed 1e-10", b5)
def b6():
    # scalar prior + fixed
    s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=gauss(), fixed=[(1, 0.3)], **KW); s.optimize(); report(s,[1])
    print("  free particles", np.asarray(s.get_results()['particles'])[:,0])
scen("B6 scalar prior + fixed 0.3", b6)
def b7():
    # sqrt transform: finite at placeholder 0 but infinite derivative
    s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[GaussPrior(ci=[0.5,5]), None], fixed=[(1, 0.3)], positive_params=False,
             param_transform=lambda p: jnp.sqrt(jnp.abs(p)) + 0.1, **KW); s.optimize(); report(s,[0,1])
scen("B7 custom transform with infinite derivative at placeholder", b7)
def b8():
    s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[gauss(), gauss()], fixed=None, **KW)
    print("  no fixed: fixed_theta", s.fixed_theta, "free_transform is param_transform", s._free_transform is s.param_transform)
    for bad in ([(1.0, 0.3)], [(np.int64(1), 0.3)], [(True, 0.3)], [(1, jnp.asarray(0.3))], [(1, np.float32(0.3))], [(1,0.3),(1,0.4)], [[1,0.3]], ((1,0.3),), []):
        try:
            s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[gauss(), None], fixed=bad, **KW)
            print("  fixed=%r -> mask %s theta %s" % (bad, np.asarray(s.fixed_mask), np.asarray(s.fixed_theta)))
        except Exception as e: print("  fixed=%r RAISED %s: %s" % (bad, type(e).__name__, str(e)[:150]))
scen("B8 parse edge cases", b8)
def b9():
    for verbose in (False, True):
        try:
            s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[None, None], fixed=[(0,1.0),(1, 0.3)], verbose=verbose, **KW)
            print(f"  all fixed verbose={verbose}: constructed"); s.optimize(); report(s,[0,1])
        except Exception as e: print(f"  all fixed verbose={verbose} RAISED {type(e).__name__}: {str(e)[:200]}")
scen("B9 all fixed", b9)
def b10():
    init = np.full((6,2), 3.0)
    s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[gauss(), None], fixed=[(1, 0.25)], theta_init=init, positive_params=False, n_iterations=2, learning_rate=0.01, progress=False, seed=1); s.optimize(); report(s,[1])
    print("  caller's init mutated?", init[:,1])
scen("B10 theta_init + no transform", b10)
def b11():
    # bool mask, 2-D rewards / multivariate
    s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[gauss(), None], fixed=[False, True], **KW); s.optimize(); report(s,[1]); print("  mask dtype", s.fixed_mask.dtype)
    s = SVGD(model=mk(), observed_data=REAL, theta_dim=2, prior=[gauss(), None], fixed=np.array([0, 1]), **KW); s.optimize(); report(s,[1]); print("  mask dtype", s.fixed_mask.dtype)
scen("B11 mask dtypes", b11)
def b12():
    s = fit(parallel='none'); s.optimize(); report(s,[1])
    s = fit(jit=False); s.optimize(); report(s,[1])
scen("B12 parallel none / jit False", b12)
