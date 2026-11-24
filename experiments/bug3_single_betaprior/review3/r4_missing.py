"""Review 3: gaps in the plan's test list, probed on the unmodified build.
Post-fix single form emulated by positive_params=False + param_transform=_sigmoid
where stated.  Fresh model object per fit."""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import numpy as np, jax, jax.numpy as jnp
from phasic import GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
from phasic.svgd import _sigmoid, _inverse_softplus
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy():
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
def beta(): return BetaPrior(ci=(0.05, 0.95))
def mk(prior, dim, **kw):
    spy, seen = make_spy()
    kw.setdefault('n_iterations', 3)
    s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=prior, n_particles=8,
             progress=False, seed=1, learning_rate=0.5, verbose=False, **kw)
    return s, seen

print('## (a) test 2 repaired: identical theta_init, emulated-single vs list form')
for dim, lst in ((1, lambda: [beta()]), (2, lambda: [beta(), beta()])):
    init = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (8, dim)))
    s1, _ = mk(beta(), dim, positive_params=False, param_transform=_sigmoid, theta_init=init); s1.optimize()
    s2, _ = mk(lst(), dim, theta_init=init); s2.optimize()
    a, b = np.asarray(s1.particles), np.asarray(s2.particles)
    print(f'  dim {dim}: array_equal {np.array_equal(a, b)}  max|diff| {np.abs(a - b).max():.3g}')

print('## (b) initial particles (constrained) for the single form today, and HPD / theta_mean range')
for label, pr, dim in (('single dim1', beta(), 1), ('single dim2', beta(), 2), ('[Beta,Beta]', [beta(), beta()], 2)):
    s, _ = mk(pr, dim); t0 = np.asarray(s.param_transform(s.theta_init)); s.optimize(); r = s.get_results()
    print(f'  {label}: init theta range [{t0.min():.4f},{t0.max():.4f}]  results particles range [{np.asarray(r["particles"]).min():.4f},{np.asarray(r["particles"]).max():.4f}] '
          f'hpd_lower {np.asarray(r["hpd_lower"]).round(4).tolist()} hpd_upper {np.asarray(r["hpd_upper"]).round(4).tolist()} theta_mean {np.asarray(s.theta_mean).round(4).tolist()}')

print('## (c) plot_pairwise_hdr on a sigmoid-transformed fit ([Beta,Beta] today = what a single BetaPrior dim-2 fit becomes)')
th = np.array([0.05, 0.3, 0.5, 0.9, 0.99])
print('  axis theta        :', th.tolist())
print('  evaluated at theta:', np.asarray(_sigmoid(_inverse_softplus(jnp.asarray(th)))).round(4).tolist(), ' (= sigmoid(inverse_softplus(theta)), svgd.py:9345 + 9402)')
s, seen = mk([beta(), beta()], 2); s.optimize(); r = s.get_results(); n0 = len(seen)
try:
    s.plot_pairwise_hdr()
    g = np.concatenate(seen[n0:]); P = np.asarray(r['particles'])
    print(f'  particles (display) range dim0 [{P[:,0].min():.3f},{P[:,0].max():.3f}] dim1 [{P[:,1].min():.3f},{P[:,1].max():.3f}]')
    print(f'  model evaluated by the plot on theta dim0 [{g[:,0].min():.3f},{g[:,0].max():.3f}] dim1 [{g[:,1].min():.3f},{g[:,1].max():.3f}]  ({len(g)} evaluations)')
except Exception as e:
    print(f'  plot_pairwise_hdr: {type(e).__name__}: {str(e)[:200]}')
for name in ('plot_posterior', 'plot_trace', 'plot_ci', 'plot_pairwise', 'plot_convergence'):
    try: getattr(s, name)(); print(f'  {name}: ok')
    except Exception as e: print(f'  {name}: {type(e).__name__}: {str(e)[:150]}')
s1, _ = mk(beta(), 1); s1.optimize()
for name in ('plot_posterior', 'plot_trace', 'plot_ci'):
    try: getattr(s1, name)(); print(f'  single dim1 {name}: ok')
    except Exception as e: print(f'  single dim1 {name}: {type(e).__name__}: {str(e)[:150]}')
try: s1.summary(); 
except Exception as e: print(f'  summary: {type(e).__name__}: {str(e)[:150]}')

print('## (d) fixed slots')
for label, pr, kw in (('single BetaPrior, dim 2, fixed=[(0, 5.0)]', beta(), dict(fixed=[(0, 5.0)])),
                      ('[None, BetaPrior], fixed=[(0, 5.0)]', [None, beta()], dict(fixed=[(0, 5.0)])),
                      ('[BetaPrior, None], fixed=[(1, 5.0)]', [beta(), None], dict(fixed=[(1, 5.0)]))):
    try:
        s, seen = mk(pr, 2, **kw); s.optimize(); v = np.concatenate(seen)
        print(f'  SVGD {label}: model saw dim0 [{v[:,0].min():.4f},{v[:,0].max():.4f}] dim1 [{v[:,1].min():.4f},{v[:,1].max():.4f}]')
    except Exception as e: print(f'  SVGD {label}: {type(e).__name__}: {str(e)[:150]}')
    try:
        seen = []
        def lp(theta, seen=seen):
            jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta); return 0.0 * jnp.sum(theta)
        pr2 = beta() if not isinstance(pr, list) else [None if p is None else beta() for p in pr]
        m = MCMC(log_prob_fn=lp, prior=pr2, theta_dim=2, n_samples=200, n_chains=2, burn_in=20, proposal_scale=1.0, verbose=False, progress=False, seed=0, **kw)
        m.run(); v = np.concatenate(seen)
        print(f'  MCMC {label}: model saw dim0 [{v[:,0].min():.4f},{v[:,0].max():.4f}] dim1 [{v[:,1].min():.4f},{v[:,1].max():.4f}]')
    except Exception as e: print(f'  MCMC {label}: {type(e).__name__}: {str(e)[:150]}')
print('  emulated all-sigmoid transform applied to the stored fixed value for 5.0 (what "every slot uses the natural transform" gives without bug 2\'s wrapper):',
      float(_sigmoid(jnp.log(jnp.exp(5.0) - 1))))

print('## (e) default / non-natural priors: transform identity today (guard reference)')
for label, pr in (('prior=None', None), ('GaussPrior single', GaussPrior(ci=[0.5, 5], lower=0.0)), ('[GaussPrior]', [GaussPrior(ci=[0.5, 5], lower=0.0)])):
    try:
        s, _ = mk(pr, 1)
        print(f'  SVGD {label}: param_transform(5)={float(np.asarray(s.param_transform(jnp.array([5.0])))[0]):.6f} has _per_dim_fns={hasattr(s.param_transform, "_per_dim_fns")}')
    except Exception as e: print(f'  SVGD {label}: {type(e).__name__}: {str(e)[:150]}')
s, _ = mk([GaussPrior(ci=[0.5, 5], lower=0.0), beta()], 2); print('  [Gauss,Beta] has _per_dim_fns:', hasattr(s.param_transform, '_per_dim_fns'))

print('## (f) BetaPrior target in phi is flat beyond the sigmoid clip (improper tails): log prior(phi) for U-shaped Beta(0.5,0.5) and Beta(2,5)')
for a, b in ((0.5, 0.5), (2.0, 5.0)):
    p = BetaPrior(alpha=a, beta=b); p._transform = _sigmoid
    print(f'  Beta({a},{b}):', {phi: round(float(p(jnp.array([phi]))), 3) for phi in (0.0, 10.0, 20.0, 25.0, 50.0, 500.0)})
