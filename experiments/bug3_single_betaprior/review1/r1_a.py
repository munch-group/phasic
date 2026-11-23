"""Review 1 (diagnosis) of bug 3: unmodified code, fresh model object per fit."""
import warnings, math
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import phasic
from phasic import Graph, GaussPrior, BetaPrior, SVGD
from phasic.svgd import _sigmoid
from phasic.mcmc import MCMC
from scipy import stats
print('phasic from', phasic.__file__)
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy(dim):
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.ones_like(times), jnp.zeros(2)
    return spy, seen
def beta(): return BetaPrior(ci=(0.05, 0.95))
b0 = beta(); print('Beta params', b0.alpha, b0.beta)
A, B = b0.alpha, b0.beta

print('\n## E1 BetaPrior.__call__ when handed softplus (what MCMC does)')
p = beta(); p._transform = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)
for phi in [-3.0, 0.0, 0.5, 0.55, 1.0, 3.0]:
    th = float(jax.nn.softplus(phi)); print(f'  phi={phi:5.2f} theta=softplus={th:.4f} logp={float(p(jnp.array([phi])))}')
print('  sample() with softplus _transform returns logit(beta draw): ', np.asarray(p.sample(jax.random.PRNGKey(0), (4,))).round(3))

print('\n## E1b MCMC positive_params=True: per-chain behaviour (flat likelihood)')
for label, mk, dim in [('single', beta, 1), ('[Beta]', lambda: [beta()], 1)]:
    for seed in range(4):
        pr = mk()
        m = MCMC(log_prob_fn=lambda th: 0.0 * jnp.sum(th), prior=pr, theta_dim=dim, n_samples=1500, n_chains=4,
                 burn_in=200, proposal_scale=1.0, positive_params=True, verbose=False, progress=False, seed=seed)
        init_theta = np.asarray(m.param_transform(m.theta_init)).ravel()
        init_lp = [float(m._log_prob(m.theta_init[c])) for c in range(4)]
        m.run(); r = m.get_results(); ch = np.asarray(r['chains'])[:, :, 0]
        print(f'  {label} seed {seed}: init theta {init_theta.round(3).tolist()} init logp {np.round(init_lp,3).tolist()} '
              f'acc {np.asarray(m.acceptance_rate).round(3).tolist()} n_unique/chain {[len(np.unique(c)) for c in ch]} '
              f'chain max {ch.max(axis=1).round(3).tolist()} chain mean {ch.mean(axis=1).round(3).tolist()}')
print(f'  expected mean of moving chains if target is Beta(a,b)*theta(1-theta)/(1-exp(-theta)) on (0,1):', end=' ')
x = np.linspace(1e-6, 1-1e-6, 200001); w = x**(A-1)*(1-x)**(B-1)*x*(1-x)/(1-np.exp(-x)); print(round(float((x*w).sum()/w.sum()), 4), '(Beta mean', round(A/(A+B), 4), ')')

print('\n## E2 MCMC positive_params=False with BetaPrior: does the untransformed path sample Beta?')
for label, mk in [('single', beta), ('[Beta]', lambda: [beta()])]:
    ps = []
    for seed in range(5):
        pr = mk()
        m = MCMC(log_prob_fn=lambda th: 0.0 * jnp.sum(th), prior=pr, theta_dim=1, n_samples=4000, n_chains=2,
                 burn_in=300, proposal_scale=0.5, positive_params=False, verbose=False, progress=False, seed=seed)
        m.run(); r = m.get_results(); P = np.asarray(r['particles'])[:, 0]
        thin = P.reshape(2, -1)[:, ::40].ravel()
        ps.append(stats.kstest(thin, 'beta', args=(A, B)).pvalue)
        if seed == 0: print(f'  {label}: min {P.min():.4f} max {P.max():.4f} mean {P.mean():.4f} acc {np.asarray(m.acceptance_rate).round(3).tolist()}')
    print(f'  {label}: KS p-values vs Beta (thinned x40) {np.round(ps,3).tolist()}')

print('\n## E3/E4 SVGD: range of the PARTICLES (not preconditioner probes); preconditioner grid shown separately')
for label, mk, dim, kw in [
    ('single Beta dim1', beta, 1, {}),
    ('[Beta] dim1', lambda: [beta()], 1, {}),
    ('single Beta dim1 pos=False', beta, 1, dict(positive_params=False)),
    ('[Beta] dim1 pos=False', lambda: [beta()], 1, dict(positive_params=False)),
    ('single Beta dim1 preconditioner=none', beta, 1, dict(preconditioner='none')),
    ('single Beta pos=False preconditioner=none', beta, 1, dict(positive_params=False, preconditioner='none')),
]:
    spy, seen = make_spy(dim); pr = mk()
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=pr, n_particles=20, n_iterations=30,
                 progress=False, seed=1, learning_rate=0.05, **kw)
        s.optimize(); r = s.get_results(); P = np.asarray(r['particles'])
        allv = np.concatenate(seen)
        print(f'  {label}: constrained particles min {P.min():.4f} max {P.max():.4f} mean {P.mean():.4f}; '
              f'n>1: {(P>1).sum()}/{P.size}; n<0: {(P<0).sum()}; all model inputs range [{allv.min():.4f}, {allv.max():.4f}]')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:200]}')

print('\n## E5 custom param_transform')
exp_t = lambda phi: jnp.exp(phi)
sig_t = lambda phi: jax.nn.sigmoid(phi)
for label, mk, dim, pt in [
    ('[Gauss(lower=None), Beta] + exp', lambda: [GaussPrior(ci=[0.5, 5]), beta()], 2, exp_t),
    ('[Beta] + exp', lambda: [beta()], 1, exp_t),
    ('single Beta + exp', beta, 1, exp_t),
    ('single Beta + custom sigmoid', beta, 1, sig_t),
    ('[Beta] + custom sigmoid', lambda: [beta()], 1, sig_t),
    ('single Gauss(lower=None) + exp', lambda: GaussPrior(ci=[0.5, 5]), 1, exp_t),
]:
    spy, seen = make_spy(dim); pr = mk(); prl = pr if isinstance(pr, list) else [pr]
    try:
        s = SVGD(model=spy, observed_data=d, theta_dim=dim, prior=pr, n_particles=20, n_iterations=5, progress=False,
                 seed=1, learning_rate=0.05, positive_params=False, param_transform=pt, preconditioner='none')
        phi = jnp.full((dim,), 0.7)
        mt = np.asarray(s.param_transform(phi)).round(4).tolist()
        prt = [float(np.asarray(q._transform(jnp.array([0.7])))[0]) for q in prl]
        init_c = np.asarray(s.param_transform(s.theta_init))
        s.optimize(); P = np.asarray(s.get_results()['particles'])
        print(f'  {label}: model transform(0.7)={mt}; prior._transform(0.7)={np.round(prt,4).tolist()}; '
              f'init constrained range {init_c.min(axis=0).round(3).tolist()}..{init_c.max(axis=0).round(3).tolist()}; final max {P.max(axis=0).round(3).tolist()}')
    except Exception as e:
        print(f'  {label}: {type(e).__name__}: {str(e)[:200]}')
g = GaussPrior(ci=[0.5, 5]); g._transform = exp_t
phi = jnp.array([0.7]); th = float(jnp.exp(0.7))
print(f'  GaussPrior with exp transform: logp={float(g(phi)):.6f}; N(theta)+log|exp Jacobian|={-0.5*((th-g.mu)/g.sigma)**2 + 0.7:.6f}; '
      f'N(theta)+softplus Jacobian={-0.5*((th-g.mu)/g.sigma)**2 - float(jax.nn.softplus(-0.7)):.6f}')
smp = np.asarray(g.sample(jax.random.PRNGKey(0), (2000, 1)))
print(f'  GaussPrior(exp transform).sample -> exp(phi) mean {np.exp(smp).mean():.3f} (prior mu {g.mu:.3f}); softplus(phi) mean {np.log1p(np.exp(smp)).mean():.3f}')

print('\n## E6 is single == list at initialisation? (plan test 2 asserts particles array-equal)')
for dim in (1, 2):
    s1 = SVGD(model=make_spy(dim)[0], observed_data=d, theta_dim=dim, prior=GaussPrior(ci=[0.5, 5], lower=0.0), n_particles=8, n_iterations=2, progress=False, seed=1, learning_rate=0.05)
    s2 = SVGD(model=make_spy(dim)[0], observed_data=d, theta_dim=dim, prior=[GaussPrior(ci=[0.5, 5], lower=0.0) for _ in range(dim)], n_particles=8, n_iterations=2, progress=False, seed=1, learning_rate=0.05)
    print(f'  Gauss dim {dim}: theta_init equal: {bool(np.array_equal(np.asarray(s1.theta_init), np.asarray(s2.theta_init)))}; max abs diff {float(np.abs(np.asarray(s1.theta_init)-np.asarray(s2.theta_init)).max()):.4f}')
m1 = MCMC(log_prob_fn=lambda th: 0.0*jnp.sum(th), prior=GaussPrior(ci=[0.5, 5], lower=0.0), theta_dim=1, n_samples=5, n_chains=2, burn_in=1, verbose=False, progress=False, seed=0)
m2 = MCMC(log_prob_fn=lambda th: 0.0*jnp.sum(th), prior=[GaussPrior(ci=[0.5, 5], lower=0.0)], theta_dim=1, n_samples=5, n_chains=2, burn_in=1, verbose=False, progress=False, seed=0)
print(f'  MCMC Gauss dim 1: theta_init equal: {bool(np.array_equal(np.asarray(m1.theta_init), np.asarray(m2.theta_init)))}')

print('\n## E7 stale _transform on a reused prior object')
bb = beta()
s = SVGD(model=make_spy(1)[0], observed_data=d, theta_dim=1, prior=[bb], n_particles=8, n_iterations=2, progress=False, seed=1, learning_rate=0.05)
print('  after SVGD list fit (positive): _transform is sigmoid:', bb._transform is BetaPrior._natural_transform or float(bb._transform(jnp.array([5.0]))[0]))
m = MCMC(log_prob_fn=lambda th: 0.0*jnp.sum(th), prior=[bb], theta_dim=1, n_samples=5, n_chains=2, burn_in=1, verbose=False, progress=False, seed=0)
print('  after constructing an MCMC with the same object: _transform(5)=', float(bb._transform(jnp.array([5.0]))[0]), ' -> the earlier SVGD object now evaluates its prior with softplus; svgd.param_transform(5)=', np.asarray(s.param_transform(jnp.array([5.0]))).tolist())
s2 = SVGD(model=make_spy(1)[0], observed_data=d, theta_dim=1, prior=[bb], n_particles=8, n_iterations=2, progress=False, seed=1, learning_rate=0.05, positive_params=False)
print('  then SVGD(positive_params=False) with the same object: param_transform', s2.param_transform, '; prior._transform still set:', bb._transform is not None,
      '; theta_init range', np.asarray(s2.theta_init).min().round(3), np.asarray(s2.theta_init).max().round(3))
