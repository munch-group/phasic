"""v2 review: decisions (a), (b), item 3, guard 12.  Runs on UNPATCHED source; v2 rules are
emulated at runtime (prior._transform reassigned after construction / BetaPrior.__call__
monkeypatched in this process only).  V2=0 gives master behaviour."""
import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from scipy import stats
from phasic import GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
import phasic.svgd as S
d = np.array([0.5, 1.0, 1.5, 2.0])
def model(theta, times, rewards=None):
    return jnp.exp(-times * jnp.sum(theta)) * jnp.sum(theta), jnp.zeros(2) + jnp.sum(theta)
def flat(theta, times, rewards=None):
    return jnp.ones_like(times) + 0.0 * jnp.sum(theta), jnp.zeros(2)
_old_call = BetaPrior.__call__
def new_call(self, phi):                       # fix item 3, natural transform only
    if self._transform is S._sigmoid:
        return jnp.sum(self.alpha * jax.nn.log_sigmoid(phi) + self.beta * jax.nn.log_sigmoid(-phi))
    return _old_call(self, phi)

print('## (b) BetaPrior whose _transform is a custom function (v2 rule) vs its sigmoid (master)')
for tname, T in [('exp', jnp.exp), ('jax.nn.sigmoid', jax.nn.sigmoid), ('0.5*sigmoid', lambda x: 0.5 * jax.nn.sigmoid(x))]:
    b = BetaPrior(alpha=2.0, beta=5.0); b._transform = T
    phis = jnp.array([-2.0, -0.5, 0.5, 2.0])
    lp = [float(b(phis[i:i+1])) for i in range(4)]; g = [float(jax.grad(lambda p: b(p))(phis[i:i+1])[0]) for i in range(4)]
    exact = [float(jax.grad(lambda p: jnp.sum((2 - 1) * jnp.log(T(p)) + (5 - 1) * jnp.log1p(-T(p)) + jnp.log(jnp.abs(jax.grad(lambda q: jnp.sum(T(q)))(p)))))(phis[i:i+1])[0]) for i in range(4)]
    print(f'  _transform={tname}: log-prior at phi=-2,-0.5,0.5,2: {np.round(lp, 3).tolist()}; d/dphi {np.round(g, 3).tolist()}; correct d/dphi {np.round(exact, 3).tolist()}')
for v2 in (False, True):
    for form, mk in [('single', lambda: BetaPrior(alpha=2.0, beta=5.0)), ('list', lambda: [BetaPrior(alpha=2.0, beta=5.0)])]:
        pr = mk(); p0 = pr if form == 'single' else pr[0]
        try:
            s = SVGD(model=model, observed_data=d, theta_dim=1, prior=pr, n_particles=8, n_iterations=20, progress=False, seed=1,
                     learning_rate=0.05, verbose=False, positive_params=False, param_transform=jnp.exp, preconditioner='none')
            if v2:
                p0._transform = jnp.exp
                s.theta_init = jnp.asarray(p0.sample(jax.random.PRNGKey(1), (8, 1)))   # v2: init drawn after wiring
            s.optimize(); P = np.asarray(s.get_results()['particles'])
            print(f'  SVGD {form} Beta + custom exp, {"v2 wiring (prior gets exp)" if v2 else "master wiring (prior keeps sigmoid)"}: init theta range '
                  f'[{float(jnp.exp(s.theta_init).min()):.3g}, {float(jnp.exp(s.theta_init).max()):.3g}]; particles nan={int(np.isnan(P).sum())}/8, range [{np.nanmin(P):.3g}, {np.nanmax(P):.3g}]')
        except Exception as e:
            print(f'  SVGD {form} v2={v2}: {type(e).__name__}: {str(e)[:160]}')
        pr = mk(); p0 = pr if form == 'single' else pr[0]
        try:
            m = MCMC(model=flat, observed_data=d, theta_dim=1, prior=pr, n_samples=300, n_chains=4, burn_in=100, verbose=False, seed=1,
                     positive_params=False, param_transform=jnp.exp, proposal_scale=1.0)
            if v2:
                p0._transform = jnp.exp
            lp0 = [float(m._log_prob(m.theta_init[c])) for c in range(4)]
            m.run(); P = np.asarray(m.get_results()['particles'])
            print(f'  MCMC {form} Beta + custom exp, v2={v2} (master MCMC already hands the custom fn): init log-prob {np.round(lp0, 2).tolist()}; acceptance {np.round(m.acceptance_rate, 2).tolist()}; '
                  f'samples range [{P.min():.3g}, {P.max():.3g}] frac>1: {np.mean(P > 1):.2f}')
        except Exception as e:
            print(f'  MCMC {form} v2={v2}: {type(e).__name__}: {str(e)[:160]}')

print('## (a) MCMC, positive_params=False, no transform, BetaPrior hard wall')
for label, mk, dim, slot, ab in [('single Beta(0.75,0.75)', lambda: BetaPrior(alpha=0.75, beta=0.75), 1, 0, (0.75, 0.75)),
                                 ('[Gauss(lower=None), Beta(2,5)]', lambda: [GaussPrior(ci=[-1, 3]), BetaPrior(alpha=2.0, beta=5.0)], 2, 1, (2.0, 5.0))]:
    pr = mk()
    m = MCMC(model=flat, observed_data=d, theta_dim=dim, prior=pr, n_samples=500, n_chains=4, burn_in=300, thin=8, verbose=False, seed=2,
             positive_params=False, proposal_scale=0.3, parallel='vmap')
    lp0 = [float(m._log_prob(m.theta_init[c])) for c in range(4)]
    m.run(); P = np.asarray(m.get_results()['particles'])[:, slot]
    ks = stats.kstest(P, stats.beta(*ab).cdf)
    print(f'  {label}: param_transform is None: {m.param_transform is None}; init log-prob finite: {bool(np.all(np.isfinite(lp0)))}; acceptance {np.round(m.acceptance_rate, 2).tolist()}; '
          f'in (0,1): {bool(np.all((P > 0) & (P < 1)))}; KS D={ks.statistic:.4f} p={ks.pvalue:.3g}')

print('## guard 12 / item 3: how far do list-form SVGD particles move when only the log-density form changes?')
for label, mk, dim in [('[Beta]', lambda: [BetaPrior(ci=(0.05, 0.95))], 1), ('[Gauss, Beta]', lambda: [GaussPrior(ci=[0.5, 5], lower=0.0), BetaPrior(ci=(0.05, 0.95))], 2)]:
    out = []
    for new in (False, True):
        BetaPrior.__call__ = new_call if new else _old_call
        def mdl(theta, times, rewards=None):          # fresh model object per fit
            return jnp.exp(-times * jnp.sum(theta)) * jnp.sum(theta), jnp.zeros(2) + jnp.sum(theta)
        s = SVGD(model=mdl, observed_data=d, theta_dim=dim, prior=mk(), n_particles=8, n_iterations=30, progress=False, seed=1, learning_rate=0.05, verbose=False)
        s.optimize(); out.append(np.asarray(s.particles))
    BetaPrior.__call__ = _old_call
    print(f'  {label}: bit-identical={np.array_equal(out[0], out[1])}; max|diff| after 30 iterations = {np.abs(out[0] - out[1]).max():.3g}')
b = BetaPrior(alpha=0.75, beta=0.75); b._transform = S._sigmoid
print('  alpha=0.75 shipped log-density at phi=-21,-25,-30:', [round(float(_old_call(b, jnp.array([x]))), 4) for x in (-21., -25., -30.)],
      '| unclipped:', [round(float(new_call(b, jnp.array([x]))), 4) for x in (-21., -25., -30.)])
