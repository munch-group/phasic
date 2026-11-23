"""Review 3, plan v2: do tests 4, 5 (alpha<1 and [Gauss,Beta] forms) and the
single-seed ESS z-check behave on a CORRECT implementation?  Fix emulated on the
unmodified build: custom sigmoid/composite transform + a BetaPrior subclass with
the plan's unclipped log-density (fix item 3).  Also: BetaPrior handed an
UNCLIPPED custom sigmoid (what v2's custom-transform rule gives SVGD single form)."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from scipy import stats
from phasic import BetaPrior, GaussPrior
from phasic.mcmc import MCMC
from phasic.svgd import _sigmoid
class Beta3(BetaPrior):
    def __call__(self, phi):
        if self._transform is not None:
            return jnp.sum(self.alpha * jax.nn.log_sigmoid(phi) + self.beta * jax.nn.log_sigmoid(-phi))
        return super().__call__(phi)
softplus = lambda phi: jnp.maximum(jax.nn.softplus(phi), 1e-9)
def thin_ks(chains, ess, cdf):
    n = chains.shape[0] * chains.shape[1]; k = max(1, int(np.ceil(2 * n / max(ess, 1.0))))
    return stats.kstest(chains[:, ::k].ravel(), cdf).pvalue
print('## test 5, alpha < 1 form: BetaPrior(ci=(0.01,0.99)), 2000x4, burn 500, 5 seeds')
ps, zs, t0 = [], [], time.time()
for seed in range(5):
    pr = Beta3(ci=(0.01, 0.99))
    m = MCMC(log_prob_fn=lambda t: 0.0 * jnp.sum(t), prior=pr, theta_dim=1, n_samples=2000, n_chains=4, burn_in=500,
             proposal_scale=1.0, positive_params=False, param_transform=_sigmoid, verbose=False, progress=False, seed=seed)
    lp0 = [float(m._log_prob(m.theta_init[c])) for c in range(4)]
    m.run(); r = m.get_results(); ch = np.asarray(r['chains'])[:, :, 0]; ess = float(np.asarray(r['ess'])[0]); d = stats.beta(pr.alpha, pr.beta)
    ps.append(thin_ks(ch, ess, d.cdf)); zs.append((ch.mean() - d.mean()) / (d.std() / np.sqrt(ess)))
    print(f'   seed {seed}: alpha {pr.alpha:.3f} beta {pr.beta:.3f} init lp finite {all(np.isfinite(lp0))} acc {np.asarray(m.acceptance_rate).round(2).tolist()} ESS {ess:.0f} '
          f'range [{ch.min():.2e},{ch.max():.6f}] thinned KS p {ps[-1]:.3f} z {zs[-1]:.2f} max|phi| {np.abs(np.asarray(r["particles_unconstrained"])).max():.1f}')
print(f'   median p {np.median(ps):.3f}  ({time.time()-t0:.0f}s)')
print('## test 5, [GaussPrior, BetaPrior] form, 3 seeds')
comp = lambda phi: jnp.concatenate([softplus(phi[..., 0:1]), _sigmoid(phi[..., 1:2])], axis=-1)
for seed in range(3):
    g, b = GaussPrior(ci=[0.5, 5], lower=0.0), Beta3(alpha=2.0, beta=5.0)
    m = MCMC(log_prob_fn=lambda t: 0.0 * jnp.sum(t), prior=[g, b], theta_dim=2, n_samples=2000, n_chains=4, burn_in=500,
             proposal_scale=1.0, positive_params=False, param_transform=comp, verbose=False, progress=False, seed=seed)
    g._transform = softplus; b._transform = _sigmoid
    m.run(); r = m.get_results(); ch = np.asarray(r['chains']); ess = np.asarray(r['ess'])
    tn = stats.truncnorm((0 - g.mu) / g.sigma, np.inf, loc=g.mu, scale=g.sigma); d = stats.beta(2, 5)
    print(f'   seed {seed}: ESS {ess.round(0).tolist()} Gauss slot thinned KS p {thin_ks(ch[:, :, 0], ess.min(), tn.cdf):.3f} (mu {g.mu:.2f} sigma {g.sigma:.2f}) '
          f'Beta slot thinned KS p {thin_ks(ch[:, :, 1], ess.min(), d.cdf):.3f}  z_gauss {(ch[:,:,0].mean()-tn.mean())/(tn.std()/np.sqrt(ess[0])):.2f} z_beta {(ch[:,:,1].mean()-d.mean())/(d.std()/np.sqrt(ess[1])):.2f}')
print('## test 4: geometric likelihood (pure JAX), truth 0.3, n=200, BetaPrior(ci=(0.5,0.95)); |mean-exact| < 4 sd/sqrt(ESS)?')
data = np.random.default_rng(7).geometric(0.3, size=200).astype(float); n, S = data.size, (data - 1).sum()
for seed in range(4):
    pr = Beta3(ci=(0.5, 0.95)); post = stats.beta(pr.alpha + n, pr.beta + S)
    m = MCMC(log_prob_fn=lambda t: n * jnp.log(t[0]) + S * jnp.log1p(-t[0]), prior=pr, theta_dim=1, n_samples=2000, n_chains=4, burn_in=500,
             proposal_scale=1.0, positive_params=False, param_transform=_sigmoid, verbose=False, progress=False, seed=seed)
    m.run(); r = m.get_results(); x = np.asarray(r['particles']).ravel(); ess = float(np.asarray(r['ess'])[0])
    print(f'   seed {seed}: mean {x.mean():.5f} exact {post.mean():.5f} sd {x.std():.5f} exact sd {post.std():.5f} ESS {ess:.0f} rhat {float(np.asarray(r["rhat"])[0]):.3f} '
          f'acc {np.asarray(m.acceptance_rate).round(2).tolist()} z {(x.mean()-post.mean())/(post.std()/np.sqrt(ess)):.2f} (bound 4)')
print('## BetaPrior handed an UNCLIPPED custom sigmoid (v2 rule: custom transform -> prior._transform = custom)')
for a, b in ((2.0, 5.0), (2.0, 1.0)):
    p = BetaPrior(alpha=a, beta=b); p._transform = jax.nn.sigmoid
    q = BetaPrior(alpha=a, beta=b); q._transform = _sigmoid
    print(f'   Beta({a},{b}) log-prior and d/dphi at phi=5, 37, 40, -800: custom unclipped',
          [(round(float(p(jnp.array([v]))), 2), round(float(jax.grad(lambda z: p(z))(jnp.array([v]))[0]), 3)) for v in (5.0, 37.0, 40.0, -800.0)],
          ' natural clipped', [round(float(q(jnp.array([v]))), 2) for v in (5.0, 37.0, 40.0, -800.0)])
