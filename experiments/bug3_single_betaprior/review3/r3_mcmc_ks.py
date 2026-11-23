"""Review 3: is plan test 3 (MCMC flat likelihood, KS vs Beta) sound?
Post-fix behaviour is EMULATED on the unmodified build with
positive_params=False + param_transform=_sigmoid (MCMC then hands sigmoid to
the prior and the model; init via inverse sigmoid) -- exactly what the fix
would wire for a single BetaPrior.  Reports KS p on raw (autocorrelated)
draws, ESS, KS p on draws thinned to ~ESS, and ESS-based z scores."""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from scipy import stats
from phasic import BetaPrior, GaussPrior
from phasic.mcmc import MCMC
from phasic.svgd import _sigmoid
def run(seed, mk, n_samples, n_chains, burn, thin=1, emulate=True, ps=1.0):
    pr = mk()
    kw = dict(positive_params=False, param_transform=_sigmoid) if emulate else dict(positive_params=True)
    m = MCMC(log_prob_fn=lambda t: 0.0 * jnp.sum(t), prior=pr, theta_dim=1, n_samples=n_samples, n_chains=n_chains,
             burn_in=burn, thin=thin, proposal_scale=ps, verbose=False, progress=False, seed=seed, **kw)
    m.run(); r = m.get_results()
    return pr, np.asarray(r['particles']).ravel(), float(np.asarray(r['ess']).ravel()[0]), float(np.asarray(r['rhat']).ravel()[0]), np.asarray(r['chains'])
for name, mk in (('Beta(ci .05-.95)', lambda: BetaPrior(ci=(0.05, 0.95))), ('Beta(2,5)', lambda: BetaPrior(alpha=2.0, beta=5.0))):
    for (ns, nc, burn) in ((400, 2, 50), (2000, 4, 500)):
        t0 = time.time(); rows = []
        for seed in range(8):
            pr, x, ess, rhat, chains = run(seed, mk, ns, nc, burn)
            dist = stats.beta(pr.alpha, pr.beta)
            p_raw = stats.kstest(x, dist.cdf).pvalue
            k = max(1, int(np.ceil(2 * x.size / max(ess, 1.0))))
            xt = chains[:, ::k, 0].ravel()
            p_thin = stats.kstest(xt, dist.cdf).pvalue
            z_mean = (x.mean() - dist.mean()) / (dist.std() / np.sqrt(ess))
            q = np.quantile(x, [0.1, 0.5, 0.9]); qerr = np.abs(dist.cdf(q) - [0.1, 0.5, 0.9]).max()
            rows.append((p_raw, p_thin, ess, rhat, z_mean, qerr, x.min(), x.max(), xt.size))
        R = np.array(rows)
        print(f'{name}  n_samples={ns} n_chains={nc} burn={burn}  ({time.time()-t0:.0f}s for 8 seeds)')
        print('   KS p raw   :', np.round(R[:, 0], 4).tolist(), ' median', round(float(np.median(R[:, 0])), 4), ' #<0.05:', int((R[:, 0] < 0.05).sum()))
        print('   KS p thin  :', np.round(R[:, 1], 4).tolist(), ' median', round(float(np.median(R[:, 1])), 4), ' #<0.05:', int((R[:, 1] < 0.05).sum()), ' n_thinned', R[:, 8].astype(int).tolist())
        print('   ESS        :', R[:, 2].round(0).tolist(), ' of', ns * nc, ' rhat max', R[:, 3].max().round(3))
        print('   z(mean|ESS):', R[:, 4].round(2).tolist())
        print('   max |F(q)-q| at q=.1,.5,.9:', R[:, 5].round(3).tolist())
        print('   range      :', R[:, 6].min(), R[:, 7].max())
print('## today (softplus wiring, positive_params=True), same statistics, 2 seeds: the test must FAIL here')
for seed in range(2):
    pr, x, ess, rhat, chains = run(seed, lambda: BetaPrior(ci=(0.05, 0.95)), 2000, 4, 500, emulate=False)
    dist = stats.beta(pr.alpha, pr.beta)
    print(f'   seed {seed}: frac>1 {np.mean(x > 1):.3f} max {x.max():.3f} mean {x.mean():.3f} KS p {stats.kstest(x, dist.cdf).pvalue:.3g} ESS {ess:.0f} nan {np.isnan(x).sum()}')
