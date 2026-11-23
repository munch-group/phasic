"""Attack 2: with a flat likelihood, does MCMC through the prototype sample the
Beta prior (KS vs scipy.stats.beta)?  Thinned to reduce autocorrelation."""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from scipy import stats
from phasic import GaussPrior, BetaPrior
from phasic.mcmc import MCMC
print('PROTO_OFF =', os.environ.get('PROTO_OFF'))
d = np.array([1.0, 2.0])
def flat(theta, times, rewards=None):
    return jnp.ones_like(times) + 0.0 * jnp.sum(theta), jnp.zeros(2)
def run(label, mk, dim, beta_slots, seed, thin=10, n=600, gauss_slots=()):
    pr = mk(); t0 = time.time()
    m = MCMC(model=flat, observed_data=d, prior=pr, theta_dim=dim, n_samples=n, n_chains=4, burn_in=600,
             thin=thin, proposal_scale=1.5, verbose=False, progress=False, seed=seed, parallel='vmap')
    m.run(); r = m.get_results(); P = np.asarray(r['particles']); U = np.asarray(r['particles_unconstrained'])
    out = []
    for i, (a, b) in beta_slots.items():
        ks = stats.kstest(P[:, i], stats.beta(a, b).cdf)
        out.append(f'slot{i} Beta({a:.3g},{b:.3g}): mean {P[:, i].mean():.4f} (true {a/(a+b):.4f}) KS D={ks.statistic:.4f} p={ks.pvalue:.3g} '
                   f'frac at clip={np.mean((P[:, i] <= 1e-9) | (P[:, i] >= 1 - 1e-9)):.4f} max|phi|={np.abs(U[:, i]).max():.1f}')
    for i, g in gauss_slots:
        ks = stats.kstest(P[:, i], stats.truncnorm((0 - g.mu) / g.sigma, np.inf, g.mu, g.sigma).cdf)
        out.append(f'slot{i} truncGauss({g.mu:.3g},{g.sigma:.3g}): mean {P[:, i].mean():.4f} KS D={ks.statistic:.4f} p={ks.pvalue:.3g}')
    print(f'  {label} seed={seed} acc={np.round(m.acceptance_rate, 2).tolist()} ({time.time() - t0:.0f}s): ' + ' | '.join(out), flush=True)
b0 = BetaPrior(ci=(0.05, 0.95)); A, B = b0.alpha, b0.beta
for seed in (0, 1, 2):
    run('single Beta(ci) dim1', lambda: BetaPrior(ci=(0.05, 0.95)), 1, {0: (A, B)}, seed)
for seed in (0, 1):
    run('[Beta(2,5)]', lambda: [BetaPrior(alpha=2.0, beta=5.0)], 1, {0: (2.0, 5.0)}, seed)
    g = GaussPrior(ci=[0.5, 5], lower=0.0)
    run('[Gauss, Beta(2,5)]', lambda: [GaussPrior(ci=[0.5, 5], lower=0.0), BetaPrior(alpha=2.0, beta=5.0)], 2, {1: (2.0, 5.0)}, seed, gauss_slots=[(0, g)])
    run('single Beta(2,5) dim2', lambda: BetaPrior(alpha=2.0, beta=5.0), 2, {0: (2.0, 5.0), 1: (2.0, 5.0)}, seed)
if not os.environ.get('PROTO_OFF'):
    # clip tail: with alpha < 1 the phi-density beyond |phi| > 20.7 is FLAT (theta clipped at 1e-9)
    for seed in (0, 1):
        run('single Beta(0.1,0.1) dim1', lambda: BetaPrior(alpha=0.1, beta=0.1), 1, {0: (0.1, 0.1)}, seed, n=1500)
        run('single Beta(0.05,0.05) dim1', lambda: BetaPrior(alpha=0.05, beta=0.05), 1, {0: (0.05, 0.05)}, seed, n=1500)
