"""Clip tail: BetaPrior with alpha, beta < 1.  theta = clip(sigmoid(phi), 1e-9, 1-1e-9)
and the Jacobian is computed from the CLIPPED theta, so the phi-density is flat
(improper) beyond |phi| > 20.7 at height exp(-20.7*alpha) relative to ~1."""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from scipy import stats
from phasic import BetaPrior
from phasic.mcmc import MCMC
class BetaExact(BetaPrior):
    """Candidate correction: density in phi from the UNCLIPPED sigmoid,
    log p(phi) = alpha*log_sigmoid(phi) + beta*log_sigmoid(-phi)  (= Beta log-density + log-Jacobian)."""
    def __call__(self, phi):
        return jnp.sum(self.alpha * jax.nn.log_sigmoid(phi) + self.beta * jax.nn.log_sigmoid(-phi))
d = np.array([1.0, 2.0])
def flat(theta, times, rewards=None):
    return jnp.ones_like(times) + 0.0 * jnp.sum(theta), jnp.zeros(2)
for ci in [(0.05, 0.95), (0.01, 0.5), (0.05, 0.5), (0.01, 0.99), (0.001, 0.999), (0.001, 0.5), (1e-4, 0.1)]:
    q = BetaPrior(ci=ci); print(f'BetaPrior(ci={ci}): alpha={q.alpha:.3f} beta={q.beta:.3f}', flush=True)
for a in (0.5, 0.1):
    b = BetaPrior(alpha=a, beta=a)
    b._transform = type(b)._natural_transform
    lp = [float(b(jnp.array([x]))) for x in (0.0, -10.0, -20.0, -25.0, -40.0, -400.0)]
    print(f'Beta({a},{a}) log-density in phi at 0,-10,-20,-25,-40,-400: {np.round(lp, 3).tolist()}', flush=True)
    for cls in (BetaPrior, BetaExact):
        seed = 0; t0 = time.time()
        m = MCMC(model=flat, observed_data=d, prior=cls(alpha=a, beta=a), theta_dim=1, n_samples=800, n_chains=4,
                 burn_in=500, thin=6, proposal_scale=1.5, verbose=False, seed=seed, parallel='vmap')
        m.run(); r = m.get_results(); P = np.asarray(r['particles'])[:, 0]; U = np.asarray(r['particles_unconstrained'])[:, 0]
        ks = stats.kstest(P, stats.beta(a, a).cdf)
        print(f'  MCMC {cls.__name__} seed={seed} ({time.time() - t0:.0f}s): KS D={ks.statistic:.4f} p={ks.pvalue:.3g}; frac of samples AT the clip (theta==1e-9 or 1-1e-9): '
              f'{np.mean((P <= 1e-9) | (P >= 1 - 1e-9)):.4f} (Beta mass outside [1e-9,1-1e-9]: {2 * stats.beta(a, a).cdf(1e-9):.4f}); max|phi|={np.abs(U).max():.1f}; '
              f'per-chain max|phi| {np.abs(np.asarray(m._chains_phi)[:, :, 0]).max(axis=1).round(1).tolist()}', flush=True)
