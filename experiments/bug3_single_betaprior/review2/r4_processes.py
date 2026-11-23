"""MCMC parallel='processes' through the prototype (pure-Python log_prob_fn)."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import numpy as np, jax.numpy as jnp
from phasic import GaussPrior, BetaPrior
from phasic.mcmc import MCMC
def lp(theta):
    return 0.0 * jnp.sum(theta)
if __name__ == '__main__':
    for label, mk, dim in [('single Beta dim2', lambda: BetaPrior(alpha=2.0, beta=5.0), 2),
                           ('[Gauss, Beta]', lambda: [GaussPrior(ci=[0.5, 5], lower=0.0), BetaPrior(alpha=2.0, beta=5.0)], 2)]:
        try:
            m = MCMC(log_prob_fn=lp, prior=mk(), theta_dim=dim, n_samples=300, n_chains=2, burn_in=100,
                     proposal_scale=1.0, verbose=False, seed=0, parallel='processes')
            m.run(); P = np.asarray(m.get_results()['particles'])
            print(f'  {label} [processes]: OK; samples min {P.min(axis=0).round(4).tolist()} max {P.max(axis=0).round(4).tolist()} mean {P.mean(axis=0).round(4).tolist()}')
        except Exception as e:
            print(f'  {label} [processes]: {type(e).__name__}: {str(e)[:300]}')
