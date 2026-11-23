import numpy as np, jax, jax.numpy as jnp, pickle
import phasic
from phasic import GaussPrior
from phasic.mcmc import MCMC
def log_prob(theta):
    return -0.5 * jnp.sum((theta[0] - 1.0) ** 2) - 0.0 * theta[1]
if __name__ == "__main__":
    print(phasic.__file__)
    for v in (1e-10, 0.3):
        try:
            m = MCMC(log_prob_fn=log_prob, prior=[GaussPrior(ci=[0.5, 5], lower=0.0), None], theta_dim=2, fixed=[(1, v)],
                     n_samples=60, n_chains=2, burn_in=10, proposal_scale=0.5, verbose=False, progress=False, seed=0,
                     positive_params=True, parallel='processes')
            m.run(); r = m.get_results()
            print(v, "uniq", np.unique(np.asarray(r['particles'])[:, 1]), "mean", float(r['theta_mean'][1]), "std", float(r['theta_std'][1]), r['hpd_lower'][1], r['hpd_upper'][1])
        except Exception as e:
            print(v, "RAISED", type(e).__name__, str(e)[:300])
