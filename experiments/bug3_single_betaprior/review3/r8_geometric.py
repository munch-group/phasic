"""Review 3: plan test 8 -- a discrete geometric model with a probability
parameter.  Is it buildable in a few lines, does the single-BetaPrior form
pass or fail TODAY, and what does the list form (correct today = emulates
the fix) give?  Prior EXCLUDES the truth (project rule).  The Beta prior is
conjugate, so the exact posterior is known: Beta(a + n, b + sum(k - 1))."""
import warnings, random; warnings.filterwarnings('ignore')
import numpy as np, jax.numpy as jnp
from scipy import stats
import phasic
from phasic import Graph, BetaPrior, ExpStepSize
def geom():
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0); v.add_edge(a, [1.0])       # per-step absorption probability = theta[0]
    return g
# model check against the closed form
m = Graph.pmf_from_graph(geom(), discrete=True)
k = np.array([1., 2., 3., 7.]); p = 0.3
print('pmf check:', np.asarray(m(jnp.asarray([p]), jnp.asarray(k))), 'closed form', p * (1 - p) ** (k - 1))
TRUE = 0.3
rng = np.random.default_rng(7); data = rng.geometric(TRUE, size=200).astype(float)
def mkprior(): return BetaPrior(ci=(0.5, 0.95))
pr = mkprior(); pd = stats.beta(pr.alpha, pr.beta)
print(f'prior Beta({pr.alpha:.3f},{pr.beta:.3f}); prior 95% interval ({pd.ppf(.025):.3f},{pd.ppf(.975):.3f}); P(theta<=TRUE)={pd.cdf(TRUE):.2e}')
post = stats.beta(pr.alpha + data.size, pr.beta + (data - 1).sum())
print(f'exact conjugate posterior: mean {post.mean():.4f} sd {post.std():.4f} 95% ({post.ppf(.025):.4f},{post.ppf(.975):.4f}); MLE {1/data.mean():.4f}')
for label, mk in (('single BetaPrior (plan test 8, today)', mkprior), ('[BetaPrior] (correct today = emulates the fix)', lambda: [mkprior()])):
    for seed in (1, 2, 3):
        np.random.seed(seed); random.seed(seed)
        try:
            s = geom().svgd(observed_data=jnp.asarray(data), discrete=True, theta_dim=1, prior=mk(), n_particles=40, n_iterations=150,
                            learning_rate=ExpStepSize(first_step=0.05, last_step=0.005, tau=50.0), progress=False, seed=seed)
            r = s.get_results(); P = np.asarray(r['particles']).ravel()
            lo, hi = float(np.asarray(r['hpd_lower'])[0]), float(np.asarray(r['hpd_upper'])[0])
            print(f'  {label} seed {seed}: particles in (0,1): {bool(np.all((P > 0) & (P < 1)))} [min {P.min():.4f} max {P.max():.4f}] mean {P.mean():.4f} sd {P.std():.4f} '
                  f'HPD ({lo:.4f},{hi:.4f}) truth in HPD: {lo <= TRUE <= hi}; |mean-exact|/sd_exact = {abs(P.mean()-post.mean())/post.std():.2f}')
        except Exception as e:
            print(f'  {label} seed {seed}: {type(e).__name__}: {str(e)[:200]}')
