"""Review 1 part c: plot_pairwise_hdr inverts every axis with inverse softplus (svgd.py:9345), also sigmoid slots."""
import warnings
import numpy as np, jax, jax.numpy as jnp
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
from phasic import GaussPrior, BetaPrior, SVGD
d = np.array([0.5, 1.0, 1.5, 2.0])
seen = []
def spy(theta, times, rewards=None):
    jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
    return jnp.ones_like(times), jnp.zeros(2)
s = SVGD(model=spy, observed_data=d, theta_dim=2, prior=[GaussPrior(ci=[0.5, 5], lower=0.0), BetaPrior(ci=(0.05, 0.95))], n_particles=20,
         n_iterations=5, progress=False, seed=1, learning_rate=0.05, preconditioner='none')
s.optimize(); n0 = len(seen)
s.plot_pairwise_hdr(return_fig=True, center_map=False, ylim=(0.2, 0.8), xlim=(1.0, 3.0), n=6, pad=0)
v = np.concatenate([a.reshape(-1, 2) for a in seen[n0:]])
y = np.unique(v[:, 1].round(10)); x = np.unique(v[:, 0].round(10))
print('Beta-slot values the model was evaluated at   :', y.round(4).tolist())
print('  their spacing (a hex lattice is evenly spaced):', np.diff(y).round(4).tolist())
g = np.log1p(np.exp(np.log(y / (1 - y))))   # softplus(logit(y)) = the grid value the plot labels this row with
print('softplus(logit(.)) of them = plotted axis value:', g.round(4).tolist())
print('  spacing                                       :', np.diff(g).round(4).tolist())
print('=> a row drawn at axis value', g.round(3).tolist()[0], 'was evaluated at theta =', y.round(3).tolist()[0], '; at', g.round(3).tolist()[-1], '->', y.round(3).tolist()[-1])
print('Gauss-slot values (control, softplus round trip) spacing:', np.diff(x).round(4).tolist()[:6])
