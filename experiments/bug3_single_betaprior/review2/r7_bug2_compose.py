"""Attack 5 (partial emulation): bug 2's wrapper installed on top of the transform the
prototype returns.  Emulated post-construction (bug 2 installs it inside the constructor),
preconditioner disabled for SVGD (bug 2 changes that call site to _free_transform; not emulated)."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
import proto
warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
from phasic import GaussPrior, BetaPrior, SVGD
from phasic.mcmc import MCMC
import phasic.svgd as S
d = np.array([0.5, 1.0, 1.5, 2.0])
def make_spy():
    seen = []
    def spy(theta, times, rewards=None):
        jax.debug.callback(lambda t: seen.append(np.atleast_2d(np.asarray(t)).copy()), theta)
        return jnp.exp(-times * jnp.sum(theta)) * jnp.sum(theta), jnp.zeros(2) + jnp.sum(theta)
    return spy, seen
def install(obj, v, dim=2):
    base = obj.param_transform; mask = np.asarray(obj.fixed_mask).astype(bool); ft = jnp.where(mask, v, 0.0)
    def with_fixed(phi):
        phi = jnp.asarray(phi)
        if phi.shape[-1] != dim: raise ValueError('shape')
        return jnp.where(mask, ft, base(phi))
    pd = getattr(base, '_per_dim_fns', None)
    if pd is not None: with_fixed._per_dim_fns = pd
    obj._free_transform = base; obj.param_transform = with_fixed
    obj.fixed_values = jnp.zeros(dim)
    obj.theta_init = jnp.where(mask[None, :], 0.0, obj.theta_init)
for v in (5.0, 0.25, 0.0):
    for label, mk in [('single Beta', lambda: BetaPrior(alpha=2.0, beta=5.0)), ('[None, Beta]', lambda: [None, BetaPrior(alpha=2.0, beta=5.0)])]:
        spy, seen = make_spy(); pr = mk()
        s = SVGD(model=spy, observed_data=d, theta_dim=2, prior=pr, n_particles=8, n_iterations=5, progress=False, seed=1,
                 learning_rate=0.05, verbose=False, fixed=[(0, v)], preconditioner='none')
        install(s, v); del seen[:]; s.optimize(); a = np.concatenate(seen); R = np.asarray(s.get_results()['particles'])
        p_i = pr if not isinstance(pr, list) else pr[1]
        print(f'SVGD {label} fixed=(0,{v}): model slot0 all == v: {bool(np.all(a[:, 0] == v))}; slot1 in (0,1): {bool(np.all((a[:, 1] > 0) & (a[:, 1] < 1)))}; '
              f'results slot0 == v: {bool(np.all(R[:, 0] == v))}; prior._transform is sigmoid: {p_i._transform is S._sigmoid}; '
              f'_free_transform works on learnable subset: {np.asarray(s._free_transform(jnp.array([0.0]))).tolist() if not hasattr(s._free_transform, "_per_dim_fns") else "composite -> uses _per_dim_fns[1]: " + str(np.asarray(s._free_transform._per_dim_fns[1](jnp.array([0.0]))).tolist())}')
        spy, seen = make_spy(); pr = mk()
        m = MCMC(model=spy, observed_data=d, theta_dim=2, prior=pr, n_samples=100, n_chains=2, burn_in=20, verbose=False, seed=1, fixed=[(0, v)])
        install(m, v); del seen[:]; m.run(); a = np.concatenate(seen); R = np.asarray(m.get_results()['particles'])
        print(f'MCMC {label} fixed=(0,{v}): model slot0 all == v: {bool(np.all(a[:, 0] == v))}; slot1 in (0,1): {bool(np.all((a[:, 1] > 0) & (a[:, 1] < 1)))}; results slot0 == v: {bool(np.all(R[:, 0] == v))}')
print('plot_pairwise_hdr inverse (svgd.py:9345) on a sigmoid slot: display 0.5 ->', float(S._sigmoid(S._inverse_softplus(jnp.array(0.5)))), '; display 0.9 ->', float(S._sigmoid(S._inverse_softplus(jnp.array(0.9)))))
