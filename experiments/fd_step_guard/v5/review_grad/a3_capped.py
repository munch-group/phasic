"""A3 (prototype worktree): the CAPPED rule h = min(1e-7, s*theta), s = 1e-4, on an Euler/automatic-granularity
model (pmf_and_moments pdf part, site C3, two-island n=8) for theta in the band the plan newly changes
(1e-7 <= theta < 1e-3: step shrinks from 1e-7 to 1e-4*theta) and below it.
Reference: 5-point central stencil in log(theta) on the shipped forward (rates << 512: G constant, forward smooth).
Reports theta-space relative error and phi-space (theta * dlogL/dtheta) absolute error; plus the forward's noise."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/experiments/fd_step_guard')
import numpy as np, jax, jax.numpy as jnp
from sites import two_island_graph, Graph
g = two_island_graph(8)
print('vertices', g.vertices_length())
T = jnp.array([0.3, 0.8, 1.5, 2.5])
M = Graph.pmf_and_moments_from_graph(g, nr_moments=2, theta_dim=2, exact_moment_grad=False)
ll = lambda th: jnp.sum(jnp.log(M(th, T)[0]))
def ref(th, i, h=2e-3):
    def f(k):
        t = np.array(th, float); t[i] *= np.exp(k * h); return float(ll(jnp.array(t)))
    return (-f(2) + 8 * f(1) - 8 * f(-1) + f(-2)) / (12 * h)      # d ll / d log theta_i
rng = np.random.default_rng(0)
for th in ([1.0, 1e-2], [1.0, 1e-4], [1.0, 1e-6], [1.0, 1e-8], [1e-4, 1e-4], [1e-5, 1.0]):
    base = float(ll(jnp.array(th)))
    noise = np.std([float(ll(jnp.array(np.array(th) * (1 + 1e-13 * rng.standard_normal(2))))) - base for _ in range(12)])
    r = np.array([ref(th, 0), ref(th, 1)])
    out = [f'theta={th} ll={base:.6g} fwd-noise(abs, ll)={noise:.1e} ref dll/dlog={r}']
    for name, env in (('today', ('old', '1')), ('capped s=1e-4', ('', '1e-4')), ('capped s=1e-3', ('', '1e-3'))):
        os.environ['PHASIC_FD_RULE'], os.environ['PHASIC_FD_REL'] = env
        if not env[0]: os.environ.pop('PHASIC_FD_RULE')
        try:
            v = np.asarray(jax.grad(ll)(jnp.array(th)))
            phi = v * np.array(th)
            out.append(f'   {name:14s}: theta-space rel err {np.abs(phi - r) / np.abs(r)}  phi-space abs err {np.abs(phi - r)}')
        except Exception as e:
            out.append(f'   {name:14s}: RAISES {type(e).__name__} {str(e)[:80]!r}')
    print('\n'.join(out), flush=True)
