"""Review 1, v2 check: proposed BetaPrior log-density a*log_sigmoid(phi)+b*log_sigmoid(-phi) vs current code (unmodified)."""
import numpy as np, jax, jax.numpy as jnp
from phasic import BetaPrior
from phasic.svgd import _sigmoid
def new(a, b): return lambda phi: jnp.sum(a * jax.nn.log_sigmoid(phi) + b * jax.nn.log_sigmoid(-phi))
for kw in [dict(alpha=1.35, beta=1.35), dict(alpha=2.0, beta=5.0), dict(alpha=0.1, beta=0.1), dict(alpha=0.75, beta=3.0)]:
    p = BetaPrior(**kw); p._transform = _sigmoid; f = new(p.alpha, p.beta)
    phis = np.concatenate([np.linspace(-20, 20, 81), [-25, -40, -700, 25, 40, 700]])
    cur = np.array([float(p(jnp.array([x]))) for x in phis]); nw = np.array([float(f(jnp.array([x]))) for x in phis])
    gc = np.array([float(jax.grad(lambda x: p(x))(jnp.array([x]))[0]) for x in phis]); gn = np.array([float(jax.grad(f)(jnp.array([x]))[0]) for x in phis])
    inside = np.abs(phis) <= 20
    print(kw, '| max |new-cur| for |phi|<=20: value %.2e (rel %.2e), grad %.2e' % (np.abs(nw-cur)[inside].max(), (np.abs(nw-cur)/np.abs(cur))[inside].max(), np.abs(gn-gc)[inside].max()))
    print('   tails phi=[-25,-40,-700,25,40,700]: current', cur[-6:].round(3).tolist(), 'grad', gc[-6:].tolist())
    print('                                       new    ', nw[-6:].round(3).tolist(), 'grad', np.round(gn[-6:], 4).tolist(), 'all finite:', bool(np.isfinite(nw).all() and np.isfinite(gn).all()))
    v2 = jnp.array([0.3, -1.2]); print('   2-vector: current %.12f new %.12f' % (float(p(v2)), float(f(v2))))
p = BetaPrior(alpha=2.0, beta=5.0)
print('no-transform branch (must stay): p(0.3)=%.6f p(1.2)=%s p(-0.1)=%s' % (float(p(jnp.array([0.3]))), float(p(jnp.array([1.2]))), float(p(jnp.array([-0.1])))))
