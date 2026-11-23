"""Gradient through the wrapper: where-NaN trap, placeholder safety, FD-NaN cotangent."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
mask = jnp.array([False, True, False]); ft = jnp.array([0.0, 0.25, 0.0])
def wrap(base):
    return lambda phi: jnp.where(mask, ft, base(phi))
def loss(theta):            # depends on every slot, incl. the fixed one
    return jnp.sum(theta ** 2) + theta[0] * theta[1] + theta[1] * theta[2]
softplus = lambda p: jnp.maximum(jax.nn.softplus(p), 1e-9)
bases = {
 'softplus+floor': softplus, 'sigmoid': jax.nn.sigmoid, 'exp': jnp.exp,
 'log (domain phi>0)': jnp.log, '1/phi': lambda p: 1.0 / p, 'sqrt': jnp.sqrt,
 'exp(1/phi)': lambda p: jnp.exp(1.0 / p),
 'cumsum(exp) [ordered, non-elementwise]': lambda p: jnp.cumsum(jnp.exp(p)),
 'softmax [simplex, non-elementwise]': jax.nn.softmax,
 'phi/sum(phi) [non-elementwise]': lambda p: p / jnp.sum(p),
 'log then cumsum [bad domain + coupled]': lambda p: jnp.cumsum(jnp.log(p)),
}
for ph in (0.0, jnp.inf, -jnp.inf, jnp.nan, 800.0):
    print(f'## placeholder phi[fixed] = {ph}')
    for name, base in bases.items():
        phi = jnp.array([0.7, ph, 1.3])
        w = wrap(base)
        th = w(phi); g = jax.grad(lambda p: loss(w(p)))(phi)
        gv = jax.vmap(jax.grad(lambda p: loss(w(p))))(jnp.stack([phi, phi]))[0]
        ok_free = bool(np.all(np.isfinite(np.asarray(g)[[0, 2]])))
        print(f'   {name:42s} theta={np.asarray(th)} grad={np.asarray(g)} free_finite={ok_free} fixed_exact0={float(g[1]) == 0.0} vmap_same={np.array_equal(np.asarray(g), np.asarray(gv), equal_nan=True)}')
print('## does the free-slot theta depend on the placeholder under a coupled transform?')
for name in ('cumsum(exp) [ordered, non-elementwise]', 'softmax [simplex, non-elementwise]'):
    w = wrap(bases[name])
    print('   ', name, [np.asarray(w(jnp.array([0.7, ph, 1.3]))).round(4).tolist() for ph in (0.0, 1.0, -3.0)])
print('## NaN cotangent arriving at the fixed slot (FD backward with a negative probe): dropped by where?')
@jax.custom_vjp
def model(theta): return jnp.sum(theta)
model.defvjp(lambda th: (jnp.sum(th), None), lambda _, ct: (jnp.array([1.0, jnp.nan, 1.0]) * ct,))
for name in ('softplus+floor', 'exp'):
    w = wrap(bases[name]); print('   wrapper  ', name, np.asarray(jax.grad(lambda p: model(w(p)))(jnp.array([0.7, 0.0, 1.3]))))
    print('   today    ', name, np.asarray(jax.grad(lambda p: model(bases[name](p)))(jnp.array([0.7, 0.0, 1.3]))))
