import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)
x = jnp.asarray(rng.standard_normal(200000) * 10.0 ** rng.uniform(-12, 3, 200000))
eps = 1e-7
f_today = jax.jit(lambda x: x / (2 * eps))
f_where = jax.jit(lambda x, m: x / jnp.where(m, x * 0 + 3.0, 2 * eps))
f_sel   = jax.jit(lambda x, m: jnp.where(m, x / 3.0, x / (2 * eps)))
m = jnp.zeros_like(x, dtype=bool)
a, b, c = map(np.asarray, (f_today(x), f_where(x, m), f_sel(x, m)))
print("eager==jit today", np.array_equal(np.asarray(x / (2*eps)), a))
print("divisor-select bit-identical:", np.array_equal(a, b), "n diff", int(np.sum(a != b)))
print("result-select bit-identical:", np.array_equal(a, c), "n diff", int(np.sum(a != c)))
# scalar-in-loop form like the sites: cotangent * (fp - fm) / (2*eps), summed
g = jnp.asarray(rng.standard_normal(200000))
t1 = jax.jit(lambda g, fp, fm: jnp.sum(g * (fp - fm) / (2 * eps)))
t2 = jax.jit(lambda g, fp, fm, m: jnp.sum(g * (fp - fm) / jnp.where(m, 3.0, 2 * eps)))
fp = x; fm = x * 0.999
print("loop-form sum bit-identical:", float(t1(g, fp, fm)) == float(t2(g, fp, fm, False)), float(t1(g,fp,fm)), float(t2(g,fp,fm,False)))
