import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
for n in (4, 8, 40, 50, 60, 100, 40000):
    for v in (0.3, 1e-8, 1.2e-4, 0.7):
        a = np.full(n, v)
        print(f'n={n:6d} v={v:g}: np.mean==v {a.mean() == v}  np.std={a.std():.2e}  jnp.mean==v {float(jnp.mean(jnp.asarray(a))) == v}  jnp.std={float(jnp.std(jnp.asarray(a))):.2e}')
