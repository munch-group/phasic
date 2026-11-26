import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
seen=[]
@jax.custom_vjp
def f(x): return jnp.sum(x**2)
def fwd(x): return f(x), x
def bwd(x, g):
    seen.append(type(x).__name__)
    return (2*g*x,)
f.defvjp(fwd,bwd)
jax.grad(f)(jnp.array([1.,2.]))
jax.jit(jax.grad(f))(jnp.array([1.,2.]))
jax.vmap(jax.grad(f))(jnp.ones((3,2)))
print(seen)
