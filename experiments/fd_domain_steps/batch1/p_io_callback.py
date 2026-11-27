"""Review m3: can the tie/zero report use io_callback(ordered=False) inside a
custom_vjp backward, under grad / jit / jit(vmap(grad)), without unrolling one
callback per particle into the compiled program?"""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

calls = []


def host(z, t):
    calls.append((np.asarray(z).shape, int(np.sum(z)), int(np.sum(t))))


def report_debug(z, t):
    jax.debug.callback(host, z, t)


def report_io(z, t):
    io_callback(host, None, z, t, ordered=False)


def make(report):
    @jax.custom_vjp
    def f(x):
        return jnp.sum(x ** 2)

    def fwd(x):
        return f(x), x

    def bwd(x, g):
        z = jnp.stack([x[0] > 0, x[1] > 0]); t = jnp.stack([x[0] > 1, x[1] > 1])
        report(z, t)
        return (2 * g * x,)
    f.defvjp(fwd, bwd)
    return f


for name, rep in (("debug.callback", report_debug), ("io_callback", report_io)):
    f = make(rep)
    calls.clear()
    g1 = jax.grad(f)(jnp.asarray([2.0, -1.0]))
    g2 = jax.jit(jax.grad(f))(jnp.asarray([2.0, -1.0]))
    for n in (8, 256):
        P = jnp.ones((n, 2)) * jnp.arange(n)[:, None]
        fn = jax.jit(jax.vmap(jax.grad(f)))
        hlo = fn.lower(P).as_text()
        t0 = time.time(); G = fn(P); G.block_until_ready(); t1 = time.time()
        print(f"{name}: n={n} callbacks-in-HLO={hlo.count('custom_call_target=\"xla_python_cpu_callback\"') + hlo.count('xla_ffi_python_cpu_callback')}"
              f" first-call {t1-t0:.3f}s grads-ok={np.allclose(np.asarray(G), 2*np.asarray(P))}")
    print(f"{name}: eager/jit grads {np.asarray(g1)} {np.asarray(g2)}; host calls {len(calls)}; sample {calls[:2]} ... {calls[-1]}")
