"""Batch F / F0 go/no-go: does a RuntimeError raised inside a
jax.pure_callback host function surface LEGIBLY under the compositions SVGD
actually runs — vmap(jit(grad(f))) and jit(vmap(grad(f)))?

dr_lax_cond_vmap_derisk.py's claim 3 tested grad and vmap(grad) only (no
jit anywhere) — the D6.1 review flagged the gap. If the diagnostic message
is NOT findable in str(e) here, the committed-raise design must go back to
the user (plan amendment 2). Build-free: jax only, no phasic import.
Exits 0 = GO, 1 = NO-GO."""
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

MARKER = "PHASIC-F0-DIAGNOSTIC: exact sojourn gradient declined at theta"


def _host(theta_np):
    raise RuntimeError(f"{MARKER}={np.asarray(theta_np).tolist()}")


@jax.custom_vjp
def f(theta):
    return jnp.sum(theta ** 2)

def f_fwd(theta):
    return f(theta), theta

def f_bwd(theta, g):
    out = jax.pure_callback(
        _host, jax.ShapeDtypeStruct(theta.shape, theta.dtype), theta,
        vmap_method='sequential')
    return (out * g,)

f.defvjp(f_fwd, f_bwd)

batch = jnp.asarray(np.random.default_rng(0).uniform(0.5, 2.0, (4, 2)))
ok = True
for name, fn in [
    ("vmap(jit(grad))", jax.vmap(jax.jit(jax.grad(f)))),
    ("jit(vmap(grad))", jax.jit(jax.vmap(jax.grad(f)))),
]:
    try:
        np.asarray(fn(batch))
        print(f"{name}: [NO-GO] no exception raised at all")
        ok = False
    except BaseException as e:
        legible = MARKER in str(e) or (e.__cause__ is not None and MARKER in str(e.__cause__))
        print(f"{name}: exception {type(e).__name__}; marker "
          f"{'FOUND -> GO' if legible else 'NOT FOUND -> NO-GO'}")
        if not legible:
            print("   str(e) head:", str(e)[:300].replace(chr(10), ' | '))
        ok &= legible

print("F0: GO (raise design viable under jit)" if ok else "F0: NO-GO (present to user)")
sys.exit(0 if ok else 1)
