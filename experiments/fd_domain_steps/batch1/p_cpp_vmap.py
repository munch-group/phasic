"""OI-123 reproducer: a pmf_from_cpp model cannot be vmapped over particles
(eager and jit work). Run with PHASIC_SOURCE_DIR pointing at a checkout."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests", "pytest"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import phasic  # noqa: F401
import _fd_golden_cells as C

f = C.build("cpp")
L = C.loss_of(f)
th = jnp.asarray([0.7, 1.3])
print("eager grad:", jax.grad(L)(th))
print("jit grad:  ", jax.jit(jax.grad(L))(th))
try:
    print("vmap grad: ", jax.vmap(jax.grad(L))(jnp.stack([th, th * 1.1])))
except Exception as e:  # noqa: BLE001
    print("vmap grad RAISES:", str(e).replace("\n", " ")[-160:])
try:
    print("vmap fwd:  ", jax.vmap(f)(jnp.stack([th, th * 1.1])))
except Exception as e:  # noqa: BLE001
    print("vmap fwd RAISES:", str(e).replace("\n", " ")[-160:])
