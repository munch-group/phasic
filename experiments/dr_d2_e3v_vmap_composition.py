"""Deferred-2 de-risk E3(v) -- the exact-VJP x custom_vmap composition
probe: "the actual hard JAX problem" (plan §4-E3 gate 5, review F3).

WHY THIS IS THE BLOCKING RISK. The production daisy sites
(`_daisy_chain_svgd_model`, src/phasic/__init__.py:5111-5200) wrap the
per-particle forward in `custom_vmap` so a batched call fuses into ONE
fat 2-D FFI call, and wrap THAT in `custom_vjp` whose FD backward calls
the same custom_vmap'd core at theta+-eps -- so the FD probes inherit
the batching rule for free. Batch H's exact FINAL-epoch term dodges the
problem a second way: its block is a construction-time numpy CONSTANT
(`_efg_block_exp_np`), which vmap simply broadcasts.

An exact INTERMEDIATE-epoch gradient can do NEITHER: it depends on the
per-particle theta, so it must be a PER-CALL host callback inside the
backward, executed under `vmap(grad(loss))(particles)`. Nothing in the
shipped codebase does that yet. If that composition is broken or
silently wrong, Deferred 2 is blocked regardless of the C math.

The probe reproduces the production skeleton exactly (custom_vmap core
with a 1-D path and a fused batched rule; custom_vjp on top) but swaps
the FD backward for an EXACT one implemented as `jax.pure_callback`
(the stand-in for a C adjoint), and checks:
  P1  vmap(grad(loss)) == analytic truth, per particle
  P2  the callback observes ONE PARTICLE per invocation (1-D theta),
      i.e. vmap pushes the batch through the callback correctly
  P3  jit(vmap(grad(loss))) also correct (compiled path)
  P4  vmap(grad()) of the FD-backward variant agrees -- so the probe's
      exact backward is measured against the mechanism production
      actually ships, not only against algebra
  P5  the fused custom_vmap rule really fires under grad (the FFI
      would reject a 3-D buffer, so a silent fall-through to the
      default batching rule is a real failure mode)
"""
import sys

import numpy as np

import phasic  # noqa: F401  (import-time JAX x64 config)
from phasic import set_log_level

set_log_level("WARNING")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import custom_batching as _cb  # noqa: E402  (production idiom)

FAILS = []
D = 3            # theta_dim
K = 4            # the internal n_unique batch (production: unique exposures)
SCALE = np.linspace(1.0, 1.6, K)[:, None] * np.ones((1, D))   # (K, D)
W = np.array([0.5, -1.25, 2.0])                                # readout weights


def check(label, ok, detail=""):
    print(f"  {label}: {'PASS' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(label)


# --------------------------------------------------------------- the "C" side
# Stand-in for the FFI forward: a nonlinear map of a (B, D) theta batch
# to (B,) outputs. Deliberately NOT expressible in JAX (numpy only, via
# callback) -- exactly like the real FFI.
def c_forward_np(theta_2d):
    t = np.asarray(theta_2d, dtype=np.float64)
    assert t.ndim == 2, f"C forward received ndim={t.ndim} (production FFI rejects 3-D)"
    return np.sum(np.sin(t) * W[None, :], axis=1)          # (B,)


def c_jacobian_np(theta_1d):
    """Stand-in for the exact C adjoint: d(out_k)/d(theta_j) for one
    particle across the K internal rows -> (K, D)."""
    t = np.asarray(theta_1d, dtype=np.float64)
    assert t.ndim == 1, f"C adjoint received ndim={t.ndim} (expected one particle)"
    tk = t[None, :] * SCALE                                 # (K, D)
    return np.cos(tk) * W[None, :] * SCALE                  # (K, D)


CALLBACK_NDIMS = []          # records what the adjoint callback observed
FORWARD_SHAPES = []          # records the FFI batch shapes seen


def _ffi(theta_2d):
    """jax.pure_callback wrapper of the 'FFI' forward."""
    def _host(t):
        t = np.asarray(t)
        FORWARD_SHAPES.append(t.shape)
        return c_forward_np(t)
    out_shape = jax.ShapeDtypeStruct((theta_2d.shape[0],), jnp.float64)
    return jax.pure_callback(_host, out_shape, theta_2d, vmap_method='sequential')


# ------------------------------------------------- production skeleton (copied)
@_cb.custom_vmap
def _core(theta_flat):
    # 1-D path: build the (K, D) internal batch, one fat call.
    theta_pk = theta_flat[None, :] * jnp.asarray(SCALE)
    return _ffi(theta_pk)                                    # (K,)


@_core.def_vmap
def _core_vmap_rule(axis_size, in_batched, theta_flat):
    del in_batched
    P = axis_size
    theta_pk = (theta_flat[:, None, :] * jnp.asarray(SCALE)[None, :, :]
                ).reshape(P * K, D)
    out = _ffi(theta_pk).reshape(P, K)
    return out, True


@jax.custom_vjp
def _autodiff(theta_flat):
    return _core(theta_flat)


def _fwd(theta_flat):
    return _core(theta_flat), theta_flat


def _bwd_exact(theta_flat, cotangent):
    """EXACT backward via a PER-CALL host callback (the thing production
    has never done under vmap). Returns J^T . cotangent."""
    def _host(t, ct):
        t = np.asarray(t)
        CALLBACK_NDIMS.append(t.ndim)
        J = c_jacobian_np(t)              # (K, D)
        return (np.asarray(ct)[None, :] @ J).reshape(-1)   # (D,)
    out_shape = jax.ShapeDtypeStruct((D,), jnp.float64)
    g = jax.pure_callback(_host, out_shape, theta_flat, cotangent,
                          vmap_method='sequential')
    return (g,)


_autodiff.defvjp(_fwd, _bwd_exact)


# FD variant -- the mechanism production actually ships, for P4
@jax.custom_vjp
def _autodiff_fd(theta_flat):
    return _core(theta_flat)


def _fwd_fd(theta_flat):
    return _core(theta_flat), theta_flat


def _bwd_fd(theta_flat, cotangent):
    eps = 1e-6
    gs = []
    for i in range(D):
        jp = _core(theta_flat.at[i].add(eps))
        jm = _core(theta_flat.at[i].add(-eps))
        gs.append(jnp.sum(cotangent * (jp - jm) / (2 * eps)))
    return (jnp.stack(gs),)


_autodiff_fd.defvjp(_fwd_fd, _bwd_fd)


def loss(theta_flat, fn=_autodiff):
    out = fn(theta_flat)
    return jnp.sum(out * jnp.arange(1.0, K + 1.0))


def analytic_grad(theta_1d):
    J = c_jacobian_np(theta_1d)                      # (K, D)
    ct = np.arange(1.0, K + 1.0)
    return ct[None, :] @ J


PARTICLES = np.array([[0.3, 1.1, -0.4],
                      [1.7, 0.2, 0.9],
                      [-0.8, 0.5, 1.3]])

print("== E3(v): exact-VJP x custom_vmap composition probe ==")

# P1 -- vmap(grad(loss)) with the EXACT per-call-callback backward
CALLBACK_NDIMS.clear(); FORWARD_SHAPES.clear()
g_vmap = np.asarray(jax.vmap(jax.grad(loss))(jnp.asarray(PARTICLES)))
g_true = np.vstack([analytic_grad(p) for p in PARTICLES])
rel = np.max(np.abs(g_vmap - g_true)) / max(np.max(np.abs(g_true)), 1e-300)
check("P1 vmap(grad(loss)) == analytic truth (exact per-call callback)",
      rel < 1e-12, f"(max rel {rel:.2e})")

# P2 -- what did the adjoint callback observe?
ndims = set(CALLBACK_NDIMS)
check("P2 adjoint callback sees ONE PARTICLE per call (ndim==1)",
      ndims == {1}, f"(observed ndims {sorted(ndims)}, "
                    f"{len(CALLBACK_NDIMS)} invocations for "
                    f"{PARTICLES.shape[0]} particles)")

# P5 -- the FFI never saw a 3-D buffer, and the fused rule fired
maxdim = max((len(s) for s in FORWARD_SHAPES), default=0)
fused = any(s == (PARTICLES.shape[0] * K, D) for s in FORWARD_SHAPES)
check("P5 forward stayed 2-D (no 3-D buffer) and the FUSED rule fired",
      maxdim == 2 and fused,
      f"(shapes {sorted(set(FORWARD_SHAPES))})")

# P3 -- compiled path
g_jit = np.asarray(jax.jit(jax.vmap(jax.grad(loss)))(jnp.asarray(PARTICLES)))
rel_jit = np.max(np.abs(g_jit - g_true)) / max(np.max(np.abs(g_true)), 1e-300)
check("P3 jit(vmap(grad(loss))) == analytic truth", rel_jit < 1e-12,
      f"(max rel {rel_jit:.2e})")

# P4 -- agreement with the shipped FD mechanism
g_fd = np.asarray(jax.vmap(jax.grad(lambda t: loss(t, fn=_autodiff_fd)))(
    jnp.asarray(PARTICLES)))
rel_fd = np.max(np.abs(g_fd - g_true)) / max(np.max(np.abs(g_true)), 1e-300)
check("P4 shipped-style FD backward agrees (sanity on the skeleton)",
      rel_fd < 1e-6, f"(max rel {rel_fd:.2e}) -- note the exact path is "
                     f"{rel_fd / max(rel, 1e-300):.0e}x more accurate")

print()
print("E3(v) COMPLETE" + ("" if not FAILS else f"; FAILURES: {FAILS}"))
sys.exit(0 if not FAILS else 1)
