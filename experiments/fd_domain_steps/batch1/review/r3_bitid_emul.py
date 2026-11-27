"""Review probe r3: bit identity of undeclared slots OUTSIDE the band, against an
emulated master (slot_rule None -> TODAY reproduces master's traced program for
undeclared slots), over a wider theta set than the goldens, in eager / jit /
vmap(grad) / jit(vmap(grad)) (SVGD's mode), including vmap batches that MIX an
in-band particle with out-of-band ones (the out-of-band rows must be unchanged),
and thetas just above the band where today's 1e-9 clamp fires (1e-7 < t < 1.01e-7)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../tests/pytest"))
import phasic
import jax, jax.numpy as jnp
import numpy as np
from phasic import _fd_steps
import _fd_golden_cells as C

_orig = _fd_steps.slot_rule


def master_rule(domain, refuses):
    return _fd_steps.TODAY if domain is None else _orig(domain, refuses)


def grads(name, rows, mode, emulate):
    _fd_steps.slot_rule = master_rule if emulate else _orig
    try:
        L = C.loss_of(C.build(name))
        X = jnp.asarray(rows, dtype=jnp.float64)
        if mode == "eager":
            return np.stack([np.asarray(jax.grad(L)(x)) for x in X])
        if mode == "jit":
            g = jax.jit(jax.grad(L))
            return np.stack([np.asarray(g(x)) for x in X])
        if mode == "vmap":
            return np.asarray(jax.vmap(jax.grad(L))(X))
        if mode == "jitvmap":
            return np.asarray(jax.jit(jax.vmap(jax.grad(L)))(X))
    finally:
        _fd_steps.slot_rule = _orig


CASES = {
    # refusing models
    "pfg_log": [[0.7, 1.3], [1.01e-7, 0.5], [3e-3, 2e-6], [1e3, 0.4]],
    "pfg_formula": [[0.7, 1.3], [1.005e-7, 2.0], [2e-7, 1e-6]],
    "pfg_dph": [[0.3, 0.4], [1.005e-7, 0.2], [0.1, 1.5e-7]],
    "pam_log": [[0.7, 1.3], [1.005e-7, 0.5], [0.4, 1.0001e-7]],
    "pam_formula": [[0.7, 1.3], [1.005e-7, 0.5], [0.4, 1.009e-7]],
    "dc_noexp": [[1.0, 0.5, 1.5, 0.3], [1.0, 1.5e-7, 2.0, 1.01e-7], [3e-6, 1e-4, 1e-4, 1e-4]],
    "dc_exp": [[1.0, 0.5, 1.5, 0.3], [1.0, 1.5e-7, 2.0, 1.01e-7]],
    "dc_efg": [[1.0, 0.5, 1.5, 0.3], [1.0, 1.5e-7, 2.0, 1.01e-7]],
    "dcjp": [[1.0, 0.5, 1.5, 0.3], [1.0, 1.5e-7, 2.0, 1.01e-7]],
    "mfg": [[0.7, 1.3], [1.01e-7, 0.5]],
    # non-refusing: must be the identical program
    "pfg_linear": [[0.7, 1.3], [5e-8, 0.3], [0.0, 0.4]],
    "pam_linear": [[0.7, 1.3], [5e-8, 0.3]],
    "jix": [[1.0, 0.5], [1e-8, 0.5]],
    "rvp_linear": [[0.7, 1.3], [5e-8, 0.3]],
}
MIXED = {   # (rows, index of the in-band row) -- other rows must be unchanged
    "dc_noexp": ([[1.0, 0.5, 1.5, 0.3], [1.0, 1e-8, 1.0, 1e-8], [2.0, 0.4, 1.1, 0.6]], 1),
    "pfg_log": ([[0.7, 1.3], [0.7, 5e-8], [1.1, 0.2]], 1),
    "pam_formula": ([[0.7, 1.3], [0.7, 5e-8], [1.1, 0.2]], 1),
}

bad = 0
for name, rows in CASES.items():
    for mode in ("eager", "jit", "vmap", "jitvmap"):
        if name in C.NO_VMAP and "vmap" in mode:
            continue
        try:
            a = grads(name, rows, mode, True); b = grads(name, rows, mode, False)
        except Exception as e:
            print(f"{name:12s} {mode:8s} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
            bad += 1
            continue
        ok = a.tobytes() == b.tobytes()
        bad += (not ok)
        print(f"{name:12s} {mode:8s} bit-identical={ok}" + ("" if ok else f"  maxrel={np.max(np.abs(a-b)/np.maximum(np.abs(a),1e-300)):.3g}"), flush=True)

for name, (rows, k) in MIXED.items():
    for mode in ("vmap", "jitvmap"):
        b = grads(name, rows, mode, False)
        keep = [i for i in range(len(rows)) if i != k]
        a = grads(name, [rows[i] for i in keep], mode, True)
        ok = a.tobytes() == b[keep].tobytes()
        bad += (not ok)
        print(f"MIXED {name:12s} {mode:8s} out-of-band rows bit-identical={ok}; in-band row finite={np.all(np.isfinite(b[k]))}", flush=True)
print("TOTAL mismatches/errors:", bad)
