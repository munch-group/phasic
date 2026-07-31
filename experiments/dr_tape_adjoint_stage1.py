"""De-risk the crux of the C plan: reverse-mode over the numeric tape.

Numeric replay: result[from] += result[to]*m, in order, seeded with s; Q=result[target].
Claim to verify: the multiplier gradient is
    dQ/dm_c = adjoint[from_c] * result_primal[to_c]
where adjoint[from_c] is read BEFORE applying command c's transpose in the reverse
walk (adjoint[to]+=adjoint[from]*m), and result_primal[to_c] is result[to_c] captured
BEFORE command c executes in the forward pass. And the seed gradient dQ/ds is the
plain reverse seed-walk (the existing sojourn adjoint).

Validated against JAX autodiff on random tapes INCLUDING from==to self-loop
commands and aliased slots (both present in phasic's real tape).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def forward_jax(seed, mult, tape, target):
    """result[from] += result[to]*m ; Q = result[target]. Pure jnp (autodiff ref)."""
    r = seed
    for c, (frm, to, _) in enumerate(tape):
        r = r.at[frm].add(r[to] * mult[c])
    return r[target]


def manual_adjoint(seed_np, mult_np, tape, target):
    n = len(seed_np); m = len(tape)
    # forward, snapshotting result[to] BEFORE each command
    r = seed_np.astype(float).copy()
    snap_to = np.empty(m)
    for c, (frm, to, _) in enumerate(tape):
        snap_to[c] = r[to]                     # result[to] at command c's time
        r[frm] += r[to] * mult_np[c]
    # reverse walk
    adj = np.zeros(n); adj[target] = 1.0
    dm = np.zeros(m)
    for c in range(m - 1, -1, -1):
        frm, to, _ = tape[c]
        dm[c] = adj[frm] * snap_to[c]          # BEFORE the transpose update
        adj[to] += adj[frm] * mult_np[c]       # seed-adjoint transpose
    return dm, adj                             # adj is dQ/dseed


def check(tape, n, target, seed, mult, tag):
    seed_j = jnp.asarray(seed, float); mult_j = jnp.asarray(mult, float)
    dm_ref = np.asarray(jax.grad(lambda mm: forward_jax(seed_j, mm, tape, target))(mult_j))
    ds_ref = np.asarray(jax.grad(lambda ss: forward_jax(ss, mult_j, tape, target))(seed_j))
    dm_man, ds_man = manual_adjoint(np.asarray(seed), np.asarray(mult), tape, target)
    em = np.max(np.abs(dm_man - dm_ref)); es = np.max(np.abs(ds_man - ds_ref))
    ok = em < 1e-9 and es < 1e-9
    print(f"  [{'OK' if ok else 'FAIL'}] {tag:38s}  max|dm-ref|={em:.2e}  max|ds-ref|={es:.2e}")
    return ok


print("Verifying dQ/dm_c = adjoint[from_c]*result[to_c] and dQ/ds = seed-walk:")
allok = True

# 1) hand example with aliasing (cmd2 re-reads slot 1 after cmd1 updated it; slot0 updated twice)
allok &= check([(0,1,0),(1,2,0),(0,1,0)], 3, 0, [2.,3.,5.], [0.7,1.3,0.9], "aliased 3-command")

# 2) self-loop command from==to (the 1/(1-q) cycle correction add_command(parent,parent,.))
allok &= check([(0,1,0),(1,1,0),(0,1,0)], 3, 0, [1.,2.,4.], [0.5, 0.3, 1.1], "self-loop from==to")

# 3) chain of self-loops + aliasing
allok &= check([(2,2,0),(1,2,0),(1,1,0),(0,1,0),(0,0,0)], 3, 0,
               [1.,1.,1.], [0.2,0.6,0.4,0.8,0.1], "self-loops + chain")

# 4) random tapes (incl. self-loops and repeated slots), several sizes/targets
rng = np.random.default_rng(0)
for trial in range(12):
    n = int(rng.integers(3, 7)); m = int(rng.integers(4, 14))
    tape = [(int(rng.integers(0, n)), int(rng.integers(0, n)), 0) for _ in range(m)]
    target = int(rng.integers(0, n))
    seed = rng.uniform(0.5, 3.0, n); mult = rng.uniform(-1.5, 1.5, m)
    allok &= check(tape, n, target, seed, mult, f"random n={n} m={m} tgt={target}")

print("\nRESULT:", "ALL PASS — formula + timing verified" if allok else "SOME FAILED")
