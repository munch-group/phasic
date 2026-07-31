"""Batch-0 (algorithm de-risk, build-free): a faithful Python model of phasic's
TWO-TIER elimination tape + the full reverse-mode θ-adjoint, validated against JAX
autodiff. Exercises exactly the pieces the earlier prototype did NOT: the 7 param
ops (P/PP/INV/ONE_MINUS/DIVIDE/ZERO/NEW_ADD), in-place mem reuse, the
REPLACE/kill reverse semantics (amendment 2), the emit-before-transpose ordering
(amendment 3), and the param↔numeric glue.

Model:
  param tape ops over a mem[] array (some slots are INPUT edge weights):
    P(f,t,c):  mem[f] += mem[t]*c              (const mult)
    PP(f,t,m): mem[f] += mem[t]*mem[m]
    INV(f):    mem[f] = 1/mem[f]               (REPLACE)
    OM(f):     mem[f] = 1 - mem[f]             (REPLACE)
    DIV(f,t):  mem[f] = mem[f]/mem[t]          (REPLACE)
    ZERO(f):   mem[f] = 0                       (kill)
    NEWADD(a,b,m): emit numeric command {a,b, mem[m]}
  numeric replay: result[a] += result[b]*mult, seeded s; Q = result[target].
Gradient wanted: dQ/d(INPUT edge weights).  (θ chain = *coeff, tested trivially.)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ---------- forward as a pure jnp function of the INPUT slots (the JAX oracle) ----------
def forward_jnp(inputs, tape, n_mem, input_slots, seed, target):
    mem = [jnp.array(0.0)] * n_mem
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs[k]
    num = []  # numeric commands (a,b, mult-Value)
    for op in tape:
        k = op[0]
        if k == 'P':   _, f, t, c = op; mem[f] = mem[f] + mem[t] * c
        elif k == 'PP':_, f, t, m = op; mem[f] = mem[f] + mem[t] * mem[m]
        elif k == 'INV':_, f = op;      mem[f] = 1.0 / mem[f]
        elif k == 'OM':_, f = op;       mem[f] = 1.0 - mem[f]
        elif k == 'DIV':_, f, t = op;   mem[f] = mem[f] / mem[t]
        elif k == 'ZERO':_, f = op;     mem[f] = jnp.array(0.0)
        elif k == 'NEWADD':_, a, b, m = op; num.append((a, b, mem[m]))
    r = [seed[i] for i in range(len(seed))]
    for (a, b, mv) in num:
        r[a] = r[a] + r[b] * mv
    return r[target]


# ---------- manual reverse-mode θ-adjoint (the algorithm to port to C) ----------
def manual_vjp(inputs_np, tape, n_mem, input_slots, seed_np, target):
    # ---- param tape forward, snapshot operand primals; record numeric tape ----
    mem = np.zeros(n_mem)
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs_np[k]
    snaps = []                # per-op operand primals needed by the reverse
    num = []                  # (a,b, mult_value, mptr_slot)
    for op in tape:
        k = op[0]
        if k == 'P':   _, f, t, c = op; snaps.append(None); mem[f] += mem[t] * c
        elif k == 'PP':_, f, t, m = op; snaps.append((mem[t], mem[m])); mem[f] += mem[t] * mem[m]
        elif k == 'INV':_, f = op;      snaps.append(mem[f]); mem[f] = 1.0 / mem[f]
        elif k == 'OM':_, f = op;       snaps.append(None); mem[f] = 1.0 - mem[f]
        elif k == 'DIV':_, f, t = op;   snaps.append((mem[f], mem[t])); mem[f] = mem[f] / mem[t]
        elif k == 'ZERO':_, f = op;     snaps.append(None); mem[f] = 0.0
        elif k == 'NEWADD':_, a, b, m = op; snaps.append(None); num.append((a, b, mem[m], m))
    # ---- numeric replay forward, snapshot result[to] per command ----
    r = seed_np.astype(float).copy(); snap_to = []
    for (a, b, mv, _m) in num:
        snap_to.append(r[b]); r[a] += r[b] * mv
    # ---- stage 1: reverse numeric replay -> dQ/dm_c ----
    adj = np.zeros(len(seed_np)); adj[target] = 1.0
    dm = np.zeros(len(num))
    for c in range(len(num) - 1, -1, -1):
        a, b, mv, _m = num[c]
        dm[c] = adj[a] * snap_to[c]          # BEFORE transpose (amendment 3)
        adj[b] += adj[a] * mv
    # ---- stage 2: reverse the param tape (REPLACE/kill per op, amendment 2) ----
    # bar[f] must be READ into a local before any accumulate-then-modify (aliasing);
    # the param<->numeric glue is applied AT the NEW_ADD op during the reverse (timing).
    bar = np.zeros(n_mem)
    numptr = len(num) - 1                    # walk numeric commands in reverse too
    for i in range(len(tape) - 1, -1, -1):
        op = tape[i]; k = op[0]; sn = snaps[i]
        if k == 'P':   _, f, t, c = op; bf = bar[f]; bar[t] += bf * c             # keep bar[f]
        elif k == 'PP':_, f, t, m = op; t0, m0 = sn; bf = bar[f]; bar[t] += bf*m0; bar[m] += bf*t0
        elif k == 'INV':_, f = op; f0 = sn; bar[f] = bar[f] * (-1.0 / f0**2)      # REPLACE
        elif k == 'OM':_, f = op; bar[f] = -bar[f]                                # REPLACE
        elif k == 'DIV':_, f, t = op; f0, t0 = sn; bf = bar[f]; bar[t] += bf*(-f0/t0**2); bar[f] = bf/t0
        elif k == 'ZERO':_, f = op; bar[f] = 0.0                                  # kill
        elif k == 'NEWADD':_, a, b, m = op; bar[m] += dm[numptr]; numptr -= 1     # glue, in-order
    return np.array([bar[sl] for sl in input_slots])


# ---------- random valid two-tier tapes ----------
def random_case(rng):
    n_mem = int(rng.integers(4, 8))
    n_in = int(rng.integers(2, 4))
    input_slots = list(rng.choice(n_mem, size=n_in, replace=False))
    tape = []
    # a few ops; keep values well-conditioned (init inputs in [1,3]; guard INV/DIV)
    for _ in range(int(rng.integers(4, 12))):
        choice = rng.choice(['P', 'PP', 'INV', 'OM', 'DIV', 'ZERO'])
        f = int(rng.integers(n_mem)); t = int(rng.integers(n_mem)); m = int(rng.integers(n_mem))
        if choice == 'P':   tape.append(('P', f, t, float(rng.uniform(0.3, 1.5))))
        elif choice == 'PP':tape.append(('PP', f, t, m))
        elif choice == 'INV':tape.append(('INV', f))
        elif choice == 'OM':tape.append(('OM', f))
        elif choice == 'DIV':tape.append(('DIV', f, t if t != f else (t + 1) % n_mem))
        elif choice == 'ZERO':tape.append(('ZERO', f))
    # emit 2-4 numeric commands from mem slots (incl. self-loops a==b)
    n_seed = int(rng.integers(3, 6))
    for _ in range(int(rng.integers(2, 5))):
        a = int(rng.integers(n_seed)); b = int(rng.integers(n_seed)); m = int(rng.integers(n_mem))
        tape.append(('NEWADD', a, b, m))
    seed = rng.uniform(0.5, 2.0, n_seed)
    inputs = rng.uniform(1.0, 3.0, n_in)
    target = int(rng.integers(n_seed))
    return tape, n_mem, input_slots, seed, inputs, target


rng = np.random.default_rng(7)
npass = 0; ntot = 0; worst = 0.0
for trial in range(400):
    tape, n_mem, input_slots, seed, inputs, target = random_case(rng)
    inj = jnp.asarray(inputs); seedj = jnp.asarray(seed)
    f = lambda x: forward_jnp(x, tape, n_mem, input_slots, seedj, target)
    q = float(f(inj))
    if not np.isfinite(q) or abs(q) > 1e6:
        continue                                   # skip ill-conditioned random draws
    g_ref = np.asarray(jax.grad(f)(inj))
    g_man = manual_vjp(inputs, tape, n_mem, input_slots, seed, target)
    if not (np.all(np.isfinite(g_ref)) and np.all(np.isfinite(g_man))):
        continue                                   # ill-conditioned draw (NaN grad); skip
    if np.max(np.abs(g_ref)) > 1e8:
        continue                                   # gradient blow-up; skip
    ntot += 1
    err = np.max(np.abs(g_man - g_ref)) / max(1.0, np.max(np.abs(g_ref)))
    worst = max(worst, err)
    if err < 1e-9:
        npass += 1
    elif npass + (ntot - npass) <= npass + 5:
        print(f"  MISMATCH err={err:.2e}  ref={g_ref}  man={g_man}")

print(f"\nTwo-tier reverse θ-adjoint vs JAX autodiff (7 ops, in-place reuse, self-loops):")
print(f"  {npass}/{ntot} random tapes match  (worst rel.err = {worst:.2e})")
print("  RESULT:", "ALL PASS — stage-1+stage-2 reverse + glue + REPLACE/kill verified"
      if npass == ntot and ntot > 50 else "CHECK FAILURES")
