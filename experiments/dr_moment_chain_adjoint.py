"""Batch-3 de-risk (build-free): the HIGHER-MOMENT reverse chain over phasic's
moment recurrence, validated against JAX autodiff.

Recurrence (graph_builder.cpp:512-550, standard moments):
    a_1 = replay(tape, seed=ones)          # ewt(ones)
    a_{j+1} = replay(tape, seed=a_j)       # ewt(a_j); SAME numeric tape, new seed
    m_k = (k+1)! * a_{k+1}[target]         # k = 0 .. K-1

Every replay shares ONE numeric tape (multipliers depend on theta/inputs, NOT on
the seed). The reverse is therefore a CHAIN: reversing replay j with an output
cotangent bar_a_j yields (i) dm[c] += adj[from]*snap_to_j[c] contributions into a
SHARED dm[] and (ii) the seed-adjoint adj[] = bar on a_{j-1}, which becomes replay
j-1's output cotangent (plus its own j!*g_{j-1} at [target]). Stage-2 (param tape
reverse -> edge grads) runs ONCE on the accumulated dm[]. This file models the
two-tier tape + the FULL moment-chain reverse and checks it vs jax.jacobian.
"""
import math, warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ---- forward as a pure jnp fn of the INPUT slots (the JAX oracle) ----
def forward_jnp(inputs, tape, n_mem, input_slots, n_seed, target, K):
    mem = [jnp.array(0.0)] * n_mem
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs[k]
    num = []
    for op in tape:
        t = op[0]
        if t == 'P':   _, f, tt, c = op; mem[f] = mem[f] + mem[tt] * c
        elif t == 'PP':_, f, tt, m = op; mem[f] = mem[f] + mem[tt] * mem[m]
        elif t == 'INV':_, f = op;      mem[f] = 1.0 / mem[f]
        elif t == 'OM':_, f = op;       mem[f] = 1.0 - mem[f]
        elif t == 'DIV':_, f, tt = op;  mem[f] = mem[f] / mem[tt]
        elif t == 'ZERO':_, f = op;     mem[f] = jnp.array(0.0)
        elif t == 'NEWADD':_, a, b, m = op; num.append((a, b, mem[m]))
    def replay(seed):
        r = list(seed)
        for (a, b, mv) in num:
            r[a] = r[a] + r[b] * mv
        return r
    a = replay([jnp.array(1.0)] * n_seed)       # a_1 = ewt(ones)
    moments = []
    for k in range(K):                          # a_{k+1}, m_k
        if k > 0:
            a = replay(a)
        moments.append(math.factorial(k + 1) * a[target])
    return jnp.stack(moments)


# ---- manual reverse-mode over the moment chain (the algorithm to port to C) ----
def manual_jac(inputs_np, tape, n_mem, input_slots, n_seed, target, K):
    # param-tape forward: snapshot operand primals; record numeric commands
    mem = np.zeros(n_mem)
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs_np[k]
    snaps, num = [], []
    for op in tape:
        t = op[0]
        if t == 'P':   _, f, tt, c = op; snaps.append(None); mem[f] += mem[tt] * c
        elif t == 'PP':_, f, tt, m = op; snaps.append((mem[tt], mem[m])); mem[f] += mem[tt]*mem[m]
        elif t == 'INV':_, f = op;      snaps.append(mem[f]); mem[f] = 1.0/mem[f]
        elif t == 'OM':_, f = op;       snaps.append(None); mem[f] = 1.0 - mem[f]
        elif t == 'DIV':_, f, tt = op;  snaps.append((mem[f], mem[tt])); mem[f] = mem[f]/mem[tt]
        elif t == 'ZERO':_, f = op;     snaps.append(None); mem[f] = 0.0
        elif t == 'NEWADD':_, a, b, m = op; snaps.append(None); num.append((a, b, mem[m], m))

    # forward moment chain: store per-replay result vectors + per-replay snap_to
    seeds = [np.ones(n_seed)]                    # seed of replay 1
    snap_tos = []
    a = None
    for j in range(1, K + 1):
        seed = seeds[-1]
        r = seed.astype(float).copy(); st = []
        for (a_, b_, mv, _m) in num:
            st.append(r[b_]); r[a_] += r[b_] * mv
        snap_tos.append(st)                      # snap_to for replay j
        seeds.append(r)                          # a_j = r ; becomes seed of replay j+1
        a = r

    # cotangent on outputs: full Jacobian => loop one-hot over the K moments
    m_c = len(num)
    J = np.zeros((K, len(input_slots)))
    for out in range(K):
        gbar = np.zeros(K); gbar[out] = 1.0
        dm = np.zeros(m_c)
        bar_out = np.zeros(n_seed)               # bar on a_j for the current j
        # reverse the chain j = K .. 1
        for j in range(K, 0, -1):
            # bar_a_j gets j! * gbar[j-1] at target, plus seed-adjoint from j+1
            bar_a = bar_out.copy()
            bar_a[target] += math.factorial(j) * gbar[j - 1]
            # reverse replay j: dm contributions + seed-adjoint
            adj = bar_a.copy(); st = snap_tos[j - 1]
            for c in range(len(num) - 1, -1, -1):
                a_, b_, mv, _m = num[c]
                dm[c] += adj[a_] * st[c]         # accumulate across replays
                adj[b_] += adj[a_] * mv
            bar_out = adj                         # seed-adjoint -> replay j-1's output cotangent
        # stage-2: reverse the param tape ONCE on the accumulated dm -> edge grads
        bar = np.zeros(n_mem); numptr = len(num) - 1
        for i in range(len(tape) - 1, -1, -1):
            op = tape[i]; t = op[0]; sn = snaps[i]
            if t == 'P':   _, f, tt, c = op; bf = bar[f]; bar[tt] += bf * c
            elif t == 'PP':_, f, tt, m = op; t0, m0 = sn; bf = bar[f]; bar[tt] += bf*m0; bar[m] += bf*t0
            elif t == 'INV':_, f = op; f0 = sn; bar[f] = bar[f]*(-1.0/f0**2)
            elif t == 'OM':_, f = op; bar[f] = -bar[f]
            elif t == 'DIV':_, f, tt = op; f0, t0 = sn; bf = bar[f]; bar[tt] += bf*(-f0/t0**2); bar[f] = bf/t0
            elif t == 'ZERO':_, f = op; bar[f] = 0.0
            elif t == 'NEWADD':_, a_, b_, m = op; bar[m] += dm[numptr]; numptr -= 1
        J[out] = np.array([bar[sl] for sl in input_slots])
    return J


def random_case(rng):
    n_mem = int(rng.integers(4, 8)); n_in = int(rng.integers(2, 4))
    input_slots = list(rng.choice(n_mem, size=n_in, replace=False))
    tape = []
    for _ in range(int(rng.integers(4, 10))):
        ch = rng.choice(['P', 'PP', 'INV', 'OM', 'DIV', 'ZERO'])
        f = int(rng.integers(n_mem)); t = int(rng.integers(n_mem)); m = int(rng.integers(n_mem))
        if ch == 'P':   tape.append(('P', f, t, float(rng.uniform(0.3, 1.5))))
        elif ch == 'PP':tape.append(('PP', f, t, m))
        elif ch == 'INV':tape.append(('INV', f))
        elif ch == 'OM':tape.append(('OM', f))
        elif ch == 'DIV':tape.append(('DIV', f, t if t != f else (t + 1) % n_mem))
        elif ch == 'ZERO':tape.append(('ZERO', f))
    n_seed = int(rng.integers(3, 6))
    for _ in range(int(rng.integers(2, 5))):
        a = int(rng.integers(n_seed)); b = int(rng.integers(n_seed)); m = int(rng.integers(n_mem))
        tape.append(('NEWADD', a, b, m))
    inputs = rng.uniform(1.0, 3.0, n_in)
    target = int(rng.integers(n_seed))
    K = int(rng.integers(2, 5))
    return tape, n_mem, input_slots, n_seed, inputs, target, K


rng = np.random.default_rng(11)
npass = ntot = 0; worst = 0.0
for trial in range(400):
    tape, n_mem, input_slots, n_seed, inputs, target, K = random_case(rng)
    inj = jnp.asarray(inputs)
    f = lambda x: forward_jnp(x, tape, n_mem, input_slots, n_seed, target, K)
    m = np.asarray(f(inj))
    if not np.all(np.isfinite(m)) or np.max(np.abs(m)) > 1e6:
        continue
    J_ref = np.asarray(jax.jacobian(f)(inj))
    J_man = manual_jac(inputs, tape, n_mem, input_slots, n_seed, target, K)
    if not (np.all(np.isfinite(J_ref)) and np.all(np.isfinite(J_man))) or np.max(np.abs(J_ref)) > 1e8:
        continue
    ntot += 1
    err = np.max(np.abs(J_man - J_ref)) / max(1.0, np.max(np.abs(J_ref)))
    worst = max(worst, err)
    if err < 1e-9:
        npass += 1
    elif ntot - npass <= 5:
        print(f"  MISMATCH K={K} err={err:.2e}\n   ref={J_ref}\n   man={J_man}")

print(f"\nMoment-chain reverse Jacobian d[m_0..m_K-1]/d[inputs] vs JAX autodiff:")
print(f"  {npass}/{ntot} random cases match  (worst rel.err = {worst:.2e})")
print("  RESULT:", "ALL PASS — moment-chain seed-adjoint reverse verified"
      if npass == ntot and ntot > 50 else "CHECK FAILURES")
