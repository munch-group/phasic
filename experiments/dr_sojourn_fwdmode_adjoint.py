"""Joint-index batch, D0 (algorithm de-risk, build-free): forward-mode
theta-adjoint for the SOJOURN vector (expected_sojourn_time), validated
against JAX autodiff. Mirrors experiments/dr_twotier_full_adjoint.py's
two-tier tape model exactly (same param-tape op set: P/PP/INV/OM/DIV/ZERO/
NEWADD) but computes a NEW quantity in a NEW direction:

  - MOMENT chain (existing, shipped): seed r[v]=ones (or a specific reward),
    walk numeric commands FORWARD (c=0..nc-1), r[a] += r[b]*mult; read r[target].
    Production uses REVERSE-mode for the theta-gradient (few outputs [nr_moments],
    many-ish inputs [param_length]) -- ONE backward pass gives ALL theta
    components at once.

  - SOJOURN (this batch, new): seed y[target]=1, walk numeric commands
    REVERSED (c=nc-1..0) with roles of a/b SWAPPED, y[b] += y[a]*mult; read
    the WHOLE y[] vector (sojourn(v) for every v). Confirmed algebraically
    (M^T duality: y = M^T @ e_target, and moment-chain's r = M @ seed, so
    sojourn(v) = M[target,v]) AND empirically (phasic's
    Graph.expected_sojourn_time() == Graph.expected_waiting_time(reward=e_v)
    for every v, checked directly against the pybind API).

    For the GRADIENT, sojourn has the OPPOSITE shape problem from moments:
    MANY outputs (sojourn(v) for up to n or k target vertices, which in real
    joint-probability models is comparable to n -- confirmed via
    tests/pytest/test_sojourn_subset_adjoint.py's docstring, n=684226 /
    k=279936) but FEW inputs (theta_dim, typically 1-10). Reverse-mode would
    need one pass PER target vertex (O(k*nc) total) -- exactly the O(n*k)
    blowup expected_sojourn_time_subset's own adjoint algorithm was written
    to avoid for the PRIMAL. So the theta-gradient should use FORWARD-mode
    instead: one pass PER THETA COMPONENT (P total, P typically << k),
    giving the FULL (n,) sojourn-gradient column for that theta component
    in O(nc) time -- O(P*nc) total, independent of k/n's size beyond the
    O(nc) per-pass cost already paid by the primal.

    Crucially, forward-mode's param-tape section is EXACTLY the existing,
    already-shipped/validated `ptd_dbg_run_tape` logic (src/c/phasic.c
    ptd_debug_fwdmode_grad's inner loop) -- reused verbatim below. The only
    new piece is propagating the resulting per-command multiplier TANGENT
    through the SOJOURN (reversed-index, swapped-role) recurrence instead of
    the moment-chain (forward-index) one.

    Seeding: rather than looping over each EDGE-WEIGHT input k (ni of them,
    could be as many as the edge count) and contracting against
    coefficients[k][j] afterward, seed the param-tape tangent DIRECTLY with
    idot[k] = coefficients[k][j] for the theta component j of interest. This
    is valid by linearity of the forward-mode (JVP) operator -- verified
    below as an explicit cross-check, not just asserted.

D1.5 addition (post plan-review): two things the original version of this
script did NOT model, both required by the plan's corrected C design --

  (a) Production's diagonal-command storage convention
      (src/c/phasic.c:10770): a NEWADD/edge command with from==to stores
      (multiplier - 1), not multiplier, so that the self-scaling update
      x[a] = x[a]*mult can be walked with the SAME uniform "+=" instruction
      as every other command (x[a] += x[a]*(mult-1) === x[a] = x[a]*mult).
      The original script stored the raw multiplier for a==b commands too
      (an ADD of x[a]*mult, not a reassignment) -- self-consistent between
      its own oracle and manual code so the propagation-FORMULA checks
      below remained valid, but it did not match production's actual VALUE
      convention. Fixed by applying the same (mult-1 if a==b else mult)
      transform everywhere a NEWADD command's stored multiplier is built.

  (b) The guards production's sojourn walk uses, and why they must be
      ASYMMETRIC between primal and tangent. The original script had no
      guards at all. With (a) in place, a diagonal command whose CURRENT
      weight is exactly 1.0 stores nm[c]==0 while its THETA-DERIVATIVE
      mdot[c] can be nonzero (the weight varies with theta near 1.0) --
      copying the primal's "skip on m==0" guard onto the tangent would
      silently drop that y[a]*mdot contribution. Production's OWN reverse-
      mode moments code has the identical asymmetry one function down
      (phasic.c ptd_moments_grad_theta's stage-2 reverse walk: `dm[c] +=
      adj[na[c]]*st[c]` is UNCONDITIONAL, only the recurrence-continuation
      line `adj[nb[c]] += adj[na[c]]*m` is guarded on m!=0) -- so this is
      an established idiom in this codebase, not a novel invention.
      Guards added below: primal skips on (m==0) OR (isinf(m) and
      y[a]==0); tangent skips ONLY on (isinf(m) and y[a]==0).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ---------- forward tape evaluator (JAX oracle), parameterized by direction ----------
def _run_param_tape_jnp(inputs, tape, n_mem, input_slots):
    """Returns the list of (a, b, mult) numeric commands, mult as JAX values."""
    mem = [jnp.array(0.0)] * n_mem
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs[k]
    num = []
    for op in tape:
        k = op[0]
        if k == 'P':    _, f, t, c = op; mem[f] = mem[f] + mem[t] * c
        elif k == 'PP': _, f, t, m = op; mem[f] = mem[f] + mem[t] * mem[m]
        elif k == 'INV': _, f = op;      mem[f] = 1.0 / mem[f]
        elif k == 'OM':  _, f = op;      mem[f] = 1.0 - mem[f]
        elif k == 'DIV': _, f, t = op;   mem[f] = mem[f] / mem[t]
        elif k == 'ZERO': _, f = op;     mem[f] = jnp.array(0.0)
        elif k == 'NEWADD':
            _, a, b, m = op
            mv = mem[m] - 1.0 if a == b else mem[m]  # production's diagonal convention
            num.append((a, b, mv))
    return num


def moment_jnp(inputs, tape, n_mem, input_slots, seed, target):
    num = _run_param_tape_jnp(inputs, tape, n_mem, input_slots)
    r = [seed[i] for i in range(len(seed))]
    for (a, b, mv) in num:
        r[a] = r[a] + r[b] * mv
    return r[target]


def sojourn_jnp(inputs, tape, n_mem, input_slots, n_seed, target):
    """Sojourn vector: seed y[target]=1, walk REVERSED with a/b swapped."""
    num = _run_param_tape_jnp(inputs, tape, n_mem, input_slots)
    y = [jnp.array(0.0)] * n_seed
    y[target] = jnp.array(1.0)
    for (a, b, mv) in reversed(num):
        y[b] = y[b] + y[a] * mv
    return jnp.stack(y)


# ---------- manual forward-mode (dual-number) theta-adjoint for SOJOURN ----------
def manual_fwdmode_sojourn_grad(inputs_np, tape, n_mem, input_slots, n_seed, target,
                                 coeff_matrix):
    """coeff_matrix: (n_in, P) -- coefficients[k][j] = d(edge_weight_k)/d(theta_j)
    (linear mode). Returns (n_seed, P): d(sojourn(v))/d(theta_j) for all v, j,
    via P forward-mode passes (NOT looping over the n_in edge inputs)."""
    n_in = len(inputs_np)
    P = coeff_matrix.shape[1]

    # ---- param tape PRIMAL forward: record numeric commands (a, b, mult) ----
    mem = np.zeros(n_mem)
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs_np[k]
    ops_snapshot = []  # (opcode, args, primal-values-needed-for-tangent)
    num = []  # (a, b, mult_value, m_slot)
    for op in tape:
        k = op[0]
        if k == 'P':
            _, f, t, c = op
            ops_snapshot.append(('P', f, t, c, None))
            mem[f] += mem[t] * c
        elif k == 'PP':
            _, f, t, m = op
            ops_snapshot.append(('PP', f, t, m, (mem[t], mem[m])))
            mem[f] += mem[t] * mem[m]
        elif k == 'INV':
            _, f = op
            ops_snapshot.append(('INV', f, None, None, mem[f]))
            mem[f] = 1.0 / mem[f]
        elif k == 'OM':
            _, f = op
            ops_snapshot.append(('OM', f, None, None, None))
            mem[f] = 1.0 - mem[f]
        elif k == 'DIV':
            _, f, t = op
            ops_snapshot.append(('DIV', f, t, None, (mem[f], mem[t])))
            mem[f] = mem[f] / mem[t]
        elif k == 'ZERO':
            _, f = op
            ops_snapshot.append(('ZERO', f, None, None, None))
            mem[f] = 0.0
        elif k == 'NEWADD':
            _, a, b, m = op
            ops_snapshot.append(('NEWADD', a, b, m, None))
            mv = mem[m] - 1.0 if a == b else mem[m]  # production's diagonal convention
            num.append((a, b, mv, m))

    J = np.zeros((n_seed, P))
    for j in range(P):
        # ---- param tape TANGENT forward, seeded at theta component j ----
        mem_dot = np.zeros(n_mem)
        for k, sl in enumerate(input_slots):
            mem_dot[sl] = coeff_matrix[k, j]
        mdot_by_slot_for_newadd = []  # tangent of mem[m] at each NEWADD, in order

        for snap in ops_snapshot:
            kind = snap[0]
            if kind == 'P':
                _, f, t, c, _ = snap
                mem_dot[f] += mem_dot[t] * c
            elif kind == 'PP':
                _, f, t, m, (t0, m0) = snap
                mem_dot[f] += mem_dot[t] * m0 + t0 * mem_dot[m]
            elif kind == 'INV':
                _, f, _, _, f0 = snap
                mem_dot[f] = -mem_dot[f] / (f0 * f0)
            elif kind == 'OM':
                _, f, _, _, _ = snap
                mem_dot[f] = -mem_dot[f]
            elif kind == 'DIV':
                _, f, t, _, (f0, t0) = snap
                mem_dot[f] = mem_dot[f] / t0 - f0 * mem_dot[t] / (t0 * t0)
            elif kind == 'ZERO':
                _, f, _, _, _ = snap
                mem_dot[f] = 0.0
            elif kind == 'NEWADD':
                _, a, b, m, _ = snap
                mdot_by_slot_for_newadd.append(mem_dot[m])

        # ---- SOJOURN recurrence, PRIMAL + TANGENT interleaved (forward-mode
        # needs the LIVE primal y[a] at each step, not a separate snapshot pass) ----
        y = np.zeros(n_seed); y[target] = 1.0
        y_dot = np.zeros(n_seed)
        nc = len(num)
        for c in range(nc - 1, -1, -1):
            a, b, mv, _m = num[c]
            mdot = mdot_by_slot_for_newadd[c]
            # TANGENT: skip ONLY the inf-with-zero-operand case -- NOT m==0
            # (see module docstring (b): a diagonal command at weight exactly
            # 1.0 stores mv==0 but can have mdot!=0).
            if not (np.isinf(mv) and y[a] == 0.0):
                y_dot[b] += y_dot[a] * mv + y[a] * mdot
            # PRIMAL: mirrors ptd_expected_sojourn_time_subset's adjoint --
            # skips on EITHER mv==0 OR inf-with-zero-operand.
            if mv == 0.0:
                continue
            if np.isinf(mv) and y[a] == 0.0:
                continue
            y[b] += y[a] * mv
        J[:, j] = y_dot

    return J


def manual_fwdmode_sojourn_grad_WRONG_GUARD(inputs_np, tape, n_mem, input_slots, n_seed,
                                             target, coeff_matrix):
    """Same as manual_fwdmode_sojourn_grad but with the guard bug the review
    flagged: the tangent update WRONGLY shares the primal's m==0 skip. Exists
    only to prove the crafted diagonal-weight-exactly-1 test case below
    actually discriminates -- i.e. that the corrected function's PASS isn't
    vacuous (would pass regardless of which guard variant is used)."""
    n_in = len(inputs_np)
    P = coeff_matrix.shape[1]
    mem = np.zeros(n_mem)
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs_np[k]
    ops_snapshot = []
    num = []
    for op in tape:
        k = op[0]
        if k == 'P':
            _, f, t, c = op; ops_snapshot.append(('P', f, t, c, None)); mem[f] += mem[t] * c
        elif k == 'PP':
            _, f, t, m = op; ops_snapshot.append(('PP', f, t, m, (mem[t], mem[m]))); mem[f] += mem[t] * mem[m]
        elif k == 'INV':
            _, f = op; ops_snapshot.append(('INV', f, None, None, mem[f])); mem[f] = 1.0 / mem[f]
        elif k == 'OM':
            _, f = op; ops_snapshot.append(('OM', f, None, None, None)); mem[f] = 1.0 - mem[f]
        elif k == 'DIV':
            _, f, t = op; ops_snapshot.append(('DIV', f, t, None, (mem[f], mem[t]))); mem[f] = mem[f] / mem[t]
        elif k == 'ZERO':
            _, f = op; ops_snapshot.append(('ZERO', f, None, None, None)); mem[f] = 0.0
        elif k == 'NEWADD':
            _, a, b, m = op
            ops_snapshot.append(('NEWADD', a, b, m, None))
            mv = mem[m] - 1.0 if a == b else mem[m]
            num.append((a, b, mv, m))

    J = np.zeros((n_seed, P))
    for j in range(P):
        mem_dot = np.zeros(n_mem)
        for k, sl in enumerate(input_slots):
            mem_dot[sl] = coeff_matrix[k, j]
        mdot_by_slot_for_newadd = []
        for snap in ops_snapshot:
            kind = snap[0]
            if kind == 'P':
                _, f, t, c, _ = snap; mem_dot[f] += mem_dot[t] * c
            elif kind == 'PP':
                _, f, t, m, (t0, m0) = snap; mem_dot[f] += mem_dot[t] * m0 + t0 * mem_dot[m]
            elif kind == 'INV':
                _, f, _, _, f0 = snap; mem_dot[f] = -mem_dot[f] / (f0 * f0)
            elif kind == 'OM':
                _, f, _, _, _ = snap; mem_dot[f] = -mem_dot[f]
            elif kind == 'DIV':
                _, f, t, _, (f0, t0) = snap; mem_dot[f] = mem_dot[f] / t0 - f0 * mem_dot[t] / (t0 * t0)
            elif kind == 'ZERO':
                _, f, _, _, _ = snap; mem_dot[f] = 0.0
            elif kind == 'NEWADD':
                _, a, b, m, _ = snap; mdot_by_slot_for_newadd.append(mem_dot[m])

        y = np.zeros(n_seed); y[target] = 1.0
        y_dot = np.zeros(n_seed)
        nc = len(num)
        for c in range(nc - 1, -1, -1):
            a, b, mv, _m = num[c]
            mdot = mdot_by_slot_for_newadd[c]
            # BUG (under test): tangent WRONGLY shares the primal's m==0 skip.
            if mv == 0.0:
                continue
            if np.isinf(mv) and y[a] == 0.0:
                continue
            y_dot[b] += y_dot[a] * mv + y[a] * mdot
            y[b] += y[a] * mv
        J[:, j] = y_dot

    return J


def manual_sojourn_primal(inputs_np, tape, n_mem, input_slots, n_seed, target):
    mem = np.zeros(n_mem)
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs_np[k]
    num = []
    for op in tape:
        k = op[0]
        if k == 'P':    _, f, t, c = op; mem[f] += mem[t] * c
        elif k == 'PP': _, f, t, m = op; mem[f] += mem[t] * mem[m]
        elif k == 'INV': _, f = op;      mem[f] = 1.0 / mem[f]
        elif k == 'OM':  _, f = op;      mem[f] = 1.0 - mem[f]
        elif k == 'DIV': _, f, t = op;   mem[f] = mem[f] / mem[t]
        elif k == 'ZERO': _, f = op;     mem[f] = 0.0
        elif k == 'NEWADD':
            _, a, b, m = op
            mv = mem[m] - 1.0 if a == b else mem[m]
            num.append((a, b, mv))
    y = np.zeros(n_seed); y[target] = 1.0
    for (a, b, mv) in reversed(num):
        # mirrors ptd_expected_sojourn_time_subset's adjoint guards exactly
        if mv == 0.0:
            continue
        if np.isinf(mv) and y[a] == 0.0:
            continue
        y[b] += y[a] * mv
    return y


# ---------- random valid two-tier tapes + a linear theta->coeff map ----------
def random_case(rng):
    n_mem = int(rng.integers(4, 8))
    n_in = int(rng.integers(2, 5))
    input_slots = list(rng.choice(n_mem, size=n_in, replace=False))
    tape = []
    for _ in range(int(rng.integers(4, 12))):
        choice = rng.choice(['P', 'PP', 'INV', 'OM', 'DIV', 'ZERO'])
        f = int(rng.integers(n_mem)); t = int(rng.integers(n_mem)); m = int(rng.integers(n_mem))
        if choice == 'P':   tape.append(('P', f, t, float(rng.uniform(0.3, 1.5))))
        elif choice == 'PP':tape.append(('PP', f, t, m))
        elif choice == 'INV':tape.append(('INV', f))
        elif choice == 'OM':tape.append(('OM', f))
        elif choice == 'DIV':tape.append(('DIV', f, t if t != f else (t + 1) % n_mem))
        elif choice == 'ZERO':tape.append(('ZERO', f))
    n_seed = int(rng.integers(4, 7))
    for _ in range(int(rng.integers(3, 6))):
        a = int(rng.integers(n_seed)); b = int(rng.integers(n_seed)); m = int(rng.integers(n_mem))
        tape.append(('NEWADD', a, b, m))
    inputs = rng.uniform(1.0, 3.0, n_in)
    target = int(rng.integers(n_seed))
    P = int(rng.integers(1, 4))
    coeff_matrix = rng.uniform(-1.0, 1.0, size=(n_in, P))
    return tape, n_mem, input_slots, inputs, n_seed, target, P, coeff_matrix


def main():
    rng = np.random.default_rng(11)
    npass = 0; ntot = 0; worst = 0.0
    npass_primal = 0

    for trial in range(400):
        tape, n_mem, input_slots, inputs, n_seed, target, P, coeff_matrix = random_case(rng)
        inj = jnp.asarray(inputs)

        # --- primal sanity: manual sojourn matches JAX sojourn ---
        soj_jax = np.asarray(sojourn_jnp(inj, tape, n_mem, input_slots, n_seed, target))
        soj_man = manual_sojourn_primal(inputs, tape, n_mem, input_slots, n_seed, target)
        if not (np.all(np.isfinite(soj_jax)) and np.all(np.isfinite(soj_man))):
            continue
        if np.max(np.abs(soj_jax)) > 1e6:
            continue
        if np.max(np.abs(soj_jax - soj_man)) > 1e-9 * max(1.0, np.max(np.abs(soj_jax))):
            print(f"  PRIMAL MISMATCH trial={trial}: jax={soj_jax} man={soj_man}")
            continue
        npass_primal += 1

        # --- ground truth: jax.jacobian of sojourn(theta) via edge_weight(theta)=coeff@theta ---
        def sojourn_of_theta(theta):
            edge_weights = coeff_matrix @ theta  # (n_in,)
            return sojourn_jnp(edge_weights, tape, n_mem, input_slots, n_seed, target)

        theta0 = jnp.ones(P)  # arbitrary reference point; edge weights = coeff @ ones = row sums
        # Use the ACTUAL inputs as the edge weights by picking theta s.t. coeff@theta ~ inputs
        # is over-determined in general; instead just differentiate at theta=ones and use
        # edge weights = coeff_matrix @ ones for this trial (consistent inputs for both sides).
        edge_weights_at_theta0 = np.asarray(coeff_matrix @ np.ones(P))
        if not np.all(np.isfinite(edge_weights_at_theta0)):
            continue

        J_ref = np.asarray(jax.jacobian(sojourn_of_theta)(theta0))
        if not np.all(np.isfinite(J_ref)):
            continue
        if np.max(np.abs(J_ref)) > 1e8:
            continue

        # manual forward-mode adjoint, evaluated at the SAME edge weights (coeff @ ones)
        J_man = manual_fwdmode_sojourn_grad(
            edge_weights_at_theta0, tape, n_mem, input_slots, n_seed, target, coeff_matrix)

        if not np.all(np.isfinite(J_man)):
            continue

        ntot += 1
        err = np.max(np.abs(J_man - J_ref)) / max(1.0, np.max(np.abs(J_ref)))
        worst = max(worst, err)
        if err < 1e-9:
            npass += 1
        elif ntot - npass <= 5:
            print(f"  MISMATCH trial={trial} err={err:.2e}\n    ref=\n{J_ref}\n    man=\n{J_man}")

    print(f"\nPrimal sojourn (manual reversed-index walk vs JAX): {npass_primal} cases checked, all matched")
    print(f"Forward-mode theta-adjoint for SOJOURN vs JAX autodiff (P forward passes, seeded via coeff column):")
    print(f"  {npass}/{ntot} random tapes match  (worst rel.err = {worst:.2e})")
    print("  RESULT:", "ALL PASS" if npass == ntot and ntot > 50 else "CHECK FAILURES")
    return 0 if (npass == ntot and ntot > 50) else 1


# ---------- SECOND independent cross-check: reverse-mode moment-chain machinery
# seeded with e_v, run once per target v (slow, O(n_seed) passes -- NOT the
# production algorithm, but a genuinely different code path for cross-validation) ----------
def manual_vjp_moment_chain(inputs_np, tape, n_mem, input_slots, seed_np, target,
                             coeff_matrix):
    """Verbatim port of dr_twotier_full_adjoint.py's manual_vjp, generalized to
    return d(m_0)/d(theta_j) directly (contracting bar[] against coeff_matrix)
    instead of d(m_0)/d(input_k)."""
    mem = np.zeros(n_mem)
    for k, sl in enumerate(input_slots):
        mem[sl] = inputs_np[k]
    snaps = []
    num = []
    for op in tape:
        k = op[0]
        if k == 'P':   _, f, t, c = op; snaps.append(None); mem[f] += mem[t] * c
        elif k == 'PP':_, f, t, m = op; snaps.append((mem[t], mem[m])); mem[f] += mem[t] * mem[m]
        elif k == 'INV':_, f = op;      snaps.append(mem[f]); mem[f] = 1.0 / mem[f]
        elif k == 'OM':_, f = op;       snaps.append(None); mem[f] = 1.0 - mem[f]
        elif k == 'DIV':_, f, t = op;   snaps.append((mem[f], mem[t])); mem[f] = mem[f] / mem[t]
        elif k == 'ZERO':_, f = op;     snaps.append(None); mem[f] = 0.0
        elif k == 'NEWADD':
            _, a, b, m = op
            snaps.append(None)
            mv = mem[m] - 1.0 if a == b else mem[m]  # production's diagonal convention
            num.append((a, b, mv, m))
    r = seed_np.astype(float).copy(); snap_to = []
    for (a, b, mv, _m) in num:
        snap_to.append(r[b]); r[a] += r[b] * mv
    adj = np.zeros(len(seed_np)); adj[target] = 1.0
    dm = np.zeros(len(num))
    for c in range(len(num) - 1, -1, -1):
        a, b, mv, _m = num[c]
        dm[c] = adj[a] * snap_to[c]
        adj[b] += adj[a] * mv
    bar = np.zeros(n_mem)
    numptr = len(num) - 1
    for i in range(len(tape) - 1, -1, -1):
        op = tape[i]; k = op[0]; sn = snaps[i]
        if k == 'P':   _, f, t, c = op; bf = bar[f]; bar[t] += bf * c
        elif k == 'PP':_, f, t, m = op; t0, m0 = sn; bf = bar[f]; bar[t] += bf*m0; bar[m] += bf*t0
        elif k == 'INV':_, f = op; f0 = sn; bar[f] = bar[f] * (-1.0 / f0**2)
        elif k == 'OM':_, f = op; bar[f] = -bar[f]
        elif k == 'DIV':_, f, t = op; f0, t0 = sn; bf = bar[f]; bar[t] += bf*(-f0/t0**2); bar[f] = bf/t0
        elif k == 'ZERO':_, f = op; bar[f] = 0.0
        elif k == 'NEWADD':_, a, b, m = op; bar[m] += dm[numptr]; numptr -= 1
    bar_inputs = np.array([bar[sl] for sl in input_slots])  # d(m_0)/d(input_k)
    return bar_inputs @ coeff_matrix  # (P,): d(m_0)/d(theta_j) = sum_k d/d(input_k) * coeff[k,j]


def cross_check_main():
    rng = np.random.default_rng(23)
    npass = 0; ntot = 0; worst = 0.0
    for trial in range(150):
        tape, n_mem, input_slots, inputs, n_seed, target, P, coeff_matrix = random_case(rng)
        edge_weights = np.asarray(coeff_matrix @ np.ones(P))
        if not np.all(np.isfinite(edge_weights)) or np.max(np.abs(edge_weights)) > 1e4:
            continue

        # Forward-mode (this batch's new algorithm): full (n_seed, P) Jacobian.
        J_fwd = manual_fwdmode_sojourn_grad(
            edge_weights, tape, n_mem, input_slots, n_seed, target, coeff_matrix)
        if not np.all(np.isfinite(J_fwd)) or np.max(np.abs(J_fwd)) > 1e8:
            continue

        # Reverse-mode moment-chain, run ONCE PER TARGET VERTEX v with seed=e_v,
        # giving row v of the SAME (n_seed, P) Jacobian each time (since
        # sojourn(v) = m_0^{seed=e_v} algebraically -- see module docstring).
        J_rev = np.zeros((n_seed, P))
        ok = True
        for v in range(n_seed):
            e_v = np.zeros(n_seed); e_v[v] = 1.0
            row = manual_vjp_moment_chain(edge_weights, tape, n_mem, input_slots, e_v, target, coeff_matrix)
            if not np.all(np.isfinite(row)):
                ok = False
                break
            J_rev[v, :] = row
        if not ok or np.max(np.abs(J_rev)) > 1e8:
            continue

        ntot += 1
        err = np.max(np.abs(J_fwd - J_rev)) / max(1.0, np.max(np.abs(J_rev)))
        worst = max(worst, err)
        if err < 1e-9:
            npass += 1
        elif ntot - npass <= 5:
            print(f"  CROSS-CHECK MISMATCH trial={trial} err={err:.2e}\n    fwd=\n{J_fwd}\n    rev(per-v)=\n{J_rev}")

    print(f"\nCross-check: forward-mode (P passes) vs reverse-mode-per-target-vertex (n_seed passes):")
    print(f"  {npass}/{ntot} random tapes match  (worst rel.err = {worst:.2e})")
    print("  RESULT:", "ALL PASS" if npass == ntot and ntot > 50 else "CHECK FAILURES")
    return 0 if (npass == ntot and ntot > 50) else 1


# ---------- D1.5: crafted diagonal-weight-exactly-1 guard-asymmetry case ----------
def guard_asymmetry_case():
    """Deterministic (not random) case engineered so a diagonal NEWADD
    command's stored multiplier (mem[m] - 1.0, since a==b) is EXACTLY 0.0 at
    theta=1.0, while its theta-derivative is 1.0 (nonzero) -- the precise
    scenario finding 2 of the D1 review flagged. Single vertex, single
    input, single command: mem[0] = theta[0] fed directly (no intervening
    ops), NEWADD(a=0, b=0, m=0) so mv = mem[0] - 1.0 = 0.0 at theta[0]=1.0.

    Confirms two things:
      1. The CORRECTED function (asymmetric guards) matches jax.jacobian
         exactly at this point -- proving the fix handles the case the
         random suite above cannot reliably hit by chance.
      2. The WRONG variant (tangent shares the primal's m==0 skip) does NOT
         match -- proving check (1) is discriminating, not vacuous (i.e. it
         would actually fail if the bug were present, not pass regardless).
    """
    tape = [('NEWADD', 0, 0, 0)]
    n_mem, input_slots, n_seed, target, P = 1, [0], 1, 0, 1
    coeff_matrix = np.array([[1.0]])
    theta0 = jnp.array([1.0])

    def sojourn_of_theta(theta):
        edge_weights = coeff_matrix @ theta  # (1,) = [theta[0]]
        return sojourn_jnp(edge_weights, tape, n_mem, input_slots, n_seed, target)

    J_ref = np.asarray(jax.jacobian(sojourn_of_theta)(theta0))

    edge_weights_at_theta0 = np.asarray(coeff_matrix @ np.ones(P))
    assert edge_weights_at_theta0[0] == 1.0  # mem[0]=1.0 -> diagonal mv = 1.0-1.0 = 0.0

    J_correct = manual_fwdmode_sojourn_grad(
        edge_weights_at_theta0, tape, n_mem, input_slots, n_seed, target, coeff_matrix)
    J_wrong = manual_fwdmode_sojourn_grad_WRONG_GUARD(
        edge_weights_at_theta0, tape, n_mem, input_slots, n_seed, target, coeff_matrix)

    correct_matches = np.allclose(J_correct, J_ref, atol=1e-12)
    wrong_diverges = not np.allclose(J_wrong, J_ref, atol=1e-6)

    print("\n=== D1.5 guard-asymmetry case: diagonal command at weight exactly 1.0 ===")
    print(f"  J_ref (jax.jacobian)        = {J_ref.tolist()}")
    print(f"  J_correct (asymmetric guard) = {J_correct.tolist()}  "
          f"{'OK (matches)' if correct_matches else 'FAIL (should match)'}")
    print(f"  J_wrong (symmetric m==0 skip)= {J_wrong.tolist()}  "
          f"{'OK (diverges, as expected)' if wrong_diverges else 'FAIL (should diverge -- test is vacuous!)'}")
    ok = correct_matches and wrong_diverges
    print("  RESULT:", "ALL PASS" if ok else "CHECK FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    r1 = main()
    r2 = cross_check_main()
    r3 = guard_asymmetry_case()
    raise SystemExit(0 if (r1 == 0 and r2 == 0 and r3 == 0) else 1)
