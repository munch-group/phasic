"""Batch H de-risk H0 -- pure-JAX one-hop oracle for the daisy final-epoch read.

Plan: b3-batchH-plan.md v2, section "De-risk phase". Findings recorded in
b3-batchH-findings.md.

Two-tier design (v2 amendment: production intermediate epochs are
granularity-limited uniformization, so 1e-10 chain parity vs a dense-expm
oracle is unreachable by construction; instead):

  Tier 1 (exact parity, same numerics): replicate the fused FFI handler
    (DaisyChainSojournFfiImpl, graph_builder_ffi.cpp:2043-2134) step by
    step through the PYBIND API (update_ipv / update_weights /
    stop_probability / expected_sojourn_time(subset)) -- this is also the
    H2(a) "Python replication" handoff-extraction route, validated here.
  Tier 2 (differentiable reference): dense JAX oracle -- expm for the
    intermediate epochs (oracle-only; the avoid-matrix-exp preference is
    about library code, an oracle NEEDS an independent method), linear
    solve for the final sojourn read. jax.jacobian over it is the ground
    truth for H0(iii)/(iv).

Checks (numbered as in the plan):
  (i)   value parity: production FFI vs Tier-1 replication (<=1e-12) and
        vs Tier-2 oracle (granularity-limited; measured, incl. a
        granularity sweep showing convergence toward the oracle).
  (ii)  LINEARITY of the final read in the raw handoff IPV -- on the
        SHIPPED path (update_ipv + expected_sojourn_time subset), not
        just the oracle (where it is linear by construction). Also
        probes the C normalization semantics explicitly.
  (iii) exact final-epoch theta block INCLUDING the r_v product-rule
        term (v2 amendment 1) vs jax.jacobian (<=1e-10 rel), evaluated
        at the oracle handoff (block correctness, isolated from
        intermediate-epoch discretization error).
  (iv)  composed chain Jacobian (exact final-epoch columns at the
        production handoff + full-chain-FD earlier columns, eps=1e-7
        central, exactly _autodiff_bwd's scheme) vs full-FD vs oracle,
        at a benign theta AND a mixed-scale theta. NOTE: at this level
        ALL slots (incl. the mutation slot the SVGD fixture pins) are
        free, so the r_v term is exercised without a fixture rewrite.
"""
import json
import sys
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic  # must precede jax import: forces x64
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv
from phasic.ffi_wrappers import (
    _make_json_serializable,
    compute_daisy_chain_sojourn_ffi,
)

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

set_log_level("WARNING")
np.random.seed(17)

REPORT = []


def check(name, value, tol=None, note=""):
    ok = (value <= tol) if tol is not None else None
    REPORT.append((name, value, tol, ok, note))
    status = "" if ok is None else ("PASS" if ok else "FAIL")
    print(f"  [{name}] {value:.3e}" + (f" (tol {tol:.0e}) {status}" if tol is not None else "") + (f"  {note}" if note else ""))


# ---------------------------------------------------------------- fixture
# test_lrt_at.py::epoch_tied_free_pair, minus the SVGD runs.
all_pairs = partial(combinations_with_replacement, r=2)
nr_samples = 4
indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr_samples)])


@with_ipv([nr_samples] + [0] * (nr_samples - 1))
def coal(state, indexer=None):
    t = []
    for i, j in all_pairs(range(indexer.lineages.state_length)):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy(); new[i] -= 1; new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t


graph = Graph(coal, indexer=indexer)
rl, mu, th = 5, 1e-4, 1 / 10_000
jpg = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=mu, discrete=False)

P = jpg.param_length()
n_epochs = 2
epoch_dts = [0.5]           # epoch_starts=[0, 0.5]
granularity = 0             # auto -- the SVGD builder default
print(f"fixture: jpg n={jpg.vertices_length()} P={P}")
assert P == 2, f"expected P=2 (coal, mutation), got {P}"

jsp = jpg.joint_stop_prob_graph()
sg = jpg.joint_sojourn_graph()

ipv_target_indices = [int(x) for x in jsp._ipv_target_indices]
t_aux_map = {int(k): int(v) for k, v in jsp._t_aux_map.items()}
t_vertex_indices = [int(x) for x in jsp._t_vertex_indices]
n_ipv = len(ipv_target_indices)

# sojourn gather/index maps -- replicated from _daisy_chain_svgd_model
# (__init__.py:4603-4620), state-matched between jsp and sg.
jsp_states = jsp.states()
jsp_ipv_pos = {tuple(int(x) for x in jsp_states[v]): k for k, v in enumerate(ipv_target_indices)}
sg_states = sg.states()
sg_state_to_idx = {tuple(int(x) for x in sg_states[v]): v for v in range(sg.vertices_length())}
sojourn_jsp_gather = [int(jsp_ipv_pos[s]) for s in sg._ipv_target_states]
sojourn_t_indices = [int(sg_state_to_idx[tuple(int(x) for x in jsp_states[t])]) for t in t_vertex_indices]
n_t = len(sojourn_t_indices)
sojourn_n_ipv = len(sojourn_jsp_gather)
print(f"jsp n={jsp.vertices_length()} n_ipv={n_ipv} n_t={n_t}; sg n={sg.vertices_length()} sojourn_n_ipv={sojourn_n_ipv}")

# initial IPV in jsp coordinates (__init__.py:4376-4379)
self_ipv_full = np.zeros(jpg.vertices_length())
for edge in jpg.starting_vertex().edges():
    self_ipv_full[edge.to().index()] = edge.weight()
initial_ipv = self_ipv_full[np.asarray(ipv_target_indices)]
print(f"initial_ipv sum={initial_ipv.sum():.6g} nonzeros={int((initial_ipv != 0).sum())}")

# ---------------------------------------------------------------- production (fused FFI)
structure = _make_json_serializable(jsp.serialize(theta_dim=P))
structure["_daisy_chain"] = {
    "n_epochs": int(n_epochs),
    "param_length": int(P),
    "t_eval": 10.0,  # unused by the sojourn handler; required key parity
    "granularity": int(granularity),
    "epoch_dts": [float(x) for x in epoch_dts],
    "ipv_target_indices": ipv_target_indices,
    "t_aux_keys": list(t_aux_map.keys()),
    "t_aux_values": [t_aux_map[k] for k in t_aux_map.keys()],
    "t_vertex_indices": t_vertex_indices,
    "sojourn_jsp_gather": sojourn_jsp_gather,
    "sojourn_t_indices": sojourn_t_indices,
}
structure_json = json.dumps(structure)
sojourn_json = json.dumps(_make_json_serializable(sg.serialize(theta_dim=P)))


def production(theta_flat):
    out = compute_daisy_chain_sojourn_ffi(
        structure_json, sojourn_json,
        jnp.asarray(theta_flat, dtype=jnp.float64),
        jnp.asarray(initial_ipv, dtype=jnp.float64),
    )
    return np.asarray(out)


# ---------------------------------------------------------------- Tier 1: pybind replication
def replicated_chain(theta_flat, gran=granularity):
    """Replicates DaisyChainSojournFfiImpl per-batch body step by step.
    Returns (final_jp, handoff_ipv_work, sojourn_ipv, mass, soj_subset, r_v)."""
    theta_flat = np.asarray(theta_flat, dtype=np.float64)
    ipv_work = np.asarray(initial_ipv, dtype=np.float64).copy()
    for e in range(n_epochs - 1):
        theta_e = theta_flat[e * P:(e + 1) * P]
        jsp.update_ipv(ipv_work)
        jsp.update_weights(theta_e.tolist())
        raw = np.asarray(jsp.stop_probability(epoch_dts[e], granularity=gran))
        new = np.empty(n_ipv)
        for k, tgt in enumerate(ipv_target_indices):
            p = raw[tgt]
            if tgt in t_aux_map:
                p += raw[t_aux_map[tgt]]
            new[k] = p
        ipv_work = new
    sojourn_ipv = ipv_work[np.asarray(sojourn_jsp_gather)]
    mass = float(sojourn_ipv.sum())
    theta_final = theta_flat[(n_epochs - 1) * P:]
    sg.update_ipv(sojourn_ipv)
    sg.update_weights(theta_final.tolist())
    soj = np.asarray(sg.expected_sojourn_time(sojourn_t_indices))
    by_idx = {v.index(): v for v in sg.vertices()}
    r_v = np.empty(n_t)
    for k, ti in enumerate(sojourn_t_indices):
        tv = by_idx[ti]
        acc = 0.0
        for e in tv.edges():
            if len(e.to().edges()) == 0:
                acc += e.weight()
        r_v[k] = acc
    return r_v * soj * mass, ipv_work, sojourn_ipv, mass, soj, r_v


# ---------------------------------------------------------------- Tier 2: dense JAX oracle
def dense_parts(ser):
    """Constant / parameterized structure of a serialized graph, in
    enumeration space (asserted == vertex-index space)."""
    n = int(ser["n_vertices"])
    vidx = np.asarray(ser["vertex_indices"]) if "vertex_indices" in ser else None
    edges = np.asarray(ser["edges"], dtype=np.float64).reshape(-1, 3)
    pedges = np.asarray(ser["param_edges"], dtype=np.float64)
    if pedges.size == 0:
        pedges = pedges.reshape(0, 2 + P)
    start_edges = np.asarray(ser["start_edges"], dtype=np.float64).reshape(-1, 2)
    return n, edges, pedges, start_edges


n_jsp, jsp_e, jsp_pe, jsp_se = dense_parts(_make_json_serializable(jsp.serialize(theta_dim=P)))
n_sg, sg_e, sg_pe, sg_se = dense_parts(_make_json_serializable(sg.serialize(theta_dim=P)))
assert n_jsp == jsp.vertices_length() and n_sg == sg.vertices_length()


def Q_of(nv, edges, pedges, theta):
    Q = jnp.zeros((nv, nv), dtype=jnp.float64)
    if edges.shape[0]:
        Q = Q.at[edges[:, 0].astype(int), edges[:, 1].astype(int)].add(jnp.asarray(edges[:, 2]))
    if pedges.shape[0]:
        w = jnp.asarray(pedges[:, 2:]) @ theta
        Q = Q.at[pedges[:, 0].astype(int), pedges[:, 1].astype(int)].add(w)
    return Q - jnp.diag(jnp.sum(Q, axis=1))


# sg partition: transient = has out-edges; absorbing = none. r_v structure.
outdeg = np.zeros(n_sg)
for arr in (sg_e, sg_pe):
    for row in arr:
        outdeg[int(row[0])] += 1
absorbing = outdeg == 0
alpha_scatter = sg_se[:, 0].astype(int)          # start-edge order == update_ipv order
assert len(alpha_scatter) == sojourn_n_ipv, (len(alpha_scatter), sojourn_n_ipv)
transient_idx = np.asarray([i for i in range(n_sg) if not absorbing[i] and i not in set(alpha_scatter.tolist()) or True])
# NOTE: keep ALL vertices in the solve; absorbing rows are zero so we
# restrict to non-absorbing rows/cols for invertibility.
trans = np.where(~absorbing)[0]
trans_pos = {int(v): i for i, v in enumerate(trans)}
t_pos_in_trans = np.asarray([trans_pos[ti] for ti in sojourn_t_indices])

# r_v(theta) = r_const + r_coeff @ theta over exit edges (t-vertex -> absorbing)
r_const = np.zeros(n_t)
r_coeff = np.zeros((n_t, P))
tpos = {ti: k for k, ti in enumerate(sojourn_t_indices)}
for row in sg_e:
    f, t_, w = int(row[0]), int(row[1]), row[2]
    if f in tpos and absorbing[t_]:
        r_const[tpos[f]] += w
for row in sg_pe:
    f, t_ = int(row[0]), int(row[1])
    if f in tpos and absorbing[t_]:
        r_coeff[tpos[f]] += row[2:]


def oracle_final_read(theta_final, alpha_raw):
    """LINEAR final read: r_v(theta) * [alpha_raw @ inv(-T)]_t. No
    normalization -- if the shipped normalize-then-mass-rescale is truly
    linear, this equals it."""
    Q = Q_of(n_sg, sg_e, sg_pe, theta_final)
    T = Q[jnp.asarray(trans)][:, jnp.asarray(trans)]
    alpha_full = jnp.zeros(n_sg, dtype=jnp.float64).at[jnp.asarray(alpha_scatter)].set(alpha_raw)
    a_t = alpha_full[jnp.asarray(trans)]
    soj = jnp.linalg.solve(-T.T, a_t)            # soj_v = [alpha @ inv(-T)]_v
    r = jnp.asarray(r_const) + jnp.asarray(r_coeff) @ theta_final
    return r * soj[jnp.asarray(t_pos_in_trans)]


def oracle_chain(theta_flat, ipv0):
    p = jnp.zeros(n_jsp, dtype=jnp.float64).at[jnp.asarray(ipv_target_indices)].set(ipv0)
    ipv = None
    for e in range(n_epochs - 1):
        theta_e = theta_flat[e * P:(e + 1) * P]
        Q = Q_of(n_jsp, jsp_e, jsp_pe, theta_e)
        pt = p @ expm(Q * epoch_dts[e])
        vals = []
        for tgt in ipv_target_indices:
            v = pt[tgt]
            if tgt in t_aux_map:
                v = v + pt[t_aux_map[tgt]]
            vals.append(v)
        ipv = jnp.stack(vals)
        p = jnp.zeros(n_jsp, dtype=jnp.float64).at[jnp.asarray(ipv_target_indices)].set(ipv)
    alpha = ipv[jnp.asarray(sojourn_jsp_gather)]
    return oracle_final_read(theta_flat[(n_epochs - 1) * P:], alpha), ipv, alpha


theta_base = np.array([th, mu, 0.5 * th, 1.3 * mu])          # epoch-distinct, all slots live
theta_mixed = np.array([2e-5, 1e-4, 3e-4, 5e-2])             # mixed-scale stressor

print("\n== (i) value parity ==")
fp_prod = production(theta_base)
fp_rep, handoff, soj_ipv_prod, mass_prod, soj_prod, rv_prod = replicated_chain(theta_base)
check("i.tier1 prod-vs-replication (same numerics)", float(np.max(np.abs(fp_prod - fp_rep)) / np.max(np.abs(fp_prod))), 1e-12)
print(f"  r_v (live graph) = {rv_prod[:3]}... vs mutation slot {theta_base[-1]:.3g}; mass={mass_prod:.6g}")
if np.isnan(fp_prod).any():
    print("  !! NaN in production output -- swallowed C failure (16b item 8); STOP AND DEBUG")

fp_oracle, handoff_oracle, alpha_oracle = oracle_chain(jnp.asarray(theta_base), jnp.asarray(initial_ipv))
fp_oracle = np.asarray(fp_oracle)
check("i.tier2 prod-vs-oracle (granularity-limited)", float(np.max(np.abs(fp_prod - fp_oracle)) / np.max(np.abs(fp_prod))))
for g in (512, 4096, 32768):
    fp_g = replicated_chain(theta_base, gran=g)[0]
    check(f"i.sweep gran={g} vs oracle", float(np.max(np.abs(fp_g - fp_oracle)) / np.max(np.abs(fp_oracle))),
          note="should shrink as gran grows")
check("i.handoff prod-vs-oracle", float(np.max(np.abs(handoff - np.asarray(handoff_oracle)))))

print("\n== (ii) linearity of the SHIPPED final read in raw handoff ipv ==")
theta_final_np = theta_base[P:]
rng = np.random.default_rng(0)
x = rng.uniform(0.1, 1.0, sojourn_n_ipv)
y = rng.uniform(0.1, 1.0, sojourn_n_ipv)
a, b = 0.37, 1.91


def shipped_final_read(alpha_raw):
    sg.update_ipv(np.asarray(alpha_raw, dtype=np.float64))
    sg.update_weights(theta_final_np.tolist())
    soj = np.asarray(sg.expected_sojourn_time(sojourn_t_indices))
    return rv_prod * soj * float(np.sum(alpha_raw)), soj


f_x, soj_x = shipped_final_read(x)
f_y, _ = shipped_final_read(y)
f_ab, _ = shipped_final_read(a * x + b * y)
check("ii.linearity shipped f(ax+by) vs a f(x)+b f(y)",
      float(np.max(np.abs(f_ab - (a * f_x + b * f_y))) / np.max(np.abs(f_ab))), 1e-12)
_, soj_2x = shipped_final_read(2.0 * x)
print(f"  normalization probe: max|soj(2x)-soj(x)| = {np.max(np.abs(soj_2x - soj_x)):.3e} "
      f"(0 => C normalizes ipv internally; mass rescale is what restores linearity)")
f_x_oracle = np.asarray(oracle_final_read(jnp.asarray(theta_final_np), jnp.asarray(x)))
check("ii.shipped-vs-oracle at same raw alpha", float(np.max(np.abs(f_x - f_x_oracle)) / np.max(np.abs(f_x))), 1e-10)

print("\n== (iii) exact final-epoch theta block (incl. r_v term) vs jax.jacobian ==")

# MPFR-conditioning-gate bookkeeping. De-risk finding (dbg_decline2-5,
# recorded in b3-batchH-findings.md): at this fixture's REALISTIC
# coalescent theta (~1e-4) with realistic handoff dynamic range (entries
# 1 -> <=1e-8), the default PHASIC_CONDITION_THRESHOLD declines the
# shipped adjoint; theta=ones never declines. The block checks below
# lift the threshold (the Batch-F-precedented per-call env knob,
# phasic.c reads it per call) and RECORD every gate event; check
# (iii.gate) then measures whether the gated answers were actually
# inaccurate (16b item 2: on the sojourn path the gate is a conservatism
# knob, not a correctness necessity -- here we quantify it).
import os

GATE_EVENTS = []


def exact_final_block(theta_final, alpha_raw, tag=""):
    """The Batch-H exact block: shipped C adjoint + r_v product-rule term.
    Returns (n_t, P) Jacobian d(final_jp)/d(theta_final) at fixed alpha.
    On a default-threshold gate decline, retries with the threshold
    lifted and records the event."""
    alpha_raw = np.asarray(alpha_raw, dtype=np.float64)
    mass = float(alpha_raw.sum())
    sg.update_ipv(alpha_raw)
    sg.update_weights(np.asarray(theta_final, dtype=np.float64).tolist())
    soj = np.asarray(sg.expected_sojourn_time(sojourn_t_indices))
    raw = sg._sojourn_grad_theta_subset(list(sojourn_t_indices))
    if not raw:
        GATE_EVENTS.append(tag or "unlabelled")
        # dbg_decline6: the true handoff's min nonzero entry is ~5e-148
        # (dynamic range ~1e148), so the conditioning estimate exceeds any
        # threshold below ~1e300 -- 1e30 is NOT enough to lift the gate at
        # a real handoff. 1e300 = effectively disable, to measure whether
        # the gated answer is accurate at all (checks iii/iv judge that).
        os.environ["PHASIC_CONDITION_THRESHOLD"] = "1e300"
        try:
            raw = sg._sojourn_grad_theta_subset(list(sojourn_t_indices))
        finally:
            os.environ.pop("PHASIC_CONDITION_THRESHOLD", None)
        if not raw:
            raise RuntimeError("adjoint declined even with the gate lifted")
    J_soj = np.asarray(raw, dtype=np.float64).reshape(n_t, P)
    r = r_const + r_coeff @ np.asarray(theta_final)
    return (r_coeff * (soj * float(mass))[:, None]) + (r[:, None] * J_soj * mass)


for label, alpha_use in (("oracle-handoff", np.asarray(alpha_oracle)), ("production-handoff", soj_ipv_prod)):
    J_exact = exact_final_block(theta_base[P:], alpha_use, tag=f"iii/{label}")
    J_orc = np.asarray(jax.jacobian(oracle_final_read, argnums=0)(jnp.asarray(theta_base[P:]), jnp.asarray(alpha_use)))
    denom = np.max(np.abs(J_orc))
    check(f"iii.block-vs-jax.jacobian @ {label}", float(np.max(np.abs(J_exact - J_orc)) / denom), 1e-10)
print(f"  gate events so far: {len(GATE_EVENTS)} {GATE_EVENTS}")
print("  (iii.gate: if the checks above PASS despite gate events, the gated "
      "answers were accurate -- the default gate is over-conservative on this "
      "path at realistic theta; 16b item 2 evidence)")

print("\n== (iv) composed chain Jacobian: exact-final + FD-earlier vs full-FD vs oracle ==")
EPS = 1e-7  # _autodiff_bwd's central-difference eps, exactly


def fd_jacobian_production(theta_flat):
    theta_flat = np.asarray(theta_flat, dtype=np.float64)
    J = np.zeros((n_t, theta_flat.size))
    for i in range(theta_flat.size):
        tp = theta_flat.copy(); tp[i] += EPS
        tm = theta_flat.copy(); tm[i] -= EPS
        J[:, i] = (production(tp) - production(tm)) / (2 * EPS)
    return J


for label, theta_use in (("benign", theta_base), ("mixed-scale", theta_mixed)):
    print(f"  -- theta variant: {label} {theta_use.tolist()}")
    J_oracle_chain = np.asarray(jax.jacobian(lambda t: oracle_chain(t, jnp.asarray(initial_ipv))[0])(jnp.asarray(theta_use)))
    J_fd = fd_jacobian_production(theta_use)
    _, handoff_u, soj_ipv_u, _, _, _ = replicated_chain(theta_use)
    J_comp = J_fd.copy()
    J_comp[:, P:] = exact_final_block(theta_use[P:], soj_ipv_u, tag=f"iv/{label}")
    denom = np.max(np.abs(J_oracle_chain))
    err_fd_final = np.max(np.abs(J_fd[:, P:] - J_oracle_chain[:, P:])) / denom
    err_comp_final = np.max(np.abs(J_comp[:, P:] - J_oracle_chain[:, P:])) / denom
    err_fd_early = np.max(np.abs(J_fd[:, :P] - J_oracle_chain[:, :P])) / denom
    check(f"iv.{label} FD error, final-epoch cols", float(err_fd_final))
    check(f"iv.{label} COMPOSED error, final-epoch cols", float(err_comp_final))
    check(f"iv.{label} FD error, earlier cols (unchanged by H)", float(err_fd_early))
    ratio = err_fd_final / err_comp_final if err_comp_final > 0 else np.inf
    check(f"iv.{label} improvement factor (final cols)", float(ratio), note=">=10 required for GO")
    # Tied variant (plan v2 amendment: tied=[(0, [0, 1])] i.e. slot 0
    # shared across both epochs). Tying is a scatter ABOVE the custom_vjp
    # boundary (verified __init__.py:4701-4702); its VJP sums slave->master
    # cotangents, so the tied master gradient is the SUM of flat columns
    # {0, P}. Checked here on the Jacobians directly.
    tied_comp = J_comp[:, 0] + J_comp[:, P]
    tied_orc = J_oracle_chain[:, 0] + J_oracle_chain[:, P]
    check(f"iv.{label} TIED master-slot composed-vs-oracle",
          float(np.max(np.abs(tied_comp - tied_orc)) / np.max(np.abs(J_oracle_chain))),
          note="tied master = flat cols 0+P summed (scatter VJP)")

print("\n== gate-decline census (H2(d) early answer) ==")
print(f"  default-threshold gate declines during (iii)/(iv): {len(GATE_EVENTS)}: {GATE_EVENTS}")

print("\n== summary ==")
fails = [r for r in REPORT if r[3] is False]
for name, value, tol, ok, note in REPORT:
    s = "PASS" if ok else ("FAIL" if ok is False else "info")
    print(f"  {s:4s}  {name}: {value:.3e}" + (f" (tol {tol:.0e})" if tol else ""))
print(f"\n{'ALL GATED CHECKS PASS' if not fails else f'{len(fails)} GATED CHECK(S) FAILED'}")
sys.exit(0 if not fails else 1)
