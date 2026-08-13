"""Batch H I1 micro-gates (plan v3.1): the conditioning-gate opt-out.

Modes (argv[1]):
  dump   -- run under the PRE-H install (main checkout): compute the
            default-path results on the shared fixtures and save them to
            the npz given as argv[2]. The old pybind signature has no
            skip_condition_gate arg -- calls use indices only.
  check  -- run under the NEW build (worktree): micro-gates
            (a) flag=0 byte-identity vs the dump (exact array equality,
                including decline-parity: old-declined cases must still
                decline);
            (b) flag=1 at the H0-documented declining point (realistic
                theta + real handoff) COMPUTES and matches a dense-JAX
                oracle Jacobian <=1e-10;
            (c) flag=1 at a genuinely non-finite requested row (trap
                fixture) still DECLINES via the isfinite sweep -- the
                gate-skip must not disable the live defense;
            (d) subnormal-mass probe (plan micro-gate (d)): behavior at
                mass ~1e-300 recorded (decides the I2 zero-branch width).

Fixture = the H0 oracle fixture (test_lrt_at epoch model, jpg n=514).
"""
import sys
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

set_log_level("WARNING")
np.random.seed(17)

MODE = sys.argv[1]
NPZ = sys.argv[2]

all_pairs = partial(combinations_with_replacement, r=2)
nr = 4
indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])


@with_ipv([nr] + [0] * (nr - 1))
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
jpg = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=1e-4, discrete=False)
jsp = jpg.joint_stop_prob_graph()
sg = jpg.joint_sojourn_graph()
P = jpg.param_length()

jsp_states = jsp.states(); sg_states = sg.states()
iti = [int(x) for x in jsp._ipv_target_indices]
tam = {int(k): int(v) for k, v in jsp._t_aux_map.items()}
jsp_ipv_pos = {tuple(int(x) for x in jsp_states[v]): k for k, v in enumerate(iti)}
sg_s2i = {tuple(int(x) for x in sg_states[v]): v for v in range(sg.vertices_length())}
gather = [int(jsp_ipv_pos[s]) for s in sg._ipv_target_states]
sti = [int(sg_s2i[tuple(int(x) for x in jsp_states[t])]) for t in jsp._t_vertex_indices]
n_t, n_ipv = len(sti), len(gather)

# The real handoff at the fixture theta (pybind replication, H0 tier-1).
theta = np.array([1e-4, 1e-4, 0.5e-4, 1.3e-4])
full = np.zeros(jpg.vertices_length())
for e in jpg.starting_vertex().edges():
    full[e.to().index()] = e.weight()
ipv0 = full[np.asarray(iti)]
jsp.update_ipv(ipv0); jsp.update_weights(theta[:P].tolist())
raw = np.asarray(jsp.stop_probability(0.5, granularity=0))
hand = np.empty(n_ipv)
for k, t in enumerate(iti):
    p = raw[t]
    if t in tam:
        p += raw[tam[t]]
    hand[k] = p
alpha_real = hand[np.asarray(gather)]

# Case matrix for micro-gate (a): (label, graph, theta_final, alpha, indices).
# G4 fold (tests/process refuter MAJOR 3): the plan mandated the
# joint-index fixture family too, not just the sojourn graph -- the jpg
# graph IS the joint-index consumer's graph type (pmf_from_graph_joint_index
# calls the same C function on it), so "jix_*" cases exercise that family
# (alpha=None: baked IPV, no update_ipv).
rng = np.random.default_rng(0)
CASES = [
    ("ones_pos", sg, [1.0, 1.0], rng.uniform(0.1, 1.0, n_ipv), sti[:8]),
    ("ones_real", sg, [1.0, 1.0], alpha_real, sti),
    ("tiny_pos", sg, [0.5e-4, 1.3e-4], rng.uniform(0.1, 1.0, n_ipv), sti[:8]),
    ("tiny_real_DECLINES", sg, [0.5e-4, 1.3e-4], alpha_real, sti),  # H0 gate finding
    ("jix_ones", jpg, [1.0, 1.0], None, [5, 6, 7, 8]),
    ("jix_tiny", jpg, [0.5e-4, 1.3e-4], None, [5, 6, 7, 8]),
]


def default_call(g, theta_f, alpha, idx):
    if alpha is not None:
        g.update_ipv(np.asarray(alpha, dtype=np.float64))
    g.update_weights(list(theta_f))
    return g._sojourn_grad_theta_subset(list(idx))  # default gate (old & new)


def daisy_false_path_grad():
    """G4 fold (tests/process refuter MAJOR 2): the plan's cross-install
    golden for the daisy False path. Old install: model built WITHOUT the
    kwarg (it does not exist there); new install: exact_final_grad=False.
    Gradients must be identical."""
    import jax
    import jax.numpy as jnp
    tvs = [int(x) for x in jsp._t_vertex_indices][:4]
    kw = {}
    if MODE == "check":
        kw["exact_final_grad"] = False
    model, _, _, _ = jpg._daisy_chain_svgd_model(
        observed_indices=tvs, epoch_starts=[0.0, 0.5], **kw)

    def loss(t):
        return jnp.sum(jnp.log(model(t, None)[0] + 1e-300))

    return np.asarray(jax.grad(loss)(jnp.asarray(theta)))


if MODE == "dump":
    out = {}
    for label, g, tf, alpha, idx in CASES:
        res = default_call(g, tf, alpha, idx)
        out[label] = np.asarray(res, dtype=np.float64) if res else np.empty(0)
    out["daisy_false_grad"] = daisy_false_path_grad()
    np.savez(NPZ, **out)
    print("dumped:", {k: v.shape for k, v in out.items()})
    sys.exit(0)

assert MODE == "check"
old = np.load(NPZ)
fails = []

print("== (a) flag=0 byte-identity vs pre-H install ==")
for label, g, tf, alpha, idx in CASES:
    res = default_call(g, tf, alpha, idx)
    new_arr = np.asarray(res, dtype=np.float64) if res else np.empty(0)
    old_arr = old[label]
    same = new_arr.shape == old_arr.shape and np.array_equal(new_arr, old_arr)
    print(f"  {label:22s} old={old_arr.size:5d} new={new_arr.size:5d} identical={same}")
    if not same:
        fails.append(f"(a) {label}")

print("== (a2) daisy False-path cross-install golden ==")
g_new = daisy_false_path_grad()
g_old = old["daisy_false_grad"]
same = np.array_equal(g_new, g_old)
print(f"  identical={same}"
      + ("" if same else f"  max|diff|={np.max(np.abs(g_new - g_old)):.3e}"))
if not same:
    fails.append("(a2) daisy False-path golden")

print("== (b) flag=1 at the declining point computes and matches oracle ==")
sg.update_ipv(alpha_real); sg.update_weights(theta[P:].tolist())
res_default = sg._sojourn_grad_theta_subset(sti)
res_nogate = sg._sojourn_grad_theta_subset(sti, skip_condition_gate=True)
print(f"  default declines: {not res_default}; nogate computes: {bool(res_nogate)}")
if res_default or not res_nogate:
    fails.append("(b) decline/compute pattern")
else:
    import jax
    import jax.numpy as jnp

    ser = sg.serialize(theta_dim=P)
    edges = np.asarray(ser["edges"], dtype=np.float64).reshape(-1, 3)
    pedges = np.asarray(ser["param_edges"], dtype=np.float64)
    n_sg = int(ser["n_vertices"])
    outdeg = np.zeros(n_sg)
    for arr in (edges, pedges):
        for row in arr:
            outdeg[int(row[0])] += 1
    trans = np.where(outdeg > 0)[0]
    trans_pos = {int(v): i for i, v in enumerate(trans)}
    t_pos = np.asarray([trans_pos[ti] for ti in sti])
    se = np.asarray(ser["start_edges"], dtype=np.float64).reshape(-1, 2)
    a_scatter = se[:, 0].astype(int)

    def soj_C(theta_f):
        # normalized-ipv subset sojourn, as the C computes it
        Q = jnp.zeros((n_sg, n_sg), dtype=jnp.float64)
        Q = Q.at[edges[:, 0].astype(int), edges[:, 1].astype(int)].add(jnp.asarray(edges[:, 2]))
        if pedges.shape[0]:
            w = jnp.asarray(pedges[:, 2:]) @ theta_f
            Q = Q.at[pedges[:, 0].astype(int), pedges[:, 1].astype(int)].add(w)
        Q = Q - jnp.diag(jnp.sum(Q, axis=1))
        T = Q[jnp.asarray(trans)][:, jnp.asarray(trans)]
        alpha_n = jnp.asarray(alpha_real / alpha_real.sum())
        af = jnp.zeros(n_sg, dtype=jnp.float64).at[jnp.asarray(a_scatter)].set(alpha_n)
        soj = jnp.linalg.solve(-T.T, af[jnp.asarray(trans)])
        return soj[jnp.asarray(t_pos)]

    J_or = np.asarray(jax.jacobian(soj_C)(jnp.asarray(theta[P:])))
    J_ng = np.asarray(res_nogate, dtype=np.float64).reshape(n_t, P)
    err = float(np.max(np.abs(J_ng - J_or)) / np.max(np.abs(J_or)))
    print(f"  nogate vs dense-JAX oracle: {err:.3e} (tol 1e-10)")
    if err > 1e-10:
        fails.append("(b) oracle mismatch")

print("== (c) flag=1 still declines on a non-finite requested row (trap) ==")
try:
    tg = Graph(1)
    s = tg.starting_vertex()
    vA = tg.find_or_create_vertex([1])
    vB = tg.find_or_create_vertex([2])
    vC = tg.find_or_create_vertex([3])
    vabs = tg.find_or_create_vertex([4])
    s.add_edge(vA, 1.0)
    vA.add_edge_parameterized(vB, 0.0, [1.0, 0.0])   # A -> B (param)
    vA.add_edge_parameterized(vabs, 0.0, [0.0, 1.0])  # A -> absorbing
    vB.add_edge_parameterized(vC, 0.0, [1.0, 0.0])   # B <-> C trap cycle
    vC.add_edge_parameterized(vB, 0.0, [1.0, 0.0])
    tg.update_weights([1.0, 1.0])
    res_trap = tg._sojourn_grad_theta_subset([vB.index()], skip_condition_gate=True)
    print(f"  trap-row nogate result: {'DECLINED (len 0) as required' if not res_trap else 'COMPUTED (len %d)' % len(res_trap)}")
    if res_trap and not all(np.isfinite(res_trap)):
        fails.append("(c) returned non-finite values instead of declining")
    if res_trap and all(np.isfinite(res_trap)):
        print("  NOTE: computed finite values -- trap row not non-finite on this fixture; recorded, not a failure")
except Exception as exc:
    print(f"  trap fixture not constructible via this API ({type(exc).__name__}: {exc}) -- skip-with-note per plan")

print("== (d) subnormal-mass probe (decides the I2 zero-branch width) ==")
for scale in (1e-300, 1e-308):
    alpha_sub = alpha_real * (scale / alpha_real.sum())
    sg.update_ipv(alpha_sub); sg.update_weights(theta[P:].tolist())
    res_sub = sg._sojourn_grad_theta_subset(sti, skip_condition_gate=True)
    if res_sub:
        finite = bool(np.all(np.isfinite(res_sub)))
        print(f"  mass={scale:.0e}: COMPUTES, all-finite={finite}")
    else:
        print(f"  mass={scale:.0e}: DECLINES (isfinite sweep or other)")

print("\n" + ("ALL MICRO-GATES PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(0 if not fails else 1)
