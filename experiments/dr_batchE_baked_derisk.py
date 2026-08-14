"""Batch E de-risk E0 -- baked-mode exact gradient (plan v2).

Runs on the PRE-E install (main checkout). Fixture: the G.1
`_base_graph` coalescent jpg (NR=3, reward_limit=4, mutation_rate=1e-4)
built with `discrete=False` (the plan's CRITICAL caveat: a default jpg
is was_dph and measures the wrong thing).

  (i)   Oracle check of the PROPOSED baked backward (scatter-add +
        quotient at unique granularity, C adjoint at uniq) against the
        SHIPPED non-baked committed exact path (itself F-gated) at the
        same per-obs granularity, and against tight central FD of the
        baked forward. Target <=1e-9 vs shipped-exact.
        Degenerate shapes: n_unique==1, n_obs==1, unsorted/duplicated.
  (ii)  Gate/probe dynamic-range sweep: the C adjoint at
        union(uniq, all_terminal) over theta in {ones, 1e-2, 1e-4,
        mixed [1e-4, 5e-2]} -- ANY decline => STOP (user decision).
        Trap disposition: scan the fixture's per-vertex sojourn for
        non-finite rows; record presence/absence.
  (iii) Front-door smoke: Graph.svgd(obs) reaches the BAKED leaf --
        spy on phasic.ffi_wrappers.compute_sojourn_times_ffi (patched
        BEFORE the call), assert a spied index-array length ==
        n_unique < n_obs.
"""
import sys
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

import jax
import jax.numpy as jnp

set_log_level("WARNING")
np.random.seed(17)

all_pairs = partial(combinations_with_replacement, r=2)
NR = 3
MU, TH = 1e-4, 1 / 10_000
indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=NR)])


@with_ipv([NR] + [0] * (NR - 1))
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
jpg = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU,
                             discrete=False)
assert not getattr(jpg, 'was_dph', True) or not jpg.get_was_dph(), \
    "fixture must be continuous (discrete=False)"

# all-terminal set, exactly as the builder derives it
all_term = []
for v in jpg.vertices():
    for e in v.edges():
        if len(e.to().edges()) == 0:
            all_term.append(v.index())
            break
all_term = sorted(set(all_term))
P = jpg.param_length()
print(f"fixture: jpg n={jpg.vertices_length()} P={P} n_terminal={len(all_term)}")

# observations: t-vertices with heavy duplication, UNSORTED
rng = np.random.default_rng(3)
obs = rng.choice(all_term, 25, replace=True).astype(np.int64)
uniq, inverse = np.unique(obs, return_inverse=True)
print(f"obs: n_obs={obs.size} n_unique={uniq.size}")

fails = []
theta_base = np.array([TH, MU])


def proposed_baked_grad(theta_np, obs_idx, g_visits):
    """The E backward: scatter-add + quotient at unique granularity,
    using the shipped C adjoint on a private clone."""
    u, inv = np.unique(obs_idx, return_inverse=True)
    g_uniq = np.zeros(u.size)
    np.add.at(g_uniq, inv, g_visits)
    cl = jpg.clone()
    cl.update_weights(list(theta_np))
    soj_u = np.asarray(cl.expected_sojourn_time([int(x) for x in u]))
    soj_all = np.asarray(cl.expected_sojourn_time(list(all_term)))
    norm = soj_all.sum()
    union_idx = np.union1d(u, np.asarray(all_term))
    raw = cl._sojourn_grad_theta_subset([int(x) for x in union_idx])
    if not raw:
        raise RuntimeError("C adjoint declined in proposed backward")
    J_union = np.asarray(raw).reshape(union_idx.size, P)
    pos = {int(v): k for k, v in enumerate(union_idx)}
    J_u = J_union[[pos[int(v)] for v in u], :]
    J_all = J_union[[pos[int(v)] for v in all_term], :]
    dnorm = J_all.sum(axis=0)
    d_probs = (J_u * norm - soj_u[:, None] * dnorm[None, :]) / norm**2
    return d_probs.T @ g_uniq


print("== (i) proposed baked backward vs shipped non-baked exact + tight FD ==")
model_nb = Graph.pmf_from_graph_joint_index(jpg, exact_grad=True)
g_visits = np.linspace(0.5, 1.5, obs.size)


def loss_nb(t):
    out, _ = model_nb(t, jnp.asarray(obs))
    return jnp.sum(jnp.asarray(g_visits) * out)


for label, o in (("main", obs), ("n_unique==1", np.full(7, int(all_term[0]))),
                 ("n_obs==1", np.asarray([int(all_term[1])]))):
    gv = np.linspace(0.5, 1.5, o.size)

    def loss_o(t, _o=o, _gv=gv):
        out, _ = model_nb(t, jnp.asarray(_o))
        return jnp.sum(jnp.asarray(_gv) * out)

    g_shipped = np.asarray(jax.grad(loss_o)(jnp.asarray(theta_base)))
    g_prop = proposed_baked_grad(theta_base, o, gv)
    rel = np.max(np.abs(g_prop - g_shipped)) / np.max(np.abs(g_shipped))
    status = "PASS" if rel < 1e-9 else "FAIL"
    print(f"  [{label:12s}] proposed vs shipped-exact rel: {rel:.3e} {status}")
    if rel >= 1e-9:
        fails.append(f"(i) {label}")

# tight FD cross-check on the main shape (baked forward simulated:
# probs at uniq gathered to obs)
def baked_forward(theta_np):
    cl = jpg.clone()
    cl.update_weights(list(theta_np))
    soj_u = np.asarray(cl.expected_sojourn_time([int(x) for x in uniq]))
    norm = np.asarray(cl.expected_sojourn_time(list(all_term))).sum()
    return (soj_u / norm)[inverse]


eps = 1e-6
g_fd = np.empty(P)
for k in range(P):
    tp = theta_base.copy(); tp[k] += eps
    tm = theta_base.copy(); tm[k] -= eps
    g_fd[k] = np.sum(g_visits * (baked_forward(tp) - baked_forward(tm))) / (2 * eps)
g_prop_main = proposed_baked_grad(theta_base, obs, g_visits)
rel_fd = np.max(np.abs(g_prop_main - g_fd)) / np.max(np.abs(g_fd))
print(f"  [tight-FD    ] proposed vs central FD rel: {rel_fd:.3e} (FD-limited)")

print("== (ii) gate/probe dynamic-range sweep + trap disposition ==")
union_probe = np.union1d(uniq, np.asarray(all_term))
for label, th in (("ones", [1.0, 1.0]), ("1e-2", [1e-2, 1e-2]),
                  ("1e-4", [TH, MU]), ("mixed", [1e-4, 5e-2])):
    cl = jpg.clone()
    cl.update_weights(list(th))
    ok = bool(cl._sojourn_grad_theta_subset([int(x) for x in union_probe]))
    print(f"  theta={label:5s}: {'COMPUTES' if ok else 'DECLINE'}")
    if not ok:
        fails.append(f"(ii) decline at {label}")
cl = jpg.clone(); cl.update_weights([TH, MU])
soj_full = np.asarray(cl.expected_sojourn_time(
    list(range(jpg.vertices_length()))))
n_nonfinite = int(np.sum(~np.isfinite(soj_full)))
print(f"  trap disposition: {n_nonfinite} non-finite sojourn rows on the "
      f"fixture ({'trap-free' if n_nonfinite == 0 else 'TRAPS PRESENT'})")

print("== (iii) front-door smoke: svgd reaches the BAKED leaf ==")
import phasic.ffi_wrappers as fw
seen_lengths = []
orig = fw.compute_sojourn_times_ffi


def spy(sd, t, vi, *a, **k):
    try:
        seen_lengths.append(int(np.asarray(vi).size))
    except Exception:
        pass
    return orig(sd, t, vi, *a, **k)


fw.compute_sojourn_times_ffi = spy
try:
    jpg_d = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU)
    jpg_d.update_weights([TH, MU])
    jpt = jpg_d.joint_prob_table()
    p = jpt["prob"] / jpt["prob"].sum()
    sample = np.random.choice(jpt.index.values, 30, p=p.to_numpy())
    obs_tuples = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
    fit = jpg.svgd(obs_tuples, n_iterations=1, n_particles=4)
    n_u_expected = len(set(map(tuple, obs_tuples)))
    baked_seen = any(l < len(obs_tuples) for l in seen_lengths)
    print(f"  spied index lengths (sample): {sorted(set(seen_lengths))[:6]}; "
          f"n_obs={len(obs_tuples)}, distinct outcomes={n_u_expected}")
    if not seen_lengths:
        fails.append("(iii) spy dead")
    elif not baked_seen:
        fails.append("(iii) no dedup-length call seen -- baked leaf not engaged?")
finally:
    fw.compute_sojourn_times_ffi = orig

print("\n" + ("E0 GO" if not fails else f"E0 NO-GO: {fails}"))
sys.exit(0 if not fails else 1)
