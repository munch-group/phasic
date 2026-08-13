"""Batch H de-risk H1 -- cost/instrument study.

Plan: b3-batchH-plan.md v2. Findings -> b3-batchH-findings.md.

H1(a) [MANDATORY, master plan 16b item 3 "evaluate during Batch H's
design"]: the offset-tape conversion (ptd_pcg_convert_to_offset) runs
fresh inside every ptd_sojourn_grad_theta_subset call. It cannot be
isolated from Python (no C-level timer exposed) -- what CAN be decided
with evidence is whether the TOTAL warm per-call adjoint cost (which
includes the conversion) is material relative to the FD backward it
replaces. If the whole adjoint call is a small fraction of one FD
backward, caching the conversion cannot buy anything measurable, and
16b item 3 is DECLINED-with-evidence for Batch H's regime.

H1(b) [CONTINGENT on H0(iv)]: J_ipv instruments -- SKIPPED. H0(iv)
showed the primary variant (exact final cols + full-chain-FD earlier
cols) reaches ~1e-12 composed error with improvement factors of
3.6e5-5.0e5x, and the earlier columns' FD error (2.4e-10) is already
4 orders below the final columns' FD error it would improve -- the
ipv_bar variant has nothing left to buy on this model class.

Also measured (feeds H2's cost model): the Python-replication handoff
extraction (intermediate epochs via pybind) vs one fused FFI call, and
the projected exact-block backward add-on vs the plan's bound
(<= 0.5x one full-FD backward).

Timings: median of N warm reps, first-call (cold tape) reported
separately. Gate lifted (PHASIC_CONDITION_THRESHOLD=1e300) for adjoint
timings -- H0 established the default gate declines 100% of realistic
calls here; timing the decline would mismeasure.
"""
import json
import os
import time
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv
from phasic.ffi_wrappers import (
    _make_json_serializable,
    compute_daisy_chain_sojourn_ffi,
)

import jax.numpy as jnp

set_log_level("WARNING")
N_REP = 20


def median_time(fn, n=N_REP):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def build_fixture(nr_samples):
    all_pairs = partial(combinations_with_replacement, r=2)
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
    jpg = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=1e-4, discrete=False)
    return jpg


def study(nr_samples, theta_flat):
    jpg = build_fixture(nr_samples)
    P = jpg.param_length()
    jsp = jpg.joint_stop_prob_graph()
    sg = jpg.joint_sojourn_graph()

    ipv_target_indices = [int(x) for x in jsp._ipv_target_indices]
    t_aux_map = {int(k): int(v) for k, v in jsp._t_aux_map.items()}
    t_vertex_indices = [int(x) for x in jsp._t_vertex_indices]
    jsp_states = jsp.states()
    jsp_ipv_pos = {tuple(int(x) for x in jsp_states[v]): k for k, v in enumerate(ipv_target_indices)}
    sg_states = sg.states()
    sg_state_to_idx = {tuple(int(x) for x in sg_states[v]): v for v in range(sg.vertices_length())}
    sojourn_jsp_gather = [int(jsp_ipv_pos[s]) for s in sg._ipv_target_states]
    sojourn_t_indices = [int(sg_state_to_idx[tuple(int(x) for x in jsp_states[t])]) for t in t_vertex_indices]

    self_ipv_full = np.zeros(jpg.vertices_length())
    for edge in jpg.starting_vertex().edges():
        self_ipv_full[edge.to().index()] = edge.weight()
    initial_ipv = self_ipv_full[np.asarray(ipv_target_indices)]

    structure = _make_json_serializable(jsp.serialize(theta_dim=P))
    structure["_daisy_chain"] = {
        "n_epochs": 2, "param_length": int(P), "t_eval": 10.0,
        "granularity": 0, "epoch_dts": [0.5],
        "ipv_target_indices": ipv_target_indices,
        "t_aux_keys": list(t_aux_map.keys()),
        "t_aux_values": [t_aux_map[k] for k in t_aux_map.keys()],
        "t_vertex_indices": t_vertex_indices,
        "sojourn_jsp_gather": sojourn_jsp_gather,
        "sojourn_t_indices": sojourn_t_indices,
    }
    structure_json = json.dumps(structure)
    sojourn_json = json.dumps(_make_json_serializable(sg.serialize(theta_dim=P)))
    theta_j = jnp.asarray(theta_flat, dtype=jnp.float64)
    ipv_j = jnp.asarray(initial_ipv, dtype=jnp.float64)

    def ffi_call():
        np.asarray(compute_daisy_chain_sojourn_ffi(structure_json, sojourn_json, theta_j, ipv_j))

    def intermediate_handoff():
        jsp.update_ipv(initial_ipv)
        jsp.update_weights(theta_flat[:P].tolist())
        raw = np.asarray(jsp.stop_probability(0.5, granularity=0))
        new = np.empty(len(ipv_target_indices))
        for k, tgt in enumerate(ipv_target_indices):
            p = raw[tgt]
            if tgt in t_aux_map:
                p += raw[t_aux_map[tgt]]
            new[k] = p
        return new[np.asarray(sojourn_jsp_gather)]

    alpha = intermediate_handoff()

    def fwd_sojourn():
        sg.update_ipv(alpha)
        sg.update_weights(theta_flat[P:].tolist())
        np.asarray(sg.expected_sojourn_time(sojourn_t_indices))

    def adjoint():
        sg.update_ipv(alpha)
        sg.update_weights(theta_flat[P:].tolist())
        out = sg._sojourn_grad_theta_subset(sojourn_t_indices)
        assert out, "adjoint declined during timing (gate should be lifted)"

    ffi_call()  # warm builder caches
    t_ffi = median_time(ffi_call)
    t_inter = median_time(intermediate_handoff)
    t_fwd = median_time(fwd_sojourn)

    os.environ["PHASIC_CONDITION_THRESHOLD"] = "1e300"
    try:
        t0 = time.perf_counter(); adjoint(); t_adj_cold = time.perf_counter() - t0
        t_adj = median_time(adjoint)
    finally:
        os.environ.pop("PHASIC_CONDITION_THRESHOLD", None)

    n_slots = len(theta_flat)
    t_fd_bwd = 2 * n_slots * t_ffi                       # _autodiff_bwd: 2 FFI calls per slot
    t_exact_addon = t_inter + t_adj                      # handoff extraction + exact block
    print(f"\n-- nr_samples={nr_samples}: jpg n={jpg.vertices_length()} jsp n={jsp.vertices_length()} "
          f"sg n={sg.vertices_length()} n_t={len(sojourn_t_indices)} n_ipv={len(ipv_target_indices)}")
    print(f"   fused FFI forward:              {t_ffi*1e3:9.3f} ms")
    print(f"   handoff extraction (pybind):    {t_inter*1e3:9.3f} ms")
    print(f"   forward subset sojourn:         {t_fwd*1e3:9.3f} ms")
    print(f"   exact adjoint (warm):           {t_adj*1e3:9.3f} ms   (cold first call: {t_adj_cold*1e3:.3f} ms)")
    print(f"   full-FD backward (2*{n_slots} FFI):    {t_fd_bwd*1e3:9.3f} ms")
    print(f"   exact-block add-on / FD bwd:    {t_exact_addon/t_fd_bwd:9.3%}   (plan bound: <=50%)")
    print(f"   adjoint / FD bwd (16b item 3):  {t_adj/t_fd_bwd:9.3%}")
    return t_adj, t_fd_bwd


print("== H1 cost study ==")
study(4, np.array([1e-4, 1e-4, 0.5e-4, 1.3e-4]))
study(5, np.array([1e-4, 1e-4, 0.5e-4, 1.3e-4]))
study(6, np.array([1e-4, 1e-4, 0.5e-4, 1.3e-4]))
print("\ndecision inputs recorded; see b3-batchH-findings.md H1 section")
