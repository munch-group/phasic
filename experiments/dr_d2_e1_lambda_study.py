"""Deferred-2 de-risk E1 (compact) -- granularity/lambda variation study.

Plan: deferred-2-daisy-intermediate-epoch-plan.md §4-E1 items 1+2, run
compactly (grid sweep in place of live SVGD trajectories -- the
trajectory recording is scoped as remaining E1 work in the findings).

Item 1: how much does the rate scale (and hence the auto-mode
granularity, `2*max(512, max_exit_rate)` per phasic.c:12648-12692) vary
across a prior-scale theta grid on the real JSP fixture? Large
variation means auto mode changes the embedded DTMC's *identity* with
theta -- FD tolerates that, an exact gradient cannot (plan §2.1).

Item 2: value error of the primal at PINNED granularity vs a high-λ
reference, at a benign theta -- sizes the safety-margin cost of
pinning (plan §7 risk 2).

Fixture: same nr=3 coalescent JSP as dr_d2_a2_value_test.py.
"""
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

set_log_level("WARNING")
import jax.numpy as jnp  # noqa: E402

all_pairs = partial(combinations_with_replacement, r=2)
NR = 3
MU = 1e-4
DTS = [0.5]

indexer = StateIndexer(
    lineages=[Property("descendants", min_value=1, max_value=NR)])


@with_ipv([NR] + [0] * (NR - 1))
def coal(state, indexer=None):
    t = []
    for i, j in all_pairs(range(indexer.lineages.state_length)):
        same = int(i == j)
        if same and state[i] < 2:
            continue
        if not same and (state[i] < 1 or state[j] < 1):
            continue
        new = state.copy()
        new[i] -= 1
        new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t


graph = Graph(coal, indexer=indexer)
jpg = graph.joint_prob_graph(indexer, reward_limit=3, mutation_rate=MU,
                             discrete=False)
jsp = jpg.joint_stop_prob_graph()
ipv_full = np.zeros(jpg.vertices_length(), dtype=np.float64)
for edge in jpg.starting_vertex().edges():
    ipv_full[edge.to().index()] = edge.weight()
initial_ipv = ipv_full[jsp._ipv_target_indices]


def max_exit_rate(theta):
    jsp.update_weights(list(theta))
    rates = []
    for v in range(jsp.vertices_length()):
        vert = jsp.vertex_at(v)
        rates.append(sum(e.weight() for e in vert.edges()))
    return max(rates)


print("== E1 item 1: rate scale across a prior-scale theta grid ==")
print("   (auto granularity = 2*max(512, max_exit_rate) -- theta-dependent)")
for th0 in (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0):
    r = max_exit_rate([th0, MU])
    auto_g = 2 * max(512.0, r)
    print(f"  theta0={th0:8.0e}: max_exit_rate={r:12.6g}  "
          f"auto-granularity={auto_g:12.6g}"
          f"{'  (floor-bound: identity FIXED)' if r <= 512 else ''}")

print("\n== E1 item 2: pinned-granularity value convergence (benign theta) ==")
theta = np.array([[1e-4, MU], [2e-4, MU]])


def primal(gran):
    p = jsp.daisy_chain_joint_probs(
        epoch_thetas=jnp.asarray(theta), epoch_dts=DTS,
        initial_ipv=initial_ipv, granularity=int(gran),
        final_read='sojourn')
    return float(jnp.sum(np.asarray(p) * np.arange(1.0, len(p) + 1.0)))


ref = primal(1 << 19)          # 524288 -- high-λ reference
print(f"  reference (g=524288): {ref:.15e}")
for g in (512, 1024, 2048, 4096, 8192, 16384, 65536):
    v = primal(g)
    print(f"  g={g:6d}: rel value err vs ref = {abs(v - ref) / abs(ref):.3e}")

print("\nE1 COMPACT STUDY COMPLETE")
