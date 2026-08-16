"""Deferred-2 de-risk E2 -- cost-model instrumentation + decision table.

Plan: deferred-2-daisy-intermediate-epoch-plan.md §4-E2. Measures the
structural drivers of the backprop-through-time cost regime on real
epoch fixtures at STAGED sizes (memory mandate: small first, stop on a
time-box; no gradient pipelines built here -- structure only):

  n      = JSP vertex count (the per-epoch state dimension)
  n_ipv  = |jsp._ipv_target_indices| (the handoff vector's length --
           the seed count forward-mode must carry, plus n_params)
  E      = parameterized edge count
  k      = steps per epoch = granularity * dt (the C loop is
           `while (time > ctx->time) step()`, ceil-like; granularity
           auto-resolves to 2*max(512, max_exit_rate) -- measured
           constant at 1024 on this fixture family, D2-E1)

and turns them into the plan's decision table:

  forward-mode  : memory O(n * (n_params + n_ipv)), history-free;
                  compute ~ (n_params + n_ipv) tangent replays
  checkpointed  : memory O(n * sqrt(k)) at ~2x compute
  reverse       : memory O(k * n) naive (infeasible at production k)

The crossover is (n_params + n_ipv) vs ~2*sqrt(k): forward-mode wins
while the seed count is below the checkpoint overhead.

Usage: python dr_d2_e2_cost_model.py [--max-nr N]
"""
import argparse
import sys
import time
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, Property, StateIndexer, set_log_level, with_ipv

set_log_level("WARNING")

MU = 1e-4
DT = 0.5
BUILD_TIMEBOX_S = 120.0

ap = argparse.ArgumentParser()
ap.add_argument("--max-nr", type=int, default=6)
args = ap.parse_args()

all_pairs = partial(combinations_with_replacement, r=2)


def build_jsp(nr):
    indexer = StateIndexer(
        lineages=[Property("descendants", min_value=1, max_value=nr)])

    @with_ipv([nr] + [0] * (nr - 1))
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

    g = Graph(coal, indexer=indexer)
    jpg = g.joint_prob_graph(indexer, reward_limit=3, mutation_rate=MU,
                             discrete=False)
    jsp = jpg.joint_stop_prob_graph()
    return jsp


def max_exit_rate(jsp, theta):
    jsp.update_weights(list(theta))
    return max(sum(e.weight() for e in jsp.vertex_at(v).edges())
               for v in range(jsp.vertices_length()))


print("== D2-E2: cost-model instrumentation (staged sizes) ==")
print(f"{'nr':>3} {'n(JSP)':>8} {'n_ipv':>7} {'P':>3} {'lambda':>8} "
      f"{'k=lam*dt':>9} {'seeds=P+n_ipv':>14} {'2*sqrt(k)':>10} "
      f"{'winner':>10} {'build_s':>8}")

rows = []
for nr in range(3, args.max_nr + 1):
    t0 = time.time()
    try:
        jsp = build_jsp(nr)
    except Exception as exc:
        print(f"{nr:>3}  BUILD FAILED: {type(exc).__name__}: {str(exc)[:60]}")
        break
    dt_build = time.time() - t0
    n = jsp.vertices_length()
    n_ipv = len(jsp._ipv_target_indices)
    P = jsp.param_length()
    rate = max_exit_rate(jsp, [1e-4] + [MU] * (P - 1))
    lam = 2.0 * max(512.0, rate)
    k = lam * DT
    seeds = P + n_ipv
    ckpt = 2.0 * np.sqrt(k)
    winner = "forward" if seeds < ckpt else "checkpoint"
    print(f"{nr:>3} {n:>8} {n_ipv:>7} {P:>3} {lam:>8.0f} {k:>9.0f} "
          f"{seeds:>14} {ckpt:>10.1f} {winner:>10} {dt_build:>8.1f}")
    rows.append(dict(nr=nr, n=n, n_ipv=n_ipv, P=P, lam=lam, k=k,
                     seeds=seeds, ckpt=ckpt, winner=winner))
    if dt_build > BUILD_TIMEBOX_S:
        print(f"    (time-box {BUILD_TIMEBOX_S:.0f}s exceeded at nr={nr}; "
              f"stopping the ladder -- memory mandate)")
        break

print()
print("== memory footprints at the measured sizes (doubles) ==")
print(f"{'nr':>3} {'fwd O(n*seeds)':>16} {'ckpt O(n*sqrt k)':>18} "
      f"{'naive O(k*n)':>14}")
for r in rows:
    fwd = r['n'] * r['seeds']
    ck = r['n'] * np.sqrt(r['k'])
    naive = r['k'] * r['n']
    print(f"{r['nr']:>3} {fwd:>16,.0f} {ck:>18,.0f} {naive:>14,.0f}")

print()
print("DECISION TABLE READING:")
if rows:
    lo, hi = rows[0], rows[-1]
    print(f"  seed count grows with n_ipv ({lo['seeds']} at nr={lo['nr']} "
          f"-> {hi['seeds']} at nr={hi['nr']}); the checkpoint threshold "
          f"2*sqrt(k) is ~{hi['ckpt']:.0f} and is theta-INDEPENDENT on "
          f"this fixture family (lambda pinned at the 1024 floor, D2-E1).")
    crossover = [r for r in rows if r['seeds'] >= r['ckpt']]
    if crossover:
        print(f"  CROSSOVER observed at nr={crossover[0]['nr']} "
              f"(n_ipv={crossover[0]['n_ipv']}): checkpointed-reverse "
              f"becomes the cheaper mode at and above this size.")
    else:
        print("  NO crossover in the measured range: forward-mode is the "
              "cheaper mode at every size measured; the crossover lies "
              f"above n_ipv={hi['n_ipv']} (extrapolate: it needs "
              f"n_ipv >~ {hi['ckpt']:.0f}).")
print("\nE2 COST-MODEL COMPLETE")
