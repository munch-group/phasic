"""Batch G.1 G0-smoke -- front-door reachability probe (the D.3 lesson).

Plan: b3-batchG1-plan.md v2. Runs on the CURRENT (pre-G.1) install; no
new kwarg involved. Proves the two target shapes construct and evaluate
THROUGH Graph.svgd (never only via direct model construction -- the
exact failure mode that killed the D.3 plan: a validator rule can
reject a configuration long before the model builder runs):

  (a) single-epoch + exposure:  svgd(obs, exposure=..., epoch_starts=[0.0])
  (b) multi-epoch, no exposure: svgd(obs, epoch_starts=[0.0, 0.5])

Each must: pass validation, build the DAISY model (proven by a spy on
phasic.ffi_wrappers.compute_daisy_chain_sojourn_ffi, patched BEFORE the
call -- the builder from-imports it at construction), run one SVGD
iteration, and end with finite particles. GO = both shapes pass.
"""
import sys
from functools import partial
from itertools import combinations_with_replacement

import numpy as np

import phasic
from phasic import Graph, LogGaussPrior, Property, StateIndexer, set_log_level, with_ipv
import phasic.ffi_wrappers as fw

set_log_level("WARNING")
np.random.seed(17)

all_pairs = partial(combinations_with_replacement, r=2)
NR = 3
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
rl, mu, th = 4, 1e-4, 1 / 10_000
jpg_d = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=mu)
jpg_d.update_weights([th, mu])
jpt = jpg_d.joint_prob_table()
p = jpt["prob"] / jpt["prob"].sum()
sample = np.random.choice(jpt.index.values, 40, p=p.to_numpy())
obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
jpg = graph.joint_prob_graph(indexer, reward_limit=rl, mutation_rate=mu, discrete=False)

calls = []
orig = fw.compute_daisy_chain_sojourn_ffi


def spy(*a, **k):
    calls.append(1)
    return orig(*a, **k)


fw.compute_daisy_chain_sojourn_ffi = spy

common = dict(fixed=[(1, mu)], prior=LogGaussPrior(ci=[1 / 50_000, 1 / 5_000]),
              n_iterations=1, n_particles=4)
fails = []

print("== shape (a): single-epoch + exposure ==")
try:
    calls.clear()
    expo = np.random.default_rng(0).uniform(0.5, 2.0, len(obs))
    fit_a = jpg.svgd(obs, exposure=expo, exposure_param_index=1,
                     epoch_starts=[0.0], **common)
    finite = bool(np.all(np.isfinite(np.asarray(fit_a.particles))))
    daisy = len(calls) > 0
    print(f"  constructed+ran: True; particles finite: {finite}; "
          f"daisy spy calls: {len(calls)} (daisy-built: {daisy})")
    if not (finite and daisy):
        fails.append("(a) finite/daisy")
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    fails.append(f"(a) {type(exc).__name__}")

print("== shape (b): multi-epoch, no exposure ==")
try:
    calls.clear()
    fit_b = jpg.svgd(obs, epoch_starts=[0.0, 0.5], **common)
    finite = bool(np.all(np.isfinite(np.asarray(fit_b.particles))))
    daisy = len(calls) > 0
    print(f"  constructed+ran: True; particles finite: {finite}; "
          f"daisy spy calls: {len(calls)} (daisy-built: {daisy})")
    if not (finite and daisy):
        fails.append("(b) finite/daisy")
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    fails.append(f"(b) {type(exc).__name__}")

print("\n" + ("GO -- both front-door shapes reachable and finite"
              if not fails else f"NO-GO: {fails}"))
sys.exit(0 if not fails else 1)
