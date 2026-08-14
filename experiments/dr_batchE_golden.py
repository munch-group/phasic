"""Batch E golden gate -- cross-install bitwise identity of the
no-kwarg default svgd-built BAKED joint-index model gradient (plan v2 I3-9;
the H micro-gate (a2) template). The compared artifact is a SINGLE
model-gradient call; the 1-iteration svgd run below only BUILDS the
model -- its RNG never enters the compared value.

  dump  <npz>  -- run under the PRE-E install (main checkout).
  check <npz>  -- run under the branch install; bitwise compare.
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
MODE, NPZ = sys.argv[1], sys.argv[2]
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
MU, TH = 1e-4, 1 / 10_000
jpg_d = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU)
jpg_d.update_weights([TH, MU])
jpt = jpg_d.joint_prob_table()
p = jpt["prob"] / jpt["prob"].sum()
sample = np.random.choice(jpt.index.values, 30, p=p.to_numpy())
obs = jpt.loc[sample, jpt.columns[:-1]].to_numpy().tolist()
jpg = graph.joint_prob_graph(indexer, reward_limit=4, mutation_rate=MU,
                             discrete=False)

fit = jpg.svgd(obs, n_iterations=1, n_particles=4)
th = jnp.asarray([TH, MU])


def loss(t):
    out, _ = fit.model(t, None)
    return jnp.sum(jnp.log(out + 1e-300))


g = np.asarray(jax.grad(loss)(th))

if MODE == "dump":
    np.savez(NPZ, g=g)
    print(f"dumped grad {g.tolist()}")
else:
    old = np.load(NPZ)["g"]
    same = np.array_equal(g, old)
    print(f"identical={same}"
          + ("" if same else f" max|diff|={np.max(np.abs(g - old)):.3e}"))
    sys.exit(0 if same else 1)
