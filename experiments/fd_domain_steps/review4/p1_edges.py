"""Review 4 probe: (a) today's daisy FD at slot = 0 (fit-time fixed 0 on
epoch_model) and (b) at slot exactly 1e-7 and just below (does today crash?),
(c) slot used additively with a large slot. Plain epoch_model, master."""
import os, sys, traceback
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from itertools import combinations_with_replacement
from phasic import Graph, with_ipv, StateIndexer, Property, set_log_level
set_log_level("ERROR")
all_pairs = partial(combinations_with_replacement, r=2)
nr = 4
indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])
@with_ipv([nr] + [0] * (nr - 1))
def coal(state, indexer=None):
    t = []
    for i, j in all_pairs(range(indexer.lineages.state_length)):
        same = int(i == j)
        if same and state[i] < 2: continue
        if not same and (state[i] < 1 or state[j] < 1): continue
        new = state.copy(); new[i] -= 1; new[j] -= 1
        new[min(i + j + 1, state.size - 1)] += 1
        t.append([new, [state[i] * (state[j] - same) / (1 + same)]])
    return t
graph = Graph(coal, indexer=indexer)
mu = 1e-4
jpg_d = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=mu)
jpg_d.update_weights([1 / 10_000, mu])
obs = jpg_d.joint_prob_table().iloc[:12][jpg_d.joint_prob_table().columns[:-1]].to_numpy().tolist()
jpg = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=mu, discrete=False)
model = jpg.epoch_model(obs, epoch_starts=[0, 0.5]).model
def loglik(theta):
    per_obs, _ = model(theta, None)
    return jnp.sum(jnp.log(per_obs + 1e-300))
g = jax.grad(loglik)
for th in ([1e-4, 0.0, 1e-4, 1e-4], [1e-4, 1e-7, 1e-4, 1e-4],
           [1e-4, 1e-7 - 1e-13, 1e-4, 1e-4], [1e-4, 1e-7 - 1e-11, 1e-4, 1e-4],
           [1e-4, 5e-8, 1e-4, 1e-4]):
    try:
        print(th, "->", np.asarray(g(jnp.array(th))), flush=True)
    except Exception as e:
        print(th, "-> RAISES", type(e).__name__, str(e)[:120], flush=True)
