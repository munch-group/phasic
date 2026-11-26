"""Review 4 probe: does the single-epoch joint-index (discrete jpg) and the
discrete/continuous pdf FD backward crash today at theta slot in (0, 1e-7)?"""
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
for disc in (True, False):
    jpg = graph.joint_prob_graph(indexer, reward_limit=5, event_rate_coefficient=1e-4, discrete=disc)
    obs = jpg._map_joint_observations_to_indices(
        [[1, 0, 0], [2, 1, 0], [0, 0, 1]] if False else
        jpg.joint_prob_table().iloc[:5][jpg.joint_prob_table().columns[:-1]].to_numpy().tolist())
    m = Graph.pmf_from_graph_joint_index(jpg, theta_dim=2, observed_indices=None)
    def ll(th):
        v, _ = m(th, jnp.asarray(obs))
        return jnp.sum(jnp.log(jnp.abs(v) + 1e-300))
    for th in ([1e-4, 1e-4], [1e-4, 5e-8], [5e-8, 1e-4]):
        try:
            print("jix discrete=%s" % disc, th, np.asarray(jax.grad(ll)(jnp.array(th))), flush=True)
        except Exception as e:
            print("jix discrete=%s" % disc, th, "RAISES", type(e).__name__, str(e)[:150], flush=True)
