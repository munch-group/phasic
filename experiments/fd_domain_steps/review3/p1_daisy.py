"""P1: epoch (daisy-chain) model from the strict-xfail pin. Reproduce master's
crash, then prototype the positive-domain rule (full and capped) and compare
with exact_final_grad (final-epoch slots) and a Richardson reference."""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from functools import partial
from itertools import combinations_with_replacement
import numpy as np, jax, jax.numpy as jnp
from phasic import Graph, with_ipv, StateIndexer, Property, set_log_level
from common import *
set_log_level("ERROR")

nr = 4
indexer = StateIndexer(lineages=[Property("descendants", min_value=1, max_value=nr)])
all_pairs = partial(combinations_with_replacement, r=2)
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
tab = jpg_d.joint_prob_table()
obs = tab.iloc[:12][tab.columns[:-1]].to_numpy().tolist()
jpg = graph.joint_prob_graph(indexer, reward_limit=5, mutation_rate=mu, discrete=False)
em = jpg.epoch_model(obs, epoch_starts=[0, 0.5])
model = em.model
oi = jpg._map_joint_observations_to_indices(obs)
m_efg, td, _, _ = jpg._daisy_chain_svgd_model(observed_indices=oi, epoch_starts=np.array([0, 0.5]),
                                             user_prior=None, user_fixed=None, exact_final_grad=True)

def F(th):
    p, _ = model(jnp.asarray(th), None)
    return np.asarray(p, dtype=np.float64)
def ll(th): return jnp.sum(jnp.log(model(th, None)[0] + 1e-300))
def ll_efg(th):
    out = m_efg(th); p = out[0] if isinstance(out, tuple) else out
    return jnp.sum(jnp.log(p + 1e-300))
g_today = jax.grad(ll); g_efg = jax.grad(ll_efg)

CASES = [[1e-4, 1e-4, 1e-4, 1e-4], [1.0, 1e-8, 1.0, 1e-8], [1.0, 1e-9, 1.0, 1e-9],
         [1e-9, 1e-4, 1e-4, 1e-4], [0.3, 2e-3, 1.5, 5e-3], [1.0, 1e-3, 1.0, 1e-3]]
SS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for base in CASES:
    th = np.array(base, float)
    cot = 1.0 / (F(th) + 1e-300)
    print(f"\n=== theta={base}", flush=True)
    t0 = time.time()
    try:
        gt = np.array(g_today(jnp.asarray(th))); err = ''
    except Exception as e:
        gt = np.full(4, np.nan); err = type(e).__name__ + ': ' + str(e).split('\n')[0][:150]
    print(f"  master jax.grad: {gt}  {err} [{time.time()-t0:.1f}s]")
    try:
        ge = np.array(g_efg(jnp.asarray(th)))
    except Exception as e:
        ge = np.full(4, np.nan); print('  efg failed', type(e).__name__, str(e)[:120])
    print(f"  exact_final_grad model (final slots 2,3 exact): {ge}")
    for i in range(4):
        ref, spread = richardson_log(F, th, i, cot, s=1e-2)
        line = f"  slot{i} th={th[i]:.1e} ref={ref: .8e} (spread {spread:.1e})"
        if i >= 2: line += f" efg-vs-ref {rel(ge[i], ref):.1e}"
        line += f" | master {rel(gt[i], ref):.1e}"
        for s in SS:
            line += f" | s={s:.0e} full {rel(fd_log(F, th, i, cot, s), ref):.1e} cap {rel(fd_log(F, th, i, cot, s, cap=1e-7), ref):.1e}"
        print(line, flush=True)
