"""De-risk for the tot_reward_limit trash-edge fix (fix/tot-reward-trash).

Defect: joint_prob_graph() evaluated the reward/trash rates at t-vertices
(base state absorbing). Under a total cap, a t-vertex sitting at the cap
got an edge to the trash vertex, so on a CONTINUOUS joint graph its
sojourn read (and hence the single-epoch likelihood) was divided by the
extra exit rate. No time passes after base absorption, so a t-vertex can
carry no mutation and no overflow.

Checks (printed), against independent closed forms:
  S  structure: no t-vertex has an edge to the trash vertex
  1  n=2, single-epoch pmf_from_graph_joint_index, normalised, every cap
     combination, continuous and discrete
  3  n=3, two features, same
  E  epoch path (epoch_model, one epoch) still agrees (it never used the
     t-vertex out-edges)

Run inside the worktree env:  pixi run python experiments/dr_tot_reward_trash.py
"""
import os
import sys
import numpy as np
import jax.numpy as jnp
from scipy.special import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests', 'pytest'))
from test_notebook_model_recovery import coalescent_indexed  # noqa: E402
from phasic import Graph  # noqa: E402


def p_n2(k, c, mu):
    q = 2 * mu / (c + 2 * mu)
    return (1 - q) * q ** k


def p_n3(k1, k2, c, mu):
    a = mu / (c + mu)
    q = 2 * mu / (c + 2 * mu)
    return sum((1 - a) * a ** n * (1 - q) * q ** (k1 - n + k2)
               * comb(k1 - n + k2, k2) * 0.5 ** (k1 - n + k2)
               for n in range(k1 + 1))


def allowed(row, rl, tl):
    return (rl is None or all(x <= rl for x in row)) and sum(row) <= tl


def t_vertices(j):
    return sorted({v.index() for v in j.vertices() for e in v.edges()
                   if len(e.to().edges()) == 0 and len(v.edges()) > 0
                   and v.index() != j.starting_vertex().index()})


def trash_index(j):
    for v in j.vertices():
        if v.state().sum() == 0 and v.edges_length() == 1:
            ch = v.edges()[0].to()
            if ch.state().sum() == 0 and ch.edges_length() == 1 \
                    and ch.edges()[0].to().index() == v.index():
                return v.index()
    return None


def rewards_of(j, vi, n_reward):
    return tuple(int(x) for x in j.vertex_at(vi).state()[-n_reward:])


worst = 0.0
CAPS = [dict(reward_limit=2), dict(tot_reward_limit=2),
        dict(reward_limit=2, tot_reward_limit=2),
        dict(reward_limit=2, tot_reward_limit=3), dict(tot_reward_limit=4)]

for n, cf, c, mu in [(2, p_n2, 1.0, 0.7), (3, p_n3, 1.1, 0.6)]:
    g, idx = coalescent_indexed(n)
    print(f'\n==== n={n}  (c={c}, mu={mu})')
    for kw in CAPS:
        rl, tl = kw.get('reward_limit'), kw.get('tot_reward_limit', np.inf)
        for disc in (False, True):
            j = g.joint_prob_graph(idx, discrete=disc, **kw)
            tv = t_vertices(j)
            tr = trash_index(j)
            bad = [v for v in tv if any(e.to().index() == tr
                                        for e in j.vertex_at(v).edges())]
            n_reward = j.state_length() - g.state_length()
            rows = [rewards_of(j, v, n_reward) for v in tv]
            feat = [r[:n - 1] for r in rows]  # descendants_1..n-1 (size-n count is always 0)
            assert all(sum(r[n - 1:]) == 0 for r in rows)
            m = Graph.pmf_from_graph_joint_index(j, theta_dim=2)
            p = np.asarray(m(jnp.array([c, mu]), jnp.array(tv, dtype=jnp.int32))[0])
            ref = np.array([cf(*r, c, mu) for r in feat])
            grid = [r for r in np.ndindex(*(12,) * (n - 1)) if allowed(r, rl, tl)]
            Z = sum(cf(*r, c, mu) for r in grid)
            sets_ok = sorted(feat) == sorted(grid)
            err = float(np.max(np.abs(p / (ref / Z) - 1)))
            worst = max(worst, err)
            print(f'  {str(kw):45s} discrete={disc!s:5s} t->trash edges={len(bad)}  '
                  f'outcome set ok={sets_ok}  max rel err={err:.2e}')
            if not disc:
                em = j.epoch_model([list(r) for r in rows], [0.0])
                pe = np.asarray(em.model(jnp.array([c, mu]))[0])
                pe = pe / pe.sum()
                e2 = float(np.max(np.abs(pe / (ref / Z) - 1)))
                print(f'  {"":45s} epoch_model [0.0] vs closed form (renormalised here) rel err={e2:.2e}')

print(f'\nworst single-epoch rel err: {worst:.2e}')
