"""joint_prob_graph() with a total reward cap (tot_reward_limit).

Regression for the t-vertex trash edge: a t-vertex (base state absorbing)
sitting at the total cap used to get an edge to the trash vertex. On a
continuous joint graph that divided its sojourn read by the extra exit
rate, so the single-epoch likelihood (pmf_from_graph_joint_index) was
wrong for every outcome at the cap (43-49% relative error on the fixtures
below). With reward_only the t-vertices feed a merged outcome vertex, so
the trash edge also lost mass on DISCRETE graphs (n=2, tot cap 2: 0.097
vs 0.177 for the capped row). Without reward_only, discrete graphs were
numerically unaffected (only the structure check fails on old code); the
epoch path was unaffected throughout.

References are independent closed forms for the standard coalescent with
infinite-sites mutations (mutation rate mu per lineage, coalescence rate c
per pair):
  n=2: P(k) = (1-q) q^k,  q = 2 mu / (c + 2 mu)
  n=3: P(k1, k2) = sum_{m<=k1} (1-a) a^m (1-q) q^s C(s, k2) 2^-s,
       a = mu/(c+mu), q = 2mu/(c+2mu), s = k1 - m + k2
and the single-epoch path normalises over the allowed outcome set.
"""
import numpy as np
import pytest
import jax.numpy as jnp
from scipy.special import comb

from phasic import Graph, Property, StateIndexer, with_ipv
from test_notebook_model_recovery import coalescent_indexed


def _p_n2(k, c, mu):
    q = 2 * mu / (c + 2 * mu)
    return (1 - q) * q ** k


def _p_n3(k1, k2, c, mu):
    a = mu / (c + mu)
    q = 2 * mu / (c + 2 * mu)
    return sum((1 - a) * a ** m * (1 - q) * q ** (k1 - m + k2)
               * comb(k1 - m + k2, k2) * 0.5 ** (k1 - m + k2)
               for m in range(k1 + 1))


_CASES = {2: (_p_n2, 1.0, 0.7), 3: (_p_n3, 1.1, 0.6)}
_CAPS = [dict(reward_limit=2), dict(tot_reward_limit=2),
         dict(reward_limit=2, tot_reward_limit=2),
         dict(reward_limit=2, tot_reward_limit=3), dict(tot_reward_limit=4)]


def _t_vertices(j):
    start = j.starting_vertex().index()
    return sorted({v.index() for v in j.vertices()
                   if v.index() != start and v.edges()
                   and any(len(e.to().edges()) == 0 for e in v.edges())})


def _trash_index(j):
    for v in j.vertices():
        if v.state().sum() == 0 and v.edges_length() == 1:
            ch = v.edges()[0].to()
            if (ch.state().sum() == 0 and ch.edges_length() == 1
                    and ch.edges()[0].to().index() == v.index()):
                return v.index()
    raise AssertionError('no trash pair')


def _allowed(row, kw):
    rl = kw.get('reward_limit')
    tl = kw.get('tot_reward_limit', np.inf)
    return (rl is None or all(x <= rl for x in row)) and sum(row) <= tl


@pytest.mark.parametrize('discrete', [False, True])
@pytest.mark.parametrize('kw', _CAPS, ids=lambda d: ','.join(f'{k}={v}' for k, v in d.items()))
@pytest.mark.parametrize('n', [2, 3])
def test_single_epoch_matches_closed_form(n, kw, discrete):
    cf, c, mu = _CASES[n]
    g, idx = coalescent_indexed(n)
    j = g.joint_prob_graph(idx, discrete=discrete, **kw)
    tv = _t_vertices(j)

    # Structure: a t-vertex has a single out-edge, to the absorbing vertex.
    trash = _trash_index(j)
    for v in tv:
        out = j.vertex_at(v).edges()
        assert len(out) == 1, f't-vertex {v} has {len(out)} out-edges'
        assert len(out[0].to().edges()) == 0  # ... and it goes to absorption
    assert trash is not None

    n_reward = j.state_length() - g.state_length()
    rows = [tuple(int(x) for x in j.vertex_at(v).state()[-n_reward:]) for v in tv]
    assert all(sum(r[n - 1:]) == 0 for r in rows)  # size-n lineage never mutates
    feats = [r[:n - 1] for r in rows]

    # Outcome set equals the closed-form allowed set (caps applied correctly).
    grid = [r for r in np.ndindex(*(12,) * (n - 1)) if _allowed(r, kw)]
    assert sorted(feats) == sorted(grid)

    m = Graph.pmf_from_graph_joint_index(j, theta_dim=2)
    p = np.asarray(m(jnp.array([c, mu]), jnp.array(tv, dtype=jnp.int32))[0])
    ref = np.array([cf(*r, c, mu) for r in feats])
    Z = sum(cf(*r, c, mu) for r in grid)
    np.testing.assert_allclose(p, ref / Z, rtol=1e-12)


@pytest.mark.parametrize('kw', [dict(tot_reward_limit=2),
                                dict(reward_limit=2, tot_reward_limit=3)],
                         ids=['tot2', 'rl2_tot3'])
def test_single_epoch_agrees_with_epoch_path(kw):
    # The epoch path never read the t-vertex out-edges (joint_stop_prob_graph
    # replaces them, joint_sojourn_graph redirects trash to absorbing); the
    # two paths must agree after normalisation.
    c, mu = 1.1, 0.6
    g, idx = coalescent_indexed(3)
    j = g.joint_prob_graph(idx, discrete=False, **kw)
    tv = _t_vertices(j)
    n_reward = j.state_length() - g.state_length()
    rows = [[int(x) for x in j.vertex_at(v).state()[-n_reward:]] for v in tv]
    m = Graph.pmf_from_graph_joint_index(j, theta_dim=2)
    p = np.asarray(m(jnp.array([c, mu]), jnp.array(tv, dtype=jnp.int32))[0])
    pe = np.asarray(j.epoch_model(rows, [0.0]).model(jnp.array([c, mu]))[0])
    np.testing.assert_allclose(p, pe / pe.sum(), rtol=1e-12)


def _coalescent_n2_with_inert_tag():
    """n=2 coalescent whose lineages carry a second, never-changing
    property, so reward_only=['descendants'] is a genuine restriction
    (with a single property it is reset to None)."""
    idx = StateIndexer(lineage=[Property('descendants', min_value=1, max_value=2),
                                Property('tag', min_value=0, max_value=1)])
    s0 = np.zeros(idx.state_length, int)
    s0[idx.lineage.props_to_index(descendants=1, tag=0)] = 2

    @with_ipv(list(s0))
    def cb(state, indexer=None):
        i = indexer.lineage.props_to_index(descendants=1, tag=0)
        if state[i] == 2:
            new = state.copy()
            new[i] = 0
            new[indexer.lineage.props_to_index(descendants=2, tag=0)] = 1
            return [[new, [1.0]]]
        return []

    return Graph(cb, indexer=idx), idx


@pytest.mark.parametrize('discrete', [False, True])
@pytest.mark.parametrize('kw', [dict(tot_reward_limit=1), dict(tot_reward_limit=2),
                                dict(reward_limit=1)],
                         ids=['tot1', 'tot2', 'rl1'])
def test_reward_only_matches_closed_form(kw, discrete):
    c, mu = 1.0, 0.7
    g, idx = _coalescent_n2_with_inert_tag()
    j = g.joint_prob_graph(idx, discrete=discrete, reward_only=['descendants'], **kw)
    tv = _t_vertices(j)
    m = Graph.pmf_from_graph_joint_index(j, theta_dim=2)
    p = np.asarray(m(jnp.array([c, mu]), jnp.array(tv, dtype=jnp.int32))[0])
    n_reward = j.state_length() - g.state_length()
    k = [int(j.vertex_at(v).state()[-n_reward:].sum()) for v in tv]
    cap = kw.get('tot_reward_limit', kw.get('reward_limit'))
    assert sorted(k) == list(range(cap + 1))
    ref = np.array([_p_n2(x, c, mu) for x in k])
    np.testing.assert_allclose(p, ref / sum(_p_n2(x, c, mu) for x in range(cap + 1)),
                               rtol=1e-12)
