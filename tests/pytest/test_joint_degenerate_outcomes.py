"""Degenerate joint-probability outcomes are merged at construction (T4).

When the base graph has several absorbing states (here: an indexed
two-island coalescent whose common ancestor can end in either island),
outcome vertices with the same reward vector used to be separate
t-vertices: 54 vertices for 27 outcomes. An observed reward vector
could not be assigned to one of them, and the observation mapping drew
one at random, so the likelihood used one vertex's probability instead
of the outcome's total. joint_prob_graph now substitutes one canonical
absorbing base state when it creates a vertex, so equal reward vectors
meet in one t-vertex (no extra vertex or edge). See
joint-prob-followups-plan.md, ticket T4.

PINS are the outcome probabilities of the UNMERGED discrete graph,
summed over each outcome's vertices, computed with the code before the
fix (discrete visit probabilities were correct there; only the grouping
was wrong). theta = (coalescence, migration, mutation) = (1.0, 0.5, 0.7),
reward_limit = 2.
"""
import numpy as np
import pytest
import jax.numpy as jnp

from phasic import Graph, Property, StateIndexer, with_ipv
from test_notebook_model_recovery import coalescent_indexed

THETA = [1.0, 0.5, 0.7]
PINS = {
    (0, 0, 0, 0): 0.23989898989898994,
    (0, 0, 0, 1): 0.0026276907553334614,
    (0, 0, 0, 2): 0.0009141886887270833,
    (0, 0, 1, 0): 0.0807343386294232,
    (0, 0, 1, 1): 0.0018811876998410006,
    (0, 0, 1, 2): 0.0006806274643450807,
    (0, 0, 2, 0): 0.027199787312260217,
    (0, 0, 2, 1): 0.0009954643516844955,
    (0, 0, 2, 2): 0.0003741480818035436,
    (1, 0, 0, 0): 0.08336202938475666,
    (1, 0, 0, 1): 0.0037095650772951674,
    (1, 0, 0, 2): 0.001621090367687475,
    (1, 0, 1, 0): 0.05628076232436143,
    (1, 0, 1, 1): 0.003352183632059152,
    (1, 0, 1, 2): 0.0014717138682419156,
    (1, 0, 2, 0): 0.028503161373978726,
    (1, 0, 2, 1): 0.0021378306618222718,
    (1, 0, 2, 2): 0.0009524417760489485,
    (2, 0, 0, 0): 0.0299951637008283,
    (2, 0, 0, 1): 0.003297182183717051,
    (2, 0, 0, 2): 0.0017376910306174768,
    (2, 0, 1, 0): 0.030179253190008297,
    (2, 0, 1, 1): 0.003609544530064187,
    (2, 0, 1, 2): 0.0018698923960329106,
    (2, 0, 2, 0): 0.020315796582901054,
    (2, 0, 2, 1): 0.0026962083053905603,
    (2, 0, 2, 2): 0.0013950092479516904,
}


def _two_island(n=2):
    idx = StateIndexer(lineage=[Property('descendants', min_value=1, max_value=n),
                                Property('pop', min_value=0, max_value=1)])
    s0 = np.zeros(idx.state_length, int)
    s0[idx.lineage.props_to_index(descendants=1, pop=0)] = n

    @with_ipv(list(s0))
    def cb(state, indexer=None):
        L = indexer.lineage
        out = []
        if sum(state) == 1:
            return []                      # common ancestor: absorbing (either island)
        for d1 in range(1, n + 1):
            for p in (0, 1):
                i = L.props_to_index(descendants=d1, pop=p)
                c = state[i]
                if c == 0:
                    continue
                new = state.copy(); new[i] -= 1
                new[L.props_to_index(descendants=d1, pop=1 - p)] += 1
                out.append([new, [0.0, 1.0]])                    # migration
                for d2 in range(d1, n + 1):
                    j = L.props_to_index(descendants=d2, pop=p)
                    pairs = c * (c - 1) / 2 if j == i else c * state[j]
                    if pairs == 0 or d1 + d2 > n:
                        continue
                    new = state.copy(); new[i] -= 1; new[j] -= 1
                    new[L.props_to_index(descendants=d1 + d2, pop=p)] += 1
                    out.append([new, [pairs, 0.0]])              # coalescence
        return out
    return Graph(cb, indexer=idx), idx


def _t_vertices(j):
    return sorted({v.index() for v in j.vertices()
                   if v.edges() and any(len(e.to().edges()) == 0 for e in v.edges())})


@pytest.fixture(scope="module")
def island():
    g, idx = _two_island()
    absorbing = [v for v in g.vertices() if not v.edges() and v.index() != g.starting_vertex().index()]
    assert len(absorbing) == 2           # the fixture really has two absorbing states
    return g, idx


@pytest.mark.parametrize("discrete", [True, False])
def test_one_vertex_per_outcome_and_pinned_probabilities(island, discrete):
    g, idx = island
    j = g.joint_prob_graph(idx, discrete=discrete, reward_limit=2)
    j.update_weights(THETA)
    t = j.joint_prob_table()
    cols = list(t.columns[:-1])
    keys = [tuple(int(x) for x in r) for r in t[cols].to_numpy()]
    assert len(keys) == len(set(keys)) == len(PINS)
    got = dict(zip(keys, t['prob'].to_numpy()))
    for k, v in PINS.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-12, err_msg=str(k))


def test_likelihoods_match_pinned_outcome_probabilities(island):
    g, idx = island
    j = g.joint_prob_graph(idx, discrete=False, reward_limit=2)
    term = _t_vertices(j)
    n_r = j.state_length() - g.state_length()
    keys = [tuple(int(x) for x in j.vertex_at(i).state()[-n_r:]) for i in term]
    ref = np.array([PINS[k] for k in keys])
    ref = ref / sum(PINS.values())
    m = Graph.pmf_from_graph_joint_index(j, theta_dim=3)
    p = np.asarray(m(jnp.array(THETA), jnp.array(term, dtype=jnp.int32))[0])
    np.testing.assert_allclose(p, ref, rtol=1e-12)
    pe = np.asarray(j.epoch_model([list(k) for k in keys], [0.0, 0.4]).model(
        jnp.array(THETA + THETA))[0])
    np.testing.assert_allclose(pe, ref, rtol=1e-9)


def test_observations_map_deterministically(island):
    g, idx = island
    j = g.joint_prob_graph(idx, discrete=False, reward_limit=2)
    obs = [list(k) for k in PINS][:10] * 3
    a = j._map_joint_observations_to_indices(obs)
    b = j._map_joint_observations_to_indices(obs, seed=7)
    assert a == b


def test_single_absorbing_state_graph_unchanged():
    # Coalescent: one absorbing state, so the canonical substitution is a
    # no-op; vertex count and outcome count are those of the plain build.
    g, idx = coalescent_indexed(3)
    j = g.joint_prob_graph(idx, discrete=True, reward_limit=2)
    assert len(_t_vertices(j)) == 9


def _two_absorbing_children():
    # One transient state (two singleton lineages in pop 0) with an edge to
    # EACH of two absorbing states (the ancestor in pop 0 at rate t0, in
    # pop 1 at rate t1). After the canonical substitution both edges point
    # at the same outcome vertex: parallel edges, which the C layer sums.
    idx = StateIndexer(lineage=[Property('descendants', min_value=1, max_value=2),
                                Property('pop', min_value=0, max_value=1)])
    L = idx.lineage
    s0 = np.zeros(idx.state_length, int)
    s0[L.props_to_index(descendants=1, pop=0)] = 2

    @with_ipv(list(s0))
    def cb(state, indexer=None):
        L = indexer.lineage
        if sum(state) == 1:
            return []
        a = np.zeros_like(state); a[L.props_to_index(descendants=2, pop=0)] = 1
        b = np.zeros_like(state); b[L.props_to_index(descendants=2, pop=1)] = 1
        return [[a, [1.0, 0.0]], [b, [0.0, 1.0]]]
    return Graph(cb, indexer=idx), idx


@pytest.mark.parametrize("discrete", [True, False])
def test_parent_with_two_absorbing_children_matches_closed_form(discrete):
    # Mutations on the two singletons at total rate r = 2 mu while the
    # transient state is left at rate lam = t0 + t1, so the number of
    # descendants_1 mutations is geometric: P(k) = (r/(r+lam))^k lam/(r+lam).
    t0, t1, mu = 1.0, 0.5, 0.7
    g, idx = _two_absorbing_children()
    j = g.joint_prob_graph(idx, discrete=discrete, reward_limit=2,
                           reward_only=['descendants'])
    j.update_weights([t0, t1, mu])
    parallel = [v for v in j.vertices()
                if len({e.to().index() for e in v.edges()}) < len(v.edges())]
    assert parallel, "fixture no longer produces parallel edges"
    t = j.joint_prob_table()
    k = t.iloc[:, 0].to_numpy().astype(int)
    r, lam = 2 * mu, t0 + t1
    ref = [(r / (r + lam)) ** x * lam / (r + lam) for x in k]
    assert len(k) == len(set(k)) == 3
    np.testing.assert_allclose(t['prob'].to_numpy(), ref, rtol=1e-12)
