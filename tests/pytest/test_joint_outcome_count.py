"""Joint-probability inference needs at least two outcomes (ticket T8).

Every joint-prob likelihood is normalised over the outcome set, so with
one outcome each observation has probability 1 for every theta and
nothing can be inferred; with none, the table construction used to fail
with an IndexError deep inside numpy indexing.
"""
import numpy as np
import pytest
import jax.numpy as jnp

from phasic import Graph, Property, StateIndexer, with_ipv
from test_notebook_model_recovery import coalescent_indexed


@pytest.fixture(scope="module")
def one_outcome():
    g, idx = coalescent_indexed(2)
    j = g.joint_prob_graph(idx, discrete=False, tot_reward_limit=0)
    return g, j


def _t_vertices(j):
    return sorted({v.index() for v in j.vertices()
                   if v.edges() and any(len(e.to().edges()) == 0 for e in v.edges())})


def test_one_outcome_table_still_readable(one_outcome):
    _, j = one_outcome
    assert len(j.joint_prob_table()) == 1


def test_one_outcome_joint_index_model_refused(one_outcome):
    _, j = one_outcome
    with pytest.raises(ValueError, match="at least 2 are needed"):
        Graph.pmf_from_graph_joint_index(j, theta_dim=2)


@pytest.mark.parametrize("call", ["svgd", "epoch_model", "probability_matching"])
def test_one_outcome_inference_refused(one_outcome, call):
    _, j = one_outcome
    obs = [[0, 0]] * 10
    with pytest.raises(ValueError, match="at least 2 are needed for inference"):
        if call == "svgd":
            j.svgd(obs, n_particles=4, n_iterations=2, fixed=[(1, 0.5)], progress=False)
        elif call == "epoch_model":
            j.epoch_model(obs, [0.0])
        else:
            j.probability_matching(obs, fixed=[(1, 0.5)])


def test_two_outcomes_accepted():
    g, idx = coalescent_indexed(2)
    j = g.joint_prob_graph(idx, discrete=False, tot_reward_limit=1)
    m = Graph.pmf_from_graph_joint_index(j, theta_dim=2)
    p = np.asarray(m(jnp.array([1.0, 0.5]),
                     jnp.array(_t_vertices(j), dtype=jnp.int32))[0])
    np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-12)


def test_zero_outcomes_clear_error():
    # A base chain that cycles forever between two states never absorbs,
    # so its joint graph has no outcome vertices.
    idx = StateIndexer(lineage=[Property('descendants', min_value=1, max_value=2)])
    s0 = [2, 0]

    @with_ipv(s0)
    def cb(state, indexer=None):
        new = state.copy()
        new[0], new[1] = state[1], state[0]
        return [[new, [1.0]]]

    g = Graph(cb, indexer=idx)
    j = g.joint_prob_graph(idx, discrete=True, reward_limit=1)
    with pytest.raises(ValueError, match="no outcome vertices"):
        j.joint_prob_table()


def test_generic_single_terminal_graph_still_accepted():
    # pmf_from_graph_joint_index is a generic API: on a NON-joint graph a
    # single terminal vertex is fine, because a non-terminal vertex read
    # against it (sojourn ratio) depends on theta. Chain s -> v3 -> v2 -> v1
    # with v2 the only terminal: sojourn(v3)/sojourn(v2) = w2/w3 =
    # (t0 + 2 t1) / (2 t0 + 3 t1). (Regression: the T8 guard first applied
    # to every graph and broke 17 tests built on such chains.)
    g = Graph(1)
    v3 = g.find_or_create_vertex([3])
    v2 = g.find_or_create_vertex([2])
    v1 = g.find_or_create_vertex([1])
    g.starting_vertex().add_edge(v3, 1.0)
    v3.add_edge(v2, [2.0, 3.0])
    v2.add_edge(v1, [1.0, 2.0])
    m = Graph.pmf_from_graph_joint_index(g, theta_dim=2)
    for t0, t1 in [(1.0, 1.0), (0.5, 2.0)]:
        p = float(np.asarray(m(jnp.array([t0, t1]),
                               jnp.array([v3.index()], dtype=jnp.int32))[0])[0])
        np.testing.assert_allclose(p, (t0 + 2 * t1) / (2 * t0 + 3 * t1), rtol=1e-10)
