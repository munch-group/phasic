"""Granularity-free final-epoch read for the daisy chain.

The final epoch of an epoch-wise (daisy-chain) joint-prob computation runs to
absorption (t -> inf), which is the one place the elimination-based
``expected_sojourn_time`` read applies. Replacing the final-epoch
``stop_probability(t_eval)`` forward solve with

    joint_prob[v] = r_v * expected_sojourn_time(v) * handoff_mass

(``r_v`` = the t-vertex's exit rate to absorption; ``handoff_mass`` = total mass
of the final-epoch handoff IPV, which ``expected_sojourn_time`` normalises away)
is EXACT and granularity-free. These tests pin that equivalence against the
existing exact ``daisy_chain_joint_probs`` (continuous ``stop_probability`` path),
and the structural invariants of ``Graph.joint_sojourn_graph``.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import Graph, Property, StateIndexer, with_ipv

N_SAMPLES = 4
MUTATION_RATE = 1.0
REWARD_LIMIT = 3


def _make_joint(discrete: bool = False) -> Graph:
    indexer = StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N_SAMPLES)]
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N_SAMPLES

    @with_ipv(ipv)
    def coal(state, indexer=None):
        tr = []
        for i, j in combinations_with_replacement(
            range(indexer.lineages.state_length), 2
        ):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            tr.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return tr

    return Graph(coal, indexer=indexer).joint_prob_graph(
        indexer, mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT,
        discrete=discrete,
    )


def _config_of(graph, idx):
    bl = graph._joint_prob_base_graph_indexer.state_length
    st = graph.states()
    return tuple(int(st[idx, k]) for k in range(bl, graph.state_length()))


def _daisy_probs(jsp, init_ipv, epoch_thetas, epoch_dts):
    import jax.numpy as jnp
    out = np.asarray(jsp.daisy_chain_joint_probs(
        epoch_thetas=jnp.asarray(epoch_thetas, dtype=jnp.float64),
        epoch_dts=list(epoch_dts), initial_ipv=init_ipv, t_eval=200.0,
    ))
    d: dict = {}
    for t_idx, p in zip(jsp._t_vertex_indices, out):
        c = _config_of(jsp, t_idx)
        d[c] = d.get(c, 0.0) + float(p)
    return d


def _sojourn_daisy_probs(cjpg, jsp, sg, init_ipv, epoch_thetas, epoch_dts):
    """Modified daisy: intermediate handoffs via stop_probability (epoch_context),
    final-epoch read via the granularity-free joint_sojourn_graph."""
    jst = jsp.states()
    sg_pos = {s: i for i, s in enumerate(sg._ipv_target_states)}

    # Chain the intermediate handoffs on the JSP graph.
    ctx = cjpg.epoch_context()
    ctx.update_ipv(init_ipv)
    h = init_ipv
    for k in range(len(epoch_dts)):
        ctx.update_ipv(h)
        ctx.update_weights(epoch_thetas[k])
        h = ctx.next_ipv(epoch_dts[k])
    handoff_mass = float(h.sum())

    # Map the handoff (JSP IPV layout) onto the sojourn graph's IPV layout by state.
    sg_ipv = np.zeros(len(sg._ipv_target_states))
    for i, jidx in enumerate(jsp._ipv_target_indices):
        s = tuple(int(x) for x in jst[jidx])
        if s in sg_pos:
            sg_ipv[sg_pos[s]] += h[i]

    sg.update_ipv(sg_ipv)
    sg.update_weights(epoch_thetas[-1])
    sojourn = np.asarray(sg.expected_sojourn_time())

    d: dict = {}
    for tv in sg._t_vertex_indices:
        v = sg.vertex_at(tv)
        r_v = sum(e.weight() for e in v.edges() if len(e.to().edges()) == 0)
        c = _config_of(sg, tv)
        d[c] = d.get(c, 0.0) + r_v * sojourn[tv] * handoff_mass
    return d


def test_joint_sojourn_graph_structure():
    cjpg = _make_joint()
    jsp = cjpg.joint_stop_prob_graph()
    sg = cjpg.joint_sojourn_graph()

    assert getattr(sg, '_joint_sojourn_graph', False) is True
    assert sg.is_discrete is False
    # Same IPV layout size as the JSP graph (both exclude trash/abs/start).
    assert len(sg._ipv_target_indices) == len(jsp._ipv_target_indices)
    assert len(sg._ipv_target_states) == len(sg._ipv_target_indices)
    # t-vertices each retain a single edge to an absorbing vertex (no trap).
    for tv in sg._t_vertex_indices:
        v = sg.vertex_at(tv)
        assert any(len(e.to().edges()) == 0 for e in v.edges())
    # No aux/trapping metadata.
    assert not hasattr(sg, '_t_aux_map')


def test_joint_sojourn_graph_rejects_non_joint_prob_graph():
    g = Graph(1)
    v = g.find_or_create_vertex([1])
    sink = g.find_or_create_vertex([0])
    v.add_edge(sink, [1.0])
    g.starting_vertex().add_edge(v, 1.0)
    with pytest.raises(ValueError, match="joint_prob_graph"):
        g.joint_sojourn_graph()


@pytest.mark.parametrize("epoch_thetas,epoch_dts", [
    ([[0.5, MUTATION_RATE]], []),                                   # 1 epoch
    ([[0.5, MUTATION_RATE], [0.2, MUTATION_RATE]], [0.5]),          # 2 epochs
    ([[1 / 0.3, MUTATION_RATE], [0.25, MUTATION_RATE]], [0.7]),     # 2 epochs, other theta
    ([[0.5, MUTATION_RATE], [0.2, MUTATION_RATE], [1.0, MUTATION_RATE]], [0.4, 0.6]),   # 3
    ([[0.5, MUTATION_RATE], [0.2, MUTATION_RATE], [1.0, MUTATION_RATE], [1 / 3, MUTATION_RATE]], [0.3, 0.4, 0.5]),  # 4
])
def test_sojourn_final_read_matches_daisy(epoch_thetas, epoch_dts):
    """The granularity-free final-epoch read reproduces the exact daisy chain."""
    cjpg = _make_joint()
    jsp = cjpg.joint_stop_prob_graph()
    sg = cjpg.joint_sojourn_graph()

    init = np.zeros(cjpg.vertices_length())
    for e in cjpg.starting_vertex().edges():
        init[e.to().index()] = e.weight()
    init_ipv = init[jsp._ipv_target_indices]

    ref = _daisy_probs(jsp, init_ipv, epoch_thetas, epoch_dts)
    got = _sojourn_daisy_probs(cjpg, jsp, sg, init_ipv, epoch_thetas, epoch_dts)

    keys = sorted(set(ref) | set(got))
    a = np.array([ref.get(k, 0.0) for k in keys])
    b = np.array([got.get(k, 0.0) for k in keys])
    np.testing.assert_allclose(b, a, atol=1e-9, rtol=1e-7)


@pytest.mark.parametrize("epoch_thetas,epoch_dts", [
    ([[0.5, MUTATION_RATE]], []),
    ([[0.5, MUTATION_RATE], [0.2, MUTATION_RATE]], [0.5]),
    ([[1 / 0.3, MUTATION_RATE], [0.25, MUTATION_RATE]], [0.7]),
    ([[0.5, MUTATION_RATE], [0.2, MUTATION_RATE], [1.0, MUTATION_RATE]], [0.4, 0.6]),
])
def test_library_final_read_sojourn_matches_stopprob(epoch_thetas, epoch_dts):
    """daisy_chain_joint_probs(final_read='sojourn') (batched C++) reproduces the
    default stop_probability path, element-wise (same t-vertex order)."""
    import jax.numpy as jnp
    cjpg = _make_joint()
    jsp = cjpg.joint_stop_prob_graph()
    init = np.zeros(cjpg.vertices_length())
    for e in cjpg.starting_vertex().edges():
        init[e.to().index()] = e.weight()
    init_ipv = init[jsp._ipv_target_indices]
    kw = dict(
        epoch_thetas=jnp.asarray(epoch_thetas, dtype=jnp.float64),
        epoch_dts=list(epoch_dts), initial_ipv=init_ipv, t_eval=200.0,
    )
    ref = np.asarray(jsp.daisy_chain_joint_probs(**kw, final_read='stopprob'))
    got = np.asarray(jsp.daisy_chain_joint_probs(**kw, final_read='sojourn'))
    np.testing.assert_allclose(got, ref, atol=1e-9, rtol=1e-7)


def test_library_final_read_sojourn_vmap_batched():
    """Batched (vmap over particles) sojourn final-read matches per-particle."""
    import jax
    import jax.numpy as jnp
    cjpg = _make_joint()
    jsp = cjpg.joint_stop_prob_graph()
    init = np.zeros(cjpg.vertices_length())
    for e in cjpg.starting_vertex().edges():
        init[e.to().index()] = e.weight()
    init_ipv = init[jsp._ipv_target_indices]

    def model(theta_flat):
        return jsp.daisy_chain_joint_probs(
            epoch_thetas=theta_flat.reshape(2, 2), epoch_dts=[0.5],
            initial_ipv=init_ipv, t_eval=200.0, final_read='sojourn',
        )

    thetas = jnp.asarray(
        [[0.5, MUTATION_RATE, 0.2, MUTATION_RATE],
         [1.0, MUTATION_RATE, 0.4, MUTATION_RATE]], dtype=jnp.float64,
    )
    batched = np.asarray(jax.vmap(model)(thetas))
    for i in range(thetas.shape[0]):
        single = np.asarray(model(thetas[i]))
        np.testing.assert_allclose(batched[i], single, rtol=1e-9, atol=1e-12)
