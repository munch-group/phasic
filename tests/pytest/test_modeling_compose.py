"""
Composability tests: base graph + add_epoch + joint_prob_graph.

These tests pin down that the indexer attached to a Graph (via the
``indexer=`` kwarg) is propagated through ``add_epoch`` as an epoch-aware
indexer, and that ``joint_prob_graph`` accepts either the original or the
augmented indexer.
"""

from itertools import combinations_with_replacement

import numpy as np
import pytest

from phasic import Graph, Property, StateIndexer, with_ipv


N_SAMPLES = 5
EPOCH_BOUNDARY = 0.5
TRUE_THETA1 = 2.0
TRUE_THETA2 = 0.5
MUTATION_RATE = 0.01
REWARD_LIMIT = 5


def _make_indexer():
    return StateIndexer(
        lineages=[Property('descendants', min_value=1, max_value=N_SAMPLES)],
    )


def _make_callback(indexer):
    ipv = [0] * indexer.state_length
    ipv[indexer.lineages.props_to_index(descendants=1)] = N_SAMPLES

    @with_ipv(ipv)
    def coal_callback(state, indexer=None):
        transitions = []
        for i, j in combinations_with_replacement(range(indexer.lineages.state_length), 2):
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            pair_count = state[i] * (state[j] - same) / (1 + same)
            transitions.append([new, [pair_count]])
        return transitions

    return coal_callback


class TestIndexerPropagation:
    """Graph stores its indexer; add_epoch produces an epoch-aware indexer."""

    def test_graph_stores_indexer(self):
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g = Graph(cb, indexer=indexer)
        assert g._indexer is indexer

    def test_graph_without_indexer_has_none(self):
        # Building from a callback without an indexer kwarg leaves _indexer None.
        # We construct a trivial empty graph by integer state_length to verify
        # the default value path; callback construction without the indexer
        # kwarg would also work but requires a callback that doesn't use one.
        g = Graph(2)
        assert g._indexer is None

    def test_add_epoch_propagates_augmented_indexer(self):
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g_cont = Graph(cb, indexer=indexer)

        g_epochs = g_cont.add_epoch(time=EPOCH_BOUNDARY)

        assert g_epochs._indexer is not None
        assert g_epochs._indexer.state_length == g_epochs.state_length()
        # base indexer preserved for chained composition
        assert g_epochs._base_indexer is indexer
        # exactly one PropertySet (joint_prob_graph contract)
        assert len(g_epochs._indexer.property_sets()) == 1
        # epoch slot was added
        slot_names = [s.name for s in g_epochs._indexer.slots()]
        assert 'epoch' in slot_names


class TestJointProbAfterEpoch:
    """joint_prob_graph composes with add_epoch."""

    def test_joint_prob_graph_accepts_original_indexer(self):
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g_cont = Graph(cb, indexer=indexer)

        g_epochs = g_cont.add_epoch(time=EPOCH_BOUNDARY)
        g_epochs.update_weights([TRUE_THETA1, 0.0, TRUE_THETA2, 1.0])

        # Passing the *original* (non-augmented) indexer must work — the graph
        # transparently substitutes its own augmented indexer.
        jg = g_epochs.joint_prob_graph(
            indexer,
            mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT,
        )

        # Joint graph state vector strictly extends the augmented base state.
        assert jg.state_length() > g_epochs.state_length()
        # Joint graph param_length = augmented param_length + 1 (mutation slot)
        assert jg.param_length() == g_epochs.param_length() + 1
        # Joint graph is non-empty and larger than the base
        assert jg.vertices_length() > g_epochs.vertices_length()
        # Joint graph carries an indexer matching its own state vector length
        assert jg._indexer is not None
        assert jg._indexer.state_length == jg.state_length()

    def test_joint_prob_graph_accepts_augmented_indexer(self):
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g_cont = Graph(cb, indexer=indexer)

        g_epochs = g_cont.add_epoch(time=EPOCH_BOUNDARY)
        g_epochs.update_weights([TRUE_THETA1, 0.0, TRUE_THETA2, 1.0])

        # Passing the augmented indexer directly must also work.
        jg = g_epochs.joint_prob_graph(
            g_epochs._indexer,
            mutation_rate=MUTATION_RATE,
            reward_limit=REWARD_LIMIT,
        )

        assert jg.vertices_length() > g_epochs.vertices_length()
        assert jg.param_length() == g_epochs.param_length() + 1

    def test_joint_prob_graph_results_match_for_either_indexer(self):
        """Same graph structure regardless of which indexer the user passes."""
        indexer = _make_indexer()
        cb = _make_callback(indexer)

        g1 = Graph(cb, indexer=indexer).add_epoch(time=EPOCH_BOUNDARY)
        g1.update_weights([TRUE_THETA1, 0.0, TRUE_THETA2, 1.0])
        jg_orig = g1.joint_prob_graph(
            indexer, mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT,
        )

        g2 = Graph(cb, indexer=indexer).add_epoch(time=EPOCH_BOUNDARY)
        g2.update_weights([TRUE_THETA1, 0.0, TRUE_THETA2, 1.0])
        jg_aug = g2.joint_prob_graph(
            g2._indexer, mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT,
        )

        assert jg_orig.state_length() == jg_aug.state_length()
        assert jg_orig.param_length() == jg_aug.param_length()
        assert jg_orig.vertices_length() == jg_aug.vertices_length()

    def test_joint_prob_graph_rejects_unrelated_indexer(self):
        """A length-mismatched indexer with no graph fallback raises clearly."""
        indexer = _make_indexer()
        cb = _make_callback(indexer)
        g_epochs = Graph(cb, indexer=indexer).add_epoch(time=EPOCH_BOUNDARY)
        g_epochs.update_weights([TRUE_THETA1, 0.0, TRUE_THETA2, 1.0])

        # Wipe the graph's own indexer to force the failure path.
        g_epochs._indexer = None
        with pytest.raises(ValueError, match="state_length"):
            g_epochs.joint_prob_graph(
                indexer, mutation_rate=MUTATION_RATE, reward_limit=REWARD_LIMIT,
            )
