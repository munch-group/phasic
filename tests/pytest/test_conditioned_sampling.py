"""
Tests for backward probabilities and conditioned path sampling.
"""

from phasic import Graph, with_ipv, StateIndexer, Property
import numpy as np
import pytest


def build_simple_chain():
    """start -> v1 -> v2 -> absorb (v3). One absorbing state."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 2.0)
    v2.add_edge(v3, 3.0)
    return g


def build_branching_graph():
    """start -> v1, v1 -> v2 (absorb) or v1 -> v3 (absorb).
    Two absorbing states with equal probability."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)
    v1.add_edge(v3, 1.0)
    return g


def build_joint_prob_coalescent():
    """Build a joint_prob_graph from a small coalescent."""
    nr_samples = 2
    indexer = StateIndexer(
        lineages=[Property('ton', min_value=1, max_value=nr_samples)]
    )
    ipv = [0] * indexer.state_length
    ipv[indexer.props_to_index(ton=1)] = nr_samples

    @with_ipv(ipv)
    def coal(state, indexer=None):
        from itertools import combinations_with_replacement
        transitions = []
        for i, j in combinations_with_replacement(range(state.size), 2):
            same = int(i == j)
            if same and state[i] < 2: continue
            if not same and (state[i] < 1 or state[j] < 1): continue
            new = state.copy()
            new[i] -= 1; new[j] -= 1
            new[min(i + j + 1, state.size - 1)] += 1
            transitions.append([new, [state[i] * (state[j] - same) / (1 + same)]])
        return transitions

    g = Graph(coal, indexer=indexer)
    jg = g.joint_prob_graph(indexer, tot_reward_limit=3, mutation_rate=1.0)
    jg.update_weights([1.0, 1.0])
    return jg, indexer


class TestBackwardProbabilities:

    def test_single_absorbing(self):
        """Simple chain: all vertices should have h=1 (only one absorbing state)."""
        g = build_simple_chain()
        # v3 (index 3) is the only absorbing state
        bp = g.backward_probabilities([3])
        # All non-absorbing vertices lead to v3
        assert bp[3] == 1.0
        assert bp[2] > 0  # v2 -> v3
        assert bp[1] > 0  # v1 -> v2 -> v3
        assert bp[0] > 0  # start -> v1 -> ... -> v3

    def test_branching_target_one(self):
        """Branching: targeting v2 should give h=0.5 at v1."""
        g = build_branching_graph()
        bp = g.backward_probabilities([2])
        assert bp[2] == 1.0  # Target
        assert bp[3] == 0.0  # Other absorbing state
        assert abs(bp[1] - 0.5) < 1e-10  # v1 -> v2 with prob 0.5

    def test_branching_target_both(self):
        """Targeting both absorbing states: h=1 everywhere."""
        g = build_branching_graph()
        bp = g.backward_probabilities([2, 3])
        assert bp[2] == 1.0
        assert bp[3] == 1.0
        assert abs(bp[1] - 1.0) < 1e-10

    def test_range_zero_to_one(self):
        """All backward probs should be in [0, 1]."""
        g = build_branching_graph()
        bp = g.backward_probabilities([2])
        assert np.all(bp >= 0)
        assert np.all(bp <= 1.0 + 1e-10)

    def test_joint_prob_graph(self):
        """Backward probs on joint_prob_graph: targets have h=1."""
        jg, indexer = build_joint_prob_coalescent()
        # Find terminal vertices (no outgoing edges)
        terminals = []
        for i, v in enumerate(jg.vertices()):
            if v.edges_length() == 0:
                terminals.append(i)

        assert len(terminals) > 0

        # Target one terminal
        bp = jg.backward_probabilities([terminals[0]])
        assert bp[terminals[0]] == 1.0
        # Other terminals should have h=0
        for t in terminals[1:]:
            assert bp[t] == 0.0


class TestConditionedPathSampling:

    def test_reaches_target(self):
        """Every conditioned path must end at a target vertex."""
        g = build_branching_graph()
        target = [2]

        for _ in range(50):
            path = g.sample_path_conditioned(target)
            last_idx = int(path['vertex_indices'][-1])
            assert last_idx in target, (
                f"Path ended at {last_idx}, expected one of {target}"
            )

    def test_never_reaches_non_target(self):
        """Conditioned on v2, should never reach v3."""
        g = build_branching_graph()
        for _ in range(50):
            path = g.sample_path_conditioned([2])
            assert int(path['vertex_indices'][-1]) == 2

    def test_frequency_matches_backward_prob(self):
        """Fraction of unconditioned paths reaching target ≈ h(start)."""
        g = build_branching_graph()
        bp = g.backward_probabilities([2])
        h_start = bp[0]  # Should be ~0.5

        n = 1000
        count = 0
        for _ in range(n):
            path = g.sample_path()
            if int(path['vertex_indices'][-1]) == 2:
                count += 1

        empirical = count / n
        assert abs(empirical - h_start) < 0.1, (
            f"Empirical {empirical:.3f} vs backward prob {h_start:.3f}"
        )

    def test_multiple_paths(self):
        """sample_path_conditioned(n=5) returns list of 5 dicts."""
        g = build_branching_graph()
        paths = g.sample_path_conditioned([2], n=5)
        assert isinstance(paths, list)
        assert len(paths) == 5
        for p in paths:
            assert int(p['vertex_indices'][-1]) == 2

    def test_joint_prob_graph_conditioned(self):
        """Conditioned sampling on joint_prob_graph reaches correct terminal."""
        jg, indexer = build_joint_prob_coalescent()

        # Use joint_prob_table to find terminal vertices (t-vertices)
        # These may have edges to trash but are the logical terminals
        jpt = jg.joint_prob_table()
        terminals = list(jpt.index)

        # Pick a target with nonzero backward probability from start
        target = None
        for t in terminals:
            bp = jg.backward_probabilities([int(t)])
            if bp[0] > 0:
                target = [int(t)]
                break

        assert target is not None, "No reachable terminal found"

        for _ in range(20):
            path = jg.sample_path_conditioned(target)
            last_idx = int(path['vertex_indices'][-1])
            assert last_idx in target, (
                f"Conditioned path ended at {last_idx}, expected {target}"
            )

    def test_entry_times_valid(self):
        """Conditioned paths should have valid entry times."""
        g = build_branching_graph()
        path = g.sample_path_conditioned([2])
        times = path['entry_times']
        assert times[0] == 0.0
        assert np.all(np.diff(times) >= 0)
