"""
Tests for Graph.sample_path() — full path sampling.
"""

from phasic import Graph, with_ipv
import numpy as np
import pytest


def build_simple_chain():
    """Build a simple 3-state chain: start -> v1 -> v2 -> absorb."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    v3 = g.find_or_create_vertex([3])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 2.0)
    v2.add_edge(v3, 3.0)
    return g


def build_exponential_graph():
    """Simple exponential: start -> v1 -> absorb with rate 1."""
    g = Graph(1)
    start = g.starting_vertex()
    v1 = g.find_or_create_vertex([1])
    v2 = g.find_or_create_vertex([2])
    start.add_edge(v1, 1.0)
    v1.add_edge(v2, 1.0)
    return g


def build_block_coalescent(n_samples):
    """Block coalescent tracking only lineage count."""

    @with_ipv([n_samples] + [0] * (n_samples - 1))
    def block_coalescent(state):
        transitions = []
        for i in range(state.size):
            for j in range(i, state.size):
                same = int(i == j)
                if same and state[i] < 2:
                    continue
                if not same and (state[i] < 1 or state[j] < 1):
                    continue
                new = state.copy()
                new[i] -= 1
                new[j] -= 1
                new[i + j + 1] += 1
                transitions.append([new, state[i] * (state[j] - same) / (1 + same)])
        return transitions

    return Graph(block_coalescent)


class TestSamplePathBasic:

    def test_returns_dict(self):
        """Single sample returns a dict."""
        g = build_simple_chain()
        path = g.sample_path()
        assert isinstance(path, dict)
        assert 'vertex_indices' in path
        assert 'entry_times' in path

    def test_returns_list_for_n_gt_1(self):
        """Multiple samples returns a list of dicts."""
        g = build_simple_chain()
        paths = g.sample_path(n=3)
        assert isinstance(paths, list)
        assert len(paths) == 3
        for p in paths:
            assert isinstance(p, dict)

    def test_starts_at_starting_vertex(self):
        """First vertex in path is the starting vertex."""
        g = build_simple_chain()
        path = g.sample_path()
        assert path['vertex_indices'][0] == g.starting_vertex().index()

    def test_entry_time_starts_at_zero(self):
        """First entry time is 0."""
        g = build_simple_chain()
        path = g.sample_path()
        assert path['entry_times'][0] == 0.0

    def test_entry_times_non_decreasing(self):
        """Entry times are monotonically non-decreasing."""
        g = build_simple_chain()
        for _ in range(10):
            path = g.sample_path()
            times = path['entry_times']
            assert np.all(np.diff(times) >= 0), f"Entry times not non-decreasing: {times}"

    def test_ends_at_absorbing_vertex(self):
        """Last vertex in path has no outgoing edges (absorbing)."""
        g = build_simple_chain()
        path = g.sample_path()
        last_idx = int(path['vertex_indices'][-1])
        # Find the vertex and check it's absorbing
        vertices = g.vertices()
        last_vertex = vertices[last_idx]
        assert last_vertex.edges_length() == 0

    def test_simple_chain_length(self):
        """Simple chain: path length = 4 (start, v1, v2, absorb)."""
        g = build_simple_chain()
        path = g.sample_path()
        assert len(path['vertex_indices']) == 4
        assert len(path['entry_times']) == 4


class TestSamplePathCoalescent:

    def test_block_coalescent_path_length(self):
        """Block coalescent with n samples: path has n+1 entries
        (start + n-1 coalescence events + absorbing state)."""
        n = 5
        g = build_block_coalescent(n)
        path = g.sample_path()
        # start -> (n-1 intermediate states) -> absorb = n+1
        assert len(path['vertex_indices']) == n + 1

    def test_sojourn_times_positive(self):
        """Sojourn times (diffs of entry times) are positive after starting vertex."""
        g = build_block_coalescent(5)
        for _ in range(10):
            path = g.sample_path()
            sojourn = np.diff(path['entry_times'])
            # First sojourn (from start) should be 0 (start has no waiting)
            # Remaining should be positive
            assert np.all(sojourn[1:] > 0), f"Sojourn times not positive: {sojourn}"


class TestSamplePathConsistency:

    def test_multiple_paths_vary(self):
        """Multiple sampled paths should not be identical."""
        g = build_block_coalescent(5)
        paths = g.sample_path(n=5)
        total_times = [p['entry_times'][-1] for p in paths]
        assert len(set(float(t) for t in total_times)) > 1, (
            "All paths have identical total times — randomness not working"
        )

    def test_vertex_indices_valid(self):
        """All vertex indices in path should be valid graph vertices."""
        g = build_block_coalescent(5)
        n_vertices = g.vertices_length()
        for _ in range(10):
            path = g.sample_path()
            for idx in path['vertex_indices']:
                assert 0 <= int(idx) < n_vertices, (
                    f"Vertex index {idx} out of range [0, {n_vertices})"
                )
