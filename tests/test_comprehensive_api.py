#!/usr/bin/env python
"""
Comprehensive test suite for the ptdalgorithms Python API.
Tests all core functionality of the Graph, Vertex, and Edge classes.
"""

import numpy as np
import ptdalgorithms as ptd
from ptdalgorithms import Graph, Vertex, Edge, MatrixRepresentation


def approx(a, b, rel=1e-5):
    """Check if two values are approximately equal."""
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


class TestGraphConstruction:
    """Test graph construction methods."""

    def test_construct_with_state_length(self):
        """Test basic graph construction with state_length."""
        g = Graph(state_length=1)
        assert g is not None
        assert g.vertices_length() == 1  # Only starting vertex
        assert g.state_length() == 1

    def test_construct_with_multidimensional_state(self):
        """Test graph with multidimensional state vectors."""
        g = Graph(state_length=3)
        assert g.state_length() == 3
        v = g.find_or_create_vertex([1, 2, 3])
        assert list(v.state()) == [1, 2, 3]

    def test_construct_with_callback(self):
        """Test graph construction with callback function."""
        def callback(state):
            if state[0] < 5:
                return [([state[0] + 1], 1.0)]
            return []

        g = Graph(callback=callback)
        assert g.vertices_length() > 1  # Should generate vertices

    def test_construct_parameterized(self):
        """Test parameterized graph construction."""
        def callback(state):
            if state[0] < 3:
                return [([state[0] + 1], 0.0, [1.0, 0.0])]
            return []

        g = Graph(callback=callback, parameterized=True)
        assert g.vertices_length() > 1

        # Check serialization detects parameterized edges
        serialized = g.serialize()
        assert serialized.get('param_length', 0) > 0

    def test_cannot_use_both_state_length_and_callback(self):
        """Test that using both state_length and callback raises error."""
        def callback(state):
            return []

        with pytest.raises(AssertionError):
            Graph(state_length=1, callback=callback)

    def test_starting_vertex(self):
        """Test that starting vertex is always present."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        assert start is not None


class TestVertexOperations:
    """Test vertex creation and manipulation."""

    def test_find_or_create_vertex(self):
        """Test vertex creation."""
        g = Graph(state_length=2)
        v1 = g.find_or_create_vertex([1, 2])
        assert v1 is not None
        assert list(v1.state()) == [1, 2]

        # Finding existing vertex should return same vertex
        v2 = g.find_or_create_vertex([1, 2])
        assert v1 == v2

    def test_create_vertex(self):
        """Test create_vertex (always creates new)."""
        g = Graph(state_length=1)
        v1 = g.create_vertex([5])
        v2 = g.create_vertex([5])
        assert v1.index() != v2.index()  # Different indices

    def test_find_vertex(self):
        """Test finding existing vertices."""
        g = Graph(state_length=1)
        v = g.find_or_create_vertex([10])

        found = g.find_vertex([10])
        assert found is not None
        assert found == v

        not_found = g.find_vertex([99])
        assert not_found is None

    def test_vertex_at(self):
        """Test accessing vertices by index."""
        g = Graph(state_length=1)
        v = g.vertex_at(0)  # Starting vertex
        assert v is not None

    def test_vertex_exists(self):
        """Test checking vertex existence."""
        g = Graph(state_length=1)
        g.find_or_create_vertex([5])

        assert g.vertex_exists([5])
        assert not g.vertex_exists([99])

    def test_vertex_state(self):
        """Test accessing vertex state."""
        g = Graph(state_length=3)
        v = g.find_or_create_vertex([1, 2, 3])
        state = v.state()
        assert list(state) == [1, 2, 3]

    def test_vertex_index(self):
        """Test vertex indexing."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        assert v1.index() != v2.index()
        assert v1.index() >= 1  # 0 is starting vertex

    def test_vertex_rate(self):
        """Test vertex rate computation."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge(v2, 2.5)
        rate = v1.rate()
        assert rate == pytest.approx(2.5)


class TestEdgeOperations:
    """Test edge creation and manipulation."""

    def test_add_edge(self):
        """Test basic edge addition."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge(v2, 1.5)

        edges = v1.edges()
        assert len(edges) > 0
        assert edges[0].weight() == pytest.approx(1.5)
        assert edges[0].to() == v2

    def test_add_edge_alias(self):
        """Test ae() alias for add_edge."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.ae(v2, 2.0)

        edges = v1.edges()
        assert len(edges) > 0
        assert edges[0].weight() == pytest.approx(2.0)

    def test_add_edge_parameterized(self):
        """Test parameterized edge addition."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge_parameterized(v2, 0.0, [1.0, 0.0])

        # Check serialization detects it
        serialized = g.serialize()
        assert serialized.get('param_length', 0) > 0

    def test_edge_weight(self):
        """Test edge weight access."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge(v2, 3.14)
        edge = v1.edges()[0]
        assert edge.weight() == pytest.approx(3.14)

    def test_edge_update_weight(self):
        """Test updating edge weight."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge(v2, 1.0)
        edge = v1.edges()[0]
        edge.update_weight(2.0)

        assert edge.weight() == pytest.approx(2.0)

    def test_edge_to_vertex(self):
        """Test edge target vertex access."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge(v2, 1.0)
        edge = v1.edges()[0]

        assert edge.to() == v2


class TestMatrixOperations:
    """Test matrix conversion operations."""

    def test_as_matrices_basic(self):
        """Test converting graph to matrices."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        start.add_edge(v1, 0.7)
        start.add_edge(v2, 0.3)
        v1.add_edge(v2, 1.0)

        g.normalize()

        matrices = g.as_matrices()
        assert isinstance(matrices, MatrixRepresentation)
        assert matrices.ipv is not None
        assert matrices.sim is not None
        assert matrices.states is not None

        # Check dimensions
        n = len(matrices.ipv)
        assert matrices.sim.shape == (n, n)
        assert len(matrices.states) == n

    def test_from_matrices_basic(self):
        """Test creating graph from matrices."""
        ipv = np.array([0.6, 0.4])
        sim = np.array([[-2.0, 1.0], [0.0, -3.0]])

        g = Graph.from_matrices(ipv, sim)
        assert g is not None
        assert g.vertices_length() > 0

    def test_from_matrices_with_states(self):
        """Test from_matrices with custom states."""
        ipv = np.array([1.0, 0.0])
        sim = np.array([[-1.0, 0.5], [0.0, -2.0]])
        states = np.array([[10], [20]], dtype=np.int32)

        g = Graph.from_matrices(ipv, sim, states)
        assert g is not None

    def test_round_trip_matrices(self):
        """Test as_matrices -> from_matrices round trip."""
        # Create original graph
        g_orig = Graph(state_length=1)
        start = g_orig.starting_vertex()
        v1 = g_orig.find_or_create_vertex([10])
        v2 = g_orig.find_or_create_vertex([20])

        start.add_edge(v1, 0.7)
        start.add_edge(v2, 0.3)
        v1.add_edge(v2, 1.5)
        g_orig.normalize()

        # Convert to matrices and back
        matrices = g_orig.as_matrices()
        g_recon = Graph.from_matrices(matrices.ipv, matrices.sim, matrices.states)

        # Compare PDFs
        times = [0.5, 1.0, 2.0]
        for t in times:
            pdf_orig = g_orig.pdf(t, 100)
            pdf_recon = g_recon.pdf(t, 100)
            assert pdf_orig == pytest.approx(pdf_recon, rel=1e-5)


class TestDistributionComputations:
    """Test distribution computation methods."""

    def test_pdf_continuous(self):
        """Test PDF computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        pdf = g.pdf(1.0, 100)
        assert pdf >= 0
        assert pdf < float('inf')

    def test_cdf_continuous(self):
        """Test CDF computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        cdf = g.cdf(1.0, 100)
        assert 0 <= cdf <= 1

    def test_pmf_discrete(self):
        """Test discrete PMF computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        pmf = g.pmf_discrete(5)
        assert pmf >= 0
        assert pmf <= 1

    def test_cdf_discrete(self):
        """Test discrete CDF computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        cdf = g.cdf_discrete(5)
        assert 0 <= cdf <= 1

    def test_stop_probability(self):
        """Test stop probability computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        prob = g.stop_probability(1.0, 100)
        assert 0 <= prob <= 1

    def test_stop_probability_discrete(self):
        """Test discrete stop probability."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        prob = g.stop_probability_discrete(5)
        assert 0 <= prob <= 1


class TestMoments:
    """Test moment computation methods."""

    def test_expectation(self):
        """Test expectation computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 2.0)
        g.normalize()

        exp = g.expectation()
        assert exp > 0
        # For exponential with rate 2, expectation should be 0.5
        assert exp == pytest.approx(0.5, rel=0.1)

    def test_variance(self):
        """Test variance computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 2.0)
        g.normalize()

        var = g.variance()
        assert var >= 0

    def test_moments(self):
        """Test general moment computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        # Test different moments
        m1 = g.moments(1)  # First moment (expectation)
        m2 = g.moments(2)  # Second moment

        assert m1 > 0
        assert m2 > 0
        assert m2 >= m1  # Second moment >= first moment

    def test_covariance(self):
        """Test covariance computation."""
        g = Graph(state_length=2)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1, 1])
        start.add_edge(v, 1.0)
        g.normalize()

        # Test with rewards
        rewards = np.array([[1, 0], [0, 1]], dtype=float)
        cov = g.covariance(rewards)
        assert cov.shape == (2, 2)

    def test_expected_waiting_time(self):
        """Test expected waiting time."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 2.0)
        g.normalize()

        wait_time = g.expected_waiting_time()
        assert wait_time > 0

    def test_expectation_discrete(self):
        """Test discrete expectation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        exp = g.expectation_discrete()
        assert exp > 0

    def test_variance_discrete(self):
        """Test discrete variance."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        var = g.variance_discrete()
        assert var >= 0


class TestSampling:
    """Test sampling methods."""

    def test_sample_continuous(self):
        """Test continuous sampling."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        sample = g.sample()
        assert sample >= 0

    def test_sample_discrete(self):
        """Test discrete sampling."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        sample = g.sample_discrete()
        assert sample >= 0
        assert isinstance(sample, (int, np.integer))

    def test_sample_multiple(self):
        """Test multiple samples."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        samples = [g.sample() for _ in range(10)]
        assert len(samples) == 10
        assert all(s >= 0 for s in samples)

    def test_random_sample_stop_vertex(self):
        """Test random sampling of stop vertex."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        start.add_edge(v1, 0.5)
        start.add_edge(v2, 0.5)
        g.normalize()

        time, vertex = g.random_sample_stop_vertex()
        assert time >= 0
        assert vertex is not None

    def test_random_sample_discrete_stop_vertex(self):
        """Test discrete random sampling of stop vertex."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        jumps, vertex = g.random_sample_discrete_stop_vertex()
        assert jumps >= 0
        assert isinstance(jumps, (int, np.integer))
        assert vertex is not None


class TestDiscretization:
    """Test discretization methods."""

    def test_discretize_basic(self):
        """Test basic discretization."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        g_discrete, rewards = g.discretize(reward_rate=0.1)

        assert g_discrete is not None
        assert g_discrete.vertices_length() >= g.vertices_length()
        assert rewards.shape[1] == g_discrete.vertices_length()

    def test_discretize_with_skip_states(self):
        """Test discretization with skip_states."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        start.add_edge(v1, 0.5)
        start.add_edge(v2, 0.5)
        g.normalize()

        g_discrete, rewards = g.discretize(reward_rate=0.1, skip_states=[1])
        assert g_discrete is not None

    def test_discretize_with_skip_slots(self):
        """Test discretization with skip_slots."""
        g = Graph(state_length=2)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1, 2])
        start.add_edge(v, 1.0)
        g.normalize()

        g_discrete, rewards = g.discretize(reward_rate=0.1, skip_slots=[0])
        assert g_discrete is not None


class TestGraphOperations:
    """Test graph manipulation operations."""

    def test_normalize(self):
        """Test graph normalization."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 2.0)

        scaling = g.normalize()
        assert scaling > 0

    def test_normalize_discrete(self):
        """Test discrete graph normalization."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)

        scaling = g.normalize_discrete()
        assert scaling > 0

    def test_copy(self):
        """Test graph copying."""
        g1 = Graph(state_length=1)
        start = g1.starting_vertex()
        v = g1.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g1.normalize()

        g2 = g1.copy()

        # Should be different objects
        assert g2 is not g1

        # But same structure
        assert g2.vertices_length() == g1.vertices_length()
        assert g2.state_length() == g1.state_length()

    def test_clone(self):
        """Test graph cloning."""
        g1 = Graph(state_length=1)
        start = g1.starting_vertex()
        v = g1.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g1.normalize()

        g2 = Graph(g1.clone())

        assert g2.vertices_length() == g1.vertices_length()

    def test_validate(self):
        """Test graph validation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)

        # Should not raise
        g.validate()

    def test_is_acyclic(self):
        """Test acyclic check."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        start.add_edge(v1, 1.0)
        v1.add_edge(v2, 1.0)

        # This should be acyclic
        assert g.is_acyclic()


class TestGraphQueries:
    """Test graph query methods."""

    def test_vertices_length(self):
        """Test vertices_length query."""
        g = Graph(state_length=1)
        initial_length = g.vertices_length()

        g.find_or_create_vertex([1])
        assert g.vertices_length() == initial_length + 1

    def test_state_length(self):
        """Test state_length query."""
        g = Graph(state_length=3)
        assert g.state_length() == 3

    def test_vertices(self):
        """Test vertices() method."""
        g = Graph(state_length=1)
        g.find_or_create_vertex([1])
        g.find_or_create_vertex([2])

        vertices = g.vertices()
        assert len(vertices) >= 3  # start + 2 created

    def test_states(self):
        """Test states() method."""
        g = Graph(state_length=1)
        g.find_or_create_vertex([1])
        g.find_or_create_vertex([2])

        states = g.states()
        assert len(states) >= 3


class TestSerialization:
    """Test serialization methods."""

    def test_serialize_basic(self):
        """Test basic serialization."""
        g = Graph(state_length=2)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1, 0])
        v2 = g.find_or_create_vertex([0, 1])
        start.add_edge(v1, 0.5)
        start.add_edge(v2, 0.5)
        v1.add_edge(v2, 1.0)
        g.normalize()

        serialized = g.serialize()

        # Check required fields
        assert 'states' in serialized
        assert 'edges' in serialized
        assert 'start_edges' in serialized
        assert 'state_dim' in serialized
        assert 'n_vertices' in serialized

        # Check dimensions
        assert serialized['state_dim'] == 2
        assert len(serialized['states']) > 0

    def test_serialize_parameterized(self):
        """Test serialization with parameterized edges."""
        g = Graph(state_length=1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge_parameterized(v2, 0.0, [1.0, 0.0])

        serialized = g.serialize()

        assert 'param_edges' in serialized
        assert 'param_length' in serialized
        assert serialized['param_length'] > 0


class TestRewardTransforms:
    """Test reward transformation methods."""

    def test_reward_transform_continuous(self):
        """Test continuous reward transformation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        rewards = np.array([[1.0], [0.0]], dtype=float)
        result = g.reward_transform(1.0, 100, rewards)

        assert result is not None
        assert len(result) > 0

    def test_reward_transform_discrete(self):
        """Test discrete reward transformation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        rewards = np.array([[1.0], [0.0]], dtype=float)
        result = g.reward_transform_discrete(5, rewards)

        assert result is not None
        assert len(result) > 0


class TestExpectedVisits:
    """Test expected visits methods."""

    def test_expected_visits_discrete(self):
        """Test expected discrete visits."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        visits = g.expected_visits_discrete(5)
        assert visits is not None
        assert len(visits) > 0

    def test_accumulated_visits_discrete(self):
        """Test accumulated discrete visits."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        visits = g.accumulated_visits_discrete(5)
        assert visits is not None
        assert len(visits) > 0


class TestResidenceTime:
    """Test residence time methods."""

    def test_expected_residence_time(self):
        """Test expected residence time."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        residence = g.expected_residence_time(1.0, 100)
        assert residence is not None
        assert len(residence) > 0

    def test_accumulated_visiting_time(self):
        """Test accumulated visiting time."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        time = g.accumulated_visiting_time(1.0, 100)
        assert time is not None
        assert len(time) > 0


class TestDistributionContext:
    """Test distribution context methods."""

    def test_distribution_context_continuous(self):
        """Test continuous distribution context."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        context = g.distribution_context(1.0, 100)
        assert context is not None

    def test_distribution_context_discrete(self):
        """Test discrete distribution context."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 1.0)
        g.normalize()

        context = g.distribution_context_discrete(5)
        assert context is not None


class TestDefect:
    """Test defect computation."""

    def test_defect(self):
        """Test defect computation."""
        g = Graph(state_length=1)
        start = g.starting_vertex()
        v = g.find_or_create_vertex([1])
        start.add_edge(v, 0.9)  # Not normalized

        defect = g.defect()
        assert defect >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
