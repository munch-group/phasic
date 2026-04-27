"""
Test suite for ...

"""

from phasic import Graph, with_ipv, MatrixRepresentation
from pytest import approx, raises
import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm

from math import factorial

# Try to import optional dependencies
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


from phasic.test_utils import (
    bin_coef,
    coalescent_manual_construction,
    coalescent_callback,
    coalescent_callback_with_ipv,
    coalescent_callback_with_abbr_ipv,
    coalescent_callback_parameterized,
)

class TestManualConstruction:
    """Test manual construction."""

    def test_starting_vertex(self):
        """Test that starting vertex is always present."""
        g = Graph(1)
        start = g.starting_vertex()
        assert start is not None


    def test_find_or_create_vertex(self):
        """Test vertex creation."""
        g = Graph(2)
        v1 = g.find_or_create_vertex([1, 2])
        assert v1 is not None
        assert list(v1.state()) == [1, 2]

        # Finding existing vertex should return same vertex
        v2 = g.find_or_create_vertex([1, 2])
        assert v1 == v2


    def test_create_vertex(self):
        """Test create_vertex (always creates new)."""
        g = Graph(1)
        v1 = g.create_vertex([5])
        v2 = g.create_vertex([5])
        assert v1.index() != v2.index()  # Different indices


    def test_find_vertex(self):
        """Test finding existing vertices."""
        g = Graph(1)
        v = g.find_or_create_vertex([10])

        found = g.find_vertex([10])
        assert found is not None
        assert found == v


    def test_find_vertex_raises(self):
        """Test finding non-existing vertices raises."""
        g = Graph(1)
        with raises(RuntimeError):
            g.find_vertex([99])


    def test_vertex_at(self):
        """Test accessing vertices by index."""
        g = Graph(1)
        v = g.vertex_at(0)  # Starting vertex
        assert v is not None


    def test_vertex_at_raises(self):
        """Test accessing vertices by index."""
        g = Graph(1)
        with raises(ValueError):
            v = g.vertex_at(99)  # Starting vertex


    def test_vertex_exists(self):
        """Test checking vertex existence."""
        g = Graph(1)
        g.find_or_create_vertex([5])
        assert g.vertex_exists([5])
        assert not g.vertex_exists([99])


    def test_vertex_rate(self):
        """Test vertex rate computation."""
        g = Graph(1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v1.add_edge(v2, 2.5)
        rate = v1.rate()
        assert rate == approx(2.5)


    def test_add_edge(self):
        """Test basic edge addition."""
        g = Graph(1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v1.add_edge(v2, 1.5)
        edges = v1.edges()
        assert len(edges) > 0
        assert edges[0].weight() == approx(1.5)
        assert edges[0].to() == v2


    def test_add_edge_alias(self):
        """Test ae() alias for add_edge."""
        g = Graph(1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        v1.ae(v2, 2.0)
        edges = v1.edges()
        assert len(edges) > 0
        assert edges[0].weight() == approx(2.0)


    def test_edge_weight(self):
        """Test edge weight access."""
        g = Graph(1)
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        v1.add_edge(v2, 3.14)
        edge = v1.edges()[0]
        assert edge.weight() == approx(3.14)


    def test_manual_construction_not_raising(self):
        """manual construction not raising."""
        graph = coalescent_manual_construction(4)


    def test_manual_construction_states(self):
        """manual construction correct states."""
        graph = coalescent_manual_construction(4)
        assert graph.vertices_length() == 6
        states = np.array([
            [0, 0, 0, 0],
            [4, 0, 0, 0],
            [2, 1, 0, 0],
            [0, 2, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=int)        
        assert np.array(graph.states(), dtype=int) == approx(states)



class TestConstructionFromMatrices:
    """Test matrix construction."""

    def test_from_matrices_ipv_sim_lists(self):
        """Test basic from_matrices functionality."""
        # Create simple 2-state phase-type
        ipv = [1, 0, 0, 0]
        sim = [
            [-6,  6,  0,  0],
            [ 0, -3,  1,  2],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1]
            ]

        # Create graph from matrices
        g = Graph.from_matrices(ipv, sim)

        assert g is not None
        assert g.vertices_length() > 0

        # Test PDF computation
        pdf = g.pdf(1.0)
        assert 0 <= pdf <= 1
        print(f"Basic from_matrices test passed (PDF at t=1.0: {pdf:.6f})")


    def test_from_matrices_basic(self):
        """Test basic from_matrices functionality."""
        # Create simple 2-state phase-type
        ipv = np.array([0.6, 0.4])
        sim = np.array([
            [-2.0, 1.0],
            [0.0, -3.0]
        ])

        # Create graph from matrices
        g = Graph.from_matrices(ipv, sim)

        assert g is not None
        assert g.vertices_length() > 0

        # Test PDF computation
        pdf = g.pdf(1.0)
        assert 0 <= pdf <= 1
        print(f"Basic from_matrices test passed (PDF at t=1.0: {pdf:.6f})")


    def test_from_matrices_with_states(self):
        """Test from_matrices with custom state vectors."""
        # ipv length must match sim dimension
        ipv = np.array([1.0, 0.0])
        sim = np.array([
            [-1.0, 0.0],
            [0.0, -2.0]
        ])

        # Test with custom states that don't conflict
        states = np.array([[10], [20]], dtype=np.int32)

        g = Graph.from_matrices(ipv, sim, states)
        assert g is not None
        assert g.vertices_length() > 0

        print("from_matrices with custom states test passed")


    def test_from_matrices_multidimensional(self):
        """Test from_matrices with multidimensional states."""
        ipv = np.array([1.0, 0, 0, 0])
        sim = np.array([
            [-6,  6,  0,  0],
            [ 0, -3,  1,  2],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1]
            ]
        )

        # 2D states
        states = np.array([
            [4, 0, 0, 0],
            [2, 1, 0, 0],
            [0, 2, 0, 0],
            [1, 0, 1, 0],
        ], dtype=np.int32)

        g = Graph.from_matrices(ipv, sim, states)
        assert g is not None
        assert g.state_length() == 4  # Should have 2D states

        print("from_matrices with multidimensional states test passed")


    def test_round_trip_simple(self):
        """Test round-trip: create graph -> as_matrices -> from_matrices."""
        # Create original graph
        graph = Graph(coalescent_callback_with_ipv)

        matrices = graph.as_matrices()

        assert isinstance(matrices, MatrixRepresentation)
        ipv = matrices.ipv
        assert sum(ipv) == approx(1.0)
        sim = matrices.sim
        states = matrices.states

        # Reconstruct from matrices
        graph_mat = Graph.from_matrices(ipv, sim, states)

        graph_mat.expectation() == approx(graph.expectation())

        graph_mat_matrices = graph_mat.as_matrices()
        graph_mat_matrices.ipv == approx(ipv)
        graph_mat_matrices.sim == approx(sim)
        graph_mat_matrices.states == approx(states)


    def test_from_matrices_validation(self):
        """Test input validation for from_matrices."""
        # Test dimension mismatch
        ipv = np.array([0.5, 0.5])
        sim_wrong = np.array([[-1.0]])  # Wrong size

        try:
            g = Graph.from_matrices(ipv, sim_wrong)
            assert False, "Should have raised error for dimension mismatch"
        except RuntimeError as e:
            assert "square" in str(e) or "dimension" in str(e)

        # Test non-square SIM
        sim_nonsquare = np.array([[-1.0, 0.0]])
        try:
            g = Graph.from_matrices(ipv, sim_nonsquare)
            assert False, "Should have raised error for non-square matrix"
        except RuntimeError as e:
            assert "square" in str(e)

        print("Input validation test passed")


    def test_from_matrices_edge_cases(self):
        """Test edge cases for from_matrices."""
        # Single state
        ipv_single = np.array([1.0])
        sim_single = np.array([[-5.0]])

        g = Graph.from_matrices(ipv_single, sim_single)
        assert g is not None
        pdf = g.pdf(0.5)
        # Just check it's a valid PDF value, exact calculation depends on implementation
        assert 0 <= pdf <= 5.0  # Max PDF for exponential with rate 5

        # Zero initial probability for some states
        ipv_sparse = np.array([0.0, 1.0, 0.0])
        sim_sparse = np.array([
            [-1.0, 0.5, 0.0],
            [0.0, -2.0, 1.0],
            [0.0, 0.0, -0.5]
        ])

        g = Graph.from_matrices(ipv_sparse, sim_sparse)
        assert g is not None

        print("Edge cases test passed")


    def test_from_matrices_performance(self):
        """Test from_matrices with larger matrices."""
        n = 10

        # Create a chain of states
        sim = np.zeros((n, n))
        for i in range(n):
            sim[i, i] = -(i + 1)  # Increasing exit rates
            if i < n - 1:
                sim[i, i + 1] = i + 1  # Transition to next state

        ipv = np.zeros(n)
        ipv[0] = 1.0  # Start in first state

        g = Graph.from_matrices(ipv, sim)
        assert g is not None
        assert g.vertices_length() >= n

        # Test PDF computation works
        pdf = g.pdf(1.0)
        assert 0 <= pdf <= 1

        print(f"Performance test passed (n={n} states)")


class TestCallbackConstruction:
    """Test construction using callback."""

    def test_callback_not_raising(self):
        """Test constructor not raising."""
        graph = Graph(coalescent_callback_with_ipv)


    def test_full_ipv(self):
        """Test ipv kwarg as (state, rate) tuples."""
        nr_samples = 4
        ipv = [([nr_samples]+[0]*(nr_samples-1), 1)]
        graph = Graph(coalescent_callback, ipv=ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == approx(sum(1 / bin_coef(n)))


    def test_abbr_ipv(self):
        """Test ipv kwarg as single state."""
        nr_samples = 4
        ipv = [nr_samples]+[0]*(nr_samples-1)
        graph = Graph(coalescent_callback, ipv=ipv)
        assert graph is not None
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == approx(sum(1 / bin_coef(n)))


    def test_full_ipv_decorated_callback(self):
        """Test ipv decorator as (state, rate) tuples."""
        nr_samples = 4
        graph = Graph(coalescent_callback_with_ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == approx(sum(1 / bin_coef(n)))


    def test_abbr_ipv_decorated_callback(self):
        """Test ipv decorator as single state."""
        nr_samples = 4
        graph = Graph(coalescent_callback_with_abbr_ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == approx(sum(1 / bin_coef(n)))


    def test_must_sum_to_one(self):
        """Test raises when IPV does not sum to one."""
        ipv = [
            ([4, 0, 0, 0], 0.5),
            ([2, 1, 0, 0], 0.7),                
            ]
        with raises(ValueError):
            graph = Graph(coalescent_callback, ipv=ipv)


    def test_raise_if_abbr_not_single_state(self):
        """Test that abbr IPV cannot have more than one state"""
        ipv = [
            [4, 0, 0, 0],
            [2, 1, 0, 0]
            ]
        with raises(TypeError):
            graph = Graph(coalescent_callback, ipv=ipv)


    def test_raise_if_ipv_empty(self):
        """Test that abbr IPV is not empty"""
        ipv = []
        with raises(ValueError):
            graph = Graph(coalescent_callback, ipv=ipv)


    def test_raise_if_ipv_not_provided(self):
        """Test that abbr IPV is exactly one state"""
        with raises(ValueError):
            graph = Graph(coalescent_callback)


    def test_raise_if_ipv_provided_to_decorated(self):
        """Test that abbr IPV is exactly one state"""
        with raises(ValueError):
            graph = Graph(coalescent_callback_with_ipv, ipv=[4, 0, 0, 0])


    def test_raise_if_duplicate_states_in_ipv(self):
        """Test that abbr IPV is exactly one state"""
        with raises(ValueError):
            graph = Graph(coalescent_callback_with_ipv, ipv=[([4, 0, 0, 0], 0.5), ([4, 0, 0, 0], 0.5)])


    def test_callback_construction_states(self):
        """manual construction correct states."""
        manual_graph = coalescent_manual_construction(4)
        callback_graph = Graph(coalescent_callback_with_ipv)
        assert manual_graph.vertices_length() == callback_graph.vertices_length()
        assert np.array(callback_graph.states()) == approx(np.array(manual_graph.states()))


    def test_callback_construction_moments(self):
        """manual construction correct states."""
        manual_graph = coalescent_manual_construction(4)
        callback_graph = Graph(coalescent_callback_with_ipv)
        assert manual_graph.expectation() == callback_graph.expectation()
        assert manual_graph.variance() == callback_graph.variance()
        assert manual_graph.moments(5) == approx(callback_graph.moments(5))


class TestCloneConstruction:
    """Test clone construction."""

    def test_clone(self):
        graph = Graph(coalescent_callback_with_ipv)
        graph.expectation()
        graph_clone =  graph.clone()
        exp = graph_clone.expectation()
        assert exp == approx(graph.expectation())

    def test_clone_is_separate(self):
        graph = Graph(coalescent_callback_with_ipv)
        graph_clone =  graph.clone()
        del graph
        graph_clone.expectation() # deleting clone should not affect orig graph
            


class TestMomentsPDF:
    """Test moments and PDF"""

    def test_moments(self):
        """Test constructor not raising."""

        ipv = np.array([1, 0, 0, 0])
        sim = np.array([
            [-6,  6,  0,  0],
            [ 0, -3,  1,  2],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1]])

        U = inv(-sim) # "Green" matrix
        ones = np.ones(sim.shape[0])

        def moment(n):
            return factorial(n) * ipv @ np.linalg.matrix_power(U, n) @ ones

        expectation = moment(1)
        variance  = moment(2) - expectation**2
        moments = np.array([moment(n) for n in range(1, 6)])

        graph = coalescent_manual_construction(4)
        assert graph.expectation() == approx(expectation)
        assert graph.variance() == approx(variance)
        assert graph.moments(5) == approx(moments)

        graph = Graph(coalescent_callback, ipv=[4, 0, 0, 0])
        assert graph.expectation() == approx(expectation)
        assert graph.variance() == approx(variance)
        assert graph.moments(5) == approx(moments)

        graph = Graph.from_matrices(ipv, sim)
        assert graph.expectation() == approx(expectation)
        assert graph.variance() == approx(variance)
        assert graph.moments(5) == approx(moments)


    def test_pdf(self):
        """Test constructor not raising."""

        ipv = np.array([1, 0, 0, 0])
        sim = np.array([
            [-6,  6,  0,  0],
            [ 0, -3,  1,  2],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1]])

        t_exit = -sim @ np.ones(sim.shape[0])

        def pdf(x):
            x = np.atleast_1d(x).astype(float)
            return np.array([ipv @ expm(sim * xi) @ t_exit for xi in x])

        xs = np.linspace(0, 3, 20)

        graph = coalescent_manual_construction(4)
        assert graph.pdf(xs) == approx(pdf(xs), rel=2e-2)

        graph = Graph(coalescent_callback, ipv=[4, 0, 0, 0])
        assert graph.pdf(xs) == approx(pdf(xs), rel=2e-2)

        graph = Graph.from_matrices(ipv, sim)
        assert graph.pdf(xs) == approx(pdf(xs), rel=2e-2)



class TestParameterized:
    """Test parameterized"""


    def test_callback_parameterized_not_raising(self):
        """Test constructor parameterized not raising."""
        graph = Graph(coalescent_callback_parameterized, ipv=[4, 0, 0, 0])



class TestRewards:
    """Test rewards"""

    def test_expectation_rewards(self):
        ...



# ## Forward computation in steps:

# - **accumulated_occupancy(...)** calls **accumulated_visits** for discrete  graphs, **accumulated_visiting_time** for continuous graphs
# - **state_probability(t)**. The probability vector p(t) where p(t)[i] = probability of being in transient state i at time t (continuous) or after n jumps (discrete). 

# ## Graph elimination

# - **expected_waiting_time(rewards)** Computes E[total accumulated reward before absorption] via graph elimination. With the default reward vector (all 1s), this gives E[T], the expected total time until absorption. The result is a vector of length n_vertices where result[0] (the starting vertex) holds the final answer. The other entries are intermediate quantities from the elimination. This is the workhorse behind moments(). Higher moments are computed by iterating reward transformations.
# - **expected_sojourn_time()**. Computes E[time in state i before absorption] for every state, starting from the initial distribution. Implemented by running expected_waiting_time with n different unit reward vectors (one-hot for each state) in a single batched pass. So expected_sojourn_time()[i] = the expected total time the chain spends in state i over its entire lifetime. You can think of it as accumulated_occupancy for infinite t.



class TestExpectedWaitingTime:
    """Test expected waiting time"""

    def test_expected_waiting_time(self):
        ...


class TestStateProbability:
    """Test state probability"""

    def test_state_probability_XXXX(self):
        ...


class TestExpectedWaitingTime:
    """Test expected waiting time"""

    def test_expected_waiting_time_XXXX(self):
        ...


class TestExpectedSojournTime:
    """Test expected sojourn time"""

    def test_expected_sojourn_time_XXXX(self):
        ...
