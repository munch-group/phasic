"""
Test suite for ...

"""

from phasic import Graph, with_ipv
import pytest
import numpy as np

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


def callback(state):
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
            new[i+j+1] += 1
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions

@with_ipv([([4, 0, 0, 0], 1)])
def callback_with_ipv(state):
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
            new[i+j+1] += 1
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions

@with_ipv([4, 0, 0, 0])
def callback_with_abbr_ipv(state):
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
            new[i+j+1] += 1
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions


@np.vectorize
def bin_coef(n):
    return n*(n-1)/2

class TestIPV:
    """Test IPV argument"""

    def test_full_ipv(self):
        """Test ipv kwarg as (state, rate) tuples."""
        nr_samples = 4
        ipv = [([nr_samples]+[0]*(nr_samples-1), 1)]
        graph = Graph(callback, ipv=ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == pytest.approx(sum(1 / bin_coef(n)))

    def test_abbr_ipv(self):
        """Test ipv kwarg as single state."""
        nr_samples = 4
        ipv = [nr_samples]+[0]*(nr_samples-1)
        graph = Graph(callback, ipv=ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == pytest.approx(sum(1 / bin_coef(n)))

    def test_full_ipv_decorated_callback(self):
        """Test ipv decorator as (state, rate) tuples."""
        nr_samples = 4
        graph = Graph(callback_with_ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == pytest.approx(sum(1 / bin_coef(n)))

    def test_abbr_ipv_decorated_callback(self):
        """Test ipv decorator as single state."""
        nr_samples = 4
        graph = Graph(callback_with_abbr_ipv)
        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == pytest.approx(sum(1 / bin_coef(n)))

    def test_must_sum_to_one(self):
        """Test raises when IPV does not sum to one."""
        ipv = [
            ([4, 0, 0, 0], 0.5),
            ([2, 1, 0, 0], 0.7),                
            ]
        with pytest.raises(ValueError):
            graph = Graph(callback, ipv=ipv)

    def test_raise_if_abbr_not_single_state(self):
        """Test that abbr IPV cannot have more than one state"""
        ipv = [
            [4, 0, 0, 0],
            [2, 1, 0, 0]
            ]
        with pytest.raises(TypeError):
            graph = Graph(callback, ipv=ipv)

    def test_raise_if_ipv_empty(self):
        """Test that abbr IPV is not empty"""
        ipv = []
        with pytest.raises(ValueError):
            graph = Graph(callback, ipv=ipv)

    def test_raise_if_ipv_not_provided(self):
        """Test that abbr IPV is exactly one state"""
        with pytest.raises(ValueError):
            graph = Graph(callback)

    def test_raise_if_ipv_provided_to_decorated(self):
        """Test that abbr IPV is exactly one state"""
        with pytest.raises(ValueError):
            graph = Graph(callback_with_ipv, ipv=[4, 0, 0, 0])

    def test_raise_if_duplicate_states_in_ipv(self):
        """Test that abbr IPV is exactly one state"""
        with pytest.raises(ValueError):
            graph = Graph(callback_with_ipv, ipv=[([4, 0, 0, 0], 0.5), ([4, 0, 0, 0], 0.5)])




class TestXXXXXXXX:
    """Test XXXX """

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_XXXXXX(self):
        """Test XXXXXX."""

        nr_samples = 4
        ipv = [([nr_samples]+[0]*(nr_samples-1), 1)]
        graph = Graph(coalescent, ipv=ipv)

        n = np.arange(2, nr_samples+1)
        assert graph.expectation() == pytest.approx(sum(1 / bin_coef(n)))

        # graph.expectation() == pytest.approx(np.array(2, 1, 2/3))

        # np.array([0.1, 0.2]) + np.array([0.2, 0.4]) == approx(np.array([0.3, 0.6]))



    # @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    # def test_configure_jax_for_environment(self):
    #     """Test JAX configuration for environment."""
    #     env = ptd.detect_environment()
    #     # Should not raise
    #     ptd.configure_jax_for_environment(env)



    # def test_invalid_state_length(self):
    #     """Test invalid state_length."""
    #     with pytest.raises((ValueError, AssertionError, TypeError)):
    #         g = Graph(-1)
