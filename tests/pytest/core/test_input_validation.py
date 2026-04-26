"""
Tests for input validation added to Graph constructor and key methods.

Ensures that invalid inputs are caught early with clear error messages
rather than propagating deep into the system and producing NaN or crashes.
"""

import numpy as np
import pytest
from phasic import Graph
from phasic import _callback


# --- Helper: build a simple parameterized graph ---

def _make_simple_graph():
    """Create a minimal parameterized graph for testing methods."""
    g = Graph(1)
    v_start = g.starting_vertex()
    v1 = g.find_or_create_vertex([2])
    v_abs = g.find_or_create_vertex([1])
    v_start.add_edge_parameterized(v1, 0.0, [1.0])
    v1.add_edge_parameterized(v_abs, 0.0, [1.0])
    return g


# ---- _callback(ipv) validation ----

class TestCallbackIPV:

    def test_invalid_ipv_type_string(self):
        with pytest.raises(TypeError, match="ipv must be a list"):
            _callback("hello")

    def test_invalid_ipv_type_int(self):
        with pytest.raises(TypeError, match="ipv must be a list"):
            _callback(42)

    def test_invalid_ipv_type_dict(self):
        with pytest.raises(TypeError, match="ipv must be a list"):
            _callback({"a": 1})

    def test_empty_ipv(self):
        with pytest.raises(ValueError, match="ipv must be non-empty"):
            _callback([])

    def test_invalid_ipv_format_mixed(self):
        """Mixed types that are neither all ints nor all [state, prob] pairs."""
        with pytest.raises(TypeError, match="ipv must be a list of ints"):
            _callback([5, [1, 2]])

    def test_invalid_ipv_pair_state_not_list(self):
        """State in [state, prob] pair is not a list."""
        with pytest.raises(TypeError, match="ipv\\[0\\]\\[0\\] must be a list"):
            _callback([[42, 1.0]])

    def test_invalid_ipv_pair_prob_not_number(self):
        """Probability in [state, prob] pair is not a number."""
        with pytest.raises(TypeError, match="ipv\\[0\\]\\[1\\] must be a number"):
            _callback([[[5, 0], "oops"]])

    def test_valid_ipv_simple(self):
        """Valid simple format should not raise."""
        dec = _callback([5])
        assert callable(dec)

    def test_valid_ipv_explicit(self):
        """Valid explicit format should not raise."""
        dec = _callback([[[5, 0], 0.7], [[4, 1], 0.3]])
        assert callable(dec)


# ---- Graph.__init__() validation ----

class TestGraphInit:

    def test_invalid_state_length_zero(self):
        with pytest.raises(ValueError, match="state_length must be >= 1"):
            Graph(0)

    def test_invalid_state_length_negative(self):
        with pytest.raises(ValueError, match="state_length must be >= 1"):
            Graph(-1)

    def test_invalid_arg_type_float(self):
        with pytest.raises(TypeError, match="First argument must be"):
            Graph(3.14)

    def test_invalid_arg_type_string(self):
        with pytest.raises(TypeError, match="First argument must be"):
            Graph("hello")

    def test_ipv_without_callback(self):
        with pytest.raises(ValueError, match="ipv is only used with a callback"):
            Graph(1, ipv=[5])

    def test_invalid_theta_dim_float(self):
        with pytest.raises(TypeError, match="theta_dim must be an integer"):
            def cb(state):
                return []
            Graph(cb, ipv=[1], theta_dim=2.5)

    def test_invalid_theta_dim_zero(self):
        with pytest.raises(ValueError, match="theta_dim must be >= 1"):
            def cb(state):
                return []
            Graph(cb, ipv=[1], theta_dim=0)

    def test_invalid_theta_dim_negative(self):
        with pytest.raises(ValueError, match="theta_dim must be >= 1"):
            def cb(state):
                return []
            Graph(cb, ipv=[1], theta_dim=-1)

    def test_valid_state_length(self):
        """Valid integer state_length should work."""
        g = Graph(1)
        assert g.vertices_length() >= 1


# ---- update_weights() validation ----

class TestUpdateWeights:

    def test_theta_nan(self):
        g = _make_simple_graph()
        with pytest.raises(ValueError, match="theta contains NaN"):
            g.update_weights(np.array([float('nan')]))

    def test_theta_inf(self):
        g = _make_simple_graph()
        with pytest.raises(ValueError, match="theta contains infinite"):
            g.update_weights(np.array([float('inf')]))

    def test_theta_wrong_dims(self):
        g = _make_simple_graph()
        with pytest.raises(ValueError, match="theta must be 1-dimensional"):
            g.update_weights(np.array([[1.0, 2.0]]))

    def test_theta_empty(self):
        g = _make_simple_graph()
        with pytest.raises(ValueError, match="theta must be non-empty"):
            g.update_weights(np.array([]))

    def test_theta_valid(self):
        """Valid theta should work."""
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))


# ---- pdf() validation ----

class TestPDF:

    def test_negative_time(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="time must be non-negative"):
            g.pdf(-1.0)

    def test_nan_time(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="time contains NaN"):
            g.pdf(float('nan'))

    def test_negative_granularity(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="granularity must be >= 0"):
            g.pdf(1.0, granularity=-1)

    def test_float_granularity(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(TypeError, match="granularity must be an integer"):
            g.pdf(1.0, granularity=1.5)

    def test_valid_pdf(self):
        """Valid pdf call should work."""
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        result = g.pdf(1.0)
        assert np.isfinite(result)


# ---- expectation() / variance() rewards validation ----

class TestExpectationVarianceRewards:

    def test_expectation_rewards_wrong_length(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        n = g.vertices_length()
        with pytest.raises(ValueError, match="rewards length"):
            g.expectation(rewards=np.ones(n + 5))

    def test_expectation_rewards_nan(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        n = g.vertices_length()
        rewards = np.ones(n)
        rewards[0] = float('nan')
        with pytest.raises(ValueError, match="rewards contains NaN"):
            g.expectation(rewards=rewards)

    def test_expectation_rewards_wrong_dims(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="rewards must be 1-dimensional"):
            g.expectation(rewards=np.ones((3, 2)))

    def test_variance_rewards_wrong_length(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        n = g.vertices_length()
        with pytest.raises(ValueError, match="rewards length"):
            g.variance(rewards=np.ones(n + 5))

    def test_variance_rewards_nan(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        n = g.vertices_length()
        rewards = np.ones(n)
        rewards[0] = float('nan')
        with pytest.raises(ValueError, match="rewards contains NaN"):
            g.variance(rewards=rewards)

    def test_valid_expectation(self):
        """Valid expectation call should work."""
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        result = g.expectation()
        assert np.isfinite(result)


# ---- reward_transform() validation ----

class TestRewardTransform:

    def test_rewards_wrong_length(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="rewards length"):
            g.reward_transform(np.ones(g.vertices_length() + 5))

    def test_rewards_nan(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        rewards = np.ones(g.vertices_length())
        rewards[0] = float('nan')
        with pytest.raises(ValueError, match="rewards contains NaN"):
            g.reward_transform(rewards)


# ---- discretize() validation ----

class TestDiscretize:

    def test_invalid_rate_string(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(TypeError, match="rate must be a number or callable"):
            g.discretize("fast")

    def test_rate_zero(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="rate must be in \\(0, 1\\)"):
            g.discretize(0.0)

    def test_rate_one(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="rate must be in \\(0, 1\\)"):
            g.discretize(1.0)

    def test_rate_negative(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="rate must be in \\(0, 1\\)"):
            g.discretize(-0.5)

    def test_rate_greater_than_one(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="rate must be in \\(0, 1\\)"):
            g.discretize(1.5)


# ---- svgd() validation ----

class TestSVGDValidation:

    def test_negative_n_particles(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="n_particles must be >= 1"):
            g.svgd(observed_data=np.array([1.0, 2.0]), n_particles=0, theta_dim=1)

    def test_zero_n_iterations(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="n_iterations must be >= 1"):
            g.svgd(observed_data=np.array([1.0, 2.0]), n_iterations=0, theta_dim=1)

    def test_negative_learning_rate(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            g.svgd(observed_data=np.array([1.0, 2.0]), learning_rate=-0.01, theta_dim=1)

    def test_negative_regularization(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="regularization must be >= 0"):
            g.svgd(observed_data=np.array([1.0, 2.0]), regularization=-1.0, theta_dim=1)

    def test_zero_nr_moments(self):
        g = _make_simple_graph()
        g.update_weights(np.array([2.0]))
        with pytest.raises(ValueError, match="nr_moments must be >= 1"):
            g.svgd(observed_data=np.array([1.0, 2.0]), nr_moments=0, theta_dim=1)
