"""
Tests for callback-based update_weights() functionality.
"""

import pytest
import numpy as np
from phasic import Graph


def test_callback_basic():
    """Test basic callback functionality"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    # Simple linear callback (should match standard mode)
    def linear_callback(theta, coeffs):
        return np.dot(coeffs, theta)

    g.update_weights([1.0, 2.0], callback=linear_callback)
    assert np.isclose(v1.edges()[0].weight(), 8.0), \
        f"Expected 8.0 (2.0*1.0 + 3.0*2.0), got {v1.edges()[0].weight()}"


def test_callback_multiplicative():
    """Test multiplicative callback (should match log mode)"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    # Multiplicative callback
    def mult_callback(theta, coeffs):
        weight = 1.0
        for c, t in zip(coeffs, theta):
            weight *= c * t
        return weight

    g.update_weights([1.0, 2.0], callback=mult_callback)
    assert np.isclose(v1.edges()[0].weight(), 12.0), \
        f"Expected 12.0 (2.0*1.0 * 3.0*2.0), got {v1.edges()[0].weight()}"


def test_callback_psmc_with_nan():
    """Test PSMC-style callback with NaN skipping"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [1.0, np.nan, 0.5])

    # PSMC-style: multiply non-NaN (coeff * theta) values
    def psmc_weight(theta, coeffs):
        weight = 1.0
        for c, t in zip(coeffs, theta):
            if not np.isnan(c):
                weight *= c * t
        return weight

    # Params: [epoch_rec_prob=0.1, unused=999, coal_rate=2.0]
    # Expected: (1.0*0.1) * (0.5*2.0) = 0.1
    g.update_weights([0.1, 999.0, 2.0], callback=psmc_weight)
    assert np.isclose(v1.edges()[0].weight(), 0.1), \
        f"Expected 0.1, got {v1.edges()[0].weight()}"


def test_callback_multiple_nans():
    """Test callback with multiple NaN coefficients"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, np.nan, np.nan, np.nan, 5.0])

    def psmc_weight(theta, coeffs):
        weight = 1.0
        for c, t in zip(coeffs, theta):
            if not np.isnan(c):
                weight *= c * t
        return weight

    # Expected: (2.0*1.0) * (5.0*5.0) = 50.0
    g.update_weights([1.0, 2.0, 3.0, 4.0, 5.0], callback=psmc_weight)
    assert np.isclose(v1.edges()[0].weight(), 50.0), \
        f"Expected 50.0, got {v1.edges()[0].weight()}"


def test_callback_exponential():
    """Test exponential weight function"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [1.0, 2.0])

    def exp_weight(theta, coeffs):
        return np.exp(np.dot(coeffs, theta))

    # exp(1.0*1.0 + 2.0*2.0) = exp(5.0) ≈ 148.41
    g.update_weights([1.0, 2.0], callback=exp_weight)
    expected = np.exp(5.0)
    assert np.isclose(v1.edges()[0].weight(), expected), \
        f"Expected {expected}, got {v1.edges()[0].weight()}"


def test_callback_saturating():
    """Test saturating weight function"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    def saturating_weight(theta, coeffs):
        linear = np.dot(coeffs, theta)
        return linear / (1.0 + linear)

    # linear = 8.0, saturated = 8.0/9.0 ≈ 0.889
    g.update_weights([1.0, 2.0], callback=saturating_weight)
    expected = 8.0 / 9.0
    assert np.isclose(v1.edges()[0].weight(), expected), \
        f"Expected {expected}, got {v1.edges()[0].weight()}"


def test_callback_negative_weight_error():
    """Test that callback returning negative weight raises error"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    def bad_callback(theta, coeffs):
        return -1.0

    with pytest.raises(RuntimeError, match="non-positive"):
        g.update_weights([1.0, 2.0], callback=bad_callback)


def test_callback_zero_weight_error():
    """Test that callback returning zero weight raises error"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    def bad_callback(theta, coeffs):
        return 0.0

    with pytest.raises(RuntimeError, match="non-positive"):
        g.update_weights([1.0, 2.0], callback=bad_callback)


def test_callback_multiple_edges():
    """Test callback updates all edges correctly"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v3 = g.create_vertex([3])

    v1.add_edge(v2, [2.0, 3.0])
    v1.add_edge(v3, [1.0, 4.0])

    def linear_callback(theta, coeffs):
        return np.dot(coeffs, theta)

    g.update_weights([1.0, 2.0], callback=linear_callback)

    # Edge 1: 2.0*1.0 + 3.0*2.0 = 8.0
    # Edge 2: 1.0*1.0 + 4.0*2.0 = 9.0
    assert np.isclose(v1.edges()[0].weight(), 8.0)
    assert np.isclose(v1.edges()[1].weight(), 9.0)


def test_callback_comparison_with_builtin_linear():
    """Test callback produces same result as built-in linear mode"""
    g1 = Graph(1)
    v1_1 = g1.create_vertex([1])
    v2_1 = g1.create_vertex([2])
    v1_1.add_edge(v2_1, [2.0, 3.0, 5.0])

    g2 = Graph(1)
    v1_2 = g2.create_vertex([1])
    v2_2 = g2.create_vertex([2])
    v1_2.add_edge(v2_2, [2.0, 3.0, 5.0])

    params = [1.0, 2.0, 3.0]

    # Built-in linear mode
    g1.update_weights(params, log=False)

    # Callback linear mode
    def linear_callback(theta, coeffs):
        return np.dot(coeffs, theta)
    g2.update_weights(params, callback=linear_callback)

    assert np.isclose(v1_1.edges()[0].weight(), v1_2.edges()[0].weight()), \
        f"Callback result {v1_2.edges()[0].weight()} != built-in {v1_1.edges()[0].weight()}"


def test_callback_comparison_with_builtin_log():
    """Test callback produces same result as built-in log mode"""
    g1 = Graph(1)
    v1_1 = g1.create_vertex([1])
    v2_1 = g1.create_vertex([2])
    v1_1.add_edge(v2_1, [2.0, 3.0, 5.0])

    g2 = Graph(1)
    v1_2 = g2.create_vertex([1])
    v2_2 = g2.create_vertex([2])
    v1_2.add_edge(v2_2, [2.0, 3.0, 5.0])

    params = [1.0, 2.0, 3.0]

    # Built-in log mode
    g1.update_weights(params, log=True)

    # Callback multiplicative mode
    def mult_callback(theta, coeffs):
        weight = 1.0
        for c, t in zip(coeffs, theta):
            weight *= c * t
        return weight
    g2.update_weights(params, callback=mult_callback)

    assert np.isclose(v1_1.edges()[0].weight(), v1_2.edges()[0].weight()), \
        f"Callback result {v1_2.edges()[0].weight()} != built-in {v1_1.edges()[0].weight()}"


def test_callback_on_constant_graph_error():
    """Test that callback on constant graph raises error"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, 5.0)  # Constant edge

    def simple_callback(theta, coeffs):
        return 1.0

    with pytest.raises(RuntimeError, match="constant graph"):
        g.update_weights([1.0], callback=simple_callback)


def test_callback_parameter_length_mismatch():
    """Test that parameter length mismatch raises error"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])  # 2 coefficients

    def simple_callback(theta, coeffs):
        return np.dot(coeffs, theta)

    # Graph expects 2 parameters, but we provide 3
    with pytest.raises(RuntimeError, match="Parameter length mismatch"):
        g.update_weights([1.0, 2.0, 3.0], callback=simple_callback)


def test_callback_receives_correct_arrays():
    """Test that callback receives correct theta and coefficient arrays"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    coeffs_expected = [2.5, 3.5, 4.5]
    v1.add_edge(v2, coeffs_expected)

    theta_expected = [1.5, 2.5, 3.5]

    received_theta = None
    received_coeffs = None

    def capture_callback(theta, coeffs):
        nonlocal received_theta, received_coeffs
        received_theta = theta.copy()
        received_coeffs = coeffs.copy()
        return np.dot(coeffs, theta)

    g.update_weights(theta_expected, callback=capture_callback)

    assert np.allclose(received_theta, theta_expected), \
        f"Theta mismatch: expected {theta_expected}, got {received_theta}"
    assert np.allclose(received_coeffs, coeffs_expected), \
        f"Coeffs mismatch: expected {coeffs_expected}, got {received_coeffs}"


def test_callback_with_numpy_functions():
    """Test callback using various numpy functions"""
    g = Graph(1)
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    # Test np.sum
    def sum_callback(theta, coeffs):
        return np.sum(coeffs * theta) + 1.0  # +1 to ensure positive

    g.update_weights([1.0, 2.0], callback=sum_callback)
    assert np.isclose(v1.edges()[0].weight(), 9.0)  # 2*1 + 3*2 + 1 = 9

    # Test np.mean
    def mean_callback(theta, coeffs):
        return np.mean(coeffs * theta) + 1.0  # +1 to ensure positive

    g.update_weights([1.0, 2.0], callback=mean_callback)
    assert np.isclose(v1.edges()[0].weight(), 5.0)  # (2*1 + 3*2)/2 + 1 = 5
