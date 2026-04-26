"""
Tests for log-space weight computation

Tests the log=True flag for update_weights() that computes edge weights
as products in log-space rather than linear combinations.
"""

import pytest
import numpy as np
from phasic import Graph


def test_log_space_simple():
    """Test basic log-space weight computation"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])

    # Add edge with coefficients [2.0, 3.0] - must be from non-starting vertex
    v1.add_edge(v2, [2.0, 3.0])

    # Test standard mode: 2.0*1.0 + 3.0*2.0 = 8.0
    g.update_weights([1.0, 2.0], log=False)
    assert np.isclose(v1.edges()[0].weight(), 8.0), \
        f"Standard mode: expected 8.0, got {v1.edges()[0].weight()}"

    # Test log mode: (2.0*1.0) * (3.0*2.0) = 2.0 * 6.0 = 12.0
    g.update_weights([1.0, 2.0], log=True)
    assert np.isclose(v1.edges()[0].weight(), 12.0), \
        f"Log mode: expected 12.0, got {v1.edges()[0].weight()}"


def test_log_space_single_param():
    """Test log-space with single parameter"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])

    v1.add_edge(v2, [5.0])

    # Standard: 5.0*3.0 = 15.0
    g.update_weights([3.0], log=False)
    assert np.isclose(v1.edges()[0].weight(), 15.0), \
        f"Standard mode: expected 15.0, got {v1.edges()[0].weight()}"

    # Log: (5.0*3.0) = 15.0 (same for single param)
    g.update_weights([3.0], log=True)
    assert np.isclose(v1.edges()[0].weight(), 15.0), \
        f"Log mode: expected 15.0, got {v1.edges()[0].weight()}"


def test_log_space_error_negative():
    """Test that log-space raises error for negative products"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])

    # Use positive coefficients so edge can be added, but negative when multiplied with param
    v1.add_edge(v2, [2.0, 3.0])

    # Use negative parameter to get negative product
    # Product: 3.0 * (-2.0) = -6.0 (negative)
    with pytest.raises(RuntimeError, match="positive"):
        g.update_weights([1.0, -2.0], log=True)


def test_log_space_error_zero():
    """Test that log-space raises error for zero products"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])

    v1.add_edge(v2, [2.0, 3.0])

    # First set non-zero weights to allow the graph to be valid
    g.update_weights([1.0, 2.0], log=False)

    # Should fail in log mode (zero product: 3.0 * 0.0 = 0)
    with pytest.raises(RuntimeError, match="positive"):
        g.update_weights([1.0, 0.0], log=True)


def test_log_space_numerical_stability():
    """Test log-space handles small values better"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])

    # Small coefficients that would underflow in normal multiplication
    v1.add_edge(v2, [1e-10, 1e-10, 1e-10])

    g.update_weights([1.0, 1.0, 1.0], log=True)
    weight = v1.edges()[0].weight()

    # Should be 1e-30 without underflow
    assert np.isclose(weight, 1e-30, rtol=1e-5), \
        f"Expected 1e-30, got {weight}"


def test_log_space_three_params():
    """Test log-space with three parameters"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])

    # Coefficients: [2.0, 3.0, 5.0]
    # Params: [1.0, 2.0, 3.0]
    v1.add_edge(v2, [2.0, 3.0, 5.0])

    # Standard: 2.0*1.0 + 3.0*2.0 + 5.0*3.0 = 2 + 6 + 15 = 23.0
    g.update_weights([1.0, 2.0, 3.0], log=False)
    assert np.isclose(v1.edges()[0].weight(), 23.0), \
        f"Standard mode: expected 23.0, got {v1.edges()[0].weight()}"

    # Log: (2.0*1.0) * (3.0*2.0) * (5.0*3.0) = 2.0 * 6.0 * 15.0 = 180.0
    g.update_weights([1.0, 2.0, 3.0], log=True)
    assert np.isclose(v1.edges()[0].weight(), 180.0), \
        f"Log mode: expected 180.0, got {v1.edges()[0].weight()}"


def test_log_space_multiple_edges():
    """Test log-space with multiple edges"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v3 = g.create_vertex([3])

    # Two edges with different coefficients
    v1.add_edge(v2, [2.0, 1.0])
    v1.add_edge(v3, [3.0, 4.0])

    # Log mode
    g.update_weights([2.0, 3.0], log=True)

    # Edge 1: (2.0*2.0) * (1.0*3.0) = 4.0 * 3.0 = 12.0
    assert np.isclose(v1.edges()[0].weight(), 12.0), \
        f"Edge 1: expected 12.0, got {v1.edges()[0].weight()}"

    # Edge 2: (3.0*2.0) * (4.0*3.0) = 6.0 * 12.0 = 72.0
    assert np.isclose(v1.edges()[1].weight(), 72.0), \
        f"Edge 2: expected 72.0, got {v1.edges()[1].weight()}"


def test_log_space_trace_evaluation():
    """Test log-space works with trace elimination"""
    from phasic.trace_elimination import record_elimination_trace, evaluate_trace

    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    # Record trace
    trace = record_elimination_trace(g, theta_dim=2)

    # Evaluate with standard mode
    result_standard = evaluate_trace(trace, np.array([1.0, 2.0]), use_log=False)

    # Evaluate with log mode
    result_log = evaluate_trace(trace, np.array([1.0, 2.0]), use_log=True)

    # Results should differ (8.0 vs 12.0 from test_log_space_simple)
    assert not np.allclose(result_standard['vertex_rates'], result_log['vertex_rates']), \
        "Standard and log mode should produce different results"


def test_log_space_instantiate_from_trace():
    """Test log-space works with instantiate_from_trace"""
    from phasic.trace_elimination import record_elimination_trace, instantiate_from_trace

    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    # Record trace
    trace = record_elimination_trace(g, theta_dim=2)

    # Instantiate with log mode
    params = np.array([1.0, 2.0])
    graph_log = instantiate_from_trace(trace, params, use_log=True)

    # Find the correct vertex with edges (index 1 based on the trace)
    vertex_with_edges = graph_log.vertices()[1]
    edges = vertex_with_edges.edges()
    assert len(edges) == 1
    assert np.isclose(edges[0].weight(), 12.0), \
        f"Expected 12.0, got {edges[0].weight()}"


def test_log_space_jax_compatibility():
    """Test log-space works with JAX transformations"""
    try:
        import jax
        import jax.numpy as jnp
        from phasic.trace_elimination import record_elimination_trace, evaluate_trace_jax
    except ImportError:
        pytest.skip("JAX not available")

    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    trace = record_elimination_trace(g, theta_dim=2)

    # Test jit
    @jax.jit
    def eval_log(params):
        return evaluate_trace_jax(trace, params, use_log=True)

    result = eval_log(jnp.array([1.0, 2.0]))
    assert result is not None
    assert 'vertex_rates' in result

    # Test grad
    def loss(params):
        result = evaluate_trace_jax(trace, params, use_log=True)
        return jnp.sum(result['vertex_rates'])

    grad_fn = jax.grad(loss)
    gradient = grad_fn(jnp.array([1.0, 2.0]))

    assert gradient.shape == (2,)
    assert not np.any(np.isnan(gradient)), \
        f"Gradient contains NaN: {gradient}"


def test_log_space_trace_error_handling():
    """Test that trace evaluation with log mode raises error for invalid inputs"""
    from phasic.trace_elimination import record_elimination_trace, evaluate_trace

    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    trace = record_elimination_trace(g, theta_dim=2)

    # Should raise error for negative product (3.0 * (-2.0) = -6.0)
    with pytest.raises(ValueError, match="positive"):
        evaluate_trace(trace, np.array([1.0, -2.0]), use_log=True)


def test_log_space_default_false():
    """Test that log defaults to False for backward compatibility"""
    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, 3.0])

    # Without log argument, should use standard mode
    # Standard: 2.0*1.0 + 3.0*2.0 = 8.0
    g.update_weights([1.0, 2.0])
    assert np.isclose(v1.edges()[0].weight(), 8.0), \
        "Default behavior should be standard mode (sum)"


def test_log_space_comparison():
    """Compare standard and log modes side by side"""
    # Standard mode graph
    g_standard = Graph(1)
    v0_s = g_standard.starting_vertex()
    v1_s = g_standard.create_vertex([1])
    v2_s = g_standard.create_vertex([2])
    v1_s.add_edge(v2_s, [2.0, 3.0, 5.0])

    # Log mode graph
    g_log = Graph(1)
    v0_l = g_log.starting_vertex()
    v1_l = g_log.create_vertex([1])
    v2_l = g_log.create_vertex([2])
    v1_l.add_edge(v2_l, [2.0, 3.0, 5.0])

    params = [1.5, 2.5, 3.5]

    # Update weights
    g_standard.update_weights(params, log=False)
    g_log.update_weights(params, log=True)

    # Standard: 2.0*1.5 + 3.0*2.5 + 5.0*3.5 = 3 + 7.5 + 17.5 = 28.0
    weight_standard = v1_s.edges()[0].weight()
    assert np.isclose(weight_standard, 28.0), \
        f"Standard: expected 28.0, got {weight_standard}"

    # Log: (2.0*1.5) * (3.0*2.5) * (5.0*3.5) = 3.0 * 7.5 * 17.5 = 393.75
    weight_log = v1_l.edges()[0].weight()
    assert np.isclose(weight_log, 393.75), \
        f"Log: expected 393.75, got {weight_log}"

    # Verify they're different
    assert not np.isclose(weight_standard, weight_log), \
        "Standard and log modes should produce different results"


# def test_nan_coefficients_standard_mode():
#     """Test NaN coefficients in standard mode (nansum behavior)"""
#     g = Graph(1)
#     v0 = g.starting_vertex()
#     v1 = g.create_vertex([1])
#     v2 = g.create_vertex([2])

#     # Coefficients with NaN (should skip middle parameter)
#     v1.add_edge(v2, [2.0, np.nan, 3.0])

#     # Standard mode: 2.0*1.0 + 3.0*3.0 = 2.0 + 9.0 = 11.0 (skips NaN)
#     g.update_weights([1.0, 2.0, 3.0], log=False)
#     assert np.isclose(v1.edges()[0].weight(), 11.0), \
#         f"Expected 11.0, got {v1.edges()[0].weight()}"


# def test_nan_coefficients_log_mode():
#     """Test NaN coefficients in log mode (nanprod behavior)"""
#     g = Graph(1)
#     v0 = g.starting_vertex()
#     v1 = g.create_vertex([1])
#     v2 = g.create_vertex([2])

#     # Coefficients with NaN (should skip middle parameter)
#     v1.add_edge(v2, [2.0, np.nan, 3.0])

#     # Log mode: (2.0*1.0) * (3.0*3.0) = 2.0 * 9.0 = 18.0 (skips NaN)
#     g.update_weights([1.0, 2.0, 3.0], log=True)
#     assert np.isclose(v1.edges()[0].weight(), 18.0), \
#         f"Expected 18.0, got {v1.edges()[0].weight()}"


# def test_nan_coefficients_multiple_nans():
#     """Test multiple NaN coefficients"""
#     g = Graph(1)
#     v0 = g.starting_vertex()
#     v1 = g.create_vertex([1])
#     v2 = g.create_vertex([2])

#     # Only first and last coefficients are used
#     v1.add_edge(v2, [2.0, np.nan, np.nan, np.nan, 5.0])

#     # Standard: 2.0*1.0 + 5.0*5.0 = 2.0 + 25.0 = 27.0
#     g.update_weights([1.0, 2.0, 3.0, 4.0, 5.0], log=False)
#     assert np.isclose(v1.edges()[0].weight(), 27.0), \
#         f"Standard mode: expected 27.0, got {v1.edges()[0].weight()}"

#     # Log: (2.0*1.0) * (5.0*5.0) = 2.0 * 25.0 = 50.0
#     g.update_weights([1.0, 2.0, 3.0, 4.0, 5.0], log=True)
#     assert np.isclose(v1.edges()[0].weight(), 50.0), \
#         f"Log mode: expected 50.0, got {v1.edges()[0].weight()}"


# def test_nan_coefficients_all_nan_standard():
#     """Test all NaN coefficients in standard mode"""
#     g = Graph(1)
#     v0 = g.starting_vertex()
#     v1 = g.create_vertex([1])
#     v2 = g.create_vertex([2])

#     v1.add_edge(v2, [np.nan, np.nan, np.nan])

#     # Standard mode: nansum([]) = 0.0
#     g.update_weights([1.0, 2.0, 3.0], log=False)
#     assert np.isclose(v1.edges()[0].weight(), 0.0), \
#         f"Expected 0.0, got {v1.edges()[0].weight()}"


# def test_nan_coefficients_all_nan_log():
#     """Test all NaN coefficients in log mode"""
#     g = Graph(1)
#     v0 = g.starting_vertex()
#     v1 = g.create_vertex([1])
#     v2 = g.create_vertex([2])

#     v1.add_edge(v2, [np.nan, np.nan, np.nan])

#     # Log mode: nanprod([]) = 1.0 (identity for multiplication)
#     g.update_weights([1.0, 2.0, 3.0], log=True)
#     assert np.isclose(v1.edges()[0].weight(), 1.0), \
#         f"Expected 1.0, got {v1.edges()[0].weight()}"


def test_nan_coefficients_trace_evaluation():
    """Test NaN coefficients work with trace evaluation"""
    from phasic.trace_elimination import record_elimination_trace, evaluate_trace

    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, np.nan, 3.0])

    trace = record_elimination_trace(g, theta_dim=3)

    # Standard mode
    result_standard = evaluate_trace(trace, np.array([1.0, 2.0, 3.0]), use_log=False)
    # Log mode
    result_log = evaluate_trace(trace, np.array([1.0, 2.0, 3.0]), use_log=True)

    # Results should differ (11.0 vs 18.0)
    assert not np.allclose(result_standard['vertex_rates'], result_log['vertex_rates'])


def test_nan_coefficients_jax_compatibility():
    """Test NaN coefficients work with JAX"""
    try:
        import jax
        import jax.numpy as jnp
        from phasic.trace_elimination import record_elimination_trace, evaluate_trace_jax
    except ImportError:
        pytest.skip("JAX not available")

    g = Graph(1)
    v0 = g.starting_vertex()
    v1 = g.create_vertex([1])
    v2 = g.create_vertex([2])
    v1.add_edge(v2, [2.0, np.nan, 3.0])

    trace = record_elimination_trace(g, theta_dim=3)

    # Test with log mode
    result = evaluate_trace_jax(trace, jnp.array([1.0, 2.0, 3.0]), use_log=True)
    assert result is not None
    assert 'vertex_rates' in result

    # Test grad works with NaN coefficients
    def loss(params):
        result = evaluate_trace_jax(trace, params, use_log=True)
        return jnp.sum(result['vertex_rates'])

    grad_fn = jax.grad(loss)
    gradient = grad_fn(jnp.array([1.0, 2.0, 3.0]))

    # Gradient should be zero for the NaN coefficient parameter
    assert gradient.shape == (3,)
    # Note: gradient[1] should be 0 since coefficient is NaN


# def test_nan_coefficients_psmc_use_case():
#     """Test PSMC-style epoch-specific parameters"""
#     g = Graph(1)
#     v0 = g.starting_vertex()
#     v1 = g.create_vertex([1])
#     v2 = g.create_vertex([2])

#     # Epoch 1: uses θ₀ and θ₂ (skips θ₁)
#     v1.add_edge(v2, [1.0, np.nan, 0.5])

#     # Params: [epoch_rec_prob=0.1, unused=999, coal_rate=2.0]
#     # Log mode: (1.0*0.1) * (0.5*2.0) = 0.1 * 1.0 = 0.1
#     g.update_weights([0.1, 999.0, 2.0], log=True)
#     assert np.isclose(v1.edges()[0].weight(), 0.1), \
#         f"Expected 0.1, got {v1.edges()[0].weight()}"


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])
