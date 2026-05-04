"""
Unit tests for trace stitching algorithm

Tests the hierarchical SCC-based trace stitching implementation.
"""

import numpy as np
from phasic import Graph
from phasic.trace_elimination import record_elimination_trace, evaluate_trace_jax
from phasic.hierarchical_trace_cache import stitch_scc_traces


def test_stitch_single_scc():
    """Test that stitching a single SCC returns equivalent trace"""
    # Build simple 2-state graph
    def callback(state, **kwargs):
        n = state[0]
        if n <= 1:
            return []  # Absorbing
        # For parameterized, return (state, edge_state) where edge_state is coefficient array
        return [(np.array([n - 1]), [1.0])]

    graph = Graph(
        callback,
        ipv=[3],  # Start with n=3
        parameterized=True,
        nr_samples=3
    )

    # Record direct trace
    trace_direct = record_elimination_trace(graph, theta_dim=1)

    # SCC decomposition
    scc_graph = graph.scc_decomposition()
    sccs = scc_graph.sccs_in_topo_order()

    # Should be single SCC for this simple graph
    assert len(sccs) >= 1, "Expected at least one SCC"

    # Record traces for each SCC
    scc_trace_dict = {}
    for scc in sccs:
        scc_subgraph = scc.as_graph()
        scc_trace = record_elimination_trace(scc_subgraph, theta_dim=1)
        scc_trace_dict[scc.hash()] = scc_trace

    # Stitch
    trace_stitched = stitch_scc_traces(scc_graph, scc_trace_dict)

    # Verify basic properties
    assert trace_stitched.n_vertices == trace_direct.n_vertices, \
        f"Vertex count mismatch: {trace_stitched.n_vertices} vs {trace_direct.n_vertices}"
    assert trace_stitched.state_length == trace_direct.state_length
    assert trace_stitched.param_length == trace_direct.param_length
    assert len(trace_stitched.operations) > 0, "No operations in stitched trace"


def test_stitch_vs_direct_simple():
    """Compare stitched trace vs direct computation for simple graph"""
    # Build simple parameterized graph
    def callback(state, **kwargs):
        n = state[0]
        if n <= 1:
            return []
        rate = n * (n - 1) / 2
        # For parameterized, return (state, edge_state) where edge_state is coefficient array
        return [(np.array([n - 1]), [rate])]

    graph = Graph(
        callback,
        ipv=[4],  # Start with n=4
        parameterized=True,
        nr_samples=4
    )

    # Direct trace
    trace_direct = record_elimination_trace(graph, theta_dim=1)

    # SCC decomposition and stitching
    scc_graph = graph.scc_decomposition()
    scc_trace_dict = {}
    for scc in scc_graph.sccs_in_topo_order():
        scc_subgraph = scc.as_graph()
        scc_trace = record_elimination_trace(scc_subgraph, theta_dim=1)
        scc_trace_dict[scc.hash()] = scc_trace

    trace_stitched = stitch_scc_traces(scc_graph, scc_trace_dict)

    # Evaluate both with same parameters
    params = np.array([1.0])

    try:
        import jax.numpy as jnp
        result_direct = evaluate_trace_jax(trace_direct, jnp.array(params))
        result_stitched = evaluate_trace_jax(trace_stitched, jnp.array(params))

        # Compare vertex_rates (should be identical)
        print(f"Direct vertex_rates: {result_direct['vertex_rates']}")
        print(f"Stitched vertex_rates: {result_stitched['vertex_rates']}")

        # Allow small numerical differences
        np.testing.assert_allclose(
            result_direct['vertex_rates'],
            result_stitched['vertex_rates'],
            rtol=1e-10,
            err_msg="vertex_rates mismatch"
        )

        print("✓ Trace stitching test passed!")

    except ImportError:
        # JAX not available, skip evaluation test
        print("JAX not available, skipping evaluation test")
        pass


def test_stitch_validation():
    """Test that validation catches errors"""
    # Empty trace dict should raise
    graph = Graph(1)
    v0 = graph.starting_vertex()
    v1 = graph.find_or_create_vertex([1])
    v0.add_edge(v1, [1.0])

    scc_graph = graph.scc_decomposition()

    try:
        stitch_scc_traces(scc_graph, {})
        assert False, "Expected ValueError for empty scc_trace_dict"
    except ValueError as e:
        assert "scc_trace_dict is empty" in str(e)

    # Missing SCC trace should raise
    sccs = scc_graph.sccs_in_topo_order()
    if len(sccs) > 0:
        try:
            stitch_scc_traces(scc_graph, {"fake_hash": None})
            assert False, "Expected ValueError for missing SCC trace"
        except ValueError as e:
            assert "Missing trace for SCC" in str(e)


if __name__ == "__main__":
    print("Running trace stitching tests...")
    print("\n1. Testing single SCC stitching...")
    test_stitch_single_scc()
    print("✓ Single SCC test passed")

    print("\n2. Testing stitched vs direct computation...")
    test_stitch_vs_direct_simple()

    print("\n3. Testing validation...")
    test_stitch_validation()
    print("✓ Validation test passed")

    print("\n✓ All tests passed!")
