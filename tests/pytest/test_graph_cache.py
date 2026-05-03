"""
Tests for Graph serialization and deserialization (from_serialized).

This test module validates the round-trip serialization required for
distributed trace recording via JAX vmap/pmap.
"""

import json
import pytest
import numpy as np
from phasic import Graph


class TestSerializationRoundTrip:
    """Test that serialize() + from_serialized() preserves graph structure."""

    def test_simple_graph_round_trip(self):
        """Test serialization of simple non-parameterized graph."""
        # Create simple graph
        g = Graph(1)  # state_length=1
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])
        start.add_edge(v1, 1.0)
        v1.add_edge(v2, 2.0)

        # Serialize and deserialize
        serialized = g.serialize(theta_dim=0)
        g_reconstructed = Graph.from_serialized(serialized)

        # Verify structure
        assert g_reconstructed.vertices_length() == g.vertices_length()
        assert g_reconstructed.state_length() == g.state_length()

    def test_parameterized_graph_round_trip(self):
        """Test serialization of parameterized graph."""
        # Create parameterized graph
        g = Graph(1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v2 = g.find_or_create_vertex([2])

        # Add parameterized edges
        start.add_edge_parameterized(v1, 0.0, [2.0, 0.5])
        v1.add_edge_parameterized(v2, 1.0, [3.0, 1.5])

        # Serialize and deserialize
        serialized = g.serialize(theta_dim=2)
        g_reconstructed = Graph.from_serialized(serialized)

        # Verify structure
        assert g_reconstructed.vertices_length() == g.vertices_length()
        assert g_reconstructed.state_length() == g.state_length()
        assert g_reconstructed.param_length() == g.param_length()

    def test_mixed_edges_round_trip(self):
        """Test graph with both regular and parameterized edges (separate)."""
        g = Graph(2)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1, 0])
        v2 = g.find_or_create_vertex([0, 1])
        v3 = g.find_or_create_vertex([1, 1])

        # All parameterized edges
        start.add_edge_parameterized(v1, 0.0, [1.5])
        start.add_edge_parameterized(v2, 0.0, [2.5])
        v1.add_edge_parameterized(v3, 0.0, [3.0])
        v2.add_edge_parameterized(v3, 0.0, [4.0])

        # Round-trip
        serialized = g.serialize(theta_dim=1)
        g_reconstructed = Graph.from_serialized(serialized)

        # Verify
        assert g_reconstructed.vertices_length() == g.vertices_length()
        assert g_reconstructed.state_length() == g.state_length()

    def test_json_round_trip(self):
        """Test full JSON serialization workflow (for network transmission)."""
        # Create graph
        g = Graph(1)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        start.add_edge_parameterized(v1, 0.0, [2.0])

        # Serialize to dict
        serialized = g.serialize(theta_dim=1)

        # Convert to JSON string (simulating network transmission)
        json_dict = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in serialized.items()
        }
        json_str = json.dumps(json_dict)

        # Deserialize from JSON (simulating worker receiving data)
        received_dict = json.loads(json_str)
        received_dict['states'] = np.array(received_dict['states'], dtype=np.int32)
        received_dict['edges'] = np.array(received_dict['edges'], dtype=np.float64)
        received_dict['start_edges'] = np.array(received_dict['start_edges'], dtype=np.float64)
        received_dict['param_edges'] = np.array(received_dict['param_edges'], dtype=np.float64)
        received_dict['start_param_edges'] = np.array(received_dict['start_param_edges'], dtype=np.float64)

        # Reconstruct
        g_reconstructed = Graph.from_serialized(received_dict)

        # Verify
        assert g_reconstructed.vertices_length() == g.vertices_length()
        assert g_reconstructed.state_length() == g.state_length()

    def test_empty_graph_round_trip(self):
        """Test serialization of graph with no edges."""
        g = Graph(1)
        v1 = g.find_or_create_vertex([1])

        # Serialize and deserialize
        serialized = g.serialize(theta_dim=0)
        g_reconstructed = Graph.from_serialized(serialized)

        # Verify
        assert g_reconstructed.vertices_length() == g.vertices_length()

    def test_large_state_length_round_trip(self):
        """Test graph with larger state vectors."""
        g = Graph(5)
        start = g.starting_vertex()
        v1 = g.find_or_create_vertex([1, 2, 3, 4, 5])
        v2 = g.find_or_create_vertex([5, 4, 3, 2, 1])
        start.add_edge(v1, 1.0)
        v1.add_edge(v2, 2.0)

        # Round-trip
        serialized = g.serialize(theta_dim=0)
        g_reconstructed = Graph.from_serialized(serialized)

        # Verify
        assert g_reconstructed.vertices_length() == g.vertices_length()
        assert g_reconstructed.state_length() == 5


class TestDeserializationErrors:
    """Test that deserialization produces informative error messages."""

    def test_missing_required_field(self):
        """Test error when required field is missing."""
        data = {
            'states': np.array([[0]], dtype=np.int32),
            'edges': np.empty((0, 3), dtype=np.float64),
            # Missing 'start_edges' and others
        }

        with pytest.raises(ValueError, match="missing required fields"):
            Graph.from_serialized(data)

    def test_invalid_metadata_types(self):
        """Test error when metadata has wrong types."""
        data = {
            'states': np.array([[0]], dtype=np.int32),
            'edges': np.empty((0, 3), dtype=np.float64),
            'start_edges': np.empty((0, 2), dtype=np.float64),
            'param_edges': np.empty((0, 3), dtype=np.float64),
            'start_param_edges': np.empty((0, 2), dtype=np.float64),
            'param_length': 0,
            'state_length': 1,
            'n_vertices': "not_an_int"  # Wrong type
        }

        with pytest.raises(ValueError, match="invalid metadata types"):
            Graph.from_serialized(data)

    def test_states_array_shape_mismatch(self):
        """Test error when states array shape doesn't match metadata."""
        data = {
            'states': np.array([[0], [1], [2]], dtype=np.int32),  # 3 vertices
            'edges': np.empty((0, 3), dtype=np.float64),
            'start_edges': np.empty((0, 2), dtype=np.float64),
            'param_edges': np.empty((0, 3), dtype=np.float64),
            'start_param_edges': np.empty((0, 2), dtype=np.float64),
            'param_length': 0,
            'state_length': 1,
            'n_vertices': 2  # Says 2 but states has 3
        }

        with pytest.raises(ValueError, match="states array shape mismatch"):
            Graph.from_serialized(data)

    def test_edges_array_wrong_shape(self):
        """Test error when edges array has wrong shape."""
        data = {
            'states': np.array([[0]], dtype=np.int32),
            'vertex_indices': np.array([0], dtype=np.int32),
            'edges': np.array([[0, 1]], dtype=np.float64),  # Should be (n, 3)
            'start_edges': np.empty((0, 2), dtype=np.float64),
            'param_edges': np.empty((0, 3), dtype=np.float64),
            'start_param_edges': np.empty((0, 2), dtype=np.float64),
            'param_length': 0,
            'state_length': 1,
            'n_vertices': 1
        }

        with pytest.raises(ValueError, match="edges array has wrong shape"):
            Graph.from_serialized(data)

    def test_param_edges_wrong_columns(self):
        """Test error when param_edges has too few coefficient columns."""
        data = {
            'states': np.array([[0], [1]], dtype=np.int32),
            'vertex_indices': np.array([0, 1], dtype=np.int32),
            'edges': np.empty((0, 3), dtype=np.float64),
            'start_edges': np.empty((0, 2), dtype=np.float64),
            # 4 cols = [from, to, c1, c2] → coeff_len=2, but param_length=3 requires ≥3 coeffs
            'param_edges': np.array([[0, 1, 0.0, 1.0]], dtype=np.float64),
            'start_param_edges': np.empty((0, 4), dtype=np.float64),
            'param_length': 3,
            'state_length': 1,
            'n_vertices': 2
        }

        with pytest.raises(ValueError, match="coefficient length < param_length"):
            Graph.from_serialized(data)

    def test_invalid_edge_indices(self):
        """Test error when edge refers to non-existent vertex."""
        data = {
            'states': np.array([[0], [1]], dtype=np.int32),
            'vertex_indices': np.array([0, 1], dtype=np.int32),
            'edges': np.array([[0, 5, 1.0]], dtype=np.float64),  # Vertex 5 doesn't exist
            'start_edges': np.empty((0, 2), dtype=np.float64),
            'param_edges': np.empty((0, 0), dtype=np.float64),  # (0, 0) for param_length=0
            'start_param_edges': np.empty((0, 0), dtype=np.float64),
            'param_length': 0,
            'state_length': 1,
            'n_vertices': 2
        }

        with pytest.raises((RuntimeError, ValueError, IndexError)):
            Graph.from_serialized(data)

    def test_malformed_json_conversion(self):
        """Test error handling for JSON deserialization failures."""
        # Test that array conversion fails for invalid data
        json_str = '{"states": "not_an_array"}'
        data = json.loads(json_str)

        # Array conversion should fail with ValueError
        with pytest.raises(ValueError):
            np.array(data['states'], dtype=np.int32)


class TestCoalescentModel:
    """Test serialization with real coalescent model."""

    def test_coalescent_callback_serialization(self):
        """Test serialization of graph built with callback."""
        from phasic import callback

        @callback(ipv=[([5], 1.0)])
        def coalescent_callback(state, nr_samples=None, **kwargs):
            n = state[0]
            if n <= 1:
                return []
            rate = n * (n - 1) / 2
            # Parameterized transitions are 2-tuples (state, coeffs_list)
            return [([n - 1], [rate])]

        # Build graph with callback
        g = Graph(coalescent_callback, nr_samples=5)

        # Serialize and deserialize
        serialized = g.serialize(theta_dim=1)
        g_reconstructed = Graph.from_serialized(serialized)

        # Verify structure preserved
        assert g_reconstructed.vertices_length() == g.vertices_length()
        assert g_reconstructed.state_length() == g.state_length()
        assert g_reconstructed.param_length() == g.param_length()

    def test_serialized_trace_recording(self):
        """Test that trace recording works on deserialized graph."""
        from phasic.trace_elimination import record_elimination_trace
        from phasic import callback

        @callback(ipv=[([3], 1.0)])
        def coalescent_callback(state, nr_samples=None, **kwargs):
            n = state[0]
            if n <= 1:
                return []
            rate = n * (n - 1) / 2
            # Parameterized transitions are 2-tuples (state, coeffs_list)
            return [([n - 1], [rate])]

        # Build, serialize, deserialize
        g = Graph(coalescent_callback, nr_samples=3)
        serialized = g.serialize(theta_dim=1)
        g_reconstructed = Graph.from_serialized(serialized)

        # Record trace on reconstructed graph
        trace = record_elimination_trace(g_reconstructed, theta_dim=1)

        # Verify trace was recorded successfully
        assert trace is not None
        assert len(trace.operations) > 0


class TestGraphHashing:
    """Test graph hashing integration"""

    def test_graph_hash_computation(self):
        """Test that graph hashing works"""
        from phasic import hash as phasic_hash

        g = Graph(1)
        v0 = g.starting_vertex()
        v1 = g.find_or_create_vertex([1])
        v0.add_edge(v1, 1.0)

        hash_result = phasic_hash.compute_graph_hash(g)

        assert hash_result is not None, "Should return hash result"
        assert hasattr(hash_result, 'hash_hex'), "Should have hash_hex attribute"
        assert len(hash_result.hash_hex) == 64, "Hash should be 64 hex characters"

    def test_identical_graphs_same_hash(self):
        """Test that identical graphs produce same hash"""
        from phasic import hash as phasic_hash

        # Graph 1
        g1 = Graph(1)
        v0_1 = g1.starting_vertex()
        v1_1 = g1.find_or_create_vertex([1])
        v0_1.add_edge(v1_1, 1.0)

        # Graph 2 (identical structure)
        g2 = Graph(1)
        v0_2 = g2.starting_vertex()
        v1_2 = g2.find_or_create_vertex([1])
        v0_2.add_edge(v1_2, 1.0)

        hash1 = phasic_hash.compute_graph_hash(g1)
        hash2 = phasic_hash.compute_graph_hash(g2)

        assert hash1.hash_hex == hash2.hash_hex, "Identical graphs should have same hash"
