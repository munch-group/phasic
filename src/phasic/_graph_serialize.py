"""serialize / from_serialized (array round-trip) for :class:`~phasic.Graph`.

Extracted verbatim from ``Graph`` (Stage-3 WS-C). Pure relocation. ``serialize``
is assigned onto Graph as-is; ``from_serialized`` is a plain function taking
``cls`` and is wrapped with ``classmethod(...)`` at the assignment in
``__init__.py`` (it constructs via ``cls(...)``, no direct Graph reference).
"""
from __future__ import annotations

from typing import Any
import numpy as np


def serialize(self, theta_dim: int | None = None) -> dict[str, np.ndarray]:
    """
    Serialize graph to array representation for efficient computation.

    Parameters
    ----------
    theta_dim : int, optional
        Number of parameters for parameterized edges. If not provided, will be
        auto-detected by probing edge states. Providing this explicitly avoids
        potential issues with auto-detection reading garbage memory.

    Returns
    -------
    dict
        Dictionary containing:
        - 'states': Array of vertex states (n_vertices, state_dim)
        - 'edges': Array of regular edges [from_idx, to_idx, weight] (n_edges, 3)
        - 'start_edges': Array of starting vertex regular edges [to_idx, weight] (n_start_edges, 2)
        - 'param_edges': Array of parameterized edges [from_idx, to_idx, x1, x2, ...] (n_param_edges, theta_dim+2)
        - 'start_param_edges': Array of starting vertex parameterized edges [to_idx, x1, x2, ...] (n_start_param_edges, theta_dim+1)
          NOTE: start_param_edges should be empty (starting edges are not parameterized)
        - 'param_length': Length of parameter vector (0 if no parameterized edges)
        - 'state_length': Integer state dimension
        - 'n_vertices': Number of vertices
    """
    vertices_list = list(self.vertices())
    n_vertices = len(vertices_list)

    if n_vertices == 0:
        raise ValueError("Graph has no vertices (except starting vertex)")

    # Extract states and create vertex index mapping
    state_length = self.state_length()
    states = np.zeros((n_vertices, state_length), dtype=np.int32)
    vertex_indices = np.zeros(n_vertices, dtype=np.int32)

    # Map vertex.index() -> enumeration position in vertices_list
    # This handles duplicate states correctly (multiple vertices with same state)
    vertex_idx_to_enum = {}

    for i, v in enumerate(vertices_list):
        state = v.state()
        states[i, :] = state
        vertex_indices[i] = v.index()
        vertex_idx_to_enum[v.index()] = i

    # Get parameter length directly from graph (set by first add_edge() call)
    start = self.starting_vertex()

    # Use provided theta_dim if given, otherwise get from graph
    if theta_dim is None:
        theta_dim = self.param_length()

    # theta_dim is now always correct (no probing needed)

    # Extract parameterized edges FIRST (needed to build exclusion set before extracting regular edges)
    start_state = tuple(start.state())

    # Extract parameterized edges between vertices (excluding starting vertex)
    # With unified interface: parameterized_edges() returns edges with coefficient arrays
    param_edges_list = []
    start_vertex_idx = start.index()
    if theta_dim > 0:  # Export all edges with coefficient arrays
        for i, v in enumerate(vertices_list):
            # Skip starting vertex edges (they're handled separately)
            # Use vertex index comparison, not state comparison (states may be duplicated)
            if v.index() == start_vertex_idx:
                continue

            from_idx = i
            for edge in v.parameterized_edges():
                to_vertex = edge.to()
                if to_vertex.index() in vertex_idx_to_enum:
                    to_idx = vertex_idx_to_enum[to_vertex.index()]
                    # Get FULL coefficient array (all coefficients, not just theta_dim)
                    # This is critical when coefficients_length > theta_dim
                    coeff_len = edge.coefficients_length()
                    edge_state = list(edge.edge_state(coeff_len))
                    # Include all parameterized edges (even with all-zero coefficients)
                    if edge_state:
                        # Store: [from_idx, to_idx, x1, x2, x3, ...]
                        param_edges_list.append([from_idx, to_idx] + edge_state)

    # Convert to numpy array - all edges should have same coefficient length
    # If not, will raise ValueError with helpful message
    if param_edges_list:
        try:
            param_edges = np.array(param_edges_list, dtype=np.float64)
        except ValueError as e:
            # Check if edges have different coefficient lengths
            lengths = set(len(row) for row in param_edges_list)
            if len(lengths) > 1:
                raise ValueError(
                    f"Graph serialization failed: parameterized edges have inconsistent coefficient lengths.\n"
                    f"  Found coefficient lengths: {sorted(lengths)}\n"
                    f"  All edges must have the same coefficient length.\n"
                    f"  Hint: Check callback function - it may be returning edges with different coefficient arrays."
                ) from e
            else:
                raise
    else:
        # Empty case - use theta_dim for consistency
        param_edges = np.empty((0, theta_dim + 2 if theta_dim > 0 else 0), dtype=np.float64)

    # Extract starting vertex parameterized edges FIRST (needed to build exclusion set)
    # NOTE: Starting vertex edges are NEVER rescaled by update_weights() (see starting vertex fix)
    # So we should NOT export them as parameterized edges - they are effectively constant
    start_param_edges_list = []
    if False:  # Starting edges are never parameterized (always constant)
        for edge in start.parameterized_edges():
            to_vertex = edge.to()
            if to_vertex.index() in vertex_idx_to_enum:
                to_idx = vertex_idx_to_enum[to_vertex.index()]
                # Get coefficient array (length is guaranteed to be theta_dim)
                edge_state = list(edge.edge_state(theta_dim))
                # Only include edges with non-empty edge states
                if edge_state and any(x != 0 for x in edge_state):
                    # Store: [to_idx, x1, x2, x3, ...]
                    start_param_edges_list.append([to_idx] + edge_state)

    start_param_edges = np.array(start_param_edges_list, dtype=np.float64) if start_param_edges_list else np.empty((0, theta_dim + 1 if theta_dim > 0 else 0), dtype=np.float64)

    # Build set of (from_idx, to_idx) pairs for parameterized edges to skip in regular edges
    param_edge_pairs = set()
    for edge_data in start_param_edges_list:
        to_idx = int(edge_data[0])
        param_edge_pairs.add((-1, to_idx))  # -1 represents starting vertex
    for edge_data in param_edges_list:
        from_idx = int(edge_data[0])
        to_idx = int(edge_data[1])
        param_edge_pairs.add((from_idx, to_idx))

    # Extract regular edges between vertices (excluding starting vertex)
    # Skip edges that have parameterized versions.
    #
    # Coefficient-less edges (constant edges with
    # coefficients_length == 0, created by add_aux_vertex_constant)
    # are emitted into a separate ``constant_edges`` list. The
    # GraphBuilder reads them and constructs them via direct
    # ptd_edge struct manipulation, bypassing the EDGE_MODE lock so
    # they can coexist with parameterised edges on the same graph.
    # This is the round-trip path for the t-aux trapping loops in
    # joint_stop_prob_graph().
    edges_list = []
    constant_edges_list = []
    start_vertex_idx = start.index()
    for i, v in enumerate(vertices_list):
        # Skip starting vertex edges (they're handled separately)
        # Use vertex index comparison, not state comparison (states may be duplicated)
        if v.index() == start_vertex_idx:
            continue

        from_idx = i
        for edge in v.edges():
            to_vertex = edge.to()
            if to_vertex.index() in vertex_idx_to_enum:
                to_idx = vertex_idx_to_enum[to_vertex.index()]
                # Skip if this edge also has a parameterized version
                if (from_idx, to_idx) not in param_edge_pairs:
                    weight = edge.weight()
                    # Route coefficient-less constant edges to a
                    # dedicated list so the deserialiser can rebuild
                    # them without triggering the EDGE_MODE lock.
                    if edge.coefficients_length() == 0:
                        constant_edges_list.append([from_idx, to_idx, weight])
                    else:
                        edges_list.append([from_idx, to_idx, weight])

    edges = np.array(edges_list, dtype=np.float64) if edges_list else np.empty((0, 3), dtype=np.float64)
    constant_edges = (
        np.array(constant_edges_list, dtype=np.float64)
        if constant_edges_list else np.empty((0, 3), dtype=np.float64)
    )

    # Extract starting vertex regular edges (skip those with parameterized versions)
    start_edges_list = []
    for edge in start.edges():
        to_vertex = edge.to()
        if to_vertex.index() in vertex_idx_to_enum:
            to_idx = vertex_idx_to_enum[to_vertex.index()]
            # Skip if this edge also has a parameterized version
            if (-1, to_idx) not in param_edge_pairs:
                weight = edge.weight()
                start_edges_list.append([to_idx, weight])

    start_edges = np.array(start_edges_list, dtype=np.float64) if start_edges_list else np.empty((0, 2), dtype=np.float64)

    _serialized = {
        'states': states,
        'vertex_indices': vertex_indices,
        'edges': edges,
        'constant_edges': constant_edges,
        'start_edges': start_edges,
        'param_edges': param_edges,
        'start_param_edges': start_param_edges,
        'param_length': theta_dim,
        'state_length': state_length,
        'n_vertices': n_vertices,
        'weight_mode': self._weight_mode,
        # Carry the elimination-ordering flag so the FFI/pybind
        # GraphBuilder that rebuilds this graph inherits it. Without
        # this, graph.dyn_ordering=True does NOT reach the rebuilt
        # graph (only the PHASIC_DYN_ORDERING env var did).
        'dyn_ordering': bool(self.dyn_ordering),
    }
    # Carry the compiled weight tape ONLY in formula mode, so the FFI
    # GraphBuilder that rebuilds this graph evaluates the formula in C.
    # Other modes' serialized output is byte-for-byte unchanged.
    if self._weight_mode == 'formula':
        tape = getattr(self, '_weight_formula_tape', None)
        if tape is None:
            raise ValueError(
                "weight_mode='formula' but no weight_formula is set; "
                "assign graph.weight_formula = '<expr>' first.")
        _serialized['weight_formula_tape'] = tape
    return _serialized


def from_serialized(cls, data: dict[str, Any]) -> Graph:
    """
    Reconstruct Graph from serialize() output.

    This method enables distributed trace recording by allowing graphs
    to be serialized to JSON, sent across the network via JAX pmap,
    and reconstructed on worker processes.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary returned by Graph.serialize() containing:
        - states: np.ndarray of shape (n_vertices, state_length)
        - edges: np.ndarray of shape (n_edges, 3) with [from_idx, to_idx, weight]
        - start_edges: np.ndarray of shape (n_start_edges, 2) with [to_idx, weight]
        - param_edges: np.ndarray of shape (n_param_edges, 2+param_length)
          Format (v0.22.0+): [from_idx, to_idx, coeff1, coeff2, ...]
        - start_param_edges: np.ndarray of shape (n_start_param_edges, 1+param_length)
          Format (v0.22.0+): [to_idx, coeff1, coeff2, ...]
        - param_length: int
        - state_length: int
        - n_vertices: int

    Returns
    -------
    Graph
        Reconstructed graph with identical structure to the original

    Raises
    ------
    ValueError
        If data is missing required fields, has wrong shapes, or malformed arrays
    RuntimeError
        If graph reconstruction fails (e.g., edges to non-existent vertices)

    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> from phasic import Graph
    >>>
    >>> # Create and serialize graph
    >>> g = Graph(1)
    >>> v0 = g.starting_vertex()
    >>> v1 = g.find_or_create_vertex([1])
    >>> v0.add_edge_parameterized(v1, 0.0, [2.0])
    >>> serialized = g.serialize(theta_dim=1)
    >>>
    >>> # Convert to JSON (for network transmission)
    >>> json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
    ...              for k, v in serialized.items()}
    >>> json_str = json.dumps(json_dict)
    >>>
    >>> # Reconstruct on worker (e.g., different machine)
    >>> received_dict = json.loads(json_str)
    >>> received_dict['states'] = np.array(received_dict['states'], dtype=np.int32)
    >>> received_dict['edges'] = np.array(received_dict['edges'], dtype=np.float64)
    >>> received_dict['start_edges'] = np.array(received_dict['start_edges'], dtype=np.float64)
    >>> received_dict['param_edges'] = np.array(received_dict['param_edges'], dtype=np.float64)
    >>> received_dict['start_param_edges'] = np.array(received_dict['start_param_edges'], dtype=np.float64)
    >>> g_reconstructed = Graph.from_serialized(received_dict)
    >>> assert g_reconstructed.vertices_length() == g.vertices_length()
    """
    import numpy as np

    # Validate required fields
    required = ['states', 'edges', 'start_edges', 'param_edges',
                'start_param_edges', 'param_length', 'state_length', 'n_vertices']
    missing = [f for f in required if f not in data]
    if missing:
        raise ValueError(
            f"Graph deserialization failed: missing required fields {missing}\n"
            f"  Available fields: {list(data.keys())}\n"
            f"  This usually indicates corrupted cache or version mismatch.\n"
            f"  Resolution: Clear cache with phasic.clear_caches() and rebuild graph"
        )

    # Extract and validate metadata
    try:
        n_vertices = int(data['n_vertices'])
        state_length = int(data['state_length'])
        param_length = int(data['param_length'])
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Graph deserialization failed: invalid metadata types\n"
            f"  n_vertices={data.get('n_vertices')!r}, "
            f"state_length={data.get('state_length')!r}, "
            f"param_length={data.get('param_length')!r}\n"
            f"  Error: {e}"
        ) from e

    # Validate and convert arrays
    try:
        states = np.asarray(data['states'], dtype=np.int32)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Graph deserialization failed: cannot convert 'states' to int32 array\n"
            f"  Type: {type(data['states'])}\n"
            f"  Error: {e}"
        ) from e

    if states.shape != (n_vertices, state_length):
        raise ValueError(
            f"Graph deserialization failed: states array shape mismatch\n"
            f"  Expected: ({n_vertices}, {state_length}) from metadata\n"
            f"  Actual: {states.shape} from 'states' field\n"
            f"  Resolution: This indicates corrupted data. Clear cache and rebuild."
        )

    try:
        vertex_indices = np.asarray(data['vertex_indices'], dtype=np.int32)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Graph deserialization failed: cannot convert 'vertex_indices' to int32 array\n"
            f"  Type: {type(data['vertex_indices'])}\n"
            f"  Error: {e}"
        ) from e

    if vertex_indices.shape != (n_vertices,):
        raise ValueError(
            f"Graph deserialization failed: vertex_indices array shape mismatch\n"
            f"  Expected: ({n_vertices},) from metadata\n"
            f"  Actual: {vertex_indices.shape} from 'vertex_indices' field\n"
            f"  Resolution: This indicates corrupted data. Clear cache and rebuild."
        )

    try:
        edges = np.asarray(data['edges'], dtype=np.float64)
        start_edges = np.asarray(data['start_edges'], dtype=np.float64)
        param_edges = np.asarray(data['param_edges'], dtype=np.float64)
        start_param_edges = np.asarray(data['start_param_edges'], dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Graph deserialization failed: cannot convert edge arrays to float64\n"
            f"  Error: {e}"
        ) from e

    # Validate edge array shapes
    # Handle empty arrays: they may be shape (0,) or (0, n_cols)
    if edges.ndim == 1:
        if edges.shape[0] != 0:
            raise ValueError(
                f"Graph deserialization failed: edges array has wrong shape\n"
                f"  Expected: (n_edges, 3) or empty (0,)\n"
                f"  Actual: {edges.shape}"
            )
        edges = edges.reshape((0, 3))
    elif edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError(
            f"Graph deserialization failed: edges array has wrong shape\n"
            f"  Expected: (n_edges, 3)\n"
            f"  Actual: {edges.shape}"
        )

    if start_edges.ndim == 1:
        if start_edges.shape[0] != 0:
            raise ValueError(
                f"Graph deserialization failed: start_edges array has wrong shape\n"
                f"  Expected: (n_start_edges, 2) or empty (0,)\n"
                f"  Actual: {start_edges.shape}"
            )
        start_edges = start_edges.reshape((0, 2))
    elif start_edges.ndim != 2 or start_edges.shape[1] != 2:
        raise ValueError(
            f"Graph deserialization failed: start_edges array has wrong shape\n"
            f"  Expected: (n_start_edges, 2)\n"
            f"  Actual: {start_edges.shape}"
        )

    # Parameterized edge format: [from_idx, to_idx, coeff1, coeff2, ...]
    # Format is now [from, to, c1, c2, ...] with 2+coefficients_length columns
    # The coefficient length may be >= param_length (edges can have extra coefficients)
    # Infer actual coefficient length from array shape
    if param_edges.ndim == 1:
        if param_edges.shape[0] != 0:
            raise ValueError(
                f"Graph deserialization failed: param_edges array has wrong shape\n"
                f"  Expected: (n_param_edges, 2+coeff_len) or empty (0,)\n"
                f"  Actual: {param_edges.shape}"
            )
        param_edges = param_edges.reshape((0, 0))
    elif param_edges.ndim == 2:
        # Validate minimum columns (at least from_idx, to_idx)
        if param_edges.shape[0] > 0 and param_edges.shape[1] < 2:
            raise ValueError(
                f"Graph deserialization failed: param_edges array has too few columns\n"
                f"  Expected: at least 2 columns (from_idx, to_idx)\n"
                f"  Actual: {param_edges.shape[1]} columns"
            )
        # Infer coefficient length from array shape
        if param_edges.shape[0] > 0:
            actual_coeff_len = param_edges.shape[1] - 2
            if actual_coeff_len < param_length:
                raise ValueError(
                    f"Graph deserialization failed: coefficient length < param_length\n"
                    f"  Coefficient length: {actual_coeff_len} (from array shape)\n"
                    f"  param_length: {param_length}\n"
                    f"  Edges must have at least param_length coefficients"
                )

    expected_start_param_edge_cols = 1 + param_length if param_length > 0 else 0
    if start_param_edges.ndim == 1:
        if start_param_edges.shape[0] != 0:
            raise ValueError(
                f"Graph deserialization failed: start_param_edges array has wrong shape\n"
                f"  Expected: (n_start_param_edges, {expected_start_param_edge_cols}) or empty (0,)\n"
                f"  Actual: {start_param_edges.shape}"
            )
        start_param_edges = start_param_edges.reshape((0, expected_start_param_edge_cols)) if expected_start_param_edge_cols > 0 else start_param_edges.reshape((0, 0))
    elif start_param_edges.ndim == 2:
        # Accept (0, 0) for param_length=0, or (n, 1+param_length) for param_length>0
        if param_length == 0:
            if start_param_edges.shape != (0, 0):
                raise ValueError(
                    f"Graph deserialization failed: start_param_edges array has wrong shape\n"
                    f"  Expected: (0, 0) when param_length=0\n"
                    f"  Actual: {start_param_edges.shape}"
                )
        elif start_param_edges.shape[1] != expected_start_param_edge_cols:
            raise ValueError(
                f"Graph deserialization failed: start_param_edges array has wrong shape\n"
                f"  Expected: (n_start_param_edges, {expected_start_param_edge_cols})\n"
                f"  Actual: {start_param_edges.shape}\n"
                f"  Note: theta_dim={theta_dim}, so columns should be 1+{theta_dim}={expected_start_param_edge_cols}\n"
                f"  Format: [to_idx, coeff1, coeff2, ...]"
            )

    # Create empty graph
    graph = cls(state_length)

    # Normally we let the first parameterized edge lock param_length, because
    # setting it early would make non-IPV CONSTANT edges require param_length
    # coefficients. But when the edges carry MORE coefficients than
    # param_length (decoupled weight_formula models, theta_dim < coefficient
    # length), letting the first edge lock it would silently raise param_length
    # to the coefficient length and DROP the intended theta dimension. In that
    # case pin it up front. A decoupled graph is necessarily fully
    # parameterized (the C layer rejects mixing constant + parameterized
    # non-IPV edges), so there are no non-IPV constant edges to conflict with.
    _coeff_len = (param_edges.shape[1] - 2) if (
        param_edges.ndim == 2 and param_edges.shape[0] > 0) else param_length
    if param_length > 0 and _coeff_len > param_length:
        graph.set_param_length(param_length)

    start = graph.starting_vertex()
    start_vertex_c_idx = start.index()  # Get C index of starting vertex

    # Create all vertices first
    # Note: The starting vertex may be included in the states array,
    # so we need to check for it and reuse it instead of creating a duplicate
    idx_to_vertex = {}
    for idx in range(n_vertices):
        state = states[idx].tolist()

        # Check if this is the starting vertex using C vertex index
        # Use vertex_indices array, not state comparison (states may be duplicated)
        if vertex_indices[idx] == start_vertex_c_idx:
            idx_to_vertex[idx] = start
        else:
            try:
                vertex = graph.find_or_create_vertex(state)
                idx_to_vertex[idx] = vertex
            except Exception as e:
                raise RuntimeError(
                    f"Graph deserialization failed: cannot create vertex {idx}\n"
                    f"  State: {state}\n"
                    f"  Error: {e}"
                ) from e

    # Add parameterized edges
    # Format (v0.22.0+): [from_idx, to_idx, coeff1, coeff2, ...]
    for edge_data in param_edges:
        from_idx = int(edge_data[0])
        to_idx = int(edge_data[1])
        edge_state = edge_data[2:].tolist()

        if from_idx < 0 or from_idx >= n_vertices:
            raise RuntimeError(
                f"Graph deserialization failed: parameterized edge has invalid from_idx\n"
                f"  from_idx={from_idx}, valid range=[0, {n_vertices})"
            )
        if to_idx < 0 or to_idx >= n_vertices:
            raise RuntimeError(
                f"Graph deserialization failed: parameterized edge has invalid to_idx\n"
                f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
            )

        try:
            from_vertex = idx_to_vertex[from_idx]
            to_vertex = idx_to_vertex[to_idx]
            from_vertex.add_edge(to_vertex, edge_state)
        except Exception as e:
            raise RuntimeError(
                f"Graph deserialization failed: cannot add parameterized edge\n"
                f"  From vertex {from_idx} (state={states[from_idx].tolist()})\n"
                f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                f"  Edge state (coefficients): {edge_state}\n"
                f"  Error: {e}"
            ) from e

    # Add starting vertex parameterized edges
    # Format (v0.22.0+): [to_idx, coeff1, coeff2, ...]
    for edge_data in start_param_edges:
        to_idx = int(edge_data[0])
        edge_state = edge_data[1:].tolist()

        if to_idx < 0 or to_idx >= n_vertices:
            raise RuntimeError(
                f"Graph deserialization failed: start parameterized edge has invalid to_idx\n"
                f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
            )

        try:
            to_vertex = idx_to_vertex[to_idx]
            start.add_edge(to_vertex, edge_state)
        except Exception as e:
            raise RuntimeError(
                f"Graph deserialization failed: cannot add start parameterized edge\n"
                f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                f"  Edge state (coefficients): {edge_state}\n"
                f"  Error: {e}"
            ) from e

    # Add regular edges (non-parameterized)
    for edge_data in edges:
        from_idx = int(edge_data[0])
        to_idx = int(edge_data[1])
        weight = float(edge_data[2])

        if from_idx < 0 or from_idx >= n_vertices:
            raise RuntimeError(
                f"Graph deserialization failed: edge has invalid from_idx\n"
                f"  from_idx={from_idx}, valid range=[0, {n_vertices})"
            )
        if to_idx < 0 or to_idx >= n_vertices:
            raise RuntimeError(
                f"Graph deserialization failed: edge has invalid to_idx\n"
                f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
            )

        try:
            from_vertex = idx_to_vertex[from_idx]
            to_vertex = idx_to_vertex[to_idx]
            from_vertex.add_edge(to_vertex, weight)
        except Exception as e:
            raise RuntimeError(
                f"Graph deserialization failed: cannot add edge\n"
                f"  From vertex {from_idx} (state={states[from_idx].tolist()})\n"
                f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                f"  Weight: {weight}\n"
                f"  Error: {e}"
            ) from e

    # Add starting vertex regular edges
    start = graph.starting_vertex()
    for edge_data in start_edges:
        to_idx = int(edge_data[0])
        weight = float(edge_data[1])

        if to_idx < 0 or to_idx >= n_vertices:
            raise RuntimeError(
                f"Graph deserialization failed: start edge has invalid to_idx\n"
                f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
            )

        try:
            to_vertex = idx_to_vertex[to_idx]
            start.add_edge(to_vertex, weight)
        except Exception as e:
            raise RuntimeError(
                f"Graph deserialization failed: cannot add start edge\n"
                f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                f"  Weight: {weight}\n"
                f"  Error: {e}"
            ) from e

    # Set param_length explicitly after all edges are added
    # This ensures graph.param_length() returns the correct value for trace recording
    if param_length > 0:
        try:
            graph.set_param_length(param_length)
        except Exception:
            pass  # May fail if no parameterized edges exist, which is fine

    # Restore the elimination-ordering flag (default False for
    # backward compatibility with caches serialized before this field
    # existed).
    graph.dyn_ordering = bool(data.get('dyn_ordering', False))

    # Restore the Python-side weight configuration. serialize() persists
    # weight_mode always and the compiled tape in formula mode; without
    # this the reconstructed graph silently reverts to 'linear' (a
    # formula/log graph then evaluates plain inner-product weights).
    graph._weight_mode = data.get('weight_mode', 'linear')
    tape = data.get('weight_formula_tape')
    if tape is not None:
        graph._weight_formula = tape.get('src')
        graph._weight_formula_tape = tape
        graph._set_weight_tape(
            list(tape['ops']), list(tape['consts']),
            int(tape['stack_depth']), int(tape['n_theta']),
            int(tape['n_coeff']))

    return graph
