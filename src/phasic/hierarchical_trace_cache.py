"""
Hierarchical SCC-Based Trace Caching

This module implements hierarchical trace caching using strongly connected
component (SCC) decomposition. Large graphs are broken into SCCs, traces
are computed in parallel, and results are stitched together.

Key Features:
- Hash-based deduplication of SCCs
- Parallel computation via vmap/pmap
- Two-level caching: full graph + individual SCCs
- Topological ordering for safe trace stitching

Author: Kasper Munch
Date: 2025-11-06
"""

import json
import hashlib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ============================================================================
# Cache Utilities
# ============================================================================

def _get_cache_path(graph_hash: str) -> Path:
    """Get cache file path for a graph hash"""
    cache_dir = Path.home() / ".phasic_cache" / "traces"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{graph_hash}.json"


def _load_trace_from_cache(graph_hash: str):
    """Load trace from cache (returns None if not found)"""
    from .trace_serialization import load_trace_from_cache
    return load_trace_from_cache(graph_hash)


def _save_trace_to_cache(graph_hash: str, trace) -> bool:
    """Save trace to cache (returns True on success)"""
    from .trace_serialization import save_trace_to_cache
    return save_trace_to_cache(graph_hash, trace)


# ============================================================================
# SCC Decomposition
# ============================================================================

def get_scc_graphs(graph, min_size: int = 50) -> List[Tuple[str, 'Graph']]:
    """
    Extract SCC subgraphs in topological order.

    Parameters
    ----------
    graph : Graph
        Input graph
    min_size : int
        Minimum vertices to subdivide (default 50)

    Returns
    -------
    List[Tuple[str, Graph]]
        List of (hash, scc_graph) pairs in topological order
    """
    scc_decomp = graph.scc_decomposition()

    result = []
    for scc in scc_decomp.sccs_in_topo_order():
        # Extract as standalone graph
        scc_graph = scc.as_graph()
        scc_hash = scc.hash()

        result.append((scc_hash, scc_graph))

    return result


# ============================================================================
# Work Collection (with deduplication)
# ============================================================================

def collect_missing_traces_batch(graph, param_length: Optional[int] = None,
                                 min_size: int = 50) -> Dict[str, str]:
    """
    Recursively collect ALL missing trace work units (deduplicated).

    This is the key improvement: collect everything first before computing.

    Parameters
    ----------
    graph : Graph
        Input graph
    param_length : int, optional
        Number of parameters
    min_size : int
        Minimum size to subdivide

    Returns
    -------
    Dict[str, str]
        Mapping: graph_hash -> serialized_graph_json
        Deduplicated by hash (same SCC across different graphs = one work unit)
    """
    from . import Graph

    work_units = {}  # hash -> serialized graph JSON

    def collect_recursive(g):
        """Recursively collect missing traces"""
        # Compute hash for this graph
        g_hash_result = g.content_hash()
        if g_hash_result is None:
            # Skip unhashable graphs
            return

        g_hash = g_hash_result

        # Check cache
        cached = _load_trace_from_cache(g_hash)
        if cached is not None:
            return  # Cache hit

        # Check if too small to subdivide
        if g.vertices_length() < min_size:
            # This is a work unit
            if g_hash not in work_units:
                # Serialize graph to JSON for cross-machine transport
                work_units[g_hash] = g.serialize()
            return

        # Subdivide into SCCs and recurse
        scc_decomp = g.scc_decomposition()
        for scc in scc_decomp.sccs_in_topo_order():
            scc_graph = scc.as_graph()
            collect_recursive(scc_graph)

    # Start recursive collection
    collect_recursive(graph)

    return work_units


# ============================================================================
# Parallel Trace Computation
# ============================================================================

def compute_trace_work_unit(hash_and_json: Tuple[str, str]) -> Tuple[str, 'EliminationTrace']:
    """
    Single work unit for vmap/pmap.

    Parameters
    ----------
    hash_and_json : Tuple[str, str]
        (graph_hash, serialized_graph_json)

    Returns
    -------
    Tuple[str, EliminationTrace]
        (hash, computed_trace)

    Notes
    -----
    - Checks cache again (race condition safety)
    - Deserializes graph from JSON
    - Computes trace via record_elimination_trace()
    - Caches result atomically
    """
    from .trace_elimination import record_elimination_trace
    from . import Graph

    graph_hash, graph_json = hash_and_json

    # Check cache again (another worker may have computed it)
    cached = _load_trace_from_cache(graph_hash)
    if cached is not None:
        return (graph_hash, cached)

    # Deserialize graph
    graph_dict = json.loads(graph_json)
    graph = Graph.deserialize(graph_dict)

    # Compute trace
    trace = record_elimination_trace(graph)

    # Cache result
    _save_trace_to_cache(graph_hash, trace)

    return (graph_hash, trace)


def compute_missing_traces_parallel(work_units: Dict[str, str],
                                   strategy: str = 'auto') -> Dict[str, 'EliminationTrace']:
    """
    Distribute work across CPUs/devices using vmap or pmap.

    Parameters
    ----------
    work_units : Dict[str, str]
        Mapping: graph_hash -> serialized_graph_json
    strategy : str, default='auto'
        Parallelization strategy:
        - 'auto': Use vmap for single machine, pmap for multi-device
        - 'vmap': Vectorize over batch (single machine, multi-CPU)
        - 'pmap': Parallelize over devices (multi-GPU or multi-machine)
        - 'sequential': No parallelization (debugging)

    Returns
    -------
    Dict[str, EliminationTrace]
        Mapping: hash -> computed_trace

    Notes
    -----
    Uses JAX vmap/pmap for automatic parallelization.
    Work units are automatically distributed across available CPUs/devices.
    """
    if not HAS_JAX:
        # Fallback to sequential
        strategy = 'sequential'

    if len(work_units) == 0:
        return {}

    # Convert to list for JAX
    work_list = list(work_units.items())

    # Auto-detect strategy
    if strategy == 'auto':
        n_devices = jax.device_count() if HAS_JAX else 1
        strategy = 'pmap' if n_devices > 1 else 'vmap'

    # ========================================================================
    # VMAP Strategy: Single machine, vectorize over batch
    # ========================================================================
    if strategy == 'vmap':
        # Vectorize over all work units
        # vmap automatically handles batch dimension
        compute_batch = jax.vmap(
            compute_trace_work_unit,
            in_axes=0,  # Vectorize over first dimension
            out_axes=0   # Output also batched
        )

        # Convert work_list to array
        work_array = jnp.array(work_list, dtype=object)

        # Compute all traces in parallel (CPU threads)
        results_array = compute_batch(work_array)

        # Convert back to dict
        return dict(results_array.tolist())

    # ========================================================================
    # PMAP Strategy: Multi-device or multi-machine
    # ========================================================================
    elif strategy == 'pmap':
        n_devices = jax.device_count()

        # Pad work to device count
        from .parallel_utils import _pad_to_devices, _shard_to_devices
        work_array = jnp.array(work_list, dtype=object)
        work_padded = _pad_to_devices(work_array, n_devices)
        work_sharded = _shard_to_devices(work_padded, n_devices)

        # Define per-device batch function
        def compute_device_batch(device_batch):
            """Compute batch of traces on one device"""
            return jax.vmap(compute_trace_work_unit)(device_batch)

        # Parallel map across devices
        results_sharded = jax.pmap(compute_device_batch)(work_sharded)

        # Flatten back to dict
        results_flat = results_sharded.reshape(-1, 2)[:len(work_list)]
        return dict(results_flat.tolist())

    # ========================================================================
    # SEQUENTIAL Strategy: No parallelization (debugging)
    # ========================================================================
    else:  # sequential
        results = {}
        for graph_hash, graph_json in work_list:
            _, trace = compute_trace_work_unit((graph_hash, graph_json))
            results[graph_hash] = trace
        return results


# ============================================================================
# Trace Stitching - Helper Functions
# ============================================================================

def _build_vertex_mappings(
    scc_graph: 'SCCGraph',
    scc_trace_dict: Dict[str, 'EliminationTrace']
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    """
    Build vertex mappings for stitching.

    With the modified SCC decomposition, SCC 0 always contains only the
    starting vertex, simplifying the mapping significantly.

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition (with starting vertex isolated in SCC 0)
    scc_trace_dict : Dict[str, EliminationTrace]
        Traces for each SCC

    Returns
    -------
    vertex_to_original : Dict[(scc_idx, scc_v_idx), orig_v_idx]
        Maps SCC trace vertex indices to original graph indices
    original_to_merged : Dict[orig_v_idx, merged_v_idx]
        Maps original graph indices to merged trace indices
    """
    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    vertex_to_original = {}
    original_to_merged = {}
    next_merged_idx = 0

    for scc_idx, scc in enumerate(sccs):
        scc_hash = scc.hash()
        scc_trace = scc_trace_dict[scc_hash]

        # Get original vertex indices for this SCC
        # These are the actual vertices from the original graph
        internal_indices = scc.internal_vertex_indices()

        # The SCC subgraph (created by as_graph()) has:
        # - Vertex 0: The subgraph's starting vertex
        # - Vertices 1, 2, 3, ...: The internal vertices

        for scc_v_idx in range(scc_trace.n_vertices):
            if scc_v_idx == 0:
                # Trace vertex 0 is the subgraph's starting vertex
                # For SCC 0, this is the original starting vertex
                # For other SCCs, this is a proxy - we'll map it to the first internal vertex
                if scc_idx == 0:
                    # SCC 0 contains only the starting vertex
                    orig_v_idx = internal_indices[0]
                else:
                    # For other SCCs, the subgraph starting vertex is a proxy
                    # It doesn't correspond to a real original vertex
                    # We'll map it to the first internal vertex as a placeholder
                    orig_v_idx = internal_indices[0] if len(internal_indices) > 0 else 0
            elif scc_v_idx - 1 < len(internal_indices):
                # Internal vertices: trace index 1, 2, 3, ... → internal_indices[0, 1, 2, ...]
                orig_v_idx = internal_indices[scc_v_idx - 1]
            else:
                raise ValueError(f"Trace has more vertices than expected for SCC {scc_idx}")

            # Record mapping
            vertex_to_original[(scc_idx, scc_v_idx)] = orig_v_idx

            # Assign merged index if not already assigned
            if orig_v_idx not in original_to_merged:
                original_to_merged[orig_v_idx] = next_merged_idx
                next_merged_idx += 1

    return vertex_to_original, original_to_merged


def _remap_operation(op: 'Operation', op_offset: int) -> 'Operation':
    """
    Remap operation indices by adding offset.

    Parameters
    ----------
    op : Operation
        Original operation from SCC trace
    op_offset : int
        Offset to add to operation indices

    Returns
    -------
    Operation
        Remapped operation for merged trace
    """
    from .trace_elimination import Operation, OpType

    if op.op_type == OpType.CONST:
        # No remapping needed
        return Operation(op_type=OpType.CONST, const_value=op.const_value)

    elif op.op_type == OpType.PARAM:
        # No remapping needed (references parameter array)
        return Operation(op_type=OpType.PARAM, param_idx=op.param_idx)

    elif op.op_type == OpType.DOT:
        # Remap operands, preserve coefficients
        return Operation(
            op_type=OpType.DOT,
            coefficients=list(op.coefficients),
            operands=[idx + op_offset for idx in op.operands]
        )

    elif op.op_type in [OpType.ADD, OpType.MUL, OpType.DIV]:
        # Remap both operands
        return Operation(
            op_type=op.op_type,
            operands=[idx + op_offset for idx in op.operands]
        )

    elif op.op_type == OpType.INV:
        # Remap single operand
        return Operation(
            op_type=OpType.INV,
            operands=[op.operands[0] + op_offset]
        )

    elif op.op_type == OpType.SUM:
        # Remap all operands
        return Operation(
            op_type=OpType.SUM,
            operands=[idx + op_offset for idx in op.operands]
        )

    else:
        raise ValueError(f"Unknown operation type: {op.op_type}")


def _add_boundary_edges(
    merged: 'EliminationTrace',
    scc_graph: 'SCCGraph',
    vertex_to_original: Dict[Tuple[int, int], int],
    original_to_merged: Dict[int, int]
) -> None:
    """
    Add boundary edges (edges crossing SCC boundaries).

    Modifies merged trace in-place.

    Parameters
    ----------
    merged : EliminationTrace
        Merged trace being constructed
    scc_graph : SCCGraph
        SCC decomposition
    vertex_to_original : Dict
        Maps (scc_idx, scc_v_idx) to original vertex indices
    original_to_merged : Dict
        Maps original vertex indices to merged vertex indices
    """
    from .trace_elimination import Operation, OpType

    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    # Build set of internal states for each SCC
    scc_vertex_sets = []
    for scc_idx, scc in enumerate(sccs):
        internal_states = set()
        scc_subgraph = scc.as_graph()
        for v_idx in range(scc_subgraph.vertices_length()):
            v = scc_subgraph.vertex_at(v_idx)
            internal_states.add(tuple(v.state()))
        scc_vertex_sets.append(internal_states)

    # Iterate through SCCs and find boundary edges
    for scc_idx, scc in enumerate(sccs):
        current_scc_states = scc_vertex_sets[scc_idx]

        # Check each vertex in this SCC
        for (map_scc_idx, _), orig_v_idx in vertex_to_original.items():
            if map_scc_idx != scc_idx:
                continue

            # Get original vertex
            orig_vertex = original_graph.vertex_at(orig_v_idx)
            merged_v_idx = original_to_merged[orig_v_idx]

            # Check each outgoing edge
            for edge_idx in range(orig_vertex.edges_length()):
                edge = orig_vertex.edge_at(edge_idx)
                target_state = tuple(edge.target.state())

                # Is this a boundary edge?
                if target_state not in current_scc_states:
                    # Get target merged index
                    target_orig_vertex = original_graph.find_vertex(edge.target.state())
                    target_orig_idx = target_orig_vertex.index()
                    target_merged_idx = original_to_merged[target_orig_idx]

                    # Create operation for edge weight
                    if edge.is_parameterized():
                        # Parameterized edge: DOT operation
                        coeffs = list(edge.edge_state)
                        param_indices = list(range(len(coeffs)))

                        edge_op = Operation(
                            op_type=OpType.DOT,
                            coefficients=np.array(coeffs),
                            operands=param_indices
                        )
                    else:
                        # Constant edge: CONST operation
                        edge_op = Operation(
                            op_type=OpType.CONST,
                            const_value=edge.weight
                        )

                    # Add operation and update trace
                    op_idx = len(merged.operations)
                    merged.operations.append(edge_op)
                    merged.edge_probs[merged_v_idx].append(op_idx)
                    merged.vertex_targets[merged_v_idx].append(target_merged_idx)


# ============================================================================
# Trace Stitching - Main Function
# ============================================================================

def stitch_scc_traces(
    scc_graph: 'SCCGraph',
    scc_trace_dict: Dict[str, 'EliminationTrace']
) -> 'EliminationTrace':
    """
    Merge SCC traces in topological order.

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition with topological ordering
    scc_trace_dict : Dict[str, EliminationTrace]
        Cached traces for each SCC (by hash)

    Returns
    -------
    EliminationTrace
        Full graph trace stitched from SCC traces

    Notes
    -----
    - Processes SCCs in topological order (dependencies first)
    - Handles boundary edges between SCCs
    - Adjusts operation indices during merge

    Algorithm:
    1. Build vertex mappings (SCC → original → merged)
    2. Initialize merged trace
    3. Process each SCC in topological order:
       a. Remap and append operations
       b. Copy and remap vertex data
    4. Add boundary edges
    5. Set starting vertex and return

    Raises
    ------
    ValueError
        If scc_trace_dict is empty or missing required traces
    """
    from .trace_elimination import EliminationTrace

    # Validate inputs
    if not scc_trace_dict:
        raise ValueError("scc_trace_dict is empty")

    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    if len(sccs) == 0:
        raise ValueError("Cannot stitch empty SCC graph")

    # Check all SCC traces present
    for scc in sccs:
        if scc.hash() not in scc_trace_dict:
            raise ValueError(f"Missing trace for SCC {scc.hash()}")

    # Get first trace for metadata
    first_trace = next(iter(scc_trace_dict.values()))

    # Step 1: Build vertex mappings
    vertex_to_original, original_to_merged = _build_vertex_mappings(
        scc_graph, scc_trace_dict
    )

    # Step 2: Initialize merged trace
    n_vertices_merged = len(original_to_merged)

    merged = EliminationTrace(
        operations=[],
        vertex_rates=np.zeros(n_vertices_merged, dtype=np.int64),
        edge_probs=[[] for _ in range(n_vertices_merged)],
        vertex_targets=[[] for _ in range(n_vertices_merged)],
        states=np.zeros(
            (n_vertices_merged, first_trace.state_length),
            dtype=first_trace.states.dtype
        ),
        starting_vertex_idx=0,  # Will be set later
        n_vertices=n_vertices_merged,
        state_length=first_trace.state_length,
        param_length=first_trace.param_length,
        reward_length=first_trace.reward_length,
        is_discrete=first_trace.is_discrete,
        metadata={}
    )

    # Step 3: Process each SCC in topological order
    op_remap = {}  # (scc_idx, scc_op_idx) -> merged_op_idx

    for scc_idx, scc in enumerate(sccs):
        scc_hash = scc.hash()
        scc_trace = scc_trace_dict[scc_hash]

        # 3a. Remap and append operations
        op_offset = len(merged.operations)

        for scc_op_idx, operation in enumerate(scc_trace.operations):
            # Remap operation indices
            new_op = _remap_operation(operation, op_offset)
            merged.operations.append(new_op)

            # Record mapping
            op_remap[(scc_idx, scc_op_idx)] = op_offset + scc_op_idx

        # 3b. Copy and remap vertex data
        for scc_v_idx in range(scc_trace.n_vertices):
            orig_v_idx = vertex_to_original[(scc_idx, scc_v_idx)]
            merged_v_idx = original_to_merged[orig_v_idx]

            # Copy state
            merged.states[merged_v_idx] = scc_trace.states[scc_v_idx]

            # Remap vertex_rates
            scc_rate_op_idx = scc_trace.vertex_rates[scc_v_idx]
            if (scc_idx, scc_rate_op_idx) in op_remap:
                merged.vertex_rates[merged_v_idx] = op_remap[(scc_idx, scc_rate_op_idx)]

            # Remap edge_probs and vertex_targets
            for j, scc_edge_op_idx in enumerate(scc_trace.edge_probs[scc_v_idx]):
                # Remap edge operation
                merged_edge_op_idx = op_remap[(scc_idx, scc_edge_op_idx)]
                merged.edge_probs[merged_v_idx].append(merged_edge_op_idx)

                # Remap target vertex
                scc_target_v_idx = scc_trace.vertex_targets[scc_v_idx][j]
                orig_target_v_idx = vertex_to_original[(scc_idx, scc_target_v_idx)]
                merged_target_v_idx = original_to_merged[orig_target_v_idx]
                merged.vertex_targets[merged_v_idx].append(merged_target_v_idx)

    # Step 4: Add boundary edges
    _add_boundary_edges(merged, scc_graph, vertex_to_original, original_to_merged)

    # Step 5: Set starting vertex
    starting_state = original_graph.starting_vertex().state()
    starting_orig_vertex = original_graph.find_vertex(starting_state)
    starting_orig_idx = starting_orig_vertex.index()
    merged.starting_vertex_idx = original_to_merged[starting_orig_idx]

    return merged


# ============================================================================
# Main Entry Point
# ============================================================================

def get_trace_hierarchical(graph,
                          param_length: Optional[int] = None,
                          min_size: int = 50,
                          parallel_strategy: str = 'auto') -> 'EliminationTrace':
    """
    Main entry point: Get trace with hierarchical caching.

    CURRENT IMPLEMENTATION (Phase 3a):
    - Simplified approach without trace stitching
    - Caches full graph traces only
    - Future: Will cache SCC-level traces for cross-graph reuse

    Workflow:
    1. Check cache for full graph → return if hit
    2. If not cached, compute trace directly for full graph
    3. Cache the result

    Parameters
    ----------
    graph : Graph
        Input graph (may be very large)
    param_length : int, optional
        Number of parameters
    min_size : int
        Minimum vertices to subdivide (currently ignored - no subdivision yet)
    parallel_strategy : str, default='auto'
        Parallelization strategy (currently ignored - no SCC parallelization yet)

    Returns
    -------
    EliminationTrace
        Complete elimination trace

    Examples
    --------
    >>> # Simple caching (no subdivision)
    >>> trace = get_trace_hierarchical(graph)
    >>>
    >>> # All parameters are accepted but min_size/parallel currently ignored
    >>> trace = get_trace_hierarchical(graph, hierarchical=True, min_size=50)

    Notes
    -----
    Phase 3a Limitations:
    - No SCC-level caching yet (coming in Phase 3b)
    - No trace stitching yet (complex algorithm, future work)
    - No parallel SCC computation yet
    - Still provides full-graph caching and clean API

    Future (Phase 3b):
    - SCC decomposition and caching
    - Trace stitching algorithm
    - Parallel SCC trace computation via vmap/pmap
    """
    from .trace_elimination import record_elimination_trace
    from . import hash as phasic_hash

    # Step 1: Try full graph hash cache
    try:
        hash_result = phasic_hash.compute_graph_hash(graph)
        graph_hash = hash_result.hash_hex
        trace = _load_trace_from_cache(graph_hash)
        if trace is not None:
            return trace  # Cache hit!
    except Exception:
        # Hash computation failed - proceed without caching
        graph_hash = None

    # Step 2: Compute trace directly (no subdivision in Phase 3a)
    # Future: Will decompose into SCCs and cache/stitch
    trace = record_elimination_trace(graph, param_length=param_length)

    # Step 3: Cache the full result
    if graph_hash is not None:
        _save_trace_to_cache(graph_hash, trace)

    return trace
