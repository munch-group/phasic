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
from .logging_config import get_logger

logger = get_logger(__name__)

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
    logger.debug("Starting SCC decomposition for graph with %d vertices",
                 graph.vertices_length())

    scc_decomp = graph.scc_decomposition()

    result = []
    scc_sizes = []
    for scc in scc_decomp.sccs_in_topo_order():
        # Extract as standalone graph
        scc_graph = scc.as_graph()
        scc_hash = scc.hash()
        scc_size = scc_graph.vertices_length()
        scc_sizes.append(scc_size)

        result.append((scc_hash, scc_graph))

    total_vertices = graph.vertices_length()
    logger.info("SCC decomposition: found %d components with sizes %s",
                len(result), scc_sizes)
    logger.debug("SCC component details:")
    for i, (hash_val, scc_g) in enumerate(result):
        size = scc_g.vertices_length()
        pct = (size / total_vertices) * 100
        logger.debug("  SCC %d: %d vertices (%.1f%% of graph), hash=%s...",
                     i, size, pct, hash_val[:16])

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

    logger.debug("Starting recursive trace collection: graph has %d vertices, min_size=%d",
                 graph.vertices_length(), min_size)

    work_units = {}  # hash -> serialized graph JSON
    cache_hits = []
    cache_misses = []
    scc_stats = {'total_sccs': 0, 'cached_sccs': 0, 'too_small': 0, 'subdivided': 0}

    def collect_recursive(g, depth=0):
        """Recursively collect missing traces"""
        indent = "  " * depth
        n_vertices = g.vertices_length()

        # Compute hash for this graph
        g_hash_result = g.content_hash()
        if g_hash_result is None:
            logger.warning("%sSkipping unhashable graph (%d vertices)", indent, n_vertices)
            return

        g_hash = g_hash_result
        logger.debug("%sChecking graph: %d vertices, hash=%s...", indent, n_vertices, g_hash[:16])

        # Check cache
        cached = _load_trace_from_cache(g_hash)
        if cached is not None:
            cache_hits.append((g_hash[:16], n_vertices))
            scc_stats['cached_sccs'] += 1
            logger.debug("%s✓ Cache hit for %d vertices", indent, n_vertices)
            return  # Cache hit

        cache_misses.append((g_hash[:16], n_vertices))
        logger.debug("%s✗ Cache miss for %d vertices", indent, n_vertices)

        # Check if too small to subdivide
        if n_vertices < min_size:
            scc_stats['too_small'] += 1
            # This is a work unit
            if g_hash not in work_units:
                # Serialize graph to JSON for cross-machine transport
                work_units[g_hash] = g.serialize()
                logger.debug("%s→ Added as work unit (%d vertices, below min_size)",
                           indent, n_vertices)
            else:
                logger.debug("%s→ Already in work units (deduplicated)", indent)
            return

        # Subdivide into SCCs and recurse
        logger.debug("%s→ Subdividing into SCCs (%d vertices >= min_size=%d)...",
                   indent, n_vertices, min_size)
        scc_decomp = g.scc_decomposition()
        sccs = list(scc_decomp.sccs_in_topo_order())
        scc_stats['total_sccs'] += len(sccs)
        scc_stats['subdivided'] += 1

        logger.debug("%s  Found %d SCCs", indent, len(sccs))
        for i, scc in enumerate(sccs):
            scc_graph = scc.as_graph()
            scc_vertices = scc_graph.vertices_length()
            logger.debug("%s  SCC %d/%d: %d vertices", indent, i+1, len(sccs), scc_vertices)
            collect_recursive(scc_graph, depth + 1)

    # Start recursive collection
    collect_recursive(graph)

    # Summary statistics
    total_cached_vertices = sum(v for _, v in cache_hits)
    total_missing_vertices = sum(v for _, v in cache_misses)
    total_vertices = graph.vertices_length()

    if total_vertices > 0:
        cached_pct = (total_cached_vertices / total_vertices) * 100
    else:
        cached_pct = 0

    logger.info("Trace collection complete: %d work units needed", len(work_units))
    logger.info("Cache statistics: %d hits, %d misses", len(cache_hits), len(cache_misses))
    logger.info("Cached vertices: %d/%d (%.1f%% of graph)",
                total_cached_vertices, total_vertices, cached_pct)
    logger.debug("SCC statistics: %s", scc_stats)

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

def _build_enhanced_scc_subgraph(
    original_graph: 'Graph',
    scc: 'SCCVertex',
    scc_graph: 'SCCGraph'
) -> 'Graph':
    """
    Build enhanced SCC subgraph that includes downstream connecting vertices.

    The enhanced subgraph contains:
    - All internal vertices of the SCC
    - Downstream connecting vertices (targets of boundary edges) as absorbing vertices
    - Boundary edges from internal vertices to connecting vertices

    This allows elimination to naturally handle boundary edges.

    Parameters
    ----------
    original_graph : Graph
        The original graph
    scc : SCCVertex
        The SCC to build subgraph for
    scc_graph : SCCGraph
        The SCC decomposition

    Returns
    -------
    Graph
        Enhanced subgraph
    """
    from . import Graph
    import numpy as np

    # Start with the standard SCC subgraph
    base_subgraph = scc.as_graph()

    # Get internal vertex indices
    internal_indices = scc.internal_vertex_indices()
    internal_states = {tuple(original_graph.vertex_at(i).state()) for i in internal_indices}

    # Find downstream connecting vertices (targets of boundary edges)
    connecting_vertices = {}  # orig_idx -> state
    boundary_edges = []  # (from_orig_idx, to_orig_idx, edge)

    for orig_v_idx in internal_indices:
        orig_vertex = original_graph.vertex_at(orig_v_idx)
        edges = orig_vertex.parameterized_edges() if original_graph.parameterized() else orig_vertex.edges()

        for edge in edges:
            target_vertex = edge.to()
            target_state = tuple(target_vertex.state())

            # Is this a boundary edge?
            if target_state not in internal_states:
                target_orig_idx = target_vertex.index()
                connecting_vertices[target_orig_idx] = target_vertex.state()
                boundary_edges.append((orig_v_idx, target_orig_idx, edge))

    # If no boundary edges, just return the base subgraph
    if not connecting_vertices:
        return base_subgraph

    # Build enhanced subgraph
    # Create new graph with same parameters
    enhanced = Graph(
        original_graph.state_length(),
        parameterized=original_graph.parameterized(),
        param_length=original_graph.param_length() if original_graph.parameterized() else 0
    )

    # Add internal vertices (copying from original graph)
    vertex_map = {}  # orig_idx -> new_vertex
    for orig_v_idx in internal_indices:
        orig_vertex = original_graph.vertex_at(orig_v_idx)
        state = orig_vertex.state()
        new_vertex = enhanced.find_or_create_vertex(state)
        vertex_map[orig_v_idx] = new_vertex

    # Add connecting vertices as absorbing vertices (no outgoing edges)
    for orig_v_idx, state in connecting_vertices.items():
        new_vertex = enhanced.find_or_create_vertex(state)
        vertex_map[orig_v_idx] = new_vertex

    # Add internal edges
    for orig_v_idx in internal_indices:
        orig_vertex = original_graph.vertex_at(orig_v_idx)
        edges = orig_vertex.parameterized_edges() if original_graph.parameterized() else orig_vertex.edges()

        for edge in edges:
            target_orig_idx = edge.to().index()
            target_state = tuple(edge.to().state())

            # Only add edges to internal vertices (not connecting vertices yet)
            if target_state in internal_states:
                from_vertex = vertex_map[orig_v_idx]
                to_vertex = vertex_map[target_orig_idx]

                if original_graph.parameterized():
                    coeffs = list(edge.edge_state(original_graph.param_length()))
                    from_vertex.add_edge_parameterized(to_vertex, 0.0, coeffs)
                else:
                    from_vertex.add_edge(to_vertex, edge.weight())

    # Add boundary edges to connecting vertices
    for from_orig_idx, to_orig_idx, edge in boundary_edges:
        from_vertex = vertex_map[from_orig_idx]
        to_vertex = vertex_map[to_orig_idx]

        if original_graph.parameterized():
            coeffs = list(edge.edge_state(original_graph.param_length()))
            from_vertex.add_edge_parameterized(to_vertex, 0.0, coeffs)
        else:
            from_vertex.add_edge(to_vertex, edge.weight())

    return enhanced


def _identify_trace_vertices(
    scc_graph: 'SCCGraph',
    scc_idx: int,
    scc_trace: 'EliminationTrace'
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Identify which trace vertices correspond to internal vs connecting vertices.

    Enhanced SCC subgraphs contain:
    - Internal vertices (belong to this SCC)
    - Connecting vertices (downstream neighbors, appear as absorbing in trace)

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition
    scc_idx : int
        Index of current SCC
    scc_trace : EliminationTrace
        Trace for this SCC (from enhanced subgraph)

    Returns
    -------
    internal_mapping : Dict[trace_v_idx, orig_v_idx]
        Maps trace vertices to original graph for internal vertices
    connecting_mapping : Dict[trace_v_idx, orig_v_idx]
        Maps trace vertices to original graph for connecting vertices
    """
    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()
    scc = sccs[scc_idx]

    internal_indices = set(scc.internal_vertex_indices())
    internal_mapping = {}
    connecting_mapping = {}

    # Map trace vertices by matching states
    for trace_v_idx in range(scc_trace.n_vertices):
        trace_state = tuple(scc_trace.states[trace_v_idx])

        # Find corresponding original vertex by state
        found = False
        for orig_v_idx in range(original_graph.vertices_length()):
            orig_vertex = original_graph.vertex_at(orig_v_idx)
            orig_state = tuple(orig_vertex.state())

            if orig_state == trace_state:
                # Check if this is internal or connecting
                if orig_v_idx in internal_indices:
                    internal_mapping[trace_v_idx] = orig_v_idx
                else:
                    connecting_mapping[trace_v_idx] = orig_v_idx
                found = True
                break

        if not found:
            # This might be the starting vertex proxy for non-SCC-0
            if scc_idx > 0 and trace_v_idx == 0:
                # Skip proxy starting vertex
                continue
            else:
                raise ValueError(f"Could not find original vertex for trace vertex {trace_v_idx} in SCC {scc_idx}")

    return internal_mapping, connecting_mapping


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
        # Remap operands, preserve coefficients as numpy array
        return Operation(
            op_type=OpType.DOT,
            coefficients=op.coefficients,  # Keep as numpy array
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


# ============================================================================
# Trace Stitching - Main Function
# ============================================================================

def record_enhanced_scc_traces(
    scc_graph: 'SCCGraph',
    param_length: int
) -> Dict[str, 'EliminationTrace']:
    """
    Record elimination traces for each SCC using enhanced subgraphs.

    Enhanced subgraphs include downstream connecting vertices as absorbing vertices,
    allowing boundary edges to be handled naturally during elimination.

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition
    param_length : int
        Number of parameters

    Returns
    -------
    Dict[str, EliminationTrace]
        Traces for each SCC, keyed by SCC hash
    """
    from .trace_elimination import record_elimination_trace

    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    scc_trace_dict = {}
    for scc in sccs:
        # Build enhanced subgraph with connecting vertices
        enhanced_subgraph = _build_enhanced_scc_subgraph(original_graph, scc, scc_graph)

        # Record trace for enhanced subgraph
        trace = record_elimination_trace(enhanced_subgraph, param_length)

        # Store with SCC hash
        scc_hash = scc.hash()
        scc_trace_dict[scc_hash] = trace

    return scc_trace_dict


def stitch_scc_traces(
    scc_graph: 'SCCGraph',
    scc_trace_dict: Dict[str, 'EliminationTrace']
) -> 'EliminationTrace':
    """
    Merge SCC traces in topological order.

    SCC traces are recorded using enhanced subgraphs that include downstream
    connecting vertices. When stitching, we merge operations for connecting
    vertices that appear in multiple traces.

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition with topological ordering
    scc_trace_dict : Dict[str, EliminationTrace]
        Traces for each SCC (recorded with enhanced subgraphs)

    Returns
    -------
    EliminationTrace
        Full graph trace stitched from SCC traces

    Notes
    -----
    - Processes SCCs in topological order (dependencies first)
    - Connecting vertices appear in both upstream and own SCC traces
    - Operations are merged for connecting vertices

    Algorithm:
    1. Initialize merged trace with all original vertices
    2. Process SCCs in topological order
    3. For each SCC:
       - Copy operations from enhanced SCC trace
       - For internal vertices: set vertex_rates, edge_probs, vertex_targets
       - Connecting vertices (from upstream SCCs) are handled automatically

    Raises
    ------
    ValueError
        If scc_trace_dict is empty or missing required traces
    """
    from .trace_elimination import EliminationTrace

    # Validate inputs
    if not scc_trace_dict:
        logger.error("Cannot stitch traces: scc_trace_dict is empty")
        raise ValueError("scc_trace_dict is empty")

    original_graph = scc_graph.original_graph()
    sccs = scc_graph.sccs_in_topo_order()

    if len(sccs) == 0:
        logger.error("Cannot stitch traces: no SCCs in graph")
        raise ValueError("Cannot stitch empty SCC graph")

    logger.debug("Starting trace stitching: %d SCCs, %d vertices total",
                 len(sccs), original_graph.vertices_length())
    logger.debug("SCC traces available: %d", len(scc_trace_dict))

    # Check all SCC traces present
    for scc in sccs:
        if scc.hash() not in scc_trace_dict:
            raise ValueError(f"Missing trace for SCC {scc.hash()}")

    # Step 1: Build mapping from original vertices to merged indices (identity mapping)
    original_to_merged = {}
    for orig_v_idx in range(original_graph.vertices_length()):
        original_to_merged[orig_v_idx] = orig_v_idx

    n_vertices_merged = original_graph.vertices_length()

    # Step 2: Initialize merged trace
    first_trace = next(iter(scc_trace_dict.values()))

    merged = EliminationTrace(
        operations=[],
        vertex_rates=np.full(n_vertices_merged, -1, dtype=np.int64),  # -1 = not set
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

    # Copy states from original graph
    for orig_v_idx in range(original_graph.vertices_length()):
        merged_v_idx = original_to_merged[orig_v_idx]
        merged.states[merged_v_idx] = original_graph.vertex_at(orig_v_idx).state()

    # Step 3: Process SCCs in topological order
    logger.debug("Processing %d SCCs in topological order...", len(sccs))
    op_remap = {}  # (scc_idx, scc_op_idx) -> merged_op_idx

    for scc_idx, scc in enumerate(sccs):
        scc_hash = scc.hash()
        scc_trace = scc_trace_dict[scc_hash]
        internal_indices = set(scc.internal_vertex_indices())

        n_internal = len(internal_indices)
        n_connecting = scc_trace.n_vertices - n_internal

        logger.debug("SCC %d/%d: hash=%s..., %d internal vertices, %d connecting vertices",
                     scc_idx + 1, len(sccs), scc_hash[:16], n_internal, n_connecting)
        logger.debug("  %d operations to merge", len(scc_trace.operations))

        # Append operations
        op_offset = len(merged.operations)
        logger.debug("  Operation offset: %d (merged trace has %d operations before this SCC)",
                     op_offset, op_offset)

        for scc_op_idx, operation in enumerate(scc_trace.operations):
            new_op = _remap_operation(operation, op_offset)
            merged.operations.append(new_op)
            op_remap[(scc_idx, scc_op_idx)] = op_offset + scc_op_idx

        logger.debug("  Copied %d operations (merged trace now has %d operations)",
                     len(scc_trace.operations), len(merged.operations))

        # Map trace vertices to original vertices by matching states
        trace_to_orig = {}
        for trace_v_idx in range(scc_trace.n_vertices):
            trace_state = tuple(scc_trace.states[trace_v_idx])
            for orig_v_idx in range(original_graph.vertices_length()):
                if tuple(original_graph.vertex_at(orig_v_idx).state()) == trace_state:
                    trace_to_orig[trace_v_idx] = orig_v_idx
                    break

        logger.debug("  Mapped %d trace vertices to original graph", len(trace_to_orig))

        # Copy data for INTERNAL vertices only (connecting vertices handled by their own SCC)
        n_vertices_set = 0
        n_edges_set = 0

        for trace_v_idx, orig_v_idx in trace_to_orig.items():
            if orig_v_idx not in internal_indices:
                continue  # Skip connecting vertices

            merged_v_idx = original_to_merged[orig_v_idx]

            # Set vertex_rates
            scc_rate_op_idx = scc_trace.vertex_rates[trace_v_idx]
            if scc_rate_op_idx >= 0 and (scc_idx, scc_rate_op_idx) in op_remap:
                merged.vertex_rates[merged_v_idx] = op_remap[(scc_idx, scc_rate_op_idx)]
            else:
                merged.vertex_rates[merged_v_idx] = scc_rate_op_idx

            n_vertices_set += 1

            # Set edge_probs and vertex_targets
            for j, scc_edge_op_idx in enumerate(scc_trace.edge_probs[trace_v_idx]):
                merged_edge_op_idx = op_remap[(scc_idx, scc_edge_op_idx)]
                merged.edge_probs[merged_v_idx].append(merged_edge_op_idx)

                scc_target_v_idx = scc_trace.vertex_targets[trace_v_idx][j]
                orig_target_v_idx = trace_to_orig[scc_target_v_idx]
                merged_target_v_idx = original_to_merged[orig_target_v_idx]
                merged.vertex_targets[merged_v_idx].append(merged_target_v_idx)

                n_edges_set += 1

        logger.debug("  Set vertex data for %d internal vertices, %d edges",
                     n_vertices_set, n_edges_set)
        logger.debug("SCC %d/%d merge complete", scc_idx + 1, len(sccs))

    # Set starting vertex
    merged.starting_vertex_idx = original_to_merged[original_graph.starting_vertex().index()]

    # Final summary statistics
    logger.info("Trace stitching complete: %d vertices, %d operations",
                merged.n_vertices, len(merged.operations))
    logger.debug("Final trace statistics:")
    logger.debug("  Vertices: %d", merged.n_vertices)
    logger.debug("  Operations: %d", len(merged.operations))
    logger.debug("  Parameters: %d", merged.param_length)
    logger.debug("  State length: %d", merged.state_length)
    logger.debug("  Starting vertex: %d", merged.starting_vertex_idx)

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
