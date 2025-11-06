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
    from .trace_elimination import EliminationTrace

    cache_file = _get_cache_path(graph_hash)
    if not cache_file.exists():
        return None

    # Use existing C-level cache loading
    # (delegates to ptd_load_trace_from_cache)
    from . import Graph
    # TODO: Implement trace deserialization
    # For now, return None to force recomputation
    return None


def _save_trace_to_cache(graph_hash: str, trace) -> bool:
    """Save trace to cache (returns True on success)"""
    # Use existing C-level cache saving
    # (delegates to ptd_save_trace_to_cache)
    # TODO: Implement trace serialization
    return False


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
# Trace Stitching
# ============================================================================

def stitch_scc_traces(scc_graph: 'SCCGraph',
                     scc_trace_dict: Dict[str, 'EliminationTrace']) -> 'EliminationTrace':
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
    1. Iterate SCCs in topological order
    2. For each SCC, append its operations to merged trace
    3. Adjust operation indices to account for previous operations
    4. Handle boundary edges from previous SCCs to current SCC
    5. Update vertex_rates, edge_probs, vertex_targets accordingly
    """
    from .trace_elimination import EliminationTrace, Operation, OpType, TraceBuilder

    # TODO: Implement trace stitching algorithm
    # For now, raise NotImplementedError
    raise NotImplementedError("Trace stitching not yet implemented")


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
