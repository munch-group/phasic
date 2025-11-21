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
import os
try:
    import multiprocess
    # Use spawn context to avoid fork() + JAX multithreading deadlock
    _mp_ctx = multiprocess.get_context('spawn')
    Pool = _mp_ctx.Pool
except ImportError:
    import multiprocessing
    # Use spawn context for consistency across platforms
    _mp_ctx = multiprocessing.get_context('spawn')
    Pool = _mp_ctx.Pool
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
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

    logger.debug("Cache query: hash=%s...", graph_hash[:16])
    trace = load_trace_from_cache(graph_hash)

    if trace is not None:
        logger.debug("  ✓ Cache HIT: hash=%s..., %d vertices, %d operations",
                     graph_hash[:16], trace.n_vertices, len(trace.operations))
    else:
        logger.debug("  ✗ Cache MISS: hash=%s...", graph_hash[:16])

    return trace


def _save_trace_to_cache(graph_hash: str, trace) -> bool:
    """Save trace to cache (returns True on success)"""
    from .trace_serialization import save_trace_to_cache

    logger.debug("Saving trace to cache: hash=%s..., %d vertices, %d operations",
                 graph_hash[:16], trace.n_vertices, len(trace.operations))

    success = save_trace_to_cache(graph_hash, trace)

    if success:
        logger.debug("  ✓ Cache save successful: hash=%s...", graph_hash[:16])
    else:
        logger.debug("  ✗ Cache save failed: hash=%s...", graph_hash[:16])

    return success


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

    logger.debug("Computing SCC decomposition...")
    scc_decomp = graph.scc_decomposition()
    logger.debug("SCC decomposition computed")

    result = []
    scc_sizes = []
    scc_list = list(scc_decomp.sccs_in_topo_order())
    logger.debug("Processing %d SCCs in topological order...", len(scc_list))

    for i, scc in enumerate(scc_list):
        logger.debug("  Extracting SCC %d/%d...", i + 1, len(scc_list))

        # Extract as standalone graph
        scc_graph = scc.as_graph()
        scc_hash = scc.hash()
        scc_size = scc_graph.vertices_length()
        scc_sizes.append(scc_size)

        logger.debug("    → SCC %d: %d vertices, hash=%s...",
                     i + 1, scc_size, scc_hash[:16])

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
                                 min_size: int = 50,
                                 verbose: bool = False) -> Dict[str, str]:
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
    Tuple[Dict[str, Graph], List[str], SCCGraph]
        - work_units: Mapping graph_hash -> graph_object (deduplicated)
        - all_scc_hashes: List of all top-level SCC hashes in topological order
        - top_level_scc_decomp: SCC decomposition of the top-level graph (for stitching)
    """
    from . import Graph

    n_vertices = graph.vertices_length()
    logger.debug("Collecting missing traces: graph has %d vertices, min_size=%d",
                 n_vertices, min_size)

    work_units = {}  # hash -> enhanced subgraph
    all_scc_hashes = []  # All SCC hashes in topological order
    cache_hits = []
    cache_misses = []

    # If graph too small, don't subdivide
    if n_vertices < min_size:
        logger.debug("Graph too small for subdivision (%d < %d), recording directly",
                    n_vertices, min_size)
        try:
            from . import hash as phasic_hash
            g_hash_result = phasic_hash.compute_graph_hash(graph)
            g_hash = g_hash_result.hash_hex
            work_units[g_hash] = graph
            all_scc_hashes = [g_hash]
            return work_units, all_scc_hashes, None
        except Exception as e:
            logger.warning("Failed to hash graph: %s", str(e))
            return {}, [], None

    # Decompose into SCCs
    logger.debug("Computing SCC decomposition...")
    scc_decomp = graph.scc_decomposition()
    sccs = list(scc_decomp.sccs_in_topo_order())
    top_level_scc_decomp = scc_decomp  # Store for stitching
    logger.debug("✓ Found %d SCCs", len(sccs))

    # Log SCC sizes
    scc_sizes = [scc.size() for scc in sccs]
    logger.debug("SCC sizes: %s", scc_sizes)

    # Group SCCs by size: large SCCs (≥ min_size) processed separately,
    # small SCCs (< min_size) should not be processed separately
    large_sccs = []  # SCCs to process separately
    small_sccs = []  # SCCs that are too small

    for i, scc in enumerate(sccs):
        scc_size = scc.size()
        if scc_size >= min_size:
            large_sccs.append((i, scc))
        else:
            small_sccs.append((i, scc))

    total_small_vertices = sum(scc.size() for _, scc in small_sccs)

    logger.info("SCC grouping: %d large SCCs (≥%d vertices), %d small SCCs (<%d vertices, %d total vertices)",
                len(large_sccs), min_size, len(small_sccs), min_size, total_small_vertices)

    # If ALL SCCs are small, fall back to recording full graph directly (no subdivision)
    if len(large_sccs) == 0:
        logger.info("All SCCs are below min_size=%d, recording full graph directly (no subdivision)", min_size)
        try:
            from . import hash as phasic_hash
            g_hash_result = phasic_hash.compute_graph_hash(graph)
            g_hash = g_hash_result.hash_hex

            # Serialize graph to JSON for distributed processing
            try:
                serialized_dict = graph.serialize(param_length=graph.param_length())
                # Convert numpy arrays to lists for JSON serialization
                json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in serialized_dict.items()}
                json_str = json.dumps(json_dict)
                logger.debug("Serialized full graph %s (%d bytes)", g_hash[:16], len(json_str))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to serialize full graph {g_hash[:16]}: {type(e).__name__}: {e}"
                ) from e

            work_units[g_hash] = json_str
            all_scc_hashes = [g_hash]
            return work_units, all_scc_hashes, None
        except Exception as e:
            logger.warning("Failed to hash graph: %s", str(e))
            return {}, [], None

    # If we have ONLY large SCCs (no small ones), proceed normally
    if len(small_sccs) == 0:
        logger.debug("All SCCs meet min_size threshold, processing all separately")

    # Process large SCCs separately (these get cached)
    # Add progress bar for large SCC collection
    if verbose:
        large_scc_iterator = tqdm(
            large_sccs,
            desc="Collecting work units",
            unit="SCC",
            leave=False
        )
    else:
        large_scc_iterator = large_sccs

    for orig_idx, scc in large_scc_iterator:
        scc_hash = scc.hash()
        scc_size = scc.size()
        all_scc_hashes.append(scc_hash)

        logger.debug("Processing LARGE SCC %d: %d vertices, hash=%s...",
                    orig_idx, scc_size, scc_hash[:16])

        # Check cache
        logger.debug("Cache query: hash=%s...", scc_hash[:16])
        cached = _load_trace_from_cache(scc_hash)

        if cached is not None:
            logger.debug("✓ Cache HIT: hash=%s..., %d vertices, %d operations",
                        scc_hash[:16], cached.n_vertices, len(cached.operations))
            cache_hits.append((scc_hash[:16], scc_size))
            logger.debug("✓ Cache hit for %d vertices", scc_size)
            continue  # Cache hit - no work needed

        cache_misses.append((scc_hash[:16], scc_size))
        logger.debug("✗ Cache miss for %d vertices", scc_size)

        # Build subgraph (first SCC vs other SCCs)
        if orig_idx == 0:
            # First SCC - contains starting vertex
            logger.debug("Building FIRST SCC subgraph (starting vertex)...")
            enhanced_subgraph, metadata = _build_first_scc_subgraph(
                graph, graph.starting_vertex().index(), scc, scc_decomp
            )
        else:
            # Other SCCs - have upstream vertices
            logger.debug("Building SCC subgraph with upstream/downstream vertices...")
            enhanced_subgraph, metadata = _build_scc_subgraph(graph, scc, scc_decomp)

        enhanced_size = enhanced_subgraph.vertices_length()
        logger.debug("✓ Subgraph built: %d vertices (%d internal + %d connecting)",
                    enhanced_size, scc_size, enhanced_size - scc_size)

        # Store metadata with subgraph for later retrieval
        # We'll store it in a dict keyed by SCC hash
        if not hasattr(collect_missing_traces_batch, '_metadata_cache'):
            collect_missing_traces_batch._metadata_cache = {}
        collect_missing_traces_batch._metadata_cache[scc_hash] = metadata

        # Serialize graph to JSON for distributed processing
        try:
            serialized_dict = enhanced_subgraph.serialize(param_length=graph.param_length())
            # Convert numpy arrays to lists for JSON serialization
            json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in serialized_dict.items()}
            json_str = json.dumps(json_dict)
            logger.debug("Serialized SCC %s (%d bytes)", scc_hash[:16], len(json_str))
        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize SCC {scc_hash[:16]}: {type(e).__name__}: {e}"
            ) from e

        # Add as work unit (keyed by SCC hash, not enhanced subgraph hash)
        work_units[scc_hash] = json_str
        logger.debug("→ Added as work unit")

        logger.debug("✓ Completed large SCC %d", orig_idx)

    # If we have a mix of large and small SCCs, warn the user
    if len(small_sccs) > 0 and len(large_sccs) > 0:
        logger.warning("Graph has %d small SCCs (<%d vertices) that will be included in large SCC subgraphs",
                      len(small_sccs), min_size)
        logger.warning("This may reduce cache reuse. Consider increasing min_size to avoid subdivision.")

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
    logger.debug("Collected %d top-level SCC hashes", len(all_scc_hashes))

    return work_units, all_scc_hashes, top_level_scc_decomp


# ============================================================================
# Parallel Trace Computation with JAX vmap/pmap
# ============================================================================

# Global work unit storage for JAX compatibility
# Maps integer index -> (hash, json_string)
# JAX can vmap over integer indices, but not Python objects
_work_unit_store: Dict[int, Tuple[str, str]] = {}


def _record_trace_callback(idx: int) -> Tuple[str, 'EliminationTrace']:
    """
    Pure Python callback for trace recording (called from JAX via pure_callback).

    This function is wrapped by JAX's pure_callback mechanism to enable vmap/pmap
    while performing Python/C++ operations (deserialization, trace recording, caching).

    Parameters
    ----------
    idx : int
        Index into _work_unit_store, or -1 for padding

    Returns
    -------
    Tuple[str, EliminationTrace]
        (graph_hash, computed_trace) or ("", None) for padding

    Raises
    ------
    RuntimeError
        If deserialization or trace computation fails

    Notes
    -----
    For vmap: Global dict works for both read and write (single process).
    For pmap: Global dict works for read only (copied to each worker process).
              Results are saved to disk cache, not written back to global dict.
    """
    from .trace_elimination import record_elimination_trace
    from phasic import Graph

    # Handle padding indices (used by pmap)
    idx_int = int(idx)
    if idx_int < 0:
        return ("", None)  # Padding sentinel

    # Load work unit from global store
    # Note: pmap workers get a COPY of the global dict at initialization
    graph_hash, json_str = _work_unit_store[idx_int]

    # Check cache again (another worker may have computed it)
    cached = _load_trace_from_cache(graph_hash)
    if cached is not None:
        logger.debug("Cache hit for %s during parallel computation", graph_hash[:16])
        return (graph_hash, cached)

    # Deserialize JSON to Graph
    try:
        graph_dict = json.loads(json_str)
        # Convert lists back to numpy arrays
        graph_dict['states'] = np.array(graph_dict['states'], dtype=np.int32)
        graph_dict['edges'] = np.array(graph_dict['edges'], dtype=np.float64)
        graph_dict['start_edges'] = np.array(graph_dict['start_edges'], dtype=np.float64)
        graph_dict['param_edges'] = np.array(graph_dict['param_edges'], dtype=np.float64)
        graph_dict['start_param_edges'] = np.array(graph_dict['start_param_edges'], dtype=np.float64)

        graph = Graph.from_serialized(graph_dict)
        logger.debug("Deserialized graph %s: %d vertices, param_length=%d",
                    graph_hash[:16], graph.vertices_length(), graph.param_length())
    except Exception as e:
        raise RuntimeError(
            f"Worker failed to deserialize graph {graph_hash[:16]}: {type(e).__name__}: {e}"
        ) from e

    # Compute trace (param_length auto-detected from graph)
    try:
        trace = record_elimination_trace(graph, param_length=None)
        logger.debug("Recorded trace for %s: %d operations", graph_hash[:16], len(trace.operations))
    except Exception as e:
        raise RuntimeError(
            f"Worker failed to record trace for {graph_hash[:16]}: {type(e).__name__}: {e}"
        ) from e

    # Cache result
    _save_trace_to_cache(graph_hash, trace)

    return (graph_hash, trace)


# ============================================================================
# PMAP Disk-Based Work Unit Distribution
# ============================================================================

# Per-process cache for loaded work units (avoids repeated disk reads)
_pmap_file_cache: Dict[str, Tuple[str, str]] = {}


def _get_pmap_shared_dir() -> Path:
    """
    Get shared directory for pmap work units.

    Returns
    -------
    Path
        Directory path (defaults to ~/.phasic_cache/pmap_work or $PHASIC_PMAP_SHARED_DIR)
    """
    import os
    env_dir = os.environ.get('PHASIC_PMAP_SHARED_DIR')
    if env_dir:
        shared_dir = Path(env_dir)
    else:
        shared_dir = Path.home() / ".phasic_cache" / "pmap_work"

    shared_dir.mkdir(parents=True, exist_ok=True)
    return shared_dir


def _write_work_unit_to_file(work_dir: Path, idx: int, graph_hash: str, json_str: str) -> Path:
    """
    Write work unit to disk file.

    Parameters
    ----------
    work_dir : Path
        Session work directory
    idx : int
        Work unit index
    graph_hash : str
        Graph hash
    json_str : str
        Serialized graph JSON

    Returns
    -------
    Path
        Path to written file
    """
    file_path = work_dir / f"work_{idx:06d}.json"
    with open(file_path, 'w') as f:
        json.dump({'graph_hash': graph_hash, 'graph_json': json_str}, f)
    return file_path


def _load_work_unit_from_file(file_path: str) -> Tuple[str, str]:
    """
    Load work unit from disk file with per-process caching.

    Parameters
    ----------
    file_path : str
        Path to work unit file

    Returns
    -------
    tuple[str, str]
        (graph_hash, json_str)

    Raises
    ------
    FileNotFoundError
        If work unit file doesn't exist
    """
    from pathlib import Path

    # Check per-process cache first
    if file_path in _pmap_file_cache:
        return _pmap_file_cache[file_path]

    # Validate file exists
    file_obj = Path(file_path)
    if not file_obj.exists():
        raise FileNotFoundError(
            f"Work unit file not found: {file_path}\n"
            f"  Directory exists: {file_obj.parent.exists()}\n"
            f"  Parent contents: {list(file_obj.parent.iterdir()) if file_obj.parent.exists() else 'N/A'}"
        )

    # Load from disk
    with open(file_path, 'r') as f:
        data = json.load(f)

    result = (data['graph_hash'], data['graph_json'])

    # Cache in this process
    _pmap_file_cache[file_path] = result

    return result


def compute_missing_traces_parallel(work_units: Dict[str, str],
                                   strategy: str = 'auto',
                                   min_size: int = 50,
                                   verbose: bool = False,
                                   n_workers: Optional[int] = None) -> Dict[str, 'EliminationTrace']:
    """
    Distribute work across CPUs/devices using multiprocessing or sequential.

    Parameters
    ----------
    work_units : Dict[str, str]
        Mapping: graph_hash -> json_string (serialized graph)
    strategy : str, default='auto'
        Parallelization strategy:
        - 'auto': Use 'vmap' (multiprocessing) if available, else 'sequential'
        - 'vmap': Use multiprocessing.Pool for CPU parallelization
        - 'sequential': No parallelization (debugging)
        - 'pmap': NOT SUPPORTED (requires JIT compilation)
    min_size : int, default=50
        Minimum vertices to subdivide (for recursive subdivision)
    n_workers : int, optional
        Number of worker processes for multiprocessing (vmap strategy).
        Default: os.cpu_count() (use all available cores)
        Set to 1 for sequential execution within vmap

    Returns
    -------
    Dict[str, EliminationTrace]
        Mapping: hash -> computed_trace

    Raises
    ------
    ValueError
        If strategy is invalid
    ImportError
        If JAX is not installed but vmap/pmap requested
    RuntimeError
        If pmap requested but <2 devices available

    Notes
    -----
    Graphs are serialized to JSON for transmission via vmap/pmap.
    Each worker deserializes independently and caches to local disk.
    """
    if len(work_units) == 0:
        return {}

    # Validate parallelization strategy
    if strategy not in ('auto', 'vmap', 'pmap', 'sequential'):
        raise ValueError(
            f"Invalid parallelization strategy: '{strategy}'\n"
            f"  Valid options: 'auto', 'vmap', 'pmap', 'sequential'\n"
            f"  Got: {strategy}"
        )

    # Auto-detect with explicit logging
    if strategy == 'auto':
        # Always use vmap (multiprocessing) for CPU parallelization
        strategy = 'vmap'
        n_cpus = os.cpu_count() or 1
        logger.info("Parallelization: Auto-selected 'vmap' (multiprocessing with %d CPUs)", n_cpus)

    # Validate JAX available for vmap/pmap
    if strategy == 'vmap' and not HAS_JAX:
        raise ImportError(
            "Cannot use strategy='vmap': JAX not installed\n"
            "  Install JAX: pip install jax jaxlib\n"
            "  Or use: strategy='sequential' or strategy='auto'"
        )

    if strategy == 'pmap':
        raise ValueError(
            "strategy='pmap' is not supported.\n"
            "  pmap requires JIT-compiled code for distribution.\n"
            "  pure_callback cannot be JIT compiled, making pmap non-functional.\n"
            "  \n"
            "  Use 'vmap' for CPU parallelization via multiprocessing.\n"
            "  Use 'sequential' for debugging.\n"
            "  Use 'auto' for automatic selection."
        )

    # Populate global work unit store and create index mapping
    _work_unit_store.clear()
    hash_to_idx = {}
    for idx, (graph_hash, json_str) in enumerate(work_units.items()):
        _work_unit_store[idx] = (graph_hash, json_str)
        hash_to_idx[graph_hash] = idx

    # Create JAX-compatible indices array
    import jax.numpy as jnp
    indices = jnp.arange(len(work_units))

    # ========================================================================
    # VMAP Strategy: Single machine, multiprocessing with expand_dims
    # ========================================================================
    if strategy == 'vmap':
        # Determine number of workers

        if n_workers is None:
            n_workers = os.cpu_count() or 1

        # If running in SLURM, respect allocated CPUs
        n_workers = os.environ.get('SLURM_JOB_CPUS_PER_NODE', n_workers)

        n_workers = max(1, min(n_workers, len(work_units)))  # Limit to work count

        logger.info("VMAP: Using multiprocessing with %d workers over %d work units",
                    n_workers, len(work_units))

        # Show progress
        if verbose:
            pbar = tqdm(
                total=0,
                desc=f"Computing {len(work_units)} traces ({n_workers} workers)",
                bar_format="{desc}: {elapsed}",
                leave=False
            )

        # Define batched computation with multiprocessing
        def _compute_traces_batch_jax(indices):
            """Compute traces in parallel using multiprocessing.Pool"""
            result_shape = jax.ShapeDtypeStruct(indices.shape, jnp.int32)

            def _callback_batch_impl(indices_array):
                """Process batch in parallel with multiprocessing"""
                # Convert to list for processing
                indices_list = indices_array.tolist()

                if n_workers == 1:
                    # Sequential execution (no multiprocessing overhead)
                    trace_results = []
                    for idx in indices_list:
                        if idx >= 0:
                            # Process directly in main process
                            graph_hash, json_str = _work_unit_store[idx]
                            # Deserialize graph from JSON
                            graph_dict = json.loads(json_str)
                            graph_dict['states'] = np.array(graph_dict['states'], dtype=np.int32)
                            graph_dict['edges'] = np.array(graph_dict['edges'], dtype=np.float64)
                            graph_dict['start_edges'] = np.array(graph_dict['start_edges'], dtype=np.float64)
                            graph_dict['param_edges'] = np.array(graph_dict['param_edges'], dtype=np.float64)
                            graph_dict['start_param_edges'] = np.array(graph_dict['start_param_edges'], dtype=np.float64)
                            from phasic import Graph
                            graph = Graph.from_serialized(graph_dict)
                            # Compute trace
                            from .trace_elimination import record_elimination_trace
                            trace = record_elimination_trace(graph, param_length=None)
                            # Save to disk cache
                            _save_trace_to_cache(graph_hash, trace)
                            trace_results.append((idx, (graph_hash, trace)))
                        else:
                            trace_results.append((idx, ("", None)))
                else:
                    # Parallel execution with Pool
                    # Define worker function that receives data explicitly
                    def _compute_trace_worker(work_item):
                        """Worker function: receives (idx, graph_hash, json_str)"""
                        import json
                        import numpy as np
                        from phasic import Graph
                        from phasic.trace_elimination import record_elimination_trace

                        idx, graph_hash, json_str = work_item
                        # Deserialize graph from JSON
                        graph_dict = json.loads(json_str)
                        graph_dict['states'] = np.array(graph_dict['states'], dtype=np.int32)
                        graph_dict['edges'] = np.array(graph_dict['edges'], dtype=np.float64)
                        graph_dict['start_edges'] = np.array(graph_dict['start_edges'], dtype=np.float64)
                        graph_dict['param_edges'] = np.array(graph_dict['param_edges'], dtype=np.float64)
                        graph_dict['start_param_edges'] = np.array(graph_dict['start_param_edges'], dtype=np.float64)
                        graph = Graph.from_serialized(graph_dict)
                        # Compute trace
                        trace = record_elimination_trace(graph, param_length=None)
                        # Save to disk cache (so parent can load it)
                        from phasic.hierarchical_trace_cache import _save_trace_to_cache
                        _save_trace_to_cache(graph_hash, trace)
                        return (idx, (graph_hash, trace))

                    # Prepare work items: (idx, graph_hash, json_str)
                    work_items = []
                    for idx in indices_list:
                        if idx >= 0:
                            graph_hash, json_str = _work_unit_store[idx]
                            work_items.append((idx, graph_hash, json_str))

                    if len(work_items) > 0:
                        # Process in parallel - data passed explicitly
                        with Pool(processes=n_workers) as pool:
                            parallel_results = pool.map(_compute_trace_worker, work_items)

                        # Convert to dict for easy lookup
                        results_dict = dict(parallel_results)

                        # Reconstruct results with padding
                        trace_results = []
                        for idx in indices_list:
                            if idx >= 0:
                                trace_results.append((idx, results_dict[idx]))
                            else:
                                trace_results.append((idx, ("", None)))
                    else:
                        # All padding
                        trace_results = [(idx, ("", None)) for idx in indices_list]

                # Store results in main process
                for idx, (graph_hash, trace) in trace_results:
                    if idx >= 0 and graph_hash:  # Skip padding
                        _work_unit_store[idx] = (graph_hash, trace)
                        logger.debug("Stored trace for hash=%s, idx=%d", graph_hash[:16], idx)

                # Return as int32 (JAX expects this dtype)
                return np.array(indices_array, dtype=np.int32)

            return jax.pure_callback(
                _callback_batch_impl,
                result_shape,
                indices,
                vmap_method='expand_dims'  # Process entire batch at once
            )

        # Process entire batch (no vmap needed!)
        completed_indices = _compute_traces_batch_jax(indices)

        if verbose:
            pbar.close()

        # Collect results from work unit store
        results = {}
        for idx in completed_indices:
            idx_int = int(idx)
            if idx_int >= 0:
                graph_hash, trace = _work_unit_store[idx_int]
                if graph_hash:  # Skip padding sentinel
                    results[graph_hash] = trace
                    logger.debug("Collected trace for %s", graph_hash[:16])

        return results

    # ========================================================================
    # SEQUENTIAL Strategy: No parallelization (debugging)
    # ========================================================================
    elif strategy == 'sequential':
        from .trace_elimination import record_elimination_trace
        from phasic import Graph

        results = {}

        # Add progress bar for sequential processing
        if verbose:
            work_iterator = tqdm(
                work_units.items(),
                desc="Computing traces (sequential)",
                unit="trace",
                leave=False
            )
        else:
            work_iterator = work_units.items()

        for graph_hash, json_str in work_iterator:
            # Check cache again (race condition safety)
            cached = _load_trace_from_cache(graph_hash)
            if cached is not None:
                results[graph_hash] = cached
                continue

            # Deserialize JSON to Graph
            try:
                graph_dict = json.loads(json_str)
                # Convert lists back to numpy arrays
                graph_dict['states'] = np.array(graph_dict['states'], dtype=np.int32)
                graph_dict['edges'] = np.array(graph_dict['edges'], dtype=np.float64)
                graph_dict['start_edges'] = np.array(graph_dict['start_edges'], dtype=np.float64)
                graph_dict['param_edges'] = np.array(graph_dict['param_edges'], dtype=np.float64)
                graph_dict['start_param_edges'] = np.array(graph_dict['start_param_edges'], dtype=np.float64)

                graph = Graph.from_serialized(graph_dict)
                logger.debug("Sequential: Deserialized SCC %s (%d vertices)",
                           graph_hash[:16], graph.vertices_length())
            except Exception as e:
                raise RuntimeError(
                    f"Sequential: Failed to deserialize {graph_hash[:16]}: {type(e).__name__}: {e}"
                ) from e

            # Record trace directly (no recursive subdivision needed)
            # Hierarchical decomposition already happened in collect_missing_traces_batch()
            # Workers should just compute traces atomically to avoid unnecessary overhead
            logger.debug(f"  Recording trace for {graph.vertices_length()} vertex subgraph")
            try:
                # Use graph's param_length explicitly instead of auto-detection (which is broken)
                trace = record_elimination_trace(graph, param_length=graph.param_length())
                logger.debug(f"  Recorded trace: {len(trace.operations)} operations, param_length={trace.param_length}")
            except Exception as e:
                raise RuntimeError(
                    f"Sequential: Failed to record trace for {graph_hash[:16]}: {type(e).__name__}: {e}"
                ) from e

            # Cache the result
            _save_trace_to_cache(graph_hash, trace)
            results[graph_hash] = trace

            # NOTE: Do NOT explicitly delete graph here!
            # The graph might be the original input (borrowed reference) or an enhanced subgraph.
            # If it's the original, deleting it causes double-free when caller tries to delete.
            # If it's a subgraph, Python GC will clean it up when work_units goes out of scope.
            # Explicit deletion was causing crashes with clone() + compute_trace()

        return results


# ============================================================================
# Trace Stitching - Helper Functions
# ============================================================================

def _find_upstream_vertices(
    original_graph: 'Graph',
    internal_indices: List[int],
    scc_graph: 'SCCGraph'
) -> List[int]:
    """
    Find vertices in upstream SCCs that connect to the current SCC.

    These are vertices that have edges TO the current SCC's internal vertices.
    They will be treated as "fake starting vertices" in the enhanced subgraph.

    Parameters
    ----------
    original_graph : Graph
        The original graph
    internal_indices : List[int]
        Internal vertex indices of the current SCC
    scc_graph : SCCGraph
        The SCC decomposition

    Returns
    -------
    List[int]
        List of original vertex indices in upstream SCCs
    """
    internal_set = set(internal_indices)
    upstream_vertices = set()

    # For each vertex in the original graph
    for v_idx in range(original_graph.vertices_length()):
        # Skip internal vertices
        if v_idx in internal_set:
            continue

        vertex = original_graph.vertex_at(v_idx)
        edges = vertex.parameterized_edges() if original_graph.parameterized() else vertex.edges()

        # Check if this vertex has edges TO any internal vertex
        for edge in edges:
            target_idx = edge.to().index()
            if target_idx in internal_set:
                # This is an upstream vertex
                upstream_vertices.add(v_idx)
                break

    return sorted(list(upstream_vertices))


def _find_upstream_connecting(
    internal_indices: List[int],
    upstream_vertices: List[int],
    original_graph: 'Graph'
) -> List[int]:
    """
    Find internal vertices that receive edges from upstream vertices.

    These are the "upstream-connecting" vertices in the enhanced subgraph ordering.

    Parameters
    ----------
    internal_indices : List[int]
        Internal vertex indices of the current SCC
    upstream_vertices : List[int]
        Upstream vertex indices
    original_graph : Graph
        The original graph

    Returns
    -------
    List[int]
        List of internal vertex indices that connect to upstream
    """
    internal_set = set(internal_indices)
    upstream_set = set(upstream_vertices)
    upstream_connecting = set()

    # Check which internal vertices receive from upstream
    for up_idx in upstream_vertices:
        up_vertex = original_graph.vertex_at(up_idx)
        edges = up_vertex.parameterized_edges() if original_graph.parameterized() else up_vertex.edges()

        for edge in edges:
            target_idx = edge.to().index()
            if target_idx in internal_set:
                upstream_connecting.add(target_idx)

    return sorted(list(upstream_connecting))


def _find_downstream_connecting(
    internal_indices: List[int],
    original_graph: 'Graph'
) -> List[int]:
    """
    Find internal vertices that have edges to vertices outside the SCC.

    These are the "downstream-connecting" vertices in the enhanced subgraph ordering.

    Parameters
    ----------
    internal_indices : List[int]
        Internal vertex indices of the current SCC
    original_graph : Graph
        The original graph

    Returns
    -------
    List[int]
        List of internal vertex indices that connect to downstream
    """
    internal_set = set(internal_indices)
    downstream_connecting = set()

    for v_idx in internal_indices:
        vertex = original_graph.vertex_at(v_idx)
        edges = vertex.parameterized_edges() if original_graph.parameterized() else vertex.edges()

        for edge in edges:
            target_idx = edge.to().index()
            if target_idx not in internal_set:
                # This internal vertex has an edge to outside the SCC
                downstream_connecting.add(v_idx)
                break

    return sorted(list(downstream_connecting))


def _find_downstream_vertices(
    original_graph: 'Graph',
    downstream_connecting: List[int],
    internal_indices: List[int]
) -> List[int]:
    """
    Find vertices that receive edges from downstream-connecting vertices.

    These are vertices outside the SCC that will be treated as "fake absorbing"
    vertices in the enhanced subgraph.

    Parameters
    ----------
    original_graph : Graph
        The original graph
    downstream_connecting : List[int]
        Downstream-connecting vertex indices
    internal_indices : List[int]
        Internal vertex indices of the current SCC

    Returns
    -------
    List[int]
        List of original vertex indices in downstream SCCs
    """
    internal_set = set(internal_indices)
    downstream_vertices = set()

    for v_idx in downstream_connecting:
        vertex = original_graph.vertex_at(v_idx)
        edges = vertex.parameterized_edges() if original_graph.parameterized() else vertex.edges()

        for edge in edges:
            target_idx = edge.to().index()
            if target_idx not in internal_set:
                downstream_vertices.add(target_idx)

    return sorted(list(downstream_vertices))


def _build_scc_subgraph(
    original_graph: 'Graph',
    scc: 'SCCVertex',
    scc_graph: 'SCCGraph'
) -> Tuple['Graph', Dict[str, any]]:
    """
    Build SCC subgraph for non-first SCCs (with upstream vertices).

    This handles SCCs that have upstream vertices from previous SCCs.
    The auto-starting vertex is NOT part of the original graph.

    Vertex ordering: {*upstream, *upstream-connecting, *internal,
                      *downstream-connecting, *downstream}

    - Upstream vertices: From previous SCCs (fake starting vertices)
    - Upstream-connecting: Internal vertices receiving from upstream
    - Internal: Pure internal SCC vertices
    - Downstream-connecting: Internal vertices connecting to downstream
    - Downstream vertices: From future SCCs (fake absorbing vertices)

    Parameters
    ----------
    original_graph : Graph
        The original graph
    scc : SCCVertex
        The SCC to build subgraph for (NOT the first SCC)
    scc_graph : SCCGraph
        The SCC decomposition

    Returns
    -------
    scc_subgraph : Graph
        Subgraph with auto-start at index 0 (not in original)
    metadata : Dict
        Contains vertex categorization and mapping:
        - 'upstream': List[int] - upstream vertex indices (original)
        - 'upstream_connecting': List[int] - upstream-connecting indices (original)
        - 'internal': List[int] - internal vertex indices (original)
        - 'downstream_connecting': List[int] - downstream-connecting indices (original)
        - 'downstream': List[int] - downstream vertex indices (original)
        - 'vertex_map': Dict[int, int] - orig_idx -> subgraph_idx
        - 'ordered_vertices': List[int | None] - trace[0]=None, trace[i+1]=ordered[i]
    """
    from . import Graph

    # Step 1: Get internal vertex indices
    internal_indices = scc.internal_vertex_indices()

    logger.debug(f"Building enhanced subgraph for SCC with {len(internal_indices)} internal vertices")

    # Step 2: Find vertex categories
    upstream_vertices = _find_upstream_vertices(original_graph, internal_indices, scc_graph)
    upstream_connecting = _find_upstream_connecting(internal_indices, upstream_vertices, original_graph)
    downstream_connecting = _find_downstream_connecting(internal_indices, original_graph)
    downstream_vertices = _find_downstream_vertices(original_graph, downstream_connecting, internal_indices)

    # Compute pure internal vertices (not connecting to upstream or downstream)
    connecting_set = set(upstream_connecting + downstream_connecting)
    internal_only = [v for v in internal_indices if v not in connecting_set]

    logger.debug(f"  Internal indices: {internal_indices}")
    logger.debug(f"  Upstream: {upstream_vertices} ({len(upstream_vertices)})")
    logger.debug(f"  Upstream-connecting: {upstream_connecting} ({len(upstream_connecting)})")
    logger.debug(f"  Internal-only: {internal_only} ({len(internal_only)})")
    logger.debug(f"  Downstream-connecting: {downstream_connecting} ({len(downstream_connecting)})")
    logger.debug(f"  Downstream: {downstream_vertices} ({len(downstream_vertices)})")

    # Step 3: Create ordered vertex list (critical for elimination!)
    # Ordering: {*upstream, *upstream-connecting, *internal, *downstream-connecting, *downstream}
    #
    # IMPORTANT: A vertex can appear in multiple categories (e.g., both upstream_connecting
    # and downstream_connecting). We must ensure each vertex appears ONLY ONCE.
    # Priority: upstream > upstream_connecting > internal_only > downstream_connecting > downstream

    ordered_vertices = []
    seen = set()

    # Add each category in priority order, skipping duplicates
    for category in [upstream_vertices, upstream_connecting, internal_only,
                     downstream_connecting, downstream_vertices]:
        for v in category:
            if v not in seen:
                ordered_vertices.append(v)
                seen.add(v)

    logger.debug(f"  Total vertices in enhanced subgraph: {len(ordered_vertices)}")

    # Verify no duplicates
    if len(ordered_vertices) != len(seen):
        logger.error("DUPLICATE VERTICES IN ordered_vertices!")
        from collections import Counter
        counts = Counter(ordered_vertices)
        duplicates = {v: c for v, c in counts.items() if c > 1}
        logger.error(f"  Duplicates: {duplicates}")
        raise ValueError(f"ordered_vertices contains duplicates: {duplicates}")

    # Step 4: Build subgraph
    scc_subgraph = Graph(
        original_graph.state_length(),
        parameterized=original_graph.parameterized()
    )

    vertex_map = {}  # orig_idx -> subgraph_vertex

    # For non-first SCCs: auto-starting vertex is NOT in original graph
    # Create NEW vertices for ALL ordered vertices (don't reuse auto-start)
    for orig_idx in ordered_vertices:
        orig_vertex = original_graph.vertex_at(orig_idx)
        new_vertex = scc_subgraph.create_vertex(orig_vertex.state())
        vertex_map[orig_idx] = new_vertex

    logger.debug(f"  Created {scc_subgraph.vertices_length()} vertices in subgraph (from {len(ordered_vertices)} ordered)")

    # Step 5: Add edges
    # Define which vertices are in the subgraph
    subgraph_vertices_set = set(ordered_vertices)
    internal_set = set(internal_indices)

    # Add edges FROM upstream vertices TO subgraph vertices (not back to their home SCC)
    for up_idx in upstream_vertices:
        up_vertex = original_graph.vertex_at(up_idx)
        edges = up_vertex.parameterized_edges() if original_graph.parameterized() else up_vertex.edges()

        for edge in edges:
            target_idx = edge.to().index()
            # Only add edges going INTO the subgraph (to internal vertices)
            if target_idx in internal_set:
                from_vertex = vertex_map[up_idx]
                to_vertex = vertex_map[target_idx]

                if original_graph.parameterized():
                    coeffs = list(edge.edge_state(original_graph.param_length()))
                    from_vertex.add_edge(to_vertex, coeffs)
                else:
                    from_vertex.add_edge(to_vertex, edge.weight())

    # Add edges FROM internal vertices (all 3 categories)
    for v_idx in internal_indices:
        vertex = original_graph.vertex_at(v_idx)
        edges = vertex.parameterized_edges() if original_graph.parameterized() else vertex.edges()

        for edge in edges:
            target_idx = edge.to().index()
            # Add edge if target is in subgraph
            if target_idx in subgraph_vertices_set:
                from_vertex = vertex_map[v_idx]
                to_vertex = vertex_map[target_idx]

                if original_graph.parameterized():
                    coeffs = list(edge.edge_state(original_graph.param_length()))
                    from_vertex.add_edge(to_vertex, coeffs)
                else:
                    from_vertex.add_edge(to_vertex, edge.weight())

    # Downstream vertices have NO outgoing edges (they're absorbing)

    logger.debug(f"  Added edges to enhanced subgraph")

    # Step 6: Create metadata
    # For non-first SCCs, the auto-starting vertex (subgraph index 0) is NOT in original graph
    # Build mapping for TRACE: trace_idx -> orig_idx
    #
    # Trace structure:
    #   trace[0] = auto-start vertex (NOT in original) → None
    #   trace[1] = created vertex 0 → ordered_vertices[0]
    #   trace[2] = created vertex 1 → ordered_vertices[1]
    #   ...
    #
    # So: trace[i] -> ordered_vertices[i-1] for i >= 1, trace[0] -> None

    trace_ordered_vertices = [None]  # trace[0] = auto-start, skip during stitching
    trace_ordered_vertices.extend(ordered_vertices)  # trace[i+1] = ordered[i]

    metadata = {
        'upstream': upstream_vertices,
        'upstream_connecting': upstream_connecting,
        'internal': internal_only,
        'downstream_connecting': downstream_connecting,
        'downstream': downstream_vertices,
        'vertex_map': {orig_idx: vertex_map[orig_idx].index() for orig_idx in ordered_vertices},
        'ordered_vertices': trace_ordered_vertices  # [None, *ordered_vertices]
    }

    logger.info(f"SCC subgraph: {scc_subgraph.vertices_length()} vertices, "
                f"param={scc_subgraph.parameterized()}, param_length={scc_subgraph.param_length()}")

    return scc_subgraph, metadata


def _build_first_scc_subgraph(
    original_graph: 'Graph',
    starting_vertex_idx: int,
    scc: 'SCCVertex',
    scc_graph: 'SCCGraph'
) -> Tuple['Graph', Dict[str, any]]:
    """
    Build special subgraph for the first SCC (contains starting vertex).

    The first SCC is special because:
    - Auto-starting vertex (index 0) IS the actual starting vertex of original graph
    - NO upstream vertices (this is the first SCC!)
    - Only contains: {starting_vertex, *downstream_connecting}

    Parameters
    ----------
    original_graph : Graph
        The original graph
    starting_vertex_idx : int
        Index of the starting vertex in original graph
    scc : SCCVertex
        The first SCC
    scc_graph : SCCGraph
        The SCC decomposition

    Returns
    -------
    first_subgraph : Graph
        Subgraph with starting vertex at index 0
    metadata : Dict
        Contains:
        - 'upstream': [] (empty - no upstream for first SCC)
        - 'internal': [starting_vertex_idx]
        - 'downstream': List[int] - downstream connecting vertices
        - 'vertex_map': Dict[int, int] - orig_idx -> subgraph_idx
        - 'ordered_vertices': List[int] - maps trace[i] -> orig_idx
    """
    from . import Graph

    logger.debug(f"Building FIRST SCC subgraph for starting vertex {starting_vertex_idx}")

    # Step 1: Get internal vertices (should be just the starting vertex for first SCC)
    internal_indices = scc.internal_vertex_indices()

    # Step 2: Find downstream connecting vertices
    # These are vertices that the starting vertex connects to (in next SCCs)
    downstream_connecting = _find_downstream_connecting(internal_indices, original_graph)
    downstream_vertices = _find_downstream_vertices(original_graph, downstream_connecting, internal_indices)

    logger.debug(f"  Starting vertex: {starting_vertex_idx}")
    logger.debug(f"  Internal indices: {internal_indices}")
    logger.debug(f"  Downstream connecting: {downstream_connecting}")
    logger.debug(f"  Downstream vertices: {downstream_vertices}")

    # Step 3: Create ordered vertex list
    # Ordering for first SCC: [starting_vertex, *downstream]
    ordered_vertices = [starting_vertex_idx] + downstream_vertices

    # Step 4: Build subgraph
    first_subgraph = Graph(
        original_graph.state_length(),
        parameterized=original_graph.parameterized()
    )

    vertex_map = {}
    auto_starting_vertex = first_subgraph.starting_vertex()

    # Reuse auto-starting vertex for the actual starting vertex
    vertex_map[starting_vertex_idx] = auto_starting_vertex

    # Create downstream vertices
    for downstream_idx in downstream_vertices:
        downstream_orig = original_graph.vertex_at(downstream_idx)
        new_vertex = first_subgraph.create_vertex(downstream_orig.state())
        vertex_map[downstream_idx] = new_vertex

    logger.debug(f"  Created {first_subgraph.vertices_length()} vertices in first subgraph")

    # Step 5: Add edges from starting vertex to downstream
    starting_vertex = original_graph.vertex_at(starting_vertex_idx)
    edges = starting_vertex.parameterized_edges() if original_graph.parameterized() else starting_vertex.edges()

    for edge in edges:
        target_idx = edge.to().index()
        # Only add edges to vertices in our subgraph
        if target_idx in vertex_map:
            from_vertex = vertex_map[starting_vertex_idx]
            to_vertex = vertex_map[target_idx]

            if original_graph.parameterized():
                coeffs = list(edge.edge_state(original_graph.param_length()))
                from_vertex.add_edge(to_vertex, coeffs)
            else:
                from_vertex.add_edge(to_vertex, edge.weight())

    # Downstream vertices are absorbing in this subgraph (no outgoing edges)

    # Step 6: Create metadata
    # For first SCC, trace mapping is direct: trace[i] -> ordered_vertices[i]
    metadata = {
        'upstream': [],  # No upstream for first SCC
        'upstream_connecting': [],
        'internal': [starting_vertex_idx],
        'downstream_connecting': downstream_connecting,
        'downstream': downstream_vertices,
        'vertex_map': {orig_idx: vertex_map[orig_idx].index() for orig_idx in ordered_vertices},
        'ordered_vertices': ordered_vertices  # Direct mapping: trace[i] -> ordered[i]
    }

    logger.info(f"First SCC subgraph: {first_subgraph.vertices_length()} vertices, "
                f"{len(ordered_vertices)} in ordered list")

    return first_subgraph, metadata


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
    sccs = list(scc_graph.sccs_in_topo_order())  # Convert to list to avoid iterator exhaustion
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


def _find_sister_vertices(
    upstream_metadata: Dict[str, any],
    downstream_metadata: Dict[str, any],
    original_graph: 'Graph'
) -> List[Tuple[int, int]]:
    """
    Find sister vertices between upstream and downstream subgraphs.

    Sister vertices are vertices with the same state vector that appear:
    - As downstream vertices in the upstream subgraph
    - As upstream vertices in the downstream subgraph

    Parameters
    ----------
    upstream_metadata : Dict
        Metadata from upstream enhanced subgraph
    downstream_metadata : Dict
        Metadata from downstream enhanced subgraph
    original_graph : Graph
        The original graph

    Returns
    -------
    List[Tuple[int, int]]
        List of (upstream_downstream_idx, downstream_upstream_idx) pairs
        where indices are original graph indices
    """
    sisters = []

    # Get downstream vertices from upstream subgraph
    upstream_downstream = upstream_metadata.get('downstream', [])

    # Get upstream vertices from downstream subgraph
    downstream_upstream = downstream_metadata.get('upstream', [])

    logger.debug(f"  Finding sisters: upstream has {len(upstream_downstream)} downstream, "
                f"downstream has {len(downstream_upstream)} upstream")

    # Match by state vector
    for up_down_idx in upstream_downstream:
        up_down_state = tuple(original_graph.vertex_at(up_down_idx).state())

        for down_up_idx in downstream_upstream:
            down_up_state = tuple(original_graph.vertex_at(down_up_idx).state())

            if up_down_state == down_up_state:
                sisters.append((up_down_idx, down_up_idx))
                logger.debug(f"    Found sister pair: {up_down_idx} (upstream→downstream) ↔ "
                           f"{down_up_idx} (downstream→upstream)")

    logger.debug(f"  Found {len(sisters)} sister vertex pairs")

    return sisters


def _get_vertex_categories_from_metadata(
    metadata: Dict[str, any]
) -> Dict[str, List[int]]:
    """
    Extract vertex categories from enhanced subgraph metadata.

    Parameters
    ----------
    metadata : Dict
        Metadata from _build_scc_subgraph or _build_first_scc_subgraph

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping category names to lists of original vertex indices:
        - 'upstream': Upstream vertices
        - 'upstream_connecting': Upstream-connecting vertices
        - 'internal': Internal vertices
        - 'downstream_connecting': Downstream-connecting vertices
        - 'downstream': Downstream vertices
    """
    return {
        'upstream': metadata.get('upstream', []),
        'upstream_connecting': metadata.get('upstream_connecting', []),
        'internal': metadata.get('internal', []),
        'downstream_connecting': metadata.get('downstream_connecting', []),
        'downstream': metadata.get('downstream', [])
    }


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
) -> Tuple[Dict[str, 'EliminationTrace'], Dict[str, Dict[str, any]]]:
    """
    Record elimination traces for each SCC using enhanced subgraphs.

    Enhanced subgraphs include upstream and downstream vertices with proper
    5-part vertex ordering for correct elimination.

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition
    param_length : int
        Number of parameters

    Returns
    -------
    scc_trace_dict : Dict[str, EliminationTrace]
        Traces for each SCC, keyed by SCC hash
    scc_metadata_dict : Dict[str, Dict]
        Metadata for each SCC's enhanced subgraph, keyed by SCC hash
    """
    from .trace_elimination import record_elimination_trace

    original_graph = scc_graph.original_graph()
    sccs = list(scc_graph.sccs_in_topo_order())

    scc_trace_dict = {}
    scc_metadata_dict = {}

    for i, scc in enumerate(sccs):
        # Build subgraph (first SCC vs other SCCs)
        if i == 0:
            # First SCC - contains starting vertex
            enhanced_subgraph, metadata = _build_first_scc_subgraph(
                original_graph, original_graph.starting_vertex().index(), scc, scc_graph
            )
        else:
            # Other SCCs - have upstream vertices
            enhanced_subgraph, metadata = _build_scc_subgraph(original_graph, scc, scc_graph)

        # Record trace for subgraph
        trace = record_elimination_trace(enhanced_subgraph, param_length)

        # Store with SCC hash
        scc_hash = scc.hash()
        scc_trace_dict[scc_hash] = trace
        scc_metadata_dict[scc_hash] = metadata

    return scc_trace_dict, scc_metadata_dict


def stitch_scc_traces(
    scc_graph: 'SCCGraph',
    scc_trace_dict: Dict[str, 'EliminationTrace'],
    scc_metadata_dict: Optional[Dict[str, Dict[str, any]]] = None,
    verbose: bool = False
) -> 'EliminationTrace':
    """
    Merge SCC traces using sister vertex merging.

    Sister vertices (vertices appearing in multiple SCCs) are merged by:
    1. Removing downstream sisters from the merged trace
    2. Attaching downstream sister's outgoing edges to upstream sister
    3. Keeping upstream sister's incoming edges

    Parameters
    ----------
    scc_graph : SCCGraph
        SCC decomposition with topological ordering
    scc_trace_dict : Dict[str, EliminationTrace]
        Traces for each SCC (recorded with enhanced subgraphs)
    scc_metadata_dict : Dict[str, Dict], optional
        Metadata for each SCC (from _build_scc_subgraph or _build_first_scc_subgraph).
        If None, metadata will be regenerated from SCC decomposition.

    Returns
    -------
    EliminationTrace
        Full graph trace stitched from SCC traces with sister merging

    Notes
    -----
    - Processes SCCs in topological order (dependencies first)
    - Sister vertices are merged to avoid duplication
    - Uses Option 2 (sequential merging) for multiple upstream SCCs
    - Metadata can be regenerated deterministically if not provided

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
    sccs = list(scc_graph.sccs_in_topo_order())

    if len(sccs) == 0:
        logger.error("Cannot stitch traces: no SCCs in graph")
        raise ValueError("Cannot stitch empty SCC graph")

    logger.info("Starting sister vertex merging: %d SCCs, %d vertices total",
                len(sccs), original_graph.vertices_length())

    # Track which vertices have been processed to avoid duplicates
    processed_vertices = set()

    # Generate metadata if not provided - INFER from traces instead of regenerating!
    if scc_metadata_dict is None:
        logger.info("Inferring SCC metadata from cached traces...")
        scc_metadata_dict = {}
        for scc in sccs:
            scc_hash = scc.hash()

            # Skip SCCs that don't have traces (small SCCs below min_size threshold)
            # These SCCs' vertices are already included in the enhanced subgraphs of large SCCs
            if scc_hash not in scc_trace_dict:
                logger.debug("  Skipping SCC %s (no trace - likely below min_size threshold)", scc_hash[:16])
                continue

            scc_trace = scc_trace_dict[scc_hash]

            # Infer ordered_vertices by matching trace states to original graph
            # NOTE: The trace may have duplicate states (e.g., starting vertex appearing multiple times)
            # We map each trace index to an original graph index, even if there are duplicates
            ordered_vertices = []

            for trace_idx in range(scc_trace.n_vertices):
                trace_state = tuple(scc_trace.states[trace_idx])

                # Find matching vertex in original graph
                found = False
                for orig_idx in range(original_graph.vertices_length()):
                    orig_state = tuple(original_graph.vertex_at(orig_idx).state())
                    if orig_state == trace_state:
                        ordered_vertices.append(orig_idx)
                        found = True
                        break

                if not found:
                    logger.error("  Could not find vertex with state %s in original graph!", trace_state)
                    raise ValueError(f"Trace state {trace_state} not found in original graph")

            # Log if there are duplicates
            if len(ordered_vertices) != len(set(ordered_vertices)):
                from collections import Counter
                counts = Counter(ordered_vertices)
                duplicates = {v: c for v, c in counts.items() if c > 1}
                logger.warning("  Trace has duplicate vertices: %s", duplicates)

            # Recompute vertex categories using same logic as _build_scc_subgraph
            internal_indices = scc.internal_vertex_indices()
            upstream_vertices = _find_upstream_vertices(original_graph, internal_indices, scc_graph)
            upstream_connecting = _find_upstream_connecting(internal_indices, upstream_vertices, original_graph)
            downstream_connecting = _find_downstream_connecting(internal_indices, original_graph)
            downstream_vertices = _find_downstream_vertices(original_graph, downstream_connecting, internal_indices)

            connecting_set = set(upstream_connecting + downstream_connecting)
            internal_only = [v for v in internal_indices if v not in connecting_set]

            # IMPORTANT: internal should be based on ordered_vertices, not internal_indices!
            # ordered_vertices may include upstream/downstream, so we need to filter them out
            upstream_set = set(upstream_vertices)
            downstream_set = set(downstream_vertices)
            internal_in_trace = [v for v in ordered_vertices if v not in upstream_set and v not in downstream_set]

            # Create full metadata with all vertex categories
            # (ordered_vertices is already deduplicated during subgraph building)
            metadata = {
                'ordered_vertices': ordered_vertices,
                'vertex_map': {orig_idx: idx for idx, orig_idx in enumerate(ordered_vertices)},
                'upstream': upstream_vertices,
                'upstream_connecting': upstream_connecting,
                'internal': internal_in_trace,  # Use the filtered internal list from ordered_vertices
                'downstream_connecting': downstream_connecting,
                'downstream': downstream_vertices
            }
            scc_metadata_dict[scc_hash] = metadata

            logger.debug("  SCC %s: inferred %d vertices, upstream=%d, internal=%d, downstream=%d",
                        scc_hash[:16], len(ordered_vertices), len(upstream_vertices), len(internal_in_trace), len(downstream_vertices))

    # Initialize merged trace with first SCC that has a trace
    first_scc = None
    first_hash = None
    first_trace = None
    first_metadata = None

    for scc in sccs:
        scc_hash = scc.hash()
        if scc_hash in scc_trace_dict:
            first_scc = scc
            first_hash = scc_hash
            first_trace = scc_trace_dict[first_hash]
            first_metadata = scc_metadata_dict[first_hash]
            break

    if first_scc is None:
        logger.error("No SCCs with traces found - this should not happen!")
        raise ValueError("No SCCs with traces in scc_trace_dict")

    logger.info("Initializing with first SCC (hash=%s..., %d vertices, %d operations)",
                first_hash[:16], first_trace.n_vertices, len(first_trace.operations))

    # Validate that metadata matches trace
    first_ordered = first_metadata['ordered_vertices']
    logger.info("  First ordered vertices: %d", len(first_ordered))
    logger.info("  First trace vertices: %d", first_trace.n_vertices)

    if len(first_ordered) != first_trace.n_vertices:
        logger.error("MISMATCH: ordered_vertices length (%d) != trace n_vertices (%d)",
                    len(first_ordered), first_trace.n_vertices)
        logger.error("  This indicates metadata doesn't match the cached trace!")
        logger.error("  ordered_vertices: %s", first_ordered)
        raise ValueError("Metadata mismatch: ordered_vertices length != trace n_vertices")

    # Create merged trace structure
    merged = EliminationTrace(
        operations=list(first_trace.operations),  # Copy operations
        vertex_rates=np.full(original_graph.vertices_length(), -1, dtype=np.int64),
        edge_probs=[[] for _ in range(original_graph.vertices_length())],
        vertex_targets=[[] for _ in range(original_graph.vertices_length())],
        states=np.array([original_graph.vertex_at(i).state()
                        for i in range(original_graph.vertices_length())]),
        starting_vertex_idx=original_graph.starting_vertex().index(),
        n_vertices=original_graph.vertices_length(),
        state_length=first_trace.state_length,
        param_length=original_graph.param_length(),
        reward_length=first_trace.reward_length,
        is_discrete=first_trace.is_discrete,
        metadata={'scc_metadata': {first_hash: first_metadata}}
    )

    # Map first SCC's trace vertices to original graph vertices
    first_ordered = first_metadata['ordered_vertices']
    for subgraph_idx, orig_idx in enumerate(first_ordered):
        # Copy vertex data from first trace to merged trace
        if first_trace.vertex_rates[subgraph_idx] >= 0:
            merged.vertex_rates[orig_idx] = first_trace.vertex_rates[subgraph_idx]

        n_edges = len(first_trace.edge_probs[subgraph_idx])
        if n_edges > 0:
            logger.debug(f"    Adding {n_edges} edges from trace[{subgraph_idx}] to merged[{orig_idx}]")

        for edge_idx, edge_prob_op in enumerate(first_trace.edge_probs[subgraph_idx]):
            merged.edge_probs[orig_idx].append(edge_prob_op)

            target_subgraph_idx = first_trace.vertex_targets[subgraph_idx][edge_idx]

            # Bounds check: target_subgraph_idx should be within first_ordered
            if target_subgraph_idx < len(first_ordered):
                target_orig_idx = first_ordered[target_subgraph_idx]
                merged.vertex_targets[orig_idx].append(target_orig_idx)
                logger.debug(f"      Edge {edge_idx}: trace[{subgraph_idx}]→trace[{target_subgraph_idx}] becomes merged[{orig_idx}]→merged[{target_orig_idx}]")
            else:
                logger.error("Target index %d out of range for first_ordered (length=%d)",
                           target_subgraph_idx, len(first_ordered))
                logger.error("  Source vertex: subgraph_idx=%d, orig_idx=%d",
                           subgraph_idx, orig_idx)
                logger.error("  First ordered vertices: %d", len(first_ordered))
                logger.error("  First trace vertices: %d", first_trace.n_vertices)
                raise IndexError(f"Target vertex index {target_subgraph_idx} out of range")

        # Mark as processed
        processed_vertices.add(orig_idx)

    logger.info("  First SCC merged: %d operations, %d vertices processed",
                len(merged.operations), len([r for r in merged.vertex_rates if r >= 0]))

    # Process remaining SCCs in topological order
    # Add progress bar for stitching
    if verbose:
        scc_iterator = tqdm(
            range(len(sccs)),
            desc="Stitching traces",
            unit="SCC",
            leave=False
        )
    else:
        scc_iterator = range(len(sccs))

    for scc_idx in scc_iterator:
        scc = sccs[scc_idx]
        scc_hash = scc.hash()

        # Skip the first SCC that we already processed
        if scc_hash == first_hash:
            logger.debug("  Skipping SCC %d/%d (already processed as first SCC)", scc_idx + 1, len(sccs))
            continue

        # Skip SCCs that don't have traces (small SCCs below min_size threshold)
        if scc_hash not in scc_trace_dict:
            logger.debug("  Skipping SCC %d/%d (hash=%s..., no trace - likely below min_size threshold)",
                        scc_idx + 1, len(sccs), scc_hash[:16])
            continue

        scc_trace = scc_trace_dict[scc_hash]
        scc_metadata = scc_metadata_dict[scc_hash]

        logger.info("Merging SCC %d/%d (hash=%s..., %d vertices, %d operations)",
                    scc_idx + 1, len(sccs), scc_hash[:16],
                    scc_trace.n_vertices, len(scc_trace.operations))

        # Find sister vertices with previous SCCs (already in merged trace)
        merged_metadata = merged.metadata.get('scc_metadata', {})
        sisters = []

        for prev_hash, prev_metadata in merged_metadata.items():
            scc_sisters = _find_sister_vertices(prev_metadata, scc_metadata, original_graph)
            sisters.extend(scc_sisters)

        logger.info("  Found %d sister vertex pairs", len(sisters))

        # Append SCC operations to merged trace
        op_offset = len(merged.operations)
        for operation in scc_trace.operations:
            remapped_op = _remap_operation(operation, op_offset)
            merged.operations.append(remapped_op)

        logger.info("  Appended %d operations (offset=%d, total now=%d)",
                    len(scc_trace.operations), op_offset, len(merged.operations))

        # Build mapping from SCC trace indices to original indices
        scc_ordered = scc_metadata['ordered_vertices']

        # Get upstream vertices in current SCC (fake starting vertices from previous SCCs)
        # These were already processed in their home SCC, so skip them here
        upstream_in_current_scc = set(scc_metadata.get('upstream', []))

        # Get downstream vertices in current SCC (fake absorbing vertices for downstream SCCs)
        # These will be processed as internal vertices when we reach their home SCC
        downstream_in_current_scc = set(scc_metadata.get('downstream', []))

        logger.debug("  Processing SCC %d: ordered=%s", scc_idx + 1, scc_ordered)
        logger.debug("    upstream_in_current: %s", upstream_in_current_scc)
        logger.debug("    downstream_in_current: %s", downstream_in_current_scc)
        logger.debug("    sisters: %s", sisters)

        # Merge vertex data
        for subgraph_idx, orig_idx in enumerate(scc_ordered):
            # SKIP None entries (auto-starting vertex, not in original graph)
            if orig_idx is None:
                logger.debug("    Skipping trace vertex %d (auto-starting vertex, not in original)",
                            subgraph_idx)
                continue

            # SKIP upstream vertices - they were already processed in their home SCC
            if orig_idx in upstream_in_current_scc:
                logger.debug("    Skipping upstream vertex %d (already processed in previous SCC)",
                            orig_idx)
                continue

            # SKIP downstream sisters - they will be processed in their home SCC
            # But first, attach their edges to upstream sister if this is a sister pair
            if orig_idx in downstream_in_current_scc:
                is_sister = any(s[1] == orig_idx for s in sisters)
                if is_sister:
                    upstream_sister = next((s[0] for s in sisters if s[1] == orig_idx), None)
                    if upstream_sister is not None:
                        logger.debug("    Attaching edges from downstream sister %d to upstream sister %d",
                                    orig_idx, upstream_sister)

                        # Attach this vertex's edges to upstream sister
                        for edge_idx, edge_prob_op in enumerate(scc_trace.edge_probs[subgraph_idx]):
                            remapped_prob_op = edge_prob_op + op_offset if edge_prob_op >= 0 else edge_prob_op
                            merged.edge_probs[upstream_sister].append(remapped_prob_op)

                            target_subgraph_idx = scc_trace.vertex_targets[subgraph_idx][edge_idx]
                            target_orig_idx = scc_ordered[target_subgraph_idx]

                            # Skip if target is None
                            if target_orig_idx is None:
                                logger.warning("    Sister edge target is None, skipping")
                                continue

                            merged.vertex_targets[upstream_sister].append(target_orig_idx)

                # Skip processing this downstream vertex's own rate/edges
                logger.debug("    Skipping downstream vertex %d (will process in home SCC)",
                            orig_idx)
                continue

            # Check if already processed (handles duplicates in ordered_vertices)
            if orig_idx in processed_vertices:
                logger.debug("    Skipping vertex %d (already processed in earlier SCC)", orig_idx)
                continue

            # Not a sister, or is upstream vertex in this SCC - process normally
            if scc_trace.vertex_rates[subgraph_idx] >= 0:
                remapped_rate = scc_trace.vertex_rates[subgraph_idx] + op_offset
                merged.vertex_rates[orig_idx] = remapped_rate

            for edge_idx, edge_prob_op in enumerate(scc_trace.edge_probs[subgraph_idx]):
                remapped_prob_op = edge_prob_op + op_offset if edge_prob_op >= 0 else edge_prob_op
                merged.edge_probs[orig_idx].append(remapped_prob_op)

                target_subgraph_idx = scc_trace.vertex_targets[subgraph_idx][edge_idx]
                target_orig_idx = scc_ordered[target_subgraph_idx]

                # Skip if target is None (shouldn't happen, but safety check)
                if target_orig_idx is None:
                    logger.warning("    Edge target is None (auto-start), skipping edge from %d", orig_idx)
                    continue

                merged.vertex_targets[orig_idx].append(target_orig_idx)

            # Mark as processed
            processed_vertices.add(orig_idx)

        # Store metadata for future sister matching
        merged.metadata['scc_metadata'][scc_hash] = scc_metadata

        logger.info("  SCC %d/%d merged: total operations=%d, vertices processed=%d",
                    scc_idx + 1, len(sccs), len(merged.operations),
                    len([r for r in merged.vertex_rates if r >= 0]))

    # Final validation
    vertices_with_rates = sum(1 for r in merged.vertex_rates if r >= 0)
    total_edges = sum(len(edge_list) for edge_list in merged.edge_probs)

    logger.info("✓ Sister merging complete: %d vertices, %d operations, %d edges",
                merged.n_vertices, len(merged.operations), total_edges)
    logger.info("  Vertices with rates: %d / %d", vertices_with_rates, merged.n_vertices)

    return merged


# ============================================================================
# Main Entry Point
# ============================================================================

def get_trace_hierarchical(graph,
                          param_length: Optional[int] = None,
                          min_size: int = 50,
                          parallel_strategy: str = 'auto',
                          use_scc_subdivision: bool = True,
                          verbose: bool = False) -> 'EliminationTrace':
    """
    Main entry point: Get trace with hierarchical SCC-based caching.

    Workflow:
    1. Check cache for full graph → return if hit
    2. If cache miss and use_scc_subdivision=True:
       a. Decompose graph into SCCs
       b. Recursively collect missing traces (with caching)
       c. Compute missing traces (potentially in parallel)
       d. Stitch SCC traces together
    3. If cache miss and use_scc_subdivision=False:
       - Compute trace directly for full graph
    4. Cache the result

    Parameters
    ----------
    graph : Graph
        Input graph (may be very large)
    param_length : int, optional
        Number of parameters (auto-detected if None)
    min_size : int, default=50
        Minimum vertices to subdivide into SCCs
        Graphs with < min_size vertices are computed directly
    parallel_strategy : str, default='auto'
        Parallelization strategy: 'auto', 'vmap', 'pmap', or 'sequential'
    use_scc_subdivision : bool, default=True
        If True, use hierarchical SCC-based subdivision and caching
        If False, compute full graph directly (Phase 3a behavior)
    verbose : bool, default=False
        If True, show progress bars for major computation stages

    Returns
    -------
    EliminationTrace
        Complete elimination trace

    Examples
    --------
    >>> # Hierarchical caching with SCC subdivision (recommended for large graphs)
    >>> trace = graph.compute_trace(hierarchical=True)
    >>>
    >>> # Adjust minimum subdivision size
    >>> trace = graph.compute_trace(hierarchical=True, min_size=100)
    >>>
    >>> # Disable SCC subdivision (simple full-graph caching)
    >>> trace = graph.compute_trace(hierarchical=True, min_size=50, use_scc_subdivision=False)

    Notes
    -----
    Benefits of SCC subdivision:
    - Reuses cached SCCs across different graphs
    - Enables parallel computation of independent components
    - Handles very large graphs (10K+ vertices)
    - Reduces redundant computation when graphs share structure
    """
    from .trace_elimination import record_elimination_trace
    from . import hash as phasic_hash

    n_vertices = graph.vertices_length()
    logger.debug("get_trace_hierarchical: graph=%d vertices, min_size=%d, use_scc_subdivision=%s",
                 n_vertices, min_size, use_scc_subdivision)

    # Check if graph is empty (happens when compute_trace() is called multiple times)
    # This is because record_elimination_trace() is destructive - it eliminates vertices
    if n_vertices == 0:
        raise ValueError(
            "Cannot compute trace: graph has no vertices. "
            "This usually means compute_trace() was called multiple times on the same graph. "
            "Note: compute_trace() with hierarchical=False is destructive and empties the graph. "
            "Use hierarchical=True (default) for caching, or create a new graph for each call."
        )

    # Step 1: Try full graph hash cache
    try:
        hash_result = phasic_hash.compute_graph_hash(graph)
        graph_hash = hash_result.hash_hex
        logger.debug("Full graph hash: %s...", graph_hash[:16])

        trace = _load_trace_from_cache(graph_hash)
        if trace is not None:
            logger.info("✓ Full graph cache HIT: returning cached trace")
            return trace  # Cache hit!

        logger.debug("Full graph cache MISS: need to compute trace")
    except Exception as e:
        # Hash computation failed - proceed without caching
        logger.warning("Failed to compute graph hash: %s", str(e))
        graph_hash = None

    # Step 2: Decide whether to use SCC subdivision
    if not use_scc_subdivision or n_vertices < min_size:
        # Compute directly without subdivision
        logger.debug("Computing trace directly (use_scc_subdivision=%s, %d < min_size=%d)",
                     use_scc_subdivision, n_vertices, min_size)
        # Use graph's param_length if not explicitly provided (auto-detection is broken)
        if param_length is None:
            param_length = graph.param_length()
        trace = record_elimination_trace(graph, param_length=param_length)
    else:
        # Use hierarchical SCC-based subdivision
        logger.info("Using hierarchical SCC subdivision (graph=%d vertices, min_size=%d)",
                    n_vertices, min_size)

        # Collect missing traces recursively
        logger.debug("Step 1: Collecting missing SCC traces...")
        work_units, all_scc_hashes, scc_decomp = collect_missing_traces_batch(graph, param_length=param_length, min_size=min_size, verbose=verbose)

        if len(work_units) > 0:
            # Compute missing traces (potentially in parallel)
            logger.debug("Step 2: Computing %d missing traces...", len(work_units))
            scc_traces = compute_missing_traces_parallel(work_units, strategy=parallel_strategy, min_size=min_size, verbose=verbose)
            logger.debug("✓ Computed %d missing traces", len(scc_traces))

            # NOTE: Do NOT explicitly delete work_units!
            # work_units may contain the original input graph (borrowed reference).
            # Explicit deletion causes double-free when caller tries to delete the same graph.
            # Python GC will clean up work_units when function returns.
        else:
            logger.debug("Step 2: No missing traces to compute (all cached)")
            scc_traces = {}

        # Build complete scc_trace_dict by loading from cache using collected hashes
        logger.debug("Step 3: Loading all %d SCC traces from cache...", len(all_scc_hashes))
        scc_trace_dict = {}

        # Add progress bar for cache loading
        if verbose:
            scc_hash_iterator = tqdm(
                enumerate(all_scc_hashes),
                total=len(all_scc_hashes),
                desc="Loading cached traces",
                unit="trace",
                leave=False
            )
        else:
            scc_hash_iterator = enumerate(all_scc_hashes)

        for i, scc_hash in scc_hash_iterator:
            logger.debug("  Loading SCC %d/%d: hash=%s...", i+1, len(all_scc_hashes), scc_hash[:16])
            trace = _load_trace_from_cache(scc_hash)
            if trace is None:
                logger.error("SCC trace not found in cache: %s... (should have been computed!)", scc_hash[:16])
                raise RuntimeError(f"Missing SCC trace for hash {scc_hash}")
            logger.info("  SCC %d/%d: %d vertices, %d operations, param_length=%d",
                       i+1, len(all_scc_hashes), trace.n_vertices, len(trace.operations), trace.param_length)
            scc_trace_dict[scc_hash] = trace

        logger.debug("✓ Loaded %d SCC traces from cache", len(scc_trace_dict))

        # Stitch traces together using the saved SCC decomposition
        # If scc_decomp is None, it means we recorded the full graph directly (no subdivision)
        if scc_decomp is None or len(scc_trace_dict) == 1:
            # Full graph was recorded directly - just use the single trace
            logger.debug("No stitching needed (single trace or no subdivision)")
            trace = list(scc_trace_dict.values())[0]
            logger.info("✓ Hierarchical trace computation complete (no stitching)")
        else:
            # Multiple SCC traces - need to stitch
            logger.debug("Step 4: Stitching %d SCC traces together...", len(scc_trace_dict))
            try:
                trace = stitch_scc_traces(scc_decomp, scc_trace_dict, verbose=verbose)
                logger.info("✓ Hierarchical trace computation complete")
            except Exception as e:
                logger.error("  ✗ Trace stitching failed: %s", str(e))
                raise

    # Step 3: Cache the full result
    if graph_hash is not None:
        _save_trace_to_cache(graph_hash, trace)

    return trace
