# Graph Caching + pmap Disk Serialization - Complete Implementation Plan

**Date**: 2025-11-14
**Author**: Claude Code
**Status**: Planning Phase

---

## Executive Summary

This document outlines the complete implementation plan for two interconnected features:

1. **Graph Caching**: Cache expensive graph construction using AST-based callback hashing
2. **pmap Disk Serialization**: Fix pmap crash by using disk-based work unit distribution

Both features share infrastructure and are designed to work together as a unified caching system.

---

## Background

### Problem 1: Expensive Graph Construction

Many phasic models are expensive to build:
- Two-locus ARG models: 600-3000+ vertices, 5-30 seconds to construct
- Complex coalescent models with large sample sizes
- State space models with intricate transition logic

Currently, graphs must be rebuilt every time a script runs, even if the callback and parameters are identical.

### Problem 2: pmap Crashes with Global State

Current pmap implementation (lines 590-660 in `hierarchical_trace_cache.py`) uses global dictionary `_work_unit_store` to pass data to workers. This fails because:
- pmap spawns separate processes (or XLA contexts)
- Global dict not shared across processes
- Workers try to access empty dict → KeyError → malformed graphs → C code crash

**Current error**:
```
Stack is empty.
 @ /Users/kmt/phasic/src/c/phasic.c (1794)
Exit code: 138 (SIGSEGV)
```

**Root cause**: Line 394 in `hierarchical_trace_cache.py`:
```python
graph_hash, json_str = _work_unit_store[idx_int]  # Empty in child processes!
```

---

## Design Decisions (User Confirmed)

1. **Callback hashing**: AST-based (structural), not source-based or Python hash()
2. **Parameter handling**: Implicit in structure (different N → different hash automatically)
3. **Cache eviction**: Manual cleanup only (no LRU auto-eviction)
4. **Implementation approach**: Unified design (build both features together)
5. **vmap preservation**: Keep existing vmap implementation unchanged (it works perfectly)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ User Code                                                        │
│   graph = Graph(callback, cache=True, nr_samples=100)          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Graph.__init__()                                                 │
│   1. Compute callback hash (AST-based, 0.1ms)                   │
│   2. Check graph cache (BEFORE building!)                       │
│   3. Cache HIT  → Load from disk (10ms) ✓                       │
│   4. Cache MISS → Build graph (5s) + Save to cache              │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ graph.compute_trace(parallel_strategy='pmap')                   │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ compute_missing_traces_parallel(strategy='pmap')                │
│   1. Write work units to shared disk                            │
│   2. Distribute file paths via pmap                             │
│   3. Workers load from disk + cache results                     │
│   4. Cleanup temp files                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: AST-Based Callback Hashing

### File: `src/phasic/callback_hash.py` (NEW)

#### Purpose
Compute stable, deterministic hashes for callback functions without building graphs.

#### Public API

```python
def hash_callback(callback: Callable, **params) -> str:
    """
    Compute stable hash for callback function + parameters.

    Uses AST-based content hashing (robust to formatting changes).

    Parameters
    ----------
    callback : Callable
        Callback function to hash
    **params : dict
        Additional parameters (nr_samples, theta, etc.)

    Returns
    -------
    str
        SHA256 hash hex string (32 characters)

    Raises
    ------
    ValueError
        If callback uses closures (captured variables)
    TypeError
        If callback is not hashable (C extensions, built-ins)

    Examples
    --------
    >>> def coalescent(state, theta=1.0):
    ...     n = state[0]
    ...     return [[n-1], [n*(n-1)/2 * theta]]

    >>> hash1 = hash_callback(coalescent, nr_samples=10, theta=1.0)
    >>> hash1
    'a3f2b8c9def12345...'

    >>> # Same callback, different formatting → SAME hash
    >>> def coalescent(state, theta=1.0):
    ...     n = state[0]
    ...     return [[n-1], [n*(n-1)/2 * theta]]  # Extra whitespace

    >>> hash2 = hash_callback(coalescent, nr_samples=10, theta=1.0)
    >>> hash1 == hash2
    True
    """
```

#### Implementation Details

**1. Version Tagging**
```python
PHASIC_CALLBACK_VERSION = "1.0"

# Increment this when hashing logic changes
# All old caches automatically invalidated
```

**2. Closure Detection**
```python
def _detect_closures(func: Callable) -> None:
    """
    Check if function uses closures (captured variables).

    Raises ValueError with helpful message if closures detected.
    """
    # Unwrap decorators first
    unwrapped = func
    while hasattr(unwrapped, '__wrapped__'):
        unwrapped = unwrapped.__wrapped__

    # Check __closure__ attribute
    if unwrapped.__closure__ is not None:
        raise ValueError(
            f"Callback '{func.__name__}' uses closures (captured variables).\n"
            "\n"
            "For caching to work, pass captured variables as explicit parameters.\n"
            "\n"
            "Example:\n"
            "  # GOOD (explicit parameter):\n"
            f"  def {func.__name__}(state, theta=2.0):\n"
            "      return [[state[0] - 1, [theta]]]\n"
            "\n"
            "  # BAD (closure - captures 'theta' from outer scope):\n"
            "  theta = 2.0\n"
            f"  def {func.__name__}(state):\n"
            "      return [[state[0] - 1, [theta]]]  # ← Captures theta\n"
            "\n"
            "Captured variables detected:\n" +
            "\n".join(f"  - {var}" for var in unwrapped.__code__.co_freevars)
        )
```

**3. AST Normalization**
```python
def _normalize_ast(node: ast.AST) -> ast.AST:
    """
    Normalize AST for stable hashing.

    Removes:
    - Position info (lineno, col_offset, end_lineno, end_col_offset)
    - Docstrings (first string in function/class/module body)

    Preserves:
    - Logic structure
    - Variable names
    - Control flow
    - Operations
    """
    # Step 1: Remove all position attributes
    for child in ast.walk(node):
        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if hasattr(child, attr):
                delattr(child, attr)

    # Step 2: Remove docstrings from functions, classes, modules
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            # Check if first statement is a string constant (docstring)
            if (child.body and
                isinstance(child.body[0], ast.Expr) and
                isinstance(child.body[0].value, ast.Constant) and
                isinstance(child.body[0].value.value, str)):
                # Remove docstring
                child.body = child.body[1:]

    return node
```

**4. Hash Computation**
```python
def hash_callback(callback: Callable, **params) -> str:
    """Main hashing function - see docstring above."""
    import hashlib
    import inspect
    import ast

    components = []

    # Component 1: Version tag
    components.append(f"version:{PHASIC_CALLBACK_VERSION}")

    # Component 2: Python version (AST structure changes between versions)
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    components.append(f"python:{py_version}")

    # Component 3: Unwrap decorators
    func = callback
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__

    # Component 4: Check for closures (reject if found)
    _detect_closures(func)

    # Component 5: Get source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise TypeError(
            f"Cannot hash callback '{func.__name__}': unable to get source code.\n"
            "This usually means the function is:\n"
            "  - A C extension function\n"
            "  - A built-in function\n"
            "  - Defined in an interactive session without proper module\n"
            "  - A lambda without accessible source\n"
            f"\nOriginal error: {e}"
        ) from e

    # Component 6: Parse to AST and normalize
    try:
        tree = ast.parse(source)
        normalized = _normalize_ast(tree)
        ast_str = ast.dump(normalized, annotate_fields=True)
        components.append(f"ast:{ast_str}")
    except SyntaxError as e:
        raise ValueError(
            f"Failed to parse callback '{func.__name__}' source code.\n"
            f"Syntax error: {e}"
        ) from e

    # Component 7: Add sorted parameters (deterministic order)
    if params:
        # Sort by key for determinism
        param_items = sorted(params.items())
        # Use repr() for consistent representation
        param_str = ",".join(f"{k}={repr(v)}" for k, v in param_items)
        components.append(f"params:{param_str}")

    # Component 8: Compute SHA256 hash
    combined = "||".join(components)
    hash_bytes = hashlib.sha256(combined.encode('utf-8')).digest()

    # Return first 32 hex chars (128 bits - sufficient for collision resistance)
    return hash_bytes.hex()[:32]
```

#### Edge Cases Handled

1. **Decorators**: Unwrap via `__wrapped__` attribute
2. **Closures**: Detect and reject with helpful error
3. **C extensions**: Catch `TypeError` from `inspect.getsource()`
4. **Lambda functions**: Work if source is accessible
5. **Methods**: Extract via `inspect.getsource()` (includes class context)
6. **Different Python versions**: Include version in hash
7. **Different parameter order**: Sort params for determinism

#### Edge Cases NOT Handled (Documented Limitations)

1. **Dynamic code generation** (`eval()`, `exec()`): Source is `"<string>"` → TypeError
2. **External dependencies**: Changes to imported functions not detected
3. **Nondeterministic callbacks**: Random/I/O operations break caching (user responsibility)
4. **Mutating callbacks**: In-place modifications break assumptions (user responsibility)

---

## Part 2: Graph Cache Infrastructure

### File: `src/phasic/graph_cache.py` (NEW)

#### Purpose
Manage persistent disk cache for graph objects using callback hashes as keys.

#### Cache Directory Structure
```
~/.phasic_cache/
├── graphs/
│   ├── a3f2b8c9def12345.json          # Cached graph
│   ├── a3f2b8c9def12345.meta.json    # Metadata (optional)
│   ├── f7e4d1c0abc98765.json
│   └── ...
├── traces/                            # Existing trace cache
│   └── ...
└── cache_info.json                    # Global cache stats (optional)
```

#### Graph JSON Format
```json
{
  "version": "0.22.0",
  "callback_hash": "a3f2b8c9def12345",
  "created_at": "2025-11-14T10:30:00Z",
  "python_version": "3.11",
  "construction_params": {
    "nr_samples": 20,
    "parameterized": true,
    "theta": 1.0
  },
  "graph_data": {
    "state_length": 1,
    "parameterized": true,
    "param_length": 3,
    "states": [[5], [4], [3], ...],
    "edges": [...],
    "start_edges": [...],
    "param_edges": [...],
    "start_param_edges": [...]
  }
}
```

#### Public API

```python
class GraphCache:
    """
    Manages persistent disk cache for Graph objects.

    Cache directory: ~/.phasic_cache/graphs/
    Cache key: callback hash (AST-based) + parameters

    Examples
    --------
    >>> cache = GraphCache()
    >>>
    >>> def callback(state, theta=1.0):
    ...     return [[state[0]-1], [theta]]
    >>>
    >>> # Save graph to cache
    >>> graph = Graph(callback, nr_samples=10, theta=1.0)
    >>> hash_key = cache.save_graph(graph, callback, theta=1.0, nr_samples=10)
    >>>
    >>> # Load graph from cache
    >>> cached = cache.load_graph(hash_key)
    >>> cached is not None
    True
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize graph cache.

        Parameters
        ----------
        cache_dir : Path, optional
            Cache directory. Defaults to ~/.phasic_cache/graphs/
        """

    def save_graph(self, graph: 'Graph', callback: Callable, **params) -> str:
        """
        Save graph to cache.

        Parameters
        ----------
        graph : Graph
            Graph object to cache
        callback : Callable
            Callback function used to build graph
        **params : dict
            Construction parameters

        Returns
        -------
        str
            Cache key (callback hash)
        """

    def load_graph(self, callback: Callable, **params) -> Optional['Graph']:
        """
        Load graph from cache.

        Parameters
        ----------
        callback : Callable
            Callback function
        **params : dict
            Construction parameters

        Returns
        -------
        Graph or None
            Cached graph if found, None otherwise
        """

    def get_or_build(self, callback: Callable, **params) -> 'Graph':
        """
        Get graph from cache or build if not found.

        Parameters
        ----------
        callback : Callable
            Callback function
        **params : dict
            Construction parameters (nr_samples, theta, etc.)

        Returns
        -------
        Graph
            Cached or newly built graph
        """

    def clear_graph_cache(self) -> int:
        """
        Clear all cached graphs.

        Returns
        -------
        int
            Number of graphs removed
        """

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns
        -------
        dict
            {'num_graphs': int, 'total_size_mb': float, ...}
        """


# Module-level convenience functions
def get_cache_dir() -> Path:
    """Get default cache directory."""

def clear_all_caches() -> None:
    """Clear both graph and trace caches."""
```

#### Implementation Details

**1. Cache Key Computation**
```python
def _compute_cache_key(self, callback: Callable, **params) -> str:
    """Compute cache key using callback hash."""
    from .callback_hash import hash_callback
    return hash_callback(callback, **params)
```

**2. Save to Cache**
```python
def save_graph(self, graph: 'Graph', callback: Callable, **params) -> str:
    # Compute cache key
    cache_key = self._compute_cache_key(callback, **params)

    # Serialize graph
    graph_data = graph.serialize()  # Existing Graph.serialize() method

    # Add metadata
    cache_entry = {
        'version': phasic.__version__,
        'callback_hash': cache_key,
        'created_at': datetime.now().isoformat(),
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
        'construction_params': params,
        'graph_data': graph_data
    }

    # Write to disk
    cache_path = self.cache_dir / f"{cache_key}.json"
    with open(cache_path, 'w') as f:
        json.dump(cache_entry, f, indent=2, default=_json_serialize_numpy)

    logger.info(f"Saved graph to cache: {cache_key[:16]}...")
    return cache_key
```

**3. Load from Cache**
```python
def load_graph(self, callback: Callable, **params) -> Optional['Graph']:
    # Compute cache key
    cache_key = self._compute_cache_key(callback, **params)

    # Check if cached file exists
    cache_path = self.cache_dir / f"{cache_key}.json"
    if not cache_path.exists():
        logger.debug(f"Cache miss: {cache_key[:16]}...")
        return None

    # Load from disk
    try:
        with open(cache_path, 'r') as f:
            cache_entry = json.load(f)

        # Version check (warn if different, but still load)
        if cache_entry['version'] != phasic.__version__:
            logger.warning(
                f"Cache version mismatch: cached with {cache_entry['version']}, "
                f"current is {phasic.__version__}"
            )

        # Deserialize graph
        from phasic import Graph
        graph_data = cache_entry['graph_data']
        graph = Graph.from_serialized(graph_data)

        logger.info(f"Cache hit: {cache_key[:16]}... ({graph.vertices_length()} vertices)")
        return graph

    except Exception as e:
        logger.error(f"Failed to load cached graph {cache_key[:16]}: {e}")
        return None
```

**4. Helper: JSON Serialization**
```python
def _json_serialize_numpy(obj):
    """Custom JSON serializer for numpy arrays."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
```

---

## Part 3: Graph.__init__() Integration

### File: `src/phasic/__init__.py` (MODIFIED)

#### Current Implementation (Line ~1470)
```python
class Graph:
    def __init__(self, callback=None, state_length=None, **kwargs):
        # ... existing initialization ...
        if callback:
            self._init_from_callback(callback, **kwargs)
        # ...
```

#### New Implementation (WITH CACHING)

```python
class Graph:
    def __init__(self, callback=None, state_length=None,
                 cache=False, force_rebuild=False, **kwargs):
        """
        Initialize Graph.

        Parameters
        ----------
        callback : callable, optional
            Callback function for graph construction
        state_length : int, optional
            Length of state vectors
        cache : bool, default=False
            Enable graph caching (saves/loads from ~/.phasic_cache/graphs/)
        force_rebuild : bool, default=False
            Force rebuild even if cache exists (useful for testing)
        **kwargs : dict
            Additional construction parameters (nr_samples, theta, etc.)
        """
        # ... existing initialization ...

        if callback:
            # NEW: Try loading from cache first (if caching enabled)
            if cache and not force_rebuild:
                cached_graph = self._load_from_cache(callback, **kwargs)
                if cached_graph is not None:
                    # Cache hit - copy cached graph data
                    self._copy_from_cached(cached_graph)
                    logger.debug("Loaded graph from cache (skipped construction)")
                    return

            # Cache miss or caching disabled - build normally
            self._init_from_callback(callback, **kwargs)

            # NEW: Save to cache after building (if caching enabled)
            if cache:
                self._save_to_cache(callback, **kwargs)
        # ...
```

#### New Methods

```python
def _load_from_cache(self, callback: Callable, **kwargs) -> Optional['Graph']:
    """
    Try loading graph from cache.

    Returns None if cache miss or error.
    """
    try:
        from .graph_cache import GraphCache
        cache = GraphCache()
        return cache.load_graph(callback, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to load from cache: {e}")
        return None

def _save_to_cache(self, callback: Callable, **kwargs) -> None:
    """
    Save graph to cache.

    Silently fails if caching error (graph is already built).
    """
    try:
        from .graph_cache import GraphCache
        cache = GraphCache()
        cache.save_graph(self, callback, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to save to cache: {e}")

def _copy_from_cached(self, cached_graph: 'Graph') -> None:
    """
    Copy data from cached graph to self.

    This avoids replacing 'self', which would break user references.
    """
    # Copy internal C++ graph pointer
    self.graph = cached_graph.graph
    # Copy any other cached attributes
    # ... (implementation depends on Graph internals)
```

---

## Part 4: pmap Disk Serialization

### File: `src/phasic/hierarchical_trace_cache.py` (MODIFIED)

#### Problem Location
- Lines 360: Global `_work_unit_store` dictionary
- Lines 523-528: Populated in parent process
- Lines 393-394, 611-617: Accessed in child pmap processes (FAILS!)

#### Solution: Disk-Based Work Unit Distribution

**Key Idea**: Write work units to shared disk, pass file paths through pmap instead of using global dict.

#### New Helper Functions (Insert after line 433)

```python
def _get_pmap_shared_dir() -> Path:
    """
    Get shared directory for pmap work units.

    Priority:
    1. PHASIC_PMAP_SHARED_DIR environment variable
    2. ~/.phasic_cache/pmap_work (for local multi-GPU)
    3. /tmp/phasic_pmap (last resort, single-machine only)

    Returns
    -------
    Path
        Directory for storing work unit files
    """
    import os
    from pathlib import Path

    # Check environment variable first (for distributed clusters)
    env_dir = os.environ.get('PHASIC_PMAP_SHARED_DIR')
    if env_dir:
        shared_dir = Path(env_dir)
        logger.debug(f"Using PHASIC_PMAP_SHARED_DIR: {shared_dir}")
        return shared_dir

    # Use cache directory for local multi-GPU
    cache_dir = Path.home() / ".phasic_cache" / "pmap_work"
    if cache_dir.parent.exists():
        logger.debug(f"Using cache directory for pmap: {cache_dir}")
        return cache_dir

    # Fallback to temp directory (single-machine only)
    temp_dir = Path(tempfile.gettempdir()) / "phasic_pmap"
    logger.debug(f"Using temp directory for pmap: {temp_dir}")
    return temp_dir


# Module-level per-process file cache
_pmap_file_cache: Dict[str, Tuple[str, str]] = {}


def _write_work_unit_to_file(work_dir: Path, idx: int,
                             graph_hash: str, json_str: str) -> str:
    """
    Write work unit to file.

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
    str
        File path for this work unit
    """
    file_path = work_dir / f"work_{idx}.json"

    data = {
        'hash': graph_hash,
        'json_str': json_str
    }

    with open(file_path, 'w') as f:
        json.dump(data, f)

    logger.debug(f"Wrote work unit {idx} to {file_path} ({len(json_str)} bytes)")
    return str(file_path)


def _load_work_unit_from_file(file_path: str) -> Tuple[str, str]:
    """
    Load work unit from file with per-process caching.

    Parameters
    ----------
    file_path : str
        Path to work unit file

    Returns
    -------
    Tuple[str, str]
        (graph_hash, json_str)
    """
    # Check per-process cache first
    if file_path in _pmap_file_cache:
        logger.debug(f"File cache hit: {file_path}")
        return _pmap_file_cache[file_path]

    # Load from disk
    logger.debug(f"Loading work unit from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    result = (data['hash'], data['json_str'])

    # Cache for future calls in this process
    _pmap_file_cache[file_path] = result

    return result
```

#### Modified: `_record_trace_callback()` (Lines 363-432)

```python
def _record_trace_callback(idx: int, use_files: bool = False,
                          file_path: str = "") -> Tuple[str, 'EliminationTrace']:
    """
    Pure Python callback for trace recording (called from JAX via pure_callback).

    Parameters
    ----------
    idx : int
        Index into _work_unit_store (vmap mode) or padding sentinel (-1)
    use_files : bool, default=False
        If True, load work unit from file_path instead of _work_unit_store
    file_path : str, default=""
        Path to work unit file (pmap mode only)

    Returns
    -------
    Tuple[str, EliminationTrace]
        (graph_hash, computed_trace) or ("", None) for padding
    """
    from .trace_elimination import record_elimination_trace
    from phasic import Graph

    # Handle padding indices (used by pmap)
    idx_int = int(idx)
    if idx_int < 0:
        return ("", None)  # Padding sentinel

    # Load work unit: from file (pmap) or global dict (vmap)
    if use_files:
        if not file_path:
            raise ValueError("pmap mode requires file_path, got empty string")
        graph_hash, json_str = _load_work_unit_from_file(file_path)
    else:
        # vmap mode - use global dict (UNCHANGED)
        graph_hash, json_str = _work_unit_store[idx_int]

    # ... rest of function unchanged (cache check, deserialization, trace recording) ...
```

#### Keep vmap Section UNCHANGED (Lines 537-588)

**NO MODIFICATIONS** - vmap works perfectly with global dict.

```python
    # ========================================================================
    # VMAP Strategy: Single machine, vectorized processing with JAX
    # ========================================================================
    if strategy == 'vmap':
        # ... existing vmap implementation (KEEP AS-IS) ...
```

#### Rewrite pmap Section (Lines 590-660)

```python
    # ========================================================================
    # PMAP Strategy: Multi-device parallelization with disk serialization
    # ========================================================================
    elif strategy == 'pmap':
        import time
        import shutil

        n_devices = jax.device_count()
        logger.info("PMAP: Using JAX pmap over %d devices with disk serialization", n_devices)

        # Step 1: Get shared directory for work unit files
        shared_dir = _get_pmap_shared_dir()

        # Step 2: Create unique session directory
        session_id = f"pmap_{os.getpid()}_{int(time.time() * 1000)}"
        work_dir = shared_dir / session_id

        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created pmap work directory: {work_dir}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create pmap work directory {work_dir}: {e}\n"
                "  Ensure PHASIC_PMAP_SHARED_DIR points to writable shared storage\n"
                "  Or use strategy='vmap' for single-machine parallelization"
            ) from e

        # Step 3: Write work units to disk
        file_paths = []
        for idx, (graph_hash, json_str) in enumerate(work_units.items()):
            file_path = _write_work_unit_to_file(work_dir, idx, graph_hash, json_str)
            file_paths.append(file_path)

        logger.info(f"Wrote {len(file_paths)} work units to {work_dir}")

        # Show progress bar
        if verbose:
            pbar = tqdm(
                total=0,
                desc=f"Computing {len(work_units)} traces (pmap, {n_devices} devices)",
                bar_format="{desc}: {elapsed}",
                leave=False
            )

        try:
            # Step 4: Define JAX-compatible wrapper with file loading
            def _compute_trace_jax_pmap(file_path_bytes):
                """JAX-compatible trace computation using pure_callback with disk I/O."""
                result_shape = jax.ShapeDtypeStruct((), jnp.int32)

                def _callback_impl(fp_bytes):
                    # Decode file path from bytes
                    file_path = fp_bytes.tobytes().decode('utf-8').rstrip('\x00')

                    if not file_path or file_path == '\x00' * len(file_path):
                        # Padding entry
                        return np.array(-1, dtype=np.int32)

                    # Load and compute trace from file
                    graph_hash, trace = _record_trace_callback(
                        idx=0,  # Unused when use_files=True
                        use_files=True,
                        file_path=file_path
                    )

                    # Return success indicator
                    return np.array(0, dtype=np.int32)

                return jax.pure_callback(
                    _callback_impl,
                    result_shape,
                    file_path_bytes,
                    vmap_method='sequential'
                )

            # Step 5: Convert file paths to fixed-length byte arrays
            max_path_len = max(len(fp) for fp in file_paths) + 1
            file_path_arrays = []
            for fp in file_paths:
                # Encode as bytes and pad to max length
                fp_bytes = fp.encode('utf-8')
                padded = fp_bytes + b'\x00' * (max_path_len - len(fp_bytes))
                file_path_arrays.append(np.frombuffer(padded, dtype=np.uint8))

            # Step 6: Pad to be divisible by n_devices
            n_work_units = len(file_path_arrays)
            padded_size = ((n_work_units + n_devices - 1) // n_devices) * n_devices

            # Add padding entries (all zeros)
            for _ in range(padded_size - n_work_units):
                file_path_arrays.append(np.zeros(max_path_len, dtype=np.uint8))

            # Step 7: Stack into JAX array and reshape for pmap
            file_paths_jax = jnp.array(np.stack(file_path_arrays))
            # Shape: (n_devices, work_per_device, max_path_len)
            reshaped_paths = file_paths_jax.reshape(n_devices, -1, max_path_len)

            # Step 8: Apply pmap over devices
            pmapped_compute = jax.pmap(jax.vmap(_compute_trace_jax_pmap))
            completed = pmapped_compute(reshaped_paths)

            logger.info("PMAP computation completed")

        finally:
            if verbose:
                pbar.close()

        # Step 9: Collect results from trace cache (workers saved traces there)
        results = {}
        for idx, (graph_hash, _) in enumerate(work_units.items()):
            trace = _load_trace_from_cache(graph_hash)
            if trace is None:
                logger.error(f"PMAP: Trace not found in cache after computation: {graph_hash[:16]}")
                raise RuntimeError(
                    f"Missing trace for {graph_hash} after pmap computation.\n"
                    "This usually means a worker failed silently."
                )
            results[graph_hash] = trace
            logger.debug("PMAP: Loaded completed trace for %s", graph_hash[:16])

        # Step 10: Cleanup - remove work directory
        try:
            shutil.rmtree(work_dir)
            logger.debug(f"Cleaned up pmap work directory: {work_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup pmap work directory {work_dir}: {e}")

        # Clear per-process file cache
        _pmap_file_cache.clear()

        return results
```

---

## Testing Strategy

### Unit Tests

#### Test 1: AST Hashing Stability
```python
def test_ast_hash_formatting_changes():
    """Test that formatting changes don't affect hash."""

    def callback_v1(state):
        n = state[0]
        if n > 0:
            return [[[n-1], [n]]]
        return []

    def callback_v2(state):
        n = state[0]
        if n > 0:
            return [[[n-1], [n]]]  # Extra whitespace
        return []

    from phasic.callback_hash import hash_callback

    hash1 = hash_callback(callback_v1, nr_samples=10)
    hash2 = hash_callback(callback_v2, nr_samples=10)

    assert hash1 == hash2, "Formatting changes should not affect hash"
```

#### Test 2: Closure Detection
```python
def test_closure_rejection():
    """Test that closures are rejected with clear error."""

    theta = 2.0
    def callback(state):
        return [[[state[0] - 1], [theta]]]  # Captures theta

    from phasic.callback_hash import hash_callback

    with pytest.raises(ValueError, match="closures"):
        hash_callback(callback, nr_samples=10)
```

#### Test 3: Graph Cache Save/Load
```python
def test_graph_cache_save_load():
    """Test saving and loading graphs from cache."""

    def callback(state, theta=1.0):
        n = state[0]
        if n <= 1:
            return []
        return [[[n-1], [n*(n-1)/2 * theta]]]

    from phasic import Graph
    from phasic.graph_cache import GraphCache

    cache = GraphCache()

    # Build graph and save to cache
    graph1 = Graph(callback, nr_samples=5, theta=1.0)
    hash_key = cache.save_graph(graph1, callback, nr_samples=5, theta=1.0)

    # Load from cache
    graph2 = cache.load_graph(callback, nr_samples=5, theta=1.0)

    assert graph2 is not None, "Should load from cache"
    assert graph2.vertices_length() == graph1.vertices_length()
```

#### Test 4: Graph.__init__() with Caching
```python
def test_graph_init_caching():
    """Test Graph.__init__() cache integration."""

    def callback(state, theta=1.0):
        n = state[0]
        if n <= 1:
            return []
        return [[[n-1], [n*(n-1)/2 * theta]]]

    from phasic import Graph, graph_cache

    # Clear cache first
    graph_cache.clear_graph_cache()

    # First build - cache miss
    import time
    start = time.time()
    graph1 = Graph(callback, cache=True, nr_samples=20, theta=1.0)
    time1 = time.time() - start

    # Second build - cache hit (should be much faster)
    start = time.time()
    graph2 = Graph(callback, cache=True, nr_samples=20, theta=1.0)
    time2 = time.time() - start

    assert time2 < time1 * 0.5, "Cache hit should be at least 2x faster"
    assert graph2.vertices_length() == graph1.vertices_length()
```

#### Test 5: pmap File Writing/Loading
```python
def test_pmap_work_unit_files():
    """Test writing and loading work unit files."""

    import tempfile
    from pathlib import Path
    from phasic.hierarchical_trace_cache import (
        _write_work_unit_to_file,
        _load_work_unit_from_file
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)

        # Write work unit
        file_path = _write_work_unit_to_file(
            work_dir, 0, "abc123", '{"test": "data"}'
        )

        # Load work unit
        hash_val, json_str = _load_work_unit_from_file(file_path)

        assert hash_val == "abc123"
        assert json_str == '{"test": "data"}'
```

### Integration Tests

#### Test 6: pmap with Disk Serialization (Local)
```python
def test_pmap_disk_serialization_local():
    """Test pmap with disk serialization on local multi-GPU."""

    # Requires: XLA_FLAGS="--xla_force_host_platform_device_count=2"

    from phasic import Graph
    from phasic.hierarchical_trace_cache import get_trace_hierarchical
    import phasic

    def callback(state, theta=1.0):
        n = state[0]
        if n <= 1:
            return []
        return [[[n-1], [n*(n-1)/2 * theta]]]

    # Build graph
    graph = Graph(callback, nr_samples=10, theta=1.0)

    # Clear caches
    phasic.clear_caches()

    # Compute trace with pmap
    trace = get_trace_hierarchical(
        graph,
        parallel_strategy='pmap',
        min_size=10
    )

    # Verify correctness
    assert trace is not None
    assert trace.n_vertices > 0
    assert len(trace.operations) > 0
```

#### Test 7: Cached Graph → pmap Traces
```python
def test_cached_graph_to_pmap_traces():
    """Test complete workflow: cache graph, then compute traces with pmap."""

    from phasic import Graph
    import phasic

    def expensive_callback(state, N=50):
        # Simulate expensive callback
        n = state[0]
        if n <= 1:
            return []
        return [[[n-1], [n*(n-1)/2]]]

    # Clear all caches
    phasic.clear_caches()

    # Step 1: Build graph with caching (cache miss)
    graph1 = Graph(expensive_callback, cache=True, nr_samples=N)

    # Step 2: Rebuild graph (cache hit - should be fast)
    graph2 = Graph(expensive_callback, cache=True, nr_samples=N)

    # Step 3: Compute traces with pmap
    trace = graph2.compute_trace(
        hierarchical=True,
        parallel_strategy='pmap',
        min_size=20
    )

    assert trace is not None
    assert graph2.vertices_length() == graph1.vertices_length()
```

#### Test 8: Rabbits Model with Caching + pmap
```python
def test_rabbits_cache_pmap():
    """Test rabbits model with graph caching and pmap trace computation."""

    import phasic
    import numpy as np

    @phasic.callback([2, 0])
    def rabbits(state):
        left, right = state
        transitions = []
        if left:
            transitions.append([[left - 1, right + 1], [left, 0, 0]])
            transitions.append([[0, right], [0, 1, 0]])
        if right:
            transitions.append([[left + 1, right - 1], [right, 0, 0]])
            transitions.append([[left, 0], [0, 0, 1]])
        return transitions

    # Build graph with caching
    graph = phasic.Graph(rabbits, cache=True)
    graph.update_weights([1.0, 2.0, 4.0])

    # Generate data
    observed_data = np.array(graph.sample(50))

    # Run SVGD with pmap
    result = graph.svgd(
        observed_data=observed_data,
        theta_dim=3,
        n_particles=2,
        n_iterations=2,
        parallel='pmap',  # Use pmap!
        verbose=True
    )

    assert result is not None
    assert result.particles.shape == (2, 3)
```

---

## Implementation Checklist

### Phase 1: Callback Hashing (~2-3 hours)
- [ ] Create `src/phasic/callback_hash.py`
- [ ] Implement `hash_callback()` with AST normalization
- [ ] Implement `_normalize_ast()`
- [ ] Implement `_detect_closures()` with helpful error
- [ ] Add version tagging (`PHASIC_CALLBACK_VERSION`)
- [ ] Write unit tests for AST hashing
- [ ] Test closure detection and rejection
- [ ] Test edge cases (decorators, methods, lambdas)

### Phase 2: Graph Cache (~3-4 hours)
- [ ] Create `src/phasic/graph_cache.py`
- [ ] Implement `GraphCache` class
- [ ] Implement `save_graph()` with JSON serialization
- [ ] Implement `load_graph()` with version checking
- [ ] Implement `get_cache_stats()`, `clear_graph_cache()`
- [ ] Create cache directory structure
- [ ] Write unit tests for graph cache
- [ ] Test save/load correctness

### Phase 3: Graph.__init__() Integration (~1-2 hours)
- [ ] Modify `src/phasic/__init__.py` (Graph class)
- [ ] Add `cache=False, force_rebuild=False` parameters
- [ ] Implement `_load_from_cache()` method
- [ ] Implement `_save_to_cache()` method
- [ ] Implement `_copy_from_cached()` method
- [ ] Write integration tests
- [ ] Test cache hit/miss performance

### Phase 4: pmap Disk Serialization (~4-5 hours)
- [ ] Modify `src/phasic/hierarchical_trace_cache.py`
- [ ] Add `_get_pmap_shared_dir()` helper
- [ ] Add `_write_work_unit_to_file()` helper
- [ ] Add `_load_work_unit_from_file()` helper
- [ ] Add `_pmap_file_cache` module variable
- [ ] Modify `_record_trace_callback()` for file loading
- [ ] Rewrite pmap section (lines 590-660)
- [ ] Keep vmap unchanged (verify no regressions)
- [ ] Test file writing/loading
- [ ] Test pmap with local multi-GPU

### Phase 5: Integration Testing (~2-3 hours)
- [ ] Test cached graph → pmap workflow
- [ ] Test rabbits model with cache + pmap
- [ ] Test on distributed cluster (if available)
- [ ] Performance benchmarking
- [ ] Error handling verification

### Phase 6: Documentation (~1-2 hours)
- [ ] Update CLAUDE.md with caching examples
- [ ] Document callback requirements (no closures)
- [ ] Add pmap shared storage setup instructions
- [ ] Update API reference
- [ ] Add tutorial notebook

### Phase 7: Cleanup and Review (~1 hour)
- [ ] Code review and refactoring
- [ ] Add logging statements
- [ ] Check error messages are helpful
- [ ] Verify all tests pass
- [ ] Update version number

---

## Performance Targets

### Graph Caching
- Cache miss (first build): Normal speed (no overhead)
- Cache hit (subsequent): **10-100x faster** than building
- Hash computation: < 1ms per callback
- Disk I/O: < 100ms for typical graphs

### pmap Disk Serialization
- File write overhead: < 10ms for 100 work units
- File load per worker: < 1ms per file (cached)
- Total pmap overhead: < 5% vs ideal pmap
- No crashes or global state issues

---

## Risk Mitigation

### Risk 1: AST Hashing Fragility
**Risk**: Python AST structure changes between versions
**Mitigation**: Include Python version in hash, warn on version mismatch
**Fallback**: User can force rebuild with `force_rebuild=True`

### Risk 2: Disk I/O Bottleneck in pmap
**Risk**: NFS/shared filesystem slow reads
**Mitigation**: Per-process file caching, small file sizes
**Fallback**: Use vmap instead (document in error message)

### Risk 3: Cache Corruption
**Risk**: Corrupted cache files break loading
**Mitigation**: Try/except with fallback to rebuild
**Recovery**: `clear_graph_cache()` utility

### Risk 4: Closure False Positives
**Risk**: Decorators create closures even for "clean" callbacks
**Mitigation**: Unwrap decorators before checking `__closure__`
**Documentation**: Document known decorator compatibility

---

## Rollback Plan

If critical issues arise:

1. **Graph caching**: Set `cache=False` by default (opt-in)
2. **pmap**: Revert to vmap-only, document pmap limitation
3. **Complete rollback**: Git revert to pre-implementation commit

All changes are additive (new files + optional parameters), so rollback is clean.

---

## Success Criteria

### Must Have (MVP)
- ✅ AST-based callback hashing works for simple callbacks
- ✅ Graph caching saves/loads correctly
- ✅ Cache hits are significantly faster (>10x)
- ✅ pmap works without crashes (local multi-GPU)
- ✅ vmap continues working (no regressions)
- ✅ Clear error messages for unsupported cases

### Nice to Have
- ✅ Distributed pmap on cluster
- ✅ Cache statistics and cleanup utilities
- ✅ Performance within 5% of ideal
- ✅ Comprehensive test coverage

### Documentation
- ✅ API reference updated
- ✅ Tutorial with examples
- ✅ Troubleshooting guide

---

## Timeline Estimate

**Total**: 14-20 hours of implementation + testing

- **Week 1 (8 hours)**: Phases 1-3 (callback hashing + graph cache + integration)
- **Week 2 (6 hours)**: Phase 4 (pmap disk serialization)
- **Week 3 (4-6 hours)**: Phases 5-7 (testing, docs, cleanup)

---

## Next Steps

1. **Review this plan** with user for approval
2. **Begin Phase 1**: Implement callback hashing
3. **Iterate**: Test each phase before moving to next
4. **Document**: Update CLAUDE.md as features complete

---

**Plan Status**: ✅ IMPLEMENTATION COMPLETE - READY FOR TESTING

**Progress Update - 2025-11-14**:

✅ **Phase 1 COMPLETE**: Added `@wraps` to `@phasic.callback` decorator
- Modified line 1396 in `src/phasic/__init__.py`
- Updated decorator check to use `hasattr(arg, '__wrapped__')` instead of `arg.__name__ != 'wrapper'`

✅ **Phase 2 COMPLETE**: Implemented AST-based callback hashing
- Created `src/phasic/callback_hash.py` (188 lines)
- `hash_callback()`: AST-based hashing with version tagging
- `_normalize_ast()`: Removes whitespace, comments, docstrings
- `_detect_closures()`: Rejects closures with helpful error messages

✅ **Phase 3 COMPLETE**: Implemented graph cache infrastructure
- Created `src/phasic/graph_cache.py` (319 lines)
- `GraphCache` class with save/load/get_or_build methods
- Modified `Graph.__init__()` to add `cache=False, force_rebuild=False` parameters
- Added `_load_from_cache()` and `_save_to_cache()` helper methods to Graph class
- Cache directory: `~/.phasic_cache/graphs/`

✅ **Phase 4 COMPLETE**: Implemented pmap disk serialization
- **Phase 4.1**: Added disk-based helper functions (lines 435-516 in hierarchical_trace_cache.py)
  - `_pmap_file_cache`: Per-process cache dict
  - `_get_pmap_shared_dir()`: Returns `~/.phasic_cache/pmap_work` or `$PHASIC_PMAP_SHARED_DIR`
  - `_write_work_unit_to_file()`: Write work unit to JSON file
  - `_load_work_unit_from_file()`: Load with per-process caching
- **Phase 4.2**: Modified `_record_trace_callback()` (lines 363-432)
  - Added `use_files=False, file_path=""` parameters
  - Supports both vmap (index-based) and pmap (file-based) modes
- **Phase 4.3**: Rewrote pmap section (lines 681-805)
  - Creates session directory with UUID
  - Writes all work units to individual files
  - Converts file paths to JAX-compatible byte arrays
  - Passes paths through pmap instead of indices
  - Workers load from disk with per-process caching
  - Results collected from trace cache on disk
  - Cleanup temp directory after completion

⏳ **Phase 5 IN PROGRESS**: Testing
- Next: Test on local multi-CPU machine, then SLURM cluster

**Decisions Made**:
1. ✅ cache=False by default (opt-in caching)
2. ✅ Test on local multi-CPU first, then SLURM cluster
3. ✅ Add `@wraps` to decorator (enables AST hashing for decorated callbacks)
4. ✅ vmap unchanged (global dict works perfectly for single-process vmap)
5. ✅ pmap uses disk-based distribution (fixes cross-process communication)
