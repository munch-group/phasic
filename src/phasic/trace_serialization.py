"""
Trace serialization for hierarchical caching

This module provides disk caching for EliminationTrace objects using C-level JSON serialization.
Traces are stored in ~/.phasic_cache/traces/ for fast retrieval across sessions.

Implementation
--------------
Uses C functions ptd_load_trace_from_cache() and ptd_save_trace_to_cache()
exposed through pybind11 bindings. The C layer handles JSON serialization,
making traces compatible across Python, C++, and R.

Environment Variables
---------------------
PHASIC_DISABLE_CACHE : str
    Set to "1" to disable all cache operations
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional

from .trace_elimination import EliminationTrace, Operation, OpType
from .logging_config import get_logger

logger = get_logger(__name__)


# Import C bindings
try:
    from .phasic_pybind import (
        _c_load_trace_from_cache,
        _c_save_trace_to_cache,
        _c_elimination_trace_destroy,
        _c_trace_get_n_vertices,
        _c_trace_get_state_length,
        _c_trace_get_param_length,
        _c_trace_get_starting_vertex_idx,
        _c_trace_get_is_discrete,
        _c_trace_get_operations_length,
        _c_trace_get_states,
        _c_trace_get_vertex_rates,
        _c_trace_get_edge_probs,
        _c_trace_get_vertex_targets,
        _c_trace_get_operation,
    )
    _HAS_C_BINDINGS = True
except ImportError as e:
    logger.warning(f"C bindings not available, cache will be disabled: {e}")
    _HAS_C_BINDINGS = False


# Map C operation types to Python OpType enum
_C_OP_TYPE_MAP = {
    0: OpType.CONST,     # PTD_OP_CONST
    1: OpType.PARAM,     # PTD_OP_PARAM
    2: OpType.DOT,       # PTD_OP_DOT
    3: OpType.ADD,       # PTD_OP_ADD
    4: OpType.MUL,       # PTD_OP_MUL
    5: OpType.DIV,       # PTD_OP_DIV
    6: OpType.INV,       # PTD_OP_INV
    7: OpType.SUM,       # PTD_OP_SUM
}

_PYTHON_OP_TYPE_MAP = {v: k for k, v in _C_OP_TYPE_MAP.items()}


def is_cache_disabled() -> bool:
    """
    Check if cache is disabled via environment variable.

    Returns
    -------
    bool
        True if PHASIC_DISABLE_CACHE=1 or C bindings unavailable
    """
    if not _HAS_C_BINDINGS:
        return True
    return os.environ.get("PHASIC_DISABLE_CACHE", "0") == "1"


def _c_trace_to_python(trace_ptr: int) -> Optional[EliminationTrace]:
    """
    Convert C trace struct to Python EliminationTrace.

    Parameters
    ----------
    trace_ptr : int
        Pointer to C struct ptd_elimination_trace

    Returns
    -------
    EliminationTrace or None
        Python trace object, or None on error
    """
    if trace_ptr == 0:
        return None

    try:
        # Get metadata
        n_vertices = _c_trace_get_n_vertices(trace_ptr)
        state_length = _c_trace_get_state_length(trace_ptr)
        param_length = _c_trace_get_param_length(trace_ptr)
        starting_vertex_idx = _c_trace_get_starting_vertex_idx(trace_ptr)
        is_discrete = _c_trace_get_is_discrete(trace_ptr)
        operations_length = _c_trace_get_operations_length(trace_ptr)

        # Get states
        states = _c_trace_get_states(trace_ptr)

        # Get vertex rates
        vertex_rates = _c_trace_get_vertex_rates(trace_ptr)

        # Get edge probs (list of lists)
        edge_probs_raw = _c_trace_get_edge_probs(trace_ptr)
        edge_probs = [[int(idx) for idx in vertex_edges] for vertex_edges in edge_probs_raw]

        # Get vertex targets (list of lists)
        vertex_targets_raw = _c_trace_get_vertex_targets(trace_ptr)
        vertex_targets = [[int(idx) for idx in targets] for targets in vertex_targets_raw]

        # Get operations
        operations = []
        for i in range(operations_length):
            op_dict = _c_trace_get_operation(trace_ptr, i)

            # Convert to Python Operation
            op_type = _C_OP_TYPE_MAP.get(op_dict['op_type'], OpType.CONST)

            # Build operation
            if op_type == OpType.CONST:
                op = Operation(
                    op_type=OpType.CONST,
                    value=op_dict['const_value']
                )
            elif op_type == OpType.PARAM:
                op = Operation(
                    op_type=OpType.PARAM,
                    operands=[op_dict['param_idx']]
                )
            elif op_type == OpType.DOT:
                op = Operation(
                    op_type=OpType.DOT,
                    coefficients=list(op_dict['coefficients']),
                    operands=list(op_dict['operands'])
                )
            else:
                # Binary/unary operations
                op = Operation(
                    op_type=op_type,
                    operands=list(op_dict['operands'])
                )

            operations.append(op)

        # Create EliminationTrace
        trace = EliminationTrace(
            operations=operations,
            vertex_rates=vertex_rates,
            edge_probs=edge_probs,
            vertex_targets=vertex_targets,
            states=states,
            starting_vertex_idx=starting_vertex_idx,
            n_vertices=n_vertices,
            state_length=state_length,
            param_length=param_length,
            reward_length=0,  # Not stored in C trace
            is_discrete=is_discrete,
            metadata={}
        )

        return trace

    except Exception as e:
        logger.error(f"Failed to convert C trace to Python: {e}")
        return None


def load_trace_from_cache(hash_hex: str) -> Optional[EliminationTrace]:
    """
    Load elimination trace from disk cache.

    Tries multiple formats in order:
    1. JSON (C-level cache, cross-language compatible)
    2. Pickle (Python-level cache)

    Parameters
    ----------
    hash_hex : str
        64-character hexadecimal hash identifying the trace

    Returns
    -------
    EliminationTrace or None
        Loaded trace, or None if not found or error

    Examples
    --------
    >>> from phasic import hash as phasic_hash
    >>> hash_result = phasic_hash.compute_graph_hash(graph)
    >>> trace = load_trace_from_cache(hash_result.hash_hex)
    >>> if trace is not None:
    ...     print("Cache hit!")
    """
    if hash_hex is None:
        return None

    if is_cache_disabled():
        return None

    # Try C JSON cache first (if available)
    if _HAS_C_BINDINGS:
        try:
            trace_ptr = _c_load_trace_from_cache(hash_hex)

            if trace_ptr != 0:
                # Convert to Python
                trace = _c_trace_to_python(trace_ptr)

                # Free C memory
                _c_elimination_trace_destroy(trace_ptr)

                if trace is not None:
                    logger.debug(f"Loaded trace from C cache (JSON): {hash_hex}")
                    return trace
        except Exception as e:
            logger.debug(f"C cache load failed, trying pickle: {e}")

    # Try pickle cache as fallback
    home = Path.home()
    cache_dir = home / ".phasic_cache" / "traces"
    cache_file = cache_dir / f"{hash_hex}.pkl"

    if cache_file.exists():
        try:
            import pickle

            with open(cache_file, 'rb') as f:
                trace = pickle.load(f)
            logger.debug(f"Loaded trace from pickle cache: {hash_hex}")
            return trace
        except Exception as e:
            logger.warning(f"Failed to load pickle cache: {e}")
            # Delete corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass

    return None  # Cache miss


def save_trace_to_cache(hash_hex: str, trace: EliminationTrace) -> bool:
    """
    Save elimination trace to disk cache.

    Traces are stored in ~/.phasic_cache/traces/ as pickle files.

    Note: Currently uses pickle serialization because C API doesn't provide
    a way to manually construct ptd_elimination_trace structs. Phase 3b will
    add C-level trace construction for cross-language compatibility.

    Parameters
    ----------
    hash_hex : str
        64-character hexadecimal hash identifying the trace
    trace : EliminationTrace
        The trace to save

    Returns
    -------
    bool
        True on success, False on error or if cache disabled

    Examples
    --------
    >>> from phasic import hash as phasic_hash
    >>> from phasic.trace_elimination import record_elimination_trace
    >>>
    >>> trace = record_elimination_trace(graph, param_length=2)
    >>> hash_result = phasic_hash.compute_graph_hash(graph)
    >>> success = save_trace_to_cache(hash_result.hash_hex, trace)
    """
    if hash_hex is None or trace is None:
        return False

    if is_cache_disabled():
        return False

    # Get cache directory
    home = Path.home()
    cache_dir = home / ".phasic_cache" / "traces"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to create cache directory: {e}")
        return False

    cache_file = cache_dir / f"{hash_hex}.pkl"

    try:
        import pickle

        # Write to temporary file first, then rename atomically
        temp_file = cache_file.with_suffix('.tmp')

        with open(temp_file, 'wb') as f:
            pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)

        temp_file.rename(cache_file)
        logger.debug(f"Saved trace to cache (pickle): {hash_hex}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save trace to cache: {e}")
        # Clean up temp file if it exists
        try:
            if temp_file.exists():
                temp_file.unlink()
        except:
            pass
        return False


def clear_cache() -> int:
    """
    Clear all cached traces (both JSON and pickle).

    Returns
    -------
    int
        Number of cache files deleted
    """
    home = Path.home()
    cache_dir = home / ".phasic_cache" / "traces"

    if not cache_dir.exists():
        return 0

    count = 0
    try:
        # Clear both .json (C cache) and .pkl (Python cache)
        for pattern in ["*.json", "*.pkl"]:
            for cache_file in cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    count += 1
                except:
                    pass
        logger.info(f"Cleared {count} traces from cache")
        return count
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")
        return count


def get_cache_info() -> dict:
    """
    Get information about the cache.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'cache_dir': str, path to cache directory
        - 'n_traces_json': int, number of JSON traces (C cache)
        - 'n_traces_pickle': int, number of pickle traces (Python cache)
        - 'total_size': int, total size in bytes
        - 'disabled': bool, whether cache is disabled
        - 'c_bindings_available': bool, whether C bindings are available
    """
    info = {
        'cache_dir': None,
        'n_traces_json': 0,
        'n_traces_pickle': 0,
        'total_size': 0,
        'disabled': is_cache_disabled(),
        'c_bindings_available': _HAS_C_BINDINGS
    }

    home = Path.home()
    cache_dir = home / ".phasic_cache" / "traces"

    if not cache_dir.exists():
        return info

    info['cache_dir'] = str(cache_dir)

    try:
        json_files = list(cache_dir.glob("*.json"))
        pkl_files = list(cache_dir.glob("*.pkl"))

        info['n_traces_json'] = len(json_files)
        info['n_traces_pickle'] = len(pkl_files)

        all_files = json_files + pkl_files
        info['total_size'] = sum(f.stat().st_size for f in all_files)
    except Exception as e:
        logger.warning(f"Failed to get cache info: {e}")

    return info
