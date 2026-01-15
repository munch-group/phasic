"""
Persistent disk cache for Graph objects using callback hashes as keys.

This module provides efficient caching of expensive graph construction
by hashing callback functions and parameters to create stable cache keys.

Cache Directory: ~/.phasic_cache/graphs/
Cache Format: {callback_hash}.json containing serialized graph + metadata

Author: Claude Code
Date: 2025-11-14
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from . import Graph

from .callback_hash import hash_callback
from .logging_config import get_logger

logger = get_logger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".phasic_cache" / "graphs"


class GraphCache:
    """
    Manages persistent disk cache for Graph objects.

    Cache directory: ~/.phasic_cache/graphs/
    Cache key: callback hash (AST-based) + parameters

    Examples
    --------
    >>> from phasic.graph_cache import GraphCache
    >>> import phasic
    >>>
    >>> @phasic.callback([5])
    >>> def callback(state, theta=1.0):
    ...     n = state[0]
    ...     if n <= 1:
    ...         return []
    ...     return [[[n-1], [n*(n-1)/2 * theta]]]
    >>>
    >>> cache = GraphCache()
    >>>
    >>> # Save graph to cache
    >>> graph = phasic.Graph(callback, nr_samples=10, theta=1.0)
    >>> hash_key = cache.save_graph(graph, callback, nr_samples=10, theta=1.0)
    >>>
    >>> # Load graph from cache
    >>> cached = cache.load_graph(callback, nr_samples=10, theta=1.0)
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
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"GraphCache initialized: {self.cache_dir}")

    def save_graph(self, graph: 'Graph', callback, **params) -> str:
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

        Raises
        ------
        ValueError
            If callback uses closures or is not hashable
        IOError
            If cache write fails
        """
        # Compute cache key
        try:
            cache_key = hash_callback(callback, **params)
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot hash callback for caching: {e}")
            raise

        # Serialize graph
        try:
            graph_data = graph.serialize()
        except Exception as e:
            logger.error(f"Failed to serialize graph: {e}")
            raise RuntimeError(f"Graph serialization failed: {e}") from e

        # Add metadata
        import phasic
        cache_entry = {
            'version': phasic.__version__,
            'callback_hash': cache_key,
            'created_at': datetime.now().isoformat(),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'construction_params': _serialize_value(params),
            'graph_data': _serialize_value(graph_data)
        }

        # Write to disk
        cache_path = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_entry, f, indent=2)
            logger.info(f"Saved graph to cache: {cache_key[:16]}... ({graph.vertices_length()} vertices)")
        except IOError as e:
            logger.error(f"Failed to write cache file {cache_path}: {e}")
            raise

        return cache_key

    def load_graph(self, callback, **params) -> Optional['Graph']:
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
        # Compute cache key
        try:
            cache_key = hash_callback(callback, **params)
        except (ValueError, TypeError) as e:
            logger.debug(f"Cannot hash callback for cache lookup: {e}")
            return None

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
            import phasic
            if cache_entry['version'] != phasic.__version__:
                logger.warning(
                    f"Cache version mismatch: cached with {cache_entry['version']}, "
                    f"current is {phasic.__version__}"
                )

            # Deserialize graph data
            graph_data = _deserialize_value(cache_entry['graph_data'])

            # Reconstruct graph
            from phasic import Graph
            graph = Graph.from_serialized(graph_data)

            logger.info(f"Cache hit: {cache_key[:16]}... ({graph.vertices_length()} vertices)")
            return graph

        except Exception as e:
            logger.error(f"Failed to load cached graph {cache_key[:16]}: {e}")
            return None

    def get_or_build(self, callback, **params) -> 'Graph':
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
        # Try loading from cache
        graph = self.load_graph(callback, **params)
        if graph is not None:
            return graph

        # Cache miss - build graph
        from phasic import Graph
        graph = Graph(callback, **params)

        # Save to cache
        try:
            self.save_graph(graph, callback, **params)
        except Exception as e:
            logger.warning(f"Failed to save graph to cache: {e}")

        return graph

    def clear_graph_cache(self) -> int:
        """
        Clear all cached graphs.

        Returns
        -------
        int
            Number of graphs removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        logger.info(f"Cleared {count} cached graphs")
        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            {'num_graphs': int, 'total_size_mb': float, ...}
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'num_graphs': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }


def _serialize_value(value: Any) -> Any:
    """
    Recursively serialize a value for JSON storage.

    Handles:
    - Primitives (int, float, str, bool, None)
    - Collections (list, dict, tuple)
    - NumPy arrays and scalars
    - Custom objects with to_dict() method

    Parameters
    ----------
    value : Any
        Value to serialize

    Returns
    -------
    Any
        JSON-serializable representation

    Raises
    ------
    TypeError
        If value is not serializable and lacks to_dict() method

    Examples
    --------
    >>> # Primitive
    >>> _serialize_value(42)
    42

    >>> # NumPy array
    >>> _serialize_value(np.array([1, 2, 3]))
    {'__type__': 'ndarray', '__data__': [1, 2, 3]}

    >>> # Custom object with to_dict()
    >>> class MyClass:
    ...     def to_dict(self): return {'x': 1}
    >>> _serialize_value(MyClass())
    {'__type__': '__main__.MyClass', '__data__': {'x': 1}}
    """
    # Primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # NumPy arrays
    if isinstance(value, np.ndarray):
        return {'__type__': 'ndarray', '__data__': value.tolist()}

    # NumPy scalars
    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    # Collections - dict
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Collections - list/tuple
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    # Custom objects with to_dict()
    if hasattr(value, 'to_dict') and callable(value.to_dict):
        serialized = value.to_dict()
        # Tag with type info for deserialization
        return {
            '__type__': f'{value.__class__.__module__}.{value.__class__.__name__}',
            '__data__': _serialize_value(serialized)
        }

    # Unsupported type
    raise TypeError(
        f"Cannot serialize parameter of type {type(value).__name__}. "
        f"To make this type cacheable, add to_dict() and from_dict() methods to the class."
    )


def _deserialize_value(value: Any) -> Any:
    """
    Recursively deserialize a value from JSON storage.

    Parameters
    ----------
    value : Any
        Value to deserialize

    Returns
    -------
    Any
        Reconstructed Python object

    Raises
    ------
    TypeError
        If custom class cannot be imported or lacks from_dict() method

    Examples
    --------
    >>> # Primitive
    >>> _deserialize_value(42)
    42

    >>> # NumPy array
    >>> _deserialize_value({'__type__': 'ndarray', '__data__': [1, 2, 3]})
    array([1, 2, 3])

    >>> # Custom class (must be importable)
    >>> _deserialize_value({'__type__': 'mymodule.MyClass', '__data__': {'x': 1}})
    <mymodule.MyClass object>
    """
    # Primitives
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Collections - list
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]

    # Collections - dict
    if isinstance(value, dict):
        # Check for type tags
        if '__type__' in value and '__data__' in value:
            type_name = value['__type__']
            data = _deserialize_value(value['__data__'])

            # NumPy array
            if type_name == 'ndarray':
                return np.array(data)

            # Custom class - import and reconstruct
            module_name, class_name = type_name.rsplit('.', 1)
            try:
                import importlib
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)

                if hasattr(cls, 'from_dict') and callable(cls.from_dict):
                    return cls.from_dict(data)
                else:
                    raise TypeError(
                        f"Class {type_name} has to_dict() but not from_dict(). "
                        f"Both methods are required for deserialization."
                    )
            except (ImportError, AttributeError) as e:
                raise TypeError(
                    f"Cannot deserialize {type_name}: {e}. "
                    f"Ensure the class is importable and has from_dict() class method."
                ) from e
        else:
            # Regular dict
            return {k: _deserialize_value(v) for k, v in value.items()}

    return value


# Module-level convenience functions
def get_cache_dir() -> Path:
    """Get default cache directory."""
    return DEFAULT_CACHE_DIR


def get_graph_cache_stats() -> Dict[str, Any]:
    """
    Get graph cache statistics.

    Convenience wrapper for GraphCache().get_cache_stats().

    Returns
    -------
    dict
        Dictionary with cache statistics:
        - num_graphs: Number of cached graphs
        - total_size_mb: Total cache size in MB
        - cache_dir: Cache directory path

    Examples
    --------
    >>> from phasic import get_graph_cache_stats
    >>> stats = get_graph_cache_stats()
    >>> print(f"Cached graphs: {stats['num_graphs']}")
    """
    cache = GraphCache()
    return cache.get_cache_stats()


def print_graph_cache_info() -> None:
    """
    Print formatted graph cache information.

    Displays statistics about cached Graph objects in a human-readable format.

    Examples
    --------
    >>> from phasic import print_graph_cache_info
    >>> print_graph_cache_info()
    Cache directory: /Users/you/.phasic_cache/graphs
    Cached graphs: 5
    Total size: 2.34 MB
    """
    stats = get_graph_cache_stats()

    print(f"Cache directory: {stats['cache_dir']}")

    if stats['num_graphs'] == 0:
        print("Status: No cached graphs")
    else:
        print(f"Cached graphs: {stats['num_graphs']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")


def clear_all_graph_caches() -> int:
    """
    Clear all graph caches.

    Returns
    -------
    int
        Number of cache files removed
    """
    cache = GraphCache()
    return cache.clear_graph_cache()
