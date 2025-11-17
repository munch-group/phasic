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
            'construction_params': params,
            'graph_data': _serialize_numpy(graph_data)
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
            graph_data = _deserialize_numpy(cache_entry['graph_data'])

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


def _serialize_numpy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to lists for JSON serialization."""
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _deserialize_numpy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lists back to numpy arrays after JSON deserialization."""
    result = {}
    for key, value in data.items():
        if key in ('states', 'edges', 'start_edges', 'param_edges', 'start_param_edges'):
            result[key] = np.array(value, dtype=np.float64 if 'edges' in key else np.int32)
        else:
            result[key] = value
    return result


# Module-level convenience functions
def get_cache_dir() -> Path:
    """Get default cache directory."""
    return DEFAULT_CACHE_DIR


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
