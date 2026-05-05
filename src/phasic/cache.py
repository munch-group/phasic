"""On-disk cache management for phasic.

phasic maintains two on-disk caches under ``~/.phasic_cache/``:

- ``~/.phasic_cache/traces/`` — Python ``EliminationTrace`` objects
  serialised as JSON or pickle. Used by ``Graph(cache_trace=True)``
  and the trace-based codegen pipeline. Managed via
  :func:`clear_trace_cache` / :func:`trace_cache_info`.

- ``~/.phasic_cache/parameterized_reward_compute/`` — C-level
  symbolic elimination output (``parameterized_reward_compute_graph``)
  serialised by ``ptd_save_parameterized_reward_compute_graph``. Used
  transparently by the FFI / pybind11 forward path; consulted on
  every fresh process before running the O(n^3) Gaussian
  elimination. Managed via :func:`clear_param_compute_cache` /
  :func:`param_compute_cache_info`.

Both caches are user-owned. Phasic does not auto-prune or
size-cap. Both honour ``PHASIC_DISABLE_CACHE=1`` to skip reads and
writes.

Example
-------
>>> import phasic.cache as cache
>>> cache.param_compute_cache_info()
{'cache_dir': '/Users/...', 'n_files': 3, 'total_size': 142336, 'disabled': False}
>>> cache.clear_param_compute_cache()
3
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Re-export the existing trace-cache helpers under more specific
# names so callers have a single place to look for cache management.
from .trace_serialization import clear_cache as _clear_trace_cache
from .trace_serialization import get_cache_info as _get_trace_cache_info
from .trace_serialization import is_cache_disabled

logger = logging.getLogger(__name__)

__all__ = [
    "is_cache_disabled",
    "clear_param_compute_cache",
    "param_compute_cache_info",
    "clear_trace_cache",
    "trace_cache_info",
    "clear_all_caches",
]


def _param_compute_cache_dir() -> Path:
    """Return the path to the parameterised compute graph cache
    directory. Does not create it."""
    return Path.home() / ".phasic_cache" / "parameterized_reward_compute"


def clear_param_compute_cache() -> int:
    """Delete every cached parameterised reward compute graph file.

    The cache directory itself is left in place. Cache misses (no
    files present) return 0; per-file deletion errors are logged but
    do not abort the bulk clear.

    Returns
    -------
    int
        Number of cache files successfully deleted.

    Notes
    -----
    The cache is keyed by the deterministic SHA-256 of graph
    structure + coefficients (``ptd_graph_content_hash``), so any
    structural change to a model produces a fresh key. Clearing the
    cache is rarely necessary in normal use — the main reasons are:

    - Reclaim disk space if you have many one-off models cached.
    - Force regeneration after upgrading phasic if you want every
      cache file rewritten with the new build's exact output (the
      file format has a magic + version header that detects
      genuine layout changes automatically, so this is usually
      unnecessary).
    - Test setups that need a clean slate.

    See Also
    --------
    param_compute_cache_info : inspect the cache contents.
    clear_trace_cache : clear the parallel ``traces/`` cache.
    """
    cache_dir = _param_compute_cache_dir()
    if not cache_dir.exists():
        return 0
    count = 0
    for cache_file in cache_dir.glob("*.bin"):
        try:
            cache_file.unlink()
            count += 1
        except OSError as exc:
            logger.warning(
                "Failed to delete %s: %s", cache_file, exc)
    logger.info("Cleared %d parameterised compute graph cache files", count)
    return count


def param_compute_cache_info() -> dict[str, Any]:
    """Return a summary of the parameterised compute graph cache.

    Returns
    -------
    dict
        Keys:

        - ``cache_dir`` (str or None): absolute path to the cache
          directory, or None if HOME is unset.
        - ``n_files`` (int): number of ``*.bin`` files in the
          directory.
        - ``total_size`` (int): total size in bytes.
        - ``disabled`` (bool): whether ``PHASIC_DISABLE_CACHE=1`` is
          set.

    Examples
    --------
    >>> info = param_compute_cache_info()
    >>> info['n_files']
    3
    >>> info['disabled']
    False
    """
    info: dict[str, Any] = {
        "cache_dir": None,
        "n_files": 0,
        "total_size": 0,
        "disabled": is_cache_disabled(),
    }
    home = os.environ.get("HOME")
    if home is None:
        return info
    cache_dir = _param_compute_cache_dir()
    info["cache_dir"] = str(cache_dir)
    if not cache_dir.exists():
        return info
    for cache_file in cache_dir.glob("*.bin"):
        try:
            info["n_files"] += 1
            info["total_size"] += cache_file.stat().st_size
        except OSError:
            # File disappeared mid-scan (concurrent clear or unlink).
            # Skip silently — the count is informational anyway.
            pass
    return info


def clear_trace_cache() -> int:
    """Delete every cached ``EliminationTrace`` (JSON and pickle).

    Counterpart to :func:`clear_param_compute_cache` for the
    ``~/.phasic_cache/traces/`` subdirectory used by the Python-side
    trace pipeline. Returns the number of files deleted.

    See Also
    --------
    trace_cache_info : inspect the trace cache.
    clear_param_compute_cache : clear the parameterised compute
        graph cache.
    """
    return _clear_trace_cache()


def trace_cache_info() -> dict[str, Any]:
    """Return a summary of the trace cache.

    Thin re-export of :func:`phasic.trace_serialization.get_cache_info`
    so callers can find both cache helpers in one module.
    """
    return _get_trace_cache_info()


def clear_all_caches() -> dict[str, int]:
    """Clear both phasic on-disk caches.

    Returns
    -------
    dict
        ``{'param_compute': N, 'traces': M}`` where ``N`` and ``M``
        are the number of files deleted from each cache.
    """
    return {
        "param_compute": clear_param_compute_cache(),
        "traces": clear_trace_cache(),
    }
