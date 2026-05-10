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


def _cache_root() -> Path:
    """Return the cache root directory.

    Honours ``PHASIC_CACHE_DIR``: if set, used verbatim. Otherwise
    falls back to ``$HOME/.phasic_cache``. This mirrors the C-side
    ``ptd_cache_root_dir`` helper so Python tools (cache_info,
    clear_cache) operate on the same directory the C cache reads
    and writes.
    """
    override = os.environ.get("PHASIC_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".phasic_cache"


def _param_compute_cache_dir() -> Path:
    """Return the path to the parameterised compute graph cache
    directory. Does not create it."""
    return _cache_root() / "parameterized_reward_compute"


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

    The cache directory holds two kinds of entries:

    - ``<hash>.bin``: parent-graph-level Stage A2 entries
      (populated by ``ptd_precompute_reward_compute_graph``).
    - ``scc_<hash>.bin``: per-SCC entries from the hierarchical
      pipeline (populated by ``ptd_scc_get_or_compute_prc``,
      WP-4 of the ``hierar-elimin-cache`` branch).

    Both kinds live in the same directory because they share the
    same on-disk format (with WP-3's revision-2 extension for SCC
    entries that need EXTERNAL pointers).

    Returns
    -------
    dict
        Keys:

        - ``cache_dir`` (str or None): absolute path to the cache
          directory, or None if HOME is unset.
        - ``n_files`` (int): total number of ``*.bin`` files.
        - ``n_parent_files`` (int): count of parent-level entries
          (filenames not starting with ``scc_``).
        - ``n_scc_files`` (int): count of per-SCC entries
          (filenames starting with ``scc_``).
        - ``total_size`` (int): total size in bytes across all
          files.
        - ``disabled`` (bool): whether ``PHASIC_DISABLE_CACHE=1``
          is set.

    Examples
    --------
    >>> info = param_compute_cache_info()
    >>> info['n_files']                  # doctest: +SKIP
    7
    >>> info['n_parent_files']           # doctest: +SKIP
    3
    >>> info['n_scc_files']              # doctest: +SKIP
    4
    """
    info: dict[str, Any] = {
        "cache_dir": None,
        "n_files": 0,
        "n_parent_files": 0,
        "n_scc_files": 0,
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
            if cache_file.name.startswith("scc_"):
                info["n_scc_files"] += 1
            else:
                info["n_parent_files"] += 1
        except OSError:
            # File disappeared mid-scan (concurrent clear or unlink).
            # Skip silently — the count is informational anyway.
            pass
    return info


def clear_trace_cache() -> int:
    """Delete every cached ``EliminationTrace`` (JSON and pickle).

    .. deprecated::
        The Python ``EliminationTrace`` path is no longer wired to the
        public ``moments()``/``expectation()``/``variance()`` entry
        points (those route directly to the C++ implementation), and
        ``Graph(cache_trace=True)`` is deprecated. Nothing in the
        public API populates ``~/.phasic_cache/traces/`` anymore.
        This helper is kept for cleaning up leftover files from
        previous phasic versions and will be removed in a future
        release.

    Counterpart to :func:`clear_param_compute_cache` for the
    ``~/.phasic_cache/traces/`` subdirectory used by the Python-side
    trace pipeline. Returns the number of files deleted.

    See Also
    --------
    trace_cache_info : inspect the trace cache.
    clear_param_compute_cache : clear the parameterised compute
        graph cache.
    """
    import warnings
    warnings.warn(
        "clear_trace_cache() is deprecated. The Python EliminationTrace "
        "cache it manages is no longer populated by any public API; "
        "this helper now only cleans up leftover files from previous "
        "phasic versions. It will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _clear_trace_cache()


def trace_cache_info() -> dict[str, Any]:
    """Return a summary of the trace cache.

    .. deprecated::
        See :func:`clear_trace_cache` — the Python EliminationTrace
        cache is no longer populated by any public API. This helper
        is kept for inspecting leftover files from previous phasic
        versions and will be removed in a future release.

    Thin re-export of :func:`phasic.trace_serialization.get_cache_info`
    so callers can find both cache helpers in one module.
    """
    import warnings
    warnings.warn(
        "trace_cache_info() is deprecated. The Python EliminationTrace "
        "cache it inspects is no longer populated by any public API. "
        "It will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_trace_cache_info()


def clear_all_caches() -> dict[str, int]:
    """Clear both phasic on-disk caches.

    Returns
    -------
    dict
        ``{'param_compute': N, 'traces': M}`` where ``N`` and ``M``
        are the number of files deleted from each cache.

    Notes
    -----
    The ``traces`` portion is deprecated — nothing in the public API
    populates ``~/.phasic_cache/traces/`` anymore. This call still
    cleans up leftover files for backwards compatibility but emits a
    ``DeprecationWarning`` from the inner :func:`clear_trace_cache`.
    """
    return {
        "param_compute": clear_param_compute_cache(),
        "traces": clear_trace_cache(),
    }


def scc_compose_stats() -> dict[str, Any]:
    """Return SCC composer telemetry counters.

    Counters are process-wide and reflect activity since process
    start (or the last call to :func:`reset_scc_compose_stats`).
    They are only populated when the hierarchical compose path is
    actually invoked — that is, when ``PHASIC_HIERAR_ELIMINATION=1``
    is set, the graph is parameterised, and ``rewards is None``.

    Returns
    -------
    dict
        A dict with the following keys, all integers:

        - ``cache_hits``: number of times the per-SCC PRC was
          loaded from disk
        - ``cache_misses``: number of times the per-SCC PRC was
          recomputed (and saved on first miss)
        - ``compose_calls``: number of ``ptd_compose_scc_prcs``
          invocations (one per top-level
          ``Graph.expected_waiting_time`` taking the hierarchical
          path)
        - ``total_compose_ns``: cumulative wall-clock time spent
          inside ``ptd_compose_scc_prcs`` (nanoseconds, monotonic
          clock)

    Examples
    --------
    >>> import os, phasic.cache as cache
    >>> os.environ["PHASIC_HIERAR_ELIMINATION"] = "1"
    >>> cache.reset_scc_compose_stats()
    >>> # ... run some hierarchical computations ...
    >>> stats = cache.scc_compose_stats()
    >>> hit_rate = stats["cache_hits"] / max(
    ...     stats["cache_hits"] + stats["cache_misses"], 1)
    """
    from phasic.phasic_pybind import _scc_compose_stats_get
    return _scc_compose_stats_get()


def reset_scc_compose_stats() -> None:
    """Reset SCC composer telemetry counters to zero.

    Useful at the start of a benchmark or test that wants to
    measure cache behaviour for a specific block of work.
    """
    from phasic.phasic_pybind import _scc_compose_stats_reset
    _scc_compose_stats_reset()
