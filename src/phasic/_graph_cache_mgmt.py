"""Local trace-cache management functions for :class:`~phasic.Graph`.

Extracted verbatim from ``Graph`` (Stage-3 WS-C). Pure relocation; bodies
unchanged (lazy local imports). Assigned onto ``Graph`` as class attributes in
``__init__.py`` so they stay direct members (docs/introspection).
"""
from __future__ import annotations


def clear_from_cache(
    self,
    graph_cache: bool = True,
    parameterized_reward_compute: bool = True,
) -> dict[str, int]:
    """Delete this graph's entries from the on-disk caches.

    Parameters
    ----------
    graph_cache : bool, default True
        If True, remove the serialised Graph entry from
        ``~/.phasic_cache/graphs/<callback_hash>.json``. Requires
        that this graph was built from a callback (so the
        callback + construction kwargs are available to recompute
        the same hash the constructor used). Manually constructed
        graphs have no callback-hash key and will raise.
    parameterized_reward_compute : bool, default True
        If True, remove this graph's Stage A2 symbolic elimination
        entry from
        ``~/.phasic_cache/parameterized_reward_compute/<content_hash>.bin``.
        Per-SCC entries (``scc_<synth_hash>.bin``) are *not* touched
        because they are content-addressed by SCC subgraph
        topology and may be shared with other graphs that have the
        same substructure.

    Returns
    -------
    dict[str, int]
        ``{'graph_cache': n, 'parameterized_reward_compute': m}``
        where each value is the number of files actually removed
        (0 if the entry was not present, or if the flag was False).

    Raises
    ------
    ValueError
        If neither flag is True.
    RuntimeError
        If ``graph_cache=True`` was requested but no callback is
        stored on this instance (e.g. graph built via
        ``Graph(state_length)`` then populated manually).

    Notes
    -----
    Missing files are not an error — they just mean the graph was
    never cached (or has already been cleared). The only conditions
    that raise are user errors (no flags, no callback key
    available).

    For bulk operations across many graphs prefer
    :func:`phasic.cache.clear_param_compute_cache` and
    :func:`phasic.clear_all_graph_caches`.

    Examples
    --------
    >>> g = Graph(model, indexer=indexer, theta_dim=1)
    >>> g.update_weights([1.0])
    >>> _ = g.expectation()  # populates parameterized_reward_compute
    >>> g.clear_from_cache(graph_cache=False) # keep graph cache
    {'graph_cache': 0, 'parameterized_reward_compute': 1}
    """
    if not graph_cache and not parameterized_reward_compute:
        raise ValueError(
            "clear_from_cache: specify at least one of "
            "graph_cache=True or parameterized_reward_compute=True"
        )

    removed = {"graph_cache": 0, "parameterized_reward_compute": 0}

    if graph_cache:
        if self._callback is None:
            raise RuntimeError(
                "clear_from_cache(graph_cache=True): this graph "
                "was not built from a callback, so it has no "
                "callback-hash key in ~/.phasic_cache/graphs/."
            )
        from .graph_cache import GraphCache
        from .callback_hash import hash_callback
        cache_key = hash_callback(self._callback, **self._callback_kwargs)
        cache_path = GraphCache().cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            cache_path.unlink()
            removed["graph_cache"] = 1

    if parameterized_reward_compute:
        from .phasic_pybind import hash as _hash_mod
        from .cache import _cache_root
        hex_hash = _hash_mod.compute_graph_hash(self).hash_hex
        cache_path = (
            _cache_root()
            / "parameterized_reward_compute"
            / f"{hex_hash}.bin"
        )
        if cache_path.exists():
            cache_path.unlink()
            removed["parameterized_reward_compute"] = 1

    return removed

def prewarm_cache(self) -> None:
    """Populate the on-disk Stage A2 cache for this graph. If parameterized, 
    update_weights must be called beforehand.

    Triggers the C-side symbolic elimination
    (``ptd_precompute_reward_compute_graph``) and writes its result to
    ``~/.phasic_cache/parameterized_reward_compute/<content_hash>.bin``.
    Subsequent processes (notebook restart, SLURM workers, fresh CLI
    runs) that build the same graph will load the cached elimination
    instead of redoing the O(n^3) work.

    If ``phasic.configure(parallel_elimination=True)`` is active and
    the graph is parameterised, the hierarchical SCC composer takes
    over: it writes one ``scc_<synth_hash>.bin`` entry per SCC
    large enough to cache (``PHASIC_MIN_SCC_SIZE_TO_CACHE``) and
    *skips* the monolithic parent ``<content_hash>.bin``. Future
    runs that want to benefit from the cache must also have
    ``parallel_elimination=True`` active — the parent file and SCC
    files are populated by different code paths and not
    interchangeable. Run ``prewarm_cache`` once under each
    configuration you plan to consume from.

    Notes
    -----
    Cost is one full elimination plus one waiting-time read; the 
    cache write is a side effect of the elimination.

    Examples
    --------
    Pre-warm on the head node before farming out to SLURM workers:

    >>> g = Graph(model, indexer=indexer, theta_dim=2)
    >>> g.prewarm_cache()
    >>> # ~/.phasic_cache/parameterized_reward_compute/<hash>.bin now exists
    >>> # Workers that rebuild the same graph will load it instantly

    Pre-warm SCC cache too (cyclic graph):

    >>> from phasic import configure
    >>> with configure(parallel_elimination=True):
    ...     g.prewarm_cache(theta=[1.0, 0.5])
    ...     # scc_*.bin entries are now also on disk
    """
    if self._last_theta is None:
        raise RuntimeError(
            "prewarm_cache: graph is parameterised but no theta "
            "has been set. Pass theta=... or call "
            "update_weights(theta) first."
        )
    _ = self.expectation()
