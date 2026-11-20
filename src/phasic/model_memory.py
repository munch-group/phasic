"""Memory held by the native (FFI) likelihood code, per model (ticket T11).

Every distinct model evaluated through the JAX FFI path keeps its parsed
structure and one graph per worker thread in memory so later calls are
fast. That is about 6 KB per vertex on joint-probability graphs, so a
session that evaluates many large models would otherwise keep all of
them until the process exits. The number kept is capped (least recently
used dropped first; ``phasic.configure(model_memory_cache_size=K)``,
default 4), and this module frees it on request.
"""
from __future__ import annotations


def _param():
    from . import phasic_pybind
    return phasic_pybind.parameterized


def clear_model_memory() -> dict:
    """Free the memory the native likelihood code keeps for every model.

    Releases the cached models of the calling thread and of the OpenMP
    worker threads (which hold most of the memory: one graph per model
    per worker) immediately. A model cached by another thread that runs
    native computations (for example one of JAX's own threads) is
    released when that thread next runs one. A computation already in
    progress keeps what it is using until it finishes. The next
    evaluation of any model rebuilds what it needs. The on-disk caches are not touched (see
    ``phasic.clear_caches()``), nor are JAX's own compilation caches
    (``jax.clear_caches()``).

    Returns
    -------
    dict
        ``model_memory_info()`` after the clear.
    """
    p = _param()
    if p._model_memory_available:
        p._model_memory_clear()
    return model_memory_info()


def model_memory_info() -> dict:
    """Counters for the memory the native likelihood code keeps per model.

    Returns
    -------
    dict
        ``ffi_available``: whether this build has the native FFI path (if
        not, nothing is cached and the other entries are 0);
        ``capacity``: how many distinct models may stay cached;
        ``models``: how many are cached now; ``live_builders``: native
        model objects alive (including ones owned by Python objects);
        ``cached_graphs``: per-thread graphs alive, over all threads.
    """
    p = _param()
    if not p._model_memory_available:
        return {'ffi_available': False, 'capacity': 0, 'models': 0,
                'live_builders': 0, 'cached_graphs': 0}
    info = dict(p._model_memory_info())
    info['ffi_available'] = True
    return info
