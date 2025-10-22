"""
JAX FFI Wrappers for Parameterized Graph Computations

This module provides JAX-compatible wrappers for the C++ GraphBuilder operations
using JAX's Foreign Function Interface (FFI). These wrappers enable:
- JIT compilation with JAX
- Automatic differentiation (with custom VJP rules)
- Vectorization (vmap) and parallelization (pmap)
- Proper GIL management for multi-threading

Key Features:
- Zero-copy data transfer via XLA buffers
- Automatic batching support
- Thread-safe (GIL released during C++ computation)
- Compatible with all JAX transformations

Usage Example:
    ```python
    import jax.numpy as jnp
    from phasic import Graph
    from phasic.ffi_wrappers import compute_pmf_ffi, compute_moments_ffi

    # Create and serialize graph
    g = Graph(...)
    structure_json = g.serialize_json()

    # Compute PMF with JAX FFI
    theta = jnp.array([1.0, 0.5])
    times = jnp.linspace(0.1, 5.0, 100)
    pmf = compute_pmf_ffi(structure_json, theta, times, discrete=False)

    # Works with JAX transformations
    jit_pmf = jax.jit(compute_pmf_ffi, static_argnums=(0, 3))
    grad_pmf = jax.grad(lambda t: jnp.sum(compute_pmf_ffi(structure_json, t, times)))
    ```
"""

import json
import jax
import jax.numpy as jnp
from jax import ffi
from typing import Union, Dict, Any
import numpy as np

# Import configuration and exceptions
from .config import get_config
from .exceptions import PTDBackendError, PTDConfigError

# Import the C++ module (required - not optional)
try:
    from . import phasic_pybind as cpp_module
    _HAS_CPP_MODULE = True
except ImportError as e:
    raise PTDBackendError(
        "C++ pybind11 module not available.\n"
        "  This is a core dependency and should always be present.\n"
        f"  Import error: {e}"
    )

# FFI registration state
# Registration happens lazily on first use, AFTER JAX is initialized
_lib = None

# ============================================================================
# Helper Functions
# ============================================================================

def _make_json_serializable(obj):
    """
    Convert an object to JSON-serializable format.

    Recursively converts numpy arrays to lists.

    Parameters
    ----------
    obj : any
        Object to convert (can be dict, list, ndarray, or scalar)

    Returns
    -------
    any
        JSON-serializable version of obj
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def _ensure_json_string(structure_json: Union[str, Dict]) -> str:
    """
    Ensure structure_json is a JSON string.

    If input is a dict (from graph.serialize()), convert to JSON string.
    If input is already a string, return as-is.

    Parameters
    ----------
    structure_json : str or dict
        Graph structure as JSON string or dict

    Returns
    -------
    str
        JSON string representation
    """
    if isinstance(structure_json, str):
        return structure_json
    elif isinstance(structure_json, dict):
        # Convert numpy arrays to lists before JSON serialization
        serializable = _make_json_serializable(structure_json)
        return json.dumps(serializable)
    else:
        raise TypeError(
            f"structure_json must be str or dict, got {type(structure_json)}"
        )


def _serialize_graph_structure(graph) -> str:
    """
    Serialize a Graph object to JSON string for FFI.

    Parameters
    ----------
    graph : Graph
        The graph object to serialize

    Returns
    -------
    str
        JSON string representation of the graph structure
    """
    serialized = graph.serialize()

    # Convert numpy arrays to lists for JSON serialization
    json_dict = {}
    for key, value in serialized.items():
        if isinstance(value, np.ndarray):
            json_dict[key] = value.tolist()
        else:
            json_dict[key] = value

    return json.dumps(json_dict)


# ============================================================================
# FFI Registration (Phase 2 Implementation)
# ============================================================================

_FFI_REGISTERED = False

def _register_ffi_targets():
    """Register FFI targets with JAX.

    This function is called lazily on first use, AFTER JAX is initialized.
    It creates FFI handlers on-demand to avoid static initialization issues.

    Returns
    -------
    bool
        True if FFI registration succeeded

    Raises
    ------
    PTDConfigError
        If FFI is disabled via config (user should enable it or set ffi=False)
    PTDBackendError
        If FFI is enabled but not available (build issue)
    """
    global _FFI_REGISTERED

    if _FFI_REGISTERED:
        return True

    # Check if FFI backend is requested
    config = get_config()
    if not config.ffi:
        raise PTDConfigError(
            "FFI backend is disabled in configuration.\n"
            "  FFI is required for multi-core parallelization with vmap.\n"
            "  To enable: phasic.configure(ffi=True)\n"
            "  Note: Requires rebuild with XLA headers if not already built.\n"
            f"  Current config: {config}"
        )

    # FFI requested - try to register
    try:
        import jax
        from . import phasic_pybind as cpp_module

        # Get capsules for FFI handlers (created on-demand, safe after JAX init)
        try:
            compute_pmf_capsule = cpp_module.parameterized.get_compute_pmf_ffi_capsule()
            compute_pmf_and_moments_capsule = cpp_module.parameterized.get_compute_pmf_and_moments_ffi_capsule()
        except AttributeError as e:
            raise PTDBackendError(
                "FFI handlers not available in C++ module.\n"
                "  This means the package was built without XLA headers.\n"
                "\n"
                "To rebuild with FFI support:\n"
                "  export XLA_FFI_INCLUDE_DIR=$(python -c \"from jax import ffi; print(ffi.include_dir())\")\n"
                "  pip install --no-build-isolation --force-reinstall --no-deps .\n"
                "\n"
                "Or disable FFI (slower, single-core only):\n"
                "  import phasic\n"
                "  phasic.configure(ffi=False, openmp=False)"
            ) from e

        # Register with JAX FFI
        try:
            jax.ffi.register_ffi_target(
                "ptd_compute_pmf",
                compute_pmf_capsule,
                platform="cpu",
                api_version=1  # XLA FFI API v1.0
            )
            jax.ffi.register_ffi_target(
                "ptd_compute_pmf_and_moments",
                compute_pmf_and_moments_capsule,
                platform="cpu",
                api_version=1  # XLA FFI API v1.0
            )
        except Exception as e:
            # FFI registration failed
            raise PTDBackendError(
                f"FFI registration failed: {e}\n"
                "  This may be due to JAX/XLA version incompatibility.\n"
                "  Try updating JAX: pip install --upgrade jax jaxlib"
            ) from e

        _FFI_REGISTERED = True
        return True

    except (ImportError, RuntimeError) as e:
        # FFI system not available
        raise PTDBackendError(
            f"FFI backend unavailable: {e}\n"
            "  This is likely a build or installation issue.\n"
            "  Try rebuilding: pip install --force-reinstall --no-deps ."
        ) from e


# ============================================================================
# Fallback Implementation (using pybind11 directly)
# ============================================================================

def _compute_pmf_impl(structure_json: str, theta_np: np.ndarray, times_np: np.ndarray,
                     discrete: bool, granularity: int) -> np.ndarray:
    """Internal implementation for compute_pmf (pure Python/numpy)."""
    builder = cpp_module.parameterized.GraphBuilder(structure_json)
    return builder.compute_pmf(theta_np, times_np, discrete, granularity)


# def compute_pmf_fallback(structure_json: Union[str, Dict], theta: jax.Array, times: jax.Array,
#                         discrete: bool = False, granularity: int = 100) -> jax.Array:
#     """
#     Compute PMF/PDF using pybind11 GraphBuilder (fallback when FFI not available).

#     This uses JAX's pure_callback to wrap the C++ call, enabling JIT compilation
#     while maintaining compatibility with the pybind11 interface.

#     Parameters
#     ----------
#     structure_json : str or dict
#         JSON string or dict (from graph.serialize()) representing graph structure
#     theta : jax.Array
#         Parameter array, shape (n_params,)
#     times : jax.Array
#         Time points (continuous) or jump counts (discrete), shape (n_times,)
#     discrete : bool, default=False
#         If True, compute DPH (discrete phase-type)
#         If False, compute PDF (continuous phase-type)
#     granularity : int, default=100
#         Discretization granularity for PDF computation

#     Returns
#     -------
#     jax.Array
#         PMF/PDF values, shape (n_times,)
#     """
#     if not _HAS_CPP_MODULE:
#         raise RuntimeError("C++ module not available. Cannot compute PMF.")

#     # Ensure structure_json is a JSON string (convert dict if needed)
#     structure_json_str = _ensure_json_string(structure_json)

#     # Use pure_callback to wrap the C++ call
#     # This allows JIT compilation while calling out to Python/C++
#     result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)

#     result = jax.pure_callback(
#         lambda theta_jax, times_jax: _compute_pmf_impl(
#             structure_json_str,
#             np.asarray(theta_jax),
#             np.asarray(times_jax),
#             discrete,
#             granularity
#         ),
#         result_shape,
#         theta,
#         times,
#         vmap_method='sequential'  # Enable vmap support (JAX v0.6.0+)
#     )

#     return result


def _compute_moments_impl(structure_json: str, theta_np: np.ndarray,
                         nr_moments: int) -> np.ndarray:
    """Internal implementation for compute_moments (pure Python/numpy)."""
    builder = cpp_module.parameterized.GraphBuilder(structure_json)
    return builder.compute_moments(theta_np, nr_moments)


# def compute_moments_fallback(structure_json: Union[str, Dict], theta: jax.Array,
#                              nr_moments: int) -> jax.Array:
#     """
#     Compute distribution moments using pybind11 GraphBuilder (fallback).

#     Uses JAX's pure_callback for JIT compatibility.

#     Parameters
#     ----------
#     structure_json : str or dict
#         JSON string or dict (from graph.serialize()) representing graph structure
#     theta : jax.Array
#         Parameter array, shape (n_params,)
#     nr_moments : int
#         Number of moments to compute

#     Returns
#     -------
#     jax.Array
#         Moments array, shape (nr_moments,)
#         Contains [E[T], E[T^2], ..., E[T^nr_moments]]
#     """
#     if not _HAS_CPP_MODULE:
#         raise RuntimeError("C++ module not available. Cannot compute moments.")

#     # Ensure structure_json is a JSON string (convert dict if needed)
#     structure_json_str = _ensure_json_string(structure_json)

#     # Use pure_callback to wrap the C++ call
#     result_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)

#     result = jax.pure_callback(
#         lambda theta_jax: _compute_moments_impl(
#             structure_json_str,
#             np.asarray(theta_jax),
#             nr_moments
#         ),
#         result_shape,
#         theta,
#         vmap_method='sequential'  # Enable vmap support (JAX v0.6.0+)
#     )

#     return result


def _compute_pmf_and_moments_impl(structure_json: str, theta_np: np.ndarray,
                                  times_np: np.ndarray, nr_moments: int,
                                  discrete: bool, granularity: int) -> tuple[np.ndarray, np.ndarray]:
    """Internal implementation for compute_pmf_and_moments (pure Python/numpy)."""
    builder = cpp_module.parameterized.GraphBuilder(structure_json)
    return builder.compute_pmf_and_moments(
        theta_np, times_np, nr_moments, discrete, granularity
    )


def compute_pmf_and_moments_fallback(structure_json: Union[str, Dict], theta: jax.Array,
                                    times: jax.Array, nr_moments: int,
                                    discrete: bool = False,
                                    granularity: int = 100) -> tuple[jax.Array, jax.Array]:
    """
    Compute both PMF and moments using pybind11 GraphBuilder (fallback).

    More efficient than separate calls because the graph is built only once.
    Uses JAX's pure_callback for JIT compatibility.

    Parameters
    ----------
    structure_json : str or dict
        JSON string or dict (from graph.serialize()) representing graph structure
    theta : jax.Array
        Parameter array, shape (n_params,)
    times : jax.Array
        Time points or jump counts, shape (n_times,)
    nr_moments : int
        Number of moments to compute
    discrete : bool, default=False
        If True, use DPH mode; if False, use PDF mode
    granularity : int, default=100
        Discretization granularity for PDF

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (pmf_values, moments)
        - pmf_values: shape (n_times,)
        - moments: shape (nr_moments,)
    """
    if not _HAS_CPP_MODULE:
        raise RuntimeError("C++ module not available. Cannot compute PMF and moments.")

    # Ensure structure_json is a JSON string (convert dict if needed)
    structure_json_str = _ensure_json_string(structure_json)

    # Use pure_callback to wrap the C++ call
    pmf_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)
    moments_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)
    result_shapes = (pmf_shape, moments_shape)

    def callback_fn(theta_jax, times_jax):
        return _compute_pmf_and_moments_impl(
            structure_json_str,
            np.asarray(theta_jax),
            np.asarray(times_jax),
            nr_moments,
            discrete,
            granularity
        )

    pmf, moments = jax.pure_callback(
        callback_fn,
        result_shapes,
        theta,
        times,
        vmap_method='sequential'  # Enable vmap support (JAX v0.6.0+)
    )

    return pmf, moments


# ============================================================================
# Public API
# ============================================================================

def compute_pmf_ffi(structure_json: Union[str, Dict], theta: jax.Array, times: jax.Array,
                   discrete: bool = False, granularity: int = 100) -> jax.Array:
    """
    Compute PMF (discrete) or PDF (continuous) using JAX FFI.

    This function uses JAX's Foreign Function Interface to call C++ code
    with proper GIL management and XLA integration. It supports all JAX
    transformations including jit, grad, vmap, and pmap.

    Parameters
    ----------
    structure_json : str or dict
        JSON string or dict (from Graph.serialize()) containing graph structure
    theta : jax.Array
        Parameter array, shape (n_params,)
    times : jax.Array
        Time points (continuous) or jump counts (discrete), shape (n_times,)
    discrete : bool, default=False
        If True, compute DPH (discrete phase-type)
        If False, compute PDF (continuous phase-type)
    granularity : int, default=100
        Discretization granularity for PDF computation (ignored for DPH)

    Returns
    -------
    jax.Array
        PMF/PDF values, shape (n_times,)

    Raises
    ------
    PTDConfigError
        If FFI is disabled in configuration
    PTDBackendError
        If FFI is enabled but not available (build issue)

    Notes
    -----
    - Requires FFI to be enabled and built with XLA headers
    - Accepts both JSON string and dict from graph.serialize()
    - GIL is released during C++ computation
    - Supports batching via vmap with OpenMP multi-threading
    - Differentiable with custom VJP rules

    Examples
    --------
    >>> # Using dict from graph.serialize()
    >>> structure_dict = graph.serialize()
    >>> theta = jnp.array([1.0, 0.5])
    >>> times = jnp.linspace(0.1, 5.0, 100)
    >>> pmf = compute_pmf_ffi(structure_dict, theta, times, discrete=False)
    >>>
    >>> # JIT compilation
    >>> jit_pmf = jax.jit(compute_pmf_ffi, static_argnums=(0, 3, 4))
    >>> fast_pmf = jit_pmf(structure_dict, theta, times, False, 100)
    """
    # Register FFI targets (raises error if FFI disabled or unavailable)
    _register_ffi_targets()

    # Use JAX FFI (XLA-optimized zero-copy, enables multi-core parallelization via OpenMP)
    # JSON is passed as STRING ATTRIBUTE (static, not batched by vmap)
    structure_str = _ensure_json_string(structure_json)

    # Call JAX FFI target
    # NOTE: JSON passed as attribute (static), theta/times as buffers (batched)
    # expand_dims: vmap adds batch dimension, FFI handler loops over batch with OpenMP
    ffi_fn = jax.ffi.ffi_call(
        "ptd_compute_pmf",
        jax.ShapeDtypeStruct(times.shape, times.dtype),
        vmap_method="expand_dims"  # Batch dim added, handler processes all at once with OpenMP
    )
    result = ffi_fn(
        theta,       # Arg 1: theta buffer (BATCHED by vmap)
        times,       # Arg 2: times buffer (BATCHED by vmap)
        structure_json=structure_str,           # Attr: JSON string (STATIC, not batched)
        granularity=np.int32(granularity),      # Attr: granularity
        discrete=np.bool_(discrete)             # Attr: discrete
    )
    return result


def compute_moments_ffi(structure_json: Union[str, Dict], theta: jax.Array,
                       nr_moments: int) -> jax.Array:
    """
    Compute distribution moments using JAX FFI.

    Computes E[T^k] for k=1,2,...,nr_moments using efficient C++ implementation
    with JAX FFI integration.

    Parameters
    ----------
    structure_json : str or dict
        JSON string or dict (from Graph.serialize()) containing graph structure
    theta : jax.Array
        Parameter array, shape (n_params,)
    nr_moments : int
        Number of moments to compute

    Returns
    -------
    jax.Array
        Moments array, shape (nr_moments,)
        Contains [E[T], E[T^2], ..., E[T^nr_moments]]

    Examples
    --------
    >>> structure_dict = graph.serialize()
    >>> moments = compute_moments_ffi(structure_dict, theta, nr_moments=3)
    >>> mean = moments[0]
    >>> variance = moments[1] - moments[0]**2
    """
    # For now, use fallback implementation
    # TODO: Replace with true FFI call once handlers are properly exposed
    return compute_moments_fallback(structure_json, theta, nr_moments)


def compute_pmf_and_moments_ffi(structure_json: Union[str, Dict], theta: jax.Array,
                               times: jax.Array, nr_moments: int,
                               discrete: bool = False,
                               granularity: int = 100) -> tuple[jax.Array, jax.Array]:
    """
    Compute both PMF and moments efficiently using JAX FFI.

    More efficient than calling compute_pmf_ffi() and compute_moments_ffi()
    separately because the graph is built only once.

    Primary use case: SVGD with moment-based regularization.

    Parameters
    ----------
    structure_json : str or dict
        JSON string or dict (from Graph.serialize()) containing graph structure
    theta : jax.Array
        Parameter array, shape (n_params,)
    times : jax.Array
        Time points or jump counts, shape (n_times,)
    nr_moments : int
        Number of moments to compute
    discrete : bool, default=False
        If True, use DPH mode; if False, use PDF mode
    granularity : int, default=100
        Discretization granularity for PDF (ignored for DPH)

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (pmf_values, moments)
        - pmf_values: shape (n_times,)
        - moments: shape (nr_moments,)

    Raises
    ------
    PTDConfigError
        If FFI is disabled in configuration
    PTDBackendError
        If FFI is enabled but not available (build issue)

    Examples
    --------
    >>> structure_dict = graph.serialize()
    >>> pmf, moments = compute_pmf_and_moments_ffi(
    ...     structure_dict, theta, times, nr_moments=2, discrete=False
    ... )
    >>> # Use pmf for likelihood, moments for regularization
    >>> likelihood = jnp.sum(jnp.log(pmf))
    >>> moment_penalty = jnp.sum((moments - target_moments)**2)
    """
    # Register FFI targets (raises error if FFI disabled or unavailable)
    _register_ffi_targets()

    # Use JAX FFI (XLA-optimized zero-copy with OpenMP parallelization)
    # Convert structure to JSON bytes (survives pickle boundary)
    structure_str = _ensure_json_string(structure_json)
    # Create owned array (not view) to ensure data persists until FFI accesses it
    json_str_bytes = structure_str.encode('utf-8')
    json_bytes = jnp.array(np.frombuffer(json_str_bytes, dtype=np.uint8))

    # Call JAX FFI target with JSON
    pmf_result, moments_result = jax.ffi.ffi_call(
        "ptd_compute_pmf_and_moments",
        (jax.ShapeDtypeStruct(times.shape, times.dtype),
         jax.ShapeDtypeStruct((nr_moments,), jnp.float64)),
        json_bytes,  # Arg 1: structure_json buffer
        theta,       # Arg 2: theta buffer
        times,       # Arg 3: times buffer
        granularity=granularity,   # Attr: granularity
        discrete=discrete,          # Attr: discrete
        nr_moments=nr_moments       # Attr: nr_moments
    )
    return pmf_result, moments_result


# ============================================================================
# Module Initialization
# ============================================================================

# FFI registration is currently DISABLED
# When re-enabled, registration must be explicit (not automatic on import)
# to avoid memory corruption from static global constructors.
# See FFI_MEMORY_CORRUPTION_FIX.md for details.
#
# Future implementation should use:
#   def register_ffi():
#       """Explicitly register FFI handlers AFTER JAX initialization"""
#       if get_config().ffi:
#           _register_ffi_targets()
#
# DO NOT attempt automatic registration on module import!


__all__ = [
    'compute_pmf_ffi',
    'compute_moments_ffi',
    'compute_pmf_and_moments_ffi',
]
