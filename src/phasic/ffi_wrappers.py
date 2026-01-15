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
            compute_moments_capsule = cpp_module.parameterized.get_compute_moments_ffi_capsule()
            compute_pmf_and_moments_capsule = cpp_module.parameterized.get_compute_pmf_and_moments_ffi_capsule()
            compute_pmf_multivariate_capsule = cpp_module.parameterized.get_compute_pmf_multivariate_ffi_capsule()
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
                "ptd_compute_moments",
                compute_moments_capsule,
                platform="cpu",
                api_version=1  # XLA FFI API v1.0
            )
            jax.ffi.register_ffi_target(
                "ptd_compute_pmf_and_moments",
                compute_pmf_and_moments_capsule,
                platform="cpu",
                api_version=1  # XLA FFI API v1.0
            )
            jax.ffi.register_ffi_target(
                "ptd_compute_pmf_multivariate",
                compute_pmf_multivariate_capsule,
                platform="cpu",
                api_version=1  # XLA FFI API v1.0
            )

            # Get sojourn times capsule
            try:
                compute_sojourn_times_capsule = cpp_module.parameterized.get_compute_sojourn_times_ffi_capsule()
            except AttributeError as e:
                raise PTDBackendError(
                    "FFI handler get_compute_sojourn_times_ffi_capsule() not available.\n"
                    "  Rebuild with: pixi run install-dev"
                ) from e

            jax.ffi.register_ffi_target(
                "ptd_compute_sojourn_times",
                compute_sojourn_times_capsule,
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


# def compute_pmf_and_moments_fallback(structure_json: Union[str, Dict], theta: jax.Array,
#                                     times: jax.Array, nr_moments: int,
#                                     discrete: bool = False,
#                                     granularity: int = 100) -> tuple[jax.Array, jax.Array]:
#     """
#     Compute both PMF and moments using pybind11 GraphBuilder (fallback).

#     More efficient than separate calls because the graph is built only once.
#     Uses JAX's pure_callback for JIT compatibility.

#     Parameters
#     ----------
#     structure_json : str or dict
#         JSON string or dict (from graph.serialize()) representing graph structure
#     theta : jax.Array
#         Parameter array, shape (n_params,)
#     times : jax.Array
#         Time points or jump counts, shape (n_times,)
#     nr_moments : int
#         Number of moments to compute
#     discrete : bool, default=False
#         If True, use DPH mode; if False, use PDF mode
#     granularity : int, default=100
#         Discretization granularity for PDF

#     Returns
#     -------
#     tuple[jax.Array, jax.Array]
#         (pmf_values, moments)
#         - pmf_values: shape (n_times,)
#         - moments: shape (nr_moments,)
#     """
#     if not _HAS_CPP_MODULE:
#         raise RuntimeError("C++ module not available. Cannot compute PMF and moments.")

#     # Ensure structure_json is a JSON string (convert dict if needed)
#     structure_json_str = _ensure_json_string(structure_json)

#     # Use pure_callback to wrap the C++ call
#     pmf_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)
#     moments_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)
#     result_shapes = (pmf_shape, moments_shape)

#     def callback_fn(theta_jax, times_jax):
#         return _compute_pmf_and_moments_impl(
#             structure_json_str,
#             np.asarray(theta_jax),
#             np.asarray(times_jax),
#             nr_moments,
#             discrete,
#             granularity
#         )

#     pmf, moments = jax.pure_callback(
#         callback_fn,
#         result_shapes,
#         theta,
#         times,
#         vmap_method='sequential'  # Enable vmap support (JAX v0.6.0+)
#     )

#     return pmf, moments


# ============================================================================
# Public API
# ============================================================================

def compute_pmf_ffi(structure_json: Union[str, Dict], theta: jax.Array, times: jax.Array,
                   discrete: bool = False, granularity: int = 0) -> jax.Array:
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
        jax.ShapeDtypeStruct(times.shape, jnp.float64),  # Force float64 output
        vmap_method="expand_dims"  # Batch dim added, handler processes all at once with OpenMP
    )
    result = ffi_fn(
        theta,       # Arg 1: theta buffer (BATCHED by vmap)
        times,       # Arg 2: times buffer (BATCHED by vmap)
        structure_json=structure_str,           # Attr: JSON string (STATIC, not batched)
        granularity=np.int32(granularity),      # Attr: granularity
        discrete=bool(discrete)                 # Attr: discrete (bool for JAX PRED type)
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
    - Supports batching via vmap
    - More efficient to use compute_pmf_and_moments_ffi() if you need both

    Examples
    --------
    >>> # Using dict from graph.serialize()
    >>> structure_dict = graph.serialize()
    >>> theta = jnp.array([1.0, 0.5])
    >>> moments = compute_moments_ffi(structure_dict, theta, nr_moments=3)
    >>> mean = moments[0]
    >>> variance = moments[1] - moments[0]**2
    >>>
    >>> # JIT compilation
    >>> jit_moments = jax.jit(compute_moments_ffi, static_argnums=(0, 2))
    >>> fast_moments = jit_moments(structure_dict, theta, 3)
    """
    # Register FFI targets (raises error if FFI disabled or unavailable)
    _register_ffi_targets()

    # Use JAX FFI (XLA-optimized zero-copy)
    # JSON is passed as STRING ATTRIBUTE (static, not batched by vmap)
    structure_str = _ensure_json_string(structure_json)

    # Call JAX FFI target
    # NOTE: JSON passed as attribute (static), theta as buffer (batched)
    # expand_dims: vmap adds batch dimension, FFI handler processes batches
    ffi_fn = jax.ffi.ffi_call(
        "ptd_compute_moments",
        jax.ShapeDtypeStruct((nr_moments,), jnp.float64),
        vmap_method="expand_dims"  # Batch dim added, handler processes all
    )
    result = ffi_fn(
        theta,       # Arg 1: theta buffer (BATCHED by vmap)
        structure_json=structure_str,        # Attr: JSON string (STATIC, not batched)
        nr_moments=np.int32(nr_moments)      # Attr: nr_moments
    )
    return result


def compute_pmf_and_moments_ffi(structure_json: Union[str, Dict], theta: jax.Array,
                               times: jax.Array, nr_moments: int,
                               discrete: bool = False,
                               granularity: int = 0,
                               rewards: jax.Array = None) -> tuple[jax.Array, jax.Array]:
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
    rewards : jax.Array or None, default=None
        Optional reward vector (one per vertex). If None, computes standard moments E[T^k].
        If provided, computes reward-transformed moments E[R·T^k].

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
    # JSON is passed as STRING ATTRIBUTE (static, not batched by vmap)
    structure_str = _ensure_json_string(structure_json)

    # Handle optional rewards: use empty array if None
    if rewards is None:
        rewards = jnp.array([], dtype=jnp.float64)
    else:
        rewards = jnp.asarray(rewards, dtype=jnp.float64)

    # Determine output shapes based on rewards dimensionality
    # NOTE: For multivariate models with 2D rewards, use pmf_and_moments_from_graph_multivariate()
    # which loops over features in Python. The 1D model always returns 1D outputs.
    # Use len(rewards.shape) instead of ndim to work with JAX tracing
    if len(rewards.shape) == 2:
        # 2D rewards (n_vertices, n_features): multivariate case
        # However, this codepath is not currently used - pmf_and_moments_from_graph_multivariate()
        # loops in Python instead. Keeping this for future optimization.
        n_features = rewards.shape[1]
        pmf_shape = jax.ShapeDtypeStruct((times.shape[0], n_features), jnp.float64)
        moments_shape = jax.ShapeDtypeStruct((n_features, nr_moments), jnp.float64)
    else:
        # No rewards or 1D rewards: univariate case
        pmf_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)  # Force float64 output
        moments_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)

    # Call JAX FFI target
    # NOTE: JSON passed as attribute (static), theta/times/rewards as buffers (batched)
    # expand_dims: vmap adds batch dimension, FFI handler loops over batch with OpenMP
    ffi_fn = jax.ffi.ffi_call(
        "ptd_compute_pmf_and_moments",
        (pmf_shape, moments_shape),
        vmap_method="expand_dims"  # Batch dim added, handler processes all at once with OpenMP
    )
    pmf_result, moments_result = ffi_fn(
        theta,       # Arg 1: theta buffer (BATCHED by vmap)
        times,       # Arg 2: times buffer (BATCHED by vmap)
        rewards,     # Arg 3: rewards buffer (empty array if None → standard moments)
        structure_json=structure_str,           # Attr: JSON string (STATIC, not batched)
        granularity=np.int32(granularity),      # Attr: granularity
        discrete=bool(discrete),                # Attr: discrete (bool for JAX PRED type)
        nr_moments=np.int32(nr_moments)         # Attr: nr_moments
    )
    return pmf_result, moments_result


def compute_pmf_multivariate_ffi(structure_json: Union[str, Dict], theta: jax.Array,
                                times: jax.Array, rewards: jax.Array,
                                discrete: bool = False, granularity: int = 0,
                                compute_joint: bool = False) -> jax.Array:
    """
    Compute multivariate PMF/PDF using JAX FFI.

    Supports multivariate observations where each observation is a vector of features.
    Each feature dimension has its own reward vector (column in rewards matrix).

    Modes:
    - Sparse mode (compute_joint=False): Independent PDF per feature, zeros for missing obs
    - Joint mode (compute_joint=True): Joint PDF across features [NOT YET IMPLEMENTED]

    Parameters
    ----------
    structure_json : str or dict
        JSON string or dict (from Graph.serialize()) containing graph structure
    theta : jax.Array
        Parameter array, shape (n_params,)
    times : jax.Array
        Time points or jump counts, shape (n_times, n_features)
        Zero entries are treated as "no observation" in sparse mode
    rewards : jax.Array
        Reward matrix, shape (n_vertices, n_features)
        Each column defines the reward vector for one feature dimension
    discrete : bool, default=False
        If True, compute DPH (discrete phase-type)
        If False, compute PDF (continuous phase-type)
    granularity : int, default=0
        Discretization granularity for PDF computation (ignored for DPH)
        If 0, uses automatic granularity (2 * max_rate)
    compute_joint : bool, default=False
        If True, compute joint PDF across features [raises error - not implemented]
        If False, compute independent PDFs (sparse mode)

    Returns
    -------
    jax.Array
        PMF/PDF values, shape (n_times, n_features)
        Zero wherever times[i,j] == 0.0 in sparse mode

    Raises
    ------
    PTDConfigError
        If FFI is disabled in configuration
    PTDBackendError
        If FFI is enabled but not available (build issue)
    ValueError
        If compute_joint=True (not yet implemented)

    Notes
    -----
    - Requires FFI to be enabled and built with XLA headers
    - GIL is released during C++ computation
    - Supports batching via vmap with OpenMP multi-threading
    - Differentiable with custom VJP rules

    Examples
    --------
    >>> # Sparse observations: 2 features, some observations missing
    >>> structure_dict = graph.serialize()
    >>> theta = jnp.array([1.0, 0.5])
    >>> times = jnp.array([
    ...     [1.5, 0.0],   # Observe feature 0 only
    ...     [0.0, 2.1],   # Observe feature 1 only
    ...     [1.2, 1.8]    # Observe both features
    ... ])  # Shape: (3, 2)
    >>> n_vertices = graph.vertices_length()
    >>> rewards = jnp.array([
    ...     [1.0, 0.5],  # Vertex 0 rewards for features 0, 1
    ...     [2.0, 1.0]   # Vertex 1 rewards for features 0, 1
    ... ])  # Shape: (n_vertices, 2)
    >>> pdf = compute_pmf_multivariate_ffi(
    ...     structure_dict, theta, times, rewards, discrete=False
    ... )
    >>> # pdf.shape = (3, 2)
    >>> # pdf[0,0] = PDF(t=1.5, rewards[:,0]), pdf[0,1] = 0.0
    >>> # pdf[1,0] = 0.0,                      pdf[1,1] = PDF(t=2.1, rewards[:,1])
    >>> # pdf[2,0] = PDF(t=1.2, rewards[:,0]), pdf[2,1] = PDF(t=1.8, rewards[:,1])
    >>>
    >>> # JIT compilation
    >>> jit_fn = jax.jit(compute_pmf_multivariate_ffi, static_argnums=(0, 4, 5, 6))
    >>> fast_pdf = jit_fn(structure_dict, theta, times, rewards, False, 100, False)
    """
    # Register FFI targets (raises error if FFI disabled or unavailable)
    _register_ffi_targets()

    # Validate compute_joint mode
    if compute_joint:
        raise ValueError(
            "Joint PDF computation not yet implemented. "
            "Use compute_joint=False for independent feature PDFs (sparse mode)."
        )

    # Use JAX FFI (XLA-optimized zero-copy with OpenMP parallelization)
    # JSON is passed as STRING ATTRIBUTE (static, not batched by vmap)
    structure_str = _ensure_json_string(structure_json)

    # Ensure inputs are JAX arrays with correct dtypes
    times = jnp.asarray(times, dtype=jnp.float64)
    rewards = jnp.asarray(rewards, dtype=jnp.float64)

    # Validate shapes
    if len(times.shape) != 2:
        raise ValueError(f"times must be 2D (n_times, n_features), got shape {times.shape}")
    if len(rewards.shape) != 2:
        raise ValueError(f"rewards must be 2D (n_vertices, n_features), got shape {rewards.shape}")
    if times.shape[1] != rewards.shape[1]:
        raise ValueError(
            f"times and rewards must have same number of features: "
            f"times.shape[1]={times.shape[1]}, rewards.shape[1]={rewards.shape[1]}"
        )

    # Output shape matches times shape
    result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)

    # Call JAX FFI target
    # NOTE: JSON passed as attribute (static), theta/times/rewards as buffers (batched)
    # expand_dims: vmap adds batch dimension, FFI handler loops over batch with OpenMP
    ffi_fn = jax.ffi.ffi_call(
        "ptd_compute_pmf_multivariate",
        result_shape,
        vmap_method="expand_dims"  # Batch dim added, handler processes all at once with OpenMP
    )
    result = ffi_fn(
        theta,       # Arg 1: theta buffer (BATCHED by vmap)
        times,       # Arg 2: times buffer (BATCHED by vmap, 2D or 3D after batch)
        rewards,     # Arg 3: rewards buffer (BATCHED by vmap, 2D or 3D after batch)
        structure_json=structure_str,           # Attr: JSON string (STATIC, not batched)
        granularity=np.int32(granularity),      # Attr: granularity
        discrete=bool(discrete),                # Attr: discrete (bool for JAX PRED type)
        compute_joint=bool(compute_joint)       # Attr: compute_joint (must be False for now)
    )
    return result


def compute_sojourn_times_ffi(structure_json: Union[str, Dict], theta: jax.Array,
                               indices: jax.Array) -> jax.Array:
    """
    Compute expected sojourn times for a subset of vertices using JAX FFI.

    Memory-efficient subset computation using n×k matrix instead of n×n.
    Supports vmap batching with OpenMP parallelization and broadcasting.

    Parameters
    ----------
    structure_json : str or dict
        JSON string or dict (from Graph.serialize()) containing graph structure
    theta : jax.Array
        Parameter array, shape (n_params,)
    indices : jax.Array
        Vertex indices to compute sojourn times for, shape (k,)
        Must be int32 dtype, non-negative, < n_vertices

    Returns
    -------
    jax.Array
        Expected sojourn times for requested vertices, shape (k,)
        sojourn_times[i] = E[time spent in vertex indices[i]]

    Raises
    ------
    PTDConfigError
        If FFI is disabled in configuration
    PTDBackendError
        If FFI is enabled but not available (build issue)
    ValueError
        If indices has wrong dtype (must be int32)

    Notes
    -----
    - Requires FFI to be enabled and built with XLA headers
    - Memory: O(n×k) vs O(n²) for full computation
    - For large graphs: 268 GB → 1.7 GB (99.4% reduction for typical use)
    - GIL is released during C++ computation
    - Supports batching via vmap with OpenMP multi-threading
    - Broadcasting: singleton indices broadcast to all theta batches
    - Differentiable with custom VJP rules (finite differences)

    Examples
    --------
    >>> # Basic usage: compute sojourn times for specific vertices
    >>> structure_dict = graph.serialize()
    >>> theta = jnp.array([1.0, 0.5])
    >>> indices = jnp.array([0, 5, 10], dtype=jnp.int32)  # Must be int32!
    >>> sojourn = compute_sojourn_times_ffi(structure_dict, theta, indices)
    >>> # sojourn.shape = (3,)
    >>>
    >>> # JIT compilation
    >>> jit_fn = jax.jit(compute_sojourn_times_ffi, static_argnums=(0,))
    >>> fast_sojourn = jit_fn(structure_dict, theta, indices)
    >>>
    >>> # Batching with vmap (OpenMP parallelization)
    >>> theta_batch = jnp.array([[1.0, 0.5], [2.0, 1.0], [1.5, 0.8]])  # (3, 2)
    >>> batched_fn = jax.vmap(
    ...     lambda t: compute_sojourn_times_ffi(structure_dict, t, indices)
    ... )
    >>> sojourn_batch = batched_fn(theta_batch)  # (3, 3) - 3 batches × 3 indices
    >>>
    >>> # Broadcasting: singleton indices with batched theta
    >>> indices_singleton = jnp.array([1, 2], dtype=jnp.int32)  # (2,)
    >>> theta_batch = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1)
    >>> # Same indices used for all 3 theta values (automatic broadcasting)
    """
    # Register FFI targets (raises error if FFI disabled or unavailable)
    _register_ffi_targets()

    # Use JAX FFI (XLA-optimized zero-copy with OpenMP parallelization)
    # JSON is passed as STRING ATTRIBUTE (static, not batched by vmap)
    structure_str = _ensure_json_string(structure_json)

    # Ensure inputs are JAX arrays with correct dtypes
    theta = jnp.asarray(theta, dtype=jnp.float64)
    indices = jnp.asarray(indices, dtype=jnp.int32)  # MUST be int32 for S32 buffer

    # Validate indices dtype
    if indices.dtype != jnp.int32:
        raise ValueError(
            f"indices must be int32 dtype, got {indices.dtype}. "
            "Convert with: jnp.array(indices, dtype=jnp.int32)"
        )

    # Validate shapes
    if len(indices.shape) != 1:
        raise ValueError(f"indices must be 1D, got shape {indices.shape}")

    # Output shape matches indices length
    result_shape = jax.ShapeDtypeStruct(indices.shape, jnp.float64)

    # Call JAX FFI target
    # NOTE: JSON passed as attribute (static), theta/indices as buffers (batched)
    # expand_dims: vmap adds batch dimension, FFI handler loops over batch with OpenMP
    ffi_fn = jax.ffi.ffi_call(
        "ptd_compute_sojourn_times",
        result_shape,
        vmap_method="expand_dims"  # Batch dim added, handler processes all at once with OpenMP
    )
    result = ffi_fn(
        theta,       # Arg 1: theta buffer (BATCHED by vmap)
        indices,     # Arg 2: indices buffer (BATCHED by vmap, or broadcast if singleton)
        structure_json=structure_str  # Attr: JSON string (STATIC, not batched)
    )
    return result


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
    'compute_pmf_multivariate_ffi',
    'compute_sojourn_times_ffi',
]
