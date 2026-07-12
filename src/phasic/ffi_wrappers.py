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

from __future__ import annotations

import json
from typing import Any

import functools

import jax
import jax.numpy as jnp
from jax import ffi
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

def _make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to JSON-serializable format.

    Recursively converts numpy arrays to lists.

    Parameters
    ----------
    obj : Any
        Object to convert (can be dict, list, ndarray, or scalar)

    Returns
    -------
    Any
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


def _ensure_json_string(structure_json: str | dict) -> str:
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


def _serialize_graph_structure(graph: Any) -> str:
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

def _register_ffi_targets() -> bool:
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
    if not config._use_ffi:
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

            # BFFG handlers: backward probabilities and conditioned path sampling
            try:
                backward_probs_capsule = cpp_module.parameterized.get_backward_probabilities_ffi_capsule()
                jax.ffi.register_ffi_target(
                    "ptd_backward_probabilities",
                    backward_probs_capsule,
                    platform="cpu",
                    api_version=1
                )

                sample_path_capsule = cpp_module.parameterized.get_sample_path_conditioned_ffi_capsule()
                jax.ffi.register_ffi_target(
                    "ptd_sample_path_conditioned",
                    sample_path_capsule,
                    platform="cpu",
                    api_version=1
                )
            except AttributeError:
                pass  # BFFG FFI handlers not available (optional)

            # Daisy-chain joint-probs handler (single FFI call replaces
            # the per-epoch pure_callback chain on the SVGD path).
            try:
                daisy_chain_capsule = cpp_module.parameterized.get_daisy_chain_joint_probs_ffi_capsule()
                jax.ffi.register_ffi_target(
                    "ptd_daisy_chain_joint_probs",
                    daisy_chain_capsule,
                    platform="cpu",
                    api_version=1,
                )
            except AttributeError:
                pass  # Daisy-chain FFI handler not available (optional)

            # Daisy-chain SOJOURN handler (granularity-free final-epoch read).
            try:
                daisy_sojourn_capsule = cpp_module.parameterized.get_daisy_chain_sojourn_ffi_capsule()
                jax.ffi.register_ffi_target(
                    "ptd_daisy_chain_sojourn",
                    daisy_sojourn_capsule,
                    platform="cpu",
                    api_version=1,
                )
            except AttributeError:
                pass  # Daisy-chain sojourn FFI handler not available (optional)

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

def compute_pmf_ffi(structure_json: str | dict, theta: jax.Array, times: jax.Array,
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


def compute_moments_ffi(structure_json: str | dict, theta: jax.Array,
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


def compute_pmf_and_moments_ffi(structure_json: str | dict, theta: jax.Array,
                               times: jax.Array, nr_moments: int,
                               discrete: bool = False,
                               granularity: int = 0,
                               rewards: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
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


def compute_pmf_multivariate_ffi(structure_json: str | dict, theta: jax.Array,
                                times: jax.Array, rewards: jax.Array,
                                discrete: bool = False, granularity: int = 0,
                                compute_joint: bool = False) -> jax.Array:
    """
    Compute multivariate PMF/PDF using JAX FFI.

    Supports multivariate observations where each observation is a vector of features.
    Each feature dimension has its own reward vector (row in rewards matrix).

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
        Reward matrix, shape ``(n_features, n_vertices)``.
        Each ROW defines the reward vector for one feature dimension.
        This orientation matches ``Graph.pmf_and_moments_from_graph_multivariate``
        and the convention documented in CLAUDE.md ("Shape Convention v0.22.22"):
        vertices are always the trailing axis.
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
        If rewards shape is not ``(n_features, n_vertices)``.

    Notes
    -----
    - Requires FFI to be enabled and built with XLA headers
    - GIL is released during C++ computation
    - Supports batching via vmap with OpenMP multi-threading
    - Differentiable with custom VJP rules
    - The C++ handler still expects ``(n_vertices, n_features)``;
      this wrapper transposes the input internally before dispatch.
      Callers must use the documented ``(n_features, n_vertices)``
      orientation.

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
    ...     [1.0, 2.0],  # Feature 0: reward at each vertex
    ...     [0.5, 1.0],  # Feature 1: reward at each vertex
    ... ])  # Shape: (n_features=2, n_vertices)
    >>> pdf = compute_pmf_multivariate_ffi(
    ...     structure_dict, theta, times, rewards, discrete=False
    ... )
    >>> # pdf.shape = (3, 2)
    >>> # pdf[0,0] = PDF(t=1.5, rewards[0, :]), pdf[0,1] = 0.0
    >>> # pdf[1,0] = 0.0,                       pdf[1,1] = PDF(t=2.1, rewards[1, :])
    >>> # pdf[2,0] = PDF(t=1.2, rewards[0, :]), pdf[2,1] = PDF(t=1.8, rewards[1, :])
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
        raise ValueError(
            f"rewards must be 2D (n_features, n_vertices), got shape {rewards.shape}"
        )
    # User-facing convention: (n_features, n_vertices). The C++ handler
    # at graph_builder_ffi.cpp:574-580 still expects
    # (n_vertices, n_features), so transpose internally before dispatch.
    n_features = rewards.shape[0]
    if times.shape[1] != n_features:
        raise ValueError(
            f"times.shape[1] (n_features={times.shape[1]}) must equal "
            f"rewards.shape[0] (n_features={n_features}). Note: rewards "
            f"must have shape (n_features, n_vertices) — if you are "
            f"passing (n_vertices, n_features), transpose it: "
            f"rewards = rewards.T."
        )
    rewards_for_ffi = jnp.swapaxes(rewards, 0, 1)  # -> (n_vertices, n_features)

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
        theta,           # Arg 1: theta buffer (BATCHED by vmap)
        times,           # Arg 2: times buffer (BATCHED by vmap, 2D or 3D after batch)
        rewards_for_ffi, # Arg 3: rewards buffer in C++ orientation
        structure_json=structure_str,           # Attr: JSON string (STATIC, not batched)
        granularity=np.int32(granularity),      # Attr: granularity
        discrete=bool(discrete),                # Attr: discrete (bool for JAX PRED type)
        compute_joint=bool(compute_joint)       # Attr: compute_joint (must be False for now)
    )
    return result


def compute_sojourn_times_ffi(structure_json: str | dict, theta: jax.Array,
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
    # Allow indices to be 1D (legacy / single particle) or 2D (broadcast or
    # batched alongside a 2D theta). The C++ handler at
    # `graph_builder_ffi.cpp:760-806` accepts either.
    if len(indices.shape) not in (1, 2):
        raise ValueError(
            f"indices must be 1D or 2D, got shape {indices.shape}"
        )

    # Output shape: when theta is 2D (batch, theta_dim) — typically because
    # the caller is dispatching its own batched call via a custom_vmap rule
    # rather than through JAX's vmap — the C++ handler returns
    # (batch, n_indices). Detect 2D theta or 2D indices and reflect that in
    # the declared result_shape; otherwise fall back to the 1D output that
    # JAX's `vmap_method="expand_dims"` rule extends automatically when
    # vmap is in play. Mirrors compute_daisy_chain_joint_probs_ffi.
    n_indices = indices.shape[-1]
    if theta.ndim == 2 or indices.ndim == 2:
        batch_size = theta.shape[0] if theta.ndim == 2 else indices.shape[0]
        result_shape = jax.ShapeDtypeStruct(
            (batch_size, n_indices), jnp.float64,
        )
    else:
        result_shape = jax.ShapeDtypeStruct((n_indices,), jnp.float64)

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


def sample_path_conditioned_ffi(
    structure_json: str | dict,
    theta: 'jax.Array',
    target_vertex: 'jax.Array',
    seed: 'jax.Array',
    max_length: int,
) -> tuple['jax.Array', 'jax.Array']:
    """Sample a conditioned path via FFI with fixed-size output.

    Parameters
    ----------
    structure_json : str or dict
        Serialized graph structure.
    theta : jax.Array
        Parameter vector, shape (n_params,).
    target_vertex : jax.Array
        Target terminal vertex index, shape (1,), int32.
    seed : jax.Array
        Random seed, shape (1,), int32.
    max_length : int
        Fixed output size. Path padded with -1/0.0 sentinels.

    Returns
    -------
    vertex_indices : jax.Array, shape (max_length,), int32
        Visited vertex indices, -1 padded.
    entry_times : jax.Array, shape (max_length,), float64
        Cumulative entry times, 0.0 padded.
    """
    _register_ffi_targets()
    structure_str = _ensure_json_string(structure_json)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    target_vertex = jnp.atleast_1d(jnp.asarray(target_vertex, dtype=jnp.int32))
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.int32))

    result_shapes = (
        jax.ShapeDtypeStruct((max_length,), jnp.int32),
        jax.ShapeDtypeStruct((max_length,), jnp.float64),
    )

    ffi_fn = jax.ffi.ffi_call(
        "ptd_sample_path_conditioned",
        result_shapes,
        vmap_method="expand_dims"
    )

    return ffi_fn(
        theta,
        target_vertex,
        seed,
        structure_json=structure_str,
        max_length=np.int32(max_length),
    )


def backward_probabilities_ffi(
    structure_json: str | dict,
    theta: 'jax.Array',
    target_vertices: 'jax.Array',
    n_vertices: int,
) -> 'jax.Array':
    """Compute backward probabilities via FFI.

    Parameters
    ----------
    structure_json : str or dict
        Serialized graph structure.
    theta : jax.Array
        Parameter vector, shape (n_params,).
    target_vertices : jax.Array
        Target terminal indices, shape (n_targets,), int32.
    n_vertices : int
        Total number of vertices (output size).

    Returns
    -------
    jax.Array, shape (n_vertices,)
        P(reach target | start at v) for each vertex.
    """
    _register_ffi_targets()
    structure_str = _ensure_json_string(structure_json)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    target_vertices = jnp.atleast_1d(jnp.asarray(target_vertices, dtype=jnp.int32))

    result_shape = jax.ShapeDtypeStruct((n_vertices,), jnp.float64)

    ffi_fn = jax.ffi.ffi_call(
        "ptd_backward_probabilities",
        result_shape,
        vmap_method="expand_dims"
    )

    return ffi_fn(
        theta,
        target_vertices,
        structure_json=structure_str,
    )


def compute_reward_visit_probability_ffi(
    structure_json: 'str | dict',
    theta: 'jax.Array',
    target_vertices: 'jax.Array',
    initial_ipv: 'jax.Array',
) -> 'jax.Array':
    """Probability of visiting any target vertex before absorption.

    Returns the scalar phase-type quantity

        p(theta) = sum_v alpha_v * h_v(theta)

    where ``h_v(theta) = P(reach target_vertices | start at v)`` is
    computed via the underlying C handler ``ptd_backward_probabilities``
    (NOT the FFI wrapper, which lacks batched-theta support). We route
    through ``jax.pure_callback`` with ``vmap_method='sequential'`` so
    vmap-over-particles dispatches one C call per particle. The
    function is JAX-differentiable via a ``custom_vjp`` with central
    finite differences.

    Parameters
    ----------
    structure_json : str or dict
        Serialized graph structure.
    theta : jax.Array
        Parameter vector, shape (n_params,).
    target_vertices : jax.Array
        Indices of vertices treated as the reachable target set,
        shape (n_targets,), int32. Typically the set of vertices with
        positive reward.
    initial_ipv : jax.Array
        Initial probability vector alpha, shape (n_vertices,).
        Nonzero only at the graph's starting vertices.

    Returns
    -------
    jax.Array, shape ()
        Scalar p(theta) in [0, 1].
    """
    initial_ipv = jnp.asarray(initial_ipv, dtype=jnp.float64)
    target_vertices = jnp.asarray(target_vertices, dtype=jnp.int32)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    return _reward_visit_probability_autodiff(
        structure_json, theta, target_vertices, initial_ipv,
    )


def _reward_visit_probability_callback(
    structure_json_str: str,
    theta_np: np.ndarray,
    target_vertices_np: np.ndarray,
    initial_ipv_np: np.ndarray,
) -> np.ndarray:
    """Concrete-Python helper: compute p(theta) without JAX.

    Builds a temporary Graph with the given theta, runs the C
    backward_probabilities helper, then takes the inner product
    against the IPV. Always returns a 0-D float64 scalar.
    """
    import phasic
    from phasic import Graph
    # Rebuild a Graph from the serialized JSON. Cache the parsed
    # builder on the structure_json_str to avoid repeated parsing.
    builder = _get_pybind_builder(structure_json_str)
    theta_np = np.asarray(theta_np, dtype=np.float64).ravel()
    targets = [int(x) for x in np.asarray(target_vertices_np).ravel()]
    # Build a concrete Graph at the given theta and use its
    # backward_probabilities Python wrapper.
    concrete = Graph(builder.build(theta_np))
    h = np.asarray(concrete.backward_probabilities(targets), dtype=np.float64)
    alpha = np.asarray(initial_ipv_np, dtype=np.float64)
    return np.array(float(np.dot(alpha, h)), dtype=np.float64)


def _get_pybind_builder(structure_json_str: str):
    """Thread-local cache for pybind GraphBuilder by structure JSON.

    Avoids reparsing the JSON on every callback invocation.
    """
    import phasic as _phasic
    cache = getattr(_get_pybind_builder, "_cache", None)
    if cache is None:
        cache = {}
        _get_pybind_builder._cache = cache
    if structure_json_str not in cache:
        cache[structure_json_str] = (
            _phasic.parameterized.GraphBuilder(structure_json_str)
        )
    return cache[structure_json_str]


def _reward_visit_probability_forward(
    structure_json, theta, target_vertices, initial_ipv,
):
    """JAX-side forward via pure_callback so vmap iterates per particle."""
    structure_str = _ensure_json_string(structure_json)
    result_shape = jax.ShapeDtypeStruct((), jnp.float64)

    def _cb(theta_jax, target_jax, ipv_jax):
        return _reward_visit_probability_callback(
            structure_str,
            np.asarray(theta_jax),
            np.asarray(target_jax),
            np.asarray(ipv_jax),
        )

    return jax.pure_callback(
        _cb,
        result_shape,
        theta,
        target_vertices,
        initial_ipv,
        vmap_method='sequential',
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _reward_visit_probability_autodiff(
    structure_json, theta, target_vertices, initial_ipv,
):
    return _reward_visit_probability_forward(
        structure_json, theta, target_vertices, initial_ipv,
    )


def _rvp_fwd(structure_json, theta, target_vertices, initial_ipv):
    out = _reward_visit_probability_forward(
        structure_json, theta, target_vertices, initial_ipv,
    )
    return out, (theta, target_vertices, initial_ipv)


@functools.lru_cache(maxsize=32)
def _weight_mode_from_json_str(structure_json: str) -> str:
    """Cached weight-mode lookup for the JSON-string form."""
    return json.loads(structure_json).get('weight_mode', 'linear')


def _weight_mode_of(structure) -> str:
    """Weight mode of a serialized graph (``Graph.serialize`` emits it).

    Callers pass EITHER the serialized dict (``g.serialize()``) or its JSON
    string, so accept both. The string form is cached — it is a static (nondiff)
    argument parsed once per trace, and the graph JSON can be large; the dict
    form is already parsed and is not hashable, so it is read directly.
    """
    if isinstance(structure, dict):
        return structure.get('weight_mode', 'linear')
    return _weight_mode_from_json_str(structure)


def _rvp_bwd(structure_json, res, cotangent):
    theta, target_vertices, initial_ipv = res
    # Relative, positivity-preserving FD step. A fixed absolute eps=1e-7 drove
    # theta_i - eps NEGATIVE for any rate below it (routine: a per-generation
    # mutation rate is ~1e-8), and the solvers accept a negative rate silently.
    # See phasic._fd_probe_points. Imported lazily: ffi_wrappers is imported
    # from phasic/__init__, so a module-level import would be circular.
    from . import _fd_probe_points
    fd_mode = _weight_mode_of(structure_json)
    n_params = int(theta.shape[-1])
    grads = []
    for i in range(n_params):
        tp, tm, denom = _fd_probe_points(theta, i, fd_mode)
        fp = _reward_visit_probability_forward(
            structure_json, tp, target_vertices, initial_ipv,
        )
        fm = _reward_visit_probability_forward(
            structure_json, tm, target_vertices, initial_ipv,
        )
        grads.append(cotangent * (fp - fm) / denom)
    return (
        jnp.stack(grads),
        jnp.zeros_like(target_vertices, dtype=jnp.float64),
        jnp.zeros_like(initial_ipv),
    )


_reward_visit_probability_autodiff.defvjp(_rvp_fwd, _rvp_bwd)


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


def compute_daisy_chain_joint_probs_ffi(
    structure_json: 'str | dict',
    theta: 'jax.Array',
    initial_ipv: 'jax.Array',
) -> 'jax.Array':
    """Run the daisy-chain joint-probs computation via JAX FFI.

    Daisy-chain metadata (epoch_dts, t_eval, n_epochs, ipv_target_indices,
    t_aux_keys, t_aux_values, t_vertex_indices, param_length) must be
    embedded in ``structure_json`` under a top-level ``"_daisy_chain"``
    object. Use ``Graph._daisy_chain_serialize(...)`` to build it.

    Parameters
    ----------
    structure_json : str or dict
        Serialized JSP graph + ``"_daisy_chain"`` metadata.
    theta : jax.Array
        Per-epoch parameters, flattened: shape ``(n_epochs * param_length,)``.
    initial_ipv : jax.Array
        IPV for epoch 0, shape ``(n_ipv,)``.

    Returns
    -------
    jax.Array
        Joint probabilities at the t-states, shape ``(n_t_vertices,)``.

    Notes
    -----
    Mirrors ``compute_sojourn_times_ffi``:
      - JSON passed as STATIC attribute (not batched).
      - theta and initial_ipv are buffers (batched by vmap).
      - vmap_method="expand_dims": OpenMP-parallel batch loop in the C
        handler.
    """
    # Register FFI targets (raises error if FFI disabled or unavailable)
    _register_ffi_targets()

    structure_str = _ensure_json_string(structure_json)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    initial_ipv = jnp.asarray(initial_ipv, dtype=jnp.float64)

    # 1D theta = single-particle daisy chain (legacy path).
    # 2D theta = batched daisy chain with shape (batch_size, n_epochs * param_length).
    # The C++ handler at graph_builder_ffi.cpp:1120-1135 already accepts
    # both 1D and 2D theta/ipv buffers. We allow either here so callers
    # (notably _daisy_chain_svgd_model with exposure) can dispatch one
    # batched FFI call instead of a Python-side lax.map fan-out.
    if theta.ndim not in (1, 2):
        raise ValueError(
            f"theta must be 1D or 2D, got shape {theta.shape}"
        )
    if initial_ipv.ndim not in (1, 2):
        raise ValueError(
            f"initial_ipv must be 1D or 2D, got shape {initial_ipv.shape}"
        )

    # Output shape: (n_t_vertices,). We need to extract n_t_vertices from
    # the JSON's _daisy_chain.t_vertex_indices length so the FFI call
    # knows the expected output shape.
    import json as _json_mod
    parsed = _json_mod.loads(structure_str) if isinstance(structure_json, str) \
        else structure_json
    if not isinstance(parsed, dict) or "_daisy_chain" not in parsed:
        raise ValueError(
            "structure_json must contain a top-level '_daisy_chain' object."
        )
    n_t = len(parsed["_daisy_chain"]["t_vertex_indices"])

    if theta.ndim == 2 or initial_ipv.ndim == 2:
        batch_size = theta.shape[0] if theta.ndim == 2 else initial_ipv.shape[0]
        result_shape = jax.ShapeDtypeStruct((batch_size, n_t), jnp.float64)
    else:
        result_shape = jax.ShapeDtypeStruct((n_t,), jnp.float64)

    ffi_fn = jax.ffi.ffi_call(
        "ptd_daisy_chain_joint_probs",
        result_shape,
        vmap_method="expand_dims",
    )
    return ffi_fn(
        theta,
        initial_ipv,
        structure_json=structure_str,
    )


def compute_daisy_chain_sojourn_ffi(
    structure_json: 'str | dict',
    sojourn_structure_json: 'str | dict',
    theta: 'jax.Array',
    initial_ipv: 'jax.Array',
) -> 'jax.Array':
    """Daisy chain with a granularity-free FINAL-epoch read via JAX FFI.

    Like :func:`compute_daisy_chain_joint_probs_ffi`, but the final epoch is
    read off the no-trapping ``sojourn_structure_json`` graph as
    ``r_v * expected_sojourn(v) * handoff_mass`` (exact, granularity-free, no
    ``t_eval``) instead of a long-time ``stop_probability`` forward solve.
    ``structure_json``'s ``"_daisy_chain"`` block must additionally carry
    ``sojourn_jsp_gather`` (len n_sojourn_ipv) and ``sojourn_t_indices``
    (len n_t, ordered to match ``t_vertex_indices``).

    Parameters mirror :func:`compute_daisy_chain_joint_probs_ffi` plus the
    extra ``sojourn_structure_json``. Output shape ``(n_t,)`` or ``(batch, n_t)``.
    """
    _register_ffi_targets()

    structure_str = _ensure_json_string(structure_json)
    sojourn_str = _ensure_json_string(sojourn_structure_json)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    initial_ipv = jnp.asarray(initial_ipv, dtype=jnp.float64)

    if theta.ndim not in (1, 2):
        raise ValueError(f"theta must be 1D or 2D, got shape {theta.shape}")
    if initial_ipv.ndim not in (1, 2):
        raise ValueError(f"initial_ipv must be 1D or 2D, got shape {initial_ipv.shape}")

    import json as _json_mod
    parsed = _json_mod.loads(structure_str) if isinstance(structure_json, str) \
        else structure_json
    if not isinstance(parsed, dict) or "_daisy_chain" not in parsed:
        raise ValueError("structure_json must contain a top-level '_daisy_chain' object.")
    n_t = len(parsed["_daisy_chain"]["sojourn_t_indices"])

    if theta.ndim == 2 or initial_ipv.ndim == 2:
        batch_size = theta.shape[0] if theta.ndim == 2 else initial_ipv.shape[0]
        result_shape = jax.ShapeDtypeStruct((batch_size, n_t), jnp.float64)
    else:
        result_shape = jax.ShapeDtypeStruct((n_t,), jnp.float64)

    ffi_fn = jax.ffi.ffi_call(
        "ptd_daisy_chain_sojourn",
        result_shape,
        vmap_method="expand_dims",
    )
    return ffi_fn(
        theta,
        initial_ipv,
        structure_json=structure_str,
        sojourn_structure_json=sojourn_str,
    )


__all__ = [
    'compute_pmf_ffi',
    'compute_moments_ffi',
    'compute_pmf_and_moments_ffi',
    'compute_pmf_multivariate_ffi',
    'compute_sojourn_times_ffi',
    'compute_daisy_chain_joint_probs_ffi',
    'compute_daisy_chain_sojourn_ffi',
    'backward_probabilities_ffi',
    'compute_reward_visit_probability_ffi',
]
