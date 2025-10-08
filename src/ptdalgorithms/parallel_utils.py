"""
Parallel Utilities for Auto-Parallelization

This module provides utilities for automatic parallelization of batch operations
using JAX's pmap and vmap transformations.

Key Features:
- Automatic data sharding for pmap
- Batch detection heuristics
- Parallel strategy decorators
- Device-aware data distribution

Usage:
    >>> from ptdalgorithms.parallel_utils import auto_parallel_batch
    >>>
    >>> @auto_parallel_batch
    >>> def compute_pdf(theta, times):
    >>>     # Your computation here
    >>>     pass
    >>>
    >>> # Automatically parallelized based on batch size
    >>> result = compute_pdf(theta_batch, times)

Author: PtDAlgorithms Team
Date: 2025-10-08
"""

import functools
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Optional JAX import
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False


# ============================================================================
# Data Sharding Utilities
# ============================================================================

def _pad_to_devices(arr: Any, n_devices: int) -> Any:
    """
    Pad array to make it divisible by number of devices.

    Parameters
    ----------
    arr : array_like
        Input array to pad
    n_devices : int
        Number of devices to shard across

    Returns
    -------
    array
        Padded array with shape divisible by n_devices
    """
    if not HAS_JAX:
        return arr

    arr = jnp.asarray(arr)
    batch_size = arr.shape[0]
    remainder = batch_size % n_devices

    if remainder == 0:
        return arr

    # Pad to next multiple of n_devices
    pad_size = n_devices - remainder
    pad_width = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
    return jnp.pad(arr, pad_width, mode='edge')


def _shard_to_devices(arr: Any, n_devices: int) -> Any:
    """
    Shard array across devices for pmap.

    Reshapes array from (batch_size, ...) to (n_devices, batch_per_device, ...)

    Parameters
    ----------
    arr : array_like
        Input array to shard
    n_devices : int
        Number of devices

    Returns
    -------
    array
        Sharded array with shape (n_devices, batch_per_device, ...)
    """
    if not HAS_JAX:
        return arr

    # Pad if necessary
    arr_padded = _pad_to_devices(arr, n_devices)

    # Reshape to (n_devices, batch_per_device, ...)
    new_shape = (n_devices, -1) + arr_padded.shape[1:]
    return arr_padded.reshape(new_shape)


def _unshard_from_devices(arr: Any, original_batch_size: int) -> Any:
    """
    Unshard array from pmap output.

    Reshapes from (n_devices, batch_per_device, ...) to (batch_size, ...)
    and removes padding.

    Parameters
    ----------
    arr : array_like
        Sharded array from pmap
    original_batch_size : int
        Original batch size before padding

    Returns
    -------
    array
        Unsharded array with original batch size
    """
    if not HAS_JAX:
        return arr

    # Reshape to (batch_size, ...)
    arr_flat = arr.reshape(-1, *arr.shape[2:])

    # Remove padding
    return arr_flat[:original_batch_size]


# ============================================================================
# Batch Detection
# ============================================================================

def is_batched(arg: Any) -> bool:
    """
    Check if input represents a batch of data.

    Heuristics:
    - Arrays with 2+ dimensions (first dim is batch)
    - Arrays with 1 dimension and size > 1 (could be batch of scalars)

    Parameters
    ----------
    arg : any
        Input to check

    Returns
    -------
    bool
        True if input appears to be batched
    """
    if not HAS_JAX:
        if isinstance(arg, np.ndarray):
            return arg.ndim >= 2 or (arg.ndim == 1 and len(arg) > 1)
        return False

    if isinstance(arg, (jnp.ndarray, np.ndarray)):
        # 2D or higher is definitely batched
        if arg.ndim >= 2:
            return True
        # 1D with size > 1 might be batched (e.g., batch of scalars)
        if arg.ndim == 1 and arg.size > 1:
            return True

    return False


def get_batch_size(arg: Any) -> int:
    """
    Get batch size from batched input.

    Parameters
    ----------
    arg : array_like
        Batched input

    Returns
    -------
    int
        Batch size (first dimension)
    """
    if hasattr(arg, 'shape'):
        return arg.shape[0]
    return 1


# ============================================================================
# Parallel Execution
# ============================================================================

def apply_pmap(func: Callable, args: Tuple, n_devices: int) -> Any:
    """
    Apply pmap parallelization to function.

    Shards inputs across devices, applies pmap, and unshards output.

    Parameters
    ----------
    func : callable
        Function to parallelize
    args : tuple
        Arguments to function
    n_devices : int
        Number of devices

    Returns
    -------
    any
        Unsharded result
    """
    if not HAS_JAX:
        raise ImportError("JAX required for pmap")

    # Get original batch size from first arg
    batch_size = get_batch_size(args[0]) if args else 1

    # Shard arguments
    sharded_args = []
    for arg in args:
        if is_batched(arg):
            sharded_args.append(_shard_to_devices(arg, n_devices))
        else:
            # Non-batched arg - broadcast to all devices
            sharded_args.append(arg)

    # Apply pmap
    pmapped_func = jax.pmap(func)
    sharded_result = pmapped_func(*sharded_args)

    # Unshard result
    result = _unshard_from_devices(sharded_result, batch_size)

    return result


def apply_vmap(func: Callable, args: Tuple) -> Any:
    """
    Apply vmap parallelization to function.

    Parameters
    ----------
    func : callable
        Function to vectorize
    args : tuple
        Arguments to function

    Returns
    -------
    any
        Vectorized result
    """
    if not HAS_JAX:
        raise ImportError("JAX required for vmap")

    # Determine which args are batched
    in_axes = []
    for arg in args:
        if is_batched(arg):
            in_axes.append(0)  # Batch along first axis
        else:
            in_axes.append(None)  # Not batched

    # Apply vmap
    vmapped_func = jax.vmap(func, in_axes=tuple(in_axes))
    result = vmapped_func(*args)

    return result


# ============================================================================
# Auto-Parallel Decorator
# ============================================================================

def auto_parallel_batch(func: Callable) -> Callable:
    """
    Decorator for automatic batch parallelization.

    Automatically applies pmap or vmap based on:
    1. Current parallel configuration
    2. Whether inputs are batched
    3. Available devices

    Parameters
    ----------
    func : callable
        Function to decorate. Should accept batched inputs.

    Returns
    -------
    callable
        Decorated function with automatic parallelization

    Examples
    --------
    >>> @auto_parallel_batch
    >>> def compute_pdf(theta, times):
    >>>     # Single computation
    >>>     return model(theta, times)
    >>>
    >>> # Automatically parallelized for batches
    >>> theta_batch = jnp.array([[1.0], [2.0], [3.0]])
    >>> times = jnp.array([1.0, 2.0, 3.0])
    >>> result = compute_pdf(theta_batch, times)  # Uses pmap or vmap

    Notes
    -----
    - For single inputs, runs serially (no overhead)
    - For batched inputs with pmap strategy, shards across devices
    - For batched inputs with vmap strategy, vectorizes
    - Falls back to manual looping if JAX not available
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if we have any batched inputs
        has_batch = any(is_batched(arg) for arg in args)

        if not has_batch:
            # Single input - no parallelization needed
            return func(*args, **kwargs)

        # Get parallel config (delayed import to avoid circular dependency)
        try:
            from .auto_parallel import get_parallel_config
            config = get_parallel_config()
        except ImportError:
            # If auto_parallel can't be imported, fall back to no config
            config = None

        if config is None or config.strategy == 'none':
            # No parallelization - use vmap if available, else loop
            if HAS_JAX:
                return apply_vmap(func, args)
            else:
                # Manual loop for numpy
                batch_size = get_batch_size(args[0])
                results = []
                for i in range(batch_size):
                    # Extract single item from each batched arg
                    single_args = []
                    for arg in args:
                        if is_batched(arg):
                            single_args.append(arg[i])
                        else:
                            single_args.append(arg)
                    results.append(func(*single_args, **kwargs))
                return np.array(results)

        elif config.strategy == 'pmap':
            # Use pmap across devices
            return apply_pmap(func, args, config.device_count)

        elif config.strategy == 'vmap':
            # Use vmap vectorization
            return apply_vmap(func, args)

        else:
            # Unknown strategy - fallback to vmap
            logger.warning(f"Unknown strategy '{config.strategy}', using vmap")
            return apply_vmap(func, args)

    return wrapper


# ============================================================================
# Batch Execution Helpers
# ============================================================================

def execute_batch(func: Callable,
                  batch_args: Tuple,
                  strategy: Optional[str] = None,
                  n_devices: Optional[int] = None) -> Any:
    """
    Execute function on batch with specified strategy.

    More explicit version of auto_parallel_batch for advanced use cases.

    Parameters
    ----------
    func : callable
        Function to execute
    batch_args : tuple
        Batched arguments
    strategy : str, optional
        'pmap', 'vmap', or 'serial'. If None, uses global config.
    n_devices : int, optional
        Number of devices for pmap. If None, uses global config.

    Returns
    -------
    any
        Batch results

    Examples
    --------
    >>> # Force pmap with 4 devices
    >>> result = execute_batch(compute_pdf, (theta_batch, times),
    ...                        strategy='pmap', n_devices=4)
    >>>
    >>> # Force vmap
    >>> result = execute_batch(compute_pdf, (theta_batch, times),
    ...                        strategy='vmap')
    """
    if strategy is None:
        # Use global config (with fallback if import fails)
        try:
            from .auto_parallel import get_parallel_config
            config = get_parallel_config()
            strategy = config.strategy if config else 'serial'
            n_devices = config.device_count if config else 1
        except ImportError:
            strategy = 'serial'
            n_devices = 1

    if strategy == 'pmap':
        if not HAS_JAX:
            raise ImportError("JAX required for pmap")
        return apply_pmap(func, batch_args, n_devices or 1)

    elif strategy == 'vmap':
        if not HAS_JAX:
            raise ImportError("JAX required for vmap")
        return apply_vmap(func, batch_args)

    else:  # serial
        # Manual loop
        batch_size = get_batch_size(batch_args[0])
        results = []
        for i in range(batch_size):
            single_args = []
            for arg in batch_args:
                if is_batched(arg):
                    single_args.append(arg[i] if HAS_JAX else arg[i])
                else:
                    single_args.append(arg)
            results.append(func(*single_args))

        if HAS_JAX:
            return jnp.array(results)
        else:
            return np.array(results)


__all__ = [
    'auto_parallel_batch',
    'execute_batch',
    'is_batched',
    'get_batch_size',
    'apply_pmap',
    'apply_vmap',
]
