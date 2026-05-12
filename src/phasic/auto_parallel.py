"""
Automatic Parallelization for PtDAlgorithms

This module provides automatic detection and configuration of parallel computing
resources across different environments (Jupyter, SLURM, local machines).

Key Features:
- Automatic environment detection (Jupyter/IPython/SLURM/script)
- JAX device configuration with proper import timing
- Resource detection (CPUs, SLURM allocations)
- Parallelization strategy selection (pmap/vmap/none)
- Graceful handling of "JAX already imported" scenarios

Usage:
    >>> from phasic.auto_parallel import init_parallel
    >>>
    >>> # Explicit initialization (recommended at top of notebook)
    >>> config = init_parallel(cpus=8)
    >>>
    >>> # Or auto-detection
    >>> config = init_parallel()  # Uses all available CPUs

Author: PtDAlgorithms Team
Date: 2025-10-08
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any
import warnings

from .logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EnvironmentInfo:
    """
    Information about the execution environment.

    Attributes
    ----------
    env_type : str
        Type of environment: 'jupyter', 'ipython', 'slurm_multi', 'slurm_single', 'script'
    is_interactive : bool
        Whether running in an interactive shell (Jupyter/IPython)
    available_cpus : int
        Number of CPUs available for computation
    slurm_info : dict or None
        SLURM environment information (if running under SLURM)
    jax_already_imported : bool
        Whether JAX has already been imported (affects XLA_FLAGS configuration)
    """
    env_type: str = 'script'
    is_interactive: bool = False
    available_cpus: int = 1
    slurm_info: dict | None = None
    jax_already_imported: bool = False

    def __str__(self) -> str:
        """Pretty print environment information."""
        lines = [
            "Environment Information:",
            f"  Type: {self.env_type}",
            f"  Interactive: {self.is_interactive}",
            f"  Available CPUs: {self.available_cpus}",
        ]
        if self.slurm_info:
            lines.extend([
                f"  SLURM Job ID: {self.slurm_info.get('job_id', 'N/A')}",
                f"  SLURM Nodes: {self.slurm_info.get('node_count', 1)}",
            ])
        lines.append(f"  JAX imported: {self.jax_already_imported}")
        return "\n".join(lines)


@dataclass
class ParallelConfig:
    """
    Configuration for parallel computation.

    Attributes
    ----------
    device_count : int
        Total number of JAX devices available
    local_device_count : int
        Number of JAX devices on this node
    strategy : str
        Parallelization strategy: 'pmap', 'vmap', or 'none'
    env_info : EnvironmentInfo or None
        Environment information
    devices : list
        List of JAX devices
    """
    device_count: int = 1
    local_device_count: int = 1
    strategy: str = 'none'
    env_info: EnvironmentInfo | None = None
    devices: list = field(default_factory=list)

    def __str__(self) -> str:
        """Pretty print parallel configuration."""
        lines = [
            "Parallel Configuration:",
            f"  Strategy: {self.strategy}",
            f"  Total devices: {self.device_count}",
            f"  Local devices: {self.local_device_count}",
        ]
        if self.env_info:
            lines.append(f"  Environment: {self.env_info.env_type}")
        return "\n".join(lines)


# ============================================================================
# Environment Detection
# ============================================================================

def detect_environment() -> EnvironmentInfo:
    """
    Detect execution environment and available resources.

    Detection priority:
    1. Check if JAX already imported (affects configuration timing)
    2. Check for SLURM environment variables
    3. Check for IPython/Jupyter
    4. Fallback to script mode

    Returns
    -------
    EnvironmentInfo
        Detected environment information

    Examples
    --------
    >>> env = detect_environment()
    >>> print(f"Running in {env.env_type} with {env.available_cpus} CPUs")
    """
    # Check if JAX already imported
    jax_imported = 'jax' in sys.modules

    # Check for IPython/Jupyter
    is_interactive = False
    env_type = 'script'

    try:
        # Try to get IPython instance
        from IPython import get_ipython
        shell = get_ipython()

        if shell is not None:
            is_interactive = True
            # Detect Jupyter vs IPython
            shell_type = str(type(shell))
            if 'ZMQInteractiveShell' in shell_type:
                env_type = 'jupyter'
            else:
                env_type = 'ipython'
    except (ImportError, NameError):
        # Not in IPython/Jupyter
        pass

    # Check SLURM (overrides interactive detection)
    slurm_info = None
    if 'SLURM_JOB_ID' in os.environ:
        # Import here to avoid circular dependency
        from .distributed_utils import detect_slurm_environment
        slurm_info = detect_slurm_environment()

        if slurm_info.get('is_slurm', False):
            if slurm_info['num_processes'] > 1:
                env_type = 'slurm_multi'
            else:
                env_type = 'slurm_single'

    # Detect available CPUs
    if slurm_info and slurm_info.get('is_slurm', False):
        # SLURM: use allocated resources
        available_cpus = slurm_info['cpus_per_task']
    else:
        # Local: use all available CPUs (or env var override)
        available_cpus = int(os.environ.get('PTDALG_CPUS', os.cpu_count() or 1))

    env_info = EnvironmentInfo(
        env_type=env_type,
        is_interactive=is_interactive,
        available_cpus=available_cpus,
        slurm_info=slurm_info,
        jax_already_imported=jax_imported
    )

    logger.debug(f"Detected environment:\n{env_info}")
    return env_info


# ============================================================================
# JAX Configuration
# ============================================================================

def _determine_strategy(env_info: EnvironmentInfo, devices: list) -> str:
    """
    Determine best parallelization strategy.

    Parameters
    ----------
    env_info : EnvironmentInfo
        Environment information
    devices : list
        List of JAX devices

    Returns
    -------
    str
        'pmap' (multiple devices), 'vmap' (vectorization), or 'none' (serial)
    """
    device_count = len(devices)

    if device_count > 1:
        # Multiple devices available - use pmap
        return 'pmap'
    elif env_info.available_cpus > 1:
        # Single device but multiple CPUs - can benefit from vmap
        return 'vmap'
    else:
        # Single CPU - no parallelization
        return 'none'


def configure_jax_for_environment(env_info: EnvironmentInfo, enable_x64: bool = True) -> ParallelConfig:
    """
    Configure JAX based on environment information.

    Handles three scenarios:
    1. JAX not imported: Set XLA_FLAGS before import (optimal)
    2. JAX already imported: Use existing config, warn if suboptimal
    3. Multi-node SLURM: Initialize JAX distributed

    Parameters
    ----------
    env_info : EnvironmentInfo
        Environment information from detect_environment()
    enable_x64 : bool, default=True
        Enable 64-bit precision in JAX

    Returns
    -------
    ParallelConfig
        Configuration for parallel computation

    Examples
    --------
    >>> env = detect_environment()
    >>> config = configure_jax_for_environment(env)
    >>> print(f"Configured {config.device_count} devices with {config.strategy}")
    """
    # Configure JAX devices BEFORE importing/initializing
    if not env_info.jax_already_imported:
        logger.info(f"Configuring JAX for {env_info.available_cpus} devices...")
        from .distributed_utils import configure_jax_devices
        configure_jax_devices(env_info.available_cpus, platform="cpu")

    # Import JAX (but don't call jax.devices() yet!)
    import jax
    from jax import config

    if enable_x64:
        config.update('jax_enable_x64', True)
        logger.debug("JAX x64 precision enabled")

    # IMPORTANT: Initialize distributed BEFORE any jax.devices() calls
    # jax.devices() initializes the XLA backend, which must happen after distributed init
    if env_info.env_type == 'slurm_multi' and env_info.slurm_info:
        from .distributed_utils import initialize_jax_distributed, get_coordinator_address

        coordinator_address = get_coordinator_address(env_info.slurm_info)

        logger.info("Initializing JAX distributed for multi-node SLURM...")
        initialize_jax_distributed(
            coordinator_address=coordinator_address,
            num_processes=env_info.slurm_info['num_processes'],
            process_id=env_info.slurm_info['process_id']
        )

    # Now safe to call jax.devices() - this initializes the XLA backend
    devices = jax.devices()
    local_devices = jax.local_devices()

    # Check if we got the expected number of devices
    if env_info.jax_already_imported and len(devices) < env_info.available_cpus:
        logger.warning(
            f"JAX already initialized with {len(devices)} device(s), "
            f"but {env_info.available_cpus} CPU(s) available.\n"
            f"  To use all CPUs, restart kernel and import phasic first:\n"
            f"    import phasic as pta\n"
            f"    pta.init_parallel(cpus={env_info.available_cpus})"
        )

    logger.info(f"JAX initialized with {len(devices)} devices")

    # Determine strategy
    strategy = _determine_strategy(env_info, devices)

    config = ParallelConfig(
        device_count=len(devices),
        local_device_count=len(local_devices),
        strategy=strategy,
        env_info=env_info,
        devices=devices
    )

    logger.info(f"\n{config}")
    return config


# Note: the previous parallel_config / disable_parallel context
# managers, and the _global_parallel_config / get_parallel_config /
# set_parallel_config helpers, were removed in the config
# refactor. The SVGD particle strategy is now configured via
# phasic.configure(svgd_strategy='auto'|'pmap'|'vmap'|'none'), and
# the same `with configure(...)` block provides the
# temporary-override semantics the old context managers offered.


__all__ = [
    'EnvironmentInfo',
    'ParallelConfig',
    'detect_environment',
    'configure_jax_for_environment',
]
