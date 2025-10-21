"""
Distributed Computing Utilities for JAX

This module provides utilities for distributed computing with JAX, with special
support for SLURM clusters. It encapsulates all the boilerplate code needed for
multi-node parallelization, making it easy to scale from single-machine to
cluster computing.

Key Features:
- Auto-detection of SLURM environment
- JAX distributed initialization
- Coordinator setup (for multi-node)
- Device configuration (CPU/GPU)
- Error handling and validation

Usage:
    >>> from phasic.distributed_utils import initialize_distributed
    >>>
    >>> # That's it! All boilerplate handled automatically
    >>> dist_info = initialize_distributed()
    >>>
    >>> # Use distributed info in your code
    >>> print(f"Process {dist_info.process_id}/{dist_info.num_processes}")
    >>> print(f"Local devices: {dist_info.local_device_count}")
    >>> print(f"Global devices: {dist_info.global_device_count}")

Author: PtDAlgorithms Team
Date: 2025-10-07
"""

import os
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """
    Configuration for distributed computing.

    Attributes
    ----------
    num_processes : int
        Total number of processes (nodes) in the cluster
    process_id : int
        Rank of this process (0 to num_processes-1)
    cpus_per_task : int
        Number of CPUs per process (local devices per node)
    coordinator_address : str
        Address of the coordinator node (host:port)
    coordinator_port : int
        Port for coordinator communication
    job_id : Optional[str]
        SLURM job ID (if running under SLURM)
    local_device_count : int
        Number of devices on this node
    global_device_count : int
        Total number of devices across all nodes
    local_devices : List
        List of local JAX devices
    global_devices : List
        List of all JAX devices (across all nodes)
    is_coordinator : bool
        True if this is the coordinator process (rank 0)
    platform : str
        Platform type ('cpu' or 'gpu')
    """
    num_processes: int = 1
    process_id: int = 0
    cpus_per_task: int = 1
    coordinator_address: str = "localhost:12345"
    coordinator_port: int = 12345
    job_id: Optional[str] = None
    local_device_count: int = 1
    global_device_count: int = 1
    local_devices: List = field(default_factory=list)
    global_devices: List = field(default_factory=list)
    is_coordinator: bool = True
    platform: str = "cpu"

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = [
            "Distributed Configuration:",
            f"  Job ID: {self.job_id or 'N/A'}",
            f"  Process: {self.process_id}/{self.num_processes}",
            f"  Coordinator: {self.coordinator_address} {'(this node)' if self.is_coordinator else ''}",
            f"  Local devices: {self.local_device_count}",
            f"  Global devices: {self.global_device_count}",
            f"  Platform: {self.platform}",
        ]
        return "\n".join(lines)


def detect_slurm_environment() -> Dict[str, Any]:
    """
    Detect and parse SLURM environment variables.

    Returns
    -------
    dict
        Dictionary with SLURM configuration:
        - 'is_slurm': bool - Whether running under SLURM
        - 'job_id': str - Job ID
        - 'process_id': int - Process rank (SLURM_PROCID)
        - 'num_processes': int - Total processes (SLURM_NTASKS)
        - 'cpus_per_task': int - CPUs per task
        - 'nodelist': str - List of nodes
        - 'node_count': int - Number of nodes
    """
    env = {}

    # Check if running under SLURM
    env['is_slurm'] = 'SLURM_JOB_ID' in os.environ

    if not env['is_slurm']:
        logger.info("Not running under SLURM - using single-node setup")
        return env

    # Parse SLURM environment variables
    env['job_id'] = os.environ.get('SLURM_JOB_ID')
    env['process_id'] = int(os.environ.get('SLURM_PROCID', 0))
    env['num_processes'] = int(os.environ.get('SLURM_NTASKS', 1))
    env['cpus_per_task'] = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    env['nodelist'] = os.environ.get('SLURM_JOB_NODELIST', '')
    env['node_count'] = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))

    logger.info(f"SLURM environment detected:")
    logger.info(f"  Job ID: {env['job_id']}")
    logger.info(f"  Process: {env['process_id']}/{env['num_processes']}")
    logger.info(f"  CPUs per task: {env['cpus_per_task']}")
    logger.info(f"  Nodes: {env['node_count']}")

    return env


def get_coordinator_address(slurm_env: Dict[str, Any], port: int = 12345) -> str:
    """
    Determine the coordinator address for JAX distributed.

    For SLURM clusters, the first node in the allocation becomes the coordinator.
    For single-node setups, uses localhost.

    Parameters
    ----------
    slurm_env : dict
        SLURM environment from detect_slurm_environment()
    port : int, default=12345
        Port for coordinator communication

    Returns
    -------
    str
        Coordinator address in format "host:port"
    """
    # Check if manually specified
    if 'SLURM_COORDINATOR_ADDRESS' in os.environ:
        host = os.environ['SLURM_COORDINATOR_ADDRESS']
        logger.info(f"Using manually specified coordinator: {host}")
        return f"{host}:{port}"

    # Single node setup
    if not slurm_env.get('is_slurm', False):
        return f"localhost:{port}"

    # Multi-node SLURM setup
    nodelist = slurm_env.get('nodelist', '')
    if not nodelist:
        logger.warning("No nodelist found, using localhost")
        return f"localhost:{port}"

    try:
        # Use scontrol to expand nodelist
        result = subprocess.run(
            ['scontrol', 'show', 'hostnames', nodelist],
            capture_output=True,
            text=True,
            check=True
        )
        nodes = result.stdout.strip().split('\n')
        if nodes and nodes[0]:
            coordinator_host = nodes[0]
            logger.info(f"Coordinator node: {coordinator_host}")
            return f"{coordinator_host}:{port}"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Could not expand nodelist: {e}")

    # Fallback
    logger.warning("Using localhost as fallback")
    return f"localhost:{port}"


def configure_jax_devices(num_devices: int, platform: str = "cpu"):
    """
    Configure JAX device count using XLA_FLAGS.

    This must be called before importing JAX.

    Parameters
    ----------
    num_devices : int
        Number of devices to create on this node
    platform : str, default="cpu"
        Platform type: "cpu" or "gpu"
    """
    # Set device count
    xla_flags = os.environ.get('XLA_FLAGS', '')
    device_flag = f"--xla_force_host_platform_device_count={num_devices}"

    # Add or update device count flag
    if '--xla_force_host_platform_device_count' in xla_flags:
        # Replace existing flag
        import re
        xla_flags = re.sub(
            r'--xla_force_host_platform_device_count=\d+',
            device_flag,
            xla_flags
        )
    else:
        # Add new flag
        xla_flags = f"{xla_flags} {device_flag}".strip()

    # Add CPU optimization flags
    if platform == "cpu":
        if '--xla_cpu_multi_thread_eigen' not in xla_flags:
            xla_flags = f"{xla_flags} --xla_cpu_multi_thread_eigen=true"

    os.environ['XLA_FLAGS'] = xla_flags
    os.environ['JAX_PLATFORMS'] = platform

    logger.info(f"Configured JAX for {num_devices} {platform.upper()} devices")
    logger.debug(f"XLA_FLAGS: {xla_flags}")


def initialize_jax_distributed(
    coordinator_address: str,
    num_processes: int,
    process_id: int
):
    """
    Initialize JAX distributed computing.

    This enables multi-node parallelization where pmap/pjit operations
    are distributed across all nodes in the cluster.

    Parameters
    ----------
    coordinator_address : str
        Address of coordinator node (host:port)
    num_processes : int
        Total number of processes in the cluster
    process_id : int
        Rank of this process (0 to num_processes-1)
    """
    import jax

    logger.info("Initializing JAX distributed...")
    logger.info(f"  Coordinator: {coordinator_address}")
    logger.info(f"  Process: {process_id}/{num_processes}")

    try:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
        logger.info("JAX distributed initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize JAX distributed: {e}")
        raise


def initialize_distributed(
    cpus_per_task: Optional[int] = None,
    coordinator_port: int = 12345,
    platform: str = "cpu",
    enable_x64: bool = True
) -> DistributedConfig:
    """
    Initialize distributed computing with automatic environment detection.

    This is the main entry point for distributed computing. It handles:
    - SLURM environment detection
    - Coordinator setup
    - JAX distributed initialization
    - Device configuration

    For single-node setups, it creates multiple local devices.
    For multi-node SLURM setups, it initializes JAX distributed.

    Parameters
    ----------
    cpus_per_task : int, optional
        Number of CPUs per task (devices per node).
        If None, auto-detected from SLURM_CPUS_PER_TASK or defaults to 1.
    coordinator_port : int, default=12345
        Port for coordinator communication (multi-node only)
    platform : str, default="cpu"
        Platform type: "cpu" or "gpu"
    enable_x64 : bool, default=True
        Enable 64-bit precision in JAX

    Returns
    -------
    DistributedConfig
        Configuration object with all distributed computing information

    Examples
    --------
    >>> # Single-node with 8 CPUs
    >>> dist_info = initialize_distributed(cpus_per_task=8)
    >>>
    >>> # Multi-node SLURM (auto-detected)
    >>> dist_info = initialize_distributed()
    >>>
    >>> # Use in your code
    >>> if dist_info.is_coordinator:
    >>>     print("I am the coordinator!")
    """
    # Detect SLURM environment
    slurm_env = detect_slurm_environment()

    # Determine number of devices per node
    if cpus_per_task is None:
        if slurm_env.get('is_slurm', False):
            cpus_per_task = slurm_env['cpus_per_task']
        else:
            cpus_per_task = int(os.environ.get('NUM_DEVICES', 1))

    # Configure JAX devices before importing jax
    configure_jax_devices(cpus_per_task, platform)

    # Now safe to import JAX
    import jax
    from jax import config

    # Enable x64 if requested
    if enable_x64:
        config.update('jax_enable_x64', True)
        logger.info("JAX x64 precision enabled")

    # Create configuration object
    dist_config = DistributedConfig(
        cpus_per_task=cpus_per_task,
        coordinator_port=coordinator_port,
        platform=platform,
    )

    # Multi-node SLURM setup
    if slurm_env.get('is_slurm', False) and slurm_env['num_processes'] > 1:
        dist_config.num_processes = slurm_env['num_processes']
        dist_config.process_id = slurm_env['process_id']
        dist_config.job_id = slurm_env['job_id']
        dist_config.is_coordinator = (dist_config.process_id == 0)

        # Get coordinator address
        coordinator_host_port = get_coordinator_address(slurm_env, coordinator_port)
        dist_config.coordinator_address = coordinator_host_port

        # Initialize JAX distributed
        initialize_jax_distributed(
            coordinator_address=coordinator_host_port,
            num_processes=dist_config.num_processes,
            process_id=dist_config.process_id
        )
    else:
        # Single-node setup
        logger.info("Single-node setup - no distributed initialization needed")
        dist_config.num_processes = 1
        dist_config.process_id = 0
        dist_config.is_coordinator = True

    # Get device information
    dist_config.local_devices = jax.local_devices()
    dist_config.global_devices = jax.devices()
    dist_config.local_device_count = len(dist_config.local_devices)
    dist_config.global_device_count = len(dist_config.global_devices)

    # Log configuration
    logger.info("\n" + str(dist_config))

    return dist_config


__all__ = [
    'DistributedConfig',
    'initialize_distributed',
    'detect_slurm_environment',
    'get_coordinator_address',
    'configure_jax_devices',
    'initialize_jax_distributed',
]
