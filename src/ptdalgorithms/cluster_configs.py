"""
Cluster Configuration Management

This module provides configuration management for different cluster setups.
Configurations can be defined in YAML files and loaded programmatically.

Usage:
    >>> from ptdalgorithms.cluster_configs import load_config, get_default_config
    >>>
    >>> # Load from YAML file
    >>> config = load_config("docs/examples/slurm_configs/production.yaml")
    >>>
    >>> # Or use defaults
    >>> config = get_default_config("small")
    >>>
    >>> print(config.nodes, config.cpus_per_node)

Author: PtDAlgorithms Team
Date: 2025-10-07
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not installed. YAML config loading disabled.")


@dataclass
class ClusterConfig:
    """
    Configuration for a SLURM cluster setup.

    Attributes
    ----------
    name : str
        Name of this configuration
    nodes : int
        Number of nodes (machines) to request
    cpus_per_node : int
        CPUs per node (devices per node)
    memory_per_cpu : str
        Memory per CPU (e.g., "4G", "8G")
    time_limit : str
        Maximum job runtime (e.g., "01:00:00", "04:00:00")
    partition : str
        SLURM partition/queue name
    qos : str, optional
        Quality of service
    coordinator_port : int
        Port for JAX coordinator
    platform : str
        Platform type: "cpu" or "gpu"
    gpus_per_node : int, optional
        Number of GPUs per node (if platform="gpu")
    network_interface : str, optional
        Network interface for inter-node communication (e.g., "ib0", "eth0")
    extra_sbatch_options : Dict[str, str]
        Additional SBATCH options
    env_vars : Dict[str, str]
        Environment variables to set
    modules_to_load : List[str]
        Modules to load before execution
    """
    name: str = "default"
    nodes: int = 1
    cpus_per_node: int = 8
    memory_per_cpu: str = "4G"
    time_limit: str = "01:00:00"
    partition: str = "compute"
    qos: Optional[str] = None
    coordinator_port: int = 12345
    platform: str = "cpu"
    gpus_per_node: Optional[int] = None
    network_interface: Optional[str] = None
    extra_sbatch_options: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    modules_to_load: List[str] = field(default_factory=list)

    @property
    def total_devices(self) -> int:
        """Total number of compute devices."""
        if self.platform == "gpu" and self.gpus_per_node:
            return self.nodes * self.gpus_per_node
        return self.nodes * self.cpus_per_node

    @property
    def total_cpus(self) -> int:
        """Total number of CPUs across all nodes."""
        return self.nodes * self.cpus_per_node

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = [
            f"Cluster Configuration: {self.name}",
            f"  Nodes: {self.nodes}",
            f"  CPUs per node: {self.cpus_per_node}",
            f"  Total devices: {self.total_devices}",
            f"  Memory per CPU: {self.memory_per_cpu}",
            f"  Time limit: {self.time_limit}",
            f"  Partition: {self.partition}",
            f"  Platform: {self.platform}",
        ]
        if self.gpus_per_node:
            lines.append(f"  GPUs per node: {self.gpus_per_node}")
        if self.network_interface:
            lines.append(f"  Network: {self.network_interface}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_yaml(self, filepath: Path):
        """Save configuration to YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML required for YAML export. Install with: pip install pyyaml")

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {filepath}")


def load_config(filepath: Path) -> ClusterConfig:
    """
    Load cluster configuration from YAML file.

    Parameters
    ----------
    filepath : Path
        Path to YAML configuration file

    Returns
    -------
    ClusterConfig
        Loaded configuration

    Examples
    --------
    >>> config = load_config("docs/examples/slurm_configs/production.yaml")
    >>> print(config.nodes, config.cpus_per_node)
    """
    if not HAS_YAML:
        raise ImportError("PyYAML required for YAML loading. Install with: pip install pyyaml")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {filepath}")
    return ClusterConfig(**data)


def get_default_config(profile: str = "small") -> ClusterConfig:
    """
    Get a pre-defined configuration profile.

    Parameters
    ----------
    profile : str
        Configuration profile name. Options:
        - "debug": Single node for debugging (1 node, 4 CPUs)
        - "small": Small cluster (2 nodes, 8 CPUs each)
        - "medium": Medium cluster (4 nodes, 16 CPUs each)
        - "large": Large cluster (8 nodes, 16 CPUs each)
        - "production": Production setup (8 nodes, 32 CPUs each)

    Returns
    -------
    ClusterConfig
        Pre-configured cluster setup

    Examples
    --------
    >>> config = get_default_config("medium")
    >>> print(config)  # 4 nodes × 16 CPUs = 64 devices
    """
    profiles = {
        "debug": ClusterConfig(
            name="debug",
            nodes=1,
            cpus_per_node=4,
            memory_per_cpu="4G",
            time_limit="00:30:00",
            partition="debug",
        ),
        "small": ClusterConfig(
            name="small",
            nodes=2,
            cpus_per_node=8,
            memory_per_cpu="4G",
            time_limit="01:00:00",
            partition="compute",
        ),
        "medium": ClusterConfig(
            name="medium",
            nodes=4,
            cpus_per_node=16,
            memory_per_cpu="8G",
            time_limit="02:00:00",
            partition="compute",
        ),
        "large": ClusterConfig(
            name="large",
            nodes=8,
            cpus_per_node=16,
            memory_per_cpu="8G",
            time_limit="04:00:00",
            partition="compute",
        ),
        "production": ClusterConfig(
            name="production",
            nodes=8,
            cpus_per_node=32,
            memory_per_cpu="16G",
            time_limit="08:00:00",
            partition="compute",
            qos="high",
        ),
    }

    if profile not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile '{profile}'. Available: {available}")

    return profiles[profile]


def validate_config(config: ClusterConfig, num_particles: int) -> bool:
    """
    Validate configuration for a given number of particles.

    Parameters
    ----------
    config : ClusterConfig
        Cluster configuration to validate
    num_particles : int
        Number of particles to distribute

    Returns
    -------
    bool
        True if configuration is valid

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    total_devices = config.total_devices

    # Check particle count is divisible by device count
    if num_particles % total_devices != 0:
        raise ValueError(
            f"Number of particles ({num_particles}) must be divisible by "
            f"total devices ({total_devices}). "
            f"Try using {total_devices * (num_particles // total_devices)} "
            f"or {total_devices * ((num_particles // total_devices) + 1)} particles."
        )

    particles_per_device = num_particles // total_devices

    # Warn if too few particles per device
    if particles_per_device < 2:
        logger.warning(
            f"Only {particles_per_device} particle(s) per device. "
            f"Consider reducing device count or increasing particle count."
        )

    # Warn if too many particles per device
    if particles_per_device > 16:
        logger.warning(
            f"{particles_per_device} particles per device may cause memory issues. "
            f"Consider increasing device count."
        )

    logger.info(f"Configuration valid: {num_particles} particles → {particles_per_device} per device")
    return True


def suggest_config(num_particles: int, particles_per_device: int = 4) -> ClusterConfig:
    """
    Suggest optimal cluster configuration for a given particle count.

    Parameters
    ----------
    num_particles : int
        Number of particles for inference
    particles_per_device : int, default=4
        Target particles per device (2-8 recommended)

    Returns
    -------
    ClusterConfig
        Suggested configuration

    Examples
    --------
    >>> config = suggest_config(num_particles=128)
    >>> print(config)  # Suggests 4 nodes × 8 CPUs = 32 devices
    """
    target_devices = num_particles // particles_per_device

    # Adjust to ensure divisibility
    while num_particles % target_devices != 0:
        target_devices += 1

    # Determine node configuration
    # Prefer 8 or 16 CPUs per node
    if target_devices <= 8:
        nodes = 1
        cpus_per_node = target_devices
    elif target_devices <= 16:
        nodes = 2
        cpus_per_node = target_devices // 2
    elif target_devices <= 64:
        nodes = target_devices // 8
        cpus_per_node = 8
    else:
        nodes = target_devices // 16
        cpus_per_node = 16

    config = ClusterConfig(
        name="auto_suggested",
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        memory_per_cpu="8G",
        time_limit="02:00:00",
    )

    logger.info(f"Suggested configuration for {num_particles} particles:")
    logger.info(f"  {nodes} nodes × {cpus_per_node} CPUs = {target_devices} devices")
    logger.info(f"  {num_particles // target_devices} particles per device")

    return config


__all__ = [
    'ClusterConfig',
    'load_config',
    'get_default_config',
    'validate_config',
    'suggest_config',
]
