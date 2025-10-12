"""
JAX Compilation Configuration

Provides configurable optimization settings for JAX/XLA compilation with:
- Persistent cross-session caching
- Parallel compilation
- Optimization level control
- Config file save/load support
- Pre-defined presets
"""

import os
import platform
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json


class CompilationConfig:
    """
    Configuration for JAX compilation optimization settings.

    This class manages JAX/XLA environment variables to control compilation
    behavior, caching, and optimization levels.

    Parameters
    ----------
    cache_dir : Path or str, optional
        Directory for persistent compilation cache across Python sessions.
        Default: ~/.jax_cache
    optimization_level : int, optional
        XLA backend optimization level (0-3):
        - 0: No optimization (fastest compile, slowest runtime)
        - 1: Basic optimization (fast compile, decent runtime)
        - 2: Moderate optimization (balanced)
        - 3: Full optimization (slow compile, fastest runtime)
        Default: 3
    parallel_compile : bool, optional
        Enable parallel compilation using all CPU cores.
        Default: True
    min_cache_time : float, optional
        Minimum compilation time (seconds) to trigger caching.
        Compilations faster than this won't be cached.
        Default: 1.0
    enable_x64 : bool, optional
        Enable 64-bit precision for JAX arrays.
        Default: True
    platform : str, optional
        JAX platform: 'cpu', 'gpu', or 'tpu'.
        Default: 'cpu'
    cpu_threads : int or None, optional
        Number of CPU threads for parallel operations.
        If None, uses all available performance cores.
        Default: None (auto-detect)

    Examples
    --------
    >>> # Use default balanced settings
    >>> config = CompilationConfig()
    >>> config.apply()

    >>> # Fast compilation for development
    >>> config = CompilationConfig.fast_compile()
    >>> config.apply()

    >>> # Maximum runtime performance
    >>> config = CompilationConfig.max_performance()
    >>> config.apply()

    >>> # Custom configuration
    >>> config = CompilationConfig(
    ...     optimization_level=2,
    ...     cache_dir='/scratch/jax_cache'
    ... )
    >>> config.save_to_file('my_config.json')

    >>> # Load from file
    >>> config = CompilationConfig.load_from_file('my_config.json')
    >>> config.apply()
    """

    def __init__(
        self,
        cache_dir: Optional[Union[Path, str]] = None,
        optimization_level: int = 2,
        parallel_compile: bool = True,
        min_cache_time: float = 1.0,
        enable_x64: bool = True,
        platform: str = 'cpu',
        cpu_threads: Optional[int] = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.jax_cache'
        self.optimization_level = optimization_level
        self.parallel_compile = parallel_compile
        self.min_cache_time = min_cache_time
        self.enable_x64 = enable_x64
        self.platform = platform
        self.cpu_threads = cpu_threads or self._get_performance_cores()

        # Validate inputs
        if not 0 <= optimization_level <= 3:
            raise ValueError(f"optimization_level must be 0-3, got {optimization_level}")
        if platform not in ('cpu', 'gpu', 'tpu'):
            raise ValueError(f"platform must be 'cpu', 'gpu', or 'tpu', got {platform}")

    @staticmethod
    def _get_performance_cores() -> int:
        """Get number of performance cores on Apple Silicon, or total CPUs otherwise"""
        try:
            # Check if we're on Apple Silicon
            if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                import subprocess
                # Get P-cores (performance cores)
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.perflevel0.physicalcpu'],
                    capture_output=True, text=True, check=True
                )
                p_cores = int(result.stdout.strip())
                return p_cores
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass

        # Fallback to total CPU count
        return multiprocessing.cpu_count() or 1

    def apply(self, force: bool = False) -> None:
        """
        Apply configuration to environment variables.

        Parameters
        ----------
        force : bool, optional
            If True, overwrite existing environment variables.
            If False, only set variables that aren't already configured.
            Default: False

        Notes
        -----
        This should be called BEFORE importing JAX for full effect.
        Some settings (like XLA_FLAGS) cannot be changed after JAX is imported.
        """
        # JAX persistent compilation cache
        if force or 'JAX_COMPILATION_CACHE_DIR' not in os.environ:
            os.environ['JAX_COMPILATION_CACHE_DIR'] = str(self.cache_dir)

        # Cache thresholds
        if force or 'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS' not in os.environ:
            os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = str(self.min_cache_time)

        if force or 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES' not in os.environ:
            os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '0'

        # 64-bit precision
        if force or 'JAX_ENABLE_X64' not in os.environ:
            os.environ['JAX_ENABLE_X64'] = str(self.enable_x64)

        # Platform selection
        if force or 'JAX_PLATFORMS' not in os.environ:
            os.environ['JAX_PLATFORMS'] = self.platform

        # Build XLA_FLAGS
        xla_flags = []

        # Optimization level
        xla_flags.append(f'--xla_backend_optimization_level={self.optimization_level}')

        # Parallel compilation
        if self.parallel_compile:
            xla_flags.append('--xla_cpu_multi_thread_eigen=true')
            xla_flags.append(f'intra_op_parallelism_threads={self.cpu_threads}')
            xla_flags.append(f'inter_op_parallelism_threads={self.cpu_threads}')

        # Combine with existing XLA_FLAGS if not forcing
        existing_xla = os.environ.get('XLA_FLAGS', '')
        if force or not existing_xla:
            os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
        else:
            # Merge with existing flags (avoid duplicates)
            all_flags = existing_xla + ' ' + ' '.join(xla_flags)
            os.environ['XLA_FLAGS'] = all_flags

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'cache_dir': str(self.cache_dir),
            'optimization_level': self.optimization_level,
            'parallel_compile': self.parallel_compile,
            'min_cache_time': self.min_cache_time,
            'enable_x64': self.enable_x64,
            'platform': self.platform,
            'cpu_threads': self.cpu_threads
        }

    def save_to_file(self, path: Union[Path, str]) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        path : Path or str
            Path to save configuration file

        Examples
        --------
        >>> config = CompilationConfig(optimization_level=2)
        >>> config.save_to_file('my_config.json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Union[Path, str]) -> 'CompilationConfig':
        """
        Load configuration from JSON file.

        Parameters
        ----------
        path : Path or str
            Path to configuration file

        Returns
        -------
        CompilationConfig
            Loaded configuration

        Examples
        --------
        >>> config = CompilationConfig.load_from_file('my_config.json')
        >>> config.apply()
        """
        path = Path(path)

        with open(path, 'r') as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def fast_compile(cls) -> 'CompilationConfig':
        """
        Preset for fast compilation (prioritize compile speed).

        Best for:
        - Development and debugging
        - Rapid iteration
        - Small models where runtime performance isn't critical

        Settings:
        - Optimization level: 1 (basic optimization)
        - Parallel compile: True
        - Min cache time: 0.5s

        Returns
        -------
        CompilationConfig
            Configuration preset
        """
        return cls(
            optimization_level=1,
            parallel_compile=True,
            min_cache_time=0.5
        )

    @classmethod
    def balanced(cls) -> 'CompilationConfig':
        """
        Preset for balanced compile/runtime performance (default).

        Best for:
        - General use
        - Medium-sized models
        - Good balance between compile time and runtime speed

        Settings:
        - Optimization level: 2 (moderate optimization)
        - Parallel compile: True
        - Min cache time: 1.0s

        Returns
        -------
        CompilationConfig
            Configuration preset
        """
        return cls(
            optimization_level=2,
            parallel_compile=True,
            min_cache_time=1.0
        )

    @classmethod
    def max_performance(cls) -> 'CompilationConfig':
        """
        Preset for maximum runtime performance.

        Best for:
        - Production workloads
        - Large models
        - Long-running computations
        - When compile time is acceptable for best runtime speed

        Settings:
        - Optimization level: 3 (full optimization)
        - Parallel compile: True
        - Min cache time: 1.0s

        Returns
        -------
        CompilationConfig
            Configuration preset
        """
        return cls(
            optimization_level=3,
            parallel_compile=True,
            min_cache_time=1.0
        )

    def __repr__(self) -> str:
        return (
            f"CompilationConfig(\n"
            f"  cache_dir={self.cache_dir},\n"
            f"  optimization_level={self.optimization_level},\n"
            f"  parallel_compile={self.parallel_compile},\n"
            f"  min_cache_time={self.min_cache_time},\n"
            f"  enable_x64={self.enable_x64},\n"
            f"  platform='{self.platform}',\n"
            f"  cpu_threads={self.cpu_threads}\n"
            f")"
        )


# Global default configuration
_default_config = None


def get_default_config() -> CompilationConfig:
    """Get the global default compilation configuration"""
    global _default_config
    if _default_config is None:
        _default_config = CompilationConfig.balanced()
    return _default_config


def set_default_config(config: CompilationConfig) -> None:
    """Set the global default compilation configuration"""
    global _default_config
    _default_config = config
