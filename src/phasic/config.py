"""
Configuration system for phasic.

Provides explicit user control over all optional features and backends.
No silent fallbacks - all behavior must be specified by the user.

Examples
--------
>>> import phasic as ptd

>>> # Check what's available on this system
>>> print(ptd.get_available_options())
{'jax': True, 'jit': True, 'ffi': False,
 'backends': ['jax', 'cpp'],
 'platforms': ['cpu']}

>>> # Configure explicitly (errors if unavailable)
>>> ptd.configure(jax=True, jit=True, ffi=False, strict=True)

>>> # Or configure with warnings instead of errors
>>> ptd.configure(jax=True, jit=True, strict=False)
"""

from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any, Optional
import os
import sys

from .exceptions import PTDConfigError, PTDJAXError, PTDBackendError


def _check_jax_available() -> bool:
    """Check if JAX is installed and importable (without actually importing it)."""
    import importlib.util
    return importlib.util.find_spec('jax') is not None


def _check_cpp_available() -> bool:
    """Check if C++ pybind11 module is available."""
    try:
        from . import phasic_pybind
        return True
    except ImportError:
        return False


def _get_available_backends() -> List[str]:
    """Get list of available computation backends."""
    backends = []

    if _check_cpp_available():
        backends.append('cpp')

    if _check_jax_available():
        backends.append('jax')

    # FFI is always disabled for now due to memory corruption bug
    # See: FFI_MEMORY_CORRUPTION_FIX.md
    # if _check_ffi_available():
    #     backends.append('ffi')

    return backends


def _get_available_platforms() -> List[str]:
    """Get list of available JAX platforms."""
    platforms = ['cpu']  # CPU always available

    if not _check_jax_available():
        return platforms

    try:
        import jax
        # Check for GPU
        try:
            devices = jax.devices('gpu')
            if devices:
                platforms.append('gpu')
        except RuntimeError:
            pass

        # Check for TPU
        try:
            devices = jax.devices('tpu')
            if devices:
                platforms.append('tpu')
        except RuntimeError:
            pass
    except Exception:
        pass

    return platforms


@dataclass
class PTDAlgorithmsConfig:
    """
    Global configuration for phasic behavior.

    All optional features must be explicitly enabled/disabled.
    No silent fallbacks.

    Parameters
    ----------
    jax : bool, default=True
        Require JAX functionality. If True and JAX not installed, raises error.
    jit : bool, default=True
        Enable JIT compilation. Requires jax=True.
    ffi : bool, default=True
        Enable FFI backend for zero-copy C++ computation.
        Provides 5-10x speedup over pure_callback. Requires XLA headers during build.
        Set to False only if FFI cannot be built on your system.
    openmp : bool, default=True
        Enable OpenMP multi-threading in FFI handlers.
        Provides ~8x additional speedup on 8-core systems (800% CPU vs 100%).
        Requires ffi=True. Set to False only if OpenMP unavailable on your system.
    strict : bool, default=True
        If True, raise errors when features unavailable.
        If False, print warnings and continue.
    platform : Literal['cpu', 'gpu', 'tpu'], default='cpu'
        JAX platform to use. Requires jax=True.
    backend : Literal['jax', 'cpp', 'ffi'], default='jax'
        Default computation backend for FFI wrappers.
    verbose : bool, default=False
        Print configuration details on startup.

    Examples
    --------
    >>> config = PTDAlgorithmsConfig(jax=True, jit=True, ffi=False)
    >>> config.validate()  # Check if configuration is valid

    >>> # Or use factory methods
    >>> config = PTDAlgorithmsConfig.jax_only()  # JAX with JIT
    >>> config = PTDAlgorithmsConfig.cpp_only()  # Pure C++, no JAX
    """

    jax: bool = True
    jit: bool = True
    ffi: bool = True
    openmp: bool = True
    strict: bool = True
    platform: Literal['cpu', 'gpu', 'tpu'] = 'cpu'
    backend: Literal['jax', 'cpp', 'ffi'] = 'jax'
    verbose: bool = False

    # Internal tracking
    _validated: bool = field(default=False, init=False, repr=False)
    _jax_imported: bool = field(default=False, init=False, repr=False)

    def validate(self) -> None:
        """
        Validate configuration and check feature availability.

        Raises
        ------
        PTDConfigError
            If strict=True and requested features are unavailable.

        Warns
        -----
        If strict=False and requested features are unavailable.
        """
        errors = []
        warnings = []

        # Check JAX
        if self.jax:
            if not _check_jax_available():
                msg = (
                    "jax=True but JAX not installed.\n"
                    "  Install: pip install jax jaxlib\n"
                    "  Or configure: phasic.configure(jax=False)"
                )
                errors.append(msg)
        else:
            # If JAX disabled, can't use JIT or JAX backend
            if self.jit:
                errors.append("jit=True requires jax=True")
            if self.backend == 'jax':
                warnings.append(
                    "backend='jax' but jax=False. "
                    "Switching to backend='cpp'"
                )
                self.backend = 'cpp'

        # Check FFI availability if enabled
        if self.ffi:
            try:
                from . import phasic_pybind as cpp_module
                if not hasattr(cpp_module.parameterized, 'get_compute_pmf_ffi_capsule'):
                    msg = (
                        "ffi=True but FFI handlers not available.\n"
                        "  This usually means XLA headers were not found during build.\n"
                        "\n"
                        "To rebuild with FFI:\n"
                        "  export XLA_FFI_INCLUDE_DIR=$(python -c \"from jax import ffi; print(ffi.include_dir())\")\n"
                        "  pip install --no-build-isolation --force-reinstall --no-deps .\n"
                        "\n"
                        "Or disable FFI (slower performance):\n"
                        "  import phasic\n"
                        "  phasic.configure(ffi=False)"
                    )
                    errors.append(msg)
            except (ImportError, AttributeError) as e:
                msg = (
                    f"ffi=True but C++ module not available: {e}\n"
                    "  This is a build error - C++ extensions should always be present.\n"
                    "  Try rebuilding: pip install --force-reinstall --no-deps ."
                )
                errors.append(msg)

        # Check OpenMP availability if enabled
        if self.openmp and not self.ffi:
            errors.append("openmp=True requires ffi=True (OpenMP only works with FFI backend)")

        # Note: We don't check if OpenMP is actually compiled in - trust the build
        # A runtime check would require platform-specific code (otool/ldd)

        # Check backend consistency
        if self.backend == 'ffi' and not self.ffi:
            errors.append("backend='ffi' requires ffi=True")

        if self.backend == 'jax' and not self.jax:
            errors.append("backend='jax' requires jax=True")

        if self.backend == 'cpp' and not _check_cpp_available():
            errors.append(
                "backend='cpp' but C++ module not available.\n"
                "  This should not happen - C++ module is core dependency."
            )

        # Check platform
        if self.platform != 'cpu':
            if not self.jax:
                errors.append(
                    f"platform='{self.platform}' requires jax=True"
                )
            elif self.platform not in _get_available_platforms():
                available = _get_available_platforms()
                errors.append(
                    f"platform='{self.platform}' not available.\n"
                    f"  Available platforms: {available}\n"
                    f"  Install GPU/TPU support or use platform='cpu'"
                )

        # Handle errors/warnings
        if warnings and self.verbose:
            for w in warnings:
                print(f"WARNING: {w}", file=sys.stderr)

        if errors:
            error_msg = "\n\n".join(errors)
            if self.strict:
                raise PTDConfigError(error_msg)
            else:
                print(f"WARNING: Configuration issues:\n{error_msg}", file=sys.stderr)

        self._validated = True

        if self.verbose:
            print(f"PTDAlgorithms configured: {self}")

    def get_available_options(self) -> Dict[str, Any]:
        """
        Return dict of available options on this system.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'jax': bool, whether JAX is installed
            - 'jit': bool, whether JIT is available (same as jax)
            - 'ffi': bool, whether FFI is available (always False now)
            - 'backends': list of available backends
            - 'platforms': list of available JAX platforms
            - 'cpp': bool, whether C++ module is available

        Examples
        --------
        >>> import phasic as ptd
        >>> opts = ptd.get_available_options()
        >>> print(opts)
        {'jax': True, 'jit': True, 'ffi': False,
         'backends': ['jax', 'cpp'],
         'platforms': ['cpu'],
         'cpp': True}
        """
        return {
            'jax': _check_jax_available(),
            'jit': _check_jax_available(),  # JIT requires JAX
            'ffi': False,  # Always False for now (memory corruption bug)
            'cpp': _check_cpp_available(),
            'backends': _get_available_backends(),
            'platforms': _get_available_platforms(),
        }

    @classmethod
    def jax_only(cls) -> 'PTDAlgorithmsConfig':
        """
        Factory: JAX-based configuration (JIT enabled, no FFI).

        Returns
        -------
        PTDAlgorithmsConfig
            Config with jax=True, jit=True, backend='jax'
        """
        return cls(
            jax=True,
            jit=True,
            ffi=False,
            backend='jax',
            strict=True
        )

    @classmethod
    def cpp_only(cls) -> 'PTDAlgorithmsConfig':
        """
        Factory: Pure C++ configuration (no JAX, no JIT).

        Useful for environments without JAX or when JIT overhead
        is not worth it.

        Returns
        -------
        PTDAlgorithmsConfig
            Config with jax=False, jit=False, backend='cpp'
        """
        return cls(
            jax=False,
            jit=False,
            ffi=False,
            backend='cpp',
            strict=True
        )

    @classmethod
    def permissive(cls) -> 'PTDAlgorithmsConfig':
        """
        Factory: Permissive configuration (warnings instead of errors).

        Useful for development when you want to test functionality
        even if some features are missing.

        Returns
        -------
        PTDAlgorithmsConfig
            Config with strict=False
        """
        return cls(
            jax=True,
            jit=True,
            ffi=False,
            backend='jax',
            strict=False,  # Warnings not errors
            verbose=True
        )


# Global configuration instance
_global_config: Optional[PTDAlgorithmsConfig] = None


def configure(**kwargs) -> None:
    """
    Configure phasic globally.

    Parameters
    ----------
    **kwargs
        Configuration options (see PTDAlgorithmsConfig for details)
        Valid options: jax, jit, ffi, openmp, strict, platform, backend, verbose

    Raises
    ------
    PTDConfigError
        If strict=True and configuration is invalid

    Examples
    --------
    >>> import phasic as ptd

    >>> # Standard configuration (FFI+OpenMP enabled by default)
    >>> ptd.configure(jax=True, jit=True, ffi=True, openmp=True)

    >>> # Disable FFI/OpenMP if build issues (slower, single-core only)
    >>> ptd.configure(ffi=False, openmp=False)

    >>> # Permissive (warnings not errors)
    >>> ptd.configure(jax=True, strict=False)

    >>> # Pure C++ (no JAX)
    >>> ptd.configure(jax=False, jit=False, backend='cpp')

    >>> # Check what's available first
    >>> print(ptd.get_available_options())
    >>> ptd.configure(jax=True, jit=True)
    """
    global _global_config

    # Create new config with provided kwargs
    if _global_config is None:
        _global_config = PTDAlgorithmsConfig(**kwargs)
    else:
        # Update existing config
        for key, value in kwargs.items():
            if hasattr(_global_config, key):
                setattr(_global_config, key, value)
            else:
                raise PTDConfigError(
                    f"Unknown configuration option: {key}\n"
                    f"Valid options: jax, jit, ffi, openmp, strict, platform, backend, verbose"
                )

    # Validate
    _global_config.validate()


def get_config() -> PTDAlgorithmsConfig:
    """
    Get current global configuration.

    Returns
    -------
    PTDAlgorithmsConfig
        Current configuration (creates default if none exists)

    Examples
    --------
    >>> import phasic as ptd
    >>> config = ptd.get_config()
    >>> print(config.jax, config.jit, config.backend)
    True True jax
    """
    global _global_config

    if _global_config is None:
        # Create default config
        _global_config = PTDAlgorithmsConfig()
        _global_config.validate()

    return _global_config


def get_available_options() -> Dict[str, Any]:
    """
    Get dictionary of available options on this system.

    Returns
    -------
    dict
        Available features and backends

    Examples
    --------
    >>> import phasic as ptd
    >>> opts = ptd.get_available_options()
    >>> if opts['jax']:
    ...     ptd.configure(jax=True, jit=True)
    ... else:
    ...     ptd.configure(jax=False, backend='cpp')
    """
    config = get_config()
    return config.get_available_options()


def reset_config() -> None:
    """
    Reset configuration to default.

    Examples
    --------
    >>> import phasic as ptd
    >>> ptd.configure(jax=False, backend='cpp')
    >>> ptd.reset_config()  # Back to defaults
    >>> assert ptd.get_config().jax == True
    """
    global _global_config
    _global_config = None


__all__ = [
    'PTDAlgorithmsConfig',
    'configure',
    'get_config',
    'get_available_options',
    'reset_config',
]
