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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
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


def _get_available_backends() -> list[str]:
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


def _get_available_platforms() -> list[str]:
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


def _check_mpfr_available() -> bool:
    """Check if MPFR high-precision support is available."""
    try:
        from . import phasic_pybind
        # MPFR is available if HAVE_MPFR was defined at compile time
        # For now, we check if libmpfr is linked (via otool/ldd would work, but complex)
        # Instead, we'll just assume it's available if phasic_pybind loaded successfully
        # A proper check could query a C++ function that returns #ifdef HAVE_MPFR
        return True  # If built with MPFR, it's always linked
    except ImportError:
        return False


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
    force_high_precision : bool, default=False
        Force MPFR high-precision arithmetic for all trace evaluations.
        Raises error if MPFR not available. Auto-activation at condition > 1e20 still occurs.
    mpfr_precision_bits : int, default=0
        MPFR precision in bits (0 = auto-determine from condition number).
        Examples: 128 (standard), 256 (high), 512 (very high), 1024 (extreme).
    condition_threshold : float, default=1e20
        Condition number threshold for auto-activating MPFR.
        Lower values are more conservative. Set to inf to disable auto-activation.
    enable_condition_warnings : bool, default=True
        Enable/disable warnings about ill-conditioned operations.

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

    # High-precision arithmetic settings (MPFR)
    force_high_precision: bool = False  # Force MPFR for all trace evaluations
    mpfr_precision_bits: int = 0  # MPFR precision in bits (0 = auto-determine from condition number)
    condition_threshold: float = 1e12  # Auto-activate MPFR when condition > threshold (lowered from 1e20 for better default)
    enable_condition_warnings: bool = True  # Log warnings for ill-conditioned operations

    # Hierarchical SCC composer settings.
    #
    # The composer routes Graph.expected_waiting_time through a
    # per-SCC compose path with disk-cached per-SCC PRCs. None
    # means "leave the corresponding env var untouched"; a
    # concrete value is written into os.environ on validate().
    hierar_elimination: bool = False  # PHASIC_HIERAR_ELIMINATION
    min_scc_size_to_cache: int | None = None  # PHASIC_MIN_SCC_SIZE_TO_CACHE; None = use C default (4)
    max_parallel_sccs: int | None = None  # PHASIC_MAX_PARALLEL_SCCS; None = OpenMP default (no cap)

    # On-disk cache settings (parameterised reward compute graph
    # + per-SCC PRCs share ~/.phasic_cache/).
    disable_cache: bool = False  # PHASIC_DISABLE_CACHE
    cache_dir: str | None = None  # PHASIC_CACHE_DIR; None = default ($HOME/.phasic_cache)

    # OpenMP thread count for the SCC composer's parallel loop
    # and any other OpenMP regions inside phasic. None = leave
    # OMP_NUM_THREADS as-is (phasic's import-time auto-detection
    # has already set it to SLURM_CPUS_PER_TASK / os.cpu_count()
    # if it was unset). Setting a concrete value writes the env
    # var, but note: OpenMP reads OMP_NUM_THREADS at library
    # load time, so changing it via configure() AFTER any
    # parallel region has run may not take effect until a fresh
    # Python process. Most reliable: set in shell before launch.
    omp_num_threads: int | None = None  # OMP_NUM_THREADS

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

        Notes
        -----
        Logs a warning if strict=False and requested features are unavailable.
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

        # Check MPFR high-precision settings
        if self.force_high_precision:
            if not _check_mpfr_available():
                msg = (
                    "force_high_precision=True but MPFR not available.\n"
                    "  MPFR is required for high-precision arithmetic.\n"
                    "\n"
                    "To rebuild with MPFR:\n"
                    "  pixi add mpfr gmp pkg-config\n"
                    "  pixi install\n"
                    "  pixi run install-dev\n"
                    "\n"
                    "Or via conda:\n"
                    "  conda install mpfr gmp pkg-config -c conda-forge\n"
                    "  pip install --no-build-isolation --force-reinstall --no-deps .\n"
                    "\n"
                    "Or disable high-precision mode:\n"
                    "  phasic.configure(force_high_precision=False)"
                )
                errors.append(msg)
            # Set environment variable for C code
            os.environ['PHASIC_FORCE_MPFR'] = '1'
        else:
            os.environ.pop('PHASIC_FORCE_MPFR', None)

        if self.mpfr_precision_bits < 0:
            errors.append("mpfr_precision_bits must be non-negative (0 = auto)")
        elif self.mpfr_precision_bits > 0 and self.mpfr_precision_bits < 53:
            errors.append("mpfr_precision_bits must be >= 53 (double precision) or 0 for auto")
        elif self.mpfr_precision_bits > 0:
            os.environ['PHASIC_MPFR_BITS'] = str(self.mpfr_precision_bits)
        else:
            os.environ.pop('PHASIC_MPFR_BITS', None)

        if self.condition_threshold <= 1.0:
            errors.append("condition_threshold must be > 1.0")
        else:
            os.environ['PHASIC_CONDITION_THRESHOLD'] = str(self.condition_threshold)

        if not self.enable_condition_warnings:
            os.environ['PHASIC_DISABLE_CONDITION_WARNINGS'] = '1'
        else:
            os.environ.pop('PHASIC_DISABLE_CONDITION_WARNINGS', None)

        # Hierarchical SCC composer settings. configure() always
        # wins over a pre-existing env var: if the field is set
        # to a concrete value (not None / not False default), we
        # overwrite the env var; if the field is None, we leave
        # the env var alone (user may have set it explicitly in
        # the shell, e.g. via SLURM job script).
        if self.hierar_elimination:
            os.environ['PHASIC_HIERAR_ELIMINATION'] = '1'
        else:
            os.environ.pop('PHASIC_HIERAR_ELIMINATION', None)

        if self.min_scc_size_to_cache is not None:
            if self.min_scc_size_to_cache < 0:
                errors.append(
                    "min_scc_size_to_cache must be non-negative")
            else:
                os.environ['PHASIC_MIN_SCC_SIZE_TO_CACHE'] = str(
                    self.min_scc_size_to_cache)

        if self.max_parallel_sccs is not None:
            if self.max_parallel_sccs < 1:
                errors.append(
                    "max_parallel_sccs must be >= 1 (use None for no cap)")
            else:
                os.environ['PHASIC_MAX_PARALLEL_SCCS'] = str(
                    self.max_parallel_sccs)

        if self.disable_cache:
            os.environ['PHASIC_DISABLE_CACHE'] = '1'
        else:
            os.environ.pop('PHASIC_DISABLE_CACHE', None)

        if self.cache_dir is not None:
            os.environ['PHASIC_CACHE_DIR'] = str(self.cache_dir)

        if self.omp_num_threads is not None:
            if self.omp_num_threads < 1:
                errors.append("omp_num_threads must be >= 1")
            else:
                os.environ['OMP_NUM_THREADS'] = str(self.omp_num_threads)

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

    def get_available_options(self) -> dict[str, Any]:
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
            'mpfr': _check_mpfr_available(),  # High-precision arithmetic
            'backends': _get_available_backends(),
            'platforms': _get_available_platforms(),
        }

    @classmethod
    def jax_only(cls) -> PTDAlgorithmsConfig:
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
    def cpp_only(cls) -> PTDAlgorithmsConfig:
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
    def permissive(cls) -> PTDAlgorithmsConfig:
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
_global_config: PTDAlgorithmsConfig | None = None


def configure(**kwargs) -> None:
    """
    Configure phasic globally.

    Parameters
    ----------
    **kwargs
        Configuration options (see PTDAlgorithmsConfig for details).
        Valid options: jax, jit, ffi, openmp, strict, platform,
        backend, verbose, force_high_precision, mpfr_precision_bits,
        condition_threshold, enable_condition_warnings,
        hierar_elimination, min_scc_size_to_cache, max_parallel_sccs,
        disable_cache, cache_dir, omp_num_threads.

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

    >>> # Hierarchical SCC composer with custom controls
    >>> ptd.configure(
    ...     hierar_elimination=True,
    ...     min_scc_size_to_cache=8,   # only cache SCCs with synth >= 8 vertices
    ...     max_parallel_sccs=4,       # cap simultaneous SCC computes at 4
    ...     cache_dir="/scratch/phasic_cache",  # shared filesystem on cluster
    ... )

    >>> # Check what's available first
    >>> print(ptd.get_available_options())
    >>> ptd.configure(jax=True, jit=True)

    Notes
    -----
    When configure() and an environment variable both set the
    same field, configure() wins. Fields with value None do not
    overwrite the corresponding env var, so leaving a knob alone
    in configure() preserves shell-level overrides (useful for
    SLURM job scripts).
    """
    global _global_config

    # Validate kwargs up-front so PTDAlgorithmsConfig() doesn't
    # raise a generic TypeError for unknown options.
    valid_fields = {f.name for f in PTDAlgorithmsConfig.__dataclass_fields__.values()
                    if not f.name.startswith("_")}
    for key in kwargs:
        if key not in valid_fields:
            raise PTDConfigError(
                f"Unknown configuration option: {key}\n"
                f"Valid options: {', '.join(sorted(valid_fields))}"
            )

    # Create new config with provided kwargs, or update existing.
    if _global_config is None:
        _global_config = PTDAlgorithmsConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            setattr(_global_config, key, value)

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
        # Check environment variables for overrides
        kwargs = {}
        # if os.getenv('PHASIC_FFI') == '0':
        #     kwargs['ffi'] = False
        #     kwargs['openmp'] = False
        if os.getenv('PHASIC_JAX') == '0':
            kwargs['jax'] = False
            kwargs['jit'] = False

        # SCC + cache env-var overrides. Mirror the C-side
        # readers so get_config() reflects what the C code will
        # actually see at runtime.
        if os.getenv('PHASIC_HIERAR_ELIMINATION') == '1':
            kwargs['hierar_elimination'] = True
        if os.getenv('PHASIC_DISABLE_CACHE') == '1':
            kwargs['disable_cache'] = True
        cache_dir_env = os.getenv('PHASIC_CACHE_DIR')
        if cache_dir_env:
            kwargs['cache_dir'] = cache_dir_env
        min_scc = os.getenv('PHASIC_MIN_SCC_SIZE_TO_CACHE')
        if min_scc is not None:
            try:
                kwargs['min_scc_size_to_cache'] = int(min_scc)
            except ValueError:
                pass  # malformed env var; let C-side default kick in
        max_par = os.getenv('PHASIC_MAX_PARALLEL_SCCS')
        if max_par is not None:
            try:
                parsed = int(max_par)
                if parsed >= 1:
                    kwargs['max_parallel_sccs'] = parsed
            except ValueError:
                pass
        omp_threads = os.getenv('OMP_NUM_THREADS')
        if omp_threads is not None:
            try:
                parsed = int(omp_threads)
                if parsed >= 1:
                    kwargs['omp_num_threads'] = parsed
            except ValueError:
                pass

        # Create default config with env overrides
        _global_config = PTDAlgorithmsConfig(**kwargs)
        _global_config.validate()

    return _global_config


def get_available_options() -> dict[str, Any]:
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
