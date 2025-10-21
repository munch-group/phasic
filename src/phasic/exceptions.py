"""
Exception classes for phasic.

Provides clear, actionable error messages when features are unavailable
or incorrectly configured.
"""


class PTDAlgorithmsError(Exception):
    """Base exception for all phasic errors."""
    pass


class PTDConfigError(PTDAlgorithmsError):
    """
    Configuration error with suggested fixes.

    Raised when the user requests functionality that is not available
    or when the configuration is invalid.

    Examples
    --------
    >>> import phasic as ptd
    >>> ptd.configure(jax=True, ffi=True)  # FFI not available
    PTDConfigError: ffi=True but FFI is currently disabled.
      Reason: Memory corruption bug (see FFI_MEMORY_CORRUPTION_FIX.md)
      Fix: Set ffi=False
    """
    pass


class PTDBackendError(PTDAlgorithmsError):
    """
    Backend not available error.

    Raised when a specific computation backend (JAX, FFI, C++) is requested
    but not available on the system.

    Examples
    --------
    >>> from phasic.ffi_wrappers import compute_pmf_ffi
    >>> pmf = compute_pmf_ffi(..., backend='ffi')
    PTDBackendError: FFI backend requested but not available.
      FFI is currently disabled (see FFI_MEMORY_CORRUPTION_FIX.md).
      Available backends: ['jax', 'cpp']
    """
    pass


class PTDFeatureError(PTDAlgorithmsError):
    """
    Feature not available on this platform.

    Raised when a feature is requested that is not supported on the
    current platform (e.g., GPU features on CPU-only system).

    Examples
    --------
    >>> ptd.configure(platform='gpu')  # On CPU-only system
    PTDFeatureError: platform='gpu' requested but no GPU available.
      Available platforms: ['cpu']
      Install GPU support or set platform='cpu'
    """
    pass


class PTDJAXError(PTDAlgorithmsError):
    """
    JAX-specific error.

    Raised when JAX is required but not installed or when JAX operations fail.

    Examples
    --------
    >>> import phasic as ptd
    >>> ptd.configure(jax=True)  # JAX not installed
    PTDJAXError: jax=True but JAX not installed.
      Install: pip install jax jaxlib
      Or: phasic.configure(jax=False)
    """
    pass


__all__ = [
    'PTDAlgorithmsError',
    'PTDConfigError',
    'PTDBackendError',
    'PTDFeatureError',
    'PTDJAXError',
]
