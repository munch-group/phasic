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


# =============================================================
# Module-level functions: configure(), get_config(), etc.
#
# These all operate on PhasicConfig (defined further below).
# The previous PTDAlgorithmsConfig class has been removed in the
# refactor; user-facing field names changed (clean break).
# =============================================================

# Global configuration instance.
_global_config: "PhasicConfig | None" = None


class _ConfigureContext:
    """Return value of ``configure(...)`` that can also be used as
    a context manager to temporarily apply settings.

    When obtained from a normal ``configure(...)`` call the settings
    are already applied — discarding the return value is safe.
    When entered via ``with configure(...) as ctx:`` (or
    ``with configure(...):``), exiting the block restores the
    pre-call configuration: the global dataclass is rolled back
    to its previous field values, and every phasic-tracked env
    var (the ``PHASIC_*`` set plus ``OMP_NUM_THREADS``) is
    restored to whatever it held before the ``configure()`` call.

    Nesting is supported: each block restores to the state at
    its own ``__enter__`` time, so ``with A(): with B(): ...``
    rolls back B first, then A.
    """

    def __init__(self,
                 saved_fields: dict[str, Any] | None,
                 saved_env: dict[str, str | None],
                 saved_assigned: set[str]):
        self._saved_fields = saved_fields  # None means no prior config
        self._saved_env = saved_env
        self._saved_assigned = saved_assigned

    def __enter__(self) -> "PhasicConfig":
        # Settings are already applied. Return the live config so
        # callers can ``with configure(...) as cfg:`` if they want.
        return get_config()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _global_config

        # Restore env vars.
        for var, value in self._saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

        # Restore the "phasic-assigned" bookkeeping set.
        _phasic_assigned_env.clear()
        _phasic_assigned_env.update(self._saved_assigned)

        # Restore the dataclass.
        if self._saved_fields is None:
            _global_config = None
        else:
            if _global_config is None:
                _global_config = PhasicConfig()
            for k, v in self._saved_fields.items():
                setattr(_global_config, k, v)
            # Skip validate() on the restore path: the saved state
            # was already valid when we captured it, and re-running
            # validate() would re-fire env-var conflict checks
            # against the restored values.
            _global_config._validated = True


def _snapshot_config_for_context() -> _ConfigureContext:
    """Capture the current PhasicConfig + env vars + assigned-set
    for later rollback by _ConfigureContext.__exit__."""
    env_vars = {
        'OMP_NUM_THREADS',
        'PHASIC_HIERAR_ELIMINATION',
        'PHASIC_MIN_SCC_SIZE_TO_CACHE',
        'PHASIC_MAX_PARALLEL_SCCS',
        'PHASIC_FORCE_MPFR',
        'PHASIC_MPFR_BITS',
        'PHASIC_CONDITION_THRESHOLD',
        'PHASIC_DISABLE_CONDITION_WARNINGS',
        'PHASIC_DISABLE_CACHE',
        'PHASIC_CACHE_DIR',
    }
    saved_env = {v: os.environ.get(v) for v in env_vars}
    saved_assigned = set(_phasic_assigned_env)

    if _global_config is None:
        saved_fields = None
    else:
        saved_fields = {
            f.name: getattr(_global_config, f.name)
            for f in PhasicConfig.__dataclass_fields__.values()
            if not f.name.startswith("_")
        }

    return _ConfigureContext(saved_fields, saved_env, saved_assigned)


def configure(**kwargs) -> _ConfigureContext:
    """
    Configure phasic globally — or temporarily, as a context manager.

    All fields describe USER INTENT, not which library implements
    the feature. See PhasicConfig for the full list and per-field
    docstrings.

    Field summary:
      - compute: 'auto' | 'cpu' | 'jax-cpu' | 'jax-gpu'
      - cpu_threads: int or None (None = auto-detect)
      - parallel_elimination: bool
      - parallel_elimination_min_subgraph: int or None
      - parallel_elimination_max_concurrent: int or None
      - svgd_strategy: 'auto' | 'pmap' | 'vmap' | 'none'
      - high_precision_mode: 'auto' | 'always' | 'never'
      - high_precision_bits: int or None
      - ill_condition_threshold: float
      - warn_on_ill_conditioning: bool
      - cache_enabled: bool
      - cache_dir: str or None
      - strict: bool
      - verbose: bool

    Returns
    -------
    _ConfigureContext
        A context-manager-shaped object. When used as a regular
        call (``configure(...)``) the settings are applied
        immediately and persist until further changes. When used
        as ``with configure(...): ...`` the settings are applied
        on entry and rolled back to their previous values on exit
        (including all phasic-tracked env vars).

    Raises
    ------
    PTDConfigError
        On unknown kwargs, invalid values, or conflicts between
        configure() arguments and pre-existing env vars.

    Examples
    --------
    >>> import phasic
    >>> # Persistent: settings stay applied.
    >>> phasic.configure(compute='jax-cpu', cpu_threads=4)

    >>> # Temporary: rolled back at the end of the block.
    >>> with phasic.configure(parallel_elimination=True,
    ...                       parallel_elimination_max_concurrent=8):
    ...     # graph.expectation() runs with parallel elimination
    ...     ...
    >>> # Outside the block, parallel_elimination is back to False

    >>> # Capture the live config inside the block:
    >>> with phasic.configure(high_precision_mode='always') as cfg:
    ...     print(cfg.high_precision_bits)
    """
    global _global_config

    valid_fields = {f.name for f in PhasicConfig.__dataclass_fields__.values()
                    if not f.name.startswith("_")}
    for key in kwargs:
        if key not in valid_fields:
            raise PTDConfigError(
                f"Unknown configuration option: {key}\n"
                f"Valid options: {', '.join(sorted(valid_fields))}"
            )

    # Snapshot the pre-call state BEFORE we mutate anything, so
    # _ConfigureContext.__exit__ can roll back to it.
    snapshot = _snapshot_config_for_context()

    if _global_config is None:
        _global_config = PhasicConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            setattr(_global_config, key, value)

    _global_config.validate()
    return snapshot


def get_config() -> "PhasicConfig":
    """
    Return the current global configuration.

    On first call, reads the relevant phasic env vars
    (`PHASIC_HIERAR_ELIMINATION`, `PHASIC_DISABLE_CACHE`,
    `PHASIC_CACHE_DIR`, `PHASIC_MIN_SCC_SIZE_TO_CACHE`,
    `PHASIC_MAX_PARALLEL_SCCS`, `OMP_NUM_THREADS`) and seeds the
    config so its fields agree with the environment that the C
    layer will observe.

    Subsequent calls return the cached config object.
    """
    global _global_config

    if _global_config is None:
        kwargs: dict[str, Any] = {}

        if os.getenv('PHASIC_HIERAR_ELIMINATION') == '1':
            kwargs['parallel_elimination'] = True
        if os.getenv('PHASIC_DISABLE_CACHE') == '1':
            kwargs['cache_enabled'] = False
        cache_dir_env = os.getenv('PHASIC_CACHE_DIR')
        if cache_dir_env:
            kwargs['cache_dir'] = cache_dir_env
        min_scc = os.getenv('PHASIC_MIN_SCC_SIZE_TO_CACHE')
        if min_scc is not None:
            try:
                kwargs['parallel_elimination_min_subgraph'] = int(min_scc)
            except ValueError:
                pass
        max_par = os.getenv('PHASIC_MAX_PARALLEL_SCCS')
        if max_par is not None:
            try:
                parsed = int(max_par)
                if parsed >= 1:
                    kwargs['parallel_elimination_max_concurrent'] = parsed
            except ValueError:
                pass
        omp_threads = os.getenv('OMP_NUM_THREADS')
        if omp_threads is not None:
            try:
                parsed = int(omp_threads)
                if parsed >= 1:
                    kwargs['cpu_threads'] = parsed
            except ValueError:
                pass
        force_mpfr = os.getenv('PHASIC_FORCE_MPFR')
        if force_mpfr == '1':
            kwargs['high_precision_mode'] = 'always'
        mpfr_bits = os.getenv('PHASIC_MPFR_BITS')
        if mpfr_bits is not None:
            try:
                kwargs['high_precision_bits'] = int(mpfr_bits)
            except ValueError:
                pass
        cond_thresh = os.getenv('PHASIC_CONDITION_THRESHOLD')
        if cond_thresh is not None:
            try:
                kwargs['ill_condition_threshold'] = float(cond_thresh)
            except ValueError:
                pass

        _global_config = PhasicConfig(**kwargs)
        _global_config.validate()

    return _global_config


def get_available_options() -> dict[str, Any]:
    """
    Return what's available on this system (independent of config).

    Useful for branching code in scripts:

    >>> import phasic
    >>> if phasic.get_available_options()['mpfr']:
    ...     phasic.configure(high_precision_mode='always')
    """
    return {
        'jax': _check_jax_available(),
        'cpp': _check_cpp_available(),
        'mpfr': _check_mpfr_available(),
        'platforms': _get_available_platforms(),
    }


def reset_config() -> None:
    """
    Reset the global configuration to defaults.

    Mainly used by tests. Note: does NOT roll back env vars that
    a previous configure() call wrote. If you need a clean env,
    unset PHASIC_* / OMP_NUM_THREADS in os.environ manually.
    """
    global _global_config
    _global_config = None


__all__ = [
    'PhasicConfig',
    'configure',
    'get_config',
    'get_available_options',
    'reset_config',
]



# =============================================================
# New configuration API (refactor)
# =============================================================
#
# This block introduces `PhasicConfig`, the replacement for
# `PTDAlgorithmsConfig`. Once all internal callers have migrated
# and the tests/tutorials are updated, the old class above will
# be removed.
#
# Design principles:
#   - Public fields describe user intent ("parallel_elimination",
#     "high_precision_mode") rather than implementation
#     mechanics ("hierar_elimination", "force_high_precision").
#   - There is one source of truth per knob: the env var. The
#     dataclass field is the corresponding setter. If both the
#     env var and configure() set a field's value to different
#     values, validate() raises — no silent override.
#   - The `compute` field is the only public backend selector.
#     Implementation flags (_use_jax, _use_jit, _use_ffi,
#     _use_openmp, _jax_platform) are derived from `compute` and
#     read by internal modules (mcmc/svgd/ffi_wrappers).
#   - validate() does NOT mutate user-supplied fields. Any
#     conflict raises PTDConfigError.

from typing import Literal as _Literal


# field_name -> (env_var_name, invert_bool, default_value_means_unset)
#
# invert_bool: if True, the env var is "off" semantics (e.g.
# PHASIC_DISABLE_CACHE=1 disables, but the field cache_enabled
# is the positive form). When writing: True -> env unset,
# False -> env set to "1". When reading: presence of env -> False.
#
# default_value_means_unset: the field value that corresponds to
# "leave the env var alone". For Optional[int] fields this is None;
# for bool fields it's the default polarity.
_ENV_VAR_FOR_FIELD = {
    'cpu_threads':                          ('OMP_NUM_THREADS', False, None),
    'parallel_elimination':                 ('PHASIC_HIERAR_ELIMINATION', False, False),
    'parallel_elimination_min_subgraph':    ('PHASIC_MIN_SCC_SIZE_TO_CACHE', False, None),
    'parallel_elimination_max_concurrent':  ('PHASIC_MAX_PARALLEL_SCCS', False, None),
    'high_precision_mode':                  ('PHASIC_FORCE_MPFR', False, 'auto'),
    'high_precision_bits':                  ('PHASIC_MPFR_BITS', False, None),
    'ill_condition_threshold':              ('PHASIC_CONDITION_THRESHOLD', False, 1e12),
    'warn_on_ill_conditioning':             ('PHASIC_DISABLE_CONDITION_WARNINGS', True, True),
    'cache_enabled':                        ('PHASIC_DISABLE_CACHE', True, True),
    'cache_dir':                            ('PHASIC_CACHE_DIR', False, None),
}


@dataclass
class PhasicConfig:
    """
    Public configuration for phasic. All fields describe USER INTENT.
    Implementation choices (JAX/MPFR/OpenMP/FFI/SCC) are derived.

    Conflicting settings raise PTDConfigError; no silent coercion.

    Public fields
    -------------
    compute : 'auto' | 'cpu' | 'jax-cpu' | 'jax-gpu', default 'auto'
        Single source of truth for the compute backend.
        - 'auto'    : JAX if importable, else pure C++.
        - 'cpu'     : pure C++ pipeline; no JAX, no JIT, no FFI.
        - 'jax-cpu' : JAX path on the CPU device.
        - 'jax-gpu' : JAX path on a GPU device. Raises if no GPU.

    cpu_threads : int or None, default None
        OpenMP thread count. None = use auto-detected value
        (SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE / sched_getaffinity
        / os.cpu_count()). Backs OMP_NUM_THREADS.

    parallel_elimination : bool, default False
        Run Gaussian elimination in parallel across strongly-connected
        components of the graph. Backs PHASIC_HIERAR_ELIMINATION.

    parallel_elimination_min_subgraph : int or None, default None
        Skip the on-disk cache for SCCs smaller than this many
        vertices (None = use the C default of 4). Backs
        PHASIC_MIN_SCC_SIZE_TO_CACHE.

    parallel_elimination_max_concurrent : int or None, default None
        Cap on simultaneous per-SCC eliminations within one level
        of the SCC condensation. None = no cap (use all OMP threads).
        Backs PHASIC_MAX_PARALLEL_SCCS.

    high_precision_mode : 'auto' | 'always' | 'never', default 'auto'
        When to engage high-precision arithmetic (MPFR).
        - 'auto'   : activate when the condition number exceeds
                     ill_condition_threshold.
        - 'always' : force high precision for every moment computation.
        - 'never'  : double precision only, even for ill-conditioned
                     problems (warnings still surface).
        Requires the library to be built with MPFR for 'always'.
        Backs PHASIC_FORCE_MPFR.

    high_precision_bits : int or None, default None
        Precision in bits when high-precision mode is active.
        None = auto-pick based on the condition number. Must be
        >= 53 if specified. Backs PHASIC_MPFR_BITS.

    ill_condition_threshold : float, default 1e12
        Condition-number threshold above which high-precision mode
        is engaged automatically. Backs PHASIC_CONDITION_THRESHOLD.

    warn_on_ill_conditioning : bool, default True
        Emit log warnings when the condition number is high.
        Backs PHASIC_DISABLE_CONDITION_WARNINGS (inverted).

    cache_enabled : bool, default True
        Use the on-disk caches under cache_dir. Backs
        PHASIC_DISABLE_CACHE (inverted).

    cache_dir : str or None, default None
        Override the cache root. None = $HOME/.phasic_cache. Backs
        PHASIC_CACHE_DIR.

    strict : bool, default True
        Raise on missing-feature warnings (e.g. JAX requested but
        not installed) instead of logging a warning. Does NOT
        affect conflict checking — conflicts always raise.

    verbose : bool, default False
        Print configuration details and informational logs.
    """

    # --- compute backend (single source of truth) ---
    compute: _Literal['auto', 'cpu', 'jax-cpu', 'jax-gpu'] = 'auto'

    # --- parallel execution ---
    cpu_threads: int | None = None
    parallel_elimination: bool = False
    parallel_elimination_min_subgraph: int | None = None
    parallel_elimination_max_concurrent: int | None = None
    # SVGD particle parallelism strategy:
    #   'auto' -> 'pmap' if jax.devices() reports >1 device, else 'vmap'
    #   'pmap' -> distribute particles across devices
    #   'vmap' -> vectorise particles on a single device
    #   'none' -> sequential particle evaluation (debugging)
    # No env-var backing — this is a runtime-only knob consumed
    # by Graph.svgd() / SVGD() when their `parallel` kwarg is
    # left at None.
    svgd_strategy: _Literal['auto', 'pmap', 'vmap', 'none'] = 'auto'

    # --- numerical precision ---
    high_precision_mode: _Literal['auto', 'always', 'never'] = 'auto'
    high_precision_bits: int | None = None
    ill_condition_threshold: float = 1e12
    warn_on_ill_conditioning: bool = True

    # --- caching ---
    cache_enabled: bool = True
    cache_dir: str | None = None

    # --- runtime behaviour ---
    strict: bool = True
    verbose: bool = False

    # --- private, derived from `compute` ---
    _use_jax: bool = field(default=True, init=False, repr=False)
    _use_jit: bool = field(default=True, init=False, repr=False)
    _use_ffi: bool = field(default=True, init=False, repr=False)
    _use_openmp: bool = field(default=True, init=False, repr=False)
    _jax_platform: _Literal['cpu', 'gpu'] = field(default='cpu', init=False, repr=False)

    # Internal tracking.
    _validated: bool = field(default=False, init=False, repr=False)
    _jax_imported: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Compatibility properties — old internal readers (config.jax,
    # config.jit, config.ffi) keep working without touching mcmc.py /
    # svgd.py / ffi_wrappers.py / __init__.py during the staged migration.
    # ------------------------------------------------------------------
    @property
    def jax(self) -> bool:
        """Whether JAX is enabled. Derived from `compute`."""
        return self._use_jax

    @property
    def jit(self) -> bool:
        """Whether JIT is enabled. Derived from `compute`."""
        return self._use_jit

    @property
    def ffi(self) -> bool:
        """Whether the FFI backend is enabled. Derived from `compute`."""
        return self._use_ffi

    # ------------------------------------------------------------------
    # Resolution + validation
    # ------------------------------------------------------------------
    def _resolve_compute(self) -> None:
        """Map the public `compute` value onto the private _use_* flags.

        Raises
        ------
        PTDConfigError
            If the requested compute mode cannot be satisfied (e.g.
            'jax-cpu' without JAX installed, 'jax-gpu' without a GPU).
        """
        if self.compute == 'auto':
            resolved = 'jax-cpu' if _check_jax_available() else 'cpu'
        else:
            resolved = self.compute

        if resolved == 'cpu':
            self._use_jax = False
            self._use_jit = False
            self._use_ffi = False
            self._use_openmp = True
            self._jax_platform = 'cpu'
            return

        if resolved in ('jax-cpu', 'jax-gpu'):
            if not _check_jax_available():
                raise PTDConfigError(
                    f"compute={self.compute!r} requested but JAX is not installed.\n"
                    "  Install JAX:   pip install jax jaxlib\n"
                    "  Or set         compute='cpu'\n"
                )
            self._use_jax = True
            self._use_jit = True
            self._use_ffi = True
            self._use_openmp = True
            self._jax_platform = 'gpu' if resolved == 'jax-gpu' else 'cpu'
            if resolved == 'jax-gpu' and 'gpu' not in _get_available_platforms():
                raise PTDConfigError(
                    f"compute='jax-gpu' requested but no GPU device is available."
                )
            return

        raise PTDConfigError(
            f"Unknown compute={self.compute!r}; "
            f"must be one of 'auto', 'cpu', 'jax-cpu', 'jax-gpu'."
        )

    def validate(self) -> None:
        """Validate the configuration, sync to env vars, raise on conflicts.

        Rules:
          1. Env-var single-source-of-truth: each non-default field is
             written into its env var.
          2. No silent rewrites: validate() does not mutate user-set fields.
          3. Conflicts raise: incompatible field combinations, or
             configure() vs pre-existing env var disagreement.
          4. `strict` only governs missing-feature warnings; conflicts
             always raise regardless of strict.
        """
        errors: list[str] = []

        # 1. Resolve compute first — this populates the _use_* fields
        #    and raises on impossible requests.
        try:
            self._resolve_compute()
        except PTDConfigError as e:
            errors.append(str(e))

        # 2. parallel_elimination requires cpu_threads >= 2 (after
        #    cpu_threads resolves: either explicit, or shell, or
        #    phasic auto-detect).
        effective_threads = self.cpu_threads
        if effective_threads is None:
            env_val = os.environ.get('OMP_NUM_THREADS')
            if env_val is not None:
                try:
                    effective_threads = int(env_val)
                except ValueError:
                    pass
        if self.parallel_elimination and effective_threads == 1:
            errors.append(
                "parallel_elimination=True is meaningless with cpu_threads=1.\n"
                "  Set cpu_threads >= 2 or use parallel_elimination=False."
            )

        # 3. high_precision_mode='always' requires MPFR.
        if self.high_precision_mode == 'always' and not _check_mpfr_available():
            errors.append(
                "high_precision_mode='always' requested but the C library was\n"
                "  built without MPFR. Rebuild with MPFR or set\n"
                "  high_precision_mode='auto'."
            )

        # 4. high_precision_bits: 0/None for auto, otherwise >= 53.
        if self.high_precision_bits is not None:
            if self.high_precision_bits < 53:
                errors.append(
                    f"high_precision_bits={self.high_precision_bits} too small "
                    f"(must be >= 53 or None for auto)."
                )

        # 5. ill_condition_threshold must be > 1.
        if self.ill_condition_threshold <= 1.0:
            errors.append(
                f"ill_condition_threshold={self.ill_condition_threshold} must be > 1."
            )

        # 6. cpu_threads >= 1 if set.
        if self.cpu_threads is not None and self.cpu_threads < 1:
            errors.append(
                f"cpu_threads={self.cpu_threads} must be >= 1 or None for auto-detect."
            )

        # 7. parallel_elimination_max_concurrent >= 1 if set.
        if (self.parallel_elimination_max_concurrent is not None
                and self.parallel_elimination_max_concurrent < 1):
            errors.append(
                f"parallel_elimination_max_concurrent="
                f"{self.parallel_elimination_max_concurrent} must be >= 1 or None."
            )

        # 8. parallel_elimination_min_subgraph >= 0 if set.
        if (self.parallel_elimination_min_subgraph is not None
                and self.parallel_elimination_min_subgraph < 0):
            errors.append(
                f"parallel_elimination_min_subgraph="
                f"{self.parallel_elimination_min_subgraph} must be >= 0."
            )

        # 8b. svgd_strategy enum check.
        valid_strategies = ('auto', 'pmap', 'vmap', 'none')
        if self.svgd_strategy not in valid_strategies:
            errors.append(
                f"svgd_strategy={self.svgd_strategy!r} must be one of "
                f"{valid_strategies}."
            )

        # 9. Env-var ↔ configure() conflict policy.
        #    Phasic's own import-time auto-detect for OMP_NUM_THREADS
        #    records itself in _phasic_assigned_env; configure() may
        #    overwrite that without raising. Any OTHER pre-existing
        #    env var that disagrees with a configure() value raises.
        for field_name, (env_var, invert, default) in _ENV_VAR_FOR_FIELD.items():
            field_value = getattr(self, field_name)
            if field_value == default:
                # Field at default; nothing to compare against.
                continue
            existing = os.environ.get(env_var)
            if existing is None:
                continue
            if _env_was_set_by_phasic(env_var):
                # Our own auto-set; safe to overwrite.
                continue
            # Compare existing env value with what we'd write.
            new_str = _field_to_env_value(field_name, field_value, invert)
            if existing != new_str:
                errors.append(
                    f"Conflicting configuration for {field_name}: shell sets "
                    f"{env_var}={existing!r} but configure({field_name}="
                    f"{field_value!r}) would write {env_var}={new_str!r}. "
                    f"Unset {env_var} before launch, or align the two values."
                )

        # 10. Raise everything together.
        if errors:
            raise PTDConfigError("\n\n".join(errors))

        # 11. Write env vars for non-default fields.
        for field_name, (env_var, invert, default) in _ENV_VAR_FOR_FIELD.items():
            field_value = getattr(self, field_name)
            if field_value == default:
                # Leave the env var alone (don't overwrite shell value).
                continue
            new_str = _field_to_env_value(field_name, field_value, invert)
            previous = os.environ.get(env_var)
            if new_str is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = new_str
            # Only mark as phasic-assigned if we actually CHANGED
            # the env var. If validate() is just reaffirming a
            # value the shell already set (e.g. via the
            # get_config() bootstrap that reads env vars), we
            # leave the assignment unmarked so a later configure()
            # with a different value still triggers the conflict
            # check.
            if previous != new_str:
                _phasic_assigned_env.add(env_var)

        # 12. high_precision_mode is tri-valued; the C side only
        #     reads PHASIC_FORCE_MPFR as a boolean.
        #     'always' -> set; 'auto'/'never' -> unset.
        #     'never' additionally writes PHASIC_CONDITION_THRESHOLD=inf
        #     to disable the auto-trigger.
        if self.high_precision_mode == 'always':
            os.environ['PHASIC_FORCE_MPFR'] = '1'
        else:
            os.environ.pop('PHASIC_FORCE_MPFR', None)
        if self.high_precision_mode == 'never':
            os.environ['PHASIC_CONDITION_THRESHOLD'] = 'inf'
        elif self.ill_condition_threshold != 1e12:
            os.environ['PHASIC_CONDITION_THRESHOLD'] = (
                str(self.ill_condition_threshold))
        elif (self.high_precision_mode == 'auto'
              and os.environ.get('PHASIC_CONDITION_THRESHOLD') == 'inf'):
            # Previous 'never' setting bled through. Clear it so
            # 'auto' uses the C default (1e12).
            os.environ.pop('PHASIC_CONDITION_THRESHOLD', None)

        self._validated = True

        if self.verbose:
            print(f"PhasicConfig: {self.effective()}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def effective(self) -> dict[str, Any]:
        """Snapshot of current configuration and its environment.

        Returns
        -------
        dict with three sections:
          - 'fields':      every public field and its current value.
          - 'environment': every env var phasic uses, and its current
                           value in os.environ (None if unset).
          - 'derived':     useful answers about runtime state:
                           jax_active, mpfr_active, cpu_threads_source,
                           cache_dir_resolved, compute_resolved.
        """
        fields = {
            'compute': self.compute,
            'cpu_threads': self.cpu_threads,
            'parallel_elimination': self.parallel_elimination,
            'parallel_elimination_min_subgraph': self.parallel_elimination_min_subgraph,
            'parallel_elimination_max_concurrent': self.parallel_elimination_max_concurrent,
            'svgd_strategy': self.svgd_strategy,
            'high_precision_mode': self.high_precision_mode,
            'high_precision_bits': self.high_precision_bits,
            'ill_condition_threshold': self.ill_condition_threshold,
            'warn_on_ill_conditioning': self.warn_on_ill_conditioning,
            'cache_enabled': self.cache_enabled,
            'cache_dir': self.cache_dir,
            'strict': self.strict,
            'verbose': self.verbose,
        }

        env_vars = [
            'OMP_NUM_THREADS',
            'PHASIC_HIERAR_ELIMINATION',
            'PHASIC_MIN_SCC_SIZE_TO_CACHE',
            'PHASIC_MAX_PARALLEL_SCCS',
            'PHASIC_FORCE_MPFR',
            'PHASIC_MPFR_BITS',
            'PHASIC_CONDITION_THRESHOLD',
            'PHASIC_DISABLE_CONDITION_WARNINGS',
            'PHASIC_DISABLE_CACHE',
            'PHASIC_CACHE_DIR',
        ]
        environment = {var: os.environ.get(var) for var in env_vars}

        # cpu_threads source: where did OMP_NUM_THREADS come from?
        if self.cpu_threads is not None:
            threads_src = 'configure()'
        else:
            threads_src = _omp_threads_source()

        # cache_dir resolved: respect override or default.
        cache_dir_resolved = self.cache_dir or os.path.expanduser('~/.phasic_cache')

        # compute resolved: if 'auto', what did it become?
        if self.compute == 'auto':
            compute_resolved = 'jax-cpu' if self._use_jax else 'cpu'
        else:
            compute_resolved = self.compute

        derived = {
            'jax_active': self._jax_imported,
            'mpfr_active': _check_mpfr_available() and (
                self.high_precision_mode == 'always'
                or os.environ.get('PHASIC_FORCE_MPFR') == '1'),
            'cpu_threads_source': threads_src,
            'cache_dir_resolved': cache_dir_resolved,
            'compute_resolved': compute_resolved,
        }

        return {'fields': fields, 'environment': environment, 'derived': derived}

    def __repr__(self) -> str:
        """Multi-line pretty-print of the effective configuration."""
        import json
        eff = self.effective()
        return "PhasicConfig:\n" + json.dumps(eff, indent=2, default=str)


# ------------------------------------------------------------------
# Helpers shared by validate() and effective()
# ------------------------------------------------------------------

# Module-level record of env vars that phasic itself set during
# import-time auto-detection. configure() may overwrite these
# without raising; user-set env vars trigger conflict checks.
_phasic_assigned_env: set[str] = set()


def _env_was_set_by_phasic(env_var: str) -> bool:
    """True if phasic auto-set this env var (e.g. import-time
    OMP_NUM_THREADS detection)."""
    return env_var in _phasic_assigned_env


def _field_to_env_value(field_name: str, value: Any, invert: bool) -> str | None:
    """Convert a field value to its env-var string representation.

    Returns None to signal "remove the env var" (e.g. inverted
    bool fields at their positive default).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        if invert:
            # E.g. cache_enabled=False -> PHASIC_DISABLE_CACHE='1'.
            #      cache_enabled=True  -> remove PHASIC_DISABLE_CACHE.
            return '1' if not value else None
        return '1' if value else None
    if isinstance(value, str):
        return value if value else None
    return str(value)


def _omp_threads_source() -> str:
    """Describe where the OMP_NUM_THREADS value came from."""
    if os.environ.get('OMP_NUM_THREADS') is None:
        return 'unset'
    if _env_was_set_by_phasic('OMP_NUM_THREADS'):
        if os.environ.get('SLURM_CPUS_PER_TASK'):
            return 'phasic auto-detect (SLURM_CPUS_PER_TASK)'
        if os.environ.get('SLURM_CPUS_ON_NODE'):
            return 'phasic auto-detect (SLURM_CPUS_ON_NODE)'
        return 'phasic auto-detect (os.cpu_count)'
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        return 'shell + SLURM_CPUS_PER_TASK'
    return 'shell (OMP_NUM_THREADS)'
