from __future__ import annotations

from ast import arg
from functools import partial
from collections import defaultdict, OrderedDict
from itertools import product, zip_longest
from unittest import result
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Any, TypeVar, Self, Tuple, List
from numpy.typing import NDArray
from collections.abc import Sequence, MutableSequence, Callable
import os
import hashlib
import subprocess
import tempfile
import ctypes
import pathlib

from functools import wraps
import numpy as np
from collections import OrderedDict

# Import configuration system FIRST (before any optional imports)
from .config import (
    configure,
    get_config,
    get_available_options,
    PhasicConfig,
    reset_config
)
from .exceptions import (
    PTDAlgorithmsError,
    PTDConfigError,
    PTDBackendError,
    PTDFeatureError,
    PTDJAXError
)
from .logging_config import (
    set_log_level,
    get_logger,
)
from .state_indexing import (
    StateSpace,  # Backward compatibility (deprecated)
    Property,
    PropertySet,
    StateIndexer,
    PropertyDict,
    StateVector
)
from .hex_grid import HexGrid, HexCell
from phasic.graph_cache import GraphCache, get_graph_cache_stats, print_graph_cache_info


def _detect_omp_num_threads() -> int:
    """Return the OpenMP thread count to use when OMP_NUM_THREADS
    is unset.

    Priority:
      1. SLURM_CPUS_PER_TASK (the per-task allocation under SLURM)
      2. SLURM_CPUS_ON_NODE (full node allocation under SLURM)
      3. os.sched_getaffinity(0) (respects Linux cgroup limits)
      4. Apple Silicon performance cores only (avoids efficiency-core
         oversubscription, which causes severe OMP stalls under FFI
         batched fanout — see commit notes)
      5. os.cpu_count() (last resort: full machine logical CPUs)
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return max(int(val), 1)
            except ValueError:
                pass
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        pass
    # Apple Silicon has both performance and efficiency cores; using all
    # of them via OpenMP `#pragma omp parallel for` produces a
    # catastrophic slowdown when batch_size is comparable to total core
    # count, because OMP's barrier waits for the slowest thread and the
    # efficiency cores are ~3x slower per element. Measured on M1 Pro
    # (8 perf + 2 efficiency cores): batched FFI sojourn-time call with
    # P=10 takes ~26 s at OMP=10 vs ~1 s at OMP=4. Prefer perf cores
    # only when we can identify them.
    try:
        if os.uname().sysname == 'Darwin':
            perf_cores = subprocess.check_output(
                ['sysctl', '-n', 'hw.perflevel0.physicalcpu'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            n = int(perf_cores)
            if n >= 1:
                return n
    except (AttributeError, subprocess.SubprocessError, ValueError,
            FileNotFoundError):
        pass
    return os.cpu_count() or 1


# Auto-detect OMP_NUM_THREADS BEFORE the pybind module loads —
# OpenMP reads the env var on library load, so setting it later
# would not take effect for the SCC composer's parallel loop.
# Users can pre-set OMP_NUM_THREADS in their shell to override.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(_detect_omp_num_threads())
    # Record that phasic set this value (not the user/shell).
    # PhasicConfig's conflict checker (in config.py) treats env
    # vars in _phasic_assigned_env as overwritable by configure();
    # user-set env vars trigger conflict-raises.
    from .config import _phasic_assigned_env as _phasic_assigned_env_set
    _phasic_assigned_env_set.add("OMP_NUM_THREADS")

# Force JAX 64-bit precision for everything that flows through the
# FFI. The C handlers require F64 buffers; without this default a
# downstream `import jax` (in a notebook, in a sibling library,
# anywhere) leaves jax_enable_x64 disabled and the FFI crashes
# with "Wrong buffer dtype: expected F64 but got F32". Setting
# the env var here — before any jax import can occur — makes the
# guarantee transparent. `_ensure_jax_active()` below also calls
# `jax.config.update('jax_enable_x64', True)` defensively for the
# case where jax was already imported before phasic loaded.
os.environ.setdefault("JAX_ENABLE_X64", "1")

# from .vscode_theme import set_phasic_theme
# from .vscode_theme import phasic_theme as theme
# from .vscode_theme import set_theme # backwards compatibility
# from . import plot

# Get configuration (creates default if none exists)
_config = get_config()

# JAX is no longer imported at module load time. The previous
# import-time block (which imported JAX, wrote XLA_FLAGS, applied
# CompilationConfig.balanced(), and installed a stdout filter) is
# now deferred to _ensure_jax_active() below, which is called
# lazily by mcmc.py / svgd.py and by configure() when compute is
# 'jax-cpu' / 'jax-gpu'. This means `import phasic` is cheap and
# side-effect-light: no JAX import, no env vars beyond
# OMP_NUM_THREADS auto-detect.
jax = None
jnp = None
HAS_JAX = False


# The _DeviceListFilter class lives at module scope so
# _ensure_jax_active() can install it. Wrapping stdout/stderr
# prevents JAX from logging "CpuDevice(id=0), CpuDevice(id=1), ..."
# at first device access.
class _DeviceListFilter:
    def __init__(self, original: Any) -> None:
        self.original = original
        self.buffer = ''

    def write(self, text: str) -> None:
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            line += '\n'
            if not ('CpuDevice' in line or 'GpuDevice' in line):
                self.original.write(line)

    def flush(self) -> None:
        if self.buffer and not ('CpuDevice' in self.buffer or 'GpuDevice' in self.buffer):
            self.original.write(self.buffer)
            self.buffer = ''
        self.original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


# ------------------------------------------------------------------
# Deferred JAX initialisation (refactor step 2)
#
# `_ensure_jax_active()` is the on-demand entry point that runs the
# same JAX setup the import-time block does above. After step 5 of
# the config refactor, the import-time block is removed and this
# helper is the sole path that touches JAX.
#
# For now (step 2) it is idempotent: if JAX was already imported by
# the top-of-module block, calling _ensure_jax_active() is a no-op.
# Callers in svgd.py / mcmc.py can already use it.
# ------------------------------------------------------------------
def _ensure_jax_active() -> None:
    """Lazily initialise JAX. Idempotent — safe to call repeatedly.

    Performs (in order):
      1. Apply CompilationConfig.balanced() defaults.
      2. Write XLA_FLAGS with multi-CPU device count.
      3. Set JAX_PLATFORMS=cpu default.
      4. Install the stdout/stderr device-list filter.
      5. import jax + enable x64.

    After this returns, ``jax`` and ``jnp`` module attributes are
    populated. Mark the active state on the global config so
    `effective()` reports it.
    """
    global jax, jnp, HAS_JAX
    if HAS_JAX:
        return

    import sys
    if 'jax' in sys.modules:
        # JAX was imported by something else (e.g. directly in the
        # user's notebook, or by another library). Pick up the
        # references — and CRITICALLY also enable x64. The env var
        # JAX_ENABLE_X64=1 (set at phasic import time) covers the
        # case where jax is imported *after* phasic, but if jax was
        # imported *before* phasic loaded, only `jax.config.update`
        # can flip the flag now. Skipping this is the root cause of
        # the FFI "expected F64 but got F32" crashes.
        import jax as _jax_mod
        was_off = not bool(_jax_mod.config.jax_enable_x64)
        _jax_mod.config.update('jax_enable_x64', True)
        import jax.numpy as _jnp_mod
        jax = _jax_mod
        jnp = _jnp_mod
        HAS_JAX = True
        if was_off:
            # Any JAX arrays created before this flip are still
            # float32. Operations on them will mix dtypes and may
            # crash the FFI later. Tell the user once.
            import warnings
            warnings.warn(
                "phasic enabled jax_enable_x64 after JAX was already "
                "imported. Any JAX arrays created before importing "
                "phasic are still float32 and may trip FFI dtype "
                "checks. Restart the kernel (or recreate arrays) if "
                "you see 'expected F64 but got F32' errors.",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            cfg = get_config()
            cfg._jax_imported = True
        except Exception:
            pass
        return

    from .jax_config import get_default_config
    get_default_config().apply(force=False)

    cpu_count = int(os.environ.get('PTDALG_CPUS',
                                   _detect_omp_num_threads()))
    xla_flags = os.environ.get('XLA_FLAGS', '')
    device_flag = f"--xla_force_host_platform_device_count={cpu_count}"
    if '--xla_force_host_platform_device_count' not in xla_flags:
        xla_flags = f"{xla_flags} {device_flag}".strip()
        os.environ['XLA_FLAGS'] = xla_flags

    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

    # Install device-list filter before JAX prints its device list.
    if not isinstance(sys.stdout, _DeviceListFilter):
        sys.stdout = _DeviceListFilter(sys.stdout)
    if not isinstance(sys.stderr, _DeviceListFilter):
        sys.stderr = _DeviceListFilter(sys.stderr)

    import jax as _jax_mod
    _jax_mod.config.update('jax_enable_x64', True)
    import jax.numpy as _jnp_mod
    jax = _jax_mod
    jnp = _jnp_mod
    HAS_JAX = True

    try:
        cfg = get_config()
        cfg._jax_imported = True
    except Exception:
        pass


# Cache for compiled libraries
_lib_cache = {}

from .phasic_pybind import *
from .phasic_pybind import Graph as _Graph
from . import _graph_plotting
from . import _graph_cache_transfer
from . import _graph_cache_mgmt
from . import _graph_reward_validation
from . import _graph_serialize
from .phasic_pybind import Vertex, Edge

# Configure package-wide logging
from .logging_config import setup_logging, get_logger
setup_logging()

# SVGD, MCMC, BFFG, and prior/optimizer/preconditioner classes
# are JAX-dependent and resolved lazily via the module-level
# __getattr__ at the end of this file. They are NOT bound at
# import time, which keeps `import phasic` from triggering the
# JAX import cascade.

# Method of moments (requires JAX via svgd dependency, but MoMResult is always available)
from .method_of_moments import MoMResult

# Probability matching for joint probability graphs
from .probability_matching import ProbMatchResult

# Optax integration (optional dependency)
try:
    from .optax_wrapper import (
        OptaxOptimizer,
        optax_adam,
        optax_adamw,
        optax_sgd,
        optax_rmsprop,
        optax_adagrad,
        optax_chain,
        optax_lion,
    )
    HAS_OPTAX = True
except ImportError:
    # Optax not installed - exports will raise ImportError when accessed
    OptaxOptimizer = None
    optax_adam = None
    optax_adamw = None
    optax_sgd = None
    optax_rmsprop = None
    optax_adagrad = None
    optax_chain = None
    optax_lion = None
    HAS_OPTAX = False

# Progress bar utilities
#from .utils import pqdm, prange # now in vscodenb

# Distributed computing utilities
from .distributed_utils import (
    DistributedConfig,
    # initialize_distributed,
    detect_slurm_environment,
    get_coordinator_address,
    configure_jax_devices,
    initialize_jax_distributed
)

# Cluster configuration management
from .cluster_configs import (
    ClusterConfig,
    load_config,
    get_default_config,
    validate_config,
    suggest_config
)

# Automatic parallelization
from .auto_parallel import (
    EnvironmentInfo,
    ParallelConfig,
    detect_environment,
    configure_jax_for_environment,
)

# Cache management (JAX compilation cache)
from .cache_manager import CacheManager, print_jax_cache_info, configure_layered_cache
from .model_export import clear_caches, clear_jax_cache, clear_model_cache, cache_info, print_model_cache_info, get_all_cache_stats, print_all_cache_info
from .trace_cache import get_trace_cache_stats, print_trace_cache_info

# Model selection (AIC, BIC, likelihood-ratio tests) for fitted SVGD
# instances. Pure-Python / numpy / scipy module — JAX-free at import time.
# Submodule access keeps the top-level namespace tidy:
#     phasic.model_selection.aic(svgd)
from . import model_selection

# On-disk cache management (~/.phasic_cache/) — symbolic compute
# graph cache (Stage A2) and Python trace cache.
from . import cache
# Graph profiler — recommends parallel_elimination / dyn_ordering / eval path.
from .profile import profile_graph, GraphProfile
from .jax_config import CompilationConfig, get_default_config, set_default_config
# from .cloud_cache import (
#     S3Backend,
#     GCSBackend,
#     AzureBlobBackend,
#     download_from_url,
#     download_from_github_release,
#     install_model_library
# )
from .trace_elimination import EliminationTrace

# Compute-graph sharing (C-elimination .bin artifacts, keyed by
# ptd_graph_content_hash). Per-graph operations live as methods on
# the Graph class (``g.pull_cache()`` / ``g.push_cache(...)``); only
# the registry-wide browser is exposed at the top level.
from .compute_repository import list_computes

# JAX FFI wrappers (optional, requires JAX)
if HAS_JAX:
    from .ffi_wrappers import (
        compute_pmf_ffi,
        compute_moments_ffi,
        compute_pmf_and_moments_ffi
    )
    from .profiling import (
        analyze_svgd_profile,
        profile_svgd
    )
else:
    compute_pmf_ffi = None
    compute_moments_ffi = None
    compute_pmf_and_moments_ffi = None
    analyze_svgd_profile = None
    profile_svgd = None

__version__ = '0.20.0'

GraphType = TypeVar('Graph')


from collections import namedtuple
MatrixRepresentation = namedtuple("MatrixRepresentation", ['ipv', 'sim', 'states', 'indices'])


# ============================================================================
# Pure Helper Functions (Computation Phase - JAX Compatible)
# ============================================================================

def _compute_pmf_from_ctypes(theta: ArrayLike, times: ArrayLike, compute_func: Any, graph_data: dict | None, granularity: int, discrete: bool) -> np.ndarray:
    """
    Pure function wrapper around ctypes PMF computation.

    No side effects - same inputs always produce same outputs.
    Compatible with JAX transformations when wrapped appropriately.
    """
    theta_np = np.asarray(theta, dtype=np.float64)
    times_np = np.asarray(times, dtype=np.float64 if not discrete else np.int32)
    output_np = np.zeros_like(times_np, dtype=np.float64)

    # Check if this is a parameterized C++ model or a from_arrays model
    if graph_data and 'states_flat' in graph_data:
        # from_arrays case: unpack graph_data (works for both discrete and continuous)
        states_flat = graph_data['states_flat']
        edges_flat = graph_data['edges_flat']
        start_edges_flat = graph_data['start_edges_flat']
        n_vertices = graph_data['n_vertices']
        state_length = graph_data['state_length']
        n_edges = graph_data['n_edges']
        n_start_edges = graph_data['n_start_edges']

        if discrete:
            # Discrete mode: no granularity parameter
            compute_func(
                states_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                n_vertices,
                state_length,
                edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(edges_flat) > 0 else None,
                n_edges,
                start_edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(start_edges_flat) > 0 else None,
                n_start_edges,
                times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                len(times_np),
                output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
        else:
            # Continuous mode: includes granularity parameter
            compute_func(
                states_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                n_vertices,
                state_length,
                edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(edges_flat) > 0 else None,
                n_edges,
                start_edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(start_edges_flat) > 0 else None,
                n_start_edges,
                times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(times_np),
                output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                granularity
            )
    else:
        # C++ build_model case (works for both discrete and continuous)
        if discrete:
            # Discrete mode: no granularity parameter
            compute_func(
                theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(theta_np),
                times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                len(times_np),
                output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
        else:
            # Continuous mode: includes granularity parameter
            compute_func(
                theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(theta_np),
                times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(times_np),
                output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                granularity
            )

    return output_np




def _create_jax_parameterized_wrapper(compute_func: Any, graph_builder: Callable, discrete: bool) -> Callable:
    """
    Create a pure JAX-compatible wrapper for parameterized models.

    Handles models where graph structure depends on parameters.

    NOTE: only used by Graph.pmf_from_graph_parameterized, which is DISABLED (see
    CLAUDE.md "Disabled paths / follow-ups"); it is unreachable today. It carries
    bug 5: `jax`/`jnp` are the module-level lazily-imported globals (None until
    _ensure_jax_active() runs, which this path never calls), and it hardcodes
    jnp.float32 where the FFI requires F64. Fix both on revival.
    """
    from jax import pure_callback

    def model_fn(theta, times, granularity=100):
        """Parameterized model with JAX compatibility"""
        def compute_impl(inputs):
            theta_arr, times_arr = inputs

            # Build graph with parameters
            theta_np = np.asarray(theta_arr)
            if theta_np.ndim == 0:
                theta_np = theta_np.reshape(1)

            # Build and serialize the graph
            graph = graph_builder(*theta_np)
            serialized = graph.serialize()

            # Prepare graph data
            graph_data = _serialize_graph_data(serialized)

            return _compute_pmf_from_ctypes(
                theta_np,
                times_arr,
                compute_func,
                graph_data,
                granularity,
                discrete
            )

        result_shape_dtypes = jax.ShapeDtypeStruct(times.shape, jnp.float32)
        return pure_callback(compute_impl, result_shape_dtypes, (theta, times))

    return model_fn


# ============================================================================
# Impure Helper Functions (Setup Phase - Run Once During Model Loading)
# ============================================================================

def _get_package_dir() -> pathlib.Path:
    """Get package root directory (caching is acceptable).

    Honours $PHASIC_SOURCE_DIR if set so wheel-installed users can still use
    the JIT-compilation paths by pointing at a source checkout. Otherwise
    derives the path from __file__, which only resolves to a real source
    tree under an editable install.
    """
    env_dir = os.environ.get('PHASIC_SOURCE_DIR')
    if env_dir:
        return pathlib.Path(env_dir)
    return pathlib.Path(__file__).parent.parent.parent


_ARTIFACT_DIR: str | None = None


def _secure_artifact_dir() -> str:
    """Return a per-process, per-user 0700 directory for generated C++
    sources and compiled shared libraries.

    Uses ``tempfile.mkdtemp`` — owned by the current user, mode 0700, with an
    unguessable name — instead of predictable, world-writable ``/tmp`` paths
    like ``/tmp/trace_log_lik_<hash>.so``. Those predictable names let a peer
    on a shared login/compute node pre-plant a malicious ``.so`` (loaded via
    ctypes with no owner check, since the hash is derivable from a public
    model) or clobber a generated source through a symlink. An unguessable
    dir owned only by us removes both vectors, while within-process caching
    (the ``skip-if-exists`` reuse) still works because the path is stable for
    the process lifetime. The directory is inherited across ``multiprocess``
    forks, so parallel trace workers share the parent's secure location.
    """
    global _ARTIFACT_DIR
    if _ARTIFACT_DIR is None:
        import tempfile
        _ARTIFACT_DIR = tempfile.mkdtemp(prefix='phasic-compiled-')
    return _ARTIFACT_DIR


def _continuous_to_discrete_moments(m: np.ndarray) -> np.ndarray:
    """Convert continuous waiting-time raw moments ``[E[T], E[T^2], ...]`` to
    discrete raw moments ``[E[N], E[N^2], ...]`` for a DPH.

    There is no native discrete-moment routine in the C layer — the ``*_discrete``
    helpers (``expectation_discrete``, ``variance_discrete``) all compute the
    continuous moments via ``expected_waiting_time`` and then apply an algebraic
    correction. This is the same correction, generalised to arbitrary order.

    Graph-independent because ``U = (I-P)^-1`` commutes with ``P``:
      continuous power moment   ``u_j        = E[T^j]/j! = a U^j 1``
      discrete factorial moment ``F_r = E[(N)_r] = r! * sum_i C(r-1,i)(-1)^i u_{r-i}``
      discrete raw moment       ``E[N^k]     = sum_r StirlingS2(k,r) F_r``
    Order 2 reduces to ``E[N^2] = m[1]-m[0]``. Mirrors
    ``GraphBuilder::continuous_to_discrete_moments`` used on the parameterized path.
    """
    from math import comb, factorial

    m = np.asarray(m, dtype=np.float64)
    k = len(m)
    if k == 0:
        return m
    u = [0.0] * (k + 1)
    for j in range(1, k + 1):
        u[j] = m[j - 1] / factorial(j)
    F = [0.0] * (k + 1)
    for r in range(1, k + 1):
        F[r] = factorial(r) * sum(
            comb(r - 1, i) * (-1) ** i * u[r - i] for i in range(r)
        )

    def _stirling2(n: int, kk: int) -> float:
        if kk == 0:
            return 1.0 if n == 0 else 0.0
        if kk > n:
            return 0.0
        if kk == n or kk == 1:
            return 1.0
        return kk * _stirling2(n - 1, kk) + _stirling2(n - 1, kk - 1)

    out = np.zeros(k)
    for kk in range(1, k + 1):
        out[kk - 1] = sum(_stirling2(kk, r) * F[r] for r in range(1, kk + 1))
    return out


def _serialize_graph_data(serialized: dict) -> dict:
    """Extract and prepare graph arrays for computation."""
    states_flat = serialized['states'].flatten()
    edges_flat = serialized['edges'].flatten() if serialized['edges'].size > 0 else np.array([], dtype=np.float64)
    start_edges_flat = serialized['start_edges'].flatten() if serialized['start_edges'].size > 0 else np.array([], dtype=np.float64)

    return {
        'states_flat': states_flat,
        'edges_flat': edges_flat,
        'start_edges_flat': start_edges_flat,
        'n_vertices': serialized['n_vertices'],
        'state_length': serialized['state_length'],
        'n_edges': len(serialized['edges']),
        'n_start_edges': len(serialized['start_edges'])
    }


def _propagate_weight_formula(src: 'Graph', dst: 'Graph') -> None:
    """Carry a ``weight_mode='formula'`` (formula string + compiled tape) from a
    source graph onto a derived graph, and install the tape on the derived live
    graph.

    Used by the joint-graph constructors (``joint_stop_prob_graph`` /
    ``joint_sojourn_graph``) so a formula set on a joint-prob graph reaches the
    daisy-chain / sojourn FFI path — the only way to use non-inner-product
    weights there (Python callbacks are rejected by validator rule R21).

    Other weight modes are intentionally left untouched, so the existing
    behavior of these derived graphs is byte-for-byte preserved.
    """
    if getattr(src, '_weight_mode', 'linear') != 'formula':
        return
    tape = getattr(src, '_weight_formula_tape', None)
    dst._weight_mode = 'formula'
    dst._weight_formula = getattr(src, '_weight_formula', None)
    dst._weight_formula_tape = tape
    if tape is not None:
        dst._set_weight_tape(list(tape['ops']), list(tape['consts']),
                             int(tape['stack_depth']), int(tape['n_theta']),
                             int(tape['n_coeff']))


def _propagate_weight_state(src: 'Graph', dst: 'Graph') -> None:
    """Copy the full Python-side weight configuration from ``src`` to a
    derived graph ``dst`` and re-install the compiled tape on ``dst``.

    Unlike :func:`_propagate_weight_formula` (formula-only, for the
    joint-graph constructors), this copies *every* mode — linear, log,
    callback, and formula — so graphs derived via ``clone()``/``copy()``,
    ``from_serialized()``, ``_rebuild_with_wider_layout()`` (add_epoch /
    discretize) and ``laplace_transform()`` do not silently revert to
    ``'linear'`` or end up in ``'formula'`` mode with no tape. The C-level
    clone does not copy the weight tape, so a formula graph must have the
    tape re-installed on the derived live graph here.
    """
    dst._weight_mode = getattr(src, '_weight_mode', 'linear')
    dst._weight_callback = getattr(src, '_weight_callback', None)
    dst._weight_formula = getattr(src, '_weight_formula', None)
    tape = getattr(src, '_weight_formula_tape', None)
    dst._weight_formula_tape = tape
    if tape is not None:
        dst._set_weight_tape(list(tape['ops']), list(tape['consts']),
                             int(tape['stack_depth']), int(tape['n_theta']),
                             int(tape['n_coeff']))


def _fixed_mask_from_fixed(fixed, theta_dim):
    """Build a ``(theta_dim,)`` 0/1 mask marking fixed parameters, or ``None``.

    Mirrors the joint_index path: accepts ``fixed`` as a list of
    ``(index, value)`` tuples or an already-built mask array. Returns ``None``
    when ``fixed`` is unset or ``theta_dim`` is unknown — in which case the model
    builders skip nothing (unchanged behavior).
    """
    if fixed is None or theta_dim is None:
        return None
    import jax.numpy as jnp
    if isinstance(fixed, list) and len(fixed) > 0 and isinstance(fixed[0], tuple):
        m = jnp.zeros(theta_dim)
        for idx, _val in fixed:
            m = m.at[idx].set(1)
        return m
    return jnp.array(fixed)


def _fixed_indices_set_from_mask(fixed_mask):
    """Python set of fixed-parameter indices from a 0/1 mask (empty if ``None``).

    Used by the non-daisy model builders' finite-difference VJPs to SKIP fixed
    dimensions (their gradient is 0 and was discarded by SVGD anyway), mirroring
    the daisy ``_autodiff_bwd`` and ``pmf_from_graph_joint_index`` paths. Skipping
    is value-preserving for the learnable dimensions.
    """
    if fixed_mask is None:
        return set()
    m = np.asarray(fixed_mask)
    return set(int(i) for i in np.where(m == 1)[0])


def _apply_weight_callback(serialized: dict, theta: np.ndarray, callback: Callable) -> dict:
    """Apply a weight callback to produce a non-parameterized serialized graph.

    For each parameterized edge, calls ``callback(theta, coefficients)`` to
    compute a concrete weight, then moves the edge into the regular edges array.

    Parameters
    ----------
    serialized : dict
        From ``Graph.serialize()``.
    theta : ndarray
        Parameter vector.
    callback : callable
        ``(theta, coefficients) -> weight``.

    Returns
    -------
    dict
        Modified serialized dict with all edges as regular (non-parameterized).
    """
    param_edges = serialized['param_edges']
    start_param_edges = serialized['start_param_edges']

    def _check_weight(w: float, theta_in, coeffs_in, where: str) -> float:
        # The C++ JSON parser rejects NaN/inf as invalid literals,
        # producing a cryptic "parse error" downstream. Catch it here
        # at the callback boundary so the user sees which call to
        # their callback produced the bad value.
        if not np.isfinite(w):
            raise ValueError(
                "weight_callback returned a non-finite value "
                f"({w!r}) for {where}.\n"
                f"  theta = {np.asarray(theta_in).tolist()}\n"
                f"  coefficients = {np.asarray(coeffs_in).tolist()}\n"
                "Phase-type edge weights must be finite non-negative "
                "floats. Inspect your weight_callback for branches "
                "that produce NaN/inf at typical theta values "
                "(e.g. binomial pmf called with p outside [0, 1], "
                "division by near-zero, log of non-positive)."
            )
        return w

    # Compute concrete weights via callback
    new_edges = list(serialized['edges'].tolist()) if len(serialized['edges']) > 0 else []
    for edge in param_edges:
        from_idx, to_idx = int(edge[0]), int(edge[1])
        coeffs = np.array(edge[2:])
        weight = _check_weight(
            float(callback(theta, coeffs)),
            theta, coeffs,
            where=f"edge {from_idx} -> {to_idx}",
        )
        new_edges.append([from_idx, to_idx, weight])

    new_start_edges = list(serialized['start_edges'].tolist()) if len(serialized['start_edges']) > 0 else []
    for edge in start_param_edges:
        to_idx = int(edge[0])
        coeffs = np.array(edge[1:])
        weight = _check_weight(
            float(callback(theta, coeffs)),
            theta, coeffs,
            where=f"start edge -> {to_idx}",
        )
        new_start_edges.append([to_idx, weight])

    result = dict(serialized)
    result['edges'] = np.array(new_edges, dtype=np.float64) if new_edges else np.empty((0, 3), dtype=np.float64)
    result['start_edges'] = np.array(new_start_edges, dtype=np.float64) if new_start_edges else np.empty((0, 2), dtype=np.float64)
    result['param_edges'] = np.empty((0, 0), dtype=np.float64)
    result['start_param_edges'] = np.empty((0, 0), dtype=np.float64)
    result['param_length'] = 0
    result['weight_mode'] = 'linear'
    return result


def _generate_cpp_from_graph(serialized: dict) -> str:
    """
    Generate C++ build_model() function from serialized graph.

    Auto-detects if graph has parameterized edges and generates appropriate code.

    Parameters
    ----------
    serialized : dict
        Dictionary from Graph.serialize() containing states, edges, and param info

    Returns
    -------
    str
        C++ code implementing build_model(const double* theta, int n_params)
    """
    states = serialized['states']
    edges = serialized['edges']
    start_edges = serialized['start_edges']
    param_edges = serialized.get('param_edges', np.array([]))
    start_param_edges = serialized.get('start_param_edges', np.array([]))
    param_length = serialized.get('param_length', 0)
    state_dim = serialized['state_length']
    n_vertices = serialized['n_vertices']

    # Generate vertex creation code
    vertex_code = []
    vertex_code.append(f"    auto start = g.starting_vertex_p();")
    vertex_code.append(f"    std::vector<phasic::Vertex*> vertices;")

    # Check if first vertex is the starting vertex (common case)
    # Starting vertex typically has state [0, 0, ...] in state_dim dimensions
    start_state = tuple([0] * state_dim)
    first_vertex_state = tuple(int(s) for s in states[0]) if n_vertices > 0 else None

    for i in range(n_vertices):
        state_vals = ", ".join(str(int(s)) for s in states[i])
        state_tuple = tuple(int(s) for s in states[i])

        # If this is the starting vertex (state is all zeros), use the start pointer
        if state_tuple == start_state:
            vertex_code.append(f"    vertices.push_back(start);  // Starting vertex")
        else:
            vertex_code.append(f"    vertices.push_back(g.find_or_create_vertex_p({{{state_vals}}}));")

    # Create a set of parameterized edge (from, to) pairs to skip in regular edges
    param_edge_pairs = set()
    for edge in start_param_edges:
        to_idx = int(edge[0])
        param_edge_pairs.add((-1, to_idx))  # -1 represents start vertex
    for edge in param_edges:
        from_idx = int(edge[0])
        to_idx = int(edge[1])
        param_edge_pairs.add((from_idx, to_idx))

    # Generate regular edge code
    edge_code = []
    edge_code.append("    // Regular (fixed weight) edges")

    for edge in start_edges:
        to_idx = int(edge[0])
        weight = edge[1]
        # Skip if this edge is also parameterized, or has NaN weight
        if (-1, to_idx) not in param_edge_pairs and not np.isnan(weight):
            edge_code.append(f"    start->add_edge(*vertices[{to_idx}], {weight});")

    for edge in edges:
        from_idx = int(edge[0])
        to_idx = int(edge[1])
        weight = edge[2]
        # Skip if this edge is also parameterized, or has NaN weight
        if (from_idx, to_idx) not in param_edge_pairs and not np.isnan(weight):
            edge_code.append(f"    vertices[{from_idx}]->add_edge(*vertices[{to_idx}], {weight});")

    # Generate parameterized edge code
    param_edge_code = []
    if param_length > 0:
        param_edge_code.append("    // Parameterized edges (weights computed from theta)")

        # Starting vertex parameterized edges
        for i, edge in enumerate(start_param_edges):
            to_idx = int(edge[0])
            coeffs = edge[1:]
            weight_terms = [f"{coeffs[j]}*theta[{j}]"
                           for j in range(len(coeffs))
                           if coeffs[j] != 0.0]
            weight_expr = " + ".join(weight_terms) if weight_terms else "0.0"
            param_edge_code.append(f"    double w_start_{to_idx} = {weight_expr};")
            param_edge_code.append(f"    start->add_edge(*vertices[{to_idx}], w_start_{to_idx});")

        # Regular vertex parameterized edges
        for i, edge in enumerate(param_edges):
            from_idx = int(edge[0])
            to_idx = int(edge[1])
            coeffs = edge[2:]
            weight_terms = [f"{coeffs[j]}*theta[{j}]"
                           for j in range(len(coeffs))
                           if coeffs[j] != 0.0]
            weight_expr = " + ".join(weight_terms) if weight_terms else "0.0"
            param_edge_code.append(f"    double w_{from_idx}_{to_idx} = {weight_expr};")
            param_edge_code.append(f"    vertices[{from_idx}]->add_edge(*vertices[{to_idx}], w_{from_idx}_{to_idx});")

    # Combine all code
    cpp_code = f'''#include "phasiccpp.h"

phasic::Graph build_model(const double* theta, int n_params) {{
    phasic::Graph g({state_dim});

{chr(10).join(vertex_code)}

{chr(10).join(edge_code)}

{chr(10).join(param_edge_code) if param_edge_code else ""}

    return g;
}}
'''
    return cpp_code










def _compile_wrapper_library(wrapper_code: str, lib_name: str, extra_includes: list[str] | None = None) -> str:
    """
    Compile C++ wrapper code to shared library.

    Handles all I/O and subprocess calls during setup phase.
    """
    pkg_dir = _get_package_dir()
    lib_path = os.path.join(_secure_artifact_dir(), f"{lib_name}.so")

    # Remove existing library if present
    if os.path.exists(lib_path):
        os.unlink(lib_path)

    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False, mode='w') as f:
        f.write(wrapper_code)
        wrapper_file = f.name

    try:
        # The JIT-compilation path needs the C/C++ sources on disk so it can
        # build a per-graph wrapper. _get_package_dir() returns
        # Path(__file__).parent.parent.parent which only resolves to a real
        # source tree under an editable install (`pip install -e .`). Wheel
        # installs put __file__ under site-packages where no `src/` exists,
        # producing an opaque "no such file" clang error. Detect that here
        # and emit a self-explanatory message.
        # Must match what CMake links into libphasiccpp (CMakeLists.txt) —
        # phasic.c calls into the SCC machinery (ptd_scc_compose_in_progress
        # etc.), so omitting scc_synthetic.c / scc_compose.c / scc_graph.cpp
        # left this JIT compile with unresolved symbols at link time.
        sources = [
            f'{pkg_dir}/src/cpp/phasiccpp.cpp',
            f'{pkg_dir}/api/cpp/scc_graph.cpp',
            f'{pkg_dir}/src/c/phasic.c',
            f'{pkg_dir}/src/c/phasic_hash.c',
            f'{pkg_dir}/src/c/phasic_log.c',
            f'{pkg_dir}/src/c/scc_synthetic.c',
            f'{pkg_dir}/src/c/scc_compose.c',
        ]
        missing = [s for s in sources if not os.path.exists(s)]
        if missing:
            raise RuntimeError(
                "JIT compilation requires the phasic C/C++ source tree on "
                "disk, but the following files are missing:\n  - "
                + "\n  - ".join(missing)
                + f"\n\nResolved package root: {pkg_dir}\n\n"
                "This usually means phasic was installed from a wheel "
                "(which does not ship the C/C++ sources) instead of as an "
                "editable install. Reinstall with:\n\n"
                "    pip install -e .\n\n"
                "from a checkout of the phasic source tree, or set "
                "PHASIC_SOURCE_DIR to point at one."
            )

        # Include set must match what CMake gives the real build
        # (CMakeLists.txt: `libphasiccpp PRIVATE src/cpp src/c`) — in
        # particular src/c, because api/c/phasic.h does
        # `#include "phasic_log.h"` and that header lives in src/c, not api/c.
        includes = [
            f'-I{pkg_dir}',
            f'-I{pkg_dir}/api/cpp',
            f'-I{pkg_dir}/api/c',
            f'-I{pkg_dir}/include',
            f'-I{pkg_dir}/src/cpp',
            f'-I{pkg_dir}/src/c',
        ]
        if extra_includes:
            includes += [f'-I{inc}' for inc in extra_includes]

        # The C sources must be compiled AS C, not as C++. A single g++ command
        # over mixed sources compiles the .c files in C++ mode, where legal C
        # (e.g. scc_compose.c's goto past an initialiser) is a hard error. CMake
        # builds them with the C compiler; mirror that: compile each translation
        # unit with the right frontend, then link with the C++ driver (the
        # public headers carry extern "C" guards, so the C objects link in
        # cleanly).
        c_sources = [s for s in sources if s.endswith('.c')]
        cxx_sources = [wrapper_file] + [s for s in sources if s.endswith('.cpp')]

        objects = []
        with tempfile.TemporaryDirectory() as objdir:
            for i, src in enumerate(c_sources):
                obj = os.path.join(objdir, f'c_{i}.o')
                _run_cc(
                    ['cc', '-O3', '-fPIC', '-std=c11', '-c', src, '-o', obj]
                    + includes,
                    src,
                )
                objects.append(obj)
            for i, src in enumerate(cxx_sources):
                obj = os.path.join(objdir, f'cxx_{i}.o')
                _run_cc(
                    ['g++', '-O3', '-fPIC', '-std=c++14', '-c', src, '-o', obj]
                    + includes,
                    src,
                )
                objects.append(obj)
            _run_cc(
                ['g++', '-shared', *objects, '-o', lib_path], 'link',
            )
    finally:
        os.unlink(wrapper_file)

    return lib_path


def _run_cc(cmd: list[str], what: str) -> None:
    """Run one compile/link step, surfacing the compiler's own diagnostics."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed ({what}):\n{result.stderr}"
        )


def _setup_ctypes_signatures(lib: Any, has_pmf: bool = True, has_dph: bool = True) -> None:
    """Configure ctypes function signatures on loaded library."""
    if has_pmf:
        lib.compute_pmf.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,                      # n_params
            ctypes.POINTER(ctypes.c_double),  # times
            ctypes.c_int,                      # n_times
            ctypes.POINTER(ctypes.c_double),  # output
            ctypes.c_int                       # granularity
        ]
        lib.compute_pmf.restype = None

    if has_dph:
        lib.compute_dph_pmf.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,                      # n_params
            ctypes.POINTER(ctypes.c_int),     # jumps
            ctypes.c_int,                      # n_jumps
            ctypes.POINTER(ctypes.c_double),  # output
        ]
        lib.compute_dph_pmf.restype = None


def _setup_ctypes_signatures_from_arrays(lib: Any, discrete: bool = False) -> None:
    """Configure ctypes signatures for from_arrays compute function."""
    if discrete:
        # Discrete mode: no granularity parameter, uses int* for jumps
        lib.compute_dph_pmf_from_arrays.argtypes = [
            ctypes.POINTER(ctypes.c_int32),    # states
            ctypes.c_int,                       # n_vertices
            ctypes.c_int,                       # state_dim
            ctypes.POINTER(ctypes.c_double),    # edges
            ctypes.c_int,                       # n_edges
            ctypes.POINTER(ctypes.c_double),    # start_edges
            ctypes.c_int,                       # n_start_edges
            ctypes.POINTER(ctypes.c_int),       # jumps (int* not double*)
            ctypes.c_int,                       # n_jumps
            ctypes.POINTER(ctypes.c_double),    # output
        ]
        lib.compute_dph_pmf_from_arrays.restype = None
    else:
        # Continuous mode: includes granularity parameter
        lib.compute_pmf_from_arrays.argtypes = [
            ctypes.POINTER(ctypes.c_int32),    # states
            ctypes.c_int,                       # n_vertices
            ctypes.c_int,                       # state_dim
            ctypes.POINTER(ctypes.c_double),    # edges
            ctypes.c_int,                       # n_edges
            ctypes.POINTER(ctypes.c_double),    # start_edges
            ctypes.c_int,                       # n_start_edges
            ctypes.POINTER(ctypes.c_double),    # times
            ctypes.c_int,                       # n_times
            ctypes.POINTER(ctypes.c_double),    # output
            ctypes.c_int                        # granularity
        ]
        lib.compute_pmf_from_arrays.restype = None


def _validate_ipv(ipv, allow_single_state=True):

    if not isinstance(ipv, (list, tuple)):
        raise TypeError(
            f"ipv must be a list, got {type(ipv).__name__}. "
            f"Example: ipv=[5] or ipv=[[[5,0,0], 1.0]]"
        )
    if len(ipv) == 0:
        raise ValueError("ipv must be non-empty")

    # Simple format: [int, int, ...] → single starting state
    if all(isinstance(x, (int, np.integer)) for x in ipv):
        pass  # OK — converted to [[state, 1.0]] below
    # Explicit format: [[state, prob], ...]
    elif all(isinstance(x, (list, tuple)) and len(x) == 2 for x in ipv):
        for i, (state, prob) in enumerate(ipv):
            if not isinstance(state, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"ipv[{i}][0] must be a list (state vector), got {type(state).__name__}"
                )
            if not isinstance(prob, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"ipv[{i}][1] must be a number (probability), got {type(prob).__name__}"
                )
    else:
        raise TypeError(
            f"ipv must be a list of ints (e.g. [5]) or list of [state, prob] pairs "
            f"(e.g. [[[5,0], 0.7], [[4,1], 0.3]]). Got: {ipv!r}"
        )

    if all(isinstance(x, (int, np.integer)) for x in ipv):
        if allow_single_state:
            ipv = [[ipv, 1.0]]
        else:
            raise TypeError("To allow specifying a single state pass as IPV, pass allow_single_state=True.")

    return ipv

def _callback(ipv: list) -> Callable:
    """
    Turn callback functions with different signatures into a common one.
    Also makes return the ipv when called with empty state.
    """
    # Validate ipv
    ipv = _validate_ipv(ipv)

    def decorator(func):
       # @wraps(func) don't use wraps to be able to check if decorated from callable name
        def wrapper(state=[], **kwargs):

            assert not (ipv is None and (state is None or len(state) == 0)), "ipv must be provided if callback does not return it"
            assert ipv is not None, "ipv must be provided when building with callback function"

            # return initial probability vector if no state is provided
            if state is None or len(state) == 0:
                assert ipv is not None, "ipv must be provided if callback does not return it"
                _, prob = zip(*ipv)
                try:
                    if sum(prob) < 0 or sum(prob) > 1:
                        raise ValueError("IPV must be non-zero and sum to at most one", ipv)
                    # if abs(sum(prob) - 1.0) > 1e-12:
                    #     raise ValueError("IPV does not sum to one", ipv)
                    return [[s, a, []] for s, a in ipv]               
                except TypeError:
                    return [[s, 1.0, a] for s, a in ipv]               


            for key, value in kwargs.items():
                if isinstance(value, int):
                    print(f"Integer argument {key}={value} will be passed to callback as float")

            try:
                transitions = func(state, **kwargs)
            except:
                # to help user a bit now the function name is 'wrapper' because of no @wraps
                print("Exception raised in callback function")
                raise
    
            for t in transitions:
                assert len(t[0]) == len(state), ("Returned state and input state must be same length", t[0], state)
                assert np.any(t[0] - state), f"Transitions returned by callback function cannot include the input state ({t[0]}): {transitions}"

            # assert all(len(t[0]) == len(state) for t in transitions), ("ipv and state vectors must be same length", transitions, state)

            # empty transitions for absorbing states
            if not transitions:
                return transitions
            
            # make sure returned types are correct
            # Handle 3-tuples: (state, weight, edge_state) for parameterized edges
            # weight is ignored (vestigial); prefer 2-tuple (state, [coeffs]) format
            if len(transitions[0]) == 3:
                return [[list(map(int, s)), float(w), list(e)] for s, w, e in transitions]
            # Handle 2-tuples with list as second element: (state, [coeffs]) - legacy format
            if isinstance(transitions[0][1], list) or isinstance(transitions[0][1], np.ndarray):
                return [[list(map(int, s)), 0.0, list(a)] for s, a in transitions]
            if isinstance(transitions[0][1], tuple):
                assert "Use lists of lists not lists of tuples for transitions"
            else:
                # Handle 2-tuples: (state, weight) for non-parameterized edges
                return [[list(map(int, s)), float(a), []] for s, a in transitions]

        # Store original function and IPV for graph caching
        wrapper.__wrapped__ = func
        wrapper.__ipv__ = ipv
        return wrapper
    return decorator

# allow _callback decorator to be imported as with_ipv
with_ipv = _callback


def _invalidates_trace(method: Callable) -> Callable:
    """Decorator that marks trace as dirty when graph structure changes.

    Used internally to invalidate the cached elimination trace when
    methods that modify graph structure are called (e.g., normalize, discretize).
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if hasattr(self, '_trace_dirty'):
            self._trace_dirty = True
        return result
    return wrapper


# Smallest positive PMF returned when a diverged theta is uncomputable, so
# log(pmf) is a large finite negative rather than -inf -> nan (F: SVGD divergence
# robustness). See _is_rate_blowup / the fail-soft callbacks in
# pmf_and_moments_from_graph.
_PMF_FLOOR = 1e-300


# The native distribution build reports an uncomputable (diverged/unscaled) rate
# in three distinct ways as the rate grows, spanning TWO exception types:
#   - rate ~2.5e8 .. 1e18: ValueError "... too many forward-algorithm steps (cap:
#     1e9)"  (api/cpp/phasiccpp.h; granularity*time exceeds the step cap)
#   - rate >~ 1e18:        RuntimeError "... too large to build/compute a
#     phase-type distribution"  (src/c/phasic.c; auto-granularity would overflow)
#   - legacy discrete:     "... Increase the granularity"
# A diverging SVGD particle enters the ValueError band FIRST, so BOTH types must
# be treated as fail-soft.
_RATE_BLOWUP_EXC = (RuntimeError, ValueError)


def _is_rate_blowup(exc: BaseException) -> bool:
    """True iff ``exc`` is a native 'implied rate too large to build a phase-type
    distribution' error — a diverged/unscaled theta, NOT a code bug.

    A single SVGD particle can wander to a theta whose implied transition rate is
    astronomically large; the native distribution build then raises and, crossing
    the jax.pure_callback boundary, aborts the whole optimize() run. The fail-soft
    callbacks catch ONLY this class of error (across both RuntimeError and
    ValueError, see ``_RATE_BLOWUP_EXC``) and return a finite penalty so the
    optimizer steps away; every other error still propagates. Matching is on the
    exact rate-blowup / step-cap messages, so genuine bugs are never swallowed.
    """
    msg = str(exc)
    return (
        "too large to build a phase-type" in msg
        or "too large to compute a phase-type" in msg
        or "too many forward-algorithm steps" in msg
        or "Increase the granularity" in msg
    )


def _rate_blowup_penalty(times, nr_moments, rewards):
    """Finite (pmf, moments) penalty matching the shapes the native
    compute_pmf_and_moments would have returned, for a diverged/uncomputable
    theta. pmf = _PMF_FLOOR (so log(pmf) is finite-negative), moments = 0."""
    times = np.asarray(times)
    if rewards is not None and np.asarray(rewards).ndim == 2:
        n_features = np.asarray(rewards).shape[1]
        pmf = np.full((times.shape[0], n_features), _PMF_FLOOR, dtype=np.float64)
        moments = np.zeros((n_features, nr_moments), dtype=np.float64)
    else:
        pmf = np.full(times.shape, _PMF_FLOOR, dtype=np.float64)
        moments = np.zeros((nr_moments,), dtype=np.float64)
    return pmf, moments


def _cdf_zero_blowup_penalty(rewards):
    """Finite cdf_zero penalty (shaped like the native output) for a diverged
    theta on the zero-inflation path. A tiny positive value keeps any downstream
    log(cdf_zero) / log(1 - cdf_zero) finite rather than nan."""
    if rewards is not None and np.asarray(rewards).ndim == 2:
        n_features = np.asarray(rewards).shape[0]
        return np.full((n_features,), _PMF_FLOOR, dtype=np.float64)
    return np.full((1,), _PMF_FLOOR, dtype=np.float64)


class Graph(_Graph):
    # def __init__(self, state_length:int=None, callback:Callable=None, ipv:list[list[int] | list[list[int] | float]] | None = None, parameterized:bool=False, **kwargs):
    def __init__(self, arg: int | Callable, ipv: list[int] | list[list[int] | float] | None = None, graph_cache: bool | None = None, **kwargs: Any) -> None:
        """
        Create a graph representing a phase-type distribution. This is the primary entry-point of the library. A starting vertex will always be added to the graph upon initialization.

        The graph can be initialized in two ways:
        - By providing a state length to create an empty graph.
        - By providing a callback function that generates the graph. The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate. For parameterized edges, the callback should return 2-tuples (state, [coefficients]). If the ipv argument is not provided, the function must return the ipv if given an empty state array as argument.

        Parameters
        ----------
        state_length :
            The length of the integer vector used to represent and reference a state, by default None
        callback :
            Callback function accepting a state and returns a list of reachable states and the corresponding transition rates, by default None.
            The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.
        graph_cache : bool or None, optional
            Per-graph override for the on-disk graph cache at
            ``~/.phasic_cache/graphs/``. The cache is keyed by callback
            source code + parameters, enabling instant loading of
            previously built graphs.

            - ``True``: opt in to caching for this graph (load on
              construction, save after build).
            - ``False``: opt out for this graph even if the global
              default is on.
            - ``None`` (default): use the global default from
              ``phasic.configure(graph_cache=...)`` — on by default.
        theta_dim : int, optional
            Number of model parameters (θ). This sets the expected length of parameter vectors
            passed to update_weights(theta).

            **Can be set at two stages:**
            1. At graph construction: `Graph(callback, theta_dim=2)`
            2. At inference time: `graph.svgd(..., theta_dim=3)` - can override if graph was modified

            The value set here establishes the initial parameter dimension. It can be overridden
            later in methods like svgd() if the graph structure has been augmented or changed.

            When theta_dim < edge coefficients_length:
            - **linear / log mode** (update_weights(theta)): ERROR - coefficient and theta lengths must match exactly
            - **callback / formula mode** (update_weights(theta, callback=...) or a weight_formula): OK - the callback receives the full coefficient vector; a formula references t0.. (theta) and c0.. (full coefficients)

            This allows storing auxiliary data in coefficient vectors for use in custom callback/formula
            weights while maintaining a compact theta parameter space. The extra coefficients are accessible
            only through the callback/formula, not in standard dot-product weight computation.

            If not provided, theta_dim (param_length) is inferred from the first edge's coefficient length.
            That default is correct for linear/log. For **formula** and **callback** weight modes you need
            not set it here — it is resolved at inference time (see ``Graph.svgd``): a formula's parameter
            count is read from the formula itself (its highest t-index + 1), and callback mode requires an
            explicit ``theta_dim``/``theta_init`` (a callback cannot be introspected).

            Example:
                >>> g.set_param_length(2)  # Set theta_dim=2 (param_length is C++ method name)
                >>> g.starting_vertex().add_edge(v1, [c1, c2, c3])  # 3 coefficients stored
                >>> g.update_weights([θ1, θ2])  # ERROR: mismatch (2 params vs 3 coeffs)
                >>> g.update_weights([θ1, θ2], lambda theta, coeffs: custom_weight(theta, coeffs))  # OK

            Default: None (auto-detect from edges)
        cache_trace : bool, optional
            If True, caches the elimination trace on the instance for repeated use.
            When False (default), trace is computed fresh each time but not stored.
            For parameterized graphs, trace-based computation is always used (O(n) memory)
            regardless of this setting - this flag only controls caching.
            Default: False
        hierarchical : bool, optional
            Deprecated alias for cache_trace. Use cache_trace instead.

        Returns
        -------
        :
            A graph object representing a phase-type distribution.

        Examples
        --------
        >>> # Normal construction
        >>> graph = Graph(callback, theta=2.0)

        >>> # With caching (fast on second call)
        >>> graph = Graph(callback, theta=2.0, cache=True)  # Builds and caches
        >>> graph = Graph(callback, theta=2.0, cache=True)  # Instant load from cache

        >>> # Using extra coefficients with callback mode
        >>> def callback_with_extra_coeffs(state):
        >>>     c1, c2, c3 = compute_coeffs(state)  # 3 coefficients (c3 is auxiliary data)
        >>>     return [(next_state, 0.0, [c1, c2, c3])]
        >>> graph = Graph(callback_with_extra_coeffs, theta_dim=2)
        >>> # Non-callback mode fails:
        >>> # graph.update_weights([1.5, 2.0])  # ERROR: 2 params vs 3 coeffs
        >>> # Callback mode works:
        >>> graph.update_weights([1.5, 2.0], lambda theta, coeffs: coeffs[0]*theta[0] + coeffs[1]*theta[1] + coeffs[2])  # OK
        """
        # DEPRECATED: cache_trace and hierarchical kwargs.
        #
        # The Python EliminationTrace path that these kwargs once gated
        # is no longer wired to the public moments()/expectation()/
        # variance() entry points (those route directly to the C++
        # super() implementation, which already uses the Stage A0-cached
        # parameterized_reward_compute_graph). Passing cache_trace=True
        # had documented numerical bugs on cyclic graphs and produced
        # RuntimeError on non-parameterised graphs (see
        # tests/pytest/failing_tests.md). The implementation is left
        # in source for the time being but the public kwarg is no
        # longer accepted.
        #
        # Internally self._cache_trace is forced to False so any
        # surviving guards in the trace machinery short-circuit
        # cleanly.
        cache_trace_in = kwargs.pop('cache_trace', None)
        hierarchical_in = kwargs.pop('hierarchical', None)
        if cache_trace_in not in (None, False) or hierarchical_in not in (None, False):
            import warnings
            warnings.warn(
                "Graph(cache_trace=...) / Graph(hierarchical=...) is "
                "deprecated and no longer takes effect. The Python "
                "EliminationTrace path that this kwarg gated has been "
                "retired from the public moments()/expectation()/"
                "variance() entry points; those now route directly to "
                "the C++ implementation. The argument is ignored; the "
                "kwarg will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        cache_trace = False  # forced

        # Original behaviour retained as commented reference:
        #
        # cache_trace = kwargs.get('cache_trace', None)
        # hierarchical = kwargs.get('hierarchical', None)
        # if cache_trace is None and hierarchical is not None:
        #     import warnings
        #     warnings.warn(
        #         "The 'hierarchical' parameter is deprecated. Use 'cache_trace' instead.",
        #         DeprecationWarning,
        #         stacklevel=2
        #     )
        #     cache_trace = hierarchical
        # elif cache_trace is None:
        #     cache_trace = False

        # Validate arg
        if isinstance(arg, int):
            if arg < 1:
                raise ValueError(f"state_length must be >= 1, got {arg}")
        elif not callable(arg) and not isinstance(arg, _Graph):
            raise TypeError(
                f"First argument must be an integer state_length, a callback function, "
                f"or a Graph instance, got {type(arg).__name__}"
            )

        # Validate ipv usage
        if not callable(arg) and ipv is not None:
            raise ValueError("ipv is only used with a callback function, not with integer state_length")

        # Validate theta_dim if provided
        theta_dim_arg = kwargs.get('theta_dim', None)
        if theta_dim_arg is not None:
            if not isinstance(theta_dim_arg, (int, np.integer)):
                raise TypeError(f"theta_dim must be an integer, got {type(theta_dim_arg).__name__}")
            if theta_dim_arg < 1:
                raise ValueError(f"theta_dim must be >= 1, got {theta_dim_arg}")

        self._joint_prob_base_graph_indexer = None  # flag to signify joint probability representation; defaults to until set internally
        self._indexer = kwargs.get('indexer', None)  # StateIndexer this graph was built against; carried for downstream composition (add_epoch, joint_prob_graph)

        # Wrap callback with IPV BEFORE cache operations to ensure consistent hashing
        callback_for_cache = arg
        if callable(arg) and ipv is not None:
            if arg.__name__ != 'wrapper':
                # Wrap with IPV now so cache hash includes it
                callback_for_cache = _callback(ipv)(arg)

        # Resolve the per-graph graph_cache flag against the global
        # default once: graph_cache=None means "use phasic.configure
        # (graph_cache=...)" (defaults True). Explicit True/False
        # bypasses the global.
        if graph_cache is None:
            from .config import get_config
            graph_cache = bool(get_config().graph_cache)

        # Try loading from cache if requested
        if callable(arg) and graph_cache:
            from .graph_cache import GraphCache
            from .logging_config import get_logger
            logger = get_logger(__name__)

            _cache = GraphCache()
            try:
                cached_graph = _cache.load_graph(callback_for_cache, **kwargs)
                if cached_graph is not None:
                    # Cache hit - initialize from cached graph
                    super().__init__(cached_graph)
                    # Copy Python attributes.
                    # The deserialized cached graph only restores graph *structure*;
                    # its _callback is always None because a Python function cannot
                    # be serialized to disk. Use the live callback/kwargs in hand
                    # (callback_for_cache is the same value the non-cache path stores
                    # below) so that extend()/add_epoch() keep working after a cache
                    # hit instead of raising "No callback available".
                    self._callback = callback_for_cache
                    self._callback_kwargs = kwargs.copy()
                    self.is_discrete = cached_graph.is_discrete if hasattr(cached_graph, 'is_discrete') else False
                    self._cache_trace = cache_trace
                    self._trace = None
                    self._trace_dirty = True
                    self._last_theta = None
                    # Match the attribute set established by the
                    # non-cache-hit path further down (lines ~1901+).
                    # Without these, calling ``serialize()`` on the
                    # cache-loaded instance raises AttributeError.
                    self._weight_mode = getattr(cached_graph, '_weight_mode', 'linear')
                    self._weight_callback = getattr(cached_graph, '_weight_callback', None)
                    self._weight_formula = getattr(cached_graph, '_weight_formula', None)
                    self._weight_formula_tape = getattr(cached_graph, '_weight_formula_tape', None)
                    if self._weight_mode == 'formula' and self._weight_formula_tape is not None:
                        # Re-install the tape on this (freshly loaded) live graph
                        # so direct graph.pdf()/update_weights honor the formula.
                        _t = self._weight_formula_tape
                        self._set_weight_tape(list(_t['ops']), list(_t['consts']),
                                              int(_t['stack_depth']),
                                              int(_t['n_theta']), int(_t['n_coeff']))
                    self._last_callback_vertices_length = self.vertices_length()
                    logger.info(f"Loaded graph from cache: {cached_graph.vertices_length()} vertices")

                    return
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                # Fall through to normal construction

        # Store callback and kwargs for later use with extend()
        self._callback = None
        self._callback_kwargs = {}

        if callable(arg):
            # Extract theta_dim from kwargs
            theta_dim = kwargs.get('theta_dim', None)

            # Remove cache_trace/hierarchical and theta_dim from kwargs before passing to C++ callback
            kwargs_for_callback = kwargs.copy()
            kwargs_for_callback.pop('cache_trace', None)
            kwargs_for_callback.pop('hierarchical', None)
            kwargs_for_callback.pop('theta_dim', None)

            # turn integer kwargs into float kwargs
            for key, value in kwargs_for_callback.items():
                if isinstance(value, int):
                    kwargs_for_callback[key] = float(value)

            # Use the wrapped callback (already done before cache operations)
            if callback_for_cache is not arg:
                # Already wrapped for cache
                arg = callback_for_cache
            elif arg.__name__ != 'wrapper':
                if ipv is None:
                    raise ValueError("When providing a function not decorated with @callback, the ipv argument must be provided")
                arg = _callback(ipv)(arg)
            else:
                if ipv is not None:
                    raise ValueError("When providing a function decorated with @callback, the ipv argument is ignored and should not be provided")

            # Store the callback and kwargs for extend() (with hierarchical)
            self._callback = arg
            self._callback_kwargs = kwargs.copy()

            # Pass theta_dim to C++ builder (C++ expects param_length keyword)
            if theta_dim is not None:
                super().__init__(callback_tuples_parameterized=partial(arg, **kwargs_for_callback), param_length=theta_dim)
            else:
                super().__init__(callback_tuples_parameterized=partial(arg, **kwargs_for_callback))
        elif isinstance(arg, int):
            super().__init__(state_length=arg)
        elif isinstance(arg, _Graph):
            super().__init__(arg)
        else:
            raise ValueError("First argument must be either an integer state length or a callback function")

        self.is_discrete = False

        # Trace caching mode for faster repeated evaluations
        self._cache_trace = cache_trace
        self._trace = None  # Cached EliminationTrace
        self._trace_dirty = True  # True = trace needs (re)computation
        self._last_theta = None  # Cached theta from update_weights()
        self._weight_mode = 'linear'  # Weight computation mode: 'linear', 'log', 'callback', or 'formula'
        self._weight_callback = None  # Custom weight callback for 'callback' mode
        self._weight_formula = None  # Formula string for 'formula' mode
        self._weight_formula_tape = None  # Compiled tape dict for 'formula' mode

        self._last_callback_vertices_length = self.vertices_length()  # Track vertices length at last callback call for extend()

        # Save to cache if requested and construction succeeded
        if callable(arg) and graph_cache:
            from .graph_cache import GraphCache
            from .logging_config import get_logger
            logger = get_logger(__name__)

            _cache = GraphCache()
            try:
                # Use same wrapped callback as load for consistent hash
                _cache.save_graph(self, callback_for_cache, **kwargs)
                logger.info(f"Saved graph to cache: {self.vertices_length()} vertices")
            except Exception as e:
                logger.warning(f"Failed to save graph to cache: {e}")


    def vertex_at(self, idx):
        if idx >= self.vertices_length():
            raise ValueError("Vertex at index does not exist")
        return super().vertex_at(idx)


    def __iter__(self):
        yield from (self.vertex_at(i) for i in range(self.vertices_length()))


    def __len__(self):
        return self.vertices_length()


    @_invalidates_trace
    def find_or_create_vertex(self, state: ArrayLike) -> Vertex:
        """Find or create a vertex with the given state.

        This method wraps the C++ implementation to track trace invalidation.
        """
        return super().find_or_create_vertex(state)

    @_invalidates_trace
    def extend(self, callback: Callable | None = None, vertex_index: int | None = None, **kwargs: Any) -> None:
        """Extend the graph by continuing to visit unvisited vertices using a callback.

        After manually adding vertices to the graph (e.g., via find_or_create_vertex),
        this method continues the callback-based graph building process, visiting all
        newly added unvisited vertices.

        Parameters
        ----------
        callback : callable, optional
            Callback function to use for extending the graph. If None, uses the
            callback from the original Graph construction. The callback should
            accept a state array and return:
            - For non-parameterized: list of (state, weight) tuples
            - For parameterized: list of (state, [coefficients]) tuples
        **kwargs
            Additional keyword arguments to pass to the callback function. If None,
            uses the kwargs from the original Graph construction.

        Raises
        ------
        RuntimeError
            If no callback is provided and the graph was not constructed with a callback.

        Examples
        --------
        >>> # Build initial graph
        >>> graph = Graph(my_callback, nr_samples=5)
        >>>
        >>> # Manually add new vertex
        >>> special_vertex = graph.find_or_create_vertex([100, 200])
        >>> graph.starting_vertex().add_edge(special_vertex, [1.5])
        >>>
        >>> # Continue with callback to explore new vertices
        >>> graph.extend()  # Uses original callback
        >>>
        >>> # Or use different callback
        >>> graph.extend(callback=my_other_callback, param=value)
        """
        # Determine which callback to use
        if callback is None:
            if self._callback is None:
                raise RuntimeError(
                    "No callback provided and graph was not constructed with a callback. "
                    "Please provide a callback to extend()."
                )
            callback = self._callback
            # Merge kwargs: stored kwargs as defaults, explicit kwargs override
            merged_kwargs = self._callback_kwargs.copy()
            merged_kwargs.update(kwargs)
            kwargs = merged_kwargs
        else:
            # If callback provided, kwargs must be explicit (don't use stored kwargs)
            pass

        # Convert integer kwargs to float
        for key, value in kwargs.items():
            if isinstance(value, int):
                kwargs[key] = float(value)

        # Create partial function with kwargs
        callback_with_kwargs = partial(callback, **kwargs)

        # Call C++ extension method
        extend_kwargs = {}
        if vertex_index is None:
            extend_kwargs['vertex_index'] = self._last_callback_vertices_length
        else:
            extend_kwargs['vertex_index'] = vertex_index
        super().extend_graph_callback_tuples_parameterized(callback_with_kwargs, **extend_kwargs)

    def _ensure_trace(self) -> EliminationTrace | None:
        """Ensure trace is available for computation.

        Returns the cached trace if caching is enabled, or computes a fresh
        trace on a cloned graph if not. This allows trace-based computation
        to work regardless of the cache_trace setting.

        Returns
        -------
        EliminationTrace or None
            The trace for computation, or None if graph is not parameterized.
        """
        if not self.parameterized():
            return None

        # If caching enabled and trace is valid, return cached trace
        if self._cache_trace:
            if self._trace is None or self._trace_dirty:
                self.compute_trace()
            return self._trace

        # Caching disabled - compute trace on cloned graph (non-destructive)
        from .hierarchical_trace_cache import get_trace_hierarchical
        graph_copy = self.clone()
        return get_trace_hierarchical(graph_copy, param_length=self.param_length())

    def _get_current_theta(self) -> np.ndarray:
        """Get cached theta values set via update_weights().

        Returns
        -------
        np.ndarray
            The parameter vector last set via update_weights().

        Raises
        ------
        RuntimeError
            If graph is not parameterized or no theta has been set.
        """
        if not self.parameterized():
            raise RuntimeError("Graph is not parameterized")

        if self._last_theta is None:
            raise RuntimeError(
                "No parameters set. Call update_weights(theta) before "
                "calling moments()/expectation()."
            )

        return self._last_theta

    @property
    def cache_trace(self) -> bool:
        """Whether this graph caches the elimination trace."""
        return self._cache_trace

    @property
    def dyn_ordering(self) -> bool:
        """Whether dynamic minimum-degree elimination ordering is enabled.

        When True, graph elimination uses dynamic minimum-degree ordering
        within each SCC, which can dramatically reduce fill-in for graphs
        with heterogeneous vertex degree (e.g., ARG models, island models
        with many populations).

        Can also be enabled globally via the PHASIC_DYN_ORDERING=1 environment
        variable (sets the default for new graphs).
        """
        return super().get_dyn_ordering()

    @dyn_ordering.setter
    def dyn_ordering(self, value: bool):
        super().set_dyn_ordering(value)

    @property
    def hierarchical(self) -> bool:
        """Deprecated: use cache_trace instead."""
        import warnings
        warnings.warn(
            "The 'hierarchical' property is deprecated. Use 'cache_trace' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._cache_trace

    @property
    def trace_valid(self) -> bool:
        """Whether the cached trace is valid (not dirty)."""
        return self._trace is not None and not self._trace_dirty

    @property
    def weight_mode(self) -> str:
        """Weight computation mode for parameterized edges.

        One of ``'linear'`` (default), ``'log'``, ``'callback'``, or
        ``'formula'``.

        - ``'linear'``: weight = Σ c_k θ_k
        - ``'log'``: weight = Π(c_k θ_k) (computed in log-space for stability)
        - ``'callback'``: weight = callback(theta, coefficients) (Python; slow)
        - ``'formula'``: weight = a compiled formula string evaluated per edge
          in C (see ``weight_formula``); fast like linear/log, stays on the
          FFI/SVGD path.

        Set ``'formula'`` indirectly by assigning ``graph.weight_formula``.
        """
        return self._weight_mode

    @weight_mode.setter
    def weight_mode(self, mode: str) -> None:
        if mode not in ('linear', 'log', 'callback', 'formula'):
            raise ValueError(
                "weight_mode must be 'linear', 'log', 'callback', or "
                f"'formula', got {mode!r}"
            )
        self._weight_mode = mode

    @property
    def weight_callback(self) -> Callable | None:
        """Custom callback for computing edge weights from theta and coefficients.

        Setting this automatically sets ``weight_mode = 'callback'``.

        The callback signature is ``(theta, coefficients) -> weight`` where
        ``theta`` is a numpy array and ``coefficients`` is a numpy array of
        edge coefficients.
        """
        return self._weight_callback

    @weight_callback.setter
    def weight_callback(self, fn: Callable | None) -> None:
        self._weight_callback = fn
        if fn is not None:
            self._weight_mode = 'callback'
        elif self._weight_mode == 'callback':
            self._weight_mode = 'linear'

    @property
    def weight_formula(self) -> str | None:
        """Per-edge weight formula string, evaluated in C (``weight_mode='formula'``).

        Assigning a formula string compiles it once into a bytecode tape and
        sets ``weight_mode = 'formula'``. The formula references the parameter
        vector as ``t0, t1, …`` and the *current edge's* coefficients as
        ``c0, c1, …`` and may use ``+ - * / **``, ``exp log sqrt logistic pow``,
        comparisons ``== != < > <= >=``, ``delta`` (Kronecker), ``and or not``,
        and ``select(cond, a, b)``. Example::

            graph.weight_formula = "exp(c0*t0 + c1*t1) + c2"

        Unlike ``weight_callback`` (Python; slow and unsupported on the
        daisy-chain path), a formula is computed in C on every theta and stays
        on the OpenMP-parallel FFI/SVGD path.

        Conditions of comparisons/``delta``/``and``/``or``/``not`` and the
        condition of ``select`` must be theta-independent (they may use ``c<j>``
        and constants but not ``t<i>``); a theta-dependent condition raises at
        assignment (use ``logistic(...)`` for smooth theta-gating).

        Assigning ``None`` clears the formula and reverts ``weight_mode`` to
        ``'linear'``.
        """
        return self._weight_formula

    @weight_formula.setter
    def weight_formula(self, formula: str | None) -> None:
        if formula is None:
            self._weight_formula = None
            self._weight_formula_tape = None
            try:
                self._clear_weight_tape()
            except Exception:
                pass
            if self._weight_mode == 'formula':
                self._weight_mode = 'linear'
            return
        from .weight_formula import compile_formula
        tape = compile_formula(formula)   # raises WeightFormulaError on bad input
        self._weight_formula = formula
        self._weight_formula_tape = tape
        self._weight_mode = 'formula'
        # Install on the live graph so the direct graph.pdf()/update_weights
        # paths also honor the formula (the FFI/SVGD path receives the tape via
        # serialize()). No silent fallback to linear on the direct path.
        self._set_weight_tape(list(tape['ops']), list(tape['consts']),
                              int(tape['stack_depth']),
                              int(tape['n_theta']), int(tape['n_coeff']))

    def _restore_weight_formula_state(self, prev_mode: str, prev_formula,
                                      prev_tape) -> None:
        """Restore weight-formula state (Python attrs + live C tape) saved
        before a one-shot ``weight_formula=`` kwarg override.

        Re-installs the prior tape on the live graph (or clears it) so the graph
        is left exactly as it was, including for the direct ``pdf()`` path. Used
        by the ``weight_formula=`` kwargs of ``update_weights`` and ``svgd``.
        """
        if prev_mode == 'formula' and prev_tape is not None:
            self._set_weight_tape(list(prev_tape['ops']), list(prev_tape['consts']),
                                  int(prev_tape['stack_depth']),
                                  int(prev_tape['n_theta']), int(prev_tape['n_coeff']))
        else:
            self._clear_weight_tape()
        self._weight_mode = prev_mode
        self._weight_formula = prev_formula
        self._weight_formula_tape = prev_tape

    def update_weights(self, theta: ArrayLike, callback: Callable | None = None,
                       log: bool = False, weight_formula: str | None = None) -> None:
        """Update parameterized edge weights with given parameters.

        This method wraps the C++ implementation to cache theta for use
        with trace-based computation.

        Parameters
        ----------
        theta : ArrayLike
            Parameter vector θ. Its required length depends on the weight mode
            (mirroring ``Graph(callback, theta_dim=...)``):

            - **linear / log** (no ``callback`` / ``weight_formula``): length must
              equal ``graph.param_length()`` (the edge coefficient length); a
              mismatch raises. Set ``theta_dim`` (= ``param_length``) at
              construction when the parameter count differs from the coefficient
              length.
            - **callback / weight_formula**: θ may be SHORTER than the edge
              coefficient length. The callback receives ``(theta, coefficients)``
              and may use any subset; a formula references ``t0..`` (θ) and
              ``c0..`` (the full per-edge coefficients). No length check is
              imposed here — the callback/formula defines how θ maps to
              coefficients.
        callback : callable, optional
            ``callback(theta, coefficients) -> weight`` for arbitrary per-edge
            weights (one-shot: applies for this call without changing the graph's
            persistent ``weight_mode``). Mutually exclusive with
            ``weight_formula``. Unlike linear/log it tolerates
            ``len(theta) != len(coefficients)``.
        log : bool, default=False
            If True, use log-space computation.
        weight_formula : str, optional
            A weight formula string (see :attr:`weight_formula`). One-shot: this
            call's edge weights are computed from the formula in C, but the
            graph's persistent ``weight_mode`` is left unchanged (mirrors the
            ``callback`` argument). Useful for verifying a formula via
            ``graph.pdf(...)`` before running SVGD. Mutually exclusive with
            ``callback``.

        Notes
        -----
        This updates edge weights on the live Graph object (C++ side).
        It does NOT invalidate the cached trace since it only changes
        parameter values, not graph structure.

        For the JAX/FFI/SVGD pipeline, use ``graph.weight_mode`` instead.
        That property controls how ``pmf_from_graph()`` and
        ``pmf_and_moments_from_graph()`` compute edge weights from theta.
        """
        theta_array = np.asarray(theta, dtype=np.float64)
        if theta_array.ndim != 1:
            raise ValueError(f"theta must be 1-dimensional, got shape {theta_array.shape}")
        if theta_array.size == 0 and callback is None:
            raise ValueError("theta must be non-empty")
        if np.any(np.isnan(theta_array)):
            raise ValueError("theta contains NaN values")
        if theta_array.size > 0 and np.any(np.isinf(theta_array)):
            raise ValueError("theta contains infinite values")
        if callback is not None and weight_formula is not None:
            raise ValueError(
                "update_weights: pass only one of callback= or weight_formula=")

        self._last_theta = np.asarray(theta)
        if weight_formula is not None:
            # One-shot formula (mirrors the callback overload): compile + install
            # the tape, fill edge weights via the C tape branch, then restore the
            # graph's prior weight state so weight_mode is not permanently
            # changed. graph.pdf() afterward reads the filled edge weights.
            _prev = (self._weight_mode, self._weight_formula,
                     self._weight_formula_tape)
            self.weight_formula = weight_formula
            try:
                return super().update_weights(theta, log=False)
            finally:
                self._restore_weight_formula_state(*_prev)
        if callback is not None:
            # C++ overload: update_weights(params, callback) - no log parameter
            return super().update_weights(theta, callback)
        else:
            # C++ overload: update_weights(params, log=False)
            return super().update_weights(theta, log=log)

    def update_ipv(self, ipv: ArrayLike) -> None:
        """Set the initial probability vector after construction.

        Parameters
        ----------
        ipv : array-like, shape (n_ipv_edges,)
            New weights for starting-vertex edges, in construction order.
            Length must equal the number of starting-vertex edges in
            the graph.

        Notes
        -----
        IPV is a property of the model, not an inference parameter. Use
        this method to:

          - set the IPV after constructing the graph if your callback
            does not bake one in;
          - re-run the same graph against a different initial
            distribution without rebuilding it;
          - propagate epoch state in a daisy chain (handled internally
            by the daisy-chain machinery between epochs).

        The symbolic compute graph cache (Stage A0) survives this call,
        so subsequent forward computations (``expectation``, ``pdf``,
        ``compute_pmf``, ...) reuse the cached elimination.

        For users on the Python ``EliminationTrace`` path
        (``cache_trace=True``), IPV remains baked in at trace-record
        time — that path is not affected by ``update_ipv``.
        """
        ipv_array = np.asarray(ipv, dtype=np.float64)
        if ipv_array.ndim != 1:
            raise ValueError(
                f"ipv must be 1-dimensional, got shape {ipv_array.shape}"
            )
        if ipv_array.size == 0:
            raise ValueError("ipv must be non-empty")
        if np.any(np.isnan(ipv_array)):
            raise ValueError("ipv contains NaN values")
        if np.any(np.isinf(ipv_array)):
            raise ValueError("ipv contains infinite values")
        self._last_ipv = ipv_array
        return super().update_ipv(ipv_array)

    # Local trace-cache management lives in _graph_cache_mgmt.py (WS-C split);
    # assigned here so they stay direct Graph members.
    clear_from_cache = _graph_cache_mgmt.clear_from_cache
    prewarm_cache = _graph_cache_mgmt.prewarm_cache

    def expected_waiting_time(self, *args: Any, **kwargs: Any) -> float:
        """
        Compute expected waiting time until absorption.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to C++ implementation.
        **kwargs : dict
            Keyword arguments passed to C++ implementation.

        Returns
        -------
        float
            Expected waiting time until absorption.

        Notes
        -----
        This method wraps the C++ implementation of expected_waiting_time.
        Only available for continuous-time phase-type distributions.
        """
        return super().expected_waiting_time(*args, **kwargs)

    
    def expected_sojourn_time(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """
        Compute expected sojourn time (residence time) in each state.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to C++ implementation.
        **kwargs : dict
            Keyword arguments passed to C++ implementation.

        Returns
        -------
        ArrayLike
            Expected sojourn time for each vertex.

        Notes
        -----
        This method wraps the C++ implementation of expected_sojourn_time.
        Returns the expected accumulated reward for the starting vertex propagated
        through all paths. This is what expectation() uses internally. The difference
        from expected_residence_time is subtle - expected_residence_time decomposes
        this into per-vertex contributions.

        Available for both continuous and discrete phase-type distributions.
        """
        return super().expected_sojourn_time(*args, **kwargs)




    def moments(self, power: int, rewards: ArrayLike = [], discrete: bool = False, **kwargs: Any) -> float:
        """
        Compute k-th moment of the phase-type distribution.

        Parameters
        ----------
        power : int
            Moment order (1 for first moment, 2 for second moment, etc.).
        rewards : ArrayLike, optional
            Reward vector for reward-transformed moments. If not provided,
            uses unit rewards (standard moments).
        discrete : bool, default=False
            If True, compute discrete-time moments (DPH distribution).
            Requires that the graph was discretized via discretize().
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float
            The k-th moment E[T^k] where T is the time until absorption.

        Notes
        -----
        If the graph is parameterized, this method uses trace-based computation
        for 5-10x speedup on repeated evaluations and O(n) memory usage.
        Otherwise falls back to direct C++ graph elimination.

        For higher moments (k > 2), numerical stability may become an issue for
        complex distributions.
        """
        # DISABLED: trace-based routing for parameterised graphs.
        # The C++ moments path (super().moments) already uses the
        # Stage A0-cached parameterized_reward_compute_graph and works
        # for parameterised graphs. Routing through
        # self._ensure_trace() + instantiate_from_trace() recorded a
        # Python trace and rebuilt a graph from it on every call —
        # strictly redundant work for the same numerical result.
        #
        # # For parameterized graphs, always use trace-based computation (O(n) memory)
        # # to avoid O(n²) matrix allocation in the C++ fallback path
        # if self.parameterized():
        #     trace = self._ensure_trace()
        #     if trace is not None:
        #         return self._moments_from_trace(power, rewards=rewards, discrete=discrete, **kwargs)

        # Direct C++ computation (works for both parameterised and
        # non-parameterised graphs).
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            # No native discrete-moment routine exists (super().moments_discrete
            # is unbound); the C layer only computes continuous waiting-time
            # moments. Convert them to discrete raw moments -- the same identity
            # variance_discrete uses (m[1]-m[0]-m[0]^2), generalised. Exact for a DPH.
            cont = super().moments(power, rewards=rewards, **kwargs)
            return _continuous_to_discrete_moments(cont)
        else:
            return super().moments(power, rewards=rewards, **kwargs)

    def expectation(self, rewards: ArrayLike = [], **kwargs: Any) -> float:
        """
        Compute expected value (first moment) of the phase-type distribution.

        Parameters
        ----------
        rewards : ArrayLike, optional
            Reward vector for reward-transformed expectation. If not provided,
            computes E[T] where T is time until absorption.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float
            Expected value E[T] or reward-transformed expectation.

        Notes
        -----
        For parameterized graphs, this method uses trace-based computation
        which requires O(n) memory instead of O(n²) for the matrix-based approach.
        Set cache_trace=True when creating the graph to cache the trace for
        faster repeated evaluations.

        This is equivalent to moments(1, rewards).
        """
        if len(rewards) > 0:
            rewards_arr = np.asarray(rewards, dtype=np.float64)
            if rewards_arr.ndim != 1:
                raise ValueError(f"rewards must be 1-dimensional, got shape {rewards_arr.shape}")
            if np.any(np.isnan(rewards_arr)):
                raise ValueError("rewards contains NaN values")
            n_vertices = self.vertices_length()
            if len(rewards_arr) != n_vertices:
                raise ValueError(
                    f"rewards length ({len(rewards_arr)}) must equal number of vertices ({n_vertices})"
                )

        # DISABLED: trace-based routing for parameterised graphs.
        # See Graph.moments() for the rationale. The
        # _expectation_from_trace path is preserved in source for
        # callers who use it directly, but the public expectation()
        # entry point now always goes through the C++ super() path.
        #
        # if self.cache_trace and self.parameterized():
        #     if self.is_discrete:
        #         raise NotImplementedError("Trace-based expectation computation not implemented yet for discrete graphs.")
        #         # return self._expectation_from_trace(rewards=rewards, discrete=True, **kwargs)
        #     else:
        #         trace = self._ensure_trace()
        #         if trace is None:
        #             raise RuntimeError("No trace, is your Graph parameterized?")
        #         return self._expectation_from_trace(rewards=rewards, discrete=False, **kwargs)

        # Direct C++ computation (works for both parameterised and
        # non-parameterised graphs).
        if self.is_discrete:
            return super().expectation_discrete(rewards=rewards, **kwargs)
        else:
            return super().expectation(rewards=rewards, **kwargs)

    def variance(self, rewards: ArrayLike = [], **kwargs: Any) -> float:
        """
        Compute variance of the phase-type distribution.

        Parameters
        ----------
        rewards : ArrayLike, optional
            Reward vector for reward-transformed variance. If not provided,
            computes Var(T) where T is time until absorption.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float
            Variance Var(T) or reward-transformed variance.

        Notes
        -----
        For parameterized graphs, this method uses trace-based computation
        which requires O(n) memory instead of O(n²) for the matrix-based approach.
        Set cache_trace=True when creating the graph to cache the trace for
        faster repeated evaluations.

        Computed as Var(T) = E[T^2] - E[T]^2 using moments.
        """
        if len(rewards) > 0:
            rewards_arr = np.asarray(rewards, dtype=np.float64)
            if rewards_arr.ndim != 1:
                raise ValueError(f"rewards must be 1-dimensional, got shape {rewards_arr.shape}")
            if np.any(np.isnan(rewards_arr)):
                raise ValueError("rewards contains NaN values")
            n_vertices = self.vertices_length()
            if len(rewards_arr) != n_vertices:
                raise ValueError(
                    f"rewards length ({len(rewards_arr)}) must equal number of vertices ({n_vertices})"
                )

        # DISABLED: trace-based routing for parameterised graphs.
        # See Graph.moments() for the rationale.
        #
        # if self.cache_trace and self.parameterized():
        #     if self.is_discrete:
        #         raise NotImplementedError("Trace-based expectation computation not implemented yet for discrete graphs.")
        #         # return self._variance_from_trace(rewards=rewards, discrete=True, **kwargs)
        #     else:
        #         trace = self._ensure_trace()
        #         if trace is None:
        #             raise RuntimeError("No trace, is your Graph parameterized?")
        #         return self._variance_from_trace(rewards=rewards, discrete=False, **kwargs)

        # Direct C++ computation (works for both parameterised and
        # non-parameterised graphs).
        if self.is_discrete:
            return super().variance_discrete(rewards=rewards, **kwargs)
        else:
            return super().variance(rewards=rewards, **kwargs)

    def covariance(self, rewards1: ArrayLike = [], rewards2: ArrayLike = [], **kwargs: Any) -> np.ndarray:
        """
        Compute covariance matrix for multivariate phase-type distributions.

        Parameters
        ----------
        rewards1 : list of float or ndarray
            The first set of rewards, which should be applied to the 
            phase-type distribution. Must have length equal to 
            `vertices_length()`.
        rewards2 : list of float or ndarray
            The second set of rewards, which should be applied to the 
            phase-type distribution. Must have length equal to 
            `vertices_length()`.
         
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Covariance matrix for the multivariate distribution.

        Notes
        -----
        This method is for multivariate phase-type distributions with
        multiple marginals. For univariate distributions, use variance().
        """
        if self.is_discrete:
            return super().covariance_discrete(rewards1=rewards1, rewards2=rewards2, **kwargs)
        else:
            return super().covariance(rewards1=rewards1, rewards2=rewards2, **kwargs)

    def pdf(self, time: float | ArrayLike, granularity: int = 0, **kwargs: Any) -> float | np.ndarray:
        """
        Compute probability density/mass function using forward algorithm.

        Parameters
        ----------
        time : float or ArrayLike
            Time point(s) at which to evaluate the PDF/PMF.
        granularity : int, optional
            Granularity for uniformization (default: 0, auto-detected as 2*max_rate).
            Higher values improve accuracy but increase computation time.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float or np.ndarray
            PDF/PMF value(s) at the specified time point(s).

        Notes
        -----
        This method uses the forward algorithm (Algorithm 4) via uniformization
        to compute the exact phase-type PDF/PMF, not an approximation.

        For continuous distributions: f(t) = α · exp(S·t) · s*
        For discrete distributions: p(n) = probability of absorption at jump n
        """
        time_arr = np.asarray(time, dtype=np.float64)
        if np.any(np.isnan(time_arr)):
            raise ValueError("time contains NaN values")
        if np.any(time_arr < 0):
            raise ValueError("time must be non-negative")
        if not isinstance(granularity, (int, np.integer)):
            raise TypeError(f"granularity must be an integer, got {type(granularity).__name__}")
        if granularity < 0:
            raise ValueError(f"granularity must be >= 0, got {granularity}")

        if self.is_discrete:
            return super().pdf_discrete(time, **kwargs)
        else:
            return super().pdf(time, granularity=granularity, **kwargs)

    def cdf(self, time: float | ArrayLike, **kwargs: Any) -> float | np.ndarray:
        """
        Compute cumulative distribution function.

        Parameters
        ----------
        time : float or ArrayLike
            Time point(s) at which to evaluate the CDF.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float or np.ndarray
            CDF value(s) P(T ≤ t) at the specified time point(s).

        Notes
        -----
        For continuous distributions: F(t) = P(T ≤ t) = 1 - α · exp(S·t) · 1
        For discrete distributions: F(n) = P(N ≤ n) = sum of PMF up to n
        """
        if self.is_discrete:
            return super().cdf_discrete(time, **kwargs)
        else:
            return super().cdf(time, **kwargs)

    def distribution_context(self, *args: Any, **kwargs: Any) -> Any:
        """
        Create a distribution context for efficient repeated sampling.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        object
            Distribution context object that can be used for efficient sampling.

        Notes
        -----
        The distribution context precomputes data structures needed for
        sampling, making repeated sample() calls much faster than sampling
        directly from the graph.
        """
        if self.is_discrete:
            return super().distribution_context_discrete(*args, **kwargs)
        else:
            return super().distribution_context(*args, **kwargs)

    def sample(
        self,
        n: int,
        *,
        validate_rewards: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate random samples from the phase-type distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        validate_rewards : bool, default True
            If True and ``rewards=`` is provided in ``kwargs``, run the
            coverage check (BFS for a trajectory that skips every
            rewarded vertex). A coverage failure means the
            reward-transformed distribution has a point mass at
            :math:`r = 0`; ``Graph.sample`` will then return some
            zero-valued samples and emits a ``UserWarning`` so the
            user knows the mixture shape. Shape errors are always
            raised regardless of this flag — wrong-length rewards
            are bugs, not modelling choices.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation,
            notably ``rewards=`` (a per-vertex reward vector).

        Returns
        -------
        np.ndarray
            Array of n samples from the distribution.

        Notes
        -----
        Sampling is done by simulating the underlying Markov chain until
        absorption. For more efficient repeated sampling, first create a
        distribution context using distribution_context().
        """
        if "rewards" in kwargs and validate_rewards:
            # Shape errors raise (always); coverage is reported
            # silently. Partial-coverage reward vectors produce a
            # mixture of point-mass-at-0 and continuous-positive
            # samples — that's the correct shape of the data, not
            # a problem to flag.
            kwargs["rewards"] = self._validate_rewards(
                kwargs["rewards"],
                allow_2d=False,
                coverage_mode="report",
                context="rewards",
            )
        if self.is_discrete:
            return np.array(super().sample_discrete(n, **kwargs))
        else:
            return np.array(super().sample(n, **kwargs))

    def sample_path(self, n: int = 1) -> dict | list[dict]:
        """
        Sample complete path(s) through the Markov chain.

        Simulates the underlying Markov chain from the starting vertex until
        absorption, recording every vertex visited and the cumulative time
        at which each vertex was entered.

        Parameters
        ----------
        n : int, default=1
            Number of paths to sample.

        Returns
        -------
        dict or list of dict
            If n=1, returns a single dict with keys:
            - 'vertex_indices': np.ndarray of vertex indices visited
            - 'entry_times': np.ndarray of cumulative entry times

            If n>1, returns a list of such dicts.

        Notes
        -----
        The first entry is always the starting vertex with entry_time=0.
        The last entry is the absorbing vertex. Sojourn times can be
        computed as the differences between consecutive entry times.

        Examples
        --------
        >>> g = Graph(...)  # build graph
        >>> path = g.sample_path()
        >>> path['vertex_indices']  # array of visited vertex indices
        >>> path['entry_times']     # cumulative times
        >>> sojourn_times = np.diff(path['entry_times'])
        """
        return super().sample_path(n)

    def backward_probabilities(self, target_vertices: list[int]) -> np.ndarray:
        """
        Compute P(reach target | start at v) for each vertex v.

        For each vertex, computes the probability of eventually reaching
        one of the target terminal states. Uses backward induction over
        the graph structure.

        Parameters
        ----------
        target_vertices : list of int
            Indices of target terminal vertices.

        Returns
        -------
        np.ndarray
            Array of length vertices_length() with backward probability
            for each vertex. Values are between 0 and 1.
        """
        return np.array(super().backward_probabilities(target_vertices))

    def sample_path_conditioned(self, target_vertices: list[int], n: int = 1) -> dict | list[dict]:
        """
        Sample path(s) conditioned on reaching a target terminal state.

        Uses guided forward sampling: at each step, the next state is
        chosen proportional to ``edge_weight * h(next_state)`` where
        ``h`` is the backward probability of reaching the target. This
        ensures every sampled path ends at one of the target vertices.

        Parameters
        ----------
        target_vertices : list of int
            Indices of target terminal vertices.
        n : int, default=1
            Number of paths to sample.

        Returns
        -------
        dict or list of dict
            Same format as sample_path().
        """
        bp = self.backward_probabilities(target_vertices)
        return super().sample_path_conditioned(bp.tolist(), n)

    def stop_probability(self, time: float | int, **kwargs: Any) -> np.ndarray:
        """
        Compute probability of being in each state at a given time.

        Parameters
        ----------
        time : float or int
            Time point (continuous) or jump number (discrete) at which to
            evaluate state probabilities.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Probability of being in each state at the specified time.

        Notes
        -----
        For continuous distributions: probability vector at time t
        For discrete distributions: probability vector after n jumps
        Computed via matrix exponentiation or uniformization.
        """
        if self.is_discrete:
            return super().stop_probability_discrete(time, **kwargs)
        else:
            return super().stop_probability(time, **kwargs)

    # Alias for stop_probability
    state_probability = stop_probability


    def accumulated_visits(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Compute expected number of visits to each state (discrete only).

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Expected number of visits to each state until absorption.

        Raises
        ------
        ValueError
            If graph is not discrete.

        Notes
        -----
        Only available for discrete-time phase-type (DPH) distributions.
        The graph must be discretized via discretize() before calling this method.
        """
        if not self.is_discrete:
            raise ValueError("accumulated_visits only valid for discrete distributions")
        return super().accumulated_visits(*args, **kwargs)

    def accumulated_visiting_time(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Compute expected time spent in each state (continuous only).

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Expected time spent in each state until absorption.

        Notes
        -----
        Only available for continuous-time phase-type distributions.
        For discrete distributions, use accumulated_visits() instead.
        """
        return super().accumulated_visiting_time(*args, **kwargs)

    def accumulated_occupancy(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Compute expected occupancy (visits or time) for each state.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Expected visits (discrete) or time (continuous) in each state.

        Notes
        -----
        This is a convenience method that dispatches to either:
        - accumulated_visits() for discrete distributions
        - accumulated_visiting_time() for continuous distributions
        """
        if self.is_discrete:
            return self.accumulated_visits(*args, **kwargs)
        else:
            return self.accumulated_visiting_time(*args, **kwargs)

    @_invalidates_trace
    def normalize(self, *args: Any, **kwargs: Any) -> float:
        """
        Normalize edge weights to make the graph a proper probability distribution.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float
            Scaling factor applied to normalize the weights.

        Notes
        -----
        This method modifies the graph in-place and invalidates any cached trace.

        For continuous distributions: Normalizes transition rates.
        For discrete distributions: Normalizes transition probabilities to sum to 1.

        The normalization ensures that the initial probability vector and
        transition matrix satisfy the requirements for a valid phase-type distribution.
        """
        if self.is_discrete:
            return super().normalize_discrete(*args, **kwargs)
        else:
            return super().normalize(*args, **kwargs)

    @_invalidates_trace
    def _discretize_inplace(self, rate: float | Callable, skip_existing: bool = False, **kwargs: Any) -> NDArray[np.int64]:
        """COMPOSABLE_MIGRATION: original in-place discretize, kept for equivalence testing."""
        # COMPOSABLE_MIGRATION: original implementation
        # if not callable(rate):
        #     if not isinstance(rate, (int, float, np.integer, np.floating)):
        #         raise TypeError(f"rate must be a number or callable, got {type(rate).__name__}")
        #     if rate <= 0 or rate >= 1:
        #         raise ValueError(f"rate must be in (0, 1), got {rate}")
        #
        # vlength = self.vertices_length()
        # aux_indices = []
        #
        # for vertex in self.vertices():
        #     if vertex.index() == self.starting_vertex().index() or not vertex.edges():
        #         continue
        #
        #     if skip_existing:
        #         has_aux, is_aux = False, False
        #         for edge in vertex.edges():
        #             if edge.to().state().sum() == 0 and edge.to().edges_length() and edge.to().edges()[0].to().index() == vertex.index():
        #                 has_aux = True
        #                 aux_indices.append(edge.to().index())
        #                 vlength -= 1
        #                 break
        #         if vertex.state().sum() == 0:
        #             is_aux = True
        #         if has_aux or is_aux:
        #             continue
        #
        #     _rate = rate(vertex.state(), **kwargs) if callable(rate) else rate
        #     aux_vertex = vertex.add_aux_vertex(_rate)
        #     aux_vertex.set_aux(True)
        #     aux_indices.append(aux_vertex.index())
        #
        # rewards = np.zeros(vlength+len(aux_indices), dtype=int)
        # for index in aux_indices:
        #     rewards[index] = 1
        #
        # weight_scaling = self.normalize()
        #
        # self.is_discrete = True
        # self.set_was_dph(True)
        #
        # return rewards

        # Functional copy of original for equivalence testing
        if not callable(rate):
            if not isinstance(rate, (int, float, np.integer, np.floating)):
                raise TypeError(f"rate must be a number or callable, got {type(rate).__name__}")
            if rate <= 0:
                raise ValueError(f"rate must be larger than 0, got {rate}")

        vlength = self.vertices_length()
        aux_indices = []

        for vertex in self.vertices():
            if vertex.index() == self.starting_vertex().index() or not vertex.edges():
                continue

            if skip_existing:
                has_aux, is_aux = False, False
                for edge in vertex.edges():
                    if edge.to().state().sum() == 0 and edge.to().edges_length() and edge.to().edges()[0].to().index() == vertex.index():
                        has_aux = True
                        aux_indices.append(edge.to().index())
                        vlength -= 1
                        break
                if vertex.state().sum() == 0:
                    is_aux = True
                if has_aux or is_aux:
                    continue

            _rate = rate(vertex.state(), **kwargs) if callable(rate) else rate
            aux_vertex = vertex.add_aux_vertex(_rate)
            aux_vertex.set_aux(True)
            aux_indices.append(aux_vertex.index())

        rewards = np.zeros(vlength+len(aux_indices), dtype=int)
        for index in aux_indices:
            rewards[index] = 1

        self.normalize()
        self.is_discrete = True
        self.set_was_dph(True)

        return rewards

    def discretize(self, rate: float | Callable, skip_existing: bool = False, **kwargs: Any) -> 'Graph':
        """
        Create a discretized copy of this graph.

        Returns a new graph with auxiliary vertices added for each transient state.
        The original graph is not modified.

        Parameters
        ----------
        rate : float or callable
            Discretization rate. If callable, receives state array and **kwargs,
            must return a scalar rate (for non-parameterized graphs) or a
            coefficient vector (for parameterized graphs).
        skip_existing : bool, optional
            If True, skip vertices that already have auxiliary vertices.

        Returns
        -------
        Graph
            New discretized graph with ``.rewards`` attribute containing the
            reward vector (1 for auxiliary vertices, 0 otherwise).
        """
        if not callable(rate):
            if not isinstance(rate, (int, float, np.integer, np.floating)):
                raise TypeError(f"rate must be a number or callable, got {type(rate).__name__}")
            if rate <= 0:
                raise ValueError(f"rate must be larger than 0, got {rate}")
            
        # For parameterized graphs with scalar rate, widen layout to add a coeff slot
        if self.parameterized() and not callable(rate):
            new_graph = self._rebuild_with_wider_layout(extra_coeff_slots=1)
            _discretize_rate_slot = new_graph.param_length() - 1
        elif self.parameterized() and callable(rate):
            # Callable on parameterized graph: caller provides coefficient vector
            new_graph = self.clone()
            _discretize_rate_slot = None
        else:
            new_graph = self.clone()
            _discretize_rate_slot = None

        vlength = new_graph.vertices_length()
        aux_indices = []

        for vertex in new_graph.vertices():
            if vertex.index() == new_graph.starting_vertex().index() or not vertex.edges():
                continue

            if skip_existing:
                has_aux, is_aux = False, False
                for edge in vertex.edges():
                    if edge.to().state().sum() == 0 and edge.to().edges_length() and edge.to().edges()[0].to().index() == vertex.index():
                        has_aux = True
                        aux_indices.append(edge.to().index())
                        vlength -= 1
                        break
                if vertex.state().sum() == 0:
                    is_aux = True
                if has_aux or is_aux:
                    continue

            _rate = rate(vertex.state(), **kwargs) if callable(rate) else rate
            # For parameterized graphs, add_aux_vertex needs a coefficient vector
            if new_graph.parameterized() and _discretize_rate_slot is not None:
                rate_coeffs = [0.0] * new_graph.param_length()
                rate_coeffs[_discretize_rate_slot] = _rate
                aux_vertex = vertex.add_aux_vertex(rate_coeffs)
            elif new_graph.parameterized():
                # Callable returned a coefficient vector
                aux_vertex = vertex.add_aux_vertex(_rate)
            else:
                aux_vertex = vertex.add_aux_vertex(_rate)
            aux_vertex.set_aux(True)
            aux_indices.append(aux_vertex.index())

        rewards = np.zeros(vlength + len(aux_indices), dtype=int)
        for index in aux_indices:
            rewards[index] = 1

        new_graph.normalize()

        new_graph.is_discrete = True
        new_graph.set_was_dph(True)

        new_graph.rewards = rewards
        return new_graph

    def add_epoch(self, time: float, callback: Callable | None = None, **kwargs: Any) -> 'Graph':
        """
        Add an epoch boundary, returning a new graph with epoch transition edges.

        Computes transition rates from stop_probability(time) / accumulated_occupancy(time)
        and wires up sister vertices in the next epoch. The first call also adds an epoch
        index to the state vector.

        Coefficient layout
        ------------------
        After at least one ``add_epoch`` call, the per-edge coefficient vector follows
        a uniform interleaved layout. With base coefficient count ``B`` (i.e. the length
        of each coefficient list returned by the original callback), epoch ``k`` owns
        slots ``[k*(B+1) .. (k+1)*(B+1) - 1]``: the first ``B`` are that epoch's
        dynamics coefficients, and the last is the transition rate from epoch ``k-1``
        into epoch ``k``. Epoch 0's transition slot is a dummy — its coefficient is
        zero on every edge, so any value supplied at that index has no effect.

        For example, with ``B = 1`` and ``N`` ``add_epoch`` calls (i.e. ``N + 1`` epochs)
        the coefficient vector accepted by ``update_weights`` is::

            [r_0, t_0_dummy, r_1, t_{0->1}, r_2, t_{1->2}, ..., r_N, t_{(N-1)->N}]

        which lets a list of per-epoch rates and times be passed directly via
        ``zip``::

            graph.update_weights([v for pair in zip(rates, times) for v in pair])

        ``param_length()`` grows from ``B`` (no epochs) to ``2*(B+1)`` after the first
        ``add_epoch`` and by ``B + 1`` per subsequent call.

        Parameters
        ----------
        time : float
            Time at which the epoch boundary occurs.
        callback : callable, optional
            Callback for building out new-epoch vertices. If None, uses the
            stored callback from graph construction.
        **kwargs
            Additional keyword arguments merged with stored callback kwargs.

        Returns
        -------
        Graph
            New graph with epoch transitions wired up.
        """
        # Reject the joint -> epoch composition order. add_epoch wires epoch
        # transitions from stop_probability/accumulated_occupancy and extends
        # the next epoch via a per-state callback — machinery defined for the
        # base coalescent graph, not for a joint-probability graph whose state
        # space already folds in reward-count dimensions, trash states, and a
        # discrete absorbing layout. Adding an epoch here would apply the epoch
        # transition to the reward-augmented joint state space, encoding a
        # different model. Epochs must be added to the base graph first; the
        # joint distribution is built last (epoch -> joint).
        if getattr(self, '_joint_prob_base_graph_indexer', None) is not None:
            raise ValueError(
                "add_epoch() is not supported on a joint-probability graph (one "
                "built via joint_prob_graph). Add epoch boundaries to the base "
                "graph BEFORE building the joint distribution, i.e. epoch -> joint:\n"
                "    base  = Graph(callback, indexer=indexer)\n"
                "    base  = base.add_epoch(t1)            # epoch(s) on the base graph\n"
                "    joint = base.joint_prob_graph(...)    # fold mutations in last\n"
                "The reverse order (joint_prob_graph then add_epoch) is rejected "
                "because the epoch transition would be applied to the reward-"
                "augmented joint state space, which encodes a different model."
            )

        if callback is None and self._callback is None:
            raise RuntimeError(
                "No callback available. Either construct the graph with a callback "
                "or provide one to add_epoch()."
            )

        # Determine if this is the first epoch (need to add state dimension)
        is_first_epoch = not hasattr(self, '_epoch_state_index')
        extra_state = 1 if is_first_epoch else 0

        # Determine the base param_length (original callback's coefficient count)
        if hasattr(self, '_base_param_length'):
            base_param_length = self._base_param_length
        else:
            base_param_length = self.param_length()

        # Each epoch owns (base_param_length + 1) slots: B dynamics slots + 1 transition slot.
        # The first add_epoch call also reserves a dummy transition slot for epoch 0
        # (already present at the front via the existing B dynamics slots), so all
        # epochs have a uniform layout. Subsequent calls add one epoch's worth of slots.
        if is_first_epoch:
            extra_coeff = base_param_length + 2  # epoch 0 dummy transition + epoch 1's (B + 1) slots
        else:
            extra_coeff = base_param_length + 1

        # Rebuild with wider layout
        new_graph = self._rebuild_with_wider_layout(
            extra_state_dims=extra_state,
            extra_coeff_slots=extra_coeff
        )

        # Track epoch metadata
        if is_first_epoch:
            new_graph._epoch_state_index = self.state_length()  # index of new epoch dim
            new_graph._n_epochs = 1
        else:
            new_graph._epoch_state_index = self._epoch_state_index
            new_graph._n_epochs = self._n_epochs + 1
        new_graph._base_param_length = base_param_length

        # Propagate an epoch-aware indexer so downstream composition
        # (joint_prob_graph, etc.) sees an indexer whose state_length matches
        # the augmented graph's state vector length. The base indexer is
        # preserved verbatim under _base_indexer for chained add_epoch calls.
        if is_first_epoch:
            base_indexer = self._indexer
        else:
            base_indexer = getattr(self, '_base_indexer', None) or self._indexer
        new_graph._base_indexer = base_indexer
        if base_indexer is not None:
            existing_slot_names = [s.name for s in base_indexer.slots()]
            if 'epoch' in existing_slot_names:
                raise ValueError(
                    "Cannot add an epoch dimension: the base indexer already "
                    "has a slot named 'epoch'."
                )
            new_graph._indexer = StateIndexer(
                *existing_slot_names, 'epoch',
                property_sets=list(base_indexer.property_sets()),
            )
        else:
            new_graph._indexer = None

        epoch_idx = new_graph._n_epochs  # current epoch number (0 was original)
        epoch_state_idx = new_graph._epoch_state_index

        # Compute transition rates on the ORIGINAL graph
        stop_probs = np.array(self.stop_probability(time))
        acc_occ = np.array(self.accumulated_occupancy(time))

        with np.errstate(invalid='ignore'):
            transition_rates = stop_probs / acc_occ

        # Uniform interleaved layout: epoch k owns slots [k*(B+1) .. (k+1)*(B+1) - 1],
        # where the first B are dynamics and the last is the transition rate from
        # epoch k-1 into epoch k (a dummy that stays 0.0 for k == 0).
        new_epoch_dynamics_start = epoch_idx * (base_param_length + 1)
        epoch_trans_coeff_idx = new_epoch_dynamics_start + base_param_length

        # Wire epoch transitions on the new graph
        n_vertices_before_extend = new_graph.vertices_length()

        for i in range(1, n_vertices_before_extend):
            vertex = new_graph.vertex_at(i)
            state = vertex.state()

            # Skip absorbing vertices
            if vertex.edges_length() == 0:
                continue

            # Only process vertices in the previous epoch
            if state[epoch_state_idx] != epoch_idx - 1:
                continue

            # Skip if transition rate is NaN (unreachable state)
            if i >= len(transition_rates) or np.isnan(transition_rates[i]):
                continue

            # Create sister vertex in new epoch
            sister_state = state.copy()
            sister_state[epoch_state_idx] = epoch_idx
            child = new_graph.find_or_create_vertex(sister_state)

            # Add epoch transition edge with rate in the epoch transition slot
            coeff = np.zeros(new_graph.param_length())
            coeff[epoch_trans_coeff_idx] = transition_rates[i]
            vertex.add_edge(child, list(coeff))

        # Prepare callback for extending new-epoch vertices
        # Use the base callback (before any epoch wrapping), not a previously wrapped callback
        if callback is not None:
            use_callback = callback
        elif hasattr(self, '_base_callback') and self._base_callback is not None:
            use_callback = self._base_callback
        else:
            use_callback = self._callback

        use_kwargs = {}
        if callback is None:
            if hasattr(self, '_base_callback_kwargs') and self._base_callback_kwargs:
                use_kwargs = self._base_callback_kwargs.copy()
            elif self._callback_kwargs:
                use_kwargs = self._callback_kwargs.copy()
        use_kwargs.update(kwargs)

        # Wrap callback to:
        # 1. Strip epoch dimension from state before passing to original callback
        # 2. Re-append epoch index to returned states
        # 3. Route base coefficients to the correct per-epoch slots
        _new_param_length = new_graph.param_length()
        _new_epoch_dynamics_start = new_epoch_dynamics_start

        def epoch_callback_wrapper(state, **kw):
            # Only generate transitions for vertices in the current epoch
            if state[epoch_state_idx] != epoch_idx:
                return []

            # Strip epoch dim added by add_epoch before passing to original callback
            base_state = np.delete(state, epoch_state_idx)

            # Call original callback
            transitions = use_callback(base_state, **kw)

            # Re-append epoch index and route coefficients to per-epoch slots
            result = []
            for transition in transitions:
                # Handle both 2-tuple (state, coeffs) and 3-tuple (state, weight, coeffs) formats
                if len(transition) == 2:
                    child_state, coeffs = transition[0], transition[1]
                elif len(transition) == 3:
                    child_state, coeffs = transition[0], transition[2]
                else:
                    raise ValueError(f"Unexpected transition format with {len(transition)} elements")

                # Re-insert epoch dimension
                new_child = np.insert(np.asarray(child_state, dtype=int),
                                      epoch_state_idx, epoch_idx)

                # Route base coefficients to this epoch's slots
                if not hasattr(coeffs, '__iter__'):
                    coeffs = [coeffs]
                coeffs_list = list(coeffs)

                # Create full-width coefficient vector with zeros
                padded = [0.0] * _new_param_length
                # Place base coefficients at the new epoch's dynamics slots
                for k, c in enumerate(coeffs_list[:base_param_length]):
                    padded[_new_epoch_dynamics_start + k] = c

                result.append([new_child, padded])
            return result

        # Extend graph to build out new-epoch vertices
        new_graph._callback = epoch_callback_wrapper
        new_graph._callback_kwargs = use_kwargs
        # Store base callback for chained add_epoch calls
        new_graph._base_callback = use_callback
        new_graph._base_callback_kwargs = use_kwargs.copy()
        new_graph.extend(epoch_callback_wrapper, **use_kwargs)

        return new_graph

    def reward_transform(
        self,
        rewards: np.ndarray,
        *,
        validate_rewards: bool = True,
    ) -> Self:
        """
        Apply reward transformation to create a new graph with modified rewards.

        Parameters
        ----------
        rewards : np.ndarray
            Reward vector of length n_vertices. Each element specifies the
            reward associated with visiting the corresponding vertex.
        validate_rewards : bool, default True
            If True, run the coverage check and emit a ``UserWarning``
            when some trajectories accumulate zero reward — i.e. the
            transformed distribution has a point mass at :math:`r = 0`.
            The transformation always proceeds regardless: an (atom +
            continuous) mixture is a legitimate phase-type distribution,
            useful for Laplace transforms, conditional expectations,
            and zero-inflated likelihood inference via ``Graph.svgd``.
            Set ``False`` to silence the warning. Shape errors are
            always raised regardless of this flag.

        Returns
        -------
        Graph
            New graph with reward-transformed distribution.

        Notes
        -----
        Reward transformation is used to compute moments and expectations
        for different reward structures. The transformation modifies the
        graph to compute E[∑ rewards[i] * time_in_state_i] instead of E[T].

        For continuous distributions: Uses continuous reward transformation.
        For discrete distributions: Uses discrete reward transformation.

        The returned graph is a new Graph object (not modified in-place).

        See Also
        --------
        moments : Compute moments with optional reward vector
        expectation : Compute expectation with optional reward vector
        """
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if np.any(np.isnan(rewards_arr)):
            raise ValueError("rewards contains NaN values")
        if validate_rewards:
            # Shape errors always raise (wrong-length rewards is a
            # bug). Coverage is reported silently: a partial-coverage
            # reward vector produces a (atom + continuous) mixture
            # graph, which is a legitimate phase-type distribution.
            self._validate_rewards(
                rewards_arr,
                allow_2d=False,
                check_coverage=True,
                coverage_mode="report",
                context="rewards",
            )

        if self.is_discrete:
            result = Graph(super().reward_transform_discrete(rewards))
            # Wrapping the C++ _Graph runs __init__, which resets
            # is_discrete=False; a reward-transformed DPH is still
            # discrete, so restore the flag.
            result.is_discrete = True
            # Propagate was_dph from the SOURCE, don't latch True: a discretize()
            # source (was_dph=True) still needs auto-normalisation, but a NATIVE
            # DPH (was_dph=False) must not be renormalised or it collapses to a
            # deterministic walk.
            result.set_was_dph(self.get_was_dph())
            return result
        else:
            return Graph(super().reward_transform(rewards))

    def reward_transform_discrete(self, rewards:np.ndarray) -> Self:
        """
        Apply reward transformation for discrete-time distributions.

        Parameters
        ----------
        rewards : np.ndarray
            Reward vector of length n_vertices.

        Returns
        -------
        Graph
            New graph with discrete reward transformation applied.

        Notes
        -----
        This method is specific to discrete-time phase-type (DPH) distributions.
        For automatic dispatch, use reward_transform() instead.

        See Also
        --------
        reward_transform : General reward transformation (dispatches to this for discrete graphs)
        """
        result = Graph(super().reward_transform_discrete(rewards))
        # Preserve discreteness across the C++ -> Python re-wrap (see
        # reward_transform); otherwise downstream pmf/moments dispatch
        # to the continuous path.
        result.is_discrete = True
        # Propagate was_dph from the source (see reward_transform): a native DPH
        # (was_dph=False) must not be latched to auto-normalise.
        result.set_was_dph(self.get_was_dph())
        return result

    def laplace_transform(self, theta: float) -> Self:
        """
        Create a Laplace-transformed graph.

        Returns a new graph where each transient state has an additional edge
        to the absorbing state with weight theta (or theta added to existing
        absorbing edge weight).

        To compute the Laplace transform value L(theta) = E[exp(-theta * T)],
        call expectation() on the result with a reward vector that is 1 for
        states that had edges to absorbing in the original graph and 0 otherwise.

        Parameters
        ----------
        theta : float
            The Laplace transform parameter.

        Returns
        -------
        Graph
            A new graph representing the Laplace-transformed distribution.

        Examples
        --------
        >>> import numpy as np
        >>> graph = Graph(coalescent_callback)
        >>> # Get reward vector: 1 for states with absorbing edges, 0 otherwise
        >>> rewards = graph.absorbing_state_rewards()
        >>> laplace_graph = graph.laplace_transform(0.5)
        >>> laplace_value = laplace_graph.expectation(rewards=rewards)

        Notes
        -----
        The Laplace transform of a phase-type distribution PH(alpha, S) is:

            L(theta) = E[exp(-theta * T)] = alpha * (theta*I - S)^(-1) * s

        where s is the exit rate vector indicating states with direct
        transitions to absorption.

        For continuous distributions only. For discrete distributions, use the
        z-transform instead.

        See Also
        --------
        absorbing_state_rewards : Get the reward vector for Laplace transform computation
        """
        if self.is_discrete:
            raise ValueError("Laplace transform only available for continuous distributions. "
                           "Use z-transform for discrete distributions.")
        result = Graph(super().laplace_transform(theta))
        # Copy metadata from source graph
        result._callback = self._callback
        result._callback_kwargs = self._callback_kwargs.copy() if self._callback_kwargs else {}
        _propagate_weight_state(self, result)
        result._last_callback_vertices_length = result.vertices_length()
        # COMPOSABLE_MIGRATION: original implementation
        # return Graph(super().laplace_transform(theta))
        return result

    # Reward-validation / vertex-index helpers live in _graph_reward_validation.py
    # (WS-C split); assigned here so they stay direct Graph members.
    absorbing_state_rewards = _graph_reward_validation.absorbing_state_rewards
    _starting_vertex_indices = _graph_reward_validation._starting_vertex_indices
    _absorbing_vertex_indices = _graph_reward_validation._absorbing_vertex_indices
    _validate_reward_coverage = _graph_reward_validation._validate_reward_coverage
    _validate_rewards = _graph_reward_validation._validate_rewards

    def _initial_probability_vector(self) -> np.ndarray:
        """Return the IPV as a length-n_vertices np.float64 array.

        Reads the synthetic start vertex's outgoing edges; the
        ``weight()`` of each edge is the initial probability mass
        placed on the target vertex. Same convention used by
        ``_starting_vertex_indices`` and ``absorbing_state_rewards``.
        """
        n_v = self.vertices_length()
        alpha = np.zeros(n_v, dtype=np.float64)
        for e in self.starting_vertex().edges():
            w = float(e.weight())
            if w > 0.0:
                alpha[int(e.to().index())] = w
        return alpha

    def reward_visit_probability(
        self,
        rewards,
        theta=None,
    ):
        """Probability of visiting any rewarded vertex before absorption.

        For a 1D reward vector ``rewards`` and (parameterized) graph at
        parameter ``theta``, returns the scalar probability that a
        trajectory absorbed from the starting vertex visits at least
        one vertex with reward > 0 before being absorbed. This is the
        phase-type quantity

            p(theta) = sum_{v in starts} alpha_v * h_v(theta)

        where ``h_v(theta) = P(reach a rewarded vertex | start at v)``
        and ``alpha`` is the IPV. ``h`` is computed via
        ``Graph.backward_probabilities`` using the rewarded vertices
        as the target set; the result depends only on graph topology,
        IPV, and theta — NOT on observed reward times.

        Parameters
        ----------
        rewards : array-like, shape (n_vertices,)
            Per-vertex reward; only the nonzero pattern is used.
        theta : array-like, optional
            Parameter vector. If None, uses the graph's current
            parameter setting. If a JAX tracer is passed, the
            JAX-differentiable FFI path is used and the return is a
            jax.Array. Otherwise the result is a Python float.

        Returns
        -------
        float or jax.Array, shape ()
            Probability p(theta) in [0, 1].

        Notes
        -----
        When p < 1, some absorbing trajectories accumulate zero
        cumulative reward. The induced reward distribution is then a
        mixture (point mass at 0 + continuous part for r > 0); use
        ``Graph.svgd(..., rewards=...)`` to fit it with a
        zero-inflated likelihood automatically.
        """
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        n_v = self.vertices_length()
        if rewards_arr.shape != (n_v,):
            raise ValueError(
                f"rewards must have shape (n_vertices={n_v},); "
                f"got shape {rewards_arr.shape}."
            )

        rewarded = [int(v) for v in np.where(rewards_arr > 0.0)[0]]
        if not rewarded:
            # No rewarded vertices: p is trivially zero.
            try:
                import jax
                if isinstance(theta, (jax.Array, jax.core.Tracer)):
                    import jax.numpy as jnp
                    return jnp.asarray(0.0, dtype=jnp.float64)
            except ImportError:
                pass
            return 0.0

        # JAX-traced theta routes through the FFI path.
        try:
            import jax
            import jax.numpy as jnp
            if isinstance(theta, (jax.Array, jax.core.Tracer)):
                from .ffi_wrappers import compute_reward_visit_probability_ffi
                structure_json = self.serialize()
                alpha = jnp.asarray(
                    self._initial_probability_vector(), dtype=jnp.float64,
                )
                return compute_reward_visit_probability_ffi(
                    structure_json,
                    theta,
                    jnp.asarray(rewarded, dtype=jnp.int32),
                    alpha,
                )
        except ImportError:
            pass

        # Concrete-theta path. Uses the graph's current parameter
        # setting if theta is None; otherwise temporarily updates
        # weights to compute h then restores them.
        if theta is not None:
            self.update_weights(np.asarray(theta, dtype=np.float64))

        h = self.backward_probabilities(rewarded)
        alpha = self._initial_probability_vector()
        return float(np.sum(alpha * h))

    def _partial_coverage_features(self, rewards) -> list[int]:
        """Return indices of features (rows) that fail the coverage check.

        Thin shim over :func:`phasic.zero_inflation.partial_coverage_features`;
        see that module for the implementation.
        """
        from .zero_inflation import partial_coverage_features
        return partial_coverage_features(self, rewards)

    def _attach_zero_inflated_term(
        self,
        model,
        *,
        rewards,
        offenders: list[int],
        observed_data,
    ) -> None:
        """Wire the zero-inflated likelihood term onto an SVGD model.

        Thin shim over
        :func:`phasic.zero_inflation.attach_zero_inflated_term`; see
        that module for the implementation.
        """
        from .zero_inflation import attach_zero_inflated_term
        attach_zero_inflated_term(
            self, model,
            rewards=rewards,
            offenders=offenders,
            observed_data=observed_data,
        )

    # serialize / from_serialized live in _graph_serialize.py (WS-C split); assigned
    # here so they stay direct Graph members (from_serialized wrapped as classmethod).
    serialize = _graph_serialize.serialize
    from_serialized = classmethod(_graph_serialize.from_serialized)

    def as_matrices(self) -> MatrixRepresentation:
        """
        Convert the graph to its matrix representation.

        Returns a NamedTuple containing the traditional phase-type distribution
        matrices and associated information.

        Returns
        -------
        MatrixRepresentation
            NamedTuple with the following attributes:
            - states: np.ndarray of shape (n_states, state_dim), dtype=int32
                State vector for each vertex
            - sim: np.ndarray of shape (n_states, n_states), dtype=float64
                Sub-intensity matrix
            - ipv: np.ndarray of shape (n_states,), dtype=float64
                Initial probability vector
            - indices: np.ndarray of shape (n_states,), dtype=int32
                1-based indices for vertices (for use with vertex_at())

        Examples
        --------
        >>> g = Graph(1)
        >>> start = g.starting_vertex()
        >>> v1 = g.find_or_create_vertex([1])
        >>> v2 = g.find_or_create_vertex([2])
        >>> start.add_edge(v1, 1.0)
        >>> v1.add_edge(v2, 2.0)
        >>> g.normalize()
        >>>
        >>> matrices = g.as_matrices()
        >>> print(matrices.sim)  # Sub-intensity matrix (attribute access)
        >>> print(matrices.ipv)  # Initial probability vector
        >>> # Can also use index access like a tuple
        >>> states, sim, ipv, indices = matrices
        """
        # Call the C++ method which returns a dict
        result_dict = super().as_matrices()

        # Convert dict to NamedTuple
        return MatrixRepresentation(
            ipv=result_dict['ipv'],
            sim=result_dict['sim'],
            states=result_dict['states'],
            indices=result_dict['indices']
        )

    @classmethod
    def from_matrices(cls, ipv: np.ndarray, sim: np.ndarray, states: np.ndarray | None = None) -> Self:
        """
        Construct a Graph from matrix representation.

        Parameters
        ----------
        ipv : np.ndarray
            Initial probability vector, shape (n_states,)
        sim : np.ndarray
            Sub-intensity matrix, shape (n_states, n_states)
        states : np.ndarray, optional
            State vectors, shape (n_states, state_dim), dtype=int32
            If None, uses default states [0], [1], [2], ...

        Returns
        -------
        Graph
            The reconstructed phase-type distribution graph

        Examples
        --------
        >>> ipv = np.array([0.6, 0.4])
        >>> sim = np.array([[-2.0, 1.0], [0.0, -3.0]])
        >>> g = Graph.from_matrices(ipv, sim)
        >>> pdf = g.pdf(1.0)
        """
        # Call the C++ static method to create the base graph
        if states is not None:
            base_graph = _Graph.from_matrices(ipv, sim, states)
        else:
            base_graph = _Graph.from_matrices(ipv, sim)

        # Wrap it in our Python Graph class to get all the Python methods
        # We need to create a new Python Graph and copy the data
        state_length = base_graph.state_length()
#        wrapped = cls(state_length)

        # This is a workaround - ideally we'd have a better way to wrap
        # For now, return the base graph which works but doesn't have our Python methods
        # TODO: Implement proper wrapping or copy constructor
        return Graph(base_graph)

    @classmethod
    def pmf_from_graph(cls, graph: Graph, discrete: bool = False, use_cache: bool = True, theta_dim: int | None = None) -> Callable:
        """
        Convert a Python-built Graph to a JAX-compatible function with full gradient support.

        This method automatically detects if the graph has parameterized edges (edges with
        state vectors) and generates optimized C++ code to enable full JAX transformations
        including gradients, vmap, and jit compilation.

        For direct C++ access without JAX wrapping, use the Graph object's methods directly:
        graph.pdf(time), graph.dph_pmf(jump), graph.moments(power), etc.

        Raises
        ------
        ImportError
            If JAX is not installed. Install with: pip install jax jaxlib

        Parameters
        ----------
        graph : Graph
            Graph built using the Python API. Can have regular edges or parameterized edges.
        discrete : bool
            If True, uses discrete phase-type distribution (DPH) computation.
            If False, uses continuous phase-type distribution (PDF).
        use_cache : bool, optional
            If True, uses symbolic DAG cache to avoid re-computing expensive symbolic
            elimination for graphs with the same structure. Default: True
            Set to False to disable caching (useful for testing).

        Returns
        -------
        callable
            If graph has parameterized edges:
                JAX-compatible function (theta, times) -> pmf_values
                Supports JIT, grad, vmap, etc.
            If graph has no parameterized edges:
                JAX-compatible function (times) -> pmf_values
                Supports JIT (backward compatible signature)

        Examples
        --------
        # Non-parameterized graph (regular edges only)
        >>> g = Graph(1)
        >>> start = g.starting_vertex()
        >>> v0 = g.find_or_create_vertex([0])
        >>> v1 = g.find_or_create_vertex([1])
        >>> start.add_edge(v0, 1.0)
        >>> v0.add_edge(v1, 2.0)  # fixed weight
        >>>
        >>> model = Graph.pmf_from_graph(g)
        >>> times = jnp.linspace(0, 5, 50)
        >>> pdf = model(times)  # No theta needed

        # Parameterized graph (with edge states for gradient support)
        >>> g = Graph(1)
        >>> start = g.starting_vertex()
        >>> v0 = g.find_or_create_vertex([0])
        >>> v1 = g.find_or_create_vertex([1])
        >>> start.add_edge(v0, 1.0)
        >>> v0.add_edge_parameterized(v1, 0.0, [2.0, 0.5])  # weight = 2.0*theta[0] + 0.5*theta[1]
        >>>
        >>> model = Graph.pmf_from_graph(g)
        >>> theta = jnp.array([1.0, 3.0])
        >>> pdf = model(theta, times)  # weight becomes 2.0*1.0 + 0.5*3.0 = 3.5
        >>>
        >>> # Full JAX support for parameterized graphs
        >>> grad_fn = jax.grad(lambda t: jnp.sum(model(t, times)))
        >>> gradient = grad_fn(theta)  # Gradients work!

        # For direct C++ access (no JAX overhead), use graph methods:
        >>> pdf_value = g.pdf(1.5)  # Direct C++ call
        >>> pmf_value = g.dph_pmf(3)  # Direct C++ call

        # With symbolic DAG caching (default)
        >>> model = Graph.pmf_from_graph(g, use_cache=True)  # First call: computes and caches
        >>> model2 = Graph.pmf_from_graph(g, use_cache=True)  # Subsequent: instant from cache!
        """
        # Check if JAX is available
        try:
            _ensure_jax_active()
        except Exception as _e:
            raise ImportError(
                "JAX is required for JAX-compatible models. Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            ) from _e

        # Note: Symbolic cache (symbolic_cache.py) has been removed as obsolete.
        # The trace-based elimination system (trace_elimination.py) is now used instead,
        # providing better performance for repeated evaluations.
        # See: CACHING_SYSTEM_OVERVIEW.md for details.

        # Serialize the graph (now includes parameterized edges)
        serialized = graph.serialize(theta_dim=theta_dim)
        # graph.parameterized() is the source of truth: it reflects edge_mode
        # locking (PARAMETERIZED set by the first array-syntax add_edge).
        # serialized['param_length'] is always >= 1 once any non-IPV edge has
        # been added, even for constant graphs (the C layer stores constant
        # weights as a single-element coefficient array), so it can't be used
        # to detect the parameterized dispatch path.
        has_param_edges = bool(graph.parameterized())

        # Generate C++ build_model() code from the serialized graph
        cpp_code = _generate_cpp_from_graph(serialized)

        # Create hash of the generated C++ code
        cpp_hash = hashlib.sha256(cpp_code.encode()).hexdigest()[:16]
        temp_file = os.path.join(_secure_artifact_dir(), f"graph_model_{cpp_hash}.cpp")

        # Write C++ code to temp file
        with open(temp_file, 'w') as f:
            f.write(cpp_code)

        # Return appropriate signature based on parameterization
        if has_param_edges and serialized.get('weight_mode') == 'callback':
            # CALLBACK MODE: Python-level weight computation before C++ call
            import json
            from .ffi_wrappers import _make_json_serializable
            from . import phasic_pybind as cpp_module

            weight_callback = graph.weight_callback
            if weight_callback is None:
                raise ValueError(
                    "Graph has weight_mode='callback' but no weight_callback set. "
                    "Set graph.weight_callback = my_callback_fn before calling pmf_from_graph()."
                )

            # Keep original serialized dict for callback application
            _serialized = serialized

            def _compute_pdf_callback(theta_np, times_np):
                """Apply weight callback, build graph, compute PDF."""
                concrete = _apply_weight_callback(_serialized, theta_np, weight_callback)
                json_str = json.dumps(_make_json_serializable(concrete))
                builder = cpp_module.parameterized.GraphBuilder(json_str)
                # theta is unused by non-parameterized builder, pass dummy
                return builder.compute_pmf(
                    np.zeros(0), times_np, discrete=discrete, granularity=0
                )

            def model_pure(theta, times):
                result_shape = jax.ShapeDtypeStruct(times.shape, times.dtype)
                return jax.pure_callback(
                    lambda t, tm: _compute_pdf_callback(
                        np.asarray(t, dtype=np.float64),
                        np.asarray(tm, dtype=np.float64)
                    ).astype(times.dtype),
                    result_shape,
                    theta,
                    times,
                    vmap_method='sequential'
                )

            # Add custom VJP for gradients (finite differences)
            @jax.custom_vjp
            def jax_model(theta, times):
                return model_pure(theta, times)

            def jax_model_fwd(theta, times):
                pdf = model_pure(theta, times)
                return pdf, (theta, times)

            def jax_model_bwd(res, g):
                theta, times = res
                n_params = theta.shape[0]
                eps = 1e-7
                theta_bar = []
                for i in range(n_params):
                    theta_plus = theta.at[i].add(eps)
                    theta_minus = theta.at[i].add(-eps)
                    pdf_plus = model_pure(theta_plus, times)
                    pdf_minus = model_pure(theta_minus, times)
                    grad_i = jnp.sum(g * (pdf_plus - pdf_minus) / (2 * eps))
                    theta_bar.append(grad_i)
                return jnp.array(theta_bar), None

            jax_model.defvjp(jax_model_fwd, jax_model_bwd)
            # Source graph reference (see ``pmf_and_moments_from_graph``).
            jax_model._source_graph = graph
            return jax_model

        elif has_param_edges:
            # PARAMETERIZED MODEL: Use FFI for multi-core parallelization
            import json
            from .ffi_wrappers import _make_json_serializable, compute_pmf_ffi
            from .config import get_config

            # Serialize graph structure to JSON (one time)
            # FFI handlers cache GraphBuilder internally (thread-local cache)
            structure_json_str = json.dumps(_make_json_serializable(serialized))

            # Check if FFI is available
            config = get_config()
            use_ffi = config._use_ffi

            if not use_ffi:
                # The pure_callback fallback below has been DISABLED.
                # phasic now requires FFI for parameterised pmf models;
                # the legacy single-core pure_callback path is left in
                # the source as a comment for reference but is no longer
                # reachable. To restore it, uncomment the block below
                # and remove this raise.
                raise PTDBackendError(
                    "pmf_from_graph requires FFI for parameterised graphs. "
                    "FFI is disabled in the current configuration. "
                    "Re-enable with phasic.configure(ffi=True), or rebuild "
                    "phasic with XLA FFI headers available."
                )
            # FFI MODE: Zero-copy XLA-optimized computation with multi-core support
            # FFI handlers cache GraphBuilder in thread-local storage
            from functools import partial

            # Create a partially applied function with static structure_json
            # This prevents vmap from adding a batch dimension to JSON
            model_ffi_partial = partial(
                compute_pmf_ffi,
                structure_json_str,  # Static: not vmapped
                discrete=discrete,   # Static: not vmapped
                granularity=0        # Static: not vmapped
            )

            def model_pure(theta, times):
                """FFI wrapper for multi-core parallelization.

                Supports: jit, vmap, pmap with true multi-core execution
                FFI caching: GraphBuilder cached by JSON structure (no repeated parsing)
                """
                return model_ffi_partial(theta=theta, times=times)

            # ---- DISABLED: legacy pure_callback fallback (no FFI) ----
            # from . import phasic_pybind as cpp_module
            #
            # # Create GraphBuilder ONCE - captured in model closure
            # builder = cpp_module.parameterized.GraphBuilder(structure_json_str)
            #
            # def _compute_pdf_cached(theta_np, times_np):
            #     """Uses cached builder - NO JSON parsing per call."""
            #     # Check if theta is batched (from vmap with expand_dims)
            #     if theta_np.ndim == 2:
            #         times_unbatched = times_np[0] if times_np.ndim == 2 else times_np
            #         results = []
            #         for theta_single in theta_np:
            #             result = builder.compute_pmf(
            #                 theta_single,
            #                 times_unbatched,
            #                 discrete=discrete,
            #                 granularity=0
            #             )
            #             results.append(result)
            #         return np.array(results)
            #     else:
            #         return builder.compute_pmf(
            #             theta_np,
            #             times_np,
            #             discrete=discrete,
            #             granularity=0
            #         )
            #
            # def model_pure(theta, times):
            #     """Pure callback wrapper (fallback when FFI disabled)."""
            #     result_shape = jax.ShapeDtypeStruct(times.shape, times.dtype)
            #     return jax.pure_callback(
            #         lambda t, tm: _compute_pdf_cached(
            #             np.asarray(t, dtype=np.float64),
            #             np.asarray(tm, dtype=np.float64)
            #         ).astype(times.dtype),
            #         result_shape,
            #         theta,
            #         times,
            #         vmap_method='expand_dims'
            #     )
            # ---- end DISABLED ----

            # Add custom VJP for gradients (finite differences)
            @jax.custom_vjp
            def jax_model(theta, times):
                return model_pure(theta, times)

            def jax_model_fwd(theta, times):
                """Forward pass: compute PDF and save inputs for backward."""
                pdf = model_pure(theta, times)
                return pdf, (theta, times)

            def jax_model_bwd(res, g):
                """Backward pass: compute gradients via finite differences."""
                theta, times = res
                n_params = theta.shape[0]
                eps = 1e-7

                # Finite difference gradients
                theta_bar = []
                for i in range(n_params):
                    theta_plus = theta.at[i].add(eps)
                    theta_minus = theta.at[i].add(-eps)

                    pdf_plus = model_pure(theta_plus, times)
                    pdf_minus = model_pure(theta_minus, times)

                    grad_i = jnp.sum(g * (pdf_plus - pdf_minus) / (2 * eps))
                    theta_bar.append(grad_i)

                return jnp.array(theta_bar), None

            jax_model.defvjp(jax_model_fwd, jax_model_bwd)
            # Source graph reference (see ``pmf_and_moments_from_graph``).
            jax_model._source_graph = graph
            return jax_model

        else:
            # NON-PARAMETERIZED MODEL: Use pmf_from_cpp (original flow)
            base_model = cls.pmf_from_cpp(temp_file, discrete=discrete)

            # Wrap to hide theta parameter
            # Return (times) -> pmf for backward compatibility
            def non_param_wrapper(times):
                # Use dummy theta (not used by non-parameterized graphs)
                # Can't use empty array due to JAX pure_callback limitations
                dummy_theta = jnp.array([0.0])
                return base_model(dummy_theta, times)
            # Source graph reference (see ``pmf_and_moments_from_graph``).
            non_param_wrapper._source_graph = graph
            return non_param_wrapper

    @classmethod
    def pmf_from_graph_parameterized(cls, graph_builder: Callable[..., Graph], discrete: bool = False) -> Callable:
        """
        Convert a parameterized Python graph builder to a JAX-compatible function.

        This allows users to define parameterized models where the graph structure
        or edge weights depend on parameters.

        Parameters
        ----------
        graph_builder : callable
            Function (theta) -> Graph that builds a graph with given parameters
        discrete : bool
            If True, uses discrete phase-type distribution (DPH) computation.
            If False, uses continuous phase-type distribution (PDF).

        Returns
        -------
        callable
            JAX-compatible function (theta, times) -> pdf_values that supports JIT, grad, vmap, etc.

        Examples
        --------
        >>> def build_exponential(rate):
        ...     g = Graph(1)
        ...     start = g.starting_vertex()
        ...     v0 = g.find_or_create_vertex([0])
        ...     v1 = g.find_or_create_vertex([1])
        ...     start.add_edge(v0, 1.0)
        ...     v0.add_edge(v1, float(rate))
        ...     return g
        >>>
        >>> model = Graph.pmf_from_graph_parameterized(build_exponential)
        >>> theta = jnp.array([1.5])
        >>> times = jnp.linspace(0, 5, 50)
        >>> pdf = model(theta, times)

        .. note::
           **DISABLED** pending a fix -- see CLAUDE.md "Disabled paths / follow-ups".
        """
        # DISABLED: this builder-based (theta -> Graph) path is unused -- SVGD, the
        # LRT, and pmf_and_moments_from_graph all use the graph-based parameterized
        # API (pmf_from_graph / pmf_and_moments_from_graph) -- and it is broken:
        #   bug 5a: never calls _ensure_jax_active(), so the module-level jax/jnp
        #           are None -> AttributeError at _create_jax_parameterized_wrapper;
        #   bug 5b: hardcodes jnp.float32 where the F64 FFI requires float64;
        #   F-001:  its discrete branch still calls g.normalize() (a native-DPH
        #           normalize that collapses the chain to a deterministic walk --
        #           the same defect fixed as "bug 4" in pmf_from_cpp).
        # See CLAUDE.md "Disabled paths / follow-ups" for the revival checklist. The
        # original implementation is preserved below (now unreachable) for revival.
        raise NotImplementedError(
            "Graph.pmf_from_graph_parameterized is disabled pending a fix "
            "(bugs 5a/5b + the F-001 normalize; see CLAUDE.md 'Disabled paths / "
            "follow-ups'). Use Graph.pmf_from_graph or Graph.pmf_and_moments_from_graph "
            "(the parameterized-graph API) instead."
        )

        # Create wrapper code (both continuous and discrete)
        wrapper_code = '''
#include "phasiccpp.h"
#include <vector>

extern "C" {
    // Continuous mode (PDF)
    void compute_pmf_from_arrays(
        const int* states, int n_vertices, int state_dim,
        const double* edges, int n_edges,
        const double* start_edges, int n_start_edges,
        const double* times, int n_times,
        double* output, int granularity
    ) {
        // Create graph
        phasic::Graph g(state_dim);
        auto start = g.starting_vertex_p();

        // Create vertices
        std::vector<phasic::Vertex*> vertices;
        for (int i = 0; i < n_vertices; i++) {
            std::vector<int> state(state_dim);
            for (int j = 0; j < state_dim; j++) {
                state[j] = states[i * state_dim + j];
            }
            auto v = g.find_or_create_vertex_p(state);
            vertices.push_back(v);
        }

        // Add edges from starting vertex
        for (int i = 0; i < n_start_edges; i++) {
            int to_idx = (int)start_edges[i * 2];
            double weight = start_edges[i * 2 + 1];
            start->add_edge(*vertices[to_idx], weight);
        }

        // Add edges between vertices
        for (int i = 0; i < n_edges; i++) {
            int from_idx = (int)edges[i * 3];
            int to_idx = (int)edges[i * 3 + 1];
            double weight = edges[i * 3 + 2];
            vertices[from_idx]->add_edge(*vertices[to_idx], weight);
        }

        // Compute PDF
        for (int i = 0; i < n_times; i++) {
            output[i] = g.pdf(times[i], granularity);
        }
    }

    // Discrete mode (DPH)
    void compute_dph_pmf_from_arrays(
        const int* states, int n_vertices, int state_dim,
        const double* edges, int n_edges,
        const double* start_edges, int n_start_edges,
        const int* jumps, int n_jumps,
        double* output
    ) {
        // Create graph (same as continuous)
        phasic::Graph g(state_dim);
        auto start = g.starting_vertex_p();

        // Create vertices
        std::vector<phasic::Vertex*> vertices;
        for (int i = 0; i < n_vertices; i++) {
            std::vector<int> state(state_dim);
            for (int j = 0; j < state_dim; j++) {
                state[j] = states[i * state_dim + j];
            }
            auto v = g.find_or_create_vertex_p(state);
            vertices.push_back(v);
        }

        // Add edges from starting vertex
        for (int i = 0; i < n_start_edges; i++) {
            int to_idx = (int)start_edges[i * 2];
            double weight = start_edges[i * 2 + 1];
            start->add_edge(*vertices[to_idx], weight);
        }

        // Add edges between vertices
        for (int i = 0; i < n_edges; i++) {
            int from_idx = (int)edges[i * 3];
            int to_idx = (int)edges[i * 3 + 1];
            double weight = edges[i * 3 + 2];
            vertices[from_idx]->add_edge(*vertices[to_idx], weight);
        }

        // Normalize for discrete mode (required for DPH)
        g.normalize();

        // Compute DPH PMF
        for (int i = 0; i < n_jumps; i++) {
            output[i] = g.dph_pmf(jumps[i]);
        }
    }
}
'''

        # Create hash for the builder function
        import inspect
        builder_source = inspect.getsource(graph_builder) if hasattr(graph_builder, '__code__') else str(graph_builder)
        builder_hash = hashlib.sha256(builder_source.encode()).hexdigest()[:16]

        # Check if already compiled
        cache_key = f"{builder_hash}_discrete_{discrete}"
        if cache_key not in _lib_cache:
            # Compile once
            lib_name = f"param_graph_{builder_hash}"
            lib_path = _compile_wrapper_library(wrapper_code, lib_name)
            # Use PyDLL instead of CDLL to manage GIL automatically
            lib = ctypes.PyDLL(lib_path)
            _setup_ctypes_signatures_from_arrays(lib, discrete=discrete)
            _lib_cache[cache_key] = lib
        else:
            lib = _lib_cache[cache_key]

        # Select appropriate compute function based on mode
        compute_func = lib.compute_dph_pmf_from_arrays if discrete else lib.compute_pmf_from_arrays

        # Create JAX-compatible wrapper using the helper
        return _create_jax_parameterized_wrapper(compute_func, graph_builder, discrete)

    @classmethod
    def pmf_from_cpp(cls, cpp_file: str | pathlib.Path, discrete: bool = False) -> Callable:
        """
        Load a phase-type model from a user's C++ file and return a JAX-compatible function.

        The C++ file should include 'user_model.h' and implement:

        phasic::Graph build_model(const double* theta, int n_params) {
            // Build and return Graph instance
        }

        For efficient repeated evaluations with the same parameters without JAX overhead,
        use load_cpp_builder() instead to get a builder function that creates Graph objects.

        Parameters
        ----------
        cpp_file : str or pathlib.Path
            Path to the user's C++ file
        discrete : bool
            If True, uses discrete phase-type distribution (DPH) computation.
            If False, uses continuous phase-type distribution (PDF).

        Raises
        ------
        ImportError
            If JAX is not installed. Install with: pip install jax jaxlib
        FileNotFoundError
            If the specified C++ file does not exist

        Returns
        -------
        callable
            JAX-compatible function (theta, times) -> pmf_values that supports JIT, grad, vmap, etc.

        Examples
        --------
        JAX-compatible approach (default - for SVGD, gradients, optimization):
        >>> model = Graph.pmf_from_cpp("my_model.cpp")
        >>> theta = jnp.array([1.0, 2.0])
        >>> times = jnp.linspace(0, 10, 100)
        >>> pmf = model(theta, times)
        >>> gradient = jax.grad(lambda p: jnp.sum(model(p, times)))(theta)

        Discrete phase-type distribution:
        >>> model = Graph.pmf_from_cpp("my_model.cpp", discrete=True)
        >>> theta = jnp.array([1.0, 2.0])
        >>> jumps = jnp.array([1, 2, 3, 4, 5])
        >>> dph_pmf = model(theta, jumps)

        For direct C++ access without JAX (faster for repeated evaluations):
        >>> builder = load_cpp_builder("my_model.cpp")
        >>> graph = builder(np.array([1.0, 2.0]))  # Build graph once
        >>> pdf1 = graph.pdf(1.0)  # Use many times
        >>> pdf2 = graph.pdf(2.0)  # No rebuild needed
        """
        cpp_path = pathlib.Path(cpp_file).absolute()
        if not cpp_path.exists():
            raise FileNotFoundError(f"C++ file not found: {cpp_file}")

        # Check if JAX is available
        try:
            _ensure_jax_active()
        except Exception as _e:
            raise ImportError(
                "JAX is required for JAX-compatible C++ models. Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            ) from _e

        # Read user's C++ code
        with open(cpp_path, 'r') as f:
            user_code = f.read()

        # Check that it implements build_model
        if "build_model" not in user_code:
            raise ValueError(
                "C++ file must implement: phasic::Graph build_model(const double* theta, int n_params)"
            )

        # Create wrapper code with both continuous and discrete computation
        wrapper_code = f'''
// Include the C++ API header (which includes the C headers)
#include "phasiccpp.h"

// Include user's model
#include "{cpp_path.absolute()}"

extern "C" {{
    // Wrapper functions that compute PMF/DPH directly
    void compute_pmf(const double* theta, int n_params,
                     const double* times, int n_times,
                     double* output, int granularity) {{
        phasic::Graph g = build_model(theta, n_params);
        for (int i = 0; i < n_times; i++) {{
            output[i] = g.pdf(times[i], granularity);
        }}
    }}

    void compute_dph_pmf(const double* theta, int n_params,
                         const int* jumps, int n_jumps,
                         double* output) {{
        phasic::Graph g = build_model(theta, n_params);
        // NO normalize() here. In a DPH the edge weights ARE the per-step
        // transition probabilities and the deficit (1 - row sum) is the
        // implicit stay-in-place probability. The continuous normalize()
        // rescales every vertex's outgoing weights to sum to 1, which turns
        // the chain into a deterministic walk: on a 2-phase chain it made
        // P(T = 2) = 1 and P(T = n) = 0 otherwise, and — because the rescaling
        // divides theta out — made the gradient identically zero. The FFI
        // handler (ComputePmfFfiImpl) calls dph_pmf directly with no
        // normalisation; match it. A model whose row sums exceed 1 is invalid
        // and the DPH context rejects it, which is the correct behaviour.
        for (int i = 0; i < n_jumps; i++) {{
            output[i] = g.dph_pmf(jumps[i]);
        }}
    }}
}}
'''

        # Compile the library
        source_hash = hashlib.md5(user_code.encode()).hexdigest()[:8]
        lib_name = f"user_model_{cpp_path.stem}_{source_hash}"

        # Check cache first
        cache_key = f"{lib_name}_discrete_{discrete}"
        if cache_key not in _lib_cache:
            lib_path = _compile_wrapper_library(wrapper_code, lib_name)
            # Use PyDLL instead of CDLL to manage GIL automatically
            lib = ctypes.PyDLL(lib_path)
            _setup_ctypes_signatures(lib, has_pmf=True, has_dph=True)
            _lib_cache[cache_key] = lib
        else:
            lib = _lib_cache[cache_key]

        # Select the appropriate compute function
        compute_func = lib.compute_dph_pmf if discrete else lib.compute_pmf

        # Create the Python wrapper function
        def pmf_function(theta, times, granularity=0):
            """Compute PMF using the loaded C++ model"""
            return _compute_pmf_from_ctypes(
                theta,
                times,
                compute_func,
                {},  # Empty graph_data for C++ models with theta
                granularity,
                discrete
            )

        # Helper function for pure callback (used in forward and backward pass)
        def _compute_pmf_pure(theta, times):
            """Pure computation without custom_vjp wrapper"""
            result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)
            return jax.pure_callback(
                lambda t, tm: pmf_function(t, tm, granularity=0).astype(np.float64),
                result_shape,
                theta,
                times,
                vmap_method='expand_dims'
            )

        # Wrap for JAX compatibility with custom VJP for gradients
        @jax.custom_vjp
        def jax_model(theta, times):
            return _compute_pmf_pure(theta, times)

        def jax_model_fwd(theta, times):
            # Call the underlying computation, not jax_model (avoid infinite recursion!)
            pmf = _compute_pmf_pure(theta, times)
            return pmf, (theta, times)

        def jax_model_bwd(res, g):
            theta, times = res
            n_params = theta.shape[0]
            eps = 1e-7

            # Finite differences for gradient
            theta_bar = []
            for i in range(n_params):
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].add(-eps)

                # Call underlying computation, not jax_model
                pmf_plus = _compute_pmf_pure(theta_plus, times)
                pmf_minus = _compute_pmf_pure(theta_minus, times)

                grad_i = jnp.sum(g * (pmf_plus - pmf_minus) / (2 * eps))
                theta_bar.append(grad_i)

            return jnp.array(theta_bar), None

        jax_model.defvjp(jax_model_fwd, jax_model_bwd)
        return jax_model


    def _map_joint_observations_to_indices(self, observed_data, *, seed=None):
        """Map joint-observation outcome tuples to joint-prob-table vertex indices.

        Each observation (a tuple over the non-``prob`` outcome columns) is
        looked up in ``self.joint_prob_table()``. A degenerate outcome that maps
        to more than one index is resolved by sampling proportional to the
        outcomes' probability. Pass ``seed`` for a deterministic tie-break; the
        default ``seed=None`` uses the global ``np.random`` stream (bit-identical
        to the historic inline mapping in ``Graph.svgd``).

        This is the single implementation shared by ``Graph.svgd`` (joint-index /
        daisy-chain paths) and ``Graph.epoch_model``; both feed the returned
        1-D index list to the daisy-chain / joint-index model as
        ``observed_indices``.
        """
        rng = np.random if seed is None else np.random.RandomState(seed)
        joint_prob_table = self.joint_prob_table()
        obs2idx = joint_prob_table.groupby(
            joint_prob_table.columns[:-1].to_list()
        ).groups
        obs_indices = []
        for obs in observed_data:
            idx = obs2idx[tuple(obs)]
            if idx.size > 1:
                # observation maps to multiple indices: sample by probability
                p = joint_prob_table.loc[idx, 'prob'].to_numpy()
                p = p / p.sum()
                chosen_idx = rng.choice(idx, p=p)
                obs_indices.append(chosen_idx.item())
            else:
                # unique index
                obs_indices.append(idx.item())
        return obs_indices


    def _daisy_chain_svgd_model(
        self,
        *,
        observed_indices,
        epoch_starts,
        t_eval: float | None = None,
        user_prior=None,
        user_fixed=None,
        user_tied=None,
        sd: float = 5.0,
        verbose: bool = False,
        granularity: int = 0,
        exposure_arr=None,
        exposure_param_index: int | None = None,
        final_read: str = 'sojourn',
        bake_fd_skip: bool = True,
        exact_final_grad: bool = False,
    ):
        """Build the daisy-chain SVGD model + prior + theta_dim.

        Internal helper for ``Graph.svgd(epoch_starts=...)``. Constructs
        the JSP graph from ``self``, fits a time-homogeneous reference
        prior via ``probability_matching`` on ``self``, broadcasts it
        across ``n_epochs`` epoch slots, and returns a model callable
        with the SVGD ``model(theta, observed_indices, rewards=None)``
        contract.

        Parameters
        ----------
        observed_indices : sequence of int
            Vertex indices in ``self`` (the source joint-prob graph)
            corresponding to each observation. These are the same
            integers the existing ``joint_index`` mode produces; the
            daisy-chain branch translates them to JSP-graph t-state
            positions internally.
        epoch_starts : array-like of float
            Epoch start times. ``epoch_starts[0] == 0``; the last
            entry starts the final epoch which runs to infinity.
            ``n_epochs = len(epoch_starts)``.
        t_eval : float, optional
            Time at which final-epoch joint stop-probabilities are
            read. Forwarded to ``daisy_chain_joint_probs``. Numeric;
            the ``'auto'`` resolution happens upstream in
            ``Graph.svgd``.
        user_prior : callable, optional
            User-supplied prior. If given, used as-is. If None, a
            data-informed prior is built from ``probability_matching``
            on ``self`` and broadcast across epochs.
        user_fixed : list of (local_idx, value) tuples, optional
            Per-parameter fixings, broadcast to the flattened theta
            shape ``(n_epochs * param_length,)``. The ``value`` may be:

            - a scalar — broadcast across every epoch (legacy form);
            - a list/array of length ``n_epochs`` — pinning the
              parameter at a different value in each epoch.

            Multiple entries can mix both forms. The returned
            ``broadcast_fixed`` is a list of
            ``(flat_idx, scalar_value)`` tuples ready for SVGD.
        verbose : bool
            Forwarded to ``probability_matching``.
        granularity : int, optional
            Uniformization granularity forwarded to the FFI handler's
            ``stop_probability`` calls. ``0`` (default) = auto.
        exact_final_grad : bool, default=False
            Batch H (``b3-batchH-plan.md`` v3.1): compute the FINAL
            epoch's theta gradient EXACTLY (the r_v product-rule term +
            the C sojourn adjoint with the conditioning gate skipped, at
            the handoff extracted by pybind replication of the fused FFI
            chain), leaving earlier epochs' slots on the unchanged
            full-chain central-difference FD. Requires
            ``final_read='sojourn'`` and ``weight_mode='linear'``
            (anything else raises ``ValueError`` at construction — no
            silent fallback). Once built, a residual C decline in the
            backward RAISES a diagnostic ``RuntimeError`` (no FD
            fallback on this path). Measured on the de-risk fixture:
            final-epoch gradient components ~3.6e5x more accurate than
            FD, at ~7.4% of the FD backward's cost (net speedup).
            Public via ``Graph.svgd(exact_final_grad=...)`` since
            Batch G.1 (rule R30 scopes it to the epoch leaf).

        Returns
        -------
        (model, theta_dim, prior)
            ``model`` is the SVGD-shaped callable; ``theta_dim`` is
            ``n_epochs * self.param_length()``; ``prior`` is a
            list-of-priors (one entry per flattened theta slot) or
            ``user_prior`` if it was supplied.
        """
        import jax
        import jax.numpy as jnp

        # Validate epoch_starts.
        es = np.asarray(epoch_starts, dtype=np.float64).ravel()
        if es.size == 0:
            raise ValueError("epoch_starts must contain at least one value.")
        if es[0] != 0.0:
            raise ValueError(
                f"epoch_starts[0] must be 0.0, got {float(es[0])}."
            )
        if not np.all(np.diff(es) > 0):
            raise ValueError(
                "epoch_starts must be strictly monotonically increasing."
            )
        n_epochs = int(es.size)
        epoch_dts = list(np.diff(es))  # length n_epochs - 1

        param_length = self.param_length()
        if param_length == 0:
            raise ValueError(
                "Graph has no parameterized edges; cannot run daisy-chain "
                "SVGD."
            )

        # The daisy-chain FFI handler uses continuous-time
        # stop_probability internally, which only works on
        # continuous-time joint-prob graphs. Discrete joint-prob
        # graphs would dispatch to stop_probability_discrete(jumps),
        # which takes integer jumps rather than continuous t. So
        # any epoch_starts value (including [0]) requires
        # discrete=False.
        if getattr(self, 'is_discrete', False):
            raise ValueError(
                "Daisy-chain SVGD (epoch_starts=...) requires a "
                "continuous-time joint-prob graph. Construct the "
                "joint-prob graph with discrete=False, e.g.\n"
                "    graph.joint_prob_graph(indexer, ..., discrete=False)"
            )

        # Build the JSP graph and the initial IPV (in JSP-graph IPV-coords).
        jsp = self.joint_stop_prob_graph()
        n_ipv = len(jsp._ipv_target_indices)
        # Read self's IPV (length self.vertices_length()) and project to
        # the JSP graph's IPV layout. self's IPV has nonzero entries
        # only at vertices the user's callback's @with_ipv covered;
        # those are vertex indices in self that align with vertex
        # indices in the JSP graph (since both graphs visit
        # self.vertices() in the same order during construction). So
        # we extract the entries at the JSP graph's _ipv_target_indices.
        self_ipv_full = np.zeros(self.vertices_length(), dtype=np.float64)
        for edge in self.starting_vertex().edges():
            self_ipv_full[edge.to().index()] = edge.weight()
        initial_ipv = self_ipv_full[jsp._ipv_target_indices]

        # Translate observed_indices (vertex indices in self) to
        # positions in jsp._t_vertex_indices. Since the t-vertex set of
        # the JSP graph is exactly the t-vertex set of self (shared
        # vertex indexing), each observed_index must be a t-vertex in
        # self.
        t_vertex_pos = {
            int(t_idx): pos for pos, t_idx in enumerate(jsp._t_vertex_indices)
        }
        try:
            observed_pos = np.asarray(
                [t_vertex_pos[int(i)] for i in observed_indices],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ValueError(
                f"observed_indices contains vertex {exc.args[0]} which is "
                "not a t-vertex in the JSP graph; observations must "
                "correspond to terminal joint-prob outcomes."
            )

        # Capture daisy-chain kwargs and build the SVGD model. The model
        # contract follows pmf_from_graph_joint_index:
        # model(theta, observed_indices_passed, rewards=None) ->
        #     (per_obs_probs, dummy_moments)
        # The third arg is the SVGD's per-observation index tensor,
        # which we ignore (already captured at construction time).
        observed_pos_jnp = jnp.asarray(observed_pos, dtype=jnp.int32)

        theta_dim = n_epochs * param_length

        # Build the broadcast `fixed` list (per-epoch fixed slots
        # become per-flattened-slot fixings) early so we can pass the
        # fixed indices into daisy_chain_joint_probs to skip FD on
        # those slots in the custom_vjp backward.
        if user_fixed is None:
            broadcast_fixed = None
            fixed_indices = None
        else:
            broadcast_fixed = []
            for entry in user_fixed:
                if not (isinstance(entry, tuple) and len(entry) == 2):
                    raise ValueError(
                        f"fixed entries must be (index, value) tuples; "
                        f"got {entry!r}"
                    )
                local_idx, value = entry
                if not (0 <= local_idx < param_length):
                    raise ValueError(
                        f"fixed index {local_idx} out of range "
                        f"[0, {param_length})"
                    )
                # Two accepted value forms:
                # - scalar: broadcast to every epoch (legacy behaviour);
                # - list/array of length n_epochs: per-epoch values.
                is_scalar = isinstance(value, (int, float, np.integer, np.floating))
                if is_scalar:
                    per_epoch_values = [float(value)] * n_epochs
                else:
                    arr = np.asarray(value, dtype=np.float64).ravel()
                    if arr.size != n_epochs:
                        raise ValueError(
                            f"fixed entry for index {local_idx}: per-epoch "
                            f"value list must have length n_epochs={n_epochs}; "
                            f"got length {arr.size}"
                        )
                    per_epoch_values = arr.tolist()
                for epoch, v in enumerate(per_epoch_values):
                    broadcast_fixed.append(
                        (epoch * param_length + local_idx, float(v))
                    )
            fixed_indices = [idx for idx, _v in broadcast_fixed]

        # Parse user_tied into a flat slave -> master map. By plan
        # convention the first epoch in each entry's list is the
        # master; the rest are slaves whose flat positions are
        # overwritten with the master's value before every forward
        # evaluation. Validation rules R16..R20 (svgd_config.py) have
        # already rejected malformed input, out-of-range indices,
        # duplicates within a group, and overlap with fixed before we
        # get here, so the parsing below assumes well-formed entries.
        #
        # The map is intentionally a plain dict (not a JAX structure):
        # _apply_tying turns it into static jnp.int32 arrays at model-
        # build time so the trace closure captures concrete arrays.
        slave_to_master_flat: dict[int, int] = {}
        if user_tied is not None:
            for entry in user_tied:
                local_idx, epochs = entry
                local_idx = int(local_idx)
                epochs_list = [int(e) for e in epochs]
                master_flat = epochs_list[0] * param_length + local_idx
                for slave_epoch in epochs_list[1:]:
                    slave_flat = slave_epoch * param_length + local_idx
                    slave_to_master_flat[slave_flat] = master_flat
        tied_slave_indices = set(slave_to_master_flat.keys())

        # Tying introspection returned to Graph.svgd so it can:
        #   - copy master->slave columns in the post-SVGD theta_init
        #     consistency step (so the initial particles tensor has
        #     matching values at master and slave positions);
        #   - hand the master/slave map to SVGD.summary so the per-
        #     parameter table can show "Tied->θ[k]" for slave rows.
        # When user_tied is None, slave_to_master is an empty dict.
        tying_info = {
            'slave_to_master': dict(slave_to_master_flat),
        }

        # Build a JAX-traceable helper that replicates the master's
        # value into every slave position. Closes over concrete int32
        # arrays so the trace context sees no leakage; behaves as a
        # no-op when there are no tied slaves.
        if tied_slave_indices:
            _slaves_arr = jnp.asarray(
                list(slave_to_master_flat.keys()), dtype=jnp.int32,
            )
            _masters_arr = jnp.asarray(
                list(slave_to_master_flat.values()), dtype=jnp.int32,
            )

            def _apply_tying(theta_flat):
                return theta_flat.at[_slaves_arr].set(theta_flat[_masters_arr])
        else:
            def _apply_tying(theta_flat):
                return theta_flat

        # Record the truly-fixed flat indices (i.e. user-supplied
        # `fixed=`) BEFORE we extend broadcast_fixed with slave entries
        # below. The FD-backward loops in both _autodiff_bwd and
        # _per_obs_bwd must skip ONLY the truly-fixed positions — slave
        # positions need their FD partials computed so the
        # _apply_tying scatter's VJP can route them back into the
        # master. Mixing the two would zero out the per-slave gradient
        # contributions and underestimate dL/d(master).
        # bake_fd_skip=True (the Graph.svgd default) skips FD partials at the
        # truly-fixed indices for efficiency. A reusable "free" model
        # (Graph.epoch_model) passes bake_fd_skip=False so NO slot is skipped:
        # the FD backward then returns correct gradients for EVERY slot, so the
        # same model object can be fit with an ARBITRARY `fixed` set at the SVGD
        # level (no build-time/fit-time superset constraint). This is
        # bit-identical to bake_fd_skip=True on the non-fixed slots, since SVGD
        # never reads a fixed slot's gradient (it optimises the learnable
        # subspace); it only changes the discarded fixed-slot partials.
        fd_skip_indices = (
            list(fixed_indices)
            if (bake_fd_skip and fixed_indices is not None) else []
        )

        # Extend broadcast_fixed to also lock slave positions. The
        # sentinel value 0.0 is overwritten by _apply_tying inside the
        # model on every forward call, so its concrete value is
        # irrelevant — it only matters that the SVGD-side fixed_mask
        # marks the slave as not-learnable.
        if tied_slave_indices:
            if broadcast_fixed is None:
                broadcast_fixed = []
                fixed_indices = []
            for slave_flat in sorted(tied_slave_indices):
                broadcast_fixed.append((slave_flat, 0.0))
                fixed_indices.append(slave_flat)

        # Two model variants:
        #   (a) no exposure — one daisy-chain FFI call, then per-obs
        #       indexing into the (n_t,) result vector.
        #   (b) exposure + exposure_param_index — per-obs theta_batch
        #       with the exposure_param_index slot broadcast across all
        #       epochs; ONE batched FFI call of shape (n_obs, n_t)
        #       (the C++ handler parallelises over the batch with
        #       OpenMP). No Python-side lax.map fan-out.
        if final_read not in ('stopprob', 'sojourn'):
            raise ValueError(
                f"final_read must be 'stopprob' or 'sojourn', got {final_read!r}."
            )

        # ---- Batch H: exact FINAL-epoch gradient (exact_final_grad) ----
        # Public since Batch G.1 via Graph.svgd(exact_final_grad=...)
        # (R30 scopes it to the epoch leaf).
        # Design of record: b3-batchH-plan.md v3.1; evidence:
        # b3-batchH-findings.md (oracle parity 1.4e-13; composed gradient
        # 3.6e5x more accurate than FD on final-epoch slots at ~7.4% of
        # the FD backward's cost). Loud scope guards -- no silent
        # fallbacks (the C adjoint's contraction is linear-only and the
        # exact block differentiates the sojourn final read).
        if exact_final_grad:
            if final_read != 'sojourn':
                raise ValueError(
                    "exact_final_grad=True requires final_read='sojourn' "
                    f"(got {final_read!r}) -- the exact block "
                    "differentiates the granularity-free sojourn read. "
                    "Drop exact_final_grad for final_read='stopprob'."
                )
            if getattr(self, '_weight_mode', 'linear') != 'linear':
                raise ValueError(
                    "exact_final_grad=True supports weight_mode='linear' "
                    f"only (this graph has weight_mode="
                    f"{self._weight_mode!r}); the C sojourn adjoint's "
                    "contraction is linear-only. Pass "
                    "exact_final_grad=False."
                )
            if getattr(self, 'was_dph', False):
                raise ValueError(
                    "exact_final_grad=True does not support was_dph "
                    "(Graph.discretize()) graphs."
                )

        def _make_exact_final_block(jsp_g, sg_g, ipv_tgt, aux_map, gather_l,
                                    sti_l, init_ipv_np, gran, dts, P_l, n_ep):
            """Batch H I2: build the host-side exact final-epoch block.

            Returns ``(block, n_t)`` where ``block(theta_full_np)`` gives
            the ``(n_t, P)`` Jacobian d(final_jp)/d(theta_final) at the
            handoff extracted from ``theta_full_np``'s earlier epochs --
            the r_v product-rule term included (final_jp = r_v * soj *
            mass with r_v theta-dependent; H0 check (iii), 1.4e-13 vs
            jax.jacobian). Uses PRIVATE clones (never the shared model
            graphs) and per-call attribute lookup on the clone (the I3
            test seam). Handoff extraction = the H0-validated pybind
            replication of the fused FFI handler (tier-1 parity 2.2e-16).
            """
            efg_jsp = jsp_g.clone()
            efg_sg = sg_g.clone()
            ser = sg_g.serialize(theta_dim=P_l)
            # serialize emits edges in ENUMERATION space while sti_l is in
            # vertex-index space; joint_sojourn_graph creates vertices in
            # index order so the two coincide -- ASSERTED, not assumed
            # (G4 fold).
            _vi = np.asarray(ser.get('vertex_indices', []))
            if _vi.size:
                assert np.array_equal(_vi, np.arange(int(ser['n_vertices']))), \
                    "sojourn-graph enumeration != vertex-index space"
            pe = np.asarray(ser['param_edges'], dtype=np.float64)
            if pe.size == 0:
                pe = pe.reshape(0, 2 + P_l)
            ce = np.asarray(ser['edges'], dtype=np.float64).reshape(-1, 3)
            # constant_edges included (G4 fold): empty for
            # joint_sojourn_graph outputs (parameterized edges only), but
            # including them makes r_v/outdeg match the C handler's live
            # edges_length criterion for any graph shape.
            kce = np.asarray(ser.get('constant_edges', []),
                             dtype=np.float64).reshape(-1, 3) \
                if np.asarray(ser.get('constant_edges', [])).size else \
                np.empty((0, 3))
            outdeg = np.zeros(int(ser['n_vertices']))
            for arr in (ce, kce, pe):
                for row in arr:
                    outdeg[int(row[0])] += 1
            absorbing = outdeg == 0
            n_t_l = len(sti_l)
            tpos = {int(ti): kk for kk, ti in enumerate(sti_l)}
            r_const = np.zeros(n_t_l)
            r_coeff = np.zeros((n_t_l, P_l))
            for row in np.concatenate([ce, kce]) if kce.size else ce:
                f_, t_ = int(row[0]), int(row[1])
                if f_ in tpos and absorbing[t_]:
                    r_const[tpos[f_]] += row[2]
            for row in pe:
                f_, t_ = int(row[0]), int(row[1])
                if f_ in tpos and absorbing[t_]:
                    r_coeff[tpos[f_]] += row[2:]
            ipv_tgt_np = np.asarray(ipv_tgt, dtype=np.int64)
            gather_np = np.asarray(gather_l, dtype=np.int64)
            sti_list = [int(x) for x in sti_l]

            def block(theta_full_np):
                th = np.asarray(theta_full_np, dtype=np.float64).ravel()
                ipv_work = np.asarray(init_ipv_np, dtype=np.float64).copy()
                for ep in range(n_ep - 1):
                    efg_jsp.update_ipv(ipv_work)
                    efg_jsp.update_weights(th[ep * P_l:(ep + 1) * P_l].tolist())
                    raw = np.asarray(
                        efg_jsp.stop_probability(dts[ep], granularity=gran)
                    )
                    if not np.all(np.isfinite(raw)):
                        raise RuntimeError(
                            "exact_final_grad: intermediate-epoch "
                            "stop_probability returned non-finite values (a "
                            "swallowed C failure surfaces as NaN on this "
                            "handler family -- master plan 16b item 8). "
                            f"theta={th.tolist()}"
                        )
                    new = np.empty(ipv_tgt_np.size)
                    for kk in range(ipv_tgt_np.size):
                        tgt = int(ipv_tgt_np[kk])
                        p_ = raw[tgt]
                        aux = aux_map.get(tgt)
                        if aux is not None:
                            p_ += raw[aux]
                        new[kk] = p_
                    ipv_work = new
                alpha = ipv_work[gather_np]
                mass = float(alpha.sum())
                theta_final = th[(n_ep - 1) * P_l:]
                if mass == 0.0:
                    # Linear-limit Jacobian: final_jp == 0 identically in
                    # theta_final at zero handoff mass. Production's
                    # FORWARD NaN-fills here (pre-existing, unchanged by
                    # Batch H); this branch returns the limit WITHOUT
                    # calling C, because the C adjoint DECLINES at a zero
                    # IPV and would otherwise trip the raise below.
                    # Subnormal-but-nonzero mass deliberately falls
                    # through to the C call -> decline -> raise
                    # (micro-gate (d) decision, b3-batchH-findings.md).
                    return np.zeros((n_t_l, P_l))
                efg_sg.update_ipv(alpha)
                efg_sg.update_weights(theta_final.tolist())
                soj = np.asarray(efg_sg.expected_sojourn_time(sti_list))
                raw_j = efg_sg._sojourn_grad_theta_subset(
                    sti_list, skip_condition_gate=True)
                if not raw_j:
                    raise RuntimeError(
                        "exact_final_grad: the exact sojourn adjoint "
                        "declined at this theta/handoff WITH the "
                        "conditioning gate skipped. Remaining causes: "
                        "allocation failure; parameterized-tape "
                        "precompute failure; the tape size guard "
                        "(L > 5e7); a non-finite Jacobian row (trap/"
                        "deficit-sink vertex, or subnormal handoff mass); "
                        "an mmap-loaded Stage-A2 tape descriptor without "
                        "input specs (PHASIC_REWARD_COMPUTE_CACHE=1 with "
                        "a warm on-disk cache); or out-of-scope tape "
                        "inputs. No FD fallback exists on this path -- "
                        "pass exact_final_grad=False or investigate. "
                        f"theta_final={theta_final.tolist()}, handoff "
                        f"mass={mass:.3e}."
                    )
                J_soj = np.asarray(raw_j, dtype=np.float64).reshape(n_t_l, P_l)
                r_v = r_const + r_coeff @ theta_final
                return (r_coeff * (soj * mass)[:, None]
                        + r_v[:, None] * J_soj * mass)

            return block, n_t_l

        if exposure_arr is None:
            # No-exposure path: inlined builder that mirrors the exposure
            # branch below MINUS the per-obs scaling. structure_json and
            # initial_ipv_arr are built ONCE here at SVGD-creation time
            # (outside any trace) so the closures captured by _forward,
            # the custom_vmap rule, and the custom_vjp wiring are
            # concrete jax arrays rather than trace-context tracers.
            # See `daisy-chain-fusion-recovery-plan.md` for the full
            # rationale: the previous shape — calling
            # `jsp.daisy_chain_joint_probs(...)` from inside model() —
            # re-defined those closures on every model invocation,
            # which leaked an outer-trace tracer into the inner jaxpr's
            # consts under `vmap(jit(grad(...)))` in some execution
            # contexts (notably ipykernel).
            from .ffi_wrappers import (
                _make_json_serializable,
                compute_daisy_chain_joint_probs_ffi,
                compute_daisy_chain_sojourn_ffi,
            )
            import json as _json_mod_local
            from jax import custom_batching as _cb_local

            theta_dim_local = self.param_length()
            initial_ipv_arr_local = jnp.asarray(initial_ipv, dtype=jnp.float64)
            # (1, n_ipv) form for the vmap rule. The FFI handler
            # broadcasts ipv_batch_size=1 against any theta_batch_size>=1
            # (see graph_builder_ffi.cpp:1261-1266), so this one row
            # serves both single-particle and multi-particle paths.
            initial_ipv_one_local = initial_ipv_arr_local[None, :]
            t_eval_resolved = (
                t_eval if t_eval is not None
                else max(float(sum(epoch_dts)) * 4.0, 10.0)
            )
            structure_local = _make_json_serializable(
                jsp.serialize(theta_dim=theta_dim_local)
            )
            structure_local["_daisy_chain"] = {
                "n_epochs": int(n_epochs),
                "param_length": int(theta_dim_local),
                "t_eval": float(t_eval_resolved),
                "granularity": int(granularity),
                "epoch_dts": [float(x) for x in epoch_dts],
                "ipv_target_indices": [int(x) for x in jsp._ipv_target_indices],
                "t_aux_keys": [int(k) for k in jsp._t_aux_map.keys()],
                "t_aux_values": [int(jsp._t_aux_map[k]) for k in jsp._t_aux_map.keys()],
                "t_vertex_indices": [int(x) for x in jsp._t_vertex_indices],
            }

            sojourn_json_local = None
            if final_read == 'sojourn':
                sg = self.joint_sojourn_graph()
                jsp_states_l = jsp.states()
                jsp_ipv_pos_l = {
                    tuple(int(x) for x in jsp_states_l[v]): k
                    for k, v in enumerate(jsp._ipv_target_indices)
                }
                sg_states_l = sg.states()
                sg_state_to_idx_l = {
                    tuple(int(x) for x in sg_states_l[v]): v
                    for v in range(sg.vertices_length())
                }
                structure_local["_daisy_chain"]["sojourn_jsp_gather"] = [
                    int(jsp_ipv_pos_l[s]) for s in sg._ipv_target_states
                ]
                structure_local["_daisy_chain"]["sojourn_t_indices"] = [
                    int(sg_state_to_idx_l[tuple(int(x) for x in jsp_states_l[t])])
                    for t in jsp._t_vertex_indices
                ]
                sojourn_json_local = _json_mod_local.dumps(
                    _make_json_serializable(sg.serialize(theta_dim=theta_dim_local))
                )

            structure_json_local = _json_mod_local.dumps(structure_local)

            # Batch H: build the exact final-epoch block (guards passed
            # above; final_read == 'sojourn' guaranteed so the sojourn
            # maps exist).
            _efg_block_np = None
            if exact_final_grad:
                _efg_block_np, _efg_n_t = _make_exact_final_block(
                    jsp, sg,
                    [int(x) for x in jsp._ipv_target_indices],
                    {int(k): int(v) for k, v in jsp._t_aux_map.items()},
                    structure_local["_daisy_chain"]["sojourn_jsp_gather"],
                    structure_local["_daisy_chain"]["sojourn_t_indices"],
                    np.asarray(initial_ipv, dtype=np.float64),
                    int(granularity), [float(x) for x in epoch_dts],
                    theta_dim_local, int(n_epochs))
                _efg_final_lo = (n_epochs - 1) * theta_dim_local

            eps_local = 1e-7
            # FD backward skips ONLY truly-fixed positions, not slave
            # positions — see fd_skip_indices computation above.
            fixed_set_local = set(int(i) for i in fd_skip_indices)

            # Forward: wrapped with custom_vmap so that under
            # vmap(grad(loss))(particles), the FD-backward's per-
            # perturbation _forward(tp)/_forward(tm) calls dispatch ONE
            # fused (P, theta_dim) FFI call per perturbation instead of
            # P separate (theta_dim,) calls fanned out by JAX's default
            # expand_dims rule. The structure_json and initial_ipv
            # captures are CONCRETE (built outside any trace at SVGD-
            # creation time), so the rule does NOT leak a tracer into
            # the inner jaxpr's consts the way the previous
            # in-`daisy_chain_joint_probs` version did.
            def _dc_call(theta_flat, ipv_arr):
                # Dispatch the daisy FFI by final-epoch read mode. Closures over
                # final_read / structure_json_local / sojourn_json_local are
                # concrete at SVGD-creation time.
                if final_read == 'sojourn':
                    return compute_daisy_chain_sojourn_ffi(
                        structure_json_local, sojourn_json_local, theta_flat, ipv_arr,
                    )
                return compute_daisy_chain_joint_probs_ffi(
                    structure_json_local, theta_flat, ipv_arr,
                )

            @_cb_local.custom_vmap
            def _forward(theta_flat: jnp.ndarray) -> jnp.ndarray:
                return _dc_call(theta_flat, initial_ipv_arr_local)

            @_forward.def_vmap
            def _forward_vmap_rule(axis_size, in_batched, theta_flat):
                # theta_flat: (axis_size, n_epochs * param_length).
                # Dispatch as one fat 2D FFI call.
                del axis_size, in_batched
                return (
                    _dc_call(theta_flat, initial_ipv_one_local),  # (1, n_ipv)
                    True,  # batched along axis 0
                )

            # custom_vjp: FD backward, skipping fixed indices.
            @jax.custom_vjp
            def _autodiff(theta_flat):
                return _forward(theta_flat)

            def _autodiff_fwd(theta_flat):
                return _forward(theta_flat), theta_flat

            def _autodiff_bwd(theta_flat, cotangent):
                n_params = theta_flat.shape[0]
                grads = []
                for i in range(n_params):
                    if i in fixed_set_local:
                        grads.append(jnp.asarray(0.0, dtype=theta_flat.dtype))
                        continue
                    tp = theta_flat.at[i].add(eps_local)
                    tm = theta_flat.at[i].add(-eps_local)
                    jp = _forward(tp)
                    jm = _forward(tm)
                    grads.append(
                        jnp.sum(cotangent * (jp - jm) / (2.0 * eps_local))
                    )
                return (jnp.stack(grads),)

            if _efg_block_np is not None:
                def _autodiff_bwd(theta_flat, cotangent):  # noqa: F811
                    """Batch H override (exact_final_grad=True): exact
                    final-epoch theta slots via the host-callback block;
                    earlier slots keep the UNCHANGED full-chain central-
                    difference FD. Static Python dispatch -- the False
                    path uses the original definition above, untouched.
                    Slot precedence (plan v3.1): fixed wins (0.0, the
                    fixed-slot contract); tied slaves get exact values
                    (the _apply_tying scatter-VJP sums slave->master).
                    """
                    n_params = theta_flat.shape[0]
                    J_final = jax.pure_callback(
                        _efg_block_np,
                        jax.ShapeDtypeStruct(
                            (_efg_n_t, theta_dim_local), jnp.float64),
                        theta_flat,
                        vmap_method='sequential',
                    )
                    exact_final = cotangent @ J_final  # (P,)
                    grads = []
                    for i in range(n_params):
                        if i in fixed_set_local:
                            grads.append(
                                jnp.asarray(0.0, dtype=theta_flat.dtype))
                            continue
                        if i >= _efg_final_lo:
                            grads.append(exact_final[i - _efg_final_lo])
                            continue
                        tp = theta_flat.at[i].add(eps_local)
                        tm = theta_flat.at[i].add(-eps_local)
                        jp = _forward(tp)
                        jm = _forward(tm)
                        grads.append(
                            jnp.sum(cotangent * (jp - jm) / (2.0 * eps_local))
                        )
                    return (jnp.stack(grads),)

            _autodiff.defvjp(_autodiff_fwd, _autodiff_bwd)

            def model(theta, _observed_arg=None, rewards=None):
                theta_arr = jnp.atleast_1d(theta).reshape(-1)
                # Replicate master values into slave positions BEFORE
                # the FFI sees theta. Cotangents at slave positions
                # are routed back to the master via the scatter's
                # standard VJP, so the FD bwd's per-slave partials
                # add into dL/d(master) automatically.
                theta_arr = _apply_tying(theta_arr)
                joint_probs = _autodiff(theta_arr)
                per_obs = joint_probs[observed_pos_jnp]
                return per_obs, jnp.zeros(2)

            # Tag the model so SVGD.__init__ keeps parallel_mode='vmap'.
            # _handles_exposure_internally is a slight misnomer on this
            # no-exposure model but is harmless: SVGD's exposure-wrapper
            # branch is gated on `self.exposure is not None`, so a no-
            # exposure model is never wrapped regardless of this tag.
            # We set both flags to reuse the existing dispatch in
            # `svgd.py:4994-5004` without touching SVGD.
            model._handles_exposure_internally = True
            model._handles_particle_vmap = True
        else:
            # Validate per-obs inputs.
            if exposure_param_index is None:
                raise ValueError(
                    "_daisy_chain_svgd_model: exposure_param_index must "
                    "be set when exposure_arr is set."
                )
            alpha_arr = jnp.asarray(exposure_arr, dtype=jnp.float64)
            if alpha_arr.ndim != 1:
                raise ValueError(
                    f"exposure_arr must be 1D, got shape {alpha_arr.shape}."
                )
            n_obs = int(alpha_arr.shape[0])
            n_obs_observed = int(observed_pos_jnp.shape[0])
            if n_obs != n_obs_observed:
                raise ValueError(
                    f"exposure_arr length ({n_obs}) does not match the "
                    f"number of observations ({n_obs_observed})."
                )

            # Auto-dedup of identical exposure values: identical alpha_i
            # produces an identical theta row, which the FFI handler
            # would otherwise compute redundantly. We dedup once at
            # model-build time so every subsequent forward call (and
            # every FD-backward perturbation) runs only the unique
            # rows; results are scattered back per-obs via inverse_idx.
            # This is bit-exact: same alpha → identical FFI output.
            # Users can amplify the benefit by pre-rounding their
            # exposures (e.g. np.round(tree_spans, -3)) before calling
            # svgd; n_obs=312 with K=30 unique rounded values runs ~10×
            # fewer chains.
            _alpha_np = np.asarray(alpha_arr)
            _unique_alphas_np, _inverse_idx_np = np.unique(
                _alpha_np, return_inverse=True
            )
            unique_alpha_arr = jnp.asarray(_unique_alphas_np, dtype=jnp.float64)
            inverse_idx_jnp = jnp.asarray(_inverse_idx_np, dtype=jnp.int32)
            n_unique = int(unique_alpha_arr.shape[0])

            # Flat-theta indices that should be scaled per observation:
            # one per epoch, all pointing to the local
            # exposure_param_index slot.
            flat_exposure_indices = jnp.asarray(
                [epoch * param_length + exposure_param_index
                 for epoch in range(n_epochs)],
                dtype=jnp.int32,
            )

            # Build the structure_json + initial_ipv_batched once outside
            # the model so the batched FFI call sees them as static
            # closures. Mirrors the structure-build path inside
            # daisy_chain_joint_probs.
            from .ffi_wrappers import (
                _make_json_serializable,
                compute_daisy_chain_joint_probs_ffi,
                compute_daisy_chain_sojourn_ffi,
            )
            import json as _json_mod_local
            theta_dim_local = self.param_length()  # = param_length here
            n_ipv_local = len(jsp._ipv_target_indices)
            initial_ipv_arr_local = jnp.asarray(initial_ipv, dtype=jnp.float64)
            # Initial IPV with leading batch axis = 1. The FFI handler
            # at graph_builder_ffi.cpp:1261-1266 broadcasts
            # ipv_batch_size=1 against any theta_batch_size>=1, so this
            # single row serves both the single-particle (B=1) and the
            # vmapped multi-particle (B=P*K) paths.
            initial_ipv_one = initial_ipv_arr_local[None, :]
            t_eval_resolved = (
                t_eval if t_eval is not None
                else max(float(sum(epoch_dts)) * 4.0, 10.0)
            )
            structure_local = _make_json_serializable(
                jsp.serialize(theta_dim=theta_dim_local)
            )
            structure_local["_daisy_chain"] = {
                "n_epochs": int(n_epochs),
                "param_length": int(theta_dim_local),
                "t_eval": float(t_eval_resolved),
                "granularity": int(granularity),
                "epoch_dts": [float(x) for x in epoch_dts],
                "ipv_target_indices": [int(x) for x in jsp._ipv_target_indices],
                "t_aux_keys": [int(k) for k in jsp._t_aux_map.keys()],
                "t_aux_values": [int(jsp._t_aux_map[k]) for k in jsp._t_aux_map.keys()],
                "t_vertex_indices": [int(x) for x in jsp._t_vertex_indices],
            }

            sojourn_json_local = None
            if final_read == 'sojourn':
                sg = self.joint_sojourn_graph()
                jsp_states_l = jsp.states()
                jsp_ipv_pos_l = {
                    tuple(int(x) for x in jsp_states_l[v]): k
                    for k, v in enumerate(jsp._ipv_target_indices)
                }
                sg_states_l = sg.states()
                sg_state_to_idx_l = {
                    tuple(int(x) for x in sg_states_l[v]): v
                    for v in range(sg.vertices_length())
                }
                structure_local["_daisy_chain"]["sojourn_jsp_gather"] = [
                    int(jsp_ipv_pos_l[s]) for s in sg._ipv_target_states
                ]
                structure_local["_daisy_chain"]["sojourn_t_indices"] = [
                    int(sg_state_to_idx_l[tuple(int(x) for x in jsp_states_l[t])])
                    for t in jsp._t_vertex_indices
                ]
                sojourn_json_local = _json_mod_local.dumps(
                    _make_json_serializable(sg.serialize(theta_dim=theta_dim_local))
                )

            structure_json_local = _json_mod_local.dumps(structure_local)

            # Per-obs forward + custom_vjp (FD) wrapping a SINGLE batched
            # FFI call. eps matches the legacy daisy_chain_joint_probs
            # FD pattern.
            eps_local = 1e-7
            # FD backward skips ONLY truly-fixed positions, not slave
            # positions — see fd_skip_indices computation above for why.
            fixed_set_local = set(int(i) for i in fd_skip_indices)

            # Precomputed scale matrix: shape (n_unique, theta_dim), 1.0
            # everywhere except in the flat_exposure_indices columns,
            # which hold the per-unique alpha values. Lifted out of the
            # forward function so it's a JIT-time constant (all inputs
            # are concrete at model-build time).
            scale_per_unique = (
                jnp.ones(
                    (n_unique, n_epochs * param_length), dtype=jnp.float64
                )
                .at[:, flat_exposure_indices]
                .multiply(unique_alpha_arr[:, None])
            )

            # Batch H: exact final-epoch block, exposure variant -- one
            # (n_t, P) block PER UNIQUE exposure value, computed at the
            # correspondingly-scaled theta, with the chain rule applied
            # SLOT-SPECIFICALLY: element-wise by
            # scale_per_unique[u, final_slice], which is 1.0 everywhere
            # except the exposure_param_index column (a blanket
            # all-columns scaling would corrupt the non-exposure final
            # slots by a factor alpha_u -- plan v3.1, C-review MAJOR 2).
            _efg_block_exp_np = None
            if exact_final_grad:
                _efg_block_core, _efg_n_t = _make_exact_final_block(
                    jsp, sg,
                    [int(x) for x in jsp._ipv_target_indices],
                    {int(k): int(v) for k, v in jsp._t_aux_map.items()},
                    structure_local["_daisy_chain"]["sojourn_jsp_gather"],
                    structure_local["_daisy_chain"]["sojourn_t_indices"],
                    np.asarray(initial_ipv, dtype=np.float64),
                    int(granularity), [float(x) for x in epoch_dts],
                    theta_dim_local, int(n_epochs))
                _efg_final_lo = (n_epochs - 1) * theta_dim_local
                _efg_scale_np = np.asarray(scale_per_unique, dtype=np.float64)

                def _efg_block_exp_np(theta_flat_np):
                    th = np.asarray(theta_flat_np, dtype=np.float64).ravel()
                    out = np.empty(
                        (n_unique, _efg_n_t, theta_dim_local))
                    for u in range(n_unique):
                        Ju = _efg_block_core(th * _efg_scale_np[u])
                        out[u] = Ju * _efg_scale_np[u, _efg_final_lo:][None, :]
                    return out

            from jax import custom_batching as _cb_local

            # Core per-particle forward, wrapped with custom_vmap so that
            # under ANY vmap composition (vmap(f), vmap(grad(f)), etc.)
            # the batched call is intercepted by our rule and dispatched
            # as a single fat FFI call of shape (P*n_unique, theta_dim).
            # Without this rule, the FFI call inside the body would get
            # auto-batched by JAX's default expand_dims rule, producing
            # a 3D theta buffer that the C++ handler rejects.
            def _dc_call(theta_b, ipv_b):
                if final_read == 'sojourn':
                    return compute_daisy_chain_sojourn_ffi(
                        structure_json_local, sojourn_json_local, theta_b, ipv_b,
                    )
                return compute_daisy_chain_joint_probs_ffi(
                    structure_json_local, theta_b, ipv_b,
                )

            @_cb_local.custom_vmap
            def _per_obs_core(theta_flat):
                # 1D input path: theta_flat shape (theta_dim,). Build a
                # (n_unique, theta_dim) batch and call FFI once.
                theta_pk = theta_flat[None, :] * scale_per_unique
                joint = _dc_call(theta_pk, initial_ipv_one)  # (n_unique, n_t)
                # Scatter to per-obs positions via inverse_idx_jnp and
                # pick each obs's own t-vertex. Two paired integer
                # arrays of equal length -> (n_obs,).
                return joint[inverse_idx_jnp, observed_pos_jnp]

            @_per_obs_core.def_vmap
            def _per_obs_core_vmap_rule(axis_size, in_batched, theta_flat):
                # theta_flat has been lifted to (axis_size, theta_dim)
                # by vmap. Fuse the leading axis with our internal
                # n_unique batch into one fat FFI call.
                del in_batched
                P = axis_size
                # (P, 1, theta_dim) * (1, n_unique, theta_dim) ->
                # (P, n_unique, theta_dim), reshape to (P*K, theta_dim).
                theta_pk = (
                    theta_flat[:, None, :] * scale_per_unique[None, :, :]
                ).reshape(P * n_unique, n_epochs * param_length)
                joint = _dc_call(theta_pk, initial_ipv_one)  # (P*n_unique, n_t)
                joint = joint.reshape(P, n_unique, -1)  # (P, n_unique, n_t)
                per_obs_2d = joint[:, inverse_idx_jnp, observed_pos_jnp]
                # out is batched along axis 0.
                return per_obs_2d, True

            # custom_vjp wraps _per_obs_core to provide the FD backward.
            # The custom_vmap rule on _per_obs_core ensures that even
            # the bwd's internal calls to _per_obs_core (which would
            # otherwise be auto-batched by vmap) go through our explicit
            # rule, producing a 2D FFI call instead of a rejected 3D one.
            @jax.custom_vjp
            def _per_obs_autodiff(theta_flat):
                return _per_obs_core(theta_flat)

            def _per_obs_fwd(theta_flat):
                return _per_obs_core(theta_flat), theta_flat

            def _per_obs_bwd(theta_flat, cotangent):
                """Central-difference VJP.

                theta_flat is 1D (theta_dim,) at the autodiff boundary
                — vmap composes with custom_vjp by tracing this bwd
                with a hidden batch axis. Operations inside the body
                are vmap'd transparently, but our _per_obs_core's
                custom_vmap rule intercepts the FFI call so the batch
                stays 2D, never 3D.
                """
                n_params = theta_flat.shape[0]
                grads = []
                for i in range(n_params):
                    if i in fixed_set_local:
                        grads.append(jnp.asarray(0.0, dtype=theta_flat.dtype))
                        continue
                    tp = theta_flat.at[i].add(eps_local)
                    tm = theta_flat.at[i].add(-eps_local)
                    jp = _per_obs_core(tp)
                    jm = _per_obs_core(tm)
                    grads.append(
                        jnp.sum(cotangent * (jp - jm) / (2.0 * eps_local))
                    )
                return (jnp.stack(grads),)

            if _efg_block_exp_np is not None:
                def _per_obs_bwd(theta_flat, cotangent):  # noqa: F811
                    """Batch H override (exact_final_grad=True), exposure
                    variant: exact final-epoch slots via per-unique
                    blocks gathered to per-obs rows; earlier slots keep
                    the UNCHANGED full-chain FD. Same slot precedence as
                    the no-exposure override (fixed wins; tied slaves
                    exact)."""
                    n_params = theta_flat.shape[0]
                    Jb = jax.pure_callback(
                        _efg_block_exp_np,
                        jax.ShapeDtypeStruct(
                            (n_unique, _efg_n_t, theta_dim_local),
                            jnp.float64),
                        theta_flat,
                        vmap_method='sequential',
                    )
                    per_obs_J = Jb[inverse_idx_jnp, observed_pos_jnp, :]
                    exact_final = cotangent @ per_obs_J  # (P,)
                    grads = []
                    for i in range(n_params):
                        if i in fixed_set_local:
                            grads.append(
                                jnp.asarray(0.0, dtype=theta_flat.dtype))
                            continue
                        if i >= _efg_final_lo:
                            grads.append(exact_final[i - _efg_final_lo])
                            continue
                        tp = theta_flat.at[i].add(eps_local)
                        tm = theta_flat.at[i].add(-eps_local)
                        jp = _per_obs_core(tp)
                        jm = _per_obs_core(tm)
                        grads.append(
                            jnp.sum(cotangent * (jp - jm) / (2.0 * eps_local))
                        )
                    return (jnp.stack(grads),)

            _per_obs_autodiff.defvjp(_per_obs_fwd, _per_obs_bwd)

            def model(theta, _observed_arg=None, rewards=None):
                theta_arr = jnp.atleast_1d(theta)
                # Replicate master values into slave positions BEFORE
                # the per-obs scaling + FFI sees theta. (Same fix as
                # the no-exposure branch; see that wrapper for the
                # gradient-routing rationale.) When no ties are
                # active, _apply_tying is the identity.
                theta_arr = _apply_tying(theta_arr)
                per_obs = _per_obs_autodiff(theta_arr)
                return per_obs, jnp.zeros(2)

            # Tags the model so SVGD.__init__ knows:
            #   _handles_exposure_internally: do NOT apply the outer
            #     _wrap_model_with_exposure wrapper (the per-obs scaling
            #     is already inside the FFI).
            #   _handles_particle_vmap: do NOT force parallel_mode='none';
            #     the custom_vmap rule above batches particles natively
            #     so vmap-over-particles fuses with the internal batch.
            model._handles_exposure_internally = True
            model._handles_particle_vmap = True

        # broadcast_fixed already built above (alongside fixed_indices)
        # so we could pass fixed_indices into the model's custom_vjp.

        # Build the prior. If user supplied one, broadcast it across
        # epochs (one slot per param_length entry per epoch). A list-
        # of-priors of length param_length is also broadcast slot-for-
        # slot. A list of length theta_dim is taken as-is.
        from .svgd import GaussPrior, Prior

        # Helper: zero out per-parameter prior entries for fixed params.
        # SVGD requires that prior_list[i] is None at fixed indices.
        fixed_indices_set = (
            set(idx for idx, _v in broadcast_fixed)
            if broadcast_fixed is not None else set()
        )

        def _mask_fixed(prior_list):
            return [
                None if i in fixed_indices_set else p
                for i, p in enumerate(prior_list)
            ]

        # Attach the tying info to the model so callers (Graph.svgd,
        # SVGD.summary) can introspect it without changing the public
        # return-tuple shape. The attribute is always set, including
        # when no tying is active (slave_to_master is empty).
        model._tying_info = tying_info
        # Preconditioner source: the daisy-chain model returns DUMMY zeros as its
        # second (moments) output, so precondition on the FIRST (probability)
        # output (ProbabilityJacobianPreconditioner). Set for BOTH the exposure
        # and no-exposure model objects (this line runs for both branches).
        model._precondition_output = 'probability'

        if user_prior is not None:
            # Determine the broadcast pattern.
            if isinstance(user_prior, list):
                if len(user_prior) == theta_dim:
                    return (
                        model,
                        theta_dim,
                        _mask_fixed(user_prior),
                        broadcast_fixed,
                    )
                if len(user_prior) == param_length:
                    out = []
                    for _epoch in range(n_epochs):
                        out.extend(user_prior)
                    return (
                        model,
                        theta_dim,
                        _mask_fixed(out),
                        broadcast_fixed,
                    )
                raise ValueError(
                    f"user prior list length {len(user_prior)} must equal "
                    f"either param_length={param_length} or "
                    f"theta_dim={theta_dim}."
                )
            # Single callable: broadcast to every per-parameter slot.
            return (
                model,
                theta_dim,
                _mask_fixed([user_prior] * theta_dim),
                broadcast_fixed,
            )

        # Otherwise: run probability_matching on self for a time-
        # homogeneous reference, then broadcast across epochs.
        from .probability_matching import probability_matching

        try:
            pm_result = probability_matching(
                self, observed_indices,
                std_multiplier=sd, verbose=verbose,
            )
            theta_hat = np.asarray(pm_result.theta, dtype=np.float64)
            theta_std = np.asarray(pm_result.std, dtype=np.float64)
        except Exception:
            # Fall through to None — SVGD will use its standard-normal
            # default prior.
            return model, theta_dim, None, broadcast_fixed

        # Per-parameter Gaussian priors, broadcast n_epochs times.
        # GaussPrior takes ci=[lo, hi]; we pass mean ± sd*std as the
        # 95% interval — same convention DataPrior uses.
        broadcast_priors = []
        for _epoch in range(n_epochs):
            for k in range(param_length):
                m = float(theta_hat[k])
                s = float(theta_std[k]) if k < theta_std.size else 1.0
                lo = m - sd * s
                hi = m + sd * s
                broadcast_priors.append(GaussPrior(ci=[lo, hi]))

        return (
            model,
            theta_dim,
            _mask_fixed(broadcast_priors),
            broadcast_fixed,
        )


    def epoch_model(
        self,
        observed_data,
        epoch_starts,
        *,
        prior=None,
        fixed=None,
        exposure=None,
        exposure_param_index: int | None = None,
        daisy_chain_t_eval: float | str | None = None,
        daisy_chain_probe_theta=None,
        daisy_chain_t_eval_tol: float = 1e-3,
        daisy_chain_granularity: int = 0,
        final_read: str = 'sojourn',
        seed: int | None = None,
        verbose: bool = False,
    ) -> "FreeEpochModel":
        """Build a reusable **free** (untied) daisy-chain epoch model.

        This is the epoch/time-inhomogeneous analogue of the public
        :meth:`pmf_and_moments_from_graph`: it returns a *reusable* model object
        instead of fitting immediately (as ``Graph.svgd(epoch_starts=...)``
        does). Build the (expensive) JSP graph + model **once**, then fit it many
        times via :meth:`FreeEpochModel.fit`. Because two fits from the same
        object share ``.model``,
        :func:`phasic.model_selection.likelihood_ratio_test(full, nested)` takes
        the fast same-model path — a free-vs-fixed nested LRT on an epoch model.
        (A *tied*-vs-free LRT instead uses ``Graph.svgd(tied=...)`` +
        ``likelihood_ratio_test`` directly; tying is out of scope here.)

        ``self`` must be a **continuous-time joint-probability graph**
        (``graph.joint_prob_graph(indexer, ..., discrete=False)``).

        Parameters
        ----------
        observed_data : sequence
            Joint observations (outcome tuples over the joint-prob table's
            non-``prob`` columns). Mapped to vertex indices internally.
        epoch_starts : array-like of float
            ``epoch_starts[0] == 0``; the rest are additional epoch start times.
            ``n_epochs = len(epoch_starts)``; the flat theta has length
            ``n_epochs * param_length``.
        prior, fixed : optional
            Per-epoch (local index) prior / fixings, broadcast across epochs —
            same semantics as ``Graph.svgd``'s ``prior``/``fixed`` under
            ``epoch_starts``. These are baked as the model's *base* restriction;
            :meth:`FreeEpochModel.fit` can add MORE (in flat index space).
        exposure, exposure_param_index : optional
            Per-observation exposure scaling; baked into the model (do not
            re-pass to ``fit``).
        daisy_chain_t_eval, daisy_chain_probe_theta, daisy_chain_t_eval_tol, daisy_chain_granularity, final_read
            As in :meth:`svgd` (the ``'auto'`` probe, granularity, sojourn read).
        seed : int, optional
            Seed for the observation-mapping tie-break (degenerate outcome
            groups). ``None`` (default) uses the global ``np.random`` stream, as
            ``Graph.svgd`` does. Pass a seed for reproducible mapping.

        Returns
        -------
        FreeEpochModel
            Carries ``model``, ``theta_dim``, ``observed_data`` (mapped indices),
            ``prior``, ``fixed`` (base), epoch metadata, and ``.fit(...)``.

        Examples
        --------
        >>> m = jpg.epoch_model(obs, epoch_starts=[0, 0.5], fixed=[(1, mu)],
        ...                     prior=LogGaussPrior(ci=[1/50_000, 1/5000]))
        >>> full   = m.fit(n_iterations=500)
        >>> nested = m.fit(fixed=[(0, 1e-4)], n_iterations=500)  # flat index
        >>> import phasic.model_selection as ms
        >>> ms.likelihood_ratio_test(full, nested)   # same-model fast path
        """
        _ensure_jax_active()
        if not self._joint_prob_base_graph_indexer:
            raise ValueError(
                "Graph.epoch_model requires a joint-probability graph. Construct "
                "one with graph.joint_prob_graph(indexer, ..., discrete=False)."
            )
        # 1. observations -> joint-prob vertex indices (shared mapping helper).
        mapped = self._map_joint_observations_to_indices(observed_data, seed=seed)
        # 2. resolve t_eval (numeric / None default / 'auto' probe).
        resolved_t_eval = self._resolve_daisy_chain_t_eval(
            daisy_chain_t_eval=daisy_chain_t_eval,
            epoch_starts=epoch_starts,
            probe_theta=daisy_chain_probe_theta,
            tol=daisy_chain_t_eval_tol,
            granularity=daisy_chain_granularity,
            verbose=verbose,
        )
        # 3. per-observation exposure array (scalar broadcasts to len(mapped)).
        _daisy_exposure = (
            np.asarray(exposure, dtype=np.float64).ravel()
            if exposure is not None else None
        )
        if _daisy_exposure is not None and _daisy_exposure.size == 1:
            _daisy_exposure = np.full(
                (len(mapped),), float(_daisy_exposure.item()), dtype=np.float64
            )
        # 4. build the FREE model (user_tied=None) with NO baked FD-skip, so ANY
        #    `fixed` set may be applied per fit at the SVGD level.
        model, theta_dim, prior_out, fixed_out = self._daisy_chain_svgd_model(
            observed_indices=mapped,
            epoch_starts=epoch_starts,
            t_eval=resolved_t_eval,
            user_prior=prior,
            user_fixed=fixed,
            user_tied=None,
            sd=5.0,
            verbose=verbose,
            granularity=daisy_chain_granularity,
            exposure_arr=_daisy_exposure,
            exposure_param_index=exposure_param_index,
            final_read=final_read,
            bake_fd_skip=False,
        )
        from .epoch_model import FreeEpochModel
        es = np.asarray(epoch_starts, dtype=np.float64).ravel()
        return FreeEpochModel(
            model=model,
            theta_dim=theta_dim,
            observed_data=mapped,
            prior=prior_out,
            fixed=fixed_out,
            n_epochs=int(es.size),
            param_length=self.param_length(),
            t_eval=resolved_t_eval,
            epoch_starts=es.tolist(),
        )


    def _resolve_inference_theta_dim(self, theta_dim, theta_init, *,
                                     callback=None, weight_formula=None,
                                     epoch_starts=None):
        """Resolve/validate theta_dim by weight mode for inference entry points
        (``svgd``, ``method_of_moments``), so the parameter dimension is always
        either explicitly given or RELIABLY inferred — never silently taken to be
        the coefficient length when that is wrong.

        - **callback** mode (``callback=`` kwarg OR the ``weight_callback``
          property): a black box; require explicit ``theta_dim``/``theta_init``
          (raises otherwise) — the callback may treat only some coefficient slots
          as parameters, so it cannot be introspected.
        - **formula** mode (``weight_formula=`` kwarg OR the ``weight_formula``
          property): inferred from the formula's ``n_theta`` (highest t-index + 1)
          when not given; an explicit ``theta_dim < n_theta`` raises.
        - **linear/log**: returned unchanged (``None`` → the caller infers
          ``param_length()``).

        Under ``epoch_starts`` the formula inference is skipped (the daisy chain
        resolves its own flat theta dimension; on a joint-prob graph
        ``param_length`` already equals ``n_theta``).
        """
        from .exceptions import SvgdConfigError
        _callback_mode = (callback is not None) or (self._weight_mode == 'callback')
        if _callback_mode and theta_dim is None and theta_init is None:
            raise SvgdConfigError(
                "callback weight mode requires an explicit theta_dim= or "
                "theta_init= argument. In callback mode the edge coefficient "
                "vector may carry auxiliary data, so theta_dim cannot be "
                "inferred from the graph without ambiguity. Pass theta_dim=<n> "
                "(or theta_init of shape (n_particles, n)) to make the "
                "parameter dimension explicit."
            )
        _fn_theta = None
        if weight_formula is not None:
            from .weight_formula import compile_formula
            _fn_theta = int(compile_formula(weight_formula).get('n_theta', 0))
        elif (self._weight_mode == 'formula'
              and self._weight_formula_tape is not None):
            _fn_theta = int(self._weight_formula_tape.get('n_theta', 0))
        if _fn_theta:
            if theta_dim is not None and theta_dim < _fn_theta:
                raise SvgdConfigError(
                    f"weight_formula uses parameters up to t{_fn_theta - 1} "
                    f"(n_theta={_fn_theta}), but theta_dim={theta_dim} is "
                    f"smaller. Increase theta_dim to at least {_fn_theta}, or "
                    f"reference fewer t-indices in the formula."
                )
            if (theta_dim is None and theta_init is None
                    and epoch_starts is None):
                theta_dim = _fn_theta
        return theta_dim


    def svgd(self,
             observed_data: ArrayLike | SparseObservations,
             discrete: bool | None = None,
             prior: Callable | None = None,
             n_particles: int | None = None,
             n_iterations: int = 1000,
             optimizer: object | None = None,
             learning_rate: float | None = None,
             bandwidth: str | float | ArrayLike = 'median_per_dim',
             theta_init: ArrayLike | None = None,
             theta_dim: int | None = None,
             return_history: bool = True,
             seed: int | None = None,
             verbose: bool = False,
             progress: bool = True,
             jit: bool | None = None,
             parallel: str | None = None,
             n_devices: int | None = None,
             precompile: bool = True,
             compilation_config: object | None = None,
             regularization: float = 0.0,
             nr_moments: int = 2,
             positive_params: bool = True,
             param_transform: Callable | None = None,
             joint_index: bool | None = None,
             rewards: ArrayLike | None = None,
             fixed: ArrayLike | None = None,
             tied: ArrayLike | None = None,
             callback: Callable | None = None,
             weight_formula: str | None = None,
             preconditioner: str | object | None = None,
             epoch_starts: ArrayLike | None = None,
             daisy_chain_t_eval: float | str | None = None,
             daisy_chain_granularity: int = 0,
             daisy_chain_probe_theta: ArrayLike | None = None,
             daisy_chain_t_eval_tol: float = 1e-3,
             final_read: str = 'sojourn',
             exposure: ArrayLike | float | None = None,
             exposure_param_index: int | None = None,
             validate_rewards: bool = True,
             quiet_assumptions: bool = False,
             exact_moment_grad: bool | None = None,
             exact_final_grad: bool | None = None,
             exact_grad: bool | None = None,
             ) -> 'SVGD':
        """
        Run Stein Variational Gradient Descent (SVGD) inference for Bayesian parameter estimation.

        SVGD finds the posterior distribution p(theta | data) by optimizing a set of particles to
        approximate the posterior. This method works with parameterized models created by
        pmf_from_graph() or pmf_from_cpp() where the model signature is model(theta, times).

        Parameters
        ----------
        observed_data : array or SparseObservations
            Observed data.  A plain 1-D array is accepted for univariate
            models (must not contain NaN).  For multivariate models use
            ``SparseObservations`` (see ``dense_to_sparse()``), in which
            case ``rewards`` must also be provided.
        discrete : bool, default=None
            If True, computes discrete PMF. If False, computes continuous PDF. If undefined it is 
            inferred from the graph.is_discrete attribute.
        prior : callable or list of Prior objects, optional
            Log prior function for parameters. Can be:
            - Single callable: prior(theta) -> scalar, applied to entire theta vector
            - List of Prior objects: One prior per parameter dimension.
              Use None for fixed parameters: prior=[GaussPrior(ci=[0,1]), None, GaussPrior(ci=[0,1])]
            - DataPrior instance: Data-informed prior estimated from the observed data.

            If None (default), constructs ``DataPrior(graph, observed_data, sd=5)``
            which uses method-of-moments (or probability matching for joint probability
            graphs) to estimate prior means from the data, with a wide spread
            (5x the asymptotic standard error).  Falls back to standard normal
            if DataPrior construction fails.

            Note: this data-driven default differs from the lower-level ``SVGD``
            class, whose ``prior=None`` is a plain standard normal
            (``-0.5 * sum(theta**2)``). Pass an explicit ``prior=`` to either API
            for identical behaviour.

            **With fixed parameters**:
            When using a list of priors with the `fixed` parameter, you must provide None
            at indices corresponding to fixed parameters. This is validated at initialization.

            Example:
                prior=[GaussPrior(ci=[0,1]), None, GaussPrior(ci=[0,1])],
                fixed=[(1, 0.5)]  # theta[1] fixed, prior[1] must be None
        n_particles : int or None, default None
            Number of SVGD particles. ``None`` resolves to ``20 * theta_dim``.
            More particles = better posterior approximation but slower.
        n_iterations : int, default=1000
            Number of SVGD optimization steps
        optimizer : object, optional
            Adaptive learning-rate optimizer from phasic (Adam, Adamelia,
            SGDMomentum, RMSprop, Adagrad), giving per-parameter step sizes —
            the recommended way to handle parameters of different magnitudes.
            When set, ``learning_rate`` must be None (R25) and ``regularization``
            must be 0 (R26). NOT the default: with ``optimizer=None`` SVGD uses a
            fixed global step (see ``learning_rate``); pass e.g.
            ``optimizer=Adam(learning_rate=0.01)`` to opt in.
        learning_rate : float or None, default=None
            Fixed (non-adaptive) SVGD step size. When None *and* ``optimizer`` is
            None, SVGD uses a constant default step (≈ ``0.001`` scaled by the
            observation count) — NOT an adaptive optimizer. A float sets the
            step explicitly. Mutually exclusive with ``optimizer`` (R25).
            For per-parameter adaptive steps use ``optimizer=Adam(...)`` instead.
        bandwidth : str, float, or np.ndarray, default='median_per_dim'
            Kernel bandwidth selection method:
            - 'median_per_dim': Per-dimension median heuristic (default). Uses a
              separate bandwidth per parameter dimension for an anisotropic kernel.
            - 'median': Scalar median heuristic (isotropic kernel)
            - float: Fixed scalar bandwidth value
            - np.ndarray: Fixed per-dimension bandwidth vector
        theta_init : ArrayLike, optional
            Initial particle positions (n_particles, theta_dim).
            If None, initializes randomly from standard normal.
        theta_dim : int, optional
            Number of model parameters (θ). An explicit value here overrides any
            ``theta_dim`` set at construction (useful if the graph was modified,
            e.g. via ``extend()``). When not given, it is resolved by weight mode
            so it is always either specified or RELIABLY inferred:

            - **linear / log** (default): inferred from the graph's
              ``param_length()`` (the parameterized-edge coefficient length;
              the two must match in these modes).
            - **formula** (``weight_formula`` set on the graph, or passed here):
              inferred from the formula itself — the highest ``t`` index it uses
              plus one (``n_theta``). A formula model therefore needs no
              ``theta_dim`` at construction. An explicit ``theta_dim`` may exceed
              ``n_theta`` (to reserve extra parameters); a smaller one raises.
            - **callback** (``callback`` passed here, or ``weight_callback`` set
              on the graph): cannot be inferred — the callback receives the full
              coefficient vector and may treat only some slots as parameters — so
              you MUST pass ``theta_dim`` (or ``theta_init``), else this raises.

            With ``epoch_starts`` (daisy chain) the per-epoch dimension comes from
            the graph and the optimised flat θ has length ``n_epochs × param_length``.
        return_history : bool, default=True
            If True, return particle positions throughout optimization
        seed : int, default=None
            Random seed for reproducibility
        verbose : bool, default=False
            Print progress information
        jit : bool or None, default=None
            Enable JIT compilation. If None, uses value from phasic.get_config().jit.
            JIT compilation provides significant speedup but adds initial compilation overhead.
        parallel : str or None, default=None
            Parallelization strategy:
            - 'vmap': Vectorize across particles (single device)
            - 'pmap': Parallelize across devices (uses multiple CPUs/GPUs)
            - 'none': No parallelization (sequential, useful for debugging)
            - None: Auto-select (pmap if multiple devices, vmap otherwise)
        n_devices : int or None, default=None
            Number of devices to use for pmap. Only used when parallel='pmap'.
            If None, uses all available devices.
        precompile : bool, default=True
            (Deprecated: use jit parameter instead)
            Precompile model and gradient functions for faster execution.
            First run will take longer but subsequent iterations will be much faster.
        compilation_config : CompilationConfig, dict, str, or pathlib.Path, optional
            JAX compilation optimization configuration. Can be:
            - CompilationConfig object from phasic.CompilationConfig
            - dict with CompilationConfig parameters
            - str/Path to JSON config file
            - None (uses default balanced configuration)
        positive_params : bool, default=True
            If True, applies softplus transformation to ensure all parameters are positive.
            Recommended for phase-type models where parameters represent rates.
            SVGD operates in unconstrained space, but model receives positive parameters.
        param_transform : callable, optional
            Custom parameter transformation function: transform(theta_unconstrained) -> theta_constrained.
            If provided, SVGD optimizes in unconstrained space and applies this transformation
            before calling the model. Cannot be used together with positive_params.
            Example: lambda theta: jnp.concatenate([jnp.exp(theta[:1]), jax.nn.softplus(theta[1:])])
        joint_index : bool or None, default=None
            Joint-index inference mode, where the likelihood is computed from
            converged visit counts (``accumulated_visits()``) rather than PDF/PMF
            values, for joint-index distributions in population genetics.

            You normally do not set this: it is **inferred from the graph**.
            - None (default): inferred True for a joint-probability graph
              (built via ``graph.joint_prob_graph(...)``), False otherwise.
            - True: explicit; requires a joint-probability graph (else R2 raises).
            - False on a joint-probability graph: rejected (R28) — these models
              only support joint-index inference. (Omit the argument instead.)

            When active:
            - observed_data are joint observations / vertex indices
            - discrete=True is forced (these models read visit counts, not a density)
            - moment regularization and rewards are not supported (R3/R4)
        exact_moment_grad : bool or None, default=None
            Controls the exact moment-gradient path of the underlying
            ``pmf_and_moments_from_graph`` model — meaningful for a
            standard (non-joint) graph with no rewards or with 1-D
            ``rewards`` (Batch A threaded rewards through the exact
            moments adjoint).

            - None (default): not forwarded; the model builder's own default
              governs (exact moment gradients ON).
            - False: force finite-difference moment gradients.
            - True: request the exact (reverse-mode) path explicitly. Note
              that the model builder's documented declines still govern —
              a NON-JAX-NATIVE callback declines STATICALLY at
              construction (callback mode is covered since Batch C for
              JAX-native callbacks; formula since Batch B, except a
              lazily-built decoupled formula graph — C param_length ≠
              theta dimension — which also declines statically),
              ``'log'``/``'formula'`` on a discrete/``was_dph`` graph
              decline too, ``rewards`` on an effectively DISCRETE model
              decline permanently (the continuous→discrete moment
              correction is mathematically invalid under reward weighting
              — a refuted derivation, not a missing feature), and
              per-theta MPFR declines remain dynamic (all INFO-logged);
              this kwarg cannot promise more than the builder delivers.

            An explicit value on any other model kind raises
            ``SvgdConfigError`` (R29) rather than being silently ignored:
            with 2-D (multivariate) ``rewards`` the kwarg's forwarding
            semantics are not defined yet (Batch G.2 — the per-feature
            exact path engages by default there);
            with ``epoch_starts`` the exact epoch-model gradient is the
            separate ``exact_final_grad`` kwarg (below);
            joint-probability models use the separate forward-mode
            ``exact_grad`` machinery of ``pmf_from_graph_joint_index``
            (default False — a different kwarg with a different default, not
            this one).
        exact_final_grad : bool or None, default=None
            Batch G.1 (public plumbing of the Batch-H epoch-model exact
            gradient). Only meaningful with ``epoch_starts``: ``True``
            computes the FINAL epoch's parameter gradients EXACTLY (the
            earlier epochs' slots keep finite differences — with
            ``epoch_starts=[0.0]`` every parameter is a final-epoch
            parameter, so the whole gradient is exact). Requires
            ``final_read='sojourn'`` and a linear weight mode; an
            explicit value anywhere else raises ``SvgdConfigError``
            (R30) rather than being silently ignored. ``None`` (default)
            = not forwarded; the epoch model keeps its own default
            (finite differences). Measured on the Batch-H fixtures:
            final-epoch gradient components ~3.6e5x more accurate than
            FD at ~7% of the FD backward's cost per model call; note the
            exact backward evaluates sequentially per unique exposure
            value per particle on the host. **Failure mode (deliberate,
            recorded):** once the exact path is active, a residual
            per-theta decline in the backward RAISES a diagnostic
            ``RuntimeError`` — under SVGD this means ONE particle whose
            theta declines halts the ENTIRE cloud mid-optimization;
            there is no silent per-particle FD fallback. This is also
            the recommended route for per-observation ``exposure``:
            ``svgd(obs, exposure=..., exposure_param_index=...,
            epoch_starts=[0.0], exact_final_grad=True)`` gives batched
            exposure handling plus fully exact gradients.
        exact_grad : bool or None, default=None
            Batch E (public plumbing of the joint-index exact gradient
            to the BAKED leaf). Only meaningful on a CONTINUOUS
            joint-probability graph (``joint_prob_graph(...,
            discrete=False)``) without ``epoch_starts`` or ``exposure``
            — the default no-epochs joint-prob SVGD case, where
            observations are baked and deduplicated. ``True`` computes
            the parameter gradient EXACTLY (forward-mode; cost scales
            with the parameter count — see
            ``pmf_from_graph_joint_index``'s ``exact_grad`` docs for
            the P~10-20 crossover vs FD). An explicit value anywhere
            else raises ``SvgdConfigError`` (R31) — including on a
            DEFAULT (discrete) joint-prob graph, where the message says
            to rebuild with ``discrete=False``. Residual builder-level
            declines (a ``theta_dim`` override; a structural probe
            failure at construction) fall back to FD with an INFO log
            — the model-level contract. ``None`` (default) = not
            forwarded. **Failure mode (user-decided 2026-08-14): the
            svgd path forwards ``exact_grad_decline='fd'`` — a particle
            whose theta the conditioning check declines gets a HOST-side
            finite-difference gradient for that particle only, with a
            WARNING logged per event, and the fit proceeds; every other
            particle stays exact. (SVGD's wide log-scale initialization
            routinely visits extreme theta ratios where the check
            declines; a hard raise would kill first fits, and lifting
            the check returns unreliable numbers there.) The MODEL-level
            kwarg default keeps the hard-raise contract
            (``exact_grad_decline='raise'``).** For baked mode the
            construction-time probe covers the exact index set of every
            future call, so index-dependent surprises cannot occur.
        rewards : ArrayLike, optional
            Reward vectors for computing reward-transformed likelihoods. Can be:
            - None: Standard phase-type likelihood (default)
            - 1D array (n_vertices,): Single reward vector for univariate models
            - 2D array (n_vertices, n_features): Multivariate rewards - one reward vector per feature
              dimension. Requires use of pmf_and_moments_from_graph_multivariate() model.
            For multivariate models, observed_data should also be 2D (n_times, n_features).
        fixed : list of (index, value) tuples or 1D array, optional
            Pin selected parameters at known constants so SVGD only
            optimises the *learnable* dimensions. Equivalent to a
            point-mass prior at ``value`` on the pinned slots, with
            those positions removed from the kernel and the gradient
            computation. Two accepted forms:

            **(a) Index/value tuples** (recommended) —
            ``[(idx, value), ...]``. Each tuple pins parameter ``idx``
            at ``value``::

                fixed=[(1, 0.01)]              # theta[1] = 0.01, rest learned
                fixed=[(0, 2.5), (2, 0.1)]     # theta[0]=2.5, theta[2]=0.1

            **(b) Binary mask** (legacy) — a 1D array of length
            ``theta_dim`` where ``1`` pins the slot at the value ``1.0``
            and ``0`` leaves it learnable. Use form (a) whenever the
            fix value is not ``1.0``::

                fixed=[0, 1]                   # theta[1] pinned at 1.0

            **When to use.** Fix a parameter when its value is known
            from prior data, when you want to test sensitivity to a
            single dimension while holding the rest at MLE, or when a
            slot is structurally unidentifiable from the observed data
            alone (e.g. a global rate scale that the model absorbs into
            another parameter).

            **Interaction with ``prior``.** If ``prior`` is a per-slot
            list, the entries at fixed indices **must** be ``None``;
            mismatches raise at construction time. A scalar ``prior``
            callable is fine and is auto-masked at the fixed slots.

            **Daisy-chain semantics** (``epoch_starts=[...]``). The
            flattened theta has shape ``(n_epochs * param_length,)``,
            but ``fixed`` entries still use the *local* per-epoch index
            (``[0, param_length)``) and are broadcast across all
            epochs. To pin a parameter at *different* values per epoch,
            pass a list/array of length ``n_epochs`` as the value::

                fixed=[(1, 1.0)]              # local_idx=1 pinned at 1.0 in EVERY epoch
                fixed=[(1, [1.0, 2.5])]       # local_idx=1: 1.0 in epoch 0, 2.5 in epoch 1
                fixed=[(0, 5.0), (1, [2.0, 8.0])]  # mix scalar and per-epoch values

            **Combination with ``tied``.** Compatible, with one rule:
            a given ``(local_idx, epoch)`` slot may be either ``fixed``
            *or* a member of a ``tied`` group, not both. Overlaps raise
            at construction time (rule R20).
        tied : list of (local_idx, [epoch_a, epoch_b, ...]) tuples, optional
            **Daisy-chain only.** Tie a parameter slot across two or
            more epochs so SVGD treats them as a *single* learnable
            value. Within each entry the first epoch is the **master**
            (the slot SVGD actually optimises); every subsequent epoch
            is a **slave** whose value is replaced with the master's
            on every forward evaluation, and whose gradient is routed
            back into the master.

            Examples::

                tied=[(0, [0, 1])]                  # local_idx 0 shared across epochs 0 and 1
                tied=[(1, [0, 2, 3])]               # local_idx 1 shared across epochs 0, 2, 3
                tied=[(0, [0, 1]), (1, [1, 2])]     # two independent ties

            **When to use.** Use ``tied`` when a population parameter
            (e.g. mutation rate per base, baseline hazard) is
            biologically constant across a subset of epochs while other
            parameters change. Tying reduces dimensionality, improves
            identifiability, and makes posteriors tighter without
            silently fusing epochs whose other parameters should
            differ.

            **Requirements.**

            - Requires ``epoch_starts=...`` (rule R16); a tie within a
              single epoch is meaningless and rejected with rule R17.
            - ``local_idx`` is the *per-epoch* index in
              ``[0, param_length)`` (rule R18); the epoch indices are
              in ``[0, n_epochs)``.
            - Each epoch index may appear at most once per tied entry
              (rule R19).
            - Each ``(local_idx, epoch)`` slot may belong to at most
              one tied group, and may not also be ``fixed`` (rule R20).

            **Combination with ``fixed``.** Compatible as long as no
            slot is claimed by both — see the rule R20 note in
            ``fixed`` above. To pin a parameter at the same value in
            every epoch use ``fixed=[(local_idx, value)]`` (broadcast).
            To pin a parameter at *different* values per epoch use the
            per-epoch list form of ``fixed``. Use ``tied`` only when
            the shared value should be *learned* rather than known a
            priori.

            **Combination with ``exposure``.** Compatible. The exposed
            parameter (``exposure_param_index``) may itself be tied
            across epochs; the per-observation exposure scaling
            multiplies the tied (shared) value in every epoch where
            the slot appears as master or slave.

            **Not available on the direct ``SVGD(model=...)`` path** —
            tying is a pre-model-construction concern owned by
            ``Graph.svgd``.
        preconditioner : str, preconditioner instance, or None, default=None
            Preconditioning method for multi-scale parameters (rescales the
            kernel so dimensions of different magnitude mix evenly). Functional
            on ALL model paths:
            - None (the default) or 'auto'/'jacobian' (recommended): Jacobian
              column-norm scaling. On standard/reward models this is the moment
              Jacobian; on joint-probability / daisy-chain models it is the
              *probability* Jacobian (``ProbabilityJacobianPreconditioner``) — the
              moments output is a dummy there, so phasic preconditions on the
              theta-dependent probability output instead. This resolution is
              announced via ``SvgdAssumptionWarning`` and shown by
              ``effective_options()``.
            - 'fisher': Fisher diagonal. Divides the score by the PMF/probability,
              so it can be unstable when those are small; on joint-prob it warns
              (R13). Prefer the default.
            - 'none': the ONLY way to disable preconditioning. (Note: ``None`` is
              the default and means 'auto'/enabled — pass the string ``'none'`` to
              turn preconditioning off.)
            - A MomentJacobianPreconditioner / ProbabilityJacobianPreconditioner /
              FisherPreconditioner instance: custom preconditioner.
            See also ``optimizer=Adam(...)`` for per-parameter adaptive *step
            sizes*, which complements (and is independent of) preconditioning.
        epoch_starts : array-like of float, optional
            Enables daisy-chain (time-inhomogeneous) inference. ``epoch_starts[0] == 0``;
            subsequent entries are the start times of additional epochs. ``n_epochs =
            len(epoch_starts)``. Each epoch fits its own ``param_length`` parameters,
            so the flattened theta has length ``n_epochs * param_length``. Requires
            a continuous-time joint-prob graph (``discrete=False``). See ``fixed`` and
            ``tied`` for how those kwargs interpret indices under daisy-chain.
        daisy_chain_t_eval : float, str, or None, default=None
            Time at which the final-epoch joint stop-probabilities are read off the
            JSP graph's t-vertices. Only used when ``epoch_starts`` is set.
            - Numeric: used as-is.
            - ``None`` (default): falls back to ``max(sum(epoch_dts) * 4, 10.0)``.
            - ``'auto'``: an adaptive probe (``Graph._probe_daisy_t_eval``) walks the
              daisy chain at ``daisy_chain_probe_theta`` and grows ``t_eval`` until
              the residual non-t-vertex transient mass falls below
              ``daisy_chain_t_eval_tol``. Typically picks a much smaller value than
              the conservative default and so cuts SVGD wall time significantly.
        daisy_chain_granularity : int, default=0
            Uniformization granularity passed to ``stop_probability`` inside the
            daisy-chain FFI handler. ``0`` = auto (the underlying C++ picks a safe
            value from the graph's max rate × time). Larger values give finer
            discretisation; smaller positive values trade accuracy for speed.
        daisy_chain_probe_theta : array-like, optional
            Theta used by the ``daisy_chain_t_eval='auto'`` probe. Shape
            ``(param_length,)`` (broadcast across all epochs) or
            ``(n_epochs, param_length)`` (per-epoch). Defaults to ones.
        daisy_chain_t_eval_tol : float, default=1e-3
            Residual-mass tolerance used by the ``daisy_chain_t_eval='auto'`` probe.
        exposure : float, array-like, or None, default=None
            Per-observation **exposure** :math:`\\alpha_i` — a known
            multiplicative scaling on a rate-typed component of
            :math:`\\boldsymbol{\\theta}`. For observation ``i`` the
            model is evaluated at :math:`\\boldsymbol{\\theta}^{(i)}`
            where :math:`\\theta^{(i)}_j = \\theta_j` for
            :math:`j \\neq k` and
            :math:`\\theta^{(i)}_k = \\theta_k \\cdot \\alpha_i`,
            with :math:`k` = ``exposure_param_index``. The exposed
            rate parameter and :math:`\\alpha_i` jointly determine
            each observation's expected event count (or hazard, or
            PMF, depending on the model).

            This is the GLM "exposure" / "offset" construct: it
            linearises the relationship between a rate parameter and an
            observation-specific outcome that scales with a known
            quantity. Concrete instances:

            - **Coalescent-with-mutation**: :math:`\\alpha_i` = segment
              length :math:`L_i` in bases; :math:`\\theta_k` is the
              per-base mutation rate.
            - **Survival / failure-time**: :math:`\\alpha_i` =
              time-at-risk for unit :math:`i`; :math:`\\theta_k` is the
              hazard rate.
            - **Spatial Poisson**: :math:`\\alpha_i` = area or volume
              of region :math:`i`; :math:`\\theta_k` is the intensity
              per unit area.

            Forms:

            - ``None`` (default): no exposure correction; existing
              behaviour.
            - scalar: same :math:`\\alpha` applied to every observation.
            - 1D array of length ``n_observations``: per-observation
              :math:`\\alpha`. For dense 2D ``observed_data`` of shape
              ``(n_observations, n_features)`` the same
              :math:`\\alpha_i` is shared across all features of
              observation :math:`i`.

            Requires ``exposure_param_index`` to be set. Not supported
            for ``SparseObservations`` (raises ``NotImplementedError``).
        exposure_param_index : int or None, default=None
            Index :math:`k` of the rate-typed parameter in
            :math:`\\boldsymbol{\\theta}` that ``exposure`` scales.
            Required when ``exposure`` is set. Must be in
            ``[0, param_length)``.

            Under daisy-chain (``epoch_starts=[…]``),
            :math:`\\boldsymbol{\\theta}` has flat layout
            ``(n_epochs * param_length,)``;
            ``exposure_param_index`` remains the *local* per-epoch
            index and is broadcast across every epoch internally.
        quiet_assumptions : bool, default False
            When False (default), emit a one-time
            :class:`~phasic.exceptions.SvgdAssumptionWarning` whenever an option
            is forced or notably assumed from the model type (e.g.
            ``discrete=True`` for a joint-index model, the preconditioner
            resolution, the daisy-chain re-derivation). Set True to suppress
            those notices; the full resolved set is still available via
            ``result.effective_options()``. (Routine inferences such as
            ``theta_dim`` from the graph are recorded silently regardless.)

        Returns
        -------
        SVGD
            The fitted :class:`SVGD` object. Posterior summaries are available
            on it (e.g. ``result.theta_mean``, ``result.theta_std``,
            ``result.particles``, ``result.summary()``), and
            ``result.effective_options()`` prints every option in effect with
            its provenance (default / user / inferred / forced).

            **Valid option combinations** are enforced up front by
            ``phasic.svgd_config`` (rules R1–R27) — see that module's docstring
            for the full matrix. Invalid combinations raise
            :class:`~phasic.exceptions.SvgdConfigError` before any model is built.

        Raises
        ------
        ImportError
            If JAX is not installed
        ValueError
            If model is not parameterized or theta_dim cannot be inferred

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from phasic import Graph
        >>>
        >>> # Build parameterized coalescent model
        >>> def coalescent_callback(state, nr_samples=3):
        ...     if len(state) == 0:
        ...         return [(np.array([nr_samples]), 1.0, [1.0])]
        ...     if state[0] > 1:
        ...         n = state[0]
        ...         rate = n * (n - 1) / 2
        ...         return [(np.array([n - 1]), 0.0, [rate])]
        ...     return []
        >>>
        >>> g = Graph.from_callback_parameterized(coalescent_callback, nr_samples=4)
        >>> model = Graph.pmf_from_graph(g, discrete=False)
        >>>
        >>> # Generate synthetic observed data
        >>> true_theta = jnp.array([2.0])
        >>> times = jnp.linspace(0.1, 3.0, 15)
        >>> observed_pdf = model(true_theta, times)
        >>>
        >>> # Run SVGD inference
        >>> results = Graph.svgd(
        ...     observed_data=observed_pdf,
        ...     theta_dim=1,
        ...     n_particles=30,
        ...     n_iterations=500,
        ...     learning_rate=0.01
        ... )
        >>>
        >>> print(f"True theta: {true_theta}")
        >>> print(f"Posterior mean: {results['theta_mean']}")
        >>> print(f"Posterior std: {results['theta_std']}")

        Notes
        -----
        - SVGD requires a parameterized model. Non-parameterized models (signature: model(times))
          cannot be used for inference as there are no parameters to estimate.
        - The likelihood is computed as sum(log(model(theta, observed_data)))
        - For better results, ensure observed_data has sufficient information about the parameters
        - Learning rate and number of iterations may need tuning for different problems
        """
        # Activate JAX on demand (deferred from import time).
        try:
            _ensure_jax_active()
        except Exception as e:
            raise ImportError(
                "JAX is required for SVGD inference. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            ) from e

        from .svgd import SVGD, StepSizeSchedule, RegularizationSchedule

        # Validate SVGD parameters
        if n_particles is not None and n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")

        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

        if learning_rate is not None and not isinstance(learning_rate, StepSizeSchedule) and isinstance(learning_rate, (int, float, np.integer, np.floating)) and learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        if not isinstance(regularization, RegularizationSchedule) and regularization < 0:
            raise ValueError(f"regularization must be >= 0, got {regularization}")
        if nr_moments < 1:
            raise ValueError(f"nr_moments must be >= 1, got {nr_moments}")

        # Centralised combinational validation. Catches invalid combos
        # (graph kind × continuous/discrete × observation shape × rewards
        # × epochs × fixed × exposure × ...) before any model is
        # constructed; see src/phasic/svgd_config.py for the rule list.
        from .svgd_config import from_svgd_call as _svgd_from_call, validate as _svgd_validate
        # Probe the regularization at the *current* schedule position (or
        # use the scalar). The validator only needs to know whether
        # regularization is positive at any point during the run.
        _reg_probe = (
            float(regularization(0)) if isinstance(regularization, RegularizationSchedule)
            else float(regularization)
        )
        _svgd_validate(_svgd_from_call(
            self,
            observed_data,
            rewards=rewards,
            fixed=fixed,
            tied=tied,
            callback=callback,
            weight_formula=weight_formula,
            epoch_starts=epoch_starts,
            exposure=exposure,
            exposure_param_index=exposure_param_index,
            param_transform=param_transform,
            positive_params=positive_params,
            preconditioner=preconditioner,
            optimizer=optimizer,
            learning_rate=learning_rate,
            regularization=_reg_probe,
            nr_moments=nr_moments,
            joint_index=joint_index,
            exact_moment_grad=exact_moment_grad,
            exact_final_grad=exact_final_grad,
            exact_grad=exact_grad,
            final_read=final_read,
        ))

        # Options ledger: record the provenance (default / user / inferred /
        # forced) of each resolved option so the returned object's
        # effective_options() can show what is in effect and which values phasic
        # chose or overrode. Surprising/overriding choices also emit a one-time
        # SvgdAssumptionWarning (suppressible with quiet_assumptions=True);
        # routine inferences are recorded silently. Captured here — before any
        # resolution below — so the original user-passed values are recorded.
        from .svgd_config import (
            SvgdOptionsLedger as _SvgdOptionsLedger,
            assume as _svgd_assume,
            LEDGER_OPTION_ORDER as _LEDGER_OPTION_ORDER,
        )
        import inspect as _inspect
        _ledger = _SvgdOptionsLedger()
        _opt_defaults = {
            _n: _p.default
            for _n, _p in _inspect.signature(Graph.svgd).parameters.items()
        }
        _opt_vals = dict(locals())
        for _name in _LEDGER_OPTION_ORDER:
            _ledger.record_user_or_default(
                _name, _opt_vals.get(_name), _opt_defaults.get(_name))
        # Originals (pre-resolution) for forced/inferred provenance below.
        _orig_discrete = discrete
        _orig_theta_dim = theta_dim
        _orig_joint_index = joint_index

        # Resolve / validate theta_dim against the weight mode (callback requires
        # it; formula infers it from n_theta; linear/log leaves it for the
        # param_length() inference below). See _resolve_inference_theta_dim.
        theta_dim = self._resolve_inference_theta_dim(
            theta_dim, theta_init,
            callback=callback, weight_formula=weight_formula,
            epoch_starts=epoch_starts,
        )
        if _orig_theta_dim is None and theta_dim is not None:
            _ledger.set_inferred('theta_dim', theta_dim, 'from weight mode')

        # Per-call override of graph.weight_callback. When the kwarg
        # is set, temporarily flip graph.weight_callback to the
        # supplied callable so the existing weight_mode='callback'
        # machinery in `Graph.pmf_and_moments_from_graph` picks it
        # up via `self.serialize()`. The graph's prior callback /
        # weight_mode are restored in the `finally:` block below,
        # regardless of whether SVGD succeeds or raises.
        #
        # Validator rule R21 has already rejected the incompatible
        # combination `callback + epoch_starts`, so the daisy-chain
        # FFI path will never silently ignore the callback in this
        # branch.
        _callback_overridden = callback is not None
        _prev_weight_callback = self._weight_callback
        _prev_weight_mode = self._weight_mode
        if _callback_overridden:
            # Setter flips _weight_mode to 'callback'; both fields
            # are restored in the finally below.
            self.weight_callback = callback

        # Per-call override of graph.weight_formula. Mirrors the callback
        # override but is ALLOWED under epoch_starts (the formula is evaluated
        # in C on the daisy-chain path). Mutually exclusive with callback
        # (validator rule R22). The setter compiles + installs the live tape and
        # sets weight_mode='formula' so the model builders pick it up via
        # serialize(); restored in the finally below.
        _formula_overridden = weight_formula is not None
        _prev_formula_state = (
            (self._weight_mode, self._weight_formula, self._weight_formula_tape)
            if _formula_overridden else None
        )
        if _formula_overridden:
            self.weight_formula = weight_formula

        try:
            # Reward-vector structural validation (shape + coverage).
            # Skipped when joint_index is active (rewards then encode joint
            # observation indices, not vertex-level reward weights).
            #
            # Partial coverage is handled silently by a zero-inflated
            # likelihood that models the point mass at r = 0
            # (trajectories that absorb without visiting any rewarded
            # vertex) alongside the continuous part. No warning is
            # emitted: this is the correct distributional response for
            # any (atom + continuous) mixture, and in many multi-feature
            # workflows partial coverage is the expected shape of the
            # data. The list of offending features is recorded on the
            # SVGD object as `svgd.zero_inflated_features` and surfaced
            # in `svgd.summary()` so the user can inspect it explicitly.
            _partial_coverage_features_for_zi: list[int] = []
            if rewards is not None and validate_rewards and not joint_index:
                rewards = self._validate_rewards(
                    rewards,
                    allow_2d=True,
                    coverage_mode="report",
                    context="rewards",
                )
                _partial_coverage_features_for_zi = (
                    self._partial_coverage_features(rewards)
                )

            # Validate observed_data: must be 1-D array or SparseObservations
            # (skip for joint probability graphs — they accept lists of tuples)
            _is_joint_graph = self._joint_prob_base_graph_indexer is not None
            from .svgd import SparseObservations as _SparseObs, is_sparse_observations
            if not is_sparse_observations(observed_data) and not joint_index and not _is_joint_graph:
                observed_data = np.asarray(observed_data, dtype=np.float64)
                if observed_data.ndim != 1:
                    raise TypeError(
                        "observed_data must be a 1-D array or SparseObservations. "
                        "For multivariate data, use dense_to_sparse()."
                    )
                if np.any(np.isnan(observed_data)):
                    raise ValueError(
                        "observed_data contains NaN values. "
                        "Use SparseObservations for data with missing values."
                    )
            elif is_sparse_observations(observed_data) and rewards is None:
                raise ValueError(
                    "rewards must be provided when using SparseObservations."
                )

            # Auto-infer theta_dim from graph if not provided
            if theta_dim is None and theta_init is None:
                theta_dim = self.param_length()
                if theta_dim == 0:
                    raise ValueError(
                        "theta_dim could not be inferred. Either the graph has no parameterized edges, "
                        "or you must specify theta_dim (or theta_init) explicitly."
                    )
                if _orig_theta_dim is None:
                    _ledger.set_inferred('theta_dim', theta_dim,
                                         'from graph.param_length()')

            if discrete is None:
                discrete = self.is_discrete
                _ledger.set_inferred('discrete', discrete, 'from graph.is_discrete')

            # Build the fixed-parameter mask once (theta_dim is resolved here)
            # so the non-daisy model builders can skip finite-difference
            # perturbations of fixed dims (joint_index/daisy already do this).
            # None ⇒ no skip (unchanged behavior).
            fixed_mask_for_model = _fixed_mask_from_fixed(fixed, theta_dim)

            # Default prior: DataPrior with sd=5 (wide, data-informed).
            # Skip when epoch_starts is given — the daisy-chain branch
            # builds its own broadcast prior with the right shape.
            if prior is None and epoch_starts is None:
                from .svgd import DataPrior
                try:
                    prior = DataPrior(
                        self, observed_data, sd=5.0, fixed=fixed,
                        theta_dim=theta_dim, discrete=discrete, verbose=verbose,
                    )
                    _ledger.set_inferred('prior', 'DataPrior(sd=5)',
                                         'default data-informed prior')
                except Exception as _prior_exc:
                    # DataPrior construction failed; SVGD falls back to a
                    # standard-normal prior. Surface this — it was silently
                    # swallowed before.
                    _ledger.set_inferred('prior', 'standard-normal',
                                         'DataPrior failed; fell back')
                    _svgd_assume(
                        "prior=None: data-informed DataPrior construction failed "
                        f"({type(_prior_exc).__name__}); falling back to a "
                        "standard-normal prior.",
                        quiet=quiet_assumptions,
                    )

            # Handle joint_index mode
            if self._joint_prob_base_graph_indexer is not None:
                logger = get_logger(__name__)

                # joint_index defaults to None and is inferred to True here for a
                # joint-prob graph (these models only support joint-index
                # inference). An explicit joint_index=False on a joint-prob graph
                # is a contradiction and was already rejected by validator rule
                # R28, so it never reaches this block. The None inference is
                # routine — recorded in the ledger but not warned about.
                if joint_index is None:
                    _ledger.set_inferred(
                        'joint_index', True,
                        'graph built with joint-probability support')
                joint_index = True # FIXME: joint_index is always True if graph supports it, so not really needed as argument

                if not self._joint_prob_base_graph_indexer:
                    raise ValueError(
                        "Graph was not constructed with joint index support. "
                        "Cannot use joint_index=True."
                    )
                # map observed data to indices in joint probability table
                # (shared helper; seed=None keeps the historic global-RNG
                # tie-break behaviour, bit-identical to the previous inline code)
                observed_data = self._map_joint_observations_to_indices(observed_data)

                # Check for unsupported combinations. The validator (R3/R4)
                # already raises on these before model construction; kept here
                # as defense-in-depth for direct callers.
                if regularization > 0:
                    raise NotImplementedError(
                        "Moment regularization is not supported with joint probability models."
                    )
                if rewards is not None:
                    raise NotImplementedError(
                        "Reward transformation is not supported with joint_index=True. "
                        "Set rewards=None or use joint_index=False."
                    )
                # Force discrete mode for joint_index: these models read
                # converged visit counts, not a continuous PDF/PMF.
                if _orig_discrete is False:
                    _svgd_assume(
                        "forcing discrete=True: joint-index models read converged "
                        "visit counts, not a continuous PDF/PMF; your "
                        "discrete=False was overridden.",
                        quiet=quiet_assumptions,
                    )
                    _ledger.set_forced('discrete', True, user_value=False,
                                       reason='joint-index reads visit counts')
                elif _orig_discrete is None:
                    _ledger.set_inferred('discrete', True, 'joint-index model')
                discrete = True

                # Preconditioner resolution: the default (None) and
                # 'auto'/'jacobian' on a joint-prob model build the
                # probability-Jacobian preconditioner (the moment Jacobian would
                # degenerate to a no-op here). Announce it.
                if preconditioner is None or (
                    isinstance(preconditioner, str)
                    and preconditioner in ('auto', 'jacobian')
                ):
                    _pc_shown = 'auto' if preconditioner is None else preconditioner
                    _ledger.set_inferred(
                        'preconditioner',
                        f"{_pc_shown} -> probability-Jacobian",
                        'joint-probability model',
                    )
                    _svgd_assume(
                        f"preconditioner={_pc_shown!r} resolves to the "
                        "probability-Jacobian preconditioner (built from the "
                        "model's probability output) for this joint-probability "
                        "model.",
                        quiet=quiet_assumptions,
                    )

                # Daisy-chain branch: when epoch_starts is provided, fit
                # n_epochs * param_length parameters under a piecewise-
                # constant time-inhomogeneous joint-prob model.
                if epoch_starts is not None:
                    resolved_t_eval = self._resolve_daisy_chain_t_eval(
                        daisy_chain_t_eval=daisy_chain_t_eval,
                        epoch_starts=epoch_starts,
                        probe_theta=daisy_chain_probe_theta,
                        tol=daisy_chain_t_eval_tol,
                        granularity=daisy_chain_granularity,
                        verbose=verbose,
                    )
                    # When exposure is set, push the per-observation rate
                    # scaling INTO the daisy-chain model: it builds a per-
                    # obs theta_batch and dispatches one batched FFI call.
                    # The outer SVGD wrapper sees the model's
                    # _handles_exposure_internally tag and does NOT apply
                    # _wrap_model_with_exposure on top.
                    _daisy_exposure = (
                        np.asarray(exposure, dtype=np.float64).ravel()
                        if exposure is not None else None
                    )
                    # Scalar exposure broadcasts to one entry per
                    # observation; the validator already enforces the
                    # length constraint.
                    if _daisy_exposure is not None and _daisy_exposure.size == 1:
                        _daisy_exposure = np.full(
                            (len(observed_data),),
                            float(_daisy_exposure.item()),
                            dtype=np.float64,
                        )
                    model, theta_dim, prior, fixed = self._daisy_chain_svgd_model(
                        observed_indices=observed_data,
                        epoch_starts=epoch_starts,
                        t_eval=resolved_t_eval,
                        user_prior=prior,
                        user_fixed=fixed,
                        user_tied=tied,
                        sd=5.0,
                        verbose=verbose,
                        granularity=daisy_chain_granularity,
                        exposure_arr=_daisy_exposure,
                        exposure_param_index=exposure_param_index,
                        final_read=final_read,
                        **({} if exact_final_grad is None
                           else {'exact_final_grad': exact_final_grad}),
                    )
                    # The daisy-chain builder re-derives theta_dim (flat
                    # n_epochs x param_length), and its own broadcast prior and
                    # fixed layout, overriding what was resolved above.
                    _ledger.set_forced(
                        'theta_dim', theta_dim, user_value=_orig_theta_dim,
                        reason='daisy-chain flat layout n_epochs x param_length')
                    _svgd_assume(
                        "daisy-chain: theta_dim, prior and fixed were re-derived "
                        "for the flat per-epoch layout "
                        "(n_epochs x param_length); per-epoch indices apply.",
                        quiet=quiet_assumptions,
                    )
                else:
                    # Parse fixed to get mask for joint_index model
                    # This allows the custom VJP to skip finite differences for fixed dimensions
                    fixed_mask_for_model = None
                    if fixed is not None:
                        import jax.numpy as jnp
                        if isinstance(fixed, list) and len(fixed) > 0 and isinstance(fixed[0], tuple):
                            fixed_mask_for_model = jnp.zeros(theta_dim)
                            for idx, _ in fixed:
                                fixed_mask_for_model = fixed_mask_for_model.at[idx].set(1)
                        else:
                            fixed_mask_for_model = jnp.array(fixed)
                    # Use joint_index specific model with fixed_mask.
                    # Bake observed_data (already mapped to vertex
                    # indices above) into the model when no per-observation
                    # exposure is in play — this enables the dedup +
                    # custom_vmap fast path (see pmf_from_graph_joint_index
                    # docstring under observed_indices). With per-obs
                    # exposure, every observation is intentionally distinct
                    # in theta-space, so dedup is invalid and we fall back
                    # to the legacy path.
                    _bake_obs = observed_data if exposure is None else None
                    model = Graph.pmf_from_graph_joint_index(
                        self, theta_dim=theta_dim,
                        fixed_mask=fixed_mask_for_model,
                        observed_indices=_bake_obs,
                        **({} if exact_grad is None
                           else {'exact_grad': exact_grad,
                                 # Batch E user decision 2026-08-14: the
                                 # svgd entry uses the per-particle
                                 # host-side FD fallback on gate declines
                                 # (WARN-logged) instead of the
                                 # model-level raise -- SVGD's wide init
                                 # makes raise-on-decline kill first fits.
                                 'exact_grad_decline': 'fd'}),
                    )
            # Auto-detect if we need multivariate model (2D rewards)
            elif rewards is not None:
                import jax.numpy as jnp
                rewards_arr = jnp.asarray(rewards, dtype=jnp.float64)  # Ensure float64 for C++ compatibility
                if rewards_arr.ndim == 2:
                    # Use multivariate model for 2D rewards
                    model = Graph.pmf_and_moments_from_graph_multivariate(
                        self, nr_moments=nr_moments, discrete=discrete,
                        use_ffi=False, theta_dim=theta_dim,
                        fixed_mask=fixed_mask_for_model,
                    )
                else:
                    # Use standard model for 1D rewards. Batch A (user
                    # decision 2026-08-14): explicit exact_moment_grad is
                    # honored on this leaf too (R29's 1-D-rewards arm was
                    # relaxed in the same batch); None stays not-forwarded.
                    _emg_kw_rw = ({} if exact_moment_grad is None
                                  else {'exact_moment_grad': exact_moment_grad})
                    model = Graph.pmf_and_moments_from_graph(
                        self, nr_moments=nr_moments, discrete=discrete,
                        theta_dim=theta_dim, fixed_mask=fixed_mask_for_model,
                        **_emg_kw_rw,
                    )
            else:
                # No rewards - use standard model. exact_moment_grad=None is
                # deliberately NOT forwarded (byte-identical to the callee's
                # own default resolution); an explicit value reaches only this
                # leaf — every other leaf rejects it at validation (R29).
                _emg_kw = ({} if exact_moment_grad is None
                           else {'exact_moment_grad': exact_moment_grad})
                model = Graph.pmf_and_moments_from_graph(
                    self, nr_moments=nr_moments, discrete=discrete,
                    theta_dim=theta_dim, fixed_mask=fixed_mask_for_model,
                    **_emg_kw,
                )

            # Zero-inflated likelihood wiring: when the rewards validator
            # flagged partial-coverage features above, attach a
            # JAX-differentiable p(theta) function and per-feature zero
            # counts so SVGD's _log_prob_unified can add the
            # n_zero * log(1 - p(theta)) term. Models without partial
            # coverage carry no attributes and the legacy path runs
            # unchanged.
            if _partial_coverage_features_for_zi:
                self._attach_zero_inflated_term(
                    model,
                    rewards=rewards,
                    offenders=_partial_coverage_features_for_zi,
                    observed_data=observed_data,
                )

            # Create SVGD object
            svgd = SVGD(
                observed_data=observed_data,
                model=model,
                prior=prior,
                n_particles=n_particles,
                n_iterations=n_iterations,
                learning_rate=learning_rate,
                bandwidth=bandwidth,
                theta_init=theta_init,
                theta_dim=theta_dim,
                seed=seed,
                verbose=verbose,
                progress=progress,
                jit=jit,
                parallel=parallel,
                n_devices=n_devices,
                precompile=precompile,
                compilation_config=compilation_config,
                regularization=regularization,
                nr_moments=nr_moments,
                positive_params=positive_params,
                param_transform=param_transform,
                rewards=rewards,
                fixed=fixed,
                optimizer=optimizer,
                preconditioner=preconditioner,
                exposure=exposure,
                exposure_param_index=exposure_param_index,
                _validated=True,  # Graph.svgd ran the validator at the top
            )

            # Attach the options ledger for svgd.effective_options(). Reconcile
            # the couple of defaults that SVGD.__init__ resolves internally.
            if _opt_vals.get('n_particles') is None and getattr(svgd, 'n_particles', None) is not None:
                _ledger.set_inferred('n_particles', svgd.n_particles,
                                     'default (20 x theta_dim)')
            if _opt_vals.get('learning_rate') is None and getattr(svgd, 'learning_rate', None) is not None:
                _ledger.set_inferred('learning_rate', svgd.learning_rate,
                                     'default step size')
            svgd._options_ledger = _ledger

            # Post-init for tied parameters: copy master column values
            # into the slave columns of svgd.theta_init so the initial
            # particles tensor is internally consistent. Slaves are
            # marked fixed by `broadcast_fixed`, so SVGD never updates
            # them — but the SVGD-side default init for fixed dims uses
            # the per-dim sentinel from `fixed_values` (0.0 for slaves)
            # which would be invisibly wrong without this step. The model
            # wrapper still applies `_apply_tying` on every forward call,
            # so even if a future code path skipped this step the FFI
            # would still see consistent theta — this just keeps the
            # exposed `svgd.theta_init` matrix in shape for inspection.
            _tying_info = getattr(model, '_tying_info', None)
            if _tying_info is not None and _tying_info.get('slave_to_master'):
                import jax.numpy as _jnp_local
                theta_init_jax = _jnp_local.asarray(svgd.theta_init)
                for slave_flat, master_flat in _tying_info['slave_to_master'].items():
                    theta_init_jax = theta_init_jax.at[:, slave_flat].set(
                        theta_init_jax[:, master_flat]
                    )
                svgd.theta_init = theta_init_jax

            # Run inference
            svgd.optimize(return_history=return_history)

            # Return results as dictionary for backward compatibility
            # return svgd.get_results()

            return svgd
        finally:
            # Restore graph.weight_callback / weight_mode if the
            # kwarg overrode them. This runs on both success and
            # exception so the graph state is not left mutated by
            # a failed SVGD call. When the kwarg was None we skip
            # the restore entirely (zero overhead in the common
            # path).
            if _callback_overridden:
                self._weight_callback = _prev_weight_callback
                self._weight_mode = _prev_weight_mode
            if _formula_overridden:
                self._restore_weight_formula_state(*_prev_formula_state)

    def mcmc(self,
             observed_data: ArrayLike | SparseObservations,
             discrete: bool | None = None,
             prior: Callable | None = None,
             n_samples: int = 10_000,
             n_chains: int = 4,
             burn_in: int = 1000,
             thin: int = 1,
             proposal_scale: float | ArrayLike | None = None,
             theta_init: ArrayLike | None = None,
             theta_dim: int | None = None,
             seed: int | None = None,
             verbose: bool = False,
             progress: bool = True,
             jit: bool | None = None,
             positive_params: bool = True,
             param_transform: Callable | None = None,
             rewards: ArrayLike | None = None,
             fixed: ArrayLike | None = None,
             ) -> object:
        """
        Run MCMC Metropolis-Hastings inference for Bayesian parameter estimation.

        Samples from the posterior distribution p(theta | data) using
        random-walk Metropolis-Hastings with multiple independent chains.

        Parameters
        ----------
        observed_data : array or SparseObservations
            Observed data. A plain 1-D array for univariate models.
            For multivariate models use ``SparseObservations``.
        discrete : bool, optional
            If True, computes discrete PMF. If False, continuous PDF.
            If None, inferred from the graph.
        prior : callable or list of Prior objects, optional
            Log prior function. If None, constructs ``DataPrior``.
        n_samples : int, default=10000
            Number of posterior samples per chain (after burn-in).
        n_chains : int, default=4
            Number of independent chains.
        burn_in : int, default=1000
            Number of initial samples to discard.
        thin : int, default=1
            Keep every thin-th sample.
        proposal_scale : float or array, optional
            Proposal standard deviation. If None, defaults to 0.1.
        theta_init : array, optional
            Initial parameter values. Shape (n_chains, theta_dim) or (theta_dim,).
        theta_dim : int, optional
            Dimension of parameter vector. Inferred from graph if None.
        seed : int, optional
            Random seed.
        verbose : bool, default=False
            Print progress information.
        progress : bool, default=True
            Display progress bar.
        jit : bool or None, default=None
            JIT-compile log-probability function.
        positive_params : bool, default=True
            Constrain parameters to positive domain via softplus.
        param_transform : callable, optional
            Custom parameter transformation.
        rewards : array, optional
            Reward vector/matrix for multivariate distributions.
        fixed : list or array, optional
            Fixed parameters: [(index, value), ...] or binary mask.

        Returns
        -------
        MCMC
            MCMC object with results accessible via get_results() and summary().

        Examples
        --------
        >>> g = Graph(...)  # parameterized graph
        >>> data = np.random.exponential(0.5, size=200)
        >>> mcmc = g.mcmc(data, n_samples=5000, n_chains=4)
        >>> mcmc.summary()
        >>> print(mcmc.get_results()['theta_mean'])
        """
        try:
            _ensure_jax_active()
        except Exception as _e:
            raise ImportError(
                "JAX is required for MCMC inference. Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            ) from _e

        from .mcmc import MCMC

        # Validate observed_data
        from .svgd import SparseObservations as _SparseObs, is_sparse_observations
        if not is_sparse_observations(observed_data):
            observed_data = np.asarray(observed_data, dtype=np.float64)
            if observed_data.ndim != 1:
                raise TypeError(
                    "observed_data must be a 1-D array or SparseObservations. "
                    "For multivariate data, use dense_to_sparse()."
                )
            if np.any(np.isnan(observed_data)):
                raise ValueError(
                    "observed_data contains NaN values. "
                    "Use SparseObservations for data with missing values."
                )
        elif is_sparse_observations(observed_data) and rewards is None:
            raise ValueError("rewards must be provided when using SparseObservations.")

        # Auto-infer theta_dim
        if theta_dim is None and theta_init is None:
            theta_dim = self.param_length()
            if theta_dim == 0:
                raise ValueError(
                    "theta_dim could not be inferred. Either the graph has no parameterized edges, "
                    "or you must specify theta_dim (or theta_init) explicitly."
                )

        if discrete is None:
            discrete = self.is_discrete

        # Default prior: DataPrior
        if prior is None:
            from .svgd import DataPrior
            try:
                prior = DataPrior(
                    self, observed_data, sd=5.0, fixed=fixed,
                    theta_dim=theta_dim, discrete=discrete, verbose=verbose,
                )
            except Exception:
                pass  # Fall through to MCMC's standard-normal default

        # Build model (MCMC only needs PMF, not moments)
        if rewards is not None:
            import jax.numpy as jnp
            rewards_arr = jnp.asarray(rewards, dtype=jnp.float64)
            if rewards_arr.ndim == 2:
                model = Graph.pmf_and_moments_from_graph_multivariate(
                    self, nr_moments=2, discrete=discrete,
                    use_ffi=False, theta_dim=theta_dim
                )
            else:
                model = Graph.pmf_and_moments_from_graph(
                    self, nr_moments=2, discrete=discrete,
                    theta_dim=theta_dim
                )
        else:
            model = Graph.pmf_and_moments_from_graph(
                self, nr_moments=2, discrete=discrete,
                theta_dim=theta_dim
            )

        mcmc = MCMC(
            model=model,
            observed_data=observed_data,
            prior=prior,
            n_samples=n_samples,
            n_chains=n_chains,
            burn_in=burn_in,
            thin=thin,
            proposal_scale=proposal_scale,
            theta_init=theta_init,
            theta_dim=theta_dim,
            seed=seed,
            verbose=verbose,
            progress=progress,
            jit=jit,
            positive_params=positive_params,
            param_transform=param_transform,
            rewards=rewards,
            fixed=fixed,
        )

        mcmc.run()
        return mcmc

    def method_of_moments(
        self,
        observed_data: ArrayLike | SparseObservations,
        nr_moments: int | None = None,
        rewards: ArrayLike | None = None,
        fixed: list[tuple[int, float]] | None = None,
        theta_dim: int | None = None,
        theta_init: ArrayLike | None = None,
        std_multiplier: float = 2.0,
        discrete: bool | None = None,
        verbose: bool = True,
        weighted: bool = True,
    ) -> MoMResult:
        """Find parameter estimates by matching model moments to sample moments.

        Solves the nonlinear least-squares problem::

            minimize ||moments_fn(theta) - sample_moments||^2
            subject to  theta > 0

        The returned ``MoMResult.prior`` list can be passed directly to
        :meth:`Graph.svgd` as the ``prior`` argument, providing data-informed
        priors centred on the method-of-moments estimates.

        Parameters
        ----------
        observed_data : array or SparseObservations
            Observed data.  A plain 1-D array is accepted for univariate
            models (must not contain NaN).  For multivariate models use
            ``SparseObservations`` (see ``dense_to_sparse()``), in which
            case ``rewards`` must also be provided.
        nr_moments : int, optional
            Number of moments to match per feature dimension.  If ``None``
            (default), automatically chosen based on ``weighted``:
            when ``weighted=True``, adaptive selection prunes high-order
            moments by condition number; when ``weighted=False``, uses
            the heuristic ``max(2 * n_free_params, 4)``.
            Still auto-increased if a user-specified value gives fewer
            equations than free parameters.
        rewards : np.ndarray, optional
            Reward vectors.  ``None`` for standard moments, 1-D for a single
            reward vector, 2-D ``(n_features, n_vertices)`` for multivariate.
        fixed : list, optional
            List of ``(index, value)`` tuples pinning specific parameters.
        theta_dim : int, optional
            Number of model parameters (θ). When ``None`` it is resolved by
            weight mode, exactly as in :meth:`svgd`: from the graph's
            ``param_length()`` (linear/log), from a ``weight_formula``'s
            ``n_theta`` (formula mode), or — for a callback installed via the
            ``weight_callback`` property — it must be given explicitly (a
            callback cannot be introspected), otherwise this raises.
        theta_init : np.ndarray, optional
            Initial guess for the *free* parameters (excluding fixed ones).
            If ``None`` a coordinate-wise grid search is used.
        std_multiplier : float
            Factor applied to the asymptotic standard error to obtain the
            prior standard deviation: ``prior_std = std_multiplier * se``.
        discrete : bool, optional
            ``True`` for discrete (PMF) models, ``False`` for continuous (PDF).
            If ``None``, inferred from ``self.is_discrete``.
        verbose : bool
            Print progress information.
        weighted : bool
            If ``True`` (default), use two-step efficient GMM with optimal
            weighting matrix ``W = Σ⁻¹``.  This down-weights high-variance
            moments and generally produces tighter standard errors.
            If ``False``, use unweighted least squares (legacy behavior).

        Returns
        -------
        MoMResult
            Dataclass with ``theta``, ``std``, ``prior``, ``success``, etc.

        Examples
        --------
        >>> g = Graph(...)  # parameterized graph
        >>> data = np.random.exponential(0.5, size=200)
        >>> mom = g.method_of_moments(data)
        >>> print(mom.theta)            # parameter estimate
        >>> svgd = g.svgd(data, prior=mom.prior)  # use as SVGD prior
        """
        from .method_of_moments import method_of_moments as _mom

        if discrete is None:
            discrete = self.is_discrete
        # Resolve/validate theta_dim by weight mode (a callback set via the
        # weight_callback property requires an explicit theta_dim/theta_init; a
        # weight_formula infers it from n_theta), matching Graph.svgd. method_of_
        # moments has no callback=/weight_formula=/epoch_starts kwargs, so those
        # modes reach here only via the graph's properties.
        theta_dim = self._resolve_inference_theta_dim(theta_dim, theta_init)
        if theta_dim is None:
            theta_dim = self.param_length()

        return _mom(
            self, observed_data,
            nr_moments=nr_moments,
            theta_dim=theta_dim,
            theta_init=theta_init,
            rewards=rewards,
            fixed=fixed,
            std_multiplier=std_multiplier,
            discrete=discrete,
            verbose=verbose,
            weighted=weighted,
        )

    def probability_matching(
        self,
        observed_data: ArrayLike,
        fixed: list[tuple[int, float]] | None = None,
        theta_dim: int | None = None,
        theta_init: ArrayLike | None = None,
        std_multiplier: float = 2.0,
        verbose: bool = True,
    ) -> ProbMatchResult:
        """Find parameter estimates by matching model probabilities to empirical probabilities.

        For joint probability graphs (created via :meth:`joint_prob_graph`),
        observations are feature-count tuples and the model outputs a
        probability table.  This method minimises the squared difference
        between model and empirical probabilities.

        The returned ``ProbMatchResult.prior`` list can be passed directly
        to :meth:`Graph.svgd` as the ``prior`` argument.

        Parameters
        ----------
        observed_data : list of tuples
            Observed feature-count tuples, e.g. ``[(0, 1, 0), (1, 0, 0), ...]``.
            Each tuple must match a row in the joint probability table.
        fixed : list, optional
            List of ``(index, value)`` tuples pinning specific parameters.
        theta_dim : int, optional
            Number of model parameters.  Inferred from the graph when ``None``.
        theta_init : np.ndarray, optional
            Initial guess for the *free* parameters (excluding fixed ones).
            If ``None`` a coordinate-wise grid search is used.
        std_multiplier : float
            Factor applied to the asymptotic standard error to obtain the
            prior standard deviation: ``prior_std = std_multiplier * se``.
        verbose : bool
            Print progress information.

        Returns
        -------
        ProbMatchResult
            Dataclass with ``theta``, ``std``, ``prior``, ``empirical_probs``,
            ``model_probs``, ``unique_indices``, etc.

        Raises
        ------
        ValueError
            If the graph is not a joint probability graph.

        Examples
        --------
        >>> jg = graph.joint_prob_graph(indexer, tot_reward_limit=2)
        >>> jg.update_weights([1.0, 0.5])
        >>> obs = [tuple(row) for row in jg.joint_prob_table().iloc[:, :-1].values]
        >>> result = jg.probability_matching(obs, fixed=[(1, 0.5)])
        >>> svgd = jg.svgd(obs, prior=result.prior, fixed=[(1, 0.5)])
        """
        from .probability_matching import probability_matching as _prob_match

        if self._joint_prob_base_graph_indexer is None:
            raise ValueError(
                "probability_matching() requires a joint probability graph. "
                "Build one with graph.joint_prob_graph(indexer, ...)."
            )

        if theta_dim is None:
            theta_dim = self.param_length()

        # Map observations (feature-count tuples) -> vertex indices
        joint_prob_table = self.joint_prob_table()
        obs2idx = joint_prob_table.groupby(
            joint_prob_table.columns[:-1].to_list()
        ).groups
        obs_indices = []
        for obs in observed_data:
            key = tuple(obs)
            idx = obs2idx[key]
            if idx.size > 1:
                p = joint_prob_table.loc[idx, 'prob'].to_numpy()
                p = p / p.sum()
                chosen_idx = np.random.choice(idx, p=p)
                obs_indices.append(chosen_idx.item())
            else:
                obs_indices.append(idx.item())

        return _prob_match(
            self, obs_indices,
            theta_dim=theta_dim,
            theta_init=theta_init,
            fixed=fixed,
            std_multiplier=std_multiplier,
            verbose=verbose,
        )

    @classmethod
    def moments_from_graph(cls, graph: Graph, nr_moments: int = 2, use_ffi: bool = False) -> Callable:
        """
        Convert a parameterized Graph to a JAX-compatible function that computes moments.

        This method creates a function that computes the first `nr_moments` moments of the
        phase-type distribution: [E[T], E[T^2], ..., E[T^nr_moments]].

        Moments are computed using the existing C++ `graph.moments(power)` method for efficiency.

        Parameters
        ----------
        graph : Graph
            Parameterized graph built using the Python API with parameterized edges.
            Must have edges created with `add_edge_parameterized()`.
        nr_moments : int, default=2
            Number of moments to compute. For example:
            - 1: Returns [E[T]] (mean only)
            - 2: Returns [E[T], E[T^2]] (mean and second moment)
            - 3: Returns [E[T], E[T^2], E[T^3]]
        use_ffi : bool, default=False
            If True, uses Foreign Function Interface approach.

        Returns
        -------
        callable
            JAX-compatible function with signature: moments_fn(theta) -> jnp.array(nr_moments,)
            Returns array of moments: [E[T], E[T^2], ..., E[T^k]]

        Examples
        --------
        >>> # Create parameterized coalescent model
        >>> def coalescent(state, nr_samples=2):
        ...     if len(state) == 0:
        ...         return [(np.array([nr_samples]), 1.0, [1.0])]
        ...     if state[0] > 1:
        ...         n = state[0]
        ...         rate = n * (n - 1) / 2
        ...         return [(np.array([n-1]), 0.0, [rate])]
        ...     return []
        >>>
        >>> graph = Graph(coalescent, nr_samples=3)
        >>> moments_fn = Graph.moments_from_graph(graph, nr_moments=2)
        >>>
        >>> # Compute moments for given theta
        >>> theta = jnp.array([0.5])
        >>> moments = moments_fn(theta)  # [E[T], E[T^2]]
        >>> print(f"Mean: {moments[0]}, Second moment: {moments[1]}")
        >>>
        >>> # Variance can be computed as: Var[T] = E[T^2] - E[T]^2
        >>> variance = moments[1] - moments[0]**2

        Notes
        -----
        - Requires graph to have parameterized edges (created with parameterized=True)
        - Moments are raw moments, not central moments
        - For variance, compute: Var[T] = E[T^2] - E[T]^2
        - For standard deviation: std[T] = sqrt(Var[T])
        """
        # Activate JAX on demand if we're going to use it.
        if not use_ffi:
            try:
                _ensure_jax_active()
            except Exception as _e:
                raise ImportError(
                    "JAX is required for JAX-compatible models. "
                    "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
                ) from _e

        import jax
        import jax.numpy as jnp

        # Serialize the graph to extract structure
        serialized = graph.serialize()
        param_length = serialized.get('param_length', 0)

        if param_length == 0:
            raise ValueError(
                "Graph must have parameterized edges to compute moments as function of theta. "
                "Create graph with parameterized=True and use add_edge_parameterized()."
            )

        # `_generate_cpp_from_graph` below emits a build_model() whose weight
        # computation is a hardcoded LINEAR dot product: the graph's weight_mode
        # never reaches the generated C++. Silently returning linear moments for
        # a 'log'/'callback'/'formula' graph is a WRONG ANSWER with no error --
        # on a test chain it returned E[T] = 0.325 where the truth was 0.75.
        # Fail loudly instead. (`pmf_and_moments_from_graph` DOES honour the
        # mode: it routes through GraphBuilder / a callback branch.)
        _wm = getattr(graph, '_weight_mode', 'linear')
        if _wm != 'linear':
            raise ValueError(
                f"moments_from_graph() supports weight_mode='linear' only, got "
                f"{_wm!r}. This path JIT-compiles a build_model() that computes "
                f"edge weights as a linear dot product, so a {_wm!r} graph would "
                f"silently get LINEAR moments rather than the ones its weight "
                f"rule implies. Use Graph.pmf_and_moments_from_graph(), which "
                f"honours every weight mode."
            )

        # Generate C++ build_model() code
        cpp_code = _generate_cpp_from_graph(serialized)

        # Create wrapper code that computes moments
        # Use expected_waiting_time() method which is available in C++ API
        wrapper_code = f'''{cpp_code}

#include <cmath>

// Helper function to compute factorial
double factorial(int n) {{
    double result = 1.0;
    for (int i = 2; i <= n; i++) {{
        result *= i;
    }}
    return result;
}}

extern "C" {{
    void compute_moments(
        const double* theta, int n_params,
        int nr_moments,
        double* output
    ) {{
        // Build graph from theta
        phasic::Graph g = build_model(theta, n_params);

        // Compute moments using expected_waiting_time() method
        // This replicates the _moments() function from pybind11 code
        std::vector<double> rewards;  // Empty rewards for standard moments
        std::vector<double> rewards2 = g.expected_waiting_time(rewards);
        std::vector<double> rewards3(rewards2.size());

        output[0] = rewards2[0];  // First moment (mean)

        // Raw moments of a phase-type: E[T^k] = k! * w^(k)[start], where
        // w^(1) = expected_waiting_time(unit rewards) and each subsequent
        // vector is obtained by feeding the PREVIOUS one back in as the reward
        // vector: w^(k+1) = expected_waiting_time(rewards = w^(k)).
        //
        // This previously did `rewards3[j] = rewards2[j] * pow(rewards2[j], i)`
        // — raising each entry to a power instead of passing it through — which
        // computes a different quantity entirely. On Erlang(2, rate=1) it
        // returned E[T^2] = 10 against a true value of 6. The bug was invisible
        // because _compile_wrapper_library could not build this wrapper at all.
        for (int i = 1; i < nr_moments; i++) {{
            for (int j = 0; j < (int)rewards3.size(); j++) {{
                rewards3[j] = rewards2[j];
            }}

            rewards2 = g.expected_waiting_time(rewards3);
            output[i] = factorial(i + 1) * rewards2[0];
        }}
    }}
}}
'''

        # Compile the wrapper
        lib_name = f"moments_{hashlib.sha256(wrapper_code.encode()).hexdigest()[:16]}"
        lib_path = _compile_wrapper_library(wrapper_code, lib_name)

        # Load the library
        lib = ctypes.PyDLL(lib_path)

        # Define the function signature
        lib.compute_moments.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,                      # n_params
            ctypes.c_int,                      # nr_moments
            ctypes.POINTER(ctypes.c_double)    # output
        ]
        lib.compute_moments.restype = None

        # Pure computation function
        def _compute_moments_pure(theta_flat):
            """Pure function for moment computation"""
            theta_np = np.asarray(theta_flat, dtype=np.float64)
            # Under vmap ('expand_dims') theta arrives 2-D with the batch as
            # the leading axis; the callback must return (B, nr_moments).
            # Without this branch, len(theta_np) is the batch size and the
            # ctypes call runs against a mis-shaped buffer.
            if theta_np.ndim == 2:
                # Preallocate so a B=0 batch returns (0, nr_moments) instead
                # of np.stack([]) raising (sibling pattern in
                # _exact_moments_jac_np).
                out = np.empty((theta_np.shape[0], nr_moments), dtype=np.float64)
                for b in range(theta_np.shape[0]):
                    row = np.ascontiguousarray(theta_np[b], dtype=np.float64)
                    out_row = np.zeros(nr_moments, dtype=np.float64)
                    lib.compute_moments(
                        row.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(row),
                        nr_moments,
                        out_row.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    )
                    out[b] = out_row
                return out
            if theta_np.ndim > 2:
                raise ValueError(
                    f"moments_from_graph: theta has ndim={theta_np.ndim}; "
                    "only 1-D theta or a single vmap batch dimension (2-D) "
                    "is supported"
                )
            output_np = np.zeros(nr_moments, dtype=np.float64)

            lib.compute_moments(
                theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(theta_np),
                nr_moments,
                output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )

            return output_np

        # Helper function for pure callback (used in forward and backward pass)
        def _compute_pure(theta):
            """Pure computation without custom_vjp wrapper"""
            theta = jnp.atleast_1d(theta)
            result_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)
            return jax.pure_callback(_compute_moments_pure, result_shape, theta, vmap_method='expand_dims')

        # Wrap for JAX compatibility with custom VJP for gradients
        @jax.custom_vjp
        def moments_fn(theta):
            """JAX-compatible moments function"""
            return _compute_pure(theta)

        def moments_fn_fwd(theta):
            # Call the underlying computation, not moments_fn (avoid infinite recursion!)
            moments = _compute_pure(theta)
            return moments, theta

        def moments_fn_bwd(theta, g):
            n_params = theta.shape[0]
            eps = 1e-7

            # Finite differences for gradient
            theta_bar = []
            for i in range(n_params):
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].add(-eps)

                # Call underlying computation, not moments_fn
                moments_plus = _compute_pure(theta_plus)
                moments_minus = _compute_pure(theta_minus)

                grad_i = jnp.sum(g * (moments_plus - moments_minus) / (2 * eps))
                theta_bar.append(grad_i)

            return (jnp.array(theta_bar),)

        moments_fn.defvjp(moments_fn_fwd, moments_fn_bwd)
        return moments_fn

    @classmethod
    def pmf_and_moments_from_graph(cls, graph: Graph, nr_moments: int = 2,
                                   discrete: bool = False, use_ffi: bool = False,
                                   theta_dim: int | None = None,
                                   fixed_mask: Any = None,
                                   exact_moment_grad: bool = True) -> Callable:
        """
        Convert a parameterized Graph to a function that computes both PMF/PDF and moments.

        This is more efficient than calling `pmf_from_graph()` and `moments_from_graph()`
        separately because it builds the graph once and computes both quantities.

        Parameters
        ----------
        graph : Graph
            Parameterized graph built using the Python API with parameterized edges.
        nr_moments : int, default=2
            Number of moments to compute
        discrete : bool, default=False
            If True, computes discrete PMF. If False, computes continuous PDF.
        use_ffi : bool, default=False
            If True, uses Foreign Function Interface approach.
        theta_dim : int, optional
            Number of parameters for parameterized edges. If not provided, will be
            auto-detected by probing edge states. Providing this explicitly avoids
            potential issues with auto-detection reading garbage memory.
        exact_moment_grad : bool, default=True
            Use the exact reverse-mode theta-adjoint for the moments gradient
            instead of finite differences. Covers weight_mode in
            {None, 'linear'} (continuous and discrete, including
            Graph.discretize()'d graphs), weight_mode='log' (continuous
            only -- a discrete/was_dph graph combined with 'log' is not
            supported and always uses finite differences), since
            Batch B, weight_mode='formula' (continuous only, and only
            when the graph's C-level param_length equals the model's
            theta dimension -- a lazily-built decoupled formula graph
            keeps finite differences; align with
            graph.set_param_length(theta_dim) before adding edges), and,
            since Batch C, weight_mode='callback' (continuous only, for
            JAX-NATIVE callbacks: written with jax.numpy ops, no
            float()/numpy conversions, no Python control flow on theta
            or coefficients -- probed with the deployed
            jax.jit(jax.grad(...)) transform at model construction;
            non-JAX-native callbacks keep finite differences PERMANENTLY,
            the honest boundary for the arbitrary-Python escape hatch;
            decoupled theta dimensions are fully supported here).
            1-D ``rewards``
            are supported by the exact path (Batch A: the adjoint re-scales
            at every stage of the moment chain, matching the
            reward-transformed forward; formula mode inherits this).
            Whenever finite
            differences are used instead -- because exact_moment_grad=False
            was passed explicitly, the graph's weight_mode is out of scope
            (a non-JAX-native callback -- permanent, by design -- or
            'log'/'formula'/'callback' on a discrete/was_dph graph,
            or 'formula' on a decoupled-theta graph),
            rewards are combined with a DISCRETE model (the
            continuous->discrete moment correction is invalid under reward
            weighting -- a refuted derivation, permanent), rewards are 2-D
            on this 1-D leaf (use pmf_and_moments_from_graph_multivariate,
            whose per-feature 1-D slices take the exact path), or
            the exact computation declines at a given theta (an
            ill-conditioned elimination tape, or a non-finite
            formula-tape gradient at a domain boundary) -- an INFO-level log message via
            ``logging.getLogger('phasic')`` states why, so the choice of
            gradient method is never silent. Set ``PHASIC_LOG_LEVEL=INFO`` (or
            ``logging.getLogger('phasic').setLevel(logging.INFO)``) to see
            these; the default log level (WARNING) suppresses them.

        Returns
        -------
        callable
            JAX-compatible function with signature:
            model(theta, times, rewards=None) -> (pmf_values, moments)

            Where:
            - theta: Parameter vector
            - times: Time points or jump counts
            - rewards: Optional reward vector (one per vertex). If None, computes standard moments.
            - pmf_values: jnp.array(len(times),) - PMF/PDF values at each time
            - moments: jnp.array(nr_moments,) - [E[T], E[T^2], ...] or [E[R·T], E[R·T^2], ...]

        Examples
        --------
        >>> # Create parameterized model
        >>> graph = Graph(coalescent, nr_samples=3)
        >>> model = Graph.pmf_and_moments_from_graph(graph, nr_moments=2)
        >>>
        >>> # Compute both PMF and moments
        >>> theta = jnp.array([0.5])
        >>> times = jnp.array([1.0, 2.0, 3.0])
        >>> pmf_vals, moments = model(theta, times)
        >>>
        >>> print(f"PMF at times: {pmf_vals}")
        >>> print(f"Moments: {moments}")  # [E[T], E[T^2]]
        >>>
        >>> # Compute reward-transformed moments
        >>> rewards = jnp.array([1.0, 2.0, 0.5, 1.5])  # One per vertex
        >>> pmf_vals, reward_moments = model(theta, times, rewards=rewards)
        >>> print(f"Reward moments: {reward_moments}")  # [E[R·T], E[R·T^2]]
        >>>
        >>> # Use in SVGD with moment regularization
        >>> svgd = SVGD(model, observed_pmf, theta_dim=1)
        >>> svgd.fit_regularized(observed_times=data, nr_moments=2, regularization=1.0)

        Notes
        -----
        - More efficient than separate calls to pmf_from_graph() and moments_from_graph()
        - Required for using moment-based regularization in SVGD.fit_regularized()
        - The moments are always computed from the same graph used for PMF/PDF
        - **Import phasic before importing jax / creating jax arrays.** phasic
          enables ``jax_enable_x64`` on import; the C FFI requires float64 buffers.
          jax arrays created *before* ``import phasic`` are float32 and trip the
          FFI check "Wrong buffer dtype: expected F64 but got F32". If you hit
          this, restart the kernel and import phasic first (or recreate the arrays).
        """
        # Activate JAX on demand if we're going to use it.
        if not use_ffi:
            try:
                _ensure_jax_active()
            except Exception as _e:
                raise ImportError(
                    "JAX is required for JAX-compatible models. "
                    "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
                ) from _e

        import jax
        import jax.numpy as jnp

        # Serialize the graph
        serialized = graph.serialize(theta_dim=theta_dim)
        param_length = serialized.get('param_length', 0)

        if param_length == 0:
            raise ValueError(
                "Graph must have parameterized edges. "
                "Create graph with parameterized=True and use add_edge_parameterized()."
            )

        # Fixed-parameter indices whose finite-difference perturbation can be
        # skipped (their gradient is 0 and SVGD discards it). Threaded from
        # Graph.svgd via fixed_mask; empty set ⇒ no skip (unchanged behavior).
        _fixed_dims = _fixed_indices_set_from_mask(fixed_mask)

        # B3 (default-on, additive): replace the finite-difference d(moments)/dθ
        # with the EXACT reverse-mode θ-adjoint Jacobian over the elimination
        # tape (fixes the mixed-scale FD defect for the WHOLE moment vector,
        # not just the first moment). Scope: weight_mode 'linear' (continuous
        # OR discrete -- both was_dph=True i.e. Graph.discretize(), and
        # was_dph=False native DPH, see ptd_moments_grad_theta_dph), 'log'
        # (continuous only), and 'formula' (continuous, ALIGNED theta-dim
        # only -- Batch B); monolithic. model_bwd
        # swaps the moments FD term for J^T·g_moments and falls back to FD if
        # the C path reports not-applicable for a given θ.
        #
        # No silent fallback: whenever FD ends up being used instead of the
        # exact path -- exact_moment_grad=False passed explicitly, weight_mode
        # out of scope (checked once here, statically), or a per-theta decline
        # inside _one() below (checked dynamically, e.g. MPFR conditioning) --
        # an INFO-level log line via get_logger(__name__) states why. Default
        # log level is WARNING (see logging_config.py), so these are invisible
        # unless the caller opts in with PHASIC_LOG_LEVEL=INFO; functional
        # behaviour is unchanged from before this line was added.
        #
        # Effective discreteness mirrors GraphBuilder::compute_pmf_and_moments's
        # own `is_disc = discrete || is_discrete_` dispatch exactly: a graph
        # flagged is_discrete produces discrete moments even when the caller
        # passes discrete=False (test_is_discrete_propagates_without_per_call_flag),
        # so the exact-grad gate must follow the SAME effective flag or it would
        # silently apply the continuous Jacobian to a discrete forward.
        _grad_logger = get_logger(__name__)
        _wm = serialized.get('weight_mode', 'linear')
        _effective_discrete = bool(discrete) or bool(serialized.get('is_discrete', False))
        # log-mode scope: continuous only. Confirmed by direct repro (not
        # assumed) that a was_dph graph combined with weight_mode='log' is
        # NOT guaranteed to fail elsewhere (a callable-rate discretize() can
        # pass log's positivity check), so `not _effective_discrete` here is
        # load-bearing, not defensive redundancy -- see
        # b3-log-weight-mode-plan.md.
        _log_scope_ok = (_wm == 'log' and not _effective_discrete)
        _linear_scope_ok = (_wm in (None, 'linear'))
        # Batch B: formula-mode scope. Continuous only, AND the model's
        # resolved theta dimension must equal the C graph's param_length
        # (the ALIGNED-graphs scope, b3-batchB-plan.md v2 SS-A): formula
        # is the only weight mode where the two legally decouple (tape
        # n_theta governs the model; a lazily-built graph locks C
        # param_length to the coefficient length), and on such a graph
        # the clone's update_weights(theta) RAISES -- inside
        # jax.pure_callback that is a hard XlaRuntimeError, not a soft
        # decline, so the mismatch must be gated STATICALLY here
        # (probe-confirmed at plan review + D-B6).
        _formula_scope_ok = (_wm == 'formula' and not _effective_discrete
                             and param_length == int(graph.param_length()))
        # Batch C: callback-mode scope. Continuous + JAX-NATIVE-UNDER-JIT
        # only. The probe uses the DEPLOYED transform jax.jit(jax.grad(f))
        # over ALL param edges' coefficient vectors (a plain-grad probe is
        # systematically lenient: data-dependent Python branching on theta
        # passes concrete tracers but raises TracerBoolConversionError
        # under jit at every call -- plan v2 SS-B; jit tracing is
        # value-independent, so construction success => success at every
        # theta). NO theta-dimension predicate: update_weights(theta,
        # callback=) accepts any theta length BY DESIGN
        # (phasiccpp.cpp:1879-1884, the PSMC-style idiom), the binp exit
        # is theta-blind, and W's rows take their length from theta
        # itself -- decoupled graphs are SUPPORTED (v2 SS-A).
        _cb_captured = None
        _cb_grad_jit = None
        _cb_probe_exc = None
        _callback_scope_ok = False
        if (_wm == 'callback' and not _effective_discrete
                and bool(exact_moment_grad)):
            _cb_captured = getattr(graph, 'weight_callback', None)
            if _cb_captured is not None:
                try:
                    _cb_grad_jit = jax.jit(jax.grad(_cb_captured, argnums=0))
                    _probe_t = jnp.ones(param_length, dtype=jnp.float64)
                    for _pe_row in serialized.get('param_edges', []):
                        _pe_c = jnp.asarray(np.asarray(_pe_row,
                                                       dtype=np.float64)[2:])
                        _cb_grad_jit(_probe_t, _pe_c)
                    _callback_scope_ok = True
                except Exception as _exc:
                    _cb_probe_exc = _exc
                    _cb_grad_jit = None
        if not bool(exact_moment_grad):
            _grad_logger.info(
                "pmf_and_moments_from_graph: exact_moment_grad=False -- using "
                "finite differences for the moments gradient."
            )
            _exact_grad_enabled = False
        elif _wm == 'formula' and _effective_discrete:
            _grad_logger.info(
                "pmf_and_moments_from_graph: exact moment gradient for "
                "weight_mode='formula' is continuous-only (discrete/"
                "was_dph graphs are out of scope) -- using finite "
                "differences for the moments gradient."
            )
            _exact_grad_enabled = False
        elif _wm == 'formula' and param_length != int(graph.param_length()):
            # G4 fold (wiring MINOR 1): direction-NEUTRAL message -- the
            # mismatch has two sub-classes (model theta_dim < C
            # param_length: the lazily-built decoupled graph; model
            # theta_dim > C param_length: a tape referencing more thetas
            # than the graph's parameter slots) and the original text
            # mis-diagnosed the second as the first.
            _grad_logger.info(
                "pmf_and_moments_from_graph: exact moment gradient for "
                "weight_mode='formula' requires the graph's C-level "
                "param_length (%d) to equal the model's theta dimension "
                "(%d). For a lazily-built decoupled graph (param_length "
                "locked to the coefficient length), call "
                "graph.set_param_length(theta_dim) BEFORE adding edges "
                "to align them -- using finite differences for the "
                "moments gradient.",
                int(graph.param_length()), param_length
            )
            _exact_grad_enabled = False
        elif _wm == 'callback' and _effective_discrete:
            _grad_logger.info(
                "pmf_and_moments_from_graph: exact moment gradient for "
                "weight_mode='callback' is continuous-only (discrete/"
                "was_dph graphs are out of scope) -- using finite "
                "differences for the moments gradient."
            )
            _exact_grad_enabled = False
        elif _wm == 'callback' and not _callback_scope_ok:
            # PERMANENT boundary (plan v2 SS-B / feasibility Q4 option 2):
            # a non-JAX-native callback cannot be differentiated. The
            # message keeps the "weight_mode" + "finite differences"
            # tokens the pre-existing out-of-scope pin greps.
            _grad_logger.info(
                "pmf_and_moments_from_graph: exact moment gradient not "
                "available for weight_mode='callback' with a non-JAX-"
                "native callback (the probe with jax.jit(jax.grad(...)) "
                "failed: %s%s). Write the callback with jax.numpy ops, "
                "no float()/numpy conversions and no Python control flow "
                "on theta or coefficients (jnp.where/lax.cond are fine) "
                "-- using finite differences for the moments gradient.",
                type(_cb_probe_exc).__name__ if _cb_probe_exc is not None
                else "no weight_callback set",
                (": " + str(_cb_probe_exc)) if _cb_probe_exc is not None
                else ""
            )
            _exact_grad_enabled = False
        elif not (_linear_scope_ok or _log_scope_ok or _formula_scope_ok
                  or _callback_scope_ok):
            _grad_logger.info(
                "pmf_and_moments_from_graph: exact moment gradient not "
                "available for weight_mode=%r (only None/'linear'/'log'/"
                "'formula'/'callback' -- 'log'/'formula'/'callback' only "
                "for continuous graphs, 'callback' additionally requiring "
                "a JAX-native callback -- is supported) -- using finite "
                "differences for "
                "the moments gradient.", _wm
            )
            _exact_grad_enabled = False
        else:
            _exact_grad_enabled = True
        _exact_moments_jac_np = None
        if _exact_grad_enabled:
            _exact_graph = graph.clone()
            _exact_K = int(nr_moments)
            _exact_is_log = _log_scope_ok
            _exact_is_formula = _formula_scope_ok
            _exact_is_callback = _callback_scope_ok

            def _exact_moments_jac_np(theta_np, rewards_np):
                # Host callback: set the private clone's weights at θ and read the
                # exact moment-vector Jacobian d[m]/dθ (nr_moments × param_length)
                # from C. NaNs when the exact path is not applicable → FD fallback
                # (logged at INFO, see _one() below). Batch A: rewards is a
                # genuine per-call array (size 0 = rewardless); under
                # vmap_method='expand_dims' the unbatched rewards arrive
                # with a leading length-1 axis -- collapse it (the same
                # disambiguation the forward callback uses).
                th = np.asarray(theta_np, dtype=np.float64)
                rw = np.asarray(rewards_np, dtype=np.float64)
                if rw.ndim > 1:
                    rw = rw[0]
                _rw_list = rw.tolist() if rw.size else []
                _shape = (_exact_K, param_length)

                def _one(t):
                    # CRITICAL: the clone's weights must be computed with the
                    # SAME weight rule the actual model uses, or the "exact"
                    # Jacobian would silently differentiate the wrong
                    # function (linear dot-product weights instead of log
                    # product weights) -- found via adversarial review of
                    # this batch's plan BEFORE this wiring was written; see
                    # b3-log-weight-mode-plan.md D1.1. (For formula mode,
                    # plain update_weights(t) reaches the C tape branch --
                    # probe-verified; log=False is correct there.)
                    #
                    # Batch B defense-in-depth (plan v2 SS-A): a raise
                    # anywhere in here would escape jax.pure_callback as a
                    # hard XlaRuntimeError -- convert to the standard
                    # NaN -> FD per-theta fallback, logged. The known raise
                    # classes are statically gated before this callback is
                    # built (the lazy-decoupled formula class; discrete x
                    # formula); this catch covers the residual surface
                    # (e.g. update_weights rejecting a non-finite formula
                    # weight at an extreme theta probe).
                    try:
                        if _exact_is_callback:
                            # Batch C: the pre-contraction binp exit +
                            # Python contraction J = binp @ W, with W rows
                            # from the construction-jitted grad of the
                            # BUILD-TIME-CAPTURED callback (v2 SS-D).
                            # Frozen rows (starting-vertex / cl==0 -- the
                            # primal's never-recomputed set, flagged
                            # C-side) are excluded from W and contribute
                            # 0; a non-finite binp on a FROZEN row is
                            # truly ignored (the matmul is MASKED to the
                            # engaged columns -- G4 fold), while any
                            # non-finite on an engaged row provably
                            # reaches J and declines below.
                            _exact_graph.update_weights(
                                t, callback=_cb_captured)
                            _bres = _exact_graph._moments_binp_exit(
                                _exact_K, rewards=_rw_list)
                            _bflat = np.asarray(_bres[0], dtype=np.float64)
                            if _bflat.size == 0:
                                J = _bflat  # C decline -> size check below
                            else:
                                _ni = len(_bres[2])
                                _binp = _bflat.reshape(_exact_K, _ni)
                                # G4 fold (wiring LOW-1): contract over the
                                # ENGAGED columns only -- an unmasked
                                # binp @ W would turn a non-finite adjoint
                                # on a frozen row into inf*0 = NaN and
                                # falsely decline; masking realizes the
                                # stated ignore-frozen semantics, matching
                                # the C kinds' skip-before-accumulate.
                                _eng = [
                                    _k for _k in range(_ni)
                                    if not _bres[2][_k]
                                ]
                                _W = np.zeros((len(_eng), param_length))
                                _tj = jnp.asarray(t, dtype=jnp.float64)
                                for _r, _k in enumerate(_eng):
                                    _W[_r] = np.asarray(_cb_grad_jit(
                                        _tj, jnp.asarray(
                                            _bres[1][_k],
                                            dtype=jnp.float64)))
                                J = np.asarray(_binp[:, _eng] @ _W,
                                               dtype=np.float64).reshape(-1)
                                if not np.all(np.isfinite(J)):
                                    _grad_logger.info(
                                        "pmf_and_moments_from_graph: exact "
                                        "moment gradient declined at "
                                        "theta=%s (non-finite callback "
                                        "gradient or adjoint on an engaged "
                                        "edge) -- using finite differences "
                                        "for this step.", t.tolist()
                                    )
                                    return np.full(_shape, np.nan)
                        elif _effective_discrete:
                            # dph+rewards is statically declined BEFORE this
                            # callback is ever built with rewards (the c2d
                            # refutation); _rw_list is always [] here -- the C
                            # wrapper's rewards_len!=0 -> -1 is defense in depth.
                            _exact_graph.update_weights(t, log=_exact_is_log)
                            J = np.asarray(
                                _exact_graph._moments_grad_theta_dph(
                                    _exact_K, t.tolist(), rewards=_rw_list),
                                dtype=np.float64)
                        elif _exact_is_formula:
                            _exact_graph.update_weights(t, log=False)
                            J = np.asarray(
                                _exact_graph._moments_grad_theta_formula(
                                    _exact_K, t.tolist(), rewards=_rw_list),
                                dtype=np.float64)
                        elif _exact_is_log:
                            _exact_graph.update_weights(t, log=True)
                            J = np.asarray(
                                _exact_graph._moments_grad_theta_log(
                                    _exact_K, t.tolist(), rewards=_rw_list),
                                dtype=np.float64)
                        else:
                            _exact_graph.update_weights(t, log=False)
                            J = np.asarray(
                                _exact_graph._moments_grad_theta(
                                    _exact_K, rewards=_rw_list),
                                dtype=np.float64)
                    except Exception as _exc:
                        _grad_logger.info(
                            "pmf_and_moments_from_graph: exact moment gradient "
                            "raised at theta=%s (%s: %s) -- using finite "
                            "differences for this step.",
                            t.tolist(), type(_exc).__name__, _exc
                        )
                        return np.full(_shape, np.nan)
                    if J.size != _exact_K * param_length:
                        _grad_logger.info(
                            "pmf_and_moments_from_graph: exact moment gradient "
                            "declined at theta=%s (ill-conditioned elimination "
                            "tape; for a was_dph graph a vertex mixing "
                            "a constant and a parameterized out-edge; for "
                            "weight_mode='formula' a non-finite formula-tape "
                            "gradient, e.g. d/dt sqrt(t-c) at its domain "
                            "boundary) -- using "
                            "finite differences for this step.", t.tolist()
                        )
                    return (J.reshape(_shape) if J.size == _exact_K * param_length
                            else np.full(_shape, np.nan))

                if th.ndim == 1:
                    return _one(th)
                out = np.empty((th.shape[0], _exact_K, param_length), dtype=np.float64)
                for _b in range(th.shape[0]):
                    out[_b] = _one(th[_b])
                return out

        # Reward-length guard. Rewards are one per vertex, so the vertex axis
        # (the LAST axis, for 1D and (n_features, n_vertices) 2D) must equal
        # vertices_length(). Without this a short vector reads out of bounds in
        # ptd_graph_reward_transform / expected_waiting_time. This is a
        # STATIC-SHAPE check (shapes are concrete even for jit/vmap tracers), so
        # it is safe under jit/grad/vmap -- unlike _validate_rewards, whose
        # np.asarray + coverage BFS would raise TracerArrayConversionError. It
        # runs at the top of every _compute_pure, so the custom_vjp fwd/bwd
        # (SVGD's grad path, which does not run the primal) are covered too.
        _n_vertices = int(serialized.get('n_vertices', 0))

        def _check_rewards_len(rewards):
            if rewards is None:
                return
            shp = jnp.asarray(rewards).shape
            if len(shp) == 0 or int(shp[-1]) != _n_vertices:
                raise ValueError(
                    f"rewards: the last axis must be n_vertices={_n_vertices} "
                    f"(one reward per vertex = vertices_length()), got shape "
                    f"{tuple(int(s) for s in shp)}."
                )

        # Callback mode: Python-level weight computation
        if serialized.get('weight_mode') == 'callback':
            import json
            from .ffi_wrappers import _make_json_serializable
            from . import phasic_pybind as cpp_module

            weight_callback = graph.weight_callback
            if weight_callback is None:
                raise ValueError(
                    "Graph has weight_mode='callback' but no weight_callback set."
                )

            _serialized = serialized

            def _compute_callback(theta_np, times_np, rewards_np=None):
                concrete = _apply_weight_callback(_serialized, theta_np, weight_callback)
                json_str = json.dumps(_make_json_serializable(concrete))
                builder = cpp_module.parameterized.GraphBuilder(json_str)
                try:
                    return builder.compute_pmf_and_moments(
                        np.zeros(0), times_np,
                        nr_moments=nr_moments, discrete=discrete,
                        granularity=0, rewards=rewards_np
                    )
                except _RATE_BLOWUP_EXC as _e:
                    if not _is_rate_blowup(_e):
                        raise
                    return _rate_blowup_penalty(times_np, nr_moments, rewards_np)

            def _compute_pure(theta, times, rewards=None):
                _check_rewards_len(rewards)
                # Determine output shapes based on rewards dimensionality.
                # Match the FFI / pybind paths (univariate vs multivariate).
                if rewards is not None and jnp.asarray(rewards).ndim == 2:
                    n_features = jnp.asarray(rewards).shape[0]
                    pmf_shape = jax.ShapeDtypeStruct(
                        (times.shape[0], n_features), jnp.float64,
                    )
                    moments_shape = jax.ShapeDtypeStruct(
                        (n_features, nr_moments), jnp.float64,
                    )
                else:
                    pmf_shape = jax.ShapeDtypeStruct(times.shape, times.dtype)
                    moments_shape = jax.ShapeDtypeStruct(
                        (nr_moments,), times.dtype,
                    )

                # Pass `rewards` as a pure_callback argument (rather than
                # capturing via closure) so this path survives jit /
                # vmap tracing — captured Python-level numpy arrays
                # turn into JAX tracers under jit and trigger
                # TracerArrayConversionError when np.asarray runs.
                # Empty-array sentinel for None mirrors the FFI path.
                if rewards is not None:
                    rewards_jax = jnp.atleast_1d(rewards).astype(jnp.float64)
                else:
                    rewards_jax = jnp.array([], dtype=jnp.float64)

                def _cb(t, tm, rw):
                    t_np = np.asarray(t, dtype=np.float64)
                    tm_np = np.asarray(tm, dtype=np.float64)
                    rw_np = np.asarray(rw, dtype=np.float64)
                    if rw_np.size == 0:
                        rw_np = None
                    pmf, moments = _compute_callback(t_np, tm_np, rw_np)
                    return (
                        pmf.astype(jnp.float64),
                        moments.astype(jnp.float64),
                    )

                return jax.pure_callback(
                    _cb, (pmf_shape, moments_shape),
                    theta, times, rewards_jax,
                    vmap_method='sequential'
                )

            # Companion closure: compute only cdf_zero (per-feature
            # atom mass at r = 0 of the reward-transformed
            # distribution) under callback weight mode. Mirrors the
            # linear/log pybind branch's `_cdf_zero_fn` so
            # `_attach_zero_inflated_term` can route through the fused
            # path rather than the legacy
            # `compute_reward_visit_probability_ffi`, which can't parse
            # a `weight_mode='callback'` JSON.
            def _compute_callback_cdf_zero(theta_np, rewards_np):
                """Apply the user's weight_callback to materialise a
                concrete graph at this theta, then call the same fused
                pybind method used by the linear/log path."""
                concrete = _apply_weight_callback(
                    _serialized, theta_np, weight_callback,
                )
                json_str = json.dumps(_make_json_serializable(concrete))
                builder_cb = cpp_module.parameterized.GraphBuilder(json_str)
                times_dummy = np.array([1.0], dtype=np.float64)
                try:
                    _, _, cz = builder_cb.compute_pmf_moments_and_cdf_zero(
                        np.zeros(0), times_dummy,
                        nr_moments=1, discrete=discrete,
                        granularity=0, rewards=rewards_np,
                    )
                except _RATE_BLOWUP_EXC as _e:
                    if not _is_rate_blowup(_e):
                        raise
                    cz = _cdf_zero_blowup_penalty(rewards_np)
                return cz

            def _cdf_zero_pure_cb(theta, rewards):
                theta = jnp.atleast_1d(theta)
                rewards_arr = jnp.atleast_1d(rewards).astype(jnp.float64)
                if rewards_arr.ndim == 2:
                    n_features = rewards_arr.shape[0]
                    out_shape = jax.ShapeDtypeStruct(
                        (n_features,), jnp.float64,
                    )
                else:
                    out_shape = jax.ShapeDtypeStruct((1,), jnp.float64)

                def callback_fn(theta_jax, rewards_jax):
                    theta_np = np.asarray(theta_jax, dtype=np.float64)
                    rewards_np = np.asarray(rewards_jax, dtype=np.float64)
                    return _compute_callback_cdf_zero(theta_np, rewards_np)

                return jax.pure_callback(
                    callback_fn,
                    out_shape,
                    theta, rewards_arr,
                    vmap_method='sequential',
                )

            @jax.custom_vjp
            def cdf_zero_fn_cb(theta, rewards):
                return _cdf_zero_pure_cb(theta, rewards)

            def cdf_zero_fwd_cb(theta, rewards):
                cz = _cdf_zero_pure_cb(theta, rewards)
                return cz, (theta, rewards)

            def cdf_zero_bwd_cb(res, g_cz):
                theta, rewards = res
                n_params = theta.shape[0]
                eps = 1e-7
                min_theta = 1e-9
                grads = []
                for i in range(n_params):
                    if i in _fixed_dims:
                        grads.append(0.0)
                        continue
                    theta_plus = theta.at[i].add(eps)
                    theta_minus = theta.at[i].set(
                        jnp.maximum(theta[i] - eps, min_theta)
                    )
                    actual_diff = theta_plus[i] - theta_minus[i]
                    cz_plus = _cdf_zero_pure_cb(theta_plus, rewards)
                    cz_minus = _cdf_zero_pure_cb(theta_minus, rewards)
                    # Guard actual_diff == 0 at extreme theta (see model_bwd).
                    _ok = actual_diff != 0.0
                    grad_i = jnp.sum(g_cz * jnp.where(
                        _ok, (cz_plus - cz_minus) / jnp.where(_ok, actual_diff, 1.0), 0.0))
                    grads.append(grad_i)
                return jnp.array(grads), None

            cdf_zero_fn_cb.defvjp(cdf_zero_fwd_cb, cdf_zero_bwd_cb)

            # Jump to the VJP wrapping below (shared with FFI/pybind paths)
            # Fall through to the VJP code at the end of this method
            use_ffi = False
            _callback_mode = True
            _callback_mode_cdf_zero_fn = cdf_zero_fn_cb
        else:
            _callback_mode = False
            _callback_mode_cdf_zero_fn = None

        if not _callback_mode:
            # Check if FFI is available - respect parameter, allow config override
            config = get_config()
            if not use_ffi:  # If explicitly disabled, respect it
                use_ffi = False
            else:  # If True or default, check config
                use_ffi = config._use_ffi  # Enable FFI for multi-core parallelization (C++ binding fixed!)

        if not _callback_mode and use_ffi:
            # FFI MODE: Zero-copy XLA-optimized computation with multi-core support
            from functools import partial
            import json
            from .ffi_wrappers import compute_pmf_and_moments_ffi, _make_json_serializable

            structure_json_str = json.dumps(_make_json_serializable(serialized))

            # Create partially applied FFI function with static parameters
            model_ffi_partial = partial(
                compute_pmf_and_moments_ffi,
                structure_json_str,
                nr_moments=nr_moments,
                discrete=discrete,
                granularity=0
            )

            # FFI mode doesn't need batching - FFI handles it natively
            def _compute_pure(theta, times, rewards=None):
                """FFI wrapper for multi-core parallelization.

                Supports: jit, vmap, pmap with true multi-core execution
                FFI caching: GraphBuilder cached by JSON structure
                """
                _check_rewards_len(rewards)
                theta = jnp.atleast_1d(theta)
                times = jnp.atleast_1d(times)
                return model_ffi_partial(theta=theta, times=times, rewards=rewards)
        elif not _callback_mode:
            # NOTE: this pybind11 GraphBuilder + pure_callback path is the
            # default for pmf_and_moments_from_graph (use_ffi=False is the
            # method-signature default), and is hit by SVGD's reward
            # paths. It is NOT disabled here because it is a primary path
            # for callers with rewards. The FFI version
            # (compute_pmf_and_moments_ffi) is functionally equivalent
            # and the use_ffi=False default could be flipped in a follow-
            # up — but doing so requires auditing every caller that omits
            # the kwarg. For now this fallback stays live.
            import json
            from . import phasic_pybind as cpp_module
            from .ffi_wrappers import _make_json_serializable

            structure_json_str = json.dumps(_make_json_serializable(serialized))

            # Create GraphBuilder ONCE - captured in model closure
            builder = cpp_module.parameterized.GraphBuilder(structure_json_str)

            def _compute_pmf_and_moments_cached(theta_np, times_np, rewards_np=None):
                """Uses cached builder - NO JSON parsing per call."""
                # Check if theta is batched (from vmap with expand_dims)
                if theta_np.ndim == 2:
                    times_unbatched = times_np[0] if times_np.ndim == 2 else times_np
                    pmf_results = []
                    moments_results = []
                    for theta_single in theta_np:
                        try:
                            pmf, moments = builder.compute_pmf_and_moments(
                                theta_single,
                                times_unbatched,
                                nr_moments=nr_moments,
                                discrete=discrete,
                                granularity=0,
                                rewards=rewards_np  # Pass optional rewards
                            )
                        except _RATE_BLOWUP_EXC as _e:
                            # Diverged particle: uncomputable rate. Penalise THIS
                            # particle only (not the whole batch) with a finite
                            # loss so the optimizer steps away instead of the run
                            # crashing across the pure_callback boundary. Only the
                            # rate-blowup error is swallowed; anything else raises.
                            if not _is_rate_blowup(_e):
                                raise
                            pmf, moments = _rate_blowup_penalty(
                                times_unbatched, nr_moments, rewards_np)
                        pmf_results.append(pmf)
                        moments_results.append(moments)
                    return np.array(pmf_results), np.array(moments_results)
                else:
                    # Unbatched case
                    try:
                        pmf, moments = builder.compute_pmf_and_moments(
                            theta_np,
                            times_np,
                            nr_moments=nr_moments,
                            discrete=discrete,
                            granularity=0,
                            rewards=rewards_np  # Pass optional rewards
                        )
                    except _RATE_BLOWUP_EXC as _e:
                        if not _is_rate_blowup(_e):
                            raise
                        pmf, moments = _rate_blowup_penalty(
                            times_np, nr_moments, rewards_np)
                    return pmf, moments

            # Helper function for pure callback (used in forward and backward pass)
            def _compute_pure(theta, times, rewards=None):
                """Pure computation without custom_vjp wrapper"""
                _check_rewards_len(rewards)
                theta = jnp.atleast_1d(theta)
                times = jnp.atleast_1d(times)

                # Determine output shapes based on rewards dimensionality
                if rewards is not None and rewards.ndim == 2:
                    # 2D rewards: multivariate case
                    n_features = rewards.shape[1]
                    pmf_shape = jax.ShapeDtypeStruct((times.shape[0], n_features), jnp.float64)
                    moments_shape = jax.ShapeDtypeStruct((n_features, nr_moments), jnp.float64)
                else:
                    # No rewards or 1D rewards: univariate case
                    pmf_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)
                    moments_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)

                # Convert rewards to JAX array to allow passing through vmap (handles both batched & unbatched)
                if rewards is not None:
                    rewards_jax = jnp.atleast_1d(rewards).astype(jnp.float64)
                else:
                    rewards_jax = jnp.array([], dtype=jnp.float64)  # Empty array as sentinel for None

                # Callback handles vmap batch dimension at runtime (not during tracing)
                def callback_fn(theta_jax, times_jax, rewards_jax):
                    """Runtime conversion (not traced) - handles vmap batching"""
                    theta_np = np.asarray(theta_jax)
                    times_np = np.asarray(times_jax)
                    rewards_np = np.asarray(rewards_jax, dtype=np.float64)

                    # Handle vmap batch dimension for rewards
                    # vmap adds a batch dimension to the front, but rewards should stay constant
                    # Original rewards: 1D (n_vertices,) or 2D (n_vertices, n_features)
                    # After vmap: 2D (batch, n_vertices) or 3D (batch, n_vertices, n_features)
                    if rewards_np.ndim == 3:
                        # 3D: batched 2D rewards, take first (all identical)
                        rewards_np = rewards_np[0]
                    elif rewards_np.ndim == 2:
                        # Could be: (batch, n_vertices) from 1D vmap, or (n_vertices, n_features) multivariate
                        # Check if theta is batched to determine if this is from vmap
                        if theta_np.ndim == 2:
                            # Batched case: take first reward vector
                            rewards_np = rewards_np[0]
                        # else: not batched, keep as-is (multivariate case)

                    # Convert empty array sentinel back to None
                    if rewards_np.size == 0:
                        rewards_np = None

                    return _compute_pmf_and_moments_cached(theta_np, times_np, rewards_np)

                result = jax.pure_callback(
                    callback_fn,
                    (pmf_shape, moments_shape),
                    theta, times, rewards_jax,  # Pass all args to allow vmap
                    vmap_method='expand_dims'
                )
                return result

            # Companion closure: compute only cdf_zero (per-feature atom
            # mass at r = 0 on the reward-transformed distribution),
            # used by the zero-inflated likelihood path. JAX-differentiable
            # via the same finite-difference pattern as the main model.
            # Calls the new pybind method `compute_pmf_moments_and_cdf_zero`
            # so the reward_transform is the same as the model path,
            # discarding pmf/moments. The (small) extra cost replaces a
            # `backward_probabilities` linear solve in
            # `compute_reward_visit_probability_ffi` per particle.
            def _compute_cdf_zero_cached(theta_np, rewards_np):
                """Uses cached builder; returns cdf_zero (1D)."""
                # Tiny times array — we only need the reward-transform
                # side-effect, not any real time evaluation. The pybind
                # method evaluates PDF at this dummy point but the cost
                # is dominated by the reward transform.
                times_np = np.array([1.0], dtype=np.float64)
                if theta_np.ndim == 2:
                    cdf_zeros = []
                    for theta_single in theta_np:
                        try:
                            _, _, cz = builder.compute_pmf_moments_and_cdf_zero(
                                theta_single, times_np,
                                nr_moments=1, discrete=discrete,
                                granularity=0, rewards=rewards_np,
                            )
                        except _RATE_BLOWUP_EXC as _e:
                            # Diverged particle on the auto-attached zero-inflation
                            # path: penalise this particle (finite) instead of
                            # aborting the run. See _is_rate_blowup.
                            if not _is_rate_blowup(_e):
                                raise
                            cz = _cdf_zero_blowup_penalty(rewards_np)
                        cdf_zeros.append(cz)
                    return np.array(cdf_zeros)
                else:
                    try:
                        _, _, cz = builder.compute_pmf_moments_and_cdf_zero(
                            theta_np, times_np,
                            nr_moments=1, discrete=discrete,
                            granularity=0, rewards=rewards_np,
                        )
                    except _RATE_BLOWUP_EXC as _e:
                        if not _is_rate_blowup(_e):
                            raise
                        cz = _cdf_zero_blowup_penalty(rewards_np)
                    return cz

            def _cdf_zero_pure(theta, rewards):
                """Pure cdf_zero computation, JAX-callback path."""
                theta = jnp.atleast_1d(theta)
                rewards_arr = jnp.atleast_1d(rewards).astype(jnp.float64)
                # Output shape: (1,) for 1D rewards, (n_features,) for 2D.
                if rewards_arr.ndim == 2:
                    n_features = rewards_arr.shape[0]
                    out_shape = jax.ShapeDtypeStruct((n_features,), jnp.float64)
                else:
                    out_shape = jax.ShapeDtypeStruct((1,), jnp.float64)

                def callback_fn(theta_jax, rewards_jax):
                    theta_np = np.asarray(theta_jax)
                    rewards_np = np.asarray(rewards_jax, dtype=np.float64)
                    if rewards_np.ndim == 3:
                        rewards_np = rewards_np[0]
                    elif rewards_np.ndim == 2 and theta_np.ndim == 2:
                        rewards_np = rewards_np[0]
                    return _compute_cdf_zero_cached(theta_np, rewards_np)

                return jax.pure_callback(
                    callback_fn,
                    out_shape,
                    theta, rewards_arr,
                    vmap_method='expand_dims',
                )

            @jax.custom_vjp
            def cdf_zero_fn(theta, rewards):
                return _cdf_zero_pure(theta, rewards)

            def cdf_zero_fwd(theta, rewards):
                cz = _cdf_zero_pure(theta, rewards)
                return cz, (theta, rewards)

            def cdf_zero_bwd(res, g_cz):
                theta, rewards = res
                n_params = theta.shape[0]
                eps = 1e-7
                min_theta = 1e-9
                grads = []
                for i in range(n_params):
                    if i in _fixed_dims:
                        grads.append(0.0)
                        continue
                    theta_plus = theta.at[i].add(eps)
                    theta_minus = theta.at[i].set(
                        jnp.maximum(theta[i] - eps, min_theta)
                    )
                    actual_diff = theta_plus[i] - theta_minus[i]
                    cz_plus = _cdf_zero_pure(theta_plus, rewards)
                    cz_minus = _cdf_zero_pure(theta_minus, rewards)
                    # Guard actual_diff == 0 at extreme (diverged) theta (see
                    # model_bwd): finite 0 gradient, not nan. Value-preserving.
                    _ok = actual_diff != 0.0
                    grad_i = jnp.sum(g_cz * jnp.where(
                        _ok, (cz_plus - cz_minus) / jnp.where(_ok, actual_diff, 1.0), 0.0))
                    grads.append(grad_i)
                return jnp.array(grads), None

            cdf_zero_fn.defvjp(cdf_zero_fwd, cdf_zero_bwd)

        # Wrap for JAX compatibility with custom VJP for gradients
        @jax.custom_vjp
        def model(theta, times, rewards=None):
            """JAX-compatible model function returning (pmf, moments)

            Parameters
            ----------
            theta : jax.Array
                Parameter vector
            times : jax.Array
                Time points
            rewards : jax.Array or None, optional
                Reward vector for reward-transformed moments

            Returns
            -------
            tuple of (jax.Array, jax.Array)
                (pmf_values, moments)
            """
            return _compute_pure(theta, times, rewards)

        def model_fwd(theta, times, rewards=None):
            # Call the underlying computation, not model (avoid infinite recursion!)
            pmf, moments = _compute_pure(theta, times, rewards)
            return (pmf, moments), (theta, times, rewards)

        def model_bwd(res, g):
            theta, times, rewards = res
            g_pmf, g_moments = g  # Unpack gradient tuple

            n_params = theta.shape[0]
            eps = 1e-7

            # B3: exact moment-vector Jacobian J = d[m]/dθ (nr_moments ×
            # n_params) via a host callback into the reverse-mode C adjoint. The
            # exact moments contribution to θ_bar is then J^T · g_moments. None
            # when disabled → pure FD. _exact_ok gates a fallback to FD if the C
            # path reported not-applicable (NaN) for this θ.
            #
            # rewards ARE threaded into the exact path since Batch A
            # (2026-08-14): 1-D rewards flow to ptd_moments_grad_theta/_log
            # via the second pure_callback argument below, and the C core
            # re-scales at EVERY stage of the forward moment chain plus the
            # matching adjoint-side VJP (seed-only scaling is provably wrong
            # for K>=2 -- b3-batchA-plan.md). Two static exceptions keep FD:
            # DISCRETE models (the continuous->discrete correction is
            # REFUTED under reward weighting) and 2-D rewards on this 1-D
            # leaf (the multivariate wrapper's per-feature 1-D slices use
            # the exact path instead). rewards'
            # None-ness/size is static per traced call (a genuine Python
            # None or a concrete shape, never a runtime-varying value), so
            # this check is safe as plain Python control flow, no
            # pure_callback needed. No silent fallback: log why, same as the
            # weight_mode/decline cases above.
            _rewards_provided = rewards is not None and jnp.asarray(rewards).size > 0
            _rewards_1d = (_rewards_provided
                           and jnp.asarray(rewards).ndim == 1)
            _exact_tbm = None
            if (_exact_grad_enabled and _rewards_provided
                    and _effective_discrete):
                # Batch A: reward-weighted DISCRETE moment gradients are
                # REFUTED (the c2d correction's derivation needs U to
                # commute with P, broken by reward scaling; 2nd moments
                # provably wrong -- plan review 2026-08-14). Static
                # decline with a truthful log; FD stays correct.
                _grad_logger.info(
                    "pmf_and_moments_from_graph: exact moment gradient "
                    "with rewards is not available for DISCRETE models "
                    "(the continuous->discrete correction is invalid "
                    "under reward weighting) -- using finite differences "
                    "for the moments gradient."
                )
            elif (_exact_grad_enabled and _rewards_provided
                    and not _rewards_1d):
                # 2-D rewards flow through this 1-D leaf's forward but the
                # exact Jacobian contraction is 1-D-only; the multivariate
                # wrapper slices per-feature 1-D and gets exact there.
                _grad_logger.info(
                    "pmf_and_moments_from_graph: exact moment gradient "
                    "supports 1-D rewards only (2-D rewards keep finite "
                    "differences on this leaf; the multivariate wrapper's "
                    "per-feature slices use the exact path)."
                )
            elif _exact_grad_enabled:
                _rw_arg = (jnp.asarray(rewards, dtype=jnp.float64)
                           if _rewards_1d
                           else jnp.zeros((0,), dtype=jnp.float64))
                _exactJ = jax.pure_callback(
                    _exact_moments_jac_np,
                    jax.ShapeDtypeStruct((nr_moments, param_length), jnp.float64),
                    theta, _rw_arg, vmap_method='expand_dims')
                _exact_ok = jnp.all(jnp.isfinite(_exactJ))
                _exact_tbm = _exactJ.T @ g_moments  # (n_params,)

            # Finite differences for gradient
            # Clamp lower perturbation to stay positive (theta comes from
            # softplus which can be as small as 1e-9, smaller than eps)
            min_theta = 1e-9
            theta_bar = []
            for i in range(n_params):
                # Skip fixed parameters: their gradient is 0 and SVGD discards
                # it (mirrors the daisy / joint_index paths). Value-preserving
                # for learnable dims; avoids 2 forward passes per fixed dim.
                if i in _fixed_dims:
                    theta_bar.append(0.0)
                    continue
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].set(jnp.maximum(theta[i] - eps, min_theta))
                actual_diff = theta_plus[i] - theta_minus[i]

                # Call underlying computation, not model
                pmf_plus, moments_plus = _compute_pure(theta_plus, times, rewards)
                pmf_minus, moments_minus = _compute_pure(theta_minus, times, rewards)

                # Combine gradients from both PMF and moments
                # Use nansum to handle NaN values in PMF (from missing observations)
                # NaN in PMF means the observation was missing, so it shouldn't contribute to gradient
                #
                # Guard actual_diff == 0: at an extreme (diverged) theta the +/-eps
                # probes collapse to the same float64 value (eps underflows against
                # a huge |theta|), so actual_diff -> 0 and the division would yield
                # nan. The forward is already fail-soft to a constant penalty there,
                # so the true FD slope is ~0; return a finite 0 gradient. This keeps
                # the likelihood score finite so the PRIOR's restoring force pulls
                # the diverged particle back, instead of a nan poisoning the coupled
                # SVGD kernel. Value-preserving for normal theta (actual_diff != 0).
                _ok = actual_diff != 0.0
                _safe = jnp.where(_ok, actual_diff, 1.0)
                pmf_diff = jnp.where(_ok, (pmf_plus - pmf_minus) / _safe, 0.0)
                grad_pmf_i = jnp.nansum(g_pmf * pmf_diff)
                moments_diff = jnp.where(_ok, (moments_plus - moments_minus) / _safe, 0.0)
                grad_moments_i = jnp.sum(g_moments * moments_diff)
                if _exact_tbm is not None:
                    # Swap the WHOLE moments FD term for the exact (J^T·g_moments)_i,
                    # keeping FD for pmf. Fall back to the FD term if the exact
                    # path was not applicable for this θ (NaN sentinel).
                    grad_moments_i = jnp.where(_exact_ok, _exact_tbm[i], grad_moments_i)
                grad_i = grad_pmf_i + grad_moments_i

                theta_bar.append(grad_i)

            return jnp.array(theta_bar), None, None  # gradients for theta, times, rewards

        model.defvjp(model_fwd, model_bwd)
        # Expose the per-feature cdf_zero closure on the pybind paths
        # (the default for Graph.svgd). Consumed by
        # _attach_zero_inflated_term to evaluate p(θ) = 1 - cdf_zero(θ)
        # without a separate backward_probabilities solve. The FFI path
        # (use_ffi=True) does not expose this attribute; its zero-
        # inflation wiring continues to use the legacy
        # compute_reward_visit_probability_ffi solve.
        if _callback_mode:
            # Callback mode also exposes the fused cdf_zero closure
            # (built above against the same concrete-graph machinery
            # the model uses). Without this, the legacy fallback in
            # _attach_zero_inflated_term would feed the raw
            # ``weight_mode='callback'`` JSON to the C++ parser, which
            # only accepts 'linear' and 'log'.
            model._cdf_zero_fn = _callback_mode_cdf_zero_fn
        elif not use_ffi:
            model._cdf_zero_fn = cdf_zero_fn
        # Source graph reference for direct ``SVGD(model=...)``
        # callers; enables auto-attachment of the zero-inflated
        # likelihood term (see ``zero_inflation.py``).
        model._source_graph = graph
        # Preconditioner source: this model returns REAL moments as its second
        # output, so MomentJacobianPreconditioner ('auto') applies here.
        model._precondition_output = 'moments'
        return model

    @classmethod
    def pmf_from_graph_joint_index(cls, graph: Graph, theta_dim: int | None = None,
                                    fixed_mask: Any = None,
                                    exclude_vertices: list[int] | None = None,
                                    observed_indices: Any = None,
                                    exact_grad: bool = False,
                                    exact_grad_decline: str = 'raise') -> Callable:
        """
        Create a JAX-compatible model for joint index distributions.

        In joint index mode, likelihood is computed from exact expected sojourn times
        rather than PDF/PMF values. The observed_data should contain vertex indices
        (integers) instead of time values.

        This is used for joint index distributions in population genetics models where
        the observed data represents which states (vertices) were visited.

        Uses the fast expected_sojourn_time() method which computes exact sojourn
        times for all states in a single pass through the elimination trace.

        Parameters
        ----------
        graph : Graph
            Parameterized graph built using the Python API with parameterized edges.
        theta_dim : int, optional
            Number of parameters for parameterized edges. If not provided, will be
            auto-detected from the graph.
        fixed_mask : jnp.ndarray, optional
            Binary mask indicating which parameters are fixed (1=fixed, 0=learnable).
            If provided, gradients for fixed dimensions will be zero in the custom VJP.
            This is used to skip finite difference computation for fixed parameters.
        exclude_vertices : list of int, optional
            Vertex indices to exclude from the normalization constant.
            Use this for ascertainment bias correction — e.g., exclude the
            zero-mutation terminal state when conditioning on observing at
            least one mutation. The excluded vertices are removed from the
            denominator, so the model returns P(s | s not in excluded set).
        observed_indices : array-like of int, optional
            When supplied, enables baked observation dedup: the model will
            be built around the unique vertex indices in ``observed_indices``
            (typically far fewer than the number of observations, with many
            repeats). The FFI sojourn-times call sees ``k = n_unique`` instead
            of ``k = n_obs`` (the FFI consumer's inner loop scales linearly
            in ``k``), and the returned per-observation array is reconstructed
            via a scatter through the inverse-index mapping. The model also
            wraps its forward in a ``custom_vmap`` rule so that under
            ``vmap(grad(loss))(particles)`` the per-particle dispatch fuses
            into a single batched FFI call.
            When ``observed_indices`` is ``None`` (default), the legacy path
            is used and ``vertex_indices`` is read from the model's runtime
            argument as before.
        exact_grad_decline : {'raise', 'fd'}, default='raise'
            What a COMMITTED exact model does when the C adjoint declines
            at a runtime theta (e.g. the conditioning check at extreme
            theta ratios). ``'raise'`` (default) = the hard-stop contract:
            a diagnostic ``RuntimeError``. ``'fd'`` = compute THAT call's
            Jacobian rows by host-side central finite differences on the
            private clone (relative steps on the raw sojourn values, then
            the exact quotient rule) and WARN-log the event — used by the
            ``Graph.svgd`` plumbing (user decision 2026-08-14) so one
            extreme particle cannot kill a whole SVGD cloud.
        exact_grad : bool, default=False
            Use the exact forward-mode theta-adjoint for the sojourn-vector
            gradient instead of finite differences (FD). Covers weight_mode
            'linear' (continuous and native DPH -- discrete=True/was_dph=
            False graphs need no special-casing here), in BOTH the runtime
            vertex_indices mode and the ``observed_indices`` BAKED/dedup
            mode (Batch E: the baked probe covers the exact static index
            union -- in baked mode the probe runs over
            ``union(unique observed indices, all_terminal)``, not the
            non-baked ``union(all_terminal, [0])`` -- so a committed
            baked model can never hit an
            index-dependent decline at call time); NOT yet supported
            for weight_mode in {'formula', 'callback'}, a ``was_dph`` graph
            (``Graph.discretize()``), or a
            ``theta_dim`` that overrides the graph's own ``param_length``
            -- each falls back to FD with an INFO-level log via
            ``logging.getLogger('phasic')`` stating why (set
            ``PHASIC_LOG_LEVEL=INFO`` to see these).

            **Commit-or-decline semantics (Batch F):** at model
            construction a one-time probe runs the exact gradient at the
            reference point ``theta=ones`` over the union of the
            all-terminal indices and vertex 0. If the probe FAILS, the whole model uses FD (logged; no
            exact-path overhead ever). If it SUCCEEDS, the model COMMITS
            to the exact path: the FD branch is never traced (so under
            SVGD's ``vmap(jit(grad(...)))`` only the exact branch runs),
            and a later per-theta decline (e.g. an ill-conditioned tape at
            one particle's theta) RAISES a diagnostic ``RuntimeError``
            instead of silently falling back -- a deliberate hard-stop
            failure mode. A NaN/inf theta raises ``ValueError`` from
            ``update_weights``; an out-of-range observed index raises
            ``ValueError`` naming the index. If your model is
            well-conditioned only away from ``theta=ones``, the probe may
            decline spuriously (logged) -- a documented limitation of the
            static reference point.

            Defaults to **False**, unlike every other B3 exact-gradient
            kwarg in this codebase (which default to ``True``). This
            function uses FORWARD-mode (one pass per theta parameter P),
            not reverse-mode, so its cost scales with P and only overtakes
            FD once P is roughly 10-20+ on representative graphs (see
            ``b3-joint-index-plan.md``'s D3 benchmark) -- unlike this P=2
            (coalescent rate + mutation rate) model's typical usage.
            The default staying ``False`` is a DELIBERATE, user-decided
            trade (2026-08-13, post-Batch-F): FD-favoured cost at small P
            and no hard-stop raises by default, versus exact gradients on
            request. Set ``exact_grad=True`` explicitly for richer models
            (P roughly 10+) or when FD's documented mixed-scale gradient
            defect (the reason this feature exists) matters more than the
            P-scaled cost -- accepting that a committed model RAISES on a
            per-theta decline instead of falling back.

        Returns
        -------
        callable
            JAX-compatible function with signature:
            model(theta, vertex_indices, rewards=None) -> (sojourn_times, dummy_moments)

            Where:
            - theta: Parameter vector
            - vertex_indices: Array of vertex indices (integers). Ignored when
              ``observed_indices`` was supplied at construction time; the
              baked indices are used instead.
            - rewards: Ignored (must be None for joint_index mode)
            - sojourn_times: Expected sojourn times for the specified vertices
            - dummy_moments: Zeros array (moments not supported in joint_index mode)

        Notes
        -----
        - Uses expected_sojourn_time() for fast exact computation
        - Much faster than iterating accumulated_visiting_time() until convergence
        - Moment regularization is not supported (regularization must be 0)
        - Reward transformation is not supported (rewards must be None)
        - When ``observed_indices`` is supplied, dedup typically gives a
          10-100x wall-clock win for SVGD on joint-prob graphs with many
          repeated observations. See ``Graph.svgd`` joint_index path.
        """
        # Check if JAX is available
        try:
            _ensure_jax_active()
        except Exception as _e:
            raise ImportError(
                "JAX is required for JAX-compatible models. Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            ) from _e

        import jax
        import jax.numpy as jnp
        from .ffi_wrappers import compute_sojourn_times_ffi

        # ComputeSojournTimesFfiImpl calls ptd_graph_update_weights(...,
        # /*use_log=*/false) directly (graph_builder_ffi.cpp:887, :941),
        # bypassing GraphBuilder and hardcoding LINEAR weights. A 'log' graph
        # would therefore silently get the linear answer (verified: identical
        # output for linear and log where the log rates differ). 'callback' and
        # 'formula' ARE honoured (see the callback branch below), so only 'log'
        # is rejected here.
        _wm = getattr(graph, '_weight_mode', 'linear')
        if _wm == 'log':
            raise ValueError(
                "pmf_from_graph_joint_index() does not support "
                "weight_mode='log': the sojourn FFI handler computes edge "
                "weights linearly, so a 'log' graph would silently receive "
                "LINEAR weights instead of the product rule 'log' implies. "
                "Use weight_mode='formula' (e.g. a product expression) or "
                "'callback', both of which are honoured."
            )

        # Serialize the graph
        serialized = graph.serialize(theta_dim=theta_dim)
        param_length_actual = serialized.get('param_length', 0)

        if param_length_actual == 0:
            raise ValueError(
                "Graph must have parameterized edges. "
                "Create graph with parameterized=True and use add_edge_parameterized()."
            )

        # Keep serialized dict for FFI (it handles JSON conversion internally)
        structure_dict = serialized

        # Find ALL terminal vertex indices at model construction time
        # Terminal vertices are those with an edge to an absorbing state (no outgoing edges)
        all_terminal_indices = []
        for vertex in graph.vertices():
            for edge in vertex.edges():
                if len(edge.to().edges()) == 0:  # points to absorbing state
                    all_terminal_indices.append(vertex.index())
                    break
        all_terminal_indices = sorted(set(all_terminal_indices))
        if exclude_vertices is not None:
            exclude_set = set(int(v) for v in exclude_vertices)
            all_terminal_indices = [v for v in all_terminal_indices if v not in exclude_set]
        all_terminal_indices_np = np.asarray(all_terminal_indices, dtype=np.int32)
        all_terminal_indices = jnp.array(all_terminal_indices, dtype=jnp.int32)

        # Dedup mapping: when SVGD supplies observed_indices, precompute
        # unique vertex indices + inverse-index mapping once at construction.
        # The FFI's per-call cost is O(commands * k); k drops from n_obs to
        # n_unique (often a 10-100x win for joint-prob SVGD). The forward
        # then scatters the small unique result back to n_obs shape so that
        # downstream consumers (SVGD's _log_lik_from_pmf) see the same
        # per-observation pmf vector as before. Mirrors the same dedup
        # pattern used in _daisy_chain_svgd_model for the per-observation
        # exposure path.
        if exact_grad_decline not in ('raise', 'fd'):
            raise ValueError(
                "exact_grad_decline must be 'raise' or 'fd', got "
                f"{exact_grad_decline!r}."
            )
        _baked = observed_indices is not None
        if _baked:
            _obs_idx_arr = np.asarray(observed_indices, dtype=np.int32)
            if _obs_idx_arr.ndim > 1:
                raise ValueError(
                    "pmf_from_graph_joint_index: observed_indices must be "
                    f"1-D vertex indices (got shape {_obs_idx_arr.shape}). "
                    "Multi-column observation OUTCOME tuples are a "
                    "Graph.svgd input format, not a model-level index "
                    "array -- raveling them would produce garbage indices."
                )
            _obs_idx_np = _obs_idx_arr.ravel()
            # Batch E: construction-time bounds validation. The sojourn FFI
            # silently NaN-fills out-of-range indices, and the exact
            # backward's per-call check does not run for baked mode (the
            # index set is static) -- validate HERE, loudly, once.
            _n_vertices_bv = int(graph.vertices_length())
            if _obs_idx_np.size and (
                    int(_obs_idx_np.min()) < 0
                    or int(_obs_idx_np.max()) >= _n_vertices_bv):
                raise ValueError(
                    "pmf_from_graph_joint_index: observed_indices out of "
                    f"range (got min={int(_obs_idx_np.min())}, "
                    f"max={int(_obs_idx_np.max())}; graph has "
                    f"{_n_vertices_bv} vertices)."
                )
            _uniq_idx_np, _inverse_idx_np = np.unique(
                _obs_idx_np, return_inverse=True,
            )
            _uniq_idx_jnp = jnp.asarray(_uniq_idx_np, dtype=jnp.int32)
            _inverse_idx_jnp = jnp.asarray(_inverse_idx_np, dtype=jnp.int32)
            _n_obs_baked = int(_obs_idx_np.size)
        else:
            _uniq_idx_np = None
            _inverse_idx_np = None
            _uniq_idx_jnp = None
            _inverse_idx_jnp = None
            _n_obs_baked = None

        # B3 joint-index extension (default-on, additive): replace the
        # finite-difference d(sojourn_probs)/dtheta with the EXACT forward-
        # mode theta-adjoint over the elimination tape
        # (Graph._sojourn_grad_theta_subset). Scope: weight_mode='linear'
        # only (log is already rejected above; 'formula'/'callback' are not
        # supported here, matching precedent from the other B3 batches);
        # was_dph excluded (needs a different, deferred quotient-rule
        # contraction -- native DPH, is_discrete=True/was_dph=False, needs
        # NO special-casing and is NOT excluded, confirmed neither
        # ComputeSojournTimesFfiImpl nor ptd_expected_sojourn_time_subset
        # branch on is_discrete); observed_indices baked mode excluded
        # (deferred -- would need a scatter-add of the upstream cotangent by
        # the inverse-index map before the quotient rule, a separate piece
        # of work). See b3-joint-index-plan.md.
        #
        # No silent fallback: whenever FD ends up being used instead --
        # exact_grad=False passed explicitly, weight_mode/was_dph/baked-mode
        # out of scope (checked once here, statically), or a per-call
        # decline inside _exact_sojourn_jac_np below (e.g. MPFR
        # conditioning or an out-of-scope tape input) -- an INFO-level log
        # line via get_logger(__name__) states why.
        _jix_grad_logger = get_logger(__name__)
        _jix_was_dph = bool(graph.get_was_dph())
        _jix_linear_scope_ok = (_wm in (None, 'linear'))
        if not bool(exact_grad):
            _jix_grad_logger.info(
                "pmf_from_graph_joint_index: exact_grad=False -- using "
                "finite differences for the sojourn gradient."
            )
            _jix_exact_enabled = False
        elif not _jix_linear_scope_ok:
            _jix_grad_logger.info(
                "pmf_from_graph_joint_index: exact sojourn gradient not "
                "available for weight_mode=%r (only 'linear'/None is "
                "supported) -- using finite differences.", _wm
            )
            _jix_exact_enabled = False
        elif _jix_was_dph:
            _jix_grad_logger.info(
                "pmf_from_graph_joint_index: exact sojourn gradient not "
                "yet supported for a was_dph graph (Graph.discretize()) -- "
                "using finite differences. Native DPH (is_discrete=True, "
                "was_dph=False) IS supported."
            )
            _jix_exact_enabled = False
        elif int(graph.param_length()) != param_length_actual:
            # The C function reads graph->param_length off the CLONE (built
            # from the ORIGINAL graph object, not the serialized/theta_dim-
            # overridden dict); if a caller passed theta_dim != the graph's
            # own param_length, the clone's Jacobian width would silently
            # disagree with param_length_actual (used for every reshape/
            # ShapeDtypeStruct below) -- decline explicitly rather than let
            # that surface as a reshape error inside a pure_callback (found
            # via adversarial review of the implemented fix).
            _jix_grad_logger.info(
                "pmf_from_graph_joint_index: exact sojourn gradient not "
                "supported when theta_dim (%r) overrides the graph's own "
                "param_length (%r) -- using finite differences.",
                param_length_actual, int(graph.param_length())
            )
            _jix_exact_enabled = False
        else:
            _jix_exact_enabled = True

        _exact_sojourn_jac_np = None
        _jix_probed_ok = False  # Batch F: latched ONCE by the construction-
                                # time probe below; model_bwd dispatches on a
                                # plain Python bool (never a traced predicate,
                                # so vmap/jit trace only the chosen branch).
        if _jix_exact_enabled:
            _jix_exact_graph = graph.clone()
            _jix_param_length = param_length_actual
            _jix_all_terminal_np = all_terminal_indices_np
            _jix_n_vertices = int(graph.vertices_length())

            def _exact_sojourn_jac_np(theta_np, vertex_indices_np):
                # Host callback: set the private clone's weights at theta and
                # read the exact sojourn-gradient Jacobian d[sojourn]/dtheta
                # from C, gathered at BOTH the observed and all-terminal
                # index sets via a SINGLE call over their union (see the
                # plan's "Wiring" section for why one call over a union beats
                # two calls over the parts). vertex_indices is a genuine
                # per-call argument here (unlike theta_dim/fixed_mask), since
                # the non-baked model reads it from the runtime observed
                # data, not from a construction-time constant -- so the
                # union must be computed INSIDE the callback (on concrete
                # values), not precomputed in Python before tracing.
                # Batch F (committed-path semantics): once the construction-
                # time probe has succeeded, there is NO per-call FD fallback
                # -- a decline here RAISES with a diagnostic message (the
                # user-decided failure mode; verified legible under
                # vmap/jit by dr_batchF_jit_raise_derisk.py).
                th = np.asarray(theta_np, dtype=np.float64)
                vi = np.asarray(vertex_indices_np, dtype=np.int64).ravel()

                # Accurate error for bad indices BEFORE blaming conditioning
                # (D6.1 review: indices[r] >= n is an index-dependent C
                # decline the probe cannot see).
                if vi.size and (vi.min() < 0 or vi.max() >= _jix_n_vertices):
                    raise ValueError(
                        "pmf_from_graph_joint_index: observed vertex index "
                        f"out of range (got min={int(vi.min())}, "
                        f"max={int(vi.max())}; graph has {_jix_n_vertices} "
                        "vertices) -- check the observed data / index "
                        "mapping."
                    )

                _jix_exact_graph.update_weights(th.tolist())
                if _baked:
                    # Batch E hoist: the baked index set is static, so the
                    # union and both position maps are construction-time
                    # constants (see _baked_union_np below).
                    union_idx = _baked_union_np
                else:
                    union_idx = np.union1d(vi, _jix_all_terminal_np)
                raw = _jix_exact_graph._sojourn_grad_theta_subset(union_idx.tolist())
                if not raw and exact_grad_decline == 'fd':
                    # Batch E (user decision 2026-08-14): HOST-side
                    # per-particle central-difference fallback. Chosen for
                    # the svgd entry point because SVGD's wide particle
                    # init routinely visits thetas the conditioning gate
                    # declines -- under raise semantics ONE such particle
                    # kills the whole cloud, and lifting the gate returns
                    # silently unreliable numbers at extreme theta ratios
                    # (measured 34-144% off tight FD at ratios >= 1e8).
                    # The fallback = RELATIVE-step central FD on the raw
                    # sojourn values + the exact quotient rule downstream
                    # (better-conditioned than _fd_theta_bar's absolute-eps
                    # FD of the normalized forward); each event is
                    # WARN-logged. Unlike a
                    # JAX-level fallback (impossible per-particle under
                    # vmap -- the D6 record), the host callback runs
                    # per-particle sequentially, so this is exact for
                    # every computable particle and FD only at declined
                    # ones.
                    _jix_grad_logger.warning(
                        "pmf_from_graph_joint_index: exact sojourn "
                        "gradient declined at theta=%s -- using a "
                        "host-side finite-difference fallback for THIS "
                        "particle only (exact_grad_decline='fd').",
                        th.tolist(),
                    )
                    J_union = np.empty(
                        (union_idx.shape[0], _jix_param_length))
                    for _k in range(_jix_param_length):
                        _h = max(abs(float(th[_k])) * 1e-6, 1e-10)
                        _tp = th.copy(); _tp[_k] += _h
                        _tm = th.copy(); _tm[_k] -= _h
                        _jix_exact_graph.update_weights(_tp.tolist())
                        _sp = np.asarray(
                            _jix_exact_graph.expected_sojourn_time(
                                union_idx.tolist()))
                        _jix_exact_graph.update_weights(_tm.tolist())
                        _sm = np.asarray(
                            _jix_exact_graph.expected_sojourn_time(
                                union_idx.tolist()))
                        J_union[:, _k] = (_sp - _sm) / (2.0 * _h)
                    if _baked:
                        return (J_union[_baked_obs_pos_np],
                                J_union[_baked_all_pos_np])
                    obs_pos = np.searchsorted(union_idx, vi)
                    all_pos = np.searchsorted(union_idx, _jix_all_terminal_np)
                    return J_union[obs_pos], J_union[all_pos]
                if not raw:
                    raise RuntimeError(
                        "pmf_from_graph_joint_index: exact sojourn gradient "
                        f"(exact_grad=True) declined at theta={th.tolist()}. "
                        "The construction-time probe (theta=ones, over the "
                        "all-terminal index union) succeeded, so the likely "
                        "causes are: an ill-conditioned elimination tape at "
                        "THIS theta (MPFR gate), a non-finite Jacobian row "
                        "at one of the REQUESTED observed vertices, or an "
                        "allocation failure. No automatic finite-difference "
                        "fallback exists once the exact path is committed -- "
                        "pass exact_grad=False, pass exact_grad_decline='fd' "
                        "for a per-call host-side FD fallback, or "
                        "investigate this theta/"
                        "index set. (A NaN/inf theta raises a ValueError "
                        "from update_weights before reaching this point.)"
                    )

                J_union = np.asarray(raw, dtype=np.float64).reshape(
                    union_idx.shape[0], _jix_param_length,
                )
                if _baked:
                    return (J_union[_baked_obs_pos_np],
                            J_union[_baked_all_pos_np])
                obs_pos = np.searchsorted(union_idx, vi)
                all_pos = np.searchsorted(union_idx, _jix_all_terminal_np)
                return J_union[obs_pos], J_union[all_pos]

            # Construction-time probe (Batch F, amendment 1): theta=ones over
            # union(all_terminal, [0]) -- the construction-known part of
            # EVERY future call's index union. This generalizes the
            # topology-only decline reasons (out-of-scope tape input, size
            # guard) to all future theta AND catches structurally non-finite
            # Jacobian rows at the terminal vertices (trap/deficit-sink
            # class), which a single-index probe would miss. A probe failure
            # is treated exactly like a static exclusion: whole-model FD, no
            # wiring overhead, never re-tried.
            if _baked:
                # Batch E: the baked probe is EXACT -- probe set == every
                # future call's set (union of the static unique observed
                # indices with all_terminal; no runtime index rows exist,
                # so F's [0] residual member is subsumed). The hoisted
                # position maps serve the per-call callback above.
                _baked_union_np = np.union1d(_uniq_idx_np.astype(np.int64),
                                             _jix_all_terminal_np)
                _baked_obs_pos_np = np.searchsorted(
                    _baked_union_np, _uniq_idx_np.astype(np.int64))
                _baked_all_pos_np = np.searchsorted(
                    _baked_union_np, _jix_all_terminal_np)
                _probe_union = _baked_union_np
            else:
                _baked_union_np = None
                _baked_obs_pos_np = None
                _baked_all_pos_np = None
                _probe_union = np.union1d(
                    np.asarray([0], dtype=np.int64), _jix_all_terminal_np)
            _probe_theta = np.ones(_jix_param_length, dtype=np.float64)
            _jix_exact_graph.update_weights(_probe_theta.tolist())
            _probe_raw = _jix_exact_graph._sojourn_grad_theta_subset(
                _probe_union.tolist())
            if (_probe_raw
                    and len(_probe_raw) == _probe_union.shape[0] * _jix_param_length):
                _jix_probed_ok = True
            elif not _probe_raw:
                _jix_grad_logger.info(
                    "pmf_from_graph_joint_index: exact sojourn gradient "
                    "declined at the construction-time probe (theta=ones, "
                    "all-terminal union) -- an out-of-scope/oversized tape, "
                    "a structurally non-finite Jacobian row at a terminal "
                    "vertex, or ill-conditioning even at this benign "
                    "reference point -- using finite differences for the "
                    "whole model. (theta=ones is the documented reference; "
                    "if your model is only well-conditioned elsewhere, this "
                    "is a known limitation of the static probe.)"
                )
            else:
                _jix_grad_logger.info(
                    "pmf_from_graph_joint_index: construction-time probe "
                    "returned an unexpected length (%d, expected %d) -- "
                    "using finite differences for the whole model.",
                    len(_probe_raw),
                    _probe_union.shape[0] * _jix_param_length,
                )

        # Callback weight mode: the C++ JSON parser only knows 'linear'
        # and 'log'. For 'callback', apply the user's weight_callback
        # per-theta to materialise concrete edge weights and route
        # through pybind's `Graph.expected_sojourn_time`. Mirrors the
        # callback handling in `pmf_and_moments_from_graph`.
        if serialized.get('weight_mode') == 'callback':
            import json
            from .ffi_wrappers import _make_json_serializable
            from . import phasic_pybind as cpp_module

            weight_callback = graph.weight_callback
            if weight_callback is None:
                raise ValueError(
                    "Graph has weight_mode='callback' but no "
                    "weight_callback set. Set graph.weight_callback "
                    "before calling pmf_from_graph_joint_index() / "
                    "joint-prob Graph.svgd()."
                )

            _serialized = serialized

            def _compute_callback(theta_np, vertex_indices_np):
                concrete = _apply_weight_callback(
                    _serialized, theta_np, weight_callback,
                )
                json_str = json.dumps(_make_json_serializable(concrete))
                builder_local = cpp_module.parameterized.GraphBuilder(json_str)
                concrete_graph = builder_local.build(np.zeros(0))
                # Sojourn times for the observed-vertex subset.
                obs_idx = [int(v) for v in vertex_indices_np.tolist()]
                obs_sojourn = np.asarray(
                    concrete_graph.expected_sojourn_time(obs_idx),
                    dtype=np.float64,
                )
                all_sojourn = np.asarray(
                    concrete_graph.expected_sojourn_time(
                        all_terminal_indices_np.tolist(),
                    ),
                    dtype=np.float64,
                )
                return obs_sojourn, all_sojourn

            if _baked:
                # Baked mode: ignore runtime vertex_indices; the FFI
                # callback works on the precomputed unique indices and
                # the result is scattered back via _inverse_idx_jnp.
                _baked_uniq_for_cb = _uniq_idx_np  # captured by closure

                def _compute_pure(theta, _vertex_indices_ignored):
                    theta = jnp.atleast_1d(theta)
                    uniq_shape = jax.ShapeDtypeStruct(
                        (int(_baked_uniq_for_cb.size),), jnp.float64,
                    )
                    all_shape = jax.ShapeDtypeStruct(
                        (int(all_terminal_indices_np.size),), jnp.float64,
                    )

                    def _cb(t):
                        return _compute_callback(
                            np.asarray(t, dtype=np.float64),
                            _baked_uniq_for_cb,
                        )

                    uniq_sojourn, all_sojourn = jax.pure_callback(
                        _cb, (uniq_shape, all_shape),
                        theta,
                        vmap_method='sequential',
                    )
                    normalization_constant = jnp.sum(all_sojourn)
                    uniq_probs = uniq_sojourn / normalization_constant
                    # Scatter from (n_unique,) back to (n_obs,) so that
                    # downstream _log_lik_from_pmf sees the same per-obs
                    # shape it always did. inverse_idx[i] = position of
                    # obs i in the unique array.
                    sojourn_probs = uniq_probs[_inverse_idx_jnp]
                    dummy_moments = jnp.zeros(2)
                    return sojourn_probs, dummy_moments
            else:
                def _compute_pure(theta, vertex_indices):
                    theta = jnp.atleast_1d(theta)
                    vertex_indices = jnp.atleast_1d(vertex_indices).astype(jnp.int32)
                    obs_shape = jax.ShapeDtypeStruct(
                        vertex_indices.shape, jnp.float64,
                    )
                    all_shape = jax.ShapeDtypeStruct(
                        (int(all_terminal_indices_np.size),), jnp.float64,
                    )

                    def _cb(t, vi):
                        return _compute_callback(
                            np.asarray(t, dtype=np.float64),
                            np.asarray(vi, dtype=np.int32),
                        )

                    obs_sojourn, all_sojourn = jax.pure_callback(
                        _cb, (obs_shape, all_shape),
                        theta, vertex_indices,
                        vmap_method='sequential',
                    )
                    normalization_constant = jnp.sum(all_sojourn)
                    sojourn_probs = obs_sojourn / normalization_constant
                    dummy_moments = jnp.zeros(2)
                    return sojourn_probs, dummy_moments

        else:
            if _baked:
                # Baked mode: precompute unique indices + inverse map at
                # construction; wrap the two FFI calls in custom_vmap rules
                # so that under SVGD's vmap(grad(loss))(particles), each
                # FD-perturbation dispatch fuses into ONE batched FFI call
                # of theta shape (P, theta_dim) instead of P serial calls.
                # Mirrors the no-exposure branch of _daisy_chain_svgd_model.
                from jax import custom_batching as _cb_jix

                @_cb_jix.custom_vmap
                def _obs_forward(theta_flat):
                    return compute_sojourn_times_ffi(
                        structure_dict, theta_flat, _uniq_idx_jnp,
                    )

                @_obs_forward.def_vmap
                def _obs_forward_vmap_rule(axis_size, in_batched, theta_flat):
                    # theta_flat: (axis_size, theta_dim). Dispatch as one
                    # batched FFI call. The C handler parallelises across
                    # the batch dim with OpenMP (within the per-call thread
                    # cap from ComputeSojournTimesFfiImpl).
                    del axis_size, in_batched
                    return (
                        compute_sojourn_times_ffi(
                            structure_dict, theta_flat, _uniq_idx_jnp,
                        ),
                        True,  # batched along axis 0
                    )

                @_cb_jix.custom_vmap
                def _all_forward(theta_flat):
                    return compute_sojourn_times_ffi(
                        structure_dict, theta_flat, all_terminal_indices,
                    )

                @_all_forward.def_vmap
                def _all_forward_vmap_rule(axis_size, in_batched, theta_flat):
                    del axis_size, in_batched
                    return (
                        compute_sojourn_times_ffi(
                            structure_dict, theta_flat, all_terminal_indices,
                        ),
                        True,
                    )

                def _compute_pure(theta, _vertex_indices_ignored):
                    """Baked-dedup FFI path. The runtime vertex_indices
                    argument is ignored — the unique indices precomputed
                    from observed_indices at construction are used instead,
                    and the result is scattered back to n_obs shape via the
                    inverse-index mapping."""
                    theta = jnp.atleast_1d(theta)

                    # Compute sojourn at unique observed vertices (small k).
                    uniq_sojourn = _obs_forward(theta)

                    # Normalisation over all terminal vertices.
                    all_sojourn = _all_forward(theta)
                    normalization_constant = jnp.sum(all_sojourn)

                    # Probabilities at unique vertices, then scatter back.
                    uniq_probs = uniq_sojourn / normalization_constant
                    sojourn_probs = uniq_probs[_inverse_idx_jnp]

                    # Dummy moments (not supported in joint_index mode)
                    dummy_moments = jnp.zeros(2)

                    return sojourn_probs, dummy_moments
            else:
                def _compute_pure(theta, vertex_indices):
                    """Pure computation using FFI for memory-efficient subset computation."""
                    theta = jnp.atleast_1d(theta)
                    vertex_indices = jnp.atleast_1d(vertex_indices).astype(jnp.int32)

                    # Compute sojourn times for observed vertices
                    sojourn_times = compute_sojourn_times_ffi(structure_dict, theta, vertex_indices)

                    # Compute normalization constant using ALL terminal vertices
                    # This is the total probability mass in the modeled state space
                    all_sojourn_times = compute_sojourn_times_ffi(structure_dict, theta, all_terminal_indices)
                    normalization_constant = jnp.sum(all_sojourn_times)

                    # Normalize to get conditional probabilities P(obs | obs ∈ modeled space)
                    # This correctly handles the deficit without biasing toward θ values
                    # that minimize deficit
                    sojourn_probs = sojourn_times / normalization_constant

                    # Dummy moments (not supported in joint_index mode)
                    dummy_moments = jnp.zeros(2)

                    return sojourn_probs, dummy_moments

        @jax.custom_vjp
        def model(theta, vertex_indices, rewards=None):
            """JAX-compatible model for joint index distributions.

            Parameters
            ----------
            theta : jax.Array
                Parameter vector
            vertex_indices : jax.Array
                Array of vertex indices (integers)
            rewards : None
                Ignored (not supported in joint_index mode)

            Returns
            -------
            tuple of (jax.Array, jax.Array)
                (converged_visits, dummy_moments)
            """
            return _compute_pure(theta, vertex_indices)

        def model_fwd(theta, vertex_indices, rewards=None):
            visits, moments = _compute_pure(theta, vertex_indices)
            return (visits, moments), (theta, vertex_indices)

        # Convert fixed_mask to a Python set of fixed indices for use in model_bwd
        # This avoids JAX tracing issues with boolean comparisons on arrays
        fixed_indices_set = set()
        if fixed_mask is not None:
            fixed_mask_np = np.asarray(fixed_mask)
            fixed_indices_set = set(np.where(fixed_mask_np == 1)[0].tolist())

        def model_bwd(res, g):
            theta, vertex_indices = res
            g_visits, g_moments = g  # Unpack gradient tuple

            n_params = theta.shape[0]
            eps = 1e-7

            def _fd_theta_bar():
                # Finite differences for gradient (unchanged from before this
                # batch). Skip perturbation for fixed parameters -- gradient
                # is zero; critical for correct chain rule when using
                # compiled_grad_reduced.
                tb = []
                for i in range(n_params):
                    if i in fixed_indices_set:
                        tb.append(jnp.zeros((), dtype=theta.dtype))
                        continue
                    theta_plus = theta.at[i].add(eps)
                    theta_minus = theta.at[i].add(-eps)
                    visits_plus, _ = _compute_pure(theta_plus, vertex_indices)
                    visits_minus, _ = _compute_pure(theta_minus, vertex_indices)
                    # Gradient only from visits (moments are dummy zeros)
                    grad_i = jnp.sum(g_visits * (visits_plus - visits_minus) / (2 * eps))
                    tb.append(grad_i)
                return jnp.stack(tb)

            if _exact_sojourn_jac_np is None or not _jix_probed_ok:
                # Static exclusion OR construction-time probe failure: pure
                # FD, no exact-path wiring traced at all.
                theta_bar = _fd_theta_bar()
            else:
                # Quotient rule: sojourn_probs[i] = obs_sojourn[i] / norm,
                # norm = sum(all_sojourn).
                # Batch F: the dispatch above is a plain Python `if` on
                # construction-time bools, NOT a traced `lax.cond` -- under
                # SVGD's vmap(jit(grad(loss)))(particles) a batched cond
                # predicate lowers to a select that computes BOTH branches
                # (FD *plus* exact, every step; empirically confirmed in
                # dr_lax_cond_vmap_derisk.py), whereas a trace-time Python
                # `if` traces only the chosen branch. The price: once
                # committed, a per-theta decline has no automatic FD
                # fallback -- the host callback RAISES (user-decided;
                # legibility under vmap/jit verified in
                # dr_batchF_jit_raise_derisk.py).
                # Normalize to 1-D, matching _compute_pure's own
                # jnp.atleast_1d handling -- res[1] (vertex_indices) is
                # whatever raw shape the caller passed to model(), and the
                # callback below always returns a raveled (k_obs, P) array,
                # so the ShapeDtypeStruct declared for it must match the
                # SAME normalized shape or a non-1-D vertex_indices would
                # raise inside the pure_callback (found via adversarial
                # review of the implemented fix).
                if _baked:
                    # Batch E: the forward gathers uniq->obs, so the VJP
                    # SCATTER-ADDS the cotangent obs->uniq, and the same
                    # quotient rule then runs at unique granularity with
                    # the STATIC index set (probe set == call set).
                    _vi_norm = _uniq_idx_jnp
                    g_contract = jnp.zeros(
                        _uniq_idx_jnp.shape[0], dtype=g_visits.dtype,
                    ).at[_inverse_idx_jnp].add(g_visits)
                else:
                    _vi_norm = jnp.atleast_1d(vertex_indices).astype(jnp.int32)
                    g_contract = g_visits
                all_sojourn_exact = compute_sojourn_times_ffi(
                    structure_dict, theta, all_terminal_indices)
                norm_exact = jnp.sum(all_sojourn_exact)
                obs_sojourn_exact = compute_sojourn_times_ffi(
                    structure_dict, theta, _vi_norm)

                J_obs, J_all = jax.pure_callback(
                    _exact_sojourn_jac_np,
                    (jax.ShapeDtypeStruct(_vi_norm.shape + (n_params,), jnp.float64),
                     jax.ShapeDtypeStruct(all_terminal_indices.shape + (n_params,), jnp.float64)),
                    theta, _vi_norm,
                    vmap_method='sequential',
                )
                dnorm_exact = jnp.sum(J_all, axis=0)
                d_probs_exact = (
                    J_obs * norm_exact - obs_sojourn_exact[:, None] * dnorm_exact[None, :]
                ) / (norm_exact ** 2)
                exact_tbm = d_probs_exact.T @ g_contract  # (n_params,)
                if fixed_indices_set:
                    _fixed_keep = jnp.array(
                        [0.0 if i in fixed_indices_set else 1.0 for i in range(n_params)],
                        dtype=exact_tbm.dtype,
                    )
                    exact_tbm = exact_tbm * _fixed_keep

                theta_bar = exact_tbm

            return theta_bar, None, None  # gradients for theta, vertex_indices, rewards

        model.defvjp(model_fwd, model_bwd)
        # Source graph reference (see ``pmf_and_moments_from_graph``).
        model._source_graph = graph
        # Preconditioner source: this joint-index model returns DUMMY zeros as
        # its second (moments) output, so the moment Jacobian would degenerate.
        # Precondition on the theta-dependent FIRST (probability) output instead
        # (ProbabilityJacobianPreconditioner). See SVGD.optimize dispatch.
        model._precondition_output = 'probability'
        return model

    @classmethod
    def pmf_and_moments_from_graph_multivariate(cls, graph: Graph, nr_moments: int = 2,
                                                discrete: bool = False, use_ffi: bool = False,
                                                theta_dim: int | None = None,
                                                fixed_mask: Any = None) -> Callable:
        """
        Create a multivariate phase-type model that handles 2D observations and rewards.

        This wrapper enables computing joint likelihoods for multivariate phase-type distributions
        where each feature dimension has its own reward vector defining the marginal distribution.

        Parameters
        ----------
        graph : Graph
            Parameterized graph built using the Python API with parameterized edges.
        nr_moments : int, default=2
            Number of moments to compute per feature dimension
        discrete : bool, default=False
            If True, computes discrete PMF. If False, computes continuous PDF.
        use_ffi : bool, default=False
            If True, uses Foreign Function Interface approach.
        theta_dim : int, optional
            Number of parameters for parameterized edges.

        Returns
        -------
        callable
            JAX-compatible function with signature:
            model(theta, times, rewards=None) -> (pmf_values, moments)

            Where:
            - theta: Parameter vector (theta_dim,)
            - times: Time points - can be 1D (n_times,) or 2D (n_times, n_features)
            - rewards: Reward vectors - can be:
              * None: Standard moments
              * 1D (n_vertices,): Single reward vector (backward compatible)
              * 2D (n_vertices, n_features): Multivariate - one reward per feature
            - pmf_values: PMF/PDF values - shape matches input:
              * 1D rewards → 1D output (n_times,)
              * 2D rewards → 2D output (n_times, n_features)
            - moments: Moment values:
              * 1D rewards → 1D output (nr_moments,)
              * 2D rewards → 2D output (n_features, nr_moments)

        Examples
        --------
        >>> # Create parameterized model
        >>> graph = Graph(coalescent, nr_samples=3)
        >>> model = Graph.pmf_and_moments_from_graph_multivariate(graph, nr_moments=2)
        >>>
        >>> # 1D case (backward compatible)
        >>> theta = jnp.array([0.5])
        >>> times = jnp.array([1.0, 2.0, 3.0])
        >>> rewards_1d = jnp.array([1.0, 2.0, 0.5, 1.5])  # (n_vertices,)
        >>> pmf_vals, moments = model(theta, times, rewards_1d)
        >>> print(pmf_vals.shape)  # (3,)
        >>> print(moments.shape)   # (2,)
        >>>
        >>> # 2D case (multivariate)
        >>> rewards_2d = jnp.array([[1.0, 2.0, 0.5, 1.5],   # Feature 1 reward vector
        ...                          [0.5, 1.0, 2.0, 0.8]])  # Feature 2 reward vector
        ...                                                  # (n_features, n_vertices)
        >>> times_2d = jnp.array([[1.0, 1.5],
        ...                        [2.0, 2.5],
        ...                        [3.0, 3.5]])  # (n_times, n_features)
        >>> pmf_vals, moments = model(theta, times_2d, rewards_2d)
        >>> print(pmf_vals.shape)  # (3, 2)
        >>> print(moments.shape)   # (2, 2)
        >>>
        >>> # Use in SVGD with 2D observations
        >>> observed_data = jnp.array([[1.5, 2.1], [0.8, 1.2], [2.3, 3.1]])
        >>> svgd = SVGD(model, observed_data, theta_dim=1, rewards=rewards_2d)
        >>> results = svgd.optimize()

        Notes
        -----
        - For 2D rewards, each feature dimension is computed independently using the
          corresponding row of the rewards matrix (rewards[j, :] for feature j)
        - Log-likelihood is computed as sum over all observation elements
        - Backward compatible: 1D rewards behave exactly as pmf_and_moments_from_graph()
        """
        # Check if JAX is available
        try:
            _ensure_jax_active()
        except Exception as _e:
            raise ImportError(
                "JAX is required for multivariate models. Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            ) from _e

        import jax
        import jax.numpy as jnp

        # Get the 1D model (forward fixed_mask so its FD VJP skips fixed dims)
        model_1d = cls.pmf_and_moments_from_graph(
            graph, nr_moments=nr_moments, discrete=discrete,
            use_ffi=use_ffi, theta_dim=theta_dim, fixed_mask=fixed_mask
        )

        def model_multivariate(theta, times, rewards=None):
            """Multivariate wrapper handling 1D and 2D rewards, including sparse observations."""
            from .svgd import SparseObservations, is_sparse_observations

            # Auto-detect dimensionality
            if rewards is None:
                # No rewards - use 1D model directly
                return model_1d(theta, times, rewards=None)

            rewards_arr = jnp.asarray(rewards)

            if rewards_arr.ndim == 1:
                # Backward compatible: 1D rewards
                return model_1d(theta, times, rewards=rewards_arr)

            elif rewards_arr.ndim == 2:
                # 2D case: Loop over features
                # Using Python loop here to avoid JAX FFI dtype issues with scan
                n_features = rewards_arr.shape[0]

                # Check if times is in sparse observation format
                if is_sparse_observations(times):
                    # Sparse format: extract observations per feature using pre-computed slices
                    pmf_list = []
                    moments_list = []

                    for j in range(n_features):
                        # Extract reward vector for feature j
                        reward_j = rewards_arr[j, :].astype(jnp.float64)

                        # Extract observations for this feature using JAX-compatible method
                        times_j = times.get_feature_values(j)

                        # Skip if no observations for this feature
                        if times_j.shape[0] == 0:
                            # Return empty PMF and moments with NaN
                            pmf_list.append(jnp.array([]))
                            moments_list.append(jnp.full(nr_moments, jnp.nan))
                            continue

                        pmf_j, moments_j = model_1d(theta, times_j, rewards=reward_j)
                        pmf_list.append(pmf_j)
                        moments_list.append(moments_j)

                    # For sparse: concatenate PMFs (not stack - different lengths per feature)
                    # Return as flat array of PMF values in same order as input
                    pmf = jnp.concatenate(pmf_list) if any(len(p) > 0 for p in pmf_list) else jnp.array([])
                    moments = jnp.stack(moments_list, axis=0)  # (n_features, nr_moments)

                    return pmf, moments

                else:
                    # Dense format: existing behavior
                    times_arr = jnp.asarray(times)
                    pmf_list = []
                    moments_list = []

                    for j in range(n_features):
                        # Extract reward vector for feature j (ensure float64 for C++ compatibility)
                        reward_j = rewards_arr[j, :].astype(jnp.float64)

                        # Extract times for feature j (support both 1D and 2D times)
                        # NaN times are handled by C++ layer - they return NaN in output
                        if times_arr.ndim == 2:
                            times_j = times_arr[:, j]
                        else:
                            times_j = times_arr  # Broadcast same times to all features

                        pmf_j, moments_j = model_1d(theta, times_j, rewards=reward_j)

                        pmf_list.append(pmf_j)
                        moments_list.append(moments_j)

                    # Stack results
                    pmf = jnp.stack(pmf_list, axis=1)  # (n_times, n_features)
                    moments = jnp.stack(moments_list, axis=0)  # (n_features, nr_moments)

                    return pmf, moments

            else:
                raise ValueError(
                    f"Rewards must be 1D (n_vertices,) or 2D (n_features, n_vertices). "
                    f"Got shape: {rewards_arr.shape}"
                )

        # Forward `_cdf_zero_fn` from the underlying 1D model so the
        # zero-inflated likelihood path in `_attach_zero_inflated_term`
        # can use it. Without this, partial-coverage 2D-rewards SVGD
        # falls back to `compute_reward_visit_probability_ffi`, which
        # can't parse a `weight_mode='callback'` JSON.
        if hasattr(model_1d, '_cdf_zero_fn'):
            model_multivariate._cdf_zero_fn = model_1d._cdf_zero_fn

        # Source graph reference (see ``pmf_and_moments_from_graph``).
        model_multivariate._source_graph = graph
        # Preconditioner source: real moments output (see pmf_and_moments_from_graph).
        model_multivariate._precondition_output = 'moments'
        return model_multivariate


    # def plot(self, *args: Any, **kwargs: Any) -> Any:
    #     """
    #     Plots the graph using graphviz. See plot::plot_graph.py for more details.

    #     Returns
    #     -------
    #     :
    #         _description_
    #     """
    #     return plot.plot_graph(self, *args, **kwargs)

    # Plotting methods live in _graph_plotting.py (WS-C god-object split); assigned
    # here as class attributes so they stay direct Graph members (docs/introspection).
    plot = _graph_plotting.plot
    plot_scc_decomp = _graph_plotting.plot_scc_decomp

    def profile(self, theta: Any = None, probe_dyn: bool | str = "auto") -> Any:
        """Profile this graph and recommend ``parallel_elimination``,
        ``dyn_ordering``, and the evaluation path (forward-PDF vs joint/sojourn).

        Thin wrapper around :func:`phasic.profile_graph`; see that function for
        details. Returns a :class:`~phasic.profile.GraphProfile`.

        >>> print(graph.profile())
        """
        from .profile import profile_graph
        return profile_graph(self, theta=theta, probe_dyn=probe_dyn)

    def copy(self) -> Self:
        """
        Returns a deep copy of the graph.

        Creates an independent copy with all vertices, edges, and metadata.
        Modifications to the copy will not affect the original graph.

        Returns
        -------
        Graph
            Deep copy of the graph
        """
        return self.clone()  # clone() already wraps in Graph(), don't double-wrap!

        # """
        # Takes a graph for a continuous distribution and turns
        # it into a descrete one (inplace). Returns a matrix of
        # rewards for computing marginal moments
        # """

    def clone(self) -> Graph:
        """Create a deep copy of this graph.

        The cloned graph preserves the cache_trace setting but starts
        with a fresh (invalid) trace cache.

        Returns
        -------
        Graph
            A new Graph instance with the same structure.
        """
        # super().clone() returns C++ _Graph, wrap it in Python Graph
        cloned = Graph(super().clone())
        cloned.is_discrete = self.is_discrete
        cloned._cache_trace = self._cache_trace
        # Carry the weight mode/callback/formula tape; the C-level clone
        # does not copy the tape and Graph(_Graph) resets to 'linear'.
        _propagate_weight_state(self, cloned)
        # Don't copy trace - clone starts fresh
        cloned._trace = None
        cloned._trace_dirty = True
        cloned._last_theta = None
        return cloned

    def _rebuild_with_wider_layout(self, extra_state_dims=0, extra_coeff_slots=0,
                                    state_fill=0, coeff_fill=0.0) -> 'Graph':
        """Rebuild graph with wider state vectors and/or coefficient arrays.

        Creates a new graph that is structurally identical to self but with
        additional state dimensions and/or coefficient slots appended to each
        vertex state and edge coefficient array respectively.

        Parameters
        ----------
        extra_state_dims : int
            Number of extra dimensions to append to each vertex state vector.
        extra_coeff_slots : int
            Number of extra slots to append to each edge coefficient array.
        state_fill : int
            Fill value for the new state dimensions (default 0).
        coeff_fill : float
            Fill value for the new coefficient slots (default 0.0).

        Returns
        -------
        Graph
            A new graph with wider layout. Python metadata is copied from self.
        """
        new_state_length = self.state_length() + extra_state_dims
        new_param_length = self.param_length() + extra_coeff_slots

        new_graph = Graph(new_state_length)

        # Set param_length BEFORE adding any edges
        if new_param_length > 0:
            new_graph.set_param_length(new_param_length)

        pad_state = np.full(extra_state_dims, state_fill, dtype=int) if extra_state_dims > 0 else np.array([], dtype=int)
        pad_coeff = [coeff_fill] * extra_coeff_slots

        # Map: old vertex index -> new vertex object
        vertex_map = {}

        # Starting vertex maps to new starting vertex
        vertex_map[self.starting_vertex().index()] = new_graph.starting_vertex()

        # Create all non-starting vertices first
        for vertex in self.vertices():
            idx = vertex.index()
            if idx == self.starting_vertex().index():
                continue
            new_state = np.append(vertex.state(), pad_state).astype(int)
            new_vertex = new_graph.find_or_create_vertex(new_state)
            vertex_map[idx] = new_vertex

        # Copy all edges with padded coefficients
        for vertex in self.vertices():
            old_idx = vertex.index()
            new_vertex = vertex_map[old_idx]
            if self.parameterized():
                for edge in vertex.parameterized_edges():
                    coeffs = list(edge.edge_state(self.param_length())) + pad_coeff
                    new_vertex.add_edge(vertex_map[edge.to().index()], coeffs)
            else:
                for edge in vertex.edges():
                    new_vertex.add_edge(vertex_map[edge.to().index()], edge.weight())

        # Copy Python metadata
        new_graph._callback = self._callback
        new_graph._callback_kwargs = self._callback_kwargs.copy() if self._callback_kwargs else {}
        # Carry the full weight state (mode + callback + formula tape).
        # Previously only mode+callback were copied, leaving a formula graph
        # in mode='formula' with tape=None (raises on serialize / silently
        # reverts to linear). Extra coeff slots are appended, so the tape's
        # original coefficient indices remain valid.
        _propagate_weight_state(self, new_graph)
        new_graph.is_discrete = self.is_discrete
        new_graph._last_callback_vertices_length = new_graph.vertices_length()

        return new_graph

    def compute_trace(self, param_length: int | None = None,
                     hierarchical: bool = True,
                     min_size: int = 50,
                     parallel: str = 'auto',
                     verbose: bool = False,
                     force: bool = False) -> EliminationTrace:
        """
        Compute elimination trace with optional hierarchical caching.

        .. deprecated::
            The Python ``EliminationTrace`` machinery is no longer
            wired to the public ``moments()`` / ``expectation()`` /
            ``variance()`` entry points (those route directly to the
            C++ implementation, which uses the Stage A0-cached
            ``parameterized_reward_compute_graph``). ``compute_trace``
            is preserved for callers that still drive the trace
            pipeline directly, but emits a ``DeprecationWarning`` and
            will be removed in a future release.

        When Graph was created with cache_trace=True, the trace is cached on the
        instance for use by moments(), expectation(), etc. In this mode, the operation
        is NON-DESTRUCTIVE (graph is preserved via cloning).

        When Graph was created with cache_trace=False (default), the operation is
        DESTRUCTIVE and will empty the graph during trace recording.

        Parameters
        ----------
        param_length : int, optional
            Number of parameters (auto-detect if None)
        hierarchical : bool, default=True
            If True, use hierarchical SCC-based caching (recommended).
            If False, use direct trace recording without caching.
            Caching provides 10-100x speedup on repeated calls.
        min_size : int, default=50
            Minimum vertices to subdivide (only used if hierarchical=True)
        parallel : str, default='auto'
            Parallelization: 'auto', 'vmap', 'pmap', or 'sequential'
        verbose : bool, default=False
            If True, show progress bars for major computation stages
        force : bool, default=False
            If True, recompute trace even if a cached trace exists and is valid.
            Only applicable when Graph was created with cache_trace=True.

        Returns
        -------
        EliminationTrace
            Elimination trace (from cache or computed)

        Notes
        -----
        **Caching Mode** (Graph created with cache_trace=True):
        - Trace is cached on the instance and reused by moments(), expectation(), etc.
        - Graph is cloned before trace recording, preserving the original
        - Subsequent calls return cached trace unless force=True or graph was modified

        **Non-Caching Mode** (default):
        - The graph elimination algorithm is DESTRUCTIVE - vertices are eliminated
        - Use disk caching (hierarchical=True parameter) to avoid re-recording

        Examples
        --------
        >>> # Cache mode - non-destructive, cached on instance
        >>> g = Graph(model, nr_samples=5, cache_trace=True)
        >>> g.normalize()
        >>> trace = g.compute_trace()  # Graph preserved, trace cached
        >>> g.update_weights([1.0, 2.0])
        >>> mean = g.expectation()  # Uses cached trace
        >>>
        >>> # Standard mode with disk caching
        >>> g = Graph(model, nr_samples=5)
        >>> trace = g.compute_trace()  # Graph emptied, trace returned
        """
        import warnings
        warnings.warn(
            "Graph.compute_trace() is deprecated. The Python "
            "EliminationTrace path it produces is no longer wired to "
            "moments() / expectation() / variance(); those route "
            "directly to the C++ implementation. compute_trace() will "
            "be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # If using instance-level cache mode, check cache first
        if self._cache_trace:
            # Return cached trace if valid and not forcing recompute
            if not force and self._trace is not None and not self._trace_dirty:
                return self._trace

            # Check if graph is empty
            if self.vertices_length() == 0:
                raise ValueError(
                    "Cannot compute trace: graph has no vertices. "
                    "The graph may have been emptied by a previous non-hierarchical operation."
                )

            # Clone graph to preserve original (non-destructive in cache mode)
            graph_copy = self.clone()

            # Compute trace on clone using hierarchical SCC caching
            from .hierarchical_trace_cache import get_trace_hierarchical
            trace = get_trace_hierarchical(
                graph_copy,
                param_length=param_length,
                min_size=min_size,
                parallel_strategy=parallel,
                verbose=verbose
            )

            # Cache trace on instance
            self._trace = trace
            self._trace_dirty = False

            return trace

        # Standard (non-caching instance) behavior - destructive
        if self.vertices_length() == 0:
            raise ValueError(
                "Cannot compute trace: graph has no vertices. "
                "This usually means compute_trace() was called multiple times on the same graph. "
                "Note: compute_trace() is DESTRUCTIVE and eliminates vertices during trace recording. "
                "You must rebuild the graph for each call, or use cache_trace=True for caching."
            )

        if hierarchical:
            from .hierarchical_trace_cache import get_trace_hierarchical
            trace = get_trace_hierarchical(
                self,
                param_length=param_length,
                min_size=min_size,
                parallel_strategy=parallel,
                verbose=verbose
            )
            return trace
        else:
            from .trace_elimination import record_elimination_trace
            return record_elimination_trace(self, theta_dim=param_length)

    # ========================================================================
    # Batch-Aware Methods (Phase 2: Auto-Parallelization)
    # ========================================================================

    # def pdf_batch(self, times: ArrayLike, granularity: int = 100) -> np.ndarray:
    #     """
    #     Compute PDF at multiple time points with automatic parallelization.

    #     Automatically uses pmap/vmap based on parallel configuration and batch size.
    #     For single values, use the standard pdf() method instead (no overhead).

    #     Parameters
    #     ----------
    #     times : ArrayLike
    #         Array of time points to evaluate PDF at
    #     granularity : int, default=100
    #         Discretization granularity for PDF computation

    #     Returns
    #     -------
    #     np.ndarray
    #         PDF values at each time point

    #     Examples
    #     --------
    #     >>> import phasic as pta
    #     >>> import numpy as np
    #     >>>
    #     >>> # Initialize parallel computing (once at notebook start)
    #     >>> config = pta.init_parallel()
    #     >>>
    #     >>> # Build a simple model
    #     >>> g = pta.Graph(1)
    #     >>> start = g.starting_vertex()
    #     >>> v1 = g.find_or_create_vertex([1])
    #     >>> start.add_edge(v1, 2.0)
    #     >>> g.normalize()
    #     >>>
    #     >>> # Compute PDF at many time points (automatically parallelized)
    #     >>> times = np.linspace(0.1, 5.0, 1000)
    #     >>> pdf_values = g.pdf_batch(times)
    #     >>>
    #     >>> # For single values, use pdf() instead:
    #     >>> single_value = g.pdf(1.0)

    #     Notes
    #     -----
    #     - Automatically parallelizes based on init_parallel() configuration
    #     - Uses pmap across devices, vmap for vectorization, or serial execution
    #     - No manual batching or parallelization code required
    #     """
    #     from .parallel_utils import is_batched

    #     times_arr = np.asarray(times)

    #     # For single values, delegate to C++ method directly
    #     if not is_batched(times_arr):
    #         return np.array([self.pdf(float(times_arr), granularity)])

    #     # For batched inputs, use vectorized numpy operations
    #     # The C++ pdf method is called for each element
    #     # This is a simple loop-based approach that can be parallelized by JAX if needed
    #     result = np.array([self.pdf(float(t), granularity) for t in times_arr])
    #     return result

    # def dph_pmf_batch(self, jumps: ArrayLike) -> np.ndarray:
    #     """
    #     Compute discrete phase-type PMF at multiple jump counts with automatic parallelization.

    #     Automatically uses pmap/vmap based on parallel configuration and batch size.
    #     For single values, use the standard dph_pmf() method instead (no overhead).

    #     Parameters
    #     ----------
    #     jumps : ArrayLike
    #         Array of jump counts (integers) to evaluate PMF at

    #     Returns
    #     -------
    #     np.ndarray
    #         PMF values at each jump count

    #     Examples
    #     --------
    #     >>> import phasic as pta
    #     >>> import numpy as np
    #     >>>
    #     >>> # Initialize parallel computing
    #     >>> config = pta.init_parallel()
    #     >>>
    #     >>> # Build and discretize a model
    #     >>> g = pta.Graph(1)
    #     >>> # ... build model ...
    #     >>> g_discrete, rewards = g.discretize(reward_rate=0.1)
    #     >>> g_discrete.normalize()
    #     >>>
    #     >>> # Compute PMF at many jump counts (automatically parallelized)
    #     >>> jumps = np.arange(0, 100)
    #     >>> pmf_values = g_discrete.dph_pmf_batch(jumps)

    #     Notes
    #     -----
    #     - Requires a discrete phase-type model (use discretize() first)
    #     - Automatically parallelizes based on init_parallel() configuration
    #     """
    #     from .parallel_utils import is_batched

    #     jumps_arr = np.asarray(jumps, dtype=np.int32)

    #     # For single values, delegate to C++ method directly
    #     if not is_batched(jumps_arr):
    #         return np.array([self.dph_pmf(int(jumps_arr))])

    #     # For batched inputs, vectorized evaluation
    #     result = np.array([self.dph_pmf(int(j)) for j in jumps_arr])
    #     return result

    # def moments_batch(self, powers: ArrayLike) -> np.ndarray:
    #     """
    #     Compute moments for multiple powers with automatic parallelization.

    #     Automatically uses pmap/vmap based on parallel configuration and batch size.
    #     For single values, use the standard moments() method instead (no overhead).

    #     Parameters
    #     ----------
    #     powers : ArrayLike
    #         Array of moment orders to compute (e.g., [1, 2, 3] for E[T], E[T^2], E[T^3])

    #     Returns
    #     -------
    #     np.ndarray
    #         Moment values for each power

    #     Examples
    #     --------
    #     >>> import phasic as pta
    #     >>> import numpy as np
    #     >>>
    #     >>> # Initialize parallel computing
    #     >>> config = pta.init_parallel()
    #     >>>
    #     >>> # Build a model
    #     >>> g = pta.Graph(1)
    #     >>> # ... build model ...
    #     >>>
    #     >>> # Compute multiple moments (automatically parallelized)
    #     >>> powers = np.arange(1, 10)  # Moments 1 through 9
    #     >>> moment_values = g.moments_batch(powers)

    #     Notes
    #     -----
    #     - Automatically parallelizes based on init_parallel() configuration
    #     - Each moment computation is independent and can be parallelized
    #     """
    #     from .parallel_utils import is_batched

    #     powers_arr = np.asarray(powers, dtype=np.int32)

    #     # For single values, delegate to C++ method directly
    #     if not is_batched(powers_arr):
    #         return np.array([self.moments(int(powers_arr))])

    #     # For batched inputs, vectorized evaluation
    #     result = np.array([self.moments(int(p)) for p in powers_arr])
    #     return result







#     def _joint_prob_reward(self,
#                            state: np.ndarray,
#                         indexer: StateIndexer,
#                         reward_indexer: StateIndexer,
#                         current_rewards: np.ndarray | None = None,
#                         mutation_rate: float = 1.0,
#                         reward_limit: int | dict = 10,
#                         tot_reward_limit: float = np.inf) -> tuple[np.ndarray, float]:

#             logger = get_logger(__name__)

#             prop_set_names = [p.name for p in indexer.property_sets()]
#             prop_set_name, *_ = prop_set_names

#             # determine reward dimensions (extension to state vector)
#             # reward_length = indexer.state_length - indexer[prop_set_name].state_length
#             reward_length = reward_indexer.state_length

#             if not isinstance(reward_limit, dict):
#                 reward_limits = np.repeat(reward_limit, reward_length)

#             if current_rewards is None:
#                 current_rewards = np.zeros(reward_length)

#             reward_rates = np.zeros(reward_length)
#             trash_rate = 0

#             reward_prop_names = set(prop.name for prop_set in reward_indexer.property_sets() for prop in prop_set.properties)

#             # for each base graph state index
#             for i in range(indexer[prop_set_name].state_length):
#                 # get properties for the property set
#                 props = indexer[prop_set_name].index_to_props(i, as_dict=True)

# #                props = indexer.index_to_props(i, as_dict=True)
#                 # for prop, value in getattr(props, prop_set_name).items():

#                 # for each property and its value
#                 for prop, value in props.items():

#                     # make flattened prop_set + property nmae
#                     _prop_name = f'{prop_set_name}_{prop}'

#                     if _prop_name not in reward_prop_names:
#                         continue

#                     reward_idx = reward_indexer.props_to_index(**{_prop_name: value})
#                     rate = state[i] * mutation_rate 

#                     # logger.debug("i: %d; prop: %s; value: %s; rate: %e; reward_idx: %d", i, repr(prop), repr(value), rate, reward_idx)
#                     if isinstance(reward_limit, dict):
#                         if current_rewards[i] + 1 > reward_limit[prop] and np.sum(current_rewards + r) <= tot_reward_limit:
#                             reward_rates[reward_idx] += rate
#                         else:
#                             trash_rate = trash_rate + rate
#                     else:
#                         r = np.zeros_like(reward_rates)
#                         r[reward_idx] = 1
#                         if (reward_limit is None or np.all(current_rewards + r <= reward_limits)) and np.sum(current_rewards + r) <= tot_reward_limit:
#                             reward_rates[reward_idx] += rate
#                         else:
#                             trash_rate = trash_rate + rate

#             return reward_rates, trash_rate


    def joint_prob_graph(self,
                        base_graph_indexer: StateIndexer | None = None,
                        reward_only: list | None = None,
                        reward_rates_callback: Callable | None = None,
                        mutation_rate: float = 1.0,
                        reward_limit: int | None = None,
                        tot_reward_limit: float = np.inf,
                        discrete: bool = True) -> Graph:
        """Build the joint-probability graph from this parameterized graph.

        Optimised reimplementation of :meth:`_joint_prob_graph` that produces
        a **bit-identical** graph (same vertices, same insertion order, same
        edges/weights, same serialization) while building much faster.

        The speedup comes from hoisting the per-base-state-index → reward-index
        mapping out of the per-vertex construction loop: that mapping is a pure
        function of the indexers and was previously recomputed (via millions of
        ``index_to_props`` / ``props_to_index`` calls) once per joint vertex.

        When a custom ``reward_rates_callback`` is supplied the fast path is
        bypassed and the callback is invoked exactly as in
        :meth:`_joint_prob_graph`, so behaviour is unchanged for custom
        callbacks.
        """

        logger = get_logger(__name__)

        if self.param_length() == 0:
            raise ValueError("Graph must have parameterized edges for joint_prob_graph.")
        if reward_limit is None and tot_reward_limit == np.inf:
            raise ValueError("Either reward_limit or tot_reward_limit must be specified.")

        if base_graph_indexer is None:
            if hasattr(self, '_indexer'):
                base_graph_indexer = self._indexer
            else:
                raise TypeError("If the graph was not created using an indexer, the base_graph_indexer kwarg must be supplied.")

        # Reconcile the supplied indexer with the graph's actual state vector
        # length. After composition (e.g. add_epoch), the graph carries a wider
        # state vector than the original indexer describes; in that case prefer
        # the graph's own _indexer, which add_epoch keeps in sync.
        graph_state_length = self.state_length()
        if base_graph_indexer.state_length != graph_state_length:
            graph_indexer = getattr(self, '_indexer', None)
            if graph_indexer is not None and graph_indexer.state_length == graph_state_length:
                logger.info(
                    "joint_prob_graph: supplied indexer state_length=%d does not match "
                    "graph state_length=%d; using graph._indexer instead.",
                    base_graph_indexer.state_length, graph_state_length,
                )
                base_graph_indexer = graph_indexer
            else:
                raise ValueError(
                    f"Indexer state_length ({base_graph_indexer.state_length}) does not "
                    f"match graph state_length ({graph_state_length}). Pass the indexer "
                    f"returned by add_epoch (graph._indexer), or rebuild the graph from "
                    f"this indexer."
                )

        if len(base_graph_indexer.property_sets()) != 1:
            raise ValueError("Indexer must have exactly one property set representing the base graph state.")

        # When the caller supplies a custom callback we cannot use the fast
        # precomputed reward path (it is specific to _joint_prob_reward). Fall
        # back to the reference implementation, which honours arbitrary
        # callbacks. Passing the resolved base_graph_indexer keeps behaviour
        # identical.
        if reward_rates_callback is not None:
            return self._joint_prob_graph(
                base_graph_indexer=base_graph_indexer,
                reward_only=reward_only,
                reward_rates_callback=reward_rates_callback,
                mutation_rate=mutation_rate,
                reward_limit=reward_limit,
                tot_reward_limit=tot_reward_limit,
                discrete=discrete,
            )

        base_starting_vertex = self.starting_vertex()

        # create indexer for rewards (each property gets its own property set)
        reward_prop_sets = []
        _rewarded_props = []
        property_set = base_graph_indexer.property_sets()[0]
        for p in property_set.properties:
            if reward_only is None or p.name in reward_only:
                _rewarded_props.append(p)
                reward_prop_sets.append(
                    PropertySet(
                        name=f'{property_set.name}_{p.name}',
                        properties=[
                            Property(f'{property_set.name}_{p.name}',
                                    min_value=p.min_value,
                                    max_value=p.max_value)
                                    ]
                        )
                    )
        kwargs = OrderedDict()
        for x in reward_prop_sets:
            kwargs[x.name] = x.properties
        reward_indexer = StateIndexer(**kwargs)
        reward_length = reward_indexer.state_length

        # append reward indexer to original indexer
        joint_graph_indexer = base_graph_indexer + reward_indexer

        # joint graph state vector length
        state_vector_length = joint_graph_indexer.state_length

        # indices for original and new parts of the state vector
        state_indices = base_graph_indexer.indices()
        reward_state_indices = np.arange(base_graph_indexer.state_length, joint_graph_indexer.state_length)

        # ---- B1: hoist the invariant base-index -> reward-index mapping ----
        # _joint_prob_reward recomputes, for every joint vertex, the same map
        # from base-state vector index i to (reward_idx, prop_name). It depends
        # only on the indexers, so build it ONCE here. This is the dominant
        # cost removed (was ~5.7M index_to_props/props_to_index calls; now a
        # few hundred). prop_name is retained for the (currently dead) dict
        # reward_limit gating to mirror _joint_prob_reward exactly.
        prop_set_name = base_graph_indexer.property_sets()[0].name
        reward_prop_names = set(
            prop.name
            for ps in reward_indexer.property_sets()
            for prop in ps.properties
        )
        base_state_length = base_graph_indexer[prop_set_name].state_length
        base_idx_to_rewards = [[] for _ in range(base_state_length)]
        for i in range(base_state_length):
            props = base_graph_indexer[prop_set_name].index_to_props(i, as_dict=True)
            for prop, value in props.items():
                _prop_name = f'{prop_set_name}_{prop}'
                if _prop_name not in reward_prop_names:
                    continue
                reward_idx = reward_indexer.props_to_index(**{_prop_name: value})
                base_idx_to_rewards[i].append((reward_idx, prop))

        # ---- B2: fast reward closure ----
        # Replicates _joint_prob_reward's per-vertex computation exactly, but
        # (a) uses the precomputed base_idx_to_rewards map instead of
        # re-deriving reward_idx, and (b) replaces the per-(i, reward_idx)
        # numpy ops (np.zeros_like / np.all / np.sum over the full reward
        # vector) with their exact scalar equivalents, which are bit-identical
        # because `r` is a one-hot vector at reward_idx:
        #   np.sum(current_rewards + r)        == cr_sum + 1   (integer rewards)
        #   np.all(current_rewards + r <= lim) == all_base_le AND
        #                                         (current_rewards[reward_idx]+1
        #                                          <= lim[reward_idx])
        # `all_base_le` and `cr_sum` are loop-invariant per vertex.
        # The i-then-prop accumulation order is preserved so the float result
        # is identical.
        _reward_limit = reward_limit
        _tot_reward_limit = tot_reward_limit
        if not isinstance(_reward_limit, dict):
            _reward_limits_arr = np.repeat(_reward_limit, reward_length)

        def fast_reward(current_state, current_rewards):
            reward_rates = np.zeros(reward_length)
            trash_rate = 0
            cr_sum = current_rewards.sum()
            # all_base_le: whether current_rewards already satisfies the
            # per-dimension limits everywhere (the part of the np.all that
            # does not depend on reward_idx). Adding the one-hot only affects
            # index reward_idx, checked separately below. Only meaningful when
            # a scalar reward_limit is set; when reward_limit is None the
            # original short-circuits before the np.all (so we must not touch
            # _reward_limits_arr, which would be an object array of None).
            if (_reward_limit is not None) and (not isinstance(_reward_limit, dict)):
                all_base_le = bool(np.all(current_rewards <= _reward_limits_arr))
            for i in range(base_state_length):
                mapped = base_idx_to_rewards[i]
                if not mapped:
                    continue
                si_rate = current_state[i] * mutation_rate
                for reward_idx, prop in mapped:
                    rate = si_rate
                    if isinstance(_reward_limit, dict):
                        # Mirrors the (currently dead / NameError-raising) dict
                        # branch of _joint_prob_reward for exact parity.
                        if current_rewards[i] + 1 > _reward_limit[prop] and (cr_sum + 1) <= _tot_reward_limit:
                            reward_rates[reward_idx] += rate
                        else:
                            trash_rate = trash_rate + rate
                    else:
                        ok_limits = (
                            _reward_limit is None
                            or (all_base_le and (current_rewards[reward_idx] + 1 <= _reward_limits_arr[reward_idx]))
                        )
                        if ok_limits and (cr_sum + 1) <= _tot_reward_limit:
                            reward_rates[reward_idx] += rate
                        else:
                            trash_rate = trash_rate + rate
            return reward_rates, trash_rate

        # create the new graph
        joint_graph = Graph(state_vector_length)
        starting_vertex = joint_graph.starting_vertex()

        # array of zeros for extension of state vector
        null_rewards = np.zeros(reward_length)

        # graph index of last vertex visited
        index = 0

        # get param_length for extracting parameterized edge coefficients
        param_length = self.param_length()
        # Base edges may carry MORE coefficients than param_length (the theta
        # dimension) when the rate depends on per-edge data beyond the optimized
        # parameters (weight_formula). Preserve the FULL coefficient vector and
        # place the mutation slot AFTER it; the joint graph's theta dimension is
        # param_length + 1 (base params + the mutation rate). For the common case
        # (coefficient length == param_length) all of this is a no-op, so existing
        # joint graphs are bit-identical.
        # Max coefficient length over all parameterized edges, floored at
        # param_length (IPV/starting edges may carry fewer coefficients, so a
        # plain "first edge" probe can undercount; the dynamics edges define the
        # true length and are >= param_length).
        base_coeff_length = param_length
        for _bi in range(self.vertices_length()):
            for _be in self.vertex_at(_bi).parameterized_edges():
                _cl = _be.coefficients_length()
                if _cl > base_coeff_length:
                    base_coeff_length = _cl
        # Pin the joint theta dimension to param_length + 1 BEFORE adding edges so
        # it doesn't lock to the (possibly longer) coefficient length.
        joint_graph.set_param_length(param_length + 1)

        # copy initial (extended) states to new graph
        for edge in base_starting_vertex.parameterized_edges():
            starting_vertex.add_edge(
            joint_graph.find_or_create_vertex(
                np.append(edge.to().state(), null_rewards).astype(int)),
            edge.weight())

        index = index + 1

        # weights of edges to trash
        trash_rates = {}

        # indices of t-states (with absorbing as only child).
        # Use a Python list (append is amortised O(1)); the reference version
        # used np.append in-loop (O(n^2)). Final content is identical after
        # np.unique below.
        t_vertex_indices_list = []

        # graph construction loop
        while index < joint_graph.vertices_length():

            # graph state
            vertex = joint_graph.vertex_at(index)
            state = vertex.state()

            # get vertex with same state in base graph
            base_state = vertex.state()[state_indices]
            base_vertex = self.find_vertex(base_state)

            # add edges and children of vertex in base graph
            for edge in base_vertex.parameterized_edges():
                # child states are copies of the base_vertex child_states
                # extended with a copy of the extended part of the
                # current state's state vector
                child_state = np.append(
                    edge.to().state(),
                    state[reward_state_indices]
                    )

                if np.all(state == child_state): # FIXME: should this ever happen?
                    continue

                # create the vertex
                child_vertex = joint_graph.find_or_create_vertex(
                    child_state
                    )

                # get the FULL edge coefficient vector (length base_coeff_length,
                # which may exceed the theta dimension param_length)
                coeffs = list(edge.edge_state(base_coeff_length))

                # Pad with 0 for mutation rate slot (at index base_coeff_length)
                coeffs.append(0)

                # add edge to the child vertex
                vertex.add_edge(child_vertex, coeffs)

                # if the base graph version of the child state was
                # absorbing, we add it to the array of t-states
                if not self.find_vertex(child_state[state_indices]).edges():
                    t_vertex_indices_list.append(child_vertex.index())

            # base part of current state
            current_state = state[state_indices]
            # extended part of current state
            current_rewards = state[reward_state_indices]

            # get rates to states representing an additional mutation
            rates, trash_rate = fast_reward(current_state, current_rewards)

            trash_rates[index] = trash_rate
            for i in range(reward_length):
                rate = rates[i]
                if rate > 0:
                    new_rewards = current_rewards.copy()
                    new_rewards[i] = new_rewards[i] + 1
                    child_state = np.append(current_state, new_rewards)
                    if not self.find_vertex(child_state[state_indices]).edges():
                        continue
                    child_vertex = joint_graph.find_or_create_vertex(child_state)
                    vertex.add_edge(child_vertex, np.append(np.zeros(base_coeff_length), rate))

            index = index + 1

        t_vertex_indices = np.unique(np.array(t_vertex_indices_list, dtype=int)).tolist()

        # create trash vertices
        trash_vertex = joint_graph.find_or_create_vertex(np.repeat(0, state_vector_length))
        trash_loop_vertex = joint_graph.create_vertex(np.repeat(0, state_vector_length))
        trash_vertex.add_edge(trash_loop_vertex, np.append(np.zeros(base_coeff_length), 1.0))
        trash_loop_vertex.add_edge(trash_vertex, np.append(np.zeros(base_coeff_length), 1.0))

        # connect edges to first trash state
        for i, rate in trash_rates.items():
            if rate > 0:
                joint_graph.vertex_at(i).add_edge(trash_vertex, np.append(np.zeros(base_coeff_length), rate))


        if reward_only is not None:
            reward_only = sorted(reward_only)
            sorted_prop_names = sorted([p.name for p in property_set.properties])
            if all(x == y for x, y in zip_longest(reward_only, sorted_prop_names)):
                # no effect anyway
                logger.info('Specified reward_only lists all properties. Set to None for same effect.')
                reward_only = None

        if reward_only is not None:

            # for sets of t-states representing the same observation, remove them from
            # the list of t-states and add prob 1 edges to a new t-state representing all
            # of them. t-states in such sets are the ones that only differ by properties not
            # in the reward_only keyword arg
            values = []
            for p in property_set.properties:
                if p.name in reward_only:
                    values.append(list(range(p.min_value, p.max_value+1)))
            idxs = []
            for tup in product(*values):
                idxs.extend(property_set.props_to_index(**dict(zip(reward_only, tup))))
            idxs = np.array(sorted(idxs))

            t_vertex_sets = defaultdict(list)
            for i in range(joint_graph.vertices_length()):
                state = joint_graph.vertex_at(i).state()
                mask = np.ones(state_vector_length, np.bool)
                mask[idxs] = 0
                if i in t_vertex_indices:
                    t_vertex_sets[tuple(state[mask].tolist())].append(i)

            for t_vertex_set in t_vertex_sets.values():
                state = np.repeat(0, state_vector_length)
                state[mask] = joint_graph.vertex_at(t_vertex_set[0]).state()[mask]
                t_set_abs = joint_graph.create_vertex(state)
                for i in t_vertex_set:
                    joint_graph.vertex_at(i).add_edge(
                        t_set_abs,
                        np.append(np.zeros(base_coeff_length), 1.0)
                        )
                    t_vertex_indices.remove(i)
                t_vertex_indices.append(t_set_abs.index())


        # the t-states represent variants of the original absorbing state
        # add a new absorbing with edges from all t-states
        new_absorbing = joint_graph.create_vertex(np.repeat(0, state_vector_length))

        for i in t_vertex_indices:
            joint_graph.vertex_at(i).add_edge(new_absorbing, np.append(np.zeros(base_coeff_length), 1.0))

        # set discrete flag for update_weights to also normalize and for
        # expected_sojourn_time to call its discrete version
        joint_graph.is_discrete = discrete
        joint_graph.set_was_dph(discrete)  # Enable auto-normalization in C update_weights()

        joint_graph._joint_prob_base_graph_indexer = base_graph_indexer
        joint_graph._rewarded_props = _rewarded_props
        # Attach the combined (base + reward) indexer so the joint graph
        # carries an indexer matching its own state vector length, mirroring
        # the convention for callback-built and epoch-augmented graphs.
        joint_graph._indexer = joint_graph_indexer

        return joint_graph


    # def _joint_prob_graph(self,
    #                     base_graph_indexer: StateIndexer | None = None,
    #                     reward_only: list | None = None,
    #                     reward_rates_callback: Callable | None = None,
    #                     mutation_rate: float = 1.0,
    #                     reward_limit: int | None = None,
    #                     tot_reward_limit: float = np.inf,
    #                     discrete: bool = True) -> Graph:
    #     """DEPRECATED reference implementation of :meth:`joint_prob_graph`.

    #     Kept verbatim for A/B comparison against the optimised
    #     ``joint_prob_graph``. Slated for removal. Do not use in new code.
    #     """

    #     logger = get_logger(__name__)

    #     if self.param_length() == 0:
    #         raise ValueError("Graph must have parameterized edges for joint_prob_graph.")
    #     if reward_limit is None and tot_reward_limit == np.inf:
    #         raise ValueError("Either reward_limit or tot_reward_limit must be specified.")

    #     if base_graph_indexer is None:
    #         if hasattr(self, '_indexer'):
    #             base_graph_indexer = self._indexer
    #         else:
    #             raise TypeError("If the graph was not created using an indexer, the base_graph_indexer kwarg must be supplied.")

    #     # Reconcile the supplied indexer with the graph's actual state vector
    #     # length. After composition (e.g. add_epoch), the graph carries a wider
    #     # state vector than the original indexer describes; in that case prefer
    #     # the graph's own _indexer, which add_epoch keeps in sync.
    #     graph_state_length = self.state_length()
    #     if base_graph_indexer.state_length != graph_state_length:
    #         graph_indexer = getattr(self, '_indexer', None)
    #         if graph_indexer is not None and graph_indexer.state_length == graph_state_length:
    #             logger.info(
    #                 "joint_prob_graph: supplied indexer state_length=%d does not match "
    #                 "graph state_length=%d; using graph._indexer instead.",
    #                 base_graph_indexer.state_length, graph_state_length,
    #             )
    #             base_graph_indexer = graph_indexer
    #         else:
    #             raise ValueError(
    #                 f"Indexer state_length ({base_graph_indexer.state_length}) does not "
    #                 f"match graph state_length ({graph_state_length}). Pass the indexer "
    #                 f"returned by add_epoch (graph._indexer), or rebuild the graph from "
    #                 f"this indexer."
    #             )

    #     if len(base_graph_indexer.property_sets()) != 1:
    #         raise ValueError("Indexer must have exactly one property set representing the base graph state.")

    #     if reward_rates_callback is None:
    #         # default to joint prob reward callback
    #         reward_rates_callback = self._joint_prob_reward

    #     base_starting_vertex = self.starting_vertex()

    #     # create indexer for rewards (each property gets its own property set)
    #     reward_prop_sets = []
    #     _rewarded_props = []
    #     property_set = base_graph_indexer.property_sets()[0]
    #     for p in property_set.properties:
    #         if reward_only is None or p.name in reward_only:
    #             _rewarded_props.append(p)                    
    #             reward_prop_sets.append(
    #                 PropertySet(
    #                     name=f'{property_set.name}_{p.name}',
    #                     properties=[
    #                         Property(f'{property_set.name}_{p.name}', 
    #                                 min_value=p.min_value, 
    #                                 max_value=p.max_value)
    #                                 ]
    #                     )
    #                 )
    #     kwargs = OrderedDict()        
    #     for x in reward_prop_sets:
    #         kwargs[x.name] = x.properties
    #     reward_indexer = StateIndexer(**kwargs)                    
    #     reward_length = reward_indexer.state_length

    #     # logger.debug(f"Reward indexer created with {reward_indexer.state_length} states: {reward_indexer}")

    #     # append reward indexer to original indexer
    #     joint_graph_indexer = base_graph_indexer + reward_indexer

    #     # joint graph state vector length
    #     state_vector_length = joint_graph_indexer.state_length

    #     # indices for original and new parts of the state vector
    #     state_indices = base_graph_indexer.indices()
    #     reward_state_indices = np.arange(base_graph_indexer.state_length, joint_graph_indexer.state_length)

    #     # create the new graph
    #     joint_graph = Graph(state_vector_length)
    #     starting_vertex = joint_graph.starting_vertex()

    #     # array of zeros for extension of state vector
    #     null_rewards = np.zeros(reward_length)

    #     # graph index of last vertex visited
    #     index = 0

    #     # get param_length for extracting parameterized edge coefficients
    #     param_length = self.param_length()

    #     # copy initial (extended) states to new graph
    #     for edge in base_starting_vertex.parameterized_edges():
    #         starting_vertex.add_edge(
    #         joint_graph.find_or_create_vertex(
    #             np.append(edge.to().state(), null_rewards).astype(int)),
    #         edge.weight())

    #     # pgbar
    #     # pgbar_prev = 0    
    #     # pgbar = tqdm(position=0, total=1, miniters=0, 
    #     #             desc='Visited / Created', bar_format='{l_bar}{bar}'
    #     #             )
    #     index = index + 1

    #     # weights of edges to trash    
    #     trash_rates = {}

    #     # indices of t-states (with absorbing as only child)
    #     t_vertex_indices = np.array([], dtype=int)

    #     # graph construction loop
    #     while index < joint_graph.vertices_length():

    #         # graph state
    #         vertex = joint_graph.vertex_at(index) 
    #         state = vertex.state()

    #         # get vertex with same state in base graph
    #         base_state = vertex.state()[state_indices]
    #         base_vertex = self.find_vertex(base_state)

    #         # add edges and children of vertex in base graph
    #         for edge in base_vertex.parameterized_edges():
    #             # child states are copies of the base_vertex child_states
    #             # extended with a with a copy of the extended part of the 
    #             # current states state vector
    #             child_state = np.append(
    #                 edge.to().state(),
    #                 state[reward_state_indices]
    #                 )

    #             if np.all(state == child_state): # FIXME: should this ever happen?
    #                 continue

    #             # create the vertex
    #             child_vertex = joint_graph.find_or_create_vertex(
    #                 child_state
    #                 )
                
    #             # get the edge state (ensuring it is param_length)
    #             coeffs = list(edge.edge_state(param_length))
                
    #             # Pad with 0 for mutation rate slot
    #             coeffs.append(0) 

    #             # add edge to the child vertex
    #             vertex.add_edge(child_vertex, coeffs)

    #             # if the base graph version of the child state was
    #             # absorbing, we add it to the array of t-states
    #             if not self.find_vertex(child_state[state_indices]).edges():
    #                 t_vertex_indices = np.append(t_vertex_indices, child_vertex.index()) 

    #         # base part of current state
    #         current_state = state[state_indices]
    #         # extended part of current state
    #         current_rewards = state[reward_state_indices]

    #         # get rates to states representing an additional mutation
    #         rates, trash_rate = reward_rates_callback(
    #             current_state, 
    #             base_graph_indexer, 
    #             reward_indexer,
    #             current_rewards, 
    #             mutation_rate=mutation_rate, 
    #             reward_limit=reward_limit, 
    #             tot_reward_limit=tot_reward_limit
    #             ) 

    #         trash_rates[index] = trash_rate
    #         for i in range(reward_length):
    #             rate = rates[i]
    #             if rate > 0:
    #                 new_rewards = current_rewards.copy()
    #                 new_rewards[i] = new_rewards[i] + 1
    #                 child_state = np.append(current_state, new_rewards)
    #                 if not self.find_vertex(child_state[state_indices]).edges():
    #                     continue
    #                 child_vertex = joint_graph.find_or_create_vertex(child_state)
    #                 vertex.add_edge(child_vertex, np.append(np.zeros(base_coeff_length), rate))
                                    
    #         index = index + 1 

    #     t_vertex_indices = np.unique(t_vertex_indices).tolist()

    #     #     pgbar_this = index/joint_graph.vertices_length()
    #     #     pgbar.update(pgbar_this - pgbar_prev)
    #     #     pgbar_prev = pgbar_this

    #     # pgbar.close()

    #     # create trash vertices
    #     trash_vertex = joint_graph.find_or_create_vertex(np.repeat(0, state_vector_length))
    #     trash_loop_vertex = joint_graph.create_vertex(np.repeat(0, state_vector_length))
    #     trash_vertex.add_edge(trash_loop_vertex, np.append(np.zeros(base_coeff_length), 1.0))
    #     trash_loop_vertex.add_edge(trash_vertex, np.append(np.zeros(base_coeff_length), 1.0))

    #     # connect edges to first trash state
    #     for i, rate in trash_rates.items():
    #         if rate > 0:
    #             joint_graph.vertex_at(i).add_edge(trash_vertex, np.append(np.zeros(base_coeff_length), rate))


    #     if reward_only is not None:
    #         reward_only = sorted(reward_only)
    #         sorted_prop_names = sorted([p.name for p in property_set.properties])
    #         if all(x == y for x, y in zip_longest(reward_only, sorted_prop_names)):
    #             # no effect anyway
    #             logger.info('Specified reward_only lists all properties. Set to None for same effect.')
    #             reward_only = None

    #     if reward_only is not None:

    #         # for sets of t-states representing the same observation, remove them from
    #         # the list of t-states and add prob 1 edges to a new t-state representing all
    #         # of them. t-states in such sets are the ones that only differ by properties not
    #         # in the reward_only keyword arg
    #         values = []
    #         for p in property_set.properties:
    #             if p.name in reward_only:
    #                 values.append(list(range(p.min_value, p.max_value+1)))
    #         idxs = []
    #         for tup in product(*values):
    #             idxs.extend(property_set.props_to_index(**dict(zip(reward_only, tup))))
    #         idxs = np.array(sorted(idxs))

    #         t_vertex_sets = defaultdict(list)
    #         for i in range(joint_graph.vertices_length()):
    #             state = joint_graph.vertex_at(i).state()
    #             mask = np.ones(state_vector_length, np.bool)
    #             mask[idxs] = 0
    #             if i in t_vertex_indices:
    #                 t_vertex_sets[tuple(state[mask].tolist())].append(i)

    #         for t_vertex_set in t_vertex_sets.values():
    #             state = np.repeat(0, state_vector_length)
    #             state[mask] = joint_graph.vertex_at(t_vertex_set[0]).state()[mask]
    #             t_set_abs = joint_graph.create_vertex(state)    
    #             for i in t_vertex_set:
    #                 joint_graph.vertex_at(i).add_edge(
    #                     t_set_abs, 
    #                     np.append(np.zeros(base_coeff_length), 1.0)
    #                     )
    #                 t_vertex_indices.remove(i)
    #             t_vertex_indices.append(t_set_abs.index())


    #     # the t-states represent variants of the original absorbing state
    #     # add a new absorbing with edges from all t-states
    #     new_absorbing = joint_graph.create_vertex(np.repeat(0, state_vector_length))
        
    #     for i in t_vertex_indices:
    #         joint_graph.vertex_at(i).add_edge(new_absorbing, np.append(np.zeros(base_coeff_length), 1.0))

    #     # set discrete flag for update_weights to also normalize and for
    #     # expected_sojourn_time to call its discrete version
    #     joint_graph.is_discrete = discrete
    #     joint_graph.set_was_dph(discrete)  # Enable auto-normalization in C update_weights()

    #     joint_graph._joint_prob_base_graph_indexer = base_graph_indexer
    #     joint_graph._rewarded_props = _rewarded_props
    #     # Attach the combined (base + reward) indexer so the joint graph
    #     # carries an indexer matching its own state vector length, mirroring
    #     # the convention for callback-built and epoch-augmented graphs.
    #     joint_graph._indexer = joint_graph_indexer

    #     return joint_graph


    def joint_stop_prob_graph(self) -> 'Graph':
        """Build the joint stop-probability graph for daisy-chained inference.

        Promotes the notebook helper (in
        ``docs/pages/tutorial/time_inhom_joint_prob.ipynb``) into the library.
        The transformation: for each "t-vertex" in the source joint-prob
        graph (a vertex that has a transition to an absorbing vertex), wire
        a "trapping aux loop" — an aux vertex with state ``[0,...,0]`` and
        bidirectional unit-weight parameterised edges to/from the t-vertex.
        Mass that reaches a t-vertex shuttles between t-vertex and aux
        forever, so ``stop_probability(t)[t_vertex] +
        stop_probability(t)[aux]`` equals the cumulative joint absorption
        mass at that t-state by time t.

        Initial-probability-vector edges (starting vertex outgoing) are
        added at construction time to **every non-aux, non-trash, non-
        absorbing vertex** with weight 0. The user calls ``update_ipv``
        before each epoch to set them; the daisy-chain loop calls it
        between epochs to propagate surviving mass forward. The IPV vector
        layout matches ``joint_stop_probabilities`` output (vertex order,
        skipping aux vertices and the trash pair, expanded back to full
        vertex space — see ``_collapse_t_aux``).

        Returns
        -------
        Graph
            New graph with attributes ``_joint_stop_prob_graph = True``,
            ``_t_vertex_indices`` (sorted list of new-graph t-vertex
            indices), ``_t_aux_map`` (dict mapping new-graph t-vertex
            index → new-graph aux index), and the propagated
            ``_joint_prob_base_graph_indexer`` / ``_rewarded_props`` /
            ``_cache_trace`` from the source.

        Raises
        ------
        ValueError
            If ``self`` is not a joint-prob graph (no
            ``_joint_prob_base_graph_indexer``) or has no parameterised
            edges (``param_length() == 0``).
        """
        if not getattr(self, '_joint_prob_base_graph_indexer', None):
            raise ValueError(
                "joint_stop_prob_graph requires a graph produced by "
                "joint_prob_graph()."
            )
        if self.param_length() == 0:
            raise ValueError(
                "joint_stop_prob_graph requires a parameterised graph; "
                "got param_length() == 0."
            )

        # Trash-pair predicate (matches notebook): two zero-state vertices
        # whose only edges are to each other.
        def _is_trash(v: Vertex) -> bool:
            if v.state().sum() != 0 or v.edges_length() != 1:
                return False
            child = v.edges()[0].to()
            if child.state().sum() != 0 or child.edges_length() != 1:
                return False
            return child.edges()[0].to().index() == v.index()

        # Identify t-vertices, trash pair, and absorbing index in the source.
        start_old = self.starting_vertex()
        t_vertex_old_indices: list[int] = []
        trash_old_indices: list[int] = []
        abs_old_index: int | None = None
        for v in self.vertices():
            if v.index() == start_old.index():
                continue
            if not v.edges():
                abs_old_index = v.index()
                continue
            for edge in v.edges():
                if len(edge.to().edges()) == 0:
                    t_vertex_old_indices.append(v.index())
                    break
            if _is_trash(v):
                trash_old_indices.append(v.index())

        t_vertex_old_indices = list(np.unique(t_vertex_old_indices))
        if len(trash_old_indices) != 2:
            raise ValueError(
                f"joint_stop_prob_graph: expected exactly 2 trash vertices "
                f"in source graph, found {len(trash_old_indices)}."
            )
        if abs_old_index is None:
            raise ValueError(
                "joint_stop_prob_graph: source graph has no absorbing vertex."
            )

        # Build the new graph.
        new = Graph(self.state_length())
        new.set_param_length(self.param_length())
        param_length = self.param_length()

        vmap: dict[int, Vertex] = {start_old.index(): new.starting_vertex()}
        for v in self.vertices():
            if v.index() == start_old.index():
                continue
            vmap[v.index()] = new.create_vertex(list(v.state()))

        # Copy interior edges (skipping trash, redirecting trash-pointers to
        # the absorbing vertex). For t-vertices, install the t-aux trapping
        # loop instead of the original outgoing edges.
        t_aux_map: dict[int, int] = {}  # new-graph t-vertex idx → new-graph aux idx
        for v in self.vertices():
            if v.index() == start_old.index() or not v.edges():
                continue
            if v.index() in trash_old_indices:
                continue

            nv = vmap[v.index()]

            if v.index() in t_vertex_old_indices:
                # Install the trapping loop as bidirectional
                # COEFFICIENT-LESS constant-weight edges. The previous
                # implementation used parameterised edges with
                # coefficient ``[1.0] * param_length``, but linear-mode
                # weight evaluation gives ``Σ 1.0 · θ_k = θ_0 + θ_1 + …``,
                # which (a) couples the trapping rate to theta and (b)
                # under per-observation exposure scaling lets the
                # mu-slot-times-alpha term inflate every t-aux edge in
                # the JSP graph, blowing up λ_max and the
                # uniformization auto-granularity. Using
                # ``add_aux_vertex_constant`` makes both directions
                # coefficient-less so ``ptd_graph_update_weights``
                # skips them; the trapping rate stays at exactly 1.0
                # regardless of theta or exposure.
                t_aux_vertex = nv.add_aux_vertex_constant(1.0)
                t_aux_map[nv.index()] = t_aux_vertex.index()
                continue

            for e in v.parameterized_edges():
                to_index = e.to().index()
                if to_index in trash_old_indices:
                    to_index = abs_old_index
                # Copy the FULL coefficient vector (may exceed param_length when
                # theta_dim < coefficient length, e.g. weight_formula on the
                # joint graph); the new graph keeps param_length = self.param_length().
                nv.add_edge(vmap[to_index], list(e.edge_state(e.coefficients_length())))

        # IPV edges: one scalar starting-vertex edge per non-aux non-trash
        # non-absorbing vertex that exists in the new graph (i.e. all the
        # vmapped vertices except the start, the absorbing vertex, the
        # trash pair, and the new aux vertices). Initial weight 0 — the
        # caller must call update_ipv before any forward computation.
        non_ipv_old_indices = (
            {start_old.index(), abs_old_index}
            | set(trash_old_indices)
        )
        # Sort by new-graph index so the IPV layout is stable and matches
        # the natural scan order used by _collapse_t_aux.
        ipv_targets = sorted(
            (vmap[old_idx].index(), vmap[old_idx])
            for old_idx in vmap
            if old_idx not in non_ipv_old_indices
        )
        ipv_target_indices = [new_idx for new_idx, _v in ipv_targets]
        for _new_idx, target in ipv_targets:
            new.starting_vertex().add_edge(target, 0.0)

        # Attach metadata read by the daisy-chain FFI handler. The C++
        # side rebuilds its own collapsed-position lookup from
        # _t_aux_map; we don't need to precompute one here.
        new._joint_prob_base_graph_indexer = self._joint_prob_base_graph_indexer
        new._rewarded_props = getattr(self, '_rewarded_props', None)
        new._joint_stop_prob_graph = True
        new._t_vertex_indices = sorted(t_aux_map.keys())
        new._t_aux_map = t_aux_map
        # Source (continuous joint-prob) graph, so the JSP graph can build the
        # no-trapping sojourn graph on demand for daisy_chain_joint_probs(
        # final_read='sojourn').
        new._joint_prob_source = self
        # New-graph vertex indices that carry IPV edges, in starting-vertex
        # edge order. update_ipv expects a vector of this length, in this
        # order.
        new._ipv_target_indices = ipv_target_indices
        new.is_discrete = self.is_discrete
        new._cache_trace = getattr(self, '_cache_trace', False)
        # Forward the indexer like joint_prob_graph does.
        if hasattr(self, '_indexer'):
            new._indexer = self._indexer
        # Carry a weight formula (if any) so the daisy-chain FFI evaluates it
        # in C; other weight modes are left untouched.
        _propagate_weight_formula(self, new)
        return new


    def joint_sojourn_graph(self) -> 'Graph':
        """Build the joint sojourn-read graph for granularity-free epoch reads.

        Sibling of :meth:`joint_stop_prob_graph`. Where the JSP graph wires a
        trapping aux loop at each t-vertex (so cumulative absorption mass is read
        via the granularity-bound ``stop_probability(t)`` forward solve), this
        variant keeps each t-vertex's original edge to the absorbing vertex (NO
        trapping) and adds the same full IPV-edge layout. That makes the **exact**
        absorption distribution from an arbitrary initial distribution readable
        granularity-free via graph elimination:

            joint_prob[t-state v] = r_v * expected_sojourn_time(v) * ipv_mass

        where ``r_v`` is the t-vertex's exit rate to absorption and ``ipv_mass``
        is the total mass of the (possibly sub-stochastic) initial distribution
        set via :meth:`update_ipv`. ``expected_sojourn_time`` normalises the IPV
        to unit mass, so the caller multiplies back by ``ipv_mass``. This is the
        granularity-free replacement for the daisy chain's final-epoch
        ``stop_probability(t_eval)`` read (the final epoch runs to absorption,
        the one place the elimination read applies); it is also exact in the
        sense that it has no finite-``t_eval`` truncation.

        Returns
        -------
        Graph
            New graph with attributes ``_joint_sojourn_graph = True``,
            ``_t_vertex_indices`` (new-graph indices of the original t-vertices),
            ``_ipv_target_indices`` / ``_ipv_target_states`` (the full IPV-edge
            layout; ``update_ipv`` expects a vector of length
            ``len(_ipv_target_indices)`` in this order), and the propagated
            ``_joint_prob_base_graph_indexer`` / ``_rewarded_props`` / ``_indexer``.

        Raises
        ------
        ValueError
            If ``self`` is not a joint-prob graph or has no parameterised edges.
        """
        if not getattr(self, '_joint_prob_base_graph_indexer', None):
            raise ValueError(
                "joint_sojourn_graph requires a graph produced by joint_prob_graph()."
            )
        if self.param_length() == 0:
            raise ValueError(
                "joint_sojourn_graph requires a parameterised graph; "
                "got param_length() == 0."
            )

        # Trash-pair predicate (matches joint_stop_prob_graph).
        def _is_trash(v: Vertex) -> bool:
            if v.state().sum() != 0 or v.edges_length() != 1:
                return False
            child = v.edges()[0].to()
            if child.state().sum() != 0 or child.edges_length() != 1:
                return False
            return child.edges()[0].to().index() == v.index()

        start_old = self.starting_vertex()
        t_vertex_old_indices: list[int] = []
        trash_old_indices: list[int] = []
        abs_old_index: int | None = None
        for v in self.vertices():
            if v.index() == start_old.index():
                continue
            if not v.edges():
                abs_old_index = v.index()
                continue
            for edge in v.edges():
                if len(edge.to().edges()) == 0:
                    t_vertex_old_indices.append(v.index())
                    break
            if _is_trash(v):
                trash_old_indices.append(v.index())

        t_vertex_old_indices = list(np.unique(t_vertex_old_indices))
        if len(trash_old_indices) != 2:
            raise ValueError(
                f"joint_sojourn_graph: expected exactly 2 trash vertices in "
                f"source graph, found {len(trash_old_indices)}."
            )
        if abs_old_index is None:
            raise ValueError(
                "joint_sojourn_graph: source graph has no absorbing vertex."
            )

        param_length = self.param_length()
        trash_set = set(trash_old_indices)

        new = Graph(self.state_length())
        new.set_param_length(param_length)

        vmap: dict[int, Vertex] = {start_old.index(): new.starting_vertex()}
        for v in self.vertices():
            if v.index() == start_old.index() or v.index() in trash_set:
                continue
            vmap[v.index()] = new.create_vertex(list(v.state()))

        # Copy interior edges verbatim (including each t-vertex's edge to the
        # absorbing vertex — NO trap), redirecting trash-pointers to absorbing.
        for v in self.vertices():
            if v.index() == start_old.index() or v.index() in trash_set:
                continue
            if not v.edges():
                continue
            nv = vmap[v.index()]
            for e in v.parameterized_edges():
                to_index = e.to().index()
                if to_index in trash_set:
                    to_index = abs_old_index
                # Copy the FULL coefficient vector (may exceed param_length when
                # theta_dim < coefficient length, e.g. weight_formula on the
                # joint graph); the new graph keeps param_length = self.param_length().
                nv.add_edge(vmap[to_index], list(e.edge_state(e.coefficients_length())))

        # Full IPV edges: one weight-0 starting-vertex edge per non-trash,
        # non-absorbing interior vertex, sorted by new-graph index (stable
        # layout; the caller sets weights via update_ipv).
        non_ipv_old_indices = {start_old.index(), abs_old_index} | trash_set
        ipv_targets = sorted(
            (vmap[old_idx].index(), old_idx)
            for old_idx in vmap
            if old_idx not in non_ipv_old_indices
        )
        ipv_target_indices = [new_idx for new_idx, _old in ipv_targets]
        ipv_target_states = [
            tuple(int(x) for x in self.vertex_at(old_idx).state())
            for _new_idx, old_idx in ipv_targets
        ]
        for _new_idx, old_idx in ipv_targets:
            new.starting_vertex().add_edge(vmap[old_idx], 0.0)

        new._joint_sojourn_graph = True
        new._joint_prob_base_graph_indexer = self._joint_prob_base_graph_indexer
        new._rewarded_props = getattr(self, '_rewarded_props', None)
        new._t_vertex_indices = sorted(vmap[o].index() for o in t_vertex_old_indices)
        new._ipv_target_indices = ipv_target_indices
        new._ipv_target_states = ipv_target_states
        # Continuous: the read is r_v * expected_sojourn_time (continuous time)
        # * ipv_mass. Not discrete -> no set_was_dph IPV auto-normalisation.
        new.is_discrete = False
        if hasattr(self, '_indexer'):
            new._indexer = self._indexer
        # Carry a weight formula (if any) so the sojourn FFI evaluates it in C.
        _propagate_weight_formula(self, new)
        return new


    def epoch_context(self) -> 'EpochContext':
        """Build a single-epoch context for reading cumulative t-state probabilities.

        Allowed only on **continuous** joint-prob graphs (built via
        ``joint_prob_graph(..., discrete=False)``). Builds the
        joint-stop-probability (JSP) graph internally, projects this
        graph's starting-vertex IPV into the JSP graph's IPV layout,
        inherits the current theta (whatever was last passed to this
        graph's ``update_weights``), and returns an :class:`EpochContext`
        ready for ``cumulative_probs`` calls.

        If you have not called ``update_weights`` on this graph yet, you
        must call ``ctx.update_weights(theta)`` on the returned context
        before reading probabilities.

        For the multi-epoch / daisy-chain reads used by SVGD, see
        :meth:`daisy_chain_joint_probs`.

        Returns
        -------
        EpochContext
            A context object whose ``cumulative_probs(t)`` returns
            ``P(absorbed via t-state by time t)`` for each t-state.
            Converges to :meth:`joint_prob_table` (built with the same
            settings and ``discrete=True``) as ``t → ∞``.

        Raises
        ------
        ValueError
            If this graph is not a joint-prob graph, or is the discrete
            variant (use :meth:`joint_prob_table` for asymptotes
            directly).
        """
        if not getattr(self, '_joint_prob_base_graph_indexer', None):
            raise ValueError(
                "epoch_context requires a graph produced by joint_prob_graph()."
            )
        if getattr(self, 'is_discrete', False):
            raise ValueError(
                "epoch_context requires a continuous joint-prob graph. The "
                "discrete variant's joint_prob_table() already returns the "
                "asymptotic probabilities directly."
            )

        jsp = self.joint_stop_prob_graph()

        # Seed JSP graph's IPV from this graph's starting-vertex edges,
        # using the same projection convention as the SVGD daisy-chain
        # entry point (see __init__.py around the
        # `initial_ipv = self_ipv_full[jsp._ipv_target_indices]` line).
        self_ipv_full = np.zeros(self.vertices_length(), dtype=np.float64)
        for edge in self.starting_vertex().edges():
            self_ipv_full[edge.to().index()] = edge.weight()
        initial_ipv = self_ipv_full[jsp._ipv_target_indices]
        jsp.update_ipv(initial_ipv)

        # Inherit the current theta from the source graph if one has been
        # set via update_weights(). Users typically call update_weights()
        # on the source graph before building the epoch context; we
        # propagate that automatically so the context is ready to read
        # without a redundant update_weights() call.
        if self._last_theta is not None:
            jsp.update_weights(self._last_theta)

        return EpochContext(jsp, source_graph=self)


    def daisy_chain_joint_probs(
        self,
        *,
        epoch_thetas,
        epoch_dts,
        initial_ipv,
        t_eval: float | None = None,
        fixed_indices=None,
        granularity: int = 0,
        final_read: str = 'sojourn',
    ):
        """JAX-traceable model: joint-probs at the t-states after a daisy chain.

        ``final_read`` selects how the FINAL epoch is read:

        - ``'sojourn'`` (default): granularity-free elimination read on the
          no-trapping sojourn graph (``r_v * expected_sojourn(v) * handoff_mass``)
          — exact, much faster, and ``t_eval`` is ignored (the final epoch runs to
          absorption by construction). Requires that this JSP graph was built by
          :meth:`joint_stop_prob_graph` (carries ``_joint_prob_source``).
        - ``'stopprob'``: the legacy ``stop_probability(t_eval)`` forward solve on
          the JSP graph (granularity-bound; subject to finite-``t_eval``
          truncation). Use it to reproduce pre-0.30 behavior or for a JSP graph
          that does not carry ``_joint_prob_source``.

        Daisies through ``len(epoch_dts)`` epoch transitions
        (``update_ipv → update_weights → stop_probability(dt)``), then
        in the final epoch sets the propagated IPV and the final-epoch
        theta on the JSP graph and reads
        ``joint_stop_probabilities(t_eval)`` at the t-state vertex
        positions. The number of epochs is ``len(epoch_thetas)``, which
        must equal ``len(epoch_dts) + 1``: each epoch *i* in
        0..n_epochs-2 has a transition of duration ``epoch_dts[i]``;
        the final epoch ``n_epochs-1`` has no transition out.

        The joint probability of an outcome is the relative probability
        of paths ending in (and being trapped in) that t-state. Because
        the JSP graph's t-aux loops trap mass, ``stop_probability(t)``
        at a t-vertex monotonically approaches the joint probability
        as ``t → ∞``. ``t_eval`` should be large enough that the
        chain has had time to absorb most mass at the t-states. For
        slow-mutation models ``t_eval`` may need to be quite large; the
        default scales with ``sum(epoch_dts)`` to provide a conservative
        starting point. For an adaptive ``t_eval`` chosen via a
        residual-mass probe, use
        ``Graph.svgd(..., daisy_chain_t_eval='auto')`` or call
        ``Graph._probe_daisy_t_eval(...)`` directly.

        Parameters
        ----------
        epoch_thetas : array-like, shape (n_epochs, theta_dim)
            Per-epoch parameter vectors. JAX-traced.
        epoch_dts : array-like, shape (n_epochs - 1,)
            Durations of the first ``n_epochs - 1`` epochs. Static
            Python sequence (not JAX-traced).
        initial_ipv : array-like, shape (n_ipv,)
            IPV for epoch 0, in the JSP graph's IPV layout.
        t_eval : float, optional
            Time at which the final-epoch joint stop-probabilities are
            read. Defaults to ``max(sum(epoch_dts) * 4, 10.0)``.
        fixed_indices : sequence of int, optional
            Flat-theta indices held fixed by SVGD. Forwarded to the
            ``custom_vjp`` backward pass to skip finite-difference
            gradient evaluations on those slots.
        granularity : int, optional
            Uniformization granularity passed to the underlying
            ``stop_probability`` call in the C++ FFI handler. ``0``
            (default) auto-picks a safe value from the graph's max rate
            and the requested time. Larger values give finer
            discretisation (slower, more accurate); smaller positive
            values trade accuracy for speed. The same granularity is
            used for every epoch's ``stop_probability`` call.

        Returns
        -------
        jax.Array, shape (n_t_vertices,)
            Joint-probs at the t-states, in the order of
            ``self._t_vertex_indices``.
        """
        if not getattr(self, '_joint_stop_prob_graph', False):
            raise ValueError(
                "daisy_chain_joint_probs requires a graph produced by "
                "joint_stop_prob_graph()."
            )

        # Both daisy FFI handlers call ptd_graph_update_weights(...,
        # /*use_log=*/false) directly (graph_builder_ffi.cpp:1528 for
        # DaisyChainJointProbs; :1782 and :1827 for DaisyChainSojourn), so a
        # 'log' graph would silently be evaluated with LINEAR weights.
        if getattr(self, '_weight_mode', 'linear') == 'log':
            raise ValueError(
                "daisy_chain_joint_probs() does not support weight_mode='log': "
                "the daisy FFI handlers compute edge weights linearly, so a "
                "'log' graph would silently receive LINEAR weights instead of "
                "the product rule 'log' implies. Use weight_mode='formula' or "
                "'callback'."
            )

        _ensure_jax_active()
        epoch_thetas_arr = jnp.asarray(epoch_thetas)
        if epoch_thetas_arr.ndim != 2:
            raise ValueError(
                f"epoch_thetas must have shape (n_epochs, theta_dim); "
                f"got ndim={epoch_thetas_arr.ndim}."
            )
        n_epochs = int(epoch_thetas_arr.shape[0])
        if n_epochs < 1:
            raise ValueError("epoch_thetas must contain at least one epoch.")
        if epoch_thetas_arr.shape[1] != self.param_length():
            raise ValueError(
                f"epoch_thetas theta_dim ({epoch_thetas_arr.shape[1]}) "
                f"does not match graph param_length ({self.param_length()})."
            )

        epoch_dts_seq = list(epoch_dts)
        if len(epoch_dts_seq) != n_epochs - 1:
            raise ValueError(
                f"epoch_dts must have length n_epochs - 1 = {n_epochs - 1}; "
                f"got {len(epoch_dts_seq)}."
            )

        initial_ipv_arr = jnp.asarray(initial_ipv, dtype=jnp.float64)
        n_ipv = len(self._ipv_target_indices)
        if initial_ipv_arr.shape != (n_ipv,):
            raise ValueError(
                f"initial_ipv must have shape ({n_ipv},); got "
                f"{initial_ipv_arr.shape}."
            )

        if t_eval is None:
            t_eval = max(float(sum(epoch_dts_seq)) * 4.0, 10.0)
        if t_eval <= 0:
            raise ValueError(f"t_eval must be > 0, got {t_eval}.")
        if not isinstance(granularity, (int, np.integer)) or granularity < 0:
            raise ValueError(
                f"granularity must be a non-negative integer, got {granularity!r}."
            )

        theta_dim = self.param_length()
        n_t = len(self._t_vertex_indices)

        # Build the structure JSON augmented with daisy-chain metadata.
        # GraphBuilder ignores unknown JSON fields, so the same JSON is
        # parsed twice in the FFI handler — once by GraphBuilder for
        # the topology, once by the daisy-chain handler for the
        # "_daisy_chain" sub-object. This mirrors how vanilla joint-
        # prob FFI passes structure_json as a single static attribute.
        from .ffi_wrappers import (
            _make_json_serializable,
            compute_daisy_chain_joint_probs_ffi,
            compute_daisy_chain_sojourn_ffi,
        )
        import json as _json_mod

        if final_read not in ('stopprob', 'sojourn'):
            raise ValueError(
                f"final_read must be 'stopprob' or 'sojourn', got {final_read!r}."
            )

        structure = _make_json_serializable(self.serialize(theta_dim=theta_dim))
        structure["_daisy_chain"] = {
            "n_epochs": int(n_epochs),
            "param_length": int(theta_dim),
            "t_eval": float(t_eval),
            "granularity": int(granularity),
            "epoch_dts": [float(x) for x in epoch_dts_seq],
            "ipv_target_indices": [int(x) for x in self._ipv_target_indices],
            "t_aux_keys": [int(k) for k in self._t_aux_map.keys()],
            "t_aux_values": [int(self._t_aux_map[k]) for k in self._t_aux_map.keys()],
            "t_vertex_indices": [int(x) for x in self._t_vertex_indices],
        }

        sojourn_json_str = None
        if final_read == 'sojourn':
            source = getattr(self, '_joint_prob_source', None)
            if source is None:
                raise ValueError(
                    "final_read='sojourn' requires a JSP graph built by "
                    "joint_stop_prob_graph() (it must carry _joint_prob_source)."
                )
            sg = source.joint_sojourn_graph()
            jsp_states = self.states()
            # JSP ipv-target state -> position in self._ipv_target_indices.
            jsp_ipv_pos = {
                tuple(int(x) for x in jsp_states[v]): k
                for k, v in enumerate(self._ipv_target_indices)
            }
            # sojourn_jsp_gather[k] = JSP ipv position whose state == sg's k-th
            # ipv target -> sojourn_ipv[k] = handoff_ipv[gather[k]].
            sojourn_jsp_gather = [jsp_ipv_pos[s] for s in sg._ipv_target_states]
            # Map each JSP t-vertex's full state to the corresponding sojourn
            # graph vertex, ordered to match self._t_vertex_indices.
            sg_states = sg.states()
            sg_state_to_idx = {
                tuple(int(x) for x in sg_states[v]): v
                for v in range(sg.vertices_length())
            }
            sojourn_t_indices = [
                sg_state_to_idx[tuple(int(x) for x in jsp_states[t])]
                for t in self._t_vertex_indices
            ]
            structure["_daisy_chain"]["sojourn_jsp_gather"] = [int(x) for x in sojourn_jsp_gather]
            structure["_daisy_chain"]["sojourn_t_indices"] = [int(x) for x in sojourn_t_indices]
            sojourn_json_str = _json_mod.dumps(
                _make_json_serializable(sg.serialize(theta_dim=theta_dim))
            )

        structure_json_str = _json_mod.dumps(structure)

        # The full forward computation as a flat-theta function. The
        # custom_vjp wrapper differentiates only theta_flat (initial_ipv
        # is closed over and treated as fixed). Single FFI call replaces
        # the per-epoch pure_callback chain.
        #
        # Note: this function is defined INSIDE daisy_chain_joint_probs,
        # which means closures over ``initial_ipv_arr`` /
        # ``structure_json_str`` are re-created on every call. That is
        # fine for ad-hoc one-shot use (forward eval, ``t_eval='auto'``
        # probe, etc.) but DOES NOT support a ``custom_vmap`` rule
        # wrapped around ``_forward`` — under ``vmap(jit(grad(...)))``
        # the rule's closure captures an enclosing-trace tracer that
        # later leaks into the inner jaxpr's consts as a
        # ``DynamicJaxprTracer``. The SVGD entry point sidesteps this
        # by building its own ``_forward`` + ``custom_vmap`` rule at
        # model-creation time in ``Graph._daisy_chain_svgd_model``'s
        # no-exposure branch (closures capture concrete arrays since
        # nothing is being traced at that point). Do not add
        # ``custom_vmap`` here.
        def _forward(theta_flat: jnp.ndarray) -> jnp.ndarray:
            if final_read == 'sojourn':
                return compute_daisy_chain_sojourn_ffi(
                    structure_json_str,
                    sojourn_json_str,
                    theta_flat,
                    initial_ipv_arr,
                )
            return compute_daisy_chain_joint_probs_ffi(
                structure_json_str,
                theta_flat,
                initial_ipv_arr,
            )

        # Wrap the forward in a custom_vjp so jax.grad works via finite
        # differences. eps=1e-7 matches the established pattern at
        # __init__.py:4322 (vanilla pmf_from_graph_joint_index).
        eps = 1e-7

        # Normalise fixed_indices to a Python set so the bwd rule can
        # skip FD evaluations on fixed slots — both saving compute and,
        # critically, preventing nonzero gradients from flowing into
        # parameters the user has pinned.
        fixed_set = (
            set(int(i) for i in fixed_indices)
            if fixed_indices is not None else set()
        )

        @jax.custom_vjp
        def _autodiff(theta_flat):
            return _forward(theta_flat)

        def _autodiff_fwd(theta_flat):
            return _forward(theta_flat), theta_flat

        def _autodiff_bwd(theta_flat, cotangent):
            n_params = theta_flat.shape[0]
            grads = []
            for i in range(n_params):
                if i in fixed_set:
                    grads.append(jnp.asarray(0.0, dtype=theta_flat.dtype))
                    continue
                tp = theta_flat.at[i].add(eps)
                tm = theta_flat.at[i].add(-eps)
                jp = _forward(tp)
                jm = _forward(tm)
                grads.append(jnp.sum(cotangent * (jp - jm) / (2.0 * eps)))
            return (jnp.stack(grads),)

        _autodiff.defvjp(_autodiff_fwd, _autodiff_bwd)

        return _autodiff(epoch_thetas_arr.reshape(-1))


    def _probe_daisy_t_eval(
        self,
        *,
        probe_thetas: np.ndarray,
        epoch_dts: list[float],
        initial_ipv: np.ndarray,
        tol: float = 1e-3,
        t_min: float | None = None,
        t_max: float | None = None,
        granularity: int = 0,
    ) -> float:
        """Pick the smallest ``t_eval`` whose residual non-t-vertex mass
        is below ``tol``.

        Mirrors the FFI handler's daisy-chain loop in pure Python: walks
        epochs 0..n_epochs-2 with `update_ipv → update_weights →
        stop_probability(dt)`, projecting the collapsed t-aux survival
        vector to the next epoch's IPV. Then in the final epoch, sets
        the propagated IPV and final-epoch theta and grows ``t`` by 1.5×
        until the residual transient mass at non-t vertices falls below
        ``tol``.

        Modifies ``self``'s IPV and weights as a side effect — the
        caller is responsible for restoring them, or for accepting that
        the next ``daisy_chain_joint_probs`` call will overwrite them.

        Parameters
        ----------
        probe_thetas : np.ndarray, shape (n_epochs, param_length)
            Per-epoch parameters used during the probe.
        epoch_dts : list of float, length n_epochs - 1
            Per-epoch durations (same as ``daisy_chain_joint_probs``).
        initial_ipv : np.ndarray, shape (n_ipv,)
            IPV for epoch 0, in the JSP graph's IPV layout.
        tol : float
            Residual-mass tolerance. Default 1e-3.
        t_min : float, optional
            Initial value to probe. Default ``max(sum(dts) * 0.5, 1.0)``.
        t_max : float, optional
            Upper bound. Default ``max(sum(dts) * 16, 40.0)``.
        granularity : int
            Forwarded to ``stop_probability``; 0 = auto.

        Returns
        -------
        float
            Chosen ``t_eval``.
        """
        if not getattr(self, '_joint_stop_prob_graph', False):
            raise ValueError(
                "_probe_daisy_t_eval requires a graph produced by "
                "joint_stop_prob_graph()."
            )

        n_epochs = int(probe_thetas.shape[0])
        if len(epoch_dts) != n_epochs - 1:
            raise ValueError(
                f"epoch_dts must have length n_epochs - 1 = {n_epochs - 1}; "
                f"got {len(epoch_dts)}."
            )

        n_vertices = self.vertices_length()
        t_aux_keys = list(self._t_aux_map.keys())
        t_aux_values = list(self._t_aux_map.values())
        aux_set = set(int(v) for v in t_aux_values)
        t_to_aux = {int(k): int(self._t_aux_map[k]) for k in t_aux_keys}
        ipv_target_indices = [int(x) for x in self._ipv_target_indices]
        t_vertex_indices = set(int(x) for x in self._t_vertex_indices)

        # Collapsed-position lookup (skip aux vertices).
        collapsed_pos = [-1] * n_vertices
        rank = 0
        for v in range(n_vertices):
            if v in aux_set:
                continue
            collapsed_pos[v] = rank
            rank += 1

        def _collapse(raw):
            """Sum t-vertex mass with its aux partner; return collapsed."""
            collapsed = np.zeros(rank, dtype=np.float64)
            for v in range(n_vertices):
                if v in aux_set:
                    continue
                p = float(raw[v])
                if v in t_to_aux:
                    p += float(raw[t_to_aux[v]])
                collapsed[collapsed_pos[v]] = p
            return collapsed

        # Walk transitions, propagating IPV.
        ipv_work = np.asarray(initial_ipv, dtype=np.float64).copy()
        for epoch in range(n_epochs - 1):
            self.update_ipv(ipv_work)
            self.update_weights(probe_thetas[epoch])
            raw = np.asarray(
                self.stop_probability(float(epoch_dts[epoch]), granularity=granularity)
            )
            collapsed = _collapse(raw)
            ipv_work = np.array(
                [collapsed[collapsed_pos[v]] for v in ipv_target_indices],
                dtype=np.float64,
            )

        # Final epoch: set IPV/theta, then grow t until residual mass
        # at non-t-vertices is below tol.
        self.update_ipv(ipv_work)
        self.update_weights(probe_thetas[n_epochs - 1])

        sum_dts = float(sum(epoch_dts))
        if t_min is None:
            t_min = max(sum_dts * 0.5, 1.0)
        if t_max is None:
            t_max = max(sum_dts * 16.0, 40.0)

        t = t_min
        while t <= t_max:
            raw = np.asarray(self.stop_probability(float(t), granularity=granularity))
            collapsed = _collapse(raw)
            non_t_mass = sum(
                float(collapsed[collapsed_pos[v]])
                for v in range(n_vertices)
                if v not in aux_set
                and v not in t_vertex_indices
                and v != self.starting_vertex().index()
            )
            if non_t_mass < tol:
                return float(t)
            t *= 1.5
        return float(t_max)


    def _resolve_daisy_chain_t_eval(
        self,
        *,
        daisy_chain_t_eval,
        epoch_starts,
        probe_theta=None,
        tol: float = 1e-3,
        granularity: int = 0,
        verbose: bool = False,
    ) -> float:
        """Resolve a ``daisy_chain_t_eval`` value (numeric, None, or
        ``'auto'``) into a numeric ``t_eval`` for the daisy-chain SVGD
        loop. ``self`` is the source joint-prob graph; we build the
        JSP graph internally for the probe.

        - Numeric: returned unchanged.
        - None: returns the legacy default ``max(sum(dts)*4, 10.0)``.
        - ``'auto'``: builds the JSP graph and calls
          ``_probe_daisy_t_eval`` with ``probe_theta`` (default
          ``[1.0, 1.0, ..., 1.0]`` per parameter slot).
        """
        es = np.asarray(epoch_starts, dtype=np.float64).ravel()
        epoch_dts = list(np.diff(es))
        sum_dts = float(sum(epoch_dts))
        legacy_default = max(sum_dts * 4.0, 10.0)

        if daisy_chain_t_eval is None:
            return legacy_default
        if isinstance(daisy_chain_t_eval, str):
            if daisy_chain_t_eval != 'auto':
                raise ValueError(
                    f"daisy_chain_t_eval must be a positive number, None, or "
                    f"'auto'; got {daisy_chain_t_eval!r}."
                )
        else:
            t_val = float(daisy_chain_t_eval)
            if t_val <= 0:
                raise ValueError(
                    f"daisy_chain_t_eval must be > 0, got {t_val}."
                )
            return t_val

        # 'auto' branch: build JSP graph, probe.
        jsp = self.joint_stop_prob_graph()
        n_epochs = int(es.size)
        param_length = self.param_length()
        if probe_theta is None:
            probe_thetas = np.ones((n_epochs, param_length), dtype=np.float64)
        else:
            probe_arr = np.asarray(probe_theta, dtype=np.float64)
            if probe_arr.ndim == 1:
                probe_thetas = np.broadcast_to(
                    probe_arr.reshape(1, -1), (n_epochs, param_length)
                ).copy()
            else:
                probe_thetas = probe_arr
            if probe_thetas.shape != (n_epochs, param_length):
                raise ValueError(
                    f"probe_theta must broadcast to shape "
                    f"({n_epochs}, {param_length}); got {probe_arr.shape}."
                )

        n_ipv = len(jsp._ipv_target_indices)
        self_ipv_full = np.zeros(self.vertices_length(), dtype=np.float64)
        for edge in self.starting_vertex().edges():
            self_ipv_full[edge.to().index()] = edge.weight()
        initial_ipv = self_ipv_full[jsp._ipv_target_indices]

        chosen = jsp._probe_daisy_t_eval(
            probe_thetas=probe_thetas,
            epoch_dts=epoch_dts,
            initial_ipv=initial_ipv,
            tol=tol,
            granularity=granularity,
        )

        if verbose:
            logger = get_logger(__name__)
            logger.info(
                "daisy_chain_t_eval='auto': probed t_eval=%.4f (legacy default "
                "would have been %.4f, ratio=%.2fx)",
                chosen, legacy_default, legacy_default / max(chosen, 1e-12),
            )
        return float(chosen)


    def _get_joint_probs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self._joint_prob_base_graph_indexer:
            raise ValueError("Graph is not a joint probability representation.")
            
        t_vertex_indices = []
        for vertex in self.vertices():
            for edge in vertex.edges():
                if len(edge.to().edges()) == 0:
                    t_vertex_indices.append(vertex.index())
                    break
        t_vertex_indices = np.unique(t_vertex_indices)
        states = self.states()

        joint_reward_state_indices = np.arange(self._joint_prob_base_graph_indexer.state_length, self.state_length())
        state_reward_matrix = states[t_vertex_indices, :][:, joint_reward_state_indices]

        idx2obs = {}
        for rewards, idx in zip(state_reward_matrix, t_vertex_indices):
            idx2obs[int(idx)] = tuple(rewards.tolist())

        t_indices = list(idx2obs.keys())
        sojourn_times = self.expected_sojourn_time(t_indices)
        assert len(sojourn_times) == len(t_indices)
        outcomes = []
        probs = []
        t_index = []
        for idx, prob in zip(t_indices, sojourn_times):
            t_index.append(idx)
            obs = idx2obs[idx]
            outcomes.append(np.array(obs))
            probs.append(prob)

        return np.array(outcomes), np.array(probs), np.array(t_index)


    def _joint_prob_columns(self) -> list[str]:
        """Column names for a joint-prob DataFrame: reward columns + 'prob' + 't_vertex_index'.

        Used by both :meth:`joint_prob_table` (which reads the asymptotic discrete
        probabilities) and :class:`EpochContext` (which reads the continuous
        cumulative probabilities at a finite time).
        """
        if not self._rewarded_props:
            raise ValueError(
                "Graph does not carry _rewarded_props; only joint-prob graphs "
                "(built via joint_prob_graph) and graphs derived from them "
                "(joint_stop_prob_graph) have column metadata."
            )
        column_names: list[str] = []
        for p in self._rewarded_props:
            for i in range(p.min_value, p.max_value + 1):
                column_names.append(f"{p.name}_{i}")
        column_names.extend(['prob', 't_vertex_index'])
        return column_names

    def joint_prob_table(self) -> pd.DataFrame:

        if not (self._joint_prob_base_graph_indexer):
            raise ValueError("Graph must be discrete and a joint probability representation.")

        outcomes, probs, t_vertex_indices = self._get_joint_probs()

        records = [[*obs, prob, idx] for obs, prob, idx in zip(outcomes, probs, t_vertex_indices)]
        joint = pd.DataFrame(records, columns=self._joint_prob_columns()).set_index('t_vertex_index')
        return joint

    # Registry cache transfer (pull/push) lives in _graph_cache_transfer.py
    # (WS-C split); assigned here so they stay direct Graph members.
    pull_cache = _graph_cache_transfer.pull_cache
    push_cache = _graph_cache_transfer.push_cache


class EpochContext:
    """Read cumulative t-state probabilities of a continuous joint-prob graph at arbitrary times.

    Built via :meth:`Graph.epoch_context`. Wraps a joint-stop-probability
    (JSP) graph plus the indexing metadata needed to collapse each
    t-vertex's mass with its trapping aux partner, so that

        M_i(t) = P(absorbed via t-state i by time t)

    can be read directly. As ``t → ∞`` each ``M_i(t)`` converges to the
    discrete asymptote :meth:`Graph.joint_prob_table` would give for the
    same model.

    The context is a thin Python wrapper — the heavy lifting (graph
    transformation, forward computation) lives on the underlying graph
    at :attr:`_graph`. Typical usage::

        cjpg = graph.joint_prob_graph(..., discrete=False)
        cjpg.update_weights(theta)        # optional — inherited below
        ctx  = cjpg.epoch_context()       # inherits current theta of cjpg
        probs = ctx.cumulative_probs(np.linspace(0, 4, 40))  # (40, n_t)

    If the source graph has had ``update_weights`` called on it, the
    context inherits that ``theta`` at construction time. Use
    :meth:`update_weights` on the context only when you want to change
    parameters after creation.

    Attributes
    ----------
    _graph : Graph
        The joint-stop-probability (JSP) graph (output of
        :meth:`Graph.joint_stop_prob_graph`). The user can pass this to
        any standard ``Graph`` method.
    _source_graph : Graph
        The continuous joint-prob graph this context was built from.
        Kept for downstream introspection (e.g. retrieving
        :meth:`joint_prob_table` of the matching discrete graph).
    _t_vertex_indices : list[int]
        Sorted t-vertex indices in the JSP graph. These are also the
        ``t_vertex_index`` keys in the source graph's
        :meth:`joint_prob_table` output.
    _t_aux_map : dict[int, int]
        ``{t_vertex_idx: aux_vertex_idx}`` mapping for the trapping
        aux-loop pairs installed by :meth:`Graph.joint_stop_prob_graph`.
    _ipv_target_indices : list[int]
        Vertex indices in the JSP graph that carry the per-epoch IPV
        edges. Propagated for symmetry with the JSP graph; rarely
        needed by users.
    """

    def __init__(self, jsp_graph: 'Graph', source_graph: 'Graph') -> None:
        if not getattr(jsp_graph, '_joint_stop_prob_graph', False):
            raise ValueError(
                "EpochContext requires a graph produced by joint_stop_prob_graph()."
            )
        self._graph = jsp_graph
        self._source_graph = source_graph
        self._t_vertex_indices = list(jsp_graph._t_vertex_indices)
        self._t_aux_map = dict(jsp_graph._t_aux_map)
        self._ipv_target_indices = list(jsp_graph._ipv_target_indices)
        self._param_length = jsp_graph.param_length()
        self._n_vertices = jsp_graph.vertices_length()

    def update_weights(self, theta: ArrayLike) -> None:
        """Set the per-epoch parameter vector ``theta`` on the underlying JSP graph."""
        self._graph.update_weights(theta)

    def update_ipv(self, ipv: ArrayLike) -> None:
        """Override the initial-probability vector on the underlying JSP graph.

        Auto-seeded from the source graph's starting-vertex edges at
        construction time (see :meth:`Graph.epoch_context`); call this
        only if you want a non-default IPV (for example, a propagated
        IPV from a previous epoch).
        """
        self._graph.update_ipv(ipv)

    def cumulative_probs(
        self,
        t: float | ArrayLike | None = None,
        *,
        tol: float = 1e-3,
        granularity: int = 0,
        table: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Cumulative absorption probability at each t-state by time ``t``.

        For each t-state ``i`` the returned value is
        ``stop_probability(t)[t_i] + stop_probability(t)[aux(t_i)]`` —
        the total mass that has ever been trapped in the (t, aux) pair
        for that state.

        Parameters
        ----------
        t : float, 1D array-like, or None, default None
            Time(s) at which to evaluate.

            * scalar → 1D array of length ``len(_t_vertex_indices)``
              (``table=False``, default) or a :class:`pandas.DataFrame`
              (``table=True``).
            * 1D array → 2D output of shape ``(len(t), n_t_states)``.
              Requires ``table=False`` (an array-valued ``t`` is
              rejected when ``table=True``).
            * ``None`` → auto-pick a ``t`` large enough that the residual
              transient mass at non-t / non-aux / non-start vertices
              falls below ``tol``. Mirrors the policy used by the C++
              daisy-chain handler in SVGD (see
              :meth:`auto_t` / :meth:`Graph._probe_daisy_t_eval`).
        tol : float, default 1e-3
            Residual-mass tolerance used when ``t`` is ``None``. Ignored
            otherwise.
        granularity : int, default 0
            Uniformization granularity forwarded to
            :meth:`Graph.stop_probability`. ``0`` lets the underlying
            implementation auto-pick a safe value.
        table : bool, default False
            If True, return a :class:`pandas.DataFrame` matching the
            layout of :meth:`Graph.joint_prob_table`. Requires scalar
            (or ``None``) ``t``.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Cumulative probabilities. See ``t`` and ``table`` for
            shape / type.
        """
        if t is None:
            t = self.auto_t(tol=tol, granularity=granularity)
        t_arr = np.asarray(t, dtype=np.float64)
        if table and t_arr.ndim != 0:
            raise ValueError(
                "table=True requires a scalar t (or t=None). For an "
                "array of times, call cumulative_probs(..., table=False) "
                "and build the frame yourself."
            )
        if t_arr.ndim == 0:
            probs = self._collapse_one(float(t_arr), granularity=granularity)
            if table:
                outcomes, _, t_vertex_indices = self._source_graph._get_joint_probs()
                records = [
                    [*obs, prob, idx]
                    for obs, prob, idx in zip(outcomes, probs, t_vertex_indices)
                ]
                return pd.DataFrame(
                    records, columns=self._source_graph._joint_prob_columns()
                ).set_index('t_vertex_index')
            return probs
        if t_arr.ndim != 1:
            raise ValueError(
                f"t must be a scalar, 1D array, or None; got ndim={t_arr.ndim}."
            )
        n_t = t_arr.shape[0]
        n_states = len(self._t_vertex_indices)
        out = np.empty((n_t, n_states), dtype=np.float64)
        for k, tk in enumerate(t_arr):
            out[k] = self._collapse_one(float(tk), granularity=granularity)
        return out

    def auto_t(
        self,
        *,
        tol: float = 1e-3,
        t_min: float = 1.0,
        t_max: float = 1024.0,
        granularity: int = 0,
    ) -> float:
        """Smallest ``t`` whose residual transient mass falls below ``tol``.

        Walks ``t = t_min, t_min*1.5, t_min*1.5^2, …`` and stops as soon
        as the sum of mass at non-t / non-aux / non-start vertices is
        below ``tol`` — i.e. once nearly all probability mass has either
        absorbed into a t-state trap or been routed to the trash/abs
        vertex. Same heuristic as the C++ daisy-chain handler and
        :meth:`Graph._probe_daisy_t_eval`.

        Parameters
        ----------
        tol : float, default 1e-3
            Residual-mass cutoff.
        t_min : float, default 1.0
            Starting probe value.
        t_max : float, default 1024.0
            Upper bound. Returned if ``tol`` is never reached.
        granularity : int, default 0
            Forwarded to :meth:`Graph.stop_probability`.

        Returns
        -------
        float
            The chosen ``t``.
        """
        start_idx = self._graph.starting_vertex().index()
        t_vertex_set = set(self._t_vertex_indices)
        aux_set = set(self._t_aux_map.values())
        # Indices whose mass we want driven to ~0 for a "settled" read.
        residual_indices = [
            v for v in range(self._n_vertices)
            if v != start_idx and v not in t_vertex_set and v not in aux_set
        ]
        t = float(t_min)
        while t <= t_max:
            raw = np.asarray(
                self._graph.stop_probability(t, granularity=granularity),
                dtype=np.float64,
            )
            residual = float(np.sum(raw[residual_indices]))
            if residual < tol:
                return t
            t *= 1.5
        return float(t_max)

    def next_ipv(
        self,
        t: float | None = None,
        *,
        tol: float = 1e-3,
        granularity: int = 0,
    ) -> np.ndarray:
        """IPV for the next epoch: per-vertex probabilities at time ``t`` collapsed onto the JSP graph's IPV layout.

        Use this to daisy-chain epochs by hand: feed the returned vector
        to ``next_ctx.update_ipv(...)`` (or to the next graph's
        ``update_ipv``) so the next epoch starts where this one left off.

        Each entry is the mass at one JSP-graph vertex listed in
        ``_ipv_target_indices``, with t-vertex / aux-partner pairs
        summed (matching :meth:`cumulative_probs`). Aux vertices, trash
        pairs, the absorbing vertex, and the starting vertex are
        excluded — i.e. the same layout the JSP graph's IPV edges use.

        For the multi-epoch SVGD pipeline, prefer
        :meth:`Graph.daisy_chain_joint_probs`; this method is the
        equivalent for ad-hoc per-epoch reads.

        Parameters
        ----------
        t : float or None, default None
            Time at which to evaluate. ``None`` → auto-pick via
            :meth:`auto_t` with the supplied ``tol``.
        tol : float, default 1e-3
            Residual-mass tolerance used when ``t`` is ``None``. Ignored
            otherwise.
        granularity : int, default 0
            Forwarded to :meth:`Graph.stop_probability`.

        Returns
        -------
        np.ndarray
            1D vector of length ``len(_ipv_target_indices)``, in the
            same order. Pass directly to the next epoch's
            ``update_ipv``.
        """
        if t is None:
            t = self.auto_t(tol=tol, granularity=granularity)
        if np.ndim(t) != 0:
            raise ValueError(
                f"next_ipv requires a scalar t or None; got ndim={np.ndim(t)}."
            )
        raw = np.asarray(
            self._graph.stop_probability(float(t), granularity=granularity),
            dtype=np.float64,
        )
        out = np.empty(len(self._ipv_target_indices), dtype=np.float64)
        for i, v in enumerate(self._ipv_target_indices):
            p = raw[v]
            if v in self._t_aux_map:
                p += raw[self._t_aux_map[v]]
            out[i] = p
        return out

    def joint_probs_table(
        self,
        t: float | None = None,
        *,
        tol: float = 1e-3,
        granularity: int = 0,
    ) -> pd.DataFrame:
        """Alias for :meth:`cumulative_probs` with ``table=True``.

        Returns a :class:`pandas.DataFrame` matching the layout of
        :meth:`Graph.joint_prob_table`. ``t`` must be a scalar (or
        ``None`` to auto-pick); use :meth:`cumulative_probs` for arrays
        of times.
        """
        return self.cumulative_probs(
            t, tol=tol, granularity=granularity, table=True
        )

    def _collapse_one(self, t: float, *, granularity: int = 0) -> np.ndarray:
        """One stop_probability call + t/aux collapse → 1D probability vector."""
        raw = np.asarray(
            self._graph.stop_probability(t, granularity=granularity),
            dtype=np.float64,
        )
        out = np.empty(len(self._t_vertex_indices), dtype=np.float64)
        for i, t_idx in enumerate(self._t_vertex_indices):
            out[i] = raw[t_idx] + raw[self._t_aux_map[t_idx]]
        return out


# Module-level utility functions

def load_cpp_builder(cpp_file: str | pathlib.Path) -> Callable:
    """
    Load a C++ model builder for direct Graph object creation without JAX wrapping.

    This function compiles a user-provided C++ file and returns a builder function
    that creates Graph objects directly. Use this when you need fast forward
    evaluations without JAX support or gradient computation.

    For gradient-based inference and automatic differentiation, use pmf_from_cpp()
    instead, which wraps the C++ model in a JAX-compatible function.

    Parameters
    ----------
    cpp_file : str or pathlib.Path
        Path to C++ file implementing the build_model() function.
        See examples/user_models/README.md for details on the required interface.

    Returns
    -------
    callable
        Builder function with signature: (theta: np.ndarray) -> Graph
        - Input: Parameter vector (numpy array)
        - Output: Graph object with standard methods (pdf, pmf, moments, etc.)

    Examples
    --------
    >>> # Load a C++ coalescent model
    >>> builder = load_cpp_builder("models/coalescent.cpp")
    >>>
    >>> # Create graph with specific parameters
    >>> graph = builder(np.array([1.0, 2.0]))
    >>>
    >>> # Use standard Graph methods for forward evaluation
    >>> pdf_value = graph.pdf(1.0)  # Direct C++ call, no JAX overhead
    >>> pmf_value = graph.dph_pmf(5)
    >>> moment = graph.moments(2)  # E[T^2]
    >>>
    >>> # For gradient-based inference, use pmf_from_cpp instead:
    >>> model = Graph.pmf_from_cpp("models/coalescent.cpp")
    >>> # Now you can use jax.grad(model) for automatic differentiation

    See Also
    --------
    Graph.pmf_from_cpp : JAX-compatible wrapper for gradient computation
    Graph.pmf_from_graph : Convert Python-built Graph to JAX function

    Notes
    -----
    - This function does NOT provide JAX integration or gradient support
    - Suitable for scenarios where you need repeated fast evaluations with different parameters
    - The C++ file must implement: Graph* build_model(const double* theta, int dim)
    - For distributed/GPU computing with JAX, use pmf_from_cpp() instead
    """
    from . import phasic_pybind
    cpp_path = pathlib.Path(cpp_file).resolve()
    if not cpp_path.exists():
        raise FileNotFoundError(f"C++ file not found: {cpp_path}")
    return phasic_pybind.load_cpp_builder(str(cpp_path))


# ============================================================================
# Automatic Parallelization API
# ============================================================================

def init_parallel(cpus: int | None = None,
                  force: bool = False,
                  enable_x64: bool = True) -> ParallelConfig:
    """
    Initialize parallel computing with automatic resource detection.

    This function configures JAX for optimal multi-CPU/device usage based on
    the execution environment. It should be called at the top of your script
    or notebook before any JAX operations for best results.

    Environment Detection:
    - Jupyter/IPython: Uses all available CPUs on local machine
    - SLURM single-node: Uses allocated CPUs (SLURM_CPUS_PER_TASK)
    - SLURM multi-node: Initializes distributed JAX across all nodes
    - Script: Uses all available CPUs

    Parameters
    ----------
    cpus : int, optional
        Number of CPUs to use. If None, auto-detects based on environment.
        - Local: os.cpu_count()
        - SLURM: SLURM_CPUS_PER_TASK
    force : bool, default=False
        If True, attempts to reconfigure even if JAX already imported.
        Note: May require kernel restart if JAX is already imported.
    enable_x64 : bool, default=True
        Enable 64-bit precision in JAX for numerical accuracy

    Returns
    -------
    ParallelConfig
        Configuration object containing:
        - device_count: Number of JAX devices available
        - strategy: Parallelization strategy ('pmap', 'vmap', or 'none')
        - env_info: Detected environment information

    Raises
    ------
    RuntimeError
        If force=True but JAX is already imported (requires kernel restart)

    Examples
    --------
    >>> # At top of Jupyter notebook - uses all available CPUs
    >>> import phasic as pta
    >>> config = pta.init_parallel()
    >>> print(f"Configured {config.device_count} devices")
    >>>
    >>> # Explicit CPU count
    >>> config = pta.init_parallel(cpus=8)
    >>>
    >>> # Now all Graph operations automatically parallelize
    >>> g = pta.Graph(...)
    >>> pdf = g.pdf_batch(times)  # Auto-parallelized!

    >>> # On SLURM cluster (auto-detects allocation)
    >>> # sbatch --cpus-per-task=16 my_script.sh
    >>> config = pta.init_parallel()  # Uses all 16 CPUs

    Notes
    -----
    - For optimal performance, call this before importing JAX or creating graphs
    - If JAX is already imported, you'll get a warning and suboptimal configuration
    - To reconfigure, restart your kernel and call init_parallel() first
    - The configuration applies globally to all subsequent phasic operations

    See Also
    --------
    detect_environment : Inspect environment without configuring
    configure : phasic.configure(svgd_strategy=...) controls SVGD
        particle parallelism strategy (auto/pmap/vmap/none).
    """
    # Detect environment
    env_info = detect_environment()

    # Override CPU count if specified
    if cpus is not None:
        env_info.available_cpus = cpus

    # Check if force is needed
    if force and env_info.jax_already_imported:
        raise RuntimeError(
            "Cannot reconfigure JAX after import. Please restart kernel and "
            "call init_parallel() before any JAX operations."
        )

    # Configure JAX for environment.
    config = configure_jax_for_environment(env_info, enable_x64=enable_x64)

    # Note: previous versions stored `config` in a module-level
    # global via set_parallel_config(). That global was unread
    # after the config refactor (SVGD reads its strategy from
    # phasic.get_config().svgd_strategy instead) and has been
    # removed. The returned ParallelConfig is still useful
    # informationally.
    return config


# ============================================================================
# Export JAX Configuration and Model Export Utilities
# ============================================================================

# Make CompilationConfig and utilities available to users
if HAS_JAX:
    from .jax_config import CompilationConfig, get_default_config as get_jax_config, set_default_config as set_jax_config
    from . import model_export

    # Expose common model_export functions at package level
    from .model_export import (
        clear_jax_cache,
        cache_info,
        print_model_cache_info,
        export_model_package,
        generate_warmup_script
    )

# ============================================================================
# Public Configuration API
# ============================================================================

# Export configuration system to package namespace
# These are already imported at the top, just documenting them as public API
__all_config__ = [
    'configure',
    'get_config',
    'get_available_options',
    'PhasicConfig',
    'reset_config',
    'PTDAlgorithmsError',
    'PTDConfigError',
    'PTDBackendError',
    'PTDFeatureError',
    'PTDJAXError',
]

# Export callback decorator for building parameterized graphs
callback = _callback


# ============================================================================
# SCCVertex.as_graph: wrap returned _Graph in the Python Graph subclass so
# users get the full Python API (e.g. Graph.from_matrices, expectation, etc.)
# instead of a bare pybind _Graph instance. The underlying C++ method returns
# a _Graph; we wrap it post-hoc here so isinstance(..., Graph) holds.
# ============================================================================
try:
    from .phasic_pybind import SCCVertex as _SCCVertex
    _scc_vertex_as_graph_raw = _SCCVertex.as_graph

    def _scc_vertex_as_graph(self):
        return Graph(_scc_vertex_as_graph_raw(self))

    _scc_vertex_as_graph.__doc__ = _scc_vertex_as_graph_raw.__doc__
    _SCCVertex.as_graph = _scc_vertex_as_graph
except (ImportError, AttributeError):
    # SCC API unavailable in this build; nothing to wrap.
    pass


# ------------------------------------------------------------------
# Lazy JAX-dependent attributes
#
# JAX-dependent exports (SVGD, MCMC, BFFG helpers, optax_*) are
# bound to None at import time so `import phasic` stays light.
# The module-level __getattr__ below triggers _ensure_jax_active()
# and rebinds the symbols on first access. After that, normal
# attribute lookup picks them up directly.
# ------------------------------------------------------------------

_JAX_LAZY_NAMES = frozenset({
    # svgd
    'SVGD',
    'Prior', 'GaussPrior', 'LogGaussPrior', 'HalfCauchyPrior', 'BetaPrior', 'DataPrior',
    'StepSizeSchedule', 'ConstantStepSize', 'ExpStepSize',
    'AdaptiveStepSize', 'WarmupExpStepSize',
    'Adam', 'Adamelia', 'SGDMomentum', 'RMSprop', 'Adagrad',
    'RegularizationSchedule', 'ConstantRegularization',
    'ExpRegularization', 'ExponentialCDFRegularization',
    'FisherPreconditioner', 'MomentJacobianPreconditioner',
    'ProbabilityJacobianPreconditioner',
    'SparseObservations', 'dense_to_sparse', 'is_sparse_observations',
    # mcmc
    'MCMC',
    # bffg
    'path_to_rewards', 'path_exit_rates', 'path_exit_rates_by_param',
    'importance_log_weight_from_rates',
    'importance_weighted_log_likelihood', 'bffg_log_prob',
})


def __getattr__(name: str):
    """Lazy resolution for JAX-dependent exports.

    On first access to any name in `_JAX_LAZY_NAMES`, this
    triggers _ensure_jax_active() (which imports JAX, sets up
    XLA_FLAGS, applies CompilationConfig.balanced(), etc.) and
    then imports the relevant module to bind the symbol on the
    package, so subsequent accesses are direct.
    """
    if name not in _JAX_LAZY_NAMES:
        raise AttributeError(f"module 'phasic' has no attribute {name!r}")

    _ensure_jax_active()

    # Import the relevant submodule and pull out the symbol.
    import importlib
    candidates = [
        ('svgd', _JAX_LAZY_NAMES - {'MCMC', 'path_to_rewards',
                                     'path_exit_rates',
                                     'path_exit_rates_by_param',
                                     'importance_log_weight_from_rates',
                                     'importance_weighted_log_likelihood',
                                     'bffg_log_prob'}),
        ('mcmc', {'MCMC'}),
        ('bffg', {'path_to_rewards', 'path_exit_rates',
                  'path_exit_rates_by_param',
                  'importance_log_weight_from_rates',
                  'importance_weighted_log_likelihood',
                  'bffg_log_prob'}),
    ]
    for modname, names in candidates:
        if name in names:
            mod = importlib.import_module(f'phasic.{modname}')
            value = getattr(mod, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'phasic' has no attribute {name!r}")


# Reusable free daisy-chain epoch model returned by Graph.epoch_model. Not
# JAX-dependent at import (SVGD is imported lazily inside FreeEpochModel.fit),
# so bind it directly rather than via the JAX-lazy __getattr__ above.
from .epoch_model import FreeEpochModel  # noqa: E402,F401
