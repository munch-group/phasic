from __future__ import annotations

from ast import arg
from functools import partial
from collections import defaultdict, OrderedDict
from itertools import product, zip_longest
from unittest import result
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Any, TypeVar, Self
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
    PTDAlgorithmsConfig,
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

# from .vscode_theme import set_phasic_theme
# from .vscode_theme import phasic_theme as theme
# from .vscode_theme import set_theme # backwards compatibility
# from . import plot

# Get configuration (creates default if none exists)
_config = get_config()

# Configure JAX environment BEFORE importing (if JAX will be used)
if _config.jax:
    import sys

    # Configure JAX for multi-CPU BEFORE importing JAX
    if 'jax' in sys.modules:
        # JAX already imported - this prevents multi-CPU configuration
        raise ImportError(
            "JAX must NOT be imported before phasic.\n"
            "This prevents multi-CPU device configuration and will cause poor performance.\n\n"
            "REQUIRED import order:\n"
            "  from phasic import Graph, SVGD, ...\n"
            "  import jax  # Import JAX AFTER phasic\n"
            "  import jax.numpy as jnp\n\n"
            "Note: phasic automatically:\n"
            "  - Enables x64 precision for accurate gradients\n"
            "  - Configures multi-CPU support (8 devices on this system)\n"
            "  - Sets up JAX compilation cache\n\n"
            "If you need to override CPU count, set PTDALG_CPUS before import:\n"
            "  export PTDALG_CPUS=4\n"
            "  python your_script.py"
        )
    else:
        # Import compilation configuration system
        from .jax_config import CompilationConfig, get_default_config, set_default_config

        # Apply default balanced configuration (includes JAX persistent cache)
        default_config = get_default_config()
        default_config.apply(force=False)  # Don't override existing user configuration

        # Detect performance cores on Apple Silicon for multi-CPU
        def get_available_cpus() -> int:
            """Get number of CPUs available to this process.

            Priority:
            1. Apple Silicon P-cores (macOS ARM64 only)
            2. SLURM allocation (SLURM_CPUS_PER_TASK or SLURM_CPUS_ON_NODE)
            3. OS-reported affinity (respects cgroups)
            4. Total CPU count (last resort)
            """
            try:
                import subprocess
                import platform

                # Check if we're on Apple Silicon
                if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                    result = subprocess.run(
                        ['sysctl', '-n', 'hw.perflevel0.physicalcpu'],
                        capture_output=True, text=True, check=True
                    )
                    p_cores = int(result.stdout.strip())
                    return p_cores
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass

            # SLURM: respect allocated CPUs, not full node count
            for var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
                val = os.environ.get(var)
                if val is not None:
                    try:
                        return max(int(val), 1)
                    except ValueError:
                        pass

            # os.sched_getaffinity respects cgroup restrictions (Linux)
            try:
                return len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                pass

            return os.cpu_count() or 1

        # Configure multi-device CPU count (for pmap)
        cpu_count = int(os.environ.get('PTDALG_CPUS', get_available_cpus()))
        xla_flags = os.environ.get('XLA_FLAGS', '')
        device_flag = f"--xla_force_host_platform_device_count={cpu_count}"

        if '--xla_force_host_platform_device_count' not in xla_flags:
            if xla_flags:
                xla_flags += f" {device_flag}"
            else:
                xla_flags = device_flag
            os.environ['XLA_FLAGS'] = xla_flags


    # Set JAX platform before import
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

    # Filter to suppress JAX device list output
    class _DeviceListFilter:
        def __init__(self, original: Any) -> None:
            self.original = original
            self.buffer = ''

        def write(self, text: str) -> None:
            # Buffer the text to check full lines
            self.buffer += text

            # Process complete lines
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                line += '\n'

                # Filter out device list lines
                if not ('CpuDevice' in line or 'GpuDevice' in line):
                    self.original.write(line)

        def flush(self) -> None:
            # Flush any remaining buffer (except device lists)
            if self.buffer and not ('CpuDevice' in self.buffer or 'GpuDevice' in self.buffer):
                self.original.write(self.buffer)
                self.buffer = ''
            self.original.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.original, name)

    # Install filter BEFORE importing JAX (and keep it active)
    if not isinstance(sys.stdout, _DeviceListFilter):
        sys.stdout = _DeviceListFilter(sys.stdout)
    if not isinstance(sys.stderr, _DeviceListFilter):
        sys.stderr = _DeviceListFilter(sys.stderr)

    # Import JAX (raise clear error if unavailable)
    try:
        import jax
        jax.config.update('jax_enable_x64', True)  # Enable 64-bit precision for accurate gradients
        import jax.numpy as jnp
        HAS_JAX = True
    except ImportError as e:
        raise PTDJAXError(
            "jax=True but JAX not installed.\n"
            "  Install: pip install jax jaxlib\n"
            "  Or configure before import: phasic.configure(jax=False)\n"
            f"  Original error: {e}"
        )
else:
    # JAX disabled by configuration
    jax = None
    jnp = None
    HAS_JAX = False

# Cache for compiled libraries
_lib_cache = {}

from .phasic_pybind import *
from .phasic_pybind import Graph as _Graph
from .phasic_pybind import Vertex, Edge

# Configure package-wide logging
from .logging_config import setup_logging, get_logger
setup_logging()

# Optional SVGD support (requires JAX)
if HAS_JAX:
    from .svgd import (
        SVGD,
        # Prior classes
        Prior,
        GaussPrior,
        HalfCauchyPrior,
        DataPrior,
        # Step size schedules
        StepSizeSchedule,
        ConstantStepSize,
        ExpStepSize,
        AdaptiveStepSize,
        WarmupExpStepSize,
        # Optimizers
        Adam,
        Adamelia,
        SGDMomentum,
        RMSprop,
        Adagrad,
        # Regularization schedules
        RegularizationSchedule,
        ConstantRegularization,
        ExpRegularization,
        ExponentialCDFRegularization,
        # # Bandwidth schedules
        # BandwidthSchedule,
        # MedianBandwidth,
        # FixedBandwidth,
        # LocalAdaptiveBandwidth
        # Preconditioning
        FisherPreconditioner,
        MomentJacobianPreconditioner,
        # Sparse observations for multivariate SVGD
        SparseObservations,
        dense_to_sparse,
        is_sparse_observations,
    )
    from .mcmc import MCMC
    from .bffg import (
        path_to_rewards,
        path_exit_rates,
        path_exit_rates_by_param,
        importance_log_weight_from_rates,
        importance_weighted_log_likelihood,
        bffg_log_prob,
    )
else:
    SVGD = None
    MCMC = None
    path_to_rewards = None
    path_exit_rates = None
    path_exit_rates_by_param = None
    importance_log_weight_from_rates = None
    importance_weighted_log_likelihood = None
    bffg_log_prob = None
    Prior = None
    GaussPrior = None
    HalfCauchyPrior = None
    DataPrior = None
    StepSizeSchedule = None
    ConstantStepSize = None
    ExpStepSize = None
    AdaptiveStepSize = None
    WarmupExpStepSize = None
    RegularizationSchedule = None
    ConstantRegularization = None
    ExpRegularization = None
    ExponentialCDFRegularization = None
    Adam = None
    Adamelia = None
    SGDMomentum = None
    RMSprop = None
    Adagrad = None
    # BandwidthSchedule = None
    # MedianBandwidth = None
    # FixedBandwidth = None
    # LocalAdaptiveBandwidth = None
    FisherPreconditioner = None
    MomentJacobianPreconditioner = None
    SparseObservations = None
    dense_to_sparse = None
    is_sparse_observations = None

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
    get_parallel_config,
    set_parallel_config,
    parallel_config,
    disable_parallel,
)

# Cache management (JAX compilation cache)
from .cache_manager import CacheManager, print_jax_cache_info, configure_layered_cache
from .model_export import clear_caches, clear_jax_cache, clear_model_cache, cache_info, print_model_cache_info, get_all_cache_stats, print_all_cache_info
from .trace_cache import get_trace_cache_stats, print_trace_cache_info

# On-disk cache management (~/.phasic_cache/) — symbolic compute
# graph cache (Stage A2) and Python trace cache.
from . import cache
from .jax_config import CompilationConfig, get_default_config, set_default_config
# from .cloud_cache import (
#     S3Backend,
#     GCSBackend,
#     AzureBlobBackend,
#     download_from_url,
#     download_from_github_release,
#     install_model_library
# )
from .trace_repository import (
    TransportBackend,
    IPFSBackend,
    TraceRegistry,
    get_trace,
    install_trace_library,
    get_ipfs_dir,
    generate_swarm_key,
    install_swarm_key,
    detect_swarm_key,
    remove_swarm_key,
    configure_bootstrap_peers,
)
from .trace_elimination import EliminationTrace


# Hash-based trace lookup (convenience wrapper)
def get_trace_by_hash(graph_hash: str, force_download: bool = False, backend: TransportBackend | None = None) -> EliminationTrace | None:
    """
    Get elimination trace by graph structure hash.

    Convenience wrapper around TraceRegistry.get_trace_by_hash().

    Parameters
    ----------
    graph_hash : str
        SHA-256 hash of graph structure (from phasic.hash.compute_graph_hash)
    force_download : bool, default=False
        If True, re-download even if cached
    backend : TransportBackend, optional
        Custom transport backend for content retrieval.

    Returns
    -------
    EliminationTrace or None
        Trace if found, None otherwise

    Examples
    --------
    >>> import phasic
    >>> import phasic.hash
    >>> graph = phasic.Graph(my_callback, nr_samples=5)
    >>> hash_result = phasic.hash.compute_graph_hash(graph)
    >>> trace = phasic.get_trace_by_hash(hash_result.hash_hex)
    >>> if trace is None:
    ...     # Record new trace
    ...     from phasic.trace_elimination import record_elimination_trace
    ...     trace = record_elimination_trace(graph, theta_dim=1)
    """
    registry = TraceRegistry(backend=backend) if backend else TraceRegistry()
    return registry.get_trace_by_hash(graph_hash, force_download=force_download)

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


def _create_jax_callback_wrapper(compute_func: Any, graph_data: dict, discrete: bool) -> Callable:
    """
    Create a pure JAX-compatible callback wrapper.

    Returns a function compatible with jax.pure_callback that maintains
    purity and supports JAX transformations.
    """
    from jax import pure_callback

    def compute_pmf_pure(times, granularity=100):
        """Pure function wrapper for JAX compatibility"""
        def compute_impl(times_arr):
            # For parameterized models, theta comes from outer scope
            # For static models, graph_data is fixed
            return _compute_pmf_from_ctypes(
                np.array([]),  # Empty theta for static graphs
                times_arr,
                compute_func,
                graph_data,
                granularity,
                discrete
            )

        result_shape_dtypes = jax.ShapeDtypeStruct(times.shape, jnp.float32)
        return pure_callback(compute_impl, result_shape_dtypes, times)

    return compute_pmf_pure


def _create_jax_parameterized_wrapper(compute_func: Any, graph_builder: Callable, discrete: bool) -> Callable:
    """
    Create a pure JAX-compatible wrapper for parameterized models.

    Handles models where graph structure depends on parameters.
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

    # Compute concrete weights via callback
    new_edges = list(serialized['edges'].tolist()) if len(serialized['edges']) > 0 else []
    for edge in param_edges:
        from_idx, to_idx = int(edge[0]), int(edge[1])
        coeffs = np.array(edge[2:])
        weight = float(callback(theta, coeffs))
        new_edges.append([from_idx, to_idx, weight])

    new_start_edges = list(serialized['start_edges'].tolist()) if len(serialized['start_edges']) > 0 else []
    for edge in start_param_edges:
        to_idx = int(edge[0])
        coeffs = np.array(edge[1:])
        weight = float(callback(theta, coeffs))
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


def _generate_cpp_from_trace(trace: EliminationTrace, observed_data: ArrayLike, granularity: int = 0) -> str:
    """
    Generate standalone C++ log-likelihood function from elimination trace.

    Creates self-contained C++ code that embeds the trace data structure and
    evaluates log-likelihood without Python dependencies. This enables fast
    SVGD evaluation with minimal overhead.

    Parameters
    ----------
    trace : EliminationTrace
        Elimination trace from record_elimination_trace()
    observed_data : ArrayLike
        Observed data points for likelihood computation
    granularity : int, default=100
        Discretization granularity for forward algorithm PDF computation

    Returns
    -------
    str
        C++ code implementing compute_log_likelihood(theta, n_params) function

    Notes
    -----
    Generated function signature:
        double compute_log_likelihood(const double* theta, int n_params)

    The function performs:
    1. Evaluates trace with parameters using embedded trace data
    2. Instantiates graph from evaluation results
    3. Computes exact PDF at all observation points
    4. Returns sum of log-probabilities
    5. Cleans up allocated memory

    Performance: O(n*m) where n = operations, m = observations
    Memory: O(n) for evaluation + O(v+e) for graph (v=vertices, e=edges)

    Examples
    --------
    >>> from phasic.trace_elimination import record_elimination_trace
    >>> trace = record_elimination_trace(graph, theta_dim=2)
    >>> observed_times = np.array([1.5, 2.3, 0.8])
    >>> cpp_code = _generate_cpp_from_trace(trace, observed_times, granularity=100)
    >>> # Compile cpp_code and use with JAX
    """
    from .trace_elimination import trace_to_c_arrays
    import numpy as np

    # Convert observed_data to numpy array
    observed_data = np.asarray(observed_data)
    n_observations = len(observed_data) if observed_data.ndim > 0 else 1

    # Serialize trace to C-compatible arrays
    arrays = trace_to_c_arrays(trace)

    # Helper function to format array as C initializer
    def format_array(arr, dtype='double'):
        if not arr:
            return f"NULL  /* empty {dtype} array */"
        if dtype == 'int' or dtype == 'size_t':
            return '{' + ', '.join(str(int(x)) for x in arr) + '}'
        else:
            return '{' + ', '.join(f'{float(x):.17e}' for x in arr) + '}'

    # Generate code
    cpp_code = f'''#include "phasiccpp.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>

// =============================================================================
// Embedded Trace Data
// =============================================================================

// Trace metadata
static const size_t N_OPERATIONS = {len(arrays['operations_types'])};
static const size_t N_VERTICES = {arrays['n_vertices']};
static const size_t STATE_LENGTH = {arrays['state_length']};
static const size_t PARAM_LENGTH = {arrays['param_length']};
static const size_t STARTING_VERTEX_IDX = {arrays['starting_vertex_idx']};
static const bool IS_DISCRETE = {'true' if arrays['is_discrete'] else 'false'};

// Operations data
static const int operations_types[] = {format_array(arrays['operations_types'], 'int')};
static const double operations_consts[] = {format_array(arrays['operations_consts'], 'double')};
static const int operations_param_indices[] = {format_array(arrays['operations_param_indices'], 'int')};
static const size_t operations_operand_counts[] = {format_array(arrays['operations_operand_counts'], 'size_t')};
static const size_t operations_operands_flat[] = {format_array(arrays['operations_operands_flat'], 'size_t')};
static const size_t operations_coeff_counts[] = {format_array(arrays['operations_coeff_counts'], 'size_t')};
static const double operations_coeffs_flat[] = {format_array(arrays['operations_coeffs_flat'], 'double')};

// Vertex data
static const size_t vertex_rates[] = {format_array(arrays['vertex_rates'], 'size_t')};
static const size_t edge_probs_counts[] = {format_array(arrays['edge_probs_counts'], 'size_t')};
static const size_t edge_probs_flat[] = {format_array(arrays['edge_probs_flat'], 'size_t')};
static const size_t vertex_targets_counts[] = {format_array(arrays['vertex_targets_counts'], 'size_t')};
static const size_t vertex_targets_flat[] = {format_array(arrays['vertex_targets_flat'], 'size_t')};
static const int states_flat[] = {format_array(arrays['states_flat'], 'int')};

// Observation data
static const size_t N_OBSERVATIONS = {n_observations};
static const double observed_times[] = {format_array(observed_data.tolist(), 'double')};
static const size_t GRANULARITY = {granularity};

// =============================================================================
// Trace Evaluation Helper
// =============================================================================

/**
 * Evaluate embedded trace with given parameters
 * Returns allocated ptd_trace_result (caller must free)
 */
static struct ptd_trace_result* evaluate_embedded_trace(const double* theta, size_t n_params) {{
    // Allocate values array
    double* values = (double*)malloc(N_OPERATIONS * sizeof(double));
    if (values == NULL) {{
        return NULL;
    }}

    // Execute operations in order
    size_t operands_offset = 0;
    size_t coeffs_offset = 0;

    for (size_t i = 0; i < N_OPERATIONS; i++) {{
        int op_type = operations_types[i];

        if (op_type == 0) {{  // CONST
            values[i] = operations_consts[i];
        }}
        else if (op_type == 1) {{  // PARAM
            int param_idx = operations_param_indices[i];
            values[i] = theta[param_idx];
        }}
        else if (op_type == 2) {{  // DOT
            size_t n_coeffs = operations_coeff_counts[i];
            double result = 0.0;
            for (size_t j = 0; j < n_coeffs; j++) {{
                result += operations_coeffs_flat[coeffs_offset + j] * theta[j];
            }}
            values[i] = result;
            coeffs_offset += n_coeffs;
        }}
        else if (op_type == 3) {{  // ADD
            values[i] = values[operations_operands_flat[operands_offset]] +
                       values[operations_operands_flat[operands_offset + 1]];
            operands_offset += operations_operand_counts[i];
        }}
        else if (op_type == 4) {{  // MUL
            values[i] = values[operations_operands_flat[operands_offset]] *
                       values[operations_operands_flat[operands_offset + 1]];
            operands_offset += operations_operand_counts[i];
        }}
        else if (op_type == 5) {{  // DIV
            values[i] = values[operations_operands_flat[operands_offset]] /
                       values[operations_operands_flat[operands_offset + 1]];
            operands_offset += operations_operand_counts[i];
        }}
        else if (op_type == 6) {{  // INV
            values[i] = 1.0 / values[operations_operands_flat[operands_offset]];
            operands_offset += operations_operand_counts[i];
        }}
        else if (op_type == 7) {{  // SUM
            double sum = 0.0;
            for (size_t j = 0; j < operations_operand_counts[i]; j++) {{
                sum += values[operations_operands_flat[operands_offset + j]];
            }}
            values[i] = sum;
            operands_offset += operations_operand_counts[i];
        }}
        else if (op_type == 8) {{  // SUB
            values[i] = values[operations_operands_flat[operands_offset]] -
                       values[operations_operands_flat[operands_offset + 1]];
            operands_offset += operations_operand_counts[i];
        }}
    }}

    // Build result structure
    struct ptd_trace_result* result = (struct ptd_trace_result*)malloc(sizeof(struct ptd_trace_result));
    if (result == NULL) {{
        free(values);
        return NULL;
    }}

    result->n_vertices = N_VERTICES;
    result->vertex_rates = (double*)malloc(N_VERTICES * sizeof(double));
    result->edge_probs = (double**)malloc(N_VERTICES * sizeof(double*));
    result->edge_probs_lengths = (size_t*)malloc(N_VERTICES * sizeof(size_t));
    result->vertex_targets = (size_t**)malloc(N_VERTICES * sizeof(size_t*));
    result->vertex_targets_lengths = (size_t*)malloc(N_VERTICES * sizeof(size_t));

    if (result->vertex_rates == NULL || result->edge_probs == NULL ||
        result->edge_probs_lengths == NULL || result->vertex_targets == NULL ||
        result->vertex_targets_lengths == NULL) {{
        free(values);
        free(result);
        return NULL;
    }}

    // Extract vertex rates
    for (size_t i = 0; i < N_VERTICES; i++) {{
        result->vertex_rates[i] = values[vertex_rates[i]];
    }}

    // Extract edge probabilities and targets
    size_t edge_offset = 0;
    size_t target_offset = 0;

    for (size_t i = 0; i < N_VERTICES; i++) {{
        size_t n_edges = edge_probs_counts[i];
        result->edge_probs_lengths[i] = n_edges;
        result->vertex_targets_lengths[i] = n_edges;

        if (n_edges > 0) {{
            result->edge_probs[i] = (double*)malloc(n_edges * sizeof(double));
            result->vertex_targets[i] = (size_t*)malloc(n_edges * sizeof(size_t));

            if (result->edge_probs[i] == NULL || result->vertex_targets[i] == NULL) {{
                free(values);
                // TODO: proper cleanup
                return NULL;
            }}

            for (size_t j = 0; j < n_edges; j++) {{
                result->edge_probs[i][j] = values[edge_probs_flat[edge_offset + j]];
                result->vertex_targets[i][j] = vertex_targets_flat[target_offset + j];
            }}

            edge_offset += n_edges;
            target_offset += n_edges;
        }} else {{
            result->edge_probs[i] = NULL;
            result->vertex_targets[i] = NULL;
        }}
    }}

    free(values);
    return result;
}}

// =============================================================================
// Main Log-Likelihood Function
// =============================================================================

/**
 * Compute log-likelihood for given parameters
 *
 * @param theta Parameter array
 * @param n_params Number of parameters (must equal PARAM_LENGTH)
 * @return Log-likelihood value, or -INFINITY on error
 */
extern "C" double compute_log_likelihood(const double* theta, int n_params) {{
    if (theta == NULL || n_params != PARAM_LENGTH) {{
        return -INFINITY;
    }}

    // 1. Evaluate trace with parameters
    struct ptd_trace_result* result = evaluate_embedded_trace(theta, n_params);
    if (result == NULL) {{
        return -INFINITY;
    }}

    // 2. Build elimination trace structure for ptd_instantiate_from_trace
    //    (We need to construct minimal trace structure with states)
    struct ptd_elimination_trace trace_struct;
    trace_struct.n_vertices = N_VERTICES;
    trace_struct.state_length = STATE_LENGTH;
    trace_struct.starting_vertex_idx = STARTING_VERTEX_IDX;

    // Allocate and populate states
    int** states = (int**)malloc(N_VERTICES * sizeof(int*));
    if (states == NULL) {{
        ptd_trace_result_destroy(result);
        return -INFINITY;
    }}

    for (size_t i = 0; i < N_VERTICES; i++) {{
        states[i] = (int*)malloc(STATE_LENGTH * sizeof(int));
        if (states[i] == NULL) {{
            for (size_t j = 0; j < i; j++) free(states[j]);
            free(states);
            ptd_trace_result_destroy(result);
            return -INFINITY;
        }}
        for (size_t j = 0; j < STATE_LENGTH; j++) {{
            states[i][j] = states_flat[i * STATE_LENGTH + j];
        }}
    }}
    trace_struct.states = states;

    // 3. Instantiate graph from trace
    struct ptd_graph* graph = ptd_instantiate_from_trace(result, &trace_struct);

    // Clean up states
    for (size_t i = 0; i < N_VERTICES; i++) {{
        free(states[i]);
    }}
    free(states);

    if (graph == NULL) {{
        ptd_trace_result_destroy(result);
        return -INFINITY;
    }}

    // 4. Compute log-likelihood by evaluating PDF at all observation points
    double log_lik = 0.0;
    double pdf_value = 0.0;
    double* pdf_gradient = NULL;  // We don't need gradients here

    for (size_t i = 0; i < N_OBSERVATIONS; i++) {{
        int status = ptd_graph_pdf_parameterized(graph, observed_times[i], GRANULARITY,
                                                  &pdf_value, pdf_gradient);
        if (status != 0) {{
            ptd_graph_destroy(graph);
            ptd_trace_result_destroy(result);
            return -INFINITY;
        }}

        // Add log(PDF) with numerical safety
        if (pdf_value <= 0.0) {{
            log_lik += -23.025850929940458;  // log(1e-10)
        }} else {{
            log_lik += log(pdf_value);
        }}
    }}

    // 5. Cleanup
    ptd_graph_destroy(graph);
    ptd_trace_result_destroy(result);

    return log_lik;
}}
'''

    return cpp_code


def _compile_trace_library(cpp_code: str, trace_hash: str) -> str:
    """
    Compile trace-based C++ code to shared library.

    Parameters
    ----------
    cpp_code : str
        C++ source code from _generate_cpp_from_trace()
    trace_hash : str
        Hash identifier for this trace (for caching)

    Returns
    -------
    str
        Path to compiled shared library

    Raises
    ------
    RuntimeError
        If compilation fails
    """
    import subprocess
    import tempfile
    import os

    lib_path = f"/tmp/trace_log_lik_{trace_hash}.so"

    # Skip compilation if library already exists
    if os.path.exists(lib_path):
        return lib_path

    # Write source to temporary file
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False, mode='w') as f:
        f.write(cpp_code)
        cpp_file = f.name

    try:
        # Get package directory for includes
        pkg_dir = _get_package_dir()

        # Compile command
        cmd = [
            'g++', '-O3', '-fPIC', '-shared', '-std=c++14',
            f'-I{pkg_dir}',
            f'-I{pkg_dir}/api/cpp',
            f'-I{pkg_dir}/api/c',
            f'-I{pkg_dir}/include',
            cpp_file,
            f'{pkg_dir}/src/c/phasic.c',
            '-o', lib_path,
            '-lm'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

        return lib_path

    finally:
        # Clean up temporary C++ file
        if os.path.exists(cpp_file):
            os.unlink(cpp_file)


def clear_trace_cache() -> int:
    """
    Clear cached compiled trace libraries.

    Removes all compiled shared libraries from /tmp/ that were generated
    by trace_to_log_likelihood() with use_cpp=True.

    Returns
    -------
    int
        Number of cache files removed

    Examples
    --------
    >>> from phasic import clear_trace_cache
    >>> n_removed = clear_trace_cache()
    >>> print(f"Removed {n_removed} cached trace libraries")
    """
    import glob
    pattern = "/tmp/trace_log_lik_*.so"
    cache_files = glob.glob(pattern)

    count = 0
    for f in cache_files:
        try:
            os.unlink(f)
            count += 1
        except OSError:
            pass  # Ignore errors (file might be in use or already deleted)

    return count


def _wrap_trace_log_likelihood_for_jax(lib_path: str, param_length: int) -> Callable:
    """
    Wrap C++ log-likelihood function for JAX compatibility.

    Creates a JAX-compatible function using jax.pure_callback that calls
    the compiled C++ log-likelihood function.

    Parameters
    ----------
    lib_path : str
        Path to compiled shared library
    param_length : int
        Number of parameters expected by the function

    Returns
    -------
    callable
        JAX-compatible log-likelihood function with signature:
        log_lik(theta: jax.numpy.ndarray) -> float

    Notes
    -----
    The returned function supports:
    - jax.jit: JIT compilation
    - jax.grad: Automatic differentiation (via finite differences)
    - jax.vmap: Vectorization over parameter batches

    The function uses pure_callback to call C++ code, which means:
    - Gradients computed via JAX's finite difference approximation
    - No direct gradient computation in C++ (yet - Phase 5 feature)
    - Each vmap call executes sequentially (no parallelization)

    Examples
    --------
    >>> lib_path = "/tmp/trace_log_lik_abc123.so"
    >>> log_lik = _wrap_trace_log_likelihood_for_jax(lib_path, theta_dim=2)
    >>> import jax.numpy as jnp
    >>> theta = jnp.array([1.0, 2.0])
    >>> ll_value = log_lik(theta)
    >>> grad = jax.grad(log_lik)(theta)
    """
    import ctypes
    import numpy as np

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required. Install with: pip install jax jaxlib")

    # Load shared library
    lib = ctypes.CDLL(lib_path)

    # Define function signature
    lib.compute_log_likelihood.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.compute_log_likelihood.restype = ctypes.c_double

    def log_lik_cpp(theta):
        """Pure Python wrapper for C++ function"""
        theta_array = np.asarray(theta, dtype=np.float64)
        if len(theta_array) != param_length:
            raise ValueError(f"Expected {param_length} parameters, got {len(theta_array)}")

        theta_ptr = theta_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result = lib.compute_log_likelihood(theta_ptr, param_length)
        return float(result)

    def log_lik_jax(theta):
        """JAX-compatible wrapper using pure_callback"""
        # Ensure theta is the right shape
        theta = jnp.atleast_1d(theta)

        return jax.pure_callback(
            log_lik_cpp,
            jax.ShapeDtypeStruct((), jnp.float64),  # Returns scalar
            theta,
            vectorized=False  # vmap will handle batching via sequential calls
        )

    return log_lik_jax


def _compile_wrapper_library(wrapper_code: str, lib_name: str, extra_includes: list[str] | None = None) -> str:
    """
    Compile C++ wrapper code to shared library.

    Handles all I/O and subprocess calls during setup phase.
    """
    pkg_dir = _get_package_dir()
    lib_path = f"/tmp/{lib_name}.so"

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
        sources = [
            f'{pkg_dir}/src/cpp/phasiccpp.cpp',
            f'{pkg_dir}/src/c/phasic.c',
            f'{pkg_dir}/src/c/phasic_hash.c',
            f'{pkg_dir}/src/c/phasic_log.c',
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

        # Base compilation command
        cmd = [
            'g++', '-O3', '-fPIC', '-shared', '-std=c++14',
            f'-I{pkg_dir}',
            f'-I{pkg_dir}/api/cpp',
            f'-I{pkg_dir}/api/c',
            f'-I{pkg_dir}/include',
        ]

        # Add extra includes if provided
        if extra_includes:
            for inc in extra_includes:
                cmd.append(f'-I{inc}')

        # Add source files
        cmd.extend([wrapper_file, *sources, '-o', lib_path])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    finally:
        os.unlink(wrapper_file)

    return lib_path


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


class SymbolicDAG:
    """
    Symbolic representation of an acyclic phase-type distribution graph.

    This class represents a graph where edges contain symbolic expression trees
    instead of concrete numeric values. This enables O(n) parameter evaluation
    instead of O(n³) graph reconstruction.

    Primary use case: SVGD and other inference algorithms that require
    evaluating the same graph structure with many different parameter vectors.

    Performance:
    - Graph elimination (once): O(n³)
    - Parameter instantiation (per particle): O(n)
    - Expected speedup for SVGD: 100-1000×

    Examples
    --------
    >>> # Create parameterized graph
    >>> g = Graph(1)
    >>> v_a = g.create_vertex([0])
    >>> v_b = g.create_vertex([1])
    >>> v_c = g.create_vertex([2])
    >>> v_a.add_edge_parameterized(v_b, 0.0, [1.0, 0.0, 0.0])  # weight = p[0]
    >>> v_b.add_edge_parameterized(v_c, 0.0, [0.0, 1.0, 0.0])  # weight = p[1]

    >>> # Perform symbolic elimination (once, O(n³))
    >>> dag = g.eliminate_to_dag()

    >>> # Instantiate with different parameters (O(n) each)
    >>> g1 = dag.instantiate([1.0, 2.0, 0.0])
    >>> g2 = dag.instantiate([3.0, 4.0, 0.0])

    >>> # Use for SVGD (100× faster than rebuilding graph for each particle)
    >>> particles = [dag.instantiate(p) for p in param_vectors]
    """

    def __init__(self, ptr: int) -> None:
        """Initialize from opaque pointer returned by Graph._eliminate_to_dag_internal()"""
        self._ptr = ptr
        self._info = None

    def instantiate(self, params: ArrayLike) -> Graph:
        """
        Evaluate expression trees with concrete parameters to create a Graph.

        This is an O(n) operation that evaluates all symbolic expressions
        with the given parameter vector. Much faster than O(n³) graph
        reconstruction!

        Parameters
        ----------
        params : numpy.typing.ArrayLike
            Parameter vector, shape ``(n_params,)``.

        Returns
        -------
        Graph
            Graph with concrete edge weights evaluated from expressions
        """
        from .phasic_pybind import _symbolic_dag_instantiate
        params_arr = np.asarray(params, dtype=np.float64)
        return _symbolic_dag_instantiate(self._ptr, params_arr)

    @property
    def info(self) -> dict[str, Any]:
        """Get metadata about the symbolic DAG"""
        if self._info is None:
            from .phasic_pybind import _symbolic_dag_get_info
            self._info = _symbolic_dag_get_info(self._ptr)
        return self._info

    @property
    def vertices_length(self) -> int:
        """Number of vertices in the DAG"""
        return self.info['vertices_length']

    @property
    def param_length(self) -> int:
        """Number of parameters required for instantiation"""
        return self.info['param_length']

    @property
    def is_acyclic(self) -> bool:
        """Whether the graph is acyclic (should always be True after elimination)"""
        return self.info['is_acyclic']

    def __del__(self) -> None:
        """Free C memory when Python object is garbage collected"""
        if hasattr(self, '_ptr') and self._ptr != 0:
            from .phasic_pybind import _symbolic_dag_destroy
            _symbolic_dag_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        return (f"SymbolicDAG(vertices={self.vertices_length}, "
                f"params={self.param_length}, acyclic={self.is_acyclic})")


class Graph(_Graph):
    # def __init__(self, state_length:int=None, callback:Callable=None, ipv:list[list[int] | list[list[int] | float]] | None = None, parameterized:bool=False, **kwargs):
    def __init__(self, arg: int | Callable, ipv: list[int] | list[list[int] | float] | None = None, graph_cache: bool = False, **kwargs: Any) -> None:
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
        graph_cache : bool, optional
            If True, attempts to load graph from disk cache. If not cached, builds graph and saves to cache.
            Cache is keyed by callback function source code + parameters, enabling instant loading of
            previously built graphs. Useful for expensive graph constructions.
            Default: False (no caching)
        theta_dim : int, optional
            Number of model parameters (θ). This sets the expected length of parameter vectors
            passed to update_weights(theta).

            **Can be set at two stages:**
            1. At graph construction: `Graph(callback, theta_dim=2)`
            2. At inference time: `graph.svgd(..., theta_dim=3)` - can override if graph was modified

            The value set here establishes the initial parameter dimension. It can be overridden
            later in methods like svgd() if the graph structure has been augmented or changed.

            When theta_dim < edge coefficients_length:
            - **Non-callback mode** (update_weights(theta)): ERROR - coefficient and theta lengths must match exactly
            - **Callback mode** (update_weights(theta, callback)): OK - callback receives full coefficient vector

            This allows storing auxiliary data in coefficient vectors for use in custom callback functions
            while maintaining a compact theta parameter space. The extra coefficients are accessible only
            through the callback, not in standard dot-product weight computation.

            If not provided, theta_dim is inferred from the first edge's coefficient length.

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
                    # Copy Python attributes
                    self._callback = cached_graph._callback if hasattr(cached_graph, '_callback') else None
                    self._callback_kwargs = cached_graph._callback_kwargs if hasattr(cached_graph, '_callback_kwargs') else {}
                    self.is_discrete = cached_graph.is_discrete if hasattr(cached_graph, 'is_discrete') else False
                    self._cache_trace = cache_trace
                    self._trace = None
                    self._trace_dirty = True
                    self._last_theta = None
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
        self._weight_mode = 'linear'  # Weight computation mode: 'linear', 'log', or 'callback'
        self._weight_callback = None  # Custom weight callback for 'callback' mode

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

        One of ``'linear'`` (default), ``'log'``, or ``'callback'``.

        - ``'linear'``: weight = Σ c_k θ_k
        - ``'log'``: weight = Π(c_k θ_k) (computed in log-space for stability)
        - ``'callback'``: weight = callback(theta, coefficients)
        """
        return self._weight_mode

    @weight_mode.setter
    def weight_mode(self, mode: str) -> None:
        if mode not in ('linear', 'log', 'callback'):
            raise ValueError(
                f"weight_mode must be 'linear', 'log', or 'callback', got {mode!r}"
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

    def update_weights(self, theta: ArrayLike, callback: Callable | None = None, log: bool = False) -> None:
        """Update parameterized edge weights with given parameters.

        This method wraps the C++ implementation to cache theta for use
        with trace-based computation.

        Parameters
        ----------
        theta : ArrayLike
            Parameter vector to set edge weights.
        callback : callable, optional
            Custom callback for weight computation.
        log : bool, default=False
            If True, use log-space computation.

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

        self._last_theta = np.asarray(theta)
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

    def _moments_from_trace(self, power: int = 1, rewards: ArrayLike | None = None, discrete: bool = False) -> float:
        """Compute moments using cached elimination trace.

        This instantiates a concrete graph from the trace with current
        parameters and computes moments on that graph.

        Parameters
        ----------
        power : int, default=1
            Moment power (1 for expectation, 2 for variance, etc.)
        rewards : ArrayLike, optional
            Reward vector for reward-transformed moments.
        discrete : bool, default=False
            If True, compute discrete-time moments (DPH distribution).
            Requires that the graph was discretized via discretize().

        Returns
        -------
        array
            Moments computed from trace-instantiated graph.
        """
        from .trace_elimination import instantiate_from_trace

        trace = self._ensure_trace()
        if trace is None:
            raise RuntimeError("_moments_from_trace called but no trace available")

        # Get current parameters
        theta = self._get_current_theta()

        # Instantiate concrete graph from trace with current parameters
        # Note: Do NOT call normalize() - the graph already has correct rates
        concrete_graph = instantiate_from_trace(trace, theta)

        # Compute moments on the concrete graph using C++ implementation
        if rewards is not None:
            return concrete_graph.moments(power, list(rewards), discrete=discrete)
        else:
            return concrete_graph.moments(power, discrete=discrete)

    def _expectation_from_trace(self, rewards: ArrayLike | None = None, discrete: bool = False, **kwargs: Any) -> float:
        """Compute expectation using cached elimination trace."""
        from .trace_elimination import instantiate_from_trace

        trace = self._ensure_trace()
        if trace is None:
            raise RuntimeError("_expectation_from_trace called but no trace available")

        theta = self._get_current_theta()
        # Note: Do NOT call normalize() - the graph already has correct rates
        concrete_graph = instantiate_from_trace(trace, theta)

        # Choose appropriate method based on discrete flag
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            if rewards is not None:
                return concrete_graph.expectation_discrete(rewards=list(rewards), **kwargs)
            else:
                return concrete_graph.expectation_discrete(**kwargs)
        else:
            if rewards is not None:
                return concrete_graph.expectation(rewards=list(rewards), **kwargs)
            else:
                return concrete_graph.expectation(**kwargs)

    def _variance_from_trace(self, rewards: ArrayLike | None = None, discrete: bool = False, **kwargs: Any) -> float:
        """Compute variance using cached elimination trace."""
        from .trace_elimination import instantiate_from_trace

        trace = self._ensure_trace()
        if trace is None:
            raise RuntimeError("_variance_from_trace called but no trace available")

        theta = self._get_current_theta()
        # Note: Do NOT call normalize() - the graph already has correct rates
        concrete_graph = instantiate_from_trace(trace, theta)

        # Choose appropriate method based on discrete flag
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            if rewards is not None:
                return concrete_graph.variance_discrete(rewards=list(rewards), **kwargs)
            else:
                return concrete_graph.variance_discrete(**kwargs)
        else:
            if rewards is not None:
                return concrete_graph.variance(rewards=list(rewards), **kwargs)
            else:
                return concrete_graph.variance(**kwargs)

    def _covariance_from_trace(self, rewards1: ArrayLike | None = None, rewards2: ArrayLike | None = None, discrete: bool = False, **kwargs: Any) -> float:
        """Compute covariance using cached elimination trace."""
        from .trace_elimination import instantiate_from_trace

        trace = self._ensure_trace()
        if trace is None:
            raise RuntimeError("_covariance_from_trace called but no trace available")

        theta = self._get_current_theta()
        # Note: Do NOT call normalize() - the graph already has correct rates
        concrete_graph = instantiate_from_trace(trace, theta)

        # Choose appropriate method based on discrete flag
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            if rewards1 is not None:
                rewards1 = list(rewards1)
            if rewards2 is not None:
                rewards2 = list(rewards2)
            return concrete_graph.covariance_discrete(rewards1=rewards1, rewards2=rewards2, **kwargs)
        else:
            if rewards1 is not None:
                rewards1 = list(rewards1)
            if rewards2 is not None:
                rewards2 = list(rewards2)
            return concrete_graph.covariance(rewards1=rewards1, rewards2=rewards2, **kwargs)

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
            return super().moments_discrete(power, rewards=rewards, **kwargs)
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
        if self.cache_trace:
            if self.is_discrete:
                raise NotImplementedError("Trace-based expectation computation not implemented yet for discrete graphs.")
                # trace = self._ensure_trace()
                # if trace is None:
                #     raise RuntimeError("No trace, is your Graph parameterized?")
                # return self._covariance_from_trace(rewards1=rewards1, rewards2=rewards2, discrete=True, **kwargs)
            else:
                trace = self._ensure_trace()
                if trace is None:
                    raise RuntimeError("No trace, is your Graph parameterized?")
                return self._covariance_from_trace(rewards1=rewards1, rewards2=rewards2, discrete=False, **kwargs)                     

        if self.is_discrete:
            return super().covariance_discrete(rewards1=rewards1, rewards2=rewards2, **kwargs)
        else:
            return super().covariance(rewards1=rewards1, rewards2=rewards2, **kwargs)

    def pdf(self, time: float | ArrayLike, **kwargs: Any) -> float | np.ndarray:
        """
        Compute probability density/mass function using forward algorithm.

        Parameters
        ----------
        time : float or ArrayLike
            Time point(s) at which to evaluate the PDF/PMF.
        granularity : int, optional
            Granularity for uniformization (default: auto-detected as 2*max_rate).
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
        granularity = kwargs.get('granularity', 0)
        if not isinstance(granularity, (int, np.integer)):
            raise TypeError(f"granularity must be an integer, got {type(granularity).__name__}")
        if granularity < 0:
            raise ValueError(f"granularity must be >= 0, got {granularity}")

        if self.is_discrete:
            return super().pdf_discrete(time, **kwargs)
        else:
            return super().pdf(time, **kwargs)

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

    def sample(self, n: int, **kwargs: Any) -> np.ndarray:
        """
        Generate random samples from the phase-type distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

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
            if rate <= 0 or rate >= 1:
                raise ValueError(f"rate must be in (0, 1), got {rate}")

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
            if rate <= 0 or rate >= 1:
                raise ValueError(f"rate must be in (0, 1), got {rate}")

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

    def reward_transform(self, rewards:np.ndarray) -> Self:
        """
        Apply reward transformation to create a new graph with modified rewards.

        Parameters
        ----------
        rewards : np.ndarray
            Reward vector of length n_vertices. Each element specifies the
            reward associated with visiting the corresponding vertex.

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
        if rewards_arr.ndim == 1:
            if len(rewards_arr) != self.vertices_length():
                raise ValueError(
                    f"rewards length ({len(rewards_arr)}) must equal number of vertices ({self.vertices_length()})"
                )

        if self.is_discrete:
            return Graph(super().reward_transform_discrete(rewards))
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
        return Graph(super().reward_transform_discrete(rewards))

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
        result._weight_mode = self._weight_mode
        result._weight_callback = self._weight_callback
        result._last_callback_vertices_length = result.vertices_length()
        # COMPOSABLE_MIGRATION: original implementation
        # return Graph(super().laplace_transform(theta))
        return result

    def absorbing_state_rewards(self) -> np.ndarray:
        """
        Get a reward vector that is 1 for states with edges to absorbing states.

        This reward vector is used for computing the Laplace transform value
        from a Laplace-transformed graph.

        Returns
        -------
        np.ndarray
            Reward vector of length n_vertices. Element i is 1.0 if vertex i
            has an edge to an absorbing state, 0.0 otherwise.

        Examples
        --------
        >>> graph = Graph(coalescent_callback)
        >>> rewards = graph.absorbing_state_rewards()
        >>> laplace_graph = graph.laplace_transform(0.5)
        >>> L_0_5 = laplace_graph.expectation(rewards=rewards)

        See Also
        --------
        laplace_transform : Create a Laplace-transformed graph
        """
        n_vertices = self.vertices_length()
        rewards = np.zeros(n_vertices)
        for i in range(n_vertices):
            vertex = self.vertex_at(i)
            # Check if any edge goes to an absorbing state (no outgoing edges)
            for edge in vertex.edges():
                to_vertex = edge.to()
                if to_vertex.edges_length() == 0:
                    rewards[i] = 1.0
                    break
        return rewards

    def serialize(self, theta_dim: int | None = None) -> dict[str, np.ndarray]:
        """
        Serialize graph to array representation for efficient computation.

        Parameters
        ----------
        theta_dim : int, optional
            Number of parameters for parameterized edges. If not provided, will be
            auto-detected by probing edge states. Providing this explicitly avoids
            potential issues with auto-detection reading garbage memory.

        Returns
        -------
        dict
            Dictionary containing:
            - 'states': Array of vertex states (n_vertices, state_dim)
            - 'edges': Array of regular edges [from_idx, to_idx, weight] (n_edges, 3)
            - 'start_edges': Array of starting vertex regular edges [to_idx, weight] (n_start_edges, 2)
            - 'param_edges': Array of parameterized edges [from_idx, to_idx, x1, x2, ...] (n_param_edges, theta_dim+2)
            - 'start_param_edges': Array of starting vertex parameterized edges [to_idx, x1, x2, ...] (n_start_param_edges, theta_dim+1)
              NOTE: start_param_edges should be empty (starting edges are not parameterized)
            - 'param_length': Length of parameter vector (0 if no parameterized edges)
            - 'state_length': Integer state dimension
            - 'n_vertices': Number of vertices
        """
        vertices_list = list(self.vertices())
        n_vertices = len(vertices_list)

        if n_vertices == 0:
            raise ValueError("Graph has no vertices (except starting vertex)")

        # Extract states and create vertex index mapping
        state_length = self.state_length()
        states = np.zeros((n_vertices, state_length), dtype=np.int32)
        vertex_indices = np.zeros(n_vertices, dtype=np.int32)

        # Map vertex.index() -> enumeration position in vertices_list
        # This handles duplicate states correctly (multiple vertices with same state)
        vertex_idx_to_enum = {}

        for i, v in enumerate(vertices_list):
            state = v.state()
            states[i, :] = state
            vertex_indices[i] = v.index()
            vertex_idx_to_enum[v.index()] = i

        # Get parameter length directly from graph (set by first add_edge() call)
        start = self.starting_vertex()

        # Use provided theta_dim if given, otherwise get from graph
        if theta_dim is None:
            theta_dim = self.param_length()

        # theta_dim is now always correct (no probing needed)

        # Extract parameterized edges FIRST (needed to build exclusion set before extracting regular edges)
        start_state = tuple(start.state())

        # Extract parameterized edges between vertices (excluding starting vertex)
        # With unified interface: parameterized_edges() returns edges with coefficient arrays
        param_edges_list = []
        start_vertex_idx = start.index()
        if theta_dim > 0:  # Export all edges with coefficient arrays
            for i, v in enumerate(vertices_list):
                # Skip starting vertex edges (they're handled separately)
                # Use vertex index comparison, not state comparison (states may be duplicated)
                if v.index() == start_vertex_idx:
                    continue

                from_idx = i
                for edge in v.parameterized_edges():
                    to_vertex = edge.to()
                    if to_vertex.index() in vertex_idx_to_enum:
                        to_idx = vertex_idx_to_enum[to_vertex.index()]
                        # Get FULL coefficient array (all coefficients, not just theta_dim)
                        # This is critical when coefficients_length > theta_dim
                        coeff_len = edge.coefficients_length()
                        edge_state = list(edge.edge_state(coeff_len))
                        # Include all parameterized edges (even with all-zero coefficients)
                        if edge_state:
                            # Store: [from_idx, to_idx, x1, x2, x3, ...]
                            param_edges_list.append([from_idx, to_idx] + edge_state)

        # Convert to numpy array - all edges should have same coefficient length
        # If not, will raise ValueError with helpful message
        if param_edges_list:
            try:
                param_edges = np.array(param_edges_list, dtype=np.float64)
            except ValueError as e:
                # Check if edges have different coefficient lengths
                lengths = set(len(row) for row in param_edges_list)
                if len(lengths) > 1:
                    raise ValueError(
                        f"Graph serialization failed: parameterized edges have inconsistent coefficient lengths.\n"
                        f"  Found coefficient lengths: {sorted(lengths)}\n"
                        f"  All edges must have the same coefficient length.\n"
                        f"  Hint: Check callback function - it may be returning edges with different coefficient arrays."
                    ) from e
                else:
                    raise
        else:
            # Empty case - use theta_dim for consistency
            param_edges = np.empty((0, theta_dim + 2 if theta_dim > 0 else 0), dtype=np.float64)

        # Extract starting vertex parameterized edges FIRST (needed to build exclusion set)
        # NOTE: Starting vertex edges are NEVER rescaled by update_weights() (see starting vertex fix)
        # So we should NOT export them as parameterized edges - they are effectively constant
        start_param_edges_list = []
        if False:  # Starting edges are never parameterized (always constant)
            for edge in start.parameterized_edges():
                to_vertex = edge.to()
                if to_vertex.index() in vertex_idx_to_enum:
                    to_idx = vertex_idx_to_enum[to_vertex.index()]
                    # Get coefficient array (length is guaranteed to be theta_dim)
                    edge_state = list(edge.edge_state(theta_dim))
                    # Only include edges with non-empty edge states
                    if edge_state and any(x != 0 for x in edge_state):
                        # Store: [to_idx, x1, x2, x3, ...]
                        start_param_edges_list.append([to_idx] + edge_state)

        start_param_edges = np.array(start_param_edges_list, dtype=np.float64) if start_param_edges_list else np.empty((0, theta_dim + 1 if theta_dim > 0 else 0), dtype=np.float64)

        # Build set of (from_idx, to_idx) pairs for parameterized edges to skip in regular edges
        param_edge_pairs = set()
        for edge_data in start_param_edges_list:
            to_idx = int(edge_data[0])
            param_edge_pairs.add((-1, to_idx))  # -1 represents starting vertex
        for edge_data in param_edges_list:
            from_idx = int(edge_data[0])
            to_idx = int(edge_data[1])
            param_edge_pairs.add((from_idx, to_idx))

        # Extract regular edges between vertices (excluding starting vertex)
        # Skip edges that have parameterized versions
        edges_list = []
        start_vertex_idx = start.index()
        for i, v in enumerate(vertices_list):
            # Skip starting vertex edges (they're handled separately)
            # Use vertex index comparison, not state comparison (states may be duplicated)
            if v.index() == start_vertex_idx:
                continue

            from_idx = i
            for edge in v.edges():
                to_vertex = edge.to()
                if to_vertex.index() in vertex_idx_to_enum:
                    to_idx = vertex_idx_to_enum[to_vertex.index()]
                    # Skip if this edge also has a parameterized version
                    if (from_idx, to_idx) not in param_edge_pairs:
                        weight = edge.weight()
                        edges_list.append([from_idx, to_idx, weight])

        edges = np.array(edges_list, dtype=np.float64) if edges_list else np.empty((0, 3), dtype=np.float64)

        # Extract starting vertex regular edges (skip those with parameterized versions)
        start_edges_list = []
        for edge in start.edges():
            to_vertex = edge.to()
            if to_vertex.index() in vertex_idx_to_enum:
                to_idx = vertex_idx_to_enum[to_vertex.index()]
                # Skip if this edge also has a parameterized version
                if (-1, to_idx) not in param_edge_pairs:
                    weight = edge.weight()
                    start_edges_list.append([to_idx, weight])

        start_edges = np.array(start_edges_list, dtype=np.float64) if start_edges_list else np.empty((0, 2), dtype=np.float64)

        return {
            'states': states,
            'vertex_indices': vertex_indices,
            'edges': edges,
            'start_edges': start_edges,
            'param_edges': param_edges,
            'start_param_edges': start_param_edges,
            'param_length': theta_dim,
            'state_length': state_length,
            'n_vertices': n_vertices,
            'weight_mode': self._weight_mode,
        }

    @classmethod
    def from_serialized(cls, data: dict[str, Any]) -> Graph:
        """
        Reconstruct Graph from serialize() output.

        This method enables distributed trace recording by allowing graphs
        to be serialized to JSON, sent across the network via JAX pmap,
        and reconstructed on worker processes.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary returned by Graph.serialize() containing:
            - states: np.ndarray of shape (n_vertices, state_length)
            - edges: np.ndarray of shape (n_edges, 3) with [from_idx, to_idx, weight]
            - start_edges: np.ndarray of shape (n_start_edges, 2) with [to_idx, weight]
            - param_edges: np.ndarray of shape (n_param_edges, 2+param_length)
              Format (v0.22.0+): [from_idx, to_idx, coeff1, coeff2, ...]
            - start_param_edges: np.ndarray of shape (n_start_param_edges, 1+param_length)
              Format (v0.22.0+): [to_idx, coeff1, coeff2, ...]
            - param_length: int
            - state_length: int
            - n_vertices: int

        Returns
        -------
        Graph
            Reconstructed graph with identical structure to the original

        Raises
        ------
        ValueError
            If data is missing required fields, has wrong shapes, or malformed arrays
        RuntimeError
            If graph reconstruction fails (e.g., edges to non-existent vertices)

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> from phasic import Graph
        >>>
        >>> # Create and serialize graph
        >>> g = Graph(1)
        >>> v0 = g.starting_vertex()
        >>> v1 = g.find_or_create_vertex([1])
        >>> v0.add_edge_parameterized(v1, 0.0, [2.0])
        >>> serialized = g.serialize(theta_dim=1)
        >>>
        >>> # Convert to JSON (for network transmission)
        >>> json_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
        ...              for k, v in serialized.items()}
        >>> json_str = json.dumps(json_dict)
        >>>
        >>> # Reconstruct on worker (e.g., different machine)
        >>> received_dict = json.loads(json_str)
        >>> received_dict['states'] = np.array(received_dict['states'], dtype=np.int32)
        >>> received_dict['edges'] = np.array(received_dict['edges'], dtype=np.float64)
        >>> received_dict['start_edges'] = np.array(received_dict['start_edges'], dtype=np.float64)
        >>> received_dict['param_edges'] = np.array(received_dict['param_edges'], dtype=np.float64)
        >>> received_dict['start_param_edges'] = np.array(received_dict['start_param_edges'], dtype=np.float64)
        >>> g_reconstructed = Graph.from_serialized(received_dict)
        >>> assert g_reconstructed.vertices_length() == g.vertices_length()
        """
        import numpy as np

        # Validate required fields
        required = ['states', 'edges', 'start_edges', 'param_edges',
                    'start_param_edges', 'param_length', 'state_length', 'n_vertices']
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(
                f"Graph deserialization failed: missing required fields {missing}\n"
                f"  Available fields: {list(data.keys())}\n"
                f"  This usually indicates corrupted cache or version mismatch.\n"
                f"  Resolution: Clear cache with phasic.clear_caches() and rebuild graph"
            )

        # Extract and validate metadata
        try:
            n_vertices = int(data['n_vertices'])
            state_length = int(data['state_length'])
            param_length = int(data['param_length'])
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Graph deserialization failed: invalid metadata types\n"
                f"  n_vertices={data.get('n_vertices')!r}, "
                f"state_length={data.get('state_length')!r}, "
                f"param_length={data.get('param_length')!r}\n"
                f"  Error: {e}"
            ) from e

        # Validate and convert arrays
        try:
            states = np.asarray(data['states'], dtype=np.int32)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Graph deserialization failed: cannot convert 'states' to int32 array\n"
                f"  Type: {type(data['states'])}\n"
                f"  Error: {e}"
            ) from e

        if states.shape != (n_vertices, state_length):
            raise ValueError(
                f"Graph deserialization failed: states array shape mismatch\n"
                f"  Expected: ({n_vertices}, {state_length}) from metadata\n"
                f"  Actual: {states.shape} from 'states' field\n"
                f"  Resolution: This indicates corrupted data. Clear cache and rebuild."
            )

        try:
            vertex_indices = np.asarray(data['vertex_indices'], dtype=np.int32)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Graph deserialization failed: cannot convert 'vertex_indices' to int32 array\n"
                f"  Type: {type(data['vertex_indices'])}\n"
                f"  Error: {e}"
            ) from e

        if vertex_indices.shape != (n_vertices,):
            raise ValueError(
                f"Graph deserialization failed: vertex_indices array shape mismatch\n"
                f"  Expected: ({n_vertices},) from metadata\n"
                f"  Actual: {vertex_indices.shape} from 'vertex_indices' field\n"
                f"  Resolution: This indicates corrupted data. Clear cache and rebuild."
            )

        try:
            edges = np.asarray(data['edges'], dtype=np.float64)
            start_edges = np.asarray(data['start_edges'], dtype=np.float64)
            param_edges = np.asarray(data['param_edges'], dtype=np.float64)
            start_param_edges = np.asarray(data['start_param_edges'], dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Graph deserialization failed: cannot convert edge arrays to float64\n"
                f"  Error: {e}"
            ) from e

        # Validate edge array shapes
        # Handle empty arrays: they may be shape (0,) or (0, n_cols)
        if edges.ndim == 1:
            if edges.shape[0] != 0:
                raise ValueError(
                    f"Graph deserialization failed: edges array has wrong shape\n"
                    f"  Expected: (n_edges, 3) or empty (0,)\n"
                    f"  Actual: {edges.shape}"
                )
            edges = edges.reshape((0, 3))
        elif edges.ndim != 2 or edges.shape[1] != 3:
            raise ValueError(
                f"Graph deserialization failed: edges array has wrong shape\n"
                f"  Expected: (n_edges, 3)\n"
                f"  Actual: {edges.shape}"
            )

        if start_edges.ndim == 1:
            if start_edges.shape[0] != 0:
                raise ValueError(
                    f"Graph deserialization failed: start_edges array has wrong shape\n"
                    f"  Expected: (n_start_edges, 2) or empty (0,)\n"
                    f"  Actual: {start_edges.shape}"
                )
            start_edges = start_edges.reshape((0, 2))
        elif start_edges.ndim != 2 or start_edges.shape[1] != 2:
            raise ValueError(
                f"Graph deserialization failed: start_edges array has wrong shape\n"
                f"  Expected: (n_start_edges, 2)\n"
                f"  Actual: {start_edges.shape}"
            )

        # Parameterized edge format: [from_idx, to_idx, coeff1, coeff2, ...]
        # Format is now [from, to, c1, c2, ...] with 2+coefficients_length columns
        # The coefficient length may be >= param_length (edges can have extra coefficients)
        # Infer actual coefficient length from array shape
        if param_edges.ndim == 1:
            if param_edges.shape[0] != 0:
                raise ValueError(
                    f"Graph deserialization failed: param_edges array has wrong shape\n"
                    f"  Expected: (n_param_edges, 2+coeff_len) or empty (0,)\n"
                    f"  Actual: {param_edges.shape}"
                )
            param_edges = param_edges.reshape((0, 0))
        elif param_edges.ndim == 2:
            # Validate minimum columns (at least from_idx, to_idx)
            if param_edges.shape[0] > 0 and param_edges.shape[1] < 2:
                raise ValueError(
                    f"Graph deserialization failed: param_edges array has too few columns\n"
                    f"  Expected: at least 2 columns (from_idx, to_idx)\n"
                    f"  Actual: {param_edges.shape[1]} columns"
                )
            # Infer coefficient length from array shape
            if param_edges.shape[0] > 0:
                actual_coeff_len = param_edges.shape[1] - 2
                if actual_coeff_len < param_length:
                    raise ValueError(
                        f"Graph deserialization failed: coefficient length < param_length\n"
                        f"  Coefficient length: {actual_coeff_len} (from array shape)\n"
                        f"  param_length: {param_length}\n"
                        f"  Edges must have at least param_length coefficients"
                    )

        expected_start_param_edge_cols = 1 + param_length if param_length > 0 else 0
        if start_param_edges.ndim == 1:
            if start_param_edges.shape[0] != 0:
                raise ValueError(
                    f"Graph deserialization failed: start_param_edges array has wrong shape\n"
                    f"  Expected: (n_start_param_edges, {expected_start_param_edge_cols}) or empty (0,)\n"
                    f"  Actual: {start_param_edges.shape}"
                )
            start_param_edges = start_param_edges.reshape((0, expected_start_param_edge_cols)) if expected_start_param_edge_cols > 0 else start_param_edges.reshape((0, 0))
        elif start_param_edges.ndim == 2:
            # Accept (0, 0) for param_length=0, or (n, 1+param_length) for param_length>0
            if param_length == 0:
                if start_param_edges.shape != (0, 0):
                    raise ValueError(
                        f"Graph deserialization failed: start_param_edges array has wrong shape\n"
                        f"  Expected: (0, 0) when param_length=0\n"
                        f"  Actual: {start_param_edges.shape}"
                    )
            elif start_param_edges.shape[1] != expected_start_param_edge_cols:
                raise ValueError(
                    f"Graph deserialization failed: start_param_edges array has wrong shape\n"
                    f"  Expected: (n_start_param_edges, {expected_start_param_edge_cols})\n"
                    f"  Actual: {start_param_edges.shape}\n"
                    f"  Note: theta_dim={theta_dim}, so columns should be 1+{theta_dim}={expected_start_param_edge_cols}\n"
                    f"  Format: [to_idx, coeff1, coeff2, ...]"
                )

        # Create empty graph
        graph = cls(state_length)

        # DO NOT set param_length here - let the first parameterized edge set it
        # Setting it early causes constant edges to require param_length coefficients
        # The graph will auto-detect mode from the first non-IPV edge added

        start = graph.starting_vertex()
        start_vertex_c_idx = start.index()  # Get C index of starting vertex

        # Create all vertices first
        # Note: The starting vertex may be included in the states array,
        # so we need to check for it and reuse it instead of creating a duplicate
        idx_to_vertex = {}
        for idx in range(n_vertices):
            state = states[idx].tolist()

            # Check if this is the starting vertex using C vertex index
            # Use vertex_indices array, not state comparison (states may be duplicated)
            if vertex_indices[idx] == start_vertex_c_idx:
                idx_to_vertex[idx] = start
            else:
                try:
                    vertex = graph.find_or_create_vertex(state)
                    idx_to_vertex[idx] = vertex
                except Exception as e:
                    raise RuntimeError(
                        f"Graph deserialization failed: cannot create vertex {idx}\n"
                        f"  State: {state}\n"
                        f"  Error: {e}"
                    ) from e

        # Add parameterized edges
        # Format (v0.22.0+): [from_idx, to_idx, coeff1, coeff2, ...]
        for edge_data in param_edges:
            from_idx = int(edge_data[0])
            to_idx = int(edge_data[1])
            edge_state = edge_data[2:].tolist()

            if from_idx < 0 or from_idx >= n_vertices:
                raise RuntimeError(
                    f"Graph deserialization failed: parameterized edge has invalid from_idx\n"
                    f"  from_idx={from_idx}, valid range=[0, {n_vertices})"
                )
            if to_idx < 0 or to_idx >= n_vertices:
                raise RuntimeError(
                    f"Graph deserialization failed: parameterized edge has invalid to_idx\n"
                    f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
                )

            try:
                from_vertex = idx_to_vertex[from_idx]
                to_vertex = idx_to_vertex[to_idx]
                from_vertex.add_edge(to_vertex, edge_state)
            except Exception as e:
                raise RuntimeError(
                    f"Graph deserialization failed: cannot add parameterized edge\n"
                    f"  From vertex {from_idx} (state={states[from_idx].tolist()})\n"
                    f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                    f"  Edge state (coefficients): {edge_state}\n"
                    f"  Error: {e}"
                ) from e

        # Add starting vertex parameterized edges
        # Format (v0.22.0+): [to_idx, coeff1, coeff2, ...]
        for edge_data in start_param_edges:
            to_idx = int(edge_data[0])
            edge_state = edge_data[1:].tolist()

            if to_idx < 0 or to_idx >= n_vertices:
                raise RuntimeError(
                    f"Graph deserialization failed: start parameterized edge has invalid to_idx\n"
                    f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
                )

            try:
                to_vertex = idx_to_vertex[to_idx]
                start.add_edge(to_vertex, edge_state)
            except Exception as e:
                raise RuntimeError(
                    f"Graph deserialization failed: cannot add start parameterized edge\n"
                    f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                    f"  Edge state (coefficients): {edge_state}\n"
                    f"  Error: {e}"
                ) from e

        # Add regular edges (non-parameterized)
        for edge_data in edges:
            from_idx = int(edge_data[0])
            to_idx = int(edge_data[1])
            weight = float(edge_data[2])

            if from_idx < 0 or from_idx >= n_vertices:
                raise RuntimeError(
                    f"Graph deserialization failed: edge has invalid from_idx\n"
                    f"  from_idx={from_idx}, valid range=[0, {n_vertices})"
                )
            if to_idx < 0 or to_idx >= n_vertices:
                raise RuntimeError(
                    f"Graph deserialization failed: edge has invalid to_idx\n"
                    f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
                )

            try:
                from_vertex = idx_to_vertex[from_idx]
                to_vertex = idx_to_vertex[to_idx]
                from_vertex.add_edge(to_vertex, weight)
            except Exception as e:
                raise RuntimeError(
                    f"Graph deserialization failed: cannot add edge\n"
                    f"  From vertex {from_idx} (state={states[from_idx].tolist()})\n"
                    f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                    f"  Weight: {weight}\n"
                    f"  Error: {e}"
                ) from e

        # Add starting vertex regular edges
        start = graph.starting_vertex()
        for edge_data in start_edges:
            to_idx = int(edge_data[0])
            weight = float(edge_data[1])

            if to_idx < 0 or to_idx >= n_vertices:
                raise RuntimeError(
                    f"Graph deserialization failed: start edge has invalid to_idx\n"
                    f"  to_idx={to_idx}, valid range=[0, {n_vertices})"
                )

            try:
                to_vertex = idx_to_vertex[to_idx]
                start.add_edge(to_vertex, weight)
            except Exception as e:
                raise RuntimeError(
                    f"Graph deserialization failed: cannot add start edge\n"
                    f"  To vertex {to_idx} (state={states[to_idx].tolist()})\n"
                    f"  Weight: {weight}\n"
                    f"  Error: {e}"
                ) from e

        # Set param_length explicitly after all edges are added
        # This ensures graph.param_length() returns the correct value for trace recording
        if param_length > 0:
            try:
                graph.set_param_length(param_length)
            except Exception:
                pass  # May fail if no parameterized edges exist, which is fine

        return graph

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
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for JAX-compatible models. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

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
        temp_file = f"/tmp/graph_model_{cpp_hash}.cpp"

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
            use_ffi = config.ffi  # User can enable with config.ffi = True

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
        """
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
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for JAX-compatible C++ models. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

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
        g.normalize();  // Normalize for discrete mode
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


    def _daisy_chain_svgd_model(
        self,
        *,
        observed_indices,
        epoch_starts,
        t_eval: float | None = None,
        user_prior=None,
        user_fixed=None,
        sd: float = 5.0,
        verbose: bool = False,
        granularity: int = 0,
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
        sd : float
            Standard-error multiplier for the broadcast prior (matches
            the ``DataPrior`` default).
        verbose : bool
            Forwarded to ``probability_matching``.
        granularity : int, optional
            Uniformization granularity forwarded to the FFI handler's
            ``stop_probability`` calls. ``0`` (default) = auto.

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
                for epoch in range(n_epochs):
                    broadcast_fixed.append(
                        (epoch * param_length + local_idx, float(value))
                    )
            fixed_indices = [idx for idx, _v in broadcast_fixed]

        def model(theta, _observed_arg=None, rewards=None):
            theta_arr = jnp.atleast_1d(theta)
            joint_probs = jsp.daisy_chain_joint_probs(
                epoch_thetas=theta_arr.reshape(n_epochs, param_length),
                epoch_dts=epoch_dts,
                initial_ipv=initial_ipv,
                t_eval=t_eval,
                fixed_indices=fixed_indices,
                granularity=granularity,
            )
            per_obs = joint_probs[observed_pos_jnp]
            return per_obs, jnp.zeros(2)

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


    def svgd(self,
             observed_data: ArrayLike | SparseObservations,
             discrete: bool | None = None,
             prior: Callable | None = None,
             n_particles: int | None = None,
             n_iterations: int = 100,
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
             joint_index: bool = False,
             rewards: ArrayLike | None = None,
             fixed: ArrayLike | None = None,
             preconditioner: str | object | None = 'auto',
             epoch_starts: ArrayLike | None = None,
             daisy_chain_t_eval: float | str | None = None,
             daisy_chain_granularity: int = 0,
             daisy_chain_probe_theta: ArrayLike | None = None,
             daisy_chain_t_eval_tol: float = 1e-3,
             ) -> dict:
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

            **With fixed parameters**:
            When using a list of priors with the `fixed` parameter, you must provide None
            at indices corresponding to fixed parameters. This is validated at initialization.

            Example:
                prior=[GaussPrior(ci=[0,1]), None, GaussPrior(ci=[0,1])],
                fixed=[(1, 0.5)]  # theta[1] fixed, prior[1] must be None
        n_particles : int, default=50
            Number of SVGD particles. More particles = better posterior approximation but slower.
        n_iterations : int, default=1000
            Number of SVGD optimization steps
        optimizer : object, optional
            Learning rate optimizer instance from phasic.optimizers. Default is Adam
            when learning_rate=None and regularization=0. Options include Adamelia, Adam,
            SGDMomentum, RMSprop, Adagrad. When an optimizer is used, the learning_rate
            parameter is ignored (the optimizer has its own learning rate).
        learning_rate : float or None, default=None
            SVGD step size. If None (default), uses Adame optimizer with adaptive
            learning rates. If a float is provided, uses fixed learning rate approach.
            Larger values = faster convergence but may be unstable.
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
            Dimension of theta parameter vector. Can be:
            - Set at graph construction: Graph(callback, theta_dim=2)
            - Overridden here for SVGD inference (if graph was modified/augmented)

            If None, inferred from the graph's parameterized edge structure via param_length().
            Only required if theta_init is None and the graph has no parameterized edges.

            The value specified here overrides any theta_dim set during graph construction,
            which is useful if you've modified the graph structure (e.g., via extend()).
        return_history : bool, default=True
            If True, return particle positions throughout optimization
        seed : int, default=None
            Random seed for reproducibility
        verbose : bool, default=True
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
        joint_index : bool, default=False
            If True, use joint index mode where observed_data contains vertex indices (integers)
            instead of time values. In this mode, likelihood is computed from converged
            accumulated_visits() values rather than PDF/PMF values. This is used for joint
            index distributions in population genetics models.

            When joint_index=True:
            - observed_data should contain vertex indices (integers)
            - Forces discrete=True behavior
            - Moment regularization is not supported (raises NotImplementedError if regularization > 0)
        rewards : ArrayLike, optional
            Reward vectors for computing reward-transformed likelihoods. Can be:
            - None: Standard phase-type likelihood (default)
            - 1D array (n_vertices,): Single reward vector for univariate models
            - 2D array (n_vertices, n_features): Multivariate rewards - one reward vector per feature
              dimension. Requires use of pmf_and_moments_from_graph_multivariate() model.
            For multivariate models, observed_data should also be 2D (n_times, n_features).
        preconditioner : str, preconditioner instance, or None, default='auto'
            Preconditioning method for multi-scale parameters:
            - 'auto' or 'jacobian': Moment Jacobian preconditioning (default, recommended).
              Uses column norms of the moment Jacobian matrix for scaling. Simpler and
              more robust than Fisher preconditioning.
            - 'fisher': Fisher diagonal preconditioning. Uses empirical Fisher information
              matrix diagonal. Can be unstable when PMF values are small.
            - None or 'none': No preconditioning (original behavior)
            - MomentJacobianPreconditioner or FisherPreconditioner instance: Custom preconditioner
        epoch_starts : array-like of float, optional
            Enables daisy-chain (time-inhomogeneous) inference. ``epoch_starts[0] == 0``;
            subsequent entries are the start times of additional epochs. ``n_epochs =
            len(epoch_starts)``. Each epoch fits its own ``param_length`` parameters,
            so the flattened theta has length ``n_epochs * param_length``. Requires
            a continuous-time joint-prob graph (``discrete=False``).
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

        Returns
        -------
        dict
            Inference results containing:
            - 'particles': Final posterior samples (n_particles, theta_dim)
            - 'theta_mean': Posterior mean estimate
            - 'theta_std': Posterior standard deviation
            - 'history': Particle evolution over iterations (if return_history=True)

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
        # Check JAX availability
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for SVGD inference. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

        from .svgd import SVGD

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

        if discrete is None:
            discrete = self.is_discrete

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
            except Exception:
                pass  # Fall through to SVGD's standard-normal default

        # Handle joint_index mode
        if self._joint_prob_base_graph_indexer is not None:
            logger = get_logger(__name__)

            if not joint_index:
                logger.info("Graph was constructed with joint index support. "
                  "joint_index=True is implied.")
            joint_index = True # FIXME: joint_index is always True if graph supports it, so not really needed as argument

            if not self._joint_prob_base_graph_indexer:
                raise ValueError(
                    "Graph was not constructed with joint index support. "
                    "Cannot use joint_index=True."
                )
            # map observed data to indices in joint probability table
            joint_prob_table = self.joint_prob_table()
            obs_indices = []
            obs2idx = joint_prob_table.groupby(joint_prob_table.columns[:-1].to_list()).groups
            for obs in observed_data:
                idx = obs2idx[tuple(obs)]
                if idx.size > 1:
                    # if observation maps to multiple indices, sample according to their probabilities
                    p = joint_prob_table.loc[idx, 'prob'].to_numpy()
                    p = p / p.sum()
                    chosen_idx = np.random.choice(idx, p=p)
                    obs_indices.append(chosen_idx.item())
                else:
                    # else just return the unique index
                    obs_indices.append(idx.item())
            observed_data = obs_indices

            # Check for unsupported combinations
            if regularization > 0:
                print("Warning: Moment regularization is not implemented with joint_index=True")
                raise NotImplementedError(
                    "Moment regularization is not supported with joint_index=True. "
                    "Set regularization=0 or use joint_index=False."
                )
            if rewards is not None:
                print("Warning: Reward transformation is not supported with joint_index=True")
                raise NotImplementedError(
                    "Reward transformation is not supported with joint_index=True. "
                    "Set rewards=None or use joint_index=False."
                )
            # Force discrete mode for joint_index
            discrete = True

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
                model, theta_dim, prior, fixed = self._daisy_chain_svgd_model(
                    observed_indices=observed_data,
                    epoch_starts=epoch_starts,
                    t_eval=resolved_t_eval,
                    user_prior=prior,
                    user_fixed=fixed,
                    sd=5.0,
                    verbose=verbose,
                    granularity=daisy_chain_granularity,
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
                # Use joint_index specific model with fixed_mask
                model = Graph.pmf_from_graph_joint_index(
                    self, theta_dim=theta_dim,
                    fixed_mask=fixed_mask_for_model,
                )
        # Auto-detect if we need multivariate model (2D rewards)
        elif rewards is not None:
            import jax.numpy as jnp
            rewards_arr = jnp.asarray(rewards, dtype=jnp.float64)  # Ensure float64 for C++ compatibility
            if rewards_arr.ndim == 2:
                # Use multivariate model for 2D rewards
                model = Graph.pmf_and_moments_from_graph_multivariate(
                    self, nr_moments=nr_moments, discrete=discrete,
                    use_ffi=False, theta_dim=theta_dim
                )
            else:
                # Use standard model for 1D rewards
                model = Graph.pmf_and_moments_from_graph(
                    self, nr_moments=nr_moments, discrete=discrete,
                    theta_dim=theta_dim
                )
        else:
            # No rewards - use standard model
            model = Graph.pmf_and_moments_from_graph(
                self, nr_moments=nr_moments, discrete=discrete,
                theta_dim=theta_dim
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
            preconditioner=preconditioner
        )

        # Run inference
        svgd.optimize(return_history=return_history)

        # Return results as dictionary for backward compatibility
        # return svgd.get_results()

        return svgd

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
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for MCMC inference. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

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
            Number of model parameters.  Inferred from the graph when ``None``.
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
        # Check if JAX is available
        if not HAS_JAX and not use_ffi:
            raise ImportError(
                "JAX is required for JAX-compatible models. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

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

        for (int i = 1; i < nr_moments; i++) {{
            // Compute higher moments
            for (int j = 0; j < (int)rewards3.size(); j++) {{
                rewards3[j] = rewards2[j] * std::pow(rewards2[j], i);
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
                                   theta_dim: int | None = None) -> Callable:
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
        """
        # Check if JAX is available
        if not HAS_JAX and not use_ffi:
            raise ImportError(
                "JAX is required for JAX-compatible models. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

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
                return builder.compute_pmf_and_moments(
                    np.zeros(0), times_np,
                    nr_moments=nr_moments, discrete=discrete,
                    granularity=0, rewards=rewards_np
                )

            def _compute_pure(theta, times, rewards=None):
                pmf_shape = jax.ShapeDtypeStruct(times.shape, times.dtype)
                moments_shape = jax.ShapeDtypeStruct((nr_moments,), times.dtype)

                def _cb(t, tm):
                    pmf, moments = _compute_callback(
                        np.asarray(t, dtype=np.float64),
                        np.asarray(tm, dtype=np.float64),
                        np.asarray(rewards, dtype=np.float64) if rewards is not None else None
                    )
                    return pmf.astype(times.dtype), moments.astype(times.dtype)

                return jax.pure_callback(
                    _cb, (pmf_shape, moments_shape),
                    theta, times,
                    vmap_method='sequential'
                )

            # Jump to the VJP wrapping below (shared with FFI/pybind paths)
            # Fall through to the VJP code at the end of this method
            use_ffi = False
            _callback_mode = True
        else:
            _callback_mode = False

        if not _callback_mode:
            # Check if FFI is available - respect parameter, allow config override
            config = get_config()
            if not use_ffi:  # If explicitly disabled, respect it
                use_ffi = False
            else:  # If True or default, check config
                use_ffi = config.ffi  # Enable FFI for multi-core parallelization (C++ binding fixed!)

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
                        pmf, moments = builder.compute_pmf_and_moments(
                            theta_single,
                            times_unbatched,
                            nr_moments=nr_moments,
                            discrete=discrete,
                            granularity=0,
                            rewards=rewards_np  # Pass optional rewards
                        )
                        pmf_results.append(pmf)
                        moments_results.append(moments)
                    return np.array(pmf_results), np.array(moments_results)
                else:
                    # Unbatched case
                    pmf, moments = builder.compute_pmf_and_moments(
                        theta_np,
                        times_np,
                        nr_moments=nr_moments,
                        discrete=discrete,
                        granularity=0,
                        rewards=rewards_np  # Pass optional rewards
                    )
                    return pmf, moments

            # Helper function for pure callback (used in forward and backward pass)
            def _compute_pure(theta, times, rewards=None):
                """Pure computation without custom_vjp wrapper"""
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

            # Finite differences for gradient
            # Clamp lower perturbation to stay positive (theta comes from
            # softplus which can be as small as 1e-9, smaller than eps)
            min_theta = 1e-9
            theta_bar = []
            for i in range(n_params):
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].set(jnp.maximum(theta[i] - eps, min_theta))
                actual_diff = theta_plus[i] - theta_minus[i]

                # Call underlying computation, not model
                pmf_plus, moments_plus = _compute_pure(theta_plus, times, rewards)
                pmf_minus, moments_minus = _compute_pure(theta_minus, times, rewards)

                # Combine gradients from both PMF and moments
                # Use nansum to handle NaN values in PMF (from missing observations)
                # NaN in PMF means the observation was missing, so it shouldn't contribute to gradient
                pmf_diff = (pmf_plus - pmf_minus) / actual_diff
                grad_pmf_i = jnp.nansum(g_pmf * pmf_diff)
                grad_moments_i = jnp.sum(g_moments * (moments_plus - moments_minus) / actual_diff)
                grad_i = grad_pmf_i + grad_moments_i

                theta_bar.append(grad_i)

            return jnp.array(theta_bar), None, None  # gradients for theta, times, rewards

        model.defvjp(model_fwd, model_bwd)
        return model

    @classmethod
    def pmf_from_graph_joint_index(cls, graph: Graph, theta_dim: int | None = None,
                                    fixed_mask: Any = None,
                                    exclude_vertices: list[int] | None = None) -> Callable:
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
        param_length : int, optional
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

        Returns
        -------
        callable
            JAX-compatible function with signature:
            model(theta, vertex_indices, rewards=None) -> (sojourn_times, dummy_moments)

            Where:
            - theta: Parameter vector
            - vertex_indices: Array of vertex indices (integers)
            - rewards: Ignored (must be None for joint_index mode)
            - sojourn_times: Expected sojourn times for the specified vertices
            - dummy_moments: Zeros array (moments not supported in joint_index mode)

        Notes
        -----
        - Uses expected_sojourn_time() for fast exact computation
        - Much faster than iterating accumulated_visiting_time() until convergence
        - Moment regularization is not supported (regularization must be 0)
        - Reward transformation is not supported (rewards must be None)
        """
        # Check if JAX is available
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for JAX-compatible models. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

        import jax
        import jax.numpy as jnp
        from .ffi_wrappers import compute_sojourn_times_ffi

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
        all_terminal_indices = jnp.array(all_terminal_indices, dtype=jnp.int32)

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
            import numpy as np
            fixed_mask_np = np.asarray(fixed_mask)
            fixed_indices_set = set(np.where(fixed_mask_np == 1)[0].tolist())

        def model_bwd(res, g):
            theta, vertex_indices = res
            g_visits, g_moments = g  # Unpack gradient tuple

            n_params = theta.shape[0]
            eps = 1e-7

            # Finite differences for gradient
            theta_bar = []
            for i in range(n_params):
                # Skip finite differences for fixed parameters - gradient is zero
                # This is critical for correct chain rule when using compiled_grad_reduced
                if i in fixed_indices_set:
                    theta_bar.append(0.0)
                    continue

                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].add(-eps)

                visits_plus, _ = _compute_pure(theta_plus, vertex_indices)
                visits_minus, _ = _compute_pure(theta_minus, vertex_indices)

                # Gradient only from visits (moments are dummy zeros)
                grad_i = jnp.sum(g_visits * (visits_plus - visits_minus) / (2 * eps))
                theta_bar.append(grad_i)

            return jnp.array(theta_bar), None, None  # gradients for theta, vertex_indices, rewards

        model.defvjp(model_fwd, model_bwd)
        return model

    @classmethod
    def pmf_and_moments_from_graph_multivariate(cls, graph: Graph, nr_moments: int = 2,
                                                discrete: bool = False, use_ffi: bool = False,
                                                theta_dim: int | None = None) -> Callable:
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
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for multivariate models. "
                "Install with: pip install 'phasic[jax]' or pip install jax jaxlib"
            )

        import jax
        import jax.numpy as jnp

        # Get the 1D model
        model_1d = cls.pmf_and_moments_from_graph(
            graph, nr_moments=nr_moments, discrete=discrete,
            use_ffi=use_ffi, theta_dim=theta_dim
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

    def plot(
        graph: Any, 
        filename: str | None = None,
        wrap: bool|int = True,
        label_fmt: Callable[[Any], str] | None = None,
        subgraphfun: Callable[..., str] | None = None,
        by_state: Callable[..., str] | None = None,
        by_index: Callable[[int], str] | None = None,
        max_nodes: int = 100,
        dark: bool | None = None,
        constraint: bool = True, ranksep: float = 1, nodesep: float = 1, rankdir: str = "LR",
        size: tuple[int, int] = (7, 7), fontsize: int = 12, rainbow: bool = True, penwidth: float = 1,
        seed: int = 1,
        **kwargs: Any) -> graphviz.Digraph | None:
        """Plot a graph using graphviz.

        Parameters
        ----------
        graph : Graph
            The phasic graph object to visualize.
        filename : str | None
            If provided, save the graph to this file. The file extension
            determines the output format (e.g., ``'graph.pdf'``).
        wrap : bool | int
            Whether to wrap vertex labels, and if so, the maximum number of
            characters per line. By default True.
        subgraphfun : Callable[..., str] | None
            Deprecated. Use ``by_state`` instead. Callback function defining
            subgraph clusters by state.
        by_state : Callable[..., str] | None
            Callback function defining subgraph clusters. Takes a state as
            input and returns a string used as the subgraph label.
        by_index : Callable[[int], str] | None
            Callback function defining subgraph clusters. Takes a vertex
            index as input and returns a string used as the subgraph label.
        max_nodes : int
            Maximum number of vertices to plot, by default 100.
        dark : bool | None
            Whether to use dark mode for the graph. Detected automatically
            from the VS Code theme if ``vscodenb`` is available.
        constraint : bool
            Graphviz constraint attribute, by default True.
        ranksep : float
            Graphviz ranksep attribute, by default 1.
        nodesep : float
            Graphviz nodesep attribute, by default 1.
        rankdir : str
            Graphviz rankdir attribute, by default ``"LR"``.
        size : tuple[int, int]
            Graphviz size as ``(width, height)``, by default ``(7, 7)``.
        fontsize : int
            Graphviz fontsize attribute, by default 12.
        rainbow : bool
            Whether to color edges with random colors, by default True.
        penwidth : float
            Graphviz penwidth attribute, by default 1.
        seed : int
            Random seed for graph layout, by default 1.
        **kwargs : Any
            Additional graphviz graph attributes.

        Returns
        -------
        graphviz.Digraph | None
            Graphviz Digraph object for display in Jupyter notebooks,
            or ``None`` if the graph exceeds ``max_nodes``.
        """


        import math
        import os
        import subprocess
        import graphviz
        from collections import defaultdict
        import seaborn as sns
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.colors
        from itertools import cycle
        from functools import partial

        from typing import Any, TypeVar
        from collections.abc import Callable, Generator

        from .logging_config import get_logger
        logger = get_logger(__name__)

        GraphType = TypeVar('Graph')


        def _get_color(n: int, lightness: float = 0.4) -> Generator[str, None, None]:
            """Generate an infinite cycle of hex color strings from a HUSL palette.

            Parameters
            ----------
            n : int
                Number of distinct colors in the palette.
            lightness : float
                Lightness parameter for the HUSL palette, by default 0.4.

            Yields
            ------
            str
                Hex color string (e.g., ``'#a1b2c3'``).
            """
            color_cycle = cycle([matplotlib.colors.to_hex(c) for c in sns.husl_palette(n, l=lightness)])
            for color in color_cycle:
                yield color

        def _format_rate(rate: float) -> str:
            """Format a transition rate for display on graph edges.

            Parameters
            ----------
            rate : float
                The transition rate value.

            Returns
            -------
            str
                Formatted string using fixed-point for integers, scientific
                notation otherwise.
            """
            if rate == round(rate):
                return f"{rate:.2f}"
            else:
                return f"{rate:.2e}"

        def format_label(vertex, wrap=True, max_cols=8):
            state = vertex.state()
            n = len(state) 
            if wrap is False or n <= max_cols:
                return ','.join(map(str, state))
            
            if wrap is True:

                target = math.isqrt(n // 2) or 1
                best = (1, n)                                                                     
                for b in range(max(1, target - 100), target + 101):                               
                    if n % b == 0:                                                                
                        a = n // b                                                                
                        if abs(a - 2 * b) < abs(best[0] - 2 * best[1]):                           
                            best = (a, b)   
                rows, cols = best

                # for i in range(int(math.sqrt(n)), 0, -1):
                #     if n % i == 0:
                #         rows, cols = i, n // i
                #         break

                cols = cols if cols < max_cols else int(math.sqrt(n)+2)
            elif isinstance(wrap, int):
                cols = wrap
            else:
                cols = 9999
            l = []
            for i in range(1+n//cols):
                r = ','.join(map(str, state[i*cols:(i+1)*cols]))
                if not r:
                    break
                l.append(r)
            return ',\n'.join(l)


        try:
            from vscodenb import is_vscode_dark_theme
            dark = is_vscode_dark_theme()
        except ImportError:
            logger.warning(f"vscodenb is not available. Defaulting to light theme.")
            dark = False

        # always light theme when executing via nbconvert
        if 'NBCONVERT_BGCOLOR' in os.environ:
            dark = False

        if label_fmt is None:
            label_fmt = partial(format_label, wrap=wrap)
        elif label_fmt is False:
            label_fmt = lambda vertex: str(vertex.index())

        subprocess.check_call(['dot', '-c']) # register layout engine

        # backwards comp
        if by_state is None and subgraphfun is not None:
            by_state = subgraphfun

        if by_state and by_index:
            assert "Do not use both by_index and by_state"

        # get matplotlib background color for graph background
        plt.ioff()
        fig, ax = plt.subplots()
        bg_color = ax.get_facecolor()
        plt.close(fig)
        plt.ion()
        bg_color = matplotlib.colors.to_hex(bg_color)

        if dark:
            edge_color = '#e6e6e6'
            node_edgecolor = '#888888'
            node_fillcolor = "#c6c6c6"
            start_edgecolor = 'black'
            start_fillcolor = '#777777'
            abs_edgecolor = 'black'
            abs_fillcolor = '#777777'
            aux_edgecolor = 'black'
            aux_fillcolor = '#3e3e3e'
            # bgcolor = '#1F1F1F'
            bgcolor = bg_color
            subgraph_label_fontcolor = '#e6e6e6'
            subgraph_bgcolor='#2e2e2e'
            subgraph_edgecolor='#e6e6e6'
            husl_colors = _get_color(10, lightness=0.7)
        else:
            edge_color = '#3e3e3e'
            node_edgecolor='black'
            node_fillcolor='#eeeeee'
            edge_color='black'
            start_edgecolor='black'
            start_fillcolor='#bbbbbb'
            abs_edgecolor='black'
            abs_fillcolor='#bbbbbb'
            aux_edgecolor='black'
            aux_fillcolor='#bbbbbb'
            # bgcolor='transparent'
            bgcolor=bg_color
            subgraph_label_fontcolor = 'black'
            # subgraph_bgcolor='white'
            subgraph_bgcolor=bg_color
            subgraph_edgecolor='black'
            husl_colors = _get_color(10, lightness=0.4)

        if graph.vertices_length() > max_nodes:
            print(f"Graph has too many nodes ({graph.vertices_length()}). Please set max_nodes to a higher value.")
            return None

        graph_attr = dict(compound='true', newrank='true', pad='0.5',
                        ranksep=str(ranksep), nodesep=str(nodesep),
                        bgcolor=bgcolor, rankdir=rankdir, ratio="auto",
                        size=f'{size[0]},{size[1]}',
                        start=str(seed),
                        fontname="Helvetica,Arial,sans-serif", **kwargs)
        node_attr = dict(style='filled', color='black',
                        fontname="Helvetica,Arial,sans-serif",
                        fontsize=str(fontsize),
                        fillcolor=str(node_fillcolor))
        edge_attr = dict(constraint='true' if constraint else 'false',
                        style='filled', labelfloat='false', labeldistance='0',
                        fontname="Helvetica,Arial,sans-serif",
                        fontsize=str(fontsize), penwidth=str(penwidth))
        dot = graphviz.Digraph(graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr)
        for i in range(graph.vertices_length()):
            vertex = graph.vertex_at(i)
            for edge in vertex.edges():
                if rainbow:
                    color = next(husl_colors)
                else:
                    color = edge_color
                dot.edge(str(vertex.index()), str(edge.to().index()),
                    xlabel=_format_rate(edge.weight()), color=color, fontcolor=color)

        subgraph_attr = dict(rank='same',
                            style='filled',
                            fillcolor=subgraph_bgcolor,
                            color=subgraph_edgecolor,
                            fontcolor=subgraph_label_fontcolor)
        subgraphs = defaultdict(list)
        for i in range(graph.vertices_length()):
            vertex = graph.vertex_at(i)
            label = label_fmt(vertex)
            if i == 0:
                dot.node(str(vertex.index()), 'S',
                        style='filled', edge_color=start_edgecolor, fillcolor=start_fillcolor)
            elif not vertex.state().sum() and vertex.rate() == 1 and len(vertex.edges()) == 1:
                dot.node(str(vertex.index()), 'AUX',
                        style='filled', edge_color=aux_edgecolor, fillcolor=aux_fillcolor)
            elif not vertex.edges():
                dot.node(str(vertex.index()), label,
                        style='filled', edge_color=abs_edgecolor, fillcolor=abs_fillcolor)
            else:
                dot.node(str(vertex.index()), label,
                        style='filled', edge_color=node_edgecolor, fillcolor=node_fillcolor)

            if i != 0:
                if by_state:
                    subgraphs[f'cluster_{by_state(vertex.state())}'].append(i)
                elif by_index:
                    subgraphs[f'cluster_{by_index(vertex.index())}'].append(i)

        if by_state or by_index:
            for sglabel in subgraphs:
                subgraph_attr['label'] = sglabel.replace('cluster_', '')
                with dot.subgraph(name=sglabel, graph_attr=subgraph_attr) as c:
                    for i in subgraphs[sglabel]:
                        vertex = graph.vertex_at(i)
                        c.node(str(vertex.index()))

        if filename:
            name, suffix = filename.rsplit('.', 1)
            dot.render(name, format=suffix, cleanup=True)

        return dot



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
        new_graph._weight_mode = self._weight_mode
        new_graph._weight_callback = self._weight_callback
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

    def eliminate_to_dag(self) -> SymbolicDAG:
        """
        Perform symbolic graph elimination to create a reusable DAG structure.

        This method performs the O(n³) graph elimination algorithm ONCE and
        returns a symbolic DAG where edges contain expression trees instead
        of concrete values. The DAG can then be instantiated with different
        parameters in O(n) time each.

        This is the key optimization for SVGD and other inference methods
        that require evaluating the same graph structure with many different
        parameter vectors.

        Returns
        -------
        SymbolicDAG
            Symbolic DAG that can be instantiated with parameters

        Raises
        ------
        RuntimeError
            If the graph is not parameterized or elimination fails

        Examples
        --------
        >>> # Create parameterized graph
        >>> g = Graph(1)
        >>> v_a = g.create_vertex([0])
        >>> v_b = g.create_vertex([1])
        >>> v_c = g.create_vertex([2])
        >>> v_a.add_edge_parameterized(v_b, 0.0, [1.0, 0.0, 0.0])
        >>> v_b.add_edge_parameterized(v_c, 0.0, [0.0, 1.0, 0.0])

        >>> # Eliminate to symbolic DAG (once)
        >>> dag = g.eliminate_to_dag()
        >>> print(dag)  # SymbolicDAG(vertices=3, params=3, acyclic=True)

        >>> # Fast instantiation for SVGD (100-1000× faster!)
        >>> for theta in particle_swarm:
        ...     g_concrete = dag.instantiate(theta)
        ...     log_prob = -g_concrete.expectation()  # Fast!

        Performance
        -----------
        - Elimination: O(n³) - performed once
        - Instantiation: O(n) - performed per particle
        - Expected speedup for SVGD: 100-1000×

        See Also
        --------
        SymbolicDAG : The returned symbolic DAG class
        SymbolicDAG.instantiate : Create concrete graph from parameters
        """
        ptr = self._eliminate_to_dag_internal()
        return SymbolicDAG(ptr)






    def _joint_prob_reward(self,
                           state: np.ndarray,
                        indexer: StateIndexer,
                        reward_indexer: StateIndexer,
                        current_rewards: np.ndarray | None = None,
                        mutation_rate: float = 1.0,
                        reward_limit: int | dict = 10,
                        tot_reward_limit: float = np.inf) -> tuple[np.ndarray, float]:

            logger = get_logger(__name__)

            prop_set_names = [p.name for p in indexer.property_sets()]
            prop_set_name, *_ = prop_set_names

            # determine reward dimensions (extension to state vector)
            # reward_length = indexer.state_length - indexer[prop_set_name].state_length
            reward_length = reward_indexer.state_length

            if not isinstance(reward_limit, dict):
                reward_limits = np.repeat(reward_limit, reward_length)

            if current_rewards is None:
                current_rewards = np.zeros(reward_length)

            reward_rates = np.zeros(reward_length)
            trash_rate = 0

            reward_prop_names = set(prop.name for prop_set in reward_indexer.property_sets() for prop in prop_set.properties)

            # for each base graph state index
            for i in range(indexer[prop_set_name].state_length):
                # get properties for the property set
                props = indexer[prop_set_name].index_to_props(i, as_dict=True)

#                props = indexer.index_to_props(i, as_dict=True)
                # for prop, value in getattr(props, prop_set_name).items():

                # for each property and its value
                for prop, value in props.items():

                    # make flattened prop_set + property nmae
                    _prop_name = f'{prop_set_name}_{prop}'

                    if _prop_name not in reward_prop_names:
                        continue

                    reward_idx = reward_indexer.props_to_index(**{_prop_name: value})
                    rate = state[i] * mutation_rate 

                    # logger.debug("i: %d; prop: %s; value: %s; rate: %e; reward_idx: %d", i, repr(prop), repr(value), rate, reward_idx)
                    if isinstance(reward_limit, dict):
                        if current_rewards[i] + 1 > reward_limit[prop] and np.sum(current_rewards + r) <= tot_reward_limit:
                            reward_rates[reward_idx] += rate
                        else:
                            trash_rate = trash_rate + rate
                    else:
                        r = np.zeros_like(reward_rates)
                        r[reward_idx] = 1
                        if (reward_limit is None or np.all(current_rewards + r <= reward_limits)) and np.sum(current_rewards + r) <= tot_reward_limit:
                            reward_rates[reward_idx] += rate
                        else:
                            trash_rate = trash_rate + rate

            return reward_rates, trash_rate


    def joint_prob_graph(self,
                        base_graph_indexer: StateIndexer | None = None,
                        reward_only: list | None = None,
                        reward_rates_callback: Callable | None = None,
                        mutation_rate: float = 1.0,
                        reward_limit: int | None = None,
                        tot_reward_limit: float = np.inf,
                        discrete: bool = True) -> Graph:

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

        if reward_rates_callback is None:
            # default to joint prob reward callback
            reward_rates_callback = self._joint_prob_reward

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

        # logger.debug(f"Reward indexer created with {reward_indexer.state_length} states: {reward_indexer}")

        # append reward indexer to original indexer
        joint_graph_indexer = base_graph_indexer + reward_indexer

        # joint graph state vector length
        state_vector_length = joint_graph_indexer.state_length

        # indices for original and new parts of the state vector
        state_indices = base_graph_indexer.indices()
        reward_state_indices = np.arange(base_graph_indexer.state_length, joint_graph_indexer.state_length)

        # create the new graph
        joint_graph = Graph(state_vector_length)
        starting_vertex = joint_graph.starting_vertex()

        # array of zeros for extension of state vector
        null_rewards = np.zeros(reward_length)

        # graph index of last vertex visited
        index = 0

        # get param_length for extracting parameterized edge coefficients
        param_length = self.param_length()

        # copy initial (extended) states to new graph
        for edge in base_starting_vertex.parameterized_edges():
            starting_vertex.add_edge(
            joint_graph.find_or_create_vertex(
                np.append(edge.to().state(), null_rewards).astype(int)),
            edge.weight())

        # pgbar
        # pgbar_prev = 0    
        # pgbar = tqdm(position=0, total=1, miniters=0, 
        #             desc='Visited / Created', bar_format='{l_bar}{bar}'
        #             )
        index = index + 1

        # weights of edges to trash    
        trash_rates = {}

        # indices of t-states (with absorbing as only child)
        t_vertex_indices = np.array([], dtype=int)

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
                # extended with a with a copy of the extended part of the 
                # current states state vector
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
                
                # get the edge state (ensuring it is param_length)
                coeffs = list(edge.edge_state(param_length))
                
                # Pad with 0 for mutation rate slot
                coeffs.append(0) 

                # add edge to the child vertex
                vertex.add_edge(child_vertex, coeffs)

                # if the base graph version of the child state was
                # absorbing, we add it to the array of t-states
                if not self.find_vertex(child_state[state_indices]).edges():
                    t_vertex_indices = np.append(t_vertex_indices, child_vertex.index()) 

            # base part of current state
            current_state = state[state_indices]
            # extended part of current state
            current_rewards = state[reward_state_indices]

            # get rates to states representing an additional mutation
            rates, trash_rate = reward_rates_callback(
                current_state, 
                base_graph_indexer, 
                reward_indexer,
                current_rewards, 
                mutation_rate=mutation_rate, 
                reward_limit=reward_limit, 
                tot_reward_limit=tot_reward_limit
                ) 

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
                    vertex.add_edge(child_vertex, np.append(np.zeros(self.param_length()), rate))
                                    
            index = index + 1 

        t_vertex_indices = np.unique(t_vertex_indices).tolist()

        #     pgbar_this = index/joint_graph.vertices_length()
        #     pgbar.update(pgbar_this - pgbar_prev)
        #     pgbar_prev = pgbar_this

        # pgbar.close()

        # create trash vertices
        trash_vertex = joint_graph.find_or_create_vertex(np.repeat(0, state_vector_length))
        trash_loop_vertex = joint_graph.create_vertex(np.repeat(0, state_vector_length))
        trash_vertex.add_edge(trash_loop_vertex, np.append(np.zeros(self.param_length()), 1.0))
        trash_loop_vertex.add_edge(trash_vertex, np.append(np.zeros(self.param_length()), 1.0))

        # connect edges to first trash state
        for i, rate in trash_rates.items():
            if rate > 0:
                joint_graph.vertex_at(i).add_edge(trash_vertex, np.append(np.zeros(self.param_length()), rate))


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
                        np.append(np.zeros(self.param_length()), 1.0)
                        )
                    t_vertex_indices.remove(i)
                t_vertex_indices.append(t_set_abs.index())


        # the t-states represent variants of the original absorbing state
        # add a new absorbing with edges from all t-states
        new_absorbing = joint_graph.create_vertex(np.repeat(0, state_vector_length))
        
        for i in t_vertex_indices:
            joint_graph.vertex_at(i).add_edge(new_absorbing, np.append(np.zeros(self.param_length()), 1.0))

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
                t_aux_vertex = new.create_vertex([0] * self.state_length())
                # Unit-weight parameterised edges in both directions; using
                # the modern list-based add_edge to avoid the
                # add_edge_parameterized deprecation warning.
                nv.add_edge(t_aux_vertex, [1.0] * param_length)
                t_aux_vertex.add_edge(nv, [1.0] * param_length)
                t_aux_map[nv.index()] = t_aux_vertex.index()
                continue

            for e in v.parameterized_edges():
                to_index = e.to().index()
                if to_index in trash_old_indices:
                    to_index = abs_old_index
                nv.add_edge(vmap[to_index], list(e.edge_state(param_length)))

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
        # New-graph vertex indices that carry IPV edges, in starting-vertex
        # edge order. update_ipv expects a vector of this length, in this
        # order.
        new._ipv_target_indices = ipv_target_indices
        new.is_discrete = self.is_discrete
        new._cache_trace = getattr(self, '_cache_trace', False)
        # Forward the indexer like joint_prob_graph does.
        if hasattr(self, '_indexer'):
            new._indexer = self._indexer
        return new


    def daisy_chain_joint_probs(
        self,
        *,
        epoch_thetas,
        epoch_dts,
        initial_ipv,
        t_eval: float | None = None,
        fixed_indices=None,
        granularity: int = 0,
    ):
        """JAX-traceable model: joint-probs at the t-states after a daisy chain.

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
        )
        import json as _json_mod

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
        structure_json_str = _json_mod.dumps(structure)

        # The full forward computation as a flat-theta function. The
        # custom_vjp wrapper differentiates only theta_flat (initial_ipv
        # is closed over and treated as fixed). Single FFI call replaces
        # the per-epoch pure_callback chain.
        def _forward(theta_flat: jnp.ndarray) -> jnp.ndarray:
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


    def joint_prob_table(self) -> pd.DataFrame:

        if not self._joint_prob_base_graph_indexer:
            raise ValueError("Graph is not a joint probability representation.")

        outcomes, probs, t_vertex_indices = self._get_joint_probs()

        records = [[*obs, prob, idx] for obs, prob, idx in zip(outcomes, probs, t_vertex_indices)]
        column_names = []

        for p in self._rewarded_props:
            for i in range(p.min_value, p.max_value+1):
                column_names.append(f"{p.name}_{i}")
        column_names.extend(['prob', 't_vertex_index'])
        joint = pd.DataFrame(records, columns=column_names).set_index('t_vertex_index')
        return joint


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
    get_parallel_config : Query current parallel configuration
    detect_environment : Inspect environment without configuring
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

    # Configure JAX for environment
    config = configure_jax_for_environment(env_info, enable_x64=enable_x64)

    # Store globally
    set_parallel_config(config)

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
    'PTDAlgorithmsConfig',
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
