from ast import arg
from functools import partial
from collections import defaultdict, OrderedDict
from unittest import result
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from typing import Any, TypeVar, List, Tuple, Dict, Union, NamedTuple, Optional, Self
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
from phasic.graph_cache import GraphCache, get_graph_cache_stats, print_graph_cache_info

# from .vscode_theme import set_phasic_theme
# from .vscode_theme import phasic_theme as theme
# from .vscode_theme import set_theme # backwards compatibility
from . import plot

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
        def get_performance_cores():
            """Get number of performance cores on Apple Silicon, or total CPUs otherwise"""
            try:
                import subprocess
                import platform

                # Check if we're on Apple Silicon
                if platform.system() == 'Darwin' and platform.machine() == 'arm64':
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
            return os.cpu_count() or 1

        # Configure multi-device CPU count (for pmap)
        cpu_count = int(os.environ.get('PTDALG_CPUS', get_performance_cores()))
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
        def __init__(self, original):
            self.original = original
            self.buffer = ''

        def write(self, text):
            # Buffer the text to check full lines
            self.buffer += text

            # Process complete lines
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                line += '\n'

                # Filter out device list lines
                if not ('CpuDevice' in line or 'GpuDevice' in line):
                    self.original.write(line)

        def flush(self):
            # Flush any remaining buffer (except device lists)
            if self.buffer and not ('CpuDevice' in self.buffer or 'GpuDevice' in self.buffer):
                self.original.write(self.buffer)
                self.buffer = ''
            self.original.flush()

        def __getattr__(self, name):
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
    )
else:
    SVGD = None
    Prior = None
    GaussPrior = None
    HalfCauchyPrior = None
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

# Hash-based trace lookup (convenience wrapper)
def get_trace_by_hash(graph_hash: str, force_download: bool = False, backend: 'TransportBackend | None' = None):
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

def _compute_pmf_from_ctypes(theta, times, compute_func, graph_data, granularity, discrete):
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


def _create_jax_callback_wrapper(compute_func, graph_data, discrete):
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


def _create_jax_parameterized_wrapper(compute_func, graph_builder, discrete):
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

def _get_package_dir():
    """Get package root directory (caching is acceptable)."""
    return pathlib.Path(__file__).parent.parent.parent


def _serialize_graph_data(serialized):
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


def _generate_cpp_from_graph(serialized):
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
            base_weight = edge[1]
            edge_state = edge[2:]
            # Generate weight computation: w = base_weight + x1*theta[0] + x2*theta[1] + ...
            # Only include non-zero coefficients for efficiency and correctness
            weight_terms = [f"{edge_state[j]}*theta[{j}]"
                           for j in range(len(edge_state))
                           if edge_state[j] != 0.0]
            if weight_terms:
                weight_expr = f"{base_weight} + " + " + ".join(weight_terms)
            else:
                # All coefficients are zero - use only base_weight
                weight_expr = f"{base_weight}"
            param_edge_code.append(f"    double w_start_{to_idx} = {weight_expr};")
            param_edge_code.append(f"    start->add_edge(*vertices[{to_idx}], w_start_{to_idx});")

        # Regular vertex parameterized edges
        for i, edge in enumerate(param_edges):
            from_idx = int(edge[0])
            to_idx = int(edge[1])
            base_weight = edge[2]
            edge_state = edge[3:]
            # Generate weight computation: w = base_weight + x1*theta[0] + x2*theta[1] + ...
            # Only include non-zero coefficients for efficiency and correctness
            weight_terms = [f"{edge_state[j]}*theta[{j}]"
                           for j in range(len(edge_state))
                           if edge_state[j] != 0.0]
            if weight_terms:
                weight_expr = f"{base_weight} + " + " + ".join(weight_terms)
            else:
                # All coefficients are zero - use only base_weight
                weight_expr = f"{base_weight}"
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


def _generate_cpp_from_trace(trace, observed_data, granularity=0):
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


def _compile_trace_library(cpp_code, trace_hash):
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


def clear_trace_cache():
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


def _wrap_trace_log_likelihood_for_jax(lib_path, param_length):
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


def _compile_wrapper_library(wrapper_code, lib_name, extra_includes=None):
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
        cmd.extend([
            wrapper_file,
            f'{pkg_dir}/src/cpp/phasiccpp.cpp',
            f'{pkg_dir}/src/c/phasic.c',
            f'{pkg_dir}/src/c/phasic_hash.c',
            '-o', lib_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    finally:
        os.unlink(wrapper_file)

    return lib_path


def _setup_ctypes_signatures(lib, has_pmf=True, has_dph=True):
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


def _setup_ctypes_signatures_from_arrays(lib, discrete=False):
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


def _callback(ipv):
    """
    Turn callback functions with different signatures into a common one.
    Also makes return the ipv when called with empty state.
    """
    if all(isinstance(x, int) for x in ipv):
        ipv = [[ipv, 1.0]]

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
                    if abs(sum(prob) - 1.0) > 1e-12:
                        raise ValueError("IPV does not sum to one", ipv)
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
            # Handle 3-tuples: (state, base_weight, edge_state) for parameterized edges
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


def _invalidates_trace(method):
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


class Graph(_Graph):
    # def __init__(self, state_length:int=None, callback:Callable=None, ipv:List[Union[List[int], List[Union[List[int], float]]]] = None, parameterized:bool=False, **kwargs):
    def __init__(self, arg:Union[int, Callable], ipv:Optional[Union[List[int], List[Union[List[int], float]]]]=None, cache_graph:bool=False, **kwargs):
        """
        Create a graph representing a phase-type distribution. This is the primary entry-point of the library. A starting vertex will always be added to the graph upon initialization.

        The graph can be initialized in two ways:
        - By providing a state length to create an empty graph.
        - By providing a callback function that generates the graph. The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate. For parameterized edges, the callback should return 3-tuples (state, weight, edge_state). If the ipv argument is not provided, the function must return the ipv if given an empty state array as argument.

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
        # Extract cache_trace flag (with hierarchical as deprecated alias)
        cache_trace = kwargs.get('cache_trace', None)
        hierarchical = kwargs.get('hierarchical', None)
        if cache_trace is None and hierarchical is not None:
            import warnings
            warnings.warn(
                "The 'hierarchical' parameter is deprecated. Use 'cache_trace' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            cache_trace = hierarchical
        elif cache_trace is None:
            cache_trace = False

        self._joint_prob_base_graph_indexer = None  # flag to signify joint probability representation; defaults to until set internally

        # Wrap callback with IPV BEFORE cache operations to ensure consistent hashing
        callback_for_cache = arg
        if callable(arg) and ipv is not None:
            if arg.__name__ != 'wrapper':
                # Wrap with IPV now so cache hash includes it
                callback_for_cache = _callback(ipv)(arg)

        # Try loading from cache if requested
        if callable(arg) and cache_graph:
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
                assert ipv is not None, "When providing a function not decorated with @callback, the ipv argument must be provided"
                arg = _callback(ipv)(arg)
            else:
                assert ipv is None, "When providing a function decorated with @callback, the ipv argument is ignored and should not be provided"

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

        # Save to cache if requested and construction succeeded
        if callable(arg) and cache_graph:
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

    @_invalidates_trace
    def find_or_create_vertex(self, state):
        """Find or create a vertex with the given state.

        This method wraps the C++ implementation to track trace invalidation.
        """
        return super().find_or_create_vertex(state)

    @_invalidates_trace
    def extend(self, callback=None, **kwargs):
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
            - For parameterized: list of (state, base_weight, edge_state) tuples
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
        >>> graph.starting_vertex().add_edge_parameterized(special_vertex, 0.0, [1.5])
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
        super().extend_graph_callback_tuples_parameterized(callback_with_kwargs)

    def _ensure_trace(self):
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

    def update_weights(self, theta, callback=None, log=False):
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
        This does NOT invalidate the cached trace since it only changes
        parameter values, not graph structure.
        """
        self._last_theta = np.asarray(theta)
        if callback is not None:
            # C++ overload: update_weights(params, callback) - no log parameter
            return super().update_weights(theta, callback)
        else:
            # C++ overload: update_weights(params, log=False)
            return super().update_weights(theta, log=log)

    def _moments_from_trace(self, power: int = 1, rewards=None):
        """Compute moments using cached elimination trace.

        This instantiates a concrete graph from the trace with current
        parameters and computes moments on that graph.

        Parameters
        ----------
        power : int, default=1
            Moment power (1 for expectation, 2 for variance, etc.)
        rewards : ArrayLike, optional
            Reward vector for reward-transformed moments.

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
            return concrete_graph.moments(power, list(rewards))
        else:
            return concrete_graph.moments(power)

    def _expectation_from_trace(self, rewards=None, discrete=False, **kwargs):
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

    def _variance_from_trace(self, rewards=None, discrete=False, **kwargs):
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

    def expected_waiting_time(self, *args, **kwargs):
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

    
    def expected_sojourn_time(self, *args, **kwargs):
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




    def moments(self, power, rewards=[], discrete=False, **kwargs):
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
        # For parameterized graphs, always use trace-based computation (O(n) memory)
        # to avoid O(n²) matrix allocation in the C++ fallback path
        if self.parameterized():
            trace = self._ensure_trace()
            if trace is not None:
                return self._moments_from_trace(power, rewards=rewards, discrete=discrete, **kwargs)

        # Fall back to direct C++ computation for non-parameterized graphs
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().moments_discrete(power, rewards=rewards, **kwargs)
        else:
            return super().moments(power, rewards=rewards, **kwargs)

    def expectation(self, rewards=[], discrete=False, **kwargs):
        """
        Compute expected value (first moment) of the phase-type distribution.

        Parameters
        ----------
        rewards : ArrayLike, optional
            Reward vector for reward-transformed expectation. If not provided,
            computes E[T] where T is time until absorption.
        discrete : bool, default=False
            If True, compute discrete-time expectation (DPH distribution).
            Requires that the graph was discretized via discretize().
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
        # For parameterized graphs, always use trace-based computation (O(n) memory)
        # to avoid O(n²) matrix allocation in the C++ fallback path
        if self.parameterized():
            trace = self._ensure_trace()
            if trace is not None:
                return self._expectation_from_trace(rewards=rewards, discrete=discrete, **kwargs)

        # Fall back to direct C++ computation for non-parameterized graphs
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().expectation_discrete(rewards=rewards, **kwargs)
        else:
            return super().expectation(rewards=rewards, **kwargs)

    def variance(self, rewards=[], discrete=False, **kwargs):
        """
        Compute variance of the phase-type distribution.

        Parameters
        ----------
        rewards : ArrayLike, optional
            Reward vector for reward-transformed variance. If not provided,
            computes Var(T) where T is time until absorption.
        discrete : bool, default=False
            If True, compute discrete-time variance (DPH distribution).
            Requires that the graph was discretized via discretize().
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
        # For parameterized graphs, always use trace-based computation (O(n) memory)
        # to avoid O(n²) matrix allocation in the C++ fallback path
        if self.parameterized():
            trace = self._ensure_trace()
            if trace is not None:
                return self._variance_from_trace(rewards=rewards, discrete=discrete, **kwargs)

        # Fall back to direct C++ computation for non-parameterized graphs
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().variance_discrete(rewards=rewards, **kwargs)
        else:
            return super().variance(rewards=rewards, **kwargs)

    def covariance(self, *args, discrete=False, **kwargs):
        """
        Compute covariance matrix for multivariate phase-type distributions.

        Parameters
        ----------
        discrete : bool, default=False
            If True, compute discrete-time covariance (DPH distribution).
            Requires that the graph was discretized via discretize().
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Covariance matrix for the multivariate distribution.

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        This method is for multivariate phase-type distributions with
        multiple marginals. For univariate distributions, use variance().
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().covariance_discrete(*args, **kwargs)
        else:
            return super().covariance(*args, **kwargs)

    def pdf(self, time, discrete=False, **kwargs):
        """
        Compute probability density/mass function using forward algorithm.

        Parameters
        ----------
        time : float or ArrayLike
            Time point(s) at which to evaluate the PDF/PMF.
        granularity : int, optional
            Granularity for uniformization (default: auto-detected as 2*max_rate).
            Higher values improve accuracy but increase computation time.
        discrete : bool, default=False
            If True, compute PMF for discrete-time distribution (DPH).
            Requires that the graph was discretized via discretize().
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float or np.ndarray
            PDF/PMF value(s) at the specified time point(s).

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        This method uses the forward algorithm (Algorithm 4) via uniformization
        to compute the exact phase-type PDF/PMF, not an approximation.

        For continuous distributions: f(t) = α · exp(S·t) · s*
        For discrete distributions: p(n) = probability of absorption at jump n
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().pdf_discrete(time, **kwargs)
        else:
            return super().pdf(time, **kwargs)

    def cdf(self, time, discrete=False, **kwargs):
        """
        Compute cumulative distribution function.

        Parameters
        ----------
        time : float or ArrayLike
            Time point(s) at which to evaluate the CDF.
        discrete : bool, default=False
            If True, compute CDF for discrete-time distribution (DPH).
            Requires that the graph was discretized via discretize().
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        float or np.ndarray
            CDF value(s) P(T ≤ t) at the specified time point(s).

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        For continuous distributions: F(t) = P(T ≤ t) = 1 - α · exp(S·t) · 1
        For discrete distributions: F(n) = P(N ≤ n) = sum of PMF up to n
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().cdf_discrete(time, **kwargs)
        else:
            return super().cdf(time, **kwargs)

    def distribution_context(self, *args, discrete=False, **kwargs):
        """
        Create a distribution context for efficient repeated sampling.

        Parameters
        ----------
        discrete : bool, default=False
            If True, create context for discrete-time distribution (DPH).
            Requires that the graph was discretized via discretize().
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        DistributionContext
            Context object that can be used for efficient sampling.

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        The distribution context precomputes data structures needed for
        sampling, making repeated sample() calls much faster than sampling
        directly from the graph.
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().distribution_context_discrete(*args, **kwargs)
        else:
            return super().distribution_context(*args, **kwargs)

    def sample(self, n, discrete=False, **kwargs):
        """
        Generate random samples from the phase-type distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        discrete : bool, default=False
            If True, sample from discrete-time distribution (DPH).
            Requires that the graph was discretized via discretize().
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Array of n samples from the distribution.

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        Sampling is done by simulating the underlying Markov chain until
        absorption. For more efficient repeated sampling, first create a
        distribution context using distribution_context().
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().sample_discrete(n, **kwargs)
        else:
            return super().sample(n, **kwargs)

    def stop_probability(self, time, discrete=False, **kwargs):
        """
        Compute probability of being in each state at a given time.

        Parameters
        ----------
        time : float or int
            Time point (continuous) or jump number (discrete) at which to
            evaluate state probabilities.
        discrete : bool, default=False
            If True, compute for discrete-time distribution (DPH).
            Requires that the graph was discretized via discretize().
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Probability of being in each state at the specified time.

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        For continuous distributions: probability vector at time t
        For discrete distributions: probability vector after n jumps
        Computed via matrix exponentiation or uniformization.
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return super().stop_probability_discrete(time, **kwargs)
        else:
            return super().stop_probability(time, **kwargs)

    # Alias for stop_probability
    state_probability = stop_probability


    def accumulated_visits(self, *args, **kwargs):
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

    def accumulated_visiting_time(self, *args, **kwargs):
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

    def accumulated_occupancy(self, *args, discrete=False, **kwargs):
        """
        Compute expected occupancy (visits or time) for each state.

        Parameters
        ----------
        discrete : bool, default=False
            If True, compute accumulated visits (discrete distribution).
            If False, compute accumulated visiting time (continuous distribution).
        *args : tuple
            Additional positional arguments passed to C++ implementation.
        **kwargs : dict
            Additional keyword arguments passed to C++ implementation.

        Returns
        -------
        np.ndarray
            Expected visits (discrete) or time (continuous) in each state.

        Raises
        ------
        ValueError
            If discrete=True but graph is not discrete.

        Notes
        -----
        This is a convenience method that dispatches to either:
        - accumulated_visits() for discrete=True
        - accumulated_visiting_time() for discrete=False
        """
        if discrete:
            if not self.is_discrete:
                raise ValueError("discrete=True only valid for discrete distributions")
            return self.accumulated_visits(*args, **kwargs)
        else:
            return self.accumulated_visiting_time(*args, **kwargs)

    @_invalidates_trace
    def normalize(self, *args, **kwargs):
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
    def discretize(self, rate, skip_existing=False, **kwargs) -> NDArray[np.int64]:
        """
        Discretizes graph inplace and returns reward matrix for added auxiliary states.

        Parameters
        ----------
        rate : 
            float or callable
        skip_existing : bool, optional
            If True, skip vertices that already have auxiliary vertices, by default False

        Returns
        -------
        :
            Reward matrix for added auxiliary states 
        """

        # if not callable(rate):
        #     def rate_fn(state):
        #         rate = rate
        #         return rate
        #     rate = rate_fn

        # new_graph = self.copy()
        vlength = self.vertices_length()

        aux_indices = []

        for vertex in self.vertices():
            if vertex.index() == self.starting_vertex().index() or not vertex.edges():
                # skip starting and absorbing nodes
                continue

            if skip_existing:
                has_aux, is_aux = False, False
                for edge in vertex.edges():
                    if edge.to().state().sum() == 0 and edge.to().edges_length() and edge.to().edges()[0].to().index() == vertex.index():
                        has_aux = True
                        aux_indices.append(edge.to().index())
                        vlength -= 1 # to not count vertex in both aux_indices and vlength
                        break
                if vertex.state().sum() == 0:
                    is_aux = True
                if has_aux or is_aux:
                    continue

            _rate = rate(vertex.state(), **kwargs) if callable(rate) else rate
            aux_vertex = vertex.add_aux_vertex(_rate)
                
            aux_indices.append(aux_vertex.index())

        rewards = np.zeros(vlength+len(aux_indices), dtype=int)
        for index in aux_indices:
            rewards[index] = 1

        weight_scaling = self.normalize()

        self.is_discrete = True
        self.set_was_dph(True)  # Enable auto-normalization in C update_weights()

        return rewards
    

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
        reward_transform : General reward transformation (dispatches to this for discrete=True)
        """
        return Graph(super().reward_transform_discrete(rewards))
    

    def serialize(self, theta_dim: int = None) -> Dict[str, np.ndarray]:
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
            'n_vertices': n_vertices
        }

    @classmethod
    def from_serialized(cls, data: Dict[str, Any]) -> 'Graph':
        """
        Reconstruct Graph from serialize() output.

        This method enables distributed trace recording by allowing graphs
        to be serialized to JSON, sent across the network via JAX pmap,
        and reconstructed on worker processes.

        Parameters
        ----------
        data : Dict[str, Any]
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

        # Note: As of v0.22.0, base_weight was removed from parameterized edges
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
                    f"  Format: [to_idx, coeff1, coeff2, ...] (no base_weight as of v0.22.0)"
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
                # As of v0.22.0, base_weight=0.0 for all parameterized edges
                from_vertex.add_edge_parameterized(to_vertex, 0.0, edge_state)
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
                # As of v0.22.0, base_weight=0.0 for all parameterized edges
                start.add_edge_parameterized(to_vertex, 0.0, edge_state)
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
    def from_matrices(cls, ipv: np.ndarray, sim: np.ndarray, states: Optional[np.ndarray] = None) -> Self:
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
    def pmf_from_graph(cls, graph: 'Graph', discrete: bool = False, use_cache: bool = True, theta_dim: int = None) -> Callable:
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
        detected_param_length = serialized.get('param_length', 0)
        has_param_edges = detected_param_length > 0

        # Generate C++ build_model() code from the serialized graph
        cpp_code = _generate_cpp_from_graph(serialized)

        # Create hash of the generated C++ code
        cpp_hash = hashlib.sha256(cpp_code.encode()).hexdigest()[:16]
        temp_file = f"/tmp/graph_model_{cpp_hash}.cpp"

        # Write C++ code to temp file
        with open(temp_file, 'w') as f:
            f.write(cpp_code)

        # Return appropriate signature based on parameterization
        if has_param_edges:
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

            if use_ffi:
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
            else:
                # pure_callback (single-core, no FFI)
                from . import phasic_pybind as cpp_module

                # Create GraphBuilder ONCE - captured in model closure
                builder = cpp_module.parameterized.GraphBuilder(structure_json_str)

                def _compute_pdf_cached(theta_np, times_np):
                    """Uses cached builder - NO JSON parsing per call."""
                    # Check if theta is batched (from vmap with expand_dims)
                    if theta_np.ndim == 2:
                        times_unbatched = times_np[0] if times_np.ndim == 2 else times_np
                        results = []
                        for theta_single in theta_np:
                            result = builder.compute_pmf(
                                theta_single,
                                times_unbatched,
                                discrete=discrete,
                                granularity=0
                            )
                            results.append(result)
                        return np.array(results)
                    else:
                        return builder.compute_pmf(
                            theta_np,
                            times_np,
                            discrete=discrete,
                            granularity=0
                        )

                def model_pure(theta, times):
                    """Pure callback wrapper (fallback when FFI disabled)."""
                    result_shape = jax.ShapeDtypeStruct(times.shape, times.dtype)
                    return jax.pure_callback(
                        lambda t, tm: _compute_pdf_cached(
                            np.asarray(t, dtype=np.float64),
                            np.asarray(tm, dtype=np.float64)
                        ).astype(times.dtype),
                        result_shape,
                        theta,
                        times,
                        vmap_method='expand_dims'
                    )

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
    def pmf_from_graph_parameterized(cls, graph_builder: Callable, discrete: bool = False) -> Callable:
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
    def pmf_from_cpp(cls, cpp_file: Union[str, pathlib.Path], discrete: bool = False) -> Callable:
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
        cpp_file : str or Path
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
        # JAX-compatible approach (default - for SVGD, gradients, optimization)
        >>> model = Graph.pmf_from_cpp("my_model.cpp")
        >>> theta = jnp.array([1.0, 2.0])
        >>> times = jnp.linspace(0, 10, 100)
        >>> pmf = model(theta, times)
        >>> gradient = jax.grad(lambda p: jnp.sum(model(p, times)))(theta)

        # Discrete phase-type distribution
        >>> model = Graph.pmf_from_cpp("my_model.cpp", discrete=True)
        >>> theta = jnp.array([1.0, 2.0])
        >>> jumps = jnp.array([1, 2, 3, 4, 5])
        >>> dph_pmf = model(theta, jumps)

        # For direct C++ access without JAX (faster for repeated evaluations):
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


    def svgd(self,
             observed_data: ArrayLike,
             discrete: Optional[bool] = None,
             prior: Optional[Callable] = None,
             n_particles: Optional[int] = None,
             n_iterations: int = 100,
             optimizer: Optional[object] = None,
             learning_rate: Optional[float] = None,
             bandwidth = 'median_per_dim',
             theta_init: Optional[ArrayLike] = None,
             theta_dim: Optional[int] = None,
             return_history: bool = True,
             seed: Optional[int] = None,
             verbose: bool = False,
             progress: bool = True,
             jit: Optional[bool] = None,
             parallel: Optional[str] = None,
             n_devices: Optional[int] = None,
             precompile: bool = True,
             compilation_config: Optional[object] = None,
             regularization: float = 0.0,
             nr_moments: int = 2,
             positive_params: bool = True,
             param_transform: Optional[Callable] = None,
             joint_index: bool = False,
             rewards: Optional[ArrayLike] = None,
             fixed: Optional[ArrayLike] = None,
             preconditioner = 'auto',
             ) -> Dict:
        """
        Run Stein Variational Gradient Descent (SVGD) inference for Bayesian parameter estimation.

        SVGD finds the posterior distribution p(theta | data) by optimizing a set of particles to
        approximate the posterior. This method works with parameterized models created by
        pmf_from_graph() or pmf_from_cpp() where the model signature is model(theta, times).

        Parameters
        ----------
        observed_data : ArrayLike
            Observed data points. For continuous models (PDF), these are time points where
            the density was observed. For discrete models (PMF), these are jump counts.
        discrete : bool, default=None
            If True, computes discrete PMF. If False, computes continuous PDF. If undefined it is 
            inferred from the graph.is_discrete attribute.
        prior : callable or list of Prior objects, optional
            Log prior function for parameters. Can be:
            - Single callable: prior(theta) -> scalar, applied to entire theta vector
            - List of Prior objects: One prior per parameter dimension.
              Use None for fixed parameters: prior=[GaussPrior(ci=[0,1]), None, GaussPrior(ci=[0,1])]

            If None, uses standard normal prior: log p(theta) = -0.5 * sum(theta^2)

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
        optimizer : Optimizer, optional
            Learning rate optimizer instance from phasic.optimizers. Default is Adam
            when learning_rate=None and regularization=0. Options include Adamelia, Adam,
            SGDMomentum, RMSprop, Adagrad. When an optimizer is used, the learning_rate
            parameter is ignored (the optimizer has its own learning rate).
        learning_rate : float or None, default=None
            SVGD step size. If None (default), uses Adame optimizer with adaptive
            learning rates. If a float is provided, uses fixed learning rate approach.
            Larger values = faster convergence but may be unstable.
        bandwidth : str, float, or array_like, default='median_per_dim'
            Kernel bandwidth selection method:
            - 'median_per_dim': Per-dimension median heuristic (default). Uses a
              separate bandwidth per parameter dimension for an anisotropic kernel.
            - 'median': Scalar median heuristic (isotropic kernel)
            - float: Fixed scalar bandwidth value
            - array_like: Fixed per-dimension bandwidth vector
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
        compilation_config : CompilationConfig, dict, str, or Path, optional
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
            model = Graph.pmf_from_graph_joint_index(self, theta_dim=theta_dim,
                                                      fixed_mask=fixed_mask_for_model)
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

    @classmethod
    def moments_from_graph(cls, graph: 'Graph', nr_moments: int = 2, use_ffi: bool = False) -> Callable:
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
    def pmf_and_moments_from_graph(cls, graph: 'Graph', nr_moments: int = 2,
                                   discrete: bool = False, use_ffi: bool = False,
                                   theta_dim: int = None) -> Callable:
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

        # Check if FFI is available - respect parameter, allow config override
        config = get_config()
        if not use_ffi:  # If explicitly disabled, respect it
            use_ffi = False
        else:  # If True or default, check config
            use_ffi = config.ffi  # Enable FFI for multi-core parallelization (C++ binding fixed!)

        if use_ffi:
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
        else:
            # Use pybind11 GraphBuilder (same as pmf_from_graph)
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
            theta_bar = []
            for i in range(n_params):
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].add(-eps)

                # Call underlying computation, not model
                pmf_plus, moments_plus = _compute_pure(theta_plus, times, rewards)
                pmf_minus, moments_minus = _compute_pure(theta_minus, times, rewards)

                # Combine gradients from both PMF and moments
                grad_pmf_i = jnp.sum(g_pmf * (pmf_plus - pmf_minus) / (2 * eps))
                grad_moments_i = jnp.sum(g_moments * (moments_plus - moments_minus) / (2 * eps))
                grad_i = grad_pmf_i + grad_moments_i

                theta_bar.append(grad_i)

            return jnp.array(theta_bar), None, None  # gradients for theta, times, rewards

        model.defvjp(model_fwd, model_bwd)
        return model

    @classmethod
    def pmf_from_graph_joint_index(cls, graph: 'Graph', theta_dim: int = None,
                                    fixed_mask: 'jnp.ndarray' = None) -> Callable:
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
        all_terminal_indices = jnp.array(sorted(set(all_terminal_indices)), dtype=jnp.int32)

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
    def pmf_and_moments_from_graph_multivariate(cls, graph: 'Graph', nr_moments: int = 2,
                                                discrete: bool = False, use_ffi: bool = False,
                                                theta_dim: int = None) -> Callable:
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
            """Multivariate wrapper handling 1D and 2D rewards"""

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
                times_arr = jnp.asarray(times)

                pmf_list = []
                moments_list = []

                for j in range(n_features):
                    # Extract reward vector for feature j (ensure float64 for C++ compatibility)
                    reward_j = rewards_arr[j, :].astype(jnp.float64)

                    # Extract times for feature j (support both 1D and 2D times)
                    if times_arr.ndim == 2:
                        times_j = times_arr[:, j]
                    else:
                        times_j = times_arr  # Broadcast same times to all features

                    # Compute PMF and moments for this feature
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

    def plot(self, *args, **kwargs):
        """
        Plots the graph using graphviz. See plot::plot_graph.py for more details.

        Returns
        -------
        :
            _description_
        """
        return plot.plot_graph(self, *args, **kwargs)

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

    def clone(self):
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

    def compute_trace(self, param_length: Optional[int] = None,
                     hierarchical: bool = True,
                     min_size: int = 50,
                     parallel: str = 'auto',
                     verbose: bool = False,
                     force: bool = False):
        """
        Compute elimination trace with optional hierarchical caching.

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

    def eliminate_to_dag(self) -> 'SymbolicDAG':
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
                           state, 
                        indexer,
                        reward_indexer,
                        current_rewards=None, 
                        mutation_rate=1.0, 
                        reward_limit=10, 
                        tot_reward_limit=np.inf):

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

            for i in range(indexer.state_length):
                props = indexer.index_to_props(i, as_dict=True)
                for prop, value in getattr(props, prop_set_name).items():
                    rate = state[i] * mutation_rate 
                    reward_idx = reward_indexer.props_to_index(**{f'{prop_set_name}_{prop}': value})

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
                        base_graph_indexer:StateIndexer,
                        reward_rates_callback:Optional[Callable]=None,
                        mutation_rate:Optional[float]=1.0,
                        reward_limit:Optional[int]=None, 
                        tot_reward_limit:Optional[float]=np.inf) -> Graph:

        if self.param_length() == 0:
            raise ValueError("Graph must have parameterized edges for joint_prob_graph.")

        if reward_rates_callback is None:
            # default to joint prob reward callback
            reward_rates_callback = self._joint_prob_reward

        if reward_limit is None and tot_reward_limit == np.inf:        
            raise ValueError("Either reward_limit or tot_reward_limit must be specified.")

        base_starting_vertex = self.starting_vertex()

        if len(base_graph_indexer.property_sets()) != 1:
            raise ValueError("Indexer must have exactly one property set representing the base graph state.")

        # indexer for rewards (each property gets its own property set)
        reward_prop_sets = []
        property_set = base_graph_indexer.property_sets()[0]
        for p in property_set.properties:
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

        # append reward indexer to original indexer
        joint_graph_indexer = base_graph_indexer + reward_indexer

        # joint graph state vector length
        state_vector_length = joint_graph_indexer.state_length

        reward_length = reward_indexer.state_length

        # indices for original and new parts of the state vector
        state_indices = base_graph_indexer.indices()
        reward_state_indices = np.arange(base_graph_indexer.state_length, joint_graph_indexer.state_length)

        # create the new graph
        joint_graph = Graph(state_vector_length)
        starting_vertex = joint_graph.starting_vertex()

        # array of zeros for extended state vector
        null_rewards = np.zeros(reward_length)

        # graph index of last vertex visited
        index = 0

        # Get param_length for extracting parameterized edge coefficients
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

            # add edges and children of base vertex
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
                base_graph_indexer, reward_indexer,
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

        # the t-states represent variants of the original absorbing state
        # add a new absorbing with edges from all t-states
        new_absorbing = joint_graph.create_vertex(np.repeat(0, state_vector_length))
        t_vertex_indices = np.unique(t_vertex_indices)
        for i in t_vertex_indices:
            joint_graph.vertex_at(i).add_edge(new_absorbing, np.append(np.zeros(self.param_length()), 1.0))

        # set discrete flag for update_weights to also normalize and for
        # expected_sojourn_time to call its discrete version
        joint_graph.is_discrete = True
        joint_graph.set_was_dph(True)  # Enable auto-normalization in C update_weights()

        joint_graph._joint_prob_base_graph_indexer = base_graph_indexer

        return joint_graph


    def _get_joint_probs(self):

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
            # print(rewards, idx)

            # if base_graph_indexer is not None:
            #     properties = [p.name for p in base_graph_indexer.properties]
            #     prop_a, prop_b = properties
            #     obs = np.zeros((base_graph_indexer.property_dict[prop_a].max_value+1,
            #                     base_graph_indexer.property_dict[prop_b].max_value+1), dtype=int)
            #     for i in range(rewards.size):
            #         if rewards[i] > 0:
            #             props = base_graph_indexer.index_to_props(i, as_dict=True)
            #             obs[props[prop_a], props[prop_b]] += rewards[i]
            #     obs = tuple([tuple(row.tolist()) for row in obs])
            #     idx2obs[obs] = int(idx)

            # else:
            #     idx2obs[int(idx)] = tuple(rewards.tolist())

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


    def joint_prob_table(self):

        if not self._joint_prob_base_graph_indexer:
            raise ValueError("Graph is not a joint probability representation.")

        outcomes, probs, t_vertex_indices = self._get_joint_probs()

        records = [[*obs, prob, idx] for obs, prob, idx in zip(outcomes, probs, t_vertex_indices)]
        column_names = []
        prop_sets = self._joint_prob_base_graph_indexer.property_sets()
        if len(prop_sets) != 1:
            raise ValueError("Indexer must have exactly one property set representing the base graph state.")
        for p in prop_sets[0].properties:
            for i in range(p.min_value, p.max_value+1):
                column_names.append(f"{p.name}_{i}")
        column_names.extend(['prob', 't_vertex_index'])
        joint = pd.DataFrame(records, columns=column_names).set_index('t_vertex_index')
        return joint


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

    def __init__(self, ptr: int):
        """Initialize from opaque pointer returned by Graph._eliminate_to_dag_internal()"""
        self._ptr = ptr
        self._info = None

    def instantiate(self, params: ArrayLike) -> 'Graph':
        """
        Evaluate expression trees with concrete parameters to create a Graph.

        This is an O(n) operation that evaluates all symbolic expressions
        with the given parameter vector. Much faster than O(n³) graph
        reconstruction!

        Parameters
        ----------
        params : array-like
            Parameter vector, shape (n_params,)

        Returns
        -------
        Graph
            Graph with concrete edge weights evaluated from expressions
        """
        from .phasic_pybind import _symbolic_dag_instantiate
        params_arr = np.asarray(params, dtype=np.float64)
        return _symbolic_dag_instantiate(self._ptr, params_arr)

    @property
    def info(self) -> Dict[str, Any]:
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

    def __del__(self):
        """Free C memory when Python object is garbage collected"""
        if hasattr(self, '_ptr') and self._ptr != 0:
            from .phasic_pybind import _symbolic_dag_destroy
            _symbolic_dag_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self):
        return (f"SymbolicDAG(vertices={self.vertices_length}, "
                f"params={self.param_length}, acyclic={self.is_acyclic})")


# Module-level utility functions

def load_cpp_builder(cpp_file: Union[str, pathlib.Path]) -> Callable:
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

def init_parallel(cpus: Optional[int] = None,
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
