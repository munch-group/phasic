from functools import partial
from collections import defaultdict
from unittest import result
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, TypeVar, List, Tuple, Dict, Union, NamedTuple, Optional
from collections.abc import Sequence, MutableSequence, Callable
import os
import hashlib
import subprocess
import tempfile
import ctypes
import pathlib

from functools import wraps
from distro import name
import numpy as np
from collections import OrderedDict, UserDict

class StateDict(UserDict):

    def __init__(self, data):
        self.data = OrderedDict(data)
        self.list = list(self.data.values())

    def __get__(self, key):
        if type(key) is int:
            return self.list[key]
        return self.data[key]

    def __set__(self, key, value):
        if type(key) is int:
            self.list[key] = value
        self.data[key] = value     

def labelled(labels):  # The factory function that accepts a parameter
    def decorator(func):
        @wraps(func)  # Apply @wraps to the wrapper
        def wrapper(arr, **kwargs):
            print(kwargs)
#            assert len(labels) == arr.size
            l = list(zip(labels, arr))
            d = StateDict(l)
            result = func(d, **kwargs)  
            print(result)
            if not result:
                return []
            return [[np.array(state, dtype=int), *rest] for state, *rest in result]
        return wrapper
    return decorator

# # state vector labels 
# labels = ['foo', 'bar', 'baz']

# @labeled(labels)
# def callback(state):

#     new_state = state.copy()
#     new_state['foo'] += 1

#     return [(new_state, 1)]

# state = np.array([1, 2, 3])
# callback(state)

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

from . import plot
from .plot import set_theme

# Optional SVGD support (requires JAX)
if HAS_JAX:
    from .svgd import (
        SVGD,
        # Step size schedules
        StepSizeSchedule,
        ConstantStepSize,
        ExponentialDecayStepSize,
        AdaptiveStepSize,
        # Regularization schedules
        RegularizationSchedule,
        ConstantRegularization,
        ExponentialDecayRegularization,
        ExponentialCDFRegularization,
        # # Bandwidth schedules
        # BandwidthSchedule,
        # MedianBandwidth,
        # FixedBandwidth,
        # LocalAdaptiveBandwidth
    )
else:
    SVGD = None
    StepSizeSchedule = None
    ConstantStepSize = None
    ExponentialDecayStepSize = None
    AdaptiveStepSize = None
    RegularizationSchedule = None
    ConstantRegularization = None
    ExponentialDecayRegularization = None
    ExponentialCDFRegularization = None
    # BandwidthSchedule = None
    # MedianBandwidth = None
    # FixedBandwidth = None
    # LocalAdaptiveBandwidth = None

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

# CPU monitoring
from .cpu_monitor import (
    CPUMonitor,
    monitor_cpu,
    CPUMonitorMagics,
    detect_compute_nodes,
    get_cached_nodes,
)

# Cache management (JAX compilation cache)
from .cache_manager import CacheManager, print_jax_cache_info, configure_layered_cache
from .model_export import clear_cache, cache_info, print_cache_info
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
    IPFSBackend,
    TraceRegistry,
    get_trace,
    install_trace_library
)

# Hash-based trace lookup (convenience wrapper)
def get_trace_by_hash(graph_hash: str, force_download: bool = False):
    """
    Get elimination trace by graph structure hash.

    Convenience wrapper around TraceRegistry.get_trace_by_hash().

    Parameters
    ----------
    graph_hash : str
        SHA-256 hash of graph structure (from phasic.hash.compute_graph_hash)
    force_download : bool, default=False
        If True, re-download even if cached

    Returns
    -------
    EliminationTrace or None
        Trace if found, None otherwise

    Examples
    --------
    >>> import phasic
    >>> import phasic.hash
    >>> graph = phasic.Graph(callback=my_callback, parameterized=True, nr_samples=5)
    >>> hash_result = phasic.hash.compute_graph_hash(graph)
    >>> trace = phasic.get_trace_by_hash(hash_result.hash_hex)
    >>> if trace is None:
    ...     # Record new trace
    ...     from phasic.trace_elimination import record_elimination_trace
    ...     trace = record_elimination_trace(graph, param_length=1)
    """
    registry = TraceRegistry()
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


# class MatrixRepresentation(NamedTuple):
#     """
#     Matrix representation of a phase-type distribution.

#     Attributes
#     ----------
#     states : np.ndarray
#         State vectors for each vertex, shape (n_states, state_dim), dtype=int32
#     sim : np.ndarray
#         Sub-intensity matrix, shape (n_states, n_states), dtype=float64
#     ipv : np.ndarray
#         Initial probability vector, shape (n_states,), dtype=float64
#     indices : np.ndarray
#         1-based indices for vertices (for use with vertex_at()), shape (n_states,), dtype=int32
#     """
#     states: np.ndarray
#     sim: np.ndarray
#     ipv: np.ndarray
#     indices: np.ndarray 

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
            edge_state = edge[1:]
            # Generate weight computation: w = x1*theta[0] + x2*theta[1] + ...
            # Only include non-zero coefficients for efficiency and correctness
            weight_terms = [f"{edge_state[j]}*theta[{j}]"
                           for j in range(len(edge_state))
                           if edge_state[j] != 0.0]
            if not weight_terms:
                # All coefficients are zero - use constant 0
                weight_terms = ["0.0"]
            weight_expr = " + ".join(weight_terms)
            param_edge_code.append(f"    double w_start_{to_idx} = {weight_expr};")
            param_edge_code.append(f"    start->add_edge(*vertices[{to_idx}], w_start_{to_idx});")

        # Regular vertex parameterized edges
        for i, edge in enumerate(param_edges):
            from_idx = int(edge[0])
            to_idx = int(edge[1])
            edge_state = edge[2:]
            # Generate weight computation - only include non-zero coefficients
            weight_terms = [f"{edge_state[j]}*theta[{j}]"
                           for j in range(len(edge_state))
                           if edge_state[j] != 0.0]
            if not weight_terms:
                # All coefficients are zero - use constant 0
                weight_terms = ["0.0"]
            weight_expr = " + ".join(weight_terms)
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
    observed_data : array_like
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
    >>> trace = record_elimination_trace(graph, param_length=2)
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
    >>> log_lik = _wrap_trace_log_likelihood_for_jax(lib_path, param_length=2)
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


# class Graph(_Graph):
#     def __init__(self, state_length=None, callback=None, initial=None, trans_as_dict=False):
#         """
#         Create a graph representing a phase-type distribution. This is the primary entry-point of the library. A starting vertex will always be added to the graph upon initialization.

#         The graph can be initialized in two ways:
#         - By providing a callback function that generates the graph. The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.
#         - By providing an initial state and a list of transitions. The initial state is a list of integers representing the initial model state. The list of transitions is a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.

#         Parameters
#         ----------
#         state_length : 
#             The length of the integer vector used to represent and reference a state, by default None
#         callback : 
#             Callback function accepting a state and returns a list of reachable states and the corresponding transition rates, by default None.
#             The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.
#         initial : 
#             A list of integers representing the initial model state, by default None
#         trans_as_dict : 
#             Whether the callback should return dictionaries with 'state' and 'weight' keys instead of tuples, by default False

#         Returns
#         -------
#         :
#             A graph object representing a phase-type distribution.
#         """

#         assert bool(callback) == bool(initial), "callback and initial_state must be both None or both not None"

#         if callback and initial:
#             if trans_as_dict:        
#                 super().__init__(callback_dicts=callback, initial_state=initial)
#             else:
#                 super().__init__(callback_tuples=callback, initial_state=initial)
#         else:
#             super().__init__(state_length)

class Graph(_Graph):
    def __init__(self, state_length:int=None, callback:Callable=None, parameterized:bool=False, **kwargs):
        """
        Create a graph representing a phase-type distribution. This is the primary entry-point of the library. A starting vertex will always be added to the graph upon initialization.

        The graph can be initialized in two ways:
        - By providing a callback function that generates the graph. The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.
        - By providing an initial state and a list of transitions. The initial state is a list of integers representing the initial model state. The list of transitions is a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.

        Parameters
        ----------
        state_length :
            The length of the integer vector used to represent and reference a state, by default None
        callback :
            Callback function accepting a state and returns a list of reachable states and the corresponding transition rates, by default None.
            The callback function should take a list of integers as its only argument and return a list of tuples, where each tuple contains a state and a list of tuples, where each tuple contains a state and a rate.
        parameterized :
            If True, the callback returns 3-tuples (state, weight, edge_state) for parameterized edges, by default False

        Returns
        -------
        :
            A graph object representing a phase-type distribution.
        """
        assert (callback is None) + (state_length is None) == 1, "Use either the state_length or callback argument"

        if callback:
            if parameterized:
                super().__init__(callback_tuples_parameterized=partial(callback, **kwargs))
            else:
                super().__init__(callback_tuples=partial(callback, **kwargs))
        else:
            super().__init__(state_length)


    # def make_discrete(self, mutation_rate, skip_states=[], skip_slots=[]):
    #     """
    #     Takes a graph for a continuous distribution and turns
    #     it into a descrete one (inplace). Returns a matrix of
    #     rewards for computing marginal moments
    #     """

    #     mutation_graph = self.copy()

    #     # save current nr of states in graph
    #     vlength = mutation_graph.vertices_length()

    #     # number of fields in state vector (assumes all are the same length)
    #     state_vector_length = len(mutation_graph.vertex_at(1).state())

    #     # list state vector fields to reward at each auxiliary node
    #     # rewarded_state_vector_indexes = [[] for _ in range(state_vector_length)]
    #     rewarded_state_vector_indexes = defaultdict(list)

    #     # loop all but starting node
    #     for i in range(1, vlength):
    #         if i in skip_states:
    #             continue
    #         vertex = mutation_graph.vertex_at(i)
    #         if vertex.rate() > 0: # not absorbing
    #             for j in range(state_vector_length):
    #                 if j in skip_slots:
    #                     continue
    #                 val = vertex.state()[j]
    #                 if val > 0: # only ones we may reward
    #                     # add auxilliary node
    #                     mutation_vertex = mutation_graph.create_vertex(np.repeat(0, state_vector_length))
    #                     mutation_vertex.add_edge(vertex, 1)
    #                     vertex.add_edge(mutation_vertex, mutation_rate*val)
    #                     # print(mutation_vertex.index(), rewarded_state_vector_indexes[j], j)
    #                     # rewarded_state_vector_indexes[mutation_vertex.index()] = rewarded_state_vector_indexes[j] + [j]
    #                     rewarded_state_vector_indexes[mutation_vertex.index()].append(j)

    #     # normalize graph
    #     weights_were_multiplied_with = mutation_graph.normalize()

    #     # build reward matrix
    #     rewards = np.zeros((mutation_graph.vertices_length(), state_vector_length))
    #     for state in rewarded_state_vector_indexes:
    #         for i in rewarded_state_vector_indexes[state]:
    #             rewards[state, i] = 1

    #     rewards = np.transpose(rewards)
    #     return NamedTuple("DiscreteGraph", (mutation_graph, rewards))


    def discretize(self, reward_rate:float, skip_states:Sequence[int]=[], 
                   skip_slots:Sequence[int]=[]) -> Tuple[GraphType, np.ndarray]:
        """Creates a graph for a discrete distribution from a continuous one.

        Creates a graph augmented with auxiliary vertices and edges to represent the discrete distribution. 

        Parameters
        ----------
        reward_rate : 
            Rate of discrete events.
        skip_states : 
            Vertex indices to not add auxiliary states to, by default []
        skip_slots : 
            State vector indices to not add rewards to, by default []

        Returns
        -------
        :
            A new graph and a matrix of rewards for computing marginal moments.

        Examples
        --------
        
        >>> from phasic import Graph
        >>> def callback(state):
        ...     return [(state[0] + 1, [(state[0], 1)])]
        >>> g = Graph(callback=callback)
        >>> g.discretize(0.1)
        >>> a = [1, 2, 3]
        >>> print([x + 3 for x in a])
        [4, 5, 6]
        >>> print("a\nb")
        a
        b            
        """

        new_graph = self.copy()

        # save current nr of states in graph
        vlength = new_graph.vertices_length()

        state_vector_length = len(new_graph.vertex_at(1).state())

        # record state vector fields for unit rewards
        rewarded_state_vector_indexes = defaultdict(list)

        # loop all but starting node
        for i in range(1, vlength):
            if i in skip_states:
                continue
            vertex = new_graph.vertex_at(i)
            if vertex.rate() > 0: # not absorbing
                for j in range(state_vector_length):
                    if j in skip_slots:
                        continue
                    val = vertex.state()[j]
                    if val > 0: # only ones we may reward
                        # add aux node
                        mutation_vertex = new_graph.create_vertex(np.repeat(0, state_vector_length))
                        mutation_vertex.add_edge(vertex, 1)
                        vertex.add_edge(mutation_vertex, reward_rate*val)
                        rewarded_state_vector_indexes[mutation_vertex.index()].append(j)

        # normalize graph
        weight_scaling = new_graph.normalize()

        # build reward matrix
        rewards = np.zeros((new_graph.vertices_length(), state_vector_length)).astype(int)
        for state in rewarded_state_vector_indexes:
            for i in rewarded_state_vector_indexes[state]:
                rewards[state, i] = 1
        rewards = np.transpose(rewards)
        return new_graph, rewards


    def serialize(self, param_length: int = None) -> Dict[str, np.ndarray]:
        """
        Serialize graph to array representation for efficient computation.

        Parameters
        ----------
        param_length : int, optional
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
            - 'param_edges': Array of parameterized edges [from_idx, to_idx, x1, x2, ...] (n_param_edges, param_length+2)
            - 'start_param_edges': Array of starting vertex parameterized edges [to_idx, x1, x2, ...] (n_start_param_edges, param_length+1)
            - 'param_length': Length of parameter vector (0 if no parameterized edges)
            - 'state_length': Integer state dimension
            - 'n_vertices': Number of vertices
        """
        vertices_list = list(self.vertices())
        n_vertices = len(vertices_list)

        if n_vertices == 0:
            raise ValueError("Graph has no vertices (except starting vertex)")

        # Extract states and create state-based mapping
        state_length = self.state_length()
        states = np.zeros((n_vertices, state_length), dtype=np.int32)
        state_to_idx = {}

        for i, v in enumerate(vertices_list):
            state = v.state()
            states[i, :] = state
            # Use tuple of state for hashable key
            state_tuple = tuple(state)
            state_to_idx[state_tuple] = i

        # Detect or use provided parameter length
        start = self.starting_vertex()

        # Track the actual coefficient length for each edge (before garbage starts)
        # Key: (from_vertex_idx, to_vertex_idx), Value: valid length
        # Use -1 for starting vertex index
        edge_valid_lengths = {}

        # If param_length provided explicitly, use it directly
        if param_length is not None:
            # Use provided value - skip auto-detection
            detected_param_length = param_length
        else:
            # Auto-detect parameter length from parameterized edges
            # Strategy: Track the highest non-zero coefficient index
            detected_param_length = 0

            # Check all vertices for parameterized edges to determine actual param usage
            # We probe up to a reasonable limit and track the highest non-zero coefficient index
            max_probe_length = 20

            # Probe edges from regular vertices
            for i, v in enumerate(vertices_list):
                v_state = tuple(v.state())
                if v_state == tuple(start.state()):
                    continue  # Skip starting vertex (handled separately)

                from_idx = i
                for edge in v.parameterized_edges():
                    to_vertex = edge.to()
                    to_state = tuple(to_vertex.state())
                    if to_state not in state_to_idx:
                        continue
                    to_idx = state_to_idx[to_state]

                    valid_length = 0
                    last_nonzero_pos = 0
                    # Probe increasing lengths until we hit garbage or exceed limit
                    for try_len in range(1, max_probe_length + 1):
                        state = edge.edge_state(try_len)
                        if len(state) == 0:
                            break

                        val = state[-1]
                        # Check for garbage (NaN, inf, or suspiciously large/tiny values)
                        # Use 1e-100 as threshold to catch denormal floats like 5e-324
                        if (np.isnan(val) or np.isinf(val) or
                            abs(val) > 1e100 or
                            (val != 0 and abs(val) < 1e-100)):
                            break

                        # Track valid length up to here (before garbage)
                        valid_length = try_len

                        if val != 0:
                            # Found non-zero: record as potential param position
                            detected_param_length = max(detected_param_length, try_len)

                    edge_valid_lengths[(from_idx, to_idx)] = valid_length

            # Probe edges from starting vertex
            for edge in start.parameterized_edges():
                to_vertex = edge.to()
                to_state = tuple(to_vertex.state())
                if to_state not in state_to_idx:
                    continue
                to_idx = state_to_idx[to_state]

                valid_length = 0
                last_nonzero_pos = 0
                for try_len in range(1, max_probe_length + 1):
                    state = edge.edge_state(try_len)
                    if len(state) == 0:
                        break

                    val = state[-1]
                    if (np.isnan(val) or np.isinf(val) or
                        abs(val) > 1e100 or
                        (val != 0 and abs(val) < 1e-100)):
                        break

                    # Track valid length up to here (before garbage)
                    valid_length = try_len

                    if val != 0:
                        # Found non-zero: record as potential param position
                        detected_param_length = max(detected_param_length, try_len)

                edge_valid_lengths[(-1, to_idx)] = valid_length

        # Use detected_param_length for all subsequent operations
        param_length = detected_param_length

        # Sanity check: param_length should never exceed max_probe_length (only applies to auto-detected)
        # If it does, we likely read garbage/adjacent memory - use a conservative fallback
        if param_length is None and param_length > max_probe_length:
            # This shouldn't happen with proper garbage detection, but guard against it
            param_length = max_probe_length
            import warnings
            warnings.warn(
                f"Detected param_length={param_length} exceeds max_probe_length={max_probe_length}. "
                f"This may indicate edge_state() is reading adjacent memory. "
                f"Capping to {max_probe_length}.",
                RuntimeWarning
            )

        # Extract parameterized edges FIRST (needed to build exclusion set before extracting regular edges)
        start_state = tuple(start.state())

        # Extract parameterized edges between vertices (excluding starting vertex)
        param_edges_list = []
        if param_length > 0:
            for i, v in enumerate(vertices_list):
                # Skip starting vertex edges (they're handled separately)
                v_state = tuple(v.state())
                if v_state == start_state:
                    continue

                from_idx = i
                for edge in v.parameterized_edges():
                    to_vertex = edge.to()
                    to_state = tuple(to_vertex.state())
                    if to_state in state_to_idx:
                        to_idx = state_to_idx[to_state]
                        # Use edge-specific valid length, padded/truncated to param_length
                        # If edge_valid_lengths is empty (when param_length explicitly provided), use param_length
                        edge_len = edge_valid_lengths.get((from_idx, to_idx), param_length)
                        if edge_len > 0:
                            edge_state = list(edge.edge_state(edge_len))
                            # Ensure exactly param_length coefficients (truncate or pad)
                            if len(edge_state) < param_length:
                                edge_state.extend([0.0] * (param_length - len(edge_state)))
                            elif len(edge_state) > param_length:
                                edge_state = edge_state[:param_length]
                            # Only include edges with non-empty edge states
                            if any(x != 0 for x in edge_state):
                                # Store: [from_idx, to_idx, x1, x2, x3, ...]
                                param_edges_list.append([from_idx, to_idx] + edge_state)

        param_edges = np.array(param_edges_list, dtype=np.float64) if param_edges_list else np.empty((0, param_length + 2 if param_length > 0 else 0), dtype=np.float64)

        # Extract starting vertex parameterized edges FIRST (needed to build exclusion set)
        start_param_edges_list = []
        if param_length > 0:
            for edge in start.parameterized_edges():
                to_vertex = edge.to()
                to_state = tuple(to_vertex.state())
                if to_state in state_to_idx:
                    to_idx = state_to_idx[to_state]
                    # Use edge-specific valid length, padded/truncated to param_length (-1 = starting vertex)
                    # If edge_valid_lengths is empty (when param_length explicitly provided), use param_length
                    edge_len = edge_valid_lengths.get((-1, to_idx), param_length)
                    if edge_len > 0:
                        edge_state = list(edge.edge_state(edge_len))
                        # Ensure exactly param_length coefficients (truncate or pad)
                        if len(edge_state) < param_length:
                            edge_state.extend([0.0] * (param_length - len(edge_state)))
                        elif len(edge_state) > param_length:
                            edge_state = edge_state[:param_length]
                        # Only include edges with non-empty edge states
                        if any(x != 0 for x in edge_state):
                            # Store: [to_idx, x1, x2, x3, ...]
                            start_param_edges_list.append([to_idx] + edge_state)

        start_param_edges = np.array(start_param_edges_list, dtype=np.float64) if start_param_edges_list else np.empty((0, param_length + 1 if param_length > 0 else 0), dtype=np.float64)

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
        for i, v in enumerate(vertices_list):
            # Skip starting vertex edges (they're handled separately)
            v_state = tuple(v.state())
            if v_state == start_state:
                continue

            from_idx = i
            for edge in v.edges():
                to_vertex = edge.to()
                to_state = tuple(to_vertex.state())
                if to_state in state_to_idx:
                    to_idx = state_to_idx[to_state]
                    # Skip if this edge also has a parameterized version
                    if (from_idx, to_idx) not in param_edge_pairs:
                        weight = edge.weight()
                        edges_list.append([from_idx, to_idx, weight])

        edges = np.array(edges_list, dtype=np.float64) if edges_list else np.empty((0, 3), dtype=np.float64)

        # Extract starting vertex regular edges (skip those with parameterized versions)
        start_edges_list = []
        for edge in start.edges():
            to_vertex = edge.to()
            to_state = tuple(to_vertex.state())
            if to_state in state_to_idx:
                to_idx = state_to_idx[to_state]
                # Skip if this edge also has a parameterized version
                if (-1, to_idx) not in param_edge_pairs:
                    weight = edge.weight()
                    start_edges_list.append([to_idx, weight])

        start_edges = np.array(start_edges_list, dtype=np.float64) if start_edges_list else np.empty((0, 2), dtype=np.float64)

        return {
            'states': states,
            'edges': edges,
            'start_edges': start_edges,
            'param_edges': param_edges,
            'start_param_edges': start_param_edges,
            'param_length': param_length,
            'state_length': state_length,
            'n_vertices': n_vertices
        }

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
    def from_matrices(cls, ipv: np.ndarray, sim: np.ndarray, states: Optional[np.ndarray] = None) -> GraphType:
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
    def pmf_from_graph(cls, graph: 'Graph', discrete: bool = False, use_cache: bool = True, param_length: int = None) -> Callable:
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
        serialized = graph.serialize(param_length=param_length)
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
                # FALLBACK MODE: pure_callback (single-core, no FFI)
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
             discrete: bool = False,
             prior: Optional[Callable] = None,
             n_particles: int = 50,
             n_iterations: int = 1000,
             learning_rate: float = 0.001,
             bandwidth: str = 'median',
             theta_init: Optional[ArrayLike] = None,
             theta_dim: Optional[int] = None,
             return_history: bool = True,
             seed: int = 42,
             verbose: bool = True,
             jit: Optional[bool] = None,
             parallel: Optional[str] = None,
             n_devices: Optional[int] = None,
             precompile: bool = True,
             compilation_config: Optional[object] = None,
             regularization=10, 
             nr_moments=2,
             positive_params: bool = True,
             param_transform: Optional[Callable] = None) -> Dict:    
    # @classmethod
    # def svgd(cls,
    #          model: Callable,
    #          observed_data: ArrayLike,
    #          prior: Optional[Callable] = None,
    #          n_particles: int = 50,
    #          n_iterations: int = 1000,
    #          learning_rate: float = 0.001,
    #          kernel: str = 'median',
    #          theta_init: Optional[ArrayLike] = None,
    #          theta_dim: Optional[int] = None,
    #          return_history: bool = True,
    #          seed: int = 42,
    #          verbose: bool = True,
    #          positive_params: bool = True,
    #          param_transform: Optional[Callable] = None) -> Dict:
        """
        Run Stein Variational Gradient Descent (SVGD) inference for Bayesian parameter estimation.

        SVGD finds the posterior distribution p(theta | data) by optimizing a set of particles to
        approximate the posterior. This method works with parameterized models created by
        pmf_from_graph() or pmf_from_cpp() where the model signature is model(theta, times).

        Parameters
        ----------
        model : callable
            JAX-compatible parameterized model from pmf_from_graph() or pmf_from_cpp().
            Must have signature: model(theta, times) -> values
        observed_data : array_like
            Observed data points. For continuous models (PDF), these are time points where
            the density was observed. For discrete models (PMF), these are jump counts.
        discrete : bool, default=False
            If True, computes discrete PMF. If False, computes continuous PDF.
        prior : callable, optional
            Log prior function: prior(theta) -> scalar.
            If None, uses standard normal prior: log p(theta) = -0.5 * sum(theta^2)
        n_particles : int, default=50
            Number of SVGD particles. More particles = better posterior approximation but slower.
        n_iterations : int, default=1000
            Number of SVGD optimization steps
        learning_rate : float, default=0.001
            SVGD step size. Larger values = faster convergence but may be unstable.
        kernel : str, default='median'
            Kernel bandwidth selection method:
            - 'median': RBF kernel with median heuristic bandwidth (default)
            - 'rbf_adaptive': RBF kernel with adaptive bandwidth
        theta_init : array_like, optional
            Initial particle positions (n_particles, theta_dim).
            If None, initializes randomly from standard normal.
        theta_dim : int, optional
            Dimension of theta parameter vector. Required if theta_init is None.
        return_history : bool, default=True
            If True, return particle positions throughout optimization
        seed : int, default=42
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
        ...     model=model,
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

        model = Graph.pmf_and_moments_from_graph(self, nr_moments=2, discrete=discrete,
                                                  param_length=theta_dim)

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
            jit=jit,
            parallel=parallel,
            n_devices=n_devices,
            precompile=precompile,
            compilation_config=compilation_config,
            regularization=regularization,
            nr_moments=nr_moments,
            positive_params=positive_params,
            param_transform=param_transform
        )

        # Run inference
        svgd.fit(return_history=return_history)

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
        >>> graph = Graph(callback=coalescent, parameterized=True, nr_samples=3)
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
                                   param_length: int = None) -> Callable:
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
        param_length : int, optional
            Number of parameters for parameterized edges. If not provided, will be
            auto-detected by probing edge states. Providing this explicitly avoids
            potential issues with auto-detection reading garbage memory.

        Returns
        -------
        callable
            JAX-compatible function with signature:
            model(theta, times) -> (pmf_values, moments)

            Where:
            - pmf_values: jnp.array(len(times),) - PMF/PDF values at each time
            - moments: jnp.array(nr_moments,) - [E[T], E[T^2], ..., E[T^k]]

        Examples
        --------
        >>> # Create parameterized model
        >>> graph = Graph(callback=coalescent, parameterized=True, nr_samples=3)
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
        serialized = graph.serialize(param_length=param_length)
        param_length = serialized.get('param_length', 0)

        if param_length == 0:
            raise ValueError(
                "Graph must have parameterized edges. "
                "Create graph with parameterized=True and use add_edge_parameterized()."
            )

        # Check if FFI is available (same pattern as pmf_from_graph)
        config = get_config()
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
            def _compute_pure(theta, times):
                """FFI wrapper for multi-core parallelization.

                Supports: jit, vmap, pmap with true multi-core execution
                FFI caching: GraphBuilder cached by JSON structure
                """
                theta = jnp.atleast_1d(theta)
                times = jnp.atleast_1d(times)
                return model_ffi_partial(theta=theta, times=times)
        else:
            # FALLBACK MODE: Use pybind11 GraphBuilder (same as pmf_from_graph)
            import json
            from . import phasic_pybind as cpp_module
            from .ffi_wrappers import _make_json_serializable

            structure_json_str = json.dumps(_make_json_serializable(serialized))

            # Create GraphBuilder ONCE - captured in model closure
            builder = cpp_module.parameterized.GraphBuilder(structure_json_str)

            def _compute_pmf_and_moments_cached(theta_np, times_np):
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
                            granularity=0
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
                        granularity=0
                    )
                    return pmf, moments

            # Helper function for pure callback (used in forward and backward pass)
            def _compute_pure(theta, times):
                """Pure computation without custom_vjp wrapper"""
                theta = jnp.atleast_1d(theta)
                times = jnp.atleast_1d(times)

                pmf_shape = jax.ShapeDtypeStruct(times.shape, jnp.float64)
                moments_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)

                result = jax.pure_callback(
                    lambda theta_jax, times_jax: _compute_pmf_and_moments_cached(
                        np.asarray(theta_jax),
                        np.asarray(times_jax)
                    ),
                    (pmf_shape, moments_shape),
                    theta, times,
                    vmap_method='expand_dims'
                )
                return result

        # Wrap for JAX compatibility with custom VJP for gradients
        @jax.custom_vjp
        def model(theta, times):
            """JAX-compatible model function returning (pmf, moments)"""
            return _compute_pure(theta, times)

        def model_fwd(theta, times):
            # Call the underlying computation, not model (avoid infinite recursion!)
            pmf, moments = _compute_pure(theta, times)
            return (pmf, moments), (theta, times)

        def model_bwd(res, g):
            theta, times = res
            g_pmf, g_moments = g  # Unpack gradient tuple

            n_params = theta.shape[0]
            eps = 1e-7

            # Finite differences for gradient
            theta_bar = []
            for i in range(n_params):
                theta_plus = theta.at[i].add(eps)
                theta_minus = theta.at[i].add(-eps)

                # Call underlying computation, not model
                pmf_plus, moments_plus = _compute_pure(theta_plus, times)
                pmf_minus, moments_minus = _compute_pure(theta_minus, times)

                # Combine gradients from both PMF and moments
                grad_pmf_i = jnp.sum(g_pmf * (pmf_plus - pmf_minus) / (2 * eps))
                grad_moments_i = jnp.sum(g_moments * (moments_plus - moments_minus) / (2 * eps))
                grad_i = grad_pmf_i + grad_moments_i

                theta_bar.append(grad_i)

            return jnp.array(theta_bar), None

        model.defvjp(model_fwd, model_bwd)
        return model

    def plot(self, *args, **kwargs):
        """
        Plots the graph using graphviz. See plot::plot_graph.py for more details.

        Returns
        -------
        :
            _description_
        """
        return plot.plot_graph(self, *args, **kwargs)

    def copy(self) -> GraphType:
        """
        Returns a deep copy of the graph.
        """
        return Graph(self.clone())

        # """
        # Takes a graph for a continuous distribution and turns
        # it into a descrete one (inplace). Returns a matrix of
        # rewards for computing marginal moments
        # """

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
    #     times : array_like
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
    #     jumps : array_like
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
    #     powers : array_like
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
        >>> g = Graph(state_length=1, parameterized=True)
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
    >>> g = Graph(state_length=1, parameterized=True)
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
        clear_cache,
        cache_info,
        print_cache_info,
        export_model_package,
        generate_warmup_script
    )


# ============================================================================
# Auto-register IPython magic for CPU monitoring
# ============================================================================

try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None and CPUMonitorMagics is not None:
        ipython.register_magics(CPUMonitorMagics)
except (ImportError, NameError):
    # Not in IPython environment or magic not available
    pass


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
