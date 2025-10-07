from functools import partial
from collections import defaultdict
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

# Optional JAX support
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

# Cache for compiled libraries
_lib_cache = {}

from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot
from .plot import set_theme

# Optional SVGD support (requires JAX)
if HAS_JAX:
    from .svgd import SVGD
else:
    SVGD = None

__version__ = '0.19.106'

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
    vertex_code.append(f"    std::vector<ptdalgorithms::Vertex*> vertices;")

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
            weight_terms = [f"{edge_state[j]}*theta[{j}]" for j in range(param_length)]
            weight_expr = " + ".join(weight_terms)
            param_edge_code.append(f"    double w_start_{to_idx} = {weight_expr};")
            param_edge_code.append(f"    start->add_edge(*vertices[{to_idx}], w_start_{to_idx});")

        # Regular vertex parameterized edges
        for i, edge in enumerate(param_edges):
            from_idx = int(edge[0])
            to_idx = int(edge[1])
            edge_state = edge[2:]
            # Generate weight computation
            weight_terms = [f"{edge_state[j]}*theta[{j}]" for j in range(param_length)]
            weight_expr = " + ".join(weight_terms)
            param_edge_code.append(f"    double w_{from_idx}_{to_idx} = {weight_expr};")
            param_edge_code.append(f"    vertices[{from_idx}]->add_edge(*vertices[{to_idx}], w_{from_idx}_{to_idx});")

    # Combine all code
    cpp_code = f'''#include "ptdalgorithmscpp.h"

ptdalgorithms::Graph build_model(const double* theta, int n_params) {{
    ptdalgorithms::Graph g({state_dim});

{chr(10).join(vertex_code)}

{chr(10).join(edge_code)}

{chr(10).join(param_edge_code) if param_edge_code else ""}

    return g;
}}
'''
    return cpp_code


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
            f'{pkg_dir}/src/cpp/ptdalgorithmscpp.cpp',
            f'{pkg_dir}/src/c/ptdalgorithms.c',
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


    def make_discrete(self, mutation_rate, skip_states=[], skip_slots=[]):
        """
        Takes a graph for a continuous distribution and turns
        it into a descrete one (inplace). Returns a matrix of
        rewards for computing marginal moments
        """

        mutation_graph = self.copy()

        # save current nr of states in graph
        vlength = mutation_graph.vertices_length()

        # number of fields in state vector (assumes all are the same length)
        state_vector_length = len(mutation_graph.vertex_at(1).state())

        # list state vector fields to reward at each auxiliary node
        # rewarded_state_vector_indexes = [[] for _ in range(state_vector_length)]
        rewarded_state_vector_indexes = defaultdict(list)

        # loop all but starting node
        for i in range(1, vlength):
            if i in skip_states:
                continue
            vertex = mutation_graph.vertex_at(i)
            if vertex.rate() > 0: # not absorbing
                for j in range(state_vector_length):
                    if j in skip_slots:
                        continue
                    val = vertex.state()[j]
                    if val > 0: # only ones we may reward
                        # add auxilliary node
                        mutation_vertex = mutation_graph.create_vertex(np.repeat(0, state_vector_length))
                        mutation_vertex.add_edge(vertex, 1)
                        vertex.add_edge(mutation_vertex, mutation_rate*val)
                        # print(mutation_vertex.index(), rewarded_state_vector_indexes[j], j)
                        # rewarded_state_vector_indexes[mutation_vertex.index()] = rewarded_state_vector_indexes[j] + [j]
                        rewarded_state_vector_indexes[mutation_vertex.index()].append(j)

        # normalize graph
        weights_were_multiplied_with = mutation_graph.normalize()

        # build reward matrix
        rewards = np.zeros((mutation_graph.vertices_length(), state_vector_length))
        for state in rewarded_state_vector_indexes:
            for i in rewarded_state_vector_indexes[state]:
                rewards[state, i] = 1

        rewards = np.transpose(rewards)
        return NamedTuple("DiscreteGraph", (mutation_graph, rewards))





    def serialize(self) -> Dict[str, np.ndarray]:
        """
        Serialize graph to array representation for efficient computation.

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

        # Detect parameter length from parameterized edges
        param_length = 0
        start = self.starting_vertex()

        # Check all vertices for parameterized edges to determine param_length
        # Strategy: Collect a few parameterized edges and check for consistency
        # All param edges should have the same length
        sample_edges = []
        for v in vertices_list:
            param_edges = v.parameterized_edges()
            if param_edges:
                sample_edges.extend(param_edges[:2])  # Take up to 2 edges per vertex
                if len(sample_edges) >= 5:  # Collect a few samples
                    break

        if sample_edges:
            # Try increasing lengths and stop when we hit uninitialized memory
            # Strategy: Uninitialized memory typically contains extremely tiny values (< 1e-300)
            # or NaN/inf. Valid edge state coefficients should be reasonable numbers.
            for try_len in range(1, 20):
                hit_garbage = False

                for edge in sample_edges:
                    state = edge.edge_state(try_len)
                    if len(state) == 0:
                        hit_garbage = True
                        break
                    last_val = state[-1]

                    # Check for obvious garbage:
                    # 1. NaN or inf
                    # 2. Extremely large values (> 1e100)
                    # 3. Extremely tiny values (< 1e-300) which indicate uninitialized memory
                    #    Note: We use 1e-300 instead of checking == 0 because 0.0 is valid
                    if (np.isnan(last_val) or np.isinf(last_val) or
                        abs(last_val) > 1e100 or
                        (last_val != 0 and abs(last_val) < 1e-300)):
                        hit_garbage = True
                        break

                if hit_garbage:
                    # At least one edge hit garbage, previous length was correct
                    break

                # This length seems valid
                param_length = try_len

        # Also check starting vertex if we haven't found param_length yet
        if param_length == 0:
            start_param_edges = start.parameterized_edges()
            if start_param_edges:
                # Use same strategy as above
                for try_len in range(1, 20):
                    edge_state = start_param_edges[0].edge_state(try_len)
                    if len(edge_state) == 0:
                        break
                    last_val = edge_state[-1]
                    if np.isnan(last_val) or np.isinf(last_val) or abs(last_val) > 1e100:
                        break
                    param_length = try_len

        # Extract regular edges between vertices (excluding starting vertex)
        edges_list = []
        start_state = tuple(start.state())
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
                    weight = edge.weight()
                    edges_list.append([from_idx, to_idx, weight])

        edges = np.array(edges_list, dtype=np.float64) if edges_list else np.empty((0, 3), dtype=np.float64)

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
                        edge_state = edge.edge_state(param_length)
                        # Only include edges with non-empty edge states
                        if len(edge_state) > 0 and any(x != 0 for x in edge_state):
                            # Store: [from_idx, to_idx, x1, x2, x3, ...]
                            param_edges_list.append([from_idx, to_idx] + list(edge_state))

        param_edges = np.array(param_edges_list, dtype=np.float64) if param_edges_list else np.empty((0, param_length + 2 if param_length > 0 else 0), dtype=np.float64)

        # Extract starting vertex regular edges
        start_edges_list = []
        for edge in start.edges():
            to_vertex = edge.to()
            to_state = tuple(to_vertex.state())
            if to_state in state_to_idx:
                to_idx = state_to_idx[to_state]
                weight = edge.weight()
                start_edges_list.append([to_idx, weight])

        start_edges = np.array(start_edges_list, dtype=np.float64) if start_edges_list else np.empty((0, 2), dtype=np.float64)

        # Extract starting vertex parameterized edges
        start_param_edges_list = []
        if param_length > 0:
            for edge in start.parameterized_edges():
                to_vertex = edge.to()
                to_state = tuple(to_vertex.state())
                if to_state in state_to_idx:
                    to_idx = state_to_idx[to_state]
                    edge_state = edge.edge_state(param_length)
                    # Only include edges with non-empty edge states
                    if len(edge_state) > 0 and any(x != 0 for x in edge_state):
                        # Store: [to_idx, x1, x2, x3, ...]
                        start_param_edges_list.append([to_idx] + list(edge_state))

        start_param_edges = np.array(start_param_edges_list, dtype=np.float64) if start_param_edges_list else np.empty((0, param_length + 1 if param_length > 0 else 0), dtype=np.float64)

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
    def pmf_from_graph(cls, graph: 'Graph', discrete: bool = False) -> Callable:
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
        """
        # Check if JAX is available
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for JAX-compatible models. "
                "Install with: pip install 'ptdalgorithms[jax]' or pip install jax jaxlib"
            )

        # Serialize the graph (now includes parameterized edges)
        serialized = graph.serialize()
        param_length = serialized.get('param_length', 0)
        has_param_edges = param_length > 0

        # Generate C++ build_model() code from the serialized graph
        cpp_code = _generate_cpp_from_graph(serialized)

        # Create hash of the generated C++ code
        cpp_hash = hashlib.sha256(cpp_code.encode()).hexdigest()[:16]
        temp_file = f"/tmp/graph_model_{cpp_hash}.cpp"

        # Write C++ code to temp file
        with open(temp_file, 'w') as f:
            f.write(cpp_code)

        # Use pmf_from_cpp() to compile and create JAX-compatible function
        # This gives us full JAX support (jit, grad, vmap)
        base_model = cls.pmf_from_cpp(temp_file, discrete=discrete)

        # Return appropriate signature based on parameterization
        if has_param_edges:
            # Parameterized: return (theta, times) -> pmf
            return base_model
        else:
            # Non-parameterized: wrap to hide theta parameter
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
#include "ptdalgorithmscpp.h"
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
        ptdalgorithms::Graph g(state_dim);
        auto start = g.starting_vertex_p();

        // Create vertices
        std::vector<ptdalgorithms::Vertex*> vertices;
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
        ptdalgorithms::Graph g(state_dim);
        auto start = g.starting_vertex_p();

        // Create vertices
        std::vector<ptdalgorithms::Vertex*> vertices;
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

        ptdalgorithms::Graph build_model(const double* theta, int n_params) {
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
                "Install with: pip install 'ptdalgorithms[jax]' or pip install jax jaxlib"
            )

        # Read user's C++ code
        with open(cpp_path, 'r') as f:
            user_code = f.read()

        # Check that it implements build_model
        if "build_model" not in user_code:
            raise ValueError(
                "C++ file must implement: ptdalgorithms::Graph build_model(const double* theta, int n_params)"
            )

        # Create wrapper code with both continuous and discrete computation
        wrapper_code = f'''
// Include the C++ API header (which includes the C headers)
#include "ptdalgorithmscpp.h"

// Include user's model
#include "{cpp_path.absolute()}"

extern "C" {{
    // Wrapper functions that compute PMF/DPH directly
    void compute_pmf(const double* theta, int n_params,
                     const double* times, int n_times,
                     double* output, int granularity) {{
        ptdalgorithms::Graph g = build_model(theta, n_params);
        for (int i = 0; i < n_times; i++) {{
            output[i] = g.pdf(times[i], granularity);
        }}
    }}

    void compute_dph_pmf(const double* theta, int n_params,
                         const int* jumps, int n_jumps,
                         double* output) {{
        ptdalgorithms::Graph g = build_model(theta, n_params);
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
            result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float32)
            return jax.pure_callback(
                lambda t, tm: pmf_function(t, tm, granularity=0).astype(np.float32),
                result_shape,
                theta,
                times,
                vmap_method='sequential'
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

    @classmethod
    def svgd(cls,
             model: Callable,
             observed_data: ArrayLike,
             prior: Optional[Callable] = None,
             n_particles: int = 50,
             n_iterations: int = 1000,
             learning_rate: float = 0.001,
             kernel: str = 'rbf_median',
             theta_init: Optional[ArrayLike] = None,
             theta_dim: Optional[int] = None,
             return_history: bool = False,
             seed: int = 42,
            verbose: bool = True) -> Dict:
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
        prior : callable, optional
            Log prior function: prior(theta) -> scalar.
            If None, uses standard normal prior: log p(theta) = -0.5 * sum(theta^2)
        n_particles : int, default=50
            Number of SVGD particles. More particles = better posterior approximation but slower.
        n_iterations : int, default=1000
            Number of SVGD optimization steps
        learning_rate : float, default=0.001
            SVGD step size. Larger values = faster convergence but may be unstable.
        kernel : str, default='rbf_median'
            Kernel bandwidth selection method:
            - 'rbf_median': RBF kernel with median heuristic bandwidth (default)
            - 'rbf_adaptive': RBF kernel with adaptive bandwidth
        theta_init : array_like, optional
            Initial particle positions (n_particles, theta_dim).
            If None, initializes randomly from standard normal.
        theta_dim : int, optional
            Dimension of theta parameter vector. Required if theta_init is None.
        return_history : bool, default=False
            If True, return particle positions throughout optimization
        seed : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Print progress information

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
        >>> from ptdalgorithms import Graph
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
                "Install with: pip install 'ptdalgorithms[jax]' or pip install jax jaxlib"
            )

        from .svgd import SVGD

        # Create SVGD object
        svgd = SVGD(
            model=model,
            observed_data=observed_data,
            prior=prior,
            n_particles=n_particles,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            kernel=kernel,
            theta_init=theta_init,
            theta_dim=theta_dim,
            seed=seed,
            verbose=verbose
        )

        # Run inference
        svgd.fit(return_history=return_history)

        # Return results as dictionary for backward compatibility
        return svgd.get_results()

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
                "Install with: pip install 'ptdalgorithms[jax]' or pip install jax jaxlib"
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
        ptdalgorithms::Graph g = build_model(theta, n_params);

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
            return jax.pure_callback(_compute_moments_pure, result_shape, theta, vmap_method='sequential')

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
                                   discrete: bool = False, use_ffi: bool = False) -> Callable:
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
                "Install with: pip install 'ptdalgorithms[jax]' or pip install jax jaxlib"
            )

        import jax
        import jax.numpy as jnp

        # Serialize the graph
        serialized = graph.serialize()
        param_length = serialized.get('param_length', 0)

        if param_length == 0:
            raise ValueError(
                "Graph must have parameterized edges. "
                "Create graph with parameterized=True and use add_edge_parameterized()."
            )

        # Generate C++ build_model() code
        cpp_code = _generate_cpp_from_graph(serialized)

        # Create wrapper that computes both PMF and moments
        granularity = 100  # Default granularity for PDF computation

        if discrete:
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
    void compute_pmf_and_moments(
        const double* theta, int n_params,
        const int* times, int n_times,
        int nr_moments,
        double* pmf_output,
        double* moments_output
    ) {{
        // Build graph from theta
        ptdalgorithms::Graph g = build_model(theta, n_params);

        // Compute PMF for discrete case
        for (int i = 0; i < n_times; i++) {{
            pmf_output[i] = g.dph_pmf(times[i]);
        }}

        // Compute moments using expected_waiting_time() method
        // This replicates the _moments() function from pybind11 code
        std::vector<double> rewards;  // Empty rewards for standard moments
        std::vector<double> rewards2 = g.expected_waiting_time(rewards);
        std::vector<double> rewards3(rewards2.size());

        moments_output[0] = rewards2[0];  // First moment (mean)

        // Compute higher moments iteratively
        for (int i = 1; i < nr_moments; i++) {{
            for (int j = 0; j < (int)rewards3.size(); j++) {{
                rewards3[j] = rewards2[j] * std::pow(rewards2[j], i);
            }}
            rewards2 = g.expected_waiting_time(rewards3);
            moments_output[i] = factorial(i + 1) * rewards2[0];
        }}
    }}
}}
'''
        else:
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
    void compute_pmf_and_moments(
        const double* theta, int n_params,
        const double* times, int n_times,
        int nr_moments,
        double* pmf_output,
        double* moments_output
    ) {{
        // Build graph from theta
        ptdalgorithms::Graph g = build_model(theta, n_params);

        // Compute PDF for continuous case
        for (int i = 0; i < n_times; i++) {{
            pmf_output[i] = g.pdf(times[i], {granularity});
        }}

        // Compute moments using expected_waiting_time() method
        // This replicates the _moments() function from pybind11 code
        std::vector<double> rewards;  // Empty rewards for standard moments
        std::vector<double> rewards2 = g.expected_waiting_time(rewards);
        std::vector<double> rewards3(rewards2.size());

        moments_output[0] = rewards2[0];  // First moment (mean)

        // Compute higher moments iteratively
        for (int i = 1; i < nr_moments; i++) {{
            for (int j = 0; j < (int)rewards3.size(); j++) {{
                rewards3[j] = rewards2[j] * std::pow(rewards2[j], i);
            }}
            rewards2 = g.expected_waiting_time(rewards3);
            moments_output[i] = factorial(i + 1) * rewards2[0];
        }}
    }}
}}
'''

        # Compile the wrapper
        lib_name = f"pmf_moments_{hashlib.sha256(wrapper_code.encode()).hexdigest()[:16]}"
        lib_path = _compile_wrapper_library(wrapper_code, lib_name)

        # Load the library
        lib = ctypes.PyDLL(lib_path)

        # Define the function signature
        if discrete:
            lib.compute_pmf_and_moments.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # theta
                ctypes.c_int,                      # n_params
                ctypes.POINTER(ctypes.c_int),      # times
                ctypes.c_int,                      # n_times
                ctypes.c_int,                      # nr_moments
                ctypes.POINTER(ctypes.c_double),   # pmf_output
                ctypes.POINTER(ctypes.c_double)    # moments_output
            ]
        else:
            lib.compute_pmf_and_moments.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # theta
                ctypes.c_int,                      # n_params
                ctypes.POINTER(ctypes.c_double),   # times
                ctypes.c_int,                      # n_times
                ctypes.c_int,                      # nr_moments
                ctypes.POINTER(ctypes.c_double),   # pmf_output
                ctypes.POINTER(ctypes.c_double)    # moments_output
            ]
        lib.compute_pmf_and_moments.restype = None

        # Pure computation function
        def _compute_pmf_and_moments_pure(theta_flat, times_flat):
            """Pure function for combined PMF and moments computation"""
            theta_np = np.asarray(theta_flat, dtype=np.float64)
            times_np = np.asarray(times_flat, dtype=np.float64 if not discrete else np.int32)

            pmf_output = np.zeros(len(times_np), dtype=np.float64)
            moments_output = np.zeros(nr_moments, dtype=np.float64)

            if discrete:
                lib.compute_pmf_and_moments(
                    theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(theta_np),
                    times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    len(times_np),
                    nr_moments,
                    pmf_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    moments_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
            else:
                lib.compute_pmf_and_moments(
                    theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(theta_np),
                    times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(times_np),
                    nr_moments,
                    pmf_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    moments_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )

            return pmf_output, moments_output

        # Helper function for pure callback (used in forward and backward pass)
        def _compute_pure(theta, times):
            """Pure computation without custom_vjp wrapper"""
            theta = jnp.atleast_1d(theta)
            times = jnp.atleast_1d(times)

            pmf_shape = jax.ShapeDtypeStruct((len(times),), jnp.float64)
            moments_shape = jax.ShapeDtypeStruct((nr_moments,), jnp.float64)

            result = jax.pure_callback(
                _compute_pmf_and_moments_pure,
                (pmf_shape, moments_shape),
                theta, times,
                vmap_method='sequential'
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
        
        >>> from ptdalgorithms import Graph
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
    from . import ptdalgorithmscpp_pybind
    cpp_path = pathlib.Path(cpp_file).resolve()
    if not cpp_path.exists():
        raise FileNotFoundError(f"C++ file not found: {cpp_path}")
    return ptdalgorithmscpp_pybind.load_cpp_builder(str(cpp_path))
