from functools import partial
from collections import defaultdict
import numpy as np
from typing import Any, TypeVar, List, Tuple, Dict, Union, NamedTuple, Optional
from collections.abc import Sequence, MutableSequence, Callable
import os
import hashlib
import subprocess
import tempfile
import ctypes
import pathlib
import jax
import jax.numpy as jnp

# Cache for compiled libraries
_lib_cache = {}

from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot
from .plot import set_theme

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
    def __init__(self, state_length:int=None, callback:Callable=None, **kwargs):
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

        Returns
        -------
        :
            A graph object representing a phase-type distribution.
        """
        assert (callback is None) + (state_length is None) == 1, "Use either the state_length or callback argument"

        if callback:
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
            - 'edges': Array of edges [from_idx, to_idx, weight] (n_edges, 3)
            - 'start_edges': Array of starting vertex edges [to_idx, weight] (n_start_edges, 2)
            - 'state_length': Integer state dimension
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

        # Extract edges between vertices
        edges_list = []
        for i, v in enumerate(vertices_list):
            from_idx = i
            for edge in v.edges():
                to_vertex = edge.to()
                to_state = tuple(to_vertex.state())
                if to_state in state_to_idx:
                    to_idx = state_to_idx[to_state]
                    weight = edge.weight()
                    edges_list.append([from_idx, to_idx, weight])

        edges = np.array(edges_list, dtype=np.float64) if edges_list else np.empty((0, 3), dtype=np.float64)

        # Extract starting vertex edges
        start = self.starting_vertex()
        start_edges_list = []
        for edge in start.edges():
            to_vertex = edge.to()
            to_state = tuple(to_vertex.state())
            if to_state in state_to_idx:
                to_idx = state_to_idx[to_state]
                weight = edge.weight()
                start_edges_list.append([to_idx, weight])

        start_edges = np.array(start_edges_list, dtype=np.float64) if start_edges_list else np.empty((0, 2), dtype=np.float64)

        return {
            'states': states,
            'edges': edges,
            'start_edges': start_edges,
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
    def from_python_graph(cls, graph: 'Graph', jax_compatible: bool = True) -> Callable:
        """
        Convert a Python-built Graph to a JAX-compatible function.

        This allows users to build graphs using the Python API and then
        convert them to efficient JAX-compatible functions without writing C++.

        Parameters
        ----------
        graph : Graph
            Graph built using the Python API
        jax_compatible : bool
            If True, returns a JAX-compatible function

        Returns
        -------
        callable
            Function (times) -> pdf_values that computes the PDF

        Examples
        --------
        >>> g = Graph(1)
        >>> start = g.starting_vertex()
        >>> v0 = g.find_or_create_vertex([0])
        >>> v1 = g.find_or_create_vertex([1])
        >>> start.add_edge(v0, 1.0)
        >>> v0.add_edge(v1, 1.0)  # rate = 1.0
        >>>
        >>> model = Graph.from_python_graph(g)
        >>> times = jnp.linspace(0, 5, 50)
        >>> pdf = model(times)
        """
        # Serialize the graph once
        serialized = graph.serialize()

        # Get package directory
        pkg_dir = pathlib.Path(__file__).parent.parent.parent

        # Create wrapper code
        wrapper_code = f'''
#include "ptdalgorithmscpp.h"
#include <vector>

extern "C" {{
    void compute_pmf_from_arrays(
        const int* states, int n_vertices, int state_dim,
        const double* edges, int n_edges,
        const double* start_edges, int n_start_edges,
        const double* times, int n_times,
        double* output, int granularity
    ) {{
        // Create graph
        ptdalgorithms::Graph g(state_dim);
        auto start = g.starting_vertex_p();

        // Create vertices
        std::vector<ptdalgorithms::Vertex*> vertices;
        for (int i = 0; i < n_vertices; i++) {{
            std::vector<int> state(state_dim);
            for (int j = 0; j < state_dim; j++) {{
                state[j] = states[i * state_dim + j];
            }}
            auto v = g.find_or_create_vertex_p(state);
            vertices.push_back(v);
        }}

        // Add edges from starting vertex
        for (int i = 0; i < n_start_edges; i++) {{
            int to_idx = (int)start_edges[i * 2];
            double weight = start_edges[i * 2 + 1];
            start->add_edge(*vertices[to_idx], weight);
        }}

        // Add edges between vertices
        for (int i = 0; i < n_edges; i++) {{
            int from_idx = (int)edges[i * 3];
            int to_idx = (int)edges[i * 3 + 1];
            double weight = edges[i * 3 + 2];
            vertices[from_idx]->add_edge(*vertices[to_idx], weight);
        }}

        // Normalize and compute PDF
        g.normalize();
        for (int i = 0; i < n_times; i++) {{
            output[i] = g.pdf(times[i], granularity);
        }}
    }}
}}
'''

        # Create hash of serialized graph
        import hashlib
        graph_hash = hashlib.sha256(str(serialized).encode()).hexdigest()[:16]

        # Compile to shared library
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper_path = pathlib.Path(tmpdir) / "wrapper.cpp"
            wrapper_path.write_text(wrapper_code)

            lib_path = pathlib.Path(tmpdir) / f"graph_{graph_hash}.so"

            compile_cmd = [
                "g++", "-O3", "-fPIC", "-shared", "-std=c++14",
                f"-I{pkg_dir}",
                f"-I{pkg_dir}/api/cpp",
                f"-I{pkg_dir}/api/c",
                f"-I{pkg_dir}/include",
                str(wrapper_path),
                f"{pkg_dir}/src/cpp/ptdalgorithmscpp.cpp",
                f"{pkg_dir}/src/c/ptdalgorithms.c",
                "-o", str(lib_path)
            ]

            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed: {result.stderr}")

            # Load library
            lib = ctypes.CDLL(str(lib_path))
            compute_func = lib.compute_pmf_from_arrays
            compute_func.argtypes = [
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
            compute_func.restype = None

            # Keep library in memory
            _lib_cache[graph_hash] = lib

        # Prepare arrays
        states_flat = serialized['states'].flatten()
        edges_flat = serialized['edges'].flatten() if serialized['edges'].size > 0 else np.array([], dtype=np.float64)
        start_edges_flat = serialized['start_edges'].flatten() if serialized['start_edges'].size > 0 else np.array([], dtype=np.float64)
        n_vertices = serialized['n_vertices']
        state_length = serialized['state_length']
        n_edges = len(serialized['edges'])
        n_start_edges = len(serialized['start_edges'])

        if jax_compatible:
            # Create JAX-compatible wrapper using pure callback
            import jax
            from jax import pure_callback

            def compute_pmf_pure(times, granularity=100):
                """Pure function wrapper for JAX compatibility"""
                def compute_impl(times_arr):
                    times_np = np.asarray(times_arr, dtype=np.float64)
                    n_times = len(times_np)
                    output = np.zeros(n_times, dtype=np.float64)

                    compute_func(
                        states_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        n_vertices,
                        state_length,
                        edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(edges_flat) > 0 else None,
                        n_edges,
                        start_edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(start_edges_flat) > 0 else None,
                        n_start_edges,
                        times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        n_times,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        granularity
                    )
                    return output

                # Use pure_callback for JAX compatibility
                result_shape_dtypes = jax.ShapeDtypeStruct(times.shape, jnp.float32)
                return pure_callback(compute_impl, result_shape_dtypes, times)

            return compute_pmf_pure
        else:
            # Return regular Python function
            def model_fn(times, granularity=100):
                times_np = np.asarray(times, dtype=np.float64)
                n_times = len(times_np)
                output = np.zeros(n_times, dtype=np.float64)

                compute_func(
                    states_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    n_vertices,
                    state_length,
                    edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(edges_flat) > 0 else None,
                    n_edges,
                    start_edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(start_edges_flat) > 0 else None,
                    n_start_edges,
                    times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    n_times,
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    granularity
                )

                return output

            return model_fn

    @classmethod
    def from_python_graph_parameterized(cls, graph_builder: Callable, jax_compatible: bool = True) -> Callable:
        """
        Convert a parameterized Python graph builder to a JAX-compatible function.

        This allows users to define parameterized models where the graph structure
        or edge weights depend on parameters.

        Parameters
        ----------
        graph_builder : callable
            Function (theta) -> Graph that builds a graph with given parameters
        jax_compatible : bool
            If True, returns a JAX-compatible function

        Returns
        -------
        callable
            Function (theta, times) -> pdf_values that computes the PDF

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
        >>> model = Graph.from_python_graph_parameterized(build_exponential)
        >>> theta = jnp.array([1.5])
        >>> times = jnp.linspace(0, 5, 50)
        >>> pdf = model(theta, times)
        """
        # Get package directory
        pkg_dir = pathlib.Path(__file__).parent.parent.parent

        # Create wrapper code
        wrapper_code = f'''
#include "ptdalgorithmscpp.h"
#include <vector>

extern "C" {{
    void compute_pmf_parameterized(
        const double* theta, int n_params,
        const double* times, int n_times,
        double* output, int granularity
    ) {{
        // This is a placeholder - actual implementation depends on the model
        // In practice, we'll generate specific code for each model
    }}

    void compute_pmf_from_arrays(
        const int* states, int n_vertices, int state_dim,
        const double* edges, int n_edges,
        const double* start_edges, int n_start_edges,
        const double* times, int n_times,
        double* output, int granularity
    ) {{
        // Create graph
        ptdalgorithms::Graph g(state_dim);
        auto start = g.starting_vertex_p();

        // Create vertices
        std::vector<ptdalgorithms::Vertex*> vertices;
        for (int i = 0; i < n_vertices; i++) {{
            std::vector<int> state(state_dim);
            for (int j = 0; j < state_dim; j++) {{
                state[j] = states[i * state_dim + j];
            }}
            auto v = g.find_or_create_vertex_p(state);
            vertices.push_back(v);
        }}

        // Add edges from starting vertex
        for (int i = 0; i < n_start_edges; i++) {{
            int to_idx = (int)start_edges[i * 2];
            double weight = start_edges[i * 2 + 1];
            start->add_edge(*vertices[to_idx], weight);
        }}

        // Add edges between vertices
        for (int i = 0; i < n_edges; i++) {{
            int from_idx = (int)edges[i * 3];
            int to_idx = (int)edges[i * 3 + 1];
            double weight = edges[i * 3 + 2];
            vertices[from_idx]->add_edge(*vertices[to_idx], weight);
        }}

        // Normalize and compute PDF
        g.normalize();
        for (int i = 0; i < n_times; i++) {{
            output[i] = g.pdf(times[i], granularity);
        }}
    }}
}}
'''

        # Create hash for the builder function
        import inspect
        builder_source = inspect.getsource(graph_builder) if hasattr(graph_builder, '__code__') else str(graph_builder)
        builder_hash = hashlib.sha256(builder_source.encode()).hexdigest()[:16]

        # Check if already compiled
        if builder_hash not in _lib_cache:
            # Compile once
            with tempfile.TemporaryDirectory() as tmpdir:
                wrapper_path = pathlib.Path(tmpdir) / "wrapper.cpp"
                wrapper_path.write_text(wrapper_code)

                lib_path = pathlib.Path(tmpdir) / f"param_graph_{builder_hash}.so"

                compile_cmd = [
                    "g++", "-O3", "-fPIC", "-shared", "-std=c++14",
                    f"-I{pkg_dir}",
                    f"-I{pkg_dir}/api/cpp",
                    f"-I{pkg_dir}/api/c",
                    f"-I{pkg_dir}/include",
                    str(wrapper_path),
                    f"{pkg_dir}/src/cpp/ptdalgorithmscpp.cpp",
                    f"{pkg_dir}/src/c/ptdalgorithms.c",
                    "-o", str(lib_path)
                ]

                result = subprocess.run(compile_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Compilation failed: {result.stderr}")

                # Load library
                lib = ctypes.CDLL(str(lib_path))
                compute_func = lib.compute_pmf_from_arrays
                compute_func.argtypes = [
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
                compute_func.restype = None

                # Keep library in memory
                _lib_cache[builder_hash] = lib
        else:
            lib = _lib_cache[builder_hash]
            compute_func = lib.compute_pmf_from_arrays

        if jax_compatible:
            # Create JAX-compatible wrapper using pure callback
            import jax
            from jax import pure_callback

            def model_fn(theta, times, granularity=100):
                """Parameterized model with JAX compatibility"""
                def compute_impl(inputs):
                    theta_arr, times_arr = inputs

                    # Build graph with parameters
                    theta_np = np.asarray(theta_arr)
                    if theta_np.ndim == 0:
                        theta_np = theta_np.reshape(1)

                    # Build the graph
                    graph = graph_builder(*theta_np)

                    # Serialize
                    serialized = graph.serialize()

                    # Prepare arrays
                    states_flat = serialized['states'].flatten()
                    edges_flat = serialized['edges'].flatten() if serialized['edges'].size > 0 else np.array([], dtype=np.float64)
                    start_edges_flat = serialized['start_edges'].flatten() if serialized['start_edges'].size > 0 else np.array([], dtype=np.float64)
                    n_vertices = serialized['n_vertices']
                    state_length = serialized['state_length']
                    n_edges = len(serialized['edges'])
                    n_start_edges = len(serialized['start_edges'])

                    times_np = np.asarray(times_arr, dtype=np.float64)
                    n_times = len(times_np)
                    output = np.zeros(n_times, dtype=np.float64)

                    compute_func(
                        states_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        n_vertices,
                        state_length,
                        edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(edges_flat) > 0 else None,
                        n_edges,
                        start_edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(start_edges_flat) > 0 else None,
                        n_start_edges,
                        times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        n_times,
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        granularity
                    )

                    return output

                # Use pure_callback for JAX compatibility
                result_shape_dtypes = jax.ShapeDtypeStruct(times.shape, jnp.float32)
                return pure_callback(compute_impl, result_shape_dtypes, (theta, times))

            return model_fn
        else:
            # Return regular Python function
            def model_fn(theta, times, granularity=100):
                # Build graph with parameters
                theta_np = np.asarray(theta)
                if theta_np.ndim == 0:
                    theta_np = theta_np.reshape(1)

                # Build the graph
                graph = graph_builder(*theta_np)

                # Serialize
                serialized = graph.serialize()

                # Prepare arrays
                states_flat = serialized['states'].flatten()
                edges_flat = serialized['edges'].flatten() if serialized['edges'].size > 0 else np.array([], dtype=np.float64)
                start_edges_flat = serialized['start_edges'].flatten() if serialized['start_edges'].size > 0 else np.array([], dtype=np.float64)
                n_vertices = serialized['n_vertices']
                state_length = serialized['state_length']
                n_edges = len(serialized['edges'])
                n_start_edges = len(serialized['start_edges'])

                times_np = np.asarray(times, dtype=np.float64)
                n_times = len(times_np)
                output = np.zeros(n_times, dtype=np.float64)

                compute_func(
                    states_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    n_vertices,
                    state_length,
                    edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(edges_flat) > 0 else None,
                    n_edges,
                    start_edges_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if len(start_edges_flat) > 0 else None,
                    n_start_edges,
                    times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    n_times,
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    granularity
                )

                return output

            return model_fn

    @classmethod
    def load_cpp_model(cls, cpp_file: Union[str, pathlib.Path], jax_compatible: bool = True, use_ffi: bool = False) -> Callable:
        """
        Load a phase-type model from a user's C++ file.

        The C++ file should include 'user_model.h' and implement:

        ptdalgorithms::Graph build_model(const double* theta, int n_params) {
            // Build and return Graph instance
        }

        Parameters
        ----------
        cpp_file : str or Path
            Path to the user's C++ file
        jax_compatible : bool
            If True, returns a JAX-compatible function. If False, returns a regular Python function.
            Ignored if use_ffi=True.
        use_ffi : bool
            If True, uses Foreign Function Interface approach that separates graph construction
            from computation. Returns a builder function that creates reusable Graph objects.
            This is more efficient for repeated evaluations with the same parameters but
            doesn't support JAX transformations directly on the returned graphs.

        Returns
        -------
        callable
            If use_ffi=False: A function (theta, times) -> pmf_values that computes the PMF.
            If use_ffi=True: A builder function (theta) -> Graph that creates Graph objects.
            If jax_compatible=True and use_ffi=False: supports JIT, grad, vmap, etc.

        Examples
        --------
        # JAX-compatible approach (default)
        >>> model = Graph.load_cpp_model("my_model.cpp")
        >>> theta = jnp.array([1.0, 2.0])
        >>> times = jnp.linspace(0, 10, 100)
        >>> pmf = model(theta, times)
        >>> gradient = jax.grad(lambda p: jnp.sum(model(p, times)))(theta)

        # FFI approach (efficient for repeated evaluations)
        >>> builder = Graph.load_cpp_model("my_model.cpp", use_ffi=True)
        >>> graph = builder(np.array([1.0, 2.0]))  # Build graph once
        >>> pdf1 = graph.pdf(1.0)  # Use many times
        >>> pdf2 = graph.pdf(2.0)  # No rebuild needed
        """
        cpp_path = pathlib.Path(cpp_file)
        if not cpp_path.exists():
            raise FileNotFoundError(f"C++ file not found: {cpp_file}")

        # If using FFI, delegate to the pybind module's load_cpp_builder
        if use_ffi:
            from . import ptdalgorithmscpp_pybind
            return ptdalgorithmscpp_pybind.load_cpp_builder(str(cpp_path))

        # Read user's C++ code
        with open(cpp_path, 'r') as f:
            user_code = f.read()

        # Check that it implements build_model
        if "build_model" not in user_code:
            raise ValueError(
                "C++ file must implement: ptdalgorithms::Graph build_model(const double* theta, int n_params)"
            )

        # Get package root directory
        pkg_dir = pathlib.Path(__file__).parent.parent.parent

        # Use a different approach: build the graph using the Python API
        # by loading the user's C++ code and calling it through ctypes
        # Create a minimal wrapper that links against the existing pybind module
        # This ensures we use the same library instance that's already initialized
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
        g.normalize();  // Normalize the graph before computing PDF
        for (int i = 0; i < n_times; i++) {{
            output[i] = g.pdf(times[i], granularity);
        }}
    }}

    void compute_dph_pmf(const double* theta, int n_params,
                         const int* jumps, int n_jumps,
                         double* output) {{
        ptdalgorithms::Graph g = build_model(theta, n_params);
        g.dph_normalize();  // Normalize for discrete phase-type
        for (int i = 0; i < n_jumps; i++) {{
            output[i] = g.dph_pmf(jumps[i]);
        }}
    }}
}}
'''

        # Compile the complete standalone library
        source_hash = hashlib.md5(user_code.encode()).hexdigest()[:8]
        lib_name = f"user_model_{cpp_path.stem}_{source_hash}"
        lib_path = f"/tmp/{lib_name}.so"

        # Always recompile for now to ensure fresh library
        # (In production, would cache more intelligently)
        if os.path.exists(lib_path):
            os.unlink(lib_path)

        if not os.path.exists(lib_path):
            with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False, mode='w') as f:
                f.write(wrapper_code)
                wrapper_file = f.name

            try:
                # Compile with source files
                # The key insight: the crash happens because ptd_vertex_create allocates
                # memory for state based on graph->state_length, but if graph is NULL
                # or uninitialized, this will segfault
                cpp_src = f'{pkg_dir}/src/cpp/ptdalgorithmscpp.cpp'
                c_src = f'{pkg_dir}/src/c/ptdalgorithms.c'

                cmd = [
                    'g++', '-O3', '-fPIC', '-shared', '-std=c++14',
                    f'-I{pkg_dir}',
                    f'-I{pkg_dir}/api/cpp',
                    f'-I{pkg_dir}/api/c',
                    f'-I{pkg_dir}/include',
                    wrapper_file,
                    cpp_src,
                    c_src,
                    '-o', lib_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Compilation failed:\n{result.stderr}")
            finally:
                os.unlink(wrapper_file)

        # Load the compiled library
        lib = ctypes.CDLL(lib_path)

        # Set up function signatures
        lib.compute_pmf.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,                      # n_params
            ctypes.POINTER(ctypes.c_double),  # times
            ctypes.c_int,                      # n_times
            ctypes.POINTER(ctypes.c_double),  # output
            ctypes.c_int                       # granularity
        ]

        lib.compute_dph_pmf.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # theta
            ctypes.c_int,                      # n_params
            ctypes.POINTER(ctypes.c_int),     # jumps
            ctypes.c_int,                      # n_jumps
            ctypes.POINTER(ctypes.c_double),  # output
        ]

        # Create the Python wrapper function
        def pmf_function(theta, times, discrete=False, granularity=0):
            """Compute PMF using the loaded C++ model"""
            theta_np = np.asarray(theta, dtype=np.float64)
            times_np = np.asarray(times, dtype=np.float64 if not discrete else np.int32)
            output_np = np.zeros_like(times_np, dtype=np.float64)

            if discrete:
                lib.compute_dph_pmf(
                    theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(theta_np),
                    times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    len(times_np),
                    output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
            else:
                lib.compute_pmf(
                    theta_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(theta_np),
                    times_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(times_np),
                    output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    granularity
                )

            return output_np

        if jax_compatible:
            # Wrap for JAX compatibility
            @jax.custom_vjp
            def jax_model(theta, times):
                result_shape = jax.ShapeDtypeStruct(times.shape, jnp.float32)
                return jax.pure_callback(
                    lambda t, tm: pmf_function(t, tm, discrete=False, granularity=0).astype(np.float32),
                    result_shape,
                    theta,
                    times,
                    vmap_method='sequential'
                )

            def jax_model_fwd(theta, times):
                pmf = jax_model(theta, times)
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

                    pmf_plus = jax_model(theta_plus, times)
                    pmf_minus = jax_model(theta_minus, times)

                    grad_i = jnp.sum(g * (pmf_plus - pmf_minus) / (2 * eps))
                    theta_bar.append(grad_i)

                return jnp.array(theta_bar), None

            jax_model.defvjp(jax_model_fwd, jax_model_bwd)

            return jax_model
        else:
            return pmf_function

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
