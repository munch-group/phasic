from functools import partial
from collections import defaultdict
import numpy as np
from typing import Any, TypeVar, List, Tuple, Dict, Union
from collections.abc import Sequence, MutableSequence, Callable
import os
import hashlib
import subprocess
import tempfile
import ctypes
import pathlib
import jax
import jax.numpy as jnp

from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot
from .plot import set_theme

__version__ = '0.19.106'

GraphType = TypeVar('Graph') 

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

    @classmethod
    def load_cpp_model(cls, cpp_file: Union[str, pathlib.Path], jax_compatible: bool = True) -> Callable:
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

        Returns
        -------
        callable
            A function (theta, times) -> pmf_values that computes the PMF.
            If jax_compatible=True, supports JIT, grad, vmap, etc.

        Examples
        --------
        >>> model = Graph.load_cpp_model("my_model.cpp")
        >>> theta = jnp.array([1.0, 2.0])
        >>> times = jnp.linspace(0, 10, 100)
        >>> pmf = model(theta, times)
        >>> gradient = jax.grad(lambda p: jnp.sum(model(p, times)))(theta)
        """
        cpp_path = pathlib.Path(cpp_file)
        if not cpp_path.exists():
            raise FileNotFoundError(f"C++ file not found: {cpp_file}")

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
        # Create a minimal wrapper that compiles user code with full Graph implementation
        wrapper_code = f'''
// Include the C++ API header
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
                # Find source files to compile
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
