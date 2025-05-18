from functools import partial
from collections import defaultdict
import numpy as np
from typing import Any, TypeVar, List, Tuple, Dict, Union
from collections.abc import Sequence, MutableSequence, Callable

from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot

__version__ = '0.19.55'

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
