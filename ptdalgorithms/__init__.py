
from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot

class Graph(_Graph):
    def __init__(self, state_length=None, callback=None, initial=None, trans_as_dict=False):
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
        initial : 
            A list of integers representing the initial model state, by default None
        trans_as_dict : 
            Whether the callback should return dictionaries with 'state' and 'weight' keys instead of tuples, by default False

        Returns
        -------
        :
            A graph object representing a phase-type distribution.
        """

        assert bool(callback) == bool(initial), "callback and initial_state must be both None or both not None"

        if callback and initial:
            if trans_as_dict:        
                super().__init__(callback_dicts=callback, initial_state=initial)
            else:
                super().__init__(callback_tuples=callback, initial_state=initial)
        else:
            super().__init__(state_length)

    def plot(self, *args, **kwargs):
        return plot.plot_graph(self, *args, **kwargs)
