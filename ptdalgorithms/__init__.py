
from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot

class Graph(_Graph):
    def __init__(self, state_length=None, callback=None, initial=None, trans_as_dict=False):

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
