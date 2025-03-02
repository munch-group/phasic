
from .ptdalgorithmscpp_pybind import *
from .ptdalgorithmscpp_pybind import Graph as _Graph
from .ptdalgorithmscpp_pybind import Vertex, Edge

from . import plot

class Graph(_Graph):
    def __init__(self, n):
        super().__init__(n)

    def plot(self, *args, **kwargs):
        return plot.plot_graph(self, *args, **kwargs)
