"""
Tests XXXX
"""


from phasic import Graph, with_ipv
import numpy as np
from pytest import approx, raises

nr_samples = 4

np.random.seed(42)
@with_ipv([nr_samples]+[0]*(nr_samples-1))
def coalescent(state):
    transitions = []
    for i in range(state.size):
        for j in range(i, state.size):            
            same = int(i == j)
            if same and state[i] < 2:
                continue
            if not same and (state[i] < 1 or state[j] < 1):
                continue 
            new = state.copy()
            new[i] -= 1
            new[j] -= 1
            new[i+j+1] += 1
            transitions.append((new, [state[i]*(state[j]-same)/(1+same)]))
    return transitions


class TestDiscretize:
    """Test discretization"""

    def test_discretize_states(self):

        graph = Graph(coalescent)

        def mutation_rate(state):
            nr_lineages = sum(state)
            return [0, nr_lineages]
        
        disc_graph = graph.discretize(mutation_rate)
        assert disc_graph.vertices_length() == graph.vertices_length() + graph.vertices_length() - 2

    def test_discretize_skip_existing(self):

        graph = Graph(coalescent)

        def mutation_rate(state):
            nr_lineages = sum(state)
            return [0, nr_lineages]

        disc_graph = graph.discretize(mutation_rate)
        disc_graph_again = graph.discretize(mutation_rate, skip_existing=True)

        assert disc_graph.vertices_length() == graph.vertices_length() + graph.vertices_length() - 2
        assert disc_graph_again.vertices_length() == disc_graph.vertices_length()         