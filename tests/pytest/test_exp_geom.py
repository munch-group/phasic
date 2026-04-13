from phasic import Graph
import numpy as np
import jax.numpy as jnp

from pytest import approx

rate = 0.1

cont_graph = Graph(1)
vertex = cont_graph.find_or_create_vertex([2])
cont_graph.starting_vertex().add_edge(vertex, 1)
abs_vertex = cont_graph.find_or_create_vertex([1])
vertex.add_edge(abs_vertex, rate)
assert cont_graph.expectation() == approx(1 / rate)
assert cont_graph.variance() == approx(1 / rate**2)


disc_graph = Graph(1)
vertex = disc_graph.find_or_create_vertex([2])
disc_graph.starting_vertex().add_edge(vertex, 1)
abs_vertex = disc_graph.find_or_create_vertex([1])
vertex.add_edge(abs_vertex, rate)
# COMPOSABLE_MIGRATION: old call
# rewards = disc_graph.discretize(1-rate)
# assert disc_graph.expectation_discrete(rewards) == approx((1 - rate)  / rate)
# assert disc_graph.variance_discrete(rewards) == approx((1 - rate) / rate**2)
disc_graph = disc_graph.discretize(1-rate)
assert disc_graph.expectation_discrete(disc_graph.rewards) == approx((1 - rate)  / rate)
assert disc_graph.variance_discrete(disc_graph.rewards) == approx((1 - rate) / rate**2)
