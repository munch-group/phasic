import math
from phasic import Graph
g = Graph(1)
v = g.find_or_create_vertex([1])
g.starting_vertex().add_edge(v, 1.0)
a = g.find_or_create_vertex([2])
v.add_edge(a, 1.0)   # Exp(1): E[T^k] = k!
m = g.moments(14)
for k in (12, 13, 14):
    print(k, m[k-1], math.factorial(k))
