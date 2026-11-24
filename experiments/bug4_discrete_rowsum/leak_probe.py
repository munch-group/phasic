"""Review finding 4: the reward-transform failure path now frees the
transformed graph, new_graph_indices and vertices. Exercise it many times;
RSS must stay flat and the error must still be raised with its message."""
import resource, numpy as np
from phasic import Graph
def g15():
    g = Graph(1); s = g.starting_vertex()
    v = g.find_or_create_vertex([1]); a = g.find_or_create_vertex([0])
    s.add_edge(v, 1.0); v.add_edge(a, 1.5); g.is_discrete = True
    return g
g = g15()
for rnd in range(3):
    for _ in range(20000):
        try:
            g.reward_transform_discrete(np.array([0, 2, 0]))
            raise SystemExit("no raise")
        except RuntimeError as e:
            assert "outgoing rate <= 1" in str(e)
    print("round", rnd, "maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
