# reviewAK p7: are fresh uncached builds bit-identical in expectation() (baseline for p6)?
import sys, numpy as np
from phasic import Graph, with_ipv
sys.path.insert(0, "experiments/bug5_graph_cache/reviewAK")
from p6_hit_vs_fresh import two_island_const, cb_const
for name, cb, ipv in [("const25", cb_const, [[[6, 0], 0.7], [[5, 1], 0.3]]), ("island40", two_island_const, [[[40, 0], 1.0]])]:
    Es = []
    for i in range(4):
        g = Graph(with_ipv(ipv)(cb), graph_cache=False)
        Es.append((g.expectation(), g.expectation()))
    print(name, [repr(e) for e in Es])
