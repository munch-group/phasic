"""Bug 4 gate step 2: bit-identity of the paths the fix must not change.

Dumps float hex of pdf, cdf, stop_probability and accumulated_visiting_time
for continuous models (coalescent, two-island, two-locus, reward-transformed,
parameterized through pmf_from_graph), and pdf(1..200) / cdf / stop_probability
for a native discrete graph whose rows all sum to at most 0.6 (no cap can
fire there). Run under the master build and under the fix build; the two dumps
must be identical.

usage: python gate_bitid.py <out_file>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "pytest"))

from phasic import Graph  # noqa: E402
from test_notebook_model_recovery import (coalescent_graph,  # noqa: E402
                                          two_island_graph, two_locus_arg)

TIMES = [0.05, 0.3, 1.0, 2.5, 6.0]
lines = []


def hx(x):
    return [float(v).hex() for v in np.ravel(np.asarray(x, dtype=float))]


def dump_continuous(tag, g):
    lines.append(f"{tag} pdf {hx([g.pdf(t) for t in TIMES])}")
    lines.append(f"{tag} cdf {hx([g.cdf(t) for t in TIMES])}")
    for t in (0.3, 2.5):
        lines.append(f"{tag} stop_probability({t}) {hx(g.stop_probability(t))}")
        lines.append(f"{tag} accumulated_visiting_time({t}) "
                     f"{hx(g.accumulated_visiting_time(t))}")


g = coalescent_graph(6)
g.update_weights([1.7])
dump_continuous("coalescent6", g)

g = two_island_graph(4)
g.update_weights([1.0, 0.3])
dump_continuous("two_island4", g)

g = two_locus_arg(3)
g.update_weights([1.0, 0.5])
dump_continuous("two_locus3", g)

g = coalescent_graph(6)
g.update_weights([1.7])
rew = np.zeros(g.vertices_length())
rew[1:-1] = np.arange(1, g.vertices_length() - 1) % 3 + 1
dump_continuous("coalescent6_rewarded", g.reward_transform(rew))

model = Graph.pmf_from_graph(coalescent_graph(6))
lines.append(f"pmf_from_graph coalescent6 "
             f"{hx(model(np.array([1.7]), np.array(TIMES)))}")


def safe_discrete():
    """Native discrete graph, every row sums to <= 0.6."""
    gd = Graph(1)
    s = gd.starting_vertex()
    a = gd.find_or_create_vertex([1])
    b = gd.find_or_create_vertex([2])
    c = gd.find_or_create_vertex([3])
    z = gd.find_or_create_vertex([0])
    s.add_edge(a, 0.6)
    s.add_edge(b, 0.4 * 0.0)  # weight 0 edge kept out of the row sum
    a.add_edge(b, 0.35)
    a.add_edge(c, 0.2)
    b.add_edge(a, 0.1)
    b.add_edge(z, 0.45)
    c.add_edge(z, 0.3)
    c.add_edge(a, 0.25)
    gd.is_discrete = True
    return gd


gd = safe_discrete()
lines.append(f"safe_discrete pdf(1..200) {hx([gd.pdf(k) for k in range(1, 201)])}")
lines.append(f"safe_discrete cdf(1..200) {hx([gd.cdf(k) for k in range(1, 201)])}")
for k in (1, 7, 40, 200):
    lines.append(f"safe_discrete stop_probability({k}) {hx(gd.stop_probability(k))}")
    lines.append(f"safe_discrete accumulated_visits({k}) {hx(gd.accumulated_visits(k))}")

with open(sys.argv[1], "w") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "lines written to", sys.argv[1])
