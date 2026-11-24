"""Crash probes, one per process: argv[1] in {ctx, mv}."""
import sys, numpy as np, phasic
from phasic import Graph
g = Graph(1); s = g.starting_vertex(); a = g.find_or_create_vertex([1]); b = g.find_or_create_vertex([2])
s.add_edge(a, 1.0); a.add_edge(b, 1.5); g.is_discrete = True
print('probe', sys.argv[1], flush=True)
if sys.argv[1] == 'ctx':
    c = g.distribution_context(); print('created', flush=True); c.step(); print('pmf', c.pmf())
elif sys.argv[1] == 'mv':
    print(g.sample_multivariate_discrete(2, np.array([[0., 0.], [1., 2.], [0., 0.]])))
elif sys.argv[1] == 'mv_ok':
    g2 = Graph(1); s = g2.starting_vertex(); a = g2.find_or_create_vertex([1]); b = g2.find_or_create_vertex([2])
    s.add_edge(a, 1.0); a.add_edge(b, 0.5); g2.is_discrete = True
    print(g2.sample_multivariate_discrete(2, np.array([[0., 0.], [1., 2.], [0., 0.]])))
