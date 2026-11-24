import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests", "pytest"))
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
for n_s, rate in ((22, 0.2), (22, 0.001)):
    g0 = coalescent_graph(n_s); g0.update_weights([0.37]); d = g0.discretize(rate); d = d[0] if isinstance(d, tuple) else d
    print('discretize', n_s, rate, d.vertices_length())
for n, rl in ((6, 3), (7, 3), (6, 4), (7, 2), (8, 2)):
    g, idx = coalescent_indexed(n); J = g.joint_prob_graph(idx, reward_limit=rl)
    print('joint', n, rl, J.vertices_length())
