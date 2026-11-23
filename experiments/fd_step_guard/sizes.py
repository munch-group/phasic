import sys, time; sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed, two_island_graph
for n in (4, 6, 8):
    t=time.time(); g, idx = coalescent_indexed(n)
    J = g.joint_prob_graph(idx, reward_limit=3, discrete=False)
    ne = sum(len(J.vertex_at(i).edges()) for i in range(J.vertices_length()))
    print(f'joint n={n}: base {g.vertices_length()} v, joint {J.vertices_length()} v, {ne} edges, build {time.time()-t:.1f}s', flush=True)
for n in (6, 10, 14):
    t=time.time(); g = two_island_graph(n)
    ne = sum(len(g.vertex_at(i).edges()) for i in range(g.vertices_length()))
    print(f'island n={n}: {g.vertices_length()} v, {ne} edges, build {time.time()-t:.1f}s', flush=True)
