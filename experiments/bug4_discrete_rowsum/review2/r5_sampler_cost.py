"""Review 2 of plan v2: cost of a draw today (check only on visited vertices)
vs the size of a whole-graph row-sum pass; and was_dph of a discrete joint graph."""
import sys, warnings, time, random
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
for n_s, th, rate in ((22, [0.37], 0.001), (22, [0.37], 0.2)):
    g0 = coalescent_graph(n_s); g0.update_weights(th); d = g0.discretize(rate); d = d[0] if isinstance(d, tuple) else d
    E = sum(len(d.vertex_at(i).edges()) for i in range(d.vertices_length()))
    random.seed(1); np.random.seed(1)
    t0 = time.perf_counter(); smp = d.sample(3000); t1 = time.perf_counter()
    print(f'coalescent n={n_s} discretize({rate}): {d.vertices_length()} vertices, {E} edges; {1e6*(t1-t0)/3000:.1f} us per draw; mean jumps {float(np.mean(smp)):.1f}')
g, idx = coalescent_indexed(5); J = g.joint_prob_graph(idx, reward_limit=3)
print('fresh discrete joint graph: is_discrete', J.is_discrete, 'was_dph', J.get_was_dph())
rs = [sum(e.weight() for e in J.vertex_at(i).edges()) for i in range(J.vertices_length()) if J.vertex_at(i).edges()]
print('  fresh row sums: max', max(rs))
for f in ('expectation', 'variance'):
    try: print(f'  fresh {f}():', getattr(J, f)())
    except Exception as e: print(f'  fresh {f}():', type(e).__name__, str(e)[:100])
