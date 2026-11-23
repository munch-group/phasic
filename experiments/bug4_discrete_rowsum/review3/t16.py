"""Cost context for plan 2.1 ("the six existing checks are replaced by calls to the helper", helper loops over ALL vertices):
the sampler checks sit inside the per-vertex walk loop. Today's sampling time vs. edges per full validation."""
import sys, time, random, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/kmt/phasic-bug4/tests/pytest')
from test_notebook_model_recovery import coalescent_graph, coalescent_indexed
for label, d in (('coalescent n=22 discretize(0.01)', (lambda g: (g.update_weights([0.37]), g.discretize(0.01))[1])(coalescent_graph(22))),
                 ('joint n=6 rl=3 [5,0.01]', (lambda gi: (lambda J: (J.update_weights([5.0, 0.01]), J)[1])(gi[0].joint_prob_graph(gi[1], reward_limit=3)))(coalescent_indexed(6)))):
    V = d.vertices_length(); E = sum(len(d.vertex_at(i).edges()) for i in range(V))
    random.seed(1); t0 = time.perf_counter(); s = d.sample(20000); dt = time.perf_counter() - t0
    mean_jumps = float(np.mean(s))
    print(f'{label}: V={V} E={E}; sample(20000) today {dt*1e3:.1f} ms = {dt/20000*1e9:.0f} ns/draw, mean jumps/draw {mean_jumps:.1f}')
    print(f'   one full-graph validation ~ E adds = {E} (>= ~{E*1.0:.0f} ns at 1 ns/edge); per DRAW that is {E*1.0/(dt/20000*1e9):.1f}x today\'s draw cost; per VISITED VERTEX {E*mean_jumps/(dt/20000*1e9):.0f}x')
