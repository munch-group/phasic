"""RA2b: why is the JSP step 3x slower per edge? IPV zero vs set; step count dependence (denormals)."""
import sys, time
import numpy as np
sys.path.insert(0, '/Users/kmt/phasic/tests/pytest')
from test_notebook_model_recovery import coalescent_indexed
g, idx = coalescent_indexed(6)
jc = g.joint_prob_graph(idx, discrete=False, reward_limit=3)
jsp = jc.joint_stop_prob_graph()
V = jsp.vertices_length(); E = sum(len(jsp.vertex_at(i).edges()) for i in range(V))
E0 = len(jsp.vertex_at(0).edges())
print('V', V, 'E', E, 'starting-vertex (IPV) edges', E0)
def run(label, G, chunks=(500, 500, 2000, 4000)):
    ctx = jsp.distribution_context(G)
    out = []
    for n in chunks:
        t0 = time.perf_counter()
        for _ in range(n): ctx.step()
        out.append((time.perf_counter() - t0) / n * 1e9 / E)
    sp = np.asarray(ctx.stop_probability(), float)
    pos = sp[sp > 0]
    print(f'{label}: ns/edge per chunk {["%.1f" % o for o in out]}; min positive prob {pos.min() if len(pos) else 0:.3g}, n denormal {(pos < 2.3e-308).sum()}', flush=True)
jsp.update_weights([1.0, 1.0]); run('IPV all zero, G=1e5', 100000)
n_ipv = E0
ipv = np.zeros(n_ipv); ipv[0] = 1.0
jsp.update_ipv(ipv); jsp.update_weights([1.0, 1.0])
run('IPV set, G=1e5 (t<=0.07)', 100000)
run('IPV set, G=30 (t up to 233)', 30)
run('IPV set, G=30 again', 30, chunks=(7000, 20000))
